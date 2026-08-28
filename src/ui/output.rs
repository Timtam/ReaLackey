//! The conversation output pane.
//!
//! When possible we host an embedded WebView2 (via `wry`) as a child of the
//! native dialog and render the conversation as HTML — Markdown formatting plus
//! collapsible `<details>` tool cards. If the WebView2 runtime is missing or
//! creation fails, everything degrades to the plain read-only edit control (the
//! previous behaviour), so there is never a broken window.
//!
//! Everything here runs on REAPER's main thread: the dialog, the pump, and the
//! FFI callbacks all fire there, so the (non-`Send`) WebView lives in a
//! main-thread `thread_local`.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::text::{html_escape, markdown_to_html};
use crate::ui::ffi;

thread_local! {
    static STATE: RefCell<Output> = RefCell::new(Output::new());
}

/// Thread-safe mirror of "is the webview live?", so the worker thread (which can't
/// touch the main-thread `STATE` thread-local) can decide whether the cut-by-text
/// editor can open before it commits to awaiting a reply.
static WEBVIEW_ACTIVE: AtomicBool = AtomicBool::new(false);

/// True once the webview's HTML/JS has finished loading (the page pings `ui:ready`).
/// `WEBVIEW_ACTIVE` flips the instant the webview is created, but its page loads
/// asynchronously — the worker waits on THIS before injecting the editor modal.
static WEBVIEW_READY: AtomicBool = AtomicBool::new(false);

/// Whether the embedded webview is live (readable from any thread).
pub fn webview_active() -> bool {
    WEBVIEW_ACTIVE.load(Ordering::Acquire)
}

/// Whether the webview's page has finished loading (readable from any thread).
pub fn webview_ready() -> bool {
    WEBVIEW_READY.load(Ordering::Acquire)
}

/// Mark the webview page ready — called when it pings `ui:ready` on load.
pub fn set_webview_ready() {
    WEBVIEW_READY.store(true, Ordering::Release);
}

struct Output {
    #[cfg(webview)]
    webview: Option<wry::WebView>,
    /// Accumulated Markdown of the assistant message currently streaming.
    assistant_md: String,
    /// Accumulated Markdown of the reasoning/"thinking" block for the current turn
    /// (streamed into a collapsible section, separate from the answer).
    reasoning_md: String,
}

impl Output {
    fn new() -> Self {
        Self {
            #[cfg(webview)]
            webview: None,
            assistant_md: String::new(),
            reasoning_md: String::new(),
        }
    }

    fn active(&self) -> bool {
        #[cfg(webview)]
        {
            self.webview.is_some()
        }
        #[cfg(not(webview))]
        {
            false
        }
    }

    fn eval(&self, js: &str) {
        #[cfg(webview)]
        if let Some(wv) = &self.webview {
            let _ = wv.evaluate_script(js);
        }
        #[cfg(not(webview))]
        {
            let _ = js;
        }
    }

    /// Call a JS helper defined in the base document with one HTML-string arg.
    fn call_js(&self, func: &str, html_arg: &str) {
        let json = serde_json::to_string(html_arg).unwrap_or_else(|_| "\"\"".into());
        self.eval(&format!("{func}({json});"));
    }

    fn user_message(&mut self, text: &str) {
        self.assistant_md.clear();
        self.reasoning_md.clear();
        if self.active() {
            // A heading so a screen reader can jump straight to each question (h).
            let html = format!("<h2 class=\"turn user\">You: {}</h2>", html_escape(text));
            self.call_js("addBlock", &html);
        } else {
            ffi::append_output(&format!("\r\nYou: {text}\r\n"));
        }
    }

    fn assistant_start(&mut self) {
        self.assistant_md.clear();
        // The answer follows the reasoning; reset the reasoning buffer so the next
        // turn's reasoning (if any) starts a fresh block.
        self.reasoning_md.clear();
        if self.active() {
            self.eval("startAssistant();");
        } else {
            ffi::append_output("Assistant: ");
        }
    }

    /// Append a streamed reasoning/"thinking" token into a collapsible block,
    /// separate from the answer. Not spoken via OSARA (only the answer is).
    fn reasoning_delta(&mut self, token: &str) {
        if !self.active() {
            return; // reasoning is a webview-only enhancement; fallback shows the answer
        }
        let first = self.reasoning_md.is_empty();
        self.reasoning_md.push_str(token);
        if first {
            self.eval("startReasoning();");
        }
        let html = markdown_to_html(&self.reasoning_md);
        self.call_js("updateReasoning", &html);
    }

    fn assistant_delta(&mut self, token: &str) {
        self.assistant_md.push_str(token);
        if self.active() {
            let html = markdown_to_html(&self.assistant_md);
            self.call_js("updateAssistant", &html);
        } else {
            ffi::append_output(token);
        }
    }

    fn tool_started(&mut self, name: &str, input: &str) {
        // The reasoning phase (if any) ended when the model started calling tools.
        self.reasoning_md.clear();
        if self.active() {
            let html = format!(
                "<details class=\"tool\"><summary>{}</summary>\
                 <pre class=\"tin\">{}</pre><div class=\"tres\"></div></details>",
                html_escape(name),
                html_escape(input)
            );
            // addTool groups consecutive tool cards into one "Used N tools"
            // collapsible (assistant text / notices between tools break the run).
            self.call_js("addTool", &html);
        } else {
            ffi::append_output(&format!("\r\n[tool: {name}]\r\n"));
        }
    }

    fn tool_finished(&mut self, is_error: bool, summary: &str) {
        if self.active() {
            let class = if is_error { "tres err" } else { "tres" };
            let html = format!("<pre class=\"{}\">{}</pre>", class, html_escape(summary));
            self.call_js("setToolResult", &html);
        }
        // The edit fallback stays terse: the tool line is enough there.
    }

    fn notice(&mut self, text: &str) {
        if self.active() {
            self.call_js(
                "addBlock",
                &format!("<div class=\"msg notice\">{}</div>", html_escape(text)),
            );
        } else {
            ffi::append_output(&format!("{text}\r\n"));
        }
    }

    fn error(&mut self, text: &str) {
        if self.active() {
            self.call_js(
                "addBlock",
                &format!("<div class=\"msg error\">{}</div>", html_escape(text)),
            );
        } else {
            ffi::append_output(&format!("\r\n[Error] {text}\r\n"));
        }
    }

    /// Put text in the webview's aria-live region so a screen reader observing
    /// the pane announces it. Plain text (set via textContent), not HTML.
    fn announce(&self, text: &str) {
        if self.active() {
            self.call_js("liveAnnounce", text);
        }
    }

    /// Update the webview's status line (visual; spoken feedback is via announce).
    fn status(&self, text: &str) {
        if self.active() {
            self.call_js("setStatus", text);
        }
    }

    /// Insert a prompt preset's body into the composer at the caret (leaving it
    /// editable before send). No-op without the webview.
    fn insert_preset(&self, body: &str) {
        if self.active() {
            self.call_js("insertPreset", body);
        }
    }

    /// Tell the composer whether a turn is in flight (gates its Escape = stop).
    fn set_generating(&self, on: bool) {
        if self.active() {
            self.eval(if on { "setGenerating(true);" } else { "setGenerating(false);" });
        }
    }

    /// Open the cut-by-text editor modal with its JSON payload (words + audio).
    fn open_cut_editor(&self, payload_json: &str) {
        if self.active() {
            self.call_js("openCutEditor", payload_json);
        }
    }

    /// Close the cut-by-text editor modal.
    fn close_cut_editor(&self) {
        if self.active() {
            self.eval("closeCutEditor();");
        }
    }
}

// ---- public API (all main-thread) -------------------------------------------

/// Create the embedded webview once the dialog exists. Idempotent; on failure
/// (or non-Windows) leaves the plain edit control in place.
pub fn ensure_created() {
    ffi::install_resize_cb();
    ffi::install_destroy_cb();
    let already = STATE.with(|c| c.borrow().active());
    if already {
        return;
    }
    #[cfg(webview)]
    {
        // Build the webview WITHOUT holding the STATE borrow: creating the child
        // window can synchronously fire WM_SIZE -> on_resize(), which borrows STATE.
        match webview_impl::create() {
            Ok(webview) => {
                STATE.with(|c| c.borrow_mut().webview = Some(webview));
                // Publish "webview is live" for the worker thread (cut-by-text editor).
                WEBVIEW_ACTIVE.store(true, Ordering::Release);
                // Hand the whole window to the webview (it hosts the conversation
                // AND the input composer now), hiding every native control, then
                // re-bound the webview to the freed-up full-window output rect.
                ffi::set_webview_active(true);
                on_resize();
                // Tab-focus forwarding into the web content is WebView2-specific;
                // on macOS WKWebView handles its own focus/keyboard.
                #[cfg(windows)]
                {
                    ffi::enable_webview_tabstop();
                    ffi::install_webview_focus_cb();
                    webview_impl::install_focus_out_handler();
                }
                // macOS: wry does NOT make a *child* WKWebView the first responder
                // (it only does so for a standalone webview), and the native dialog
                // controls that SWELL would otherwise focus are now hidden — so
                // without this the user has no keyboard path into the composer.
                // `WebView::focus()` calls `window.makeFirstResponder(webview)`;
                // then land the caret in the message box, mirroring the Windows
                // on-focus flow (`on_webview_focus`).
                #[cfg(target_os = "macos")]
                STATE.with(|c| {
                    let out = c.borrow();
                    if let Some(wv) = &out.webview {
                        let _ = wv.focus();
                    }
                    out.eval("focusInput();");
                });
                // No console message on success — ShowConsoleMsg pops the console
                // window open, which is unwanted on a normal launch.
            }
            Err(_e) => {
                // On an actual failure the webview stays None, so the plain
                // edit-control fallback takes over automatically. On Windows,
                // surface WHY in the console (the one case worth popping it).
                #[cfg(windows)]
                console(&format!(
                    "ReaLackey: HTML pane unavailable, using plain text output. \
                     Reason: {_e}\n"
                ));
            }
        }
    }
}

/// Print a line to REAPER's console (main-thread REAPER handle). Used ONLY for
/// the webview-failure diagnostic (Windows) — routine status goes via OSARA / the
/// pane, so a normal launch never opens the console.
#[cfg(windows)]
fn console(msg: &str) {
    let _ = crate::reaper::api::with(|r| r.show_console_msg(msg));
}

/// Drop the webview when its parent dialog is destroyed, so it never lingers
/// with a dangling parent (and `ensure_created` will rebuild on re-open).
pub fn on_destroy() {
    // The webview is going away: unblock any worker awaiting a cut-by-text editor
    // reply (it would otherwise hang), and mark the pane inactive + not-ready (a
    // rebuild re-pings ui:ready when its page reloads).
    WEBVIEW_ACTIVE.store(false, Ordering::Release);
    WEBVIEW_READY.store(false, Ordering::Release);
    crate::ui::bridge::cancel_editor();
    #[cfg(webview)]
    {
        // Take the webview out (releasing the borrow) before dropping it. Dropping
        // it closes the WebView2 controller, which must happen while the module is
        // still attached (never at DLL detach) — see `control_surface::close_no_reset`.
        let webview = STATE.with(|c| c.borrow_mut().webview.take());
        drop(webview);
    }
}

/// Re-bound the webview to the output area after a dialog resize.
pub fn on_resize() {
    #[cfg(webview)]
    STATE.with(|c| {
        let out = c.borrow();
        if let Some(wv) = &out.webview {
            if let Some((x, y, w, h)) = ffi::output_bounds() {
                webview_impl::set_bounds(wv, x, y, w, h);
            }
        }
    });
}

pub fn user_message(text: &str) {
    STATE.with(|c| c.borrow_mut().user_message(text));
}
pub fn assistant_start() {
    STATE.with(|c| c.borrow_mut().assistant_start());
}
pub fn assistant_delta(token: &str) {
    STATE.with(|c| c.borrow_mut().assistant_delta(token));
}

/// Stream a reasoning/"thinking" token into a collapsible block (main thread).
pub fn reasoning_delta(token: &str) {
    STATE.with(|c| c.borrow_mut().reasoning_delta(token));
}
pub fn tool_started(name: &str, input: &str) {
    STATE.with(|c| c.borrow_mut().tool_started(name, input));
}
pub fn tool_finished(is_error: bool, summary: &str) {
    STATE.with(|c| c.borrow_mut().tool_finished(is_error, summary));
}
pub fn notice(text: &str) {
    STATE.with(|c| c.borrow_mut().notice(text));
}
pub fn error(text: &str) {
    STATE.with(|c| c.borrow_mut().error(text));
}
pub fn announce(text: &str) {
    STATE.with(|c| c.borrow().announce(text));
}
/// Open the cut-by-text editor modal in the webview (main thread).
pub fn open_cut_editor(payload_json: &str) {
    STATE.with(|c| c.borrow().open_cut_editor(payload_json));
}
/// Tell the editor whether the clip report reached the clipboard (main thread).
pub fn report_copied(ok: bool) {
    STATE.with(|c| {
        c.borrow()
            .eval(if ok { "cutReportDone(true);" } else { "cutReportDone(false);" })
    });
}
/// Clear the visible conversation log (main thread).
pub fn clear_log() {
    STATE.with(|c| c.borrow().eval("clearLog();"));
}
/// Hand the editor the real cut segments for its preview (main thread).
pub fn send_preview(json: &str) {
    STATE.with(|c| c.borrow().call_js("cutPreviewSegments", json));
}
/// Close the cut-by-text editor modal (main thread).
pub fn close_cut_editor() {
    STATE.with(|c| c.borrow().close_cut_editor());
}
/// Speak `text` to the screen reader exactly ONCE. Prefer OSARA (focus-independent
/// and cross-platform — it reaches the reader whether or not the chat pane is
/// focused); fall back to the webview aria-live region only when OSARA isn't
/// present. Announcing through BOTH — as the old code did — made a reader that
/// observes each channel (OSARA installed AND focus in the chat pane, the common
/// case) speak everything twice. Main thread only, like both underlying calls.
pub fn speak(text: &str) {
    crate::reaper::osara::announce(text);
    if !crate::reaper::osara::is_running() {
        announce(text);
    }
}
/// Update the webview status line. No-op in the plain-text fallback (the native
/// status field is driven separately via `ffi::set_status`).
pub fn status(text: &str) {
    STATE.with(|c| c.borrow().status(text));
}

/// Insert a prompt preset's body into the chat composer (main thread). Triggered
/// from the webview preset picker; no-op when the webview isn't up.
pub fn insert_preset(body: &str) {
    STATE.with(|c| c.borrow().insert_preset(body));
}
/// Mirror the "generating" state into the webview composer (gates Esc = stop).
pub fn set_generating(on: bool) {
    STATE.with(|c| c.borrow().set_generating(on));
}

/// The webview host window just gained keyboard focus (the user Tabbed onto it).
/// Push focus straight into the web content so the very first Tab lands there
/// instead of stopping silently on the empty host window. Fired from the C++
/// subclass on `WM_SETFOCUS`.
pub fn on_webview_focus() {
    #[cfg(windows)]
    {
        // Clone the controller out from under the borrow before touching COM.
        let controller = STATE.with(|c| {
            use wry::WebViewExtWindows;
            c.borrow().webview.as_ref().map(|wv| wv.controller())
        });
        if let Some(controller) = controller {
            webview_impl::move_focus_into_content(&controller);
            // Land the caret in the composer so the user can just start typing.
            STATE.with(|c| c.borrow().eval("focusInput();"));
        }
    }
}

// ---- wry hosting (Windows: WebView2, macOS: WKWebView) ----------------------

#[cfg(webview)]
mod webview_impl {
    use std::path::PathBuf;

    use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle, WindowHandle};
    #[cfg(target_os = "macos")]
    use raw_window_handle::AppKitWindowHandle;
    #[cfg(windows)]
    use raw_window_handle::Win32WindowHandle;
    #[cfg(windows)]
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Controller, ICoreWebView2MoveFocusRequestedEventArgs,
        COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC,
    };
    #[cfg(windows)]
    use webview2_com::MoveFocusRequestedEventHandler;
    #[cfg(not(target_os = "macos"))]
    use wry::dpi::{PhysicalPosition, PhysicalSize};
    #[cfg(target_os = "macos")]
    use wry::dpi::{LogicalPosition, LogicalSize};
    use wry::http::Request;
    use wry::{Rect, WebContext, WebView, WebViewBuilder};

    use crate::ui::ffi;

    /// Borrow-only wrapper so wry can host the webview as a child of the dialog.
    /// The value is the dialog's native handle: a Win32 `HWND` on Windows, and on
    /// macOS the SWELL `HWND`, which *is* an `NSView` (SWELL_hwndChild : NSView).
    struct Host(isize);

    impl HasWindowHandle for Host {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            // SAFETY (both arms): the dialog owns the native window/view and
            // outlives the webview, which is created, used, and dropped on this
            // (main) thread while it exists.
            #[cfg(windows)]
            {
                let hwnd = std::num::NonZeroIsize::new(self.0).ok_or(HandleError::Unavailable)?;
                let mut handle = Win32WindowHandle::new(hwnd);
                handle.hinstance = None;
                Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
            }
            #[cfg(target_os = "macos")]
            {
                let ns_view = std::ptr::NonNull::new(self.0 as *mut std::ffi::c_void)
                    .ok_or(HandleError::Unavailable)?;
                let handle = AppKitWindowHandle::new(ns_view);
                Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::AppKit(handle)) })
            }
        }
    }

    const BASE_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; font-src data:;">
<style>
:root{color-scheme:dark;}
html,body{margin:0;padding:0;height:100%;font:13px/1.55 "Segoe UI",system-ui,sans-serif;background:#1e1e1e;color:#e6e6e6;}
body{display:flex;flex-direction:column;height:100vh;}
#log{flex:1 1 auto;overflow-y:auto;padding:10px;}
.msg{margin:0 0 12px;word-wrap:break-word;overflow-wrap:anywhere;}
/* Each turn starts with a heading so a screen reader can jump between them (h). */
h2.turn{font-size:13px;font-weight:700;margin:16px 0 4px;line-height:1.4;scroll-margin:8px;}
h2.turn:focus{outline:2px solid #0e639c;outline-offset:1px;}
h2.turn.user{color:#9cdcfe;}
h2.turn.assistant{color:#569cd6;}
.body{margin:0 0 12px;}
.msg.notice{color:#c9a227;font-style:italic;}
.msg.error{color:#f48771;white-space:pre-wrap;}
.body p{margin:.4em 0;}
.body pre{background:#111;padding:8px;border-radius:5px;overflow:auto;}
.body code{background:#111;padding:0 3px;border-radius:3px;}
.body a{color:#4ea1ff;}
details.tool{border:1px solid #3a3a3a;border-radius:6px;margin:8px 0;background:#232323;}
details.tool>summary{cursor:pointer;padding:5px 9px;list-style:none;color:#b5cea8;}
details.tool>summary::-webkit-details-marker{display:none;}
details.tool>summary::before{content:"\25b8  ";}
details.tool[open]>summary::before{content:"\25be  ";}
details.tool pre{margin:0;padding:8px;background:#151515;overflow:auto;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;}
.tres.err{color:#f48771;}
/* Grouping of consecutive tool cards ("Used N tools"). */
details.toolgroup{border:1px solid #3a3a3a;border-radius:6px;margin:8px 0;background:#232323;}
details.toolgroup>summary.tgsum{cursor:pointer;padding:5px 9px;list-style:none;color:#b5cea8;font-weight:600;}
details.toolgroup>summary.tgsum::-webkit-details-marker{display:none;}
details.toolgroup>summary.tgsum::before{content:"\25b8  ";}
details.toolgroup[open]>summary.tgsum::before{content:"\25be  ";}
.tgbody{padding:0 8px 4px;}
.tgbody details.tool{margin:6px 0;background:#1b1b1b;}
/* Reasoning / "thinking": a muted collapsible block, collapsed by default. */
details.reasoning{border:1px solid #333;border-radius:6px;margin:8px 0;background:#1b1b1b;}
details.reasoning>summary{cursor:pointer;padding:5px 9px;list-style:none;color:#8a8a8a;font-style:italic;}
details.reasoning>summary::-webkit-details-marker{display:none;}
details.reasoning>summary::before{content:"\25b8  ";}
details.reasoning[open]>summary::before{content:"\25be  ";}
details.reasoning .rbody{padding:0 9px 6px;color:#9a9a9a;font-size:12px;}
details.reasoning .rbody pre{background:#111;padding:8px;border-radius:5px;overflow:auto;}
.sr{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;}
#status{flex:0 0 auto;padding:2px 10px;color:#9a9a9a;font-size:12px;min-height:15px;}
#composer{flex:0 0 auto;display:flex;gap:6px;padding:8px 10px 10px;border-top:1px solid #3a3a3a;background:#252526;}
#msg{flex:1 1 auto;min-width:0;resize:none;overflow-y:auto;font:inherit;color:#e6e6e6;background:#1e1e1e;border:1px solid #3a3a3a;border-radius:4px;padding:6px 8px;}
#send{flex:0 0 auto;padding:6px 16px;cursor:pointer;color:#fff;background:#0e639c;border:1px solid #1177bb;border-radius:4px;font:inherit;}
#send:hover{background:#1177bb;}
#preset{flex:0 0 auto;padding:6px 12px;cursor:pointer;color:#e6e6e6;background:#333;border:1px solid #3a3a3a;border-radius:4px;font:inherit;}
#preset:hover{background:#3d3d3d;}
/* Optional, collapsed keyboard-shortcut help under the composer. Deliberately
   NOT tied to the message field (no aria-describedby): a screen reader must not
   re-read it on every focus of the field. */
details.help{flex:0 0 auto;padding:0 10px 8px;color:#9a9a9a;font-size:12px;}
details.help summary{cursor:pointer;padding:2px 0;}
details.help ul{margin:4px 0 0;padding-left:18px;}
details.help li{margin:2px 0;}
/* Cut-by-text editor modal (overlays the whole pane). */
#cutModal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:9999;}
#cutModal[hidden]{display:none;}
.cut-card{display:flex;flex-direction:column;width:min(760px,94vw);height:min(88vh,680px);background:#1e1e1e;border:1px solid #3a3a3a;border-radius:10px;overflow:hidden;}
.cut-head{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid #3a3a3a;}
.cut-title{flex:1;font-weight:700;}
.cut-item{color:#9a9a9a;font-weight:400;margin-left:8px;font-size:12px;}
.cut-ico{background:#2a2a2a;color:#e6e6e6;border:1px solid #3a3a3a;border-radius:5px;padding:4px 8px;cursor:pointer;font:inherit;font-size:12px;}
.cut-ico:hover{background:#333;}
.cut-transport{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid #3a3a3a;}
#cutPlay{width:34px;font-size:14px;}
.cut-summary{color:#9a9a9a;font-size:12px;}
.cut-grid{flex:1 1 auto;overflow-y:auto;padding:14px 16px;line-height:2.1;font-size:16px;outline:none;}
.cut-grid:focus{box-shadow:inset 0 0 0 2px #0e639c;}
.cut-sent{display:inline;}
.cut-tok{border-radius:4px;padding:1px 3px;}
.cut-tok.cur{box-shadow:0 0 0 2px #4ea1ff;}
.cut-tok.sel{background:#264f78;}
.cut-tok.rm{opacity:.45;text-decoration:line-through;color:#8a8a8a;}
.cut-foot{display:flex;align-items:center;gap:8px;padding:10px 14px;border-top:1px solid #3a3a3a;}
.cut-hint{flex:1;color:#8a8a8a;font-size:11px;}
/* Everything the editor announces is ALSO shown here. The live region is
   sr-only by necessity, so without this every announcement -- the word under
   the caret, an undo, a failed copy -- reached screen-reader users only, and a
   sighted user saw nothing at all. */
.cut-say{color:#e6e6e6;}
.cut-say:not(:empty)+#cutKeyHint{display:none;}
.cut-foot button{padding:6px 14px;border-radius:5px;border:1px solid #3a3a3a;background:#2a2a2a;color:#e6e6e6;cursor:pointer;font:inherit;}
.cut-primary{background:#0e639c;border-color:#1177bb;color:#fff;}
</style></head><body>
<div id="live" class="sr" aria-live="polite" aria-atomic="true"></div>
<!-- NOT a live region: streaming re-renders it token-by-token; role="log"/
     aria-live here makes a screen reader announce every token. The final answer
     is spoken via #live + OSARA; turns are navigable by their h2 headings. -->
<div id="log"></div>
<div id="status" role="status" aria-atomic="true" aria-label="Assistant status">Ready.</div>
<form id="composer">
<textarea id="msg" rows="1" aria-label="Message the assistant" placeholder="Ask the assistant…"></textarea>
<button id="preset" type="button" aria-label="Insert a saved prompt preset">Presets</button>
<button id="send" type="submit">Send</button>
<button id="clearchat" type="button" aria-label="Clear the conversation history">Clear</button>
<button id="closewin" type="button" aria-label="Close the assistant window">Close</button>
</form>
<details class="help"><summary>Keyboard shortcuts</summary><ul><li>Enter sends; Shift+Enter starts a new line.</li><li>Alt+1 through Alt+0 read that message; press the same combo again quickly to copy it.</li><li>Alt+P inserts a saved prompt preset.</li><li>Escape stops the assistant while it is working.</li><li>Command/Ctrl+W closes the window (it keeps your conversation and reopens instantly).</li></ul></details>
<div id="cutModal" hidden role="dialog" aria-modal="true" aria-label="Cut by text editor">
  <div class="cut-card">
    <div class="cut-head">
      <div class="cut-title">Cut by text<span id="cutItem" class="cut-item"></span></div>
      <button id="cutAudioMode" type="button" class="cut-ico" title="Cycle audio feedback">Audio: both</button>
      <button id="cutUndo" type="button" class="cut-ico" title="Undo (Ctrl+Z)">Undo</button>
      <button id="cutRedo" type="button" class="cut-ico" title="Redo (Ctrl+Y)">Redo</button>
      <button id="cutKeysBtn" type="button" class="cut-ico" aria-expanded="false"
        aria-controls="cutKeys" title="Show keyboard shortcuts">Keys</button>
      <button id="cutReport" type="button" class="cut-ico"
        title="Copy a diagnostic report for this clip to the clipboard">Report</button>
    </div>
    <div id="cutKeys" hidden role="region" aria-label="Keyboard shortcuts">
      <ul>
        <li><b>Up</b> / <b>Down</b> — previous / next sentence (reads the whole sentence)</li>
        <li><b>Left</b> / <b>Right</b> — previous / next word, within the sentence</li>
        <li><b>Shift</b> + arrows — extend the selection by word or sentence</li>
        <li><b>Space</b> — select or deselect the word at the cursor</li>
        <li><b>Delete</b> / <b>Backspace</b> — remove the word or selection (reversible)</li>
        <li><b>Ctrl+Z</b> / <b>Ctrl+Y</b> — undo / redo, before anything is cut</li>
        <li><b>Home</b> / <b>End</b> — start / end of the sentence</li>
        <li><b>Ctrl+Home</b> / <b>Ctrl+End</b> — start / end of the transcript</li>
        <li><b>Escape</b> — clear the selection; again to cancel</li>
        <li><b>Tab</b> — leave the transcript for the buttons</li>
        <li><b>Play</b> — hear the edited result; <b>Audio</b> cycles what each move plays</li>
      </ul>
    </div>
    <div class="cut-transport">
      <button id="cutPlay" type="button" class="cut-ico" aria-label="Play the edited result">&#9654;</button>
      <div id="cutSummary" class="cut-summary"></div>
    </div>
    <div id="cutGrid" class="cut-grid" role="application" tabindex="0" aria-label="Transcript. Arrow keys move by word and sentence; Space selects; Delete removes; Ctrl+Z undoes; Tab leaves to the buttons."></div>
    <div class="cut-foot">
      <span class="cut-hint"><span id="cutStatus" class="cut-say"></span><span id="cutKeyHint">Up/Down sentence, Left/Right word, Space select, Del remove, Ctrl+Z undo, Tab to buttons</span></span>
      <button id="cutCancel" type="button">Cancel</button>
      <button id="cutConfirm" type="button" class="cut-primary">Confirm cut</button>
    </div>
  </div>
</div>
<script>
function sd(){var l=document.getElementById('log');if(l)l.scrollTop=l.scrollHeight;}
function addBlock(h){var l=document.getElementById('log');if(l){l.insertAdjacentHTML('beforeend',h);sd();}}
function tgCount(g){var b=g.querySelector('.tgbody');var n=b.querySelectorAll('details.tool').length;g.querySelector('.tgsum').textContent='Used '+n+' tool'+(n===1?'':'s');}
function newToolGroup(l){var g=document.createElement('details');g.className='toolgroup';g.innerHTML='<summary class="tgsum" aria-label="tool group"></summary><div class="tgbody"></div>';return g;}
// Group CONSECUTIVE tool cards under one collapsible. A lone tool stays a single
// card; the second consecutive tool promotes the pair into a "Used N tools" group.
// Anything else added to the log (assistant heading, notice, error) ends the run.
function addTool(h){
  var l=document.getElementById('log');if(!l)return;
  var last=l.lastElementChild,cl=last&&last.classList;
  if(cl&&cl.contains('toolgroup')){var b=last.querySelector('.tgbody');b.insertAdjacentHTML('beforeend',h);tgCount(last);sd();return;}
  if(cl&&cl.contains('tool')){var g=newToolGroup(l);l.replaceChild(g,last);var gb=g.querySelector('.tgbody');gb.appendChild(last);gb.insertAdjacentHTML('beforeend',h);tgCount(g);sd();return;}
  l.insertAdjacentHTML('beforeend',h);sd();
}
function startAssistant(){var o=document.getElementById('cur');if(o)o.removeAttribute('id');addBlock('<h2 class="turn assistant">Assistant</h2><div class="body" id="cur"></div>');}
function updateAssistant(h){var c=document.getElementById('cur');if(c){c.innerHTML=h;sd();}}
// Reasoning/"thinking": a collapsible block streamed before the answer, collapsed
// by default (secondary to the answer, and never spoken as the final answer).
function startReasoning(){var o=document.getElementById('rcur');if(o)o.removeAttribute('id');addBlock('<details class="reasoning"><summary>Reasoning</summary><div class="rbody" id="rcur"></div></details>');}
function updateReasoning(h){var c=document.getElementById('rcur');if(c){c.innerHTML=h;sd();}}
function setToolResult(h){var l=document.querySelectorAll('#log details.tool');if(l.length){var t=l[l.length-1].querySelector('.tres');if(t){t.innerHTML=h;sd();}}}
var _liveT=null,_liveAlt=false;
// Announce via the aria-live region. Set the text IMMEDIATELY (the old 60 ms
// clear-then-set delayed every announcement and, under fast navigation, let
// overlapping timers clobber each other so words were silently skipped). The
// toggled zero-width space guarantees the text differs from the previous
// announcement — screen readers drop a live-region update that repeats the same
// string, which otherwise silences repeated words ("the" ... "the").
function liveAnnounce(t){var l=document.getElementById('live');if(!l||!t)return;showSaid(t);if(_liveT){clearTimeout(_liveT);_liveT=null;}_liveAlt=!_liveAlt;l.textContent=t+(_liveAlt?'\u200B':'');_liveT=setTimeout(function(){l.textContent='';_liveT=null;},4000);}
function setStatus(t){var s=document.getElementById('status');if(s)s.textContent=t;}
// A screen-reader-only announcement is invisible by construction, so mirror it to
// a VISIBLE line: the editor's own footer while the editor is open, and the chat
// pane's status line otherwise. Cleared on the same timer as the live region so
// the footer falls back to showing the key hints.
var _sayT=null;
function showSaid(t){
  var c=document.getElementById('cutStatus'),
      open=document.getElementById('cutModal');
  if(_sayT){clearTimeout(_sayT);_sayT=null;}
  if(c&&open&&!open.hidden){
    c.textContent=t;
    _sayT=setTimeout(function(){c.textContent='';_sayT=null;},4000);
  }else{
    if(c)c.textContent='';
    setStatus(t);
  }
}
function clearLog(){var l=document.getElementById('log');if(l)l.innerHTML='';
  var st=document.getElementById('status');if(st)st.textContent='Ready.';
  if(window.liveAnnounce) liveAnnounce('Conversation cleared.');}
function focusInput(){var m=document.getElementById('msg');if(m)m.focus();}
// Prompt presets: the host shows a native picker; on a choice it calls
// insertPreset with the chosen body, spliced at the caret so it stays editable.
function pickPreset(){if(window.ipc)window.ipc.postMessage(JSON.stringify({t:'presets:pick'}));}
function insertPreset(text){var m=document.getElementById('msg');if(!m||!text)return;var a=m.selectionStart,b=m.selectionEnd,v=m.value;m.value=v.slice(0,a)+text+v.slice(b);var c=a+text.length;m.selectionStart=m.selectionEnd=c;grow();m.focus();}
// Copy `t` to the clipboard. execCommand works in the webview under a user
// gesture (about:blank isn't a secure context, so navigator.clipboard may be
// blocked); fall back to it. Restores focus after the hidden-textarea trick.
function copyText(t){var ok=false;try{var p=document.activeElement,ta=document.createElement('textarea');ta.value=t;ta.style.position='fixed';ta.style.left='-9999px';ta.style.top='0';document.body.appendChild(ta);ta.focus();ta.select();try{ok=document.execCommand('copy');}catch(e){}document.body.removeChild(ta);if(p&&p.focus)p.focus();}catch(e){}if(!ok&&navigator.clipboard&&navigator.clipboard.writeText){try{navigator.clipboard.writeText(t);ok=true;}catch(e){}}return ok;}
// Message navigation: each turn is an h2.turn (user or assistant). msgText pulls
// the message's plain text — a user turn's own text, or the assistant turn's body.
function msgHeadings(){return Array.prototype.slice.call(document.querySelectorAll('#log h2.turn'));}
function msgText(h){if(!h)return '';if(h.classList.contains('assistant')){var b=h.nextElementSibling;return (b&&b.classList&&b.classList.contains('body'))?b.textContent.trim():'';}return h.textContent.trim().replace(/^You:\s*/,'');}
var lastMsgNav={n:0,t:0};
// Alt+N reads message N; a quick second Alt+N copies it. The read goes through the
// aria-live region (the same path the copy confirmation uses) rather than moving DOM
// focus to the heading: VoiceOver on the macOS WKWebView only intermittently announces
// a programmatic focus() on a non-interactive (tabindex=-1) element, which made the
// read fire "sometimes". aria-live is reliable on both NVDA and VoiceOver. We announce
// the message's actual text (a user turn's prompt, or the assistant turn's answer), so
// navigating to an assistant message now reads the response, not just the "Assistant"
// heading. Focus stays in the composer; scroll still brings the message into view.
function gotoMessage(n){var hs=msgHeadings();if(n<1||n>hs.length){liveAnnounce(hs.length?('Only '+hs.length+' message'+(hs.length===1?'':'s')+'.'):'No messages yet.');return;}var h=hs[n-1],now=(new Date()).getTime();if(lastMsgNav.n===n&&now-lastMsgNav.t<600){lastMsgNav={n:0,t:0};liveAnnounce(copyText(msgText(h))?'Message copied.':'Copy failed.');return;}lastMsgNav={n:n,t:now};h.scrollIntoView({block:'center'});liveAnnounce(msgText(h)||h.textContent.trim());}
var generating=false;
function setGenerating(b){generating=!!b;}
function grow(){var m=document.getElementById('msg');if(!m)return;m.style.height='auto';var max=Math.round(window.innerHeight*0.4);m.style.height=Math.min(m.scrollHeight,max)+'px';}
(function(){
  var f=document.getElementById('composer'),m=document.getElementById('msg');
  function send(){var t=m.value;if(!t.trim())return;m.value='';grow();window.ipc.postMessage(JSON.stringify({t:'submit',text:t}));}
  f.addEventListener('submit',function(e){e.preventDefault();send();});
  // Bridge clipboard for the composer. A SWELL-hosted WKWebView on macOS doesn't
  // get native Cmd+C/X/V, so route the Cmd variants (metaKey) through the host,
  // which reads/writes the system pasteboard. Ctrl variants (Windows) fall through
  // to the webview's own, already-working native editing.
  function selText(){ return m.value.substring(m.selectionStart,m.selectionEnd); }
  function clipKey(e){
    if(!e.metaKey||e.ctrlKey||e.altKey) return false;
    var k=(e.key||'').toLowerCase();
    if(k==='c'||k==='x'){ var s=selText(); if(s){ window.ipc.postMessage(JSON.stringify({t:'clip:set',text:s}));
        if(k==='x'){ var a=m.selectionStart,b=m.selectionEnd; m.value=m.value.slice(0,a)+m.value.slice(b); m.selectionStart=m.selectionEnd=a; grow(); } }
      e.preventDefault(); return true; }
    if(k==='v'){ window.ipc.postMessage(JSON.stringify({t:'clip:paste'})); e.preventDefault(); return true; }
    return false;
  }
  m.addEventListener('keydown',function(e){
    // Enter sends; Shift+Enter is a newline. Skip while an IME composition is
    // in progress (isComposing) so committing the composition doesn't submit.
    if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}
    // Escape stops an in-flight turn — but only while generating, so a stray/
    // reflexive Escape at rest does nothing.
    else if(e.key==='Escape'&&generating){e.preventDefault();window.ipc.postMessage(JSON.stringify({t:'cancel'}));}
    else clipKey(e);
  });
  m.addEventListener('input',grow);
  var pbtn=document.getElementById('preset');if(pbtn)pbtn.addEventListener('click',pickPreset);
  var kb=document.getElementById('cutKeysBtn'),kp=document.getElementById('cutKeys');
  if(kb&&kp) kb.addEventListener('click',function(){
    var open=kp.hidden; kp.hidden=!open; kb.setAttribute('aria-expanded',open?'true':'false');
    if(window.liveAnnounce) liveAnnounce(open?'Keyboard shortcuts shown':'Keyboard shortcuts hidden');
  });
  var rb=document.getElementById('cutReport');
  if(rb) rb.addEventListener('click',function(){
    rb.textContent='Copying…';
    if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'cut:report'}));
  });
  // Confirm on BOTH channels: the label is the sighted signal, the live region the
  // spoken one — a copy that silently did nothing is indistinguishable from success.
  window.cutReportDone=function(ok){
    var b=document.getElementById('cutReport'); if(!b)return;
    var msg=ok?'Report copied to the clipboard':'No analysis available to report';
    b.textContent=ok?'Copied':'No data';
    if(window.liveAnnounce) liveAnnounce(msg);
    setTimeout(function(){ b.textContent='Report'; },2500);
  };
  var cbtn=document.getElementById('clearchat');
  if(cbtn) cbtn.addEventListener('click',function(){
    if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'chat:clear'}));
  });
  var xbtn=document.getElementById('closewin');if(xbtn)xbtn.addEventListener('click',function(){window.ipc.postMessage(JSON.stringify({t:'window:close'}));});
  // Cmd/Ctrl+W closes the window everywhere (works even when the webview has focus,
  // which on macOS is the only reliable way for a VoiceOver user to close it).
  document.addEventListener('keydown',function(e){
    if((e.metaKey||e.ctrlKey)&&!e.altKey&&!e.shiftKey&&(e.key||'').toLowerCase()==='w'){
      e.preventDefault(); if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'window:close'})); }
  });
  grow();focusInput();
})();
// Links open in the user's default browser; the chat pane must never navigate
// away from the conversation. preventDefault on EVERY <a> click (keyboard Enter
// on a link also fires click, so this covers screen-reader activation), and hand
// http(s)/mailto URLs to the host to launch externally.
document.addEventListener('click',function(e){
  var a=e.target&&e.target.closest?e.target.closest('a[href]'):null;if(!a)return;
  e.preventDefault();
  var u=a.getAttribute('href')||'';
  if(/^(https?:\/\/|mailto:)/i.test(u)&&window.ipc)window.ipc.postMessage(JSON.stringify({t:'openurl',url:u}));
});
// Alt+1..9 / Alt+0 (=10) / Alt+key-right-of-0 (=11) jump between messages; a quick
// second press of the same combo copies that message. Uses e.code so it's layout-
// independent (and preventDefault stops mac Option+digit typing a special char).
document.addEventListener('keydown',function(e){
  if(!e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;
  var c=e.code,n=0;
  if(/^Digit[1-9]$/.test(c))n=+c.slice(5);
  else if(c==='Digit0')n=10;
  else if(c==='Minus')n=11;
  else return;
  e.preventDefault();
  gotoMessage(n);
});
// Alt+P opens the prompt-preset picker. e.code is layout-independent, and
// preventDefault stops mac Option+P typing a special character (\u{03c0}).
document.addEventListener('keydown',function(e){
  if(!e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;
  if(e.code!=='KeyP')return;
  e.preventDefault();
  pickPreset();
});
// ---- Cut-by-text editor (modal). openCutEditor(payload)/closeCutEditor() are
// called from Rust; the user's confirm/cancel posts {t:'cut:save'|'cut:cancel'}. --
(function(){
  var st=null,G=null,ctx=null,buf=null,curSrc=null,playTimer=null,preview=[];
  var MODES=['both','audio','spoken'];
  var cancelArmed=false,cancelTimer=null;
  function $(id){return document.getElementById(id);}
  function announce(t){ if(window.liveAnnounce) liveAnnounce(t); }
  function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  // Strip PUNCTUATION for announcements, not "everything non-ASCII": the old
  // [^A-Za-z0-9'] deleted umlauts and every non-Latin letter, so the screen
  // reader was fed "zrtlich" for "zärtlich". List what to remove, keep the rest.
  function bare(s){ return (s||'').replace(/["'.,!?;:()\[\]{}«»„“”‚‘’…\/\\-]/g,''); }
  function isArrow(k){ return k==='ArrowLeft'||k==='ArrowRight'||k==='ArrowUp'||k==='ArrowDown'; }
  function b64bytes(b){ var s=atob(b),n=s.length,u=new Uint8Array(n); for(var i=0;i<n;i++)u[i]=s.charCodeAt(i); return u; }

  window.openCutEditor=function(json){
    try{
      var d=(typeof json==='string')?JSON.parse(json):json;
      // Merge continuation tokens (j-flagged by the host: Whisper split ONE
      // spoken word, e.g. "89"+"-Jährige") into a single navigable token that
      // spans head start to tail end — including any bogus gap between the
      // halves, so the whole compound auditions as the one word it is. `n`
      // remembers how many transcript words each token stands for, so the
      // keep flags sent back to the host stay per-WORD (indices are
      // load-bearing there).
      var raw=(d.words||[]),words=[];
      for(var wi=0;wi<raw.length;wi++){ var rw=raw[wi];
        var g=words.length?words[words.length-1]:null;
        if(rw.j && g){
          g.t=(g.t||'')+(rw.t||'');
          g.end=rw.end;
          if(typeof rw.f==='number') g.f=rw.f;
          g.n=(g.n||1)+1;
        } else { rw.n=1; words.push(rw); }
      }
      st={words:words,caret:0,anchor:null,mode:'both',undo:[],redo:[],duration:(d.duration||0)};
      cancelArmed=false;
      ctx=null;buf=null;
      try{ var AC=window.AudioContext||window.webkitAudioContext;
        // latencyHint 'interactive' asks for the smallest output buffer — this is
        // keypress-driven feedback, so latency matters more than power draw.
        if(AC && d.wav){ try{ ctx=new AC({latencyHint:'interactive'}); }catch(_){ ctx=new AC(); }
          ctx.decodeAudioData(b64bytes(d.wav).buffer,function(b){buf=b;},function(){buf=null;}); } }catch(e){ctx=null;buf=null;}
      $('cutItem').textContent=d.item?(' — '+d.item):'';
      $('cutAudioMode').textContent='Audio: both';
      G=$('cutGrid'); render();
      inertComposer(true);
      $('cutModal').hidden=false;
      setTimeout(function(){ if(G)G.focus(); },30);
      announce('Cut by text editor. '+st.words.length+' words. Arrows navigate, Delete removes, Tab to the buttons, Confirm to cut.');
    }catch(e){ try{ if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'cut:cancel'})); }catch(_){} }
  };
  window.closeCutEditor=function(){
    stopAll(); var m=$('cutModal'); if(m)m.hidden=true; inertComposer(false);
    st=null; buf=null; if(ctx){ try{ctx.close();}catch(e){} ctx=null; }
    try{ focusInput(); }catch(e){}
  };
  function inertComposer(on){ ['msg','send','preset'].forEach(function(id){ var el=$(id); if(!el)return;
    if(on){ el.setAttribute('data-pt', el.getAttribute('tabindex')||''); el.setAttribute('tabindex','-1'); }
    else { var p=el.getAttribute('data-pt'); if(p==='') el.removeAttribute('tabindex'); else if(p!=null) el.setAttribute('tabindex',p); el.removeAttribute('data-pt'); } }); }

  function maxSent(){ return st.words.length?st.words[st.words.length-1].s:0; }
  function firstOf(s){ for(var i=0;i<st.words.length;i++) if(st.words[i].s===s) return i; return 0; }
  function lastOf(s){ var r=0; for(var i=0;i<st.words.length;i++) if(st.words[i].s===s) r=i; return r; }
  function wordR(i){ var n=i+1; return (n<st.words.length && st.words[n].s===st.words[i].s)?n:i; }
  function wordL(i){ var n=i-1; return (n>=0 && st.words[n].s===st.words[i].s)?n:i; }
  function sentText(s){ var o=[],rm=0; for(var i=0;i<st.words.length;i++){ if(st.words[i].s===s){ o.push(st.words[i].t); if(st.words[i].rm)rm++; } } return o.join(' ')+(rm?(' ('+rm+' removed)'):''); }
  function stWord(){ var w=st.words[st.caret]; return bare(w.t)+(w.sel?', selected':'')+(w.rm?', removed':''); }
  function clearSel(){ for(var i=0;i<st.words.length;i++) st.words[i].sel=false; st.anchor=null; }
  function extend(){ if(st.anchor===null) st.anchor=st.caret; var a=Math.min(st.anchor,st.caret),b=Math.max(st.anchor,st.caret); for(var i=0;i<st.words.length;i++) st.words[i].sel=(i>=a&&i<=b); }

  function summary(){
    var rmDur=0,spans=0,prev=false;
    for(var j=0;j<st.words.length;j++){ var w2=st.words[j]; if(w2.rm){ rmDur+=Math.max(0,w2.end-w2.start); if(!prev)spans++; } prev=w2.rm; }
    var total=st.duration||sumDur(),kept=Math.max(0,total-rmDur);
    $('cutSummary').textContent='Cutting '+spans+' span'+(spans===1?'':'s')+' · '+rmDur.toFixed(1)+' s removed · '+(total>0?Math.round(kept/total*100):100)+'% kept';
  }
  // Build the token DOM ONCE (on open). The token set never changes — only their
  // state does — so navigation must not rebuild it.
  function render(){
    var h='',cs=-1;
    for(var i=0;i<st.words.length;i++){ var w=st.words[i];
      if(w.s!==cs){ if(cs!==-1)h+='</span> '; h+='<span class="cut-sent">'; cs=w.s; }
      h+='<span class="cut-tok'+(i===st.caret?' cur':'')+(w.sel?' sel':'')+(w.rm?' rm':'')+'" data-i="'+i+'">'+esc(w.t)+'</span> ';
    }
    if(cs!==-1)h+='</span>';
    G.innerHTML=h;
    summary();
  }
  // Incremental repaint: sync only the classes that changed. Rebuilding innerHTML
  // on every arrow key was both slow on long transcripts AND tore down the DOM
  // under the screen reader mid-announcement, which dropped spoken words.
  function paint(){
    if(!G) return;
    var els=G.getElementsByClassName('cut-tok'),n=Math.min(els.length,st.words.length);
    for(var i=0;i<n;i++){ var w=st.words[i];
      var c='cut-tok'+(i===st.caret?' cur':'')+(w.sel?' sel':'')+(w.rm?' rm':'');
      if(els[i].className!==c) els[i].className=c;
    }
    var cur=els[st.caret];
    if(cur&&cur.scrollIntoView){ try{ cur.scrollIntoView({block:'nearest'}); }catch(_){ } }
    summary();
  }
  function sumDur(){ var t=0; for(var i=0;i<st.words.length;i++) t+=Math.max(0,st.words[i].end-st.words[i].start); return t; }

  function stopSnip(){ if(curSrc){ try{curSrc.stop();}catch(e){} curSrc=null; } }
  function stopPreview(){ preview.forEach(function(s){ try{s.stop();}catch(e){} }); preview=[]; }
  function stopAll(){ if(playTimer){clearTimeout(playTimer);playTimer=null;} stopSnip(); stopPreview(); }
  function resume(){ if(ctx && ctx.state==='suspended'){ try{ctx.resume();}catch(e){} } }
  // Whisper's word boundaries come from attention alignment, not a forced aligner,
  // so they run a few tens of ms early/late and clip onsets. Pad each snippet a
  // little (and a bit more at the tail) so you hear the WHOLE word — this only
  // affects preview playback, never where the cut lands.
  // Pads are capped at HALF the actual gap to the neighbouring word, so a fixed pad
  // can't bleed into the next word on a fast read with short breaks (the old fixed
  // 90 ms tail played ~60 ms of the next word whenever the gap was 30 ms). Floors
  // keep a minimum pad, because the pad exists to stop Whisper's tight boundaries
  // clipping the onset — deriving it purely from the gap would collapse it to zero
  // exactly on the connected speech that needs it most.
  var PAD_IN=0.04,PAD_OUT=0.09,PAD_IN_MIN=0.015,PAD_OUT_MIN=0.02,FADE=0.012;
  function playRange(s,e,i){
    if(!ctx||!buf)return; stopSnip();
    try{
      var lead=PAD_IN,tail=PAD_OUT;
      if(st&&typeof i==='number'&&st.words){
        var w=st.words[i],p=st.words[i-1],nx=st.words[i+1];
        // Measured extent when the host could determine it: what you hear is then
        // exactly what deleting this word would remove.
        if(w&&typeof w.o==='number'&&typeof w.f==='number'&&w.f>w.o){ s=w.o; e=w.f; }
        var pe=(p&&typeof p.f==='number')?p.f:(p?p.end:null);
        var ns=(nx&&typeof nx.o==='number')?nx.o:(nx?nx.start:null);
        if(pe!==null) lead=Math.min(PAD_IN,Math.max(0,0.5*(s-pe)));
        if(ns!==null) tail=Math.min(PAD_OUT,Math.max(0,0.5*(ns-e)));
      }
      var o=Math.max(0,s-lead), d=Math.max(0.02,(e-o)+tail);
      if(buf.duration) d=Math.min(d,Math.max(0.02,buf.duration-o));
      var n=ctx.createBufferSource(); n.buffer=buf;
      // Short fade-out so any residual bleed reads as decay, and a hard stop
      // mid-waveform can't click.
      var g=ctx.createGain(); n.connect(g); g.connect(ctx.destination);
      var t0=ctx.currentTime, f=Math.min(FADE,d/3);
      g.gain.setValueAtTime(1,t0+Math.max(0,d-f));
      g.gain.linearRampToValueAtTime(0.0001,t0+d);
      n.start(0,o,d); curSrc=n;
    }catch(_){} }
  // `fast` = a single deliberate keypress: start the audio NOW. Only auto-repeat
  // (holding an arrow) settles first, so holding a key doesn't machine-gun the
  // audio. The old code delayed EVERY press by 140 ms, which is what made single
  // arrow presses feel sluggish.
  function snippet(s,e,fast,i){ if(playTimer){clearTimeout(playTimer);playTimer=null;} stopSnip(); if(st.mode==='spoken')return;
    if(fast){ playRange(s,e,i); return; }
    playTimer=setTimeout(function(){ playRange(s,e,i); },110); }
  // Preview the edited result. A kept run is broken ONLY where a word was actually
  // removed — never on a duration threshold. The old code split runs at any gap over
  // 50 ms, which silently deleted every natural pause from the preview, so the
  // preview was always tighter than the real cut and could never be trusted to judge
  // it. Pauses BETWEEN kept words are part of the audio and stay.
  // Ask the host for the segments the REAL cut would leave. The transcript times we
  // hold are 50-200 ms out and the cut uses measured boundaries, so previewing from
  // them auditions something that never gets written.
  function playEdited(){ if(!ctx||!buf){ announce('No audio to preview'); return; }
    try{ if(window.ipc){ window.ipc.postMessage(JSON.stringify({t:'cut:preview',
      keep: keepFlags()})); announce('Preparing preview…'); return; } }catch(e){}
    playSegments(null); }
  window.cutPreviewSegments=function(json){
    var segs=null; try{ segs=(typeof json==='string')?JSON.parse(json):json; }catch(e){}
    playSegments(segs && segs.length ? segs : null); };
  function playSegments(segs){ if(!ctx||!buf||!st)return; resume(); stopAll();
    var ranges=[];
    if(segs){ for(var j=0;j<segs.length;j++) ranges.push({s:segs[j][0], e:segs[j][1]}); }
    else {
      // Fallback only when the host could not measure: transcript times.
      var cur=null;
      for(var i=0;i<st.words.length;i++){ var w=st.words[i];
        if(w.rm){ if(cur){ranges.push(cur);cur=null;} continue; }
        if(cur) cur.e=Math.max(cur.e,w.end); else cur={s:w.start,e:w.end};
      }
      if(cur)ranges.push(cur);
    }
    if(!ranges.length){ announce('Everything is removed'); return; }
    // Butt the kept runs together — that IS the cut. Half the pause on each side of
    // a removed span survives the real cut, so approximate that here too.
    var at=ctx.currentTime+0.03;
    ranges.forEach(function(r,k){ try{
      // No invented lead-in: these ARE the cut boundaries.
      var o=Math.max(0,r.s);
      var d=Math.max(0.02,r.e-o);
      if(buf.duration) d=Math.min(d,Math.max(0.02,buf.duration-o));
      var n=ctx.createBufferSource(); n.buffer=buf; n.connect(ctx.destination);
      n.start(at,o,d); preview.push(n); at+=d; }catch(_){} });
    announce('Playing the edited result');
  }

  function snap(){ return st.words.map(function(w){return w.rm;}); }
  function apply(s){ for(var i=0;i<st.words.length;i++) st.words[i].rm=!!s[i]; }
  function pushUndo(){ st.undo.push(snap()); st.redo.length=0; }
  function rmSummary(){ var n=st.words.filter(function(w){return w.rm;}).length; return n+' word'+(n===1?'':'s')+' to remove'; }
  function undo(){ if(!st.undo.length){announce('Nothing to undo');return;} st.redo.push(snap()); apply(st.undo.pop()); paint(); announce('Undone. '+rmSummary()); }
  function redo(){ if(!st.redo.length){announce('Nothing to redo');return;} st.undo.push(snap()); apply(st.redo.pop()); paint(); announce('Redone. '+rmSummary()); }
  function toggleRemove(){ pushUndo(); var any=st.words.some(function(w){return w.sel;});
    if(any){ var all=st.words.filter(function(w){return w.sel;}).every(function(w){return w.rm;}); for(var i=0;i<st.words.length;i++) if(st.words[i].sel) st.words[i].rm=!all; clearSel(); }
    else { st.words[st.caret].rm=!st.words[st.caret].rm; } }

  function afterNav(sentence,fast){ paint(); var w=st.words[st.caret];
    if(st.mode!=='audio') announce(sentence?sentText(w.s):stWord());
    snippet(w.start,w.end,fast!==false,st.caret); }

  function focusables(){ return Array.prototype.slice.call(document.querySelectorAll('#cutModal button, #cutGrid')); }
  function postCancel(){ try{ if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'cut:cancel'})); }catch(e){} closeCutEditor(); }
  function requestCancel(){ var pending=st.words.some(function(w){return w.rm;})||st.undo.length;
    if(pending && !cancelArmed){ cancelArmed=true; announce('You have edits. Press Cancel or Escape again to discard.'); if(cancelTimer)clearTimeout(cancelTimer); cancelTimer=setTimeout(function(){cancelArmed=false;},4000); return; }
    postCancel(); }
  // Per-WORD keep flags: merged tokens expand back to one flag per transcript
  // word (the host's indices are load-bearing).
  function keepFlags(){ var k=[]; st.words.forEach(function(w){ var v=!w.rm; for(var m=0;m<(w.n||1);m++) k.push(v); }); return k; }
  function confirmCut(){ var keep=keepFlags(); try{ if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'cut:save',keep:keep})); }catch(e){} closeCutEditor(); }

  document.addEventListener('keydown',function(e){
    if(!st) return; var m=$('cutModal'); if(!m||m.hidden) return;
    var k=e.key,mod=e.ctrlKey||e.metaKey,inGrid=(document.activeElement===G);
    if(k==='Tab'){ var f=focusables(); if(f.length){ var first=f[0],last=f[f.length-1];
      if(e.shiftKey && document.activeElement===first){ e.preventDefault(); last.focus(); }
      else if(!e.shiftKey && document.activeElement===last){ e.preventDefault(); first.focus(); } }
      e.stopPropagation(); return; }
    if(mod && (k==='z'||k==='Z')){ e.preventDefault(); e.stopPropagation(); if(e.shiftKey)redo(); else undo(); return; }
    if(mod && (k==='y'||k==='Y')){ e.preventDefault(); e.stopPropagation(); redo(); return; }
    e.stopPropagation(); // keep the composer's Alt+N/Alt+P/Enter off while the modal is open
    if(!inGrid) return;  // let the footer buttons handle their own Enter/Space
    resume();
    var handled=true;
    // A held-down arrow (auto-repeat) settles before playing; a single deliberate
    // press plays immediately.
    var fast=!e.repeat;
    if(mod && k==='Home'){ st.caret=0; clearSel(); afterNav(false,fast); }
    else if(mod && k==='End'){ st.caret=st.words.length-1; clearSel(); afterNav(false,fast); }
    else if(e.shiftKey && isArrow(k)){
      if(k==='ArrowRight') st.caret=wordR(st.caret);
      else if(k==='ArrowLeft') st.caret=wordL(st.caret);
      else if(k==='ArrowDown') st.caret=firstOf(Math.min(maxSent(),st.words[st.caret].s+1));
      else st.caret=firstOf(Math.max(0,st.words[st.caret].s-1));
      extend(); afterNav(k==='ArrowUp'||k==='ArrowDown',fast);
    }
    else if(k==='ArrowRight'){ st.caret=wordR(st.caret); clearSel(); afterNav(false,fast); }
    else if(k==='ArrowLeft'){ st.caret=wordL(st.caret); clearSel(); afterNav(false,fast); }
    else if(k==='ArrowDown'){ st.caret=firstOf(Math.min(maxSent(),st.words[st.caret].s+1)); clearSel(); afterNav(true,fast); }
    else if(k==='ArrowUp'){ st.caret=firstOf(Math.max(0,st.words[st.caret].s-1)); clearSel(); afterNav(true,fast); }
    else if(k==='Home'){ st.caret=firstOf(st.words[st.caret].s); clearSel(); afterNav(false,fast); }
    else if(k==='End'){ st.caret=lastOf(st.words[st.caret].s); clearSel(); afterNav(false,fast); }
    else if(k===' '||k==='Spacebar'){ var w=st.words[st.caret]; w.sel=!w.sel; st.anchor=w.sel?st.caret:null; paint(); if(st.mode!=='audio') announce(bare(w.t)+(w.sel?', selected':', deselected')); snippet(w.start,w.end,true,st.caret); }
    else if(k==='Delete'||k==='Backspace'){ toggleRemove(); paint(); announce('Removed. '+rmSummary()); }
    else if(k==='Escape'){ var anySel=st.words.some(function(w){return w.sel;}); if(anySel){ clearSel(); paint(); announce('Selection cleared'); } else { requestCancel(); } }
    else handled=false;
    if(handled) e.preventDefault();
  },true);

  $('cutAudioMode').addEventListener('click',function(){ var i=MODES.indexOf(st.mode); st.mode=MODES[(i+1)%MODES.length]; this.textContent='Audio: '+st.mode; announce('Audio feedback: '+st.mode); if(G)G.focus(); });
  $('cutUndo').addEventListener('click',function(){ undo(); if(G)G.focus(); });
  $('cutRedo').addEventListener('click',function(){ redo(); if(G)G.focus(); });
  $('cutPlay').addEventListener('click',function(){ playEdited(); });
  $('cutCancel').addEventListener('click',function(){ requestCancel(); });
  $('cutConfirm').addEventListener('click',function(){ confirmCut(); });
  $('cutGrid').addEventListener('click',function(e){ var el=e.target.closest('.cut-tok'); if(!el||!st)return; st.caret=+el.getAttribute('data-i'); clearSel(); afterNav(false,true); G.focus(); });
})();
// Tell the host the page has loaded (the worker waits on this before opening the
// cut-by-text editor modal, so it isn't injected before openCutEditor exists).
try{ if(window.ipc) window.ipc.postMessage(JSON.stringify({t:'ui:ready'})); }catch(e){}
</script></body></html>"#;

    // WebView2 is COM and requires the calling (UI) thread to be in a
    // single-threaded apartment. wry does not initialize COM when hosting in a
    // foreign HWND, so do it ourselves; harmless if the thread is already STA
    // (returns S_FALSE) and we deliberately never CoUninitialize.
    #[cfg(windows)]
    #[link(name = "ole32")]
    extern "system" {
        fn CoInitializeEx(reserved: *mut std::ffi::c_void, coinit: u32) -> i32;
    }
    #[cfg(windows)]
    const COINIT_APARTMENTTHREADED: u32 = 0x2;

    pub fn create() -> Result<WebView, String> {
        let hwnd = ffi::get_hwnd() as isize;
        if hwnd == 0 {
            return Err("dialog window handle is null".into());
        }
        let host = Host(hwnd);
        let (x, y, w, h) = ffi::output_bounds().ok_or("output area bounds unavailable")?;
        #[cfg(windows)]
        unsafe {
            CoInitializeEx(std::ptr::null_mut(), COINIT_APARTMENTTHREADED)
        };

        // WebView2's default user-data folder sits next to the host exe
        // (reaper.exe, in read-only Program Files) -> E_ACCESSDENIED, so point it
        // at a writable per-user folder. WKWebView uses its default data store.
        #[cfg(windows)]
        let data_dir: Option<PathBuf> = {
            let dir = user_data_dir()?;
            let _ = std::fs::create_dir_all(&dir);
            Some(dir)
        };
        #[cfg(not(windows))]
        let data_dir: Option<PathBuf> = None;
        // wry requires the WebContext to OUTLIVE the WebView: dropping it while the
        // webview is still alive breaks custom protocols (notably on macOS). It used
        // to be a local that died at the end of this function. The pane lives for the
        // process, so leak it deliberately — one small allocation per webview build,
        // and builds happen at most once per REAPER session.
        let web_context: &'static mut WebContext = Box::leak(Box::new(WebContext::new(data_dir)));

        WebViewBuilder::new_with_web_context(web_context)
            .with_bounds(bounds(x, y, w, h))
            .with_html(BASE_HTML)
            .with_transparent(false)
            // The chat composer lives in the HTML now; its Send/Enter posts here.
            // Panic-guarded: this fires from a WebView2/COM callback, and a panic
            // must never unwind across that boundary (design N3). Take the body as
            // an owned String first so the catch_unwind closure is UnwindSafe.
            .with_ipc_handler(|req: Request<String>| {
                let body = req.into_body();
                let _ = std::panic::catch_unwind(move || {
                    crate::ui::bridge::on_webview_message(&body);
                });
            })
            // Backstop to the in-page click handler: the pane loads via
            // NavigateToString (about:blank), so any http(s) navigation can only
            // come from a link. Cancel it and open it externally instead — the
            // conversation must never be replaced by a web page.
            .with_navigation_handler(|uri: String| {
                let lower = uri.to_ascii_lowercase();
                if lower.starts_with("http://") || lower.starts_with("https://") {
                    let _ = std::panic::catch_unwind(move || crate::ui::bridge::open_url(&uri));
                    false // deny in-pane navigation
                } else {
                    true // the pane's own content (about:blank, data:) — allow
                }
            })
            .build_as_child(&host)
            .map_err(|e| e.to_string())
    }

    #[cfg(windows)]
    fn user_data_dir() -> Result<PathBuf, String> {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or("LOCALAPPDATA is not set")?;
        Ok(base.join("ReaLackey").join("WebView2"))
    }

    pub fn set_bounds(webview: &WebView, x: i32, y: i32, w: i32, h: i32) {
        let _ = webview.set_bounds(bounds(x, y, w, h));
    }

    fn bounds(x: i32, y: i32, w: i32, h: i32) -> Rect {
        // WebView2 (Windows) positions the child in device pixels, and the Win32
        // client rect we get is already in device pixels, so there it's Physical.
        //
        // wry's WKWebView path instead DIVIDES the incoming Rect by the backing
        // scale factor (`bounds.to_logical(backingScaleFactor)`) to get the NSView
        // frame in points. But SWELL's geometry is ALREADY in Cocoa points, so on
        // macOS we must hand wry Logical units — otherwise `to_logical` divides a
        // second time and the pane renders at 1/scale size in the top-left corner
        // on a Retina display. (Logical.to_logical(sf) is the identity.)
        #[cfg(target_os = "macos")]
        {
            Rect {
                position: LogicalPosition::new(x, y).into(),
                size: LogicalSize::new(w.max(1), h.max(1)).into(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Rect {
                position: PhysicalPosition::new(x, y).into(),
                size: PhysicalSize::new(w.max(1) as u32, h.max(1) as u32).into(),
            }
        }
    }

    /// Move keyboard focus from the (empty) host window into the web content.
    /// Called when the host gains focus (window activation / Tab onto the host).
    /// PROGRAMMATIC hands the WebView2 focus without picking an element; the
    /// caller then runs `focusInput()` to land the caret in the composer.
    #[cfg(windows)]
    pub fn move_focus_into_content(controller: &ICoreWebView2Controller) {
        // SAFETY: called on the main (UI) thread that owns the controller.
        unsafe {
            let _ = controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
        }
    }

    /// Register a MoveFocusRequested handler so that when Tab would walk off the
    /// end (or start) of the web content, focus wraps back to the composer inside
    /// the webview instead of escaping to the (now hidden) native controls. The
    /// whole window is the webview, so focus should never leave it.
    #[cfg(windows)]
    pub fn install_focus_out_handler() {
        use wry::WebViewExtWindows;
        let controller =
            super::STATE.with(|c| c.borrow().webview.as_ref().map(|wv| wv.controller()));
        let Some(controller) = controller else {
            return;
        };

        let handler = MoveFocusRequestedEventHandler::create(Box::new(
            move |_ctrl: Option<ICoreWebView2Controller>,
                  args: Option<ICoreWebView2MoveFocusRequestedEventArgs>|
                  -> windows_core::Result<()> {
                if let Some(args) = args {
                    // Keep focus in the pane: send the caret back to the composer.
                    super::STATE.with(|c| c.borrow().eval("focusInput();"));
                    // SAFETY: fires on the UI thread; `args` is a live COM pointer.
                    unsafe { args.SetHandled(true)? };
                }
                Ok(())
            },
        ));

        // WebView2 AddRefs the handler and keeps it alive; the registration lives
        // as long as the controller (i.e. the webview) does, so we drop `token`.
        let mut token: i64 = 0;
        // SAFETY: main-thread COM call on a live controller.
        unsafe {
            let _ = controller.add_MoveFocusRequested(&handler, &mut token);
        }
    }
}
