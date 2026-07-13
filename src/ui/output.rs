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

use crate::text::{html_escape, markdown_to_html};
use crate::ui::ffi;

thread_local! {
    static STATE: RefCell<Output> = RefCell::new(Output::new());
}

struct Output {
    #[cfg(webview)]
    webview: Option<wry::WebView>,
    /// Accumulated Markdown of the assistant message currently streaming.
    assistant_md: String,
}

impl Output {
    fn new() -> Self {
        Self {
            #[cfg(webview)]
            webview: None,
            assistant_md: String::new(),
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
        if self.active() {
            self.eval("startAssistant();");
        } else {
            ffi::append_output("Assistant: ");
        }
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

    /// Tell the composer whether a turn is in flight (gates its Escape = stop).
    fn set_generating(&self, on: bool) {
        if self.active() {
            self.eval(if on { "setGenerating(true);" } else { "setGenerating(false);" });
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
/// Update the webview status line. No-op in the plain-text fallback (the native
/// status field is driven separately via `ffi::set_status`).
pub fn status(text: &str) {
    STATE.with(|c| c.borrow().status(text));
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
h2.turn{font-size:13px;font-weight:700;margin:16px 0 4px;line-height:1.4;}
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
.sr{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;}
#status{flex:0 0 auto;padding:2px 10px;color:#9a9a9a;font-size:12px;min-height:15px;}
#composer{flex:0 0 auto;display:flex;gap:6px;padding:8px 10px 10px;border-top:1px solid #3a3a3a;background:#252526;}
#msg{flex:1 1 auto;min-width:0;resize:none;overflow-y:auto;font:inherit;color:#e6e6e6;background:#1e1e1e;border:1px solid #3a3a3a;border-radius:4px;padding:6px 8px;}
#send{flex:0 0 auto;padding:6px 16px;cursor:pointer;color:#fff;background:#0e639c;border:1px solid #1177bb;border-radius:4px;font:inherit;}
#send:hover{background:#1177bb;}
</style></head><body>
<div id="live" class="sr" aria-live="polite" aria-atomic="true"></div>
<!-- NOT a live region: streaming re-renders it token-by-token; role="log"/
     aria-live here makes a screen reader announce every token. The final answer
     is spoken via #live + OSARA; turns are navigable by their h2 headings. -->
<div id="log"></div>
<div id="status" role="status" aria-atomic="true" aria-label="Assistant status">Ready.</div>
<form id="composer">
<textarea id="msg" rows="1" aria-label="Message the assistant" placeholder="Ask the assistant…  (Enter to send, Shift+Enter = new line)"></textarea>
<button id="send" type="submit">Send</button>
</form><script>
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
function setToolResult(h){var l=document.querySelectorAll('#log details.tool');if(l.length){var t=l[l.length-1].querySelector('.tres');if(t){t.innerHTML=h;sd();}}}
function liveAnnounce(t){var l=document.getElementById('live');if(!l)return;l.textContent='';setTimeout(function(){l.textContent=t;setTimeout(function(){l.textContent='';},2500);},60);}
function setStatus(t){var s=document.getElementById('status');if(s)s.textContent=t;}
function focusInput(){var m=document.getElementById('msg');if(m)m.focus();}
var generating=false;
function setGenerating(b){generating=!!b;}
function grow(){var m=document.getElementById('msg');if(!m)return;m.style.height='auto';var max=Math.round(window.innerHeight*0.4);m.style.height=Math.min(m.scrollHeight,max)+'px';}
(function(){
  var f=document.getElementById('composer'),m=document.getElementById('msg');
  function send(){var t=m.value;if(!t.trim())return;m.value='';grow();window.ipc.postMessage(JSON.stringify({t:'submit',text:t}));}
  f.addEventListener('submit',function(e){e.preventDefault();send();});
  m.addEventListener('keydown',function(e){
    // Enter sends; Shift+Enter is a newline. Skip while an IME composition is
    // in progress (isComposing) so committing the composition doesn't submit.
    if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}
    // Escape stops an in-flight turn — but only while generating, so a stray/
    // reflexive Escape at rest does nothing.
    else if(e.key==='Escape'&&generating){e.preventDefault();window.ipc.postMessage(JSON.stringify({t:'cancel'}));}
  });
  m.addEventListener('input',grow);
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
        let mut web_context = WebContext::new(data_dir);

        WebViewBuilder::new_with_web_context(&mut web_context)
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
