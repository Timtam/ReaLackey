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
    #[cfg(windows)]
    webview: Option<wry::WebView>,
    /// Accumulated Markdown of the assistant message currently streaming.
    assistant_md: String,
}

impl Output {
    fn new() -> Self {
        Self {
            #[cfg(windows)]
            webview: None,
            assistant_md: String::new(),
        }
    }

    fn active(&self) -> bool {
        #[cfg(windows)]
        {
            self.webview.is_some()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    fn eval(&self, js: &str) {
        #[cfg(windows)]
        if let Some(wv) = &self.webview {
            let _ = wv.evaluate_script(js);
        }
        #[cfg(not(windows))]
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
            self.call_js("addBlock", &html);
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
    #[cfg(windows)]
    {
        // Build the webview WITHOUT holding the STATE borrow: creating the child
        // window can synchronously fire WM_SIZE -> on_resize(), which borrows STATE.
        match webview_impl::create() {
            Ok(webview) => {
                STATE.with(|c| c.borrow_mut().webview = Some(webview));
                ffi::set_output_edit_visible(false);
                ffi::enable_webview_tabstop();
                ffi::install_webview_focus_cb();
                #[cfg(windows)]
                webview_impl::install_focus_out_handler();
                console("REAPER AI Assistant: HTML output pane active.\n");
            }
            Err(e) => {
                console(&format!(
                    "REAPER AI Assistant: HTML pane unavailable, using plain text output. \
                     Reason: {e}\n"
                ));
            }
        }
    }
    #[cfg(not(windows))]
    {
        console("REAPER AI Assistant: plain text output (webview only on Windows).\n");
    }
}

/// Print a line to REAPER's console (main-thread REAPER handle).
fn console(msg: &str) {
    let _ = crate::reaper::api::with(|r| r.show_console_msg(msg));
}

/// Drop the webview when its parent dialog is destroyed, so it never lingers
/// with a dangling parent (and `ensure_created` will rebuild on re-open).
pub fn on_destroy() {
    #[cfg(windows)]
    {
        // Take the webview out (releasing the borrow) before dropping it.
        let webview = STATE.with(|c| c.borrow_mut().webview.take());
        drop(webview);
    }
}

/// Re-bound the webview to the output area after a dialog resize.
pub fn on_resize() {
    #[cfg(windows)]
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
        }
    }
}

// ---- Windows: the wry hosting ----------------------------------------------

#[cfg(windows)]
mod webview_impl {
    use std::num::NonZeroIsize;
    use std::path::PathBuf;

    use raw_window_handle::{
        HandleError, HasWindowHandle, RawWindowHandle, Win32WindowHandle, WindowHandle,
    };
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Controller, ICoreWebView2MoveFocusRequestedEventArgs,
        COREWEBVIEW2_MOVE_FOCUS_REASON, COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT,
        COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC,
    };
    use webview2_com::MoveFocusRequestedEventHandler;
    use wry::dpi::{PhysicalPosition, PhysicalSize};
    use wry::{Rect, WebContext, WebView, WebViewBuilder};

    use crate::ui::ffi;

    /// Borrow-only wrapper so wry can host a webview in the dialog's HWND.
    struct Host(NonZeroIsize);

    impl HasWindowHandle for Host {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            let mut handle = Win32WindowHandle::new(self.0);
            handle.hinstance = None;
            // SAFETY: the dialog owns the HWND and outlives the webview, which is
            // created, used, and dropped on this (main) thread while it exists.
            Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    const BASE_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; font-src data:;">
<style>
:root{color-scheme:dark;}
html,body{margin:0;padding:0;height:100%;font:13px/1.55 "Segoe UI",system-ui,sans-serif;background:#1e1e1e;color:#e6e6e6;}
#log{padding:10px;}
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
.sr{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;}
</style></head><body>
<div id="live" class="sr" aria-live="polite" aria-atomic="true"></div>
<div id="log"></div><script>
function sd(){window.scrollTo(0,document.body.scrollHeight);}
function addBlock(h){var l=document.getElementById('log');if(l){l.insertAdjacentHTML('beforeend',h);sd();}}
function startAssistant(){var o=document.getElementById('cur');if(o)o.removeAttribute('id');addBlock('<h2 class="turn assistant">Assistant</h2><div class="body" id="cur"></div>');}
function updateAssistant(h){var c=document.getElementById('cur');if(c){c.innerHTML=h;sd();}}
function setToolResult(h){var l=document.querySelectorAll('#log details.tool');if(l.length){var t=l[l.length-1].querySelector('.tres');if(t){t.innerHTML=h;sd();}}}
function liveAnnounce(t){var l=document.getElementById('live');if(!l)return;l.textContent='';setTimeout(function(){l.textContent=t;setTimeout(function(){l.textContent='';},2500);},60);}
</script></body></html>"#;

    // WebView2 is COM and requires the calling (UI) thread to be in a
    // single-threaded apartment. wry does not initialize COM when hosting in a
    // foreign HWND, so do it ourselves; harmless if the thread is already STA
    // (returns S_FALSE) and we deliberately never CoUninitialize.
    #[link(name = "ole32")]
    extern "system" {
        fn CoInitializeEx(reserved: *mut std::ffi::c_void, coinit: u32) -> i32;
    }
    const COINIT_APARTMENTTHREADED: u32 = 0x2;

    pub fn create() -> Result<WebView, String> {
        let hwnd = ffi::get_hwnd() as isize;
        let host = Host(NonZeroIsize::new(hwnd).ok_or("dialog window handle is null")?);
        let (x, y, w, h) = ffi::output_bounds().ok_or("output area bounds unavailable")?;
        unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_APARTMENTTHREADED) };

        // WebView2's default user-data folder sits next to the host exe
        // (reaper.exe, in read-only Program Files) -> E_ACCESSDENIED. Point it at
        // a writable per-user folder instead. `web_context` only needs to live
        // until build_as_child returns.
        let data_dir = user_data_dir()?;
        let _ = std::fs::create_dir_all(&data_dir);
        let mut web_context = WebContext::new(Some(data_dir));

        WebViewBuilder::new_with_web_context(&mut web_context)
            .with_bounds(bounds(x, y, w, h))
            .with_html(BASE_HTML)
            .with_transparent(false)
            .build_as_child(&host)
            .map_err(|e| e.to_string())
    }

    fn user_data_dir() -> Result<PathBuf, String> {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or("LOCALAPPDATA is not set")?;
        Ok(base.join("REAPER-AI-Assistant").join("WebView2"))
    }

    pub fn set_bounds(webview: &WebView, x: i32, y: i32, w: i32, h: i32) {
        let _ = webview.set_bounds(bounds(x, y, w, h));
    }

    fn bounds(x: i32, y: i32, w: i32, h: i32) -> Rect {
        Rect {
            position: PhysicalPosition::new(x, y).into(),
            size: PhysicalSize::new(w.max(1) as u32, h.max(1) as u32).into(),
        }
    }

    /// Move keyboard focus from the (empty) host window into the web content.
    /// Called when the host gains focus via Tab.
    ///
    /// We use PROGRAMMATIC, not NEXT: the pane is a read-only log with no
    /// tab-stops, so NEXT would find nothing to land on and immediately fire
    /// MoveFocusRequested, bouncing focus straight back out. PROGRAMMATIC gives
    /// the document itself focus so the user can actually read/scroll it; a
    /// subsequent Tab then exits cleanly via the MoveFocusRequested handler.
    pub fn move_focus_into_content(controller: &ICoreWebView2Controller) {
        // SAFETY: called on the main (UI) thread that owns the controller.
        unsafe {
            let _ = controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
        }
    }

    /// Register a MoveFocusRequested handler so that when Tab walks off the last
    /// (or first) element of the web content, we hand focus to the native control
    /// after/before the webview instead of letting WebView2 trap it inside.
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
                    // SAFETY: fires on the UI thread; `args` is a live COM pointer.
                    let mut reason = COREWEBVIEW2_MOVE_FOCUS_REASON(0);
                    unsafe { args.Reason(&mut reason)? };
                    let forward = reason == COREWEBVIEW2_MOVE_FOCUS_REASON_NEXT;
                    ffi::focus_after_webview(forward);
                    // We took focus over; stop WebView2 from cycling back inside.
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
