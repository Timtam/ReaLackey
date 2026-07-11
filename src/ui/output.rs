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
            let html = format!(
                "<div class=\"msg user\"><span class=\"who\">You</span>{}</div>",
                html_escape(text)
            );
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
        if let Some(webview) = webview_impl::create() {
            STATE.with(|c| c.borrow_mut().webview = Some(webview));
            ffi::set_output_edit_visible(false);
        }
    }
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

// ---- Windows: the wry hosting ----------------------------------------------

#[cfg(windows)]
mod webview_impl {
    use std::num::NonZeroIsize;

    use raw_window_handle::{
        HandleError, HasWindowHandle, RawWindowHandle, Win32WindowHandle, WindowHandle,
    };
    use wry::dpi::{PhysicalPosition, PhysicalSize};
    use wry::{Rect, WebView, WebViewBuilder};

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
.msg .who{font-weight:600;margin-right:6px;color:#569cd6;}
.msg.user{color:#9cdcfe;}
.msg.notice{color:#c9a227;font-style:italic;}
.msg.error{color:#f48771;white-space:pre-wrap;}
.msg.assistant p{margin:.4em 0;}
.msg.assistant pre{background:#111;padding:8px;border-radius:5px;overflow:auto;}
.msg.assistant code{background:#111;padding:0 3px;border-radius:3px;}
.msg.assistant a{color:#4ea1ff;}
details.tool{border:1px solid #3a3a3a;border-radius:6px;margin:8px 0;background:#232323;}
details.tool>summary{cursor:pointer;padding:5px 9px;list-style:none;color:#b5cea8;}
details.tool>summary::-webkit-details-marker{display:none;}
details.tool>summary::before{content:"\25b8  ";}
details.tool[open]>summary::before{content:"\25be  ";}
details.tool pre{margin:0;padding:8px;background:#151515;overflow:auto;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;}
.tres.err{color:#f48771;}
</style></head><body><div id="log"></div><script>
function sd(){window.scrollTo(0,document.body.scrollHeight);}
function addBlock(h){var l=document.getElementById('log');if(l){l.insertAdjacentHTML('beforeend',h);sd();}}
function startAssistant(){var o=document.getElementById('cur');if(o)o.removeAttribute('id');addBlock('<div class="msg assistant" id="cur"></div>');}
function updateAssistant(h){var c=document.getElementById('cur');if(c){c.innerHTML=h;sd();}}
function setToolResult(h){var l=document.querySelectorAll('#log details.tool');if(l.length){var t=l[l.length-1].querySelector('.tres');if(t){t.innerHTML=h;sd();}}}
</script></body></html>"#;

    pub fn create() -> Option<WebView> {
        let hwnd = ffi::get_hwnd() as isize;
        let host = Host(NonZeroIsize::new(hwnd)?);
        let (x, y, w, h) = ffi::output_bounds()?;
        WebViewBuilder::new()
            .with_bounds(bounds(x, y, w, h))
            .with_html(BASE_HTML)
            .with_transparent(false)
            .build_as_child(&host)
            .ok()
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
}
