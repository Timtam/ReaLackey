//! Routes dialog callbacks (which fire on the main thread) to the worker.
//!
//! The design's C-ABI callbacks carry no user pointer, so the target sender
//! lives in a process-global `OnceLock`. `send` on a tokio unbounded channel is
//! sync and thread-safe, so calling it from the main thread is fine.

use std::sync::OnceLock;

use tokio::sync::mpsc::UnboundedSender;

use crate::ai::protocol::MainTask;

static TASK_TX: OnceLock<UnboundedSender<MainTask>> = OnceLock::new();

pub fn set_task_sender(tx: UnboundedSender<MainTask>) {
    let _ = TASK_TX.set(tx);
}

/// "Send" pressed.
pub fn submit(text: String) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return;
    }
    if let Some(tx) = TASK_TX.get() {
        let _ = tx.send(MainTask::Prompt(text));
    }
}

/// A message posted from the webview composer via `window.ipc.postMessage`.
/// Runs on the main thread (wry dispatches the IPC handler there). Shape:
/// `{"t":"submit","text":"…"}` or `{"t":"cancel"}`.
pub fn on_webview_message(json: &str) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return;
    };
    match v.get("t").and_then(|t| t.as_str()) {
        Some("submit") => {
            if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                submit(text.to_string());
            }
        }
        Some("cancel") => cancel(),
        // A link was clicked in the chat: open it in the user's default browser
        // instead of navigating the pane away from the conversation.
        Some("openurl") => {
            if let Some(url) = v.get("url").and_then(|u| u.as_str()) {
                open_url(url);
            }
        }
        // The composer's "Presets" button / Alt+P: show the native preset picker
        // and insert the chosen prompt into the composer.
        Some("presets:pick") => pick_preset(),
        _ => {}
    }
}

/// Show the native picker of saved prompt presets and, on a choice, insert its
/// body into the chat composer. Runs on the main thread (wry dispatches the IPC
/// handler there), where the native menu and the webview both live.
fn pick_preset() {
    use crate::ui::presets_ui::display_name;
    let presets = crate::prompts::registry::list();
    if presets.is_empty() {
        crate::ui::output::announce(
            "No presets saved yet. Add one from the Extensions menu, ReaLackey, Prompt presets.",
        );
        return;
    }
    // Non-empty, one-line labels so the popup's index maps 1:1 to `presets`
    // (ui_popup_menu drops empty lines).
    let labels: Vec<String> = presets.iter().map(|p| display_name(&p.name)).collect();
    let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    let choice = crate::ui::ffi::popup_menu(&refs);
    if choice == 0 || choice > presets.len() {
        return; // cancelled
    }
    let preset = &presets[choice - 1];
    crate::ui::output::insert_preset(&preset.body);
    // Announce the name through both accessible channels (matching the project's
    // dual-channel convention): OSARA speaks directly (focus-independent, Windows)
    // and the webview aria-live region carries it for VoiceOver on macOS.
    let msg = format!("Preset inserted: {}.", display_name(&preset.name));
    crate::reaper::osara::announce(&msg);
    crate::ui::output::announce(&msg);
}

/// Open an http(s)/mailto URL (a link the user clicked in the chat) in the
/// default browser. Model-generated links are restricted to these schemes — a
/// local `file://` or custom scheme is never launched.
pub(crate) fn open_url(url: &str) {
    if is_launchable_url(url) {
        open_external(url.trim());
    }
}

/// Whether a clicked URL may be launched externally. Only web and mail schemes —
/// leading whitespace is trimmed first so `" javascript:…"` can't sneak through.
fn is_launchable_url(url: &str) -> bool {
    let l = url.trim().to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://") || l.starts_with("mailto:")
}

#[cfg(windows)]
fn open_external(url: &str) {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "shell32")]
    extern "system" {
        fn ShellExecuteW(
            hwnd: isize,
            op: *const u16,
            file: *const u16,
            params: *const u16,
            dir: *const u16,
            show: i32,
        ) -> isize;
    }
    let wide = |s: &str| -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    };
    let op = wide("open");
    let file = wide(url);
    // SW_SHOWNORMAL = 1; null hwnd. Best-effort — the returned HINSTANCE is ignored.
    unsafe {
        ShellExecuteW(
            0,
            op.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
        );
    }
}

#[cfg(target_os = "macos")]
fn open_external(url: &str) {
    // Hand the URL to LaunchServices via `open`, which routes it to the user's
    // default browser / mail client. Best-effort: a spawn failure is ignored (the
    // chat pane has no surface for this error and this must never panic — it runs
    // from a webview IPC callback).
    let _ = std::process::Command::new("open").arg(url).spawn();
}

#[cfg(all(not(windows), not(target_os = "macos")))]
fn open_external(_url: &str) {}

/// "Confirm" pressed (Phase 3 mutations). No-op in Phase 0.
pub fn confirm(_confirm_id: i32) {}

/// Dialog closed / "Stopp": abort any in-flight generation and disarm pixel
/// control (a physical kill switch for the Tier-B click/drag capability).
pub fn cancel() {
    crate::tools::disarm_pixel_control();
    if let Some(tx) = TASK_TX.get() {
        let _ = tx.send(MainTask::Cancel);
    }
}

#[cfg(test)]
mod tests {
    use super::is_launchable_url;

    #[test]
    fn only_web_and_mail_schemes_launch() {
        for ok in ["https://example.com", "http://x", "HTTPS://X", "mailto:a@b.com"] {
            assert!(is_launchable_url(ok), "{ok} should launch");
        }
        for bad in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            " javascript:alert(1)",
            "data:text/html,x",
            "ftp://x",
            "vbscript:x",
            "",
        ] {
            assert!(!is_launchable_url(bad), "{bad} must NOT launch");
        }
    }
}
