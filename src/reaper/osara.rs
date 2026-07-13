//! OSARA screen-reader output with graceful degradation.
//!
//! OSARA registers `osara_outputMessage(const char*)` via the REAPER plugin
//! API; we resolve it with REAPER's `GetFunc`. Crucially, resolution is **lazy**:
//! REAPER loads extensions alphabetically, so `reaper_realackey` can load
//! before OSARA, at which point `GetFunc("osara_outputMessage")` is still null.
//! We resolve on demand (through the main-thread plug-in context) and cache the
//! function once OSARA appears. A persistently null result means OSARA isn't
//! installed — announcements become no-ops and the output pane is the fallback.

use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// `void osara_outputMessage(const char* message)`.
type OsaraOutputMessage = unsafe extern "C" fn(*const std::ffi::c_char);

static OSARA_FN: OnceLock<OsaraOutputMessage> = OnceLock::new();
/// Set once OSARA is detected. Readable from ANY thread (unlike `resolve`, which
/// needs the main-thread plug-in context) so the worker can tailor the system
/// prompt for a screen-reader user. OSARA never unloads mid-session, so once
/// true it stays true.
static OSARA_RUNNING: AtomicBool = AtomicBool::new(false);

fn resolve() -> Option<OsaraOutputMessage> {
    if let Some(f) = OSARA_FN.get() {
        return Some(*f);
    }
    // Resolve via the plug-in context reachable from the main-thread REAPER
    // handle. `with` returns None off the main thread / before the handle is set.
    let ptr: *mut c_void = crate::reaper::api::with(|reaper| unsafe {
        reaper
            .low()
            .plugin_context()
            .GetFunc(c"osara_outputMessage".as_ptr())
    })?;
    if ptr.is_null() {
        return None; // OSARA not (yet) present; try again next time.
    }
    // Pointer-to-fn transmute: same size, C ABI known exactly.
    let f: OsaraOutputMessage =
        unsafe { std::mem::transmute::<*mut c_void, OsaraOutputMessage>(ptr) };
    let _ = OSARA_FN.set(f);
    OSARA_RUNNING.store(true, Ordering::Relaxed);
    Some(f)
}

/// Whether OSARA has been detected this session — i.e. the user is a screen-reader
/// user. Safe to call from any thread. Populated on the main thread by
/// [`refresh_running`] (and by any [`announce`]).
pub fn is_running() -> bool {
    OSARA_RUNNING.load(Ordering::Relaxed)
}

/// Main-thread probe for OSARA; caches the result. Cheap and a no-op once found.
/// Called from the control-surface `run` loop so detection survives REAPER
/// loading OSARA after us (see the lazy-resolution note above).
pub fn refresh_running() {
    if OSARA_RUNNING.load(Ordering::Relaxed) {
        return;
    }
    let _ = resolve();
}

/// Speak a message (main thread only). No-op if OSARA is absent.
pub fn announce(msg: &str) {
    if let Some(f) = resolve() {
        if let Ok(c) = CString::new(msg) {
            unsafe { f(c.as_ptr()) };
        }
    }
}
