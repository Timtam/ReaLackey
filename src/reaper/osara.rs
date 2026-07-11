//! OSARA screen-reader output with graceful degradation.
//!
//! OSARA registers `osara_outputMessage(const char*)` via the REAPER plugin
//! API; we resolve it with `GetFunc("osara_outputMessage")` (no `API_` prefix).
//! A null pointer means OSARA isn't installed — announcements become no-ops and
//! the native output field remains the accessible fallback.

use std::ffi::{c_char, c_void, CString};

/// `void osara_outputMessage(const char* message)`.
type OsaraOutputMessage = unsafe extern "C" fn(*const c_char);

#[derive(Debug, Clone, Copy)]
pub struct Osara {
    func: Option<OsaraOutputMessage>,
}

impl Osara {
    /// Wrap the raw pointer from `GetFunc("osara_outputMessage")`.
    pub fn from_ptr(ptr: *mut c_void) -> Self {
        let func = if ptr.is_null() {
            None
        } else {
            // Pointer-to-fn transmute: same size, C ABI known exactly.
            Some(unsafe { std::mem::transmute::<*mut c_void, OsaraOutputMessage>(ptr) })
        };
        Osara { func }
    }

    pub fn is_available(&self) -> bool {
        self.func.is_some()
    }

    /// Speak a message (main thread only). No-op if OSARA is absent.
    pub fn announce(&self, msg: &str) {
        if let Some(f) = self.func {
            if let Ok(c) = CString::new(msg) {
                unsafe { f(c.as_ptr()) };
            }
        }
    }
}
