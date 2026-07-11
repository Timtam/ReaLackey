//! REAPER's native single-field input box (`GetUserInputs`), used for entering
//! the API key. Resolved as a raw function pointer at load so it can be called
//! from a static action callback without a REAPER session handle. Main thread
//! only (the call is modal).

use std::ffi::{c_char, c_void, CString};
use std::os::raw::c_int;
use std::sync::OnceLock;

use reaper_low::PluginContext;

type GetUserInputsFn = unsafe extern "C" fn(
    title: *const c_char,
    num_inputs: c_int,
    captions_csv: *const c_char,
    retvals_csv: *mut c_char,
    retvals_csv_sz: c_int,
) -> bool;

static GET_USER_INPUTS: OnceLock<GetUserInputsFn> = OnceLock::new();

/// Resolve `GetUserInputs` from the plugin context (call once at load).
pub fn init(context: &PluginContext) {
    let ptr = unsafe { context.GetFunc(c"GetUserInputs".as_ptr()) };
    if !ptr.is_null() {
        let f: GetUserInputsFn = unsafe { std::mem::transmute::<*mut c_void, GetUserInputsFn>(ptr) };
        let _ = GET_USER_INPUTS.set(f);
    }
}

/// Show a single-field input box and return the entered text (None on cancel or
/// if the API is unavailable). Main thread only.
pub fn get_user_input(title: &str, caption: &str) -> Option<String> {
    let f = *GET_USER_INPUTS.get()?;
    let title_c = CString::new(title).ok()?;
    let caption_c = CString::new(caption).ok()?;

    const CAP: usize = 4096;
    let mut buf = vec![0u8; CAP];
    let ok = unsafe {
        f(
            title_c.as_ptr(),
            1,
            caption_c.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            CAP as c_int,
        )
    };
    if !ok {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(CAP);
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}
