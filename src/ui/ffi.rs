//! FFI to the C++/SWELL shim (cpp/ui_shim.cpp), statically linked by build.rs.
//!
//! Every `ui_*` call other than `init`/`install_callbacks` must happen on the
//! REAPER main thread (the shim touches native window handles). The `#[no_mangle]`-
//! free callback thunks are panic-guarded so a Rust panic never unwinds across
//! the C-ABI boundary (design N3).

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

extern "C" {
    fn ui_init(hinst: *mut c_void, reaper_get_func: *mut c_void);
    fn ui_set_callbacks(
        on_submit: extern "C" fn(*const c_char),
        on_confirm: extern "C" fn(c_int),
        on_cancel: extern "C" fn(),
    );
    fn ui_show(parent_hwnd: *mut c_void);
    fn ui_append_output(utf8: *const c_char);
    fn ui_set_status(utf8: *const c_char);
    fn ui_close();
    fn ui_add_menu_item(hmenu: *mut c_void, label: *const c_char, command_id: c_int);
    fn ui_create_submenu() -> *mut c_void;
    fn ui_attach_submenu(parent_hmenu: *mut c_void, submenu: *mut c_void, title: *const c_char);
}

/// One-time init. `get_func` is REAPER's `rec->GetFunc` (used by SWELL on
/// non-Windows; ignored on Windows).
pub fn init(hinst: *mut c_void, get_func: *mut c_void) {
    unsafe { ui_init(hinst, get_func) }
}

pub fn install_callbacks() {
    unsafe { ui_set_callbacks(on_submit, on_confirm, on_cancel) }
}

pub fn show(parent_hwnd: *mut c_void) {
    unsafe { ui_show(parent_hwnd) }
}

pub fn append_output(text: &str) {
    if let Some(c) = to_cstring(text) {
        unsafe { ui_append_output(c.as_ptr()) }
    }
}

pub fn set_status(text: &str) {
    if let Some(c) = to_cstring(text) {
        unsafe { ui_set_status(c.as_ptr()) }
    }
}

#[allow(dead_code)]
pub fn close() {
    unsafe { ui_close() }
}

/// Append a menu item (bound to `command_id`) to a native `HMENU`.
pub fn add_menu_item(hmenu: *mut c_void, label: &str, command_id: c_int) {
    if let Some(c) = to_cstring(label) {
        unsafe { ui_add_menu_item(hmenu, c.as_ptr(), command_id) }
    }
}

/// Create an empty popup submenu (returns its `HMENU`).
pub fn create_submenu() -> *mut c_void {
    unsafe { ui_create_submenu() }
}

/// Attach `submenu` to `parent_hmenu` under `title`.
pub fn attach_submenu(parent_hmenu: *mut c_void, submenu: *mut c_void, title: &str) {
    if let Some(c) = to_cstring(title) {
        unsafe { ui_attach_submenu(parent_hmenu, submenu, c.as_ptr()) }
    }
}

fn to_cstring(s: &str) -> Option<CString> {
    // Interior NULs would truncate; replace them defensively.
    CString::new(s.replace('\0', " ")).ok()
}

// ---- callback thunks (fire on the main thread; never unwind across FFI) ------

extern "C" fn on_submit(text: *const c_char) {
    let _ = std::panic::catch_unwind(|| {
        let s = unsafe { cstr_to_string(text) };
        crate::ui::bridge::submit(s);
    });
}

extern "C" fn on_confirm(confirm_id: c_int) {
    let _ = std::panic::catch_unwind(|| {
        crate::ui::bridge::confirm(confirm_id);
    });
}

extern "C" fn on_cancel() {
    let _ = std::panic::catch_unwind(|| {
        crate::ui::bridge::cancel();
    });
}

unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}
