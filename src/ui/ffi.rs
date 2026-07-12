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
    fn ui_get_hwnd() -> *mut c_void;
    fn ui_output_bounds(x: *mut c_int, y: *mut c_int, w: *mut c_int, h: *mut c_int) -> c_int;
    fn ui_set_webview_active(active: c_int);
    fn ui_translate_accel(msg: *mut c_void) -> c_int;
    fn ui_set_resize_cb(on_resize: extern "C" fn());
    fn ui_set_destroy_cb(on_destroy: extern "C" fn());
    fn ui_enable_webview_tabstop();
    fn ui_set_webview_focus_cb(on_focus: extern "C" fn());
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

/// Destroy the dialog window. Currently unused (the window persists for the
/// process; see the webview `ManuallyDrop` note in `output.rs`).
#[allow(dead_code)]
pub fn close() {
    unsafe { ui_close() }
}

/// The dialog's native window handle (null if the dialog isn't created yet).
pub fn get_hwnd() -> *mut c_void {
    unsafe { ui_get_hwnd() }
}

/// The output area's rect in dialog client pixels, if the dialog exists.
pub fn output_bounds() -> Option<(i32, i32, i32, i32)> {
    let (mut x, mut y, mut w, mut h) = (0, 0, 0, 0);
    let ok = unsafe { ui_output_bounds(&mut x, &mut y, &mut w, &mut h) };
    (ok != 0).then_some((x, y, w, h))
}

/// Hand the whole window to the webview (hide all native controls, fill client).
pub fn set_webview_active(active: bool) {
    unsafe { ui_set_webview_active(active as c_int) }
}

/// Route a keystroke aimed at our window (via REAPER's accelerator queue).
/// `msg` points to a Win32/SWELL `MSG`. Returns 0 (not ours) / 1 (eat) / -1 (pass on).
pub fn translate_accel(msg: *mut c_void) -> i32 {
    unsafe { ui_translate_accel(msg) }
}

/// Register the resize thunk so the webview re-bounds with the dialog.
pub fn install_resize_cb() {
    unsafe { ui_set_resize_cb(on_resize) }
}

/// Register the destroy thunk so the webview is dropped when the dialog is.
pub fn install_destroy_cb() {
    unsafe { ui_set_destroy_cb(on_destroy) }
}

/// Make the webview host window a keyboard tab-stop (call after creation).
pub fn enable_webview_tabstop() {
    unsafe { ui_enable_webview_tabstop() }
}

/// Register the "webview host focused" thunk (forwards focus into the content).
pub fn install_webview_focus_cb() {
    unsafe { ui_set_webview_focus_cb(on_webview_focus) }
}

extern "C" fn on_webview_focus() {
    let _ = std::panic::catch_unwind(crate::ui::output::on_webview_focus);
}

extern "C" fn on_resize() {
    let _ = std::panic::catch_unwind(crate::ui::output::on_resize);
}

extern "C" fn on_destroy() {
    let _ = std::panic::catch_unwind(crate::ui::output::on_destroy);
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
