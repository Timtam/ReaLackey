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
    fn ui_set_provider_cbs(
        on_list: extern "C" fn(*mut c_char, c_int),
        on_action: extern "C" fn(c_int, c_int) -> c_int,
    );
    fn ui_show_providers();
    fn ui_popup_menu(items_newline: *const c_char) -> c_int;
    fn ui_message_box(title: *const c_char, text: *const c_char, flags: c_int) -> c_int;
    fn ui_set_provider_edit_cbs(
        on_init: extern "C" fn(),
        on_fetch: extern "C" fn(),
        on_ok: extern "C" fn() -> c_int,
        on_key: extern "C" fn(c_int),
    );
    fn ui_show_provider_edit() -> c_int;
    fn ui_pe_set_text(ctrl: c_int, utf8: *const c_char);
    fn ui_pe_get_text(ctrl: c_int, buf: *mut c_char, buf_sz: c_int);
    fn ui_pe_set_check(ctrl: c_int, checked: c_int);
    fn ui_pe_get_check(ctrl: c_int) -> c_int;
    fn ui_pe_show(ctrl: c_int, visible: c_int);
    fn ui_pe_set_list(ctrl: c_int, items_newline: *const c_char);
    fn ui_pe_get_sel(ctrl: c_int) -> c_int;
    fn ui_pe_set_sel(ctrl: c_int, index: c_int);
    fn ui_set_preset_cbs(
        on_list: extern "C" fn(*mut c_char, c_int),
        on_action: extern "C" fn(c_int, c_int) -> c_int,
    );
    fn ui_show_presets();
    fn ui_set_preset_edit_cbs(on_init: extern "C" fn(), on_ok: extern "C" fn() -> c_int);
    fn ui_show_preset_edit() -> c_int;
    fn ui_preset_set_text(ctrl: c_int, utf8: *const c_char);
    fn ui_preset_get_text(ctrl: c_int, buf: *mut c_char, buf_sz: c_int);
    fn ui_find_window_by_title(needle: *const c_char) -> *mut c_void;
}

// Provider settings dialog control ids — MUST match cpp/resource.h.
pub const PE_LABEL: c_int = 1021;
pub const PE_BASEURL: c_int = 1022;
pub const PE_BASEURL_LBL: c_int = 1023;
pub const PE_MODEL: c_int = 1024;
pub const PE_MAXTOK: c_int = 1026;
pub const PE_VISION: c_int = 1027;
pub const PE_AUDIO: c_int = 1036;
pub const PE_THINKING: c_int = 1037;
pub const PE_KEY: c_int = 1028;
pub const PE_KEYHINT: c_int = 1029;
pub const PE_MAXTURNS: c_int = 1030;
pub const PE_KEYLIST: c_int = 1031;

// Prompt-preset edit dialog control ids — MUST match cpp/resource.h.
pub const PRE_NAME: c_int = 1051;
pub const PRE_BODY: c_int = 1052;

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

/// Find the first visible top-level window whose title contains `needle`
/// (case-insensitive), as an `isize` HWND, or `None`. Used to locate REAPER's
/// floating Video window for capture (REAPER has no API for its handle).
pub fn find_window_by_title(needle: &str) -> Option<isize> {
    let c = to_cstring(needle)?;
    let h = unsafe { ui_find_window_by_title(c.as_ptr()) } as isize;
    (h != 0).then_some(h)
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

/// Register the provider-dialog callbacks (list + row actions). Call once at init.
pub fn install_provider_cbs() {
    unsafe { ui_set_provider_cbs(prov_list, prov_action) }
}

/// Register the provider *settings* dialog callbacks. Call once at init.
pub fn install_provider_edit_cbs() {
    unsafe { ui_set_provider_edit_cbs(pe_init, pe_fetch, pe_ok, pe_key) }
}

/// Show the modal provider settings dialog; true if the user pressed OK.
pub fn show_provider_edit() -> bool {
    unsafe { ui_show_provider_edit() != 0 }
}

/// Set a settings-dialog text control (valid only from the dialog callbacks).
pub fn pe_set_text(ctrl: c_int, text: &str) {
    if let Some(c) = to_cstring(text) {
        unsafe { ui_pe_set_text(ctrl, c.as_ptr()) }
    }
}

/// Read a settings-dialog text control (valid only from the dialog callbacks).
pub fn pe_get_text(ctrl: c_int) -> String {
    const CAP: usize = 8192;
    let mut buf = vec![0u8; CAP];
    unsafe { ui_pe_get_text(ctrl, buf.as_mut_ptr() as *mut c_char, CAP as c_int) };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(CAP);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Set a settings-dialog checkbox.
pub fn pe_set_check(ctrl: c_int, checked: bool) {
    unsafe { ui_pe_set_check(ctrl, checked as c_int) }
}

/// Read a settings-dialog checkbox.
pub fn pe_get_check(ctrl: c_int) -> bool {
    unsafe { ui_pe_get_check(ctrl) != 0 }
}

/// Show/hide a settings-dialog control (used to hide base-URL/vision for Anthropic).
pub fn pe_show(ctrl: c_int, visible: bool) {
    unsafe { ui_pe_show(ctrl, visible as c_int) }
}

/// Fill a settings-dialog listbox with `items` (already display-masked), in order.
pub fn pe_set_list(ctrl: c_int, items: &[String]) {
    if let Some(c) = to_cstring(&items.join("\n")) {
        unsafe { ui_pe_set_list(ctrl, c.as_ptr()) }
    }
}

/// The selected row of a settings-dialog listbox, or `None` if nothing is selected.
pub fn pe_get_sel(ctrl: c_int) -> Option<usize> {
    let sel = unsafe { ui_pe_get_sel(ctrl) };
    (sel >= 0).then_some(sel as usize)
}

/// Select row `index` in a settings-dialog listbox.
pub fn pe_set_sel(ctrl: c_int, index: usize) {
    unsafe { ui_pe_set_sel(ctrl, index as c_int) }
}

extern "C" fn pe_init() {
    let _ = std::panic::catch_unwind(crate::ui::providers_ui::edit_dialog_init);
}

extern "C" fn pe_fetch() {
    let _ = std::panic::catch_unwind(crate::ui::providers_ui::edit_dialog_fetch);
}

extern "C" fn pe_ok() -> c_int {
    // On panic, close the dialog (1) rather than leaving it stuck open.
    std::panic::catch_unwind(|| crate::ui::providers_ui::edit_dialog_ok() as c_int).unwrap_or(1)
}

extern "C" fn pe_key(action: c_int) {
    let _ = std::panic::catch_unwind(|| crate::ui::providers_ui::edit_dialog_key(action));
}

/// Register the prompt-preset list/action callbacks. Call once at init.
pub fn install_preset_cbs() {
    unsafe { ui_set_preset_cbs(preset_list, preset_action) }
}

/// Register the prompt-preset edit sub-dialog callbacks. Call once at init.
pub fn install_preset_edit_cbs() {
    unsafe { ui_set_preset_edit_cbs(preset_init, preset_ok) }
}

/// Show the modal prompt-preset management dialog (main thread only).
pub fn show_presets() {
    unsafe { ui_show_presets() }
}

/// Show the modal preset edit sub-dialog; true if the user pressed OK.
pub fn show_preset_edit() -> bool {
    unsafe { ui_show_preset_edit() != 0 }
}

/// Set a preset-edit text control (valid only from the edit-dialog callbacks).
pub fn preset_set_text(ctrl: c_int, text: &str) {
    if let Some(c) = to_cstring(text) {
        unsafe { ui_preset_set_text(ctrl, c.as_ptr()) }
    }
}

/// Read a preset-edit text control (valid only from the edit-dialog callbacks).
/// Uses a generous buffer since the body field holds a whole prompt.
pub fn preset_get_text(ctrl: c_int) -> String {
    const CAP: usize = 65536;
    let mut buf = vec![0u8; CAP];
    unsafe { ui_preset_get_text(ctrl, buf.as_mut_ptr() as *mut c_char, CAP as c_int) };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(CAP);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

extern "C" fn preset_list(buf: *mut c_char, buf_sz: c_int) {
    let _ = std::panic::catch_unwind(|| {
        let text = crate::ui::presets_ui::list_text();
        unsafe { write_cstr(buf, buf_sz, &text) };
    });
}

extern "C" fn preset_action(action: c_int, index: c_int) -> c_int {
    std::panic::catch_unwind(|| crate::ui::presets_ui::on_action(action, index) as c_int)
        .unwrap_or(0)
}

extern "C" fn preset_init() {
    let _ = std::panic::catch_unwind(crate::ui::presets_ui::edit_dialog_init);
}

extern "C" fn preset_ok() -> c_int {
    // On panic, close the dialog (1) rather than leaving it stuck open.
    std::panic::catch_unwind(|| crate::ui::presets_ui::edit_dialog_ok() as c_int).unwrap_or(1)
}

/// Show the modal provider-management dialog (main thread only).
pub fn show_providers() {
    unsafe { ui_show_providers() }
}

/// Show a modal popup menu of `items` at the cursor; returns the 1-based index of
/// the chosen item, or 0 if cancelled. Main thread only.
pub fn popup_menu(items: &[&str]) -> usize {
    let Some(c) = to_cstring(&items.join("\n")) else {
        return 0;
    };
    let r = unsafe { ui_popup_menu(c.as_ptr()) };
    if r > 0 {
        r as usize
    } else {
        0
    }
}

/// Show a modal message box. `question` = Yes/No (returns true on Yes); otherwise
/// an OK box (returns true on OK). Main thread only.
pub fn message_box(title: &str, text: &str, question: bool) -> bool {
    match (to_cstring(title), to_cstring(text)) {
        (Some(t), Some(m)) => {
            unsafe { ui_message_box(t.as_ptr(), m.as_ptr(), question as c_int) != 0 }
        }
        _ => false,
    }
}

/// Fill `buf` (capacity `buf_sz`, incl. NUL) with `s`, truncated on a char
/// boundary. No-op on a null/zero buffer.
unsafe fn write_cstr(buf: *mut c_char, buf_sz: c_int, s: &str) {
    if buf.is_null() || buf_sz <= 0 {
        return;
    }
    let cap = buf_sz as usize;
    let mut end = s.len().min(cap - 1);
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let bytes = &s.as_bytes()[..end];
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
    *buf.add(bytes.len()) = 0;
}

extern "C" fn prov_list(buf: *mut c_char, buf_sz: c_int) {
    let _ = std::panic::catch_unwind(|| {
        let text = crate::ui::providers_ui::list_text();
        unsafe { write_cstr(buf, buf_sz, &text) };
    });
}

extern "C" fn prov_action(action: c_int, index: c_int) -> c_int {
    std::panic::catch_unwind(|| crate::ui::providers_ui::on_action(action, index) as c_int)
        .unwrap_or(0)
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

// ---- macOS-only window helpers (used by the capture/input backends) ----------
// Cross-platform in the shim (Win32/SWELL), but only the macOS Rust backends need
// them (Windows uses the `windows` crate directly), so they're gated to avoid
// dead-code warnings on Windows.
#[cfg(target_os = "macos")]
extern "C" {
    fn ui_window_rect(
        hwnd: *mut c_void,
        x: *mut c_int,
        y: *mut c_int,
        w: *mut c_int,
        h: *mut c_int,
    ) -> c_int;
    fn ui_window_to_front(hwnd: *mut c_void);
}

/// Window rect (screen-space x, y, w, h) via the host toolkit. macOS only.
#[cfg(target_os = "macos")]
pub fn window_rect(hwnd: isize) -> Option<(i32, i32, i32, i32)> {
    let (mut x, mut y, mut w, mut h) = (0, 0, 0, 0);
    let ok = unsafe { ui_window_rect(hwnd as *mut c_void, &mut x, &mut y, &mut w, &mut h) };
    (ok != 0).then_some((x, y, w, h))
}

/// Un-minimize + raise a window. macOS only.
#[cfg(target_os = "macos")]
pub fn window_to_front(hwnd: isize) {
    unsafe { ui_window_to_front(hwnd as *mut c_void) }
}
