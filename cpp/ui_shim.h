// ui_shim.h — the narrow, stable C-ABI the Rust core drives.
//
// Design: Rust owns all logic/concurrency/networking; C++ owns only the GUI
// surface (a modeless native dialog). Strings are UTF-8, NUL-terminated, and
// borrowed for the duration of the call only. ALL ui_* calls other than
// ui_init/ui_set_callbacks MUST happen on REAPER's main thread.
#ifndef RAAI_UI_SHIM_H
#define RAAI_UI_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Callbacks Rust registers. They fire on the main thread (dialog message loop).
typedef void (*on_submit_cb)(const char* utf8_text);  // "Send" pressed
typedef void (*on_confirm_cb)(int confirm_id);        // "Confirm" (Phase 3)
typedef void (*on_cancel_cb)(void);                   // dialog closed / "Stopp"
typedef void (*on_resize_cb)(void);                   // dialog resized (reflow webview)
typedef void (*on_destroy_cb)(void);                  // dialog HWND destroyed (drop webview)

// One-time init. On non-Windows, `reaper_get_func` (REAPER's rec->GetFunc) is
// forwarded to SWELL so it can resolve the host's SWELL API. On Windows,
// `reaper_get_func` is ignored and `hinst` is the extension module handle used
// to load the dialog resource.
void ui_init(void* hinst, void* reaper_get_func);

void ui_set_callbacks(on_submit_cb on_submit,
                      on_confirm_cb on_confirm,
                      on_cancel_cb on_cancel);

// Create/show the modeless dialog. `parent_hwnd` = REAPER main window.
void ui_show(void* parent_hwnd);

void ui_append_output(const char* utf8);  // append to the read-only log
void ui_set_status(const char* utf8);      // set the status field
void ui_close(void);                       // destroy the dialog

// --- webview hosting (Rust hosts an embedded WebView2 in the output area) -----
// The dialog's native HWND (as a void*), or NULL if not created yet.
void* ui_get_hwnd(void);
// The output area's rect in dialog client pixels. Returns 1 on success.
int ui_output_bounds(int* x, int* y, int* w, int* h);
// Show/hide the plain output edit (hidden when the webview takes over).
void ui_set_output_edit_visible(int visible);
// Register a callback fired (main thread) whenever the dialog is resized.
void ui_set_resize_cb(on_resize_cb on_resize);
// Register a callback fired (main thread) when the dialog HWND is destroyed, so
// Rust can drop the embedded webview before its parent window goes away.
void ui_set_destroy_cb(on_destroy_cb on_destroy);

// Append a menu item wired to a REAPER command id. Used to add an entry to
// REAPER's Extensions menu from a hookcustommenu callback (main thread only).
void ui_add_menu_item(void* hmenu, const char* label, int command_id);

// Create a popup submenu (returns its HMENU), and attach a submenu to a parent
// menu under a title. Used to group the extension's entries under one submenu.
void* ui_create_submenu(void);
void ui_attach_submenu(void* parent_hmenu, void* submenu, const char* title);

#ifdef __cplusplus
}
#endif

#endif // RAAI_UI_SHIM_H
