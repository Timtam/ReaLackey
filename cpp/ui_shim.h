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
