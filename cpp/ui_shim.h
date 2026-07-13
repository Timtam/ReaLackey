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
typedef void (*on_webview_focus_cb)(void);            // webview host gained focus (enter content)

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
// Hand the whole window to the webview: hide all native controls and stretch
// the (webview-tracked) output rect to fill the client area.
void ui_set_webview_active(int active);
// REAPER accelerator hook (registered via plugin_register "accelerator"). Routes
// keystrokes aimed at our window to it (so REAPER doesn't eat them as shortcuts)
// and lets the dialog manager handle Tab/Esc for native controls. `msg` is a
// pointer to a Win32/SWELL MSG. Returns 0 (not ours) / 1 (eaten) / -1 (pass on).
int ui_translate_accel(void* msg);
// Register a callback fired (main thread) whenever the dialog is resized.
void ui_set_resize_cb(on_resize_cb on_resize);
// Register a callback fired (main thread) when the dialog HWND is destroyed, so
// Rust can drop the embedded webview before its parent window goes away.
void ui_set_destroy_cb(on_destroy_cb on_destroy);
// Give the embedded webview's host window WS_TABSTOP (so Tab reaches it) and
// subclass it so a WM_SETFOCUS fires `on_webview_focus` — letting Rust move
// focus into the web content on the first Tab (no silent stop).
void ui_enable_webview_tabstop(void);
// Register the "webview host focused" callback.
void ui_set_webview_focus_cb(on_webview_focus_cb on_focus);
// Move keyboard focus to the native control after (forward=1) or before
// (forward=0) the webview — used when the user Tabs out of the web content.
void ui_focus_after_webview(int forward);

// Window geometry + raise via the host toolkit (Win32 on Windows, SWELL on
// macOS/Linux) — cross-platform helpers the macOS capture/input backends use to
// stay off platform-specific Rust APIs. `ui_window_rect` fills screen-space
// x/y/w/h and returns 1 on success. Main thread only.
int ui_window_rect(void* hwnd, int* x, int* y, int* w, int* h);
void ui_window_to_front(void* hwnd);

// Append a menu item wired to a REAPER command id. Used to add an entry to
// REAPER's Extensions menu from a hookcustommenu callback (main thread only).
void ui_add_menu_item(void* hmenu, const char* label, int command_id);

// Create a popup submenu (returns its HMENU), and attach a submenu to a parent
// menu under a title. Used to group the extension's entries under one submenu.
void* ui_create_submenu(void);
void ui_attach_submenu(void* parent_hmenu, void* submenu, const char* title);

// --- provider management dialog (Phase 5, M4) --------------------------------
// Rust fills `buf` (UTF-8, NUL-terminated, capacity `buf_sz`) with the provider
// labels, one per line, in list order; the default provider is marked ("* ").
typedef void (*prov_list_cb)(char* buf, int buf_sz);
// Rust performs an action on the selected row: action 0=add, 1=edit, 2=delete,
// 3=set-default; `index` is the selected row (or -1 if none). Returns 1 if the
// list changed (the dialog then repopulates the listbox), 0 otherwise. Rust may
// open nested modal dialogs (e.g. ui_popup_menu) inside this call.
typedef int (*prov_action_cb)(int action, int index);
void ui_set_provider_cbs(prov_list_cb on_list, prov_action_cb on_action);
// Show the modal provider-management dialog. Main thread only.
void ui_show_providers(void);
// Show a modal popup menu of newline-separated items at the cursor and return
// the 1-based index of the chosen item, or 0 if cancelled. Used for the "Add"
// preset picker. Main thread only.
int ui_popup_menu(const char* items_newline);
// Show a modal message box. flags: 0 = info/OK box, 1 = Yes/No question. Returns
// 1 for OK/Yes, 0 for No/Cancel. Main thread only.
int ui_message_box(const char* title, const char* text, int flags);

// --- provider settings dialog (Phase 5, M5) ----------------------------------
// A real dialog (add / edit one account) with a Model field next to a "Fetch
// models" button. Rust drives it through these callbacks + control accessors.
typedef void (*pe_init_cb)(void);   // WM_INITDIALOG: Rust prefills the fields
typedef void (*pe_fetch_cb)(void);  // "Fetch models" clicked: Rust fetches+picks
typedef int (*pe_ok_cb)(void);      // OK clicked: Rust saves; returns 1=close, 0=keep open
// A key-list button was pressed: action 0=add, 1=delete, 2=move up, 3=move down.
// Rust mutates its working list and repopulates the listbox via ui_pe_set_list.
typedef void (*pe_key_cb)(int action);
void ui_set_provider_edit_cbs(pe_init_cb on_init, pe_fetch_cb on_fetch, pe_ok_cb on_ok,
                              pe_key_cb on_key);
// Show the modal settings dialog; returns 1 if the user pressed OK, 0 on cancel.
int ui_show_provider_edit(void);
// Field accessors, valid only from the callbacks while the dialog is open. `ctrl`
// is an ID_PE_* control id.
void ui_pe_set_text(int ctrl, const char* utf8);
void ui_pe_get_text(int ctrl, char* buf, int buf_sz);
void ui_pe_set_check(int ctrl, int checked);
int ui_pe_get_check(int ctrl);
void ui_pe_show(int ctrl, int visible);
// Listbox accessors for the key list (`ctrl` = ID_PE_KEYLIST). `items` is a
// newline-separated, already-masked display list in priority order.
void ui_pe_set_list(int ctrl, const char* items_newline);
int ui_pe_get_sel(int ctrl);          // selected row, or -1 if none
void ui_pe_set_sel(int ctrl, int index);

#ifdef __cplusplus
}
#endif

#endif // RAAI_UI_SHIM_H
