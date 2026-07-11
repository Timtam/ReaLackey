// ui_shim.cpp — modeless native dialog + C-ABI implementation.
//
// The SAME source builds on all three platforms: on Windows against the real
// Win32 API; on macOS/Linux against SWELL (swell.h stands in for windows.h and
// translates to Cocoa/GTK). REAPER already provides the SWELL implementation —
// this shim only calls it (see build.rs: it compiles the SWELL *modstub* on
// non-Windows so those calls resolve against the host).
//
// Text is UTF-8 across the C-ABI. On Windows we convert to UTF-16 and use the
// wide (-W) Win32 calls so umlauts render correctly; SWELL is natively UTF-8
// so it uses the -A calls directly.

#ifdef _WIN32
  #include <windows.h>
  #include <string>
  #define RAAI_DLGRET INT_PTR CALLBACK
#else
  #include "swell.h"
  #include <string>
  #define RAAI_DLGRET WDL_DLGRET
#endif

#include "resource.h"
#include "ui_shim.h"

// SWELL_dllMain lives in the separately-compiled modstub (build.rs, non-Windows,
// -DSWELL_PROVIDED_BY_APP). Declaring it lets ui_init hand REAPER's GetFunc to
// SWELL before any dialog call.
#ifndef _WIN32
extern "C" int SWELL_dllMain(HINSTANCE, DWORD, LPVOID);
#endif

static HINSTANCE     g_hinst = NULL;
static HWND          g_dlg   = NULL;
static on_submit_cb  g_on_submit  = NULL;
static on_confirm_cb g_on_confirm = NULL;
static on_cancel_cb  g_on_cancel  = NULL;

// ---- text helpers (UTF-8 <-> platform) --------------------------------------
#ifdef _WIN32
static std::wstring to_wide(const char* s) {
  if (!s) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  if (n <= 0) return std::wstring();
  std::wstring w((size_t)(n - 1), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
  return w;
}
static std::string to_utf8(const wchar_t* w) {
  if (!w) return std::string();
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
  if (n <= 0) return std::string();
  std::string s((size_t)(n - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, NULL, NULL);
  return s;
}
#endif

static void set_ctrl_text(HWND hwnd, int id, const char* utf8) {
  if (!hwnd || !utf8) return;
#ifdef _WIN32
  SetDlgItemTextW(hwnd, id, to_wide(utf8).c_str());
#else
  SetDlgItemTextA(hwnd, id, utf8);
#endif
}

static void append_ctrl_text(HWND edit, const char* utf8) {
  if (!edit || !utf8) return;
#ifdef _WIN32
  int len = GetWindowTextLengthW(edit);
  SendMessageW(edit, EM_SETSEL, (WPARAM)len, (LPARAM)len);
  SendMessageW(edit, EM_REPLACESEL, FALSE, (LPARAM)to_wide(utf8).c_str());
  SendMessageW(edit, EM_SCROLLCARET, 0, 0);
#else
  int len = GetWindowTextLength(edit);
  SendMessage(edit, EM_SETSEL, (WPARAM)len, (LPARAM)len);
  SendMessage(edit, EM_REPLACESEL, FALSE, (LPARAM)utf8);
  SendMessage(edit, EM_SCROLLCARET, 0, 0);
#endif
}

static std::string get_ctrl_text(HWND hwnd, int id) {
#ifdef _WIN32
  wchar_t buf[8192];
  buf[0] = 0;
  GetDlgItemTextW(hwnd, id, buf, (int)(sizeof(buf) / sizeof(buf[0])));
  return to_utf8(buf);
#else
  char buf[8192];
  buf[0] = 0;
  GetDlgItemTextA(hwnd, id, buf, (int)sizeof(buf));
  return std::string(buf);
#endif
}

// ---- dialog procedure -------------------------------------------------------
static RAAI_DLGRET DialogProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_INITDIALOG: {
      g_dlg = hwnd;
      // Raise the output edit's text limit so a long conversation isn't clipped.
      HWND out = GetDlgItem(hwnd, ID_OUTPUT_EDIT);
      if (out) SendMessage(out, EM_SETLIMITTEXT, (WPARAM)0, 0);
      set_ctrl_text(hwnd, ID_STATUS_TEXT, "Ready.");
      return TRUE;  // let the dialog manager set default focus
    }

    case WM_COMMAND:
      switch (LOWORD(wParam)) {
        case ID_SUBMIT_BTN: {
          std::string text = get_ctrl_text(hwnd, ID_INPUT_EDIT);
          set_ctrl_text(hwnd, ID_INPUT_EDIT, "");
          if (g_on_submit) g_on_submit(text.c_str());
          return TRUE;
        }
        case ID_CONFIRM_BTN:
          if (g_on_confirm) g_on_confirm(0);
          return TRUE;
        case IDCANCEL:            // "Close" / Esc: HIDE, don't destroy, so the
          ShowWindow(hwnd, SW_HIDE);   // conversation history survives and the
          return TRUE;                 // action can re-show it (side-by-side use).
      }
      return FALSE;

    case WM_CLOSE:                 // window [x]: hide, keep the window + history.
      ShowWindow(hwnd, SW_HIDE);
      return TRUE;

    case WM_DESTROY:
      if (g_on_cancel) g_on_cancel();
      g_dlg = NULL;
      return TRUE;
  }
  return FALSE;  // not handled: default processing
}

// ---- C-ABI ------------------------------------------------------------------
extern "C" void ui_init(void* hinst, void* reaper_get_func) {
  g_hinst = (HINSTANCE)hinst;
#ifndef _WIN32
  // Wire REAPER's SWELL into this module BEFORE any dialog call.
  SWELL_dllMain(g_hinst, DLL_PROCESS_ATTACH, (LPVOID)reaper_get_func);
#else
  (void)reaper_get_func;
#endif
}

extern "C" void ui_set_callbacks(on_submit_cb on_submit,
                                 on_confirm_cb on_confirm,
                                 on_cancel_cb on_cancel) {
  g_on_submit  = on_submit;
  g_on_confirm = on_confirm;
  g_on_cancel  = on_cancel;
}

extern "C" void ui_show(void* parent_hwnd) {
  if (g_dlg) {
    ShowWindow(g_dlg, SW_SHOW);
    SetForegroundWindow(g_dlg);
    return;
  }
  g_dlg = CreateDialogParam(g_hinst, MAKEINTRESOURCE(ID_ASSISTANT_DLG),
                            (HWND)parent_hwnd, DialogProc, 0);
  if (g_dlg) {
    ShowWindow(g_dlg, SW_SHOW);
    SetForegroundWindow(g_dlg);
  }
}

extern "C" void ui_append_output(const char* utf8) {  // MAIN THREAD ONLY
  if (!g_dlg) return;
  append_ctrl_text(GetDlgItem(g_dlg, ID_OUTPUT_EDIT), utf8);
}

extern "C" void ui_set_status(const char* utf8) {  // MAIN THREAD ONLY
  set_ctrl_text(g_dlg, ID_STATUS_TEXT, utf8);
}

extern "C" void ui_close(void) {
  if (g_dlg) DestroyWindow(g_dlg);
}

extern "C" void ui_add_menu_item(void* hmenu, const char* label, int command_id) {
  if (!hmenu || !label) return;
  HMENU m = (HMENU)hmenu;
#ifdef _WIN32
  AppendMenuW(m, MF_STRING, (UINT_PTR)command_id, to_wide(label).c_str());
#else
  AppendMenu(m, MF_STRING, (UINT_PTR)command_id, label);
#endif
}

extern "C" void* ui_create_submenu(void) {
  return (void*)CreatePopupMenu();
}

extern "C" void ui_attach_submenu(void* parent_hmenu, void* submenu, const char* title) {
  if (!parent_hmenu || !submenu || !title) return;
  HMENU parent = (HMENU)parent_hmenu;
#ifdef _WIN32
  AppendMenuW(parent, MF_POPUP, (UINT_PTR)submenu, to_wide(title).c_str());
#else
  AppendMenu(parent, MF_POPUP, (UINT_PTR)submenu, title);
#endif
}
