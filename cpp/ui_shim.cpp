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
  #define RAAI_DLGRET INT_PTR
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
static HWND          g_webview_host = NULL;  // wry's host child (webview), if any
static bool          g_webview_active = false; // true once the webview took over
static on_submit_cb  g_on_submit  = NULL;
static on_confirm_cb g_on_confirm = NULL;
static on_cancel_cb  g_on_cancel  = NULL;
static on_resize_cb  g_on_resize  = NULL;
static on_destroy_cb g_on_destroy = NULL;
static on_webview_focus_cb g_on_webview_focus = NULL;
static prov_list_cb   g_prov_list   = NULL;
static prov_action_cb g_prov_action = NULL;
static HWND        g_prov_dlg = NULL;  // provider LIST dialog, while open
static HWND        g_pe_dlg   = NULL;  // provider settings dialog, while open
static pe_init_cb  g_pe_init  = NULL;
static pe_fetch_cb g_pe_fetch = NULL;
static pe_ok_cb    g_pe_ok    = NULL;
static pe_key_cb   g_pe_key   = NULL;  // key-list button (add/delete/move)

// Classic subclass of the webview host so focus entering it (Tab) is forwarded
// into the web content via Rust (WebView2 MoveFocus). Windows-only.
#ifdef _WIN32
static WNDPROC g_webview_prev_proc = NULL;
static LRESULT CALLBACK WebviewSubclassProc(HWND h, UINT msg, WPARAM w, LPARAM l) {
  LRESULT r = g_webview_prev_proc ? CallWindowProc(g_webview_prev_proc, h, msg, w, l)
                                  : DefWindowProc(h, msg, w, l);
  if (msg == WM_SETFOCUS && g_on_webview_focus) g_on_webview_focus();
  return r;
}
#endif

// Reflow the dialog's controls to fill the client area, so the window (and the
// embedded webview, which tracks the output rect) is comfortably resizable.
static void layout_controls(HWND hwnd) {
  RECT rc;
  GetClientRect(hwnd, &rc);
  const int cw = rc.right, ch = rc.bottom;
  HWND out = GetDlgItem(hwnd, ID_OUTPUT_EDIT);
  // When the webview is active it owns the whole window (it hosts its own input
  // composer). The webview tracks the output-edit rect, so stretch that to fill
  // the client area; the other native controls are hidden (see set_webview_active).
  if (g_webview_active) {
    if (out) SetWindowPos(out, NULL, 0, 0, cw, ch, SWP_NOZORDER | SWP_NOACTIVATE);
    return;
  }
  const int m = 6, bw = 60, bh = 22, ih = 22, sh = 16;
  int closeY = ch - m - bh;
  int inputY = closeY - m - ih;
  int statusY = inputY - m - sh;
  int outH = statusY - 2 * m;
  if (outH < 20) outH = 20;
  HWND st  = GetDlgItem(hwnd, ID_STATUS_TEXT);
  HWND in  = GetDlgItem(hwnd, ID_INPUT_EDIT);
  HWND sb  = GetDlgItem(hwnd, ID_SUBMIT_BTN);
  HWND cb  = GetDlgItem(hwnd, IDCANCEL);
  if (out) SetWindowPos(out, NULL, m, m, cw - 2 * m, outH, SWP_NOZORDER | SWP_NOACTIVATE);
  if (st)  SetWindowPos(st, NULL, m, statusY, cw - 2 * m, sh, SWP_NOZORDER | SWP_NOACTIVATE);
  if (in)  SetWindowPos(in, NULL, m, inputY, cw - 3 * m - bw, ih, SWP_NOZORDER | SWP_NOACTIVATE);
  if (sb)  SetWindowPos(sb, NULL, cw - m - bw, inputY, bw, bh, SWP_NOZORDER | SWP_NOACTIVATE);
  if (cb)  SetWindowPos(cb, NULL, cw - m - bw, closeY, bw, bh, SWP_NOZORDER | SWP_NOACTIVATE);
}

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
  SetDlgItemText(hwnd, id, utf8);
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
  GetDlgItemText(hwnd, id, buf, (int)sizeof(buf));
  return std::string(buf);
#endif
}

// ---- dialog procedure -------------------------------------------------------
static RAAI_DLGRET DialogProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_INITDIALOG: {
      g_dlg = hwnd;
#ifdef _WIN32
      // Raise the output edit's text limit so a long conversation isn't clipped.
      // (SWELL edit controls have no small default limit, so this is Win32-only.)
      HWND out = GetDlgItem(hwnd, ID_OUTPUT_EDIT);
      if (out) SendMessage(out, EM_SETLIMITTEXT, (WPARAM)0, 0);
#endif
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
          if (g_on_cancel) g_on_cancel();  // conversation history survives; but
          ShowWindow(hwnd, SW_HIDE);       // disarm pixel control + stop any turn
          return TRUE;                     // so "close" is a real kill switch.
      }
      return FALSE;

    case WM_SIZE:
      layout_controls(hwnd);
      if (g_on_resize) g_on_resize();   // let Rust re-bound the webview
      return TRUE;

    case WM_CLOSE:                 // window [x]: hide, keep the window + history,
      if (g_on_cancel) g_on_cancel();  // but disarm pixel control + stop the turn.
      ShowWindow(hwnd, SW_HIDE);
      return TRUE;

    case WM_DESTROY:
      if (g_on_destroy) g_on_destroy();  // drop the webview while its parent lives
      if (g_on_cancel) g_on_cancel();
      g_dlg = NULL;
      g_webview_host = NULL;
      g_webview_active = false;
      return TRUE;
  }
  return FALSE;  // not handled: default processing
}

// ---- provider management dialog (Phase 5, M4) -------------------------------
// Add a UTF-8 string to a listbox or menu using the platform-correct call.
static void add_list_item(HWND lb, const char* utf8) {
#ifdef _WIN32
  SendMessageW(lb, LB_ADDSTRING, 0, (LPARAM)to_wide(utf8).c_str());
#else
  SendMessage(lb, LB_ADDSTRING, 0, (LPARAM)utf8);
#endif
}

// Repopulate the listbox from Rust, preserving the selection index if possible.
static void populate_prov_list(HWND hwnd) {
  HWND lb = GetDlgItem(hwnd, ID_PROV_LIST);
  if (!lb) return;
  int sel = (int)SendMessage(lb, LB_GETCURSEL, 0, 0);
  SendMessage(lb, LB_RESETCONTENT, 0, 0);
  if (g_prov_list) {
    char buf[8192];
    buf[0] = 0;
    g_prov_list(buf, (int)sizeof(buf));
    std::string s(buf);
    size_t start = 0;
    while (start <= s.size()) {
      size_t nl = s.find('\n', start);
      std::string line =
          (nl == std::string::npos) ? s.substr(start) : s.substr(start, nl - start);
      if (!line.empty()) add_list_item(lb, line.c_str());
      if (nl == std::string::npos) break;
      start = nl + 1;
    }
  }
  int count = (int)SendMessage(lb, LB_GETCOUNT, 0, 0);
  if (count > 0) {
    if (sel < 0) sel = 0;
    if (sel >= count) sel = count - 1;
    SendMessage(lb, LB_SETCURSEL, (WPARAM)sel, 0);
  }
}

// Run a provider action in Rust for the selected row; repopulate if it changed.
static void do_prov_action(HWND hwnd, int action) {
  if (!g_prov_action) return;
  HWND lb = GetDlgItem(hwnd, ID_PROV_LIST);
  int sel = lb ? (int)SendMessage(lb, LB_GETCURSEL, 0, 0) : -1;
  if (g_prov_action(action, sel)) populate_prov_list(hwnd);
}

static RAAI_DLGRET ProvidersProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_INITDIALOG:
      g_prov_dlg = hwnd;  // so the settings dialog can own (disable) this one
      populate_prov_list(hwnd);
      return TRUE;
    case WM_COMMAND:
      switch (LOWORD(wParam)) {
        case ID_PROV_ADD:     do_prov_action(hwnd, 0); return TRUE;
        case ID_PROV_EDIT:    do_prov_action(hwnd, 1); return TRUE;
        case ID_PROV_DELETE:  do_prov_action(hwnd, 2); return TRUE;
        case ID_PROV_DEFAULT: do_prov_action(hwnd, 3); return TRUE;
        case ID_PROV_LIST:
          if (HIWORD(wParam) == LBN_DBLCLK) { do_prov_action(hwnd, 1); return TRUE; }
          return FALSE;
        case IDCANCEL:
          EndDialog(hwnd, 0);
          return TRUE;
      }
      return FALSE;
    case WM_DESTROY:
      g_prov_dlg = NULL;
      return FALSE;
  }
  return FALSE;
}

// ---- provider settings dialog (Phase 5, M5) ---------------------------------
static RAAI_DLGRET ProviderEditProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_INITDIALOG:
      g_pe_dlg = hwnd;
      if (g_pe_init) g_pe_init();  // Rust prefills the fields
      return TRUE;                 // default focus (first tabstop)
    case WM_COMMAND:
      switch (LOWORD(wParam)) {
        case ID_PE_FETCH:
          if (g_pe_fetch) g_pe_fetch();
          return TRUE;
        // Key-list buttons: Rust mutates its working list and repopulates.
        case ID_PE_KEYADD:   if (g_pe_key) g_pe_key(0); return TRUE;
        case ID_PE_KEYDEL:   if (g_pe_key) g_pe_key(1); return TRUE;
        case ID_PE_KEYUP:    if (g_pe_key) g_pe_key(2); return TRUE;
        case ID_PE_KEYDOWN:  if (g_pe_key) g_pe_key(3); return TRUE;
        case IDOK:
          // Rust saves and decides whether to close (0 = keep open on error).
          if (!g_pe_ok || g_pe_ok()) EndDialog(hwnd, 1);
          return TRUE;
        case IDCANCEL:
          EndDialog(hwnd, 0);
          return TRUE;
      }
      return FALSE;
    case WM_DESTROY:
      g_pe_dlg = NULL;
      return FALSE;
  }
  return FALSE;
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
  // NULL owner: an independent top-level window, NOT owned by REAPER's main
  // window — so it can sit alongside REAPER (behind it, on another monitor,
  // minimized independently) instead of always floating on top. REAPER still
  // pumps our messages (same thread) and, with the accelerator hook registered
  // (ui_translate_accel), routes keyboard to us. parent_hwnd is now unused.
  (void)parent_hwnd;
  g_dlg = CreateDialogParam(g_hinst, MAKEINTRESOURCE(ID_ASSISTANT_DLG),
                            NULL, DialogProc, 0);
  if (g_dlg) {
#ifdef _WIN32
    // Give the unowned window its own taskbar button so the user can Alt+Tab to
    // it and find it in the taskbar while working in REAPER.
    LONG_PTR ex = GetWindowLongPtr(g_dlg, GWL_EXSTYLE);
    SetWindowLongPtr(g_dlg, GWL_EXSTYLE, ex | WS_EX_APPWINDOW);
#endif
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

// ---- webview hosting --------------------------------------------------------
extern "C" void* ui_get_hwnd(void) {
  return (void*)g_dlg;
}

extern "C" int ui_output_bounds(int* x, int* y, int* w, int* h) {
  if (!g_dlg || !x || !y || !w || !h) return 0;
  HWND edit = GetDlgItem(g_dlg, ID_OUTPUT_EDIT);
  if (!edit) return 0;
  RECT r;
  if (!GetWindowRect(edit, &r)) return 0;
  POINT tl = { r.left, r.top };
  POINT br = { r.right, r.bottom };
  ScreenToClient(g_dlg, &tl);
  ScreenToClient(g_dlg, &br);
  *x = tl.x;
  *y = tl.y;
  *w = br.x - tl.x;
  *h = br.y - tl.y;
  return 1;
}

extern "C" void ui_set_output_edit_visible(int visible) {
  if (!g_dlg) return;
  HWND edit = GetDlgItem(g_dlg, ID_OUTPUT_EDIT);
  if (edit) ShowWindow(edit, visible ? SW_SHOW : SW_HIDE);
}

extern "C" void ui_set_resize_cb(on_resize_cb on_resize) {
  g_on_resize = on_resize;
}

extern "C" void ui_set_destroy_cb(on_destroy_cb on_destroy) {
  g_on_destroy = on_destroy;
}

extern "C" void ui_enable_webview_tabstop(void) {
  // The embedded webview is Windows-only (wry/WebView2), so this whole
  // focus-forwarding dance is Win32-only; a no-op on SWELL.
#ifdef _WIN32
  if (!g_dlg) return;
  // The webview's host window is the first direct child of the dialog that is
  // NOT one of our known controls. Give it WS_TABSTOP (so the Tab cycle lands on
  // it) and subclass it so focus is forwarded into the web content.
  for (HWND child = GetWindow(g_dlg, GW_CHILD); child; child = GetWindow(child, GW_HWNDNEXT)) {
    int id = GetDlgCtrlID(child);
    if (id == ID_OUTPUT_EDIT || id == ID_STATUS_TEXT || id == ID_INPUT_EDIT ||
        id == ID_SUBMIT_BTN || id == IDCANCEL) {
      continue;
    }
    g_webview_host = child;
    LONG_PTR style = GetWindowLongPtr(child, GWL_STYLE);
    SetWindowLongPtr(child, GWL_STYLE, style | WS_TABSTOP);
    if (!g_webview_prev_proc) {
      g_webview_prev_proc =
          (WNDPROC)SetWindowLongPtr(child, GWLP_WNDPROC, (LONG_PTR)WebviewSubclassProc);
    }
    break; // one host window is enough
  }
#endif
}

// Give the whole window to the webview: hide every native control (the webview
// hosts its own conversation + input composer) and stretch the output rect the
// webview tracks to fill the client. Called once after the webview is created.
extern "C" void ui_set_webview_active(int active) {
  if (!g_dlg) return;
  g_webview_active = active ? true : false;
  static const int ids[] = { ID_OUTPUT_EDIT, ID_STATUS_TEXT, ID_INPUT_EDIT,
                             ID_SUBMIT_BTN, IDCANCEL };
  for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
    HWND c = GetDlgItem(g_dlg, ids[i]);
    if (c) ShowWindow(c, g_webview_active ? SW_HIDE : SW_SHOW);
  }
  layout_controls(g_dlg);
}

// Keyboard router for REAPER's accelerator queue (registered via plugin_register
// "accelerator"). Two jobs, both only for keystrokes aimed at OUR window:
//   1. Stop REAPER from swallowing them as global shortcuts (so typing in the
//      webview types, and doesn't e.g. start/stop playback). We return -1
//      ("pass on to my window") for anything the webview/composer should get.
//   2. Let the native dialog manager handle Tab/Esc/mnemonics for the NATIVE
//      controls (the non-webview fallback), but NOT while the webview owns focus
//      — WebView2 does its own Tab cycling and IsDialogMessage would yank focus
//      out of the composer on the first Tab.
// Returns: 0 = not our window (let REAPER process), 1 = eaten, -1 = pass on.
extern "C" int ui_translate_accel(void* msgp) {
#ifdef _WIN32
  if (!g_dlg || !msgp) return 0;
  MSG* msg = (MSG*)msgp;
  if (msg->hwnd != g_dlg && !IsChild(g_dlg, msg->hwnd)) return 0; // not ours
  // WM_SYSKEY*/VK_MENU (Alt) would be DROPPED by the plain "pass on" (-1);
  // force-deliver them (-20) so Alt, Alt+mnemonics and Alt+F4 reach the window.
  if (msg->message == WM_SYSKEYDOWN || msg->message == WM_SYSKEYUP) return -20;
  // When the webview owns the whole window it handles ALL its own keys; the
  // dialog manager has nothing to navigate, so never run IsDialogMessage — on a
  // focus transient it could steal Tab or turn Esc into hide-window.
  if (g_webview_active) return -1;
  HWND focus = GetFocus();
  bool in_webview = g_webview_host &&
      (focus == g_webview_host || IsChild(g_webview_host, focus));
  if (!in_webview && IsDialogMessage(g_dlg, msg)) return 1; // dialog handled it
  return -1; // ours: deliver to the window, don't apply REAPER shortcuts
#else
  // macOS/SWELL: REAPER's accelerator hook (registered at the Front position) sees
  // every keystroke before the Cocoa responder chain, so keys the user types into
  // the webview composer would otherwise fire as global REAPER actions (Space =
  // Play/Stop, R = record, ...). Mirror the Win32 intent: for any key targeting our
  // dialog subtree, return -1 ("pass on to my window") so REAPER does not consume
  // it and the keystroke reaches the webview. IsChild walks the NSView hierarchy
  // (isDescendantOf:), so it recognises the wry-injected WKWebView subview even
  // though SWELL never created it.
  if (!g_dlg || !msgp) return 0;
  MSG* msg = (MSG*)msgp;
  if (msg->hwnd != g_dlg && !IsChild(g_dlg, msg->hwnd)) return 0; // not ours
  return -1; // ours: deliver to the webview, don't apply REAPER shortcuts
#endif
}

extern "C" void ui_set_webview_focus_cb(on_webview_focus_cb on_focus) {
  g_on_webview_focus = on_focus;
}

extern "C" void ui_focus_after_webview(int forward) {
  // Webview-only (Windows): hand focus off after a Tab out of the web content.
#ifdef _WIN32
  if (!g_dlg) return;
  // Forward tab-out -> the input box (next control); backward -> the Close
  // button (the control before the output area in the cycle).
  HWND target = GetDlgItem(g_dlg, forward ? ID_INPUT_EDIT : IDCANCEL);
  if (target) {
    SendMessage(g_dlg, WM_NEXTDLGCTL, (WPARAM)target, TRUE);
  }
#else
  (void)forward;
#endif
}

extern "C" int ui_window_rect(void* hwnd, int* x, int* y, int* w, int* h) {
  if (!hwnd || !x || !y || !w || !h) return 0;
  RECT r;
  if (!GetWindowRect((HWND)hwnd, &r)) return 0;
  *x = r.left;
  *y = r.top;
  *w = r.right - r.left;
  *h = r.bottom - r.top;
  return 1;
}

extern "C" void ui_window_to_front(void* hwnd) {
  if (!hwnd) return;
  HWND h = (HWND)hwnd;
#ifdef _WIN32
  if (IsIconic(h)) ShowWindow(h, SW_RESTORE);
#endif
  SetForegroundWindow(h);
}

// Find the first VISIBLE top-level window whose title contains `needle`
// (case-insensitive). Used to locate REAPER's floating Video window (title
// "Video Window") for capture, since REAPER exposes no API for its HWND.
// Returns NULL if none matches. Main thread only.
struct FindWinCtx { std::string needle; HWND found; };

static BOOL CALLBACK find_win_cb(HWND hwnd, LPARAM lp) {
  FindWinCtx* ctx = (FindWinCtx*)lp;
  if (!IsWindowVisible(hwnd)) return TRUE;
#ifdef _WIN32
  // Only OUR (REAPER's) windows — EnumWindows is desktop-wide on Win32, so without
  // this a foreign window merely CONTAINING the title (e.g. a browser tab about the
  // REAPER video window) could be matched and its pixels captured/uploaded. On
  // macOS/Linux SWELL enumerates only in-process windows, so no filter is needed.
  DWORD pid = 0;
  GetWindowThreadProcessId(hwnd, &pid);
  if (pid != GetCurrentProcessId()) return TRUE;
#endif
  char title[512];
  title[0] = 0;
  GetWindowText(hwnd, title, (int)sizeof(title));
  std::string t(title);
  for (char& c : t) {
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
  }
  if (t.find(ctx->needle) != std::string::npos) {
    ctx->found = hwnd;
    return FALSE; // stop enumerating
  }
  return TRUE;
}

extern "C" void* ui_find_window_by_title(const char* needle_utf8) {
  if (!needle_utf8 || !needle_utf8[0]) return NULL;
  std::string needle(needle_utf8);
  for (char& c : needle) {
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
  }
  FindWinCtx ctx{needle, NULL};
  EnumWindows(find_win_cb, (LPARAM)&ctx);
  return (void*)ctx.found;
}

extern "C" void ui_add_menu_item(void* hmenu, const char* label, int command_id) {
  if (!hmenu || !label) return;
  HMENU m = (HMENU)hmenu;
#ifdef _WIN32
  AppendMenuW(m, MF_STRING, (UINT_PTR)command_id, to_wide(label).c_str());
#else
  InsertMenu(m, -1, MF_BYPOSITION | MF_STRING, (UINT_PTR)command_id, label);
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
  InsertMenu(parent, -1, MF_BYPOSITION | MF_POPUP, (UINT_PTR)submenu, title);
#endif
}

extern "C" void ui_set_provider_cbs(prov_list_cb on_list, prov_action_cb on_action) {
  g_prov_list   = on_list;
  g_prov_action = on_action;
}

extern "C" void ui_show_providers(void) {
  // Modal, parented to the assistant window if it exists (else REAPER's front
  // window). Rust owns the list/action logic via the registered callbacks.
  HWND parent = g_dlg ? g_dlg : GetForegroundWindow();
  DialogBoxParam(g_hinst, MAKEINTRESOURCE(ID_PROVIDERS_DLG), parent, ProvidersProc, 0);
}

extern "C" int ui_popup_menu(const char* items_newline) {
  if (!items_newline) return 0;
  HMENU menu = CreatePopupMenu();
  if (!menu) return 0;
  std::string s(items_newline);
  size_t start = 0;
  int idx = 1;
  while (start <= s.size()) {
    size_t nl = s.find('\n', start);
    std::string line =
        (nl == std::string::npos) ? s.substr(start) : s.substr(start, nl - start);
    if (!line.empty()) {
#ifdef _WIN32
      AppendMenuW(menu, MF_STRING, (UINT_PTR)idx, to_wide(line.c_str()).c_str());
#else
      InsertMenu(menu, -1, MF_BYPOSITION | MF_STRING, (UINT_PTR)idx, line.c_str());
#endif
      idx++;
    }
    if (nl == std::string::npos) break;
    start = nl + 1;
  }
  HWND owner = g_pe_dlg ? g_pe_dlg : (g_prov_dlg ? g_prov_dlg : (g_dlg ? g_dlg : GetForegroundWindow()));
  POINT pt;
  GetCursorPos(&pt);
  int chosen = (int)TrackPopupMenu(
      menu, TPM_RETURNCMD | TPM_NONOTIFY | TPM_LEFTALIGN | TPM_TOPALIGN,
      pt.x, pt.y, 0, owner, NULL);
  DestroyMenu(menu);
  return chosen;  // 0 = cancelled, else 1-based index
}

extern "C" int ui_message_box(const char* title, const char* text, int flags) {
  HWND owner = g_pe_dlg ? g_pe_dlg : (g_prov_dlg ? g_prov_dlg : (g_dlg ? g_dlg : GetForegroundWindow()));
  UINT type = (flags == 1) ? (MB_YESNO | MB_ICONQUESTION) : (MB_OK | MB_ICONINFORMATION);
#ifdef _WIN32
  int r = MessageBoxW(owner, to_wide(text ? text : "").c_str(),
                      to_wide(title ? title : "").c_str(), type);
#else
  int r = MessageBox(owner, text ? text : "", title ? title : "", type);
#endif
  return (r == IDYES || r == IDOK) ? 1 : 0;
}

extern "C" void ui_set_provider_edit_cbs(pe_init_cb on_init, pe_fetch_cb on_fetch,
                                         pe_ok_cb on_ok, pe_key_cb on_key) {
  g_pe_init  = on_init;
  g_pe_fetch = on_fetch;
  g_pe_ok    = on_ok;
  g_pe_key   = on_key;
}

extern "C" int ui_show_provider_edit(void) {
  // Own the provider LIST dialog (its actual parent modal) so it's disabled
  // while settings is open; fall back to the assistant window, then foreground.
  HWND parent = g_prov_dlg ? g_prov_dlg : (g_dlg ? g_dlg : GetForegroundWindow());
  INT_PTR r = DialogBoxParam(g_hinst, MAKEINTRESOURCE(ID_PROVIDER_EDIT_DLG), parent,
                             ProviderEditProc, 0);
  return r == 1 ? 1 : 0;
}

extern "C" void ui_pe_set_text(int ctrl, const char* utf8) {
  if (g_pe_dlg) set_ctrl_text(g_pe_dlg, ctrl, utf8);
}

extern "C" void ui_pe_get_text(int ctrl, char* buf, int buf_sz) {
  if (!buf || buf_sz <= 0) return;
  buf[0] = 0;
  if (!g_pe_dlg) return;
  std::string s = get_ctrl_text(g_pe_dlg, ctrl);
  int n = (int)s.size();
  if (n > buf_sz - 1) n = buf_sz - 1;
  if (n > 0) memcpy(buf, s.data(), (size_t)n);
  buf[n] = 0;
}

extern "C" void ui_pe_set_check(int ctrl, int checked) {
  if (g_pe_dlg) CheckDlgButton(g_pe_dlg, ctrl, checked ? BST_CHECKED : BST_UNCHECKED);
}

extern "C" int ui_pe_get_check(int ctrl) {
  return (g_pe_dlg && IsDlgButtonChecked(g_pe_dlg, ctrl) == BST_CHECKED) ? 1 : 0;
}

extern "C" void ui_pe_show(int ctrl, int visible) {
  if (!g_pe_dlg) return;
  HWND c = GetDlgItem(g_pe_dlg, ctrl);
  if (c) ShowWindow(c, visible ? SW_SHOW : SW_HIDE);
}

// Fill the key listbox from a newline-separated (already-masked) item list, in
// order. Rust drives it after every add/delete/move; selection is set separately.
extern "C" void ui_pe_set_list(int ctrl, const char* items_newline) {
  if (!g_pe_dlg) return;
  HWND lb = GetDlgItem(g_pe_dlg, ctrl);
  if (!lb) return;
  SendMessage(lb, LB_RESETCONTENT, 0, 0);
  std::string s(items_newline ? items_newline : "");
  size_t start = 0;
  while (start <= s.size()) {
    size_t nl = s.find('\n', start);
    std::string line =
        (nl == std::string::npos) ? s.substr(start) : s.substr(start, nl - start);
    if (!line.empty()) add_list_item(lb, line.c_str());
    if (nl == std::string::npos) break;
    start = nl + 1;
  }
}

extern "C" int ui_pe_get_sel(int ctrl) {
  if (!g_pe_dlg) return -1;
  HWND lb = GetDlgItem(g_pe_dlg, ctrl);
  if (!lb) return -1;
  int sel = (int)SendMessage(lb, LB_GETCURSEL, 0, 0);
  return sel;  // LB_ERR is -1
}

extern "C" void ui_pe_set_sel(int ctrl, int index) {
  if (!g_pe_dlg) return;
  HWND lb = GetDlgItem(g_pe_dlg, ctrl);
  if (lb) SendMessage(lb, LB_SETCURSEL, (WPARAM)index, 0);
}

// ---- SWELL dialog/menu resource registration (macOS/Linux) ------------------
// On non-Windows there is no rc.exe: SWELL learns about our dialog templates ONLY
// from the tables swell_resgen.php generates from assistant.rc (see build.rs),
// pulled in here. WITHOUT these includes SWELL has no record of our dialogs, so
// CreateDialogParam / DialogBoxParam(MAKEINTRESOURCE(...)) return NULL at runtime
// and "Open window" / "Providers" open nothing (focus just returns to REAPER).
// swell-dlggen.h / swell-menugen.h define the macros the generated tables use;
// each must precede its table. assistant.rc has no MENU, so the menu table is
// empty but still generated.
#ifndef _WIN32
#include "swell-dlggen.h"
// SWELL doesn't define these Win32 listbox style flags — its listboxes notify and
// hold strings unconditionally — so map them to 0 to let the generated table
// (which carries the .rc's LISTBOX styles verbatim) compile.
#ifndef LBS_NOTIFY
#define LBS_NOTIFY 0
#endif
#ifndef LBS_HASSTRINGS
#define LBS_HASSTRINGS 0
#endif
#include "assistant.rc_mac_dlg"
#include "swell-menugen.h"
#include "assistant.rc_mac_menu"
#endif
