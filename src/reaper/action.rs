//! Registers the extension's actions ("Open window" and "Providers") and mirrors
//! them into REAPER's Extensions menu. Both actions share one `HookCommand` that
//! dispatches on the command id. The callbacks are static, so the command ids and
//! the REAPER main window handle live in process globals.

use std::error::Error;
use std::ffi::c_void;
use std::sync::OnceLock;

use reaper_medium::{
    AcceleratorPosition, CommandId, Hmenu, HookCommand, HookCustomMenu, MenuHookFlag,
    OwnedGaccelRegister, ReaperSession, ReaperStr, ToggleAction, ToggleActionResult,
    TranslateAccel, TranslateAccelArgs, TranslateAccelResult,
};

use crate::ai::protocol::TranscribeOutput;
use crate::ui;

static CMD_OPEN: OnceLock<u32> = OnceLock::new();
static CMD_PROVIDERS: OnceLock<u32> = OnceLock::new();
static CMD_PRESETS: OnceLock<u32> = OnceLock::new();
static CMD_AUTOAPPROVE: OnceLock<u32> = OnceLock::new();
static CMD_TRANSCRIBE_NOTES: OnceLock<u32> = OnceLock::new();
static CMD_TRANSCRIBE_TEXT: OnceLock<u32> = OnceLock::new();
static CMD_TRANSCRIBE_SRT: OnceLock<u32> = OnceLock::new();
static CMD_CUT_EDITOR: OnceLock<u32> = OnceLock::new();
static MAIN_HWND: OnceLock<usize> = OnceLock::new();

struct Commands;

impl HookCommand for Commands {
    fn call(command_id: CommandId, _flag: i32) -> bool {
        let id = command_id.get();
        if Some(id) == CMD_OPEN.get().copied() {
            if let Some(h) = MAIN_HWND.get().copied() {
                ui::ffi::show(h as *mut c_void);
                // Create the embedded webview now that the dialog HWND exists
                // (idempotent; no-op if already created or unavailable).
                ui::output::ensure_created();
            }
            true
        } else if Some(id) == CMD_PROVIDERS.get().copied() {
            ui::ffi::show_providers();
            true
        } else if Some(id) == CMD_PRESETS.get().copied() {
            ui::ffi::show_presets();
            true
        } else if Some(id) == CMD_AUTOAPPROVE.get().copied() {
            // Advanced mode: apply the model's edits without a per-request
            // confirmation. Speak the new state so it's clear what changed.
            let on = crate::providers::registry::toggle_auto_approve();
            ui::output::speak(if on {
                "Advanced mode on. The assistant applies edits without asking."
            } else {
                "Advanced mode off. The assistant asks before applying edits."
            });
            true
        } else if Some(id) == CMD_TRANSCRIBE_NOTES.get().copied() {
            crate::ui::bridge::transcribe(TranscribeOutput::Notes);
            true
        } else if Some(id) == CMD_TRANSCRIBE_TEXT.get().copied() {
            crate::ui::bridge::transcribe(TranscribeOutput::Text);
            true
        } else if Some(id) == CMD_TRANSCRIBE_SRT.get().copied() {
            crate::ui::bridge::transcribe(TranscribeOutput::Srt);
            true
        } else if Some(id) == CMD_CUT_EDITOR.get().copied() {
            // Cut by text: just kick off the worker. The window/webview is opened
            // lazily only once transcription succeeds and the editor is about to
            // show — so an error (no provider, no word timings) is announced via
            // OSARA without popping the pane and stealing focus.
            crate::ui::bridge::open_cut_editor();
            true
        } else {
            false
        }
    }
}

/// Keyboard router for our assistant window. Registered in REAPER's accelerator
/// queue so that — now the window is unowned (not a child of REAPER's main
/// window) — keystrokes aimed at it still reach it (Tab/Esc for the native
/// fallback controls) and REAPER does NOT swallow them as global actions while
/// the window is focused (critical for typing in the webview composer). All the
/// window-specific logic lives in the shim (`ui_translate_accel`).
struct AccelHook;

impl TranslateAccel for AccelHook {
    fn call(&mut self, args: TranslateAccelArgs) -> TranslateAccelResult {
        // The shim needs a Win32/SWELL MSG pointer to call IsDialogMessage.
        let mut msg = args.msg.raw();
        match ui::ffi::translate_accel(&mut msg as *mut _ as *mut c_void) {
            1 => TranslateAccelResult::Eat,
            -1 => TranslateAccelResult::PassOnToWindow,
            // macOS, cut-by-text editor open: the shim claimed an editing key
            // (2 = plain, 3 = Shift held). Eat it and drive the editor by JS
            // injection — this never depends on WKWebView keyboard focus, the
            // lane that kept failing on live mac tests. Runs on the main
            // thread, where eval is safe.
            r @ (2 | 3) => {
                if let Some(name) = editor_key_name(msg.wParam as u32) {
                    crate::ui::output::editor_host_key(name, r == 3);
                }
                TranslateAccelResult::Eat
            }
            // macOS: hand the raw NSEvent back to Cocoa so the WKWebView handles
            // native editing itself (Cmd+C/V/X/A, arrows, typing) instead of REAPER
            // swallowing it (e.g. Cmd+V hitting REAPER's Edit > Paste).
            -10 => TranslateAccelResult::ProcessEventRaw,
            // Deliver Alt/WM_SYSKEY* to the window (plain pass-on drops them).
            -20 => TranslateAccelResult::ForcePassOnToWindow,
            _ => TranslateAccelResult::NotOurWindow,
        }
    }
}

/// The DOM `KeyboardEvent.key` name for a host-routed editor key's VK code —
/// exactly the set the shim claims while the editor is open.
fn editor_key_name(vk: u32) -> Option<&'static str> {
    Some(match vk {
        0x25 => "ArrowLeft",
        0x26 => "ArrowUp",
        0x27 => "ArrowRight",
        0x28 => "ArrowDown",
        0x24 => "Home",
        0x23 => "End",
        0x20 => " ",
        0x2E => "Delete",
        0x08 => "Backspace",
        0x1B => "Escape",
        _ => return None,
    })
}

/// Reports the on/off state of our toggleable commands. REAPER queries this
/// whenever it needs the state — drawing the Extensions-menu checkmark, the
/// Actions list's State column, toolbar button states — so the display can
/// never go stale, unlike state baked into a menu label at build time.
struct ToggleState;

impl ToggleAction for ToggleState {
    fn call(command_id: CommandId) -> ToggleActionResult {
        if CMD_AUTOAPPROVE.get().copied() == Some(command_id.get()) {
            if crate::providers::registry::auto_approve() {
                ToggleActionResult::On
            } else {
                ToggleActionResult::Off
            }
        } else {
            ToggleActionResult::NotRelevant
        }
    }
}

/// Adds a "ReaLackey" submenu (holding all our entries) to REAPER's
/// Extensions menu, wired to the same command ids as the actions.
struct ExtMenu;

impl HookCustomMenu for ExtMenu {
    fn call(menuidstr: &ReaperStr, menu: Hmenu, flag: MenuHookFlag) {
        // REAPER calls this with `Init` when it wants us to populate the menu.
        if flag != MenuHookFlag::Init || menuidstr.as_c_str() != c"Main extensions" {
            return;
        }
        let parent: *mut c_void = menu.as_ptr().cast();
        let submenu = ui::ffi::create_submenu();
        if submenu.is_null() {
            return;
        }
        if let Some(id) = CMD_OPEN.get().copied() {
            ui::ffi::add_menu_item(submenu, "Open window", id as i32);
        }
        if let Some(id) = CMD_PROVIDERS.get().copied() {
            ui::ffi::add_menu_item(submenu, "Providers\u{2026}", id as i32);
        }
        if let Some(id) = CMD_PRESETS.get().copied() {
            ui::ffi::add_menu_item(submenu, "Prompt presets\u{2026}", id as i32);
        }
        if let Some(id) = CMD_AUTOAPPROVE.get().copied() {
            // Plain label: the on/off state is reported through the
            // `toggleaction` hook (ToggleState below) and REAPER draws it as a
            // native menu checkmark, which screen readers announce as
            // "checked". The state used to live in the label TEXT — but macOS
            // builds its Cocoa menu once and re-fires this menu hook rarely if
            // ever, so the label froze at the startup state and always read
            // "off" (live mac report). The checkmark is queried when the menu
            // opens, on both platforms, and the Actions list's State column
            // shows on/off for free.
            ui::ffi::add_menu_item(submenu, "Advanced mode (auto-approve edits)", id as i32);
        }
        if let Some(id) = CMD_TRANSCRIBE_NOTES.get().copied() {
            ui::ffi::add_menu_item(submenu, "Transcribe selected item \u{2192} notes", id as i32);
        }
        if let Some(id) = CMD_TRANSCRIBE_TEXT.get().copied() {
            ui::ffi::add_menu_item(submenu, "Transcribe selected item \u{2192} text file", id as i32);
        }
        if let Some(id) = CMD_TRANSCRIBE_SRT.get().copied() {
            ui::ffi::add_menu_item(submenu, "Transcribe selected item \u{2192} SRT file", id as i32);
        }
        // "Cut selected item by text" is intentionally NOT in the menu — it stays a
        // registered action (bind it to a key via Actions), just not a menu entry.
        ui::ffi::attach_submenu(parent, submenu, "ReaLackey");
    }
}

/// Show the assistant window and bring up its webview, on the main thread. Called
/// (via a UiEvent) when the cut-by-text editor is about to open, so the pane only
/// appears when it's actually needed. Idempotent — no-op if already shown/created.
pub fn ensure_window_shown() {
    if let Some(h) = MAIN_HWND.get().copied() {
        ui::ffi::show(h as *mut c_void);
        ui::output::ensure_created();
    }
}

pub fn register(session: &mut ReaperSession) -> Result<(), Box<dyn Error>> {
    // REAPER main window (parent for the modeless dialog).
    let hwnd = session.reaper().get_main_hwnd();
    let _ = MAIN_HWND.set(hwnd.as_ptr() as usize);

    // Action: open the assistant window.
    let cmd_open = session.plugin_register_add_command_id("RAAI_OpenAssistant")?;
    let _ = CMD_OPEN.set(cmd_open.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_open,
        "ReaLackey: Open window",
    ))?;

    // Action: manage providers (add / edit / delete / set-default), including
    // per-provider API keys (this superseded the standalone "Set API key" action).
    let cmd_providers = session.plugin_register_add_command_id("RAAI_Providers")?;
    let _ = CMD_PROVIDERS.set(cmd_providers.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_providers,
        "ReaLackey: Providers",
    ))?;

    // Action: manage prompt presets (reusable prompts inserted into the composer).
    // No default key binding — the user binds it in REAPER's Actions list.
    let cmd_presets = session.plugin_register_add_command_id("RAAI_Presets")?;
    let _ = CMD_PRESETS.set(cmd_presets.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_presets,
        "ReaLackey: Prompt presets",
    ))?;

    // Action: toggle "advanced mode" — apply the model's edits without asking for
    // confirmation each time. Bindable to a key from REAPER's Actions list.
    let cmd_autoapprove = session.plugin_register_add_command_id("RAAI_ToggleAutoApprove")?;
    let _ = CMD_AUTOAPPROVE.set(cmd_autoapprove.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_autoapprove,
        "ReaLackey: Toggle advanced mode (auto-approve edits)",
    ))?;

    // Actions: transcribe the SELECTED item's audio to text, straight to a
    // destination — no chat needed. Bindable to keys from REAPER's Actions list.
    let cmd_tr_notes = session.plugin_register_add_command_id("RAAI_TranscribeToNotes")?;
    let _ = CMD_TRANSCRIBE_NOTES.set(cmd_tr_notes.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_tr_notes,
        "ReaLackey: Transcribe selected item to its notes",
    ))?;
    let cmd_tr_text = session.plugin_register_add_command_id("RAAI_TranscribeToText")?;
    let _ = CMD_TRANSCRIBE_TEXT.set(cmd_tr_text.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_tr_text,
        "ReaLackey: Transcribe selected item to a text file",
    ))?;
    let cmd_tr_srt = session.plugin_register_add_command_id("RAAI_TranscribeToSrt")?;
    let _ = CMD_TRANSCRIBE_SRT.set(cmd_tr_srt.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_tr_srt,
        "ReaLackey: Transcribe selected item to an SRT subtitle file",
    ))?;

    // Action: open the cut-by-text editor on the selected item (transcribe, edit
    // the text in a modal, then cut what was removed). Bindable from the Actions list.
    let cmd_cut_editor = session.plugin_register_add_command_id("RAAI_CutByText")?;
    let _ = CMD_CUT_EDITOR.set(cmd_cut_editor.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_cut_editor,
        "ReaLackey: Cut selected item by text",
    ))?;

    // One handler dispatches all command ids.
    session.plugin_register_add_hook_command::<Commands>()?;

    // Keyboard router for the (unowned) assistant window: keeps Tab/Esc working
    // and stops REAPER from eating keystrokes meant for the webview composer.
    session.plugin_register_add_accelerator_register(Box::new(AccelHook), AcceleratorPosition::Front)?;

    // Mirror the actions into REAPER's Extensions menu.
    session.reaper().add_extensions_main_menu();
    session.plugin_register_add_hook_custom_menu::<ExtMenu>()?;
    // On/off state provider for toggleable commands (menu checkmark, Actions
    // list, toolbars) — queried live, so it can't go stale on any platform.
    session.plugin_register_add_toggle_action::<ToggleState>()?;
    Ok(())
}
