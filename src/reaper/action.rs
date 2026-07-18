//! Registers the extension's actions ("Open window" and "Providers") and mirrors
//! them into REAPER's Extensions menu. Both actions share one `HookCommand` that
//! dispatches on the command id. The callbacks are static, so the command ids and
//! the REAPER main window handle live in process globals.

use std::error::Error;
use std::ffi::c_void;
use std::sync::OnceLock;

use reaper_medium::{
    AcceleratorPosition, CommandId, Hmenu, HookCommand, HookCustomMenu, MenuHookFlag,
    OwnedGaccelRegister, ReaperSession, ReaperStr, TranslateAccel, TranslateAccelArgs,
    TranslateAccelResult,
};

use crate::ui;

static CMD_OPEN: OnceLock<u32> = OnceLock::new();
static CMD_PROVIDERS: OnceLock<u32> = OnceLock::new();
static CMD_PRESETS: OnceLock<u32> = OnceLock::new();
static CMD_AUTOAPPROVE: OnceLock<u32> = OnceLock::new();
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
            // Label carries the current state (read by the screen reader on menu
            // open) — the menu-item API here has no separate checkmark flag.
            let label = if crate::providers::registry::auto_approve() {
                "Advanced mode (auto-approve edits): on"
            } else {
                "Advanced mode (auto-approve edits): off"
            };
            ui::ffi::add_menu_item(submenu, label, id as i32);
        }
        ui::ffi::attach_submenu(parent, submenu, "ReaLackey");
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

    // One handler dispatches all command ids.
    session.plugin_register_add_hook_command::<Commands>()?;

    // Keyboard router for the (unowned) assistant window: keeps Tab/Esc working
    // and stops REAPER from eating keystrokes meant for the webview composer.
    session.plugin_register_add_accelerator_register(Box::new(AccelHook), AcceleratorPosition::Front)?;

    // Mirror the actions into REAPER's Extensions menu.
    session.reaper().add_extensions_main_menu();
    session.plugin_register_add_hook_custom_menu::<ExtMenu>()?;
    Ok(())
}
