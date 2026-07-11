//! Registers the extension's actions ("Open window" and "Set Anthropic API
//! key") and mirrors them into REAPER's Extensions menu. Both actions share one
//! `HookCommand` that dispatches on the command id. The callbacks are static, so
//! the command ids and the REAPER main window handle live in process globals.

use std::error::Error;
use std::ffi::c_void;
use std::sync::OnceLock;

use reaper_medium::{
    CommandId, Hmenu, HookCommand, HookCustomMenu, MenuHookFlag, OwnedGaccelRegister,
    ReaperSession, ReaperStr,
};

use crate::ai::protocol::UiEvent;
use crate::reaper::prompt;
use crate::{config, ui};

static CMD_OPEN: OnceLock<u32> = OnceLock::new();
static CMD_SETKEY: OnceLock<u32> = OnceLock::new();
static MAIN_HWND: OnceLock<usize> = OnceLock::new();

struct Commands;

impl HookCommand for Commands {
    fn call(command_id: CommandId, _flag: i32) -> bool {
        let id = command_id.get();
        if Some(id) == CMD_OPEN.get().copied() {
            if let Some(h) = MAIN_HWND.get().copied() {
                ui::ffi::show(h as *mut c_void);
            }
            true
        } else if Some(id) == CMD_SETKEY.get().copied() {
            prompt_and_store_key();
            true
        } else {
            false
        }
    }
}

/// Prompt for the Anthropic API key (native input box) and store it.
fn prompt_and_store_key() {
    let caption = if config::has_api_key() {
        "Anthropic API key (a key is already set):"
    } else {
        "Anthropic API key:"
    };
    let Some(input) = prompt::get_user_input("REAPER AI Assistant", caption) else {
        return; // cancelled or API unavailable
    };
    let input = input.trim();
    if input.is_empty() {
        return;
    }
    match config::set_api_key(input) {
        Ok(()) => {
            ui::bridge::emit(UiEvent::Status("Anthropic API key saved.".into()));
            ui::bridge::emit(UiEvent::Announce("Anthropic API key saved.".into()));
        }
        Err(e) => {
            // The key works for this session; only persistence failed.
            ui::bridge::emit(UiEvent::Status(
                "API key set for this session (not persisted).".into(),
            ));
            ui::bridge::emit(UiEvent::Announce(format!(
                "API key set for this session. Could not persist it: {e}"
            )));
        }
    }
}

/// Adds a "REAPER AI Assistant" submenu (holding all our entries) to REAPER's
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
        if let Some(id) = CMD_SETKEY.get().copied() {
            ui::ffi::add_menu_item(submenu, "Set Anthropic API key", id as i32);
        }
        ui::ffi::attach_submenu(parent, submenu, "REAPER AI Assistant");
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
        "REAPER AI Assistant: Open window",
    ))?;

    // Action: set the Anthropic API key.
    let cmd_key = session.plugin_register_add_command_id("RAAI_SetApiKey")?;
    let _ = CMD_SETKEY.set(cmd_key.get());
    session.plugin_register_add_gaccel(OwnedGaccelRegister::without_key_binding(
        cmd_key,
        "REAPER AI Assistant: Set Anthropic API key",
    ))?;

    // One handler dispatches both command ids.
    session.plugin_register_add_hook_command::<Commands>()?;

    // Mirror the actions into REAPER's Extensions menu.
    session.reaper().add_extensions_main_menu();
    session.plugin_register_add_hook_custom_menu::<ExtMenu>()?;
    Ok(())
}
