//! Prompt-preset dialog logic. Two native dialogs live in the C++ shim:
//!   * the preset *list* (`ui_show_presets`) — this module supplies its name
//!     list and row actions (add / edit / delete);
//!   * the preset *edit* sub-dialog (`ui_show_preset_edit`) — a Name field and a
//!     multiline Prompt-text field, driven through the `edit_dialog_*` callbacks.
//!
//! Everything runs on the REAPER main thread (the dialogs are modal and open
//! nested modal boxes, all main-thread only), so the in-flight edit session is
//! kept in a thread-local. Mirrors [`crate::ui::providers_ui`] with a simpler
//! model (no keys, no reorder).

use std::cell::RefCell;

use crate::prompts::registry::{self, PromptPreset};
use crate::reaper::osara;
use crate::ui;

/// Newline-separated preset names for the listbox, in registry order.
pub fn list_text() -> String {
    registry::list()
        .iter()
        .map(|p| display_name(&p.name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A one-line, non-empty display label. The listbox and the popup menu both split
/// on '\n' and drop empty lines, which would shift every row off its preset — so
/// collapse newlines and never return an empty string.
pub fn display_name(name: &str) -> String {
    let s = name.replace(['\r', '\n'], " ");
    let t = s.trim();
    if t.is_empty() {
        "(unnamed)".to_string()
    } else {
        t.to_string()
    }
}

/// Run a row action (0=add, 1=edit, 2=delete). Returns true if the preset list
/// changed (the dialog then repopulates its listbox).
pub fn on_action(action: i32, index: i32) -> bool {
    match action {
        0 => add_preset(),
        1 => edit_preset(index),
        2 => delete_preset(index),
        _ => false,
    }
}

// ---- the in-flight edit/add session -----------------------------------------

struct EditSession {
    /// None = adding a new preset; Some(id) = editing that existing preset.
    id: Option<String>,
    name: String,
    body: String,
    /// Set true once the preset was saved (so the list dialog repopulates).
    changed: bool,
}

thread_local! {
    static SESSION: RefCell<Option<EditSession>> = const { RefCell::new(None) };
}

// ---- actions ----------------------------------------------------------------

fn add_preset() -> bool {
    run_edit_dialog(EditSession {
        id: None,
        name: String::new(),
        body: String::new(),
        changed: false,
    })
}

fn edit_preset(index: i32) -> bool {
    let Some(p) = preset_at(index) else {
        return false;
    };
    run_edit_dialog(EditSession {
        id: Some(p.id),
        name: p.name,
        body: p.body,
        changed: false,
    })
}

/// Show the modal edit dialog for `session`; returns whether it changed the store
/// (so the list dialog repopulates).
fn run_edit_dialog(session: EditSession) -> bool {
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    let _ = ui::ffi::show_preset_edit(); // modal; fires the edit_dialog_* callbacks
    SESSION.with(|s| s.borrow_mut().take().map(|x| x.changed).unwrap_or(false))
}

fn delete_preset(index: i32) -> bool {
    let Some(p) = preset_at(index) else {
        return false;
    };
    let confirmed = ui::ffi::message_box(
        "Delete preset",
        &format!(
            "Delete the preset \"{}\"? This cannot be undone.",
            display_name(&p.name)
        ),
        true,
    );
    if !confirmed {
        return false;
    }
    match registry::remove(&p.id) {
        Ok(()) => {
            osara::announce("Preset deleted.");
            true
        }
        Err(e) => {
            ui::ffi::message_box("Delete preset", &format!("Could not delete preset: {e}"), false);
            false
        }
    }
}

// ---- edit dialog callbacks (fired by the C++ modal loop) --------------------

/// Prefill the edit dialog's Name + Prompt-text fields (WM_INITDIALOG).
pub fn edit_dialog_init() {
    SESSION.with(|s| {
        let b = s.borrow();
        let Some(sess) = b.as_ref() else {
            return;
        };
        ui::ffi::preset_set_text(ui::ffi::PRE_NAME, &sess.name);
        ui::ffi::preset_set_text(ui::ffi::PRE_BODY, &sess.body);
    });
}

/// OK clicked: validate, save (add or update), and report whether the dialog
/// should close (true = close; false = keep open so the user can fix an error).
pub fn edit_dialog_ok() -> bool {
    let name = ui::ffi::preset_get_text(ui::ffi::PRE_NAME).trim().to_string();
    // The shim already normalizes the multiline body to LF-only line endings.
    let body = ui::ffi::preset_get_text(ui::ffi::PRE_BODY);
    if name.is_empty() {
        // (The OK button is disabled while the name is empty; this guards the
        // Enter-in-field path and any race.)
        ui::ffi::message_box("Prompt preset", "Give the preset a name.", false);
        return false;
    }
    if body.trim().is_empty() {
        ui::ffi::message_box(
            "Prompt preset",
            "Enter the prompt text this preset should insert.",
            false,
        );
        return false;
    }

    let id = SESSION.with(|s| s.borrow().as_ref().and_then(|x| x.id.clone()));
    let is_edit = id.is_some();
    let result = match &id {
        Some(id) => registry::update(id, name.clone(), body),
        None => registry::add(name.clone(), body).map(|_| ()),
    };
    match result {
        Ok(()) => {
            osara::announce(&format!(
                "Preset {name} {}.",
                if is_edit { "updated" } else { "saved" }
            ));
            SESSION.with(|s| {
                if let Some(x) = s.borrow_mut().as_mut() {
                    x.changed = true;
                }
            });
            true // close
        }
        Err(e) => {
            ui::ffi::message_box("Prompt preset", &format!("Could not save preset: {e}"), false);
            false // keep the dialog open so the user can correct it
        }
    }
}

// ---- helpers ----------------------------------------------------------------

/// The preset at a listbox row (registry order), if the index is valid.
fn preset_at(index: i32) -> Option<PromptPreset> {
    let i = usize::try_from(index).ok()?;
    registry::list().into_iter().nth(i)
}
