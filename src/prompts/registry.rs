//! Prompt-preset registry: the user's saved reusable prompts.
//!
//! Each preset is a name + a plain-text body. They persist to `presets.json` in
//! the same config dir as `providers.json` (portable, under REAPER's resource
//! path — the dir resolution is reused from [`crate::providers::registry`]).
//! Global, not per-project: a preset is a user-level asset wanted in every
//! project. Ordering is the `Vec` position; there is no reorder UI in v1, so new
//! presets simply append.

use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use serde::{Deserialize, Serialize};

/// One saved prompt preset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptPreset {
    /// Stable id (JSON key / selection key). Unique within the store.
    pub id: String,
    /// Short title shown in the manager list and the insert picker.
    pub name: String,
    /// The prompt text inserted into the composer.
    pub body: String,
    /// Reserved for a future optional description; unwired in v1 (no edit control,
    /// never displayed). Present so it can be added later without a format break.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default)]
    presets: Vec<PromptPreset>,
}

static STORE: LazyLock<RwLock<Store>> = LazyLock::new(|| RwLock::new(load()));

// ---- public API -------------------------------------------------------------

/// Force the store to load now (first run = empty; no file is written until the
/// user adds a preset). Call once at startup on the main thread, after the REAPER
/// handle is published (the config dir needs it).
pub fn init() {
    LazyLock::force(&STORE);
}

/// All presets, in list order.
pub fn list() -> Vec<PromptPreset> {
    STORE.read().unwrap().presets.clone()
}

/// Add a new preset (appended). Returns its generated id.
pub fn add(name: String, body: String) -> Result<String, String> {
    let mut s = STORE.write().unwrap();
    let id = unique_id(&s.presets, &name);
    s.presets.push(PromptPreset {
        id: id.clone(),
        name,
        body,
        description: None,
    });
    save(&s)?;
    Ok(id)
}

/// Update an existing preset's name + body. Errors on an unknown id.
pub fn update(id: &str, name: String, body: String) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let slot = s
        .presets
        .iter_mut()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("unknown preset id: {id}"))?;
    slot.name = name;
    slot.body = body;
    save(&s)
}

/// Remove a preset. Errors on an unknown id.
pub fn remove(id: &str) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let before = s.presets.len();
    s.presets.retain(|p| p.id != id);
    if s.presets.len() == before {
        return Err(format!("unknown preset id: {id}"));
    }
    save(&s)
}

// ---- helpers ----------------------------------------------------------------

/// A slug-ish stable id derived from the name, uniquified against existing ids.
/// Falls back to "preset" when the name has no id-safe characters.
fn unique_id(existing: &[PromptPreset], name: &str) -> String {
    let base = slugify(name);
    let base = if base.is_empty() {
        "preset".to_string()
    } else {
        base
    };
    if !existing.iter().any(|p| p.id == base) {
        return base;
    }
    let mut n = 2;
    loop {
        let cand = format!("{base}-{n}");
        if !existing.iter().any(|p| p.id == cand) {
            return cand;
        }
        n += 1;
    }
}

/// Lowercase ASCII-alphanumeric run, hyphen-separated (non-alphanumerics collapse
/// to single hyphens, trimmed). Non-ASCII is dropped, so a name of only non-ASCII
/// yields an empty string (the caller then falls back to "preset").
fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut pending_sep = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('-');
            }
            pending_sep = false;
            out.push(c.to_ascii_lowercase());
        } else {
            pending_sep = true;
        }
    }
    out
}

// ---- persistence ------------------------------------------------------------

fn config_path() -> Option<PathBuf> {
    crate::providers::registry::config_dir().map(|d| d.join("presets.json"))
}

fn load() -> Store {
    if let Some(path) = config_path() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(store) = serde_json::from_str::<Store>(&text) {
                return store;
            }
        }
    }
    Store::default()
}

fn save(store: &Store) -> Result<(), String> {
    let path = config_path().ok_or("no config directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{slugify, unique_id, PromptPreset};

    fn preset(id: &str) -> PromptPreset {
        PromptPreset {
            id: id.to_string(),
            name: id.to_string(),
            body: String::new(),
            description: None,
        }
    }

    #[test]
    fn slugify_makes_id_safe_slugs() {
        assert_eq!(slugify("Master for streaming"), "master-for-streaming");
        assert_eq!(slugify("  Trim & Fade!! "), "trim-fade");
        assert_eq!(slugify("EQ v2"), "eq-v2");
        // Non-ASCII collapses away (caller falls back to "preset").
        assert_eq!(slugify("\u{4e2d}\u{6587}"), "");
    }

    #[test]
    fn unique_id_disambiguates_collisions() {
        let existing = vec![preset("master"), preset("master-2")];
        // Fresh name -> its slug.
        assert_eq!(unique_id(&existing, "Trim"), "trim");
        // Colliding name -> next free suffix (skips the taken -2).
        assert_eq!(unique_id(&existing, "Master"), "master-3");
        // Empty slug -> "preset".
        assert_eq!(unique_id(&[], "\u{4e2d}"), "preset");
    }
}
