//! Runtime configuration.
//!
//! The Anthropic API key is resolved from, in order: an in-memory cache (set via
//! the UI this session), the OS credential store (Windows Credential Manager via
//! `keyring`), and the `ANTHROPIC_API_KEY` environment variable. Storing a key
//! persists it to the credential store (design §kap-sec).

use std::sync::RwLock;

static CACHED_KEY: RwLock<Option<String>> = RwLock::new(None);

const KEYRING_SERVICE: &str = "reaper-ai-assistant";
const KEYRING_ACCOUNT: &str = "anthropic-api-key";

/// Load a persisted/env key into the in-memory cache at startup.
pub fn init_key_cache() {
    if let Some(k) = keyring_get().or_else(env_key) {
        *CACHED_KEY.write().unwrap() = Some(k);
    }
}

/// The Anthropic API key, if configured.
pub fn api_key() -> Option<String> {
    if let Some(k) = CACHED_KEY.read().unwrap().clone() {
        return Some(k);
    }
    env_key()
}

pub fn has_api_key() -> bool {
    api_key().is_some()
}

/// Set (or, with an empty string, clear) the API key. Always updates the
/// in-memory cache so it takes effect immediately; the returned `Result`
/// reflects only whether persistence to the credential store succeeded.
pub fn set_api_key(key: &str) -> Result<(), String> {
    let key = key.trim().to_string();
    *CACHED_KEY.write().unwrap() = if key.is_empty() { None } else { Some(key.clone()) };
    if key.is_empty() {
        keyring_delete()
    } else {
        keyring_set(&key)
    }
}

/// Default model. Dateless-but-pinned snapshot id (do not append a date suffix).
pub fn default_model() -> String {
    std::env::var("RAAI_MODEL").unwrap_or_else(|_| "claude-opus-4-8".to_string())
}

/// System prompt. Establishes the role and how to use the read tools
/// (design §kap-llm). Grows as more capabilities land.
pub fn system_prompt() -> String {
    "You are an AI assistant embedded in the REAPER digital audio workstation. \
     You can inspect the user's project through tools: the project summary, the \
     track list, a track's FX chain and its parameters, selected media items, an \
     item's take FX and their parameters, the list of installed plugins, and the \
     currently focused FX window. When a question depends on the current project \
     state, call the appropriate tool instead of guessing, and chain tools when \
     needed (e.g. resolve the focused FX, then read its parameters). When \
     explaining parameters, prefer the human-readable display value over the raw \
     normalized 0..1 value. Answer concisely. You cannot modify the project yet — \
     tools for making changes arrive in a later phase."
        .to_string()
}

// ---- sources ----------------------------------------------------------------

fn env_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn keyring_entry() -> keyring::Result<keyring::Entry> {
    keyring::Entry::new(KEYRING_SERVICE, KEYRING_ACCOUNT)
}

fn keyring_get() -> Option<String> {
    keyring_entry()
        .ok()?
        .get_password()
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn keyring_set(key: &str) -> Result<(), String> {
    keyring_entry()
        .and_then(|e| e.set_password(key))
        .map_err(|e| e.to_string())
}

fn keyring_delete() -> Result<(), String> {
    match keyring_entry().and_then(|e| e.delete_credential()) {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
