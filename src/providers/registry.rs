//! Provider registry: the user's configured LLM accounts (design §kap-providers).
//!
//! Each entry is a configured *account* — adapter kind, endpoint, model, token
//! limit — with its API key held separately in the OS credential store (never in
//! the JSON, never in the WebView). The non-secret list plus the chosen default
//! account persist to `providers.json` in the config dir. This is the config
//! layer (Phase 5, M1); adapters are built from it in later milestones.

use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use serde::{Deserialize, Serialize};

/// Credential-store service name (shared with the legacy single-key setup).
const KEYRING_SERVICE: &str = "reaper-ai-assistant";
/// Legacy account name for the single Anthropic key (pre-multi-provider). The
/// seeded `anthropic` account reuses it so existing keys keep working.
const LEGACY_KEY_ACCOUNT: &str = "anthropic-api-key";

/// Which adapter drives an account.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterKind {
    /// Native Anthropic Messages API.
    Anthropic,
    /// Shared OpenAI-compatible `/chat/completions` endpoint (OpenAI, Groq,
    /// Gemini-compat, DeepSeek, xAI, OpenRouter, Ollama/LM Studio, custom).
    OpenAiCompatible,
}

/// One configured provider account. The API key is NOT stored here — it lives in
/// the OS credential store, addressed by `id` (see [`account_for`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Stable internal id (credential-store key, JSON key). Unique.
    pub id: String,
    /// Human-readable name shown in the list.
    pub label: String,
    pub kind: AdapterKind,
    /// Endpoint base URL — OpenAI-compatible accounts only (preset or custom).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub model: String,
    pub max_tokens: u32,
    /// Max agentic tool-call turns per user message (bounded loop). Per-provider
    /// so a cheap/local account can iterate more than a metered one. Defaulted for
    /// configs saved before this field existed.
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    /// Whether the chosen model accepts image input (vision). Per-MODEL, not
    /// per-provider (gpt-4o yes, plain Llama no), so it lives here rather than as
    /// an adapter constant. Anthropic accounts are always vision-capable; this is
    /// only consulted for OpenAI-compatible accounts (see `build_provider`).
    #[serde(default)]
    pub supports_images: bool,
    /// Whether the chosen model accepts audio input ("listening"). Per-MODEL and
    /// rare (OpenAI gpt-audio, Gemini, OpenRouter audio models); false for
    /// Anthropic/Ollama/Groq/xAI. Gates the `listen_to_audio` tool.
    #[serde(default)]
    pub supports_audio: bool,
}

impl ProviderConfig {
    /// Whether this account has what it needs to send: Anthropic requires a key;
    /// an OpenAI-compatible account needs an endpoint (key optional — local
    /// servers are keyless).
    pub fn can_send(&self) -> bool {
        match self.kind {
            AdapterKind::Anthropic => resolve_key(self).is_some(),
            AdapterKind::OpenAiCompatible => self.base_url.is_some(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Store {
    /// Id of the account that drives the conversation.
    #[serde(default)]
    default: Option<String>,
    #[serde(default)]
    providers: Vec<ProviderConfig>,
}

static STORE: LazyLock<RwLock<Store>> = LazyLock::new(|| RwLock::new(load_or_seed()));

// ---- public API -------------------------------------------------------------

/// Force the store to load (or seed) now. Call once at startup so the first-run
/// seed + legacy-key migration happen predictably on the main thread.
pub fn init() {
    LazyLock::force(&STORE);
}

/// The configured accounts, in list order (drives the provider dialog).
pub fn list() -> Vec<ProviderConfig> {
    STORE.read().unwrap().providers.clone()
}

/// The default account id, if any.
pub fn default_id() -> Option<String> {
    STORE.read().unwrap().default.clone()
}

/// The default account's config (the one that drives the conversation).
pub fn active() -> Option<ProviderConfig> {
    let s = STORE.read().unwrap();
    let id = s.default.as_ref()?;
    s.providers.iter().find(|p| &p.id == id).cloned()
}

/// Look up one account by id.
pub fn get(id: &str) -> Option<ProviderConfig> {
    STORE.read().unwrap().providers.iter().find(|p| p.id == id).cloned()
}

/// Set the default account. Errors if the id is unknown.
pub fn set_default(id: &str) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    if !s.providers.iter().any(|p| p.id == id) {
        return Err(format!("unknown provider id: {id}"));
    }
    s.default = Some(id.to_string());
    save(&s)
}

/// Add a new account (optionally with a key). Errors on a duplicate id. Becomes
/// the default if it is the first account.
pub fn add(cfg: ProviderConfig, key: Option<&str>) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    if s.providers.iter().any(|p| p.id == cfg.id) {
        return Err(format!("provider id already exists: {}", cfg.id));
    }
    if let Some(k) = key {
        set_key(&cfg.id, k)?;
    }
    if s.default.is_none() {
        s.default = Some(cfg.id.clone());
    }
    s.providers.push(cfg);
    save(&s)
}

/// Update an existing account's config. `key_change`: `None` = leave the key
/// as-is, `Some(None)` = clear it, `Some(Some(k))` = set it.
pub fn update(cfg: ProviderConfig, key_change: Option<Option<&str>>) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let slot = s
        .providers
        .iter_mut()
        .find(|p| p.id == cfg.id)
        .ok_or_else(|| format!("unknown provider id: {}", cfg.id))?;
    *slot = cfg.clone();
    match key_change {
        None => {}
        Some(Some(k)) => set_key(&cfg.id, k)?,
        Some(None) => delete_key(&cfg.id),
    }
    save(&s)
}

/// Remove an account (and its stored key). If it was the default, the default
/// moves to the first remaining account (or none).
pub fn remove(id: &str) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let before = s.providers.len();
    s.providers.retain(|p| p.id != id);
    if s.providers.len() == before {
        return Err(format!("unknown provider id: {id}"));
    }
    delete_key(id);
    if s.default.as_deref() == Some(id) {
        s.default = s.providers.first().map(|p| p.id.clone());
    }
    save(&s)
}

/// The resolved key for an account id (credential store, with an env fallback
/// for Anthropic). `None` if not set. Used by the adapter factory + M4 dialog.
pub fn key_for(id: &str) -> Option<String> {
    get(id).and_then(|c| resolve_key(&c))
}

/// The resolved key of the default account.
pub fn active_key() -> Option<String> {
    active().and_then(|c| resolve_key(&c))
}

/// Set (or, with an empty string, clear) the default account's key. Used by the
/// legacy "Set API key" action until the provider dialog (M4) replaces it.
pub fn set_active_key(key: &str) -> Result<(), String> {
    let id = default_id().ok_or("no active provider configured")?;
    set_key(&id, key)
}

/// Whether the default account can currently send (key/endpoint present).
pub fn active_can_send() -> bool {
    active().map(|c| c.can_send()).unwrap_or(false)
}

// ---- key resolution (credential store) --------------------------------------

/// Credential-store account name for a provider id. The seeded `anthropic`
/// account reuses the legacy name so pre-existing keys keep working.
fn account_for(id: &str) -> String {
    if id == "anthropic" {
        LEGACY_KEY_ACCOUNT.to_string()
    } else {
        format!("apikey:{id}")
    }
}

/// Stored key (credential store), plus an env fallback for Anthropic.
fn resolve_key(cfg: &ProviderConfig) -> Option<String> {
    if let Some(k) = keyring_get(&account_for(&cfg.id)) {
        return Some(k);
    }
    if cfg.kind == AdapterKind::Anthropic {
        return std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty());
    }
    None
}

fn set_key(id: &str, key: &str) -> Result<(), String> {
    let key = key.trim();
    if key.is_empty() {
        delete_key(id);
        return Ok(());
    }
    keyring_entry(&account_for(id))
        .and_then(|e| e.set_password(key))
        .map_err(|e| e.to_string())
}

fn delete_key(id: &str) {
    if let Ok(e) = keyring_entry(&account_for(id)) {
        let _ = e.delete_credential();
    }
}

fn keyring_entry(account: &str) -> keyring::Result<keyring::Entry> {
    keyring::Entry::new(KEYRING_SERVICE, account)
}

fn keyring_get(account: &str) -> Option<String> {
    keyring_entry(account)
        .ok()?
        .get_password()
        .ok()
        .filter(|s| !s.trim().is_empty())
}

// ---- persistence ------------------------------------------------------------

fn config_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"));
    base.map(|b| b.join("REAPER-AI-Assistant"))
}

fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("providers.json"))
}

fn load_or_seed() -> Store {
    if let Some(path) = config_path() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(mut store) = serde_json::from_str::<Store>(&text) {
                if !store.providers.is_empty() {
                    migrate(&mut store);
                    return store;
                }
            }
        }
    }
    // First run: seed a Claude account mirroring the previous fixed settings.
    let store = Store {
        default: Some("anthropic".to_string()),
        providers: vec![ProviderConfig {
            id: "anthropic".to_string(),
            label: "Claude (Anthropic)".to_string(),
            kind: AdapterKind::Anthropic,
            base_url: None,
            model: default_anthropic_model(),
            max_tokens: 8192,
            max_turns: default_max_turns(),
            supports_images: true,
            supports_audio: false,
        }],
    };
    let _ = save(&store);
    store
}

/// In-memory migration for configs written before a field existed. Anthropic
/// (Claude) models are all vision-capable, so ensure the flag is set for them
/// even when an older `providers.json` omitted it (`serde` would default false).
fn migrate(store: &mut Store) {
    for p in &mut store.providers {
        if p.kind == AdapterKind::Anthropic {
            p.supports_images = true;
        }
    }
}

fn save(store: &Store) -> Result<(), String> {
    let path = config_path().ok_or("no config directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

/// Default Claude model id (env override kept for parity with the old config).
fn default_anthropic_model() -> String {
    std::env::var("RAAI_MODEL").unwrap_or_else(|_| "claude-opus-4-8".to_string())
}

/// Default max agentic turns for a new/legacy account (see `config::max_turns`).
fn default_max_turns() -> u32 {
    25
}
