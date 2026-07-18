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

/// Credential-store service name (the "section" every API key lives under).
const KEYRING_SERVICE: &str = "realackey";
/// The pre-rename service name. `migrate_keyring` copies keys from here into the
/// current service on startup (then removes the old copy) so keys are coherent.
const PRIOR_KEYRING_SERVICE: &str = "reaper-ai-assistant";
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
    /// Perplexity Agent API (OpenAI **Responses** protocol) — multi-provider
    /// models with client-side function tools and built-in web grounding.
    PerplexityAgent,
}

/// What an account is FOR — an axis orthogonal to [`AdapterKind`] (the wire
/// protocol). A transcription account still has a wire protocol (an
/// OpenAI-compatible `/v1/audio/transcriptions` endpoint), but it drives a
/// different capability than chat. Defaults to `Chat` so accounts saved before
/// this field existed keep working.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProviderRole {
    /// Drives the conversation (an [`crate::providers::LlmProvider`]).
    #[default]
    Chat,
    /// Speech-to-text (a [`crate::providers::transcription::TranscriptionProvider`]).
    Transcription,
}

impl AdapterKind {
    /// Whether the endpoint is built into the adapter (no user-supplied base URL):
    /// Anthropic and Perplexity have fixed endpoints; OpenAI-compatible does not.
    pub fn has_fixed_endpoint(self) -> bool {
        !matches!(self, AdapterKind::OpenAiCompatible)
    }

    /// Whether the adapter cannot send without an API key (no keyless/local mode).
    /// OpenAI-compatible allows keyless local servers (Ollama/LM Studio); the
    /// cloud adapters always need a key.
    pub fn requires_key(self) -> bool {
        matches!(
            self,
            AdapterKind::Anthropic | AdapterKind::PerplexityAgent
        )
    }
}

/// One configured provider account. The API key is NOT stored here — it lives in
/// the OS credential store, addressed by `id` (see [`account_for`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Stable internal id (credential-store key, JSON key). Unique.
    pub id: String,
    /// Human-readable name shown in the list.
    pub label: String,
    /// What the account is for (chat vs transcription). Defaults to `Chat` for
    /// configs written before roles existed.
    #[serde(default)]
    pub role: ProviderRole,
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
    /// Whether to request Anthropic extended thinking (reasoning). Anthropic only —
    /// OpenAI-compatible models expose reasoning inherently, with no request flag.
    #[serde(default)]
    pub thinking: bool,
}

impl ProviderConfig {
    /// Whether this account has what it needs to send: Anthropic requires a key;
    /// an OpenAI-compatible account needs an endpoint (key optional — local
    /// servers are keyless).
    pub fn can_send(&self) -> bool {
        if self.kind.requires_key() {
            resolve_key(self).is_some()
        } else {
            self.base_url.is_some()
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Store {
    /// Id of the account that drives the conversation (the CHAT default). Kept as
    /// `default` for backward compatibility with pre-role `providers.json` files.
    #[serde(default)]
    default: Option<String>,
    /// Id of the account used for transcription (speech-to-text), if any.
    #[serde(default)]
    default_transcription: Option<String>,
    #[serde(default)]
    providers: Vec<ProviderConfig>,
    /// "Advanced mode": apply mutating tools WITHOUT asking for confirmation.
    /// Off by default; toggled from the Extensions menu / an action. The
    /// `RAAI_CONFIRM` env var overrides it (see `config::confirmation_required`).
    #[serde(default)]
    auto_approve: bool,
}

impl Store {
    /// The stored default id for a role (borrowed).
    fn role_default(&self, role: ProviderRole) -> &Option<String> {
        match role {
            ProviderRole::Chat => &self.default,
            ProviderRole::Transcription => &self.default_transcription,
        }
    }

    /// Mutable slot for a role's default id.
    fn role_default_mut(&mut self, role: ProviderRole) -> &mut Option<String> {
        match role {
            ProviderRole::Chat => &mut self.default,
            ProviderRole::Transcription => &mut self.default_transcription,
        }
    }
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

/// The configured accounts of one role, in list order (drives a role's tab).
pub fn list_by_role(role: ProviderRole) -> Vec<ProviderConfig> {
    STORE
        .read()
        .unwrap()
        .providers
        .iter()
        .filter(|p| p.role == role)
        .cloned()
        .collect()
}

/// Whether `id` is the default account FOR ITS OWN ROLE (drives the list marker,
/// so a transcription default is marked even though it isn't the chat default).
pub fn is_default(id: &str) -> bool {
    let s = STORE.read().unwrap();
    match s.providers.iter().find(|p| p.id == id) {
        Some(p) => s.role_default(p.role).as_deref() == Some(id),
        None => false,
    }
}

/// The default account's config (the one that drives the conversation — chat).
pub fn active() -> Option<ProviderConfig> {
    active_for(ProviderRole::Chat)
}

/// The default account's config for a role (e.g. the active transcription account).
pub fn active_for(role: ProviderRole) -> Option<ProviderConfig> {
    let s = STORE.read().unwrap();
    let id = s.role_default(role).as_ref()?;
    s.providers.iter().find(|p| &p.id == id).cloned()
}

/// Look up one account by id.
pub fn get(id: &str) -> Option<ProviderConfig> {
    STORE.read().unwrap().providers.iter().find(|p| p.id == id).cloned()
}

/// "Advanced mode": whether the assistant applies mutating tools without asking.
/// Persisted; `RAAI_CONFIRM` overrides it (see `config::confirmation_required`).
pub fn auto_approve() -> bool {
    STORE.read().unwrap().auto_approve
}

/// Flip advanced mode and persist. Returns the new state.
pub fn toggle_auto_approve() -> bool {
    let mut s = STORE.write().unwrap();
    s.auto_approve = !s.auto_approve;
    let now = s.auto_approve;
    let _ = save(&s);
    now
}

/// Set the default account for ITS role (a chat account becomes the chat default,
/// a transcription account the transcription default). Errors if the id is unknown.
pub fn set_default(id: &str) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let role = s
        .providers
        .iter()
        .find(|p| p.id == id)
        .map(|p| p.role)
        .ok_or_else(|| format!("unknown provider id: {id}"))?;
    *s.role_default_mut(role) = Some(id.to_string());
    save(&s)
}

/// Add a new account. Its API keys are set separately via [`set_keys`]. Errors on
/// a duplicate id. Becomes the default for its role if that role has none yet.
pub fn add(cfg: ProviderConfig) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    if s.providers.iter().any(|p| p.id == cfg.id) {
        return Err(format!("provider id already exists: {}", cfg.id));
    }
    if s.role_default(cfg.role).is_none() {
        *s.role_default_mut(cfg.role) = Some(cfg.id.clone());
    }
    s.providers.push(cfg);
    save(&s)
}

/// Update an existing account's config. Its API keys are managed separately via
/// [`set_keys`]. Errors on an unknown id.
pub fn update(cfg: ProviderConfig) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let slot = s
        .providers
        .iter_mut()
        .find(|p| p.id == cfg.id)
        .ok_or_else(|| format!("unknown provider id: {}", cfg.id))?;
    *slot = cfg;
    save(&s)
}

/// Remove an account (and its stored key). If it was its role's default, that
/// default moves to the first remaining account OF THE SAME ROLE (or none).
pub fn remove(id: &str) -> Result<(), String> {
    let mut s = STORE.write().unwrap();
    let Some(role) = s.providers.iter().find(|p| p.id == id).map(|p| p.role) else {
        return Err(format!("unknown provider id: {id}"));
    };
    s.providers.retain(|p| p.id != id);
    delete_key(id);
    if s.role_default(role).as_deref() == Some(id) {
        let next = s.providers.iter().find(|p| p.role == role).map(|p| p.id.clone());
        *s.role_default_mut(role) = next;
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

/// Every usable key for an account id, in priority order — the FAILOVER list the
/// worker rotates through when a key hits its limit. Includes the Anthropic
/// env-var fallback. Empty if the id is unknown or no key is configured.
pub fn keys_for(id: &str) -> Vec<String> {
    get(id).map(|c| resolve_keys(&c)).unwrap_or_default()
}

/// The keys actually held in the credential store for an account id, in order
/// (NO env fallback) — what the settings dialog shows and edits. `Some(vec)` on a
/// successful read (empty = no entry stored); `None` if the store could NOT be
/// read (locked / unavailable), so the caller can avoid treating a read failure as
/// "no keys" and then overwriting real keys with an empty list.
pub fn stored_keys(id: &str) -> Option<Vec<String>> {
    match keyring_entry(&account_for(id)).and_then(|e| e.get_password()) {
        Ok(s) => Some(split_keys(&s)),
        Err(keyring::Error::NoEntry) => Some(Vec::new()),
        Err(_) => None, // locked / unavailable — unknown, not "empty"
    }
}

/// Replace an account's ordered key list (first = tried first). An empty list
/// clears the entry.
pub fn set_keys(id: &str, keys: &[String]) -> Result<(), String> {
    let joined = keys
        .iter()
        .map(|k| k.trim())
        .filter(|k| !k.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        delete_key(id);
        return Ok(());
    }
    keyring_entry(&account_for(id))
        .and_then(|e| e.set_password(&joined))
        .map_err(|e| e.to_string())
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

/// All stored keys for an account, in priority order (first = tried first). The
/// keys live newline-joined in ONE credential-store entry (a pre-multi-key single
/// key is just a one-line list, so no migration is needed). An Anthropic account
/// with no stored keys falls back to the ANTHROPIC_API_KEY env var as one key.
fn resolve_keys(cfg: &ProviderConfig) -> Vec<String> {
    let mut keys = keyring_get(&account_for(&cfg.id))
        .map(|s| split_keys(&s))
        .unwrap_or_default();
    if keys.is_empty() && cfg.kind == AdapterKind::Anthropic {
        if let Some(k) = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            keys.push(k);
        }
    }
    keys
}

/// The first usable key, if any. The many single-key call sites (`can_send`,
/// `key_for`, `active_key`, model fetching) just want one working key.
fn resolve_key(cfg: &ProviderConfig) -> Option<String> {
    resolve_keys(cfg).into_iter().next()
}

/// Split a stored key blob into trimmed, non-empty keys — one per line, in order.
/// API keys never contain a newline, so a line split is a safe list encoding.
fn split_keys(blob: &str) -> Vec<String> {
    blob.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

fn delete_key(id: &str) {
    if let Ok(e) = keyring_entry(&account_for(id)) {
        let _ = e.delete_credential();
    }
}

fn keyring_entry_in(service: &str, account: &str) -> keyring::Result<keyring::Entry> {
    keyring::Entry::new(service, account)
}

fn keyring_entry(account: &str) -> keyring::Result<keyring::Entry> {
    keyring_entry_in(KEYRING_SERVICE, account)
}

fn keyring_get_in(service: &str, account: &str) -> Option<String> {
    keyring_entry_in(service, account)
        .ok()?
        .get_password()
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn keyring_get(account: &str) -> Option<String> {
    keyring_get_in(KEYRING_SERVICE, account)
}

/// Copy each provider's key from the pre-rename credential service into the
/// current one ("realackey"), then remove the old copy — so keys live coherently
/// under one section. Idempotent: an already-present new entry short-circuits.
/// The keyring has no enumerate, so only ids present in the store are migrated
/// (any key for a since-deleted provider stays under the old service, unused).
fn migrate_keyring(providers: &[ProviderConfig]) {
    for p in providers {
        let account = account_for(&p.id);
        if keyring_get_in(KEYRING_SERVICE, &account).is_some() {
            continue; // already under the new service
        }
        let Some(key) = keyring_get_in(PRIOR_KEYRING_SERVICE, &account) else {
            continue; // nothing to migrate for this id
        };
        // Write under the new service and verify before dropping the old copy.
        let moved = keyring_entry_in(KEYRING_SERVICE, &account)
            .and_then(|e| e.set_password(&key))
            .is_ok()
            && keyring_get_in(KEYRING_SERVICE, &account).as_deref() == Some(key.as_str());
        if moved {
            if let Ok(e) = keyring_entry_in(PRIOR_KEYRING_SERVICE, &account) {
                let _ = e.delete_credential();
            }
        }
    }
}

// ---- persistence ------------------------------------------------------------

fn config_base() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
    }
}

/// The config directory (`<REAPER resource path>/ReaLackey`, portable), shared
/// with sibling stores like the prompt-preset registry (`presets.json`).
pub(crate) fn config_dir() -> Option<PathBuf> {
    // Portable: live under REAPER's resource path (like OSARA), so a portable
    // REAPER install carries ReaLackey's config with it. Needs the REAPER API,
    // which is available by the time the store loads (init runs after api::set).
    if let Some(p) = crate::reaper::api::with(|r| {
        r.get_resource_path(|rp| rp.join("ReaLackey").into_std_path_buf())
    }) {
        return Some(p);
    }
    // Fallback if the API isn't ready: the per-user app dir.
    config_base().map(|b| b.join("ReaLackey"))
}

/// One-time copy of the pre-portable config (which lived in the per-user app
/// dir, under the current name or the original "REAPER-AI-Assistant") into the
/// resource-path location, so the move keeps the user's saved providers. Copy,
/// not rename, to survive a cross-volume portable install; the stale old file is
/// harmless and can be removed by hand.
fn migrate_config() {
    let (Some(new_path), Some(base)) = (config_path(), config_base()) else {
        return;
    };
    if new_path.exists() {
        return;
    }
    let old = ["ReaLackey", "REAPER-AI-Assistant"]
        .into_iter()
        .map(|n| base.join(n).join("providers.json"))
        .find(|p| p.exists());
    if let Some(old_path) = old {
        // Don't migrate onto itself (e.g. resource path == app dir on some setups).
        if old_path == new_path {
            return;
        }
        if let Some(parent) = new_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::copy(&old_path, &new_path);
    }
}

fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("providers.json"))
}

fn load_or_seed() -> Store {
    migrate_config(); // bring a pre-portable config into the resource path
    if let Some(path) = config_path() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(mut store) = serde_json::from_str::<Store>(&text) {
                if !store.providers.is_empty() {
                    migrate(&mut store);
                    migrate_keyring(&store.providers);
                    return store;
                }
            }
        }
    }
    // First run: seed a Claude account mirroring the previous fixed settings.
    let store = Store {
        default: Some("anthropic".to_string()),
        default_transcription: None,
        providers: vec![ProviderConfig {
            id: "anthropic".to_string(),
            label: "Claude (Anthropic)".to_string(),
            role: ProviderRole::Chat,
            kind: AdapterKind::Anthropic,
            base_url: None,
            model: default_anthropic_model(),
            max_tokens: 8192,
            max_turns: default_max_turns(),
            supports_images: true,
            supports_audio: false,
            thinking: false,
        }],
        auto_approve: false,
    };
    let _ = save(&store);
    migrate_keyring(&store.providers);
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

#[cfg(test)]
mod tests {
    use super::{split_keys, ProviderRole, Store};

    #[test]
    fn provider_role_defaults_to_chat_for_pre_role_configs() {
        // A providers.json written before roles existed omits the field entirely.
        let json = r#"{ "id":"x", "label":"X", "kind":"OpenAiCompatible",
                        "model":"gpt-4o", "max_tokens":4096 }"#;
        let cfg: super::ProviderConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.role, ProviderRole::Chat);
    }

    #[test]
    fn provider_role_round_trips_and_defaults() {
        assert_eq!(ProviderRole::default(), ProviderRole::Chat);
        for r in [ProviderRole::Chat, ProviderRole::Transcription] {
            let s = serde_json::to_string(&r).unwrap();
            assert_eq!(serde_json::from_str::<ProviderRole>(&s).unwrap(), r);
        }
        // Transcription serializes as its variant name (stable on-disk form).
        assert_eq!(
            serde_json::to_string(&ProviderRole::Transcription).unwrap(),
            "\"Transcription\""
        );
    }

    #[test]
    fn store_routes_each_role_to_its_own_default_slot() {
        let mut s = Store::default();
        *s.role_default_mut(ProviderRole::Chat) = Some("c".into());
        *s.role_default_mut(ProviderRole::Transcription) = Some("t".into());
        assert_eq!(s.role_default(ProviderRole::Chat).as_deref(), Some("c"));
        assert_eq!(
            s.role_default(ProviderRole::Transcription).as_deref(),
            Some("t")
        );
        // The two slots are independent (setting one never disturbs the other).
        assert_eq!(s.default.as_deref(), Some("c"));
        assert_eq!(s.default_transcription.as_deref(), Some("t"));
    }

    #[test]
    fn split_keys_parses_ordered_list_and_is_backward_compatible() {
        // A pre-multi-key single stored key (no newline) -> a one-element list.
        assert_eq!(split_keys("sk-single"), vec!["sk-single".to_string()]);
        // Multiple keys keep their order.
        assert_eq!(
            split_keys("sk-a\nsk-b\nsk-c"),
            vec!["sk-a".to_string(), "sk-b".to_string(), "sk-c".to_string()]
        );
        // Blank lines and surrounding whitespace are dropped/trimmed.
        assert_eq!(
            split_keys("  sk-a  \n\n  sk-b\n"),
            vec!["sk-a".to_string(), "sk-b".to_string()]
        );
        // An empty / whitespace-only blob is no keys.
        assert!(split_keys("").is_empty());
        assert!(split_keys("   \n  \n").is_empty());
    }
}
