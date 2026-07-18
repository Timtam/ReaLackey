//! Provider management dialog logic (Phase 5, M4/M5).
//!
//! Two native dialogs live in the C++ shim:
//!   * the provider *list* (`ui_show_providers`) — this module supplies its label
//!     list and row actions (add / edit / delete / set-default);
//!   * the provider *settings* dialog (`ui_show_provider_edit`) — a real dialog
//!     with a Model field next to a "Fetch models" button; this module drives it
//!     through the `edit_dialog_*` callbacks below.
//!
//! Everything here runs on the REAPER main thread (the dialogs are modal and open
//! nested modal menus / boxes, all main-thread only), so the in-flight edit
//! session is kept in a thread-local.

use std::cell::RefCell;

use crate::providers::models_api;
use crate::providers::registry::{self, AdapterKind, ProviderConfig, ProviderRole};
use crate::reaper::osara;
use crate::ui;

/// A ready-made account the "Add" picker offers. Model ids are sensible defaults
/// the user can edit; endpoints are the providers' OpenAI-compatible bases.
struct Preset {
    /// Label shown in the Add popup menu.
    menu: &'static str,
    /// Stable id base (uniquified if it collides).
    id: &'static str,
    /// Default provider label.
    label: &'static str,
    kind: AdapterKind,
    /// OpenAI-compatible endpoint (empty for Anthropic / a blank custom slot).
    base_url: &'static str,
    model: &'static str,
    max_tokens: u32,
    /// Default vision support for the preset's default model (overridable, and
    /// re-derived when the user picks a model via "Fetch models…").
    vision: bool,
}

/// The shipped presets. Claude is included so it can be re-added if deleted; the
/// rest are OpenAI-compatible endpoints plus a blank custom slot.
const PRESETS: &[Preset] = &[
    Preset {
        menu: "Claude (Anthropic)",
        id: "anthropic",
        label: "Claude (Anthropic)",
        kind: AdapterKind::Anthropic,
        base_url: "",
        model: "claude-opus-4-8",
        max_tokens: 8192,
        vision: true,
    },
    Preset {
        menu: "OpenAI",
        id: "openai",
        label: "OpenAI",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://api.openai.com/v1",
        model: "gpt-4o",
        max_tokens: 4096,
        vision: true,
    },
    Preset {
        menu: "Groq",
        id: "groq",
        label: "Groq",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://api.groq.com/openai/v1",
        model: "llama-3.3-70b-versatile",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "OpenRouter",
        id: "openrouter",
        label: "OpenRouter",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://openrouter.ai/api/v1",
        model: "openai/gpt-4o",
        max_tokens: 4096,
        vision: true,
    },
    Preset {
        menu: "DeepSeek",
        id: "deepseek",
        label: "DeepSeek",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://api.deepseek.com",
        model: "deepseek-chat",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "xAI (Grok)",
        id: "xai",
        label: "xAI (Grok)",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://api.x.ai/v1",
        model: "grok-2-latest",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "Gemini (OpenAI-compatible)",
        id: "gemini",
        label: "Gemini",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        model: "gemini-3.5-flash",
        max_tokens: 4096,
        vision: true,
    },
    // Perplexity's Agent API (OpenAI Responses protocol): multi-provider models
    // with client-side function tools + built-in web grounding. Fixed endpoint,
    // needs a key. Model ids span providers (openai/…, anthropic/…, sonar-…);
    // pick a strong agentic model — the seed is just a starting point.
    Preset {
        menu: "Perplexity (Agent API, web-grounded)",
        id: "perplexity",
        label: "Perplexity",
        kind: AdapterKind::PerplexityAgent,
        base_url: "",
        model: "openai/gpt-5.1",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "Ollama (local)",
        id: "ollama",
        label: "Ollama (local)",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "http://localhost:11434/v1",
        model: "llama3.2",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "LM Studio (local)",
        id: "lmstudio",
        label: "LM Studio (local)",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "http://localhost:1234/v1",
        model: "local-model",
        max_tokens: 4096,
        vision: false,
    },
    // oMLX: a native MLX inference server for Apple Silicon (continuous batching,
    // SSD KV cache) — faster than Ollama on a Mac. Exposes an OpenAI-compatible
    // endpoint on :8000, so it rides the same adapter. Model ids come from the
    // user's local model directory ("Fetch models…" lists them via /v1/models).
    Preset {
        menu: "oMLX (local, Apple Silicon)",
        id: "omlx",
        label: "oMLX (local)",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "http://localhost:8000/v1",
        model: "qwen3-8b",
        max_tokens: 4096,
        vision: false,
    },
    Preset {
        menu: "Custom endpoint\u{2026}",
        id: "custom",
        label: "Custom",
        kind: AdapterKind::OpenAiCompatible,
        base_url: "",
        model: "",
        max_tokens: 4096,
        vision: false,
    },
];

/// Newline-separated labels for the listbox, in registry order. The default
/// account is marked with a leading `*`; accounts that can't yet send show why.
pub fn list_text() -> String {
    let default = registry::default_id();
    registry::list()
        .iter()
        .map(|p| {
            let is_default = default.as_deref() == Some(p.id.as_str());
            let mark = if is_default { "* " } else { "   " };
            let status = if p.can_send() {
                ""
            } else if p.kind.requires_key() {
                "  \u{2014} needs API key"
            } else if p.base_url.is_none() {
                "  \u{2014} needs endpoint URL"
            } else {
                ""
            };
            format!("{mark}{}  ({}){}", p.label, p.model, status)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run a row action (0=add, 1=edit, 2=delete, 3=set-default). Returns true if
/// the provider list changed (the dialog then repopulates its listbox).
pub fn on_action(action: i32, index: i32) -> bool {
    match action {
        0 => add_provider(),
        1 => edit_provider(index),
        2 => delete_provider(index),
        3 => set_default(index),
        _ => false,
    }
}

// ---- the in-flight edit/add session -----------------------------------------

/// State for one run of the settings dialog. Held in a thread-local because the
/// dialog's init/fetch/ok callbacks (fired by the C++ modal loop on the main
/// thread) need it, and it must outlive each individual callback.
struct ProvSession {
    /// Adding a new account vs. editing an existing one.
    is_new: bool,
    /// Existing id (edit) or a pre-generated unique id (add).
    id: String,
    kind: AdapterKind,
    // Prefill values for the fields (used by `edit_dialog_init`).
    label: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    max_turns: u32,
    vision: bool,
    /// Whether the account's model accepts audio input ("Supports audio" checkbox).
    /// Seeded from the id heuristic (see add/fetch); user-overridable.
    audio: bool,
    /// Whether Anthropic extended thinking is on ("Extended thinking" checkbox,
    /// Anthropic accounts only).
    thinking: bool,
    /// The working list of API keys, in priority order (top tried first). Edited
    /// live via the Add / Delete / Move up / Move down buttons; saved on OK.
    keys: Vec<String>,
    /// Whether `keys` was read back from the credential store successfully. False
    /// means the store couldn't be read (locked/unavailable) — so we must NOT
    /// treat the empty `keys` as authoritative and overwrite the real stored keys.
    keys_loaded: bool,
    /// Whether the user changed the key list in this dialog session. We only
    /// rewrite the credential store on OK when this is set, so an unrelated edit
    /// (or a failed read) never clobbers the stored keys.
    keys_dirty: bool,
    /// Anthropic account whose only credential is the ANTHROPIC_API_KEY env var
    /// (no stored keys) — shown as a hint so the list isn't misread as unconfigured.
    env_active: bool,
    /// Set true once the account was successfully saved (so the list repopulates).
    changed: bool,
}

thread_local! {
    static SESSION: RefCell<Option<ProvSession>> = const { RefCell::new(None) };
}

// ---- actions ----------------------------------------------------------------

fn add_provider() -> bool {
    let labels: Vec<&str> = PRESETS.iter().map(|p| p.menu).collect();
    let choice = ui::ffi::popup_menu(&labels);
    if choice == 0 || choice > PRESETS.len() {
        return false; // cancelled
    }
    let preset = &PRESETS[choice - 1];
    run_settings_dialog(ProvSession {
        is_new: true,
        id: unique_id(preset.id),
        kind: preset.kind,
        label: preset.label.to_string(),
        base_url: preset.base_url.to_string(),
        model: preset.model.to_string(),
        max_tokens: preset.max_tokens,
        max_turns: 25,
        vision: preset.vision,
        audio: infer_audio(preset.model),
        thinking: false,
        keys: Vec::new(),
        keys_loaded: true, // a brand-new account genuinely has no stored keys yet
        keys_dirty: false,
        env_active: false,
        changed: false,
    })
}

fn edit_provider(index: i32) -> bool {
    let Some(cfg) = provider_at(index) else {
        return false;
    };
    // A read failure (locked store) must be distinguished from "no keys", or an
    // OK after an unrelated edit would overwrite the real keys with an empty list.
    let (keys, keys_loaded) = match registry::stored_keys(&cfg.id) {
        Some(k) => (k, true),
        None => (Vec::new(), false),
    };
    // Anthropic account running purely off the ANTHROPIC_API_KEY env var (nothing
    // stored) — surface that so an empty list doesn't read as "unconfigured".
    let env_active = keys_loaded
        && keys.is_empty()
        && cfg.kind == AdapterKind::Anthropic
        && !registry::keys_for(&cfg.id).is_empty();
    run_settings_dialog(ProvSession {
        is_new: false,
        id: cfg.id.clone(),
        kind: cfg.kind,
        label: cfg.label.clone(),
        base_url: cfg.base_url.clone().unwrap_or_default(),
        model: cfg.model.clone(),
        max_tokens: cfg.max_tokens,
        max_turns: cfg.max_turns,
        vision: cfg.supports_images,
        audio: cfg.supports_audio,
        thinking: cfg.thinking,
        keys,
        keys_loaded,
        keys_dirty: false,
        env_active,
        changed: false,
    })
}

/// Show the modal settings dialog for `session`; returns whether it changed the
/// registry (so the list dialog repopulates).
fn run_settings_dialog(session: ProvSession) -> bool {
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    let _ = ui::ffi::show_provider_edit(); // modal; fires the edit_dialog_* callbacks
    SESSION.with(|s| s.borrow_mut().take().map(|x| x.changed).unwrap_or(false))
}

fn delete_provider(index: i32) -> bool {
    let Some(cfg) = provider_at(index) else {
        return false;
    };
    if registry::list().len() <= 1 {
        ui::ffi::message_box(
            "Delete provider",
            "This is the only provider. Add another before deleting it.",
            false,
        );
        return false;
    }
    let confirmed = ui::ffi::message_box(
        "Delete provider",
        &format!(
            "Delete provider \"{}\"? Its stored API key will be removed.",
            cfg.label
        ),
        true,
    );
    if !confirmed {
        return false;
    }
    match registry::remove(&cfg.id) {
        Ok(()) => {
            osara::announce("Provider deleted.");
            true
        }
        Err(e) => {
            ui::ffi::message_box("Delete provider", &format!("Could not delete provider: {e}"), false);
            false
        }
    }
}

fn set_default(index: i32) -> bool {
    let Some(cfg) = provider_at(index) else {
        return false;
    };
    if registry::default_id().as_deref() == Some(cfg.id.as_str()) {
        osara::announce(&format!("{} is already the default.", cfg.label));
        return false; // no change
    }
    match registry::set_default(&cfg.id) {
        Ok(()) => {
            osara::announce(&format!("{} is now the default provider.", cfg.label));
            true
        }
        Err(e) => {
            ui::ffi::message_box("Set default", &format!("Could not set default: {e}"), false);
            false
        }
    }
}

// ---- settings dialog callbacks (fired by the C++ modal loop) -----------------

/// Prefill the settings dialog fields from the session (WM_INITDIALOG).
pub fn edit_dialog_init() {
    SESSION.with(|s| {
        let b = s.borrow();
        let Some(sess) = b.as_ref() else {
            return;
        };
        ui::ffi::pe_set_text(ui::ffi::PE_LABEL, &sess.label);
        ui::ffi::pe_set_text(ui::ffi::PE_MODEL, &sess.model);
        ui::ffi::pe_set_text(ui::ffi::PE_MAXTOK, &sess.max_tokens.to_string());
        ui::ffi::pe_set_text(ui::ffi::PE_MAXTURNS, &sess.max_turns.to_string());
        ui::ffi::pe_set_text(ui::ffi::PE_KEY, "");
        // Fixed-endpoint providers (Anthropic, Perplexity) hide the base-URL row and
        // the vision/audio checkboxes. OpenAI-compatible shows all three. The
        // Anthropic-only "Extended thinking" toggle shares the vision row (they never
        // coexist); it's hidden for every other kind.
        if sess.kind.has_fixed_endpoint() {
            ui::ffi::pe_show(ui::ffi::PE_BASEURL, false);
            ui::ffi::pe_show(ui::ffi::PE_BASEURL_LBL, false);
            ui::ffi::pe_show(ui::ffi::PE_VISION, false);
            ui::ffi::pe_show(ui::ffi::PE_AUDIO, false);
        } else {
            ui::ffi::pe_set_text(ui::ffi::PE_BASEURL, &sess.base_url);
            ui::ffi::pe_set_check(ui::ffi::PE_VISION, sess.vision);
            ui::ffi::pe_set_check(ui::ffi::PE_AUDIO, sess.audio);
        }
        if sess.kind == AdapterKind::Anthropic {
            ui::ffi::pe_set_check(ui::ffi::PE_THINKING, sess.thinking);
        } else {
            ui::ffi::pe_show(ui::ffi::PE_THINKING, false);
        }
        // Fill the key list (masked) + its summary hint.
        repopulate_keys(&sess.keys, None, sess.keys_loaded, sess.env_active);
    });
}

/// Refill the key listbox from `keys` (masked), set the summary hint, and (if
/// given) select row `selected`. Touches only the dialog, never SESSION.
/// `loaded` = whether the stored keys were read successfully; `env_active` = an
/// Anthropic account running off ANTHROPIC_API_KEY with nothing stored.
fn repopulate_keys(keys: &[String], selected: Option<usize>, loaded: bool, env_active: bool) {
    let masked: Vec<String> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| format!("{}. {}", i + 1, mask_key(k)))
        .collect();
    ui::ffi::pe_set_list(ui::ffi::PE_KEYLIST, &masked);
    let hint = if !loaded {
        "Couldn't read the stored keys (credential store locked?). Editing keys here \
         could overwrite the existing ones."
            .to_string()
    } else {
        match keys.len() {
            0 if env_active => {
                "Using the ANTHROPIC_API_KEY environment variable. Add a key here to override it."
                    .to_string()
            }
            0 => "No keys yet - type one above and press Add. (Local servers can stay keyless.)"
                .to_string(),
            1 => "1 key configured.".to_string(),
            n => format!("{n} keys - the top is used; on a limit it falls back down the list."),
        }
    };
    ui::ffi::pe_set_text(ui::ffi::PE_KEYHINT, &hint);
    if let Some(i) = selected {
        if !keys.is_empty() {
            ui::ffi::pe_set_sel(ui::ffi::PE_KEYLIST, i.min(keys.len() - 1));
        }
    }
}

/// A masked, display-safe form of a key: the last 4 characters behind bullets, so
/// the list is legible (which key is which) without exposing the secret.
fn mask_key(key: &str) -> String {
    let k = key.trim();
    let n = k.chars().count();
    if n <= 4 {
        "\u{2022}\u{2022}\u{2022}\u{2022}".to_string()
    } else {
        let tail: String = k.chars().skip(n - 4).collect();
        format!("\u{2022}\u{2022}\u{2022}\u{2022}{tail}")
    }
}

/// A key-list button was pressed: 0=add, 1=delete, 2=move up, 3=move down. Mutate
/// the working list, then refresh the listbox and announce the result. Main thread.
pub fn edit_dialog_key(action: i32) {
    let refreshed = SESSION.with(|s| {
        let mut b = s.borrow_mut();
        let sess = b.as_mut()?;
        // Default to keeping the current selection so a no-op (boundary move, empty
        // Add, etc.) never silently deselects the row — jarring under a screen reader.
        let mut select: Option<usize> = ui::ffi::pe_get_sel(ui::ffi::PE_KEYLIST);
        let mut announce: Option<String> = None;
        match action {
            0 => {
                let field = ui::ffi::pe_get_text(ui::ffi::PE_KEY).trim().to_string();
                if field.is_empty() {
                    announce = Some("Type a key in the field first, then press Add.".into());
                } else if sess.keys.iter().any(|k| k == &field) {
                    announce = Some("That key is already in the list.".into());
                } else {
                    sess.keys.push(field);
                    sess.keys_dirty = true;
                    ui::ffi::pe_set_text(ui::ffi::PE_KEY, "");
                    select = Some(sess.keys.len() - 1);
                    announce = Some(format!("Key added. {} in the list.", sess.keys.len()));
                }
            }
            1 => match ui::ffi::pe_get_sel(ui::ffi::PE_KEYLIST) {
                Some(i) if i < sess.keys.len() => {
                    sess.keys.remove(i);
                    sess.keys_dirty = true;
                    select = (!sess.keys.is_empty()).then(|| i.min(sess.keys.len() - 1));
                    announce = Some(format!("Key removed. {} left.", sess.keys.len()));
                }
                _ => announce = Some("Select a key to delete.".into()),
            },
            2 | 3 => {
                let up = action == 2;
                match ui::ffi::pe_get_sel(ui::ffi::PE_KEYLIST) {
                    Some(i) if i < sess.keys.len() => {
                        let target = if up {
                            i.checked_sub(1)
                        } else if i + 1 < sess.keys.len() {
                            Some(i + 1)
                        } else {
                            None
                        };
                        if let Some(j) = target {
                            sess.keys.swap(i, j);
                            sess.keys_dirty = true;
                            select = Some(j);
                            announce = Some(format!("Key moved {}.", if up { "up" } else { "down" }));
                        } else {
                            announce =
                                Some(format!("Already at the {}.", if up { "top" } else { "bottom" }));
                        }
                    }
                    _ => announce = Some("Select a key to move.".into()),
                }
            }
            _ => {}
        }
        Some((sess.keys.clone(), select, sess.keys_loaded, sess.env_active, announce))
    });
    if let Some((keys, select, loaded, env_active, announce)) = refreshed {
        repopulate_keys(&keys, select, loaded, env_active);
        if let Some(msg) = announce {
            osara::announce(&msg);
        }
    }
}

/// "Fetch models" clicked: fetch the list using the endpoint + key currently in
/// the dialog, let the user pick, and set the Model field + vision checkbox.
pub fn edit_dialog_fetch() {
    let Some((kind, id)) =
        SESSION.with(|s| s.borrow().as_ref().map(|sess| (sess.kind, sess.id.clone())))
    else {
        return;
    };

    // Read live (possibly-unsaved) endpoint + key from the dialog. For Anthropic
    // the base-URL field is hidden/empty; models_api uses the fixed endpoint.
    let base = ui::ffi::pe_get_text(ui::ffi::PE_BASEURL);
    let key_field = ui::ffi::pe_get_text(ui::ffi::PE_KEY);
    let key = if key_field.trim().is_empty() {
        registry::key_for(&id) // fall back to the stored key (edit)
    } else {
        Some(key_field.trim().to_string())
    };
    let key_missing = key.as_deref().map(str::trim).unwrap_or("").is_empty();

    if kind == AdapterKind::OpenAiCompatible && base.trim().is_empty() {
        ui::ffi::message_box(
            "Fetch models",
            "Enter the Base URL first, then fetch the model list.",
            false,
        );
        return;
    }

    // Anthropic always needs a key — hitting the API without one just returns an
    // opaque HTTP error, so guide the user instead. (OpenAI-compatible endpoints
    // may be keyless local servers like Ollama/LM Studio, so those are allowed to
    // proceed and only get the hint if the request actually fails.)
    if kind == AdapterKind::Anthropic && key_missing {
        ui::ffi::message_box(
            "Fetch models",
            "Enter your API key first, then fetch the model list.",
            false,
        );
        return;
    }

    osara::announce("Fetching models\u{2026}");
    let models = match models_api::fetch_models(kind, base.trim(), key.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            // A missing key is the most common cause of an auth/404 failure; call it
            // out explicitly since the raw provider error rarely makes it obvious.
            // Not for Perplexity Agent, whose fetch never hits the network (it just
            // returns type-a-model-id guidance), so a key hint would be misleading.
            let hint = if key_missing && kind != AdapterKind::PerplexityAgent {
                "\n\nNo API key was entered. Most cloud providers require one: type your key in \
                 the \u{201c}API key\u{201d} field, then try Fetch models again. (Local providers \
                 such as Ollama or LM Studio don\u{2019}t need a key.)"
            } else {
                ""
            };
            ui::ffi::message_box(
                "Fetch models",
                &format!("Could not fetch models: {e}{hint}"),
                false,
            );
            return;
        }
    };
    if models.is_empty() {
        osara::announce("The provider returned no models.");
        return;
    }

    let refs: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    let choice = ui::ffi::popup_menu(&refs);
    if choice == 0 || choice > models.len() {
        return; // cancelled — user can still type a model by hand
    }
    let mi = &models[choice - 1];
    // Trust the provider's reported vision flag; infer from the id otherwise. Audio
    // isn't reported by the model list, so seed it from the id (the checkbox overrides).
    let vision = mi.vision.unwrap_or_else(|| infer_vision(&mi.id));
    let audio = infer_audio(&mi.id);
    ui::ffi::pe_set_text(ui::ffi::PE_MODEL, &mi.id);
    // Only OpenAI-compatible accounts show the vision/audio checkboxes to sync.
    if kind == AdapterKind::OpenAiCompatible {
        ui::ffi::pe_set_check(ui::ffi::PE_VISION, vision);
        ui::ffi::pe_set_check(ui::ffi::PE_AUDIO, audio);
    }
    osara::announce(&format!("Model set to {}.", mi.id));
}

/// OK clicked: read the fields, save (add or update), and report whether the
/// dialog should close (true = close; false = keep open so the user can fix an
/// error).
pub fn edit_dialog_ok() -> bool {
    let Some((is_new, id, kind, def_label, def_model, def_max, def_turns, keys, keys_dirty)) =
        SESSION.with(|s| {
            s.borrow().as_ref().map(|x| {
                (
                    x.is_new,
                    x.id.clone(),
                    x.kind,
                    x.label.clone(),
                    x.model.clone(),
                    x.max_tokens,
                    x.max_turns,
                    x.keys.clone(),
                    x.keys_dirty,
                )
            })
        })
    else {
        return true;
    };

    let label = nonempty(ui::ffi::pe_get_text(ui::ffi::PE_LABEL), def_label);
    let model = nonempty(ui::ffi::pe_get_text(ui::ffi::PE_MODEL), def_model);
    // Max tokens: empty keeps the current value; a non-empty but invalid entry is
    // rejected with feedback (silent revert would be invisible to a screen reader).
    let max_raw = ui::ffi::pe_get_text(ui::ffi::PE_MAXTOK);
    let max_trimmed = max_raw.trim();
    let max_tokens = if max_trimmed.is_empty() {
        def_max
    } else {
        match max_trimmed.parse::<u32>() {
            Ok(n) if n > 0 => n,
            _ => {
                ui::ffi::message_box(
                    "Provider settings",
                    "Max tokens must be a positive whole number.",
                    false,
                );
                return false; // keep the dialog open so the user can fix it
            }
        }
    };
    // Tool steps (max agentic turns): same empty=keep / invalid=reject rule.
    // Clamped to 1..=200 by config::max_turns at use; reject 0/garbage here.
    let turns_raw = ui::ffi::pe_get_text(ui::ffi::PE_MAXTURNS);
    let turns_trimmed = turns_raw.trim();
    let max_turns = if turns_trimmed.is_empty() {
        def_turns
    } else {
        match turns_trimmed.parse::<u32>() {
            Ok(n) if (1..=200).contains(&n) => n,
            _ => {
                ui::ffi::message_box(
                    "Provider settings",
                    "Tool steps must be a whole number between 1 and 200.",
                    false,
                );
                return false;
            }
        }
    };
    let (base_url, vision) = if kind.has_fixed_endpoint() {
        // No user base URL. Anthropic is always vision-capable; Perplexity's Agent
        // adapter doesn't bridge images (v1), so it's not.
        (None, kind == AdapterKind::Anthropic)
    } else {
        let b = ui::ffi::pe_get_text(ui::ffi::PE_BASEURL).trim().to_string();
        (
            (!b.is_empty()).then_some(b),
            ui::ffi::pe_get_check(ui::ffi::PE_VISION),
        )
    };
    let key = ui::ffi::pe_get_text(ui::ffi::PE_KEY).trim().to_string();

    let name = label.clone(); // for the announcement (cfg takes ownership of label)
    // Audio input (listening) is a per-model capability set with the "Supports
    // audio" checkbox — the id heuristic only seeds its default (add/fetch). This
    // lets locally-run multimodal models (e.g. Gemma) enable audio even though
    // their id can't be reliably classified. Anthropic models don't take audio.
    let supports_audio =
        kind == AdapterKind::OpenAiCompatible && ui::ffi::pe_get_check(ui::ffi::PE_AUDIO);
    // Extended thinking (reasoning) is Anthropic-only — the checkbox is shown just
    // for Anthropic accounts; OpenAI-compatible models expose reasoning inherently.
    let thinking = kind == AdapterKind::Anthropic && ui::ffi::pe_get_check(ui::ffi::PE_THINKING);
    let cfg = ProviderConfig {
        id,
        label,
        // Every provider added through this dialog today is a chat account; the
        // role-aware UI (transcription tab) will thread the real role through here.
        role: ProviderRole::Chat,
        kind,
        base_url,
        model,
        max_tokens,
        max_turns,
        supports_images: vision,
        supports_audio,
        thinking,
    };

    // The final key list is the working list plus a key typed into the field but
    // not yet pressed Add (a convenience for the common single-key case). A typed
    // key counts as a change, so it triggers the save below.
    let mut final_keys = keys;
    let mut dirty = keys_dirty;
    if !key.is_empty() && !final_keys.iter().any(|k| k == &key) {
        final_keys.push(key);
        dirty = true;
    }

    let pid = cfg.id.clone();
    let result = if is_new {
        registry::add(cfg)
    } else {
        registry::update(cfg)
    };

    match result {
        Ok(()) => {
            // Only rewrite the credential store when the user actually changed the
            // keys — so an unrelated edit (or a failed read that showed an empty
            // list) never clobbers the stored keys. Surface a write failure instead
            // of silently reporting success with the keys lost.
            if dirty {
                if let Err(e) = registry::set_keys(&pid, &final_keys) {
                    ui::ffi::message_box(
                        "Provider settings",
                        &format!(
                            "The provider was saved, but its API key(s) could not be stored: {e}\n\n\
                             Open the provider again and re-enter the keys."
                        ),
                        false,
                    );
                }
            }
            osara::announce(&format!(
                "Provider {name} {}.",
                if is_new { "added" } else { "updated" }
            ));
            SESSION.with(|s| {
                if let Some(x) = s.borrow_mut().as_mut() {
                    x.changed = true;
                }
            });
            true // close
        }
        Err(e) => {
            ui::ffi::message_box(
                if is_new { "Add provider" } else { "Edit provider" },
                &format!("Could not save provider: {e}"),
                false,
            );
            false // keep the dialog open so the user can correct it
        }
    }
}

// ---- helpers ----------------------------------------------------------------

/// The provider at a listbox row (registry order), if the index is valid.
fn provider_at(index: i32) -> Option<ProviderConfig> {
    let i = usize::try_from(index).ok()?;
    registry::list().into_iter().nth(i)
}

/// A trimmed value, or `fallback` if it was left empty.
fn nonempty(v: String, fallback: String) -> String {
    let t = v.trim();
    if t.is_empty() {
        fallback
    } else {
        t.to_string()
    }
}

/// Guess whether a model accepts image input from its id — the fallback when the
/// provider's model list doesn't report vision explicitly. Overridable by the
/// user via the vision checkbox in the settings dialog.
fn infer_vision(model: &str) -> bool {
    let m = model.to_lowercase();
    const HINTS: &[&str] = &[
        "vision",
        "-vl",
        "vl-",
        "llava",
        "pixtral",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gemini",
        "gemma", // Google Gemma 3 / 3n / 4 — all vision-capable (run locally too)
        "claude",
        "grok-2-vision",
        "grok-3",
        "grok-4",
        "llama-4",
        "llama3.2-vision",
        "internvl",
        "moondream",
        "minicpm-v",
    ];
    HINTS.iter().any(|h| m.contains(h))
}

/// Guess whether a model accepts audio INPUT from its id — audio is rare, so a
/// small heuristic beats a dedicated control: OpenAI's `*-audio-*`/`gpt-audio`
/// and all Gemini chat models accept audio; everything else does not.
fn infer_audio(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("audio") || m.contains("gemini")
}

/// A provider id derived from `base`, uniquified against the existing ids.
fn unique_id(base: &str) -> String {
    let existing: Vec<String> = registry::list().into_iter().map(|p| p.id).collect();
    if !existing.iter().any(|e| e == base) {
        return base.to_string();
    }
    let mut n = 2;
    loop {
        let cand = format!("{base}-{n}");
        if !existing.iter().any(|e| e == &cand) {
            return cand;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{infer_audio, infer_vision};

    #[test]
    fn gemma_is_recognised_as_vision_capable() {
        // Gemma (3 / 3n / 4) is multimodal; recognized across common id spellings.
        assert!(infer_vision("gemma-3-12b"));
        assert!(infer_vision("gemma3n:e4b"));
        assert!(infer_vision("gemma-4-12b"));
        // A plain text model is still not treated as vision-capable.
        assert!(!infer_vision("llama3.2"));
    }

    #[test]
    fn audio_inference_seeds_known_audio_models_only() {
        // The heuristic only seeds the checkbox default; the user can override it.
        assert!(infer_audio("gpt-audio"));
        assert!(infer_audio("gemini-3.5-flash"));
        // Gemma's id can't be classified, so it defaults off and relies on the
        // checkbox (gemma != gemini, so no accidental match).
        assert!(!infer_audio("gemma-4-12b"));
        assert!(!infer_audio("llama3.2"));
    }
}
