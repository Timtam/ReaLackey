//! Fetch a provider's available models so the dialog can offer a pick-list
//! instead of a hand-typed model id (Phase 5, M5).
//!
//! `GET {base_url}/models` is near-universal across OpenAI-compatible providers
//! (OpenAI, Groq, OpenRouter, DeepSeek, xAI, Gemini's OpenAI-compat endpoint,
//! Ollama, LM Studio) — all return `{ "data": [ { "id": … } ] }`. Anthropic is
//! the outlier (different auth headers, richer per-model `capabilities`). Only
//! a few providers report vision support in the list; where they don't, the
//! caller infers it from the model id.
//!
//! The call is synchronous (a short-lived current-thread runtime) because it is
//! invoked from the modal provider dialog on the REAPER main thread. It carries
//! a connect + overall timeout so an unreachable endpoint can't hang the UI.

use std::time::Duration;

use serde_json::Value;

use crate::providers::registry::AdapterKind;

/// One model as offered to the user, with vision support if the provider states it.
pub struct ModelInfo {
    pub id: String,
    /// `Some(true/false)` when the provider's list reports image support
    /// (Anthropic, OpenRouter, xAI); `None` when it must be inferred.
    pub vision: Option<bool>,
}

/// Fetch and parse the model list for an account. Blocks the caller (main thread,
/// modal) for at most the configured timeout. Returns a friendly error string.
pub fn fetch_models(
    kind: AdapterKind,
    base_url: &str,
    key: Option<&str>,
) -> Result<Vec<ModelInfo>, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let out = rt.block_on(fetch_async(kind, base_url, key));
    // Detach — don't join — any blocking task still running (notably a hung
    // getaddrinfo DNS lookup, which reqwest can't cancel): dropping the runtime
    // normally blocks the caller until such a task finishes, which would extend
    // the modal main-thread freeze past our connect/read timeouts.
    rt.shutdown_background();
    out
}

async fn fetch_async(
    kind: AdapterKind,
    base_url: &str,
    key: Option<&str>,
) -> Result<Vec<ModelInfo>, String> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(6))
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    let req = match kind {
        AdapterKind::Anthropic => {
            // Fixed endpoint; x-api-key + anthropic-version (not Bearer).
            let mut r = client
                .get("https://api.anthropic.com/v1/models")
                .header("anthropic-version", "2023-06-01");
            if let Some(k) = key {
                r = r.header("x-api-key", k);
            }
            r
        }
        AdapterKind::OpenAiCompatible => {
            let base = base_url.trim_end_matches('/');
            if base.is_empty() {
                return Err("no base URL set for this provider".into());
            }
            let mut r = client.get(format!("{base}/models"));
            if let Some(k) = key {
                r = r.header("authorization", format!("Bearer {k}"));
            }
            r
        }
    };

    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(300).collect();
        return Err(format!("HTTP {status}: {snippet}"));
    }
    let val: Value = resp.json().await.map_err(|e| e.to_string())?;
    parse_models(&val)
}

/// Parse the `{ "data": [ { "id", … } ] }` list (both OpenAI-style and Anthropic
/// use `data`), pulling a vision flag where the provider exposes one.
fn parse_models(val: &Value) -> Result<Vec<ModelInfo>, String> {
    let arr = val
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or("unexpected response (no \"data\" array)")?;
    let mut out: Vec<ModelInfo> = arr
        .iter()
        .filter_map(|m| {
            // Skip entries without a usable id: an empty id would also desync the
            // pick-menu index mapping (ui_popup_menu drops empty lines).
            let id = m
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?;
            Some(ModelInfo {
                id: id.to_string(),
                vision: detect_vision(m),
            })
        })
        .collect();
    if out.is_empty() {
        return Err("no models returned".into());
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Read an explicit vision flag from a model entry, across the three shapes that
/// carry one. `None` if the provider doesn't say.
fn detect_vision(m: &Value) -> Option<bool> {
    // Anthropic: capabilities.image_input.supported
    if let Some(b) = m
        .pointer("/capabilities/image_input/supported")
        .and_then(|v| v.as_bool())
    {
        return Some(b);
    }
    // OpenRouter: architecture.input_modalities contains "image"
    if let Some(mods) = m
        .pointer("/architecture/input_modalities")
        .and_then(|v| v.as_array())
    {
        return Some(mods.iter().any(|x| x.as_str() == Some("image")));
    }
    // xAI /v1/language-models: top-level input_modalities
    if let Some(mods) = m.get("input_modalities").and_then(|v| v.as_array()) {
        return Some(mods.iter().any(|x| x.as_str() == Some("image")));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_style_list_without_vision_flag() {
        let v = json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o", "object": "model" },
                { "id": "gpt-3.5-turbo", "object": "model" }
            ]
        });
        let models = parse_models(&v).unwrap();
        assert_eq!(models.len(), 2);
        // Sorted; no explicit vision flag from OpenAI's list.
        assert_eq!(models[0].id, "gpt-3.5-turbo");
        assert_eq!(models[0].vision, None);
    }

    #[test]
    fn reads_anthropic_and_openrouter_vision_flags() {
        let anthropic = json!({
            "data": [ { "id": "claude-opus-4-8",
                        "capabilities": { "image_input": { "supported": true } } } ]
        });
        assert_eq!(parse_models(&anthropic).unwrap()[0].vision, Some(true));

        let openrouter = json!({
            "data": [
                { "id": "vision/model", "architecture": { "input_modalities": ["text", "image"] } },
                { "id": "text/only",    "architecture": { "input_modalities": ["text"] } }
            ]
        });
        let m = parse_models(&openrouter).unwrap();
        // Sorted: "text/only" before "vision/model".
        assert_eq!(m[0].vision, Some(false));
        assert_eq!(m[1].vision, Some(true));
    }

    #[test]
    fn empty_or_shapeless_is_an_error() {
        assert!(parse_models(&json!({ "data": [] })).is_err());
        assert!(parse_models(&json!({ "models": [] })).is_err());
    }

    #[test]
    fn skips_entries_with_empty_or_missing_id() {
        let v = json!({ "data": [
            { "id": "" },
            { "object": "model" },
            { "id": "real-model" }
        ]});
        let m = parse_models(&v).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].id, "real-model");
    }
}
