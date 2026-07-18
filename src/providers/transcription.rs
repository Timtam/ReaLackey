//! Speech-to-text (transcription) providers — a sibling capability to chat.
//!
//! A transcription account is the `Transcription` [`ProviderRole`]. v1 has one
//! adapter: an OpenAI-compatible `POST {base_url}/audio/transcriptions` (multipart
//! upload, `response_format=verbose_json` for timestamped segments). That one shape
//! covers cloud OpenAI Whisper AND local servers that mirror it — whisper.cpp's
//! server, faster-whisper, LocalAI — the same "one adapter, many backends" leverage
//! the chat OpenAI-compatible adapter has. (Ollama itself does not serve ASR; point
//! at a separate local whisper server.)
//!
//! The timestamped [`Transcript`] is the shared currency the chat agent (via a
//! transcribe tool) and direct REAPER actions both consume — see the
//! `transcription-toolkit-wishlist` project note.
//!
//! Consumers (the transcribe tool + actions) land in a follow-up; the whole module
//! is phased-in, hence the module-level dead-code allowance.
#![allow(dead_code)]

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::providers::registry::ProviderConfig;
use crate::providers::ProviderError;

/// One timestamped chunk of transcribed speech.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// Start time in seconds, relative to the start of the clip.
    pub start: f64,
    /// End time in seconds, relative to the start of the clip.
    pub end: f64,
    pub text: String,
}

/// A completed transcription: the full text plus timestamped segments (empty if
/// the server returned only plain text).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The language the server reports, verbatim, when it reports one — OpenAI's
    /// verbose_json returns the full English name (e.g. "english"), not a code.
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

/// The audio to transcribe. The multipart upload needs the bytes plus a filename
/// whose extension tells the server the container format.
pub struct AudioClip {
    pub bytes: Vec<u8>,
    /// e.g. `audio.wav` — the extension is how the server infers the format.
    pub filename: String,
    /// e.g. `audio/wav`.
    pub mime: String,
}

/// Per-call transcription options (the model + endpoint live on the provider).
#[derive(Clone, Debug, Default)]
pub struct TranscribeOptions {
    /// ISO-639-1 language hint, or `None` to let the model auto-detect.
    pub language: Option<String>,
    /// Optional context/style prompt (proper nouns, formatting hints).
    pub prompt: Option<String>,
}

#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    /// Transcribe `clip`, honoring `cancel` promptly. Returns the full text plus
    /// timestamped segments (segments may be empty for a text-only server).
    async fn transcribe(
        &self,
        clip: AudioClip,
        opts: &TranscribeOptions,
        cancel: CancellationToken,
    ) -> Result<Transcript, ProviderError>;
}

/// Build the transcription adapter for a configured `Transcription` account.
/// v1 always uses the OpenAI-compatible audio endpoint (cloud or local); the
/// `AdapterKind` is not yet consulted (a future native adapter could branch on it).
pub fn build_transcriber(
    cfg: &ProviderConfig,
    key: Option<String>,
) -> Box<dyn TranscriptionProvider> {
    Box::new(OpenAiTranscriber::new(
        cfg.base_url.clone().unwrap_or_default(),
        key,
        cfg.model.clone(),
    ))
}

// ---- OpenAI-compatible /audio/transcriptions adapter ------------------------

pub struct OpenAiTranscriber {
    client: reqwest::Client,
    /// Endpoint base, e.g. `https://api.openai.com/v1` (trailing slash trimmed).
    base_url: String,
    /// Bearer key; `None` for a keyless local server.
    key: Option<String>,
    /// Model id. `whisper-1` (or a local whisper model) gives timestamped
    /// segments; OpenAI's `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` transcribe
    /// well but return TEXT ONLY (no segment timestamps — see `response_format_for`).
    model: String,
}

impl OpenAiTranscriber {
    pub fn new(base_url: impl Into<String>, key: Option<String>, model: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url,
            key,
            model: model.into(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url)
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAiTranscriber {
    async fn transcribe(
        &self,
        clip: AudioClip,
        opts: &TranscribeOptions,
        cancel: CancellationToken,
    ) -> Result<Transcript, ProviderError> {
        let part = reqwest::multipart::Part::bytes(clip.bytes)
            .file_name(clip.filename)
            .mime_str(&clip.mime)
            .map_err(|e| ProviderError::Other(e.to_string()))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone())
            // verbose_json gives per-segment timestamps + detected language — but
            // OpenAI's gpt-4o-transcribe family rejects it (whisper-1/local only).
            .text("response_format", response_format_for(&self.model));
        if let Some(lang) = opts.language.as_ref().filter(|l| !l.trim().is_empty()) {
            form = form.text("language", lang.clone());
        }
        if let Some(prompt) = opts.prompt.as_ref().filter(|p| !p.trim().is_empty()) {
            form = form.text("prompt", prompt.clone());
        }

        let mut builder = self.client.post(self.endpoint());
        if let Some(k) = &self.key {
            builder = builder.header("authorization", format!("Bearer {k}"));
        }
        let send = builder.multipart(form).send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            r = send => r.map_err(|e| ProviderError::Http { status: None, message: e.to_string() })?,
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status: Some(status.as_u16()),
                message: format!("API {status}: {text}"),
            });
        }

        let body = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            b = resp.text() => b.map_err(|e| ProviderError::Http { status: None, message: e.to_string() })?,
        };
        Ok(parse_response(&body))
    }
}

/// Parse a transcription response body into a [`Transcript`]. Handles the
/// `verbose_json` object (text + language + segments) and, defensively, a
/// plain-text or bare-`json` body (some local servers ignore `response_format`)
/// by treating the whole body as the transcript text with no segments.
fn parse_response(body: &str) -> Transcript {
    match serde_json::from_str::<Value>(body) {
        Ok(v) if v.is_object() => Transcript {
            text: v
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .trim()
                .to_string(),
            language: v
                .get("language")
                .and_then(|l| l.as_str())
                .map(|s| s.to_string()),
            segments: v
                .get("segments")
                .and_then(|s| s.as_array())
                .map(|arr| arr.iter().filter_map(parse_segment).collect())
                .unwrap_or_default(),
        },
        // Not a JSON object (plain text, or a bare JSON string/number): use it
        // verbatim as the text.
        _ => Transcript {
            text: body.trim().to_string(),
            language: None,
            segments: Vec::new(),
        },
    }
}

/// The `response_format` to request for `model`. `verbose_json` (timestamped
/// segments + detected language) is supported by whisper-1 and by local whisper
/// servers (whisper.cpp, faster-whisper, LocalAI), but OpenAI's `gpt-4o-transcribe`
/// / `gpt-4o-mini-transcribe` reject it with a 400 and only do `json`/`text`. So
/// those fall back to `json` — full text, but no segment timestamps (features that
/// need timing, like subtitles or cut-by-text, want whisper-1 or a local server).
fn response_format_for(model: &str) -> &'static str {
    if model.starts_with("gpt-4o") {
        "json"
    } else {
        "verbose_json"
    }
}

fn parse_segment(seg: &Value) -> Option<Segment> {
    let start = seg.get("start")?.as_f64()?;
    let end = seg.get("end")?.as_f64()?;
    let text = seg.get("text")?.as_str()?.trim().to_string();
    Some(Segment { start, end, text })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_verbose_json_with_segments() {
        let body = r#"{
            "task": "transcribe",
            "language": "english",
            "duration": 3.2,
            "text": "Hello world. Bring the vocals up.",
            "segments": [
                { "id": 0, "start": 0.0, "end": 1.4, "text": " Hello world." },
                { "id": 1, "start": 1.4, "end": 3.2, "text": " Bring the vocals up." }
            ]
        }"#;
        let t = parse_response(body);
        assert_eq!(t.text, "Hello world. Bring the vocals up.");
        assert_eq!(t.language.as_deref(), Some("english"));
        assert_eq!(t.segments.len(), 2);
        assert_eq!(t.segments[0].start, 0.0);
        assert_eq!(t.segments[0].end, 1.4);
        // Leading/trailing whitespace on segment text is trimmed.
        assert_eq!(t.segments[0].text, "Hello world.");
        assert_eq!(t.segments[1].text, "Bring the vocals up.");
    }

    #[test]
    fn falls_back_to_plain_text_body() {
        // A local server that ignored response_format and returned raw text.
        let t = parse_response("  just the words  ");
        assert_eq!(t.text, "just the words");
        assert!(t.language.is_none());
        assert!(t.segments.is_empty());
    }

    #[test]
    fn json_without_segments_keeps_text() {
        let t = parse_response(r#"{ "text": "no timestamps here" }"#);
        assert_eq!(t.text, "no timestamps here");
        assert!(t.segments.is_empty());
    }

    #[test]
    fn malformed_segments_are_skipped_not_fatal() {
        // A segment missing `end` is dropped; the well-formed one survives.
        let body = r#"{ "text": "x", "segments": [
            { "start": 0.0, "text": "no end" },
            { "start": 1.0, "end": 2.0, "text": "ok" }
        ]}"#;
        let t = parse_response(body);
        assert_eq!(t.segments.len(), 1);
        assert_eq!(t.segments[0].text, "ok");
    }

    #[test]
    fn endpoint_appends_audio_path_and_trims_slash() {
        let t = OpenAiTranscriber::new("https://api.openai.com/v1/", None, "whisper-1");
        assert_eq!(t.endpoint(), "https://api.openai.com/v1/audio/transcriptions");
    }

    #[test]
    fn response_format_avoids_verbose_json_for_gpt_4o_models() {
        // whisper-1 + local models get timestamped segments.
        assert_eq!(response_format_for("whisper-1"), "verbose_json");
        assert_eq!(response_format_for("Systran/faster-whisper-large-v3"), "verbose_json");
        // gpt-4o-transcribe family rejects verbose_json -> request json instead.
        assert_eq!(response_format_for("gpt-4o-transcribe"), "json");
        assert_eq!(response_format_for("gpt-4o-mini-transcribe"), "json");
    }
}
