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

// ---- chunking (long audio) --------------------------------------------------

/// Minimum chunk length to upload. Many ASR endpoints (OpenAI Whisper) reject
/// audio shorter than ~0.1 s with a 400, so a sub-minimum window is folded into
/// its neighbour (or skipped when it's the whole clip).
pub const MIN_CHUNK_SECONDS: f64 = 0.1;

/// Split a total duration into transcription windows of at most `chunk_seconds`,
/// covering `[0, total)`. Whisper caps uploads (~25 MB ≈ ~13 min at 16 kHz mono),
/// so long items are transcribed in pieces and stitched. A short clip yields a
/// single window — chunking is the general path, N=1 is just its degenerate case.
///
/// Fixed boundaries can clip a word at a seam; a future refinement can snap seams
/// to detected silence (the DSP layer already finds silent regions). Kept as one
/// function so that upgrade is local.
pub fn plan_chunks(total_seconds: f64, chunk_seconds: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if total_seconds <= 0.0 || chunk_seconds <= 0.0 {
        return out;
    }
    let mut start = 0.0;
    // Small epsilon so floating error near the end doesn't emit a zero-length tail.
    while start < total_seconds - 1e-6 {
        let len = chunk_seconds.min(total_seconds - start);
        out.push((start, len));
        start += len;
    }
    // Fold a too-short trailing chunk into its predecessor so a multi-chunk run
    // never ends on a sub-minimum clip the endpoint would 400 on (which would abort
    // the whole run and discard the already-transcribed chunks). A whole clip that
    // is itself below the minimum stays a single short chunk — the caller skips it.
    if out.len() >= 2 {
        let last = out[out.len() - 1].1;
        if last < MIN_CHUNK_SECONDS {
            let prev = out.len() - 2;
            out[prev].1 += last;
            out.pop();
        }
    }
    out
}

/// Merge per-chunk transcripts (each paired with its start offset in seconds within
/// the item) into one item-relative transcript: every segment's timestamps are
/// shifted by its chunk's offset, texts are joined, and the language is taken from
/// the first chunk that reports one.
pub fn merge_transcripts(chunks: &[(f64, Transcript)]) -> Transcript {
    let mut text_parts: Vec<String> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut language: Option<String> = None;
    for (offset, t) in chunks {
        let trimmed = t.text.trim();
        if !trimmed.is_empty() {
            text_parts.push(trimmed.to_string());
        }
        if language.is_none() {
            language = t.language.clone();
        }
        for s in &t.segments {
            segments.push(Segment {
                start: s.start + offset,
                end: s.end + offset,
                text: s.text.clone(),
            });
        }
    }
    Transcript {
        text: text_parts.join(" "),
        language,
        segments,
    }
}

// ---- output formatting ------------------------------------------------------

/// Format a transcript's segments as an SRT subtitle file, using its (item-relative)
/// timestamps. Empty when there are no segments (e.g. a text-only model).
pub fn to_srt(t: &Transcript) -> String {
    let mut out = String::new();
    for (i, s) in t.segments.iter().enumerate() {
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&srt_time(s.start));
        out.push_str(" --> ");
        out.push_str(&srt_time(s.end));
        out.push('\n');
        out.push_str(s.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// Seconds -> `HH:MM:SS,mmm` (the SRT timecode form). Negative clamps to zero.
fn srt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        total_s / 3600,
        (total_s / 60) % 60,
        total_s % 60,
        ms
    )
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
        // A generous per-request timeout backstops a server that accepts the upload
        // then stalls (reqwest has no default timeout, so send() would hang forever).
        // 5 min comfortably covers uploading + transcribing a ~10-minute chunk.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
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
    fn plan_chunks_covers_the_whole_duration() {
        // Short clip -> a single chunk (the N=1 general case).
        assert_eq!(plan_chunks(30.0, 600.0), vec![(0.0, 30.0)]);
        // Exact multiple -> equal chunks, no zero-length tail.
        assert_eq!(plan_chunks(1200.0, 600.0), vec![(0.0, 600.0), (600.0, 600.0)]);
        // Remainder -> a short final chunk.
        let c = plan_chunks(1300.0, 600.0);
        assert_eq!(c, vec![(0.0, 600.0), (600.0, 600.0), (1200.0, 100.0)]);
        // The windows tile [0, total) with no gaps or overlaps.
        let covered: f64 = c.iter().map(|(_, l)| l).sum();
        assert!((covered - 1300.0).abs() < 1e-9);
        // Degenerate inputs -> no chunks (never a zero/negative window).
        assert!(plan_chunks(0.0, 600.0).is_empty());
        assert!(plan_chunks(100.0, 0.0).is_empty());
    }

    #[test]
    fn srt_formats_segments_with_timecodes() {
        let t = Transcript {
            text: "hi there".into(),
            language: None,
            segments: vec![
                Segment { start: 0.0, end: 1.5, text: "hi".into() },
                Segment { start: 3661.5, end: 3661.75, text: "there".into() },
            ],
        };
        let srt = to_srt(&t);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhi\n\n"), "{srt}");
        // Hours/minutes/millis all carry: 3661.5 s = 01:01:01,500.
        assert!(srt.contains("2\n01:01:01,500 --> 01:01:01,750\nthere\n\n"), "{srt}");
        // No segments (text-only model) -> empty SRT.
        let empty = Transcript { text: "x".into(), language: None, segments: vec![] };
        assert!(to_srt(&empty).is_empty());
    }

    #[test]
    fn plan_chunks_folds_a_sub_minimum_tail() {
        // A 0.05 s tail would 400 on Whisper -> fold it into the previous chunk.
        assert_eq!(plan_chunks(600.05, 600.0), vec![(0.0, 600.05)]);
        // A comfortably-long tail is kept as its own chunk.
        assert_eq!(plan_chunks(700.0, 600.0), vec![(0.0, 600.0), (600.0, 100.0)]);
        // A whole clip below the minimum stays one short chunk (caller skips it).
        assert_eq!(plan_chunks(0.05, 600.0), vec![(0.0, 0.05)]);
    }

    #[test]
    fn merge_offsets_segment_timestamps_by_chunk_start() {
        let c0 = Transcript {
            text: "hello world".into(),
            language: Some("english".into()),
            segments: vec![Segment { start: 0.0, end: 2.0, text: "hello world".into() }],
        };
        let c1 = Transcript {
            text: "second part".into(),
            language: Some("english".into()),
            segments: vec![Segment { start: 1.0, end: 3.0, text: "second part".into() }],
        };
        // Chunk 1 started 600 s into the item.
        let merged = merge_transcripts(&[(0.0, c0), (600.0, c1)]);
        assert_eq!(merged.text, "hello world second part");
        assert_eq!(merged.language.as_deref(), Some("english"));
        assert_eq!(merged.segments.len(), 2);
        // First chunk's segment is unshifted; second is shifted by its 600 s offset.
        assert_eq!(merged.segments[0].start, 0.0);
        assert_eq!(merged.segments[1].start, 601.0);
        assert_eq!(merged.segments[1].end, 603.0);
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
