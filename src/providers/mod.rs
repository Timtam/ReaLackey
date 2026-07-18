//! LLM provider abstraction. Phase 0 shipped one adapter (native Anthropic);
//! Phase 1 adds tool use. Later phases add a shared OpenAI-compatible adapter
//! (design §kap-llm).

pub mod anthropic;
pub mod models_api;
pub mod openai_compat;
pub mod perplexity_agent;
pub mod registry;
pub mod sse;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

/// Model capabilities that gate features (vision, direct audio). Cross-cut per
/// design §kap-capabilities. Surfaced in the UI from Phase 5.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub supports_images: bool,
    pub supports_audio: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One block inside a tool result. Anthropic allows `tool_result.content` to be
/// an array of text and image blocks; a plain text result stays a single Text.
#[derive(Clone, Debug)]
pub enum ResultBlock {
    Text(String),
    /// An image the model can see (vision), e.g. a plugin-GUI screenshot.
    Image {
        media_type: String,
        data_base64: String,
    },
    /// Audio an audio-capable model can hear, e.g. a rendered clip. `format` is
    /// the bare container name (e.g. "wav") for the OpenAI `input_audio` part.
    Audio {
        format: String,
        data_base64: String,
    },
}

/// A single content block within a message (mirrors the Anthropic content model).
#[derive(Clone, Debug)]
pub enum Content {
    Text(String),
    /// An Anthropic extended-thinking block, kept as the ready-to-send Anthropic
    /// JSON so it can be replayed VERBATIM in the assistant turn on the next request
    /// (Anthropic requires the unmodified thinking block, with its signature, to
    /// precede the tool_use it thought about, or the request 400s). Other adapters
    /// ignore it. Only present when a provider has extended thinking enabled.
    Thinking {
        block: Value,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        /// Google Gemini thinking models attach an opaque `thought_signature` to
        /// each function call that MUST be echoed back on the next turn, or the
        /// request 400s. Opaque to us; `None` for every other provider.
        thought_signature: Option<String>,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ResultBlock>,
        is_error: bool,
    },
}

impl Content {
    /// Build a text-only tool result (the common case; keeps call sites terse).
    pub fn tool_result_text(
        tool_use_id: impl Into<String>,
        text: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Content::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: vec![ResultBlock::Text(text.into())],
            is_error,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Vec<Content>,
}

impl ChatMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.into())],
        }
    }
}

/// A tool definition sent to the model (JSON-Schema `input_schema`).
#[derive(Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDef>,
}

#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // token accounting: surfaced in a later phase
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Why a model turn ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other,
}

impl StopReason {
    pub fn from_wire(s: &str) -> Self {
        match s {
            "tool_use" => StopReason::ToolUse,
            "end_turn" => StopReason::EndTurn,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::Other,
        }
    }
}

/// Provider-neutral streaming event. The worker maps these onto `UiEvent`s and
/// drives the tool loop.
#[allow(dead_code)] // `Done.usage` consumed once token display lands
#[derive(Debug)]
pub enum ChatEvent {
    TextDelta(String),
    /// A streamed reasoning / "thinking" token (shown separately from the answer;
    /// not part of the final answer and not spoken via OSARA).
    ReasoningDelta(String),
    /// A COMPLETED Anthropic thinking block (verbatim JSON), to store in history and
    /// replay in the assistant turn. Anthropic-only; carries the signature the API
    /// validates on replay.
    ThinkingBlock(Value),
    ToolCall {
        id: String,
        name: String,
        input: Value,
        /// Gemini's per-call `thought_signature` (see `Content::ToolUse`); `None`
        /// for providers that don't emit one.
        thought_signature: Option<String>,
    },
    Done {
        stop_reason: StopReason,
        usage: Option<Usage>,
    },
    Error(String),
}

#[allow(dead_code)] // `Other` is a catch-all for future adapters
#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    /// A failed HTTP exchange. `status` is the response code when the failure was
    /// an HTTP error response (vs. a transport/stream error, where it is `None`),
    /// so the worker can tell a per-key limit from a malformed request.
    #[error("HTTP{}: {message}", .status.map(|s| format!(" {s}")).unwrap_or_default())]
    Http { status: Option<u16>, message: String },
    #[error("missing API key: set {0}")]
    MissingKey(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

impl ProviderError {
    /// Whether this error means the current API KEY cannot serve the request —
    /// rate-limit / quota exhausted (429) or auth / billing / permission
    /// (401/402/403) — so rotating to the next configured key may succeed. A
    /// malformed request (400) or a transport error is deliberately NOT this: it
    /// would fail identically on every key, so rotating would just burn them all.
    pub fn is_key_exhausted(&self) -> bool {
        matches!(
            self,
            ProviderError::Http { status: Some(429 | 401 | 402 | 403), .. }
        )
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    // `id` is the extensibility surface used from Phase 5.
    #[allow(dead_code)]
    fn id(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;

    /// Stream one model turn, emitting `ChatEvent`s on `tx` until done. Must
    /// observe `cancel` promptly and stop.
    async fn chat(
        &self,
        req: ChatRequest,
        tx: Sender<ChatEvent>,
        cancel: CancellationToken,
    ) -> Result<(), ProviderError>;
}

/// Build the adapter for a configured account using a SPECIFIC key — the worker
/// passes each key of the account's failover list in turn, rebuilding when it
/// rotates to the next one on a per-key limit. `key` is `None` for keyless local
/// servers (Ollama/LM Studio).
pub fn build_provider_with_key(
    cfg: &registry::ProviderConfig,
    key: Option<String>,
) -> Box<dyn LlmProvider> {
    match cfg.kind {
        registry::AdapterKind::Anthropic => {
            Box::new(anthropic::AnthropicProvider::with_key(key, cfg.thinking))
        }
        registry::AdapterKind::OpenAiCompatible => {
            Box::new(openai_compat::OpenAiCompatProvider::new(
                cfg.base_url.clone().unwrap_or_default(),
                key,
                cfg.supports_images,
                cfg.supports_audio,
            ))
        }
        // Web grounding (Perplexity's built-in web_search tool) is always on for
        // this provider in v1 — it's the whole reason to pick it. A per-provider
        // toggle can come later.
        registry::AdapterKind::PerplexityAgent => Box::new(
            perplexity_agent::PerplexityAgentProvider::with_key(key, true),
        ),
    }
}
