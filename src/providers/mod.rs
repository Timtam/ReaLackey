//! LLM provider abstraction. Phase 0 shipped one adapter (native Anthropic);
//! Phase 1 adds tool use. Later phases add a shared OpenAI-compatible adapter
//! (design §kap-llm).

pub mod anthropic;
pub mod registry;

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
}

/// A single content block within a message (mirrors the Anthropic content model).
#[derive(Clone, Debug)]
pub enum Content {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
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
    ToolCall {
        id: String,
        name: String,
        input: Value,
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
    #[error("HTTP: {0}")]
    Http(String),
    #[error("missing API key: set {0}")]
    MissingKey(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    // `id`/`capabilities` are the extensibility surface used from Phase 5.
    #[allow(dead_code)]
    fn id(&self) -> &'static str;
    #[allow(dead_code)]
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
