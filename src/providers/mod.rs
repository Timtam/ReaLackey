//! LLM provider abstraction. Phase 0 ships one adapter (native Anthropic).
//! Later phases add a shared OpenAI-compatible adapter (design §kap-llm).

pub mod anthropic;

use async_trait::async_trait;
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

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub messages: Vec<ChatMessage>,
}

#[allow(dead_code)] // token accounting: surfaced in a later phase
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Provider-neutral streaming event. The worker maps these onto `UiEvent`s.
#[allow(dead_code)] // `Done.usage` consumed once token display lands
#[derive(Debug)]
pub enum ChatEvent {
    TextDelta(String),
    // ToolCall { .. }  // Phase 1+
    Done { usage: Option<Usage> },
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
    // `id`/`capabilities` are the extensibility surface used from Phase 5
    // (provider/model switching + capability signalling in the UI).
    #[allow(dead_code)]
    fn id(&self) -> &'static str;
    #[allow(dead_code)]
    fn capabilities(&self) -> Capabilities;

    /// Stream a completion, emitting `ChatEvent`s on `tx` until done. Must
    /// observe `cancel` promptly and stop.
    async fn chat(
        &self,
        req: ChatRequest,
        tx: Sender<ChatEvent>,
        cancel: CancellationToken,
    ) -> Result<(), ProviderError>;
}
