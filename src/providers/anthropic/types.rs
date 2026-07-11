//! Wire types for the Anthropic Messages API (request + SSE stream events).
//! Serialized/deserialized directly with serde — no official Rust SDK.

// These mirror the on-the-wire protocol. Several fields are deserialized for
// completeness but only consumed in later phases (tool_use, thinking, usage
// display), so silence dead-code noise deliberately here.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

use crate::providers::{ChatRequest, Role};

// ---- Request ----------------------------------------------------------------

#[derive(Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<WireMessage>,
}

#[derive(Serialize)]
pub struct WireMessage {
    pub role: &'static str,
    pub content: String,
}

pub fn build_request(req: &ChatRequest) -> MessagesRequest {
    let messages = req
        .messages
        .iter()
        .map(|m| WireMessage {
            role: match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            },
            content: m.content.clone(),
        })
        .collect();
    MessagesRequest {
        model: req.model.clone(),
        max_tokens: req.max_tokens,
        stream: true,
        system: req.system.clone(),
        messages,
    }
}

// ---- Streaming (SSE) response ------------------------------------------------

/// Top-level SSE event, tagged by the internal "type" field. `#[serde(other)]`
/// keeps the parser forward-compatible with new event types.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessageStub,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: Delta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<WireUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ApiError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageStub {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
// Variant names mirror the wire tags (`text_delta`, ...) via serde rename_all;
// renaming them to satisfy the lint would change the deserialized tags.
#[allow(clippy::enum_variant_names)]
pub enum Delta {
    TextDelta {
        text: String,
    },
    /// Tool-call arguments arrive as partial JSON fragments; concatenate per
    /// block index and parse once at `ContentBlockStop` (Phase 1+).
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        #[serde(default)]
        thinking: String,
    },
    SignatureDelta {
        #[serde(default)]
        signature: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct MessageDeltaBody {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct WireUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    /// CUMULATIVE on message_delta — take the last value, do not sum.
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

impl From<WireUsage> for crate::providers::Usage {
    fn from(u: WireUsage) -> Self {
        crate::providers::Usage {
            input_tokens: u.input_tokens.unwrap_or(0),
            output_tokens: u.output_tokens.unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub message: String,
}
