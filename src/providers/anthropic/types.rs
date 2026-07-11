//! Wire types for the Anthropic Messages API (request builder + SSE stream events).
//! Serialized/deserialized directly with serde — no official Rust SDK.

// Several stream fields are deserialized for completeness but only consumed in
// later phases (thinking, usage display), so silence dead-code noise here.
#![allow(dead_code)]

use serde::Deserialize;
use serde_json::{json, Value};

use crate::providers::{ChatRequest, Content, Role};

// ---- Request ----------------------------------------------------------------

/// Build the JSON request body (content blocks + tools + streaming).
pub fn build_request(req: &ChatRequest) -> Value {
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": match m.role { Role::User => "user", Role::Assistant => "assistant" },
                "content": m.content.iter().map(content_to_json).collect::<Vec<_>>(),
            })
        })
        .collect();

    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": true,
        "messages": messages,
    });
    if let Some(system) = &req.system {
        body["system"] = json!(system);
    }
    if !req.tools.is_empty() {
        body["tools"] = json!(req
            .tools
            .iter()
            .map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }))
            .collect::<Vec<_>>());
    }
    body
}

fn content_to_json(c: &Content) -> Value {
    match c {
        Content::Text(text) => json!({ "type": "text", "text": text }),
        Content::ToolUse { id, name, input } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        Content::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            });
            if *is_error {
                v["is_error"] = json!(true);
            }
            v
        }
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
    /// block index and parse once at `ContentBlockStop`.
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
