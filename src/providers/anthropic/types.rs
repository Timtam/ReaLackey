//! Wire types for the Anthropic Messages API (request builder + SSE stream events).
//! Serialized/deserialized directly with serde — no official Rust SDK.

// Several stream fields are deserialized for completeness but only consumed in
// later phases (thinking, usage display), so silence dead-code noise here.
#![allow(dead_code)]

use serde::Deserialize;
use serde_json::{json, Value};

use crate::providers::{ChatRequest, Content, ResultBlock, Role};

// ---- Request ----------------------------------------------------------------

/// Build the JSON request body (content blocks + tools + streaming). `thinking`
/// requests extended thinking (adaptive) on models that support it.
pub fn build_request(req: &ChatRequest, thinking: bool) -> Value {
    // When thinking is OFF, drop any thinking blocks that a prior (thinking-on)
    // turn left in the history: the API rejects thinking blocks on a request that
    // doesn't enable thinking, so a mid-conversation toggle-off would otherwise 400
    // every subsequent turn. They only anchor a following tool_use *when thinking is
    // on*, so removing them here is safe.
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": match m.role { Role::User => "user", Role::Assistant => "assistant" },
                "content": m.content.iter()
                    .filter(|c| thinking || !matches!(c, Content::Thinking { .. }))
                    .map(content_to_json)
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": true,
        "messages": messages,
    });
    // Extended thinking (per-provider setting). Adaptive lets the model self-regulate
    // depth — it thinks more on hard tasks, less on easy ones. GA on Claude 4.6+
    // (Opus 4.8); older models don't support it and would 400. When on, the model
    // emits thinking blocks (summarized on 4.x) that we must replay verbatim in the
    // assistant turn — handled via Content::Thinking.
    if thinking {
        body["thinking"] = json!({ "type": "adaptive" });
    }
    // Prompt caching: the tools + system prefix is byte-identical across the turns
    // of one agentic loop, so mark it cacheable. A cache_control breakpoint caches
    // the contiguous prefix up to that block in Anthropic's tools -> system ->
    // messages order, so a marker on the last tool caches the tools, and one on the
    // system block caches tools + system. Cache READS are ~10% of the price AND are
    // not counted against the rate limit, so repeated turns re-read the ~10k static
    // prefix nearly for free. GA on current models — no beta header. Off switch:
    // RAAI_PROMPT_CACHE=off (for an endpoint that rejects the field).
    let cache = prompt_cache_enabled();
    if let Some(system) = &req.system {
        body["system"] = if cache {
            json!([{ "type": "text", "text": system, "cache_control": { "type": "ephemeral" } }])
        } else {
            json!(system)
        };
    }
    if !req.tools.is_empty() {
        let last = req.tools.len() - 1;
        body["tools"] = json!(req
            .tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut tv = json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                });
                if cache && i == last {
                    tv["cache_control"] = json!({ "type": "ephemeral" });
                }
                tv
            })
            .collect::<Vec<_>>());
    }
    body
}

/// Whether to attach `cache_control` markers (prompt caching). On by default;
/// set `RAAI_PROMPT_CACHE=off` (or 0/false/no) to disable — e.g. if a proxy in
/// front of the API rejects the field.
fn prompt_cache_enabled() -> bool {
    match std::env::var("RAAI_PROMPT_CACHE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no"
        ),
        Err(_) => true,
    }
}

fn content_to_json(c: &Content) -> Value {
    match c {
        Content::Text(text) => json!({ "type": "text", "text": text }),
        // Replayed verbatim (already the exact Anthropic block, with its signature).
        // Anthropic requires the unmodified thinking block before the tool_use.
        Content::Thinking { block } => block.clone(),
        Content::ToolUse {
            id, name, input, ..
        } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        Content::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            // Backward-compatible: a single text block serializes to the plain
            // string form (byte-identical to before image support). Only emit an
            // array of blocks when there is an image (or multiple blocks).
            let content_json = match content.as_slice() {
                // Defensive: an empty result would otherwise emit `[]`, which the
                // API rejects. Can't happen today (tools always add ≥1 block).
                [] => json!(""),
                [ResultBlock::Text(t)] => json!(t),
                blocks => json!(blocks.iter().map(result_block_to_json).collect::<Vec<_>>()),
            };
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_json,
            });
            if *is_error {
                v["is_error"] = json!(true);
            }
            v
        }
    }
}

fn result_block_to_json(b: &ResultBlock) -> Value {
    match b {
        ResultBlock::Text(text) => json!({ "type": "text", "text": text }),
        ResultBlock::Image {
            media_type,
            data_base64,
        } => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data_base64,
            }
        }),
        // The Anthropic Messages API has no audio input (Claude accounts report
        // supports_audio=false, so listen_to_audio isn't offered); degrade to text
        // defensively in case an audio block reaches here after a provider switch.
        ResultBlock::Audio { .. } => {
            json!({ "type": "text", "text": "[audio omitted: this model has no audio input]" })
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
    /// Extended-thinking block start (its text + signature arrive as deltas).
    Thinking {
        #[serde(default)]
        signature: String,
    },
    /// A redacted (encrypted) thinking block — carries opaque `data`, no deltas.
    RedactedThinking {
        #[serde(default)]
        data: String,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ResultBlock;

    #[test]
    fn text_only_tool_result_serializes_as_a_plain_string() {
        // Adding image support must NOT change the wire shape of ordinary
        // (text-only) tool results: `content` stays a string, not an array.
        let c = Content::tool_result_text("toolu_1", "the answer is 42", false);
        let v = content_to_json(&c);
        assert_eq!(v["type"], json!("tool_result"));
        assert_eq!(v["tool_use_id"], json!("toolu_1"));
        assert_eq!(v["content"], json!("the answer is 42"));
        assert!(v["content"].is_string());
        assert!(v.get("is_error").is_none());
    }

    #[test]
    fn error_tool_result_sets_is_error() {
        let c = Content::tool_result_text("toolu_2", "boom", true);
        let v = content_to_json(&c);
        assert_eq!(v["is_error"], json!(true));
    }

    #[test]
    fn image_tool_result_serializes_as_a_block_array() {
        let c = Content::ToolResult {
            tool_use_id: "toolu_3".into(),
            content: vec![
                ResultBlock::Text("screenshot attached".into()),
                ResultBlock::Image {
                    media_type: "image/png".into(),
                    data_base64: "AAAA".into(),
                },
            ],
            is_error: false,
        };
        let v = content_to_json(&c);
        let blocks = v["content"].as_array().expect("content should be an array");
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0],
            json!({ "type": "text", "text": "screenshot attached" })
        );
        assert_eq!(
            blocks[1],
            json!({
                "type": "image",
                "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" }
            })
        );
    }

    #[test]
    fn image_tool_result_with_error_keeps_is_error_at_top_level() {
        let c = Content::ToolResult {
            tool_use_id: "toolu_4".into(),
            content: vec![ResultBlock::Text("failed but here is context".into())],
            is_error: true,
        };
        let v = content_to_json(&c);
        // Single text block still serializes as a string, and is_error is set.
        assert!(v["content"].is_string());
        assert_eq!(v["is_error"], json!(true));
    }

    #[test]
    fn empty_tool_result_does_not_emit_an_empty_array() {
        let c = Content::ToolResult {
            tool_use_id: "toolu_5".into(),
            content: vec![],
            is_error: false,
        };
        let v = content_to_json(&c);
        // An empty array would be rejected by the API; we emit an empty string.
        assert_eq!(v["content"], json!(""));
    }

    #[test]
    fn caching_marks_the_system_block_and_the_last_tool() {
        use crate::providers::{ChatMessage, ToolDef};
        // Default (no RAAI_PROMPT_CACHE env) → caching on.
        let req = ChatRequest {
            model: "claude-opus-4-8".into(),
            system: Some("you are a helpful DAW assistant".into()),
            max_tokens: 1024,
            messages: vec![ChatMessage::user_text("hi")],
            tools: vec![
                ToolDef { name: "a".into(), description: "first".into(), input_schema: json!({"type":"object"}) },
                ToolDef { name: "b".into(), description: "second".into(), input_schema: json!({"type":"object"}) },
            ],
        };
        let body = build_request(&req, false);
        // System is the block-array form with a cache breakpoint (not a bare string).
        assert!(body["system"].is_array(), "system should be a block array when caching");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["system"][0]["text"], "you are a helpful DAW assistant");
        // Only the LAST tool carries the breakpoint (it caches the whole tools prefix).
        assert!(body["tools"][0].get("cache_control").is_none(), "first tool: no breakpoint");
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn thinking_flag_toggles_the_adaptive_thinking_param() {
        use crate::providers::{ChatMessage, ToolDef};
        let req = ChatRequest {
            model: "claude-opus-4-8".into(),
            system: Some("sys".into()),
            max_tokens: 1024,
            messages: vec![ChatMessage::user_text("hi")],
            tools: vec![ToolDef {
                name: "a".into(),
                description: "d".into(),
                input_schema: json!({"type":"object"}),
            }],
        };
        // Off → no thinking key at all (older models 400 on an explicit block).
        assert!(build_request(&req, false).get("thinking").is_none());
        // On → adaptive form (GA on 4.6+/Opus 4.8; budget_tokens is deprecated).
        assert_eq!(build_request(&req, true)["thinking"], json!({ "type": "adaptive" }));
    }

    #[test]
    fn thinking_blocks_are_dropped_from_history_when_thinking_is_off() {
        use crate::providers::{ChatMessage, Content, Role};
        // An assistant turn that a thinking-on run left behind: a thinking block
        // followed by its answer text.
        let assistant = ChatMessage {
            role: Role::Assistant,
            content: vec![
                Content::Thinking {
                    block: json!({ "type": "thinking", "thinking": "hmm", "signature": "sig" }),
                },
                Content::Text("answer".into()),
            ],
        };
        let req = ChatRequest {
            model: "claude-opus-4-8".into(),
            system: None,
            max_tokens: 256,
            messages: vec![ChatMessage::user_text("hi"), assistant],
            tools: vec![],
        };
        // Thinking OFF: the block is stripped, only the text survives — otherwise the
        // API 400s on a thinking block sent without thinking enabled.
        let off = build_request(&req, false);
        let content = off["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
        // Thinking ON: the block is preserved verbatim (with its signature), first.
        let on = build_request(&req, true);
        let content = on["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["signature"], "sig");
    }
}
