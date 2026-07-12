//! Shared OpenAI-compatible `/chat/completions` adapter (streaming, tool calls).
//!
//! One adapter covers OpenAI, Groq, DeepSeek, xAI, OpenRouter, Google Gemini's
//! OpenAI-compatible endpoint, and local servers (Ollama, LM Studio) — only the
//! base URL, model and key differ (design §kap-llm). Raw REST/SSE via reqwest +
//! serde; no vendor SDK.
//!
//! Vision note: OpenAI's `role:"tool"` messages are text-only, so an image tool
//! result (our `capture_view`) can't ride in the tool reply the way Anthropic
//! allows. When the configured model supports vision we bridge it the standard
//! way: the tool reply stays text, and the image is appended as a FOLLOWING
//! `role:"user"` message with an `image_url` data-URI part (see `build_messages`).
//! `supports_images` is per-account (vision is per-model) and gates both this
//! bridge and whether `capture_view` is offered at all.

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::providers::{
    Capabilities, ChatEvent, ChatRequest, Content, LlmProvider, ProviderError, ResultBlock, Role,
    StopReason,
};

pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    /// Endpoint base, e.g. `https://api.openai.com/v1` (trailing slash trimmed).
    base_url: String,
    /// Optional bearer key (local servers are keyless).
    api_key: Option<String>,
    /// Whether the configured model accepts image input (per-account, per-model).
    supports_images: bool,
    /// Whether the configured model accepts audio input (per-account, per-model).
    supports_audio: bool,
}

impl OpenAiCompatProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        supports_images: bool,
        supports_audio: bool,
    ) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url,
            api_key,
            supports_images,
            supports_audio,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn id(&self) -> &'static str {
        "openai_compat"
    }

    fn capabilities(&self) -> Capabilities {
        // Vision/audio are per-model: the account config decides (image + audio
        // tool results are bridged into a following user message — see build_messages).
        Capabilities {
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
        }
    }

    async fn chat(
        &self,
        req: ChatRequest,
        tx: Sender<ChatEvent>,
        cancel: CancellationToken,
    ) -> Result<(), ProviderError> {
        let body = build_body(&req, self.supports_images, self.supports_audio);

        let mut builder = self
            .client
            .post(self.endpoint())
            .header("content-type", "application/json");
        if let Some(key) = &self.api_key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }
        let send = builder.json(&body).send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            r = send => r.map_err(|e| ProviderError::Http(e.to_string()))?,
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let msg = format!("API {status}: {text}");
            let _ = tx.send(ChatEvent::Error(msg.clone())).await;
            return Err(ProviderError::Http(msg));
        }

        let mut byte_stream = resp.bytes_stream();
        let mut sse = SseData::default();
        // Tool-call fragments accumulate by their streamed index until the turn ends.
        let mut tools: BTreeMap<u32, ToolAcc> = BTreeMap::new();
        let mut stop_reason = StopReason::EndTurn;

        'stream: loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                c = byte_stream.next() => c,
            };
            match chunk {
                None => break,
                Some(Err(e)) => {
                    let _ = tx.send(ChatEvent::Error(e.to_string())).await;
                    return Err(ProviderError::Http(e.to_string()));
                }
                Some(Ok(bytes)) => {
                    for payload in sse.feed(&bytes) {
                        if payload.trim() == "[DONE]" {
                            break 'stream;
                        }
                        // A streamed error object (some gateways send these inline).
                        if let Ok(err) = serde_json::from_str::<ErrorEnvelope>(&payload) {
                            if let Some(e) = err.error {
                                let _ = tx.send(ChatEvent::Error(e.message)).await;
                                continue;
                            }
                        }
                        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&payload) else {
                            continue; // ignore keep-alives / unknown shapes
                        };
                        for choice in chunk.choices {
                            if let Some(text) = choice.delta.content {
                                if !text.is_empty()
                                    && tx.send(ChatEvent::TextDelta(text)).await.is_err()
                                {
                                    return Ok(());
                                }
                            }
                            for tc in choice.delta.tool_calls.unwrap_or_default() {
                                let acc = tools.entry(tc.index).or_default();
                                if let Some(id) = tc.id {
                                    acc.id = id;
                                }
                                if let Some(f) = tc.function {
                                    if let Some(name) = f.name {
                                        acc.name = name;
                                    }
                                    if let Some(args) = f.arguments {
                                        acc.args.push_str(&args);
                                    }
                                }
                            }
                            if let Some(reason) = choice.finish_reason {
                                stop_reason = map_finish(&reason);
                            }
                        }
                    }
                }
            }
        }

        // Flush accumulated tool calls (in streamed index order).
        for (_, acc) in tools {
            if acc.name.is_empty() {
                continue;
            }
            let input: Value = if acc.args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&acc.args).unwrap_or_else(|_| json!({}))
            };
            if stop_reason == StopReason::EndTurn {
                stop_reason = StopReason::ToolUse;
            }
            if tx
                .send(ChatEvent::ToolCall {
                    id: acc.id,
                    name: acc.name,
                    input,
                })
                .await
                .is_err()
            {
                return Ok(());
            }
        }

        let _ = tx
            .send(ChatEvent::Done {
                stop_reason,
                usage: None,
            })
            .await;
        Ok(())
    }
}

#[derive(Default)]
struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

fn map_finish(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        _ => StopReason::Other,
    }
}

// ---- request building (neutral ChatRequest -> OpenAI body) -------------------

fn build_body(req: &ChatRequest, vision: bool, audio: bool) -> Value {
    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": true,
        "messages": build_messages(req, vision, audio),
    });
    if !req.tools.is_empty() {
        body["tools"] = json!(req
            .tools
            .iter()
            .map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            }))
            .collect::<Vec<_>>());
    }
    body
}

/// Flatten our messages into OpenAI's role model. A single neutral message can
/// expand to several: assistant tool calls become one assistant message with
/// `tool_calls`; each tool result becomes its own `role:"tool"` message. Because
/// tool messages are text-only, any image (when `vision`) or audio (when `audio`)
/// in a tool result is bridged into a trailing `role:"user"` message with
/// `image_url` / `input_audio` parts.
fn build_messages(req: &ChatRequest, vision: bool, audio: bool) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    if let Some(system) = &req.system {
        out.push(json!({ "role": "system", "content": system }));
    }
    for m in &req.messages {
        match m.role {
            Role::User => {
                let mut text = String::new();
                // Image/audio pulled out of tool results, to append AFTER all the
                // tool replies for this turn (OpenAI requires the tool messages to
                // directly answer the assistant's tool_calls; a following user
                // message carrying the media is then valid).
                let mut media_parts: Vec<Value> = Vec::new();
                for c in &m.content {
                    match c {
                        Content::Text(t) => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        // Tool results become their own `tool` message, keyed to
                        // the matching assistant tool call.
                        Content::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            out.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": result_blocks_to_text(content, vision, audio),
                            }));
                            for b in content {
                                match b {
                                    ResultBlock::Image {
                                        media_type,
                                        data_base64,
                                    } if vision => {
                                        media_parts.push(json!({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": format!("data:{media_type};base64,{data_base64}")
                                            }
                                        }));
                                    }
                                    // Audio uses a bare base64 `data` + `format`,
                                    // NOT a data: URI (unlike images).
                                    ResultBlock::Audio {
                                        format,
                                        data_base64,
                                    } if audio => {
                                        media_parts.push(json!({
                                            "type": "input_audio",
                                            "input_audio": { "data": data_base64, "format": format }
                                        }));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Content::ToolUse { .. } => {}
                    }
                }
                if !text.is_empty() {
                    out.push(json!({ "role": "user", "content": text }));
                }
                if !media_parts.is_empty() {
                    let mut parts = vec![json!({
                        "type": "text",
                        "text": "Media returned by the tool call(s) above:"
                    })];
                    parts.extend(media_parts);
                    out.push(json!({ "role": "user", "content": parts }));
                }
            }
            Role::Assistant => {
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for c in &m.content {
                    match c {
                        Content::Text(t) => text.push_str(t),
                        Content::ToolUse { id, name, input } => {
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                                }
                            }));
                        }
                        Content::ToolResult { .. } => {}
                    }
                }
                let mut msg = json!({ "role": "assistant" });
                // content must be present; null when the turn was tool-calls only.
                msg["content"] = if text.is_empty() {
                    Value::Null
                } else {
                    json!(text)
                };
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                out.push(msg);
            }
        }
    }
    out
}

/// OpenAI tool-result content is a plain string. Join text blocks; replace any
/// image/audio block with a textual note. When the capability is on the media is
/// delivered in a following user message (see build_messages); else it's dropped.
fn result_blocks_to_text(blocks: &[ResultBlock], vision: bool, audio: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        match b {
            ResultBlock::Text(t) => parts.push(t.clone()),
            ResultBlock::Image { .. } => parts.push(
                if vision {
                    "[image provided in the following message]".into()
                } else {
                    "[image omitted: this model has no vision]".into()
                },
            ),
            ResultBlock::Audio { .. } => parts.push(
                if audio {
                    "[audio provided in the following message]".into()
                } else {
                    "[audio omitted: this model has no audio input]".into()
                },
            ),
        }
    }
    parts.join("\n")
}

// ---- streaming (SSE) response ------------------------------------------------

/// Minimal SSE framer that yields each event's `data:` payload. Buffers raw
/// bytes and only splits complete lines, so a multibyte codepoint split across
/// network chunks is never corrupted.
#[derive(Default)]
struct SseData {
    buf: Vec<u8>,
    data: String,
}

impl SseData {
    fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(std::mem::take(&mut self.data));
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest);
            }
            // `:` comments / keep-alives and other field lines are ignored.
        }
        out
    }
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FnDelta>,
}

#[derive(Deserialize)]
struct FnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{ChatMessage, ToolDef};

    #[test]
    fn tool_round_trip_maps_to_openai_roles() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            system: Some("be terse".into()),
            max_tokens: 100,
            messages: vec![
                ChatMessage::user_text("hi"),
                ChatMessage {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "call_1".into(),
                        name: "get_tracks".into(),
                        input: json!({ "include_fx": true }),
                    }],
                },
                ChatMessage {
                    role: Role::User,
                    content: vec![Content::tool_result_text("call_1", "3 tracks", false)],
                },
            ],
            tools: vec![ToolDef {
                name: "get_tracks".into(),
                description: "list tracks".into(),
                input_schema: json!({ "type": "object" }),
            }],
        };
        let body = build_body(&req, false, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        // Assistant tool call -> tool_calls with a JSON-string arguments field.
        assert_eq!(msgs[2]["role"], "assistant");
        assert!(msgs[2]["content"].is_null());
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "get_tracks");
        assert!(msgs[2]["tool_calls"][0]["function"]["arguments"].is_string());
        // Tool result -> role "tool".
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_1");
        assert_eq!(msgs[3]["content"], "3 tracks");
        // Tools advertised in OpenAI function shape.
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_tracks");
    }

    /// Build a one-turn history where capture_view returned a text+image result.
    fn image_tool_req() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".into(),
            system: None,
            max_tokens: 100,
            messages: vec![
                ChatMessage {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "call_1".into(),
                        name: "capture_view".into(),
                        input: json!({}),
                    }],
                },
                ChatMessage {
                    role: Role::User,
                    content: vec![Content::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: vec![
                            ResultBlock::Text("captured".into()),
                            ResultBlock::Image {
                                media_type: "image/png".into(),
                                data_base64: "AAAA".into(),
                            },
                        ],
                        is_error: false,
                    }],
                },
            ],
            tools: vec![],
        }
    }

    #[test]
    fn image_tool_result_bridges_to_a_following_user_message() {
        let body = build_body(&image_tool_req(), true, false);
        let msgs = body["messages"].as_array().unwrap();
        // assistant tool call, then the (text-only) tool reply, then a user
        // message carrying the image as an image_url data-URI part.
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "call_1");
        assert!(msgs[1]["content"].as_str().unwrap().contains("captured"));
        assert!(msgs[1]["content"].is_string()); // tool content stays text
        assert_eq!(msgs[2]["role"], "user");
        let parts = msgs[2]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn image_tool_result_dropped_when_vision_off() {
        let body = build_body(&image_tool_req(), false, false);
        let msgs = body["messages"].as_array().unwrap();
        // No trailing user image message; the tool reply notes the omission.
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "tool");
        assert!(msgs[1]["content"].as_str().unwrap().contains("omitted"));
    }

    #[test]
    fn audio_tool_result_bridges_to_input_audio_part() {
        let req = ChatRequest {
            model: "gpt-audio".into(),
            system: None,
            max_tokens: 100,
            messages: vec![
                ChatMessage {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "call_1".into(),
                        name: "listen_to_audio".into(),
                        input: json!({}),
                    }],
                },
                ChatMessage {
                    role: Role::User,
                    content: vec![Content::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: vec![
                            ResultBlock::Text("rendered 5 s".into()),
                            ResultBlock::Audio {
                                format: "wav".into(),
                                data_base64: "QQQQ".into(),
                            },
                        ],
                        is_error: false,
                    }],
                },
            ],
            tools: vec![],
        };
        let body = build_body(&req, false, true);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "tool");
        assert!(msgs[1]["content"].is_string()); // tool content stays text
        let parts = msgs[2]["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "input_audio");
        assert_eq!(parts[1]["input_audio"]["data"], "QQQQ");
        assert_eq!(parts[1]["input_audio"]["format"], "wav");
        assert_eq!(msgs.len(), 3);
        // With audio off, the clip is dropped and the omission noted.
        let off = build_body(&req, false, false);
        let m = off["messages"].as_array().unwrap();
        assert_eq!(m.len(), 2);
        assert!(m[1]["content"].as_str().unwrap().contains("omitted"));
    }

    #[test]
    fn sse_framer_splits_data_payloads() {
        let mut s = SseData::default();
        let out = s.feed(b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(out, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn parses_a_streamed_text_then_tool_call() {
        // Two content deltas, then a tool call split across two argument chunks.
        let mut s = SseData::default();
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut text = String::new();
        let mut tools: BTreeMap<u32, ToolAcc> = BTreeMap::new();
        let mut done = false;
        for payload in s.feed(stream.as_bytes()) {
            if payload.trim() == "[DONE]" {
                done = true;
                break;
            }
            let chunk: ChatChunk = serde_json::from_str(&payload).unwrap();
            for choice in chunk.choices {
                if let Some(t) = choice.delta.content {
                    text.push_str(&t);
                }
                for tc in choice.delta.tool_calls.unwrap_or_default() {
                    let acc = tools.entry(tc.index).or_default();
                    if let Some(id) = tc.id {
                        acc.id = id;
                    }
                    if let Some(f) = tc.function {
                        if let Some(n) = f.name {
                            acc.name = n;
                        }
                        if let Some(a) = f.arguments {
                            acc.args.push_str(&a);
                        }
                    }
                }
            }
        }
        assert!(done);
        assert_eq!(text, "Hello");
        let acc = tools.get(&0).unwrap();
        assert_eq!(acc.id, "c1");
        assert_eq!(acc.name, "f");
        let v: Value = serde_json::from_str(&acc.args).unwrap();
        assert_eq!(v["x"], 1);
    }
}
