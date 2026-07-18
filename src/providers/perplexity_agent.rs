//! Perplexity **Agent API** adapter (`POST https://api.perplexity.ai/v1/agent`).
//!
//! Unlike the plain Sonar `/chat/completions` endpoint (which has no tool calling
//! at all), the Agent API speaks the OpenAI **Responses** protocol: a typed
//! `input` item list instead of `messages`, flat `function` tool objects, and
//! `function_call` / `function_call_output` items for the tool round-trip. It is
//! multi-provider (Sonar plus `openai/…`, `anthropic/…`, etc.) and always runs
//! with Perplexity's built-in `web_search` tool so answers are web-grounded while
//! the model drives ReaLackey's REAPER tools client-side.
//!
//! Streaming is SSE; each event's type lives in the JSON payload's `type` field
//! (the framer discards the `event:` line). We parse the payloads as untyped
//! `Value`s and match on `type`, which tolerates the API's many event kinds and
//! any future additions. Untested against the live API from the dev machine — the
//! mapping follows the documented Responses shape; validate with a real key.

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::providers::sse::SseFramer;
use crate::providers::{
    Capabilities, ChatEvent, ChatRequest, Content, LlmProvider, ProviderError, ResultBlock, Role,
    StopReason,
};

const ENDPOINT: &str = "https://api.perplexity.ai/v1/agent";

pub struct PerplexityAgentProvider {
    client: reqwest::Client,
    /// Bearer key. Required — the Agent API has no keyless mode.
    key: Option<String>,
    /// Attach Perplexity's built-in `web_search` tool so the model can ground its
    /// answers on live web results in the same loop it drives REAPER tools.
    web_search: bool,
}

impl PerplexityAgentProvider {
    pub fn with_key(key: Option<String>, web_search: bool) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            key,
            web_search,
        }
    }
}

impl Default for PerplexityAgentProvider {
    fn default() -> Self {
        Self::with_key(None, true)
    }
}

/// A function call being assembled from stream events, keyed by its output-item id.
#[derive(Default)]
struct ToolAcc {
    call_id: String,
    name: String,
    args: String,
}

#[async_trait]
impl LlmProvider for PerplexityAgentProvider {
    fn id(&self) -> &'static str {
        "perplexity_agent"
    }

    fn capabilities(&self) -> Capabilities {
        // v1: text + tools + web grounding. Image/audio input isn't bridged yet.
        Capabilities {
            supports_images: false,
            supports_audio: false,
        }
    }

    async fn chat(
        &self,
        req: ChatRequest,
        tx: Sender<ChatEvent>,
        cancel: CancellationToken,
    ) -> Result<(), ProviderError> {
        let key = self
            .key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| ProviderError::MissingKey("Perplexity API key".into()))?;
        let body = build_body(&req, self.web_search);

        let send = self
            .client
            .post(ENDPOINT)
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            r = send => r.map_err(|e| ProviderError::Http { status: None, message: e.to_string() })?,
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let msg = format!("API {status}: {text}");
            let _ = tx.send(ChatEvent::Error(msg.clone())).await;
            return Err(ProviderError::Http {
                status: Some(status.as_u16()),
                message: msg,
            });
        }

        let mut byte_stream = resp.bytes_stream();
        let mut sse = SseFramer::default();
        // Function calls assembled across events, keyed by output-item id. A BTreeMap
        // (not a HashMap) so the end-of-stream flush below is deterministic.
        let mut tools: BTreeMap<String, ToolAcc> = BTreeMap::new();
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
                    return Err(ProviderError::Http { status: None, message: e.to_string() });
                }
                Some(Ok(bytes)) => {
                    for payload in sse.feed(&bytes) {
                        if payload.trim() == "[DONE]" {
                            break 'stream;
                        }
                        let Ok(ev) = serde_json::from_str::<Value>(&payload) else {
                            continue; // ignore keep-alives / non-JSON frames
                        };
                        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                            // Answer text.
                            "response.output_text.delta" => {
                                if let Some(d) = ev.get("delta").and_then(|d| d.as_str()) {
                                    if !d.is_empty()
                                        && tx.send(ChatEvent::TextDelta(d.to_string())).await.is_err()
                                    {
                                        return Ok(());
                                    }
                                }
                            }
                            // Reasoning / thinking summary (display-only, like the
                            // OpenAI-compatible reasoning path).
                            "response.reasoning_text.delta"
                            | "response.reasoning_summary_text.delta" => {
                                if let Some(d) = ev.get("delta").and_then(|d| d.as_str()) {
                                    if !d.is_empty()
                                        && tx
                                            .send(ChatEvent::ReasoningDelta(d.to_string()))
                                            .await
                                            .is_err()
                                    {
                                        return Ok(());
                                    }
                                }
                            }
                            // A new output item begins — register function calls so
                            // their argument deltas have somewhere to accumulate.
                            "response.output_item.added" => {
                                if let Some(item) = ev.get("item") {
                                    if item.get("type").and_then(|t| t.as_str())
                                        == Some("function_call")
                                    {
                                        if let Some(id) = item_id(item) {
                                            let acc = tools.entry(id).or_default();
                                            if let Some(c) =
                                                item.get("call_id").and_then(|v| v.as_str())
                                            {
                                                acc.call_id = c.to_string();
                                            }
                                            if let Some(n) =
                                                item.get("name").and_then(|v| v.as_str())
                                            {
                                                acc.name = n.to_string();
                                            }
                                            if let Some(a) =
                                                item.get("arguments").and_then(|v| v.as_str())
                                            {
                                                acc.args.push_str(a);
                                            }
                                        }
                                    }
                                }
                            }
                            // Streamed function-call arguments.
                            "response.function_call_arguments.delta" => {
                                if let (Some(id), Some(d)) = (
                                    ev.get("item_id").and_then(|v| v.as_str()),
                                    ev.get("delta").and_then(|v| v.as_str()),
                                ) {
                                    tools.entry(id.to_string()).or_default().args.push_str(d);
                                }
                            }
                            // An output item finished. For a function call, emit it.
                            "response.output_item.done" => {
                                if let Some(item) = ev.get("item") {
                                    if item.get("type").and_then(|t| t.as_str())
                                        == Some("function_call")
                                    {
                                        let acc = item_id(item).and_then(|id| tools.remove(&id));
                                        if let Some(call) = finalize_call(item, acc) {
                                            stop_reason = StopReason::ToolUse;
                                            if tx.send(call).await.is_err() {
                                                return Ok(());
                                            }
                                        }
                                    }
                                }
                            }
                            // Terminal events.
                            "response.completed" | "response.done" => break 'stream,
                            "response.failed" | "response.error" | "error" => {
                                let msg = ev
                                    .pointer("/response/error/message")
                                    .or_else(|| ev.pointer("/error/message"))
                                    .or_else(|| ev.get("message"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("agent request failed")
                                    .to_string();
                                let _ = tx.send(ChatEvent::Error(msg)).await;
                            }
                            _ => {} // other event kinds (item.added parts, web_search, …) ignored
                        }
                    }
                }
            }
        }

        // Defensive flush: emit any function call that was opened (output_item.added
        // + argument deltas) but never closed by an output_item.done — e.g. if the
        // server ends at response.completed without a per-item done event. The
        // Responses spec always sends done (so this is normally empty), but this
        // adapter targets a non-OpenAI reimplementation, so flush rather than
        // silently drop the call. Mirrors the OpenAI-compatible adapter.
        for (_, acc) in std::mem::take(&mut tools) {
            if acc.name.is_empty() || acc.call_id.is_empty() {
                continue;
            }
            let input: Value = if acc.args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&acc.args).unwrap_or_else(|_| json!({}))
            };
            stop_reason = StopReason::ToolUse;
            if tx
                .send(ChatEvent::ToolCall {
                    id: acc.call_id,
                    name: acc.name,
                    input,
                    thought_signature: None,
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

/// The output-item id used to correlate `function_call_arguments.delta` events
/// (their `item_id`) with the item's `added` / `done` events (their `id`).
fn item_id(item: &Value) -> Option<String> {
    item.get("id").and_then(|v| v.as_str()).map(str::to_string)
}

/// Turn a completed `function_call` output item (plus any accumulated argument
/// deltas) into a [`ChatEvent::ToolCall`]. Prefers the item's own fields; falls
/// back to the accumulator for anything the terminal item left empty.
fn finalize_call(item: &Value, acc: Option<ToolAcc>) -> Option<ChatEvent> {
    let acc = acc.unwrap_or_default();
    // call_id links the eventual function_call_output back to this call.
    let call_id = item
        .get("call_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| (!acc.call_id.is_empty()).then_some(acc.call_id))
        // Last resort: the item id, so a result can still be paired.
        .or_else(|| item_id(item))?;
    let name = item
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| (!acc.name.is_empty()).then_some(acc.name))?;
    let args_str = item
        .get("arguments")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .unwrap_or(acc.args);
    let input: Value = if args_str.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&args_str).unwrap_or_else(|_| json!({}))
    };
    Some(ChatEvent::ToolCall {
        id: call_id,
        name,
        input,
        thought_signature: None,
    })
}

// ---- request building (neutral ChatRequest -> Responses body) ----------------

fn build_body(req: &ChatRequest, web_search: bool) -> Value {
    let mut body = json!({
        "model": req.model,
        "stream": true,
        "max_output_tokens": req.max_tokens,
        "input": build_input(req),
    });
    if let Some(system) = &req.system {
        // Responses API carries the system prompt as top-level `instructions`.
        body["instructions"] = json!(system);
    }
    let mut tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            })
        })
        .collect();
    if web_search {
        tools.push(json!({ "type": "web_search" }));
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    body
}

/// Flatten neutral messages into the Responses `input` item list: plain
/// `{role, content}` message items for text, `function_call` items for the
/// assistant's tool calls, and `function_call_output` items for tool results.
fn build_input(req: &ChatRequest) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for m in &req.messages {
        match m.role {
            Role::User => {
                let mut text = String::new();
                for c in &m.content {
                    match c {
                        Content::Text(t) => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        // Each tool result is its own item, keyed to the call_id of
                        // the assistant function_call it answers.
                        Content::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            out.push(json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": result_blocks_to_text(content),
                            }));
                        }
                        // Tool-use never appears on a user turn; thinking blocks are
                        // Anthropic-only.
                        Content::ToolUse { .. } | Content::Thinking { .. } => {}
                    }
                }
                if !text.is_empty() {
                    out.push(json!({ "role": "user", "content": text }));
                }
            }
            Role::Assistant => {
                for c in &m.content {
                    match c {
                        Content::Text(t) => {
                            if !t.is_empty() {
                                out.push(json!({ "role": "assistant", "content": t }));
                            }
                        }
                        Content::ToolUse {
                            id, name, input, ..
                        } => {
                            out.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(input)
                                    .unwrap_or_else(|_| "{}".into()),
                            }));
                        }
                        Content::ToolResult { .. } | Content::Thinking { .. } => {}
                    }
                }
            }
        }
    }
    out
}

/// A tool result's text. Image/audio blocks aren't sent to the Agent API (no
/// multimodal bridge yet); they become a short note so the model isn't confused
/// by a silently missing result.
fn result_blocks_to_text(blocks: &[ResultBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        match b {
            ResultBlock::Text(t) => parts.push(t.clone()),
            ResultBlock::Image { .. } => {
                parts.push("[image omitted: this provider has no vision]".into())
            }
            ResultBlock::Audio { .. } => {
                parts.push("[audio omitted: this provider has no audio input]".into())
            }
        }
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{ChatMessage, ToolDef};

    fn tool_req() -> ChatRequest {
        ChatRequest {
            model: "openai/gpt-5.1".into(),
            system: Some("be terse".into()),
            max_tokens: 256,
            messages: vec![
                ChatMessage::user_text("mute track 1"),
                ChatMessage {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "call_1".into(),
                        name: "set_mute".into(),
                        input: json!({ "track": 1, "mute": true }),
                        thought_signature: None,
                    }],
                },
                ChatMessage {
                    role: Role::User,
                    content: vec![Content::tool_result_text("call_1", "muted track 1", false)],
                },
            ],
            tools: vec![ToolDef {
                name: "set_mute".into(),
                description: "mute or unmute a track".into(),
                input_schema: json!({ "type": "object" }),
            }],
        }
    }

    #[test]
    fn build_body_uses_responses_shape() {
        let body = build_body(&tool_req(), true);
        // System -> instructions; max tokens -> max_output_tokens; streaming on.
        assert_eq!(body["instructions"], "be terse");
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "openai/gpt-5.1");
        // Function tool is FLAT (name/description/parameters at top level), not
        // nested under a "function" key like chat-completions.
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "set_mute");
        assert!(body["tools"][0]["parameters"].is_object());
        assert!(body["tools"][0].get("function").is_none());
        // web_search is appended as the last tool when grounding is on.
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.last().unwrap()["type"], "web_search");
    }

    #[test]
    fn web_search_omitted_when_disabled() {
        let body = build_body(&tool_req(), false);
        let has_web = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["type"] == "web_search");
        assert!(!has_web);
    }

    #[test]
    fn input_maps_tool_round_trip_to_responses_items() {
        let body = build_body(&tool_req(), false);
        let input = body["input"].as_array().unwrap();
        // user prompt -> message item
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "mute track 1");
        // assistant tool call -> function_call item with a JSON-string arguments
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["name"], "set_mute");
        assert!(input[1]["arguments"].is_string());
        // tool result -> function_call_output keyed by the same call_id
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["output"], "muted track 1");
    }

    #[test]
    fn finalize_call_prefers_item_then_accumulator() {
        // Complete item carries everything.
        let item = json!({
            "type": "function_call", "id": "fc_1", "call_id": "call_9",
            "name": "get_tracks", "arguments": "{\"include_fx\":true}"
        });
        let ChatEvent::ToolCall { id, name, input, .. } =
            finalize_call(&item, None).unwrap()
        else {
            panic!("expected a tool call");
        };
        assert_eq!(id, "call_9");
        assert_eq!(name, "get_tracks");
        assert_eq!(input["include_fx"], true);

        // Terminal item lacks arguments; fall back to what the deltas accumulated.
        let bare = json!({ "type": "function_call", "id": "fc_2", "call_id": "call_10", "name": "f" });
        let acc = ToolAcc {
            call_id: "call_10".into(),
            name: "f".into(),
            args: "{\"x\":1}".into(),
        };
        let ChatEvent::ToolCall { input, .. } = finalize_call(&bare, Some(acc)).unwrap() else {
            panic!("expected a tool call");
        };
        assert_eq!(input["x"], 1);
    }

    /// Drive the streaming event parser the way `chat()` does, over a synthetic
    /// Responses stream: reasoning, text, then a function call split across a
    /// `added` + two argument deltas + `done`.
    #[test]
    fn streaming_parses_text_and_a_split_function_call() {
        let stream = concat!(
            "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"hmm \"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Done\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"set_mute\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"track\\\":\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"1}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"set_mute\"}}\n\n",
            "data: {\"type\":\"response.completed\"}\n\n",
        );

        let mut sse = SseFramer::default();
        let mut tools: BTreeMap<String, ToolAcc> = BTreeMap::new();
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut call: Option<(String, String, Value)> = None;
        let mut completed = false;

        for payload in sse.feed(stream.as_bytes()) {
            let ev: Value = serde_json::from_str(&payload).unwrap();
            match ev["type"].as_str().unwrap() {
                "response.output_text.delta" => text.push_str(ev["delta"].as_str().unwrap()),
                "response.reasoning_text.delta" => {
                    reasoning.push_str(ev["delta"].as_str().unwrap())
                }
                "response.output_item.added" => {
                    let item = &ev["item"];
                    let acc = tools.entry(item_id(item).unwrap()).or_default();
                    acc.call_id = item["call_id"].as_str().unwrap().into();
                    acc.name = item["name"].as_str().unwrap().into();
                }
                "response.function_call_arguments.delta" => {
                    tools
                        .entry(ev["item_id"].as_str().unwrap().into())
                        .or_default()
                        .args
                        .push_str(ev["delta"].as_str().unwrap());
                }
                "response.output_item.done" => {
                    let item = &ev["item"];
                    let acc = item_id(item).and_then(|id| tools.remove(&id));
                    if let ChatEvent::ToolCall { id, name, input, .. } =
                        finalize_call(item, acc).unwrap()
                    {
                        call = Some((id, name, input));
                    }
                }
                "response.completed" => completed = true,
                _ => {}
            }
        }

        assert_eq!(text, "Done");
        assert_eq!(reasoning, "hmm ");
        assert!(completed);
        let (id, name, input) = call.expect("a function call");
        assert_eq!(id, "call_1");
        assert_eq!(name, "set_mute");
        // Arguments reassembled from the two deltas (the done item had none).
        assert_eq!(input["track"], 1);
    }

    /// A function call opened by `added` + argument deltas but NOT closed by an
    /// `output_item.done` (the stream just ends at `response.completed`) must still
    /// be flushed, not dropped. Exercises the end-of-stream flush logic.
    #[test]
    fn unclosed_function_call_is_flushed_at_end_of_stream() {
        let stream = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_9\",\"call_id\":\"call_9\",\"name\":\"get_tracks\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_9\",\"delta\":\"{\\\"fx\\\":true}\"}\n\n",
            "data: {\"type\":\"response.completed\"}\n\n",
        );
        let mut sse = SseFramer::default();
        let mut tools: BTreeMap<String, ToolAcc> = BTreeMap::new();
        for payload in sse.feed(stream.as_bytes()) {
            let ev: Value = serde_json::from_str(&payload).unwrap();
            match ev["type"].as_str().unwrap() {
                "response.output_item.added" => {
                    let item = &ev["item"];
                    let acc = tools.entry(item_id(item).unwrap()).or_default();
                    acc.call_id = item["call_id"].as_str().unwrap().into();
                    acc.name = item["name"].as_str().unwrap().into();
                }
                "response.function_call_arguments.delta" => {
                    tools
                        .entry(ev["item_id"].as_str().unwrap().into())
                        .or_default()
                        .args
                        .push_str(ev["delta"].as_str().unwrap());
                }
                _ => {}
            }
        }
        // No output_item.done fired, so the call is still pending — the flush emits it.
        let mut flushed: Vec<(String, String, Value)> = Vec::new();
        for (_, acc) in std::mem::take(&mut tools) {
            if acc.name.is_empty() || acc.call_id.is_empty() {
                continue;
            }
            let input: Value = serde_json::from_str(&acc.args).unwrap_or_else(|_| json!({}));
            flushed.push((acc.call_id, acc.name, input));
        }
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].0, "call_9");
        assert_eq!(flushed[0].1, "get_tracks");
        assert_eq!(flushed[0].2["fx"], true);
    }
}
