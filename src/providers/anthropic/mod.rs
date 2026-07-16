//! Native Anthropic Messages API adapter (streaming, reqwest + serde).
//! Endpoint POST https://api.anthropic.com/v1/messages, `anthropic-version: 2023-06-01`.

mod stream;
pub mod types;

use std::collections::HashMap;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::config;
use crate::providers::{
    Capabilities, ChatEvent, ChatRequest, LlmProvider, ProviderError, StopReason,
};
use types::{build_request, ContentBlock, Delta, StreamEvent};

const ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: reqwest::Client,
    /// The API key this instance sends. The worker builds one provider per key in
    /// the account's failover list; `None` falls back to config/env at send time.
    key: Option<String>,
    /// Whether to request extended thinking (per-provider setting).
    thinking: bool,
}

impl AnthropicProvider {
    /// Build with a specific key (a `None` falls back to the active account's
    /// resolved key / `ANTHROPIC_API_KEY` at send time) and the account's
    /// extended-thinking setting.
    pub fn with_key(key: Option<String>, thinking: bool) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            key,
            thinking,
        }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::with_key(None, false)
    }
}

/// Per-content-block state while streaming.
enum Block {
    Text,
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    /// An extended-thinking block: accumulates its text (for display + replay) and
    /// its signature (required verbatim on replay).
    Thinking {
        text: String,
        signature: String,
    },
    /// A `redacted_thinking` block — opaque encrypted `data`, replayed verbatim.
    Redacted {
        data: String,
    },
    Ignore,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_images: true,
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
            .or_else(config::api_key)
            .ok_or_else(|| ProviderError::MissingKey("ANTHROPIC_API_KEY".into()))?;
        let body = build_request(&req, self.thinking);

        let send = self
            .client
            .post(ENDPOINT)
            .header("x-api-key", key)
            .header("anthropic-version", API_VERSION)
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
        let mut parser = stream::SseParser::new();
        let mut blocks: HashMap<u32, Block> = HashMap::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage: Option<crate::providers::Usage> = None;

        loop {
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
                    for ev in parser.feed(&bytes) {
                        match ev {
                            StreamEvent::ContentBlockStart {
                                index,
                                content_block,
                            } => {
                                let block = match content_block {
                                    ContentBlock::ToolUse { id, name } => Block::ToolUse {
                                        id,
                                        name,
                                        json: String::new(),
                                    },
                                    ContentBlock::Text { .. } => Block::Text,
                                    ContentBlock::Thinking { signature } => Block::Thinking {
                                        text: String::new(),
                                        signature,
                                    },
                                    ContentBlock::RedactedThinking { data } => {
                                        Block::Redacted { data }
                                    }
                                    ContentBlock::Other => Block::Ignore,
                                };
                                blocks.insert(index, block);
                            }
                            StreamEvent::ContentBlockDelta { index, delta } => match delta {
                                Delta::TextDelta { text } => {
                                    if tx.send(ChatEvent::TextDelta(text)).await.is_err() {
                                        return Ok(());
                                    }
                                }
                                Delta::InputJsonDelta { partial_json } => {
                                    if let Some(Block::ToolUse { json, .. }) =
                                        blocks.get_mut(&index)
                                    {
                                        json.push_str(&partial_json);
                                    }
                                }
                                // Thinking text: show it live AND keep it for replay.
                                Delta::ThinkingDelta { thinking } => {
                                    if let Some(Block::Thinking { text, .. }) =
                                        blocks.get_mut(&index)
                                    {
                                        text.push_str(&thinking);
                                    }
                                    if tx.send(ChatEvent::ReasoningDelta(thinking)).await.is_err() {
                                        return Ok(());
                                    }
                                }
                                Delta::SignatureDelta { signature: s } => {
                                    if let Some(Block::Thinking { signature, .. }) =
                                        blocks.get_mut(&index)
                                    {
                                        signature.push_str(&s);
                                    }
                                }
                                _ => {}
                            },
                            StreamEvent::ContentBlockStop { index } => match blocks.remove(&index) {
                                Some(Block::ToolUse { id, name, json }) => {
                                    let input: Value =
                                        serde_json::from_str(&json).unwrap_or_else(|_| json!({}));
                                    if tx
                                        .send(ChatEvent::ToolCall {
                                            id,
                                            name,
                                            input,
                                            thought_signature: None,
                                        })
                                        .await
                                        .is_err()
                                    {
                                        return Ok(());
                                    }
                                }
                                // Emit the completed thinking block for the worker to
                                // store and replay VERBATIM (signature included).
                                Some(Block::Thinking { text, signature }) => {
                                    let block = json!({
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": signature,
                                    });
                                    if tx.send(ChatEvent::ThinkingBlock(block)).await.is_err() {
                                        return Ok(());
                                    }
                                }
                                Some(Block::Redacted { data }) => {
                                    let block = json!({ "type": "redacted_thinking", "data": data });
                                    if tx.send(ChatEvent::ThinkingBlock(block)).await.is_err() {
                                        return Ok(());
                                    }
                                }
                                _ => {}
                            },
                            StreamEvent::MessageDelta { delta, usage: u } => {
                                if let Some(sr) = delta.stop_reason {
                                    stop_reason = StopReason::from_wire(&sr);
                                }
                                if let Some(u) = u {
                                    usage = Some(u.into());
                                }
                            }
                            StreamEvent::Error { error } => {
                                let _ = tx
                                    .send(ChatEvent::Error(format!(
                                        "{}: {}",
                                        error.kind, error.message
                                    )))
                                    .await;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let _ = tx.send(ChatEvent::Done { stop_reason, usage }).await;
        Ok(())
    }
}
