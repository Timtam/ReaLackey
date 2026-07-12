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
}

impl AnthropicProvider {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
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
        let key = config::api_key()
            .ok_or_else(|| ProviderError::MissingKey("ANTHROPIC_API_KEY".into()))?;
        let body = build_request(&req);

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
                    return Err(ProviderError::Http(e.to_string()));
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
                                _ => {}
                            },
                            StreamEvent::ContentBlockStop { index } => {
                                if let Some(Block::ToolUse { id, name, json }) =
                                    blocks.remove(&index)
                                {
                                    let input: Value =
                                        serde_json::from_str(&json).unwrap_or_else(|_| json!({}));
                                    if tx
                                        .send(ChatEvent::ToolCall { id, name, input })
                                        .await
                                        .is_err()
                                    {
                                        return Ok(());
                                    }
                                }
                            }
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
