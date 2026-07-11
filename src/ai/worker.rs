//! The AI-core worker: a dedicated OS thread hosting a tokio runtime that runs
//! the agent loop (stream a turn -> execute tool calls on the main thread ->
//! feed results back -> repeat) and forwards output to the main thread. Never
//! touches the REAPER API or dialog directly.

use std::sync::Arc;

use crossbeam_channel::Sender as CbSender;
use serde_json::{json, Value};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::config;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::{
    ChatEvent, ChatMessage, ChatRequest, Content, LlmProvider, ResultBlock, Role, StopReason,
};
use crate::tools::{self, ReaperOp, ToolOutcome};

/// Safety cap on tool-call iterations per user prompt.
const MAX_TURNS: usize = 8;

/// Spawn the worker on its own thread. Returns immediately.
pub fn spawn(
    task_rx: UnboundedReceiver<MainTask>,
    ui_tx: CbSender<UiEvent>,
    op_tx: CbSender<ReaperOp>,
) {
    let ui_tx_err = ui_tx.clone();
    let build = std::thread::Builder::new().name("raai-worker".into());
    let spawned = build.spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = ui_tx_err.send(UiEvent::Error(format!("Tokio runtime: {e}")));
                return;
            }
        };
        rt.block_on(run(task_rx, ui_tx, op_tx));
    });
    if let Err(e) = spawned {
        eprintln!("raai: failed to spawn worker thread: {e}");
    }
}

async fn run(
    mut task_rx: UnboundedReceiver<MainTask>,
    ui_tx: CbSender<UiEvent>,
    op_tx: CbSender<ReaperOp>,
) {
    let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new());
    let mut history: Vec<ChatMessage> = Vec::new();

    while let Some(task) = task_rx.recv().await {
        match task {
            MainTask::Cancel => { /* nothing in flight */ }
            MainTask::Prompt(prompt) => {
                handle_prompt(&provider, &mut history, &ui_tx, &mut task_rx, &op_tx, prompt).await;
            }
        }
    }
}

/// One accumulated model turn.
struct TurnResult {
    text: String,
    tool_calls: Vec<(String, String, Value)>, // (id, name, input)
    stop_reason: StopReason,
    aborted: bool, // cancelled or errored
}

async fn handle_prompt(
    provider: &Arc<dyn LlmProvider>,
    history: &mut Vec<ChatMessage>,
    ui_tx: &CbSender<UiEvent>,
    task_rx: &mut UnboundedReceiver<MainTask>,
    op_tx: &CbSender<ReaperOp>,
    prompt: String,
) {
    if !config::has_api_key() {
        let _ = ui_tx.send(UiEvent::UserMessage(prompt));
        let _ = ui_tx.send(UiEvent::Error(
            "No Anthropic API key set. Use Extensions -> REAPER AI Assistant -> \
             Set Anthropic API key (or set the ANTHROPIC_API_KEY environment variable)."
                .into(),
        ));
        let _ = ui_tx.send(UiEvent::Status("No API key.".into()));
        let _ = ui_tx.send(UiEvent::Done);
        return;
    }

    let _ = ui_tx.send(UiEvent::UserMessage(prompt.clone()));
    let _ = ui_tx.send(UiEvent::Status("Thinking…".into()));

    history.push(ChatMessage::user_text(prompt));

    let tools = tools::definitions();
    let cancel = CancellationToken::new();
    let mut final_answer = String::new();
    let mut truncated = false;

    for turn in 0..MAX_TURNS {
        let req = ChatRequest {
            model: config::default_model(),
            system: Some(config::system_prompt()),
            max_tokens: config::max_output_tokens(),
            messages: history.clone(),
            tools: tools.clone(),
        };

        let result = run_turn(provider, ui_tx, task_rx, &cancel, req).await;

        // Record the assistant turn (text + tool_use blocks) in history. On an
        // aborted turn, skip the tool_use blocks so history never contains a
        // tool_use without a following tool_result (which would 400 next time).
        let mut content = Vec::new();
        if !result.text.is_empty() {
            content.push(Content::Text(result.text.clone()));
            final_answer = result.text.clone();
        }
        if !result.aborted {
            for (id, name, input) in &result.tool_calls {
                content.push(Content::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
        }
        if !content.is_empty() {
            history.push(ChatMessage {
                role: Role::Assistant,
                content,
            });
        }

        if result.aborted {
            break;
        }

        // Continue the loop only if the model asked for tools.
        if result.stop_reason == StopReason::ToolUse && !result.tool_calls.is_empty() {
            let mut results = Vec::new();
            for (id, name, input) in result.tool_calls {
                let input_pretty = serde_json::to_string_pretty(&input).unwrap_or_default();
                let _ = ui_tx.send(UiEvent::ToolStarted {
                    name: name.clone(),
                    input: input_pretty,
                });
                let outcome = run_tool(ui_tx, op_tx, &name, input).await;
                let _ = ui_tx.send(UiEvent::ToolFinished {
                    is_error: outcome.is_error,
                    summary: truncate_summary(&outcome.content),
                });
                let ToolOutcome {
                    content,
                    is_error,
                    image,
                } = outcome;
                let result = match image {
                    // Common case: a text-only result (byte-identical wire form).
                    None => Content::tool_result_text(id, content, is_error),
                    // Vision: text + an image block the model can see.
                    Some(img) => Content::ToolResult {
                        tool_use_id: id,
                        content: vec![
                            ResultBlock::Text(content),
                            ResultBlock::Image {
                                media_type: img.media_type,
                                data_base64: img.data_base64,
                            },
                        ],
                        is_error,
                    },
                };
                results.push(result);
            }
            history.push(ChatMessage {
                role: Role::User,
                content: results,
            });

            if turn + 1 == MAX_TURNS {
                let _ = ui_tx.send(UiEvent::Error("Reached the tool-call limit.".into()));
            }
            continue;
        }
        // A non-tool turn is the final answer. If the model ran into the output
        // limit, the text ends mid-thought — flag it so we don't stop silently.
        truncated = result.stop_reason == StopReason::MaxTokens;
        break;
    }

    if !final_answer.is_empty() {
        // Announce the final answer as one sense-unit (design §kap-a11y), with
        // Markdown stripped so the screen reader speaks prose, not "hash"/"star".
        let mut spoken = crate::text::strip_markdown(&final_answer);
        if truncated {
            spoken.push_str("\n\nNote: this response was cut off at the length limit.");
        }
        if !spoken.trim().is_empty() {
            let _ = ui_tx.send(UiEvent::Announce(spoken));
        }
    }
    if truncated {
        // Visible marker in the pane too, so a cut-off answer never looks complete.
        let _ = ui_tx.send(UiEvent::Notice(
            "Response cut off at the length limit — ask me to continue for the rest.".into(),
        ));
    }
    let _ = ui_tx.send(UiEvent::Status("Ready.".into()));
    let _ = ui_tx.send(UiEvent::Done);
}

/// Run one streaming model turn, forwarding text to the UI and collecting tool
/// calls. Watches `task_rx` so a Cancel aborts promptly.
async fn run_turn(
    provider: &Arc<dyn LlmProvider>,
    ui_tx: &CbSender<UiEvent>,
    task_rx: &mut UnboundedReceiver<MainTask>,
    cancel: &CancellationToken,
    req: ChatRequest,
) -> TurnResult {
    let (ev_tx, mut ev_rx) = mpsc::channel::<ChatEvent>(64);
    let p = provider.clone();
    let c = cancel.clone();
    let handle = tokio::spawn(async move { p.chat(req, ev_tx, c).await });

    let mut out = TurnResult {
        text: String::new(),
        tool_calls: Vec::new(),
        stop_reason: StopReason::EndTurn,
        aborted: false,
    };
    let mut assistant_started = false;

    loop {
        tokio::select! {
            ev = ev_rx.recv() => match ev {
                Some(ChatEvent::TextDelta(d)) => {
                    if !assistant_started {
                        assistant_started = true;
                        let _ = ui_tx.send(UiEvent::AssistantStart);
                    }
                    out.text.push_str(&d);
                    let _ = ui_tx.send(UiEvent::AssistantDelta(d));
                }
                Some(ChatEvent::ToolCall { id, name, input }) => {
                    out.tool_calls.push((id, name, input));
                }
                Some(ChatEvent::Done { stop_reason, .. }) => {
                    out.stop_reason = stop_reason;
                }
                Some(ChatEvent::Error(e)) => {
                    let _ = ui_tx.send(UiEvent::Error(e));
                    out.aborted = true;
                }
                None => break, // provider finished and dropped the sender
            },
            t = task_rx.recv() => match t {
                Some(MainTask::Cancel) | None => {
                    cancel.cancel();
                    out.aborted = true;
                    let _ = ui_tx.send(UiEvent::Status("Cancelled.".into()));
                }
                Some(MainTask::Prompt(_)) => {
                    // Phase 0/1: one generation at a time.
                    let _ = ui_tx.send(UiEvent::Status(
                        "Please wait until the current answer is finished…".into(),
                    ));
                }
            },
        }
    }

    let _ = handle.await;
    out
}

/// Run a tool, gating mutating tools behind a user confirmation (default on).
async fn run_tool(
    ui_tx: &CbSender<UiEvent>,
    op_tx: &CbSender<ReaperOp>,
    name: &str,
    input: Value,
) -> ToolOutcome {
    // Screen capture sends pixels to the cloud, so it is ALWAYS consent-gated
    // (data protection), independent of the mutation-confirm toggle, and asked
    // before the tool runs. capture_view is not a mutation, so the preview path
    // below never applies to it.
    if let Some(consent) = tools::consent_prompt(name, &input) {
        let _ = ui_tx.send(UiEvent::Notice(format!("{consent}?")));
        let _ = ui_tx.send(UiEvent::Announce(format!("{consent}. Allow?")));
        let approved = confirm(
            op_tx,
            format!("{consent}?\n\nThe screenshot will be sent to the cloud AI provider."),
        )
        .await;
        if !approved {
            let _ = ui_tx.send(UiEvent::Notice("Screenshot declined.".into()));
            return ToolOutcome::ok(
                json!({ "captured": false, "reason": "user declined the screenshot" }).to_string(),
            );
        }
        return exec_tool(op_tx, name.to_string(), input).await;
    }

    // `preview` returns Some only for mutating tools; those require confirmation.
    if let Some(preview) = tools::preview(name, &input) {
        if config::confirmation_required() {
            let _ = ui_tx.send(UiEvent::Notice(format!("Proposed change: {preview}")));
            let _ = ui_tx.send(UiEvent::Announce(format!("Proposed change: {preview}. Confirm?")));
            let confirmed = confirm(
                op_tx,
                format!("The assistant proposes this change:\n\n{preview}\n\nApply it?"),
            )
            .await;
            if !confirmed {
                let _ = ui_tx.send(UiEvent::Notice("Declined.".into()));
                return ToolOutcome::ok(
                    json!({ "applied": false, "reason": "user declined the change" }).to_string(),
                );
            }
            let outcome = exec_tool(op_tx, name.to_string(), input).await;
            let _ = ui_tx.send(UiEvent::Notice("Applied.".into()));
            return outcome;
        }
    }
    exec_tool(op_tx, name.to_string(), input).await
}

/// Cap a tool result shown in the UI card (the full result still goes to the model).
fn truncate_summary(s: &str) -> String {
    const MAX: usize = 4000;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut t: String = s.chars().take(MAX).collect();
    t.push_str("\n… (truncated)");
    t
}

/// Ask the user to confirm a change via a native message box (main thread).
async fn confirm(op_tx: &CbSender<ReaperOp>, message: String) -> bool {
    let (tx, rx) = oneshot::channel();
    if op_tx.send(ReaperOp::Confirm { message, reply: tx }).is_err() {
        return false;
    }
    rx.await.unwrap_or(false)
}

/// Send a tool to the main thread for execution and await its result.
async fn exec_tool(op_tx: &CbSender<ReaperOp>, name: String, input: Value) -> ToolOutcome {
    let (reply_tx, reply_rx) = oneshot::channel();
    if op_tx
        .send(ReaperOp::Tool {
            name,
            input,
            reply: reply_tx,
        })
        .is_err()
    {
        return ToolOutcome::error("{\"error\":\"main thread unavailable\"}");
    }
    reply_rx
        .await
        .unwrap_or_else(|_| ToolOutcome::error("{\"error\":\"no reply from main thread\"}"))
}
