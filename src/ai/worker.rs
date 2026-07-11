//! The AI-core worker: a dedicated OS thread hosting a tokio runtime that runs
//! the agent loop (stream a turn -> execute tool calls on the main thread ->
//! feed results back -> repeat) and forwards output to the main thread. Never
//! touches the REAPER API or dialog directly.

use std::sync::Arc;

use crossbeam_channel::Sender as CbSender;
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::config;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::{
    ChatEvent, ChatMessage, ChatRequest, Content, LlmProvider, Role, StopReason,
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
        let _ = ui_tx.send(UiEvent::AssistantDelta(format!("\r\nYou: {prompt}\r\n")));
        let _ = ui_tx.send(UiEvent::Error(
            "No Anthropic API key set. Use Extensions -> REAPER AI Assistant -> \
             Set Anthropic API key (or set the ANTHROPIC_API_KEY environment variable)."
                .into(),
        ));
        let _ = ui_tx.send(UiEvent::Status("No API key.".into()));
        let _ = ui_tx.send(UiEvent::Done);
        return;
    }

    let _ = ui_tx.send(UiEvent::AssistantDelta(format!(
        "\r\nYou: {prompt}\r\nAssistant: "
    )));
    let _ = ui_tx.send(UiEvent::Status("Thinking…".into()));

    history.push(ChatMessage::user_text(prompt));

    let tools = tools::definitions();
    let cancel = CancellationToken::new();
    let mut final_answer = String::new();

    for turn in 0..MAX_TURNS {
        let req = ChatRequest {
            model: config::default_model(),
            system: Some(config::system_prompt()),
            max_tokens: 1024,
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
                let _ = ui_tx.send(UiEvent::AssistantDelta(format!("\r\n[tool: {name}]\r\n")));
                let outcome = exec_tool(op_tx, name, input).await;
                results.push(Content::ToolResult {
                    tool_use_id: id,
                    content: outcome.content,
                    is_error: outcome.is_error,
                });
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
        break;
    }

    if !final_answer.is_empty() {
        let _ = ui_tx.send(UiEvent::AssistantDelta("\r\n".into()));
        // Announce the final answer as one sense-unit (design §kap-a11y).
        let _ = ui_tx.send(UiEvent::Announce(final_answer));
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

    loop {
        tokio::select! {
            ev = ev_rx.recv() => match ev {
                Some(ChatEvent::TextDelta(d)) => {
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

/// Send a tool to the main thread for execution and await its result.
async fn exec_tool(op_tx: &CbSender<ReaperOp>, name: String, input: Value) -> ToolOutcome {
    let (reply_tx, reply_rx) = oneshot::channel();
    if op_tx
        .send(ReaperOp {
            name,
            input,
            reply: reply_tx,
        })
        .is_err()
    {
        return ToolOutcome {
            content: "{\"error\":\"main thread unavailable\"}".into(),
            is_error: true,
        };
    }
    reply_rx.await.unwrap_or(ToolOutcome {
        content: "{\"error\":\"no reply from main thread\"}".into(),
        is_error: true,
    })
}
