//! The AI-core worker: a dedicated OS thread hosting a tokio runtime that runs
//! the agent loop (stream a turn -> execute tool calls on the main thread ->
//! feed results back -> repeat) and forwards output to the main thread. Never
//! touches the REAPER API or dialog directly.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::Sender as CbSender;
use serde_json::{json, Value};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::config;
use crate::providers::registry::{self, AdapterKind};
use crate::providers::{
    ChatEvent, ChatMessage, ChatRequest, Content, LlmProvider, ResultBlock, Role, StopReason,
};
use crate::tools::{self, ReaperOp, ToolOutcome};


/// True while a prompt is being processed. The main-thread pump reads this to
/// emit a periodic "still working" announcement (see `control_surface::run`).
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Bumped once per prompt so the pump can tell two back-to-back generations
/// apart (and reset its "still working" timer) even if it never samples the idle
/// gap between them.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Whether the worker is currently processing a prompt.
pub fn is_generating() -> bool {
    GENERATING.load(Ordering::Relaxed)
}

/// A monotonically increasing id for the current generation (see [`GENERATION`]).
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

/// Clears [`GENERATING`] when a prompt finishes, on every exit path.
struct GeneratingGuard;
impl Drop for GeneratingGuard {
    fn drop(&mut self) {
        GENERATING.store(false, Ordering::Relaxed);
    }
}

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
    let mut history: Vec<ChatMessage> = Vec::new();
    // Per-provider "which failover key are we on" cursor, remembered across
    // messages this session so an exhausted key isn't re-tried first every time.
    let mut key_cursor: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    while let Some(task) = task_rx.recv().await {
        match task {
            MainTask::Cancel => { /* nothing in flight */ }
            MainTask::Prompt(prompt) => {
                handle_prompt(
                    &mut history,
                    &ui_tx,
                    &mut task_rx,
                    &op_tx,
                    &mut key_cursor,
                    prompt,
                )
                .await;
            }
        }
    }
}

/// One accumulated model turn.
struct TurnResult {
    text: String,
    tool_calls: Vec<(String, String, Value, Option<String>)>, // (id, name, input, thought_signature)
    stop_reason: StopReason,
    aborted: bool, // cancelled or errored
    /// The provider error message, if the turn errored (not set on a plain
    /// cancel). The caller decides whether to show it or rotate to another key.
    error: Option<String>,
    /// Whether the error means THIS API key can't serve the request (rate-limit /
    /// quota / auth) — i.e. rotating to the next key may help. See
    /// [`ProviderError::is_key_exhausted`](crate::providers::ProviderError).
    key_exhausted: bool,
    /// The user cancelled this turn. Distinct from an error `aborted`: we must
    /// never rotate keys or show a provider error when the user asked to stop.
    cancelled: bool,
}

async fn handle_prompt(
    history: &mut Vec<ChatMessage>,
    ui_tx: &CbSender<UiEvent>,
    task_rx: &mut UnboundedReceiver<MainTask>,
    op_tx: &CbSender<ReaperOp>,
    key_cursor: &mut std::collections::HashMap<String, usize>,
    prompt: String,
) {
    // Resolve the active (default) provider account for this prompt, so switching
    // the default takes effect from the next message.
    let Some(cfg) = registry::active() else {
        let _ = ui_tx.send(UiEvent::UserMessage(prompt));
        let _ = ui_tx.send(UiEvent::Error(
            "No provider configured. Add one via Extensions -> ReaLackey.".into(),
        ));
        let _ = ui_tx.send(UiEvent::Status("No provider.".into()));
        let _ = ui_tx.send(UiEvent::Done);
        return;
    };
    if !cfg.can_send() {
        let _ = ui_tx.send(UiEvent::UserMessage(prompt));
        let msg = match cfg.kind {
            AdapterKind::Anthropic => format!(
                "No API key for \"{}\". Set it via Extensions -> ReaLackey \
                 (or the ANTHROPIC_API_KEY environment variable).",
                cfg.label
            ),
            AdapterKind::OpenAiCompatible => format!(
                "Provider \"{}\" needs an endpoint URL (and usually an API key).",
                cfg.label
            ),
        };
        let _ = ui_tx.send(UiEvent::Error(msg));
        let _ = ui_tx.send(UiEvent::Status("Not configured.".into()));
        let _ = ui_tx.send(UiEvent::Done);
        return;
    }
    // The account's ordered failover keys (empty = a keyless local server; one in
    // the common case). `key_idx` starts from the session cursor so we resume on
    // the last working key, clamped in case the list shrank since.
    let keys = registry::keys_for(&cfg.id);
    let mut key_idx = key_cursor
        .get(&cfg.id)
        .copied()
        .unwrap_or(0)
        .min(keys.len().saturating_sub(1));
    let key_at = |i: usize| keys.get(i).cloned();

    // Capabilities (and thus the tool set + prompt) are key-independent, so derive
    // them from one throwaway build; the per-turn attempt loop rebuilds per key.
    let caps = crate::providers::build_provider_with_key(&cfg, key_at(key_idx)).capabilities();

    let _ = ui_tx.send(UiEvent::UserMessage(prompt.clone()));
    let _ = ui_tx.send(UiEvent::Status("Thinking…".into()));

    // Mark generation active (the pump announces "still working" every ~10s while
    // this holds) and speak once, now, that work has started. The guard clears
    // the flag on every exit path (success, error, cancel).
    GENERATION.fetch_add(1, Ordering::Relaxed);
    GENERATING.store(true, Ordering::Relaxed);
    let _generating = GeneratingGuard;
    let _ = ui_tx.send(UiEvent::Announce("Working on it…".into()));

    history.push(ChatMessage::user_text(prompt));

    let tools = tools::definitions(caps.supports_images, caps.supports_audio);
    let cancel = CancellationToken::new();
    let mut final_answer = String::new();
    let mut truncated = false;
    // Approve applying changes ONCE per user request (not once per change): the
    // model reveals changes across turns, so a per-turn prompt still asks many
    // times. None = not yet asked; Some(v) = the user's decision for this request.
    let mut changes_decision: Option<bool> = None;

    let max_turns = config::max_turns(cfg.max_turns);
    for turn in 0..max_turns {
        let req = ChatRequest {
            model: cfg.model.clone(),
            system: Some(config::system_prompt(
                caps.supports_images,
                caps.supports_audio,
                crate::reaper::osara::is_running(),
            )),
            max_tokens: cfg.max_tokens,
            messages: history.clone(),
            tools: tools.clone(),
        };

        // Per-turn attempt loop: on a per-key limit (rate/quota/auth), rotate to
        // the next configured key and retry the SAME turn. `tried` is per-turn, so
        // each turn gets one full rotation before we give up.
        let mut tried = 0usize;
        let result = loop {
            let provider: Arc<dyn LlmProvider> =
                crate::providers::build_provider_with_key(&cfg, key_at(key_idx)).into();
            let r = run_turn(&provider, ui_tx, task_rx, &cancel, req.clone()).await;

            // Rotate only when the KEY is the problem, the user didn't cancel, we
            // have another key, and nothing was streamed yet (so a retry can't
            // duplicate partial output).
            if keys.len() > 1 && r.key_exhausted && r.text.is_empty() && !r.cancelled {
                let from = key_idx;
                key_idx = (key_idx + 1) % keys.len();
                key_cursor.insert(cfg.id.clone(), key_idx);
                tried += 1;
                if tried >= keys.len() {
                    // A full rotation and every key failed — all exhausted. Show
                    // the last error once and stop.
                    let last = r.error.clone().unwrap_or_else(|| "no key available".into());
                    let _ = ui_tx.send(UiEvent::Error(format!(
                        "All {} API keys for \"{}\" are exhausted or rejected. Last error: {last}",
                        keys.len(),
                        cfg.label
                    )));
                    break TurnResult {
                        aborted: true,
                        error: None, // already shown
                        ..r
                    };
                }
                let note = format!(
                    "API key #{} hit a limit or was rejected — switching to key #{} of {}.",
                    from + 1,
                    key_idx + 1,
                    keys.len()
                );
                let _ = ui_tx.send(UiEvent::Notice(note.clone()));
                let _ = ui_tx.send(UiEvent::Announce(note));
                continue; // retry the same turn with the next key
            }

            // Not a rotation. On success remember this key; on a non-rotating error
            // surface it (run_turn no longer emits provider errors inline). A cancel
            // shows nothing — run_turn already set the "Cancelled." status.
            if !r.aborted {
                key_cursor.insert(cfg.id.clone(), key_idx);
            } else if !r.cancelled {
                if let Some(err) = &r.error {
                    let _ = ui_tx.send(UiEvent::Error(err.clone()));
                }
            }
            break r;
        };

        // Record the assistant turn (text + tool_use blocks) in history. On an
        // aborted turn, skip the tool_use blocks so history never contains a
        // tool_use without a following tool_result (which would 400 next time).
        let mut content = Vec::new();
        if !result.text.is_empty() {
            content.push(Content::Text(result.text.clone()));
            final_answer = result.text.clone();
        }
        if !result.aborted {
            for (id, name, input, thought_signature) in &result.tool_calls {
                content.push(Content::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    thought_signature: thought_signature.clone(),
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
            // Ask for permission to apply changes ONCE per request — the first
            // turn that proposes a change — then remember the decision for the
            // rest of the request (further changes just proceed, each announced).
            if changes_decision.is_none()
                && config::confirmation_required()
                && result
                    .tool_calls
                    .iter()
                    .any(|(_, n, i, _)| tools::preview(n, i).is_some())
            {
                changes_decision =
                    Some(confirm_apply_changes(ui_tx, op_tx, &result.tool_calls).await);
            }
            let mutations_ok = changes_decision.unwrap_or(true);

            let mut results = Vec::new();
            for (id, name, input, _sig) in result.tool_calls {
                let input_pretty = serde_json::to_string_pretty(&input).unwrap_or_default();
                let _ = ui_tx.send(UiEvent::ToolStarted {
                    name: name.clone(),
                    input: input_pretty,
                });
                let outcome = run_tool(ui_tx, op_tx, &name, input, mutations_ok).await;
                let _ = ui_tx.send(UiEvent::ToolFinished {
                    is_error: outcome.is_error,
                    summary: truncate_summary(&outcome.content),
                });
                let ToolOutcome {
                    content,
                    is_error,
                    image,
                    audio,
                } = outcome;
                let result = if image.is_none() && audio.is_none() {
                    // Common case: a text-only result (byte-identical wire form).
                    Content::tool_result_text(id, content, is_error)
                } else {
                    // Media result: text + an image the model can see and/or an
                    // audio clip it can hear.
                    let mut blocks = vec![ResultBlock::Text(content)];
                    if let Some(img) = image {
                        blocks.push(ResultBlock::Image {
                            media_type: img.media_type,
                            data_base64: img.data_base64,
                        });
                    }
                    if let Some(au) = audio {
                        blocks.push(ResultBlock::Audio {
                            format: au.format,
                            data_base64: au.data_base64,
                        });
                    }
                    Content::ToolResult {
                        tool_use_id: id,
                        content: blocks,
                        is_error,
                    }
                };
                results.push(result);
            }
            history.push(ChatMessage {
                role: Role::User,
                content: results,
            });

            if turn + 1 == max_turns {
                // Not an error — the task just ran long (e.g. an iterative
                // capture→click→verify GUI session). History is preserved, so a
                // follow-up "continue" resumes exactly where this left off.
                let msg = format!(
                    "Paused after {max_turns} tool steps for this message. Say \"continue\" and \
                     I'll pick up where I left off (or raise RAAI_MAX_TURNS)."
                );
                let _ = ui_tx.send(UiEvent::Notice(msg.clone()));
                let _ = ui_tx.send(UiEvent::Announce(msg));
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
        error: None,
        key_exhausted: false,
        cancelled: false,
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
                Some(ChatEvent::ToolCall { id, name, input, thought_signature }) => {
                    out.tool_calls.push((id, name, input, thought_signature));
                }
                Some(ChatEvent::Done { stop_reason, .. }) => {
                    out.stop_reason = stop_reason;
                }
                Some(ChatEvent::Error(e)) => {
                    // Don't show it here: the caller decides whether to surface the
                    // error or quietly rotate to the next key.
                    out.error = Some(e);
                    out.aborted = true;
                }
                None => break, // provider finished and dropped the sender
            },
            t = task_rx.recv() => match t {
                Some(MainTask::Cancel) | None => {
                    cancel.cancel();
                    out.aborted = true;
                    out.cancelled = true;
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

    // The task's return carries the structured error (with an HTTP status), which
    // the channel `ChatEvent::Error` string can't. Use it to classify whether the
    // key is exhausted (rotate) vs. a genuine failure, and to fill the message if
    // the channel didn't. A plain cancel is not an error.
    if let Ok(Err(err)) = handle.await {
        if !matches!(err, crate::providers::ProviderError::Cancelled) {
            out.key_exhausted = err.is_key_exhausted();
            if out.error.is_none() {
                out.error = Some(err.to_string());
            }
            out.aborted = true;
        }
    }
    out
}

/// Run a tool. Mutating tools were already confirmed for the whole turn in one
/// batch (`mutations_ok`); screenshot/pixel tools keep their own per-use gates.
async fn run_tool(
    ui_tx: &CbSender<UiEvent>,
    op_tx: &CbSender<ReaperOp>,
    name: &str,
    input: Value,
    mutations_ok: bool,
) -> ToolOutcome {
    // Tier-B pixel input: arm-per-task consent. The user approves once, then the
    // assistant may click/drag in plugin windows (announced each time) until it
    // is disarmed. GUI clicks bypass REAPER's undo, so this is a distinct gate,
    // always enforced regardless of the mutation-confirm toggle.
    if tools::is_pixel_tool(name) {
        if !tools::is_pixel_armed() {
            let msg = "The assistant wants to operate a plugin's on-screen controls by \
                       synthesizing real mouse clicks/drags — for GUI-only controls that have no \
                       automatable parameter (e.g. a Kontakt mode switch). It will briefly move \
                       the mouse cursor inside the plugin window.\n\nIMPORTANT: these GUI actions \
                       CANNOT be undone by REAPER's undo.\n\nAllow the assistant to click in \
                       plugin windows for this session? (You can cancel any action, or tell it to \
                       stop.)";
            let approved = confirm(op_tx, msg.to_string()).await;
            if !approved {
                let _ = ui_tx.send(UiEvent::Notice("Pixel control declined.".into()));
                return ToolOutcome::ok(
                    json!({ "done": false, "reason": "user declined pixel control" }).to_string(),
                );
            }
            tools::arm_pixel_control();
            let _ = ui_tx.send(UiEvent::Notice(
                "Pixel control armed for this session.".into(),
            ));
            let _ = ui_tx.send(UiEvent::Announce("Pixel control armed.".into()));
        }
        // Announce each action so a synthesized click is never silent.
        let desc = pixel_action_desc(name, &input);
        let _ = ui_tx.send(UiEvent::Notice(desc.clone()));
        let _ = ui_tx.send(UiEvent::Announce(desc));
        return exec_tool(op_tx, name.to_string(), input).await;
    }

    // Sending a screenshot or an audio clip to the cloud is ALWAYS consent-gated
    // (data protection), independent of the mutation-confirm toggle, and asked
    // before the tool runs. These are not mutations, so the preview path below
    // never applies to them.
    if let Some(consent) = tools::consent_prompt(name, &input) {
        let _ = ui_tx.send(UiEvent::Notice(format!("{consent}?")));
        let _ = ui_tx.send(UiEvent::Announce(format!("{consent}. Allow?")));
        let approved = confirm(
            op_tx,
            format!("{consent}?\n\nThis will be sent to the cloud AI provider."),
        )
        .await;
        if !approved {
            let _ = ui_tx.send(UiEvent::Notice("Declined.".into()));
            return ToolOutcome::ok(
                json!({ "declined": true, "reason": "user declined" }).to_string(),
            );
        }
        return exec_tool(op_tx, name.to_string(), input).await;
    }

    // `preview` returns Some only for mutating tools. Their confirmation was
    // already handled for the whole turn as a single batch prompt, so here we
    // only honour a declined batch.
    if tools::preview(name, &input).is_some() && !mutations_ok {
        return ToolOutcome::ok(
            json!({ "applied": false, "reason": "user declined the change" }).to_string(),
        );
    }
    exec_tool(op_tx, name.to_string(), input).await
}

/// Ask the user ONCE per request whether the assistant may apply changes. Lists
/// the changes proposed so far; the model may make more this request, each
/// announced and undo-wrapped. The caller only invokes this when confirmation is
/// on and a change is actually proposed.
async fn confirm_apply_changes(
    ui_tx: &CbSender<UiEvent>,
    op_tx: &CbSender<ReaperOp>,
    calls: &[(String, String, Value, Option<String>)],
) -> bool {
    let previews: Vec<String> = calls
        .iter()
        .filter_map(|(_, name, input, _)| tools::preview(name, input))
        .collect();
    let list = previews
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{}. {}", i + 1, p))
        .collect::<Vec<_>>()
        .join("\n");
    let count = previews.len();
    let lead = if count == 1 {
        "The assistant wants to make this change".to_string()
    } else {
        format!("The assistant wants to make these {count} changes")
    };
    let _ = ui_tx.send(UiEvent::Notice(format!("{lead}:\n{list}")));
    let _ = ui_tx.send(UiEvent::Announce(format!(
        "{lead}, and possibly more for this request. Allow it to apply changes?"
    )));
    let approved = confirm(
        op_tx,
        format!(
            "{lead}:\n\n{list}\n\nIt may make further changes for this request; each is announced \
             and can be undone. Allow the assistant to apply changes?"
        ),
    )
    .await;
    let _ = ui_tx.send(UiEvent::Notice(
        if approved {
            "Applying changes."
        } else {
            "Declined."
        }
        .into(),
    ));
    approved
}

/// A short, spoken-friendly description of a pixel action for the announcement.
fn pixel_action_desc(name: &str, input: &Value) -> String {
    let n = |k: &str| input.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    match name {
        "plugin_click" => {
            let kind = if input
                .get("double")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Double-clicking"
            } else {
                "Clicking"
            };
            format!("{} the plugin at {}, {}.", kind, n("x"), n("y"))
        }
        "plugin_drag" => format!(
            "Dragging in the plugin from {}, {} to {}, {}.",
            n("x1"),
            n("y1"),
            n("x2"),
            n("y2")
        ),
        "plugin_type" => {
            let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let shown: String = text.chars().take(40).collect();
            format!("Typing into the plugin: {shown}")
        }
        "plugin_scroll" => format!("Scrolling the plugin at {}, {}.", n("x"), n("y")),
        _ => "Operating the plugin.".to_string(),
    }
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
    if op_tx
        .send(ReaperOp::Confirm { message, reply: tx })
        .is_err()
    {
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
    // Bounded wait: nearly all tools reply within a tick; the post-FX render is
    // the slow case (it renders synchronously on the main thread, window capped
    // at 30 s). A generous ceiling means a main-thread callback that never fires
    // can't hang the agent forever.
    match tokio::time::timeout(std::time::Duration::from_secs(90), reply_rx).await {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(_)) => ToolOutcome::error("{\"error\":\"no reply from main thread\"}"),
        Err(_) => ToolOutcome::error("{\"error\":\"the tool timed out\"}"),
    }
}
