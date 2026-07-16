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
    // Progressive tool disclosure (opt-in): the tools loaded on demand this
    // SESSION. Persisted across messages so a category loaded for one message
    // stays available for the next — otherwise every message would re-pay a
    // load_tools hop for tools it already discovered.
    let mut active_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

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
                    &mut active_tools,
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

/// How many most-recent media-bearing tool results keep their inline base64 live.
/// Older ones are evicted to a text placeholder. Tunable via `RAAI_MEDIA_KEEP`
/// (0 = evict all past media immediately; capped at 20).
fn media_keep_recent() -> usize {
    std::env::var("RAAI_MEDIA_KEEP")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .min(20)
}

const IMG_EVICTED: &str =
    "[image from an earlier step, dropped to save tokens — call capture_view again if you need it]";
const AUDIO_EVICTED: &str =
    "[audio clip from an earlier step, dropped to save tokens — call listen_to_audio again if needed]";

/// Replace inline image/audio base64 in older history messages with a short text
/// placeholder, keeping the `keep_recent` most recent media-bearing tool results
/// intact. Mutates in place (permanent) and is idempotent — an already-evicted
/// block is plain text and is skipped. A media-bearing result carrying several
/// blocks (e.g. a video clip's frames) counts as ONE and is kept or evicted whole.
fn evict_stale_media(history: &mut [ChatMessage], keep_recent: usize) {
    let has_media = |m: &ChatMessage| {
        m.content.iter().any(|c| {
            matches!(c, Content::ToolResult { content, .. }
                if content.iter().any(|b| matches!(b, ResultBlock::Image { .. } | ResultBlock::Audio { .. })))
        })
    };
    let mut seen = 0usize;
    for msg in history.iter_mut().rev() {
        if !has_media(msg) {
            continue;
        }
        seen += 1;
        if seen <= keep_recent {
            continue; // keep the most recent captures live
        }
        for c in &mut msg.content {
            if let Content::ToolResult { content, .. } = c {
                for b in content.iter_mut() {
                    match b {
                        ResultBlock::Image { .. } => *b = ResultBlock::Text(IMG_EVICTED.into()),
                        ResultBlock::Audio { .. } => *b = ResultBlock::Text(AUDIO_EVICTED.into()),
                        ResultBlock::Text(_) => {}
                    }
                }
            }
        }
    }
}

/// Handle the `load_tools` meta-tool (progressive disclosure): activate the tools
/// matching the query for the rest of the SESSION and report them to the model.
/// Pure worker-side state — no REAPER access and no consent (it only widens the
/// offered set; each tool keeps its own gates when actually called).
fn handle_load_tools(
    input: &Value,
    active: &mut std::collections::HashSet<String>,
    supports_images: bool,
    supports_audio: bool,
) -> ToolOutcome {
    let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("").trim();
    if query.is_empty() {
        return ToolOutcome::error(
            json!({ "error": "provide a 'query' describing the capability you need" }).to_string(),
        );
    }
    let matches = tools::find_matching_tools(query, supports_images, supports_audio);
    if matches.is_empty() {
        return ToolOutcome::ok(
            json!({
                "loaded": [],
                "note": format!("No tools matched \"{query}\". Try a broader capability query, or use run_action for a REAPER action."),
            })
            .to_string(),
        );
    }
    let loaded: Vec<Value> = matches
        .iter()
        .map(|d| {
            active.insert(d.name.clone());
            json!({ "name": d.name, "summary": first_sentence(&d.description) })
        })
        .collect();
    ToolOutcome::ok(
        json!({
            "loaded": loaded,
            "note": "These tools are now available — call them on your NEXT turn (they were not in this turn's tool list).",
        })
        .to_string(),
    )
}

/// First sentence (or ~160 bytes) of a description, for the load_tools listing.
fn first_sentence(desc: &str) -> String {
    let mut end = desc.find(". ").map(|i| i + 1).unwrap_or(desc.len()).min(160);
    while end > 0 && !desc.is_char_boundary(end) {
        end -= 1;
    }
    desc[..end].trim().to_string()
}

async fn handle_prompt(
    history: &mut Vec<ChatMessage>,
    ui_tx: &CbSender<UiEvent>,
    task_rx: &mut UnboundedReceiver<MainTask>,
    op_tx: &CbSender<ReaperOp>,
    key_cursor: &mut std::collections::HashMap<String, usize>,
    active_tools: &mut std::collections::HashSet<String>,
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

    // Progressive tool disclosure (opt-in): pre-load likely tools from the prompt
    // text with NO round-trip, so common tasks don't spend a load_tools hop.
    let progressive = tools::progressive_enabled();
    if progressive {
        for name in tools::preseed_from_prompt(&prompt, caps.supports_images, caps.supports_audio) {
            active_tools.insert(name);
        }
    }
    history.push(ChatMessage::user_text(prompt));

    // Non-progressive: the full tool set (identical every turn). Progressive: the
    // tools array is rebuilt per turn from CORE + the session's loaded set.
    let all_tools = tools::definitions(caps.supports_images, caps.supports_audio);
    let cancel = CancellationToken::new();
    let mut final_answer = String::new();
    let mut truncated = false;
    // Whether the model ran any tool this request, and whether the loop ended with
    // a clean (non-aborted, non-tool) final turn — used to avoid finishing SILENTLY
    // when the model returns nothing (common with local Ollama models that don't
    // reliably do tool use: the request succeeds but yields no text and no tools).
    let mut did_tool_work = false;
    let mut clean_finish = false;
    // Approve applying changes ONCE per user request (not once per change): the
    // model reveals changes across turns, so a per-turn prompt still asks many
    // times. None = not yet asked; Some(v) = the user's decision for this request.
    let mut changes_decision: Option<bool> = None;

    let max_turns = config::max_turns(cfg.max_turns);
    for turn in 0..max_turns {
        // Media (screenshots / audio / video-clip frames) is inline base64 in the
        // history and would otherwise be re-uploaded on every later turn forever.
        // Keep only the most recent captures live; replace older ones with a text
        // placeholder so they stop costing tokens. Permanent + idempotent.
        evict_stale_media(&mut *history, media_keep_recent());
        // In progressive mode the offered set can grow across turns (after a
        // load_tools call), so rebuild it each turn from the session's active set.
        let turn_tools = if progressive {
            tools::core_and_active(active_tools, caps.supports_images, caps.supports_audio)
        } else {
            all_tools.clone()
        };
        let req = ChatRequest {
            model: cfg.model.clone(),
            system: Some(config::system_prompt(
                caps.supports_images,
                caps.supports_audio,
                crate::reaper::osara::is_running(),
            )),
            max_tokens: cfg.max_tokens,
            messages: history.clone(),
            tools: turn_tools,
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
            did_tool_work = true;
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
                let outcome = if progressive && name == "load_tools" {
                    // Worker-side: activate the matching tools for the rest of the
                    // session (no REAPER op, no consent — it only widens the offered
                    // set). They become callable on the NEXT request.
                    handle_load_tools(&input, active_tools, caps.supports_images, caps.supports_audio)
                } else {
                    run_tool(
                        ui_tx,
                        op_tx,
                        task_rx,
                        &name,
                        input,
                        mutations_ok,
                        caps.supports_audio,
                    )
                    .await
                };
                let _ = ui_tx.send(UiEvent::ToolFinished {
                    is_error: outcome.is_error,
                    summary: truncate_summary(&outcome.content),
                });
                let ToolOutcome {
                    content,
                    is_error,
                    images,
                    audio,
                } = outcome;
                let result = if images.is_empty() && audio.is_none() {
                    // Common case: a text-only result (byte-identical wire form).
                    Content::tool_result_text(id, content, is_error)
                } else {
                    // Media result: text + one or more images the model can see
                    // (a video clip is several frames) and/or an audio clip.
                    let mut blocks = vec![ResultBlock::Text(content)];
                    for img in images {
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
        clean_finish = true;
        break;
    }

    // The model finished cleanly but said nothing (no text, no tools) — don't go
    // silent to "Ready.", which reads as a crash. Tell the user what happened.
    // (Aborts/errors/cancels already showed a message; the max-turns case showed a
    // "paused" notice; both leave clean_finish false, so this only fires on a
    // genuinely empty answer.)
    if clean_finish && final_answer.trim().is_empty() {
        let msg = if did_tool_work {
            "The model ran the actions above but ended without a final message — they may have \
             completed; check the result, or ask me to summarise."
                .to_string()
        } else {
            "The model returned an empty response — no answer and no action taken. Local models \
             sometimes do this when they don't support tool use for a request, or the prompt is \
             too large. Try rephrasing, or a model with function-calling support."
                .to_string()
        };
        let _ = ui_tx.send(UiEvent::Notice(msg.clone()));
        let _ = ui_tx.send(UiEvent::Announce(msg));
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
                // Reasoning is shown in its own block; it is NOT part of the final
                // answer (not stored in history, not spoken as the answer).
                Some(ChatEvent::ReasoningDelta(d)) => {
                    let _ = ui_tx.send(UiEvent::ReasoningDelta(d));
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
    task_rx: &mut UnboundedReceiver<MainTask>,
    name: &str,
    input: Value,
    mutations_ok: bool,
    supports_audio: bool,
) -> ToolOutcome {
    // A video clip is a multi-step capture (seek -> settle -> grab, per frame) that
    // must yield the main thread between frames so the Video window re-renders — a
    // single main-thread op can't. It's orchestrated here in the worker instead.
    if name == "capture_video_clip" {
        return capture_video_clip(ui_tx, op_tx, task_rx, input, supports_audio).await;
    }

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

/// Orchestrate a `capture_video_clip`: one consent, then per frame seek the edit
/// cursor and — after a short `sleep` that frees REAPER's main thread so the Video
/// window re-renders — capture that frame; finally render the span's audio and
/// restore the cursor. Runs in the worker (async): the between-frame sleeps are the
/// whole point, since a single main-thread op couldn't yield for the re-render. The
/// internal sub-ops (`__video_begin`, `set_edit_cursor`, `capture_view`,
/// `listen_to_audio`) are dispatched directly via `exec_tool`, which bypasses their
/// own per-call consent — the one consent above already covers the whole clip.
async fn capture_video_clip(
    ui_tx: &CbSender<UiEvent>,
    op_tx: &CbSender<ReaperOp>,
    task_rx: &mut UnboundedReceiver<MainTask>,
    input: Value,
    supports_audio: bool,
) -> ToolOutcome {
    // One consent for the whole clip (frames + audio are sent to the cloud).
    let consent = tools::consent_prompt("capture_video_clip", &input)
        .unwrap_or_else(|| "The assistant wants to capture a short video clip".to_string());
    let _ = ui_tx.send(UiEvent::Notice(format!("{consent}?")));
    let _ = ui_tx.send(UiEvent::Announce(format!("{consent}. Allow?")));
    if !confirm(
        op_tx,
        format!("{consent}?\n\nThis will be sent to the cloud AI provider."),
    )
    .await
    {
        let _ = ui_tx.send(UiEvent::Notice("Declined.".into()));
        return ToolOutcome::ok(json!({ "declined": true, "reason": "user declined" }).to_string());
    }

    let frames = input
        .get("frames")
        .and_then(|v| v.as_u64())
        .unwrap_or(6)
        .clamp(2, 12) as usize;
    let want_audio = supports_audio
        && input
            .get("include_audio")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
    // Settle after each seek before capturing, so the Video window has re-rendered
    // to the new cursor. Tunable on-device (a heavy video-FX chain may need longer).
    let settle_ms = std::env::var("RAAI_VIDEO_SETTLE_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(250)
        .clamp(0, 5000);

    // Resolve the range + original cursor and validate the Video window (one op).
    let begin = exec_tool(op_tx, "__video_begin".to_string(), input.clone()).await;
    if begin.is_error {
        return begin; // e.g. Video window closed, or no range/time-selection
    }
    let b: Value = serde_json::from_str(&begin.content).unwrap_or_default();
    let t0 = b["t0"].as_f64().unwrap_or(0.0);
    let t1 = b["t1"].as_f64().unwrap_or(t0);
    let orig = b["orig_cursor"].as_f64().unwrap_or(t0);
    let span = (t1 - t0).max(0.0);
    let times: Vec<f64> = if frames <= 1 || span <= 0.0 {
        vec![t0]
    } else {
        (0..frames)
            .map(|k| t0 + span * (k as f64) / ((frames - 1) as f64))
            .collect()
    };

    // Restore the edit cursor to where the user had it (best-effort, on every exit).
    let restore_cursor =
        json!({ "position": orig, "seek_playback": false, "move_view": false });

    // Per frame: seek -> settle (main thread free, video re-renders) -> capture.
    // Each captured frame is paired with its ACTUAL timestamp, so a frame that
    // fails to capture drops its time too and never mis-labels later frames.
    let mut frames_out: Vec<(f64, tools::CapturedImage)> = Vec::new();
    for &tk in &times {
        // Honor a Stop pressed mid-clip: the tool loop doesn't otherwise drain
        // task_rx, so a Cancel would sit unprocessed until the whole clip finished.
        match task_rx.try_recv() {
            Ok(MainTask::Cancel) => {
                let _ =
                    exec_tool(op_tx, "set_edit_cursor".to_string(), restore_cursor.clone()).await;
                let _ = ui_tx.send(UiEvent::Status("Cancelled.".into()));
                return ToolOutcome::ok(
                    json!({ "cancelled": true, "captured_frames": frames_out.len() }).to_string(),
                );
            }
            Ok(MainTask::Prompt(_)) => {
                let _ = ui_tx.send(UiEvent::Status(
                    "Please wait until the current answer is finished…".into(),
                ));
            }
            _ => {}
        }
        let _ = exec_tool(
            op_tx,
            "set_edit_cursor".to_string(),
            json!({ "position": tk, "seek_playback": false, "move_view": false }),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(settle_ms)).await;
        let frame = exec_tool(op_tx, "capture_view".to_string(), json!({ "target": "video" })).await;
        if let Some(img) = frame.images.into_iter().next() {
            frames_out.push((tk, img));
        }
    }

    // The clip's audio (master mix over the span), for audio-capable models only.
    // listen_to_audio caps the render (LISTEN_MAX_SECONDS), so surface whether the
    // audio actually covers the whole clip — otherwise a >cap span invites bogus
    // A/V-sync reasoning against audio that stops early.
    let (audio, audio_seconds, audio_truncated) = if want_audio && span > 0.0 {
        let a = exec_tool(
            op_tx,
            "listen_to_audio".to_string(),
            json!({ "target": "master", "start": t0, "length": span }),
        )
        .await;
        let meta: Value = serde_json::from_str(&a.content).unwrap_or_default();
        let secs = meta.get("rendered_seconds").and_then(|v| v.as_f64());
        let trunc = meta.get("truncated").and_then(|v| v.as_bool()).unwrap_or(false);
        (a.audio, secs, trunc)
    } else {
        (None, None, false)
    };

    let _ = exec_tool(op_tx, "set_edit_cursor".to_string(), restore_cursor).await;

    if frames_out.is_empty() {
        return ToolOutcome::error(
            json!({
                "error": "captured no video frames — is the Video window open and floating (View -> Video)?"
            })
            .to_string(),
        );
    }
    let r2 = |x: f64| (x * 100.0).round() / 100.0;
    let dropped = times.len() - frames_out.len();
    // frame_times is 1:1 with the attached images (dropped frames omitted from both).
    let frame_times: Vec<f64> = frames_out.iter().map(|(t, _)| r2(*t)).collect();
    let images: Vec<tools::CapturedImage> = frames_out.into_iter().map(|(_, img)| img).collect();

    let mut note = String::from(
        "A sequence of video frames is attached as images, in time order; the i-th image \
         corresponds to frame_times[i]. ",
    );
    if dropped > 0 {
        note.push_str(
            "Some requested frames could not be captured and were omitted from BOTH the images \
             and frame_times (so they stay aligned). ",
        );
    }
    note.push_str(match (audio.is_some(), audio_truncated) {
        (true, false) => "The clip's audio is attached; judge motion, cuts, transitions and \
                          audio/video sync.",
        (true, true) => "The clip's audio is attached but covers only the first part of the range \
                         (see audio_seconds); judge motion, cuts and transitions, and limit \
                         audio/video-sync judgments to that window.",
        (false, _) => "No audio is attached; judge motion, cuts and transitions.",
    });

    let summary = json!({
        "captured_frames": images.len(),
        "requested_frames": frames,
        "dropped_frames": dropped,
        "start": r2(t0),
        "end": r2(t1),
        "span_seconds": r2(span),
        "frame_times": frame_times,
        "audio_attached": audio.is_some(),
        "audio_seconds": audio_seconds,
        "audio_truncated": audio_truncated,
        "note": note,
    })
    .to_string();
    ToolOutcome {
        content: summary,
        is_error: false,
        images,
        audio,
    }
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
    // Nearly all tools reply within a tick, and the post-FX render (the slow case)
    // is capped at 30 s, so a 90 s ceiling catches a main-thread callback that
    // never fires without hanging the agent. But some tools legitimately BLOCK on a
    // modal dialog until the user responds — add_action_shortcut opens REAPER's
    // key-assignment dialog, and run_action can invoke an action that opens a
    // Preferences/Render dialog. For those, 90 s would falsely report "timed out"
    // (dropping the real result and possibly prompting a duplicate) while the user
    // is still interacting, so give them a long ceiling that still bails on a true
    // hang.
    let timeout = if matches!(name.as_str(), "add_action_shortcut" | "run_action") {
        std::time::Duration::from_secs(3600)
    } else {
        std::time::Duration::from_secs(90)
    };
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
    match tokio::time::timeout(timeout, reply_rx).await {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(_)) => ToolOutcome::error("{\"error\":\"no reply from main thread\"}"),
        Err(_) => ToolOutcome::error("{\"error\":\"the tool timed out\"}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_result(tag: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: vec![Content::ToolResult {
                tool_use_id: tag.into(),
                content: vec![
                    ResultBlock::Text(format!("captured {tag}")),
                    ResultBlock::Image {
                        media_type: "image/png".into(),
                        data_base64: "BIGBASE64".into(),
                    },
                ],
                is_error: false,
            }],
        }
    }

    fn image_count(m: &ChatMessage) -> usize {
        m.content
            .iter()
            .flat_map(|c| match c {
                Content::ToolResult { content, .. } => content.clone(),
                _ => vec![],
            })
            .filter(|b| matches!(b, ResultBlock::Image { .. }))
            .count()
    }

    #[test]
    fn evict_keeps_recent_media_and_drops_older() {
        let mut history = vec![
            img_result("old"),                     // 0: oldest media -> evicted
            ChatMessage::user_text("some text"),   // 1: no media
            img_result("recent"),                  // 2: newest media -> kept
        ];
        evict_stale_media(&mut history, 1);
        assert_eq!(image_count(&history[0]), 0, "old media evicted");
        assert_eq!(image_count(&history[2]), 1, "recent media kept");
        // The evicted slot became a text placeholder pointing at re-capture.
        if let Content::ToolResult { content, .. } = &history[0].content[0] {
            assert!(content.iter().any(|b| matches!(b, ResultBlock::Text(t) if t.contains("capture_view"))));
        } else {
            panic!("expected a tool result");
        }
    }

    #[test]
    fn evict_is_idempotent() {
        let mut history = vec![img_result("a"), img_result("b")];
        evict_stale_media(&mut history, 1);
        let after_first = image_count(&history[0]);
        evict_stale_media(&mut history, 1);
        assert_eq!(image_count(&history[0]), after_first, "second pass changes nothing");
        assert_eq!(image_count(&history[1]), 1, "most recent still live");
    }

    #[test]
    fn evict_keep_zero_drops_all() {
        let mut history = vec![img_result("a"), img_result("b")];
        evict_stale_media(&mut history, 0);
        assert_eq!(image_count(&history[0]) + image_count(&history[1]), 0);
    }
}
