//! The AI-core worker: a dedicated OS thread hosting a tokio runtime that runs
//! the agent loop, streams from the provider, and forwards results to the main
//! thread. Never touches the REAPER API or the dialog directly — everything goes
//! back over the `UiEvent` channel drained by `ControlSurface::run()`.

use std::sync::Arc;

use crossbeam_channel::Sender as UiSender;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::config;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::{ChatEvent, ChatMessage, ChatRequest, LlmProvider, Role};

/// Spawn the worker on its own thread. Returns immediately.
pub fn spawn(task_rx: UnboundedReceiver<MainTask>, ui_tx: UiSender<UiEvent>) {
    let ui_tx_err = ui_tx.clone();
    let build = std::thread::Builder::new().name("raai-worker".into());
    let spawned = build.spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = ui_tx_err.send(UiEvent::Error(format!("Tokio-Runtime: {e}")));
                return;
            }
        };
        rt.block_on(run(task_rx, ui_tx));
    });
    if let Err(e) = spawned {
        // Extremely unlikely; surface it rather than silently losing the worker.
        eprintln!("raai: failed to spawn worker thread: {e}");
    }
}

async fn run(mut task_rx: UnboundedReceiver<MainTask>, ui_tx: UiSender<UiEvent>) {
    let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new());
    let mut history: Vec<ChatMessage> = Vec::new();

    while let Some(task) = task_rx.recv().await {
        match task {
            MainTask::Cancel => { /* nothing in flight */ }
            MainTask::Prompt(prompt) => {
                handle_prompt(&provider, &mut history, &ui_tx, &mut task_rx, prompt).await;
            }
        }
    }
}

async fn handle_prompt(
    provider: &Arc<dyn LlmProvider>,
    history: &mut Vec<ChatMessage>,
    ui_tx: &UiSender<UiEvent>,
    task_rx: &mut UnboundedReceiver<MainTask>,
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

    history.push(ChatMessage {
        role: Role::User,
        content: prompt,
    });

    let req = ChatRequest {
        model: config::default_model(),
        system: Some(config::system_prompt()),
        max_tokens: 1024,
        messages: history.clone(),
    };

    let cancel = CancellationToken::new();
    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel::<ChatEvent>(64);
    let provider2 = provider.clone();
    let cancel2 = cancel.clone();
    let handle = tokio::spawn(async move { provider2.chat(req, ev_tx, cancel2).await });

    let mut full = String::new();
    let mut errored = false;

    loop {
        tokio::select! {
            maybe_ev = ev_rx.recv() => match maybe_ev {
                Some(ChatEvent::TextDelta(d)) => {
                    full.push_str(&d);
                    let _ = ui_tx.send(UiEvent::AssistantDelta(d));
                }
                Some(ChatEvent::Error(e)) => {
                    errored = true;
                    let _ = ui_tx.send(UiEvent::Error(e));
                }
                Some(ChatEvent::Done { .. }) => {}
                None => break, // provider finished and dropped the sender
            },
            maybe_task = task_rx.recv() => match maybe_task {
                Some(MainTask::Cancel) | None => {
                    cancel.cancel();
                }
                Some(MainTask::Prompt(_)) => {
                    // Phase 0: one generation at a time. Ignore concurrent
                    // prompts while streaming (rare; documented limitation).
                    let _ = ui_tx.send(UiEvent::Status(
                        "Please wait until the current answer is finished…".into(),
                    ));
                }
            },
        }
    }

    let _ = handle.await;

    if !full.is_empty() {
        history.push(ChatMessage {
            role: Role::Assistant,
            content: full.clone(),
        });
        let _ = ui_tx.send(UiEvent::AssistantDelta("\r\n".into()));
        // Announce the full answer as one sense-unit (design §kap-a11y).
        let _ = ui_tx.send(UiEvent::Announce(full));
    }

    let _ = ui_tx.send(UiEvent::Status(if errored { "Error." } else { "Ready." }.into()));
    let _ = ui_tx.send(UiEvent::Done);
}
