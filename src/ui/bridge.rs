//! Routes dialog callbacks (which fire on the main thread) to the worker.
//!
//! The design's C-ABI callbacks carry no user pointer, so the target sender
//! lives in a process-global `OnceLock`. `send` on a tokio unbounded channel is
//! sync and thread-safe, so calling it from the main thread is fine.

use std::sync::OnceLock;

use crossbeam_channel::Sender;
use tokio::sync::mpsc::UnboundedSender;

use crate::ai::protocol::{MainTask, UiEvent};

static TASK_TX: OnceLock<UnboundedSender<MainTask>> = OnceLock::new();
static UI_TX: OnceLock<Sender<UiEvent>> = OnceLock::new();

pub fn set_task_sender(tx: UnboundedSender<MainTask>) {
    let _ = TASK_TX.set(tx);
}

pub fn set_ui_sender(tx: Sender<UiEvent>) {
    let _ = UI_TX.set(tx);
}

/// Emit a `UiEvent` to the main-thread pump (for feedback from main-thread code
/// such as the "set API key" action).
pub fn emit(ev: UiEvent) {
    if let Some(tx) = UI_TX.get() {
        let _ = tx.send(ev);
    }
}

/// "Send" pressed.
pub fn submit(text: String) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return;
    }
    if let Some(tx) = TASK_TX.get() {
        let _ = tx.send(MainTask::Prompt(text));
    }
}

/// "Confirm" pressed (Phase 3 mutations). No-op in Phase 0.
pub fn confirm(_confirm_id: i32) {}

/// Dialog closed / "Stopp": abort any in-flight generation and disarm pixel
/// control (a physical kill switch for the Tier-B click/drag capability).
pub fn cancel() {
    crate::tools::disarm_pixel_control();
    if let Some(tx) = TASK_TX.get() {
        let _ = tx.send(MainTask::Cancel);
    }
}
