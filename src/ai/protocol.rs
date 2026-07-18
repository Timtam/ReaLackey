//! Typed messages crossing the main-thread <-> worker boundary.
//!
//! - `MainTask` travels UI/main-thread -> worker (via a tokio mpsc).
//! - `UiEvent` travels worker -> main thread, via a crossbeam channel drained
//!   from `ControlSurface::run()`, where it is safe to touch the dialog + OSARA.

/// A unit of work requested by the user.
#[derive(Debug, Clone)]
pub enum MainTask {
    /// Send a prompt to the model.
    Prompt(String),
    /// Abort the current generation.
    Cancel,
    /// Transcribe the selected item and write the result, triggered by a bindable
    /// REAPER action (no chat involved). Runs in the worker so the HTTP call is off
    /// the main thread.
    Transcribe(TranscribeOutput),
}

/// Where a transcription action writes its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscribeOutput {
    /// Into the selected item's notes (the text travels with the item).
    Notes,
    /// A plain-text `.txt` file next to the item's source media.
    Text,
    /// An SRT subtitle file (with timecodes) next to the item's source media.
    Srt,
}

/// Something to reflect in the UI / screen reader. Handled ONLY on the main
/// thread (see `reaper::control_surface::PumpSurface`), which renders it into the
/// HTML output pane (or the plain edit-control fallback).
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// The user's prompt (starts a new exchange).
    UserMessage(String),
    /// Begin a fresh assistant message block (before its first delta).
    AssistantStart,
    /// Append a streamed token to the current assistant message.
    AssistantDelta(String),
    /// Append a streamed reasoning/"thinking" token — rendered in a collapsible
    /// block, separate from the answer, and NOT spoken as the final answer.
    ReasoningDelta(String),
    /// A tool call was started (name + pretty-printed input).
    ToolStarted { name: String, input: String },
    /// A tool call finished (its result/outcome summary).
    ToolFinished { is_error: bool, summary: String },
    /// A neutral inline note (proposed change, applied, declined, …).
    Notice(String),
    /// Replace the status line.
    Status(String),
    /// Speak a full sense-unit via OSARA (screen reader).
    Announce(String),
    /// Generation finished (success or handled error).
    Done,
    /// Surface an error in the log + status + OSARA.
    Error(String),
    /// Open the modeless transcription progress dialog (a transcription action, for
    /// sighted feedback). Carries the initial status line.
    ProgressOpen(String),
    /// Update the progress dialog: bar percent (0..=100) + status line.
    ProgressUpdate { percent: u8, message: String },
    /// Close the progress dialog.
    ProgressClose,
}
