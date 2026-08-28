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
    /// Drop the conversation history. Makes switching provider cheap: the next turn
    /// starts fresh instead of re-sending an entire chat to a different model (which
    /// costs tokens, and can fail outright where the two providers disagree about
    /// message shape — a thinking block or a tool-call turn the new one won't accept).
    ClearHistory,
    /// Transcribe the selected item and write the result, triggered by a bindable
    /// REAPER action (no chat involved). Runs in the worker so the HTTP call is off
    /// the main thread.
    Transcribe(TranscribeOutput),
    /// Open the cut-by-text editor on the selected item, triggered by a bindable
    /// REAPER action: transcribe it, show the editor, then cut what the user
    /// removed. Runs in the worker (transcription is async HTTP).
    OpenCutEditor,
    /// Download the local forced-alignment files (model + ONNX Runtime) into
    /// the resource-path models dir. Triggered from the provider settings
    /// dialog when "Refine word timings locally" is enabled and the files are
    /// missing. Runs in the worker: it is a large streamed HTTP download.
    DownloadAlignModel,
}

/// The outcome of a cut-by-text editor session, sent from the webview (main
/// thread) back to the waiting worker.
#[derive(Debug, Clone)]
pub enum EditorResult {
    /// The user confirmed: `keep[i]` is true for each word to keep (false = cut).
    Save { keep: Vec<bool> },
    /// The user cancelled (or the editor was dismissed) — change nothing.
    Cancel,
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
    /// Open the cut-by-text editor modal in the webview. Carries the JSON payload
    /// (words with times + sentence ids, and the item's audio as base64 WAV).
    OpenCutEditor(String),
    /// Close the cut-by-text editor modal (e.g. the turn was cancelled).
    CloseCutEditor,
}
