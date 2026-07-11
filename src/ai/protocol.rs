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
}

/// Something to reflect in the UI / screen reader. Handled ONLY on the main
/// thread (see `reaper::control_surface::PumpSurface`).
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// Append a chunk to the read-only conversation log.
    AssistantDelta(String),
    /// Replace the status line.
    Status(String),
    /// Speak a full sense-unit via OSARA (screen reader).
    Announce(String),
    /// Generation finished (success or handled error).
    Done,
    /// Surface an error in the log + status + OSARA.
    Error(String),
}
