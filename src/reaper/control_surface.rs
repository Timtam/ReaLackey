//! Main-thread pump. Registered as a `ControlSurface`, whose `run()` REAPER
//! calls ~30x/second on the main thread — the one place it is safe to touch the
//! dialog, OSARA, and the REAPER API. It drains two worker->main channels:
//! `UiEvent`s (streamed output/status) and `ReaperOp`s (tool executions +
//! confirmation prompts), and periodically samples the undo label to build the
//! activity trail.

use std::cell::Cell;

use crossbeam_channel::Receiver;
use reaper_medium::{ControlSurface, MessageBoxResult, MessageBoxType};

use crate::ai::protocol::UiEvent;
use crate::reaper::{api, history, osara};
use crate::tools::{self, ReaperOp, ToolOutcome};
use crate::ui;

/// Poll the undo label every N ticks (~30 ticks/sec, so ~2x/second).
const UNDO_POLL_TICKS: u32 = 15;

/// How often to announce the current activity while the worker is generating.
const WORKING_ANNOUNCE_SECS: u64 = 10;

/// A short, present-continuous phrase for what a tool is doing — shown in the status
/// line and spoken by the heartbeat while the tool runs. Falls back to the
/// humanised tool name so any tool still gets a sensible label.
fn tool_activity(name: &str) -> String {
    let phrase = match name {
        "transcribe_item" => "Transcribing the audio",
        "listen_to_audio" => "Listening to the audio",
        "capture_view" => "Looking at the screen",
        "capture_video_clip" => "Watching the video",
        "measure_loudness"
        | "analyze_processed_audio"
        | "analyze_track_audio"
        | "analyze_item_audio"
        | "analyze_audio_timeline" => "Analysing the audio",
        "load_tools" => "Loading more tools",
        "run_action" => "Running a REAPER action",
        n if n.starts_with("get_") || n.starts_with("search_") => "Reading the project",
        // Anything else (mostly the mutating tools): humanise the name.
        _ => return format!("Running: {}", name.replace('_', " ")),
    };
    phrase.to_string()
}

thread_local! {
    /// Set while `run()` is on the stack. A tool invoked from `run()` — the
    /// offline post-FX render, or a modal confirmation — pumps REAPER's message
    /// loop, and REAPER re-invokes `run()` from inside it. The nested call must
    /// return immediately: a second `&mut self` is unsound, and we must not touch
    /// the REAPER API while a render is mid-flight.
    static IN_RUN: Cell<bool> = const { Cell::new(false) };
}

/// Clears [`IN_RUN`] when the outer `run()` returns.
struct RunGuard;
impl Drop for RunGuard {
    fn drop(&mut self) {
        IN_RUN.with(|c| c.set(false));
    }
}

#[derive(Debug)]
pub struct PumpSurface {
    ui_rx: Receiver<UiEvent>,
    op_rx: Receiver<ReaperOp>,
    tick: u32,
    /// When the current generation started (None when idle). Wall-clock, so the
    /// "still working" cadence stays accurate even across a blocking tool.
    work_start: Option<std::time::Instant>,
    /// When we last announced "still working".
    last_working_announce: Option<std::time::Instant>,
    /// The worker generation id the timer is anchored to (distinguishes two
    /// back-to-back prompts even if we never sample the idle gap between them).
    work_gen: Option<u64>,
    /// The generating state last pushed to the webview (gates its Esc = stop).
    webview_generating: bool,
    /// What the assistant is currently doing (drives the status line + what the
    /// periodic heartbeat speaks) — derived from the event stream: a tool's activity,
    /// reasoning, writing the reply, or a worker-supplied status. Not an elapsed
    /// timer.
    activity: String,
}

impl PumpSurface {
    pub fn new(ui_rx: Receiver<UiEvent>, op_rx: Receiver<ReaperOp>) -> Self {
        Self {
            ui_rx,
            op_rx,
            tick: 0,
            work_start: None,
            last_working_announce: None,
            work_gen: None,
            webview_generating: false,
            activity: "Working\u{2026}".to_string(),
        }
    }

    /// While the worker is generating, announce "still working" every
    /// `WORKING_ANNOUNCE_SECS` so the user (perhaps focused in REAPER) hears that
    /// it's still going. Runs from the pump, so it never fires during a blocking
    /// tool and never bursts afterwards. Also mirrors the generating state into
    /// the webview so its composer knows whether Escape should stop the turn.
    fn poll_working_announcement(&mut self) {
        let generating = crate::ai::worker::is_generating();
        if generating != self.webview_generating {
            self.webview_generating = generating;
            crate::ui::output::set_generating(generating);
        }
        if !generating {
            self.work_start = None;
            self.last_working_announce = None;
            self.work_gen = None;
            return;
        }
        let now = std::time::Instant::now();
        // A new prompt (even one that started in the same frame the previous one
        // ended) bumps the generation id — re-anchor the timer to it.
        let gen = crate::ai::worker::generation();
        if self.work_gen != Some(gen) {
            self.work_gen = Some(gen);
            self.work_start = Some(now);
            self.last_working_announce = None;
        }
        let start = self.work_start.unwrap_or(now);
        let elapsed = now.duration_since(start).as_secs();
        let since_last = self
            .last_working_announce
            .map_or(elapsed, |t| now.duration_since(t).as_secs());
        if elapsed >= WORKING_ANNOUNCE_SECS && since_last >= WORKING_ANNOUNCE_SECS {
            // Speak WHAT it's doing (the current activity), not how long it's been —
            // "Transcribing the audio", "Reasoning…", "Reading the project", etc.
            let msg = if self.activity.trim().is_empty() {
                "Still working\u{2026}".to_string()
            } else {
                self.activity.clone()
            };
            crate::ui::output::speak(&msg);
            self.last_working_announce = Some(now);
        }
    }

    fn handle_ui(&mut self, ev: UiEvent) {
        use crate::ui::output;
        match ev {
            UiEvent::UserMessage(s) => {
                output::user_message(&s);
                self.set_activity("Thinking\u{2026}");
            }
            UiEvent::AssistantStart => {
                output::assistant_start();
                self.set_activity("Writing the reply\u{2026}");
            }
            UiEvent::AssistantDelta(s) => output::assistant_delta(&s),
            UiEvent::ReasoningDelta(s) => {
                output::reasoning_delta(&s);
                self.set_activity("Reasoning\u{2026}");
            }
            UiEvent::ToolStarted { name, input } => {
                output::tool_started(&name, &input);
                let a = tool_activity(&name);
                self.set_activity(&a);
            }
            UiEvent::ToolFinished { is_error, summary } => {
                output::tool_finished(is_error, &summary);
                // Result is in; the model will reason about it next.
                self.set_activity("Thinking\u{2026}");
            }
            UiEvent::Notice(s) => output::notice(&s),
            // A worker-supplied status (e.g. "Transcribing… part 2/3") IS the current
            // activity — show it and let the heartbeat speak it.
            UiEvent::Status(s) => self.set_activity(&s),
            UiEvent::Announce(s) => {
                // One spoken channel (OSARA when present, else the aria-live
                // region) so a reader observing both doesn't hear it twice.
                output::speak(&s);
            }
            UiEvent::Done => self.set_activity("Ready."),
            UiEvent::Error(e) => {
                output::error(&e);
                self.set_activity("Error.");
                output::speak(&format!("Error: {e}"));
            }
            UiEvent::ProgressOpen(msg) => ui::ffi::progress_open(&msg),
            UiEvent::ProgressUpdate { percent, message } => {
                ui::ffi::progress_update(percent, &message)
            }
            UiEvent::ProgressClose => ui::ffi::progress_close(),
        }
    }

    /// Set the current activity + push it to the status line (webview + native
    /// fallback). Only pushes on a change, so a stream of same-value updates (e.g.
    /// reasoning deltas) doesn't spam the status.
    fn set_activity(&mut self, s: &str) {
        if self.activity == s {
            return;
        }
        self.activity = s.to_string();
        ui::ffi::set_status(s);
        crate::ui::output::status(s);
    }

    fn handle_op(&self, op: ReaperOp) {
        match op {
            ReaperOp::Tool { name, input, reply } => {
                let outcome = api::with(|reaper| tools::execute(reaper, &name, &input))
                    .unwrap_or_else(|| {
                        ToolOutcome::error("{\"error\":\"REAPER API unavailable\"}")
                    });
                let _ = reply.send(outcome);
            }
            ReaperOp::Confirm { message, reply } => {
                // Native, screen-reader-accessible Yes/No confirmation.
                let yes = api::with(|reaper| {
                    matches!(
                        reaper.show_message_box(
                            message.as_str(),
                            "ReaLackey",
                            MessageBoxType::YesNo,
                        ),
                        MessageBoxResult::Yes
                    )
                })
                .unwrap_or(false);
                let _ = reply.send(yes);
            }
        }
    }

    fn poll_undo_history(&self) {
        let label = api::with(|reaper| {
            reaper.undo_can_undo_2(
                reaper_medium::ProjectContext::CurrentProject,
                |s: &reaper_medium::ReaperStr| {
                    String::from_utf8_lossy(s.as_c_str().to_bytes()).into_owned()
                },
            )
        })
        .flatten();
        history::observe(label);
    }
}

impl ControlSurface for PumpSurface {
    fn run(&mut self) {
        // Skip re-entrant calls (REAPER pumps its loop during a render/modal that
        // we started from here, and re-invokes run() nested — see IN_RUN).
        if IN_RUN.with(|c| c.replace(true)) {
            return;
        }
        let _run_guard = RunGuard;

        // Bounded drains so a burst never starves REAPER's main loop.
        for _ in 0..512 {
            match self.ui_rx.try_recv() {
                Ok(ev) => self.handle_ui(ev),
                Err(_) => break,
            }
        }
        for _ in 0..64 {
            match self.op_rx.try_recv() {
                Ok(op) => self.handle_op(op),
                Err(_) => break,
            }
        }

        self.tick = self.tick.wrapping_add(1);
        if self.tick % UNDO_POLL_TICKS == 0 {
            self.poll_undo_history();
            // Detect OSARA (it may load after us) so the worker can tailor the
            // system prompt for a screen-reader user. No-op once found.
            osara::refresh_running();
        }
        self.poll_working_announcement();
    }
}
