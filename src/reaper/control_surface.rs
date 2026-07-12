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
}

impl PumpSurface {
    pub fn new(ui_rx: Receiver<UiEvent>, op_rx: Receiver<ReaperOp>) -> Self {
        Self {
            ui_rx,
            op_rx,
            tick: 0,
        }
    }

    fn handle_ui(&self, ev: UiEvent) {
        use crate::ui::output;
        match ev {
            UiEvent::UserMessage(s) => output::user_message(&s),
            UiEvent::AssistantStart => output::assistant_start(),
            UiEvent::AssistantDelta(s) => output::assistant_delta(&s),
            UiEvent::ToolStarted { name, input } => output::tool_started(&name, &input),
            UiEvent::ToolFinished { is_error, summary } => {
                output::tool_finished(is_error, &summary)
            }
            UiEvent::Notice(s) => output::notice(&s),
            UiEvent::Status(s) => ui::ffi::set_status(&s),
            UiEvent::Announce(s) => {
                // Two accessible channels: OSARA (speaks directly, focus-
                // independent) and the webview's aria-live region (read by the
                // screen reader when the pane is observed).
                osara::announce(&s);
                output::announce(&s);
            }
            UiEvent::Done => {}
            UiEvent::Error(e) => {
                output::error(&e);
                ui::ffi::set_status("Error.");
                let msg = format!("Error: {e}");
                osara::announce(&msg);
                output::announce(&msg);
            }
        }
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
                            "REAPER AI Assistant",
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
        }
    }
}
