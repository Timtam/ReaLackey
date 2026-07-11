//! Main-thread pump. Registered as a `ControlSurface`, whose `run()` REAPER
//! calls ~30x/second on the main thread — the one place it is safe to touch the
//! dialog, OSARA, and the REAPER API. It drains two worker->main channels:
//! `UiEvent`s (streamed output/status) and `ReaperOp`s (tool executions).

use crossbeam_channel::Receiver;
use reaper_medium::ControlSurface;

use crate::ai::protocol::UiEvent;
use crate::reaper::{api, osara::Osara};
use crate::tools::{self, ReaperOp, ToolOutcome};
use crate::ui;

#[derive(Debug)]
pub struct PumpSurface {
    ui_rx: Receiver<UiEvent>,
    op_rx: Receiver<ReaperOp>,
    osara: Osara,
}

impl PumpSurface {
    pub fn new(ui_rx: Receiver<UiEvent>, op_rx: Receiver<ReaperOp>, osara: Osara) -> Self {
        Self {
            ui_rx,
            op_rx,
            osara,
        }
    }

    fn handle_ui(&self, ev: UiEvent) {
        match ev {
            UiEvent::AssistantDelta(s) => ui::ffi::append_output(&s),
            UiEvent::Status(s) => ui::ffi::set_status(&s),
            UiEvent::Announce(s) => self.osara.announce(&s),
            UiEvent::Done => {}
            UiEvent::Error(e) => {
                ui::ffi::append_output(&format!("\r\n[Error] {e}\r\n"));
                ui::ffi::set_status("Error.");
                self.osara.announce(&format!("Error: {e}"));
            }
        }
    }

    fn handle_op(&self, op: ReaperOp) {
        let outcome = api::with(|reaper| tools::execute(reaper, &op.name, &op.input))
            .unwrap_or_else(|| ToolOutcome {
                content: "{\"error\":\"REAPER API unavailable\"}".into(),
                is_error: true,
            });
        let _ = op.reply.send(outcome);
    }
}

impl ControlSurface for PumpSurface {
    fn run(&mut self) {
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
    }
}
