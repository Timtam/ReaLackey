//! Main-thread pump. Registered as a `ControlSurface`, whose `run()` REAPER
//! calls ~30x/second on the main thread — the one place it is safe to touch the
//! dialog and OSARA. It drains `UiEvent`s produced by the worker thread.

use crossbeam_channel::Receiver;
use reaper_medium::ControlSurface;

use crate::ai::protocol::UiEvent;
use crate::reaper::osara::Osara;
use crate::ui;

#[derive(Debug)]
pub struct PumpSurface {
    rx: Receiver<UiEvent>,
    osara: Osara,
}

impl PumpSurface {
    pub fn new(rx: Receiver<UiEvent>, osara: Osara) -> Self {
        Self { rx, osara }
    }

    fn handle(&self, ev: UiEvent) {
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
}

impl ControlSurface for PumpSurface {
    fn run(&mut self) {
        // Bounded drain so a burst never starves REAPER's main loop.
        for _ in 0..512 {
            match self.rx.try_recv() {
                Ok(ev) => self.handle(ev),
                Err(_) => break,
            }
        }
    }
}
