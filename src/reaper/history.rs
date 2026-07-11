//! Best-effort activity trail. REAPER's C API exposes only the *next* undoable
//! action label (`Undo_CanUndo2`), not the full undo stack. The main-thread pump
//! samples that label periodically; whenever it changes, we append it here,
//! building a rolling log of what the user (or the assistant) has been doing.
//! The `get_undo_history` tool reads this.

use std::collections::VecDeque;
use std::sync::Mutex;

const CAP: usize = 200;

static HISTORY: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
static LAST: Mutex<Option<String>> = Mutex::new(None);

/// Feed the current "next undo" label (main thread). Records it when it changes.
pub fn observe(current: Option<String>) {
    let mut last = LAST.lock().unwrap();
    if *last == current {
        return;
    }
    if let Some(label) = &current {
        if let Ok(mut h) = HISTORY.lock() {
            h.push_back(label.clone());
            while h.len() > CAP {
                h.pop_front();
            }
        }
    }
    *last = current;
}

/// Snapshot of the recorded labels, oldest first.
pub fn snapshot() -> Vec<String> {
    HISTORY
        .lock()
        .map(|h| h.iter().cloned().collect())
        .unwrap_or_default()
}
