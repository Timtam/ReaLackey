//! Main-thread re-entrancy guard.
//!
//! Some operations we run on the main thread pump REAPER's message loop — an
//! offline render via `Main_OnCommand`, and native modal dialogs (confirmation
//! boxes, the API-key input). While that inner loop runs, REAPER can re-invoke
//! our OTHER main-thread callbacks: the `ControlSurface` pump (`run()`) and our
//! hook-command actions. Those must NOT call the REAPER API re-entrantly while
//! REAPER is mid-operation (especially mid-render) — doing so crashes the host
//! with a C++ fault that the Rust panic firewall cannot catch.
//!
//! So: wrap any message-pumping operation in [`enter()`], and have every
//! main-thread callback that touches the REAPER API bail when [`is_busy()`] is
//! true (or, for the pump, when `enter()` yields `None`).

use std::cell::Cell;

thread_local! {
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// True while a main-thread operation that pumps the message loop is in progress
/// (so we are nested inside REAPER's own processing and must not re-enter it).
pub fn is_busy() -> bool {
    BUSY.with(|c| c.get())
}

/// Mark the main thread busy until the returned guard drops. Yields `None` if it
/// was already busy (i.e. this call is itself a nested/re-entrant invocation),
/// so callers can skip their work.
#[must_use]
pub fn enter() -> Option<BusyGuard> {
    if BUSY.with(|c| c.replace(true)) {
        None
    } else {
        Some(BusyGuard)
    }
}

/// Clears the busy flag on drop.
pub struct BusyGuard;

impl Drop for BusyGuard {
    fn drop(&mut self) {
        BUSY.with(|c| c.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enter_is_exclusive_and_resets_on_drop() {
        assert!(!is_busy());
        {
            let g = enter();
            assert!(g.is_some());
            assert!(is_busy());
            // A nested enter() must fail while the first guard is alive.
            assert!(enter().is_none());
            assert!(is_busy());
        }
        // Guard dropped -> not busy, and enter() works again.
        assert!(!is_busy());
        assert!(enter().is_some());
    }
}
