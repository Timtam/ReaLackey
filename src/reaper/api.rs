//! Main-thread access to the REAPER API for code that isn't handed a handle
//! (the `ControlSurface` pump that executes tools). The session is leaked at a
//! stable address, so a pointer to its `Reaper` stays valid for the process
//! lifetime. Stored in a thread-local: set on the main thread at load, read on
//! the main thread from `run()`, so no cross-thread sharing occurs.

use std::cell::Cell;

use reaper_medium::{MainThreadScope, Reaper};

thread_local! {
    static REAPER: Cell<*const Reaper<MainThreadScope>> = const { Cell::new(std::ptr::null()) };
}

/// Register the (leaked, `'static`) main-thread REAPER handle. Main thread only.
pub fn set(reaper: &'static Reaper<MainThreadScope>) {
    REAPER.with(|c| c.set(reaper as *const _));
}

/// Run `f` with the main-thread REAPER handle, if available. Main thread only.
pub fn with<R>(f: impl FnOnce(&Reaper<MainThreadScope>) -> R) -> Option<R> {
    REAPER.with(|c| {
        let ptr = c.get();
        if ptr.is_null() {
            None
        } else {
            // SAFETY: pointer is to the leaked, never-dropped session's Reaper,
            // and this is only ever read on the main thread.
            Some(f(unsafe { &*ptr }))
        }
    })
}
