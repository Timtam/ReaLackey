//! Wiring: build the channels, spawn the worker, register the main-thread pump
//! and the action, and keep everything alive for the plug-in's lifetime.

use std::error::Error;
use std::ffi::c_void;

use reaper_low::PluginContext;
use reaper_medium::ReaperSession;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::ai::worker;
use crate::reaper::action;
use crate::reaper::control_surface::PumpSurface;
use crate::reaper::osara::Osara;
use crate::ui;

/// Owns the session (whose Drop would unregister everything) + the task sender.
/// Deliberately leaked so registrations persist for the plug-in's lifetime.
struct AppState {
    _session: ReaperSession,
    _task_tx: tokio::sync::mpsc::UnboundedSender<MainTask>,
}

pub fn init(context: PluginContext) -> Result<(), Box<dyn Error>> {
    // Extract what we need from the low-level context BEFORE it's moved into
    // the medium session.
    let osara_ptr = unsafe { context.GetFunc(c"osara_outputMessage".as_ptr()) };
    let hinst = context.h_instance().cast::<c_void>();

    // REAPER's GetFunc, passed to SWELL on non-Windows (ignored on Windows).
    #[cfg(windows)]
    let get_func: *mut c_void = std::ptr::null_mut();
    #[cfg(not(windows))]
    let get_func: *mut c_void = context
        .to_raw()
        .GetFunc
        .map(|f| f as *mut c_void)
        .unwrap_or(std::ptr::null_mut());

    let osara = Osara::from_ptr(osara_ptr);

    // Resolve REAPER's native input box (for the "set API key" action) and load
    // any persisted/env API key into the cache — both before `context` moves.
    crate::reaper::prompt::init(&context);
    crate::config::init_key_cache();

    // Load the medium-level session (consumes the context).
    let mut session = ReaperSession::load(context);

    // Bring up the C++/SWELL UI shim and wire the dialog callbacks.
    ui::ffi::init(hinst, get_func);
    ui::ffi::install_callbacks();

    // Channels: worker -> main (crossbeam), main/UI -> worker (tokio mpsc).
    let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiEvent>();
    let (task_tx, task_rx) = tokio::sync::mpsc::unbounded_channel::<MainTask>();
    ui::bridge::set_task_sender(task_tx.clone());
    ui::bridge::set_ui_sender(ui_tx.clone());

    // Worker thread: agent loop + HTTP/SSE.
    worker::spawn(task_rx, ui_tx);

    // Main-thread pump drains UiEvents ~30x/s and updates the dialog / OSARA.
    session.plugin_register_add_csurf_inst(Box::new(PumpSurface::new(ui_rx, osara)))?;

    // "Open Assistant" action.
    action::register(&mut session)?;

    // Readiness feedback.
    session
        .reaper()
        .show_console_msg("REAPER AI Assistant loaded.\n");
    if osara.is_available() {
        osara.announce("REAPER AI Assistant loaded.");
    }

    // Keep the session (and thus all registrations) alive forever.
    std::mem::forget(AppState {
        _session: session,
        _task_tx: task_tx,
    });
    Ok(())
}
