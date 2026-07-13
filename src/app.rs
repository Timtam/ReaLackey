//! Wiring: build the channels, spawn the worker, register the main-thread pump
//! and the action, and keep everything alive for the plug-in's lifetime.

use std::error::Error;
use std::ffi::c_void;

use reaper_low::PluginContext;
use reaper_medium::ReaperSession;

use crate::ai::protocol::{MainTask, UiEvent};
use crate::ai::worker;
use crate::reaper::action;
use crate::reaper::api;
use crate::reaper::control_surface::PumpSurface;
use crate::reaper::osara;
use crate::tools::ReaperOp;
use crate::ui;

/// Owns the session (whose Drop would unregister everything) + the task sender.
/// Deliberately leaked so registrations persist for the plug-in's lifetime and
/// the session sits at a stable address (see `api::set`).
struct AppState {
    session: ReaperSession,
    _task_tx: tokio::sync::mpsc::UnboundedSender<MainTask>,
}

pub fn init(context: PluginContext) -> Result<(), Box<dyn Error>> {
    // Extract what we need from the low-level context BEFORE it's moved into
    // the medium session. (OSARA is resolved lazily at announce time, since it
    // may load after us — see reaper::osara.)
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

    // Resolve REAPER's native input box (for the "set API key" action) and load
    // any persisted/env API key into the cache — both before `context` moves.
    crate::reaper::prompt::init(&context);
    crate::config::init_key_cache();

    // Load the medium-level session (consumes the context).
    let mut session = ReaperSession::load(context);

    // Bring up the C++/SWELL UI shim and wire the dialog callbacks.
    ui::ffi::init(hinst, get_func);
    ui::ffi::install_callbacks();
    ui::ffi::install_provider_cbs();
    ui::ffi::install_provider_edit_cbs();

    // Channels: worker -> main (crossbeam) for UI events and tool ops;
    // main/UI -> worker (tokio mpsc) for user intents.
    let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiEvent>();
    let (op_tx, op_rx) = crossbeam_channel::unbounded::<ReaperOp>();
    let (task_tx, task_rx) = tokio::sync::mpsc::unbounded_channel::<MainTask>();
    ui::bridge::set_task_sender(task_tx.clone());
    ui::bridge::set_ui_sender(ui_tx.clone());

    // Worker thread: agent loop + HTTP/SSE + tool orchestration.
    worker::spawn(task_rx, ui_tx, op_tx);

    // Main-thread pump: drains UiEvents (output/status) and ReaperOps (tool
    // executions) ~30x/s — the only place the dialog / OSARA / REAPER API run.
    session.plugin_register_add_csurf_inst(Box::new(PumpSurface::new(ui_rx, op_rx)))?;

    // "Open Assistant" action.
    action::register(&mut session)?;

    // Readiness feedback.
    session
        .reaper()
        .show_console_msg("ReaLackey loaded.\n");
    // Lazy: a no-op if OSARA hasn't loaded yet (it usually loads after us).
    osara::announce("ReaLackey loaded.");

    // Leak the app state so the session (and all registrations) live at a stable
    // address for the process lifetime, then publish the main-thread REAPER
    // handle for the pump's tool execution.
    let app: &'static AppState = Box::leak(Box::new(AppState {
        session,
        _task_tx: task_tx,
    }));
    api::set(app.session.reaper());
    Ok(())
}
