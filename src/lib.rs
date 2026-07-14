//! ReaLackey — a single-process native REAPER extension that
//! integrates large language models into the DAW workflow.
//!
//! Architecture (all in one address space):
//!   * `reaper`    — REAPER API + OSARA + dialog, main thread.
//!   * `ai`        — provider-agnostic agent loop, worker thread (tokio).
//!   * `providers` — LLM adapters (native Anthropic in Phase 0).
//!   * `ui`        — Rust side of the C++/SWELL dialog shim.
//!   * `tools`     — the tool/function catalog the model drives.
//!   * `dsp`       — pure-Rust audio feature extraction (loudness/spectral).
//!
//! Phase 0 scope: the extension loads, the shim shows the modeless dialog,
//! OSARA is detected, and a hello-world streaming round-trip with Claude runs
//! (no tools yet).

mod ai;
mod app;
mod config;
mod dsp;
mod prompts;
mod providers;
mod reaper;
mod text;
mod tools;
mod ui;

use std::error::Error;

use reaper_low::PluginContext;

// Windows module entry. Hand-written (NOT `reaper_low::dll_main!()`) so we can
// SKIP `execute_plugin_destroy_hooks()` at DLL_PROCESS_DETACH. That call lazily
// initializes reaper-low's `PLUGIN_DESTROY_HOOKS` (a `LazyLock<Fragile<…>>`) for
// the FIRST time during teardown, and `Fragile`'s init calls
// `std::thread::current()`, which panics once TLS is being destroyed and aborts
// REAPER on every exit (confirmed via a crash-dump backtrace: DllMain → Once::call
// → Fragile init → thread::current::init_current → panic_cannot_unwind → fastfail).
// We register no destroy hooks, so there is nothing to run anyway.
#[cfg(target_family = "windows")]
#[no_mangle]
#[allow(non_snake_case)]
extern "system" fn DllMain(
    hinstance: reaper_low::raw::HINSTANCE,
    reason: u32,
    _reserved: *const u8,
) -> u32 {
    if reason == reaper_low::raw::DLL_PROCESS_ATTACH {
        let _ = reaper_low::register_hinstance(hinstance);
    }
    1
}

// Non-Windows SWELL entry (its DLL_PROCESS_DETACH path does NOT call
// execute_plugin_destroy_hooks, so it's safe to keep as-is).
reaper_low::swell_dll_main!();

/// Extension entry point (the C `ReaperPluginEntry` symbol REAPER resolves).
///
/// REAPER calls this with a NON-null `rec` at load, and with a NULL `rec` when it
/// UNLOADS us — which happens during REAPER's shutdown, while the module is still
/// attached (message loop alive, before DLL detach). We use that null call to
/// tear the WebView2 controller down at a safe time: the window is unowned, so
/// REAPER no longer destroys it (and its webview) for us during shutdown, and if
/// the WebView2 COM state is still live at DLL detach its teardown aborts REAPER.
///
/// # Safety
/// Called by REAPER across the C ABI; both paths are panic-guarded (design N3).
#[no_mangle]
unsafe extern "C" fn ReaperPluginEntry(
    h_instance: reaper_low::raw::HINSTANCE,
    rec: *mut reaper_low::raw::reaper_plugin_info_t,
) -> std::os::raw::c_int {
    if rec.is_null() {
        // REAPER unloading us (at quit), still module-attached: drop the webview
        // now (closes the WebView2 controller) so it doesn't tear down during the
        // later CRT TLS pass. Panic-guarded across the C ABI.
        let _ = std::panic::catch_unwind(|| {
            crate::ui::output::on_destroy();
        });
        return 0;
    }
    let static_context = reaper_low::static_plugin_context();
    reaper_low::bootstrap_extension_plugin(h_instance, rec, static_context, plugin_main)
}

/// The load path: guard the FFI boundary so a panic can never unwind into REAPER.
fn plugin_main(context: PluginContext) -> Result<(), Box<dyn Error>> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || app::init(context))) {
        Ok(res) => res,
        Err(_) => Err("Panic while loading the ReaLackey".into()),
    }
}
