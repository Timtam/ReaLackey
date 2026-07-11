//! REAPER AI Assistant — a single-process native REAPER extension that
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
mod providers;
mod reaper;
mod text;
mod tools;
mod ui;

use std::error::Error;

use reaper_low::PluginContext;
use reaper_macros::reaper_extension_plugin;

/// Extension entry point. The macro generates the C `ReaperPluginEntry` symbol
/// and hands us the low-level `PluginContext`. We guard the FFI boundary so a
/// panic can never unwind into REAPER (design N3).
#[reaper_extension_plugin]
fn plugin_main(context: PluginContext) -> Result<(), Box<dyn Error>> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || app::init(context))) {
        Ok(res) => res,
        Err(_) => Err("Panic while loading the REAPER AI Assistant".into()),
    }
}
