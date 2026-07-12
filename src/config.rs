//! Runtime configuration. API keys, endpoints, models and token limits are owned
//! per configured provider by [`crate::providers::registry`] (design §kap-providers);
//! the functions here resolve them for the *active* (default) provider so the
//! rest of the code stays provider-agnostic.

use crate::providers::registry;

/// Load the provider registry at startup (seeds a Claude account on first run,
/// migrates the legacy single key).
pub fn init_key_cache() {
    registry::init();
}

/// The active provider's API key, if configured.
pub fn api_key() -> Option<String> {
    registry::active_key()
}

/// Whether the active provider can send (key/endpoint present).
pub fn has_api_key() -> bool {
    registry::active_can_send()
}

/// Set (or, with an empty string, clear) the active provider's key. Used by the
/// legacy "Set API key" action until the provider dialog (M4) supersedes it.
pub fn set_api_key(key: &str) -> Result<(), String> {
    registry::set_active_key(key)
}

/// The active provider's model id.
pub fn default_model() -> String {
    registry::active()
        .map(|c| c.model)
        .unwrap_or_else(|| "claude-opus-4-8".to_string())
}

/// The active provider's per-turn output-token limit.
pub fn max_output_tokens() -> u32 {
    registry::active().map(|c| c.max_tokens).unwrap_or(8192)
}

/// Whether mutating tools require user confirmation (design: configurable,
/// default on). Set `RAAI_CONFIRM=off` (or 0/false/no) to disable.
pub fn confirmation_required() -> bool {
    match std::env::var("RAAI_CONFIRM") {
        Ok(v) => !matches!(
            v.trim().to_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => true,
    }
}

/// System prompt. Establishes the role and how to use the read tools
/// (design §kap-llm). Grows as more capabilities land.
pub fn system_prompt() -> String {
    "You are an AI assistant embedded in the REAPER digital audio workstation. \
     You can inspect the project through read tools (project summary — including \
     the project name and file path, which often hint at the project's intent — \
     tracks, \
     track/take FX and their parameters, selected items, installed plugins, and \
     the focused FX window, MIDI notes of a take, a track's sends/receives, and \
     its automation envelopes and points, project and per-track notes, project \
     markers and regions, the tempo/time-signature map, a take's stretch markers, \
     the project's render settings, item/take/track properties and \
     grouping, and DSP audio analysis — loudness (LUFS), peak/RMS, clipping, \
     and a spectral profile — of a take or track's pre-FX source audio, or, via \
     a brief offline render, of PROCESSED post-FX audio (an item through its \
     take+track FX, a track's processed output, or the full master mix incl. \
     master FX)) and make changes through \
     mutating \
     tools (add an FX, set an FX parameter, enable/bypass an FX, write MIDI \
     notes, delete MIDI notes, create a MIDI item, create tracks, \
     add/adjust/remove track sends, write \
     automation points, add/delete markers and regions, edit the tempo map and \
     project tempo, add/delete take stretch markers, change render settings, \
     edit item properties (fades, length, mute, loop, snap, color), take \
     properties (start offset, rate, pitch, pan, channel mode, name), and track \
     settings (visibility, height, folder nesting, mute/solo, color, name); \
     copy/move/delete items, copy a take, duplicate/delete tracks, and manage \
     track groups and item groups). \
     When composing MIDI, read the existing take (and its \
     neighbouring items via include_neighbors) first so new material fits the \
     key, tempo, and surrounding parts. When a question depends on the \
     current project state, call a tool instead of guessing, and chain tools when \
     needed (e.g. resolve the focused FX, then read its parameters). Prefer \
     human-readable display values over raw normalized 0..1 values. Before making \
     a change, briefly explain what you intend to do; every change is shown to the \
     user for confirmation and is wrapped in a labelled undo block, so both you \
     and the user can undo it. When you plan SEVERAL independent changes (e.g. \
     configuring a plugin's parameters), make them together in ONE step (multiple \
     tool calls in the same turn) so the user can approve them all with a single \
     confirmation, instead of one at a time. You can undo/redo actions and read the recent-action \
     history (get_undo_history) to understand the user's workflow and suggest \
     improvements. You have a persistent per-project memory saved in the \
     project file: at the START of a session call get_project_memory to recall \
     context, and use set_project_memory to record decisions, TODOs, and \
     progress as you work. You can also read/append the project's Notes and \
     per-track notes. \
     For plugin GUIs a screen reader cannot read (custom-drawn interfaces, meters, \
     waveforms), you can SEE them with capture_view (each capture asks the user for \
     consent). Having seen a control, PREFER to act through the parameter API — \
     set_fx_param or set_fx_param_by_steps (relative nudges like 'a bit more') — \
     because those are undoable. Only for GUI-only controls that have NO host \
     parameter (e.g. a Kontakt mode or patch switch) fall back to plugin_click / \
     plugin_drag, giving pixel coordinates read from the most recent capture_view \
     image of that plugin; the user must arm pixel control first (they are prompted \
     once). Those synthesized clicks are NOT undoable by REAPER, so after each one \
     call capture_view again to verify, and work in small steps. When the user says \
     to stop operating the GUI, or you are done, call disable_pixel_control. \
     Answer concisely."
        .to_string()
}
