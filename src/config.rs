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
///
/// The audio (`listen_to_audio`) and vision (`capture_view`, pixel control)
/// paragraphs are included ONLY when the active model actually has that
/// capability — mirroring the tool-list gating in [`crate::tools::definitions`].
/// Otherwise a text-only model reads that it can hear/see, offers the tool to
/// the user, and then discovers the tool was never in its toolset.
///
/// When `screen_reader` is set (OSARA detected — see [`crate::reaper::osara`]),
/// a paragraph tells the model the user is blind, so it stops giving visual
/// directions ("look for the cog icon") and instead sees GUIs itself / prefers
/// keyboard- and action-based paths.
pub fn system_prompt(supports_images: bool, supports_audio: bool, screen_reader: bool) -> String {
    let mut prompt = String::from(
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
     master FX), plus professional loudness metering — integrated LUFS, loudness \
     range (LRA), true-peak (dBTP), and momentary/short-term maxima — of the \
     processed master or a track (measure_loudness)) and make changes through \
     mutating \
     tools (add an FX, set an FX parameter, enable/bypass an FX, write MIDI \
     notes, delete MIDI notes, create a MIDI item, create tracks, \
     add/adjust/remove track sends, create automation envelopes (an FX parameter \
     via create_fx_envelope, a track volume/pan/mute envelope via \
     create_track_envelope, or a SEND/RECEIVE volume/pan/mute envelope via \
     create_send_envelope — then automate points using the envelope_track_index \
     and envelope_index it returns) and write/edit/delete their points (insert_envelope_point, \
     set_envelope_point, delete_envelope_point/delete_envelope_points, clear_envelope) \
     — create the envelope first if it does not exist yet, then add points; \
     add/delete markers and regions, edit the tempo map and \
     project tempo, add/delete take stretch markers, change render settings, \
     edit item properties (fades, length, mute, loop, snap, color), take \
     properties (start offset, rate, pitch, pan, channel mode, name), and track \
     settings (visibility, height, folder nesting, mute/solo, color, name); \
     copy/move/delete items, create empty items, copy a take, duplicate/delete \
     tracks, remove track/take FX, edit markers and regions, delete automation \
     points, and manage track groups and item groups; and control the session: \
     transport (play/stop/pause/record), move the edit cursor, change the playback \
     speed and ruler unit, and toggle global options (metronome, repeat, snapping, \
     ripple editing). Track/take volume and pan, record-arm/mute/solo, and free \
     item positioning are set via the track/item/take property tools). \
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
     per-track notes. ",
    );

    // Audio: only advertise listening when the active model can actually hear.
    if supports_audio {
        prompt.push_str(
    "You can LISTEN to a short rendered clip of the \
     processed master or a track (listen_to_audio) to judge tone, balance, noise \
     and artifacts directly — use it only when hearing genuinely helps, and (like \
     screenshots) each listen asks the user for consent. ",
        );
    }

    // Vision + pixel control: only advertise seeing/clicking GUIs when the active
    // model can actually process images.
    if supports_images {
        prompt.push_str(
    "For plugin GUIs a screen reader cannot read (custom-drawn interfaces, meters, \
     waveforms), you can SEE them with capture_view (each capture asks the user for \
     consent). To look at a SPECIFIC plugin the user hasn't focused (e.g. one you \
     just added), pass its track_index and fx_index to capture_view — it opens the \
     plugin's window for you, so you don't need the user to bring it to the front. \
     Having seen a control, PREFER to act through the parameter API — \
     set_fx_param or set_fx_param_by_steps (relative nudges like 'a bit more') — \
     because those are undoable. Only for GUI-only controls that have NO host \
     parameter (e.g. a Kontakt mode or patch switch) fall back to plugin_click / \
     plugin_drag, giving pixel coordinates read from the most recent capture_view \
     image of that plugin; the user must arm pixel control first (they are prompted \
     once). Those synthesized clicks are NOT undoable by REAPER, so after each one \
     call capture_view again to verify, and work in small steps. When the user says \
     to stop operating the GUI, or you are done, call disable_pixel_control. ",
        );
    }

    // Screen-reader user (OSARA detected): stop the model from giving sighted
    // directions and steer it toward accessible paths.
    if screen_reader {
        prompt.push_str(
    "The user is BLIND and operates REAPER with a screen reader (NVDA + OSARA). \
     Never give visual directions: do not tell them to 'look at' or 'see' anything, \
     or to find something by its icon, colour, or on-screen position (e.g. 'the cog \
     icon in the top-right'). Prefer keyboard-driven paths — REAPER actions (name the \
     action), OSARA commands, and the parameter API — over mouse or visual navigation, \
     and refer to controls by their name or label. Report results in words (values, \
     states, names), not visual layout. ",
        );
        if supports_images {
            prompt.push_str(
    "When a value or control lives on a plugin or custom GUI the screen reader cannot \
     read, do NOT ask the user to find it — YOU are their eyes: use capture_view to see \
     it yourself and read the relevant values back, then act through the parameter API \
     (preferred) or, only for a control with no host parameter, plugin_click/plugin_drag. ",
            );
        }
    }

    prompt.push_str("Answer concisely.");
    prompt
}

#[cfg(test)]
mod tests {
    use super::system_prompt;

    // The system prompt must never advertise a capability whose tool is gated
    // out of the toolset — otherwise the model offers e.g. listen_to_audio to a
    // text-only account, the user accepts, and the tool isn't there.
    #[test]
    fn prompt_hides_audio_without_support() {
        let p = system_prompt(true, false, false);
        assert!(!p.contains("listen_to_audio"), "must not mention audio tool");
        assert!(!p.contains("LISTEN"), "must not offer listening");
    }

    #[test]
    fn prompt_hides_vision_without_support() {
        let p = system_prompt(false, true, false);
        assert!(!p.contains("capture_view"), "must not mention vision tool");
        assert!(!p.contains("plugin_click"), "must not mention pixel control");
    }

    #[test]
    fn prompt_shows_capabilities_when_supported() {
        let p = system_prompt(true, true, false);
        assert!(p.contains("listen_to_audio"), "audio-capable: offer it");
        assert!(p.contains("capture_view"), "vision-capable: offer it");
    }

    #[test]
    fn prompt_text_only_offers_neither() {
        let p = system_prompt(false, false, false);
        assert!(!p.contains("listen_to_audio"));
        assert!(!p.contains("capture_view"));
        assert!(p.ends_with("Answer concisely."));
    }

    // Screen-reader (OSARA) framing is opt-in and must not leak to sighted users.
    #[test]
    fn prompt_omits_screen_reader_framing_by_default() {
        let p = system_prompt(true, true, false);
        assert!(!p.contains("BLIND"), "no blind-user framing when OSARA absent");
    }

    #[test]
    fn prompt_adds_screen_reader_framing_when_flagged() {
        let p = system_prompt(true, true, true);
        assert!(p.contains("BLIND"), "must state the user is blind");
        assert!(p.contains("cog icon"), "must forbid visual directions");
        // Vision-capable: tell it to be the user's eyes with capture_view.
        assert!(p.contains("YOU are their eyes"));
    }

    #[test]
    fn prompt_screen_reader_without_vision_omits_capture_view() {
        // No vision: keep the blind-user framing but don't offer capture_view.
        let p = system_prompt(false, false, true);
        assert!(p.contains("BLIND"));
        assert!(!p.contains("capture_view"), "no vision tool without images");
    }
}
