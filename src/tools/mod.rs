//! Tool / function catalog (design §kap-tools).
//!
//! Phases 1–2: read-only context tools. Each tool executes on the REAPER main
//! thread (via [`crate::reaper::api`]) and returns JSON that is fed back to the
//! model as a `tool_result`. Track FX is available through reaper-medium; take
//! FX and installed-FX enumeration drop to the low-level API. Mutating tools
//! (Undo-wrapped, confirmation-gated) arrive in Phase 3.

use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, Ordering};

use reaper_medium::{
    AddFxBehavior, FxLocation, FxShowInstruction, ItemAttributeKey, MainThreadScope,
    MasterTrackBehavior, MediaItem, MediaItemTake, MediaTrack, PositionInSeconds, ProjectContext,
    Reaper, ReaperNormalizedFxParamValue, ReaperStr, SendTarget, TrackDefaultsBehavior,
    TrackFxChainType, TrackFxLocation, TrackLocation, TrackSendAttributeKey, TrackSendCategory,
    TrackSendDirection, UndoScope,
};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::providers::ToolDef;

const NAME_BUF: u32 = 256;
const DEFAULT_LIMIT: usize = 200;
/// Shared schema description for the `chain` selector on the track-FX tools.
const CHAIN_DESC: &str = "which FX chain: 'normal' (default; the track's output FX), \
    'input' (the track's record/input FX, applied before recording), or 'monitor' (REAPER's \
    global Monitoring FX, which live on the master track — track_index is ignored)";

/// A request from the worker for the main thread to run.
pub enum ReaperOp {
    /// Execute a tool and return its outcome.
    Tool {
        name: String,
        input: Value,
        reply: oneshot::Sender<ToolOutcome>,
    },
    /// Ask the user to confirm a proposed change (native Yes/No message box).
    Confirm {
        message: String,
        reply: oneshot::Sender<bool>,
    },
}

/// An image produced by a tool (a screenshot), returned to the model as an
/// Anthropic image content block so it can see visual-only UI.
pub struct CapturedImage {
    pub media_type: String,
    pub data_base64: String,
}

/// An audio clip produced by a tool, returned to an audio-capable model so it can
/// hear the actual sound. `format` is the bare container name (e.g. "wav").
pub struct CapturedAudio {
    pub format: String,
    pub data_base64: String,
}

/// The result of running a tool.
pub struct ToolOutcome {
    /// JSON (or plain text) summary fed back to the model as the tool result.
    pub content: String,
    pub is_error: bool,
    /// Images attached to the tool result (vision tools). Usually 0 or 1; a video
    /// clip attaches several frames, in time order.
    pub images: Vec<CapturedImage>,
    /// An optional audio clip attached to the tool result (audio tools only).
    pub audio: Option<CapturedAudio>,
}

impl ToolOutcome {
    /// A text result (no media).
    pub fn text(content: impl Into<String>, is_error: bool) -> Self {
        Self {
            content: content.into(),
            is_error,
            images: Vec::new(),
            audio: None,
        }
    }

    /// A successful text result.
    pub fn ok(content: impl Into<String>) -> Self {
        Self::text(content, false)
    }

    /// An error result.
    pub fn error(content: impl Into<String>) -> Self {
        Self::text(content, true)
    }

    /// A successful result carrying an image for the model to see.
    pub fn with_image(content: impl Into<String>, image: CapturedImage) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            images: vec![image],
            audio: None,
        }
    }

    /// A successful result carrying an audio clip for the model to hear.
    pub fn with_audio(content: impl Into<String>, audio: CapturedAudio) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            images: Vec::new(),
            audio: Some(audio),
        }
    }
}

/// A tool that only makes sense when the model can see images (Phase 7). These
/// are withheld from models without vision (`supports_images == false`).
fn is_vision_tool(name: &str) -> bool {
    name == "capture_view"
        || name == "capture_video_clip"
        || is_pixel_tool(name)
        || name == "disable_pixel_control"
}

/// A tool that only makes sense when the model can hear audio. Withheld from
/// models without audio input (`supports_audio == false`).
fn is_audio_tool(name: &str) -> bool {
    name == "listen_to_audio"
}

/// Tool definitions advertised to the model. Vision/pixel tools are only included
/// when the model can see images; the audio tool only when it can hear.
pub fn definitions(supports_images: bool, supports_audio: bool) -> Vec<ToolDef> {
    let obj = |props: Value, required: Value| json!({ "type": "object", "properties": props, "required": required });
    let empty = || json!({ "type": "object", "properties": {} });
    let mut defs = vec![
        ToolDef {
            name: "get_project_summary".into(),
            description: "Lightweight snapshot of the current REAPER project: project name and \
                          file path (the name often hints at the project's intent), tempo (BPM), \
                          total track count, number of selected tracks and items, and the \
                          edit cursor position in seconds."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "get_tracks".into(),
            description: "List all tracks with their 0-based index, name, and whether selected."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "get_track_fx".into(),
            description: "List the FX chain of a track: for each FX its 0-based index, name, \
                          and enabled/offline state. Use 'chain' for input or monitoring FX."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer", "description": "0-based track index" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "get_fx_params".into(),
            description: "List the parameters of a track FX: index, name, formatted display \
                          value, and normalized value (0..1)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer", "description": "0-based FX index in the chain" },
                    "limit": { "type": "integer", "description": "max parameters to return (default 200)" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index"]),
            ),
        },
        ToolDef {
            name: "get_selected_items".into(),
            description: "List selected media items: project item index, track index, position \
                          and length (seconds), active take name, and take FX count."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "get_take_fx".into(),
            description: "List the FX chain of an item's active take (index, name, enabled/offline)."
                .into(),
            input_schema: obj(
                json!({ "item_index": { "type": "integer", "description": "0-based project media item index" } }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "get_take_fx_params".into(),
            description: "List the parameters of a take FX: index, name, formatted value, \
                          normalized value (0..1)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "limit": { "type": "integer", "description": "max parameters (default 200)" }
                }),
                json!(["item_index", "fx_index"]),
            ),
        },
        ToolDef {
            name: "list_installed_fx".into(),
            description: "List installed plugins (name + identifier, which encodes the type: \
                          VST/VST3/AU/CLAP/JS). Use 'filter' to narrow by substring."
                .into(),
            input_schema: obj(
                json!({
                    "filter": { "type": "string", "description": "case-insensitive substring to match against the name" },
                    "limit": { "type": "integer", "description": "max results (default 200)" }
                }),
                json!([]),
            ),
        },
        ToolDef {
            name: "get_focused_fx".into(),
            description: "Identify the currently or last focused FX window (the plugin the user \
                          is looking at): whether it is a track or take FX and its location."
                .into(),
            input_schema: empty(),
        },
        // --- mutating tools (confirmation-gated, Undo-wrapped) ---
        ToolDef {
            name: "add_fx".into(),
            description: "Add an FX/plugin by name to a track's FX chain (or its input/monitoring \
                          chain via 'chain'). This CHANGES the project: it is confirmed by the user \
                          and wrapped in a labelled undo block."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_name": { "type": "string", "description": "plugin name, e.g. \"ReaEQ\" or \"VST3: Serum\"" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_name"]),
            ),
        },
        ToolDef {
            name: "set_fx_param".into(),
            description: "Set a track-FX parameter to a normalized value in 0..1. CHANGES the \
                          project (confirmed + undo-wrapped). Call get_fx_params first to choose \
                          the parameter index and understand its current value."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "param_index": { "type": "integer" },
                    "value": { "type": "number", "description": "normalized value 0..1" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index", "param_index", "value"]),
            ),
        },
        ToolDef {
            name: "set_fx_enabled".into(),
            description: "Enable or bypass a track FX. CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "enabled": { "type": "boolean" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index", "enabled"]),
            ),
        },
        ToolDef {
            name: "get_fx_preset".into(),
            description: "Read the current preset of a track FX: its name, 0-based preset_index \
                          (-1 when no preset is active), and the total preset_count. \
                          matches_preset is false when the FX state has been edited away from the \
                          stored preset. FX index from get_track_fx."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index"]),
            ),
        },
        ToolDef {
            name: "set_fx_preset".into(),
            description: "Load (select) a preset for a track FX. Provide EXACTLY ONE of: name \
                          (exact preset name), index (0-based preset index), or navigate \
                          ('next'/'previous'). NOTE: REAPER's API can only LOAD existing presets — \
                          it cannot SAVE a new preset (that must be done from the FX window's \
                          preset menu). CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "name": { "type": "string", "description": "exact preset name to load" },
                    "index": { "type": "integer", "description": "0-based preset index to load" },
                    "navigate": { "type": "string", "enum": ["next", "previous"], "description": "step to the next/previous preset" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index"]),
            ),
        },
        ToolDef {
            name: "get_take_fx_preset".into(),
            description: "Read the current preset of a take (item) FX — name, 0-based preset_index \
                          (-1 when none), total preset_count, and matches_preset. take_index \
                          defaults to the active take. FX index from get_take_fx."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "take_index": { "type": "integer" }
                }),
                json!(["item_index", "fx_index"]),
            ),
        },
        ToolDef {
            name: "set_take_fx_preset".into(),
            description: "Load (select) a preset for a take (item) FX. Provide EXACTLY ONE of name, \
                          index, or navigate ('next'/'previous'); take_index defaults to the active \
                          take. Like set_fx_preset, the API can only LOAD presets, not save new \
                          ones. CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "take_index": { "type": "integer" },
                    "name": { "type": "string", "description": "exact preset name to load" },
                    "index": { "type": "integer", "description": "0-based preset index to load" },
                    "navigate": { "type": "string", "enum": ["next", "previous"], "description": "step to the next/previous preset" }
                }),
                json!(["item_index", "fx_index"]),
            ),
        },
        // --- undo / history (reversible; not confirmation-gated) ---
        ToolDef {
            name: "undo".into(),
            description: "Undo the most recent action (made by either the user or the assistant). \
                          Reversible with redo."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "redo".into(),
            description: "Redo the most recently undone action.".into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "get_undo_history".into(),
            description: "Inspect undo state: the next undo/redo action labels plus a best-effort \
                          log of recent undo-point labels (a trail of what the user has been doing) \
                          for workflow suggestions."
                .into(),
            input_schema: empty(),
        },
        // --- MIDI (Phase 4) ---
        ToolDef {
            name: "get_take_midi".into(),
            description: "Read the MIDI notes of a media item's active take (pitch, note name, \
                          velocity, channel, timing in seconds and PPQ). Set include_neighbors to \
                          also read overlapping items on the adjacent tracks for harmonic context."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer", "description": "0-based project media item index" },
                    "include_neighbors": { "type": "boolean", "description": "also include overlapping items on adjacent tracks" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "insert_midi_notes".into(),
            description: "Insert MIDI notes into an item's active take. CHANGES the project \
                          (confirmed + undo-wrapped). Times are in quarter notes relative to the \
                          item start."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "notes": {
                        "type": "array",
                        "description": "notes to insert",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pitch": { "type": "integer", "description": "MIDI pitch 0-127 (60 = middle C / C4)" },
                                "start_qn": { "type": "number", "description": "start in quarter notes from the item start" },
                                "length_qn": { "type": "number", "description": "length in quarter notes" },
                                "velocity": { "type": "integer", "description": "1-127, default 96" },
                                "channel": { "type": "integer", "description": "0-15, default 0" }
                            },
                            "required": ["pitch", "start_qn", "length_qn"]
                        }
                    }
                }),
                json!(["item_index", "notes"]),
            ),
        },
        ToolDef {
            name: "create_midi_item".into(),
            description: "Create an empty MIDI item on a track. CHANGES the project (confirmed + \
                          undo-wrapped). Position and length are in quarter notes. Returns the new \
                          item_index for use with insert_midi_notes."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "start_qn": { "type": "number", "description": "start position in quarter notes" },
                    "length_qn": { "type": "number", "description": "length in quarter notes" }
                }),
                json!(["track_index", "start_qn", "length_qn"]),
            ),
        },
        // --- track routing / sends (Phase 4) ---
        ToolDef {
            name: "get_track_sends".into(),
            description: "List a track's sends (destination track, volume, pan, mute) and receives."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "add_send".into(),
            description: "Create a send from one track to another. CHANGES the project (confirmed + \
                          undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "src_track_index": { "type": "integer" },
                    "dest_track_index": { "type": "integer" }
                }),
                json!(["src_track_index", "dest_track_index"]),
            ),
        },
        ToolDef {
            name: "set_send_param".into(),
            description: "Set a send's volume, pan, or mute. CHANGES the project (confirmed + \
                          undo-wrapped). volume is a linear amplitude (1.0 = 0 dB), pan is -1..1, \
                          mute is 0 or 1."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "send_index": { "type": "integer" },
                    "param": { "type": "string", "enum": ["volume", "pan", "mute"] },
                    "value": { "type": "number" }
                }),
                json!(["track_index", "send_index", "param", "value"]),
            ),
        },
        ToolDef {
            name: "remove_send".into(),
            description: "Remove a send from a track. CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "send_index": { "type": "integer" }
                }),
                json!(["track_index", "send_index"]),
            ),
        },
        // --- automation / envelopes (Phase 5) ---
        ToolDef {
            name: "get_track_envelopes".into(),
            description: "List a track's automation envelopes (index, name, point count, \
                          automation-item count)."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "get_envelope_points".into(),
            description: "Read the points of a track envelope: time (seconds), value (in the \
                          envelope's native units), shape, tension, selected."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "envelope_index": { "type": "integer" },
                    "limit": { "type": "integer", "description": "max points (default 200)" }
                }),
                json!(["track_index", "envelope_index"]),
            ),
        },
        ToolDef {
            name: "get_automation_items".into(),
            description: "List the automation items on a track envelope (position, length, \
                          play rate, pool id)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "envelope_index": { "type": "integer" }
                }),
                json!(["track_index", "envelope_index"]),
            ),
        },
        ToolDef {
            name: "insert_envelope_point".into(),
            description: "Insert (or replace) an automation point on a track envelope. CHANGES the \
                          project (confirmed + undo-wrapped). Read get_envelope_points first to \
                          understand the value scale. shape: 0=linear, 1=square, 2=slow, 3=fast \
                          start, 4=fast end, 5=bezier."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "envelope_index": { "type": "integer" },
                    "time": { "type": "number", "description": "position in seconds" },
                    "value": { "type": "number", "description": "value in the envelope's native units" },
                    "shape": { "type": "integer", "description": "0=linear (default) .. 5=bezier" }
                }),
                json!(["track_index", "envelope_index", "time", "value"]),
            ),
        },
        // --- notes & per-project memory ---
        ToolDef {
            name: "get_project_notes".into(),
            description: "Read the project's Notes (the user-visible Project Notes field)."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "set_project_notes".into(),
            description: "Write the project's Notes field. Appends by default (set append=false to \
                          replace). Undo-wrapped so the user can revert."
                .into(),
            input_schema: obj(
                json!({
                    "text": { "type": "string" },
                    "append": { "type": "boolean", "description": "append (default true) vs replace" }
                }),
                json!(["text"]),
            ),
        },
        ToolDef {
            name: "get_track_notes".into(),
            description: "Read the per-track notes the assistant stores on a track (persists in the project)."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "set_track_notes".into(),
            description: "Set the per-track notes for a track (replaces; persists in the project). \
                          Undo-wrapped."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "text": { "type": "string" }
                }),
                json!(["track_index", "text"]),
            ),
        },
        ToolDef {
            name: "get_project_memory".into(),
            description: "Read the assistant's persistent per-project memory. With a key, returns \
                          that entry's value; without a key, lists all key/value entries. Use this \
                          at the start of a session to recall context and progress."
                .into(),
            input_schema: obj(
                json!({ "key": { "type": "string", "description": "omit to list all entries" } }),
                json!([]),
            ),
        },
        ToolDef {
            name: "set_project_memory".into(),
            description: "Store a key/value entry in the assistant's persistent per-project memory \
                          (saved in the project file). Use this to remember decisions, TODOs, and \
                          progress across sessions. Not shown to the user and not confirmation-gated."
                .into(),
            input_schema: obj(
                json!({
                    "key": { "type": "string" },
                    "value": { "type": "string" }
                }),
                json!(["key", "value"]),
            ),
        },
        ToolDef {
            name: "delete_project_memory".into(),
            description: "Delete an entry from the assistant's per-project memory.".into(),
            input_schema: obj(
                json!({ "key": { "type": "string" } }),
                json!(["key"]),
            ),
        },
        // --- markers & regions ---
        ToolDef {
            name: "get_markers".into(),
            description: "List all project markers and regions: kind (marker/region), the \
                          user-facing index number, position (seconds), region end (for regions), \
                          name, and color (native integer, 0 = default)."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "add_marker".into(),
            description: "Add a project marker at a position (seconds). CHANGES the project \
                          (confirmed + undo-wrapped). Returns the new marker's index number."
                .into(),
            input_schema: obj(
                json!({
                    "position": { "type": "number", "description": "position in seconds" },
                    "name": { "type": "string", "description": "marker name (optional)" },
                    "index_number": { "type": "integer", "description": "desired display number, or -1 to auto-assign (default)" }
                }),
                json!(["position"]),
            ),
        },
        ToolDef {
            name: "add_region".into(),
            description: "Add a region spanning start..end (seconds). CHANGES the project \
                          (confirmed + undo-wrapped). Returns the new region's index number."
                .into(),
            input_schema: obj(
                json!({
                    "start": { "type": "number", "description": "region start in seconds" },
                    "end": { "type": "number", "description": "region end in seconds" },
                    "name": { "type": "string", "description": "region name (optional)" },
                    "index_number": { "type": "integer", "description": "desired display number, or -1 to auto-assign (default)" }
                }),
                json!(["start", "end"]),
            ),
        },
        ToolDef {
            name: "delete_marker".into(),
            description: "Delete a marker or region by its display index number (as reported by \
                          get_markers). CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "index_number": { "type": "integer", "description": "the marker/region display index number" },
                    "is_region": { "type": "boolean", "description": "true to delete a region, false (default) for a marker" }
                }),
                json!(["index_number"]),
            ),
        },
        // --- tempo / time-signature map ---
        ToolDef {
            name: "get_tempo_markers".into(),
            description: "List the tempo/time-signature markers: index, time (seconds), measure, \
                          beat, BPM, time signature (numerator/denominator), and whether the tempo \
                          transition is linear."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "add_tempo_marker".into(),
            description: "Add a tempo and/or time-signature marker at a time (seconds). CHANGES the \
                          project (confirmed + undo-wrapped). Pass timesig 0/0 (default) for a \
                          tempo-only marker."
                .into(),
            input_schema: obj(
                json!({
                    "time": { "type": "number", "description": "position in seconds" },
                    "bpm": { "type": "number", "description": "tempo in BPM" },
                    "timesig_num": { "type": "integer", "description": "time-signature numerator, 0 = no change (default)" },
                    "timesig_denom": { "type": "integer", "description": "time-signature denominator, 0 = no change (default)" },
                    "linear": { "type": "boolean", "description": "linear tempo transition to the next marker (default false)" }
                }),
                json!(["time", "bpm"]),
            ),
        },
        ToolDef {
            name: "delete_tempo_marker".into(),
            description: "Delete a tempo/time-signature marker by index (as reported by \
                          get_tempo_markers). CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({ "index": { "type": "integer", "description": "0-based tempo marker index" } }),
                json!(["index"]),
            ),
        },
        ToolDef {
            name: "set_project_tempo".into(),
            description: "Set the project's (master) tempo in BPM. CHANGES the project (confirmed + \
                          undo-wrapped). For tempo changes at a specific time, use add_tempo_marker."
                .into(),
            input_schema: obj(
                json!({ "bpm": { "type": "number", "description": "tempo in BPM" } }),
                json!(["bpm"]),
            ),
        },
        // --- stretch markers (per take) ---
        ToolDef {
            name: "get_stretch_markers".into(),
            description: "List the stretch markers of an item's active take: index, position in the \
                          take (seconds), source-media position (seconds), and slope."
                .into(),
            input_schema: obj(
                json!({ "item_index": { "type": "integer", "description": "0-based project media item index" } }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "add_stretch_marker".into(),
            description: "Add a stretch marker to an item's active take at a position (seconds \
                          within the take). CHANGES the project (confirmed + undo-wrapped). \
                          Optionally pin it to a source-media position."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "position": { "type": "number", "description": "position within the take, in seconds" },
                    "src_position": { "type": "number", "description": "source-media position in seconds (optional)" }
                }),
                json!(["item_index", "position"]),
            ),
        },
        ToolDef {
            name: "delete_stretch_marker".into(),
            description: "Delete stretch marker(s) from an item's active take, starting at \
                          marker_index. CHANGES the project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "marker_index": { "type": "integer", "description": "0-based stretch marker index" },
                    "count": { "type": "integer", "description": "how many to remove (default 1)" }
                }),
                json!(["item_index", "marker_index"]),
            ),
        },
        // --- render settings ---
        ToolDef {
            name: "get_render_settings".into(),
            description: "Read the project's render settings: mode bitmask, bounds flag, channel \
                          count, sample rate, custom start/end, tail, add-to-project, dither, and \
                          the output directory and file-name pattern."
                .into(),
            input_schema: empty(),
        },
        ToolDef {
            name: "set_render_setting".into(),
            description: "Set one render setting by key. CHANGES the project (confirmed + \
                          undo-wrapped). Provide 'value' for numeric keys (e.g. RENDER_SRATE, \
                          RENDER_CHANNELS, RENDER_BOUNDSFLAG, RENDER_SETTINGS, RENDER_STARTPOS, \
                          RENDER_ENDPOS) or 'text' for string keys (RENDER_FILE, RENDER_PATTERN)."
                .into(),
            input_schema: obj(
                json!({
                    "key": { "type": "string", "description": "GetSetProjectInfo key, e.g. \"RENDER_SRATE\" or \"RENDER_FILE\"" },
                    "value": { "type": "number", "description": "numeric value (for numeric keys)" },
                    "text": { "type": "string", "description": "string value (for string keys)" }
                }),
                json!(["key"]),
            ),
        },
        // --- item properties ---
        ToolDef {
            name: "get_item_properties".into(),
            description: "Read a media item's properties: position, length, volume, mute, \
                          loop_source, lock, snap_offset, fade in/out lengths, shapes and \
                          directions, auto-fade lengths, group_id, color, all_takes_play, plus \
                          the take count."
                .into(),
            input_schema: obj(
                json!({ "item_index": { "type": "integer", "description": "0-based project media item index" } }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "set_item_property".into(),
            description: "Set one media item property to a numeric value. CHANGES the project \
                          (confirmed + undo-wrapped). property is one of: position, length, volume, \
                          mute (0/1), loop_source (0/1), lock (0/1), snap_offset, fade_in_len, \
                          fade_out_len, fade_in_len_auto (-1=off), fade_out_len_auto, fade_in_shape \
                          (0..6), fade_out_shape, fade_in_dir (-1..1), fade_out_dir, group_id, \
                          color (native color|0x1000000; 0 clears), all_takes_play (0/1). Times are \
                          in seconds."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "property": { "type": "string" },
                    "value": { "type": "number" }
                }),
                json!(["item_index", "property", "value"]),
            ),
        },
        ToolDef {
            name: "set_item_edge".into(),
            description: "Move a media item's left or right EDGE to an absolute timeline time \
                          (seconds) — trimming or extending it. CHANGES the project (confirmed + \
                          undo-wrapped, ONE undo point). The RIGHT edge just changes the item's \
                          length. The LEFT edge moves the item start AND shifts EVERY take's \
                          start-in-source offset (each by the move scaled by its own playrate) so \
                          the audio content stays put across all takes — the correct 'trim left \
                          edge' behaviour; extending the left edge before the source start yields \
                          leading silence (or loops if loop_source is on). Use this instead of \
                          composing set_item_property/set_take_property by hand."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "edge": { "type": "string", "enum": ["left", "right"], "description": "which edge to move" },
                    "time": { "type": "number", "description": "absolute timeline position (seconds) for the edge" }
                }),
                json!(["item_index", "edge", "time"]),
            ),
        },
        // --- take properties ---
        ToolDef {
            name: "get_take_properties".into(),
            description: "Read a take's properties (defaults to the active take): name, \
                          start_offset (seconds into the source), volume, pan, playrate, pitch \
                          (semitones), preserve_pitch, channel_mode, color, and the source file \
                          and length."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "take_index": { "type": "integer", "description": "0-based take index; omit for the active take" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "set_take_property".into(),
            description: "Set one take property. CHANGES the project (confirmed + undo-wrapped). \
                          property is one of: start_offset (seconds), volume (linear), pan (-1..1), \
                          playrate, pitch (semitones), preserve_pitch (0/1), channel_mode (int), \
                          color, or name (pass 'text' instead of 'value'). Defaults to the active \
                          take unless take_index is given."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "property": { "type": "string" },
                    "value": { "type": "number", "description": "numeric value (for non-name properties)" },
                    "text": { "type": "string", "description": "string value (for property=name)" },
                    "take_index": { "type": "integer" }
                }),
                json!(["item_index", "property"]),
            ),
        },
        ToolDef {
            name: "set_active_take".into(),
            description: "Choose which take of an item is the active (playing) take. CHANGES the \
                          project (confirmed + undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "take_index": { "type": "integer" }
                }),
                json!(["item_index", "take_index"]),
            ),
        },
        // --- track settings ---
        ToolDef {
            name: "get_track_properties".into(),
            description: "Read a track's settings: name, mute, solo, volume, pan, visible_tcp, \
                          visible_mixer, height, height_lock, folder_depth, folder_compact, \
                          free_mode, color, rec_arm, rec_monitor."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "set_track_property".into(),
            description: "Set one track setting. CHANGES the project (confirmed + undo-wrapped). \
                          property is one of: mute (0/1), solo (0/1/2), volume (linear), pan \
                          (-1..1), visible_tcp (0/1), visible_mixer (0/1), height (px; 0=auto), \
                          height_lock (0/1), folder_depth (1 opens a folder, -1/-2.. closes), \
                          folder_compact (0/1/2), free_mode (0/1), color, rec_arm (0/1), \
                          rec_monitor (0/1/2), or name (pass 'text' instead of 'value')."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "property": { "type": "string" },
                    "value": { "type": "number" },
                    "text": { "type": "string", "description": "string value (for property=name)" }
                }),
                json!(["track_index", "property"]),
            ),
        },
        // --- grouping ---
        ToolDef {
            name: "get_track_group_membership".into(),
            description: "List which track groups (1..64) the track belongs to, per grouping \
                          parameter (e.g. VOLUME_LEAD, MUTE_FOLLOW, VOLUME_VCA_LEAD). Only \
                          parameters with any membership are returned."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "set_track_group_membership".into(),
            description: "Add or remove a track from a grouping. CHANGES the project (confirmed + \
                          undo-wrapped). group is 1..64; param is a REAPER group-name such as \
                          VOLUME_LEAD, VOLUME_FOLLOW, PAN_LEAD, MUTE_FOLLOW, SOLO_LEAD, \
                          VOLUME_VCA_LEAD, VOLUME_VCA_FOLLOW (LEAD = controls the group, FOLLOW = \
                          follows it); member true adds, false removes."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "group": { "type": "integer", "description": "group number 1..64" },
                    "param": { "type": "string", "description": "uppercase REAPER group-name, e.g. VOLUME_LEAD" },
                    "member": { "type": "boolean" }
                }),
                json!(["track_index", "group", "param", "member"]),
            ),
        },
        ToolDef {
            name: "group_items".into(),
            description: "Put several media items into a shared item group (so they select/move \
                          together). CHANGES the project (confirmed + undo-wrapped). Omit group_id \
                          to allocate a fresh unused group. Pass group_id 0 to ungroup."
                .into(),
            input_schema: obj(
                json!({
                    "item_indices": { "type": "array", "items": { "type": "integer" }, "description": "0-based project media item indices" },
                    "group_id": { "type": "integer", "description": "shared group id; omit to auto-allocate, 0 to ungroup" }
                }),
                json!(["item_indices"]),
            ),
        },
        // --- copy / move / delete ---
        ToolDef {
            name: "copy_item".into(),
            description: "Duplicate a media item (with all its takes) onto a track. CHANGES the \
                          project (confirmed + undo-wrapped). Defaults to the same track/position \
                          as the source; pass dest_track_index and/or position (seconds) to place \
                          the copy. Returns the new item_index."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "dest_track_index": { "type": "integer", "description": "target track (default: same as source)" },
                    "position": { "type": "number", "description": "new start in seconds (default: same as source)" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "move_item".into(),
            description: "Move an existing media item to another track and/or position. CHANGES \
                          the project (confirmed + undo-wrapped). Provide dest_track_index and/or \
                          position (seconds)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "dest_track_index": { "type": "integer" },
                    "position": { "type": "number" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "delete_item".into(),
            description: "Delete a media item from its track. CHANGES the project (confirmed + \
                          undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({ "item_index": { "type": "integer" } }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "duplicate_track".into(),
            description: "Duplicate a track (its FX, envelopes, routing and items) as a new track \
                          immediately below it. CHANGES the project (confirmed + undo-wrapped). \
                          Returns the new track_index."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "delete_track".into(),
            description: "Delete a track and all of its items. CHANGES the project (confirmed + \
                          undo-wrapped)."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "copy_take".into(),
            description: "Copy a take from one item to another as a new (inactive) take. CHANGES \
                          the project (confirmed + undo-wrapped). Supports plain file-based audio \
                          takes; for MIDI, in-project, or section/reverse sources use copy_item to \
                          duplicate the whole item. Defaults to the source item's active take."
                .into(),
            input_schema: obj(
                json!({
                    "src_item_index": { "type": "integer" },
                    "dest_item_index": { "type": "integer" },
                    "take_index": { "type": "integer", "description": "which take of the source; omit for its active take" }
                }),
                json!(["src_item_index", "dest_item_index"]),
            ),
        },
        // --- audio analysis (Phase 6) ---
        ToolDef {
            name: "analyze_item_audio".into(),
            description: "Analyse the audio of a media item's take (the source audio, PRE-FX): \
                          peak and RMS dBFS, crest factor, DC offset, clipping, integrated \
                          loudness (LUFS, BS.1770), and a rough spectral profile — centroid, \
                          dominant frequency, and low/mid/high energy balance. Reads up to 30 s. \
                          Defaults to the active take."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer", "description": "0-based project media item index" },
                    "take_index": { "type": "integer", "description": "0-based take index; omit for the active take" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "analyze_track_audio".into(),
            description: "Analyse a track's audio (its summed item output, PRE-FX and pre-fader): \
                          peak/RMS dBFS, crest factor, clipping, integrated loudness (LUFS), and a \
                          rough spectral profile. Optionally restrict to a start/length in seconds; \
                          reads up to 30 s. Note this is the source material BEFORE the track's FX \
                          chain and fader, not the processed output."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "start": { "type": "number", "description": "start in seconds (default: track start)" },
                    "length": { "type": "number", "description": "seconds to analyse (default: whole track, capped at 30 s)" }
                }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "analyze_audio_timeline".into(),
            description: "Analyse audio OVER TIME (a time-SERIES, not one aggregate number) for a \
                          take or a track: a per-window RMS/peak level envelope, detected SILENT \
                          regions, optional TRANSIENT onset times, and optional tracking of a single \
                          FREQUENCY's level over time. Read-only, local DSP (nothing is uploaded). \
                          Give item_index (+optional take_index) OR track_index (+optional \
                          start/length). Reads up to 30 s per call; all times reported are absolute \
                          timeline seconds. Use it to find silence, locate transients, or monitor \
                          how one frequency behaves across a longer passage."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer", "description": "analyse this item's take" },
                    "take_index": { "type": "integer", "description": "0-based; omit for the active take" },
                    "track_index": { "type": "integer", "description": "analyse this track's summed pre-FX audio instead of an item" },
                    "start": { "type": "number", "description": "track only: start time in seconds (default: track start)" },
                    "length": { "type": "number", "description": "seconds to analyse (capped at 30)" },
                    "window_ms": { "type": "number", "description": "envelope/silence window in ms (default 50)" },
                    "silence_threshold_db": { "type": "number", "description": "RMS dBFS below which a window counts as silent (default -60)" },
                    "min_silence_ms": { "type": "number", "description": "shortest silent region to report, in ms (default 200)" },
                    "detect_transients": { "type": "boolean", "description": "also return transient onset times (default false)" },
                    "target_hz": { "type": "number", "description": "if set, track this frequency's level over time" }
                }),
                json!([]),
            ),
        },
        // --- advanced: FX config + raw track state chunk (e.g. Video processor code) ---
        ToolDef {
            name: "get_fx_config".into(),
            description: "Read a named config value of a track FX (TrackFX_GetNamedConfigParm) — \
                          e.g. 'fx_type' (VST/JS/VIDEO/...), 'fx_name', 'fx_ident', 'renamed_name'. \
                          Read-only. 'fx_type' == 'VIDEO' confirms an FX is the Video processor. Its \
                          parameters and presets use the normal FX tools (get_fx_params / \
                          set_fx_param / get_fx_preset / set_fx_preset); only its EEL CODE needs the \
                          track state chunk below."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer" },
                    "name": { "type": "string", "description": "config parameter name, e.g. 'fx_type'" }
                }),
                json!(["track_index", "fx_index", "name"]),
            ),
        },
        ToolDef {
            name: "get_track_state_chunk".into(),
            description: "Read a track's full REAPER state chunk (RPP-format text: its FX chain — \
                          including the Video processor's EEL code — envelopes, item list, etc.). \
                          Read-only, and can be large. Use it for advanced edits the typed tools \
                          don't cover (notably reading/editing Video processor code), then write it \
                          back with set_track_state_chunk (edit minimally — change only what you \
                          intend)."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer" } }),
                json!(["track_index"]),
            ),
        },
        ToolDef {
            name: "set_track_state_chunk".into(),
            description: "Replace a track's full state chunk (RPP-format text from \
                          get_track_state_chunk). CHANGES the project (confirmed + undo-wrapped). \
                          ADVANCED/powerful: REAPER rejects a malformed chunk and the edit is \
                          undoable, but only pass a chunk derived from a FRESH, non-truncated \
                          get_track_state_chunk with a minimal, intentional change (e.g. the Video \
                          processor's code). Do not hand-craft chunks."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "chunk": { "type": "string", "description": "the full replacement state chunk" }
                }),
                json!(["track_index", "chunk"]),
            ),
        },
        // --- track/MIDI creation & deletion ---
        ToolDef {
            name: "create_track".into(),
            description: "Insert a new track. CHANGES the project (confirmed + undo-wrapped). \
                          'index' is the 0-based position (default: after the last track); 'name' \
                          is optional. Returns the new track_index."
                .into(),
            input_schema: obj(
                json!({
                    "index": { "type": "integer", "description": "0-based insert position (default: end)" },
                    "name": { "type": "string", "description": "track name (optional)" }
                }),
                json!([]),
            ),
        },
        ToolDef {
            name: "delete_midi_notes".into(),
            description: "Delete MIDI notes from a media item's active take. CHANGES the project \
                          (confirmed + undo-wrapped). With no filters it deletes ALL notes; \
                          otherwise only notes matching the pitch range (pitch_min/pitch_max, \
                          0-127) and/or time range (start_time/end_time in seconds, matched on the \
                          note's start)."
                .into(),
            input_schema: obj(
                json!({
                    "item_index": { "type": "integer" },
                    "pitch_min": { "type": "integer", "description": "lowest MIDI pitch to delete (0-127)" },
                    "pitch_max": { "type": "integer", "description": "highest MIDI pitch to delete (0-127)" },
                    "start_time": { "type": "number", "description": "only notes starting at/after this time (seconds)" },
                    "end_time": { "type": "number", "description": "only notes starting at/before this time (seconds)" }
                }),
                json!(["item_index"]),
            ),
        },
        ToolDef {
            name: "set_fx_param_by_steps".into(),
            description: "Nudge a track-FX parameter UP or DOWN by a relative amount, instead of \
                          setting an absolute 0..1 value. Use this when you can SEE a knob/slider \
                          (e.g. via capture_view) and want to move it 'a little' without computing \
                          the exact normalized value — it reads the current value, steps it, and \
                          reports the new display value so you can judge the effect and repeat. \
                          CHANGES the project (confirmed + undo-wrapped). direction is 'up' or \
                          'down'; step_count is how many steps (default 1); step_size is the \
                          normalized amount per step in 0..1 (default 0.01)."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_index": { "type": "integer", "description": "0-based FX index in the track chain" },
                    "param_index": { "type": "integer" },
                    "direction": { "type": "string", "enum": ["up", "down"], "description": "nudge direction" },
                    "step_count": { "type": "integer", "description": "number of steps to move (default 1)" },
                    "step_size": { "type": "number", "description": "normalized amount per step, 0..1 (default 0.01)" },
                    "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
                }),
                json!(["track_index", "fx_index", "param_index", "direction"]),
            ),
        },
        // --- vision (Phase 7) ---
        ToolDef {
            name: "capture_view".into(),
            description: "Screenshot visual-only UI the REAPER API can't express (custom plugin \
                          GUIs, meters, waveforms), returned as an image to reason about. Needs \
                          user consent each time (sent to the AI provider); call only when seeing \
                          genuinely helps. Seeing is not acting — to CHANGE anything use the \
                          parameter tools (set_fx_param), not this. Pass track_index + fx_index \
                          (from get_track_fx) to open and capture a SPECIFIC unfocused track FX \
                          (the reliable way to view a plugin you just added); these take precedence \
                          over target. Otherwise target: 'focused_plugin' (default), 'reaper_main', \
                          'full_screen', or 'video' (REAPER's Video window — the processed video \
                          frame at the cursor, with all video FX applied; the Video window must be \
                          open and floating). Capturing 'video' repeatedly during playback is a way \
                          to watch video processing, but each frame is a separate consent-gated \
                          still, not a live stream."
                .into(),
            input_schema: obj(
                json!({
                    "target": {
                        "type": "string",
                        "enum": ["focused_plugin", "reaper_main", "full_screen", "video"],
                        "description": "what to capture when track_index/fx_index are not given (default 'focused_plugin')"
                    },
                    "track_index": {
                        "type": "integer",
                        "description": "0-based track index of a specific track FX to open + capture (use together with fx_index)"
                    },
                    "fx_index": {
                        "type": "integer",
                        "description": "0-based FX index within that track's chain to open + capture (use together with track_index)"
                    }
                }),
                json!([]),
            ),
        },
        ToolDef {
            name: "capture_video_clip".into(),
            description: "See REAPER's video output AS A SHORT CLIP, not a single still: captures \
                          several evenly-spaced frames of the Video window across a time range (by \
                          stepping the edit cursor, with playback stopped) and returns them as images \
                          in time order, plus the clip's audio for audio-capable models. Use this \
                          over repeated capture_view('video') to judge motion, cuts, transitions, \
                          text/credits timing, or audio/video sync. One consent covers the whole clip. \
                          The Video window must be open and floating (View -> Video). Range: pass \
                          start + length (seconds), or leave both out to use the current time \
                          selection. frames defaults to 6 (2-12). More frames = more detail but more \
                          cost. Seeing is not acting."
                .into(),
            input_schema: obj(
                json!({
                    "start": { "type": "number", "description": "clip start in seconds (omit to use the time selection)" },
                    "length": { "type": "number", "description": "clip length in seconds (omit to use the time selection)" },
                    "frames": { "type": "integer", "description": "how many frames to sample across the range (default 6, clamped 2-12)" },
                    "include_audio": { "type": "boolean", "description": "also render+attach the clip's audio (default true; ignored for models without audio input)" }
                }),
                json!([]),
            ),
        },
        // --- Tier B: operate GUI-only plugin controls via synthesized input ---
        ToolDef {
            name: "plugin_click".into(),
            description: "Click a GUI-only control (no host parameter, e.g. a Kontakt mode/patch \
                          toggle) in the FOCUSED plugin by synthesizing a real mouse click; for \
                          normal knobs/sliders prefer set_fx_param(_by_steps), which are undoable. \
                          x,y are PIXELS in the latest capture_view image (top-left 0,0), clamped \
                          to the window. Just CALL it — the first pixel action asks the user once \
                          to allow pixel control; never tell the user to enable it themselves \
                          (no such setting exists). NOT undoable by REAPER — call capture_view after \
                          to verify."
                .into(),
            input_schema: obj(
                json!({
                    "x": { "type": "integer", "description": "x pixel in the plugin screenshot" },
                    "y": { "type": "integer", "description": "y pixel in the plugin screenshot" },
                    "double": { "type": "boolean", "description": "double-click instead of single (default false)" }
                }),
                json!(["x", "y"]),
            ),
        },
        ToolDef {
            name: "plugin_type".into(),
            description: "Type text into the FOCUSED plugin's currently focused control — e.g. a \
                          value field you just clicked (plugin_click) to enter a number or name. \
                          Set submit=true to press Enter afterward. Same rules as plugin_click: \
                          pixel control is requested automatically on first use, NOT undoable by \
                          REAPER; verify with capture_view."
                .into(),
            input_schema: obj(
                json!({
                    "text": { "type": "string", "description": "text to type into the focused field" },
                    "submit": { "type": "boolean", "description": "press Enter after typing (default false)" }
                }),
                json!(["text"]),
            ),
        },
        ToolDef {
            name: "plugin_scroll".into(),
            description: "Mouse-wheel scroll inside the FOCUSED plugin's window at pixel (x,y) from \
                          the most recent capture_view image — e.g. to scroll a preset/browser \
                          list. clicks is the number of wheel notches: positive scrolls up/away, \
                          negative down/toward. Pixel control is requested automatically on first \
                          use; verify with capture_view."
                .into(),
            input_schema: obj(
                json!({
                    "x": { "type": "integer" },
                    "y": { "type": "integer" },
                    "clicks": { "type": "integer", "description": "wheel notches; +up / -down" }
                }),
                json!(["x", "y", "clicks"]),
            ),
        },
        ToolDef {
            name: "plugin_drag".into(),
            description: "Drag within the FOCUSED plugin's GUI from (x1,y1) to (x2,y2) — the way a \
                          knob is turned (usually a vertical drag). Coordinates are PIXELS in the \
                          most recent capture_view image, clamped to the window. Same rules as \
                          plugin_click: pixel control is requested automatically on first use, and \
                          the change is NOT undoable by REAPER; verify with capture_view afterwards."
                .into(),
            input_schema: obj(
                json!({
                    "x1": { "type": "integer" },
                    "y1": { "type": "integer" },
                    "x2": { "type": "integer" },
                    "y2": { "type": "integer" }
                }),
                json!(["x1", "y1", "x2", "y2"]),
            ),
        },
        ToolDef {
            name: "disable_pixel_control".into(),
            description: "Turn OFF pixel control (stop being able to click/drag in plugin GUIs). \
                          Call this when the user asks you to stop operating a plugin's GUI or says \
                          you're done. Always allowed; does not change the project."
                .into(),
            input_schema: empty(),
        },
    ];
    defs.push(ToolDef {
        name: "analyze_processed_audio".into(),
        description: "Analyse PROCESSED (post-FX) audio by doing a short offline render and \
                      measuring the result — the same metrics as analyze_track_audio (peak/RMS, \
                      loudness LUFS, clipping, spectral profile) but WITH the FX applied. \
                      target 'master' = full mix (all track FX + master FX); 'track' (with \
                      track_index) = that track through its FX + master; 'item' (with \
                      item_index) = that item through its take + track FX. Briefly renders \
                      offline (REAPER may be momentarily unresponsive; not cancellable). Window \
                      capped at 30 s. Use analyze_track_audio / analyze_item_audio for the raw \
                      pre-FX source."
            .into(),
        input_schema: obj(
            json!({
                "target": { "type": "string", "enum": ["master", "track", "item"], "description": "'master' = full mix, 'track' = one track's processed output, 'item' = one item through its take+track FX" },
                "track_index": { "type": "integer", "description": "required when target is 'track'" },
                "item_index": { "type": "integer", "description": "required when target is 'item'" },
                "start": { "type": "number", "description": "start in seconds for master/track (default: time selection or 0)" },
                "length": { "type": "number", "description": "seconds to render for master/track (default: content/selection, capped at 30 s)" }
            }),
            json!(["target"]),
        ),
    });
    defs.push(ToolDef {
        name: "listen_to_audio".into(),
        description: "LISTEN to the processed audio: briefly renders the master mix or a track's \
                      post-FX output to a short clip and returns it as audio for you to HEAR, so you \
                      can judge tone, balance, artifacts, noise, timing — things the numeric DSP \
                      metrics don't convey. Only call this when actually hearing the sound helps; it \
                      needs the user's consent (the clip is sent to the AI provider). target \
                      'master' = full mix; 'track' (with track_index) = one track's processed \
                      output. Optional start/length seconds (else the time selection, else the \
                      start; capped at 20 s to keep the clip small). For numbers (loudness, peak, \
                      spectrum) use analyze_processed_audio / measure_loudness instead."
            .into(),
        input_schema: obj(
            json!({
                "target": { "type": "string", "enum": ["master", "track"], "description": "'master' = full mix, 'track' = one track's processed output" },
                "track_index": { "type": "integer", "description": "0-based track index (required when target is 'track')" },
                "start": { "type": "number", "description": "start in seconds (default: time selection or 0)" },
                "length": { "type": "number", "description": "seconds to render (default: content/selection, capped at 20 s)" }
            }),
            json!(["target"]),
        ),
    });
    defs.push(ToolDef {
        name: "measure_loudness".into(),
        description: "Measure professional loudness of the PROCESSED output: briefly renders the mix \
                      offline through the FX chain, then measures it with BS.1770 DSP. Returns \
                      integrated loudness (LUFS-I), loudness range (LRA in LU), true-peak (dBTP), \
                      momentary/short-term LUFS maxima, sample peak (dBFS), RMS and clipping — the \
                      EBU R128 / streaming-target metrics. target 'master' measures the full mix \
                      (incl. master FX); 'track' (with track_index) measures a track's post-FX \
                      output. Optional start/length seconds (else the time selection, else up to \
                      30 s of content). Nothing is left in the project and there is no undo point. \
                      REAPER is briefly unresponsive during the render; not cancellable."
            .into(),
        input_schema: obj(
            json!({
                "target": { "type": "string", "enum": ["master", "track"], "description": "'master' = full mix, 'track' = one track's processed output" },
                "track_index": { "type": "integer", "description": "0-based track index (required when target is 'track')" },
                "start": { "type": "number", "description": "start in seconds (default: time selection or 0)" },
                "length": { "type": "number", "description": "seconds to measure (default: content/selection, capped at 30 s)" }
            }),
            json!(["target"]),
        ),
    });
    // --- transport / timeline / global settings ---
    defs.push(ToolDef {
        name: "get_transport".into(),
        description: "Read the transport + timeline state: play_state \
                      ('stopped'|'playing'|'paused'|'recording'), play_position and edit_cursor \
                      (seconds), playrate (1.0 = normal speed), tempo (BPM), repeat/loop on/off, \
                      and the ruler time unit."
            .into(),
        input_schema: empty(),
    });
    defs.push(ToolDef {
        name: "transport".into(),
        description: "Control the transport. action: play, stop, pause, record, or toggle_play \
                      (play/stop). Recording writes to record-armed tracks (arm via \
                      set_track_property rec_arm); stop ends play or recording. Not undo-wrapped."
            .into(),
        input_schema: obj(
            json!({ "action": { "type": "string", "enum": ["play", "stop", "pause", "record", "toggle_play"] } }),
            json!(["action"]),
        ),
    });
    defs.push(ToolDef {
        name: "set_edit_cursor".into(),
        description: "Move the edit cursor to a position in seconds. seek_playback (default false) \
                      also moves the play cursor during playback; move_view (default false) scrolls \
                      the arrange view to show it."
            .into(),
        input_schema: obj(
            json!({
                "position": { "type": "number", "description": "seconds" },
                "seek_playback": { "type": "boolean" },
                "move_view": { "type": "boolean" }
            }),
            json!(["position"]),
        ),
    });
    defs.push(ToolDef {
        name: "get_time_selection".into(),
        description: "Read the current time selection (the arrange loop/edit range): start, end and \
                      length in seconds, and is_set (false when there is no selection). Read-only."
            .into(),
        input_schema: empty(),
    });
    defs.push(ToolDef {
        name: "set_time_selection".into(),
        description: "Set the time selection (and the matching loop range) to [start, end] in \
                      seconds; start/end are ordered and clamped to >= 0. seek (default false) also \
                      moves the edit cursor to the start (and seeks playback there). Transient \
                      view/selection state, like moving the cursor — not undo-wrapped. Useful to \
                      mark a range for the analyze_*/render/measure_loudness tools, which default to \
                      the time selection when no explicit range is given."
            .into(),
        input_schema: obj(
            json!({
                "start": { "type": "number", "description": "selection start in seconds" },
                "end": { "type": "number", "description": "selection end in seconds" },
                "seek": { "type": "boolean", "description": "also move the edit cursor to the start (default false)" }
            }),
            json!(["start", "end"]),
        ),
    });
    defs.push(ToolDef {
        name: "clear_time_selection".into(),
        description: "Remove the time selection (and loop range). Transient, not undo-wrapped."
            .into(),
        input_schema: empty(),
    });
    // --- REAPER actions + keyboard shortcuts ---
    defs.push(ToolDef {
        name: "search_actions".into(),
        description: "Search REAPER's Main action list by name (case-insensitive substring). \
                      Returns matching actions with their command_id, name, toggle_state \
                      (on/off/null), and bound key shortcuts. Use this to find an action to run \
                      (run_action) or whose shortcuts to inspect/edit. Results are capped (limit, \
                      default 40); 'truncated' says whether more matched."
            .into(),
        input_schema: obj(
            json!({
                "query": { "type": "string", "description": "text to match in the action name" },
                "limit": { "type": "integer", "description": "max results (default 40, max 200)" }
            }),
            json!(["query"]),
        ),
    });
    defs.push(ToolDef {
        name: "get_action_info".into(),
        description: "Read one action's name, toggle_state (on/off/null), and key shortcuts. \
                      command is a numeric id (e.g. '40364') or a named command (e.g. '_SWS_ABOUT'). \
                      Read-only."
            .into(),
        input_schema: obj(
            json!({ "command": { "type": "string", "description": "numeric id or named command" } }),
            json!(["command"]),
        ),
    });
    defs.push(ToolDef {
        name: "run_action".into(),
        description: "Run a REAPER action by command id or named command (Main action section). \
                      CHANGES the project (confirmed). A catch-all for anything without a dedicated \
                      tool; the action manages its own undo. Prefer the dedicated tools where they \
                      exist. Returns the action's new toggle_state where applicable."
            .into(),
        input_schema: obj(
            json!({ "command": { "type": "string", "description": "numeric id or named command" } }),
            json!(["command"]),
        ),
    });
    defs.push(ToolDef {
        name: "delete_action_shortcut".into(),
        description: "Remove a keyboard shortcut from an action. CHANGES the key map (confirmed). \
                      command is the action; shortcut_index is 0-based into the action's shortcut \
                      list (see get_action_info/search_actions). Returns the remaining shortcuts."
            .into(),
        input_schema: obj(
            json!({
                "command": { "type": "string", "description": "numeric id or named command" },
                "shortcut_index": { "type": "integer", "description": "0-based index of the shortcut to remove" }
            }),
            json!(["command", "shortcut_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "add_action_shortcut".into(),
        description: "Assign a NEW keyboard shortcut to an action. REAPER has no API to set a key \
                      directly, so this opens its interactive key-assignment dialog for the USER to \
                      press the desired combo (confirmed before opening). Use delete_action_shortcut \
                      to remove one non-interactively."
            .into(),
        input_schema: obj(
            json!({ "command": { "type": "string", "description": "numeric id or named command" } }),
            json!(["command"]),
        ),
    });
    defs.push(ToolDef {
        name: "set_playrate".into(),
        description: "Set the master playback speed (playrate). 1.0 = normal, 2.0 = double, 0.5 = \
                      half; usable range ~0.25..4.0. Pitch is preserved unless REAPER's 'preserve \
                      pitch when changing playrate' is off."
            .into(),
        input_schema: obj(json!({ "rate": { "type": "number" } }), json!(["rate"])),
    });
    defs.push(ToolDef {
        name: "set_ruler_unit".into(),
        description: "Set the timeline/ruler time unit. unit: minutes_seconds, measures_beats, \
                      measures_beats_minutes, seconds, samples, or hmsf (hours:minutes:seconds:frames)."
            .into(),
        input_schema: obj(
            json!({ "unit": { "type": "string", "enum": ["minutes_seconds", "measures_beats", "measures_beats_minutes", "seconds", "samples", "hmsf"] } }),
            json!(["unit"]),
        ),
    });
    defs.push(ToolDef {
        name: "get_global_toggles".into(),
        description: "Read common project/global on-off states: metronome, repeat (loop), snap \
                      (snapping), and ripple editing mode ('off'|'one_track'|'all_tracks')."
            .into(),
        input_schema: empty(),
    });
    defs.push(ToolDef {
        name: "set_global_toggle".into(),
        description: "Turn a global option on or off. name: metronome, repeat, or snap. on = the \
                      desired state. (For ripple editing use set_ripple_mode.)"
            .into(),
        input_schema: obj(
            json!({ "name": { "type": "string", "enum": ["metronome", "repeat", "snap"] }, "on": { "type": "boolean" } }),
            json!(["name", "on"]),
        ),
    });
    defs.push(ToolDef {
        name: "set_ripple_mode".into(),
        description: "Set ripple-editing mode: off, one_track, or all_tracks.".into(),
        input_schema: obj(
            json!({ "mode": { "type": "string", "enum": ["off", "one_track", "all_tracks"] } }),
            json!(["mode"]),
        ),
    });
    // --- CRUD completeness ---
    defs.push(ToolDef {
        name: "remove_fx".into(),
        description: "Remove an FX from a track's FX chain by index (get_track_fx for indices; use \
                      'chain' for input or monitoring FX). CHANGES the project (confirmed + \
                      undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "fx_index": { "type": "integer" },
                "chain": { "type": "string", "enum": ["normal", "input", "monitor"], "description": CHAIN_DESC }
            }),
            json!(["track_index", "fx_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "remove_take_fx".into(),
        description: "Remove an FX from a take's FX chain by index (defaults to the active take). \
                      CHANGES the project (confirmed + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "item_index": { "type": "integer" },
                "fx_index": { "type": "integer" },
                "take_index": { "type": "integer", "description": "0-based; omit for the active take" }
            }),
            json!(["item_index", "fx_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "create_empty_item".into(),
        description: "Create an empty media item (no take) on a track at a position, with a length \
                      in seconds. CHANGES the project (confirmed + undo-wrapped). For a MIDI item \
                      use create_midi_item."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "position": { "type": "number", "description": "seconds" },
                "length": { "type": "number", "description": "seconds" }
            }),
            json!(["track_index", "position", "length"]),
        ),
    });
    defs.push(ToolDef {
        name: "update_marker".into(),
        description: "Edit an existing marker by its index_number (from get_markers). Optionally set \
                      position (seconds), name, and/or color (native colour|0x1000000; omit or 0 = \
                      keep current). CHANGES the project (confirmed + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "index_number": { "type": "integer" },
                "position": { "type": "number" },
                "name": { "type": "string" },
                "color": { "type": "integer" }
            }),
            json!(["index_number"]),
        ),
    });
    defs.push(ToolDef {
        name: "update_region".into(),
        description: "Edit an existing region by its index_number (from get_markers). Optionally set \
                      start and/or end (seconds), name, and/or color. CHANGES the project \
                      (confirmed + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "index_number": { "type": "integer" },
                "start": { "type": "number" },
                "end": { "type": "number" },
                "name": { "type": "string" },
                "color": { "type": "integer" }
            }),
            json!(["index_number"]),
        ),
    });
    defs.push(ToolDef {
        name: "delete_envelope_points".into(),
        description: "Delete all envelope points in the time range [start, end] seconds on a track \
                      envelope (index from get_track_envelopes). CHANGES the project (confirmed + \
                      undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "envelope_index": { "type": "integer" },
                "start": { "type": "number" },
                "end": { "type": "number" }
            }),
            json!(["track_index", "envelope_index", "start", "end"]),
        ),
    });
    defs.push(ToolDef {
        name: "create_fx_envelope".into(),
        description: "Create (or get, if it already exists) the automation envelope for a track FX \
                      PARAMETER, so it can then be automated with insert_envelope_point. CHANGES \
                      the project (confirmed + undo-wrapped). Returns the new envelope_index. \
                      Indices from get_track_fx / get_fx_params."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "fx_index": { "type": "integer" },
                "param_index": { "type": "integer" }
            }),
            json!(["track_index", "fx_index", "param_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "create_track_envelope".into(),
        description: "Create a built-in track automation envelope so insert_envelope_point can \
                      target it. kind: volume, pan, or mute. If it already exists it is returned \
                      unchanged (never hidden). CHANGES the project (confirmed + undo-wrapped). \
                      Returns envelope_index."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "kind": { "type": "string", "enum": ["volume", "pan", "mute"] }
            }),
            json!(["track_index", "kind"]),
        ),
    });
    defs.push(ToolDef {
        name: "set_envelope_point".into(),
        description: "Edit an existing envelope point by its 0-based point_index (order from \
                      get_envelope_points). Optionally set time (seconds), value, and/or shape \
                      (0=linear, 1=square, 2=slow start/end, 3=fast start, 4=fast end, 5=bezier); \
                      omitted fields are left unchanged. CHANGES the project (confirmed + \
                      undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "envelope_index": { "type": "integer" },
                "point_index": { "type": "integer" },
                "time": { "type": "number" },
                "value": { "type": "number" },
                "shape": { "type": "integer" }
            }),
            json!(["track_index", "envelope_index", "point_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "delete_envelope_point".into(),
        description: "Delete a single envelope point by its 0-based point_index. (To delete a time \
                      range of points, use delete_envelope_points.) CHANGES the project (confirmed \
                      + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "envelope_index": { "type": "integer" },
                "point_index": { "type": "integer" }
            }),
            json!(["track_index", "envelope_index", "point_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "clear_envelope".into(),
        description: "Remove ALL points from an envelope, clearing its automation. NOTE: REAPER has \
                      no API to delete the envelope lane object itself — the (empty) lane remains; \
                      it can be hidden manually. CHANGES the project (confirmed + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "envelope_index": { "type": "integer" }
            }),
            json!(["track_index", "envelope_index"]),
        ),
    });
    defs.push(ToolDef {
        name: "create_send_envelope".into(),
        description: "Create a send/receive automation envelope (volume, pan, or mute) so it can \
                      be automated with the point tools. category: 'send' (default) reads \
                      (track_index, send_index) as a send on the source track; 'receive' as a \
                      receive on the destination — same routing either way. Returned unchanged if \
                      it already exists. IMPORTANT: to add points afterwards use the RETURNED \
                      envelope_track_index + envelope_index (the envelope is owned by the source \
                      track, which may differ from the index you passed). Fails safe (no change) if \
                      it can't confirm creation. CHANGES the project (confirmed + undo-wrapped)."
            .into(),
        input_schema: obj(
            json!({
                "track_index": { "type": "integer" },
                "send_index": { "type": "integer" },
                "kind": { "type": "string", "enum": ["volume", "pan", "mute"] },
                "category": { "type": "string", "enum": ["send", "receive"] }
            }),
            json!(["track_index", "send_index", "kind"]),
        ),
    });
    if !supports_images {
        defs.retain(|d| !is_vision_tool(&d.name));
    }
    if !supports_audio {
        defs.retain(|d| !is_audio_tool(&d.name));
    }
    defs
}

#[cfg(test)]
mod definition_tests {
    #[test]
    fn processed_render_tool_advertised() {
        // Guards against a regression where the post-FX render tool is silently
        // dropped from the advertised set.
        let defs = super::definitions(true, false);
        assert!(
            defs.iter().any(|d| d.name == "analyze_processed_audio"),
            "analyze_processed_audio must be advertised"
        );
    }

    #[test]
    fn loudness_probe_advertised_regardless_of_vision() {
        // measure_loudness is not a vision tool, so it must be offered whether or
        // not the model supports images.
        for supports_images in [true, false] {
            let defs = super::definitions(supports_images, false);
            assert!(
                defs.iter().any(|d| d.name == "measure_loudness"),
                "measure_loudness must be advertised (supports_images={supports_images})"
            );
        }
    }

    #[test]
    fn transport_and_crud_tools_advertised() {
        let defs = super::definitions(false, false);
        for name in [
            "transport",
            "get_transport",
            "set_edit_cursor",
            "set_playrate",
            "set_ruler_unit",
            "get_global_toggles",
            "set_global_toggle",
            "set_ripple_mode",
            "remove_fx",
            "remove_take_fx",
            "create_empty_item",
            "update_marker",
            "update_region",
            "delete_envelope_points",
        ] {
            assert!(
                defs.iter().any(|d| d.name == name),
                "{name} must be advertised"
            );
        }
    }

    #[test]
    fn fx_chain_parse_and_raw_index() {
        use super::FxChain;
        assert!(matches!(FxChain::parse("").unwrap(), FxChain::Normal));
        assert!(matches!(FxChain::parse("normal").unwrap(), FxChain::Normal));
        assert!(matches!(FxChain::parse("input").unwrap(), FxChain::Input));
        assert!(matches!(FxChain::parse("MONITOR").unwrap(), FxChain::Monitor));
        assert!(FxChain::parse("bogus").is_err());
        // input/monitor address the record chain by OR-ing 0x1000000; normal doesn't.
        assert_eq!(FxChain::parse("normal").unwrap().raw_index(2), 2);
        assert_eq!(FxChain::parse("input").unwrap().raw_index(2), 2 | 0x0100_0000);
        assert_eq!(FxChain::parse("monitor").unwrap().raw_index(0), 0x0100_0000);
    }

    #[test]
    fn fx_preset_tools_advertised_and_gated() {
        use serde_json::json;
        let defs = super::definitions(false, false);
        for name in [
            "get_fx_preset",
            "set_fx_preset",
            "get_take_fx_preset",
            "set_take_fx_preset",
        ] {
            assert!(defs.iter().any(|d| d.name == name), "{name} advertised");
        }
        // Loading a preset changes the FX state -> confirmation-gated.
        assert!(super::preview("set_fx_preset", &json!({})).is_some());
        assert!(super::preview("set_take_fx_preset", &json!({})).is_some());
        // Reads are not gated.
        assert!(super::preview("get_fx_preset", &json!({})).is_none());
        assert!(super::preview("get_take_fx_preset", &json!({})).is_none());
    }

    #[test]
    fn envelope_tools_advertised_and_gated() {
        use serde_json::json;
        let defs = super::definitions(false, false);
        for name in [
            "create_fx_envelope",
            "create_track_envelope",
            "create_send_envelope",
            "set_envelope_point",
            "delete_envelope_point",
            "clear_envelope",
        ] {
            assert!(
                defs.iter().any(|d| d.name == name),
                "{name} must be advertised"
            );
            assert!(
                super::preview(name, &json!({})).is_some(),
                "{name} must be confirmation-gated"
            );
        }
    }

    #[test]
    fn crud_mutations_are_confirmation_gated() {
        // The content-changing new tools must require confirmation; the transport/
        // control tools must NOT (they aren't content mutations).
        use serde_json::json;
        for name in [
            "remove_fx",
            "remove_take_fx",
            "create_empty_item",
            "update_marker",
            "update_region",
            "delete_envelope_points",
        ] {
            assert!(
                super::preview(name, &json!({})).is_some(),
                "{name} must be confirmation-gated"
            );
        }
        for name in ["transport", "set_edit_cursor", "set_playrate", "set_global_toggle"] {
            assert!(
                super::preview(name, &json!({})).is_none(),
                "{name} must not be confirmation-gated"
            );
        }
    }

    #[test]
    fn listen_tool_gated_on_audio() {
        // listen_to_audio only appears when the model can hear.
        assert!(
            !super::definitions(true, false)
                .iter()
                .any(|d| d.name == "listen_to_audio"),
            "listen_to_audio must be withheld without audio support"
        );
        assert!(
            super::definitions(false, true)
                .iter()
                .any(|d| d.name == "listen_to_audio"),
            "listen_to_audio must be advertised with audio support"
        );
    }
}

/// Execute a tool by name on the main thread. Never panics.
pub fn execute(reaper: &Reaper<MainThreadScope>, name: &str, input: &Value) -> ToolOutcome {
    // Vision/audio tools return media alongside their text, so they bypass the
    // plain `dispatch` (which only yields JSON) and build their own outcome.
    if name == "capture_view" {
        return capture_view(reaper, input);
    }
    if name == "listen_to_audio" {
        return listen_to_audio(reaper, input);
    }
    match dispatch(reaper, name, input) {
        Ok(v) => ToolOutcome::ok(v.to_string()),
        Err(e) => ToolOutcome::error(json!({ "error": e }).to_string()),
    }
}

/// Capture a screenshot and return it to the model as an image (Phase 7 vision).
/// The user has already consented by the time this runs (see [`consent_prompt`]).
/// Supports the REAPER main window and the focused plugin's window (full-screen
/// lands with the robustness pass).
fn capture_view(reaper: &Reaper<MainThreadScope>, input: &Value) -> ToolOutcome {
    // Description-only targets (main window / full screen) can be huge, so cap
    // their long edge to stay under the vision API's size limit. Plugin windows
    // are left 1:1 so Tier-B click coordinates map exactly to the window.
    const MAX_EDGE: u32 = 1280;

    // Explicit track FX (track_index + fx_index) opens + captures that specific
    // plugin regardless of focus — the reliable path for weaker models and for a
    // plugin the user hasn't focused. It takes precedence over `target`.
    let (shot, target_label): (Result<_, _>, String) =
        match (opt_u32(input, "track_index"), opt_u32(input, "fx_index")) {
            (Some(ti), Some(fi)) => {
                let hwnd = match open_track_fx_window(reaper, ti, fi) {
                    Ok(h) => h,
                    Err(e) => return ToolOutcome::error(json!({ "error": e }).to_string()),
                };
                (crate::ui::screenshot::capture_hwnd(hwnd, true, None), "track_fx".to_string())
            }
            _ => {
                let target = input
                    .get("target")
                    .and_then(|v| v.as_str())
                    .unwrap_or("focused_plugin");
                let shot = match target {
                    "focused_plugin" => {
                        let hwnd = match resolve_focused_fx_hwnd(reaper) {
                            Ok(h) => h,
                            Err(e) => return ToolOutcome::error(json!({ "error": e }).to_string()),
                        };
                        crate::ui::screenshot::capture_hwnd(hwnd, true, None)
                    }
                    "reaper_main" => {
                        let hwnd = reaper.get_main_hwnd().as_ptr() as isize;
                        crate::ui::screenshot::capture_hwnd(hwnd, false, Some(MAX_EDGE))
                    }
                    "full_screen" => crate::ui::screenshot::capture_screen(Some(MAX_EDGE)),
                    "video" => {
                        // REAPER renders the fully-processed video frame (all video
                        // FX applied) into its Video window at the play/edit cursor.
                        // There's no API for that window's handle, so find the
                        // floating window by title and screen-capture it.
                        match crate::ui::ffi::find_window_by_title("Video Window") {
                            Some(hwnd) => {
                                crate::ui::screenshot::capture_hwnd(hwnd, true, Some(MAX_EDGE))
                            }
                            None => {
                                return ToolOutcome::error(
                                    json!({
                                        "error": "The REAPER Video window isn't open (or is docked). Open it via View -> Video and make it a floating window, then try again."
                                    })
                                    .to_string(),
                                );
                            }
                        }
                    }
                    other => {
                        return ToolOutcome::error(
                            json!({
                                "error": format!(
                                    "capture target '{other}' is not supported; use 'focused_plugin', 'reaper_main', 'full_screen', 'video', or pass track_index+fx_index"
                                )
                            })
                            .to_string(),
                        );
                    }
                };
                (shot, target.to_string())
            }
        };

    match shot {
        Ok(shot) => {
            let summary = json!({
                "captured": true,
                "target": target_label,
                "width": shot.width,
                "height": shot.height,
                "note": "The screenshot is attached as an image for you to view.",
            })
            .to_string();
            ToolOutcome::with_image(
                summary,
                CapturedImage {
                    media_type: "image/png".into(),
                    data_base64: shot.png_base64,
                },
            )
        }
        Err(e) => {
            ToolOutcome::error(json!({ "error": format!("screenshot failed: {e}") }).to_string())
        }
    }
}

/// Internal op backing `capture_video_clip` (not advertised to the model). Resolve
/// the capture range and the current edit-cursor position (so it can be restored),
/// and fail fast if the Video window isn't open/floating. Range = explicit
/// `start`+`length`, else the current time selection.
fn video_begin(reaper: &Reaper<MainThreadScope>, input: &Value) -> Result<Value, String> {
    if crate::ui::ffi::find_window_by_title("Video Window").is_none() {
        return Err("The REAPER Video window isn't open (or is docked). Open it via View -> Video \
                    and make it a floating window, then try again."
            .to_string());
    }
    let low = reaper.low();
    // The Video window follows the PLAY cursor while the transport runs, so seeking
    // the EDIT cursor per frame would leave every frame identical. Require a stopped
    // transport (bit 0 = playing, bit 2 = recording) and let the user/model stop it.
    let play_state = low.GetPlayState();
    if play_state & 1 != 0 || play_state & 4 != 0 {
        return Err("Playback is running. Stop the transport first — while playing, REAPER's Video \
                    window follows the play cursor, so the captured frames wouldn't match the \
                    requested range."
            .to_string());
    }
    let orig = low.GetCursorPosition();
    let (t0, t1) = match (
        input.get("start").and_then(|v| v.as_f64()),
        input.get("length").and_then(|v| v.as_f64()),
    ) {
        (Some(s), Some(l)) if l > 0.0 => (s.max(0.0), (s + l).max(0.0)),
        _ => {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            unsafe { low.GetSet_LoopTimeRange(false, false, &mut a, &mut b, false) };
            if b > a {
                (a, b)
            } else {
                return Err("No time range given and no time selection is set. Pass start+length, \
                            or set a time selection first."
                    .to_string());
            }
        }
    };
    Ok(json!({ "t0": t0, "t1": t1, "orig_cursor": orig }))
}

/// Open (float) a specific track FX's window and return its `HWND` as an `isize`,
/// so a plugin the user hasn't focused can still be screenshotted / driven.
fn open_track_fx_window(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
) -> Result<isize, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    float_track_fx(reaper, track, TrackFxLocation::NormalFxChain(fx_index))
        .ok_or_else(|| format!("could not open the floating window for FX {fx_index} on track {track_index}"))
}

/// Return a track FX's floating-window `HWND` (as `isize`), forcing the window
/// open if it isn't already. `None` only if the window can't be opened. Shared by
/// the focused-FX path and the explicit track+fx path.
fn float_track_fx(
    reaper: &Reaper<MainThreadScope>,
    track: MediaTrack,
    fx_location: TrackFxLocation,
) -> Option<isize> {
    // SAFETY: track is a live pointer from the medium API; main thread.
    if let Some(h) = unsafe { reaper.track_fx_get_floating_window(track, fx_location) } {
        return Some(h.as_ptr() as isize);
    }
    unsafe {
        reaper.track_fx_show(track, FxShowInstruction::ShowFloatingWindow(fx_location));
    }
    unsafe { reaper.track_fx_get_floating_window(track, fx_location) }.map(|h| h.as_ptr() as isize)
}

/// Resolve the currently/last focused FX to its floating-window `HWND` (as an
/// `isize`), so it can be screenshotted. Embedded (in-chain) FX have no window
/// of their own, so we force a floating window first. Covers track FX and take
/// FX; the master track is handled via `TrackLocation::MasterTrack`.
fn resolve_focused_fx_hwnd(reaper: &Reaper<MainThreadScope>) -> Result<isize, String> {
    let res = reaper
        .get_touched_or_focused_fx_currently_focused_fx()
        .ok_or("no plugin is focused — open or click a plugin window first")?;

    match res.fx {
        FxLocation::TrackFx {
            track_location,
            fx_location,
        } => {
            let project = ProjectContext::CurrentProject;
            let track = track_from_location(reaper, project, track_location)?;
            float_track_fx(reaper, track, fx_location)
                .ok_or_else(|| "could not open the plugin's floating window".to_string())
        }
        FxLocation::TakeFx {
            item_index,
            take_index,
            fx_index,
            ..
        } => {
            let item = reaper
                .get_media_item(ProjectContext::CurrentProject, item_index)
                .ok_or_else(|| format!("no media item at index {item_index}"))?;
            let low = reaper.low();
            // SAFETY: main thread; item is a live pointer from the medium API.
            let take = unsafe { low.GetMediaItemTake(item.as_ptr(), take_index as c_int) };
            if take.is_null() {
                return Err(format!("no take at index {take_index}"));
            }
            let idx = fx_index as c_int;
            // SAFETY: take valid, main thread.
            let existing = unsafe { low.TakeFX_GetFloatingWindow(take, idx) };
            if !existing.is_null() {
                return Ok(existing as isize);
            }
            unsafe { low.TakeFX_Show(take, idx, 3) }; // 3 = show floating window
            let h = unsafe { low.TakeFX_GetFloatingWindow(take, idx) };
            if h.is_null() {
                Err("could not open the take FX floating window".to_string())
            } else {
                Ok(h as isize)
            }
        }
    }
}

/// The `MediaTrack` for a focused-FX track location (master or a normal track).
fn track_from_location(
    reaper: &Reaper<MainThreadScope>,
    project: ProjectContext,
    loc: TrackLocation,
) -> Result<MediaTrack, String> {
    match loc {
        TrackLocation::MasterTrack => Ok(reaper.get_master_track(project)),
        TrackLocation::NormalTrack(index) => reaper
            .get_track(project, index)
            .ok_or_else(|| format!("no track at index {index}")),
    }
}

/// Synthesize a left click at image-space `(x, y)` in the focused plugin's
/// window (Phase 7 Tier B — for GUI-only controls with no host parameter). The
/// user has already armed pixel control by the time this runs. Coordinates are
/// clamped to the window in the input backend, so a click can't leave it.
fn plugin_click(
    reaper: &Reaper<MainThreadScope>,
    x: i32,
    y: i32,
    double: bool,
) -> Result<Value, String> {
    let hwnd = resolve_focused_fx_hwnd(reaper)?;
    if double {
        crate::ui::input::double_click(hwnd, x, y)?;
    } else {
        crate::ui::input::click(hwnd, x, y)?;
    }
    Ok(json!({
        "clicked": true, "double": double, "x": x, "y": y,
        "note": "Synthesized a GUI click. This is NOT undoable by REAPER. Use \
                 capture_view to verify the result changed as intended."
    }))
}

/// Type text into the focused plugin's focused control (e.g. a value field just
/// clicked). See [`plugin_click`] for the safety model.
fn plugin_type(
    reaper: &Reaper<MainThreadScope>,
    text: &str,
    submit: bool,
) -> Result<Value, String> {
    let hwnd = resolve_focused_fx_hwnd(reaper)?;
    crate::ui::input::type_text(hwnd, text, submit)?;
    Ok(json!({
        "typed": true, "chars": text.chars().count(), "submit": submit,
        "note": "Typed into the plugin's focused control. NOT undoable by REAPER; \
                 use capture_view to verify."
    }))
}

/// Mouse-wheel scroll within the focused plugin's window (e.g. a browser list).
/// `clicks` is wheel notches; positive scrolls up/away, negative down/toward.
fn plugin_scroll(
    reaper: &Reaper<MainThreadScope>,
    x: i32,
    y: i32,
    clicks: i32,
) -> Result<Value, String> {
    let hwnd = resolve_focused_fx_hwnd(reaper)?;
    crate::ui::input::scroll(hwnd, x, y, clicks)?;
    Ok(json!({
        "scrolled": true, "x": x, "y": y, "clicks": clicks,
        "note": "Scrolled the plugin. Use capture_view to verify."
    }))
}

/// Synthesize a left-button drag from `(x1, y1)` to `(x2, y2)` in the focused
/// plugin's window (e.g. turning a knob vertically). See [`plugin_click`].
fn plugin_drag(
    reaper: &Reaper<MainThreadScope>,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
) -> Result<Value, String> {
    let hwnd = resolve_focused_fx_hwnd(reaper)?;
    crate::ui::input::drag(hwnd, x1, y1, x2, y2)?;
    Ok(json!({
        "dragged": true, "from": [x1, y1], "to": [x2, y2],
        "note": "Synthesized a GUI drag. This is NOT undoable by REAPER. Use \
                 capture_view to verify the result."
    }))
}

fn dispatch(reaper: &Reaper<MainThreadScope>, name: &str, input: &Value) -> Result<Value, String> {
    match name {
        "get_project_summary" => Ok(get_project_summary(reaper)),
        "get_tracks" => Ok(get_tracks(reaper)),
        "get_track_fx" => get_track_fx(
            reaper,
            req_u32(input, "track_index")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "get_fx_params" => get_fx_params(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            opt_usize(input, "limit").unwrap_or(DEFAULT_LIMIT),
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "get_selected_items" => Ok(get_selected_items(reaper)),
        "get_take_fx" => get_take_fx(reaper, req_u32(input, "item_index")?),
        "get_take_fx_params" => get_take_fx_params(
            reaper,
            req_u32(input, "item_index")?,
            req_u32(input, "fx_index")?,
            opt_usize(input, "limit").unwrap_or(DEFAULT_LIMIT),
        ),
        "list_installed_fx" => Ok(list_installed_fx(
            reaper,
            opt_str(input, "filter"),
            opt_usize(input, "limit").unwrap_or(DEFAULT_LIMIT),
        )),
        "get_focused_fx" => Ok(get_focused_fx(reaper)),
        // mutating
        "add_fx" => add_fx(
            reaper,
            req_u32(input, "track_index")?,
            req_str(input, "fx_name")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "set_fx_param" => set_fx_param(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_u32(input, "param_index")?,
            req_f64(input, "value")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "set_fx_param_by_steps" => set_fx_param_by_steps(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_u32(input, "param_index")?,
            req_str(input, "direction")?,
            opt_u32(input, "step_count").unwrap_or(1),
            input
                .get("step_size")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.01),
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        // Tier B pixel input (arm-gated in the worker before reaching here).
        "plugin_click" => plugin_click(
            reaper,
            req_u32(input, "x")? as i32,
            req_u32(input, "y")? as i32,
            opt_bool(input, "double").unwrap_or(false),
        ),
        "plugin_drag" => plugin_drag(
            reaper,
            req_u32(input, "x1")? as i32,
            req_u32(input, "y1")? as i32,
            req_u32(input, "x2")? as i32,
            req_u32(input, "y2")? as i32,
        ),
        "plugin_type" => plugin_type(
            reaper,
            req_str(input, "text")?,
            opt_bool(input, "submit").unwrap_or(false),
        ),
        "plugin_scroll" => plugin_scroll(
            reaper,
            req_u32(input, "x")? as i32,
            req_u32(input, "y")? as i32,
            input.get("clicks").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        ),
        "disable_pixel_control" => {
            disarm_pixel_control();
            Ok(json!({ "pixel_control": "disabled" }))
        }
        "set_fx_enabled" => set_fx_enabled(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_bool(input, "enabled")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "get_fx_preset" => get_fx_preset(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "set_fx_preset" => set_fx_preset(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            opt_str(input, "name"),
            input.get("index").and_then(|v| v.as_i64()),
            opt_str(input, "navigate"),
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "get_take_fx_preset" => get_take_fx_preset(
            reaper,
            req_u32(input, "item_index")?,
            req_u32(input, "fx_index")?,
            opt_u32(input, "take_index"),
        ),
        "set_take_fx_preset" => set_take_fx_preset(
            reaper,
            req_u32(input, "item_index")?,
            req_u32(input, "fx_index")?,
            opt_u32(input, "take_index"),
            opt_str(input, "name"),
            input.get("index").and_then(|v| v.as_i64()),
            opt_str(input, "navigate"),
        ),
        // MIDI
        "get_take_midi" => get_take_midi(
            reaper,
            req_u32(input, "item_index")?,
            opt_bool(input, "include_neighbors").unwrap_or(false),
        ),
        "insert_midi_notes" => insert_midi_notes(
            reaper,
            req_u32(input, "item_index")?,
            input
                .get("notes")
                .ok_or_else(|| "missing 'notes' array".to_string())?,
        ),
        "create_midi_item" => create_midi_item(
            reaper,
            req_u32(input, "track_index")?,
            req_f64(input, "start_qn")?,
            req_f64(input, "length_qn")?,
        ),
        // routing
        "get_track_sends" => get_track_sends(reaper, req_u32(input, "track_index")?),
        "add_send" => add_send(
            reaper,
            req_u32(input, "src_track_index")?,
            req_u32(input, "dest_track_index")?,
        ),
        "set_send_param" => set_send_param(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "send_index")?,
            req_str(input, "param")?,
            req_f64(input, "value")?,
        ),
        "remove_send" => remove_send(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "send_index")?,
        ),
        // automation / envelopes
        "get_track_envelopes" => get_track_envelopes(reaper, req_u32(input, "track_index")?),
        "get_envelope_points" => get_envelope_points(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
            opt_usize(input, "limit").unwrap_or(DEFAULT_LIMIT),
        ),
        "get_automation_items" => get_automation_items(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
        ),
        "insert_envelope_point" => insert_envelope_point(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
            req_f64(input, "time")?,
            req_f64(input, "value")?,
            input.get("shape").and_then(|v| v.as_i64()).unwrap_or(0) as c_int,
        ),
        "delete_envelope_points" => delete_envelope_points(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
            req_f64(input, "start")?,
            req_f64(input, "end")?,
        ),
        "create_fx_envelope" => create_fx_envelope(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_u32(input, "param_index")?,
        ),
        "create_track_envelope" => {
            create_track_envelope(reaper, req_u32(input, "track_index")?, req_str(input, "kind")?)
        }
        "set_envelope_point" => set_envelope_point(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
            req_u32(input, "point_index")?,
            input.get("time").and_then(|v| v.as_f64()),
            input.get("value").and_then(|v| v.as_f64()),
            input.get("shape").and_then(|v| v.as_i64()).map(|s| s as c_int),
        ),
        "delete_envelope_point" => delete_envelope_point(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
            req_u32(input, "point_index")?,
        ),
        "clear_envelope" => clear_envelope(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "envelope_index")?,
        ),
        "create_send_envelope" => create_send_envelope(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "send_index")?,
            req_str(input, "kind")?,
            opt_str(input, "category").unwrap_or("send"),
        ),
        // transport / timeline / global settings
        "get_transport" => Ok(get_transport(reaper)),
        "transport" => transport(reaper, req_str(input, "action")?),
        "set_edit_cursor" => set_edit_cursor(
            reaper,
            req_f64(input, "position")?,
            opt_bool(input, "seek_playback").unwrap_or(false),
            opt_bool(input, "move_view").unwrap_or(false),
        ),
        // Internal (not model-advertised) op for capture_video_clip: resolve the
        // capture range + save the current cursor, failing fast if the Video window
        // isn't available. Orchestrated from the worker (see run_tool).
        "__video_begin" => video_begin(reaper, input),
        "get_time_selection" => Ok(get_time_selection(reaper)),
        "set_time_selection" => set_time_selection(
            reaper,
            req_f64(input, "start")?,
            req_f64(input, "end")?,
            opt_bool(input, "seek").unwrap_or(false),
        ),
        "clear_time_selection" => clear_time_selection(reaper),
        "search_actions" => search_actions(
            reaper,
            req_str(input, "query")?,
            opt_u32(input, "limit").unwrap_or(40) as usize,
        ),
        "get_action_info" => get_action_info(reaper, req_str(input, "command")?),
        "run_action" => run_action(reaper, req_str(input, "command")?),
        "delete_action_shortcut" => delete_action_shortcut(
            reaper,
            req_str(input, "command")?,
            req_u32(input, "shortcut_index")?,
        ),
        "add_action_shortcut" => add_action_shortcut(reaper, req_str(input, "command")?),
        "set_playrate" => set_playrate(reaper, req_f64(input, "rate")?),
        "set_ruler_unit" => set_ruler_unit(reaper, req_str(input, "unit")?),
        "get_global_toggles" => Ok(get_global_toggles(reaper)),
        "set_global_toggle" => {
            set_global_toggle(reaper, req_str(input, "name")?, req_bool(input, "on")?)
        }
        "set_ripple_mode" => set_ripple_mode(reaper, req_str(input, "mode")?),
        // CRUD completeness
        "remove_fx" => remove_fx(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            opt_str(input, "chain").unwrap_or("normal"),
        ),
        "remove_take_fx" => remove_take_fx(
            reaper,
            req_u32(input, "item_index")?,
            req_u32(input, "fx_index")?,
            opt_u32(input, "take_index"),
        ),
        "create_empty_item" => create_empty_item(
            reaper,
            req_u32(input, "track_index")?,
            req_f64(input, "position")?,
            req_f64(input, "length")?,
        ),
        "update_marker" => update_marker(
            reaper,
            req_i64(input, "index_number")? as c_int,
            false,
            input.get("position").and_then(|v| v.as_f64()),
            None,
            opt_str(input, "name"),
            input.get("color").and_then(|v| v.as_i64()).map(|c| c as c_int),
        ),
        "update_region" => update_marker(
            reaper,
            req_i64(input, "index_number")? as c_int,
            true,
            input.get("start").and_then(|v| v.as_f64()),
            input.get("end").and_then(|v| v.as_f64()),
            opt_str(input, "name"),
            input.get("color").and_then(|v| v.as_i64()).map(|c| c as c_int),
        ),
        // notes & per-project memory
        "get_project_notes" => Ok(get_project_notes(reaper)),
        "set_project_notes" => Ok(set_project_notes(
            reaper,
            req_str(input, "text")?,
            opt_bool(input, "append").unwrap_or(true),
        )),
        "get_track_notes" => get_track_notes(reaper, req_u32(input, "track_index")?),
        "set_track_notes" => set_track_notes(
            reaper,
            req_u32(input, "track_index")?,
            req_str(input, "text")?,
        ),
        "get_project_memory" => get_project_memory(reaper, opt_str(input, "key")),
        "set_project_memory" => set_project_memory(
            reaper,
            req_str(input, "key")?,
            input.get("value").and_then(|v| v.as_str()).unwrap_or(""),
        ),
        "delete_project_memory" => delete_project_memory(reaper, req_str(input, "key")?),
        // markers & regions
        "get_markers" => Ok(get_markers(reaper)),
        "add_marker" => add_marker(
            reaper,
            req_f64(input, "position")?,
            opt_str(input, "name").unwrap_or(""),
            input
                .get("index_number")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as c_int,
        ),
        "add_region" => add_region(
            reaper,
            req_f64(input, "start")?,
            req_f64(input, "end")?,
            opt_str(input, "name").unwrap_or(""),
            input
                .get("index_number")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as c_int,
        ),
        "delete_marker" => delete_marker(
            reaper,
            req_i64(input, "index_number")? as c_int,
            opt_bool(input, "is_region").unwrap_or(false),
        ),
        // tempo / time-signature map
        "get_tempo_markers" => Ok(get_tempo_markers(reaper)),
        "add_tempo_marker" => add_tempo_marker(
            reaper,
            req_f64(input, "time")?,
            req_f64(input, "bpm")?,
            input
                .get("timesig_num")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as c_int,
            input
                .get("timesig_denom")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as c_int,
            opt_bool(input, "linear").unwrap_or(false),
        ),
        "delete_tempo_marker" => delete_tempo_marker(reaper, req_i64(input, "index")? as c_int),
        "set_project_tempo" => set_project_tempo(reaper, req_f64(input, "bpm")?),
        // stretch markers
        "get_stretch_markers" => get_stretch_markers(reaper, req_u32(input, "item_index")?),
        "add_stretch_marker" => add_stretch_marker(
            reaper,
            req_u32(input, "item_index")?,
            req_f64(input, "position")?,
            input.get("src_position").and_then(|v| v.as_f64()),
        ),
        "delete_stretch_marker" => delete_stretch_marker(
            reaper,
            req_u32(input, "item_index")?,
            req_i64(input, "marker_index")? as c_int,
            input
                .get("count")
                .and_then(|v| v.as_i64())
                .map(|c| c as c_int),
        ),
        // render settings
        "get_render_settings" => Ok(get_render_settings(reaper)),
        "set_render_setting" => set_render_setting(
            reaper,
            req_str(input, "key")?,
            input.get("value").and_then(|v| v.as_f64()),
            opt_str(input, "text"),
        ),
        // item properties
        "get_item_properties" => get_item_properties(reaper, req_u32(input, "item_index")?),
        "set_item_property" => set_item_property(
            reaper,
            req_u32(input, "item_index")?,
            req_str(input, "property")?,
            req_f64(input, "value")?,
        ),
        "set_item_edge" => set_item_edge(
            reaper,
            req_u32(input, "item_index")?,
            req_str(input, "edge")?,
            req_f64(input, "time")?,
        ),
        // take properties
        "get_take_properties" => get_take_properties(
            reaper,
            req_u32(input, "item_index")?,
            opt_u32(input, "take_index"),
        ),
        "set_take_property" => set_take_property(
            reaper,
            req_u32(input, "item_index")?,
            req_str(input, "property")?,
            input.get("value").and_then(|v| v.as_f64()),
            opt_str(input, "text"),
            opt_u32(input, "take_index"),
        ),
        "set_active_take" => set_active_take(
            reaper,
            req_u32(input, "item_index")?,
            req_u32(input, "take_index")?,
        ),
        // track settings
        "get_track_properties" => get_track_properties(reaper, req_u32(input, "track_index")?),
        "set_track_property" => set_track_property(
            reaper,
            req_u32(input, "track_index")?,
            req_str(input, "property")?,
            input.get("value").and_then(|v| v.as_f64()),
            opt_str(input, "text"),
        ),
        // grouping
        "get_track_group_membership" => {
            get_track_group_membership(reaper, req_u32(input, "track_index")?)
        }
        "set_track_group_membership" => set_track_group_membership(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "group")?,
            req_str(input, "param")?,
            req_bool(input, "member")?,
        ),
        "group_items" => {
            let arr = input
                .get("item_indices")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "missing 'item_indices' array".to_string())?;
            let idxs: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
            group_items(
                reaper,
                &idxs,
                input.get("group_id").and_then(|v| v.as_i64()),
            )
        }
        // copy / move / delete
        "copy_item" => copy_item(
            reaper,
            req_u32(input, "item_index")?,
            opt_u32(input, "dest_track_index"),
            input.get("position").and_then(|v| v.as_f64()),
        ),
        "move_item" => move_item(
            reaper,
            req_u32(input, "item_index")?,
            opt_u32(input, "dest_track_index"),
            input.get("position").and_then(|v| v.as_f64()),
        ),
        "delete_item" => delete_item(reaper, req_u32(input, "item_index")?),
        "duplicate_track" => duplicate_track(reaper, req_u32(input, "track_index")?),
        "delete_track" => delete_track(reaper, req_u32(input, "track_index")?),
        "copy_take" => copy_take(
            reaper,
            req_u32(input, "src_item_index")?,
            req_u32(input, "dest_item_index")?,
            opt_u32(input, "take_index"),
        ),
        // audio analysis
        "analyze_item_audio" => analyze_item_audio(
            reaper,
            req_u32(input, "item_index")?,
            opt_u32(input, "take_index"),
        ),
        "analyze_track_audio" => analyze_track_audio(
            reaper,
            req_u32(input, "track_index")?,
            input.get("start").and_then(|v| v.as_f64()),
            input.get("length").and_then(|v| v.as_f64()),
        ),
        "analyze_audio_timeline" => analyze_audio_timeline(
            reaper,
            opt_u32(input, "item_index"),
            opt_u32(input, "take_index"),
            opt_u32(input, "track_index"),
            input.get("start").and_then(|v| v.as_f64()),
            input.get("length").and_then(|v| v.as_f64()),
            input.get("window_ms").and_then(|v| v.as_f64()).unwrap_or(50.0),
            input.get("silence_threshold_db").and_then(|v| v.as_f64()).unwrap_or(-60.0),
            input.get("min_silence_ms").and_then(|v| v.as_f64()).unwrap_or(200.0),
            input.get("detect_transients").and_then(|v| v.as_bool()).unwrap_or(false),
            input.get("target_hz").and_then(|v| v.as_f64()),
        ),
        "analyze_processed_audio" => analyze_processed_audio(reaper, input),
        "measure_loudness" => measure_loudness(reaper, input),
        // advanced: FX config + raw track state chunk
        "get_fx_config" => get_fx_config(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_str(input, "name")?,
        ),
        "get_track_state_chunk" => {
            get_track_state_chunk(reaper, req_u32(input, "track_index")?)
        }
        "set_track_state_chunk" => set_track_state_chunk(
            reaper,
            req_u32(input, "track_index")?,
            req_str(input, "chunk")?,
        ),
        // track / MIDI creation & deletion
        "create_track" => create_track(reaper, opt_u32(input, "index"), opt_str(input, "name")),
        "delete_midi_notes" => delete_midi_notes(
            reaper,
            req_u32(input, "item_index")?,
            input.get("pitch_min").and_then(|v| v.as_i64()),
            input.get("pitch_max").and_then(|v| v.as_i64()),
            input.get("start_time").and_then(|v| v.as_f64()),
            input.get("end_time").and_then(|v| v.as_f64()),
        ),
        // undo / history
        "undo" => Ok(undo(reaper)),
        "redo" => Ok(redo(reaper)),
        "get_undo_history" => Ok(get_undo_history(reaper)),
        other => Err(format!("unknown tool: {other}")),
    }
}

/// Human-readable preview for a mutating tool call, or None for read/undo tools.
/// Returning `Some` marks the tool as requiring confirmation.
// ---- Tier-B pixel control (Phase 7): arm-per-task state ----------------------
// When the user "arms" pixel control, the assistant may synthesize clicks/drags
// into plugin windows (for GUI-only controls with no host parameter) until it is
// disarmed. Arming is human-gated (worker consent); disarming is always allowed.
static PIXEL_ARMED: AtomicBool = AtomicBool::new(false);

/// Whether the assistant is currently allowed to synthesize plugin-GUI input.
pub fn is_pixel_armed() -> bool {
    PIXEL_ARMED.load(Ordering::Relaxed)
}

/// Arm pixel control (call only after explicit user consent).
pub fn arm_pixel_control() {
    PIXEL_ARMED.store(true, Ordering::Relaxed);
}

/// Disarm pixel control (always safe; e.g. the dialog closed or the user said stop).
pub fn disarm_pixel_control() {
    PIXEL_ARMED.store(false, Ordering::Relaxed);
}

/// Whether a tool synthesizes plugin-GUI input (gated by the arm consent).
pub fn is_pixel_tool(name: &str) -> bool {
    matches!(
        name,
        "plugin_click" | "plugin_drag" | "plugin_type" | "plugin_scroll"
    )
}

/// A mandatory consent gate for tools that send data off the machine (Phase 7
/// screen capture). Returns `Some(description-of-what-is-captured)` for such
/// tools, or `None`. Unlike mutation confirmation, this gate is ALWAYS enforced
/// (data protection) and is independent of the mutation-confirm toggle.
pub fn consent_prompt(name: &str, input: &Value) -> Option<String> {
    match name {
        "capture_view" => {
            // Default MUST match capture_view's own default (focused_plugin), or the
            // consent dialog would name a different window than the one captured.
            let target = input
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("focused_plugin");
            // Every screenshot is consent-gated, always — a capture sends pixels
            // to the cloud, so it is never covered by the pixel-control arm (which
            // only authorizes LOCAL input), and consent is never persistent (per
            // the design's data-protection rule). The arm and this gate are
            // deliberately independent.
            let what = match target {
                "reaper_main" => "the REAPER main window",
                "focused_plugin" => "the focused plugin window",
                "full_screen" => "the entire screen",
                "video" => "the REAPER video window (the processed video frame)",
                _ => "a window",
            };
            Some(format!(
                "The assistant wants to take a screenshot of {what}"
            ))
        }
        // Rendering a clip and sending it to the cloud provider is the same class
        // of data-protection concern as a screenshot — always consent-gated.
        "listen_to_audio" => {
            let what = match input.get("target").and_then(|v| v.as_str()) {
                Some("track") => "a track's processed audio",
                _ => "the processed master audio",
            };
            Some(format!("The assistant wants to render and listen to {what}"))
        }
        // A video clip = several frames of the Video window + the span's audio, all
        // sent to the cloud. One consent covers the whole clip (the internal frame
        // captures / audio render are orchestrated in the worker and are not each
        // re-prompted).
        "capture_video_clip" => Some(
            "The assistant wants to capture a short video clip (several frames of the REAPER Video \
             window, plus the clip's audio) and send it to the AI provider"
                .to_string(),
        ),
        _ => None,
    }
}

pub fn preview(name: &str, input: &Value) -> Option<String> {
    let show = |k: &str| {
        input
            .get(k)
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into())
    };
    match name {
        "add_fx" => Some(format!(
            "Add FX {} to track {}{}",
            input
                .get("fx_name")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "?".into()),
            show("track_index"),
            chain_suffix(input),
        )),
        "set_fx_param" => Some(format!(
            "Set track {} FX {} parameter {} to {} (normalized 0..1){}",
            show("track_index"),
            show("fx_index"),
            show("param_index"),
            show("value"),
            chain_suffix(input),
        )),
        "set_fx_param_by_steps" => Some(format!(
            "Nudge track {} FX {} parameter {} {} by {} step(s){}",
            show("track_index"),
            show("fx_index"),
            show("param_index"),
            input
                .get("direction")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            input
                .get("step_count")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "1".into()),
            chain_suffix(input),
        )),
        "set_fx_preset" => Some(format!(
            "Load a preset ({}) for track {} FX {}{}",
            preset_sel_desc(input),
            show("track_index"),
            show("fx_index"),
            chain_suffix(input),
        )),
        "set_take_fx_preset" => Some(format!(
            "Load a preset ({}) for item {} take FX {}",
            preset_sel_desc(input),
            show("item_index"),
            show("fx_index"),
        )),
        "set_fx_enabled" => Some(format!(
            "{} track {} FX {}{}",
            if input
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Enable"
            } else {
                "Bypass"
            },
            show("track_index"),
            show("fx_index"),
            chain_suffix(input),
        )),
        "insert_midi_notes" => Some(format!(
            "Insert {} MIDI note(s) into item {}",
            input
                .get("notes")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
            show("item_index"),
        )),
        "create_midi_item" => Some(format!(
            "Create a MIDI item on track {} (start {} QN, length {} QN)",
            show("track_index"),
            show("start_qn"),
            show("length_qn"),
        )),
        "add_send" => Some(format!(
            "Add a send from track {} to track {}",
            show("src_track_index"),
            show("dest_track_index"),
        )),
        "set_send_param" => Some(format!(
            "Set send {} {} to {} on track {}",
            show("send_index"),
            input.get("param").and_then(|v| v.as_str()).unwrap_or("?"),
            show("value"),
            show("track_index"),
        )),
        "remove_send" => Some(format!(
            "Remove send {} from track {}",
            show("send_index"),
            show("track_index"),
        )),
        "insert_envelope_point" => Some(format!(
            "Insert an automation point on track {} envelope {} at time {} = {}",
            show("track_index"),
            show("envelope_index"),
            show("time"),
            show("value"),
        )),
        "delete_envelope_points" => Some(format!(
            "Delete automation points on track {} envelope {} from {} s to {} s",
            show("track_index"),
            show("envelope_index"),
            show("start"),
            show("end"),
        )),
        "create_fx_envelope" => Some(format!(
            "Create an automation envelope for FX {} param {} on track {}",
            show("fx_index"),
            show("param_index"),
            show("track_index"),
        )),
        "create_track_envelope" => Some(format!(
            "Create the {} envelope on track {}",
            show("kind"),
            show("track_index"),
        )),
        "set_envelope_point" => Some(format!(
            "Edit point {} of envelope {} on track {}",
            show("point_index"),
            show("envelope_index"),
            show("track_index"),
        )),
        "delete_envelope_point" => Some(format!(
            "Delete point {} of envelope {} on track {}",
            show("point_index"),
            show("envelope_index"),
            show("track_index"),
        )),
        "clear_envelope" => Some(format!(
            "Clear all points from envelope {} on track {}",
            show("envelope_index"),
            show("track_index"),
        )),
        "create_send_envelope" => Some(format!(
            "Create the {} envelope for {} {} on track {}",
            show("kind"),
            opt_str(input, "category").unwrap_or("send"),
            show("send_index"),
            show("track_index"),
        )),
        "remove_fx" => Some(format!(
            "Remove FX {} from track {}{}",
            show("fx_index"),
            show("track_index"),
            chain_suffix(input),
        )),
        "remove_take_fx" => Some(format!(
            "Remove take FX {} from item {}",
            show("fx_index"),
            show("item_index"),
        )),
        "create_empty_item" => Some(format!(
            "Create an empty item on track {} at {} s ({} s long)",
            show("track_index"),
            show("position"),
            show("length"),
        )),
        "update_marker" => Some(format!("Edit marker number {}", show("index_number"))),
        "update_region" => Some(format!("Edit region number {}", show("index_number"))),
        "add_marker" => Some(format!(
            "Add marker {} at {} s",
            input
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "(unnamed)".into()),
            show("position"),
        )),
        "add_region" => Some(format!(
            "Add region {} from {} s to {} s",
            input
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "(unnamed)".into()),
            show("start"),
            show("end"),
        )),
        "delete_marker" => Some(format!(
            "Delete {} number {}",
            if input
                .get("is_region")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "region"
            } else {
                "marker"
            },
            show("index_number"),
        )),
        "add_tempo_marker" => Some(format!(
            "Add tempo marker at {} s = {} BPM",
            show("time"),
            show("bpm"),
        )),
        "delete_tempo_marker" => Some(format!("Delete tempo marker {}", show("index"))),
        "set_project_tempo" => Some(format!("Set project tempo to {} BPM", show("bpm"))),
        "add_stretch_marker" => Some(format!(
            "Add stretch marker to item {} at {} s",
            show("item_index"),
            show("position"),
        )),
        "delete_stretch_marker" => Some(format!(
            "Delete stretch marker {} from item {}",
            show("marker_index"),
            show("item_index"),
        )),
        "set_render_setting" => Some(format!(
            "Set render setting {} to {}",
            input.get("key").and_then(|v| v.as_str()).unwrap_or("?"),
            input
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .or_else(|| input.get("value").map(|v| v.to_string()))
                .unwrap_or_else(|| "?".into()),
        )),
        "set_item_property" => Some(format!(
            "Set item {} {} to {}",
            show("item_index"),
            input
                .get("property")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            show("value"),
        )),
        "set_item_edge" => Some(format!(
            "Move item {} {} edge to {}s",
            show("item_index"),
            input.get("edge").and_then(|v| v.as_str()).unwrap_or("?"),
            show("time"),
        )),
        "set_track_state_chunk" => Some(format!(
            "Replace the full state chunk of track {}",
            show("track_index"),
        )),
        "run_action" => Some(format!(
            "Run REAPER action {}",
            input.get("command").and_then(|v| v.as_str()).unwrap_or("?"),
        )),
        "delete_action_shortcut" => Some(format!(
            "Remove shortcut #{} from action {}",
            show("shortcut_index"),
            input.get("command").and_then(|v| v.as_str()).unwrap_or("?"),
        )),
        "add_action_shortcut" => Some(format!(
            "Open the key-assignment dialog for action {}",
            input.get("command").and_then(|v| v.as_str()).unwrap_or("?"),
        )),
        "set_take_property" => Some(format!(
            "Set take {} of item {} to {}",
            input
                .get("property")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            show("item_index"),
            input
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .or_else(|| input.get("value").map(|v| v.to_string()))
                .unwrap_or_else(|| "?".into()),
        )),
        "set_active_take" => Some(format!(
            "Set active take {} on item {}",
            show("take_index"),
            show("item_index"),
        )),
        "set_track_property" => Some(format!(
            "Set track {} {} to {}",
            show("track_index"),
            input
                .get("property")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            input
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .or_else(|| input.get("value").map(|v| v.to_string()))
                .unwrap_or_else(|| "?".into()),
        )),
        "set_track_group_membership" => Some(format!(
            "{} track {} {} group {}",
            if input
                .get("member")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Add"
            } else {
                "Remove"
            },
            show("track_index"),
            input.get("param").and_then(|v| v.as_str()).unwrap_or("?"),
            show("group"),
        )),
        "group_items" => Some(format!(
            "Group {} item(s)",
            input
                .get("item_indices")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
        )),
        "copy_item" => Some(format!("Copy item {}", show("item_index"))),
        "move_item" => Some(format!("Move item {}", show("item_index"))),
        "delete_item" => Some(format!("Delete item {}", show("item_index"))),
        "duplicate_track" => Some(format!("Duplicate track {}", show("track_index"))),
        "delete_track" => Some(format!("Delete track {}", show("track_index"))),
        "copy_take" => Some(format!(
            "Copy a take from item {} to item {}",
            show("src_item_index"),
            show("dest_item_index"),
        )),
        "create_track" => Some(format!(
            "Create a track{}",
            input
                .get("name")
                .and_then(|v| v.as_str())
                .map(|n| format!(" named \"{n}\""))
                .unwrap_or_default(),
        )),
        "delete_midi_notes" => Some(format!(
            "Delete MIDI notes from item {}",
            show("item_index")
        )),
        _ => None,
    }
}

// ---- tools ------------------------------------------------------------------

fn get_project_summary(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let cursor = reaper
        .get_cursor_position_ex(project)
        .map(|p| p.get())
        .unwrap_or(0.0);
    let low = reaper.low();
    // Project file name (e.g. "song.rpp") and full path; empty when unsaved.
    let name = read_string(1024, |b, s| {
        unsafe { low.GetProjectName(CUR_PROJ, b, s) };
        true
    })
    .unwrap_or_default();
    let path = read_string(4096, |b, s| {
        unsafe { low.EnumProjects(-1, b, s) };
        true
    })
    .unwrap_or_default();
    json!({
        "project_name": if name.is_empty() { "(unsaved)".to_string() } else { name },
        "project_path": path,
        "tempo": reaper.master_get_tempo().get(),
        "track_count": reaper.count_tracks(project),
        "selected_tracks": reaper.count_selected_tracks_2(project, MasterTrackBehavior::ExcludeMasterTrack),
        "selected_items": reaper.count_selected_media_items(project),
        "edit_cursor_seconds": cursor,
    })
}

fn get_tracks(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let selected = selected_track_set(reaper);
    let mut tracks = Vec::new();
    for i in 0..reaper.count_tracks(project) {
        if let Some(t) = reaper.get_track(project, i) {
            tracks.push(json!({
                "index": i,
                "name": track_name(reaper, t),
                "selected": selected.contains(&t),
            }));
        }
    }
    json!({ "tracks": tracks })
}

/// Which FX chain a track/monitor FX tool targets. `Normal` = the track's main
/// (output) FX; `Input` = its input/record FX (applied before recording);
/// `Monitor` = REAPER's global Monitoring FX, which live on the master track's
/// input chain (so `track_index` is ignored for `Monitor`).
#[derive(Clone, Copy)]
enum FxChain {
    Normal,
    Input,
    Monitor,
}

impl FxChain {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "" | "normal" | "main" | "output" => Ok(FxChain::Normal),
            "input" | "rec" | "record" => Ok(FxChain::Input),
            "monitor" | "mon" | "monitoring" => Ok(FxChain::Monitor),
            other => Err(format!(
                "unknown chain '{other}' (use normal, input, or monitor)"
            )),
        }
    }

    /// Input or monitor (both are addressed as an input/record chain) vs normal.
    fn is_input(self) -> bool {
        !matches!(self, FxChain::Normal)
    }

    fn location(self, fx_index: u32) -> TrackFxLocation {
        if self.is_input() {
            TrackFxLocation::InputFxChain(fx_index)
        } else {
            TrackFxLocation::NormalFxChain(fx_index)
        }
    }

    fn chain_type(self) -> TrackFxChainType {
        if self.is_input() {
            TrackFxChainType::InputFxChain
        } else {
            TrackFxChainType::NormalFxChain
        }
    }

    /// Raw FX index for the low-level `TrackFX_*` calls that take a plain int:
    /// input/monitor FX are addressed by OR-ing 0x1000000 into the index.
    fn raw_index(self, fx_index: u32) -> c_int {
        if self.is_input() {
            (fx_index | 0x0100_0000) as c_int
        } else {
            fx_index as c_int
        }
    }

    /// The track this chain lives on. Monitoring FX are on the master track, so
    /// `track_index` is ignored for `Monitor`.
    fn track(
        self,
        reaper: &Reaper<MainThreadScope>,
        track_index: u32,
    ) -> Result<MediaTrack, String> {
        let project = ProjectContext::CurrentProject;
        match self {
            FxChain::Monitor => Ok(reaper.get_master_track(project)),
            _ => reaper
                .get_track(project, track_index)
                .ok_or_else(|| format!("no track at index {track_index}")),
        }
    }

    /// Number of FX in this chain on the given track.
    fn count(self, reaper: &Reaper<MainThreadScope>, track: MediaTrack) -> u32 {
        if self.is_input() {
            unsafe { reaper.track_fx_get_rec_count(track) }
        } else {
            unsafe { reaper.track_fx_get_count(track) }
        }
    }

    fn label(self) -> &'static str {
        match self {
            FxChain::Normal => "normal",
            FxChain::Input => "input",
            FxChain::Monitor => "monitor",
        }
    }
}

fn get_track_fx(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let track = chain.track(reaper, track_index)?;
    // SAFETY: track just obtained from REAPER, used on the main thread here.
    let count = chain.count(reaper, track);
    let mut fx = Vec::new();
    for i in 0..count {
        let loc = chain.location(i);
        let name = unsafe { reaper.track_fx_get_fx_name(track, loc, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
        let enabled = unsafe { reaper.track_fx_get_enabled(track, loc) };
        let offline = unsafe { reaper.track_fx_get_offline(track, loc) };
        fx.push(json!({ "index": i, "name": name, "enabled": enabled, "offline": offline }));
    }
    Ok(json!({ "track_index": track_index, "chain": chain.label(), "fx": fx }))
}

fn get_fx_params(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    limit: usize,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let track = chain.track(reaper, track_index)?;
    let loc = chain.location(fx_index);
    let num = unsafe { reaper.track_fx_get_num_params(track, loc) };
    let cap = (limit as u32).min(num);
    let mut params = Vec::new();
    for p in 0..cap {
        let name = unsafe { reaper.track_fx_get_param_name(track, loc, p, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
        let value = unsafe { reaper.track_fx_get_formatted_param_value(track, loc, p, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
        let norm = unsafe { reaper.track_fx_get_param_normalized(track, loc, p) }.get();
        params.push(json!({ "index": p, "name": name, "value": value, "normalized": norm }));
    }
    Ok(json!({
        "track_index": track_index,
        "fx_index": fx_index,
        "chain": chain.label(),
        "param_count": num,
        "truncated": num > cap,
        "params": params,
    }))
}

fn get_selected_items(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let item_index_of = media_item_index_map(reaper);
    let track_index_of = track_index_map(reaper);
    let low = reaper.low();

    let mut items = Vec::new();
    for s in 0..reaper.count_selected_media_items(project) {
        if let Some(item) = reaper.get_selected_media_item(project, s) {
            // SAFETY: item just obtained from REAPER, used on the main thread.
            let position =
                unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Position) };
            let length =
                unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Length) };
            let take = unsafe { reaper.get_active_take(item) };
            let take_name = take
                .map(|t| {
                    reaper.get_take_name(t, |r| {
                        r.map(|s| reaper_string(s.as_c_str().to_bytes()))
                            .unwrap_or_default()
                    })
                })
                .unwrap_or_default();
            let take_fx_count = take
                .map(|t| unsafe { low.TakeFX_GetCount(t.as_ptr()) })
                .unwrap_or(0);
            let track_index = unsafe { reaper.get_media_item_track(item) }
                .and_then(|tr| track_index_of.get(&tr).copied());
            items.push(json!({
                "item_index": item_index_of.get(&item).copied(),
                "track_index": track_index,
                "position": position,
                "length": length,
                "take_name": take_name,
                "take_fx_count": take_fx_count,
            }));
        }
    }
    json!({ "selected_items": items })
}

fn get_take_fx(reaper: &Reaper<MainThreadScope>, item_index: u32) -> Result<Value, String> {
    let item = reaper
        .get_media_item(ProjectContext::CurrentProject, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    let count = unsafe { low.TakeFX_GetCount(t) };
    let mut fx = Vec::new();
    for i in 0..count {
        let name = read_string(NAME_BUF as usize, |b, s| unsafe {
            low.TakeFX_GetFXName(t, i, b, s)
        })
        .unwrap_or_default();
        let enabled = unsafe { low.TakeFX_GetEnabled(t, i) };
        let offline = unsafe { low.TakeFX_GetOffline(t, i) };
        fx.push(json!({ "index": i, "name": name, "enabled": enabled, "offline": offline }));
    }
    Ok(json!({ "item_index": item_index, "take_fx": fx }))
}

fn get_take_fx_params(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    fx_index: u32,
    limit: usize,
) -> Result<Value, String> {
    let item = reaper
        .get_media_item(ProjectContext::CurrentProject, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    let fx = fx_index as c_int;
    let num = unsafe { low.TakeFX_GetNumParams(t, fx) }.max(0);
    let cap = (limit as c_int).min(num);
    let mut params = Vec::new();
    for p in 0..cap {
        let name = read_string(NAME_BUF as usize, |b, s| unsafe {
            low.TakeFX_GetParamName(t, fx, p, b, s)
        })
        .unwrap_or_default();
        let value = read_string(NAME_BUF as usize, |b, s| unsafe {
            low.TakeFX_GetFormattedParamValue(t, fx, p, b, s)
        })
        .unwrap_or_default();
        let norm = unsafe { low.TakeFX_GetParamNormalized(t, fx, p) };
        params.push(json!({ "index": p, "name": name, "value": value, "normalized": norm }));
    }
    Ok(json!({
        "item_index": item_index,
        "fx_index": fx_index,
        "param_count": num,
        "truncated": num > cap,
        "params": params,
    }))
}

fn list_installed_fx(
    reaper: &Reaper<MainThreadScope>,
    filter: Option<&str>,
    limit: usize,
) -> Value {
    let low = reaper.low();
    let filter_lc = filter.map(|f| f.to_lowercase());
    let mut matched = Vec::new();
    let mut total_matched = 0usize;
    let mut i: c_int = 0;
    loop {
        let mut name_ptr: *const c_char = std::ptr::null();
        let mut ident_ptr: *const c_char = std::ptr::null();
        let ok = unsafe { low.EnumInstalledFX(i, &mut name_ptr, &mut ident_ptr) };
        if !ok {
            break;
        }
        i += 1;
        let name = unsafe { cstr_to_string(name_ptr) };
        if let Some(f) = &filter_lc {
            if !name.to_lowercase().contains(f) {
                continue;
            }
        }
        total_matched += 1;
        if matched.len() < limit {
            let ident = unsafe { cstr_to_string(ident_ptr) };
            matched.push(json!({ "name": name, "ident": ident }));
        }
        if i > 100_000 {
            break; // safety bound
        }
    }
    json!({
        "total_matched": total_matched,
        "returned": matched.len(),
        "truncated": total_matched > matched.len(),
        "fx": matched,
    })
}

fn get_focused_fx(reaper: &Reaper<MainThreadScope>) -> Value {
    match reaper.get_touched_or_focused_fx_currently_focused_fx() {
        None => json!({ "focused_fx": Value::Null }),
        Some(res) => {
            let still = res.is_still_focused;
            match res.fx {
                FxLocation::TrackFx {
                    track_location,
                    fx_location,
                } => json!({
                    "is_still_focused": still,
                    "kind": "track_fx",
                    "track": track_location_json(track_location),
                    "fx": fx_location_json(fx_location),
                }),
                FxLocation::TakeFx {
                    track_index,
                    item_index,
                    take_index,
                    fx_index,
                } => json!({
                    "is_still_focused": still,
                    "kind": "take_fx",
                    "track_index": track_index,
                    "item_index": item_index,
                    "take_index": take_index,
                    "fx_index": fx_index,
                }),
            }
        }
    }
}

// ---- mutating tools (Undo-wrapped) ------------------------------------------

fn add_fx(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_name: &str,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    reaper.undo_begin_block_2(project);
    // SAFETY: track valid, main thread.
    let result = unsafe {
        reaper.track_fx_add_by_name_add(
            track,
            fx_name,
            chain.chain_type(),
            AddFxBehavior::AlwaysAdd,
        )
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add FX \"{fx_name}\" to track {track_index} ({} chain)", chain.label()),
        UndoScope::All,
    );
    match result {
        Ok(fx_index) => Ok(json!({
            "added": true, "track_index": track_index, "chain": chain.label(),
            "fx_index": fx_index, "name": fx_name
        })),
        Err(_) => Err(format!("could not add FX \"{fx_name}\" (name not found?)")),
    }
}

fn set_fx_param(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    param_index: u32,
    value: f64,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    let loc = chain.location(fx_index);
    let v = value.clamp(0.0, 1.0);
    reaper.undo_begin_block_2(project);
    let result = unsafe {
        reaper.track_fx_set_param_normalized(
            track,
            loc,
            param_index,
            ReaperNormalizedFxParamValue::new(v),
        )
    };
    let display =
        unsafe { reaper.track_fx_get_formatted_param_value(track, loc, param_index, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} FX {fx_index} param {param_index} to {v:.4}"),
        UndoScope::All,
    );
    result
        .map(|_| {
            json!({
                "set": true, "track_index": track_index, "chain": chain.label(),
                "fx_index": fx_index, "param_index": param_index, "normalized": v,
                "display_value": display
            })
        })
        .map_err(|_| "failed to set parameter".to_string())
}

/// Nudge a track-FX parameter by a relative number of normalized steps (Phase 7:
/// lets the model act on a knob it can only SEE, without display-value math —
/// REAPER exposes no display→normalized conversion). Reads the current value,
/// applies `direction * step_count * step_size`, clamps to 0..1, and reports the
/// new display value. Undo-wrapped, mirroring `set_fx_param`.
#[allow(clippy::too_many_arguments)] // a flat, self-explanatory nudge parameter list
fn set_fx_param_by_steps(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    param_index: u32,
    direction: &str,
    step_count: u32,
    step_size: f64,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let sign = match direction.trim().to_lowercase().as_str() {
        "up" | "increase" | "+" => 1.0,
        "down" | "decrease" | "-" => -1.0,
        other => return Err(format!("direction must be 'up' or 'down' (got '{other}')")),
    };
    // Sanitize step size: finite, positive, at most a full sweep.
    let step = if step_size.is_finite() && step_size > 0.0 {
        step_size.min(1.0)
    } else {
        0.01
    };
    let steps = step_count.max(1);
    let delta = sign * (steps as f64) * step;

    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    let loc = chain.location(fx_index);

    // SAFETY: track valid, main thread.
    let current = unsafe { reaper.track_fx_get_param_normalized(track, loc, param_index) }.get();
    let target = (current + delta).clamp(0.0, 1.0);

    reaper.undo_begin_block_2(project);
    let result = unsafe {
        reaper.track_fx_set_param_normalized(
            track,
            loc,
            param_index,
            ReaperNormalizedFxParamValue::new(target),
        )
    };
    let display =
        unsafe { reaper.track_fx_get_formatted_param_value(track, loc, param_index, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
    reaper.undo_end_block_2(
        project,
        format!("AI: nudge track {track_index} FX {fx_index} param {param_index} {direction} by {steps}x{step:.4}"),
        UndoScope::All,
    );
    result
        .map(|_| {
            json!({
                "set": true, "track_index": track_index, "chain": chain.label(),
                "fx_index": fx_index, "param_index": param_index, "direction": direction,
                "steps": steps, "step_size": step, "previous_normalized": current,
                "normalized": target, "display_value": display
            })
        })
        .map_err(|_| "failed to set parameter".to_string())
}

fn set_fx_enabled(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    enabled: bool,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    let loc = chain.location(fx_index);
    reaper.undo_begin_block_2(project);
    unsafe { reaper.track_fx_set_enabled(track, loc, enabled) };
    reaper.undo_end_block_2(
        project,
        format!(
            "AI: {} track {track_index} FX {fx_index}",
            if enabled { "enable" } else { "bypass" }
        ),
        UndoScope::All,
    );
    Ok(json!({
        "track_index": track_index, "chain": chain.label(),
        "fx_index": fx_index, "enabled": enabled
    }))
}

// ---- undo / history ---------------------------------------------------------

fn undo_label(reaper: &Reaper<MainThreadScope>, project: ProjectContext) -> Option<String> {
    reaper.undo_can_undo_2(project, |s: &ReaperStr| {
        reaper_string(s.as_c_str().to_bytes())
    })
}

fn redo_label(reaper: &Reaper<MainThreadScope>, project: ProjectContext) -> Option<String> {
    reaper.undo_can_redo_2(project, |s: &ReaperStr| {
        reaper_string(s.as_c_str().to_bytes())
    })
}

fn undo(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let action = undo_label(reaper, project);
    let ok = reaper.undo_do_undo_2(project);
    json!({ "undone": ok, "action": action })
}

fn redo(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let action = redo_label(reaper, project);
    let ok = reaper.undo_do_redo_2(project);
    json!({ "redone": ok, "action": action })
}

fn get_undo_history(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    json!({
        "next_undo": undo_label(reaper, project),
        "next_redo": redo_label(reaper, project),
        "recent_actions": crate::reaper::history::snapshot(),
        "note": "recent_actions is a best-effort log of undo-point labels observed over time \
                 (most recent last); REAPER's API does not expose the full undo stack.",
    })
}

// ---- MIDI (Phase 4) ---------------------------------------------------------

fn read_take_notes(reaper: &Reaper<MainThreadScope>, take: MediaItemTake) -> Vec<Value> {
    let t = take.as_ptr();
    let low = reaper.low();
    let mut notecnt: c_int = 0;
    let mut cc: c_int = 0;
    let mut syx: c_int = 0;
    unsafe { low.MIDI_CountEvts(t, &mut notecnt, &mut cc, &mut syx) };
    let mut notes = Vec::new();
    for i in 0..notecnt {
        let mut selected = false;
        let mut muted = false;
        let mut sppq = 0.0f64;
        let mut eppq = 0.0f64;
        let mut chan: c_int = 0;
        let mut pitch: c_int = 0;
        let mut vel: c_int = 0;
        let ok = unsafe {
            low.MIDI_GetNote(
                t,
                i,
                &mut selected,
                &mut muted,
                &mut sppq,
                &mut eppq,
                &mut chan,
                &mut pitch,
                &mut vel,
            )
        };
        if !ok {
            continue;
        }
        let start_time = unsafe { low.MIDI_GetProjTimeFromPPQPos(t, sppq) };
        let end_time = unsafe { low.MIDI_GetProjTimeFromPPQPos(t, eppq) };
        notes.push(json!({
            "pitch": pitch, "note": note_name(pitch), "velocity": vel, "channel": chan,
            "start_time": start_time, "end_time": end_time,
            "start_ppq": sppq, "end_ppq": eppq,
            "selected": selected, "muted": muted,
        }));
    }
    notes
}

fn get_take_midi(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    include_neighbors: bool,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = reaper
        .get_media_item(project, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let notes = read_take_notes(reaper, take);
    let mut result = json!({ "item_index": item_index, "note_count": notes.len(), "notes": notes });

    if include_neighbors {
        let track_index_of = track_index_map(reaper);
        let this_idx = unsafe { reaper.get_media_item_track(item) }
            .and_then(|tr| track_index_of.get(&tr).copied());
        let start = unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Position) };
        let end =
            start + unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Length) };
        let mut neighbors = Vec::new();
        if let Some(ti) = this_idx {
            let want: Vec<u32> = [ti.checked_sub(1), Some(ti + 1)]
                .into_iter()
                .flatten()
                .collect();
            for i in 0..reaper.count_media_items(project) {
                let Some(other) = reaper.get_media_item(project, i) else {
                    continue;
                };
                if other == item {
                    continue;
                }
                let oidx = unsafe { reaper.get_media_item_track(other) }
                    .and_then(|tr| track_index_of.get(&tr).copied());
                let Some(oi) = oidx.filter(|oi| want.contains(oi)) else {
                    continue;
                };
                let os =
                    unsafe { reaper.get_media_item_info_value(other, ItemAttributeKey::Position) };
                let oe = os
                    + unsafe { reaper.get_media_item_info_value(other, ItemAttributeKey::Length) };
                if oe <= start || os >= end {
                    continue; // no time overlap
                }
                if let Some(otake) = unsafe { reaper.get_active_take(other) } {
                    neighbors.push(json!({
                        "item_index": i, "track_index": oi, "notes": read_take_notes(reaper, otake)
                    }));
                }
            }
        }
        result["neighbor_items"] = json!(neighbors);
    }
    Ok(result)
}

fn insert_midi_notes(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    notes: &Value,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = reaper
        .get_media_item(project, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let arr = notes.as_array().ok_or("'notes' must be an array")?;
    let t = take.as_ptr();
    let low = reaper.low();

    let item_start = unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Position) };
    let item_start_qn = reaper
        .time_map_2_time_to_qn(
            project,
            PositionInSeconds::new(item_start).unwrap_or_default(),
        )
        .get();
    let no_sort = true;
    let mut inserted = 0;

    reaper.undo_begin_block_2(project);
    for n in arr {
        let (pitch, start_qn, length_qn) = match (
            n.get("pitch").and_then(|v| v.as_i64()),
            n.get("start_qn").and_then(|v| v.as_f64()),
            n.get("length_qn").and_then(|v| v.as_f64()),
        ) {
            (Some(p), Some(s), Some(l)) => (p as c_int, s, l),
            _ => continue, // skip malformed note
        };
        let vel = n.get("velocity").and_then(|v| v.as_i64()).unwrap_or(96) as c_int;
        let chan = n.get("channel").and_then(|v| v.as_i64()).unwrap_or(0) as c_int;
        let sppq = unsafe { low.MIDI_GetPPQPosFromProjQN(t, item_start_qn + start_qn) };
        let eppq = unsafe { low.MIDI_GetPPQPosFromProjQN(t, item_start_qn + start_qn + length_qn) };
        let ok =
            unsafe { low.MIDI_InsertNote(t, false, false, sppq, eppq, chan, pitch, vel, &no_sort) };
        if ok {
            inserted += 1;
        }
    }
    unsafe { low.MIDI_Sort(t) };
    reaper.undo_end_block_2(
        project,
        format!("AI: insert {inserted} MIDI note(s) into item {item_index}"),
        UndoScope::All,
    );
    Ok(json!({ "inserted": inserted, "item_index": item_index }))
}

fn create_midi_item(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    start_qn: f64,
    length_qn: f64,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let qn_in = true;
    reaper.undo_begin_block_2(project);
    let item_ptr = unsafe {
        low.CreateNewMIDIItemInProj(track.as_ptr(), start_qn, start_qn + length_qn, &qn_in)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: create MIDI item on track {track_index}"),
        UndoScope::All,
    );
    let item_index =
        MediaItem::new(item_ptr).and_then(|it| media_item_index_map(reaper).get(&it).copied());
    Ok(json!({
        "created": !item_ptr.is_null(),
        "track_index": track_index,
        "item_index": item_index,
    }))
}

fn create_track(
    reaper: &Reaper<MainThreadScope>,
    index: Option<u32>,
    name: Option<&str>,
) -> Result<Value, String> {
    if let Some(n) = name {
        if n.as_bytes().contains(&0) {
            return Err("name contains a NUL byte".to_string());
        }
    }
    let project = ProjectContext::CurrentProject;
    let count = reaper.count_tracks(project);
    let idx = index.unwrap_or(count).min(count);
    reaper.undo_begin_block_2(project);
    reaper.insert_track_at_index(idx, TrackDefaultsBehavior::AddDefaultEnvAndFx);
    if let Some(n) = name {
        if let Some(track) = reaper.get_track(project, idx) {
            unsafe { reaper.get_set_media_track_info_set_name(track, n) };
        }
    }
    reaper.undo_end_block_2(
        project,
        format!("AI: create track at {idx}"),
        UndoScope::All,
    );
    reaper.low().TrackList_AdjustWindows(false);
    reaper.update_arrange();
    Ok(json!({ "created": true, "track_index": idx, "name": name }))
}

fn delete_midi_notes(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    pitch_min: Option<i64>,
    pitch_max: Option<i64>,
    start_time: Option<f64>,
    end_time: Option<f64>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    let mut note_count: c_int = 0;
    let mut cc: c_int = 0;
    let mut syx: c_int = 0;
    unsafe { low.MIDI_CountEvts(t, &mut note_count, &mut cc, &mut syx) };
    let filter_time = start_time.is_some() || end_time.is_some();
    let mut to_delete = Vec::new();
    for i in 0..note_count {
        let mut selected = false;
        let mut muted = false;
        let mut sppq = 0.0f64;
        let mut eppq = 0.0f64;
        let mut chan: c_int = 0;
        let mut pitch: c_int = 0;
        let mut vel: c_int = 0;
        let ok = unsafe {
            low.MIDI_GetNote(
                t,
                i,
                &mut selected,
                &mut muted,
                &mut sppq,
                &mut eppq,
                &mut chan,
                &mut pitch,
                &mut vel,
            )
        };
        if !ok {
            continue;
        }
        if pitch_min.is_some_and(|p| (pitch as i64) < p) {
            continue;
        }
        if pitch_max.is_some_and(|p| (pitch as i64) > p) {
            continue;
        }
        if filter_time {
            let start = unsafe { low.MIDI_GetProjTimeFromPPQPos(t, sppq) };
            if start_time.is_some_and(|s| start < s) || end_time.is_some_and(|e| start > e) {
                continue;
            }
        }
        to_delete.push(i);
    }
    reaper.undo_begin_block_2(project);
    // Delete from the highest index down so earlier indices stay valid.
    for &i in to_delete.iter().rev() {
        unsafe { low.MIDI_DeleteNote(t, i) };
    }
    unsafe { low.MIDI_Sort(t) };
    reaper.undo_end_block_2(
        project,
        format!(
            "AI: delete {} MIDI note(s) from item {item_index}",
            to_delete.len()
        ),
        UndoScope::All,
    );
    Ok(json!({ "deleted": to_delete.len(), "item_index": item_index }))
}

fn note_name(pitch: c_int) -> String {
    if !(0..=127).contains(&pitch) {
        return pitch.to_string();
    }
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[(pitch % 12) as usize], pitch / 12 - 1)
}

// ---- track routing / sends (Phase 4) ----------------------------------------

fn get_track_sends(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let track_index_of = track_index_map(reaper);

    let mut sends = Vec::new();
    for i in 0..unsafe { reaper.get_track_num_sends(track, TrackSendCategory::Send) } {
        let name = unsafe { reaper.get_track_send_name(track, i, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
        let dest =
            unsafe { reaper.get_track_send_info_desttrack(track, TrackSendDirection::Send, i) }
                .ok()
                .and_then(|d| track_index_of.get(&d).copied());
        sends.push(json!({
            "index": i,
            "name": name,
            "dest_track_index": dest,
            "volume": unsafe { reaper.get_track_send_info_value(track, TrackSendCategory::Send, i, TrackSendAttributeKey::Vol) },
            "pan": unsafe { reaper.get_track_send_info_value(track, TrackSendCategory::Send, i, TrackSendAttributeKey::Pan) },
            "muted": unsafe { reaper.get_track_send_info_value(track, TrackSendCategory::Send, i, TrackSendAttributeKey::Mute) } != 0.0,
        }));
    }

    let mut receives = Vec::new();
    for i in 0..unsafe { reaper.get_track_num_sends(track, TrackSendCategory::Receive) } {
        let src =
            unsafe { reaper.get_track_send_info_desttrack(track, TrackSendDirection::Receive, i) }
                .ok()
                .and_then(|d| track_index_of.get(&d).copied());
        receives.push(json!({
            "index": i,
            "src_track_index": src,
            "volume": unsafe { reaper.get_track_send_info_value(track, TrackSendCategory::Receive, i, TrackSendAttributeKey::Vol) },
        }));
    }
    Ok(json!({ "track_index": track_index, "sends": sends, "receives": receives }))
}

fn add_send(
    reaper: &Reaper<MainThreadScope>,
    src_track_index: u32,
    dest_track_index: u32,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let src = reaper
        .get_track(project, src_track_index)
        .ok_or_else(|| format!("no track at index {src_track_index}"))?;
    let dest = reaper
        .get_track(project, dest_track_index)
        .ok_or_else(|| format!("no track at index {dest_track_index}"))?;
    reaper.undo_begin_block_2(project);
    let res = unsafe { reaper.create_track_send(src, SendTarget::OtherTrack(dest)) };
    reaper.undo_end_block_2(
        project,
        format!("AI: add send from track {src_track_index} to track {dest_track_index}"),
        UndoScope::All,
    );
    res.map(|idx| {
        json!({
            "added": true, "src_track_index": src_track_index,
            "dest_track_index": dest_track_index, "send_index": idx
        })
    })
    .map_err(|_| "failed to create send".to_string())
}

fn set_send_param(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    send_index: u32,
    param: &str,
    value: f64,
) -> Result<Value, String> {
    let key = match param {
        "volume" | "vol" => TrackSendAttributeKey::Vol,
        "pan" => TrackSendAttributeKey::Pan,
        "mute" => TrackSendAttributeKey::Mute,
        other => {
            return Err(format!(
                "unknown send param '{other}' (use volume, pan, or mute)"
            ))
        }
    };
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    reaper.undo_begin_block_2(project);
    let res = unsafe {
        reaper.set_track_send_info_value(track, TrackSendCategory::Send, send_index, key, value)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: set send {send_index} {param} on track {track_index}"),
        UndoScope::All,
    );
    res.map(|_| {
        json!({
            "set": true, "track_index": track_index, "send_index": send_index,
            "param": param, "value": value
        })
    })
    .map_err(|_| "failed to set send parameter".to_string())
}

fn remove_send(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    send_index: u32,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    reaper.undo_begin_block_2(project);
    let res = unsafe { reaper.remove_track_send(track, TrackSendCategory::Send, send_index) };
    reaper.undo_end_block_2(
        project,
        format!("AI: remove send {send_index} from track {track_index}"),
        UndoScope::All,
    );
    res.map(|_| json!({ "removed": true, "track_index": track_index, "send_index": send_index }))
        .map_err(|_| "failed to remove send".to_string())
}

// ---- automation / envelopes (Phase 5) ---------------------------------------

fn get_track_envelopes(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let tp = track.as_ptr();
    let count = unsafe { low.CountTrackEnvelopes(tp) };
    let mut envs = Vec::new();
    for i in 0..count {
        let env = unsafe { low.GetTrackEnvelope(tp, i) };
        if env.is_null() {
            continue;
        }
        let name = read_string(NAME_BUF as usize, |b, s| unsafe {
            low.GetEnvelopeName(env, b, s)
        })
        .unwrap_or_default();
        envs.push(json!({
            "index": i,
            "name": name,
            "point_count": unsafe { low.CountEnvelopePoints(env) },
            "automation_item_count": unsafe { low.CountAutomationItems(env) },
        }));
    }
    Ok(json!({ "track_index": track_index, "envelopes": envs }))
}

fn get_envelope_points(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
    limit: usize,
) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    let count = unsafe { low.CountEnvelopePoints(env) };
    let cap = (limit as c_int).min(count);
    let mut points = Vec::new();
    for i in 0..cap {
        let mut time = 0.0f64;
        let mut value = 0.0f64;
        let mut shape: c_int = 0;
        let mut tension = 0.0f64;
        let mut selected = false;
        let ok = unsafe {
            low.GetEnvelopePoint(
                env,
                i,
                &mut time,
                &mut value,
                &mut shape,
                &mut tension,
                &mut selected,
            )
        };
        if !ok {
            continue;
        }
        points.push(json!({
            "index": i, "time": time, "value": value,
            "shape": shape, "tension": tension, "selected": selected
        }));
    }
    Ok(json!({
        "track_index": track_index,
        "envelope_index": envelope_index,
        "point_count": count,
        "truncated": count > cap,
        "points": points,
    }))
}

fn get_automation_items(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    let count = unsafe { low.CountAutomationItems(env) };
    let get = |i: c_int, key: &CStr| unsafe {
        low.GetSetAutomationItemInfo(env, i, key.as_ptr(), 0.0, false)
    };
    let mut items = Vec::new();
    for i in 0..count {
        items.push(json!({
            "index": i,
            "position": get(i, c"D_POSITION"),
            "length": get(i, c"D_LENGTH"),
            "playrate": get(i, c"D_PLAYRATE"),
            "pool_id": get(i, c"D_POOL_ID"),
        }));
    }
    Ok(json!({
        "track_index": track_index,
        "envelope_index": envelope_index,
        "automation_items": items,
    }))
}

fn insert_envelope_point(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
    time: f64,
    value: f64,
    shape: c_int,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    let mut no_sort = false;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.InsertEnvelopePoint(env, time, value, shape, 0.0, false, &mut no_sort) };
    unsafe { low.Envelope_SortPoints(env) };
    reaper.undo_end_block_2(
        project,
        format!("AI: insert envelope point on track {track_index} envelope {envelope_index}"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({
            "inserted": true, "track_index": track_index,
            "envelope_index": envelope_index, "time": time, "value": value, "shape": shape
        }))
    } else {
        Err("failed to insert envelope point".to_string())
    }
}

// ---- notes & per-project memory ---------------------------------------------

const NOTES_BUF: usize = 256 * 1024;
const MEMORY_EXT: &CStr = c"REAPER_AI_Assistant";
const TRACK_NOTES_KEY: &str = "raai_notes";

fn read_project_notes(low: &reaper_low::Reaper) -> String {
    read_string(NOTES_BUF, |b, s| {
        unsafe { low.GetSetProjectNotes(std::ptr::null_mut(), false, b, s) };
        true
    })
    .unwrap_or_default()
}

fn get_project_notes(reaper: &Reaper<MainThreadScope>) -> Value {
    json!({ "notes": read_project_notes(reaper.low()) })
}

fn set_project_notes(reaper: &Reaper<MainThreadScope>, text: &str, append: bool) -> Value {
    let low = reaper.low();
    let new_text = if append {
        let current = read_project_notes(low);
        if current.trim().is_empty() {
            text.to_string()
        } else {
            format!("{current}\n{text}")
        }
    } else {
        text.to_string()
    };
    let mut bytes = new_text.into_bytes();
    bytes.push(0); // NUL terminator
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    unsafe {
        low.GetSetProjectNotes(
            std::ptr::null_mut(),
            true,
            bytes.as_mut_ptr() as *mut c_char,
            bytes.len() as c_int,
        );
    }
    reaper.undo_end_block_2(
        project,
        "AI: update project notes".to_string(),
        UndoScope::All,
    );
    json!({ "saved": true, "appended": append })
}

fn get_track_notes(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let notes = unsafe {
        reaper.get_set_media_track_info_get_ext(track, TRACK_NOTES_KEY, |s: &ReaperStr| {
            reaper_string(s.as_c_str().to_bytes())
        })
    }
    .unwrap_or_default();
    Ok(json!({ "track_index": track_index, "notes": notes }))
}

fn set_track_notes(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    text: &str,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    reaper.undo_begin_block_2(project);
    unsafe {
        reaper.get_set_media_track_info_set_ext(track, TRACK_NOTES_KEY, text);
    }
    reaper.undo_end_block_2(
        project,
        format!("AI: update notes for track {track_index}"),
        UndoScope::All,
    );
    Ok(json!({ "saved": true, "track_index": track_index }))
}

fn get_project_memory(
    reaper: &Reaper<MainThreadScope>,
    key: Option<&str>,
) -> Result<Value, String> {
    let low = reaper.low();
    match key {
        Some(k) => {
            let key_c = CString::new(k).map_err(|_| "invalid key".to_string())?;
            let value = read_string(NOTES_BUF, |b, s| unsafe {
                low.GetProjExtState(
                    std::ptr::null_mut(),
                    MEMORY_EXT.as_ptr(),
                    key_c.as_ptr(),
                    b,
                    s,
                ) > 0
            })
            .unwrap_or_default();
            Ok(json!({ "key": k, "value": value }))
        }
        None => {
            let mut entries = Vec::new();
            let mut i: c_int = 0;
            loop {
                let mut keyb = vec![0u8; 1024];
                let mut valb = vec![0u8; NOTES_BUF];
                let ok = unsafe {
                    low.EnumProjExtState(
                        std::ptr::null_mut(),
                        MEMORY_EXT.as_ptr(),
                        i,
                        keyb.as_mut_ptr() as *mut c_char,
                        keyb.len() as c_int,
                        valb.as_mut_ptr() as *mut c_char,
                        valb.len() as c_int,
                    )
                };
                if !ok {
                    break;
                }
                entries.push(json!({ "key": buf_to_string(&keyb), "value": buf_to_string(&valb) }));
                i += 1;
                if i > 10_000 {
                    break;
                }
            }
            Ok(json!({ "memory": entries }))
        }
    }
}

fn set_project_memory(
    reaper: &Reaper<MainThreadScope>,
    key: &str,
    value: &str,
) -> Result<Value, String> {
    let low = reaper.low();
    let key_c = CString::new(key).map_err(|_| "invalid key (contains a NUL byte)".to_string())?;
    let val_c = CString::new(value).map_err(|_| "value contains a NUL byte".to_string())?;
    unsafe {
        low.SetProjExtState(
            std::ptr::null_mut(),
            MEMORY_EXT.as_ptr(),
            key_c.as_ptr(),
            val_c.as_ptr(),
        );
    }
    Ok(json!({ "saved": true, "key": key }))
}

fn delete_project_memory(reaper: &Reaper<MainThreadScope>, key: &str) -> Result<Value, String> {
    let low = reaper.low();
    let key_c = CString::new(key).map_err(|_| "invalid key".to_string())?;
    // An empty value removes the key.
    unsafe {
        low.SetProjExtState(
            std::ptr::null_mut(),
            MEMORY_EXT.as_ptr(),
            key_c.as_ptr(),
            c"".as_ptr(),
        );
    }
    Ok(json!({ "deleted": true, "key": key }))
}

fn buf_to_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    reaper_string(&buf[..end])
}

// ---- markers, regions, tempo, stretch markers, render settings --------------

/// The null project pointer denotes the current project for the low-level API,
/// matching how the notes/memory tools address the active project.
const CUR_PROJ: *mut reaper_low::raw::ReaProject = std::ptr::null_mut();

fn get_markers(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let mut out = Vec::new();
    let mut i: c_int = 0;
    loop {
        let mut is_rgn = false;
        let mut pos = 0.0f64;
        let mut rgn_end = 0.0f64;
        let mut name_ptr: *const c_char = std::ptr::null();
        let mut index_number: c_int = 0;
        let mut color: c_int = 0;
        let next = unsafe {
            low.EnumProjectMarkers3(
                CUR_PROJ,
                i,
                &mut is_rgn,
                &mut pos,
                &mut rgn_end,
                &mut name_ptr,
                &mut index_number,
                &mut color,
            )
        };
        if next == 0 {
            break; // no marker/region at index i
        }
        let name = unsafe { cstr_to_string(name_ptr) };
        let mut entry = json!({
            "kind": if is_rgn { "region" } else { "marker" },
            "index_number": index_number,
            "position": pos,
            "name": name,
            "color": color,
        });
        if is_rgn {
            entry["region_end"] = json!(rgn_end);
        }
        out.push(entry);
        i += 1;
        if i > 1_000_000 {
            break; // safety bound
        }
    }
    json!({ "markers": out })
}

fn add_marker(
    reaper: &Reaper<MainThreadScope>,
    position: f64,
    name: &str,
    want_index: c_int,
) -> Result<Value, String> {
    let name_c = CString::new(name).map_err(|_| "name contains a NUL byte".to_string())?;
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let id = unsafe {
        low.AddProjectMarker2(
            CUR_PROJ,
            false,
            position,
            0.0,
            name_c.as_ptr(),
            want_index,
            0,
        )
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add marker at {position:.3}s"),
        UndoScope::All,
    );
    Ok(json!({ "added": id >= 0, "index_number": id, "position": position }))
}

fn add_region(
    reaper: &Reaper<MainThreadScope>,
    start: f64,
    end: f64,
    name: &str,
    want_index: c_int,
) -> Result<Value, String> {
    let name_c = CString::new(name).map_err(|_| "name contains a NUL byte".to_string())?;
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let id = unsafe {
        low.AddProjectMarker2(CUR_PROJ, true, start, end, name_c.as_ptr(), want_index, 0)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add region {start:.3}s..{end:.3}s"),
        UndoScope::All,
    );
    Ok(json!({ "added": id >= 0, "index_number": id, "start": start, "end": end }))
}

fn delete_marker(
    reaper: &Reaper<MainThreadScope>,
    index_number: c_int,
    is_region: bool,
) -> Result<Value, String> {
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.DeleteProjectMarker(CUR_PROJ, index_number, is_region) };
    reaper.undo_end_block_2(
        project,
        format!(
            "AI: delete {} {index_number}",
            if is_region { "region" } else { "marker" }
        ),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "deleted": true, "index_number": index_number, "is_region": is_region }))
    } else {
        Err(format!(
            "no {} with index number {index_number}",
            if is_region { "region" } else { "marker" }
        ))
    }
}

// ---- transport / timeline / global settings --------------------------------

/// Ruler time-unit names ↔ REAPER "View: Time unit for ruler: …" action ids.
const RULER_UNITS: &[(&str, c_int)] = &[
    ("minutes_seconds", 40365),
    ("measures_beats_minutes", 40366),
    ("measures_beats", 40367),
    ("seconds", 40368),
    ("samples", 40369),
    ("hmsf", 40370),
];

fn current_ruler_unit(low: &reaper_low::Reaper) -> &'static str {
    for (name, id) in RULER_UNITS {
        if low.GetToggleCommandState(*id) == 1 {
            return name;
        }
    }
    "unknown"
}

fn get_transport(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let state = low.GetPlayState();
    let play_state = if state & 4 != 0 {
        "recording"
    } else if state & 2 != 0 {
        "paused"
    } else if state & 1 != 0 {
        "playing"
    } else {
        "stopped"
    };
    let (mut ts_start, mut ts_end) = (0.0f64, 0.0f64);
    unsafe { low.GetSet_LoopTimeRange(false, false, &mut ts_start, &mut ts_end, false) };
    let time_selection = if ts_end > ts_start {
        json!({ "start": ts_start, "end": ts_end, "length": ts_end - ts_start })
    } else {
        Value::Null
    };
    json!({
        "play_state": play_state,
        "play_position": low.GetPlayPosition(),
        "edit_cursor": low.GetCursorPosition(),
        "playrate": unsafe { low.Master_GetPlayRate(CUR_PROJ) },
        "tempo": low.Master_GetTempo(),
        "repeat": low.GetSetRepeat(-1) != 0,
        "ruler_unit": current_ruler_unit(low),
        "time_selection": time_selection,
    })
}

/// Read the current time selection (arrange loop/edit range). Read-only.
fn get_time_selection(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let (mut s, mut e) = (0.0f64, 0.0f64);
    unsafe { low.GetSet_LoopTimeRange(false, false, &mut s, &mut e, false) };
    json!({
        "start": s,
        "end": e,
        "length": (e - s).max(0.0),
        "is_set": e > s,
    })
}

/// Set the time selection (and matching loop range) to [start, end] seconds.
/// Transient project state — like the edit cursor, it's not undo-wrapped.
fn set_time_selection(
    reaper: &Reaper<MainThreadScope>,
    start: f64,
    end: f64,
    seek: bool,
) -> Result<Value, String> {
    if !start.is_finite() || !end.is_finite() {
        return Err("start and end must be finite numbers of seconds".to_string());
    }
    // Order defensively and clamp to >= 0.
    let (mut s, mut e) = if start <= end { (start, end) } else { (end, start) };
    s = s.max(0.0);
    e = e.max(s);
    let low = reaper.low();
    // Set the time selection; also set the loop range to match (REAPER links them
    // by default), so the loop bar reflects the selection. `seek` (allowautoseek)
    // moves PLAYBACK to the start when playing.
    unsafe { low.GetSet_LoopTimeRange(true, false, &mut s, &mut e, seek) };
    unsafe { low.GetSet_LoopTimeRange(true, true, &mut s, &mut e, false) };
    if seek {
        // Move the EDIT cursor to the start too (and scroll it into view).
        low.SetEditCurPos(s, true, false);
    }
    reaper.update_arrange();
    Ok(json!({ "start": s, "end": e, "length": e - s, "seek": seek }))
}

/// Remove the time selection (and loop range). Transient — not undo-wrapped.
fn clear_time_selection(reaper: &Reaper<MainThreadScope>) -> Result<Value, String> {
    let low = reaper.low();
    // A zero-length range means "no time selection" (REAPER's remove-selection).
    let (mut z0, mut z1) = (0.0f64, 0.0f64);
    unsafe { low.GetSet_LoopTimeRange(true, false, &mut z0, &mut z1, false) };
    unsafe { low.GetSet_LoopTimeRange(true, true, &mut z0, &mut z1, false) };
    reaper.update_arrange();
    Ok(json!({ "cleared": true }))
}

// ---- REAPER actions + keyboard shortcuts (main action section) --------------

/// Resolve an action `command` — a numeric id ("40364") or a named command
/// ("_SWS_ABOUT") — to a numeric command id, verifying it names a real action.
fn resolve_command(
    low: &reaper_low::Reaper,
    section: *mut reaper_low::raw::KbdSectionInfo,
    command: &str,
) -> Result<i32, String> {
    let t = command.trim();
    let cmd = if let Ok(n) = t.parse::<i32>() {
        if n == 0 {
            return Err("invalid action command '0'".to_string());
        }
        n
    } else if t.is_empty() || t.as_bytes().contains(&0) {
        return Err("invalid action command".to_string());
    } else {
        let mut bytes = t.as_bytes().to_vec();
        bytes.push(0);
        let id = unsafe { low.NamedCommandLookup(bytes.as_ptr() as *const c_char) };
        if id == 0 {
            return Err(format!(
                "unknown action '{command}' (not a numeric id or a known named command)"
            ));
        }
        id
    };
    // A numeric id is otherwise unvalidated; confirm it names a real action
    // (kbd_getTextFromCmd yields an empty name for an unknown command id), so we
    // never run — or report success for — a nonexistent action.
    if action_name(low, section, cmd).is_empty() {
        return Err(format!("no action with command id {cmd}"));
    }
    Ok(cmd)
}

/// The key shortcuts bound to `cmd`, as human-readable strings (e.g. "Ctrl+S").
fn action_shortcuts(
    low: &reaper_low::Reaper,
    section: *mut reaper_low::raw::KbdSectionInfo,
    cmd: i32,
) -> Vec<String> {
    let n = unsafe { low.CountActionShortcuts(section, cmd) };
    (0..n)
        .filter_map(|i| read_string(512, |b, s| unsafe {
            low.GetActionShortcutDesc(section, cmd, i, b, s)
        }))
        .collect()
}

/// The display name of `cmd`, or an empty string.
fn action_name(
    low: &reaper_low::Reaper,
    section: *mut reaper_low::raw::KbdSectionInfo,
    cmd: i32,
) -> String {
    let p = unsafe { low.kbd_getTextFromCmd(cmd, section) };
    if p.is_null() {
        String::new()
    } else {
        unsafe { cstr_to_string(p) }
    }
}

/// -1 → not a toggle; 0 → off; 1 → on (JSON: null/"off"/"on").
fn toggle_json(state: i32) -> Value {
    match state {
        0 => json!("off"),
        1 => json!("on"),
        _ => Value::Null,
    }
}

/// Name + toggle state + shortcuts for one command id.
fn action_json(
    low: &reaper_low::Reaper,
    section: *mut reaper_low::raw::KbdSectionInfo,
    cmd: i32,
) -> Value {
    let toggle = unsafe { low.GetToggleCommandState2(section, cmd) };
    json!({
        "command_id": cmd,
        "name": action_name(low, section, cmd),
        "toggle_state": toggle_json(toggle),
        "shortcuts": action_shortcuts(low, section, cmd),
    })
}

fn get_action_info(reaper: &Reaper<MainThreadScope>, command: &str) -> Result<Value, String> {
    let low = reaper.low();
    let section = low.SectionFromUniqueID(0);
    if section.is_null() {
        return Err("could not resolve the main action section".to_string());
    }
    let cmd = resolve_command(low, section, command)?;
    Ok(action_json(low, section, cmd))
}

fn search_actions(
    reaper: &Reaper<MainThreadScope>,
    query: &str,
    limit: usize,
) -> Result<Value, String> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return Err("provide a non-empty search query".to_string());
    }
    let low = reaper.low();
    let section = low.SectionFromUniqueID(0);
    if section.is_null() {
        return Err("could not resolve the main action section".to_string());
    }
    let limit = limit.clamp(1, 200);
    let mut matches: Vec<Value> = Vec::new();
    let mut total = 0usize;
    let mut idx: c_int = 0;
    loop {
        let mut name_ptr: *const c_char = std::ptr::null();
        let cmd = unsafe { low.kbd_enumerateActions(section, idx, &mut name_ptr) };
        if cmd == 0 || idx > 200_000 {
            break; // 0 = end of list; the cap is a runaway guard
        }
        idx += 1;
        // Use kbd_getTextFromCmd for the authoritative display name (the
        // enumerator's nameOut can be the internal command name for some actions).
        let name = action_name(low, section, cmd);
        if name.to_lowercase().contains(&q) {
            total += 1;
            if matches.len() < limit {
                matches.push(action_json(low, section, cmd));
            }
        }
    }
    Ok(json!({
        "query": query,
        "total_matched": total,
        "returned": matches.len(),
        "truncated": total > matches.len(),
        "actions": matches,
    }))
}

fn run_action(reaper: &Reaper<MainThreadScope>, command: &str) -> Result<Value, String> {
    let low = reaper.low();
    let section = low.SectionFromUniqueID(0);
    if section.is_null() {
        return Err("could not resolve the main action section".to_string());
    }
    let cmd = resolve_command(low, section, command)?;
    let name = action_name(low, section, cmd);
    // REAPER actions manage their own undo, so don't wrap this in an undo block.
    low.Main_OnCommand(cmd, 0);
    let toggle = unsafe { low.GetToggleCommandState2(section, cmd) };
    Ok(json!({
        "ran": true, "command_id": cmd, "name": name, "toggle_state": toggle_json(toggle),
    }))
}

fn delete_action_shortcut(
    reaper: &Reaper<MainThreadScope>,
    command: &str,
    shortcut_index: u32,
) -> Result<Value, String> {
    let low = reaper.low();
    let section = low.SectionFromUniqueID(0);
    if section.is_null() {
        return Err("could not resolve the main action section".to_string());
    }
    let cmd = resolve_command(low, section, command)?;
    let count = unsafe { low.CountActionShortcuts(section, cmd) };
    let i = i32::try_from(shortcut_index).map_err(|_| "shortcut_index out of range".to_string())?;
    if i >= count {
        return Err(format!(
            "action has {count} shortcut(s); no shortcut at index {shortcut_index}"
        ));
    }
    let ok = unsafe { low.DeleteActionShortcut(section, cmd, i) };
    if !ok {
        return Err("REAPER could not delete the shortcut".to_string());
    }
    Ok(json!({
        "deleted": true, "command_id": cmd, "shortcuts": action_shortcuts(low, section, cmd),
    }))
}

fn add_action_shortcut(reaper: &Reaper<MainThreadScope>, command: &str) -> Result<Value, String> {
    let low = reaper.low();
    let section = low.SectionFromUniqueID(0);
    if section.is_null() {
        return Err("could not resolve the main action section".to_string());
    }
    let cmd = resolve_command(low, section, command)?;
    // REAPER has no API to bind a specific key; the only way to ADD a shortcut is
    // its interactive dialog. Passing the current count opens the "add new" slot;
    // the user then presses the key combo to assign.
    let count = unsafe { low.CountActionShortcuts(section, cmd) };
    let hwnd = low.GetMainHwnd();
    unsafe { low.DoActionShortcutDialog(hwnd, section, cmd, count) };
    Ok(json!({
        "command_id": cmd,
        "shortcuts": action_shortcuts(low, section, cmd),
        "note": "Opened REAPER's key-assignment dialog for you to press the shortcut; the list reflects any key added.",
    }))
}

fn transport(reaper: &Reaper<MainThreadScope>, action: &str) -> Result<Value, String> {
    let cmd = match action {
        "play" => 1007,
        "stop" => 1016,
        "pause" => 1008,
        "record" => 1013,
        "toggle_play" => 40073,
        other => return Err(format!("unknown transport action '{other}'")),
    };
    reaper.low().Main_OnCommand(cmd, 0);
    Ok(json!({ "action": action, "state": get_transport(reaper) }))
}

fn set_edit_cursor(
    reaper: &Reaper<MainThreadScope>,
    position: f64,
    seek_playback: bool,
    move_view: bool,
) -> Result<Value, String> {
    if !position.is_finite() {
        return Err("position must be a finite number of seconds".to_string());
    }
    let low = reaper.low();
    low.SetEditCurPos(position.max(0.0), move_view, seek_playback);
    Ok(json!({ "edit_cursor": low.GetCursorPosition() }))
}

fn set_playrate(reaper: &Reaper<MainThreadScope>, rate: f64) -> Result<Value, String> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err("playrate must be a positive number".to_string());
    }
    let low = reaper.low();
    low.CSurf_OnPlayRateChange(rate.clamp(0.01, 16.0));
    Ok(json!({ "playrate": unsafe { low.Master_GetPlayRate(CUR_PROJ) } }))
}

fn set_ruler_unit(reaper: &Reaper<MainThreadScope>, unit: &str) -> Result<Value, String> {
    let id = RULER_UNITS
        .iter()
        .find(|(n, _)| *n == unit)
        .map(|(_, id)| *id)
        .ok_or_else(|| format!("unknown ruler unit '{unit}'"))?;
    reaper.low().Main_OnCommand(id, 0);
    Ok(json!({ "ruler_unit": current_ruler_unit(reaper.low()) }))
}

fn get_global_toggles(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let ripple = if low.GetToggleCommandState(40311) == 1 {
        "all_tracks"
    } else if low.GetToggleCommandState(40310) == 1 {
        "one_track"
    } else {
        "off"
    };
    json!({
        "metronome": low.GetToggleCommandState(40364) == 1,
        "repeat": low.GetSetRepeat(-1) != 0,
        "snap": low.GetToggleCommandState(1157) == 1,
        "ripple": ripple,
    })
}

fn set_global_toggle(
    reaper: &Reaper<MainThreadScope>,
    name: &str,
    on: bool,
) -> Result<Value, String> {
    let low = reaper.low();
    match name {
        "metronome" => {
            if (low.GetToggleCommandState(40364) == 1) != on {
                low.Main_OnCommand(40364, 0);
            }
        }
        "snap" => {
            if (low.GetToggleCommandState(1157) == 1) != on {
                low.Main_OnCommand(1157, 0);
            }
        }
        "repeat" => {
            low.GetSetRepeat(if on { 1 } else { 0 });
        }
        other => {
            return Err(format!("unknown toggle '{other}' (use metronome, repeat, or snap)"))
        }
    }
    Ok(get_global_toggles(reaper))
}

fn set_ripple_mode(reaper: &Reaper<MainThreadScope>, mode: &str) -> Result<Value, String> {
    let id = match mode {
        "off" => 40309,
        "one_track" => 40310,
        "all_tracks" => 40311,
        other => return Err(format!("unknown ripple mode '{other}'")),
    };
    reaper.low().Main_OnCommand(id, 0);
    Ok(get_global_toggles(reaper))
}

// ---- CRUD completeness ------------------------------------------------------

fn remove_fx(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { reaper.low().TrackFX_Delete(track.as_ptr(), chain.raw_index(fx_index)) };
    reaper.undo_end_block_2(
        project,
        format!("AI: remove FX {fx_index} from track {track_index} ({} chain)", chain.label()),
        UndoScope::All,
    );
    if ok {
        Ok(json!({
            "removed": true, "track_index": track_index,
            "chain": chain.label(), "fx_index": fx_index
        }))
    } else {
        Err(format!("no FX at index {fx_index} on track {track_index} ({} chain)", chain.label()))
    }
}

fn remove_take_fx(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    fx_index: u32,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let take = match take_index {
        Some(ti) => unsafe { low.GetMediaItemTake(item.as_ptr(), ti as c_int) },
        None => unsafe { low.GetActiveTake(item.as_ptr()) },
    };
    if take.is_null() {
        return Err("no such take".to_string());
    }
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.TakeFX_Delete(take, fx_index as c_int) };
    reaper.undo_end_block_2(
        project,
        format!("AI: remove take FX {fx_index} from item {item_index}"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "removed": true, "item_index": item_index, "fx_index": fx_index }))
    } else {
        Err(format!("no take FX at index {fx_index}"))
    }
}

// ---- FX presets -------------------------------------------------------------
// REAPER can LOAD presets (by name, index, or next/previous) and read the current
// one, but exposes NO API to SAVE a new preset — that stays a manual FX-window
// action. Track and take FX share the exact same call shape (only the handle
// differs), so these mirror each other closely.

/// Fill a preset-name buffer and return `(name, matches_preset)`. The REAPER
/// `*FX_GetPreset` bool means "current state exactly matches a stored preset";
/// the name buffer is filled regardless, so — unlike `read_string` — we keep it.
fn read_preset_name(f: impl FnOnce(*mut c_char, c_int) -> bool) -> (String, bool) {
    let mut buf = vec![0u8; NAME_BUF as usize];
    let matches = f(buf.as_mut_ptr() as *mut c_char, NAME_BUF as c_int);
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    (reaper_string(&buf[..end]), matches)
}

/// Validate that exactly one preset selector was given and resolve `navigate`
/// into a move delta (`Some(+1/-1)` for navigate, `None` for name/index).
fn preset_nav_move(
    name: Option<&str>,
    index: Option<i64>,
    navigate: Option<&str>,
) -> Result<Option<c_int>, String> {
    let selectors = name.is_some() as u8 + index.is_some() as u8 + navigate.is_some() as u8;
    if selectors != 1 {
        return Err("provide exactly one of: name, index, or navigate".into());
    }
    match navigate {
        Some("next") => Ok(Some(1)),
        Some("previous") | Some("prev") => Ok(Some(-1)),
        Some(other) => Err(format!(
            "navigate must be 'next' or 'previous' (got '{other}')"
        )),
        None => Ok(None),
    }
}

/// Confirm-prompt suffix naming a non-normal FX chain (empty for the normal chain),
/// so a blind user approving a change knows it targets input/monitoring FX.
fn chain_suffix(input: &Value) -> &'static str {
    match input.get("chain").and_then(|v| v.as_str()).unwrap_or("normal") {
        "input" | "rec" | "record" => " [input FX]",
        "monitor" | "mon" | "monitoring" => " [monitoring FX]",
        _ => "",
    }
}

/// Short human description of the chosen preset selector, for the confirm prompt.
fn preset_sel_desc(input: &Value) -> String {
    if let Some(n) = input.get("name").and_then(|v| v.as_str()) {
        format!("name '{n}'")
    } else if let Some(i) = input.get("index").and_then(|v| v.as_i64()) {
        format!("index {i}")
    } else if let Some(nav) = input.get("navigate").and_then(|v| v.as_str()) {
        format!("{nav} preset")
    } else {
        "?".into()
    }
}

fn get_fx_preset(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    chain: &str,
) -> Result<Value, String> {
    let chain = FxChain::parse(chain)?;
    let track = chain.track(reaper, track_index)?;
    let low = reaper.low();
    let fx = chain.raw_index(fx_index);
    let (name, matches) =
        read_preset_name(|b, s| unsafe { low.TrackFX_GetPreset(track.as_ptr(), fx, b, s) });
    let mut count: c_int = 0;
    let index = unsafe { low.TrackFX_GetPresetIndex(track.as_ptr(), fx, &mut count) };
    Ok(json!({
        "track_index": track_index,
        "chain": chain.label(),
        "fx_index": fx_index,
        "preset": name,
        "preset_index": index,
        "preset_count": count,
        "matches_preset": matches,
    }))
}

fn set_fx_preset(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    name: Option<&str>,
    index: Option<i64>,
    navigate: Option<&str>,
    chain: &str,
) -> Result<Value, String> {
    // Everything fallible happens BEFORE the undo block so we never leave it open.
    let chain = FxChain::parse(chain)?;
    let nav = preset_nav_move(name, index, navigate)?;
    let name_c = match name {
        Some(n) => Some(CString::new(n).map_err(|_| "preset name has a NUL byte".to_string())?),
        None => None,
    };
    let project = ProjectContext::CurrentProject;
    let track = chain.track(reaper, track_index)?;
    let low = reaper.low();
    let fx = chain.raw_index(fx_index);
    reaper.undo_begin_block_2(project);
    let ok = if let Some(c) = &name_c {
        unsafe { low.TrackFX_SetPreset(track.as_ptr(), fx, c.as_ptr()) }
    } else if let Some(i) = index {
        unsafe { low.TrackFX_SetPresetByIndex(track.as_ptr(), fx, i as c_int) }
    } else {
        unsafe { low.TrackFX_NavigatePresets(track.as_ptr(), fx, nav.unwrap_or(0)) }
    };
    let (current, matches) =
        read_preset_name(|b, s| unsafe { low.TrackFX_GetPreset(track.as_ptr(), fx, b, s) });
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} FX {fx_index} preset"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({
            "set": true, "track_index": track_index, "chain": chain.label(),
            "fx_index": fx_index, "preset": current, "matches_preset": matches,
        }))
    } else {
        Err("failed to set preset (unknown preset name/index, or none to navigate to)".into())
    }
}

fn get_take_fx_preset(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    fx_index: u32,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let take = match take_index {
        Some(ti) => unsafe { low.GetMediaItemTake(item.as_ptr(), ti as c_int) },
        None => unsafe { low.GetActiveTake(item.as_ptr()) },
    };
    if take.is_null() {
        return Err("no such take".to_string());
    }
    let fx = fx_index as c_int;
    let (name, matches) = read_preset_name(|b, s| unsafe { low.TakeFX_GetPreset(take, fx, b, s) });
    let mut count: c_int = 0;
    let index = unsafe { low.TakeFX_GetPresetIndex(take, fx, &mut count) };
    Ok(json!({
        "item_index": item_index,
        "fx_index": fx_index,
        "preset": name,
        "preset_index": index,
        "preset_count": count,
        "matches_preset": matches,
    }))
}

fn set_take_fx_preset(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    fx_index: u32,
    take_index: Option<u32>,
    name: Option<&str>,
    index: Option<i64>,
    navigate: Option<&str>,
) -> Result<Value, String> {
    let nav = preset_nav_move(name, index, navigate)?;
    let name_c = match name {
        Some(n) => Some(CString::new(n).map_err(|_| "preset name has a NUL byte".to_string())?),
        None => None,
    };
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let take = match take_index {
        Some(ti) => unsafe { low.GetMediaItemTake(item.as_ptr(), ti as c_int) },
        None => unsafe { low.GetActiveTake(item.as_ptr()) },
    };
    if take.is_null() {
        return Err("no such take".to_string());
    }
    let fx = fx_index as c_int;
    reaper.undo_begin_block_2(project);
    let ok = if let Some(c) = &name_c {
        unsafe { low.TakeFX_SetPreset(take, fx, c.as_ptr()) }
    } else if let Some(i) = index {
        unsafe { low.TakeFX_SetPresetByIndex(take, fx, i as c_int) }
    } else {
        unsafe { low.TakeFX_NavigatePresets(take, fx, nav.unwrap_or(0)) }
    };
    let (current, matches) =
        read_preset_name(|b, s| unsafe { low.TakeFX_GetPreset(take, fx, b, s) });
    reaper.undo_end_block_2(
        project,
        format!("AI: set item {item_index} take FX {fx_index} preset"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({
            "set": true, "item_index": item_index, "fx_index": fx_index,
            "preset": current, "matches_preset": matches,
        }))
    } else {
        Err("failed to set preset (unknown preset name/index, or none to navigate to)".into())
    }
}

fn create_empty_item(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    position: f64,
    length: f64,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    if !position.is_finite() || !length.is_finite() || length <= 0.0 {
        return Err("position and length must be finite and length > 0".to_string());
    }
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    let item = unsafe { low.AddMediaItemToTrack(track.as_ptr()) };
    if !item.is_null() {
        unsafe {
            low.SetMediaItemInfo_Value(item, c"D_POSITION".as_ptr(), position.max(0.0));
            low.SetMediaItemInfo_Value(item, c"D_LENGTH".as_ptr(), length);
            low.UpdateArrange();
        }
    }
    reaper.undo_end_block_2(
        project,
        format!("AI: create empty item on track {track_index}"),
        UndoScope::All,
    );
    let item_index =
        MediaItem::new(item).and_then(|it| media_item_index_map(reaper).get(&it).copied());
    Ok(json!({
        "created": !item.is_null(),
        "track_index": track_index,
        "item_index": item_index,
        "position": position,
        "length": length,
    }))
}

/// Edit an existing marker (`is_region=false`) or region by its index_number:
/// read the current values, apply the given overrides, write them back.
#[allow(clippy::too_many_arguments)]
fn update_marker(
    reaper: &Reaper<MainThreadScope>,
    index_number: c_int,
    is_region: bool,
    pos_or_start: Option<f64>,
    end: Option<f64>,
    name: Option<&str>,
    color: Option<c_int>,
) -> Result<Value, String> {
    let low = reaper.low();
    // Find the marker/region's current values by enumeration.
    let mut i: c_int = 0;
    let mut found: Option<(f64, f64, String, c_int)> = None;
    loop {
        let mut is_rgn = false;
        let mut pos = 0.0f64;
        let mut rgn_end = 0.0f64;
        let mut name_ptr: *const c_char = std::ptr::null();
        let mut num: c_int = 0;
        let mut col: c_int = 0;
        let next = unsafe {
            low.EnumProjectMarkers3(
                CUR_PROJ, i, &mut is_rgn, &mut pos, &mut rgn_end, &mut name_ptr, &mut num, &mut col,
            )
        };
        if next == 0 {
            break;
        }
        if num == index_number && is_rgn == is_region {
            found = Some((pos, rgn_end, unsafe { cstr_to_string(name_ptr) }, col));
            break;
        }
        i += 1;
        if i > 1_000_000 {
            break;
        }
    }
    let (cur_pos, cur_end, cur_name, cur_color) = found.ok_or_else(|| {
        format!(
            "no {} with index number {index_number}",
            if is_region { "region" } else { "marker" }
        )
    })?;

    let new_pos = pos_or_start.unwrap_or(cur_pos).max(0.0);
    let new_end = if is_region {
        end.unwrap_or(cur_end).max(0.0)
    } else {
        0.0
    };
    let new_name = name.map(|s| s.to_string()).unwrap_or(cur_name);
    // color: omit or 0 = keep current.
    let new_color = match color {
        Some(c) if c != 0 => c,
        _ => cur_color,
    };
    let name_c = CString::new(new_name).map_err(|_| "name contains a NUL byte".to_string())?;

    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let ok = unsafe {
        low.SetProjectMarker3(
            CUR_PROJ,
            index_number,
            is_region,
            new_pos,
            new_end,
            name_c.as_ptr(),
            new_color,
        )
    };
    reaper.undo_end_block_2(
        project,
        format!(
            "AI: edit {} {index_number}",
            if is_region { "region" } else { "marker" }
        ),
        UndoScope::All,
    );
    if ok {
        Ok(json!({
            "updated": true, "index_number": index_number, "is_region": is_region,
            "position": new_pos, "end": if is_region { Some(new_end) } else { None }
        }))
    } else {
        Err("failed to update the marker/region".to_string())
    }
}

fn delete_envelope_points(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
    start: f64,
    end: f64,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    if !start.is_finite() || !end.is_finite() || end < start {
        return Err("start/end must be finite with end >= start".to_string());
    }
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.DeleteEnvelopePointRange(env, start, end) };
    unsafe { low.Envelope_SortPoints(env) };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete envelope points on track {track_index} envelope {envelope_index}"),
        UndoScope::All,
    );
    Ok(json!({
        "deleted": ok, "track_index": track_index, "envelope_index": envelope_index,
        "start": start, "end": end
    }))
}

// ---- envelope create / edit / clear -----------------------------------------

/// Find a track envelope's 0-based index in the track's envelope list.
fn envelope_index_of(
    low: &reaper_low::Reaper,
    track: MediaTrack,
    env: *mut reaper_low::raw::TrackEnvelope,
) -> Option<u32> {
    if env.is_null() {
        return None;
    }
    let count = unsafe { low.CountTrackEnvelopes(track.as_ptr()) };
    for i in 0..count {
        if unsafe { low.GetTrackEnvelope(track.as_ptr(), i) } == env {
            return Some(i as u32);
        }
    }
    None
}

fn create_fx_envelope(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    param_index: u32,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    // create=true: creates the parameter envelope if it doesn't exist yet.
    let env = unsafe {
        low.GetFXEnvelope(track.as_ptr(), fx_index as c_int, param_index as c_int, true)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: create FX envelope (track {track_index} fx {fx_index} param {param_index})"),
        UndoScope::All,
    );
    if env.is_null() {
        return Err("could not create the FX parameter envelope (bad fx/param index?)".into());
    }
    Ok(json!({
        "created": true,
        "track_index": track_index,
        "fx_index": fx_index,
        "param_index": param_index,
        "envelope_index": envelope_index_of(low, track, env),
    }))
}

fn create_track_envelope(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    kind: &str,
) -> Result<Value, String> {
    // Built-in envelope name (for GetTrackEnvelopeByName) + the "toggle visible"
    // action that creates it when it doesn't exist yet.
    let (name, action): (&str, c_int) = match kind {
        "volume" => ("Volume", 40406),
        "pan" => ("Pan", 40407),
        "mute" => ("Mute", 40866),
        other => return Err(format!("unknown envelope kind '{other}' (use volume, pan, or mute)")),
    };
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let name_c = CString::new(name).map_err(|_| "bad envelope name".to_string())?;

    // Already exists? Return it — never toggle (that would HIDE a visible one).
    let existing = unsafe { low.GetTrackEnvelopeByName(track.as_ptr(), name_c.as_ptr()) };
    if !existing.is_null() {
        return Ok(json!({
            "created": false,
            "already_existed": true,
            "track_index": track_index,
            "kind": kind,
            "envelope_index": envelope_index_of(low, track, existing),
        }));
    }

    // Create via the toggle-visible action (operates on the selected track);
    // restore the prior track selection afterwards.
    let prior = selected_track_set(reaper);
    reaper.undo_begin_block_2(project);
    unsafe { low.SetOnlyTrackSelected(track.as_ptr()) };
    low.Main_OnCommand(action, 0);
    for i in 0..reaper.count_tracks(project) {
        if let Some(t) = reaper.get_track(project, i) {
            unsafe { low.SetTrackSelected(t.as_ptr(), prior.contains(&t)) };
        }
    }
    reaper.undo_end_block_2(
        project,
        format!("AI: create {name} envelope on track {track_index}"),
        UndoScope::All,
    );
    let env = unsafe { low.GetTrackEnvelopeByName(track.as_ptr(), name_c.as_ptr()) };
    Ok(json!({
        "created": !env.is_null(),
        "track_index": track_index,
        "kind": kind,
        "envelope_index": envelope_index_of(low, track, env),
    }))
}

fn set_envelope_point(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
    point_index: u32,
    time: Option<f64>,
    value: Option<f64>,
    shape: Option<c_int>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    // SetEnvelopePointEx: a null in-pointer leaves that field unchanged.
    let mut t = time.unwrap_or(0.0);
    let mut v = value.unwrap_or(0.0);
    let mut s = shape.unwrap_or(0);
    let time_ptr = if time.is_some() { &mut t as *mut f64 } else { std::ptr::null_mut() };
    let value_ptr = if value.is_some() { &mut v as *mut f64 } else { std::ptr::null_mut() };
    let shape_ptr = if shape.is_some() { &mut s as *mut c_int } else { std::ptr::null_mut() };
    let mut no_sort = false;
    reaper.undo_begin_block_2(project);
    let ok = unsafe {
        low.SetEnvelopePointEx(
            env,
            -1,
            point_index as c_int,
            time_ptr,
            value_ptr,
            shape_ptr,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut no_sort,
        )
    };
    unsafe { low.Envelope_SortPoints(env) };
    reaper.undo_end_block_2(
        project,
        format!("AI: edit envelope point {point_index} (track {track_index} env {envelope_index})"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "updated": true, "point_index": point_index }))
    } else {
        Err(format!("no point at index {point_index}"))
    }
}

fn delete_envelope_point(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
    point_index: u32,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.DeleteEnvelopePointEx(env, -1, point_index as c_int) };
    unsafe { low.Envelope_SortPoints(env) };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete envelope point {point_index}"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "deleted": true, "point_index": point_index }))
    } else {
        Err(format!("no point at index {point_index}"))
    }
}

fn clear_envelope(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    envelope_index: u32,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let low = reaper.low();
    let env = unsafe { low.GetTrackEnvelope(track.as_ptr(), envelope_index as c_int) };
    if env.is_null() {
        return Err(format!("no envelope at index {envelope_index}"));
    }
    reaper.undo_begin_block_2(project);
    // Delete every point (a range spanning the whole project timeline).
    let ok = unsafe { low.DeleteEnvelopePointRange(env, -1.0e18, 1.0e18) };
    unsafe { low.Envelope_SortPoints(env) };
    reaper.undo_end_block_2(
        project,
        format!("AI: clear envelope {envelope_index} on track {track_index}"),
        UndoScope::All,
    );
    Ok(json!({ "cleared": ok, "track_index": track_index, "envelope_index": envelope_index }))
}

/// Line indices of every top-level `AUXRECV` (a receive) in a single track's
/// state chunk. "Top-level" = a direct child of the outer `<TRACK` block (nesting
/// depth 1), so `AUXRECV`-like tokens buried inside FX/item sub-blocks can't match.
/// A line whose trimmed form starts with `<` opens a nested block; one starting
/// with `>` closes it.
fn top_level_auxrecv_lines(lines: &[&str]) -> Vec<usize> {
    let mut depth: i32 = 0;
    let mut out = Vec::new();
    for (i, raw) in lines.iter().enumerate() {
        let t = raw.trim_start();
        if t.starts_with('>') {
            depth -= 1;
        } else if t.starts_with('<') {
            depth += 1;
        } else if depth == 1 && t.split_whitespace().next() == Some("AUXRECV") {
            out.push(i);
        }
    }
    out
}

/// Whether a send-envelope block with `block_tag` already exists in the sibling
/// run that follows the `AUXRECV` line at `target` (its volume/pan/mute envelopes
/// live as `<AUX*ENV>` blocks right after it). Stops at the next receive or at the
/// first non-envelope sub-block (`<FXCHAIN`, `<ITEM`, …) that ends the routing header.
fn send_env_present_after(lines: &[&str], target: usize, block_tag: &str) -> bool {
    const AUX: [&str; 3] = ["<AUXVOLENV", "<AUXPANENV", "<AUXMUTEENV"];
    let mut depth: i32 = 1; // we start just after the AUXRECV token, inside <TRACK
    let mut i = target + 1;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if t.starts_with('>') {
            depth -= 1;
            if depth < 1 {
                break; // left the <TRACK block
            }
        } else if t.starts_with('<') {
            if depth == 1 {
                let tag = t.split_whitespace().next().unwrap_or("");
                if tag == block_tag {
                    return true;
                }
                if !AUX.contains(&tag) {
                    break; // reached FXCHAIN/ITEM/etc — routing header is over
                }
            }
            depth += 1;
        } else if depth == 1 && t.split_whitespace().next() == Some("AUXRECV") {
            break; // reached the next receive
        }
        i += 1;
    }
    false
}

/// Create a send/receive automation envelope (send volume/pan/mute).
///
/// REAPER exposes no create API for these (unlike `GetFXEnvelope`), so the only
/// mechanism is to splice an `<AUX*ENV>` block into the DESTINATION track's state
/// chunk after the relevant `AUXRECV` line. Every reviewer-flagged risk is guarded:
/// the target `AUXRECV` is found by depth-aware parsing confined to the routing
/// header; the receive-count is cross-checked against the API; duplicates are
/// prevented by scanning the chunk (not by the P_ENV read, which may be wrong);
/// bracket balance is validated before applying; and success is gated on RE-READING
/// the chunk to confirm REAPER kept the block — if the tag is wrong REAPER silently
/// drops it, we detect the absence and report a hard error, leaving the track
/// effectively unchanged. `P_ENV` is used only to resolve the live pointer for the
/// returned index and is never load-bearing for correctness.
fn create_send_envelope(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    send_index: u32,
    kind: &str,
    category: &str,
) -> Result<Value, String> {
    // (read parmname, chunk block tag, seeded inner lines). ARM 0 matches REAPER's
    // own default so a fresh envelope isn't overwritten under write-automation. One
    // neutral point (1/0/1 = unity gain / centre / audible) guarantees REAPER keeps
    // and enumerates the envelope on round-trip.
    let (read_parm, block_tag, inner): (&str, &str, &[&str]) = match kind {
        "volume" => (
            "P_ENV:<VOLENV",
            "<AUXVOLENV",
            &[
                "ACT 1 -1",
                "VIS 1 1 1",
                "LANEHEIGHT 0 0",
                "ARM 0",
                "DEFSHAPE 0 -1 -1",
                "VOLTYPE 1",
                "PT 0 1 0",
            ],
        ),
        "pan" => (
            "P_ENV:<PANENV",
            "<AUXPANENV",
            &[
                "ACT 1 -1",
                "VIS 1 1 1",
                "LANEHEIGHT 0 0",
                "ARM 0",
                "DEFSHAPE 0 -1 -1",
                "PT 0 0 0",
            ],
        ),
        "mute" => (
            "P_ENV:<MUTEENV",
            "<AUXMUTEENV",
            &[
                "ACT 1 -1",
                "VIS 1 1 1",
                "LANEHEIGHT 0 0",
                "ARM 0",
                "DEFSHAPE 1 -1 -1",
                "PT 0 1 1",
            ],
        ),
        other => {
            return Err(format!(
                "unknown envelope kind '{other}' (use volume, pan, or mute)"
            ))
        }
    };

    let project = ProjectContext::CurrentProject;
    let low = reaper.low();
    let idx_map = track_index_map(reaper);

    // Normalise the request to (source track A, destination track B, receive index
    // j on B). The chunk edit is always on B; the envelope ends up owned by A.
    let (src, dest, recv_index) = match category {
        "send" => {
            let a = reaper
                .get_track(project, track_index)
                .ok_or_else(|| format!("no track at index {track_index}"))?;
            let n = unsafe { reaper.get_track_num_sends(a, TrackSendCategory::Send) };
            if send_index >= n {
                return Err(format!(
                    "track {track_index} has no send #{send_index} (it has {n})"
                ));
            }
            let b = unsafe {
                reaper.get_track_send_info_desttrack(a, TrackSendDirection::Send, send_index)
            }
            .map_err(|_| {
                "this send has no destination track (hardware-output sends are not supported)"
                    .to_string()
            })?;
            // Rank of this send among A's sends that target B; the k-th A->B send
            // corresponds to the k-th A->B receive (REAPER keeps them in order).
            let mut k = 0u32;
            for i2 in 0..send_index {
                if let Ok(d) =
                    unsafe { reaper.get_track_send_info_desttrack(a, TrackSendDirection::Send, i2) }
                {
                    if d == b {
                        k += 1;
                    }
                }
            }
            let rn = unsafe { reaper.get_track_num_sends(b, TrackSendCategory::Receive) };
            let mut seen = 0u32;
            let mut found = None;
            for r in 0..rn {
                if let Ok(s) =
                    unsafe { reaper.get_track_send_info_desttrack(b, TrackSendDirection::Receive, r) }
                {
                    if s == a {
                        if seen == k {
                            found = Some(r);
                            break;
                        }
                        seen += 1;
                    }
                }
            }
            let j = found.ok_or_else(|| {
                "could not map this send to a receive on the destination track".to_string()
            })?;
            (a, b, j)
        }
        "receive" => {
            let b = reaper
                .get_track(project, track_index)
                .ok_or_else(|| format!("no track at index {track_index}"))?;
            let n = unsafe { reaper.get_track_num_sends(b, TrackSendCategory::Receive) };
            if send_index >= n {
                return Err(format!(
                    "track {track_index} has no receive #{send_index} (it has {n})"
                ));
            }
            let a = unsafe {
                reaper.get_track_send_info_desttrack(b, TrackSendDirection::Receive, send_index)
            }
            .map_err(|_| "this receive has no source track".to_string())?;
            (a, b, send_index)
        }
        other => {
            return Err(format!("unknown category '{other}' (use send or receive)"));
        }
    };

    let src_index = idx_map.get(&src).copied();
    let dest_index = idx_map.get(&dest).copied();
    let parm_c = CString::new(read_parm).map_err(|_| "bad envelope parm".to_string())?;

    // Resolve the live envelope pointer (best-effort; may be null if the P_ENV read
    // string doesn't resolve send envelopes on this build — not relied on for safety).
    let resolve_env = || -> *mut reaper_low::raw::TrackEnvelope {
        unsafe {
            low.GetSetTrackSendInfo(
                dest.as_ptr(),
                -1,
                recv_index as c_int,
                parm_c.as_ptr(),
                std::ptr::null_mut(),
            ) as *mut reaper_low::raw::TrackEnvelope
        }
    };
    // The envelope is enumerated on whichever endpoint owns it — discover it by
    // pointer identity rather than assuming the source, so the returned index is
    // always usable by the point tools.
    let owner_of = |env: *mut reaper_low::raw::TrackEnvelope| -> (Option<u32>, Option<u32>) {
        if let Some(ix) = envelope_index_of(low, src, env) {
            return (src_index, Some(ix));
        }
        if let Some(ix) = envelope_index_of(low, dest, env) {
            return (dest_index, Some(ix));
        }
        (None, None)
    };

    let chunk = read_chunk(|b, s| unsafe { low.GetTrackStateChunk(dest.as_ptr(), b, s, false) })
        .ok_or_else(|| "could not read the destination track's state chunk".to_string())?;
    let lines: Vec<&str> = chunk.lines().collect();

    // Cross-check the parsed receive count against the API before trusting the
    // positional match — a mismatch means our parse is off; abort rather than risk
    // editing the wrong routing.
    let recvs = top_level_auxrecv_lines(&lines);
    let api_recvs = unsafe { reaper.get_track_num_sends(dest, TrackSendCategory::Receive) };
    if recvs.len() as u32 != api_recvs {
        return Err(format!(
            "destination track chunk parse mismatch ({} AUXRECV lines vs {api_recvs} receives); \
             aborting to avoid corruption",
            recvs.len()
        ));
    }
    let target = *recvs.get(recv_index as usize).ok_or_else(|| {
        format!("could not locate receive #{recv_index} in the destination track chunk")
    })?;

    // Idempotency: chunk-scan (independent of the P_ENV read) prevents a duplicate
    // block if the envelope already exists.
    if send_env_present_after(&lines, target, block_tag) {
        let env = resolve_env();
        let (owner_ti, env_ix) = owner_of(env);
        return Ok(json!({
            "created": false,
            "already_existed": true,
            "kind": kind,
            "category": category,
            "source_track_index": src_index,
            "dest_track_index": dest_index,
            "receive_index": recv_index,
            "envelope_track_index": owner_ti,
            "envelope_index": env_ix,
        }));
    }

    let indent: String = lines[target]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();

    // Splice the envelope block in immediately after the AUXRECV line.
    let mut new_lines: Vec<String> = Vec::with_capacity(lines.len() + inner.len() + 2);
    for (i, raw) in lines.iter().enumerate() {
        new_lines.push((*raw).to_string());
        if i == target {
            new_lines.push(format!("{indent}{block_tag}"));
            for l in inner {
                new_lines.push(format!("{indent}  {l}"));
            }
            new_lines.push(format!("{indent}>"));
        }
    }

    // Validate bracket balance of the spliced chunk before applying it.
    let mut depth: i32 = 0;
    for raw in &new_lines {
        let t = raw.trim_start();
        if t.starts_with('>') {
            depth -= 1;
            if depth < 0 {
                return Err("internal error: chunk became unbalanced; aborting".into());
            }
        } else if t.starts_with('<') {
            depth += 1;
        }
    }
    if depth != 0 {
        return Err("internal error: chunk not balanced after edit; aborting".into());
    }

    let mut new_chunk = new_lines.join("\n");
    if chunk.ends_with('\n') {
        new_chunk.push('\n');
    }
    let chunk_c =
        CString::new(new_chunk).map_err(|_| "the edited chunk contains a NUL byte".to_string())?;

    reaper.undo_begin_block_2(project);
    let applied = unsafe { low.SetTrackStateChunk(dest.as_ptr(), chunk_c.as_ptr(), false) };
    reaper.undo_end_block_2(
        project,
        format!("AI: create send {kind} envelope"),
        UndoScope::All,
    );
    low.TrackList_AdjustWindows(false);
    reaper.update_arrange();
    if !applied {
        return Err("REAPER rejected the modified track chunk; no change was made".into());
    }

    // Success gate: re-read and confirm REAPER kept the block. If the tag is wrong,
    // REAPER drops the unknown block on parse (leaving the rest of the track intact)
    // and this scan fails — we report an error rather than a false success.
    let verify =
        read_chunk(|b, s| unsafe { low.GetTrackStateChunk(dest.as_ptr(), b, s, false) })
            .ok_or_else(|| "could not re-read the destination chunk to verify".to_string())?;
    let vlines: Vec<&str> = verify.lines().collect();
    let vrecvs = top_level_auxrecv_lines(&vlines);
    let kept = vrecvs
        .get(recv_index as usize)
        .is_some_and(|&vt| send_env_present_after(&vlines, vt, block_tag));
    if !kept {
        return Err(format!(
            "applied the chunk but REAPER did not keep the {block_tag} block — the send-envelope \
             chunk tag is likely different on this REAPER build, so no envelope was created (the \
             destination track is otherwise unchanged). Create one send envelope manually once so \
             its exact chunk format can be confirmed."
        ));
    }

    let env = resolve_env();
    let (owner_ti, env_ix) = owner_of(env);
    let mut out = json!({
        "created": true,
        "kind": kind,
        "category": category,
        "source_track_index": src_index,
        "dest_track_index": dest_index,
        "receive_index": recv_index,
        "envelope_track_index": owner_ti,
        "envelope_index": env_ix,
    });
    if env_ix.is_none() {
        out["note"] = json!(
            "Envelope created, but its live index could not be resolved automatically; call \
             get_track_envelopes on the source track to find it, then use the point tools."
        );
    }
    Ok(out)
}

fn get_tempo_markers(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let count = unsafe { low.CountTempoTimeSigMarkers(CUR_PROJ) };
    let mut markers = Vec::new();
    for i in 0..count {
        let mut timepos = 0.0f64;
        let mut measurepos: c_int = 0;
        let mut beatpos = 0.0f64;
        let mut bpm = 0.0f64;
        let mut num: c_int = 0;
        let mut denom: c_int = 0;
        let mut linear = false;
        let ok = unsafe {
            low.GetTempoTimeSigMarker(
                CUR_PROJ,
                i,
                &mut timepos,
                &mut measurepos,
                &mut beatpos,
                &mut bpm,
                &mut num,
                &mut denom,
                &mut linear,
            )
        };
        if !ok {
            continue;
        }
        markers.push(json!({
            "index": i,
            "time": timepos,
            "measure": measurepos,
            "beat": beatpos,
            "bpm": bpm,
            "timesig_num": num,
            "timesig_denom": denom,
            "linear": linear,
        }));
    }
    json!({ "tempo_markers": markers })
}

fn add_tempo_marker(
    reaper: &Reaper<MainThreadScope>,
    time: f64,
    bpm: f64,
    timesig_num: c_int,
    timesig_denom: c_int,
    linear: bool,
) -> Result<Value, String> {
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let ok = unsafe {
        low.AddTempoTimeSigMarker(CUR_PROJ, time, bpm, timesig_num, timesig_denom, linear)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add tempo marker at {time:.3}s ({bpm:.3} BPM)"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "added": true, "time": time, "bpm": bpm }))
    } else {
        Err("failed to add tempo marker".to_string())
    }
}

fn delete_tempo_marker(reaper: &Reaper<MainThreadScope>, index: c_int) -> Result<Value, String> {
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.DeleteTempoTimeSigMarker(CUR_PROJ, index) };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete tempo marker {index}"),
        UndoScope::All,
    );
    if ok {
        Ok(json!({ "deleted": true, "index": index }))
    } else {
        Err(format!("no tempo marker at index {index}"))
    }
}

fn set_project_tempo(reaper: &Reaper<MainThreadScope>, bpm: f64) -> Result<Value, String> {
    if !(bpm.is_finite() && bpm > 0.0) {
        return Err("bpm must be a positive number".to_string());
    }
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    // want_undo=false: our own block already records the undo point.
    unsafe { low.SetCurrentBPM(CUR_PROJ, bpm, false) };
    reaper.undo_end_block_2(
        project,
        format!("AI: set project tempo to {bpm:.3} BPM"),
        UndoScope::All,
    );
    Ok(json!({ "set": true, "bpm": bpm }))
}

fn get_stretch_markers(reaper: &Reaper<MainThreadScope>, item_index: u32) -> Result<Value, String> {
    let item = reaper
        .get_media_item(ProjectContext::CurrentProject, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    let count = unsafe { low.GetTakeNumStretchMarkers(t) };
    let mut markers = Vec::new();
    for i in 0..count {
        let mut pos = 0.0f64;
        let mut srcpos = 0.0f64;
        let ret = unsafe { low.GetTakeStretchMarker(t, i, &mut pos, &mut srcpos) };
        if ret < 0 {
            continue;
        }
        let slope = unsafe { low.GetTakeStretchMarkerSlope(t, i) };
        markers.push(json!({
            "index": i,
            "position": pos,
            "src_position": srcpos,
            "slope": slope,
        }));
    }
    Ok(json!({ "item_index": item_index, "stretch_markers": markers }))
}

fn add_stretch_marker(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    position: f64,
    src_position: Option<f64>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = reaper
        .get_media_item(project, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    // idx = -1 adds a new marker.
    let idx = match src_position {
        Some(s) => unsafe { low.SetTakeStretchMarker(t, -1, position, &s) },
        None => unsafe { low.SetTakeStretchMarker(t, -1, position, std::ptr::null()) },
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add stretch marker to item {item_index} at {position:.3}s"),
        UndoScope::All,
    );
    Ok(json!({ "added": idx >= 0, "index": idx, "item_index": item_index, "position": position }))
}

fn delete_stretch_marker(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    marker_index: c_int,
    count: Option<c_int>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = reaper
        .get_media_item(project, item_index)
        .ok_or_else(|| format!("no media item at index {item_index}"))?;
    let take = unsafe { reaper.get_active_take(item) }.ok_or("item has no active take")?;
    let t = take.as_ptr();
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    let removed = match count {
        Some(c) => unsafe { low.DeleteTakeStretchMarkers(t, marker_index, &c) },
        None => unsafe { low.DeleteTakeStretchMarkers(t, marker_index, std::ptr::null()) },
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete {removed} stretch marker(s) from item {item_index}"),
        UndoScope::All,
    );
    Ok(json!({ "removed": removed, "item_index": item_index, "marker_index": marker_index }))
}

/// Buffer for render string values (paths / patterns). GetSetProjectInfo_String
/// takes no size argument, so the buffer just needs to be comfortably large.
const RENDER_STR_BUF: usize = 8192;

fn get_render_settings(reaper: &Reaper<MainThreadScope>) -> Value {
    let low = reaper.low();
    let num = |key: &CStr| unsafe { low.GetSetProjectInfo(CUR_PROJ, key.as_ptr(), 0.0, false) };
    let text = |key: &CStr| {
        read_string(RENDER_STR_BUF, |b, _s| unsafe {
            low.GetSetProjectInfo_String(CUR_PROJ, key.as_ptr(), b, false)
        })
        .unwrap_or_default()
    };
    json!({
        "render_settings": num(c"RENDER_SETTINGS"),
        "bounds_flag": num(c"RENDER_BOUNDSFLAG"),
        "channels": num(c"RENDER_CHANNELS"),
        "sample_rate": num(c"RENDER_SRATE"),
        "start_position": num(c"RENDER_STARTPOS"),
        "end_position": num(c"RENDER_ENDPOS"),
        "tail_flag": num(c"RENDER_TAILFLAG"),
        "tail_ms": num(c"RENDER_TAILMS"),
        "add_to_project": num(c"RENDER_ADDTOPROJ"),
        "dither": num(c"RENDER_DITHER"),
        "render_file": text(c"RENDER_FILE"),
        "render_pattern": text(c"RENDER_PATTERN"),
    })
}

fn set_render_setting(
    reaper: &Reaper<MainThreadScope>,
    key: &str,
    value: Option<f64>,
    text: Option<&str>,
) -> Result<Value, String> {
    let key_c = CString::new(key).map_err(|_| "key contains a NUL byte".to_string())?;
    let low = reaper.low();
    let project = ProjectContext::CurrentProject;
    // Validate before opening the undo block so it always stays balanced.
    let result = if let Some(t) = text {
        let mut bytes = t.as_bytes().to_vec();
        bytes.push(0); // NUL terminator
        reaper.undo_begin_block_2(project);
        let ok = unsafe {
            low.GetSetProjectInfo_String(
                CUR_PROJ,
                key_c.as_ptr(),
                bytes.as_mut_ptr() as *mut c_char,
                true,
            )
        };
        reaper.undo_end_block_2(
            project,
            format!("AI: set render setting {key}"),
            UndoScope::All,
        );
        json!({ "set": ok, "key": key, "text": t })
    } else if let Some(v) = value {
        reaper.undo_begin_block_2(project);
        unsafe { low.GetSetProjectInfo(CUR_PROJ, key_c.as_ptr(), v, true) };
        reaper.undo_end_block_2(
            project,
            format!("AI: set render setting {key}"),
            UndoScope::All,
        );
        json!({ "set": true, "key": key, "value": v })
    } else {
        return Err("provide 'value' (number) or 'text' (string)".to_string());
    };
    Ok(result)
}

// ---- item / take / track properties, grouping, copy/move -------------------

/// Friendly-name -> raw REAPER key tables. Used by both the readers (iterate
/// all) and the setters (look up one). Values flow through the low-level
/// `*Info_Value` functions as f64.
const ITEM_PROPS: &[(&str, &CStr)] = &[
    ("position", c"D_POSITION"),
    ("length", c"D_LENGTH"),
    ("volume", c"D_VOL"),
    ("mute", c"B_MUTE"),
    ("loop_source", c"B_LOOPSRC"),
    ("lock", c"C_LOCK"),
    ("snap_offset", c"D_SNAPOFFSET"),
    ("fade_in_len", c"D_FADEINLEN"),
    ("fade_out_len", c"D_FADEOUTLEN"),
    ("fade_in_len_auto", c"D_FADEINLEN_AUTO"),
    ("fade_out_len_auto", c"D_FADEOUTLEN_AUTO"),
    ("fade_in_shape", c"C_FADEINSHAPE"),
    ("fade_out_shape", c"C_FADEOUTSHAPE"),
    ("fade_in_dir", c"D_FADEINDIR"),
    ("fade_out_dir", c"D_FADEOUTDIR"),
    ("group_id", c"I_GROUPID"),
    ("color", c"I_CUSTOMCOLOR"),
    ("all_takes_play", c"B_ALLTAKESPLAY"),
];

const TAKE_PROPS: &[(&str, &CStr)] = &[
    ("start_offset", c"D_STARTOFFS"),
    ("volume", c"D_VOL"),
    ("pan", c"D_PAN"),
    ("playrate", c"D_PLAYRATE"),
    ("pitch", c"D_PITCH"),
    ("preserve_pitch", c"B_PPITCH"),
    ("channel_mode", c"I_CHANMODE"),
    ("color", c"I_CUSTOMCOLOR"),
];

const TRACK_PROPS: &[(&str, &CStr)] = &[
    ("mute", c"B_MUTE"),
    ("solo", c"I_SOLO"),
    ("volume", c"D_VOL"),
    ("pan", c"D_PAN"),
    ("visible_tcp", c"B_SHOWINTCP"),
    ("visible_mixer", c"B_SHOWINMIXER"),
    ("height", c"I_HEIGHTOVERRIDE"),
    ("height_lock", c"B_HEIGHTLOCK"),
    ("folder_depth", c"I_FOLDERDEPTH"),
    ("folder_compact", c"I_FOLDERCOMPACT"),
    ("free_mode", c"B_FREEMODE"),
    ("color", c"I_CUSTOMCOLOR"),
    ("rec_arm", c"I_RECARM"),
    ("rec_monitor", c"I_RECMON"),
];

/// Numeric take attributes copied when duplicating a take.
const TAKE_COPY_KEYS: &[&CStr] = &[
    c"D_STARTOFFS",
    c"D_VOL",
    c"D_PAN",
    c"D_PLAYRATE",
    c"D_PITCH",
    c"B_PPITCH",
    c"I_CHANMODE",
    c"I_CUSTOMCOLOR",
];

/// Track grouping parameters queried by get_track_group_membership.
const GROUP_PARAMS: &[&CStr] = &[
    c"VOLUME_LEAD",
    c"VOLUME_FOLLOW",
    c"PAN_LEAD",
    c"PAN_FOLLOW",
    c"WIDTH_LEAD",
    c"WIDTH_FOLLOW",
    c"MUTE_LEAD",
    c"MUTE_FOLLOW",
    c"SOLO_LEAD",
    c"SOLO_FOLLOW",
    c"RECARM_LEAD",
    c"RECARM_FOLLOW",
    c"POLARITY_LEAD",
    c"POLARITY_FOLLOW",
    c"AUTOMODE_LEAD",
    c"AUTOMODE_FOLLOW",
    c"VOLUME_VCA_LEAD",
    c"VOLUME_VCA_FOLLOW",
];

/// Every group-name REAPER's GetSetTrackGroupMembership accepts (current
/// LEAD/FOLLOW names plus the deprecated MASTER/SLAVE aliases). Used to reject
/// typos, which would otherwise silently no-op while reporting success.
const GROUP_PARAM_NAMES: &[&str] = &[
    "MEDIA_EDIT_LEAD",
    "MEDIA_EDIT_FOLLOW",
    "VOLUME_LEAD",
    "VOLUME_FOLLOW",
    "VOLUME_VCA_LEAD",
    "VOLUME_VCA_FOLLOW",
    "VOLUME_VCA_FOLLOW_ISPREFX",
    "PAN_LEAD",
    "PAN_FOLLOW",
    "WIDTH_LEAD",
    "WIDTH_FOLLOW",
    "MUTE_LEAD",
    "MUTE_FOLLOW",
    "SOLO_LEAD",
    "SOLO_FOLLOW",
    "RECARM_LEAD",
    "RECARM_FOLLOW",
    "POLARITY_LEAD",
    "POLARITY_FOLLOW",
    "AUTOMODE_LEAD",
    "AUTOMODE_FOLLOW",
    "VOLUME_REVERSE",
    "PAN_REVERSE",
    "WIDTH_REVERSE",
    "NO_LEAD_WHEN_FOLLOW",
    // deprecated pre-v6.12 aliases, still accepted by REAPER
    "MEDIA_EDIT_MASTER",
    "MEDIA_EDIT_SLAVE",
    "VOLUME_MASTER",
    "VOLUME_SLAVE",
    "VOLUME_VCA_MASTER",
    "VOLUME_VCA_SLAVE",
    "PAN_MASTER",
    "PAN_SLAVE",
    "WIDTH_MASTER",
    "WIDTH_SLAVE",
    "MUTE_MASTER",
    "MUTE_SLAVE",
    "SOLO_MASTER",
    "SOLO_SLAVE",
    "RECARM_MASTER",
    "RECARM_SLAVE",
    "POLARITY_MASTER",
    "POLARITY_SLAVE",
    "AUTOMODE_MASTER",
    "AUTOMODE_SLAVE",
];

fn lookup_key(table: &[(&str, &'static CStr)], name: &str) -> Option<&'static CStr> {
    table.iter().find(|e| e.0 == name).map(|e| e.1)
}

fn item_at(reaper: &Reaper<MainThreadScope>, index: u32) -> Result<MediaItem, String> {
    reaper
        .get_media_item(ProjectContext::CurrentProject, index)
        .ok_or_else(|| format!("no media item at index {index}"))
}

fn track_at(reaper: &Reaper<MainThreadScope>, index: u32) -> Result<MediaTrack, String> {
    reaper
        .get_track(ProjectContext::CurrentProject, index)
        .ok_or_else(|| format!("no track at index {index}"))
}

fn resolve_take(
    reaper: &Reaper<MainThreadScope>,
    item: MediaItem,
    take_index: Option<u32>,
) -> Result<MediaItemTake, String> {
    match take_index {
        Some(i) => {
            if i > i32::MAX as u32 {
                return Err(format!("take index {i} out of range"));
            }
            let ptr = unsafe { reaper.low().GetTake(item.as_ptr(), i as c_int) };
            MediaItemTake::new(ptr).ok_or_else(|| format!("no take at index {i}"))
        }
        None => unsafe { reaper.get_active_take(item) }
            .ok_or_else(|| "item has no active take".to_string()),
    }
}

fn get_item_properties(reaper: &Reaper<MainThreadScope>, item_index: u32) -> Result<Value, String> {
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let ip = item.as_ptr();
    let mut props = serde_json::Map::new();
    for &(name, key) in ITEM_PROPS {
        let v = unsafe { low.GetMediaItemInfo_Value(ip, key.as_ptr()) };
        props.insert(name.to_string(), json!(v));
    }
    let take_count = unsafe { low.CountTakes(ip) };
    Ok(json!({ "item_index": item_index, "take_count": take_count, "properties": props }))
}

fn set_item_property(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    property: &str,
    value: f64,
) -> Result<Value, String> {
    let key = lookup_key(ITEM_PROPS, property)
        .ok_or_else(|| format!("unknown item property '{property}'"))?;
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.SetMediaItemInfo_Value(item.as_ptr(), key.as_ptr(), value) };
    unsafe { low.UpdateItemInProject(item.as_ptr()) };
    reaper.undo_end_block_2(
        project,
        format!("AI: set item {item_index} {property}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    if ok {
        Ok(json!({ "set": true, "item_index": item_index, "property": property, "value": value }))
    } else {
        Err(format!("failed to set item property '{property}'"))
    }
}

/// Move an item's left or right edge to an absolute timeline time, atomically.
/// Right edge = a length change. Left edge = move position + shift the take's
/// source offset (scaled by playrate) so the content stays put + shrink length.
fn set_item_edge(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    edge: &str,
    time: f64,
) -> Result<Value, String> {
    /// Smallest length we allow — a zero/negative-length item is invalid.
    const MIN_LEN: f64 = 1e-6;
    let edge = edge.trim().to_ascii_lowercase();
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let ip = item.as_ptr();
    let old_pos = unsafe { low.GetMediaItemInfo_Value(ip, c"D_POSITION".as_ptr()) };
    let old_len = unsafe { low.GetMediaItemInfo_Value(ip, c"D_LENGTH".as_ptr()) };

    match edge.as_str() {
        "right" => {
            let new_len = time - old_pos;
            if new_len < MIN_LEN {
                return Err(format!(
                    "right edge {time:.6}s is at or before the item start {old_pos:.6}s"
                ));
            }
            reaper.undo_begin_block_2(project);
            unsafe { low.SetMediaItemInfo_Value(ip, c"D_LENGTH".as_ptr(), new_len) };
            unsafe { low.UpdateItemInProject(ip) };
            reaper.undo_end_block_2(
                project,
                format!("AI: set item {item_index} right edge"),
                UndoScope::All,
            );
            reaper.update_arrange();
            Ok(json!({
                "set": true, "item_index": item_index, "edge": "right",
                "position": old_pos, "length": new_len, "right_edge": old_pos + new_len,
            }))
        }
        "left" => {
            let delta = time - old_pos; // > 0 trims from the left, < 0 extends left
            let new_len = old_len - delta;
            if new_len < MIN_LEN {
                return Err(format!(
                    "left edge {time:.6}s would leave a non-positive length (item ends at {:.6}s)",
                    old_pos + old_len
                ));
            }
            reaper.undo_begin_block_2(project);
            unsafe { low.SetMediaItemInfo_Value(ip, c"D_POSITION".as_ptr(), time) };
            // The position change applies to EVERY take, so keep them all aligned:
            // shift each take's start-in-source by delta * its OWN playrate
            // (D_STARTOFFS is in SOURCE seconds). Do NOT clamp — REAPER represents a
            // pre-source region (leading silence / loop padding) with a negative
            // offset, and clamping to 0 would slide the content instead. Empty items
            // have no takes, so only the position/length move.
            let take_count = unsafe { low.CountTakes(ip) };
            let active = unsafe { low.GetMediaItemInfo_Value(ip, c"I_CURTAKE".as_ptr()) } as c_int;
            let mut active_off: Value = json!(null);
            let mut extended_before_source = false;
            for ti in 0..take_count {
                let tp = unsafe { low.GetTake(ip, ti) };
                if tp.is_null() {
                    continue;
                }
                let playrate = {
                    let r = unsafe { low.GetMediaItemTakeInfo_Value(tp, c"D_PLAYRATE".as_ptr()) };
                    if r > 0.0 { r } else { 1.0 }
                };
                let old_off = unsafe { low.GetMediaItemTakeInfo_Value(tp, c"D_STARTOFFS".as_ptr()) };
                let new_off = old_off + delta * playrate;
                if new_off < -1e-9 {
                    extended_before_source = true;
                }
                unsafe { low.SetMediaItemTakeInfo_Value(tp, c"D_STARTOFFS".as_ptr(), new_off) };
                if ti == active {
                    active_off = json!(new_off);
                }
            }
            unsafe { low.SetMediaItemInfo_Value(ip, c"D_LENGTH".as_ptr(), new_len) };
            unsafe { low.UpdateItemInProject(ip) };
            reaper.undo_end_block_2(
                project,
                format!("AI: set item {item_index} left edge"),
                UndoScope::All,
            );
            reaper.update_arrange();
            let mut out = json!({
                "set": true, "item_index": item_index, "edge": "left",
                "position": time, "length": new_len,
                "start_offset": active_off, "takes_adjusted": take_count,
            });
            if extended_before_source {
                out["note"] = json!(
                    "extended before the source start; the leading region plays silence (or loops \
                     if loop_source is on)"
                );
            }
            Ok(out)
        }
        other => Err(format!("edge must be 'left' or 'right', got '{other}'")),
    }
}

/// Read a named FX config value (TrackFX_GetNamedConfigParm), e.g. "fx_type".
fn get_fx_config(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    name: &str,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    if name.as_bytes().contains(&0) {
        return Err("name contains a NUL byte".to_string());
    }
    let fx = i32::try_from(fx_index).map_err(|_| "fx_index out of range".to_string())?;
    let mut nb = name.as_bytes().to_vec();
    nb.push(0);
    let val = read_string(8192, |b, s| unsafe {
        low.TrackFX_GetNamedConfigParm(track.as_ptr(), fx, nb.as_ptr() as *const c_char, b, s)
    });
    match val {
        Some(v) => Ok(json!({
            "track_index": track_index, "fx_index": fx_index, "name": name, "value": v,
        })),
        None => Err(format!(
            "no config value '{name}' for FX {fx_index} on track {track_index}"
        )),
    }
}

/// Read a track's full RPP-format state chunk (contains the Video processor's EEL
/// code, among everything else). Read-only.
fn get_track_state_chunk(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    // Use the shared grow-to-fit reader (GetTrackStateChunk silently truncates a
    // too-small buffer): it grows up to 64 MiB and returns None rather than a
    // truncated chunk, so we never hand back (and can't write back) a partial one.
    let chunk = read_chunk(|b, s| unsafe {
        low.GetTrackStateChunk(track.as_ptr(), b, s, false)
    })
    .ok_or("could not read the track state chunk (too large or unavailable)")?;
    Ok(json!({ "track_index": track_index, "chunk": chunk }))
}

/// Replace a track's full state chunk. Mutation (confirmed + undo-wrapped).
fn set_track_state_chunk(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    chunk: &str,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    if chunk.as_bytes().contains(&0) {
        return Err("chunk contains a NUL byte".to_string());
    }
    let mut bytes = chunk.as_bytes().to_vec();
    bytes.push(0);
    let project = ProjectContext::CurrentProject;
    reaper.undo_begin_block_2(project);
    let ok = unsafe {
        low.SetTrackStateChunk(track.as_ptr(), bytes.as_ptr() as *const c_char, false)
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} state chunk"),
        UndoScope::All,
    );
    reaper.update_arrange();
    if ok {
        Ok(json!({ "set": true, "track_index": track_index }))
    } else {
        Err("REAPER rejected the state chunk (likely malformed)".to_string())
    }
}

fn get_take_properties(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let item = item_at(reaper, item_index)?;
    let take = resolve_take(reaper, item, take_index)?;
    let low = reaper.low();
    let tp = take.as_ptr();
    let mut props = serde_json::Map::new();
    for &(name, key) in TAKE_PROPS {
        let v = unsafe { low.GetMediaItemTakeInfo_Value(tp, key.as_ptr()) };
        props.insert(name.to_string(), json!(v));
    }
    let name = take_name(reaper, take);
    let source = unsafe { low.GetMediaItemTake_Source(tp) };
    let (source_len, is_qn, source_file) = if source.is_null() {
        (0.0, false, String::new())
    } else {
        let mut lengthis_qn = false;
        let len = unsafe { low.GetMediaSourceLength(source, &mut lengthis_qn) };
        let file = read_string(4096, |b, s| {
            unsafe { low.GetMediaSourceFileName(source, b, s) };
            true
        })
        .unwrap_or_default();
        (len, lengthis_qn, file)
    };
    Ok(json!({
        "item_index": item_index,
        "take_index": take_index,
        "name": name,
        "properties": props,
        "source_length": source_len,
        "source_length_is_qn": is_qn,
        "source_file": source_file,
    }))
}

fn set_take_property(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    property: &str,
    value: Option<f64>,
    text: Option<&str>,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let take = resolve_take(reaper, item, take_index)?;
    let low = reaper.low();
    let tp = take.as_ptr();
    if property == "name" {
        let text = text.ok_or_else(|| "property 'name' requires 'text'".to_string())?;
        if text.as_bytes().contains(&0) {
            return Err("'text' contains a NUL byte".to_string());
        }
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        reaper.undo_begin_block_2(project);
        let ok = unsafe {
            low.GetSetMediaItemTakeInfo_String(
                tp,
                c"P_NAME".as_ptr(),
                bytes.as_mut_ptr() as *mut c_char,
                true,
            )
        };
        reaper.undo_end_block_2(
            project,
            format!("AI: rename take of item {item_index}"),
            UndoScope::All,
        );
        reaper.update_arrange();
        return if ok {
            Ok(json!({ "set": true, "item_index": item_index, "property": "name", "text": text }))
        } else {
            Err("failed to set take name".to_string())
        };
    }
    let key = lookup_key(TAKE_PROPS, property)
        .ok_or_else(|| format!("unknown take property '{property}'"))?;
    let value = value.ok_or_else(|| format!("property '{property}' requires 'value'"))?;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.SetMediaItemTakeInfo_Value(tp, key.as_ptr(), value) };
    reaper.undo_end_block_2(
        project,
        format!("AI: set take {property} of item {item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    if ok {
        Ok(json!({ "set": true, "item_index": item_index, "property": property, "value": value }))
    } else {
        Err(format!("failed to set take property '{property}'"))
    }
}

fn set_active_take(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    take_index: u32,
) -> Result<Value, String> {
    if take_index > i32::MAX as u32 {
        return Err(format!("take index {take_index} out of range"));
    }
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let ptr = unsafe { low.GetTake(item.as_ptr(), take_index as c_int) };
    let take = MediaItemTake::new(ptr).ok_or_else(|| format!("no take at index {take_index}"))?;
    reaper.undo_begin_block_2(project);
    unsafe { low.SetActiveTake(take.as_ptr()) };
    unsafe { low.UpdateItemInProject(item.as_ptr()) };
    reaper.undo_end_block_2(
        project,
        format!("AI: set active take {take_index} on item {item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    Ok(json!({ "set": true, "item_index": item_index, "take_index": take_index }))
}

fn get_track_properties(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    let trp = track.as_ptr();
    let mut props = serde_json::Map::new();
    for &(name, key) in TRACK_PROPS {
        let v = unsafe { low.GetMediaTrackInfo_Value(trp, key.as_ptr()) };
        props.insert(name.to_string(), json!(v));
    }
    Ok(
        json!({ "track_index": track_index, "name": track_name(reaper, track), "properties": props }),
    )
}

fn set_track_property(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    property: &str,
    value: Option<f64>,
    text: Option<&str>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    if property == "name" {
        let text = text.ok_or_else(|| "property 'name' requires 'text'".to_string())?;
        if text.as_bytes().contains(&0) {
            return Err("'text' contains a NUL byte".to_string());
        }
        reaper.undo_begin_block_2(project);
        unsafe { reaper.get_set_media_track_info_set_name(track, text) };
        reaper.undo_end_block_2(
            project,
            format!("AI: rename track {track_index}"),
            UndoScope::All,
        );
        return Ok(
            json!({ "set": true, "track_index": track_index, "property": "name", "text": text }),
        );
    }
    let key = lookup_key(TRACK_PROPS, property)
        .ok_or_else(|| format!("unknown track property '{property}'"))?;
    let value = value.ok_or_else(|| format!("property '{property}' requires 'value'"))?;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.SetMediaTrackInfo_Value(track.as_ptr(), key.as_ptr(), value) };
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} {property}"),
        UndoScope::All,
    );
    // Visibility/height/folder changes only take visual effect after a layout pass.
    low.TrackList_AdjustWindows(false);
    reaper.update_arrange();
    if ok {
        Ok(json!({ "set": true, "track_index": track_index, "property": property, "value": value }))
    } else {
        Err(format!("failed to set track property '{property}'"))
    }
}

fn get_track_group_membership(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    let trp = track.as_ptr();
    let mut groups = serde_json::Map::new();
    for key in GROUP_PARAMS {
        let lo = unsafe { low.GetSetTrackGroupMembership(trp, key.as_ptr(), 0, 0) };
        let hi = unsafe { low.GetSetTrackGroupMembershipHigh(trp, key.as_ptr(), 0, 0) };
        if lo == 0 && hi == 0 {
            continue;
        }
        let mut nums = Vec::new();
        for b in 0..32u32 {
            if lo & (1u32 << b) != 0 {
                nums.push(b + 1);
            }
        }
        for b in 0..32u32 {
            if hi & (1u32 << b) != 0 {
                nums.push(b + 33);
            }
        }
        groups.insert(key.to_str().unwrap_or_default().to_string(), json!(nums));
    }
    Ok(json!({
        "track_index": track_index,
        "groups": groups,
        "note": "each entry lists the group numbers (1..64) the track belongs to for that parameter",
    }))
}

fn set_track_group_membership(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    group: u32,
    param: &str,
    member: bool,
) -> Result<Value, String> {
    if !(1..=64).contains(&group) {
        return Err("group must be between 1 and 64".to_string());
    }
    if !GROUP_PARAM_NAMES.contains(&param) {
        return Err(format!(
            "unknown group parameter '{param}'; expected one of e.g. VOLUME_LEAD, VOLUME_FOLLOW, \
             MUTE_FOLLOW, SOLO_LEAD, VOLUME_VCA_LEAD, VOLUME_VCA_FOLLOW"
        ));
    }
    let param_c = CString::new(param).map_err(|_| "invalid param".to_string())?;
    let project = ProjectContext::CurrentProject;
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    let trp = track.as_ptr();
    reaper.undo_begin_block_2(project);
    let previous = if group <= 32 {
        let bit = 1u32 << (group - 1);
        let val = if member { bit } else { 0 };
        unsafe { low.GetSetTrackGroupMembership(trp, param_c.as_ptr(), bit, val) }
    } else {
        let bit = 1u32 << (group - 33);
        let val = if member { bit } else { 0 };
        unsafe { low.GetSetTrackGroupMembershipHigh(trp, param_c.as_ptr(), bit, val) }
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} {param} group {group}"),
        UndoScope::All,
    );
    Ok(json!({
        "set": true, "track_index": track_index, "group": group, "param": param,
        "member": member, "previous_mask": previous
    }))
}

fn group_items(
    reaper: &Reaper<MainThreadScope>,
    item_indices: &[u32],
    group_id: Option<i64>,
) -> Result<Value, String> {
    if item_indices.is_empty() {
        return Err("provide at least one item index".to_string());
    }
    let project = ProjectContext::CurrentProject;
    let items: Vec<MediaItem> = item_indices
        .iter()
        .map(|&i| item_at(reaper, i))
        .collect::<Result<_, _>>()?;
    let low = reaper.low();
    let gid = match group_id {
        Some(g) => g as f64,
        None => {
            // Allocate a fresh group id: one past the largest in use.
            let mut max_g = 0i64;
            for i in 0..reaper.count_media_items(project) {
                if let Some(it) = reaper.get_media_item(project, i) {
                    let g =
                        unsafe { low.GetMediaItemInfo_Value(it.as_ptr(), c"I_GROUPID".as_ptr()) }
                            as i64;
                    max_g = max_g.max(g);
                }
            }
            (max_g + 1) as f64
        }
    };
    reaper.undo_begin_block_2(project);
    for it in &items {
        unsafe { low.SetMediaItemInfo_Value(it.as_ptr(), c"I_GROUPID".as_ptr(), gid) };
    }
    reaper.undo_end_block_2(
        project,
        format!("AI: group {} item(s)", items.len()),
        UndoScope::All,
    );
    reaper.update_arrange();
    Ok(json!({ "grouped": items.len(), "group_id": gid as i64 }))
}

fn copy_item(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    dest_track_index: Option<u32>,
    position: Option<f64>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    // Resolve the destination track before opening the undo block.
    let dest_track = match dest_track_index {
        Some(ti) => track_at(reaper, ti)?,
        None => {
            let ptr = unsafe { low.GetMediaItemTrack(item.as_ptr()) };
            MediaTrack::new(ptr).ok_or_else(|| "could not resolve the source track".to_string())?
        }
    };
    let chunk = read_chunk(|b, s| unsafe { low.GetItemStateChunk(item.as_ptr(), b, s, false) })
        .ok_or_else(|| "could not read the item's state chunk".to_string())?;
    // Strip the item/take GUID lines so REAPER assigns fresh ones to the copy.
    let chunk = strip_chunk_lines(&chunk, &["IGUID", "GUID"]);
    let chunk_c = CString::new(chunk).map_err(|_| "item chunk contains a NUL byte".to_string())?;
    reaper.undo_begin_block_2(project);
    let new_item = MediaItem::new(unsafe { low.AddMediaItemToTrack(dest_track.as_ptr()) });
    let ok = match new_item {
        Some(ni) => {
            let set_ok = unsafe { low.SetItemStateChunk(ni.as_ptr(), chunk_c.as_ptr(), false) };
            if set_ok {
                if let Some(pos) = position {
                    unsafe { low.SetMediaItemInfo_Value(ni.as_ptr(), c"D_POSITION".as_ptr(), pos) };
                }
                unsafe { low.UpdateItemInProject(ni.as_ptr()) };
            } else {
                // Roll back the blank item we just added so we don't leave junk.
                unsafe { low.DeleteTrackMediaItem(dest_track.as_ptr(), ni.as_ptr()) };
            }
            set_ok
        }
        None => false,
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: copy item {item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    if !ok {
        return Err("failed to copy item".to_string());
    }
    let new_index = new_item.and_then(|ni| media_item_index_map(reaper).get(&ni).copied());
    Ok(json!({ "copied": true, "source_item_index": item_index, "new_item_index": new_index }))
}

fn move_item(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    dest_track_index: Option<u32>,
    position: Option<f64>,
) -> Result<Value, String> {
    if dest_track_index.is_none() && position.is_none() {
        return Err("provide dest_track_index and/or position".to_string());
    }
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    // Resolve destination before the undo block so failures stay balanced.
    let dest = match dest_track_index {
        Some(ti) => Some(track_at(reaper, ti)?),
        None => None,
    };
    let low = reaper.low();
    reaper.undo_begin_block_2(project);
    let mut reparented = false;
    if let Some(d) = dest {
        reparented = unsafe { low.MoveMediaItemToTrack(item.as_ptr(), d.as_ptr()) };
    }
    if let Some(pos) = position {
        unsafe { low.SetMediaItemInfo_Value(item.as_ptr(), c"D_POSITION".as_ptr(), pos) };
    }
    unsafe { low.UpdateItemInProject(item.as_ptr()) };
    reaper.undo_end_block_2(
        project,
        format!("AI: move item {item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    let new_index = media_item_index_map(reaper).get(&item).copied();
    Ok(json!({ "moved": true, "item_index": new_index, "track_changed": reparented }))
}

fn delete_item(reaper: &Reaper<MainThreadScope>, item_index: u32) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let item = item_at(reaper, item_index)?;
    let low = reaper.low();
    let track = MediaTrack::new(unsafe { low.GetMediaItemTrack(item.as_ptr()) })
        .ok_or_else(|| "could not resolve the item's track".to_string())?;
    reaper.undo_begin_block_2(project);
    let ok = unsafe { low.DeleteTrackMediaItem(track.as_ptr(), item.as_ptr()) };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete item {item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    if ok {
        Ok(json!({ "deleted": true, "item_index": item_index }))
    } else {
        Err("failed to delete item".to_string())
    }
}

fn duplicate_track(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let src = track_at(reaper, track_index)?;
    let low = reaper.low();
    let chunk = read_chunk(|b, s| unsafe { low.GetTrackStateChunk(src.as_ptr(), b, s, false) })
        .ok_or_else(|| "could not read the track's state chunk".to_string())?;
    // Strip the track's own GUID so the duplicate gets a fresh TRACKID; a
    // collision would break routing / grouping / VCA references keyed by GUID.
    let chunk = strip_chunk_lines(&chunk, &["TRACKID"]);
    let chunk_c = CString::new(chunk).map_err(|_| "track chunk contains a NUL byte".to_string())?;
    let new_idx = track_index + 1;
    reaper.undo_begin_block_2(project);
    reaper.insert_track_at_index(new_idx, TrackDefaultsBehavior::OmitDefaultEnvAndFx);
    let result = match reaper.get_track(project, new_idx) {
        Some(new_track) => {
            if unsafe { low.SetTrackStateChunk(new_track.as_ptr(), chunk_c.as_ptr(), false) } {
                Ok(new_idx)
            } else {
                Err("failed to apply the track chunk to the new track".to_string())
            }
        }
        None => Err("could not fetch the inserted track".to_string()),
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: duplicate track {track_index}"),
        UndoScope::All,
    );
    low.TrackList_AdjustWindows(false);
    reaper.update_arrange();
    result.map(|ni| json!({ "duplicated": true, "source_track_index": track_index, "new_track_index": ni }))
}

fn delete_track(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = track_at(reaper, track_index)?;
    reaper.undo_begin_block_2(project);
    unsafe { reaper.delete_track(track) };
    reaper.undo_end_block_2(
        project,
        format!("AI: delete track {track_index}"),
        UndoScope::All,
    );
    reaper.low().TrackList_AdjustWindows(false);
    reaper.update_arrange();
    Ok(json!({ "deleted": true, "track_index": track_index }))
}

fn copy_take(
    reaper: &Reaper<MainThreadScope>,
    src_item_index: u32,
    dest_item_index: u32,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let src_item = item_at(reaper, src_item_index)?;
    let dest_item = item_at(reaper, dest_item_index)?;
    let src_take = resolve_take(reaper, src_item, take_index)?;
    let low = reaper.low();
    let stp = src_take.as_ptr();
    let source = unsafe { low.GetMediaItemTake_Source(stp) };
    if source.is_null() {
        return Err("the source take has no media source".to_string());
    }
    // Recreating from the file loses section/reverse trimming, so refuse those.
    let src_type = {
        let mut buf = vec![0u8; 64];
        unsafe {
            low.GetMediaSourceType(source, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int)
        };
        buf_to_string(&buf)
    };
    if src_type.eq_ignore_ascii_case("SECTION") {
        return Err(
            "copy_take cannot faithfully copy a section/reverse take source; use copy_item to \
             duplicate the whole item."
                .to_string(),
        );
    }
    let file = read_string(4096, |b, s| {
        unsafe { low.GetMediaSourceFileName(source, b, s) };
        true
    })
    .unwrap_or_default();
    if file.trim().is_empty() {
        return Err(
            "copy_take supports file-based audio takes; this take has an in-project source \
             (e.g. MIDI). Use copy_item to duplicate the whole item."
                .to_string(),
        );
    }
    let file_c =
        CString::new(file).map_err(|_| "source filename contains a NUL byte".to_string())?;
    let take_name_src = take_name(reaper, src_take);
    // Remember the destination's active take so the copy lands as an inactive take.
    let prior_active = unsafe { reaper.get_active_take(dest_item) };
    // Create the source before opening the undo block; destroy it if we bail.
    let new_src = unsafe { low.PCM_Source_CreateFromFile(file_c.as_ptr()) };
    if new_src.is_null() {
        return Err("could not create a source from the take's file".to_string());
    }
    reaper.undo_begin_block_2(project);
    let new_take_ptr = unsafe { low.AddTakeToMediaItem(dest_item.as_ptr()) };
    if new_take_ptr.is_null() {
        unsafe { low.PCM_Source_Destroy(new_src) };
        reaper.undo_end_block_2(
            project,
            "AI: copy take (failed)".to_string(),
            UndoScope::All,
        );
        return Err("could not add a take to the destination item".to_string());
    }
    if !unsafe { low.SetMediaItemTake_Source(new_take_ptr, new_src) } {
        // Source not adopted — free it so we don't leak, and report failure.
        unsafe { low.PCM_Source_Destroy(new_src) };
        reaper.undo_end_block_2(
            project,
            "AI: copy take (failed)".to_string(),
            UndoScope::All,
        );
        return Err("could not attach the source to the new take".to_string());
    }
    for key in TAKE_COPY_KEYS {
        let v = unsafe { low.GetMediaItemTakeInfo_Value(stp, key.as_ptr()) };
        unsafe { low.SetMediaItemTakeInfo_Value(new_take_ptr, key.as_ptr(), v) };
    }
    if !take_name_src.is_empty() {
        if let Ok(name_c) = CString::new(take_name_src) {
            let mut bytes = name_c.into_bytes_with_nul();
            unsafe {
                low.GetSetMediaItemTakeInfo_String(
                    new_take_ptr,
                    c"P_NAME".as_ptr(),
                    bytes.as_mut_ptr() as *mut c_char,
                    true,
                )
            };
        }
    }
    // Keep the previously-active take active (if the item had one).
    if let Some(pa) = prior_active {
        unsafe { low.SetActiveTake(pa.as_ptr()) };
    }
    unsafe { low.UpdateItemInProject(dest_item.as_ptr()) };
    reaper.undo_end_block_2(
        project,
        format!("AI: copy take to item {dest_item_index}"),
        UndoScope::All,
    );
    reaper.update_arrange();
    let new_take_index = unsafe { low.CountTakes(dest_item.as_ptr()) } - 1;
    Ok(json!({
        "copied": true,
        "src_item_index": src_item_index,
        "dest_item_index": dest_item_index,
        "new_take_index": new_take_index,
    }))
}

// ---- audio analysis (Phase 6) ----------------------------------------------

/// Fixed analysis sample rate; the accessor resamples to this, and it matches
/// the BS.1770 K-weighting coefficients in `crate::dsp`.
const ANALYZE_SR: c_int = 48_000;
/// Cap on how much audio one call reads/analyses, to bound the main-thread cost
/// (accessor reads must run on the main thread).
const MAX_ANALYZE_SECONDS: f64 = 30.0;
/// Read the accessor in blocks of this many frames so no single call is huge.
const READ_BLOCK_FRAMES: usize = ANALYZE_SR as usize; // 1 second

/// Read interleaved f64 samples from an audio accessor over `[start, start+len)`
/// at [`ANALYZE_SR`] / `channels`, block by block. Generic over the opaque
/// accessor pointer type (which reaper-low does not export a nameable alias for).
/// Reads samples and reports whether any block failed (GetAudioAccessorSamples
/// returns -1 on error), so a read failure isn't silently reported as silence.
fn read_accessor_samples<A>(
    low: &reaper_low::Reaper,
    accessor: *mut A,
    channels: c_int,
    start: f64,
    length: f64,
) -> (Vec<f64>, bool) {
    let ch = channels.max(1) as usize;
    let total = (length * ANALYZE_SR as f64).round().max(0.0) as usize;
    let mut out: Vec<f64> = Vec::with_capacity(total * ch);
    let mut had_error = false;
    let mut done = 0usize;
    while done < total {
        let n = READ_BLOCK_FRAMES.min(total - done);
        let t = start + done as f64 / ANALYZE_SR as f64;
        let mut buf = vec![0.0f64; n * ch];
        let ret = unsafe {
            low.GetAudioAccessorSamples(
                accessor as *mut _,
                ANALYZE_SR,
                channels,
                t,
                n as c_int,
                buf.as_mut_ptr(),
            )
        };
        if ret < 0 {
            had_error = true;
        }
        out.extend_from_slice(&buf);
        done += n;
    }
    (out, had_error)
}

fn audio_result_json(
    mut base: Value,
    start: f64,
    length: f64,
    truncated: bool,
    pre_fx: bool,
    read_error: bool,
    features: crate::dsp::AudioFeatures,
) -> Value {
    base["analysis_start"] = json!(start);
    base["analysis_length"] = json!(length);
    base["truncated"] = json!(truncated);
    base["pre_fx"] = json!(pre_fx);
    if read_error {
        base["read_error"] = json!(true);
    }
    base["features"] = serde_json::to_value(features).unwrap_or(Value::Null);
    base
}

fn analyze_item_audio(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
    take_index: Option<u32>,
) -> Result<Value, String> {
    let item = item_at(reaper, item_index)?;
    let take = resolve_take(reaper, item, take_index)?;
    let low = reaper.low();
    // Channel count from the take's source (mono vs. stereo), clamped to 2.
    let source = unsafe { low.GetMediaItemTake_Source(take.as_ptr()) };
    let channels = if source.is_null() {
        2
    } else {
        unsafe { low.GetMediaSourceNumChannels(source) }.clamp(1, 2)
    };
    let acc = unsafe { low.CreateTakeAudioAccessor(take.as_ptr()) };
    if acc.is_null() {
        return Err("could not create an audio accessor for the take".to_string());
    }
    let acc_start = unsafe { low.GetAudioAccessorStartTime(acc) };
    let acc_end = unsafe { low.GetAudioAccessorEndTime(acc) };
    let available = (acc_end - acc_start).max(0.0);
    let length = available.min(MAX_ANALYZE_SECONDS);
    let (samples, read_error) = read_accessor_samples(low, acc, channels, acc_start, length);
    unsafe { low.DestroyAudioAccessor(acc) };
    let features = crate::dsp::analyze(&samples, channels as usize, ANALYZE_SR as f64);
    Ok(audio_result_json(
        json!({ "item_index": item_index, "take_index": take_index }),
        acc_start,
        length,
        available > MAX_ANALYZE_SECONDS,
        true,
        read_error,
        features,
    ))
}

fn analyze_track_audio(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    param_start: Option<f64>,
    param_length: Option<f64>,
) -> Result<Value, String> {
    let track = track_at(reaper, track_index)?;
    let low = reaper.low();
    let channels = (unsafe { low.GetMediaTrackInfo_Value(track.as_ptr(), c"I_NCHAN".as_ptr()) }
        as c_int)
        .clamp(1, 2);
    let acc = unsafe { low.CreateTrackAudioAccessor(track.as_ptr()) };
    if acc.is_null() {
        return Err("could not create an audio accessor for the track".to_string());
    }
    let acc_start = unsafe { low.GetAudioAccessorStartTime(acc) };
    let acc_end = unsafe { low.GetAudioAccessorEndTime(acc) };
    let start = param_start.unwrap_or(acc_start).max(acc_start);
    let available = (acc_end - start).max(0.0);
    let requested = param_length.unwrap_or(available).clamp(0.0, available);
    let length = requested.min(MAX_ANALYZE_SECONDS);
    let (samples, read_error) = read_accessor_samples(low, acc, channels, start, length);
    unsafe { low.DestroyAudioAccessor(acc) };
    let features = crate::dsp::analyze(&samples, channels as usize, ANALYZE_SR as f64);
    Ok(audio_result_json(
        json!({ "track_index": track_index }),
        start,
        length,
        requested > MAX_ANALYZE_SECONDS,
        true,
        read_error,
        features,
    ))
}

/// Time-series (over-time) analysis of a take or track: RMS/peak envelope, silent
/// regions, and optionally transient onsets + a single-frequency level track.
/// Read-only (local DSP), so no confirmation/consent gate.
#[allow(clippy::too_many_arguments)]
fn analyze_audio_timeline(
    reaper: &Reaper<MainThreadScope>,
    item_index: Option<u32>,
    take_index: Option<u32>,
    track_index: Option<u32>,
    param_start: Option<f64>,
    param_length: Option<f64>,
    window_ms: f64,
    silence_threshold_db: f64,
    min_silence_ms: f64,
    detect_transients: bool,
    target_hz: Option<f64>,
) -> Result<Value, String> {
    let low = reaper.low();
    // Resolve an audio accessor + channel count + read-start + the TIMELINE offset
    // to add so reported times are absolute, for an item take XOR a track (both
    // accessor creators return the same handle type).
    //
    // A track accessor's start time IS project time. A take accessor's is
    // item-local (0 = the item's start), so its results must be offset by the
    // item's timeline position D_POSITION to become absolute.
    let (acc, channels, acc_start, available, time_offset, label) = if let Some(ii) = item_index {
        let item = item_at(reaper, ii)?;
        let take = resolve_take(reaper, item, take_index)?;
        let source = unsafe { low.GetMediaItemTake_Source(take.as_ptr()) };
        let channels = if source.is_null() {
            2
        } else {
            unsafe { low.GetMediaSourceNumChannels(source) }.clamp(1, 2)
        };
        let acc = unsafe { low.CreateTakeAudioAccessor(take.as_ptr()) };
        if acc.is_null() {
            return Err("could not create an audio accessor for the take".to_string());
        }
        let s = unsafe { low.GetAudioAccessorStartTime(acc) };
        let e = unsafe { low.GetAudioAccessorEndTime(acc) };
        let item_pos = unsafe { low.GetMediaItemInfo_Value(item.as_ptr(), c"D_POSITION".as_ptr()) };
        (
            acc,
            channels,
            s,
            (e - s).max(0.0),
            item_pos + s,
            json!({ "item_index": ii, "take_index": take_index, "item_position": item_pos }),
        )
    } else if let Some(ti) = track_index {
        let track = track_at(reaper, ti)?;
        let channels = (unsafe {
            low.GetMediaTrackInfo_Value(track.as_ptr(), c"I_NCHAN".as_ptr())
        } as c_int)
            .clamp(1, 2);
        let acc = unsafe { low.CreateTrackAudioAccessor(track.as_ptr()) };
        if acc.is_null() {
            return Err("could not create an audio accessor for the track".to_string());
        }
        let s0 = unsafe { low.GetAudioAccessorStartTime(acc) };
        let e = unsafe { low.GetAudioAccessorEndTime(acc) };
        let s = param_start.unwrap_or(s0).max(s0);
        (acc, channels, s, (e - s).max(0.0), s, json!({ "track_index": ti }))
    } else {
        return Err("provide item_index or track_index".to_string());
    };

    let requested = param_length.unwrap_or(available).clamp(0.0, available);
    let length = requested.min(MAX_ANALYZE_SECONDS);
    let (samples, read_error) = read_accessor_samples(low, acc, channels, acc_start, length);
    unsafe { low.DestroyAudioAccessor(acc) };

    let opts = crate::dsp::TimelineOpts {
        window_ms,
        silence_threshold_db,
        min_silence_ms,
        detect_transients,
        target_hz,
        time_offset,
        max_points: 512,
    };
    let analysis = crate::dsp::analyze_timeline(&samples, channels as usize, ANALYZE_SR as f64, &opts);

    let mut out = json!({
        "source": label,
        "analysis_start": time_offset,
        "analysis_length": length,
        "truncated": requested > MAX_ANALYZE_SECONDS,
        "analysis": serde_json::to_value(&analysis).unwrap_or(Value::Null),
    });
    if read_error {
        out["read_error"] = json!(true);
    }
    Ok(out)
}

// ---- processed (post-FX) audio via an offline render ------------------------

/// Default render window (seconds) when no time selection or explicit range.
const PROCESSED_DEFAULT_SECONDS: f64 = 20.0;
/// An item longer than this is rejected for a processed render (item renders
/// span the whole item, so this bounds the temp-file size, the decode
/// allocation, and the synchronous main-thread render/analysis time).
// The render runs synchronously and freezes REAPER's UI for its duration, so
// keep the worst case bounded (matches the master/track window cap).
const MAX_PROCESSED_ITEM_SECONDS: f64 = 30.0;
/// Numeric render settings we override and must restore afterwards.
const RENDER_NUM_KEYS: &[&CStr] = &[
    c"RENDER_SETTINGS",
    c"RENDER_BOUNDSFLAG",
    c"RENDER_STARTPOS",
    c"RENDER_ENDPOS",
    c"RENDER_SRATE",
    c"RENDER_CHANNELS",
    c"RENDER_TAILFLAG",
    c"RENDER_ADDTOPROJ",
    c"RENDER_NORMALIZE",
    c"RENDER_DITHER",
];
/// String render settings we override and must restore afterwards. RENDER_FORMAT2
/// is included so any secondary render the user had configured is disabled during
/// the probe (one output file) and restored after.
const RENDER_STR_KEYS: &[&CStr] = &[
    c"RENDER_FORMAT",
    c"RENDER_FORMAT2",
    c"RENDER_FILE",
    c"RENDER_PATTERN",
];

fn proj_info_get(low: &reaper_low::Reaper, key: &CStr) -> f64 {
    unsafe { low.GetSetProjectInfo(CUR_PROJ, key.as_ptr(), 0.0, false) }
}

fn proj_info_set(low: &reaper_low::Reaper, key: &CStr, value: f64) {
    unsafe { low.GetSetProjectInfo(CUR_PROJ, key.as_ptr(), value, true) };
}

fn proj_info_str_get(low: &reaper_low::Reaper, key: &CStr) -> String {
    read_string(RENDER_STR_BUF, |b, _s| unsafe {
        low.GetSetProjectInfo_String(CUR_PROJ, key.as_ptr(), b, false)
    })
    .unwrap_or_default()
}

/// Snapshot a string render setting for later restore. Returns None when the get
/// fails, so restore can skip it rather than clobber the live value with "".
/// Uses a large buffer — RENDER_FORMAT is a base64 sink config that can be KBs.
fn proj_info_str_snapshot(low: &reaper_low::Reaper, key: &CStr) -> Option<String> {
    read_string(256 * 1024, |b, _s| unsafe {
        low.GetSetProjectInfo_String(CUR_PROJ, key.as_ptr(), b, false)
    })
}

fn proj_info_str_set(low: &reaper_low::Reaper, key: &CStr, value: &str) {
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    unsafe {
        low.GetSetProjectInfo_String(
            CUR_PROJ,
            key.as_ptr(),
            bytes.as_mut_ptr() as *mut c_char,
            true,
        )
    };
}

/// RAII guard that restores the render settings (and, for a track probe, the
/// track selection) and deletes the temp render file when it drops — so an early
/// return or a panic can never leave the user's project reconfigured.
struct RenderStateGuard<'a> {
    reaper: &'a Reaper<MainThreadScope>,
    num: Vec<f64>,
    strs: Vec<Option<String>>,
    /// (selected non-master tracks, was the master track selected) — track probe only.
    track_selection: Option<(std::collections::HashSet<MediaTrack>, bool)>,
    /// Selected media items — item probe only.
    item_selection: Option<std::collections::HashSet<MediaItem>>,
    path: Option<String>,
}

impl Drop for RenderStateGuard<'_> {
    fn drop(&mut self) {
        let low = self.reaper.low();
        if let Some((selected, master_selected)) = &self.track_selection {
            let project = ProjectContext::CurrentProject;
            for i in 0..self.reaper.count_tracks(project) {
                if let Some(t) = self.reaper.get_track(project, i) {
                    unsafe { low.SetTrackSelected(t.as_ptr(), selected.contains(&t)) };
                }
            }
            let master = unsafe { low.GetMasterTrack(CUR_PROJ) };
            if !master.is_null() {
                unsafe { low.SetTrackSelected(master, *master_selected) };
            }
        }
        if let Some(items) = &self.item_selection {
            let project = ProjectContext::CurrentProject;
            unsafe { low.SelectAllMediaItems(CUR_PROJ, false) };
            // Iterate CURRENT items and re-select by set membership, so a stored
            // pointer for an item deleted meanwhile is never dereferenced (mirrors
            // the track restore above).
            for i in 0..self.reaper.count_media_items(project) {
                if let Some(it) = self.reaper.get_media_item(project, i) {
                    if items.contains(&it) {
                        unsafe { low.SetMediaItemSelected(it.as_ptr(), true) };
                    }
                }
            }
        }
        for (k, v) in RENDER_NUM_KEYS.iter().zip(&self.num) {
            proj_info_set(low, k, *v);
        }
        for (k, v) in RENDER_STR_KEYS.iter().zip(&self.strs) {
            if let Some(s) = v {
                proj_info_str_set(low, k, s);
            }
        }
        if let Some(p) = &self.path {
            if !p.is_empty() {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}

/// Analyse PROCESSED (post-FX) audio: briefly render the master mix, a track's
/// processed output, or a single item (through its take + track FX) to a temp
/// WAV, decode + analyse it. All settings/selection changes are undone (and the
/// temp file removed) by a Drop guard on every path. Runs synchronously on the
/// main thread (REAPER is briefly unresponsive during the render).
fn analyze_processed_audio(
    reaper: &Reaper<MainThreadScope>,
    input: &Value,
) -> Result<Value, String> {
    let target = req_str(input, "target")?;
    let track_index = opt_u32(input, "track_index");
    let item_index = opt_u32(input, "item_index");
    let param_start = input.get("start").and_then(|v| v.as_f64());
    let param_length = input.get("length").and_then(|v| v.as_f64());
    let (track, item) = match target {
        "master" => (None, None),
        "track" => {
            let ti =
                track_index.ok_or_else(|| "target 'track' requires track_index".to_string())?;
            (Some(track_at(reaper, ti)?), None)
        }
        "item" => {
            let ii = item_index.ok_or_else(|| "target 'item' requires item_index".to_string())?;
            (None, Some(item_at(reaper, ii)?))
        }
        other => {
            return Err(format!(
                "unknown target '{other}' (use 'master', 'track', or 'item')"
            ))
        }
    };
    let low = reaper.low();

    // Choose the render mode, bounds and window per target.
    let render_settings: f64;
    let bounds_flag: f64;
    let rstart: f64;
    let rlen: f64;
    let truncated: bool;
    let chain: &str;
    if let Some(it) = item {
        // 32 = render selected media items (through take FX + track FX, no master);
        // 4 = selected-media-items bounds (renders the item over its own extent).
        let pos = unsafe { low.GetMediaItemInfo_Value(it.as_ptr(), c"D_POSITION".as_ptr()) };
        let len = unsafe { low.GetMediaItemInfo_Value(it.as_ptr(), c"D_LENGTH".as_ptr()) };
        if !(pos.is_finite() && len.is_finite()) || len <= 0.0 {
            return Err("item has an invalid or zero length".to_string());
        }
        if len > MAX_PROCESSED_ITEM_SECONDS {
            return Err(format!(
                "item is longer than {MAX_PROCESSED_ITEM_SECONDS:.0} s; analyse a track window instead"
            ));
        }
        render_settings = 32.0;
        bounds_flag = 4.0;
        rstart = pos;
        rlen = len;
        truncated = false;
        chain = "take FX + track FX";
    } else {
        // master / track: explicit range, else the time selection, else the
        // project content, capped at the analysis limit; custom time bounds.
        let (mut sel_start, mut sel_end) = (0.0f64, 0.0f64);
        unsafe { low.GetSet_LoopTimeRange(false, false, &mut sel_start, &mut sel_end, false) };
        let has_selection = sel_end > sel_start;
        let start = param_start
            .unwrap_or(if has_selection { sel_start } else { 0.0 })
            .max(0.0);
        let default_len = if has_selection && param_start.is_none() {
            sel_end - sel_start
        } else {
            let master = unsafe { low.GetMasterTrack(CUR_PROJ) };
            let acc = unsafe { low.CreateTrackAudioAccessor(master) };
            if acc.is_null() {
                PROCESSED_DEFAULT_SECONDS
            } else {
                let end = unsafe { low.GetAudioAccessorEndTime(acc) };
                unsafe { low.DestroyAudioAccessor(acc) };
                let content = (end - start).max(0.0);
                if content > 0.0 {
                    content
                } else {
                    PROCESSED_DEFAULT_SECONDS
                }
            }
        };
        let requested_len = param_length.unwrap_or(default_len);
        let len = requested_len.clamp(0.0, MAX_ANALYZE_SECONDS);
        if !start.is_finite() || !len.is_finite() || len <= 0.0 {
            return Err("invalid or empty render window".to_string());
        }
        render_settings = if track.is_some() { 128.0 } else { 0.0 };
        bounds_flag = 0.0;
        rstart = start;
        rlen = len;
        truncated = requested_len.is_finite() && requested_len > MAX_ANALYZE_SECONDS;
        chain = if track.is_some() {
            "track FX + master FX"
        } else {
            "full mix including master FX"
        };
    }
    let rend = rstart + rlen;

    // Snapshot BEFORE mutating; the guard restores everything on drop.
    let track_selection = track.map(|_| {
        let master = unsafe { low.GetMasterTrack(CUR_PROJ) };
        let master_selected = !master.is_null()
            && unsafe { low.GetMediaTrackInfo_Value(master, c"I_SELECTED".as_ptr()) } != 0.0;
        (selected_track_set(reaper), master_selected)
    });
    let item_selection = item.map(|_| selected_item_set(reaper));
    let mut guard = RenderStateGuard {
        reaper,
        num: RENDER_NUM_KEYS
            .iter()
            .map(|k| proj_info_get(low, k))
            .collect(),
        strs: RENDER_STR_KEYS
            .iter()
            .map(|k| proj_info_str_snapshot(low, k))
            .collect(),
        track_selection,
        item_selection,
        path: None,
    };

    // Configure the render: 32-bit float WAV, 48 kHz stereo, no tail/normalize/
    // dither, don't add to project.
    let tmp_dir = std::env::temp_dir().to_string_lossy().to_string();
    let base = format!("raai_render_probe_{}", std::process::id());
    // 32-bit float WAV, as REAPER's own base64 render-config: fourcc "evaw" +
    // 0x20 (32-bit) + float flag. A BARE "evaw" base64-decodes to only 3 bytes —
    // too short for the config struct — so REAPER read past the buffer and
    // corrupted the heap (this was the crash). Float avoids render-time clamping,
    // so peak/clip analysis stays accurate. RENDER_FORMAT2 is left untouched.
    proj_info_str_set(low, c"RENDER_FORMAT", "ZXZhdyAAAQ==");
    proj_info_set(low, c"RENDER_SETTINGS", render_settings);
    proj_info_set(low, c"RENDER_BOUNDSFLAG", bounds_flag);
    proj_info_set(low, c"RENDER_STARTPOS", rstart);
    proj_info_set(low, c"RENDER_ENDPOS", rend);
    proj_info_set(low, c"RENDER_SRATE", ANALYZE_SR as f64);
    proj_info_set(low, c"RENDER_CHANNELS", 2.0);
    proj_info_set(low, c"RENDER_TAILFLAG", 0.0);
    proj_info_set(low, c"RENDER_ADDTOPROJ", 0.0);
    proj_info_set(low, c"RENDER_NORMALIZE", 0.0);
    proj_info_set(low, c"RENDER_DITHER", 0.0);
    proj_info_str_set(low, c"RENDER_FILE", &tmp_dir);
    proj_info_str_set(low, c"RENDER_PATTERN", &base);
    if let Some(t) = track {
        unsafe { low.SetOnlyTrackSelected(t.as_ptr()) };
    }
    if let Some(it) = item {
        unsafe { low.SelectAllMediaItems(CUR_PROJ, false) };
        unsafe { low.SetMediaItemSelected(it.as_ptr(), true) };
    }
    let path = proj_info_str_get(low, c"RENDER_TARGETS")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    guard.path = Some(path.clone());

    let outcome: Result<(Vec<f64>, usize, f64), String> = if path.is_empty() {
        Err("REAPER reported no render target path".to_string())
    } else {
        // Pre-delete so an existing file can't trigger a modal overwrite prompt.
        let _ = std::fs::remove_file(&path);
        // 42230 = File: Render project, using the most recent render settings.
        low.Main_OnCommand(42230, 0);
        std::fs::read(&path)
            .map_err(|e| format!("could not read the rendered file: {e}"))
            .and_then(|bytes| crate::dsp::parse_wav(&bytes))
    };

    // `?` here drops `guard`, which restores settings/selection and deletes the file.
    let (samples, channels, sr) = outcome?;
    let features = crate::dsp::analyze(&samples, channels, sr);
    Ok(audio_result_json(
        json!({ "target": target, "track_index": track_index, "item_index": item_index, "chain": chain }),
        rstart,
        rlen,
        truncated,
        false,
        false,
        features,
    ))
}

// ---- processed master/track clip: offline render + decode (shared) ----------

/// A decoded processed-output clip from an offline render.
struct RenderedClip {
    samples: Vec<f64>,
    channels: usize,
    sample_rate: f64,
    rstart: f64,
    rlen: f64,
    truncated: bool,
}

/// Render the processed master mix (`select_track` = None) or a track's post-FX
/// output (`Some(track)`) over a window to a temp WAV and decode it. Window:
/// explicit range, else the time selection, else project content, capped at
/// `cap_seconds`. All render/selection changes + the temp file are reverted by a
/// Drop guard on every path. Shared by `measure_loudness` and `listen_to_audio`.
fn render_master_or_track(
    reaper: &Reaper<MainThreadScope>,
    select_track: Option<MediaTrack>,
    param_start: Option<f64>,
    param_length: Option<f64>,
    cap_seconds: f64,
) -> Result<RenderedClip, String> {
    let is_master = select_track.is_none();
    let low = reaper.low();

    let (mut sel_start, mut sel_end) = (0.0f64, 0.0f64);
    unsafe { low.GetSet_LoopTimeRange(false, false, &mut sel_start, &mut sel_end, false) };
    let has_selection = sel_end > sel_start;
    let start = param_start
        .unwrap_or(if has_selection { sel_start } else { 0.0 })
        .max(0.0);
    let default_len = if has_selection && param_start.is_none() {
        sel_end - sel_start
    } else {
        let master = unsafe { low.GetMasterTrack(CUR_PROJ) };
        let acc = unsafe { low.CreateTrackAudioAccessor(master) };
        if acc.is_null() {
            PROCESSED_DEFAULT_SECONDS
        } else {
            let end = unsafe { low.GetAudioAccessorEndTime(acc) };
            unsafe { low.DestroyAudioAccessor(acc) };
            let content = (end - start).max(0.0);
            if content > 0.0 {
                content
            } else {
                PROCESSED_DEFAULT_SECONDS
            }
        }
    };
    let requested_len = param_length.unwrap_or(default_len);
    let rlen = requested_len.clamp(0.0, cap_seconds);
    if !start.is_finite() || !rlen.is_finite() || rlen <= 0.0 {
        return Err("invalid or empty render window".to_string());
    }
    let truncated = requested_len.is_finite() && requested_len > cap_seconds;
    let (rstart, rend) = (start, start + rlen);

    // Snapshot render state (+ track selection for a track target) BEFORE mutating.
    let track_selection = select_track.map(|_| {
        let master = unsafe { low.GetMasterTrack(CUR_PROJ) };
        let master_selected = !master.is_null()
            && unsafe { low.GetMediaTrackInfo_Value(master, c"I_SELECTED".as_ptr()) } != 0.0;
        (selected_track_set(reaper), master_selected)
    });
    let mut guard = RenderStateGuard {
        reaper,
        num: RENDER_NUM_KEYS.iter().map(|k| proj_info_get(low, k)).collect(),
        strs: RENDER_STR_KEYS
            .iter()
            .map(|k| proj_info_str_snapshot(low, k))
            .collect(),
        track_selection,
        item_selection: None,
        path: None,
    };

    let tmp_dir = std::env::temp_dir().to_string_lossy().to_string();
    let base = format!("raai_clip_probe_{}", std::process::id());
    proj_info_str_set(low, c"RENDER_FORMAT", "ZXZhdyAAAQ==");
    proj_info_set(low, c"RENDER_SETTINGS", if is_master { 0.0 } else { 128.0 });
    proj_info_set(low, c"RENDER_BOUNDSFLAG", 0.0);
    proj_info_set(low, c"RENDER_STARTPOS", rstart);
    proj_info_set(low, c"RENDER_ENDPOS", rend);
    proj_info_set(low, c"RENDER_SRATE", ANALYZE_SR as f64);
    proj_info_set(low, c"RENDER_CHANNELS", 2.0);
    proj_info_set(low, c"RENDER_TAILFLAG", 0.0);
    proj_info_set(low, c"RENDER_ADDTOPROJ", 0.0);
    proj_info_set(low, c"RENDER_NORMALIZE", 0.0);
    proj_info_set(low, c"RENDER_DITHER", 0.0);
    proj_info_str_set(low, c"RENDER_FILE", &tmp_dir);
    proj_info_str_set(low, c"RENDER_PATTERN", &base);
    if let Some(t) = select_track {
        unsafe { low.SetOnlyTrackSelected(t.as_ptr()) };
    }
    let path = proj_info_str_get(low, c"RENDER_TARGETS")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    guard.path = Some(path.clone());
    if path.is_empty() {
        return Err("REAPER reported no render target path".to_string());
    }
    let _ = std::fs::remove_file(&path);
    // 42230 = File: Render project, using the most recent render settings.
    low.Main_OnCommand(42230, 0);

    // `?` here drops `guard`, which restores settings/selection and deletes the file.
    let bytes =
        std::fs::read(&path).map_err(|e| format!("could not read the rendered file: {e}"))?;
    let (samples, channels, sample_rate) = crate::dsp::parse_wav(&bytes)?;
    Ok(RenderedClip {
        samples,
        channels,
        sample_rate,
        rstart,
        rlen,
        truncated,
    })
    // Drop guard here: restore render settings/selection, delete temp file.
}

/// Resolve a master/track target to the render's `select_track` (None = master).
fn master_or_track_select(
    reaper: &Reaper<MainThreadScope>,
    target: &str,
    track_index: Option<u32>,
) -> Result<Option<MediaTrack>, String> {
    match target {
        "master" => Ok(None),
        "track" => {
            let ti = track_index.ok_or_else(|| "target 'track' requires track_index".to_string())?;
            Ok(Some(track_at(reaper, ti)?))
        }
        other => Err(format!("unknown target '{other}' (use 'master' or 'track')")),
    }
}

// ---- loudness of the processed output (offline render + BS.1770 DSP) --------

/// Measure professional loudness (integrated LUFS, LRA, true-peak, momentary /
/// short-term maxima) of the PROCESSED output via an offline render + our BS.1770
/// DSP. (An earlier design inserted REAPER's loudness_meter JSFX and read its
/// params, but an offline render processes a discarded copy of the chain, so a
/// live FX instance's sliders are never updated — the meter read back silence.)
fn measure_loudness(reaper: &Reaper<MainThreadScope>, input: &Value) -> Result<Value, String> {
    let target = req_str(input, "target")?;
    let track_index = opt_u32(input, "track_index");
    let select = master_or_track_select(reaper, target, track_index)?;
    let clip = render_master_or_track(
        reaper,
        select,
        input.get("start").and_then(|v| v.as_f64()),
        input.get("length").and_then(|v| v.as_f64()),
        MAX_ANALYZE_SECONDS,
    )?;
    let f = crate::dsp::analyze(&clip.samples, clip.channels, clip.sample_rate);

    Ok(json!({
        "target": target,
        "track_index": track_index,
        "chain": if select.is_none() { "full mix including master FX" } else { "track FX + master FX" },
        "measurement_start": clip.rstart,
        "measurement_length": clip.rlen,
        "truncated": clip.truncated,
        "source": "offline render + BS.1770 DSP",
        "loudness": {
            "integrated_lufs": f.integrated_lufs,
            "loudness_range_lu": f.loudness_range_lu,
            "true_peak_dbtp": f.true_peak_dbtp,
            "momentary_lufs_max": f.momentary_lufs_max,
            "short_term_lufs_max": f.short_term_lufs_max,
            "peak_dbfs": f.peak_dbfs,
            "rms_dbfs": f.rms_dbfs,
            "clipping": f.clipping,
        }
    }))
}

/// Render a short processed-audio clip and return it to an audio-capable model to
/// HEAR (Phase 6, direct listening). Capped at 20 s; encoded as PCM16 WAV. Consent
/// is enforced by the worker before this runs (the clip is uploaded to the cloud).
fn listen_to_audio(reaper: &Reaper<MainThreadScope>, input: &Value) -> ToolOutcome {
    use base64::Engine as _;
    const LISTEN_MAX_SECONDS: f64 = 20.0;

    let target = match req_str(input, "target") {
        Ok(t) => t,
        Err(e) => return ToolOutcome::error(json!({ "error": e }).to_string()),
    };
    let track_index = opt_u32(input, "track_index");
    let select = match master_or_track_select(reaper, target, track_index) {
        Ok(s) => s,
        Err(e) => return ToolOutcome::error(json!({ "error": e }).to_string()),
    };
    let clip = match render_master_or_track(
        reaper,
        select,
        input.get("start").and_then(|v| v.as_f64()),
        input.get("length").and_then(|v| v.as_f64()),
        LISTEN_MAX_SECONDS,
    ) {
        Ok(c) => c,
        Err(e) => return ToolOutcome::error(json!({ "error": e }).to_string()),
    };

    let wav = crate::dsp::encode_pcm16_wav(&clip.samples, clip.channels, clip.sample_rate);
    let data_base64 = base64::engine::general_purpose::STANDARD.encode(&wav);
    let summary = json!({
        "target": target,
        "track_index": track_index,
        "rendered_seconds": (clip.rlen * 100.0).round() / 100.0,
        "truncated": clip.truncated,
        "note": "A short audio clip is attached for you to listen to.",
    })
    .to_string();
    ToolOutcome::with_audio(
        summary,
        CapturedAudio {
            format: "wav".into(),
            data_base64,
        },
    )
}

// ---- helpers ----------------------------------------------------------------

fn selected_item_set(reaper: &Reaper<MainThreadScope>) -> std::collections::HashSet<MediaItem> {
    let project = ProjectContext::CurrentProject;
    let mut set = std::collections::HashSet::new();
    for i in 0..reaper.count_selected_media_items(project) {
        if let Some(it) = reaper.get_selected_media_item(project, i) {
            set.insert(it);
        }
    }
    set
}

fn selected_track_set(reaper: &Reaper<MainThreadScope>) -> std::collections::HashSet<MediaTrack> {
    let project = ProjectContext::CurrentProject;
    let mut set = std::collections::HashSet::new();
    for i in 0..reaper.count_selected_tracks_2(project, MasterTrackBehavior::ExcludeMasterTrack) {
        if let Some(t) =
            reaper.get_selected_track_2(project, i, MasterTrackBehavior::ExcludeMasterTrack)
        {
            set.insert(t);
        }
    }
    set
}

fn track_index_map(reaper: &Reaper<MainThreadScope>) -> HashMap<MediaTrack, u32> {
    let project = ProjectContext::CurrentProject;
    let mut map = HashMap::new();
    for i in 0..reaper.count_tracks(project) {
        if let Some(t) = reaper.get_track(project, i) {
            map.insert(t, i);
        }
    }
    map
}

fn media_item_index_map(
    reaper: &Reaper<MainThreadScope>,
) -> HashMap<reaper_medium::MediaItem, u32> {
    let project = ProjectContext::CurrentProject;
    let mut map = HashMap::new();
    for i in 0..reaper.count_media_items(project) {
        if let Some(it) = reaper.get_media_item(project, i) {
            map.insert(it, i);
        }
    }
    map
}

fn track_name(reaper: &Reaper<MainThreadScope>, track: MediaTrack) -> String {
    // SAFETY: track is valid and only used on the main thread here.
    unsafe {
        reaper.get_set_media_track_info_get_name(track, |s| reaper_string(s.as_c_str().to_bytes()))
    }
    .unwrap_or_default()
}

/// Read a take's name via the pointer-returning `GetTakeName` (medium wrapper).
/// This avoids the caller-sized-buffer `GetSetMediaItemTakeInfo_String(P_NAME)`
/// form, which has no length argument and would overflow on a long name.
fn take_name(reaper: &Reaper<MainThreadScope>, take: MediaItemTake) -> String {
    reaper.get_take_name(take, |r| {
        r.map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default()
    })
}

/// Drop chunk lines whose first whitespace-delimited token is one of `keys`, so
/// REAPER assigns fresh identifiers (GUIDs) when the chunk is re-applied to a
/// duplicate — otherwise the copy shares the source's GUID, corrupting anything
/// that resolves objects by GUID (routing, grouping/VCA, associations).
fn strip_chunk_lines(chunk: &str, keys: &[&str]) -> String {
    let mut out = String::with_capacity(chunk.len());
    for line in chunk.lines() {
        let token = line
            .trim_start()
            .split(|c: char| c.is_whitespace())
            .next()
            .unwrap_or("");
        if keys.contains(&token) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn track_location_json(loc: TrackLocation) -> Value {
    match loc {
        TrackLocation::MasterTrack => json!("master"),
        TrackLocation::NormalTrack(i) => json!(i),
    }
}

fn fx_location_json(loc: TrackFxLocation) -> Value {
    match loc {
        TrackFxLocation::NormalFxChain(i) => json!({ "chain": "normal", "index": i }),
        TrackFxLocation::InputFxChain(i) => json!({ "chain": "input", "index": i }),
        _ => json!({ "chain": "other" }),
    }
}

fn reaper_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Fill a buffer via a REAPER low-level call and read back the NUL-terminated
/// UTF-8 string. Returns None if the call reports failure.
fn read_string(cap: usize, f: impl FnOnce(*mut c_char, c_int) -> bool) -> Option<String> {
    let mut buf = vec![0u8; cap];
    let ok = f(buf.as_mut_ptr() as *mut c_char, cap as c_int);
    if !ok {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(cap);
    Some(reaper_string(&buf[..end]))
}

/// Read a REAPER state chunk via a fill-a-buffer call, growing the buffer until
/// the whole chunk fits (chunks are unbounded — a busy track can be MBs). These
/// "NeedBig" APIs silently truncate (still returning success) when the buffer is
/// too small, so completeness is judged by whether the buffer was left
/// *unfilled*: REAPER stops early only when it has written the entire chunk, so
/// at least one spare byte past the NUL terminator proves nothing was cut off.
/// Relying on a trailing `>` is unsound — chunks are full of nested `>` lines.
fn read_chunk(f: impl Fn(*mut c_char, c_int) -> bool) -> Option<String> {
    let mut cap: usize = 512 * 1024;
    let max: usize = 64 * 1024 * 1024;
    loop {
        let mut buf = vec![0u8; cap];
        let ok = f(buf.as_mut_ptr() as *mut c_char, cap as c_int);
        let end = buf.iter().position(|&b| b == 0).unwrap_or(cap);
        if ok && end < cap - 1 {
            return Some(reaper_string(&buf[..end]));
        }
        if cap >= max {
            return None; // give up rather than apply a possibly-truncated chunk
        }
        cap *= 2;
    }
}

/// # Safety
/// `ptr` must be null or a valid NUL-terminated C string owned by REAPER.
unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn req_u32(input: &Value, key: &str) -> Result<u32, String> {
    input
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .ok_or_else(|| format!("missing or invalid '{key}' (expected a non-negative integer)"))
}

fn req_i64(input: &Value, key: &str) -> Result<i64, String> {
    input
        .get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("missing or invalid '{key}' (expected an integer)"))
}

fn opt_usize(input: &Value, key: &str) -> Option<usize> {
    input.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
}

fn opt_u32(input: &Value, key: &str) -> Option<u32> {
    input.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn opt_bool(input: &Value, key: &str) -> Option<bool> {
    input.get(key).and_then(|v| v.as_bool())
}

fn opt_str<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

fn req_str<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    let s = opt_str(input, key)
        .ok_or_else(|| format!("missing or invalid '{key}' (expected a non-empty string)"))?;
    // reaper-medium's ReaperStringArg panics on interior NUL bytes; reject them
    // here so model input can never unwind across the FFI boundary.
    if s.as_bytes().contains(&0) {
        return Err(format!("'{key}' contains a NUL byte"));
    }
    Ok(s)
}

fn req_f64(input: &Value, key: &str) -> Result<f64, String> {
    input
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing or invalid '{key}' (expected a number)"))
}

fn req_bool(input: &Value, key: &str) -> Result<bool, String> {
    input
        .get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("missing or invalid '{key}' (expected true or false)"))
}
