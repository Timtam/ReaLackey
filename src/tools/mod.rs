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

use reaper_medium::{
    AddFxBehavior, FxLocation, ItemAttributeKey, MainThreadScope, MasterTrackBehavior, MediaItem,
    MediaItemTake, MediaTrack, PositionInSeconds, ProjectContext, Reaper,
    ReaperNormalizedFxParamValue, ReaperStr, SendTarget, TrackFxChainType, TrackFxLocation,
    TrackLocation, TrackSendAttributeKey, TrackSendCategory, TrackSendDirection, UndoScope,
};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::providers::ToolDef;

const NAME_BUF: u32 = 256;
const DEFAULT_LIMIT: usize = 200;

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

/// The result of running a tool.
pub struct ToolOutcome {
    pub content: String,
    pub is_error: bool,
}

/// Tool definitions advertised to the model.
pub fn definitions() -> Vec<ToolDef> {
    let obj = |props: Value, required: Value| {
        json!({ "type": "object", "properties": props, "required": required })
    };
    let empty = || json!({ "type": "object", "properties": {} });
    vec![
        ToolDef {
            name: "get_project_summary".into(),
            description: "Lightweight snapshot of the current REAPER project: tempo (BPM), \
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
                          and enabled/offline state."
                .into(),
            input_schema: obj(
                json!({ "track_index": { "type": "integer", "description": "0-based track index" } }),
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
                    "fx_index": { "type": "integer", "description": "0-based FX index in the track chain" },
                    "limit": { "type": "integer", "description": "max parameters to return (default 200)" }
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
            description: "Add an FX/plugin by name to a track's FX chain. This CHANGES the \
                          project: it is confirmed by the user and wrapped in a labelled undo block."
                .into(),
            input_schema: obj(
                json!({
                    "track_index": { "type": "integer" },
                    "fx_name": { "type": "string", "description": "plugin name, e.g. \"ReaEQ\" or \"VST3: Serum\"" }
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
                    "value": { "type": "number", "description": "normalized value 0..1" }
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
                    "enabled": { "type": "boolean" }
                }),
                json!(["track_index", "fx_index", "enabled"]),
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
    ]
}

/// Execute a tool by name on the main thread. Never panics.
pub fn execute(reaper: &Reaper<MainThreadScope>, name: &str, input: &Value) -> ToolOutcome {
    match dispatch(reaper, name, input) {
        Ok(v) => ToolOutcome {
            content: v.to_string(),
            is_error: false,
        },
        Err(e) => ToolOutcome {
            content: json!({ "error": e }).to_string(),
            is_error: true,
        },
    }
}

fn dispatch(reaper: &Reaper<MainThreadScope>, name: &str, input: &Value) -> Result<Value, String> {
    match name {
        "get_project_summary" => Ok(get_project_summary(reaper)),
        "get_tracks" => Ok(get_tracks(reaper)),
        "get_track_fx" => get_track_fx(reaper, req_u32(input, "track_index")?),
        "get_fx_params" => get_fx_params(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            opt_usize(input, "limit").unwrap_or(DEFAULT_LIMIT),
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
        "add_fx" => add_fx(reaper, req_u32(input, "track_index")?, req_str(input, "fx_name")?),
        "set_fx_param" => set_fx_param(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_u32(input, "param_index")?,
            req_f64(input, "value")?,
        ),
        "set_fx_enabled" => set_fx_enabled(
            reaper,
            req_u32(input, "track_index")?,
            req_u32(input, "fx_index")?,
            req_bool(input, "enabled")?,
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
            input.get("notes").ok_or_else(|| "missing 'notes' array".to_string())?,
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
        // notes & per-project memory
        "get_project_notes" => Ok(get_project_notes(reaper)),
        "set_project_notes" => Ok(set_project_notes(
            reaper,
            req_str(input, "text")?,
            opt_bool(input, "append").unwrap_or(true),
        )),
        "get_track_notes" => get_track_notes(reaper, req_u32(input, "track_index")?),
        "set_track_notes" => {
            set_track_notes(reaper, req_u32(input, "track_index")?, req_str(input, "text")?)
        }
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
            input.get("index_number").and_then(|v| v.as_i64()).unwrap_or(-1) as c_int,
        ),
        "add_region" => add_region(
            reaper,
            req_f64(input, "start")?,
            req_f64(input, "end")?,
            opt_str(input, "name").unwrap_or(""),
            input.get("index_number").and_then(|v| v.as_i64()).unwrap_or(-1) as c_int,
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
            input.get("timesig_num").and_then(|v| v.as_i64()).unwrap_or(0) as c_int,
            input.get("timesig_denom").and_then(|v| v.as_i64()).unwrap_or(0) as c_int,
            opt_bool(input, "linear").unwrap_or(false),
        ),
        "delete_tempo_marker" => {
            delete_tempo_marker(reaper, req_i64(input, "index")? as c_int)
        }
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
            input.get("count").and_then(|v| v.as_i64()).map(|c| c as c_int),
        ),
        // render settings
        "get_render_settings" => Ok(get_render_settings(reaper)),
        "set_render_setting" => set_render_setting(
            reaper,
            req_str(input, "key")?,
            input.get("value").and_then(|v| v.as_f64()),
            opt_str(input, "text"),
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
pub fn preview(name: &str, input: &Value) -> Option<String> {
    let show = |k: &str| input.get(k).map(|v| v.to_string()).unwrap_or_else(|| "?".into());
    match name {
        "add_fx" => Some(format!(
            "Add FX {} to track {}",
            input
                .get("fx_name")
                .and_then(|v| v.as_str())
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "?".into()),
            show("track_index"),
        )),
        "set_fx_param" => Some(format!(
            "Set track {} FX {} parameter {} to {} (normalized 0..1)",
            show("track_index"),
            show("fx_index"),
            show("param_index"),
            show("value"),
        )),
        "set_fx_enabled" => Some(format!(
            "{} track {} FX {}",
            if input.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
                "Enable"
            } else {
                "Bypass"
            },
            show("track_index"),
            show("fx_index"),
        )),
        "insert_midi_notes" => Some(format!(
            "Insert {} MIDI note(s) into item {}",
            input.get("notes").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
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
            if input.get("is_region").and_then(|v| v.as_bool()).unwrap_or(false) {
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
    json!({
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

fn get_track_fx(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    // SAFETY: track just obtained from REAPER, used on the main thread here.
    let count = unsafe { reaper.track_fx_get_count(track) };
    let mut fx = Vec::new();
    for i in 0..count {
        let loc = TrackFxLocation::NormalFxChain(i);
        let name = unsafe { reaper.track_fx_get_fx_name(track, loc, NAME_BUF) }
            .ok()
            .map(|s| reaper_string(s.as_c_str().to_bytes()))
            .unwrap_or_default();
        let enabled = unsafe { reaper.track_fx_get_enabled(track, loc) };
        let offline = unsafe { reaper.track_fx_get_offline(track, loc) };
        fx.push(json!({ "index": i, "name": name, "enabled": enabled, "offline": offline }));
    }
    Ok(json!({ "track_index": track_index, "fx": fx }))
}

fn get_fx_params(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    limit: usize,
) -> Result<Value, String> {
    let track = reaper
        .get_track(ProjectContext::CurrentProject, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let loc = TrackFxLocation::NormalFxChain(fx_index);
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
            let position = unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Position) };
            let length = unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Length) };
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
        let name =
            read_string(NAME_BUF as usize, |b, s| unsafe { low.TakeFX_GetParamName(t, fx, p, b, s) })
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
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    reaper.undo_begin_block_2(project);
    // SAFETY: track valid, main thread.
    let result = unsafe {
        reaper.track_fx_add_by_name_add(
            track,
            fx_name,
            TrackFxChainType::NormalFxChain,
            AddFxBehavior::AlwaysAdd,
        )
    };
    reaper.undo_end_block_2(
        project,
        format!("AI: add FX \"{fx_name}\" to track {track_index}"),
        UndoScope::All,
    );
    match result {
        Ok(fx_index) => Ok(json!({
            "added": true, "track_index": track_index, "fx_index": fx_index, "name": fx_name
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
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let loc = TrackFxLocation::NormalFxChain(fx_index);
    let v = value.clamp(0.0, 1.0);
    reaper.undo_begin_block_2(project);
    let result = unsafe {
        reaper.track_fx_set_param_normalized(track, loc, param_index, ReaperNormalizedFxParamValue::new(v))
    };
    let display = unsafe { reaper.track_fx_get_formatted_param_value(track, loc, param_index, NAME_BUF) }
        .ok()
        .map(|s| reaper_string(s.as_c_str().to_bytes()))
        .unwrap_or_default();
    reaper.undo_end_block_2(
        project,
        format!("AI: set track {track_index} FX {fx_index} param {param_index} to {v:.4}"),
        UndoScope::All,
    );
    result
        .map(|_| json!({
            "set": true, "track_index": track_index, "fx_index": fx_index,
            "param_index": param_index, "normalized": v, "display_value": display
        }))
        .map_err(|_| "failed to set parameter".to_string())
}

fn set_fx_enabled(
    reaper: &Reaper<MainThreadScope>,
    track_index: u32,
    fx_index: u32,
    enabled: bool,
) -> Result<Value, String> {
    let project = ProjectContext::CurrentProject;
    let track = reaper
        .get_track(project, track_index)
        .ok_or_else(|| format!("no track at index {track_index}"))?;
    let loc = TrackFxLocation::NormalFxChain(fx_index);
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
    Ok(json!({ "track_index": track_index, "fx_index": fx_index, "enabled": enabled }))
}

// ---- undo / history ---------------------------------------------------------

fn undo_label(reaper: &Reaper<MainThreadScope>, project: ProjectContext) -> Option<String> {
    reaper.undo_can_undo_2(project, |s: &ReaperStr| reaper_string(s.as_c_str().to_bytes()))
}

fn redo_label(reaper: &Reaper<MainThreadScope>, project: ProjectContext) -> Option<String> {
    reaper.undo_can_redo_2(project, |s: &ReaperStr| reaper_string(s.as_c_str().to_bytes()))
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
                t, i, &mut selected, &mut muted, &mut sppq, &mut eppq, &mut chan, &mut pitch,
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
        let end = start + unsafe { reaper.get_media_item_info_value(item, ItemAttributeKey::Length) };
        let mut neighbors = Vec::new();
        if let Some(ti) = this_idx {
            let want: Vec<u32> = [ti.checked_sub(1), Some(ti + 1)].into_iter().flatten().collect();
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
                let os = unsafe { reaper.get_media_item_info_value(other, ItemAttributeKey::Position) };
                let oe = os + unsafe { reaper.get_media_item_info_value(other, ItemAttributeKey::Length) };
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
        let ok = unsafe { low.MIDI_InsertNote(t, false, false, sppq, eppq, chan, pitch, vel, &no_sort) };
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
        let dest = unsafe { reaper.get_track_send_info_desttrack(track, TrackSendDirection::Send, i) }
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
        let src = unsafe { reaper.get_track_send_info_desttrack(track, TrackSendDirection::Receive, i) }
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
    res.map(|idx| json!({
        "added": true, "src_track_index": src_track_index,
        "dest_track_index": dest_track_index, "send_index": idx
    }))
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
        other => return Err(format!("unknown send param '{other}' (use volume, pan, or mute)")),
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
    res.map(|_| json!({
        "set": true, "track_index": track_index, "send_index": send_index,
        "param": param, "value": value
    }))
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

fn get_track_envelopes(reaper: &Reaper<MainThreadScope>, track_index: u32) -> Result<Value, String> {
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
        let name = read_string(NAME_BUF as usize, |b, s| unsafe { low.GetEnvelopeName(env, b, s) })
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
            low.GetEnvelopePoint(env, i, &mut time, &mut value, &mut shape, &mut tension, &mut selected)
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
    let get = |i: c_int, key: &CStr| unsafe { low.GetSetAutomationItemInfo(env, i, key.as_ptr(), 0.0, false) };
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
    reaper.undo_end_block_2(project, "AI: update project notes".to_string(), UndoScope::All);
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

fn get_project_memory(reaper: &Reaper<MainThreadScope>, key: Option<&str>) -> Result<Value, String> {
    let low = reaper.low();
    match key {
        Some(k) => {
            let key_c = CString::new(k).map_err(|_| "invalid key".to_string())?;
            let value = read_string(NOTES_BUF, |b, s| unsafe {
                low.GetProjExtState(std::ptr::null_mut(), MEMORY_EXT.as_ptr(), key_c.as_ptr(), b, s) > 0
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
        low.SetProjExtState(std::ptr::null_mut(), MEMORY_EXT.as_ptr(), key_c.as_ptr(), c"".as_ptr());
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
        low.AddProjectMarker2(CUR_PROJ, false, position, 0.0, name_c.as_ptr(), want_index, 0)
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

fn get_stretch_markers(
    reaper: &Reaper<MainThreadScope>,
    item_index: u32,
) -> Result<Value, String> {
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

// ---- helpers ----------------------------------------------------------------

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
