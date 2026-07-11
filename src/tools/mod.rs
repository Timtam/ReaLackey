//! Tool / function catalog (design §kap-tools).
//!
//! Phases 1–2: read-only context tools. Each tool executes on the REAPER main
//! thread (via [`crate::reaper::api`]) and returns JSON that is fed back to the
//! model as a `tool_result`. Track FX is available through reaper-medium; take
//! FX and installed-FX enumeration drop to the low-level API. Mutating tools
//! (Undo-wrapped, confirmation-gated) arrive in Phase 3.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
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
    opt_str(input, key).ok_or_else(|| format!("missing or invalid '{key}' (expected a non-empty string)"))
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
