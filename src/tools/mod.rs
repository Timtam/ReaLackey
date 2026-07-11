//! Tool / function catalog (design §kap-tools).
//!
//! Phase 1: read-only context tools. Each tool executes on the REAPER main
//! thread (via [`crate::reaper::api`]) and returns a JSON string that is fed
//! back to the model as a `tool_result`. Mutating tools (Undo-wrapped,
//! confirmation-gated) arrive in Phase 3.

use std::collections::HashSet;

use reaper_medium::{
    MainThreadScope, MasterTrackBehavior, ProjectContext, Reaper,
};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::providers::ToolDef;

/// A request from the worker to run a tool on the main thread.
pub struct ReaperOp {
    pub name: String,
    pub input: Value,
    pub reply: oneshot::Sender<ToolOutcome>,
}

/// The result of running a tool.
pub struct ToolOutcome {
    pub content: String,
    pub is_error: bool,
}

/// Tool definitions advertised to the model.
pub fn definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "get_project_summary".into(),
            description: "Get a lightweight snapshot of the current REAPER project: \
                          tempo (BPM), total track count, number of selected tracks and \
                          items, and the edit cursor position in seconds."
                .into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "get_tracks".into(),
            description: "List all tracks in the current project with their 0-based index, \
                          name, and whether they are currently selected."
                .into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
    ]
}

/// Execute a tool by name on the main thread. Never panics: unknown tools and
/// errors are returned as error outcomes.
pub fn execute(reaper: &Reaper<MainThreadScope>, name: &str, _input: &Value) -> ToolOutcome {
    let result: Result<Value, String> = match name {
        "get_project_summary" => Ok(get_project_summary(reaper)),
        "get_tracks" => Ok(get_tracks(reaper)),
        other => Err(format!("unknown tool: {other}")),
    };
    match result {
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

fn get_project_summary(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let tempo = reaper.master_get_tempo().get();
    let track_count = reaper.count_tracks(project);
    let selected_tracks =
        reaper.count_selected_tracks_2(project, MasterTrackBehavior::ExcludeMasterTrack);
    let selected_items = reaper.count_selected_media_items(project);
    let cursor = reaper
        .get_cursor_position_ex(project)
        .map(|p| p.get())
        .unwrap_or(0.0);
    json!({
        "tempo": tempo,
        "track_count": track_count,
        "selected_tracks": selected_tracks,
        "selected_items": selected_items,
        "edit_cursor_seconds": cursor,
    })
}

fn get_tracks(reaper: &Reaper<MainThreadScope>) -> Value {
    let project = ProjectContext::CurrentProject;
    let count = reaper.count_tracks(project);

    // Build the set of selected tracks so each track can be flagged.
    let sel_count =
        reaper.count_selected_tracks_2(project, MasterTrackBehavior::ExcludeMasterTrack);
    let mut selected = HashSet::new();
    for i in 0..sel_count {
        if let Some(t) =
            reaper.get_selected_track_2(project, i, MasterTrackBehavior::ExcludeMasterTrack)
        {
            selected.insert(t);
        }
    }

    let mut tracks = Vec::new();
    for i in 0..count {
        if let Some(t) = reaper.get_track(project, i) {
            // SAFETY: `t` was just obtained from REAPER and is used only on the
            // main thread within this call.
            let name = unsafe {
                reaper.get_set_media_track_info_get_name(t, |s| {
                    String::from_utf8_lossy(s.as_c_str().to_bytes()).into_owned()
                })
            }
            .unwrap_or_default();
            tracks.push(json!({
                "index": i,
                "name": name,
                "selected": selected.contains(&t),
            }));
        }
    }
    json!({ "tracks": tracks })
}
