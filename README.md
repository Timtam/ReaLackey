# ReaLackey

A single-process native **REAPER extension** (Rust) that integrates large
language models into the DAW workflow: it knows the project state, answers
questions, and — in later phases — makes changes itself (instantiate plugins,
set parameters, write MIDI), analyses audio via DSP, and reads the screen
on-demand. Accessibility is designed in from the start via native controls and
**OSARA** screen-reader announcements.

> Working title — the project name is still open. The German design document is
> intentionally **not** part of this repository (it is gitignored).

## Status: Phase 0 (skeleton) — complete

The roadmap runs Phase 0 → Phase 8. Phases 0–1 are implemented and building:

| Piece | State |
|---|---|
| Loads as a REAPER extension (`reaper-rs`, `#[reaper_extension_plugin]`) | ✅ compiles/links |
| Actions + Extensions submenu ("Open window", "Set Anthropic API key") | ✅ |
| Modeless native dialog via a thin C++ shim (Win32 now; SWELL for mac/linux) | ✅ Windows |
| OSARA detected (`GetFunc("osara_outputMessage")`) with graceful degradation | ✅ |
| tokio worker thread + main-thread pump (channels) | ✅ |
| Native Anthropic Messages adapter, streaming SSE (`reqwest`+`serde`) | ✅ (parser unit-tested) |
| API key entry (native input box) + OS credential-store storage | ✅ |
| **Phase 1:** tool-use agent loop | ✅ code complete |
| **Phase 2:** plugin/FX + item read tools (incl. take FX) | ✅ code complete |
| **Phase 3:** safe mutations (confirm + Undo) + undo tools | ✅ code complete |
| **Phase 4:** MIDI composition + track routing | ✅ code complete |
| **+ Automation:** envelopes read/write + automation items | ✅ code complete |
| **+ Arrangement:** markers/regions, tempo map, stretch markers, render settings | ✅ code complete |
| **+ Editing:** item/take/track properties, grouping, copy/move/delete | ✅ code complete |
| **Phase 6:** DSP audio analysis (loudness/spectral) via audio accessors | ✅ code complete |

**Read tools** (main thread, results fed back to the model): `get_project_summary`,
`get_tracks`, `get_track_fx`, `get_fx_params`, `get_selected_items`,
`get_take_fx`, `get_take_fx_params`, `list_installed_fx`, `get_focused_fx`,
`get_take_midi` (notes, optionally incl. neighbouring items), `get_track_sends`,
`get_track_envelopes`, `get_envelope_points`, `get_automation_items`,
`get_project_notes`, `get_track_notes`, `get_project_memory`,
`get_markers`, `get_tempo_markers`, `get_stretch_markers`, `get_render_settings`,
`get_item_properties`, `get_take_properties`, `get_track_properties`,
`get_track_group_membership`, `analyze_item_audio`, `analyze_track_audio`,
`analyze_processed_audio`.

**Per-project notes & memory:** the assistant can read/append the project's Notes
and per-track notes (undo-wrapped), and keeps a **persistent per-project memory**
(`get_project_memory` / `set_project_memory` / `delete_project_memory`) stored in
the `.rpp` via project ext-state — a keyed scratchpad it uses to remember
decisions, TODOs, and progress across sessions. Memory writes are metadata (not
confirmation-gated), so the assistant can track progress freely.

**Mutating tools** — FX: `add_fx`, `set_fx_param`, `set_fx_enabled`; MIDI:
`insert_midi_notes` (quarter-note timing), `create_midi_item`; routing:
`add_send`, `set_send_param`, `remove_send`; automation: `insert_envelope_point`;
markers/regions: `add_marker`, `add_region`, `delete_marker`; tempo map:
`add_tempo_marker`, `delete_tempo_marker`, `set_project_tempo`; stretch markers:
`add_stretch_marker`, `delete_stretch_marker`; render: `set_render_setting`;
item/take/track editing: `set_item_property`, `set_take_property`,
`set_active_take`, `set_track_property`, `set_track_group_membership`,
`group_items`; arrangement edits: `copy_item`, `move_item`, `delete_item`,
`copy_take`, `duplicate_track`, `delete_track`.
Every change is shown to the user
for confirmation (a native, screen-reader-accessible Yes/No box) and wrapped in
a **labelled Undo block** (`AI: …`) so both the user and the assistant can revert
it. Confirmation is on by default (`RAAI_CONFIRM=off` to disable). The assistant
can also `undo`/`redo`, and `get_undo_history` returns the next undo/redo labels
plus a rolling log of recent actions (sampled from `Undo_CanUndo2`) so it can
comment on the user's workflow.

**Audio analysis (Phase 6):** `analyze_item_audio` / `analyze_track_audio` read
samples through a REAPER audio accessor (main thread) and run a pure-Rust DSP
pass (`src/dsp/`) — sample peak & RMS (dBFS), crest factor, DC offset, clipping,
**integrated loudness (LUFS, ITU-R BS.1770 gated)**, and a rough spectral profile
(centroid, dominant frequency, low/mid/high balance) via a hand-rolled FFT. The
DSP is host-independent and **unit-tested** (FFT vs. DFT, loudness, spectral
centroid). Note the accessor returns the **pre-FX, pre-fader source audio**, and
reads are capped at 30 s.

**Processed (post-FX) audio** — `analyze_processed_audio` returns the same metrics
for audio *with* the FX applied, via a short offline render: `target: "master"`
renders the full mix (all track FX + master FX); `target: "track"` renders one
track through its FX and the master; `target: "item"` renders one item through
its take FX and its track's FX (no master). It forces a temp WAV (`RENDER_FORMAT="evaw"`),
reads the exact output path from `RENDER_TARGETS`, renders with the "most recent
render settings" action, decodes the WAV in Rust (a unit-tested parser), analyses
it, deletes the temp file, and **saves/restores every render setting and the
track selection** it touches. Capped at 30 s.

Later: shared OpenAI-compatible provider adapter (P5), the real-time JSFX
measurement-probe path (the second Phase-6 mechanism, for live/streaming metering),
screen vision (P7), docking + distribution (P8).

## Architecture

Everything lives in one address space (no IPC, no local server):

```
src/
  lib.rs            entry point (#[reaper_extension_plugin]) + FFI panic guard
  app.rs            wiring: channels, worker, pump + action registration
  reaper/           MAIN THREAD: REAPER API + OSARA + dialog
    action.rs         "Open Assistant" command registration
    control_surface.rs  run() ~30x/s drains UiEvents -> dialog/OSARA
    osara.rs          osara_outputMessage resolution + announce
  ai/               WORKER THREAD (tokio)
    protocol.rs       MainTask / UiEvent channel messages
    worker.rs         agent loop; maps provider events -> UiEvents
  providers/        LLM adapters
    mod.rs            LlmProvider trait + Capabilities
    anthropic/        native Messages API adapter (types.rs, stream.rs SSE parser)
  ui/               Rust side of the C++ shim
    ffi.rs            extern "C" decls + panic-guarded callback thunks
    bridge.rs         routes dialog callbacks -> worker
    output.rs         HTML conversation pane: embeds a WebView2 (wry) as a child
                      of the dialog and renders markdown->HTML + collapsible tool
                      cards; falls back to the plain edit control if unavailable
  text.rs           markdown -> clean prose (OSARA) and markdown -> HTML (pane)
  tools/            tool/function catalog the model drives (read + mutating)
  dsp/              pure-Rust audio feature extraction (loudness, spectral)
cpp/
  resource.h        control IDs (shared by rc.exe AND swell_resgen.php)
  assistant.rc      ONE Win32 DIALOGEX resource (all platforms)
  ui_shim.{h,cpp}   DialogProc + the C-ABI Rust drives
build.rs            compiles the C++ shim (cc) + the .rc (embed-resource)
```

**Threading rule:** the REAPER API and the dialog are main-thread-only. The
worker never touches them directly — it sends structured `UiEvent`s (user
message, assistant delta, tool started/finished, notice, status, …) over a
channel drained by `ControlSurface::run()` on the main thread, which renders
them into the HTML pane (or the edit-control fallback). The embedded WebView2 is
`!Send` and lives in a main-thread `thread_local`; it needs the Edge **WebView2
runtime** (present on current Windows), and gracefully falls back to the plain
edit control if that runtime or hosting is unavailable. The final answer is also
spoken via OSARA with markdown stripped to plain prose.

## Build

Requirements (Windows, the primary target so far):

- Rust (stable), `x86_64-pc-windows-msvc` toolchain.
- Visual Studio Build Tools (MSVC linker + the Windows SDK's `rc.exe`).
- `libclang` — bundled with Visual Studio; `.cargo/config.toml` points bindgen
  at it. Adjust that path if your VS install differs, or unset it if LLVM is on
  `PATH`.

```sh
cargo build            # -> target/debug/reaper_realackey.dll
cargo test             # runs the SSE parser unit tests
cargo build --release  # optimized (LTO)
```

`reaper-rs` is pulled from **git** (crates.io is frozen at a 2020 `0.1.0`); all
four crates are pinned to the same rev in `Cargo.toml`.

### macOS / Linux (not yet built here)

The SWELL path is stubbed in `build.rs`. It additionally needs a WDL checkout at
`vendor/WDL` (`git clone https://github.com/justinfrankel/WDL vendor/WDL`) and
**PHP** on `PATH` (for `swell_resgen.php`).

## Install & run

1. Copy `reaper_realackey.dll` into REAPER's `UserPlugins` directory.
   (The file name **must** start with `reaper_` or REAPER won't load it.)
2. Start REAPER → the console prints "ReaLackey loaded." on load.
3. Set your Anthropic API key: **Extensions → ReaLackey → Set
   Anthropic API key** (also in the Actions list). It's stored in the OS
   credential store (Windows Credential Manager) and persists across restarts.
   Alternatively, set the `ANTHROPIC_API_KEY` environment variable. Optional:
   `RAAI_MODEL` (default `claude-opus-4-8`).
4. **Extensions → ReaLackey → Open window** (or the action) opens the
   dialog. Type a message, press **Send**; the reply streams into the log.

## Verified here vs. pending validation

**Verified in this environment:** clean `cargo build`/`clippy`, the crate links
to `reaper_realackey.dll`, the dialog resource is embedded, and the SSE parser
passes unit tests (including byte-split streams and unknown events).

**Needs a real REAPER host + screen reader (open items from the design):**

- Does the extension load and does the action appear / open the dialog?
- Does `osara_outputMessage` speak when a native control has focus? (design a11y
  validation task a)
- Windows text is Unicode-correct (UTF-8↔UTF-16); confirm German glyphs render.
- Full keyboard flow (Enter-to-send) may need REAPER's accelerator hook —
  currently the **Send** button works; Enter routing is a follow-up.
- macOS/VoiceOver + SWELL (design a11y validation task b).

## License

MIT OR Apache-2.0.
