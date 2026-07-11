# REAPER AI Assistant

A single-process native **REAPER extension** (Rust) that integrates large
language models into the DAW workflow: it knows the project state, answers
questions, and — in later phases — makes changes itself (instantiate plugins,
set parameters, write MIDI), analyses audio via DSP, and reads the screen
on-demand. Accessibility is designed in from the start via native controls and
**OSARA** screen-reader announcements.

> Working title — the project name is still open. The German design document is
> intentionally **not** part of this repository (it is gitignored).

## Status: Phase 0 (skeleton) — complete

The roadmap runs Phase 0 → Phase 8. Phase 0 is implemented and building:

| Piece | State |
|---|---|
| Loads as a REAPER extension (`reaper-rs`, `#[reaper_extension_plugin]`) | ✅ compiles/links |
| "Open Assistant" action registered (Actions list / shortcut) | ✅ |
| Modeless native dialog via a thin C++ shim (Win32 now; SWELL for mac/linux) | ✅ Windows |
| OSARA detected (`GetFunc("osara_outputMessage")`) with graceful degradation | ✅ |
| tokio worker thread + main-thread pump (channels) | ✅ |
| Native Anthropic Messages adapter, streaming SSE (`reqwest`+`serde`) | ✅ (parser unit-tested) |
| Hello-world round-trip with Claude, **no tools yet** | ✅ code complete |

Later phases add: reading context tools (P1), plugin advice (P2), safe
Undo-wrapped mutations (P3), MIDI composition (P4), the shared OpenAI-compatible
provider adapter (P5), audio analysis via JSFX probe + Rust DSP (P6), screen
vision (P7), docking + distribution (P8).

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
  tools/ dsp/       reserved for later phases
cpp/
  resource.h        control IDs (shared by rc.exe AND swell_resgen.php)
  assistant.rc      ONE Win32 DIALOGEX resource (all platforms)
  ui_shim.{h,cpp}   DialogProc + the C-ABI Rust drives
build.rs            compiles the C++ shim (cc) + the .rc (embed-resource)
```

**Threading rule:** the REAPER API and the dialog are main-thread-only. The
worker never touches them directly — it sends `UiEvent`s over a channel drained
by `ControlSurface::run()` on the main thread.

## Build

Requirements (Windows, the primary target so far):

- Rust (stable), `x86_64-pc-windows-msvc` toolchain.
- Visual Studio Build Tools (MSVC linker + the Windows SDK's `rc.exe`).
- `libclang` — bundled with Visual Studio; `.cargo/config.toml` points bindgen
  at it. Adjust that path if your VS install differs, or unset it if LLVM is on
  `PATH`.

```sh
cargo build            # -> target/debug/reaper_aiassistant.dll
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

1. Copy `reaper_aiassistant.dll` into REAPER's `UserPlugins` directory.
   (The file name **must** start with `reaper_` or REAPER won't load it.)
2. Start REAPER → the console prints "REAPER AI Assistant loaded." on load.
3. Set your Anthropic API key: **Extensions → REAPER AI Assistant → Set
   Anthropic API key** (also in the Actions list). It's stored in the OS
   credential store (Windows Credential Manager) and persists across restarts.
   Alternatively, set the `ANTHROPIC_API_KEY` environment variable. Optional:
   `RAAI_MODEL` (default `claude-opus-4-8`).
4. **Extensions → REAPER AI Assistant → Open window** (or the action) opens the
   dialog. Type a message, press **Send**; the reply streams into the log.

## Verified here vs. pending validation

**Verified in this environment:** clean `cargo build`/`clippy`, the crate links
to `reaper_aiassistant.dll`, the dialog resource is embedded, and the SSE parser
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
