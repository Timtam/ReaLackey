# ReaLackey

**Your AI lackey inside REAPER.** ReaLackey is a native REAPER extension (written
in Rust) that puts a capable AI assistant right in your DAW. Ask it about your
project, have it explain what's going on, or just tell it what you want done —
add an EQ, write a MIDI bassline, balance your levels, tidy up markers — and it
does the work, every change wrapped in a labelled undo block you can revert.

It was built **accessibility-first**, for producers who work with a screen
reader (NVDA + OSARA) — but it's just as useful with your eyes open. When a
plugin's GUI can't be read by a screen reader, ReaLackey can *look at it for
you* and operate it.

> One process, no servers, no copy-paste. It talks to the model over the network
> and drives REAPER directly through its API.

---

## What it can do

ReaLackey drives REAPER through ~100 tools. It reads the project to answer
questions accurately (rather than guessing), and makes changes on your behalf —
each one confirmed and undoable. Highlights:

- **Tracks, FX & routing** — list and read tracks, FX and their parameters; add,
  remove, bypass and configure FX (including a track's **input** and the master's
  **monitoring** FX chains); read and load FX **presets**; manage sends/receives;
  set volume/pan, arm/mute/solo.
- **MIDI & composition** — read a take's notes (with neighbouring context), write
  and delete notes, create MIDI items.
- **Automation** — create envelopes (FX-parameter, track volume/pan/mute, and
  send/receive) and write, edit, or clear their points.
- **Arrangement & editing** — markers and regions, the tempo/time-signature map,
  take stretch markers, render settings; item/take/track properties, grouping,
  and copy/move/delete/duplicate.
- **Transport & session** — play/stop/record, move the edit cursor, change the
  playback rate and ruler unit, toggle metronome/snap/ripple.
- **Listen & measure** — pure-Rust DSP analysis of a take or track: loudness
  (integrated **LUFS**, LRA, true-peak), peak/RMS, clipping and a spectral
  profile — for the raw source *or* the **processed** post-FX signal. On
  audio-capable models it can even **listen** to a rendered clip to judge tone.
- **See inaccessible GUIs** — for custom-drawn plugin interfaces a screen reader
  can't parse, ReaLackey takes a screenshot, reasons about it, and (with your
  one-time consent) clicks/drags/types directly in the plugin window. It always
  prefers the undoable parameter API and only falls back to pixel input for
  controls that have no automatable parameter (e.g. a Kontakt patch switch).
- **Per-project memory** — a scratchpad stored in the `.rpp` so it remembers
  decisions, TODOs and progress across sessions, plus read/append access to the
  project and per-track notes.

## Bring your own model

ReaLackey speaks to Claude natively and to everything else through the
OpenAI-compatible API, so you can pick whatever suits your budget and privacy:

**Claude (Anthropic)** · **OpenAI** · **Google Gemini** · **Groq** ·
**OpenRouter** · **DeepSeek** · **xAI (Grok)** · **Ollama** (local) ·
**LM Studio** (local) · or any **custom** OpenAI-compatible endpoint.

Manage accounts in **Extensions → ReaLackey → Providers**: add/edit/delete,
pick a default, fetch the model list from the provider, and set per-provider
options (model, max tokens, tool-step limit, vision). Vision and audio tools are
offered only when the selected model actually supports them. Running a local
model (Ollama/LM Studio) means **no rate limits and no cost**.

## Accessibility

ReaLackey is designed to be driven entirely by keyboard and screen reader:

- Final answers are spoken through **OSARA**; the chat pane is navigable by
  headings, links open in your browser, and the status line carries a
  `role="status"` so you can query it on demand.
- When OSARA is running, the assistant is told it's talking to a blind user and
  avoids "look for the cog icon" style directions — it reads controls out or
  operates them itself.
- Every destructive action asks first, via a native, screen-reader-accessible
  Yes/No box.

## Install

1. Download the plug-in for your platform from the
   [latest release](../../releases/latest) — `reaper_realackey.dll` on Windows,
   or `reaper_realackey.dylib` on macOS *(experimental — it builds, but hasn't
   been validated in a live host yet)*. Or build it yourself (below).
2. Copy it into REAPER's `UserPlugins` folder. *(Find it via Options → Show
   REAPER resource path.)* The filename **must** start with `reaper_`.
3. Restart REAPER — ReaLackey loads silently (no console window, and no
   screen-reader chatter over REAPER's own launch feedback).
4. **Extensions → ReaLackey → Providers** → add a provider and paste your API
   key (stored securely in your OS credential store, never in a file).
5. **Extensions → ReaLackey → Open window** — type a message, press **Send**, and
   the reply streams into the chat.

## Configuration

- **Config is portable.** Your provider list lives under REAPER's *resource path*
  (`…/ReaLackey/providers.json`), so a portable REAPER install carries it along.
- **API keys** live in the OS credential store (Windows Credential Manager /
  macOS Keychain / Linux Secret Service) — never in plain text.
- **Environment overrides** (all optional): `RAAI_CONFIRM=off` disables the
  change-confirmation prompt; `RAAI_MAX_TURNS=N` overrides the per-provider
  tool-step limit; `RAAI_MODEL` sets the default Claude model.

## Safety model

- **Mutations are confirmed and undoable.** Every change is shown for approval
  and wrapped in a labelled `AI: …` undo block, so you and the assistant can both
  revert it.
- **Uploads are consent-gated.** Sending a screenshot or an audio clip to the
  cloud provider always asks first, every time.
- **Pixel control is opt-in.** Synthetic clicks/drags into plugin GUIs (which
  REAPER can't undo) require a one-time per-session approval; "close the window"
  is a hard kill switch that disarms it.

## Build from source

**Windows** (the primary, tested target):

```sh
cargo build            # -> target/debug/reaper_realackey.dll
cargo test
cargo build --release  # optimized
```

You'll need the Rust `x86_64-pc-windows-msvc` toolchain, the Visual Studio Build
Tools (MSVC linker + the Windows SDK's `rc.exe`), and `libclang` (bundled with
Visual Studio — `.cargo/config.toml` points bindgen at it; adjust if your VS
install differs). `reaper-rs` is pulled from git and pinned to one rev.

**macOS / Linux** (via SWELL): additionally needs a WDL checkout at `vendor/WDL`
(`git clone https://github.com/justinfrankel/WDL vendor/WDL`) and **PHP** on
`PATH` (for `swell_resgen.php`). The macOS build is compiled and linked in CI on
every push.

## Platform status

| Platform | Build | Runtime |
|---|---|---|
| **Windows** | ✅ CI | ✅ used daily |
| **macOS** | ✅ CI (compiles + links) | ⏳ not yet validated in a host |
| **Linux** | 🚧 SWELL path present | ⏳ not yet validated |

The chat pane is an embedded HTML view — **WebView2** on Windows, **WKWebView**
on macOS — falling back to a native edit control on Linux. Screen capture and
synthetic input have Windows (GDI / SendInput) and macOS (Core Graphics /
CGEvent) backends. The macOS backends and webview compile and link in CI but
have not yet been exercised in a live REAPER host.

## Contributing

Issues and PRs welcome. The codebase is English-only. Changelog entries go under
`## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) using
[Keep a Changelog](https://keepachangelog.com/) style; releases roll that section
into a version automatically (see `.github/workflows/release.yml`).

## License

MIT OR Apache-2.0.
