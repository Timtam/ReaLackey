# Changelog

All notable changes to ReaLackey are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
for [Semantic Versioning](https://semver.org/).

Add new entries under `## [Unreleased]`. Triggering a release rolls that section
into a versioned heading and attaches its entries to the GitHub release — see
`.github/workflows/release.yml`.

## [Unreleased]

### Added

- Action & keyboard-shortcut tools: `search_actions` (find actions by name),
  `get_action_info` (an action's name, toggle state, and bound shortcuts),
  `run_action` (run any action by id or named command — a catch-all for anything
  without a dedicated tool), `delete_action_shortcut`, and `add_action_shortcut`
  (opens REAPER's key-assignment dialog, since the API can't bind a key directly).
- Time-selection tools: `get_time_selection`, `set_time_selection` (with an
  optional `seek` to move the edit cursor to the start), and
  `clear_time_selection`; the time selection is also reported by `get_transport`.
  The assistant can now mark a range and then analyse, render, or measure it (the
  analysis/render tools default to the time selection).

### Fixed

- macOS: keys pressed in the chat composer (e.g. arrow keys) could jump focus out
  to REAPER's main window, because REAPER consumed them as global shortcuts. The
  window now claims keystrokes whenever it is the front window.

## [0.2.0] - 2026-07-14

### Added

- Item-edge trimming (`set_item_edge`): move an item's left or right edge to an
  absolute time in one undo block — the left edge shifts the take's source offset
  so the audio content stays put.
- Over-time audio analysis (`analyze_audio_timeline`): a level envelope, silent
  regions, transient onsets, and single-frequency tracking across a passage —
  a time-series, not just one aggregate number.
- Video production support: `capture_view` can now snapshot REAPER's Video window
  (the processed frame, with video FX applied) so the assistant can see it; the
  Video processor's parameters and presets already work via the FX tools, and new
  `get_fx_config` / `get_track_state_chunk` / `set_track_state_chunk` tools reach
  its EEL code and other advanced RPP-level edits.
- Multiple API keys per provider, with automatic failover. A provider can hold an
  ordered list of keys (add / delete / move up / move down in the settings dialog);
  the top key is used and, on a quota or auth error, the assistant switches to the
  next key — announced in the chat pane and via the screen reader — until it finds
  one that works or all are exhausted. Useful when you have several keys for the
  same provider (e.g. Gemini's free tier).

### Removed

- The standalone "Set Anthropic API key" entry in the Extensions → ReaLackey
  menu. API keys are now managed entirely in the Providers dialog (per provider,
  as a key list), so the separate action was redundant.

### Fixed

- **macOS: the assistant windows never opened.** "Open window" and "Providers"
  in the Extensions menu did nothing (focus just returned to REAPER). The SWELL
  dialog-resource tables generated from `assistant.rc` weren't compiled into the
  extension, so the native dialogs couldn't be created; they now are.

## [0.1.0] - 2026-07-13

### Added

- Native REAPER extension with an AI assistant: a modeless chat window — an
  embedded HTML pane (WebView2 on Windows, WKWebView on macOS) with a native
  edit-control fallback on Linux — and an Extensions → ReaLackey menu.
- ~100 tools spanning tracks/FX (incl. input & monitoring FX chains and preset
  load), MIDI, sends/receives, automation envelopes (track, FX-parameter and
  send/receive), markers/regions, tempo map, stretch markers, render settings,
  item/take/track properties, grouping, copy/move/delete, and transport.
- Multi-provider support: Claude (native) plus any OpenAI-compatible endpoint —
  OpenAI, Gemini, Groq, OpenRouter, DeepSeek, xAI, Ollama, LM Studio, or custom —
  managed from a Providers dialog with model fetching and per-provider settings.
- Vision: capture and reason about custom plugin GUIs, and (opt-in) operate them
  with synthetic clicks/drags/typing when a control has no automatable parameter.
- Audio: pure-Rust DSP analysis (LUFS/LRA/true-peak, peak/RMS, clipping, spectral
  profile) of raw or processed post-FX audio, plus listening to a rendered clip
  on audio-capable models.
- Accessibility: OSARA announcements, screen-reader-aware prompting, keyboard
  flow, `role="status"` status line, and consent gates for any cloud upload.
- Per-project memory and notes stored in the `.rpp`.
- Portable config under REAPER's resource path; API keys in the OS credential
  store.

## [0.0.0] - 2026-07-13

- Pre-release development. This is the starting point of the changelog; the
  first tagged release will collect the entries above.
