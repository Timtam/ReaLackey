# Changelog

All notable changes to ReaLackey are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
for [Semantic Versioning](https://semver.org/).

Add new entries under `## [Unreleased]`. Triggering a release rolls that section
into a versioned heading and attaches its entries to the GitHub release — see
`.github/workflows/release.yml`.

## [Unreleased]

### Added

- Multiple API keys per provider, with automatic failover. A provider can hold an
  ordered list of keys (add / delete / move up / move down in the settings dialog);
  the top key is used and, on a quota or auth error, the assistant switches to the
  next key — announced in the chat pane and via the screen reader — until it finds
  one that works or all are exhausted. Useful when you have several keys for the
  same provider (e.g. Gemini's free tier).

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
