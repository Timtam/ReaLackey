# Changelog

All notable changes to ReaLackey are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
for [Semantic Versioning](https://semver.org/).

Add new entries under `## [Unreleased]`. Triggering a release rolls that section
into a versioned heading and attaches its entries to the GitHub release — see
`.github/workflows/release.yml`.

## [Unreleased]

### Added

- **Speech-to-text (transcription)** — a new provider *type* alongside chat. The
  **Providers dialog is now tabbed by role** (Chat / Transcription); on the
  **Transcription** tab, **Add → "OpenAI Whisper"** (or **"Local Whisper server"**)
  and set it as the default, and the assistant can **transcribe a media item's audio
  to text** with timestamps — ask it to caption, summarise, or find spoken words in
  the selected item, then act on the text. Cloud transcription is consent-gated (the
  audio is uploaded); a local whisper server (whisper.cpp, faster-whisper, LocalAI)
  keeps it on-machine. Long items are transcribed in chunks automatically. `whisper-1`
  (or a local whisper model) returns timestamped segments; OpenAI's `gpt-4o-transcribe`
  returns text only. Each role has its own default account. (Dedicated one-click
  REAPER actions — transcript to item notes, plain text, and SRT — are coming next.)
- **Advanced mode (auto-approve edits)**: a toggle that lets the assistant apply
  changes **without asking for confirmation each time**. Flip it from **Extensions
  → ReaLackey → "Advanced mode (auto-approve edits)"** (the menu item shows the
  current on/off state) or bind the **"ReaLackey: Toggle advanced mode"** action to
  a key. Off by default; the state persists. The `RAAI_CONFIRM` environment
  variable still overrides it. (Edits remain undoable in REAPER, and each tool the
  assistant runs is still announced — you're just not prompted per change.)

### Fixed

- **Fetch models** now authenticates with the **top (highest-priority) key from the
  key list** — the one that would actually be used to send — instead of the "Add
  key" input field or the stale saved key. Reordering or editing the key list and
  then fetching (before saving) used the wrong key; typing into the Add-key field no
  longer overrides the configured list either. (Falls back to a just-typed key only
  when the list is empty, e.g. a brand-new account before you press Add.)
- Screen-reader announcements were spoken **twice** (most noticeably under
  VoiceOver): the assistant announced through **both** OSARA and the webview's
  aria-live region, so a reader observing both channels heard everything doubled.
  It now speaks through **one** channel — OSARA when it's present (focus-independent,
  both platforms), and the aria-live region only as the fallback when OSARA isn't —
  so each announcement is spoken once.
- The message field no longer makes a screen reader recite a long instruction
  string **every time it gets focus**. Its accessible name is now just "Message
  the assistant" and the placeholder is a short prompt; the Enter / Alt+number /
  Alt+P shortcuts moved into an **optional, collapsed "Keyboard shortcuts"**
  section under the composer. Nothing is announced when you focus the field —
  previously the full instructions lived in the field's accessible name and
  placeholder, so VoiceOver (and NVDA) re-read them on every focus.
- macOS: **Alt+number message reading** is now reliable. Jumping to a message
  (Alt+1 … Alt+0) read it only intermittently on the Mac because it relied on
  moving focus to the message heading, and VoiceOver only sometimes announces a
  programmatic focus change on the WebKit view. The read now goes through the
  same aria-live channel the copy confirmation uses — reliable on VoiceOver and
  NVDA alike — and it announces the message's actual text, so navigating to an
  assistant message now reads the response rather than just "Assistant".

### Added

- **Perplexity (Agent API)** provider (Add → "Perplexity (Agent API, web-grounded)").
  Unlike Perplexity's plain Sonar endpoint — which can't call tools and so can't
  drive REAPER — the Agent API speaks the OpenAI **Responses** protocol with
  client-side function calling, so the assistant can control REAPER *and* ground
  its answers on live web results (`web_search`) in one loop. It's multi-provider:
  pick a strong agentic model by id (`openai/gpt-5.1`, `anthropic/claude-…`, or
  `sonar-…`) — the Model field is free-text (the Agent API has no model list to
  fetch). Fixed endpoint, needs a Perplexity API key; web grounding is always on.
- **oMLX** provider preset (Add → "oMLX (local, Apple Silicon)"). oMLX is a native
  MLX inference server for Apple Silicon (continuous batching, SSD KV cache) that's
  faster than Ollama on a Mac. It exposes an OpenAI-compatible endpoint, so it uses
  the existing adapter — the preset just points at `http://localhost:8000/v1`; pick
  your model with **Fetch models…**. (Any oMLX instance already worked via the
  generic "OpenAI-compatible" provider; this is just one-click setup.)

## [0.3.1] - 2026-07-17

### Fixed

- The **Gemini** provider preset now defaults to `gemini-3.5-flash`. The previous
  default, `gemini-2.0-flash`, has been retired by Google, so adding a Gemini
  provider seeded a model that no longer exists. `gemini-3.5-flash` is the current
  latest stable Flash model (multimodal, free-tier accessible). You can still pick
  another model with **Fetch models…** — e.g. `gemini-2.5-flash` or
  `gemini-2.5-flash-lite` for higher free-tier throughput.
- OpenAI-compatible providers: newer OpenAI models (**GPT-5**, the **o-series**)
  no longer fail with `Unsupported parameter: 'max_tokens'`. Those models require
  `max_completion_tokens` instead of `max_tokens`; the adapter now sends the right
  field for `api.openai.com`, and for any other endpoint that needs it, it retries
  once transparently and remembers the choice for the rest of the session. Servers
  that only understand `max_tokens` (Ollama, LM Studio, DeepSeek, Groq, …) are
  unaffected.

## [0.3.0] - 2026-07-16

### Added

- Reasoning display: models that expose their chain-of-thought — DeepSeek-R1,
  Qwen3, and Ollama "thinking" models over the OpenAI-compatible endpoint (via
  `reasoning_content`) — now stream it into a collapsible **Reasoning** block above
  the answer. It's shown separately from the reply and is not spoken as the final
  answer.
- Anthropic **extended thinking**, as a per-provider toggle ("Extended thinking
  (reasoning)" in the provider settings dialog, Anthropic only). When on, the
  request asks for adaptive thinking — the model reasons more on hard tasks, less
  on easy ones — and streams that reasoning into the same collapsible **Reasoning**
  block as above. The thinking blocks are stored and replayed verbatim (with their
  signatures) so multi-step tool-use conversations keep their reasoning continuity;
  toggling thinking back off cleanly drops them from the ongoing conversation.
  Requires a model that supports adaptive thinking (Claude Opus 4.8 / 4.7 / 4.6,
  Sonnet 4.6+); older models reject it.
- Lower per-request token usage (helps free-tier keys, which meter tokens/minute
  and re-charge the whole tool list + prompt every agentic turn):
  - **Trimmed** the tool descriptions and system prompt (deduped boilerplate now
    stated once) — ~12% off the static overhead, no behaviour change.
  - **Media eviction**: past screenshots/audio/video-clip frames were re-uploaded
    every later turn; now only the most recent captures stay live (older ones
    become a placeholder). Tunable via `RAAI_MEDIA_KEEP` (default 2).
  - **Prompt caching** on the Anthropic path (`cache_control` on the tools+system
    prefix): repeated turns re-read it as a rate-limit-free cache read.
    `RAAI_PROMPT_CACHE=off` to disable.
  - **Progressive tool disclosure** (opt-in `RAAI_PROGRESSIVE_TOOLS=on`): only a
    core set + a `load_tools(query)` loader are sent; the model pulls in the rest
    by capability on demand (session-persisted, with keyword pre-loading). Cuts the
    tool payload ~70–90% for the turn — aimed at free tiers like Gemini.
- Video-clip vision (`capture_video_clip`): instead of a single screenshot of
  REAPER's Video window, the assistant can grab **several frames across a time
  range** (stepping the edit cursor, playback stopped) and — for audio-capable
  models — the **clip's audio**, so it can reason about motion, cuts, transitions,
  on-screen text timing, and A/V sync. One consent covers the whole clip. Range
  defaults to the time selection; frame count defaults to 6 (2–12). The
  seek-to-frame settle delay is tunable via `RAAI_VIDEO_SETTLE_MS` (default 250 ms)
  if a heavy video-FX chain needs longer to re-render.
- Per-provider **"Supports audio (listening)"** checkbox in the provider settings
  dialog, next to "Supports images". Audio input was previously auto-detected from
  the model id only, so a locally-run multimodal model (e.g. Google **Gemma**
  3n/4 via Ollama or LM Studio) had no way to enable listening; now you can toggle
  it explicitly. Gemma is also recognized as vision-capable by default. (Whether a
  given local server accepts audio input is up to the server/model.)
- Prompt presets: save reusable prompts and drop them into the chat composer
  instead of retyping. Manage them (add / edit / delete) from **Extensions →
  ReaLackey → "Prompt presets…"**; in the chat window the **Presets** button (or
  **Alt+P**) opens a picker that inserts the chosen prompt into the composer so
  you can tweak it before sending. Stored globally in `presets.json`, next to
  `providers.json`.
- Chat message navigation + copy: **Alt+1 … Alt+0** (and the key right of 0) jump
  to that message and move focus to it (the screen reader reads it); a quick second
  press of the same combo copies that message — your request or the model's
  response — to the clipboard.
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

- The assistant no longer finishes **silently** when the model returns an empty
  response (no answer and no tool call). Previously the status just went to
  "Ready." with nothing shown — which reads as a crash — most visibly with local
  models (Ollama) that don't reliably do tool use. It now says the response was
  empty and hints at likely causes.
- macOS: the release `.dylib` is now a **universal binary (Apple Silicon + Intel)**.
  It previously shipped arm64-only, so it silently failed to load in REAPER on
  Intel Macs or under Rosetta.
- macOS: the release is now **Developer-ID-signed and Apple-notarized** (when the
  repo's signing secrets are configured — see `docs/macos-notarization.md`), so
  it loads without the manual quarantine step. Without the secrets it falls back
  to an ad-hoc signature (users then run `xattr -dr com.apple.quarantine`).
- macOS: keyboard handling in the chat window. Keys typed in the composer could be
  swallowed by REAPER — arrow keys jumping focus to the arrange, and Cmd+C/V/X not
  reaching the text field (they hit REAPER's Edit menu). When the window is in
  front, its keystrokes are now handed to the webview's native editing
  (copy/paste/cut/select-all, arrow keys, typing) instead of REAPER's global
  shortcuts/menu.

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
