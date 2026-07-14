//! Prompt presets: reusable prompt bodies the user saves and inserts into the
//! chat composer for repetitive tasks. Stored globally (not per-project) in
//! `presets.json`, next to `providers.json`. The bodies are plain user text, so
//! (unlike API keys) they live in the JSON file, never the credential store.

pub mod registry;
