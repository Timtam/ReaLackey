//! AI core: the provider-agnostic worker that runs the agent loop off the main
//! thread and the message protocol between it and the REAPER main thread.

pub mod protocol;
pub mod worker;
