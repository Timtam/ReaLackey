//! UI: the Rust side of the C++/SWELL dialog shim.
//! `ffi` declares/wraps the C-ABI; `bridge` routes dialog callbacks to the worker.

pub mod bridge;
pub mod ffi;
pub mod output;
