//! UI: the Rust side of the C++/SWELL dialog shim.
//! `ffi` declares/wraps the C-ABI; `bridge` routes dialog callbacks to the worker.

pub mod bridge;
pub mod ffi;
pub mod input;
pub mod output;
pub mod presets_ui;
pub mod providers_ui;
pub mod screenshot;
