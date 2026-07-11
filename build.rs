// Build script for the REAPER AI Assistant.
//
// Responsibilities:
//   1. Compile the thin C++ UI shim (cpp/ui_shim.cpp) via the `cc` crate and
//      statically link it into the cdylib. On Windows it uses the real Win32
//      API; on macOS/Linux it uses SWELL (swell.h).
//   2. Windows: compile the Win32 dialog resource (cpp/assistant.rc) with the
//      MSVC resource compiler via `embed-resource` (cc cannot handle .rc).
//   3. macOS/Linux: compile the ONE SWELL "modstub" glue file (never the real
//      SWELL implementation — REAPER already provides it) with
//      -DSWELL_PROVIDED_BY_APP, and run swell_resgen (PHP) over the .rc to
//      produce the native dialog/menu resource tables.
//
// The macOS/Linux paths require a WDL/swell checkout under vendor/WDL and PHP
// on PATH; they are stubbed here Windows-first and guarded by target_os so the
// Windows build has no such dependency.

use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // WDL checkout (only needed on non-Windows for swell.h + the modstub).
    let wdl = "vendor/WDL/WDL";

    // --- (1) Always compile our own C++ shim ---------------------------------
    let mut shim = cc::Build::new();
    shim.cpp(true).file("cpp/ui_shim.cpp").include("cpp");
    if target_os != "windows" {
        shim.include(format!("{wdl}/swell"));
    }
    if target_os == "macos" {
        shim.cpp_set_stdlib("c++");
    }
    shim.compile("raai_ui_shim");

    // --- (2) Windows: compile the dialog resource ----------------------------
    if target_os == "windows" {
        // Links the compiled .res into the cdylib. Needs rc.exe (Windows SDK).
        let _ = embed_resource::compile("cpp/assistant.rc", embed_resource::NONE);
    }

    // --- (3) macOS/Linux: SWELL modstub + resource generation ----------------
    if target_os == "linux" || target_os == "macos" {
        assert!(
            Path::new(&format!("{wdl}/swell")).exists(),
            "Non-Windows build needs a WDL checkout at vendor/WDL (git clone https://github.com/justinfrankel/WDL vendor/WDL)"
        );
        let mut sw = cc::Build::new();
        sw.cpp(true)
            .warnings(false)
            .define("SWELL_PROVIDED_BY_APP", None)
            .include(format!("{wdl}/swell"));
        if target_os == "macos" {
            sw.file(format!("{wdl}/swell/swell-modstub.mm"));
            sw.cpp_set_stdlib("c++");
            sw.flag("-x").flag("objective-c++");
            println!("cargo:rustc-link-lib=framework=AppKit");
        } else {
            sw.file(format!("{wdl}/swell/swell-modstub-generic.cpp"));
        }
        sw.compile("raai_swell");

        // Generate SWELL dialog/menu tables from the SAME .rc (needs PHP).
        let status = std::process::Command::new("php")
            .arg(format!("{wdl}/swell/swell_resgen.php"))
            .arg("cpp/assistant.rc")
            .status()
            .expect("failed to run swell_resgen.php (is PHP installed?)");
        assert!(status.success(), "swell_resgen.php failed");
    }

    println!("cargo:rerun-if-changed=cpp/ui_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/ui_shim.h");
    println!("cargo:rerun-if-changed=cpp/assistant.rc");
    println!("cargo:rerun-if-changed=cpp/resource.h");
    println!("cargo:rerun-if-changed=build.rs");
}
