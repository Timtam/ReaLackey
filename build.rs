// Build script for the ReaLackey.
//
// Responsibilities:
//   1. macOS/Linux: run swell_resgen (PHP) over cpp/assistant.rc to produce the
//      native SWELL dialog/menu resource TABLES (assistant.rc_mac_dlg /
//      _mac_menu). This MUST happen before (2) because the shim #includes those
//      tables so SWELL knows about our dialogs at runtime — without them
//      CreateDialogParam/DialogBoxParam return NULL and no window opens.
//   2. Compile the thin C++ UI shim (cpp/ui_shim.cpp) via the `cc` crate and
//      statically link it into the cdylib. On Windows it uses the real Win32
//      API; on macOS/Linux it uses SWELL (swell.h) + the tables from (1).
//   3. Windows: compile the Win32 dialog resource (cpp/assistant.rc) with the
//      MSVC resource compiler via `embed-resource` (cc cannot handle .rc).
//   4. macOS/Linux: compile the ONE SWELL "modstub" glue file (never the real
//      SWELL implementation — REAPER already provides it) with
//      -DSWELL_PROVIDED_BY_APP.
//
// The macOS/Linux paths require a WDL/swell checkout under vendor/WDL and PHP
// on PATH; they are stubbed here Windows-first and guarded by target_os so the
// Windows build has no such dependency.

use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // `webview` cfg: the wry-backed HTML conversation pane is available where wry
    // has a working backend we host as a child of the dialog — Windows (WebView2)
    // and macOS (WKWebView). Elsewhere the plain edit-control fallback is used.
    println!("cargo::rustc-check-cfg=cfg(webview)");
    if target_os == "windows" || target_os == "macos" {
        println!("cargo::rustc-cfg=webview");
    }

    // WDL checkout (only needed on non-Windows for swell.h + the modstub).
    let wdl = "vendor/WDL/WDL";
    let non_windows = target_os == "linux" || target_os == "macos";

    // --- (1) macOS/Linux: generate the SWELL dialog/menu tables from the .rc,
    //     BEFORE compiling the shim (which #includes them). ---------------------
    if non_windows {
        assert!(
            Path::new(&format!("{wdl}/swell")).exists(),
            "Non-Windows build needs a WDL checkout at vendor/WDL (git clone https://github.com/justinfrankel/WDL vendor/WDL)"
        );
        // swell_resgen.php writes cpp/assistant.rc_mac_dlg and _mac_menu next to
        // the .rc; ui_shim.cpp #includes them (see the bottom of that file).
        let status = std::process::Command::new("php")
            .arg(format!("{wdl}/swell/swell_resgen.php"))
            .arg("cpp/assistant.rc")
            .status()
            .expect("failed to run swell_resgen.php (is PHP installed?)");
        assert!(status.success(), "swell_resgen.php failed");
    }

    // --- (2) Always compile our own C++ shim ---------------------------------
    let mut shim = cc::Build::new();
    shim.cpp(true).file("cpp/ui_shim.cpp").include("cpp");
    if non_windows {
        // SWELL_PROVIDED_BY_APP makes swell.h declare the SWELL API as `extern "C"`
        // function pointers (resolved at load from the host). The shim MUST use the
        // same define as the modstub, or its calls get C++-mangled names that don't
        // match the modstub's C symbols → link errors. The swell include dir is also
        // where swell-dlggen.h / swell-menugen.h live (used by the generated tables).
        shim.include(format!("{wdl}/swell"))
            .define("SWELL_PROVIDED_BY_APP", None);
    }
    if target_os == "macos" {
        shim.cpp_set_stdlib("c++");
    }
    shim.compile("raai_ui_shim");

    // --- (3) Windows: compile the dialog resource ----------------------------
    if target_os == "windows" {
        // Links the compiled .res into the cdylib. Needs rc.exe (Windows SDK).
        let _ = embed_resource::compile("cpp/assistant.rc", embed_resource::NONE);
    }

    // --- (4) macOS/Linux: SWELL modstub --------------------------------------
    if non_windows {
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
    }

    println!("cargo:rerun-if-changed=cpp/ui_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/ui_shim.h");
    println!("cargo:rerun-if-changed=cpp/assistant.rc");
    println!("cargo:rerun-if-changed=cpp/resource.h");
    println!("cargo:rerun-if-changed=build.rs");
}
