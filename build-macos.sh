#!/usr/bin/env bash
# Local macOS build helper: produce a UNIVERSAL (Apple Silicon + Intel) dylib with
# the name REAPER expects (reaper_realackey.dylib), ad-hoc signed so it loads. A
# plain `cargo build` only makes a single-arch, lib-prefixed dylib for the host.
#
# This is the local counterpart to the release pipeline
# (.github/scripts/build-macos-universal.sh), which additionally Developer-ID
# signs + notarizes using repo secrets. Output: target/release/reaper_realackey.dylib
set -euo pipefail
cd "$(dirname "$0")"

ASSET="reaper_realackey.dylib"

# Prefer the rustup toolchain so a Homebrew (or other) rustc on PATH doesn't
# shadow it and break the target-specific builds.
if command -v rustup >/dev/null 2>&1; then
  export PATH="$(dirname "$(rustup which cargo)"):$PATH"
fi

# bindgen (via reaper-rs) needs libclang; point at it if the caller hasn't.
if [ -z "${LIBCLANG_PATH:-}" ]; then
  for d in /Library/Developer/CommandLineTools/usr/lib \
           /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib; do
    if [ -d "$d" ]; then export LIBCLANG_PATH="$d"; echo "LIBCLANG_PATH=$d"; break; fi
  done
fi

# build.rs needs a WDL/SWELL checkout and PHP (swell_resgen.php).
if [ ! -d "vendor/WDL/WDL/swell" ]; then
  echo "error: vendor/WDL not found — run:" >&2
  echo "  git clone https://github.com/justinfrankel/WDL vendor/WDL" >&2
  exit 1
fi
command -v php >/dev/null 2>&1 || echo "warning: php not on PATH; build.rs needs it for SWELL resgen" >&2

echo "Adding macOS targets…"
rustup target add aarch64-apple-darwin x86_64-apple-darwin

echo "Building aarch64 (Apple Silicon)…"
cargo build --release --target aarch64-apple-darwin
echo "Building x86_64 (Intel)…"
cargo build --release --target x86_64-apple-darwin

echo "Merging into a universal binary: target/release/$ASSET"
mkdir -p target/release
lipo -create -output "target/release/$ASSET" \
  target/aarch64-apple-darwin/release/libreaper_realackey.dylib \
  target/x86_64-apple-darwin/release/libreaper_realackey.dylib

# lipo drops the linker's ad-hoc signature and arm64 refuses to load an unsigned
# binary, so re-sign ad-hoc. (Not notarization — the release build does that.)
codesign --force --sign - "target/release/$ASSET"

echo "== architectures =="
lipo -info "target/release/$ASSET"
echo
echo "Built target/release/$ASSET — copy it into REAPER's UserPlugins folder."
