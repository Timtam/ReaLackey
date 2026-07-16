#!/bin/bash
# Local build script for ReaLackey on macOS to compile a universal dylib
# with the correct name expected by REAPER.
set -e

# Make sure we are in the script's directory
cd "$(dirname "$0")"

ASSET="reaper_realackey.dylib"
PROFILE="release"
CARGO_FLAGS="--release"

# Resolve rustup toolchain path to bypass any conflicting global package-manager (e.g. Homebrew) rustc installs
if command -v rustup &> /dev/null; then
    echo "Resolving rustup toolchain..."
    export RUSTC="$(rustup which rustc)"
    CARGO="$(rustup which cargo)"
    echo "Using RUSTC=$RUSTC"
    echo "Using CARGO=$CARGO"
else
    echo "Warning: rustup not found. Falling back to default system cargo."
    CARGO="cargo"
fi

echo "Adding macOS cross-compilation targets..."
rustup target add x86_64-apple-darwin aarch64-apple-darwin

# Check for PHP (needed by build.rs for SWELL resgen)
if ! command -v php &> /dev/null; then
    echo "Warning: php could not be found on PATH. If build.rs fails, make sure PHP is installed."
fi

# Ensure WDL submodule is checked out
if [ ! -d "vendor/WDL/WDL/swell" ]; then
    echo "Error: vendor/WDL directory not found or empty."
    echo "Please run: git clone https://github.com/justinfrankel/WDL vendor/WDL"
    exit 1
fi

# Set LIBCLANG_PATH to Xcode's CommandLineTools if not already set (needed by bindgen)
if [ -z "$LIBCLANG_PATH" ]; then
    if [ -d "/Library/Developer/CommandLineTools/usr/lib" ]; then
        export LIBCLANG_PATH="/Library/Developer/CommandLineTools/usr/lib"
    elif [ -d "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib" ]; then
        export LIBCLANG_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib"
    fi
    if [ -n "$LIBCLANG_PATH" ]; then
        echo "Setting LIBCLANG_PATH to $LIBCLANG_PATH"
    fi
fi

echo "Building aarch64 (Apple Silicon) target..."
$CARGO build $CARGO_FLAGS --target aarch64-apple-darwin

echo "Building x86_64 (Intel) target..."
$CARGO build $CARGO_FLAGS --target x86_64-apple-darwin

echo "Merging slices into universal binary: target/$PROFILE/$ASSET..."
mkdir -p "target/$PROFILE"
lipo -create -output "target/$PROFILE/$ASSET" \
  target/aarch64-apple-darwin/$PROFILE/libreaper_realackey.dylib \
  target/x86_64-apple-darwin/$PROFILE/libreaper_realackey.dylib

echo "Applying ad-hoc code signature..."
codesign --force --sign - "target/$PROFILE/$ASSET"

echo "== architectures =="
lipo -info "target/$PROFILE/$ASSET"

echo "Success! Built target/$PROFILE/$ASSET"
