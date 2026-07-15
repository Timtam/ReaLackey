#!/usr/bin/env bash
# Build the universal macOS dylib (arm64 + x86_64) and Developer-ID sign + notarize
# it. Shared by the Release workflow and the manual macOS-notarization-test
# workflow, so testing the latter validates the exact path a release uses.
#
# Without the signing secrets it falls back to an ad-hoc signature (the universal
# binary still works, but users must clear quarantine with xattr). Everything is
# read from the environment:
#   ASSET  - output dylib name, e.g. reaper_realackey.dylib   (required)
#   MACOS_CERTIFICATE_P12_BASE64 / MACOS_CERTIFICATE_PASSWORD
#   MACOS_NOTARY_KEY_P8_BASE64 / MACOS_NOTARY_KEY_ID / MACOS_NOTARY_ISSUER_ID
set -euo pipefail
: "${ASSET:?set ASSET to the output dylib name}"
: "${RUNNER_TEMP:=${TMPDIR:-/tmp}}"

# --- universal build (arm64 + x86_64) --------------------------------------
rustup target add aarch64-apple-darwin x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin
mkdir -p dist
lipo -create -output "dist/$ASSET" \
  target/aarch64-apple-darwin/release/libreaper_realackey.dylib \
  target/x86_64-apple-darwin/release/libreaper_realackey.dylib
echo "== architectures ==" && lipo -info "dist/$ASSET"

# --- ad-hoc fallback when no signing secrets are configured -----------------
if [ -z "${MACOS_CERTIFICATE_P12_BASE64:-}" ]; then
  echo "::warning::No macOS signing secrets set — ad-hoc signing only. Users must run: xattr -dr com.apple.quarantine <dylib>."
  codesign --force --sign - "dist/$ASSET"
  codesign -dvv "dist/$ASSET" 2>&1 || true
  exit 0
fi
: "${MACOS_CERTIFICATE_PASSWORD:?}" "${MACOS_NOTARY_KEY_P8_BASE64:?}" \
  "${MACOS_NOTARY_KEY_ID:?}" "${MACOS_NOTARY_ISSUER_ID:?}"

# --- import the Developer ID cert into a throwaway keychain -----------------
KEYCHAIN="$RUNNER_TEMP/build.keychain-db"
KEYCHAIN_PW="$(openssl rand -base64 24)"
printf '%s' "$MACOS_CERTIFICATE_P12_BASE64" | base64 -D > "$RUNNER_TEMP/cert.p12"
printf '%s' "$MACOS_NOTARY_KEY_P8_BASE64"   | base64 -D > "$RUNNER_TEMP/notary.p8"
security create-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
security import "$RUNNER_TEMP/cert.p12" -k "$KEYCHAIN" -P "$MACOS_CERTIFICATE_PASSWORD" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PW" "$KEYCHAIN" >/dev/null
security list-keychains -d user -s "$KEYCHAIN"

# --- sign with the Developer ID identity (hardened runtime + timestamp) -----
SIGN_ID="$(security find-identity -v -p codesigning "$KEYCHAIN" | awk '/Developer ID Application/ {print $2; exit}')"
if [ -z "$SIGN_ID" ]; then
  echo "::error::No 'Developer ID Application' identity found in the imported certificate."
  exit 1
fi
codesign --force --timestamp --options runtime --sign "$SIGN_ID" "dist/$ASSET"
codesign --verify --strict --verbose=2 "dist/$ASSET"

# --- notarize (a bare .dylib can't be stapled; Gatekeeper checks online) ----
ditto -c -k "dist/$ASSET" "$RUNNER_TEMP/notarize.zip"
if ! xcrun notarytool submit "$RUNNER_TEMP/notarize.zip" \
       --key "$RUNNER_TEMP/notary.p8" --key-id "$MACOS_NOTARY_KEY_ID" \
       --issuer "$MACOS_NOTARY_ISSUER_ID" --wait --timeout 30m 2>&1 | tee "$RUNNER_TEMP/notary.log"; then
  SID="$(awk '/id:/ {print $2; exit}' "$RUNNER_TEMP/notary.log" || true)"
  [ -n "$SID" ] && xcrun notarytool log "$SID" --key "$RUNNER_TEMP/notary.p8" \
    --key-id "$MACOS_NOTARY_KEY_ID" --issuer "$MACOS_NOTARY_ISSUER_ID" || true
  echo "::error::Notarization failed."
  exit 1
fi
echo "== signature ==" && codesign -dvv "dist/$ASSET" 2>&1 || true
echo "Signed + notarized OK."
rm -f "$RUNNER_TEMP/cert.p12" "$RUNNER_TEMP/notary.p8"
