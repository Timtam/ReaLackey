# macOS signing & notarization — one-time setup

The Release workflow signs the macOS `.dylib` with your **Developer ID** and
**notarizes** it with Apple, so users can load it without clearing the download
quarantine. It only does this when the five secrets below are present — without
them, the release still produces a working universal binary, just ad-hoc-signed
(users then run `xattr -dr com.apple.quarantine …`).

You do this setup **once**. It needs a **Mac** (for Keychain Access) and a paid
**Apple Developer Program** membership (~$99/year — the "Developer ID" certificate
and the notary service are not available on a free account).

> Why notarize a loose `.dylib`? A REAPER extension is a bare dylib, and Apple's
> `stapler` can't attach a ticket to a loose dylib (only to `.app`/`.dmg`/`.pkg`).
> That's fine: once the dylib is notarized, Gatekeeper verifies it **online** the
> first time REAPER loads it, which clears the block. (Offline-first-load is the
> one gap — the `xattr` step remains a fallback.)

---

## 1. Create a "Developer ID Application" certificate and export it as `.p12`

You end up with a `DeveloperID.p12` holding the certificate **and its private
key**. Two ways:

### 1a. Without a Mac — OpenSSL (Windows Git Bash / WSL / Linux)

Everything here works in a browser + a POSIX shell. `openssl` ships with Git for
Windows (Git Bash), WSL, and Linux.

```sh
# (i) generate a private key + a certificate signing request (CSR)
openssl genrsa -out devid_key.pem 2048
openssl req -new -key devid_key.pem -out devid.csr \
  -subj "/emailAddress=YOU@EXAMPLE.COM/CN=YOUR NAME/C=US"
```

Now, in the browser, turn the CSR into a certificate:

- <https://developer.apple.com/account> → **Certificates, IDs & Profiles** →
  **Certificates** → **+** → **Developer ID Application** → upload `devid.csr` →
  **Download** `developerID_application.cer`.
  *(Only the account's **Account Holder** can create Developer ID certificates.)*

```sh
# (ii) Apple's cert is DER; convert to PEM
openssl x509 -inform DER -in developerID_application.cer -out devid_cert.pem

# (iii) bundle key + cert into a .p12.
#   -legacy: needed on OpenSSL 3.x (Git Bash) so macOS's `security import` can read
#   the file. If your openssl is 1.1.x it will reject -legacy — just drop it.
openssl pkcs12 -export -legacy \
  -inkey devid_key.pem -in devid_cert.pem \
  -out DeveloperID.p12 -passout pass:CHOOSE_A_PASSWORD
```

`CHOOSE_A_PASSWORD` becomes the `MACOS_CERTIFICATE_PASSWORD` secret.

> If a later release fails at signing with a "unable to build chain" / authority
> error, embed Apple's intermediate too: download the *Developer ID Certification
> Authority* cert from <https://www.apple.com/certificateauthority/>, convert it to
> PEM, and add `-certfile DeveloperID_intermediate.pem` to the `pkcs12 -export`
> line. (Usually unnecessary — the GitHub runner already trusts Apple's
> intermediates.)

### 1b. With a Mac — Xcode + Keychain Access

Xcode → **Settings → Accounts → Manage Certificates → + → Developer ID
Application**; then Keychain Access → **login → My Certificates** → right-click the
`Developer ID Application: …` identity → **Export** as `.p12` with a password.

## 2. Create an App Store Connect API key (for the notary service)

1. Go to <https://appstoreconnect.apple.com> → **Users and Access** → **Integrations**
   tab → **App Store Connect API** (Team Keys).
2. Click **+** (Generate API Key). Name it e.g. `notary`, give it the **Developer**
   access role (bump to *App Manager* if notarization is later rejected for
   permissions).
3. **Download** the `AuthKey_XXXXXXXXXX.p8` — you can only download it **once**.
4. Note two IDs on that page:
   - the **Key ID** (next to the key, e.g. `ABCD123456`),
   - the **Issuer ID** (near the top of the Keys section, a UUID).

## 3. Base64-encode the two files

The secrets must be text, so base64 the two binaries. The workflow decodes them on
macOS and doesn't care whether the base64 is wrapped or on one line.

**Git Bash / WSL / Linux:**
```sh
base64 DeveloperID.p12          > p12.b64   # -> MACOS_CERTIFICATE_P12_BASE64
base64 AuthKey_ABCD123456.p8    > p8.b64    # -> MACOS_NOTARY_KEY_P8_BASE64
```

**Windows PowerShell:**
```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("DeveloperID.p12"))       | Set-Clipboard
[Convert]::ToBase64String([IO.File]::ReadAllBytes("AuthKey_ABCD123456.p8")) | Set-Clipboard
```

**macOS:** `base64 -i DeveloperID.p12 | pbcopy` (and the `.p8`).

Paste the resulting text straight into each secret.

## 4. Add the five GitHub secrets

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository
secret**. Add exactly these names:

| Secret | Value |
|---|---|
| `MACOS_CERTIFICATE_P12_BASE64` | base64 of the `.p12` (step 3) |
| `MACOS_CERTIFICATE_PASSWORD`   | the `.p12` export password (step 1) |
| `MACOS_NOTARY_KEY_P8_BASE64`   | base64 of the `.p8` (step 3) |
| `MACOS_NOTARY_KEY_ID`          | the Key ID (step 2) |
| `MACOS_NOTARY_ISSUER_ID`       | the Issuer ID (step 2) |

The signing identity and Team ID are derived automatically from the certificate,
so there's nothing else to add. A temporary keychain password is generated per run.

## 5. Release and verify

Run the **Release** workflow (Actions → Release → *Run workflow*). The **CI logs
are your verification** (no Mac needed): the macOS job's **"Sign & notarize"** step
must show the `Developer ID Application` identity, `codesign --verify` passing, and
`notarytool … status: Accepted`. The step hard-fails the release if signing or
notarization fails, so a misconfigured secret can't silently ship an unsigned
binary — it just stops the release.

If you (or a tester) have a Mac, you can double-check the downloaded dylib:

```sh
codesign -dvv reaper_realackey.dylib        # Authority = Developer ID Application: …
xcrun stapler validate reaper_realackey.dylib   # a loose dylib can't be stapled — expect
                                                 # "does not support stapling"; that's OK
```

## Rotating / troubleshooting

- **Release fails at `security import` (MAC/decrypt error):** your `.p12` used
  OpenSSL-3 encryption macOS can't read — re-run the `pkcs12 -export` with
  `-legacy` (see step 1a).
- **Developer ID Application cert missing / can't be created on the portal:** only
  the **Account Holder** role can create Developer ID certificates, and there's a
  small per-account limit — revoke an unused one if you've hit it.
- **`.p8` lost:** you can't re-download it — revoke the key and make a new one, then
  update `MACOS_NOTARY_KEY_P8_BASE64` / `MACOS_NOTARY_KEY_ID`.
- **Certificate expired** (Developer ID certs last 5 years): create a new one and
  re-export the `.p12`.
- **Notarization rejected:** the workflow prints the full `notarytool log` on
  failure — it lists the exact reason (unsigned nested code, missing timestamp,
  wrong cert type, etc.).
- **Secrets removed:** the release still works (ad-hoc fallback); users just need
  the `xattr` step again.
