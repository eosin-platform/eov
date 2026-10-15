# macOS packaging

`build.sh` builds `eov.app`: it compiles eov and `eov-update-helper`, bundles
the non-system dynamic libraries into `Contents/Frameworks`, signs the bundle
and produces the zipped release artifact. The recommended way to install eov
on macOS is the Homebrew cask (`brew install --cask eosin-platform/eov/eov`);
the zip is what the cask installs.

## What the build checks

- **Linkage (`otool -L`).** Every reference in `eov` and the bundled dylibs
  must be a system library (`/usr/lib`, `/System`) or a file inside the
  bundle; absolute paths such as `/opt/homebrew/...` or the CI build prefix
  fail the build, as do absolute `LC_RPATH` entries. `eov-update-helper` may
  only link system libraries: during an update eov runs it from a private
  temporary directory, where `@executable_path/../Frameworks` does not exist.
- **Smoke tests.** `eov --version` runs from the finished bundle, and
  `eov-update-helper --self-test` runs from a temporary copy, both without
  `DYLD_LIBRARY_PATH`.
- **Minimum macOS version.** eov, the helper, OpenSlide and libdicom are built
  for `MACOSX_DEPLOYMENT_TARGET` (default: `LSMinimumSystemVersion` in
  `assets/macos/Info.plist`, 11.0). Libraries taken from Homebrew are built
  for the build machine's macOS, so the build reads every binary's `minos`
  and raises `LSMinimumSystemVersion` to the newest one: macOS then refuses to
  open eov on an older release instead of crashing in dyld, the self-updater
  refuses updates the Mac cannot run, and the Homebrew cask's
  `depends_on macos:` is derived from the shipped bundles. Set
  `MACOS_STRICT_MINIMUM=1` to fail instead; supporting older macOS releases
  means building those libraries with the same deployment target.
- **Licences.** `Contents/Resources` holds eov's licences, the third-party
  notices (`COPYING.third-party`) and `licenses/LGPL-2.1.txt` for OpenSlide.
- **File types and links.** `Info.plist` declares the slide formats
  (`CFBundleDocumentTypes`, `UTImportedTypeDeclarations`; eov is the default
  for the vendor formats and an alternative for TIFF and DICOM) and the `eov`
  URL scheme (`CFBundleURLTypes`). The tests of `app/src/file_types` check
  the declarations against the formats eov opens; `plutil -lint` checks the
  rendered file.

## Requirements

- macOS with the Xcode command line tools (`codesign`, `install_name_tool`,
  `otool`)
- Homebrew build dependencies needed to compile OpenSlide from source, or
  `OPENSLIDE_LIB_DIR` set to a prebuilt OpenSlide install
- Rust toolchain with Cargo

Typical Homebrew dependencies:

```bash
brew install cairo gdk-pixbuf glib jpeg-turbo libpng libtiff libxml2 meson ninja openjpeg pkg-config sqlite zstd
```

## Usage

Build a zipped `.app` bundle:

```bash
./packaging/macos/build.sh
```

Build for a specific target triple:

```bash
RUST_TARGET=aarch64-apple-darwin ./packaging/macos/build.sh
RUST_TARGET=x86_64-apple-darwin ./packaging/macos/build.sh
```

Useful environment overrides:

- `VERSION`, `RUST_TARGET`, `ARCH_LABEL`, `ARCHIVE_BASENAME`, `DIST_DIR`
- `ICNS_SOURCE`
- `MACOSX_DEPLOYMENT_TARGET`, `MACOS_STRICT_MINIMUM`
- `CODESIGN_IDENTITY` (default `-`, ad-hoc)
- `NOTARIZE`, `APPLE_API_KEY_PATH`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER`

By default, `build.sh` compiles OpenSlide from
`Yanstart/openslide@00bebfb6a412485f7fbfe7fdb5919a3868594c13` (pinned commit
and checksums, see `packaging/shared/build-openslide.sh`) into a private
staging prefix before building the app.

## Signing and notarization

Without credentials the bundle is signed ad hoc and the cask tells users how
to open it (System Settings → Privacy & Security → Open Anyway).

With a Developer ID Application certificate, set `CODESIGN_IDENTITY` to its
name: the dylibs, the helper and the app are then signed with the hardened
runtime, a secure timestamp and `entitlements.plist`
(`com.apple.security.cs.disable-library-validation`, because eov loads plugin
libraries that are not signed with eov's Team ID). `NOTARIZE=1` then submits
the bundle with `xcrun notarytool` (App Store Connect API key) and staples the
ticket before zipping.

The release workflow does this when these repository secrets exist:

| Secret | Contents |
|---|---|
| `MACOS_CERTIFICATE_P12` | Base64 of the Developer ID Application certificate and key (`.p12`) |
| `MACOS_CERTIFICATE_PASSWORD` | Password of that `.p12` |
| `MACOS_CODESIGN_IDENTITY` | e.g. `Developer ID Application: Example Ltd (TEAMID)` |
| `APPLE_API_KEY_P8` | Base64 of the App Store Connect API key (`AuthKey_XXXX.p8`) |
| `APPLE_API_KEY_ID` | The key's id |
| `APPLE_API_ISSUER` | The issuer id |
