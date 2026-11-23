# macOS packaging

`build.sh` builds `eov.app`: it compiles eov and `eov-update-helper`, bundles
the non-system dynamic libraries into `Contents/Frameworks`, signs the bundle
and produces the zipped release artifact. The recommended way to install eov
on macOS is the Homebrew cask (`brew install --cask eosin-platform/eov/eov`);
the zip is what the cask installs.

## What the build bundles

Every library eov loads, directly or through other libraries, that does not
come with macOS (`/usr/lib`, `/System`) is copied into `Contents/Frameworks`.
References are resolved the way dyld resolves them on the build machine:
absolute paths, `@loader_path`, `@executable_path`, and `@rpath` through the
`LC_RPATH` entries of the library and of the libraries that load it. Homebrew
libraries reference their siblings through `@rpath` or `@loader_path`
(`libwebp.7.dylib` links `@rpath/libsharpyuv.0.dylib`, found through
`LC_RPATH @loader_path/../lib`).
dyld's fallback paths (`/usr/local/lib`, Homebrew's prefix on Intel Macs) are
deliberately not searched, so a library found only there fails the build
instead of being missing on users' Macs. eov then references the copies as
`@executable_path/../Frameworks/<name>`, and the libraries reference each
other as `@loader_path/<name>`. The `LC_RPATH` entries that point outside the
bundle are deleted, so nothing in the bundle depends on `@rpath`.

## What the build checks

- **Linkage (`otool -l`).** Every library `eov` and the bundled dylibs load
  must be a system library or a file inside the bundle, referenced through
  `@executable_path` or `@loader_path`. Absolute paths such as
  `/opt/homebrew/...` or the CI build prefix fail the build, as do `@rpath`
  references and `LC_RPATH` entries that point outside the bundle.
  `eov-update-helper` may only link system libraries: during an update eov
  runs it from a private temporary directory, where
  `@executable_path/../Frameworks` does not exist.
- **Smoke tests.** `eov --version` runs from a copy of the finished bundle in
  a temporary directory, without `DYLD_LIBRARY_PATH` and with dyld's fallback
  limited to `/usr/lib`, the way it starts on a Mac without Homebrew.
  `DYLD_PRINT_LIBRARIES` lists every image dyld loads, and each must come
  from the copy or from macOS (dyld ignores the variable under the hardened
  runtime; the build warns and checks only that eov starts).
  `eov-update-helper --self-test` runs from a temporary copy.
- **Tests without a Mac.** `test-bundle-libraries.sh` builds Mach-O libraries
  laid out like Homebrew's, bundles them with `build.sh`'s functions and
  checks the result, and that the linkage check rejects broken bundles. It
  runs on macOS with the Xcode tools and elsewhere with LLVM's `clang`,
  `ld64.lld`, `llvm-otool` and `llvm-install-name-tool` (Debian and Ubuntu:
  `apt install clang lld llvm`); CI runs it on Linux.
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
brew install cairo glib jpeg-turbo libpng libtiff libxml2 meson ninja openjpeg pkg-config sqlite zstd
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
`eosin-platform/openslide@eb6dcaeca5503e4574f1187e5ff65e9cea28e2d8` (pinned commit
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
