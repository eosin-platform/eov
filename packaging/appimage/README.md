# AppImage packaging

This directory contains the Linux AppImage packaging glue for eov.

AppImage is the primary direct-download Linux artifact for this repository.
OpenSlide remains dynamically linked for LGPL-2.1 compliance. The build script
bundles the OpenSlide shared library and its required runtime shared libraries
into the AppDir/AppImage instead of statically linking OpenSlide into eov.

## Requirements

- Rust and Cargo
- OpenSlide installed on the host so the Rust binary links dynamically during
  the release build
- Common Linux GUI development packages needed by Slint and winit on the host
- `linuxdeploy` preferred, or `appimagetool` as a fallback
- `ldd`, `ldconfig`, and standard POSIX shell utilities

On Debian or Ubuntu hosts, a typical starting point is:

```bash
sudo apt install build-essential cargo pkg-config libopenslide-dev libfontconfig-dev libx11-dev libxkbcommon-dev libwayland-dev libegl1-mesa-dev
```

You still need either `linuxdeploy` or `appimagetool` available in `PATH`.

## Build

From the repository root:

```bash
./packaging/appimage/build.sh
```

Environment overrides are available for CI and local packaging:

```bash
APP_ID=io.eosin.eov \
APP_NAME=eov \
DIST_DIR="$PWD/dist" \
./packaging/appimage/build.sh
```

Useful overrides:

- `APP_ID`
- `APP_NAME`
- `DIST_DIR`
- `BINARY_PATH`
- `ICON_SOURCE`
- `OPENSLIDE_LIB_PATH`

## Notes

- The script intentionally avoids static linking for OpenSlide.
- The script bundles `libopenslide.so` and its non-blacklisted dependent shared
  libraries into `AppDir/usr/lib`.
- glibc and related loader libraries are intentionally skipped rather than
  bundled.
- If this project ever carries local OpenSlide patches, those patches must be
  made available to recipients under LGPL-2.1 terms.
