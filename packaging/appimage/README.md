# AppImage packaging

This directory contains the Linux AppImage packaging for eov.

AppImage is the primary direct-download Linux artifact. OpenSlide stays
dynamically linked for LGPL-2.1 compliance: `build.sh` bundles the OpenSlide
shared library and the libraries it needs into `usr/lib` instead of linking
OpenSlide statically into eov.

## How the AppImage is put together

- **glibc baseline.** glibc cannot be bundled, so the newest glibc symbol any
  bundled file needs is the oldest glibc the AppImage runs on. Release builds
  run in an AlmaLinux 8 container and are checked against **glibc 2.28**
  (RHEL/Rocky/AlmaLinux 8, Debian 10, Ubuntu 18.10 and newer) by
  `check-glibc.sh`.
- **No `LD_LIBRARY_PATH`.** eov finds `usr/lib` through its `RUNPATH`
  (`$ORIGIN/../lib`) and each bundled library finds its neighbours through
  `$ORIGIN`, both set with `patchelf`. `AppRun` only starts eov, so programs
  eov launches (`xdg-open`, file managers, browsers) never load eov's copies.
  eov also gives them its launch environment without the AppImage runtime's
  `APPIMAGE`, `APPDIR`, `ARGV0` and `OWD`; should `AppRun` ever have to set
  a variable, it must save the launch value (`EOV_ORIG_<NAME>`, listed in
  `EOV_SAVED_ENV`), which `scripts/check_packaging.py` enforces.
- **Desktop integration.** The AppDir carries the desktop entry (with the
  slide MIME types and `x-scheme-handler/eov`) and
  `usr/share/mime/packages/io.eosin.eov.xml`. Integration tools such as
  AppImageLauncher install them; without one, `eov register-file-types`
  installs them for the current user, pointing at the AppImage itself.
- **System libraries stay on the host.** The glibc family and the libraries
  in the [AppImage excludelist](https://github.com/AppImageCommunity/pkg2appimage/blob/master/excludelist)
  (libstdc++, libgcc_s, GL/EGL/DRM, X11/xcb, Wayland, fontconfig, freetype,
  zlib, ...) are never bundled: an old copy loaded first would break newer
  host components, such as GPU drivers, that need the host's version.
- **Verification.** Before packing, `build.sh` checks that eov and every
  bundled library resolve without `LD_LIBRARY_PATH`, that OpenSlide resolves
  to the bundled copy, that `AppRun --version` starts, and the glibc floor.
- **Pinned tools.** `fetch-tools.sh` downloads appimagetool and the AppImage
  type-2 runtime at fixed versions and verifies their SHA-256s.
- **Licences.** `usr/share/doc/eov` holds eov's licences, the third-party
  notices (`COPYING.third-party`) and `licenses/LGPL-2.1.txt`.

## Release build (as CI does it)

```bash
docker run --rm -it -v "$PWD:/src" -w /src almalinux:8 bash
# inside the container:
packaging/appimage/install-build-deps-el8.sh
curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
eval "$(packaging/appimage/fetch-tools.sh /tmp/appimage-tools | sed 's/^/export /')"
APPIMAGE_EXTRACT_AND_RUN=1 APPIMAGE_MAX_GLIBC=2.28 packaging/appimage/build.sh
```

`APPIMAGE_EXTRACT_AND_RUN=1` lets appimagetool (itself an AppImage) run
without FUSE, as in containers.

## Local build

On any distribution with the build dependencies, `patchelf` and `appimagetool`
in `PATH`:

```bash
./packaging/appimage/build.sh
```

On Debian or Ubuntu, a typical starting point is:

```bash
sudo apt install build-essential curl git libcairo2-dev libfontconfig-dev libgdk-pixbuf-2.0-dev libglib2.0-dev libjpeg-turbo8-dev libopenjp2-7-dev libpng-dev libsqlite3-dev libtiff-dev libxml2-dev libzstd-dev meson ninja-build patchelf pkg-config zlib1g-dev
```

A local build reports its glibc requirement but only enforces one when
`APPIMAGE_MAX_GLIBC` is set; built on a current distribution, the AppImage
needs that distribution's glibc.

Useful overrides:

- `APP_ID`, `APP_NAME`, `DIST_DIR`, `OUTPUT_PATH`
- `BINARY_PATH` (default `$CARGO_TARGET_DIR/release/eov`)
- `ICON_SOURCE`
- `OPENSLIDE_LIB_PATH` (use an existing OpenSlide instead of building the fork)
- `APPIMAGE_MAX_GLIBC`, `APPIMAGE_RUNTIME_FILE`, `APPIMAGETOOL`

## Notes

- By default the script builds OpenSlide from
  `Yanstart/openslide@00bebfb6a412485f7fbfe7fdb5919a3868594c13` with
  `packaging/shared/build-openslide.sh` (pinned commit and checksums) into a
  private prefix before building eov.
- If this project ever carries local OpenSlide patches, those patches must be
  made available to recipients under LGPL-2.1 terms.
- The AppImage updates itself (`eov update`): the new file is downloaded and
  verified next to the running one, then renamed over it in one atomic step.
