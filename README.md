# eov — A lightweight WSI viewer

[![CI Status](https://github.com/eosin-platform/eov/actions/workflows/ci.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/ci.yml)
![Multi-Arch](https://img.shields.io/badge/arch-x86__64%20%7C%20arm64-blue)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-0366d6)](https://github.com/eosin-platform/eov#license)
[![Linux AppImage Build Status](https://github.com/eosin-platform/eov/actions/workflows/build-appimage.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/build-appimage.yml)
[![Linux Flatpak Build Status](https://github.com/eosin-platform/eov/actions/workflows/build-flatpak.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/build-release.yml)
[![Windows Build Status](https://github.com/eosin-platform/eov/actions/workflows/build-windows-zip.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/build-windows-zip.yml)
[![Status: Actively Maintained](https://img.shields.io/badge/status-actively%20maintained-2ea44f)](https://github.com/eosin-platform/eov/pulse)

<p align="center">
    <img src="images/eov.webp" width="256">
</p>


eov is a desktop viewer for whole-slide images built in Rust. It fills a niche in the WSI ecosystem: a small, high-performance workbench for quickly viewing WSI files on your local machine. The feature scope is intentionally narrow with its design principle of "small Linux utility for WSI".

Whereas the sister project [Eosin](https://github.com/eosin-platform/eosin) solves the institution-scale WSI problem, eov aims to provide researchers (and anyone else interested in WSI!) with frictionless viewer capabilities free of extraneous dependencies (e.g. servers, cloud infrastructure).

The name `eov` has no canonical expansion.

## Installation

Windows and Linux are currently supported. MacOS support is pending.

Prebuilt binaries can be downloaded from the [Releases](https://github.com/eosin-platform/eov/releases) page.

Both x86_64 and arm64 builds for all supported platforms are available. Make you sure you select the right architecture!

### Example WSI Files

To make testing easier, here are a few (relatively) small WSI files that can be opened by `eov`. These are clear cell renal carcinoma slides from [CPTAC-CRCC](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/):
- [C3L-00004-21.svs](https://cptac.nyc3.digitaloceanspaces.com/images/CPTAC-CCRCC/C3L-00004-21.svs) (169MB)
- [C3L-00088-22.svs](https://cptac.nyc3.digitaloceanspaces.com/images/CPTAC-CCRCC/C3L-00088-22.svs) (322MB)

### Linux
There are two methods of installation: AppImage and Flatpak. The AppImage is directly executable:

```bash
# Make it executable
chmod +x eov-v0.1.20-linux-x86_64.AppImage

# Run it directly
./eov-v0.1.20-linux-x86_64.AppImage

# Install it system-wide (optional)
sudo mv eov-v0.1.20-linux-x86_64.AppImage /usr/local/bin/eov

# Start the installed app with a nice, short command.
eov
```

### MacOS
Prebuilt binaries are available via brew and direct download (`.dmg` file) (TODO)

### Windows
Tagged releases publish a Windows zip to GitHub Releases automatically. Run `eov.exe` within the zip to start the program. Only the portable version is available; no Windows installer is planned.

If you want the `eov` command to be available via PATH (e.g. for command prompt or PowerShell) you can do [adding `C:/path/to/eov` to System Variables](https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574%28v%3Doffice.14%29) (given `C:/path/to/eov/eov.exe` reflects your directory structure).

## Overview

eov opens pyramid-based whole-slide image files through OpenSlide and presents them in a desktop viewer designed for fast inspection.

Current capabilities include:

- Open WSI files from the file picker, drag and drop, recent-files list, or the command line.
- A tab- and pane-based layout
- Duplicate tabs into additional panes for side-by-side comparison.
- Drag tabs between panes, reorder tabs, and create splits by dropping onto pane edges.
- Pan and zoom smoothly with on-demand tile loading and cached rendering.
- Toggle between CPU and GPU rendering, with automatic fallback to CPU when GPU rendering is unavailable.
- A minimap thumbnail with viewport navigation and a zoom slider.
- Basic "Region of Interest" and "Measure Distance" tools.
- Various quality of life enhancements expected from modern software packages

## Screenshots

<p align="center">
    <img src="images/screenshot.webp" width="512">
    &nbsp;
    <img src="images/screenshot-2.webp" width="512">
</p>

## Supported Formats

eov relies on [OpenSlide](https://openslide.org/) for slide access, so the formats it can open are the formats OpenSlide supports on the host system. The application explicitly offers these common extensions in the file picker:

- `.svs`
- `.tif`
- `.tiff`
- `.ndpi`
- `.vms`
- `.vmu`
- `.scn`
- `.mrxs`
- `.bif`

If OpenSlide can open the file, eov should be able to load it.


## CLI

The application can be launched as a desktop viewer or used through a small CLI surface:

```text
eov [OPTIONS] [FILES]...
eov probe <FILE>
eov recent list
eov config-path
```

Examples:

```bash
eov slide.svs
eov slide1.svs slide2.svs slide3.svs
eov --debug --backend gpu slide.svs
eov --cache-size 512 --max-tiles 4096 slide.svs
eov --gpu slide.svs
eov --log-level debug probe fixtures/C3L-00088-22.svs
eov --config /tmp/config.toml config-path
eov recent list
```

Notable options:

- `--backend auto|cpu|gpu`
- `--cpu` and `--gpu` as shorthands for `--backend cpu|gpu`
- `--debug` to enable debug overlays in the UI
- `--log-level error|warn|info|debug|trace`
- `--cache-size <MB>` to set the tile-cache budget in megabytes. Default and recommended value: `256`.
- `--max-tiles <COUNT>` to cap the number of cached tiles. Default and recommended value: `2048`.
- `--config <PATH>` to override the active config file path for the current process

## Architecture

This repository is a Cargo workspace with two crates:

- `common`: WSI access, tile management, caching, viewport math, and benchmarks.
- `app`: the desktop application built with Slint + CPU/GPU rendering paths.

At a high level, the flow is:

1. Open a slide through OpenSlide.
2. Read slide metadata and pyramid levels.
3. Compute visible tiles for each active viewport.
4. Load tiles in the background and cache them.
5. Render the composed image through the selected backend.

## Requirements

To **manually** build eov and run the binary, you need:

- A recent Rust toolchain with Cargo.
- OpenSlide installed on the system, including the development package needed for linking.
- The native libraries required by Slint, winit, and the selected graphics stack on your platform.

On Linux, the most important dependency is usually OpenSlide itself. Depending on your distribution, you may also need the usual X11, Wayland, EGL, and font development packages used by Rust GUI applications.

## Building

Build the release binary from the workspace root:

```bash
cargo build --bin eov --release
```

For a development build:

```bash
cargo build --bin eov
```

## Packaging

Linux packaging assets live under `assets/linux/` and `packaging/`.

- AppImage is the first-class direct-download Linux artifact.
- Flatpak support is included for sandboxed distribution.
- All built packages use the OpenSlide shared library and required runtime shared libraries into the AppDir/AppImage instead of statically linking them (pursuant to LGPL compliance).

Current packaging entry points:

- `./packaging/appimage/build.sh`
- `./packaging/flatpak/build.sh`

## Configuration And Persistence

eov currently persists two kinds of state:

- Preferred render backend in `~/.eov/config.toml` by default.
- Recently opened files in the XDG config directory under `~/.eov/recent_files.txt`.

You can override the render-backend config path with the `EOV_CONFIG` environment variable.

Example backend config:

```toml
render_backend = "gpu"
```

## Repository Layout

```text
.
├── Cargo.toml            # Workspace definition
├── app/                  # Desktop application crate
│   ├── src/              # Application logic, rendering, callbacks, state
│   └── ui/               # Slint UI components
├── common/               # Shared WSI, tile, cache, and viewport code
│   ├── src/
│   └── benches/
└── fixtures/             # Sample data used for local testing/benchmarks
```

## Status

The project is functional and already supports the core interactive viewing workflow, but it is still evolving. Expect the UI, configuration layout, and supported workflows to continue changing as the viewer matures.

Contributions are welcome. This project is part of the [Eosin Platform](https://github.com/eosin-platform).


## License

Apache 2.0 / MIT dual license.