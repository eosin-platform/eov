# eov — A lightweight WSI viewer

[![CI Status](https://github.com/eosin-platform/eov/actions/workflows/ci.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/ci.yml)
[![Release Status](https://github.com/eosin-platform/eov/actions/workflows/release.yml/badge.svg)](https://github.com/eosin-platform/eov/actions/workflows/release.yml)
![Multi-Arch](https://img.shields.io/badge/arch-x86__64%20%7C%20arm64-blue)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-0366d6)](https://github.com/eosin-platform/eov#license)
[![Status: Actively Maintained](https://img.shields.io/badge/status-actively%20maintained-2ea44f)](https://github.com/eosin-platform/eov/pulse)

<p align="center">
    <img src="images/eov.webp" width="256">
</p>
<p align="center">
    <a href="https://eov.sh">Website</a>
</p>


eov is a desktop viewer for whole-slide images built in Rust. It fills a niche in the WSI ecosystem: a small, high-performance workbench for quickly viewing WSI files on your local machine. The feature scope is intentionally narrow with its design principle of "small Linux utility for WSI".

Whereas the sister project [Eosin](https://github.com/eosin-platform/eosin) solves the institution-scale WSI problem, eov aims to provide researchers (and anyone else interested in WSI!) with frictionless viewer capabilities free of extraneous dependencies (e.g. servers, cloud infrastructure).

The name `eov` has no canonical expansion.

## Installation

Windows, macOS, and Linux are supported. Prebuilt binaries can be downloaded from the [Releases](https://github.com/eosin-platform/eov/releases) page.

Both x86_64 and arm64 builds for all supported platforms are available. Make you sure you select the right architecture!

### Linux
There are two methods of installation: AppImage and Flatpak. The AppImage is directly executable:

```bash
# Make it executable (ensure correct filename!)
chmod +x eov-v0.2.8-linux-x86_64.AppImage

# Run it directly
./eov-v0.2.8-linux-x86_64.AppImage

# Install it system-wide (optional)
sudo mv eov-v0.2.8-linux-x86_64.AppImage /usr/local/bin/eov

# Start the installed app with a nice, short command.
eov
```

### macOS
The `.app` file is available via the Releases page. Download and open the `.app` file to run it.

For Intel-based Macs, download the release with `x86` in the name. Apple M-series machines require the `arm64` bundle.

Installation via `brew` is currently a work-in-progress. 

### Windows
A zip file containing a portable Windows build is available on the Releases page. Extract and run `eov.exe` within the zip to start the program. Only the portable version is available; no Windows installer is planned.

If you want the `eov` command to be available via PATH (e.g. for command prompt or PowerShell) you can do this by [adding `C:/path/to/eov` to System Variables](https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574%28v%3Doffice.14%29) (given `C:/path/to/eov/eov.exe` reflects your directory structure).

### Example WSI Files

To make testing easier, here are a few (relatively) small WSI files that can be opened by `eov`. These are clear cell renal carcinoma slides from [CPTAC-CRCC](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/):
- [C3L-00004-21.svs](https://cptac.nyc3.digitaloceanspaces.com/images/CPTAC-CCRCC/C3L-00004-21.svs) (169MB)
- [C3L-00088-22.svs](https://cptac.nyc3.digitaloceanspaces.com/images/CPTAC-CCRCC/C3L-00088-22.svs) (322MB)

## Overview

eov opens pyramid-based whole-slide image files through OpenSlide and presents them in a desktop viewer designed for fast inspection.

Current capabilities include:

- Open WSI files from the file picker, drag and drop, recent-files list, or the command line.
- A tab- and pane-based layout
- Duplicate tabs into additional panes for side-by-side comparison.
- Drag tabs between panes, reorder tabs, and create splits by dropping onto pane edges.
- Pan and zoom smoothly with on-demand tile loading and cached rendering.
- Toggle between CPU and GPU rendering, with automatic fallback to CPU when GPU rendering is unavailable.
- Adaptive [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling) (high quality), [Trilinear](https://en.wikipedia.org/wiki/Trilinear_filtering) (industry standard, performant), and [Bilinear](https://en.wikipedia.org/wiki/Bilinear_interpolation) (fast) texture filtering.
- Real-time image adjustments: sharpness (unsharp mask), gamma, brightness, and contrast sliders applied via convolution + per-channel LUT (CPU) or per-pixel shader (GPU).
- Stain normalization: Macenko (SVD/PCA angular-percentile) and Vahadane (sparse non-negative dictionary learning) methods, both running on CPU and GPU backends with matched output.
- Color deconvolution: separate and visualize individual H&E stain channels with per-channel intensity control and channel isolation, on both CPU and GPU.
- A calibrated scale bar with automatic unit selection (µm, mm, inches) when microns-per-pixel metadata is available.
- A minimap thumbnail with viewport navigation and a zoom slider.
- Basic "Region of Interest" and "Measure Distance" tools.
- Export Image: right-click the viewport to export the current view as PNG or JPEG with configurable DPI, filtering mode, stain normalization, color deconvolution, and optional measurement/ROI overlays.
- Various quality of life enhancements expected from modern software packages

## Screenshots

<p align="center">
    <img src="images/screenshot-0.webp" width="512">
    &nbsp;
    <img src="images/screenshot-1.webp" width="512">
    &nbsp;
    <img src="images/screenshot-2.webp" width="512">
</p>

## Texture Filtering

eov provides three texture filtering modes, selectable from the viewport context menu:

| Mode | Description |
|------|-------------|
| **Bilinear** | Single mip-level sampling. Fastest, but can exhibit aliasing at low zoom. |
| **Trilinear** | Bilinear sampling with mip-level blending for smooth transitions across zoom levels. The default mode. |
| **Lanczos-3** | High-quality resampling using a windowed sinc kernel (a=3). Adaptively blends with trilinear at very low zoom to avoid ringing artifacts. |

Both CPU and GPU backends support all three modes. On the GPU path, trilinear blending is performed by sampling fine and coarse mip textures and mixing in the fragment shader. Lanczos resampling is implemented as a separable kernel evaluated per-pixel in a dedicated shader.

## Image Adjustments

Real-time image adjustment controls are available in the HUD settings panel:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Sharpness | 0.0 – 1.0 | 0.0 | Unsharp mask via 4-connected Laplacian kernel. 0 = disabled (bypass), 1 = maximum sharpening. |
| Gamma | 0.2 – 3.0 | 1.0 | Power-law intensity curve. Values below 1.0 brighten shadows; above 1.0 darkens them. |
| Brightness | -1.0 – 1.0 | 0.0 | Additive offset applied after gamma correction. |
| Contrast | 0.2 – 3.0 | 1.0 | Multiplicative scaling around the midpoint (0.5) after brightness. |

The CPU backend applies sharpening as a Laplacian convolution pass over the composited buffer, followed by gamma/brightness/contrast through a precomputed 256-entry lookup table. The GPU backend applies the same pipeline per-pixel in the fragment shader.

## Stain Normalization

eov includes two H&E stain normalization methods for histology images, selectable from the HUD settings panel:

- **Macenko** ([Macenko et al., 2009](https://www.researchgate.net/publication/221624097_A_Method_for_Normalizing_Histology_Slides_for_Quantitative_Analysis)): Projects tissue optical density onto its principal stain plane via SVD, then identifies hematoxylin and eosin vectors from angular percentile extremes (1st and 99th percentile).
- **Vahadane** ([Vahadane et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27164577/)): Learns a two-component stain dictionary via sparse non-negative matrix factorization with L1 (LASSO) regularization, producing a physically-motivated decomposition that preserves local tissue structure.

Both methods share a common normalization pipeline: RGB → optical density conversion, tissue masking, stain matrix estimation, least-squares concentration solve, 99th-percentile concentration scaling to a reference target, and reconstruction. The CPU and GPU backends produce matched output — stain matrix estimation runs on the CPU from raw tile data, while per-pixel normalization runs either as a CPU buffer pass or as a GPU fragment shader transform.

## Color Deconvolution

eov supports real-time H&E color deconvolution, which separates a histology image into its constituent hematoxylin and eosin stain channels using the Beer–Lambert optical density model ([Ruifrok & Johnston, 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/)). Controls are available in the HUD settings panel:

- **Channel visibility**: toggle hematoxylin and eosin channels on or off independently.
- **Channel isolation**: view a single stain channel in isolation to inspect hematoxylin (nuclei) or eosin (cytoplasm/stroma) contributions separately.
- **Intensity control**: adjust per-channel stain intensity to enhance or suppress individual components.

When stain normalization is active, the deconvolution reuses the slide-specific stain matrix estimated by the normalization step, so channel separation reflects the actual stain vectors of the current slide. Otherwise the default Ruifrok & Johnston reference matrix is used. Both CPU and GPU backends are supported.

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
eov --cpu slide.svs
eov --gpu --filtering-mode lanczos slide.svs
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

## Configuration And Persistence

eov currently persists two kinds of state:

- Preferred render backend in `~/.eov/config.toml` by default.
- Recently opened files in the directory under `~/.eov/recent_files.txt`.

You can override the render-backend config path with the `EOV_CONFIG` environment variable.

Example backend config:

```toml
render_backend = "gpu"
filtering_mode = "trilinear"
```

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

## Building

### Environment

To **manually** build eov and run the binary, you need:

- A recent Rust toolchain with Cargo.
- OpenSlide installed on the system, including the development package needed for linking.
- The native libraries required by Slint, winit, and the selected graphics stack on your platform.

On Linux, the most important dependency is usually OpenSlide itself. Depending on your distribution, you may also need the usual X11, Wayland, EGL, and font development packages used by Rust GUI applications. Generally, you can just run this you're good to go:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cargo \
    curl \
    file \
    libegl1-mesa-dev \
    libfontconfig-dev \
    libopenslide-dev \
    libwayland-dev \
    libx11-dev \
    libxkbcommon-dev \
    pkg-config \
    squashfs-tools \
    zsync
```

### How to Build

Build the release binary from the workspace root:

```bash
cargo build --bin eov
```

Build and run (for development):

```bash
cargo run --bin eov -- path/to/myfile.svs
```

For a release build:

```bash
cargo build --bin eov --release
```

## Packaging

Packaging assets live under `assets/` and `packaging/`.

- AppImage is the first-class direct-download Linux artifact.
- Flatpak support is included for sandboxed distribution.
- macOS packaging is source-controlled via `packaging/macos/` and produces a bundled `.app` archive.
- All built packages use the OpenSlide shared library and required runtime shared libraries into the AppDir/AppImage instead of statically linking them (pursuant to LGPL compliance).

Current packaging entry points:

- `./packaging/appimage/build.sh`
- `./packaging/macos/build.sh`
- `./packaging/flatpak/build.sh`

## Repository Layout

```text
.
├── Cargo.toml            # Workspace definition
├── app/                  # Desktop application crate
│   ├── src/              # Application logic, rendering, callbacks, state
│   └── ui/               # Slint UI components
├── common/               # Shared WSI, tile, cache, and viewport code
│   ├── src/
│   └── benches/          # Benchmarks for various core functions
└── fixtures/             # Sample data used for local testing/benchmarks
```

## Status

The project is functional and already supports the core interactive viewing workflow, but it is still evolving. Expect the UI, configuration layout, and supported workflows to continue changing as the viewer matures.

Contributions are welcome. This project is part of the [Eosin Platform](https://github.com/eosin-platform).


## License

All of the code in this repository is released under Apache 2.0 / MIT dual license.

### Dependency License Notes

- [OpenSlide](https://openslide.org/) is released under LGPL. EOV's releases bundle OpenSlide as a dynamic library to comply with the terms of LGPL.
- [Slint](https://slint.dev/) is used under the free license, which is permissive outside of embedded environments.
