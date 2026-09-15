# eov

[![crates.io](https://img.shields.io/crates/v/eov.svg)](https://crates.io/crates/eov)
[![License](https://img.shields.io/crates/l/eov.svg)](https://github.com/eosin-platform/eov#license)

`eov` is a lightweight, cross-platform desktop viewer for pyramid-based whole-slide images (WSI). It is built with [Rust](https://www.rust-lang.org/) and [Slint](https://slint.dev/), and opens slides locally through [OpenSlide](https://openslide.org/).

The application is designed for fast inspection without a server or cloud service. It provides smooth pan and zoom, tab and pane layouts, CPU and GPU rendering, image adjustments, stain normalization, color deconvolution, a minimap, a calibrated scale bar, region-of-interest and distance measurement tools, image export, and dataset patch extraction.

This package installs the `eov` desktop application. It is not a library crate.

## Installation

### Cargo

Install the application from [crates.io](https://crates.io/crates/eov):

```bash
cargo install eov
```

The Cargo build requires OpenSlide and the native dependencies used by Slint and the platform graphics stack. On Debian or Ubuntu, the OpenSlide development package is:

```bash
sudo apt-get install libopenslide-dev
```

You may also need the usual X11 or Wayland, Vulkan, font, and `pkg-config` development packages for your distribution.

For prebuilt AppImage, Flatpak, macOS, and Windows packages, see the [latest releases](https://github.com/eosin-platform/eov/releases/latest).

## Usage

Open one or more slides:

```bash
eov slide.svs
eov slide1.svs slide2.svs slide3.svs
```

Positional arguments define panes. Comma-separated files share a pane as tabs, and the last file in each group is selected:

```bash
eov a.svs,b.svs c.tif d.svs
```

The command-line interface also provides inspection and dataset utilities:

```text
eov [OPTIONS] [FILES]...
eov probe <FILE>
eov recent list
eov config-path
eov dataset patches <INPUTS>... --out <DIR> --tile-size <PX> --stride <PX>
```

Useful options include:

- `--backend auto|cpu|gpu` to select the rendering backend.
- `--cpu` and `--gpu` as shortcuts for `--backend cpu` and `--backend gpu`.
- `--filtering-mode auto|bilinear|trilinear|lanczos` to select texture filtering.
- `--debug` to enable debug overlays.
- `--log-level error|warn|info|debug|trace` to control logging.
- `--cache-size <MB>` and `--max-tiles <COUNT>` to tune the tile cache.
- `--config <PATH>` to override the configuration file for the current process.
- `--plugin-dir <PATH>` to choose where packaged plugins are discovered.

Run `eov --help` or `eov dataset patches --help` for the complete, version-specific help text.

### Dataset patches

Extract a deterministic grid of full image tiles from one or more slides or directories:

```bash
eov dataset patches path/to/slides/ \
	--out dataset/ \
	--tile-size 512 \
	--stride 512 \
	--metadata csv \
	--white-threshold 0.8
```

Directories are searched recursively for supported slide formats. Partial edge tiles are skipped. The optional `--metadata csv|json` flag writes per-tile metadata, and `--white-threshold` can omit tiles that are mostly white.

## Supported formats

Format support is provided by OpenSlide and depends on the OpenSlide version available on the host system. Common extensions exposed by `eov` include:

`.svs`, `.tif`, `.tiff`, `.dcm`, `.ndpi`, `.vms`, `.vmu`, `.scn`, `.mrxs`, `.svslide`, `.bif`, and `.czi`.

If OpenSlide can open a file, `eov` should be able to load it.

## Configuration and plugins

By default, `eov` stores its render preferences in `~/.eov/config.toml` and recently opened files in `~/.eov/recent_files.txt`. Set `EOV_CONFIG` or pass `--config <PATH>` to use a different configuration file.

Example configuration:

```toml
render_backend = "gpu"
filtering_mode = "trilinear"
```

Native plugins are discovered from `~/.eov/plugins/` by default. The `--plugin-dir` option can point to another directory. The [annotations plugin](https://github.com/eosin-platform/eov-annotations-plugin) adds point and polygon annotations, while the [gamepad plugin](https://github.com/eosin-platform/eov-gamepad-plugin) adds controller support.

## Links

- [Project website](https://eov.sh)
- [Source repository and full documentation](https://github.com/eosin-platform/eov)
- [OpenSlide](https://openslide.org/)

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/main/LICENSE-MIT), [Apache-2.0](https://github.com/eosin-platform/eov/blob/main/LICENSE-APACHE-2.0), or [GPLv3](https://github.com/eosin-platform/eov/blob/main/LICENSE-GPLv3), at your option.

The application also depends on OpenSlide and Slint. See the [repository license notes](https://github.com/eosin-platform/eov#dependency-license-notes) for details about those dependencies.
