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

The Cargo build requires OpenSlide 4 and the native dependencies used by Slint and the platform graphics stack. Where the distribution packages OpenSlide 4 (for example Debian 13 or Homebrew), install its development package:

```bash
sudo apt-get install libopenslide-dev   # Debian/Ubuntu
brew install openslide                  # macOS
```

Older distributions ship OpenSlide 3; build OpenSlide 4 with the repository's [`packaging/shared/build-openslide.sh`](https://github.com/eosin-platform/eov/blob/master/packaging/shared/build-openslide.sh) and point `PKG_CONFIG_PATH` (or `OPENSLIDE_LIB_DIR`) at it. You may also need the usual X11 or Wayland, Vulkan, font, and `pkg-config` development packages for your distribution.

A Cargo installation updates itself with `eov update` (which runs `cargo install eov --version ...`) when the running binary is the one Cargo installed.

For prebuilt AppImage, Flatpak, macOS, and Windows packages, see the [latest releases](https://github.com/eosin-platform/eov/releases/latest).

## Usage

Open one or more slides. They open as tabs of one pane, in order, with the first one selected:

```bash
eov slide.svs
eov slide1.svs slide2.svs slide3.svs
```

`--pane` opens more panes side by side, and `--goto`/`--zoom` open the first slide at a place (level-0 pixels, or micrometres with `um`) and magnification (`20x`, a zoom factor such as `0.5`, or `fit`):

```bash
eov he.svs --pane ki67.svs
eov slide.svs --goto 300um,200um --zoom 20x
eov 'eov://open?path=%2Fslides%2Fslide.svs&x=300um&y=200um&zoom=20x'
```

An argument that exists is taken as it is, commas included; one that does not but contains commas is a list of files (`a.svs,b.svs`). `file://` URIs are decoded. Files that cannot be opened are reported in the window, and the rest still opens. While eov runs, `eov FILES` hands the files to it, which opens them as tabs and comes to the front, instead of starting a second eov; `--new-instance`, `single_instance = false` in the config and options such as `--cpu` or `--debug` start a separate one. `eov://open?path=…&x=…&y=…&zoom=…` links (percent-encoded path) work like `--goto` and `--zoom`; on Windows, links to network paths are only followed for servers listed in `trusted_link_servers`.

`eov register-file-types` makes the file manager offer eov for slide files and use it for `eov://` links, for the current user: the registry on Windows, `~/.local/share` on Linux (for a `cargo install` or the AppImage). `eov unregister-file-types` undoes it. The macOS app and the Flatpak declare their file types themselves.

The command-line interface also provides inspection and dataset utilities:

```text
eov [OPTIONS] [FILES]... [--pane FILES]... [--goto X,Y] [--zoom ZOOM]
eov probe [--json] <FILE>
eov recent list
eov config-path
eov paths [--json]
eov register-file-types|unregister-file-types [--dry-run]
eov dataset patches <INPUTS>... --out <DIR> --tile-size <PX> --stride <PX>
eov update [--release <VERSION>] [--app-only] [--from <DIR>] [--rollback] [--dry-run] [--json] [--allow-network] [-y]
eov plugin ls|list [--json] [--allow-network]
eov plugin install|add <SOURCE> [--force] [--dry-run] [--json] [--allow-network] [-y]
eov plugin update|upgrade <SOURCE> [--force] [--dry-run] [--json] [--allow-network] [-y]
eov plugin remove|rm|uninstall <PLUGIN|FILE> [--dry-run] [--json] [-y]
eov plugin info|inspect <PLUGIN> [-o plain|toml|json]
eov plugin disable|enable <PLUGIN>
eov plugin lock [-o <FILE>]
eov plugin sync <LOCKFILE> [--from <DIR>] [--prune] [--dry-run] [--allow-network] [-y]
eov plugin gc
eov call <METHOD> [PARAMS] [--out <FILE>]
eov script-endpoint [--json]
```

Useful options include:

- `--backend auto|cpu|gpu` to select the rendering backend.
- `--cpu` and `--gpu` as shortcuts for `--backend cpu` and `--backend gpu`.
- `--filtering-mode auto|bilinear|trilinear|lanczos` to select texture filtering.
- `--debug` to enable debug overlays.
- `--log-level error|warn|info|debug|trace` to control logging.
- `--cache-size <MB>` and `--max-tiles <COUNT>` to tune the tile cache (at most 1 TiB and 16777216 tiles).
- `--config <PATH>` to override the configuration file for the current process (a leading `~` is expanded).
- `--plugin-dir <PATH>` to choose where packaged plugins are discovered.
- `--window-width`, `--window-height` and `--window-x`/`--window-y` (given together; Wayland ignores positions).
- `--new-instance` to start a separate eov while one is running.
- `--safe-mode` (or `EOV_SAFE_MODE=1`) to start without loading any plugin.

`eov probe` prints a slide's size, levels, pixel size, objective, stain, bounds, ICC profile size, associated images, quickhash and every OpenSlide property (`--json` for scripts). `eov paths` lists every file and folder eov keeps data in, with their sizes.

Scripts drive the running eov through its local, authenticated endpoint (a per-user Unix socket or localhost port; nothing listens on the network): `eov call slides`, `eov call region.read '{"x": 0, "y": 0, "width": 512, "height": 512}' --out tile.png`, or the dependency-free Python client in the repository's `clients/python`. They open slides, move views, read regions, set the ROI, add measurements and bookmarks, quantify stains and export views; `scripting = false` in the config turns it off. See "Scripting" in the [full documentation](https://github.com/eosin-platform/eov#scripting).

Run `eov --help` or `eov dataset patches --help` for the complete, version-specific help text.

Update commands are always explicit. eov makes no network requests during normal startup. GitHub metadata requires interactive consent or `--allow-network`; `-y` only skips the confirmation of the planned changes, and end of input counts as "no". `--dry-run` and `--json` show the plan without changing anything. Official plugin versions and hashes are pinned by the installed or target eov release manifest, and a signed manifest is verified against release keys built into eov. `eov update --rollback` restores what the last update replaced, `eov update --from <dir>` installs a verified offline bundle, and administrators can restrict updates and plugins with a policy file (`/etc/eov/policy.toml`, `/Library/Application Support/eov/policy.toml` or `%ProgramData%\eov\policy.toml`; see the [repository documentation](https://github.com/eosin-platform/eov#application-and-plugin-updates)). Proxies (`HTTPS_PROXY`, `NO_PROXY`) and the operating system's certificate store are used. Self-update metadata support begins at `v0.4.5`.

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

By default, `eov` stores its preferences in `~/.eov/config.toml`, recently opened files in `~/.eov/recent_files.toml` (`remember_recent_files = false` in the config turns the history off) and the last session in `~/.eov/session.toml`; `eov paths` lists every location. Set `EOV_CONFIG` or pass `--config <PATH>` to use a different configuration file.

Example configuration:

```toml
render_backend = "gpu"
filtering_mode = "trilinear"
single_instance = true                  # false: every launch starts its own eov
scripting = true                        # false: refuse every call from scripts
trusted_link_servers = ["slides01"]     # Windows: network paths eov:// links may open
```

Native plugins are discovered from `~/.eov/plugins/` by default. The `--plugin-dir` option can point to another directory. The [annotations plugin](https://github.com/eosin-platform/eov-annotations-plugin) adds point and polygon annotations, while the [gamepad plugin](https://github.com/eosin-platform/eov-gamepad-plugin) adds controller support. Manage them with `eov plugin` (install, update, remove, disable, enable, lock and sync); packages are `.eop` files (tar, optionally zstd-compressed) that eov extracts into a content-addressed cache.

## Links

- [Project website](https://eov.sh)
- [Source repository and full documentation](https://github.com/eosin-platform/eov)
- [OpenSlide](https://openslide.org/)

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/master/LICENSE-MIT), [Apache-2.0](https://github.com/eosin-platform/eov/blob/master/LICENSE-APACHE-2.0), or [GPLv3](https://github.com/eosin-platform/eov/blob/master/LICENSE-GPLv3), at your option.

The application also depends on OpenSlide and Slint. See the [repository license notes](https://github.com/eosin-platform/eov#dependency-license-notes) for details about those dependencies.
