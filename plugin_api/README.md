# eov-plugin-api

[![Crates.io](https://img.shields.io/crates/v/eov-plugin-api.svg)](https://crates.io/crates/eov-plugin-api)
[![Documentation](https://docs.rs/eov-plugin-api/badge.svg)](https://docs.rs/eov-plugin-api)
[![License](https://img.shields.io/crates/l/eov-plugin-api.svg)](https://github.com/eosin-platform/eov#license)

The API for native plugins of [eov](https://eov.sh), a cross-platform
whole-slide image viewer: a safe Rust SDK, the small layout-checked ABI it
runs on, the JSON protocol between plugin and host, the `plugin.toml`
manifest, and a mock host for unit tests.

## Overview

A plugin is a native library (`cdylib`) that implements the
[`Plugin`](https://docs.rs/eov-plugin-api/latest/eov_plugin_api/sdk/trait.Plugin.html)
trait and exports it with `export_plugin!`. eov loads it from a `.eop`
package, checks that it was built for a compatible API and ABI before any
plugin code runs, and then talks to it through two objects:

- the plugin's object, which receives events (files opened, views changed,
  toolbar clicks, tool gestures, UI callbacks, …) on eov's UI thread with no
  host locks held, and runs viewport filters on render worker threads;
- the [`Host`](https://docs.rs/eov-plugin-api/latest/eov_plugin_api/sdk/struct.Host.html)
  handle, which the plugin may call **from any thread**: queries are answered
  from a published snapshot and changes are queued, so a plugin can never
  block or deadlock eov.

Control messages are JSON ([`protocol`](https://docs.rs/eov-plugin-api/latest/eov_plugin_api/protocol/));
pixels (region reads, rasters, filter frames) cross the boundary as binary
buffers. Every call into a plugin runs inside `catch_unwind`: a panicking
plugin is stopped and reported instead of taking eov down.

## A minimal plugin

```toml
# Cargo.toml
[package]
name = "hello"
version = "1.0.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
eov-plugin-api = "1.0"
```

```rust
use eov_plugin_api::prelude::*;

struct Hello {
    host: Host,
}

impl Plugin for Hello {
    fn create(host: Host, _init: PluginInit) -> Result<Self, String> {
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: "hello".into(),
            tooltip: "Say hello".into(),
            icon_svg: "<svg xmlns='http://www.w3.org/2000/svg'/>".into(),
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        Ok(Self { host })
    }

    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        match event {
            HostEvent::ToolbarButtonClicked { .. } => {
                self.host.notify(NotifyLevel::Info, "Hello from a plugin");
            }
            HostEvent::FileOpened { file } => {
                self.host.info(format!("{} opened", file.filename));
            }
            _ => {}
        }
        Ok(())
    }
}

eov_plugin_api::export_plugin!(Hello);
```

`plugin_api/example-plugin/` in the eov repository is a complete plugin: a
sidebar written in Slint, a point tool whose markers live in an overlay
layer and can be undone, a context-menu item, a HUD button, a CPU filter,
persisted settings and a background job. It is also used as the host's
end-to-end test.

## What the host offers

The `Host` methods, grouped (see the API docs for every type):

| Area | Methods |
|---|---|
| Feedback | `log`/`info`/`warn`/`error`, `notify`, `notify_error`, `set_status`, `start_job` (status-bar progress with cancel) |
| State | `snapshot` (panes, views, files, focus), `file_info`, `file_properties`, `roi`, `measurements`, `subscribe` (limit the events you receive) |
| Pixels | `read_region` (any level, straight or premultiplied alpha, through a plugin-private handle), `read_associated_image` (label, macro, thumbnail) |
| Overlays | `create_layer` → `Layer`: `upsert`, `remove_objects`, `clear`, `select`, `highlight`, `put_raster` (split large rasters into chunk objects to update parts of them), `set_visible`, `set_visible_in_pane`, `set_min_zoom`, `update` |
| UI | `add_toolbar_button`, `add_hud_button`, `register_tool` (host-drawn shapes, point placement, momentary hotkeys, custom tools with raw pointer and key input; `ToolSpec.keys` claims only the keys a tool uses, leaving the rest to eov), `register_hotkey`, `set_badge` (notices fixed to the panes), `open_surface` (sidebars, modals and windows from `.slint` files with properties and callbacks), `confirm`, `open_file_dialog`, `save_file_dialog`, `copy_text` |
| Navigation | `set_view`, `set_view_orientation`, `pan`, `zoom`, `fit`, `frame`, `display_adjustments`, `cycle_pane`, `cycle_tab`, `series`, `open_file`, `request_render` |
| Undo | `push_undo`, `clear_undo` (one undo stack shared with eov, scoped by slide; you get `Undo`/`Redo` events) |
| Filters | `register_filter`, `set_filter_enabled` (CPU RGBA filters run in the render path) |
| Settings | `setting`, `set_setting`, `remove_setting` (persisted per plugin) |
| Models | `fetch_model` (a model from eov's model store, downloaded and verified when it is not there yet; the answer arrives as `HostEvent::ModelFetched`), `resolve_model`, `models` |
| Batch runs | `register_batch_command` (offered by `eov batch run <plugin>:<command> SLIDES --out DIR`; each slide arrives as `HostEvent::BatchRun` with its output folder and parameters), `batch_progress`, `finish_batch` (status, the files written and a summary for the run manifest); `PluginInit.headless` says when there is no window |

**Models.** eov keeps the models of AI plugins in one store per user,
by the SHA-256 of each model's main file, so a model is downloaded once
for every plugin and checked against the hash you pin.
`fetch_model(ModelFetchArgs { url, sha256, size, file_name, name,
version, companions })` names the main file by its SHA-256 (and size,
when known: a download stops as soon as it is larger) and lists files
that must sit next to it, such as an ONNX model's external weights
(`ModelFileSpec`: a SHA-256, and a URL relative to the main file's or
just a `file_name` in the same folder). A model already stored is
answered at once; otherwise eov asks the user, naming your plugin, the
model and the servers, downloads it with progress in the status bar
(resuming an interrupted download), verifies every file and answers
`ModelFetched { request_id, path, model, error, cancelled }`. The
administrator's policy can refuse downloads, restrict them to the
site's servers or provide the site's copy of a file by its SHA-256, in
which case `url` may be `None`. eov tells the user why a download
failed, so treat `error` as state to show, not a message to repeat;
`cancelled` means the user declined. Batch runs never download: they
answer from the store. `resolve_model(sha256)` returns a stored model
(`ModelRecord`: the main file's `path`, its companions, where it came
from, when and for whom) without downloading, and `models()` lists the
store. A host without a model store answers `HostError::is_unsupported`;
let the user choose the file instead.

Surfaces are `.slint` files compiled by eov's own Slint (see `[ui] slint` in
the manifest); the exported component does not need to inherit `Window`.
eov keeps keys typed into a surface's text fields away from its shortcuts
unless `capture_keyboard` is off. One Slint 1.17 pitfall: wrapped `Text`
inside `if` or `for` needs `height: self.preferred-height`, or it is sized
for the wrong width.

Hotkeys (of tools, buttons and `register_hotkey`) cannot take a key eov
uses: the keys in its Keyboard Shortcuts panel (F1), which users can
change in `config.toml`, and Escape, Enter, Delete, Backspace, Tab and the
plain arrows. A refused key arrives as `RegistrationRefused`, so offer
users a way to pick another.

Multichannel slides (fluorescence and multiplexed immunofluorescence
OME-TIFF) list their channels in `FileInfo::channel_names` and their OME
metadata as `ome.*` entries of `file_properties`. `read_region` returns
them as RGBA like any slide: the visible channels composited with the
colours, display windows and gammas the user chose in the Channels panel,
exactly what the viewer shows; when the user changes them, later reads
change too. Raw channel values are not available in plugin API 1.x.

Coordinates are level-0 pixels of the slide (x right, y down). Layer ids are
unique per plugin across all open slides, so include the file id when a
slide gets its own layer (`format!("cells-{file_id}")`). Panes are named by
stable ids that survive reordering.

A pane's view may be turned: `ViewInfo::rotation_deg` (clockwise, any angle
from 0 up to 360; quarter turns are exactly 90, 180 and 270, other angles
are kept to a millionth of a degree) and then `ViewInfo::flipped` (mirrored
left to right). `view.orientation` (`Host::set_view_orientation`) takes any
finite angle and turns the view about the pane centre. Level-0
coordinates, overlays, measurements and the ROI are unaffected; only the
mapping to the pane changes. Use `ViewInfo::image_to_screen` and
`screen_to_image` for pane positions instead of `(x - visible.x) * zoom`
(`visible` is the bounding box of the visible area). Filter frames arrive
as shown on screen, turned; their `FilterView` carries no orientation
(the FFI struct is frozen), so read it from the pane's `ViewInfo` in the
snapshot. They hold the slide's pixels after eov's stain and tone
adjustments, and are transparent outside the slide: eov converts the
colours to sRGB (colour management), composites the background and
sharpens after the filters.

Pixels from `read_region` are the slide's own colours, as scanned. When a
slide has an ICC profile (`FileInfo::has_icc_profile`), eov shows and
exports it converted to sRGB unless the user turns that off;
`DisplayAdjustments::color_managed` (from `display_adjustments`, and in
`HostEvent::DisplayChanged`) says whether a pane does.

## Slide formats

A plugin can read slide formats eov does not read itself, such as Philips
iSyntax or Olympus VSI: declare them in the manifest, so eov knows their
file extensions without running any plugin code, and implement
`Plugin::open_slide`:

```toml
[[slide_formats]]
id = "isyntax"                 # unique within the plugin
name = "Philips iSyntax"       # shown in the file picker
extensions = ["isyntax"]       # without the dot, any case
```

```rust
impl Plugin for MyFormats {
    fn open_slide(&self, request: &SlideOpenRequest) -> Result<Box<dyn SlideReader>, String> {
        // request.format_id names the [[slide_formats]] entry,
        // request.location the file (an absolute path).
        Ok(Box::new(MySlide::open(&request.location)?))
    }
    // create, on_event, … as usual
}

impl SlideReader for MySlide {
    fn info(&self) -> SlideInfo { /* levels, mpp, objective, properties, associated images */ }
    fn read_region(&self, region: &SlideRegion, out: &mut [u8]) -> Result<(), String> {
        /* region.width × region.height RGBA (straight alpha) of level
           region.level, at (region.x, region.y) in that level's pixels */
    }
}
```

Then the file picker, the series bar, folder scans and `eov batch` offer
those files, and opening one asks the plugin. eov's own readers keep the
formats they read: a plugin that claims `tif` is asked only for TIFFs they
cannot open. `SlideInfo::validate` lists what eov refuses (levels that grow,
inconsistent downsamples, oversized images); optional values it cannot use
are ignored.

- **Threads.** eov reads regions from its render workers, several threads at
  once on the same reader, with none of its locks held: a reader is
  `Send + Sync` (guard a decoder that is not with a lock of your own). Reads
  are at most 4096 × 4096 pixels and lie inside their level; areas outside
  the slide never reach the plugin.
- **Failures.** An `Err` fails that read (the tile is tried again later); a
  panic fails it, makes the plugin's slides fail from then on, and stops the
  plugin, as a panicking event handler does.
- **Where.** Slides open in eov's window and in `eov batch`. `eov dataset`,
  `eov qc`, `eov convert` and `eov probe` do not run plugins, so they report
  such files as unsupported. Only files are opened for now: URI schemes
  (`schemes = ["omero"]`) are accepted in the manifest but not yet offered.
- **Compatibility.** Libraries built before slide sources were added (their
  root module lacks the `slide_source` field) still load, without formats;
  a library with formats loads into an eov without them, which ignores
  `[[slide_formats]]`.
- **Testing.** `TestHarness::open_slide` opens a slide through the same FFI
  path eov uses; the example plugin's checkerboard format
  (`example-plugin`, `.excheck` files) shows the whole thing.

## Plugin manifest

Each package has a `plugin.toml` at its root:

```toml
id = "hello"                        # ^[a-z0-9][a-z0-9_]{0,63}$
name = "Hello"
version = "1.0.0"
description = "Shown in the Plugins panel"
repository = "https://github.com/example/hello"
api = ">=1.0, <2"                   # eov-plugin-api requirement

[environment]
version = ">=1.0.0, <2.0.0"         # eov versions the plugin supports

[icon]                              # optional; or kind = "file" with path = "…"
kind = "svg"
data = "<svg xmlns='http://www.w3.org/2000/svg'/>"

[ui]                                # optional, for plugins with .slint files
slint = ">=1.17"
```

Requirements without an upper bound are treated as `<next major>` so that a
plugin is never loaded into an eov it was not built for. Unknown keys are
ignored, so manifests written for newer hosts still load. Paths are relative
to the plugin root; absolute paths and `..` are rejected.

## Packages

A `.eop` package is a tar archive (optionally zstd-compressed) holding
`plugin.toml`, the library named after the id (`lib<id>.so`,
`lib<id>.dylib` or `<id>.dll`), and any `.slint` files and assets:

```text
hello.eop
├── plugin.toml
├── libhello.so
└── ui/
    └── sidebar.slint
```

Install it with `eov plugin install ./hello.eop`, or copy it into
`~/.eov/plugins/` (or the directory given with `--plugin-dir`). eov extracts
packages into a content-addressed cache and refuses archives with absolute
paths, `..`, links or special files.

## Compatibility

- **API:** plugins built against any `1.x` release load into any eov that
  speaks plugin API 1. Newer methods called on an older host fail with
  `FfiErrorKind::Unsupported`, which the SDK reports as a `HostError` you
  can match on; events a plugin does not know arrive as
  `HostEvent::Unknown` and are ignored.
- **ABI:** the layouts of the root module, the plugin object and the host
  object are checked when the library is loaded. Their method tables are
  frozen for 1.x; new capabilities arrive as JSON methods and events, so a
  plugin built against 1.0 keeps loading into every 1.x eov.
- A different major version, or a library that is not an eov plugin, is
  refused with a message in the Plugins panel; it never runs.

## Testing plugins

`eov_plugin_api::testing` has a `MockHost` that records every call and keeps
layers, surfaces, settings and undo entries in memory, and a `TestHarness`
that creates your plugin against it and sends it events:

```rust,ignore
use eov_plugin_api::testing::TestHarness;

let harness = TestHarness::<Hello>::new().unwrap();
harness.send(HostEvent::ToolbarButtonClicked { button_id: "hello".into() }).unwrap();
assert_eq!(harness.state().notifications.len(), 1);
```

`test_init()` gives each plugin instance its own temporary data and cache
directories, so tests never touch a real eov installation.

The mock keeps a model store too: put models in it with
`state().add_model(record)` (what `resolve_model` and `models` answer),
read the requests your plugin made from `state().model_fetches` (or
`last_model_fetch()`), and answer them as eov would with
`finish_model_fetch(request_id, Ok(record) | Err(message))` or
`cancel_model_fetch(request_id)`.

## Related projects

- [eov](https://github.com/eosin-platform/eov)
- [Annotations plugin](https://github.com/eosin-platform/eov-annotations-plugin)
- [Gamepad plugin](https://github.com/eosin-platform/eov-gamepad-plugin)
- [API documentation](https://docs.rs/eov-plugin-api)

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/master/LICENSE-MIT),
[Apache-2.0](https://github.com/eosin-platform/eov/blob/master/LICENSE-APACHE-2.0),
or [GPLv3](https://github.com/eosin-platform/eov/blob/master/LICENSE-GPLv3), at
your option.
