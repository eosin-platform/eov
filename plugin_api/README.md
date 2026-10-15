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

Surfaces are `.slint` files compiled by eov's own Slint (see `[ui] slint` in
the manifest); the exported component does not need to inherit `Window`.
eov keeps keys typed into a surface's text fields away from its shortcuts
unless `capture_keyboard` is off. One Slint 1.17 pitfall: wrapped `Text`
inside `if` or `for` needs `height: self.preferred-height`, or it is sized
for the wrong width.

Coordinates are level-0 pixels of the slide (x right, y down). Layer ids are
unique per plugin across all open slides, so include the file id when a
slide gets its own layer (`format!("cells-{file_id}")`). Panes are named by
stable ids that survive reordering.

A pane's view may be turned: `ViewInfo::rotation_deg` (clockwise, in steps
of 90°) and then `ViewInfo::flipped` (mirrored left to right). Level-0
coordinates, overlays, measurements and the ROI are unaffected; only the
mapping to the pane changes. Use `ViewInfo::image_to_screen` and
`screen_to_image` for pane positions instead of `(x - visible.x) * zoom`
(`visible` is the bounding box of the visible area). Filter frames arrive
as shown on screen, turned; their `FilterView` carries no orientation
(the FFI struct is frozen), so read it from the pane's `ViewInfo` in the
snapshot.

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

## Related projects

- [eov](https://github.com/eosin-platform/eov)
- [Annotations plugin](https://github.com/eosin-platform/eov-annotations-plugin)
- [Gamepad plugin](https://github.com/eosin-platform/eov-gamepad-plugin)
- [API documentation](https://docs.rs/eov-plugin-api)

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/main/LICENSE-MIT),
[Apache-2.0](https://github.com/eosin-platform/eov/blob/main/LICENSE-APACHE-2.0),
or [GPLv3](https://github.com/eosin-platform/eov/blob/main/LICENSE-GPLv3), at
your option.
