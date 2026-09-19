# eov-plugin-api

[![Crates.io](https://img.shields.io/crates/v/eov-plugin-api.svg)](https://crates.io/crates/eov-plugin-api)
[![Documentation](https://docs.rs/eov-plugin-api/badge.svg)](https://docs.rs/eov-plugin-api)
[![License](https://img.shields.io/crates/l/eov-plugin-api.svg)](https://github.com/eosin-platform/eov#license)

Shared API and ABI definitions for native plugins for [eov](https://eov.sh), a
cross-platform whole-slide image viewer.

This crate is intended for plugin authors and host integrations. It does not
load or package plugin binaries itself. The eov application discovers plugin
packages, validates their manifests, loads their native libraries, and passes
host callbacks through the contracts defined here.

## Add the dependency

Keep the crate version aligned with the eov host version that will load the
plugin:

```toml
[dependencies]
eov-plugin-api = "0.4.6"
abi_stable = "0.11"
```

Native plugins should build both a dynamic library for eov and an `rlib` when
they also contain Rust tests or reusable internal code:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

## What this crate provides

- `eov_plugin_api::ffi` contains the `abi_stable` types used across the native
	plugin boundary, including `PluginVTable`, `HostApiVTable`, toolbar and HUD
	registrations, host snapshots, viewport overlays, and filter callbacks.
- `PluginManifest` and `IconDescriptor` describe and validate `plugin.toml`.
- `Plugin`, `HostContext`, and the related registration types provide a Rust
	plugin abstraction for in-process integrations.
- `ViewportFilter` supports CPU RGBA8 post-processing and GPU filter
	integrations. The FFI module exposes the corresponding ABI-safe types.
- `PluginError` and `PluginResult` provide errors shared by the Rust-level API.

The dynamic-library path uses `eov_plugin_api::ffi`. The Rust-level traits and
the FFI types are related APIs, but they are not interchangeable at the ABI
boundary: exported callbacks must use the `abi_stable` types from `ffi`.

## Plugin manifest

Each plugin package contains a `plugin.toml` at its root. The fields below are
the fields understood by `PluginManifest`:

```toml
id = "example_plugin"
name = "Example Plugin"
version = "0.1.0"
description = "Optional plugin description"
repository = "https://github.com/example/example-plugin"
entry_ui = "ui/example-panel.slint"
entry_component = "ExamplePanel"

[environment]
version = ">=0.4.1"

[icon]
kind = "svg"
data = "<svg viewBox=\"0 0 24 24\">...</svg>"

[[toolbar_buttons]]
button_id = "open_panel"
tooltip = "Open example panel"
action_id = "open_panel"
```

`entry_ui` and `entry_component` are optional for plugins without a Slint
window. Icon files can be used instead of inline SVG data:

```toml
[icon]
kind = "file"
path = "icons/example.svg"
```

UI and icon paths are relative to the plugin root. Absolute paths and `..`
path traversal are rejected by manifest validation. A manifest can be parsed
from a string or file and its referenced files can then be checked:

```rust
use eov_plugin_api::PluginManifest;
use std::path::Path;

fn load_manifest(plugin_root: &Path) -> eov_plugin_api::PluginResult<PluginManifest> {
		let manifest = PluginManifest::from_file(&plugin_root.join("plugin.toml"))?;
		manifest.validate_files(plugin_root)?;
		Ok(manifest)
}
```

## Native plugin entry point

Native plugins are loaded through `abi_stable`. A plugin dynamic library
exports the following symbol and returns a fully populated `PluginVTable`:

```rust,ignore
use eov_plugin_api::ffi::PluginVTable;

#[unsafe(no_mangle)]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
		make_plugin_vtable()
}
```

The vtable covers plugin initialization, toolbar and HUD actions, callbacks
from plugin UI, viewport context-menu actions, viewport overlays, annotation
events, undo/redo, and CPU/GPU viewport filters. `HostApiVTable` is supplied by
the host and lets a plugin query application state, read slide regions, open
files, move the active viewport, request rendering, update controls, and write
host log messages.

For example, a toolbar callback uses ABI-safe strings and vectors:

```rust
use abi_stable::std_types::{ROption, RVec};
use eov_plugin_api::ffi::ToolbarButtonFFI;

fn toolbar_buttons() -> RVec<ToolbarButtonFFI> {
		RVec::from(vec![ToolbarButtonFFI {
				button_id: "open_panel".into(),
				tooltip: "Open example panel".into(),
				icon_svg: "<svg viewBox=\"0 0 24 24\">...</svg>".into(),
				action_id: "open_panel".into(),
				tool_mode: ROption::RNone,
				hotkey: ROption::RNone,
		}])
}
```

Do not pass ordinary Rust `String`, `Vec`, or other Rust-owned values across
the exported ABI. Use the stable types in `eov_plugin_api::ffi` and
`abi_stable::std_types` instead.

## Runtime and packaging

The eov host normally discovers `.eop` plugin packages in `~/.eov/plugins/`.
Use `--plugin-dir` to select another directory. A package contains the
manifest, the platform-specific plugin library, and any UI or icon files it
references. Packaging and installation are host-level concerns; see eov's
[plugin documentation](https://github.com/eosin-platform/eov#plugins) for the
current package layout and loading workflow.

The plugin system is experimental. Build a plugin against the same
`eov-plugin-api` release used by the target eov host, and treat changes to the
vtable or manifest contract as release compatibility changes.

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
