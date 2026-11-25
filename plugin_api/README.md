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
unless `capture_keyboard` is off. eov compiles surfaces without a Slint
edition, so the behavior changes that Slint ties to an edition do not apply
to them. One pitfall that remains: wrapped `Text` inside `if` or `for`
needs `height: self.preferred-height`, or it is sized for the wrong width.
Experimental Slint features are not available to surfaces: eov ignores
`SLINT_ENABLE_EXPERIMENTAL_FEATURES`.

Hotkeys (of tools, buttons and `register_hotkey`) cannot take a key eov
uses: the keys in its Keyboard Shortcuts panel (F1), which users can
change in `config.toml`, and Escape, Enter, Delete, Backspace, Tab and the
plain arrows. A refused key arrives as `RegistrationRefused`, so offer
users a way to pick another; to say why, show its `shown_reason` (the
reason in the interface's language; hosts with plugin API 1.0 send none)
when it is there, else `i18n::isolate(reason)` inside your own translated
text.

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
name = "Philips iSyntax"       # the format's name (not shown yet)
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
  plugin, as a panicking event handler does. Write the errors of
  `open_slide` and of a reader in English, as eov's own readers do: eov
  logs them and hands them to scripts as they are, and shows them to the
  user inside its own translated message.
- **Where.** Slides open in eov's window and in the commands that read
  slides (`eov batch`, `eov dataset patches`, `eov qc`, `eov convert`,
  `eov probe`), which start the installed plugins with formats without a
  window. Only files are opened for now: URI schemes
  (`schemes = ["omero"]`) are accepted in the manifest but not yet offered.
- **Compatibility.** Libraries built before slide sources were added (their
  root module lacks the `slide_source` field) still load, without formats;
  a library with formats loads into an eov without them, which ignores
  `[[slide_formats]]`.
- **Testing.** `TestHarness::open_slide` opens a slide through the same FFI
  path eov uses; the example plugin's checkerboard format
  (`example-plugin`, `.excheck` files) shows the whole thing.

## Translations

eov's interface is available in 31 languages ([`i18n::LANGUAGES`](https://docs.rs/eov-plugin-api/latest/eov_plugin_api/i18n/)),
and a plugin can follow the language the user chose:

1. Declare its catalogs in `plugin.toml`:

   ```toml
   [i18n]
   catalogs = "translations"   # <catalogs>/<tag>/LC_MESSAGES/<domain>.po
   domain = "hello"            # default: the plugin id
   ```

2. Mark texts in `.slint` surfaces with `@tr("…")` (eov compiles surfaces
   with the plugin's catalogs; the default context is the component name),
   and texts made in Rust with `tr!` (in the prelude), which takes the same
   placeholders and plural syntax:

   ```rust
   self.host.notify(NotifyLevel::Info, tr!("Counted {n} cell" | "Counted {n} cells" % count));
   let tooltip = tr!("Toolbar" => "Count cells");
   ```

3. Handle `HostEvent::LanguageChanged { language }`: the SDK has already
   switched `tr!` to the new language; hand eov again what it shows for you
   (toolbar tooltips, tool names, surface properties that hold text).
   Surfaces that declare `in property <bool> eov-rtl;` and
   `in property <string> eov-language;` get the writing direction (Arabic,
   Persian, Hebrew are right to left) and the language's tag.

4. Ship `translations/<tag>/LC_MESSAGES/<domain>.po` in the `.eop`. The
   Plugins panel shows the plugin's `name` and `description` from catalog
   entries with the context `plugin.toml`. Each catalog's plural rule
   (`Plural-Forms`) must be its language's, as `i18n::LANGUAGES` lists
   it: any spelling that gives every number the same form will do, and a
   catalog without one takes the language's. eov leaves a catalog with
   another rule out of the plugin's surfaces (they show in English in that
   language). A text with fewer plural forms than its language has shows
   in English: in surfaces always, in Rust for the numbers whose form is
   missing. `i18n.py update --plugin` writes the right header and as many
   forms as the language has.

**Warnings and errors.** eov writes a plugin's `host.warn`/`host.error`
messages to its log as they are, and shows the same text to the user as a
notification. eov's log stays in English, so tell the user in their
language with `host.notify` or `host.notify_error` and `tr!`, and log in
English with `host.info` or `host.debug`. For a text that is both shown
and logged, `i18n::shown_and_english` writes it in both forms from the same
code:

```rust
let (shown, english) = i18n::shown_and_english(|| tr!("{} failed", i18n::isolate(&name)));
self.host.info(english);
self.host.notify(NotifyLevel::Warning, shown);
```

It runs the closure twice, the second time in English, so the closure
should only make the text. Values made inside it (with `tr!`,
`i18n::isolate`, `i18n::format_decimal`) follow the language; a text made
earlier is inserted as it is, so keep such a text in both forms.
`i18n::in_english(|| …)` runs any code as in English: on the calling
thread only, for Rust code only (surfaces' `@tr` texts keep the
interface's language), until the code returns or panics, and it can be
nested. A plugin that hands translated text to `warn` or `error` has it
shown in the user's language, and accepts that eov's log gets that
language too.

Errors a plugin returns: an `Err` from `Plugin::create` is logged, written
to the audit log and shown in the Plugins panel as it is, so keep it
English (and tell the user with `notify`), as are the errors of
`open_slide` and slide readers (see [Slide formats](#slide-formats)); an
`Err` from `on_event` or a surface callback is shown to the user as a
warning, so translate it.

Overlay labels are also burned into exported images and copies of the
view. Those travel beyond the interface's language, and eov draws their
text with a font that cannot draw every letter: it has most Latin letters
and the Greek alphabet, but not, for example, Romanian ș and ț, Vietnamese
letters with two diacritics or Greek letters with an accent (eov writes
its own labels in English, with `.` decimals). Give a translated label its
English form for them:

```rust
let object = OverlayObject::new("region", shape)
    .with_label(tr!("Region · {} mm²", i18n::format_decimal(area, 2)))
    .with_export_label(format!(
        "Region · {} mm²",
        i18n::format_decimal_for(i18n::source(), area, 2)
    ));
```

That is the English text filled in the same way, without `i18n::isolate`;
labels that are names users wrote stay as they are. A `Shape::Text` is
burned in as it is, so put translated text in labels. Hosts with plugin
API 1.0 ignore `export_label`.

eov's `tools/i18n/i18n.py update --plugin <dir>` extracts the texts into
`translations/<domain>.pot` and merges every catalog. It reads the
`@tr` texts of every `.slint` file in the plugin's folder (with
`slint-tr-extractor`; not in `target/`, hidden folders, or the `tests/`,
`benches/` and `examples/` folders of its crates), the `tr!` texts of the
module tree of the plugin's crate (with `xtr`, from `[lib] path` in its
`Cargo.toml`, else `src/lib.rs`), and the manifest's texts. It warns about
Rust files that call `tr!` outside that tree: the texts of a plugin's
other crates are not extracted. `i18n::check_catalog` checks a catalog in
tests. A missing catalog or text shows in English. Insert file names and
paths into translated texts with `i18n::isolate`, show decimal numbers
with `i18n::format_decimal`, read numbers users type with
`i18n::parse_decimal`, put separately translated sentences together with
`i18n::join_sentences`, and never translate log messages (at any level),
file contents or data from slides. eov hosts with plugin API 1.0 send no
language: translated plugins run there in English.
See [docs/translating.md](https://github.com/eosin-platform/eov/blob/master/docs/translating.md)
for the whole workflow.

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

[i18n]                              # optional: translations (see Translations)
catalogs = "translations"
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
├── ui/
│   └── sidebar.slint
└── translations/                 # with [i18n]; see Translations
    ├── de/LC_MESSAGES/hello.po
    └── ja/LC_MESSAGES/hello.po
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
- **Source:** minor releases add public fields to protocol and manifest
  structs and to event variants (1.1: `PluginInit::language`,
  `DisplayAdjustments::shown_badge` and `stain_colors`,
  `OverlayObject::export_label`, `HostEvent::RegistrationRefused`'s
  `shown_reason`, `PluginManifest::i18n`). Build such structs with
  `..Default::default()`, `..OverlayObject::new(..)` or
  `..testing::test_init(..)` (or with their constructors and `with_*`
  methods), and match structs and event variants with `..`, so that a
  minor update never stops your plugin compiling. (`PluginManifest` has a
  `Default` since 1.1; `PluginManifest::from_toml` reads and validates a
  manifest.) Minor releases may also add languages: `i18n::LANGUAGES` is a
  slice (iterate `i18n::LANGUAGES`, not `&LANGUAGES`), and
  `i18n::Language` is `#[non_exhaustive]`, so read its fields rather than
  build or destructure one.

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
directories, so tests never touch a real eov installation. Unlike eov, the
mock does not turn `host.warn`/`host.error` into notifications: find them
in `calls_to("log")` by their `level`.

**Languages in tests.** The language `tr!` translates with belongs to the
whole plugin library: creating any harness sets it from
`PluginInit::language` (`test_init` gives English), and
`HostEvent::LanguageChanged` switches it for every test in the binary.
Since tests run in parallel, run the tests that switch languages in an
integration-test binary of their own (`tests/language.rs`), one at a time
behind a lock, as eov's own plugins do. Alternatively, the
`thread-local-language` feature on the dev-dependency
(`eov-plugin-api = { version = "1.1", features = ["thread-local-language"] }`
under `[dev-dependencies]`) gives each test thread its own language; texts
your plugin makes on threads of its own are then in English, since new
threads start in English.

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
