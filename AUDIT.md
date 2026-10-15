# eov audit: bugs, edge cases, unimplemented features, architecture and pathology-workbench gaps

| | |
|---|---|
| **Date** | 2026-10-15 |
| **Revision audited** | `eov` master @ `e54b16c` ("add cellaris plugin"), version 0.4.9 |
| **Submodules** | `plugins/annotations` @ `293ef43` (0.2.4) · `plugins/gamepad` @ `59770dc` (0.2.5) · `plugins/cellaris` @ `23290b2` (0.1.0) · `plugins/eovae/HistoVAE` @ `480e979`; `plugins/eovae` 0.1.0 is in-tree |
| **Size** | ≈ 39k lines app · 7.6k common · 1.6k plugin_api · 8.8k annotations · 8.1k eovae · 4.4k gamepad · 2.3k cellaris (Rust + Slint) |
| **Toolchain used** | Rust 1.98.1 · OpenSlide 4.0.0 · Linux x86_64 |

## Contents
1. [Summary](#1-summary)
2. [Architectural assessment](#2-architectural-assessment) (cross-cutting, with a proposed plugin-API redesign)
3. [Plugin API and host-side plugin runtime](#3-plugin-api-and-host-side-plugin-runtime) (`PLG-*`)
4. [Core WSI access, tiling and caching](#4-core-wsi-access-tiling-and-caching-common) (`WSI-*`)
5. [Rendering pipeline](#5-rendering-pipeline-appsrcrenderrs-gpurs-tile_loaderrs-backendrs) (`RND-*`)
6. [Colour science, image export and dataset extraction](#6-colour-science-image-export-and-dataset-extraction) (`CLR-*`, `EXP-*`, `DS-*`)
7. [Application state, tools and interaction](#7-application-state-tools-and-interaction-appsrcstaters-callbacksrs-toolsrs-ui_updaters-pane_uirs-appuislint) (`UI-*`)
8. [Shell, CLI, configuration, persistence and docs](#8-application-shell-cli-configuration-persistence-and-documentation) (`SH-*`)
9. [Installation, self-update, plugin management, release engineering](#9-installation-self-update-plugin-management-and-release-engineering) (`UPD-*`)
10. [Plugin: annotations](#10-plugin-annotations-pluginsannotations-v024-git-submodule) (`ANN-*`)
11. [Plugin: eovae](#11-plugin-eovae-pluginseovae-in-tree-workspace-member-model-from-the-histovae-submodule) (`VAE-*`)
12. [Plugin: cellaris](#12-plugin-cellaris-pluginscellaris-added-in-e54b16c-not-a-workspace-member) (`CEL-*`)
13. [Plugin: gamepad](#13-plugin-gamepad-pluginsgamepad-v025-git-submodule) (`GPD-*`)
14. [Features pathologists and WSI users would expect](#14-features-pathologists-and-other-wsi-workbench-users-would-expect)
15. [Tests, CI and build health](#15-tests-ci-and-build-health)
16. [Suggested remediation roadmap](#16-suggested-remediation-roadmap)
17. [Appendix: method and verification](#17-appendix-method-and-verification)

## How to read this document

Each finding has an id (`AREA-n`), a severity, a confidence, its locations (`path:line` at the audited revision), the problem, its impact, and a suggested fix. Low-severity items are grouped into one bundled entry per area.

**Severity:**

| Level | Meaning |
|---|---|
| **CRITICAL** | Data loss or corruption, a hang or crash on a common path, memory unsafety reachable in normal use, or a security hole. |
| **HIGH** | Wrong results a user would rely on (measurements, colours, coordinates, predictions); a crash on a plausible edge case; a broken feature. |
| **MEDIUM** | Edge-case bugs, leaks, performance problems users will notice, or architecture that blocks extension. |
| **LOW** | Minor UX issues, code quality, documentation drift, test gaps. |

**Confidence:**

| Level | Meaning |
|---|---|
| **Confirmed** | Traced through the code, and often reproduced with a scratch program or a numerical check. |
| **Likely** | The code path is confirmed, but the runtime effect depends on a platform or data that wasn't available (Windows/macOS runtime, MIRAX/DICOM data, real Homebrew output). |
| **Speculative** | Plausible, but not verified. |

Findings marked *re-verified* were checked again directly in the source by the lead reviewer, in addition to the auditor who found them (§17).

---

## 1. Summary

eov is a capable, fast core viewer. Its f64 viewport math, bordered tiles, GPU trilinear and Lanczos paths, hardened `.eop` extraction and an architecturally enforced network-consent gate show careful engineering, and clippy is clean.

The audit nevertheless found **5 distinct critical defects**: two deterministic UI deadlocks, silent corruption of ML datasets, a self-update that can delete a user's folder, and an unchecked native plugin ABI. It also found about **45 distinct high-severity issues**, many of which put wrong numbers or wrong colours in front of a user who would trust them.

Most defects trace back to a few structural choices:

1. One global `RwLock<AppState>` held across rendering, UI rebuilds, slide I/O and plugin callbacks.
2. Pane identity is an index, and pointer events don't carry a pane id.
3. Host and plugins poll each other instead of exchanging events.
4. The plugin ABI is a flat, unversioned, annotation-shaped bag of mandatory function pointers, passed by value, with no plugin instance and no threading contract.

§2 proposes a target design and a migration order.

As a *pathology workbench*, eov is missing table-stakes features:

- the slide label and macro image;
- rotation;
- magnification presets and keyboard navigation;
- linked navigation across different slides;
- a real measurement suite (area, perimeter, calibration);
- session restore and bookmarks;
- ICC colour management;
- annotation interchange (GeoJSON/QuPath);
- a counting tool and HPF reticle;
- support for multiplex fluorescence.

Several of these are blocked by plugin-API limitations, not just unimplemented (§14).

### 1.1 Findings by area

| Area | Critical | High | Medium | Low (bundled) |
|---|---|---|---|---|
| Plugin API and host runtime (`PLG`) | 2 | 4 | 11 | 2 |
| Core WSI, tiles, cache (`WSI`) | – | 4 | 10 | 1 |
| Rendering (`RND`) | – | 4 | 5 | 1 |
| Colour, export, dataset (`CLR`/`EXP`/`DS`) | 1 | 6 | 11 | 2 |
| App state and interaction (`UI`) | – | 6 | 9 | 1 |
| Shell, CLI, config, docs (`SH`) | 1 | 1 | 13 | 5 |
| Install, update, packaging (`UPD`) | 1 | 5 | 8 | 1 |
| Annotations plugin (`ANN`) | – | 3 | 14 | 1 |
| eovae plugin (`VAE`) | 1¹ | 9 | 6 | 1 |
| Cellaris plugin (`CEL`) | – | 5² | 5 | 1 |
| Gamepad plugin (`GPD`) | – | 3 | 4 | 1 |
| **Total (169 entries)** | **6** | **50** | **96** | **17** |

¹ VAE-1 is the plugin side of PLG-2. ² CEL-1 and CEL-2 restate PLG-4 and PLG-2 from the plugin's side. A few other HIGH entries are cross-references in the same way (UPD-6 ≈ PLG-1, GPD-3 ≈ UPD-5).

### 1.2 Fix first

| # | Id | Sev. | Problem | Who hits it |
|---|---|---|---|---|
| 1 | **SH-1** | CRITICAL | Re-opening a slide that is already open deadlocks the UI. A read guard is held while `request_render_loop` takes the write lock (`app/src/file_ops.rs:480-492`). *Re-verified.* | Anyone who clicks the top "Open Recent" entry, re-drops a slide, or runs `eov a.svs,a.svs`. |
| 2 | **PLG-2** | CRITICAL | Filter-state sync calls into plugins under the `AppState` write lock. eovae's `set_filter_enabled` calls `request_render`, which re-locks and hangs forever. *Re-verified.* | Users with eovae plus any plain-action toolbar button (cellaris toggle, gamepad window close, a failed sidebar). |
| 3 | **DS-1** | CRITICAL | Multi-slide dataset extraction reads every slide after the first through the *first* slide's OpenSlide handle. A persistent rayon pool plus a `thread_local!` that is never reset (`common/src/dataset/pipeline.rs:418`, `:519-538`). *Re-verified.* | Every `eov dataset patches a b …` run and every GUI export of more than one slide. Silently corrupted training data. |
| 4 | **UPD-1** | CRITICAL | Windows portable self-update renames the folder that contains `eov.exe` and then `remove_dir_all`s the original. The zip has no top-level folder (`app/src/update_helper.rs:240-294`). *Re-verified.* | Users who extracted eov into Downloads, the Desktop or a shared tools folder. Permanent loss of that folder's contents. |
| 5 | **PLG-1** | CRITICAL | The plugin ABI is loaded with no layout or version check, and both vtables cross by value. The ABI already changed in patch release 0.4.9. *Re-verified.* | Any plugin built against a different `eov-plugin-api` 0.4.x (cellaris floats on `"0.4"`). Undefined behaviour or a crash. |
| 6 | **UI-1** | HIGH | Point and polygon clicks in an unfocused split pane are saved on the *focused* pane's slide. *Re-verified.* | Anyone annotating in split view. |
| 7 | **CLR-1…4** | HIGH | Stain normalisation is re-fitted from whatever is visible, so colours change as you pan. Glass is recoloured purple. Deconvolution after normalisation uses the wrong inverse, and any slider change recolours RBCs, pigment and DAB. | Anyone using stain normalisation or deconvolution. |
| 8 | **UI-5 / UI-6** | HIGH | Measurements use the mean MPP, or zoom-dependent screen pixels when MPP is missing. The scale bar labels 25 µm as "0.03 mm". *UI-6 re-verified.* | Anyone measuring tumour size or margins. |
| 9 | **WSI-2** | HIGH | Transparent "no data" regions are never composited onto a background: dark in the viewer, black in JPEG exports, and written into datasets. | MIRAX, BIF, Philips and sparse-DICOM users. |
| 10 | **WSI-3** | HIGH | The thumbnail decodes the whole coarsest level on the UI thread under the write lock, which means multi-GB allocations for single- and two-level slides, including every sibling in a browsed folder. *Re-verified.* | Users of flat TIFF, DICOM or "extracted region" files. |
| 11 | **RND-4 / SH-2** | HIGH | No startup without Vulkan or Metal (VDI, RDP, VMs), and on Windows every CLI and startup error is invisible. | Hospital IT environments and Windows CLI users. |
| 12 | **RND-1 / RND-3 / RND-2** | HIGH | The render loop never idles. HiDPI is ignored, giving half-resolution imagery. GPU texture arrays overflow and silently drop draws. *RND-1 re-verified; RND-2's 128 MiB budget re-verified, drop counts simulated.* | Everyone, especially laptop and Retina users. |
| 13 | **UI-4** | HIGH | `status-text` is never rendered, so open, export and clipboard errors are invisible. *Re-verified.* | Everyone. |
| 14 | **PLG-4 / PLG-5** | HIGH | The newest plugin (cellaris) can't load (missing `[environment]`). The overlay API copies every object every frame under the global lock, so cell-level overlays are not viable. *PLG-4 re-verified.* | Plugin users and authors. |
| 15 | **ANN-1 / ANN-2 / ANN-3** | HIGH | Annotation writes are one fsync'd transaction per row on the UI thread (a 500-vertex polygon takes 4 s). The density panel goes stale after edits, and area is wrong for self-intersecting polygons. | Annotation and mitosis-counting users. |
| 16 | **GPD-1 / GPD-2** | HIGH | The gamepad worker deadlocks against the UI thread. Controller input isn't focus-gated, and the default right stick injects **system-wide** pointer motion. *Re-verified.* | Gamepad users, including in other apps. |
| 17 | **EXP-1 / EXP-3** | HIGH / MED | High-DPI exports silently drop full-resolution tiles beyond 4096. The JPEG quality setting is ignored, and no DPI or MPP metadata is written. *EXP-3 re-verified.* | Anyone preparing figures. |
| 18 | **VAE-3 / VAE-4 / VAE-5 / VAE-6** | HIGH | eovae analyses the wrong pyramid level on Aperio slides, mis-normalises the model's input and output, mixes results across slides, and keeps 448 KiB per tile forever. *VAE-3 and VAE-5 re-verified.* | eovae users. |
| 19 | **CEL-3 / CEL-4** | HIGH | Cellaris ignores MPP, so predictions on 20× slides are out of distribution, and its runtime loading works only on Windows. | Cellaris users. |
| 20 | **UPD-2 … UPD-5** | HIGH | macOS update aborts without `brew`; Homebrew ownership detection is broken; the macOS helper links a CI path; the Flatpak has no network and no input devices. | macOS and Flatpak users. |

### 1.3 Strengths worth preserving
- **Viewport math.** f64 throughout, with exact float blit rectangles and 1-px tile borders, so there is no zoom jitter and no bilinear seams.
- **Network-consent gate.** Updates use a `NetworkPermission` token that every HTTP call requires, manual redirect handling with an HTTPS allow-list, SHA-256 verification before any mutation, and no telemetry.
- **Hardened `.eop` extraction.** It rejects traversal, symlinks, hardlinks and devices, and inspection never loads native code.
- **Deterministic behaviour where it matters.** Dataset grids are deterministic, stale async results are rejected via revision counters, and CPU render jobs are cancelled cooperatively.
- **A clean, readable codebase.** No clippy warnings and meaningful module boundaries, with 189 passing tests once OpenSlide can be linked.

---

## 2. Architectural assessment

The core viewer is fundamentally sound. f64 viewport math, bordered tiles, exact float blit rectangles, a careful network-consent design and hardened archive extraction are all good engineering. Most defects in this audit trace back to a small number of structural decisions. Fixing those decisions removes whole classes of bugs, which is cheaper than fixing the bugs one at a time.

### 2.1 One global lock, held across everything

`Arc<parking_lot::RwLock<AppState>>` is the single source of truth for the UI callbacks, the 60 Hz render timer, the plugin host and plugin worker threads. Hot paths hold it for long, open-ended spans:

- **Rendering and UI rebuild.** The render path and `update_tabs` hold the **write** lock while rebuilding the whole UI model, which includes calling into every plugin (`app/src/render.rs:542`, `:728`, `:822-823`).
- **Slide I/O.** Opening a slide and generating its thumbnail run under the write lock (`app/src/file_ops.rs:501-589`).
- **Plugin reads.** Plugin `read_region` holds the read lock while OpenSlide decodes (`app/src/plugin_host.rs:1171-1187`).
- **Locks that aren't re-entrant.** `parking_lot` locks are not re-entrant and are task-fair: a queued writer blocks new readers. Any nested acquisition on one thread therefore deadlocks, and any long hold stalls every other thread.

The audit found **five independent deadlocks**, all following from this design:

1. Re-opening an already-open file (SH-1).
2. The toolbar filter sync re-entering through eovae (PLG-2).
3. eovae's worker/filter lock-order inversion (VAE-2).
4. The gamepad worker blocking on the UI thread while holding its mutex (GPD-1).
5. Latent: the `stats`/`current_bytes` inversion in the tile cache (WSI-11).

**Direction:**
- Hold short critical sections that copy data out, then work without the lock.
- Never call plugins or Slint while holding host locks.
- Publish immutable snapshots (for example an `ArcSwap<HostSnapshot>`) for readers.
- Make every cross-thread host request post-and-return.
- In debug builds, add a `debug_assert!` that `request_render_loop` and the plugin dispatch helpers are entered with no lock held.

### 2.2 Work on the UI thread

These all run synchronously on the Slint event loop:

- **Opening slides.** This includes full-level thumbnail reads, which can mean multi-GB allocations (WSI-3).
- **Folder scans.** Directory enumeration plus `canonicalize` on every entry (UI-14).
- **Image export** at high DPI (EXP-2).
- **Stain-matrix fitting** on every tile-load epoch (CLR-1, CLR-7).
- **CPU-backend compositing** while panning.
- **Plugin CPU filters** (PLG-12, CEL-8).
- **Plugin `.slint` compilation** on every open (PLG-15).
- **Full-file SHA-256 hashing** of multi-GB slides by the annotations plugin (ANN-14).
- **eovae's CUDA preflight**, which can take up to 20 s (VAE-11).
- **Unpacking a 4 GB plugin package** before the window even appears (SH-12).

The "Loading…" indicator never renders because the event loop is blocked while it is shown. Moving this work onto workers is a prerequisite for a responsive app on network shares (SMB/NFS), which is how slides are usually stored in hospitals.

### 2.3 Pane identity is "index into a Vec", and many callbacks don't say which pane they're for

- **State indexed by position.**
  - `OpenFile.pane_states: Vec<Option<FilePaneState>>` is indexed by pane position.
  - CPU render jobs are keyed by pane index (`app/src/render_pool.rs:114`).
  - Per-pane Slint models and caches live in thread-local vectors.
  - Inserting or removing a pane shifts all of these.
- **Callbacks without a pane id.** Viewport pan, zoom, tool and minimap callbacks carry no pane id; Rust falls back to `state.focused_pane`. Slint only updates that focus before *some* events.
- **Resulting bugs:**
  - annotation clicks in one pane are written to the other pane's slide (UI-1);
  - drag-to-split produces duplicate tab ids (UI-2);
  - CPU jobs are orphaned after a split (RND-9);
  - hover is hit-tested against the wrong pane (UI-16);
  - ROIs and measurements are shared between panes (UI-13).
- **Direction:** give panes stable ids, as tabs already have, and pass the pane id with every pointer event.

### 2.4 Polling instead of events

The host polls plugins:

- overlay getters every frame and on every pointer move;
- plugin window properties every 200 ms;
- filter states after some clicks.

Plugins poll the host in turn: the gamepad does a `get_snapshot`/`set_active_viewport` read-modify-write every 12 ms, and eovae runs progress loops. Nothing is pushed:

- there are no file-opened/closed events;
- there are no viewport-changed events;
- there is no "my filters changed" signal.

The costs:

- The render loop rarely becomes idle (RND-1).
- Plugin caches grow without bound because plugins never learn that a slide was closed.
- Overlay performance scales with the total object count rather than the visible count (PLG-5), which rules out cell-level analysis plugins.

### 2.5 The plugin ABI is a flat bag of mandatory function pointers with no versioning, no instance, and domain-specific vocabulary

PLG-1 to PLG-19 cover the details. The recurring themes are:

- **No version safety.** The ABI is a raw symbol plus by-value structs, so any change is undefined behaviour for existing binaries.
- **No plugin instance.** Callbacks receive no plugin object, so every plugin keeps its state in global `Mutex`es, which are poisoned forever by a single panic.
- **Annotation-shaped API.** Its vocabulary (point/polygon "annotation" tools and overlay structs) stands in for a generic drawing and input API. Every other plugin bends around it: eovae draws grid lines as polygons, and cellaris uses the polygon-annotation tool as a region picker.
- **Hard-coded host behaviour.** Gaps in the API are patched in the host with checks on specific plugin ids and on "magic" property names (PLG-8).
- **No services.** There is no plugin root directory, data directory, settings store, background-job API, notifications, focus state or non-blocking UI dispatch. Each plugin re-invents these, often badly. The gamepad statically links a second, inert copy of Slint, winit and wgpu just to reach an event loop, and cellaris hard-codes `~/.eov/plugins/cellaris`.
- **Shared single-slot UI surfaces** with no ownership checks (PLG-10).
- **A published API that is not implemented.** The GPU filter path (PLG-6), the DMA-BUF descriptor, the Rust `Plugin` trait and the static registry are all part of the documented API, but none of them is actually wired up.

### 2.6 Recommended target design for the plugin system (a single breaking change, e.g. eov 0.5 / plugin API 1.0)

1. **ABI and versioning.**
   - Export an `abi_stable` root module (`#[export_root_module]`), loaded with `RootModule::load_from_file` so that layouts are checked at load.
   - Make `PluginModule` and `HostApi` prefix types (`#[sabi(kind(Prefix))]`), with every optional callback as `Option<extern "C" fn>`.
   - Add an explicit `abi_version` / `struct_size` / capability-bits handshake.
   - Manifests declare `api = ">=1.0, <2"`, derived automatically from the API crate the plugin linked against.
   - Enums cross the boundary as `u32` plus `TryFrom`.
2. **Instances.** `create(host: HostApiRef, ctx: HostCtx) -> RResult<PluginBox, RString>` with `#[sabi_trait]`, so callbacks receive `&self` instead of reaching for globals. `HostCtx` is an opaque, per-plugin, revocable pointer. Add `shutdown()`.
3. **Threading contract.**
   - Every host API is callable from any thread and never blocks on the UI; UI mutations are queued.
   - `get_snapshot` reads a published lock-free snapshot.
   - The host calls plugins only on the UI thread, with **no host locks held**, under a time budget.
   - A panic is caught at the boundary and the plugin is quarantined.
4. **Events instead of polling.** The plugin declares an event mask. The host delivers events for:
   - file opened/closed and active file/pane/tab changed;
   - viewport changed (debounced);
   - tool activated/deactivated;
   - pointer and key input when the plugin owns the tool;
   - selection and hover;
   - focus;
   - surface shown/hidden;
   - shutdown.
5. **Retained overlay layers.**
   - Calls: `layer_create`, `upsert` / `remove` / `clear`, `set_layer_style`, `set_visible` and `set_opacity`.
   - Primitives: points, polylines, polygons with holes, rectangles, ellipses, text and images (heatmaps), with RGBA fill and stroke, stroke width and dash, sizes in screen or physical units, and z-order.
   - The host owns an R-tree, viewport culling, LOD and clustering, instanced GPU drawing, and hit-testing, and emits `Picked`/`Hover` events.
   - Hover and selection styling are host-applied style overrides.
   - The same layers are drawn into exports and clipboard copies.
6. **Tools.**
   - `register_tool(kind = point | polyline | polygon | rect | ellipse | freehand | custom)`, with pointer events that carry buttons, modifiers and both image- and screen-space coordinates.
   - The host's editing gestures become optional helpers.
   - The current tool and ROI are visible to plugins.
7. **UI surfaces with handles.**
   - `open_surface(kind = sidebar | window | modal | hud_menu, ui, component, size_hint) -> SurfaceId`.
   - Typed, diffed `set_properties(surface, …)` pushed by the plugin.
   - Callbacks are tagged with the surface id and can return values.
   - Compiled definitions are cached.
   - The plugin declares the Slint version it targets.
8. **Services.**
   - Directories and settings: `plugin_root()`, `data_dir()`, `cache_dir()` and a namespaced key/value settings store.
   - Jobs and feedback: `spawn_job` / `job_progress` / `cancel`, `notify(level, text)` (toasts and status bar) and `post_to_ui`.
   - Navigation and file access: relative `pan_by` / `zoom_by` for a given pane, and file dialogs with multiple filters and directory picking.
   - Slide information: slide properties, associated images and ICC profile.
9. **Filters.**
   - Namespaced ids, declared priority, and a push API for the enabled state.
   - The CPU contract includes stride, ROI and timestamp, and filters run off the UI thread.
   - GPU filters are WGSL modules the host runs inside its own pass. The host never hands a raw `VkQueue` to a plugin.
10. **SDK and trust.**
    - Publish an `eov-plugin-sdk` crate. Its `export_plugin!` macro generates the vtable, `catch_unwind` wrappers and default stubs, and the crate also provides a test harness with a mock host.
    - Validate plugin ids with a regex and check that paths stay inside the plugin's root.
    - Sign official plugins, verify hashes at load, ask for consent the first time an unknown hash is loaded, and add a Plugins management panel.

**Migration order:**
- **Before the redesign (these are unsafe today):** fix PLG-2, SH-1 and the general lock discipline (2.1), add `catch_unwind`, and fix the cellaris manifest.
- **Then:** ship the versioned ABI (items 1–3) together with the SDK (item 10) as a single breaking release, so that the existing plugins are recompiled exactly once.
- **Last:** the event model, overlays, tools, surfaces and services (items 4–9), each added as an optional prefix field that does not break existing plugins.

### 2.7 Other cross-cutting observations

- **Dead or vestigial code that misleads readers and costs build time:**
  - `AsyncTileLoader` (`common/src/cache.rs:302-410`, broken, and the reason tokio is a dependency);
  - the static plugin registry and Rust `Plugin` trait route;
  - the DMA-BUF / `nix` / `gpu_interop` path;
  - the hidden `eov plugin-window` subcommand, which loads any directory's native library outside the consent flow (`app/src/cli.rs:239-248`);
  - the gamepad's subprocess-era disk sync and heartbeat files, plus its whole `slint` dependency;
  - a duplicate, unused dataset pipeline that has the same critical bug (`common/src/dataset/pipeline.rs:122-404`);
  - unused crates: `wgpu` 28 (all code uses `slint::wgpu_29`), `nix`, `dashmap`, `uuid` and `thiserror` in `app`;
  - `TileRequestSignature`, computed every frame and never read.
- **Divergent duplicate pipelines.** Viewport CPU rendering, the CPU preview, export and the GPU shader each apply sharpening, stain normalisation, deconvolution and adjustments in different orders, coordinate conventions and rounding modes. As a result the README's "matched output" claim is not true (CLR-5, WSI-7).
- **Identity of a slide.**
  - The tile cache uses a per-session counter.
  - Annotations and eovae use a SHA-256 of 256 sampled 1 KiB chunks of the opened file (`common/src/file_id.rs:54-77`). For multi-file formats (`.mrxs`, `.vms`, DICOM) that file is only an index.
  - The annotations plugin keeps its in-memory cache keyed by path.
  
  None of these uses OpenSlide's `openslide.quickhash-1`. A single, documented slide identity, made of quickhash plus dimensions plus MPP and exposed through the plugin API, would fix WSI-13, ANN-7, ANN-8 and VAE-5 together.
- **Privacy.** Full slide paths often contain accession numbers or patient names. They are logged at info level (about 18 times per open), written into annotation exports and the eovae DB, exposed to every plugin through `HostSnapshot.recent_files`, and stored unencrypted in config directories that are often roaming or backed up. The label image (PHI) is not displayed today, but once it is, de-identification controls will be needed (§14).

---

## 3. Plugin API and host-side plugin runtime

Scope: `plugin_api/`, `app/src/plugins/`, `app/src/plugin_host.rs`, `app/src/viewport_filter.rs`, plugin dispatch in `app/src/main.rs` and `app/src/callbacks.rs`, and the `eov_get_plugin_vtable` implementations of all four plugins.

### 3.1 How it works today

- **Package and discovery.** A plugin is a native cdylib packed into a plain tar (`.eop`). Each package found in `~/.eov/plugins` is extracted to `~/.cache/eov/plugin-packages/<stem>-<len>-<mtime>-<pathhash>/`. Its `plugin.toml` is then parsed and filtered by `environment.version`, a semver range compared with the host *application* version, and duplicates are removed by id.
- **Loading.** The library is opened with `abi_stable::RawLibrary`, and the raw symbol `eov_get_plugin_vtable` is called (`app/src/plugins/manager.rs:179-194`).
- **The contract.** It is two flat `#[repr(C)]` structs of function pointers, both passed by value:
  - `PluginVTable`: 24 mandatory entries, plugin → host.
  - `HostApiVTable`: `context: u64` plus 26 entries, host → plugin, delivered via `set_host_api` (`plugin_api/src/ffi.rs:223-417`).
- **No plugin instance.** No instance pointer is passed to plugin callbacks, so every plugin keeps its state in process-global `Mutex` statics.
- **Polling model.** Everything is pull/poll:
  - The host calls the overlay getters (points, polygons, overlay component and properties) for every plugin and every pane on every rendered frame, holding the `AppState` write lock (`app/src/render.rs:542`, `:728`, `:823` → `app/src/ui_update.rs:325-330`).
  - It calls them again on every pointer move for hit-testing.
  - Plugin windows are re-polled every 200 ms.
- **Host calls from plugins.**
  - Data calls (`get_snapshot`, `read_region`, `get_level_info`) take the global `parking_lot::RwLock<AppState>` on the caller's thread.
  - UI calls are marshalled through `run_on_ui_thread` (`app/src/plugin_host.rs:1989-2020`). On the UI thread this runs synchronously and re-enters the caller's stack; on other threads it blocks on a channel.
- **Plugin UI.** It is `.slint` source compiled at runtime by `slint-interpreter`. Properties travel as JSON strings and callbacks as `(name, args_json)`.

### 3.2 Findings

#### PLG-1 · CRITICAL · The binary contract is never version- or layout-checked, and both vtables cross the boundary by value
- **Where:** `app/src/plugins/manager.rs:179-194` (the same unchecked load is repeated at `app/src/plugins/mod.rs:350-358`, `:397-405`); `plugin_api/src/ffi.rs:223-294`, `:305-417`, `:461`. Manifests: `plugins/eovae/plugin.toml` (`>=0.4.1`), `plugins/cellaris/Cargo.toml` (`eov-plugin-api = "0.4"` from crates.io).
- **Confidence:** Confirmed. I re-verified the loader. The ABI history was checked by diffing the crates.io 0.4.4, 0.4.5, 0.4.8 and 0.4.9 sources.
- **Problem:**
  - `RawLibrary::load_at` plus a raw `get::<GetPluginVTableFn>` performs none of abi_stable's checks: no `RootModule`, no `TypeLayout` comparison, no library-version check.
  - Neither vtable carries an ABI version or a size field.
  - The `SAFETY:` comment ("both compile against the same `plugin_api` crate, so the layout is guaranteed to match") is false. Plugins are separate cdylibs built against any `0.4.x`.
  - `PluginVTable` is returned by value through a hidden return slot. A plugin compiled with a larger struct writes past the host's stack slot. A plugin compiled with a smaller struct leaves trailing function pointers uninitialised, and the host calls them later.
  - `HostApiVTable` is passed by value, so a plugin built against a newer API reads past the end of what an older host passed.
  - The only guard is `environment.version`, which constrains the host app version, not the API the plugin was compiled against.
  - The `#[repr(u32)]` enums `HostToolModeFFI` and `HostLogLevelFFI` are closed. An unknown discriminant from a newer plugin is undefined behaviour in the host.
- **Impact:** This has already happened once in a *patch* release. 0.4.9 appended `get_level_info`, so `HostApiVTable` grew from 208 to 216 bytes. A plugin built against 0.4.9 that calls `get_level_info`, with a manifest like eovae's `>=0.4.1`, loaded into eov 0.4.1–0.4.8, calls a garbage pointer. The first field ever added to `PluginVTable` will corrupt the stack at load for every out-of-tree plugin built against an older 0.4.x. Official plugin release workflows build against eov's master HEAD (`plugins/gamepad/.github/workflows/release.yml:82-83`), so unreleased layout changes can ship.
- **Fix:**
  - Export an abi_stable root module (`#[export_root_module]` + `RootModule::load_from_file`) so layouts are checked at load time.
  - Make both vtables prefix types (`#[sabi(kind(Prefix))]`, `#[sabi(last_prefix_field)]`) with `Option<extern fn>` for optional entries.
  - Add an explicit `abi_version`/`struct_size` handshake.
  - Use `u32` plus `TryFrom` (or `NonExhaustive`) for enums.
  - Put an API-version requirement in the manifest, and treat any vtable change as a minor (0.x) bump.
  - Fix the `SAFETY` comment.

#### PLG-2 · CRITICAL · Deterministic UI deadlock: filter-state sync calls into plugins while holding the `AppState` write lock, and eovae calls back into the host
- **Where:** `app/src/main.rs:642-646` → `app/src/plugins/manager.rs:528-536` → `app/src/viewport_filter.rs:39-46`, `:143-146` → `plugins/eovae/src/lib.rs:464-466` → `plugins/eovae/src/state.rs:197-201` → `app/src/plugin_host.rs:1458-1460`, `:1989-2003`, `:3711-3719`.
- **Confidence:** Confirmed. Three auditors traced it independently and I re-verified it. Not executed.
- **Problem:**
  - After a toolbar action that is neither a sidebar toggle nor a tool button, the handler does `let mut s = filter_state.write(); pm.sync_filter_states(&s.filter_chain);`.
  - `sync_filter_states` re-applies every plugin's enabled flag through `FfiViewportFilter::set_enabled`, which calls the plugin's `set_filter_enabled` unconditionally, even when nothing changed.
  - eovae's `set_filter_enabled` calls `host.request_render`. On the UI thread, `run_on_ui_thread` executes inline and calls `runtime.state.write()` on the same `Arc<RwLock<AppState>>` (`plugin_host.rs:427`).
  - `parking_lot` locks are not re-entrant, so the thread blocks forever.
- **Impact:** With eovae installed, eov freezes permanently in any of these cases:
  - clicking any plain-action toolbar button, for example cellaris's "toggle_cellaris" (once its manifest is fixed, see PLG-4) or any third-party button;
  - closing the gamepad window from the toolbar (`open_window = false` → `Handled`);
  - clicking Annotations when its `sync_active_file()` fails (`plugins/annotations/src/lib.rs:98-113`);
  - clicking eovae's own button when its sidebar fails to build.

  Unsaved work in other plugins is lost. The HUD path (`main.rs:909-918`) avoids the hang by cloning the chain first, and cellaris carries a comment warning about exactly this hazard (`plugins/cellaris/src/lib.rs:568-569`).
- **Fix:**
  - Never call plugin code while holding `AppState`, the filter-chain lock or `HOST_CONTEXTS`.
  - Forward `set_filter_enabled` only on real changes.
  - Make UI-mutating host calls post-and-return, using a dirty flag or `invoke_from_event_loop`, instead of executing inside the caller's stack.
  - In eovae, record the flag only.

#### PLG-3 · HIGH · Plugin callbacks generally run with host locks held, and the host API is blocking and not re-entrant
- **Where:**
  - Filters and overlay getters under `state.write()`: `app/src/render.rs:532-535`, `:542`, `:728`, `:823`, `:1170-1179`; `app/src/ui_update.rs:325-330`, `:510`.
  - Hit-test and context-menu getters under `state.read()`: `app/src/callbacks.rs:2659-2676`, `:3316-3336`, `:1614-1617`.
  - HUD getters under the read lock: `app/src/plugin_host.rs:2125-2140`, `:2195`, `:2230`.
  - `get_viewport_overlay_component` is called while the `HOST_CONTEXTS` `std::sync::Mutex` is held (`plugin_host.rs:2395-2401`), and every `ffi_*` re-locks it (`:3279-3310`).
  - Filters also run inside the thread-local `PANE_RENDER_CACHE` `RefCell` borrow (`render.rs:232-268`, `:2455-2534`).
  - `pm.borrow_mut()` is held across plugin FFI calls (`main.rs:585-669`, `:678-725`, `:901-918`; `callbacks.rs:2750`, `:3414`, `:3508`, `:3706`).
- **Confidence:** Confirmed (mechanism). Concrete instances are PLG-2, GPD-1 and VAE-2.
- **Problem:** There is no documented re-entrancy or threading contract. Inside a getter or filter:
  - Any state-writing host call (`request_render`, `set_*_button_active`, `set_undo_redo_state`, `show_sidebar`, `set_active_viewport`) deadlocks the UI thread.
  - `get_snapshot`, `read_region` and `get_level_info` deadlock inside filters, because the write lock is held.
  - Even `log_message` from `get_viewport_overlay_component` re-locks a std mutex on the same thread.
  - Cross-thread hazards:
    - eovae's analysis worker holds its plugin mutex while calling `get_snapshot` (`plugins/eovae/src/analysis.rs:393-399`). Meanwhile the UI thread holds the write lock and calls eovae's filter, which takes that mutex (`plugins/eovae/src/filter.rs:13-14`, `:79`). This is an ABBA deadlock that re-arms on every pan.
    - The gamepad worker blocks on the UI thread while holding its mutex, and the UI thread then polls the gamepad (GPD-1).
  - `run_on_ui_thread` turns every UI-touching call from a worker into a blocking round trip. Plugins that report progress (eovae, 2–4 refreshes per tile) flood the UI thread.
- **Fix:**
  - Snapshot what plugins need, release every lock, then call plugins.
  - Make every host API callable from any thread and non-blocking: queue UI mutations; serve `get_snapshot` from a published, lock-free copy (for example an `ArcSwap<HostSnapshot>` refreshed by the UI thread).
  - Document, per callback, which host calls are allowed.

#### PLG-4 · HIGH · The newest plugin (cellaris) cannot load: `environment` became mandatory with no migration, and the README manifest format is wrong
- **Where:** `plugins/cellaris/plugin.toml:1-13` (no `[environment]`); `plugin_api/src/manifest.rs:51-52` (no `#[serde(default)]`); `app/src/plugins/discovery.rs:189-227` (skip with `warn!` only); `README.md:356-381`; `plugins/cellaris/README.md:108`.
- **Confidence:** Confirmed. I re-verified it; another auditor parsed the file with eov-plugin-api 0.4.9 and got "missing field `environment`".
- **Problem and impact:**
  - **Required field.** The field became required between v0.4.1 and v0.4.4 (commit 2c151a2). Cellaris says it was "verified loading in the EOV v0.3.1 release binary".
  - **Discovery and management.** On the current host:
    - discovery skips the plugin with a log line only;
    - `eov plugin install ./cellaris.eop` fails;
    - `eov plugin remove cellaris` says nothing matches.
    
    The user sees no button and gets no explanation.
  - **README format.** The main README documents a `[plugin]` table and `[icon] svg = … / file = …`. `PluginManifest` accepts neither: it needs top-level fields and `kind = "svg"`, `data = …`. The README example also omits `[environment]`.
  - **`toolbar_buttons`.** The top-level `toolbar_buttons` array is parsed but ignored for native plugins (PLG-19).
- **Fix:**
  - Add `[environment] version = ">=0.4.9, <0.5"` to cellaris.
  - Either make `environment` optional with a visible warning, or surface the rejection in the UI.
  - Add a doc test that parses the README example, and a plugin-CI step that runs `PluginManifest::from_toml` on every `plugin.toml`.

#### PLG-5 · HIGH · The overlay API copies every object every frame and on every pointer move, under the write lock, with no culling
- **Where:** `app/src/render.rs:542`, `:728`, `:823` → `app/src/ui_update.rs:325-330`, `:510`; `app/src/plugin_host.rs:105-120`, `:624-684`, `:686-710`, `:712-833`, `:835-933`, `:935-1056`; `app/ui/viewport.slint:285-299`; `app/src/callbacks.rs:2659-2823`, `:2952-3043`, `:3316-3336`; `plugins/annotations/src/lib.rs:201-333`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Per frame (up to 60 fps), per pane, under the write lock**, the host:
    - calls every plugin's `get_viewport_overlay_points` and `get_viewport_overlay_polygons`;
    - in polygon-tool mode, fetches the polygons a second time for vertex boxes;
    - allocates one `RString` id per object;
    - converts each vertex to screen space and builds an SVG path string per polygon (`format!("{:.3}")` per vertex);
    - instantiates one full-viewport Slint `Path` per polygon and two Rectangles per point.
  - **Culling:** none. The annotations plugin ignores the viewport bounds it receives.
  - **Pointer moves:** each one re-fetches the full lists for hover hit-tests. The polygon tool does separate vertex, edge and polygon passes, and the vertex/edge passes clone the vertex vector per candidate vertex, so O(V²) allocations per move.
  - **Overlay components:** `refresh_viewport_overlay_factories` also queries every plugin's overlay component and properties every frame.
- **Impact:**
  - Cell-level data is the main use case for an analysis plugin. A slide with 50k nucleus polygons means about 50k path strings re-tessellated per frame and four full copies per mouse move. Panning stalls, and because this runs under the write lock it also starves plugin background reads.
  - While an annotation is selected, the annotations plugin's 120 ms "pulse" thread keeps the loop hot even when idle (ANN-11).
- **Fix:**
  - Replace polling with a retained, host-owned overlay store. Plugins push upserts and removals per layer; the host keeps an R-tree and does culling, level of detail (LOD), batched or instanced GPU rendering, and hit-testing, and emits `Picked`/`Hover` events.
  - As an interim step, add a plugin revision counter (re-fetch only on change), pass bounds and zoom so plugins can cull, and build overlays outside the lock.

#### PLG-6 · HIGH · Plugin GPU viewport filters are not implemented; filters that declare GPU support get a hard-coded grayscale pass
- **Where:** `plugin_api/src/ffi.rs:407-448`; `plugin_api/src/viewport_filter.rs:7-13`, `:34-130`; `app/src/viewport_filter.rs:84-88`, `:134-172`; `app/src/render.rs:1959-1999`, `:2078`; `app/src/gpu.rs:384-432` (`fs_grayscale`), `:1156-1160`, `:1182-1191`, `:1579-1699`; `app/src/gpu_interop.rs:81-139` (`#[allow(dead_code)]`).
- **Confidence:** Confirmed. I re-verified that `apply_filter_gpu` is never called except by a test stub, and that `has_enabled_gpu_filters()` only toggles `fs_grayscale`.
- **Problem:**
  - The host never builds a `GpuFilterContextFFI`, never calls `apply_filter_gpu`, and never uses `DmaBufDescriptor`. The `nix` dependency, added for SCM_RIGHTS fd passing, is unused.
  - On the GPU backend (the default), a filter that declares both CPU and GPU support is not called. Instead the host enables a built-in **grayscale** shader. GPU-only filters never run on either backend.
  - The documented contract would be unsound if implemented as written:
    - it hands plugins wgpu's raw `VkQueue`, although Vulkan requires external synchronisation of submits;
    - it gives no image layout or ownership rules;
    - nothing times out if a plugin never signals the fence;
    - wgpu-hal does not enable `VK_KHR_external_fence_fd`.
  - The handle extraction is Linux/Vulkan-only: macOS and Windows log "Failed to extract Vulkan handles" on every start. The `VkCommandPool` and `VkFence` it creates are never destroyed.
- **Impact:** No current plugin is affected (eovae and cellaris declare `supports_gpu: false`). A third-party plugin that follows the published API gets grayscale output on GPU and nothing on CPU.
- **Fix:**
  - Now: mark the GPU ABI experimental, fall back to `apply_cpu` for any CPU-capable filter, reject GPU-only filters, delete the grayscale stand-in, and remove `DmaBufDescriptor` and `GpuFilterContext` from the public API.
  - Later: let plugins supply WGSL that the host runs inside its own wgpu pass.

#### PLG-7 · MEDIUM · A panic on either side of the boundary aborts the whole process
- **Where:**
  - Every `ffi_*` in `app/src/plugin_host.rs:3538-3880`, and every `(vtable.f)(…)` call site.
  - `lock().unwrap()` at `plugin_host.rs:403-405`, `:3279-3319`.
  - Plugin exports, for example `plugins/annotations/src/lib.rs:133`, `:212` and `plugins/eovae/src/lib.rs:89`, `:179`, `:248`, `:433`.
- **Confidence:** Confirmed (there is no `catch_unwind` outside `test_support`).
- **Problem:**
  - Since Rust 1.81, a panic that unwinds out of an `extern "C"` function aborts the process.
  - Every plugin guards its globals with `std::sync::Mutex` and `unwrap()`. A panic in any plugin worker poisons the mutex, and the next host-driven getter (every frame) aborts eov with no dialog and no chance to save.
  - Host-side panics inside `ffi_*` (a `RefCell` double borrow, a poisoned mutex, an allocation failure in `read_region`) also abort.
  - Concrete triggers:
    - ANN-4: a crafted colour string makes the annotations plugin panic every time the slide is opened.
    - VAE-12: an unexpected output shape panics an eovae worker.
- **Fix:**
  - Wrap host exports in `catch_unwind` and return `RErr`.
  - Ship an SDK `export_plugin!` macro that wraps plugin exports the same way and generates the no-op stubs.
  - Quarantine a plugin after a caught panic and tell the user.

#### PLG-8 · MEDIUM · The host hard-codes behaviour for specific plugin ids and "magic" property names
- **Where:**
  - **gamepad:**
    - forced window width: `app/src/plugins/mod.rs:41-45`;
    - `mapping-rows` is not re-pushed: `mod.rs:142`.
  - **eovae:**
    - filter re-sync after HUD clicks: `app/src/main.rs:909-918`;
    - polygon fill/stroke alpha keyed off the `grid-` id prefix: `app/src/plugin_host.rs:73-103`;
    - trace properties: `plugin_host.rs:3334-3345`;
    - the visualization-mode dropdown labels and action ids are compiled into the **host** UI: `app/ui/hud-toolbar.slint:75-82`, `:207-219`, `:258-296`.
  - **annotations:**
    - `row-hovered` interception and `hovered-row-id` push: `plugin_host.rs:226-292`, `:3100-3105`, `:3425-3428`;
    - a click on empty space always sends "deselect" to `"annotations"`: `app/src/callbacks.rs:3653-3671`;
    - "Delete Point Annotation" invokes a UI callback literally named `delete-annotation-clicked`: `callbacks.rs:1381-1398`, `:1586-1594`.
  - **magic property names:** `capture-hotkeys` (`plugin_host.rs:3471-3486`, `:3507-3522`), `dismiss-popups` (`:3488-3505`), `hud-button/{id}/active` (`:2121-2140`).
- **Confidence:** Confirmed.
- **Impact:**
  - Third-party plugins cannot get any of these behaviours: deselection, hover sync, filter sync, menu buttons, window-size hints.
  - Without the annotations plugin, every click on empty space logs an error.
  - A plugin that reuses one of the magic names by accident triggers host behaviour.
  - Renaming an official plugin silently breaks it.
  - The forced gamepad width (`clamp(2·0.76·parent, 1040, 1760)`) runs off-screen on 1366-px laptops.
- **Fix:** Replace each special case with a declared capability or an explicit API:
  - window-size hints;
  - a per-property refresh policy;
  - a `filters_changed` notification;
  - per-object style;
  - generic selection and hover events;
  - a generic "delete selected" command;
  - menu/dropdown HUD buttons with checked state;
  - documented UI-contract properties.

#### PLG-9 · MEDIUM · Annotation-specific concepts are baked into the ABI and host state, and generic primitives are missing
- **Where:** `plugin_api/src/ffi.rs:62-87` (overlay structs carry `annotation_id` and a fixed RGB ring/fill); `:145-153` (`PointAnnotation`/`PolygonAnnotation` tool modes); `:356-388` (placed/moved/selected callbacks); `app/src/state.rs:560-592` (annotation editing state lives in `AppState`); `app/src/plugin_host.rs:57-103`.
- **Confidence:** Confirmed.
- **Problem:**
  - Plugins can draw only two kinds of overlay:
    - points with a screen-px diameter, an RGB ring and a host-chosen hover style;
    - simple polygons with an RGB fill, host-chosen alpha and a 1-px stroke.
  - Missing: lines, rectangles, ellipses, holes/multipolygons, text labels, RGBA, stroke width or dash, z-order, physical-unit sizes and LOD.
  - Tools are limited to two host-implemented annotation tools, whose gestures (vertex drag, edge insert, double-click delete) are hard-wired.
- **Consequences:**
  - eovae draws its grid lines as thin polygons, which is what forced the host hack in PLG-8.
  - cellaris reuses `PolygonAnnotation` as a region tool. Its readout outline inherits the editing gestures, which go to a no-op `on_polygon_annotation_moved`, so dragging it makes it snap back.
  - `HostToolMode` is a closed enum, so no plugin can define a new tool.
- **Fix:**
  - Generic styled overlay primitives (see §2.6).
  - A tool-registration API that delivers pointer events with buttons, modifiers, wheel input and image plus screen coordinates.
  - Make the host's editing gestures opt-in helpers.

#### PLG-10 · MEDIUM · Shared UI slots (sidebar, modal, confirmation, undo/redo) have no ownership
- **Where:**
  - **Sidebar refresh:** `refresh_sidebar` ignores the caller (`app/src/plugin_host.rs:3376-3401`, `:3776-3784`). The gamepad uses it to refresh its *window* (`plugins/gamepad/src/state.rs:565-569`).
  - **Sidebar toggle:** a repeated, identical `show_sidebar` hides the sidebar instead (`plugin_host.rs:1706-1733`, re-verified).
  - **Confirmation dialog:** a single slot that later requests overwrite (`:1554`, `:1594`).
  - **Undo/redo owner:** the first plugin, in registration order, that has ever published `enabled: true` (`:2231-2240`, re-verified).
  - **Modal dialog:** `show_modal_dialog` replaces another plugin's modal (`:1928-1959`).
  - **Undo context:** `on_undo`/`on_redo` carry no file or pane (`plugin_api/src/ffi.rs:365-368`).
- **Confidence:** Confirmed.
- **Impact:**
  - Every gamepad slider event, and eovae's 1 Hz progress loop, re-pulls the full annotations sidebar JSON tree.
  - Plugins must probe state before calling `show_sidebar` (cellaris does).
  - A second confirmation request makes the first plugin's confirm/cancel callbacks never fire. The startup "prune old plugins" dialog shares that slot.
  - The annotations plugin publishes `enabled: true` at startup, so no other plugin can ever own Ctrl+Z.
  - Host edits (ROI, measurement) are not undoable at all.
- **Fix:**
  - Per-plugin surface handles with ownership checks.
  - "Show" semantics for `show_sidebar`, with a separate `toggle`.
  - Per-plugin confirmation queues.
  - A host-level undo stack that records which plugin, file and pane each action belongs to.

#### PLG-11 · MEDIUM · `read_region` holds the global lock during slide I/O, has no size cap, silently shifts negative coordinates, and its coordinate and alpha conventions are undocumented
- **Where:** `app/src/plugin_host.rs:1171-1187`, `:3557-3574`; `common/src/wsi.rs:338-353`; `plugin_api/src/ffi.rs:226-234`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Lock held during decode.** The `AppState` read lock is held for the whole OpenSlide decode. parking_lot's fair RwLock makes the UI's per-frame `write()` wait behind plugin reads (eovae runs N loader threads; cellaris reads continuously), so the UI stutters during analysis.
  - **No size cap.** A 100k × 80k level-0 read means a 32 GB allocation, and the allocation failure aborts eov.
  - **Negative coordinates.** `x.max(0) as u32` returns pixels offset from the requested region. OpenSlide itself handles negative coordinates by returning transparent pixels.
  - **Undocumented conventions.** Nothing says that x/y are level-0 coordinates while width and height are in level pixels, or that the RGBA is premultiplied, with alpha 0 where nothing was scanned (WSI-2).
  - **Shared handle.** The read goes through the UI's own OpenSlide handle, so one corrupt tile read by a plugin poisons level selection for the viewer (WSI-1).
- **Fix:**
  - Clone an `Arc` reader under the lock and read after releasing it.
  - Use a separate handle.
  - Cap `w·h` and return `RErr` above the cap.
  - Pad out-of-range regions.
  - Document the coordinate frames and the alpha convention, or composite onto the background colour.

#### PLG-12 · MEDIUM · Filter chain: global ids, random order, missed state changes, unchecked buffer contract
- **Where:** `app/src/main.rs:324-339`, `:642-646`, `:909-918`; `app/src/viewport_filter.rs:34-46`, `:62-66`, `:156-171`; `app/src/plugins/manager.rs:84`, `:522-536`; `plugin_api/src/viewport_filter.rs:5`.
- **Confidence:** Confirmed.
- **Problem:**
  - Filter ids are not namespaced, and `set_enabled` acts on the first match, so two plugins with the same id control each other's filter.
  - Order follows `HashMap` iteration (random per run), while the docs promise "registration order". Cellaris and eovae overlays therefore stack in a random order.
  - The host learns about enabled-state changes only by polling `get_viewport_filters` after some toolbar clicks, plus the eovae-only HUD path. A plugin that toggles its filter from a sidebar or worker is ignored until some unrelated click; eovae's own model load/clear shows this (VAE-16).
  - The `bool` returned by `apply_filter_cpu` is ignored.
  - `len` is computed as `w*h*4` in `u32` rather than passed as `data.len()`.
  - Filters run synchronously on the UI thread under the write lock, with no time budget.
- **Fix:**
  - Namespace ids as `plugin:filter`.
  - Use a deterministic order (declared priority, then plugin id) and add a host UI listing filters with toggle and reorder.
  - Add a push API for enabled state.
  - Pass and validate the length and stride.
  - Run filters outside the lock.

#### PLG-13 · MEDIUM · Plugin ids are not validated, yet they build native-library paths and are parsed as delimiters
- **Where:** `plugin_api/src/manifest.rs:84-96`; `plugin_api/src/ffi.rs:468-476`; `app/src/plugins/manager.rs:120-123`; `app/src/plugins/mod.rs:341-342`; `app/src/plugin_host.rs:1159`; `app/src/callbacks.rs:1381-1385`, `:1400-1404`; `app/src/plugins/toolbar.rs:68`; `app/src/update/plugins.rs:1269-1280`.
- **Confidence:** Likely. The Windows behaviour follows from `Path::join`'s documented prefix rules; the code path is traced.
- **Problem:**
  - Ids only have to be non-empty, and `lib{id}.so` / `{id}.dll` is joined onto the extraction root.
  - On Windows, `C:\x\evil` or the drive-relative `C:evil` makes `join` discard the root, so a package with no native code can make eov load a DLL from elsewhere, including a network share.
  - On Unix, `/` traverses within the package.
  - Only `eov plugin install` validates ids, rejecting `/` and `\` but not `:`. Discovery validates nothing, and it also loads hand-copied `.eop` files.
  - `:` in an id breaks context-menu parsing (`split_once(':')`) and the `plugin:button` duplicate key.
  - `-` in an id produces a library name that fails to load, logged only at info level.
  - `entry_ui` and icon paths have the same Windows prefix issue; the `..` check is a substring match.
- **Fix:** Validate `^[a-z0-9][a-z0-9_]{0,63}$` in `PluginManifest::validate`, and require the canonicalised library and UI paths to stay inside the extraction root.

#### PLG-14 · MEDIUM · Anything in the plugin directory is loaded without consent or an integrity check
- **Where:** `app/src/plugins/discovery.rs:125-231`, `:336-381`; `app/src/plugins/manager.rs:116-139`; `app/src/plugin_host.rs:466-476`, `:3279-3319`.
- **Confidence:** Confirmed.
- **Problem:**
  - The "non-official plugin" warning exists only in `eov plugin install`.
  - At startup, every `.eop` in the plugin directory (or in any `--plugin-dir`) is loaded with no prompt, no signature check, and no comparison against the SHA-256 recorded in the managed registry.
  - Cached extractions are reused, including their `.so`, without re-hashing until the package's size or mtime changes.
  - Host contexts are small sequential integers, so any plugin can call the host as another plugin.
  - No UI lists the loaded plugins, their versions or where they came from.
- **Fix:**
  - Verify the registry hash at load.
  - Ask the user the first time an unknown hash is loaded.
  - Key extractions by content hash and verify the library hash.
  - Use opaque, per-plugin, revocable contexts.
  - Sign official plugins.
  - Add a "Plugins" panel (§14.4).

#### PLG-15 · MEDIUM · The runtime Slint UI plumbing is fragile and polling-based
- **Where:** `app/src/plugin_host.rs:1627-1629`, `:2558-2639`, `:2851-2894`, `:2950-3009`, `:3089-3096`, `:3321-3374`; `app/src/plugins/mod.rs:152-158`, `:191-325`; `plugin_api/src/ffi.rs:326-330`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Compilation:**
    - Every sidebar or modal open recompiles its `.slint` on the UI thread; nothing is cached.
    - One plugin's overlay-component compile error breaks the shared overlay layer for *all* plugins. The failure isn't cached, so it is recompiled and logged every frame. This is latent: no current plugin uses overlay components.
    - The plugin's component name is pasted unescaped into generated Slint source.
  - **Property synchronisation:**
    - Overlay bindings are fixed from the first viewport's property set.
    - The first undecodable property aborts all remaining property updates.
    - Sidebars get a brand-new model on every refresh, so Slint re-instantiates every row.
  - **Callbacks:**
    - If any argument fails to convert, the plugin receives `[]`.
    - Callbacks cannot return values.
    - Dispatch is inconsistent: windows are synchronous, sidebars and modals are deferred by a timer, confirmations are synchronous.
    - A single `get_sidebar_properties` and a single `on_ui_callback` serve the sidebar, modal, window and confirmation, so a plugin cannot tell which surface spoke.
  - **Refresh cost:** plugin windows re-pull every property every 200 ms.
  - **Logging:** `log_message` only reaches `tracing`, so users never see plugin errors (ANN-15, CEL-6, VAE-7).
- **Fix:**
  - Cache compiled definitions keyed by path and mtime, and compile each overlay independently.
  - Validate component names.
  - Use surface handles with push-based, diffed `set_properties`.
  - Report and skip bad properties instead of aborting.
  - Use one dispatch policy.
  - Route `log_message` warnings and errors to a user-visible notification area.

#### PLG-16 · MEDIUM · Cross-plugin ordering is nondeterministic, and polygon hit-testing selects the bottom-most shape
- **Where:** `app/src/plugin_host.rs:386`, `:994-1000`, `:2147-2152`, `:2395-2418`, `:3312-3319`; `app/src/plugins/manager.rs:84`, `:522-524`; `app/src/main.rs:328`, `:393`; `app/ui/viewport.slint:285-299`.
- **Confidence:** Confirmed.
- **Problem:**
  - `HashMap` iteration with a random seed drives several orderings: overlay z-order, context-menu item order, hit-test priority, filter order and `set_host_api` order. All of them change between runs.
  - Polygon hit-testing returns the first match, which is the bottom-most shape drawn. Clicking inside a small hotspot drawn on top of a large outline selects the outline.
- **Fix:**
  - Use a sorted `Vec` or `BTreeMap` keyed by declared priority, then plugin id.
  - Hit-test in reverse draw order, or prefer the smallest containing polygon.

#### PLG-17 · MEDIUM · Tool ownership between host and plugins is unarbitrated and invisible to plugins
- **Where:** `app/src/plugin_host.rs:564-574`, `:1406-1456`; `app/src/state.rs:1378-1399`, `:1444-1464`; `plugin_api/src/host.rs:7-13`, `:73-85`.
- **Confidence:** Confirmed.
- **Problem:**
  - Any plugin can call `set_active_tool`:
    - Setting `Navigate` wipes the user's ROI and measurement (UI-13).
    - Setting `PointAnnotation`/`PolygonAnnotation` silently steals another plugin's tool, and any in-progress polygon is dropped.
  - Plugins are never told that their tool was deactivated, and the current tool is not in `HostSnapshot`.
  - `set_active_tool` returns `Ok` before the change has been applied.
  - The host's ROI and measurements are not exposed to plugins, so eovae and cellaris cannot run on "the selected region".
- **Fix:**
  - Add `on_tool_changed` and include the current tool in the snapshot.
  - Define ownership rules.
  - Expose the ROI, measurements and selection through the API.

#### PLG-18 · LOW · Smaller host-runtime defects
- **Persisted sidebar path:** the persisted sidebar stores an absolute `ui_path` into the *versioned* extraction cache, so after an update the old UI is restored, or the restore fails (`app/src/plugin_host.rs:1741-1746`, `:2280-2291`; `app/src/main.rs:42-72`). Cellaris documents a workaround (`plugins/cellaris/src/region.rs:255-279`). Persist plugin-relative paths instead.
- **Broken sidebar can't be closed:** on the UI thread, `show_sidebar` compiles the UI *before* deciding to hide it. A sidebar whose UI file vanished (for example after `eov plugin update`) cannot be closed (`plugin_host.rs:1687-1692` vs `:1796-1808`).
- **Window leaks:** plugin windows and library handles leak on every open/close. Close only hides the window; re-opening re-`dlopen`s the library, `Box::leak`s the handle, compiles a new instance and pushes it onto `PLUGIN_WINDOWS` (`app/src/plugins/mod.rs:73-94`, `:313-321`, `:350-358`).
- **Toolbar and hotkeys:**
  - The toolbar logic assumes `button_id == action_id`, because Slint passes only `action-id` (`app/ui/window-toolbar.slint:332`; `app/src/main.rs:580-626`).
  - Hotkeys are stored raw but compared lower-cased, so `"P"` never fires (`main.rs:176-179`; `plugin_host.rs:576-585`).
  - Hotkeys are single characters, apply to tool buttons only, and conflicts between them are not detected.
- **Unvalidated plugin input:** NaN or infinite values from plugins pass `clamp` straight into viewport state (`plugin_host.rs:1228-1237`, `:1272-1281`), and the sidebar width is only checked to be > 0 (`:2276-2278`).
- **Call order:** plugin getters are called before `set_host_api` (`main.rs:318-339` vs `:391-403`).

#### PLG-19 · LOW · Dead or misleading API surface and documentation
- **Where and problem:**
  - **Dead static path.** The static registry's `register` is `#[cfg(test)]` (`app/src/plugins/registry.rs:28-36`), so the static-plugin code in `manager.rs:140-174`, `:271-292` and `:309-362` is dead in production.
  - **Dead window requests.** `host_context.rs` collects window requests that are never drained (`:32`, `:83-87`).
  - **Ignored `open_window`.** `ActionResponseFFI.open_window` is ignored outside `on_action` (`manager.rs:391`, `:425`; `main.rs:682-683`, `:709-710`).
  - **Unused manifest fields.** `icon` and `toolbar_buttons` are parsed but unused for native plugins (`plugin_api/src/manifest.rs:7-15`, `:46-50`), yet eovae, gamepad and cellaris still declare `[[toolbar_buttons]]`.
  - **Unused or host-internal types.** `ViewportFilter`/`CpuFrameBuffer`/`GpuFilterContext`/`DmaBufDescriptor` (`plugin_api/src/viewport_filter.rs:41-130`) and parts of `plugin_api/src/host.rs:32-42`, `:66-70`.
  - **Wrong docs.** `README.md:328`, `:383-414` and `plugin_api/src/lib.rs:8-13` document the Rust `Plugin`/`HostContext` trait route, which cannot work. The sample calls a non-existent `open_plugin_window(plugin_root, &self.manifest)` and says to "register in main.rs (static registration in v1)". The real mechanism, the FFI vtable, is documented only in `plugin_api/README.md`.
- **Confidence:** Confirmed.
- **Fix:** Delete the static path (or move it behind a test feature), remove the unused types and fields, and rewrite the plugin docs around the vtable or a new SDK crate.

### 3.3 Missing plugin-API capabilities

These are listed in rough priority order. Each one is something an in-tree plugin works around today.

1. **A versioned, negotiable ABI** with optional entries. All 24 vtable entries are mandatory today; the gamepad plugin implements 20 no-op stubs.
2. **Lifecycle and events:**
   - init, shutdown or unload, and runtime enable/disable;
   - file opened and closed (plugin caches grow without bound, e.g. annotations' `state.files` and eovae's tile cache);
   - active pane, tab or viewport changed (plugins poll every frame);
   - tool deactivated, Escape, and sidebar hidden;
   - window focus. Without it the gamepad drives eov from the background (GPD-2).
3. **Services:**
   - `plugin_root()`;
   - `data_dir()` and `cache_dir()`. Cellaris hard-codes `~/.eov/plugins/cellaris`, ignoring `--plugin-dir`; others write into `dirs::config_dir()`.
   - a namespaced settings store;
   - `spawn_job`, progress and cancellation;
   - `notify` (toasts and status bar);
   - a non-blocking `post_to_ui`. The gamepad links its own copy of Slint, winit and wgpu just to call `invoke_from_event_loop`, which can only reach its private, inert Slint instance.
4. **Generic overlays.** Lines and polylines, rectangles and ellipses, multipolygons with holes, text labels and markers, RGBA fill and stroke, stroke width and dash, z-order, per-layer opacity and visibility, sizes in physical units, plus host-side spatial indexing, culling and hit-test events.
5. **Input.**
   - Custom tools with pointer events (buttons, modifiers, wheel) and cursor control.
   - Hotkeys with modifiers or non-text keys, and conflict detection.
   - Per-pane viewport control. `set_active_viewport` and friends act only on the focused pane, even though HUD and context-menu callbacks carry a pane index.
   - Relative navigation calls (`pan_by`, `zoom_by`), so plugins can steer the view without read-modify-write races (GPD-4).
6. **Access to host objects:** ROI, measurements, selection, the current tool, slide properties (the full OpenSlide property map, associated images and ICC profile) and the stain-normalisation state.
7. **Integration:**
   - plugin overlays included in image export and clipboard copy (they are dropped today, `app/src/callbacks.rs:915-924`);
   - a host-level undo history;
   - multi-filter and directory file dialogs;
   - menu or dropdown HUD buttons.
8. **Management UI:** a panel listing loaded and failed plugins, their versions, origin and errors, with enable/disable.
9. **UI compatibility contract.** Plugin `.slint` files are compiled against whatever interpreter version the host has, with no declared requirement and no theme contract.

---

## 4. Core WSI access, tiling and caching (`common/`)

Scope: `common/src/{wsi,tile,cache,viewport,file_id,imaging,render,blitter,overlay,postprocess,formatting}.rs`, and how `app/` uses them.

The auditor read the openslide-rs 0.2.1 source and ran experiments against OpenSlide 4.0.0 (via ctypes) on the repo's fixture slides. Standalone copies of `cache.rs` and `blitter.rs` were built to test the concurrency and sampling claims.

### 4.1 Findings

#### WSI-1 · HIGH · OpenSlide errors are sticky: one bad tile silently disables a handle, with no detection or reopen
- **Where:** `common/src/wsi.rs:330-372`, `:311-315` (`best_level_for_downsample` → `unwrap_or(0)`); `app/src/tile_loader.rs:84-88`, `:248-259`; `app/src/plugin_host.rs:1171-1186`; `app/src/render.rs:597-603`; `app/src/ui_update.rs:353-356`.
- **Confidence:** Confirmed. In a copy of `C3L-00165-21.svs` with 20 MB corrupted, re-reading a previously good tile after one failed `read_region` returned all-zero pixels, and `get_best_level_for_downsample(32)` went from 2 to −1.
- **Problem:**
  - OpenSlide puts the whole handle into an error state on the first decode or I/O error, and every later call on that handle fails.
  - `WsiFile` keeps the dead handle and maps each failure to `Error::ReadTile`.
  - `best_level_for_downsample` turns the −1 into level 0.
- **Impact:**
  - A truncated copy, or a single network read error, poisons the tile worker that decoded it. The failed tile is retried about every 12 frames on whichever worker is free, so the workers are poisoned one after another. Within seconds no new tiles load, and the user is told nothing.
  - If a plugin reads through the host, it uses the **UI** handle. After a failure there, every zoom resolves to level 0, so a fit-to-window view of a large slide requests level-0 tiles and the disk is hammered.
- **Fix:**
  - Check `openslide_get_error` after a failure and return a distinct `HandlePoisoned` error.
  - Reopen the handle, rate-limited.
  - Blacklist only the bad tile and draw a placeholder for it.
  - Compute the best level in Rust from the cached `properties.levels`.
  - Warn the user.

#### WSI-2 · HIGH · Transparent, premultiplied OpenSlide pixels are never composited onto a background
- **Where:**
  - `common/src/wsi.rs:355-371` only swaps BGRA→RGBA.
  - `common/src/blitter.rs:394-413`, `:740-750` overwrite the destination's RGBA, alpha included.
  - Export: `common/src/export.rs:59`, `:162-168` fill a white background that the blits then overwrite. `app/src/callbacks.rs:4426-4442` (JPEG) drops alpha.
  - GPU and CPU rendering: `app/src/gpu.rs:1013`, `:1059`, `:1131`, `:1541` (`blend: None`, dark clear colour); `app/src/render.rs:331` (`Image::from_rgba8` assumes straight alpha); `common/src/blitter.rs:870` (the CPU preview forces alpha to 255).
  - Dataset: `common/src/dataset/output.rs:57-82`, whose white filter ignores alpha.
- **Confidence:** Confirmed. In `fixtures/OS-2.bif`, a read at (15000,10000), in the gap between `openslide.region[0]` and `[1]`, returns ARGB `0x00000000`. Coarse-level edge pixels are partially transparent, e.g. `0x3a373737`.
- **Problem:**
  - No code reads `openslide.background-color`.
  - Multi-region and sparse formats (Ventana BIF, MIRAX, Philips TIFF, sparse DICOM) return transparent "no data" areas and premultiplied edge pixels, and eov treats them as opaque straight-alpha RGBA.
- **Impact:**
  - In the viewer, gaps show the dark UI background instead of glass, and flip between background and black while panning.
  - Exported JPEGs get black holes, and PNGs get transparent holes with dark fringes.
  - Dataset extraction writes black "empty glass" patches into ML datasets (DS-5).
  - Cellaris feeds those pixels to its model as black and counts them as dense tissue (CEL-7).
- **Fix:**
  - In `read_region`/`load_tile_sync`, composite onto `openslide.background-color` (default white) to produce opaque RGBA.
  - Keep an alpha-preserving variant for callers that need it.
  - Add a regression test that the BIF gap reads as white.

#### WSI-3 · HIGH · Thumbnails decode the *entire* coarsest level, on the UI thread, while holding the `AppState` write lock
- **Where:** `common/src/wsi.rs:410-432` (`thumbnail()` computes the target size and then ignores it); `app/src/file_ops.rs:190-232` (same algorithm), called from `open_file` inside `state.write()` (`:501-589`, `:572`) and for **every slide in a browsed folder** (`:273-280`). openslide-rs allocates `w·h·4` up front (`bindings.rs:209-210`).
- **Confidence:** Confirmed (code path, verified by the lead reviewer); memory figures computed.
- **Impact:**
  - A single-level tiled TIFF (common for Aperio "extract region" output, base-only DICOM and many converted files) at 100k × 80k means a 32 GB allocation, which aborts or gets OOM-killed.
  - A two-level slide (downsamples 1 and 4) allocates 2 GB and freezes the UI for seconds.
  - Merely opening a sibling file in the same folder triggers the same read for these files in the series worker, even with the series bar hidden.
- **Fix:**
  - Prefer OpenSlide's `thumbnail` or `macro` associated image.
  - Otherwise pick the coarsest level that is still at least the target size, cap the pixels read, and sample sparsely beyond the cap.
  - Run it off the UI thread, outside the lock.
  - Return `(w, h, rgba)` so the consumer stops guessing the dimensions (RND-10).

#### WSI-4 · HIGH · The tile cache is effectively FIFO, and the default budget is smaller than one frame's working set
- **Where:** `common/src/cache.rs:139-146` (`peek` never touches the LRU; only `get` does, `:103-137`). Every app lookup uses `peek` (`app/src/render.rs:428`, `:445`, `:1451`, `:2178`, `:2678`; `app/src/tile_loader.rs:143`, `:241`). Defaults are 256 MiB and 2048 tiles (`app/src/cli.rs:291-307`).
- **Confidence:** Confirmed: re-verified, plus a scratch test. In a 3-tile cache, peeking tile 0 a hundred times and then inserting tile 3 evicts tile 0. Working-set sizes come from a simulation.
- **Impact:**
  - Eviction follows insertion order. The coarse fallback tiles loaded when a file opens, which are drawn every frame, are evicted first. So are the tiles of a static comparison pane.
  - Tiles are 266 KiB, 1 MiB or 4.2 MiB. The wanted set per pane can exceed the 256 MiB default: 265 MiB at 1920×1045 / ds 20, and 356 MiB at 3840×2125 / ds 12.
  - At those zooms the cache endlessly loads, evicts still-visible tiles and re-requests them: constant disk I/O, flicker between sharp and blurry, and a render loop that never idles.
- **Fix:**
  - Touch the LRU for drawn tiles, with a cheap `touch(&[TileCoord])` or an atomic "last used frame" stamp.
  - Pin the current frame's tiles and the coarsest levels.
  - Size the default from the display (at least about 3× one frame's working set), and prefer evicting tiles of hidden files.
  - Evict tiles of closed files immediately. Today they stay, and a reopened file gets a new id, so they are never reused.

#### WSI-5 · MEDIUM · The visible-tile range is clamped with the 256-px grid instead of the level's real tile size
- **Where:** `common/src/tile.rs:371`, `:383-386` (re-verified). The app's `tile_request_signature` (`app/src/render.rs:483-507`) gets this right, so the signature and the tile list disagree.
- **Confidence:** Confirmed. With `C3L-00088-22.svs` at 1920×1080 in fit-to-window view (level 2, 1024-px tiles, a real 3×3 grid), it returns 12 coordinates when settled (3 invalid) and 20 while moving (11 invalid).
- **Impact:** Non-existent tiles are requested whenever the view touches the right or bottom edge, which is always true in the default fit-to-window view. They fail with `InvalidCoordinates`, are retried forever, use up slots in the `take(100)`/`take(500)` caps, and keep the loop alive.
- **Fix:** Clamp with `tiles_x(tile_size_for_level(level))`, and add a test with mixed tile sizes.

#### WSI-6 · MEDIUM · The trilinear LOD bias is applied in level-index units, so 4× pyramids switch to a much finer level
- **Where:** `common/src/render.rs:23`, `:70-91`, `:97-142`. Used at `app/src/render.rs:597-612`; caps at `:427`, `:1450`, `:2177`, `:2677` and `app/src/tile_loader.rs:310`.
- **Confidence:** Confirmed, using the C3L-00088-22 downsamples (1 / 4.0003 / 16.004).
- **Problem:** `lod − 0.25` is computed in level-index space. On a ratio-4 pyramid (Aperio), one index step is two octaves, so the bias is half an octave. For target downsamples from 4 to 5.66 the "fine" level becomes level 0, even though its blend weight is only 1–25%.
- **Impact:**
  - At downsample 5 at 1080p, the app wants 836 level-0 tiles (3225 at 4K) instead of 15 level-1 tiles.
  - The loader keeps the 500 tiles nearest the centre, while the renderer draws the first 500 cached tiles in row-major order. The result is strips of fallback quality, wasted decoding, cache churn, and moiré from sampling L0 at about 5.7× minification with a 2×2 footprint.
- **Fix:**
  - Apply the bias in log₂(downsample).
  - Drop the fine level when its weight is below ε or it exceeds a pixel budget.
  - Sort before both caps.

#### WSI-7 · MEDIUM · The CPU samplers use a top-left pixel convention, while the GPU and the overlays use pixel centres
- **Where:** `common/src/blitter.rs:74-88` (`src_coord`), `:109-159`, `:172-220`. Used by `app/src/render_pool.rs:305-349` and `common/src/export.rs:321-392`. The GPU uses `uv*size−0.5` (`app/src/gpu.rs:350-356`).
- **Confidence:** Confirmed by compiling the real `blitter.rs`. A black|white 2-px tile blitted at 4× gives `[0,63,127,191,255,…]`, where pixel-centre sampling gives `[0,0,31,95,159,223,255,255]`.
- **Impact:**
  - In CPU mode and in every export, the image is offset by 0.5·(z−1) screen px relative to annotations, measurements and the GPU image: about 1.5 px at 4× and about 50 px at maximum zoom.
  - Switching backends visibly shifts the slide.
  - Fixed-point paths also truncate (`>>16`, `>>8`), so the CPU output is 0.5–1 grey level darker than the GPU's.
- **Fix:** Use `(d+0.5−x0)·s−0.5`, round before shifting, and add golden-image tests that compare the CPU path, the GPU path and the overlay geometry.

#### WSI-8 · MEDIUM · Lanczos-3, the export default, aliases when minifying, seams at tile borders, and is slow
- **Where:** `common/src/blitter.rs:172-220`, `:435-549`; `common/src/tile.rs:288` (`BORDER = 1`); `common/src/export.rs:52`; `app/src/render.rs:29-40`.
- **Confidence:** Confirmed from the code (no visual capture).
- **Problem:**
  - The taps are fixed at −2…+3 source pixels whatever the scale. On ratio-4 pyramids the per-level scale ranges from 0.25 to 1, so there is no low-pass filtering.
  - Taps beyond the 1-px tile border are clamped, so results differ on either side of a tile seam.
  - It is not separable: 36 f64 multiply-adds per channel per pixel, with bounds checks. The README calls it "separable".
- **Fix:**
  - Widen the support by 1/scale, or use trilinear when minifying.
  - Give Lanczos tiles a border of at least 3 px.
  - Make it separable, with f32 weights.

#### WSI-9 · MEDIUM · Up to 18 OpenSlide handles per slide, each with a private ~32 MB cache, and the native tile grid is ignored
- **Where:** `common/src/wsi.rs:149-153` (the doc says to use one handle per worker, with the thread-safety reasoning backwards), `:200-211`, `:286-298`; `app/src/tile_loader.rs:18-22`, `:84-88`; `app/src/file_ops.rs:499`, `:538`; `common/Cargo.toml:20`.
- **Confidence:** Confirmed. Process RSS was 52 MB with one warmed handle and 279 MB with eight.
- **Problem:**
  - OpenSlide handles are thread-safe, and openslide-rs marks them `Send + Sync`. Even so, each loader worker (clamp(nproc, 4, 16) of them), the UI and the `TileManager` each open their own handle.
  - Opening a handle re-parses the slide, which is expensive for MIRAX and DICOM, and each handle has its own cache that no budget accounts for.
  - eov's 256/512/1024 tiles ignore the native 240/256-px tiles, so boundary tiles are decoded more than once.
- **Impact:** On a 32-thread machine that is up to about 576 MB of hidden cache per slide, so four tabs exceed 2 GB.
- **Fix:**
  - Use a shared handle or a small pool.
  - Use one global `openslide_cache` sized from the user's budget. This needs openslide-rs ≥ 2.x, which also exposes ICC and associated-image ICC.
  - Align eov tiles to the native grid.

#### WSI-10 · MEDIUM · `visible_tiles` keeps the first 4096 tiles in row-major order and logs a warning on every call, so single-level slides load only a strip along the top
- **Where:** `common/src/tile.rs:388-413`; `app/src/tile_loader.rs:299-310` (the centre sort happens after the cap); `app/src/render.rs:417-427`; `app/src/cli.rs:556` (warn is the default log level).
- **Confidence:** Confirmed.
- **Impact:** A 100k × 100k single-level TIFF in fit-to-window view has 152,881 candidate tiles. Only the top ~10 rows survive the cap, and `warn!` fires several times per frame.
- **Fix:** Select by distance from the centre before truncating, rate-limit the warning, and synthesise virtual low-resolution levels for slides with no pyramid.

#### WSI-11 · MEDIUM · Lock-order inversion between `stats` and `current_bytes` deadlocks the tile cache
- **Where:** `common/src/cache.rs:206-216` (stats → bytes) vs `:264-273` (bytes → stats inside the eviction loop).
- **Confidence:** Confirmed. A stress test with 8 inserting threads and 4 calling `stats()` deadlocked after 3.2 s. It is latent: nothing calls `stats()` today.
- **Fix:** Use atomics, or release one lock before taking the other.

#### WSI-12 · MEDIUM · `blitter::reproject_frame` is a safe public function that does unchecked indexing without validating sizes
- **Where:** `common/src/blitter.rs:759-874` (`get_unchecked` at `:840-853`).
- **Confidence:** Confirmed. It is not reachable today, because the only caller (`app/src/render.rs:2482`) passes consistent sizes.
- **Problem:** A stale width or height paired with a resized buffer is undefined behaviour reachable from safe code, in a crate published on crates.io.
- **Fix:** Validate the buffer lengths, as `blit_tile` already does, or mark the function `unsafe`.

#### WSI-13 · MEDIUM · The slide fingerprint, used as the key for annotations and eovae results, is weak and only covers the file that was opened
- **Where:** `common/src/file_id.rs:54-77`; `:43-52` (`cached_sha256` is keyed by path and never invalidated). Consumers: `plugins/annotations/src/db.rs:151-158`, `plugins/annotations/src/operations.rs:74-76`, `plugins/eovae/src/analysis.rs:1516-1520`.
- **Confidence:** Likely (no MIRAX or DICOM fixtures to test with).
- **Problem:**
  - It is a SHA-256 of 256 sampled 1 KiB chunks.
  - For `.mrxs`, `.vms` or a single DICOM instance, the hashed file is only an index or one instance, not the image data.
  - The file length is not included, and `openslide.quickhash-1` is not used.
- **Impact:**
  - The series browser lists every `.dcm` file separately, so opening the same DICOM slide through a different instance gives a different key and the annotations seem lost.
  - A re-scanned MIRAX slide that keeps its `.mrxs` keeps the old key.
  - A file replaced in place during a session keeps a stale SHA.
- **Fix:**
  - Use quickhash-1 when available, plus a format-aware hash over all member files and their lengths, and record the dimensions and MPP.
  - Invalidate the cache when (mtime, length) changes.
  - Expose this one slide identity through the plugin API (§2.7).

#### WSI-14 · MEDIUM · `openslide.bounds-*` and `openslide.region[i]` are ignored
- **Where:** `common/src/viewport.rs:66-98`; `common/src/wsi.rs:237-252`.
- **Confidence:** Confirmed that they are never read anywhere. The effect on MIRAX is Likely.
- **Impact:** Fit-to-view, pan limits, the minimap and thumbnails all use the full level-0 canvas. On MIRAX and some Hamamatsu/Philips slides the tissue appears as a small island in a mostly empty frame. Annotation and eovae coordinates are also raw canvas coordinates; ANN-17 covers the export side.
- **Fix:** Parse the bounds and regions into `WsiProperties` and use them for the initial fit, minimap crop, pan clamping and thumbnails.

#### WSI-15 · LOW · Other core defects
- **Minimap rectangle.** It is wrong when the view crosses a slide edge, because `x` and `width` are clamped independently. The minimap has no `clip`, so the rectangle can extend outside it (`common/src/viewport.rs:180-188`; `app/ui/minimap.slint:41-49`).
- **Metadata parsing:**
  - Objective power is parsed as `u32` by openslide-rs, so 2.5×, 1.25× and 6.3× objectives become `None` and the zoom display falls back to percent.
  - MPP is parsed as f32, with no check for ≤ 0 or NaN.
  - Scan date is read only for Aperio.
  - MPP is displayed with 2 decimals (0.4942 → "0.49").
  - 999.96 µm is labelled "1000.0 µm" instead of switching to mm (`common/src/wsi.rs:243-247`; `common/src/formatting.rs:1-10`).
- **Stain label heuristic.** It reports "H&E" for any token containing "h&e", e.g. "Smith&Evans 2019" or "PATH&ENDO_case12" (`common/src/wsi.rs:43-84`).
- **First zoom after idle.** The first mouse-wheel zoom after the app has been idle jumps instead of animating, because `zoom_at` doesn't reset `last_update` (`common/src/viewport.rs:606-624`, `:514-517`).
- **Overlay text fonts.** Overlay text uses hard-coded system font paths. Where they don't exist (some distros and runtimes, and modern macOS, which no longer ships `SFNSText.ttf`), measurement labels are silently missing from exports (`common/src/overlay.rs:375-401`). Embed a font with `include_bytes!`.
- **Dead or broken public APIs:**
  - `AsyncTileLoader` never removes entries, because `loading.clone()` deep-copies the DashMap (`common/src/cache.rs:302-410`).
  - `tiles_to_evict` reorders the LRU and returns duplicates (`:277-292`).
  - `prefetch_tiles` produces children beyond the level and is O(n²) (`common/src/tile.rs:421-494`).
  - `read_tile_with_size` underflows (`common/src/wsi.rs:404-405`).
  - `set_tile_size(0)` panics (`tile.rs:384`).
  - `level_count()−1` underflows on a slide with 0 levels.
- **Cache accounting.** The LRU and the DashMap can desync under concurrent inserts, leaving an orphaned tile that is never evicted. Hit and miss counts are only updated in `get()`, which is never called (`cache.rs:92-94`, `:149-174`, `:219-262`).
- **Latent hazards in helpers:**
  - `fast_fill_rgba` mishandles unaligned slices (`blitter.rs:228-245`).
  - `blend_buffers` underflows when `blend > 1` (`:553-567`).
  - The postprocess LUT truncates (`postprocess.rs:166`).
  - `par_chunks_mut(4)` creates one rayon task per pixel (`:168`).
  - Sharpening copies the whole frame on every call (`:190`).
- **Dependencies and robustness:**
  - openslide-rs 0.2.1 (2022) has no ICC or cache API.
  - Its default `deepzoom` feature pulls in `image` 0.24 and `regex`.
  - It passes `path.display()` to C, so non-UTF-8 paths cannot be opened.
  - It leaks the handle when `openslide_open` returns a handle that is already in an error state.
  - `tokio` is used only by the dead `AsyncTileLoader`.
  - No `rust-version` is declared, although the code needs Rust ≥ 1.88.

### 4.2 What is good here
- Viewport math is f64 throughout.
- Blit rectangles are exact floats, which avoids zoom jitter.
- Tile borders are 1 px, which avoids bilinear seams.
- Cache keys include the tile size.
- The app sorts requests centre-first and removes duplicates.
- No integer overflow was found for realistic slide sizes. Coordinates are bounded by openslide-rs's `u32` addressing, about 4.29 G px.

---

## 5. Rendering pipeline (`app/src/render*.rs`, `gpu*.rs`, `tile_loader.rs`, `backend.rs`)

Rendering is driven by a repeating 16 ms Slint `Timer` (`app/src/main.rs:219-261`), which calls `render::update_and_render` until it returns `false`. Each tick it does the following:

1. Sizes each pane in **logical** pixels.
2. Picks trilinear levels.
3. Builds a centre-sorted list of wanted tiles and hands each file's union to its `TileLoader`. Each loader runs 4–16 OS threads with one OpenSlide handle each.
4. Renders from the shared `TileCache`, using `peek`.

There are two backends:

- **GPU (the default):** draws tiles from `TileArray` 2D-array textures in Slint's `BeforeRendering` notifier, on Slint's own wgpu-29 device.
- **CPU:** composites on a rayon pool and applies the result on the next tick. While moving, it shows a synchronous reprojection preview.

### 5.1 Findings

#### RND-1 · HIGH · The render loop never goes idle in the default Trilinear mode, or while an ROI exists
- **Where:**
  - `app/src/render.rs:1259-1268` (CPU) and `:1824-1834` (GPU), both containing `new_tiles_loaded = … || !cached_tiles.coarse_tiles.is_empty() || …` (re-verified)
  - early exits at `:1303-1324` and `:1883-1896`
  - `app/src/main.rs:254-258`
  - ROI "marching ants" keep `keep_running` true: `app/src/tools.rs:151-168`
- **Confidence:** Confirmed.
- **Problem:** "Any coarse tile is cached" stays true on every tick whenever the view sits between pyramid levels, which is the normal case in Trilinear mode (and in Lanczos below 0.12 zoom). The tile epoch already reports newly loaded tiles, so this extra term is not needed.
- **Impact:**
  - **CPU backend:** a full-frame multi-threaded render job is resubmitted forever, with a full-frame clone, copy and texture upload on the UI thread for each.
  - **GPU backend:** every tick re-hashes all draws, runs `update_tabs` (including FFI calls into every plugin for overlays, and an `fs::metadata()` per pane, `app/src/ui_update.rs:139-141`), and requests a redraw. The debug FPS overlay shows about 60 fps while idle.
  - The result is constant CPU and GPU load and battery drain. On a hung network mount, the per-frame `stat()` blocks the UI.
- **Fix:**
  - Drop the term, or compare against the previous frame's coarse count.
  - Animate the marching ants in Slint.
  - Cache the file size.
  - Add a regression test asserting that `update_and_render` returns `false` once everything is loaded.

#### RND-2 · HIGH · GPU tile texture arrays hold fewer tiles than a single frame needs; the overflow is silently dropped
- **Where:** `app/src/gpu.rs:17` (`MAX_TILE_ARRAY_BYTES = 128 MiB`, re-verified), `:706-724` (eviction protects current-frame tiles, then `pop()?`), `:1391-1427` (draws skipped; a coarse partner that fails to insert falls back to the fine layer but keeps `mip_blend`).
- **Confidence:** Confirmed by code; the drop counts come from a simulation of the exact selection, caps and array logic.
- **Problem:**
  - 128 MiB is 31 layers of 1026² tiles, 127 of 514² and 256 of 258².
  - A fine draw with a coarse partner goes into the array sized for the larger of the two.
  - Once every layer is in use for the current frame, later draws are skipped without a log. These are the last fine tiles in row-major order, i.e. the bottom of the pane.
- **Impact:** For a 4-level 100k×80k SVS with every tile cached:

  | Pane | Downsample | Draws dropped |
  |---|---|---|
  | 1920×1045 | 16 | 132 of 164 |
  | 3840×2125 | 16 | 502 of 534 |

  The bottom of the pane shows blurry fallback with seams, and split view is worse.
- **Fix:**
  - Size arrays from per-frame demand: grow them, or keep several arrays per size class.
  - Pad tiles to fixed size classes, as the doc comment at `gpu.rs:830-833` already claims.
  - Don't spend layers on fully covered fallback tiles.
  - Set `mip_blend = 0` when the coarse partner fails.
  - Count and log overflow.

#### RND-3 · HIGH · Rendering ignores the HiDPI scale factor, so imagery is half-resolution on Retina and 200% displays
- **Where:** `app/src/render.rs:563-568`, `:862-867`, `:1921-1922`; `app/src/gpu.rs:1199-1243`; `app/ui/viewport.slint:106-111` (`image-fit: fill`). There is no `scale_factor` anywhere in the render path (re-verified).
- **Confidence:** Confirmed.
- **Problem:**
  - The viewport size, level choice, CPU buffers and GPU surfaces are all in logical pixels, and Slint stretches the result to the physical area.
  - So "100%" means 1 image pixel per logical pixel, and every view is 2× upsampled on a 2× display.
  - Pathologists judge nuclear detail at native resolution, so this directly hurts diagnostic image quality on the most common modern laptops.
- **Fix:**
  - Render at logical × `scale_factor`, choose levels from the physical zoom, and keep overlay math in logical units.
  - Do RND-1, RND-2 and WSI-4 first, because physical-resolution rendering roughly quadruples tile demand.

#### RND-4 · HIGH · eov cannot start without a GPU-backed Vulkan adapter (Windows/Linux) or Metal (macOS); the "CPU renderer" is not a fallback
- **Where:**
  - `app/src/backend.rs:13-34`, `:53-74` (`Backends::VULKAN` everywhere except macOS; `femtovg-wgpu` is the only UI renderer)
  - `app/src/main.rs:301`, `:419` (`gpu_backend_available = true` hard-coded), `:426-430` (the "GPU renderer unavailable" message can never appear)
  - Slint panics in `i-slint-core-1.17.1/graphics/wgpu_29.rs:299` ("Failed to find an appropriate adapter")
- **Confidence:** Confirmed code path; Likely for the exact failure mode (not run on a Vulkan-less host).
- **Problem:**
  - The UI toolkit itself needs Vulkan or Metal through wgpu.
  - `--cpu` only changes how slide pixels are composited, so it does not help.
  - CPU adapters such as lavapipe are rejected unless `SLINT_WGPU_CPU=1` is set.
- **Impact:**
  - It fails in hospital VDI/Citrix/RDP sessions, VirtualBox and Hyper-V VMs, Windows machines without a Vulkan ICD, Windows-on-ARM devices without Vulkan, and headless CI.
  - On Windows the process exits without any message, because the release binary has no console (SH-2).
  - The README's "Toggle between CPU and GPU rendering (Vulkan)" implies the CPU mode avoids this.
- **Fix:**
  - Allow `Backends::PRIMARY` (DX12 on Windows) and retry with GL or CPU adapters.
  - Enable Slint's software renderer as the last resort.
  - Derive `gpu_backend_available` from what actually initialised.
  - Show a native error dialog on failure.

#### RND-5 · MEDIUM · Stale tile requests are never cancelled, and request priority is lost
- **Where:** `app/src/tile_loader.rs:1-4`, `:36-39` (the docs promise "automatic cancellation of stale requests"), `:125-182`, `:232-264`; `app/src/render.rs:570-635`, `:756-789` (the union goes through a `HashSet`, so order becomes random).
- **Confidence:** Confirmed.
- **Problem:** The generation counter only delays retries of failed tiles; workers load every coordinate they receive. When the channel (128–512 slots) is full, it is the *new* requests that are dropped. The centre-first, fine-before-fallback order is lost in the `HashSet` union.
- **Impact:** After a fast pan or zoom on network storage, workers first decode up to 512 off-screen tiles before reaching the current view.
- **Fix:** Tag requests with a generation and have workers skip stale ones, or keep a priority queue that is replaced every frame. Keep wanted lists ordered (`IndexSet`).

#### RND-6 · MEDIUM · CPU-only plugin filters quietly move the GPU backend onto a slow single-threaded CPU path
- **Where:** `app/src/render.rs:1959-1999`; the "parallel" compositor is a sequential loop (`:2537-2565`); all coarser levels are composited every frame with no occlusion test (`:1433-1491`, `:2158-2219`, `:2441-2535`); a full-frame clone and copy per completion (`:1166`, `:1181-1198`).
- **Confidence:** Confirmed.
- **Impact:**
  - Enabling the cellaris or eovae overlay (both CPU-only) switches every pane to CPU compositing under the write lock, while the UI still says "GPU".
  - Panning a 4K pane drops frames, with 4–10× overdraw on deep pyramids.
- **Fix:**
  - Render on the GPU, read the frame back once, apply the CPU filter on a worker, then upload the result.
  - Use `par_chunks_mut`, cull fallback levels that are fully covered, and reuse buffers.

#### RND-7 · MEDIUM · GPU memory grows without bound, and there is no wgpu error or device-lost handling
- **Where:** `app/src/gpu.rs:830-833` (the doc claims power-of-two size classes), `:1391-1406` (arrays are keyed by the exact padded size), `:1184-1197`, `:1232` (surfaces are never released when a pane closes). There is no `on_uncaptured_error` handler, and wgpu's default handler panics. `ensure_surface` has no `max_texture_dimension_2d` check.
- **Confidence:** Confirmed for the growth; Likely for the panics.
- **Impact:**
  - Each distinct edge-tile size creates another array of up to 128 MiB, never freed: about 724 MiB for one SVS and about 1.3 GiB for three slides.
  - Running out of VRAM, a TDR/driver reset, an eGPU unplug, or an oversized window crashes the app.
- **Fix:**
  - Use fixed size classes with an overall VRAM budget, and free arrays in LRU order.
  - Install error and device-lost handlers that fall back to the CPU backend.

#### RND-8 · MEDIUM · Tile-loader threads poll, run per file, and are joined on the UI thread
- **Where:** `app/src/tile_loader.rs:18-30`, `:71-104`, `:193-202` (`drop(self.request_tx.clone())` is a no-op), `:232-238` (10 ms `recv_timeout`).
- **Confidence:** Confirmed.
- **Impact:**
  - With ten open slides on a 16-core machine there are about 160 threads and about 16k idle wake-ups per second.
  - Closing a tab during a slow network read stalls the UI, because the loader joins its workers on the UI thread.
- **Fix:** Use one shared, bounded worker pool for all files, with blocking receives and disconnect-based shutdown, and detach rather than join when a file closes.

#### RND-9 · MEDIUM · CPU render jobs are keyed by pane index, so inserting or removing a pane orphans them
- **Where:** `app/src/render_pool.rs:114`, `:159-168`; `app/src/state.rs:1005-1038`; `app/src/render.rs:1146-1149`, `:1319`, `:1415-1418`.
- **Confidence:** Likely.
- **Impact:** After a split, a moved pane can keep a pending job id that is either discarded or cancelled by the new pane's job. The pane stays at preview quality until it is moved again.
- **Fix:** Key jobs by stable pane id plus file id, and invalidate them on pane insert or remove.

#### RND-10 · LOW · Smaller rendering defects
- **Double post-processing in the CPU preview.** Reprojected pixels are post-processed twice: the cached frame is already post-processed, and gamma, stain normalisation, deconvolution and sharpening are applied again (`app/src/render.rs:2482-2509`, `:1166`). Bands or colour shifts are visible while panning with adjustments active.
- **Misregistered overlays.** Filters for a completed CPU frame receive the *current* viewport rather than the frame's (`render.rs:1159-1165`), so eovae and cellaris overlays are misregistered on a resize or a jump.
- **Missing minimap.** The minimap is blank when the slide's smallest level is ≤150 px. `generate_thumbnail` throws away the real dimensions, the consumer assumes 150 px, the buffer is too short, and this is retried every frame (`app/src/file_ops.rs:211-217`, `:629-631`; `render.rs:512-525`; `app/src/blitter.rs:52-55`). This is common for libvips, QuPath, NDPI and DICOM pyramids.
- **Cleanup and leftovers:**
  - A `VkCommandPool` and a `VkFence` are created per `RenderingSetup` and never destroyed (`app/src/gpu_interop.rs:110-124`).
  - `TileRequestSignature` is computed every frame and never read (`state.rs:301`, `:445`; `render.rs:483-510`).
  - There is debug logging gated on `pane.0 == 1` (`render.rs:672-715`, `:1836-1856`).
  - The recycled-buffer pool is never trimmed (`render_pool.rs:139-156`).
  - The 62.5 Hz timer is not vsync-aligned (`main.rs:105-106`).
- **Zoom display.** Optical zoom shows "0x" below 0.05× (`app/src/zoom_display.rs:99-106`), and the zoom slider tooltip shows % while the HUD shows optical magnification.

### 5.2 What is good here
- Fingerprint-based skipping of unchanged GPU frames.
- Cooperative per-pane CPU job cancellation.
- The CPU path's snapshot/commit structure, which keeps compositing out of the lock.
- Use of Slint's own wgpu device, which avoids cross-device sharing.
- Correct UV, border and Lanczos clamping for padded tiles.

---

## 6. Colour science, image export and dataset extraction

Scope: `common/src/stain.rs`, `common/src/postprocess.rs`, `common/src/export.rs`, `common/src/dataset/*`; their consumers in `app/src/render*.rs` and `app/src/callbacks.rs`; the WGSL shaders in `app/src/gpu.rs`; the export and dataset dialogs; and `eov dataset patches`.

The auditor checked the numbers by running copies of the real `stain.rs`/`postprocess.rs` on synthetic H&E data, built from a Ruifrok–Johnston (R&J) source matrix plus noise. `fixtures/` is gitignored, and no real-background measurements were made.

### 6.1 Stain normalisation and colour deconvolution

#### CLR-1 · HIGH · Stain normalisation is re-fitted from whatever is on screen, per pane, every time a tile loads
- **Where:**
  - `app/src/render.rs:457-481`: the cache is valid only while the tile epoch is unchanged (`:472-474`).
  - `app/src/render.rs:1571-1592`, `:2023-2042`, `:2301-2330`: the fits.
  - `app/src/state.rs:457-462`: the cache is per pane.
  - `app/src/tile_loader.rs:251`: the epoch bumps for every loaded tile.
  - `common/src/export.rs:237-247`: export re-fits from its own tiles.
- **Confidence:** Confirmed.
- **Problem:**
  - The Macenko or Vahadane fit uses the tiles currently visible. It re-runs on the UI thread whenever any tile arrives, separately for each pane.
  - The 96-DPI preview and the final export each fit again, from different tile sets.
  - The CPU and GPU paths fit from different inputs.
- **Impact:**
  - The same nucleus pixel (115,86,175) maps to:
    - (117,91,176) in a nuclei-dense view
    - (108,86,172) in a stroma view
    - (111,68,161) in a glassy view
  - In a stroma-only view `scale_h` reaches 3.45 and a nucleus renders almost black: (3,1,41).
  - Two panes on the same slide disagree, and export ≠ preview ≠ screen.
  - Colour changing as you pan undermines the purpose of normalisation, and makes it unsafe for side-by-side comparison.
- **Fix:**
  - Fit once per slide, on a low-resolution level with a tissue mask, off the UI thread.
  - Cache the result by file fingerprint and method, and share it across panes, preview and export.
  - Add "fit to ROI", "lock" and "reference slide" options.

#### CLR-2 · HIGH · The tissue mask lets glass background into the fit, so glass-dominated views get recoloured
- **Where:** `common/src/stain.rs:33-37`, `:88-94`, `:432-449`, `:483-486` (`MIN_TISSUE_PIXELS = 100`, an absolute count), `:529-531`.
- **Confidence:** Confirmed (numerical).
- **Problem:**
  - Macenko's β threshold should drop a pixel if *any* channel's OD is below 0.15. Here it is applied to the *sum* of the three ODs, so glass at about ≤240 RGB counts as tissue.
  - I0 is fixed at 255, and there is no minimum tissue *fraction*.
- **Impact:**
  - With background at 240, 232 or 225, the H-vector error is 14.6°, 12.1° or 11.1°, compared with about 1° using a per-channel β.
  - In a glass-only view (236–240), glass at (236,234,239) becomes purple: (129,77,169) with Macenko, (143,115,192) with Vahadane.
  - It is triggered just by panning to a tissue edge on scanners with a 230–240 background.
- **Fix:**
  - Apply β per channel, or use a LAB luminance mask.
  - Estimate I0 from a bright percentile.
  - Require a minimum tissue fraction; otherwise fall back to the slide-level or last-good parameters.

#### CLR-3 · HIGH · Deconvolution after normalisation applies the *source* inverse to pixels that are already in the *target* basis
- **Where:** `common/src/stain.rs:661-702`, `:23-26`, `:611`. Applied after normalisation in `app/src/render_pool.rs:235-249`, `:275-281`, `common/src/export.rs:237-266`, and `app/src/gpu.rs:178`, `:380`.
- **Confidence:** Confirmed numerically by two independent auditors.
- **Problem:**
  - After normalisation, OD = T·(scale∘c).
  - Deconvolution then computes pinv(S_src)·OD and reconstructs with T. That is not the identity, so the channels mix.
  - `README.md:180` advertises exactly this combination ("deconvolution reuses the slide-specific stain matrix").
- **Impact:**
  - A pixel with H = 1.0 and E = 0.5 decomposes as H = 1.14, E = 0.30.
  - In the isolated-H view of a normalised slide, a pure-eosin pixel reads grey 235 instead of 254.
- **Fix:**
  - Use the target inverse after normalisation, or better, fuse the two steps: compute c from the raw pixels, apply the normalisation scale and the user's intensities in concentration space, and reconstruct once.
  - Add a synthetic-stain unit test.

#### CLR-4 · HIGH · Any non-default deconvolution slider recolours everything that is not pure H or E
- **Where:** `common/src/stain.rs:670-678`, `:784-805`; `app/src/gpu.rs:125-139`, `:307-320`.
- **Confidence:** Confirmed (numerical).
- **Problem:** Reconstruction is Σ kᵢ·cᵢ·sᵢ, built from a two-column pseudo-inverse with c clamped ≥ 0. The out-of-plane (residual) absorbance is discarded, so even 1.01 is not close to the identity.
- **Impact:** With H × 1.01 and no normalisation:

  | Structure | Original RGB | Rendered RGB |
  |---|---|---|
  | Red blood cells | (200,60,70) | (185,47,152) |
  | Brown pigment / DAB | (150,100,60) | (135,73,168) |
  | Black ink | (40,40,45) | (35,28,113) |

  Boosting haematoxylin to bring out nuclei therefore changes the colour of pigments, RBCs and IHC chromogen, which is diagnostically misleading.
- **Fix:** Use OD_out = OD_in + Σ(kᵢ−1)·cᵢ·sᵢ, which keeps the residual, or add a third (residual or DAB) vector.

#### CLR-5 · MEDIUM · CPU and GPU output are not "matched" (contrary to README:115 and :171)
- **Where:** `app/src/gpu.rs:115-124`/`:299-306` vs `common/src/stain.rs:771-783`; `gpu.rs:155-179`/`:334-381` vs `app/src/render_pool.rs:235-268`; `common/src/postprocess.rs:24`, `:72`; `stain.rs:82-84`, `:807-809`.
- **Confidence:** Confirmed.
- **Differences:**
  1. GPU isolated-channel mode ignores the channel's intensity and visibility. H at 2× gives grey 23 on CPU and 77 on GPU, and a hidden channel shows white on CPU.
  2. GPU sharpening happens in texel space, *before* stain normalisation. Its strength therefore scales with zoom, and its trilinear centre sample mixes coarse and fine levels (`4·s·b·(c−f)` tile-shaped artefacts). CPU sharpening is in screen space, *after* deconvolution.
  3. The CPU path truncates where the GPU rounds. With brightness +0.002, 255 of the 256 LUT entries come out one level darker.
  4. The CPU sampler has a pixel-centre offset (WSI-7).
- **Impact:** Export is CPU-only, so GPU users' exports differ from what they saw on screen.
- **Fix:**
  - Honour intensity and visibility in the shader.
  - Do one screen-space sharpen pass at the same point in both pipelines.
  - Round on the CPU.
  - Add golden-image tests that compare the two backends.

#### CLR-6 · MEDIUM · The fitted stain matrix is never sanity-checked
- **Where:** `common/src/stain.rs:106-109` (the degenerate case returns an all-zero inverse), `:133-137` (scales up to 1970), `:228-265`, `:419-422`.
- **Confidence:** Confirmed (numerical).
- **Impact:**
  - A uniform region gives coincident vectors, a zero inverse, and every tissue pixel renders white.
  - A view containing pen marks gives inverse entries around ±20, so a nucleus becomes (143,0,43).
  - A stroma-only view gives H and E vectors only 22° apart.
- **Fix:**
  - Reject ill-conditioned fits, using a minimum angle between the vectors and a condition-number check.
  - Clamp the scale factors.
  - Fall back to the slide-level parameters.

#### CLR-7 · MEDIUM · Stain fitting and the per-pixel passes are expensive, single-threaded and on the UI thread
- **Where:** `app/src/render.rs:1578-1585`, `:2024-2035`, `:2308-2319`, `:2509`; `common/src/stain.rs:528-540`, `:754-810`.
- **Confidence:** Confirmed (measured with a release build of the real code).
- **Measurements:**
  - Fit, on every tile epoch: Macenko 12 ms, Vahadane 25 ms.
  - Over a 2560×1440 frame: normalisation 64 ms and deconvolution 65 ms, each single-threaded.
  - The CPU backend's interactive preview runs both synchronously, giving about 7 fps while panning.
- **Fix:**
  - Fit once per slide (CLR-1).
  - Use `par_chunks_mut`.
  - Use a 256-entry −ln lookup table.

#### CLR-8 · MEDIUM · The H&E model is applied to IHC slides, turning DAB brown into eosin pink
- **Where:** `common/src/stain.rs:23-26`, `:260-263`, `:389-392`. The stain label is parsed in `common/src/wsi.rs:63-87` but never used.
- **Confidence:** Confirmed (numerical).
- **Impact:** DAB at (189,139,115) becomes (232,115,201) with Macenko. There is no H-DAB or custom-vector deconvolution, and no DAB quantification, which IHC work depends on.
- **Fix:**
  - Warn or block when the stain label is not H&E.
  - Add stain presets (H&E, H-DAB, H-E-DAB, Masson trichrome, etc.) and custom vectors, including a "measure vectors from ROI" option.

#### CLR-9 · MEDIUM · The "Adjustments" indicator ignores stain normalisation and deconvolution
- **Where:** `app/ui/viewport.slint:638`, `:670`.
- **Confidence:** Confirmed.
- **Problem:** The HUD button lights up only for sharpness, gamma, brightness and contrast.
- **Impact:** A view that is normalised, or that has altered H/E intensities, carries no on-screen flag, so a reader may not realise the colours are synthetic. This is a patient-safety concern if eov is ever used near diagnosis.
- **Fix:**
  - Include every colour-altering setting in the indicator.
  - Show a persistent on-viewport badge, e.g. "Colour-normalised (Macenko)".
  - Burn the same badge into exports.

### 6.2 Image export

#### EXP-1 · HIGH · High-DPI exports silently drop full-resolution tiles beyond the first 4096
- **Where:** `common/src/export.rs:129-137` → `common/src/tile.rs:388-411` (the cap is 4096, applied in row-major order, with only a `tracing::warn`). Also `tile.rs:384`, `:386` (WSI-5).
- **Confidence:** Confirmed (code plus a simulation of the same arithmetic).
- **Impact:**
  - The bottom rows of fine tiles are never loaded, and upsampled coarser levels fill in for them without any notice to the user.
  - Affected cases:
    - 1920×1080 at 300 DPI: capped at zooms 0.057–0.088, the typical low-power figure views;
    - 2560×1440 at 600 DPI: capped at zooms 0.007–0.117;
    - 3840×2160: capped even at 150 DPI.
- **Fix:** Use an export-specific, uncapped tile enumeration with band rendering, and report any shortfall to the user.

#### EXP-2 · MEDIUM · Export (preview and final) runs synchronously on the UI thread, with unbounded memory
- **Where:**
  - `app/src/callbacks.rs:1048`, `:4323-4373`, `:4383-4460`
  - `common/src/export.rs:172-202` (also loads every coarser level synchronously), `:296-318`
  - `common/src/blitter.rs:435-549` (single-threaded Lanczos)
  - `app/ui/export-dialog.slint:925-928`
- **Confidence:** Confirmed.
- **Problem:**
  - Decoding, filtering, stain processing and encoding all run on the Slint event loop, holding `state.read()`, with no progress or cancel.
  - The "thumbnail-sized" preview is actually a full viewport render, and each slider release renders it twice (once immediately, once debounced).
- **Impact:** A 600-DPI export of a 4K view needs about 1.3 GB of RGBA, about 1 GB more for the JPEG RGB copy, and up to about 1 GB of tiles. The UI freezes for tens of seconds.
- **Fix:**
  - Run export on a worker thread with progress and cancel.
  - Render in bands and stream them to the encoder.
  - Render the preview at ≤512 px, once per change.

#### EXP-3 · MEDIUM · Export ignores the JPEG quality setting and writes no resolution metadata
- **Where:** `app/src/callbacks.rs:996` (`jpeg_quality: 85`), `:4399-4401`, `:4425-4443` (`image::save_buffer`, re-verified); `app/ui/export-dialog.slint:1729-1743`.
- **Confidence:** Confirmed. In image 0.25, `save_buffer` → `JpegEncoder::new` uses quality 75, and no `PixelDensity` is set.
- **Problem:**
  - The quality slider has no effect.
  - Neither format carries the resolution: JPEG density is aspect-only, and PNG has no pHYs chunk, so a "300 DPI" figure opens at 72 or 96 DPI. No µm/px is recorded anywhere.
  - The format is inferred from the typed extension, so a name with no extension fails, and a mismatched extension silently overrides the format selected in the dialog.
- **Fix:**
  - Use `JpegEncoder::new_with_quality` with `set_pixel_density`.
  - Write pHYs through the `png` crate.
  - Write µm/px into the file metadata (PNG `tEXt`, TIFF tags).
  - Append the extension when it is missing.

#### EXP-4 · MEDIUM · The export dialog's RGB fields show values ×255, and editing one corrupts the other channels
- **Where:** `app/ui/export-dialog.slint:629-638`, `:663-671`, `:696-704`.
- **Confidence:** Confirmed: Slint's `color.red`/`green`/`blue` are `Int32` 0–255 (`i-slint-compiler-1.17.1/expression_tree.rs:236`), re-verified.
- **Problem:**
  - Each field displays `current-color.red * 255`, e.g. "11730".
  - Editing R calls `Colors.rgb(r, green*255, blue*255)`, which clamps every other non-zero channel to 255.
- **Impact:** Custom measurement and ROI overlay colours come out different in exported figures.
- **Fix:** Use the channels directly.

#### EXP-5 · LOW · Other export defects
- **Stale settings.** The export dialog reuses cached settings: HUD adjustments are not re-read after the first open, and the cache is shared across files (`app/src/callbacks.rs:1003-1035`, `:4323-4336`, `:4467-4473`).
- **Wrong pane.** The ROI overlay is drawn even when it belongs to another pane, and the export uses `focused_pane` rather than the pane the dialog was opened for (`:830-888`, `:938-944`, `:1019-1025`, `:4349`, `:4405`).
- **Measurement labels:**
  - They use the mean of mpp_x and mpp_y.
  - Without MPP they fall back to the length in export pixels, which changes with DPI.
  - They ignore the HUD unit setting, and the label border is always green.
  - They are silently dropped when no system font is found (`:745-751`, `:801-810`; `common/src/overlay.rs:375-400`, `:457-468`).
- **Lanczos aliasing.** Export defaults to Lanczos even when minifying (WSI-8), while the viewport switches to trilinear adaptively.
- **Doc/comment mismatch.** `common/src/export.rs:3-5` says Lanczos is the default, but the dialog defaults to Trilinear.

### 6.3 Dataset patch extraction

#### DS-1 · CRITICAL · In a multi-slide run, every slide after the first is read through the *first* slide's OpenSlide handle
- **Where:** `common/src/dataset/pipeline.rs:418` (the rayon pool is built once per run), `:519-538` (the thread-local slot is filled only when empty, and is never compared with `slide_path` or reset). The dead duplicate `run_dataset_patches` at `:126`, `:246-267` has the same bug.
- **Confidence:** Confirmed, and re-verified by the lead reviewer. The auditor reproduced the exact structure (a persistent rayon pool with a `thread_local!` inside `pool.install`): slide 1 had 0 of 2000 tiles wrong, slides 2 and 3 had 2000 of 2000 served by slide 1's handle.
- **Problem:** A `thread_local!` declared inside a closure is still a single static, and pool threads live across the slide loop. A worker that opened slide A therefore runs `read_region` on A using slide B's coordinates.
- **Impact:** This hits `eov dataset patches a.svs b.svs …` and the GUI dataset export with two or more open slides, whose default inputs are all open files.
  - Tiles under `slides/b/` contain A's pixels.
  - Where B's coordinates fall outside A, OpenSlide returns transparent pixels with no error. These pass the white filter and are written.
  - The metadata labels these tiles with B's path, SHA-256 and MPP.
  - If slide 1 is small, only some threads hold its handle, so slide 2 becomes a non-deterministic mix of both slides.
  
  The result is silently corrupted ML training data.
- **Fix:**
  - Store `(PathBuf, WsiFile)` in the thread-local and reopen when the path differs, or build the pool per slide, or keep a per-slide handle `Vec` indexed by `rayon::current_thread_index()`.
  - Delete the duplicate pipeline.
  - Add a two-slide integration test.

#### DS-2 · HIGH · Slides with the same file stem overwrite each other's tiles
- **Where:** `common/src/dataset/output.rs:9-11`, `:25-31`, `:87-92`; `pipeline.rs:460`, `:499`, `:557`.
- **Confidence:** Confirmed.
- **Problem:** Output directories and tile names come only from `file_stem`, and every non-UTF-8 stem becomes `"unknown"`.
- **Impact:** `cases/001/HE.svs` and `cases/002/HE.svs` both write to `slides/HE/…`, and so do `x.svs` and `x.ndpi`. The second slide's tiles overwrite the first's, and the metadata ends up with two records for each tile path.
- **Fix:**
  - Detect duplicate stems up front.
  - Disambiguate them, either with stem + a short hash of the canonical path or by mirroring the input-relative directories.
  - Add a `slide_id` column to the metadata.

#### DS-3 · MEDIUM · A run is all-or-nothing, metadata is written only at the end, and exit codes are wrong
- **Where:** `pipeline.rs:211`, `:495` (`?` on hashing), `:500-508`, `:540-546`, `:560-565`, `:590-596`, `:612`; `app/src/cli.rs:749-760`, `:808-824`; `app/src/callbacks.rs:4735-4743`, `:4789-4834`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Errors.** A single tile read or write error, or a hashing error, aborts the whole run. Completed slides are left as PNGs with no metadata.
  - **Cancel.** Cancelling writes no metadata, and a run cannot be resumed.
  - **CLI exit codes.** The CLI exits 0 on cancel, and also when every input was invalid.
  - **GUI hang.** In the GUI, a `Disconnected` result channel (worker panic) is never handled, so the modal progress dialog never closes.
- **Fix:**
  - Isolate errors per slide.
  - Write metadata incrementally (CSV or JSONL), so it survives errors and cancellation.
  - Skip existing tiles when resuming.
  - Exit non-zero on errors and 130 on Ctrl-C.
  - Handle a disconnected channel by closing the dialog.

#### DS-4 · MEDIUM · Parameters are unbounded: a tiny stride or a huge tile size aborts the process
- **Where:** `common/src/dataset/grid.rs:39-42` (`Vec::with_capacity(nx*ny)`); `app/src/cli.rs:140-145`, `:158-162`; `app/ui/dataset-export-dialog.slint:255-340`; `app/src/callbacks.rs:4687-4700`.
- **Confidence:** Confirmed.
- **Impact:**
  - A 100k×80k slide with tile 512 and stride 1 needs a 126 GB allocation, and the process aborts. In the GUI this kills the viewer.
  - Empty or non-numeric GUI fields become 1, so clearing both tile size and stride gives a 1×1 grid.
  - `--white-threshold 0` (or any negative value) skips every tile, and NaN keeps every tile.
  - The thread count is unbounded.
- **Fix:**
  - Validate ranges and show the projected tile count before starting.
  - Generate coordinates lazily.
  - Restrict the threshold to (0, 1].

#### DS-5 · MEDIUM · Empty "no data" regions are exported as black or transparent patches
- **Where:** `common/src/dataset/output.rs:36-45`, `:57-82`; this is the dataset side of WSI-2.
- **Confidence:** Likely. The code path is confirmed, and it relies on OpenSlide's documented transparent output.
- **Problem and impact:** Transparent (0,0,0,0) pixels never count as "white", so empty-glass areas of MRXS, BIF and sparse TIFF slides are exported as black or transparent tiles, and premultiplied edge pixels are written as straight alpha.
- **Fix:**
  - Composite onto the background colour.
  - Treat alpha < 255 as background.
  - Write RGB PNGs.

#### DS-6 · LOW · Smaller dataset defects
- **Input discovery** (`common/src/dataset/discovery.rs:70-100`; `pipeline.rs:24-47`):
  - Every `.dcm` file in a DICOM series is treated as a separate slide, so the same slide is extracted many times.
  - Mask TIFFs and other non-slide TIFFs next to slides are extracted.
  - Symlink loops recurse.
  - The full-file SHA-256 runs serially, cannot be cancelled, and for MRXS, VMS and DICOM only hashes the index file.
- **Progress reporting:**
  - The expected-tile total grows as slides open, so the progress bar jumps backwards.
  - "Tiles exported" also counts tiles skipped as white.
  - The ETA can read "1m60s" (`dataset-export-dialog.slint:515`).
- **GUI dialog:**
  - The default output folder is built from `$HOME`, which is unset on Windows, giving `C:\tmp\Documents\EOV\Export` (`app/src/callbacks.rs:4545-4546`).
  - The progress poll timer's `Rc` captures itself, leaking one timer per export (`:4748-4819`).
- **Docs and defaults drift:**
  - `README.md:322` says no metadata is written without `--metadata`, but both CSV and JSON are written when it is omitted (`pipeline.rs:342-345`, `cli.rs:127-128`).
  - `config.rs:33-36` documents a default threshold of `Some(0.9)`, while the CLI default is `None` and the GUI's is 0.8.
  - The help text says "exceeds" but the code uses `>=`.
  - The stain module's `maxCRef` values belong to Macenko's HERef matrix, not to the R&J target used here (`stain.rs:28-30`).
- **Dead code and tests:** `DatasetPatchesProgress` is unused, and there are no pipeline integration tests.

### 6.4 What is good here
- The patch grid is deterministic and records are sorted.
- Only full tiles are emitted.
- The cancel flag is checked per tile.
- OD clamping avoids log(0).
- Overlays are drawn after post-processing.
- The WGSL and Rust uniform layouts match (144 B), and the textures are Unorm, so there is no double sRGB conversion.
- Vahadane is a simplified alternating LASSO / least-squares scheme rather than SPAMS dictionary learning. It is deterministic, and adequate if it is fitted once per slide.

---

## 7. Application state, tools and interaction (`app/src/state.rs`, `callbacks.rs`, `tools.rs`, `ui_update.rs`, `pane_ui.rs`, `app/ui/*.slint`)

### State model
- **Open files.** Each `OpenFile` owns its WSI handle, `TileManager` and `TileLoader`, *one* ROI, a measurement list, and `pane_states: Vec<Option<FilePaneState>>` indexed by pane position.
- **Panes and tabs.** Panes hold tabs. Duplicate tabs are aliases of a backing file, and each tab's view state (`FilePaneState`) is swapped in when the tab is activated.

### Callbacks
- **Pattern.** About 100 Slint callbacks each follow the same pattern: take the write lock, mutate, call `update_tabs`/`refresh_tab_ui`, then `request_render_loop`.
- **Missing pane id.** Viewport pan, zoom, tool and minimap callbacks don't carry a pane id, so Rust uses `state.focused_pane` (§2.3).
- **Keyboard shortcuts** come from two places:
  - A winit event hook handles `f`, `m`, `+`, `-` and Ctrl+Shift+W.
  - Slint FocusScopes handle Ctrl+W, Esc, Enter, undo/redo and plugin hotkeys.

### 7.1 Findings

#### UI-1 · HIGH · Point/polygon clicks in an unfocused split pane are written to the *focused* pane's slide
- **Where:** `app/ui/viewport.slint:466-483`, `:515-519`, `:540-561`; `app/ui/panes.slint:206-218`; `app/src/callbacks.rs:1225-1231`, `:3684-3755`; `app/src/ui_update.rs:415-419`.
- **Confidence:** Confirmed (re-verified).
- **Why it happens:**
  - In Point/Polygon mode, a click on empty space only arms a pan. No `tool-mouse-down` is sent, so `pane-focused` is never called.
  - `tool-mouse-move` and `tool-mouse-up` don't focus the pane either.
  - `viewport-interaction-started` receives a pane index but ignores it (`move |_pane|`).
  - On release, Rust converts the pane-local coordinates using the focused pane's viewport and places the annotation on *that* file.
- **Impact:** With slide A on the left (focused) and slide B on the right, clicking B with the point tool persists a point on slide A, at B's local position. Polygon vertices clicked in B extend a polygon on A. This silently corrupts annotation data.
- **Fix:** Pass the pane index in every `tool-mouse-*` and `interaction-started` callback. Focus the pane on pointer-down for every tool, and ignore move/up events from a pane other than the one where the gesture began.

#### UI-2 · HIGH · Drag-to-split uses the source pane's index from before the new pane is inserted
- **Where:** `app/src/state.rs:1183-1192` (`split_tab_to_new_pane` calls `insert_pane`, then `move_tab_between_panes(id, source_pane, …)` with the stale index; re-verified), `:1005-1018`, `:1153-1181`; `app/src/callbacks.rs:3818-3865`; `app/ui/app-window.slint:694-713`; `app/ui/panes.slint:380` ("Create Left Pane").
- **Confidence:** Confirmed by replaying the functions in a Python port.
- **Impact:**
  - Single pane `[a,b]`, drag `b` to the left edge: the result is `[[], [a,b]]`. An empty pane appears, the tab doesn't move, and focus lands on the empty pane.
  - Panes `[a,x],[b]`, drag `b` to the left edge of pane 0: the result is `[[b],[a,x],[b]]`, the **same tab id in two panes**.
  - After `collapse_empty_panes`, callers re-focus the target pane by its old index, so keyboard shortcuts and the HUD act on the wrong pane.
- **Fix:** Shift `source_pane` when `insert_at <= source.0`, or look the source up by tab id. After any move or collapse, find the tab's final pane by id and focus that.

#### UI-3 · HIGH · The minimap's drag-start/end callbacks never reach Rust, so the pane is stuck "dragging": wheel zoom stops working and the render loop spins
- **Where:** `app/ui/panes.slint:140-300`: the `Viewport` instance doesn't bind `minimap-drag-start`/`minimap-drag-end`, although `PaneView` declares them (`:50-51`) and `SplitPaneContainer` forwards them (`:555-560`). Re-verified. Also `app/ui/viewport.slint:585-590`, `app/src/callbacks.rs:2343-2414`, `common/src/viewport.rs:410-433`, `:606-627`, `:716-746`.
- **Confidence:** Confirmed.
- **Why it happens:** Clicking the minimap calls `drag_center_to`, which calls `start_external_drag()`. Nothing ever calls `end_external_drag()`, and while `is_dragging` is set, `ViewportState::update()` returns early.
- **Impact:**
  - After any minimap click, wheel and `+`/`-` zoom on that pane stop working.
  - The CPU stays busy at 60 fps.
  - With viewport lock on, the stuck state is copied into the locked panes.
  - The accumulated zoom ticks apply in one jump on the next pan.
- **Fix:** Bind both callbacks, calling `pane-focused` first. As a defensive measure, have `zoom_at` clear a dangling external drag.

#### UI-4 · HIGH · Error messages are never shown: `status-text` is set everywhere but rendered nowhere
- **Where:** `app/ui/viewport.slint:37` (declared, never displayed), `app/ui/panes.slint:16`, `:152`, `app/ui/app-window.slint:63`, `:537` (re-verified). Producers include `app/src/file_ops.rs:540-545`, `:620-624` and `app/src/callbacks.rs:1444-1451`, `:1668-1672`, `:4417-4419`, `:4454-4457`, `:4682-4685`, `:4810-4815`.
- **Confidence:** Confirmed.
- **Impact:**
  - A corrupt or unsupported slide, an export with a bad extension, an unwritable directory, or a dataset export failure: nothing visible happens. The toast is used only for successes.
  - Plugin errors also go only to the log (PLG-15).
  - Users conclude the app is broken, or worse, that an export succeeded.
- **Fix:** Add a status bar plus an error variant of the toast, and use a modal dialog naming the file for open failures.

#### UI-5 · HIGH · Measurement values are wrong when MPP is missing, only partly present, or anisotropic
- **Where:** `app/src/tools.rs:75-81`, `:121`, `:142`; `app/ui/viewport.slint:277`, `:391`; `app/src/callbacks.rs:744-751`, `:801-810`; `app/ui/scale-bar.slint:9-12`.
- **Confidence:** Confirmed.
- **Why it happens:**
  - The code uses `mpp = (mpp_x + mpp_y) / 2`, and only when **both** values are present; otherwise the distance is 0 µm.
  - The "px" fallback is computed from **screen** coordinates, so it changes with zoom: a structure 1000 level-0 px long reads "100 px" at zoom 0.1 and "4000 px" at zoom 4. In exports it is measured in export-bitmap pixels.
  - For non-square pixels the length must be `hypot(dx·mpp_x, dy·mpp_y)`.
- **Impact:** Wrong measurements, which pathologists use for tumour size, margins and depth of invasion.
- **Fix:**
  - Compute per axis.
  - Fall back to the axis that is available, with a warning.
  - Report uncalibrated lengths in level-0 px, clearly labelled as such.
  - Let the user enter a calibration when MPP is missing (§14).

#### UI-6 · HIGH · Scale-bar labels are rounded to the wrong value in mm mode and below 1 µm
- **Where:** `app/ui/scale-bar.slint:34-35` (re-verified).
- **Confidence:** Confirmed.
- **Why it happens:**
  - In mm mode the label is `round(nice_um/10)/100`, so a 25 µm bar reads "0.03 mm" (a 20% error), 5 µm reads "0.01 mm", and ≤2.5 µm reads "0 mm".
  - In µm mode the label is `round(nice_um)`, so 2.5 µm reads "3 µm" and 0.25 µm reads "0 µm".
  - The bar is also scaled by the average of `mpp_x` and `mpp_y`.
- **Impact:** At native 40× (0.25 µm/px) with units set to mm, the most-used calibration aid is off by 20%.
- **Fix:**
  - Format the chosen nice value exactly, to about 3 significant digits.
  - Offer only nice values the selected unit can represent.
  - Scale a horizontal bar with `mpp_x`.

#### UI-7 · MEDIUM · Esc, Enter and Ctrl+W go to the viewport while host dialogs are open, and Esc deletes the ROI and measurement
- **Where:** `app/ui/viewport.slint:433-442`, `:733-750`; `app/ui/app-window.slint:354-377`; `app/src/state.rs:1413-1442`; `app/src/callbacks.rs:3777-3793`, `:3916-3933`; `app/ui/tab-controls.slint:833-862`.
- **Confidence:** Likely (based on Slint focus semantics; the dialogs never take focus).
- **Impact:**
  - Right-click → Export Image → Esc: the dialog stays open, and the ROI and measurement the user meant to export are deleted.
  - Ctrl+W closes the tab underneath an open export dialog.
  - Context menus can't be dismissed with Esc.
- **Fix:** Give each modal a FocusScope and focus it when the modal opens. Esc should cancel only an in-progress gesture.

#### UI-8 · MEDIUM · The window-level `f`/`m`/`+`/`-` shortcuts fire while the user is typing, and behind dialogs
- **Where:** `app/src/callbacks.rs:4898-4994` (a winit hook returns `PreventDefault` before Slint sees the key); `app/ui/export-dialog.slint:113-128`, `:1235-1238`.
- **Confidence:** Confirmed.
- **Impact:**
  - Typing "-30" into a brightness field gives "30", and the viewport behind the dialog zooms out.
  - `f` and `m` act on the viewport behind modal dialogs.
  - These keys silently shadow any plugin hotkey bound to them.
- **Fix:** Route the shortcuts through the Slint FocusScope chain, so a focused text input consumes them, and keep a single shortcut table that plugin hotkeys are validated against.

#### UI-9 · MEDIUM · On macOS, undo/redo is bound to Control instead of ⌘
- **Where:** `app/ui/app-window.slint:381-403` (it checks `e.modifiers.meta` when `use-native-window-controls` is set); `app/ui/window-toolbar.slint:38-45` (the tooltip says ⌘+Z).
- **Confidence:** Confirmed. In Slint, `control` is ⌘ on macOS and `meta` is the Control key (`i-slint-common-1.17.1/builtin_structs.rs:43-48`, re-verified).
- **Fix:** Use `e.modifiers.control` on every platform, as Ctrl+W already does.

#### UI-10 · MEDIUM · In-progress tool state survives tab, pane and file changes, so one gesture can mix coordinates from two slides
- **Where:** `app/src/tools.rs:205-224`; `app/src/state.rs:968-985`, `:1209-1217`, `:1556-1564`; `app/src/callbacks.rs:1960-1976`, `:3718-3737`.
- **Confidence:** Confirmed.
- **Impact:** With the measure tool, click once on slide A, switch to slide B, and click again: a measurement is committed to B using A's start point. An in-progress polygon behaves the same way.
- **Fix:** Reset the interaction state when the tab, pane or file changes, or store the pane and file in the state and discard events that don't match.

#### UI-11 · MEDIUM · "Split Right" on a Home tab reuses the same tab id, so opening a file collapses the split
- **Where:** `app/src/state.rs:816-826`, `:1262-1269`; `app/src/file_ops.rs:514-516`, `:584-586`; `app/src/callbacks.rs:2131-2153`.
- **Confidence:** Confirmed.
- **Fix:** Create a new Home tab for the new pane, and close only the target pane's Home tab.

#### UI-12 · MEDIUM · Dropping a file on a pane edge can leave an empty pane, and the split drop zones don't work on macOS or Wayland
- **Where:** `app/src/callbacks.rs:164-244`, `:2495-2526`, `:5071-5113`; `app/src/file_ops.rs:457-494`, `:620-624`.
- **Confidence:** Confirmed.
- **Why it happens:**
  - The new pane is inserted before the file is opened. If the open fails, or the file is already open elsewhere, the empty pane stays.
  - The cursor-position query exists only for X11 and Win32. On macOS and Wayland every drop goes to pane 0, and the split zones never highlight.
  - On Wayland, winit doesn't deliver drop events at all (SH-9).
  - Dropping multiple files on an edge creates one pane per file.
- **Fix:** Open the file first, insert the pane only on success, and use platform pointer positions.

#### UI-13 · MEDIUM · ROI and Measure tools: no toggle-off (the repo's `TODO`), results discarded on Navigate/Esc, one of each per file shared across panes, nothing persisted or undoable
- **Where:** `TODO:1`; `app/ui/window-toolbar.slint:179-194`; `app/src/callbacks.rs:2611-2633`; `app/src/state.rs:351-354`, `:1378-1442`; `app/src/tools.rs:214-221`, `:299-302`, `:319-320`.
- **Confidence:** Confirmed.
- **Problems:**
  - Clicking an active ROI or Measure button again re-selects it instead of turning it off. Plugin tool buttons do toggle (`app/src/main.rs:122-174`).
  - Switching to Navigate, pressing Esc, or a plugin calling `set_active_tool(Navigate)` clears the active file's ROI and measurement, while other files keep theirs.
  - Each new measurement replaces the previous one.
  - Drawing an ROI in pane B erases pane A's ROI for the same file.
  - Nothing is saved, undoable or visible to plugins.
- **Fix:**
  - Make the buttons toggle.
  - Keep committed ROIs and measurements independent of the current tool, with an explicit delete.
  - Allow several of each, stored per tab, persisted with the session, and exposed through the plugin API.

#### UI-14 · MEDIUM · The series bar does O(n²) `canonicalize` calls and synchronous folder scans on the UI thread, and its thumbnail workers are never cancelled
- **Where:** `app/src/state.rs:16-18`, `:874-944`, `:1631-1686`, `:1780-1812`; `app/src/file_ops.rs:50-83`, `:234-313`, `:362-411`; `app/src/callbacks.rs:451-472`.
- **Confidence:** Confirmed code; the impact on large or network folders is Likely.
- **Problem:**
  - **Re-canonicalising.** Entry paths are already canonical, yet `sync_active_file_id`, which runs several times per click, canonicalises every entry again, and each thumbnail result does the same.
  - **Synchronous scans.** `read_dir` plus a stat and a canonicalize per entry runs synchronously every time a file is opened from a new folder, even when the bar is hidden.
  - **Stale workers.** Workers for previously visited folders keep opening and thumbnailing every slide, which on these slides hits WSI-3.
- **Fix:** Index entries in a `HashMap`, enumerate in the background, have workers check a revision number in their loop, and do nothing while the bar is hidden.

#### UI-15 · MEDIUM · Scroll handling: the step ignores the scroll delta, horizontal scroll zooms out, and there is no pinch or trackpad pan
- **Where:** `app/ui/viewport.slint:97`, `:566-572`; `app/src/callbacks.rs:2295-2306`.
- **Confidence:** Confirmed.
- **Impact:**
  - Every scroll event zooms by ×1.3225 regardless of its magnitude, so trackpads and free-spin wheels hit the zoom limits almost instantly.
  - `delta-y == 0` (Shift-wheel, wheel tilt, a horizontal swipe) takes the zoom-*out* branch.
  - There is no pinch-zoom and no two-finger pan, which are the primary gestures on laptops.
- **Fix:** Use `factor = base^(delta_y/step)` with a pixel/line distinction, map horizontal and trackpad scrolling to panning, and handle pinch gestures.

#### UI-16 · LOW · Smaller interaction defects
- **Measure tool threshold.** The click-vs-drag threshold is 5 level-0 px, not screen px, so a 1-px jitter at zoom 0.02 commits a tiny measurement (`app/src/tools.rs:306-327`).
- **Units setting.** The HUD "Units" (µm/mm/in) affects only the scale bar. Measurement labels always choose µm or mm automatically, with 2 decimals on screen and 1 in exports.
- **Viewport lock:**
  - Keyboard `F`, `+` and `-` don't propagate to locked panes (`callbacks.rs:4948-4968` vs `:4140-4176`).
  - The lock links only panes showing the *same* file, and gives no feedback otherwise (`state.rs:1864-1931`).
- **Hover in the wrong pane.** Hover in an unfocused pane is hit-tested against the focused pane, so annotations highlight in the wrong pane and the sidebar's hover row flickers (`callbacks.rs:2938-3363`).
- **Tab bookkeeping:**
  - The replacement tab chosen after a close doesn't restore its own view snapshot (`state.rs:1081-1098`).
  - `close_file` filters instance states after the aliases have already been removed (`:1531-1540`).
  - `refresh_tab_ui` destroys every pane's models and caches on each tab close or move, so all panes go blank until they re-render (`main.rs:116-120`; `pane_ui.rs:144-154`).
  - A tab dropped outside the panes moves to pane 0.
  - The per-tab "Close All" closes every pane.
- **Context menus and the HUD panel:**
  - Context menus near the window edge render off-screen.
  - `separator-after` is never drawn.
  - The delete and polygon icons fall back to the open-file icon (`tab-controls.slint:841-914`).
  - Clicks on the HUD settings panel background likely pass through to the viewport (`hud-settings.slint:94-657`).
- **Pane geometry mismatches:**
  - Rust clamps the pane width to `max(100.0)`, but Slint doesn't (`render.rs:563-567`).
  - The drop-zone gap is 1 px, but the layout gap is 6 px.
  - There is no cap on the number of panes.
- **Dead or half-wired UI:**
  - The deconvolution visibility toggles have no UI.
  - The HUD fit and zoom ± callbacks are never invoked, and `ZoomInToolButton` is unused.
  - `file-dropped` is never fired.
  - The tab branches of `on_context_menu_command` are unreachable.
  - `close-split` and `split-position` exist, but there is no splitter.
  - The legacy primary/secondary pane properties remain.
  - `update_tool_overlays` is a no-op.
  - The series card's `metadata_tooltip` is computed but never shown.
- **Miscellaneous:**
  - Minimap navigation logs at `info!` twice per mouse move.
  - Ctrl+Shift+W quits immediately, with no plugin shutdown hook.
  - The "Loading…" indicator can never show, because opening is synchronous.
  - `tab_ids_for_pane_mut` and `active_tab_id_for_pane_mut` index without bounds checks (not currently reachable).

### 7.2 What is good here
- Revision counters reject stale results from the series workers.
- Each tab keeps its own viewport snapshot.
- Plugin toolbar dispatch is deferred with `Timer::single_shot(0)`, which avoids `RefCell` re-entrancy.
- `is_known_tab` guards.
- Tab and pane cycling uses `rem_euclid`, so it wraps correctly.
- Escape closes every overlay dialog (at the window level).

---

## 8. Application shell, CLI, configuration, persistence and documentation

Scope: `app/src/main.rs`, `cli.rs` (everything except update/plugin commands), `config.rs`, `clipboard.rs`, `file_ops.rs`, `backend.rs`, `build.rs`, `assets/`, all READMEs, the `TODO` file and the workspace manifest.

### Startup sequence
1. Parse the command line with clap and apply the `--config` override.
2. Initialise tracing, and run the CLI subcommand if one was given.
3. Read the config three separate times.
4. Select the Slint backend: Vulkan-only, or Metal-only on macOS.
5. Discover and activate plugins *synchronously*, before any window exists. This step extracts `.eop` tarballs and calls into the plugin libraries.
6. Create `AppWindow`, set up the plugin UI runtime, call `set_host_api`, and restore the persisted sidebar.
7. Open the files given on the command line, then call `ui.run()`.

### Persistence locations
- **Config:** `~/.eov/config.toml`, holding `render_backend`, `filtering_mode` and `active_sidebar`.
- **Recent files:** `dirs::config_dir()/eov/recent_files.txt`.
- **Caches:** `dirs::cache_dir()/eov/{plugin-packages,series_thumbnails}`.

### 8.1 Findings

#### SH-1 · CRITICAL · Re-opening a file that is already open deadlocks the UI thread
- **Where:** `app/src/file_ops.rs:457-494`. The read guard is taken at `:480`, and it is still alive when `request_render_loop` runs at `:492`, which calls `state.write()` at `app/src/main.rs:226`. Re-verified by the lead reviewer.
- **Callers:**
  - File → Open (`app/src/callbacks.rs:1140`)
  - Home-screen recent list (`:1206`)
  - Open Recent menu (`:1358`)
  - OS drag-and-drop, centre and edge (`:2487`, `:2518`)
  - Open Folder, when its first file is already open (`:632`)
  - The command line (`app/src/main.rs:470-477`)
  - The host API's `open_file` (`app/src/plugin_host.rs:1211-1226`)
- **Confidence:** Confirmed. A scratch program reproducing the same `parking_lot` pattern blocks until a watchdog fires.
- **Impact:** Any of the following freezes eov permanently:
  - choosing the top "Open Recent" entry, which is usually the slide already on screen;
  - File → Open on an open slide;
  - dropping an open slide onto a pane again;
  - re-opening a recent folder;
  - running `eov a.svs,a.svs`, which hangs before any window appears;
  - a plugin calling `open_file` on an open slide.

  This is probably the most frequently hit defect in the audit.
- **Fix:**
  - Put the read guard in its own block, as `focus_existing_file` (`callbacks.rs:474-505`) already does.
  - Add a regression test using `try_write_for`.
  - Add a debug assertion that `request_render_loop` is entered with no lock held.

#### SH-2 · HIGH · On Windows, release builds discard all CLI output, prompts and startup errors
- **Where:** `app/src/main.rs:2` (`windows_subsystem = "windows"`, with no `AttachConsole`); `.github/workflows/build-windows-zip.yml:276`; `app/src/update/prompt.rs:10-11`; `app/src/update/network.rs:24`; `README.md:65`, which tells users to add eov to PATH for cmd and PowerShell.
- **Confidence:** Confirmed (standard Windows behaviour).
- **Impact:**
  - **Silent commands.** `eov --help`, `probe`, `plugin ls`, `plugin info`, `dataset patches`, `config-path` and `recent list` print nothing, and cmd returns immediately, so a dataset export runs invisibly.
  - **Update and install.** `eov update` and `plugin install` fail because stdin is not a terminal, and the error is invisible too. Only `--allow-network -y` works, and it runs blind.
  - **Startup failures.** A bad config, a missing file, a missing Vulkan driver (RND-4) or a panic makes the app vanish without a message.
- **Fix:**
  - When a subcommand, `--help`/`--version` or a parse error is present, call `AttachConsole(ATTACH_PARENT_PROCESS)` and re-open the standard handles, or ship a separate console `eov-cli.exe`.
  - Show fatal startup errors in a native message box (`rfd::MessageDialog`).
  - Spawn the update helper with `CREATE_NO_WINDOW` and have it write a result log.

#### SH-3 · MEDIUM · A malformed, hand-edited or newer `config.toml` stops the app from starting; saving drops unknown keys and isn't atomic
- **Where:** `app/src/main.rs:274-275`; `app/src/config.rs:41`, `:103-127`, `:160-183`; `app/src/cli.rs:47`, `:56`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Startup:**
    - Parse errors, including unknown enum values, propagate out of `main`.
    - The config file spells the option `lanczos3`, but the CLI spells it `lanczos`, so writing `filtering_mode = "lanczos"` makes eov exit at launch.
    - A file written by a newer version breaks after `eov update --release <older>`.
    - A torn write breaks it as well.
  - **Saving:**
    - `save_config_field` re-serialises through `AppConfig`, which drops unknown keys and comments.
    - The write is a plain `fs::write`, with no temp file and rename.
    - Because saving also parses the file first, the UI cannot repair a broken config.
- **Fix:**
  - On a parse error: warn, back up the file and continue with defaults.
  - Accept aliases such as `lanczos`.
  - Use `toml_edit` so unknown keys and comments survive.
  - Write atomically.
  - Load the config once.

#### SH-4 · MEDIUM · Unit tests write to the developer's real config, plugin config and cache, and did so during this audit
- **Where:** `app/src/plugin_host.rs:4131-4180` and `app/src/plugins/manager.rs:767-813` (through `config::save_active_sidebar`); `plugins/eovae/src/sidebar.rs:66-77`, `:815-860`, `:1077-1123` (`persist_sidebar_preferences`); `app/src/config.rs:90-101`; `app/src/plugins/discovery.rs:315-320`, `:773-780`; `app/src/test_support.rs` (no environment isolation).
- **Confidence:** Confirmed by evidence on disk. The `cargo test --workspace` run made for this audit (14 Oct 2026, 23:16:28) caused all three of the following:
  - It rewrote `~/.eov/config.toml`, which now contains only an `[active_sidebar]` block pointing at the repo's working-copy `plugins/eovae/ui/eovae-sidebar.slint`.
  - It rewrote `~/.config/eov/eovae.json` with test values: `default_model_path: null`, 2 threads, batch 64.
  - It added about 25 fixture extractions under `~/.cache/eov/plugin-packages`, which now holds 49 entries in total, including fixtures from earlier runs.
- **Impact:**
  - Running the test suite silently changes the user's app state. Here, the next launch will load the working copy's eovae UI against the installed `eovae.eop`, and any persisted eovae model path is gone.
  - Parallel tests race on the same files.
  - The cache grows on every run.
- **Fix:**
  - Add a test-harness init that sets the config override and points `HOME` and `XDG_*` at a temporary directory.
  - Add `EOVAE_CONFIG_PATH` and `EOV_CACHE_DIR` overrides.
  - Make discovery's cache root injectable.

#### SH-5 · MEDIUM · OpenSlide linking relies on the linker's default search path, which fails with rust-lld on common setups
- **Where:** `common/Cargo.toml:20` (openslide-rs 0.2.1 uses a bare `#[link(name="openslide")]`, with no build script and no pkg-config); `app/build.rs:8-12`; `.github/workflows/ci.yml:141-146` (CI works around it with `RUSTFLAGS`/`LIBRARY_PATH`); `README.md:463-487`.
- **Confidence:** Confirmed on this machine. With OpenSlide 4.0.0 in `/usr/local/lib/x86_64-linux-gnu`, `cargo test --workspace` failed to link (`rust-lld: error: unable to find library -lopenslide`) and passed with `LIBRARY_PATH` set.
- **Problem:**
  - Rust ≥ 1.90 links with lld by default on x86_64 Linux, and lld lacks GNU ld's built-in multiarch `/usr/local` search path.
  - The only hook, `OPENSLIDE_LIB_DIR`, lives in the leaf app crate, so it does nothing for `eov-common`'s tests or for plugins. It is also undocumented and has no `rerun-if-env-changed`.
  - A source-built OpenSlide 4, which DICOM support needs, installs to `/usr/local` by default.
- **Fix:**
  - Add a `common/build.rs` that runs `pkg_config::probe_library("openslide")`, with an `OPENSLIDE_LIB_DIR` override and `rerun-if-env-changed`.
  - Document the setup, including `git submodule update --init --recursive`.
  - Declare `rust-version` (≥ 1.88).

#### SH-6 · MEDIUM · Workspace manifest problems: `exclude` is a no-op, cellaris can't be built in-tree, and submodules are hard requirements
- **Where:** `Cargo.toml:2-10`; `plugins/cellaris/Cargo.toml` (no `[workspace]` table); `plugins/cellaris/package.py:31-32`; `.gitmodules`; `app/Cargo.toml:142`.
- **Confidence:** Confirmed. `cargo metadata` lists annotations as a member, and `cargo metadata --manifest-path plugins/cellaris/Cargo.toml` fails with "current package believes it's in a workspace when it's not".
- **Problem:**
  - `plugins/annotations` appears in both `members` and `exclude`, and the member entry wins, so `exclude` does nothing.
  - Cellaris is in neither list, so it can't be built inside the eov checkout, which breaks `package.py`.
  - A non-recursive clone fails every cargo command, because the submodule members' manifests are missing.
  - The app's dev-dependency on eovae pulls ort, sqlite-vec and onnx-extractor into `cargo test -p eov`.
- **Fix:**
  - Set `exclude = ["plugins/cellaris"]`, or add cellaris as a member.
  - Document the submodule requirement.
  - Replace the eovae dev-dependency with a tiny fixture plugin.

#### SH-7 · MEDIUM · Launch arguments: one bad path aborts startup, commas split file names, and `file://` URIs are rejected
- **Where:** `app/src/cli.rs:431-448` (`split(',')` at `:436`, `trim` at `:437`), `:839-849`; `app/src/main.rs:924-950`; `assets/linux/io.eosin.eov.desktop:7` (`Exec=eov %U`).
- **Confidence:** Confirmed code. That GNOME passes URIs for `%U` is Likely.
- **Impact:**
  - `eov "Case 12, H&E.svs"` exits with "input path does not exist".
  - "Open With" from a file manager passes a `file:///…%20…` URI, and the process exits without showing a window.
  - Selecting 12 slides in a file manager opens 12 side-by-side panes.
  - Non-UTF-8 paths are rejected.
- **Fix:**
  - Collect errors per file and show them in the window instead of aborting.
  - Decode `file://` URIs with `url::Url::to_file_path`.
  - Use `OsString` arguments.
  - Treat an argument that exists on disk as a literal path before splitting on commas, or replace the comma syntax with a repeatable `--pane` flag.
  - Use `%F` in the `.desktop` file.
  - Open extra files as tabs.

#### SH-8 · MEDIUM · No single-instance mode and no OS file associations
- **Where:** `assets/linux/io.eosin.eov.desktop:1-13` (no `MimeType=`); `assets/macos/Info.plist` (no `CFBundleDocumentTypes`/`UTImportedTypeDeclarations`). There is no IPC code, and nothing handles macOS open-document events.
- **Confidence:** Confirmed (absence).
- **Impact:**
  - File managers don't offer eov for `.svs`, `.ndpi` or `.mrxs` files.
  - On macOS, double-clicking a slide or dropping it on the Dock icon delivers nothing.
  - Every launch is a new process with its own 256 MB tile cache, GPU context and plugins (two gamepad listeners, for example), and the instances clobber each other's recent-files list, config and plugin databases.
- **Fix:**
  - Forward paths to a running instance through a local socket, a named pipe or D-Bus `org.freedesktop.Application`.
  - Ship MIME and UTI declarations.
  - Handle `application:openURLs:` on macOS.

#### SH-9 · MEDIUM · Drag-and-drop doesn't work on Wayland
- **Where:** `app/src/callbacks.rs:5100-5114` (winit `DroppedFile`). winit 0.30.13 implements drop and hover events only for X11, Windows and macOS. The Slint `file-dropped` handler (`callbacks.rs:2422-2453`) is dead, because `viewport.slint:66` never fires it.
- **Confidence:** Confirmed against the winit source.
- **Impact:** On a default GNOME or KDE Wayland session, and in the Flatpak (which only has `fallback-x11`), dropping slides onto the window does nothing and gives no feedback. The README advertises drag-and-drop.
- **Fix:** Document the limitation and the `WINIT_UNIX_BACKEND=x11` workaround, follow winit's Wayland drag-and-drop work, and remove the dead handler.

#### SH-10 · MEDIUM · Copying an image to the clipboard freezes the UI for up to 2 s on Linux, and doesn't work in the Flatpak under Wayland
- **Where:** `app/src/clipboard.rs:111-123` (the UI thread blocks on `recv`), `:155-168`, `:190-202` (`wait_until(now + 2s)`); `app/Cargo.toml:91` (arboard is built without `wayland-data-control`); `packaging/flatpak/io.eosin.eov.yml:9-14`.
- **Confidence:** Confirmed for the freeze; Likely for the Flatpak.
- **Fix:**
  - Drop `wait_until`, which is unnecessary in a long-lived worker, and don't block the UI.
  - Enable `wayland-data-control`.
  - Show copy failures to the user.
  - Remove the dead `Rc<RefCell<Option<Clipboard>>>` parameter.

#### SH-11 · MEDIUM · Opening slides and browsing folders block the UI thread
- **Where:** `app/src/file_ops.rs:496-589` (`WsiFile::open` at `:499`; the second open at `:538` and the thumbnail at `:572` both run under `state.write()`), `:50-83`, `:390`.
- **Confidence:** Confirmed.
- **Impact:**
  - Opening a slide from SMB or NFS, or a large MRXS or DICOM slide, freezes the window for seconds, and the "Opening…" indicator never renders.
  - Each folder you navigate into spawns a worker that opens every slide in it (WSI-3, UI-14).
- **Fix:** Open slides on a worker thread and install the result on the UI thread.

#### SH-12 · MEDIUM · Plugin packages are extracted synchronously before the window appears, and the extraction cache is never pruned
- **Where:** `app/src/main.rs:314-321`; `app/src/plugins/discovery.rs:189`, `:336-404`.
- **Confidence:** Confirmed. On this machine `eovae.eop` is 4.1 GB, and its extraction is 3.9 GB of the cache.
- **Impact:**
  - The first launch after installing or updating a plugin silently untars gigabytes before showing any window.
  - Every version's extraction is kept. Each eovae reinstall orphans about 4 GB.
- **Fix:**
  - Extract at install time, or in the background with visible progress.
  - Garbage-collect extractions that no installed package references.
  - Support compressed `.eop` packages with deduplication.

#### SH-13 · MEDIUM · One misbehaving plugin takes down the host, and there is no safe mode
- **Where:** `app/src/plugins/manager.rs:116-170`, `:179-252`; `app/src/main.rs:328-339` vs `:391-403`.
- **Confidence:** Confirmed.
- **Problem:**
  - A panic or crash inside any vtable call, including at startup, aborts eov (PLG-7).
  - There is no `--no-plugins`, `--safe-mode` or disabled-plugins list. The only workaround is deleting the `.eop` or pointing `--plugin-dir` somewhere else.
  - `set_host_api` is called *after* registration and the filter queries.
  - There is no shutdown hook.
- **Fix:**
  - Add a `--safe-mode` flag and a `disabled_plugins` setting.
  - Disable a plugin automatically after it crashes at startup, and offer to re-enable it.
  - Call `set_host_api` first.
  - Add an `on_shutdown` callback.

#### SH-14 · MEDIUM · Recent-files persistence is lossy, racy and leaks identifiers, and is documented in the wrong place
- **Where:** `app/src/state.rs:129-130`, `:195-215`, `:721-785`; `app/src/callbacks.rs:1198-1217`; `README.md:431`; `app/README.md:103`; `HostSnapshot.recent_files`.
- **Confidence:** Confirmed code. The startup hang on dead mounts is Likely.
- **Problem:**
  - **Location.** The file lives in `dirs::config_dir()/eov/`, not `~/.eov/` as documented, and it ignores `--config`/`EOV_CONFIG`.
  - **Format.** It is one path per line, written with `to_string_lossy` and read back trimmed, so paths containing newlines, leading or trailing spaces, or non-UTF-8 bytes are lost.
  - **Writing.** The file is rewritten non-atomically under the write lock on every focus change, and concurrent instances overwrite each other (the last writer wins).
  - **Missing entries.** Entries on an unmounted drive are dropped at startup and then erased on the next save.
  - **Blocking checks.** `exists()` and `canonicalize` run on every entry at startup, which blocks on a hard NFS mount.
  - **Privacy.** The list can't be cleared, and every plugin can read it through the host snapshot. File names are often patient or accession identifiers.
- **Fix:**
  - Use an OS-string-safe structured format with atomic writes, and merge with the file on disk before writing.
  - Keep missing entries (greyed out) instead of dropping them.
  - Do path checks lazily.
  - Add "Clear history" and a setting to disable history.
  - Don't expose the list to plugins without a capability.
  - Fix the docs.

#### SH-15 · MEDIUM · Windows paths and "Open containing folder" are fragile
- **Where:** `app/src/file_ops.rs:450`; `app/src/state.rs:195-197`; `app/src/callbacks.rs:1529-1541`, `:2102-2121`, `:2587-2603`; `packaging/appimage/AppRun:6-14`.
- **Confidence:** Confirmed that `canonicalize` returns verbatim `\\?\` paths. The effects in Explorer and the AppImage are Likely.
- **Problem:**
  - **Verbatim paths.** `\\?\C:\…` and `\\?\UNC\server\share\…` paths are displayed, copied by "Copy path", saved to the recent list, and passed to `open::that`.
  - **Leaked environment.** In the AppImage, the bundled `LD_LIBRARY_PATH` leaks into `xdg-open` and the file manager.
  - **No selection, no error.** The file itself is never selected in the file manager, and failures are only logged.
- **Fix:**
  - Use `dunce::canonicalize`.
  - Reveal the file with `explorer /select,`, `open -R` or `FileManager1.ShowItems`.
  - Restore the original environment for child processes.

#### SH-16 · LOW · Logs go to stdout; there is no log file and no panic hook
- **Where:** `app/src/cli.rs:552-565`; `README.md:227`.
- **Confidence:** Confirmed: the default writer in tracing-subscriber 0.3 is stdout.
- **Problem:**
  - Warnings mix into machine-readable output. For example, `plugin info -o toml > file` also captures warnings from inspecting the plugin.
  - GUI launches keep no diagnostics.
  - Full slide paths, which may contain patient identifiers, are logged at info level about 18 times per slide open.
- **Fix:**
  - Log to stderr.
  - Add an optional rotating log file and an "Open log folder" action.
  - Add a panic hook that writes a crash report and shows a dialog.
  - Redact paths at info level.

#### SH-17 · LOW · CLI ergonomics and limits
- **`--max-tiles`.** Large values preallocate the LRU table: `1e9` commits about 2 GB at startup and `1e10` aborts. The flag has no upper bound (`app/src/cli.rs:300-307`; `common/src/cache.rs:89-94`).
- **Help text.** Most flags and subcommands have no doc comments, so `--help` output is bare (`cli.rs:251-341`).
- **Tilde expansion.** `~` is expanded for `--plugin-dir` but not for `--config`, so `--config=~/x.toml` writes `./~/x.toml`.
- **Window position.** It is applied only when both `--window-x` and `--window-y` are given, and is ignored on Wayland.
- **`config-path`.** It reports only the config file. There is no way to list every storage location: recent files, plugin cache, thumbnail cache (never evicted), and plugin databases.
- **`probe`.** It has no JSON output, no full property dump, no associated images, bounds, ICC profile or quickhash, and it doesn't print the stain label it already parses (`cli.rs:863-915`).

#### SH-18 · LOW · A dead, hidden `plugin-window` subcommand loads any directory's native library outside the install and consent flow
- **Where:** `app/src/cli.rs:239-248`, `:665-668`; `app/src/plugins/mod.rs:376-425`.
- **Confidence:** Confirmed: nothing spawns it.
- **Fix:** Remove it.

#### SH-19 · LOW · Packaging metadata drift and licence-compliance gaps
- **Where and problem:**
  - `assets/linux/io.eosin.eov.desktop:11` sets `StartupWMClass=eov`, but Slint sets the X11 class to `io.eosin.eov`.
  - `assets/linux/io.eosin.eov.metainfo.xml:5`: the licence field omits GPL-3.0.
  - `metainfo.xml:26` dates 0.4.9 to 2026-04-10, but the release commit is from 2026-09-22.
  - `assets/linux/COPYING.third-party:3-4`, `:20` point at upstream OpenSlide, but every package is built from the `Yanstart/openslide@00bebfb` fork.
  - **The macOS bundle and the Windows zip contain no LGPL notice or licence text for the bundled OpenSlide.**
  - Bundled ONNX Runtime and CUDA libraries in the plugin packages carry no licence or third-party notices.
- **Confidence:** Confirmed. The effect of `StartupWMClass` is Likely.
- **Fix:**
  - Correct each value.
  - Name the OpenSlide fork, its commit, and how to obtain its source.
  - Ship the licence texts in every package.

#### SH-20 · LOW · Documentation drift
Each item gives the claim in the docs and what the code actually does:
- **Plugin docs:**
  - The README's plugin manifest example (`README.md:354-381`) fails the real parser ("missing field `kind`"). It uses a `[plugin]` table and the wrong `[icon]` keys, and it omits the required `[environment]`.
  - The "Writing a Plugin" section (`README.md:383-414`) documents a trait route that cannot work, with a non-existent `open_plugin_window` signature and "static registration in v1" (PLG-19).
  - `plugin_api/src/lib.rs:10-13`, `app/src/plugins/registry.rs:3-7` and `app/src/plugins/mod.rs:5` describe directory-based discovery and a `register_builtin_plugins()` function that doesn't exist.
- **Self-update floor.** `README.md:266` and `:272` say self-update metadata starts at v0.4.8. The code (`app/src/update/manifest.rs:10`) and `app/README.md:76` say 0.4.5; the drift came from a mass version replace in aefd06e.
- **Dataset metadata.** `README.md:314-322` and `app/README.md:91` say metadata is written only with `--metadata`, but both CSV and JSON are written when it is omitted.
- **Persisted state.** `README.md:428-433` says eov persists "two kinds of state". It also stores `filtering_mode` and `active_sidebar`, and the recent-files list lives somewhere else.
- **Filtering mode.** `README.md:141` says it is chosen from the viewport context menu; it is actually a combo box in the window toolbar.
- **Workspace layout:**
  - `README.md:444-449` says "four crates", but the workspace has six members, cellaris is outside it, and three of the four plugins are submodules.
  - The layout section (`:524-543`) omits cellaris, `assets/`, `packaging/` and `scripts/`, and lists `fixtures/`, which is gitignored but used in the example at `:227`.
- **Build instructions:**
  - `README.md:491-495` says "Build the release binary" and then shows a debug build command.
  - The apt `cargo` in `:470-487` is too old: the code needs Rust ≥ 1.88 (edition 2024, let-chains, `as_chunks`).
  - Undocumented: submodule init, `OPENSLIDE_LIB_DIR`, and the need for OpenSlide ≥ 4 for DICOM (CZI support comes only from the fork).
- **Licence links.** The links in `app/README.md:122`, `common/README.md:118-120` and `plugin_api/README.md:166-168` use `blob/main/`, but the only branch is `master` (Likely 404).
- **Dev helper script.** `README.md:420` recommends `scripts/run-with-plugins.sh`. It silently overwrites the user's installed `~/.eov/plugins/*.eop`, deletes the extraction cache, hard-codes CUDA 12.9, and sets permissive eovae debug variables that users won't have (VAE-8).
- **README feature claims** contradicted elsewhere in this report:
  - "matched" CPU and GPU output (CLR-5)
  - "separable" Lanczos (WSI-8)
  - "deconvolution reuses the slide-specific matrix" being correct (CLR-3)
  - thumbnails (`common/README.md:11`, WSI-3)
  - the "Toggle between CPU and GPU (Vulkan)" fallback (RND-4)
  - drag-and-drop (SH-9)

---

## 9. Installation, self-update, plugin management and release engineering

Scope: `app/src/update/*`, `app/src/update_helper.rs`, `app/src/distribution.rs`, `app/src/plugins/discovery.rs` (packages), the update and plugin subcommands, `scripts/`, `packaging/`, `.github/workflows/`, and the plugin repos' packaging and release scripts. The implementation was compared against the original specification in `notes.txt`.

### How the subsystem works
- **Network consent.** `network.rs` defines a `NetworkPermission` token that can't be created by accident. The ureq agent requires it on every call, and ureq is used nowhere else.
- **Redirects.** These are followed manually, with only HTTPS GitHub hosts allowed.
- **Release manifests.** They are parsed into typed structs with SHA-256 and URL-namespace validation.
- **Plugin install flow.** plan → confirm → download → SHA-256 → stage → inspect → atomic replace, with a provenance registry.
- **App updates** use a strategy per distribution:
  - AppImage: swap the file in place.
  - Flatpak: download the bundle.
  - Cargo and Homebrew: delegate to the package manager.
  - Windows portable and macOS bundle: stage the update and hand over to `eov-update-helper`, which swaps the trees after eov exits.
- **Spec conformance.** The specification is mostly implemented: the CLI surface and aliases, schema-1 plus legacy manifests, the catalog lock, the registry and the helper.

### 9.1 Findings

#### UPD-1 · CRITICAL · Windows portable self-update replaces, then permanently deletes, the whole folder that contains `eov.exe`
- **Where:** `app/src/update/app.rs:698-705`, `:713-716`, `:836-853`, `:858-871`; `app/src/update_helper.rs:240-245`, `:294`; `.github/workflows/build-windows-zip.yml:326-330`; `README.md:63-65`.
- **Confidence:** Confirmed code path, re-verified by the lead reviewer. Whether a given user hits it depends on where they extracted the zip.
- **Problem:**
  - The "installation tree" is simply `current_exe().parent()`, with no marker file or list of shipped files.
  - The helper:
    1. renames that folder to `<parent>\.eov-backup-*`;
    2. renames the staged tree into its place;
    3. finally calls `fs::remove_dir_all(backup)`. That is a permanent delete that bypasses the Recycle Bin.
  - The release zip is built from `staging\eov\*`, so the files sit at the archive root with no top-level folder. "Extract Here" in 7-Zip or WinRAR, or dragging the files out, puts `eov.exe` directly into Downloads, the Desktop or a shared tools folder. The README then suggests putting that folder on PATH.
- **Impact:** A user who extracted into `C:\Users\me\Downloads` (or `C:\Tools`, next to other tools) runs `eov.exe update --allow-network -y`:
  - the writability checks pass;
  - Downloads becomes `.eov-backup-…`;
  - a fresh eov tree becomes `Downloads`;
  - **the user's previous Downloads folder is deleted.**
- **Fix:**
  - Ship a top-level `eov/` folder plus a manifest (e.g. `eov-portable.json`) listing the shipped files.
  - Refuse to swap unless that marker exists and the folder holds only listed files, or replace just the known files one by one.
  - Never `remove_dir_all` a directory that hasn't been proven to be eov's. Keep the backup until the next successful launch, or move it to the Recycle Bin.
  - Document "extract into a dedicated folder".

#### UPD-2 · HIGH · On a Mac without `brew` on PATH, updating a manually installed app bundle aborts
- **Where:** `app/src/update/homebrew.rs:34`, `:113-119`; `app/src/update/app.rs:430-434`.
- **Confidence:** Confirmed.
- **Problem:** `probe()` propagates the `NotFound` error from `Command::new("brew")`. "No Homebrew" should simply mean "manual bundle", but it becomes a fatal error. This happens after the user has confirmed and the plugins have been downloaded, and the staged `.part` files are left behind.
- **Fix:**
  - Treat `NotFound` as a manual bundle.
  - Also check `/opt/homebrew/bin/brew` and `/usr/local/bin/brew`.
  - Probe during planning, and show the chosen strategy in the plan.

#### UPD-3 · HIGH · Homebrew ownership detection cannot recognise the official tap's cask
- **Where:** `app/src/update/homebrew.rs:38-43`, `:113-129`, `:174-195`; `packaging/homebrew-eov/Casks/eov.rb:21-22`.
- **Confidence:** Likely. This is based on how Homebrew's `brew list` works; brew is not available on the audit machine.
- **Problem:**
  - `brew list --cask --full-name` prints tap-qualified names (`eosin-platform/eov/eov`), which the filter discards.
  - Even after a match, `brew list --cask --full-name <cask>` prints the cask name, not the artifact paths.
  - The unit tests use invented output.
- **Impact:** A Homebrew-installed `eov.app` is treated as a manual bundle, and the helper swaps it behind brew's back, which desynchronises brew's state. Otherwise every Homebrew user gets "ownership is ambiguous".
- **Fix:**
  - Use `brew info --cask --json=v2 --installed`.
  - Resolve the app path from the cask's `app` artifact.
  - Build test fixtures from real brew output.

#### UPD-4 · HIGH · The macOS `eov-update-helper` keeps an absolute CI path to libopenslide, and is run from `$TMPDIR`
- **Where:** `app/build.rs:12` (a package with no lib target passes `rustc-link-lib` to *every* bin; verified with a scratch crate); `app/Cargo.toml:15-21`; `packaging/macos/build.sh:215-240`, `:287-298`, `:316-325`; `app/src/update/app.rs:889-911`.
- **Confidence:** Likely. ld64 retention of the dylib dependency was not checked on a Mac.
- **Problem:**
  - `bundle_direct_dependencies` rewrites install names only in the main binary.
  - The build checks only that the helper file exists.
  - Even with `@executable_path/../Frameworks`, the copy placed in `$TMPDIR` couldn't resolve the library.
- **Impact:**
  1. eov prints "will restart to finish the update" and exits.
  2. The helper then dies in dyld ("Library not loaded").
  3. Nothing is replaced or relaunched.
  4. A hidden staged bundle, the transaction file and the staged plugins are left behind.
- **Fix:**
  - Don't link openslide into the helper: move the link directive into `eov-common`, or give `app` a lib target.
  - Assert `otool -L` output in CI.
  - Smoke-run the helper before exiting eov.

#### UPD-5 · HIGH · The Flatpak sandbox breaks the updater, the gamepad plugin and folder-based workflows
- **Where:** `packaging/flatpak/io.eosin.eov.yml:10-15` (`finish-args`: `ipc`, `fallback-x11`, `wayland`, `--device=dri`, `--filesystem=home`; re-verified); `app/src/update/app.rs:360-374`, `:1144-1149`, `:1162-1163`.
- **Confidence:** Confirmed (manifest), Likely (runtime effects).
- **Problem:**
  - **No network.** There is no `--share=network`, so `eov update` and `eov plugin install` fail with DNS errors.
  - **Update bundle path.** The bundle is written with `create_new` to a fixed path, so a re-run fails with EEXIST. The printed instruction always uses `--user`.
  - **No input devices.** Without `--device=input` (Flatpak ≥ 1.15.6) or `--device=all`, the gamepad plugin cannot see controllers.
  - **Home only.** With `--filesystem=home`, slides under `/media` or `/mnt` are reachable only through the file-chooser portal. The portal hands over a single file, which breaks MIRAX sidecar directories and folder series.
  - **Clipboard and drag-and-drop.** See SH-9 and SH-10.
- **Fix:**
  - Grant `--share=network`. Access is already gated by the consent prompt.
  - Grant `--device=input`.
  - Document `flatpak override` for other filesystems, or request `--filesystem=host:ro`.
  - Write the update bundle via a temp file and rename.

#### UPD-6 · HIGH · Plugin compatibility metadata cannot prevent installing or loading ABI-incompatible plugins
- **Where:** See PLG-1. Also `plugins/gamepad/.github/workflows/release.yml:82-83` and `build-linux-eop.yml:67-95` (official plugin releases build against eov master HEAD); `plugins/gamepad/plugin.toml:9-10`; `plugins/eovae/plugin.toml:6-7`; `plugins/cellaris/Cargo.toml:21`.
- **Confidence:** Confirmed (mechanism).
- **Problem:**
  - Install and discovery accept any package whose open-ended `environment` range matches.
  - That range is written by hand and says nothing about the API the plugin was compiled against.
  - Plugin release builds have no lockfiles.
- **Fix:**
  - Derive the range (e.g. `>=0.4.9, <0.5.0`) from the API version actually linked.
  - Build plugins against released eov tags.
  - Check the ABI symbol during inspection as well as at load.

#### UPD-7 · MEDIUM · Windows-specific update failures
- **Running from inside the install folder.** `cd C:\Tools\eov; .\eov.exe update` fails with a sharing violation, because the helper inherits eov's working directory and the shell's open handle blocks the rename. There is no retry, and a staged copy is left behind on each attempt (`app/src/update/app.rs:905-909`; `app/src/update_helper.rs:240-245`). *Likely.* Fix: spawn the helper with `current_dir(temp_dir())` and retry renames.
- **Downgrading to v0.4.5 is impossible.** v0.4.5 is the specification's mandated floor. Staged-tree validation requires `eov-update-helper.exe`, which the v0.4.5 zip doesn't contain (`app.rs:731-739`). *Confirmed* via `git show v0.4.5:…`.
- **Helper robustness:**
  - "Process is alive" is decided by `OpenProcess` succeeding, which PID reuse can fool; the timeout is 60 s.
  - Renames aren't retried while antivirus holds locks.
  - A failed rollback leaves the install at a hidden path.
  - The backup is deleted before the relaunch has been shown to work.
  - Staging leftovers remain after a failure.
  
  Locations: `update_helper.rs:58-92`, `:185-219`, `:294`. *Likely.*

#### UPD-8 · MEDIUM · Homebrew "latest" delegation is never verified, yet plugins are aligned to the target version anyway
- **Where:** `app/src/update/app.rs:436-445`; `app/src/update/homebrew.rs:93-111`; `.github/workflows/release.yml:208-285` vs `:422-507`.
- **Confidence:** Likely.
- **Problem:**
  - `brew upgrade --cask eov` runs without refreshing the tap, and the release workflow publishes the GitHub release *before* it updates the tap. So on release day brew does nothing.
  - eov still prints "Homebrew updated EOV to X" and installs plugins that require the new version, and those plugins are then skipped on the next launch.
  - For prerelease targets, eov uninstalls the current cask before discovering that `eov@<pre>` doesn't exist.
- **Fix:**
  - Run `brew update`, or install the exact versioned cask.
  - Read back the installed version before aligning plugins.
  - Reject prerelease targets before uninstalling anything.

#### UPD-9 · MEDIUM · A third-party GitHub install can silently replace an unrelated installed plugin
- **Where:** `app/src/update/plugins.rs:353-399`, `:419-435`, `:480-498`, `:628-642`; `app/src/update/manifest.rs:41-52`.
- **Confidence:** Confirmed.
- **Problem:**
  - `[plugin].id` is optional, so the plan shows "ID: (discovered after download)".
  - After download, any inner id is accepted, and `replace_package_atomically` removes every package with that id without asking again.
  - A fork that declares `id = "annotations"` therefore replaces the official plugin, and the provenance silently flips to `github`.
- **Fix:**
  - Require ids for third-party sources.
  - Re-plan and re-prompt when an id collides with an installed plugin.
  - Refuse to shadow official ids unless the user forces it.

#### UPD-10 · MEDIUM · One failing plugin aborts the whole `eov update`, including the app update
- **Where:** `app/src/update/app.rs:276-298`, `:339`, `:545-582`. Only third-party *resolution* errors are downgraded to warnings (`:211-215`).
- **Confidence:** Confirmed.
- **Problem:** This contradicts spec §26. A third-party repository that deletes or re-uploads an asset blocks every future app update.
- **Fix:** Skip the plugin with a stated reason and continue. Fail hard only on errors with the app's own artifact.

#### UPD-11 · MEDIUM · Integrity relies only on hashes published in the same place as the artifacts; nothing is signed
- **Where:** `app/src/update/manifest.rs:21-38`; `app/src/update/plugins.rs:567-626`; `app/src/update/app.rs:1060-1076`; `scripts/release_infrastructure.py:728-743`. Plugin release workflows upload with `--clobber`.
- **Confidence:** Confirmed (by design).
- **Problem:**
  - Every expected SHA-256 lives in a `release.toml` stored in the same GitHub release as the artifact.
  - The eov catalog pins plugin *versions*, not bytes.
  - This protects against corrupted downloads, but not against a compromised maintainer token or account. That matters because plugins are native code that eov runs in-process.
- **Fix:**
  - Put per-platform plugin artifact hashes into the eov release catalog.
  - Sign `release.toml` (minisign or sigstore) with a public key embedded in the binary.
  - Enable GitHub immutable releases.
  - Sign the Windows binaries (Authenticode) and notarize the macOS app.

#### UPD-12 · MEDIUM · Plugin-cache invalidation can break a running GUI, and the cache and staging files leak without bound
- **Where:** `app/src/plugins/discovery.rs:63-118`, `:383-404`; `app/src/update/plugins.rs:217-219`, `:283`, `:461-470`, `:1175-1178`; `app/src/update/app.rs:354-356`, `:434`, `:461-474`, `:546-552`.
- **Confidence:** Confirmed (leaks); Likely (effect on a running GUI).
- **Problem:**
  - **Over-broad deletion.** Invalidation `remove_dir_all`s every cache entry whose name starts with `<stem>-`, which also matches unrelated packages such as `gamepad-extras-*`.
  - **Running GUI.** The GUI loads `.slint` files from the cache lazily, so after `eov plugin update annotations` the running GUI can't open that plugin's sidebar. On Windows the delete half-succeeds because the DLL is locked.
  - **Leaks.** Staged packages are extracted into the shared cache and never removed, the helper never invalidates old entries, nothing garbage-collects the cache, and early returns leak the staged app tree.
- **Fix:**
  - Key cache entries by content hash, matched exactly.
  - Run GC at startup rather than deleting entries that may be in use.
  - Clean up staging on every error path.

#### UPD-13 · MEDIUM · Network and platform reach
- **Proxies and TLS interception.** The updater fails behind an HTTP proxy or a TLS-intercepting firewall, which is common in hospitals. ureq is built with only `tls` (rustls with bundled webpki roots), with no `proxy-from-env` and no native certificate store (`app/Cargo.toml:104`; `app/src/update/network.rs:68-76`). *Confirmed.*
- **AppImage glibc floor.** The AppImage requires glibc ≥ 2.39 because it is built on ubuntu-24.04 and glibc can't be bundled. It therefore fails on RHEL/Rocky 8–9, Ubuntu 22.04 and Debian 12, which are common in clinical and research IT (`.github/workflows/build-appimage.yml:58-72`; `packaging/appimage/build.sh:31-45`, `:290-291`). *Likely.* Fix: build in an older baseline container and check the maximum symbol version in CI.
- **macOS deployment target.** The bundle declares macOS 11 (`assets/macos/Info.plist:24-25`, cask `:big_sur`), but its dylibs are built for the runner's macOS without `MACOSX_DEPLOYMENT_TARGET` (`.github/workflows/build-macos-app.yml:55-76`; `packaging/macos/build.sh:215-276`). *Likely.*
- **Leaking `LD_LIBRARY_PATH`.** The AppImage's `AppRun` leaks the bundled `LD_LIBRARY_PATH` into every child process: `xdg-open`, file managers and browsers (`packaging/appimage/AppRun:6-12`). *Likely.*

#### UPD-14 · MEDIUM · CI exercises none of the platform-specific updater code, and supply-chain hygiene is weak
- **Where:** `.github/workflows/ci.yml:21-31`, `:92-153`; `.github/workflows/release.yml:19-117`; `.github/workflows/build-appimage.yml:152-165`; plugin workflows.
- **Confidence:** Confirmed.
- **Problem:**
  - CI runs on Linux only, so the `cfg(windows)` helper, Homebrew and bundle code are first compiled at release time and never tested.
  - The Python release tests are never run, and two of the plugin repos' Python tests currently fail.
  - A release doesn't require green CI, and `/noci` commits skip CI entirely.
  - Actions are pinned to mutable tags rather than commit SHAs.
  - The "continuous" builds of linuxdeploy and appimagetool, plus uthash and libdicom, are fetched without checksums.
  - Plugin builds have no lockfiles.
  - `CARGO_REGISTRY_TOKEN` is present while build scripts run.
- **Fix:**
  - Add Windows and macOS CI jobs, and run the Python tests.
  - Gate releases on green CI for the source SHA.
  - Pin actions by SHA, and pin tools by version and SHA-256.
  - Commit the plugin lockfiles.
  - Use crates.io trusted publishing.

#### UPD-15 · LOW · Smaller update and packaging defects
- **AppImage replacement isn't atomic.** The running file is renamed to a backup *before* the new one is renamed in, and the directory isn't fsynced. A crash in between leaves only the hidden backup (`app/src/update/app.rs:1113-1127`).
- **Prompts:**
  - Ctrl-D (EOF) counts as "Y" at both the network-consent and the confirmation prompts.
  - A no-op update still prompts and then reports "Update applied successfully" (`app/src/update/network.rs:37-44`; `prompt.rs:17-19`).
- **The update plan display:**
  - Plugin upgrades that will be skipped on Flatpak or Unknown distributions are still listed (spec §39).
  - Distribution names are printed in Debug form (`MacosBundle`).
  - There is no Official / Third-party / Unmanaged grouping, and no explanation for incompatible plugins (spec §24/§26).
  - Every network failure is reported as "No published release manifest exists".
- **`eov plugin ls`:**
  - Managed GitHub plugins show as "unmanaged".
  - Declining network access mislabels official plugins.
  - Without `--allow-network` in a non-interactive shell it fails instead of listing local plugins, which is always the case on Windows (SH-2).
- **`plugin info` and `plugin remove`:**
  - `info -o toml` fails on duplicate ids.
  - Neither command shows which duplicate discovery will load (spec §23).
  - Packages that fail to parse are invisible, so they can't be removed (spec §14).
- **The release tooling's semver evaluator disagrees with Rust's `semver`** (`scripts/release_infrastructure.py:182-269`): `>0.4`, `||`, a bare `0.4.1`, `~1`, `^0.0`, `<=0.4` and `>= 0.4.1` (with a space) are all handled differently. The pipeline could therefore publish a catalog entry that eov rejects.
- **Version edge cases.** Prerelease hosts treat every plugin as incompatible (`>=0.4.1` does not match `0.5.0-rc.1`). A plain `eov update` run from a build newer than the latest release plans a downgrade.
- **Plugin id and filename handling** (overlaps PLG-13):
  - Ids may be `C:x` or `a:b` (an NTFS alternate data stream), or a reserved device name.
  - `<id>.eop` is overwritten even when it belongs to an unrelated package.
  - Repositories are compared as raw strings.
- **Third-party resolution:**
  - It calls the unauthenticated GitHub API, which allows 60 requests per hour, and reads only the first page.
  - It downloads every listed release's manifest even when the latest one is compatible.
  - It records a synthesised `v{version}` tag.
  - It rejects tags containing `/`.
  - It never compares the artifact URL's tag with the requested one (spec §42).
- **Memory.** Downloads are buffered entirely in memory, with no size cap on manifests, and every package is re-hashed for each command.
- **Temporary files.** Temp names are pid + nanoseconds rather than random (spec §48), and the helper is copied to a guessable temp path and then executed.
- **Release order.** The GitHub release goes live *before* the crates.io publish and the Homebrew tap update.
- **Cargo ownership check.** It looks only at `$CARGO_HOME/bin/eov`, ignoring `CARGO_INSTALL_ROOT` and `.crates2.json` (spec §32).
- **Homebrew cask.** The `zap` paths don't match where eov actually writes: `~/.eov`, `~/Library/Application Support/eov` and `~/Library/Caches/eov`.
- **Archive permissions.** tar-rs applies archive modes with mask 0, so world-writable files can land in the plugin cache (use `set_mask(0o022)`).
- **Tests.** There is no injectable HTTP transport, so installs, planning, consent versus `-y`, SHA and manifest mismatches, and rollback are untested. Discovery tests also write into the real `~/.cache` (SH-4).

### 9.2 What is good here
- The network-permission token is threaded through every HTTP call.
- Redirects are followed manually against an HTTPS host allow-list.
- Artifact URLs are pinned to the repository's release namespace.
- Direct downloads are verified with SHA-256 before anything is changed.
- `.eop` extraction rejects symlinks, hardlinks, devices, FIFOs, traversal, absolute and drive paths, and multiple manifest roots, and inspection never loads native code.
- The canonical `<id>.eop` naming makes downgrades work.
- The registry is written with temp file + rename.
- The helper rejects unknown transaction fields.
- The release flow builds first and tags later, verifies the draft, and byte-compares `release.toml`.

---

## 10. Plugin: annotations (`plugins/annotations`, v0.2.4, git submodule)

### How it works
- **Global state.** One `Mutex<PluginState>` (`src/state.rs:43-68`) holds everything:
  - an in-memory cache of layers with points and polygons in level-0 pixel coordinates, keyed by the slide's *path string*;
  - per-path selection, visibility and collapse maps;
  - **one global undo and redo stack** (256 entries);
  - the state of pending import and delete dialogs.
- **Storage.** A single SQLite file, `dirs::config_dir()/eov/annotations.db` or `$EOV_ANNOTATIONS_DB` (`src/db.rs:11-35`).
  - Every operation opens a new connection and re-runs the whole DDL.
  - Layers are keyed by the 32-byte sampled fingerprint (WSI-13).
  - There are no transactions, no indexes, no WAL and no schema version.
- **Host interaction.** The host polls `get_viewport_overlay_*` every frame and on every hit test. The plugin returns *all* annotations of the pane's file (`src/lib.rs:201-333`).
- **Sidebar.** It receives about 20 JSON properties on every refresh, including one tree row per annotation.
- **Features:**
  - layers: create, rename, recolour, hide and delete;
  - points and polygons: place, move and delete;
  - key/value metadata per annotation;
  - JSON export and import, with a full-file SHA-256 check and a merge/replace/skip conflict dialog;
  - inverse-action undo;
  - a density panel for the selected polygon: area, point count, count per mm², per-HPF counts and an exact Poisson interval.

### 10.1 Findings

#### ANN-1 · HIGH · Every write is a separate fsync'd transaction on the UI thread: seconds-to-minutes freezes, and multi-step edits aren't atomic
- **Where:**
  - polygon placement: `src/operations.rs:2182-2208`
  - move and vertex edit: `:1827-1841`
  - import: `:594-625`, `:702-742`
  - undo/redo: `src/history.rs:365-409`, `:565-600`
  - metadata: `src/db.rs:210-238`
- **Confidence:** Confirmed. The lead reviewer re-verified that there are no `BEGIN`, `transaction`, `journal_mode` or `synchronous` calls anywhere. Timings were measured with the exact schema and statement pattern on SQLite 3.45 (NVMe/ext4).
- **Measurements:**

  | Operation | As written | In one transaction |
  |---|---|---|
  | Place a 50-vertex polygon | 518 ms | 10.5 ms |
  | Place a 500-vertex polygon | 4.0 s | 11 ms |
  | Import, per point | 16 ms (≈2.7 min for 10k points) | — |

  - Dragging one vertex rewrites every vertex.
  - Undoing a large layer deletion replays every row.
- **Atomicity:** An error part-way through leaves a partial polygon or ghost rows, while the in-memory state is unchanged. Possible causes:
  - `SQLITE_BUSY` from a second eov instance (5 s timeout);
  - a full disk;
  - a CHECK constraint failure;
  - the user force-quitting the "hung" app.

  A move is recorded as Delete + Create (`operations.rs:1774-1795`), so a failure between the two halves deletes the annotation.
- **Fix:**
  - Keep one persistent connection and run the DDL once.
  - Use `journal_mode = WAL` and `synchronous = NORMAL`.
  - Wrap every logical operation in a single transaction, using `prepare_cached` statements.
  - Update memory only after a successful commit.
  - Move the I/O off the UI thread.

#### ANN-2 · HIGH · The density/area panel goes stale after a polygon is reshaped or moved, or a point is moved
- **Where:** `src/lib.rs:371-383`, `:418-429` (the move callbacks only `request_render`; re-verified); the host doesn't refresh either (`app/src/callbacks.rs:2748-2763`, `:3409-3430`, `:3505-3523`). Density is computed only in `get_sidebar_properties` (`src/sidebar.rs:571`).
- **Confidence:** Confirmed.
- **Impact:** Select a region and the panel shows "Region 0.85 mm², Mitoses: 12 points, 14.1 / mm²". Drag vertices to exclude necrosis, or drag a counted mitosis out of the region, and the panel keeps the old area, count and density until some unrelated click refreshes it. Pathologists copy these numbers into reports.
- **Fix:** Call `refresh_sidebar` after every successful move or vertex edit, and recompute only the affected layer's density.

#### ANN-3 · HIGH · Area and point counting use different rules for self-intersecting polygons, and nothing prevents such polygons
- **Where:** `src/density.rs:23-33` (signed shoelace, then `abs`), `:40-55` (even-odd point-in-polygon), `:109-112`; the only geometry check is `len() < 3` (`src/operations.rs:1805`, `:2151`); vertex drag and edge insert can create crossings (`app/src/callbacks.rs:100-127`).
- **Confidence:** Confirmed numerically.
- **Impact:**
  - A bowtie `(0,0),(1000,1000),(1000,0),(0,1000)` has area 0, so the panel says "That polygon encloses no area", although both lobes are filled on screen.
  - A twisted polygon under-reports its area by 16.5%, which over-reports density by about 20%.
  - Mitotic density is a grading criterion for breast carcinoma, GIST, neuroendocrine tumours and others, so a 20% error can change a grade.
- **Fix:**
  - Reject or repair self-intersections when a polygon is placed or edited, and tell the user.
  - Or compute the area with the same fill rule as the counting, e.g. as the union of simple pieces via `geo` BooleanOps.

#### ANN-4 · MEDIUM · A crafted or corrupted import makes eov abort every time that slide is opened
- **Where:** `src/model.rs:148-160` (`hex_color_to_rgb` checks the *byte* length and then slices at byte offsets; re-verified); callers `src/lib.rs:232`, `:301`, `src/sidebar.rs:40`; import inserts without validation at `src/operations.rs:707-741`.
- **Confidence:** Confirmed. A standalone reproduction exits with code 134 (abort).
- **Problem:**
  - `"##aébcd"` satisfies `CHECK(length(color)=7)`, because SQLite counts characters.
  - After trimming it is 6 bytes long, and byte 2 falls inside "é", so the slice panics. This happens in an `extern "C"` callback, so the process aborts.
  - The layer is already persisted, so every later open of that slide aborts eov until someone hand-edits the DB.
  - Also unvalidated on import:
    - names over 255 characters, and colours that aren't 7 characters long, fail part-way through the import;
    - polygons with 0–2 vertices and out-of-image coordinates are accepted;
    - a 0-vertex polygon gives `frame_sidebar_annotation` a NaN centre (`src/sidebar.rs:270-322`), which is passed into the viewport.
- **Fix:**
  - Validate before writing: colour `^#[0-9A-Fa-f]{6}$`, name length, at least 3 finite vertices, metadata size.
  - Wrap every FFI entry point in `catch_unwind` (PLG-7).

#### ANN-5 · MEDIUM · Import isn't atomic: "Replace" can lose the existing layer, and "Cancel" makes already-imported layers impossible to undo
- **Where:** `src/operations.rs:517-520`, `:838-870` (the existing layer is deleted from the DB first), `:910-920`, `:1186-1190`; `src/history.rs:279-284` (a failed undo is pushed back on top of the stack).
- **Confidence:** Confirmed (traced).
- **Impact:**
  - **Replace that fails:** import a conflicting layer whose colour is `#FF0000FF` and choose Replace.
    - The old layer is deleted from the DB and from memory.
    - The insert of the new layer then fails its CHECK.
    - The error is only logged, and the deletion is never recorded in the undo history.
  - **Cancel part-way:** after choosing Replace at the first conflict and Cancel at the second, the layers already applied can't be undone.
  - **Stuck undo stack:** an undo that fails is pushed back on top every time it's tried, so all older history becomes unreachable.
- **Fix:**
  - Validate first, then apply each import in a transaction.
  - On Cancel, either roll back or record what was already applied as one undoable action.

#### ANN-6 · MEDIUM · All slides share one undo/redo stack, which can undo edits on a closed slide with nothing visible on screen
- **Where:** `src/state.rs:59-60`; `src/history.rs:241-258` (always `enabled: true`), `:260-317`, `:319-332` (the file cache is never evicted).
- **Confidence:** Confirmed.
- **Impact:** Place 20 points on slide A, switch to slide B, place one point, then press Ctrl+Z twice. The second undo deletes a point on A, even if A's tab is closed, and nothing on screen changes.
- **Fix:** Keep one stack per slide, and publish undo/redo availability for the active slide whenever it changes. This needs the host's active-file-changed event (§3.3).

#### ANN-7 · MEDIUM · The in-memory cache is keyed by path string and never invalidated, while the DB is keyed by fingerprint
- **Where:** `src/state.rs:45`; `src/operations.rs:64-87`; `src/history.rs:565-600`.
- **Confidence:** Confirmed within one process; Likely across two processes.
- **Impact:**
  - **Same slide opened through two paths** (a copy, a symlink, UNC vs mapped drive, different letter case on Windows): the two paths get independent caches. Delete layer L in one tab, then place a point into L in the other: the insert fails its foreign key, which is only logged, so the click does nothing.
  - **Two eov instances:** each misses the other's edits. A layer deleted in one instance cascade-deletes points the other instance added, and the undo snapshot doesn't include them, so they are lost permanently.
  - **File replaced at the same path:** new annotations are stored under the old slide's identity.
- **Fix:**
  - Key the cache by fingerprint, and re-check size and mtime on load.
  - Check `PRAGMA data_version` before applying cached state.
  - Warn when a second instance is running (SH-8).

#### ANN-8 · MEDIUM · Slide identity is a sampled byte hash with no recovery path
- **Where:** `common/src/file_id.rs:54-77`; `src/db.rs:46-54` (no filename, size or dimensions are stored).
- **Confidence:** Likely for in-place modification; Speculative for MIRAX collisions.
- **Impact:**
  - **In-place changes** orphan all of a slide's annotations, with no UI to find or relink them. Examples: editing a TIFF tag, de-identification that strips the label image, re-saving the file.
  - **MIRAX:** the hash covers only the `.mrxs` index, so identical or empty index stubs would share annotations.
  - **Copies** of a slide deliberately share annotations, which surprises users doing an independent second read.
- **Fix:**
  - Store `openslide.quickhash-1` together with the filename, dimensions and MPP.
  - Add an "orphaned annotations → relink to this slide" view.
  - Make sharing between copies explicit.

#### ANN-9 · MEDIUM · An import lands on whichever slide is active when the dialog is answered, not the slide it was started on
- **Where:** `src/operations.rs:971-1001`, `:1132-1191`; `app/src/callbacks.rs:1960-1971`.
- **Confidence:** Confirmed (traced).
- **Problem:**
  - Start an import on slide A. While the "different WSI" warning is showing, click slide B's tab, then click Import.
  - The layers are written under B's fingerprint, although the SHA-256 check was made against A.
  - Conflict detection is re-run against B.
- **Fix:**
  - Store the target path and fingerprint in `PendingImport`.
  - Cancel the import if that file is closed.

#### ANN-10 · MEDIUM · Uncommitted metadata edits are lost, and global hotkeys can get stuck disabled
- **Where:** `ui/annotations-sidebar.slint:36-37`, `:707-761`, `:771-801`; `app/src/plugin_host.rs:3363`; `app/src/main.rs:863-866`.
- **Confidence:** Likely. Traced through the Slint 1.17 sources; not run.
- **Problem:**
  - Metadata inputs commit only on Enter or when they lose focus, and focus is tracked with a counter.
  - Clicking "Add metadata row", ×, or a tree row doesn't move focus, but it does refresh the sidebar.
  - The refresh replaces the model, which destroys the focused TextInput without firing its focus-change callback.
- **Impact:**
  - The typed value is lost.
  - The counter stays at 1, so `capture-hotkeys` stays true. The host then disables Ctrl+Z, Ctrl+Shift+Z and the tool hotkeys until the sidebar is closed and reopened.
- **Fix:**
  - Commit drafts on `edited` (debounced).
  - Derive `capture-hotkeys` from actual focus rather than a counter.
  - Don't replace models while a field is being edited.

#### ANN-11 · MEDIUM · The selection "pulse" thread forces about 8 full re-renders per second indefinitely, and the colour never actually pulses
- **Where:** `src/operations.rs:47-62`; `src/lib.rs:335-338` (`pulsing_selection_color` ignores its input); `app/src/plugin_host.rs:1458-1473`.
- **Confidence:** Confirmed.
- **Problem:**
  - Each call takes the `AppState` write lock, rebuilds the overlay factories and re-polls every overlay.
  - With no file-closed event, a selection left on a closed slide keeps this running for the rest of the session.
- **Fix:**
  - Remove the thread and draw a static highlight.
  - Clear per-file state on the host's file-closed event (§3.3).

#### ANN-12 · MEDIUM · Overlay and sidebar costs grow with the total number of annotations
- **Where:**
  - overlays: `src/lib.rs:201-333` (no culling; dummy structs are built for polygons in the point path, `:254-265`);
  - sidebar: `src/sidebar.rs:24-85`, `:589-592`;
  - density: `src/density.rs:116-124`, O(points × vertices) on every refresh;
  - layout: `ui/annotations-sidebar.slint:296-300` (`VerticalLayout` + `for` rather than a virtualised `ListView`);
  - host: a brand-new model on every refresh (`app/src/plugin_host.rs:3321-3374`).
- **Confidence:** Confirmed (behaviour); magnitude estimated.
- **Impact:**
  - Each polygon-tool mouse move re-fetches every polygon three times.
  - Every click, placement or toggle serialises about 260 B of JSON per annotation, and Slint re-instantiates every row.
  - With an imported 50k-cell layer, each refresh becomes a multi-megabyte, multi-second rebuild.
  - Tree rows are labelled only "Point" or "Polygon".
- **Fix:**
  - Cull with a spatial index; the bundled SQLite has RTREE.
  - Add a revision counter, and provide a plugin-side hit test.
  - Send layer summaries plus a paged model to the sidebar.
  - Cache the density.

#### ANN-13 · MEDIUM · Failures are retried every frame, a read-only database is unusable, and the sidebar can show the previous slide
- **Where:** `src/operations.rs:64-87` (no negative cache), `:170-185`; `src/lib.rs:204-210`, `:273-279`; `src/db.rs:138-144` (opening the DB always writes, via `INSERT OR IGNORE`).
- **Confidence:** Confirmed. The read-only failure was reproduced with sqlite3.
- **Problem:**
  - **Retries.** A failed fingerprint or DB open is retried for every overlay call: 256 file reads, a DB open with up to a 5 s busy wait, and an error log line, several times per frame.
  - **Read-only DB.** Opening the database always writes, so a read-only one fails with "attempt to write a readonly database". A teaching DB shared read-only through `$EOV_ANNOTATIONS_DB` can't even be viewed.
  - **Stale sidebar.** When the sync fails, the previous slide's annotations stay in the sidebar, and their density is computed with the new slide's MPP.
- **Fix:**
  - Cache failures per path.
  - Open read-only when the file isn't writable, and skip the DDL.
  - Switch the active slide even when loading it fails, and show an error state.

#### ANN-14 · MEDIUM · The first import or export hashes the whole slide file on the UI thread, while holding the plugin mutex
- **Where:** `src/operations.rs:1100-1108`, `:1963-1976`; `common/src/file_id.rs:43-52`.
- **Confidence:** Confirmed.
- **Impact:**
  - **Freeze.** Hashing a 1–10 GB slide, often on a network share, freezes the app for seconds to minutes with no progress indicator.
  - **Stale hash.** A slide replaced at the same path keeps its stale hash for the rest of the session.
  - **Spurious warning.** Exports from older versions stored the sampled fingerprint instead, so importing them always triggers the mismatch warning.
- **Fix:**
  - Hash in the background with progress, or compare quickhash instead.
  - Key the hash cache on (path, size, mtime).

#### ANN-15 · MEDIUM · Errors are never shown to the user
- **Where:** `src/sidebar.rs:531-533`; `src/lib.rs:119-121`, `:367`, `:381`, `:394`, `:427`, `:440`; the host only logs (`app/src/plugin_host.rs:1657-1666`).
- **Confidence:** Confirmed.
- **Impact:**
  - **Export.** An export that fails on a permissions error shows nothing.
  - **Import.** Importing a QuPath GeoJSON does nothing visible; the only trace is "missing field file_path" in the log.
  - **Rename.** Renaming a layer to more than 255 characters silently reverts.
- **Fix:**
  - Show a banner or dialog for errors.
  - Enforce length limits in the UI.

#### ANN-16 · MEDIUM · The schema has no version, no migrations and no indexes
- **Where:** `src/db.rs:37-149`, `:59`, `:288`, `:307-326`.
- **Confidence:** Confirmed (measured).
- **Problem:**
  - There is no `user_version`.
  - Adding a new annotation type means changing the `type` CHECK constraint, which in SQLite requires rebuilding the table.
  - Neither `annotation_layers(fingerprint)` nor `annotations(annotation_layer_id)` is indexed. With 400k annotations, opening a slide takes 41 ms, against 2.4 ms with indexes, and the cost grows linearly with the total DB size.
- **Fix:** Add migrations driven by `user_version`, and create both indexes.

#### ANN-17 · MEDIUM · The export format has no version or image context, and leaks the absolute local path
- **Where:** `src/model.rs:93-140`; `src/operations.rs:1957-2054`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Missing context.** The export records no format version, image size, MPP, units, coordinate convention or OpenSlide bounds offset (WSI-14).
  - **PHI in paths.** The absolute `file_path` is always included, and clinical folder and file names often contain accession numbers or patient names.
  - **Non-atomic write.** `fs::write` is not atomic.
- **Fix:**
  - Add a `format_version` and slide descriptors to the export.
  - Make exporting the path optional.
  - Write to a temp file and rename it into place.
  - Add GeoJSON (QuPath-compatible) as the first interchange format (§14).

#### ANN-18 · LOW · Smaller annotations defects
- **UI state isn't persisted.** The selected layer, visibility and collapse state are lost on restart.
  - Hidden layers reappear.
  - On a slide that already has "Tumor" and "Mitoses" layers, the first point creates a new "Untitled" layer, unsorted at index 0. That creation isn't recorded in undo, so undoing the point leaves an empty layer behind (`src/state.rs:49-53`; `src/operations.rs:1193-1254`).
- **Geometry isn't validated.** Points and vertices outside the image are stored and counted. Collinear, duplicate-vertex and zero-area polygons are accepted; they can't be hit-tested, only deleted from the sidebar.
- **Overlapping polygons.** A click selects the bottom-most polygon (PLG-16).
- **Release hygiene:**
  - The Python release test expects 0.2.2, but `Cargo.toml` says 0.2.4, so it fails.
  - CI only runs `cargo build`.
  - A workflow input is interpolated straight into shell.
  - There is no `Cargo.lock`.
  - Only `density.rs` has Rust tests.
- **README drift:**
  - It says import and HUD buttons are unimplemented; both exist.
  - It says the export carries the "fingerprint"; it actually carries the full SHA-256.
  - It says a single click focuses an annotation; it's a double-click.
  - It says hidden layers disappear; they are shown dimmed.
- **UI polish:**
  - The delete dialog says "Delete Permanently", although the deletion can be undone.
  - Area is printed in mm² with 4 decimals, so a cell-sized region shows "0.0000".
  - Framing a point zooms to 1.9× native.
  - Tooltips are set but never rendered.
  - There are no accessibility properties.

### 10.2 Missing annotation features (detail in §14.2)
- **Shapes:**
  - Only point and polygon are supported.
  - Missing:
    - rectangle;
    - ellipse (the `annotation_ellipses` table exists but is unused, `src/db.rs:72-80`);
    - brush/freehand (the bitmask tables exist but are unused, `:96-119`);
    - polyline/ruler;
    - arrow;
    - text label.
  - No holes or multipolygons, so an excluded lumen inflates the area.
- **Interchange formats:** only the plugin's own JSON. No GeoJSON (QuPath), ASAP XML, Aperio ImageScope XML, NDPA or DICOM SR/ANN, for import or export.
- **Classes:** no reusable class or ontology across slides, so layers are per-slide only.
- **Naming and search:** no per-annotation names, search, filter or sort.
- **Measurements:**
  - No perimeter, ruler, coordinates in µm, per-annotation area in the tree, or measurement export.
  - The HPF field size is hard-coded to FN 22 at 40× (`src/density.rs:14-15`).
- **Layer notes:** half-implemented. The field exists in the model, DB and export, but there is no UI.
- **Styling:**
  - No per-layer opacity, outline-only mode or stroke width; the host hard-codes a 25% fill (`app/src/plugin_host.rs:61-63`).
  - The point size is fixed at 12 px.
- **Editing:**
  - No lock, copy/paste, duplicate, multi-select, move-to-layer or bulk delete.
  - No Delete-key shortcut and no keyboard navigation of the tree.
- **Review and provenance:**
  - No per-annotation author or status (draft/verified) and no audit trail.
  - Timestamps are never shown.
- **Data management:**
  - No backup, no "export all slides", no orphan browser.
  - The single DB file is the only copy of every annotation.
- **Data sources:** only "Local"; the selector is a stub (`src/sidebar.rs:436`).
- **Density:** no option to exclude hidden layers from the count. They are counted today (`src/density.rs:116-124`).

### 10.3 What is good here
- The SQL is fully parameterised.
- Deletes cascade through foreign keys.
- The inverse-action undo design is sound.
- The exact Garwood Poisson interval and the HPF formula are correct and tested.
- The import conflict UI (merge/replace/skip, apply to all) is good.
- Release artifacts use immutable URLs and SHA-256 hashes.

---

## 11. Plugin: eovae (`plugins/eovae`, in-tree workspace member; model from the `HistoVAE` submodule)

eovae is an ONNX "reconstruction-error" (VAE anomaly) explorer.

- **Sidebar.** The VAE toolbar button opens a sidebar with three sections:
  - *Model:* load an `.onnx` file and pick its input and output tensors.
  - *Analysis:* "Mip Level" (1x/2x/4x/8x), threads, batch size, and Viewport / Whole Slide / Cancel.
  - *Results:* a 12-bin histogram of per-tile mean absolute error (MAE), summary statistics and the top-12 "hot" tiles, with jump-to.
- **HUD buttons.** There are two per-pane HUD buttons: a tile grid overlay and a visualisation-mode menu. The menu's items are hard-coded in the *host*.
- **Viewport filter.** A CPU-only filter paints Reconstruction, Difference or Error Map views over the viewport. It shows stripes on tiles not yet analysed and automatically analyses up to 64 visible missing tiles.
- **Analysis pipeline:**
  1. Plan tiles in level-0 coordinates.
  2. Read them with the host's `read_region`.
  3. Run ONNX Runtime (`ort`, `load-dynamic`). CUDA is used strictly and batched if the viewer renders on the GPU; otherwise inference is CPU, one tile per run.
  4. Store the results in a global map and in SQLite (`~/.config/eov/eovae-latent-cache.db`, with sqlite-vec registered).
- **Reference model.** `HistoVae_reconstruct.onnx` has input `[32,3,256,256]` and output `[32,3,256,256]`, both in [-1, 1], with its weights in an external `.onnx.data` file. It has no latent output.
- **Packaging.** ONNX Runtime and CUDA/cuDNN are copied by `build.rs` from the developer's `HistoVAE/.venv` and `/usr/local/cuda-12.x`, then packed into the `.eop`.

### 11.1 Findings

#### VAE-1 · CRITICAL · See PLG-2
`set_filter_enabled_ffi` calls `request_render` while the host holds the `AppState` write lock (`src/lib.rs:464-466`, `src/state.rs:197-201`), which deadlocks the UI deterministically.
- **Plugin-side fix:** only record the enabled flag inside host→plugin notifications; never call back into the host from them.

#### VAE-2 · HIGH · Lock-order inversion between eovae's global mutex and the host's `AppState` lock
- **Where:**
  - `src/analysis.rs:394-407`: the worker holds `plugin_state()` while it calls `host.get_snapshot`.
  - `src/sidebar.rs:452-472`: `start_model_load` checks the render backend while holding the lock.
  - `src/filter.rs:14`, `:50`, `:306`: the UI thread holds the `AppState` write lock while the filter takes `plugin_state` up to three times.
  - `app/src/render.rs:532-535`, `:1168-1177`.
- **Confidence:** Likely. Both lock orders are confirmed in the code; the race itself was not reproduced.
- **Problem:** parking_lot locks are task-fair, so a queued writer blocks new readers. eovae's loader threads hold the read lock during OpenSlide reads, which widens the window.
- **Impact:** intermittent full-app freezes while browsing with the overlay enabled.
- **Fix:** take host snapshots *before* locking plugin state, and hold the plugin mutex only for short data copies. On the host side, release `AppState` before running filters (PLG-3).

#### VAE-3 · HIGH · The pyramid is assumed to halve at each level (downsample = 2^mip), which is wrong for Aperio SVS
- **Where:** `src/analysis.rs:234-248`, `:1315-1336` (re-verified: `downsample = 1u64 << mip_level.min(3)` while `level: mip_level` is passed to OpenSlide); `src/filter.rs:239-240`; `src/lib.rs:316`; `src/sidebar.rs:268`, `:762-769`; `ui/eovae-viewport-grid.slint:45-46`.
- **Confidence:** Confirmed. `get_level_info` (added to the ABI for exactly this) and `level_count` are never used.
- **Impact:** The repo's fixtures `C3L-00004-21.svs` and `C3L-00088-22.svs` have downsamples [1, 4, 8.002] and [1, 4, 16.004].
  - "2x" reads level 1 (4×) into a 512-px footprint, so the overlay is shifted by up to half a tile, and hot regions and jumps point at the wrong tissue.
  - "4x" is really 8× or 16×.
  - "8x" asks for a level 3 that doesn't exist. Every read fails, and the failure is reported as "Analysis cancelled" (VAE-7).
- **Fix:**
  - Call `get_level_info` and map the requested downsample (or a target MPP) to the best level, using that level's real downsample for the world step. Alternatively, read a scaled region and resample.
  - Label options with the real downsamples and disable levels that don't exist.
  - Bump the manifest requirement to `>=0.4.9`.

#### VAE-4 · HIGH · Pre- and post-processing don't match the model's contract
- **Where:** `src/model.rs:1539-1566`, `:1667-1683`. The contract is defined in `HistoVAE/pytorch/src/create_onnx.py:14-20`, `:29-32` and `data.py:196-207`, `:275-278`.
- **Confidence:** Confirmed. Impact measured on 32 CAMELYON17 tiles from `fixtures/patient_198_node_0.tif`, using the shipped ONNX model and Python onnxruntime 1.25.
- **Problem:**
  - Training composites RGBA onto white, turns near-black pixels (< 4) white, and maps to [-1, 1]. The plugin feeds `rgb/255` (range [0, 1]) and ignores alpha.
  - The output is multiplied by 255 if the tile's maximum is ≤ 1.5 and by 1 otherwise, then truncated. A tanh output needs `(x+1)·127.5`.
- **Impact:**
  - MAE is +50% too high (0.0078 vs 0.0052), with a channel bias of −0.97/−2.58/−1.37 grey levels against −0.35/−1.09/−0.55.
  - The histogram and heatmap are therefore biased.
  - A model trained with ImageNet mean/std would be far worse.
- **Fix:**
  - Add explicit preprocessing settings (input range, mean/std, alpha compositing, black-to-white), with defaults read from ONNX `metadata_props` or a sidecar file.
  - Denormalise outputs using the declared range.
  - Include a preprocessing version in cache keys.

#### VAE-5 · HIGH · Results are not keyed by slide, so overlays and statistics leak between slides and panes
- **Where:** `src/model.rs:100-105` (the namespace is model path | input | output | layout | tile size, with no slide in it; re-verified); `src/sidebar.rs:26-28`; `src/filter.rs:25-29`, `:52-58`; `src/analysis.rs:366-382`, `:516-525`. `README.md:30` claims the slide path is part of the key.
- **Confidence:** Confirmed.
- **Impact:**
  - Analyse slide A, then open slide B with the same model and level. B shows A's reconstructions and error maps at the same coordinates, never re-analyses them, and the statistics mix both slides.
  - In split view, the unfocused pane's missing tiles are read from the *focused* slide, because jobs use `get_snapshot().active_file`, not `viewport.file_id`.
- **Fix:** Include the slide identity in the namespace and tile keys, and use the pane's `viewport.file_id`.

#### VAE-6 · HIGH · Memory grows without bound: about 448 KiB is kept forever for every analysed tile
- **Where:** `src/analysis.rs:113-126`, `:516-525`, `:652-661`, `:978-991`, `:1370-1404`, `:1477`, `:1503-1504`; `src/state.rs:108`; `src/filter.rs:280-341`.
- **Confidence:** Confirmed (arithmetic plus code reading).
- **Problem:**
  - Each tile stores three images: `reconstruction_rgb` (192 KiB), `difference_rgb` (192 KiB, never rendered) and `error_map_luma` (64 KiB). Nothing is ever evicted.
  - Whole-slide runs preload every cached tile from the DB.
- **Impact:**
  - `fixtures/OS-2.bif` at "1x" is about 134k tiles, roughly 58 GB.
  - In CPU mode, auto-analysis feeds itself: each completed job triggers a render, which requests the next 64 tiles. A zoomed-out view left in Reconstruction mode therefore becomes a level-0 whole-slide analysis that ends in out-of-memory.
- **Fix:**
  - A byte-budgeted LRU, and drop `difference_rgb`.
  - Keep low-resolution error maps for zoomed-out views, and load DB tiles lazily.
  - Make auto-analysis zoom-aware.

#### VAE-7 · HIGH · Every analysis error is reported as "Analysis cancelled" and never logged
- **Where:** `src/analysis.rs:310-337`, `:600-606`, `:639`, `:1105-1111`, `:1198-1206`, `:1241-1246`.
- **Confidence:** Confirmed for the GPU path; Likely for the CPU multi-loader path.
- **Problem:** The user-cancel flag doubles as an internal abort signal. Loaders set it on `read_region` errors and GPU inference sets it on failure, so the job records `Cancelled`.
- **Impact:** CUDA being unavailable, running out of GPU memory, a model shape or dtype mismatch, or a missing level all show "Cancelled after N s", and the real reason is never shown or logged.
- **Fix:**
  - Use separate flags for a user cancel and an internal abort.
  - Rank errors so real failures beat channel-disconnect noise.
  - Always log the first real error and show it to the user.

#### VAE-8 · HIGH · The inference device is tied to the viewer's render backend, and CUDA is strict by default
- **Where:** `src/model.rs:238-246`, `:292-322`, `:847-880`, `:1850-1869`; `src/analysis.rs:396-399`, `:432`; `src/sidebar.rs:665-690`; `scripts/run-with-plugins.sh:15` (sets `EOVAE_DIAG_ALLOW_CPU_FALLBACK`).
- **Confidence:** Confirmed.
- **Problem:** `prefer_gpu` means "the viewer renders on the GPU", which forces a CUDA-only session with CPU fallback disabled. There is no device selector.
- **Impact:**
  - Users rendering on AMD, Intel or Apple GPUs are refused ("GPU Analysis Unavailable"), and auto-analysis shows "cancelled" forever.
  - The workaround, switching the *viewer* to CPU rendering, is undocumented.
  - The dev launcher sets the fallback environment variable, so developers never see the strict behaviour.
- **Fix:**
  - Add a device setting: Auto / CPU / CUDA, and later DirectML and CoreML.
  - In Auto, try CUDA, fall back to CPU, and show which device is in use.

#### VAE-9 · HIGH · The CPU path wastes 32× compute with the shipped static-batch model, and doesn't skip background
- **Where:** `src/analysis.rs:450-692`, `:502-507`, `:634-639`, `:1056-1076`; `src/model.rs:763-780`, `:1491-1493`.
- **Confidence:** Confirmed (measured with Python onnxruntime, CPU, 8 threads).
- **Problem:** Each single tile is padded to the model's fixed batch of 32. `skip_background` is applied only on the GPU path.
- **Impact:**
  - One padded batch takes 2.38 s, against 0.074 s per tile when batched.
  - A 64-tile auto job takes about 2.5 minutes, and a 50k-tile slide about 33 hours, including the glass.
- **Fix:** Batch on the CPU too, skip background on both paths, and warn when the model has a static batch size.

#### VAE-10 · HIGH · On the GPU path, skipped background tiles are never recorded, so they stay "pending" forever and auto-analysis stalls
- **Where:** `src/analysis.rs:1056-1076`; `src/filter.rs:51-63`, `:79-84`, `:286-321`.
- **Confidence:** Confirmed (logic).
- **Impact:**
  1. In the default fit view, the glass is at the top-left.
  2. The first auto job requests 64 background tiles, and all of them are skipped.
  3. The next request key is identical, so no new job starts.
  4. The whole view stays striped, and no tissue is ever analysed.
- **Fix:**
  - Store lightweight "background" entries and exclude them from both the pending display and future requests.
  - Request tiles from the centre of the view outwards.

#### VAE-11 · MEDIUM · Clicking Analyze builds a CUDA session synchronously on the UI thread, every time
- **Where:** `src/sidebar.rs:597-690`; `src/model.rs:26`, `:339-389`, `:894-908`; `app/src/plugin_host.rs:3455-3468`.
- **Confidence:** Confirmed.
- **Problem:** Each click runs `gpu_analysis_preflight`, which:
  - preloads about 2.5 GB of CUDA and cuDNN libraries on first use;
  - builds a strict session, waiting up to 20 s;
  - then discards it.
- **Impact:** The UI freezes for several seconds on the first click, and for up to 20 s on a slow or broken CUDA setup.
- **Fix:** Run the preflight on a worker thread, and cache the result per model and device.

#### VAE-12 · MEDIUM · Persistence and caching correctness
- **Disk cache silently disabled during GPU runs** (*Likely*):
  - The postprocessing thread `try_send`s whole 64-tile batches into a 32-slot queue.
  - Each SQLite insert is about 460 KB in autocommit mode, so the writer can't keep up.
  - The first `Full` error disables persistence for the rest of the run. Only about 32 tiles are saved.
  - Locations: `src/db.rs:11`, `:299-315`; `src/analysis.rs:105`, `:968-976`.
- **Stale reconstructions can be served** (*Likely*):
  - The model hash covers only the `.onnx` file, not the `.onnx.data` weights.
  - The cache key omits the selected input/output, layout, preprocessing, device and precision (TF32 on CUDA), and the plugin version.
  - Locations: `src/analysis.rs:1510-1535`; `src/db.rs:13-23`, `:391`.
- **The database is unbounded:**
  - Each row is about 448 KiB, which adds up to tens of GB per slide, model and level.
  - It lives in the config directory, which is often backed up or roaming.
  - There is no eviction, size display or clear UI.
  - Each row stores the `file_path`, which often contains PHI.
  - Migrations are ad-hoc `ALTER TABLE` statements with no `user_version`.
  - Locations: `src/db.rs:123-178`, `:338-362`, `:394-407`.
- **Jobs are not cancelled when the configuration changes**, and concurrent jobs share one slot:
  - The first job becomes uncancellable and can keep writing results for an old namespace for hours.
  - When it finishes, it reports "Completed" while the second job is still running.
  - Locations: `src/analysis.rs:270-296`; `src/sidebar.rs:451-472`, `:571-595`, `:762-789`, `:862-904`; `src/state.rs:216-238`.
- **An unexpected output size panics the worker:**
  - The panic happens out of bounds, with no `catch_unwind`, and leaves the job "Running" forever.
  - The Analyze buttons stay disabled until a restart (`src/analysis.rs:299-340`, `:1365-1390`).
- **Fix:**
  - Use backpressure and batched transactions for DB writes.
  - Hash the external weights too, and complete the cache key.
  - Move the DB to the cache directory, with limits, a clear UI and proper migrations.
  - Give each job an id, and cancel and join the old job before starting a new one.
  - Validate output shapes, and wrap job bodies in `catch_unwind`.

#### VAE-13 · MEDIUM · Performance: the UI thread is flooded and the per-frame filter cost is high
- **Progress reporting floods the UI thread:**
  - Loaders and the inference loop force a sidebar refresh plus a render 2–4 times per tile. Each one is a blocking round trip to the UI thread that re-sends about 40 JSON properties.
  - Status-pulse and run-canceller threads are spawned per tile.
  - Locations: `src/analysis.rs:565-571`, `:590-596`, `:704-767`.
- **The filter is expensive on every frame** (*Likely*):
  - It formats a `format!` id for every visible tile: about 134k for OS-2.bif in fit view at mip 0.
  - It composites per pixel in f64, single-threaded, on the UI thread, while both locks are held.
  - Statistics are rebuilt with an O(n log n) sort every 16 tiles.
  - The hover outline clones a 448 KiB tile on every poll.
  - Locations: `src/filter.rs:44-87`, `:118-187`, `:258-278`; `src/state.rs:255-297`.
- **Fixed overheads per job:**
  - A full-file SHA-256 of the slide: 2.7 GB for OS-2.bif, which takes minutes over NFS and can't be cancelled.
  - A new ORT session is built for every job.
  - An exhaustive cuDNN search took 28.4 s on the first CUDA run on an RTX 5090, with no progress shown.
  - Locations: `src/analysis.rs:464`, `:614`, `:1119`, `:1516-1523`; `src/db.rs:364-416`.
- **Fix:**
  - Throttle UI updates to 4–10 Hz.
  - Use integer tile keys and a spatial index, and parallelise compositing.
  - Use the host's fingerprint instead of a full-file hash.
  - Cache ORT sessions.
  - Show a "warming up GPU" status.

#### VAE-14 · MEDIUM · Not portable, and provisioning is neither reproducible nor lean
- **Platform breakage:**
  - `libloading::os::unix` means the crate doesn't compile on Windows. Because eovae is a workspace member and an app dev-dependency, `cargo test --workspace` also fails on Windows.
  - ONNX Runtime discovery reads `/proc/self/maps` and looks for `.so` names, so it never matches on macOS or Windows.
  - The sqlite-vec transmute uses `*mut *mut i8`, which won't compile where `c_char` is `u8` (aarch64/ARM Linux).
  - Locations: `src/model.rs:30`, `:977-984`, `:1077-1106`; `src/db.rs:470-481`.
- **`build.rs`** (`build.rs:61-188`):
  - It copies libraries from a gitignored developer virtualenv and hard-coded `/usr/local/cuda-12.x` paths.
  - It writes outside `OUT_DIR`.
  - It compares files by size only, and copies symlinks as duplicate files.
  - The venv holds both onnxruntime and onnxruntime-gpu, and cuDNN for both cu12 and cu13, so what gets copied depends on install order.
  - CI has no venv, so it builds a plugin with no runtime at all.
- **Package size and preloading:**
  - The installed `eovae.eop` is **4,146,657,280 bytes**, an uncompressed tar. It contains libcublasLt (749 MB) twice, libcufft twice, and an unused TensorRT provider.
  - All CUDA libraries are preloaded with `RTLD_GLOBAL`, even for CPU inference.
- **README errors:** it claims static vendoring, "one tile at a time", and a GPU toggle that doesn't exist.
- **Fix:**
  - Use cfg-gated loaders and platform library names.
  - Fetch pinned ONNX Runtime builds with checksums in a packaging step.
  - Stage only SONAME files, and locate CUDA lazily.
  - Compress packages. This needs host support (SH-12).

#### VAE-15 · MEDIUM · Model handling
- **Models that can't run are accepted:**
  - Float16 tensors are offered, but tensors are always built as f32.
  - `ort::inputs![tensor]` feeds inputs by position, so `selected_input` is ignored and other required inputs are never fed.
  - Changing the selected input doesn't re-infer the layout.
  - HistoVAE "encode" exports fail output selection (`mu`/`logvar`).
  - Locations: `src/model.rs:440-445`, `:474-481`, `:684-702`, `:1493`.
- **No resolution matching:**
  - The fixture SVS files are 0.494 µm/px, while HistoVAE was trained on CAMELYON17 level 0 at 0.227 µm/px.
  - So "1x" feeds tissue at half the training magnification, raising errors everywhere, with no warning to the user.
  - Locations: `src/analysis.rs:254-268`, `:1315-1343`.
- **Fix:**
  - Feed inputs by name.
  - Validate dtype, shape and required inputs at load time with a real session.
  - Store the training MPP in the model metadata, pick or resample to match it, and warn on a mismatch.

#### VAE-16 · MEDIUM · Host integration
- **Stale filter state:** the host's filter-enabled flag is out of date after a model is loaded or cleared, because the host only re-polls after eovae HUD clicks (PLG-12). `apply_overlay` doesn't check whether a model is loaded, so after "Clear" the slide is covered in stripes and CPU rendering stays forced on (`src/lib.rs:133-140`, `:432-442`; `src/filter.rs:13-42`).
- **Host-side special cases:** the visualisation-mode menu labels and action ids are compiled into the host UI, the polygon alpha is chosen by id prefix, and the filter re-sync is eovae-only (PLG-8).
- **Tests overwrite the developer's config:** they write the real `~/.config/eov/eovae.json`, and there is no path override. This happened during this audit (SH-4; `src/sidebar.rs:66-77`, `:815-860`, `:1077-1123`).

#### VAE-17 · LOW · Smaller eovae defects
- **Overlay ignores display adjustments.** The reconstruction overlay is drawn after gamma, contrast and stain normalisation, so toggling Original/Reconstruction shows colour shifts that are not model error (*Likely*).
- **Inconsistent edge-tile coverage:**
  - Whole-slide and viewport analysis skip partial tiles at the right and bottom edges, which can leave up to 2047 px unanalysed at 8x.
  - Auto-analysis includes those partial tiles, reads past the image, and feeds the transparent area as black.
  - Images smaller than one tile report "complete" with zero tiles.
- **Statistics and quantisation:**
  - Truncating the reconstruction to u8 inflates MAE by about 27%.
  - Histogram labels have 2 decimals, so typical ranges render as "0.00–0.00".
  - Percentiles are nearest-rank.
  - Statistics cover whatever tiles happened to be visited, not a defined ROI or run.
- **Configuration persistence is fragile:**
  - Read-modify-write without an atomic rename.
  - A corrupt JSON file blocks all future saves.
  - The file is written on every keystroke, on the UI thread.
  - Clearing the Threads field persists 1.
- **Errors and dialogs:**
  - Errors are logged only.
  - Status lines are elided at 18 px.
  - The GPU warning's "Dismiss" button still proceeds.
- **Jump-to-region** uses `set_active_viewport(center, 1.0)` instead of `frame_active_rect`.
- **Dead code:**
  - The grid component is never loaded.
  - The sidebar mode selector is unwired. If it were wired, `update_mode` would self-deadlock.
  - The test `clears_cache_when_namespace_changes` never calls the function it is named after.
- **Manifest range:** it is `>=0.4.1`, while the other plugins declare `>=0.4.9`.

### 11.2 Missing features for a research user
- **Similarity search and embedding exploration:**
  - sqlite-vec is registered, but there is no `vec0` table and no KNN query.
  - Embeddings are written but never read.
  - The reference exports produce no latent output.
- **Export:** per-tile CSV or GeoJSON of scores, heatmap images, "send top regions to the annotations plugin", and NPY/Parquet embeddings.
- **ROI-restricted analysis:** `on_polygon_annotation_placed` is a stub, and the host's ROI isn't available to plugins.
- **Device selection**, a preprocessing UI, and model metadata (MPP, normalisation, tile size).
- **Tissue detection** with visible controls.
- **Heatmap display:** a legend or colorbar, absolute scales (today's scale is the p05–p95 of whatever was analysed), opacity, and a threshold.
- **Reproducibility:** per-run metadata, run history, cohort batch runs, a headless mode, ETA, and pause/resume.
- **Model distribution:** there is no bundled or downloadable default model, and no instructions for obtaining one.

### 11.3 What is good here
- Generation counters drop stale model loads.
- Model loading and dialogs run on background threads.
- In-flight ORT runs are cancelled with `RunOptions::terminate`.
- Session builds are bounded by a watchdog.
- Channels are bounded.
- The database uses WAL.
- JSON parsing of UI callback arguments is tolerant.

---

## 12. Plugin: cellaris (`plugins/cellaris`, added in e54b16c; not a workspace member)

Cellaris predicts three Orion multiplex-immunofluorescence markers (Hoechst, Pan-CK and SMA) from H&E and shows them as a CPU viewport filter. It also has a polygon "readout" sidebar.

It is an edition-2021 cdylib that depends on crates.io `eov-plugin-api = "0.4"` (0.4.9 today, with no lockfile) and on `ort` 2.0-rc with `load-dynamic`.

**Overlay pipeline:**
1. The filter (on the UI thread, under the write lock) maps the viewport to 128-px level-0 cells.
2. It queues the 256-px windows that are needed.
3. One worker reads each window with `host.read_region` and runs ONNX at batch size 1, on RGB scaled to [−1, 1].
4. The worker converts the outputs to u8 planes with arcsinh(x/5) and fixed display windows.
5. The filter blends four raised-cosine-feathered windows per cell (the weights correctly sum to 1) and alpha-blends the result onto the frame.

Inference runs on **raw level-0 slide pixels**, not on the framebuffer. Predictions therefore don't change with zoom, display adjustments or HiDPI, which is a good design choice. The model `markers.onnx` and the runtime are loaded from `~/.eov/plugins/cellaris/`, or from `CELLARIS_MODEL` / `CELLARIS_ORT_DYLIB`.

### 12.1 Findings

#### CEL-1 · HIGH · The current host rejects the plugin: `plugin.toml` has no `[environment]` (PLG-4)
- **Confirmed:** parsing it with eov-plugin-api 0.4.9 gives "missing field `environment`".
- **Consequences:**
  - Discovery skips the plugin and logs a warning.
  - `eov plugin install` fails.
  - The README's "verified loading in the EOV v0.3.1 release binary" (`README.md:108`) predates the field becoming mandatory.
  - `description` and `repository` are also missing.
- **Fix:**
  - Add `[environment] version = ">=0.4.9, <0.5"`.
  - Add a CI step that parses the manifest with `PluginManifest::from_toml`.

#### CEL-2 · HIGH · Once loadable, toggling the overlay hangs eov whenever eovae is also installed (PLG-2)
- **Why:** `toggle_cellaris` is a plain toolbar action, so it is the first in-tree plugin action to reach the deadlocking filter-sync path.

#### CEL-3 · HIGH · Inference ignores the slide's pixel size (MPP)
- **Where:** `src/lib.rs:216-224`; `src/region.rs:19`, `:364-372`, `:479-483`; `src/model.rs:116-125`; `README.md:69-70`.
- **Confidence:** Likely. The absence of any MPP check is confirmed; the training resolution of about 0.325 µm/px is inferred from the README's own region table (0.980 mm² ÷ 36,258 blocks ≈ 5.2 µm per 16-px block).
- **Problem:**
  - Every window is 256 level-0 pixels regardless of the scanner: about 83 µm at 0.325 µm/px, 64 µm at 0.25 µm/px (40×), and 128 µm at 0.5 µm/px (20×).
  - The "about one cell" 16-px density block is also defined in pixels.
  - The model carries no MPP metadata, and `get_level_info` is never called.
  - Slides with unknown MPP are not flagged.
- **Impact:** On a 20× (0.5 µm/px) slide, cells appear about 35% smaller than in training. Predictions and r² are then out of distribution, but they are displayed exactly as usual.
- **Fix:**
  - Record the training MPP in the ONNX metadata.
  - Pick the nearest level and resample to that MPP.
  - Define blocks in µm.
  - Refuse to run, or warn, when MPP is unknown or more than about 10% off.

#### CEL-4 · HIGH · ONNX Runtime loading only works as documented on Windows, and a failure panics the worker
- **Where:** `src/model.rs:89-101`; `package.py:25`, `:43-44`; `README.md:90-95`, `:134-135`. The panics are in `ort-2.0.0-rc.13/src/lib.rs:151-152`, `:234`.
- **Confidence:** Confirmed.
- **Problem:**
  - The plugin looks for a file named `onnxruntime.dll` on *every* OS.
  - On Linux and macOS, ort then falls back to system paths. If no runtime is found, or the one found is older than 1.24, `ort::api()` panics.
  - Errors when locating the runtime, or when loading the one named in `CELLARIS_ORT_DYLIB`, are discarded. This contradicts the README's "reported rather than skipped".
  - The runtime that `package.py --onnxruntime` bundles is extracted into the host's cache, where the plugin never looks. The plugin can't find its own root, because no host API exposes it.
- **Fix:**
  - Use OS-specific library names, and report load and version errors.
  - Add host `plugin_root()` and `data_dir()` (§3.3) and search there.
  - Alternatively, link the runtime statically.

#### CEL-5 · HIGH · Region-readout statistics are computed on unblended 256-px windows
- **Where:** `src/region.rs:145-216`, `:359-402`; `README.md:24-27`, `:80`.
- **Confidence:** Likely.
- **Problem:**
  - The overlay uses a 128-px stride with feathering because unblended windows have seams. The README itself gives 0.91 blended vs 4.88 unblended at the joins.
  - The readout, though, uses non-overlapping 256-px windows. About 23% of its 16-px blocks (the outer ring) carry window-edge bias.
- **Impact:**
  - The seam variance inflates the spread and lowers r², pushing the reading towards "this marker is more than density". That is the very conclusion the README warns against drawing.
  - The numbers also don't match what the overlay shows.
- **Fix:** Measure on the feathered 128-px reconstruction, reusing the overlay's windows.

#### CEL-6 · MEDIUM · Failures are silent and sticky
- **Where:** `src/lib.rs:133-178`, `:225-237`, `:278-293`; `src/model.rs:70-76`; `src/region.rs:331-357`.
- **Confidence:** Confirmed.
- **Problem:**
  - All diagnostics go to `eprintln!`, which is invisible in Windows release builds and in macOS bundles. `log_message` is never used.
  - A model-load error is cached in a `OnceLock`, and the worker exits without being respawned. Installing the model afterwards therefore requires restarting eov.
  - A failed read or inference leaves its key in `requested` forever, so those cells are never painted.
  - A panic while loading leaves the readout stuck on "Measuring: 0 of 0 windows".
- **Fix:**
  - Use `host.log_message` plus a visible status line.
  - Retry loading when the overlay is toggled.
  - Un-request failed keys, with backoff.
  - Wrap loading and inference in `catch_unwind`.

#### CEL-7 · MEDIUM · Correctness of what is painted and measured
- **Split view starves one pane** (*Likely*). Both panes share one global `visible` set, so each pane's filter call replaces the other's. The pane filtered first rarely gets its windows computed, and a pane with more than 600 cells clears visibility for both (`src/lib.rs:53`, `:209-213`, `:398-425`).
- **Predicts and paints outside the slide** (*Confirmed*):
  - Cell ranges are clamped at 0 but not to the image size, so windows entirely off the slide are read, inferred (about 1.2 s each) and painted.
  - Transparent pixels arrive as RGB 0,0,0 (WSI-2).
  - In the readout, transparent no-data areas inside the image bounds have OD ≈ 2.4 and pass the 0.04 tissue threshold (`src/lib.rs:390-418`; `src/region.rs:166-214`).
- **Fix:**
  - Track visibility per pane and use the union.
  - Clamp to the image bounds, and honour alpha (composite onto white for the model, and skip those blocks in the readout).

#### CEL-8 · MEDIUM · The overlay's cost lands on the UI thread, and it forces CPU rendering everywhere
- **Where:** `src/lib.rs:311-352`, `:430-489`, `:561-562`; `app/src/render.rs:532-534`, `:1959-1999`.
- **Confidence:** Confirmed for the painting cost (benchmarked on a replica of the paint loop); Likely for the ORT contention.
- **Problem:**
  - The per-pixel f64 blend costs about 10 ms per 1080p frame and about 40 ms per 4K frame, per pane, even when every cell is already cached.
  - `supports_gpu = false` makes the host switch every pane to CPU rendering, even when only the amber "too many cells" frame is drawn.
  - ORT uses all physical cores at batch size 1 and processes windows in raster order, saturating the CPU for minutes and competing with tile loading.
- **Fix:**
  - Paint into a cached overlay layer (§2.6, item 5).
  - Cap the ORT threads and batch the windows.
  - Process windows outward from the centre of the view.

#### CEL-9 · MEDIUM · Clinical-safety and labelling gaps
- **Where:** `src/lib.rs:95-115`, `:358-492`; `src/overlay.rs:5-33`; `ui/readout.slint:113-118`.
- **Confidence:** Confirmed (absence).
- **Problem:**
  - **No safety framing:** nothing says "research use only" or "not for diagnosis".
  - **No context on the overlay:** no legend, no "predicted" label, no model name or version, and no indication of which areas are still being computed.
  - **No validity checks:** the display windows were fixed from one cohort, nothing detects out-of-domain input (IHC, frozen sections, other stains or scanners), and there is no uncertainty or calibration display.
  - **Visual problems:**
    - Glass is painted too, appearing dark at 67% opacity.
    - The green Pan-CK and amber SMA overlays are a red-green colour-blind confusion pair.
    - Opacity is fixed, and markers can't be toggled individually.
- **Impact:** Virtual staining is very easy to mistake for a measurement.
- **Fix:**
  - Show a persistent "Predicted — research use only" badge and legend on the viewport.
  - Add a model card in the sidebar.
  - Add domain checks and a tissue mask.
  - Use a colour-blind-safe palette, with opacity and per-marker toggles.

#### CEL-10 · MEDIUM · Build and release
- **Workspace membership:** the plugin is in neither `members` nor `exclude` of the eov workspace, so it can't be built inside the eov checkout, and `package.py` fails (SH-6).
- **Untested UI:** the host's tests that compile plugin UIs don't include `readout.slint`.
- **Unpinned dependencies:** `Cargo.lock` is gitignored, so both `eov-plugin-api = "0.4"` and the pre-release `ort` float.
- **CI coverage:** CI runs only on Ubuntu and Windows. There is no macOS job, no packaging job and no release workflow.

#### CEL-11 · LOW · Smaller cellaris defects
- **Model output isn't validated:**
  - The output shape is never checked, so a smaller output panics the slice at `src/lib.rs:241`.
  - Channel names are split on commas without trimming.
  - Any `raw_intensity` value other than exactly `"false"` counts as raw.
  - Only the first input is fed.
- **The region outline behaves like an editable annotation** that the plugin ignores:
  - Dragging it snaps back.
  - Nothing is re-measured.
  - The sidebar doesn't name the slide, and its numbers stay after switching files.
  - "Close" spawns an OS thread just to hide the sidebar (`src/lib.rs:597-617`, `:660-667`).
- **Documentation:**
  - Install instructions cover Windows only.
  - "A path that does not exist is reported" is false for the runtime path.
  - The bundled runtime is never used.
  - The manifest declares only one of the two toolbar buttons.
- **Licensing and provenance:**
  - The model isn't shipped or linked. There is no checksum, version or model card, and the terms of the Orion training data aren't stated.
  - Redistributing ONNX Runtime requires its licence and third-party notices, which the package doesn't include.
  - The licence is declared as `GPL-3.0-or-later`, while eov declares `GPL-3.0`.
- **Caches:**
  - Up to about 250 MB of window and cell caches survive after a slide is closed.
  - Nothing is persisted, so the roughly 10 minutes of CPU work for one 1:1 view is redone every session.

### 12.2 Missing features
- MPP-matched inference, stain normalisation of the model's input, and a tissue mask.
- GPU or batched inference, a persistent prediction cache, and whole-slide or ROI batch runs with export of the predicted channels (e.g. OME-TIFF).
- **Display controls:**
  - a legend and a research-use-only badge;
  - opacity, per-marker toggles and display windows;
  - marker choice. `DEFAULT_MARKERS` is hard-coded although the model has 19 channels.
- **Model management:** picking a model file, showing its metadata, version and hash, and download-and-verify.
- **Readout:**
  - copy or export the numbers;
  - use annotations-plugin polygons as regions;
  - compare several regions;
  - a cancel button.
- **Platform support:** Linux and macOS runtime support, plus macOS CI.
- **A way to view real Orion/CyCIF OME-TIFF data.** Users can't check predictions against measured multiplex IF in eov, because the viewer can't open those files (§14.3).

### 12.3 What is good here
- Inference runs on raw level-0 pixels, not on the rendered framebuffer.
- The feathering maths is correct.
- Markers are resolved by name.
- The filter callback never calls back into the host.
- Panics stay in worker threads and never cross `extern "C"`.
- Lock order is consistent.

---

## 13. Plugin: gamepad (`plugins/gamepad`, v0.2.5, git submodule)

**Input handling:**
- `set_host_api` starts a single `gilrs` worker thread that lives for the whole process and polls every 12 ms (`src/state.rs:307-314`, `:659-799`).
- **Buttons** cycle pane, tab or series, open a series entry, fit, or nudge the zoom, all through host API calls.
- **Sticks and triggers** do a `get_snapshot` → `set_active_viewport` read-modify-write on every tick.
- **The right stick, by default,** injects OS-level pointer motion: enigo (XTest, SendInput, CGEvent or libei), or the XDG RemoteDesktop portal via `ashpd` on Wayland (`:958-1108`).

**Settings window and state:**
- The settings window is a separate top-level `.slint` window. The host polls it every 200 ms.
- State lives in one global `Mutex<PluginState>`, persisted as JSON under `dirs::config_dir()/eov/plugins/gamepad/`.

**Leftover code:** code from a removed out-of-process `eov plugin-window` mode is still active (§13.1, GPD-6).

### 13.1 Findings

#### GPD-1 · HIGH · The worker holds the plugin mutex while blocking on the UI thread, which deadlocks the whole app
- **Where:** `src/state.rs:791-795` (re-verified: `let state = plugin_state().lock().unwrap(); persist_runtime_state(&state); refresh_window_ui_if_available();`), `:565-569`, `:861-863`.
- **How it deadlocks:**
  - On the worker thread, `refresh_window_ui_if_available()` → host `refresh_sidebar` → `run_on_ui_thread` posts to the event loop and blocks in `rx.recv()`.
  - On the UI thread, the 200 ms window refresh (`app/src/plugins/mod.rs:305-311`) and every window callback lock the same mutex (`src/window.rs:15`, `:83`; `src/state.rs:316-317`, `:1365-1370`).
  - Each thread then waits for the other forever.
- **Confidence:** Confirmed lock-order inversion; whether it hits is timing-dependent.
- **Triggers:**
  - a change to the device list;
  - a controller connecting or disconnecting;
  - pressing a button mapped to "Enable / Pause".
- **Impact:** The README's own setup flow (open the Gamepad window, then connect the controller, `README.md:191-198`) can freeze eov. So can a Bluetooth pad sleeping and reconnecting while the window is open. Recovery means killing the process.
- **Fix:** Drop the guard before persisting or calling the host (persist from a clone). On the host side, make UI requests from worker threads non-blocking (PLG-3).

#### GPD-2 · HIGH · Input is not limited to when eov has focus, and the default mapping injects system-wide pointer motion
- **Where:** `src/state.rs:737-764`, `:875-927`, `:1062-1108`, `:1252-1253` (the default right stick is `MouseHorizontal`/`MouseVertical`; re-verified); `plugin_api/src/ffi.rs:205-219` (the snapshot has no focus field).
- **Confidence:** Confirmed.
  - Linux evdev and macOS IOHIDManager deliver controller events regardless of focus.
  - On Windows, gilrs's WGI backend delivers them only while eov is focused.
- **Impact:**
  - **Background input on Linux and macOS.** While the user plays a game or uses another app with the controller:
    - the foreground app's cursor moves;
    - eov silently cycles tabs, zooms and opens series slides in the background.
  - **macOS permissions.** enigo's default settings trigger the Accessibility permission prompt for the *whole eov process*. Once it is granted, every in-process plugin can synthesise system-wide input.
- **Fix:**
  - Add window focus to the host snapshot, or have the host drop plugin navigation while unfocused.
  - Replace OS-level injection with a host-provided virtual cursor or viewport-pointer API.
  - If OS injection stays, make it opt-in and explain it in the UI.

#### GPD-3 · HIGH · In the Flatpak the plugin can't see any controllers
- **Where and why:** The Flatpak grants only `--device=dri` (UPD-5). gilrs needs `/dev/input/event*` and udev.
- **Confidence:** Likely.
- **Fix:**
  - Grant `--device=input`.
  - Detect this case and show a specific message.

#### GPD-4 · MEDIUM · Input handling defects
- **Runaway after a disconnect:**
  - Stale axis values are kept after a disconnect or device switch. gilrs emits no zeroing events, and `axis_values` is never cleared.
  - A pad that sleeps with a stick deflected keeps panning or zooming to the clamp.
  - This also calls `set_active_viewport` and re-renders every 12 ms indefinitely (`src/state.rs:671`, `:743-787`, `:950-955`).
- **"Pause" doesn't pause buttons.** Only the continuous mappings check `controller_enabled` (`:748-752`, `:801-884`).
- **Mouse-permission flow:**
  - The portal handshake runs `block_on` on the input thread, so no controller input is processed while its dialog is open.
  - It asks to persist the grant, but passes no restore token and discards the one it gets, so the prompt reappears every session.
  - No parent window id is passed.
  - Any error permanently latches the backend as "unavailable" until restart, including macOS's `NoPermission` right after the prompt is shown.
  - On wlroots compositors the X11 fallback only moves the XWayland pointer (`:976-1105`).
- **Navigation is an absolute read-modify-write from a worker:**
  - `set_center_zoom` calls `stop()`, cancelling fit and zoom animations and fighting concurrent mouse drags.
  - If focus changes between the read and the write, pane A's centre is applied to pane B.
  - Speed is per loop iteration rather than per unit time (`:936-955`, `:1119-1133`; `app/src/plugin_host.rs:1228-1247`).
- **Analog processing:**
  - The dead zone is a step with no rescale, and it is per-axis (square), so diagonals snap to the axes.
  - Triggers that report a full-range axis map non-monotonically.
  - The response curves are fixed (`:1110-1117`, `:1201-1243`).
- **Fix:**
  - Clear axis state on disconnect and device change, or read live state each tick.
  - Gate every action on the enabled flag.
  - Create the injection backend off the input thread, persist the restore token, and retry with backoff.
  - Add host relative-navigation calls (`pan_by`, `zoom_by`) and scale motion by elapsed time.
  - Use a radial dead zone with rescale.

#### GPD-5 · MEDIUM · Constant 83 Hz polling and repeated disk I/O, some of it on the UI thread
- **Where:** `src/state.rs:616-657`, `:677-685`, `:766`, `:880-886`; `src/window.rs:15`, `:83`.
- **Confidence:** Confirmed.
- **Problem:**
  - Every 12 ms, from app start and even with no controller, the worker enumerates devices twice and clones the whole state.
  - Every 250 ms it re-reads and re-parses `state.json`.
  - With the window open, each 200 ms refresh and each callback re-reads that file on the UI thread.
  - Each slider move writes the state and profile files and lists the profiles directory.
- **Impact:** Laptops never reach deep idle, and slider drags stutter on roaming or NFS home directories.
- **Fix:**
  - Use event-driven waits, with long sleeps when idle.
  - Remove the disk re-sync.
  - Debounce writes off the UI thread, with an atomic rename.

#### GPD-6 · MEDIUM · Leftovers from the out-of-process window design
- **Where:** `src/state.rs:316-322`, `:399-415`, `:1332`, `:1365-1440`; `Cargo.toml:20-27`; `app/src/cli.rs:239-248`; `app/src/plugins/mod.rs:381-425`.
- **Confidence:** Confirmed.
- **Problem:**
  - **Disk state sync.** The plugin still syncs state through disk files, which loses updates in-process: the worker reads stale content, the UI writes, then the worker applies the stale copy.
  - **Close-request watcher.** It calls the plugin's own statically linked Slint, whose event-loop proxy is never set, so it never works.
  - **The `slint` dependency.** It exists only for this dead path, and pulls winit 0.30, wgpu 29 (unstable features) and femtovg into the cdylib.
  - **Slint version skew.** At runtime the plugin's Slint copy is inert and no Slint types cross the FFI, so host/plugin version skew is harmless today.
  - **"Toggle Panel" mapping.** It does nothing when hosted: `toggle_window()` returns `true` immediately when a host API exists (`:399-403`, `:844-846`).
- **Fix:**
  - Delete the disk sync, the heartbeat and close-request code, and the `slint` dependency.
  - Add a host API to open and close a plugin's window.

#### GPD-7 · MEDIUM · Host-side window handling
- The host leaks a window instance and a library handle on every open/close cycle, and recompiles about 900 lines of Slint each time (PLG-18).
- The forced width, `clamp(2·0.76·parent, 1040, 1760)`, runs off-screen on 1366-px laptops (PLG-8), although the `.slint` file already has a narrow layout below 780 px.
- `refresh_sidebar` refreshes whichever sidebar is open, so every gamepad slider event re-pulls the annotations sidebar (PLG-10).

#### GPD-8 · LOW · Smaller gamepad defects
- **Profile names:**
  - The field trims on every keystroke, so typing "Xbox Elite" gives "XboxElite".
  - Names are used as raw path components: "a/b" silently fails to save, and an absolute name writes or deletes `<name>.json` outside the profiles directory.
  - Delete asks for no confirmation.
  - Typing an existing profile's name and then moving a slider overwrites that profile (`src/state.rs:437`, `:546-548`, `:592-596`, `:629-631`).
- **Profile format:**
  - There is no version and no `#[serde(default)]`, so adding any field makes every existing profile and `state.json` fail to parse.
  - The enabled flag and preferred device are never persisted.
  - The profile list includes any file stem in the directory (`:231-239`, `:486-496`, `:598-607`, `:1314-1344`, `:1466-1485`).
- **Slider ranges vs. clamps:**
  - The dead-zone slider goes to 1.0, but the value is clamped at 0.95.
  - Sensitivity sliders span 0.1–2.0, while typed values may be anything from 0.05 to 4.0.
  - The knob can draw outside its track.
- **Device identification:** two identical controllers can't be told apart, because the device key ignores `GamepadId`.
- **Status text:**
  - It shows Debug-formatted power info, e.g. "Discharging(80)".
  - It is never refreshed, so it still says "Ready" while paused.
- **Tests and CI:**
  - The Python release test fails: it hard-codes 0.2.3, but the plugin is 0.2.5.
  - There are no Rust tests, and no workflow runs tests or clippy.
  - There is no `Cargo.lock`.
  - Release builds compile against eov master.
  - A workflow input is interpolated into bash.
  - Actions are pinned by tag.
  - `.gitignore` lacks `target/`.
- **Multiple instances:** two eov instances share `state.json` and both act on the same pad, so both pan the slide and both inject pointer motion.

### 13.2 Missing features
- **Mouse mode is half-implemented.** The right stick moves the cursor, but no button can click, drag or scroll.
- **Held-button repeat.** `pressed_buttons` is tracked but unused.
- **Items the README lists as not implemented:** profile export, chords, automatic per-controller profiles and haptics.
- **Workbench actions:**
  - jump to an objective-equivalent magnification (5×/10×/20×/40×) or to 1:1;
  - field-by-field "snake" screening with overlap, as used for cytology and mitosis screening;
  - a centre reticle with "drop point" and "start/finish polygon" buttons;
  - toggling overlays and filters;
  - next or previous slide in a case;
  - bookmarks and screenshots;
  - undo and redo.

---

## 14. Features pathologists and other WSI-workbench users would expect

eov presents itself as a small "Linux-style utility" for WSI, with a proper workbench experience delivered by plugins. This section is written against that positioning. Each item is marked as belonging in the **host** or a **plugin**, and whether the **plugin API needs to grow** for it.

Comparison points:
- **Desktop workbenches:** QuPath, ASAP, Aperio ImageScope, NDP.view2, CaseViewer/SlideViewer, Sedeen.
- **Viewers in clinical image-management systems:** Sectra, Philips IntelliSite and Leica/Aperio GT.

Status was checked against the code (§1.3 describes the method).

### 14.1 Viewing and navigation (host)

| Feature | Status in eov | Why it matters | Where it belongs |
|---|---|---|---|
| **Slide label and macro/overview image** | Missing. OpenSlide exposes the `label`, `macro` and `thumbnail` associated images, and openslide-rs 0.2.1 already has `read_associated_image`, but nothing uses them. The HUD shows 8 fixed fields. | Reading the label is how a pathologist confirms case/accession/block identity before diagnosing. The macro image shows tissue fragments outside the scan area, pen marks and the coverslip. | Host: a "Slide" panel with label and overview. Include a **hide label** privacy toggle (the label is PHI) for teaching and screenshots. Expose the images through the plugin API. |
| **Rotation (90° steps and free) and flip** | Missing everywhere: viewport, minimap, export and the plugin API. | Tissue is oriented to anatomy (epidermis or mucosa up). Comparing levels or serial sections often needs rotation. | Host viewport transform. The plugin API must expose it, because overlay and plugin coordinates must stay in level-0 space. |
| **Objective-equivalent magnification presets** | Partial. The HUD shows optical zoom when objective power is known, and you can type a zoom into a field. There are no 1.25×/2.5×/5×/10×/20×/40×/1:1 buttons or keys. Fractional objectives (2.5×) are lost (WSI-15). The 1/2 keys are taken by the annotations plugin. | Pathologists work in magnification steps ("scan at 4×, confirm at 40×"). | Host. Derive the magnification from MPP when objective power is missing (≈ 10 µm/px ÷ MPP). |
| **Keyboard-driven navigation** | Minimal. `f`, `m`, `+`/`-` (these also fire inside text fields, UI-8), Ctrl+W and undo/redo exist. Missing: arrow/WASD panning, PageUp/PageDown, Ctrl+O, Ctrl+Tab, number keys for magnification, F11, next/previous slide. | Pathologists screen whole slides with the keyboard for hours, and mouse-only navigation causes strain. | Host: one remappable shortcut table that is ⌘/Ctrl-aware and validated against plugin hotkeys. |
| **Linked navigation across *different* slides** (H&E vs IHC, levels, re-cuts) | Missing. The viewport lock only links panes showing the *same* file. | Side-by-side H&E/IHC is routine, for example HER2, Ki-67, or a lymphoma panel with 6–10 stains. | Host. Use per-pane offset, rotation and scale with manual landmark alignment, and later automatic registration. Synchronise zoom in **physical** units (MPP), because slides differ in MPP. |
| **Fit to tissue / scanned region** | Missing: `openslide.bounds-*` and `openslide.region[]` are ignored (WSI-14). | MIRAX and some other formats show a small island of tissue in a mostly empty canvas. | Host. |
| **Measurement suite** | Partial. There is one 2-point line per file, wrong for anisotropic or missing MPP (UI-5). No area/perimeter, polyline, angle, circle, multiple persistent measurements, list or export. | Tumour size, margins, depth of invasion (melanoma Breslow depth) and lesion area are reportable values. | Host measurement layer with a list panel, copyable values, unit settings and persistence. |
| **Manual calibration when MPP is missing** | Missing. | Scans exported to plain TIFF or JPEG often lose MPP, which makes every measurement and the scale bar disappear. | Host: per-slide µm/px or objective entry, persisted (keyed by slide identity). |
| **Session restore, bookmarks and navigation history** | Missing. Only the sidebar is persisted. | "Reopen my 12 slides where I left them", saved regions of interest for teaching or tumour boards, and back/forward through visited fields. | Host: a per-slide last view, named bookmarks with thumbnails, and Alt+←/→ history. Bookmarks should be shareable (export/import) and deep-linkable (§14.4). |
| **Full-screen / presentation mode** | Missing. | Tumour boards, teaching and multi-head microscope replacement. | Host. |
| **Status bar and error reporting** | Missing. `status-text` is never rendered (UI-4). | Without it, users can't tell "slide unsupported" from "app hung". | Host. |
| **Colour management (ICC)** | Missing. The repo's own fixtures carry ICC profiles (`openslide.icc-size`: about 142 KB for Aperio and 1.8 MB for BIF). openslide-rs 0.2.1 has no ICC API. | Scanner-to-display colour fidelity is a stated concern of regulators and standards bodies for WSI, and the same slide looks different on different scanners without it. | Host. Upgrade to openslide-rs 2.x (`read_icc_profile`), convert source → display with a 3D LUT on GPU and CPU (`moxcms` is already in the dependency tree via `image`), and show a "colour-managed" indicator. |
| **Indication that colours are altered** | Missing (CLR-9). | Stain normalisation and deconvolution are synthetic colour. | Host badge, also burned into exports. |
| **Full metadata/property browser** | Partial: 8 fields in the HUD; `probe` prints a subset. | Scanner model, compression, focus and scan settings, ICC, bounds, DICOM attributes and the stain label matter for QC and research provenance. | Host: a searchable, copyable table of every property. |
| **HiDPI-correct rendering, trackpad pinch/pan, pixel-level ("nearest") view above 1:1** | Missing (RND-3, UI-15). | Retina laptops and trackpads are the norm. Nearest-neighbour view helps judge scanner artefacts. | Host. |
| **Copy/snapshot with scale bar and overlays** | Partial. Copy Image has no overlays or scale bar. Export has measurement and ROI overlays only, with no scale bar and no plugin overlays. | Figures, reports and tumour-board slides need a calibrated scale bar. | Host: the export and clipboard paths should draw the scale bar and the retained overlay layers. |
| **Case and slide organisation** | Partial. The series bar browses one folder. It has no sort, filter or search and no grouping by case or accession; it shows a separate entry for every DICOM instance; label thumbnails aren't shown; there are no next/previous slide shortcuts. | A case is typically 5–30 slides (blocks × stains). Reviewing them in order is the core clinical loop. | Host. Group by filename pattern, barcode or DICOM Study/Series UID. |
| **Runs without Vulkan/Metal** (VDI/Citrix/RDP, VMs) | Missing (RND-4). | Many hospital desktops are virtualised. | Host. |
| **Open from the file manager, single instance, deep links** | Missing (SH-8). | "Open this slide at this coordinate" from a LIS/LIMS or a spreadsheet of findings. | Host: MIME/UTI declarations, IPC forwarding, `eov slide.svs --goto x,y --zoom 20x` and an `eov://` URL scheme. |

### 14.2 Annotation, counting and quantification (annotations plugin + host API)

- **Shape tools.**
  - Exist: point and polygon only.
  - Missing: rectangle, ellipse, brush/freehand with eraser, polyline/ruler, arrow, text label, and holes/multipolygons. The DB already has `annotation_ellipses` and bitmask tables that are never used.
  - Needs: the plugin API's tool registration and generic overlay primitives (PLG-9).
- **Classes / ontology.** Reusable classes across slides (Tumour, Stroma, Necrosis, Mitosis, Lymphocyte, …) with colours, hotkeys, per-class visibility and opacity, and counts. Layers are per-slide today, and the HPF size is hard-coded.
- **Cell counting tool.**
  - Click-to-count with a running tally per class, plus derived indices: Ki-67 % positive, mitoses per mm² or per N HPF, and PD-L1 TPS/CPS helpers.
  - Undo per click; keyboard class switching.
  - "Count only within region." The density panel already computes area, count/mm² and an exact Poisson interval for a selected polygon, which is a good foundation, but it is stale after edits (ANN-2) and inconsistent for self-intersecting shapes (ANN-3).
- **HPF / field-area reticle.**
  - A circle or square of a configurable area (field number 20/22/25/26.5 at a chosen objective, or N mm²) that follows the cursor or can be stamped.
  - Mitotic thresholds are increasingly stated per mm²; for example, GIST risk uses mitoses per 5 mm².
  - Today the field area is hard-coded to field number 22 at 40× (`plugins/annotations/src/density.rs:14-15`).
- **Measurements on annotations.** Area, perimeter, longest diameter (Feret) and minimum caliper in physical units, shown in the tree and exported to CSV.
- **Interchange formats, import and export.**
  - GeoJSON (QuPath-compatible) first.
  - Then ASAP XML, Aperio ImageScope XML, Hamamatsu NDPA, and DICOM SR / DICOM Microscopy Bulk Simple Annotations.
  - Document the coordinate conventions: level-0 pixels, the bounds offset, units and Y-down.
  - Today only the plugin's own JSON is supported, and a QuPath GeoJSON fails silently (ANN-15).
- **Review and provenance.**
  - Per-annotation author, timestamps, status (draft / verified / rejected), comments, locking and an audit trail.
  - Blinded multi-reader copies for research studies.
- **Annotation-driven dataset extraction.** Patches inside or labelled by annotations, plus binary or multiclass masks at a chosen MPP. The README already acknowledges this is missing.
- **External analysis results as overlays.**
  - Load cell detections or segmentations (HoVer-Net, StarDist, CellViT GeoJSON, often 10⁵–10⁶ objects) and aligned heatmaps (TIFF/PNG plus MPP).
  - Needs: class colours, opacity, legend and level of detail.
  - Blocked today by the copy-everything overlay API (PLG-5).
- **Backup and portability.** "Export all slides", backup/restore of the database, relinking orphaned annotations (ANN-8), and per-slide undo (ANN-6).

### 14.3 Formats and imaging modalities

| Capability | Status | Notes |
|---|---|---|
| **Multichannel fluorescence and multiplex IF** (OME-TIFF, QPTIFF/Akoya PhenoCycler, CZI fluorescence, Orion and CyCIF OME-TIFF) | Missing. OpenSlide reads only RGB brightfield. | Needs a channel list with per-channel colour/LUT, min/max/gamma, visibility and presets. Notably, cellaris *predicts* Orion multiplex IF channels, yet eov cannot open real Orion OME-TIFF data to compare against. Implementation needs a second reader (an OME-TIFF SubIFD pyramid reader on the `tiff` crate, or bfconvert/raw2ometiff conversion) behind a `SlideSource` trait. |
| **Philips iSyntax** | Missing (not in OpenSlide). | Philips IntelliSite is widely deployed clinically. libisyntax is a possible route. |
| **Olympus/Evident VSI, Leica LIF**, others | Missing. | Common in research. Bio-Formats-class readers. |
| **DICOM WSI** | Supported via OpenSlide 4. However, the series browser lists every instance, dataset extraction treats each `.dcm` as a separate slide, and slide identity uses one instance. | Group by Series Instance UID, and key identity on DICOM UIDs. |
| **Z-stacks / focal planes** | Missing. | Cytology and some NDPI/VMS/DICOM scans are multi-focal. Needs a focus slider. |
| **Remote slides: DICOMweb (WADO-RS), HTTP range reads, OMERO** | Missing. This is deliberate, given the offline stance. | Suggest a **"slide source" plugin type**, which the current API cannot express because the host is hard-wired to OpenSlide. That keeps the host offline while enabling PACS/VNA access. |
| **Writing slides**: export a region or the whole slide as pyramidal (Big)TIFF/OME-TIFF; de-identified copies | Missing. Export is PNG/JPEG of the viewport only (EXP-3). | Needed for sharing, teaching sets and conversions. De-identification must strip the label, macro and PHI properties. |

### 14.4 Platform, plugins and administration

- **Plugins panel** (GUI):
  - List loaded and failed plugins with version, origin (official / GitHub / local), signature status, errors and resource use.
  - Enable/disable, update and remove from the GUI.
  - The CLI is invisible on Windows (SH-2), and there is no disabled state in the registry.
- **Safe mode:** `--safe-mode`, plus automatic disabling of a plugin that crashed at startup (SH-13).
- **Updates from the GUI**, with release notes shown in the plan, `--dry-run` and `--json` output for IT, offline/air-gapped updates (`eov update --from <dir>` with verified manifests), and a rollback command (UPD-1, UPD-11).
- **Enterprise networks:** proxy support and the OS certificate store (UPD-13).
- **Signed binaries:** Authenticode and notarization. Also signed plugin catalogs and plugins (UPD-11).
- **Managed deployment:**
  - a system-wide plugin directory;
  - a policy file to disable self-update or third-party plugins;
  - a lockfile of plugins (IDs, versions and SHAs) for reproducible workstations.
- **Single instance and multi-instance safety:** locking for the registry, recent files and plugin databases (SH-8, SH-14, ANN-7).
- **Accessibility:**
  - accessible names for the toolbar and HUD (most tooltips are set but never rendered);
  - keyboard-only operation;
  - colour-blind-safe overlay palettes (CEL-9);
  - a light/high-contrast theme. Only a dark theme exists, and the export dialog alone has 105 hard-coded colours.
- **Localisation:** there is no i18n. All UI strings are hard-coded English; Slint's `@tr()` is unused.
- **Diagnostics:** a log file, a crash report dialog and "open log folder" (SH-16).

### 14.5 Analysis and research workflows

- **IHC quantification:**
  - H-DAB and other stain presets, plus custom stain vectors measured from an ROI (CLR-8);
  - DAB thresholding and positive-pixel count;
  - helpers for nucleus detection and H-score.
- **Tissue detection and QC:** Otsu or saturation tissue masks at low resolution, used for fit-to-tissue, dataset extraction and stain fitting; blur, fold and pen-mark detection overlays.
- **Stain normalisation as a workflow:** a reference slide or ROI, lock, and save/load of stain vectors, fitted once per slide (CLR-1).
- **Dataset extraction:**
  - choose a target MPP or magnification (only level 0 today);
  - tissue-mask filtering with a minimum tissue fraction;
  - annotation labels and masks;
  - optional stain normalisation;
  - output as RGB PNG, JPEG, HDF5 or zarr;
  - coordinates compatible with CLAM, TIAToolbox and slideflow;
  - resume, and per-slide caps or random sampling;
  - richer metadata: tissue fraction, vendor, objective and slide identity.
- **Headless and batch mode:** run plugin analyses (eovae, cellaris) over a cohort from the CLI, write results as GeoJSON or CSV, and produce reproducible run manifests (model hash including external weights, preprocessing, level/MPP, device, plugin version).
- **Scripting:** Python bindings or a local JSON-RPC interface, so researchers can drive the viewer and read slide regions without writing a native plugin. Today plugins must be Rust cdylibs, a high bar for researchers.
- **Model management** for AI plugins: model cards, MPP and normalisation metadata, versioning, download-and-verify, and a default or reference model.

### 14.6 Safety, privacy and intended use

- **State the intended use** ("research and education; not for primary diagnosis") in the README, the About box and AI-plugin UIs. eov is not a cleared medical device (FDA, IVDR). Cellaris's predicted channels in particular need a persistent "Predicted — research use only" badge (CEL-9).
- **Make synthetic colour obvious** (CLR-9), and include processing provenance (normalisation method, deconvolution settings, plugin filters) in exported figures and metadata.
- **De-identification:**
  - hide the label in the UI and in screenshots;
  - strip PHI-bearing properties and associated images when exporting or converting;
  - stop logging full paths at info level (SH-16);
  - stop exporting absolute paths in annotation JSON (ANN-17) and in the eovae DB (VAE-12);
  - gate plugin access to recent-file paths (SH-14).
- **Audit trail** of plugin loads, updates and annotation changes, for research reproducibility and later clinical use.

---

## 15. Tests, CI and build health

### 15.1 What was run for this audit (Rust 1.98.1, Linux x86_64, OpenSlide 4.0.0 in `/usr/local`)

- **`cargo clippy --workspace --all-targets`:** clean, with **no warnings**.
- **`cargo test --workspace --no-fail-fast`:** failed to link (`rust-lld: unable to find library -lopenslide`, see SH-5).
- **The same run with `LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu`:** passed. Results per crate:

| Crate | Passed | Ignored | Approx. source lines |
|---|---|---|---|
| `eov` (app) | 90 | 2 (large local fixtures) | 39k |
| `eov-update-helper` | 4 | 0 | 0.5k |
| `eov-common` | 48 | 0 | 7.6k |
| `eov-plugin-api` | 13 | 1 doctest | 1.6k |
| `eovae` | 31 | 2 (need ONNX Runtime with GPU) | 8.1k |
| `annotations` | 3 | 0 | 8.8k |
| `gamepad` | **0** | 0 | 4.4k |
| `cellaris` | not built (not a workspace member; SH-6) | | 2.3k |
| **Total** | **189** | **5** | |

> ⚠️ Running the tests modified the auditor's real `~/.eov/config.toml` and `~/.config/eov/eovae.json`, and added fixture extractions to `~/.cache/eov/plugin-packages`. See SH-4.

### 15.2 Gaps, ordered by the risk they leave uncovered

1. **Concurrency and lock discipline.** There are no tests and no debug assertions for re-entrant locking. Five deadlocks were found by reading code (§2.1). Two cheap guards would catch most of them:
   - a `debug_assert!(state.try_write().is_some())` at the top of `request_render_loop`;
   - a test-mode wrapper that panics when a plugin callback is invoked while `AppState` is locked.
2. **Multi-slide dataset extraction.** No integration test; one would have caught DS-1 (CRITICAL).
3. **Plugin ABI.** No test loads a plugin built against a different `eov-plugin-api` version, checks layouts or round-trips the vtable. Manifests are not parsed in plugin CI, which would have caught the cellaris load failure (PLG-4). The README's manifest example is not doc-tested: the crate's README is not wired in with `#![doc = include_str!]`.
4. **Fixture tests pass silently in CI.** `fixtures/` is gitignored, and the slide-based tests in `common/` return early when files are missing (`common/src/wsi.rs:446-490`; `common/src/tile.rs:529-564`), so they always "pass" there. Fixes:
   - publish a small CC-BY fixture set, for example from the OpenSlide test data;
   - mark these tests `#[ignore]` with a reason instead of returning early.
5. **Rendering correctness.** No golden-image tests compare CPU, GPU and export output, or check overlay alignment. They would catch WSI-7, CLR-5 and RND-2.
6. **Colour science.** The stain tests synthesise data from the *target* matrix, and cover no background, collinear or IHC cases (CLR-2, CLR-6, CLR-8).
7. **`blitter.rs`, `overlay.rs`, `postprocess.rs`, `common/src/render.rs` and the minimap math have no tests.**
8. **Updater.** There is no injectable HTTP transport, so installs, planning, consent versus `-y`, SHA mismatches and rollback are untested. Homebrew parsing is tested against invented output (UPD-3). The Windows helper and the macOS bundle code are first compiled at release time (UPD-14).
9. **Plugins:**
   - **annotations:** only `density.rs` is tested. The DB, history and import/export round-trips are not.
   - **gamepad:** has no Rust tests.
   - **eovae:** the tests cover helpers only, overwrite the user's config, and one test (`clears_cache_when_namespace_changes`) never calls the function it is named after.
   - **cellaris:** has no tests in eov CI.
   - **Plugin repos:** the Python release tests in the annotations and gamepad repos currently fail (stale hard-coded versions).
10. **Test isolation.** Tests write to the real `~/.eov`, `~/.config/eov` and `~/.cache/eov` (SH-4).

### 15.3 CI and release pipeline (detail in UPD-14)
- **Platform coverage:** CI runs on Linux only, and doesn't run the Python release tests.
- **Release gating:** releases don't require green CI, and `/noci` commits skip it.
- **Plugin builds:** they run against eov master HEAD, without lockfiles.
- **Pinning:** actions are pinned to mutable tags, and some tools are fetched as "continuous" builds without checksums.
- **Secrets:** `CARGO_REGISTRY_TOKEN` is exposed to build scripts.
- **Suggested additions:**
  - Windows and macOS matrix jobs, building every plugin against the PR's API;
  - `cargo machete` and `cargo deny`;
  - a glibc-symbol-version check on the AppImage;
  - `otool -L` checks on macOS binaries;
  - manifest parsing for every `plugins/*/plugin.toml`;
  - a smoke test that starts the app headless with Slint's testing backend and opens a fixture.

---

## 16. Suggested remediation roadmap

### Phase 0: stop data loss and hangs (small, local fixes; days)

| # | Item | Size of fix |
|---|---|---|
| 1 | **SH-1**: put the read guard in `file_ops.rs:480-492` in its own block | one line |
| 2 | **PLG-2**: release `AppState` before `sync_filter_states`, and forward `set_filter_enabled` only when the value changes. In eovae, don't call the host from `set_filter_enabled` | small |
| 3 | **DS-1 / DS-2**: give each slide its own handle (or pool) in the dataset pipeline, and disambiguate stems. Delete the duplicate pipeline | small |
| 4 | **UPD-1**: until a marker file and shipped-file manifest exist, refuse portable self-update unless the folder contains only known files. Ship the zip with a top-level `eov/` folder | small |
| 5 | **UI-1**: pass the pane id in every `tool-mouse-*` callback, and focus the pane on pointer-down | small |
| 6 | **GPD-1**: drop the guard before `refresh_window_ui_if_available()` | one line |
| 7 | **PLG-4 / CEL-1**: add `[environment]` to cellaris, and add a CI step that parses every manifest | trivial |
| 8 | **UI-3**: bind `minimap-drag-start`/`-end` in `PaneView` | trivial |
| 9 | **RND-1**: drop `!coarse_tiles.is_empty()` from `new_tiles_loaded` | one line |
| 10 | **UI-4**: render `status-text` and add an error toast | small |
| 11 | **UI-5 / UI-6**: compute distance per axis, label uncalibrated values honestly, and fix the scale-bar label formatting | small |
| 12 | **ANN-1 / ANN-2 / ANN-4**: wrap writes in transactions with WAL; refresh the sidebar after moves; validate colours, names and vertices | small–medium |
| 13 | **EXP-3 / EXP-4**: honour JPEG quality, write DPI metadata, and fix the colour fields | small |
| 14 | **SH-4**: isolate tests from the real `$HOME` | small |
| 15 | **UPD-2**: treat a missing `brew` as a manual bundle | trivial |

### Phase 1: correctness and robustness (weeks)

- **Lock discipline (§2.1, PLG-3):**
  - never call plugins or Slint while holding a host lock;
  - make host APIs non-blocking;
  - publish snapshots;
  - wrap all FFI in `catch_unwind` (PLG-7).
- **Pane identity (§2.3):** use stable pane ids and pass them with every pointer event.
- **WSI core:**
  - handle-health tracking and reopen (WSI-1);
  - composite onto `background-color` (WSI-2);
  - bounded, off-thread thumbnails using associated images (WSI-3);
  - a real LRU with pinning (WSI-4);
  - the per-level clamp fix (WSI-5);
  - LOD bias in log₂ space (WSI-6);
  - a shared OpenSlide handle pool and cache (WSI-9);
  - bounds and regions (WSI-14).
- **Rendering:**
  - GPU texture arrays sized to demand (RND-2);
  - Vulkan, DX12 and software fallbacks (RND-4);
  - stale-request cancellation (RND-5);
  - a GPU memory budget and device-lost handling (RND-7);
  - then HiDPI (RND-3).
- **Colour:**
  - fit stain parameters once per slide, off the UI thread (CLR-1);
  - per-channel β (CLR-2);
  - a fused normalisation and deconvolution pipeline that keeps the residual (CLR-3, CLR-4);
  - sanity checks (CLR-6);
  - CPU/GPU parity with golden-image tests (CLR-5);
  - an "altered colour" badge (CLR-9).
- **Export:** uncapped tile enumeration (EXP-1), and a worker thread with progress (EXP-2).
- **Platform:**
  - Windows console attach (SH-2);
  - robust config loading (SH-3);
  - OpenSlide discovery via pkg-config (SH-5);
  - workspace fixes (SH-6);
  - command-line argument handling (SH-7);
  - Flatpak permissions (UPD-5);
  - macOS helper linking (UPD-4);
  - Homebrew detection (UPD-3);
  - proxy support (UPD-13);
  - CI on all three OSes (UPD-14).
- **Plugins:**
  - eovae: level mapping, preprocessing, per-slide keys, memory budget, error reporting, device choice and batching (VAE-3 … VAE-10);
  - cellaris: MPP matching, runtime loading and readout blending (CEL-3 … CEL-5);
  - gamepad: focus gating and no OS-level injection by default (GPD-2).

### Phase 2: plugin API 1.0 (one deliberate breaking release)

Implement §2.6 in this order:

1. Versioned prefix-type ABI with a handshake.
2. Plugin instances and a threading contract.
3. An SDK crate with `export_plugin!`, a mock host, and examples.
4. Events.
5. Retained overlay layers, with tool registration.
6. UI surfaces with handles.
7. Services: directories, settings, jobs, notifications, relative navigation, and slide metadata and associated images.
8. Namespaced, ordered filters, plus GPU filters run by the host from plugin-supplied WGSL.
9. Trust: id validation, hash and signature checks at load, and a Plugins panel.

At the same time, remove the hard-coded plugin ids from the host (PLG-8) and the dead API surface (PLG-19).

### Phase 3: workbench features (ordered by value to pathologists)

1. Slide label and overview panel, with a privacy toggle.
2. Keyboard navigation and magnification presets, with a single remappable shortcut table.
3. Measurement suite (line, polyline, area, perimeter, angle, HPF circle) with manual calibration.
4. Rotation and flip.
5. Linked navigation across different slides, with per-pane offset and rotation.
6. Session restore, bookmarks and navigation history.
7. ICC colour management.
8. Annotation shapes and classes, with GeoJSON import/export (QuPath interoperability).
9. Counting tool, with live indices and an HPF reticle.
10. Plugins panel, safe mode and GUI updates.
11. Case/series organisation, with DICOM series grouping.
12. Full-screen/presentation mode, single-instance mode, file associations and deep links.

### Phase 4: research platform
- **Multichannel and multiplex fluorescence**, via a `SlideSource` abstraction and an OME-TIFF reader.
- **Batch runs and integration:**
  - headless and batch plugin runs, with run manifests;
  - a scripting interface;
  - import of external overlays (cell detections, heatmaps).
- **Dataset extraction** at a target MPP, with tissue masks, annotation labels and resume.
- **IHC quantification** tools.
- **Writing pyramidal TIFF/OME-TIFF** and de-identified slide copies.

---

## 17. Appendix: method and verification

### 17.1 Method
- **Coverage.** The codebase (about 72k lines of Rust and Slint, plus build, packaging and release scripts) was split into ten scopes:
  1. plugin ABI and host runtime
  2. install/update/packaging
  3. `common` core
  4. colour, export and dataset
  5. rendering
  6. app state and interaction
  7. shell, CLI, config and docs
  8. the annotations plugin
  9. the eovae plugin
  10. the gamepad and cellaris plugins
- **Reading.** Each scope was audited by reading every file in full and tracing call paths across crates.
- **Scratch reproductions.** Where useful, claims were checked with small scratch programs outside the repository:
  - ctypes experiments against OpenSlide 4.0.0 on the repository's fixture slides (sticky errors, transparent gaps, per-handle memory);
  - standalone copies of `cache.rs`, `blitter.rs`, `stain.rs` and `postprocess.rs` for numerical and concurrency checks;
  - reproductions of the `thread_local!`-in-a-rayon-pool and `parking_lot` re-entrancy patterns;
  - SQLite timing with the annotations schema;
  - Python onnxruntime runs of the shipped eovae model;
  - parsing every plugin manifest with the real `eov-plugin-api` 0.4.9;
  - cargo workspace and linker reproductions.
- **Read-only.** No repository file was modified apart from adding this `AUDIT.md`.
- **Lead re-verification.** The lead reviewer independently re-read the code behind every CRITICAL finding and most HIGH findings, marked *re-verified* in the text:
  - SH-1, PLG-1, PLG-2 (the full call chain), PLG-4, PLG-6, PLG-10 (the undo-owner and toggle semantics);
  - DS-1, UPD-1, UI-1, UI-2, UI-3, UI-4, UI-6, UI-9;
  - WSI-3, WSI-4, WSI-5, RND-1, RND-3, RND-4 (the backend selection);
  - EXP-3, EXP-4, ANN-1 (no transactions), ANN-2, ANN-4;
  - VAE-3, VAE-5, GPD-1, GPD-2, UPD-5;
  - and the cellaris `[environment]` omission.

### 17.2 Not verified at runtime
- **Not run:** the GUI was not launched. Hangs and deadlocks were established from code and minimal reproductions of the same locking patterns, not by running eov.
- **Windows and macOS behaviour:** console attachment, the helper swap, Homebrew output, dyld resolution, Gatekeeper and TCC, and the WGI focus behaviour. The affected findings are marked *Likely*.
- **Data not available:** MIRAX, DICOM and Philips slides (only SVS, BIF and a CAMELYON TIFF were available), so the bounds, region and fingerprint effects on those formats are *Likely*.
- **Visual effects:** Lanczos aliasing, seams and GPU texture-array dropouts were derived from the code and simulations, not screenshots.
- **Hardware:** no Vulkan-less host was available to test RND-4's exact failure mode.

### 17.3 Side effects of this audit on the auditor's machine
Running `cargo test --workspace` (the only build action that executes code) modified three things in the home directory, at 2026-10-14 23:16:28:
- `~/.eov/config.toml`
- `~/.config/eov/eovae.json`
- new fixture entries under `~/.cache/eov/plugin-packages`

This is finding SH-4. The previous contents of the two config files are unknown.
- `~/.eov/config.toml` now holds only an `[active_sidebar]` entry pointing at the working-copy eovae UI. Deleting that block restores the default startup.
- Check `eovae.json`. It holds the default model path and the thread and batch settings.
