# eovae: VAE reconstruction-error analysis for eov

eovae runs an ONNX autoencoder (such as [HistoVAE](HistoVAE/)) over a slide
and shows where the reconstruction differs from the tissue: unusual tissue
reconstructs worse. It is a research tool, **not for diagnosis**.

- **Error map**: a whole-slide heat map of per-tile error (colour-blind-safe
  viridis ramp, scaled to the p05–p95 of the analysed tiles, with a legend).
- **Reconstruction** and **Difference**: the model's reconstruction (shown
  with the pane's display adjustments) and the per-pixel error of the
  visible area, at screen resolution.
- **Results**: error histogram, summary statistics, the highest-error
  regions (hover to outline, click to frame), how much of the slide is
  covered, and CSV export with a provenance manifest.
- A **tile grid** overlay, and automatic analysis of the visible tiles while
  an overlay is shown.
- **Batch runs** over a cohort without a window (`eov batch run
  eovae:analyze`): per-tile scores, a heat map and a summary with the run's
  provenance for each slide.

eovae 1.0 needs eov 1.0 or newer (plugin API 1.0).

## Using it

Open the sidebar with the VAE toolbar button.

1. **Model → Load ONNX…** picks a model. eovae checks it can feed it, hashes
   it with its external weights and prepares the inference engine in the
   background; the sidebar shows the device it runs on.
2. **Analysis** sets the **device** (Auto, CPU or CUDA; independent of how eov
   renders), the **resolution** (Auto matches the model's training µm/px when
   both are known; otherwise pick one of the slide's real pyramid levels) and
   threads/batch size. **Viewport**, **ROI** (the tiles under the slide's
   region of interest, while it has one) and **Whole slide** start a job;
   progress and Cancel are also in eov's job list.
3. Choose an overlay in a pane's HUD menu (or under **Results**, for the
   focused pane). Each pane has its own overlay and tile grid, so two panes
   can show the same slide as reconstruction and error map side by side. In
   overlay modes the visible tiles are analysed automatically when you are
   zoomed in far enough (at most 400 tiles, each at least 24 px on screen).

Background (glass) tiles are detected with the rule HistoVAE's training used
and skipped, on every device. Tiles that cannot be read are marked and
retried by later runs; a job fails, with the reason, only when many do.

Reconstructions are shown the way their pane shows the slide: eovae applies
the pane's gamma, brightness and contrast with eov's tone curve, and its
sharpening while the overlay is at screen resolution (zoomed in past the
analysed level it is coarser than the screen), and follows changes within
about a second. Stain
normalisation, colour deconvolution and isolated stains cannot be reproduced;
while a pane uses them the sidebar says so, and they should be turned off
before comparing Original and Reconstruction. The error map and the
difference are computed on the slide's own pixels, so display adjustments do
not change them.

With an eov that cannot hide a layer in some panes only, every pane of a
slide shows the overlay chosen last, and eov's log says so.

## Models

eovae accepts a model when it can feed it:

- exactly one input: a 4-D float32 or float16 RGB image, NCHW or NHWC, square
  (`[N, 3, S, S]`); fixed or dynamic batch and size;
- at least one output with the input's layout and size (the reconstruction).
  Several image outputs can be chosen between; an extra non-image float
  output (e.g. `mu`) is stored with the results as an embedding.

Encoder-only exports (outputs `mu`, `logvar`) are rejected with a hint.
Models with a **fixed batch** are fed full batches (the last one of a job is
padded); export with a dynamic batch axis for faster small runs.

### Preprocessing contract

How pixels become model inputs must match training, or every error is
biased. eovae applies, in order: the defaults (HistoVAE's training pipeline:
transparent pixels composited onto white, pixels darker than 4 in every
channel made white, RGB mapped to [-1, 1], output mapped back with
`(x + 1) · 127.5`), a built-in card for models it knows by hash, the model's
ONNX `metadata_props`, a sidecar file `<model>.onnx.eovae.json`, and your
changes in **Model → Preprocessing**. The sidebar says which source applies.

| key (`eovae.` prefix in metadata) | values |
| --- | --- |
| `input_range`, `output_range` | `-1,1`, `0,1`, `0,255` |
| `input_mean` + `input_std`, `output_mean` + `output_std` | three values (or one), applied to `v / 255` |
| `background` | `white` or `black` (alpha compositing) |
| `black_to_white` | threshold (`4`), or `off` |
| `channel_order` | `rgb` or `bgr` |
| `mpp` | µm per pixel the model was trained at |
| `tile_size` | input size for models with dynamic spatial dimensions |

Example sidecar:

```json
{ "mpp": 0.5, "input_range": "0,1", "output_range": "0,1" }
```

The HistoVAE reference model (`HistoVAE/pretrained/HistoVae_reconstruct.onnx`
with its `.onnx.data` weights) is recognised and declares 0.227 µm/px
(CAMELYON17 level 0). On a 0.49 µm/px slide the sidebar warns that errors are
inflated by the resolution mismatch.

## Results and storage

Results are keyed by **slide** (the host's slide identity, or a content
fingerprint of the file and its pyramid until the host provides one) and by
**run**: a hash of the model and its weights, input and output, layout, tile
size, preprocessing contract and version, level and downsample, device,
numeric precision (TF32 on CUDA) and plugin version. Changing any of these
starts a separate result set; changing one while a job runs cancels the job.

- **Memory**: per-tile scores are small; tile pixels live in an LRU cache
  limited to 512 MB by default (`memory_limit_mb`). Evicted pixels are
  recomputed when a view needs them. The error map is sent to eov in chunks
  of 64×64 map pixels, and a job re-sends only the chunks its new tiles
  touch (its colour scale is held steady while the job runs and set exactly
  when it ends).
- **Disk**: `results.sqlite` in eov's cache directory for eovae, limited to
  2 GB by default (`cache_limit_mb`), least recently used runs first. It
  stores scores, background markers and embeddings, and the reconstructions
  only if **Keep reconstructions on disk** is on. It stores no file paths.
  **Cache → Clear cache…** empties it.
- **Settings** are stored by eov (per-plugin settings), not in your home
  directory. Settings of eovae 0.x (`eovae.json`) are not imported, and its
  old database (`eovae-latent-cache.db` in eov's config directory, which
  contains slide paths) is no longer used and can be deleted.

**Export CSV…** writes one row per tile (`col,row,x,y,width,height,level,
state,mae,max_error`, level-0 coordinates of the in-slide part of the tile)
and `<name>.manifest.json` with the run specification and slide identity.

## Batch runs

```sh
eov batch run eovae:analyze SLIDES… --out DIR [--param name=value …]
```

eov starts eovae without a window and sends it one slide at a time
(`eov batch list` shows the command and its parameters). For each slide,
eovae analyses every tile at the model's resolution, skipping glass, and
writes into `DIR/<slide>/`:

- `tiles.csv`: one row per tile: `col,row` in the grid, `x,y,width,height`
  in level-0 pixels (the part inside the slide), `width_um,height_um`,
  `level`, `state` (`analysed`, `background` or `unreadable`), a
  `background` flag (0/1), and the anomaly score `mae` (mean absolute
  reconstruction error over the tile's pixels and RGB channels, 0–1) with
  `max_error`, the largest per-pixel error;
- `heatmap.png`: one pixel per tile (`col,row`), viridis from the colour
  range's low to high end, unreadable tiles red, glass and tiles not
  analysed transparent. Each pixel covers `level0_px_per_pixel` level-0
  pixels, given in the summary and in the PNG's text chunks;
- `summary.json`: tiles analysed, background, unreadable and not analysed;
  the score distribution (count, mean, std, min, p05, p25, median, p75,
  p95, max); the model's file, SHA-256 over the model and its external
  weights, model card, input, output, layout, tile size and training µm/px;
  the preprocessing (its source and the full contract); the level,
  downsample and µm/px used and the ratio to the model's; the device asked
  for and used, its precision and any CPU fallback, the batch size and
  threads; the ONNX Runtime and plugin versions; the run key; warnings. The
  same summary goes into eov's `DIR/batch.json`, with eov's version, the
  plugin package's SHA-256 and the parameters.

The settings chosen in eov's VAE sidebar apply: the model, device, batch
size, threads, TF32, resolution, background skipping and preprocessing
changes. They are the parameters' defaults, so `batch.json` records them,
and a parameter overrides one for the run only (nothing is saved):

| parameter | values |
| --- | --- |
| `model` | path of an `.onnx` file |
| `device` | `auto` (CUDA, else the CPU), `cpu`, `cuda` |
| `batch_size`, `threads` | numbers (`threads=0`: every core) |
| `tf32`, `skip_background` | `true`, `false` |
| `resolution` | `auto` (the level closest to the model's µm/px) or a downsample, e.g. `4` |
| `heat_range` | `auto` (p05–p95 of each slide) or `lo,hi` (the same colour scale for every slide) |
| `cache` | `off` (default: every tile is computed, nothing stored), `store` (results are also kept for eov's viewer), `reuse` (tiles stored before for the same run, as keyed above, are taken from the cache) |

A slide that cannot be analysed (an unreadable file, a model that cannot
run, CUDA asked for but unavailable) is reported as failed in `batch.json`
with the reason, and the run goes on with the next one. Memory stays
bounded on any slide size: tiles stream through the pipeline and only one
score per tile is kept. For example, to compare a cohort on one scale with
CUDA:

```sh
eov batch run eovae:analyze cohort/ --out results/ \
    --param device=cuda --param heat_range=0,0.05
```

## ONNX Runtime and CUDA

eovae opens ONNX Runtime (1.22 or newer) when the first model loads, from the
first of:

1. `EOVAE_ORT_LIB` (or `ORT_DYLIB_PATH`): a library file;
2. the library chosen with **Model → Runtime…** (a runtime already loaded in
   this session stays until eov restarts);
3. `onnxruntime/` in the plugin package (official packages bundle 1.28.3);
4. the system library search path (`libonnxruntime.so.1`,
   `libonnxruntime.dylib`, `onnxruntime.dll`).

CUDA is used only when selected (or with Auto) and only if the runtime has
the CUDA provider (the `cuda12`/`cuda13` builds). CUDA and cuDNN are never
bundled; they come from the system. eovae preloads them from
`EOVAE_CUDA_LIB_DIR` and `EOVAE_CUDNN_LIB_DIR` (or `CUDA_PATH`/`CUDA_HOME`)
when set, and otherwise relies on the normal library search path. With Auto,
any CUDA problem falls back to the CPU and the sidebar says why; with CUDA
selected, the problem is shown instead. The first CUDA run is a short
warm-up (the cuDNN algorithm search is heuristic).

## Building

```sh
cargo build --release --manifest-path plugins/eovae/Cargo.toml
```

eovae is its own Cargo workspace (eov's workspace does not include it). The
default `onnx` feature adds the ONNX Runtime backend; `--no-default-features`
builds a plugin without inference, which is how the tests run.

```sh
cargo test --manifest-path plugins/eovae/Cargo.toml
```

The tests run the plugin against the plugin API's mock host with a synthetic
slide and a test inference engine; no ONNX Runtime, GPU or slide files are
needed, and nothing outside temporary directories is touched. Two optional
checks need more:

- `scripts/reference_mae.py <slide> <model> <dir>` (Python with numpy,
  Pillow, openslide-python, onnxruntime) writes reference errors computed
  with HistoVAE's training preprocessing; then
  `EOVAE_ORT_LIB=… EOVAE_REFERENCE_TILES=<dir> cargo test --release
  matches_the_python_reference -- --ignored` compares eovae's pipeline with
  it (agreement to about 1e-7).
- `scripts/check_ui.sh` compiles the sidebar with the Slint version eov uses
  and checks every pushed property against its declared type.
- `EOVAE_ORT_LIB=… cargo test --release batch_runs_with_onnx_runtime --
  --ignored` runs the batch command with ONNX Runtime and the HistoVAE
  reference model (another model: `EOVAE_BATCH_MODEL`).

## Packaging

```sh
python3 plugins/eovae/scripts/package_eop.py \
    --library plugins/eovae/target/release/libeovae.so \
    --output dist/eovae-linux-x64.eop --fetch-ort
```

`scripts/run-with-plugins.sh` (eov's development launcher) packages the
debug build with `--ort-dir "$EOVAE_ORT_DIR"` when that is set, else with
`--no-ort`.

An `.eop` is a tar of `plugin.toml`, the library, `ui/` and an optional
`onnxruntime/` folder; `--zstd` compresses it (eov 1.0 reads plain and
zstd-compressed packages; the script uses Python 3.14's zstd module, the
`zstandard` module or the `zstd` command). `--fetch-ort` downloads the
pinned ONNX Runtime 1.28.3 for `--platform` (`linux-x64`, `linux-aarch64`,
`osx-arm64`, `win-x64`, `win-arm64`; default: this machine) and
`--ort-flavor` (`cpu`, `cuda12`, `cuda13`) and verifies its SHA-256;
`--ort-archive` uses an archive you downloaded, `--ort-dir` (or
`EOVAE_ORT_DIR`) a folder of libraries, and `--no-ort` bundles none. Only the files the runtime needs are staged, under
their SONAME (no symlinks or duplicates), with its LICENSE and
ThirdPartyNotices; CUDA and cuDNN are never included. Archives are
reproducible (`SOURCE_DATE_EPOCH`, default 0; compressed ones for a given
zstd implementation). A Linux CPU package is about 30 MB, or about 9 MB
with `--zstd`.

## Module map

`src/plugin.rs` (state, events, model loading, engines, jobs),
`src/presenter.rs` (pushes the sidebar, overlays, HUD and settings),
`src/sidebar.rs`, `src/batch.rs` (`eov batch run eovae:analyze`),
`src/pipeline.rs` (one job), `src/engine.rs` and `src/ort_backend.rs`
(inference), `src/tiling.rs`, `src/contract.rs`, `src/model.rs`,
`src/onnx_scan.rs`, `src/run.rs`, `src/slide.rs`, `src/results.rs`,
`src/db.rs`, `src/render.rs`, `src/png.rs`, `src/stats.rs`.
