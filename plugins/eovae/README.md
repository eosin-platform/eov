# EOV VAE Plugin

`plugins/eovae` adds ONNX-based VAE reconstruction analysis for whole-slide images inside EOV.

## What it does

- Loads a single `.onnx` VAE model.
- Inspects model inputs and outputs, including names, element types, and shapes.
- Infers likely image input/output tensors and the expected layout (`NCHW` or `NHWC`).
- Analyzes either the current viewport or the whole slide in background worker threads.
- Caches reconstructed tiles in memory.
- Visualizes `Original`, `Reconstruction`, `Error Map`, and `Difference` through the plugin viewport filter API.
- Tracks aggregate error statistics and the highest-error regions.
- Exposes jump actions from the sidebar to frame suspicious regions.
- Supports cooperative cancellation between tiles.

## Current assumptions

- The model has one image-like tensor input and at least one image-like tensor output.
- Input and output tensors are float tensors.
- Reconstruction output is spatially aligned with the input tile.
- Inference currently runs one tile at a time. The job pipeline is structured so batching can be added without changing the sidebar or filter contract.

## Notes

- Only `.onnx` files are accepted.
- The crate vendors ONNX Runtime during the build. On Linux, the core runtime is linked from the vendored static archive and any ONNX Runtime shared sidecars are copied into the Cargo output directory for packaging.
- `plugins/eovae/scripts/package_eop.py` expects `--native-lib-dir` to point at the Cargo output directory such as `target/x86_64-unknown-linux-gnu/release`, where `build.rs` stages vendored ONNX Runtime shared artifacts for inclusion in the `.eop` archive.
- The plugin library is built with an `$ORIGIN` or `@loader_path` rpath so the extracted plugin resolves the vendored ONNX Runtime from the same directory.
- Cache invalidation is tied to slide path, model path, selected input/output, inferred layout, and tile size.
- GPU execution providers are requested when enabled in the sidebar, but the plugin still falls back to CPU-compatible extraction for output tensors.
