# eov-common

`eov-common` is a Rust library for working with pyramid-based whole-slide
images (WSI). It contains the reusable data and rendering primitives used by
the [eov WSI viewer](https://github.com/eosin-platform/eov), but it does not
include the desktop application or its UI.

## Features

- OpenSlide-backed access to WSI files through a small, safe in-house
  binding (no third-party binding crate; non-UTF-8 paths work on Unix).
- One shared, thread-safe OpenSlide handle per slide (`WsiFile` is cheap to
  clone), one process-wide OpenSlide tile cache, and automatic, rate-limited
  recovery from OpenSlide's sticky errors: a corrupt tile fails on its own
  instead of disabling the slide (`WsiFile::health` reports what happened).
- Region and tile reads that composite OpenSlide's premultiplied pixels onto
  the slide's background colour, so gaps and areas outside the scan read as
  glass (`AlphaMode` keeps the alpha channel when needed).
- Full property map, associated images (label, macro, thumbnail), slide and
  associated-image ICC profiles, content bounds and scanned regions, and
  validated metadata (MPP, fractional objective power, scan date, stain).
- Thumbnails with bounded memory and I/O for every slide, including
  single-level ones.
- Synthetic low-resolution levels for slides without a deep enough pyramid.
- Tile coordinates, tile loading, and a thread-safe, byte-budgeted LRU tile
  cache with per-frame protection, pinning and per-file eviction.
- Viewport state and render-level calculations for bilinear, trilinear, and
  Lanczos filtering workflows; CPU samplers use the same pixel-centre
  convention as the GPU, and Lanczos is separable and anti-aliases when
  minifying.
- A format-aware `SlideIdentity` for keying per-slide data.
- RGBA image utilities, cropping, overlays, export settings, and image
  post-processing helpers.
- Stain normalization and color deconvolution primitives for histology images.
- Deterministic fixed-grid dataset patch extraction with optional CSV or JSON
  metadata.

The supported file formats are the formats available through
[OpenSlide](https://openslide.org/), including common formats such as SVS,
TIFF, NDPI, SCN, MRXS, BIF, CZI, and DICOM whole-slide images.

## Installation

Add the crate to your project:

```toml
[dependencies]
eov-common = "1.0"
```

`eov-common` links the OpenSlide C library directly. **OpenSlide 4.0 or
newer is required** (the crate uses the shared-cache and ICC-profile APIs
introduced in 4.0). The build script finds the library with `pkg-config`;
set `OPENSLIDE_LIB_DIR` to a directory containing the library (on Windows,
the `openslide.lib` import library) to override it. Rust 1.88 or newer is
required.

On Debian or Ubuntu releases that ship OpenSlide 4, the development package
is typically:

```bash
sudo apt-get install libopenslide-dev
```

See the [OpenSlide installation documentation](https://openslide.org/download/)
for other platforms and package managers.

## Quick Start

Open a slide, inspect its pyramid metadata, and read the first tile as RGBA
bytes:

```rust,no_run
use eov_common::WsiFile;

fn main() -> eov_common::Result<()> {
    let slide = WsiFile::open("slide.svs")?;
    let properties = slide.properties();

    println!(
        "{}: {}x{} ({} pyramid levels)",
        properties.filename,
        properties.width,
        properties.height,
        properties.levels.len(),
    );

    let rgba = slide.read_tile(0, 0, 0)?;
    println!("first tile: {} bytes", rgba.len());

    // A bounded thumbnail, whatever the slide's pyramid looks like.
    let thumbnail = slide.thumbnail(256)?;
    println!("thumbnail: {}x{}", thumbnail.width, thumbnail.height);

    Ok(())
}
```

`WsiFile::read_region` and `WsiFile::read_tile` return packed, opaque RGBA
pixel data. For viewer-style loading, pass a `WsiFile` to `TileManager` and
use `TileCache` to reuse decoded tiles. Give every worker thread a clone of
one `WsiFile` (or one shared `TileManager`) rather than opening the slide
again:

```rust,no_run
use eov_common::{TileCache, TileCoord, TileManager, WsiFile};
use std::sync::Arc;

fn load_first_tile(path: &str) -> eov_common::Result<()> {
    let cache = Arc::new(TileCache::new());
    let slide = WsiFile::open(path)?.with_synthetic_levels();
    let manager = TileManager::new(slide, 0).with_tile_cache(Arc::clone(&cache));
    let coord = TileCoord::new(0, 0, 0, 0, manager.tile_size_for_level(0));
    let tile = manager.load_tile_cached(coord)?;
    println!("tile {}x{} (+{} px border)", tile.width, tile.height, tile.border);
    Ok(())
}
```

## Main API Areas

- [`WsiFile`](https://docs.rs/eov-common/latest/eov_common/struct.WsiFile.html)
	and [`WsiProperties`](https://docs.rs/eov-common/latest/eov_common/struct.WsiProperties.html)
	provide slide access and metadata; the `openslide` module is the raw
	binding.
- [`TileManager`](https://docs.rs/eov-common/latest/eov_common/struct.TileManager.html),
	[`TileData`](https://docs.rs/eov-common/latest/eov_common/struct.TileData.html),
	and [`TileCoord`](https://docs.rs/eov-common/latest/eov_common/struct.TileCoord.html)
	support tiled loading.
- [`TileCache`](https://docs.rs/eov-common/latest/eov_common/struct.TileCache.html)
	provides concurrent caching with tile-count and byte-size limits;
	`CacheBudget::for_display` sizes it for a display.
- The `dataset` module provides fixed-grid patch extraction and metadata types
	for machine-learning workflows.
- The `imaging`, `postprocess`, `stain`, `overlay`, `render`, `blitter` and
	`viewport` modules provide reusable image and viewing primitives.

See the [API documentation](https://docs.rs/eov-common) for the complete
public interface.

## Benchmarks

```bash
cargo bench -p eov-common
```

The slide benchmarks use the repository's (git-ignored) `fixtures/`
directory and are skipped when it is missing; the blitter, cache and
viewport benchmarks run everywhere.

## Project

`eov-common` is part of the [eov workspace](https://github.com/eosin-platform/eov).
The repository README documents the desktop viewer, command-line interface,
packaging, and application-level features.

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/master/LICENSE-MIT),
[Apache-2.0](https://github.com/eosin-platform/eov/blob/master/LICENSE-APACHE-2.0),
or [GPLv3](https://github.com/eosin-platform/eov/blob/master/LICENSE-GPLv3), at
your option.

This crate links OpenSlide, which is licensed separately under the LGPL 2.1.
