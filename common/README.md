# eov-common

`eov-common` is a Rust library for working with pyramid-based whole-slide
images (WSI). It contains the reusable data and rendering primitives used by
the [eov WSI viewer](https://github.com/eosin-platform/eov), but it does not
include the desktop application or its UI.

## Features

- OpenSlide-backed access to WSI files and slide metadata.
- Pyramid-level inspection, region reads, tile reads, and thumbnails.
- Tile coordinates, tile loading, and a thread-safe LRU tile cache.
- Viewport state and render-level calculations for bilinear, trilinear, and
	Lanczos filtering workflows.
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
eov-common = "0.4"
```

`eov-common` uses [`openslide-rs`](https://crates.io/crates/openslide-rs), so
OpenSlide must be installed and discoverable by the linker on the target
system. On Debian or Ubuntu, the development package is typically:

```bash
sudo apt-get install libopenslide-dev
```

See the [OpenSlide installation documentation](https://openslide.org/download/)
for other platforms and package managers.

## Quick Start

Open a slide, inspect its pyramid metadata, and read the first tile as RGBA
bytes:

```rust
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

		Ok(())
}
```

`WsiFile::read_region` and `WsiFile::read_tile` return packed RGBA pixel data.
For viewer-style loading, pass a `WsiFile` to `TileManager` and use
`TileCache` to reuse decoded tiles:

```rust
use eov_common::{TileCache, TileCoord, TileManager, WsiFile};

fn load_first_tile(path: &str) -> eov_common::Result<()> {
		let manager = TileManager::new(WsiFile::open(path)?, 0);
		let coord = TileCoord::new(0, 0, 0, 0, manager.tile_size());
		let tile = manager.load_tile_sync(coord)?;

		let cache = TileCache::new();
		cache.insert(tile);

		Ok(())
}
```

## Main API Areas

- [`WsiFile`](https://docs.rs/eov-common/latest/eov_common/struct.WsiFile.html)
	and [`WsiProperties`](https://docs.rs/eov-common/latest/eov_common/struct.WsiProperties.html)
	provide slide access and metadata.
- [`TileManager`](https://docs.rs/eov-common/latest/eov_common/struct.TileManager.html),
	[`TileData`](https://docs.rs/eov-common/latest/eov_common/struct.TileData.html),
	and [`TileCoord`](https://docs.rs/eov-common/latest/eov_common/struct.TileCoord.html)
	support tiled loading.
- [`TileCache`](https://docs.rs/eov-common/latest/eov_common/struct.TileCache.html)
	provides concurrent caching with tile-count and byte-size limits.
- The `dataset` module provides fixed-grid patch extraction and metadata types
	for machine-learning workflows.
- The `imaging`, `postprocess`, `stain`, `overlay`, `render`, and `viewport`
	modules provide reusable image and viewing primitives.

See the [API documentation](https://docs.rs/eov-common) for the complete
public interface.

## Project

`eov-common` is part of the [eov workspace](https://github.com/eosin-platform/eov).
The repository README documents the desktop viewer, command-line interface,
packaging, and application-level features.

## License

Licensed under [MIT](https://github.com/eosin-platform/eov/blob/main/LICENSE-MIT),
[Apache-2.0](https://github.com/eosin-platform/eov/blob/main/LICENSE-APACHE-2.0),
or [GPLv3](https://github.com/eosin-platform/eov/blob/main/LICENSE-GPLv3), at
your option.

This crate depends on OpenSlide, which is licensed separately under the LGPL.
