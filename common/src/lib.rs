//! Common crate for eov WSI viewer
//!
//! This crate provides high-performance WSI (Whole Slide Image) file handling,
//! tiling, caching, and rendering utilities.
//!
//! * [`wsi`]: [`WsiFile`], a cheap-to-clone handle to a slide opened with
//!   OpenSlide (4.0 or newer, linked through the in-house [`openslide`]
//!   binding) or, for OME-TIFF, with eov's own reader. Region reads return
//!   opaque RGBA composited onto the slide's background; handles are shared
//!   per slide, recover from OpenSlide's sticky errors, and use one
//!   process-wide tile cache. Also bounded thumbnails, associated images,
//!   ICC profiles, the full property map, content bounds and synthetic
//!   levels for flat slides.
//! * [`channels`]: the channels of multichannel (fluorescence) slides and
//!   their display: colours, windows, gamma, palettes, auto contrast.
//! * [`tile`] and [`cache`]: tile addressing, loading and a byte-budgeted
//!   LRU [`TileCache`] with per-frame protection and pinning.
//! * [`render`], [`blitter`], [`viewport`]: level selection, pixel-centre
//!   CPU resampling and viewport math.
//! * [`file_id`]: [`SlideIdentity`], the stable key for per-slide data.
//! * [`convert`]: pyramidal TIFF and OME-TIFF copies of a slide or a
//!   region, optionally de-identified.

pub mod blitter;
pub mod cache;
pub mod channels;
pub mod convert;
pub mod dataset;
pub mod error;
pub mod export;
pub mod file_id;
pub mod formatting;
pub mod heatmap;
pub mod imaging;
pub mod job;
pub(crate) mod ome_tiff;
pub mod openslide;
pub mod overlay;
pub mod paths;
pub mod postprocess;
pub mod processing;
pub mod qc;
pub mod render;
pub mod stain;
pub mod tile;
pub mod tissue;
pub mod viewport;
pub mod wsi;

#[cfg(test)]
pub(crate) mod test_slides;
/// Writers for small synthetic slides (uncompressed TIFF pyramids that
/// OpenSlide opens), for this crate's tests and, with the `test-support`
/// feature, for other crates' tests.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support;

/// Compiles the README's examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub use cache::{CacheBudget, TileCache};
pub use error::{Error, Result};
pub use export::ExportSettings;
pub use file_id::SlideIdentity;
pub use formatting::{
    format_decimal, format_file_size, format_length_um, format_mpp, format_optional_decimal,
    format_significant, format_u64,
};
pub use imaging::{
    MeasurementUnit, RgbaImageData, StainNormalization, crop_image_to_viewport_bounds,
    crop_transparent_edges,
};
pub use render::{
    FilteringMode, RenderBackend, TrilinearLevels, TrilinearOptions, calculate_trilinear_levels,
};
pub use tile::{TileCoord, TileData, TileManager, TileRange};
pub use viewport::{Viewport, ViewportState};
pub use wsi::{
    AlphaMode, AssociatedImage, HealthState, OpenOptions, SlideBounds, SlideHealth, Thumbnail,
    ThumbnailOptions, ThumbnailRegion, ThumbnailSource, WsiFile, WsiLevel, WsiProperties,
};
