//! Common crate for eov WSI viewer
//!
//! This crate provides high-performance WSI (Whole Slide Image) file handling,
//! tiling, caching, and rendering utilities.
//!
//! * [`wsi`]: [`WsiFile`], a cheap-to-clone handle to a slide opened with
//!   OpenSlide (4.0 or newer, linked through the in-house [`openslide`]
//!   binding). Region reads return opaque RGBA composited onto the slide's
//!   background; handles are shared per slide, recover from OpenSlide's
//!   sticky errors, and use one process-wide OpenSlide cache. Also bounded
//!   thumbnails, associated images, ICC profiles, the full property map,
//!   content bounds and synthetic levels for flat slides.
//! * [`tile`] and [`cache`]: tile addressing, loading and a byte-budgeted
//!   LRU [`TileCache`] with per-frame protection and pinning.
//! * [`render`], [`blitter`], [`viewport`]: level selection, pixel-centre
//!   CPU resampling and viewport math.
//! * [`file_id`]: [`SlideIdentity`], the stable key for per-slide data.

pub mod blitter;
pub mod cache;
pub mod dataset;
pub mod error;
pub mod export;
pub mod file_id;
pub mod formatting;
pub mod imaging;
pub mod openslide;
pub mod overlay;
pub mod paths;
pub mod postprocess;
pub mod render;
pub mod stain;
pub mod tile;
pub mod viewport;
pub mod wsi;

#[cfg(test)]
pub(crate) mod test_support;

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
