//! Common crate for eov WSI viewer
//!
//! This crate provides high-performance WSI (Whole Slide Image) file handling,
//! tiling, caching, and rendering utilities.

pub mod blitter;
pub mod cache;
pub mod dataset;
pub mod error;
pub mod export;
pub mod file_id;
pub mod formatting;
pub mod imaging;
pub mod overlay;
pub mod postprocess;
pub mod render;
pub mod stain;
pub mod tile;
pub mod viewport;
pub mod wsi;

pub use cache::TileCache;
pub use error::{Error, Result};
pub use export::ExportSettings;
pub use formatting::{format_decimal, format_file_size, format_optional_decimal, format_u64};
pub use imaging::{
    MeasurementUnit, RgbaImageData, StainNormalization, crop_image_to_viewport_bounds,
    crop_transparent_edges,
};
pub use render::{FilteringMode, RenderBackend, TrilinearLevels, calculate_trilinear_levels};
pub use tile::{TileCoord, TileData, TileManager};
pub use viewport::{Viewport, ViewportState};
pub use wsi::{WsiFile, WsiLevel, WsiProperties};
