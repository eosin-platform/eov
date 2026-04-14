//! Configuration types for dataset generation.

use std::path::PathBuf;

/// Image metadata format for the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataFormat {
    Csv,
    Json,
}

/// Configuration for the `dataset patches` pipeline.
///
/// This struct is designed to be constructed from CLI arguments or from a
/// future GUI dialog—keep it free of app-specific types.
#[derive(Debug, Clone)]
pub struct DatasetPatchesConfig {
    /// One or more input paths. Each may be a slide file or a directory
    /// containing slide files.
    pub inputs: Vec<PathBuf>,
    /// Output directory root.
    pub output_dir: PathBuf,
    /// Tile width and height in pixels (tiles are square).
    pub tile_size: u32,
    /// Step size between tile origins in pixels.
    pub stride: u32,
    /// Optional metadata output format.
    pub metadata_format: Option<MetadataFormat>,
}
