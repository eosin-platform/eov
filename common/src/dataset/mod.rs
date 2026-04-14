//! Dataset generation utilities for ML workflows.
//!
//! This module provides deterministic fixed-grid patch extraction from whole-slide
//! images. The core logic lives here (in `common`) so that both the CLI and a
//! future GUI export dialog can reuse it.

mod config;
mod discovery;
mod grid;
mod metadata;
mod output;
mod pipeline;

pub use config::{DatasetPatchesConfig, MetadataFormat};
pub use discovery::{expand_inputs, is_supported_slide_extension, SUPPORTED_SLIDE_EXTENSIONS};
pub use grid::generate_patch_coords;
pub use metadata::TileRecord;
pub use pipeline::{run_dataset_patches, DatasetPatchesReport, SlideReport, SlideSkipReason};
