//! Dataset generation for ML workflows: deterministic patch extraction from
//! whole-slide images, shared by the CLI (`eov dataset patches`) and the GUI.
//!
//! [`run_dataset`] takes the basic grid ([`DatasetPatchesConfig`]) plus
//! [`DatasetPatchesOptions`]: target resolution (MPP or magnification),
//! tissue-mask filtering, stain normalisation of patches, GeoJSON
//! annotation-driven extraction with labels and masks, per-slide limits
//! with seeded sampling, JPEG output, resume and dry runs. Progress and
//! cancellation go through a shared [`DatasetControl`]; run it on a worker
//! thread.
//!
//! # Output layout
//!
//! ```text
//! <out>/
//!   slides/<slide_id>/<slide_id>_x{X:06}_y{Y:06}_s{tile}.png   RGB tiles (level-0 origin)
//!   masks/<slide_id>/...png                                     8-bit masks (optional)
//!   metadata.csv | metadata.jsonl | metadata.json               one record per tile
//!   slides.csv | slides.jsonl                                   one record per slide
//!   dataset.json                                                parameters, status, totals
//!   classes.json                                                mask class indices (optional)
//! ```

mod annotations;
mod config;
mod discovery;
mod grid;
mod hashing;
mod manifest;
mod metadata;
mod output;
mod pipeline;
mod resample;
mod sampling;

pub use annotations::{AnnotationError, AnnotationPolygon, AnnotationSet, Coverage, UNCLASSIFIED};
pub use config::{
    AnnotationOptions, AnnotationSource, DatasetConfigError, DatasetPatchesConfig,
    DatasetPatchesOptions, ExistingOutput, MaskOutput, MetadataFormat, PatchImageFormat,
    PatchNormalization, STRIDE_RANGE, THREADS_RANGE, TILE_SIZE_RANGE, TargetResolution,
    TissueFilter,
};
pub use discovery::{
    Discovery, SUPPORTED_SLIDE_EXTENSIONS, SkippedInput, SlideInput, dicom_series_uid,
    discover_slides, expand_inputs, is_supported_slide_extension, non_slide_tiff_reason,
};
pub use grid::{PatchCoord, PatchGrid, generate_patch_coords};
pub use hashing::slide_sha256;
pub use manifest::{DatasetManifest, ExtractionParameters, RunStatus, RunTotals};
pub use metadata::{SlideRecord, SlideStatus, TileRecord};
pub use pipeline::{
    DatasetControl, DatasetError, DatasetPatchesProgress, DatasetPatchesReport, DatasetPhase,
    DatasetPlan, PlannedSlide, SlideGeometry, SlideReport, SlideSkipReason, plan_dataset,
    run_dataset, run_dataset_patches, run_dataset_patches_with_progress,
};
pub use resample::resample_rgba;
