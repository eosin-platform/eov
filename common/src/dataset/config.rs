//! Configuration types for dataset generation, and their validation (DS-4).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::stain::{NormalizationTarget, StainFitMethod};
use crate::tissue::TissueMethod;

/// Per-tile metadata format(s) written next to the tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataFormat {
    /// `metadata.csv` only.
    Csv,
    /// `metadata.jsonl` (written incrementally) and `metadata.json` (a JSON
    /// array, written when the run ends).
    Json,
    /// `metadata.jsonl` only.
    JsonLines,
}

/// Basic grid parameters of a `dataset patches` run.
///
/// This struct keeps its pre-1.0 fields; everything else is in
/// [`DatasetPatchesOptions`] (see [`run_dataset`](super::run_dataset)).
#[derive(Debug, Clone)]
pub struct DatasetPatchesConfig {
    /// Slide files or directories (searched recursively).
    pub inputs: Vec<PathBuf>,
    /// Output directory root.
    pub output_dir: PathBuf,
    /// Tile width and height in output pixels (tiles are square).
    pub tile_size: u32,
    /// Step between tile origins, in output pixels.
    pub stride: u32,
    /// `None` writes CSV and JSON (`metadata.csv`, `metadata.jsonl`,
    /// `metadata.json`); `Some` writes only that format.
    pub metadata_format: Option<MetadataFormat>,
    /// Worker threads.
    pub threads: usize,
    /// Skip tiles whose fraction of nearly-white pixels (all channels ≥ 220,
    /// or transparent) is **at least** this value, in `(0, 1]`. `None`
    /// keeps all tiles. The GUI's default is 0.8.
    pub white_threshold: Option<f32>,
}

/// Output resolution of the patches.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum TargetResolution {
    /// Micrometres per pixel. Requires the slide's MPP.
    Mpp(f64),
    /// Objective-equivalent magnification (e.g. 20 for "20×"); converted
    /// with the slide's objective power, or as ≈ 10 µm/px ÷ magnification
    /// when only the MPP is known.
    Magnification(f64),
}

/// Tissue-mask filtering, as an alternative or addition to the white
/// threshold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TissueFilter {
    pub method: TissueMethod,
    /// Minimum fraction of tissue pixels in a patch, in `[0, 1]`.
    pub min_fraction: f32,
    /// Pixel budget for the low-resolution mask used to skip obvious
    /// background before reading patches.
    pub mask_max_pixels: u64,
}

impl Default for TissueFilter {
    fn default() -> Self {
        Self {
            method: TissueMethod::Otsu,
            min_fraction: 0.5,
            mask_max_pixels: 4_000_000,
        }
    }
}

/// Stain normalisation of the written patches (one slide-level fit per
/// slide).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchNormalization {
    pub method: StainFitMethod,
    /// `None` = Macenko's reference appearance.
    pub target: Option<NormalizationTarget>,
}

/// Where annotation files come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "path", rename_all = "kebab-case")]
pub enum AnnotationSource {
    /// One GeoJSON file, used for every slide (meant for single-slide runs).
    File(PathBuf),
    /// A directory holding `<stem>.geojson`, `<stem>.json` or
    /// `<file name>.geojson` per slide.
    Directory(PathBuf),
    /// Next to each slide, with the same names as for `Directory`.
    Sidecar,
}

/// Per-patch mask output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MaskOutput {
    /// 255 inside any selected annotation, 0 elsewhere.
    Binary,
    /// Class index (see `classes.json`; 0 = background).
    Multiclass,
}

/// Region/annotation-driven extraction from GeoJSON polygons (QuPath-style
/// `FeatureCollection` in level-0 pixel coordinates).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnnotationOptions {
    pub source: AnnotationSource,
    /// Classes to use (empty = all). Matched case-insensitively.
    pub include_classes: Vec<String>,
    /// Classes to ignore.
    pub exclude_classes: Vec<String>,
    /// Only extract patches with at least `min_coverage` of their area
    /// inside selected annotations. When `false`, every patch is kept and
    /// labelled.
    pub restrict_to_annotations: bool,
    /// In `(0, 1]`.
    pub min_coverage: f32,
    pub masks: Option<MaskOutput>,
}

impl Default for AnnotationOptions {
    fn default() -> Self {
        Self {
            source: AnnotationSource::Sidecar,
            include_classes: Vec::new(),
            exclude_classes: Vec::new(),
            restrict_to_annotations: true,
            min_coverage: 0.5,
            masks: None,
        }
    }
}

/// Image format of the written patches. Always RGB (DS-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "kebab-case")]
pub enum PatchImageFormat {
    #[default]
    Png,
    Jpeg {
        quality: u8,
    },
}

impl PatchImageFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpg",
        }
    }
}

/// What to do when the output directory already holds a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExistingOutput {
    /// Continue an interrupted run with the same parameters: tiles already
    /// recorded in the metadata are skipped. A dataset made with different
    /// parameters is an error.
    #[default]
    Resume,
    /// Start again: metadata files are replaced and tiles overwritten.
    Overwrite,
}

/// Everything beyond the basic grid ([`DatasetPatchesConfig`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetPatchesOptions {
    /// Output resolution; `None` = level 0.
    pub resolution: Option<TargetResolution>,
    pub tissue_filter: Option<TissueFilter>,
    pub stain_normalization: Option<PatchNormalization>,
    pub annotations: Option<AnnotationOptions>,
    /// Write at most this many tiles per slide.
    pub max_tiles_per_slide: Option<u64>,
    /// Sample candidate positions in a seeded random order (with
    /// `max_tiles_per_slide`). Without a seed, positions are spread evenly
    /// over the slide.
    pub seed: Option<u64>,
    pub image_format: PatchImageFormat,
    pub existing: ExistingOutput,
    /// Compute the full-file SHA-256 of every slide (all member files for
    /// multi-file formats). Cancellable; can be slow on network storage.
    pub hash_files: bool,
    /// Refuse slides whose grid has more candidate positions than this.
    pub max_candidates_per_slide: u64,
    /// Only plan the run (validate, open slides, count tiles); write nothing.
    pub dry_run: bool,
    /// Name slides by an opaque id derived from their content
    /// (`slide-<hash>`) in folders, file names and metadata, and leave out
    /// their paths and file names, input paths and annotation files, for
    /// datasets shared outside the institution. Labels and macro images are
    /// never part of a dataset.
    pub deidentify: bool,
    /// Leave out patches over cells that tissue QC marks
    /// ([`crate::qc`]): pen marks, possible folds, possibly out of focus.
    pub qc_exclusion: Option<QcExclusion>,
}

/// Which tissue-QC marks keep patches out of a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QcExclusion {
    pub pen: bool,
    pub folds: bool,
    pub blur: bool,
}

impl QcExclusion {
    /// The [`crate::qc`] mark bits to leave out.
    pub fn marks(&self) -> u8 {
        (if self.pen { crate::qc::PEN } else { 0 })
            | (if self.folds { crate::qc::FOLD } else { 0 })
            | (if self.blur { crate::qc::BLUR } else { 0 })
    }

    /// Everything QC marks.
    pub fn all() -> Self {
        Self {
            pen: true,
            folds: true,
            blur: true,
        }
    }
}

impl Default for DatasetPatchesOptions {
    fn default() -> Self {
        Self {
            resolution: None,
            tissue_filter: None,
            stain_normalization: None,
            annotations: None,
            max_tiles_per_slide: None,
            seed: None,
            image_format: PatchImageFormat::Png,
            existing: ExistingOutput::Resume,
            hash_files: true,
            max_candidates_per_slide: 10_000_000,
            dry_run: false,
            deidentify: false,
            qc_exclusion: None,
        }
    }
}

/// Allowed tile sizes.
pub const TILE_SIZE_RANGE: (u32, u32) = (8, 8192);
/// Allowed strides.
pub const STRIDE_RANGE: (u32, u32) = (1, 65_536);
/// Allowed thread counts.
pub const THREADS_RANGE: (usize, usize) = (1, 512);

/// A configuration problem (DS-4). Displayed messages are user-facing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DatasetConfigError {
    #[error("no input paths were given")]
    NoInputs,
    #[error("tile size must be between {min} and {max} pixels (got {value})")]
    TileSize { value: u32, min: u32, max: u32 },
    #[error("stride must be between {min} and {max} pixels (got {value})")]
    Stride { value: u32, min: u32, max: u32 },
    #[error("threads must be between {min} and {max} (got {value})")]
    Threads {
        value: usize,
        min: usize,
        max: usize,
    },
    #[error("white threshold must be in (0, 1] (got {0})")]
    WhiteThreshold(f32),
    #[error("minimum tissue fraction must be in [0, 1] (got {0})")]
    TissueFraction(f32),
    #[error("tissue saturation threshold must be in (0, 1) (got {0})")]
    TissueThreshold(f32),
    #[error("target resolution must be a positive number (got {0})")]
    Resolution(f64),
    #[error("max tiles per slide must be at least 1")]
    MaxTiles,
    #[error("JPEG quality must be between 1 and 100 (got {0})")]
    JpegQuality(u8),
    #[error("annotation coverage must be in (0, 1] (got {0})")]
    Coverage(f32),
    #[error("max candidates per slide must be at least 1")]
    MaxCandidates,
}

impl DatasetPatchesConfig {
    /// Checks every parameter against its documented range.
    pub fn validate(&self, options: &DatasetPatchesOptions) -> Result<(), DatasetConfigError> {
        if self.inputs.is_empty() {
            return Err(DatasetConfigError::NoInputs);
        }
        let (min, max) = TILE_SIZE_RANGE;
        if !(min..=max).contains(&self.tile_size) {
            return Err(DatasetConfigError::TileSize {
                value: self.tile_size,
                min,
                max,
            });
        }
        let (min, max) = STRIDE_RANGE;
        if !(min..=max).contains(&self.stride) {
            return Err(DatasetConfigError::Stride {
                value: self.stride,
                min,
                max,
            });
        }
        let (min, max) = THREADS_RANGE;
        if !(min..=max).contains(&self.threads) {
            return Err(DatasetConfigError::Threads {
                value: self.threads,
                min,
                max,
            });
        }
        if let Some(t) = self.white_threshold
            && !(t.is_finite() && t > 0.0 && t <= 1.0)
        {
            return Err(DatasetConfigError::WhiteThreshold(t));
        }
        if let Some(filter) = &options.tissue_filter {
            if !(filter.min_fraction.is_finite() && (0.0..=1.0).contains(&filter.min_fraction)) {
                return Err(DatasetConfigError::TissueFraction(filter.min_fraction));
            }
            if let TissueMethod::Saturation { threshold } = filter.method
                && !(threshold.is_finite() && threshold > 0.0 && threshold < 1.0)
            {
                return Err(DatasetConfigError::TissueThreshold(threshold));
            }
        }
        if let Some(resolution) = options.resolution {
            let value = match resolution {
                TargetResolution::Mpp(v) | TargetResolution::Magnification(v) => v,
            };
            if !(value.is_finite() && value > 0.0) {
                return Err(DatasetConfigError::Resolution(value));
            }
        }
        if options.max_tiles_per_slide == Some(0) {
            return Err(DatasetConfigError::MaxTiles);
        }
        if let PatchImageFormat::Jpeg { quality } = options.image_format
            && !(1..=100).contains(&quality)
        {
            return Err(DatasetConfigError::JpegQuality(quality));
        }
        if let Some(annotations) = &options.annotations
            && !(annotations.min_coverage.is_finite()
                && annotations.min_coverage > 0.0
                && annotations.min_coverage <= 1.0)
        {
            return Err(DatasetConfigError::Coverage(annotations.min_coverage));
        }
        if options.max_candidates_per_slide == 0 {
            return Err(DatasetConfigError::MaxCandidates);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DatasetPatchesConfig {
        DatasetPatchesConfig {
            inputs: vec![PathBuf::from("a.svs")],
            output_dir: PathBuf::from("out"),
            tile_size: 512,
            stride: 512,
            metadata_format: None,
            threads: 4,
            white_threshold: Some(0.8),
        }
    }

    #[test]
    fn defaults_validate() {
        assert_eq!(config().validate(&DatasetPatchesOptions::default()), Ok(()));
    }

    #[test]
    fn out_of_range_parameters_are_rejected() {
        let options = DatasetPatchesOptions::default();
        let check = |f: &dyn Fn(&mut DatasetPatchesConfig)| {
            let mut c = config();
            f(&mut c);
            c.validate(&options)
        };
        assert!(matches!(
            check(&|c| c.tile_size = 1),
            Err(DatasetConfigError::TileSize { .. })
        ));
        assert!(matches!(
            check(&|c| c.tile_size = 100_000),
            Err(DatasetConfigError::TileSize { .. })
        ));
        assert!(matches!(
            check(&|c| c.stride = 0),
            Err(DatasetConfigError::Stride { .. })
        ));
        assert!(matches!(
            check(&|c| c.threads = 0),
            Err(DatasetConfigError::Threads { .. })
        ));
        assert!(matches!(
            check(&|c| c.threads = 100_000),
            Err(DatasetConfigError::Threads { .. })
        ));
        for bad in [0.0, -0.5, 1.5, f32::NAN] {
            assert!(
                matches!(
                    check(&|c| c.white_threshold = Some(bad)),
                    Err(DatasetConfigError::WhiteThreshold(_))
                ),
                "{bad}"
            );
        }
        assert_eq!(check(&|c| c.white_threshold = Some(1.0)), Ok(()));
        assert!(matches!(
            check(&|c| c.inputs.clear()),
            Err(DatasetConfigError::NoInputs)
        ));
    }

    #[test]
    fn option_ranges_are_checked() {
        let c = config();
        let bad = |o: DatasetPatchesOptions| c.validate(&o).is_err();
        assert!(bad(DatasetPatchesOptions {
            resolution: Some(TargetResolution::Mpp(0.0)),
            ..Default::default()
        }));
        assert!(bad(DatasetPatchesOptions {
            max_tiles_per_slide: Some(0),
            ..Default::default()
        }));
        assert!(bad(DatasetPatchesOptions {
            image_format: PatchImageFormat::Jpeg { quality: 0 },
            ..Default::default()
        }));
        assert!(bad(DatasetPatchesOptions {
            tissue_filter: Some(TissueFilter {
                min_fraction: 1.5,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert!(bad(DatasetPatchesOptions {
            annotations: Some(AnnotationOptions {
                min_coverage: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        }));
    }

    #[test]
    fn options_round_trip_through_json() {
        let options = DatasetPatchesOptions {
            resolution: Some(TargetResolution::Magnification(20.0)),
            tissue_filter: Some(TissueFilter::default()),
            max_tiles_per_slide: Some(100),
            seed: Some(7),
            ..Default::default()
        };
        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(
            serde_json::from_str::<DatasetPatchesOptions>(&json).unwrap(),
            options
        );
    }
}
