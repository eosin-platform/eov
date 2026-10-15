//! The `dataset patches` pipeline: discovery → planning → per-slide
//! extraction → incremental metadata.
//!
//! * **Isolation (DS-3).** A slide that cannot be opened, hashed or read
//!   is recorded as skipped/failed and the run continues; unreadable tiles
//!   are retried once with a fresh handle, counted and reported. Only
//!   output write errors (e.g. a full disk) stop a run.
//! * **Incremental, resumable output (DS-3).** Tile records are appended
//!   to `metadata.csv` / `metadata.jsonl` after every chunk, slide records
//!   to `slides.csv` / `slides.jsonl`, and `dataset.json` records the
//!   parameters and status (also when a run stops with an error). A re-run
//!   with the same parameters repairs what a crash left behind, skips the
//!   slides an earlier run completed and the tiles it recorded, and ends
//!   with one record per slide; different parameters are refused.
//! * **Accurate progress (DS-6).** All slides are planned before
//!   extraction, so the total never grows; written, existing, skipped and
//!   failed tiles are counted separately.
//! * **Determinism.** Grids, sampling orders and metadata order depend only
//!   on the inputs and parameters, never on thread timing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use rayon::prelude::*;
use tracing::warn;

use super::annotations::AnnotationSet;
use super::config::{
    AnnotationSource, DatasetConfigError, DatasetPatchesConfig, DatasetPatchesOptions,
    ExistingOutput, MaskOutput, TargetResolution,
};
use super::discovery::{SkippedInput, SlideInput, discover_slides};
use super::grid::PatchGrid;
use super::manifest::{
    DatasetManifest, ExtractionParameters, MANIFEST_FILE, MANIFEST_FORMAT_VERSION, RunStatus,
    RunTotals, now,
};
use super::metadata::{self, MetadataWriter, SlideRecord, SlideStatus, TileRecord};
use super::output;
use super::resample::resample_rgba;
use super::sampling;
use crate::WsiFile;
use crate::stain::{
    FitChecks, NormalizationTarget, StainEstimateOptions, StainTransform, StainTransformSpec,
    estimate_slide_stain,
};
use crate::tissue::{self, TissueMask, TissueMethod};

// ─── Progress and control ────────────────────────────────────────────────────

/// What a run is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetPhase {
    Discovering,
    Planning,
    Hashing,
    Extracting,
    Finishing,
    Done,
}

impl DatasetPhase {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Discovering,
            1 => Self::Planning,
            2 => Self::Hashing,
            3 => Self::Extracting,
            4 => Self::Finishing,
            _ => Self::Done,
        }
    }
}

/// A progress snapshot. `tiles_done` never decreases and never exceeds
/// `tiles_total`, which is fixed once planning ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetPatchesProgress {
    pub phase: DatasetPhase,
    /// 1-based index of the slide being processed (0 before the first).
    pub current_slide: u64,
    pub total_slides: u64,
    pub tiles_done: u64,
    pub tiles_total: u64,
    pub tiles_written: u64,
    /// Already present from an earlier, interrupted run.
    pub tiles_existing: u64,
    /// Rejected by the white, tissue or annotation filters.
    pub tiles_skipped: u64,
    pub tiles_failed: u64,
}

impl DatasetPatchesProgress {
    pub fn fraction(&self) -> f32 {
        if self.tiles_total == 0 {
            0.0
        } else {
            (self.tiles_done as f64 / self.tiles_total as f64).clamp(0.0, 1.0) as f32
        }
    }
}

/// Cancellation and progress shared between a run and its UI.
#[derive(Debug, Default)]
pub struct DatasetControl {
    cancel: AtomicBool,
    phase: AtomicU8,
    current_slide: AtomicU64,
    total_slides: AtomicU64,
    tiles_done: AtomicU64,
    tiles_total: AtomicU64,
    written: AtomicU64,
    existing: AtomicU64,
    skipped: AtomicU64,
    failed: AtomicU64,
}

impl DatasetControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation; the run stops after the current chunk,
    /// keeping everything written so far consistent.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn progress(&self) -> DatasetPatchesProgress {
        DatasetPatchesProgress {
            phase: DatasetPhase::from_u8(self.phase.load(Ordering::Relaxed)),
            current_slide: self.current_slide.load(Ordering::Relaxed),
            total_slides: self.total_slides.load(Ordering::Relaxed),
            tiles_done: self.tiles_done.load(Ordering::Relaxed),
            tiles_total: self.tiles_total.load(Ordering::Relaxed),
            tiles_written: self.written.load(Ordering::Relaxed),
            tiles_existing: self.existing.load(Ordering::Relaxed),
            tiles_skipped: self.skipped.load(Ordering::Relaxed),
            tiles_failed: self.failed.load(Ordering::Relaxed),
        }
    }

    fn set_phase(&self, phase: DatasetPhase) {
        self.phase.store(phase as u8, Ordering::Relaxed);
    }

    fn done(&self, n: u64) {
        if n > 0 {
            self.tiles_done.fetch_add(n, Ordering::Relaxed);
        }
    }
}

// ─── Reports and errors ──────────────────────────────────────────────────────

/// Why a slide was not (fully) processed.
#[derive(Debug, Clone, PartialEq)]
pub enum SlideSkipReason {
    /// The slide could not be opened.
    OpenError(String),
    /// The slide is smaller than one patch.
    TooSmall { width: u64, height: u64 },
    /// A target resolution was requested but the slide has no pixel size
    /// (or objective power, for magnifications).
    NoPixelSize,
    /// The target resolution is finer than the scan.
    ResolutionTooFine { slide_mpp: f64, target_mpp: f64 },
    /// The grid has more candidate positions than allowed (DS-4).
    TooManyCandidates { candidates: u64, limit: u64 },
    /// Annotation-restricted extraction, but no annotations for this slide.
    NoAnnotations,
    /// Processing failed (hashing, stain fit or too many unreadable tiles).
    Failed(String),
    /// The run was cancelled during this slide.
    Cancelled,
}

impl std::fmt::Display for SlideSkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenError(e) => write!(f, "cannot open: {e}"),
            Self::TooSmall { width, height } => {
                write!(f, "{width}×{height} is smaller than one patch")
            }
            Self::NoPixelSize => write!(f, "no pixel size (MPP) to reach the target resolution"),
            Self::ResolutionTooFine {
                slide_mpp,
                target_mpp,
            } => write!(
                f,
                "target {target_mpp:.4} µm/px is finer than the scan ({slide_mpp:.4} µm/px)"
            ),
            Self::TooManyCandidates { candidates, limit } => write!(
                f,
                "{candidates} candidate patches exceed the limit of {limit}; increase the stride \
                 or set a per-slide maximum"
            ),
            Self::NoAnnotations => write!(f, "no annotations for this slide"),
            Self::Failed(e) => write!(f, "failed: {e}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Per-slide statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SlideReport {
    pub path: PathBuf,
    /// Unique key used in output paths.
    pub slide_id: String,
    pub status: SlideStatus,
    pub tiles_written: u64,
    pub tiles_existing: u64,
    pub tiles_skipped_white: u64,
    pub tiles_skipped_tissue: u64,
    pub tiles_skipped_annotation: u64,
    /// Over tissue-QC marks the run excludes.
    pub tiles_skipped_qc: u64,
    pub tiles_failed: u64,
    /// Set when the slide was skipped or failed.
    pub skipped: Option<SlideSkipReason>,
    /// The first few tile errors.
    pub errors: Vec<String>,
}

/// Summary of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetPatchesReport {
    pub input_count: usize,
    pub discovered_slides: usize,
    /// Slides completely or partially processed.
    pub processed_slides: usize,
    /// Slides skipped or failed.
    pub skipped_slides: usize,
    pub failed_slides: usize,
    /// Tiles written in this run.
    pub total_tiles: u64,
    pub total_tiles_skipped_white: u64,
    pub total_tiles_skipped_tissue: u64,
    pub total_tiles_skipped_annotation: u64,
    pub total_tiles_skipped_qc: u64,
    pub total_tiles_existing: u64,
    pub total_tiles_failed: u64,
    /// Planned tiles (the progress total).
    pub planned_tiles: u64,
    pub metadata_paths: Vec<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub slides: Vec<SlideReport>,
    pub input_errors: Vec<(PathBuf, String)>,
    /// Files deliberately not used (non-slide TIFFs, extra DICOM instances).
    pub skipped_inputs: Vec<SkippedInput>,
    pub cancelled: bool,
    pub dry_run: bool,
}

impl DatasetPatchesReport {
    /// `true` when anything failed: unopenable or failed slides, unreadable
    /// tiles, or unresolvable inputs.
    pub fn has_errors(&self) -> bool {
        self.failed_slides > 0
            || self.total_tiles_failed > 0
            || !self.input_errors.is_empty()
            || self
                .slides
                .iter()
                .any(|s| matches!(s.skipped, Some(SlideSkipReason::OpenError(_))))
    }

    /// The run status recorded in the manifest.
    pub fn status(&self) -> RunStatus {
        if self.cancelled {
            RunStatus::Cancelled
        } else if self.has_errors() {
            RunStatus::Partial
        } else {
            RunStatus::Complete
        }
    }
}

/// An error that stops the whole run.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error(transparent)]
    Config(#[from] DatasetConfigError),
    #[error("no slide files were found in the inputs")]
    NoSlides {
        input_errors: Vec<(PathBuf, String)>,
    },
    #[error("{0}")]
    OutputConflict(String),
    #[error("cannot write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("annotations: {0}")]
    Annotations(String),
}

impl From<DatasetError> for crate::Error {
    fn from(err: DatasetError) -> Self {
        crate::Error::Io(std::io::Error::other(err.to_string()))
    }
}

fn write_err(path: &Path) -> impl FnOnce(std::io::Error) -> DatasetError + '_ {
    move |source| DatasetError::Write {
        path: path.to_path_buf(),
        source,
    }
}

// ─── Planning ────────────────────────────────────────────────────────────────

/// How patches of one slide are read.
#[derive(Debug, Clone, PartialEq)]
pub struct SlideGeometry {
    pub width: u64,
    pub height: u64,
    pub level: u32,
    pub level_downsample: f64,
    /// Level-0 pixels per output pixel, per axis.
    pub downsample: (f64, f64),
    /// Size read at `level` before resampling to `tile_size`.
    pub read_size: (u32, u32),
    pub grid: PatchGrid,
    pub output_mpp: (Option<f64>, Option<f64>),
}

/// A slide as planned.
#[derive(Debug, Clone)]
pub struct PlannedSlide {
    pub input: SlideInput,
    pub key: String,
    pub geometry: Result<SlideGeometry, SlideSkipReason>,
    pub props: Option<crate::WsiProperties>,
    /// Tiles this slide contributes to the progress total.
    pub planned_tiles: u64,
}

/// The result of planning a run (also what a dry run returns).
#[derive(Debug, Clone)]
pub struct DatasetPlan {
    pub slides: Vec<PlannedSlide>,
    pub input_errors: Vec<(PathBuf, String)>,
    pub skipped_inputs: Vec<SkippedInput>,
    pub planned_tiles: u64,
    /// Annotation classes used for multiclass masks, in index order.
    pub classes: Vec<String>,
}

fn valid(v: Option<f64>) -> Option<f64> {
    v.filter(|m| m.is_finite() && *m > 0.0)
}

/// Snap factors within 2 % of a value to it (0.2501 µm/px is 0.25 µm/px).
fn snap(value: f64, to: f64) -> f64 {
    if ((value - to) / to).abs() < 0.02 {
        to
    } else {
        value
    }
}

fn geometry(
    wsi: &WsiFile,
    config: &DatasetPatchesConfig,
    options: &DatasetPatchesOptions,
) -> Result<SlideGeometry, SlideSkipReason> {
    let props = wsi.properties();
    let (mpp_x, mpp_y) = (valid(props.mpp_x), valid(props.mpp_y));
    let (ds_x, ds_y) = match options.resolution {
        None => (1.0, 1.0),
        Some(TargetResolution::Mpp(target)) => {
            let (mx, my) = match (mpp_x, mpp_y) {
                (Some(x), Some(y)) => (x, y),
                (Some(m), None) | (None, Some(m)) => (m, m),
                (None, None) => return Err(SlideSkipReason::NoPixelSize),
            };
            (target / mx, target / my)
        }
        Some(TargetResolution::Magnification(mag)) => {
            if let Some(objective) = props.objective_power.filter(|o| *o > 0.0) {
                let ds = objective / mag;
                (ds, ds)
            } else {
                // ≈ 10 µm/px at 1×.
                let target = 10.0 / mag;
                match (mpp_x, mpp_y) {
                    (Some(x), Some(y)) => (target / x, target / y),
                    (Some(m), None) | (None, Some(m)) => (target / m, target / m),
                    (None, None) => return Err(SlideSkipReason::NoPixelSize),
                }
            }
        }
    };
    let (ds_x, ds_y) = (snap(ds_x, 1.0), snap(ds_y, 1.0));
    if ds_x < 0.95 || ds_y < 0.95 {
        let slide_mpp = mpp_x.or(mpp_y).unwrap_or(0.0);
        return Err(SlideSkipReason::ResolutionTooFine {
            slide_mpp,
            target_mpp: slide_mpp * ds_x.min(ds_y),
        });
    }
    let (ds_x, ds_y) = (ds_x.max(1.0), ds_y.max(1.0));
    // Coarsest level that is not coarser than the target (never upsample).
    let target = ds_x.min(ds_y);
    let mut level = 0;
    let mut level_ds = 1.0;
    for l in 0..wsi.level_count() {
        if let Some(info) = wsi.level(l)
            && info.downsample <= target * 1.02
            && info.downsample >= level_ds
        {
            level = l;
            level_ds = info.downsample;
        }
    }
    let ts = config.tile_size as f64;
    let read_size = (
        ((ts * ds_x / level_ds).round() as u32).max(1),
        ((ts * ds_y / level_ds).round() as u32).max(1),
    );
    let grid = PatchGrid::new(
        props.width,
        props.height,
        ts * ds_x,
        ts * ds_y,
        config.stride as f64 * ds_x,
        config.stride as f64 * ds_y,
    );
    if grid.is_empty() {
        return Err(SlideSkipReason::TooSmall {
            width: props.width,
            height: props.height,
        });
    }
    if grid.len() > options.max_candidates_per_slide {
        return Err(SlideSkipReason::TooManyCandidates {
            candidates: grid.len(),
            limit: options.max_candidates_per_slide,
        });
    }
    Ok(SlideGeometry {
        width: props.width,
        height: props.height,
        level,
        level_downsample: level_ds,
        downsample: (ds_x, ds_y),
        read_size,
        grid,
        output_mpp: (mpp_x.map(|m| m * ds_x), mpp_y.map(|m| m * ds_y)),
    })
}

fn annotation_path(source: &AnnotationSource, slide: &Path) -> Option<PathBuf> {
    let dir = match source {
        AnnotationSource::File(path) => return Some(path.clone()),
        AnnotationSource::Directory(dir) => dir.clone(),
        AnnotationSource::Sidecar => slide.parent()?.to_path_buf(),
    };
    let stem = slide.file_stem()?.to_string_lossy().into_owned();
    let name = slide.file_name()?.to_string_lossy().into_owned();
    [
        format!("{stem}.geojson"),
        format!("{stem}.json"),
        format!("{name}.geojson"),
    ]
    .into_iter()
    .map(|f| dir.join(f))
    .find(|p| p.is_file())
}

fn load_annotations(
    options: &DatasetPatchesOptions,
    slide: &Path,
) -> Result<Option<(PathBuf, AnnotationSet)>, DatasetError> {
    let Some(ann) = &options.annotations else {
        return Ok(None);
    };
    let Some(path) = annotation_path(&ann.source, slide) else {
        return Ok(None);
    };
    let mut set =
        AnnotationSet::load(&path).map_err(|e| DatasetError::Annotations(e.to_string()))?;
    set.retain_classes(&ann.include_classes, &ann.exclude_classes);
    Ok(Some((path, set)))
}

/// Discovers, opens and plans every slide without writing anything.
pub fn plan_dataset(
    config: &DatasetPatchesConfig,
    options: &DatasetPatchesOptions,
) -> Result<DatasetPlan, DatasetError> {
    config.validate(options)?;
    let discovery = discover_slides(&config.inputs);
    if discovery.slides.is_empty() {
        return Err(DatasetError::NoSlides {
            input_errors: discovery.errors,
        });
    }
    let keys = output::unique_slide_keys_for(&discovery.slides);
    let mut used_keys = std::collections::HashSet::new();
    let mut slides = Vec::with_capacity(discovery.slides.len());
    let mut classes = std::collections::BTreeSet::new();
    let mut planned_tiles = 0u64;
    for (input, key) in discovery.slides.into_iter().zip(keys) {
        let opened = WsiFile::open(&input.path);
        let key = if options.deidentify {
            output::deidentified_key(opened.as_ref().ok(), &input.path, &mut used_keys)
        } else {
            key
        };
        let (geometry, props) = match opened {
            Ok(wsi) => (
                geometry(&wsi, config, options),
                Some(wsi.properties().clone()),
            ),
            Err(e) => (Err(SlideSkipReason::OpenError(e.to_string())), None),
        };
        let mut geometry = geometry;
        if geometry.is_ok()
            && let Some(ann) = &options.annotations
        {
            match load_annotations(options, &input.path)? {
                Some((_, set)) => classes.extend(set.class_names()),
                None if ann.restrict_to_annotations => {
                    geometry = Err(SlideSkipReason::NoAnnotations)
                }
                None => {}
            }
        }
        let planned = match &geometry {
            Ok(g) => options
                .max_tiles_per_slide
                .map_or(g.grid.len(), |k| k.min(g.grid.len())),
            Err(_) => 0,
        };
        planned_tiles += planned;
        slides.push(PlannedSlide {
            input,
            key,
            geometry,
            props,
            planned_tiles: planned,
        });
    }
    Ok(DatasetPlan {
        slides,
        input_errors: discovery.errors,
        skipped_inputs: discovery.skipped,
        planned_tiles,
        classes: classes.into_iter().collect(),
    })
}

// ─── Running ─────────────────────────────────────────────────────────────────

/// Shared state of one run.
struct Run<'a> {
    config: &'a DatasetPatchesConfig,
    options: &'a DatasetPatchesOptions,
    control: &'a DatasetControl,
    /// Tiles recorded by earlier runs into the same output (when resuming).
    recorded: HashSet<String>,
    /// The latest record of each slide from earlier runs (when resuming).
    previous: HashMap<String, SlideRecord>,
    classes: Vec<String>,
    pool: rayon::ThreadPool,
}

/// What happened to one candidate position.
enum Outcome {
    Written(Box<Accepted>),
    Existing,
    SkippedWhite,
    SkippedTissue,
    SkippedAnnotation,
    SkippedQc,
    Failed(String),
}

struct Accepted {
    record: TileRecord,
    image: Vec<u8>,
    mask: Option<Vec<u8>>,
}

/// Per-slide context for the parallel stage.
struct SlideJob<'a> {
    planned: &'a PlannedSlide,
    geometry: &'a SlideGeometry,
    props: &'a crate::WsiProperties,
    fingerprint: String,
    sha256: String,
    tissue: Option<(Option<TissueMask>, f32)>,
    stain: Option<(StainTransform, String)>,
    annotations: Option<AnnotationSet>,
    /// Tissue-QC marks and the bits the run leaves out.
    qc: Option<(crate::qc::QcResult, u8)>,
    handles: Vec<Mutex<Option<WsiFile>>>,
}

/// How many sampled candidates to process next under a per-slide limit of
/// `limit` tiles, after `tried` candidates gave `accepted` tiles: about as
/// many as the acceptance rate so far says are still needed (plus a
/// margin), at least one per thread. Whole chunks would mostly be thrown
/// away when the limit is small.
fn sampled_chunk_size(limit: u64, accepted: u64, tried: u64, threads: u64) -> u64 {
    let remaining = limit.saturating_sub(accepted);
    let rate = if tried == 0 {
        1.0
    } else {
        (accepted as f64 / tried as f64).max(0.01)
    };
    let needed = (remaining as f64 / rate * 1.25).ceil();
    (needed.min(u32::MAX as f64) as u64).max(threads)
}

/// The fast, sampled slide fingerprint as hex (empty if it cannot be
/// computed).
fn fingerprint_hex(path: &Path) -> String {
    crate::file_id::compute_fingerprint(path)
        .map(|d| crate::file_id::hex_digest(&d))
        .unwrap_or_default()
}

fn read_with_retry(job: &SlideJob, x: u64, y: u64) -> crate::Result<Vec<u8>> {
    let slot_index = rayon::current_thread_index().unwrap_or(0) % job.handles.len().max(1);
    let mut slot = job.handles[slot_index]
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let (rw, rh) = job.geometry.read_size;
    for attempt in 0..2 {
        if slot.is_none() {
            *slot = Some(WsiFile::open(&job.planned.input.path)?);
        }
        let wsi = slot.as_ref().expect("opened above");
        match wsi.read_region(x as i64, y as i64, job.geometry.level, rw, rh) {
            Ok(data) => return Ok(data),
            Err(err) if attempt == 0 => {
                // OpenSlide errors poison the handle (WSI-1): reopen once.
                warn!(
                    "tile ({x}, {y}) of {}: {err}; retrying",
                    job.planned.input.path.display()
                );
                *slot = None;
            }
            Err(err) => return Err(err),
        }
    }
    unreachable!("the loop returns on the second attempt")
}

impl Run<'_> {
    /// A slide's path as written to the metadata ("" when de-identified).
    fn public_path(&self, path: &Path) -> String {
        if self.options.deidentify {
            String::new()
        } else {
            path.display().to_string()
        }
    }

    /// An error message as written to the metadata: when de-identified,
    /// the slide's path and file name are replaced by its id.
    fn public_message(&self, message: String, planned: &PlannedSlide) -> String {
        if !self.options.deidentify {
            return message;
        }
        let path = &planned.input.path;
        let mut message = message.replace(&path.display().to_string(), &planned.key);
        for part in [path.file_name(), path.file_stem()].into_iter().flatten() {
            let part = part.to_string_lossy();
            if !part.is_empty() {
                message = message.replace(&*part, &planned.key);
            }
        }
        message
    }

    fn tile_path(&self, key: &str, x: u64, y: u64) -> String {
        let ext = self.options.image_format.extension();
        let name = output::tile_filename_with_extension(key, x, y, self.config.tile_size, ext);
        format!("slides/{key}/{name}")
    }

    fn tile_paths(&self, key: &str, x: u64, y: u64) -> (String, String) {
        (
            self.tile_path(key, x, y),
            output::mask_relative_path(key, x, y, self.config.tile_size),
        )
    }

    /// When resuming: the report for a slide that an earlier run completed
    /// and whose recorded tiles are all still on disk. Such a slide needs no
    /// work at all — no hashing, tissue mask, stain fit or pixel reads (the
    /// positions an earlier run rejected are not recorded, so visiting the
    /// grid again would re-read them all).
    fn already_complete(
        &self,
        planned: &PlannedSlide,
        geometry: &SlideGeometry,
        fingerprint: &str,
    ) -> Option<SlideReport> {
        let previous = self.previous.get(&planned.key)?;
        let path = &planned.input.path;
        if previous.status != SlideStatus::Complete || previous.slide_path != self.public_path(path)
        {
            return None;
        }
        if previous.fingerprint.as_deref() != Some(fingerprint) {
            warn!(
                "{} changed since an earlier run extracted it; checking it again (recorded \
                 patches are kept)",
                path.display()
            );
            return None;
        }
        let mut existing = 0u64;
        for index in 0..geometry.grid.len() {
            let coord = geometry.grid.get(index);
            let tile_path = self.tile_path(&planned.key, coord.x, coord.y);
            if self.recorded.contains(&tile_path) {
                if !self.config.output_dir.join(&tile_path).is_file() {
                    // Process the slide again; that rewrites missing tiles.
                    return None;
                }
                existing += 1;
            }
        }
        if existing != previous.written + previous.existing {
            // The tile metadata does not match the slide record (edited or
            // partly lost): process the slide again.
            return None;
        }
        self.control.existing.fetch_add(existing, Ordering::Relaxed);
        self.control.done(planned.planned_tiles);
        Some(SlideReport {
            path: path.clone(),
            slide_id: planned.key.clone(),
            status: SlideStatus::Complete,
            tiles_written: 0,
            tiles_existing: existing,
            tiles_skipped_white: 0,
            tiles_skipped_tissue: 0,
            tiles_skipped_annotation: 0,
            tiles_skipped_qc: 0,
            tiles_failed: 0,
            skipped: None,
            errors: Vec::new(),
        })
    }

    /// Cheap filters that need no pixels (low-resolution tissue mask,
    /// annotation coverage).
    fn prefilter(&self, job: &SlideJob, x: u64, y: u64) -> Option<Outcome> {
        let (fx, fy) = (job.geometry.grid.footprint_x, job.geometry.grid.footprint_y);
        if let Some((qc, marks)) = &job.qc
            && qc.rect_marked(x as f64, y as f64, fx, fy, *marks)
        {
            return Some(Outcome::SkippedQc);
        }
        if let (Some((Some(mask), _)), Some(filter)) = (&job.tissue, &self.options.tissue_filter) {
            // Lenient: the mask is coarse, the patch decides.
            if mask.fraction(x as f64, y as f64, fx, fy) < filter.min_fraction * 0.5 {
                return Some(Outcome::SkippedTissue);
            }
        }
        if let (Some(set), Some(ann)) = (&job.annotations, &self.options.annotations)
            && ann.restrict_to_annotations
        {
            if !set.may_intersect(x as f64, y as f64, fx, fy) {
                return Some(Outcome::SkippedAnnotation);
            }
            if set.coverage(x as f64, y as f64, fx, fy, 16).any < ann.min_coverage {
                return Some(Outcome::SkippedAnnotation);
            }
        }
        None
    }

    fn process(&self, job: &SlideJob, index: u64, prefiltered: bool) -> Outcome {
        let coord = job.geometry.grid.get(index);
        let (x, y) = (coord.x, coord.y);
        let (tile_path, mask_path) = self.tile_paths(&job.planned.key, x, y);
        if self.recorded.contains(&tile_path) && self.config.output_dir.join(&tile_path).is_file() {
            return Outcome::Existing;
        }
        if !prefiltered && let Some(outcome) = self.prefilter(job, x, y) {
            return outcome;
        }
        let mut rgba = match read_with_retry(job, x, y) {
            Ok(d) => d,
            Err(e) => return Outcome::Failed(format!("({x}, {y}): {e}")),
        };
        output::composite_onto_white(&mut rgba);
        let ts = self.config.tile_size;
        let (rw, rh) = job.geometry.read_size;
        if (rw, rh) != (ts, ts) {
            rgba = resample_rgba(&rgba, rw, rh, ts, ts);
        }
        let white_fraction = output::white_fraction(&rgba);
        if let Some(t) = self.config.white_threshold
            && white_fraction >= t
        {
            return Outcome::SkippedWhite;
        }
        let tissue_fraction = match (&job.tissue, &self.options.tissue_filter) {
            (Some((_, threshold)), Some(filter)) => {
                let f = tissue::tissue_fraction_rgba(&rgba, *threshold);
                if f < filter.min_fraction {
                    return Outcome::SkippedTissue;
                }
                Some(f)
            }
            _ => None,
        };
        if let Some((transform, _)) = &job.stain {
            transform.apply_rgba(&mut rgba);
        }
        let (fx, fy) = (job.geometry.grid.footprint_x, job.geometry.grid.footprint_y);
        let (mut label, mut label_coverage, mut mask) = (None, None, None);
        if let (Some(set), Some(ann)) = (&job.annotations, &self.options.annotations) {
            let coverage = set.coverage(x as f64, y as f64, fx, fy, 32);
            if let Some((class, fraction)) = coverage.majority() {
                label = Some(class.to_string());
                label_coverage = Some(fraction);
            }
            if let Some(kind) = ann.masks {
                let classes = &self.classes;
                let value = |class: &str| -> u8 {
                    match kind {
                        MaskOutput::Binary => 255,
                        MaskOutput::Multiclass => classes
                            .iter()
                            .position(|c| c == class)
                            .map_or(0, |i| (i + 1).min(255) as u8),
                    }
                };
                let raster = set.rasterize(x as f64, y as f64, fx, fy, ts, ts, &value);
                match output::encode_mask(&raster, ts, ts) {
                    Ok(bytes) => mask = Some(bytes),
                    Err(e) => return Outcome::Failed(format!("mask ({x}, {y}): {e}")),
                }
            }
        }
        let image = match output::encode_tile(&rgba, ts, ts, self.options.image_format) {
            Ok(bytes) => bytes,
            Err(e) => return Outcome::Failed(format!("encode ({x}, {y}): {e}")),
        };
        let props = job.props;
        Outcome::Written(Box::new(Accepted {
            record: TileRecord {
                slide_path: self.public_path(&job.planned.input.path),
                fingerprint: job.fingerprint.clone(),
                file_sha256: job.sha256.clone(),
                slide_stem: if self.options.deidentify {
                    job.planned.key.clone()
                } else {
                    output::slide_base_name(&job.planned.input)
                },
                tile_path,
                x,
                y,
                tile_size: ts,
                width: ts,
                height: ts,
                slide_width: props.width,
                slide_height: props.height,
                level: job.geometry.level,
                mpp_x: props.mpp_x,
                mpp_y: props.mpp_y,
                slide_id: job.planned.key.clone(),
                level0_width: fx,
                level0_height: fy,
                downsample: job.geometry.downsample.0,
                output_mpp_x: job.geometry.output_mpp.0,
                output_mpp_y: job.geometry.output_mpp.1,
                tissue_fraction,
                white_fraction,
                label,
                label_coverage,
                mask_path: mask.as_ref().map(|_| mask_path),
                vendor: props.vendor.clone(),
                objective_power: props.objective_power,
                stain_normalization: job.stain.as_ref().map(|(_, s)| s.clone()),
            },
            image,
            mask,
        }))
    }
}

#[derive(Default)]
struct SlideCounts {
    written: u64,
    existing: u64,
    white: u64,
    tissue: u64,
    annotation: u64,
    qc: u64,
    failed: u64,
    errors: Vec<String>,
}

impl Run<'_> {
    /// Writes accepted tiles (in parallel) and appends their records.
    fn commit(
        &self,
        accepted: Vec<Accepted>,
        writer: &mut MetadataWriter,
        dirs: &mut HashSet<PathBuf>,
    ) -> Result<(), DatasetError> {
        let root = &self.config.output_dir;
        for a in &accepted {
            for rel in std::iter::once(&a.record.tile_path).chain(a.record.mask_path.as_ref()) {
                let path = root.join(rel);
                if let Some(parent) = path.parent()
                    && dirs.insert(parent.to_path_buf())
                {
                    std::fs::create_dir_all(parent).map_err(write_err(parent))?;
                }
            }
        }
        self.pool.install(|| {
            accepted
                .par_iter()
                .try_for_each(|a| -> Result<(), DatasetError> {
                    let path = root.join(&a.record.tile_path);
                    output::write_atomic(&path, &a.image).map_err(write_err(&path))?;
                    if let (Some(rel), Some(bytes)) = (&a.record.mask_path, &a.mask) {
                        let path = root.join(rel);
                        output::write_atomic(&path, bytes).map_err(write_err(&path))?;
                    }
                    Ok(())
                })
        })?;
        let records: Vec<TileRecord> = accepted.into_iter().map(|a| a.record).collect();
        let jsonl = root.join(metadata::JSONL_FILE);
        writer.append_tiles(&records).map_err(write_err(&jsonl))
    }

    /// Processes one planned slide. Returns its report and the record to
    /// append (`None` when the record of an earlier run stands).
    fn slide(
        &self,
        planned: &PlannedSlide,
        writer: &mut MetadataWriter,
    ) -> Result<(SlideReport, Option<SlideRecord>), DatasetError> {
        if let Ok(geometry) = &planned.geometry
            && self.previous.contains_key(&planned.key)
        {
            self.control.set_phase(DatasetPhase::Extracting);
            let fingerprint = fingerprint_hex(&planned.input.path);
            if let Some(report) = self.already_complete(planned, geometry, &fingerprint) {
                return Ok((report, None));
            }
        }
        let (report, record) = self.extract(planned, writer)?;
        Ok((report, Some(record)))
    }

    /// Extracts one planned slide. Returns its report and record.
    fn extract(
        &self,
        planned: &PlannedSlide,
        writer: &mut MetadataWriter,
    ) -> Result<(SlideReport, SlideRecord), DatasetError> {
        let Some(props) = &planned.props else {
            let reason = planned
                .geometry
                .clone()
                .err()
                .unwrap_or_else(|| SlideSkipReason::Failed("the slide was not opened".into()));
            return Ok(self.skipped(planned, reason));
        };
        let geometry = match &planned.geometry {
            Ok(g) => g,
            Err(reason) => return Ok(self.skipped(planned, reason.clone())),
        };
        let path = &planned.input.path;
        let fail = |reason: String| Ok(self.skipped(planned, SlideSkipReason::Failed(reason)));

        // Identity.
        self.control.set_phase(DatasetPhase::Hashing);
        let fingerprint = fingerprint_hex(path);
        let sha256 = if self.options.hash_files {
            match super::hashing::slide_sha256(&planned.input.members, &self.control.cancel) {
                Ok(Some(hash)) => hash,
                Ok(None) => return Ok(self.skipped(planned, SlideSkipReason::Cancelled)),
                Err(e) => return fail(format!("hashing: {e}")),
            }
        } else {
            String::new()
        };
        self.control.set_phase(DatasetPhase::Extracting);

        let wsi = match WsiFile::open(path) {
            Ok(w) => w,
            Err(e) => return Ok(self.skipped(planned, SlideSkipReason::OpenError(e.to_string()))),
        };
        let tissue = match &self.options.tissue_filter {
            None => None,
            Some(filter) => {
                match TissueMask::compute(&wsi, filter.method, filter.mask_max_pixels) {
                    Ok(mask) => {
                        let threshold = match (&mask, filter.method) {
                            (Some(m), _) => m.threshold(),
                            (None, TissueMethod::Saturation { threshold }) => threshold,
                            (None, TissueMethod::Otsu) => 0.08,
                        };
                        Some((mask, threshold))
                    }
                    Err(e) => return fail(format!("tissue mask: {e}")),
                }
            }
        };
        let stain = match &self.options.stain_normalization {
            None => None,
            Some(norm) => {
                let estimate =
                    match estimate_slide_stain(&wsi, norm.method, &StainEstimateOptions::default())
                    {
                        Ok(e) => e,
                        Err(e) => return fail(format!("stain fit: {e}")),
                    };
                let target = norm
                    .target
                    .clone()
                    .unwrap_or_else(NormalizationTarget::macenko_reference);
                match StainTransformSpec::normalization(&estimate, &target, &FitChecks::default())
                    .map_err(|e| e.to_string())
                    .and_then(|s| s.compile().map_err(|e| e.to_string()))
                {
                    Ok(t) => Some((t, estimate.summary())),
                    Err(e) => return fail(format!("stain normalisation: {e}")),
                }
            }
        };
        let (annotation_file, annotations) = match load_annotations(self.options, path)? {
            Some((p, set)) => (Some(p.display().to_string()), Some(set)),
            None => (None, None),
        };
        let qc = match self.options.qc_exclusion.filter(|q| q.marks() != 0) {
            None => None,
            Some(exclusion) => {
                let options = crate::qc::QcOptions {
                    sharpness: exclusion.blur,
                    ..crate::qc::QcOptions::default()
                };
                let cancelled = AtomicBool::new(false);
                match crate::qc::run(&wsi, &options, &|_| {}, &cancelled) {
                    Ok(result) => result.map(|result| (result, exclusion.marks())),
                    Err(e) => return fail(format!("tissue QC: {e}")),
                }
            }
        };
        drop(wsi);

        let job = SlideJob {
            planned,
            geometry,
            props,
            fingerprint,
            sha256,
            tissue,
            stain,
            annotations,
            qc,
            handles: (0..self.pool.current_num_threads().max(1))
                .map(|_| Mutex::new(None))
                .collect(),
        };
        let mut counts = SlideCounts::default();
        let mut dirs = HashSet::new();
        let threads = self.pool.current_num_threads().max(1) as u64;
        let chunk = (threads * 16).clamp(16, 4096);
        let error_limit = (planned.planned_tiles / 100).max(100);
        let limit = self.options.max_tiles_per_slide;
        let mut done_units = 0u64;
        let mut status = SlideStatus::Complete;
        let mut failure: Option<SlideSkipReason> = None;

        // Candidate order: all positions in raster order, or — with a
        // per-slide limit — the prefiltered positions in sampled order.
        let sampled: Option<Vec<u64>> = limit.map(|_| {
            let eligible: Vec<u64> = self.pool.install(|| {
                (0..geometry.grid.len())
                    .into_par_iter()
                    .filter(|&i| {
                        let c = geometry.grid.get(i);
                        self.prefilter(&job, c.x, c.y).is_none()
                    })
                    .collect()
            });
            match self.options.seed {
                Some(seed) => {
                    let mut order = eligible;
                    sampling::shuffle(&mut order, sampling::slide_seed(seed, &planned.key));
                    order
                }
                None => sampling::spread_order(&eligible),
            }
        });
        let total = sampled
            .as_ref()
            .map_or(geometry.grid.len(), |s| s.len() as u64);

        let mut start = 0u64;
        while start < total {
            if self.control.is_cancelled() {
                status = SlideStatus::Cancelled;
                failure = Some(SlideSkipReason::Cancelled);
                break;
            }
            let size = match limit {
                Some(k) => sampled_chunk_size(k, counts.written + counts.existing, start, threads)
                    .min(chunk),
                None => chunk,
            };
            let end = (start + size).min(total);
            let indices: Vec<u64> = match &sampled {
                Some(order) => order[start as usize..end as usize].to_vec(),
                None => (start..end).collect(),
            };
            let outcomes: Vec<Outcome> = self.pool.install(|| {
                indices
                    .par_iter()
                    .map(|&i| self.process(&job, i, sampled.is_some()))
                    .collect()
            });
            let mut accepted = Vec::new();
            for outcome in outcomes {
                let full = limit.is_some_and(|k| counts.written + counts.existing >= k);
                if full {
                    break;
                }
                let unit = match outcome {
                    Outcome::Written(a) => {
                        counts.written += 1;
                        self.control.written.fetch_add(1, Ordering::Relaxed);
                        accepted.push(*a);
                        1
                    }
                    Outcome::Existing => {
                        counts.existing += 1;
                        self.control.existing.fetch_add(1, Ordering::Relaxed);
                        1
                    }
                    Outcome::SkippedWhite
                    | Outcome::SkippedTissue
                    | Outcome::SkippedAnnotation
                    | Outcome::SkippedQc => {
                        match outcome {
                            Outcome::SkippedWhite => counts.white += 1,
                            Outcome::SkippedTissue => counts.tissue += 1,
                            Outcome::SkippedQc => counts.qc += 1,
                            _ => counts.annotation += 1,
                        }
                        self.control.skipped.fetch_add(1, Ordering::Relaxed);
                        u64::from(limit.is_none())
                    }
                    Outcome::Failed(message) => {
                        counts.failed += 1;
                        self.control.failed.fetch_add(1, Ordering::Relaxed);
                        if counts.errors.len() < 10 {
                            counts.errors.push(message);
                        }
                        u64::from(limit.is_none())
                    }
                };
                done_units += unit;
                self.control.done(unit);
            }
            self.commit(accepted, writer, &mut dirs)?;
            if counts.failed > error_limit {
                status = SlideStatus::Failed;
                failure = Some(SlideSkipReason::Failed(format!(
                    "{} tiles could not be read (first: {})",
                    counts.failed,
                    counts.errors.first().cloned().unwrap_or_default()
                )));
                break;
            }
            if limit.is_some_and(|k| counts.written + counts.existing >= k) {
                break;
            }
            start = end;
        }
        // Account for planned tiles that were not reached (limits, failure,
        // cancellation) so progress ends exactly at the total.
        self.control
            .done(planned.planned_tiles.saturating_sub(done_units));
        if status == SlideStatus::Complete && counts.failed > 0 {
            status = SlideStatus::Partial;
        }

        let record = SlideRecord {
            slide_id: planned.key.clone(),
            slide_path: self.public_path(path),
            status,
            message: failure
                .as_ref()
                .map(|f| self.public_message(f.to_string(), planned)),
            vendor: props.vendor.clone(),
            objective_power: props.objective_power,
            mpp_x: props.mpp_x,
            mpp_y: props.mpp_y,
            width: props.width,
            height: props.height,
            level_count: props.levels.len() as u32,
            read_level: Some(geometry.level),
            downsample: Some(geometry.downsample.0),
            output_mpp: geometry.output_mpp.0,
            tissue_fraction: job
                .tissue
                .as_ref()
                .and_then(|(m, _)| m.as_ref().map(TissueMask::slide_fraction)),
            candidates: geometry.grid.len(),
            written: counts.written,
            existing: counts.existing,
            skipped_white: counts.white,
            skipped_tissue: counts.tissue + counts.qc,
            skipped_annotation: counts.annotation,
            failed: counts.failed,
            fingerprint: Some(job.fingerprint.clone()),
            file_sha256: (!job.sha256.is_empty()).then(|| job.sha256.clone()),
            stain_normalization: job.stain.as_ref().map(|(_, s)| s.clone()),
            annotation_file: annotation_file.filter(|_| !self.options.deidentify),
        };
        let report = SlideReport {
            path: path.clone(),
            slide_id: planned.key.clone(),
            status,
            tiles_written: counts.written,
            tiles_existing: counts.existing,
            tiles_skipped_white: counts.white,
            tiles_skipped_tissue: counts.tissue,
            tiles_skipped_annotation: counts.annotation,
            tiles_skipped_qc: counts.qc,
            tiles_failed: counts.failed,
            skipped: failure,
            errors: counts.errors,
        };
        Ok((report, record))
    }

    fn skipped(
        &self,
        planned: &PlannedSlide,
        reason: SlideSkipReason,
    ) -> (SlideReport, SlideRecord) {
        self.control.done(planned.planned_tiles);
        let status = match reason {
            SlideSkipReason::OpenError(_) | SlideSkipReason::Failed(_) => SlideStatus::Failed,
            SlideSkipReason::Cancelled => SlideStatus::Cancelled,
            _ => SlideStatus::Skipped,
        };
        let props = planned.props.as_ref();
        let record = SlideRecord {
            slide_id: planned.key.clone(),
            slide_path: self.public_path(&planned.input.path),
            status,
            message: Some(self.public_message(reason.to_string(), planned)),
            vendor: props.and_then(|p| p.vendor.clone()),
            objective_power: props.and_then(|p| p.objective_power),
            mpp_x: props.and_then(|p| p.mpp_x),
            mpp_y: props.and_then(|p| p.mpp_y),
            width: props.map_or(0, |p| p.width),
            height: props.map_or(0, |p| p.height),
            level_count: props.map_or(0, |p| p.levels.len() as u32),
            read_level: None,
            downsample: None,
            output_mpp: None,
            tissue_fraction: None,
            candidates: planned.geometry.as_ref().map_or(0, |g| g.grid.len()),
            written: 0,
            existing: 0,
            skipped_white: 0,
            skipped_tissue: 0,
            skipped_annotation: 0,
            failed: 0,
            fingerprint: None,
            file_sha256: None,
            stain_normalization: None,
            annotation_file: None,
        };
        let report = SlideReport {
            path: planned.input.path.clone(),
            slide_id: planned.key.clone(),
            status,
            tiles_written: 0,
            tiles_existing: 0,
            tiles_skipped_white: 0,
            tiles_skipped_tissue: 0,
            tiles_skipped_annotation: 0,
            tiles_skipped_qc: 0,
            tiles_failed: 0,
            skipped: Some(reason),
            errors: Vec::new(),
        };
        (report, record)
    }
}

const OUTPUT_FILES: &[&str] = &[
    metadata::CSV_FILE,
    metadata::JSONL_FILE,
    metadata::JSON_FILE,
    metadata::SLIDES_CSV_FILE,
    metadata::SLIDES_JSONL_FILE,
    MANIFEST_FILE,
    "classes.json",
];

/// Prepares the output directory; returns whether the run resumes.
fn prepare_output(
    config: &DatasetPatchesConfig,
    options: &DatasetPatchesOptions,
    parameters: &ExtractionParameters,
) -> Result<bool, DatasetError> {
    let dir = &config.output_dir;
    std::fs::create_dir_all(dir).map_err(write_err(dir))?;
    let has_metadata = OUTPUT_FILES.iter().any(|f| dir.join(f).exists());
    match options.existing {
        ExistingOutput::Overwrite => {
            for f in OUTPUT_FILES {
                let path = dir.join(f);
                if path.exists() {
                    std::fs::remove_file(&path).map_err(write_err(&path))?;
                }
            }
            Ok(false)
        }
        ExistingOutput::Resume => {
            let manifest =
                DatasetManifest::load(dir).map_err(write_err(&dir.join(MANIFEST_FILE)))?;
            match manifest {
                Some(m) if &m.parameters == parameters => Ok(true),
                Some(_) => Err(DatasetError::OutputConflict(format!(
                    "{} already holds a dataset made with different parameters; use another \
                     output directory, or overwrite it",
                    dir.display()
                ))),
                None if has_metadata => Err(DatasetError::OutputConflict(format!(
                    "{} holds dataset metadata without a manifest (from an older eov); use \
                     another output directory, or overwrite it",
                    dir.display()
                ))),
                None => Ok(false),
            }
        }
    }
}

/// Runs the pipeline.
pub fn run_dataset(
    config: &DatasetPatchesConfig,
    options: &DatasetPatchesOptions,
    control: &DatasetControl,
) -> Result<DatasetPatchesReport, DatasetError> {
    control.set_phase(DatasetPhase::Discovering);
    config.validate(options)?;
    control.set_phase(DatasetPhase::Planning);
    let plan = plan_dataset(config, options)?;
    control
        .total_slides
        .store(plan.slides.len() as u64, Ordering::Relaxed);
    control
        .tiles_total
        .store(plan.planned_tiles, Ordering::Relaxed);

    let mut report = DatasetPatchesReport {
        input_count: config.inputs.len(),
        discovered_slides: plan.slides.len(),
        processed_slides: 0,
        skipped_slides: 0,
        failed_slides: 0,
        total_tiles: 0,
        total_tiles_skipped_white: 0,
        total_tiles_skipped_tissue: 0,
        total_tiles_skipped_annotation: 0,
        total_tiles_skipped_qc: 0,
        total_tiles_existing: 0,
        total_tiles_failed: 0,
        planned_tiles: plan.planned_tiles,
        metadata_paths: Vec::new(),
        manifest_path: None,
        slides: Vec::new(),
        input_errors: plan.input_errors.clone(),
        skipped_inputs: plan.skipped_inputs.clone(),
        cancelled: false,
        dry_run: options.dry_run,
    };
    if options.dry_run {
        for planned in &plan.slides {
            if let Err(reason) = &planned.geometry {
                report.skipped_slides += 1;
                report.slides.push(SlideReport {
                    path: planned.input.path.clone(),
                    slide_id: planned.key.clone(),
                    status: SlideStatus::Skipped,
                    tiles_written: 0,
                    tiles_existing: 0,
                    tiles_skipped_white: 0,
                    tiles_skipped_tissue: 0,
                    tiles_skipped_annotation: 0,
                    tiles_skipped_qc: 0,
                    tiles_failed: 0,
                    skipped: Some(reason.clone()),
                    errors: Vec::new(),
                });
            }
        }
        control.set_phase(DatasetPhase::Done);
        return Ok(report);
    }

    let parameters = ExtractionParameters::new(config, options);
    let resume = prepare_output(config, options, &parameters)?;
    let previous = if resume {
        metadata::prepare_resume(&config.output_dir, config.metadata_format)
            .map_err(write_err(&config.output_dir))?
    } else {
        metadata::ResumeState::default()
    };
    let started = DatasetManifest::load(&config.output_dir)
        .ok()
        .flatten()
        .filter(|_| resume)
        .map_or_else(now, |m| m.started);
    let mut manifest = DatasetManifest {
        format_version: MANIFEST_FORMAT_VERSION,
        software: crate::export::software_name(),
        parameters,
        inputs: if options.deidentify {
            Vec::new()
        } else {
            config
                .inputs
                .iter()
                .map(|p| p.display().to_string())
                .collect()
        },
        status: RunStatus::Running,
        started,
        updated: now(),
        totals: RunTotals::default(),
        classes: plan.classes.clone(),
    };
    let manifest_path = config.output_dir.join(MANIFEST_FILE);
    manifest
        .save(&config.output_dir)
        .map_err(write_err(&manifest_path))?;

    // From here on the manifest records how the run ended, also when it
    // stops with an error.
    let result = extract_planned(
        config,
        options,
        control,
        &plan,
        previous,
        resume,
        &mut report,
    );
    manifest.status = match &result {
        Err(_) => RunStatus::Failed,
        Ok(()) => report.status(),
    };
    manifest.updated = now();
    manifest.totals = RunTotals {
        slides: report.slides.len() as u64,
        slides_failed: report.failed_slides as u64,
        tiles_written: report.total_tiles,
        tiles_existing: report.total_tiles_existing,
        tiles_skipped_white: report.total_tiles_skipped_white,
        tiles_skipped_tissue: report.total_tiles_skipped_tissue,
        tiles_skipped_annotation: report.total_tiles_skipped_annotation,
        tiles_skipped_qc: report.total_tiles_skipped_qc,
        tiles_failed: report.total_tiles_failed,
    };
    let saved = manifest
        .save(&config.output_dir)
        .map_err(write_err(&manifest_path));
    report.manifest_path = Some(manifest_path);
    control.set_phase(DatasetPhase::Done);
    result.and(saved).map(|()| report)
}

/// Writes the classes file, extracts every planned slide and finishes the
/// metadata, accumulating into `report`.
fn extract_planned(
    config: &DatasetPatchesConfig,
    options: &DatasetPatchesOptions,
    control: &DatasetControl,
    plan: &DatasetPlan,
    previous: metadata::ResumeState,
    resume: bool,
    report: &mut DatasetPatchesReport,
) -> Result<(), DatasetError> {
    if !plan.classes.is_empty() {
        let classes: serde_json::Map<String, serde_json::Value> =
            std::iter::once(("0".to_string(), "background".into()))
                .chain(
                    plan.classes
                        .iter()
                        .enumerate()
                        .map(|(i, c)| ((i + 1).to_string(), c.clone().into())),
                )
                .collect();
        let path = config.output_dir.join("classes.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&classes).unwrap_or_default() + "\n",
        )
        .map_err(write_err(&path))?;
    }
    let mut writer = MetadataWriter::open(&config.output_dir, config.metadata_format, resume)
        .map_err(write_err(&config.output_dir))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .thread_name(|i| format!("ds-patch-{i}"))
        .build()
        .map_err(|e| DatasetError::Write {
            path: config.output_dir.clone(),
            source: std::io::Error::other(format!("cannot start worker threads: {e}")),
        })?;
    let run = Run {
        config,
        options,
        control,
        recorded: previous.tiles,
        previous: previous.slides,
        classes: plan.classes.clone(),
        pool,
    };

    let slides_path = config.output_dir.join(metadata::SLIDES_JSONL_FILE);
    let mut result = Ok(());
    for (index, planned) in plan.slides.iter().enumerate() {
        let outcome = if control.is_cancelled() {
            report.cancelled = true;
            let (slide_report, record) = run.skipped(planned, SlideSkipReason::Cancelled);
            Ok((slide_report, Some(record)))
        } else {
            control
                .current_slide
                .store(index as u64 + 1, Ordering::Relaxed);
            run.slide(planned, &mut writer)
        };
        match outcome.and_then(|(slide_report, record)| {
            if let Some(record) = record {
                writer
                    .append_slide(&record)
                    .map_err(write_err(&slides_path))?;
            }
            Ok(slide_report)
        }) {
            Ok(slide_report) => report.slides.push(slide_report),
            Err(err) => {
                result = Err(err);
                break;
            }
        }
        if control.is_cancelled() {
            report.cancelled = true;
        }
    }

    for slide in &report.slides {
        match slide.status {
            SlideStatus::Complete | SlideStatus::Partial => report.processed_slides += 1,
            SlideStatus::Failed => {
                report.failed_slides += 1;
                report.skipped_slides += 1;
            }
            SlideStatus::Skipped | SlideStatus::Cancelled => report.skipped_slides += 1,
        }
        report.total_tiles += slide.tiles_written;
        report.total_tiles_existing += slide.tiles_existing;
        report.total_tiles_skipped_white += slide.tiles_skipped_white;
        report.total_tiles_skipped_tissue += slide.tiles_skipped_tissue;
        report.total_tiles_skipped_annotation += slide.tiles_skipped_annotation;
        report.total_tiles_skipped_qc += slide.tiles_skipped_qc;
        report.total_tiles_failed += slide.tiles_failed;
    }
    control.set_phase(DatasetPhase::Finishing);
    // Finish the metadata even after an error, so that what was written is
    // complete and consistent; report the first error.
    let finished = writer.finish().map_err(write_err(&config.output_dir));
    if let Ok(paths) = &finished {
        report.metadata_paths = paths.clone();
    }
    result.and(finished.map(drop))
}

// ─── Pre-1.0 entry points ────────────────────────────────────────────────────

fn legacy_options() -> DatasetPatchesOptions {
    DatasetPatchesOptions {
        // The pre-1.0 GUI always wrote into the same folder, replacing the
        // metadata; keep that behaviour for these entry points.
        existing: ExistingOutput::Overwrite,
        ..DatasetPatchesOptions::default()
    }
}

/// Runs the pipeline with default options (replacing existing metadata).
pub fn run_dataset_patches(config: &DatasetPatchesConfig) -> crate::Result<DatasetPatchesReport> {
    run_dataset(config, &legacy_options(), &DatasetControl::new()).map_err(Into::into)
}

/// Sets a flag when dropped — also while unwinding from a panic.
struct SetOnDrop<'a>(&'a AtomicBool);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Runs `work` while a helper thread forwards `cancel` to `control` and
/// calls `mirror` every 50 ms. The helper stops when `work` returns or
/// panics (a panic then propagates instead of leaving the scope waiting
/// for the helper forever).
fn with_progress_mirror<R>(
    control: &DatasetControl,
    cancel: &AtomicBool,
    mirror: &(dyn Fn() + Sync),
    work: impl FnOnce() -> R,
) -> R {
    let finished = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                if cancel.load(Ordering::Relaxed) {
                    control.cancel();
                }
                mirror();
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let result = {
            let _finished = SetOnDrop(&finished);
            work()
        };
        mirror();
        result
    })
}

/// Pre-1.0 entry point with caller-owned progress counters.
///
/// `progress_tiles` receives the number of processed tiles (written,
/// skipped, existing or failed) and `progress_total_tiles_expected` the
/// planned total, which is final before extraction starts.
pub fn run_dataset_patches_with_progress(
    config: &DatasetPatchesConfig,
    cancel: &AtomicBool,
    progress_tiles: &AtomicU64,
    progress_current_slide: &AtomicU64,
    progress_total_slides: &AtomicU64,
    progress_total_tiles_expected: &AtomicU64,
) -> crate::Result<DatasetPatchesReport> {
    let control = DatasetControl::new();
    let mirror = || {
        let p = control.progress();
        progress_tiles.store(p.tiles_done, Ordering::Relaxed);
        progress_current_slide.store(p.current_slide, Ordering::Relaxed);
        progress_total_slides.store(p.total_slides, Ordering::Relaxed);
        progress_total_tiles_expected.store(p.tiles_total, Ordering::Relaxed);
    };
    with_progress_mirror(&control, cancel, &mirror, || {
        run_dataset(config, &legacy_options(), &control)
    })
    .map_err(Into::into)
}

#[cfg(test)]
mod tests;
