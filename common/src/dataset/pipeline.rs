//! Core dataset-patches pipeline.
//!
//! Orchestrates input discovery → grid generation → tile extraction → image
//! writing → metadata export. Designed for reuse from both the CLI and a
//! future GUI dialog.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rayon::prelude::*;
use tracing::warn;

use crate::WsiFile;

use super::config::{DatasetPatchesConfig, MetadataFormat};
use super::discovery::expand_inputs;
use super::grid::generate_patch_coords;
use super::metadata::{self, TileRecord};
use super::output;

fn slide_file_ids(slide_path: &std::path::Path) -> crate::Result<(String, String)> {
    let fingerprint = crate::file_id::compute_fingerprint(slide_path).map_err(|err| {
        crate::Error::Io(std::io::Error::new(
            err.kind(),
            format!(
                "failed to compute slide fingerprint for {}: {err}",
                slide_path.display()
            ),
        ))
    })?;
    let file_sha256 = crate::file_id::cached_sha256(slide_path).map_err(|err| {
        crate::Error::Io(std::io::Error::new(
            err.kind(),
            format!(
                "failed to compute slide SHA-256 for {}: {err}",
                slide_path.display()
            ),
        ))
    })?;
    Ok((
        crate::file_id::hex_digest(&fingerprint),
        crate::file_id::hex_digest(&file_sha256),
    ))
}

/// Reason a slide was skipped during processing.
#[derive(Debug, Clone)]
pub enum SlideSkipReason {
    /// The slide could not be opened.
    OpenError(String),
    /// The slide is smaller than `tile_size` in at least one dimension.
    TooSmall { width: u64, height: u64 },
}

/// Per-slide statistics.
#[derive(Debug, Clone)]
pub struct SlideReport {
    pub path: PathBuf,
    pub tiles_written: u64,
    pub tiles_skipped_white: u64,
    pub skipped: Option<SlideSkipReason>,
}

/// Summary report returned by [`run_dataset_patches`].
#[derive(Debug)]
pub struct DatasetPatchesReport {
    /// Number of raw input paths provided.
    pub input_count: usize,
    /// Number of slide files discovered after expansion.
    pub discovered_slides: usize,
    /// Number of slides that were successfully processed.
    pub processed_slides: usize,
    /// Number of slides that were skipped (open error or too small).
    pub skipped_slides: usize,
    /// Total tiles written across all slides.
    pub total_tiles: u64,
    /// Total tiles skipped because they were mostly white.
    pub total_tiles_skipped_white: u64,
    /// Paths to the metadata files that were written (CSV and JSON).
    pub metadata_paths: Vec<PathBuf>,
    /// Per-slide details.
    pub slides: Vec<SlideReport>,
    /// Input-level errors (nonexistent paths, unsupported extensions, etc.).
    pub input_errors: Vec<(PathBuf, String)>,
}

/// Progress snapshot for the dataset-patches pipeline.
#[derive(Debug, Clone)]
pub struct DatasetPatchesProgress {
    /// 1-based index of the slide currently being processed.
    pub current_slide: usize,
    /// Total number of slides to process.
    pub total_slides: usize,
    /// Tiles written so far across all slides.
    pub tiles_exported: u64,
    /// Total tiles expected across all slides (may grow as slides are opened).
    pub total_tiles_expected: u64,
    /// Wall-clock time elapsed since the pipeline started.
    pub elapsed: std::time::Duration,
}

/// Run the full dataset-patches extraction pipeline.
///
/// This is the main entrypoint for the feature. It:
///
/// 1. Resolves input paths into individual slide files.
/// 2. Creates the output directory structure.
/// 3. Iterates slides, generating a fixed grid of patch coordinates.
/// 4. Reads each patch from level 0 via OpenSlide and writes it as PNG.
/// 5. Optionally writes per-tile metadata in CSV or JSON format.
/// 6. Returns a structured report suitable for CLI display or GUI feedback.
///
/// Tile extraction is parallelised across a rayon thread pool. Each worker
/// thread opens its own `WsiFile` handle so there is no shared-lock
/// contention on a single OpenSlide descriptor.
///
/// If a slide cannot be opened, it is skipped and the failure is recorded in
/// the report. Processing continues with remaining slides.
pub fn run_dataset_patches(config: &DatasetPatchesConfig) -> crate::Result<DatasetPatchesReport> {
    let cancel = AtomicBool::new(false);
    let tiles = AtomicU64::new(0);
    let current_slide = AtomicU64::new(0);
    let total_slides = AtomicU64::new(0);
    let total_tiles_expected = AtomicU64::new(0);
    run_dataset_patches_with_progress(
        config,
        &cancel,
        &tiles,
        &current_slide,
        &total_slides,
        &total_tiles_expected,
    )
}

/// Run the dataset-patches pipeline with cancellation and progress reporting.
///
/// `cancel` — set to `true` from another thread to abort early.
/// `progress` — shared progress counters updated atomically by workers.
pub fn run_dataset_patches_with_progress(
    config: &DatasetPatchesConfig,
    cancel: &AtomicBool,
    progress_tiles: &AtomicU64,
    progress_current_slide: &AtomicU64,
    progress_total_slides: &AtomicU64,
    progress_total_tiles_expected: &AtomicU64,
) -> crate::Result<DatasetPatchesReport> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .thread_name(|i| format!("ds-patch-{i}"))
        .build()
        .map_err(|e| {
            crate::Error::Io(std::io::Error::other(format!(
                "failed to create thread pool: {e}"
            )))
        })?;

    let (slide_paths, input_errors) = expand_inputs(&config.inputs);

    if slide_paths.is_empty() && input_errors.is_empty() {
        return Err(crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no valid slide files found in the provided inputs",
        )));
    }

    progress_total_slides.store(slide_paths.len() as u64, Ordering::Relaxed);

    std::fs::create_dir_all(&config.output_dir).map_err(|e| {
        crate::Error::Io(std::io::Error::new(
            e.kind(),
            format!(
                "failed to create output directory {}: {e}",
                config.output_dir.display()
            ),
        ))
    })?;

    let mut all_records: Vec<TileRecord> = Vec::new();
    let mut slide_reports: Vec<SlideReport> = Vec::new();
    let mut total_tiles: u64 = 0;

    let slide_keys = output::unique_slide_keys(&slide_paths);

    for (slide_idx, slide_path) in slide_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        progress_current_slide.store((slide_idx + 1) as u64, Ordering::Relaxed);

        let stem = slide_keys[slide_idx].clone();

        let wsi = match WsiFile::open(slide_path) {
            Ok(w) => w,
            Err(e) => {
                warn!("Skipping {}: {e}", slide_path.display());
                slide_reports.push(SlideReport {
                    path: slide_path.clone(),
                    tiles_written: 0,
                    tiles_skipped_white: 0,
                    skipped: Some(SlideSkipReason::OpenError(e.to_string())),
                });
                continue;
            }
        };

        let props = wsi.properties().clone();
        drop(wsi);

        let coords =
            generate_patch_coords(props.width, props.height, config.tile_size, config.stride);

        if coords.is_empty() {
            slide_reports.push(SlideReport {
                path: slide_path.clone(),
                tiles_written: 0,
                tiles_skipped_white: 0,
                skipped: Some(SlideSkipReason::TooSmall {
                    width: props.width,
                    height: props.height,
                }),
            });
            continue;
        }

        let (fingerprint, file_sha256) = slide_file_ids(slide_path)?;

        progress_total_tiles_expected.fetch_add(coords.len() as u64, Ordering::Relaxed);

        let tiles_dir = output::slide_tiles_dir(&config.output_dir, &stem);
        std::fs::create_dir_all(&tiles_dir).map_err(|e| {
            crate::Error::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "failed to create tile directory {}: {e}",
                    tiles_dir.display()
                ),
            ))
        })?;

        let tile_size = config.tile_size;
        let output_dir = &config.output_dir;
        let white_threshold = config.white_threshold;
        let first_error: Mutex<Option<crate::Error>> = Mutex::new(None);
        let tile_records: Mutex<Vec<TileRecord>> = Mutex::new(Vec::new());
        let slide_tile_count = AtomicU64::new(0);
        let slide_white_count = AtomicU64::new(0);

        // One OpenSlide handle per worker thread, scoped to *this* slide.
        // Worker threads persist across slides, so a thread-local cache would
        // keep serving the first slide's handle for every later slide.
        let handle_slots: Vec<Mutex<Option<WsiFile>>> = (0..pool.current_num_threads().max(1))
            .map(|_| Mutex::new(None))
            .collect();

        pool.install(|| {
            coords.par_iter().for_each(|coord| {
                if cancel.load(Ordering::Relaxed) || first_error.lock().unwrap().is_some() {
                    return;
                }

                let slot_index = rayon::current_thread_index().unwrap_or(0) % handle_slots.len();
                let read_result = {
                    let mut slot = handle_slots[slot_index].lock().unwrap();
                    if slot.is_none() {
                        match WsiFile::open(slide_path) {
                            Ok(w) => *slot = Some(w),
                            Err(e) => {
                                *first_error.lock().unwrap() = Some(e);
                                return;
                            }
                        }
                    }
                    let wsi = slot.as_ref().expect("slot initialized above");
                    wsi.read_region(coord.x as i64, coord.y as i64, 0, tile_size, tile_size)
                };

                let data = match read_result {
                    Ok(d) => d,
                    Err(e) => {
                        *first_error.lock().unwrap() = Some(e);
                        return;
                    }
                };

                // Skip tiles that are almost completely white.
                if let Some(thresh) = white_threshold
                    && output::is_tile_mostly_white(&data, thresh)
                {
                    slide_white_count.fetch_add(1, Ordering::Relaxed);
                    progress_tiles.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                let rel_path = output::tile_relative_path(&stem, coord.x, coord.y, tile_size);
                let abs_path = output_dir.join(&rel_path);

                if let Err(e) = output::write_tile_png(&abs_path, &data, tile_size, tile_size) {
                    *first_error.lock().unwrap() = Some(crate::Error::Io(std::io::Error::other(
                        format!("failed to write tile {}: {e}", abs_path.display()),
                    )));
                    return;
                }

                slide_tile_count.fetch_add(1, Ordering::Relaxed);
                progress_tiles.fetch_add(1, Ordering::Relaxed);

                tile_records.lock().unwrap().push(TileRecord {
                    slide_path: slide_path.display().to_string(),
                    fingerprint: fingerprint.clone(),
                    file_sha256: file_sha256.clone(),
                    slide_stem: stem.clone(),
                    tile_path: rel_path,
                    x: coord.x,
                    y: coord.y,
                    tile_size,
                    width: tile_size,
                    height: tile_size,
                    slide_width: props.width,
                    slide_height: props.height,
                    level: 0,
                    mpp_x: props.mpp_x,
                    mpp_y: props.mpp_y,
                });
            });
        });

        if cancel.load(Ordering::Relaxed) {
            break;
        }

        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        let slide_tiles = slide_tile_count.load(Ordering::Relaxed);
        let slide_white = slide_white_count.load(Ordering::Relaxed);
        let mut records = tile_records.into_inner().unwrap();
        records.sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));
        all_records.append(&mut records);
        total_tiles += slide_tiles;
        slide_reports.push(SlideReport {
            path: slide_path.clone(),
            tiles_written: slide_tiles,
            tiles_skipped_white: slide_white,
            skipped: None,
        });
    }

    let metadata_paths = if !cancel.load(Ordering::Relaxed) {
        let mut paths = Vec::new();
        let write_csv =
            config.metadata_format.is_none() || config.metadata_format == Some(MetadataFormat::Csv);
        let write_json = config.metadata_format.is_none()
            || config.metadata_format == Some(MetadataFormat::Json);

        if write_csv {
            let csv_path = config.output_dir.join("metadata.csv");
            metadata::write_csv(&all_records, &csv_path).map_err(|e| {
                crate::Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to write metadata CSV {}: {e}", csv_path.display()),
                ))
            })?;
            paths.push(csv_path);
        }

        if write_json {
            let json_path = config.output_dir.join("metadata.json");
            metadata::write_json(&all_records, &json_path).map_err(|e| {
                crate::Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to write metadata JSON {}: {e}", json_path.display()),
                ))
            })?;
            paths.push(json_path);
        }

        paths
    } else {
        Vec::new()
    };

    let processed = slide_reports.iter().filter(|r| r.skipped.is_none()).count();
    let skipped = slide_reports.iter().filter(|r| r.skipped.is_some()).count();
    let total_white: u64 = slide_reports.iter().map(|r| r.tiles_skipped_white).sum();

    Ok(DatasetPatchesReport {
        input_count: config.inputs.len(),
        discovered_slides: slide_paths.len(),
        processed_slides: processed,
        skipped_slides: skipped,
        total_tiles,
        total_tiles_skipped_white: total_white,
        metadata_paths,
        slides: slide_reports,
        input_errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_tiled_tiff;

    fn config(inputs: Vec<PathBuf>, output_dir: PathBuf) -> DatasetPatchesConfig {
        DatasetPatchesConfig {
            inputs,
            output_dir,
            tile_size: 128,
            stride: 128,
            metadata_format: Some(MetadataFormat::Json),
            threads: 3,
            white_threshold: None,
        }
    }

    fn centre_colour(path: &std::path::Path) -> [u8; 3] {
        let image = image::open(path).expect("tile image").to_rgba8();
        let px = image.get_pixel(image.width() / 2, image.height() / 2).0;
        [px[0], px[1], px[2]]
    }

    fn tiles_below(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut tiles: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("tile directory")
            .map(|entry| entry.unwrap().path())
            .collect();
        tiles.sort();
        tiles
    }

    #[test]
    fn every_slide_is_read_from_its_own_file() {
        let temp = tempfile::tempdir().unwrap();
        let colours = [[200, 40, 40], [40, 40, 200], [40, 160, 40]];
        let mut inputs = Vec::new();
        for (index, colour) in colours.iter().enumerate() {
            let path = temp.path().join(format!("slide{index}.tif"));
            write_tiled_tiff(&path, 512, 384, 64, |_, _| *colour).unwrap();
            inputs.push(path);
        }
        let out = temp.path().join("out");
        let report = run_dataset_patches(&config(inputs, out.clone())).unwrap();
        assert_eq!(report.processed_slides, 3);
        assert_eq!(report.total_tiles, 3 * 4 * 3);
        for (index, colour) in colours.iter().enumerate() {
            let tiles = tiles_below(&out.join("slides").join(format!("slide{index}")));
            assert_eq!(tiles.len(), 12);
            for tile in tiles {
                assert_eq!(&centre_colour(&tile), colour, "{}", tile.display());
            }
        }
    }

    #[test]
    fn slides_with_the_same_stem_do_not_overwrite_each_other() {
        let temp = tempfile::tempdir().unwrap();
        let a = temp.path().join("case1").join("HE.tif");
        let b = temp.path().join("case2").join("HE.tif");
        std::fs::create_dir_all(a.parent().unwrap()).unwrap();
        std::fs::create_dir_all(b.parent().unwrap()).unwrap();
        write_tiled_tiff(&a, 256, 256, 64, |_, _| [220, 30, 30]).unwrap();
        write_tiled_tiff(&b, 256, 256, 64, |_, _| [30, 30, 220]).unwrap();
        let out = temp.path().join("out");
        let report = run_dataset_patches(&config(vec![a, b], out.clone())).unwrap();
        assert_eq!(report.total_tiles, 8);
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(out.join("slides"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        dirs.sort();
        assert_eq!(dirs.len(), 2, "each slide needs its own directory: {dirs:?}");
        let mut seen: Vec<[u8; 3]> = dirs
            .iter()
            .map(|dir| centre_colour(&tiles_below(dir)[0]))
            .collect();
        seen.sort();
        assert_eq!(seen, vec![[30, 30, 220], [220, 30, 30]]);
    }
}
