//! Core dataset-patches pipeline.
//!
//! Orchestrates input discovery → grid generation → tile extraction → image
//! writing → metadata export. Designed for reuse from both the CLI and a
//! future GUI dialog.

use std::path::PathBuf;
use std::time::Instant;

use tracing::{info, warn};

use crate::WsiFile;

use super::config::{DatasetPatchesConfig, MetadataFormat};
use super::discovery::expand_inputs;
use super::grid::generate_patch_coords;
use super::metadata::{self, TileRecord};
use super::output;

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
    /// Path to the metadata file, if one was written.
    pub metadata_path: Option<PathBuf>,
    /// Per-slide details.
    pub slides: Vec<SlideReport>,
    /// Input-level errors (nonexistent paths, unsupported extensions, etc.).
    pub input_errors: Vec<(PathBuf, String)>,
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
/// If a slide cannot be opened, it is skipped and the failure is recorded in
/// the report. Processing continues with remaining slides.
pub fn run_dataset_patches(config: &DatasetPatchesConfig) -> crate::Result<DatasetPatchesReport> {
    let start = Instant::now();

    // --- 1. Resolve inputs ---
    let (slide_paths, input_errors) = expand_inputs(&config.inputs);

    if slide_paths.is_empty() && input_errors.is_empty() {
        return Err(crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no valid slide files found in the provided inputs",
        )));
    }

    info!(
        "Discovered {} slide(s) from {} input path(s) ({} input error(s))",
        slide_paths.len(),
        config.inputs.len(),
        input_errors.len(),
    );

    // --- 2. Create base output directory ---
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

    // --- 3-6. Process each slide ---
    for slide_path in &slide_paths {
        let stem = output::slide_stem(slide_path);

        // Try to open the slide.
        let wsi = match WsiFile::open(slide_path) {
            Ok(w) => w,
            Err(e) => {
                warn!("Skipping {}: {e}", slide_path.display());
                slide_reports.push(SlideReport {
                    path: slide_path.clone(),
                    tiles_written: 0,
                    skipped: Some(SlideSkipReason::OpenError(e.to_string())),
                });
                continue;
            }
        };

        let props = wsi.properties();

        // Generate patch coordinates.
        let coords = generate_patch_coords(
            props.width,
            props.height,
            config.tile_size,
            config.stride,
        );

        if coords.is_empty() {
            info!(
                "Slide {} ({}×{}) is smaller than tile_size {}; skipping",
                stem, props.width, props.height, config.tile_size
            );
            slide_reports.push(SlideReport {
                path: slide_path.clone(),
                tiles_written: 0,
                skipped: Some(SlideSkipReason::TooSmall {
                    width: props.width,
                    height: props.height,
                }),
            });
            continue;
        }

        // Create per-slide output directory.
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

        info!(
            "Extracting {} tiles from {} ({}×{}) into {}",
            coords.len(),
            stem,
            props.width,
            props.height,
            tiles_dir.display(),
        );

        let mut slide_tiles: u64 = 0;

        for coord in &coords {
            // Read a tile_size × tile_size region at level 0.
            let data = wsi.read_region(
                coord.x as i64,
                coord.y as i64,
                0,
                config.tile_size,
                config.tile_size,
            )?;

            let rel_path =
                output::tile_relative_path(&stem, coord.x, coord.y, config.tile_size);
            let abs_path = config.output_dir.join(&rel_path);

            output::write_tile_png(&abs_path, &data, config.tile_size, config.tile_size)
                .map_err(|e| {
                    crate::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to write tile {}: {e}", abs_path.display()),
                    ))
                })?;

            if config.metadata_format.is_some() {
                all_records.push(TileRecord {
                    slide_path: slide_path.display().to_string(),
                    slide_stem: stem.clone(),
                    tile_path: rel_path,
                    x: coord.x,
                    y: coord.y,
                    tile_size: config.tile_size,
                    width: config.tile_size,
                    height: config.tile_size,
                    slide_width: props.width,
                    slide_height: props.height,
                    level: 0,
                    mpp_x: props.mpp_x,
                    mpp_y: props.mpp_y,
                });
            }

            slide_tiles += 1;
        }

        total_tiles += slide_tiles;
        slide_reports.push(SlideReport {
            path: slide_path.clone(),
            tiles_written: slide_tiles,
            skipped: None,
        });
    }

    // --- 7. Write metadata ---
    let metadata_path = match config.metadata_format {
        Some(MetadataFormat::Csv) => {
            let p = config.output_dir.join("metadata.csv");
            metadata::write_csv(&all_records, &p).map_err(|e| {
                crate::Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to write metadata CSV {}: {e}", p.display()),
                ))
            })?;
            info!("Wrote {} tile records to {}", all_records.len(), p.display());
            Some(p)
        }
        Some(MetadataFormat::Json) => {
            let p = config.output_dir.join("metadata.json");
            metadata::write_json(&all_records, &p).map_err(|e| {
                crate::Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to write metadata JSON {}: {e}", p.display()),
                ))
            })?;
            info!("Wrote {} tile records to {}", all_records.len(), p.display());
            Some(p)
        }
        None => None,
    };

    let processed = slide_reports
        .iter()
        .filter(|r| r.skipped.is_none())
        .count();
    let skipped = slide_reports.iter().filter(|r| r.skipped.is_some()).count();

    info!(
        "Dataset extraction complete in {:.1}s: {} slide(s) processed, {} skipped, {} tile(s) written",
        start.elapsed().as_secs_f64(),
        processed,
        skipped,
        total_tiles,
    );

    Ok(DatasetPatchesReport {
        input_count: config.inputs.len(),
        discovered_slides: slide_paths.len(),
        processed_slides: processed,
        skipped_slides: skipped,
        total_tiles,
        metadata_path,
        slides: slide_reports,
        input_errors,
    })
}
