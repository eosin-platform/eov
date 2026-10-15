//! Writing slides (AUDIT §14.3, §14.6): pyramidal TIFF and OME-TIFF copies
//! of a whole slide or a region, optionally de-identified.
//!
//! # Output
//!
//! * **TIFF** ([`SlideFormat::Tiff`]): a tiled TIFF, or BigTIFF when the
//!   file does not fit 4 GiB (decided when the file is complete, so small
//!   copies stay classic TIFFs). Every pyramid level is its own tiled
//!   image file directory (IFD), full resolution first, the others marked
//!   reduced-resolution (`NewSubfileType` 1) and halving down to one tile:
//!   OpenSlide's "generic tiled TIFF", which eov, QuPath, libvips and
//!   most other tools read as a pyramid.
//! * **OME-TIFF** ([`SlideFormat::OmeTiff`]): OME-XML (2016-06 schema) in
//!   the first IFD, the reduced levels as SubIFDs of the full-resolution
//!   IFD (Bio-Formats 6 / QuPath / OMERO pyramids). OpenSlide does not
//!   read SubIFDs and sees the full-resolution image only.
//!
//! Tiles are 256 or 512 px, compressed with baseline JPEG (YCbCr, shared
//! `JPEGTables`, abbreviated tile streams, as libtiff writes them; chroma
//! subsampled 2×2 below quality 90) or lossless Deflate with the
//! horizontal predictor. Lower levels are the 2×2 mean of the level above,
//! built while the full-resolution tiles are read (see `pyramid`), in
//! bounded memory. The pixel size goes into `XResolution`/`YResolution`
//! (pixels per centimetre), which OpenSlide reports as
//! `openslide.mpp-x`/`-y`, so measurements work on the copy; the pixel
//! size, objective power, stain and source format are also written as
//! `key = value` lines in the `ImageDescription` (OpenSlide's
//! `openslide.comment`) or the OME-XML. The slide's ICC profile is kept
//! unless [`ConvertOptions::keep_icc_profile`] is off.
//!
//! The copy is written to [`partial_path`] and renamed into place when it
//! is complete, so an error or a cancellation never leaves a truncated
//! file behind.
//!
//! # De-identification
//!
//! With [`ConvertOptions::deidentify`], the copy has no associated images
//! (no label, macro or thumbnail, which can show the label), and its
//! metadata is built from an allow-list of fields: geometry, pixel size,
//! objective power, stain and, with [`ConvertOptions::keep_scanner`], the
//! scanner vendor and model. Dates, barcodes, file names, paths, users,
//! comments, slide and device ids and every other vendor property are
//! dropped, and the description says that it is a de-identified copy made
//! by eov, with the version and date. This covers metadata and associated
//! images only: text or marks *in the tissue image itself* (a label inside
//! the scanned area, pen marks, burned-in annotations) are copied as they
//! are; choose a region that leaves them out.
//!
//! # Example
//!
//! ```no_run
//! use eov_common::WsiFile;
//! use eov_common::convert::{ConvertOptions, convert_slide};
//! use eov_common::job::JobControl;
//!
//! # fn demo() -> Result<(), eov_common::convert::ConvertError> {
//! let slide = WsiFile::open("case.svs")?;
//! let options = ConvertOptions {
//!     deidentify: true,
//!     ..ConvertOptions::default()
//! };
//! let control = JobControl::new(); // share via Arc; poll progress(), call cancel()
//! let report = convert_slide(&slide, "teaching/slide-01.tif".as_ref(), &options, &control)?;
//! println!("{} levels, {} bytes", report.plan.levels.len(), report.bytes);
//! # Ok(()) }
//! ```

mod metadata;
mod pyramid;
mod source;
#[cfg(test)]
mod tests;
mod tiff_file;

use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use crate::job::JobControl;
use crate::wsi::{SlideBounds, WsiFile, best_level_for_downsample};
use metadata::{CopyMetadata, SlideFacts, SourceSampling};
use pyramid::{Geometry, LevelTiles, TileCodec, TileEncoder, deflate_rows, write_pyramid};
use source::{Mapping, SlideTiles};
use tiff_file::{Ifd, TiffFile, Value};

/// Tile sizes the command line and the dialog offer.
pub const TILE_SIZES: [u32; 2] = [256, 512];
/// The default tile size.
pub const DEFAULT_TILE_SIZE: u32 = 512;
/// The default JPEG quality (chroma at full resolution).
pub const DEFAULT_JPEG_QUALITY: u8 = 90;
/// A target pixel size within this fraction of a stored level's is served
/// from that level's pixels instead of being resampled.
pub const LEVEL_MATCH_TOLERANCE: f64 = 0.01;
/// Largest width or height of a copy.
pub const MAX_SIDE: u64 = (1 << 31) - 1;
/// Smallest width or height of a copy.
pub const MIN_SIDE: u64 = 16;

/// The container written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlideFormat {
    /// Pyramidal (Big)TIFF, one IFD per level.
    #[default]
    Tiff,
    /// OME-TIFF with a SubIFD pyramid.
    OmeTiff,
}

impl SlideFormat {
    /// The format a file name asks for: `.ome.tif`, `.ome.tiff` and
    /// `.ome.btf` are OME-TIFF, anything else TIFF.
    pub fn from_path(path: &Path) -> Self {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if [".ome.tif", ".ome.tiff", ".ome.btf", ".ome.tf2", ".ome.tf8"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            Self::OmeTiff
        } else {
            Self::Tiff
        }
    }

    /// The usual file-name suffix (without the leading dot).
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Tiff => "tif",
            Self::OmeTiff => "ome.tif",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Tiff => "Pyramidal TIFF",
            Self::OmeTiff => "OME-TIFF",
        }
    }
}

/// Tile compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlideCompression {
    /// Baseline JPEG, `quality` 1–100.
    Jpeg { quality: u8 },
    /// Lossless Deflate.
    Deflate,
}

impl Default for SlideCompression {
    fn default() -> Self {
        Self::Jpeg {
            quality: DEFAULT_JPEG_QUALITY,
        }
    }
}

/// The resolution of the copy's full-resolution level.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputResolution {
    /// The slide's level 0.
    #[default]
    Full,
    /// A stored pyramid level, at its native pixels.
    Level(u32),
    /// This many micrometres per pixel: a stored level within
    /// [`LEVEL_MATCH_TOLERANCE`], else resampled with Lanczos-3 from the
    /// next finer stored level.
    Mpp(f64),
}

/// What [`convert_slide`] writes.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOptions {
    /// Rectangle to copy, in level-0 pixels. `None` copies the area that
    /// holds image data (the slide's bounds, else the whole slide).
    pub region: Option<SlideBounds>,
    pub resolution: OutputResolution,
    pub compression: SlideCompression,
    /// Tile width and height: a multiple of 16 from 64 to 2048 (the
    /// command line offers 256 and 512).
    pub tile_size: u32,
    pub format: SlideFormat,
    /// Leave out associated images and every property not on the
    /// allow-list (see the module documentation).
    pub deidentify: bool,
    /// With `deidentify`, still record the scanner vendor and model.
    pub keep_scanner: bool,
    /// Copy the slide's ICC colour profile.
    pub keep_icc_profile: bool,
    /// Write a BigTIFF even when a classic TIFF would do.
    pub bigtiff: bool,
    /// Replace an existing output file.
    pub overwrite: bool,
    /// Pixel size (µm per level-0 pixel, x and y) to use instead of the
    /// slide's, e.g. a manual calibration.
    pub mpp_override: Option<(f64, f64)>,
    /// Threads that read and compress tiles; `None` uses the global rayon
    /// pool (every core). A viewer leaves some cores to itself.
    pub threads: Option<usize>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            region: None,
            resolution: OutputResolution::Full,
            compression: SlideCompression::default(),
            tile_size: DEFAULT_TILE_SIZE,
            format: SlideFormat::Tiff,
            deidentify: false,
            keep_scanner: false,
            keep_icc_profile: true,
            bigtiff: false,
            overwrite: false,
            mpp_override: None,
            threads: None,
        }
    }
}

/// Why a conversion failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConvertError {
    /// The options cannot work for this slide or output.
    #[error("{0}")]
    Invalid(String),
    /// The output exists and [`ConvertOptions::overwrite`] is off.
    #[error("{} already exists", .0.display())]
    OutputExists(PathBuf),
    #[error("cancelled")]
    Cancelled,
    /// Reading the slide failed.
    #[error("{0}")]
    Slide(#[from] crate::Error),
    /// Writing the copy failed.
    #[error("{0}")]
    Io(#[from] io::Error),
}

/// What a conversion will write, worked out without writing anything.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertPlan {
    /// The rectangle copied, in level-0 pixels.
    pub region: SlideBounds,
    /// The stored level the pixels come from.
    pub source_level: u32,
    /// Whether they are resampled (a pixel size between stored levels).
    pub resampled: bool,
    /// Sizes of the copy's levels, full resolution first.
    pub levels: Vec<(u32, u32)>,
    pub tile_size: u32,
    /// µm per pixel of the copy's full-resolution level.
    pub mpp: Option<(f64, f64)>,
    /// Objective-equivalent magnification of the full-resolution level.
    pub objective_power: Option<f64>,
    /// Tiles over all levels.
    pub tiles: u64,
    /// Associated images that will be copied (none when de-identifying).
    pub associated_images: Vec<String>,
    pub format: SlideFormat,
    mapping: Mapping,
    sampling: SourceSampling,
}

impl ConvertPlan {
    /// Uncompressed size of all tiles, in bytes.
    pub fn raw_bytes(&self) -> u64 {
        self.tiles * u64::from(self.tile_size) * u64::from(self.tile_size) * 3
    }
}

/// What [`convert_slide`] wrote.
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub path: PathBuf,
    pub plan: ConvertPlan,
    /// Size of the file.
    pub bytes: u64,
    pub bigtiff: bool,
    /// Tiles whose slide data could not be read; they hold the slide's
    /// background colour.
    pub unreadable_tiles: u64,
    /// Associated images copied.
    pub associated_images: Vec<String>,
    /// Whether the ICC profile was copied.
    pub icc_profile: bool,
    /// Things that did not stop the copy (an unreadable associated image or
    /// ICC profile).
    pub warnings: Vec<String>,
    pub elapsed: Duration,
}

impl ConvertReport {
    /// A warning for the user when the copy is incomplete.
    pub fn shortfall_message(&self) -> Option<String> {
        (self.unreadable_tiles > 0).then(|| {
            format!(
                "{} of {} tiles could not be read from the slide and show its background colour",
                self.unreadable_tiles, self.plan.tiles
            )
        })
    }
}

/// The objective power an eov copy records in its description
/// (`Objective power = 20`), for slides whose reader only exposes the
/// description: OpenSlide's generic TIFF driver reports the pixel size of a
/// copy but no `openslide.objective-power`. `None` for descriptions eov did
/// not write.
pub fn objective_power_from_description(description: &str) -> Option<f64> {
    let mut lines = description.lines();
    let headline = lines.next()?;
    if !(headline.starts_with("Slide copy made by eov ")
        || headline.starts_with("De-identified slide copy made by eov "))
    {
        return None;
    }
    lines
        .take_while(|line| *line != "Source properties:")
        .find_map(|line| line.strip_prefix("Objective power = "))
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

/// The temporary file a conversion to `output` writes before renaming it:
/// `<name>.<process id>.eov-partial` in the same folder.
pub fn partial_path(output: &Path) -> PathBuf {
    let mut name = output
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "copy".into());
    name.push(format!(".{}.eov-partial", std::process::id()));
    output.with_file_name(name)
}

/// The pixel size of `wsi`'s level 0 (both axes), from `mpp_override` or
/// the slide; one recorded axis is used for both.
fn level0_mpp(wsi: &WsiFile, options: &ConvertOptions) -> Option<(f64, f64)> {
    let valid = |v: f64| v.is_finite() && v > 0.0;
    if let Some((x, y)) = options.mpp_override
        && valid(x)
        && valid(y)
    {
        return Some((x, y));
    }
    let props = wsi.properties();
    match (props.mpp_x, props.mpp_y) {
        (Some(x), Some(y)) => Some((x, y)),
        (Some(v), None) | (None, Some(v)) => Some((v, v)),
        (None, None) => None,
    }
}

fn invalid(message: impl Into<String>) -> ConvertError {
    ConvertError::Invalid(message.into())
}

/// Works out what [`convert_slide`] will write, and checks the options.
pub fn plan_conversion(
    wsi: &WsiFile,
    options: &ConvertOptions,
) -> Result<ConvertPlan, ConvertError> {
    let wsi = wsi.without_synthetic_levels();
    let props = wsi.properties();
    let tile = options.tile_size;
    if !(64..=2048).contains(&tile) || !tile.is_multiple_of(16) {
        return Err(invalid(format!(
            "tile size {tile} is not a multiple of 16 between 64 and 2048"
        )));
    }
    if let SlideCompression::Jpeg { quality } = options.compression
        && !(1..=100).contains(&quality)
    {
        return Err(invalid(format!(
            "JPEG quality {quality} is not between 1 and 100"
        )));
    }
    let full = SlideBounds::full(props.width, props.height);
    let region = match options.region {
        Some(region) if region.is_empty() => return Err(invalid("the region is empty")),
        Some(region) => region.intersect(&full).ok_or_else(|| {
            invalid(format!(
                "the region {},{},{},{} is outside the slide ({} × {} pixels)",
                region.x, region.y, region.width, region.height, props.width, props.height
            ))
        })?,
        None => props.content_bounds().intersect(&full).unwrap_or(full),
    };
    let mpp0 = level0_mpp(&wsi, options);
    let level_mpp = |index: usize| -> Option<(f64, f64)> {
        let level = &props.levels[index];
        mpp0.map(|(x, y)| {
            (
                x * props.width as f64 / level.width as f64,
                y * props.height as f64 / level.height as f64,
            )
        })
    };

    let native = |index: usize| {
        let level = &props.levels[index];
        let ds = level.downsample;
        let x0 = (region.x as f64 / ds).floor() as i64;
        let y0 = (region.y as f64 / ds).floor() as i64;
        let x1 = ((region.right() as f64 / ds).ceil() as i64).min(level.width as i64);
        let y1 = ((region.bottom() as f64 / ds).ceil() as i64).min(level.height as i64);
        (
            Mapping::Native {
                level: index as u32,
                downsample: ds,
                origin: (x0, y0),
            },
            (x1 - x0).max(1) as u64,
            (y1 - y0).max(1) as u64,
            level_mpp(index),
            SourceSampling::Level {
                level: index as u32,
                downsample: ds,
            },
            ds,
        )
    };

    let (mapping, width, height, mpp, sampling, downsample) = match options.resolution {
        OutputResolution::Full => native(0),
        OutputResolution::Level(level) => {
            if level as usize >= props.levels.len() {
                return Err(invalid(format!(
                    "the slide has no level {level}; its levels are 0 to {}",
                    props.levels.len() - 1
                )));
            }
            native(level as usize)
        }
        OutputResolution::Mpp(target) => {
            if !target.is_finite() || target <= 0.0 {
                return Err(invalid(format!("{target} is not a pixel size in µm")));
            }
            let Some((mx, my)) = mpp0 else {
                return Err(invalid(
                    "the slide does not record its pixel size, so a target pixel size cannot be \
                     used; choose a level instead",
                ));
            };
            let (fx, fy) = (target / mx, target / my);
            if fx.min(fy) < 1.0 - 1e-6 {
                return Err(invalid(format!(
                    "{target} µm/px is finer than the slide's full resolution ({} µm/px); copies \
                     are never upsampled",
                    metadata_number(mx.min(my))
                )));
            }
            let matching = (0..props.levels.len()).find(|&index| {
                level_mpp(index).is_some_and(|(x, y)| {
                    ((x + y) / 2.0 / target - 1.0).abs() <= LEVEL_MATCH_TOLERANCE
                })
            });
            match matching {
                Some(index) => native(index),
                None => {
                    let index = best_level_for_downsample(&props.levels, fx.min(fy)) as usize;
                    let level = &props.levels[index];
                    let ds = level.downsample;
                    let x0 = region.x as f64 / ds;
                    let y0 = region.y as f64 / ds;
                    let extent = (
                        x0.floor() as i64,
                        y0.floor() as i64,
                        ((region.right() as f64 / ds).ceil() as i64).min(level.width as i64),
                        ((region.bottom() as f64 / ds).ceil() as i64).min(level.height as i64),
                    );
                    (
                        Mapping::Resampled {
                            level: index as u32,
                            downsample: ds,
                            scale: (fx / ds, fy / ds),
                            origin: (x0, y0),
                            extent,
                        },
                        (region.width as f64 / fx).ceil().max(1.0) as u64,
                        (region.height as f64 / fy).ceil().max(1.0) as u64,
                        Some((target, target)),
                        SourceSampling::Resampled {
                            downsample: (fx + fy) / 2.0,
                        },
                        (fx + fy) / 2.0,
                    )
                }
            }
        }
    };
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(invalid(format!(
            "the copy would be {width} × {height} pixels; at most {MAX_SIDE} per side"
        )));
    }
    if width < MIN_SIDE || height < MIN_SIDE {
        return Err(invalid(format!(
            "the copy would be {width} × {height} pixels; it must be at least {MIN_SIDE} × \
             {MIN_SIDE} (choose a larger region or a finer resolution)"
        )));
    }
    let (width, height) = (width as u32, height as u32);
    let geometry = match options.compression {
        SlideCompression::Jpeg { .. } => Geometry::new(width, height, tile),
        // See `Geometry::whole_tiles`: no level of a lossless copy may be
        // smaller than a tile, so small copies get smaller tiles.
        SlideCompression::Deflate => {
            let fits = (width.min(height) / 16 * 16).max(16);
            Geometry::whole_tiles(width, height, tile.min(fits))
        }
    };
    let tile = geometry.tile;
    let objective_power = props.objective_power.map(|power| match (mpp0, mpp) {
        (Some((x0, y0)), Some((x, y))) => power * (x0 + y0) / (x + y),
        _ => power / downsample,
    });
    let associated_images = if options.deidentify {
        Vec::new()
    } else {
        wsi.associated_images()
            .iter()
            .map(|image| image.name.clone())
            .collect()
    };
    Ok(ConvertPlan {
        region,
        source_level: mapping.level(),
        resampled: matches!(mapping, Mapping::Resampled { .. }),
        tiles: geometry.total_tiles(),
        levels: geometry.levels,
        tile_size: tile,
        mpp,
        objective_power,
        associated_images,
        format: options.format,
        mapping,
        sampling,
    })
}

fn metadata_number(value: f64) -> String {
    crate::format_decimal(value)
}

/// The canonical form of a path that may not exist yet.
fn canonical_target(path: &Path) -> PathBuf {
    if let Ok(canonical) = fs::canonicalize(path) {
        return canonical;
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    match (fs::canonicalize(parent), path.file_name()) {
        (Ok(parent), Some(name)) => parent.join(name),
        _ => path.to_path_buf(),
    }
}

/// Refuses outputs that cannot or must not be written: a folder, a missing
/// folder, the slide itself (or one of its files), or an existing file
/// without `overwrite`. [`convert_slide`] checks this too; call it to fail
/// early.
pub fn check_output(wsi: &WsiFile, output: &Path, overwrite: bool) -> Result<(), ConvertError> {
    if output.file_name().is_none() {
        return Err(invalid(format!("{} is not a file name", output.display())));
    }
    if output.is_dir() {
        return Err(invalid(format!("{} is a folder", output.display())));
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "the folder {} does not exist",
            parent.display()
        )));
    }
    let target = canonical_target(output);
    let props = wsi.properties();
    let mut sources = vec![props.path.clone()];
    if let Ok(members) = crate::file_id::slide_member_files(&props.path, props.vendor.as_deref()) {
        sources.extend(members);
    }
    if sources
        .iter()
        .any(|source| canonical_target(source) == target)
    {
        return Err(invalid(format!(
            "{} is the slide being copied (or one of its files)",
            output.display()
        )));
    }
    if output.exists() && !overwrite {
        return Err(ConvertError::OutputExists(output.to_path_buf()));
    }
    Ok(())
}

/// The temporary output: removed unless [`persist`](Self::persist)ed.
struct PartialFile {
    path: PathBuf,
    file: Option<File>,
}

impl PartialFile {
    fn create(path: &Path) -> Result<Self, ConvertError> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|err| {
                if err.kind() == io::ErrorKind::AlreadyExists {
                    invalid(format!(
                        "{} exists: another conversion is writing this file, or an interrupted \
                         one left it behind (delete it and try again)",
                        path.display()
                    ))
                } else {
                    ConvertError::Io(err)
                }
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
        })
    }

    fn file(&self) -> &File {
        self.file.as_ref().expect("open until persisted")
    }

    fn persist(mut self, to: &Path) -> io::Result<()> {
        if let Some(file) = self.file.take() {
            file.sync_all()?;
        }
        fs::rename(&self.path, to)?;
        self.path = PathBuf::new();
        Ok(())
    }
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        self.file = None;
        if !self.path.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Writes a copy of `wsi` to `output` as `options` say, reporting
/// progress (in tiles) and honouring cancellation through `control`.
///
/// Run it on a worker thread; tiles are read and compressed on the rayon
/// pool. On error or cancellation nothing is left at `output` (an
/// existing file there is only replaced on success).
pub fn convert_slide(
    wsi: &WsiFile,
    output: &Path,
    options: &ConvertOptions,
    control: &JobControl,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let wsi = wsi.without_synthetic_levels();
    let plan = plan_conversion(&wsi, options)?;
    check_output(&wsi, output, options.overwrite)?;
    control.set_total(plan.tiles + plan.associated_images.len() as u64);

    let partial = PartialFile::create(&partial_path(output))?;
    let written = match options.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|index| format!("eov-convert-{index}"))
            .build()
            .map_err(|err| ConvertError::Io(io::Error::other(err)))?
            .install(|| write_copy(&wsi, &plan, options, partial.file(), control))?,
        None => write_copy(&wsi, &plan, options, partial.file(), control)?,
    };
    if control.is_cancelled() {
        return Err(ConvertError::Cancelled);
    }
    if !options.overwrite && output.exists() {
        return Err(ConvertError::OutputExists(output.to_path_buf()));
    }
    partial.persist(output)?;
    Ok(ConvertReport {
        path: output.to_path_buf(),
        plan,
        bytes: written.bytes,
        bigtiff: written.bigtiff,
        unreadable_tiles: written.unreadable_tiles,
        associated_images: written.associated_images,
        icc_profile: written.icc_profile,
        warnings: written.warnings,
        elapsed: started.elapsed(),
    })
}

struct Written {
    bytes: u64,
    bigtiff: bool,
    unreadable_tiles: u64,
    associated_images: Vec<String>,
    icc_profile: bool,
    warnings: Vec<String>,
}

const NEW_SUBFILE_TYPE: u16 = 254;
const IMAGE_WIDTH: u16 = 256;
const IMAGE_LENGTH: u16 = 257;
const BITS_PER_SAMPLE: u16 = 258;
const COMPRESSION: u16 = 259;
const PHOTOMETRIC: u16 = 262;
const IMAGE_DESCRIPTION: u16 = 270;
const MAKE: u16 = 271;
const MODEL: u16 = 272;
const STRIP_OFFSETS: u16 = 273;
const SAMPLES_PER_PIXEL: u16 = 277;
const ROWS_PER_STRIP: u16 = 278;
const STRIP_BYTE_COUNTS: u16 = 279;
const X_RESOLUTION: u16 = 282;
const Y_RESOLUTION: u16 = 283;
const PLANAR_CONFIGURATION: u16 = 284;
const RESOLUTION_UNIT: u16 = 296;
const SOFTWARE: u16 = 305;
const PREDICTOR: u16 = 317;
const TILE_WIDTH: u16 = 322;
const TILE_LENGTH: u16 = 323;
const TILE_OFFSETS: u16 = 324;
const TILE_BYTE_COUNTS: u16 = 325;
const JPEG_TABLES: u16 = 347;
const YCBCR_SUBSAMPLING: u16 = 530;
const REFERENCE_BLACK_WHITE: u16 = 532;
const ICC_PROFILE: u16 = 34675;

/// `value` as a TIFF rational with the largest power-of-ten denominator
/// that keeps the numerator in 32 bits.
fn rational(value: f64) -> (u32, u32) {
    for exponent in (0..=6).rev() {
        let denominator = 10u32.pow(exponent);
        let numerator = (value * f64::from(denominator)).round();
        if numerator >= 1.0 && numerator <= f64::from(u32::MAX) {
            return (numerator as u32, denominator);
        }
    }
    (value.round().clamp(1.0, f64::from(u32::MAX)) as u32, 1)
}

/// Text for a TIFF `ASCII` field (7-bit, transliterated).
fn ascii(text: &str) -> Value {
    let mut bytes = crate::export::tiff::ascii_value(text);
    bytes.pop(); // the NUL, which the writer adds
    Value::Ascii(bytes)
}

/// The tags every RGB image shares.
fn rgb_ifd(width: u32, height: u32, reduced: bool, software: &str) -> Ifd {
    let mut ifd = Ifd::default();
    ifd.set(NEW_SUBFILE_TYPE, Value::Long(vec![u32::from(reduced)]));
    ifd.set(IMAGE_WIDTH, Value::Long(vec![width]));
    ifd.set(IMAGE_LENGTH, Value::Long(vec![height]));
    ifd.set(BITS_PER_SAMPLE, Value::Short(vec![8, 8, 8]));
    ifd.set(SAMPLES_PER_PIXEL, Value::Short(vec![3]));
    ifd.set(PLANAR_CONFIGURATION, Value::Short(vec![1]));
    ifd.set(SOFTWARE, ascii(software));
    ifd
}

/// The IFD of pyramid level `level`.
fn level_ifd(
    level: usize,
    geometry: &Geometry,
    tiles: &LevelTiles,
    codec: &TileCodec,
    mpp: Option<(f64, f64)>,
    software: &str,
) -> Ifd {
    let (width, height) = geometry.levels[level];
    let mut ifd = rgb_ifd(width, height, level > 0, software);
    match codec {
        TileCodec::Jpeg(encoder) => {
            ifd.set(COMPRESSION, Value::Short(vec![7]));
            ifd.set(PHOTOMETRIC, Value::Short(vec![6]));
            ifd.set(JPEG_TABLES, Value::Undefined(encoder.tables()));
            ifd.set(
                YCBCR_SUBSAMPLING,
                Value::Short(encoder.ycbcr_subsampling().to_vec()),
            );
            ifd.set(
                REFERENCE_BLACK_WHITE,
                Value::Rational(vec![
                    (0, 1),
                    (255, 1),
                    (128, 1),
                    (255, 1),
                    (128, 1),
                    (255, 1),
                ]),
            );
        }
        TileCodec::Deflate => {
            ifd.set(COMPRESSION, Value::Short(vec![8]));
            ifd.set(PHOTOMETRIC, Value::Short(vec![2]));
            ifd.set(PREDICTOR, Value::Short(vec![2]));
        }
    }
    if let Some((mpp_x, mpp_y)) = mpp {
        let (w0, h0) = geometry.levels[0];
        // Pixels per centimetre of this level.
        let x = 10_000.0 / (mpp_x * f64::from(w0) / f64::from(width));
        let y = 10_000.0 / (mpp_y * f64::from(h0) / f64::from(height));
        ifd.set(X_RESOLUTION, Value::Rational(vec![rational(x)]));
        ifd.set(Y_RESOLUTION, Value::Rational(vec![rational(y)]));
        ifd.set(RESOLUTION_UNIT, Value::Short(vec![3]));
    }
    ifd.set(TILE_WIDTH, Value::Long(vec![geometry.tile]));
    ifd.set(TILE_LENGTH, Value::Long(vec![geometry.tile]));
    ifd.set(TILE_OFFSETS, Value::Offsets(tiles.offsets.clone()));
    ifd.set(TILE_BYTE_COUNTS, Value::Offsets(tiles.counts.clone()));
    ifd
}

/// Writes an associated image as Deflate strips and returns its IFD.
fn write_associated<W: io::Write + io::Seek>(
    file: &mut TiffFile<W>,
    name: &str,
    image: &crate::RgbaImageData,
    software: &str,
) -> io::Result<Ifd> {
    let (width, height) = (image.width as u32, image.height as u32);
    let stride = image.width * 3;
    let rows_per_strip = ((256 * 1024) / stride.max(1)).clamp(1, image.height.max(1));
    let rgb: Vec<u8> = image
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    let mut offsets = Vec::new();
    let mut counts = Vec::new();
    for strip in rgb.chunks(rows_per_strip * stride) {
        let bytes = deflate_rows(strip, image.width)?;
        offsets.push(file.append(&bytes)?);
        counts.push(bytes.len() as u64);
    }
    let mut ifd = rgb_ifd(width, height, false, software);
    ifd.set(COMPRESSION, Value::Short(vec![8]));
    ifd.set(PHOTOMETRIC, Value::Short(vec![2]));
    ifd.set(PREDICTOR, Value::Short(vec![2]));
    ifd.set(IMAGE_DESCRIPTION, ascii(name));
    ifd.set(STRIP_OFFSETS, Value::Offsets(offsets));
    ifd.set(ROWS_PER_STRIP, Value::Long(vec![rows_per_strip as u32]));
    ifd.set(STRIP_BYTE_COUNTS, Value::Offsets(counts));
    Ok(ifd)
}

fn write_copy(
    wsi: &WsiFile,
    plan: &ConvertPlan,
    options: &ConvertOptions,
    file: &File,
    control: &JobControl,
) -> Result<Written, ConvertError> {
    let mut tiff = TiffFile::new(BufWriter::with_capacity(4 << 20, file))?;
    let software = crate::export::software_name();
    let props = wsi.properties();
    let mut warnings = Vec::new();

    // Associated images (never for de-identified copies: plan has none).
    let mut associated = Vec::new();
    for name in &plan.associated_images {
        if control.is_cancelled() {
            return Err(ConvertError::Cancelled);
        }
        match wsi.read_associated_image(name) {
            Ok(image) if image.width > 0 && image.height > 0 => {
                let ifd = write_associated(&mut tiff, name, &image, &software)?;
                associated.push((name.clone(), image.width as u32, image.height as u32, ifd));
            }
            Ok(_) => {}
            Err(err) => warnings.push(format!("the {name} image was not copied: {err}")),
        }
        control.advance(1);
    }

    // The pyramid.
    let geometry = Geometry {
        tile: plan.tile_size,
        levels: plan.levels.clone(),
    };
    let tiles = SlideTiles {
        wsi: wsi.clone(),
        mapping: plan.mapping.clone(),
        tile: plan.tile_size,
        width: geometry.levels[0].0,
        height: geometry.levels[0].1,
        background: props.background_color,
    };
    let codec = match options.compression {
        SlideCompression::Jpeg { quality } => pyramid::jpeg_codec(quality),
        SlideCompression::Deflate => TileCodec::Deflate,
    };
    let encoder = TileEncoder::new(codec, plan.tile_size);
    let pyramid = write_pyramid(&mut tiff, &geometry, &tiles, &encoder, control)?;

    // Metadata.
    let facts = SlideFacts {
        properties: wsi.property_map(),
        file_name: &props.filename,
        stain: props.stain.as_deref(),
        vendor: props.vendor.as_deref(),
    };
    let metadata = CopyMetadata::new(
        &facts,
        options.deidentify,
        options.keep_scanner,
        plan.mpp,
        plan.objective_power,
        plan.region,
        plan.sampling.clone(),
        SystemTime::now(),
    );
    let icc = if options.keep_icc_profile {
        match wsi.icc_profile() {
            Ok(profile) => profile.filter(|profile| !profile.is_empty()),
            Err(err) => {
                warnings.push(format!("the ICC profile was not copied: {err}"));
                None
            }
        }
    } else {
        None
    };

    let mut levels: Vec<Ifd> = pyramid
        .levels
        .iter()
        .enumerate()
        .map(|(level, tiles)| {
            level_ifd(
                level,
                &geometry,
                tiles,
                encoder.codec(),
                plan.mpp,
                &software,
            )
        })
        .collect();
    let first = &mut levels[0];
    let description = match options.format {
        SlideFormat::Tiff => ascii(&metadata.description()),
        SlideFormat::OmeTiff => {
            let name = if options.deidentify {
                "slide".to_string()
            } else {
                Path::new(&props.filename)
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "slide".to_string())
            };
            let mut images = vec![(name, geometry.levels[0].0, geometry.levels[0].1)];
            images.extend(
                associated
                    .iter()
                    .map(|(name, w, h, _)| (name.clone(), *w, *h)),
            );
            // OME-XML is ASCII already (character references).
            Value::Ascii(
                metadata
                    .ome_xml(&images, &metadata::random_uuid())
                    .into_bytes(),
            )
        }
    };
    first.set(IMAGE_DESCRIPTION, description);
    if let Some(make) = &metadata.make {
        first.set(MAKE, ascii(make));
    }
    if let Some(model) = &metadata.model {
        first.set(MODEL, ascii(model));
    }
    if let Some(profile) = &icc {
        first.set(ICC_PROFILE, Value::Undefined(profile.clone()));
    }
    let chain: Vec<Ifd> = match options.format {
        SlideFormat::Tiff => levels
            .into_iter()
            .chain(associated.iter().map(|(_, _, _, ifd)| ifd.clone()))
            .collect(),
        SlideFormat::OmeTiff => {
            let mut levels = levels.into_iter();
            let mut first = levels.next().expect("a pyramid has a level");
            first.sub_ifds = levels.collect();
            std::iter::once(first)
                .chain(associated.iter().map(|(_, _, _, ifd)| ifd.clone()))
                .collect()
        }
    };
    let (writer, bigtiff) = tiff.finish(&chain, options.bigtiff)?;
    let file = writer.into_inner().map_err(|err| err.into_error())?;
    Ok(Written {
        bytes: file.metadata()?.len(),
        bigtiff,
        unreadable_tiles: pyramid.unreadable_tiles,
        associated_images: associated.into_iter().map(|(name, ..)| name).collect(),
        icc_profile: icc.is_some(),
        warnings,
    })
}
