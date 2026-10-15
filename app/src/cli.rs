use crate::backend::WindowGeometry;
use crate::config;
use crate::launch::{OpenRequest, Point, ViewRequest, Zoom};
use crate::update;
use anyhow::{Result, bail};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use eov_common::{
    FilteringMode, RenderBackend,
    dataset::{self, DatasetPatchesConfig, MetadataFormat},
};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Largest `--cache-size`, in MiB (1 TiB).
const MAX_CACHE_MIB: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum CliBackend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl CliBackend {
    fn render_backend_override(self) -> Option<RenderBackend> {
        match self {
            Self::Auto => None,
            Self::Cpu => Some(RenderBackend::Cpu),
            Self::Gpu => Some(RenderBackend::Gpu),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum CliFilteringMode {
    #[default]
    Auto,
    Bilinear,
    Trilinear,
    Lanczos,
}

impl CliFilteringMode {
    fn filtering_mode_override(self) -> Option<FilteringMode> {
        match self {
            Self::Auto => None,
            Self::Bilinear => Some(FilteringMode::Bilinear),
            Self::Trilinear => Some(FilteringMode::Trilinear),
            Self::Lanczos => Some(FilteringMode::Lanczos3),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum CliLogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl CliLogLevel {
    pub(crate) fn as_filter(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

#[derive(Debug, Subcommand)]
enum RecentCommand {
    /// Print the recently opened slides and folders, newest first.
    List,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliMetadataFormat {
    /// metadata.csv only.
    Csv,
    /// metadata.jsonl (written as tiles complete) and metadata.json (an array, written at the end).
    Json,
    /// metadata.jsonl only.
    Jsonl,
}

impl CliMetadataFormat {
    fn to_common(self) -> MetadataFormat {
        match self {
            Self::Csv => MetadataFormat::Csv,
            Self::Json => MetadataFormat::Json,
            Self::Jsonl => MetadataFormat::JsonLines,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliTissueMethod {
    /// Otsu threshold on the slide's saturation histogram.
    Otsu,
    /// A fixed saturation threshold (--tissue-threshold).
    Saturation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliStainMethod {
    Macenko,
    Vahadane,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliMaskKind {
    /// 255 inside selected annotations, 0 elsewhere.
    Binary,
    /// Class index per pixel (see classes.json; 0 = background).
    Multiclass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliPatchFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Subcommand)]
enum DatasetCommand {
    /// Extract image patches from whole-slide images.
    ///
    /// Reads one or more slide files (or directories of slide files) and
    /// writes a deterministic grid of RGB patches to the output directory.
    /// Only full patches are emitted. Metadata is written as patches
    /// complete, so an interrupted run can be resumed by running the same
    /// command again.
    ///
    /// Supported slide formats: .svs, .tif, .tiff, .ndpi, .vms, .vmu,
    /// .scn, .mrxs, .svslide, .bif, .czi, .dcm (DICOM instances of one
    /// series are treated as one slide).
    ///
    /// Examples:
    ///   eov dataset patches slide.svs --out ds/ --tile-size 512 --stride 512
    ///   eov dataset patches slides/ --out ds/ --tile-size 256 --stride 256 --target-mpp 0.5 --min-tissue 0.5
    ///   eov dataset patches a.svs --out ds/ --tile-size 512 --stride 256 --annotations sidecar --include-class Tumor --masks binary
    ///   eov dataset patches slides/ --out ds/ --tile-size 224 --stride 224 --max-tiles-per-slide 500 --seed 1 --normalize macenko
    #[command(after_help = "OUTPUT LAYOUT:\n  \
    ds/\n    \
      slides/<slide-id>/<slide-id>_x000000_y000000_s512.png   (X, Y = level-0 origin)\n    \
      masks/<slide-id>/...png                                (with --masks)\n    \
      metadata.csv, metadata.jsonl, metadata.json             (one record per patch)\n    \
      slides.csv, slides.jsonl                                (one record per slide)\n    \
      dataset.json                                            (parameters, status, totals)\n    \
      classes.json                                            (with annotations)\n\n  \
  If --metadata is given, only that format is written; otherwise CSV and JSON are.\n\n\
EXIT STATUS:\n  \
  0    every slide was processed\n  \
  1    some slides or patches failed (see the summary and slides.csv)\n  \
  2    invalid arguments, no slides found, or an incompatible output directory\n  \
  130  cancelled with Ctrl-C (metadata so far is kept; re-run to resume)")]
    Patches(Box<DatasetPatchesArgs>),
}

#[derive(Debug, clap::Args)]
struct DatasetPatchesArgs {
    /// One or more slide files or directories containing slides.
    /// Directories are searched recursively for supported slide formats.
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Output directory for patches and metadata.
    #[arg(long, required = true)]
    out: PathBuf,

    /// Patch width and height in output pixels (8–8192).
    #[arg(long, required = true, value_parser = clap::value_parser!(u32).range(8..=8192))]
    tile_size: u32,

    /// Step between patch origins in output pixels (1–65536).
    #[arg(long, required = true, value_parser = clap::value_parser!(u32).range(1..=65536))]
    stride: u32,

    /// Write per-patch metadata in only this format.
    #[arg(long, value_enum)]
    metadata: Option<CliMetadataFormat>,

    /// Worker threads (1–512). Defaults to the number of CPU cores.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=512))]
    threads: Option<u32>,

    /// Skip patches whose fraction of nearly-white or transparent pixels is
    /// at least this value, in (0, 1]. For example, 0.9 skips patches that
    /// are 90 % white or more. Omit to keep all patches.
    #[arg(long)]
    white_threshold: Option<f32>,

    /// Extract at this resolution in micrometres per pixel (read from the
    /// best pyramid level and resampled).
    #[arg(long, conflicts_with = "target_magnification")]
    target_mpp: Option<f64>,

    /// Extract at this objective-equivalent magnification (e.g. 20).
    #[arg(long)]
    target_magnification: Option<f64>,

    /// Keep only patches with at least this fraction of tissue, in [0, 1]
    /// (enables tissue detection).
    #[arg(long)]
    min_tissue: Option<f32>,

    /// How tissue is detected.
    #[arg(long, value_enum, default_value_t = CliTissueMethod::Otsu)]
    tissue_method: CliTissueMethod,

    /// Saturation threshold in (0, 1) for --tissue-method saturation.
    #[arg(long, default_value_t = 0.08)]
    tissue_threshold: f32,

    /// Stain-normalise patches (one slide-level fit per slide).
    #[arg(long, value_enum)]
    normalize: Option<CliStainMethod>,

    /// GeoJSON annotations (QuPath FeatureCollection, level-0 pixels): a
    /// file, a directory with <stem>.geojson per slide, or "sidecar" for
    /// files next to the slides. Patches are restricted to annotations
    /// unless --label-all is given.
    #[arg(long)]
    annotations: Option<String>,

    /// Annotation classes to use (repeatable; default: all).
    #[arg(long = "include-class")]
    include_class: Vec<String>,

    /// Annotation classes to ignore (repeatable).
    #[arg(long = "exclude-class")]
    exclude_class: Vec<String>,

    /// Minimum fraction of a patch inside selected annotations, in (0, 1].
    #[arg(long, default_value_t = 0.5)]
    min_coverage: f32,

    /// Keep every patch (not only annotated ones) and label it with the
    /// majority annotation class.
    #[arg(long)]
    label_all: bool,

    /// Write a mask image per patch.
    #[arg(long, value_enum)]
    masks: Option<CliMaskKind>,

    /// Write at most this many patches per slide.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    max_tiles_per_slide: Option<u64>,

    /// Sample patch positions randomly with this seed (with
    /// --max-tiles-per-slide); otherwise positions are spread evenly.
    #[arg(long, requires = "max_tiles_per_slide")]
    seed: Option<u64>,

    /// Patch image format (always RGB).
    #[arg(long, value_enum, default_value_t = CliPatchFormat::Png)]
    format: CliPatchFormat,

    /// JPEG quality (1–100).
    #[arg(long, default_value_t = 90, value_parser = clap::value_parser!(u8).range(1..=100))]
    jpeg_quality: u8,

    /// Replace an existing dataset in the output directory instead of
    /// resuming it.
    #[arg(long)]
    overwrite: bool,

    /// Skip the full-file SHA-256 of each slide.
    #[arg(long)]
    no_hash: bool,

    /// Refuse slides with more candidate positions than this.
    #[arg(long, default_value_t = 10_000_000, value_parser = clap::value_parser!(u64).range(1..))]
    max_candidates: u64,

    /// Plan only: open the slides and report the patch counts without
    /// writing anything.
    #[arg(long)]
    dry_run: bool,

    /// Name slides by opaque ids derived from their content, and write no
    /// slide paths or file names, for datasets shared outside the
    /// institution.
    #[arg(long)]
    deidentify: bool,
}

/// A parsed `eov dataset patches` invocation.
struct DatasetJob {
    config: DatasetPatchesConfig,
    options: dataset::DatasetPatchesOptions,
}

impl DatasetPatchesArgs {
    fn into_job(self) -> DatasetJob {
        let threads = self.threads.map(|n| n as usize).unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(dataset::THREADS_RANGE.1)
        });
        let resolution = match (self.target_mpp, self.target_magnification) {
            (Some(mpp), _) => Some(dataset::TargetResolution::Mpp(mpp)),
            (None, Some(mag)) => Some(dataset::TargetResolution::Magnification(mag)),
            (None, None) => None,
        };
        let tissue_filter = self.min_tissue.map(|min_fraction| dataset::TissueFilter {
            method: match self.tissue_method {
                CliTissueMethod::Otsu => eov_common::tissue::TissueMethod::Otsu,
                CliTissueMethod::Saturation => eov_common::tissue::TissueMethod::Saturation {
                    threshold: self.tissue_threshold,
                },
            },
            min_fraction,
            ..dataset::TissueFilter::default()
        });
        let annotations = self.annotations.map(|source| {
            let source = if source.eq_ignore_ascii_case("sidecar") {
                dataset::AnnotationSource::Sidecar
            } else {
                let path = PathBuf::from(source);
                if path.is_dir() {
                    dataset::AnnotationSource::Directory(path)
                } else {
                    dataset::AnnotationSource::File(path)
                }
            };
            dataset::AnnotationOptions {
                source,
                include_classes: self.include_class,
                exclude_classes: self.exclude_class,
                restrict_to_annotations: !self.label_all,
                min_coverage: self.min_coverage,
                masks: self.masks.map(|m| match m {
                    CliMaskKind::Binary => dataset::MaskOutput::Binary,
                    CliMaskKind::Multiclass => dataset::MaskOutput::Multiclass,
                }),
            }
        });
        DatasetJob {
            config: DatasetPatchesConfig {
                inputs: self.inputs,
                output_dir: self.out,
                tile_size: self.tile_size,
                stride: self.stride,
                metadata_format: self.metadata.map(|m| m.to_common()),
                threads,
                white_threshold: self.white_threshold,
            },
            options: dataset::DatasetPatchesOptions {
                resolution,
                tissue_filter,
                stain_normalization: self.normalize.map(|m| dataset::PatchNormalization {
                    method: match m {
                        CliStainMethod::Macenko => eov_common::stain::StainFitMethod::Macenko,
                        CliStainMethod::Vahadane => eov_common::stain::StainFitMethod::Vahadane,
                    },
                    target: None,
                }),
                annotations,
                max_tiles_per_slide: self.max_tiles_per_slide,
                seed: self.seed,
                image_format: match self.format {
                    CliPatchFormat::Png => dataset::PatchImageFormat::Png,
                    CliPatchFormat::Jpeg => dataset::PatchImageFormat::Jpeg {
                        quality: self.jpeg_quality,
                    },
                },
                existing: if self.overwrite {
                    dataset::ExistingOutput::Overwrite
                } else {
                    dataset::ExistingOutput::Resume
                },
                hash_files: !self.no_hash,
                max_candidates_per_slide: self.max_candidates,
                dry_run: self.dry_run,
                deidentify: self.deidentify,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum CliPluginOutputFormat {
    Plain,
    Toml,
    Json,
}

/// Options shared by plugin commands that install packages.
#[derive(Clone, Debug, clap::Args)]
pub(crate) struct PluginInstallArgs {
    /// An official plugin id (`gamepad`, `gamepad@1.0.1`), a GitHub
    /// repository (`github.com/owner/repo[@tag]`) or a local `.eop` file.
    source: String,
    /// Skip the confirmation prompt (network access still needs
    /// --allow-network or interactive consent).
    #[arg(short = 'y', long)]
    yes: bool,
    /// Allow GitHub requests without asking.
    #[arg(long)]
    allow_network: bool,
    /// The plugin id a GitHub release must contain (required when the
    /// release does not declare one).
    #[arg(long, value_name = "ID")]
    id: Option<String>,
    /// Replace a plugin installed from a different source, or install a
    /// third-party package that uses an official plugin id.
    #[arg(long)]
    force: bool,
    /// Show what would be installed without changing anything.
    #[arg(long)]
    dry_run: bool,
    /// Print machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Debug, Subcommand)]
pub(crate) enum PluginCommand {
    /// List installed plugins, compared with the official catalog when
    /// network access is allowed.
    #[command(alias = "list")]
    Ls {
        /// Fetch the official catalog without asking.
        #[arg(long)]
        allow_network: bool,
        /// Never contact GitHub; list installed plugins only.
        #[arg(long, conflicts_with = "allow_network")]
        offline: bool,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Install a plugin.
    #[command(alias = "add")]
    Install(PluginInstallArgs),
    /// Update an installed plugin from its recorded source.
    #[command(alias = "upgrade")]
    Update(PluginInstallArgs),
    /// Remove a plugin (every package with its id), or a broken package by
    /// file name or path.
    #[command(alias = "rm", alias = "uninstall")]
    Remove {
        /// A plugin id, or a package's file name or path.
        plugin: String,
        /// Do not ask for confirmation.
        #[arg(short = 'y', long)]
        yes: bool,
        /// Show what would be removed without changing anything.
        #[arg(long)]
        dry_run: bool,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show a plugin's metadata, provenance and compatibility.
    #[command(alias = "inspect")]
    Info {
        /// A plugin id, or a package's file name or path.
        plugin: String,
        /// Output format.
        #[arg(short = 'o', long, value_enum, default_value_t = CliPluginOutputFormat::Plain)]
        output: CliPluginOutputFormat,
    },
    /// Keep a plugin installed but do not load it.
    Disable {
        /// The plugin's id.
        id: String,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Load a previously disabled plugin again.
    Enable {
        /// The plugin's id.
        id: String,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Write a lockfile of the installed plugins (ids, versions, sources and
    /// SHA-256 hashes), e.g. `eov plugin lock > eov-plugins.lock`.
    Lock {
        /// Write to this file instead of standard output.
        #[arg(short = 'o', long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Install and replace plugins so they match a lockfile.
    Sync {
        /// A lockfile written by `eov plugin lock`.
        lockfile: PathBuf,
        /// Take packages with matching SHA-256 from this directory instead
        /// of downloading them.
        #[arg(long, value_name = "DIR")]
        from: Option<PathBuf>,
        /// Also remove installed plugins the lockfile does not list.
        #[arg(long)]
        prune: bool,
        /// Do not ask for confirmation (network access still needs
        /// --allow-network or interactive consent).
        #[arg(short = 'y', long)]
        yes: bool,
        /// Allow GitHub requests without asking.
        #[arg(long)]
        allow_network: bool,
        /// Show what would change without changing anything.
        #[arg(long)]
        dry_run: bool,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Delete unused extracted plugin packages from the cache now.
    Gc {
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Options of `eov update`.
#[derive(Clone, Debug, clap::Args)]
pub(crate) struct UpdateArgs {
    /// Install exactly this release (allows downgrades), e.g. 1.0.1.
    #[arg(long, value_name = "VERSION")]
    release: Option<String>,
    /// Update eov only, not its plugins.
    #[arg(long)]
    app_only: bool,
    /// Skip the confirmation prompt (network access still needs
    /// --allow-network or interactive consent).
    #[arg(short = 'y', long)]
    yes: bool,
    /// Allow GitHub requests without asking.
    #[arg(long)]
    allow_network: bool,
    /// Update from an offline bundle (a directory with release.toml and the
    /// release artifacts) instead of GitHub.
    #[arg(long, value_name = "DIR_OR_FILE", conflicts_with = "allow_network")]
    from: Option<PathBuf>,
    /// Trust an offline bundle whose release.toml is not signed with a
    /// trusted eov release key.
    #[arg(long, requires = "from")]
    allow_unsigned: bool,
    /// Restore the installation (and plugins) that the last update replaced.
    #[arg(long, conflicts_with_all = ["release", "from", "app_only"])]
    rollback: bool,
    /// Show the plan without changing anything.
    #[arg(long)]
    dry_run: bool,
    /// Print machine-readable JSON.
    #[arg(long)]
    json: bool,
    /// Do not start eov again after a helper-based update.
    #[arg(long)]
    no_relaunch: bool,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Update eov and its plugins (or roll back the last update).
    #[command(alias = "upgrade")]
    Update(UpdateArgs),
    /// Install, update, remove, enable, disable and inspect plugins.
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
    /// Print a slide's metadata.
    ///
    /// Size, pyramid levels, pixel size, objective, stain, scanned bounds
    /// and regions, ICC profile, associated images, quickhash and every
    /// OpenSlide property.
    Probe {
        /// The slide.
        file: PathBuf,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Recently opened slides and folders.
    Recent {
        #[command(subcommand)]
        command: RecentCommand,
    },
    /// Print the path of the configuration file in use.
    ConfigPath,
    /// List every file and folder eov keeps data in, with their sizes.
    Paths {
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Offer eov for slide files and eov:// links (this user only).
    ///
    /// Windows: writes HKEY_CURRENT_USER\Software\Classes. Linux (AppImage,
    /// cargo install): writes the MIME package, a desktop entry and an icon
    /// to ~/.local/share. The macOS app and the Flatpak declare them
    /// already.
    RegisterFileTypes {
        /// Show what would change without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Undo `eov register-file-types`.
    UnregisterFileTypes {
        /// Show what would change without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Dataset generation utilities for ML workflows.
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    /// Copy a slide, or a region of it, into a pyramidal TIFF or OME-TIFF,
    /// optionally de-identified.
    ///
    /// Reads any slide eov opens and writes a tiled pyramid (BigTIFF when it
    /// does not fit 4 GiB) that eov, OpenSlide, QuPath and libvips read, with
    /// the pixel size, objective power, stain and ICC profile of the slide,
    /// so measurements still work on the copy. The copy is written to a
    /// temporary file next to OUTPUT and renamed when it is complete:
    /// Ctrl-C or an error never leaves a half-written file.
    #[command(after_help = CONVERT_AFTER_HELP)]
    Convert(Box<ConvertArgs>),
}

const CONVERT_AFTER_HELP: &str = "\
EXAMPLES:
  eov convert slide.ndpi slide.tif
  eov convert slide.svs teaching/case-01.tif --deidentify
  eov convert slide.mrxs roi.ome.tif --region 20000,15000,8192,8192 --compression deflate
  eov convert slide.svs small.tif --mpp 1.0 --quality 80 --tile-size 256

DE-IDENTIFICATION:
  --deidentify leaves out the label, macro and thumbnail images and keeps
  only an allow-list of metadata: the geometry, pixel size, objective power,
  stain and (with --keep-scanner) the scanner vendor and model. Dates,
  barcodes, file names, users, comments, slide and device ids and all other
  vendor properties are dropped, and the description says it is a
  de-identified copy made by eov, with its version and the date. It does
  not remove text or marks inside the scanned tissue image itself (a label
  inside the scan area, pen marks, burned-in annotations): check the image,
  and choose a --region that leaves them out. Never use the original file
  name for the copy if the name identifies the patient.

OUTPUT:
  --format tiff (the default, or any name not ending in .ome.tif) writes each
  pyramid level as its own image, as OpenSlide's generic TIFF expects;
  --format ome-tiff writes OME-XML and a SubIFD pyramid for QuPath,
  Bio-Formats and OMERO (OpenSlide, and so eov 1.0, only sees its full
  resolution). JPEG tiles are YCbCr, chroma subsampled below quality 90;
  deflate is lossless.

EXIT STATUS:
  0    the copy was written
  1    the copy failed, or was written with tiles the slide could not supply
  2    invalid arguments, or OUTPUT exists (use --overwrite)
  130  cancelled with Ctrl-C (nothing is left behind)";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliCompression {
    /// Baseline JPEG (lossy, small).
    Jpeg,
    /// Deflate (lossless, larger).
    Deflate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliSlideFormat {
    /// Pyramidal TIFF, one image per level (OpenSlide generic TIFF).
    Tiff,
    /// OME-TIFF with the reduced levels as SubIFDs.
    OmeTiff,
}

/// `X,Y,WIDTH,HEIGHT` in level-0 pixels.
fn parse_region(text: &str) -> std::result::Result<eov_common::SlideBounds, String> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    let usage = || {
        format!(
            "expected X,Y,WIDTH,HEIGHT in level-0 pixels (e.g. 1000,2000,4096,4096), got {text:?}"
        )
    };
    let [x, y, width, height] = parts[..] else {
        return Err(usage());
    };
    let (Ok(x), Ok(y), Ok(width), Ok(height)) = (
        x.parse::<i64>(),
        y.parse::<i64>(),
        width.parse::<u64>(),
        height.parse::<u64>(),
    ) else {
        return Err(usage());
    };
    if width == 0 || height == 0 {
        return Err("the region's width and height must be at least 1".to_string());
    }
    Ok(eov_common::SlideBounds {
        x,
        y,
        width,
        height,
    })
}

#[derive(Debug, clap::Args)]
struct ConvertArgs {
    /// The slide to copy (any format eov opens).
    input: PathBuf,

    /// The file to write: .tif or .tiff, or .ome.tif for OME-TIFF.
    output: PathBuf,

    /// Copy only this rectangle, in level-0 pixels (default: the part of
    /// the slide that holds image data).
    #[arg(long, value_name = "X,Y,W,H", value_parser = parse_region, allow_hyphen_values = true)]
    region: Option<eov_common::SlideBounds>,

    /// Copy this pyramid level (0 = full resolution) at its native pixels.
    #[arg(long, value_name = "N", conflicts_with = "mpp")]
    level: Option<u32>,

    /// Copy at this pixel size in micrometres: a pyramid level within 1 %
    /// is used as it is, otherwise the next finer level is resampled.
    #[arg(long, value_name = "UM")]
    mpp: Option<f64>,

    /// Tile compression.
    #[arg(long, value_enum, default_value_t = CliCompression::Jpeg)]
    compression: CliCompression,

    /// JPEG quality, 1–100 (default 90); chroma is subsampled below 90.
    #[arg(long, value_name = "Q", value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: Option<u8>,

    /// Tile width and height in pixels.
    #[arg(
        long,
        value_name = "PX",
        default_value = "512",
        value_parser = clap::builder::TypedValueParser::map(
            clap::builder::PossibleValuesParser::new(["256", "512"]),
            |size: String| size.parse::<u32>().unwrap_or(512),
        )
    )]
    tile_size: u32,

    /// Container (default: ome-tiff for names ending in .ome.tif, .ome.tiff
    /// or .ome.btf, else tiff).
    #[arg(long, value_enum)]
    format: Option<CliSlideFormat>,

    /// Leave out the label, macro and thumbnail images and every property
    /// but geometry, pixel size, objective power and stain (see
    /// DE-IDENTIFICATION).
    #[arg(long)]
    deidentify: bool,

    /// With --deidentify, still record the scanner vendor and model.
    #[arg(long, requires = "deidentify")]
    keep_scanner: bool,

    /// Do not copy the slide's ICC colour profile.
    #[arg(long)]
    no_icc_profile: bool,

    /// Write a BigTIFF even when the copy would fit a classic TIFF.
    #[arg(long)]
    bigtiff: bool,

    /// Replace OUTPUT if it exists.
    #[arg(long)]
    overwrite: bool,
}

/// A parsed `eov convert` invocation.
#[derive(Debug)]
struct ConvertJob {
    input: PathBuf,
    output: PathBuf,
    options: eov_common::convert::ConvertOptions,
}

impl ConvertArgs {
    fn to_job(&self) -> Result<ConvertJob> {
        use eov_common::convert::{
            ConvertOptions, DEFAULT_JPEG_QUALITY, OutputResolution, SlideCompression, SlideFormat,
        };
        validate_input_file(&self.input)?;
        let compression = match (self.compression, self.quality) {
            (CliCompression::Jpeg, quality) => SlideCompression::Jpeg {
                quality: quality.unwrap_or(DEFAULT_JPEG_QUALITY),
            },
            (CliCompression::Deflate, None) => SlideCompression::Deflate,
            (CliCompression::Deflate, Some(_)) => {
                bail!("--quality applies to --compression jpeg only (deflate is lossless)")
            }
        };
        let resolution = match (self.level, self.mpp) {
            (Some(level), _) => OutputResolution::Level(level),
            (None, Some(mpp)) => OutputResolution::Mpp(mpp),
            (None, None) => OutputResolution::Full,
        };
        let format = match self.format {
            Some(CliSlideFormat::Tiff) => SlideFormat::Tiff,
            Some(CliSlideFormat::OmeTiff) => SlideFormat::OmeTiff,
            None => SlideFormat::from_path(&self.output),
        };
        Ok(ConvertJob {
            input: self.input.clone(),
            output: self.output.clone(),
            options: ConvertOptions {
                region: self.region,
                resolution,
                compression,
                tile_size: self.tile_size,
                format,
                deidentify: self.deidentify,
                keep_scanner: self.keep_scanner,
                keep_icc_profile: !self.no_icc_profile,
                bigtiff: self.bigtiff,
                overwrite: self.overwrite,
                mpp_override: None,
                threads: None,
            },
        })
    }
}

const AFTER_HELP: &str = "\
FILES AND PANES:
  Every FILE opens as a tab of one pane; the first one is selected. An
  argument that exists on disk is taken as it is, commas included; one that
  does not but contains commas is a list of files (a.svs,b.svs). A FILE may
  also be a folder (its slides appear in the series bar), a file:// URI or
  an eov:// link. Files that cannot be opened are reported in the window.

    eov a.svs b.svs c.svs                  one pane, three tabs
    eov a.svs --pane b.svs                 two panes side by side
    eov --pane a.svs,b.svs --pane c.svs    two panes, the first with two tabs

DEEP LINKS:
  --goto and --zoom set the view of the first slide:

    eov slide.svs --goto 1200,800 --zoom 20x
    eov slide.svs --goto 300um,200um --zoom 0.5

  An eov:// link does the same; percent-encode the path (+ is a space):

    eov 'eov://open?path=%2Fslides%2Fslide.svs&x=300um&y=200um&zoom=20x'

SINGLE INSTANCE:
  While eov runs, `eov FILES` hands the files to it, which opens them as
  tabs, and exits. Options that configure the viewer itself (--cpu,
  --debug, --window-x, ...) start a separate eov, as do --new-instance and
  single_instance = false in config.toml.";

#[derive(Debug, Parser)]
#[command(
    name = "eov",
    version,
    about = "A lightweight, cross-platform WSI viewer.",
    propagate_version = true,
    after_help = AFTER_HELP
)]
struct Cli {
    /// Slides or folders to open as tabs of one pane (see FILES AND PANES).
    #[arg(value_name = "FILES")]
    files: Vec<OsString>,

    /// Open FILES in another pane, side by side; repeat for more panes.
    /// FILES may list several slides, separated by commas.
    #[arg(long, value_name = "FILES")]
    pane: Vec<OsString>,

    /// Centre the first slide on X,Y: level-0 pixels (1200,800) or
    /// micrometres (300um,200um).
    #[arg(long, value_name = "X,Y", value_parser = crate::launch::parse_point)]
    goto: Option<Point>,

    /// Zoom the first slide: a magnification (20x), screen pixels per slide
    /// pixel (0.5; 1 or 1:1 is full resolution), or fit.
    #[arg(long, value_name = "ZOOM", value_parser = crate::launch::parse_zoom)]
    zoom: Option<Zoom>,

    /// Start a separate eov even if one is running (see SINGLE INSTANCE).
    #[arg(long)]
    new_instance: bool,

    #[command(subcommand)]
    command: Option<CliCommand>,

    /// Show the frame-rate overlay and other debugging aids.
    #[arg(short, long, global = true)]
    debug: bool,

    /// Rendering backend (auto: the saved choice, else the GPU).
    #[arg(long, value_enum, default_value_t = CliBackend::Auto, global = true)]
    backend: CliBackend,

    /// Render on the CPU (same as --backend cpu).
    #[arg(long, action = ArgAction::SetTrue, global = true)]
    cpu: bool,

    /// Render on the GPU (same as --backend gpu).
    #[arg(long, action = ArgAction::SetTrue, global = true)]
    gpu: bool,

    /// Texture filtering (auto: the saved choice, else trilinear).
    #[arg(long, value_enum, default_value_t = CliFilteringMode::Auto, global = true)]
    filtering_mode: CliFilteringMode,

    /// What to log on standard error (default: RUST_LOG, else warnings).
    #[arg(long, value_enum, global = true)]
    log_level: Option<CliLogLevel>,

    /// Tile-cache budget in MiB, at most 1048576 (default: sized for the
    /// display).
    #[arg(
        long,
        value_name = "MB",
        value_parser = clap::value_parser!(u64).range(1..=MAX_CACHE_MIB),
        global = true
    )]
    cache_size: Option<u64>,

    /// Maximum number of cached tiles, at most 16777216 (default: sized for
    /// the display).
    #[arg(
        long,
        value_name = "COUNT",
        value_parser = clap::value_parser!(u64).range(1..=eov_common::cache::MAX_TILES_LIMIT as u64),
        global = true
    )]
    max_tiles: Option<u64>,

    /// Use this configuration file instead of ~/.eov/config.toml (or
    /// EOV_CONFIG); a leading ~ is expanded.
    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    /// Window width in logical pixels.
    #[arg(
        long,
        value_name = "PX",
        value_parser = clap::value_parser!(u32).range(1..),
        global = true
    )]
    window_width: Option<u32>,

    /// Window height in logical pixels.
    #[arg(
        long,
        value_name = "PX",
        value_parser = clap::value_parser!(u32).range(1..),
        global = true
    )]
    window_height: Option<u32>,

    /// Window position from the screen's left edge in logical pixels; give
    /// --window-y too. Wayland ignores it: the compositor places windows.
    #[arg(
        long,
        value_name = "PX",
        global = true,
        requires = "window_y",
        allow_negative_numbers = true
    )]
    window_x: Option<i32>,

    /// Window position from the screen's top edge in logical pixels; give
    /// --window-x too. Wayland ignores it: the compositor places windows.
    #[arg(
        long,
        value_name = "PX",
        global = true,
        requires = "window_x",
        allow_negative_numbers = true
    )]
    window_y: Option<i32>,

    /// Directory with plugin packages (`.eop`). Default: ~/.eov/plugins, or
    /// EOV_PLUGIN_DIR; a leading ~ is expanded.
    #[arg(long, value_name = "PATH", global = true)]
    plugin_dir: Option<PathBuf>,

    /// Link the views of panes that show the same slide.
    #[arg(long, action = ArgAction::SetTrue, global = true)]
    viewport_lock: bool,

    /// Start without loading any plugins, for example after a plugin made
    /// eov crash at startup (also enabled by EOV_SAFE_MODE=1).
    #[arg(long, action = ArgAction::SetTrue, global = true)]
    safe_mode: bool,
}

enum CommandAction {
    LaunchUi,
    ApplicationUpdate(UpdateArgs),
    Plugin(PluginCommand),
    Probe { file: PathBuf, json: bool },
    RecentList,
    ConfigPath,
    Paths { json: bool },
    FileTypes { register: bool, dry_run: bool },
    DatasetPatches(Box<DatasetJob>),
    Convert(Box<ConvertArgs>),
}

pub(crate) struct LaunchOptions {
    pub(crate) debug_mode: bool,
    /// Positional files (the current pane's tabs).
    files: Vec<OsString>,
    /// `--pane` values.
    panes: Vec<OsString>,
    /// `--goto` and `--zoom`.
    view: ViewRequest,
    /// `--new-instance`.
    new_instance: bool,
    /// An option that configures this process was given, so the launch is
    /// not handed to a running eov.
    configures_process: bool,
    pub(crate) render_backend_override: Option<RenderBackend>,
    pub(crate) filtering_mode_override: Option<FilteringMode>,
    pub(crate) viewport_lock_enabled: bool,
    pub(crate) log_level: Option<CliLogLevel>,
    /// Fixed tile-cache limits; `None` sizes the cache for the display.
    pub(crate) cache_size_bytes: Option<usize>,
    pub(crate) max_tiles: Option<usize>,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) window_geometry: WindowGeometry,
    pub(crate) plugin_dir: PathBuf,
    /// `--safe-mode` / `EOV_SAFE_MODE=1`: the GUI must not discover, extract
    /// or load any plugin (and should say so in the window).
    #[allow(dead_code)] // honoured by main() once the plugin host checks it
    pub(crate) safe_mode: bool,
    command: CommandAction,
}

/// The command line, without the `-psn_…` argument macOS adds when it
/// starts an app from the Finder on some versions.
fn launch_arguments() -> Vec<OsString> {
    let mut arguments: Vec<OsString> = std::env::args_os().collect();
    if cfg!(target_os = "macos") {
        let mut index = 0;
        arguments.retain(|argument| {
            index += 1;
            index == 1 || !argument.to_string_lossy().starts_with("-psn_")
        });
    }
    arguments
}

pub(crate) fn parse_launch_options() -> Result<LaunchOptions> {
    launch_options(Cli::parse_from(launch_arguments()))
}

fn launch_options(cli: Cli) -> Result<LaunchOptions> {
    let cache_size_bytes = cli
        .cache_size
        .map(|mib| {
            mib.checked_mul(1024 * 1024)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| anyhow::anyhow!("--cache-size is too large for this platform"))
        })
        .transpose()?;

    let max_tiles = cli
        .max_tiles
        .map(|count| {
            usize::try_from(count)
                .map_err(|_| anyhow::anyhow!("--max-tiles is too large for this platform"))
        })
        .transpose()?;

    let opens_files = !cli.files.is_empty() || !cli.pane.is_empty();
    if cli.command.is_some()
        && (opens_files || cli.goto.is_some() || cli.zoom.is_some() || cli.new_instance)
    {
        bail!(
            "files, --pane, --goto, --zoom and --new-instance cannot be combined with a subcommand"
        );
    }
    if (cli.goto.is_some() || cli.zoom.is_some()) && !opens_files {
        bail!("--goto and --zoom need a slide to open");
    }

    let shorthand_backend = match (cli.cpu, cli.gpu) {
        (true, true) => {
            bail!("--cpu and --gpu are mutually exclusive; choose only one rendering override")
        }
        (true, false) => Some(RenderBackend::Cpu),
        (false, true) => Some(RenderBackend::Gpu),
        (false, false) => None,
    };

    let backend_flag = cli.backend.render_backend_override();
    let render_backend_override = match (backend_flag, shorthand_backend) {
        (Some(explicit), Some(shorthand)) if explicit != shorthand => {
            bail!(
                "--backend {} conflicts with shorthand override; use only one backend selector",
                cli.backend.as_str()
            )
        }
        (Some(explicit), _) => Some(explicit),
        (None, Some(shorthand)) => Some(shorthand),
        (None, None) => None,
    };

    let safe_mode = cli.safe_mode || safe_mode_from_env();
    let configures_process = cli.debug
        || cli.backend != CliBackend::Auto
        || cli.cpu
        || cli.gpu
        || cli.filtering_mode != CliFilteringMode::Auto
        || cli.log_level.is_some()
        || cli.cache_size.is_some()
        || cli.max_tiles.is_some()
        || cli.window_width.is_some()
        || cli.window_height.is_some()
        || cli.window_x.is_some()
        || cli.window_y.is_some()
        || cli.viewport_lock
        || safe_mode;

    let command = match cli.command {
        Some(CliCommand::Update(args)) => CommandAction::ApplicationUpdate(args),
        Some(CliCommand::Plugin { command }) => CommandAction::Plugin(command),
        Some(CliCommand::Probe { file, json }) => {
            validate_input_file(&file)?;
            CommandAction::Probe { file, json }
        }
        Some(CliCommand::Recent {
            command: RecentCommand::List,
        }) => CommandAction::RecentList,
        Some(CliCommand::ConfigPath) => CommandAction::ConfigPath,
        Some(CliCommand::Paths { json }) => CommandAction::Paths { json },
        Some(CliCommand::RegisterFileTypes { dry_run }) => CommandAction::FileTypes {
            register: true,
            dry_run,
        },
        Some(CliCommand::UnregisterFileTypes { dry_run }) => CommandAction::FileTypes {
            register: false,
            dry_run,
        },

        Some(CliCommand::Dataset { command }) => match command {
            DatasetCommand::Patches(args) => {
                CommandAction::DatasetPatches(Box::new(args.into_job()))
            }
        },
        Some(CliCommand::Convert(args)) => CommandAction::Convert(args),
        None => CommandAction::LaunchUi,
    };

    let plugin_dir = resolve_plugin_dir(cli.plugin_dir.as_deref());

    Ok(LaunchOptions {
        debug_mode: cli.debug,
        files: cli.files,
        panes: cli.pane,
        view: ViewRequest {
            center: cli.goto,
            zoom: cli.zoom,
        },
        new_instance: cli.new_instance,
        configures_process,
        render_backend_override,
        filtering_mode_override: cli.filtering_mode.filtering_mode_override(),
        viewport_lock_enabled: cli.viewport_lock,
        log_level: cli.log_level,
        cache_size_bytes,
        max_tiles,
        config_path: cli.config,
        window_geometry: WindowGeometry {
            width: cli.window_width,
            height: cli.window_height,
            x: cli.window_x,
            y: cli.window_y,
        },
        plugin_dir,
        safe_mode,
        command,
    })
}

/// Whether `EOV_SAFE_MODE` asks for safe mode (any value but empty or `0`).
fn safe_mode_from_env() -> bool {
    std::env::var_os("EOV_SAFE_MODE").is_some_and(|value| !value.is_empty() && value != "0")
}

/// Default plugin directory: `~/.eov/plugins/` (or `$EOV_PLUGIN_DIR`).
pub(crate) fn default_plugin_dir() -> PathBuf {
    crate::paths::plugin_dir()
}

/// Resolve `--plugin-dir` with `~` expansion, falling back to the default.
pub(crate) fn resolve_plugin_dir(explicit: Option<&Path>) -> PathBuf {
    match explicit {
        Some(p) => expand_tilde(p),
        None => default_plugin_dir(),
    }
}

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &Path) -> PathBuf {
    crate::paths::expand_tilde(path)
}

pub(crate) fn apply_config_override(config_path: Option<&PathBuf>) -> Result<()> {
    if let Some(path) = config_path {
        config::set_config_path_override(expand_tilde(path))?;
    }

    Ok(())
}

impl LaunchOptions {
    /// Whether this run opens the GUI (rather than running a subcommand).
    pub(crate) fn launches_ui(&self) -> bool {
        matches!(self.command, CommandAction::LaunchUi)
    }

    /// What to open. Call after the config override is applied: links
    /// consult `trusted_link_servers`.
    pub(crate) fn open_request(&self) -> OpenRequest {
        let cwd = std::env::current_dir().unwrap_or_default();
        crate::launch::request_from_arguments(
            &self.files,
            &self.panes,
            self.view,
            &cwd,
            &crate::launch::find_on_disk,
            &config::load_trusted_link_servers(),
        )
    }

    /// Whether this launch may hand its files to an eov that is running
    /// (SH-8): the GUI, without --new-instance or options that configure
    /// the process (they need a process of their own).
    pub(crate) fn may_forward(&self) -> bool {
        self.launches_ui() && !self.new_instance && !self.configures_process
    }
}

/// Lets a subcommand whose output goes to a closed pipe (`eov probe x |
/// head`) end quietly, as other command-line tools do, instead of
/// panicking in `println!`. The viewer keeps ignoring SIGPIPE: its sockets
/// report closed peers as errors.
fn end_quietly_on_closed_pipe() {
    #[cfg(unix)]
    // SAFETY: restoring the default disposition of one signal, before any
    // other thread exists.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

pub(crate) fn maybe_run_cli_command(launch_options: &LaunchOptions) -> Result<bool> {
    if !launch_options.launches_ui() {
        end_quietly_on_closed_pipe();
    }
    match &launch_options.command {
        CommandAction::LaunchUi => Ok(false),
        CommandAction::ApplicationUpdate(args) => {
            run_update_command(&launch_options.plugin_dir, args)?;
            Ok(true)
        }
        CommandAction::Plugin(command) => {
            run_plugin_command(&launch_options.plugin_dir, command)?;
            Ok(true)
        }
        CommandAction::Probe { file, json } => {
            crate::probe::run(file, *json)?;
            Ok(true)
        }
        CommandAction::RecentList => {
            print_recent_files();
            Ok(true)
        }
        CommandAction::ConfigPath => {
            println!("{}", config::resolve_config_path()?.display());
            Ok(true)
        }
        CommandAction::Paths { json } => {
            print_storage_locations(&launch_options.plugin_dir, *json)?;
            Ok(true)
        }
        CommandAction::FileTypes { register, dry_run } => {
            crate::file_types::run_cli(*register, *dry_run)?;
            Ok(true)
        }
        CommandAction::DatasetPatches(job) => {
            run_dataset_patches_cli(job)?;
            Ok(true)
        }
        CommandAction::Convert(args) => {
            run_convert_cli(args);
            Ok(true)
        }
    }
}

/// Runs `eov update`. With `--json`, failures are reported as a JSON object
/// on standard output as well.
fn run_update_command(plugin_dir: &Path, args: &UpdateArgs) -> Result<()> {
    let options = update::UpdateOptions {
        request: update::UpdateRequest {
            release: args.release.clone(),
            app_only: args.app_only,
            from: args.from.clone(),
            allow_unsigned: args.allow_unsigned,
            allow_network: args.allow_network,
        },
        yes: args.yes,
        dry_run: args.dry_run,
        json: args.json,
        rollback: args.rollback,
        relaunch: !args.no_relaunch,
    };
    let result = update::run_update_cli(plugin_dir, &options);
    if args.json
        && let Err(error) = &result
    {
        println!("{}", serde_json::json!({ "error": format!("{error:#}") }));
    }
    // A helper-based update continues after this process exits.
    result.map(|_| ())
}

/// Runs an `eov plugin` subcommand. With `--json`, failures are reported as
/// a JSON object on standard output as well.
fn run_plugin_command(plugin_dir: &Path, command: &PluginCommand) -> Result<()> {
    use update::plugins::{self, InstallOptions, PluginContext, RemoveOptions, SyncOptions};

    let (yes, json) = match command {
        PluginCommand::Ls { json, .. }
        | PluginCommand::Disable { json, .. }
        | PluginCommand::Enable { json, .. }
        | PluginCommand::Lock { json, .. }
        | PluginCommand::Gc { json } => (false, *json),
        PluginCommand::Install(args) | PluginCommand::Update(args) => (args.yes, args.json),
        PluginCommand::Remove { yes, json, .. } | PluginCommand::Sync { yes, json, .. } => {
            (*yes, *json)
        }
        PluginCommand::Info { output, .. } => (false, *output == CliPluginOutputFormat::Json),
    };
    let result = PluginContext::for_cli(plugin_dir, yes, json).and_then(|mut ctx| match command {
        PluginCommand::Ls {
            allow_network,
            offline,
            ..
        } => plugins::list(&mut ctx, *allow_network, *offline),
        PluginCommand::Install(args) | PluginCommand::Update(args) => plugins::install(
            &mut ctx,
            &args.source,
            &InstallOptions {
                allow_network: args.allow_network,
                dry_run: args.dry_run,
                force: args.force,
                id: args.id.clone(),
                require_existing: matches!(command, PluginCommand::Update(_)),
            },
        ),
        PluginCommand::Remove {
            plugin, dry_run, ..
        } => plugins::remove(&mut ctx, plugin, &RemoveOptions { dry_run: *dry_run }),
        PluginCommand::Info { plugin, output } => plugins::info(
            &ctx,
            plugin,
            match output {
                CliPluginOutputFormat::Plain => update::PluginOutputFormat::Plain,
                CliPluginOutputFormat::Toml => update::PluginOutputFormat::Toml,
                CliPluginOutputFormat::Json => update::PluginOutputFormat::Json,
            },
        ),
        PluginCommand::Disable { id, .. } => plugins::set_enabled(&ctx, id, false),
        PluginCommand::Enable { id, .. } => plugins::set_enabled(&ctx, id, true),
        PluginCommand::Lock { output, .. } => plugins::lock(&ctx, output.as_deref()),
        PluginCommand::Sync {
            lockfile,
            from,
            prune,
            allow_network,
            dry_run,
            ..
        } => plugins::sync(
            &mut ctx,
            lockfile,
            &SyncOptions {
                allow_network: *allow_network,
                dry_run: *dry_run,
                prune: *prune,
                from: from.clone(),
            },
        ),
        PluginCommand::Gc { .. } => plugins::collect_garbage(&ctx),
    });
    if json && let Err(error) = &result {
        println!("{}", serde_json::json!({ "error": format!("{error:#}") }));
    }
    result
}

/// `eov dataset patches` exit statuses (see the subcommand help).
const DATASET_EXIT_FAILURES: i32 = 1;
const DATASET_EXIT_USAGE: i32 = 2;
const DATASET_EXIT_INTERRUPTED: i32 = 130;

fn dataset_exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code)
}

fn print_dataset_plan(plan: &dataset::DatasetPlan) {
    for (path, err) in &plan.input_errors {
        eprintln!("warning: {}: {}", path.display(), err);
    }
    for skipped in &plan.skipped_inputs {
        eprintln!(
            "note: skipping {}: {}",
            skipped.path.display(),
            skipped.reason
        );
    }
    for slide in &plan.slides {
        match &slide.geometry {
            Ok(g) => println!(
                "{}: {} patches (level {}, {:.3} level-0 px per output px{})",
                slide.input.path.display(),
                slide.planned_tiles,
                g.level,
                g.downsample.0,
                g.output_mpp
                    .0
                    .map(|m| format!(", {m:.4} µm/px"))
                    .unwrap_or_default()
            ),
            Err(reason) => eprintln!("warning: {}: {}", slide.input.path.display(), reason),
        }
    }
    println!();
    println!(
        "Dry run: {} patch(es) would be extracted from {} slide(s).",
        plan.planned_tiles,
        plan.slides.iter().filter(|s| s.geometry.is_ok()).count()
    );
}

fn print_dataset_report(report: &dataset::DatasetPatchesReport) {
    for (path, err) in &report.input_errors {
        eprintln!("warning: {}: {}", path.display(), err);
    }
    for skipped in &report.skipped_inputs {
        eprintln!(
            "note: skipped {}: {}",
            skipped.path.display(),
            skipped.reason
        );
    }
    for slide in &report.slides {
        match &slide.skipped {
            Some(reason) => {
                let level = if matches!(slide.status, dataset::SlideStatus::Failed) {
                    "error"
                } else {
                    "warning"
                };
                eprintln!("{level}: {}: {}", slide.path.display(), reason);
            }
            None => {
                let mut parts = vec![format!("{} written", slide.tiles_written)];
                if slide.tiles_existing > 0 {
                    parts.push(format!("{} already present", slide.tiles_existing));
                }
                let skipped = slide.tiles_skipped_white
                    + slide.tiles_skipped_tissue
                    + slide.tiles_skipped_annotation;
                if skipped > 0 {
                    parts.push(format!("{skipped} skipped"));
                }
                if slide.tiles_failed > 0 {
                    parts.push(format!("{} unreadable", slide.tiles_failed));
                }
                println!("{}: {}", slide.path.display(), parts.join(", "));
            }
        }
        for err in &slide.errors {
            eprintln!("  error: {err}");
        }
    }
    println!();
    println!(
        "Done: {} slide(s) processed, {} skipped, {} failed; {} patch(es) written, {} already \
         present, {} skipped as white, {} as non-tissue, {} outside annotations, {} unreadable.",
        report.processed_slides,
        report.skipped_slides - report.failed_slides,
        report.failed_slides,
        report.total_tiles,
        report.total_tiles_existing,
        report.total_tiles_skipped_white,
        report.total_tiles_skipped_tissue,
        report.total_tiles_skipped_annotation,
        report.total_tiles_failed,
    );
    for meta in &report.metadata_paths {
        println!("Metadata: {}", meta.display());
    }
    if let Some(manifest) = &report.manifest_path {
        println!("Manifest: {}", manifest.display());
    }
}

fn run_dataset_patches_cli(job: &DatasetJob) -> Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    if job.options.dry_run {
        return match dataset::plan_dataset(&job.config, &job.options) {
            Ok(plan) => {
                print_dataset_plan(&plan);
                Ok(())
            }
            Err(err) => report_dataset_error(err),
        };
    }

    let control = Arc::new(dataset::DatasetControl::new());
    {
        let control = Arc::clone(&control);
        let presses = AtomicU32::new(0);
        let _ = ctrlc::set_handler(move || {
            if presses.fetch_add(1, Ordering::Relaxed) == 0 {
                control.cancel();
                eprintln!(
                    "\nCancelling after the current batch of patches \
                     (press Ctrl-C again to stop immediately)..."
                );
            } else {
                dataset_exit(DATASET_EXIT_INTERRUPTED);
            }
        });
    }

    let config = job.config.clone();
    let options = job.options.clone();
    let worker_control = Arc::clone(&control);
    let handle = std::thread::Builder::new()
        .name("dataset-patches".into())
        .spawn(move || dataset::run_dataset(&config, &options, &worker_control))
        .expect("failed to spawn dataset-patches thread");

    let pb = indicatif::ProgressBar::new(0);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} patches \
                 ({eta} remaining) {msg}",
            )
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(200));
    let update = |pb: &indicatif::ProgressBar| {
        let p = control.progress();
        if p.tiles_total > 0 {
            pb.set_length(p.tiles_total);
        }
        pb.set_position(p.tiles_done);
        pb.set_message(format!(
            "slide {}/{} · {} written · {} skipped · {} failed",
            p.current_slide, p.total_slides, p.tiles_written, p.tiles_skipped, p.tiles_failed
        ));
    };
    while !handle.is_finished() {
        update(&pb);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    update(&pb);
    pb.finish_and_clear();

    match handle.join().expect("dataset-patches thread panicked") {
        Ok(report) => {
            print_dataset_report(&report);
            if report.cancelled {
                eprintln!("Cancelled. Run the same command again to resume.");
                dataset_exit(DATASET_EXIT_INTERRUPTED);
            }
            if report.has_errors() {
                dataset_exit(DATASET_EXIT_FAILURES);
            }
            Ok(())
        }
        Err(err) => report_dataset_error(err),
    }
}

/// Prints a run-level dataset error and exits with the matching status.
fn report_dataset_error(err: dataset::DatasetError) -> Result<()> {
    match err {
        dataset::DatasetError::Config(e) => {
            eprintln!("error: {e}");
            dataset_exit(DATASET_EXIT_USAGE);
        }
        dataset::DatasetError::NoSlides { input_errors } => {
            for (path, e) in &input_errors {
                eprintln!("error: {}: {}", path.display(), e);
            }
            eprintln!("error: no slide files were found in the inputs");
            dataset_exit(DATASET_EXIT_USAGE);
        }
        dataset::DatasetError::OutputConflict(message) => {
            eprintln!("error: {message}");
            eprintln!("hint: pass --overwrite to replace it, or choose another --out directory");
            dataset_exit(DATASET_EXIT_USAGE);
        }
        other => Err(other.into()),
    }
}

/// `eov convert` exit statuses (see the subcommand help).
const CONVERT_EXIT_FAILED: i32 = 1;
const CONVERT_EXIT_USAGE: i32 = 2;
const CONVERT_EXIT_INTERRUPTED: i32 = 130;

/// What a conversion will do, for the terminal.
fn describe_conversion(
    job: &ConvertJob,
    source: &eov_common::WsiProperties,
    plan: &eov_common::convert::ConvertPlan,
) -> String {
    use eov_common::convert::SlideCompression;
    let (width, height) = plan.levels[0];
    let mut lines = vec![format!(
        "Copying {} ({}, {} × {} px)",
        job.input.display(),
        source.vendor.as_deref().unwrap_or("unknown format"),
        source.width,
        source.height
    )];
    let mut what = vec![
        plan.format.label().to_string(),
        format!("{width} × {height} px"),
    ];
    if let Some((x, y)) = plan.mpp {
        what.push(if (x - y).abs() < 1e-9 {
            format!("{} µm/px", eov_common::format_decimal(x))
        } else {
            format!(
                "{} × {} µm/px",
                eov_common::format_decimal(x),
                eov_common::format_decimal(y)
            )
        });
    }
    what.push(format!("{} levels", plan.levels.len()));
    what.push(match job.options.compression {
        SlideCompression::Jpeg { quality } => {
            format!("{}-px JPEG tiles (quality {quality})", plan.tile_size)
        }
        SlideCompression::Deflate => format!("{}-px lossless tiles", plan.tile_size),
    });
    if plan.resampled {
        what.push(format!("resampled from level {}", plan.source_level));
    } else if plan.source_level > 0 {
        what.push(format!("from level {}", plan.source_level));
    }
    if job.options.deidentify {
        what.push("de-identified".to_string());
    }
    lines.push(format!(
        "  to {}: {}",
        job.output.display(),
        what.join(", ")
    ));
    if !plan.associated_images.is_empty() {
        lines.push(format!(
            "  with the {} image(s); pass --deidentify to leave them out",
            plan.associated_images.join(", ")
        ));
    }
    lines.join("\n")
}

/// `"1h 02m"`, `"4m 05s"`, `"12s"`.
fn format_elapsed(elapsed: std::time::Duration) -> String {
    let total = elapsed.as_secs_f64().round() as u64;
    let (hours, minutes, seconds) = (total / 3600, (total / 60) % 60, total % 60);
    if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

/// Runs `eov convert`, exiting with its status (see the help) unless the
/// copy was written completely.
fn run_convert_cli(args: &ConvertArgs) {
    use eov_common::convert::{self, ConvertError};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    let job = match args.to_job() {
        Ok(job) => job,
        Err(err) => {
            eprintln!("error: {err:#}");
            dataset_exit(CONVERT_EXIT_USAGE);
        }
    };
    let wsi = match eov_common::WsiFile::open(&job.input) {
        Ok(wsi) => wsi,
        Err(err) => {
            eprintln!("error: cannot open {}: {err}", job.input.display());
            dataset_exit(CONVERT_EXIT_FAILED);
        }
    };
    let plan = match convert::plan_conversion(&wsi, &job.options).and_then(|plan| {
        convert::check_output(&wsi, &job.output, job.options.overwrite).map(|()| plan)
    }) {
        Ok(plan) => plan,
        Err(ConvertError::OutputExists(path)) => {
            eprintln!("error: {} already exists", path.display());
            eprintln!("hint: pass --overwrite to replace it, or choose another name");
            dataset_exit(CONVERT_EXIT_USAGE);
        }
        Err(err) => {
            eprintln!("error: {err}");
            dataset_exit(CONVERT_EXIT_USAGE);
        }
    };
    eprintln!("{}", describe_conversion(&job, wsi.properties(), &plan));

    let control = Arc::new(eov_common::job::JobControl::new());
    {
        let control = Arc::clone(&control);
        let partial = convert::partial_path(&job.output);
        let presses = AtomicU32::new(0);
        let _ = ctrlc::set_handler(move || {
            if presses.fetch_add(1, Ordering::Relaxed) == 0 {
                control.cancel();
                eprintln!("\nCancelling (press Ctrl-C again to stop at once)...");
            } else {
                // The worker would remove it; it does not get the chance.
                let _ = std::fs::remove_file(&partial);
                dataset_exit(CONVERT_EXIT_INTERRUPTED);
            }
        });
    }
    let worker = {
        let control = Arc::clone(&control);
        let options = job.options.clone();
        let output = job.output.clone();
        std::thread::Builder::new()
            .name("eov-convert".into())
            .spawn(move || convert::convert_slide(&wsi, &output, &options, &control))
            .expect("failed to spawn the conversion thread")
    };

    let pb = indicatif::ProgressBar::new(plan.tiles + plan.associated_images.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles \
                 ({eta} remaining)",
            )
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(200));
    while !worker.is_finished() {
        let progress = control.progress();
        if progress.total > 0 {
            pb.set_length(progress.total);
        }
        pb.set_position(progress.done);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    pb.finish_and_clear();

    match worker.join().expect("the conversion thread panicked") {
        Ok(report) => {
            for warning in &report.warnings {
                eprintln!("warning: {warning}");
            }
            let (w0, h0) = report.plan.levels[0];
            let (wn, hn) = *report.plan.levels.last().unwrap_or(&(w0, h0));
            let mut facts = vec![
                format_bytes(report.bytes),
                format!(
                    "{} levels ({w0} × {h0} to {wn} × {hn} px)",
                    report.plan.levels.len()
                ),
            ];
            if report.bigtiff {
                facts.push("BigTIFF".to_string());
            }
            if job.options.deidentify {
                facts.push("de-identified".to_string());
            }
            if !report.associated_images.is_empty() {
                facts.push(format!("with {}", report.associated_images.join(", ")));
            }
            if report.icc_profile {
                facts.push("ICC profile".to_string());
            }
            println!(
                "Wrote {}: {}, in {}",
                report.path.display(),
                facts.join(", "),
                format_elapsed(report.elapsed)
            );
            if let Some(shortfall) = report.shortfall_message() {
                eprintln!("warning: {shortfall}");
                dataset_exit(CONVERT_EXIT_FAILED);
            }
        }
        Err(ConvertError::Cancelled) => {
            eprintln!("Cancelled; nothing was written.");
            dataset_exit(CONVERT_EXIT_INTERRUPTED);
        }
        Err(ConvertError::OutputExists(path)) => {
            eprintln!("error: {} already exists", path.display());
            eprintln!("hint: pass --overwrite to replace it, or choose another name");
            dataset_exit(CONVERT_EXIT_USAGE);
        }
        Err(ConvertError::Invalid(message)) => {
            eprintln!("error: {message}");
            dataset_exit(CONVERT_EXIT_USAGE);
        }
        Err(err) => {
            eprintln!("error: could not write {}: {err}", job.output.display());
            dataset_exit(CONVERT_EXIT_FAILED);
        }
    }
}

fn validate_input_file(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("input file does not exist: {}", path.display());
    }

    if !path.is_file() {
        bail!("input path is not a file: {}", path.display());
    }

    Ok(())
}

/// A size for people (`1.5 GiB`).
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["bytes", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} bytes")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// `eov paths`: every place eov keeps data, with what it holds and its
/// size on disk.
fn print_storage_locations(plugin_dir: &Path, json: bool) -> Result<()> {
    let policy_file = update::policy::policy_path();
    let locations = crate::paths::storage_locations(
        &config::resolve_config_path()?,
        plugin_dir,
        policy_file.as_deref(),
        &update::policy::current().plugin_dirs,
    );
    if json {
        let entries: Vec<serde_json::Value> = locations
            .iter()
            .map(|location| {
                let bytes = crate::paths::disk_usage(&location.path);
                serde_json::json!({
                    "id": location.id,
                    "description": location.description,
                    "path": location.path.display().to_string(),
                    "kind": location.kind,
                    "exists": bytes.is_some(),
                    "bytes": bytes,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }
    let width = locations
        .iter()
        .map(|location| location.description.chars().count())
        .max()
        .unwrap_or_default();
    for location in &locations {
        let size = match crate::paths::disk_usage(&location.path) {
            Some(bytes) => format_bytes(bytes),
            None => "not created yet".to_string(),
        };
        println!(
            "{:width$}  {}  ({size})",
            location.description,
            location.path.display()
        );
    }
    Ok(())
}

fn print_recent_files() {
    let recent_files = crate::recent::load();
    if recent_files.is_empty() {
        println!("No recent files.");
        return;
    }

    for recent_file in recent_files {
        println!("{}", recent_file.path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_plugin_dir_is_plugins_below_isolated_home() {
        let dir = default_plugin_dir();
        let root = crate::paths::isolate_for_tests();
        assert!(
            dir.ends_with("plugins") && dir.starts_with(root),
            "expected the default plugin dir inside the test root, got {}",
            dir.display()
        );
    }

    #[test]
    fn resolve_plugin_dir_uses_default_when_none() {
        let dir = resolve_plugin_dir(None);
        assert_eq!(dir, default_plugin_dir());
    }

    #[test]
    fn resolve_plugin_dir_expands_tilde() {
        let dir = resolve_plugin_dir(Some(Path::new("~/my_plugins")));
        // Should not start with ~
        assert!(!dir.to_string_lossy().starts_with('~'));
        assert!(dir.to_string_lossy().ends_with("my_plugins"));
    }

    #[test]
    fn resolve_plugin_dir_absolute_path_unchanged() {
        let dir = resolve_plugin_dir(Some(Path::new("/opt/eov/plugins")));
        assert_eq!(dir, PathBuf::from("/opt/eov/plugins"));
    }

    #[test]
    fn resolve_plugin_dir_relative_path_unchanged() {
        let dir = resolve_plugin_dir(Some(Path::new("relative/plugins")));
        assert_eq!(dir, PathBuf::from("relative/plugins"));
    }

    #[test]
    fn expand_tilde_only_prefix() {
        // A path that contains ~ in the middle should not be expanded
        let path = expand_tilde(Path::new("/home/user/~config"));
        assert_eq!(path, PathBuf::from("/home/user/~config"));
    }

    fn parse(arguments: &[&str]) -> Result<LaunchOptions> {
        let cli = Cli::try_parse_from(std::iter::once("eov").chain(arguments.iter().copied()))?;
        launch_options(cli)
    }

    #[test]
    fn files_panes_and_views_are_parsed() {
        let options = parse(&[
            "a.svs",
            "b.svs",
            "--pane",
            "c.svs,d.svs",
            "--goto",
            "300um,200um",
            "--zoom",
            "20x",
        ])
        .unwrap();
        assert_eq!(options.files, [OsString::from("a.svs"), "b.svs".into()]);
        assert_eq!(options.panes, [OsString::from("c.svs,d.svs")]);
        assert_eq!(options.view.zoom, Some(Zoom::Magnification(20.0)));
        assert!(options.launches_ui() && options.may_forward());

        assert!(parse(&["--goto", "1,2"]).is_err(), "--goto needs a slide");
        assert!(parse(&["a.svs", "--goto", "1um,2"]).is_err());
        assert!(parse(&["a.svs", "--zoom", "huge"]).is_err());
        assert!(parse(&["--pane", "a.svs", "probe", "b.svs"]).is_err());
    }

    #[test]
    fn a_double_dash_makes_everything_a_file() {
        let options = parse(&["--", "probe", "--cpu"]).unwrap();
        assert!(options.launches_ui());
        assert_eq!(options.files, [OsString::from("probe"), "--cpu".into()]);
    }

    #[test]
    fn options_that_configure_the_process_start_a_new_one() {
        assert!(parse(&["a.svs"]).unwrap().may_forward());
        assert!(
            parse(&["--config", "~/x.toml", "a.svs"])
                .unwrap()
                .may_forward()
        );
        for configured in [
            &["--cpu", "a.svs"][..],
            &["--debug", "a.svs"],
            &["--window-width", "800", "a.svs"],
            &["--log-level", "debug"],
            &["--new-instance", "a.svs"],
        ] {
            assert!(
                !parse(configured).unwrap().may_forward(),
                "{configured:?} must not forward"
            );
        }
        assert!(!parse(&["recent", "list"]).unwrap().may_forward());
    }

    #[test]
    fn limits_and_window_positions_are_checked() {
        assert!(parse(&["--max-tiles", "16777216"]).is_ok());
        assert!(parse(&["--max-tiles", "16777217"]).is_err());
        assert!(parse(&["--max-tiles", "0"]).is_err());
        assert!(parse(&["--cache-size", "1048577"]).is_err());
        assert!(parse(&["--window-x", "10"]).is_err(), "needs --window-y");
        let options = parse(&["--window-x", "-200", "--window-y", "40"]).unwrap();
        assert_eq!(options.window_geometry.x, Some(-200));
        assert_eq!(options.window_geometry.y, Some(40));
    }

    #[test]
    fn the_help_text_documents_every_option_and_command() {
        use clap::CommandFactory;
        let command = Cli::command();
        for argument in command.get_arguments() {
            assert!(
                argument.get_help().is_some(),
                "--{} has no help text",
                argument.get_id()
            );
        }
        fn check(command: &clap::Command) {
            for subcommand in command.get_subcommands() {
                assert!(
                    subcommand.get_about().is_some(),
                    "`{}` has no help text",
                    subcommand.get_name()
                );
                for argument in subcommand.get_arguments() {
                    assert!(
                        argument.get_help().is_some() || argument.is_hide_set(),
                        "`{} {}` has no help text",
                        subcommand.get_name(),
                        argument.get_id()
                    );
                }
                check(subcommand);
            }
        }
        check(&command);
    }

    fn convert_args(arguments: &[&str]) -> std::result::Result<Box<ConvertArgs>, String> {
        let cli = Cli::try_parse_from(
            ["eov", "convert"]
                .into_iter()
                .chain(arguments.iter().copied()),
        )
        .map_err(|err| err.to_string())?;
        match cli.command {
            Some(CliCommand::Convert(args)) => Ok(args),
            other => Err(format!("not a convert command: {other:?}")),
        }
    }

    #[test]
    fn convert_options_are_parsed_and_checked() {
        use eov_common::convert::{OutputResolution, SlideCompression, SlideFormat};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("slide.svs");
        std::fs::write(&input, b"").unwrap();
        let input = input.to_str().unwrap();

        let args = convert_args(&[input, "out.tif"]).unwrap();
        let job = args.to_job().unwrap();
        assert_eq!(
            job.options.compression,
            SlideCompression::Jpeg { quality: 90 }
        );
        assert_eq!(job.options.tile_size, 512);
        assert_eq!(job.options.format, SlideFormat::Tiff);
        assert_eq!(job.options.resolution, OutputResolution::Full);
        assert!(!job.options.deidentify && job.options.keep_icc_profile);

        let args = convert_args(&[
            input,
            "roi.ome.tif",
            "--region",
            "-10,20,300,400",
            "--mpp",
            "0.5",
            "--quality",
            "75",
            "--tile-size",
            "256",
            "--deidentify",
            "--keep-scanner",
            "--no-icc-profile",
            "--bigtiff",
            "--overwrite",
        ])
        .unwrap();
        let job = args.to_job().unwrap();
        assert_eq!(job.options.format, SlideFormat::OmeTiff, "from the name");
        assert_eq!(job.options.resolution, OutputResolution::Mpp(0.5));
        assert_eq!(
            job.options.compression,
            SlideCompression::Jpeg { quality: 75 }
        );
        assert_eq!(job.options.tile_size, 256);
        let region = job.options.region.unwrap();
        assert_eq!(
            (region.x, region.y, region.width, region.height),
            (-10, 20, 300, 400)
        );
        assert!(job.options.deidentify && job.options.keep_scanner && job.options.bigtiff);
        assert!(!job.options.keep_icc_profile && job.options.overwrite);

        let args = convert_args(&[input, "x.ome.tif", "--format", "tiff", "--level", "2"]).unwrap();
        let job = args.to_job().unwrap();
        assert_eq!(job.options.format, SlideFormat::Tiff, "--format wins");
        assert_eq!(job.options.resolution, OutputResolution::Level(2));

        // Refused by the parser.
        for bad in [
            &[input, "o.tif", "--level", "1", "--mpp", "1"][..],
            &[input, "o.tif", "--tile-size", "300"],
            &[input, "o.tif", "--quality", "0"],
            &[input, "o.tif", "--region", "1,2,3"],
            &[input, "o.tif", "--region", "1,2,0,4"],
            &[input, "o.tif", "--keep-scanner"],
            &[input],
        ] {
            assert!(convert_args(bad).is_err(), "{bad:?} must be refused");
        }
        // Refused when the job is built.
        let args = convert_args(&[
            input,
            "o.tif",
            "--compression",
            "deflate",
            "--quality",
            "80",
        ])
        .unwrap();
        assert!(args.to_job().unwrap_err().to_string().contains("--quality"));
        let args = convert_args(&["/no/such/slide.svs", "o.tif"]).unwrap();
        assert!(args.to_job().is_err());
    }

    #[test]
    fn regions_parse_as_level_0_rectangles() {
        let region = parse_region(" 100, 200 ,300,400").unwrap();
        assert_eq!(
            (region.x, region.y, region.width, region.height),
            (100, 200, 300, 400)
        );
        assert!(parse_region("1,2,3,4,5").is_err());
        assert!(
            parse_region("a,b,c,d")
                .unwrap_err()
                .contains("X,Y,WIDTH,HEIGHT")
        );
        assert!(parse_region("0,0,-5,5").is_err());
    }
}
