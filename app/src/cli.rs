use crate::backend::WindowGeometry;
use crate::config;
use crate::update;
use anyhow::{Result, bail};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use eov_common::{
    FilteringMode, RenderBackend, WsiFile,
    dataset::{self, DatasetPatchesConfig, MetadataFormat},
    format_optional_decimal,
};
use std::path::{Path, PathBuf};

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
    fn as_filter(self) -> &'static str {
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
        plugin: String,
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
        plugin: String,
        #[arg(short = 'o', long, value_enum, default_value_t = CliPluginOutputFormat::Plain)]
        output: CliPluginOutputFormat,
    },
    /// Keep a plugin installed but do not load it.
    Disable {
        id: String,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Load a previously disabled plugin again.
    Enable {
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
        lockfile: PathBuf,
        /// Take packages with matching SHA-256 from this directory instead
        /// of downloading them.
        #[arg(long, value_name = "DIR")]
        from: Option<PathBuf>,
        /// Also remove installed plugins the lockfile does not list.
        #[arg(long)]
        prune: bool,
        #[arg(short = 'y', long)]
        yes: bool,
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
    Probe {
        file: PathBuf,
    },
    Recent {
        #[command(subcommand)]
        command: RecentCommand,
    },
    ConfigPath,
    /// Dataset generation utilities for ML workflows.
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
}

#[derive(Debug, Parser)]
#[command(
    name = "eov",
    version,
    about = "A lightweight, cross-platform WSI viewer.",
    propagate_version = true,
    after_help = "PANE LAYOUT:\n  \
    Positional arguments define pane splits. Each argument is a comma-separated\n  \
    list of files to open as tabs in one pane. The last file in each group\n  \
    becomes the active tab.\n\n  \
    Examples:\n    \
      eov slide.svs                       # one pane, one tab\n    \
      eov a.svs,b.svs                     # one pane, two tabs (b.svs active)\n    \
      eov a.svs,b.svs c.tif d.svs         # three panes"
)]
struct Cli {
    #[arg(value_name = "FILES")]
    pane_groups: Vec<String>,

    #[command(subcommand)]
    command: Option<CliCommand>,

    #[arg(short, long, global = true)]
    debug: bool,

    #[arg(long, value_enum, default_value_t = CliBackend::Auto, global = true)]
    backend: CliBackend,

    #[arg(long, action = ArgAction::SetTrue, global = true)]
    cpu: bool,

    #[arg(long, action = ArgAction::SetTrue, global = true)]
    gpu: bool,

    #[arg(long, value_enum, default_value_t = CliFilteringMode::Auto, global = true)]
    filtering_mode: CliFilteringMode,

    #[arg(long, value_enum, global = true)]
    log_level: Option<CliLogLevel>,

    /// Tile-cache budget in MiB (default: sized for the display).
    #[arg(
        long,
        value_name = "MB",
        value_parser = clap::value_parser!(u64).range(1..),
        global = true
    )]
    cache_size: Option<u64>,

    /// Maximum number of cached tiles (default: sized for the display).
    #[arg(
        long,
        value_name = "COUNT",
        value_parser = clap::value_parser!(u64).range(1..),
        global = true
    )]
    max_tiles: Option<u64>,

    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PX",
        value_parser = clap::value_parser!(u32).range(1..),
        global = true
    )]
    window_width: Option<u32>,

    #[arg(
        long,
        value_name = "PX",
        value_parser = clap::value_parser!(u32).range(1..),
        global = true
    )]
    window_height: Option<u32>,

    #[arg(long, value_name = "PX", global = true)]
    window_x: Option<i32>,

    #[arg(long, value_name = "PX", global = true)]
    window_y: Option<i32>,

    /// Directory to search for plugin packages.
    /// Defaults to ~/.eov/plugins/ and scans for `.eop` files.
    #[arg(long, value_name = "PATH", global = true)]
    plugin_dir: Option<PathBuf>,

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
    Probe(PathBuf),
    RecentList,
    ConfigPath,
    DatasetPatches(Box<DatasetJob>),
}

pub(crate) struct PaneSpec {
    pub(crate) files: Vec<PathBuf>,
}

pub(crate) struct LaunchOptions {
    pub(crate) debug_mode: bool,
    pub(crate) panes_to_open: Vec<PaneSpec>,
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

pub(crate) fn parse_launch_options() -> Result<LaunchOptions> {
    let cli = Cli::parse();

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

    if cli.command.is_some() && !cli.pane_groups.is_empty() {
        bail!("file arguments cannot be combined with a subcommand");
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

    let panes_to_open = cli
        .pane_groups
        .into_iter()
        .map(|group| {
            let files = group
                .split(',')
                .map(|entry| {
                    let path = PathBuf::from(entry.trim());
                    validate_launch_input_path(&path)?;
                    Ok(path)
                })
                .collect::<Result<Vec<_>>>()?;
            if files.is_empty() {
                bail!("empty pane group in arguments");
            }
            Ok(PaneSpec { files })
        })
        .collect::<Result<Vec<_>>>()?;

    let command = match cli.command {
        Some(CliCommand::Update(args)) => CommandAction::ApplicationUpdate(args),
        Some(CliCommand::Plugin { command }) => CommandAction::Plugin(command),
        Some(CliCommand::Probe { file }) => {
            validate_input_file(&file)?;
            CommandAction::Probe(file)
        }
        Some(CliCommand::Recent {
            command: RecentCommand::List,
        }) => CommandAction::RecentList,
        Some(CliCommand::ConfigPath) => CommandAction::ConfigPath,
        Some(CliCommand::Dataset { command }) => match command {
            DatasetCommand::Patches(args) => {
                CommandAction::DatasetPatches(Box::new(args.into_job()))
            }
        },
        None => CommandAction::LaunchUi,
    };

    let plugin_dir = resolve_plugin_dir(cli.plugin_dir.as_deref());

    Ok(LaunchOptions {
        debug_mode: cli.debug,
        panes_to_open,
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
        safe_mode: cli.safe_mode || safe_mode_from_env(),
        command,
    })
}

/// Whether `EOV_SAFE_MODE` asks for safe mode (any value but empty or `0`).
fn safe_mode_from_env() -> bool {
    std::env::var_os("EOV_SAFE_MODE").is_some_and(|value| !value.is_empty() && value != "0")
}

pub(crate) fn init_tracing(log_level: Option<CliLogLevel>) {
    let env_filter = match log_level {
        Some(level) => tracing_subscriber::EnvFilter::new(level.as_filter()),
        None => tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
    }
    .add_directive(
        "sctk_adwaita::buttons=error"
            .parse()
            .expect("valid sctk_adwaita buttons log directive"),
    );

    tracing_subscriber::fmt().with_env_filter(env_filter).init();
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

pub(crate) fn maybe_run_cli_command(launch_options: &LaunchOptions) -> Result<bool> {
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
        CommandAction::Probe(path) => {
            probe_file(path)?;
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
        CommandAction::DatasetPatches(job) => {
            run_dataset_patches_cli(job)?;
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

fn validate_input_file(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("input file does not exist: {}", path.display());
    }

    if !path.is_file() {
        bail!("input path is not a file: {}", path.display());
    }

    Ok(())
}

fn validate_launch_input_path(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("input path does not exist: {}", path.display());
    }

    if !path.is_file() && !path.is_dir() {
        bail!("input path is not a file or directory: {}", path.display());
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

fn probe_file(path: &Path) -> Result<()> {
    let wsi = WsiFile::open(path)?;
    let properties = wsi.properties();

    println!("File: {}", properties.path.display());
    println!("Filename: {}", properties.filename);
    println!("Dimensions: {}x{}", properties.width, properties.height);
    println!("Levels: {}", properties.levels.len());
    println!(
        "Vendor: {}",
        properties.vendor.as_deref().unwrap_or("unknown")
    );
    println!(
        "MPP: {} x {}",
        format_optional_decimal(properties.mpp_x),
        format_optional_decimal(properties.mpp_y)
    );
    println!(
        "Objective: {}",
        format_optional_decimal(properties.objective_power)
    );
    println!(
        "Scan date: {}",
        properties.scan_date.as_deref().unwrap_or("unknown")
    );
    println!("Base tile size: {}", wsi.tile_size());
    println!();
    println!("Levels:");

    for level in &properties.levels {
        let tile_width = level
            .tile_width
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let tile_height = level
            .tile_height
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        println!(
            "  L{}: {}x{} downsample {:.2} tile {}x{} preferred-tile {}",
            level.level,
            level.width,
            level.height,
            level.downsample,
            tile_width,
            tile_height,
            wsi.tile_size_for_level(level.level)
        );
    }

    Ok(())
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
}
