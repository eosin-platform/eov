use crate::backend::WindowGeometry;
use crate::config;
use crate::state::AppState;
use anyhow::{Result, bail};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use common::{
    FilteringMode, RenderBackend, WsiFile,
    cache::{DEFAULT_CACHE_SIZE_BYTES, DEFAULT_MAX_TILES},
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
    Csv,
    Json,
}

impl CliMetadataFormat {
    fn to_common(self) -> MetadataFormat {
        match self {
            Self::Csv => MetadataFormat::Csv,
            Self::Json => MetadataFormat::Json,
        }
    }
}

#[derive(Debug, Subcommand)]
enum DatasetCommand {
    /// Extract fixed-grid image patches from whole-slide images.
    ///
    /// Reads one or more slide files (or directories of slide files) and
    /// writes a deterministic grid of tile images to the output directory.
    /// Only full tiles are emitted; partial edge tiles that would extend
    /// beyond the slide bounds are skipped.
    ///
    /// Supported slide formats: .svs, .tif, .tiff, .ndpi, .vms, .vmu,
    /// .scn, .mrxs, .svslide, .bif, .czi, .dcm
    ///
    /// Examples:
    ///   eov dataset patches slide.svs --out ds/ --tile-size 512 --stride 512
    ///   eov dataset patches slide1.svs slide2.svs --out ds/ --tile-size 512 --stride 512 --metadata csv
    ///   eov dataset patches path/to/slides/ --out ds/ --tile-size 512 --stride 512 --metadata json
    #[command(after_help = "OUTPUT LAYOUT:\n  \
    ds/\n    \
      slides/\n        \
        <slide-stem>/\n          \
        <slide-stem>_x000000_y000000_s512.png\n          \
        <slide-stem>_x000512_y000000_s512.png\n          \
        ...\n    \
      metadata.csv\n    \
      metadata.json\n\n  \
  If --metadata is specified, only that format is written.\n  \
  If omitted, both CSV and JSON are written.")]
    Patches {
        /// One or more slide files or directories containing slides.
        /// Directories are searched recursively for supported slide formats.
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory for extracted tiles and metadata.
        #[arg(long, required = true)]
        out: PathBuf,

        /// Tile width and height in pixels (tiles are square).
        #[arg(long, required = true, value_parser = clap::value_parser!(u32).range(1..))]
        tile_size: u32,

        /// Step size between tile origins in pixels.
        #[arg(long, required = true, value_parser = clap::value_parser!(u32).range(1..))]
        stride: u32,

        /// Emit per-tile metadata in only the given format (csv or json).
        /// If omitted, both CSV and JSON metadata files are written.
        #[arg(long, value_enum)]
        metadata: Option<CliMetadataFormat>,

        /// Number of worker threads for parallel tile extraction.
        /// Each thread opens its own slide file handle for maximum throughput.
        /// Defaults to the number of available CPU cores.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        threads: Option<u32>,

        /// Skip tiles where the fraction of nearly-white pixels exceeds this
        /// threshold (0.0–1.0). For example, 0.9 skips tiles that are ≥90%
        /// white. Omit to keep all tiles.
        #[arg(long, value_parser = clap::value_parser!(f32))]
        white_threshold: Option<f32>,
    },
}

#[derive(Debug, Subcommand)]
enum CliCommand {
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
    /// Open a packaged plugin's .slint window in a standalone process.
    ///
    /// This is an internal subcommand spawned by the host when a Rust plugin
    /// requests a window. It runs its own Slint event loop so the plugin
    /// window doesn't conflict with the main window's wgpu renderer.
    #[command(hide = true)]
    PluginWindow {
        /// Path to the extracted plugin root directory (contains plugin.toml).
        plugin_dir: PathBuf,
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

    #[arg(
        long,
        value_name = "MB",
        default_value_t = (DEFAULT_CACHE_SIZE_BYTES / (1024 * 1024)) as u64,
        value_parser = clap::value_parser!(u64).range(1..),
        global = true
    )]
    cache_size: u64,

    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = DEFAULT_MAX_TILES as u64,
        value_parser = clap::value_parser!(u64).range(1..),
        global = true
    )]
    max_tiles: u64,

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

    /// Start the gRPC extension host server on this port.
    /// When set, external plugins can connect via gRPC to register
    /// viewport filters and exchange pixel data.
    #[arg(long, value_name = "PORT", global = true)]
    extension_host_port: Option<u16>,

    #[arg(long, action = ArgAction::SetTrue, global = true)]
    viewport_lock: bool,
}

enum CommandAction {
    LaunchUi,
    Probe(PathBuf),
    RecentList,
    ConfigPath,
    DatasetPatches(DatasetPatchesConfig),
    PluginWindow(PathBuf),
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
    pub(crate) cache_size_bytes: usize,
    pub(crate) max_tiles: usize,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) window_geometry: WindowGeometry,
    pub(crate) plugin_dir: PathBuf,
    pub(crate) extension_host_port: Option<u16>,
    command: CommandAction,
}

pub(crate) fn parse_launch_options() -> Result<LaunchOptions> {
    let cli = Cli::parse();

    let cache_size_bytes = cli
        .cache_size
        .checked_mul(1024 * 1024)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow::anyhow!("--cache-size is too large for this platform"))?;

    let max_tiles = usize::try_from(cli.max_tiles)
        .map_err(|_| anyhow::anyhow!("--max-tiles is too large for this platform"))?;

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
        Some(CliCommand::Probe { file }) => {
            validate_input_file(&file)?;
            CommandAction::Probe(file)
        }
        Some(CliCommand::Recent {
            command: RecentCommand::List,
        }) => CommandAction::RecentList,
        Some(CliCommand::ConfigPath) => CommandAction::ConfigPath,
        Some(CliCommand::Dataset { command }) => match command {
            DatasetCommand::Patches {
                inputs,
                out,
                tile_size,
                stride,
                metadata,
                threads,
                white_threshold,
            } => {
                let threads = threads.map(|n| n as usize).unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                });
                CommandAction::DatasetPatches(DatasetPatchesConfig {
                    inputs,
                    output_dir: out,
                    tile_size,
                    stride,
                    metadata_format: metadata.map(|m| m.to_common()),
                    threads,
                    white_threshold,
                })
            }
        },
        Some(CliCommand::PluginWindow { plugin_dir }) => CommandAction::PluginWindow(plugin_dir),
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
        extension_host_port: cli.extension_host_port,
        command,
    })
}

pub(crate) fn init_tracing(log_level: Option<CliLogLevel>) {
    let env_filter = match log_level {
        Some(level) => tracing_subscriber::EnvFilter::new(level.as_filter()),
        None => tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
    };

    tracing_subscriber::fmt().with_env_filter(env_filter).init();
}

/// Default plugin directory: `~/.eov/plugins/`
pub(crate) fn default_plugin_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".eov")
        .join("plugins")
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
    let s = path.to_string_lossy();
    if (s.starts_with("~/") || s == "~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(s.strip_prefix("~/").unwrap_or(""));
    }
    path.to_path_buf()
}

pub(crate) fn apply_config_override(config_path: Option<&PathBuf>) -> Result<()> {
    if let Some(path) = config_path {
        config::set_config_path_override(path.clone())?;
    }

    Ok(())
}

pub(crate) fn maybe_run_cli_command(launch_options: &LaunchOptions) -> Result<bool> {
    match &launch_options.command {
        CommandAction::LaunchUi => Ok(false),
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
        CommandAction::DatasetPatches(config) => {
            run_dataset_patches_cli(config)?;
            Ok(true)
        }
        CommandAction::PluginWindow(plugin_root) => {
            crate::plugins::run_plugin_window_standalone(plugin_root)?;
            Ok(true)
        }
    }
}

fn run_dataset_patches_cli(config: &DatasetPatchesConfig) -> Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let cancel = Arc::new(AtomicBool::new(false));
    let progress_tiles = Arc::new(AtomicU64::new(0));
    let progress_current_slide = Arc::new(AtomicU64::new(0));
    let progress_total_slides = Arc::new(AtomicU64::new(0));
    let progress_total_tiles_expected = Arc::new(AtomicU64::new(0));

    // Install Ctrl-C handler for graceful cancellation.
    {
        let cancel = Arc::clone(&cancel);
        let _ = ctrlc::set_handler(move || {
            cancel.store(true, Ordering::Relaxed);
            eprintln!("\nCancelling...");
        });
    }

    let pt = Arc::clone(&progress_tiles);
    let pcs = Arc::clone(&progress_current_slide);
    let pts = Arc::clone(&progress_total_slides);
    let ptte = Arc::clone(&progress_total_tiles_expected);
    let cancel_bg = Arc::clone(&cancel);
    let config_clone = config.clone();

    // Run the pipeline on a background thread so we can drive the progress bar
    // on the main thread.
    let handle = std::thread::Builder::new()
        .name("dataset-patches".into())
        .spawn(move || {
            dataset::run_dataset_patches_with_progress(
                &config_clone,
                &cancel_bg,
                &pt,
                &pcs,
                &pts,
                &ptte,
            )
        })
        .expect("failed to spawn dataset-patches thread");

    // Drive a progress bar on the main thread.
    let pb = indicatif::ProgressBar::new(0);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta} remaining)")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(200));

    loop {
        let total = progress_total_tiles_expected.load(Ordering::Relaxed);
        let done = progress_tiles.load(Ordering::Relaxed);
        if total > 0 {
            pb.set_length(total);
        }
        pb.set_position(done);

        if handle.is_finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Final update.
    let total = progress_total_tiles_expected.load(Ordering::Relaxed);
    let done = progress_tiles.load(Ordering::Relaxed);
    if total > 0 {
        pb.set_length(total);
    }
    pb.set_position(done);
    pb.finish_and_clear();

    let report = handle.join().expect("dataset-patches thread panicked")?;

    if cancel.load(Ordering::Relaxed) {
        eprintln!(
            "Export cancelled. {} tile(s) were written before cancellation.",
            report.total_tiles
        );
        return Ok(());
    }

    // Print input-level errors.
    for (path, err) in &report.input_errors {
        eprintln!("warning: {}: {}", path.display(), err);
    }

    // Print per-slide results.
    for slide in &report.slides {
        match &slide.skipped {
            Some(dataset::SlideSkipReason::OpenError(msg)) => {
                eprintln!("warning: skipped {}: {}", slide.path.display(), msg);
            }
            Some(dataset::SlideSkipReason::TooSmall { width, height }) => {
                eprintln!(
                    "warning: skipped {} ({}×{} is smaller than tile size)",
                    slide.path.display(),
                    width,
                    height,
                );
            }
            None => {
                if slide.tiles_skipped_white > 0 {
                    println!(
                        "{}: {} tile(s) written, {} skipped (white)",
                        slide.path.display(),
                        slide.tiles_written,
                        slide.tiles_skipped_white,
                    );
                } else {
                    println!(
                        "{}: {} tile(s) written",
                        slide.path.display(),
                        slide.tiles_written,
                    );
                }
            }
        }
    }

    // Summary.
    println!();
    println!(
        "Done: {} slide(s) processed, {} skipped, {} tile(s) written, {} tile(s) skipped (white)",
        report.processed_slides,
        report.skipped_slides,
        report.total_tiles,
        report.total_tiles_skipped_white,
    );
    for meta in &report.metadata_paths {
        println!("Metadata: {}", meta.display());
    }

    // Non-zero exit if any slides were skipped due to errors.
    if report
        .slides
        .iter()
        .any(|s| matches!(s.skipped, Some(dataset::SlideSkipReason::OpenError(_))))
    {
        bail!(
            "{} slide(s) could not be opened",
            report
                .slides
                .iter()
                .filter(|s| matches!(s.skipped, Some(dataset::SlideSkipReason::OpenError(_))))
                .count()
        );
    }

    Ok(())
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
    let state = AppState::new();
    if state.recent_files.is_empty() {
        println!("No recent files.");
        return;
    }

    for recent_file in state.recent_files {
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
    fn default_plugin_dir_ends_with_eov_plugins() {
        let dir = default_plugin_dir();
        assert!(
            dir.ends_with(".eov/plugins"),
            "expected default plugin dir to end with .eov/plugins, got {}",
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
