//! Headless batch runs: `eov batch run eovae:analyze SLIDES --out DIR`
//! (AUDIT §14.5).
//!
//! eov starts the plugin without a window and sends each slide as a
//! `BatchRun` event. `analyze` runs the VAE over every tile of the slide at
//! the model's resolution, skipping glass, and writes into the slide's
//! output folder:
//!
//! - `tiles.csv`: one row per tile: its position and size in level-0 pixels
//!   and µm, its state and background flag, and its reconstruction error
//!   (mean and largest absolute error, the anomaly score);
//! - `heatmap.png`: the error map, one pixel per tile (its level-0 scale is
//!   in the summary and in the PNG's text chunks);
//! - `summary.json`: the score distribution and tile counts, and the run's
//!   provenance: model file and SHA-256 (with its external weights),
//!   preprocessing, level and µm/px, device and precision, ONNX Runtime and
//!   plugin versions, warnings. The same summary goes into eov's
//!   `batch.json`.
//!
//! The sidebar's settings apply (model, device, batch size, threads, TF32,
//! resolution, background skipping, preprocessing changes) unless a
//! parameter overrides them for the run; parameters are never saved.
//! Memory stays bounded however large the slide: tiles stream through the
//! pipeline and only one score per tile is kept.

use crate::config::{Config, max_threads};
use crate::contract::CONTRACT_VERSION;
use crate::db::{DbTile, ResultCache, RunWrite, SlideKeys};
use crate::engine::{Backend, DevicePreference, InferenceEngine};
use crate::model::{self, ModelInfo};
use crate::pipeline::{JobControl, JobOutcome, JobSink, JobSpec, Progress, TileOutcome, run_job};
use crate::plugin::{Core, HostPixels, engine_request_for, format_eta, lock};
use crate::render;
use crate::results::{Counts, Need, TileState};
use crate::run::RunSpec;
use crate::sidebar::thousands;
use crate::slide;
use crate::stats::{self, percentile};
use crate::tiling::{Resolution, SlideGeometry, TileCoord, TileGrid, choose_level};
use eov_plugin_api::prelude::*;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The batch command.
pub const ANALYZE: &str = "analyze";
pub const TILES_CSV: &str = "tiles.csv";
pub const HEATMAP_PNG: &str = "heatmap.png";
pub const SUMMARY_JSON: &str = "summary.json";

/// Progress goes to eov at most this often.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

const MAE: &str = "mean absolute error of the reconstruction over the tile's in-slide pixels and RGB channels, 0-1";

fn device_key(device: DevicePreference) -> &'static str {
    match device {
        DevicePreference::Auto => "auto",
        DevicePreference::Cpu => "cpu",
        DevicePreference::Cuda => "cuda",
    }
}

fn resolution_value(resolution: Resolution) -> Value {
    match resolution {
        Resolution::Auto => json!("auto"),
        Resolution::Downsample { value } => json!(value),
    }
}

/// The command eovae offers. The defaults are the saved settings, so eov's
/// `batch.json` records what a run used.
pub fn command_spec(config: &Config) -> BatchCommandSpec {
    let param = |name: &str, description: &str, default: Value| BatchParam {
        name: name.into(),
        description: description.into(),
        default,
    };
    BatchCommandSpec {
        command_id: ANALYZE.into(),
        description: "VAE reconstruction error of every tissue tile: tiles.csv, heatmap.png and summary.json per slide".into(),
        params: vec![
            param(
                "model",
                "the ONNX model (default: the one chosen in eov's VAE sidebar)",
                config.model_path.clone().map(Value::from).unwrap_or(Value::Null),
            ),
            param("device", "auto (CUDA, else the CPU), cpu or cuda", json!(device_key(config.device))),
            param(
                "batch_size",
                "tiles per inference batch, for models without a fixed batch",
                json!(config.batch_size),
            ),
            param("threads", "CPU inference threads (0: every core)", json!(config.threads)),
            param("tf32", "allow TF32 on CUDA (faster, slightly different results)", json!(config.tf32)),
            param(
                "resolution",
                "auto (tiles resampled to the model's µm/px) or a downsample, e.g. 4 (that level's own pixels)",
                resolution_value(config.resolution),
            ),
            param("skip_background", "skip glass tiles", json!(config.skip_background)),
            param(
                "heat_range",
                "error-map colours: auto (p05-p95 of each slide) or lo,hi (mean absolute error; the same scale for every slide)",
                json!("auto"),
            ),
            param(
                "cache",
                "off; store (keep the results for eov's viewer); or reuse (also take tiles analysed before with the same model and settings)",
                json!("off"),
            ),
        ],
    }
}

/// Whether a run uses eovae's result cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheUse {
    /// Every tile is computed; nothing is stored.
    Off,
    /// Results are also stored, so eov's viewer shows them.
    Store,
    /// Tiles stored for exactly this run (same model, weights,
    /// preprocessing, level, device, precision and plugin version) are
    /// taken from the cache; the rest are computed and stored.
    Reuse,
}

impl CacheUse {
    fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Store => "store",
            Self::Reuse => "reuse",
        }
    }
}

/// A run's settings: the saved ones with the run's parameters over them.
#[derive(Clone, Debug)]
pub struct BatchSettings {
    pub config: Config,
    pub model: Option<PathBuf>,
    /// A fixed error-map colour range; `None`: p05–p95 of each slide.
    pub heat_range: Option<(f64, f64)>,
    pub cache: CacheUse,
}

fn integer(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| {
            value
                .as_f64()
                .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                .map(|f| f as u64)
        })
        .or_else(|| value.as_str()?.trim().parse().ok())
}

fn boolean(value: &Value) -> Option<bool> {
    value
        .as_bool()
        .or_else(
            || match value.as_str()?.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => Some(true),
                "false" | "no" | "off" | "0" => Some(false),
                _ => None,
            },
        )
        .or_else(|| value.as_u64().filter(|n| *n <= 1).map(|n| n == 1))
}

fn number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
        .filter(|f| f.is_finite())
}

fn resolution(value: &Value) -> Option<Resolution> {
    if value
        .as_str()
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("auto"))
    {
        return Some(Resolution::Auto);
    }
    number(value)
        .filter(|value| *value >= 1.0)
        .map(|value| Resolution::Downsample { value })
}

/// `auto`, `lo,hi` or `[lo, hi]`.
fn heat_range(value: &Value) -> Option<Option<(f64, f64)>> {
    let pair: Vec<f64> = match value {
        Value::String(text) if text.trim().eq_ignore_ascii_case("auto") => return Some(None),
        Value::String(text) => text
            .split(',')
            .map(|part| part.trim().parse::<f64>().ok())
            .collect::<Option<_>>()?,
        Value::Array(items) => items.iter().map(Value::as_f64).collect::<Option<_>>()?,
        _ => return None,
    };
    match pair[..] {
        [lo, hi] if lo.is_finite() && hi.is_finite() && 0.0 <= lo && lo < hi => {
            Some(Some((lo, hi)))
        }
        _ => None,
    }
}

impl BatchSettings {
    pub fn parse(saved: &Config, params: &Map<String, Value>) -> Result<Self, String> {
        let mut settings = Self {
            config: saved.clone(),
            model: saved.model_path.as_ref().map(PathBuf::from),
            heat_range: None,
            cache: CacheUse::Off,
        };
        for (name, value) in params {
            if value.is_null() {
                continue;
            }
            let invalid =
                |expected: &str| format!("--param {name}: expected {expected}, not {value}");
            let config = &mut settings.config;
            match name.as_str() {
                "model" => {
                    let path = value
                        .as_str()
                        .filter(|path| !path.trim().is_empty())
                        .ok_or_else(|| invalid("the path of an .onnx file"))?;
                    settings.model = Some(PathBuf::from(path));
                }
                "device" => {
                    config.device = value
                        .as_str()
                        .and_then(DevicePreference::parse)
                        .ok_or_else(|| invalid("auto, cpu or cuda"))?;
                }
                "batch_size" => {
                    config.batch_size = integer(value)
                        .filter(|n| (1..=256).contains(n))
                        .ok_or_else(|| invalid("a number from 1 to 256"))?
                        as usize;
                }
                "threads" => {
                    config.threads = integer(value)
                        .ok_or_else(|| invalid("a number of threads (0: every core)"))?
                        .min(max_threads() as u64) as usize;
                }
                "tf32" => config.tf32 = boolean(value).ok_or_else(|| invalid("true or false"))?,
                "skip_background" => {
                    config.skip_background =
                        boolean(value).ok_or_else(|| invalid("true or false"))?;
                }
                "resolution" => {
                    config.resolution = resolution(value)
                        .ok_or_else(|| invalid("auto or a downsample of at least 1"))?;
                }
                "heat_range" => {
                    settings.heat_range = heat_range(value)
                        .ok_or_else(|| invalid("auto or lo,hi with 0 <= lo < hi"))?;
                }
                "cache" => {
                    settings.cache = match value.as_str().map(str::trim) {
                        Some("off") => CacheUse::Off,
                        Some("store") => CacheUse::Store,
                        Some("reuse") => CacheUse::Reuse,
                        _ => return Err(invalid("off, store or reuse")),
                    };
                }
                other => return Err(format!("{ANALYZE} takes no parameter '{other}'")),
            }
        }
        Ok(settings)
    }
}

/// The batch run in progress (eov sends one slide at a time).
#[derive(Default)]
pub struct BatchRunner {
    running: Mutex<Option<Arc<JobControl>>>,
    /// The last run's model, kept for the next slides.
    model: Mutex<Option<Arc<ModelInfo>>>,
}

impl BatchRunner {
    /// Stops the running slide (eov is shutting down).
    pub fn cancel(&self) {
        if let Some(control) = lock(&self.running).as_ref() {
            control.cancel();
        }
    }
}

/// One slide of a batch run.
#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub run_id: String,
    pub command_id: String,
    pub file: FileInfo,
    pub output_dir: PathBuf,
    pub params: Map<String, Value>,
}

fn failure(run_id: &str, message: impl Into<String>) -> BatchFinishArgs {
    BatchFinishArgs {
        run_id: run_id.to_string(),
        ok: false,
        message: Some(message.into()),
        ..Default::default()
    }
}

/// Analyses a slide on a worker thread; eov hears back through
/// `finish_batch`, always (also when the analysis fails or panics).
pub fn start<B: Backend>(core: &Arc<Core<B>>, request: BatchRequest) {
    if request.command_id != ANALYZE {
        let _ = core.host.finish_batch(failure(
            &request.run_id,
            format!(
                "eovae has no batch command '{}' (it offers '{ANALYZE}')",
                request.command_id
            ),
        ));
        return;
    }
    let control = JobControl::new();
    {
        let mut running = lock(&core.batch.running);
        if running.is_some() {
            drop(running);
            let _ = core.host.finish_batch(failure(
                &request.run_id,
                "eovae is still analysing the previous slide",
            ));
            return;
        }
        *running = Some(Arc::clone(&control));
    }
    core.spawn("batch", move |core| {
        let finish = catch_unwind(AssertUnwindSafe(|| analyze(&core, &request, &control)))
            .unwrap_or_else(|_| Err("eovae failed unexpectedly while analysing the slide".into()))
            .unwrap_or_else(|message| failure(&request.run_id, message));
        // Free before answering: eov sends the next slide right away.
        *lock(&core.batch.running) = None;
        let _ = core.host.finish_batch(finish);
    });
}

/// The run's model: the last one when the path is the same.
fn batch_model<B: Backend>(
    core: &Core<B>,
    settings: &BatchSettings,
    control: &JobControl,
) -> Result<Arc<ModelInfo>, String> {
    let path = settings
        .model
        .clone()
        .ok_or("no model: pass --param model=PATH, or choose one in eov's VAE sidebar first")?;
    if let Some(model) = lock(&core.batch.model).as_ref().filter(|m| m.path == path) {
        return Ok(Arc::clone(model));
    }
    let model = model::inspect(&path, control.stop_flag())
        .map(Arc::new)
        .map_err(|error| format!("the model {} cannot be used: {error}", path.display()))?;
    *lock(&core.batch.model) = Some(Arc::clone(&model));
    Ok(model)
}

fn analyze<B: Backend>(
    core: &Arc<Core<B>>,
    request: &BatchRequest,
    control: &Arc<JobControl>,
) -> Result<BatchFinishArgs, String> {
    let started = Instant::now();
    let saved = lock(&core.state).config.clone();
    let settings = BatchSettings::parse(&saved, &request.params)?;
    let say = |message: &str| {
        let _ = core
            .host
            .batch_progress(&request.run_id, None, Some(message));
    };
    say("Loading the model");
    let model = batch_model(core, &settings, control)?;
    say("Preparing the inference engine");
    let engine_request = engine_request_for(&settings.config, &model);
    let mut engine = core
        .acquire_engine(&engine_request, control.stop_flag(), true)
        .map_err(|error| {
            if control.is_cancelled() {
                "cancelled".to_string()
            } else {
                format!("the model cannot run: {error}")
            }
        })?;
    let result = catch_unwind(AssertUnwindSafe(|| {
        analyze_slide(
            core,
            request,
            control,
            &settings,
            &model,
            engine.as_mut(),
            started,
        )
    }));
    // Kept for the next slide.
    core.engine.release(&engine_request.signature(), engine);
    result.unwrap_or_else(|_| Err("eovae failed unexpectedly while analysing the slide".into()))
}

/// Where a batch run's tiles go: one score per tile in memory, and the
/// result cache when the run stores its results.
struct BatchSink<'a> {
    host: &'a Host,
    run_id: &'a str,
    store: Option<Store<'a>>,
    device: String,
    states: Mutex<HashMap<u64, TileState>>,
    reported: Mutex<Option<Instant>>,
}

struct Store<'a> {
    cache: &'a ResultCache,
    slide: SlideKeys,
    run_key: String,
    manifest: String,
    grid: TileGrid,
}

impl JobSink for BatchSink<'_> {
    fn deliver(&self, outcomes: Vec<TileOutcome>) {
        let mut writes = Vec::new();
        {
            let mut states = lock(&self.states);
            for outcome in outcomes {
                states.insert(outcome.coord.key(), outcome.state);
                // Tile pixels are not needed here and are dropped.
                if self.store.is_some() {
                    writes.push(DbTile {
                        coord: outcome.coord,
                        state: outcome.state,
                        embedding: outcome.embedding,
                        payload: None,
                    });
                }
            }
        }
        if let Some(store) = &self.store
            && !writes.is_empty()
        {
            // Blocks while the cache thread is behind (backpressure).
            store.cache.write(RunWrite {
                slide: store.slide.clone(),
                run_key: store.run_key.clone(),
                manifest: store.manifest.clone(),
                cols: store.grid.cols,
                rows: store.grid.rows,
                tiles: writes,
            });
        }
    }

    fn progress(&self, progress: &Progress) {
        {
            let mut reported = lock(&self.reported);
            if progress.done < progress.total
                && reported.is_some_and(|at| at.elapsed() < PROGRESS_INTERVAL)
            {
                return;
            }
            *reported = Some(Instant::now());
        }
        let eta = format_eta(progress);
        let message = format!(
            "{} of {} tiles on {}{}",
            thousands(progress.done as f64),
            thousands(progress.total as f64),
            self.device,
            if eta.is_empty() {
                String::new()
            } else {
                format!(", {eta}")
            }
        );
        let _ = self
            .host
            .batch_progress(self.run_id, Some(progress.fraction()), Some(&message));
    }
}

/// Writes `name` in `dir` through a temporary file, so a failed run leaves
/// no half-written results.
fn write_file(
    dir: &Path,
    name: &str,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
) -> Result<(), String> {
    let path = dir.join(name);
    let temp = dir.join(format!(".{name}.part"));
    let result = std::fs::File::create(&temp).and_then(|file| {
        let mut out = BufWriter::new(file);
        write(&mut out)?;
        out.flush()?;
        drop(out);
        std::fs::rename(&temp, &path)
    });
    result.map_err(|error| {
        let _ = std::fs::remove_file(&temp);
        format!("could not write {name}: {error}")
    })
}

fn write_csv(
    out: &mut dyn Write,
    grid: &TileGrid,
    mpp: Option<(f64, f64)>,
    tiles: &[(TileCoord, TileState)],
) -> std::io::Result<()> {
    writeln!(
        out,
        "col,row,x,y,width,height,width_um,height_um,level,state,background,mae,max_error"
    )?;
    let um = |px: f64, axis: fn((f64, f64)) -> f64| {
        mpp.map(|mpp| format!("{:.2}", px * axis(mpp)))
            .unwrap_or_default()
    };
    for (coord, state) in tiles {
        let rect = grid.valid_footprint(*coord);
        let (label, background, mae, max) = match state {
            TileState::Analysed { mae, max } => {
                ("analysed", 0, format!("{mae:.6}"), format!("{max:.6}"))
            }
            TileState::Background => ("background", 1, String::new(), String::new()),
            TileState::Failed => ("unreadable", 0, String::new(), String::new()),
        };
        writeln!(
            out,
            "{},{},{:.0},{:.0},{:.0},{:.0},{},{},{},{label},{background},{mae},{max}",
            coord.col,
            coord.row,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            um(rect.width, |(x, _)| x),
            um(rect.height, |(_, y)| y),
            grid.level
        )?;
    }
    Ok(())
}

/// The error map's colour range: p05–p95 of the analysed tiles, as in
/// eov's viewer.
fn auto_range(summary: Option<&stats::Summary>) -> (f64, f64) {
    match summary {
        Some(s) if s.count >= 2 && s.p95 > s.p05 => (s.p05, s.p95),
        Some(s) => (0.0, s.max.max(1e-6)),
        None => (0.0, 1.0),
    }
}

/// The score distribution of the analysed tiles.
fn distribution(tiles: &[(TileCoord, TileState)]) -> (Option<stats::Summary>, Value) {
    let mut values: Vec<f32> = tiles.iter().filter_map(|(_, state)| state.mae()).collect();
    let Some(summary) = stats::summarize(&mut values) else {
        return (None, Value::Null);
    };
    let variance = values
        .iter()
        .map(|v| (f64::from(*v) - summary.mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let value = json!({
        "metric": MAE,
        "count": summary.count,
        "mean": summary.mean,
        "std": variance.sqrt(),
        "min": summary.min,
        "p05": summary.p05,
        "p25": percentile(&values, 0.25),
        "median": summary.median,
        "p75": percentile(&values, 0.75),
        "p95": summary.p95,
        "max": summary.max,
    });
    (Some(summary), value)
}

fn analyze_slide<B: Backend>(
    core: &Arc<Core<B>>,
    request: &BatchRequest,
    control: &Arc<JobControl>,
    settings: &BatchSettings,
    model: &Arc<ModelInfo>,
    engine: &mut dyn InferenceEngine,
    started: Instant,
) -> Result<BatchFinishArgs, String> {
    let file = &request.file;
    let config = &settings.config;
    let info = engine.info().clone();
    let identity = slide::resolve(file);
    let geometry = SlideGeometry::from_file_info(file);
    let contract = config.contract_for(model);
    let choice = choose_level(&geometry, config.resolution, contract.mpp)
        .ok_or("the slide reports no pyramid levels")?;
    let grid = TileGrid::for_choice(&geometry, &choice, model.tile_size);
    let output = config.output_for(model);
    let spec = RunSpec::new(model, &output, &contract, &grid, choice.mpp, &info);
    let run_key = spec.key();
    // The cache needs to know the slide again later.
    let cached = settings.cache != CacheUse::Off && identity.persistent;

    // Tiles stored for exactly this run.
    let mut stored = HashMap::new();
    if settings.cache == CacheUse::Reuse && identity.persistent {
        for (coord, state) in core.cache.load_run(&identity.keys, &run_key) {
            let usable = match state {
                TileState::Analysed { .. } => true,
                TileState::Background => config.skip_background,
                TileState::Failed => false,
            };
            if usable && grid.contains(coord) {
                stored.insert(coord.key(), state);
            }
        }
    }
    let from_cache = stored.len();
    let tiles: Vec<(TileCoord, Need)> = grid
        .full_range()
        .iter()
        .filter(|coord| !stored.contains_key(&coord.key()))
        .map(|coord| (coord, Need::Score))
        .collect();
    let sink = BatchSink {
        host: &core.host,
        run_id: &request.run_id,
        store: cached.then(|| Store {
            cache: &core.cache,
            slide: identity.keys.clone(),
            run_key: run_key.clone(),
            manifest: spec.manifest(),
            grid,
        }),
        device: info.label.clone(),
        states: Mutex::new(stored),
        reported: Mutex::new(None),
    };
    let job = JobSpec {
        file_id: file.file_id,
        grid,
        tiles,
        contract: contract.clone(),
        layout: model.layout,
        skip_background: config.skip_background,
        batch_size: config.batch_size,
        readers: 4,
    };
    let outcome = if job.tiles.is_empty() {
        JobOutcome::Completed(Counts::default())
    } else {
        let pixels = HostPixels {
            host: core.host.clone(),
        };
        run_job(&job, &pixels, engine, &sink, control)
    };
    if cached {
        core.cache.flush();
    }

    // Results, in row-major order.
    let mut tiles: Vec<(TileCoord, TileState)> = sink
        .states
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .map(|(key, state)| (TileCoord::from_key(key), state))
        .collect();
    tiles.sort_by_key(|(coord, _)| coord.key());
    let mut counts = Counts::default();
    for (_, state) in &tiles {
        match state {
            TileState::Analysed { .. } => counts.analysed += 1,
            TileState::Background => counts.background += 1,
            TileState::Failed => counts.failed += 1,
        }
    }
    let (complete, problem) = match &outcome {
        JobOutcome::Completed(_) => (true, None),
        JobOutcome::Cancelled(_) => (false, Some("cancelled".to_string())),
        JobOutcome::Failed { error, .. } => (false, Some(error.clone())),
    };
    let (summary_stats, scores) = distribution(&tiles);
    let range = settings
        .heat_range
        .unwrap_or_else(|| auto_range(summary_stats.as_ref()));
    let mpp_xy = geometry.mpp_xy;
    let (step_x, step_y) = (grid.step_x(), grid.step_y());

    let mut warnings = model.warnings.clone();
    let level_mpp = choice.level_mpp(&geometry);
    if choice.mismatched()
        && let (Some(level_mpp), Some(model_mpp)) = (level_mpp, contract.mpp)
    {
        warnings.push(if choice.resample.is_some() {
            format!(
                "The slide's finest level is {level_mpp:.3} µm/px, coarser than the model's {model_mpp:.3} µm/px: tiles were interpolated and errors are inflated."
            )
        } else {
            format!(
                "The slide was analysed at {level_mpp:.3} µm/px but the model was trained at {model_mpp:.3} µm/px: errors are inflated by the difference."
            )
        });
    }
    if contract.mpp.is_some() && choice.mpp.is_none() {
        warnings.push(format!(
            "The slide has no µm/px calibration, so the model's training resolution could not be matched; level {} was used.",
            grid.level
        ));
    }
    if let Some(reason) = &info.fallback {
        warnings.push(format!("Inference ran on the CPU: {reason}"));
    }
    if counts.failed > 0 {
        warnings.push(format!("{} tiles could not be read.", counts.failed));
    }
    if settings.cache != CacheUse::Off && !identity.persistent {
        warnings.push(
            "The slide has no identity or readable file to recognise it by, so the result cache was not used."
                .to_string(),
        );
    }
    let overridden = config
        .overrides
        .get(&model.sha256)
        .is_some_and(|o| !o.is_empty());
    let mut covered = vec![model.file_name.clone()];
    covered.extend(model.summary.external_data.iter().cloned());

    let mut summary = json!({
        "command": ANALYZE,
        "complete": complete,
        "plugin": {
            "id": "eovae",
            "version": env!("CARGO_PKG_VERSION"),
            "contract_version": CONTRACT_VERSION,
        },
        "run_key": run_key,
        "model": {
            "file": model.file_name,
            "sha256": model.sha256,
            "sha256_covers": covered,
            "bytes": model.bytes,
            "card": model.card,
            "input": model.input.name,
            "input_type": model.input_elem.label(),
            "output": output,
            "layout": model.layout,
            "tile_size": model.tile_size,
            "fixed_batch": model.fixed_batch,
            "trained_mpp": contract.mpp,
        },
        "preprocessing": {
            "source": if overridden {
                format!("eovae settings (the model declares: {})", model.contract_source.label())
            } else {
                model.contract_source.label().to_string()
            },
            "description": contract.summary(),
            "contract": contract,
        },
        "resolution": {
            "requested": resolution_value(config.resolution),
            "level": grid.level,
            "downsample": grid.downsample,
            "level_mpp": level_mpp,
            "mpp": choice.mpp,
            "slide_mpp": mpp_xy.map(|(x, y)| [x, y]),
            "model_mpp": contract.mpp,
            "resampled": grid.resampled(),
            "level0_px_per_model_px": [grid.scale_x, grid.scale_y],
            "ratio_to_model": choice.mismatch,
            "mismatched": choice.mismatched(),
        },
        "device": {
            "requested": device_key(config.device),
            "used": info.device.key(),
            "label": info.label,
            "precision": info.precision,
            "fallback": info.fallback,
            "batch_size": model.fixed_batch.unwrap_or(config.batch_size),
            "threads": config.effective_threads(),
        },
        "onnx_runtime": core.backend.runtime_version(),
        "tiles": {
            "cols": grid.cols,
            "rows": grid.rows,
            "total": grid.tile_count(),
            "analysed": counts.analysed,
            "background": counts.background,
            "unreadable": counts.failed,
            "not_analysed": grid.tile_count().saturating_sub(tiles.len() as u64),
            "from_cache": from_cache,
            "skip_background": config.skip_background,
            "tile_px": grid.tile,
            "tile_level0_px": [step_x, step_y],
            "tile_um": mpp_xy.map(|(x, y)| [step_x * x, step_y * y]),
        },
        "scores": scores,
        "heatmap": {
            "file": HEATMAP_PNG,
            "width": grid.cols,
            "height": grid.rows,
            "level0_px_per_pixel": [step_x, step_y],
            "um_per_pixel": mpp_xy.map(|(x, y)| [step_x * x, step_y * y]),
            "range": [range.0, range.1],
            "range_from": if settings.heat_range.is_some() { "heat_range parameter" } else { "p05-p95 of this slide" },
            "colors": "viridis from range[0] to range[1]; red: unreadable; transparent: glass or not analysed",
        },
        "cache": settings.cache.key(),
        "warnings": warnings,
    });

    // The files.
    let dir = &request.output_dir;
    let mut outputs = Vec::new();
    let mut errors = Vec::new();
    if !tiles.is_empty() {
        match write_file(dir, TILES_CSV, |out| write_csv(out, &grid, mpp_xy, &tiles)) {
            Ok(()) => outputs.push(TILES_CSV.to_string()),
            Err(error) => errors.push(error),
        }
        let image = render::tile_map(&grid, &tiles, range.0 as f32, range.1 as f32);
        let text = [
            ("Software", format!("eovae {}", env!("CARGO_PKG_VERSION"))),
            ("eovae:level0_px_per_pixel", format!("{step_x},{step_y}")),
            ("eovae:mae_range", format!("{},{}", range.0, range.1)),
            ("eovae:level", grid.level.to_string()),
        ];
        let text: Vec<(&str, &str)> = text.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let png = crate::png::encode_rgba(image.width, image.height, &image.data, &text);
        match write_file(dir, HEATMAP_PNG, |out| out.write_all(&png)) {
            Ok(()) => outputs.push(HEATMAP_PNG.to_string()),
            Err(error) => errors.push(error),
        }
    }
    summary["seconds"] = json!(started.elapsed().as_secs_f64());
    let mut listed = outputs.clone();
    listed.push(SUMMARY_JSON.to_string());
    summary["outputs"] = json!(listed);
    match write_file(dir, SUMMARY_JSON, |out| {
        serde_json::to_writer_pretty(&mut *out, &summary).map_err(std::io::Error::other)?;
        writeln!(out)
    }) {
        Ok(()) => outputs.push(SUMMARY_JSON.to_string()),
        Err(error) => errors.push(error),
    }

    let described = format!(
        "{} tissue tiles scored, {} background, {} unreadable at level {} on {}",
        thousands(counts.analysed as f64),
        thousands(counts.background as f64),
        thousands(counts.failed as f64),
        grid.level,
        info.label
    );
    let message = match (problem, errors.is_empty()) {
        (None, true) => described,
        (None, false) => errors.join("; "),
        (Some(problem), _) => {
            let mut parts = vec![format!("{problem} ({described})")];
            parts.extend(errors.iter().cloned());
            parts.join("; ")
        }
    };
    Ok(BatchFinishArgs {
        run_id: request.run_id.clone(),
        ok: complete && errors.is_empty(),
        message: Some(message),
        outputs,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn parameters_override_the_saved_settings() {
        let saved = Config {
            model_path: Some("/models/vae.onnx".into()),
            batch_size: 8,
            ..Default::default()
        };
        let settings = BatchSettings::parse(&saved, &Map::new()).unwrap();
        assert_eq!(
            settings.model.as_deref(),
            Some(Path::new("/models/vae.onnx"))
        );
        assert_eq!(settings.config.batch_size, 8);
        assert_eq!(settings.cache, CacheUse::Off);
        let settings = BatchSettings::parse(
            &saved,
            &params(json!({
                "model": "other.onnx",
                "device": "CPU",
                "batch_size": "32",
                "threads": 2,
                "tf32": "no",
                "resolution": 4,
                "skip_background": false,
                "heat_range": "0.01, 0.05",
                "cache": "reuse",
            })),
        )
        .unwrap();
        assert_eq!(settings.model.as_deref(), Some(Path::new("other.onnx")));
        assert_eq!(settings.config.device, DevicePreference::Cpu);
        assert_eq!(settings.config.batch_size, 32);
        assert_eq!(settings.config.threads, 2.min(max_threads()));
        assert!(!settings.config.tf32 && !settings.config.skip_background);
        assert_eq!(
            settings.config.resolution,
            Resolution::Downsample { value: 4.0 }
        );
        assert_eq!(settings.heat_range, Some((0.01, 0.05)));
        assert_eq!(settings.cache, CacheUse::Reuse);
        // The saved settings are not changed.
        assert_eq!(saved.batch_size, 8);
        let auto = BatchSettings::parse(
            &saved,
            &params(json!({"resolution": "auto", "heat_range": [0, 1]})),
        )
        .unwrap();
        assert_eq!(auto.config.resolution, Resolution::Auto);
        assert_eq!(auto.heat_range, Some((0.0, 1.0)));
    }

    #[test]
    fn bad_parameters_say_what_is_expected() {
        let saved = Config::default();
        for (name, value, expected) in [
            ("device", json!("tpu"), "auto, cpu or cuda"),
            ("batch_size", json!(0), "1 to 256"),
            ("tf32", json!("maybe"), "true or false"),
            ("resolution", json!(0.5), "downsample of at least 1"),
            ("heat_range", json!("0.5,0.1"), "lo < hi"),
            ("cache", json!("sometimes"), "off, store or reuse"),
            ("model", json!(""), "path"),
        ] {
            let mut map = Map::new();
            map.insert(name.into(), value);
            let error = BatchSettings::parse(&saved, &map).unwrap_err();
            assert!(error.contains(expected), "{name}: {error}");
        }
        let mut map = Map::new();
        map.insert("colour".into(), json!("red"));
        assert!(
            BatchSettings::parse(&saved, &map)
                .unwrap_err()
                .contains("colour")
        );
    }

    #[test]
    fn the_command_offers_the_saved_settings_as_defaults() {
        let config = Config {
            model_path: Some("/m.onnx".into()),
            device: DevicePreference::Cuda,
            batch_size: 24,
            ..Default::default()
        };
        let spec = command_spec(&config);
        assert_eq!(spec.command_id, ANALYZE);
        let default = |name: &str| {
            spec.params
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.default.clone())
                .unwrap()
        };
        assert_eq!(default("model"), json!("/m.onnx"));
        assert_eq!(default("device"), json!("cuda"));
        assert_eq!(default("batch_size"), json!(24));
        assert_eq!(default("cache"), json!("off"));
        // Every default parses.
        let all: Map<String, Value> = spec
            .params
            .iter()
            .map(|p| (p.name.clone(), p.default.clone()))
            .collect();
        let settings = BatchSettings::parse(&config, &all).unwrap();
        assert_eq!(settings.config, config);
        assert!(command_spec(&Config::default()).params[0].default.is_null());
    }
}
