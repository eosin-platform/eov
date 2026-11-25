//! The plugin: state, events, model loading, engines and jobs.
//!
//! # Threads and locks
//!
//! - Host events and UI callbacks arrive on the UI thread. Their handlers
//!   only update [`State`], mark work for the presenter and start threads;
//!   nothing slow runs there (VAE-11, VAE-13).
//! - Our locks (`state`, `results`, the engine cell) are held only for short
//!   copies and **never across a host call**, so the lock-order inversions
//!   and re-entrancy deadlocks of eovae 0.x cannot happen (VAE-1, VAE-2).
//!   When two are needed they are taken one after the other, never nested.
//! - Analysis jobs run on their own thread. Starting a job cancels the
//!   previous one, and the new thread joins it before it starts, so exactly
//!   one job writes results at a time (VAE-12).
//! - The presenter thread ([`crate::presenter`]) pushes everything visible
//!   (sidebar, overlays, HUD, status, settings) at a bounded rate.

use crate::config::Config;
use crate::contract::Contract;
use crate::db::{DbTile, ResultCache, RunWrite};
use crate::engine::{Backend, EngineInfo, EngineRequest, InferenceEngine};
use crate::message::{Message, cancelled};
use crate::model::{self, ModelInfo};
use crate::pipeline::{
    JobControl, JobOutcome, JobSink, JobSpec, PixelSource, Progress, TileOutcome, run_job,
};
use crate::render::Mode;
use crate::results::{Counts, Need, ResultKey, ResultStore};
use crate::run::RunSpec;
use crate::slide::{self, SlideIdentity};
use crate::tiling::{LevelChoice, SlideGeometry, TileCoord, TileGrid, choose_level};
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::i18n;
use eov_plugin_api::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const TOOLBAR_BUTTON: &str = "eovae";
pub const HUD_GRID: &str = "grid";
pub const HUD_VIEW: &str = "view";
pub const SIDEBAR_UI: &str = "ui/eovae-sidebar.slint";
pub const SIDEBAR_COMPONENT: &str = "EovaeSidebar";

const TOOL_ICON: &str = include_str!("../ui/icons/tool.svg");
const GRID_ICON: &str = include_str!("../ui/icons/grid.svg");
const VIEW_ICON: &str = include_str!("../ui/icons/visualize-transform.svg");

// The tooltips of eovae's buttons, handed to eov at the start and again
// when the language changes.

fn toolbar_tooltip() -> String {
    tr!("VAE reconstruction analysis")
}

fn grid_tooltip() -> String {
    tr!("VAE tile grid")
}

fn view_tooltip() -> String {
    tr!("VAE overlay")
}

/// Automatic analysis stops at this many visible tiles (zoomed out, the
/// view would turn into a whole-slide run, VAE-6)…
pub const AUTO_MAX_TILES: u64 = 400;
/// …or when tiles are smaller than this on screen.
pub const AUTO_MIN_TILE_PX: f64 = 24.0;

/// The error map of a slide (chunked rasters), shown in the panes whose
/// overlay is the error map.
pub fn heat_layer(file_id: FileId) -> String {
    format!("vae-{file_id}")
}

/// One pane's reconstruction or difference mosaic, shown only in that pane.
pub fn mosaic_layer(file_id: FileId, pane: PaneId) -> String {
    format!("vae-{file_id}-pane-{pane}")
}

pub fn grid_layer(file_id: FileId) -> String {
    format!("vae-grid-{file_id}")
}

pub fn mark_layer(file_id: FileId) -> String {
    format!("vae-mark-{file_id}")
}

/// Poison-tolerant locking: a panic elsewhere must not wedge the plugin.
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Warns the user in their language and writes the same text in English to
/// eov's log, both from `write` (see [`i18n::shown_and_english`]). eov
/// shows a plugin's logged warnings and errors to the user as they are
/// (PLG-15), so `Host::warn` and `Host::error` would show the log's
/// English: problems are logged at info level and notified separately.
fn warn_user(host: &Host, write: impl Fn() -> String) {
    let (shown, english) = i18n::shown_and_english(write);
    host.info(english);
    host.notify(NotifyLevel::Warning, shown);
}

/// `text` for eov's log: without the marks [`i18n::isolate`] puts around
/// names in right-to-left languages.
fn log_text(text: &str) -> String {
    text.replace(['\u{2068}', '\u{2069}'], "")
}

/// What a panic said, when it said it in text.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    payload
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub enum ModelState {
    None,
    Loading { path: PathBuf },
    Ready { info: Arc<ModelInfo> },
    Failed { path: PathBuf, error: Message },
}

/// What the engine cell last reported.
#[derive(Default)]
pub struct EngineStatus {
    /// Progress of a build in progress.
    pub building: Option<String>,
    /// Info of the engine built for a results signature.
    pub info: Option<(String, EngineInfo)>,
    /// The last build failure: (signature, error).
    pub error: Option<(String, Message)>,
}

/// What one pane shows of a slide.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaneOverlay {
    pub mode: Mode,
    pub grid: bool,
}

pub struct FileState {
    pub info: FileInfo,
    pub geometry: SlideGeometry,
    /// `None` while being resolved.
    pub identity: Option<SlideIdentity>,
    /// Each pane's overlay of this slide (panes are the host's stable ids).
    pub panes: BTreeMap<PaneId, PaneOverlay>,
    /// The last overlay chosen in any pane: what every pane shows when the
    /// host cannot show layers in some panes only.
    pub shared: PaneOverlay,
    /// The host's region of interest on this slide.
    pub roi: Option<Rect>,
}

impl FileState {
    fn new(file: &FileInfo) -> Self {
        Self {
            geometry: SlideGeometry::from_file_info(file),
            info: file.clone(),
            identity: None,
            panes: BTreeMap::new(),
            shared: PaneOverlay::default(),
            roi: None,
        }
    }

    /// What `pane` shows of this slide. `per_pane` is false when the host
    /// draws a slide's layers in all its panes alike.
    pub fn overlay(&self, pane: Option<PaneId>, per_pane: bool) -> PaneOverlay {
        match pane.filter(|_| per_pane) {
            Some(pane) => self.panes.get(&pane).copied().unwrap_or_default(),
            None => self.shared,
        }
    }

    /// Changes what `pane` shows of this slide.
    pub fn update_overlay(
        &mut self,
        pane: Option<PaneId>,
        per_pane: bool,
        change: impl FnOnce(&mut PaneOverlay),
    ) {
        let mut overlay = self.overlay(pane, per_pane);
        change(&mut overlay);
        if let Some(pane) = pane.filter(|_| per_pane) {
            self.panes.insert(pane, overlay);
        }
        self.shared = overlay;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PaneState {
    pub file_id: Option<FileId>,
    pub view: Option<ViewInfo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Viewport,
    WholeSlide,
    /// Tiles intersecting the host's region of interest.
    Roi,
    /// Visible tiles while an overlay is shown.
    Auto,
}

impl JobKind {
    pub fn explicit(self) -> bool {
        self != Self::Auto
    }

    /// The running job, as the sidebar names it.
    pub fn title(self) -> String {
        match self {
            Self::WholeSlide => tr!("Whole slide"),
            Self::Viewport => tr!("Viewport"),
            Self::Roi => tr!("Region of interest"),
            Self::Auto => tr!("Visible tiles"),
        }
    }

    /// What a finished job's summary is about, as a label ("Viewport:
    /// everything was already analysed."); sentences name it themselves
    /// (see [`JobSummary::text`]).
    fn subject(self) -> String {
        match self {
            Self::WholeSlide => tr!("Whole slide"),
            Self::Viewport => tr!("Viewport"),
            Self::Roi => tr!("Region of interest"),
            Self::Auto => tr!("Automatic analysis"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct JobRequest {
    pub kind: JobKind,
    pub file_id: FileId,
    pub pane: Option<PaneId>,
    pub view: Option<ViewInfo>,
}

pub struct RunningJob {
    pub id: u64,
    pub request: JobRequest,
    pub control: Arc<JobControl>,
    pub host_job: Option<String>,
    pub progress: Progress,
    pub phase: String,
    pub started: Instant,
}

/// How the last job ended; the sidebar shows it until the next one, in
/// the language of the interface (see [`Self::text`]).
#[derive(Clone, Debug)]
pub struct JobSummary {
    pub kind: JobKind,
    pub outcome: JobOutcome,
    pub elapsed: Duration,
}

/// How a job's run ended.
struct JobEnd {
    outcome: JobOutcome,
    /// The job was in eov's job panel and failed there: eov has told the
    /// user (its "… failed" notification, with the error).
    reported: bool,
}

impl From<JobOutcome> for JobEnd {
    fn from(outcome: JobOutcome) -> Self {
        Self {
            outcome,
            reported: false,
        }
    }
}

pub struct State {
    pub config: Config,
    /// When the config last changed and is not yet saved.
    pub config_changed: Option<Instant>,
    pub model: ModelState,
    pub model_generation: u64,
    pub model_cancel: Arc<AtomicBool>,
    pub engine: EngineStatus,
    pub files: BTreeMap<FileId, FileState>,
    pub panes: BTreeMap<PaneId, PaneState>,
    pub focused_pane: Option<PaneId>,
    pub active_file: Option<FileId>,
    pub job: Option<RunningJob>,
    pub last_job: Option<JobSummary>,
    /// Automatic analysis waits for the user to zoom in: the number of
    /// visible tiles.
    pub auto_hint: Option<u64>,
    pub sidebar: Option<Surface>,
    /// Pending dialog request ids.
    pub model_dialog: Option<String>,
    pub runtime_dialog: Option<String>,
    pub export_dialog: Option<(String, FileId)>,
    pub clear_confirm: Option<String>,
    pub hover: Option<(FileId, TileCoord)>,
    pub pulse: Option<(FileId, TileCoord, Instant)>,
    /// Whether the host can show a layer in some panes only
    /// (`layer.set_pane_visibility`); without it every pane of a slide shows
    /// the same overlay.
    pub pane_layers: bool,
    /// The `model.fetch` request waiting for eov's answer.
    pub model_fetch: Option<String>,
    /// Why the last download did not happen (eov has told the user too).
    pub download_note: Option<String>,
    /// The main file of the download target, when eov's model store has
    /// it (looked up when eovae starts and after the address changes).
    pub download_stored: Option<String>,
    /// The host keeps models (eov with `model.fetch`).
    pub model_store: bool,
}

impl State {
    fn new(config: Config) -> Self {
        Self {
            model_fetch: None,
            download_note: None,
            download_stored: None,
            model_store: true,
            config,
            config_changed: None,
            model: ModelState::None,
            model_generation: 0,
            model_cancel: Arc::new(AtomicBool::new(false)),
            engine: EngineStatus::default(),
            files: BTreeMap::new(),
            panes: BTreeMap::new(),
            focused_pane: None,
            active_file: None,
            job: None,
            last_job: None,
            auto_hint: None,
            sidebar: None,
            model_dialog: None,
            runtime_dialog: None,
            export_dialog: None,
            clear_confirm: None,
            hover: None,
            pulse: None,
            pane_layers: true,
        }
    }

    /// What `pane` shows of `file_id`.
    pub fn overlay(&self, file_id: FileId, pane: Option<PaneId>) -> PaneOverlay {
        self.files
            .get(&file_id)
            .map(|file| file.overlay(pane, self.pane_layers))
            .unwrap_or_default()
    }

    pub fn model_info(&self) -> Option<&Arc<ModelInfo>> {
        match &self.model {
            ModelState::Ready { info } => Some(info),
            _ => None,
        }
    }

    pub fn touch_config(&mut self) {
        self.config_changed = Some(Instant::now());
    }

    /// The engine the current settings ask for.
    pub fn engine_request(&self) -> Option<EngineRequest> {
        Some(engine_request_for(&self.config, self.model_info()?))
    }

    /// The file a pane shows.
    pub fn pane_file(&self, pane: PaneId) -> Option<FileId> {
        self.panes.get(&pane).and_then(|p| p.file_id)
    }
}

/// What "Download" asks eov for: the site's own model when its SHA-256 is
/// set, else the reference model, whose card pins both files. Without an
/// address here, eov uses the site's copy named in its policy, if any.
pub fn download_args(config: &Config) -> ModelFetchArgs {
    let reference = model::reference();
    match &config.download_sha256 {
        Some(sha256) if *sha256 != reference.main().sha256 => ModelFetchArgs {
            url: config.download_url.clone(),
            sha256: sha256.clone(),
            // What eov's dialogs call the model, and its name in eov's
            // model store, which keeps it: the address's file name, the
            // same in every language. Without an address, eov names the
            // model itself, in the language of the interface.
            name: config.download_url.as_deref().and_then(url_file_name),
            ..Default::default()
        },
        _ => {
            let main = reference.main();
            ModelFetchArgs {
                url: config
                    .download_url
                    .clone()
                    .or(reference.url.map(str::to_string)),
                sha256: main.sha256.to_string(),
                size: Some(main.size),
                file_name: Some(main.file_name.to_string()),
                name: Some(reference.name.to_string()),
                version: None,
                // The weights, from the same folder as the model.
                companions: reference.files[1..]
                    .iter()
                    .map(|file| ModelFileSpec {
                        sha256: file.sha256.to_string(),
                        url: None,
                        file_name: Some(file.file_name.to_string()),
                        size: Some(file.size),
                    })
                    .collect(),
            }
        }
    }
}

/// The file name at the end of an address (`our%20vae.onnx` is
/// `our vae.onnx`, as eov names the file it downloads); `None` when the
/// address names no file.
fn url_file_name(url: &str) -> Option<String> {
    let url = url.split(['?', '#']).next().unwrap_or_default();
    // The path starts after the scheme and the server.
    let path = match url.split_once("://") {
        Some((_, rest)) => rest.split_once('/')?.1,
        None => url,
    };
    let segment = path.rsplit('/').next().filter(|name| !name.is_empty())?;
    Some(percent_decoded(segment).unwrap_or_else(|| segment.to_string()))
}

/// `%XX` escapes decoded; `None` when they do not make printable text.
fn percent_decoded(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if let [b'%', high, low, ..] = bytes[i..]
            && let (Some(high), Some(low)) =
                (char::from(high).to_digit(16), char::from(low).to_digit(16))
        {
            decoded.push((high * 16 + low) as u8);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded)
        .ok()
        .filter(|text| !text.chars().any(char::is_control))
}

/// The engine `config` asks for to run `model`.
pub fn engine_request_for(config: &Config, model: &Arc<ModelInfo>) -> EngineRequest {
    EngineRequest {
        output: config.output_for(model),
        latent: model.latent.as_ref().map(|l| l.name.clone()),
        model: Arc::clone(model),
        device: config.device,
        threads: config.effective_threads(),
        tf32: config.tf32,
        batch_size: config.batch_size,
        runtime_path: config.runtime_path.as_ref().map(PathBuf::from),
    }
}

/// Engine settings that change results (not threads or batch size).
pub fn results_signature(request: &EngineRequest) -> String {
    format!(
        "{}|{}|{:?}|{}",
        request.model.sha256, request.output, request.device, request.tf32
    )
}

/// Why a slide cannot be analysed right now.
#[derive(Clone, Debug, PartialEq)]
pub enum NoRun {
    NoModel,
    ModelLoading,
    ModelFailed(Message),
    EngineNotReady,
    EngineFailed(Message),
    SlideResolving,
    NoLevels,
    UnknownFile,
}

impl NoRun {
    /// Why, in the interface's language and in English (when it ends a
    /// job, eov's log gets the English).
    pub fn message(&self) -> Message {
        Message::new(|| self.text())
    }

    /// Why, in the language being written (see [`Message`]).
    pub fn text(&self) -> String {
        match self {
            Self::NoModel => tr!("Load an ONNX model first."),
            Self::ModelLoading => tr!("The model is still loading."),
            Self::ModelFailed(error) => tr!("The model could not be loaded: {}", error),
            Self::EngineNotReady => tr!("Preparing the inference engine…"),
            Self::EngineFailed(error) => error.to_string(),
            // Translators: eovae is working out which slide file this is (its
            // content fingerprint), to find its stored results; it does not
            // identify a patient.
            Self::SlideResolving => tr!("Identifying the slide…"),
            Self::NoLevels => tr!("The slide reports no pyramid levels."),
            Self::UnknownFile => tr!("The slide is not open."),
        }
    }
}

/// Everything about analysing one slide with the current settings.
#[derive(Clone, Debug)]
pub struct RunContext {
    pub file_id: FileId,
    pub key: ResultKey,
    pub slide: SlideIdentity,
    pub spec: Arc<RunSpec>,
    pub grid: TileGrid,
    pub choice: LevelChoice,
    pub contract: Contract,
    pub model: Arc<ModelInfo>,
}

pub fn run_context(state: &State, file_id: FileId) -> Result<RunContext, NoRun> {
    let file = state.files.get(&file_id).ok_or(NoRun::UnknownFile)?;
    let model = match &state.model {
        ModelState::Ready { info } => Arc::clone(info),
        ModelState::Loading { .. } => return Err(NoRun::ModelLoading),
        ModelState::Failed { error, .. } => return Err(NoRun::ModelFailed(error.clone())),
        ModelState::None => return Err(NoRun::NoModel),
    };
    let request = state.engine_request().ok_or(NoRun::NoModel)?;
    let signature = results_signature(&request);
    let info = match &state.engine.info {
        Some((built, info)) if *built == signature => info,
        _ => {
            return Err(match &state.engine.error {
                Some((failed, error)) if *failed == request.signature() => {
                    NoRun::EngineFailed(error.clone())
                }
                _ => NoRun::EngineNotReady,
            });
        }
    };
    let identity = file.identity.clone().ok_or(NoRun::SlideResolving)?;
    let contract = state.config.contract_for(&model);
    let choice = choose_level(&file.geometry, state.config.resolution, contract.mpp)
        .ok_or(NoRun::NoLevels)?;
    let grid = TileGrid::for_choice(&file.geometry, &choice, model.tile_size);
    let output = state.config.output_for(&model);
    let spec = RunSpec::new(&model, &output, &contract, &grid, choice.mpp, info);
    let key = ResultKey::new(&identity.key, &spec.key());
    Ok(RunContext {
        file_id,
        key,
        slide: identity,
        spec: Arc::new(spec),
        grid,
        choice,
        contract,
        model,
    })
}

// ---------------------------------------------------------------------------
// Dirty flags for the presenter
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
pub struct DirtyFlags {
    pub sidebar: bool,
    pub hud: bool,
    pub status: bool,
    pub all_overlays: bool,
    pub overlays: BTreeSet<FileId>,
    /// The language changed: texts eov keeps are handed over again.
    pub language: bool,
}

impl DirtyFlags {
    pub fn any(&self) -> bool {
        self.sidebar
            || self.hud
            || self.status
            || self.all_overlays
            || !self.overlays.is_empty()
            || self.language
    }
}

#[derive(Default)]
pub struct Dirty {
    flags: Mutex<DirtyFlags>,
    wake: Condvar,
}

impl Dirty {
    pub fn mark(&self, f: impl FnOnce(&mut DirtyFlags)) {
        f(&mut lock(&self.flags));
        self.wake.notify_all();
    }

    pub fn everything(&self) {
        self.mark(|d| {
            d.sidebar = true;
            d.hud = true;
            d.status = true;
            d.all_overlays = true;
        });
    }

    /// Waits up to `timeout` for work and takes it.
    pub fn wait(&self, timeout: Duration) -> DirtyFlags {
        let mut flags = lock(&self.flags);
        if !flags.any() {
            flags = self
                .wake
                .wait_timeout(flags, timeout)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
        std::mem::take(&mut *flags)
    }

    pub fn wake(&self) {
        self.wake.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Engine cell
// ---------------------------------------------------------------------------

#[derive(Default)]
struct EngineSlot {
    signature: Option<String>,
    engine: Option<Box<dyn InferenceEngine>>,
    busy: bool,
    failed: Option<(String, Message)>,
}

/// Holds the one inference engine; built on worker threads, reused while
/// the settings are unchanged (VAE-11, VAE-13).
#[derive(Default)]
pub struct EngineCell {
    slot: Mutex<EngineSlot>,
    changed: Condvar,
}

impl EngineCell {
    /// Forgets the engine (the model changed). An engine in use is dropped
    /// when it is returned.
    pub fn discard(&self) {
        let mut slot = lock(&self.slot);
        slot.engine = None;
        slot.signature = None;
        slot.failed = None;
        drop(slot);
        self.changed.notify_all();
    }

    pub fn release(&self, signature: &str, engine: Box<dyn InferenceEngine>) {
        let mut slot = lock(&self.slot);
        slot.busy = false;
        if slot.signature.as_deref() == Some(signature) {
            slot.engine = Some(engine);
        }
        drop(slot);
        self.changed.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

pub struct Core<B: Backend> {
    pub host: Host,
    pub backend: B,
    pub plugin_root: PathBuf,
    /// Started for `eov batch` (no window).
    pub headless: bool,
    /// The headless batch run in progress.
    pub batch: crate::batch::BatchRunner,
    pub state: Mutex<State>,
    pub results: Mutex<ResultStore>,
    pub cache: ResultCache,
    pub engine: EngineCell,
    pub dirty: Dirty,
    pub stopping: Arc<AtomicBool>,
    job_thread: Mutex<Option<JoinHandle<()>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    pub presenter: Mutex<Option<JoinHandle<()>>>,
    next_id: AtomicU64,
}

impl<B: Backend> Core<B> {
    pub fn create(host: Host, init: PluginInit) -> Result<Arc<Self>, String> {
        let (config, unreadable) = Config::from_settings(&init.settings);
        let plugin_root = PathBuf::from(&init.plugin_root);
        let cache = ResultCache::open(&PathBuf::from(&init.cache_dir), config.cache_limit_bytes());
        // Without a window there is nothing to show.
        if !init.headless {
            host.add_toolbar_button(ToolbarButtonSpec {
                button_id: TOOLBAR_BUTTON.into(),
                tooltip: toolbar_tooltip(),
                icon_svg: TOOL_ICON.into(),
                ..Default::default()
            })
            .map_err(String::from)?;
            host.add_hud_button(HudButtonSpec {
                button_id: HUD_GRID.into(),
                tooltip: grid_tooltip(),
                icon_svg: GRID_ICON.into(),
                ..Default::default()
            })
            .map_err(String::from)?;
            host.add_hud_button(HudButtonSpec {
                button_id: HUD_VIEW.into(),
                tooltip: view_tooltip(),
                icon_svg: VIEW_ICON.into(),
                menu: view_menu(Mode::Original),
                ..Default::default()
            })
            .map_err(String::from)?;
        }
        // `eov batch run eovae:analyze` (hosts without batch runs refuse it).
        let _ = host.register_batch_command(crate::batch::command_spec(&config));
        if let Some(error) = unreadable {
            warn_user(&host, || {
                // Translators: eovae is this plugin's name; {} says why, in English
                tr!(
                    "eovae settings were unreadable and were reset ({})",
                    i18n::isolate(&error)
                )
            });
        }
        let results = ResultStore::new(config.memory_limit_bytes());
        let core = Arc::new(Self {
            backend: B::new(&plugin_root),
            plugin_root,
            headless: init.headless,
            batch: crate::batch::BatchRunner::default(),
            state: Mutex::new(State::new(config)),
            results: Mutex::new(results),
            cache,
            engine: EngineCell::default(),
            dirty: Dirty::default(),
            stopping: Arc::new(AtomicBool::new(false)),
            job_thread: Mutex::new(None),
            workers: Mutex::new(Vec::new()),
            presenter: Mutex::new(None),
            next_id: AtomicU64::new(1),
            host,
        });
        if !core.headless {
            let presenter = Arc::clone(&core);
            let handle = std::thread::Builder::new()
                .name("eovae-presenter".into())
                .spawn(move || crate::presenter::run(presenter))
                .map_err(|e| e.to_string())?;
            *lock(&core.presenter) = Some(handle);
        }
        Ok(core)
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Runs `f` on a tracked worker thread.
    pub fn spawn(self: &Arc<Self>, name: &str, f: impl FnOnce(Arc<Self>) + Send + 'static) {
        let core = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name(format!("eovae-{name}"))
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| f(Arc::clone(&core))));
                if let Err(payload) = result {
                    // English in the log, the user's language on screen
                    // (see `warn_user`).
                    core.host.info("an eovae worker thread failed unexpectedly");
                    // Translators: the title of an error notification; eovae is
                    // this plugin's name
                    let message = tr!("An eovae worker thread failed unexpectedly");
                    match panic_text(&*payload) {
                        Some(detail) => core.host.notify_error(message, detail),
                        None => core.host.notify(NotifyLevel::Error, message),
                    }
                }
            });
        if let Ok(handle) = spawned {
            let mut workers = lock(&self.workers);
            workers.retain(|w| !w.is_finished());
            workers.push(handle);
        }
    }

    pub fn file_ids(&self) -> Vec<FileId> {
        lock(&self.state).files.keys().copied().collect()
    }

    // -- events ----------------------------------------------------------------

    pub fn on_event(self: &Arc<Self>, event: HostEvent) -> Result<(), String> {
        match event {
            HostEvent::Ready { restore_sidebar } => self.on_ready(restore_sidebar),
            HostEvent::FileOpened { file } => self.register_file(file),
            HostEvent::FileClosed { file_id } => self.on_file_closed(file_id),
            HostEvent::ActiveChanged { pane, file_id } => self.on_active_changed(pane, file_id),
            HostEvent::ViewChanged { view } => self.on_view_changed(view),
            HostEvent::ToolbarButtonClicked { button_id } if button_id == TOOLBAR_BUTTON => {
                self.toggle_sidebar()
            }
            HostEvent::HudButtonClicked {
                button_id,
                item_id,
                pane,
                view,
            } => self.on_hud(&button_id, item_id.as_deref(), pane, view),
            HostEvent::DialogResult { request_id, paths } => self.on_dialog(&request_id, paths),
            HostEvent::ConfirmResult {
                request_id,
                confirmed,
            } => self.on_confirm(&request_id, confirmed),
            HostEvent::ModelFetched {
                request_id,
                path,
                error,
                cancelled,
                ..
            } => self.on_model_fetched(&request_id, path, error, cancelled),
            HostEvent::JobCancelRequested { job_id } => {
                let state = lock(&self.state);
                if let Some(job) = state
                    .job
                    .as_ref()
                    .filter(|j| j.host_job.as_deref() == Some(&job_id))
                {
                    job.control.cancel();
                }
                Ok(())
            }
            HostEvent::BatchRun {
                run_id,
                command_id,
                file,
                output_dir,
                params,
            } => {
                crate::batch::start(
                    self,
                    crate::batch::BatchRequest {
                        run_id,
                        command_id,
                        file,
                        output_dir: PathBuf::from(output_dir),
                        params,
                    },
                );
                Ok(())
            }
            HostEvent::RoiChanged { file_id, roi } => {
                if let Some(file) = lock(&self.state).files.get_mut(&file_id) {
                    file.roi = roi;
                }
                self.dirty.mark(|d| d.sidebar = true);
                Ok(())
            }
            HostEvent::SurfaceClosed { surface_id } => {
                let mut state = lock(&self.state);
                if state.sidebar.as_ref().is_some_and(|s| s.id() == surface_id) {
                    state.sidebar = None;
                    drop(state);
                    let _ = self.host.set_toolbar_button_active(TOOLBAR_BUTTON, false);
                }
                Ok(())
            }
            HostEvent::LanguageChanged { .. } => {
                self.on_language_changed();
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The user chose another language: `tr!` answers in it already (the
    /// sidebar's `@tr` texts changed by themselves). eov shows what eovae
    /// handed over in the old language until it is handed over again: the
    /// tooltips and the overlay menu here, the batch command, and through
    /// the presenter every pane's menu, the status text, the layers' names
    /// and the sidebar's texts.
    fn on_language_changed(self: &Arc<Self>) {
        if !self.headless {
            let _ = self.host.update_toolbar_button(ButtonUpdate {
                button_id: TOOLBAR_BUTTON.into(),
                tooltip: Some(toolbar_tooltip()),
                ..Default::default()
            });
            let _ = self.host.update_hud_button(ButtonUpdate {
                button_id: HUD_GRID.into(),
                tooltip: Some(grid_tooltip()),
                ..Default::default()
            });
            let _ = self.host.update_hud_button(ButtonUpdate {
                button_id: HUD_VIEW.into(),
                tooltip: Some(view_tooltip()),
                menu: Some(view_menu(Mode::Original)),
                ..Default::default()
            });
        }
        let config = lock(&self.state).config.clone();
        let _ = self
            .host
            .register_batch_command(crate::batch::command_spec(&config));
        self.dirty.mark(|d| {
            d.language = true;
            d.sidebar = true;
            d.hud = true;
            d.status = true;
        });
    }

    fn on_ready(self: &Arc<Self>, restore_sidebar: bool) -> Result<(), String> {
        if let Ok(snapshot) = self.host.snapshot() {
            for file in snapshot.files.clone() {
                self.register_file(file)?;
            }
            let mut state = lock(&self.state);
            for pane in &snapshot.panes {
                state.panes.insert(
                    pane.pane,
                    PaneState {
                        file_id: pane.file_id,
                        view: pane.view,
                    },
                );
            }
            state.focused_pane = snapshot.focused_pane;
            state.active_file = snapshot.focused_pane.and_then(|pane| state.pane_file(pane));
        }
        if restore_sidebar {
            self.open_sidebar()?;
        }
        // Batch runs load the model they are asked for.
        let model_path = {
            let state = lock(&self.state);
            (matches!(state.model, ModelState::None) && !self.headless)
                .then(|| state.config.model_path.clone())
                .flatten()
        };
        if let Some(path) = model_path {
            self.load_model(PathBuf::from(path), false);
        }
        if !self.headless {
            self.refresh_download();
        }
        self.dirty.everything();
        Ok(())
    }

    /// Starts tracking a slide and resolves its identity off the UI thread.
    pub fn register_file(self: &Arc<Self>, file: FileInfo) -> Result<(), String> {
        let file_id = file.file_id;
        {
            let mut state = lock(&self.state);
            if state.files.contains_key(&file_id) {
                return Ok(());
            }
            state.files.insert(file_id, FileState::new(&file));
        }
        // Later changes arrive as `RoiChanged` events (on this thread, so
        // none can be missed in between).
        let roi = self.host.roi(file_id).ok().flatten();
        if let Some(entry) = lock(&self.state).files.get_mut(&file_id) {
            entry.roi = roi;
        }
        self.spawn("identity", move |core| {
            let identity = slide::resolve(&file);
            let mut state = lock(&core.state);
            if let Some(entry) = state.files.get_mut(&file_id) {
                entry.identity = Some(identity);
            }
            drop(state);
            core.dirty.mark(|d| {
                d.overlays.insert(file_id);
                d.sidebar = true;
            });
        });
        self.dirty.mark(|d| {
            d.overlays.insert(file_id);
            d.sidebar = true;
            d.hud = true;
        });
        Ok(())
    }

    /// Makes sure `file_id` is tracked (events can mention files opened
    /// before the plugin started).
    fn ensure_file(self: &Arc<Self>, file_id: FileId) {
        if lock(&self.state).files.contains_key(&file_id) {
            return;
        }
        if let Ok(info) = self.host.file_info(file_id) {
            let _ = self.register_file(info);
        }
    }

    fn on_file_closed(self: &Arc<Self>, file_id: FileId) -> Result<(), String> {
        let slide_key = {
            let mut state = lock(&self.state);
            if let Some(job) = state.job.as_ref().filter(|j| j.request.file_id == file_id) {
                job.control.cancel();
            }
            for pane in state.panes.values_mut() {
                if pane.file_id == Some(file_id) {
                    *pane = PaneState::default();
                }
            }
            if state.active_file == Some(file_id) {
                state.active_file = None;
            }
            if state.hover.is_some_and(|(f, _)| f == file_id) {
                state.hover = None;
            }
            let key = state
                .files
                .remove(&file_id)
                .and_then(|file| file.identity)
                .map(|identity| identity.key);
            // The same slide may be open twice; keep what the other uses.
            key.filter(|key| {
                !state
                    .files
                    .values()
                    .any(|f| f.identity.as_ref().is_some_and(|i| i.key == *key))
            })
        };
        // Memory goes; the disk cache keeps the results.
        if let Some(key) = slide_key {
            lock(&self.results).drop_slide(&key);
        }
        self.dirty.everything();
        Ok(())
    }

    fn on_active_changed(
        self: &Arc<Self>,
        pane: Option<PaneId>,
        file_id: Option<FileId>,
    ) -> Result<(), String> {
        let roi = file_id.map(|file_id| {
            self.ensure_file(file_id);
            self.refresh_identity(file_id);
            // `RoiChanged` keeps it current; hosts without the event are
            // caught up when the focus changes.
            (file_id, self.host.roi(file_id))
        });
        {
            let mut state = lock(&self.state);
            if let Some((file_id, Ok(roi))) = roi
                && let Some(file) = state.files.get_mut(&file_id)
            {
                file.roi = roi;
            }
            state.focused_pane = pane;
            state.active_file = file_id;
            if let Some(pane) = pane {
                let entry = state.panes.entry(pane).or_default();
                if entry.file_id != file_id {
                    entry.view = None;
                }
                entry.file_id = file_id;
            }
        }
        // A pane that switched slides drops what it showed of the old one.
        self.dirty.mark(|d| {
            d.sidebar = true;
            d.hud = true;
            d.all_overlays = true;
        });
        Ok(())
    }

    /// Forgets panes the host no longer has.
    pub fn retain_panes(&self, live: &BTreeSet<PaneId>) {
        let mut state = lock(&self.state);
        state.panes.retain(|pane, _| live.contains(pane));
        for file in state.files.values_mut() {
            file.panes.retain(|pane, _| live.contains(pane));
        }
    }

    /// The host cannot show layers per pane: every pane of a slide shows
    /// the overlay chosen last.
    pub fn without_pane_layers(&self) {
        let changed = std::mem::replace(&mut lock(&self.state).pane_layers, false);
        if changed {
            warn_user(&self.host, || {
                tr!(
                    "This eov cannot show layers in some panes only: every pane of a slide shows the same VAE overlay."
                )
            });
            self.dirty.everything();
        }
    }

    /// Adopts a slide identity the host provides after the slide was
    /// registered under eovae's fallback fingerprint: in-memory results move
    /// to the new key (the disk cache migrates on its next lookup).
    fn refresh_identity(self: &Arc<Self>, file_id: FileId) {
        // Only slides keyed by eovae's own fallback can gain an identity.
        let pending = lock(&self.state)
            .files
            .get(&file_id)
            .and_then(|file| file.identity.as_ref())
            .is_some_and(|identity| identity.keys.identity.is_none());
        if !pending {
            return;
        }
        let Ok(info) = self.host.file_info(file_id) else {
            return;
        };
        let Some(identity) = info.identity.clone().filter(|id| !id.trim().is_empty()) else {
            return;
        };
        let renamed = {
            let mut state = lock(&self.state);
            let Some(file) = state.files.get_mut(&file_id) else {
                return;
            };
            let Some(current) = file.identity.as_mut() else {
                return;
            };
            if current.key == identity {
                return;
            }
            let old = std::mem::replace(&mut current.key, identity.clone());
            current.keys.identity = Some(identity.clone());
            current.persistent = true;
            file.info = info;
            (old, identity)
        };
        lock(&self.results).rename_slide(&renamed.0, &renamed.1);
        self.dirty.mark(|d| {
            d.overlays.insert(file_id);
            d.sidebar = true;
        });
    }

    fn on_view_changed(self: &Arc<Self>, view: ViewInfo) -> Result<(), String> {
        self.ensure_file(view.file_id);
        self.refresh_identity(view.file_id);
        {
            let mut state = lock(&self.state);
            state.panes.insert(
                view.pane,
                PaneState {
                    file_id: Some(view.file_id),
                    view: Some(view),
                },
            );
        }
        self.dirty.mark(|d| {
            d.overlays.insert(view.file_id);
            d.hud = true;
            d.sidebar = true;
        });
        self.maybe_auto(view);
        Ok(())
    }

    fn on_hud(
        self: &Arc<Self>,
        button_id: &str,
        item_id: Option<&str>,
        pane: PaneId,
        view: Option<ViewInfo>,
    ) -> Result<(), String> {
        let file_id = match view.map(|v| v.file_id) {
            Some(file_id) => Some(file_id),
            None => lock(&self.state).pane_file(pane),
        };
        let Some(file_id) = file_id else {
            return Ok(());
        };
        self.ensure_file(file_id);
        if let Some(view) = view {
            lock(&self.state).panes.insert(
                pane,
                PaneState {
                    file_id: Some(file_id),
                    view: Some(view),
                },
            );
        }
        match (button_id, item_id) {
            (HUD_GRID, _) => {
                let mut state = lock(&self.state);
                let per_pane = state.pane_layers;
                if let Some(file) = state.files.get_mut(&file_id) {
                    file.update_overlay(Some(pane), per_pane, |o| o.grid = !o.grid);
                }
            }
            (HUD_VIEW, Some(item)) => {
                let Some(mode) = Mode::from_id(item) else {
                    return Ok(());
                };
                self.set_mode(file_id, Some(pane), mode, view)?;
            }
            _ => return Ok(()),
        }
        self.dirty.mark(|d| {
            d.overlays.insert(file_id);
            d.hud = true;
            d.sidebar = true;
        });
        Ok(())
    }

    /// Changes what a pane's overlay of a slide shows.
    pub fn set_mode(
        self: &Arc<Self>,
        file_id: FileId,
        pane: Option<PaneId>,
        mode: Mode,
        view: Option<ViewInfo>,
    ) -> Result<(), String> {
        let needs_model = {
            let mut state = lock(&self.state);
            let per_pane = state.pane_layers;
            if let Some(file) = state.files.get_mut(&file_id) {
                file.update_overlay(pane, per_pane, |o| o.mode = mode);
            }
            mode != Mode::Original
                && matches!(state.model, ModelState::None | ModelState::Failed { .. })
        };
        if needs_model {
            self.host.notify(
                NotifyLevel::Info,
                tr!("Load an ONNX model in the VAE sidebar to analyse this slide."),
            );
            if lock(&self.state).sidebar.is_none() {
                self.open_sidebar()?;
            }
        }
        self.dirty.mark(|d| {
            d.overlays.insert(file_id);
            d.hud = true;
            d.sidebar = true;
        });
        let view = view.or_else(|| {
            let state = lock(&self.state);
            pane.or(state.focused_pane)
                .and_then(|pane| state.panes.get(&pane))
                .and_then(|pane| pane.view)
                .filter(|view| view.file_id == file_id)
        });
        if let Some(view) = view {
            self.maybe_auto(view);
        }
        Ok(())
    }

    // -- sidebar ----------------------------------------------------------------

    pub fn toggle_sidebar(self: &Arc<Self>) -> Result<(), String> {
        let open = lock(&self.state).sidebar.clone();
        match open {
            Some(surface) if surface.is_open() => {
                lock(&self.state).sidebar = None;
                surface.close().map_err(String::from)?;
                self.host
                    .set_toolbar_button_active(TOOLBAR_BUTTON, false)
                    .map_err(String::from)
            }
            _ => self.open_sidebar(),
        }
    }

    pub fn open_sidebar(self: &Arc<Self>) -> Result<(), String> {
        let surface = self
            .host
            .open_surface(
                SurfaceSpec::new(SurfaceKind::Sidebar, SIDEBAR_UI, SIDEBAR_COMPONENT)
                    .with_title(tr!("VAE analysis"))
                    .with_size(360.0, None),
            )
            .map_err(String::from)?;
        lock(&self.state).sidebar = Some(surface);
        self.host
            .set_toolbar_button_active(TOOLBAR_BUTTON, true)
            .map_err(String::from)?;
        self.dirty.mark(|d| d.sidebar = true);
        Ok(())
    }

    // -- dialogs ------------------------------------------------------------------

    fn on_dialog(self: &Arc<Self>, request_id: &str, paths: Vec<String>) -> Result<(), String> {
        let (model, runtime, export) = {
            let mut state = lock(&self.state);
            let model = (state.model_dialog.as_deref() == Some(request_id))
                .then(|| state.model_dialog.take())
                .flatten();
            let runtime = (state.runtime_dialog.as_deref() == Some(request_id))
                .then(|| state.runtime_dialog.take())
                .flatten();
            let export = state
                .export_dialog
                .as_ref()
                .filter(|(id, _)| id == request_id)
                .map(|(_, file)| *file);
            if export.is_some() {
                state.export_dialog = None;
            }
            (model.is_some(), runtime.is_some(), export)
        };
        let Some(path) = paths.into_iter().next() else {
            return Ok(());
        };
        if model {
            self.load_model(PathBuf::from(path), true);
        } else if runtime {
            let mut state = lock(&self.state);
            state.config.runtime_path = Some(path);
            state.touch_config();
            drop(state);
            self.host.notify(
                NotifyLevel::Info,
                tr!(
                    "The ONNX Runtime library is used from the next start of eov if a runtime is already loaded."
                ),
            );
            self.prebuild_engine();
        } else if let Some(file_id) = export {
            self.spawn("export", move |core| {
                core.export_csv(file_id, PathBuf::from(path))
            });
        }
        self.dirty.mark(|d| d.sidebar = true);
        Ok(())
    }

    fn on_confirm(self: &Arc<Self>, request_id: &str, confirmed: bool) -> Result<(), String> {
        let ours = {
            let mut state = lock(&self.state);
            let ours = state.clear_confirm.as_deref() == Some(request_id);
            if ours {
                state.clear_confirm = None;
            }
            ours
        };
        if ours && confirmed {
            self.spawn("clear", |core| {
                match core.cache.clear() {
                    Ok(()) => core.host.notify(
                        NotifyLevel::Success,
                        tr!("The VAE result cache was cleared."),
                    ),
                    Err(error) => core
                        .host
                        .notify_error(tr!("Could not clear the VAE result cache"), error),
                }
                core.dirty.mark(|d| d.sidebar = true);
            });
        }
        Ok(())
    }

    // -- model ----------------------------------------------------------------------

    /// Inspects and loads a model on a worker thread; `persist` remembers it
    /// as the default model.
    pub fn load_model(self: &Arc<Self>, path: PathBuf, persist: bool) {
        let (generation, cancel) = {
            let mut state = lock(&self.state);
            state.model_cancel.store(true, Ordering::SeqCst);
            state.model_generation += 1;
            state.model_cancel = Arc::new(AtomicBool::new(false));
            state.model = ModelState::Loading { path: path.clone() };
            state.engine = EngineStatus::default();
            if let Some(job) = &state.job {
                job.control.cancel();
            }
            (state.model_generation, Arc::clone(&state.model_cancel))
        };
        self.engine.discard();
        self.dirty.everything();
        self.spawn("model", move |core| {
            let result = catch_unwind(AssertUnwindSafe(|| model::inspect(&path, &cancel)))
                .unwrap_or_else(|_| Err(Message::new(|| tr!("the model file could not be read"))));
            let mut state = lock(&core.state);
            if state.model_generation != generation {
                return;
            }
            let failure = match result {
                Ok(info) => {
                    if persist {
                        state.config.model_path = Some(path.to_string_lossy().into_owned());
                        state.touch_config();
                    }
                    state.model = ModelState::Ready {
                        info: Arc::new(info),
                    };
                    None
                }
                Err(error) => {
                    state.model = ModelState::Failed {
                        path: path.clone(),
                        error: error.clone(),
                    };
                    Some(error)
                }
            };
            drop(state);
            core.dirty.everything();
            match failure {
                Some(error) => {
                    core.host
                        .notify_error(tr!("The VAE model could not be loaded"), error.shown);
                }
                None => core.prebuild(),
            }
        });
    }

    // -- model downloads -------------------------------------------------------

    /// Looks the download target up in eov's model store.
    pub fn refresh_download(&self) {
        let args = download_args(&lock(&self.state).config);
        let found = self.host.resolve_model(&args.sha256);
        let mut state = lock(&self.state);
        match found {
            Ok(record) => {
                state.model_store = true;
                state.download_stored = record
                    .filter(|record| {
                        args.companions.iter().all(|c| {
                            record.has_companion(&c.sha256, c.file_name.as_deref().unwrap_or(""))
                        })
                    })
                    .map(|record| record.path);
            }
            Err(error) => {
                state.model_store = !error.is_unsupported();
                state.download_stored = None;
            }
        }
        drop(state);
        self.dirty.mark(|d| d.sidebar = true);
    }

    /// "Download": asks eov for the model. eov answers at once when it has
    /// it; otherwise it asks the user and downloads it, and the answer
    /// arrives as `ModelFetched`.
    pub fn download_model(self: &Arc<Self>) {
        let args = {
            let state = lock(&self.state);
            if state.model_fetch.is_some() {
                return;
            }
            download_args(&state.config)
        };
        let asked = self.host.fetch_model(args);
        let mut state = lock(&self.state);
        match asked {
            Ok(request) => {
                state.model_fetch = Some(request);
                state.download_note = None;
            }
            Err(error) if error.is_unsupported() => {
                state.model_store = false;
                // Translators: "Load ONNX…" is the sidebar's button of that name
                state.download_note = Some(tr!(
                    "This eov cannot download models: load the model file with Load ONNX…"
                ));
            }
            Err(error) => state.download_note = Some(error.to_string()),
        }
        drop(state);
        self.dirty.mark(|d| d.sidebar = true);
    }

    fn on_model_fetched(
        self: &Arc<Self>,
        request_id: &str,
        path: Option<String>,
        error: Option<String>,
        cancelled: bool,
    ) -> Result<(), String> {
        {
            let mut state = lock(&self.state);
            if state.model_fetch.as_deref() != Some(request_id) {
                return Ok(());
            }
            state.model_fetch = None;
            // eov has told the user why; the sidebar keeps it in view.
            state.download_note = error.filter(|_| path.is_none() && !cancelled);
            if path.is_some() {
                state.download_stored = path.clone();
            }
        }
        if let Some(path) = path {
            // The downloaded model becomes the default, here and in batch
            // runs.
            self.load_model(PathBuf::from(path), true);
        }
        self.dirty.mark(|d| d.sidebar = true);
        Ok(())
    }

    pub fn clear_model(self: &Arc<Self>) {
        {
            let mut state = lock(&self.state);
            state.model_cancel.store(true, Ordering::SeqCst);
            state.model_generation += 1;
            state.model = ModelState::None;
            state.engine = EngineStatus::default();
            state.config.model_path = None;
            state.touch_config();
            if let Some(job) = &state.job {
                job.control.cancel();
            }
        }
        self.engine.discard();
        self.dirty.everything();
    }

    /// Builds the engine for the current settings in the background, so the
    /// first analysis does not wait (and the run key becomes known).
    pub fn prebuild_engine(self: &Arc<Self>) {
        self.spawn("engine", |core| core.prebuild());
    }

    fn prebuild(self: &Arc<Self>) {
        let (request, stop) = {
            let state = lock(&self.state);
            // A newer or cleared model cancels the build.
            (state.engine_request(), Arc::clone(&state.model_cancel))
        };
        let Some(request) = request else {
            return;
        };
        if let Ok(engine) = self.acquire_engine(&request, &stop, true) {
            self.engine.release(&request.signature(), engine);
        }
    }

    fn set_engine_status(&self, message: Option<String>) {
        lock(&self.state).engine.building = message;
        self.dirty.mark(|d| {
            d.sidebar = true;
            d.status = true;
        });
    }

    /// Takes the engine for `request`, building it when needed. Waits while
    /// another thread builds or uses it.
    pub fn acquire_engine(
        self: &Arc<Self>,
        request: &EngineRequest,
        stop: &Arc<AtomicBool>,
        retry_failed: bool,
    ) -> Result<Box<dyn InferenceEngine>, Message> {
        let signature = request.signature();
        let mut slot = lock(&self.engine.slot);
        loop {
            if stop.load(Ordering::SeqCst) || self.stopping.load(Ordering::SeqCst) {
                return Err(cancelled());
            }
            if !slot.busy {
                if slot.signature.as_deref() == Some(signature.as_str())
                    && let Some(engine) = slot.engine.take()
                {
                    slot.busy = true;
                    return Ok(engine);
                }
                if let Some((failed, error)) = &slot.failed
                    && *failed == signature
                    && !retry_failed
                {
                    return Err(error.clone());
                }
                // Build. The old engine goes first (GPU memory).
                slot.busy = true;
                slot.engine = None;
                slot.signature = None;
                drop(slot);
                let flags = [Arc::clone(stop), Arc::clone(&self.stopping)];
                let built = catch_unwind(AssertUnwindSafe(|| {
                    self.backend.create(
                        request,
                        &|message| self.set_engine_status(Some(message.to_string())),
                        &flags,
                    )
                }))
                .unwrap_or_else(|_| {
                    Err(Message::new(|| {
                        tr!("the inference engine failed unexpectedly")
                    }))
                });
                let stopped = stop.load(Ordering::SeqCst) || self.stopping.load(Ordering::SeqCst);
                let mut slot = lock(&self.engine.slot);
                let result = match built {
                    Ok(engine) => {
                        slot.signature = Some(signature.clone());
                        slot.failed = None;
                        Ok(engine)
                    }
                    Err(_) if stopped => {
                        // Abandoned, not broken: build again when asked.
                        slot.busy = false;
                        drop(slot);
                        self.engine.changed.notify_all();
                        lock(&self.state).engine.building = None;
                        self.dirty.mark(|d| d.sidebar = true);
                        return Err(cancelled());
                    }
                    Err(error) => {
                        slot.busy = false;
                        slot.failed = Some((signature.clone(), error.clone()));
                        Err(error)
                    }
                };
                drop(slot);
                self.engine.changed.notify_all();
                {
                    let mut state = lock(&self.state);
                    state.engine.building = None;
                    match &result {
                        Ok(engine) => {
                            state.engine.info =
                                Some((results_signature(request), engine.info().clone()));
                            state.engine.error = None;
                        }
                        Err(error) => state.engine.error = Some((signature.clone(), error.clone())),
                    }
                }
                if let Ok(engine) = &result
                    && let Some(reason) = &engine.info().fallback
                {
                    self.host.notify(
                        NotifyLevel::Warning,
                        tr!(
                            "VAE inference runs on the CPU: {}",
                            crate::engine::reason_text(reason)
                        ),
                    );
                }
                if let Err(error) = &result {
                    self.host
                        .notify_error(tr!("The VAE model cannot run"), error.shown.clone());
                }
                self.dirty.everything();
                return result;
            }
            slot = self
                .engine
                .changed
                .wait_timeout(slot, Duration::from_millis(100))
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }

    // -- jobs ---------------------------------------------------------------------------

    /// Analyses visible tiles while an overlay is shown, unless an explicit
    /// job is running or the view is too far zoomed out.
    pub fn maybe_auto(self: &Arc<Self>, view: ViewInfo) {
        let start = {
            let mut state = lock(&self.state);
            let mode = state.overlay(view.file_id, Some(view.pane)).mode;
            if !state.config.auto_analyze
                || mode == Mode::Original
                || !matches!(state.model, ModelState::Ready { .. })
                || state
                    .job
                    .as_ref()
                    .is_some_and(|j| j.request.kind.explicit())
            {
                false
            } else {
                match run_context(&state, view.file_id) {
                    Ok(ctx) => {
                        let range = ctx.grid.range_in(&view.visible);
                        let tile_px = ctx.grid.screen_size(view.zoom);
                        if range.len() > AUTO_MAX_TILES || tile_px < AUTO_MIN_TILE_PX {
                            state.auto_hint = Some(range.len());
                            false
                        } else {
                            state.auto_hint = None;
                            true
                        }
                    }
                    // Without an engine yet, the job builds it.
                    Err(NoRun::EngineNotReady | NoRun::SlideResolving) => true,
                    Err(_) => false,
                }
            }
        };
        self.dirty.mark(|d| d.sidebar = true);
        if start {
            self.start_job(JobRequest {
                kind: JobKind::Auto,
                file_id: view.file_id,
                pane: Some(view.pane),
                view: Some(view),
            });
        }
    }

    /// Starts a job, cancelling the running one; the new job's thread waits
    /// for the old one to finish first.
    pub fn start_job(self: &Arc<Self>, request: JobRequest) {
        if self.stopping.load(Ordering::SeqCst) {
            return;
        }
        let id = self.next_id();
        let control = JobControl::new();
        let mut thread = lock(&self.job_thread);
        {
            let mut state = lock(&self.state);
            if let Some(previous) = state.job.take() {
                previous.control.cancel();
            }
            state.job = Some(RunningJob {
                id,
                request: request.clone(),
                control: Arc::clone(&control),
                host_job: None,
                progress: Progress::default(),
                phase: tr!("Starting…"),
                started: Instant::now(),
            });
        }
        let previous = thread.take();
        let core = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("eovae-job".to_string())
            .spawn(move || {
                if let Some(previous) = previous {
                    let _ = previous.join();
                }
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    core.execute_job(id, &request, &control)
                }));
                core.finish_job(id, &request, &control, outcome);
            });
        *thread = spawned.ok();
        drop(thread);
        self.dirty.mark(|d| {
            d.sidebar = true;
            d.status = true;
        });
    }

    pub fn cancel_job(&self) {
        if let Some(job) = &lock(&self.state).job {
            job.control.cancel();
        }
        self.dirty.mark(|d| d.sidebar = true);
    }

    fn set_phase(&self, id: u64, phase: &str) {
        let mut state = lock(&self.state);
        if let Some(job) = state.job.as_mut().filter(|j| j.id == id) {
            job.phase = phase.to_string();
        }
        drop(state);
        self.dirty.mark(|d| {
            d.sidebar = true;
            d.status = true;
        });
    }

    fn execute_job(
        self: &Arc<Self>,
        id: u64,
        request: &JobRequest,
        control: &Arc<JobControl>,
    ) -> Result<JobEnd, Message> {
        if control.is_cancelled() {
            return Ok(JobOutcome::Cancelled(Counts::default()).into());
        }
        let engine_request = lock(&self.state)
            .engine_request()
            .ok_or_else(|| NoRun::NoModel.message())?;
        self.set_phase(id, &tr!("Preparing the model…"));
        let mut engine = match self.acquire_engine(
            &engine_request,
            control.stop_flag(),
            request.kind.explicit(),
        ) {
            Ok(engine) => engine,
            Err(_) if control.is_cancelled() => {
                return Ok(JobOutcome::Cancelled(Counts::default()).into());
            }
            Err(error) => return Err(error),
        };
        let result = self.run_with_engine(id, request, control, engine.as_mut());
        self.engine.release(&engine_request.signature(), engine);
        result
    }

    fn run_with_engine(
        self: &Arc<Self>,
        id: u64,
        request: &JobRequest,
        control: &Arc<JobControl>,
        engine: &mut dyn InferenceEngine,
    ) -> Result<JobEnd, Message> {
        let waited = Instant::now();
        let (ctx, config, mode) = loop {
            let attempt = {
                let state = lock(&self.state);
                run_context(&state, request.file_id).map(|ctx| {
                    let mode = state.overlay(request.file_id, request.pane).mode;
                    (ctx, state.config.clone(), mode)
                })
            };
            match attempt {
                Ok(found) => break found,
                Err(NoRun::SlideResolving | NoRun::EngineNotReady)
                    if !control.stopping() && waited.elapsed() < Duration::from_secs(60) =>
                {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) if control.is_cancelled() => {
                    return Ok(JobOutcome::Cancelled(Counts::default()).into());
                }
                Err(reason) => return Err(reason.message()),
            }
        };
        let (tiles, need_payloads) = match request.kind {
            JobKind::WholeSlide => (ctx.grid.full_range().iter().collect::<Vec<_>>(), false),
            JobKind::Roi => {
                let roi = self
                    .host
                    .roi(request.file_id)
                    // eov's reason, in English.
                    .map_err(|e| Message::plain(e.message))?
                    .ok_or_else(|| {
                        Message::new(|| tr!("Draw a region of interest on the slide first."))
                    })?;
                let range = ctx.grid.range_in(&roi);
                let center = ctx
                    .grid
                    .tile_units(roi.x + roi.width / 2.0, roi.y + roi.height / 2.0);
                (range.center_out(center), mode.needs_pixels())
            }
            JobKind::Viewport | JobKind::Auto => {
                let view = request
                    .view
                    .or_else(|| {
                        let state = lock(&self.state);
                        request
                            .pane
                            .and_then(|pane| state.panes.get(&pane))
                            .and_then(|p| p.view)
                    })
                    .filter(|view| view.file_id == request.file_id)
                    .ok_or_else(|| Message::new(|| tr!("The slide is not shown in a pane.")))?;
                let range = ctx.grid.range_in(&view.visible);
                let center = ctx.grid.tile_units(view.center_x, view.center_y);
                (range.center_out(center), mode.needs_pixels())
            }
        };
        self.ensure_loaded(&ctx);
        let mut needs = lock(&self.results).needs(
            &ctx.key,
            tiles.iter().copied(),
            need_payloads,
            config.skip_background,
        );
        // Pixels kept on disk spare recomputing them.
        if config.keep_pixels && ctx.slide.persistent {
            let wanted: Vec<TileCoord> = needs
                .iter()
                .filter(|(_, need)| *need == Need::Payload)
                .map(|(coord, _)| *coord)
                .collect();
            if !wanted.is_empty() {
                let loaded = self
                    .cache
                    .load_payloads(&ctx.slide.keys, &ctx.spec.key(), wanted);
                if !loaded.is_empty() {
                    let mut results = lock(&self.results);
                    for (coord, payload) in loaded {
                        if let Some(state) = results.run(&ctx.key).and_then(|run| run.state(coord))
                        {
                            results.record(
                                &ctx.key,
                                ctx.grid,
                                coord,
                                state,
                                Some(Arc::new(payload)),
                            );
                        }
                    }
                    needs = results.needs(
                        &ctx.key,
                        tiles.iter().copied(),
                        need_payloads,
                        config.skip_background,
                    );
                }
                self.dirty.mark(|d| {
                    d.overlays.insert(request.file_id);
                });
            }
        }
        if needs.is_empty() {
            return Ok(JobOutcome::Completed(Counts::default()).into());
        }
        let host_job = if request.kind.explicit() {
            let title = match request.kind {
                JobKind::WholeSlide => tr!("VAE: whole slide"),
                JobKind::Roi => tr!("VAE: region of interest"),
                _ => tr!("VAE: viewport"),
            };
            self.host.start_job(title, true).ok()
        } else {
            None
        };
        {
            let mut state = lock(&self.state);
            if let Some(job) = state.job.as_mut().filter(|j| j.id == id) {
                job.host_job = host_job.as_ref().map(|j| j.id().to_string());
                job.phase = tr!("Analysing on {}", engine.info().text());
                job.progress.total = needs.len();
            }
        }
        let spec = JobSpec {
            file_id: request.file_id,
            grid: ctx.grid,
            tiles: needs,
            contract: ctx.contract.clone(),
            layout: ctx.model.layout,
            skip_background: config.skip_background,
            batch_size: config.batch_size,
            readers: if request.kind == JobKind::WholeSlide {
                4
            } else {
                2
            },
        };
        let sink = CoreSink {
            core: self,
            id,
            ctx: &ctx,
            keep_pixels: config.keep_pixels,
            host_job: host_job.as_ref(),
        };
        let pixels = HostPixels {
            host: self.host.clone(),
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_job(&spec, &pixels, engine, &sink, control)
        }))
        .unwrap_or_else(|_| JobOutcome::Failed {
            error: Message::new(|| tr!("internal error while analysing tiles")),
            counts: Counts::default(),
        });
        // eov tells the user that a job of its job panel failed, in their
        // language.
        let reported = host_job.is_some() && matches!(outcome, JobOutcome::Failed { .. });
        if let Some(job) = host_job {
            job.finish(match &outcome {
                JobOutcome::Failed { error, .. } => Some(error.shown.clone()),
                _ => None,
            });
        }
        Ok(JobEnd { outcome, reported })
    }

    /// Loads a run's stored results once.
    pub fn ensure_loaded(&self, ctx: &RunContext) {
        let loaded = lock(&self.results)
            .run(&ctx.key)
            .is_some_and(|run| run.loaded_from_disk);
        if loaded {
            return;
        }
        let stored = if ctx.slide.persistent {
            self.cache.load_run(&ctx.slide.keys, &ctx.spec.key())
        } else {
            Vec::new()
        };
        lock(&self.results).merge_loaded(&ctx.key, ctx.grid, stored);
    }

    fn finish_job(
        self: &Arc<Self>,
        id: u64,
        request: &JobRequest,
        control: &JobControl,
        outcome: std::thread::Result<Result<JobEnd, Message>>,
    ) {
        let JobEnd { outcome, reported } = match outcome {
            Ok(Ok(end)) => end,
            Ok(Err(error)) => JobOutcome::Failed {
                error,
                counts: Counts::default(),
            }
            .into(),
            Err(payload) => {
                let panic = panic_text(&*payload).unwrap_or_default();
                JobOutcome::Failed {
                    error: Message::new(|| tr!("internal error: {}", panic)),
                    counts: Counts::default(),
                }
                .into()
            }
        };
        let (elapsed, superseded) = {
            let state = lock(&self.state);
            let current = state.job.as_ref().filter(|j| j.id == id);
            (
                current.map(|j| j.started.elapsed()).unwrap_or_default(),
                current.is_none(),
            )
        };
        let summary = JobSummary {
            kind: request.kind,
            outcome: outcome.clone(),
            elapsed,
        };
        let failed = summary.failed();
        {
            let mut state = lock(&self.state);
            if state.job.as_ref().is_some_and(|j| j.id == id) {
                state.job = None;
            }
            if (request.kind.explicit() || failed) && !(superseded && control.is_cancelled()) {
                state.last_job = Some(summary);
            }
        }
        if let JobOutcome::Failed { error, .. } = &outcome {
            // The log gets the error in English (see `warn_user`). The user
            // is told once, in their language: by eov when the job was in
            // its job panel, else here (automatic jobs never are; explicit
            // ones that fail before they start analysing are not yet).
            self.host
                .info(format!("VAE analysis failed: {}", log_text(&error.english)));
            if !reported {
                self.host
                    .notify_error(tr!("VAE analysis failed"), error.shown.clone());
            }
        }
        self.dirty.everything();
        if request.kind.explicit() && !self.stopping.load(Ordering::SeqCst) {
            let view = {
                let state = lock(&self.state);
                state
                    .focused_pane
                    .and_then(|pane| state.panes.get(&pane))
                    .and_then(|pane| pane.view)
            };
            if let Some(view) = view {
                self.maybe_auto(view);
            }
        }
    }

    // -- export ------------------------------------------------------------------------

    /// Writes the active run's tile scores as CSV, plus a JSON manifest of
    /// how they were computed (§14.5). No slide path is written.
    fn export_csv(self: &Arc<Self>, file_id: FileId, path: PathBuf) {
        let found = run_context(&lock(&self.state), file_id);
        let ctx = match found {
            Ok(ctx) => ctx,
            Err(reason) => {
                self.host
                    .notify_error(tr!("Nothing to export"), reason.text());
                return;
            }
        };
        self.ensure_loaded(&ctx);
        let mut rows: Vec<(TileCoord, crate::results::TileState)> = lock(&self.results)
            .run(&ctx.key)
            .map(|run| run.iter().collect())
            .unwrap_or_default();
        rows.sort_by_key(|(coord, _)| coord.key());
        let mut csv = String::from("col,row,x,y,width,height,level,state,mae,max_error\n");
        for (coord, state) in &rows {
            let rect = ctx.grid.valid_footprint(*coord);
            let (label, mae, max) = match state {
                crate::results::TileState::Analysed { mae, max } => {
                    ("analysed", format!("{mae:.6}"), format!("{max:.6}"))
                }
                crate::results::TileState::Background => {
                    ("background", String::new(), String::new())
                }
                crate::results::TileState::Failed => ("unreadable", String::new(), String::new()),
            };
            csv.push_str(&format!(
                "{},{},{:.0},{:.0},{:.0},{:.0},{},{label},{mae},{max}\n",
                coord.col, coord.row, rect.x, rect.y, rect.width, rect.height, ctx.grid.level
            ));
        }
        let manifest = serde_json::json!({
            "generator": format!("eovae {}", env!("CARGO_PKG_VERSION")),
            "run_key": ctx.spec.key(),
            "run": &*ctx.spec,
            "slide": {
                "identity": ctx.slide.keys.identity,
                "fingerprint": ctx.slide.keys.fingerprint,
                "width": ctx.grid.width0,
                "height": ctx.grid.height0,
            },
            "grid": {
                "tile": ctx.grid.tile,
                "cols": ctx.grid.cols,
                "rows": ctx.grid.rows,
                "level": ctx.grid.level,
                "downsample": ctx.grid.downsample,
                "level0_px_per_model_px": [ctx.grid.scale_x, ctx.grid.scale_y],
            },
            "coordinates": "level-0 pixels of the slide",
            "mae": "mean absolute error over in-slide pixels and RGB channels, 0-1",
        });
        let manifest_path = path.with_extension("manifest.json");
        let written = std::fs::write(&path, csv).and_then(|()| {
            std::fs::write(
                &manifest_path,
                serde_json::to_string_pretty(&manifest).unwrap_or_default(),
            )
        });
        match written {
            Ok(()) => self.host.notify(
                NotifyLevel::Success,
                tr!(
                    "Exported {n} tile to {file}" | "Exported {n} tiles to {file}" % rows.len(),
                    file = i18n::isolate(path.display())
                ),
            ),
            Err(error) => self
                .host
                .notify_error(tr!("The export failed"), error.to_string()),
        }
    }

    // -- shutdown ------------------------------------------------------------------------

    pub fn persist_config(&self) {
        let config = {
            let mut state = lock(&self.state);
            if state.config_changed.take().is_none() {
                return;
            }
            state.config.clone()
        };
        if let Err(error) = self.host.set_setting(crate::config::SETTINGS_KEY, &config) {
            warn_user(&self.host, || {
                // Translators: eovae is this plugin's name; {} is eov's reason, in English
                tr!(
                    "eovae settings could not be saved: {}",
                    i18n::isolate(&error)
                )
            });
        }
    }

    pub fn shutdown(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        {
            let state = lock(&self.state);
            state.model_cancel.store(true, Ordering::SeqCst);
            if let Some(job) = &state.job {
                job.control.cancel();
            }
        }
        self.batch.cancel();
        self.dirty.wake();
        self.engine.changed.notify_all();
        if let Some(handle) = lock(&self.job_thread).take() {
            let _ = handle.join();
        }
        if let Some(handle) = lock(&self.presenter).take() {
            let _ = handle.join();
        }
        let workers: Vec<JoinHandle<()>> = std::mem::take(&mut *lock(&self.workers));
        for worker in workers {
            let _ = worker.join();
        }
        self.persist_config();
        self.cache.shutdown();
    }
}

/// The HUD overlay menu with the current mode checked.
pub fn view_menu(current: Mode) -> Vec<MenuItem> {
    Mode::ALL
        .into_iter()
        .map(|mode| MenuItem {
            item_id: mode.id().to_string(),
            label: mode.label(),
            checked: Some(mode == current),
            ..Default::default()
        })
        .collect()
}

/// A duration for people (`12 s`, `3m 05s`, `1h 02m`).
fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match seconds {
        // Translators: a duration in seconds
        0..=59 => tr!("{} s", seconds),
        // Translators: a duration in minutes and seconds, e.g. "3m 05s"
        60..=3599 => tr!(
            "{minutes}m {seconds}s",
            minutes = seconds / 60,
            seconds = format!("{:02}", seconds % 60)
        ),
        // Translators: a duration in hours and minutes, e.g. "1h 02m"
        _ => tr!(
            "{hours}h {minutes}m",
            hours = seconds / 3600,
            minutes = format!("{:02}", seconds / 60 % 60)
        ),
    }
}

/// The time a job still needs, for the sidebar and the job's progress.
pub fn format_eta(progress: &Progress) -> String {
    progress
        .eta()
        .map(|eta| tr!("about {} left", format_duration(eta)))
        .unwrap_or_default()
}

/// The tiles a job analysed, found to be glass and could not read.
fn describe_counts(counts: &Counts) -> String {
    let (analysed, background, unreadable) = (counts.analysed, counts.background, counts.failed);
    match (background > 0, unreadable > 0) {
        (false, false) => tr!("{analysed} analysed", analysed),
        (true, false) => tr!(
            "{analysed} analysed, {background} background",
            analysed,
            background
        ),
        (false, true) => tr!(
            "{analysed} analysed, {unreadable} unreadable",
            analysed,
            unreadable
        ),
        (true, true) => tr!(
            "{analysed} analysed, {background} background, {unreadable} unreadable",
            analysed,
            background,
            unreadable
        ),
    }
}

impl JobSummary {
    pub fn failed(&self) -> bool {
        matches!(self.outcome, JobOutcome::Failed { .. })
    }

    /// The summary in the language of the interface. Each kind of job has
    /// whole sentences of its own, so that translations can make the verb
    /// agree with what finished, was cancelled or failed.
    pub fn text(&self) -> String {
        use JobKind::{Auto, Roi, Viewport, WholeSlide};
        let (tiles, total) = match &self.outcome {
            JobOutcome::Completed(counts)
            | JobOutcome::Cancelled(counts)
            | JobOutcome::Failed { counts, .. } => (counts, counts.total()),
        };
        let duration = format_duration(self.elapsed);
        let counts = describe_counts(tiles);
        match (&self.outcome, self.kind) {
            (JobOutcome::Completed(_), kind) if total == 0 => {
                let what = kind.subject();
                // Translators: {what}: Whole slide, Viewport, Region of interest or Automatic analysis
                tr!("{what}: everything was already analysed.", what)
            }
            // Translators: the analysis of the whole slide (the button Whole slide) ended; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Completed(_), WholeSlide) => tr!(
                "Whole slide finished in {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: the analysis of the tiles in view (the button Viewport) ended; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Completed(_), Viewport) => tr!(
                "Viewport finished in {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: the analysis of the region of interest ended; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Completed(_), Roi) => tr!(
                "Region of interest finished in {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Completed(_), Auto) => tr!(
                "Automatic analysis finished in {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: the analysis of the whole slide (the button Whole slide) was cancelled; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Cancelled(_), WholeSlide) => tr!(
                "Whole slide cancelled after {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: the analysis of the tiles in view (the button Viewport) was cancelled; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Cancelled(_), Viewport) => tr!(
                "Viewport cancelled after {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: the analysis of the region of interest was cancelled; {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Cancelled(_), Roi) => tr!(
                "Region of interest cancelled after {duration}: {counts}.",
                duration,
                counts
            ),
            // Translators: {duration}: e.g. "3m 05s"; {counts}: e.g. "120 analysed, 8 background"
            (JobOutcome::Cancelled(_), Auto) => tr!(
                "Automatic analysis cancelled after {duration}: {counts}.",
                duration,
                counts
            ),
            (JobOutcome::Failed { error, .. }, WholeSlide) if total > 0 => {
                // Translators: the analysis of the whole slide (the button Whole slide) failed; {counts}: e.g. "120 analysed, 8 background"; {error}: why
                tr!("Whole slide failed ({counts}): {error}", counts, error)
            }
            (JobOutcome::Failed { error, .. }, Viewport) if total > 0 => {
                // Translators: the analysis of the tiles in view (the button Viewport) failed; {counts}: e.g. "120 analysed, 8 background"; {error}: why
                tr!("Viewport failed ({counts}): {error}", counts, error)
            }
            (JobOutcome::Failed { error, .. }, Roi) if total > 0 => {
                // Translators: the analysis of the region of interest failed; {counts}: e.g. "120 analysed, 8 background"; {error}: why
                tr!(
                    "Region of interest failed ({counts}): {error}",
                    counts,
                    error
                )
            }
            (JobOutcome::Failed { error, .. }, Auto) if total > 0 => {
                // Translators: {counts}: e.g. "120 analysed, 8 background"; {error}: why
                tr!(
                    "Automatic analysis failed ({counts}): {error}",
                    counts,
                    error
                )
            }
            (JobOutcome::Failed { error, .. }, WholeSlide) => {
                // Translators: the analysis of the whole slide (the button Whole slide) failed; {error}: why
                tr!("Whole slide failed: {error}", error)
            }
            (JobOutcome::Failed { error, .. }, Viewport) => {
                // Translators: the analysis of the tiles in view (the button Viewport) failed; {error}: why
                tr!("Viewport failed: {error}", error)
            }
            (JobOutcome::Failed { error, .. }, Roi) => {
                // Translators: the analysis of the region of interest failed; {error}: why
                tr!("Region of interest failed: {error}", error)
            }
            (JobOutcome::Failed { error, .. }, Auto) => {
                // Translators: {error}: why
                tr!("Automatic analysis failed: {error}", error)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline plumbing
// ---------------------------------------------------------------------------

/// Slide pixels through the host (any thread, never blocks the UI).
pub(crate) struct HostPixels {
    pub host: Host,
}

impl PixelSource for HostPixels {
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, Message> {
        // eov's reason, in English.
        let image = self
            .host
            .read_region(*request)
            .map_err(|e| Message::plain(e.message))?;
        if image.width != request.width || image.height != request.height {
            return Err(Message::new(|| {
                // Translators: why a part of the slide could not be read: the
                // host (eov) gave an image of another size than asked for, in
                // pixels.
                tr!(
                    "the host returned {width}×{height} for a {asked_width}×{asked_height} region",
                    width = image.width,
                    height = image.height,
                    asked_width = request.width,
                    asked_height = request.height
                )
            }));
        }
        Ok(image.data)
    }
}

/// Where a job's tiles go: memory, the disk cache, the presenter.
struct CoreSink<'a, B: Backend> {
    core: &'a Arc<Core<B>>,
    id: u64,
    ctx: &'a RunContext,
    keep_pixels: bool,
    host_job: Option<&'a Job>,
}

impl<B: Backend> JobSink for CoreSink<'_, B> {
    fn deliver(&self, outcomes: Vec<TileOutcome>) {
        let persistent = self.ctx.slide.persistent;
        let mut writes = Vec::with_capacity(outcomes.len());
        {
            let mut results = lock(&self.core.results);
            for outcome in outcomes {
                let payload = outcome.payload.map(Arc::new);
                results.record(
                    &self.ctx.key,
                    self.ctx.grid,
                    outcome.coord,
                    outcome.state,
                    payload.clone(),
                );
                if persistent {
                    writes.push(DbTile {
                        coord: outcome.coord,
                        state: outcome.state,
                        embedding: outcome.embedding,
                        payload: if self.keep_pixels { payload } else { None },
                    });
                }
            }
        }
        if !writes.is_empty() {
            // Blocks while the cache thread is behind (backpressure).
            self.core.cache.write(RunWrite {
                slide: self.ctx.slide.keys.clone(),
                run_key: self.ctx.spec.key(),
                manifest: self.ctx.spec.manifest(),
                cols: self.ctx.grid.cols,
                rows: self.ctx.grid.rows,
                tiles: writes,
            });
        }
        self.core.dirty.mark(|d| {
            d.overlays.insert(self.ctx.file_id);
            d.sidebar = true;
        });
    }

    fn progress(&self, progress: &Progress) {
        {
            let mut state = lock(&self.core.state);
            if let Some(job) = state.job.as_mut().filter(|j| j.id == self.id) {
                job.progress = *progress;
            }
        }
        if let Some(job) = self.host_job {
            let done = progress.done;
            let message = match format_eta(progress) {
                eta if eta.is_empty() => {
                    // Translators: the progress of an analysis, e.g. "40 of 120 tiles"
                    tr!(
                        "{done} of {n} tile" | "{done} of {n} tiles" % progress.total,
                        done
                    )
                }
                // Translators: {eta} is the time left, e.g. "about 2m 05s left"
                eta => tr!(
                    "{done} of {n} tile, {eta}" | "{done} of {n} tiles, {eta}" % progress.total,
                    done,
                    eta
                ),
            };
            job.update(Some(progress.fraction()), Some(&message));
        }
        self.core.dirty.mark(|d| {
            d.sidebar = true;
            d.status = true;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_site_model_is_named_by_its_file_in_every_language() {
        let name = |url: Option<&str>| {
            download_args(&Config {
                download_url: url.map(str::to_string),
                download_sha256: Some("b".repeat(64)),
                ..Default::default()
            })
            .name
        };
        // eov shows the name in its dialogs and keeps it in its model
        // store: the file's name, as eov names the file it downloads.
        assert_eq!(
            name(Some("https://models.example.org/vae/our-vae.onnx")).as_deref(),
            Some("our-vae.onnx")
        );
        assert_eq!(
            name(Some(
                "https://models.example.org/our%20vae%C3%A9.onnx?key=1#top"
            ))
            .as_deref(),
            Some("our vaeé.onnx")
        );
        // Escapes that make no printable text stay as they are.
        assert_eq!(
            name(Some("https://models.example.org/a%FF%2.onnx")).as_deref(),
            Some("a%FF%2.onnx")
        );
        assert_eq!(
            name(Some("https://models.example.org/a%0Ab.onnx")).as_deref(),
            Some("a%0Ab.onnx")
        );
        // Without a file (or an address), eov names the model itself.
        assert_eq!(name(Some("https://models.example.org/")), None);
        assert_eq!(name(Some("https://models.example.org")), None);
        assert_eq!(name(None), None);
        // The reference model keeps its proper name.
        assert_eq!(
            download_args(&Config::default()).name.as_deref(),
            Some(model::reference().name)
        );
    }

    #[test]
    fn job_summaries_are_whole_sentences_of_their_job() {
        let tiles = Counts {
            analysed: 30,
            background: 2,
            failed: 0,
        };
        let text = |kind, outcome| {
            JobSummary {
                kind,
                outcome,
                elapsed: Duration::from_secs(75),
            }
            .text()
        };
        let failed = |counts| JobOutcome::Failed {
            error: Message::plain("disk full"),
            counts,
        };
        // The English is what eovae always said.
        assert_eq!(
            text(JobKind::WholeSlide, JobOutcome::Completed(tiles)),
            "Whole slide finished in 1m 15s: 30 analysed, 2 background."
        );
        assert_eq!(
            text(JobKind::Viewport, JobOutcome::Cancelled(tiles)),
            "Viewport cancelled after 1m 15s: 30 analysed, 2 background."
        );
        assert_eq!(
            text(JobKind::Roi, failed(tiles)),
            "Region of interest failed (30 analysed, 2 background): disk full"
        );
        assert_eq!(
            text(JobKind::Auto, failed(Counts::default())),
            "Automatic analysis failed: disk full"
        );
        assert_eq!(
            text(JobKind::Viewport, JobOutcome::Completed(Counts::default())),
            "Viewport: everything was already analysed."
        );
        // Every job and outcome has a sentence of its own: none is made
        // by putting a job's name in front of a shared one.
        let kinds = [
            JobKind::WholeSlide,
            JobKind::Viewport,
            JobKind::Roi,
            JobKind::Auto,
        ];
        let outcomes = [
            JobOutcome::Completed(tiles),
            JobOutcome::Cancelled(tiles),
            failed(tiles),
            failed(Counts::default()),
        ];
        for outcome in outcomes {
            let texts: BTreeSet<String> = kinds
                .iter()
                .map(|kind| text(*kind, outcome.clone()))
                .collect();
            assert_eq!(texts.len(), kinds.len(), "{texts:?}");
        }
    }
}
