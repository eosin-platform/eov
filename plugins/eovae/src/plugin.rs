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

/// Automatic analysis stops at this many visible tiles (zoomed out, the
/// view would turn into a whole-slide run, VAE-6)…
pub const AUTO_MAX_TILES: u64 = 400;
/// …or when tiles are smaller than this on screen.
pub const AUTO_MIN_TILE_PX: f64 = 24.0;

pub fn raster_layer(file_id: FileId) -> String {
    format!("vae-{file_id}")
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

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub enum ModelState {
    None,
    Loading { path: PathBuf },
    Ready { info: Arc<ModelInfo> },
    Failed { path: PathBuf, error: String },
}

/// What the engine cell last reported.
#[derive(Default)]
pub struct EngineStatus {
    /// Progress of a build in progress.
    pub building: Option<String>,
    /// Info of the engine built for a results signature.
    pub info: Option<(String, EngineInfo)>,
    /// The last build failure: (signature, error).
    pub error: Option<(String, String)>,
}

pub struct FileState {
    pub info: FileInfo,
    pub geometry: SlideGeometry,
    /// `None` while being resolved.
    pub identity: Option<SlideIdentity>,
    pub mode: Mode,
    pub grid: bool,
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

#[derive(Clone, Debug)]
pub struct JobSummary {
    pub text: String,
    pub error: bool,
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
    pub auto_hint: Option<String>,
    pub sidebar: Option<Surface>,
    /// Pending dialog request ids.
    pub model_dialog: Option<String>,
    pub runtime_dialog: Option<String>,
    pub export_dialog: Option<(String, FileId)>,
    pub clear_confirm: Option<String>,
    pub hover: Option<(FileId, TileCoord)>,
    pub pulse: Option<(FileId, TileCoord, Instant)>,
}

impl State {
    fn new(config: Config) -> Self {
        Self {
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
        }
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
        let model = self.model_info()?;
        Some(EngineRequest {
            output: self.config.output_for(model),
            latent: model.latent.as_ref().map(|l| l.name.clone()),
            model: Arc::clone(model),
            device: self.config.device,
            threads: self.config.effective_threads(),
            tf32: self.config.tf32,
            batch_size: self.config.batch_size,
            runtime_path: self.config.runtime_path.as_ref().map(PathBuf::from),
        })
    }

    /// The file a pane shows.
    pub fn pane_file(&self, pane: PaneId) -> Option<FileId> {
        self.panes.get(&pane).and_then(|p| p.file_id)
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
    ModelFailed(String),
    EngineNotReady,
    EngineFailed(String),
    SlideResolving,
    NoLevels,
    UnknownFile,
}

impl NoRun {
    pub fn message(&self) -> String {
        match self {
            Self::NoModel => "Load an ONNX model first.".to_string(),
            Self::ModelLoading => "The model is still loading.".to_string(),
            Self::ModelFailed(error) => format!("The model could not be loaded: {error}"),
            Self::EngineNotReady => "Preparing the inference engine…".to_string(),
            Self::EngineFailed(error) => error.clone(),
            Self::SlideResolving => "Identifying the slide…".to_string(),
            Self::NoLevels => "The slide reports no pyramid levels.".to_string(),
            Self::UnknownFile => "The slide is not open.".to_string(),
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
    let grid = TileGrid::new(&file.geometry, &choice.level, model.tile_size);
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
}

impl DirtyFlags {
    pub fn any(&self) -> bool {
        self.sidebar || self.hud || self.status || self.all_overlays || !self.overlays.is_empty()
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
    failed: Option<(String, String)>,
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
        let (config, warning) = Config::from_settings(&init.settings);
        let plugin_root = PathBuf::from(&init.plugin_root);
        let cache = ResultCache::open(&PathBuf::from(&init.cache_dir), config.cache_limit_bytes());
        host.add_toolbar_button(ToolbarButtonSpec {
            button_id: TOOLBAR_BUTTON.into(),
            tooltip: "VAE reconstruction analysis".into(),
            icon_svg: TOOL_ICON.into(),
            ..Default::default()
        })
        .map_err(String::from)?;
        host.add_hud_button(HudButtonSpec {
            button_id: HUD_GRID.into(),
            tooltip: "VAE tile grid".into(),
            icon_svg: GRID_ICON.into(),
            ..Default::default()
        })
        .map_err(String::from)?;
        host.add_hud_button(HudButtonSpec {
            button_id: HUD_VIEW.into(),
            tooltip: "VAE overlay".into(),
            icon_svg: VIEW_ICON.into(),
            menu: view_menu(Mode::Original),
            ..Default::default()
        })
        .map_err(String::from)?;
        if let Some(warning) = warning {
            host.warn(warning);
        }
        let results = ResultStore::new(config.memory_limit_bytes());
        let core = Arc::new(Self {
            backend: B::new(&plugin_root),
            plugin_root,
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
        let presenter = Arc::clone(&core);
        let handle = std::thread::Builder::new()
            .name("eovae-presenter".into())
            .spawn(move || crate::presenter::run(presenter))
            .map_err(|e| e.to_string())?;
        *lock(&core.presenter) = Some(handle);
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
                if result.is_err() {
                    core.host
                        .error("an eovae worker thread failed unexpectedly");
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
            HostEvent::SurfaceClosed { surface_id } => {
                let mut state = lock(&self.state);
                if state.sidebar.as_ref().is_some_and(|s| s.id() == surface_id) {
                    state.sidebar = None;
                    drop(state);
                    let _ = self.host.set_toolbar_button_active(TOOLBAR_BUTTON, false);
                }
                Ok(())
            }
            _ => Ok(()),
        }
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
        let model_path = {
            let state = lock(&self.state);
            matches!(state.model, ModelState::None)
                .then(|| state.config.model_path.clone())
                .flatten()
        };
        if let Some(path) = model_path {
            self.load_model(PathBuf::from(path), false);
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
            state.files.insert(
                file_id,
                FileState {
                    geometry: SlideGeometry::from_file_info(&file),
                    info: file.clone(),
                    identity: None,
                    mode: Mode::Original,
                    grid: false,
                },
            );
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
            state
                .files
                .remove(&file_id)
                .and_then(|file| file.identity)
                .map(|identity| identity.key)
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
        if let Some(file_id) = file_id {
            self.ensure_file(file_id);
        }
        {
            let mut state = lock(&self.state);
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
        self.dirty.mark(|d| {
            d.sidebar = true;
            d.hud = true;
        });
        Ok(())
    }

    fn on_view_changed(self: &Arc<Self>, view: ViewInfo) -> Result<(), String> {
        self.ensure_file(view.file_id);
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
                if let Some(file) = state.files.get_mut(&file_id) {
                    file.grid = !file.grid;
                }
            }
            (HUD_VIEW, Some(item)) => {
                let Some(mode) = Mode::from_id(item) else {
                    return Ok(());
                };
                self.set_mode(file_id, mode, view)?;
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

    /// Changes what a slide's overlay shows.
    pub fn set_mode(
        self: &Arc<Self>,
        file_id: FileId,
        mode: Mode,
        view: Option<ViewInfo>,
    ) -> Result<(), String> {
        let needs_model = {
            let mut state = lock(&self.state);
            if let Some(file) = state.files.get_mut(&file_id) {
                file.mode = mode;
            }
            mode != Mode::Original
                && matches!(state.model, ModelState::None | ModelState::Failed { .. })
        };
        if needs_model {
            self.host.notify(
                NotifyLevel::Info,
                "Load an ONNX model in the VAE sidebar to analyse this slide.",
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
            state
                .focused_pane
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
                    .with_title("VAE analysis")
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
                "The ONNX Runtime library is used from the next start of eov if a runtime is already loaded.",
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
                    Ok(()) => core
                        .host
                        .notify(NotifyLevel::Success, "The VAE result cache was cleared."),
                    Err(error) => core
                        .host
                        .notify_error("Could not clear the VAE result cache", error),
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
                .unwrap_or_else(|_| Err("the model file could not be read".to_string()));
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
                        .notify_error("The VAE model could not be loaded", error);
                }
                None => core.prebuild(),
            }
        });
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
    ) -> Result<Box<dyn InferenceEngine>, String> {
        let signature = request.signature();
        let mut slot = lock(&self.engine.slot);
        loop {
            if stop.load(Ordering::SeqCst) || self.stopping.load(Ordering::SeqCst) {
                return Err("cancelled".to_string());
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
                .unwrap_or_else(|_| Err("the inference engine failed unexpectedly".to_string()));
                let cancelled = stop.load(Ordering::SeqCst) || self.stopping.load(Ordering::SeqCst);
                let mut slot = lock(&self.engine.slot);
                let result = match built {
                    Ok(engine) => {
                        slot.signature = Some(signature.clone());
                        slot.failed = None;
                        Ok(engine)
                    }
                    Err(_) if cancelled => {
                        // Abandoned, not broken: build again when asked.
                        slot.busy = false;
                        drop(slot);
                        self.engine.changed.notify_all();
                        lock(&self.state).engine.building = None;
                        self.dirty.mark(|d| d.sidebar = true);
                        return Err("cancelled".to_string());
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
                        format!("VAE inference runs on the CPU: {reason}"),
                    );
                }
                if let Err(error) = &result {
                    self.host
                        .notify_error("The VAE model cannot run", error.clone());
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
            let mode = state.files.get(&view.file_id).map(|f| f.mode);
            if !state.config.auto_analyze
                || mode.is_none_or(|m| m == Mode::Original)
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
                            state.auto_hint = Some(format!(
                                "Zoom in to analyse automatically ({} tiles visible), or run Viewport or Whole slide.",
                                range.len()
                            ));
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
                phase: "Starting…".to_string(),
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
    ) -> Result<JobOutcome, String> {
        if control.is_cancelled() {
            return Ok(JobOutcome::Cancelled(Counts::default()));
        }
        let engine_request = lock(&self.state)
            .engine_request()
            .ok_or_else(|| NoRun::NoModel.message())?;
        self.set_phase(id, "Preparing the model…");
        let mut engine = match self.acquire_engine(
            &engine_request,
            control.stop_flag(),
            request.kind.explicit(),
        ) {
            Ok(engine) => engine,
            Err(_) if control.is_cancelled() => {
                return Ok(JobOutcome::Cancelled(Counts::default()));
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
    ) -> Result<JobOutcome, String> {
        let waited = Instant::now();
        let (ctx, config, mode) = loop {
            let attempt = {
                let state = lock(&self.state);
                run_context(&state, request.file_id).map(|ctx| {
                    let mode = state
                        .files
                        .get(&request.file_id)
                        .map(|f| f.mode)
                        .unwrap_or_default();
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
                    return Ok(JobOutcome::Cancelled(Counts::default()));
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
                    .map_err(String::from)?
                    .ok_or_else(|| "Draw a region of interest on the slide first.".to_string())?;
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
                    .ok_or_else(|| "The slide is not shown in a pane.".to_string())?;
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
            return Ok(JobOutcome::Completed(Counts::default()));
        }
        let host_job = if request.kind.explicit() {
            let title = match request.kind {
                JobKind::WholeSlide => "VAE: whole slide",
                JobKind::Roi => "VAE: region of interest",
                _ => "VAE: viewport",
            };
            self.host.start_job(title, true).ok()
        } else {
            None
        };
        {
            let mut state = lock(&self.state);
            if let Some(job) = state.job.as_mut().filter(|j| j.id == id) {
                job.host_job = host_job.as_ref().map(|j| j.id().to_string());
                job.phase = format!("Analysing on {}", engine.info().label);
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
            error: "internal error while analysing tiles".to_string(),
            counts: Counts::default(),
        });
        if let Some(job) = host_job {
            job.finish(match &outcome {
                JobOutcome::Failed { error, .. } => Some(error.clone()),
                _ => None,
            });
        }
        Ok(outcome)
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
        outcome: std::thread::Result<Result<JobOutcome, String>>,
    ) {
        let outcome = match outcome {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(error)) => JobOutcome::Failed {
                error,
                counts: Counts::default(),
            },
            Err(payload) => JobOutcome::Failed {
                error: format!(
                    "internal error: {}",
                    payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_default()
                ),
                counts: Counts::default(),
            },
        };
        let (elapsed, superseded) = {
            let state = lock(&self.state);
            let current = state.job.as_ref().filter(|j| j.id == id);
            (
                current.map(|j| j.started.elapsed()).unwrap_or_default(),
                current.is_none(),
            )
        };
        let summary = summarize_outcome(request.kind, &outcome, elapsed);
        let failed = matches!(outcome, JobOutcome::Failed { .. });
        {
            let mut state = lock(&self.state);
            if state.job.as_ref().is_some_and(|j| j.id == id) {
                state.job = None;
            }
            if let Some(summary) = summary.clone()
                && (request.kind.explicit() || failed)
                && !(superseded && control.is_cancelled())
            {
                state.last_job = Some(summary);
            }
        }
        if let JobOutcome::Failed { error, .. } = &outcome {
            self.host.error(format!("VAE analysis failed: {error}"));
            if !request.kind.explicit() {
                // Explicit jobs report through the host job; automatic ones
                // would otherwise fail silently.
                self.host.notify_error("VAE analysis failed", error.clone());
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
                    .notify_error("Nothing to export", reason.message());
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
            "grid": {"tile": ctx.grid.tile, "cols": ctx.grid.cols, "rows": ctx.grid.rows, "level": ctx.grid.level, "downsample": ctx.grid.downsample},
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
                format!("Exported {} tiles to {}", rows.len(), path.display()),
            ),
            Err(error) => self
                .host
                .notify_error("The export failed", error.to_string()),
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
            self.host
                .warn(format!("eovae settings could not be saved: {error}"));
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
            label: mode.label().to_string(),
            checked: Some(mode == current),
            ..Default::default()
        })
        .collect()
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match seconds {
        0..=59 => format!("{seconds} s"),
        60..=3599 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h {:02}m", seconds / 3600, seconds / 60 % 60),
    }
}

pub fn format_eta(progress: &Progress) -> String {
    progress
        .eta()
        .map(|eta| format!("about {} left", format_duration(eta)))
        .unwrap_or_default()
}

fn summarize_outcome(kind: JobKind, outcome: &JobOutcome, elapsed: Duration) -> Option<JobSummary> {
    let describe = |counts: &Counts| {
        let mut parts = vec![format!("{} analysed", counts.analysed)];
        if counts.background > 0 {
            parts.push(format!("{} background", counts.background));
        }
        if counts.failed > 0 {
            parts.push(format!("{} unreadable", counts.failed));
        }
        parts.join(", ")
    };
    let what = match kind {
        JobKind::WholeSlide => "Whole slide",
        JobKind::Viewport => "Viewport",
        JobKind::Roi => "Region of interest",
        JobKind::Auto => "Automatic analysis",
    };
    match outcome {
        JobOutcome::Completed(counts) if counts.total() == 0 => Some(JobSummary {
            text: format!("{what}: everything was already analysed."),
            error: false,
        }),
        JobOutcome::Completed(counts) => Some(JobSummary {
            text: format!(
                "{what} finished in {}: {}.",
                format_duration(elapsed),
                describe(counts)
            ),
            error: false,
        }),
        JobOutcome::Cancelled(counts) => Some(JobSummary {
            text: format!(
                "{what} cancelled after {}: {}.",
                format_duration(elapsed),
                describe(counts)
            ),
            error: false,
        }),
        JobOutcome::Failed { error, counts } => Some(JobSummary {
            text: if counts.total() > 0 {
                format!("{what} failed ({}): {error}", describe(counts))
            } else {
                format!("{what} failed: {error}")
            },
            error: true,
        }),
    }
}

// ---------------------------------------------------------------------------
// Pipeline plumbing
// ---------------------------------------------------------------------------

/// Slide pixels through the host (any thread, never blocks the UI).
struct HostPixels {
    host: Host,
}

impl PixelSource for HostPixels {
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String> {
        let image = self.host.read_region(*request).map_err(|e| e.message)?;
        if image.width != request.width || image.height != request.height {
            return Err(format!(
                "the host returned {}×{} for a {}×{} region",
                image.width, image.height, request.width, request.height
            ));
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
            let message = format!(
                "{} of {} tiles{}",
                progress.done,
                progress.total,
                match format_eta(progress) {
                    eta if eta.is_empty() => String::new(),
                    eta => format!(", {eta}"),
                }
            );
            job.update(Some(progress.fraction()), Some(&message));
        }
        self.core.dirty.mark(|d| {
            d.sidebar = true;
            d.status = true;
        });
    }
}
