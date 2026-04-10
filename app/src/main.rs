// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod blitter;
mod callbacks;
mod config;
mod file_ops;
mod gpu;
mod render;
mod state;
mod tile_loader;
mod tools;
mod ui_update;

use anyhow::{Result, bail};
use common::{TileCache, ViewportState};
use gpu::GpuRenderer;
use parking_lot::RwLock;
use slint::{BackendSelector, Image, SharedString, Timer, TimerMode, VecModel};
use state::{AppState, PaneId, RenderBackend};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

#[derive(Clone, Default)]
struct PaneRenderCacheEntry {
    content: Option<Image>,
    minimap_thumbnail: Option<Image>,
}

#[derive(Clone)]
struct PaneUiModels {
    tabs: Rc<VecModel<TabData>>,
    measurements: Rc<VecModel<MeasurementLine>>,
}

impl Default for PaneUiModels {
    fn default() -> Self {
        Self {
            tabs: Rc::new(VecModel::default()),
            measurements: Rc::new(VecModel::default()),
        }
    }
}

thread_local! {
    static GPU_RENDERER_HANDLE: RefCell<Option<Rc<RefCell<GpuRenderer>>>> = const { RefCell::new(None) };
    static PANE_RENDER_CACHE: RefCell<Vec<PaneRenderCacheEntry>> = const { RefCell::new(Vec::new()) };
    static PANE_VIEW_MODEL: RefCell<Rc<VecModel<PaneViewData>>> = RefCell::new(Rc::new(VecModel::default()));
    static PANE_UI_MODELS: RefCell<Vec<PaneUiModels>> = const { RefCell::new(Vec::new()) };
}

// Debug macro - set to no-op for release, enable for debugging
#[allow(unused_macros)]
macro_rules! dbg_print {
    ($($arg:tt)*) => {{
        // Uncomment for verbose debugging:
        // eprintln!($($arg)*);
        // let _ = std::io::stderr().flush();
    }};
}

slint::include_modules!();

/// Frame rate for viewport updates
const TARGET_FPS: f64 = 60.0;
const FRAME_DURATION_MS: u64 = (1000.0 / TARGET_FPS) as u64;

struct LaunchOptions {
    debug_mode: bool,
    files_to_open: Vec<PathBuf>,
    render_backend_override: Option<RenderBackend>,
}

fn parse_launch_options() -> Result<LaunchOptions> {
    let mut debug_mode = false;
    let mut files_to_open = Vec::new();
    let mut render_backend_override = None;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--debug" | "-d" => debug_mode = true,
            "--cpu" => {
                if render_backend_override == Some(RenderBackend::Gpu) {
                    bail!(
                        "--cpu and --gpu are mutually exclusive; choose only one rendering override"
                    );
                }
                render_backend_override = Some(RenderBackend::Cpu);
            }
            "--gpu" => {
                if render_backend_override == Some(RenderBackend::Cpu) {
                    bail!(
                        "--cpu and --gpu are mutually exclusive; choose only one rendering override"
                    );
                }
                render_backend_override = Some(RenderBackend::Gpu);
            }
            _ => {
                let path = PathBuf::from(&arg);
                if path.exists() {
                    files_to_open.push(path);
                }
            }
        }
    }

    Ok(LaunchOptions {
        debug_mode,
        files_to_open,
        render_backend_override,
    })
}

fn select_backend() -> Result<bool> {
    let gpu_result = BackendSelector::new()
        .backend_name("winit".to_string())
        .renderer_name("femtovg-wgpu".to_string())
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::default())
        .select();

    match gpu_result {
        Ok(()) => Ok(true),
        Err(err) => {
            warn!(
                "GPU backend unavailable, falling back to CPU renderer: {}",
                err
            );
            BackendSelector::new()
                .backend_name("winit".to_string())
                .renderer_name("femtovg".to_string())
                .select()?;
            Ok(false)
        }
    }
}

fn pane_from_index(index: i32) -> PaneId {
    PaneId(index.max(0) as usize)
}

pub(crate) fn with_pane_render_cache<T>(
    pane_count: usize,
    f: impl FnOnce(&mut Vec<PaneRenderCacheEntry>) -> T,
) -> T {
    PANE_RENDER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() < pane_count {
            cache.resize_with(pane_count, PaneRenderCacheEntry::default);
        } else if cache.len() > pane_count {
            cache.truncate(pane_count);
        }
        f(&mut cache)
    })
}

pub(crate) fn set_cached_pane_content(pane: PaneId, image: Image) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].content = Some(image);
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, "set pane cached content");
    }
}

pub(crate) fn set_cached_pane_minimap(pane: PaneId, image: Option<Image>) {
    let has_minimap = image.is_some();
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].minimap_thumbnail = image;
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, has_minimap, "set pane cached minimap");
    }
}

fn with_pane_view_model<T>(f: impl FnOnce(&Rc<VecModel<PaneViewData>>) -> T) -> T {
    PANE_VIEW_MODEL.with(|model| f(&model.borrow()))
}

fn with_pane_ui_models<T>(pane_count: usize, f: impl FnOnce(&mut Vec<PaneUiModels>) -> T) -> T {
    PANE_UI_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        if models.len() < pane_count {
            models.resize_with(pane_count, PaneUiModels::default);
        } else if models.len() > pane_count {
            models.truncate(pane_count);
        }
        f(&mut models)
    })
}

fn reset_pane_ui_state() {
    PANE_RENDER_CACHE.with(|cache| cache.borrow_mut().clear());
    PANE_UI_MODELS.with(|models| models.borrow_mut().clear());
    PANE_VIEW_MODEL.with(|model| {
        let model = model.borrow_mut();
        model.clear();
    });
}

fn insert_pane_ui_state(new_pane: PaneId, source_pane: Option<PaneId>) {
    PANE_RENDER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let insert_index = new_pane.0.min(cache.len());
        let entry = source_pane
            .and_then(|pane| cache.get(pane.0).cloned())
            .unwrap_or_default();
        cache.insert(insert_index, entry);
    });

    PANE_UI_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        let insert_index = new_pane.0.min(models.len());
        models.insert(insert_index, PaneUiModels::default());
    });
}

pub(crate) fn clear_cached_pane(pane: PaneId) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0] = PaneRenderCacheEntry::default();
    });

    if pane.0 == 1 {
        debug!(pane = pane.0, "cleared pane cache");
    }
}

fn refresh_tab_ui(ui: &AppWindow, state: &AppState) {
    reset_pane_ui_state();
    update_tabs(ui, state);
}

fn slider_value_to_zoom(value: f32) -> f64 {
    ui_update::slider_value_to_zoom(value)
}

fn set_gpu_renderer_handle(renderer: Rc<RefCell<GpuRenderer>>) {
    GPU_RENDERER_HANDLE.with(|handle| {
        *handle.borrow_mut() = Some(renderer);
    });
}

pub(crate) fn with_gpu_renderer<R>(f: impl FnOnce(&Rc<RefCell<GpuRenderer>>) -> R) -> Option<R> {
    GPU_RENDERER_HANDLE.with(|handle| handle.borrow().as_ref().map(f))
}

fn copy_text_to_clipboard(clipboard: &Rc<RefCell<Option<arboard::Clipboard>>>, text: String) {
    let mut clipboard_handle = clipboard.borrow_mut();
    if clipboard_handle.is_none() {
        match arboard::Clipboard::new() {
            Ok(new_clipboard) => {
                *clipboard_handle = Some(new_clipboard);
            }
            Err(err) => {
                warn!("Failed to initialize clipboard: {}", err);
                return;
            }
        }
    }

    if let Some(clipboard) = clipboard_handle.as_mut()
        && let Err(err) = clipboard.set_text(text)
    {
        warn!("Failed to copy text to clipboard: {}", err);
        *clipboard_handle = None;
    }
}

fn request_render_loop(
    render_timer: &Rc<Timer>,
    ui_weak: &slint::Weak<AppWindow>,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
){
    let should_start = {
        let mut state = state.write();
        state.request_render();
        if state.render_loop_running {
            false
        } else {
            state.render_loop_running = true;
            true
        }
    };

    if !should_start {
        return;
    }

    let timer_for_callback = Rc::clone(render_timer);
    let ui_weak = ui_weak.clone();
    let state = Arc::clone(state);
    let tile_cache = Arc::clone(tile_cache);
    render_timer.start(
        TimerMode::Repeated,
        Duration::from_millis(FRAME_DURATION_MS),
        move || {
            let Some(ui) = ui_weak.upgrade() else {
                timer_for_callback.stop();
                state.write().render_loop_running = false;
                return;
            };

            if !render::update_and_render(&ui, &state, &tile_cache) {
                timer_for_callback.stop();
                let mut state = state.write();
                state.render_loop_running = false;
            }
        },
    );
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting EosMol WSI Viewer");

    let launch_options = parse_launch_options()?;
    let persisted_backend = config::load_render_backend()?;
    let initial_backend = launch_options
        .render_backend_override
        .or(persisted_backend)
        .unwrap_or(RenderBackend::Cpu);

    if launch_options.debug_mode {
        info!("Debug mode enabled - FPS overlay will be shown");
    }

    if !launch_options.files_to_open.is_empty() {
        info!(
            "Opening {} file(s) from command line",
            launch_options.files_to_open.len()
        );
    }

    let gpu_backend_available = select_backend()?;

    let state = Arc::new(RwLock::new(AppState::new(launch_options.debug_mode)));
    let tile_cache = Arc::new(TileCache::new());

    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();
    let gpu_renderer = Rc::new(RefCell::new(GpuRenderer::new()));
    GpuRenderer::install(&ui, Rc::clone(&gpu_renderer))?;
    set_gpu_renderer_handle(Rc::clone(&gpu_renderer));

    let render_timer = Rc::new(Timer::default());

    setup_callbacks(
        &ui,
        Arc::clone(&state),
        Arc::clone(&tile_cache),
        Rc::clone(&render_timer),
    );

    ui.set_debug_mode(launch_options.debug_mode);
    {
        let mut state = state.write();
        state.gpu_backend_available = gpu_backend_available;
        state.select_render_backend(initial_backend);
        update_render_backend(&ui, &state);
        if initial_backend == RenderBackend::Gpu && state.render_backend != RenderBackend::Gpu {
            ui.set_status_text(SharedString::from(
                "GPU renderer unavailable; using CPU renderer",
            ));
        }
    }

    {
        let state = state.read();
        update_recent_files(&ui, &state);
    }

    if launch_options.files_to_open.is_empty() {
        {
            let mut state = state.write();
            state.create_home_tab();
        }
        let state = state.read();
        update_tabs(&ui, &state);
    }

    for path in launch_options.files_to_open {
        open_file(&ui, &state, &tile_cache, &render_timer, path);
    }

    request_render_loop(&render_timer, &ui_weak, &state, &tile_cache);

    dbg_print!("[MAIN] Timer started, running UI");

    ui.run()?;

    info!("Application shutting down");
    Ok(())
}

fn setup_callbacks(
    ui: &AppWindow,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
) {
    callbacks::setup_callbacks(ui, state, tile_cache, render_timer);
}

fn open_file(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
) {
    let pane_count = state.read().panes.len().max(1);
    with_pane_render_cache(pane_count, |pane_render_cache| {
        with_pane_ui_models(pane_count, |pane_ui_models| {
            with_pane_view_model(|pane_view_model| {
                file_ops::open_file(
                    ui,
                    state,
                    tile_cache,
                    render_timer,
                    path,
                    pane_render_cache,
                    pane_ui_models,
                    pane_view_model,
                );
            });
        });
    });
}

pub(crate) fn update_tabs(ui: &AppWindow, state: &AppState) {
    let pane_count = state.panes.len().max(1);
    with_pane_render_cache(pane_count, |pane_render_cache| {
        with_pane_ui_models(pane_count, |pane_ui_models| {
            with_pane_view_model(|pane_view_model| {
                ui_update::update_tabs(ui, state, pane_render_cache, pane_ui_models, pane_view_model);
            });
        });
    });
}

fn update_recent_files(ui: &AppWindow, state: &AppState) {
    ui_update::update_recent_files(ui, state)
}

fn build_recent_menu_items(state: &AppState) -> Vec<ContextMenuItem> {
    ui_update::build_recent_menu_items(state)
}

fn update_render_backend(ui: &AppWindow, state: &AppState) {
    ui_update::update_render_backend(ui, state)
}

// ============ Tool handling functions ============

fn update_tool_state(ui: &AppWindow, state: &AppState) {
    tools::update_tool_state(ui, state)
}

fn update_tool_overlays(ui: &AppWindow, state: &AppState) {
    tools::update_tool_overlays(ui, state)
}

fn handle_tool_mouse_down(state: &mut AppState, screen_x: f64, screen_y: f64) {
    tools::handle_tool_mouse_down(state, screen_x, screen_y)
}

fn handle_tool_mouse_move(state: &mut AppState, screen_x: f64, screen_y: f64) {
    tools::handle_tool_mouse_move(state, screen_x, screen_y)
}

fn handle_tool_mouse_up(state: &mut AppState, screen_x: f64, screen_y: f64) {
    tools::handle_tool_mouse_up(state, screen_x, screen_y)
}
