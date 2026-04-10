// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod gpu;
mod render;
mod state;
mod tile_loader;

use anyhow::{Result, bail};
use common::{
    TileCache, TileManager, Viewport, ViewportState, WsiFile,
    viewport::{MAX_ZOOM, MIN_ZOOM},
};
use gpu::{GpuRenderer, SurfaceSlot, TileDraw};
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{
    BackendSelector, Image, Model, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode,
    VecModel,
};
use state::{AppState, OpenFile, PaneId, RenderBackend, TileRequestSignature};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tile_loader::{TileLoader, calculate_wanted_tiles};
use tracing::{error, info, trace, warn};

/// Coarse tile blend data: (tile_data, coarse_offset, coarse_size, blend_factor)
type CoarseBlendData = (Arc<common::TileData>, [f32; 2], [f32; 2], f32);

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

#[derive(Default)]
struct PaneRenderOutcome {
    image: Option<Image>,
    keep_running: bool,
    rendered: bool,
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

fn ui_render_mode(backend: RenderBackend) -> RenderMode {
    match backend {
        RenderBackend::Cpu => RenderMode::Cpu,
        RenderBackend::Gpu => RenderMode::Gpu,
    }
}

fn pane_from_index(index: i32) -> PaneId {
    PaneId(index.max(0) as usize)
}

fn with_pane_render_cache<T>(pane_count: usize, f: impl FnOnce(&mut Vec<PaneRenderCacheEntry>) -> T) -> T {
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

fn set_cached_pane_content(pane: PaneId, image: Image) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].content = Some(image);
    });
}

fn set_cached_pane_minimap(pane: PaneId, image: Option<Image>) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0].minimap_thumbnail = image;
    });
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

fn model_matches<T>(model: &VecModel<T>, data: &[T]) -> bool
where
    T: Clone + PartialEq + 'static,
{
    model.row_count() == data.len()
        && data
            .iter()
            .enumerate()
            .all(|(index, value)| model.row_data(index).as_ref() == Some(value))
}

fn pane_view_data_changed(
    existing: &PaneViewData,
    next: &PaneViewData,
) -> bool {
    existing.id != next.id
        || existing.content != next.content
        || existing.viewport_info != next.viewport_info
        || existing.minimap_thumbnail != next.minimap_thumbnail
        || existing.minimap_rect != next.minimap_rect
        || existing.is_home_tab != next.is_home_tab
        || existing.zoom_slider_position != next.zoom_slider_position
        || existing.roi_rect != next.roi_rect
        || existing.candidate_measurement != next.candidate_measurement
        || existing.is_loading != next.is_loading
}

fn clear_cached_pane(pane: PaneId) {
    with_pane_render_cache(pane.0 + 1, |cache| {
        cache[pane.0] = PaneRenderCacheEntry::default();
    });
}

fn refresh_tab_ui(ui: &AppWindow, state: &AppState) {
    reset_pane_ui_state();
    update_tabs(ui, state);
}

fn zoom_to_slider_value(zoom: f64) -> f32 {
    let log_min = MIN_ZOOM.ln();
    let log_max = MAX_ZOOM.ln();
    let normalized =
        ((zoom.clamp(MIN_ZOOM, MAX_ZOOM).ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    normalized as f32
}

fn slider_value_to_zoom(value: f32) -> f64 {
    let clamped = value.clamp(0.0, 1.0) as f64;
    let log_min = MIN_ZOOM.ln();
    let log_max = MAX_ZOOM.ln();
    (log_min + (log_max - log_min) * clamped).exp()
}

fn set_gpu_renderer_handle(renderer: Rc<RefCell<GpuRenderer>>) {
    GPU_RENDERER_HANDLE.with(|handle| {
        *handle.borrow_mut() = Some(renderer);
    });
}

fn with_gpu_renderer<R>(f: impl FnOnce(&Rc<RefCell<GpuRenderer>>) -> R) -> Option<R> {
    GPU_RENDERER_HANDLE.with(|handle| handle.borrow().as_ref().map(f))
}

fn copy_text_to_clipboard(
    clipboard: &Rc<RefCell<Option<arboard::Clipboard>>>,
    text: String,
) {
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
) {
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

            if !update_and_render(&ui, &state, &tile_cache) {
                timer_for_callback.stop();
                let mut state = state.write();
                state.render_loop_running = false;
            }
        },
    );
}

fn main() -> Result<()> {
    // Initialize logging - RUST_LOG env controls level
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

    // Create application state
    let state = Arc::new(RwLock::new(AppState::new(launch_options.debug_mode)));
    let tile_cache = Arc::new(TileCache::new());

    // Create UI
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();
    let gpu_renderer = Rc::new(RefCell::new(GpuRenderer::new()));
    GpuRenderer::install(&ui, Rc::clone(&gpu_renderer))?;
    set_gpu_renderer_handle(Rc::clone(&gpu_renderer));

    let render_timer = Rc::new(Timer::default());

    // Set up callbacks
    setup_callbacks(
        &ui,
        Arc::clone(&state),
        Arc::clone(&tile_cache),
        Rc::clone(&render_timer),
    );

    // Set debug mode on UI
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

    // Initialize recent files list
    {
        let state = state.read();
        update_recent_files(&ui, &state);
    }

    // Open files from command line
    for path in launch_options.files_to_open {
        open_file(&ui, &state, &tile_cache, &render_timer, path);
    }

    request_render_loop(&render_timer, &ui_weak, &state, &tile_cache);

    dbg_print!("[MAIN] Timer started, running UI");

    // Run application
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
    let ui_weak = ui.as_weak();
    let clipboard = Rc::new(RefCell::new(None));

    // Open file callback
    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_file_requested(move || {
            let ui = ui_weak.upgrade().unwrap();

            // Show file dialog
            let dialog = FileDialog::new()
                .add_filter(
                    "WSI Files",
                    &[
                        "svs", "tif", "tiff", "ndpi", "vms", "vmu", "scn", "mrxs", "bif",
                    ],
                )
                .add_filter("All Files", &["*"]);

            if let Some(path) = dialog.pick_file() {
                open_file(&ui, &state, &tile_cache, &render_timer, path);
            }
        });
    }

    // New tab callback - creates a home tab
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_new_tab_requested(move |pane| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.set_focused_pane(pane_from_index(pane));
                    state.create_home_tab();
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Open recent file callback
    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_recent_file(move |path_str| {
            let path = PathBuf::from(path_str.as_str());
            if path.exists()
                && let Some(ui) = ui_weak.upgrade()
            {
                open_file(&ui, &state, &tile_cache, &render_timer, path);
            }
        });
    }

    // Open-button context menu for recent files
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_open_recent_menu_requested(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                let items = {
                    let state = state_handle.read();
                    build_recent_menu_items(&state)
                };
                ui.set_context_menu_items(Rc::new(VecModel::from(items)).into());
                ui.set_context_menu_tab_id(-1);
                ui.set_drag_source_pane(-1);
                ui.set_context_menu_x(x);
                ui.set_context_menu_y(y);
                ui.set_context_menu_visible(true);
            }
        });
    }

    // Context menu command handler
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();
        let clipboard = Rc::clone(&clipboard);

        ui.on_context_menu_command(move |id| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            ui.set_context_menu_visible(false);

            let command = id.to_string();
            if let Some(path_str) = command.strip_prefix("recent-file:") {
                let path = PathBuf::from(path_str);
                if path.exists() {
                    open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
                }
                return;
            }

            let pane = pane_from_index(ui.get_drag_source_pane());
            let tab_id = ui.get_context_menu_tab_id();

            match command.as_str() {
                "close" => {
                    {
                        let mut state = state_handle.write();
                        state.set_focused_pane(pane);
                        state.close_tab_in_pane(pane, tab_id);
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-others" => {
                    {
                        let mut state = state_handle.write();
                        let ids_to_close: Vec<i32> = state
                            .tabs_for_pane(pane)
                            .iter()
                            .copied()
                            .filter(|&id| id != tab_id)
                            .collect();
                        for id in ids_to_close {
                            state.set_focused_pane(pane);
                            state.close_tab_in_pane(pane, id);
                        }
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-right" => {
                    {
                        let mut state = state_handle.write();
                        let all_tabs: Vec<i32> = state.tabs_for_pane(pane).to_vec();
                        let mut found = false;
                        let ids_to_close: Vec<i32> = all_tabs
                            .iter()
                            .filter_map(|&id| {
                                if id == tab_id {
                                    found = true;
                                    None
                                } else if found {
                                    Some(id)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        for id in ids_to_close {
                            state.set_focused_pane(pane);
                            state.close_tab_in_pane(pane, id);
                        }
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "close-all" => {
                    {
                        let mut state = state_handle.write();
                        state.close_all_tabs();
                    }
                    let state = state_handle.read();
                    refresh_tab_ui(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                "open-folder" => {
                    let state = state_handle.read();
                    if let Some(file) = state.get_file(tab_id)
                        && let Some(parent) = file.path.parent()
                        && let Err(err) = open::that(parent)
                    {
                        error!("Failed to open folder: {}", err);
                    }
                }
                "copy-path" => {
                    let state = state_handle.read();
                    if let Some(file) = state.get_file(tab_id) {
                        copy_text_to_clipboard(&clipboard, file.path.display().to_string());
                    }
                }
                "split-right" => {
                    let split_enabled = {
                        let mut state = state_handle.write();
                        let source_pane = pane;
                        let new_pane = state.insert_pane(source_pane.0 + 1);
                        state.duplicate_tab_to_pane(tab_id, new_pane);
                        state.set_focused_pane(new_pane);
                        state.request_render();
                        state.split_enabled
                    };
                    reset_pane_ui_state();
                    ui.set_split_enabled(split_enabled);
                    let state = state_handle.read();
                    ui.set_focused_pane(state.focused_pane.as_index());
                    update_tabs(&ui, &state);
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
                _ => {}
            }
        });
    }

    // Render backend callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_render_backend_selected(move |mode| {
            let backend = match mode {
                RenderMode::Cpu => RenderBackend::Cpu,
                RenderMode::Gpu => RenderBackend::Gpu,
            };
            let gpu_fallback = {
                let mut state = state_handle.write();
                state.select_render_backend(backend);
                if let Err(err) = config::save_render_backend(state.render_backend) {
                    warn!("Failed to save render backend config: {}", err);
                }
                backend == RenderBackend::Gpu && state.render_backend != RenderBackend::Gpu
            };

            if let Some(ui) = ui_weak.upgrade() {
                let state = state_handle.read();
                update_render_backend(&ui, &state);
                if gpu_fallback {
                    ui.set_status_text(SharedString::from(
                        "GPU renderer unavailable; using CPU renderer",
                    ));
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_request_render(move || {
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Tab activated callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tab_activated(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.activate_tab_in_pane(pane_from_index(pane), id);
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Tab close callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tab_close_requested(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    state.set_focused_pane(pane_id);
                    state.close_tab_in_pane(pane_id, id);
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Close other tabs
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_other_tabs(move |pane, keep_id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    let ids_to_close: Vec<i32> = state
                        .tabs_for_pane(pane_id)
                        .iter()
                        .copied()
                        .filter(|&id| id != keep_id)
                        .collect();
                    for id in ids_to_close {
                        state.set_focused_pane(pane_id);
                        state.close_tab_in_pane(pane_id, id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Close tabs to the right
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_tabs_to_right(move |pane, from_id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    let all_tabs: Vec<i32> = state.tabs_for_pane(pane_id).to_vec();
                    let mut found = false;
                    let ids_to_close: Vec<i32> = all_tabs
                        .iter()
                        .filter_map(|&id| {
                            if id == from_id {
                                found = true;
                                None
                            } else if found {
                                Some(id)
                            } else {
                                None
                            }
                        })
                        .collect();
                    for id in ids_to_close {
                        state.set_focused_pane(pane_id);
                        state.close_tab_in_pane(pane_id, id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Close all tabs
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_all_tabs(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.close_all_tabs();
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Open containing folder
    {
        let state = Arc::clone(&state);

        ui.on_open_containing_folder(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id)
                && let Some(parent) = file.path.parent()
                && let Err(e) = open::that(parent)
            {
                error!("Failed to open folder: {}", e);
            }
        });
    }

    // Copy path
    {
        let state = Arc::clone(&state);
        let clipboard = Rc::clone(&clipboard);

        ui.on_copy_path(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id) {
                copy_text_to_clipboard(&clipboard, file.path.display().to_string());
            }
        });
    }

    // Split right / move between panes
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_right(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                let (split_enabled, focused_pane) = {
                    let mut state = state_handle.write();
                    let source_pane = pane_from_index(pane);
                    let new_pane = state.insert_pane(source_pane.0 + 1);
                    state.duplicate_tab_to_pane(id, new_pane);
                    state.set_focused_pane(new_pane);
                    state.request_render();
                    (state.split_enabled, state.focused_pane)
                };
                reset_pane_ui_state();
                ui.set_split_enabled(split_enabled);
                ui.set_focused_pane(focused_pane.as_index());
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Pane focused callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_pane_focused(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                state.request_render();
            }

            if let Some(ui) = ui_weak.upgrade() {
                ui.set_focused_pane(pane);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Split position changed callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_position_changed(move |pos| {
            {
                let mut state = state_handle.write();
                state.split_position = pos;
                state.request_render();
            }

            if let Some(ui) = ui_weak.upgrade() {
                ui.set_split_position(pos);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Viewport pan
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_pan(move |dx, dy| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.viewport.pan(dx as f64, dy as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Viewport start pan
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_start_pan(move |x, y| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.start_drag(x as f64, y as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Viewport end pan
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_end_pan(move || {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.end_drag();
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Viewport zoom
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_zoom(move |factor, x, y| {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.zoom_at(factor as f64, x as f64, y as f64);
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Fit to view callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_fit_to_view(move || {
            {
                let mut state = state_handle.write();
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.fit_to_view();
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Minimap navigation callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_minimap_navigate(move |nx, ny| {
            info!("Minimap navigate called: nx={}, ny={}", nx, ny);
            {
                let mut state = state_handle.write();
                let mut changed = false;

                if let Some(vs) = state.active_viewport_mut() {
                    let nx = (nx as f64).clamp(0.0, 1.0);
                    let ny = (ny as f64).clamp(0.0, 1.0);
                    vs.stop();
                    let vp = &mut vs.viewport;
                    let new_x = nx * vp.image_width;
                    let new_y = ny * vp.image_height;
                    info!(
                        "Setting viewport center: ({}, {}) -> ({}, {})",
                        vp.center.x, vp.center.y, new_x, new_y
                    );
                    vp.center.x = new_x;
                    vp.center.y = new_y;
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // File dropped callback
    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui.as_weak();

        ui.on_file_dropped(move |path_str| {
            // Parse dropped path - may include file:// prefix or be a path list
            let path_string = path_str.to_string();
            let paths: Vec<&str> = path_string
                .lines()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            for path_str in paths {
                let path = if path_str.starts_with("file://") {
                    PathBuf::from(path_str.strip_prefix("file://").unwrap_or(path_str))
                } else {
                    PathBuf::from(path_str)
                };

                if path.exists()
                    && let Some(ui) = ui_weak.upgrade()
                {
                    open_file(&ui, &state, &tile_cache, &render_timer, path);
                }
            }
        });
    }

    // Tool selected callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tool_selected(move |tool_type| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let tool = match tool_type {
                        ToolType::Navigate => state::Tool::Navigate,
                        ToolType::RegionOfInterest => state::Tool::RegionOfInterest,
                        ToolType::MeasureDistance => state::Tool::MeasureDistance,
                    };
                    state.set_tool(tool);
                }
                let state = state_handle.read();
                update_tool_state(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Tool mouse events
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_down(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_down(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_move(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_move(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_tool_mouse_up(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    handle_tool_mouse_up(&mut state, x as f64, y as f64);
                }
                let state = state_handle.read();
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Cancel tool callback
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_cancel_tool(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.cancel_tool();
                }
                let state = state_handle.read();
                update_tool_state(&ui, &state);
                update_tool_overlays(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_move_tab_to_pane(move |id, from, to| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let target_pane = pane_from_index(to);
                    state.move_tab_between_panes(id, pane_from_index(from), target_pane);
                    state.set_focused_pane(target_pane);
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_tab_by_drop(move |id, to| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let source_pane = state
                        .panes
                        .iter()
                        .enumerate()
                        .find(|(_, pane_state)| pane_state.tabs.contains(&id))
                        .map(|(index, _)| PaneId(index))
                        .unwrap_or(PaneId::PRIMARY);
                    state.split_tab_to_new_pane(id, source_pane, to.max(0) as usize);
                }
                reset_pane_ui_state();
                let state = state_handle.read();
                ui.set_split_enabled(state.split_enabled);
                ui.set_focused_pane(state.focused_pane.as_index());
                update_tabs(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Reorder tab within the same pane
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_reorder_tab(move |pane, id, new_index| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.reorder_tab(pane_from_index(pane), id, new_index);
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_zoom_slider_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let mut changed = false;
                if let Some(viewport) = state.active_viewport_mut() {
                    viewport.zoom_to(slider_value_to_zoom(value));
                    changed = true;
                }
                if changed {
                    state.request_render();
                }
            }

            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Close active tab (Ctrl+W)
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_close_active_tab(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane = state.focused_pane;
                    if let Some(active_id) = state.active_tab_id_for_pane(pane) {
                        state.close_tab_in_pane(pane, active_id);
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Window minimize
    {
        let ui_weak = ui_weak.clone();

        ui.on_window_minimize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.window().set_minimized(true);
            }
        });
    }

    // Window maximize/restore
    {
        let ui_weak = ui_weak.clone();

        ui.on_window_maximize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let maximized = ui.window().is_maximized();
                ui.window().set_maximized(!maximized);
            }
        });
    }

    // Window close
    {
        let ui_weak = ui_weak.clone();

        ui.on_window_close(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let _ = ui.hide();
                let _ = slint::quit_event_loop();
            }
        });
    }

    // Window drag (custom title bar) — use a drag-gesture approach:
    // 1. Let all pointer-down events through to Slint (so toolbar buttons work).
    // 2. Track whether the press was in the draggable titlebar zone.
    // 3. On cursor move, if the mouse has moved beyond a threshold, dispatch a
    //    synthetic pointer-release to Slint (clearing its grab state) and then
    //    call drag_window() to hand off to the compositor.
    // 4. Double-click in the drag zone toggles maximize/restore.
    {
        use slint::winit_030::{EventResult, WinitWindowAccessor, winit};
        use std::cell::Cell;

        // Physical pixel position of the last cursor move
        let last_cursor = Rc::new(Cell::new((0.0f64, 0.0f64)));
        let modifiers = Rc::new(Cell::new(winit::keyboard::ModifiersState::default()));
        // Physical pixel position where the left button was pressed (None = not in drag zone)
        let drag_press: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        // Double-click tracking
        let last_press_time = Rc::new(Cell::new(std::time::Instant::now()));
        let dbl_count = Rc::new(Cell::new(0u32));

        ui.window().on_winit_window_event({
            let last_cursor = Rc::clone(&last_cursor);
            let modifiers = Rc::clone(&modifiers);
            let drag_press = Rc::clone(&drag_press);
            let last_press_time = Rc::clone(&last_press_time);
            let dbl_count = Rc::clone(&dbl_count);
            let ui_weak = ui_weak.clone();

            move |slint_window, event| {
                match event {
                    winit::event::WindowEvent::ModifiersChanged(next_modifiers) => {
                        modifiers.set(next_modifiers.state());
                        EventResult::Propagate
                    }

                    winit::event::WindowEvent::KeyboardInput { event, .. } => {
                        let modifier_state = modifiers.get();
                        let close_app_shortcut = event.state == winit::event::ElementState::Pressed
                            && !event.repeat
                            && modifier_state.control_key()
                            && modifier_state.shift_key()
                            && matches!(
                                event.physical_key,
                                winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW)
                            );

                        if close_app_shortcut {
                            if let Some(ui) = ui_weak.upgrade() {
                                let _ = ui.hide();
                                let _ = slint::quit_event_loop();
                            }
                            return EventResult::PreventDefault;
                        }

                        EventResult::Propagate
                    }

                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        last_cursor.set((position.x, position.y));

                        // Check drag threshold (5 physical pixels)
                        if let Some((px, py)) = drag_press.get() {
                            let dx = position.x - px;
                            let dy = position.y - py;
                            if dx * dx + dy * dy > 25.0 {
                                drag_press.set(None);

                                // Dispatch synthetic pointer-release so Slint clears
                                // any pressed/grabbed state from the original click.
                                let scale = slint_window.scale_factor();
                                let lx = (px / scale as f64) as f32;
                                let ly = (py / scale as f64) as f32;
                                let _ = slint_window.try_dispatch_event(
                                    slint::platform::WindowEvent::PointerReleased {
                                        position: slint::LogicalPosition::new(lx, ly),
                                        button: slint::platform::PointerEventButton::Left,
                                    },
                                );

                                // Now hand off to compositor for native window move
                                slint_window.with_winit_window(|w| {
                                    let _ = w.drag_window();
                                });
                                return EventResult::PreventDefault;
                            }
                        }
                        EventResult::Propagate
                    }

                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        let (cx, cy) = last_cursor.get();
                        let scale = slint_window.scale_factor() as f64;
                        let ly = cy / scale;
                        let lx = cx / scale;

                        let toolbar_height = 40.0;
                        let win_w = slint_window.size().width as f64 / scale;
                        let buttons_right_start = win_w - 138.0;

                        if ly < toolbar_height && lx < buttons_right_start {
                            // Press is in the draggable toolbar zone — record it,
                            // but let Slint see the event so buttons still work.
                            drag_press.set(Some((cx, cy)));

                            // Double-click detection
                            let now = std::time::Instant::now();
                            if now.duration_since(last_press_time.get()).as_millis() < 400 {
                                dbl_count.set(dbl_count.get() + 1);
                            } else {
                                dbl_count.set(1);
                            }
                            last_press_time.set(now);

                            if dbl_count.get() >= 2 {
                                dbl_count.set(0);
                                drag_press.set(None);
                                let maximized = slint_window.is_maximized();
                                slint_window.set_maximized(!maximized);
                                return EventResult::PreventDefault;
                            }
                        }
                        EventResult::Propagate
                    }

                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Released,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        drag_press.set(None);
                        EventResult::Propagate
                    }

                    _ => EventResult::Propagate,
                }
            }
        });
    }
}

fn open_file(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
) {
    dbg_print!("[OPEN] Opening file: {}", path.display());
    ui.set_is_loading(true);
    ui.set_status_text(SharedString::from(format!("Opening {}...", path.display())));

    match WsiFile::open(&path) {
        Ok(wsi) => {
            dbg_print!("[OPEN] File opened successfully");
            let id = {
                let mut state_guard = state.write();

                // If current active tab is a home tab, close it
                if let Some(active_id) = state_guard.active_file_id
                    && state_guard.is_home_tab(active_id)
                {
                    state_guard.close_home_tab(active_id);
                }

                // Get viewport size from the focused pane (use reasonable defaults if not yet laid out)
                let pane_count = state_guard.panes.len().max(1) as f64;
                let pane_gap = 6.0;
                let ui_width = ((ui.get_content_area_width() as f64)
                    - pane_gap * (pane_count - 1.0))
                    / pane_count;
                let ui_height = ui.get_content_area_height() as f64 - 35.0;
                let viewport_width = if ui_width > 0.0 { ui_width } else { 1024.0 };
                let viewport_height = if ui_height > 0.0 { ui_height } else { 768.0 };

                let props = wsi.properties();
                let viewport = ViewportState::new(
                    viewport_width,
                    viewport_height,
                    props.width as f64,
                    props.height as f64,
                );

                // Give the tile manager its own OpenSlide handle so tile loading
                // never contends with the UI's metadata handle.
                let tile_manager_wsi = match wsi.reopen() {
                    Ok(tile_manager_wsi) => tile_manager_wsi,
                    Err(err) => {
                        error!("Failed to open dedicated tile-manager handle: {}", err);
                        ui.set_is_loading(false);
                        ui.set_status_text(SharedString::from(format!("Error: {}", err)));
                        return;
                    }
                };
                let tile_manager = Arc::new(TileManager::new(tile_manager_wsi));

                // Create background tile loader (tiles are loaded on-demand)
                let tile_loader =
                    TileLoader::new(Arc::clone(&tile_manager), Arc::clone(tile_cache));

                // Start loading tiles immediately using the initial viewport bounds
                // This ensures tiles begin loading before the first render
                let bounds = viewport.viewport.bounds();
                let best_level =
                    wsi.best_level_for_downsample(viewport.viewport.effective_downsample());
                let initial_tiles = calculate_wanted_tiles(
                    &tile_manager,
                    best_level,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                    bounds.bottom,
                    1,
                );
                tile_loader.set_wanted_tiles(initial_tiles);

                // Generate small thumbnail for minimap (lazy - only reads what's needed)
                let thumbnail = generate_thumbnail(&wsi, 150);

                state_guard.add_file(
                    path.clone(),
                    wsi,
                    tile_manager,
                    tile_loader,
                    viewport,
                    thumbnail,
                )
            };

            let level_count = {
                let state_guard = state.read();
                update_tabs(ui, &state_guard);
                update_recent_files(ui, &state_guard);
                state_guard
                    .get_file(id)
                    .map(|f| f.wsi.level_count())
                    .unwrap_or(0)
            };

            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!(
                "Opened {} ({} levels)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                level_count
            )));

            info!("Successfully opened file with {} levels", level_count);

            request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
        }
        Err(e) => {
            error!("Failed to open file: {}", e);
            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!("Error: {}", e)));
        }
    }
}

fn generate_thumbnail(wsi: &WsiFile, max_size: u32) -> Option<Vec<u8>> {
    // Use the lowest resolution level for thumbnail
    let level = wsi.level_count().saturating_sub(1);
    let level_info = wsi.level(level)?;

    // Calculate thumbnail dimensions maintaining aspect ratio
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (thumb_w, thumb_h) = if aspect > 1.0 {
        (max_size, (max_size as f64 / aspect).max(1.0) as u32)
    } else {
        ((max_size as f64 * aspect).max(1.0) as u32, max_size)
    };

    // If level is small enough, read it directly
    if level_info.width <= max_size as u64 * 2 && level_info.height <= max_size as u64 * 2 {
        match wsi.read_region(
            0,
            0,
            level,
            level_info.width as u32,
            level_info.height as u32,
        ) {
            Ok(data) => {
                if level_info.width <= max_size as u64 && level_info.height <= max_size as u64 {
                    return Some(data);
                }
                // Resize to thumbnail size
                if let Some(img) = image::RgbaImage::from_raw(
                    level_info.width as u32,
                    level_info.height as u32,
                    data,
                ) {
                    let resized = image::imageops::resize(
                        &img,
                        thumb_w,
                        thumb_h,
                        image::imageops::FilterType::Triangle,
                    );
                    return Some(resized.into_raw());
                }
            }
            Err(e) => {
                warn!("Failed to generate thumbnail: {}", e);
            }
        }
    }

    // Level is too large - read the full level anyway (it's still the smallest level)
    // and resize down to thumbnail size
    match wsi.read_region(
        0,
        0,
        level,
        level_info.width as u32,
        level_info.height as u32,
    ) {
        Ok(data) => {
            if let Some(img) =
                image::RgbaImage::from_raw(level_info.width as u32, level_info.height as u32, data)
            {
                let resized = image::imageops::resize(
                    &img,
                    thumb_w,
                    thumb_h,
                    image::imageops::FilterType::Triangle,
                );
                Some(resized.into_raw())
            } else {
                None
            }
        }
        Err(e) => {
            warn!("Failed to generate thumbnail: {}", e);
            None
        }
    }
}

fn update_tabs(ui: &AppWindow, state: &AppState) {
    let build_pane_tabs = |pane: PaneId| {
        state
            .tabs_for_pane(pane)
            .iter()
            .filter_map(|&id| {
                if state.is_home_tab(id) {
                    Some(TabData {
                        id,
                        title: SharedString::from("Home"),
                        path: SharedString::new(),
                        is_modified: false,
                        is_active: Some(id) == state.active_tab_id_for_pane(pane),
                        is_home: true,
                    })
                } else {
                    state.get_file(id).map(|file| TabData {
                        id: file.id,
                        title: SharedString::from(file.filename.clone()),
                        path: SharedString::from(file.path.display().to_string()),
                        is_modified: false,
                        is_active: Some(file.id) == state.active_tab_id_for_pane(pane),
                        is_home: false,
                    })
                }
            })
            .collect::<Vec<_>>()
    };

    let pane_models = with_pane_render_cache(state.panes.len(), |cache| {
        with_pane_ui_models(state.panes.len(), |pane_ui_models| {
            state
                .panes
                .iter()
                .enumerate()
                .map(|(pane_index, _)| {
                    let pane = PaneId(pane_index);
                    let tabs = build_pane_tabs(pane);
                    let (roi_rect, measurements, candidate_measurement) =
                        pane_overlay_data(state, pane);
                    let mut viewport_info = hidden_viewport_info();
                    let mut minimap_rect = full_minimap_rect();
                    let mut zoom_slider_position = 0.5;

                    if let Some(file_id) = state.active_file_id_for_pane(pane)
                        && let Some(file) = state.get_file(file_id)
                        && let Some(viewport_state) = pane_viewport_state(file, pane)
                    {
                        let vp = &viewport_state.viewport;
                        viewport_info = ViewportInfo {
                            center_x: vp.center.x as f32,
                            center_y: vp.center.y as f32,
                            zoom: vp.zoom as f32,
                            image_width: vp.image_width as f32,
                            image_height: vp.image_height as f32,
                            level: file
                                .wsi
                                .best_level_for_downsample(vp.effective_downsample())
                                as i32,
                        };
                        let rect = vp.minimap_rect();
                        minimap_rect = MinimapRect {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                        };
                        zoom_slider_position = zoom_to_slider_value(vp.zoom);
                    }

                    let pane_models = &pane_ui_models[pane_index];
                    if !model_matches(&pane_models.tabs, &tabs) {
                        pane_models.tabs.set_vec(tabs.clone());
                    }
                    if !model_matches(&pane_models.measurements, &measurements) {
                        pane_models.measurements.set_vec(measurements.clone());
                    }

                    let cached = cache.get(pane_index).cloned().unwrap_or_default();
                    PaneViewData {
                        id: pane.as_index(),
                        tabs: pane_models.tabs.clone().into(),
                        content: cached.content.unwrap_or_default(),
                        viewport_info,
                        minimap_thumbnail: cached.minimap_thumbnail.unwrap_or_default(),
                        minimap_rect,
                        is_home_tab: state.is_home_tab_active_in_pane(pane),
                        zoom_slider_position,
                        roi_rect,
                        measurements: pane_models.measurements.clone().into(),
                        candidate_measurement,
                        is_loading: ui.get_is_loading() && pane == state.focused_pane,
                    }
                })
                .collect::<Vec<_>>()
        })
    });

    with_pane_view_model(|pane_model| {
        let mut row_count = pane_model.row_count();
        while row_count > pane_models.len() {
            row_count -= 1;
            pane_model.remove(row_count);
        }
        for (index, pane_data) in pane_models.into_iter().enumerate() {
            if index < row_count {
                let should_update = pane_model
                    .row_data(index)
                    .map(|existing| pane_view_data_changed(&existing, &pane_data))
                    .unwrap_or(true);
                if should_update {
                    pane_model.set_row_data(index, pane_data);
                }
            } else {
                pane_model.push(pane_data);
            }
        }
        ui.set_panes(pane_model.clone().into());
    });
    ui.set_split_enabled(state.panes.len() > 1);
    ui.set_focused_pane(state.focused_pane.as_index());
}

/// Update the recent files list in the UI
fn update_recent_files(ui: &AppWindow, state: &AppState) {
    let recent: Vec<RecentFileData> = state
        .recent_files
        .iter()
        .take(5) // Show at most 5 in the UI
        .map(|f| RecentFileData {
            path: SharedString::from(f.path.display().to_string()),
            name: SharedString::from(f.name.clone()),
        })
        .collect();

    let model = Rc::new(VecModel::from(recent));
    ui.set_recent_files(model.into());
}

fn build_recent_menu_items(state: &AppState) -> Vec<ContextMenuItem> {
    if state.recent_files.is_empty() {
        return vec![ContextMenuItem {
            id: SharedString::from("recent-empty"),
            label: SharedString::from("No recent files"),
            shortcut: SharedString::new(),
            enabled: false,
            separator_after: false,
        }];
    }

    state
        .recent_files
        .iter()
        .take(10)
        .map(|file| ContextMenuItem {
            id: SharedString::from(format!("recent-file:{}", file.path.display())),
            label: SharedString::from(file.name.clone()),
            shortcut: SharedString::new(),
            enabled: true,
            separator_after: false,
        })
        .collect()
}

fn update_render_backend(ui: &AppWindow, state: &AppState) {
    ui.set_render_mode(ui_render_mode(state.render_backend));
    ui.set_gpu_rendering_available(state.gpu_backend_available);
}

fn tile_request_signature(
    tile_manager: &TileManager,
    viewport: &Viewport,
    level: u32,
    margin_tiles: i32,
) -> Option<TileRequestSignature> {
    let level_info = tile_manager.wsi().level(level)?;
    let bounds = viewport.bounds();
    let tile_size = tile_manager.tile_size_for_level(level);
    let tile_size_f64 = tile_size as f64;
    let level_left = bounds.left / level_info.downsample;
    let level_top = bounds.top / level_info.downsample;
    let level_right = bounds.right / level_info.downsample;
    let level_bottom = bounds.bottom / level_info.downsample;
    let margin_tiles_i64 = margin_tiles.max(0) as i64;

    Some(TileRequestSignature {
        level,
        margin_tiles,
        start_x: ((level_left / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        start_y: ((level_top / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        end_x: ((level_right / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_x(tile_size)),
        end_y: ((level_bottom / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_y(tile_size)),
        tile_size,
    })
}

fn thumbnail_image_for_file(file: &OpenFile) -> Option<Image> {
    let Some(ref thumb_data) = file.thumbnail else {
        return None;
    };

    let level = file.wsi.level_count().saturating_sub(1);
    let Some(level_info) = file.wsi.level(level) else {
        return None;
    };
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (width, height) = if aspect > 1.0 {
        (150u32, (150.0 / aspect) as u32)
    } else {
        ((150.0 * aspect) as u32, 150u32)
    };

    create_image_buffer(thumb_data, width.max(1), height.max(1)).map(Image::from_rgba8)
}

fn update_and_render(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let mut state = state.write();
    let render_backend = state.render_backend;
    let pane_count = state.panes.len();
    let active_file_ids: Vec<Option<i32>> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| state.active_file_id_for_pane(PaneId(pane_index)))
        .collect();
    if active_file_ids.iter().all(Option::is_none) {
        for pane_index in 0..pane_count {
            clear_cached_pane(PaneId(pane_index));
            state.set_last_rendered_file_id(PaneId(pane_index), None);
        }
        update_tabs(ui, &state);
        return false;
    }

    let render_requested = std::mem::take(&mut state.needs_render);

    // Update ant offset for marching ants animation (0.5 pixels per frame at 60fps = 30 pixels/sec)
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;

    let content_width = (ui.get_content_area_width() as f64).max(100.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(100.0);
    let pane_gap = 6.0;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(100.0);

    let mut wanted_tiles_by_file: HashMap<i32, HashSet<common::TileCoord>> = HashMap::new();
    for (pane_index, file_id) in active_file_ids.iter().copied().enumerate() {
        let Some(file_id) = file_id else {
            continue;
        };
        let pane = PaneId(pane_index);
        let Some(file_index) = state.open_files.iter().position(|f| f.id == file_id) else {
            continue;
        };

        let Some((viewport_state, last_request)) = ({
            let file = &mut state.open_files[file_index];
            match file.pane_state_mut(pane) {
                Some(pane_state) => {
                    pane_state.viewport.set_size(pane_width, content_height);
                    Some((pane_state.viewport.clone(), pane_state.last_request))
                }
                None => None,
            }
        }) else {
            continue;
        };

        let vp = &viewport_state.viewport;
        let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
        let request = {
            let file = &state.open_files[file_index];
            let trilinear =
                render::calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
            let level = trilinear.level_fine;
            tile_request_signature(&file.tile_manager, vp, level, margin_tiles).map(|signature| {
                let wanted = calculate_wanted_tiles(
                    &file.tile_manager,
                    level,
                    vp.bounds().left,
                    vp.bounds().top,
                    vp.bounds().right,
                    vp.bounds().bottom,
                    margin_tiles,
                );
                (signature, wanted)
            })
        };
        if let Some((signature, wanted)) = request {
            if last_request != Some(signature) {
                if let Some(file) = state.open_files.get_mut(file_index) {
                    if let Some(pane_state) = file.pane_state_mut(pane) {
                        pane_state.last_request = Some(signature);
                    }
                }
            }
            wanted_tiles_by_file
                .entry(file_id)
                .or_default()
                .extend(wanted);
        }
    }

    for (file_id, wanted_tiles) in wanted_tiles_by_file {
        if let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) {
            file.tile_loader.set_wanted_tiles(wanted_tiles.into_iter().collect());
        }
    }

    let mut keep_running = render_requested;
    let mut rendered_frame = false;

    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let file_switched = state.last_rendered_file_id(pane) != file_id;
        keep_running |= file_switched;

        let Some(file_id) = file_id else {
            clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
            clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let minimap_missing = with_pane_render_cache(pane_count, |cache| {
            cache.get(pane.0)
                .and_then(|entry| entry.minimap_thumbnail.as_ref())
                .is_none()
        });
        let content_missing = with_pane_render_cache(pane_count, |cache| {
            cache.get(pane.0)
                .and_then(|entry| entry.content.as_ref())
                .is_none()
        });

        if file_switched {
            if let Some(pane_state) = file.pane_state_mut(pane) {
                pane_state.frame_count = 0;
            }
        }
        if file_switched || minimap_missing {
            set_cached_pane_minimap(pane, thumbnail_image_for_file(file));
        }

        let outcome = render_pane_to_image(
            ui,
            file,
            pane,
            tile_cache,
            pane_width,
            content_height,
            file_switched,
            render_backend,
        );
        keep_running |= outcome.keep_running;
        rendered_frame |= outcome.rendered;
        if let Some(image) = outcome.image {
            set_cached_pane_content(pane, image);
            state.set_last_rendered_file_id(pane, Some(file_id));
        } else if content_missing {
            state.set_last_rendered_file_id(pane, None);
            keep_running = true;
        } else {
            state.set_last_rendered_file_id(pane, Some(file_id));
        }
    }

    update_tabs(ui, &state);
    keep_running |= has_active_roi_overlay(&state);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

fn render_pane_to_image(
    ui: &AppWindow,
    file: &mut OpenFile,
    pane: PaneId,
    tile_cache: &Arc<TileCache>,
    target_width: f64,
    target_height: f64,
    force_render: bool,
    render_backend: RenderBackend,
) -> PaneRenderOutcome {
    let (
        animating,
        viewport_state,
        frame_count,
        last_render_zoom,
        last_render_center_x,
        last_render_center_y,
        last_render_width,
        last_render_height,
        last_render_level,
        previous_tiles_loaded,
        last_seen_tile_epoch,
    ) = {
        let Some(pane_state) = file.pane_state_mut(pane) else {
            return PaneRenderOutcome::default();
        };

        let animating = pane_state.viewport.update();
        pane_state.viewport.set_size(target_width, target_height);
        (
            animating,
            pane_state.viewport.clone(),
            pane_state.frame_count,
            pane_state.last_render_zoom,
            pane_state.last_render_center_x,
            pane_state.last_render_center_y,
            pane_state.last_render_width,
            pane_state.last_render_height,
            pane_state.last_render_level,
            pane_state.tiles_loaded_since_render,
            pane_state.last_seen_tile_epoch,
        )
    };

    let vp = &viewport_state.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = frame_count == 0;
    let viewport_changed = (last_render_zoom - vp_zoom).abs() > 0.001
        || (last_render_center_x - vp_center_x).abs() > 1.0
        || (last_render_center_y - vp_center_y).abs() > 1.0
        || (last_render_width - vp_width).abs() > 1.0
        || (last_render_height - vp_height).abs() > 1.0;

    let trilinear = render::calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine;
    let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
    let level_changed = level != last_render_level;

    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    let cached_tiles: Vec<_> = visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.get(coord).map(|data| (*coord, data)))
        .collect();
    let cached_count = cached_tiles.len() as u32;

    let cached_coarse_tiles: Vec<_> = if trilinear.level_fine != trilinear.level_coarse && trilinear.blend > 0.01 {
        file.tile_manager
            .visible_tiles_with_margin(
                trilinear.level_coarse,
                bounds.left,
                bounds.top,
                bounds.right,
                bounds.bottom,
                margin_tiles,
            )
            .into_iter()
            .filter_map(|coord| tile_cache.get(&coord).map(|data| (coord, data)))
            .collect()
    } else {
        Vec::new()
    };

    let loaded_tile_epoch = file.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > last_seen_tile_epoch;
    let new_tiles_loaded = cached_count > if level_changed { 0 } else { previous_tiles_loaded }
        || !cached_coarse_tiles.is_empty()
        || tile_epoch_advanced;
    let tiles_pending = file.tile_loader.pending_count() > 0;

    let keep_running = animating || new_tiles_loaded || tiles_pending;
    if !force_render && !is_first_frame && !viewport_changed && !level_changed && !new_tiles_loaded {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.frame_count += 1;
        pane_state.last_render_time = std::time::Instant::now();
        pane_state.last_render_zoom = vp_zoom;
        pane_state.last_render_center_x = vp_center_x;
        pane_state.last_render_center_y = vp_center_y;
        pane_state.last_render_width = vp_width;
        pane_state.last_render_height = vp_height;
        pane_state.last_render_level = level;
        pane_state.tiles_loaded_since_render = cached_count;
        pane_state.last_seen_tile_epoch = loaded_tile_epoch;
    }

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if render_backend == RenderBackend::Gpu {
        let draws = collect_tile_draws(file, tile_cache, vp, trilinear);
        let image = with_gpu_renderer(|renderer| {
            renderer
                .borrow_mut()
                .queue_frame(SurfaceSlot(pane.0), render_width, render_height, draws)
        })
        .flatten();

        if image.is_some() {
            ui.window().request_redraw();
        } else if let Some(pane_state) = file.pane_state_mut(pane) {
            // The first render attempt can happen before Slint has finished WGPU setup.
            // Keep this pane in a first-frame state so the idle-frame optimization retries.
            pane_state.frame_count = 0;
            pane_state.last_render_level = u32::MAX;
            pane_state.tiles_loaded_since_render = 0;
        }
        let rendered = image.is_some();

        return PaneRenderOutcome {
            image,
            keep_running: keep_running || !rendered,
            rendered,
        };
    }

    let level_info = match file.wsi.level(level) {
        Some(info) => info.clone(),
        None => {
            return PaneRenderOutcome {
                image: None,
                keep_running,
                rendered: false,
            };
        }
    };

    let level_count = file.wsi.level_count();
    let mut fallback_blits = Vec::new();
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level).cloned() else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };

            let fb_image_x =
                fb_coord.x as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_y =
                fb_coord.y as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_x_end =
                fb_image_x + fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_y_end =
                fb_image_y + fallback_tile.height as f64 * fallback_level_info.downsample;

            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((fb_image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((fb_image_y_end - bounds.top) * vp.zoom).round() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            fallback_blits.push((
                fallback_tile,
                screen_x,
                screen_y,
                screen_w,
                screen_h,
            ));
        }
    }

    let Some(pane_state) = file.pane_state_mut(pane) else {
        return PaneRenderOutcome::default();
    };
    let buffer_size = (render_width * render_height * 4) as usize;
    if pane_state.render_buffer.len() != buffer_size {
        pane_state.render_buffer.resize(buffer_size, 0);
    }
    fast_fill_rgba(&mut pane_state.render_buffer, 30, 30, 30, 255);
    let buffer = &mut pane_state.render_buffer[..];

    for (fallback_tile, screen_x, screen_y, screen_w, screen_h) in fallback_blits {
        blit_tile(
            buffer,
            render_width,
            render_height,
            &fallback_tile.data,
            fallback_tile.width,
            fallback_tile.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }

    for (coord, tile_data) in cached_tiles.iter() {
        let image_x = coord.x as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_y = coord.y as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_x_end = image_x + tile_data.width as f64 * level_info.downsample;
        let image_y_end = image_y + tile_data.height as f64 * level_info.downsample;
        let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
        let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
        let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
        let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;
        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;

        blit_tile(
            buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }

    PaneRenderOutcome {
        image: create_image_buffer(buffer, render_width, render_height).map(Image::from_rgba8),
        keep_running,
        rendered: true,
    }
}

fn render_viewport_to_buffer(
    ui: &AppWindow,
    file: &mut OpenFile,
    tile_cache: &Arc<TileCache>,
    _is_primary: bool,
) -> bool {
    dbg_print!("[RENDER] start");
    let vp = &file.viewport.viewport;

    // Copy viewport values for dirty tracking
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;

    trace!(
        "render: viewport {}x{} zoom={}",
        vp_width, vp_height, vp_zoom
    );

    // Is this the first frame? (must render at least once)
    let is_first_frame = file.frame_count == 0;

    // Check if viewport changed since last render
    // Use larger thresholds to avoid rendering for imperceptible changes
    let viewport_changed = (file.last_render_zoom - vp_zoom).abs() > 0.001
        || (file.last_render_center_x - vp_center_x).abs() > 1.0
        || (file.last_render_center_y - vp_center_y).abs() > 1.0
        || (file.last_render_width - vp_width).abs() > 1.0
        || (file.last_render_height - vp_height).abs() > 1.0;

    // Calculate trilinear levels for smooth mip transitions
    let trilinear = render::calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine; // Primary level for rendering
    let tile_size = file.tile_manager.tile_size_for_level(level);
    let margin_tiles = if file.viewport.is_moving() { 1 } else { 0 };

    // Check if level changed - must reset tile tracking and force re-render
    let level_changed = level != file.last_render_level;
    if level_changed {
        // Reset tile count since we're now tracking a different set of tiles
        file.tiles_loaded_since_render = 0;
    }

    trace!(
        "render: trilinear levels fine={} coarse={} blend={:.3}",
        trilinear.level_fine, trilinear.level_coarse, trilinear.blend
    );

    // Get visible tiles using viewport bounds
    let bounds = vp.bounds();
    trace!(
        "render: bounds left={} top={} right={} bottom={}",
        bounds.left, bounds.top, bounds.right, bounds.bottom
    );

    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );

    trace!("render: visible_tiles count={}", visible_tiles.len());

    // Limit visible tiles to a reasonable maximum (typical viewport needs ~200-500 tiles when zoomed in)
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();

    // Pre-fetch all visible tiles from cache NOW to avoid race conditions
    // Holding Arc references prevents eviction during render
    let cached_tiles: Vec<_> = visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.get(coord).map(|data| (*coord, data)))
        .collect();
    let cached_count = cached_tiles.len() as u32;

    // For trilinear, also fetch coarse level tiles
    let coarse_visible_tiles =
        if trilinear.level_fine != trilinear.level_coarse && trilinear.blend > 0.01 {
            file.tile_manager.visible_tiles_with_margin(
                trilinear.level_coarse,
                bounds.left,
                bounds.top,
                bounds.right,
                bounds.bottom,
                margin_tiles,
            )
        } else {
            Vec::new()
        };

    let cached_coarse_tiles: Vec<_> = coarse_visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.get(coord).map(|data| (*coord, data)))
        .collect();

    let new_tiles_loaded =
        cached_count > file.tiles_loaded_since_render || !cached_coarse_tiles.is_empty();

    trace!(
        "render: cached fine={} coarse={} new_tiles={}",
        cached_count,
        cached_coarse_tiles.len(),
        new_tiles_loaded
    );

    if let Some(signature) = tile_request_signature(&file.tile_manager, vp, level, margin_tiles)
        && file.last_primary_request != Some(signature)
    {
        let wanted = calculate_wanted_tiles(
            &file.tile_manager,
            level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        );

        trace!("render: wanted tiles={}", wanted.len());
        file.tile_loader.set_wanted_tiles(wanted);
        file.last_primary_request = Some(signature);
        trace!("render: set_wanted_tiles done");
    }

    // Skip rendering if nothing changed
    // - Always render first frame
    // - Always render if viewport changed (pan/zoom)
    // - Always render if level changed (different tiles needed)
    // - Always render if new tiles loaded at current level
    if !is_first_frame && !viewport_changed && !level_changed && !new_tiles_loaded {
        trace!("render: skipping (no changes)");
        return false;
    }

    dbg_print!("[RENDER] proceeding with render");

    // Update tracking state
    file.frame_count += 1;
    file.last_render_time = std::time::Instant::now();
    file.last_render_zoom = vp_zoom;
    file.last_render_center_x = vp_center_x;
    file.last_render_center_y = vp_center_y;

    dbg_print!("[RENDER] tracking state updated");
    file.last_render_width = vp_width;
    file.last_render_height = vp_height;
    file.last_render_level = level;
    file.tiles_loaded_since_render = cached_count;

    // Use persistent buffer for rendering
    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;

    dbg_print!("[RENDER] buffer {}x{}", render_width, render_height);

    if render_width == 0 || render_height == 0 {
        return false;
    }

    let buffer_size = (render_width * render_height * 4) as usize;
    dbg_print!("[RENDER] buffer_size={}", buffer_size);

    // Resize buffer if needed (only reallocates if capacity insufficient)
    if file.render_buffer.len() != buffer_size {
        dbg_print!("[RENDER] resizing buffer to {}", buffer_size);
        file.render_buffer.resize(buffer_size, 0);
    }

    dbg_print!("[RENDER] clearing buffer len={}", file.render_buffer.len());

    // Fast clear to dark background using u32 writes (RGBA = 30,30,30,255 = 0xFF1E1E1E in little-endian)
    fast_fill_rgba(&mut file.render_buffer, 30, 30, 30, 255);

    dbg_print!("[RENDER] buffer cleared");

    let buffer = &mut file.render_buffer[..];
    dbg_print!("[RENDER] got buffer slice");

    let level_info = match file.wsi.level(level) {
        Some(info) => info,
        None => return false,
    };
    dbg_print!(
        "[RENDER] got level_info: downsample={}",
        level_info.downsample
    );

    // First pass: render lower-resolution fallback tiles
    // These provide immediate visual feedback while high-res tiles load
    // We render ALL cached low-res tiles that overlap the viewport, then high-res tiles overdraw
    let mut _fallbacks_rendered = 0;
    let level_count = file.wsi.level_count();

    // Iterate from lowest resolution (highest level index) to one above current
    // This ensures higher quality fallbacks draw on top of lower quality ones
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue; // Skip current and higher resolution levels
        }

        let fallback_level_info = match file.wsi.level(fallback_level) {
            Some(info) => info,
            None => continue,
        };

        // Get visible tiles at this fallback level
        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };

            // Calculate image coordinates for this fallback tile
            let fb_image_x =
                fb_coord.x as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_y =
                fb_coord.y as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_x_end =
                fb_image_x + fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_y_end =
                fb_image_y + fallback_tile.height as f64 * fallback_level_info.downsample;

            // Convert to screen coordinates using round() for consistent boundaries
            // Using round() ensures adjacent tiles share exact boundaries (no gaps/overlaps)
            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((fb_image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((fb_image_y_end - bounds.top) * vp.zoom).round() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            // Render the entire fallback tile at its correct screen position
            blit_tile(
                buffer,
                render_width,
                render_height,
                &fallback_tile.data,
                fallback_tile.width,
                fallback_tile.height,
                screen_x,
                screen_y,
                screen_w,
                screen_h,
            );
            _fallbacks_rendered += 1;
        }
    }

    dbg_print!(
        "[RENDER] first pass: rendered {} fallbacks",
        _fallbacks_rendered
    );

    dbg_print!(
        "[RENDER] starting second pass (high-res tiles), count={}",
        cached_tiles.len()
    );

    // Trilinear blending disabled - causes pixel corruption at level boundaries
    // TODO: Fix trilinear edge cases before re-enabling
    let _coarse_level_info = file.wsi.level(trilinear.level_coarse);
    let do_trilinear = false;

    dbg_print!(
        "[RENDER] trilinear: blend={:.3} do_trilinear={}",
        trilinear.blend,
        do_trilinear
    );

    // Second pass: render high-res tiles with optional trilinear blending
    // Use pre-fetched tiles to avoid race conditions with cache eviction
    let mut _tiles_blitted = 0;
    for (coord, tile_data) in cached_tiles.iter() {
        dbg_print!("[RENDER] tile {:?}", coord);

        // Calculate screen position based on image coordinates
        // Each tile at (coord.x, coord.y) covers [coord.x * tile_size * ds, (coord.x+1) * tile_size * ds) in image space
        let image_x = coord.x as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_y = coord.y as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_x_end = image_x + tile_data.width as f64 * level_info.downsample;
        let image_y_end = image_y + tile_data.height as f64 * level_info.downsample;

        // Convert to screen coordinates using round() for consistent boundaries
        // Using round() ensures adjacent tiles share exact boundaries (no gaps/overlaps)
        let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
        let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
        let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
        let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;

        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;

        dbg_print!(
            "[RENDER] blit tile {:?}: screen=({},{}) size={}x{}",
            coord,
            screen_x,
            screen_y,
            screen_w,
            screen_h
        );

        // Trilinear is disabled for now - just use bilinear
        // Try trilinear blending if enabled
        if do_trilinear {
            let coarse_info = _coarse_level_info.unwrap();

            // Find the fine tile's position in image coordinates (level 0)
            let fine_image_w = tile_data.width as f64 * level_info.downsample;
            let fine_image_h = tile_data.height as f64 * level_info.downsample;

            // Convert to coarse level coordinates
            let coarse_tile_x = image_x / coarse_info.downsample;
            let coarse_tile_y = image_y / coarse_info.downsample;

            // Find which coarse tile contains this region
            let coarse_tile_idx_x = (coarse_tile_x / tile_size as f64).floor() as u64;
            let coarse_tile_idx_y = (coarse_tile_y / tile_size as f64).floor() as u64;

            // Look for the coarse tile in our cached tiles
            let coarse_tile = cached_coarse_tiles.iter().find(|(c, _)| {
                c.level == trilinear.level_coarse
                    && c.x == coarse_tile_idx_x
                    && c.y == coarse_tile_idx_y
            });

            if let Some((coarse_coord, coarse_data)) = coarse_tile {
                // Calculate where within the coarse tile our fine region maps to
                let coarse_tile_origin_x = coarse_coord.x as f64 * coarse_coord.tile_size as f64;
                let coarse_tile_origin_y = coarse_coord.y as f64 * coarse_coord.tile_size as f64;

                // Source region within coarse tile
                let coarse_src_x = coarse_tile_x - coarse_tile_origin_x;
                let coarse_src_y = coarse_tile_y - coarse_tile_origin_y;
                let coarse_src_w = fine_image_w / coarse_info.downsample;
                let coarse_src_h = fine_image_h / coarse_info.downsample;

                // Clamp source region to actual coarse tile bounds to prevent stretched edge artifacts
                let coarse_src_x_clamped = coarse_src_x.max(0.0);
                let coarse_src_y_clamped = coarse_src_y.max(0.0);
                let coarse_src_w_clamped =
                    (coarse_src_w).min(coarse_data.width as f64 - coarse_src_x_clamped);
                let coarse_src_h_clamped =
                    (coarse_src_h).min(coarse_data.height as f64 - coarse_src_y_clamped);

                // Only do trilinear if we have a valid coarse region
                if coarse_src_w_clamped > 0.0 && coarse_src_h_clamped > 0.0 {
                    blit_tile_trilinear(
                        buffer,
                        render_width,
                        render_height,
                        &tile_data.data,
                        tile_data.width,
                        tile_data.height,
                        &coarse_data.data,
                        coarse_data.width,
                        coarse_data.height,
                        screen_x,
                        screen_y,
                        screen_w,
                        screen_h,
                        coarse_src_x_clamped,
                        coarse_src_y_clamped,
                        coarse_src_w_clamped,
                        coarse_src_h_clamped,
                        trilinear.blend,
                    );
                    _tiles_blitted += 1;
                    dbg_print!("[RENDER] trilinear blit done");
                    continue;
                }
            }
        }

        // Fallback to standard bilinear blit
        blit_tile(
            buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
        _tiles_blitted += 1;
        dbg_print!("[RENDER] blit done, total={}", _tiles_blitted);
    }

    dbg_print!("[RENDER] blitted {} tiles total", _tiles_blitted);

    dbg_print!("[RENDER] creating image buffer");

    // Create image from buffer and update UI
    if let Some(pixel_buffer) = create_image_buffer(buffer, render_width, render_height) {
        dbg_print!("[RENDER] updating UI");
        ui.set_viewport_content(Image::from_rgba8(pixel_buffer));
        dbg_print!("[RENDER] done");
        return true;
    }

    dbg_print!("[RENDER] done");
    false
}

fn render_viewport_gpu(
    ui: &AppWindow,
    file: &mut OpenFile,
    tile_cache: &Arc<TileCache>,
    slot: SurfaceSlot,
) -> bool {
    let vp = &file.viewport.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = file.frame_count == 0;
    let viewport_changed = (file.last_render_zoom - vp_zoom).abs() > 0.001
        || (file.last_render_center_x - vp_center_x).abs() > 1.0
        || (file.last_render_center_y - vp_center_y).abs() > 1.0
        || (file.last_render_width - vp_width).abs() > 1.0
        || (file.last_render_height - vp_height).abs() > 1.0;

    let trilinear = render::calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine;
    let margin_tiles = if file.viewport.is_moving() { 1 } else { 0 };
    let level_changed = level != file.last_render_level;
    if level_changed {
        file.tiles_loaded_since_render = 0;
    }

    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    let cached_tiles: Vec<_> = visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.get(coord).map(|data| (*coord, data)))
        .collect();
    let coarse_cached_count =
        if trilinear.level_fine != trilinear.level_coarse && trilinear.blend > 0.01 {
            file.tile_manager
                .visible_tiles_with_margin(
                    trilinear.level_coarse,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                    bounds.bottom,
                    margin_tiles,
                )
                .into_iter()
                .filter(|coord| tile_cache.contains(coord))
                .count() as u32
        } else {
            0
        };
    let cached_count = cached_tiles.len() as u32 + coarse_cached_count;
    let new_tiles_loaded = cached_count > file.tiles_loaded_since_render;

    if let Some(signature) = tile_request_signature(&file.tile_manager, vp, level, margin_tiles)
        && file.last_primary_request != Some(signature)
    {
        let wanted = calculate_wanted_tiles(
            &file.tile_manager,
            level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        );
        file.tile_loader.set_wanted_tiles(wanted);
        file.last_primary_request = Some(signature);
    }

    if !is_first_frame && !viewport_changed && !level_changed && !new_tiles_loaded {
        return false;
    }

    file.frame_count += 1;
    file.last_render_time = std::time::Instant::now();
    file.last_render_zoom = vp_zoom;
    file.last_render_center_x = vp_center_x;
    file.last_render_center_y = vp_center_y;
    file.last_render_width = vp_width;
    file.last_render_height = vp_height;
    file.last_render_level = level;
    file.tiles_loaded_since_render = cached_count;

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return false;
    }

    let draws = collect_tile_draws(file, tile_cache, vp, trilinear);
    let image = with_gpu_renderer(|renderer| {
        renderer
            .borrow_mut()
            .queue_frame(slot, render_width, render_height, draws)
    })
    .flatten();

    if let Some(image) = image {
        ui.set_viewport_content(image);
        ui.window().request_redraw();
        true
    } else {
        false
    }
}

fn render_secondary_viewport_gpu(
    ui: &AppWindow,
    file: &mut OpenFile,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let Some(ref secondary) = file.secondary_viewport else {
        return false;
    };

    let vp = &secondary.viewport;
    let trilinear = render::calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine;
    let margin_tiles = if secondary.is_moving() { 1 } else { 0 };
    let bounds = vp.bounds();

    if let Some(signature) = tile_request_signature(&file.tile_manager, vp, level, margin_tiles)
        && file.last_secondary_request != Some(signature)
    {
        let wanted = calculate_wanted_tiles(
            &file.tile_manager,
            level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        );
        file.tile_loader.set_wanted_tiles(wanted);
        file.last_secondary_request = Some(signature);
    }

    let render_width = vp.width as u32;
    let render_height = vp.height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return false;
    }

    let draws = collect_tile_draws(file, tile_cache, vp, trilinear);
    let image = with_gpu_renderer(|renderer| {
        renderer.borrow_mut().queue_frame(
            SurfaceSlot::SECONDARY,
            render_width,
            render_height,
            draws,
        )
    })
    .flatten();

    if let Some(image) = image {
        ui.set_secondary_viewport_content(image);
        ui.window().request_redraw();
        true
    } else {
        false
    }
}

fn collect_tile_draws(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: render::TrilinearLevels,
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    let bounds = vp.bounds();
    let level_count = file.wsi.level_count();

    for fallback_level in (0..level_count).rev() {
        if fallback_level <= trilinear.level_fine {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level) else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for coord in fallback_tiles.iter().take(100) {
            let Some(tile_data) = tile_cache.get(coord) else {
                continue;
            };
            if let Some(draw) = tile_draw_from_tile(
                vp,
                bounds.left,
                bounds.top,
                fallback_level_info.downsample,
                *coord,
                tile_data,
                None,
            ) {
                draws.push(draw);
            }
        }
    }

    let Some(level_info) = file.wsi.level(trilinear.level_fine) else {
        return draws;
    };

    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        trilinear.level_fine,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        0,
    );

    for coord in visible_tiles.iter().take(500) {
        let Some(tile_data) = tile_cache.get(coord) else {
            continue;
        };
        let coarse_blend = coarse_blend_for_tile(
            file,
            tile_cache,
            trilinear,
            level_info.downsample,
            *coord,
            &tile_data,
        );
        if let Some(draw) = tile_draw_from_tile(
            vp,
            bounds.left,
            bounds.top,
            level_info.downsample,
            *coord,
            tile_data,
            coarse_blend,
        ) {
            draws.push(draw);
        }
    }

    draws
}

fn coarse_blend_for_tile(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    trilinear: render::TrilinearLevels,
    fine_downsample: f64,
    fine_coord: common::TileCoord,
    fine_tile: &Arc<common::TileData>,
) -> Option<CoarseBlendData> {
    const COARSE_BOUNDARY_EPSILON: f64 = 1e-3;

    if trilinear.level_fine == trilinear.level_coarse || trilinear.blend <= 0.01 {
        return None;
    }

    let coarse_info = file.wsi.level(trilinear.level_coarse)?;
    let image_x = fine_coord.x as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let image_y = fine_coord.y as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let fine_image_w = fine_tile.width as f64 * fine_downsample;
    let fine_image_h = fine_tile.height as f64 * fine_downsample;
    let coarse_tile_size = file
        .tile_manager
        .tile_size_for_level(trilinear.level_coarse) as f64;
    let image_x_end = image_x + fine_image_w;
    let image_y_end = image_y + fine_image_h;

    let coarse_tile_x = image_x / coarse_info.downsample;
    let coarse_tile_y = image_y / coarse_info.downsample;
    let coarse_tile_x_end =
        ((image_x_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_x);
    let coarse_tile_y_end =
        ((image_y_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_y);
    let coarse_start_tile_x = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_start_tile_y = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_x = (coarse_tile_x_end / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_y = (coarse_tile_y_end / coarse_tile_size).floor().max(0.0) as u64;

    // A single coarse tile is only valid when it fully covers the fine tile's footprint.
    // Clamping a cross-boundary sample region collapses it into a repeated source row/column.
    if coarse_start_tile_x != coarse_end_tile_x || coarse_start_tile_y != coarse_end_tile_y {
        return None;
    }

    let coarse_coord = common::TileCoord::new(
        trilinear.level_coarse,
        coarse_start_tile_x,
        coarse_start_tile_y,
        coarse_tile_size as u32,
    );

    let coarse_tile = tile_cache.get(&coarse_coord)?;
    let coarse_origin_x = coarse_coord.x as f64 * coarse_coord.tile_size as f64;
    let coarse_origin_y = coarse_coord.y as f64 * coarse_coord.tile_size as f64;
    let coarse_src_x = (coarse_tile_x - coarse_origin_x).max(0.0);
    let coarse_src_y = (coarse_tile_y - coarse_origin_y).max(0.0);
    let coarse_src_w = (fine_image_w / coarse_info.downsample).max(0.0);
    let coarse_src_h = (fine_image_h / coarse_info.downsample).max(0.0);
    let coarse_src_x_end = coarse_src_x + coarse_src_w;
    let coarse_src_y_end = coarse_src_y + coarse_src_h;

    if coarse_src_x_end <= coarse_src_x
        || coarse_src_y_end <= coarse_src_y
        || coarse_src_x_end > coarse_tile.width as f64 + COARSE_BOUNDARY_EPSILON
        || coarse_src_y_end > coarse_tile.height as f64 + COARSE_BOUNDARY_EPSILON
    {
        return None;
    }

    let coarse_width = coarse_tile.width as f64;
    let coarse_height = coarse_tile.height as f64;

    Some((
        coarse_tile,
        [
            (coarse_src_x / coarse_width) as f32,
            (coarse_src_y / coarse_height) as f32,
        ],
        [
            (coarse_src_x_end / coarse_width) as f32,
            (coarse_src_y_end / coarse_height) as f32,
        ],
        trilinear.blend as f32,
    ))
}

fn tile_draw_from_tile(
    vp: &Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
    coord: common::TileCoord,
    tile_data: Arc<common::TileData>,
    coarse_blend: Option<CoarseBlendData>,
) -> Option<TileDraw> {
    let image_x = coord.x as f64 * coord.tile_size as f64 * downsample;
    let image_y = coord.y as f64 * coord.tile_size as f64 * downsample;
    let image_x_end = image_x + tile_data.width as f64 * downsample;
    let image_y_end = image_y + tile_data.height as f64 * downsample;

    let screen_x = ((image_x - bounds_left) * vp.zoom).round() as i32;
    let screen_y = ((image_y - bounds_top) * vp.zoom).round() as i32;
    let screen_x_end = ((image_x_end - bounds_left) * vp.zoom).round() as i32;
    let screen_y_end = ((image_y_end - bounds_top) * vp.zoom).round() as i32;
    let screen_w = screen_x_end - screen_x;
    let screen_h = screen_y_end - screen_y;

    if screen_w <= 0 || screen_h <= 0 {
        return None;
    }

    let (coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend) = coarse_blend
        .map(|(coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend)| {
            (Some(coarse_tile), coarse_uv_min, coarse_uv_max, mip_blend)
        })
        .unwrap_or((None, [0.0, 0.0], [1.0, 1.0], 0.0));

    Some(TileDraw {
        tile: tile_data,
        coarse_tile,
        screen_x,
        screen_y,
        screen_w,
        screen_h,
        coarse_uv_min,
        coarse_uv_max,
        mip_blend,
    })
}

fn render_secondary_viewport(
    ui: &AppWindow,
    file: &mut OpenFile,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let Some(ref secondary) = file.secondary_viewport else {
        return false;
    };

    let vp = &secondary.viewport;
    let margin_tiles = if secondary.is_moving() { 1 } else { 0 };

    // Determine best level for current zoom
    let level = file
        .wsi
        .best_level_for_downsample(vp.effective_downsample());
    // Get visible tiles using viewport bounds
    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );

    // Limit visible tiles to a reasonable maximum
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();

    if let Some(signature) = tile_request_signature(&file.tile_manager, vp, level, margin_tiles)
        && file.last_secondary_request != Some(signature)
    {
        let wanted = calculate_wanted_tiles(
            &file.tile_manager,
            level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        );
        file.tile_loader.set_wanted_tiles(wanted);
        file.last_secondary_request = Some(signature);
    }

    // Reuse persistent buffer for rendering (avoid per-frame allocation)
    let render_width = vp.width as u32;
    let render_height = vp.height.max(1.0) as u32;

    if render_width == 0 || render_height == 0 {
        return false;
    }

    let buffer_size = (render_width * render_height * 4) as usize;

    // Resize buffer only if dimensions changed
    if file.secondary_render_buffer.len() != buffer_size {
        file.secondary_render_buffer.resize(buffer_size, 0);
    }

    // Fast clear to dark background
    fast_fill_rgba(&mut file.secondary_render_buffer, 30, 30, 30, 255);

    let buffer = &mut file.secondary_render_buffer;

    let level_info = match file.wsi.level(level) {
        Some(info) => info,
        None => return false,
    };

    let level_count = file.wsi.level_count();

    // First pass: render fallback tiles (whole tiles at lower resolutions)
    // Iterate from lowest resolution to highest, so better quality draws on top
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue; // Skip current and higher resolution levels
        }

        let fallback_level_info = match file.wsi.level(fallback_level) {
            Some(info) => info,
            None => continue,
        };

        // Get visible tiles at this fallback level
        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };

            // Calculate image coordinates for this fallback tile
            let fb_image_x =
                fb_coord.x as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_y =
                fb_coord.y as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_x_end =
                fb_image_x + fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_y_end =
                fb_image_y + fallback_tile.height as f64 * fallback_level_info.downsample;

            // Convert to screen coordinates using round() for consistent boundaries
            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((fb_image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((fb_image_y_end - bounds.top) * vp.zoom).round() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            blit_tile(
                buffer,
                render_width,
                render_height,
                &fallback_tile.data,
                fallback_tile.width,
                fallback_tile.height,
                screen_x,
                screen_y,
                screen_w,
                screen_h,
            );
        }
    }

    // Second pass: render high-res tiles that are available
    for coord in &visible_tiles {
        let Some(tile_data) = tile_cache.get(coord) else {
            continue;
        };

        // Calculate image coordinates for this tile
        let image_x = coord.x as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_y = coord.y as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_x_end = image_x + tile_data.width as f64 * level_info.downsample;
        let image_y_end = image_y + tile_data.height as f64 * level_info.downsample;

        // Convert to screen coordinates using round() for consistent boundaries
        let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
        let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
        let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
        let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;
        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;

        blit_tile(
            buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }

    // Create image from buffer
    if let Some(pixel_buffer) = create_image_buffer(buffer, render_width, render_height) {
        ui.set_secondary_viewport_content(Image::from_rgba8(pixel_buffer));
        return true;
    }

    false
}

/// Fast fill RGBA buffer with a single color using u32 writes
/// This is ~4x faster than byte-by-byte writes on most architectures
#[inline(always)]
fn fast_fill_rgba(buffer: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    // Pack RGBA into u32 (little-endian: ABGR order in memory)
    let pixel = u32::from_ne_bytes([r, g, b, a]);

    // Cast buffer to u32 slice for faster writes
    // SAFETY: Buffer length is always multiple of 4 (RGBA pixels)
    if buffer.len() >= 4 && buffer.len().is_multiple_of(4) {
        let (prefix, pixels, suffix) = unsafe { buffer.align_to_mut::<u32>() };

        // Handle unaligned prefix bytes
        for chunk in prefix.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }

        // Fast fill aligned u32s
        pixels.fill(pixel);

        // Handle unaligned suffix bytes
        for chunk in suffix.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }
    } else {
        // Fallback for small or misaligned buffers
        for chunk in buffer.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }
    }
}

/// Optimized bilinear tile blitter with cache-friendly access patterns
/// Uses fixed-point arithmetic and minimizes bounds checking
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn blit_tile(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dest_x: i32,
    dest_y: i32,
    scaled_width: i32,
    scaled_height: i32,
) {
    if scaled_width <= 0 || scaled_height <= 0 {
        return;
    }

    // Early return if tile is entirely off-screen
    if dest_x + scaled_width <= 0 || dest_y + scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }

    // Calculate visible region (clamp to destination bounds)
    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + scaled_height).min(dest_height as i32) as u32;

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    // Use standard texture mapping: src_width / scaled_width ratio
    // This ensures each source pixel covers exactly (scaled_width / src_width) dest pixels
    // avoiding edge stretching at tile boundaries
    let scale_x_fp = ((src_width as u64) << 16) / (scaled_width as u64).max(1);
    let scale_y_fp = ((src_height as u64) << 16) / (scaled_height as u64).max(1);

    let src_width_minus_1 = src_width.saturating_sub(1);
    let src_height_minus_1 = src_height.saturating_sub(1);
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (src_width * 4) as usize;

    // Pre-validate that source has enough data for any valid sample
    // Max source index = (src_height-1) * src_stride + (src_width-1) * 4 + 3
    let src_max_idx = src_height_minus_1 as usize * src_stride + src_width_minus_1 as usize * 4 + 3;
    if src.len() <= src_max_idx {
        return; // Source buffer too small
    }

    // Pre-validate destination bounds
    let dest_max_idx = (end_y - 1) as usize * dest_stride + (end_x - 1) as usize * 4 + 3;
    if dest.len() <= dest_max_idx {
        return; // Destination buffer too small
    }

    // Now we can use unchecked indexing safely
    for y in start_y..end_y {
        let local_y = (y as i32 - dest_y) as u64;
        let src_y_fp = (local_y * scale_y_fp) as u32;
        let y0 = (src_y_fp >> 16).min(src_height_minus_1);
        let y1 = (y0 + 1).min(src_height_minus_1);
        let fy = (src_y_fp & 0xFFFF) >> 8; // 8-bit fraction
        let inv_fy = 256 - fy;

        let dest_row = y as usize * dest_stride;
        let src_row0 = y0 as usize * src_stride;
        let src_row1 = y1 as usize * src_stride;

        for x in start_x..end_x {
            let local_x = (x as i32 - dest_x) as u64;
            let src_x_fp = (local_x * scale_x_fp) as u32;
            let x0 = (src_x_fp >> 16).min(src_width_minus_1);
            let x1 = (x0 + 1).min(src_width_minus_1);
            let fx = (src_x_fp & 0xFFFF) >> 8; // 8-bit fraction
            let inv_fx = 256 - fx;

            let x0_4 = x0 as usize * 4;
            let x1_4 = x1 as usize * 4;
            let dest_idx = dest_row + x as usize * 4;

            // Bilinear interpolation using fixed-point math
            // Weight factors (8-bit precision, sum to 256*256 = 65536)
            let w00 = inv_fx * inv_fy;
            let w10 = fx * inv_fy;
            let w01 = inv_fx * fy;
            let w11 = fx * fy;

            // SAFETY: We pre-validated all bounds above
            unsafe {
                let s00 = src.get_unchecked(src_row0 + x0_4..src_row0 + x0_4 + 4);
                let s10 = src.get_unchecked(src_row0 + x1_4..src_row0 + x1_4 + 4);
                let s01 = src.get_unchecked(src_row1 + x0_4..src_row1 + x0_4 + 4);
                let s11 = src.get_unchecked(src_row1 + x1_4..src_row1 + x1_4 + 4);
                let d = dest.get_unchecked_mut(dest_idx..dest_idx + 4);

                // Unrolled RGBA interpolation
                d[0] = ((s00[0] as u32 * w00
                    + s10[0] as u32 * w10
                    + s01[0] as u32 * w01
                    + s11[0] as u32 * w11)
                    >> 16) as u8;
                d[1] = ((s00[1] as u32 * w00
                    + s10[1] as u32 * w10
                    + s01[1] as u32 * w01
                    + s11[1] as u32 * w11)
                    >> 16) as u8;
                d[2] = ((s00[2] as u32 * w00
                    + s10[2] as u32 * w10
                    + s01[2] as u32 * w01
                    + s11[2] as u32 * w11)
                    >> 16) as u8;
                d[3] = ((s00[3] as u32 * w00
                    + s10[3] as u32 * w10
                    + s01[3] as u32 * w01
                    + s11[3] as u32 * w11)
                    >> 16) as u8;
            }
        }
    }
}

/// Trilinear blit: blends between two mip levels for smooth level transitions.
/// This performs bilinear sampling on both levels and blends the results.
///
/// # Arguments
/// * `dest` - Destination buffer (RGBA)
/// * `dest_width`, `dest_height` - Destination dimensions
/// * `src_fine` - Source buffer from higher resolution (lower level index) tile
/// * `src_fine_width`, `src_fine_height` - Fine source dimensions
/// * `src_coarse` - Source buffer from lower resolution (higher level index) tile  
/// * `src_coarse_width`, `src_coarse_height` - Coarse source dimensions
/// * `dest_x`, `dest_y` - Destination position for fine tile
/// * `fine_scaled_width`, `fine_scaled_height` - Scaled size for fine tile
/// * `coarse_offset_x`, `coarse_offset_y` - Offset within coarse tile for this region
/// * `blend` - Blend factor: 0.0 = use fine, 1.0 = use coarse
#[allow(clippy::too_many_arguments)]
fn blit_tile_trilinear(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src_fine: &[u8],
    src_fine_width: u32,
    src_fine_height: u32,
    src_coarse: &[u8],
    src_coarse_width: u32,
    src_coarse_height: u32,
    dest_x: i32,
    dest_y: i32,
    fine_scaled_width: i32,
    fine_scaled_height: i32,
    coarse_src_x: f64,
    coarse_src_y: f64,
    coarse_src_w: f64,
    coarse_src_h: f64,
    blend: f64,
) {
    if fine_scaled_width <= 0 || fine_scaled_height <= 0 {
        return;
    }

    // Early return if tile is entirely off-screen
    if dest_x + fine_scaled_width <= 0 || dest_y + fine_scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }

    // Calculate visible region (clamp to destination bounds)
    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + fine_scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + fine_scaled_height).min(dest_height as i32) as u32;

    // Fixed-point scale factors for fine level
    // Map destination [0, scaled_width-1] to source [0, src_width-1] for proper edge sampling
    let denom_x = (fine_scaled_width - 1).max(1) as u64;
    let denom_y = (fine_scaled_height - 1).max(1) as u64;
    let scale_x_fine_fp = (((src_fine_width - 1).max(1) as u64) << 16) / denom_x;
    let scale_y_fine_fp = (((src_fine_height - 1).max(1) as u64) << 16) / denom_y;

    // Scale factors to map from destination pixel to coarse source pixel
    let scale_x_coarse = coarse_src_w / (fine_scaled_width as f64).max(1.0);
    let scale_y_coarse = coarse_src_h / (fine_scaled_height as f64).max(1.0);

    let src_fine_width_minus_1 = src_fine_width - 1;
    let src_fine_height_minus_1 = src_fine_height - 1;
    let src_coarse_width_minus_1 = src_coarse_width - 1;
    let src_coarse_height_minus_1 = src_coarse_height - 1;

    let dest_stride = dest_width * 4;
    let src_fine_stride = src_fine_width * 4;
    let src_coarse_stride = src_coarse_width * 4;

    // Convert blend to fixed-point (256 scale)
    let blend_coarse = (blend * 256.0).clamp(0.0, 256.0) as u32;
    let blend_fine = 256 - blend_coarse;

    for y in start_y..end_y {
        let local_y = (y as i32 - dest_y) as f64;

        // Fine level sampling coords
        let src_y_fine_fp = (local_y as u64 * scale_y_fine_fp) as u32;
        let y0_fine = (src_y_fine_fp >> 16).min(src_fine_height_minus_1);
        let y1_fine = (y0_fine + 1).min(src_fine_height_minus_1);
        let fy_fine = (src_y_fine_fp & 0xFFFF) >> 8;
        let inv_fy_fine = 256 - fy_fine;

        // Coarse level sampling coords (clamp to valid range)
        let coarse_y = (coarse_src_y + local_y * scale_y_coarse).max(0.0);
        let y0_coarse = (coarse_y.floor() as u32).min(src_coarse_height_minus_1);
        let y1_coarse = (y0_coarse + 1).min(src_coarse_height_minus_1);
        let fy_coarse = (((coarse_y - coarse_y.floor()) * 256.0) as u32).min(255);
        let inv_fy_coarse = 256 - fy_coarse;

        let dest_row = (y * dest_stride) as usize;
        let src_fine_row0 = (y0_fine * src_fine_stride) as usize;
        let src_fine_row1 = (y1_fine * src_fine_stride) as usize;
        let src_coarse_row0 = (y0_coarse * src_coarse_stride) as usize;
        let src_coarse_row1 = (y1_coarse * src_coarse_stride) as usize;

        for x in start_x..end_x {
            let local_x = (x as i32 - dest_x) as f64;

            // Fine level X coords
            let src_x_fine_fp = (local_x as u64 * scale_x_fine_fp) as u32;
            let x0_fine = (src_x_fine_fp >> 16).min(src_fine_width_minus_1);
            let x1_fine = (x0_fine + 1).min(src_fine_width_minus_1);
            let fx_fine = (src_x_fine_fp & 0xFFFF) >> 8;
            let inv_fx_fine = 256 - fx_fine;

            // Coarse level X coords (clamp to valid range)
            let coarse_x = (coarse_src_x + local_x * scale_x_coarse).max(0.0);
            let x0_coarse = (coarse_x.floor() as u32).min(src_coarse_width_minus_1);
            let x1_coarse = (x0_coarse + 1).min(src_coarse_width_minus_1);
            let fx_coarse = (((coarse_x - coarse_x.floor()) * 256.0) as u32).min(255);
            let inv_fx_coarse = 256 - fx_coarse;

            // Fine level indices
            let idx_fine_00 = src_fine_row0 + (x0_fine * 4) as usize;
            let idx_fine_10 = src_fine_row0 + (x1_fine * 4) as usize;
            let idx_fine_01 = src_fine_row1 + (x0_fine * 4) as usize;
            let idx_fine_11 = src_fine_row1 + (x1_fine * 4) as usize;

            // Coarse level indices
            let idx_coarse_00 = src_coarse_row0 + (x0_coarse * 4) as usize;
            let idx_coarse_10 = src_coarse_row0 + (x1_coarse * 4) as usize;
            let idx_coarse_01 = src_coarse_row1 + (x0_coarse * 4) as usize;
            let idx_coarse_11 = src_coarse_row1 + (x1_coarse * 4) as usize;

            let dest_idx = dest_row + (x * 4) as usize;

            let fine_valid = idx_fine_11 + 3 < src_fine.len();
            let coarse_valid = idx_coarse_11 + 3 < src_coarse.len();

            if dest_idx + 3 < dest.len() {
                // Bilinear weights
                let w_fine_00 = inv_fx_fine * inv_fy_fine;
                let w_fine_10 = fx_fine * inv_fy_fine;
                let w_fine_01 = inv_fx_fine * fy_fine;
                let w_fine_11 = fx_fine * fy_fine;

                let w_coarse_00 = inv_fx_coarse * inv_fy_coarse;
                let w_coarse_10 = fx_coarse * inv_fy_coarse;
                let w_coarse_01 = inv_fx_coarse * fy_coarse;
                let w_coarse_11 = fx_coarse * fy_coarse;

                for c in 0..4 {
                    // Sample from fine level
                    let fine_sample = if fine_valid {
                        (src_fine[idx_fine_00 + c] as u32 * w_fine_00
                            + src_fine[idx_fine_10 + c] as u32 * w_fine_10
                            + src_fine[idx_fine_01 + c] as u32 * w_fine_01
                            + src_fine[idx_fine_11 + c] as u32 * w_fine_11)
                            >> 16
                    } else {
                        0
                    };

                    // Sample from coarse level
                    let coarse_sample = if coarse_valid {
                        (src_coarse[idx_coarse_00 + c] as u32 * w_coarse_00
                            + src_coarse[idx_coarse_10 + c] as u32 * w_coarse_10
                            + src_coarse[idx_coarse_01 + c] as u32 * w_coarse_01
                            + src_coarse[idx_coarse_11 + c] as u32 * w_coarse_11)
                            >> 16
                    } else if fine_valid {
                        fine_sample // Fallback to fine if coarse not available
                    } else {
                        0
                    };

                    // Blend between levels
                    let blended = if fine_valid && coarse_valid {
                        (fine_sample * blend_fine + coarse_sample * blend_coarse) >> 8
                    } else if fine_valid {
                        fine_sample
                    } else if coarse_valid {
                        coarse_sample
                    } else {
                        0
                    };

                    dest[dest_idx + c] = blended.min(255) as u8;
                }
            }
        }
    }
}

fn create_image_buffer(
    data: &[u8],
    width: u32,
    height: u32,
) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let expected_len = (width * height * 4) as usize;
    if data.len() < expected_len {
        return None;
    }

    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    buffer
        .make_mut_bytes()
        .copy_from_slice(&data[..expected_len]);
    Some(buffer)
}

// ============ Tool handling functions ============

fn update_tool_state(ui: &AppWindow, state: &AppState) {
    let tool_type = match state.current_tool {
        state::Tool::Navigate => ToolType::Navigate,
        state::Tool::RegionOfInterest => ToolType::RegionOfInterest,
        state::Tool::MeasureDistance => ToolType::MeasureDistance,
    };
    ui.set_current_tool(tool_type);
}

fn hidden_roi_rect(ant_offset: f32) -> ROIRect {
    ROIRect {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: 0.0,
        visible: false,
        ant_offset,
    }
}

fn hidden_measurement_line() -> MeasurementLine {
    MeasurementLine {
        x1: 0.0,
        y1: 0.0,
        x2: 0.0,
        y2: 0.0,
        distance_um: 0.0,
        visible: false,
    }
}

fn set_pane_roi_rect(ui: &AppWindow, pane: PaneId, rect: ROIRect) {
    match pane {
        PaneId::Primary => ui.set_primary_roi_rect(rect),
        PaneId::SECONDARY => ui.set_secondary_roi_rect(rect),
        _ => {}
    }
}

fn set_pane_measurements(ui: &AppWindow, pane: PaneId, lines: Vec<MeasurementLine>) {
    let model = Rc::new(VecModel::from(lines));
    match pane {
        PaneId::Primary => ui.set_primary_measurements(model.into()),
        PaneId::SECONDARY => ui.set_secondary_measurements(model.into()),
        _ => {}
    }
}

fn set_pane_candidate_measurement(ui: &AppWindow, pane: PaneId, line: MeasurementLine) {
    match pane {
        PaneId::Primary => ui.set_primary_candidate_measurement(line),
        PaneId::SECONDARY => ui.set_secondary_candidate_measurement(line),
        _ => {}
    }
}

fn clear_tool_overlays(ui: &AppWindow, ant_offset: f32) {
    set_pane_roi_rect(ui, PaneId::Primary, hidden_roi_rect(ant_offset));
    set_pane_roi_rect(ui, PaneId::SECONDARY, hidden_roi_rect(ant_offset));
    set_pane_measurements(ui, PaneId::Primary, Vec::new());
    set_pane_measurements(ui, PaneId::SECONDARY, Vec::new());
    set_pane_candidate_measurement(ui, PaneId::Primary, hidden_measurement_line());
    set_pane_candidate_measurement(ui, PaneId::SECONDARY, hidden_measurement_line());
}

fn hidden_viewport_info() -> ViewportInfo {
    ViewportInfo {
        center_x: 0.0,
        center_y: 0.0,
        zoom: 1.0,
        image_width: 1.0,
        image_height: 1.0,
        level: 0,
    }
}

fn full_minimap_rect() -> MinimapRect {
    MinimapRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    }
}

fn pane_viewport_state(file: &OpenFile, pane: PaneId) -> Option<&ViewportState> {
    file.pane_state(pane)
        .map(|pane_state| &pane_state.viewport)
        .or(match pane {
            PaneId::Primary => Some(&file.viewport),
            PaneId::SECONDARY => file.secondary_viewport.as_ref(),
            _ => None,
        })
}

fn pane_overlay_data(
    state: &AppState,
    pane: PaneId,
) -> (ROIRect, Vec<MeasurementLine>, MeasurementLine) {
    let ant_offset = state.ant_offset;
    let mut roi_rect = hidden_roi_rect(ant_offset);
    let mut measurement_lines = Vec::new();
    let mut candidate_measurement = hidden_measurement_line();

    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };
    let Some(file) = state.get_file(file_id) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };
    let Some(viewport_state) = pane_viewport_state(file, pane) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };

    let vp = &viewport_state.viewport;
    let bounds = vp.bounds();
    let mpp = file
        .wsi
        .properties()
        .mpp_x
        .zip(file.wsi.properties().mpp_y)
        .map(|(mx, my)| (mx + my) / 2.0)
        .unwrap_or(0.0);

    let committed_roi = file.roi.filter(|roi| roi.pane == pane);
    let roi_to_display = if state.focused_pane == pane && state.current_tool == state::Tool::RegionOfInterest {
        if let state::ToolInteractionState::Dragging(start) = state.tool_state {
            state
                .candidate_point
                .map(|end| state::RegionOfInterest::from_points(start, end, pane))
                .or(committed_roi)
        } else {
            committed_roi
        }
    } else {
        committed_roi
    };

    if let Some(roi) = roi_to_display {
        roi_rect = ROIRect {
            x: ((roi.x - bounds.left) * vp.zoom) as f32,
            y: ((roi.y - bounds.top) * vp.zoom) as f32,
            width: (roi.width * vp.zoom) as f32,
            height: (roi.height * vp.zoom) as f32,
            visible: true,
            ant_offset,
        };
    }

    measurement_lines = file
        .measurements
        .iter()
        .filter(|measurement| measurement.pane == pane)
        .map(|measurement| {
            let p1 = vp.image_to_screen(measurement.start.x, measurement.start.y);
            let p2 = vp.image_to_screen(measurement.end.x, measurement.end.y);
            MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um: (measurement.distance() * mpp) as f32,
                visible: true,
            }
        })
        .collect();

    if state.focused_pane == pane && state.current_tool == state::Tool::MeasureDistance {
        if let (
            state::ToolInteractionState::Dragging(start)
            | state::ToolInteractionState::FirstPointPlaced(start),
            Some(end),
        ) = (state.tool_state, state.candidate_point)
        {
            let p1 = vp.image_to_screen(start.x, start.y);
            let p2 = vp.image_to_screen(end.x, end.y);
            let candidate = state::Measurement { pane, start, end };
            candidate_measurement = MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um: (candidate.distance() * mpp) as f32,
                visible: true,
            };
        }
    }

    (roi_rect, measurement_lines, candidate_measurement)
}

fn has_active_roi_overlay(state: &AppState) -> bool {
    if state.current_tool == state::Tool::RegionOfInterest
        && matches!(state.tool_state, state::ToolInteractionState::Dragging(_))
        && state.candidate_point.is_some()
    {
        return true;
    }

    state.panes.iter().enumerate().any(|(pane_index, _)| {
        let pane = PaneId(pane_index);
        state
            .active_file_id_for_pane(pane)
            .and_then(|id| state.get_file(id))
            .and_then(|file| file.roi)
            .map(|roi| roi.pane == pane)
            .unwrap_or(false)
    })
}

fn update_tool_overlays(_ui: &AppWindow, _state: &AppState) {
}

fn handle_tool_mouse_down(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = state::ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    match state.current_tool {
        state::Tool::Navigate => {
            // Should not happen - Navigate uses LMB for panning
        }
        state::Tool::RegionOfInterest => {
            state.tool_state = state::ToolInteractionState::Dragging(point);
            state.candidate_point = Some(point);
        }
        state::Tool::MeasureDistance => {
            // If we already have a first point placed (click-click mode),
            // this second click commits the measurement.
            if let state::ToolInteractionState::FirstPointPlaced(start) = state.tool_state {
                let measurement = state::Measurement {
                    pane: focused_pane,
                    start,
                    end: point,
                };
                file.measurements.clear();
                file.measurements.push(measurement);
                state.tool_state = state::ToolInteractionState::Idle;
                state.candidate_point = None;
            } else {
                // Start a new measurement — clear any previous one
                file.measurements.clear();
                state.tool_state = state::ToolInteractionState::Dragging(point);
                state.candidate_point = Some(point);
            }
        }
    }
}

fn handle_tool_mouse_move(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;

    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = state::ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    // Update candidate point during drag or first-point-placed
    match state.tool_state {
        state::ToolInteractionState::Dragging(_)
        | state::ToolInteractionState::FirstPointPlaced(_) => {
            state.candidate_point = Some(point);
        }
        _ => {}
    }
}

fn handle_tool_mouse_up(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;
    let current_tool = state.current_tool;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let end_point = state::ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    if let state::ToolInteractionState::Dragging(start) = state.tool_state {
        match current_tool {
            state::Tool::Navigate => {}
            state::Tool::RegionOfInterest => {
                // Create ROI from the two points
                let roi = state::RegionOfInterest::from_points(start, end_point, focused_pane);
                if roi.is_valid() {
                    file.roi = Some(roi);
                }
                state.tool_state = state::ToolInteractionState::Idle;
                state.candidate_point = None;
            }
            state::Tool::MeasureDistance => {
                // Check if the mouse moved enough to be a drag
                let dx = end_point.x - start.x;
                let dy = end_point.y - start.y;
                let dist_sq = dx * dx + dy * dy;
                // Threshold: 5 pixels in image space
                if dist_sq > 25.0 {
                    // Drag completed — commit measurement
                    let measurement = state::Measurement {
                        pane: focused_pane,
                        start,
                        end: end_point,
                    };
                    file.measurements.clear();
                    file.measurements.push(measurement);
                    state.tool_state = state::ToolInteractionState::Idle;
                    state.candidate_point = None;
                } else {
                    // Click (mouse didn't move much) — enter click-click mode
                    state.tool_state = state::ToolInteractionState::FirstPointPlaced(start);
                    // Keep candidate_point so the preview line follows the mouse
                }
            }
        }
    } else {
        // For FirstPointPlaced, mouse-up is ignored (commit happens on next mouse-down)
        // Reset for other states
        if !matches!(
            state.tool_state,
            state::ToolInteractionState::FirstPointPlaced(_)
        ) {
            state.tool_state = state::ToolInteractionState::Idle;
            state.candidate_point = None;
        }
    }
}
