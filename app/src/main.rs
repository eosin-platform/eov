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
    BackendSelector, Image, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel,
};
use state::{AppState, OpenFile, PaneId, RenderBackend, TileRequestSignature};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tile_loader::{TileLoader, calculate_wanted_tiles};
use tracing::{error, info, trace, warn};

/// Coarse tile blend data: (tile_data, coarse_offset, coarse_size, blend_factor)
type CoarseBlendData = (Arc<common::TileData>, [f32; 2], [f32; 2], f32);

thread_local! {
    static GPU_RENDERER_HANDLE: RefCell<Option<Rc<RefCell<GpuRenderer>>>> = const { RefCell::new(None) };
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
    if index == 1 {
        PaneId::Secondary
    } else {
        PaneId::Primary
    }
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
                "GPU renderer unavailable, using CPU renderer",
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
                        "GPU renderer unavailable, using CPU renderer",
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
                update_tabs(&ui, &state);
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
                update_tabs(&ui, &state);
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
                update_tabs(&ui, &state);
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
                    state.open_files.clear();
                    state.home_tabs.clear();
                    state.primary_tabs.clear();
                    state.secondary_tabs.clear();
                    state.primary_active_tab_id = None;
                    state.secondary_active_tab_id = None;
                    state.split_enabled = false;
                    state.set_focused_pane(PaneId::Primary);
                    state.active_file_id = None;
                    state.request_render();
                }
                let state = state_handle.read();
                update_tabs(&ui, &state);
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

        ui.on_copy_path(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id)
                && let Ok(mut clipboard) = arboard::Clipboard::new()
            {
                let _ = clipboard.set_text(file.path.display().to_string());
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
                    if !state.split_enabled {
                        state.activate_tab_in_pane(PaneId::Primary, id);
                        state.enable_split();
                        state.duplicate_tab_to_pane(id, PaneId::Secondary);
                        info!("Split view enabled");
                    } else if source_pane == PaneId::Primary {
                        state.move_tab_between_panes(id, PaneId::Primary, PaneId::Secondary);
                    } else {
                        state.move_tab_between_panes(id, PaneId::Secondary, PaneId::Primary);
                    }
                    (state.split_enabled, state.focused_pane)
                };
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
                let active_id = state.active_file_id;
                let focused_pane = state.focused_pane;
                let split_enabled = state.split_enabled;
                let mut changed = false;

                if let Some(file) = state
                    .open_files
                    .iter_mut()
                    .find(|f| Some(f.id) == active_id)
                {
                    let nx = (nx as f64).clamp(0.0, 1.0);
                    let ny = (ny as f64).clamp(0.0, 1.0);
                    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
                        file.secondary_viewport.as_mut()
                    } else {
                        Some(&mut file.viewport)
                    };
                    if let Some(vs) = viewport_state {
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
                    state.move_tab_between_panes(id, pane_from_index(from), pane_from_index(to));
                }
                let state = state_handle.read();
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
                update_tabs(&ui, &state);
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
        // Physical pixel position where the left button was pressed (None = not in drag zone)
        let drag_press: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        // Double-click tracking
        let last_press_time = Rc::new(Cell::new(std::time::Instant::now()));
        let dbl_count = Rc::new(Cell::new(0u32));

        ui.window().on_winit_window_event({
            let last_cursor = Rc::clone(&last_cursor);
            let drag_press = Rc::clone(&drag_press);
            let last_press_time = Rc::clone(&last_press_time);
            let dbl_count = Rc::clone(&dbl_count);

            move |slint_window, event| {
                match event {
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
                let target_pane = if state_guard.split_enabled {
                    state_guard.focused_pane
                } else {
                    PaneId::Primary
                };
                let (ui_width, ui_height) = match target_pane {
                    PaneId::Primary => (
                        ui.get_viewport_width() as f64,
                        ui.get_viewport_height() as f64,
                    ),
                    PaneId::Secondary => (
                        ui.get_secondary_viewport_width() as f64,
                        ui.get_secondary_viewport_height() as f64,
                    ),
                };
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

    ui.set_primary_tabs(Rc::new(VecModel::from(build_pane_tabs(PaneId::Primary))).into());
    ui.set_secondary_tabs(Rc::new(VecModel::from(build_pane_tabs(PaneId::Secondary))).into());
    ui.set_primary_is_home_tab(state.is_home_tab_active_in_pane(PaneId::Primary));
    ui.set_secondary_is_home_tab(state.is_home_tab_active_in_pane(PaneId::Secondary));
    ui.set_split_enabled(state.split_enabled);
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

fn set_minimap_thumbnail_for_file(ui: &AppWindow, file: &OpenFile, secondary: bool) {
    let Some(ref thumb_data) = file.thumbnail else {
        return;
    };

    let level = file.wsi.level_count().saturating_sub(1);
    let Some(level_info) = file.wsi.level(level) else {
        return;
    };
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (width, height) = if aspect > 1.0 {
        (150u32, (150.0 / aspect) as u32)
    } else {
        ((150.0 * aspect) as u32, 150u32)
    };

    if let Some(buffer) = create_image_buffer(thumb_data, width.max(1), height.max(1)) {
        let image = Image::from_rgba8(buffer);
        if secondary {
            ui.set_secondary_minimap_thumbnail(image);
        } else {
            ui.set_minimap_thumbnail(image);
        }
    }
}

fn update_and_render(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let mut state = state.write();
    let primary_file_id = state.active_file_id_for_pane(PaneId::Primary);
    let secondary_file_id = if state.split_enabled {
        state.active_file_id_for_pane(PaneId::Secondary)
    } else {
        None
    };
    if primary_file_id.is_none() && secondary_file_id.is_none() {
        ui.set_roi_rect(ROIRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            visible: false,
            ant_offset: 0.0,
        });
        return false;
    }

    let split_enabled = state.split_enabled;
    let render_backend = state.render_backend;
    let focused_pane = state.focused_pane;

    // Capture tool state for ROI calculation (before borrowing file mutably)
    let tool_state = state.tool_state;
    let candidate_point = state.candidate_point;
    let current_tool = state.current_tool;
    let render_requested = std::mem::take(&mut state.needs_render);

    // Update ant offset for marching ants animation (0.5 pixels per frame at 60fps = 30 pixels/sec)
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;
    let ant_offset = state.ant_offset;

    let primary_file_switched = state.last_primary_rendered_file_id != primary_file_id;
    let secondary_file_switched = state.last_secondary_rendered_file_id != secondary_file_id;
    let mut keep_running = render_requested || primary_file_switched || secondary_file_switched;
    let mut rendered_frame = false;

    if let Some(file_id) = primary_file_id {
        let primary_width = (ui.get_viewport_width() as f64).max(100.0);
        let primary_height = (ui.get_viewport_height() as f64).max(100.0);

        {
            let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
                return false;
            };

            if primary_file_switched {
                file.frame_count = 0;
                set_minimap_thumbnail_for_file(ui, file, false);
            }

            let primary_animating = file.viewport.update();
            file.viewport.set_size(primary_width, primary_height);
            let tile_epoch = file.tile_loader.loaded_epoch();
            let new_primary_tiles = tile_epoch != file.last_seen_tile_epoch;
            if new_primary_tiles {
                file.last_seen_tile_epoch = tile_epoch;
            }
            let tiles_pending = file.tile_loader.pending_count() > 0;
            keep_running |= primary_animating || new_primary_tiles || tiles_pending;

            let vp = &file.viewport.viewport;
            ui.set_viewport_info(ViewportInfo {
                center_x: vp.center.x as f32,
                center_y: vp.center.y as f32,
                zoom: vp.zoom as f32,
                image_width: vp.image_width as f32,
                image_height: vp.image_height as f32,
                level: file
                    .wsi
                    .best_level_for_downsample(vp.effective_downsample())
                    as i32,
            });
            ui.set_zoom_slider_position(zoom_to_slider_value(vp.zoom));

            let rect = vp.minimap_rect();
            ui.set_minimap_rect(MinimapRect {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            });

            rendered_frame |= if render_backend == RenderBackend::Gpu
                && with_gpu_renderer(|renderer| renderer.borrow().is_ready()).unwrap_or(false)
            {
                render_viewport_gpu(ui, file, tile_cache, SurfaceSlot::Primary)
            } else {
                render_viewport_to_buffer(ui, file, tile_cache, true)
            };

            if focused_pane == PaneId::Primary || !split_enabled {
                let vp = &file.viewport.viewport;
                let bounds = vp.bounds();
                let roi_to_display = if current_tool == state::Tool::RegionOfInterest {
                    if let state::ToolInteractionState::Dragging(start) = tool_state {
                        if let Some(end) = candidate_point {
                            Some(state::RegionOfInterest::from_points(start, end))
                        } else {
                            file.roi
                        }
                    } else {
                        file.roi
                    }
                } else {
                    file.roi
                };

                if let Some(roi) = roi_to_display {
                    let screen_x = (roi.x - bounds.left) * vp.zoom;
                    let screen_y = (roi.y - bounds.top) * vp.zoom;
                    let screen_w = roi.width * vp.zoom;
                    let screen_h = roi.height * vp.zoom;

                    ui.set_roi_rect(ROIRect {
                        x: screen_x as f32,
                        y: screen_y as f32,
                        width: screen_w as f32,
                        height: screen_h as f32,
                        visible: true,
                        ant_offset,
                    });
                } else {
                    ui.set_roi_rect(ROIRect {
                        x: 0.0,
                        y: 0.0,
                        width: 0.0,
                        height: 0.0,
                        visible: false,
                        ant_offset,
                    });
                }
                keep_running |= roi_to_display.is_some();
            }
        }
    }

    if split_enabled && let Some(file_id) = secondary_file_id {
        let secondary_width = (ui.get_secondary_viewport_width() as f64).max(100.0);
        let secondary_height = (ui.get_secondary_viewport_height() as f64).max(100.0);

        {
            let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
                return false;
            };

            if secondary_file_switched {
                set_minimap_thumbnail_for_file(ui, file, true);
            }

            let secondary_overlay = if let Some(ref mut secondary) = file.secondary_viewport {
                let secondary_animating = secondary.update();
                secondary.set_size(secondary_width, secondary_height);
                keep_running |= secondary_animating;

                let vp2 = &secondary.viewport;
                ui.set_secondary_viewport_info(ViewportInfo {
                    center_x: vp2.center.x as f32,
                    center_y: vp2.center.y as f32,
                    zoom: vp2.zoom as f32,
                    image_width: vp2.image_width as f32,
                    image_height: vp2.image_height as f32,
                    level: file
                        .wsi
                        .best_level_for_downsample(vp2.effective_downsample())
                        as i32,
                });
                ui.set_secondary_zoom_slider_position(zoom_to_slider_value(vp2.zoom));

                let rect2 = vp2.minimap_rect();
                ui.set_secondary_minimap_rect(MinimapRect {
                    x: rect2.x,
                    y: rect2.y,
                    width: rect2.width,
                    height: rect2.height,
                });

                Some((vp2.bounds(), vp2.zoom))
            } else {
                None
            };

            rendered_frame |= if render_backend == RenderBackend::Gpu
                && with_gpu_renderer(|renderer| renderer.borrow().is_ready()).unwrap_or(false)
            {
                render_secondary_viewport_gpu(ui, file, tile_cache)
            } else {
                render_secondary_viewport(ui, file, tile_cache)
            };

            if focused_pane == PaneId::Secondary
                && let Some((bounds, secondary_zoom)) = secondary_overlay
            {
                let roi_to_display = if current_tool == state::Tool::RegionOfInterest {
                    if let state::ToolInteractionState::Dragging(start) = tool_state {
                        if let Some(end) = candidate_point {
                            Some(state::RegionOfInterest::from_points(start, end))
                        } else {
                            file.roi
                        }
                    } else {
                        file.roi
                    }
                } else {
                    file.roi
                };

                if let Some(roi) = roi_to_display {
                    let screen_x = (roi.x - bounds.left) * secondary_zoom;
                    let screen_y = (roi.y - bounds.top) * secondary_zoom;
                    let screen_w = roi.width * secondary_zoom;
                    let screen_h = roi.height * secondary_zoom;

                    ui.set_roi_rect(ROIRect {
                        x: screen_x as f32,
                        y: screen_y as f32,
                        width: screen_w as f32,
                        height: screen_h as f32,
                        visible: true,
                        ant_offset,
                    });
                } else {
                    ui.set_roi_rect(ROIRect {
                        x: 0.0,
                        y: 0.0,
                        width: 0.0,
                        height: 0.0,
                        visible: false,
                        ant_offset,
                    });
                }
                keep_running |= roi_to_display.is_some();
            }
        }
    }

    // Update measurement overlays to track viewport changes during pan/zoom
    update_tool_overlays(ui, &state);

    state.last_primary_rendered_file_id = primary_file_id;
    state.last_secondary_rendered_file_id = secondary_file_id;

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
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
    with_gpu_renderer(|renderer| {
        renderer
            .borrow_mut()
            .queue_frame(ui, slot, render_width, render_height, draws)
    })
    .unwrap_or(false)
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
    with_gpu_renderer(|renderer| {
        renderer.borrow_mut().queue_frame(
            ui,
            SurfaceSlot::Secondary,
            render_width,
            render_height,
            draws,
        )
    })
    .unwrap_or(false)
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

fn update_tool_overlays(ui: &AppWindow, state: &AppState) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let viewport_state = if state.split_enabled && state.focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
    };
    let vp = &viewport_state.viewport;
    let bounds = vp.bounds();

    // Microns per pixel (average of x and y, fallback to 0 if unknown)
    let mpp = file
        .wsi
        .properties()
        .mpp_x
        .zip(file.wsi.properties().mpp_y)
        .map(|(mx, my)| (mx + my) / 2.0)
        .unwrap_or(0.0);

    // Update ROI overlay
    let ant_offset = state.ant_offset;
    if let Some(roi) = &file.roi {
        let screen_x = (roi.x - bounds.left) * vp.zoom;
        let screen_y = (roi.y - bounds.top) * vp.zoom;
        let screen_w = roi.width * vp.zoom;
        let screen_h = roi.height * vp.zoom;

        ui.set_roi_rect(ROIRect {
            x: screen_x as f32,
            y: screen_y as f32,
            width: screen_w as f32,
            height: screen_h as f32,
            visible: true,
            ant_offset,
        });
    } else {
        ui.set_roi_rect(ROIRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            visible: false,
            ant_offset,
        });
    }

    // Update committed measurements (image-space → screen-space)
    let measurement_lines: Vec<MeasurementLine> = file
        .measurements
        .iter()
        .map(|m| {
            let p1 = vp.image_to_screen(m.start.x, m.start.y);
            let p2 = vp.image_to_screen(m.end.x, m.end.y);
            let dist_px = m.distance();
            let distance_um = (dist_px * mpp) as f32;
            MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um,
                visible: true,
            }
        })
        .collect();
    let measurements_model = std::rc::Rc::new(slint::VecModel::from(measurement_lines));
    ui.set_measurements(measurements_model.into());

    // Update candidate measurement
    if let (
        state::ToolInteractionState::Dragging(start)
        | state::ToolInteractionState::FirstPointPlaced(start),
        Some(end),
    ) = (&state.tool_state, &state.candidate_point)
    {
        if state.current_tool == state::Tool::MeasureDistance {
            let p1 = vp.image_to_screen(start.x, start.y);
            let p2 = vp.image_to_screen(end.x, end.y);
            let candidate_m = state::Measurement {
                start: *start,
                end: *end,
            };
            let distance_um = (candidate_m.distance() * mpp) as f32;

            ui.set_candidate_measurement(MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um,
                visible: true,
            });
        } else if state.current_tool == state::Tool::RegionOfInterest {
            let roi = state::RegionOfInterest::from_points(*start, *end);
            let screen_x = (roi.x - bounds.left) * vp.zoom;
            let screen_y = (roi.y - bounds.top) * vp.zoom;
            let screen_w = roi.width * vp.zoom;
            let screen_h = roi.height * vp.zoom;

            ui.set_roi_rect(ROIRect {
                x: screen_x as f32,
                y: screen_y as f32,
                width: screen_w as f32,
                height: screen_h as f32,
                visible: true,
                ant_offset,
            });
        }
    } else {
        ui.set_candidate_measurement(MeasurementLine {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            distance_um: 0.0,
            visible: false,
        });
    }
}

fn handle_tool_mouse_down(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;
    let split_enabled = state.split_enabled;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
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
    let split_enabled = state.split_enabled;

    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
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
    let split_enabled = state.split_enabled;
    let current_tool = state.current_tool;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let viewport_state = if split_enabled && focused_pane == PaneId::Secondary {
        file.secondary_viewport.as_ref().unwrap_or(&file.viewport)
    } else {
        &file.viewport
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
                let roi = state::RegionOfInterest::from_points(start, end_point);
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
