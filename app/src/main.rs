// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod blitter;
mod callbacks;
mod cli;
mod clipboard;
mod config;
mod file_ops;
mod gpu;
mod pane_ui;
mod plugins;
mod render;
mod render_pool;
mod stain;
mod state;
mod tile_loader;
mod tools;
mod ui_update;

use anyhow::Result;
use common::{RenderBackend, TileCache};
use gpu::GpuRenderer;
use parking_lot::RwLock;
use slint::{SharedString, Timer, TimerMode};
use state::{AppState, PaneId};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

use backend::select_backend;
use cli::{apply_config_override, init_tracing, maybe_run_cli_command, parse_launch_options};
pub(crate) use clipboard::{
    capture_pane_clipboard_image, copy_image_to_clipboard, copy_text_to_clipboard,
    crop_image_to_viewport_bounds,
};
pub(crate) use pane_ui::{
    PaneRenderCacheEntry, PaneUiModels, clear_cached_pane, insert_pane_ui_state, pane_from_index,
    set_cached_pane_content, set_cached_pane_cpu_result, set_cached_pane_minimap,
    with_gpu_renderer, with_pane_render_cache,
};
use pane_ui::{
    reset_pane_ui_state, set_gpu_renderer_handle, with_pane_ui_models, with_pane_view_model,
};

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
const APP_XDG_ID: &str = "io.eosin.eov";

fn refresh_tab_ui(ui: &AppWindow, state: &AppState) {
    reset_pane_ui_state();
    update_tabs(ui, state);
}

fn slider_value_to_zoom(value: f32) -> f64 {
    ui_update::slider_value_to_zoom(value)
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

            if !render::update_and_render(&ui, &state, &tile_cache) {
                timer_for_callback.stop();
                let mut state = state.write();
                state.render_loop_running = false;
            }
        },
    );
}

fn main() -> Result<()> {
    let launch_options = parse_launch_options()?;
    apply_config_override(launch_options.config_path.as_ref())?;
    init_tracing(launch_options.log_level);

    if maybe_run_cli_command(&launch_options)? {
        return Ok(());
    }

    info!("Starting EOV WSI Viewer");

    let persisted_backend = config::load_render_backend()?;
    let persisted_filtering = config::load_filtering_mode()?;
    let initial_filtering = launch_options
        .filtering_mode_override
        .or(persisted_filtering);
    let initial_backend = launch_options
        .render_backend_override
        .or(persisted_backend)
        .unwrap_or(RenderBackend::Gpu);

    if launch_options.debug_mode {
        info!("Debug mode enabled - FPS overlay will be shown");
    }

    if !launch_options.panes_to_open.is_empty() {
        let total_files: usize = launch_options
            .panes_to_open
            .iter()
            .map(|p| p.files.len())
            .sum();
        info!(
            "Opening {} file(s) across {} pane(s) from command line",
            total_files,
            launch_options.panes_to_open.len()
        );
    }

    let gpu_backend_available = select_backend(launch_options.window_geometry)?;
    slint::set_xdg_app_id(APP_XDG_ID)?;

    let state = Arc::new(RwLock::new(AppState::new()));
    let tile_cache = Arc::new(TileCache::with_limits(
        launch_options.max_tiles,
        launch_options.cache_size_bytes,
    ));
    render_pool::init_global()?;

    // ----- Plugin system initialization -----
    let plugin_manager = Rc::new(RefCell::new(plugins::PluginManager::new(
        launch_options.plugin_dir.clone(),
    )));
    {
        let mut pm = plugin_manager.borrow_mut();
        pm.discover();
        if let Err(e) = pm.activate_all() {
            tracing::warn!("Plugin activation error: {e}");
        }
        info!(
            "Plugin system ready: {} toolbar button(s) from {} plugin(s)",
            pm.toolbar.len(),
            pm.descriptors.len()
        );
    }

    let ui = AppWindow::new()?;
    ui.set_use_native_window_controls(cfg!(target_os = "macos"));

    // Apply CLI window size override. This must happen after AppWindow::new() so it
    // takes priority over the preferred-width/preferred-height set in the .slint file.
    {
        let geom = launch_options.window_geometry;
        if geom.width.is_some() || geom.height.is_some() {
            let current = ui.window().size();
            let scale = ui.window().scale_factor();
            let current_logical = current.to_logical(scale);
            let w = geom.width.unwrap_or(current_logical.width as u32);
            let h = geom.height.unwrap_or(current_logical.height as u32);
            ui.window()
                .set_size(slint::LogicalSize::new(w as f32, h as f32));
        }
    }

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
        Rc::clone(&plugin_manager),
    );

    // Set plugin toolbar buttons on the UI
    {
        let pm = plugin_manager.borrow();
        let buttons: Vec<crate::PluginButtonData> = pm
            .toolbar
            .buttons()
            .iter()
            .map(|b| crate::PluginButtonData {
                plugin_id: SharedString::from(&b.plugin_id),
                button_id: SharedString::from(&b.button_id),
                tooltip: SharedString::from(&b.tooltip),
                action_id: SharedString::from(&b.action_id),
            })
            .collect();
        let model = std::rc::Rc::new(slint::VecModel::from(buttons));
        ui.set_plugin_buttons(slint::ModelRc::from(model));
    }

    ui.set_debug_mode(launch_options.debug_mode);
    {
        let mut state = state.write();
        state.gpu_backend_available = gpu_backend_available;
        state.select_render_backend(initial_backend);
        if let Some(filtering) = initial_filtering {
            state.select_filtering_mode(filtering);
        }
        update_render_backend(&ui, &state);
        update_filtering_mode(&ui, &state);
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

    if launch_options.panes_to_open.is_empty() {
        {
            let mut state = state.write();
            state.create_home_tab();
        }
        let state = state.read();
        update_tabs(&ui, &state);
    } else {
        prepare_launch_panes(&ui, &state, launch_options.panes_to_open.len());
    }

    for (pane_index, pane_spec) in launch_options.panes_to_open.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        {
            let mut state = state.write();
            state.set_focused_pane(pane);
        }
        ui.set_focused_pane(pane.as_index());

        for path in pane_spec.files {
            open_file(&ui, &state, &tile_cache, &render_timer, path);
        }
    }

    if state.read().split_enabled {
        {
            let mut state = state.write();
            state.set_focused_pane(PaneId::PRIMARY);
        }
        let state = state.read();
        ui.set_split_enabled(state.split_enabled);
        ui.set_focused_pane(state.focused_pane.as_index());
        update_tabs(&ui, &state);
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
    plugin_manager: Rc<RefCell<plugins::PluginManager>>,
) {
    callbacks::setup_callbacks(ui, state, tile_cache, render_timer);

    // Plugin button click callback — dispatch to plugin manager and open windows
    let pm = Rc::clone(&plugin_manager);
    ui.on_plugin_button_clicked(move |plugin_id, action_id| {
        let plugin_id = plugin_id.to_string();
        let action_id = action_id.to_string();
        info!("Plugin button clicked: {plugin_id}:{action_id}");

        let mut pm = pm.borrow_mut();
        match pm.handle_action(&plugin_id, &action_id) {
            Ok((window_requests, vtable)) => {
                for req in window_requests {
                    if let Some(vt) = vtable {
                        crate::plugins::schedule_open_plugin_window(req, vt);
                    } else {
                        tracing::warn!(
                            "Window requested for plugin '{}' but no vtable available",
                            plugin_id
                        );
                    }
                }
            }
            Err(e) => {
                tracing::error!("Plugin action error: {e}");
            }
        }
    });
}

fn prepare_launch_panes(ui: &AppWindow, state: &Arc<RwLock<AppState>>, pane_count: usize) {
    if pane_count <= 1 {
        return;
    }

    let mut inserted_panes = Vec::new();
    {
        let mut state = state.write();
        while state.panes.len() < pane_count {
            let next_index = state.panes.len();
            let source_pane = PaneId(next_index.saturating_sub(1));
            let new_pane = state.insert_pane(next_index);
            inserted_panes.push((new_pane, source_pane));
        }
        state.set_focused_pane(PaneId::PRIMARY);
    }

    for (new_pane, source_pane) in inserted_panes {
        insert_pane_ui_state(new_pane, Some(source_pane));
    }

    let state = state.read();
    ui.set_split_enabled(state.split_enabled);
    ui.set_focused_pane(state.focused_pane.as_index());
    update_tabs(ui, &state);
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
                    file_ops::OpenFileUiContext {
                        pane_render_cache,
                        pane_ui_models,
                        pane_view_model,
                    },
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
                ui_update::update_tabs(
                    ui,
                    state,
                    pane_render_cache,
                    pane_ui_models,
                    pane_view_model,
                );
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

fn update_filtering_mode(ui: &AppWindow, state: &AppState) {
    ui_update::update_filtering_mode(ui, state)
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
