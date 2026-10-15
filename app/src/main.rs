// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audit;
mod backend;
mod batch;
mod blitter;
mod bookmarks;
mod cache_budget;
mod calibration_store;
mod callbacks;
mod channel_store;
mod channels;
mod cli;
mod clipboard;
mod color;
mod config;
mod dataset_dialog;
mod desktop;
mod distribution;
mod export_dialog;
mod file_ops;
mod file_types;
mod gpu;
mod history;
mod host_ui;
mod instance;
mod interaction;
mod launch;
mod logging;
#[cfg(target_os = "macos")]
mod macos_open;
mod measurement_store;
mod overlay;
mod pane_ui;
mod paths;
mod plugin_runtime;
mod plugins;
mod probe;
mod qc_cli;
mod quantify_panel;
mod recent;
mod render;
mod render_pool;
mod save_copy;
mod scripting;
mod session;
mod shortcuts;
mod slide_info;
mod slide_link;
mod state;
#[cfg(test)]
mod test_support;
mod tile_loader;
mod ui_update;
mod update;
mod version;
mod win_console;
mod zoom_display;

use anyhow::Result;
use eov_common::{RenderBackend, TileCache};
use gpu::GpuRenderer;
use parking_lot::RwLock;
use slint::{SharedString, Timer, TimerMode};
use state::{AppState, PaneId};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use std::time::Duration;
use tracing::{info, warn};

use backend::select_backend;
use cli::{apply_config_override, maybe_run_cli_command, parse_launch_options};
pub(crate) use pane_ui::{
    PaneRenderCacheEntry, PaneUiModels, clear_cached_pane, insert_pane_ui_state, pane_from_index,
    set_cached_pane_content, set_cached_pane_cpu_result, set_cached_pane_minimap,
    with_gpu_renderer, with_pane_render_cache,
};
use pane_ui::{set_gpu_renderer_handle, with_pane_ui_models, with_pane_view_model};

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

/// Refreshes tabs and panes after tabs moved, closed or panes changed; the
/// per-pane caches follow the panes, so nothing goes blank (UI-16).
fn refresh_tab_ui(ui: &AppWindow, state: &AppState) {
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
    // Before anything can change it: programs eov starts get this
    // environment back (SH-15).
    desktop::remember_launch_environment();
    // Windows GUI build: attach to the terminal for CLI use, or show fatal
    // errors in a message box (SH-2). No-op elsewhere.
    win_console::init();
    let launch_options = parse_launch_options()?;
    apply_config_override(launch_options.config_path.as_ref())?;
    logging::init(launch_options.log_level, launch_options.launches_ui());
    if launch_options.launches_ui() {
        logging::install_panic_hook();
    }

    if maybe_run_cli_command(&launch_options)? {
        return Ok(());
    }

    // Arguments that cannot be used are reported here for the terminal and
    // in the window once it is up (SH-7).
    let open_request = launch_options.open_request();
    for problem in &open_request.problems {
        eprintln!("eov: {problem}");
    }

    // One eov per user and setup (SH-8): hand the request to the running
    // eov, before this one loads plugins, opens a window or a log file.
    // An eov that does not hand over still takes the endpoint for scripts
    // when no other eov holds it.
    let mut instance_server = None;
    let forwards = launch_options.may_forward() && config::load_single_instance();
    let scripting_enabled = config::load_scripting();
    if forwards || scripting_enabled {
        let endpoint = instance::Endpoint::for_setup(
            &config::resolve_config_path()?,
            &launch_options.plugin_dir,
        );
        if forwards {
            match instance::claim(&endpoint, &open_request, instance::Timing::default()) {
                instance::Claim::Forwarded => {
                    info!("Handed the files to the running eov");
                    return Ok(());
                }
                instance::Claim::Primary(server) => instance_server = Some(server),
                instance::Claim::Standalone(reason) => warn!("{reason}"),
            }
        } else {
            match instance::Server::bind(&endpoint) {
                Ok(server) => instance_server = server,
                Err(error) => warn!("Scripts cannot reach this eov: {error}"),
            }
            if instance_server.is_none() {
                info!("Another eov serves scripts for this setup");
            }
        }
    }
    logging::start_session_log();

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

    if !open_request.opens_nothing() {
        info!(
            "Opening {} file(s) in {} pane(s) from the command line",
            open_request.tabs.len() + open_request.panes.iter().map(Vec::len).sum::<usize>(),
            1 + open_request.panes.len()
        );
    }

    // The software renderer draws the window when no GPU can (RND-4).
    let ui_renderer = select_backend(launch_options.window_geometry)
        .inspect_err(backend::report_startup_error)?;
    slint::set_xdg_app_id(APP_XDG_ID)?;

    let mut initial_state = AppState::new();
    initial_state.viewport_lock_enabled = launch_options.viewport_lock_enabled;
    let state = Arc::new(RwLock::new(initial_state));
    let tile_cache = Arc::new(TileCache::with_budget(cache_budget::init(
        launch_options.cache_size_bytes,
        launch_options.max_tiles,
    )));
    render_pool::init_global()?;

    info!(
        "Creating application window (DISPLAY={:?}, WAYLAND_DISPLAY={:?})",
        std::env::var("DISPLAY").ok(),
        std::env::var("WAYLAND_DISPLAY").ok()
    );
    let ui = AppWindow::new()
        .map_err(anyhow::Error::from)
        .inspect_err(backend::report_startup_error)?;
    info!("Application window created");
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
    info!("Installing GPU renderer");
    // Slides can be drawn on the GPU only when the window is drawn with wgpu.
    let gpu_unavailable = match &ui_renderer {
        backend::UiRenderer::Gpu => GpuRenderer::install(&ui, Rc::clone(&gpu_renderer))
            .err()
            .map(|err| format!("the window does not support GPU rendering ({err})")),
        backend::UiRenderer::Software(reason) => Some(reason.clone()),
    };
    set_gpu_renderer_handle(Rc::clone(&gpu_renderer));
    info!("GPU renderer installed");

    // Before plugins start: their hotkeys are checked against it.
    shortcuts::install_from_config();
    host_ui::key_help::apply_labels(&ui);
    let render_timer = Rc::new(Timer::default());
    let host = host_ui::init(
        &ui,
        &state,
        &tile_cache,
        &render_timer,
        host_ui::InitOptions {
            plugin_dir: launch_options.plugin_dir.clone(),
            safe_mode: launch_options.safe_mode,
        },
    );
    info!("Plugin host initialized");

    callbacks::setup_callbacks(
        &ui,
        Arc::clone(&state),
        Arc::clone(&tile_cache),
        Rc::clone(&render_timer),
    );
    info!("UI callbacks installed");

    ui.set_debug_mode(launch_options.debug_mode);
    {
        let mut state = state.write();
        state.gpu_backend_available = gpu_unavailable.is_none();
        state.select_render_backend(initial_backend);
        if let Some(filtering) = initial_filtering {
            state.select_filtering_mode(filtering);
        }
        update_render_backend(&ui, &state);
        update_filtering_mode(&ui, &state);
        if let Some(reason) = &gpu_unavailable {
            tracing::warn!("GPU renderer unavailable: {reason}");
            let message = "GPU renderer unavailable; slides are drawn by the CPU renderer";
            ui.set_status_text(SharedString::from(message));
            host_ui::with_host(|host| {
                host.notify(
                    eov_plugin_api::protocol::NotifyLevel::Warning,
                    message,
                    Some(reason),
                    None,
                )
            });
        }
    }

    {
        let state = state.read();
        update_recent_files(&ui, &state);
    }

    let problems = open_request.problems.clone();
    if open_request.opens_nothing() {
        // Reopen the last session's slides where they were (AUDIT §14.1);
        // the Home tab otherwise, or when the files given could not be
        // used.
        if !problems.is_empty() || !session::restore(&ui, &state, &tile_cache, &render_timer) {
            {
                let mut state = state.write();
                state.create_home_tab();
            }
            let state = state.read();
            update_tabs(&ui, &state);
            info!("Initialized home tab state");
        }
    } else {
        launch::open(
            &ui,
            &state,
            &tile_cache,
            &render_timer,
            launch::OpenRequest {
                problems: Vec::new(),
                ..open_request
            },
        );
    }

    request_render_loop(&render_timer, &ui_weak, &state, &tile_cache);
    info!("Initial render loop requested");

    info!("Showing application window");
    ui.show()?;
    info!("Application window shown");
    backend::warn_if_position_ignored(&ui, launch_options.window_geometry);

    host.ready();
    launch::report_problems(&problems);

    // Files handed over by eovs started from now on open here, and scripts'
    // calls are answered (the listener only queues them; the event loop
    // handles them).
    if let Some(server) = instance_server.as_mut() {
        server.serve(
            instance::Timing::default(),
            instance::Handlers {
                open: Arc::new(|request, answer| {
                    let _ = slint::invoke_from_event_loop(move || {
                        let _ = answer.send(launch::open_handed_over(request));
                    });
                }),
                script: Some(scripting::handler(
                    scripting_enabled,
                    Arc::clone(&host.shared),
                )),
            },
        );
    }
    // Slides opened from the Finder and eov:// links arrive as Apple
    // events, also while the event loop starts (SH-8).
    #[cfg(target_os = "macos")]
    macos_open::install();

    info!("Entering Slint UI event loop");

    // Save the session now and then, so a crash loses little.
    let session_timer = Timer::default();
    {
        let state = Arc::clone(&state);
        session_timer.start(TimerMode::Repeated, Duration::from_secs(60), move || {
            session::save(&state.read());
        });
    }

    ui.run()?;

    // Copies still being written are cancelled, so no partial file is left.
    save_copy::shutdown();
    // A new eov may take over while this one finishes.
    drop(instance_server);
    info!("Application shutting down");
    session::save(&state.read());
    host.shutdown();
    Ok(())
}

pub(crate) fn prepare_launch_panes(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    pane_count: usize,
) {
    if pane_count <= 1 {
        return;
    }

    let mut inserted_panes = Vec::new();
    {
        let mut state = state.write();
        while state.panes.len() < pane_count {
            let next_index = state.panes.len();
            let source_pane = PaneId(next_index.saturating_sub(1));
            let Some(new_pane) = state.insert_pane(next_index) else {
                tracing::warn!(
                    "{}; the other panes' slides open as tabs",
                    crate::state::MAX_PANES_MESSAGE
                );
                break;
            };
            inserted_panes.push((state.pane_uid(new_pane), state.pane_uid(source_pane)));
        }
        state.set_focused_pane(PaneId::PRIMARY);
    }

    for (new_pane, source_pane) in inserted_panes {
        insert_pane_ui_state(new_pane, source_pane);
    }

    let state = state.read();
    ui.set_split_enabled(state.split_enabled);
    ui.set_focused_pane(state.focused_pane.as_index());
    update_tabs(ui, &state);
}

pub(crate) fn open_file(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
) {
    open_file_with_mode(
        ui,
        state,
        tile_cache,
        render_timer,
        path,
        file_ops::OpenFileMode::ReuseExistingTab,
    );
}

/// Opens one slide of a launch (see [`file_ops::Placement`]).
pub(crate) fn open_file_placed(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    placement: file_ops::Placement,
) {
    open_file_in(
        ui,
        state,
        tile_cache,
        render_timer,
        path,
        file_ops::OpenFileMode::ReuseExistingTab,
        Some(placement),
    );
}

fn open_file_with_mode(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    mode: file_ops::OpenFileMode,
) {
    open_file_in(ui, state, tile_cache, render_timer, path, mode, None);
}

fn open_file_in(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    mode: file_ops::OpenFileMode,
    placement: Option<file_ops::Placement>,
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
                    mode,
                    placement,
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
