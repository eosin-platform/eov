// Prevent console window in addition to Slint window in Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod blitter;
mod callbacks;
mod cli;
mod clipboard;
mod config;
mod extension_host;
mod file_ops;
mod gpu;
mod gpu_interop;
mod pane_ui;
mod plugin_host;
mod plugins;
mod render;
mod render_pool;
mod stain;
mod state;
mod tile_loader;
mod tools;
mod ui_update;
mod viewport_filter;

/// Compiled gRPC proto module.
pub(crate) mod eov_extension {
    tonic::include_proto!("eov.extension");
}

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

fn restore_persisted_sidebar(plugin_manager: &plugins::PluginManager) {
    let Ok(Some(sidebar)) = config::load_active_sidebar() else {
        return;
    };
    let Some(vtable) = plugin_manager
        .loaded_vtables()
        .find(|(plugin_id, _)| *plugin_id == sidebar.plugin_id)
        .map(|(_, vtable)| *vtable)
    else {
        return;
    };
    let Some(descriptor) = plugin_manager.descriptor(&sidebar.plugin_id) else {
        return;
    };

    if let Err(err) = plugin_host::show_sidebar(
        &sidebar.plugin_id,
        Some(&descriptor.root),
        Some(vtable),
        plugin_api::SidebarRequest {
            button_id: sidebar.button_id.clone(),
            width_px: sidebar.width_px,
            ui_path: sidebar.ui_path.clone(),
            component: sidebar.component.clone(),
        },
    ) {
        tracing::warn!(
            "Failed to restore persisted sidebar '{}' on startup: {err}",
            sidebar.plugin_id
        );
    }
}
use std::time::{Duration, Instant};
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
const MOMENTARY_TOOL_HOLD_THRESHOLD: Duration = Duration::from_millis(120);

fn plugin_trace(message: impl AsRef<str>) {
    if std::env::var_os("EOV_PLUGIN_TRACE").is_some() {
        eprintln!("[main] {}", message.as_ref());
    }
}

fn refresh_tab_ui(ui: &AppWindow, state: &AppState) {
    reset_pane_ui_state();
    update_tabs(ui, state);
    let _ = crate::plugin_host::refresh_active_sidebar();
}

fn deactivate_active_plugin_tool_if_matching(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    plugin_id: &str,
    action_id: &str,
) -> bool {
    let should_deactivate = {
        let state = state.read();
        state.local_plugin_buttons.iter().find_map(|button| {
            (button.plugin_id == plugin_id && button.action_id == action_id)
                .then_some(button.tool_mode)
                .flatten()
        })
    };

    let Some(tool_mode) = should_deactivate else {
        return false;
    };

    {
        let mut app_state = state.write();
        let is_active = match tool_mode {
            plugin_api::HostToolMode::Navigate => {
                app_state.current_tool == crate::state::Tool::Navigate
            }
            plugin_api::HostToolMode::RegionOfInterest => {
                app_state.current_tool == crate::state::Tool::RegionOfInterest
            }
            plugin_api::HostToolMode::MeasureDistance => {
                app_state.current_tool == crate::state::Tool::MeasureDistance
            }
            plugin_api::HostToolMode::PointAnnotation => {
                app_state.current_tool == crate::state::Tool::PointAnnotation
                    && app_state.active_tool_plugin_id.as_deref() == Some(plugin_id)
            }
            plugin_api::HostToolMode::PolygonAnnotation => {
                app_state.current_tool == crate::state::Tool::PolygonAnnotation
                    && app_state.active_tool_plugin_id.as_deref() == Some(plugin_id)
            }
        };
        if !is_active {
            return false;
        }

        app_state.set_tool(crate::state::Tool::Navigate);
        plugin_host::sync_tool_button_states(&mut app_state);
    }

    let app_state = state.read();
    update_tool_state(ui, &app_state);
    let _ = plugin_host::refresh_plugin_buttons();
    true
}

fn normalize_hotkey_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

fn tool_selection_for_button(
    button: &plugin_api::ToolbarButtonRegistration,
) -> Option<state::ToolSelection> {
    let tool = match button.tool_mode? {
        plugin_api::HostToolMode::Navigate => state::Tool::Navigate,
        plugin_api::HostToolMode::RegionOfInterest => state::Tool::RegionOfInterest,
        plugin_api::HostToolMode::MeasureDistance => state::Tool::MeasureDistance,
        plugin_api::HostToolMode::PointAnnotation => state::Tool::PointAnnotation,
        plugin_api::HostToolMode::PolygonAnnotation => state::Tool::PolygonAnnotation,
    };
    Some(state::ToolSelection {
        tool,
        plugin_id: matches!(
            tool,
            state::Tool::PointAnnotation | state::Tool::PolygonAnnotation
        )
        .then(|| button.plugin_id.clone()),
    })
}

fn selection_matches_button(
    app_state: &AppState,
    button: &plugin_api::ToolbarButtonRegistration,
) -> bool {
    let Some(target) = tool_selection_for_button(button) else {
        return false;
    };
    app_state.current_tool == target.tool
        && (!matches!(
            target.tool,
            state::Tool::PointAnnotation | state::Tool::PolygonAnnotation
        ) || app_state.active_tool_plugin_id == target.plugin_id)
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

    select_backend(launch_options.window_geometry)?;
    slint::set_xdg_app_id(APP_XDG_ID)?;

    let state = Arc::new(RwLock::new(AppState::new()));
    let tile_cache = Arc::new(TileCache::with_limits(
        launch_options.max_tiles,
        launch_options.cache_size_bytes,
    ));
    render_pool::init_global()?;

    // ----- Extension host (gRPC server for external plugins) -----
    let extension_host_port = launch_options
        .extension_host_port
        .or_else(|| config::load_extension_host_port().ok().flatten());

    // Build and leak a multi-threaded Tokio runtime for the async gRPC server
    // and any other async work (export, etc.).
    let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let tokio_handle = tokio_runtime.handle().clone();

    // Store the shared handles in AppState so the render pipeline can reach them.
    {
        let mut s = state.write();
        s.tokio_handle = Some(tokio_handle.clone());
    }

    let extension_host_state = {
        let s = state.read();
        Arc::clone(&s.extension_host_state)
    };

    if let Some(port) = extension_host_port {
        let ext_state = Arc::clone(&extension_host_state);
        let app_state = Arc::clone(&state);
        tokio_handle.spawn(async move {
            if let Err(e) = extension_host::start_extension_host(port, ext_state, app_state).await {
                tracing::error!("Extension host server failed: {e}");
            }
        });
        // Set the env var so spawned plugin processes can discover the host.
        // SAFETY: called before spawning any plugin child processes.
        unsafe {
            std::env::set_var("EOV_EXTENSION_HOST", format!("grpc://localhost:{port}"));
        }
        info!("Extension host gRPC server starting on port {port}");
    }

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

        // Register FFI viewport filters into the shared filter chain.
        {
            let state = state.read();
            let mut chain = state.filter_chain.write();
            for (_plugin_id, vtable) in pm.loaded_vtables() {
                let filters = (vtable.get_viewport_filters)();
                for f in filters.iter() {
                    let wrapper = crate::viewport_filter::FfiViewportFilter::new(*vtable, f);
                    chain.register(f.filter_id.to_string(), Box::new(wrapper));
                    info!(
                        "Registered viewport filter '{}' from plugin '{}'",
                        f.filter_id, _plugin_id
                    );
                }
            }
        }

        info!(
            "Plugin system ready: {} toolbar button(s), {} HUD toolbar button(s) from {} plugin(s)",
            pm.toolbar.len(),
            pm.hud_toolbar.len(),
            pm.descriptors.len()
        );

        state.write().local_plugin_buttons = pm.toolbar.buttons().to_vec();
        state.write().local_hud_plugin_buttons = pm.hud_toolbar.buttons().to_vec();
        state.write().local_plugin_undo_redo_states = pm.undo_redo_states.clone();
        state.write().local_plugin_undo_redo_order = pm.undo_redo_order.clone();
    }
    let old_plugin_package_files = plugin_manager.borrow().old_plugin_package_files().to_vec();

    info!(
        "Creating application window (DISPLAY={:?}, WAYLAND_DISPLAY={:?})",
        std::env::var("DISPLAY").ok(),
        std::env::var("WAYLAND_DISPLAY").ok()
    );
    let ui = AppWindow::new()?;
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
    GpuRenderer::install(&ui, Rc::clone(&gpu_renderer))?;
    set_gpu_renderer_handle(Rc::clone(&gpu_renderer));
    info!("GPU renderer installed");

    let render_timer = Rc::new(Timer::default());
    plugin_host::init_ui_runtime(&ui, &state, &tile_cache, &render_timer);
    plugin_host::refresh_plugin_buttons().map_err(anyhow::Error::msg)?;
    info!("Plugin UI runtime initialized");

    {
        let pm = plugin_manager.borrow();
        for (plugin_id, vtable) in pm.loaded_vtables() {
            let Some(descriptor) = pm.descriptor(plugin_id) else {
                continue;
            };
            (vtable.set_host_api)(plugin_host::build_host_api(
                plugin_id,
                &descriptor.root,
                &state,
                *vtable,
            ));
        }
        restore_persisted_sidebar(&pm);
    }

    setup_callbacks(
        &ui,
        Arc::clone(&state),
        Arc::clone(&tile_cache),
        Rc::clone(&render_timer),
        Rc::clone(&plugin_manager),
    );
    info!("UI callbacks installed");

    ui.set_debug_mode(launch_options.debug_mode);
    {
        let mut state = state.write();
        state.gpu_backend_available = true;
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
        info!("Initialized home tab state");
    } else {
        prepare_launch_panes(&ui, &state, launch_options.panes_to_open.len());
        info!("Prepared launch panes from CLI input");
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
    info!("Initial render loop requested");

    info!("Showing application window");
    ui.show()?;
    info!("Application window shown");

    if !old_plugin_package_files.is_empty() {
        plugin_host::show_old_plugin_versions_dialog(old_plugin_package_files)
            .map_err(anyhow::Error::msg)?;
    }

    info!("Entering Slint UI event loop");

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
    callbacks::setup_callbacks(
        ui,
        state.clone(),
        tile_cache.clone(),
        render_timer.clone(),
        Rc::clone(&plugin_manager),
    );

    // Plugin button click callback — dispatch to plugin manager and open windows
    let pm = Rc::clone(&plugin_manager);
    let filter_state = Arc::clone(&state);
    let rerender_state = Arc::clone(&state);
    let rerender_timer = Rc::clone(&render_timer);
    let rerender_ui = ui.as_weak();
    let rerender_cache = Arc::clone(&tile_cache);
    ui.on_plugin_button_clicked(move |plugin_id, action_id| {
        let plugin_id = plugin_id.to_string();
        let action_id = action_id.to_string();
        info!("Plugin button clicked: {plugin_id}:{action_id}");
        plugin_trace(format!(
            "toolbar click plugin={} action={}",
            plugin_id, action_id
        ));

        let pm = Rc::clone(&pm);
        let filter_state = Arc::clone(&filter_state);
        let rerender_state = Arc::clone(&rerender_state);
        let rerender_timer = Rc::clone(&rerender_timer);
        let rerender_ui = rerender_ui.clone();
        let rerender_cache = Arc::clone(&rerender_cache);
        Timer::single_shot(Duration::from_millis(0), move || {
            plugin_trace(format!(
                "toolbar deferred start plugin={} action={}",
                plugin_id, action_id
            ));
            if let Err(err) = crate::plugin_host::dismiss_active_sidebar_popups() {
                tracing::debug!(
                    "Failed to dismiss active sidebar popups before toolbar action: {err}"
                );
            }
            if let Some(ui) = rerender_ui.upgrade()
                && deactivate_active_plugin_tool_if_matching(
                    &ui,
                    &rerender_state,
                    &plugin_id,
                    &action_id,
                )
            {
                request_render_loop(
                    &rerender_timer,
                    &rerender_ui,
                    &rerender_state,
                    &rerender_cache,
                );
                plugin_trace(format!(
                    "toolbar deferred handled as tool toggle plugin={} action={}",
                    plugin_id, action_id
                ));
                return;
            }

            let extension_host_state = {
                let state = filter_state.read();
                Arc::clone(&state.extension_host_state)
            };
            if crate::extension_host::has_remote_toolbar_action(
                &extension_host_state,
                &plugin_id,
                &action_id,
            ) {
                match crate::extension_host::dispatch_remote_toolbar_action(
                    &extension_host_state,
                    &plugin_id,
                    &action_id,
                ) {
                    Ok(true) => {
                        plugin_trace(format!(
                            "toolbar deferred handled remotely plugin={} action={}",
                            plugin_id, action_id
                        ));
                        return;
                    }
                    Ok(false) => {}
                    Err(err) => {
                        tracing::error!("Remote plugin action error: {err}");
                        plugin_trace(format!(
                            "toolbar deferred remote error plugin={} action={} err={}",
                            plugin_id, action_id, err
                        ));
                        return;
                    }
                }
            }

            let sidebar_toggle_was_active = {
                let state = rerender_state.read();
                state.active_sidebar_button() == Some((plugin_id.as_str(), action_id.as_str()))
            };

            let mut pm = pm.borrow_mut();
            plugin_trace(format!(
                "toolbar deferred calling handle_action plugin={} action={}",
                plugin_id, action_id
            ));
            match pm.handle_action(&plugin_id, &action_id) {
                Ok(plugins::ActionOutcome::RustPluginWindow { plugin_root }) => {
                    plugin_trace(format!(
                        "toolbar deferred rust window plugin={} action={}",
                        plugin_id, action_id
                    ));
                    crate::plugins::spawn_rust_plugin_window(&plugin_root);
                }
                Ok(plugins::ActionOutcome::PythonSpawn {
                    script_path,
                    plugin_root,
                }) => {
                    plugin_trace(format!(
                        "toolbar deferred python spawn plugin={} action={}",
                        plugin_id, action_id
                    ));
                    // If the Python plugin has been spawned before (and presumably
                    // registered remote filters), toggle those filters on/off rather
                    // than spawning a second instance.
                    if pm.spawned_python_plugins.contains(&plugin_id) {
                        {
                            let mut s = filter_state.write();
                            s.extension_host_state.write().toggle_all_filters();
                            s.bump_filter_revision();
                        }
                        request_render_loop(
                            &rerender_timer,
                            &rerender_ui,
                            &rerender_state,
                            &rerender_cache,
                        );
                    } else {
                        pm.spawned_python_plugins.insert(plugin_id.clone());
                        crate::plugins::spawn_python_plugin(
                            &script_path,
                            &plugin_root,
                            Some(&action_id),
                        );
                    }
                }
                Ok(plugins::ActionOutcome::Handled) => {
                    plugin_trace(format!(
                        "toolbar deferred handled locally plugin={} action={}",
                        plugin_id, action_id
                    ));
                    let (sidebar_toggle_is_active, handled_action_is_tool_button) = {
                        let state = rerender_state.read();
                        (
                            state.active_sidebar_button()
                                == Some((plugin_id.as_str(), action_id.as_str())),
                            state.local_plugin_buttons.iter().any(|button| {
                                button.plugin_id == plugin_id
                                    && button.action_id == action_id
                                    && button.tool_mode.is_some()
                            }),
                        )
                    };
                    if sidebar_toggle_was_active || sidebar_toggle_is_active {
                        plugin_trace(format!(
                            "toolbar deferred sidebar toggle complete plugin={} action={}",
                            plugin_id, action_id
                        ));
                        return;
                    }
                    if handled_action_is_tool_button {
                        plugin_trace(format!(
                            "toolbar deferred tool activation complete plugin={} action={}",
                            plugin_id, action_id
                        ));
                        return;
                    }
                    {
                        let mut s = filter_state.write();
                        pm.sync_filter_states(&s.filter_chain);
                        s.bump_filter_revision();
                    }
                    if let Some(ui) = rerender_ui.upgrade() {
                        let state = rerender_state.read();
                        update_tool_state(&ui, &state);
                    }
                    request_render_loop(
                        &rerender_timer,
                        &rerender_ui,
                        &rerender_state,
                        &rerender_cache,
                    );
                    plugin_trace(format!(
                        "toolbar deferred finished plugin={} action={}",
                        plugin_id, action_id
                    ));
                }
                Err(e) => {
                    tracing::error!("Plugin action error: {e}");
                    plugin_trace(format!(
                        "toolbar deferred action error plugin={} action={} err={}",
                        plugin_id, action_id, e
                    ));
                }
            }
        });
    });

    let pm = Rc::clone(&plugin_manager);
    let rerender_state = Arc::clone(&state);
    let rerender_timer = Rc::clone(&render_timer);
    let rerender_ui = ui.as_weak();
    let rerender_cache = Arc::clone(&tile_cache);
    ui.on_plugin_undo_requested(move |plugin_id| {
        let plugin_id = plugin_id.to_string();
        let mut pm = pm.borrow_mut();
        match pm.handle_undo(&plugin_id) {
            Ok(plugins::ActionOutcome::Handled)
            | Ok(plugins::ActionOutcome::RustPluginWindow { .. })
            | Ok(plugins::ActionOutcome::PythonSpawn { .. }) => {
                if let Some(ui) = rerender_ui.upgrade() {
                    let state = rerender_state.read();
                    update_tool_state(&ui, &state);
                    let _ = plugin_host::refresh_plugin_buttons();
                }
                request_render_loop(
                    &rerender_timer,
                    &rerender_ui,
                    &rerender_state,
                    &rerender_cache,
                );
            }
            Err(err) => tracing::error!("Plugin undo error: {err}"),
        }
    });

    let pm = Rc::clone(&plugin_manager);
    let rerender_state = Arc::clone(&state);
    let rerender_timer = Rc::clone(&render_timer);
    let rerender_ui = ui.as_weak();
    let rerender_cache = Arc::clone(&tile_cache);
    ui.on_plugin_redo_requested(move |plugin_id| {
        let plugin_id = plugin_id.to_string();
        let mut pm = pm.borrow_mut();
        match pm.handle_redo(&plugin_id) {
            Ok(plugins::ActionOutcome::Handled)
            | Ok(plugins::ActionOutcome::RustPluginWindow { .. })
            | Ok(plugins::ActionOutcome::PythonSpawn { .. }) => {
                if let Some(ui) = rerender_ui.upgrade() {
                    let state = rerender_state.read();
                    update_tool_state(&ui, &state);
                    let _ = plugin_host::refresh_plugin_buttons();
                }
                request_render_loop(
                    &rerender_timer,
                    &rerender_ui,
                    &rerender_state,
                    &rerender_cache,
                );
            }
            Err(err) => tracing::error!("Plugin redo error: {err}"),
        }
    });

    let hotkey_state = Arc::clone(&state);
    let hotkey_ui = ui.as_weak();
    ui.on_plugin_tool_hotkey_pressed(move |text, repeat| {
        if plugin_host::active_sidebar_captures_hotkeys()
            || plugin_host::active_modal_captures_hotkeys()
        {
            return false;
        }

        let Some(key) = normalize_hotkey_text(text.as_str()) else {
            return false;
        };

        if repeat {
            let mut app_state = hotkey_state.write();
            if let Some(active) = app_state.temporary_tool_override_mut(&key) {
                active.saw_repeat = true;
                return true;
            }
            return false;
        }

        let Some(button) = ({
            let app_state = hotkey_state.read();
            plugin_host::hotkey_button_for_key(&app_state, &key)
        }) else {
            return false;
        };

        let Some(target) = tool_selection_for_button(&button) else {
            return false;
        };

        let target_was_active = {
            let app_state = hotkey_state.read();
            selection_matches_button(&app_state, &button)
        };

        {
            let mut app_state = hotkey_state.write();
            let restore = (!target_was_active).then(|| app_state.current_tool_selection());
            app_state.push_temporary_tool_override(state::TemporaryToolOverride {
                hotkey: key,
                plugin_id: button.plugin_id.clone(),
                action_id: button.action_id.clone(),
                target,
                restore,
                target_was_active,
                pressed_at: Instant::now(),
                saw_repeat: false,
            });
        }

        if !target_was_active && let Some(ui) = hotkey_ui.upgrade() {
            ui.invoke_plugin_button_clicked(button.plugin_id.into(), button.action_id.into());
        }

        true
    });

    let hotkey_state = Arc::clone(&state);
    let hotkey_ui = ui.as_weak();
    let hotkey_render_state = Arc::clone(&state);
    let hotkey_render_timer = Rc::clone(&render_timer);
    let hotkey_render_cache = Arc::clone(&tile_cache);
    let hotkey_plugin_manager = Rc::clone(&plugin_manager);
    ui.on_plugin_tool_hotkey_released(move |text| {
        if plugin_host::active_sidebar_captures_hotkeys()
            || plugin_host::active_modal_captures_hotkeys()
        {
            return false;
        }

        let Some(key) = normalize_hotkey_text(text.as_str()) else {
            return false;
        };

        let Some(entry) = ({
            let mut app_state = hotkey_state.write();
            app_state.take_temporary_tool_override(&key)
        }) else {
            return false;
        };

        let was_held =
            entry.saw_repeat || entry.pressed_at.elapsed() >= MOMENTARY_TOOL_HOLD_THRESHOLD;

        if was_held {
            if entry.target.tool == state::Tool::PolygonAnnotation
                && let Err(err) = crate::callbacks::confirm_active_polygon_annotation(
                    &hotkey_state,
                    &hotkey_plugin_manager,
                )
            {
                tracing::error!("Polygon annotation confirmation error: {err}");
            }
            if let Some(restore) = entry.restore
                && let Some(ui) = hotkey_ui.upgrade()
            {
                {
                    let mut app_state = hotkey_state.write();
                    app_state.apply_tool_selection(&restore);
                    plugin_host::sync_tool_button_states(&mut app_state);
                }
                {
                    let app_state = hotkey_state.read();
                    update_tool_state(&ui, &app_state);
                }
                let _ = plugin_host::refresh_plugin_buttons();
                request_render_loop(
                    &hotkey_render_timer,
                    &hotkey_ui,
                    &hotkey_render_state,
                    &hotkey_render_cache,
                );
            }
        } else if entry.target_was_active
            && let Some(ui) = hotkey_ui.upgrade()
            && deactivate_active_plugin_tool_if_matching(
                &ui,
                &hotkey_state,
                &entry.plugin_id,
                &entry.action_id,
            )
        {
            request_render_loop(
                &hotkey_render_timer,
                &hotkey_ui,
                &hotkey_render_state,
                &hotkey_render_cache,
            );
        }

        true
    });

    ui.on_global_hotkeys_enabled(|| {
        !plugin_host::active_sidebar_captures_hotkeys()
            && !plugin_host::active_modal_captures_hotkeys()
    });

    let modal_dismiss_ui = ui.as_weak();
    ui.on_plugin_modal_dismiss_requested(move || {
        if let Err(err) = plugin_host::dismiss_active_modal_dialog() {
            tracing::error!("Failed to dismiss plugin modal: {err}");
        }
        if let Some(ui) = modal_dismiss_ui.upgrade() {
            ui.invoke_focus_keyboard();
        }
    });

    let pm = Rc::clone(&plugin_manager);
    let filter_state = Arc::clone(&state);
    ui.on_hud_plugin_button_clicked(move |pane, plugin_id, action_id| {
        let pane = PaneId(pane.max(0) as usize);
        let plugin_id = plugin_id.to_string();
        let action_id = action_id.to_string();
        info!(
            "HUD plugin button clicked: pane={} {plugin_id}:{action_id}",
            pane.0
        );

        let viewport = match crate::plugin_host::viewport_snapshot_for_pane(&filter_state, pane) {
            Some(viewport) => viewport,
            None => {
                tracing::warn!(
                    "HUD plugin button click ignored without viewport: pane={}",
                    pane.0
                );
                return;
            }
        };

        let extension_host_state = {
            let state = filter_state.read();
            Arc::clone(&state.extension_host_state)
        };
        if crate::extension_host::has_remote_hud_toolbar_action(
            &extension_host_state,
            &plugin_id,
            &action_id,
        ) {
            match crate::extension_host::dispatch_remote_hud_toolbar_action(
                &extension_host_state,
                &plugin_id,
                &action_id,
                viewport.clone(),
            ) {
                Ok(true) => return,
                Ok(false) => {}
                Err(err) => {
                    tracing::error!("Remote HUD plugin action error: {err}");
                    return;
                }
            }
        }

        let mut pm = pm.borrow_mut();
        if let Err(err) = pm.handle_hud_action(&plugin_id, &action_id, &viewport) {
            tracing::error!("Local HUD plugin action error: {err}");
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
