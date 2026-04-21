use crate::{AppWindow, open_file, request_render_loop};
use abi_stable::std_types::{ROption, RResult, RString, RVec};
use common::viewport::{MAX_ZOOM, MIN_ZOOM};
use common::{FilteringMode, RenderBackend, TileCache};
use parking_lot::RwLock;
use plugin_api::IconDescriptor;
use plugin_api::ffi::{
    ActiveSidebarFFI, HostApiVTable, HostLogLevelFFI, HostSnapshotFFI, OpenFileInfoFFI,
    ViewportSnapshotFFI,
};
use slint::{
    ComponentFactory, ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, Timer,
    VecModel,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::state::{AppState, PaneId};

struct HostApiContext {
    plugin_id: String,
    plugin_root: PathBuf,
    state: Arc<RwLock<AppState>>,
    on_ui_callback: Option<extern "C" fn(RString)>,
}

struct UiRuntime {
    ui_weak: slint::Weak<AppWindow>,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
}

static HOST_CONTEXTS: OnceLock<Mutex<HashMap<u64, HostApiContext>>> = OnceLock::new();
static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
static UI_THREAD_ID: OnceLock<std::thread::ThreadId> = OnceLock::new();

thread_local! {
    static UI_RUNTIME: RefCell<Option<UiRuntime>> = const { RefCell::new(None) };
}

fn host_contexts() -> &'static Mutex<HashMap<u64, HostApiContext>> {
    HOST_CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn init_ui_runtime(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
) {
    let _ = UI_THREAD_ID.set(std::thread::current().id());
    UI_RUNTIME.with(|slot| {
        *slot.borrow_mut() = Some(UiRuntime {
            ui_weak: ui.as_weak(),
            state: Arc::clone(state),
            tile_cache: Arc::clone(tile_cache),
            render_timer: Rc::clone(render_timer),
        });
    });
}

pub(crate) fn build_host_api(
    plugin_id: &str,
    plugin_root: &Path,
    state: &Arc<RwLock<AppState>>,
    on_ui_callback: Option<extern "C" fn(RString)>,
) -> HostApiVTable {
    let context = NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
    host_contexts().lock().unwrap().insert(
        context,
        HostApiContext {
            plugin_id: plugin_id.to_string(),
            plugin_root: plugin_root.to_path_buf(),
            state: Arc::clone(state),
            on_ui_callback,
        },
    );

    HostApiVTable {
        context,
        get_snapshot: ffi_get_snapshot,
        read_region: ffi_read_region,
        open_file: ffi_open_file,
        set_active_viewport: ffi_set_active_viewport,
        fit_active_viewport: ffi_fit_active_viewport,
        frame_active_rect: ffi_frame_active_rect,
        set_toolbar_button_active: ffi_set_toolbar_button_active,
        set_hud_toolbar_button_active: ffi_set_hud_toolbar_button_active,
        show_sidebar: ffi_show_sidebar,
        hide_sidebar: ffi_hide_sidebar,
        log_message: ffi_log_message,
    }
}

pub(crate) fn refresh_plugin_buttons() -> Result<(), String> {
    run_on_ui_thread(refresh_plugin_buttons_in_ui)
}

pub(crate) fn set_local_toolbar_button_active(
    plugin_id: &str,
    button_id: &str,
    active: bool,
) -> Result<(), String> {
    let plugin_id = plugin_id.to_string();
    let button_id = button_id.to_string();
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            let button = state
                .local_plugin_buttons
                .iter_mut()
                .find(|button| button.plugin_id == plugin_id && button.button_id == button_id)
                .ok_or_else(|| {
                    format!(
                        "local toolbar button '{}:{}' is not registered",
                        plugin_id, button_id
                    )
                })?;
            button.active = active;
        }
        refresh_plugin_buttons_in_ui(runtime)
    })
}

pub(crate) fn set_local_hud_toolbar_button_active(
    plugin_id: &str,
    button_id: &str,
    active: bool,
) -> Result<(), String> {
    let plugin_id = plugin_id.to_string();
    let button_id = button_id.to_string();
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            let button = state
                .local_hud_plugin_buttons
                .iter_mut()
                .find(|button| button.plugin_id == plugin_id && button.button_id == button_id)
                .ok_or_else(|| {
                    format!(
                        "local HUD toolbar button '{}:{}' is not registered",
                        plugin_id, button_id
                    )
                })?;
            button.active = active;
        }
        refresh_plugin_buttons_in_ui(runtime)
    })
}

pub(crate) fn request_filter_repaint() -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            state.bump_filter_revision();
        }
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn snapshot(state: &Arc<RwLock<AppState>>) -> plugin_api::HostSnapshot {
    snapshot_from_state(&state.read())
}

pub(crate) fn viewport_snapshot_for_pane(
    state: &Arc<RwLock<AppState>>,
    pane: PaneId,
) -> Option<plugin_api::ViewportSnapshot> {
    let guard = state.read();
    guard
        .active_file_id_for_pane(pane)
        .and_then(|file_id| guard.get_file(file_id))
        .and_then(|file| file.pane_state(pane))
        .map(|pane_state| to_viewport_snapshot(&pane_state.viewport, pane))
}

pub(crate) fn read_region(
    state: &Arc<RwLock<AppState>>,
    file_id: i32,
    level: u32,
    x: i64,
    y: i64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let guard = state.read();
    let file = guard
        .get_file(file_id)
        .ok_or_else(|| format!("file '{file_id}' not found"))?;
    file.wsi
        .read_region(x, y, level, width, height)
        .map_err(|err| err.to_string())
}

pub(crate) fn open_file_path(path: PathBuf) -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;
        open_file(
            &ui,
            &runtime.state,
            &runtime.tile_cache,
            &runtime.render_timer,
            path,
        );
        Ok(())
    })
}

pub(crate) fn set_active_viewport(center_x: f64, center_y: f64, zoom: f64) -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            let viewport = state
                .active_viewport_mut()
                .ok_or_else(|| "no active viewport".to_string())?;
            viewport.set_center_zoom(center_x, center_y, zoom.clamp(MIN_ZOOM, MAX_ZOOM));
        }
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn fit_active_viewport() -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            let viewport = state
                .active_viewport_mut()
                .ok_or_else(|| "no active viewport".to_string())?;
            viewport.smooth_fit_to_view();
        }
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn frame_active_rect(x: f64, y: f64, width: f64, height: f64) -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            let viewport = state
                .active_viewport_mut()
                .ok_or_else(|| "no active viewport".to_string())?;
            viewport.smooth_frame_rect(x, y, width, height);
        }
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn log_message(plugin_id: &str, level: plugin_api::HostLogLevel, message: &str) {
    let message = format!("plugin[{plugin_id}]: {message}");
    match level {
        plugin_api::HostLogLevel::Trace => tracing::trace!("{message}"),
        plugin_api::HostLogLevel::Debug => tracing::debug!("{message}"),
        plugin_api::HostLogLevel::Info => tracing::info!("{message}"),
        plugin_api::HostLogLevel::Warn => tracing::warn!("{message}"),
        plugin_api::HostLogLevel::Error => tracing::error!("{message}"),
    }
}

pub(crate) fn show_sidebar(
    plugin_id: &str,
    plugin_root: Option<&Path>,
    on_ui_callback: Option<extern "C" fn(RString)>,
    request: plugin_api::SidebarRequest,
) -> Result<(), String> {
    let resolved_request = resolve_sidebar_request(plugin_root, request)?;
    let plugin_id = plugin_id.to_string();
    run_on_ui_thread(move |runtime| {
        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;

        let should_hide = {
            let state = runtime.state.read();
            state.has_matching_sidebar_request(&plugin_id, &resolved_request)
        };

        let factory = if should_hide {
            None
        } else {
            Some(build_sidebar_factory(
                &plugin_id,
                &resolved_request.ui_path,
                &resolved_request.component,
                on_ui_callback,
            )?)
        };

        {
            let mut state = runtime.state.write();
            let previous_button = state
                .active_sidebar_button()
                .map(|(owner, button)| (owner.to_string(), button.to_string()));

            if should_hide {
                state.clear_active_sidebar();
                if let Some((owner, button)) = previous_button {
                    set_toolbar_button_active_in_state(&mut state, &owner, &button, false);
                }
                ui.set_plugin_sidebar_factory(ComponentFactory::default());
                ui.set_show_plugin_sidebar(false);
                ui.set_plugin_sidebar_width(0.0);
            } else {
                if let Some((owner, button)) = previous_button {
                    set_toolbar_button_active_in_state(&mut state, &owner, &button, false);
                }
                if let Some(button_id) = resolved_request.button_id.as_deref() {
                    set_toolbar_button_active_in_state(&mut state, &plugin_id, button_id, true);
                }
                state.set_sidebar_from_request(plugin_id.clone(), resolved_request.clone());
                ui.set_plugin_sidebar_factory(factory.expect("sidebar factory missing"));
                ui.set_show_plugin_sidebar(true);
                ui.set_plugin_sidebar_width(resolved_request.width_px as f32);
            }
        }

        refresh_plugin_buttons_in_ui(runtime)?;
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn hide_sidebar(plugin_id: &str) -> Result<(), String> {
    let plugin_id = plugin_id.to_string();
    run_on_ui_thread(move |runtime| {
        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;
        let mut state = runtime.state.write();
        let Some((owner, button_id)) = state
            .active_sidebar_button()
            .map(|(owner, button)| (owner.to_string(), button.to_string()))
        else {
            return Ok(());
        };
        if owner != plugin_id {
            return Ok(());
        }

        state.clear_active_sidebar();
        set_toolbar_button_active_in_state(&mut state, &owner, &button_id, false);
        ui.set_plugin_sidebar_factory(ComponentFactory::default());
        ui.set_show_plugin_sidebar(false);
        ui.set_plugin_sidebar_width(0.0);

        refresh_plugin_buttons_in_ui(runtime)?;
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

fn run_on_ui_thread<R: Send + 'static>(
    f: impl FnOnce(&UiRuntime) -> Result<R, String> + Send + 'static,
) -> Result<R, String> {
    if UI_THREAD_ID
        .get()
        .is_some_and(|thread_id| *thread_id == std::thread::current().id())
    {
        return UI_RUNTIME.with(|slot| {
            let runtime = slot.borrow();
            let runtime = runtime
                .as_ref()
                .ok_or_else(|| "UI runtime is not initialized".to_string())?;
            f(runtime)
        });
    }

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slint::invoke_from_event_loop(move || {
        let result = UI_RUNTIME.with(|slot| {
            let runtime = slot.borrow();
            let runtime = runtime
                .as_ref()
                .ok_or_else(|| "UI runtime is not initialized".to_string())?;
            f(runtime)
        });
        let _ = tx.send(result);
    })
    .map_err(|err| format!("failed to schedule UI task: {err}"))?;

    rx.recv()
        .map_err(|err| format!("failed to receive UI task result: {err}"))?
}

fn snapshot_from_state(state: &AppState) -> plugin_api::HostSnapshot {
    let focused_pane = state.focused_pane;
    let active_file = state
        .active_file_id_for_pane(focused_pane)
        .and_then(|file_id| state.get_file(file_id))
        .map(to_open_file_info);
    let active_viewport = state
        .active_file_id_for_pane(focused_pane)
        .and_then(|file_id| state.get_file(file_id))
        .and_then(|file| file.pane_state(focused_pane))
        .map(|pane_state| to_viewport_snapshot(&pane_state.viewport, focused_pane));

    plugin_api::HostSnapshot {
        app_name: "eov".to_string(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        render_backend: render_backend_label(state.render_backend).to_string(),
        filtering_mode: filtering_mode_label(state.filtering_mode).to_string(),
        split_enabled: state.split_enabled,
        focused_pane: focused_pane.0 as u32,
        open_files: state.open_files.iter().map(to_open_file_info).collect(),
        active_file,
        active_viewport,
        recent_files: state
            .recent_files
            .iter()
            .map(|file| file.path.to_string_lossy().into_owned())
            .collect(),
        active_sidebar: state.active_sidebar().cloned(),
    }
}

fn to_open_file_info(file: &crate::state::OpenFile) -> plugin_api::OpenFileInfo {
    let props = file.wsi.properties();
    plugin_api::OpenFileInfo {
        file_id: file.id,
        path: props.path.to_string_lossy().into_owned(),
        filename: props.filename.clone(),
        width: props.width,
        height: props.height,
        level_count: props.levels.len() as u32,
        vendor: props.vendor.clone(),
        mpp_x: props.mpp_x,
        mpp_y: props.mpp_y,
        objective_power: props.objective_power,
        scan_date: props.scan_date.clone(),
    }
}

fn to_viewport_snapshot(
    viewport: &common::ViewportState,
    pane: PaneId,
) -> plugin_api::ViewportSnapshot {
    let bounds = viewport.viewport.bounds();
    plugin_api::ViewportSnapshot {
        pane_index: pane.0 as u32,
        center_x: viewport.viewport.center.x,
        center_y: viewport.viewport.center.y,
        zoom: viewport.viewport.zoom,
        width: viewport.viewport.width,
        height: viewport.viewport.height,
        image_width: viewport.viewport.image_width,
        image_height: viewport.viewport.image_height,
        bounds_left: bounds.left,
        bounds_top: bounds.top,
        bounds_right: bounds.right,
        bounds_bottom: bounds.bottom,
    }
}

fn render_backend_label(backend: RenderBackend) -> &'static str {
    match backend {
        RenderBackend::Cpu => "cpu",
        RenderBackend::Gpu => "gpu",
    }
}

fn filtering_mode_label(mode: FilteringMode) -> &'static str {
    match mode {
        FilteringMode::Bilinear => "bilinear",
        FilteringMode::Trilinear => "trilinear",
        FilteringMode::Lanczos3 => "lanczos3",
    }
}

fn image_from_icon_descriptor(icon: &IconDescriptor) -> Image {
    match icon {
        IconDescriptor::Svg { data } => image_from_svg(data),
        IconDescriptor::File { path } => {
            Image::load_from_path(path).unwrap_or_else(|_| empty_image())
        }
    }
}

fn refresh_plugin_buttons_in_ui(runtime: &UiRuntime) -> Result<(), String> {
    let ui = runtime
        .ui_weak
        .upgrade()
        .ok_or_else(|| "application window is no longer available".to_string())?;
    let state = runtime.state.read();
    let remote_buttons = {
        let extension_state = state.extension_host_state.read();
        (
            extension_state.toolbar_buttons.clone(),
            extension_state.hud_toolbar_buttons.clone(),
        )
    };
    let remote_toolbar_keys: HashSet<(String, String)> = remote_buttons
        .0
        .iter()
        .map(|button| (button.plugin_id.clone(), button.button_id.clone()))
        .collect();
    let remote_hud_toolbar_keys: HashSet<(String, String)> = remote_buttons
        .1
        .iter()
        .map(|button| (button.plugin_id.clone(), button.button_id.clone()))
        .collect();
    let buttons: Vec<crate::PluginButtonData> = state
        .local_plugin_buttons
        .iter()
        .filter(|button| {
            !remote_toolbar_keys.contains(&(button.plugin_id.clone(), button.button_id.clone()))
        })
        .map(|button| crate::PluginButtonData {
            plugin_id: button.plugin_id.clone().into(),
            button_id: button.button_id.clone().into(),
            tooltip: button.tooltip.clone().into(),
            icon: image_from_icon_descriptor(&button.icon),
            action_id: button.action_id.clone().into(),
            active: button.active,
        })
        .chain(
            remote_buttons
                .0
                .into_iter()
                .map(|button| crate::PluginButtonData {
                    plugin_id: button.plugin_id.into(),
                    button_id: button.button_id.into(),
                    tooltip: button.tooltip.into(),
                    icon: image_from_svg(&button.icon_svg),
                    action_id: button.action_id.into(),
                    active: button.active,
                }),
        )
        .collect();
    let hud_buttons: Vec<crate::HudToolbarButtonData> = state
        .local_hud_plugin_buttons
        .iter()
        .filter(|button| {
            !remote_hud_toolbar_keys.contains(&(button.plugin_id.clone(), button.button_id.clone()))
        })
        .map(|button| crate::HudToolbarButtonData {
            plugin_id: button.plugin_id.clone().into(),
            button_id: button.button_id.clone().into(),
            tooltip: button.tooltip.clone().into(),
            icon: image_from_icon_descriptor(&button.icon),
            action_id: button.action_id.clone().into(),
            active: button.active,
        })
        .chain(
            remote_buttons
                .1
                .into_iter()
                .map(|button| crate::HudToolbarButtonData {
                    plugin_id: button.plugin_id.into(),
                    button_id: button.button_id.into(),
                    tooltip: button.tooltip.into(),
                    icon: image_from_svg(&button.icon_svg),
                    action_id: button.action_id.into(),
                    active: button.active,
                }),
        )
        .collect();
    ui.set_plugin_buttons(ModelRc::from(std::rc::Rc::new(VecModel::from(buttons))));
    ui.set_plugin_hud_buttons(ModelRc::from(std::rc::Rc::new(VecModel::from(hud_buttons))));
    Ok(())
}

fn resolve_sidebar_request(
    plugin_root: Option<&Path>,
    mut request: plugin_api::SidebarRequest,
) -> Result<plugin_api::SidebarRequest, String> {
    if request.width_px == 0 {
        return Err("sidebar width must be greater than zero".to_string());
    }

    let ui_path = PathBuf::from(&request.ui_path);
    let resolved_path = if ui_path.is_absolute() {
        ui_path
    } else {
        let root = plugin_root.ok_or_else(|| {
            format!(
                "sidebar ui path '{}' is relative but no plugin root is available",
                request.ui_path
            )
        })?;
        root.join(ui_path)
    };

    request.ui_path = resolved_path.to_string_lossy().into_owned();
    Ok(request)
}

fn build_sidebar_factory(
    plugin_id: &str,
    ui_path: &str,
    component: &str,
    on_ui_callback: Option<extern "C" fn(RString)>,
) -> Result<ComponentFactory, String> {
    let ui_path = PathBuf::from(ui_path);
    let source = std::fs::read_to_string(&ui_path)
        .map_err(|err| format!("failed to read sidebar UI {}: {err}", ui_path.display()))?;

    let compiler = slint_interpreter::Compiler::default();
    let result = spin_on(compiler.build_from_source(source, ui_path.clone()));
    let mut has_errors = false;
    for diag in result.diagnostics() {
        if diag.level() == slint_interpreter::DiagnosticLevel::Error {
            has_errors = true;
            tracing::error!("Sidebar Slint compile error for '{}': {diag}", plugin_id);
        }
    }
    if has_errors {
        return Err(format!(
            "failed to compile sidebar UI '{}' for plugin '{}'",
            ui_path.display(),
            plugin_id
        ));
    }

    let definition = result.component(component).ok_or_else(|| {
        format!(
            "component '{}' not found in sidebar UI '{}' for plugin '{}'",
            component,
            ui_path.display(),
            plugin_id
        )
    })?;
    let plugin_id = plugin_id.to_string();

    Ok(ComponentFactory::new(move |ctx| {
        let instance: slint_interpreter::ComponentInstance = match definition.create_embedded(ctx) {
            Ok(instance) => instance,
            Err(err) => {
                tracing::error!(
                    "Failed to create embedded sidebar component for '{}': {err}",
                    plugin_id
                );
                return None;
            }
        };

        if let Some(on_ui_callback) = on_ui_callback {
            for name in definition.callbacks() {
                let callback_name = name.to_string();
                if instance
                    .set_callback(&name, move |_args| {
                        (on_ui_callback)(RString::from(callback_name.as_str()));
                        slint_interpreter::Value::Void
                    })
                    .is_err()
                {
                    tracing::info!(
                        "Could not wire sidebar callback '{}' for plugin '{}'",
                        name,
                        plugin_id
                    );
                }
            }
        }

        Some(instance)
    }))
}

fn set_toolbar_button_active_in_state(
    state: &mut AppState,
    plugin_id: &str,
    button_id: &str,
    active: bool,
) {
    if let Some(button) = state
        .local_plugin_buttons
        .iter_mut()
        .find(|button| button.plugin_id == plugin_id && button.button_id == button_id)
    {
        button.active = active;
        return;
    }

    let mut extension_state = state.extension_host_state.write();
    if let Some(button) = extension_state
        .toolbar_buttons
        .iter_mut()
        .find(|button| button.plugin_id == plugin_id && button.button_id == button_id)
    {
        button.active = active;
    }
}

fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => {}
        }
    }
}

fn image_from_svg(svg: &str) -> Image {
    if svg.trim().is_empty() {
        empty_image()
    } else {
        Image::load_from_svg_data(svg.as_bytes()).unwrap_or_else(|_| empty_image())
    }
}

fn empty_image() -> Image {
    Image::from_rgba8_premultiplied(SharedPixelBuffer::<Rgba8Pixel>::new(1, 1))
}

fn to_snapshot_ffi(snapshot: plugin_api::HostSnapshot) -> HostSnapshotFFI {
    HostSnapshotFFI {
        app_name: RString::from(snapshot.app_name),
        app_version: RString::from(snapshot.app_version),
        render_backend: RString::from(snapshot.render_backend),
        filtering_mode: RString::from(snapshot.filtering_mode),
        split_enabled: snapshot.split_enabled,
        focused_pane: snapshot.focused_pane,
        open_files: snapshot
            .open_files
            .into_iter()
            .map(to_open_file_info_ffi)
            .collect(),
        active_file: snapshot.active_file.map(to_open_file_info_ffi).into(),
        active_viewport: snapshot
            .active_viewport
            .map(to_viewport_snapshot_ffi)
            .into(),
        recent_files: snapshot
            .recent_files
            .into_iter()
            .map(RString::from)
            .collect(),
        active_sidebar: snapshot.active_sidebar.map(to_active_sidebar_ffi).into(),
    }
}

fn to_active_sidebar_ffi(sidebar: plugin_api::ActiveSidebar) -> ActiveSidebarFFI {
    ActiveSidebarFFI {
        plugin_id: RString::from(sidebar.plugin_id),
        button_id: sidebar.button_id.map(RString::from).into(),
        width_px: sidebar.width_px,
        ui_path: RString::from(sidebar.ui_path),
        component: RString::from(sidebar.component),
    }
}

fn to_open_file_info_ffi(file: plugin_api::OpenFileInfo) -> OpenFileInfoFFI {
    OpenFileInfoFFI {
        file_id: file.file_id,
        path: RString::from(file.path),
        filename: RString::from(file.filename),
        width: file.width,
        height: file.height,
        level_count: file.level_count,
        vendor: file.vendor.map(RString::from).into(),
        mpp_x: file.mpp_x.into(),
        mpp_y: file.mpp_y.into(),
        objective_power: file.objective_power.into(),
        scan_date: file.scan_date.map(RString::from).into(),
    }
}

fn to_viewport_snapshot_ffi(viewport: plugin_api::ViewportSnapshot) -> ViewportSnapshotFFI {
    ViewportSnapshotFFI {
        pane_index: viewport.pane_index,
        center_x: viewport.center_x,
        center_y: viewport.center_y,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        image_width: viewport.image_width,
        image_height: viewport.image_height,
        bounds_left: viewport.bounds_left,
        bounds_top: viewport.bounds_top,
        bounds_right: viewport.bounds_right,
        bounds_bottom: viewport.bounds_bottom,
    }
}

fn host_log_level(level: HostLogLevelFFI) -> plugin_api::HostLogLevel {
    match level {
        HostLogLevelFFI::Trace => plugin_api::HostLogLevel::Trace,
        HostLogLevelFFI::Debug => plugin_api::HostLogLevel::Debug,
        HostLogLevelFFI::Info => plugin_api::HostLogLevel::Info,
        HostLogLevelFFI::Warn => plugin_api::HostLogLevel::Warn,
        HostLogLevelFFI::Error => plugin_api::HostLogLevel::Error,
    }
}

fn context_state(context: u64) -> Result<Arc<RwLock<AppState>>, RString> {
    host_contexts()
        .lock()
        .unwrap()
        .get(&context)
        .map(|ctx| Arc::clone(&ctx.state))
        .ok_or_else(|| RString::from(format!("unknown host API context '{context}'")))
}

fn context_plugin_id(context: u64) -> Option<String> {
    host_contexts()
        .lock()
        .unwrap()
        .get(&context)
        .map(|ctx| ctx.plugin_id.clone())
}

fn context_plugin_root(context: u64) -> Option<PathBuf> {
    host_contexts()
        .lock()
        .unwrap()
        .get(&context)
        .map(|ctx| ctx.plugin_root.clone())
}

fn context_ui_callback(context: u64) -> Option<extern "C" fn(RString)> {
    host_contexts()
        .lock()
        .unwrap()
        .get(&context)
        .and_then(|ctx| ctx.on_ui_callback)
}

extern "C" fn ffi_get_snapshot(context: u64) -> HostSnapshotFFI {
    match context_state(context) {
        Ok(state) => to_snapshot_ffi(snapshot(&state)),
        Err(_) => HostSnapshotFFI {
            app_name: RString::from("eov"),
            app_version: RString::from(env!("CARGO_PKG_VERSION")),
            render_backend: RString::from("unknown"),
            filtering_mode: RString::from("unknown"),
            split_enabled: false,
            focused_pane: 0,
            open_files: RVec::new(),
            active_file: ROption::RNone,
            active_viewport: ROption::RNone,
            recent_files: RVec::new(),
            active_sidebar: ROption::RNone,
        },
    }
}

extern "C" fn ffi_read_region(
    context: u64,
    file_id: i32,
    level: u32,
    x: i64,
    y: i64,
    width: u32,
    height: u32,
) -> RResult<RVec<u8>, RString> {
    let state = match context_state(context) {
        Ok(state) => state,
        Err(err) => return RResult::RErr(err),
    };
    match read_region(&state, file_id, level, x, y, width, height) {
        Ok(data) => RResult::ROk(RVec::from(data)),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_open_file(context: u64, path: RString) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match open_file_path(PathBuf::from(path.as_str())) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_set_active_viewport(
    context: u64,
    center_x: f64,
    center_y: f64,
    zoom: f64,
) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match set_active_viewport(center_x, center_y, zoom) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_fit_active_viewport(context: u64) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match fit_active_viewport() {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_frame_active_rect(
    context: u64,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match frame_active_rect(x, y, width, height) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_set_toolbar_button_active(
    context: u64,
    button_id: RString,
    active: bool,
) -> RResult<(), RString> {
    let Some(plugin_id) = context_plugin_id(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    match set_local_toolbar_button_active(&plugin_id, button_id.as_str(), active) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_set_hud_toolbar_button_active(
    context: u64,
    button_id: RString,
    active: bool,
) -> RResult<(), RString> {
    let Some(plugin_id) = context_plugin_id(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    match set_local_hud_toolbar_button_active(&plugin_id, button_id.as_str(), active) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_show_sidebar(
    context: u64,
    button_id: RString,
    width_px: u32,
    ui_path: RString,
    component: RString,
) -> RResult<(), RString> {
    let Some(plugin_id) = context_plugin_id(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    let request = plugin_api::SidebarRequest {
        button_id: (!button_id.is_empty()).then(|| button_id.to_string()),
        width_px,
        ui_path: ui_path.to_string(),
        component: component.to_string(),
    };
    match show_sidebar(
        &plugin_id,
        context_plugin_root(context).as_deref(),
        context_ui_callback(context),
        request,
    ) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_hide_sidebar(context: u64) -> RResult<(), RString> {
    let Some(plugin_id) = context_plugin_id(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    match hide_sidebar(&plugin_id) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_log_message(context: u64, level: HostLogLevelFFI, message: RString) {
    if let Some(plugin_id) = context_plugin_id(context) {
        log_message(&plugin_id, host_log_level(level), message.as_str());
    }
}
