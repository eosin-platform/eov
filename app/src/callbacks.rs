use crate::config;
use crate::host_ui::with_host;
use crate::state::{self, AppState, HudSettings, PaneId};
use crate::ui_update::build_recent_folder_menu_items;
use crate::zoom_display::parse_zoom_input;
use crate::{
    AppWindow, FilteringMode as SlintFilteringMode, MeasurementUnit as SlintMeasurementUnit,
    RenderMode, StainNormalization as SlintStainNormalization, build_recent_menu_items,
    insert_pane_ui_state, open_file_with_mode, pane_from_index, refresh_tab_ui,
    request_render_loop, slider_value_to_zoom, update_filtering_mode, update_recent_files,
    update_render_backend, update_tabs,
};
use eov_common::viewport::ZOOM_FACTOR;
use eov_common::{FilteringMode, MeasurementUnit, RenderBackend, StainNormalization, TileCache};
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// Edge drops this close together are one multi-file drop.
const EDGE_DROP_GROUPING: Duration = Duration::from_millis(400);

thread_local! {
    /// When and into which pane (stable id) the last edge drop went.
    static LAST_EDGE_DROP: Cell<Option<(std::time::Instant, u32)>> = const { Cell::new(None) };
}

pub(crate) const ACTION_ZOOM_FACTOR: f64 = ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR
    * ZOOM_FACTOR;
#[cfg(target_os = "macos")]
const MACOS_TOOLBAR_HEIGHT: f64 = 40.0;

fn active_hud_mut(state: &mut AppState) -> Option<&mut HudSettings> {
    let pane = state.focused_pane;
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file_mut(file_id)?;
    let pane_state = file.pane_state_mut(pane)?;
    Some(&mut pane_state.hud)
}

fn measurement_unit_from_slint(unit: SlintMeasurementUnit) -> MeasurementUnit {
    match unit {
        SlintMeasurementUnit::Um => MeasurementUnit::Um,
        SlintMeasurementUnit::Mm => MeasurementUnit::Mm,
        SlintMeasurementUnit::Inches => MeasurementUnit::Inches,
    }
}

fn stain_normalization_from_slint(sn: SlintStainNormalization) -> StainNormalization {
    match sn {
        SlintStainNormalization::None => StainNormalization::None,
        SlintStainNormalization::Macenko => StainNormalization::Macenko,
        SlintStainNormalization::Vahadane => StainNormalization::Vahadane,
    }
}

/// Query the cursor position in window-local physical pixels via X11.
#[cfg(target_os = "linux")]
fn query_dnd_cursor_physical(ui: &AppWindow) -> Option<(f64, f64)> {
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
    use slint::winit_030::WinitWindowAccessor;

    ui.window().with_winit_window(|w| {
        let wh = w.window_handle().ok()?;
        let dh = w.display_handle().ok()?;

        match (wh.as_ref(), dh.as_ref()) {
            (RawWindowHandle::Xlib(wh), RawDisplayHandle::Xlib(dh)) => {
                let display = dh.display?.as_ptr();
                let window = wh.window;
                let xlib = x11_dl::xlib::Xlib::open().ok()?;
                unsafe {
                    let mut root = 0;
                    let mut child = 0;
                    let mut root_x = 0;
                    let mut root_y = 0;
                    let mut win_x = 0;
                    let mut win_y = 0;
                    let mut mask = 0;

                    let result = (xlib.XQueryPointer)(
                        display as *mut _,
                        window,
                        &mut root,
                        &mut child,
                        &mut root_x,
                        &mut root_y,
                        &mut win_x,
                        &mut win_y,
                        &mut mask,
                    );

                    if result != 0 {
                        Some((win_x as f64, win_y as f64))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        }
    })?
}

/// Query the cursor position in window-local physical pixels via Win32 API.
#[cfg(target_os = "windows")]
fn query_dnd_cursor_physical(ui: &AppWindow) -> Option<(f64, f64)> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use slint::winit_030::WinitWindowAccessor;
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::Graphics::Gdi::ScreenToClient;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;

    ui.window().with_winit_window(|w| {
        let wh = w.window_handle().ok()?;
        match wh.as_ref() {
            RawWindowHandle::Win32(wh) => {
                let hwnd = wh.hwnd.get() as isize;
                unsafe {
                    let mut pt = POINT { x: 0, y: 0 };
                    if GetCursorPos(&mut pt) == 0 {
                        return None;
                    }
                    if ScreenToClient(hwnd, &mut pt) == 0 {
                        return None;
                    }
                    Some((pt.x as f64, pt.y as f64))
                }
            }
            _ => None,
        }
    })?
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn query_dnd_cursor_physical(_ui: &AppWindow) -> Option<(f64, f64)> {
    None
}

/// Query cursor position during DnD and return logical coordinates.
fn query_dnd_cursor_logical(ui: &AppWindow) -> Option<(f32, f32)> {
    let (px, py) = query_dnd_cursor_physical(ui)?;
    let scale = ui.window().scale_factor() as f64;
    Some(((px / scale) as f32, (py / scale) as f32))
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSPoint {
    x: f64,
    y: f64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSSize {
    width: f64,
    height: f64,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NSRect {
    origin: NSPoint,
    size: NSSize,
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSPoint {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGPoint=dd}") }
    }
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSSize {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGSize=dd}") }
    }
}

#[cfg(target_os = "macos")]
unsafe impl objc::Encode for NSRect {
    fn encode() -> objc::Encoding {
        unsafe { objc::Encoding::from_str("{CGRect={CGPoint=dd}{CGSize=dd}}") }
    }
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
fn align_macos_window_controls(slint_window: &slint::Window) {
    use objc::runtime::Object;
    use objc::{sel, sel_impl};
    use slint::winit_030::WinitWindowAccessor;
    use slint::winit_030::winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    slint_window.with_winit_window(|w| {
        let Ok(window_handle) = w.window_handle() else {
            return;
        };

        let RawWindowHandle::AppKit(handle) = window_handle.as_ref() else {
            return;
        };

        let ns_view = handle.ns_view.as_ptr().cast::<Object>();

        unsafe {
            let ns_window: *mut Object = objc::msg_send![ns_view, window];
            if ns_window.is_null() {
                return;
            }

            for button_kind in [0usize, 1, 2] {
                let button: *mut Object =
                    objc::msg_send![ns_window, standardWindowButton: button_kind];
                if button.is_null() {
                    continue;
                }

                let frame: NSRect = objc::msg_send![button, frame];
                let origin = NSPoint {
                    x: frame.origin.x,
                    y: (MACOS_TOOLBAR_HEIGHT - frame.size.height) / 2.0,
                };
                let _: () = objc::msg_send![button, setFrameOrigin: origin];
            }
        }
    });
}

/// Frames the slide's region of interest, or the whole slide.
pub(crate) fn frame_active_viewport(state: &mut AppState) -> bool {
    let pane = state.focused_pane;
    let roi = state
        .active_file_id_for_pane(pane)
        .and_then(|file_id| state.get_file(file_id))
        .and_then(|file| file.roi);

    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };

    if let Some(roi) = roi {
        viewport.smooth_frame_rect(roi.x, roi.y, roi.width, roi.height);
    } else {
        viewport.smooth_fit_to_view();
    }

    true
}

/// The objective-equivalent magnification of a slide's full resolution:
/// the scanner's objective power, or about 10 µm ÷ pixel size when only
/// the pixel size is known (0.25 µm/px ≈ 40×), including a manual one.
pub(crate) fn native_magnification(file: &state::OpenFile) -> Option<f64> {
    crate::launch::native_magnification(
        file.wsi.properties().objective_power,
        crate::host_ui::calibration(file).average(),
    )
}

/// Zooms the focused pane to an objective-equivalent `magnification`
/// (`None` = one screen pixel per slide pixel). `Err` when the slide's
/// magnification is unknown.
pub(crate) fn magnify_active_viewport(
    state: &mut AppState,
    magnification: Option<f64>,
) -> Result<bool, String> {
    let pane = if state.split_enabled {
        state.focused_pane
    } else {
        PaneId::PRIMARY
    };
    let native = state
        .active_file_id_for_pane(pane)
        .and_then(|file_id| state.get_file(file_id))
        .map(native_magnification);
    let zoom = match (magnification, native) {
        (_, None) => return Ok(false),
        (None, _) => 1.0,
        (Some(target), Some(Some(native))) => target / native,
        (Some(_), Some(None)) => {
            return Err("This slide records neither its objective nor its pixel size".into());
        }
    };
    let Some(viewport) = state.active_viewport_mut() else {
        return Ok(false);
    };
    viewport.zoom_to(zoom);
    Ok(true)
}

/// Turns the focused pane's slide `quarter_turns` × 90° clockwise as seen on
/// screen (negative: counterclockwise), about the view centre; a fitted
/// view stays fitted. Level-0 coordinates are unaffected. Callers mirror
/// locked panes and request a render (as with the other `*_active_viewport`
/// actions).
pub(crate) fn rotate_active_viewport(state: &mut AppState, quarter_turns: i32) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };
    let orientation = viewport.viewport.orientation.turned(quarter_turns);
    viewport.set_orientation(orientation);
    true
}

/// Mirrors the focused pane's slide left to right as seen on screen.
pub(crate) fn flip_active_viewport(state: &mut AppState) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };
    let orientation = viewport.viewport.orientation.mirrored();
    viewport.set_orientation(orientation);
    true
}

/// Shows the focused pane's slide upright and unmirrored again.
pub(crate) fn reset_active_orientation(state: &mut AppState) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };
    let changed = !viewport.viewport.orientation.is_identity();
    viewport.set_orientation(eov_common::viewport::Orientation::IDENTITY);
    changed
}

/// `hud-orientation-changed` actions (see `hud-toolbar.slint`).
const HUD_ROTATE_LEFT: i32 = 0;
const HUD_ROTATE_RIGHT: i32 = 1;
const HUD_FLIP: i32 = 2;

pub(crate) fn zoom_active_viewport(state: &mut AppState, factor: f64) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };

    let center_x = viewport.viewport.width / 2.0;
    let center_y = viewport.viewport.height / 2.0;
    viewport.zoom_at_discrete(factor, center_x, center_y);
    true
}

pub(crate) fn toggle_minimap_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_minimap = {
        let mut state = state.write();
        state.toggle_minimap();
        state.show_minimap
    };
    ui.set_show_minimap(show_minimap);
}

fn toggle_series_bar_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_series_bar = {
        let mut state = state.write();
        state.toggle_series_bar();
        state.bottom_panel_visible && state.bottom_panel_kind == state::BottomPanelKind::Series
    };
    ui.set_show_series_bar(show_series_bar);
}

fn toggle_metadata_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_metadata = {
        let mut state = state.write();
        state.toggle_metadata();
        state.show_metadata
    };
    ui.set_show_metadata(show_metadata);
}

fn toggle_scale_bar_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_scale_bar = {
        let mut state = state.write();
        state.toggle_scale_bar();
        state.show_scale_bar
    };
    ui.set_show_scale_bar(show_scale_bar);
}

fn toggle_viewport_lock(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let viewport_lock_enabled = {
        let mut state = state.write();
        state.toggle_viewport_lock();
        state.viewport_lock_enabled
    };
    ui.set_viewport_lock_enabled(viewport_lock_enabled);
}

pub(crate) fn toggle_series_bar_requested(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
) {
    toggle_series_bar_visibility(ui, state);
    let mut initialized_home = false;
    if state.read().bottom_panel_visible {
        initialized_home = state.read().current_series_path().is_none();
        ensure_series_home_if_needed(ui, state);
        crate::file_ops::ensure_series_thumbnails(ui, state);
    }
    if !initialized_home {
        let state = state.read();
        update_tabs(ui, &state);
    }
    request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
}

fn open_path_with_series(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    mode: crate::file_ops::OpenFileMode,
) {
    let normalized_path = crate::paths::canonical(&path);
    let skip_series_sync = normalized_path.parent().is_some_and(|parent| {
        let guard = state.read();
        guard.find_tab_by_path(&normalized_path).is_some()
            && guard
                .current_series_path()
                .is_some_and(|current| current == parent)
    });

    if !skip_series_sync {
        crate::file_ops::set_series_from_file_parent(ui, state, &normalized_path);
    }
    open_file_with_mode(ui, state, tile_cache, render_timer, normalized_path, mode);
}

fn focus_existing_file(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: &std::path::Path,
) -> bool {
    let normalized_path = crate::paths::canonical(path);
    let Some((pane, tab_id)) = state.read().find_tab_by_path(&normalized_path) else {
        return false;
    };

    {
        let mut state = state.write();
        state.activate_tab_in_pane(pane, tab_id);
        state.add_to_recent(&normalized_path);
    }

    {
        let state = state.read();
        update_tabs(ui, &state);
    }

    ui.set_is_loading(false);
    ui.set_status_text(SharedString::from(format!(
        "Focused {}",
        normalized_path.display()
    )));
    request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
    true
}

fn open_existing_file_in_new_tab(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: &std::path::Path,
) -> bool {
    let normalized_path = crate::paths::canonical(path);
    let Some((_, tab_id)) = state.read().find_tab_by_path(&normalized_path) else {
        return false;
    };

    {
        let mut state = state.write();
        let target_pane = if state.split_enabled {
            state.focused_pane
        } else {
            crate::state::PaneId::PRIMARY
        };
        let duplicated_id = state.duplicate_tab_to_pane(tab_id, target_pane);
        state.activate_tab_in_pane(target_pane, duplicated_id);
        state.add_to_recent(&normalized_path);
    }

    {
        let state = state.read();
        update_tabs(ui, &state);
    }

    ui.set_is_loading(false);
    ui.set_status_text(SharedString::from(format!(
        "Opened new tab for {}",
        normalized_path.display()
    )));
    request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
    true
}

fn default_series_home() -> Option<PathBuf> {
    dirs::home_dir().or_else(|| std::env::current_dir().ok())
}

fn navigate_series_to_folder(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    folder: PathBuf,
    navigation: state::SeriesNavigationMode,
) {
    crate::file_ops::set_series_from_folder(ui, state, folder, navigation, false);
}

fn ensure_series_home_if_needed(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let should_initialize = state.read().current_series_path().is_none();
    if should_initialize && let Some(home) = default_series_home() {
        let home = crate::paths::canonical(&home);
        let revision = {
            let mut state = state.write();
            state.set_opened_series(
                home.clone(),
                Vec::new(),
                state::SeriesNavigationMode::Initialize,
            )
        };

        {
            let state = state.read();
            update_tabs(ui, &state);
        }

        crate::file_ops::load_series_entries_async(
            ui.as_weak(),
            Arc::clone(state),
            home,
            revision,
            false,
        );
    }
}

pub(crate) fn activate_series_entry_path(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
) {
    let path = crate::paths::canonical(&path);
    if !path.exists() {
        return;
    }

    if path.is_dir() {
        navigate_series_to_folder(ui, state, path, state::SeriesNavigationMode::Push);
    } else if !focus_existing_file(ui, state, tile_cache, render_timer, &path) {
        open_path_with_series(
            ui,
            state,
            tile_cache,
            render_timer,
            path,
            crate::file_ops::OpenFileMode::ReuseExistingTab,
        );
    }
}

/// Selects the next (or previous) entry of the series bar and opens it.
pub(crate) fn cycle_series(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    offset: i32,
) {
    let path = state.write().cycle_series_selection(offset);
    if let Some(path) = path {
        activate_series_entry_path(ui, state, tile_cache, render_timer, path);
    }
}

/// Opens a recently used slide or folder. A missing file is reported by
/// the open itself; the entry stays (its drive may just be unmounted).
fn open_recent(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    entry: crate::recent::RecentFile,
) {
    if entry.is_dir {
        open_folder_as_series(ui, state, entry.path);
    } else {
        open_path_with_series(
            ui,
            state,
            tile_cache,
            render_timer,
            entry.path,
            crate::file_ops::OpenFileMode::ReuseExistingTab,
        );
    }
}

pub(crate) fn open_folder_as_series(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    folder: PathBuf,
) {
    let normalized_folder = crate::paths::canonical(&folder);
    {
        let mut state = state.write();
        state.set_bottom_panel_visible(true, state::BottomPanelKind::Series);
        state.add_to_recent(&normalized_folder);
    }
    {
        let state = state.read();
        update_recent_files(ui, &state);
    }

    let navigation = if state.read().current_series_path().is_some() {
        state::SeriesNavigationMode::Push
    } else {
        state::SeriesNavigationMode::Initialize
    };
    // The folder is listed in the background; its first slide opens then.
    crate::file_ops::set_series_from_folder(ui, state, normalized_folder, navigation, true);
}

fn apply_filtering_mode(
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    ui_weak: &slint::Weak<AppWindow>,
    mode: FilteringMode,
) {
    {
        let mut state = state.write();
        state.select_filtering_mode(mode);
        if let Err(err) = config::save_filtering_mode(state.filtering_mode) {
            warn!("Failed to save filtering mode config: {}", err);
        }
    }
    if let Some(ui) = ui_weak.upgrade() {
        let state = state.read();
        update_filtering_mode(&ui, &state);
    }
    if let Some(ui) = ui_weak.upgrade() {
        request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
    }
}

pub fn setup_callbacks(
    ui: &AppWindow,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
) {
    let ui_weak = ui.as_weak();

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_file_requested(move || {
            let ui = ui_weak.upgrade().unwrap();

            let dialog = FileDialog::new()
                .add_filter("WSI Files", eov_common::dataset::SUPPORTED_SLIDE_EXTENSIONS)
                .add_filter("All Files", &["*"]);

            if let Some(path) = dialog.pick_file() {
                open_path_with_series(
                    &ui,
                    &state,
                    &tile_cache,
                    &render_timer,
                    path,
                    crate::file_ops::OpenFileMode::ReuseExistingTab,
                );
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_open_folder_requested(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if let Some(folder) = FileDialog::new().pick_folder() {
                open_folder_as_series(&ui, &state_handle, folder);
            }
        });
    }

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
                refresh_tab_ui(&ui, &state);
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_recent_file(move |path_str| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            // The home screen shows display strings; map back to the entry.
            let entry = state
                .read()
                .recent_files
                .iter()
                .find(|entry| entry.path.display().to_string() == path_str.as_str())
                .cloned()
                .unwrap_or_else(|| {
                    let path = PathBuf::from(path_str.as_str());
                    let is_dir = path.is_dir();
                    crate::recent::RecentFile::new(path, is_dir)
                });
            open_recent(&ui, &state, &tile_cache, &render_timer, entry);
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_interaction_started(move |_pane| {
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

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

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_series_open_folder_menu_requested(move |x, y| {
            if let Some(ui) = ui_weak.upgrade() {
                let items = {
                    let state = state_handle.read();
                    build_recent_folder_menu_items(&state)
                };
                ui.set_context_menu_items(Rc::new(VecModel::from(items)).into());
                ui.set_context_menu_tab_id(-1);
                ui.set_drag_source_pane(-1);
                ui.set_context_menu_series_path(SharedString::new());
                ui.set_context_menu_x(x);
                ui.set_context_menu_y(y);
                ui.set_context_menu_visible(true);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_scale_bar_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_scale_bar_visibility(&ui, &state_handle);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_viewport_lock_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_viewport_lock(&ui, &state_handle);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_context_menu_command(move |id| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            ui.set_context_menu_visible(false);

            let command = id.to_string();
            if let Some(index) = command.strip_prefix("recent-file:") {
                let entry = index
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| state_handle.read().recent_files.get(index).cloned());
                if let Some(entry) = entry {
                    open_recent(&ui, &state_handle, &tile_cache, &render_timer, entry);
                }
                return;
            }
            if command == "recent-clear" {
                state_handle.write().clear_recent();
                update_recent_files(&ui, &state_handle.read());
                return;
            }

            if command == "open-folder-dialog" {
                if let Some(folder) = FileDialog::new().pick_folder() {
                    open_folder_as_series(&ui, &state_handle, folder);
                }
                return;
            }

            let pane = pane_from_index(ui.get_drag_source_pane());
            let tab_id = ui.get_context_menu_tab_id();

            if with_host(|host| host.context_menu_command(&command)) == Some(true) {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                return;
            }

            match command.as_str() {
                "viewport-copy-image" => {
                    crate::export_dialog::copy_view(&ui, &state_handle, &tile_cache, pane);
                }
                "viewport-export-image" => {
                    crate::export_dialog::open(&ui, &state_handle, &tile_cache, pane);
                }
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
                    let path = state_handle
                        .read()
                        .get_file(tab_id)
                        .map(|file| file.path.clone());
                    if let Some(path) = path {
                        crate::desktop::reveal(path);
                    }
                }
                "copy-path" => {
                    let state = state_handle.read();
                    if let Some(file) = state.get_file(tab_id) {
                        crate::clipboard::copy_text(file.path.display().to_string());
                    }
                }
                "split-right" => {
                    let split = {
                        let mut state = state_handle.write();
                        let source_pane = pane;
                        state.insert_pane(source_pane.0 + 1).map(|new_pane| {
                            state.duplicate_tab_to_pane(tab_id, new_pane);
                            state.set_focused_pane(new_pane);
                            state.request_render();
                            (
                                state.split_enabled,
                                state.pane_uid(source_pane),
                                state.pane_uid(new_pane),
                            )
                        })
                    };
                    let Some((split_enabled, source_uid, new_uid)) = split else {
                        ui.set_status_text(crate::state::MAX_PANES_MESSAGE.into());
                        return;
                    };
                    insert_pane_ui_state(new_uid, source_uid);
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

            // If switching to CPU while Lanczos is active, show confirmation dialog
            let has_lanczos = {
                let state = state_handle.read();
                matches!(state.filtering_mode, FilteringMode::Lanczos3)
            };
            if backend == RenderBackend::Cpu && has_lanczos {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_lanczos_confirm_from_backend_switch(true);
                    ui.set_lanczos_confirm_visible(true);
                }
                return;
            }

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

    // Filtering mode selection with Lanczos confirmation dialog (CPU only)
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_filtering_mode_selected(move |slint_mode| {
            let mode = match slint_mode {
                SlintFilteringMode::Bilinear => FilteringMode::Bilinear,
                SlintFilteringMode::Trilinear => FilteringMode::Trilinear,
                SlintFilteringMode::Lanczos3 => FilteringMode::Lanczos3,
            };

            let (current_mode, is_cpu) = {
                let state = state_handle.read();
                (
                    state.filtering_mode,
                    state.render_backend == RenderBackend::Cpu,
                )
            };
            let is_lanczos = matches!(mode, FilteringMode::Lanczos3);
            let was_lanczos = matches!(current_mode, FilteringMode::Lanczos3);

            // If switching into Lanczos on CPU, show in-window confirmation dialog
            if is_lanczos && !was_lanczos && is_cpu {
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_lanczos_confirm_visible(true);
                }
            } else {
                apply_filtering_mode(&state_handle, &tile_cache, &render_timer, &ui_weak, mode);
            }
        });
    }

    // Lanczos confirmation dialog accepted — keep Lanczos on CPU
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_accepted(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if from_backend_switch {
                // Switch to CPU and keep Lanczos
                {
                    let mut state = state_handle.write();
                    state.select_render_backend(RenderBackend::Cpu);
                    if let Err(err) = config::save_render_backend(state.render_backend) {
                        warn!("Failed to save render backend config: {}", err);
                    }
                }
                if let Some(ui) = ui_weak.upgrade() {
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                }
                if let Some(ui) = ui_weak.upgrade() {
                    request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                }
            } else {
                apply_filtering_mode(
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    &ui_weak,
                    FilteringMode::Lanczos3,
                );
            }
        });
    }

    // Lanczos confirmation dialog — use trilinear instead
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_use_trilinear(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if from_backend_switch {
                // Switch to CPU and change filtering to Trilinear
                {
                    let mut state = state_handle.write();
                    state.select_render_backend(RenderBackend::Cpu);
                    if let Err(err) = config::save_render_backend(state.render_backend) {
                        warn!("Failed to save render backend config: {}", err);
                    }
                }
                if let Some(ui) = ui_weak.upgrade() {
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                }
            }
            apply_filtering_mode(
                &state_handle,
                &tile_cache,
                &render_timer,
                &ui_weak,
                FilteringMode::Trilinear,
            );
        });
    }

    // Lanczos confirmation dialog cancelled
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_lanczos_confirm_cancelled(move || {
            let from_backend_switch = ui_weak
                .upgrade()
                .map(|ui| ui.get_lanczos_confirm_from_backend_switch())
                .unwrap_or(false);

            if let Some(ui) = ui_weak.upgrade() {
                if from_backend_switch {
                    // Revert backend combobox (stay on GPU)
                    let state = state_handle.read();
                    update_render_backend(&ui, &state);
                } else {
                    // Revert filtering combobox
                    let state = state_handle.read();
                    update_filtering_mode(&ui, &state);
                }
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_series_bar_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_series_bar_requested(&ui, &state_handle, &tile_cache, &render_timer);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_series_go_back(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let Some(target) = state_handle.read().series_back_target() else {
                return;
            };
            navigate_series_to_folder(
                &ui,
                &state_handle,
                target,
                state::SeriesNavigationMode::Back,
            );
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_series_go_forward(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let Some(target) = state_handle.read().series_forward_target() else {
                return;
            };
            navigate_series_to_folder(
                &ui,
                &state_handle,
                target,
                state::SeriesNavigationMode::Forward,
            );
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_series_go_up(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let Some(target) = state_handle.read().series_up_target() else {
                return;
            };
            navigate_series_to_folder(
                &ui,
                &state_handle,
                target,
                state::SeriesNavigationMode::Push,
            );
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_minimap_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_minimap_visibility(&ui, &state_handle);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_toggle_metadata_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                toggle_metadata_visibility(&ui, &state_handle);
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

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_tab_activated(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    state.activate_tab_in_pane(pane_id, id);
                }
                let state = state_handle.read();
                ui.set_focused_pane(pane);
                update_tabs(&ui, &state);
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
                ui.invoke_focus_keyboard();
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
                ui.invoke_focus_keyboard();
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
                ui.invoke_focus_keyboard();
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

        ui.on_close_all_tabs(move || {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    state.close_all_tabs();
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
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

        ui.on_close_pane_tabs(move |pane| {
            if let Some(ui) = ui_weak.upgrade() {
                {
                    let mut state = state_handle.write();
                    let pane_id = pane_from_index(pane);
                    // The pane goes away with its last tab; track it by id.
                    if let Some(uid) = state.pane_uid(pane_id) {
                        let ids: Vec<i32> = state.tabs_for_pane(pane_id).to_vec();
                        for id in ids {
                            let Some(pane) = state.pane_by_uid(uid) else {
                                break;
                            };
                            state.set_focused_pane(pane);
                            state.close_tab_in_pane(pane, id);
                        }
                    }
                }
                let state = state_handle.read();
                refresh_tab_ui(&ui, &state);
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);

        ui.on_open_containing_folder(move |id| {
            let path = state.read().get_file(id).map(|file| file.path.clone());
            if let Some(path) = path {
                crate::desktop::reveal(path);
            }
        });
    }

    {
        let state = Arc::clone(&state);

        ui.on_copy_path(move |id| {
            let state = state.read();
            if let Some(file) = state.get_file(id) {
                crate::clipboard::copy_text(file.path.display().to_string());
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_right(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                let split = {
                    let mut state = state_handle.write();
                    let source_pane = pane_from_index(pane);
                    state.insert_pane(source_pane.0 + 1).map(|new_pane| {
                        state.duplicate_tab_to_pane(id, new_pane);
                        state.set_focused_pane(new_pane);
                        state.request_render();
                        (
                            state.split_enabled,
                            state.focused_pane,
                            state.pane_uid(source_pane),
                            state.pane_uid(new_pane),
                        )
                    })
                };
                let Some((split_enabled, focused_pane, source_uid, new_uid)) = split else {
                    ui.set_status_text(crate::state::MAX_PANES_MESSAGE.into());
                    return;
                };
                insert_pane_ui_state(new_uid, source_uid);
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

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_pane_focused(move |pane| {
            let focus_changed = {
                let state = state_handle.read();
                state.focused_pane != pane_from_index(pane)
            };
            if !focus_changed {
                return;
            }

            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                state.request_render();
            }

            if let Some(ui) = ui_weak.upgrade() {
                ui.set_focused_pane(pane);
                let state = state_handle.read();
                update_tabs(&ui, &state);
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

        ui.on_viewport_fit_to_view(move || {
            {
                let mut state = state_handle.write();
                let changed = frame_active_viewport(&mut state);
                let synced = state.mirror_locked_viewports_from_active();
                if changed || synced {
                    state.request_render();
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

        ui.on_minimap_drag_start(move || {
            {
                let mut state = state_handle.write();
                if let Some(vs) = state.active_viewport_mut() {
                    vs.start_external_drag();
                    state.request_render();
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

        ui.on_minimap_navigate(move |nx, ny| {
            {
                let mut state = state_handle.write();
                let mut changed = false;

                // The minimap shows the thumbnail's region (the content
                // bounds of sparse slides), not necessarily the canvas.
                let pane = if state.split_enabled {
                    state.focused_pane
                } else {
                    PaneId::PRIMARY
                };
                let region = state
                    .active_file_id_for_pane(pane)
                    .and_then(|file_id| state.get_file(file_id))
                    .and_then(|file| file.thumbnail.as_ref())
                    .map(|thumbnail| thumbnail.region);
                if let Some(vs) = state.active_viewport_mut() {
                    let nx = (nx as f64).clamp(0.0, 1.0);
                    let ny = (ny as f64).clamp(0.0, 1.0);
                    // The minimap shows the slide turned like the view.
                    let (nx, ny) = vs.viewport.orientation.to_image_normalized(nx, ny);
                    let vp = &mut vs.viewport;
                    let (new_x, new_y) = match region {
                        Some(region) => (
                            region.x as f64 + nx * region.width as f64,
                            region.y as f64 + ny * region.height as f64,
                        ),
                        None => (nx * vp.image_width, ny * vp.image_height),
                    };
                    vs.drag_center_to(new_x, new_y);
                    changed = true;
                }
                changed |= state.mirror_locked_viewports_from_active();
                if changed {
                    state.request_render();
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

        ui.on_minimap_drag_end(move || {
            {
                let mut state = state_handle.write();
                if let Some(vs) = state.active_viewport_mut() {
                    vs.end_external_drag();
                    state.request_render();
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

        ui.on_os_file_drop(move |pane, side, path_str| {
            let path = PathBuf::from(path_str.as_str());
            if !path.exists() {
                return;
            }
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if path.is_dir() {
                open_folder_as_series(&ui, &state_handle, path);
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
                return;
            }

            let target_pane = pane_from_index(pane);

            if side == -1 {
                // Drop in center: open as tab in the target pane
                {
                    let mut state = state_handle.write();
                    state.set_focused_pane(target_pane);
                }
                ui.set_focused_pane(pane);
                open_path_with_series(
                    &ui,
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    path,
                    crate::file_ops::OpenFileMode::ReuseExistingTab,
                );
            } else {
                // Drop on an edge: open in a new pane. Files dropped
                // together share that pane (one drop event per file).
                let recent = LAST_EDGE_DROP
                    .with(Cell::get)
                    .filter(|(at, _)| at.elapsed() < EDGE_DROP_GROUPING)
                    .and_then(|(_, uid)| state_handle.read().pane_by_uid(uid));
                let target_uid = {
                    let mut state = state_handle.write();
                    let pane = match recent {
                        Some(pane) => pane,
                        None => {
                            let insert_index = if side == 0 {
                                target_pane.0
                            } else {
                                target_pane.0 + 1
                            };
                            match state.insert_pane(insert_index) {
                                Some(new_pane) => {
                                    // The target pane moved right if the
                                    // new one went before it.
                                    let source = if insert_index <= target_pane.0 {
                                        PaneId(target_pane.0 + 1)
                                    } else {
                                        target_pane
                                    };
                                    insert_pane_ui_state(
                                        state.pane_uid(new_pane),
                                        state.pane_uid(source),
                                    );
                                    new_pane
                                }
                                // With the most panes, the slide opens in
                                // the pane it was dropped on.
                                None => target_pane,
                            }
                        }
                    };
                    state.set_focused_pane(pane);
                    state.request_render();
                    state.pane_uid(pane)
                };
                if let Some(uid) = target_uid {
                    LAST_EDGE_DROP.with(|last| last.set(Some((std::time::Instant::now(), uid))));
                }
                {
                    let state = state_handle.read();
                    ui.set_split_enabled(state.split_enabled);
                    ui.set_focused_pane(state.focused_pane.as_index());
                    update_tabs(&ui, &state);
                }
                // The new pane shows the slide even when it is open
                // elsewhere; if opening fails, the empty pane goes away.
                open_path_with_series(
                    &ui,
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    path,
                    crate::file_ops::OpenFileMode::ForceNewTab,
                );
            }
            request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_series_item_activated(move |path_str| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            activate_series_entry_path(
                &ui,
                &state_handle,
                &tile_cache,
                &render_timer,
                PathBuf::from(path_str.as_str()),
            );
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_series_item_open_in_new_tab(move |path_str| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            let path = PathBuf::from(path_str.as_str());
            if path.exists()
                && path.is_file()
                && !open_existing_file_in_new_tab(
                    &ui,
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    &path,
                )
            {
                open_path_with_series(
                    &ui,
                    &state_handle,
                    &tile_cache,
                    &render_timer,
                    path,
                    crate::file_ops::OpenFileMode::ForceNewTab,
                );
            }
        });
    }

    {
        ui.on_series_item_open_containing_folder(move |path_str| {
            crate::desktop::reveal(PathBuf::from(path_str.as_str()));
        });
    }

    ui.on_series_item_copy_path(move |path_str| {
        crate::clipboard::copy_text(path_str.to_string());
    });

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
                    // Empty panes collapse: focus wherever the tab ended up.
                    if let Some(pane) = state.pane_of_tab(id) {
                        state.set_focused_pane(pane);
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

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_tab_by_drop(move |id, to| {
            if let Some(ui) = ui_weak.upgrade() {
                let (source_uid, new_uid, full) = {
                    let mut state = state_handle.write();
                    let full = state.panes.len() >= crate::state::MAX_PANES;
                    let source_uid = state.pane_of_tab(id).and_then(|pane| state.pane_uid(pane));
                    let target = state.split_tab_to_new_pane(id, to.max(0) as usize);
                    (
                        source_uid,
                        target.and_then(|pane| state.pane_uid(pane)),
                        full,
                    )
                };
                if full {
                    ui.set_status_text(crate::state::MAX_PANES_MESSAGE.into());
                }
                insert_pane_ui_state(new_uid, source_uid);
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
                changed |= state.mirror_locked_viewports_from_active();
                if changed {
                    state.request_render();
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
                ui.invoke_focus_keyboard();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // HUD callbacks
    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_hud_toggle_dropdown(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.hud_dropdown_open = !hud.hud_dropdown_open;
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
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

        ui.on_hud_gamma_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.gamma = value;
                }
                state.request_render();
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

        ui.on_hud_sharpness_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.sharpness = value;
                }
                state.request_render();
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

        ui.on_hud_brightness_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.brightness = value;
                }
                state.request_render();
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

        ui.on_hud_contrast_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.contrast = value;
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    {
        let state_handle = Arc::clone(&state);
        let ui_weak = ui_weak.clone();

        ui.on_hud_measurement_unit_changed(move |pane, unit| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.measurement_unit = measurement_unit_from_slint(unit);
                }
            }
            // Measurement labels are written in the chosen unit.
            with_host(|host| host.resync_all_host_layers());
            if let Some(ui) = ui_weak.upgrade() {
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

        ui.on_hud_stain_normalization_changed(move |pane, sn| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.stain_normalization = stain_normalization_from_slint(sn);
                }
                state.request_render();
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

        ui.on_hud_reset_adjustments(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.reset_adjustments();
                }
                state.request_render();
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

        ui.on_hud_zoom_input(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let mut changed = false;
                let objective_power = state
                    .active_file_id_for_pane(state.focused_pane)
                    .and_then(|file_id| state.get_file(file_id))
                    .map(native_magnification)
                    .unwrap_or(None);
                let requested_zoom = parse_zoom_input(value.as_str(), objective_power);
                if let Some(viewport) = state.active_viewport_mut()
                    && let Some(value) = requested_zoom
                {
                    viewport.zoom_to(value);
                    changed = true;
                }
                changed |= state.mirror_locked_viewports_from_active();
                if changed {
                    state.request_render();
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

        ui.on_hud_zoom_in(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let changed = zoom_active_viewport(&mut state, ACTION_ZOOM_FACTOR);
                let synced = state.mirror_locked_viewports_from_active();
                if changed || synced {
                    state.request_render();
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

        ui.on_hud_zoom_out(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let changed = zoom_active_viewport(&mut state, 1.0 / ACTION_ZOOM_FACTOR);
                let synced = state.mirror_locked_viewports_from_active();
                if changed || synced {
                    state.request_render();
                }
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // Rotation and mirroring of a pane's view (HUD buttons).
    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_hud_orientation_changed(move |pane, action| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                let changed = match action {
                    HUD_ROTATE_LEFT => rotate_active_viewport(&mut state, -1),
                    HUD_ROTATE_RIGHT => rotate_active_viewport(&mut state, 1),
                    HUD_FLIP => flip_active_viewport(&mut state),
                    _ => reset_active_orientation(&mut state),
                };
                let synced = state.mirror_locked_viewports_from_active();
                if changed || synced {
                    state.request_render();
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

        ui.on_hud_deconv_h_intensity_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.deconv_hematoxylin_intensity = value;
                }
                state.request_render();
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

        ui.on_hud_deconv_e_intensity_changed(move |pane, value| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.deconv_eosin_intensity = value;
                }
                state.request_render();
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

        ui.on_hud_deconv_h_visibility_toggled(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.deconv_hematoxylin_visible = !hud.deconv_hematoxylin_visible;
                }
                state.request_render();
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

        ui.on_hud_deconv_e_visibility_toggled(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    hud.deconv_eosin_visible = !hud.deconv_eosin_visible;
                }
                state.request_render();
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

        ui.on_hud_deconv_h_view_toggled(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    use crate::state::IsolatedChannel;
                    hud.deconv_isolated_channel =
                        if hud.deconv_isolated_channel == IsolatedChannel::Hematoxylin {
                            IsolatedChannel::None
                        } else {
                            IsolatedChannel::Hematoxylin
                        };
                }
                state.request_render();
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

        ui.on_hud_deconv_e_view_toggled(move |pane| {
            {
                let mut state = state_handle.write();
                state.set_focused_pane(pane_from_index(pane));
                if let Some(hud) = active_hud_mut(&mut state) {
                    use crate::state::IsolatedChannel;
                    hud.deconv_isolated_channel =
                        if hud.deconv_isolated_channel == IsolatedChannel::Eosin {
                            IsolatedChannel::None
                        } else {
                            IsolatedChannel::Eosin
                        };
                }
                state.request_render();
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
            }
        });
    }

    // -- Export dialog callbacks --
    ui.on_export_dialog_settings_changed(crate::export_dialog::settings_changed);

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        ui.on_export_dialog_request_preview(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::export_dialog::request_preview(&ui, &state, &tile_cache);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        ui.on_export_dialog_export_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::export_dialog::export(&ui, &state, &tile_cache);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        ui.on_export_dialog_cancel_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::export_dialog::close(&ui);
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let ui_weak = ui_weak.clone();
        ui.on_export_dialog_reset_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::export_dialog::reset(&ui, &state, &tile_cache);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_minimize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.window().set_minimized(true);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_maximize(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let maximized = ui.window().is_maximized();
                ui.window().set_maximized(!maximized);
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();

        ui.on_window_close(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let _ = ui.hide();
                let _ = slint::quit_event_loop();
            }
        });
    }

    // -- Slide information panel --
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_requested(move |tab_id| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::open_for_tab(&ui, &state, tab_id);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_hide_label_toggled(move |hidden| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::set_hide_label(&ui, hidden);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_filter_edited(move |filter| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::filter_edited(&ui, &filter);
            }
        });
    }
    ui.on_slide_info_copy_all(crate::slide_info::copy_all);
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_pixel_size_set(move |text| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::set_pixel_size(&ui, &state, Some(&text));
            }
        });
    }
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_pixel_size_cleared(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::set_pixel_size(&ui, &state, None);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_slide_info_copy_row(move |index| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::slide_info::copy_row(&ui, index);
            }
        });
    }

    // -- Dataset export dialog callbacks --
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_open_dialog(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::dataset_dialog::open(&ui, &state);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_add_input(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::dataset_dialog::add_inputs(&ui);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_remove_input(move |index| {
            if let Some(ui) = ui_weak.upgrade() {
                crate::dataset_dialog::remove_input(&ui, index);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_change_output_dir(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::dataset_dialog::choose_output_dir(&ui);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_cancel_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_dataset_export_dialog_visible(false);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_settings_changed(move |mut settings| {
            // Edits clear a previous validation error.
            if !settings.error.is_empty()
                && let Some(ui) = ui_weak.upgrade()
            {
                settings.error = SharedString::new();
                ui.set_dataset_export_settings(settings);
            }
        });
    }
    {
        let ui_weak = ui_weak.clone();
        ui.on_dataset_export_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                crate::dataset_dialog::start(&ui);
            }
        });
    }

    #[cfg(target_os = "macos")]
    {
        let ui_weak = ui_weak.clone();
        Timer::single_shot(Duration::from_millis(0), move || {
            if let Some(ui) = ui_weak.upgrade() {
                align_macos_window_controls(ui.window());
            }
        });
    }

    {
        use slint::winit_030::{EventResult, WinitWindowAccessor, winit};

        let state_handle = Arc::clone(&state);
        let last_cursor = Rc::new(Cell::new((0.0f64, 0.0f64)));
        let modifiers = Rc::new(Cell::new(winit::keyboard::ModifiersState::default()));
        let drag_press: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        let last_press_time = Rc::new(Cell::new(std::time::Instant::now()));
        let dbl_count = Rc::new(Cell::new(0u32));
        let os_file_hovering = Rc::new(Cell::new(false));
        let dnd_poll_timer = Rc::new(Timer::default());

        ui.window().on_winit_window_event({
            let last_cursor = Rc::clone(&last_cursor);
            let modifiers = Rc::clone(&modifiers);
            let drag_press = Rc::clone(&drag_press);
            let last_press_time = Rc::clone(&last_press_time);
            let dbl_count = Rc::clone(&dbl_count);
            let os_file_hovering = Rc::clone(&os_file_hovering);
            let dnd_poll_timer = Rc::clone(&dnd_poll_timer);
            let ui_weak = ui_weak.clone();
            let state_handle = Arc::clone(&state_handle);
            let tile_cache = Arc::clone(&tile_cache);
            let render_timer = Rc::clone(&render_timer);

            move |slint_window, event| match event {
                winit::event::WindowEvent::Resized(_physical_size) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        request_render_loop(
                            &render_timer,
                            &ui.as_weak(),
                            &state_handle,
                            &tile_cache,
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::ScaleFactorChanged { .. } => {
                    if let Some(ui) = ui_weak.upgrade() {
                        request_render_loop(
                            &render_timer,
                            &ui.as_weak(),
                            &state_handle,
                            &tile_cache,
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::ModifiersChanged(next_modifiers) => {
                    modifiers.set(next_modifiers.state());
                    EventResult::Propagate
                }
                winit::event::WindowEvent::Focused(_) => {
                    crate::host_ui::window_focus_changed();
                    EventResult::Propagate
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    last_cursor.set((position.x, position.y));

                    if let Some((px, py)) = drag_press.get() {
                        let dx = position.x - px;
                        let dy = position.y - py;
                        if dx * dx + dy * dy > 25.0 {
                            drag_press.set(None);

                            let scale = slint_window.scale_factor();
                            let lx = (px / scale as f64) as f32;
                            let ly = (py / scale as f64) as f32;
                            let _ = slint_window.try_dispatch_event(
                                slint::platform::WindowEvent::PointerReleased {
                                    position: slint::LogicalPosition::new(lx, ly),
                                    button: slint::platform::PointerEventButton::Left,
                                },
                            );

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
                    let in_drag_area = if let Some(ui) = ui_weak.upgrade() {
                        let drag_start = ui.get_toolbar_drag_area_x() as f64;
                        let drag_end = drag_start + ui.get_toolbar_drag_area_width() as f64;

                        ly < toolbar_height && lx >= drag_start && lx < drag_end
                    } else {
                        false
                    };

                    if in_drag_area {
                        drag_press.set(Some((cx, cy)));

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
                winit::event::WindowEvent::HoveredFile(_path) => {
                    os_file_hovering.set(true);
                    // Start a polling timer to track cursor position during DnD.
                    // On X11, winit doesn't emit CursorMoved during external DnD,
                    // so we query the pointer position directly from X11.
                    {
                        let ui_weak = ui_weak.clone();
                        let dnd_poll_timer_inner = Rc::clone(&dnd_poll_timer);
                        let os_file_hovering = Rc::clone(&os_file_hovering);
                        dnd_poll_timer.start(
                            slint::TimerMode::Repeated,
                            Duration::from_millis(16),
                            move || {
                                if !os_file_hovering.get() {
                                    dnd_poll_timer_inner.stop();
                                    return;
                                }
                                let Some(ui) = ui_weak.upgrade() else {
                                    dnd_poll_timer_inner.stop();
                                    return;
                                };
                                if let Some((lx, ly)) = query_dnd_cursor_logical(&ui) {
                                    ui.invoke_update_os_file_hover(lx, ly);
                                }
                            },
                        );
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::DroppedFile(path) => {
                    os_file_hovering.set(false);
                    dnd_poll_timer.stop();
                    if let Some(ui) = ui_weak.upgrade() {
                        // Query final cursor position for accurate drop zone
                        if let Some((lx, ly)) = query_dnd_cursor_logical(&ui) {
                            ui.invoke_update_os_file_hover(lx, ly);
                        }
                        let pane = ui.get_os_file_hover_pane();
                        let side = ui.get_os_file_hover_side();
                        ui.invoke_clear_os_file_hover();
                        let path_str = path.to_string_lossy().to_string();
                        ui.invoke_os_file_drop(pane, side, SharedString::from(path_str));
                    }
                    EventResult::Propagate
                }
                winit::event::WindowEvent::HoveredFileCancelled => {
                    os_file_hovering.set(false);
                    dnd_poll_timer.stop();
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.invoke_clear_os_file_hover();
                    }
                    EventResult::Propagate
                }
                _ => EventResult::Propagate,
            }
        });
    }
}
