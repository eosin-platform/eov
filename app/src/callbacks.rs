use crate::config;
use crate::state::{self, AppState, PaneId, RenderBackend};
use crate::{
    AppWindow, RenderMode, ToolType, build_recent_menu_items, copy_text_to_clipboard,
    handle_tool_mouse_down, handle_tool_mouse_move, handle_tool_mouse_up, insert_pane_ui_state,
    open_file, pane_from_index, refresh_tab_ui, request_render_loop, slider_value_to_zoom,
    update_render_backend, update_tabs, update_tool_overlays, update_tool_state,
};
use common::TileCache;
use common::viewport::ZOOM_FACTOR;
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

const ACTION_ZOOM_FACTOR: f64 = ZOOM_FACTOR * ZOOM_FACTOR;

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

fn frame_active_viewport(state: &mut AppState) -> bool {
    let pane = state.focused_pane;
    let roi = state
        .active_file_id_for_pane(pane)
        .and_then(|file_id| state.get_file(file_id))
        .and_then(|file| file.roi.filter(|roi| roi.pane == pane));

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

fn zoom_active_viewport(state: &mut AppState, factor: f64) -> bool {
    let Some(viewport) = state.active_viewport_mut() else {
        return false;
    };

    let center_x = viewport.viewport.width / 2.0;
    let center_y = viewport.viewport.height / 2.0;
    viewport.zoom_at_discrete(factor, center_x, center_y);
    true
}

fn toggle_minimap_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_minimap = {
        let mut state = state.write();
        state.toggle_minimap();
        state.show_minimap
    };
    ui.set_show_minimap(show_minimap);
}

fn toggle_metadata_visibility(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let show_metadata = {
        let mut state = state.write();
        state.toggle_metadata();
        state.show_metadata
    };
    ui.set_show_metadata(show_metadata);
}

pub fn setup_callbacks(
    ui: &AppWindow,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
) {
    let ui_weak = ui.as_weak();
    let clipboard = Rc::new(RefCell::new(None));

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_open_file_requested(move || {
            let ui = ui_weak.upgrade().unwrap();

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
                    let (split_enabled, source_pane, new_pane) = {
                        let mut state = state_handle.write();
                        let source_pane = pane;
                        let new_pane = state.insert_pane(source_pane.0 + 1);
                        state.duplicate_tab_to_pane(tab_id, new_pane);
                        state.set_focused_pane(new_pane);
                        state.request_render();
                        (state.split_enabled, source_pane, new_pane)
                    };
                    insert_pane_ui_state(new_pane, Some(source_pane));
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

        ui.on_frame_requested(move || {
            {
                let mut state = state_handle.write();
                if frame_active_viewport(&mut state) {
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

        ui.on_zoom_in_requested(move || {
            {
                let mut state = state_handle.write();
                if zoom_active_viewport(&mut state, ACTION_ZOOM_FACTOR) {
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

        ui.on_zoom_out_requested(move || {
            {
                let mut state = state_handle.write();
                if zoom_active_viewport(&mut state, 1.0 / ACTION_ZOOM_FACTOR) {
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

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_split_right(move |pane, id| {
            if let Some(ui) = ui_weak.upgrade() {
                let (split_enabled, focused_pane, source_pane) = {
                    let mut state = state_handle.write();
                    let source_pane = pane_from_index(pane);
                    let new_pane = state.insert_pane(source_pane.0 + 1);
                    state.duplicate_tab_to_pane(id, new_pane);
                    state.set_focused_pane(new_pane);
                    state.request_render();
                    (state.split_enabled, state.focused_pane, source_pane)
                };
                insert_pane_ui_state(focused_pane, Some(source_pane));
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
                    viewport.stop();
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

    {
        let state_handle = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui_weak.clone();

        ui.on_viewport_fit_to_view(move || {
            {
                let mut state = state_handle.write();
                if frame_active_viewport(&mut state) {
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

    {
        let state = Arc::clone(&state);
        let tile_cache = Arc::clone(&tile_cache);
        let render_timer = Rc::clone(&render_timer);
        let ui_weak = ui.as_weak();

        ui.on_file_dropped(move |path_str| {
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

            let target_pane = pane_from_index(pane);

            if side == -1 {
                // Drop in center: open as tab in the target pane
                {
                    let mut state = state_handle.write();
                    state.set_focused_pane(target_pane);
                }
                ui.set_focused_pane(pane);
                open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
            } else {
                // Drop on edge: create a new pane split
                let insert_index = if side == 0 {
                    target_pane.0
                } else {
                    target_pane.0 + 1
                };
                let (new_pane, source_pane) = {
                    let mut state = state_handle.write();
                    let source = target_pane;
                    let new_pane = state.insert_pane(insert_index);
                    state.set_focused_pane(new_pane);
                    state.request_render();
                    (new_pane, source)
                };
                insert_pane_ui_state(new_pane, Some(source_pane));
                {
                    let state = state_handle.read();
                    ui.set_split_enabled(state.split_enabled);
                    ui.set_focused_pane(state.focused_pane.as_index());
                    update_tabs(&ui, &state);
                }
                open_file(&ui, &state_handle, &tile_cache, &render_timer, path);
            }
            request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
        });
    }

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
                let (source_pane_for_cache, target_pane_for_cache) = {
                    let mut state = state_handle.write();
                    let source_pane = state
                        .panes
                        .iter()
                        .enumerate()
                        .find(|(_, pane_state)| pane_state.tabs.contains(&id))
                        .map(|(index, _)| PaneId(index))
                        .unwrap_or(PaneId::PRIMARY);
                    state.split_tab_to_new_pane(id, source_pane, to.max(0) as usize);
                    (source_pane, state.focused_pane)
                };
                insert_pane_ui_state(target_pane_for_cache, Some(source_pane_for_cache));
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
            }
            if let Some(ui) = ui_weak.upgrade() {
                request_render_loop(&render_timer, &ui.as_weak(), &state_handle, &tile_cache);
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
                winit::event::WindowEvent::ModifiersChanged(next_modifiers) => {
                    modifiers.set(next_modifiers.state());
                    EventResult::Propagate
                }
                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    let modifier_state = modifiers.get();
                    let plain_shortcut = event.state == winit::event::ElementState::Pressed
                        && !event.repeat
                        && !modifier_state.control_key()
                        && !modifier_state.alt_key()
                        && !modifier_state.super_key();
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

                    let handled = if plain_shortcut {
                        use winit::keyboard::{Key, KeyCode, PhysicalKey};

                        let shortcut = match (&event.logical_key, &event.physical_key) {
                            (Key::Character(text), _) if text.eq_ignore_ascii_case("f") => {
                                Some("frame")
                            }
                            (Key::Character(text), _) if text.eq_ignore_ascii_case("m") => {
                                Some("toggle-minimap")
                            }
                            (Key::Character(text), _) if text == "+" => Some("zoom-in"),
                            (Key::Character(text), _) if text == "-" => Some("zoom-out"),
                            (_, PhysicalKey::Code(KeyCode::NumpadAdd)) => Some("zoom-in"),
                            (_, PhysicalKey::Code(KeyCode::NumpadSubtract)) => Some("zoom-out"),
                            _ => None,
                        };

                        if let Some(shortcut) = shortcut {
                            if let Some(ui) = ui_weak.upgrade() {
                                match shortcut {
                                    "frame" => {
                                        let mut state = state_handle.write();
                                        if frame_active_viewport(&mut state) {
                                            state.request_render();
                                        }
                                    }
                                    "zoom-in" => {
                                        let mut state = state_handle.write();
                                        if zoom_active_viewport(&mut state, ACTION_ZOOM_FACTOR) {
                                            state.request_render();
                                        }
                                    }
                                    "zoom-out" => {
                                        let mut state = state_handle.write();
                                        if zoom_active_viewport(
                                            &mut state,
                                            1.0 / ACTION_ZOOM_FACTOR,
                                        ) {
                                            state.request_render();
                                        }
                                    }
                                    "toggle-minimap" => {
                                        toggle_minimap_visibility(&ui, &state_handle);
                                    }
                                    _ => {}
                                }
                                request_render_loop(
                                    &render_timer,
                                    &ui.as_weak(),
                                    &state_handle,
                                    &tile_cache,
                                );
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if handled {
                        return EventResult::PreventDefault;
                    }

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
                    let win_w = slint_window.size().width as f64 / scale;
                    let toolbar_action_width = if let Some(ui) = ui_weak.upgrade() {
                        ui.get_toolbar_action_width() as f64
                    } else {
                        0.0
                    };
                    let buttons_right_start = if cfg!(target_os = "macos") {
                        win_w
                    } else {
                        win_w - 138.0
                    };

                    if ly < toolbar_height && lx >= toolbar_action_width && lx < buttons_right_start
                    {
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
