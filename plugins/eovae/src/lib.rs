mod analysis;
mod db;
mod filter;
mod model;
mod sidebar;
mod state;
mod stats;

use abi_stable::std_types::{ROption, RString, RVec};
use plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayComponentRequestFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};
use serde_json::json;
use sidebar::{get_sidebar_properties, initialize_from_config, on_sidebar_callback, show_sidebar};
use state::{VisualizationMode, plugin_state, request_render_if_available, set_host_api};
use std::time::Duration;

const BUTTON_ID: &str = "toggle_eovae";
const GRID_BUTTON_ID: &str = "toggle_grid";
const VISUALIZATION_BUTTON_ID: &str = "visualization_mode";
const VISUALIZATION_ORIGINAL_ACTION_ID: &str = "set_visualization_original";
const VISUALIZATION_RECONSTRUCTION_ACTION_ID: &str = "set_visualization_reconstruction";
const VISUALIZATION_DIFFERENCE_ACTION_ID: &str = "set_visualization_difference";
const VISUALIZATION_ERROR_MAP_ACTION_ID: &str = "set_visualization_error_map";
const FILTER_ID: &str = "eovae_overlay";
const TOOL_ICON: &str = include_str!("../ui/icons/tool.svg");
const GRID_ICON: &str = include_str!("../ui/icons/grid.svg");
const VISUALIZE_TRANSFORM_ICON: &str = include_str!("../ui/icons/visualize-transform.svg");

fn plugin_trace(message: impl AsRef<str>) {
    if std::env::var_os("EOV_PLUGIN_TRACE").is_some() {
        eprintln!("[eovae] {}", message.as_ref());
    }
}

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    set_host_api(host_api);
    initialize_from_config();
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: RString::from(BUTTON_ID),
        tooltip: RString::from("VAE"),
        icon_svg: RString::from(TOOL_ICON),
        action_id: RString::from(BUTTON_ID),
        tool_mode: ROption::RNone,
        hotkey: ROption::RNone,
    }])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::from(vec![
        HudToolbarButtonFFI {
            button_id: GRID_BUTTON_ID.into(),
            tooltip: "Toggle tile grid".into(),
            icon_svg: GRID_ICON.into(),
            toggled_icon_svg: ROption::RNone,
            action_id: GRID_BUTTON_ID.into(),
        },
        HudToolbarButtonFFI {
            button_id: VISUALIZATION_BUTTON_ID.into(),
            tooltip: "Visualize reconstruction".into(),
            icon_svg: VISUALIZE_TRANSFORM_ICON.into(),
            toggled_icon_svg: ROption::RNone,
            action_id: VISUALIZATION_BUTTON_ID.into(),
        },
    ])
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    plugin_trace(format!("on_action action_id={}", action_id));
    if action_id.as_str() == BUTTON_ID {
        plugin_trace("show_sidebar begin");
        show_sidebar();
        plugin_trace("show_sidebar returned");
    }
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_hud_action_ffi(
    action_id: RString,
    viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    if action_id.as_str() == GRID_BUTTON_ID {
        let mut state = plugin_state().lock().unwrap();
        if !state.pane_grid_enabled.insert(viewport.pane_index) {
            state.pane_grid_enabled.remove(&viewport.pane_index);
        }
        drop(state);
        request_render_if_available();
    } else if let Some(mode) = visualization_mode_for_action(action_id.as_str()) {
        set_visualization_mode(mode, viewport);
    }
    ActionResponseFFI { open_window: false }
}

fn visualization_mode_for_action(action_id: &str) -> Option<VisualizationMode> {
    match action_id {
        VISUALIZATION_ORIGINAL_ACTION_ID => Some(VisualizationMode::Original),
        VISUALIZATION_RECONSTRUCTION_ACTION_ID => Some(VisualizationMode::Reconstruction),
        VISUALIZATION_DIFFERENCE_ACTION_ID => Some(VisualizationMode::Difference),
        VISUALIZATION_ERROR_MAP_ACTION_ID => Some(VisualizationMode::ErrorMap),
        _ => None,
    }
}

fn set_visualization_mode(mode: VisualizationMode, viewport: ViewportSnapshotFFI) {
    let active = {
        let mut state = plugin_state().lock().unwrap();
        state
            .pane_visualization_modes
            .insert(viewport.pane_index, mode);
        if mode == VisualizationMode::Original {
            state
                .pane_auto_viewport_request_keys
                .remove(&viewport.pane_index);
        }
        state.visualization_mode != VisualizationMode::Original
            || state
                .pane_visualization_modes
                .values()
                .any(|pane_mode| *pane_mode != VisualizationMode::Original)
    };

    state::set_hud_toolbar_button_active_if_available(VISUALIZATION_BUTTON_ID, active);
    request_render_if_available();
}

fn any_visualization_enabled(state: &state::PluginState) -> bool {
    state.model.is_some()
        && (state.visualization_mode != VisualizationMode::Original
            || state
                .pane_visualization_modes
                .values()
                .any(|mode| *mode != VisualizationMode::Original))
}

extern "C" fn on_ui_callback_ffi(callback_name: RString, args_json: RString) {
    plugin_trace(format!(
        "ui_callback name={} args={}",
        callback_name, args_json
    ));
    on_sidebar_callback(callback_name.as_str(), args_json.as_str());
    plugin_trace(format!("ui_callback done name={}", callback_name));
}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    get_sidebar_properties()
}

extern "C" fn get_viewport_context_menu_items_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    RVec::new()
}

extern "C" fn on_viewport_context_menu_action_ffi(
    _item_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_overlay_points_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    RVec::new()
}

extern "C" fn get_viewport_overlay_polygons_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    let mut polygons = Vec::new();
    let mut clear_pulse = false;
    let state = plugin_state().lock().unwrap();

    let hovered_region = state
        .hovered_region_id
        .as_deref()
        .and_then(|id| state.cache.get(id))
        .filter(|entry| entry.namespace == state.cache_namespace)
        .map(|entry| entry.tile.clone());
    let pulsing_region = state
        .pulsing_region_id
        .as_deref()
        .and_then(|id| state.cache.get(id))
        .filter(|entry| entry.namespace == state.cache_namespace)
        .map(|entry| entry.tile.clone());

    if state.pane_grid_enabled.contains(&viewport.pane_index) {
        let mip_level = state.config.mip_level;
        let tile_size = state
            .model
            .as_ref()
            .map(|model| model.summary.tile_size as f64)
            .unwrap_or(256.0);
        polygons.extend(grid_overlay_polygons(&viewport, tile_size, mip_level));
    }

    if let Some(tile) = hovered_region.as_ref().filter(|tile| {
        pulsing_region
            .as_ref()
            .is_none_or(|pulse| pulse.id() != tile.id())
    }) {
        polygons.push(region_polygon(&tile.id(), tile, 0xF1, 0xC4, 0x0F));
    }

    if let Some(tile) = pulsing_region {
        let elapsed = state
            .pulsing_region_started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or(Duration::ZERO);
        if elapsed >= Duration::from_millis(1300) {
            clear_pulse = true;
        } else {
            polygons.push(region_polygon(
                &format!("pulse:{}", tile.id()),
                &tile,
                0xF1,
                0xC4,
                0x0F,
            ));
        }
    }

    drop(state);

    if clear_pulse {
        let mut state = plugin_state().lock().unwrap();
        state.pulsing_region_id = None;
        state.pulsing_region_started_at = None;
    }

    RVec::from(polygons)
}

extern "C" fn get_viewport_overlay_component_ffi() -> ROption<ViewportOverlayComponentRequestFFI> {
    ROption::RNone
}

extern "C" fn get_viewport_overlay_properties_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<UiPropertyFFI> {
    let state = plugin_state().lock().unwrap();
    let grid_visible = state.pane_grid_enabled.contains(&viewport.pane_index);
    let mip_level = state.config.mip_level;
    let tile_size = state
        .model
        .as_ref()
        .map(|model| model.summary.tile_size)
        .unwrap_or(256);

    RVec::from(vec![
        UiPropertyFFI {
            name: "grid-visible".into(),
            json_value: json!(grid_visible).to_string().into(),
        },
        UiPropertyFFI {
            name: format!("hud-button/{GRID_BUTTON_ID}/active").into(),
            json_value: json!(grid_visible).to_string().into(),
        },
        UiPropertyFFI {
            name: "tile-size".into(),
            json_value: json!(tile_size as f64).to_string().into(),
        },
        UiPropertyFFI {
            name: "mip-level".into(),
            json_value: json!(mip_level as f64).to_string().into(),
        },
    ])
}

fn region_polygon(
    annotation_id: &str,
    tile: &crate::analysis::AnalyzedTile,
    red: u8,
    green: u8,
    blue: u8,
) -> ViewportOverlayPolygonFFI {
    ViewportOverlayPolygonFFI {
        annotation_id: annotation_id.into(),
        vertices: vec![
            ViewportOverlayVertexFFI {
                x_level0: tile.x as f64,
                y_level0: tile.y as f64,
            },
            ViewportOverlayVertexFFI {
                x_level0: (tile.x + tile.width as u64) as f64,
                y_level0: tile.y as f64,
            },
            ViewportOverlayVertexFFI {
                x_level0: (tile.x + tile.width as u64) as f64,
                y_level0: (tile.y + tile.height as u64) as f64,
            },
            ViewportOverlayVertexFFI {
                x_level0: tile.x as f64,
                y_level0: (tile.y + tile.height as u64) as f64,
            },
        ]
        .into(),
        fill_red: red,
        fill_green: green,
        fill_blue: blue,
    }
}

fn grid_overlay_polygons(
    viewport: &ViewportSnapshotFFI,
    tile_size: f64,
    mip_level: u32,
) -> Vec<ViewportOverlayPolygonFFI> {
    let step = (tile_size * (1u64 << mip_level) as f64).max(1.0);
    let pixel_world_x = ((viewport.bounds_right - viewport.bounds_left) / viewport.width.max(1.0))
        .abs()
        .max(1e-6);
    let pixel_world_y = ((viewport.bounds_bottom - viewport.bounds_top) / viewport.height.max(1.0))
        .abs()
        .max(1e-6);
    let half_thickness_x = pixel_world_x * 0.5;
    let half_thickness_y = pixel_world_y * 0.5;
    let mut polygons = Vec::new();

    let first_x = (viewport.bounds_left / step).floor() * step;
    let mut x = first_x;
    let mut vertical_index = 0usize;
    while x <= viewport.bounds_right + step && vertical_index < 512 {
        polygons.push(ViewportOverlayPolygonFFI {
            annotation_id: format!("grid-v-{vertical_index}").into(),
            vertices: vec![
                ViewportOverlayVertexFFI {
                    x_level0: x - half_thickness_x,
                    y_level0: viewport.bounds_top,
                },
                ViewportOverlayVertexFFI {
                    x_level0: x + half_thickness_x,
                    y_level0: viewport.bounds_top,
                },
                ViewportOverlayVertexFFI {
                    x_level0: x + half_thickness_x,
                    y_level0: viewport.bounds_bottom,
                },
                ViewportOverlayVertexFFI {
                    x_level0: x - half_thickness_x,
                    y_level0: viewport.bounds_bottom,
                },
            ]
            .into(),
            fill_red: 0x00,
            fill_green: 0x00,
            fill_blue: 0x00,
        });
        x += step;
        vertical_index += 1;
    }

    let first_y = (viewport.bounds_top / step).floor() * step;
    let mut y = first_y;
    let mut horizontal_index = 0usize;
    while y <= viewport.bounds_bottom + step && horizontal_index < 512 {
        polygons.push(ViewportOverlayPolygonFFI {
            annotation_id: format!("grid-h-{horizontal_index}").into(),
            vertices: vec![
                ViewportOverlayVertexFFI {
                    x_level0: viewport.bounds_left,
                    y_level0: y - half_thickness_y,
                },
                ViewportOverlayVertexFFI {
                    x_level0: viewport.bounds_right,
                    y_level0: y - half_thickness_y,
                },
                ViewportOverlayVertexFFI {
                    x_level0: viewport.bounds_right,
                    y_level0: y + half_thickness_y,
                },
                ViewportOverlayVertexFFI {
                    x_level0: viewport.bounds_left,
                    y_level0: y + half_thickness_y,
                },
            ]
            .into(),
            fill_red: 0x00,
            fill_green: 0x00,
            fill_blue: 0x00,
        });
        y += step;
        horizontal_index += 1;
    }

    polygons
}

extern "C" fn on_point_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_polygon_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn on_undo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_redo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_point_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_polygon_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    let state = plugin_state().lock().unwrap();
    let enabled = any_visualization_enabled(&state);
    RVec::from(vec![ViewportFilterFFI {
        filter_id: FILTER_ID.into(),
        name: "VAE Overlay".into(),
        supports_cpu: true,
        supports_gpu: false,
        enabled,
    }])
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    rgba_data: *mut u8,
    len: u32,
    width: u32,
    height: u32,
    viewport: *const ViewportSnapshotFFI,
) -> bool {
    if rgba_data.is_null() || viewport.is_null() {
        return false;
    }
    let data = unsafe { std::slice::from_raw_parts_mut(rgba_data, len as usize) };
    let viewport = unsafe { &*viewport };
    filter::apply_overlay(data, width, height, viewport)
}

extern "C" fn apply_filter_gpu_ffi(_filter_id: RString, _ctx: *const GpuFilterContextFFI) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {
    request_render_if_available();
}

extern "C" fn on_viewport_annotation_selected_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
) {
}

#[cfg_attr(feature = "export-vtable-symbol", unsafe(no_mangle))]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        set_host_api: set_host_api_ffi,
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        get_hud_toolbar_buttons: get_hud_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_hud_action: on_hud_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
        get_sidebar_properties: get_sidebar_properties_ffi,
        get_viewport_context_menu_items: get_viewport_context_menu_items_ffi,
        on_viewport_context_menu_action: on_viewport_context_menu_action_ffi,
        get_viewport_overlay_points: get_viewport_overlay_points_ffi,
        get_viewport_overlay_polygons: get_viewport_overlay_polygons_ffi,
        get_viewport_overlay_component: get_viewport_overlay_component_ffi,
        get_viewport_overlay_properties: get_viewport_overlay_properties_ffi,
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_undo: on_undo_ffi,
        on_redo: on_redo_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        on_viewport_annotation_selected: on_viewport_annotation_selected_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}
