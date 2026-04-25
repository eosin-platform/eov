mod analysis;
mod filter;
mod model;
mod sidebar;
mod state;
mod stats;

use abi_stable::std_types::{ROption, RString, RVec};
use plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayPointFFI, ViewportOverlayPolygonFFI, ViewportOverlayVertexFFI,
    ViewportSnapshotFFI,
};
use sidebar::{get_sidebar_properties, initialize_from_config, on_sidebar_callback, show_sidebar};
use state::{request_render_if_available, set_host_api};

const BUTTON_ID: &str = "toggle_eovae";
const FILTER_ID: &str = "eovae_overlay";
const TOOL_ICON: &str = include_str!("../ui/icons/tool.svg");

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
    RVec::new()
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
    _action_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_ui_callback_ffi(callback_name: RString, args_json: RString) {
    plugin_trace(format!("ui_callback name={} args={}", callback_name, args_json));
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
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    RVec::new()
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
    RVec::from(vec![ViewportFilterFFI {
        filter_id: FILTER_ID.into(),
        name: "VAE Overlay".into(),
        supports_cpu: true,
        supports_gpu: false,
        enabled: true,
    }])
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    rgba_data: *mut u8,
    len: u32,
    width: u32,
    height: u32,
) -> bool {
    if rgba_data.is_null() {
        return false;
    }
    let data = unsafe { std::slice::from_raw_parts_mut(rgba_data, len as usize) };
    filter::apply_overlay(data, width, height)
}

extern "C" fn apply_filter_gpu_ffi(_filter_id: RString, _ctx: *const GpuFilterContextFFI) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {
    request_render_if_available();
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
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_undo: on_undo_ffi,
        on_redo: on_redo_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}