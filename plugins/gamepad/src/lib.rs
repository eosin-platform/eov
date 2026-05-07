mod sidebar;
mod state;

use abi_stable::std_types::{ROption, RString, RVec};
use plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayComponentRequestFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};

const TOOLBAR_BUTTON_ID: &str = "toggle_gamepad";
const TOOLBAR_ICON: &str = include_str!("../ui/icons/gamepad.svg");

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    state::set_host_api(host_api);
    state::ensure_worker_started();
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: TOOLBAR_BUTTON_ID.into(),
        tooltip: "Gamepad".into(),
        icon_svg: TOOLBAR_ICON.into(),
        action_id: TOOLBAR_BUTTON_ID.into(),
        tool_mode: ROption::RNone,
        hotkey: ROption::RNone,
    }])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    if action_id.as_str() == TOOLBAR_BUTTON_ID {
        return ActionResponseFFI {
            open_window: state::toggle_window(),
        };
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
    sidebar::on_sidebar_callback(callback_name.as_str(), args_json.as_str());
}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    sidebar::get_sidebar_properties()
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

extern "C" fn get_viewport_overlay_component_ffi() -> ROption<ViewportOverlayComponentRequestFFI> {
    ROption::RNone
}

extern "C" fn get_viewport_overlay_properties_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<UiPropertyFFI> {
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
    RVec::new()
}

extern "C" fn apply_cpu_filter_ffi(
    _filter_id: RString,
    _rgba_data: *mut u8,
    _rgba_len: u32,
    _width: u32,
    _height: u32,
    _viewport: *const ViewportSnapshotFFI,
) -> bool {
    false
}

extern "C" fn apply_gpu_filter_ffi(
    _filter_id: RString,
    _ctx: *const GpuFilterContextFFI,
) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {}

extern "C" fn on_viewport_annotation_selected_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
) {
}

#[unsafe(no_mangle)]
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
        apply_filter_cpu: apply_cpu_filter_ffi,
        apply_filter_gpu: apply_gpu_filter_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}