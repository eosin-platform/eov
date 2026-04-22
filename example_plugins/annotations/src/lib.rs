mod db;
mod model;
mod operations;
mod sidebar;
mod state;

use abi_stable::std_types::{ROption, RString, RVec};
use model::{Annotation, hex_color_to_rgb};
use operations::{
    ensure_loaded_for_viewport, move_point_annotation, move_polygon_annotation,
    persist_point_annotation, persist_polygon_annotation, refresh_sidebar_if_available,
    request_render_if_available, start_point_annotation_flow, start_polygon_annotation_flow,
    sync_active_file,
};
use plugin_api::ffi::{
    ActionResponseFFI, GpuFilterContextFFI, HostApiVTable, HostLogLevelFFI, HostToolModeFFI,
    HudToolbarButtonFFI, PluginVTable, ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI,
    ViewportFilterFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};
use sidebar::{get_sidebar_properties, on_sidebar_callback};
use state::{host_api, log_message, plugin_state, set_host_api};

const SIDEBAR_WIDTH_PX: u32 = 300;
const SIDEBAR_UI_PATH: &str = "ui/annotations-sidebar.slint";
const SIDEBAR_COMPONENT: &str = "AnnotationsSidebar";

const ACTION_TOGGLE_SIDEBAR: &str = "toggle_annotations";
const ACTION_CREATE_POINT: &str = "create_point_annotation";
const ACTION_CREATE_POLYGON: &str = "create_polygon_annotation";
const VIEWPORT_MENU_CREATE_POINT: &str = "create_point";
const VIEWPORT_MENU_CREATE_POLYGON: &str = "create_polygon";

const SIDEBAR_ICON_SVG: &str = include_str!("../../../app/ui/icons/annotations.svg");
const POINT_ICON_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.5" fill="currentColor"/></svg>"#;
const POLYGON_ICON_SVG: &str = include_str!("../ui/icons/polygon_annotation.svg");

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    set_host_api(host_api);
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![
        ToolbarButtonFFI {
            button_id: RString::from(ACTION_TOGGLE_SIDEBAR),
            tooltip: RString::from("Annotations"),
            icon_svg: RString::from(SIDEBAR_ICON_SVG),
            action_id: RString::from(ACTION_TOGGLE_SIDEBAR),
            tool_mode: ROption::RNone,
            hotkey: ROption::RNone,
        },
        ToolbarButtonFFI {
            button_id: RString::from(ACTION_CREATE_POINT),
            tooltip: RString::from("Create Point Annotation"),
            icon_svg: RString::from(POINT_ICON_SVG),
            action_id: RString::from(ACTION_CREATE_POINT),
            tool_mode: ROption::RSome(HostToolModeFFI::PointAnnotation),
            hotkey: ROption::RSome("1".into()),
        },
        ToolbarButtonFFI {
            button_id: RString::from(ACTION_CREATE_POLYGON),
            tooltip: RString::from("Create Polygon Annotation"),
            icon_svg: RString::from(POLYGON_ICON_SVG),
            action_id: RString::from(ACTION_CREATE_POLYGON),
            tool_mode: ROption::RSome(HostToolModeFFI::PolygonAnnotation),
            hotkey: ROption::RSome("2".into()),
        },
    ])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    let result = match action_id.as_str() {
        ACTION_TOGGLE_SIDEBAR => sync_active_file().and_then(|_| {
            let Some(host_api) = host_api() else {
                return Err("host API is not available".to_string());
            };
            (host_api.show_sidebar)(
                host_api.context,
                RString::from(ACTION_TOGGLE_SIDEBAR),
                SIDEBAR_WIDTH_PX,
                RString::from(SIDEBAR_UI_PATH),
                RString::from(SIDEBAR_COMPONENT),
            )
            .into_result()
            .map_err(|err| format!("failed to toggle annotations sidebar: {err}"))
        }),
        ACTION_CREATE_POINT => start_point_annotation_flow(),
        ACTION_CREATE_POLYGON => start_polygon_annotation_flow(),
        _ => Ok(()),
    };

    if let Err(err) = result {
        log_message(HostLogLevelFFI::Error, err);
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
    on_sidebar_callback(callback_name.as_str(), args_json.as_str());
}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    get_sidebar_properties()
}

extern "C" fn get_viewport_context_menu_items_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    if viewport.file_id < 0 || viewport.file_path.is_empty() {
        return RVec::new();
    }

    RVec::from(vec![
        ViewportContextMenuItemFFI {
            item_id: VIEWPORT_MENU_CREATE_POINT.into(),
            label: "Create Point".into(),
            icon: "point".into(),
            enabled: true,
        },
        ViewportContextMenuItemFFI {
            item_id: VIEWPORT_MENU_CREATE_POLYGON.into(),
            label: "Create Polygon".into(),
            icon: "polygon".into(),
            enabled: true,
        },
    ])
}

extern "C" fn on_viewport_context_menu_action_ffi(
    item_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    if item_id.as_str() == VIEWPORT_MENU_CREATE_POINT
        && let Err(err) = start_point_annotation_flow()
    {
        log_message(HostLogLevelFFI::Error, err);
    } else if item_id.as_str() == VIEWPORT_MENU_CREATE_POLYGON
        && let Err(err) = start_polygon_annotation_flow()
    {
        log_message(HostLogLevelFFI::Error, err);
    }

    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_overlay_points_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    if viewport.file_id < 0 || viewport.file_path.is_empty() {
        return RVec::new();
    }
    if let Err(err) = ensure_loaded_for_viewport(&viewport) {
        log_message(HostLogLevelFFI::Error, err);
        return RVec::new();
    }

    let state = plugin_state().lock().unwrap();
    let Some(loaded) = state.files.get(viewport.file_path.as_str()) else {
        return RVec::new();
    };
    let hidden_sets = state.hidden_sets_by_file.get(viewport.file_path.as_str());

    let points = loaded
        .annotation_sets
        .iter()
        .filter(|set| !hidden_sets.is_some_and(|hidden| hidden.contains(&set.id)))
        .flat_map(|set| {
            let (ring_red, ring_green, ring_blue) = hex_color_to_rgb(&set.color_hex);
            set.annotations
                .iter()
                .map(move |annotation| match annotation {
                    Annotation::Point(point) => ViewportOverlayPointFFI {
                        annotation_id: point.id.clone().into(),
                        x_level0: point.x_level0,
                        y_level0: point.y_level0,
                        diameter_px: 12.0,
                        ring_red,
                        ring_green,
                        ring_blue,
                    },
                    Annotation::Polygon(_) => ViewportOverlayPointFFI {
                        annotation_id: "".into(),
                        x_level0: 0.0,
                        y_level0: 0.0,
                        diameter_px: 0.0,
                        ring_red,
                        ring_green,
                        ring_blue,
                    },
                })
        })
        .filter(|point| !point.annotation_id.is_empty())
        .collect::<Vec<_>>();
    RVec::from(points)
}

extern "C" fn get_viewport_overlay_polygons_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    if viewport.file_id < 0 || viewport.file_path.is_empty() {
        return RVec::new();
    }
    if let Err(err) = ensure_loaded_for_viewport(&viewport) {
        log_message(HostLogLevelFFI::Error, err);
        return RVec::new();
    }

    let state = plugin_state().lock().unwrap();
    let Some(loaded) = state.files.get(viewport.file_path.as_str()) else {
        return RVec::new();
    };
    let hidden_sets = state.hidden_sets_by_file.get(viewport.file_path.as_str());

    let polygons = loaded
        .annotation_sets
        .iter()
        .filter(|set| !hidden_sets.is_some_and(|hidden| hidden.contains(&set.id)))
        .flat_map(|set| {
            let (fill_red, fill_green, fill_blue) = hex_color_to_rgb(&set.color_hex);
            set.annotations
                .iter()
                .filter_map(move |annotation| match annotation {
                    Annotation::Polygon(polygon) => Some(ViewportOverlayPolygonFFI {
                        annotation_id: polygon.id.clone().into(),
                        vertices: polygon
                            .vertices
                            .iter()
                            .map(|vertex| ViewportOverlayVertexFFI {
                                x_level0: vertex.x_level0,
                                y_level0: vertex.y_level0,
                            })
                            .collect(),
                        fill_red,
                        fill_green,
                        fill_blue,
                    }),
                    Annotation::Point(_) => None,
                })
        })
        .collect::<Vec<_>>();
    RVec::from(polygons)
}

extern "C" fn on_point_annotation_placed_ffi(
    viewport: ViewportSnapshotFFI,
    x_level0: f64,
    y_level0: f64,
) {
    match persist_point_annotation(&viewport, x_level0, y_level0) {
        Ok(()) => {
            refresh_sidebar_if_available();
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

extern "C" fn on_point_annotation_moved_ffi(
    viewport: ViewportSnapshotFFI,
    annotation_id: RString,
    x_level0: f64,
    y_level0: f64,
) {
    match move_point_annotation(&viewport, annotation_id.as_str(), x_level0, y_level0) {
        Ok(()) => {
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

extern "C" fn on_polygon_annotation_placed_ffi(
    viewport: ViewportSnapshotFFI,
    vertices: RVec<ViewportOverlayVertexFFI>,
) {
    match persist_polygon_annotation(&viewport, vertices.as_slice()) {
        Ok(()) => {
            refresh_sidebar_if_available();
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

extern "C" fn on_polygon_annotation_moved_ffi(
    viewport: ViewportSnapshotFFI,
    annotation_id: RString,
    vertices: RVec<ViewportOverlayVertexFFI>,
) {
    match move_polygon_annotation(&viewport, annotation_id.as_str(), vertices.as_slice()) {
        Ok(()) => {
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    RVec::new()
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    _rgba_data: *mut u8,
    _len: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

extern "C" fn apply_filter_gpu_ffi(_filter_id: RString, _ctx: *const GpuFilterContextFFI) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {}

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
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}
