use crate::{AppWindow, open_file, request_render_loop};
use abi_stable::std_types::{ROption, RResult, RString, RVec};
use common::viewport::{MAX_ZOOM, MIN_ZOOM};
use common::{FilteringMode, RenderBackend, TileCache};
use parking_lot::RwLock;
use plugin_api::HostToolMode;
use plugin_api::IconDescriptor;
use plugin_api::ffi::{
    ActiveSidebarFFI, ConfirmationDialogRequestFFI, HostApiVTable, HostLogLevelFFI,
    HostSnapshotFFI, HostToolModeFFI, OpenFileInfoFFI, PluginVTable, UiPropertyFFI,
    ViewportContextMenuItemFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportSnapshotFFI,
};
use slint::{
    Color, ComponentFactory, ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, Timer,
    VecModel,
};
use slint_interpreter::json::{value_from_json_str, value_to_json};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::state::{AppState, PaneId};

#[derive(Clone)]
pub(crate) struct PluginPolygonDragCandidate {
    pub handle: crate::state::PluginAnnotationHandle,
    pub vertices: Vec<crate::state::ImagePoint>,
}

#[derive(Clone)]
pub(crate) struct PluginPolygonVertexDragCandidate {
    pub handle: crate::state::PluginPolygonVertexHandle,
    pub vertices: Vec<crate::state::ImagePoint>,
}

#[derive(Clone)]
pub(crate) struct PluginPolygonEdgeInsertCandidate {
    pub handle: crate::state::PluginPolygonEdgeHandle,
    pub vertices: Vec<crate::state::ImagePoint>,
}

struct OverlayPolygonShape {
    plugin_id: String,
    annotation_id: String,
    vertices: Vec<crate::state::ImagePoint>,
    fill_color: Color,
    stroke_color: Color,
}

fn polygon_fill_color(red: u8, green: u8, blue: u8, hovered: bool) -> Color {
    Color::from_argb_u8(if hovered { 0x58 } else { 0x40 }, red, green, blue)
}

fn polygon_path_commands(vertices: &[crate::state::ImagePoint]) -> String {
    let mut commands = String::new();
    for (index, vertex) in vertices.iter().enumerate() {
        let _ = write!(
            &mut commands,
            "{} {:.3} {:.3} ",
            if index == 0 { 'M' } else { 'L' },
            vertex.x,
            vertex.y
        );
    }
    if vertices.len() >= 3 {
        commands.push('Z');
    }
    commands.trim().to_string()
}

fn point_in_polygon(
    point: crate::state::ImagePoint,
    vertices: &[crate::state::ImagePoint],
) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = vertices.last().copied().unwrap_or_default();
    for current in vertices.iter().copied() {
        let crosses = ((current.y > point.y) != (previous.y > point.y))
            && (point.x
                < (previous.x - current.x) * (point.y - current.y)
                    / ((previous.y - current.y) + f64::EPSILON)
                    + current.x);
        if crosses {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn preview_polygon_vertices(
    drag_state: &crate::state::PluginPolygonDragState,
    current: crate::state::ImagePoint,
) -> Vec<crate::state::ImagePoint> {
    let dx = current.x - drag_state.start_pointer.x;
    let dy = current.y - drag_state.start_pointer.y;
    drag_state
        .vertices
        .iter()
        .map(|vertex| crate::state::ImagePoint {
            x: vertex.x + dx,
            y: vertex.y + dy,
        })
        .collect()
}

fn preview_polygon_vertex_drag(
    drag_state: &crate::state::PluginPolygonVertexDragState,
    current: crate::state::ImagePoint,
) -> Vec<crate::state::ImagePoint> {
    let mut vertices = drag_state.vertices.clone();
    if let Some(vertex) = vertices.get_mut(drag_state.vertex_index) {
        *vertex = current;
    }
    vertices
}

fn closest_point_on_segment(
    point: crate::state::ImagePoint,
    start: crate::state::ImagePoint,
    end: crate::state::ImagePoint,
) -> (crate::state::ImagePoint, f64) {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_sq = dx * dx + dy * dy;
    if length_sq <= f64::EPSILON {
        return (start, 0.0);
    }
    let t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq;
    let t = t.clamp(0.0, 1.0);
    (
        crate::state::ImagePoint {
            x: start.x + dx * t,
            y: start.y + dy * t,
        },
        t,
    )
}

struct HostApiContext {
    plugin_id: String,
    plugin_root: PathBuf,
    state: Arc<RwLock<AppState>>,
    vtable: PluginVTable,
}

struct UiRuntime {
    ui_weak: slint::Weak<AppWindow>,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    render_timer: Rc<Timer>,
}

struct ActiveSidebarInstance {
    plugin_id: String,
    instance: slint_interpreter::Weak<slint_interpreter::ComponentInstance>,
}

struct PendingPluginConfirmation {
    vtable: PluginVTable,
    confirm_callback: Option<String>,
    confirm_args_json: String,
    cancel_callback: Option<String>,
    cancel_args_json: String,
}

static HOST_CONTEXTS: OnceLock<Mutex<HashMap<u64, HostApiContext>>> = OnceLock::new();
static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
static UI_THREAD_ID: OnceLock<std::thread::ThreadId> = OnceLock::new();
static PENDING_PLUGIN_CONFIRMATION: OnceLock<Mutex<Option<PendingPluginConfirmation>>> =
    OnceLock::new();

thread_local! {
    static UI_RUNTIME: RefCell<Option<UiRuntime>> = const { RefCell::new(None) };
    static ACTIVE_SIDEBAR_INSTANCE: RefCell<Option<ActiveSidebarInstance>> = const { RefCell::new(None) };
}

fn host_contexts() -> &'static Mutex<HashMap<u64, HostApiContext>> {
    HOST_CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pending_plugin_confirmation() -> &'static Mutex<Option<PendingPluginConfirmation>> {
    PENDING_PLUGIN_CONFIRMATION.get_or_init(|| Mutex::new(None))
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
    vtable: PluginVTable,
) -> HostApiVTable {
    let context = NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
    host_contexts().lock().unwrap().insert(
        context,
        HostApiContext {
            plugin_id: plugin_id.to_string(),
            plugin_root: plugin_root.to_path_buf(),
            state: Arc::clone(state),
            vtable,
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
        set_active_tool: ffi_set_active_tool,
        request_render: ffi_request_render,
        set_toolbar_button_active: ffi_set_toolbar_button_active,
        set_hud_toolbar_button_active: ffi_set_hud_toolbar_button_active,
        show_sidebar: ffi_show_sidebar,
        refresh_sidebar: ffi_refresh_sidebar,
        hide_sidebar: ffi_hide_sidebar,
        show_confirmation_dialog: ffi_show_confirmation_dialog,
        save_file_dialog: ffi_save_file_dialog,
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

pub(crate) fn sync_tool_button_states(state: &mut AppState) {
    let point_owner = state.active_tool_plugin_id.as_deref();
    let current_tool = state.current_tool;

    for button in &mut state.local_plugin_buttons {
        if let Some(tool_mode) = button.tool_mode {
            button.active =
                tool_mode_matches_state(tool_mode, &button.plugin_id, current_tool, point_owner);
        }
    }
}

pub(crate) fn hotkey_button_for_key(
    state: &AppState,
    key: &str,
) -> Option<plugin_api::ToolbarButtonRegistration> {
    state
        .local_plugin_buttons
        .iter()
        .find(|button| button.hotkey.as_deref() == Some(key) && button.tool_mode.is_some())
        .cloned()
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
        .and_then(|file| {
            file.pane_state(pane)
                .map(|pane_state| to_viewport_snapshot(file, &pane_state.viewport, pane))
        })
}

pub(crate) fn viewport_overlay_points_for_pane(
    state: &AppState,
    pane: PaneId,
) -> Vec<crate::PluginOverlayPoint> {
    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };
    let Some(pane_state) = file.pane_state(pane) else {
        return Vec::new();
    };

    let snapshot = to_viewport_snapshot(file, &pane_state.viewport, pane);
    let snapshot_ffi = to_viewport_snapshot_ffi(snapshot);
    let vp = &pane_state.viewport.viewport;
    let hovered = state.hovered_plugin_annotation.as_ref();
    let dragged = state.dragged_plugin_point.as_ref();
    let dragged_position = state.dragged_plugin_point_position;
    let active_tool_plugin_id = state.active_tool_plugin_id.as_deref();
    let point_tool_active = state.current_tool == crate::state::Tool::PointAnnotation;

    local_plugin_vtables()
        .into_iter()
        .flat_map(|(plugin_id, vtable)| {
            (vtable.get_viewport_overlay_points)(snapshot_ffi.clone())
                .into_iter()
                .map(move |point| (plugin_id.clone(), point))
        })
        .map(|(plugin_id, point): (String, ViewportOverlayPointFFI)| {
            let annotation_id = point.annotation_id.to_string();
            let (x_level0, y_level0) = if dragged.is_some_and(|handle| {
                handle.plugin_id == plugin_id && handle.annotation_id == annotation_id
            }) {
                dragged_position
                    .map(|preview| (preview.x_level0, preview.y_level0))
                    .unwrap_or((point.x_level0, point.y_level0))
            } else {
                (point.x_level0, point.y_level0)
            };
            let screen = vp.image_to_screen(x_level0, y_level0);
            let ring_color = if point_tool_active
                && active_tool_plugin_id == Some(plugin_id.as_str())
                && hovered.is_some_and(|handle| {
                    handle.plugin_id == plugin_id && handle.annotation_id == annotation_id
                }) {
                Color::from_rgb_u8(0xF1, 0xC4, 0x0F)
            } else {
                Color::from_rgb_u8(point.ring_red, point.ring_green, point.ring_blue)
            };
            crate::PluginOverlayPoint {
                plugin_id: plugin_id.into(),
                annotation_id: annotation_id.into(),
                x: screen.x as f32,
                y: screen.y as f32,
                diameter_px: point.diameter_px,
                ring_color,
            }
        })
        .collect()
}

pub(crate) fn hit_test_overlay_point_for_pane(
    state: &AppState,
    pane: PaneId,
    screen_x: f32,
    screen_y: f32,
    plugin_id_filter: Option<&str>,
) -> Option<crate::state::PluginAnnotationHandle> {
    viewport_overlay_points_for_pane(state, pane)
        .into_iter()
        .filter(|point| {
            plugin_id_filter.is_none_or(|plugin_id| point.plugin_id.as_str() == plugin_id)
        })
        .filter_map(|point| {
            let dx = screen_x - point.x;
            let dy = screen_y - point.y;
            let radius = point.diameter_px * 0.5 + 3.0;
            let distance_sq = dx * dx + dy * dy;
            (distance_sq <= radius * radius).then_some((distance_sq, point))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, point)| crate::state::PluginAnnotationHandle {
            plugin_id: point.plugin_id.to_string(),
            annotation_id: point.annotation_id.to_string(),
        })
}

fn overlay_polygon_shapes_for_pane(state: &AppState, pane: PaneId) -> Vec<OverlayPolygonShape> {
    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };
    let Some(pane_state) = file.pane_state(pane) else {
        return Vec::new();
    };

    let snapshot = to_viewport_snapshot(file, &pane_state.viewport, pane);
    let snapshot_ffi = to_viewport_snapshot_ffi(snapshot);
    let hovered = state.hovered_plugin_annotation.as_ref();
    let dragged = state.dragged_plugin_polygon.as_ref();
    let dragged_state = state.dragged_plugin_polygon_state.as_ref();
    let dragged_position = state.dragged_plugin_polygon_position;
    let dragged_vertex = state.dragged_plugin_polygon_vertex.as_ref();
    let dragged_vertex_state = state.dragged_plugin_polygon_vertex_state.as_ref();
    let dragged_vertex_position = state.dragged_plugin_polygon_vertex_position;
    let active_tool_plugin_id = state.active_tool_plugin_id.as_deref();
    let polygon_tool_active = state.current_tool == crate::state::Tool::PolygonAnnotation;

    let mut polygons = local_plugin_vtables()
        .into_iter()
        .flat_map(|(plugin_id, vtable)| {
            (vtable.get_viewport_overlay_polygons)(snapshot_ffi.clone())
                .into_iter()
                .map(move |polygon| (plugin_id.clone(), polygon))
        })
        .map(
            |(plugin_id, polygon): (String, ViewportOverlayPolygonFFI)| {
                let annotation_id = polygon.annotation_id.to_string();
                let vertices = if dragged.is_some_and(|handle| {
                    handle.plugin_id == plugin_id && handle.annotation_id == annotation_id
                }) {
                    match (dragged_state, dragged_position) {
                        (Some(drag_state), Some(current)) => {
                            preview_polygon_vertices(drag_state, current)
                        }
                        _ => polygon
                            .vertices
                            .iter()
                            .map(|vertex| crate::state::ImagePoint {
                                x: vertex.x_level0,
                                y: vertex.y_level0,
                            })
                            .collect(),
                    }
                } else if dragged_vertex.is_some_and(|handle| {
                    handle.plugin_id == plugin_id && handle.annotation_id == annotation_id
                }) {
                    match (dragged_vertex_state, dragged_vertex_position) {
                        (Some(drag_state), Some(current)) => {
                            preview_polygon_vertex_drag(drag_state, current)
                        }
                        _ => polygon
                            .vertices
                            .iter()
                            .map(|vertex| crate::state::ImagePoint {
                                x: vertex.x_level0,
                                y: vertex.y_level0,
                            })
                            .collect(),
                    }
                } else {
                    polygon
                        .vertices
                        .iter()
                        .map(|vertex| crate::state::ImagePoint {
                            x: vertex.x_level0,
                            y: vertex.y_level0,
                        })
                        .collect()
                };
                let hovered = polygon_tool_active
                    && active_tool_plugin_id == Some(plugin_id.as_str())
                    && hovered.is_some_and(|handle| {
                        handle.plugin_id == plugin_id && handle.annotation_id == annotation_id
                    });
                OverlayPolygonShape {
                    plugin_id,
                    annotation_id,
                    vertices,
                    fill_color: polygon_fill_color(
                        polygon.fill_red,
                        polygon.fill_green,
                        polygon.fill_blue,
                        hovered,
                    ),
                    stroke_color: if hovered {
                        Color::from_rgb_u8(0xF1, 0xC4, 0x0F)
                    } else {
                        Color::from_rgb_u8(0x00, 0x00, 0x00)
                    },
                }
            },
        )
        .collect::<Vec<_>>();

    if state.current_tool == crate::state::Tool::PolygonAnnotation
        && pane == state.focused_pane
        && !state.polygon_candidate_vertices.is_empty()
    {
        let mut candidate_vertices = state.polygon_candidate_vertices.clone();
        if let Some(hover) = state.polygon_candidate_hover {
            candidate_vertices.push(hover);
        }
        polygons.push(OverlayPolygonShape {
            plugin_id: String::new(),
            annotation_id: String::new(),
            vertices: candidate_vertices,
            fill_color: Color::from_argb_u8(0x40, 0xFF, 0xD1, 0x66),
            stroke_color: Color::from_rgb_u8(0xF1, 0xC4, 0x0F),
        });
    }

    polygons
}

pub(crate) fn viewport_overlay_polygon_vertex_boxes_for_pane(
    state: &AppState,
    pane: PaneId,
) -> Vec<crate::PluginOverlayVertexBox> {
    if state.current_tool != crate::state::Tool::PolygonAnnotation {
        return Vec::new();
    }

    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };
    let Some(pane_state) = file.pane_state(pane) else {
        return Vec::new();
    };
    let vp = &pane_state.viewport.viewport;

    let active = state
        .dragged_plugin_polygon_vertex
        .as_ref()
        .map(|handle| (handle.plugin_id.as_str(), handle.annotation_id.as_str()))
        .or_else(|| {
            state
                .dragged_plugin_polygon
                .as_ref()
                .map(|handle| (handle.plugin_id.as_str(), handle.annotation_id.as_str()))
        })
        .or_else(|| {
            state
                .hovered_plugin_annotation
                .as_ref()
                .map(|handle| (handle.plugin_id.as_str(), handle.annotation_id.as_str()))
        });

    let Some((plugin_id, annotation_id)) = active else {
        return Vec::new();
    };

    let hovered_vertex = state.hovered_plugin_polygon_vertex.as_ref();
    let hovered_edge = state.hovered_plugin_polygon_edge.as_ref();

    overlay_polygon_shapes_for_pane(state, pane)
        .into_iter()
        .find(|polygon| polygon.plugin_id == plugin_id && polygon.annotation_id == annotation_id)
        .map(|polygon| {
            let mut boxes = polygon
                .vertices
                .iter()
                .enumerate()
                .map(|(vertex_index, vertex)| {
                    let screen = vp.image_to_screen(vertex.x, vertex.y);
                    let is_hovered = hovered_vertex.is_some_and(|handle| {
                        handle.plugin_id == polygon.plugin_id
                            && handle.annotation_id == polygon.annotation_id
                            && handle.vertex_index == vertex_index
                    });
                    crate::PluginOverlayVertexBox {
                        plugin_id: polygon.plugin_id.clone().into(),
                        annotation_id: polygon.annotation_id.clone().into(),
                        vertex_index: vertex_index as i32,
                        x: screen.x as f32,
                        y: screen.y as f32,
                        size_px: 8.0,
                        fill_color: if is_hovered {
                            Color::from_rgb_u8(0xF1, 0xC4, 0x0F)
                        } else {
                            Color::from_rgb_u8(0x00, 0x00, 0x00)
                        },
                        border_color: if is_hovered {
                            Color::from_rgb_u8(0x00, 0x00, 0x00)
                        } else {
                            Color::from_rgb_u8(0xF1, 0xC4, 0x0F)
                        },
                    }
                })
                .collect::<Vec<_>>();

            if let Some(edge) = hovered_edge.filter(|edge| {
                edge.plugin_id == polygon.plugin_id && edge.annotation_id == polygon.annotation_id
            }) {
                let screen = vp.image_to_screen(edge.position.x, edge.position.y);
                boxes.push(crate::PluginOverlayVertexBox {
                    plugin_id: polygon.plugin_id.clone().into(),
                    annotation_id: polygon.annotation_id.clone().into(),
                    vertex_index: edge.insert_index as i32,
                    x: screen.x as f32,
                    y: screen.y as f32,
                    size_px: 8.0,
                    fill_color: Color::from_rgb_u8(0xF1, 0xC4, 0x0F),
                    border_color: Color::from_rgb_u8(0x00, 0x00, 0x00),
                });
            }

            boxes
        })
        .unwrap_or_default()
}

pub(crate) fn viewport_overlay_polygons_for_pane(
    state: &AppState,
    pane: PaneId,
) -> Vec<crate::PluginOverlayPolygon> {
    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };
    let Some(pane_state) = file.pane_state(pane) else {
        return Vec::new();
    };
    let vp = &pane_state.viewport.viewport;

    overlay_polygon_shapes_for_pane(state, pane)
        .into_iter()
        .filter_map(|polygon| {
            let screen_vertices = polygon
                .vertices
                .iter()
                .map(|vertex| {
                    let screen = vp.image_to_screen(vertex.x, vertex.y);
                    crate::state::ImagePoint {
                        x: screen.x,
                        y: screen.y,
                    }
                })
                .collect::<Vec<_>>();
            let commands = polygon_path_commands(&screen_vertices);
            (!commands.is_empty()).then_some(crate::PluginOverlayPolygon {
                plugin_id: polygon.plugin_id.into(),
                annotation_id: polygon.annotation_id.into(),
                path_commands: commands.into(),
                fill_color: polygon.fill_color,
                stroke_color: polygon.stroke_color,
            })
        })
        .collect()
}

pub(crate) fn hit_test_overlay_polygon_for_pane(
    state: &AppState,
    pane: PaneId,
    screen_x: f32,
    screen_y: f32,
    plugin_id_filter: Option<&str>,
) -> Option<PluginPolygonDragCandidate> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file(file_id)?;
    let pane_state = file.pane_state(pane)?;
    let image = pane_state
        .viewport
        .screen_to_image(screen_x as f64, screen_y as f64);
    let point = crate::state::ImagePoint {
        x: image.0,
        y: image.1,
    };

    overlay_polygon_shapes_for_pane(state, pane)
        .into_iter()
        .filter(|polygon| {
            !polygon.annotation_id.is_empty()
                && plugin_id_filter.is_none_or(|plugin_id| polygon.plugin_id == plugin_id)
        })
        .find(|polygon| point_in_polygon(point, &polygon.vertices))
        .map(|polygon| PluginPolygonDragCandidate {
            handle: crate::state::PluginAnnotationHandle {
                plugin_id: polygon.plugin_id,
                annotation_id: polygon.annotation_id,
            },
            vertices: polygon.vertices,
        })
}

pub(crate) fn hit_test_overlay_polygon_vertex_for_pane(
    state: &AppState,
    pane: PaneId,
    screen_x: f32,
    screen_y: f32,
    plugin_id_filter: Option<&str>,
) -> Option<PluginPolygonVertexDragCandidate> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file(file_id)?;
    let pane_state = file.pane_state(pane)?;
    let vp = &pane_state.viewport.viewport;

    overlay_polygon_shapes_for_pane(state, pane)
        .into_iter()
        .filter(|polygon| {
            !polygon.annotation_id.is_empty()
                && plugin_id_filter.is_none_or(|plugin_id| polygon.plugin_id == plugin_id)
        })
        .flat_map(|polygon| {
            let plugin_id = polygon.plugin_id.clone();
            let annotation_id = polygon.annotation_id.clone();
            let vertices = polygon.vertices.clone();
            polygon
                .vertices
                .into_iter()
                .enumerate()
                .map(move |(vertex_index, vertex)| {
                    let screen = vp.image_to_screen(vertex.x, vertex.y);
                    let dx = screen_x as f64 - screen.x;
                    let dy = screen_y as f64 - screen.y;
                    let distance_sq = dx * dx + dy * dy;
                    (
                        distance_sq,
                        PluginPolygonVertexDragCandidate {
                            handle: crate::state::PluginPolygonVertexHandle {
                                plugin_id: plugin_id.clone(),
                                annotation_id: annotation_id.clone(),
                                vertex_index,
                            },
                            vertices: vertices.clone(),
                        },
                    )
                })
        })
        .filter(|(distance_sq, _)| *distance_sq <= 36.0)
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, candidate)| candidate)
}

pub(crate) fn hit_test_overlay_polygon_edge_for_pane(
    state: &AppState,
    pane: PaneId,
    screen_x: f32,
    screen_y: f32,
    plugin_id_filter: Option<&str>,
) -> Option<PluginPolygonEdgeInsertCandidate> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file(file_id)?;
    let pane_state = file.pane_state(pane)?;
    let vp = &pane_state.viewport.viewport;
    let pointer = crate::state::ImagePoint {
        x: screen_x as f64,
        y: screen_y as f64,
    };

    overlay_polygon_shapes_for_pane(state, pane)
        .into_iter()
        .filter(|polygon| {
            !polygon.annotation_id.is_empty()
                && polygon.vertices.len() >= 2
                && plugin_id_filter.is_none_or(|plugin_id| polygon.plugin_id == plugin_id)
        })
        .flat_map(|polygon| {
            let plugin_id = polygon.plugin_id.clone();
            let annotation_id = polygon.annotation_id.clone();
            let vertices = polygon.vertices.clone();
            let screen_vertices = polygon
                .vertices
                .iter()
                .map(|vertex| {
                    let screen = vp.image_to_screen(vertex.x, vertex.y);
                    crate::state::ImagePoint {
                        x: screen.x,
                        y: screen.y,
                    }
                })
                .collect::<Vec<_>>();

            (0..screen_vertices.len()).map(move |index| {
                let next_index = (index + 1) % screen_vertices.len();
                let (closest_screen, t) = closest_point_on_segment(
                    pointer,
                    screen_vertices[index],
                    screen_vertices[next_index],
                );
                let dx = pointer.x - closest_screen.x;
                let dy = pointer.y - closest_screen.y;
                let insertion_point = crate::state::ImagePoint {
                    x: vertices[index].x + (vertices[next_index].x - vertices[index].x) * t,
                    y: vertices[index].y + (vertices[next_index].y - vertices[index].y) * t,
                };
                let insert_index = if next_index == 0 {
                    vertices.len()
                } else {
                    next_index
                };
                (
                    dx * dx + dy * dy,
                    PluginPolygonEdgeInsertCandidate {
                        handle: crate::state::PluginPolygonEdgeHandle {
                            plugin_id: plugin_id.clone(),
                            annotation_id: annotation_id.clone(),
                            insert_index,
                            position: insertion_point,
                        },
                        vertices: vertices.clone(),
                    },
                )
            })
        })
        .filter(|(distance_sq, _)| *distance_sq <= 36.0)
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, candidate)| candidate)
}

pub(crate) fn viewport_context_menu_items_for_pane(
    state: &AppState,
    pane: PaneId,
) -> Vec<crate::ContextMenuItem> {
    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };
    let Some(pane_state) = file.pane_state(pane) else {
        return Vec::new();
    };

    let snapshot = to_viewport_snapshot(file, &pane_state.viewport, pane);
    let snapshot_ffi = to_viewport_snapshot_ffi(snapshot);

    local_plugin_vtables()
        .into_iter()
        .flat_map(|(plugin_id, vtable)| {
            (vtable.get_viewport_context_menu_items)(snapshot_ffi.clone())
                .into_iter()
                .map(
                    move |item: ViewportContextMenuItemFFI| crate::ContextMenuItem {
                        id: format!("plugin-viewport:{}:{}", plugin_id, item.item_id).into(),
                        label: item.label.to_string().into(),
                        icon: item.icon.to_string().into(),
                        shortcut: slint::SharedString::default(),
                        enabled: item.enabled,
                        separator_after: false,
                    },
                )
        })
        .collect()
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

pub(crate) fn set_active_tool(plugin_id: &str, tool: HostToolModeFFI) -> Result<(), String> {
    let plugin_id = plugin_id.to_string();
    run_on_ui_thread(move |runtime| {
        {
            let mut state = runtime.state.write();
            match tool {
                HostToolModeFFI::Navigate => state.set_tool(crate::state::Tool::Navigate),
                HostToolModeFFI::RegionOfInterest => {
                    state.set_tool(crate::state::Tool::RegionOfInterest)
                }
                HostToolModeFFI::MeasureDistance => {
                    state.set_tool(crate::state::Tool::MeasureDistance)
                }
                HostToolModeFFI::PointAnnotation => state.set_plugin_annotation_tool(
                    crate::state::Tool::PointAnnotation,
                    plugin_id.clone(),
                ),
                HostToolModeFFI::PolygonAnnotation => state.set_plugin_annotation_tool(
                    crate::state::Tool::PolygonAnnotation,
                    plugin_id.clone(),
                ),
            }
            sync_tool_button_states(&mut state);
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

pub(crate) fn request_render() -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        runtime.state.write().request_render();
        request_render_loop(
            &runtime.render_timer,
            &runtime.ui_weak,
            &runtime.state,
            &runtime.tile_cache,
        );
        Ok(())
    })
}

pub(crate) fn save_file_dialog(
    default_file_name: String,
    filter_name: String,
    extension: String,
) -> Result<String, String> {
    run_on_ui_thread(move |_runtime| {
        let dialog = rfd::FileDialog::new()
            .add_filter(&filter_name, &[extension.as_str()])
            .set_file_name(default_file_name);
        dialog
            .save_file()
            .map(|path| path.to_string_lossy().into_owned())
            .ok_or_else(|| "save dialog was cancelled".to_string())
    })
}

pub(crate) fn show_confirmation_dialog(
    vtable: PluginVTable,
    request: ConfirmationDialogRequestFFI,
) -> Result<(), String> {
    let title = request.title.to_string();
    let message = request.message.to_string();
    let confirm_label = request.confirm_label.to_string();
    let cancel_label = request.cancel_label.to_string();
    let confirm_callback = match request.confirm_callback {
        ROption::RSome(value) => Some(value.to_string()),
        ROption::RNone => None,
    };
    let confirm_args_json = match request.confirm_args_json {
        ROption::RSome(value) => value.to_string(),
        ROption::RNone => "[]".to_string(),
    };
    let cancel_callback = match request.cancel_callback {
        ROption::RSome(value) => Some(value.to_string()),
        ROption::RNone => None,
    };
    let cancel_args_json = match request.cancel_args_json {
        ROption::RSome(value) => value.to_string(),
        ROption::RNone => "[]".to_string(),
    };

    run_on_ui_thread(move |runtime| {
        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;
        *pending_plugin_confirmation().lock().unwrap() = Some(PendingPluginConfirmation {
            vtable,
            confirm_callback,
            confirm_args_json,
            cancel_callback,
            cancel_args_json,
        });
        ui.set_plugin_confirm_title(title.into());
        ui.set_plugin_confirm_message(message.into());
        ui.set_plugin_confirm_confirm_label(confirm_label.into());
        ui.set_plugin_confirm_cancel_label(cancel_label.into());
        ui.set_plugin_confirm_visible(true);
        Ok(())
    })
}

pub(crate) fn handle_confirmation_dialog_response(confirmed: bool) -> Result<(), String> {
    run_on_ui_thread(move |runtime| {
        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;
        ui.set_plugin_confirm_visible(false);

        let pending = pending_plugin_confirmation().lock().unwrap().take();
        let Some(pending) = pending else {
            return Ok(());
        };

        let (callback_name, args_json) = if confirmed {
            (pending.confirm_callback, pending.confirm_args_json)
        } else {
            (pending.cancel_callback, pending.cancel_args_json)
        };

        if let Some(callback_name) = callback_name {
            (pending.vtable.on_ui_callback)(callback_name.into(), args_json.into());
        }

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
    vtable: Option<PluginVTable>,
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
                vtable,
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
                ACTIVE_SIDEBAR_INSTANCE.with(|slot| {
                    *slot.borrow_mut() = None;
                });
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
        ACTIVE_SIDEBAR_INSTANCE.with(|slot| {
            *slot.borrow_mut() = None;
        });
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
        .and_then(|file| {
            file.pane_state(focused_pane)
                .map(|pane_state| to_viewport_snapshot(file, &pane_state.viewport, focused_pane))
        });

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
    file: &crate::state::OpenFile,
    viewport: &common::ViewportState,
    pane: PaneId,
) -> plugin_api::ViewportSnapshot {
    let bounds = viewport.viewport.bounds();
    plugin_api::ViewportSnapshot {
        pane_index: pane.0 as u32,
        file_id: file.id,
        file_path: file.path.to_string_lossy().into_owned(),
        filename: file.filename.clone(),
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
    vtable: Option<PluginVTable>,
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

        if let Some(vtable) = vtable
            && let Err(err) = apply_sidebar_properties(&plugin_id, vtable, &instance)
        {
            tracing::error!(
                "Failed to apply sidebar properties for '{}': {err}",
                plugin_id
            );
        }

        ACTIVE_SIDEBAR_INSTANCE.with(|slot| {
            *slot.borrow_mut() = Some(ActiveSidebarInstance {
                plugin_id: plugin_id.clone(),
                instance: instance.as_weak(),
            });
        });

        for name in definition.callbacks() {
            let callback_name = name.to_string();
            let plugin_id = plugin_id.clone();
            let Some(vtable) = vtable else {
                continue;
            };
            if instance
                .set_callback(&name, move |args| {
                    let args_json = serde_json::Value::Array(
                        args.iter()
                            .map(value_to_json)
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_default(),
                    )
                    .to_string();
                    (vtable.on_ui_callback)(
                        RString::from(callback_name.as_str()),
                        RString::from(args_json),
                    );
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

fn tool_mode_matches_state(
    tool_mode: HostToolMode,
    plugin_id: &str,
    current_tool: crate::state::Tool,
    point_owner: Option<&str>,
) -> bool {
    match tool_mode {
        HostToolMode::Navigate => current_tool == crate::state::Tool::Navigate,
        HostToolMode::RegionOfInterest => current_tool == crate::state::Tool::RegionOfInterest,
        HostToolMode::MeasureDistance => current_tool == crate::state::Tool::MeasureDistance,
        HostToolMode::PointAnnotation => {
            current_tool == crate::state::Tool::PointAnnotation && point_owner == Some(plugin_id)
        }
        HostToolMode::PolygonAnnotation => {
            current_tool == crate::state::Tool::PolygonAnnotation && point_owner == Some(plugin_id)
        }
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
        file_id: viewport.file_id,
        file_path: RString::from(viewport.file_path),
        filename: RString::from(viewport.filename),
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

fn context_vtable(context: u64) -> Option<PluginVTable> {
    host_contexts()
        .lock()
        .unwrap()
        .get(&context)
        .map(|ctx| ctx.vtable)
}

fn local_plugin_vtables() -> Vec<(String, PluginVTable)> {
    host_contexts()
        .lock()
        .unwrap()
        .values()
        .map(|ctx| (ctx.plugin_id.clone(), ctx.vtable))
        .collect()
}

fn apply_sidebar_properties(
    plugin_id: &str,
    vtable: PluginVTable,
    instance: &slint_interpreter::ComponentInstance,
) -> Result<(), String> {
    let property_types: Vec<_> = instance
        .definition()
        .properties_and_callbacks()
        .filter_map(|(name, (ty, _visibility))| ty.is_property_type().then_some((name, ty)))
        .collect();

    for UiPropertyFFI { name, json_value } in (vtable.get_sidebar_properties)().into_iter() {
        let property_name = name.to_string();
        let Some((_, property_type)) = property_types
            .iter()
            .find(|(candidate, _)| candidate == &property_name)
        else {
            tracing::debug!(
                "Ignoring sidebar property '{}' from plugin '{}' because the component does not expose it",
                property_name,
                plugin_id
            );
            continue;
        };
        let value = value_from_json_str(property_type, json_value.as_str()).map_err(|err| {
            format!(
                "failed to decode sidebar property '{}:{}' from JSON: {err}",
                plugin_id, property_name
            )
        })?;
        instance
            .set_property(&property_name, value)
            .map_err(|err| {
                format!(
                    "failed to set sidebar property '{}:{}': {err}",
                    plugin_id, property_name
                )
            })?;
    }

    Ok(())
}

pub(crate) fn refresh_active_sidebar() -> Result<(), String> {
    run_on_ui_thread(|runtime| {
        let active_sidebar = runtime.state.read().active_sidebar().cloned();
        let Some(sidebar) = active_sidebar else {
            ACTIVE_SIDEBAR_INSTANCE.with(|slot| {
                *slot.borrow_mut() = None;
            });
            return Ok(());
        };
        let Some((_, vtable)) = local_plugin_vtables()
            .into_iter()
            .find(|(candidate, _)| candidate == &sidebar.plugin_id)
        else {
            return Ok(());
        };

        let refreshed = ACTIVE_SIDEBAR_INSTANCE.with(|slot| -> Result<bool, String> {
            let active = slot.borrow();
            let Some(active) = active.as_ref() else {
                return Ok::<bool, String>(false);
            };
            if active.plugin_id != sidebar.plugin_id {
                return Ok::<bool, String>(false);
            }
            let Some(instance) = active.instance.upgrade() else {
                return Ok::<bool, String>(false);
            };
            apply_sidebar_properties(&sidebar.plugin_id, vtable, &instance)?;
            Ok::<bool, String>(true)
        })?;

        if refreshed {
            return Ok(());
        }

        let ui = runtime
            .ui_weak
            .upgrade()
            .ok_or_else(|| "application window is no longer available".to_string())?;
        let factory = build_sidebar_factory(
            &sidebar.plugin_id,
            &sidebar.ui_path,
            &sidebar.component,
            Some(vtable),
        )?;
        ui.set_plugin_sidebar_factory(factory);
        ui.set_show_plugin_sidebar(true);
        ui.set_plugin_sidebar_width(sidebar.width_px as f32);
        Ok(())
    })
}

pub(crate) fn invoke_local_ui_callback(
    plugin_id: &str,
    callback_name: &str,
    args_json: &str,
) -> Result<(), String> {
    let Some((_, vtable)) = local_plugin_vtables()
        .into_iter()
        .find(|(candidate, _)| candidate == plugin_id)
    else {
        return Err(format!("local plugin '{plugin_id}' is not loaded"));
    };

    (vtable.on_ui_callback)(callback_name.into(), args_json.into());
    Ok(())
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

extern "C" fn ffi_set_active_tool(context: u64, tool: HostToolModeFFI) -> RResult<(), RString> {
    let Some(plugin_id) = context_plugin_id(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    match set_active_tool(&plugin_id, tool) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_request_render(context: u64) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match request_render() {
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
        context_vtable(context),
        request,
    ) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_refresh_sidebar(context: u64) -> RResult<(), RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match refresh_active_sidebar() {
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

extern "C" fn ffi_show_confirmation_dialog(
    context: u64,
    request: ConfirmationDialogRequestFFI,
) -> RResult<(), RString> {
    let Some(vtable) = context_vtable(context) else {
        return RResult::RErr(RString::from("invalid host API context"));
    };
    match show_confirmation_dialog(vtable, request) {
        Ok(()) => RResult::ROk(()),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_save_file_dialog(
    context: u64,
    default_file_name: RString,
    filter_name: RString,
    extension: RString,
) -> RResult<RString, RString> {
    if context_state(context).is_err() {
        return RResult::RErr(RString::from("invalid host API context"));
    }
    match save_file_dialog(
        default_file_name.to_string(),
        filter_name.to_string(),
        extension.to_string(),
    ) {
        Ok(path) => RResult::ROk(RString::from(path)),
        Err(err) => RResult::RErr(RString::from(err)),
    }
}

extern "C" fn ffi_log_message(context: u64, level: HostLogLevelFFI, message: RString) {
    if let Some(plugin_id) = context_plugin_id(context) {
        log_message(&plugin_id, host_log_level(level), message.as_str());
    }
}
