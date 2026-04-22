//! UI state update functions
//!
//! This module contains functions for updating UI elements like tabs,
//! recent files, and render backend settings.

use crate::state::{AppState, HudSettings, PaneId};
use crate::tools::{pane_overlay_data, pane_viewport_state};
use crate::{
    ContextMenuItem, FilteringMode as SlintFilteringMode, HudSettings as SlintHudSettings,
    IsolatedChannel as SlintIsolatedChannel, MeasurementUnit as SlintMeasurementUnit, MetadataItem,
    MinimapRect, PaneRenderCacheEntry, PaneUiModels, PaneViewData, RecentFileData, RenderMode,
    StainNormalization as SlintStainNormalization, TabData, ViewportInfo,
};
use common::viewport::{MAX_ZOOM, MIN_ZOOM};
use common::{
    FilteringMode, MeasurementUnit, RenderBackend, StainNormalization, format_decimal,
    format_file_size, format_u64,
};
use slint::{Image, Model, SharedString, VecModel};
use std::{fs, rc::Rc};

fn build_metadata_items(
    state: &AppState,
    pane: PaneId,
    viewport_info: &ViewportInfo,
) -> Vec<MetadataItem> {
    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return Vec::new();
    };
    let Some(file) = state.get_file(file_id) else {
        return Vec::new();
    };

    let properties = file.wsi.properties();
    let vendor = properties.vendor.as_deref().unwrap_or("Unknown");
    let objective = properties
        .objective_power
        .map(|value| format!("{}x", format_decimal(value)))
        .unwrap_or_else(|| "N/A".to_string());
    let mpp = properties
        .mpp_x
        .zip(properties.mpp_y)
        .map(|(x, y)| format!("{} x {} µm/px", format_decimal(x), format_decimal(y)))
        .unwrap_or_else(|| "N/A".to_string());
    let scan_date = properties.scan_date.as_deref().unwrap_or("Unknown");
    let levels = format!(
        "{} levels | L{}",
        properties.levels.len(),
        viewport_info.level.max(0)
    );
    let file_size = fs::metadata(&file.path)
        .map(|metadata| format_file_size(metadata.len()))
        .unwrap_or_else(|_| "Unknown".to_string());

    vec![
        MetadataItem {
            label: SharedString::from("Slide"),
            value: SharedString::from(properties.filename.clone()),
        },
        MetadataItem {
            label: SharedString::from("Dimensions"),
            value: SharedString::from(format!(
                "{} x {} px",
                format_u64(properties.width),
                format_u64(properties.height)
            )),
        },
        MetadataItem {
            label: SharedString::from("Pyramid"),
            value: SharedString::from(levels),
        },
        MetadataItem {
            label: SharedString::from("File Size"),
            value: SharedString::from(file_size),
        },
        MetadataItem {
            label: SharedString::from("Vendor"),
            value: SharedString::from(vendor),
        },
        MetadataItem {
            label: SharedString::from("Objective"),
            value: SharedString::from(objective),
        },
        MetadataItem {
            label: SharedString::from("MPP"),
            value: SharedString::from(mpp),
        },
        MetadataItem {
            label: SharedString::from("Scan Date"),
            value: SharedString::from(scan_date),
        },
    ]
}

/// Convert RenderBackend to UI RenderMode
pub fn ui_render_mode(backend: RenderBackend) -> RenderMode {
    match backend {
        RenderBackend::Cpu => RenderMode::Cpu,
        RenderBackend::Gpu => RenderMode::Gpu,
    }
}

/// Convert zoom level to slider value (0.0-1.0)
pub fn zoom_to_slider_value(zoom: f64) -> f32 {
    let log_min = MIN_ZOOM.ln();
    let log_max = MAX_ZOOM.ln();
    let normalized =
        ((zoom.clamp(MIN_ZOOM, MAX_ZOOM).ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    normalized as f32
}

/// Convert slider value (0.0-1.0) to zoom level
pub fn slider_value_to_zoom(value: f32) -> f64 {
    let clamped = value.clamp(0.0, 1.0) as f64;
    let log_min = MIN_ZOOM.ln();
    let log_max = MAX_ZOOM.ln();
    (log_min + (log_max - log_min) * clamped).exp()
}

/// Check if a VecModel matches a slice of data
pub fn model_matches<T>(model: &VecModel<T>, data: &[T]) -> bool
where
    T: Clone + PartialEq + 'static,
{
    model.row_count() == data.len()
        && data
            .iter()
            .enumerate()
            .all(|(index, value)| model.row_data(index).as_ref() == Some(value))
}

/// Check if pane view data has changed (excluding content images)
pub fn pane_view_data_changed(existing: &PaneViewData, next: &PaneViewData) -> bool {
    existing.id != next.id
        || existing.content != next.content
        || existing.viewport_info != next.viewport_info
        || existing.minimap_thumbnail != next.minimap_thumbnail
        || existing.minimap_rect != next.minimap_rect
        || existing.metadata_items != next.metadata_items
        || existing.is_home_tab != next.is_home_tab
        || existing.zoom_slider_position != next.zoom_slider_position
        || existing.roi_rect != next.roi_rect
        || existing.plugin_overlay_points != next.plugin_overlay_points
        || existing.candidate_measurement != next.candidate_measurement
        || existing.is_loading != next.is_loading
        || existing.hud != next.hud
}

/// Create a hidden viewport info (default values)
pub fn hidden_viewport_info() -> ViewportInfo {
    ViewportInfo {
        center_x: 0.0,
        center_y: 0.0,
        zoom: 1.0,
        image_width: 1.0,
        image_height: 1.0,
        level: 0,
    }
}

/// Create a full minimap rect (showing entire image)
pub fn full_minimap_rect() -> MinimapRect {
    MinimapRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    }
}

/// Update the tabs UI from the current state
pub fn update_tabs(
    ui: &crate::AppWindow,
    state: &AppState,
    pane_render_cache: &mut Vec<PaneRenderCacheEntry>,
    pane_ui_models: &mut Vec<PaneUiModels>,
    pane_view_model: &Rc<VecModel<PaneViewData>>,
) {
    // Ensure cache and models are sized correctly
    if pane_render_cache.len() < state.panes.len() {
        pane_render_cache.resize_with(state.panes.len(), PaneRenderCacheEntry::default);
    } else if pane_render_cache.len() > state.panes.len() {
        pane_render_cache.truncate(state.panes.len());
    }

    if pane_ui_models.len() < state.panes.len() {
        pane_ui_models.resize_with(state.panes.len(), PaneUiModels::default);
    } else if pane_ui_models.len() > state.panes.len() {
        pane_ui_models.truncate(state.panes.len());
    }

    let build_pane_tabs = |pane: PaneId| {
        state
            .tabs_for_pane(pane)
            .iter()
            .filter_map(|&tab_id| {
                if state.is_home_tab(tab_id) {
                    Some(TabData {
                        id: tab_id,
                        title: SharedString::from("Home"),
                        path: SharedString::new(),
                        is_modified: false,
                        is_active: Some(tab_id) == state.active_tab_id_for_pane(pane),
                        is_focused: pane == state.focused_pane
                            && Some(tab_id) == state.active_tab_id_for_pane(pane),
                        is_home: true,
                    })
                } else {
                    state.get_file(tab_id).map(|file| TabData {
                        id: tab_id,
                        title: SharedString::from(file.filename.clone()),
                        path: SharedString::from(file.path.display().to_string()),
                        is_modified: false,
                        is_active: Some(tab_id) == state.active_tab_id_for_pane(pane),
                        is_focused: pane == state.focused_pane
                            && Some(tab_id) == state.active_tab_id_for_pane(pane),
                        is_home: false,
                    })
                }
            })
            .collect::<Vec<_>>()
    };

    let pane_models: Vec<PaneViewData> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| {
            let pane = PaneId(pane_index);
            let tabs = build_pane_tabs(pane);
            let (roi_rect, measurements, candidate_measurement) = pane_overlay_data(state, pane);
            let plugin_overlay_points =
                crate::plugin_host::viewport_overlay_points_for_pane(state, pane);
            let mut viewport_info = hidden_viewport_info();
            let mut minimap_rect = full_minimap_rect();
            let mut zoom_slider_position = 0.5;

            if let Some(file_id) = state.active_file_id_for_pane(pane)
                && let Some(file) = state.get_file(file_id)
                && let Some(viewport_state) = pane_viewport_state(file, pane)
            {
                let vp = &viewport_state.viewport;
                viewport_info = ViewportInfo {
                    center_x: vp.center.x as f32,
                    center_y: vp.center.y as f32,
                    zoom: vp.zoom as f32,
                    image_width: vp.image_width as f32,
                    image_height: vp.image_height as f32,
                    level: file
                        .wsi
                        .best_level_for_downsample(vp.effective_downsample())
                        as i32,
                };
                let rect = vp.minimap_rect();
                minimap_rect = MinimapRect {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                };
                zoom_slider_position = zoom_to_slider_value(vp.zoom);
            }

            let metadata_items = build_metadata_items(state, pane, &viewport_info);

            // Build HUD settings from per-pane state
            let hud = if let Some(file_id) = state.active_file_id_for_pane(pane)
                && let Some(file) = state.get_file(file_id)
            {
                let props = file.wsi.properties();
                let mpp_x = props.mpp_x.unwrap_or(0.0) as f32;
                let mpp_y = props.mpp_y.unwrap_or(0.0) as f32;
                file.pane_state(pane)
                    .map(|ps| ui_hud_settings(&ps.hud, mpp_x, mpp_y))
                    .unwrap_or_default()
            } else {
                SlintHudSettings::default()
            };

            let pane_ui = &pane_ui_models[pane_index];
            if !model_matches(&pane_ui.tabs, &tabs) {
                pane_ui.tabs.set_vec(tabs.clone());
            }
            if !model_matches(&pane_ui.measurements, &measurements) {
                pane_ui.measurements.set_vec(measurements.clone());
            }

            let cached = pane_render_cache
                .get(pane_index)
                .cloned()
                .unwrap_or_default();
            PaneViewData {
                id: pane.as_index(),
                tabs: pane_ui.tabs.clone().into(),
                content: cached.content.unwrap_or_default(),
                viewport_info,
                minimap_thumbnail: cached.minimap_thumbnail.unwrap_or_default(),
                minimap_rect,
                metadata_items: metadata_items.as_slice().into(),
                is_home_tab: state.is_home_tab_active_in_pane(pane),
                zoom_slider_position,
                roi_rect,
                measurements: pane_ui.measurements.clone().into(),
                plugin_overlay_points: plugin_overlay_points.as_slice().into(),
                point_tool_hover_active: state.current_tool == crate::state::Tool::PointAnnotation
                    && pane == state.focused_pane
                    && state.hovered_plugin_point.is_some(),
                candidate_measurement,
                is_loading: ui.get_is_loading() && pane == state.focused_pane,
                hud,
            }
        })
        .collect();

    // Update the view model
    let mut row_count = pane_view_model.row_count();
    while row_count > pane_models.len() {
        row_count -= 1;
        pane_view_model.remove(row_count);
    }
    for (index, pane_data) in pane_models.into_iter().enumerate() {
        if index < row_count {
            let next_pane_data = pane_view_model
                .row_data(index)
                .map(|existing| {
                    let mut merged = pane_data.clone();
                    if merged.content == Image::default() && existing.content != Image::default() {
                        merged.content = existing.content.clone();
                    }
                    if merged.minimap_thumbnail == Image::default()
                        && existing.minimap_thumbnail != Image::default()
                    {
                        merged.minimap_thumbnail = existing.minimap_thumbnail.clone();
                    }
                    let should_update = pane_view_data_changed(&existing, &merged);
                    (merged, should_update)
                })
                .unwrap_or((pane_data, true));
            let (next_pane_data, should_update) = next_pane_data;
            if should_update {
                pane_view_model.set_row_data(index, next_pane_data);
            }
        } else {
            pane_view_model.push(pane_data);
        }
    }
    ui.set_panes(pane_view_model.clone().into());
    ui.set_split_enabled(state.panes.len() > 1);
    ui.set_focused_pane(state.focused_pane.as_index());
    ui.set_show_minimap(state.show_minimap);
    ui.set_show_metadata(state.show_metadata);
    ui.set_show_scale_bar(state.show_scale_bar);
    ui.set_show_plugin_sidebar(state.has_active_sidebar());
    ui.set_plugin_sidebar_width(state.plugin_sidebar_width_px());
}

/// Update the recent files list in the UI
pub fn update_recent_files(ui: &crate::AppWindow, state: &AppState) {
    let recent: Vec<RecentFileData> = state
        .recent_files
        .iter()
        .take(5) // Show at most 5 in the UI
        .map(|f| RecentFileData {
            path: SharedString::from(f.path.display().to_string()),
            name: SharedString::from(f.name.clone()),
        })
        .collect();

    let model = Rc::new(VecModel::from(recent));
    ui.set_recent_files(model.into());
}

/// Build context menu items for recent files
pub fn build_recent_menu_items(state: &AppState) -> Vec<ContextMenuItem> {
    if state.recent_files.is_empty() {
        return vec![ContextMenuItem {
            id: SharedString::from("recent-empty"),
            label: SharedString::from("No recent files"),
            icon: SharedString::new(),
            shortcut: SharedString::new(),
            enabled: false,
            separator_after: false,
        }];
    }

    state
        .recent_files
        .iter()
        .take(10)
        .map(|file| ContextMenuItem {
            id: SharedString::from(format!("recent-file:{}", file.path.display())),
            label: SharedString::from(file.name.clone()),
            icon: SharedString::new(),
            shortcut: SharedString::new(),
            enabled: true,
            separator_after: false,
        })
        .collect()
}

/// Update the render backend UI state
pub fn update_render_backend(ui: &crate::AppWindow, state: &AppState) {
    ui.set_render_mode(ui_render_mode(state.render_backend));
    ui.set_gpu_rendering_available(state.gpu_backend_available);
}

/// Convert FilteringMode to Slint FilteringMode
pub fn ui_filtering_mode(mode: FilteringMode) -> SlintFilteringMode {
    match mode {
        FilteringMode::Bilinear => SlintFilteringMode::Bilinear,
        FilteringMode::Trilinear => SlintFilteringMode::Trilinear,
        FilteringMode::Lanczos3 => SlintFilteringMode::Lanczos3,
    }
}

/// Convert MeasurementUnit to Slint MeasurementUnit
fn ui_measurement_unit(unit: MeasurementUnit) -> SlintMeasurementUnit {
    match unit {
        MeasurementUnit::Um => SlintMeasurementUnit::Um,
        MeasurementUnit::Mm => SlintMeasurementUnit::Mm,
        MeasurementUnit::Inches => SlintMeasurementUnit::Inches,
    }
}

/// Convert StainNormalization to Slint StainNormalization
fn ui_stain_normalization(sn: StainNormalization) -> SlintStainNormalization {
    match sn {
        StainNormalization::None => SlintStainNormalization::None,
        StainNormalization::Macenko => SlintStainNormalization::Macenko,
        StainNormalization::Vahadane => SlintStainNormalization::Vahadane,
    }
}

/// Convert HudSettings to Slint HudSettings
fn ui_hud_settings(hud: &HudSettings, mpp_x: f32, mpp_y: f32) -> SlintHudSettings {
    SlintHudSettings {
        show_hud_toolbar: hud.show_hud_toolbar,
        hud_dropdown_open: hud.hud_dropdown_open,
        sharpness: hud.sharpness,
        gamma: hud.gamma,
        brightness: hud.brightness,
        contrast: hud.contrast,
        measurement_unit: ui_measurement_unit(hud.measurement_unit),
        stain_normalization: ui_stain_normalization(hud.stain_normalization),
        mpp_x,
        mpp_y,
        deconv_hematoxylin_intensity: hud.deconv_hematoxylin_intensity,
        deconv_hematoxylin_visible: hud.deconv_hematoxylin_visible,
        deconv_eosin_intensity: hud.deconv_eosin_intensity,
        deconv_eosin_visible: hud.deconv_eosin_visible,
        deconv_isolated_channel: ui_isolated_channel(hud.deconv_isolated_channel),
    }
}

fn ui_isolated_channel(ic: crate::state::IsolatedChannel) -> SlintIsolatedChannel {
    match ic {
        crate::state::IsolatedChannel::None => SlintIsolatedChannel::None,
        crate::state::IsolatedChannel::Hematoxylin => SlintIsolatedChannel::Hematoxylin,
        crate::state::IsolatedChannel::Eosin => SlintIsolatedChannel::Eosin,
    }
}

/// Update the filtering mode UI state
pub fn update_filtering_mode(ui: &crate::AppWindow, state: &AppState) {
    ui.set_filtering_mode(ui_filtering_mode(state.filtering_mode));
}
