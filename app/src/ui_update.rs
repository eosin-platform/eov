//! UI state update functions
//!
//! This module contains functions for updating UI elements like tabs,
//! recent files, and render backend settings.

use crate::pane_ui::{series_items_model, series_items_revision, set_series_items_revision};
use crate::state::{AppState, HudSettings, PaneId};
use crate::zoom_display::zoom_display_info;
use crate::{
    BottomPanelKind as SlintBottomPanelKind, ContextMenuItem, FilteringMode as SlintFilteringMode,
    HudSettings as SlintHudSettings, IsolatedChannel as SlintIsolatedChannel,
    MeasurementUnit as SlintMeasurementUnit, MetadataItem, MinimapRect, PaneRenderCacheEntry,
    PaneUiModels, PaneViewData, RecentFileData, RenderMode, SeriesItemData,
    StainNormalization as SlintStainNormalization, TabData, ViewportInfo,
};
use eov_common::viewport::{MAX_ZOOM, MIN_ZOOM};
use eov_common::{
    FilteringMode, MeasurementUnit, RenderBackend, StainNormalization, format_decimal,
    format_file_size, format_u64,
};
use slint::ComponentHandle;
use slint::{Image, Model, SharedString, VecModel};
use std::rc::Rc;

fn ui_bottom_panel_kind(kind: crate::state::BottomPanelKind) -> SlintBottomPanelKind {
    match kind {
        crate::state::BottomPanelKind::None => SlintBottomPanelKind::None,
        crate::state::BottomPanelKind::Series => SlintBottomPanelKind::Series,
    }
}

fn series_thumbnail_image(thumbnail: Option<&crate::state::SeriesThumbnail>) -> Image {
    thumbnail
        .and_then(|thumbnail| {
            crate::blitter::create_image_buffer(&thumbnail.rgba, thumbnail.width, thumbnail.height)
                .map(Image::from_rgba8)
        })
        .unwrap_or_default()
}

fn series_index_label(index: usize) -> String {
    let mut value = index;
    let mut label = Vec::new();

    loop {
        label.push((b'A' + (value % 26) as u8) as char);
        if value < 26 {
            break;
        }
        value = value / 26 - 1;
    }

    label.iter().rev().collect()
}

fn update_series_items_model(series: &crate::state::OpenedSeries) -> Rc<VecModel<SeriesItemData>> {
    let model = series_items_model();

    let mut row_count = model.row_count();
    while row_count > series.entries.len() {
        row_count -= 1;
        model.remove(row_count);
    }

    let mut file_index = 0usize;
    for (index, entry) in series.entries.iter().enumerate() {
        let entry_path = SharedString::from(entry.path.display().to_string());
        let existing = model.row_data(index);
        let thumbnail = existing
            .as_ref()
            .filter(|item| {
                item.path == entry_path && item.has_thumbnail == entry.thumbnail.is_some()
            })
            .map(|item| item.thumbnail.clone())
            .unwrap_or_else(|| series_thumbnail_image(entry.thumbnail.as_ref()));
        let series_label = if entry.is_directory {
            SharedString::default()
        } else {
            let label = series_index_label(file_index);
            file_index += 1;
            SharedString::from(label)
        };
        let next = SeriesItemData {
            path: entry_path,
            name: SharedString::from(entry.filename.clone()),
            series_label,
            objective_label: SharedString::from(entry.objective_label.clone().unwrap_or_default()),
            stain_label: SharedString::from(entry.stain_label.clone().unwrap_or_default()),
            is_directory: entry.is_directory,
            metadata_tooltip: SharedString::from(entry.metadata_tooltip.clone()),
            thumbnail,
            has_thumbnail: entry.thumbnail.is_some(),
            is_loading: entry.thumbnail_loading,
            is_selected: false,
        };

        if index < row_count {
            if existing.as_ref() != Some(&next) {
                model.set_row_data(index, next);
            }
        } else {
            model.push(next);
            row_count += 1;
        }
    }

    model
}

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
    let file_size = file
        .file_size
        .map(format_file_size)
        .unwrap_or_else(|| "Unknown".to_string());

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
        || existing.zoom_display_text != next.zoom_display_text
        || existing.zoom_input_text != next.zoom_input_text
        || existing.zoom_tooltip != next.zoom_tooltip
        || existing.minimap_thumbnail != next.minimap_thumbnail
        || existing.minimap_rect != next.minimap_rect
        || existing.metadata_items != next.metadata_items
        || existing.is_home_tab != next.is_home_tab
        || existing.zoom_slider_position != next.zoom_slider_position
        || existing.uid != next.uid
        || existing.overlay != next.overlay
        || existing.overlay_x != next.overlay_x
        || existing.overlay_y != next.overlay_y
        || existing.overlay_width != next.overlay_width
        || existing.overlay_height != next.overlay_height
        || existing.overlay_labels != next.overlay_labels
        || existing.cursor != next.cursor
        || existing.hud_buttons != next.hud_buttons
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
    // The caches follow the panes by identity, not position.
    let uids: Vec<u32> = state.panes.iter().map(|pane| pane.uid).collect();
    crate::pane_ui::sync_pane_caches(&uids, pane_render_cache, pane_ui_models);

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
            let mut viewport_info = hidden_viewport_info();
            let mut minimap_rect = full_minimap_rect();
            let mut zoom_slider_position = 0.5;
            let mut zoom_display = zoom_display_info(1.0, None);

            if let Some(file_id) = state.active_file_id_for_pane(pane)
                && let Some(file) = state.get_file(file_id)
            {
                zoom_display = zoom_display_info(1.0, crate::callbacks::native_magnification(file));
            }

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
                    // Frames are rendered at physical resolution (RND-3).
                    level: file.tile_manager.wsi().best_level_for_downsample(
                        vp.effective_downsample() / f64::from(ui.window().scale_factor()).max(1e-6),
                    ) as i32,
                };
                // Relative to what the thumbnail shows (the content bounds
                // of sparse slides).
                let rect = match file.thumbnail.as_ref() {
                    Some(thumbnail) => vp.minimap_rect_within(&thumbnail.region),
                    None => viewport_state.minimap_rect(),
                };
                minimap_rect = MinimapRect {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                };
                zoom_slider_position = zoom_to_slider_value(vp.zoom);
                zoom_display =
                    zoom_display_info(vp.zoom, crate::callbacks::native_magnification(file));
            }

            let metadata_items = build_metadata_items(state, pane, &viewport_info);

            // Build HUD settings from per-pane state
            let hud = if let Some(file_id) = state.active_file_id_for_pane(pane)
                && let Some(file) = state.get_file(file_id)
            {
                let calibration = crate::host_ui::calibration(file);
                let mpp_x = calibration.mpp_x.unwrap_or(0.0) as f32;
                let mpp_y = calibration.mpp_y.unwrap_or(0.0) as f32;
                let color = crate::color::hud_color_info(file, pane);
                file.pane_state(pane)
                    .map(|ps| {
                        let mut hud = ui_hud_settings(&ps.hud, mpp_x, mpp_y, &color);
                        // A horizontal bar measures with the horizontal pixel
                        // size (UI-6).
                        let spacing = crate::host_ui::calibration(file).spacing();
                        hud.plugin_badges = crate::host_ui::with_host(|host| {
                            host.shared.badges_for(i64::from(file.id)).join("\n")
                        })
                        .unwrap_or_default()
                        .into();
                        if let Some((um_x, _, _)) = spacing
                            && let Some((width, label)) = eov_common::overlay::scale_bar_for_view(
                                um_x / ps.viewport.viewport.zoom.max(1e-12),
                                SCALE_BAR_MAX_PX,
                                ps.hud.measurement_unit,
                            )
                        {
                            hud.scale_bar_width = width as f32;
                            hud.scale_bar_label = SharedString::from(label);
                        }
                        hud
                    })
                    .unwrap_or_default()
            } else {
                SlintHudSettings::default()
            };

            let pane_ui = &pane_ui_models[pane_index];
            if !model_matches(&pane_ui.tabs, &tabs) {
                pane_ui.tabs.set_vec(tabs.clone());
            }

            let cached = pane_render_cache
                .get(pane_index)
                .cloned()
                .unwrap_or_default();
            let uid = state.pane_uid(pane).unwrap_or_default();
            let active_file = state
                .active_file_id_for_pane(pane)
                .and_then(|file_id| state.get_file(file_id));
            let overlay = crate::host_ui::with_host(|host| {
                host.pane_overlay(
                    pane_index,
                    uid,
                    active_file.map(|file| i64::from(file.id)),
                    active_file
                        .and_then(|file| file.pane_state(pane))
                        .map(|ps| &ps.viewport.viewport),
                    active_file
                        .map(crate::host_ui::calibration)
                        .unwrap_or_default(),
                    active_file
                        .and_then(|file| file.pane_state(pane))
                        .map(|ps| ps.hud.measurement_unit)
                        .unwrap_or_default(),
                    f64::from(ui.window().scale_factor()),
                )
            });
            let overlay = overlay.unwrap_or_else(crate::host_ui::PaneOverlayFields::empty);
            PaneViewData {
                id: pane.as_index(),
                tabs: pane_ui.tabs.clone().into(),
                content: cached.content.unwrap_or_default(),
                viewport_info,
                zoom_display_text: SharedString::from(zoom_display.text),
                zoom_input_text: SharedString::from(zoom_display.input_text),
                zoom_tooltip: SharedString::from(zoom_display.tooltip),
                minimap_thumbnail: cached.minimap_thumbnail.unwrap_or_default(),
                minimap_rect,
                metadata_items: metadata_items.as_slice().into(),
                is_home_tab: state.is_home_tab_active_in_pane(pane),
                zoom_slider_position,
                uid: uid as i32,
                overlay: overlay.image,
                overlay_x: overlay.x,
                overlay_y: overlay.y,
                overlay_width: overlay.width,
                overlay_height: overlay.height,
                overlay_labels: overlay.labels,
                cursor: overlay.cursor,
                hud_buttons: overlay.hud_buttons,
                is_loading: crate::file_ops::opening_in_pane(uid, pane == state.focused_pane),
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
    ui.set_show_series_bar(
        state.bottom_panel_visible
            && state.bottom_panel_kind == crate::state::BottomPanelKind::Series,
    );
    ui.set_bottom_panel_kind(ui_bottom_panel_kind(state.bottom_panel_kind));
    ui.set_bottom_panel_height(state.bottom_panel_height_px());
    ui.set_viewport_lock_enabled(state.viewport_lock_enabled);
    let selected_series_path = state
        .selected_series_path()
        .map(|path| SharedString::from(path.display().to_string()))
        .unwrap_or_default();
    ui.set_series_selected_path(selected_series_path);
    if let Some(series) = state.opened_series.as_ref() {
        if series_items_revision() != state.series_content_revision {
            ui.set_series_items(update_series_items_model(series).into());
            set_series_items_revision(state.series_content_revision);
        } else {
            ui.set_series_items(series_items_model().into());
        }
        ui.set_series_can_go_back(state.series_can_go_back());
        ui.set_series_can_go_forward(state.series_can_go_forward());
        ui.set_series_can_go_up(state.series_can_go_up());
        ui.set_bottom_panel_status_left(SharedString::from(
            series.current_path.display().to_string(),
        ));
        ui.set_bottom_panel_status_right(SharedString::from(format!(
            "{} items",
            series.entries.len()
        )));
    } else {
        let model = series_items_model();
        if model.row_count() > 0 {
            model.set_vec(Vec::new());
        }
        ui.set_series_items(model.into());
        set_series_items_revision(0);
        ui.set_series_selected_path(SharedString::new());
        ui.set_series_can_go_back(false);
        ui.set_series_can_go_forward(false);
        ui.set_series_can_go_up(false);
        ui.set_bottom_panel_status_left(SharedString::new());
        ui.set_bottom_panel_status_right(SharedString::new());
    }
    let live: Vec<u32> = state.panes.iter().map(|p| p.uid).collect();
    crate::host_ui::with_host(|host| host.retain_panes(&live));
}

fn pane_viewport_state(
    file: &crate::state::OpenFile,
    pane: PaneId,
) -> Option<&eov_common::ViewportState> {
    file.pane_state(pane).map(|pane_state| &pane_state.viewport)
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
    let mut items = vec![ContextMenuItem {
        id: SharedString::from("open-folder-dialog"),
        label: SharedString::from("Open Folder..."),
        icon: SharedString::new(),
        shortcut: SharedString::new(),
        enabled: true,
        separator_after: !state.recent_files.is_empty(),
    }];

    if state.recent_files.is_empty() {
        items.push(ContextMenuItem {
            id: SharedString::from("recent-empty"),
            label: SharedString::from("No recent files"),
            icon: SharedString::new(),
            shortcut: SharedString::new(),
            enabled: false,
            separator_after: false,
        });
        return items;
    }

    // Entries are referred to by position, so any path (including ones
    // that are not UTF-8) opens exactly.
    items.extend(
        state
            .recent_files
            .iter()
            .enumerate()
            .take(10)
            .map(|(index, file)| ContextMenuItem {
                id: SharedString::from(format!("recent-file:{index}")),
                label: SharedString::from(file.name.clone()),
                icon: SharedString::new(),
                shortcut: SharedString::new(),
                enabled: true,
                separator_after: false,
            }),
    );
    if let Some(last) = items.last_mut() {
        last.separator_after = true;
    }
    items.push(ContextMenuItem {
        id: SharedString::from("recent-clear"),
        label: SharedString::from("Clear Recent Files"),
        icon: SharedString::new(),
        shortcut: SharedString::new(),
        enabled: true,
        separator_after: false,
    });
    items
}

pub fn build_recent_folder_menu_items(state: &AppState) -> Vec<ContextMenuItem> {
    let recent_folders: Vec<_> = state
        .recent_files
        .iter()
        .enumerate()
        .filter(|(_, file)| file.is_dir)
        .take(10)
        .collect();

    let mut items = vec![ContextMenuItem {
        id: SharedString::from("open-folder-dialog"),
        label: SharedString::from("Open Folder..."),
        icon: SharedString::new(),
        shortcut: SharedString::new(),
        enabled: true,
        separator_after: !recent_folders.is_empty(),
    }];

    if recent_folders.is_empty() {
        items.push(ContextMenuItem {
            id: SharedString::from("recent-empty"),
            label: SharedString::from("No opened folders"),
            icon: SharedString::new(),
            shortcut: SharedString::new(),
            enabled: false,
            separator_after: false,
        });
        return items;
    }

    items.extend(
        recent_folders
            .into_iter()
            .map(|(index, file)| ContextMenuItem {
                id: SharedString::from(format!("recent-file:{index}")),
                label: SharedString::from(file.name.clone()),
                icon: SharedString::new(),
                shortcut: SharedString::new(),
                enabled: true,
                separator_after: false,
            }),
    );

    items
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

/// Convert HudSettings (and what the colour pipeline makes of them) to
/// Slint HudSettings
fn ui_hud_settings(
    hud: &HudSettings,
    mpp_x: f32,
    mpp_y: f32,
    color: &crate::color::HudColorInfo,
) -> SlintHudSettings {
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
        modified: color.modified,
        color_badge: SharedString::from(color.badge.as_str()),
        color_warning: SharedString::from(color.warnings.join("\n")),
        scale_bar_width: 0.0,
        scale_bar_label: SharedString::new(),
        plugin_badges: SharedString::new(),
    }
}

/// Longest on-screen scale bar, in logical pixels.
const SCALE_BAR_MAX_PX: f64 = 140.0;

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

#[cfg(test)]
mod tests {
    use super::{series_index_label, ui_hud_settings};

    #[test]
    fn hud_settings_carry_the_colour_state() {
        let color = crate::color::HudColorInfo {
            modified: true,
            badge: "Colour-normalised (Macenko) · Sharpened".into(),
            warnings: vec!["first".into(), "second".into()],
        };
        let hud = ui_hud_settings(&crate::state::HudSettings::default(), 0.25, 0.5, &color);
        assert!(hud.modified);
        assert_eq!(hud.color_badge, "Colour-normalised (Macenko) · Sharpened");
        assert_eq!(hud.color_warning, "first\nsecond");
        assert_eq!((hud.mpp_x, hud.mpp_y), (0.25, 0.5));

        let plain = ui_hud_settings(
            &crate::state::HudSettings::default(),
            0.0,
            0.0,
            &crate::color::HudColorInfo::default(),
        );
        assert!(!plain.modified && plain.color_badge.is_empty() && plain.color_warning.is_empty());
    }

    #[test]
    fn series_index_label_uses_excel_style_letters() {
        assert_eq!(series_index_label(0), "A");
        assert_eq!(series_index_label(25), "Z");
        assert_eq!(series_index_label(26), "AA");
        assert_eq!(series_index_label(51), "AZ");
        assert_eq!(series_index_label(52), "BA");
        assert_eq!(series_index_label(701), "ZZ");
        assert_eq!(series_index_label(702), "AAA");
    }
}
