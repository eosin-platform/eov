//! File operations for opening and managing WSI files
//!
//! This module contains functions for opening files and generating thumbnails.

use crate::state::{AppState, PaneId};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use crate::ui_update::{update_recent_files, update_tabs};
use crate::{PaneRenderCacheEntry, PaneUiModels, PaneViewData, request_render_loop};
use common::{TileCache, TileManager, ViewportState, WsiFile};
use parking_lot::RwLock;
use slint::{ComponentHandle, SharedString, Timer, VecModel};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Open a file and add it to the application state
pub fn open_file(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    path: PathBuf,
    pane_render_cache: &mut Vec<PaneRenderCacheEntry>,
    pane_ui_models: &mut Vec<PaneUiModels>,
    pane_view_model: &Rc<VecModel<PaneViewData>>,
) {
    ui.set_is_loading(true);
    ui.set_status_text(SharedString::from(format!("Opening {}...", path.display())));

    match WsiFile::open(&path) {
        Ok(wsi) => {
            let id = {
                let mut state_guard = state.write();

                let target_pane = if state_guard.split_enabled {
                    state_guard.focused_pane
                } else {
                    PaneId::PRIMARY
                };

                // Replace the focused pane's active home tab with the opened file.
                if let Some(active_id) = state_guard.active_tab_id_for_pane(target_pane)
                    && state_guard.is_home_tab(active_id)
                {
                    state_guard.close_home_tab(active_id);
                }

                // Get viewport size from the focused pane (use reasonable defaults if not yet laid out)
                let pane_count = state_guard.panes.len().max(1) as f64;
                let pane_gap = 6.0;
                let ui_width = ((ui.get_content_area_width() as f64)
                    - pane_gap * (pane_count - 1.0))
                    / pane_count;
                let ui_height = ui.get_content_area_height() as f64 - 35.0;
                let viewport_width = if ui_width > 0.0 { ui_width } else { 1024.0 };
                let viewport_height = if ui_height > 0.0 { ui_height } else { 768.0 };

                let props = wsi.properties();
                let viewport = ViewportState::new(
                    viewport_width,
                    viewport_height,
                    props.width as f64,
                    props.height as f64,
                );

                // Give the tile manager its own OpenSlide handle so tile loading
                // never contends with the UI's metadata handle.
                let tile_manager_wsi = match wsi.reopen() {
                    Ok(tile_manager_wsi) => tile_manager_wsi,
                    Err(err) => {
                        error!("Failed to open dedicated tile-manager handle: {}", err);
                        ui.set_is_loading(false);
                        ui.set_status_text(SharedString::from(format!("Error: {}", err)));
                        return;
                    }
                };
                let tile_manager = Arc::new(TileManager::new(tile_manager_wsi));

                // Create background tile loader (tiles are loaded on-demand)
                let tile_loader =
                    TileLoader::new(Arc::clone(&tile_manager), Arc::clone(tile_cache));

                // Start loading tiles immediately using the initial viewport bounds
                // This ensures tiles begin loading before the first render
                let bounds = viewport.viewport.bounds();
                let best_level =
                    wsi.best_level_for_downsample(viewport.viewport.effective_downsample());
                let initial_tiles = calculate_wanted_tiles(
                    &tile_manager,
                    best_level,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                    bounds.bottom,
                    1,
                );
                tile_loader.set_wanted_tiles(initial_tiles);

                // Generate small thumbnail for minimap (lazy - only reads what's needed)
                let thumbnail = generate_thumbnail(&wsi, 150);

                state_guard.add_file(
                    path.clone(),
                    wsi,
                    tile_manager,
                    tile_loader,
                    viewport,
                    thumbnail,
                )
            };

            let level_count = {
                let state_guard = state.read();
                update_tabs(
                    ui,
                    &state_guard,
                    pane_render_cache,
                    pane_ui_models,
                    pane_view_model,
                );
                update_recent_files(ui, &state_guard);
                state_guard
                    .get_file(id)
                    .map(|f| f.wsi.level_count())
                    .unwrap_or(0)
            };

            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!(
                "Opened {} ({} levels)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                level_count
            )));

            info!("Successfully opened file with {} levels", level_count);

            request_render_loop(render_timer, &ui.as_weak(), state, tile_cache);
        }
        Err(e) => {
            error!("Failed to open file: {}", e);
            ui.set_is_loading(false);
            ui.set_status_text(SharedString::from(format!("Error: {}", e)));
        }
    }
}

/// Generate a thumbnail image from the lowest resolution level of the WSI
pub fn generate_thumbnail(wsi: &WsiFile, max_size: u32) -> Option<Vec<u8>> {
    // Use the lowest resolution level for thumbnail
    let level = wsi.level_count().saturating_sub(1);
    let level_info = wsi.level(level)?;

    // Calculate thumbnail dimensions maintaining aspect ratio
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (thumb_w, thumb_h) = if aspect > 1.0 {
        (max_size, (max_size as f64 / aspect).max(1.0) as u32)
    } else {
        ((max_size as f64 * aspect).max(1.0) as u32, max_size)
    };

    // If level is small enough, read it directly
    if level_info.width <= max_size as u64 * 2 && level_info.height <= max_size as u64 * 2 {
        match wsi.read_region(
            0,
            0,
            level,
            level_info.width as u32,
            level_info.height as u32,
        ) {
            Ok(data) => {
                if level_info.width <= max_size as u64 && level_info.height <= max_size as u64 {
                    return Some(data);
                }
                // Resize to thumbnail size
                if let Some(img) = image::RgbaImage::from_raw(
                    level_info.width as u32,
                    level_info.height as u32,
                    data,
                ) {
                    let resized = image::imageops::resize(
                        &img,
                        thumb_w,
                        thumb_h,
                        image::imageops::FilterType::Triangle,
                    );
                    return Some(resized.into_raw());
                }
            }
            Err(e) => {
                warn!("Failed to generate thumbnail: {}", e);
            }
        }
    }

    // Level is too large - read the full level anyway (it's still the smallest level)
    // and resize down to thumbnail size
    match wsi.read_region(
        0,
        0,
        level,
        level_info.width as u32,
        level_info.height as u32,
    ) {
        Ok(data) => {
            if let Some(img) =
                image::RgbaImage::from_raw(level_info.width as u32, level_info.height as u32, data)
            {
                let resized = image::imageops::resize(
                    &img,
                    thumb_w,
                    thumb_h,
                    image::imageops::FilterType::Triangle,
                );
                Some(resized.into_raw())
            } else {
                None
            }
        }
        Err(e) => {
            warn!("Failed to generate thumbnail: {}", e);
            None
        }
    }
}
