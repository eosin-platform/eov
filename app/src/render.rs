//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.

use crate::AppWindow;
use crate::blitter;
use crate::gpu::{SurfaceSlot, TileDraw};
use crate::state::{AppState, OpenFile, PaneId, RenderBackend, TileRequestSignature};
use crate::tile_loader::calculate_wanted_tiles;
use crate::tools;
use common::{TileCache, TileCoord, TileManager, Viewport, WsiFile};
use parking_lot::RwLock;
use slint::{ComponentHandle, Image};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

/// Texture filtering mode for rendering
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Nearest-neighbor sampling (fastest, pixelated)
    #[default]
    Nearest,
    /// Bilinear filtering within a single mip level (smooth but may show level transitions)
    Bilinear,
    /// Trilinear filtering: bilinear sampling with mip-level blending (smoothest transitions)
    Trilinear,
}

/// Render quality settings
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct RenderQuality {
    /// Texture filtering mode
    pub filter_mode: FilterMode,
    /// Anti-aliasing for text overlay
    pub antialias: bool,
    /// Show tile boundaries for debugging
    pub show_tile_boundaries: bool,
    /// Show debug info overlay
    pub show_debug_info: bool,
}

impl Default for RenderQuality {
    fn default() -> Self {
        Self {
            filter_mode: FilterMode::Trilinear, // Use trilinear for best quality
            antialias: true,
            show_tile_boundaries: false,
            show_debug_info: false,
        }
    }
}

/// Result of trilinear level calculation
#[derive(Debug, Clone, Copy)]
pub struct TrilinearLevels {
    /// The higher resolution (lower index) level
    pub level_fine: u32,
    /// The lower resolution (higher index) level
    pub level_coarse: u32,
    /// Blend factor: 0.0 = use level_fine, 1.0 = use level_coarse
    pub blend: f64,
}

/// Calculate the two mip levels to blend for trilinear filtering
pub fn calculate_trilinear_levels(wsi: &WsiFile, target_downsample: f64) -> TrilinearLevels {
    let level_count = wsi.level_count();

    if level_count == 0 {
        return TrilinearLevels {
            level_fine: 0,
            level_coarse: 0,
            blend: 0.0,
        };
    }

    if level_count == 1 {
        return TrilinearLevels {
            level_fine: 0,
            level_coarse: 0,
            blend: 0.0,
        };
    }

    // Find the best level (where pixel density is closest to 1:1)
    let best_level = wsi.best_level_for_downsample(target_downsample);

    let best_info = match wsi.level(best_level) {
        Some(info) => info,
        None => {
            return TrilinearLevels {
                level_fine: 0,
                level_coarse: 0,
                blend: 0.0,
            };
        }
    };

    // Determine if we should blend with the next finer or coarser level
    // If target_downsample > best_level's downsample, we're between best and next coarser
    // If target_downsample < best_level's downsample, we're between best and next finer
    let (level_fine, level_coarse) = if target_downsample >= best_info.downsample {
        // Blend between best (fine) and next coarser level
        if best_level + 1 < level_count {
            (best_level, best_level + 1)
        } else {
            // At coarsest level, no blending
            return TrilinearLevels {
                level_fine: best_level,
                level_coarse: best_level,
                blend: 0.0,
            };
        }
    } else {
        // Blend between previous finer level and best (coarse)
        if best_level > 0 {
            (best_level - 1, best_level)
        } else {
            // At finest level, no blending
            return TrilinearLevels {
                level_fine: 0,
                level_coarse: 0,
                blend: 0.0,
            };
        }
    };

    // Calculate blend factor using log space for perceptually linear transitions
    let fine_info = wsi.level(level_fine);
    let coarse_info = wsi.level(level_coarse);

    let (fine_ds, coarse_ds) = match (fine_info, coarse_info) {
        (Some(f), Some(c)) => (f.downsample, c.downsample),
        _ => {
            return TrilinearLevels {
                level_fine,
                level_coarse,
                blend: 0.0,
            };
        }
    };

    // Log-space interpolation for smooth transitions
    let log_target = target_downsample.ln();
    let log_fine = fine_ds.ln();
    let log_coarse = coarse_ds.ln();

    let blend = if (log_coarse - log_fine).abs() < 0.001 {
        0.0
    } else {
        ((log_target - log_fine) / (log_coarse - log_fine)).clamp(0.0, 1.0)
    };

    TrilinearLevels {
        level_fine,
        level_coarse,
        blend,
    }
}

/// Render statistics for performance monitoring
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// Number of tiles rendered this frame
    pub tiles_rendered: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Frame render time in microseconds
    pub render_time_us: u64,
}

/// Calculate the optimal level for a given zoom
#[allow(dead_code)]
pub fn optimal_level(wsi: &WsiFile, zoom: f64) -> u32 {
    // Target: find level where pixel density is close to 1:1
    let target_downsample = 1.0 / zoom;
    wsi.best_level_for_downsample(target_downsample)
}

/// Calculate visible tile range for the viewport
#[allow(dead_code)]
pub fn visible_tile_range(
    viewport: &Viewport,
    level: u32,
    wsi: &WsiFile,
    tile_size: u32,
) -> Option<TileRange> {
    let level_info = wsi.level(level)?;
    let bounds = viewport.bounds();

    // Convert viewport bounds to level coordinates
    let level_left = bounds.left / level_info.downsample;
    let level_top = bounds.top / level_info.downsample;
    let level_right = bounds.right / level_info.downsample;
    let level_bottom = bounds.bottom / level_info.downsample;

    // Calculate tile indices
    let ts = tile_size as f64;
    let start_x = ((level_left / ts).floor() as i64 - 1).max(0) as u64;
    let start_y = ((level_top / ts).floor() as i64 - 1).max(0) as u64;
    let end_x = ((level_right / ts).ceil() as u64 + 1).min(level_info.tiles_x(tile_size));
    let end_y = ((level_bottom / ts).ceil() as u64 + 1).min(level_info.tiles_y(tile_size));

    Some(TileRange {
        level,
        start_x,
        start_y,
        end_x,
        end_y,
    })
}

/// Range of tiles to render
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TileRange {
    pub level: u32,
    pub start_x: u64,
    pub start_y: u64,
    pub end_x: u64,
    pub end_y: u64,
}

#[allow(dead_code)]
impl TileRange {
    /// Iterate over all tile coordinates in this range
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        let tile_size = 256;
        (self.start_y..self.end_y).flat_map(move |y| {
            (self.start_x..self.end_x).map(move |x| TileCoord::new(self.level, x, y, tile_size))
        })
    }

    /// Get the total number of tiles in this range
    pub fn tile_count(&self) -> usize {
        ((self.end_x - self.start_x) * (self.end_y - self.start_y)) as usize
    }
}

/// Bilinear interpolation for pixel sampling
#[allow(dead_code)]
pub fn bilinear_sample(data: &[u8], width: u32, height: u32, x: f64, y: f64) -> [u8; 4] {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get_pixel = |px: u32, py: u32| -> [f64; 4] {
        let idx = ((py * width + px) * 4) as usize;
        if idx + 3 < data.len() {
            [
                data[idx] as f64,
                data[idx + 1] as f64,
                data[idx + 2] as f64,
                data[idx + 3] as f64,
            ]
        } else {
            [0.0, 0.0, 0.0, 255.0]
        }
    };

    let p00 = get_pixel(x0, y0);
    let p10 = get_pixel(x1, y0);
    let p01 = get_pixel(x0, y1);
    let p11 = get_pixel(x1, y1);

    let mut result = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] * (1.0 - fx) + p10[i] * fx;
        let bottom = p01[i] * (1.0 - fx) + p11[i] * fx;
        result[i] = (top * (1.0 - fy) + bottom * fy).clamp(0.0, 255.0) as u8;
    }

    result
}

type CoarseBlendData = (Arc<common::TileData>, [f32; 2], [f32; 2], f32);

#[derive(Default)]
struct PaneRenderOutcome {
    image: Option<Image>,
    keep_running: bool,
    rendered: bool,
}

fn tile_request_signature(
    tile_manager: &TileManager,
    viewport: &Viewport,
    level: u32,
    margin_tiles: i32,
) -> Option<TileRequestSignature> {
    let level_info = tile_manager.wsi().level(level)?;
    let bounds = viewport.bounds();
    let tile_size = tile_manager.tile_size_for_level(level);
    let tile_size_f64 = tile_size as f64;
    let level_left = bounds.left / level_info.downsample;
    let level_top = bounds.top / level_info.downsample;
    let level_right = bounds.right / level_info.downsample;
    let level_bottom = bounds.bottom / level_info.downsample;
    let margin_tiles_i64 = margin_tiles.max(0) as i64;

    Some(TileRequestSignature {
        level,
        margin_tiles,
        start_x: ((level_left / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        start_y: ((level_top / tile_size_f64).floor() as i64 - margin_tiles_i64).max(0) as u64,
        end_x: ((level_right / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_x(tile_size)),
        end_y: ((level_bottom / tile_size_f64).ceil() as u64 + margin_tiles_i64 as u64)
            .min(level_info.tiles_y(tile_size)),
        tile_size,
    })
}

pub(crate) fn thumbnail_image_for_file(file: &OpenFile) -> Option<Image> {
    let thumb_data = file.thumbnail.as_ref()?;

    let level = file.wsi.level_count().saturating_sub(1);
    let level_info = file.wsi.level(level)?;
    let aspect = level_info.width as f64 / level_info.height as f64;
    let (width, height) = if aspect > 1.0 {
        (150u32, (150.0 / aspect) as u32)
    } else {
        ((150.0 * aspect) as u32, 150u32)
    };

    blitter::create_image_buffer(thumb_data, width.max(1), height.max(1)).map(Image::from_rgba8)
}

pub(crate) fn update_and_render(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) -> bool {
    let mut state = state.write();
    let render_backend = state.render_backend;
    let pane_count = state.panes.len();
    let active_file_ids: Vec<Option<i32>> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| state.active_file_id_for_pane(PaneId(pane_index)))
        .collect();
    if active_file_ids.iter().all(Option::is_none) {
        for pane_index in 0..pane_count {
            crate::clear_cached_pane(PaneId(pane_index));
            state.set_last_rendered_file_id(PaneId(pane_index), None);
        }
        crate::update_tabs(ui, &state);
        return false;
    }

    let render_requested = std::mem::take(&mut state.needs_render);
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;

    let content_width = (ui.get_content_area_width() as f64).max(100.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(100.0);
    let pane_gap = 6.0;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(100.0);

    let mut wanted_tiles_by_file: HashMap<i32, HashSet<common::TileCoord>> = HashMap::new();
    for (pane_index, file_id) in active_file_ids.iter().copied().enumerate() {
        let Some(file_id) = file_id else {
            continue;
        };
        let pane = PaneId(pane_index);
        let Some(file_index) = state.open_files.iter().position(|f| f.id == file_id) else {
            continue;
        };

        let Some((viewport_state, last_request)) = ({
            let file = &mut state.open_files[file_index];
            match file.pane_state_mut(pane) {
                Some(pane_state) => {
                    pane_state.viewport.set_size(pane_width, content_height);
                    Some((pane_state.viewport.clone(), pane_state.last_request))
                }
                None => None,
            }
        }) else {
            continue;
        };

        let vp = &viewport_state.viewport;
        let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
        let request = {
            let file = &state.open_files[file_index];
            let trilinear = calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
            let level = trilinear.level_fine;
            tile_request_signature(&file.tile_manager, vp, level, margin_tiles).map(|signature| {
                let wanted = calculate_wanted_tiles(
                    &file.tile_manager,
                    level,
                    vp.bounds().left,
                    vp.bounds().top,
                    vp.bounds().right,
                    vp.bounds().bottom,
                    margin_tiles,
                );
                (signature, wanted)
            })
        };
        if let Some((signature, wanted)) = request {
            if last_request != Some(signature)
                && let Some(file) = state.open_files.get_mut(file_index)
                && let Some(pane_state) = file.pane_state_mut(pane)
            {
                pane_state.last_request = Some(signature);
            }
            wanted_tiles_by_file
                .entry(file_id)
                .or_default()
                .extend(wanted);
        }
    }

    for (file_id, wanted_tiles) in wanted_tiles_by_file {
        if let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) {
            file.tile_loader
                .set_wanted_tiles(wanted_tiles.into_iter().collect());
        }
    }

    let mut keep_running = render_requested;
    let mut rendered_frame = false;

    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let file_switched = state.last_rendered_file_id(pane) != file_id;
        keep_running |= file_switched;

        let Some(file_id) = file_id else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let minimap_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.minimap_thumbnail.as_ref())
                .is_none()
        });
        let content_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.content.as_ref())
                .is_none()
        });
        let force_render = file_switched || content_missing;

        if pane.0 == 1 {
            debug!(
                pane = pane.0,
                file_id,
                file_switched,
                force_render,
                content_missing,
                minimap_missing,
                "pane cache state before render"
            );
        }

        if file_switched
            && let Some(pane_state) = file.pane_state_mut(pane)
        {
            pane_state.frame_count = 0;
        }
        if file_switched || minimap_missing {
            crate::set_cached_pane_minimap(pane, thumbnail_image_for_file(file));
        }

        let outcome = render_pane_to_image(
            ui,
            file,
            pane,
            tile_cache,
            pane_width,
            content_height,
            force_render,
            render_backend,
        );

        if pane.0 == 1 {
            debug!(
                pane = pane.0,
                file_id,
                rendered = outcome.rendered,
                image = outcome.image.is_some(),
                keep_running = outcome.keep_running,
                "pane render outcome"
            );
        }

        keep_running |= outcome.keep_running;
        rendered_frame |= outcome.rendered;
        if let Some(image) = outcome.image {
            crate::set_cached_pane_content(pane, image);
            state.set_last_rendered_file_id(pane, Some(file_id));
        } else {
            state.set_last_rendered_file_id(pane, Some(file_id));
            keep_running |= content_missing;
        }
    }

    crate::update_tabs(ui, &state);
    keep_running |= tools::has_active_roi_overlay(&state);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

#[allow(clippy::too_many_arguments)]
fn render_pane_to_image(
    ui: &AppWindow,
    file: &mut OpenFile,
    pane: PaneId,
    tile_cache: &Arc<TileCache>,
    target_width: f64,
    target_height: f64,
    force_render: bool,
    render_backend: RenderBackend,
) -> PaneRenderOutcome {
    let (
        animating,
        viewport_state,
        frame_count,
        last_render_zoom,
        last_render_center_x,
        last_render_center_y,
        last_render_width,
        last_render_height,
        last_render_level,
        previous_tiles_loaded,
        last_seen_tile_epoch,
    ) = {
        let Some(pane_state) = file.pane_state_mut(pane) else {
            return PaneRenderOutcome::default();
        };

        let animating = pane_state.viewport.update();
        pane_state.viewport.set_size(target_width, target_height);
        (
            animating,
            pane_state.viewport.clone(),
            pane_state.frame_count,
            pane_state.last_render_zoom,
            pane_state.last_render_center_x,
            pane_state.last_render_center_y,
            pane_state.last_render_width,
            pane_state.last_render_height,
            pane_state.last_render_level,
            pane_state.tiles_loaded_since_render,
            pane_state.last_seen_tile_epoch,
        )
    };

    let vp = &viewport_state.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = frame_count == 0;
    let viewport_changed = (last_render_zoom - vp_zoom).abs() > 0.001
        || (last_render_center_x - vp_center_x).abs() > 1.0
        || (last_render_center_y - vp_center_y).abs() > 1.0
        || (last_render_width - vp_width).abs() > 1.0
        || (last_render_height - vp_height).abs() > 1.0;

    let trilinear = calculate_trilinear_levels(&file.wsi, vp.effective_downsample());
    let level = trilinear.level_fine;
    let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
    let level_changed = level != last_render_level;

    let bounds = vp.bounds();
    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        margin_tiles,
    );
    let visible_tiles: Vec<_> = visible_tiles.into_iter().take(500).collect();
    let cached_tiles: Vec<_> = visible_tiles
        .iter()
        .filter_map(|coord| tile_cache.get(coord).map(|data| (*coord, data)))
        .collect();
    let cached_count = cached_tiles.len() as u32;

    let cached_coarse_tiles: Vec<_> =
        if trilinear.level_fine != trilinear.level_coarse && trilinear.blend > 0.01 {
            file.tile_manager
                .visible_tiles_with_margin(
                    trilinear.level_coarse,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                    bounds.bottom,
                    margin_tiles,
                )
                .into_iter()
                .filter_map(|coord| tile_cache.get(&coord).map(|data| (coord, data)))
                .collect()
        } else {
            Vec::new()
        };

    let loaded_tile_epoch = file.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > last_seen_tile_epoch;
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            previous_tiles_loaded
        }
        || !cached_coarse_tiles.is_empty()
        || tile_epoch_advanced;
    let tiles_pending = file.tile_loader.pending_count() > 0;

    let keep_running = animating || new_tiles_loaded || tiles_pending;

    if pane.0 == 1 {
        debug!(
            pane = pane.0,
            zoom = vp_zoom,
            width = vp_width,
            height = vp_height,
            level,
            cached_count,
            coarse_tiles = cached_coarse_tiles.len(),
            previous_tiles_loaded,
            tile_epoch_advanced,
            tiles_pending,
            viewport_changed,
            level_changed,
            force_render,
            keep_running,
            "pane render decision inputs"
        );
    }

    if !force_render && !is_first_frame && !viewport_changed && !level_changed && !new_tiles_loaded
    {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.frame_count += 1;
        pane_state.last_render_time = std::time::Instant::now();
        pane_state.last_render_zoom = vp_zoom;
        pane_state.last_render_center_x = vp_center_x;
        pane_state.last_render_center_y = vp_center_y;
        pane_state.last_render_width = vp_width;
        pane_state.last_render_height = vp_height;
        pane_state.last_render_level = level;
        pane_state.tiles_loaded_since_render = cached_count;
        pane_state.last_seen_tile_epoch = loaded_tile_epoch;
    }

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if render_backend == RenderBackend::Gpu {
        let draws = collect_tile_draws(file, tile_cache, vp, trilinear);
        let slot = match pane {
            PaneId::PRIMARY => SurfaceSlot::PRIMARY,
            PaneId::SECONDARY => SurfaceSlot::SECONDARY,
            _ => SurfaceSlot(pane.0),
        };
        let image = crate::with_gpu_renderer(|renderer| {
            renderer
                .borrow_mut()
                .queue_frame(slot, render_width, render_height, draws)
        })
        .flatten();

        if image.is_some() {
            ui.window().request_redraw();
        } else if let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.frame_count = 0;
            pane_state.last_render_level = u32::MAX;
            pane_state.tiles_loaded_since_render = 0;
        }
        let rendered = image.is_some();

        return PaneRenderOutcome {
            image,
            keep_running: keep_running || !rendered,
            rendered,
        };
    }

    let level_info = match file.wsi.level(level) {
        Some(info) => info.clone(),
        None => {
            return PaneRenderOutcome {
                image: None,
                keep_running,
                rendered: false,
            };
        }
    };

    let level_count = file.wsi.level_count();
    let mut fallback_blits = Vec::new();
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level).cloned() else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for fb_coord in fallback_tiles.iter().take(100) {
            let Some(fallback_tile) = tile_cache.get(fb_coord) else {
                continue;
            };

            let fb_image_x =
                fb_coord.x as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_y =
                fb_coord.y as f64 * fb_coord.tile_size as f64 * fallback_level_info.downsample;
            let fb_image_x_end =
                fb_image_x + fallback_tile.width as f64 * fallback_level_info.downsample;
            let fb_image_y_end =
                fb_image_y + fallback_tile.height as f64 * fallback_level_info.downsample;

            let screen_x = ((fb_image_x - bounds.left) * vp.zoom).round() as i32;
            let screen_y = ((fb_image_y - bounds.top) * vp.zoom).round() as i32;
            let screen_x_end = ((fb_image_x_end - bounds.left) * vp.zoom).round() as i32;
            let screen_y_end = ((fb_image_y_end - bounds.top) * vp.zoom).round() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            fallback_blits.push((fallback_tile, screen_x, screen_y, screen_w, screen_h));
        }
    }

    let Some(pane_state) = file.pane_state_mut(pane) else {
        return PaneRenderOutcome::default();
    };
    let buffer_size = (render_width * render_height * 4) as usize;
    if pane_state.render_buffer.len() != buffer_size {
        pane_state.render_buffer.resize(buffer_size, 0);
    }
    blitter::fast_fill_rgba(&mut pane_state.render_buffer, 30, 30, 30, 255);
    let buffer = &mut pane_state.render_buffer[..];

    for (fallback_tile, screen_x, screen_y, screen_w, screen_h) in fallback_blits {
        blitter::blit_tile(
            buffer,
            render_width,
            render_height,
            &fallback_tile.data,
            fallback_tile.width,
            fallback_tile.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }

    for (coord, tile_data) in cached_tiles.iter() {
        let image_x = coord.x as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_y = coord.y as f64 * coord.tile_size as f64 * level_info.downsample;
        let image_x_end = image_x + tile_data.width as f64 * level_info.downsample;
        let image_y_end = image_y + tile_data.height as f64 * level_info.downsample;
        let screen_x = ((image_x - bounds.left) * vp.zoom).round() as i32;
        let screen_y = ((image_y - bounds.top) * vp.zoom).round() as i32;
        let screen_x_end = ((image_x_end - bounds.left) * vp.zoom).round() as i32;
        let screen_y_end = ((image_y_end - bounds.top) * vp.zoom).round() as i32;
        let screen_w = screen_x_end - screen_x;
        let screen_h = screen_y_end - screen_y;

        blitter::blit_tile(
            buffer,
            render_width,
            render_height,
            &tile_data.data,
            tile_data.width,
            tile_data.height,
            screen_x,
            screen_y,
            screen_w,
            screen_h,
        );
    }

    PaneRenderOutcome {
        image: blitter::create_image_buffer(buffer, render_width, render_height)
            .map(Image::from_rgba8),
        keep_running,
        rendered: true,
    }
}

fn collect_tile_draws(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: TrilinearLevels,
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    let bounds = vp.bounds();
    let level_count = file.wsi.level_count();

    for fallback_level in (0..level_count).rev() {
        if fallback_level <= trilinear.level_fine {
            continue;
        }

        let Some(fallback_level_info) = file.wsi.level(fallback_level) else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        for coord in fallback_tiles.iter().take(100) {
            let Some(tile_data) = tile_cache.get(coord) else {
                continue;
            };
            if let Some(draw) = tile_draw_from_tile(
                vp,
                bounds.left,
                bounds.top,
                fallback_level_info.downsample,
                *coord,
                tile_data,
                None,
            ) {
                draws.push(draw);
            }
        }
    }

    let Some(level_info) = file.wsi.level(trilinear.level_fine) else {
        return draws;
    };

    let visible_tiles = file.tile_manager.visible_tiles_with_margin(
        trilinear.level_fine,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        0,
    );

    for coord in visible_tiles.iter().take(500) {
        let Some(tile_data) = tile_cache.get(coord) else {
            continue;
        };
        let coarse_blend = coarse_blend_for_tile(
            file,
            tile_cache,
            trilinear,
            level_info.downsample,
            *coord,
            &tile_data,
        );
        if let Some(draw) = tile_draw_from_tile(
            vp,
            bounds.left,
            bounds.top,
            level_info.downsample,
            *coord,
            tile_data,
            coarse_blend,
        ) {
            draws.push(draw);
        }
    }

    draws
}

fn coarse_blend_for_tile(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine_coord: common::TileCoord,
    fine_tile: &Arc<common::TileData>,
) -> Option<CoarseBlendData> {
    const COARSE_BOUNDARY_EPSILON: f64 = 1e-3;

    if trilinear.level_fine == trilinear.level_coarse || trilinear.blend <= 0.01 {
        return None;
    }

    let coarse_info = file.wsi.level(trilinear.level_coarse)?;
    let image_x = fine_coord.x as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let image_y = fine_coord.y as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let fine_image_w = fine_tile.width as f64 * fine_downsample;
    let fine_image_h = fine_tile.height as f64 * fine_downsample;
    let coarse_tile_size = file
        .tile_manager
        .tile_size_for_level(trilinear.level_coarse) as f64;
    let image_x_end = image_x + fine_image_w;
    let image_y_end = image_y + fine_image_h;

    let coarse_tile_x = image_x / coarse_info.downsample;
    let coarse_tile_y = image_y / coarse_info.downsample;
    let coarse_tile_x_end =
        ((image_x_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_x);
    let coarse_tile_y_end =
        ((image_y_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_y);
    let coarse_start_tile_x = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_start_tile_y = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_x = (coarse_tile_x_end / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_y = (coarse_tile_y_end / coarse_tile_size).floor().max(0.0) as u64;

    if coarse_start_tile_x != coarse_end_tile_x || coarse_start_tile_y != coarse_end_tile_y {
        return None;
    }

    let coarse_coord = common::TileCoord::new(
        trilinear.level_coarse,
        coarse_start_tile_x,
        coarse_start_tile_y,
        coarse_tile_size as u32,
    );

    let coarse_tile = tile_cache.get(&coarse_coord)?;
    let coarse_origin_x = coarse_coord.x as f64 * coarse_coord.tile_size as f64;
    let coarse_origin_y = coarse_coord.y as f64 * coarse_coord.tile_size as f64;
    let coarse_src_x = (coarse_tile_x - coarse_origin_x).max(0.0);
    let coarse_src_y = (coarse_tile_y - coarse_origin_y).max(0.0);
    let coarse_src_w = (fine_image_w / coarse_info.downsample).max(0.0);
    let coarse_src_h = (fine_image_h / coarse_info.downsample).max(0.0);
    let coarse_src_x_end = coarse_src_x + coarse_src_w;
    let coarse_src_y_end = coarse_src_y + coarse_src_h;

    if coarse_src_x_end <= coarse_src_x
        || coarse_src_y_end <= coarse_src_y
        || coarse_src_x_end > coarse_tile.width as f64 + COARSE_BOUNDARY_EPSILON
        || coarse_src_y_end > coarse_tile.height as f64 + COARSE_BOUNDARY_EPSILON
    {
        return None;
    }

    let coarse_width = coarse_tile.width as f64;
    let coarse_height = coarse_tile.height as f64;

    Some((
        coarse_tile,
        [
            (coarse_src_x / coarse_width) as f32,
            (coarse_src_y / coarse_height) as f32,
        ],
        [
            (coarse_src_x_end / coarse_width) as f32,
            (coarse_src_y_end / coarse_height) as f32,
        ],
        trilinear.blend as f32,
    ))
}

fn tile_draw_from_tile(
    vp: &Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
    coord: common::TileCoord,
    tile_data: Arc<common::TileData>,
    coarse_blend: Option<CoarseBlendData>,
) -> Option<TileDraw> {
    let image_x = coord.x as f64 * coord.tile_size as f64 * downsample;
    let image_y = coord.y as f64 * coord.tile_size as f64 * downsample;
    let image_x_end = image_x + tile_data.width as f64 * downsample;
    let image_y_end = image_y + tile_data.height as f64 * downsample;

    let screen_x = ((image_x - bounds_left) * vp.zoom).round() as i32;
    let screen_y = ((image_y - bounds_top) * vp.zoom).round() as i32;
    let screen_x_end = ((image_x_end - bounds_left) * vp.zoom).round() as i32;
    let screen_y_end = ((image_y_end - bounds_top) * vp.zoom).round() as i32;
    let screen_w = screen_x_end - screen_x;
    let screen_h = screen_y_end - screen_y;

    if screen_w <= 0 || screen_h <= 0 {
        return None;
    }

    let (coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend) = coarse_blend
        .map(|(coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend)| {
            (Some(coarse_tile), coarse_uv_min, coarse_uv_max, mip_blend)
        })
        .unwrap_or((None, [0.0, 0.0], [1.0, 1.0], 0.0));

    Some(TileDraw {
        tile: tile_data,
        coarse_tile,
        screen_x,
        screen_y,
        screen_w,
        screen_h,
        coarse_uv_min,
        coarse_uv_max,
        mip_blend,
    })
}
