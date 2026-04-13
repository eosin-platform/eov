//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.

use crate::AppWindow;
use crate::blitter;
use crate::gpu::{QueuedFrame, SurfaceSlot, TileDraw};
use crate::state::{
    AppState, FilteringMode, OpenFile, PaneId, RenderBackend, TileRequestSignature,
};
use crate::tile_loader::calculate_wanted_tiles;
use crate::tools;
use common::{TileCache, TileManager, Viewport, WsiFile};
use parking_lot::RwLock;
use slint::{ComponentHandle, Image};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

/// Adaptive Lanczos-to-Trilinear blending weight based on zoom level.
///
/// At very low zoom, trilinear filtering produces better results than Lanczos.
/// This function returns a weight in \[0.0, 1.0\] controlling the mix:
///   zoom >= 0.12 → 1.0 (100% Lanczos)
///   0.07–0.12   → linear blend Lanczos → Trilinear
///   zoom < 0.07 → 0.0 (100% Trilinear)
fn lanczos_adaptive_weight(zoom: f64) -> f64 {
    const LANCZOS_FULL: f64 = 0.12;
    const TRILINEAR_FULL: f64 = 0.07;
    if zoom >= LANCZOS_FULL {
        1.0
    } else if zoom <= TRILINEAR_FULL {
        0.0
    } else {
        (zoom - TRILINEAR_FULL) / (LANCZOS_FULL - TRILINEAR_FULL)
    }
}

/// Apply gamma, brightness, and contrast adjustments to an RGBA pixel buffer.
fn apply_adjustments(buffer: &mut [u8], gamma: f32, brightness: f32, contrast: f32) {
    // Pre-compute a 256-entry lookup table for the combined transformation.
    // Pipeline: input → gamma → brightness → contrast → clamp
    let inv_gamma = if gamma > 0.001 { 1.0 / gamma } else { 1.0 };
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let normalized = i as f32 / 255.0;
        // Apply gamma
        let g = normalized.powf(inv_gamma);
        // Apply brightness (additive)
        let b = g + brightness;
        // Apply contrast (multiply around midpoint 0.5)
        let c = (b - 0.5) * contrast + 0.5;
        *entry = (c * 255.0).clamp(0.0, 255.0) as u8;
    }
    // Apply LUT to RGB channels (skip alpha every 4th byte)
    for chunk in buffer.chunks_exact_mut(4) {
        chunk[0] = lut[chunk[0] as usize];
        chunk[1] = lut[chunk[1] as usize];
        chunk[2] = lut[chunk[2] as usize];
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

type CoarseBlendData = (Arc<common::TileData>, [f32; 2], [f32; 2], f32);

#[derive(Default)]
struct PaneRenderOutcome {
    image: Option<Image>,
    keep_running: bool,
    rendered: bool,
}

struct RenderPaneRequest<'a> {
    ui: &'a AppWindow,
    tile_cache: &'a Arc<TileCache>,
    target_width: f64,
    target_height: f64,
    force_render: bool,
    render_backend: RenderBackend,
    filtering_mode: FilteringMode,
}

#[derive(Clone, Copy)]
struct TileProjection<'a> {
    viewport: &'a Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
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
    let filtering_mode = state.filtering_mode;
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

        if file_switched && let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.frame_count = 0;
        }
        if file_switched || minimap_missing {
            crate::set_cached_pane_minimap(pane, thumbnail_image_for_file(file));
        }

        let outcome = render_pane_to_image(
            file,
            pane,
            RenderPaneRequest {
                ui,
                tile_cache,
                target_width: pane_width,
                target_height: content_height,
                force_render,
                render_backend,
                filtering_mode,
            },
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

fn render_pane_to_image(
    file: &mut OpenFile,
    pane: PaneId,
    request: RenderPaneRequest<'_>,
) -> PaneRenderOutcome {
    let RenderPaneRequest {
        ui,
        tile_cache,
        target_width,
        target_height,
        force_render,
        render_backend,
        filtering_mode,
    } = request;

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
        hud_sharpness,
        hud_gamma,
        hud_brightness,
        hud_contrast,
        hud_stain_normalization,
        last_render_sharpness,
        last_render_gamma,
        last_render_brightness,
        last_render_contrast,
        last_render_stain_normalization,
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
            pane_state.hud.sharpness,
            pane_state.hud.gamma,
            pane_state.hud.brightness,
            pane_state.hud.contrast,
            pane_state.hud.stain_normalization,
            pane_state.last_render_sharpness,
            pane_state.last_render_gamma,
            pane_state.last_render_brightness,
            pane_state.last_render_contrast,
            pane_state.last_render_stain_normalization,
        )
    };

    let vp = &viewport_state.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = frame_count == 0;
    let viewport_changed = animating
        || (last_render_zoom - vp_zoom).abs() > 0.001
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
        .filter_map(|coord| tile_cache.peek(coord).map(|data| (*coord, data)))
        .collect();
    let cached_count = cached_tiles.len() as u32;

    // Only fetch coarse tiles for trilinear blending
    let lanczos_weight = if filtering_mode == FilteringMode::Lanczos3 {
        lanczos_adaptive_weight(vp_zoom)
    } else {
        1.0
    };
    // Trilinear blend needed for explicit Trilinear mode or adaptive Lanczos at low zoom
    let use_trilinear_blend = filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0);
    let cached_coarse_tiles: Vec<_> = if use_trilinear_blend
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
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
            .filter_map(|coord| tile_cache.peek(&coord).map(|data| (coord, data)))
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

    let adjustments_changed = (hud_sharpness - last_render_sharpness).abs() > 0.001
        || (hud_gamma - last_render_gamma).abs() > 0.001
        || (hud_brightness - last_render_brightness).abs() > 0.001
        || (hud_contrast - last_render_contrast).abs() > 0.001
        || hud_stain_normalization != last_render_stain_normalization;

    if !force_render
        && !is_first_frame
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
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
        pane_state.last_render_gamma = hud_gamma;
        pane_state.last_render_brightness = hud_brightness;
        pane_state.last_render_contrast = hud_contrast;
        pane_state.last_render_sharpness = hud_sharpness;
        pane_state.last_render_stain_normalization = hud_stain_normalization;
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
        // Adaptive Lanczos: at low zoom, switch to trilinear on GPU
        let gpu_filtering = if filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0 {
            FilteringMode::Trilinear
        } else {
            filtering_mode
        };
        let gpu_use_trilinear = gpu_filtering == FilteringMode::Trilinear;
        let gpu_trilinear = if gpu_use_trilinear {
            trilinear
        } else {
            TrilinearLevels {
                level_fine: trilinear.level_fine,
                level_coarse: trilinear.level_fine,
                blend: 0.0,
            }
        };
        let draws = collect_tile_draws(file, tile_cache, vp, gpu_trilinear, gpu_filtering);
        let stain_params = crate::stain::compute_gpu_stain_params(&draws, hud_stain_normalization);
        let slot = match pane {
            PaneId::PRIMARY => SurfaceSlot::PRIMARY,
            PaneId::SECONDARY => SurfaceSlot::SECONDARY,
            _ => SurfaceSlot(pane.0),
        };
        let image = crate::with_gpu_renderer(|renderer| {
            renderer.borrow_mut().queue_frame(
                slot,
                QueuedFrame {
                    width: render_width,
                    height: render_height,
                    draws,
                    gamma: hud_gamma,
                    brightness: hud_brightness,
                    contrast: hud_contrast,
                    sharpness: hud_sharpness,
                    stain_norm_enabled: stain_params.enabled,
                    inv_stain_r0: stain_params.inv_stain_r0,
                    inv_stain_r1: stain_params.inv_stain_r1,
                },
            )
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

    // Detect whether the viewport is actively moving (drag, inertia, zoom
    // animation, navigation). While moving on CPU we skip expensive
    // post-processing (Lanczos, sharpening, stain normalization) and use
    // fused trilinear instead, matching the browser-style fluid path.
    let is_moving = viewport_state.is_moving() || animating;

    let level_count = file.wsi.level_count();
    let mut fallback_blits: Vec<(Arc<common::TileData>, blitter::BlitRect)> = Vec::new();
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
            let Some(fallback_tile) = tile_cache.peek(fb_coord) else {
                continue;
            };

            let fb_origin_x = fb_coord.x as f64 * fb_coord.tile_size as f64;
            let fb_origin_y = fb_coord.y as f64 * fb_coord.tile_size as f64;
            let fb_image_x = fb_origin_x * fallback_level_info.downsample;
            let fb_image_y = fb_origin_y * fallback_level_info.downsample;
            let fb_image_x_end =
                (fb_origin_x + fallback_tile.width as f64) * fallback_level_info.downsample;
            let fb_image_y_end =
                (fb_origin_y + fallback_tile.height as f64) * fallback_level_info.downsample;

            let exact_sx = (fb_image_x - bounds.left) * vp.zoom;
            let exact_sy = (fb_image_y - bounds.top) * vp.zoom;
            let exact_sx_end = (fb_image_x_end - bounds.left) * vp.zoom;
            let exact_sy_end = (fb_image_y_end - bounds.top) * vp.zoom;
            let screen_x = exact_sx.floor() as i32;
            let screen_y = exact_sy.floor() as i32;
            let screen_x_end = exact_sx_end.floor() as i32;
            let screen_y_end = exact_sy_end.floor() as i32;
            let screen_w = screen_x_end - screen_x;
            let screen_h = screen_y_end - screen_y;

            if screen_w <= 0 || screen_h <= 0 {
                continue;
            }

            fallback_blits.push((
                fallback_tile,
                blitter::BlitRect {
                    x: screen_x,
                    y: screen_y,
                    width: screen_w,
                    height: screen_h,
                    exact_x: exact_sx,
                    exact_y: exact_sy,
                    exact_width: exact_sx_end - exact_sx,
                    exact_height: exact_sy_end - exact_sy,
                },
            ));
        }
    }

    // Build a coarse-tile lookup for fused trilinear blitting.
    // Maps (tile_x, tile_y) → Arc<TileData> for the coarse level so
    // per-fine-tile blits can find their coarse counterpart in O(1).
    let coarse_tile_map: HashMap<(u64, u64), Arc<common::TileData>> = cached_coarse_tiles
        .iter()
        .map(|(coord, data)| ((coord.x, coord.y), Arc::clone(data)))
        .collect();

    let coarse_info_opt = if use_trilinear_blend
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
        file.wsi.level(trilinear.level_coarse).cloned()
    } else {
        None
    };

    let coarse_tile_size = if coarse_info_opt.is_some() {
        file.tile_manager
            .tile_size_for_level(trilinear.level_coarse) as f64
    } else {
        0.0
    };

    let Some(_pane_state) = file.pane_state_mut(pane) else {
        return PaneRenderOutcome::default();
    };
    // Render directly into SharedPixelBuffer to avoid an intermediate copy.
    let mut pixel_buffer =
        slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(render_width, render_height);
    let buffer = pixel_buffer.make_mut_bytes();
    blitter::fast_fill_rgba(buffer, 30, 30, 30, 255);

    // While moving, always use bilinear (+ fused trilinear if applicable).
    // When settled, use the user's chosen filtering mode.
    let effective_filtering = if is_moving && render_backend == RenderBackend::Cpu {
        if use_trilinear_blend {
            FilteringMode::Trilinear
        } else {
            FilteringMode::Bilinear
        }
    } else {
        filtering_mode
    };

    let is_adaptive_lanczos = effective_filtering == FilteringMode::Lanczos3;
    let effective_lanczos_weight = if is_adaptive_lanczos {
        lanczos_adaptive_weight(vp_zoom)
    } else {
        1.0
    };
    let effective_trilinear = effective_filtering == FilteringMode::Trilinear
        || (is_adaptive_lanczos && effective_lanczos_weight < 1.0);

    // Helper: compute the BlitRect for a fine tile
    let fine_blit_rect = |coord: &common::TileCoord, tile_data: &common::TileData| -> blitter::BlitRect {
        let origin_x = coord.x as f64 * coord.tile_size as f64;
        let origin_y = coord.y as f64 * coord.tile_size as f64;
        let image_x = origin_x * level_info.downsample;
        let image_y = origin_y * level_info.downsample;
        let image_x_end = (origin_x + tile_data.width as f64) * level_info.downsample;
        let image_y_end = (origin_y + tile_data.height as f64) * level_info.downsample;
        let exact_sx = (image_x - bounds.left) * vp.zoom;
        let exact_sy = (image_y - bounds.top) * vp.zoom;
        let exact_sx_end = (image_x_end - bounds.left) * vp.zoom;
        let exact_sy_end = (image_y_end - bounds.top) * vp.zoom;
        let screen_x = exact_sx.floor() as i32;
        let screen_y = exact_sy.floor() as i32;
        let screen_x_end = exact_sx_end.floor() as i32;
        let screen_y_end = exact_sy_end.floor() as i32;
        blitter::BlitRect {
            x: screen_x,
            y: screen_y,
            width: screen_x_end - screen_x,
            height: screen_y_end - screen_y,
            exact_x: exact_sx,
            exact_y: exact_sy,
            exact_width: exact_sx_end - exact_sx,
            exact_height: exact_sy_end - exact_sy,
        }
    };

    // --- Blit fallback tiles (always bilinear, no trilinear) ---
    for (fallback_tile, rect) in &fallback_blits {
        blitter::blit_tile(
            buffer,
            render_width,
            render_height,
            blitter::TileSrc {
                data: &fallback_tile.data,
                width: fallback_tile.width,
                height: fallback_tile.height,
                border: fallback_tile.border,
            },
            *rect,
        );
    }

    // --- Blit fine tiles with optional fused trilinear ---
    let do_fused_trilinear = effective_trilinear
        && coarse_info_opt.is_some()
        && trilinear.blend > 0.01
        && !coarse_tile_map.is_empty();

    for (coord, tile_data) in cached_tiles.iter() {
        let rect = fine_blit_rect(coord, tile_data);

        // Try fused trilinear path: find the matching coarse tile and blit both in one pass
        if do_fused_trilinear {
            if let Some(ref coarse_info) = coarse_info_opt {
                let fine_ds = level_info.downsample;
                let coarse_ds = coarse_info.downsample;
                let image_x = coord.x as f64 * coord.tile_size as f64 * fine_ds;
                let image_y = coord.y as f64 * coord.tile_size as f64 * fine_ds;
                let fine_image_w = tile_data.width as f64 * fine_ds;
                let fine_image_h = tile_data.height as f64 * fine_ds;
                let coarse_tile_x = image_x / coarse_ds;
                let coarse_tile_y = image_y / coarse_ds;
                let ctx = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
                let cty = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;

                if let Some(coarse_tile) = coarse_tile_map.get(&(ctx, cty)) {
                    let coarse_origin_x = ctx as f64 * coarse_tile_size;
                    let coarse_origin_y = cty as f64 * coarse_tile_size;
                    let cu_min = ((coarse_tile_x - coarse_origin_x) / coarse_tile.width as f64)
                        .clamp(0.0, 1.0) as f32;
                    let cv_min = ((coarse_tile_y - coarse_origin_y) / coarse_tile.height as f64)
                        .clamp(0.0, 1.0) as f32;
                    let cu_max = (cu_min as f64
                        + fine_image_w / coarse_ds / coarse_tile.width as f64)
                        .clamp(0.0, 1.0) as f32;
                    let cv_max = (cv_min as f64
                        + fine_image_h / coarse_ds / coarse_tile.height as f64)
                        .clamp(0.0, 1.0) as f32;

                    blitter::blit_tile_trilinear(
                        buffer,
                        render_width,
                        render_height,
                        blitter::TileSrc {
                            data: &tile_data.data,
                            width: tile_data.width,
                            height: tile_data.height,
                            border: tile_data.border,
                        },
                        &blitter::CoarseSrc {
                            data: &coarse_tile.data,
                            width: coarse_tile.width,
                            height: coarse_tile.height,
                            border: coarse_tile.border,
                            uv_min: [cu_min, cv_min],
                            uv_max: [cu_max, cv_max],
                            blend: trilinear.blend as f32,
                        },
                        rect,
                    );
                    continue;
                }
            }
        }

        // Non-trilinear path or no coarse tile found: use the selected filter
        if is_adaptive_lanczos && effective_lanczos_weight >= 1.0 {
            blitter::blit_tile_lanczos3(
                buffer,
                render_width,
                render_height,
                blitter::TileSrc {
                    data: &tile_data.data,
                    width: tile_data.width,
                    height: tile_data.height,
                    border: tile_data.border,
                },
                rect,
            );
        } else {
            blitter::blit_tile(
                buffer,
                render_width,
                render_height,
                blitter::TileSrc {
                    data: &tile_data.data,
                    width: tile_data.width,
                    height: tile_data.height,
                    border: tile_data.border,
                },
                rect,
            );
        }
    }

    // Post-processing: skip expensive operations while moving for fluid panning.
    let skip_expensive_postprocess = is_moving && render_backend == RenderBackend::Cpu;

    // Post-processing: apply stain normalization if enabled (cached)
    if !skip_expensive_postprocess && hud_stain_normalization != crate::state::StainNormalization::None {
        // Use cached stain params if the tile epoch and method haven't changed.
        let need_recompute = file
            .pane_state(pane)
            .map(|ps| {
                ps.cached_stain_params.is_none()
                    || ps.stain_params_epoch != loaded_tile_epoch
                    || ps.stain_params_method != hud_stain_normalization
            })
            .unwrap_or(true);

        if need_recompute {
            let tile_slices: Vec<&[u8]> = cached_tiles
                .iter()
                .map(|(_, td)| td.data.as_slice())
                .chain(fallback_blits.iter().map(|(td, ..)| td.data.as_slice()))
                .collect();
            let params =
                crate::stain::compute_cpu_stain_params(hud_stain_normalization, &tile_slices);
            if let Some(ps) = file.pane_state_mut(pane) {
                ps.cached_stain_params = Some(params);
                ps.stain_params_epoch = loaded_tile_epoch;
                ps.stain_params_method = hud_stain_normalization;
            }
        }

        if let Some(ps) = file.pane_state(pane) {
            if let Some(ref params) = ps.cached_stain_params {
                crate::stain::apply_stain_params_to_buffer(buffer, params);
            }
        }
    }

    // Post-processing: apply sharpening (unsharp mask) if enabled
    // Reuse scratch buffer to avoid per-frame allocation.
    if !skip_expensive_postprocess && hud_sharpness > 0.001 {
        apply_sharpening_reuse(file, pane, buffer, render_width, render_height, hud_sharpness);
    }

    // Post-processing: apply gamma, brightness, contrast if they differ from defaults
    let has_adjustments = (hud_gamma - 1.0).abs() > 0.001
        || hud_brightness.abs() > 0.001
        || (hud_contrast - 1.0).abs() > 0.001;
    if has_adjustments {
        apply_adjustments(buffer, hud_gamma, hud_brightness, hud_contrast);
    }

    PaneRenderOutcome {
        image: Some(Image::from_rgba8(pixel_buffer)),
        keep_running,
        rendered: true,
    }
}

/// Apply sharpening reusing a persistent scratch buffer stored in
/// `FilePaneState` to avoid a full-frame allocation every frame.
fn apply_sharpening_reuse(
    file: &mut OpenFile,
    pane: PaneId,
    buffer: &mut [u8],
    width: u32,
    height: u32,
    sharpness: f32,
) {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return;
    }
    let needed = buffer.len();
    // Resize scratch buffer once; subsequent frames reuse the allocation.
    if let Some(ps) = file.pane_state_mut(pane) {
        if ps.scratch_buffer.len() < needed {
            ps.scratch_buffer.resize(needed, 0);
        }
        ps.scratch_buffer[..needed].copy_from_slice(buffer);
    }
    let Some(ps) = file.pane_state(pane) else {
        return;
    };
    let src = &ps.scratch_buffer;
    let stride = w * 4;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * stride + x * 4;
            let n = idx - stride;
            let s = idx + stride;
            let we = idx - 4;
            let e = idx + 4;
            for c in 0..3 {
                let center = src[idx + c] as f32;
                let neighbors =
                    src[n + c] as f32 + src[s + c] as f32 + src[we + c] as f32 + src[e + c] as f32;
                let detail = center * 4.0 - neighbors;
                let sharpened = center + sharpness * detail;
                buffer[idx + c] = sharpened.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn collect_tile_draws(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: TrilinearLevels,
    filtering_mode: FilteringMode,
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    let bounds = vp.bounds();
    let base_projection = TileProjection {
        viewport: vp,
        bounds_left: bounds.left,
        bounds_top: bounds.top,
        downsample: 1.0,
    };
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
            let Some(tile_data) = tile_cache.peek(coord) else {
                continue;
            };
            let projection = TileProjection {
                downsample: fallback_level_info.downsample,
                ..base_projection
            };
            if let Some(draw) =
                tile_draw_from_tile(projection, *coord, tile_data, None, filtering_mode)
            {
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
        let Some(tile_data) = tile_cache.peek(coord) else {
            continue;
        };
        let projection = TileProjection {
            downsample: level_info.downsample,
            ..base_projection
        };
        let coarse_blend = coarse_blend_for_tile(
            file,
            tile_cache,
            trilinear,
            level_info.downsample,
            *coord,
            &tile_data,
        );
        if let Some(draw) =
            tile_draw_from_tile(projection, *coord, tile_data, coarse_blend, filtering_mode)
        {
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
        fine_coord.file_id,
        trilinear.level_coarse,
        coarse_start_tile_x,
        coarse_start_tile_y,
        coarse_tile_size as u32,
    );

    let coarse_tile = tile_cache.peek(&coarse_coord)?;
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
    projection: TileProjection<'_>,
    coord: common::TileCoord,
    tile_data: Arc<common::TileData>,
    coarse_blend: Option<CoarseBlendData>,
    filtering_mode: FilteringMode,
) -> Option<TileDraw> {
    let TileProjection {
        viewport: vp,
        bounds_left,
        bounds_top,
        downsample,
    } = projection;

    let origin_x = coord.x as f64 * coord.tile_size as f64;
    let origin_y = coord.y as f64 * coord.tile_size as f64;
    let image_x = origin_x * downsample;
    let image_y = origin_y * downsample;
    let image_x_end = (origin_x + tile_data.width as f64) * downsample;
    let image_y_end = (origin_y + tile_data.height as f64) * downsample;

    let screen_x = ((image_x - bounds_left) * vp.zoom) as f32;
    let screen_y = ((image_y - bounds_top) * vp.zoom) as f32;
    let screen_x_end = ((image_x_end - bounds_left) * vp.zoom) as f32;
    let screen_y_end = ((image_y_end - bounds_top) * vp.zoom) as f32;
    let screen_w = screen_x_end - screen_x;
    let screen_h = screen_y_end - screen_y;

    if screen_w <= 0.0 || screen_h <= 0.0 {
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
        filtering_mode,
    })
}
