//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.

use crate::AppWindow;
use crate::blitter;
use crate::gpu::{QueuedFrame, SurfaceSlot, TileDraw};
use crate::render_pool::{
    CachedCpuFrame, CpuBlitCommand, CpuBlitKind, CpuRenderJob, CpuRenderPostProcess,
};
use crate::state::{
    AppState, FilteringMode, OpenFile, PaneId, RenderBackend, StainNormalization,
    TileRequestSignature,
};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use crate::tools;
use common::{TileCache, TileManager, Viewport, WsiFile};
use parking_lot::RwLock;
use rayon::prelude::*;
use slint::{ComponentHandle, Image};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

/// Conservative negative LOD bias applied only to the trilinear mip path.
/// This offsets a slight tendency to choose overly coarse mip blends while
/// keeping the result stable.
pub const TRILINEAR_LOD_BIAS: f64 = -0.25;

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

/// Result of trilinear level calculation
#[derive(Debug, Clone, Copy)]
pub struct TrilinearLevels {
    /// The higher resolution (lower index) level
    pub level_fine: u32,
    /// The lower resolution (higher index) level
    pub level_coarse: u32,
    /// Blend factor: 0.0 = use level_fine, 1.0 = use level_coarse
    pub blend: f64,
    /// Continuous LOD before the optional trilinear-only bias is applied.
    /// Kept for debugger inspection in quality tuning.
    pub lod_before_bias: f64,
    /// Continuous LOD after applying the trilinear-only bias and clamping.
    pub lod_after_bias: f64,
}

fn trilinear_lod_bias_enabled(filtering_mode: FilteringMode) -> bool {
    // Keep the correction restricted to the explicit Trilinear mode.
    // Lanczos remains unbiased even when it internally falls back to
    // trilinear blending at very low zoom.
    filtering_mode == FilteringMode::Trilinear
}

fn single_level_trilinear(level: u32) -> TrilinearLevels {
    let lod = level as f64;
    TrilinearLevels {
        level_fine: level,
        level_coarse: level,
        blend: 0.0,
        lod_before_bias: lod,
        lod_after_bias: lod,
    }
}

fn continuous_trilinear_lod(
    target_downsample: f64,
    fine_downsample: f64,
    coarse_downsample: f64,
    level_fine: u32,
) -> f64 {
    let log_target = target_downsample.max(f64::MIN_POSITIVE).ln();
    let log_fine = fine_downsample.max(f64::MIN_POSITIVE).ln();
    let log_coarse = coarse_downsample.max(f64::MIN_POSITIVE).ln();

    let blend = if (log_coarse - log_fine).abs() < 0.001 {
        0.0
    } else {
        ((log_target - log_fine) / (log_coarse - log_fine)).clamp(0.0, 1.0)
    };

    level_fine as f64 + blend
}

fn finalize_trilinear_levels(
    level_count: u32,
    lod_before_bias: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let max_lod = level_count.saturating_sub(1) as f64;
    let lod_after_bias = if apply_lod_bias {
        (lod_before_bias + TRILINEAR_LOD_BIAS).clamp(0.0, max_lod)
    } else {
        lod_before_bias.clamp(0.0, max_lod)
    };
    let level_fine = lod_after_bias.floor() as u32;
    let level_coarse = lod_after_bias.ceil() as u32;

    TrilinearLevels {
        level_fine,
        level_coarse,
        blend: lod_after_bias - level_fine as f64,
        lod_before_bias,
        lod_after_bias,
    }
}

/// Calculate the two mip levels to blend for trilinear filtering
pub fn calculate_trilinear_levels(
    wsi: &WsiFile,
    target_downsample: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let level_count = wsi.level_count();

    if level_count == 0 {
        return single_level_trilinear(0);
    }

    if level_count == 1 {
        return single_level_trilinear(0);
    }

    // Find the best level (where pixel density is closest to 1:1)
    let best_level = wsi.best_level_for_downsample(target_downsample);

    let best_info = match wsi.level(best_level) {
        Some(info) => info,
        None => {
            return single_level_trilinear(0);
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
            return single_level_trilinear(best_level);
        }
    } else {
        // Blend between previous finer level and best (coarse)
        if best_level > 0 {
            (best_level - 1, best_level)
        } else {
            // At finest level, no blending
            return single_level_trilinear(0);
        }
    };

    let lod_before_bias = match (wsi.level(level_fine), wsi.level(level_coarse)) {
        (Some(fine_info), Some(coarse_info)) => continuous_trilinear_lod(
            target_downsample,
            fine_info.downsample,
            coarse_info.downsample,
            level_fine,
        ),
        _ => {
            return single_level_trilinear(level_fine);
        }
    };

    finalize_trilinear_levels(level_count, lod_before_bias, apply_lod_bias)
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

#[derive(Clone)]
struct CpuPaneSnapshot {
    pane: PaneId,
    file_id: i32,
    file_switched: bool,
    content_missing: bool,
    minimap_missing: bool,
    minimap_image: Option<Image>,
    tile_manager: Arc<TileManager>,
    tile_loader: Arc<TileLoader>,
    viewport_state: common::ViewportState,
    frame_count: u32,
    last_render_zoom: f64,
    last_render_center_x: f64,
    last_render_center_y: f64,
    last_render_width: f64,
    last_render_height: f64,
    last_render_level: u32,
    previous_tiles_loaded: u32,
    last_seen_tile_epoch: u64,
    hud_sharpness: f32,
    hud_gamma: f32,
    hud_brightness: f32,
    hud_contrast: f32,
    hud_stain_normalization: StainNormalization,
    last_render_sharpness: f32,
    last_render_gamma: f32,
    last_render_brightness: f32,
    last_render_contrast: f32,
    last_render_stain_normalization: StainNormalization,
    pending_cpu_job_id: Option<u64>,
    needs_settled_cpu_render: bool,
    cached_stain_params: Option<crate::stain::StainNormParams>,
    stain_params_epoch: u64,
    stain_params_method: StainNormalization,
    filtering_mode: FilteringMode,
}

struct CpuFrameSnapshot {
    pane_count: usize,
    render_requested: bool,
    cpu_jobs_pending: bool,
    panes: Vec<Option<CpuPaneSnapshot>>,
}

struct CpuRenderCommit {
    frame_update: Option<CpuFrameStateUpdate>,
    pending_cpu_job_id: Option<Option<u64>>,
    needs_settled_cpu_render: Option<bool>,
    stain_cache_update: Option<(crate::stain::StainNormParams, u64, StainNormalization)>,
}

struct CpuFrameStateUpdate {
    frame_count: u32,
    last_render_time: std::time::Instant,
    last_render_zoom: f64,
    last_render_center_x: f64,
    last_render_center_y: f64,
    last_render_width: f64,
    last_render_height: f64,
    last_render_level: u32,
    tiles_loaded_since_render: u32,
    last_seen_tile_epoch: u64,
    last_render_gamma: f32,
    last_render_brightness: f32,
    last_render_contrast: f32,
    last_render_sharpness: f32,
    last_render_stain_normalization: StainNormalization,
}

struct CpuPaneExecution {
    pane: PaneId,
    file_id: i32,
    content_missing: bool,
    outcome: PaneRenderOutcome,
    commit: CpuRenderCommit,
}

#[derive(Clone, Copy)]
struct TileProjection<'a> {
    viewport: &'a Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
}

struct CachedVisibleTiles {
    fine_tiles: Vec<(common::TileCoord, Arc<common::TileData>)>,
    coarse_tiles: Vec<(common::TileCoord, Arc<common::TileData>)>,
}

#[derive(Clone, Copy)]
struct RenderAdjustments {
    sharpness: f32,
    gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_normalization: StainNormalization,
}

impl RenderAdjustments {
    fn changed_from(self, previous: Self) -> bool {
        (self.sharpness - previous.sharpness).abs() > 0.001
            || (self.gamma - previous.gamma).abs() > 0.001
            || (self.brightness - previous.brightness).abs() > 0.001
            || (self.contrast - previous.contrast).abs() > 0.001
            || self.stain_normalization != previous.stain_normalization
    }
}

fn requested_trilinear_blend(filtering_mode: FilteringMode, zoom: f64) -> (f64, bool) {
    let lanczos_weight = if filtering_mode == FilteringMode::Lanczos3 {
        lanczos_adaptive_weight(zoom)
    } else {
        1.0
    };
    let use_trilinear_blend = filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0);
    (lanczos_weight, use_trilinear_blend)
}

fn effective_cpu_filtering_mode(
    filtering_mode: FilteringMode,
    use_trilinear_blend: bool,
    is_moving: bool,
) -> FilteringMode {
    if is_moving {
        if use_trilinear_blend {
            FilteringMode::Trilinear
        } else {
            FilteringMode::Bilinear
        }
    } else {
        filtering_mode
    }
}

fn filter_uses_trilinear(filtering_mode: FilteringMode, lanczos_weight: f64) -> bool {
    filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0)
}

fn collect_cached_visible_tiles(
    tile_manager: &TileManager,
    tile_cache: &Arc<TileCache>,
    viewport: &Viewport,
    trilinear: TrilinearLevels,
    margin_tiles: i32,
    fetch_coarse_tiles: bool,
) -> CachedVisibleTiles {
    let bounds = viewport.bounds();
    let fine_tiles = tile_manager
        .visible_tiles_with_margin(
            trilinear.level_fine,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        )
        .into_iter()
        .take(500)
        .filter_map(|coord| tile_cache.peek(&coord).map(|data| (coord, data)))
        .collect();

    let coarse_tiles = if fetch_coarse_tiles
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
        tile_manager
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

    CachedVisibleTiles {
        fine_tiles,
        coarse_tiles,
    }
}

fn resolve_stain_params_from_tiles(
    method: StainNormalization,
    loaded_tile_epoch: u64,
    cached_stain_params: Option<crate::stain::StainNormParams>,
    cached_epoch: u64,
    cached_method: StainNormalization,
    tile_slices: &[&[u8]],
) -> (
    Option<crate::stain::StainNormParams>,
    Option<(crate::stain::StainNormParams, u64, StainNormalization)>,
) {
    if method == StainNormalization::None {
        return (None, None);
    }

    let cache_valid = cached_stain_params.is_some()
        && cached_epoch == loaded_tile_epoch
        && cached_method == method;
    if cache_valid {
        return (cached_stain_params, None);
    }

    let params = crate::stain::compute_cpu_stain_params(method, tile_slices);
    (Some(params), Some((params, loaded_tile_epoch, method)))
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
    let (completed_cpu_frame, cpu_jobs_pending, render_backend) = {
        let mut state = state.write();
        let (completed_cpu_frame, cpu_jobs_pending) = apply_completed_cpu_renders(&mut state);
        (completed_cpu_frame, cpu_jobs_pending, state.render_backend)
    };

    if render_backend == RenderBackend::Cpu {
        return update_and_render_cpu(ui, state, tile_cache, completed_cpu_frame, cpu_jobs_pending);
    }

    let mut state = state.write();
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

    let render_requested = std::mem::take(&mut state.needs_render) || completed_cpu_frame;
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
            let trilinear = calculate_trilinear_levels(
                &file.wsi,
                vp.effective_downsample(),
                trilinear_lod_bias_enabled(filtering_mode),
            );
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

    let mut keep_running = render_requested || cpu_jobs_pending;
    let mut rendered_frame = completed_cpu_frame;

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

fn update_and_render_cpu(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    completed_cpu_frame: bool,
    cpu_jobs_pending: bool,
) -> bool {
    let Some(frame) = collect_cpu_frame_snapshot(ui, state, completed_cpu_frame, cpu_jobs_pending)
    else {
        return false;
    };

    let mut wanted_tiles_by_file: HashMap<i32, (Arc<TileLoader>, HashSet<common::TileCoord>)> =
        HashMap::new();

    for snapshot in frame.panes.iter().flatten() {
        let vp = &snapshot.viewport_state.viewport;
        let margin_tiles = if snapshot.viewport_state.is_moving() {
            1
        } else {
            0
        };
        let trilinear = calculate_trilinear_levels(
            snapshot.tile_manager.wsi(),
            vp.effective_downsample(),
            trilinear_lod_bias_enabled(snapshot.filtering_mode),
        );
        let wanted = calculate_wanted_tiles(
            &snapshot.tile_manager,
            trilinear.level_fine,
            vp.bounds().left,
            vp.bounds().top,
            vp.bounds().right,
            vp.bounds().bottom,
            margin_tiles,
        );

        let entry = wanted_tiles_by_file
            .entry(snapshot.file_id)
            .or_insert_with(|| (Arc::clone(&snapshot.tile_loader), HashSet::new()));
        entry.1.extend(wanted);
    }

    for (_, (tile_loader, wanted_tiles)) in wanted_tiles_by_file {
        tile_loader.set_wanted_tiles(wanted_tiles.into_iter().collect());
    }

    let mut keep_running = frame.render_requested || frame.cpu_jobs_pending;
    let mut rendered_frame = completed_cpu_frame;

    for pane_index in 0..frame.pane_count {
        let pane = PaneId(pane_index);
        let Some(snapshot) = frame
            .panes
            .get(pane_index)
            .and_then(|snapshot| snapshot.clone())
        else {
            crate::clear_cached_pane(pane);
            let mut state = state.write();
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        if snapshot.file_switched || snapshot.minimap_missing {
            crate::set_cached_pane_minimap(snapshot.pane, snapshot.minimap_image.clone());
        }

        let execution = render_cpu_pane_from_snapshot(&snapshot, tile_cache);
        keep_running |= execution.outcome.keep_running || snapshot.file_switched;
        rendered_frame |= execution.outcome.rendered;

        if let Some(image) = execution.outcome.image.clone() {
            crate::set_cached_pane_content(snapshot.pane, image);
        }

        apply_cpu_render_commit(state, execution);
    }

    let mut state = state.write();
    crate::update_tabs(ui, &state);
    keep_running |= tools::has_active_roi_overlay(&state);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

fn collect_cpu_frame_snapshot(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    completed_cpu_frame: bool,
    cpu_jobs_pending: bool,
) -> Option<CpuFrameSnapshot> {
    let mut state = state.write();
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
        return None;
    }

    let render_requested = std::mem::take(&mut state.needs_render) || completed_cpu_frame;
    state.ant_offset = (state.ant_offset + 0.5) % 16.0;

    let content_width = (ui.get_content_area_width() as f64).max(100.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(100.0);
    let pane_gap = 6.0;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(100.0);

    let mut panes = Vec::with_capacity(pane_count);
    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let Some(file_id) = file_id else {
            panes.push(None);
            continue;
        };

        let file_switched = state.last_rendered_file_id(pane) != Some(file_id);
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

        let filtering_mode = state.filtering_mode;
        let Some(file) = state.open_files.iter_mut().find(|file| file.id == file_id) else {
            panes.push(None);
            continue;
        };
        let tile_manager = Arc::clone(&file.tile_manager);
        let tile_loader = Arc::clone(&file.tile_loader);
        let minimap_image = if file_switched || minimap_missing {
            thumbnail_image_for_file(file)
        } else {
            None
        };

        let Some((
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
            pending_cpu_job_id,
            needs_settled_cpu_render,
            cached_stain_params,
            stain_params_epoch,
            stain_params_method,
        )) = (match file.pane_state_mut(pane) {
            Some(pane_state) => {
                if file_switched {
                    pane_state.frame_count = 0;
                }
                let _ = pane_state.viewport.update();
                pane_state.viewport.set_size(pane_width, content_height);
                Some((
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
                    pane_state.pending_cpu_job_id,
                    pane_state.needs_settled_cpu_render,
                    pane_state.cached_stain_params,
                    pane_state.stain_params_epoch,
                    pane_state.stain_params_method,
                ))
            }
            None => None,
        })
        else {
            panes.push(None);
            continue;
        };

        let vp = &viewport_state.viewport;
        let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
        let trilinear = calculate_trilinear_levels(
            tile_manager.wsi(),
            vp.effective_downsample(),
            trilinear_lod_bias_enabled(filtering_mode),
        );
        let request_signature =
            tile_request_signature(&tile_manager, vp, trilinear.level_fine, margin_tiles);
        if let Some(pane_state) = file.pane_state_mut(pane)
            && pane_state.last_request != request_signature
        {
            pane_state.last_request = request_signature;
        }

        panes.push(Some(CpuPaneSnapshot {
            pane,
            file_id,
            file_switched,
            content_missing,
            minimap_missing,
            minimap_image,
            tile_manager,
            tile_loader,
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
            pending_cpu_job_id,
            needs_settled_cpu_render,
            cached_stain_params,
            stain_params_epoch,
            stain_params_method,
            filtering_mode,
        }));
    }

    Some(CpuFrameSnapshot {
        pane_count,
        render_requested,
        cpu_jobs_pending,
        panes,
    })
}

fn apply_cpu_render_commit(state: &Arc<RwLock<AppState>>, execution: CpuPaneExecution) {
    let mut state = state.write();
    let pane = execution.pane;
    let file_id = execution.file_id;

    if let Some(file) = state.open_files.iter_mut().find(|file| file.id == file_id)
        && let Some(pane_state) = file.pane_state_mut(pane)
    {
        if let Some(frame_update) = execution.commit.frame_update {
            pane_state.frame_count = frame_update.frame_count;
            pane_state.last_render_time = frame_update.last_render_time;
            pane_state.last_render_zoom = frame_update.last_render_zoom;
            pane_state.last_render_center_x = frame_update.last_render_center_x;
            pane_state.last_render_center_y = frame_update.last_render_center_y;
            pane_state.last_render_width = frame_update.last_render_width;
            pane_state.last_render_height = frame_update.last_render_height;
            pane_state.last_render_level = frame_update.last_render_level;
            pane_state.tiles_loaded_since_render = frame_update.tiles_loaded_since_render;
            pane_state.last_seen_tile_epoch = frame_update.last_seen_tile_epoch;
            pane_state.last_render_gamma = frame_update.last_render_gamma;
            pane_state.last_render_brightness = frame_update.last_render_brightness;
            pane_state.last_render_contrast = frame_update.last_render_contrast;
            pane_state.last_render_sharpness = frame_update.last_render_sharpness;
            pane_state.last_render_stain_normalization =
                frame_update.last_render_stain_normalization;
        }

        if let Some(pending_cpu_job_id) = execution.commit.pending_cpu_job_id {
            pane_state.pending_cpu_job_id = pending_cpu_job_id;
        }
        if let Some(needs_settled_cpu_render) = execution.commit.needs_settled_cpu_render {
            pane_state.needs_settled_cpu_render = needs_settled_cpu_render;
        }
        if let Some((params, epoch, method)) = execution.commit.stain_cache_update {
            pane_state.cached_stain_params = Some(params);
            pane_state.stain_params_epoch = epoch;
            pane_state.stain_params_method = method;
        }
    }

    if execution.outcome.image.is_some() || execution.content_missing {
        state.set_last_rendered_file_id(pane, Some(file_id));
    }
}

fn apply_completed_cpu_renders(state: &mut AppState) -> (bool, bool) {
    let Some(pool) = crate::render_pool::global() else {
        return (false, false);
    };

    let mut applied = false;

    while let Some(result) = pool.try_recv() {
        let pane = PaneId(result.pane_index);
        let is_active = state.active_file_id_for_pane(pane) == Some(result.file_id);
        let Some(file) = state
            .open_files
            .iter_mut()
            .find(|file| file.id == result.file_id)
        else {
            pool.recycle_buffer(result.pixels);
            continue;
        };
        let Some(pane_state) = file.pane_state_mut(pane) else {
            pool.recycle_buffer(result.pixels);
            continue;
        };

        if pane_state.pending_cpu_job_id != Some(result.job_id) {
            pool.recycle_buffer(result.pixels);
            continue;
        }

        pane_state.pending_cpu_job_id = None;
        pane_state.needs_settled_cpu_render = !result.settled_quality;

        if !is_active {
            pool.recycle_buffer(result.pixels);
            continue;
        }

        let Some(buffer) =
            blitter::create_image_buffer(&result.pixels, result.width, result.height)
        else {
            pool.recycle_buffer(result.pixels);
            continue;
        };

        crate::set_cached_pane_cpu_result(
            pane,
            Image::from_rgba8(buffer),
            CachedCpuFrame {
                file_id: result.file_id,
                width: result.width,
                height: result.height,
                viewport: result.viewport,
                pixels: result.pixels,
            },
        );
        state.set_last_rendered_file_id(pane, Some(result.file_id));
        applied = true;
    }

    let pending = state
        .open_files
        .iter()
        .flat_map(|file| file.pane_states.iter().flatten())
        .any(|pane_state| pane_state.pending_cpu_job_id.is_some());

    if applied {
        state.request_render();
    }

    (applied, pending)
}

fn render_cpu_pane_from_snapshot(
    snapshot: &CpuPaneSnapshot,
    tile_cache: &Arc<TileCache>,
) -> CpuPaneExecution {
    let vp = &snapshot.viewport_state.viewport;
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = snapshot.frame_count == 0;
    let is_moving = snapshot.viewport_state.is_moving();
    let viewport_changed = is_moving
        || (snapshot.last_render_zoom - vp_zoom).abs() > 0.001
        || (snapshot.last_render_center_x - vp_center_x).abs() > 1.0
        || (snapshot.last_render_center_y - vp_center_y).abs() > 1.0
        || (snapshot.last_render_width - vp_width).abs() > 1.0
        || (snapshot.last_render_height - vp_height).abs() > 1.0;

    let trilinear = calculate_trilinear_levels(
        snapshot.tile_manager.wsi(),
        vp.effective_downsample(),
        trilinear_lod_bias_enabled(snapshot.filtering_mode),
    );
    let level = trilinear.level_fine;
    let margin_tiles = if is_moving { 1 } else { 0 };
    let level_changed = level != snapshot.last_render_level;
    let bounds = vp.bounds();
    let (lanczos_weight, use_trilinear_blend) =
        requested_trilinear_blend(snapshot.filtering_mode, vp_zoom);
    let cached_tiles = collect_cached_visible_tiles(
        &snapshot.tile_manager,
        tile_cache,
        vp,
        trilinear,
        margin_tiles,
        use_trilinear_blend,
    );
    let cached_count = cached_tiles.fine_tiles.len() as u32;

    let loaded_tile_epoch = snapshot.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > snapshot.last_seen_tile_epoch;
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            snapshot.previous_tiles_loaded
        }
        || !cached_tiles.coarse_tiles.is_empty()
        || tile_epoch_advanced;
    let tiles_pending = snapshot.tile_loader.pending_count() > 0;
    let keep_running = is_moving || new_tiles_loaded || tiles_pending;

    let adjustments_changed = RenderAdjustments {
        sharpness: snapshot.hud_sharpness,
        gamma: snapshot.hud_gamma,
        brightness: snapshot.hud_brightness,
        contrast: snapshot.hud_contrast,
        stain_normalization: snapshot.hud_stain_normalization,
    }
    .changed_from(RenderAdjustments {
        sharpness: snapshot.last_render_sharpness,
        gamma: snapshot.last_render_gamma,
        brightness: snapshot.last_render_brightness,
        contrast: snapshot.last_render_contrast,
        stain_normalization: snapshot.last_render_stain_normalization,
    });

    let mut commit = CpuRenderCommit {
        frame_update: None,
        pending_cpu_job_id: None,
        needs_settled_cpu_render: None,
        stain_cache_update: None,
    };

    if !snapshot.file_switched
        && !snapshot.content_missing
        && !is_first_frame
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
        && !snapshot.needs_settled_cpu_render
    {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running: keep_running || snapshot.pending_cpu_job_id.is_some(),
                rendered: false,
            },
            commit,
        };
    }

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running,
                rendered: false,
            },
            commit,
        };
    }

    commit.frame_update = Some(CpuFrameStateUpdate {
        frame_count: snapshot.frame_count + 1,
        last_render_time: std::time::Instant::now(),
        last_render_zoom: vp_zoom,
        last_render_center_x: vp_center_x,
        last_render_center_y: vp_center_y,
        last_render_width: vp_width,
        last_render_height: vp_height,
        last_render_level: level,
        tiles_loaded_since_render: cached_count,
        last_seen_tile_epoch: loaded_tile_epoch,
        last_render_gamma: snapshot.hud_gamma,
        last_render_brightness: snapshot.hud_brightness,
        last_render_contrast: snapshot.hud_contrast,
        last_render_sharpness: snapshot.hud_sharpness,
        last_render_stain_normalization: snapshot.hud_stain_normalization,
    });

    let Some(level_info) = snapshot.tile_manager.wsi().level(level).cloned() else {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running: keep_running || snapshot.pending_cpu_job_id.is_some(),
                rendered: false,
            },
            commit,
        };
    };

    let mut pending_cpu_job_id = snapshot.pending_cpu_job_id;
    if pending_cpu_job_id.is_some() && (viewport_changed || level_changed || adjustments_changed) {
        commit.pending_cpu_job_id = Some(None);
        pending_cpu_job_id = None;
    }
    if is_moving {
        commit.needs_settled_cpu_render = Some(true);
    }

    let filtering_mode =
        effective_cpu_filtering_mode(snapshot.filtering_mode, use_trilinear_blend, is_moving);
    let is_adaptive_lanczos = filtering_mode == FilteringMode::Lanczos3;
    let effective_lanczos_weight = if is_adaptive_lanczos {
        lanczos_weight
    } else {
        1.0
    };
    let effective_trilinear = filter_uses_trilinear(filtering_mode, effective_lanczos_weight);

    let level_count = snapshot.tile_manager.wsi().level_count();
    let mut fallback_commands: Vec<CpuBlitCommand> = Vec::new();
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue;
        }
        let Some(fallback_level_info) = snapshot.tile_manager.wsi().level(fallback_level).cloned()
        else {
            continue;
        };
        let fallback_tiles = snapshot.tile_manager.visible_tiles(
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

            fallback_commands.push(CpuBlitCommand {
                tile: fallback_tile,
                rect: blitter::BlitRect {
                    x: screen_x,
                    y: screen_y,
                    width: screen_w,
                    height: screen_h,
                    exact_x: exact_sx,
                    exact_y: exact_sy,
                    exact_width: exact_sx_end - exact_sx,
                    exact_height: exact_sy_end - exact_sy,
                },
                kind: CpuBlitKind::Bilinear,
            });
        }
    }

    let fine_blit_rect =
        |coord: &common::TileCoord, tile_data: &common::TileData| -> blitter::BlitRect {
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

    let cached_coarse_tile_map: HashMap<_, _> = cached_tiles.coarse_tiles.iter().cloned().collect();
    let do_fused_trilinear = effective_trilinear
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
        && !cached_coarse_tile_map.is_empty();
    let coarse_blends: HashMap<_, _> = if do_fused_trilinear {
        cached_tiles
            .fine_tiles
            .iter()
            .filter_map(|(coord, tile)| {
                coarse_blend_for_tile(
                    &snapshot.tile_manager,
                    &cached_coarse_tile_map,
                    trilinear,
                    level_info.downsample,
                    *coord,
                    tile,
                )
                .map(|blend| (*coord, blend))
            })
            .collect()
    } else {
        HashMap::new()
    };

    let mut fine_commands: Vec<CpuBlitCommand> = Vec::with_capacity(cached_tiles.fine_tiles.len());
    for (coord, tile_data) in cached_tiles.fine_tiles.iter() {
        let rect = fine_blit_rect(coord, tile_data);
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }
        let kind = if let Some((coarse_tile, uv_min, uv_max, blend)) = coarse_blends.get(coord) {
            CpuBlitKind::Trilinear {
                coarse_tile: Arc::clone(coarse_tile),
                uv_min: *uv_min,
                uv_max: *uv_max,
                blend: *blend,
            }
        } else if is_adaptive_lanczos && effective_lanczos_weight >= 1.0 {
            CpuBlitKind::Lanczos3
        } else {
            CpuBlitKind::Bilinear
        };
        fine_commands.push(CpuBlitCommand {
            tile: Arc::clone(tile_data),
            rect,
            kind,
        });
    }

    let stain_params = if !is_moving && snapshot.hud_stain_normalization != StainNormalization::None
    {
        let tile_slices: Vec<&[u8]> = cached_tiles
            .fine_tiles
            .iter()
            .map(|(_, td)| td.data.as_slice())
            .chain(fallback_commands.iter().map(|cmd| cmd.tile.data.as_slice()))
            .collect();
        let (params, cache_update) = resolve_stain_params_from_tiles(
            snapshot.hud_stain_normalization,
            loaded_tile_epoch,
            snapshot.cached_stain_params,
            snapshot.stain_params_epoch,
            snapshot.stain_params_method,
            &tile_slices,
        );
        if let Some(cache_update) = cache_update {
            commit.stain_cache_update = Some(cache_update);
        }
        params
    } else {
        None
    };

    let preview_image = if is_moving {
        render_cached_preview(
            snapshot.pane,
            snapshot.file_id,
            vp,
            render_width,
            render_height,
            &fallback_commands,
            &fine_commands,
        )
    } else {
        None
    };

    if pending_cpu_job_id.is_none()
        && let Some(pool) = crate::render_pool::global()
    {
        let job_id = pool.next_job_id();
        pool.submit(CpuRenderJob {
            pane_index: snapshot.pane.0,
            file_id: snapshot.file_id,
            job_id,
            width: render_width,
            height: render_height,
            viewport: vp.clone(),
            background_rgba: [30, 30, 30, 255],
            fallback_blits: fallback_commands,
            fine_blits: fine_commands,
            postprocess: CpuRenderPostProcess {
                stain_params: if is_moving { None } else { stain_params },
                sharpness: if is_moving {
                    0.0
                } else {
                    snapshot.hud_sharpness
                },
                gamma: snapshot.hud_gamma,
                brightness: snapshot.hud_brightness,
                contrast: snapshot.hud_contrast,
            },
            settled_quality: !is_moving,
        });
        commit.pending_cpu_job_id = Some(Some(job_id));
        commit.needs_settled_cpu_render = Some(is_moving);
        pending_cpu_job_id = Some(job_id);
    }

    let outcome = if let Some(image) = preview_image {
        PaneRenderOutcome {
            image: Some(image),
            keep_running: true,
            rendered: true,
        }
    } else {
        PaneRenderOutcome {
            image: None,
            keep_running: keep_running
                || pending_cpu_job_id.is_some()
                || snapshot.needs_settled_cpu_render,
            rendered: false,
        }
    };

    CpuPaneExecution {
        pane: snapshot.pane,
        file_id: snapshot.file_id,
        content_missing: snapshot.content_missing,
        outcome,
        commit,
    }
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
        pending_cpu_job_id,
        needs_settled_cpu_render,
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
            pane_state.pending_cpu_job_id,
            pane_state.needs_settled_cpu_render,
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

    let trilinear = calculate_trilinear_levels(
        &file.wsi,
        vp.effective_downsample(),
        trilinear_lod_bias_enabled(filtering_mode),
    );
    let level = trilinear.level_fine;
    let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
    let level_changed = level != last_render_level;

    let bounds = vp.bounds();
    let (lanczos_weight, use_trilinear_blend) = requested_trilinear_blend(filtering_mode, vp_zoom);
    let cached_tiles = collect_cached_visible_tiles(
        &file.tile_manager,
        tile_cache,
        vp,
        trilinear,
        margin_tiles,
        use_trilinear_blend,
    );
    let cached_count = cached_tiles.fine_tiles.len() as u32;

    let loaded_tile_epoch = file.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > last_seen_tile_epoch;
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            previous_tiles_loaded
        }
        || !cached_tiles.coarse_tiles.is_empty()
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
            trilinear_lod_before_bias = trilinear.lod_before_bias,
            trilinear_lod_after_bias = trilinear.lod_after_bias,
            cached_count,
            coarse_tiles = cached_tiles.coarse_tiles.len(),
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

    let adjustments_changed = RenderAdjustments {
        sharpness: hud_sharpness,
        gamma: hud_gamma,
        brightness: hud_brightness,
        contrast: hud_contrast,
        stain_normalization: hud_stain_normalization,
    }
    .changed_from(RenderAdjustments {
        sharpness: last_render_sharpness,
        gamma: last_render_gamma,
        brightness: last_render_brightness,
        contrast: last_render_contrast,
        stain_normalization: last_render_stain_normalization,
    });

    if !force_render
        && !is_first_frame
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
        && !needs_settled_cpu_render
    {
        return PaneRenderOutcome {
            image: None,
            keep_running: keep_running || pending_cpu_job_id.is_some(),
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
            single_level_trilinear(trilinear.level_fine)
        };
        let draws = collect_tile_draws_from_cached(
            file,
            tile_cache,
            vp,
            gpu_trilinear,
            gpu_filtering,
            &cached_tiles.fine_tiles,
            &cached_tiles.coarse_tiles,
        );
        let tile_slices: Vec<&[u8]> = draws.iter().map(|draw| draw.tile.data.as_slice()).collect();
        let (stain_params, stain_cache_update) = resolve_stain_params_from_tiles(
            hud_stain_normalization,
            loaded_tile_epoch,
            file.pane_state(pane).and_then(|ps| ps.cached_stain_params),
            file.pane_state(pane)
                .map(|ps| ps.stain_params_epoch)
                .unwrap_or(0),
            file.pane_state(pane)
                .map(|ps| ps.stain_params_method)
                .unwrap_or(StainNormalization::None),
            &tile_slices,
        );
        if let Some((params, epoch, method)) = stain_cache_update
            && let Some(pane_state) = file.pane_state_mut(pane)
        {
            pane_state.cached_stain_params = Some(params);
            pane_state.stain_params_epoch = epoch;
            pane_state.stain_params_method = method;
        }
        let stain_params = stain_params.unwrap_or_default();
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
                keep_running: keep_running || pending_cpu_job_id.is_some(),
                rendered: false,
            };
        }
    };

    let is_moving = viewport_state.is_moving() || animating;
    let mut pending_cpu_job_id = pending_cpu_job_id;

    if pending_cpu_job_id.is_some() && (viewport_changed || level_changed || adjustments_changed) {
        if let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.pending_cpu_job_id = None;
        }
        pending_cpu_job_id = None;
    }

    if is_moving && let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.needs_settled_cpu_render = true;
    }

    let effective_filtering =
        effective_cpu_filtering_mode(filtering_mode, use_trilinear_blend, is_moving);

    let is_adaptive_lanczos = effective_filtering == FilteringMode::Lanczos3;
    let effective_lanczos_weight = if is_adaptive_lanczos {
        lanczos_weight
    } else {
        1.0
    };
    let effective_trilinear = filter_uses_trilinear(effective_filtering, effective_lanczos_weight);

    let level_count = file.wsi.level_count();
    let mut fallback_commands: Vec<CpuBlitCommand> = Vec::new();
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

            fallback_commands.push(CpuBlitCommand {
                tile: fallback_tile,
                rect: blitter::BlitRect {
                    x: screen_x,
                    y: screen_y,
                    width: screen_w,
                    height: screen_h,
                    exact_x: exact_sx,
                    exact_y: exact_sy,
                    exact_width: exact_sx_end - exact_sx,
                    exact_height: exact_sy_end - exact_sy,
                },
                kind: CpuBlitKind::Bilinear,
            });
        }
    }

    let fine_blit_rect =
        |coord: &common::TileCoord, tile_data: &common::TileData| -> blitter::BlitRect {
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

    let cached_coarse_tile_map: HashMap<_, _> = cached_tiles.coarse_tiles.iter().cloned().collect();
    let do_fused_trilinear = effective_trilinear
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
        && !cached_coarse_tile_map.is_empty();
    let coarse_blends: HashMap<_, _> = if do_fused_trilinear {
        cached_tiles
            .fine_tiles
            .iter()
            .filter_map(|(coord, tile)| {
                coarse_blend_for_tile(
                    &file.tile_manager,
                    &cached_coarse_tile_map,
                    trilinear,
                    level_info.downsample,
                    *coord,
                    tile,
                )
                .map(|blend| (*coord, blend))
            })
            .collect()
    } else {
        HashMap::new()
    };

    let mut fine_commands: Vec<CpuBlitCommand> = Vec::with_capacity(cached_tiles.fine_tiles.len());
    for (coord, tile_data) in cached_tiles.fine_tiles.iter() {
        let rect = fine_blit_rect(coord, tile_data);
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }

        let kind = if let Some((coarse_tile, uv_min, uv_max, blend)) = coarse_blends.get(coord) {
            CpuBlitKind::Trilinear {
                coarse_tile: Arc::clone(coarse_tile),
                uv_min: *uv_min,
                uv_max: *uv_max,
                blend: *blend,
            }
        } else if is_adaptive_lanczos && effective_lanczos_weight >= 1.0 {
            CpuBlitKind::Lanczos3
        } else {
            CpuBlitKind::Bilinear
        };

        fine_commands.push(CpuBlitCommand {
            tile: Arc::clone(tile_data),
            rect,
            kind,
        });
    }

    let stain_params =
        if !is_moving && hud_stain_normalization != crate::state::StainNormalization::None {
            let tile_slices: Vec<&[u8]> = cached_tiles
                .fine_tiles
                .iter()
                .map(|(_, td)| td.data.as_slice())
                .chain(fallback_commands.iter().map(|cmd| cmd.tile.data.as_slice()))
                .collect();
            let (params, cache_update) = resolve_stain_params_from_tiles(
                hud_stain_normalization,
                loaded_tile_epoch,
                file.pane_state(pane).and_then(|ps| ps.cached_stain_params),
                file.pane_state(pane)
                    .map(|ps| ps.stain_params_epoch)
                    .unwrap_or(0),
                file.pane_state(pane)
                    .map(|ps| ps.stain_params_method)
                    .unwrap_or(StainNormalization::None),
                &tile_slices,
            );
            if let Some((params, epoch, method)) = cache_update
                && let Some(ps) = file.pane_state_mut(pane)
            {
                ps.cached_stain_params = Some(params);
                ps.stain_params_epoch = epoch;
                ps.stain_params_method = method;
            }
            params
        } else {
            None
        };

    let preview_image = if is_moving {
        render_cached_preview(
            pane,
            file.id,
            vp,
            render_width,
            render_height,
            &fallback_commands,
            &fine_commands,
        )
    } else {
        None
    };

    let submit_cpu_job = !is_moving || pending_cpu_job_id.is_none();
    if submit_cpu_job && pending_cpu_job_id.is_none() {
        let Some(pool) = crate::render_pool::global() else {
            return PaneRenderOutcome::default();
        };

        let job_id = pool.next_job_id();
        pool.submit(CpuRenderJob {
            pane_index: pane.0,
            file_id: file.id,
            job_id,
            width: render_width,
            height: render_height,
            viewport: vp.clone(),
            background_rgba: [30, 30, 30, 255],
            fallback_blits: fallback_commands,
            fine_blits: fine_commands,
            postprocess: CpuRenderPostProcess {
                stain_params: if is_moving { None } else { stain_params },
                sharpness: if is_moving { 0.0 } else { hud_sharpness },
                gamma: hud_gamma,
                brightness: hud_brightness,
                contrast: hud_contrast,
            },
            settled_quality: !is_moving,
        });

        if let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.pending_cpu_job_id = Some(job_id);
            pane_state.needs_settled_cpu_render = is_moving;
        }
        pending_cpu_job_id = Some(job_id);
    }

    if let Some(image) = preview_image {
        return PaneRenderOutcome {
            image: Some(image),
            keep_running: true,
            rendered: true,
        };
    }

    PaneRenderOutcome {
        image: None,
        keep_running: keep_running || pending_cpu_job_id.is_some() || needs_settled_cpu_render,
        rendered: false,
    }
}

fn render_cached_preview(
    pane: PaneId,
    file_id: i32,
    viewport: &Viewport,
    render_width: u32,
    render_height: u32,
    fallback_commands: &[CpuBlitCommand],
    fine_commands: &[CpuBlitCommand],
) -> Option<Image> {
    crate::with_pane_render_cache(pane.0 + 1, |cache| {
        let entry = cache.get_mut(pane.0)?;
        let cpu_frame = entry.cpu_frame.as_ref()?;
        if cpu_frame.file_id != file_id {
            return None;
        }

        let needed = (render_width as usize) * (render_height as usize) * 4;
        if entry.preview_buffer.len() < needed {
            entry.preview_buffer.resize(needed, 0);
        }
        entry.preview_width = render_width;
        entry.preview_height = render_height;
        let preview = &mut entry.preview_buffer[..needed];
        blitter::reproject_frame(
            preview,
            render_width,
            render_height,
            blitter::FrameSrc {
                pixels: &cpu_frame.pixels,
                width: cpu_frame.width,
                height: cpu_frame.height,
            },
            cpu_frame,
            viewport,
            [30, 30, 30, 255],
        );

        composite_commands_into_preview_parallel(
            preview,
            render_width,
            render_height,
            fallback_commands,
        );
        composite_commands_into_preview_parallel(
            preview,
            render_width,
            render_height,
            fine_commands,
        );

        blitter::create_image_buffer(preview, render_width, render_height).map(Image::from_rgba8)
    })
}

fn composite_commands_into_preview_parallel(
    buffer: &mut [u8],
    render_width: u32,
    render_height: u32,
    commands: &[CpuBlitCommand],
) {
    if commands.is_empty() || render_width == 0 || render_height == 0 {
        return;
    }

    let stride = render_width as usize * 4;
    let rows_per_chunk = ((render_height as usize) / rayon::current_num_threads()).max(32);

    buffer
        .par_chunks_mut(rows_per_chunk * stride)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start_row = chunk_index * rows_per_chunk;
            let chunk_rows = chunk.len() / stride;
            let chunk_height = chunk_rows as u32;
            let row_offset = start_row as i32;

            for command in commands {
                composite_command_into_preview_chunk(
                    chunk,
                    render_width,
                    chunk_height,
                    row_offset,
                    command,
                );
            }
        });
}

fn composite_command_into_preview_chunk(
    buffer: &mut [u8],
    render_width: u32,
    render_height: u32,
    row_offset: i32,
    command: &CpuBlitCommand,
) {
    let mut rect = command.rect;
    rect.y -= row_offset;
    rect.exact_y -= row_offset as f64;

    match &command.kind {
        CpuBlitKind::Bilinear => blitter::blit_tile(
            buffer,
            render_width,
            render_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
            rect,
        ),
        CpuBlitKind::Lanczos3 => blitter::blit_tile_lanczos3(
            buffer,
            render_width,
            render_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
            rect,
        ),
        CpuBlitKind::Trilinear {
            coarse_tile,
            uv_min,
            uv_max,
            blend,
        } => blitter::blit_tile_trilinear(
            buffer,
            render_width,
            render_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
            &blitter::CoarseSrc {
                data: &coarse_tile.data,
                width: coarse_tile.width,
                height: coarse_tile.height,
                border: coarse_tile.border,
                uv_min: *uv_min,
                uv_max: *uv_max,
                blend: *blend,
            },
            rect,
        ),
    }
}

fn collect_tile_draws_from_cached(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: TrilinearLevels,
    filtering_mode: FilteringMode,
    cached_tiles: &[(common::TileCoord, Arc<common::TileData>)],
    cached_coarse_tiles: &[(common::TileCoord, Arc<common::TileData>)],
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
    let coarse_tile_map: HashMap<common::TileCoord, Arc<common::TileData>> = if filtering_mode
        == FilteringMode::Trilinear
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
        cached_coarse_tiles.iter().cloned().collect()
    } else {
        HashMap::new()
    };

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

    for (coord, tile_data) in cached_tiles.iter() {
        let projection = TileProjection {
            downsample: level_info.downsample,
            ..base_projection
        };
        let coarse_blend = coarse_blend_for_tile(
            &file.tile_manager,
            &coarse_tile_map,
            trilinear,
            level_info.downsample,
            *coord,
            tile_data,
        );
        if let Some(draw) = tile_draw_from_tile(
            projection,
            *coord,
            Arc::clone(tile_data),
            coarse_blend,
            filtering_mode,
        ) {
            draws.push(draw);
        }
    }

    draws
}

fn coarse_blend_for_tile(
    tile_manager: &TileManager,
    coarse_tiles: &HashMap<common::TileCoord, Arc<common::TileData>>,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine_coord: common::TileCoord,
    fine_tile: &Arc<common::TileData>,
) -> Option<CoarseBlendData> {
    const COARSE_BOUNDARY_EPSILON: f64 = 1e-3;

    if trilinear.level_fine == trilinear.level_coarse || trilinear.blend <= 0.01 {
        return None;
    }

    let coarse_info = tile_manager.wsi().level(trilinear.level_coarse)?;
    let image_x = fine_coord.x as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let image_y = fine_coord.y as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let fine_image_w = fine_tile.width as f64 * fine_downsample;
    let fine_image_h = fine_tile.height as f64 * fine_downsample;
    let coarse_tile_size = tile_manager.tile_size_for_level(trilinear.level_coarse) as f64;
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

    let coarse_tile = coarse_tiles.get(&coarse_coord)?.clone();
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

#[cfg(test)]
mod tests {
    use super::{TRILINEAR_LOD_BIAS, finalize_trilinear_levels};

    #[test]
    fn trilinear_bias_shifts_lod_toward_finer_mips() {
        let unbiased = finalize_trilinear_levels(6, 2.5, false);
        let biased = finalize_trilinear_levels(6, 2.5, true);

        assert_eq!(unbiased.level_fine, 2);
        assert_eq!(unbiased.level_coarse, 3);
        assert!((unbiased.blend - 0.5).abs() < 1e-6);

        assert_eq!(biased.level_fine, 2);
        assert_eq!(biased.level_coarse, 3);
        assert!((biased.lod_after_bias - (2.5 + TRILINEAR_LOD_BIAS)).abs() < 1e-6);
        assert!((biased.blend - 0.25).abs() < 1e-6);
    }

    #[test]
    fn trilinear_bias_clamps_at_finest_level() {
        let biased = finalize_trilinear_levels(6, 0.1, true);

        assert_eq!(biased.level_fine, 0);
        assert_eq!(biased.level_coarse, 0);
        assert_eq!(biased.blend, 0.0);
        assert_eq!(biased.lod_after_bias, 0.0);
    }

    #[test]
    fn trilinear_bias_clamps_at_coarsest_level() {
        let biased = finalize_trilinear_levels(6, 5.5, true);

        assert_eq!(biased.level_fine, 5);
        assert_eq!(biased.level_coarse, 5);
        assert_eq!(biased.blend, 0.0);
        assert_eq!(biased.lod_after_bias, 5.0);
    }
}
