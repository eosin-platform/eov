use crate::analysis::{start_viewport_analysis, TileCacheEntry, should_render_overlay};
use crate::state::{AnalysisPhase, VisualizationMode, plugin_state};
use std::collections::HashSet;

const MAX_PENDING_PLACEHOLDERS: usize = 512;

pub fn apply_overlay(
    rgba_data: &mut [u8],
    width: u32,
    height: u32,
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
) -> bool {
    let (mode, entries, tile_size, mip_level, namespace, error_p05, error_p95) = {
        let state = plugin_state().lock().unwrap();
        let mode = state
            .pane_visualization_modes
            .get(&viewport.pane_index)
            .copied()
            .unwrap_or(state.visualization_mode);
        let tile_size = state
            .model
            .as_ref()
            .map(|model| model.summary.tile_size)
            .unwrap_or(256);
        let namespace = state
            .model
            .as_ref()
            .map(|model| format!("{}|mip{}", model.summary.identity(), state.config.mip_level))
            .unwrap_or_else(|| state.cache_namespace.clone());
        let entries = state
            .cache
            .values()
            .filter(|entry| entry.namespace == namespace)
            .cloned()
            .collect::<Vec<_>>();
        (
            mode,
            entries,
            tile_size,
            state.config.mip_level,
            namespace,
            state.error_stats.p05,
            state.error_stats.p95,
        )
    };

    if !should_render_overlay(mode) {
        return false;
    }

    let grid = viewport_tile_grid(viewport, tile_size, mip_level);
    let visible_tiles = visible_viewport_tiles(&grid, MAX_PENDING_PLACEHOLDERS);
    let cached_tile_ids = entries
        .iter()
        .filter(|entry| tile_can_render(mode, &entry.tile))
        .map(|entry| entry.tile.id())
        .collect::<HashSet<_>>();
    let visible_cached_count = entries
        .iter()
        .filter(|entry| intersects_viewport(entry, viewport) && tile_can_render(mode, &entry.tile))
        .count();
    let missing_tiles = visible_tiles
        .into_iter()
        .filter(|tile| !cached_tile_ids.contains(&tile.id))
        .collect::<Vec<_>>();

    let needs_analysis = visible_cached_count < grid.total_tiles;
    maybe_start_viewport_analysis(viewport, needs_analysis, &grid, &namespace, mip_level);

    let mut applied = false;

    for entry in entries {
        if !intersects_viewport(&entry, viewport) {
            continue;
        }
        composite_tile(
            rgba_data,
            width,
            height,
            viewport,
            &entry,
            mode,
            error_p05,
            error_p95,
        );
        applied = true;
    }

    for tile in &missing_tiles {
        paint_pending_tile(rgba_data, width, height, viewport, tile);
        applied = true;
    }

    applied
}

#[derive(Clone, Debug)]
struct VisibleTile {
    id: String,
    x: u64,
    y: u64,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug)]
struct ViewportTileGrid {
    start_x: u64,
    start_y: u64,
    end_x: u64,
    end_y: u64,
    step: u64,
    total_tiles: usize,
}

fn intersects_viewport(
    entry: &TileCacheEntry,
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
) -> bool {
    let right = entry.tile.x as f64 + entry.tile.width as f64;
    let bottom = entry.tile.y as f64 + entry.tile.height as f64;
    !(right < viewport.bounds_left
        || bottom < viewport.bounds_top
        || entry.tile.x as f64 > viewport.bounds_right
        || entry.tile.y as f64 > viewport.bounds_bottom)
}

fn composite_tile(
    rgba_data: &mut [u8],
    frame_width: u32,
    frame_height: u32,
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
    entry: &TileCacheEntry,
    mode: VisualizationMode,
    error_p05: f64,
    error_p95: f64,
) {
    let tile = &entry.tile;
    if !tile_can_render(mode, tile) {
        return;
    }

    let view_width = (viewport.bounds_right - viewport.bounds_left).max(1.0);
    let view_height = (viewport.bounds_bottom - viewport.bounds_top).max(1.0);

    let sx0 =
        (((tile.x as f64 - viewport.bounds_left) / view_width) * frame_width as f64).floor() as i32;
    let sy0 = (((tile.y as f64 - viewport.bounds_top) / view_height) * frame_height as f64).floor()
        as i32;
    let sx1 = ((((tile.x + tile.width as u64) as f64 - viewport.bounds_left) / view_width)
        * frame_width as f64)
        .ceil() as i32;
    let sy1 = ((((tile.y + tile.height as u64) as f64 - viewport.bounds_top) / view_height)
        * frame_height as f64)
        .ceil() as i32;
    let sample_width = tile.sample_width.max(1);
    let sample_height = tile.sample_height.max(1);

    for sy in sy0.max(0)..sy1.min(frame_height as i32) {
        for sx in sx0.max(0)..sx1.min(frame_width as i32) {
            let tx = (((sx - sx0).max(0) as f64 / (sx1 - sx0).max(1) as f64) * sample_width as f64)
                .floor()
                .clamp(0.0, (sample_width.saturating_sub(1)) as f64) as usize;
            let ty = (((sy - sy0).max(0) as f64 / (sy1 - sy0).max(1) as f64) * sample_height as f64)
                .floor()
                .clamp(0.0, (sample_height.saturating_sub(1)) as f64) as usize;
            let frame_index = (sy as usize * frame_width as usize + sx as usize) * 4;
            let tile_rgb_index = (ty * sample_width as usize + tx) * 3;
            let tile_luma_index = ty * sample_width as usize + tx;

            match mode {
                VisualizationMode::Original => {}
                VisualizationMode::Reconstruction => {
                    rgba_data[frame_index] = tile.reconstruction_rgb[tile_rgb_index];
                    rgba_data[frame_index + 1] = tile.reconstruction_rgb[tile_rgb_index + 1];
                    rgba_data[frame_index + 2] = tile.reconstruction_rgb[tile_rgb_index + 2];
                }
                VisualizationMode::Difference => {
                    rgba_data[frame_index] = tile.difference_rgb[tile_rgb_index];
                    rgba_data[frame_index + 1] = tile.difference_rgb[tile_rgb_index + 1];
                    rgba_data[frame_index + 2] = tile.difference_rgb[tile_rgb_index + 2];
                }
                VisualizationMode::ErrorMap => {
                    let _ = tile_luma_index;
                    let (red, green, blue) = tile_error_color(tile.mean_absolute_error, error_p05, error_p95);
                    rgba_data[frame_index] = blend(rgba_data[frame_index], red, 180);
                    rgba_data[frame_index + 1] = blend(rgba_data[frame_index + 1], green, 180);
                    rgba_data[frame_index + 2] = blend(rgba_data[frame_index + 2], blue, 180);
                }
            }
        }
    }
}

fn tile_can_render(mode: VisualizationMode, tile: &crate::analysis::AnalyzedTile) -> bool {
    if tile.sample_width == 0 || tile.sample_height == 0 {
        return false;
    }
    let rgb_len = tile.sample_width as usize * tile.sample_height as usize * 3;
    match mode {
        VisualizationMode::Original => false,
        VisualizationMode::Reconstruction => rgb_len > 0 && tile.reconstruction_rgb.len() >= rgb_len,
        VisualizationMode::Difference => rgb_len > 0 && tile.difference_rgb.len() >= rgb_len,
        VisualizationMode::ErrorMap => true,
    }
}

fn tile_error_color(value: f64, p05: f64, p95: f64) -> (u8, u8, u8) {
    let span = (p95 - p05).max(1e-6);
    let ratio = ((value - p05) / span).clamp(0.0, 1.0) as f32;
    if ratio <= 0.5 {
        let t = ratio / 0.5;
        lerp_color((0x39, 0xB5, 0x4A), (0xF2, 0xD5, 0x4A), t)
    } else {
        let t = (ratio - 0.5) / 0.5;
        lerp_color((0xF2, 0xD5, 0x4A), (0xD9, 0x43, 0x43), t)
    }
}

fn lerp_color(start: (u8, u8, u8), end: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let blend = |left: u8, right: u8| -> u8 {
        (left as f32 + (right as f32 - left as f32) * t.clamp(0.0, 1.0)).round() as u8
    };
    (
        blend(start.0, end.0),
        blend(start.1, end.1),
        blend(start.2, end.2),
    )
}

fn blend(base: u8, overlay: u8, alpha: u8) -> u8 {
    let alpha = alpha as u16;
    let inv = 255u16.saturating_sub(alpha);
    (((base as u16 * inv) + (overlay as u16 * alpha)) / 255) as u8
}

fn viewport_tile_grid(
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
    tile_size: u32,
    mip_level: u32,
) -> ViewportTileGrid {
    let downsample = 1u64 << mip_level.min(3);
    let step = tile_size as u64 * downsample;
    let right = viewport.bounds_right.max(0.0) as u64;
    let bottom = viewport.bounds_bottom.max(0.0) as u64;
    let image_width = viewport.image_width.max(0.0) as u64;
    let image_height = viewport.image_height.max(0.0) as u64;
    let start_x = ((viewport.bounds_left.max(0.0) as u64) / step) * step;
    let start_y = ((viewport.bounds_top.max(0.0) as u64) / step) * step;
    let end_x = right.min(image_width);
    let end_y = bottom.min(image_height);
    let columns = if end_x > start_x {
        ((end_x - start_x) / step) as usize + usize::from((end_x - start_x) % step != 0)
    } else {
        0
    };
    let rows = if end_y > start_y {
        ((end_y - start_y) / step) as usize + usize::from((end_y - start_y) % step != 0)
    } else {
        0
    };

    ViewportTileGrid {
        start_x,
        start_y,
        end_x,
        end_y,
        step,
        total_tiles: rows.saturating_mul(columns),
    }
}

fn visible_viewport_tiles(grid: &ViewportTileGrid, limit: usize) -> Vec<VisibleTile> {
    let mut tiles = Vec::new();
    let mut y = grid.start_y;
    while y < grid.end_y && tiles.len() < limit {
        let mut x = grid.start_x;
        while x < grid.end_x && tiles.len() < limit {
            let width = grid.step as u32;
            let height = grid.step as u32;
            tiles.push(VisibleTile {
                id: format!("{}:{}:{}:{}", x, y, width, height),
                x,
                y,
                width,
                height,
            });
            x += grid.step;
        }
        y += grid.step;
    }
    tiles
}

fn maybe_start_viewport_analysis(
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
    needs_analysis: bool,
    grid: &ViewportTileGrid,
    namespace: &str,
    mip_level: u32,
) {
    let request_key = if !needs_analysis || grid.total_tiles == 0 {
        None
    } else {
        Some(format!(
            "{}:{}:{}:{}:{}:{}:{}",
            namespace, grid.start_x, grid.start_y, grid.end_x, grid.end_y, grid.step, grid.total_tiles
        ))
    };

    let maybe_request = {
        let mut state = plugin_state().lock().unwrap();
        if request_key.is_none() {
            state.pane_auto_viewport_request_keys.remove(&viewport.pane_index);
            return;
        }
        if state.analysis_phase == AnalysisPhase::Running
            || state
                .pane_auto_viewport_request_keys
                .get(&viewport.pane_index)
                .map(String::as_str)
                == request_key.as_deref()
        {
            return;
        }
        let Some(model) = state.model.clone() else {
            return;
        };
        if let Some(request_key) = request_key {
            state
                .pane_auto_viewport_request_keys
                .insert(viewport.pane_index, request_key);
        }
        Some(model)
    };

    if let Some(model) = maybe_request {
        start_viewport_analysis(model, viewport.clone(), namespace.to_string(), mip_level);
    }
}

fn paint_pending_tile(
    rgba_data: &mut [u8],
    frame_width: u32,
    frame_height: u32,
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
    tile: &VisibleTile,
) {
    let view_width = (viewport.bounds_right - viewport.bounds_left).max(1.0);
    let view_height = (viewport.bounds_bottom - viewport.bounds_top).max(1.0);
    let sx0 = (((tile.x as f64 - viewport.bounds_left) / view_width) * frame_width as f64).floor()
        as i32;
    let sy0 = (((tile.y as f64 - viewport.bounds_top) / view_height) * frame_height as f64).floor()
        as i32;
    let sx1 = ((((tile.x + tile.width as u64) as f64 - viewport.bounds_left) / view_width)
        * frame_width as f64)
        .ceil() as i32;
    let sy1 = ((((tile.y + tile.height as u64) as f64 - viewport.bounds_top) / view_height)
        * frame_height as f64)
        .ceil() as i32;

    for sy in sy0.max(0)..sy1.min(frame_height as i32) {
        for sx in sx0.max(0)..sx1.min(frame_width as i32) {
            let frame_index = (sy as usize * frame_width as usize + sx as usize) * 4;
            let local_x = sx - sx0.max(0);
            let local_y = sy - sy0.max(0);
            let stripe = (((local_x + local_y) / 8) % 2) == 0;
            let chevron = (((local_x - local_y).abs() / 12) % 2) == 0;
            let alpha = if stripe ^ chevron { 128 } else { 56 };
            rgba_data[frame_index] = blend(rgba_data[frame_index], 0xF2, alpha);
            rgba_data[frame_index + 1] = blend(rgba_data[frame_index + 1], 0xD5, alpha);
            rgba_data[frame_index + 2] = blend(rgba_data[frame_index + 2], 0x4A, alpha);
        }
    }
}
