use crate::analysis::{TileCacheEntry, should_render_overlay};
use crate::state::{VisualizationMode, host_api, plugin_state};

pub fn apply_overlay(rgba_data: &mut [u8], width: u32, height: u32) -> bool {
    let (mode, entries) = {
        let state = plugin_state().lock().unwrap();
        let mode = state.visualization_mode;
        let entries = state
            .cache
            .values()
            .filter(|entry| entry.namespace == state.cache_namespace)
            .cloned()
            .collect::<Vec<_>>();
        (mode, entries)
    };

    if !should_render_overlay(mode) || entries.is_empty() {
        return false;
    }

    let host = match host_api() {
        Some(host) => host,
        None => return false,
    };
    let snapshot = (host.get_snapshot)(host.context);
    let viewport = match snapshot.active_viewport.into_option() {
        Some(viewport) => viewport,
        None => return false,
    };

    if should_render_overlay(mode) {
        for entry in entries {
            if !intersects_viewport(&entry, &viewport) {
                continue;
            }
            composite_tile(rgba_data, width, height, &viewport, &entry, mode);
        }
    }

    true
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
) {
    let view_width = (viewport.bounds_right - viewport.bounds_left).max(1.0);
    let view_height = (viewport.bounds_bottom - viewport.bounds_top).max(1.0);
    let tile = &entry.tile;
    let has_rgb_payload = tile.reconstruction_rgb.len()
        >= tile.sample_width as usize * tile.sample_height as usize * 3
        && tile.difference_rgb.len()
            >= tile.sample_width as usize * tile.sample_height as usize * 3;
    let has_luma_payload =
        tile.error_map_luma.len() >= tile.sample_width as usize * tile.sample_height as usize;

    if matches!(mode, VisualizationMode::Reconstruction | VisualizationMode::Difference)
        && !has_rgb_payload
    {
        return;
    }
    if mode == VisualizationMode::ErrorMap && !has_luma_payload {
        return;
    }

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
                    let heat = tile.error_map_luma[tile_luma_index];
                    let (red, green, blue) = heatmap_color(heat);
                    rgba_data[frame_index] = blend(rgba_data[frame_index], red, 180);
                    rgba_data[frame_index + 1] = blend(rgba_data[frame_index + 1], green, 180);
                    rgba_data[frame_index + 2] = blend(rgba_data[frame_index + 2], blue, 180);
                }
            }
        }
    }
}

fn heatmap_color(value: u8) -> (u8, u8, u8) {
    let ratio = value as f32 / 255.0;
    let red = (255.0 * ratio).round() as u8;
    let green = (255.0 * (1.0 - (ratio - 0.4).abs() * 1.4).clamp(0.0, 1.0)).round() as u8;
    let blue = (255.0 * (1.0 - ratio)).round() as u8;
    (red, green, blue)
}

fn blend(base: u8, overlay: u8, alpha: u8) -> u8 {
    let alpha = alpha as u16;
    let inv = 255u16.saturating_sub(alpha);
    (((base as u16 * inv) + (overlay as u16 * alpha)) / 255) as u8
}
