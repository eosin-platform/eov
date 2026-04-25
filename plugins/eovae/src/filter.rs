use crate::analysis::{TileCacheEntry, start_viewport_analysis, should_render_overlay};
use crate::state::{
    VisualizationMode, host_api, mark_auto_update, plugin_state, should_auto_update,
};
use std::time::Duration;

pub fn apply_overlay(rgba_data: &mut [u8], width: u32, height: u32) -> bool {
    let (mode, entries, hovered_entry, pulse_entry, pulse_started_at) = {
        let state = plugin_state().lock().unwrap();
        let mode = state.visualization_mode;
        let entries = state
            .cache
            .values()
            .filter(|entry| entry.namespace == state.cache_namespace)
            .cloned()
            .collect::<Vec<_>>();
        let hovered_entry = state
            .hovered_region_id
            .as_deref()
            .and_then(|id| state.cache.get(id))
            .filter(|entry| entry.namespace == state.cache_namespace)
            .cloned();
        let pulse_entry = state
            .pulsing_region_id
            .as_deref()
            .and_then(|id| state.cache.get(id))
            .filter(|entry| entry.namespace == state.cache_namespace)
            .cloned();
        (
            mode,
            entries,
            hovered_entry,
            pulse_entry,
            state.pulsing_region_started_at,
        )
    };

    let has_highlight = hovered_entry.is_some() || pulse_entry.is_some();

    if (!should_render_overlay(mode) || entries.is_empty()) && !has_highlight {
        maybe_schedule_auto_viewport_analysis();
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

    if let Some(entry) = hovered_entry.filter(|entry| intersects_viewport(entry, &viewport)) {
        draw_highlight_outline(
            rgba_data,
            width,
            height,
            &viewport,
            &entry,
            (0xF1, 0xC4, 0x0F),
            2,
        );
    }

    if let Some(entry) = pulse_entry.filter(|entry| intersects_viewport(entry, &viewport)) {
        let elapsed = pulse_started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or(Duration::ZERO);
        if elapsed >= Duration::from_millis(1300) {
            let mut state = plugin_state().lock().unwrap();
            state.pulsing_region_id = None;
            state.pulsing_region_started_at = None;
        } else {
            let phase = (elapsed.as_secs_f32() / 1.3) * std::f32::consts::TAU * 2.0;
            let pulse = ((phase.sin() + 1.0) * 0.5).clamp(0.0, 1.0);
            let color = (
                (0xF1 as f32 * pulse).round() as u8,
                (0xC4 as f32 * pulse).round() as u8,
                (0x0F as f32 * pulse).round() as u8,
            );
            draw_highlight_outline(rgba_data, width, height, &viewport, &entry, color, 3);
        }
    }

    true
}

fn maybe_schedule_auto_viewport_analysis() {
    let host = match host_api() {
        Some(host) => host,
        None => return,
    };
    let snapshot = (host.get_snapshot)(host.context);
    let viewport = match snapshot.active_viewport.into_option() {
        Some(viewport) => viewport,
        None => return,
    };
    let viewport_key = format!(
        "{:.3}:{:.3}:{:.3}:{:.3}:mip{}",
        viewport.bounds_left,
        viewport.bounds_top,
        viewport.bounds_right,
        viewport.bounds_bottom,
        {
            let state = plugin_state().lock().unwrap();
            state.config.mip_level
        },
    );
    if !should_auto_update(&viewport_key) {
        return;
    }
    let (model, namespace, mip_level) = {
        let state = plugin_state().lock().unwrap();
        (state.model.clone(), state.cache_namespace.clone(), state.config.mip_level)
    };
    if let Some(model) = model {
        mark_auto_update(viewport_key);
        start_viewport_analysis(model, viewport, namespace, mip_level);
    }
}

fn intersects_viewport(entry: &TileCacheEntry, viewport: &plugin_api::ffi::ViewportSnapshotFFI) -> bool {
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
    let sx0 = (((tile.x as f64 - viewport.bounds_left) / view_width) * frame_width as f64).floor() as i32;
    let sy0 = (((tile.y as f64 - viewport.bounds_top) / view_height) * frame_height as f64).floor() as i32;
    let sx1 = ((((tile.x + tile.width as u64) as f64 - viewport.bounds_left) / view_width) * frame_width as f64).ceil() as i32;
    let sy1 = ((((tile.y + tile.height as u64) as f64 - viewport.bounds_top) / view_height) * frame_height as f64).ceil() as i32;
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

fn draw_highlight_outline(
    rgba_data: &mut [u8],
    frame_width: u32,
    frame_height: u32,
    viewport: &plugin_api::ffi::ViewportSnapshotFFI,
    entry: &TileCacheEntry,
    color: (u8, u8, u8),
    thickness: i32,
) {
    let view_width = (viewport.bounds_right - viewport.bounds_left).max(1.0);
    let view_height = (viewport.bounds_bottom - viewport.bounds_top).max(1.0);
    let tile = &entry.tile;
    let sx0 = (((tile.x as f64 - viewport.bounds_left) / view_width) * frame_width as f64).floor() as i32;
    let sy0 = (((tile.y as f64 - viewport.bounds_top) / view_height) * frame_height as f64).floor() as i32;
    let sx1 = ((((tile.x + tile.width as u64) as f64 - viewport.bounds_left) / view_width) * frame_width as f64).ceil() as i32;
    let sy1 = ((((tile.y + tile.height as u64) as f64 - viewport.bounds_top) / view_height) * frame_height as f64).ceil() as i32;

    if sx1 <= sx0 || sy1 <= sy0 {
        return;
    }

    for offset in 0..thickness.max(1) {
        let left = (sx0 - offset).max(0);
        let right = (sx1 + offset - 1).min(frame_width as i32 - 1);
        let top = (sy0 - offset).max(0);
        let bottom = (sy1 + offset - 1).min(frame_height as i32 - 1);

        for sx in left..=right {
            paint_pixel(rgba_data, frame_width, frame_height, sx, top, color);
            paint_pixel(rgba_data, frame_width, frame_height, sx, bottom, color);
        }
        for sy in top..=bottom {
            paint_pixel(rgba_data, frame_width, frame_height, left, sy, color);
            paint_pixel(rgba_data, frame_width, frame_height, right, sy, color);
        }
    }
}

fn paint_pixel(
    rgba_data: &mut [u8],
    frame_width: u32,
    frame_height: u32,
    x: i32,
    y: i32,
    color: (u8, u8, u8),
) {
    if x < 0 || y < 0 || x >= frame_width as i32 || y >= frame_height as i32 {
        return;
    }
    let index = (y as usize * frame_width as usize + x as usize) * 4;
    rgba_data[index] = color.0;
    rgba_data[index + 1] = color.1;
    rgba_data[index + 2] = color.2;
}