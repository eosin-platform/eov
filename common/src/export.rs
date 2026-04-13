//! CPU-only export renderer for WSI viewports.
//!
//! Produces the highest quality RGBA image for a given viewport at an
//! arbitrary DPI. Uses Lanczos-3 filtering by default, with full stain
//! normalization and post-processing support.

use std::collections::HashMap;
use std::sync::Arc;

use crate::blitter::{self, BlitRect, CoarseSrc, TileSrc};
use crate::postprocess;
use crate::render::{FilteringMode, TrilinearLevels, calculate_trilinear_levels};
use crate::stain;
use crate::{
    RgbaImageData, StainNormalization, TileCache, TileCoord, TileData, TileManager, Viewport,
};

/// Export-specific settings that control how the image is rendered.
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Output DPI (96 = screen resolution).
    pub dpi: u32,
    /// Filtering mode for tile resampling.
    pub filtering_mode: FilteringMode,
    /// Stain normalization method.
    pub stain_normalization: StainNormalization,
    /// Sharpening amount (0.0 = off).
    pub sharpness: f32,
    /// Gamma correction (1.0 = identity).
    pub gamma: f32,
    /// Brightness offset (0.0 = identity).
    pub brightness: f32,
    /// Contrast multiplier (1.0 = identity).
    pub contrast: f32,
    /// Background colour (RGBA).
    pub background_rgba: [u8; 4],
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            dpi: 96,
            filtering_mode: FilteringMode::Lanczos3,
            stain_normalization: StainNormalization::None,
            sharpness: 0.0,
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
            background_rgba: [255, 255, 255, 255],
        }
    }
}

/// Construct the viewport used for export rendering at the given DPI.
///
/// This is useful for converting image coordinates to export-buffer
/// coordinates when drawing overlays.
pub fn export_viewport(viewport: &Viewport, dpi: u32) -> Viewport {
    let scale = dpi.max(1) as f64 / 96.0;
    let export_width = (viewport.width * scale).round().max(1.0);
    let export_height = (viewport.height * scale).round().max(1.0);
    let export_zoom = viewport.zoom * scale;
    Viewport {
        center: viewport.center,
        zoom: export_zoom,
        width: export_width,
        height: export_height,
        image_width: viewport.image_width,
        image_height: viewport.image_height,
    }
}

/// Render the current viewport at the requested DPI, applying all export
/// settings. Returns `None` if the viewport or WSI is in an invalid state.
///
/// This function is CPU-only and intended for off-screen export. It
/// loads any missing tiles synchronously before rendering.
pub fn render_export(
    tile_manager: &TileManager,
    tile_cache: &TileCache,
    viewport: &Viewport,
    settings: &ExportSettings,
) -> Option<RgbaImageData> {
    let wsi = tile_manager.wsi();
    if wsi.level_count() == 0 || viewport.width <= 0.0 || viewport.height <= 0.0 {
        return None;
    }

    // ── Compute output dimensions ──
    let scale = settings.dpi.max(1) as f64 / 96.0;
    let export_width = (viewport.width * scale).round().max(1.0) as u32;
    let export_height = (viewport.height * scale).round().max(1.0) as u32;
    let export_zoom = viewport.zoom * scale;

    // Build an export viewport that covers the same image area but at
    // the higher DPI pixel density.
    let export_vp = Viewport {
        center: viewport.center,
        zoom: export_zoom,
        width: export_width as f64,
        height: export_height as f64,
        image_width: viewport.image_width,
        image_height: viewport.image_height,
    };

    let target_downsample = 1.0 / export_zoom;
    let apply_lod_bias = settings.filtering_mode == FilteringMode::Trilinear;
    let trilinear = calculate_trilinear_levels(wsi, target_downsample, apply_lod_bias);
    let level = trilinear.level_fine;
    let level_info = wsi.level(level)?.clone();
    let bounds = export_vp.bounds();

    // ── Collect/load tiles ──
    let fine_coords = tile_manager.visible_tiles_with_margin(
        level,
        bounds.left,
        bounds.top,
        bounds.right,
        bounds.bottom,
        0,
    );
    let fine_tiles = load_or_peek_tiles(tile_manager, tile_cache, &fine_coords);

    let use_trilinear = settings.filtering_mode == FilteringMode::Trilinear
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01;

    let coarse_tiles: HashMap<TileCoord, Arc<TileData>> = if use_trilinear {
        let coarse_coords = tile_manager.visible_tiles_with_margin(
            trilinear.level_coarse,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            0,
        );
        load_or_peek_tiles(tile_manager, tile_cache, &coarse_coords)
            .into_iter()
            .collect()
    } else {
        HashMap::new()
    };

    // ── Build blit commands ──
    let buf_len = export_width as usize * export_height as usize * 4;
    let mut buffer = vec![0u8; buf_len];
    blitter::fast_fill_rgba(
        &mut buffer,
        settings.background_rgba[0],
        settings.background_rgba[1],
        settings.background_rgba[2],
        settings.background_rgba[3],
    );

    // Fallback blits from coarser pyramid levels
    let level_count = wsi.level_count();
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= level {
            continue;
        }
        let Some(fb_info) = wsi.level(fallback_level) else {
            continue;
        };
        let fb_coords = tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );
        for (_, tile) in load_or_peek_tiles(tile_manager, tile_cache, &fb_coords) {
            let coord = tile.coord;
            let rect = tile_screen_rect(&export_vp, &bounds, fb_info.downsample, &coord, &tile);
            if rect.width <= 0 || rect.height <= 0 {
                continue;
            }
            blit_command(
                &mut buffer,
                export_width,
                export_height,
                &tile,
                rect,
                &FilteringMode::Bilinear,
                None,
            );
        }
    }

    // Fine blits
    for (coord, tile) in &fine_tiles {
        let rect = tile_screen_rect(&export_vp, &bounds, level_info.downsample, coord, tile);
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }

        let coarse_blend = if use_trilinear {
            compute_coarse_blend(
                tile_manager,
                &coarse_tiles,
                trilinear,
                level_info.downsample,
                *coord,
                tile,
            )
        } else {
            None
        };

        blit_command(
            &mut buffer,
            export_width,
            export_height,
            tile,
            rect,
            &settings.filtering_mode,
            coarse_blend.as_ref(),
        );
    }

    // ── Post-processing ──
    // Stain normalization
    if settings.stain_normalization != StainNormalization::None {
        let tile_slices: Vec<&[u8]> = fine_tiles
            .iter()
            .map(|(_, td)| td.data.as_slice())
            .collect();
        let params = stain::compute_cpu_stain_params(settings.stain_normalization, &tile_slices);
        stain::apply_stain_params_to_buffer(&mut buffer, &params);
    }

    // Sharpening
    if settings.sharpness > 0.001 {
        postprocess::apply_sharpening(&mut buffer, export_width, export_height, settings.sharpness);
    }

    // Gamma / brightness / contrast
    let has_adjustments = (settings.gamma - 1.0).abs() > 0.001
        || settings.brightness.abs() > 0.001
        || (settings.contrast - 1.0).abs() > 0.001;
    if has_adjustments {
        postprocess::apply_adjustments(
            &mut buffer,
            settings.gamma,
            settings.brightness,
            settings.contrast,
        );
    }

    Some(RgbaImageData {
        width: export_width as usize,
        height: export_height as usize,
        pixels: buffer,
    })
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Load tiles from cache, or synchronously from disk if missing.
fn load_or_peek_tiles(
    tile_manager: &TileManager,
    tile_cache: &TileCache,
    coords: &[TileCoord],
) -> Vec<(TileCoord, Arc<TileData>)> {
    coords
        .iter()
        .filter_map(|coord| {
            if let Some(cached) = tile_cache.peek(coord) {
                return Some((*coord, cached));
            }
            // Synchronous fallback: load tile from the WSI file directly.
            match tile_manager.load_tile_sync(*coord) {
                Ok(tile) => {
                    tile_cache.insert(tile);
                    tile_cache.peek(coord).map(|arc| (*coord, arc))
                }
                Err(_) => None,
            }
        })
        .collect()
}

/// Compute the screen-space blit rect for a tile.
fn tile_screen_rect(
    vp: &Viewport,
    bounds: &crate::viewport::ViewportBounds,
    downsample: f64,
    coord: &TileCoord,
    tile: &TileData,
) -> BlitRect {
    let origin_x = coord.x as f64 * coord.tile_size as f64;
    let origin_y = coord.y as f64 * coord.tile_size as f64;
    let image_x = origin_x * downsample;
    let image_y = origin_y * downsample;
    let image_x_end = (origin_x + tile.width as f64) * downsample;
    let image_y_end = (origin_y + tile.height as f64) * downsample;
    let exact_sx = (image_x - bounds.left) * vp.zoom;
    let exact_sy = (image_y - bounds.top) * vp.zoom;
    let exact_sx_end = (image_x_end - bounds.left) * vp.zoom;
    let exact_sy_end = (image_y_end - bounds.top) * vp.zoom;
    let screen_x = exact_sx.floor() as i32;
    let screen_y = exact_sy.floor() as i32;
    let screen_x_end = exact_sx_end.floor() as i32;
    let screen_y_end = exact_sy_end.floor() as i32;
    BlitRect {
        x: screen_x,
        y: screen_y,
        width: screen_x_end - screen_x,
        height: screen_y_end - screen_y,
        exact_x: exact_sx,
        exact_y: exact_sy,
        exact_width: exact_sx_end - exact_sx,
        exact_height: exact_sy_end - exact_sy,
    }
}

/// Dispatch a single blit command.
fn blit_command(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    tile: &TileData,
    rect: BlitRect,
    filtering: &FilteringMode,
    coarse: Option<&CoarseBlendInfo>,
) {
    let src = TileSrc {
        data: &tile.data,
        width: tile.width,
        height: tile.height,
        border: tile.border,
    };

    match filtering {
        FilteringMode::Lanczos3 => {
            blitter::blit_tile_lanczos3(buffer, width, height, src, rect);
        }
        FilteringMode::Trilinear => {
            if let Some(coarse) = coarse {
                let csrc = CoarseSrc {
                    data: &coarse.tile.data,
                    width: coarse.tile.width,
                    height: coarse.tile.height,
                    border: coarse.tile.border,
                    uv_min: coarse.uv_min,
                    uv_max: coarse.uv_max,
                    blend: coarse.blend,
                };
                blitter::blit_tile_trilinear(buffer, width, height, src, &csrc, rect);
            } else {
                blitter::blit_tile(buffer, width, height, src, rect);
            }
        }
        FilteringMode::Bilinear => {
            blitter::blit_tile(buffer, width, height, src, rect);
        }
    }
}

struct CoarseBlendInfo {
    tile: Arc<TileData>,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    blend: f32,
}

/// Determine the coarse tile and UV sub-region that covers a fine tile
/// for trilinear blending.
fn compute_coarse_blend(
    tile_manager: &TileManager,
    coarse_tiles: &HashMap<TileCoord, Arc<TileData>>,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine_coord: TileCoord,
    fine_tile: &TileData,
) -> Option<CoarseBlendInfo> {
    const EPSILON: f64 = 1e-3;

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
    let coarse_tile_x_end = ((image_x_end - EPSILON) / coarse_info.downsample).max(coarse_tile_x);
    let coarse_tile_y_end = ((image_y_end - EPSILON) / coarse_info.downsample).max(coarse_tile_y);
    let coarse_start_tile_x = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_start_tile_y = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_x = (coarse_tile_x_end / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_y = (coarse_tile_y_end / coarse_tile_size).floor().max(0.0) as u64;

    if coarse_start_tile_x != coarse_end_tile_x || coarse_start_tile_y != coarse_end_tile_y {
        return None;
    }

    let coarse_coord = TileCoord::new(
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
        || coarse_src_x_end > coarse_tile.width as f64 + EPSILON
        || coarse_src_y_end > coarse_tile.height as f64 + EPSILON
    {
        return None;
    }

    let cw = coarse_tile.width as f64;
    let ch = coarse_tile.height as f64;

    Some(CoarseBlendInfo {
        tile: coarse_tile,
        uv_min: [(coarse_src_x / cw) as f32, (coarse_src_y / ch) as f32],
        uv_max: [
            (coarse_src_x_end / cw) as f32,
            (coarse_src_y_end / ch) as f32,
        ],
        blend: trilinear.blend as f32,
    })
}
