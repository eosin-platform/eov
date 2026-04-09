//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.

use common::{TileCoord, Viewport, WsiFile};

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
        return TrilinearLevels { level_fine: 0, level_coarse: 0, blend: 0.0 };
    }
    
    if level_count == 1 {
        return TrilinearLevels { level_fine: 0, level_coarse: 0, blend: 0.0 };
    }
    
    // Find the best level (where pixel density is closest to 1:1)
    let best_level = wsi.best_level_for_downsample(target_downsample);
    
    let best_info = match wsi.level(best_level) {
        Some(info) => info,
        None => return TrilinearLevels { level_fine: 0, level_coarse: 0, blend: 0.0 },
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
            return TrilinearLevels { level_fine: best_level, level_coarse: best_level, blend: 0.0 };
        }
    } else {
        // Blend between previous finer level and best (coarse)
        if best_level > 0 {
            (best_level - 1, best_level)
        } else {
            // At finest level, no blending
            return TrilinearLevels { level_fine: 0, level_coarse: 0, blend: 0.0 };
        }
    };
    
    // Calculate blend factor using log space for perceptually linear transitions
    let fine_info = wsi.level(level_fine);
    let coarse_info = wsi.level(level_coarse);
    
    let (fine_ds, coarse_ds) = match (fine_info, coarse_info) {
        (Some(f), Some(c)) => (f.downsample, c.downsample),
        _ => return TrilinearLevels { level_fine, level_coarse, blend: 0.0 },
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
    
    TrilinearLevels { level_fine, level_coarse, blend }
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
pub fn bilinear_sample(
    data: &[u8],
    width: u32,
    height: u32,
    x: f64,
    y: f64,
) -> [u8; 4] {
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
