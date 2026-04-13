//! Tile management for WSI rendering
//!
//! This module handles tile coordinates, tile requests, and tile data management
//! for efficient rendering of whole slide images.

use crate::{Error, Result, WsiFile};
use tracing::trace;

/// Build a uniformly-bordered RGBA buffer from an expanded-region read.
///
/// The read data has variable border amounts on each side (limited by image
/// bounds).  This function copies it into a buffer that always has exactly
/// `border` pixels on every side, edge-clamping where the image boundary
/// prevented reading real neighbor data.
fn create_bordered_tile(
    read_data: &[u8],
    read_w: u32,
    read_h: u32,
    inner_w: u32,
    inner_h: u32,
    border: u32,
    avail_left: u32,
    avail_top: u32,
) -> Vec<u8> {
    let out_w = (inner_w + 2 * border) as usize;
    let out_h = (inner_h + 2 * border) as usize;
    let dst_stride = out_w * 4;
    let src_stride = read_w as usize * 4;
    let mut out = vec![0u8; out_h * dst_stride];

    // Offset in the output where the read data is placed.
    let dx = (border - avail_left) as usize;
    let dy = (border - avail_top) as usize;

    // Step 1: copy the read data into the output buffer.
    for y in 0..read_h as usize {
        let src_off = y * src_stride;
        let dst_off = (dy + y) * dst_stride + dx * 4;
        out[dst_off..dst_off + src_stride]
            .copy_from_slice(&read_data[src_off..src_off + src_stride]);
    }

    // Step 2: fill missing left-border columns by replicating the first
    // available column in each row that has data.
    if dx > 0 {
        for iy in dy..dy + read_h as usize {
            let src_off = iy * dst_stride + dx * 4;
            let pixel = [
                out[src_off],
                out[src_off + 1],
                out[src_off + 2],
                out[src_off + 3],
            ];
            for ix in 0..dx {
                let off = iy * dst_stride + ix * 4;
                out[off..off + 4].copy_from_slice(&pixel);
            }
        }
    }

    // Step 3: fill missing right-border columns.
    let right_data_end = dx + read_w as usize;
    if right_data_end < out_w {
        for iy in dy..dy + read_h as usize {
            let src_off = iy * dst_stride + (right_data_end - 1) * 4;
            let pixel = [
                out[src_off],
                out[src_off + 1],
                out[src_off + 2],
                out[src_off + 3],
            ];
            for ix in right_data_end..out_w {
                let off = iy * dst_stride + ix * 4;
                out[off..off + 4].copy_from_slice(&pixel);
            }
        }
    }

    // Step 4: fill missing top-border rows (copies full rows including L/R borders).
    if dy > 0 {
        let src_row_start = dy * dst_stride;
        for iy in 0..dy {
            let dst_row_start = iy * dst_stride;
            out.copy_within(src_row_start..src_row_start + dst_stride, dst_row_start);
        }
    }

    // Step 5: fill missing bottom-border rows.
    let bottom_data_end = dy + read_h as usize;
    if bottom_data_end < out_h {
        let src_row_start = (bottom_data_end - 1) * dst_stride;
        for iy in bottom_data_end..out_h {
            let dst_row_start = iy * dst_stride;
            out.copy_within(src_row_start..src_row_start + dst_stride, dst_row_start);
        }
    }

    out
}

/// Tile coordinate identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    /// File identifier (distinguishes tiles from different open files)
    pub file_id: i32,
    /// Level index
    pub level: u32,
    /// X tile index
    pub x: u64,
    /// Y tile index
    pub y: u64,
    /// Tile size used to address this level
    pub tile_size: u32,
}

impl TileCoord {
    pub fn new(file_id: i32, level: u32, x: u64, y: u64, tile_size: u32) -> Self {
        Self {
            file_id,
            level,
            x,
            y,
            tile_size,
        }
    }
}

/// Tile data with RGBA pixels
#[derive(Debug, Clone)]
pub struct TileData {
    /// Tile coordinate
    pub coord: TileCoord,
    /// RGBA pixel data — layout is data_width() × data_height() × 4 bytes.
    pub data: Vec<u8>,
    /// Inner tile width (content, excluding border padding)
    pub width: u32,
    /// Inner tile height (content, excluding border padding)
    pub height: u32,
    /// Border padding in pixels on each side (0 = legacy, 1 = standard).
    /// When border > 0, `data` is `(width + 2*border) × (height + 2*border) × 4` bytes.
    pub border: u32,
}

impl TileData {
    /// Create a new tile data instance
    pub fn new(coord: TileCoord, data: Vec<u8>, width: u32, height: u32, border: u32) -> Self {
        Self {
            coord,
            data,
            width,
            height,
            border,
        }
    }

    /// Full data width including border padding on both sides.
    #[inline]
    pub fn data_width(&self) -> u32 {
        self.width + 2 * self.border
    }

    /// Full data height including border padding on both sides.
    #[inline]
    pub fn data_height(&self) -> u32 {
        self.height + 2 * self.border
    }

    /// Create a placeholder tile (checkerboard pattern for debugging)
    pub fn placeholder(coord: TileCoord, tile_size: u32) -> Self {
        let mut data = vec![0u8; (tile_size * tile_size * 4) as usize];

        // Create checkerboard pattern
        for y in 0..tile_size {
            for x in 0..tile_size {
                let idx = ((y * tile_size + x) * 4) as usize;
                let is_light = ((x / 16) + (y / 16)) % 2 == 0;
                let color = if is_light { 200 } else { 150 };
                data[idx] = color; // R
                data[idx + 1] = color; // G
                data[idx + 2] = color; // B
                data[idx + 3] = 255; // A
            }
        }

        Self {
            coord,
            data,
            width: tile_size,
            height: tile_size,
            border: 0,
        }
    }
}

/// Priority for tile loading
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TilePriority {
    /// Low priority (prefetch)
    Low = 0,
    /// Normal priority (visible)
    Normal = 1,
    /// High priority (center of view)
    High = 2,
}

/// Request to load a tile
#[derive(Debug, Clone)]
pub struct TileRequest {
    pub coord: TileCoord,
    pub priority: TilePriority,
}

/// Tile manager handles tile loading and coordination
pub struct TileManager {
    /// WSI file reference
    wsi: WsiFile,
    /// File identifier (propagated into every TileCoord this manager creates)
    file_id: i32,
    /// Tile size
    tile_size: u32,
}

impl TileManager {
    /// Create a new tile manager for the given WSI file
    pub fn new(wsi: WsiFile, file_id: i32) -> Self {
        let tile_size = wsi.tile_size();
        Self {
            wsi,
            file_id,
            tile_size,
        }
    }

    /// Get the file identifier
    pub fn file_id(&self) -> i32 {
        self.file_id
    }

    /// Get the tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    pub fn tile_size_for_level(&self, level: u32) -> u32 {
        self.wsi.tile_size_for_level(level)
    }

    /// Get the WSI file reference
    pub fn wsi(&self) -> &WsiFile {
        &self.wsi
    }

    /// Load a tile synchronously with a 1-pixel overlap border for seamless
    /// bilinear/Lanczos filtering at tile boundaries.
    pub fn load_tile_sync(&self, coord: TileCoord) -> Result<TileData> {
        let level_info = self
            .wsi
            .level(coord.level)
            .ok_or(Error::InvalidLevel(coord.level, self.wsi.level_count() - 1))?;

        // Inner tile bounds in level coordinates
        let tile_start_x = coord.x * coord.tile_size as u64;
        let tile_start_y = coord.y * coord.tile_size as u64;

        let inner_w =
            (level_info.width.saturating_sub(tile_start_x)).min(coord.tile_size as u64) as u32;
        let inner_h =
            (level_info.height.saturating_sub(tile_start_y)).min(coord.tile_size as u64) as u32;

        if inner_w == 0 || inner_h == 0 {
            return Err(Error::InvalidCoordinates {
                x: coord.x as i64,
                y: coord.y as i64,
                level: coord.level,
            });
        }

        trace!(
            "Loading tile {:?}, inner size: {}x{}",
            coord, inner_w, inner_h
        );

        const BORDER: u32 = 1;

        // Compute how many border pixels are available on each side
        // (clamped to image bounds at this level).
        let avail_left = tile_start_x.min(BORDER as u64) as u32;
        let avail_top = tile_start_y.min(BORDER as u64) as u32;
        let avail_right = (level_info
            .width
            .saturating_sub(tile_start_x + inner_w as u64))
        .min(BORDER as u64) as u32;
        let avail_bottom = (level_info
            .height
            .saturating_sub(tile_start_y + inner_h as u64))
        .min(BORDER as u64) as u32;

        // Expanded read region
        let read_x = tile_start_x - avail_left as u64;
        let read_y = tile_start_y - avail_top as u64;
        let read_w = avail_left + inner_w + avail_right;
        let read_h = avail_top + inner_h + avail_bottom;

        // Convert to level-0 coordinates for read_region
        let read_x0 = (read_x as f64 * level_info.downsample) as i64;
        let read_y0 = (read_y as f64 * level_info.downsample) as i64;

        let read_data = self
            .wsi
            .read_region(read_x0, read_y0, coord.level, read_w, read_h)?;

        // Build the uniformly-bordered output buffer.
        let data = create_bordered_tile(
            &read_data, read_w, read_h, inner_w, inner_h, BORDER, avail_left, avail_top,
        );

        Ok(TileData::new(coord, data, inner_w, inner_h, BORDER))
    }

    /// Calculate visible tiles for a given viewport
    ///
    /// Arguments:
    /// - level: the resolution level to use
    /// - bounds_left, bounds_top, bounds_right, bounds_bottom: visible area in level 0 coordinates
    pub fn visible_tiles(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
    ) -> Vec<TileCoord> {
        self.visible_tiles_with_margin(
            level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
            1,
        )
    }

    pub fn visible_tiles_with_margin(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
        margin_tiles: i32,
    ) -> Vec<TileCoord> {
        let level_info = match self.wsi.level(level) {
            Some(info) => info,
            None => return Vec::new(),
        };

        let tile_size = self.tile_size_for_level(level) as f64;

        // Convert bounds from level 0 to current level coordinates
        let level_left = bounds_left / level_info.downsample;
        let level_top = bounds_top / level_info.downsample;
        let level_right = bounds_right / level_info.downsample;
        let level_bottom = bounds_bottom / level_info.downsample;

        // Calculate tile range (with 1 tile margin for smooth scrolling)
        let margin_tiles = margin_tiles.max(0) as i64;
        let start_tile_x = ((level_left / tile_size).floor() as i64 - margin_tiles).max(0) as u64;
        let start_tile_y = ((level_top / tile_size).floor() as i64 - margin_tiles).max(0) as u64;
        let end_tile_x = ((level_right / tile_size).ceil() as u64 + margin_tiles as u64)
            .min(level_info.tiles_x(self.tile_size));
        let end_tile_y = ((level_bottom / tile_size).ceil() as u64 + margin_tiles as u64)
            .min(level_info.tiles_y(self.tile_size));

        // Adaptive cap: allow larger visible sets on larger windows / zoomed-out views.
        let tile_count =
            (end_tile_x.saturating_sub(start_tile_x)) * (end_tile_y.saturating_sub(start_tile_y));
        let max_tiles = tile_count.min(4096) as usize;
        if tile_count > 4096 {
            tracing::warn!(
                "visible_tiles would return {} tiles (capped), level={}, bounds=({:.0},{:.0})-({:.0},{:.0})",
                tile_count,
                level,
                bounds_left,
                bounds_top,
                bounds_right,
                bounds_bottom
            );
        }

        let mut tiles = Vec::with_capacity(max_tiles);

        for y in start_tile_y..end_tile_y {
            for x in start_tile_x..end_tile_x {
                tiles.push(TileCoord::new(self.file_id, level, x, y, tile_size as u32));
                if tiles.len() >= max_tiles {
                    return tiles;
                }
            }
        }

        tiles
    }

    /// Calculate tiles to prefetch (adjacent zoom levels)
    ///
    /// Level numbering: 0 = highest resolution (most tiles), higher levels = lower resolution
    pub fn prefetch_tiles(&self, current_tiles: &[TileCoord], level: u32) -> Vec<TileCoord> {
        let mut prefetch = Vec::new();
        let level_count = self.wsi.level_count();

        // Prefetch from the next LOWER resolution level (level + 1) as fallback
        // Each tile at current level maps to one parent tile at level + 1
        if level + 1 < level_count {
            for tile in current_tiles {
                let Some(level_info) = self.wsi.level(tile.level) else {
                    continue;
                };
                let parent_tile_size = self.tile_size_for_level(level + 1);
                let Some(parent_level_info) = self.wsi.level(level + 1) else {
                    continue;
                };
                let image_x = tile.x as f64 * tile.tile_size as f64 * level_info.downsample;
                let image_y = tile.y as f64 * tile.tile_size as f64 * level_info.downsample;
                let parent_span = parent_tile_size as f64 * parent_level_info.downsample;
                let parent_x = (image_x / parent_span).floor() as u64;
                let parent_y = (image_y / parent_span).floor() as u64;
                let coord = TileCoord::new(
                    self.file_id,
                    level + 1,
                    parent_x,
                    parent_y,
                    parent_tile_size,
                );
                if !prefetch.contains(&coord) {
                    prefetch.push(coord);
                }
            }
        }

        // Prefetch from the next HIGHER resolution level (level - 1) for when user zooms in
        // Each tile at current level corresponds to 4 child tiles at level - 1
        if level > 0 {
            for tile in current_tiles {
                let Some(level_info) = self.wsi.level(tile.level) else {
                    continue;
                };
                let child_level = level - 1;
                let child_tile_size = self.tile_size_for_level(child_level);
                let Some(child_level_info) = self.wsi.level(child_level) else {
                    continue;
                };
                let image_left = tile.x as f64 * tile.tile_size as f64 * level_info.downsample;
                let image_top = tile.y as f64 * tile.tile_size as f64 * level_info.downsample;
                let image_right = image_left + tile.tile_size as f64 * level_info.downsample;
                let image_bottom = image_top + tile.tile_size as f64 * level_info.downsample;
                let child_span = child_tile_size as f64 * child_level_info.downsample;
                let start_x = (image_left / child_span).floor().max(0.0) as u64;
                let start_y = (image_top / child_span).floor().max(0.0) as u64;
                let end_x = (image_right / child_span).ceil().max(0.0) as u64;
                let end_y = (image_bottom / child_span).ceil().max(0.0) as u64;

                for child_y in start_y..end_y {
                    for child_x in start_x..end_x {
                        let coord = TileCoord::new(
                            self.file_id,
                            child_level,
                            child_x,
                            child_y,
                            child_tile_size,
                        );
                        if !prefetch.contains(&coord) {
                            prefetch.push(coord);
                        }
                    }
                }
            }
        }

        prefetch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_test_file() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("fixtures/C3L-00088-22.svs")
    }

    #[test]
    fn test_tile_coord_equality() {
        let a = TileCoord::new(0, 0, 1, 2, 256);
        let b = TileCoord::new(0, 0, 1, 2, 256);
        let c = TileCoord::new(0, 0, 1, 3, 256);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_placeholder_tile() {
        let coord = TileCoord::new(0, 0, 0, 0, 256);
        let tile = TileData::placeholder(coord, 256);

        assert_eq!(tile.width, 256);
        assert_eq!(tile.height, 256);
        assert_eq!(tile.data.len(), 256 * 256 * 4);
    }

    #[test]
    fn test_load_tile_sync() {
        let path = get_test_file();
        if !path.exists() {
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI file");
        let manager = TileManager::new(wsi, 0);

        let coord = TileCoord::new(0, 0, 0, 0, 256);
        let tile = manager.load_tile_sync(coord).expect("Failed to load tile");

        assert!(tile.width > 0);
        assert!(tile.height > 0);
        assert!(!tile.data.is_empty());
    }

    #[test]
    fn test_visible_tiles_calculation() {
        let path = get_test_file();
        if !path.exists() {
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI file");
        let manager = TileManager::new(wsi, 0);

        // Get tiles visible in a 1024x768 viewport at level 0, origin
        let tiles = manager.visible_tiles(0, 0.0, 0.0, 1024.0, 768.0);

        assert!(!tiles.is_empty());
        // At zoom 1.0 and viewport 1024x768 with 256 size tiles,
        // we should have roughly 4x3 + margin tiles
        assert!(tiles.len() >= 12);
    }
}
