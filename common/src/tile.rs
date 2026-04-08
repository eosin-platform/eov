//! Tile management for WSI rendering
//!
//! This module handles tile coordinates, tile requests, and tile data management
//! for efficient rendering of whole slide images.

use crate::{WsiFile, Result, Error};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::trace;

/// Tile coordinate identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    /// Level index
    pub level: u32,
    /// X tile index
    pub x: u64,
    /// Y tile index
    pub y: u64,
}

impl TileCoord {
    pub fn new(level: u32, x: u64, y: u64) -> Self {
        Self { level, x, y }
    }
}

/// Tile data with RGBA pixels
#[derive(Debug, Clone)]
pub struct TileData {
    /// Tile coordinate
    pub coord: TileCoord,
    /// RGBA pixel data (width * height * 4 bytes)
    pub data: Vec<u8>,
    /// Actual tile width (may be smaller at edges)
    pub width: u32,
    /// Actual tile height (may be smaller at edges)
    pub height: u32,
}

impl TileData {
    /// Create a new tile data instance
    pub fn new(coord: TileCoord, data: Vec<u8>, width: u32, height: u32) -> Self {
        Self { coord, data, width, height }
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
                data[idx] = color;     // R
                data[idx + 1] = color; // G
                data[idx + 2] = color; // B
                data[idx + 3] = 255;   // A
            }
        }
        
        Self {
            coord,
            data,
            width: tile_size,
            height: tile_size,
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
    /// Tile size
    tile_size: u32,
    /// Request sender channel
    request_tx: Option<mpsc::Sender<TileRequest>>,
    /// Response receiver channel
    response_rx: Option<Arc<RwLock<mpsc::Receiver<TileData>>>>,
}

impl TileManager {
    /// Create a new tile manager for the given WSI file
    pub fn new(wsi: WsiFile) -> Self {
        let tile_size = wsi.tile_size();
        Self {
            wsi,
            tile_size,
            request_tx: None,
            response_rx: None,
        }
    }

    /// Get the tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Get the WSI file reference
    pub fn wsi(&self) -> &WsiFile {
        &self.wsi
    }

    /// Load a tile synchronously
    pub fn load_tile_sync(&self, coord: TileCoord) -> Result<TileData> {
        let level_info = self.wsi.level(coord.level)
            .ok_or(Error::InvalidLevel(coord.level, self.wsi.level_count() - 1))?;

        // Calculate actual tile dimensions (may be smaller at edges)
        let tile_start_x = coord.x * self.tile_size as u64;
        let tile_start_y = coord.y * self.tile_size as u64;
        
        let actual_width = (level_info.width.saturating_sub(tile_start_x))
            .min(self.tile_size as u64) as u32;
        let actual_height = (level_info.height.saturating_sub(tile_start_y))
            .min(self.tile_size as u64) as u32;

        if actual_width == 0 || actual_height == 0 {
            return Err(Error::InvalidCoordinates {
                x: coord.x as i64,
                y: coord.y as i64,
                level: coord.level,
            });
        }

        trace!(
            "Loading tile {:?}, actual size: {}x{}",
            coord, actual_width, actual_height
        );

        let data = self.wsi.read_tile(coord.level, coord.x, coord.y)?;

        Ok(TileData::new(coord, data, actual_width, actual_height))
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
        self.visible_tiles_with_margin(level, bounds_left, bounds_top, bounds_right, bounds_bottom, 1)
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

        let tile_size = self.tile_size as f64;
        
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
        let tile_count = (end_tile_x.saturating_sub(start_tile_x)) * (end_tile_y.saturating_sub(start_tile_y));
        let max_tiles = tile_count.min(4096) as usize;
        if tile_count > 4096 {
            tracing::warn!(
                "visible_tiles would return {} tiles (capped), level={}, bounds=({:.0},{:.0})-({:.0},{:.0})",
                tile_count, level, bounds_left, bounds_top, bounds_right, bounds_bottom
            );
        }

        let mut tiles = Vec::with_capacity(max_tiles);

        for y in start_tile_y..end_tile_y {
            for x in start_tile_x..end_tile_x {
                tiles.push(TileCoord::new(level, x, y));
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
    pub fn prefetch_tiles(
        &self,
        current_tiles: &[TileCoord],
        level: u32,
    ) -> Vec<TileCoord> {
        let mut prefetch = Vec::new();
        let level_count = self.wsi.level_count();

        // Prefetch from the next LOWER resolution level (level + 1) as fallback
        // Each tile at current level maps to one parent tile at level + 1
        if level + 1 < level_count {
            for tile in current_tiles {
                // Parent tile covers 2x2 area of current tiles
                let parent_x = tile.x / 2;
                let parent_y = tile.y / 2;
                let coord = TileCoord::new(level + 1, parent_x, parent_y);
                if !prefetch.contains(&coord) {
                    prefetch.push(coord);
                }
            }
        }

        // Prefetch from the next HIGHER resolution level (level - 1) for when user zooms in
        // Each tile at current level corresponds to 4 child tiles at level - 1
        if level > 0 {
            for tile in current_tiles {
                for dy in 0..2u64 {
                    for dx in 0..2u64 {
                        let child_x = tile.x * 2 + dx;
                        let child_y = tile.y * 2 + dy;
                        let coord = TileCoord::new(level - 1, child_x, child_y);
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
        let a = TileCoord::new(0, 1, 2);
        let b = TileCoord::new(0, 1, 2);
        let c = TileCoord::new(0, 1, 3);
        
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_placeholder_tile() {
        let coord = TileCoord::new(0, 0, 0);
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
        let manager = TileManager::new(wsi);
        
        let coord = TileCoord::new(0, 0, 0);
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
        let manager = TileManager::new(wsi);
        
        // Get tiles visible in a 1024x768 viewport at level 0, origin
        let tiles = manager.visible_tiles(0, 0.0, 0.0, 1024.0, 768.0, 1.0);
        
        assert!(!tiles.is_empty());
        // At zoom 1.0 and viewport 1024x768 with 256 size tiles,
        // we should have roughly 4x3 + margin tiles
        assert!(tiles.len() >= 12);
    }
}
