//! WSI (Whole Slide Image) file handling
//!
//! This module provides abstraction over various WSI formats including
//! SVS, TIFF, and other formats supported by OpenSlide.

use crate::{Error, Result};
use openslide_rs::{OpenSlide, Address, Region, Size};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Default tile size for rendering
pub const DEFAULT_TILE_SIZE: u32 = 256;

/// WSI file level information
#[derive(Debug, Clone)]
pub struct WsiLevel {
    /// Level index (0 = highest resolution)
    pub level: u32,
    /// Width at this level in pixels
    pub width: u64,
    /// Height at this level in pixels
    pub height: u64,
    /// Downsample factor from level 0
    pub downsample: f64,
    /// Tile width (if available)
    pub tile_width: Option<u32>,
    /// Tile height (if available)  
    pub tile_height: Option<u32>,
}

impl WsiLevel {
    /// Calculate number of tiles in X direction at this level
    pub fn tiles_x(&self, tile_size: u32) -> u64 {
        (self.width + tile_size as u64 - 1) / tile_size as u64
    }

    /// Calculate number of tiles in Y direction at this level
    pub fn tiles_y(&self, tile_size: u32) -> u64 {
        (self.height + tile_size as u64 - 1) / tile_size as u64
    }
}

/// WSI file properties
#[derive(Debug, Clone)]
pub struct WsiProperties {
    /// File path
    pub path: PathBuf,
    /// File name
    pub filename: String,
    /// Vendor name (e.g., "aperio", "hamamatsu")
    pub vendor: Option<String>,
    /// Microns per pixel at level 0
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    /// Objective power (e.g., 20, 40)
    pub objective_power: Option<f64>,
    /// Scan date if available
    pub scan_date: Option<String>,
    /// All available levels
    pub levels: Vec<WsiLevel>,
    /// Total width at level 0
    pub width: u64,
    /// Total height at level 0
    pub height: u64,
}

/// WSI file wrapper.
///
/// OpenSlide itself is thread-safe for read operations, so each worker should
/// own its own handle rather than sharing one behind a mutex.
pub struct WsiFile {
    /// OpenSlide handle
    slide: OpenSlide,
    /// Cached properties
    properties: WsiProperties,
    /// Tile size to use
    tile_size: u32,
}

impl WsiFile {
    /// Open a WSI file from the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        info!("Opening WSI file: {}", path.display());

        let slide = OpenSlide::new(path)
            .map_err(|e| Error::OpenFile(format!("{}: {}", path.display(), e)))?;

        let level_count = slide.get_level_count()
            .map_err(|e| Error::OpenSlide(e.to_string()))?;

        debug!("WSI has {} levels", level_count);

        // Collect level information
        let mut levels = Vec::with_capacity(level_count as usize);
        for level in 0..level_count {
            let size = slide.get_level_dimensions(level)
                .map_err(|e| Error::OpenSlide(format!("Failed to get dimensions for level {}: {}", level, e)))?;
            
            let downsample = slide.get_level_downsample(level)
                .map_err(|e| Error::OpenSlide(format!("Failed to get downsample for level {}: {}", level, e)))?;
            
            // Get tile dimensions from properties if available
            let tile_width = slide.properties.openslide_properties.levels
                .get(level as usize)
                .and_then(|l| l.tile_width);
            let tile_height = slide.properties.openslide_properties.levels
                .get(level as usize)
                .and_then(|l| l.tile_height);

            levels.push(WsiLevel {
                level,
                width: size.w as u64,
                height: size.h as u64,
                downsample,
                tile_width,
                tile_height,
            });

            debug!("Level {}: {}x{} (downsample: {:.2})", level, size.w, size.h, downsample);
        }

        // Get base dimensions
        let size0 = slide.get_level0_dimensions()
            .map_err(|e| Error::OpenSlide(e.to_string()))?;

        // Extract properties
        let openslide_props = &slide.properties.openslide_properties;
        let aperio_props = &slide.properties.aperio_properties;
        
        let properties = WsiProperties {
            path: path.to_path_buf(),
            filename: path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            vendor: openslide_props.vendor.clone(),
            mpp_x: openslide_props.mpp_x.map(|v| v as f64),
            mpp_y: openslide_props.mpp_y.map(|v| v as f64),
            objective_power: openslide_props.objective_power.map(|v| v as f64),
            scan_date: aperio_props.date.clone(),
            levels,
            width: size0.w as u64,
            height: size0.h as u64,
        };

        info!(
            "Opened {} ({}x{}, {} levels, vendor: {:?})",
            properties.filename,
            properties.width,
            properties.height,
            properties.levels.len(),
            properties.vendor
        );

        Ok(Self {
            slide,
            properties,
            tile_size: DEFAULT_TILE_SIZE,
        })
    }

    /// Open a fresh handle to the same underlying file.
    pub fn reopen(&self) -> Result<Self> {
        Self::open(&self.properties.path)
    }

    /// Get file properties
    pub fn properties(&self) -> &WsiProperties {
        &self.properties
    }

    /// Get tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Set tile size
    pub fn set_tile_size(&mut self, size: u32) {
        self.tile_size = size;
    }

    /// Get the number of levels
    pub fn level_count(&self) -> u32 {
        self.properties.levels.len() as u32
    }

    /// Get the best level for the given downsample factor
    pub fn best_level_for_downsample(&self, downsample: f64) -> u32 {
        self.slide.get_best_level_for_downsample(downsample)
            .unwrap_or(0)
    }

    /// Get level information
    pub fn level(&self, level: u32) -> Option<&WsiLevel> {
        self.properties.levels.get(level as usize)
    }

    /// Read a region from the slide
    /// 
    /// # Arguments
    /// * `x` - X coordinate at level 0
    /// * `y` - Y coordinate at level 0
    /// * `level` - The level to read from
    /// * `width` - Width in pixels at the target level
    /// * `height` - Height in pixels at the target level
    pub fn read_region(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        if level >= self.level_count() {
            return Err(Error::InvalidLevel(level, self.level_count() - 1));
        }

        // Clamp coordinates to valid range for u32
        let x_u32 = x.max(0) as u32;
        let y_u32 = y.max(0) as u32;
        
        let region = Region {
            address: Address { x: x_u32, y: y_u32 },
            level,
            size: Size { w: width, h: height },
        };

        // read_region returns BGRA data, we need to convert to RGBA
        let mut data = self.slide.read_region(&region)
            .map_err(|e| Error::ReadTile {
                level,
                x: x as u64,
                y: y as u64,
                message: e.to_string(),
            })?;

        // Convert BGRA to RGBA
        for chunk in data.chunks_exact_mut(4) {
            chunk.swap(0, 2); // Swap B and R
        }

        Ok(data)
    }

    /// Read a tile at the specified coordinates
    /// 
    /// # Arguments
    /// * `level` - The level to read from
    /// * `tile_x` - Tile X coordinate
    /// * `tile_y` - Tile Y coordinate
    pub fn read_tile(&self, level: u32, tile_x: u64, tile_y: u64) -> Result<Vec<u8>> {
        let level_info = self.level(level)
            .ok_or(Error::InvalidLevel(level, self.level_count() - 1))?;

        // Calculate pixel coordinates at level 0
        let x = (tile_x * self.tile_size as u64) as f64 * level_info.downsample;
        let y = (tile_y * self.tile_size as u64) as f64 * level_info.downsample;

        // Clamp tile size to not exceed level bounds
        let tile_start_x = tile_x * self.tile_size as u64;
        let tile_start_y = tile_y * self.tile_size as u64;
        
        let actual_width = (level_info.width - tile_start_x).min(self.tile_size as u64) as u32;
        let actual_height = (level_info.height - tile_start_y).min(self.tile_size as u64) as u32;

        self.read_region(x as i64, y as i64, level, actual_width, actual_height)
    }

    /// Get the thumbnail image if available
    pub fn thumbnail(&self, max_size: u32) -> Result<Vec<u8>> {
        // Find the best level for thumbnail generation
        let aspect = self.properties.width as f64 / self.properties.height as f64;
        let (_thumb_w, _thumb_h) = if aspect > 1.0 {
            (max_size, (max_size as f64 / aspect) as u32)
        } else {
            ((max_size as f64 * aspect) as u32, max_size)
        };

        // Use the lowest resolution level
        let level = self.level_count() - 1;
        let level_info = self.level(level).unwrap();
        
        // Read the entire level and resize
        self.read_region(0, 0, level, level_info.width as u32, level_info.height as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_file() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("fixtures/C3L-00088-22.svs")
    }

    #[test]
    fn test_open_svs_file() {
        let path = get_test_file();
        if !path.exists() {
            eprintln!("Skipping test: fixture file not found at {:?}", path);
            return;
        }
        
        let wsi = WsiFile::open(&path).expect("Failed to open WSI file");
        assert!(wsi.level_count() > 0);
        assert!(wsi.properties().width > 0);
        assert!(wsi.properties().height > 0);
    }

    #[test]
    fn test_read_tile() {
        let path = get_test_file();
        if !path.exists() {
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI file");
        let tile = wsi.read_tile(0, 0, 0).expect("Failed to read tile");
        
        // Should have RGBA data
        assert!(!tile.is_empty());
    }

    #[test]
    fn test_best_level_for_downsample() {
        let path = get_test_file();
        if !path.exists() {
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI file");
        
        // At downsample 1.0, should return level 0
        let level = wsi.best_level_for_downsample(1.0);
        assert_eq!(level, 0);
        
        // At very high downsample, should return highest level
        let level = wsi.best_level_for_downsample(1000.0);
        assert!(level < wsi.level_count());
    }
}
