//! WSI (Whole Slide Image) file handling
//!
//! [`WsiFile`] wraps an OpenSlide handle (see [`crate::openslide`]) and
//! caches everything that does not change while a slide is open: the level
//! geometry, the full property map and the parsed metadata in
//! [`WsiProperties`].
//!
//! # Coordinates
//!
//! `x`/`y` arguments of region reads are **level-0** pixel coordinates;
//! `width`/`height` are pixels **of the requested level**. This matches
//! OpenSlide's `openslide_read_region`.

mod metadata;

use crate::openslide::OpenSlide;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

#[cfg(test)]
pub(crate) use metadata::{extract_stain_from_text, parse_stain_value_from_token};

/// Default tile size for rendering
pub const DEFAULT_TILE_SIZE: u32 = 256;
pub const MEDIUM_TILE_SIZE: u32 = 512;
pub const COARSE_TILE_SIZE: u32 = 1024;

/// Largest region (in pixels) a single read may allocate: 2^28 pixels, i.e.
/// 1 GiB of RGBA. Larger requests fail with [`Error::RegionTooLarge`]
/// instead of aborting the process on allocation failure.
pub const MAX_REGION_PIXELS: u64 = 1 << 28;

/// A rectangle in level-0 pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlideBounds {
    pub x: i64,
    pub y: i64,
    pub width: u64,
    pub height: u64,
}

impl SlideBounds {
    /// The whole level-0 canvas of a `width` x `height` slide.
    pub fn full(width: u64, height: u64) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// Exclusive right edge.
    pub fn right(&self) -> i64 {
        self.x.saturating_add(self.width as i64)
    }

    /// Exclusive bottom edge.
    pub fn bottom(&self) -> i64 {
        self.y.saturating_add(self.height as i64)
    }

    /// Centre point.
    pub fn center(&self) -> (f64, f64) {
        (
            self.x as f64 + self.width as f64 / 2.0,
            self.y as f64 + self.height as f64 / 2.0,
        )
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Overlap of two rectangles, `None` if they do not overlap.
    pub fn intersect(&self, other: &SlideBounds) -> Option<SlideBounds> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        (right > x && bottom > y).then(|| SlideBounds {
            x,
            y,
            width: (right - x) as u64,
            height: (bottom - y) as u64,
        })
    }

    /// Smallest rectangle containing both.
    pub fn union(&self, other: &SlideBounds) -> SlideBounds {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        SlideBounds {
            x,
            y,
            width: (self.right().max(other.right()) - x) as u64,
            height: (self.bottom().max(other.bottom()) - y) as u64,
        }
    }
}

/// WSI file level information
#[derive(Debug, Clone, PartialEq)]
pub struct WsiLevel {
    /// Level index (0 = highest resolution)
    pub level: u32,
    /// Width at this level in pixels
    pub width: u64,
    /// Height at this level in pixels
    pub height: u64,
    /// Downsample factor from level 0
    pub downsample: f64,
    /// Native tile width of the file (if the format has a tile grid)
    pub tile_width: Option<u32>,
    /// Native tile height of the file (if the format has a tile grid)
    pub tile_height: Option<u32>,
}

impl WsiLevel {
    /// Calculate number of tiles in X direction at this level
    pub fn tiles_x(&self, tile_size: u32) -> u64 {
        self.width.div_ceil(tile_size.max(1) as u64)
    }

    /// Calculate number of tiles in Y direction at this level
    pub fn tiles_y(&self, tile_size: u32) -> u64 {
        self.height.div_ceil(tile_size.max(1) as u64)
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
    /// Microns per pixel at level 0 along X (`openslide.mpp-x`), validated
    /// to be finite and positive.
    pub mpp_x: Option<f64>,
    /// Microns per pixel at level 0 along Y (`openslide.mpp-y`).
    pub mpp_y: Option<f64>,
    /// Objective power (e.g. 2.5, 20, 40), parsed as a float.
    pub objective_power: Option<f64>,
    /// Scan date if available, from the vendor-specific property.
    pub scan_date: Option<String>,
    /// Stain label if available in slide metadata
    pub stain: Option<String>,
    /// All available levels
    pub levels: Vec<WsiLevel>,
    /// Total width at level 0
    pub width: u64,
    /// Total height at level 0
    pub height: u64,
    /// Bounding box of the non-empty region (`openslide.bounds-*`), when
    /// the format reports one (MIRAX, some Hamamatsu, Leica and Philips
    /// slides).
    pub bounds: Option<SlideBounds>,
    /// Scanned regions (`openslide.region[i].*`), when the format reports
    /// them (e.g. Ventana BIF). Areas outside every region hold no data.
    pub regions: Vec<SlideBounds>,
    /// Colour of areas without image data (`openslide.background-color`),
    /// white when the slide does not specify one.
    pub background_color: [u8; 3],
    /// `openslide.quickhash-1`, when OpenSlide could compute it.
    pub quickhash1: Option<String>,
    /// `openslide.comment`
    pub comment: Option<String>,
    /// Size of the slide's ICC profile in bytes (`openslide.icc-size`).
    pub icc_profile_size: Option<u64>,
}

impl WsiProperties {
    /// The area of the slide that holds image data: the bounds rectangle
    /// if present, else the union of the regions, else the whole canvas.
    ///
    /// Use it for the initial fit, pan limits, the minimap and thumbnails.
    pub fn content_bounds(&self) -> SlideBounds {
        if let Some(bounds) = self.bounds {
            return bounds;
        }
        self.regions
            .iter()
            .copied()
            .reduce(|a, b| a.union(&b))
            .unwrap_or_else(|| SlideBounds::full(self.width, self.height))
    }

    /// Mean of `mpp_x` and `mpp_y` when both are known, else the one that is.
    pub fn mpp(&self) -> Option<f64> {
        match (self.mpp_x, self.mpp_y) {
            (Some(x), Some(y)) => Some((x + y) / 2.0),
            (x, y) => x.or(y),
        }
    }
}

/// OpenSlide's best-level rule, computed from cached level data: the
/// coarsest level whose downsample does not exceed `downsample`.
pub fn best_level_for_downsample(levels: &[WsiLevel], downsample: f64) -> u32 {
    let at_least_finest = matches!(
        downsample.partial_cmp(&levels.first().map_or(1.0, |level| level.downsample)),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    );
    if levels.is_empty() || !at_least_finest {
        // Too small (or NaN): the finest level.
        return 0;
    }
    levels
        .iter()
        .skip(1)
        .position(|level| downsample < level.downsample)
        .map(|index| index as u32)
        .unwrap_or(levels.len() as u32 - 1)
}

/// State shared by every [`WsiFile`] that refers to the same open slide.
pub(crate) struct SharedSlide {
    slide: OpenSlide,
    properties: WsiProperties,
    property_map: BTreeMap<String, String>,
}

/// WSI file wrapper.
pub struct WsiFile {
    shared: std::sync::Arc<SharedSlide>,
    /// Tile size to use
    tile_size: u32,
}

fn read_metadata(path: &Path, slide: &OpenSlide) -> Result<(WsiProperties, BTreeMap<String, String>)> {
    let property_map = slide.properties()?;
    let level_count = slide.level_count()?;
    if level_count == 0 {
        return Err(Error::InvalidFormat(format!(
            "{}: slide has no pyramid levels",
            path.display()
        )));
    }
    debug!("WSI has {} levels", level_count);

    let mut levels = Vec::with_capacity(level_count as usize);
    for level in 0..level_count {
        let (width, height) = slide.level_dimensions(level)?;
        let downsample = slide.level_downsample(level)?;
        let (tile_width, tile_height) = metadata::level_tile_size(&property_map, level);
        if width == 0 || height == 0 {
            return Err(Error::InvalidFormat(format!(
                "{}: level {level} is empty",
                path.display()
            )));
        }
        debug!(
            "Level {}: {}x{} (downsample: {:.2})",
            level, width, height, downsample
        );
        levels.push(WsiLevel {
            level,
            width,
            height,
            downsample,
            tile_width,
            tile_height,
        });
    }

    let (width, height) = (levels[0].width, levels[0].height);
    let get = |key: &str| property_map.get(key).cloned();
    let properties = WsiProperties {
        path: path.to_path_buf(),
        filename: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        vendor: get("openslide.vendor"),
        mpp_x: metadata::parse_mpp(&property_map, "openslide.mpp-x"),
        mpp_y: metadata::parse_mpp(&property_map, "openslide.mpp-y"),
        objective_power: metadata::parse_objective_power(&property_map),
        scan_date: metadata::parse_scan_date(&property_map),
        stain: metadata::extract_stain_metadata(&property_map),
        bounds: metadata::parse_bounds(&property_map, width, height),
        regions: metadata::parse_regions(&property_map, width, height),
        background_color: metadata::parse_background_color(&property_map),
        quickhash1: get("openslide.quickhash-1"),
        comment: get("openslide.comment"),
        icc_profile_size: get("openslide.icc-size")
            .and_then(|value| value.trim().parse().ok())
            .filter(|&size: &u64| size > 0),
        levels,
        width,
        height,
    };
    Ok((properties, property_map))
}

impl WsiFile {
    /// Open a WSI file from the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        info!("Opening WSI file: {}", path.display());

        let slide = OpenSlide::open(path)?;
        let (properties, property_map) = read_metadata(path, &slide)?;

        info!(
            "Opened {} ({}x{}, {} levels, vendor: {:?})",
            properties.filename,
            properties.width,
            properties.height,
            properties.levels.len(),
            properties.vendor
        );

        Ok(Self {
            shared: std::sync::Arc::new(SharedSlide {
                slide,
                properties,
                property_map,
            }),
            tile_size: DEFAULT_TILE_SIZE,
        })
    }

    /// Open a fresh handle to the same underlying file.
    pub fn reopen(&self) -> Result<Self> {
        let mut reopened = Self::open(&self.shared.properties.path)?;
        reopened.tile_size = self.tile_size;
        Ok(reopened)
    }

    /// Get file properties
    pub fn properties(&self) -> &WsiProperties {
        &self.shared.properties
    }

    /// The value of an OpenSlide property, e.g. `"aperio.AppMag"`.
    ///
    /// All properties are read once when the slide is opened.
    pub fn property(&self, name: &str) -> Option<&str> {
        self.shared.property_map.get(name).map(String::as_str)
    }

    /// Every OpenSlide property, ordered by name.
    pub fn property_map(&self) -> &BTreeMap<String, String> {
        &self.shared.property_map
    }

    /// Get tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Get the preferred tile size for a specific pyramid level.
    pub fn tile_size_for_level(&self, level: u32) -> u32 {
        let Some(level_info) = self.level(level) else {
            return self.tile_size;
        };

        if level_info.downsample >= 16.0 {
            COARSE_TILE_SIZE
        } else if level_info.downsample >= 4.0 {
            MEDIUM_TILE_SIZE
        } else {
            DEFAULT_TILE_SIZE
        }
    }

    /// Set the tile size used by [`read_tile`](Self::read_tile). A size of 0
    /// is ignored.
    pub fn set_tile_size(&mut self, size: u32) {
        if size == 0 {
            tracing::warn!("ignoring tile size 0");
            return;
        }
        self.tile_size = size;
    }

    /// Get the number of levels
    pub fn level_count(&self) -> u32 {
        self.shared.properties.levels.len() as u32
    }

    fn max_level(&self) -> u32 {
        self.level_count().saturating_sub(1)
    }

    /// The best level for displaying `downsample` (OpenSlide's rule: the
    /// coarsest level whose downsample does not exceed it).
    ///
    /// Computed in Rust from the level data cached at open, so it stays
    /// correct even when the OpenSlide handle is in an error state.
    pub fn best_level_for_downsample(&self, downsample: f64) -> u32 {
        best_level_for_downsample(&self.shared.properties.levels, downsample)
    }

    /// Get level information
    pub fn level(&self, level: u32) -> Option<&WsiLevel> {
        self.shared.properties.levels.get(level as usize)
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
            return Err(Error::InvalidLevel(level, self.max_level()));
        }
        let pixels = width as u64 * height as u64;
        if pixels > MAX_REGION_PIXELS {
            return Err(Error::RegionTooLarge {
                width: width as u64,
                height: height as u64,
                max_pixels: MAX_REGION_PIXELS,
            });
        }

        let mut argb = vec![0u32; pixels as usize];
        self.shared
            .slide
            .read_region_argb(&mut argb, x, y, level, width, height)
            .map_err(|e| Error::ReadTile {
                level,
                x: x.max(0) as u64,
                y: y.max(0) as u64,
                message: e.to_string(),
            })?;

        // OpenSlide returns native-endian ARGB; emit RGBA bytes.
        let mut data = Vec::with_capacity(argb.len() * 4);
        for pixel in argb {
            data.extend_from_slice(&[
                (pixel >> 16) as u8,
                (pixel >> 8) as u8,
                pixel as u8,
                (pixel >> 24) as u8,
            ]);
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
        self.read_tile_with_size(level, tile_x, tile_y, self.tile_size)
    }

    /// Read a tile at the specified coordinates using an explicit tile size.
    ///
    /// Edge tiles are clipped to the level; tiles entirely outside the level
    /// fail with [`Error::InvalidCoordinates`].
    pub fn read_tile_with_size(
        &self,
        level: u32,
        tile_x: u64,
        tile_y: u64,
        tile_size: u32,
    ) -> Result<Vec<u8>> {
        let level_info = self
            .level(level)
            .ok_or(Error::InvalidLevel(level, self.max_level()))?;
        let invalid = || Error::InvalidCoordinates {
            x: tile_x.min(i64::MAX as u64) as i64,
            y: tile_y.min(i64::MAX as u64) as i64,
            level,
        };
        if tile_size == 0 {
            return Err(Error::InvalidArgument("tile size 0".to_string()));
        }

        let tile_start_x = tile_x.checked_mul(tile_size as u64).ok_or_else(invalid)?;
        let tile_start_y = tile_y.checked_mul(tile_size as u64).ok_or_else(invalid)?;
        let actual_width = level_info
            .width
            .saturating_sub(tile_start_x)
            .min(tile_size as u64) as u32;
        let actual_height = level_info
            .height
            .saturating_sub(tile_start_y)
            .min(tile_size as u64) as u32;
        if actual_width == 0 || actual_height == 0 {
            return Err(invalid());
        }

        // Pixel coordinates at level 0
        let x = (tile_start_x as f64 * level_info.downsample) as i64;
        let y = (tile_start_y as f64 * level_info.downsample) as i64;
        self.read_region(x, y, level, actual_width, actual_height)
    }

    /// Get the thumbnail image if available
    pub fn thumbnail(&self, max_size: u32) -> Result<Vec<u8>> {
        let _ = max_size;
        // Use the lowest resolution level
        let level = self.max_level();
        let level_info = self
            .level(level)
            .ok_or(Error::InvalidLevel(level, self.max_level()))?;

        self.read_region(
            0,
            0,
            level,
            level_info.width.min(u32::MAX as u64) as u32,
            level_info.height.min(u32::MAX as u64) as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffLevel, TiffSpec, write_tiff};

    fn gradient(x: u32, y: u32) -> [u8; 3] {
        [(x % 251) as u8, (y % 241) as u8, ((x * 7 + y * 3) % 256) as u8]
    }

    fn pyramid_file(dir: &Path) -> PathBuf {
        let path = dir.join("pyramid.tiff");
        write_tiff(&path, &TiffSpec::pyramid(512, 384, 64, 3), gradient).unwrap();
        path
    }

    #[test]
    fn opens_a_synthetic_pyramid() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        assert_eq!(wsi.level_count(), 3);
        let props = wsi.properties();
        assert_eq!((props.width, props.height), (512, 384));
        assert_eq!(props.vendor.as_deref(), Some("generic-tiff"));
        assert_eq!(wsi.level(2).unwrap().width, 128);
        assert_eq!(wsi.level(1).unwrap().tile_width, Some(64));
        assert_eq!(wsi.property("openslide.vendor"), Some("generic-tiff"));
        assert!(wsi.property_map().contains_key("openslide.level-count"));
        assert_eq!(props.content_bounds(), SlideBounds::full(512, 384));
    }

    #[test]
    fn reads_regions_as_rgba() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        let rgba = wsi.read_region(100, 50, 0, 3, 2).unwrap();
        assert_eq!(rgba.len(), 3 * 2 * 4);
        let [r, g, b] = gradient(100, 50);
        assert_eq!(&rgba[..4], &[r, g, b, 255]);
        let [r, g, b] = gradient(102, 51);
        assert_eq!(&rgba[20..24], &[r, g, b, 255]);
    }

    #[test]
    fn tile_reads_clip_edges_and_reject_tiles_outside_the_level() {
        let dir = tempfile::tempdir().unwrap();
        let mut wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        // 512 = 2 * 200 + 112.
        let edge = wsi.read_tile_with_size(0, 2, 0, 200).unwrap();
        assert_eq!(edge.len(), 112 * 200 * 4);
        // Previously underflowed and panicked.
        assert!(matches!(
            wsi.read_tile_with_size(0, 99, 0, 200),
            Err(Error::InvalidCoordinates { .. })
        ));
        assert!(matches!(
            wsi.read_tile_with_size(7, 0, 0, 200),
            Err(Error::InvalidLevel(7, 2))
        ));
        wsi.set_tile_size(0);
        assert_eq!(wsi.tile_size(), DEFAULT_TILE_SIZE);
        assert!(wsi.read_tile(0, 0, 0).is_ok());
    }

    #[test]
    fn rejects_oversized_reads_before_allocating() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        assert!(matches!(
            wsi.read_region(0, 0, 0, u32::MAX, u32::MAX),
            Err(Error::RegionTooLarge { .. })
        ));
    }

    #[test]
    fn best_level_matches_openslide() {
        let dir = tempfile::tempdir().unwrap();
        let path = pyramid_file(dir.path());
        let wsi = WsiFile::open(&path).unwrap();
        let raw = OpenSlide::open(&path).unwrap();
        for ds in [0.1, 0.5, 1.0, 1.5, 1.999, 2.0, 2.5, 3.99, 4.0, 4.1, 100.0, f64::INFINITY] {
            assert_eq!(
                wsi.best_level_for_downsample(ds),
                raw.best_level_for_downsample(ds).unwrap(),
                "downsample {ds}"
            );
        }
        assert_eq!(wsi.best_level_for_downsample(f64::NAN), 0);
        assert_eq!(best_level_for_downsample(&[], 4.0), 0);
    }

    #[test]
    fn parses_metadata_from_an_aperio_style_slide() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.svs");
        write_tiff(&path, &TiffSpec::aperio(128, 96, 32, 4), gradient).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let props = wsi.properties();
        assert_eq!(props.vendor.as_deref(), Some("aperio"));
        // Fractional objectives used to be dropped (parsed as u32).
        assert_eq!(props.objective_power, Some(2.5));
        assert_eq!(props.mpp_x, Some(0.4942));
        assert_eq!(props.mpp(), Some(0.4942));
        assert_eq!(props.scan_date.as_deref(), Some("08/11/16 10:18:38"));
        assert_eq!(props.stain.as_deref(), Some("H&E"));
        assert_eq!(props.background_color, [255, 255, 255]);
        assert!(props.quickhash1.is_some());
    }

    #[test]
    fn parses_resolution_and_date_of_a_generic_tiff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tiff");
        let spec = TiffSpec {
            levels: vec![TiffLevel::new(64, 64, 32)],
            pixels_per_cm: Some(40_000),
            date_time: Some("2024:01:02 03:04:05".to_string()),
            description: Some("Smith&Evans 2019".to_string()),
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, gradient).unwrap();
        let props = WsiFile::open(&path).unwrap().properties().clone();
        assert!((props.mpp_x.unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(props.objective_power, None);
        assert_eq!(props.scan_date.as_deref(), Some("2024:01:02 03:04:05"));
        assert_eq!(props.stain, None, "free text must not be read as H&E");
    }

    #[test]
    fn missing_files_and_non_slides_are_distinct_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            WsiFile::open(dir.path().join("absent.svs")),
            Err(Error::FileNotFound(_))
        ));
        let text = dir.path().join("notes.txt");
        std::fs::write(&text, "not a slide").unwrap();
        assert!(matches!(WsiFile::open(&text), Err(Error::UnsupportedFormat(_))));
    }

    #[test]
    fn parses_explicit_stain_tokens() {
        assert_eq!(
            parse_stain_value_from_token("Stain = Masson trichrome"),
            Some("Masson trichrome".to_string())
        );
        assert_eq!(
            parse_stain_value_from_token("stain_name: PAS"),
            Some("PAS".to_string())
        );
    }

    #[test]
    fn detects_common_he_stain_markers() {
        assert_eq!(
            extract_stain_from_text("Case 12 | H&E | archived slide"),
            Some("H&E".to_string())
        );
        assert_eq!(
            extract_stain_from_text("hematoxylin and eosin section"),
            Some("H&E".to_string())
        );
    }
}
