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
//! OpenSlide's `openslide_read_region`. Negative coordinates and areas
//! outside the slide are allowed and read as "no data".
//!
//! # Sharing and threads
//!
//! A `WsiFile` is a cheap handle (`Clone` is an `Arc` clone) to one shared,
//! thread-safe OpenSlide handle per slide, and [`WsiFile::open`] reuses the
//! slide when the same unchanged file is already open anywhere in the
//! process (see [`registry`]). Give every worker thread a clone instead of
//! opening the slide again; all handles use one process-wide OpenSlide tile
//! cache ([`set_openslide_cache_capacity`]).
//!
//! # Errors and recovery
//!
//! OpenSlide errors are sticky. A `WsiFile` detects a poisoned handle,
//! re-runs the failed read on a freshly opened handle and only reports
//! [`Error::CorruptRegion`] when the data is bad on both; the poisoned
//! handle is replaced either way, with rate-limited reopening. Reads that
//! cannot be served while recovery is rate-limited fail with the transient
//! [`Error::HandlePoisoned`]. [`WsiFile::health`] reports what happened so
//! a UI can tell the user.

mod health;
mod metadata;
mod pixels;
pub mod registry;

use crate::openslide::OpenSlide;
use crate::{Error, Result};
use health::{BadKey, Geometry, HandleCell};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info};

pub use health::{HealthState, SlideHealth};
pub use pixels::AlphaMode;
pub use registry::{
    DEFAULT_OPENSLIDE_CACHE_BYTES, openslide_cache_capacity, recommended_openslide_cache_bytes,
    set_openslide_cache_capacity,
};

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
    pub(crate) handle: HandleCell,
    pub(crate) canonical_path: PathBuf,
    pub(crate) stamp: registry::FileStamp,
    /// Whether [`WsiFile::open`] may hand this slide to other callers.
    pub(crate) shareable: bool,
    properties: WsiProperties,
    property_map: BTreeMap<String, String>,
}

/// How [`WsiFile::open_with`] opens a slide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOptions {
    /// Reuse (and offer for reuse) the handle of an identical slide that is
    /// already open in this process. Default `true`.
    pub share: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { share: true }
    }
}

/// An open whole-slide image.
///
/// Cloning is cheap and shares the underlying OpenSlide handle; see the
/// [module documentation](self) for the threading and recovery model.
#[derive(Clone)]
pub struct WsiFile {
    shared: Arc<SharedSlide>,
    /// Tile size to use
    tile_size: u32,
}

impl std::fmt::Debug for WsiFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsiFile")
            .field("path", &self.shared.properties.path)
            .field("levels", &self.shared.properties.levels.len())
            .field("tile_size", &self.tile_size)
            .finish_non_exhaustive()
    }
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
    /// Opens a slide, reusing the shared handle if the same unchanged file
    /// is already open in this process.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default())
    }

    /// Opens a slide with a handle of its own that is never shared with
    /// other `open` calls (clones of the result still share it).
    pub fn open_exclusive<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, &OpenOptions { share: false })
    }

    /// Opens a slide with explicit [`OpenOptions`].
    pub fn open_with<P: AsRef<Path>>(path: P, options: &OpenOptions) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(registry::not_found(path));
        }
        let stamp = registry::FileStamp::of(path)?;
        let canonical_path = registry::canonical(path);

        if options.share
            && let Some(shared) = registry::find_shared(&canonical_path, &stamp)
        {
            debug!("Reusing open slide handle for {}", path.display());
            return Ok(Self {
                shared,
                tile_size: DEFAULT_TILE_SIZE,
            });
        }

        info!("Opening WSI file: {}", path.display());
        let slide = OpenSlide::open(path)?;
        let (properties, property_map) = read_metadata(path, &slide)?;
        let geometry = Geometry::of(&slide)?;

        info!(
            "Opened {} ({}x{}, {} levels, vendor: {:?})",
            properties.filename,
            properties.width,
            properties.height,
            properties.levels.len(),
            properties.vendor
        );

        let shared = Arc::new(SharedSlide {
            handle: HandleCell::new(
                path.to_path_buf(),
                slide,
                geometry,
                registry::attach_global_cache,
            ),
            canonical_path,
            stamp,
            shareable: options.share,
            properties,
            property_map,
        });
        Ok(Self {
            shared: registry::register(shared),
            tile_size: DEFAULT_TILE_SIZE,
        })
    }

    /// Returns a handle to the same slide, re-validating the file: if it is
    /// unchanged this shares the open handle (like [`Clone`]); if it changed
    /// on disk, the new file is opened.
    pub fn reopen(&self) -> Result<Self> {
        let mut reopened = Self::open_with(
            &self.shared.properties.path,
            &OpenOptions {
                share: self.shared.shareable,
            },
        )?;
        reopened.tile_size = self.tile_size;
        Ok(reopened)
    }

    /// Whether two `WsiFile`s share the same OpenSlide handle.
    pub fn shares_handle_with(&self, other: &WsiFile) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// A snapshot of the health of the slide's OpenSlide handle.
    pub fn health(&self) -> SlideHealth {
        self.shared.handle.health()
    }

    /// Forgets the regions remembered as unreadable, so they are tried again
    /// (for example after the user restored the file or the network).
    pub fn clear_bad_regions(&self) {
        self.shared.handle.clear_bad_regions();
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

    /// Reads a region as opaque RGBA (`width * height * 4` bytes).
    ///
    /// OpenSlide's premultiplied pixels are composited over the slide's
    /// background colour ([`WsiProperties::background_color`], white unless
    /// the slide specifies one), so gaps between scanned regions and areas
    /// outside the slide read as glass, never as black or transparent. Use
    /// [`read_region_with`](Self::read_region_with) to keep the alpha channel.
    ///
    /// # Arguments
    /// * `x` - X coordinate at level 0 (may be negative)
    /// * `y` - Y coordinate at level 0 (may be negative)
    /// * `level` - The level to read from
    /// * `width` - Width in pixels at the target level
    /// * `height` - Height in pixels at the target level
    ///
    /// # Errors
    /// [`Error::InvalidLevel`], [`Error::RegionTooLarge`] (above
    /// [`MAX_REGION_PIXELS`]), the transient [`Error::HandlePoisoned`] and
    /// the permanent [`Error::CorruptRegion`].
    pub fn read_region(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        self.read_region_with(x, y, level, width, height, AlphaMode::Composited)
    }

    /// Like [`read_region`](Self::read_region), with an explicit
    /// [`AlphaMode`].
    pub fn read_region_with(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
        mode: AlphaMode,
    ) -> Result<Vec<u8>> {
        let len = self.region_byte_len(level, width, height)?;
        let mut out = pixels::try_alloc(len)?;
        self.read_region_into(x, y, level, width, height, mode, &mut out)?;
        Ok(out)
    }

    /// Reads a region into `out`, which must hold exactly
    /// `width * height * 4` bytes. Avoids an allocation per read.
    #[allow(clippy::too_many_arguments)]
    pub fn read_region_into(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
        mode: AlphaMode,
        out: &mut [u8],
    ) -> Result<()> {
        let len = self.region_byte_len(level, width, height)?;
        if out.len() != len {
            return Err(Error::InvalidArgument(format!(
                "output buffer holds {} bytes, region needs {len}",
                out.len()
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let key = BadKey::Region {
            level,
            x,
            y,
            width,
            height,
        };
        let background = self.shared.properties.background_color;

        // OpenSlide writes native-endian u32 ARGB. Let it write straight into
        // `out` when the buffer happens to be 4-byte aligned (it practically
        // always is), then convert in place.
        // SAFETY: every bit pattern is a valid u32 and u8.
        let (prefix, aligned, suffix) = unsafe { out.align_to_mut::<u32>() };
        if prefix.is_empty() && suffix.is_empty() {
            self.shared.handle.run(Some(key), |slide| {
                slide.read_region_argb(aligned, x, y, level, width, height)
            })?;
            pixels::argb_bytes_to_rgba_in_place(out, mode, background);
        } else {
            let mut argb = vec![0u32; len / 4];
            self.shared.handle.run(Some(key), |slide| {
                slide.read_region_argb(&mut argb, x, y, level, width, height)
            })?;
            pixels::argb_to_rgba(&argb, out, mode, background);
        }
        Ok(())
    }

    /// Validates a region read and returns its size in bytes.
    fn region_byte_len(&self, level: u32, width: u32, height: u32) -> Result<usize> {
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
        Ok(pixels as usize * 4)
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

    /// A 128x128 two-level pyramid whose level-0 tile (1, 1) is unreadable.
    fn broken_file(dir: &Path) -> PathBuf {
        let path = dir.join("broken.tiff");
        let spec = TiffSpec {
            levels: vec![
                TiffLevel::new(128, 128, 32).with_broken_tile(1, 1),
                TiffLevel::new(64, 64, 32),
            ],
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, gradient).unwrap();
        path
    }

    #[test]
    fn a_bad_tile_poisons_nothing_but_itself() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(broken_file(dir.path())).unwrap();
        let good = wsi.read_region(0, 0, 0, 32, 32).unwrap();

        let err = wsi.read_region(32, 32, 0, 32, 32).unwrap_err();
        assert!(matches!(err, Error::CorruptRegion { level: 0, x: 32, y: 32, .. }), "{err:?}");
        assert!(!err.is_transient());

        // The shared handle was replaced: good tiles read again, correctly.
        assert_eq!(wsi.read_region(0, 0, 0, 32, 32).unwrap(), good);
        assert!(wsi.read_region(0, 0, 1, 64, 64).is_ok());
        assert_eq!(wsi.best_level_for_downsample(32.0), 1);

        let health = wsi.health();
        assert_eq!(health.state, HealthState::Healthy);
        assert_eq!(health.poison_events, 1);
        assert_eq!(health.bad_regions, 1);
        assert!(health.reopens >= 1);
        assert!(health.last_error.is_some());

        // The bad region now fails fast without poisoning the handle again.
        assert!(matches!(
            wsi.read_region(32, 32, 0, 32, 32),
            Err(Error::CorruptRegion { .. })
        ));
        assert_eq!(wsi.health().poison_events, 1);

        // After clearing, the region is retried (and found bad again).
        wsi.clear_bad_regions();
        assert_eq!(wsi.health().bad_regions, 0);
        assert!(wsi.read_region(32, 32, 0, 32, 32).is_err());
        assert_eq!(wsi.health().poison_events, 2);
    }

    #[test]
    fn collateral_failures_on_shared_handles_are_retried_transparently() {
        let dir = tempfile::tempdir().unwrap();
        let path = broken_file(dir.path());
        let a = WsiFile::open(&path).unwrap();
        let b = WsiFile::open(&path).unwrap();
        assert!(a.shares_handle_with(&b));

        // Poison the shared handle behind `b`'s back, as another worker would.
        {
            let slot = a.shared.handle.current();
            let mut buf = vec![0u32; 32 * 32];
            assert!(slot.slide.read_region_argb(&mut buf, 32, 32, 0, 32, 32).is_err());
            assert!(slot.slide.is_poisoned());
        }
        assert_eq!(b.health().state, HealthState::Recovering);
        assert!(b.read_region(64, 0, 0, 32, 32).is_ok());
        assert_eq!(a.health().state, HealthState::Healthy);
    }

    #[test]
    fn concurrent_readers_survive_a_bad_tile() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(broken_file(dir.path())).unwrap();
        let expected = wsi.read_region(96, 96, 0, 32, 32).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let wsi = wsi.clone();
                let expected = &expected;
                scope.spawn(move || {
                    for _ in 0..40 {
                        assert_eq!(&wsi.read_region(96, 96, 0, 32, 32).unwrap(), expected);
                    }
                });
            }
            let wsi = wsi.clone();
            scope.spawn(move || {
                for _ in 0..5 {
                    assert!(wsi.read_region(32, 32, 0, 32, 32).is_err());
                }
            });
        });
        assert_eq!(wsi.health().bad_regions, 1);
        assert_eq!(wsi.health().state, HealthState::Healthy);
    }

    #[test]
    fn reopening_backs_off_when_the_file_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let path = broken_file(dir.path());
        let wsi = WsiFile::open_exclusive(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let err = wsi.read_region(32, 32, 0, 32, 32).unwrap_err();
        assert!(matches!(err, Error::HandlePoisoned(_)), "{err:?}");
        assert!(err.is_transient());
        let health = wsi.health();
        assert_eq!(health.state, HealthState::Failing);
        assert_eq!(health.failed_reopens, 1);

        // Within the backoff window nothing is reopened.
        assert!(matches!(
            wsi.read_region(0, 0, 0, 32, 32),
            Err(Error::HandlePoisoned(_))
        ));
        assert_eq!(wsi.health().failed_reopens, 1);
        // Metadata stays available.
        assert_eq!(wsi.level_count(), 2);
        assert_eq!(wsi.best_level_for_downsample(2.0), 1);
    }

    #[test]
    fn open_shares_handles_of_unchanged_files_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = pyramid_file(dir.path());
        let a = WsiFile::open(&path).unwrap();
        let b = WsiFile::open(&path).unwrap();
        assert!(a.shares_handle_with(&b));
        assert!(a.shares_handle_with(&a.clone()));
        assert!(a.shares_handle_with(&a.reopen().unwrap()));

        let exclusive = WsiFile::open_exclusive(&path).unwrap();
        assert!(!exclusive.shares_handle_with(&a));
        assert!(!WsiFile::open(&path).unwrap().shares_handle_with(&exclusive));

        // Replace the file: a new open must not reuse the stale handle.
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_tiff(&path, &TiffSpec::pyramid(256, 256, 64, 2), gradient).unwrap();
        let fresh = WsiFile::open(&path).unwrap();
        assert!(!fresh.shares_handle_with(&a));
        assert_eq!(fresh.properties().width, 256);
        assert_eq!(a.properties().width, 512);
    }

    #[test]
    fn handles_use_the_resizable_global_cache() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        let before = wsi.read_region(0, 0, 0, 128, 128).unwrap();
        let capacity = openslide_cache_capacity();
        set_openslide_cache_capacity(8 * 1024 * 1024).unwrap();
        assert_eq!(openslide_cache_capacity(), 8 * 1024 * 1024);
        assert_eq!(wsi.read_region(0, 0, 0, 128, 128).unwrap(), before);
        set_openslide_cache_capacity(capacity).unwrap();
        assert_eq!(recommended_openslide_cache_bytes(0), 32 * 1024 * 1024);
        assert_eq!(recommended_openslide_cache_bytes(1 << 30), 128 * 1024 * 1024);
    }

    #[test]
    fn gaps_and_areas_outside_the_slide_read_as_opaque_background() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sparse.tiff");
        let spec = TiffSpec {
            levels: vec![TiffLevel::new(64, 64, 32).with_missing_tile(1, 0)],
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, |_, _| [10, 20, 30]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        assert_eq!(wsi.properties().background_color, [255, 255, 255]);

        // A tile the scanner never wrote: transparent in OpenSlide.
        let gap = wsi.read_region(32, 0, 0, 32, 32).unwrap();
        assert!(gap.as_chunks::<4>().0.iter().all(|p| *p == [255, 255, 255, 255]));
        let raw = wsi
            .read_region_with(32, 0, 0, 32, 32, AlphaMode::Straight)
            .unwrap();
        assert!(raw.as_chunks::<4>().0.iter().all(|p| p[3] == 0));

        // Negative coordinates and the area past the edge are padded, not
        // shifted.
        let edge = wsi.read_region(-2, 62, 0, 4, 4).unwrap();
        let px = |x: usize, y: usize| &edge[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(px(0, 0), [255, 255, 255, 255]);
        assert_eq!(px(2, 0), [10, 20, 30, 255]);
        assert_eq!(px(2, 2), [255, 255, 255, 255]);
        let premul = wsi
            .read_region_with(-2, 62, 0, 4, 4, AlphaMode::Premultiplied)
            .unwrap();
        assert_eq!(&premul[..4], [0, 0, 0, 0]);
        assert_eq!(&premul[8..12], [10, 20, 30, 255]);
    }

    #[test]
    fn read_region_into_reuses_buffers_and_checks_their_size() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        let mut buf = vec![0u8; 8 * 8 * 4];
        wsi.read_region_into(0, 0, 1, 8, 8, AlphaMode::Composited, &mut buf)
            .unwrap();
        assert_eq!(buf, wsi.read_region(0, 0, 1, 8, 8).unwrap());
        // A misaligned buffer works too.
        let mut backing = vec![0u8; 8 * 8 * 4 + 1];
        wsi.read_region_into(0, 0, 1, 8, 8, AlphaMode::Composited, &mut backing[1..])
            .unwrap();
        assert_eq!(&backing[1..], &buf[..]);
        assert!(matches!(
            wsi.read_region_into(0, 0, 1, 8, 8, AlphaMode::Composited, &mut buf[..12]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(wsi.read_region(0, 0, 0, 0, 5).unwrap().is_empty());
    }

    /// The audit's reproduction: the gap between `openslide.region[0]` and
    /// `[1]` of `OS-2.bif` must read as white glass.
    #[test]
    #[ignore = "needs the OS-2.bif fixture in ../fixtures"]
    fn bif_region_gap_reads_as_white() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/OS-2.bif");
        let wsi = WsiFile::open(&path).unwrap();
        assert_eq!(wsi.properties().regions.len(), 2);
        let gap = wsi.read_region(15_000, 10_000, 0, 8, 8).unwrap();
        assert!(gap.as_chunks::<4>().0.iter().all(|p| *p == [255, 255, 255, 255]));
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
