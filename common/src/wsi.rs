//! WSI (Whole Slide Image) file handling
//!
//! [`WsiFile`] reads a slide through a backend ([`source`]): OpenSlide (see
//! [`crate::openslide`]) for brightfield formats, or eov's own reader for
//! OME-TIFF, including multichannel fluorescence ([`crate::channels`]). It
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
//! thread-safe backend per slide, and [`WsiFile::open`] reuses the slide
//! when the same unchanged file is already open anywhere in the process
//! (see [`registry`]). Give every worker thread a clone instead of opening
//! the slide again; all slides use one process-wide cache of native tiles
//! ([`set_openslide_cache_capacity`]).
//!
//! # Multichannel slides
//!
//! Reads of a multichannel slide return its channels composited to RGB with
//! the slide's [`ChannelDisplay`], which every clone shares. After
//! [`WsiFile::set_channel_display`], tiles cached from earlier reads show
//! the old display: drop them (e.g. [`TileCache::remove_file`]).
//!
//! [`TileCache::remove_file`]: crate::TileCache::remove_file
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

pub(crate) mod health;
pub(crate) mod metadata;
mod openslide_source;
pub(crate) mod pixels;
pub mod registry;
pub(crate) mod source;
mod synthetic;
mod thumbnail;

use crate::channels::{ChannelDisplay, ChannelSet};
use crate::imaging::RgbaImageData;
use crate::{Error, Result};
use source::{Region, SlideSource};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tracing::{debug, info};

pub use health::{HealthState, SlideHealth};
pub use pixels::AlphaMode;
pub use registry::{
    DEFAULT_OPENSLIDE_CACHE_BYTES, openslide_cache_capacity, recommended_openslide_cache_bytes,
    set_openslide_cache_capacity,
};
pub use synthetic::{
    EXACT_SYNTHESIS_BUDGET, SPARSE_SYNTHESIS_SAMPLES, SYNTHETIC_LEVEL_MAX_SIZE,
    SYNTHETIC_LEVEL_RATIO,
};
pub use thumbnail::{Thumbnail, ThumbnailOptions, ThumbnailRegion, ThumbnailSource};

/// [`WsiProperties::vendor`] (and `openslide.vendor`) of slides read by
/// eov's OME-TIFF reader.
pub use crate::ome_tiff::VENDOR as OME_TIFF_VENDOR;

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

/// Largest associated image [`WsiFile::read_associated_image`] decodes.
pub const MAX_ASSOCIATED_IMAGE_PIXELS: u64 = 1 << 28;

/// An associated image of a slide, such as `label`, `macro` or `thumbnail`.
///
/// The `label` image usually shows the case/accession number and sometimes
/// the patient's name: treat it as protected health information (offer a
/// "hide label" option, strip it from de-identified exports).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssociatedImage {
    pub name: String,
    pub width: u64,
    pub height: u64,
    /// Size of its ICC profile in bytes, if it has one.
    pub icc_profile_size: Option<u64>,
}

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
    /// `true` for levels eov synthesises from the coarsest real level (see
    /// [`WsiFile::with_synthetic_levels`]); they are not stored in the file.
    pub synthetic: bool,
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

/// Rounds `target` to the nearest positive multiple of a square native tile
/// size (64..=1024 px); other grids leave `target` unchanged.
pub fn aligned_tile_size(
    target: u32,
    native_width: Option<u32>,
    native_height: Option<u32>,
) -> u32 {
    match (native_width, native_height) {
        (Some(w), Some(h)) if w == h && (64..=COARSE_TILE_SIZE).contains(&w) => {
            let multiple = (f64::from(target) / f64::from(w)).round().max(1.0) as u32;
            w * multiple
        }
        _ => target,
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
    pub(crate) source: Box<dyn SlideSource>,
    pub(crate) canonical_path: PathBuf,
    pub(crate) stamp: registry::FileStamp,
    /// Whether [`WsiFile::open`] may hand this slide to other callers.
    pub(crate) shareable: bool,
    properties: WsiProperties,
    property_map: BTreeMap<String, String>,
    associated_images: Vec<AssociatedImage>,
    icc_profile: parking_lot::Mutex<Option<Option<Vec<u8>>>>,
    /// Properties of the view with synthetic levels, built on first use.
    synthetic_view: OnceLock<Arc<WsiProperties>>,
    /// Slide identity, computed on first use.
    identity: OnceLock<crate::file_id::SlideIdentity>,
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
    /// Levels exposed by this view: `None` for the file's real levels,
    /// `Some` for real plus synthetic levels.
    view: Option<Arc<WsiProperties>>,
    /// Tile size to use
    tile_size: u32,
}

impl std::fmt::Debug for WsiFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsiFile")
            .field("path", &self.shared.properties.path)
            .field("levels", &self.properties().levels.len())
            .field("tile_size", &self.tile_size)
            .finish_non_exhaustive()
    }
}

/// DICOM Microscopy Bulk Simple Annotations Storage.
const BULK_ANNOTATIONS_SOP_CLASS: &str = "1.2.840.10008.5.1.4.1.1.91.1";

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
                view: None,
                tile_size: DEFAULT_TILE_SIZE,
            });
        }

        // Paths can carry patient identifiers: the full path only at debug.
        info!(
            "Opening WSI file: {}",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        tracing::debug!("Opening WSI file: {}", path.display());
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
            && let Some(header) = crate::dataset::DicomHeader::read(path, 1 << 16)
            && let Some(reason) = header.non_slide_reason()
        {
            // Annotations exported next to a slide, reports: say what the
            // file is rather than that OpenSlide cannot read it.
            let hint = if header.sop_class_uid.as_deref() == Some(BULK_ANNOTATIONS_SOP_CLASS) {
                " (open its slide and import it with the annotations plugin)"
            } else {
                ""
            };
            return Err(Error::UnsupportedFormat(format!("{reason}{hint}")));
        }
        let source::Opened {
            source,
            properties,
            property_map,
            associated_images,
        } = source::open(path)?;

        info!(
            "Opened {} ({}x{}, {} levels, vendor: {:?})",
            properties.filename,
            properties.width,
            properties.height,
            properties.levels.len(),
            properties.vendor
        );

        let shared = Arc::new(SharedSlide {
            source,
            canonical_path,
            stamp,
            shareable: options.share,
            properties,
            property_map,
            associated_images,
            icc_profile: parking_lot::Mutex::new(None),
            synthetic_view: OnceLock::new(),
            identity: OnceLock::new(),
        });
        Ok(Self {
            shared: registry::register(shared),
            view: None,
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
        if self.has_synthetic_levels() {
            reopened = reopened.with_synthetic_levels();
        }
        Ok(reopened)
    }

    /// A view of this slide with synthetic low-resolution levels appended
    /// after the coarsest real level, for slides without a pyramid deep
    /// enough to draw at fit zoom (see [`SYNTHETIC_LEVEL_MAX_SIZE`]).
    ///
    /// Real level indices are unchanged; synthetic levels are flagged with
    /// [`WsiLevel::synthetic`]. Use the view for rendering (level selection,
    /// [`TileManager`](crate::TileManager)); keep the plain view for
    /// metadata displays and analysis, which should see real levels only.
    /// Slides whose pyramid is deep enough get no synthetic levels.
    ///
    /// Reading a synthetic level box-filters the coarsest real level:
    /// exactly when that decodes at most [`EXACT_SYNTHESIS_BUDGET`] base
    /// pixels (estimate with [`synthesis_cost`](Self::synthesis_cost)),
    /// otherwise approximately from at most [`SPARSE_SYNTHESIS_SAMPLES`]
    /// point samples. A [`TileManager`](crate::TileManager) with a tile
    /// cache builds coarse synthetic tiles exactly from cached tiles of the
    /// next finer level instead.
    pub fn with_synthetic_levels(&self) -> WsiFile {
        let view = self
            .shared
            .synthetic_view
            .get_or_init(|| {
                let mut props = self.shared.properties.clone();
                let extra = synthetic::synthetic_levels(&props.levels);
                props.levels.extend(extra);
                Arc::new(props)
            })
            .clone();
        WsiFile {
            shared: Arc::clone(&self.shared),
            view: (view.levels.len() > self.shared.properties.levels.len()).then_some(view),
            tile_size: self.tile_size,
        }
    }

    /// The same slide with only its real levels.
    pub fn without_synthetic_levels(&self) -> WsiFile {
        WsiFile {
            shared: Arc::clone(&self.shared),
            view: None,
            tile_size: self.tile_size,
        }
    }

    /// Whether this view includes synthetic levels.
    pub fn has_synthetic_levels(&self) -> bool {
        self.view.is_some()
    }

    /// Number of levels stored in the file (excluding synthetic ones).
    pub fn real_level_count(&self) -> u32 {
        self.shared.properties.levels.len() as u32
    }

    /// Properties with the file's real levels only, whatever the view.
    pub fn real_properties(&self) -> &WsiProperties {
        &self.shared.properties
    }

    /// Whether two `WsiFile`s share the same OpenSlide handle.
    pub fn shares_handle_with(&self, other: &WsiFile) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// The slide's stable [`SlideIdentity`](crate::file_id::SlideIdentity),
    /// the key for annotations and analysis results. Computed on first use
    /// (it samples every member file: call it off the UI thread) and cached
    /// for the lifetime of the open slide.
    pub fn identity(&self) -> std::io::Result<crate::file_id::SlideIdentity> {
        if let Some(identity) = self.shared.identity.get() {
            return Ok(identity.clone());
        }
        let identity = crate::file_id::SlideIdentity::compute(self)?;
        Ok(self.shared.identity.get_or_init(|| identity).clone())
    }

    /// A snapshot of the health of the slide's backend (for OpenSlide, its
    /// shared handle).
    pub fn health(&self) -> SlideHealth {
        self.shared.source.health()
    }

    /// Forgets the regions remembered as unreadable, so they are tried again
    /// (for example after the user restored the file or the network).
    pub fn clear_bad_regions(&self) {
        self.shared.source.clear_bad_regions();
    }

    /// The channels of a multichannel slide (OME-TIFF fluorescence), with
    /// their names, colours and sample statistics; `None` for RGB slides.
    pub fn channels(&self) -> Option<&ChannelSet> {
        self.shared.source.channels()
    }

    /// Whether the slide has channels that are composited for display.
    pub fn is_multichannel(&self) -> bool {
        self.channels().is_some()
    }

    /// How the channels are drawn now; shared by every clone of the slide.
    pub fn channel_display(&self) -> Option<ChannelDisplay> {
        self.shared.source.channel_display()
    }

    /// How the channels are drawn when the slide opens: colours from the
    /// file or the channel names, auto-contrast windows.
    pub fn default_channel_display(&self) -> Option<ChannelDisplay> {
        self.shared.source.default_channel_display()
    }

    /// Changes how the channels are drawn, for every reader of the slide.
    /// Windows and gammas are sanitised; the number of channels must match.
    ///
    /// Tiles read before the change keep the old display: drop them from
    /// any cache (see the [module documentation](self)).
    pub fn set_channel_display(&self, display: ChannelDisplay) -> Result<()> {
        self.shared.source.set_channel_display(display)
    }

    /// The focal planes of a Z-stack (OME-TIFF); 1 for most slides.
    pub fn focal_planes(&self) -> u32 {
        self.shared.source.focal_planes()
    }

    /// The focal plane shown (0 = the first).
    pub fn focal_plane(&self) -> u32 {
        self.shared.source.focal_plane()
    }

    /// Shows focal plane `z` from now on, for every reader of the slide.
    /// Cached tiles of the previous plane must be dropped by the caller (as
    /// after [`set_channel_display`](Self::set_channel_display)).
    pub fn set_focal_plane(&self, z: u32) -> Result<()> {
        self.shared.source.set_focal_plane(z)
    }

    /// Get file properties. `levels` includes synthetic levels when this
    /// is a [`with_synthetic_levels`](Self::with_synthetic_levels) view.
    pub fn properties(&self) -> &WsiProperties {
        self.view.as_deref().unwrap_or(&self.shared.properties)
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

    /// The slide's associated images (label, macro, thumbnail, ...), sorted
    /// by name.
    pub fn associated_images(&self) -> &[AssociatedImage] {
        &self.shared.associated_images
    }

    /// The associated image called `name`, if the slide has one.
    pub fn associated_image(&self, name: &str) -> Option<&AssociatedImage> {
        self.shared
            .associated_images
            .iter()
            .find(|image| image.name == name)
    }

    /// Decodes an associated image as opaque RGBA (composited onto the
    /// slide's background colour like [`read_region`](Self::read_region)).
    pub fn read_associated_image(&self, name: &str) -> Result<RgbaImageData> {
        self.read_associated_image_with(name, AlphaMode::Composited)
    }

    /// Decodes an associated image with an explicit [`AlphaMode`].
    pub fn read_associated_image_with(&self, name: &str, mode: AlphaMode) -> Result<RgbaImageData> {
        let info = self
            .associated_image(name)
            .ok_or_else(|| Error::AssociatedImageNotFound(name.to_string()))?;
        let pixels = info.width.saturating_mul(info.height);
        if pixels > MAX_ASSOCIATED_IMAGE_PIXELS {
            return Err(Error::RegionTooLarge {
                width: info.width,
                height: info.height,
                max_pixels: MAX_ASSOCIATED_IMAGE_PIXELS,
            });
        }
        self.shared.source.read_associated_image(name, mode)
    }

    /// The slide's ICC colour profile, if it has one. Pixels are returned
    /// in the device colour space this profile describes. Cached after the
    /// first successful read.
    pub fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        if let Some(cached) = self.shared.icc_profile.lock().as_ref() {
            return Ok(cached.clone());
        }
        if self.shared.properties.icc_profile_size.is_none() {
            return Ok(None);
        }
        let profile = self.shared.source.icc_profile()?;
        *self.shared.icc_profile.lock() = Some(profile.clone());
        Ok(profile)
    }

    /// The ICC profile of an associated image, if it has one.
    pub fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let info = self
            .associated_image(name)
            .ok_or_else(|| Error::AssociatedImageNotFound(name.to_string()))?;
        if info.icc_profile_size.is_none() {
            return Ok(None);
        }
        self.shared.source.associated_image_icc_profile(name)
    }

    /// Get tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// The tile size eov uses to address `level`.
    ///
    /// Coarser levels use larger tiles (256, 512 or 1024 px by downsample),
    /// rounded to a multiple of the file's native tile size when the format
    /// has a square native grid of at most 1024 px (e.g. 240 px for Aperio
    /// gives 240/480/960), so each native tile is decoded for exactly one eov
    /// tile instead of being split across up to four.
    pub fn tile_size_for_level(&self, level: u32) -> u32 {
        let Some(level_info) = self.level(level) else {
            return self.tile_size;
        };
        if level_info.synthetic {
            // Synthetic tiles are computed, not decoded: small tiles keep
            // each one cheap.
            return DEFAULT_TILE_SIZE;
        }

        let target = if level_info.downsample >= 16.0 {
            COARSE_TILE_SIZE
        } else if level_info.downsample >= 4.0 {
            MEDIUM_TILE_SIZE
        } else {
            DEFAULT_TILE_SIZE
        };
        aligned_tile_size(target, level_info.tile_width, level_info.tile_height)
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
        self.properties().levels.len() as u32
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
        best_level_for_downsample(&self.properties().levels, downsample)
    }

    /// Get level information
    pub fn level(&self, level: u32) -> Option<&WsiLevel> {
        self.properties().levels.get(level as usize)
    }

    /// Reads a region as opaque RGBA (`width * height * 4` bytes).
    ///
    /// OpenSlide's premultiplied pixels are composited over the slide's
    /// background colour ([`WsiProperties::background_color`], white unless
    /// the slide specifies one), so gaps between scanned regions and areas
    /// outside the slide read as glass, never as black or transparent. Use
    /// [`read_region_with`](Self::read_region_with) to keep the alpha channel.
    ///
    /// Multichannel slides return their visible channels composited with
    /// the current [`channel_display`](Self::channel_display), over a black
    /// background.
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
        if level >= self.real_level_count() {
            return self.read_synthetic_into(x, y, level, width, height, mode, out);
        }
        let region = Region {
            x,
            y,
            level,
            width,
            height,
        };
        self.shared.source.read_region_into(region, mode, out)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openslide::OpenSlide;
    use crate::test_support::{TiffLevel, TiffSpec, write_tiff};

    fn gradient(x: u32, y: u32) -> [u8; 3] {
        [
            (x % 251) as u8,
            (y % 241) as u8,
            ((x * 7 + y * 3) % 256) as u8,
        ]
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
    fn dicom_annotations_are_not_opened_as_slides() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.annotations.dcm");
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        bytes.extend_from_slice(&[0x02, 0x00, 0x02, 0x00, b'U', b'I']);
        bytes.extend_from_slice(&(BULK_ANNOTATIONS_SOP_CLASS.len() as u16).to_le_bytes());
        bytes.extend_from_slice(BULK_ANNOTATIONS_SOP_CLASS.as_bytes());
        std::fs::write(&path, bytes).unwrap();
        let error = WsiFile::open(&path).unwrap_err().to_string();
        assert!(
            error.contains("DICOM Microscopy Bulk Simple Annotations, not a whole-slide image")
                && error.contains("annotations plugin"),
            "{error}"
        );
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
    fn tile_sizes_follow_the_native_grid() {
        assert_eq!(aligned_tile_size(256, Some(240), Some(240)), 240);
        assert_eq!(aligned_tile_size(512, Some(240), Some(240)), 480);
        assert_eq!(aligned_tile_size(1024, Some(240), Some(240)), 960);
        assert_eq!(aligned_tile_size(256, Some(512), Some(512)), 512);
        assert_eq!(aligned_tile_size(1024, Some(512), Some(512)), 1024);
        assert_eq!(aligned_tile_size(256, Some(256), Some(256)), 256);
        // Non-square, tiny, huge or unknown grids are not followed.
        assert_eq!(aligned_tile_size(256, Some(240), Some(120)), 256);
        assert_eq!(aligned_tile_size(256, Some(16), Some(16)), 256);
        assert_eq!(aligned_tile_size(256, Some(4096), Some(4096)), 256);
        assert_eq!(aligned_tile_size(512, None, None), 512);

        let dir = tempfile::tempdir().unwrap();
        let wsi = WsiFile::open(pyramid_file(dir.path())).unwrap();
        // Native 64-px tiles: 256 = 4 * 64.
        assert_eq!(wsi.tile_size_for_level(0), 256);
        assert_eq!(wsi.tile_size_for_level(2), 512);
        assert_eq!(wsi.tile_size_for_level(99), wsi.tile_size());
    }

    #[test]
    fn best_level_matches_openslide() {
        let dir = tempfile::tempdir().unwrap();
        let path = pyramid_file(dir.path());
        let wsi = WsiFile::open(&path).unwrap();
        let raw = OpenSlide::open(&path).unwrap();
        for ds in [
            0.1,
            0.5,
            1.0,
            1.5,
            1.999,
            2.0,
            2.5,
            3.99,
            4.0,
            4.1,
            100.0,
            f64::INFINITY,
        ] {
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
        assert!(matches!(
            WsiFile::open(&text),
            Err(Error::UnsupportedFormat(_))
        ));
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
        assert!(
            matches!(
                err,
                Error::CorruptRegion {
                    level: 0,
                    x: 32,
                    y: 32,
                    ..
                }
            ),
            "{err:?}"
        );
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
            let slot = a.shared.source.openslide_handle().unwrap().current();
            let mut buf = vec![0u32; 32 * 32];
            assert!(
                slot.slide
                    .read_region_argb(&mut buf, 32, 32, 0, 32, 32)
                    .is_err()
            );
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
    fn a_burst_of_bad_tiles_leaves_the_shared_handle_usable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.tiff");
        // Like a truncated copy: the bottom half of level 0 is unreadable.
        let mut level0 = TiffLevel::new(256, 256, 32);
        for ty in 4..8 {
            for tx in 0..8 {
                level0 = level0.with_broken_tile(tx, ty);
            }
        }
        let spec = TiffSpec {
            levels: vec![level0, TiffLevel::new(128, 128, 32)],
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, gradient).unwrap();
        let wsi = WsiFile::open(&path).unwrap();

        for ty in 4..8 {
            for tx in 0..8 {
                let read = wsi.read_region(tx * 32, ty * 32, 0, 32, 32);
                assert!(read.is_err(), "tile {tx},{ty}");
            }
            // Good tiles keep reading between the bad ones.
            assert!(wsi.read_region(0, 0, 0, 32, 32).is_ok());
        }
        let health = wsi.health();
        assert_eq!(health.state, HealthState::Healthy);
        assert!(health.bad_regions >= 16, "{health:?}");
        assert!(wsi.read_region(0, 0, 1, 128, 128).is_ok());
    }

    #[test]
    fn unverified_failures_fail_fast_until_a_probe_confirms_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.tiff");
        let mut level0 = TiffLevel::new(256, 256, 32);
        for ty in 3..8 {
            for tx in 0..8 {
                level0 = level0.with_broken_tile(tx, ty);
            }
        }
        let spec = TiffSpec {
            levels: vec![level0, TiffLevel::new(128, 128, 32)],
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, gradient).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let read = |tx: i64, ty: i64| wsi.read_region(tx * 32, ty * 32, 0, 32, 32);

        // 40 distinct bad tiles use up the probe budget (32 tokens).
        let mut confirmed = 0;
        let mut provisional = Vec::new();
        for ty in 3..8 {
            for tx in 0..8 {
                match read(tx, ty) {
                    Err(Error::CorruptRegion { .. }) => confirmed += 1,
                    Err(Error::HandlePoisoned(_)) => provisional.push((tx, ty)),
                    other => panic!("unexpected {other:?}"),
                }
                // The shared handle is restored every time.
                assert!(read(0, 0).is_ok());
            }
        }
        assert!(confirmed >= 32, "{confirmed}");
        assert!(!provisional.is_empty());
        let (tx, ty) = provisional[0];
        // While provisional, the region fails fast (no reopen) and the error
        // stays transient.
        let reopens = wsi.health().reopens;
        assert!(matches!(read(tx, ty), Err(Error::HandlePoisoned(_))));
        assert_eq!(wsi.health().reopens, reopens, "fail-fast must not reopen");
        // Later, with probe budget again, the region is read, probed and
        // confirmed.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(matches!(read(tx, ty), Err(Error::CorruptRegion { .. })));
        assert!(matches!(read(tx, ty), Err(Error::CorruptRegion { .. })));
        assert!(read(0, 0).is_ok());
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
        assert_eq!(
            recommended_openslide_cache_bytes(1 << 30),
            128 * 1024 * 1024
        );
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
        assert!(
            gap.as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [255, 255, 255, 255])
        );
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
        assert!(
            gap.as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [255, 255, 255, 255])
        );
    }

    #[test]
    fn exposes_associated_images_and_icc_profiles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.svs");
        let mut spec = TiffSpec::aperio(128, 96, 32, 4);
        spec.icc_profile = Some(vec![7; 300]);
        write_tiff(&path, &spec, gradient).unwrap();
        let wsi = WsiFile::open(&path).unwrap();

        let names: Vec<_> = wsi
            .associated_images()
            .iter()
            .map(|i| i.name.as_str())
            .collect();
        assert_eq!(names, ["label", "macro", "thumbnail"]);
        let label = wsi.associated_image("label").unwrap();
        assert_eq!((label.width, label.height), (24, 16));
        assert_eq!(label.icc_profile_size, None);

        let image = wsi.read_associated_image("label").unwrap();
        assert_eq!((image.width, image.height), (24, 16));
        assert!(
            image
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [10, 20, 30, 255])
        );
        assert!(matches!(
            wsi.read_associated_image("barcode"),
            Err(Error::AssociatedImageNotFound(_))
        ));

        assert_eq!(wsi.properties().icc_profile_size, Some(300));
        assert_eq!(wsi.icc_profile().unwrap(), Some(vec![7; 300]));
        assert_eq!(wsi.icc_profile().unwrap(), Some(vec![7; 300]), "cached");
        assert_eq!(wsi.associated_image_icc_profile("label").unwrap(), None);
        assert!(wsi.associated_image_icc_profile("barcode").is_err());

        // Slides without a profile.
        let plain = WsiFile::open(pyramid_file(dir.path())).unwrap();
        assert_eq!(plain.icc_profile().unwrap(), None);
        assert!(plain.associated_images().is_empty());
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
