//! The backends a [`WsiFile`](super::WsiFile) reads slides through.
//!
//! [`open`] picks one per file: eov's OME-TIFF reader
//! ([`crate::ome_tiff`]) for OME-TIFF, recognised by its extension or by
//! OME-XML in the first ImageDescription, and OpenSlide
//! ([`super::openslide_source`]) for everything else. The rest of eov only
//! sees `WsiFile`: a backend reads regions of real levels and associated
//! images, and `WsiFile` builds synthetic levels, thumbnails and tiles on
//! top, the same way for every backend.

use super::{AlphaMode, AssociatedImage, SlideHealth, WsiProperties};
use crate::channels::{ChannelDisplay, ChannelSet};
use crate::imaging::RgbaImageData;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;
use tracing::{debug, warn};

/// A region of a real level: `x`, `y` in level-0 pixels (may be negative),
/// `width` x `height` pixels of `level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    pub x: i64,
    pub y: i64,
    pub level: u32,
    pub width: u32,
    pub height: u32,
}

/// Reads one slide. Shared by every clone of a `WsiFile` and called from
/// any thread, so implementations synchronise internally.
pub(crate) trait SlideSource: Send + Sync {
    /// Reads `region` into `out` (`width * height * 4` bytes) as RGBA in
    /// `mode`. The caller has checked the level and the size. Areas
    /// without image data read as transparent, or as the slide's
    /// background colour when composited.
    fn read_region_into(&self, region: Region, mode: AlphaMode, out: &mut [u8]) -> Result<()>;

    /// Decodes the associated image `name`, which the caller has checked
    /// exists and is not too large.
    fn read_associated_image(&self, name: &str, mode: AlphaMode) -> Result<RgbaImageData>;

    /// The slide's ICC profile.
    fn icc_profile(&self) -> Result<Option<Vec<u8>>>;

    /// The ICC profile of an associated image.
    fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>>;

    fn health(&self) -> SlideHealth;

    /// Forgets regions remembered as unreadable.
    fn clear_bad_regions(&self);

    /// The channels of a multichannel slide; `None` for RGB slides.
    fn channels(&self) -> Option<&ChannelSet> {
        None
    }

    /// How the channels are drawn now.
    fn channel_display(&self) -> Option<ChannelDisplay> {
        None
    }

    /// How the channels are drawn when the slide opens.
    fn default_channel_display(&self) -> Option<ChannelDisplay> {
        None
    }

    /// Focal planes the slide has (Z-stacks; 1 for most slides).
    fn focal_planes(&self) -> u32 {
        1
    }

    /// The focal plane reads come from.
    fn focal_plane(&self) -> u32 {
        0
    }

    /// Reads come from focal plane `z` from now on.
    fn set_focal_plane(&self, z: u32) -> Result<()> {
        if z == 0 {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "the slide has one focal plane".to_string(),
            ))
        }
    }

    /// Changes how the channels are drawn, for every later read.
    fn set_channel_display(&self, _display: ChannelDisplay) -> Result<()> {
        Err(Error::InvalidArgument(
            "the slide has no channels to display".to_string(),
        ))
    }

    /// The OpenSlide handle behind the source, for tests that poison it.
    #[cfg(test)]
    fn openslide_handle(&self) -> Option<&super::health::HandleCell> {
        None
    }
}

/// A slide opened by a backend: the source and what it read up front.
pub(crate) struct Opened {
    pub source: Box<dyn SlideSource>,
    pub properties: WsiProperties,
    pub property_map: BTreeMap<String, String>,
    pub associated_images: Vec<AssociatedImage>,
}

/// Opens `path` with the backend that reads it.
///
/// OME-TIFF goes to eov's reader. If that fails, OpenSlide is tried too
/// (it reads some RGB OME-TIFFs, e.g. with compressions eov does not
/// decode); when both fail, the OME-TIFF reader's error says more.
pub(crate) fn open(path: &Path) -> Result<Opened> {
    if crate::ome_tiff::has_ome_extension(path) || crate::ome_tiff::is_ome_tiff(path) {
        match crate::ome_tiff::open(path) {
            Ok(opened) => return Ok(opened),
            Err(ome_error) => {
                debug!("OME-TIFF reader failed: {ome_error}; trying OpenSlide");
                return match super::openslide_source::open(path) {
                    Ok(opened) => {
                        warn!(
                            "{} is read by OpenSlide, without its OME metadata: {ome_error}",
                            path.file_name().unwrap_or_default().to_string_lossy()
                        );
                        Ok(opened)
                    }
                    Err(_) => Err(ome_error),
                };
            }
        }
    }
    super::openslide_source::open(path)
}
