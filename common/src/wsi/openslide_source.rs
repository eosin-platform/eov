//! The OpenSlide backend: brightfield formats through the shared,
//! self-healing OpenSlide handle ([`HandleCell`]).

use super::health::{BadKey, Geometry, HandleCell};
use super::source::{Opened, Region, SlideSource};
use super::{
    AlphaMode, AssociatedImage, MAX_ASSOCIATED_IMAGE_PIXELS, SlideHealth, WsiLevel, WsiProperties,
    metadata, pixels, registry,
};
use crate::imaging::RgbaImageData;
use crate::openslide::OpenSlide;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tracing::debug;

pub(crate) struct OpenSlideSource {
    handle: Arc<HandleCell>,
    background: [u8; 3],
}

/// Opens `path` with OpenSlide and reads its metadata.
pub(crate) fn open(path: &Path) -> Result<Opened> {
    let slide = OpenSlide::open(path)?;
    let (properties, property_map) = read_metadata(path, &slide)?;
    let associated_images = read_associated_images(&slide, &property_map)?;
    let geometry = Geometry::of(&slide)?;
    let handle = Arc::new(HandleCell::new(
        path.to_path_buf(),
        slide,
        geometry,
        registry::attach_global_cache,
    ));
    registry::register_openslide(&handle);
    Ok(Opened {
        source: Box::new(OpenSlideSource {
            handle,
            background: properties.background_color,
        }),
        properties,
        property_map,
        associated_images,
    })
}

fn read_associated_images(
    slide: &OpenSlide,
    property_map: &BTreeMap<String, String>,
) -> Result<Vec<AssociatedImage>> {
    let mut names = slide.associated_image_names()?;
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let (width, height) = slide.associated_image_dimensions(&name)?;
            let icc_profile_size = property_map
                .get(&format!("openslide.associated.{name}.icc-size"))
                .and_then(|value| value.trim().parse().ok())
                .filter(|&size: &u64| size > 0);
            Ok(AssociatedImage {
                name,
                width,
                height,
                icc_profile_size,
            })
        })
        .collect()
}

fn read_metadata(
    path: &Path,
    slide: &OpenSlide,
) -> Result<(WsiProperties, BTreeMap<String, String>)> {
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
            synthetic: false,
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

impl SlideSource for OpenSlideSource {
    fn read_region_into(&self, region: Region, mode: AlphaMode, out: &mut [u8]) -> Result<()> {
        let Region {
            x,
            y,
            level,
            width,
            height,
        } = region;
        let key = BadKey::Region {
            level,
            x,
            y,
            width,
            height,
        };
        // OpenSlide writes native-endian u32 ARGB. Let it write straight
        // into `out` when the buffer happens to be 4-byte aligned (it
        // practically always is), then convert in place.
        // SAFETY: every bit pattern is a valid u32 and u8.
        let (prefix, aligned, suffix) = unsafe { out.align_to_mut::<u32>() };
        if prefix.is_empty() && suffix.is_empty() {
            self.handle.run(Some(key), |slide| {
                slide.read_region_argb(aligned, x, y, level, width, height)
            })?;
            pixels::argb_bytes_to_rgba_in_place(out, mode, self.background);
        } else {
            let mut argb = vec![0u32; out.len() / 4];
            self.handle.run(Some(key), |slide| {
                slide.read_region_argb(&mut argb, x, y, level, width, height)
            })?;
            pixels::argb_to_rgba(&argb, out, mode, self.background);
        }
        Ok(())
    }

    fn read_associated_image(&self, name: &str, mode: AlphaMode) -> Result<RgbaImageData> {
        let (width, height, argb) = self
            .handle
            .run(Some(BadKey::AssociatedImage(name.to_string())), |slide| {
                slide.read_associated_image_argb(name, MAX_ASSOCIATED_IMAGE_PIXELS)
            })?;
        let mut out = pixels::try_alloc(argb.len() * 4)?;
        pixels::argb_to_rgba(&argb, &mut out, mode, self.background);
        Ok(RgbaImageData {
            width: width as usize,
            height: height as usize,
            pixels: out,
        })
    }

    fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        self.handle
            .run(Some(BadKey::IccProfile(None)), |slide| slide.icc_profile())
    }

    fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.handle
            .run(Some(BadKey::IccProfile(Some(name.to_string()))), |slide| {
                slide.associated_image_icc_profile(name)
            })
    }

    fn health(&self) -> SlideHealth {
        self.handle.health()
    }

    fn clear_bad_regions(&self) {
        self.handle.clear_bad_regions();
    }

    #[cfg(test)]
    fn openslide_handle(&self) -> Option<&HandleCell> {
        Some(&self.handle)
    }
}
