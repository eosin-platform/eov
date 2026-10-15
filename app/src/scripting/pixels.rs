//! `region.read`: pixels of an open slide, or of any slide file without
//! opening it in the window, at a pyramid level or any downsample.
//!
//! Reads run on the connection's thread (at most [`super::HEAVY_CALLS`]
//! at once), never on the UI thread. A region is at most
//! [`MAX_REGION_SIDE`] pixels wide and high and [`MAX_REGION_PIXELS`] in
//! all, which keeps the answer (base64) within the protocol's 64 MiB.

use super::params::{self, MAX_COORDINATE};
use super::{CallResult, ErrorKind, RpcError, Session};
use base64::Engine;
use eov_common::WsiFile;
use eov_common::wsi::AlphaMode;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Largest region returned, in pixels (32 MiB of RGBA, 43 MiB in base64).
pub(crate) const MAX_REGION_PIXELS: u64 = 8 << 20;
/// Largest width or height of a region.
pub(crate) const MAX_REGION_SIDE: u32 = 16384;
/// Largest region read from the slide before resampling to a downsample
/// between two levels (128 MiB of RGBA).
const MAX_SOURCE_PIXELS: u64 = 32 << 20;
/// Slides a connection keeps open after reading them by path.
const KEPT_SLIDES: usize = 4;

/// Slides read by path, most recently used last.
#[derive(Default)]
pub(super) struct SlideCache {
    slides: Vec<(PathBuf, WsiFile)>,
}

impl SlideCache {
    /// The slide at `path` (canonical), opened once per connection and
    /// opened again when the file changed.
    fn open(&mut self, path: &Path) -> Result<WsiFile, RpcError> {
        let unreadable =
            |error: eov_common::Error| RpcError::failed(format!("cannot read the slide: {error}"));
        let wsi = match self.slides.iter().position(|(kept, _)| kept == path) {
            Some(index) => self.slides.remove(index).1.reopen().map_err(unreadable)?,
            None => WsiFile::open(path).map_err(unreadable)?,
        };
        if self.slides.len() >= KEPT_SLIDES {
            self.slides.remove(0);
        }
        self.slides.push((path.to_path_buf(), wsi.clone()));
        Ok(wsi.without_synthetic_levels())
    }
}

/// How pixels come back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Format {
    /// A PNG file (RGB, or RGBA with `alpha`).
    #[default]
    Png,
    /// Raw RGBA, row by row (4 bytes per pixel).
    Rgba,
    /// Raw RGB, row by row (3 bytes per pixel).
    Rgb,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Rgba => "rgba",
            Format::Rgb => "rgb",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RegionRead {
    /// An open slide (`file_id` of `slides`); default: the focused one.
    #[serde(default)]
    slide: Option<i64>,
    /// Or a slide file, read without opening it in the window.
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    os_path: Option<String>,
    /// Top-left corner in level-0 pixels.
    x: f64,
    y: f64,
    /// Size in pixels of the result.
    width: u32,
    height: u32,
    /// A pyramid level, read as it is stored...
    #[serde(default)]
    level: Option<u32>,
    /// ...or level-0 pixels per result pixel (from the best level,
    /// resampled)...
    #[serde(default)]
    downsample: Option<f64>,
    /// ...or micrometres per result pixel. Default: level 0.
    #[serde(default)]
    mpp: Option<f64>,
    #[serde(default)]
    format: Format,
    /// Keep transparency where the slide has no data (instead of its
    /// background colour).
    #[serde(default)]
    alpha: bool,
}

/// A region to read, checked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Region {
    pub(super) x: i64,
    pub(super) y: i64,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) scale: Scale,
    pub(super) alpha: bool,
}

/// The resolution of a region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Scale {
    Level(u32),
    /// Level-0 pixels per result pixel.
    Downsample(f64),
    /// Micrometres per result pixel.
    Mpp(f64),
}

/// Pixels read for a region.
#[derive(Debug)]
pub(super) struct Pixels {
    /// RGBA, straight alpha.
    pub(super) rgba: Vec<u8>,
    /// The level read.
    pub(super) level: u32,
    /// Level-0 pixels per result pixel.
    pub(super) downsample: f64,
}

pub(super) fn region_read(session: &mut Session, params: RegionRead) -> CallResult {
    let region = checked_region(&params)?;
    let path = params::path_from(params.path.clone(), params.os_path.clone())?;
    if path.is_some() && params.slide.is_some() {
        return Err(RpcError::invalid("give slide or path, not both"));
    }
    if params.alpha && params.format == Format::Rgb {
        return Err(RpcError::invalid(
            "rgb has no alpha channel; ask for rgba or png",
        ));
    }
    let (wsi, mpp) = match path {
        Some(path) => {
            let path = params::slide_file(&path)?;
            let wsi = session.slides.open(&path)?;
            let mpp = crate::interaction::Calibration::new(
                wsi.properties().mpp_x,
                wsi.properties().mpp_y,
            )
            .average();
            (wsi, mpp)
        }
        None => {
            // The viewer's own handle, without the UI thread.
            let published = session.published();
            let (file_id, file) = super::published_slide(&published, params.slide)?;
            (
                session.open_slide(file_id)?,
                crate::interaction::Calibration::new(file.info.mpp_x, file.info.mpp_y).average(),
            )
        }
    };
    let _permit = super::HEAVY.acquire();
    let pixels = read(&wsi, &region, mpp)?;
    let data = encode(
        &pixels.rgba,
        region.width,
        region.height,
        params.format,
        params.alpha,
    )?;
    Ok(json!({
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "level": pixels.level,
        "downsample": pixels.downsample,
        "mpp": mpp.map(|mpp| mpp * pixels.downsample),
        "format": params.format.name(),
        "alpha": params.alpha,
        "data": base64::engine::general_purpose::STANDARD.encode(data),
    }))
}

/// Checks a request's size, position and resolution (the level is checked
/// against the slide when it is read).
fn checked_region(params: &RegionRead) -> Result<Region, RpcError> {
    let (width, height) = (params.width, params.height);
    if width == 0 || height == 0 {
        return Err(RpcError::invalid("width and height must be at least 1"));
    }
    if width > MAX_REGION_SIDE
        || height > MAX_REGION_SIDE
        || u64::from(width) * u64::from(height) > MAX_REGION_PIXELS
    {
        return Err(RpcError::too_large(format!(
            "a region is at most {MAX_REGION_SIDE} pixels wide and high and {MAX_REGION_PIXELS} \
             pixels in all ({width} × {height} asked); read it in tiles"
        )));
    }
    let coordinate = |value: f64, name: &str| {
        if value.is_finite() && value.abs() <= MAX_COORDINATE {
            Ok(value.round() as i64)
        } else {
            Err(RpcError::invalid(format!(
                "{name} is not a coordinate on a slide"
            )))
        }
    };
    let scale = match (params.level, params.downsample, params.mpp) {
        (Some(level), None, None) => Scale::Level(level),
        (None, Some(downsample), None) => Scale::Downsample(downsample),
        (None, None, Some(mpp)) if mpp.is_finite() && mpp > 0.0 => Scale::Mpp(mpp),
        (None, None, Some(_)) => return Err(RpcError::invalid("mpp must be a positive number")),
        (None, None, None) => Scale::Level(0),
        _ => {
            return Err(RpcError::invalid("give one of level, downsample and mpp"));
        }
    };
    Ok(Region {
        x: coordinate(params.x, "x")?,
        y: coordinate(params.y, "y")?,
        width,
        height,
        scale,
        alpha: params.alpha,
    })
}

/// Reads `region` of `wsi`, whose pixel size is `mpp` µm (a worker
/// thread).
pub(super) fn read(wsi: &WsiFile, region: &Region, mpp: Option<f64>) -> Result<Pixels, RpcError> {
    let failed =
        |error: eov_common::Error| RpcError::failed(format!("cannot read the region: {error}"));
    let mode = if region.alpha {
        AlphaMode::Premultiplied
    } else {
        AlphaMode::Composited
    };
    let level_count = wsi.level_count();
    let scale = match region.scale {
        Scale::Mpp(target) => Scale::Downsample(
            target
                / mpp.ok_or_else(|| {
                    RpcError::invalid("the slide records no pixel size; give level or downsample")
                })?,
        ),
        scale => scale,
    };
    let (level, downsample) = match scale {
        Scale::Level(level) => {
            let info = wsi.level(level).ok_or_else(|| {
                RpcError::invalid(format!(
                    "the slide has levels 0 to {}, not {level}",
                    level_count.saturating_sub(1)
                ))
            })?;
            (level, info.downsample)
        }
        Scale::Downsample(downsample) => {
            if !(downsample.is_finite() && (1.0..=MAX_COORDINATE).contains(&downsample)) {
                return Err(RpcError::invalid(format!(
                    "a downsample of {downsample} is not at least 1 (full resolution)"
                )));
            }
            let level = wsi.best_level_for_downsample(downsample);
            (level, downsample)
        }
        Scale::Mpp(_) => unreachable!("resolved above"),
    };
    let level_downsample = wsi.level(level).map_or(1.0, |info| info.downsample);
    let ratio = downsample / level_downsample;
    let mut rgba = if (ratio - 1.0).abs() < 1e-9 {
        wsi.read_region_with(region.x, region.y, level, region.width, region.height, mode)
            .map_err(failed)?
    } else {
        // Between two levels: read the finer one and resample.
        let source_width = (f64::from(region.width) * ratio).ceil().max(1.0);
        let source_height = (f64::from(region.height) * ratio).ceil().max(1.0);
        if source_width * source_height > MAX_SOURCE_PIXELS as f64 {
            return Err(RpcError::too_large(format!(
                "this downsample reads {} × {} pixels of level {level}, more than {MAX_SOURCE_PIXELS}; \
                 read a smaller region, or at a level",
                source_width, source_height
            )));
        }
        let (source_width, source_height) = (source_width as u32, source_height as u32);
        let source = wsi
            .read_region_with(region.x, region.y, level, source_width, source_height, mode)
            .map_err(failed)?;
        eov_common::dataset::resample_rgba(
            &source,
            source_width,
            source_height,
            region.width,
            region.height,
        )
    };
    if region.alpha {
        // Premultiplied (which resamples correctly) to straight.
        crate::host_ui::finish_alpha(&mut rgba, 0);
    }
    Ok(Pixels {
        rgba,
        level,
        downsample,
    })
}

/// The bytes of `format` for RGBA pixels.
fn encode(
    rgba: &[u8],
    width: u32,
    height: u32,
    format: Format,
    alpha: bool,
) -> Result<Vec<u8>, RpcError> {
    let rgb = || -> Vec<u8> {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect()
    };
    match format {
        Format::Rgba => Ok(rgba.to_vec()),
        Format::Rgb => Ok(rgb()),
        Format::Png => {
            use image::ImageEncoder;
            use image::codecs::png::{CompressionType, FilterType, PngEncoder};
            let mut png = Vec::new();
            let encoder =
                PngEncoder::new_with_quality(&mut png, CompressionType::Fast, FilterType::Adaptive);
            let written = if alpha {
                encoder.write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
            } else {
                encoder.write_image(&rgb(), width, height, image::ExtendedColorType::Rgb8)
            };
            written.map_err(|error| {
                RpcError::new(
                    ErrorKind::Internal,
                    format!("cannot encode the PNG: {error}"),
                )
            })?;
            Ok(png)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1024 × 512 slide with levels 1, 2 and 4, coloured by position.
    fn slide(dir: &Path) -> WsiFile {
        let path = dir.join("pyramid.tif");
        eov_common::test_support::write_tiff(
            &path,
            &eov_common::test_support::TiffSpec::pyramid(1024, 512, 256, 3),
            |x, y| [(x / 4) as u8, (y / 2) as u8, 77],
        )
        .unwrap();
        WsiFile::open(&path).unwrap()
    }

    fn region(x: i64, y: i64, width: u32, height: u32, scale: Scale) -> Region {
        Region {
            x,
            y,
            width,
            height,
            scale,
            alpha: false,
        }
    }

    fn params(json: serde_json::Value) -> RegionRead {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn levels_are_read_as_stored() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = slide(dir.path());
        let pixels = read(&wsi, &region(400, 200, 16, 8, Scale::Level(0)), None).unwrap();
        assert_eq!((pixels.level, pixels.downsample), (0, 1.0));
        assert_eq!(pixels.rgba.len(), 16 * 8 * 4);
        assert_eq!(&pixels.rgba[..4], &[100, 100, 77, 255]);
        // Level 1 halves: its pixel (0, 0) is level-0 (400, 200) here.
        let coarse = read(&wsi, &region(400, 200, 4, 4, Scale::Level(1)), None).unwrap();
        assert_eq!(coarse.downsample, 2.0);
        assert!(coarse.rgba[0].abs_diff(100) <= 1 && coarse.rgba[1].abs_diff(100) <= 1);
        assert_eq!(
            read(&wsi, &region(0, 0, 4, 4, Scale::Level(9)), None)
                .unwrap_err()
                .kind,
            ErrorKind::InvalidParams
        );
    }

    #[test]
    fn downsamples_between_levels_are_resampled() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = slide(dir.path());
        // Exactly a level: read as stored.
        let exact = read(&wsi, &region(0, 0, 64, 32, Scale::Downsample(4.0)), None).unwrap();
        assert_eq!((exact.level, exact.downsample), (2, 4.0));
        // Between levels 1 and 2: from level 1, resampled.
        let between = read(&wsi, &region(0, 0, 100, 50, Scale::Downsample(3.0)), None).unwrap();
        assert_eq!((between.level, between.downsample), (1, 3.0));
        assert_eq!(between.rgba.len(), 100 * 50 * 4);
        // Result pixel (50, 25) covers level-0 (150..153, 75..78).
        let at = (25 * 100 + 50) * 4;
        assert!(between.rgba[at].abs_diff(37) <= 2, "{}", between.rgba[at]);
        assert!(
            between.rgba[at + 1].abs_diff(38) <= 2,
            "{}",
            between.rgba[at + 1]
        );
        for bad in [0.5, f64::NAN, f64::INFINITY] {
            assert!(read(&wsi, &region(0, 0, 4, 4, Scale::Downsample(bad)), None).is_err());
        }
        // Too much to read for the result.
        let wide = region(0, 0, 6000, 6000, Scale::Downsample(1.9));
        assert_eq!(
            read(&wsi, &wide, None).unwrap_err().kind,
            ErrorKind::TooLarge
        );
        // Micrometres per pixel, from the slide's pixel size.
        let mpp = read(&wsi, &region(0, 0, 8, 8, Scale::Mpp(1.0)), Some(0.25)).unwrap();
        assert_eq!((mpp.level, mpp.downsample), (2, 4.0));
        assert_eq!(
            read(&wsi, &region(0, 0, 8, 8, Scale::Mpp(1.0)), None)
                .unwrap_err()
                .kind,
            ErrorKind::InvalidParams
        );
    }

    #[test]
    fn transparency_is_kept_on_request() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = slide(dir.path());
        // Half outside the slide.
        let mut outside = region(1016, 0, 16, 1, Scale::Level(0));
        let opaque = read(&wsi, &outside, None).unwrap();
        assert_eq!(opaque.rgba[15 * 4 + 3], 255, "composited by default");
        outside.alpha = true;
        let kept = read(&wsi, &outside, None).unwrap();
        assert_eq!(kept.rgba[15 * 4 + 3], 0);
        assert_eq!(kept.rgba[3], 255);
        assert_eq!(&kept.rgba[..3], &opaque.rgba[..3]);
    }

    #[test]
    fn regions_are_bounded() {
        let ok = params(json!({"x": 1.4, "y": 2.6, "width": 10, "height": 20}));
        let region = checked_region(&ok).unwrap();
        assert_eq!((region.x, region.y), (1, 3));
        assert_eq!(region.scale, Scale::Level(0));
        for (json, kind) in [
            (
                json!({"x": 0, "y": 0, "width": 0, "height": 1}),
                ErrorKind::InvalidParams,
            ),
            (
                json!({"x": 0, "y": 0, "width": 16385, "height": 1}),
                ErrorKind::TooLarge,
            ),
            (
                json!({"x": 0, "y": 0, "width": 4096, "height": 4096}),
                ErrorKind::TooLarge,
            ),
            (
                json!({"x": 1e300, "y": 0, "width": 1, "height": 1}),
                ErrorKind::InvalidParams,
            ),
            (
                json!({"x": 0, "y": 0, "width": 1, "height": 1, "level": 1, "downsample": 2}),
                ErrorKind::InvalidParams,
            ),
            (
                json!({"x": 0, "y": 0, "width": 1, "height": 1, "mpp": -1}),
                ErrorKind::InvalidParams,
            ),
        ] {
            assert_eq!(
                checked_region(&params(json.clone())).unwrap_err().kind,
                kind,
                "{json}"
            );
        }
        assert!(
            serde_json::from_value::<RegionRead>(
                json!({"x": 0, "y": 0, "width": 1, "height": 1, "zoom": 2})
            )
            .is_err(),
            "unknown parameters are refused"
        );
        // 4096 × 2048 is exactly the limit.
        assert!(
            checked_region(&params(
                json!({"x": 0, "y": 0, "width": 4096, "height": 2048})
            ))
            .is_ok()
        );
    }

    #[test]
    fn pixels_are_encoded_as_asked() {
        let rgba = [10, 20, 30, 255, 40, 50, 60, 128];
        assert_eq!(encode(&rgba, 2, 1, Format::Rgba, false).unwrap(), rgba);
        assert_eq!(
            encode(&rgba, 2, 1, Format::Rgb, false).unwrap(),
            [10, 20, 30, 40, 50, 60]
        );
        let png = encode(&rgba, 2, 1, Format::Png, false).unwrap();
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.color(), image::ColorType::Rgb8);
        assert_eq!(decoded.to_rgb8().into_raw(), [10, 20, 30, 40, 50, 60]);
        let png = encode(&rgba, 2, 1, Format::Png, true).unwrap();
        assert_eq!(
            image::load_from_memory(&png).unwrap().to_rgba8().into_raw(),
            rgba
        );
    }

    #[test]
    fn slides_read_by_path_are_kept_open() {
        let dir = tempfile::tempdir().unwrap();
        let first = slide(dir.path());
        let path = crate::paths::canonical(&first.properties().path);
        let mut cache = SlideCache::default();
        let a = cache.open(&path).unwrap();
        let b = cache.open(&path).unwrap();
        assert!(a.shares_handle_with(&b));
        assert_eq!(cache.slides.len(), 1);
        assert!(cache.open(&dir.path().join("missing.tif")).is_err());
    }
}
