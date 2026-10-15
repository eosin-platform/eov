//! Bounded thumbnail generation.
//!
//! The previous implementation decoded the *entire* coarsest level, which is
//! the whole image for single-level slides: a 100k x 80k flat TIFF meant a
//! 32 GB allocation. [`WsiFile::thumbnail_with`] instead tries, in order:
//!
//! 1. OpenSlide's `thumbnail` (or `macro`) associated image, when its aspect
//!    ratio shows that it depicts the same area as the slide and it is at
//!    least as large as the requested thumbnail;
//! 2. the coarsest pyramid level that is still at least the requested size,
//!    if reading it stays within [`ThumbnailOptions::max_source_pixels`];
//! 3. a too-small associated image, as is;
//! 4. sparse sampling of that level on a grid of at most
//!    [`ThumbnailOptions::max_sparse_samples`] points.
//!
//! Every path has bounded memory and I/O. The result carries its real
//! dimensions, so callers never have to guess them.

use super::{SlideBounds, WsiFile, pixels};
use crate::imaging::RgbaImageData;
use crate::{Error, Result};

/// Relative aspect-ratio difference up to which an associated image is
/// considered to show the same area as the slide.
const ASPECT_TOLERANCE: f64 = 0.02;

/// Which part of the slide a thumbnail shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThumbnailRegion {
    /// The full level-0 canvas.
    #[default]
    FullSlide,
    /// [`WsiProperties::content_bounds`](super::WsiProperties::content_bounds):
    /// the scanned area of MIRAX and other sparse slides.
    ContentBounds,
}

/// Options for [`WsiFile::thumbnail_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct ThumbnailOptions {
    /// Longest side of the thumbnail in pixels.
    pub max_size: u32,
    /// Area of the slide to show.
    pub region: ThumbnailRegion,
    /// Whether the `thumbnail`/`macro` associated images may be used.
    pub use_associated: bool,
    /// Largest pyramid read (in source pixels) allowed for a thumbnail.
    pub max_source_pixels: u64,
    /// Number of point samples used when no level is small enough.
    pub max_sparse_samples: u32,
}

impl Default for ThumbnailOptions {
    fn default() -> Self {
        Self {
            max_size: 256,
            region: ThumbnailRegion::FullSlide,
            use_associated: true,
            max_source_pixels: 16 * 1024 * 1024,
            max_sparse_samples: 64 * 64,
        }
    }
}

impl ThumbnailOptions {
    /// Default options with a given longest side.
    pub fn new(max_size: u32) -> Self {
        Self {
            max_size,
            ..Self::default()
        }
    }
}

/// Where a thumbnail's pixels came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThumbnailSource {
    /// An OpenSlide associated image (`thumbnail` or `macro`).
    AssociatedImage(String),
    /// A pyramid level, read in full and downscaled.
    Level(u32),
    /// Point samples of a level that was too large to read.
    Sampled { level: u32, samples: u32 },
}

/// A thumbnail and its provenance.
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// Opaque RGBA pixels; `width`/`height` are the real dimensions, which
    /// may be smaller than requested for sampled or small slides.
    pub image: RgbaImageData,
    pub source: ThumbnailSource,
    /// The level-0 rectangle the thumbnail shows.
    pub region: SlideBounds,
}

/// Largest size with the aspect ratio of `w` x `h` whose longest side is at
/// most `max_size` (and never larger than `w` x `h` itself).
fn fit(w: u64, h: u64, max_size: u32) -> (u32, u32) {
    let max_size = u64::from(max_size.max(1));
    if w <= max_size && h <= max_size {
        return (w.max(1) as u32, h.max(1) as u32);
    }
    if w >= h {
        let th = (h as f64 * max_size as f64 / w as f64).round().max(1.0);
        (max_size as u32, th as u32)
    } else {
        let tw = (w as f64 * max_size as f64 / h as f64).round().max(1.0);
        (tw as u32, max_size as u32)
    }
}

fn aspect_matches(a: (u64, u64), b: (u64, u64)) -> bool {
    if a.0 == 0 || a.1 == 0 || b.0 == 0 || b.1 == 0 {
        return false;
    }
    let ra = a.0 as f64 / a.1 as f64;
    let rb = b.0 as f64 / b.1 as f64;
    (ra / rb - 1.0).abs() <= ASPECT_TOLERANCE
}

fn resize(image: RgbaImageData, width: u32, height: u32) -> RgbaImageData {
    if image.width == width as usize && image.height == height as usize {
        return image;
    }
    let pixels = pixels::resize_area(
        &image.pixels,
        image.width,
        image.height,
        width as usize,
        height as usize,
    );
    RgbaImageData {
        width: width as usize,
        height: height as usize,
        pixels,
    }
}

impl WsiFile {
    /// A thumbnail whose longest side is at most `max_size`, with default
    /// [`ThumbnailOptions`]. Memory and I/O are bounded for every slide.
    pub fn thumbnail(&self, max_size: u32) -> Result<RgbaImageData> {
        self.thumbnail_with(&ThumbnailOptions::new(max_size))
            .map(|thumbnail| thumbnail.image)
    }

    /// A bounded thumbnail; see the [module documentation](self) for the
    /// strategy. Run it off the UI thread: it performs slide I/O.
    pub fn thumbnail_with(&self, options: &ThumbnailOptions) -> Result<Thumbnail> {
        // Real levels only: synthetic ones would make "cheap" reads expensive.
        let props = self.real_properties();
        let full = SlideBounds::full(props.width, props.height);
        let region = match options.region {
            ThumbnailRegion::FullSlide => full,
            ThumbnailRegion::ContentBounds => props.content_bounds(),
        };
        if region.is_empty() {
            return Err(Error::InvalidArgument("empty thumbnail region".to_string()));
        }
        let (tw, th) = fit(region.width, region.height, options.max_size);

        // 1. A suitable associated image.
        let mut small_associated = None;
        if options.use_associated && region == full {
            for name in ["thumbnail", "macro"] {
                let Some(info) = self.associated_image(name) else {
                    continue;
                };
                if !aspect_matches((info.width, info.height), (region.width, region.height)) {
                    continue;
                }
                if info.width >= u64::from(tw) && info.height >= u64::from(th) {
                    if let Ok(image) = self.read_associated_image(name) {
                        return Ok(Thumbnail {
                            image: resize(image, tw, th),
                            source: ThumbnailSource::AssociatedImage(name.to_string()),
                            region,
                        });
                    }
                } else if small_associated.is_none() {
                    small_associated = Some(name);
                }
            }
        }

        // 2. The coarsest level that is still at least the target size.
        let levels = &props.levels;
        let level_size = |level: &super::WsiLevel| {
            (
                (region.width as f64 / level.downsample).ceil() as u64,
                (region.height as f64 / level.downsample).ceil() as u64,
            )
        };
        let level = levels
            .iter()
            .rev()
            .find(|level| {
                let (w, h) = level_size(level);
                w >= u64::from(tw) && h >= u64::from(th)
            })
            .unwrap_or(&levels[0]);
        let (lw, lh) = level_size(level);
        if lw.saturating_mul(lh) <= options.max_source_pixels {
            let pixels = self.read_region(
                region.x,
                region.y,
                level.level,
                lw.max(1) as u32,
                lh.max(1) as u32,
            )?;
            let image = RgbaImageData {
                width: lw.max(1) as usize,
                height: lh.max(1) as usize,
                pixels,
            };
            return Ok(Thumbnail {
                image: resize(image, tw, th),
                source: ThumbnailSource::Level(level.level),
                region,
            });
        }

        // 3. An associated image that is smaller than requested.
        if let Some(name) = small_associated
            && let Ok(image) = self.read_associated_image(name)
        {
            return Ok(Thumbnail {
                image,
                source: ThumbnailSource::AssociatedImage(name.to_string()),
                region,
            });
        }

        // 4. Sparse sampling: one small read per output pixel.
        let samples = u64::from(options.max_sparse_samples.max(1));
        let (sw, sh) = if u64::from(tw) * u64::from(th) <= samples {
            (tw, th)
        } else {
            let scale = (samples as f64 / (f64::from(tw) * f64::from(th))).sqrt();
            (
                ((f64::from(tw) * scale).floor() as u32).max(1),
                ((f64::from(th) * scale).floor() as u32).max(1),
            )
        };
        let background = props.background_color;
        let mut pixels = Vec::with_capacity(sw as usize * sh as usize * 4);
        for j in 0..sh {
            for i in 0..sw {
                // Centre of the sample's footprint, in level-0 coordinates.
                let x = region.x as f64 + (f64::from(i) + 0.5) * region.width as f64 / f64::from(sw);
                let y =
                    region.y as f64 + (f64::from(j) + 0.5) * region.height as f64 / f64::from(sh);
                let px = self
                    .read_region(x.floor() as i64, y.floor() as i64, level.level, 1, 1)
                    .ok()
                    .and_then(|p| p.try_into().ok())
                    .unwrap_or([background[0], background[1], background[2], 255]);
                pixels.extend_from_slice(&px);
            }
        }
        Ok(Thumbnail {
            image: RgbaImageData {
                width: sw as usize,
                height: sh as usize,
                pixels,
            },
            source: ThumbnailSource::Sampled {
                level: level.level,
                samples: sw * sh,
            },
            region,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffSpec, write_tiff};

    #[test]
    fn fits_preserving_aspect_without_upscaling() {
        assert_eq!(fit(1000, 500, 100), (100, 50));
        assert_eq!(fit(500, 1000, 100), (50, 100));
        assert_eq!(fit(50, 20, 100), (50, 20));
        assert_eq!(fit(100_000, 1, 150), (150, 1));
        assert!(aspect_matches((761, 768), (35_855, 36_162)));
        assert!(aspect_matches((1024, 688), (114_943, 76_349)));
        assert!(!aspect_matches((1008, 3008), (114_943, 76_349)));
    }

    #[test]
    fn prefers_a_matching_associated_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.svs");
        write_tiff(&path, &TiffSpec::aperio(256, 192, 32, 4), |_, _| [1, 2, 3]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();

        // The associated thumbnail is 32x24 and has the slide's aspect.
        let thumb = wsi.thumbnail_with(&ThumbnailOptions::new(16)).unwrap();
        assert_eq!(thumb.source, ThumbnailSource::AssociatedImage("thumbnail".into()));
        assert_eq!((thumb.image.width, thumb.image.height), (16, 12));
        assert_eq!(&thumb.image.pixels[..4], [200, 180, 210, 255]);

        // Larger than the associated image: the pyramid is used instead.
        let thumb = wsi.thumbnail_with(&ThumbnailOptions::new(60)).unwrap();
        assert_eq!(thumb.source, ThumbnailSource::Level(1));
        assert_eq!((thumb.image.width, thumb.image.height), (60, 45));
        assert_eq!(&thumb.image.pixels[..4], [1, 2, 3, 255]);

        let mut options = ThumbnailOptions::new(16);
        options.use_associated = false;
        assert_eq!(wsi.thumbnail_with(&options).unwrap().source, ThumbnailSource::Level(1));
    }

    #[test]
    fn single_level_slides_are_sampled_within_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat.tiff");
        write_tiff(&path, &TiffSpec::single(512, 256, 64), |x, _| {
            if x < 256 { [255, 0, 0] } else { [0, 0, 255] }
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap();

        // Cheap enough: read the only level and downscale.
        let thumb = wsi.thumbnail_with(&ThumbnailOptions::new(64)).unwrap();
        assert_eq!(thumb.source, ThumbnailSource::Level(0));
        assert_eq!((thumb.image.width, thumb.image.height), (64, 32));

        // Over the source budget: sparse samples, never the whole level.
        let options = ThumbnailOptions {
            max_size: 64,
            max_source_pixels: 10_000,
            max_sparse_samples: 128,
            ..ThumbnailOptions::default()
        };
        let thumb = wsi.thumbnail_with(&options).unwrap();
        let ThumbnailSource::Sampled { level: 0, samples } = thumb.source else {
            panic!("expected sampling, got {:?}", thumb.source);
        };
        assert!(samples <= 128);
        assert_eq!((thumb.image.width * thumb.image.height) as u32, samples);
        assert_eq!(thumb.image.pixels.len(), samples as usize * 4);
        // Left half red, right half blue.
        assert_eq!(&thumb.image.pixels[..4], [255, 0, 0, 255]);
        let last = thumb.image.pixels.len() - 4;
        assert_eq!(&thumb.image.pixels[last..], [0, 0, 255, 255]);
        assert_eq!(wsi.thumbnail(64).unwrap().width, 64);
    }

    #[test]
    #[ignore = "needs the C3L-00088-22.svs and OS-2.bif fixtures in ../fixtures"]
    fn real_slides_use_their_associated_thumbnail() {
        let fixtures = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
        for name in ["C3L-00088-22.svs", "OS-2.bif"] {
            let wsi = WsiFile::open(fixtures.join(name)).unwrap();
            let thumb = wsi.thumbnail_with(&ThumbnailOptions::new(150)).unwrap();
            assert_eq!(thumb.source, ThumbnailSource::AssociatedImage("thumbnail".into()));
            assert_eq!(thumb.image.width.max(thumb.image.height), 150, "{name}");
        }
    }

    #[test]
    fn content_bounds_default_to_the_full_slide() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat.tiff");
        write_tiff(&path, &TiffSpec::single(128, 64, 64), |_, _| [9, 9, 9]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let options = ThumbnailOptions {
            region: ThumbnailRegion::ContentBounds,
            ..ThumbnailOptions::new(32)
        };
        let thumb = wsi.thumbnail_with(&options).unwrap();
        assert_eq!(thumb.region, SlideBounds::full(128, 64));
        assert_eq!((thumb.image.width, thumb.image.height), (32, 16));
    }
}
