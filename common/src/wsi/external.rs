//! Slide readers from outside this crate: the slide sources of plugins (see
//! the app's `plugin_runtime::slide_sources`). [`register_opener`] makes a
//! reader's file extensions known; [`super::source::open`] asks it for files
//! with those extensions that eov's own readers cannot open (they keep the
//! formats they read).
//!
//! A reader gives the slide's metadata once ([`ExternalSlideInfo`]) and reads
//! regions of a level in that level's pixels as straight RGBA; this module
//! turns that into a [`SlideSource`]: level-0 coordinates, the parts outside
//! the slide as "no data", large reads in parts, and the alpha modes.

use super::source::{Opened, Region, SlideSource};
use super::{AlphaMode, AssociatedImage, SlideBounds, SlideHealth, WsiLevel, WsiProperties};
use crate::imaging::RgbaImageData;
use crate::{Error, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest region, in pixels, asked of a reader at once (larger reads arrive
/// in parts).
pub const MAX_EXTERNAL_READ_PIXELS: u64 = 4096 * 4096;

/// One pyramid level of an external slide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalLevel {
    pub width: u64,
    pub height: u64,
    /// Level-0 size over this level's (1 for level 0).
    pub downsample: f64,
    pub tile_width: Option<u32>,
    pub tile_height: Option<u32>,
}

/// What an external reader says about a slide, already checked by the
/// opener (levels that make sense, sizes in range).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalSlideInfo {
    /// Full resolution first.
    pub levels: Vec<ExternalLevel>,
    /// `openslide.vendor`: the scanner or format.
    pub vendor: String,
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    pub objective_power: Option<f64>,
    pub scan_date: Option<String>,
    pub comment: Option<String>,
    pub stain: Option<String>,
    /// Colour of areas without image data.
    pub background: [u8; 3],
    /// The level-0 rectangle with image data, when smaller than the slide.
    pub bounds: Option<SlideBounds>,
    /// Every property, shown in Slide Information (`openslide.*` keys are
    /// eov's own and set from the fields above).
    pub properties: BTreeMap<String, String>,
    /// Name, width and height of each associated image.
    pub associated_images: Vec<(String, u32, u32)>,
}

/// An open external slide. Read from several threads at once.
pub trait ExternalSlide: Send + Sync {
    /// Fills `out` (`width·height·4` bytes, zeroed: transparent) with the
    /// straight RGBA pixels of `level` at (`x`, `y`), in that level's
    /// pixels. The region lies inside the level and has at most
    /// [`MAX_EXTERNAL_READ_PIXELS`] pixels.
    fn read_region(
        &self,
        level: u32,
        x: u64,
        y: u64,
        width: u32,
        height: u32,
        out: &mut [u8],
    ) -> std::result::Result<(), String>;

    /// The associated image `name` as straight RGBA: width, height, pixels.
    fn read_associated_image(&self, name: &str)
    -> std::result::Result<(u32, u32, Vec<u8>), String>;
}

/// Opens the slides of some formats (a plugin's slide source).
pub trait ExternalOpener: Send + Sync {
    /// Who reads them, for messages ("the plugin 'iSyntax reader'").
    fn reader_name(&self) -> String;

    /// File extensions it reads, in lower case without the dot.
    fn extensions(&self) -> Vec<String>;

    /// Opens `path`, a file with one of [`extensions`](Self::extensions).
    fn open(
        &self,
        path: &Path,
    ) -> std::result::Result<(ExternalSlideInfo, Box<dyn ExternalSlide>), String>;
}

type Openers = Vec<(String, Arc<dyn ExternalOpener>)>;

static OPENERS: RwLock<Openers> = RwLock::new(Vec::new());

/// Makes `opener` read its formats, under `key` (the plugin id), replacing
/// an earlier opener with that key.
pub fn register_opener(key: &str, opener: Arc<dyn ExternalOpener>) {
    let mut openers = OPENERS.write();
    openers.retain(|(existing, _)| existing != key);
    openers.push((key.to_string(), opener));
    openers.sort_by(|a, b| a.0.cmp(&b.0));
}

/// Forgets the opener registered under `key` (the plugin stopped). Slides it
/// opened keep their readers until they are closed.
pub fn unregister_opener(key: &str) {
    OPENERS.write().retain(|(existing, _)| existing != key);
}

/// Every file extension an external reader reads (lower case, sorted).
pub fn external_extensions() -> Vec<String> {
    let mut extensions: Vec<String> = OPENERS
        .read()
        .iter()
        .flat_map(|(_, opener)| opener.extensions())
        .collect();
    extensions.sort();
    extensions.dedup();
    extensions
}

/// The opener for `path`'s extension, the first by key when several claim
/// it.
pub(crate) fn opener_for(path: &Path) -> Option<Arc<dyn ExternalOpener>> {
    let extension = path.extension()?.to_string_lossy().to_ascii_lowercase();
    OPENERS
        .read()
        .iter()
        .find(|(_, opener)| opener.extensions().contains(&extension))
        .map(|(_, opener)| Arc::clone(opener))
}

/// Opens `path` with `opener`.
pub(crate) fn open(path: &Path, opener: &dyn ExternalOpener) -> Result<Opened> {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (info, slide) = opener.open(path).map_err(|error| {
        Error::OpenFile(format!(
            "{file_name}: {} could not open it: {error}",
            opener.reader_name()
        ))
    })?;
    if info.levels.is_empty() {
        return Err(Error::InvalidFormat(format!(
            "{file_name}: {} gave no levels",
            opener.reader_name()
        )));
    }
    let base = info.levels[0];
    let properties = WsiProperties {
        path: path.to_path_buf(),
        filename: file_name,
        vendor: Some(info.vendor.clone()),
        mpp_x: info.mpp_x,
        mpp_y: info.mpp_y,
        objective_power: info.objective_power,
        scan_date: info.scan_date.clone(),
        stain: info.stain.clone(),
        levels: info
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| WsiLevel {
                level: index as u32,
                width: level.width,
                height: level.height,
                downsample: level.downsample,
                tile_width: level.tile_width,
                tile_height: level.tile_height,
                synthetic: false,
            })
            .collect(),
        width: base.width,
        height: base.height,
        bounds: info.bounds,
        regions: Vec::new(),
        background_color: info.background,
        quickhash1: None,
        comment: info.comment.clone(),
        icc_profile_size: None,
    };
    let mut property_map = info.properties.clone();
    property_map.retain(|key, _| !key.starts_with("openslide."));
    let mut set = |key: &str, value: String| {
        property_map.insert(key.to_string(), value);
    };
    set("openslide.vendor", info.vendor.clone());
    set("openslide.level-count", info.levels.len().to_string());
    for (index, level) in info.levels.iter().enumerate() {
        let prefix = format!("openslide.level[{index}]");
        set(&format!("{prefix}.width"), level.width.to_string());
        set(&format!("{prefix}.height"), level.height.to_string());
        set(
            &format!("{prefix}.downsample"),
            level.downsample.to_string(),
        );
        if let (Some(w), Some(h)) = (level.tile_width, level.tile_height) {
            set(&format!("{prefix}.tile-width"), w.to_string());
            set(&format!("{prefix}.tile-height"), h.to_string());
        }
    }
    if let Some(mpp) = info.mpp_x {
        set("openslide.mpp-x", mpp.to_string());
    }
    if let Some(mpp) = info.mpp_y {
        set("openslide.mpp-y", mpp.to_string());
    }
    if let Some(power) = info.objective_power {
        set("openslide.objective-power", power.to_string());
    }
    let [r, g, b] = info.background;
    set(
        "openslide.background-color",
        format!("{r:02X}{g:02X}{b:02X}"),
    );
    for (name, width, height) in &info.associated_images {
        set(
            &format!("openslide.associated.{name}.width"),
            width.to_string(),
        );
        set(
            &format!("openslide.associated.{name}.height"),
            height.to_string(),
        );
    }
    let associated_images = info
        .associated_images
        .iter()
        .map(|(name, width, height)| AssociatedImage {
            name: name.clone(),
            width: u64::from(*width),
            height: u64::from(*height),
            icc_profile_size: None,
        })
        .collect();
    Ok(Opened {
        source: Box::new(ExternalSource {
            reader: opener.reader_name(),
            info,
            slide,
            failing: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }),
        properties,
        property_map,
        associated_images,
    })
}

struct ExternalSource {
    reader: String,
    info: ExternalSlideInfo,
    slide: Box<dyn ExternalSlide>,
    /// The last read failed.
    failing: AtomicBool,
    last_error: Mutex<Option<String>>,
}

/// A level-0 coordinate in pixels of a level of `downsample` (as the
/// OME-TIFF reader maps them).
fn level_coordinate(x: i64, downsample: f64) -> i64 {
    if downsample == 1.0 {
        return x;
    }
    ((x as f64 - 0.5) / downsample - 1e-9).ceil() as i64
}

impl ExternalSource {
    fn fail(&self, message: String) -> Error {
        self.failing.store(true, Ordering::Relaxed);
        *self.last_error.lock() = Some(message.clone());
        Error::OpenFile(format!("{}: {message}", self.reader))
    }
}

impl SlideSource for ExternalSource {
    fn read_region_into(&self, region: Region, mode: AlphaMode, out: &mut [u8]) -> Result<()> {
        let level = self.info.levels[region.level as usize];
        let background = self.info.background;
        let no_data = match mode {
            AlphaMode::Composited => [background[0], background[1], background[2], 255],
            AlphaMode::Straight | AlphaMode::Premultiplied => [0, 0, 0, 0],
        };
        for px in out.as_chunks_mut::<4>().0 {
            *px = no_data;
        }
        // The part inside the level, in level pixels.
        let x0 = level_coordinate(region.x, level.downsample);
        let y0 = level_coordinate(region.y, level.downsample);
        let (w, h) = (i64::from(region.width), i64::from(region.height));
        let (left, top) = (x0.max(0), y0.max(0));
        let right = (x0 + w).min(level.width as i64);
        let bottom = (y0 + h).min(level.height as i64);
        if right <= left || bottom <= top {
            return Ok(());
        }
        let clip_width = (right - left) as u32;
        // Rows per part so a part has at most MAX_EXTERNAL_READ_PIXELS.
        let rows_per_part =
            (MAX_EXTERNAL_READ_PIXELS / u64::from(clip_width)).clamp(1, 4096) as i64;
        let mut row = top;
        let mut part = Vec::new();
        while row < bottom {
            let rows = (bottom - row).min(rows_per_part);
            let len = clip_width as usize * rows as usize * 4;
            part.clear();
            part.resize(len, 0);
            self.slide
                .read_region(
                    region.level,
                    left as u64,
                    row as u64,
                    clip_width,
                    rows as u32,
                    &mut part,
                )
                .map_err(|error| self.fail(error))?;
            for r in 0..rows {
                let src = &part[r as usize * clip_width as usize * 4..][..clip_width as usize * 4];
                let out_row = (row + r - y0) as usize;
                let out_col = (left - x0) as usize;
                let dst = &mut out[(out_row * region.width as usize + out_col) * 4..]
                    [..clip_width as usize * 4];
                for (d, s) in dst
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(src.as_chunks::<4>().0)
                {
                    *d = convert(*s, mode, background);
                }
            }
            row += rows;
        }
        self.failing.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn read_associated_image(&self, name: &str, mode: AlphaMode) -> Result<RgbaImageData> {
        let Some(&(_, width, height)) = self
            .info
            .associated_images
            .iter()
            .find(|(listed, _, _)| listed == name)
        else {
            return Err(Error::AssociatedImageNotFound(name.to_string()));
        };
        let (got_width, got_height, mut pixels) = self
            .slide
            .read_associated_image(name)
            .map_err(|error| self.fail(error))?;
        if (got_width, got_height) != (width, height)
            || pixels.len() != width as usize * height as usize * 4
        {
            return Err(self.fail(format!(
                "the associated image '{name}' came as {got_width}×{got_height} ({} bytes), not the {width}×{height} listed",
                pixels.len()
            )));
        }
        for px in pixels.as_chunks_mut::<4>().0 {
            *px = convert(*px, mode, self.info.background);
        }
        Ok(RgbaImageData {
            width: width as usize,
            height: height as usize,
            pixels,
        })
    }

    fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn associated_image_icc_profile(&self, _name: &str) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn health(&self) -> SlideHealth {
        SlideHealth {
            state: if self.failing.load(Ordering::Relaxed) {
                super::health::HealthState::Failing
            } else {
                super::health::HealthState::Healthy
            },
            last_error: self.last_error.lock().clone(),
            ..SlideHealth::default()
        }
    }

    fn clear_bad_regions(&self) {}
}

/// A straight RGBA pixel in `mode`.
fn convert([r, g, b, a]: [u8; 4], mode: AlphaMode, background: [u8; 3]) -> [u8; 4] {
    match mode {
        AlphaMode::Straight => {
            if a == 0 {
                [0, 0, 0, 0]
            } else {
                [r, g, b, a]
            }
        }
        AlphaMode::Premultiplied => {
            let mul = |c: u8| ((u16::from(c) * u16::from(a) + 127) / 255) as u8;
            [mul(r), mul(g), mul(b), a]
        }
        AlphaMode::Composited => {
            let blend = |c: u8, bg: u8| {
                ((u16::from(c) * u16::from(a) + u16::from(bg) * (255 - u16::from(a)) + 127) / 255)
                    as u8
            };
            [
                blend(r, background[0]),
                blend(g, background[1]),
                blend(b, background[2]),
                255,
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WsiFile;

    /// A two-level slide whose pixel at level `l`, (`x`, `y`) is
    /// `[x % 256, y % 256, l * 100, 255]`, with an opaque "label".
    struct Gradient {
        reads: std::sync::atomic::AtomicUsize,
    }

    impl ExternalSlide for Gradient {
        fn read_region(
            &self,
            level: u32,
            x: u64,
            y: u64,
            width: u32,
            height: u32,
            out: &mut [u8],
        ) -> std::result::Result<(), String> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            if x == 999 {
                return Err("the server went away".into());
            }
            for row in 0..height as u64 {
                for col in 0..width as u64 {
                    let at = ((row * u64::from(width) + col) * 4) as usize;
                    out[at..at + 4].copy_from_slice(&[
                        ((x + col) % 256) as u8,
                        ((y + row) % 256) as u8,
                        (level * 100) as u8,
                        255,
                    ]);
                }
            }
            Ok(())
        }

        fn read_associated_image(
            &self,
            name: &str,
        ) -> std::result::Result<(u32, u32, Vec<u8>), String> {
            match name {
                "label" => Ok((2, 1, vec![1, 2, 3, 255, 4, 5, 6, 255])),
                _ => Err(format!("no {name}")),
            }
        }
    }

    struct GradientFormat;

    impl ExternalOpener for GradientFormat {
        fn reader_name(&self) -> String {
            "the test reader".into()
        }

        fn extensions(&self) -> Vec<String> {
            vec!["grad".into()]
        }

        fn open(
            &self,
            _path: &Path,
        ) -> std::result::Result<(ExternalSlideInfo, Box<dyn ExternalSlide>), String> {
            let info = ExternalSlideInfo {
                levels: vec![
                    ExternalLevel {
                        width: 1000,
                        height: 600,
                        downsample: 1.0,
                        tile_width: Some(256),
                        tile_height: Some(256),
                    },
                    ExternalLevel {
                        width: 250,
                        height: 150,
                        downsample: 4.0,
                        tile_width: None,
                        tile_height: None,
                    },
                ],
                vendor: "test".into(),
                mpp_x: Some(0.5),
                mpp_y: Some(0.5),
                background: [10, 20, 30],
                properties: BTreeMap::from([("test.key".into(), "value".into())]),
                associated_images: vec![("label".into(), 2, 1)],
                ..ExternalSlideInfo::default()
            };
            Ok((
                info,
                Box::new(Gradient {
                    reads: Default::default(),
                }),
            ))
        }
    }

    #[test]
    fn external_slides_open_and_read_through_wsi_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.grad");
        std::fs::write(&path, b"not a format eov reads").unwrap();
        // Unknown before the reader is registered, read after.
        assert!(WsiFile::open(&path).is_err());
        register_opener("test-reader", Arc::new(GradientFormat));
        assert!(external_extensions().contains(&"grad".to_string()));
        let wsi = WsiFile::open(&path).unwrap();
        let props = wsi.properties();
        assert_eq!((props.width, props.height), (1000, 600));
        assert_eq!(props.levels.len(), 2);
        assert_eq!(props.mpp_x, Some(0.5));
        assert_eq!(wsi.property("test.key"), Some("value"));
        assert_eq!(wsi.property("openslide.vendor"), Some("test"));

        // Level 0: the reader's pixels; level 1: level pixel coordinates.
        let pixels = wsi.read_region(10, 20, 0, 4, 2).unwrap();
        assert_eq!(&pixels[..4], [10, 20, 0, 255]);
        let level1 = wsi.read_region(400, 80, 1, 2, 2).unwrap();
        assert_eq!(&level1[..4], [100, 20, 100, 255]);
        // Outside the slide: the background.
        let edge = wsi.read_region(998, 0, 0, 4, 1).unwrap();
        assert_eq!(&edge[8..12], [10, 20, 30, 255]);
        // Large reads arrive in parts and come out whole.
        let big = wsi.read_region(0, 0, 0, 1000, 600).unwrap();
        assert_eq!(
            &big[(599 * 1000 + 999) * 4..][..4],
            [(999 % 256) as u8, (599 % 256) as u8, 0, 255]
        );
        // A failing read fails, and the slide says so.
        assert!(wsi.read_region(999, 0, 0, 1, 1).is_err());
        assert!(wsi.health().last_error.is_some());
        // Associated images.
        let label = wsi.read_associated_image("label").unwrap();
        assert_eq!(&label.pixels[..4], [1, 2, 3, 255]);
        // Several threads at once.
        std::thread::scope(|scope| {
            for t in 0..4u64 {
                let wsi = wsi.clone();
                scope.spawn(move || {
                    for i in 0..20u64 {
                        let x = (t * 100 + i) as i64;
                        let px = wsi.read_region(x, 5, 0, 1, 1).unwrap();
                        assert_eq!(&px[..4], [(x % 256) as u8, 5, 0, 255]);
                    }
                });
            }
        });

        unregister_opener("test-reader");
        assert!(!external_extensions().contains(&"grad".to_string()));
    }

    #[test]
    fn straight_and_premultiplied_reads_keep_alpha() {
        let half = [200, 100, 50, 128];
        assert_eq!(convert(half, AlphaMode::Straight, [0, 0, 0]), half);
        assert_eq!(
            convert(half, AlphaMode::Premultiplied, [0, 0, 0]),
            [100, 50, 25, 128]
        );
        assert_eq!(
            convert(half, AlphaMode::Composited, [255, 255, 255]),
            [227, 177, 152, 255]
        );
        assert_eq!(
            convert([9, 9, 9, 0], AlphaMode::Straight, [0, 0, 0]),
            [0, 0, 0, 0]
        );
    }
}
