//! eov's OME-TIFF reader.
//!
//! OpenSlide reads RGB brightfield slides only. This reader opens OME-TIFF
//! (`.ome.tif`, `.ome.tiff`, `.ome.btf`, or any TIFF whose first
//! ImageDescription is OME-XML), which covers brightfield RGB as well as
//! multichannel fluorescence and multiplexed immunofluorescence (CyCIF,
//! Orion and other OME-TIFF exports):
//!
//! * TIFF and BigTIFF, either byte order, one file;
//! * RGB (8-bit, interleaved or planar), or any number of channels of 8,
//!   16 or 32-bit integer or 32/64-bit float samples, one plane per channel
//!   or several channels as samples of one plane;
//! * pyramids stored as SubIFDs (the current OME convention), as separate
//!   images of decreasing size, or as reduced-resolution IFDs in the main
//!   chain; tiles or strips; the compressions of [`codec`];
//! * every focal plane and time point of Z-stacks and time series (one
//!   shown at a time);
//! * images named "label", "macro" or "thumbnail" as associated images.
//!
//! Multichannel reads are composited to RGB with the slide's
//! [`ChannelDisplay`] (see [`crate::channels`]); decoded tiles are cached
//! as raw samples in [`cache::CHUNKS`], so changing the display costs no
//! decoding. Everything that is known to be unreadable when the file is
//! opened (a compression, a sample type, a channel stored in another file)
//! fails the open with an error that says so.

pub(crate) mod cache;
pub(crate) mod codec;
pub(crate) mod tiff;
pub(crate) mod xml;

use self::cache::{CHUNKS, ChunkKey};
use self::codec::DecodedChunk;
use self::tiff::{Ifd, Planar, SampleType, TiffFile};
use self::xml::{OmeDocument, OmeImage, Pixels};
use crate::channels::{
    ChannelDisplay, ChannelInfo, ChannelLut, ChannelSet, ChannelStats, Compositor,
};
use crate::imaging::RgbaImageData;
use crate::wsi::source::{Opened, Pin, Region, SlideSource};
use crate::wsi::{
    AlphaMode, AssociatedImage, HealthState, MAX_ASSOCIATED_IMAGE_PIXELS, SlideHealth, WsiLevel,
    WsiProperties, pixels,
};
use crate::{Error, Result};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, warn};

/// `openslide.vendor` of slides this reader opens.
pub const VENDOR: &str = "ome-tiff";

/// File name endings that mean OME-TIFF.
const OME_SUFFIXES: &[&str] = &[".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf"];

/// How much of the first ImageDescription is read to recognise OME-XML.
const SNIFF_BYTES: u64 = 4096;

/// Samples per channel read for the statistics behind auto contrast.
const STATS_PIXELS: u64 = 1 << 20;

/// Chunks remembered as undecodable (they fail fast until cleared).
const MAX_BAD_CHUNKS: usize = 4096;

/// IFDs outside the images' planes inspected for a pyramid in the main
/// chain.
const MAX_EXTRA_IFDS: usize = 4096;

/// Channels accepted (the largest cyclic-immunofluorescence panels have
/// about a hundred).
const MAX_CHANNELS: usize = 1024;
/// Focal planes offered at most (the rest of a deeper stack is not shown).
const MAX_FOCAL_PLANES: u64 = 512;
/// Planes (focal planes × time points) offered at most; later time points
/// are not shown.
const MAX_PLANE_STACKS: u64 = 4096;

static NEXT_SLIDE_ID: AtomicU64 = AtomicU64::new(1);

/// Whether the file name ends in an OME-TIFF extension.
pub(crate) fn has_ome_extension(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    OME_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

/// Whether `path` is a TIFF whose first ImageDescription is OME-XML. Reads
/// the header, one directory and the start of the description.
pub(crate) fn is_ome_tiff(path: &Path) -> bool {
    let Ok(tiff) = TiffFile::open(path) else {
        return false;
    };
    matches!(
        tiff.first_description_start(SNIFF_BYTES),
        Ok(Some(text)) if xml::looks_like_ome(&text)
    )
}

/// Why a chunk could not be read.
enum ChunkError {
    /// The file could not be read (gone, network down): transient.
    Io(std::io::Error),
    /// The data is damaged: permanent.
    Data(String),
}

/// Reads and decodes chunks of one file through the shared cache.
struct ChunkReader {
    id: u64,
    tiff: TiffFile,
    bad: Mutex<HashMap<ChunkKey, String>>,
    last_error: Mutex<Option<String>>,
    failing: AtomicBool,
}

impl ChunkReader {
    fn new(tiff: TiffFile) -> Self {
        Self {
            id: NEXT_SLIDE_ID.fetch_add(1, Ordering::Relaxed),
            tiff,
            bad: Mutex::new(HashMap::new()),
            last_error: Mutex::new(None),
            failing: AtomicBool::new(false),
        }
    }

    /// Chunk (`tx`, `ty`) of `ifd` holding `sample`, decoded.
    fn chunk(
        &self,
        ifd: &Ifd,
        tx: u32,
        ty: u32,
        sample: u16,
    ) -> std::result::Result<Arc<DecodedChunk>, ChunkError> {
        let index = ifd.chunk_index(tx, ty, sample);
        let key = ChunkKey {
            slide: self.id,
            ifd: ifd.offset,
            chunk: index,
        };
        if let Some(chunk) = CHUNKS.get(&key) {
            return Ok(chunk);
        }
        if let Some(message) = self.bad.lock().get(&key) {
            return Err(ChunkError::Data(message.clone()));
        }
        let compressed = match self.tiff.read_chunk(ifd, index) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                let chunk = Arc::new(DecodedChunk::missing(
                    ifd.chunk_width,
                    ifd.chunk_rows(ty),
                    ifd.samples_per_chunk(),
                    ifd.sample_type,
                ));
                CHUNKS.insert(key, Arc::clone(&chunk));
                return Ok(chunk);
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::InvalidData
                ) =>
            {
                return Err(self.remember(key, error.to_string()));
            }
            Err(error) => {
                self.failing.store(true, Ordering::Relaxed);
                *self.last_error.lock() = Some(error.to_string());
                return Err(ChunkError::Io(error));
            }
        };
        self.failing.store(false, Ordering::Relaxed);
        match codec::decode(ifd, self.tiff.order, ty, &compressed) {
            Ok(chunk) => {
                let chunk = Arc::new(chunk);
                CHUNKS.insert(key, Arc::clone(&chunk));
                Ok(chunk)
            }
            Err(message) => Err(self.remember(key, message)),
        }
    }

    fn remember(&self, key: ChunkKey, message: String) -> ChunkError {
        let mut bad = self.bad.lock();
        if bad.len() >= MAX_BAD_CHUNKS {
            bad.clear();
        }
        bad.insert(key, message.clone());
        *self.last_error.lock() = Some(message.clone());
        ChunkError::Data(message)
    }
}

/// Where one channel (or the RGB image) of one level is stored.
#[derive(Clone)]
struct PlaneRef {
    ifd: Arc<Ifd>,
    /// The sample of the IFD's pixels that holds the channel.
    sample: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// One 8-bit colour plane, shown as is.
    Rgb,
    /// Grey-value channels, composited.
    Channels,
}

/// Where the planes of an image are.
struct ImagePlanes {
    layout: Layout,
    /// One per channel; one (with every colour sample) for RGB.
    planes: Vec<PlaneRef>,
    /// The channels come from the samples of one colour plane.
    from_color_samples: bool,
}

struct Level {
    width: u64,
    height: u64,
    downsample: f64,
    planes: Vec<PlaneRef>,
}

enum Content {
    Rgb,
    Channels {
        set: ChannelSet,
        defaults: ChannelDisplay,
        display: RwLock<Arc<Compositor>>,
    },
}

struct Associated {
    name: String,
    ifd: Arc<Ifd>,
}

/// An open OME-TIFF.
pub(crate) struct OmeTiffSource {
    reader: ChunkReader,
    levels: Vec<Level>,
    /// The planes of every level for each time point and focal plane:
    /// `[t * size_z + z][level][channel]` (`[0]` is the levels' own).
    stacks: Vec<Vec<Vec<PlaneRef>>>,
    /// Focal planes per time point.
    size_z: usize,
    /// The focal plane and time point reads come from.
    focal_plane: AtomicUsize,
    time_point: AtomicUsize,
    content: Content,
    background: [u8; 3],
    associated: Vec<Associated>,
}

impl Drop for OmeTiffSource {
    fn drop(&mut self) {
        CHUNKS.remove_slide(self.reader.id);
    }
}

/// Parses IFDs on demand, once each.
struct IfdTable<'a> {
    tiff: &'a TiffFile,
    chain: Vec<u64>,
    parsed: HashMap<u64, Arc<Ifd>>,
}

impl IfdTable<'_> {
    fn by_offset(&mut self, offset: u64) -> Result<Arc<Ifd>> {
        if let Some(ifd) = self.parsed.get(&offset) {
            return Ok(Arc::clone(ifd));
        }
        let ifd = Arc::new(self.tiff.ifd(offset)?);
        self.parsed.insert(offset, Arc::clone(&ifd));
        Ok(ifd)
    }

    fn by_index(&mut self, index: u64) -> Result<Arc<Ifd>> {
        let offset = *self
            .chain
            .get(index as usize)
            .ok_or_else(|| self.tiff.invalid(format!("there is no IFD {index}")))?;
        self.by_offset(offset)
    }
}

/// The associated-image name an OME image's name stands for.
fn associated_name(image: &OmeImage) -> Option<&'static str> {
    let name = image.name.as_deref()?.to_ascii_lowercase();
    ["label", "macro", "thumbnail"]
        .into_iter()
        .find(|kind| name.contains(kind))
}

fn pixel_count(pixels: &Pixels) -> u64 {
    pixels.size_x.saturating_mul(pixels.size_y)
}

/// The image to show: the largest that is not a label, macro or
/// thumbnail.
fn main_image(doc: &OmeDocument) -> usize {
    let candidates: Vec<usize> = (0..doc.images.len())
        .filter(|&i| associated_name(&doc.images[i]).is_none())
        .collect();
    let candidates = if candidates.is_empty() {
        (0..doc.images.len()).collect()
    } else {
        candidates
    };
    candidates
        .into_iter()
        .rev()
        .max_by_key(|&i| pixel_count(&doc.images[i].pixels))
        .unwrap_or(0)
}

/// Where focal plane `z` at time point `t` of every channel of `pixels` is
/// stored.
fn image_planes(
    table: &mut IfdTable<'_>,
    pixels: &Pixels,
    own_file: &str,
    own_uuid: Option<&str>,
    (z, t): (u64, u64),
) -> Result<ImagePlanes> {
    let channels = pixels.effective_size_c();
    if channels > MAX_CHANNELS as u64 {
        return Err(Error::Unsupported(format!(
            "{channels} channels (at most {MAX_CHANNELS} are supported)"
        )));
    }
    let channels = channels as usize;
    let map = pixels.plane_ifds(table.chain.len() as u64, own_file, own_uuid);
    let sizes = (channels as u64, pixels.size_z.max(1), pixels.size_t.max(1));
    let mut ifds: Vec<Option<Arc<Ifd>>> = Vec::with_capacity(channels);
    for c in 0..channels {
        let plane = pixels.dimension_order.plane_index((c as u64, z, t), sizes);
        ifds.push(match map.get(&plane) {
            Some(&index) => Some(table.by_index(index)?),
            None => None,
        });
    }
    let first = ifds[0].clone().ok_or_else(|| {
        Error::Unsupported(
            "the image's first channel is stored in another file (multi-file OME-TIFF sets \
             are not supported)"
                .to_string(),
        )
    })?;
    let in_first = ifds
        .iter()
        .all(|ifd| ifd.as_ref().is_none_or(|ifd| ifd.offset == first.offset));
    if first.is_color() && first.sample_type == SampleType::U8 && (channels == 1 || in_first) {
        return Ok(ImagePlanes {
            layout: Layout::Rgb,
            planes: vec![PlaneRef {
                ifd: first,
                sample: 0,
            }],
            from_color_samples: false,
        });
    }
    if channels == 1 && first.is_color() {
        // Colour with more than 8 bits: show its samples as channels.
        let planes = (0..3)
            .map(|sample| PlaneRef {
                ifd: Arc::clone(&first),
                sample,
            })
            .collect();
        return Ok(ImagePlanes {
            layout: Layout::Channels,
            planes,
            from_color_samples: true,
        });
    }
    // Several channels may share an IFD as its samples; hand them out in
    // channel order.
    let mut taken: HashMap<u64, u16> = HashMap::new();
    let mut planes = Vec::with_capacity(channels);
    for (c, ifd) in ifds.into_iter().enumerate() {
        let ifd = match ifd {
            Some(ifd) => ifd,
            None if channels <= usize::from(first.samples_per_pixel) => Arc::clone(&first),
            None => {
                return Err(Error::Unsupported(format!(
                    "channel {c} is stored in another file (multi-file OME-TIFF sets are not \
                     supported)"
                )));
            }
        };
        let next = taken.entry(ifd.offset).or_insert(0);
        if *next >= ifd.samples_per_pixel {
            return Err(table.tiff.invalid(format!(
                "more channels are stored in the IFD at {} than it has samples",
                ifd.offset
            )));
        }
        let sample = *next;
        *next += 1;
        if (ifd.width, ifd.height, ifd.sample_type)
            != (first.width, first.height, first.sample_type)
        {
            return Err(Error::Unsupported(format!(
                "channel {c} differs in size or sample type from the first channel"
            )));
        }
        planes.push(PlaneRef { ifd, sample });
    }
    Ok(ImagePlanes {
        layout: Layout::Channels,
        planes,
        from_color_samples: false,
    })
}

/// Checks that every IFD of `planes` can be decoded.
fn check_decodable(planes: &[PlaneRef]) -> Result<()> {
    for plane in planes {
        if let Some(reason) = codec::unsupported(&plane.ifd) {
            return Err(Error::Unsupported(reason));
        }
    }
    Ok(())
}

/// Whether `planes` can be the level after `previous`: consistent sizes and
/// types, decodable, and smaller.
fn valid_level(
    planes: &[PlaneRef],
    base: &[PlaneRef],
    previous: (u32, u32),
    layout: Layout,
) -> bool {
    let Some(first) = planes.first() else {
        return false;
    };
    let (width, height) = (first.ifd.width, first.ifd.height);
    let smaller = width <= previous.0 && height <= previous.1 && (width, height) != previous;
    smaller
        && planes.len() == base.len()
        && planes.iter().zip(base).all(|(plane, base)| {
            plane.ifd.width == width
                && plane.ifd.height == height
                && plane.ifd.sample_type == base.ifd.sample_type
                && plane.sample < plane.ifd.samples_per_pixel
                && (layout == Layout::Channels || plane.ifd.is_color())
                && codec::unsupported(&plane.ifd).is_none()
        })
}

/// How a file stores its pyramid, for the property map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PyramidKind {
    None,
    SubIfds,
    Images,
    ReducedIfds,
}

impl PyramidKind {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SubIfds => "SubIFDs",
            Self::Images => "separate images",
            Self::ReducedIfds => "reduced-resolution IFDs",
        }
    }
}

/// The levels below level 0, from wherever the file keeps them.
fn pyramid(
    table: &mut IfdTable<'_>,
    doc: &OmeDocument,
    main: usize,
    base: &ImagePlanes,
    own: (&str, Option<&str>),
    zt: (u64, u64),
) -> (Vec<Vec<PlaneRef>>, PyramidKind) {
    let base_size = (base.planes[0].ifd.width, base.planes[0].ifd.height);
    let accept = |candidates: Vec<Vec<PlaneRef>>| -> Vec<Vec<PlaneRef>> {
        let mut levels = Vec::new();
        let mut previous = base_size;
        for planes in candidates {
            if !valid_level(&planes, &base.planes, previous, base.layout) {
                debug!("stopping the pyramid at an inconsistent level");
                break;
            }
            previous = (planes[0].ifd.width, planes[0].ifd.height);
            levels.push(planes);
        }
        levels
    };

    // SubIFDs of every plane.
    let depth = base
        .planes
        .iter()
        .map(|plane| plane.ifd.sub_ifds.len())
        .min()
        .unwrap_or(0);
    let mut candidates = Vec::new();
    for k in 0..depth {
        let level: Result<Vec<PlaneRef>> = base
            .planes
            .iter()
            .map(|plane| {
                Ok(PlaneRef {
                    ifd: table.by_offset(plane.ifd.sub_ifds[k])?,
                    sample: plane.sample,
                })
            })
            .collect();
        match level {
            Ok(level) => candidates.push(level),
            Err(error) => {
                warn!("ignoring SubIFD level {}: {error}", k + 1);
                break;
            }
        }
    }
    let levels = accept(candidates);
    if !levels.is_empty() {
        return (levels, PyramidKind::SubIfds);
    }

    // Other images of the same channels, smaller.
    let main_pixels = &doc.images[main].pixels;
    let mut others: Vec<usize> = (0..doc.images.len())
        .filter(|&i| i != main && associated_name(&doc.images[i]).is_none())
        .filter(|&i| {
            let pixels = &doc.images[i].pixels;
            pixels.effective_size_c() == main_pixels.effective_size_c()
                && pixels.pixel_type == main_pixels.pixel_type
                && pixel_count(pixels) < pixel_count(main_pixels)
        })
        .collect();
    others.sort_by_key(|&i| std::cmp::Reverse(pixel_count(&doc.images[i].pixels)));
    let candidates: Vec<Vec<PlaneRef>> = others
        .into_iter()
        .filter_map(|i| image_planes(table, &doc.images[i].pixels, own.0, own.1, zt).ok())
        .filter(|planes| planes.layout == base.layout)
        .map(|planes| planes.planes)
        .collect();
    let levels = accept(candidates);
    if !levels.is_empty() {
        return (levels, PyramidKind::Images);
    }

    // Reduced-resolution IFDs in the main chain that no image claims,
    // grouped by size: one per distinct level-0 IFD, in the same order.
    let mut claimed = std::collections::HashSet::new();
    for image in &doc.images {
        claimed.extend(
            image
                .pixels
                .plane_ifds(table.chain.len() as u64, own.0, own.1)
                .into_values(),
        );
    }
    let mut distinct: Vec<u64> = Vec::new();
    for plane in &base.planes {
        if !distinct.contains(&plane.ifd.offset) {
            distinct.push(plane.ifd.offset);
        }
    }
    let extra: Vec<u64> = (0..table.chain.len() as u64)
        .filter(|index| !claimed.contains(index))
        .take(MAX_EXTRA_IFDS)
        .collect();
    let mut groups: Vec<Vec<Arc<Ifd>>> = Vec::new();
    for index in extra {
        let Ok(ifd) = table.by_index(index) else {
            continue;
        };
        if ifd.new_subfile_type & 1 == 0 {
            continue;
        }
        match groups.last_mut() {
            Some(group)
                if group.len() < distinct.len()
                    && (group[0].width, group[0].height) == (ifd.width, ifd.height) =>
            {
                group.push(ifd)
            }
            _ => groups.push(vec![ifd]),
        }
    }
    let candidates: Vec<Vec<PlaneRef>> = groups
        .into_iter()
        .filter(|group| group.len() == distinct.len())
        .map(|group| {
            base.planes
                .iter()
                .map(|plane| {
                    let position = distinct
                        .iter()
                        .position(|&offset| offset == plane.ifd.offset)
                        .unwrap_or(0);
                    PlaneRef {
                        ifd: Arc::clone(&group[position]),
                        sample: plane.sample,
                    }
                })
                .collect()
        })
        .collect();
    let levels = accept(candidates);
    if !levels.is_empty() {
        return (levels, PyramidKind::ReducedIfds);
    }
    (Vec::new(), PyramidKind::None)
}

/// A level-0 coordinate in pixels of a level of `downsample`.
///
/// Callers derive level-0 coordinates from level pixels by truncating
/// (`(x * downsample) as i64`) or by rounding; this maps both back to the
/// same pixel for every downsample above 1.5 (and exactly at 1).
fn level_coordinate(x: i64, downsample: f64) -> i64 {
    if downsample == 1.0 {
        return x;
    }
    // The epsilon keeps a rounded position that lands exactly on a level
    // pixel from being pushed to the next one by floating-point error.
    ((x as f64 - 0.5) / downsample - 1e-9).ceil() as i64
}

/// The part of a read that lies inside the level, in level pixels.
struct Clip {
    x0: u64,
    y0: u64,
    x1: u64,
    y1: u64,
    /// Level position of the read's first pixel.
    origin_x: i64,
    origin_y: i64,
    /// Width of the read.
    out_width: usize,
}

impl Clip {
    fn new(
        origin_x: i64,
        origin_y: i64,
        width: u32,
        height: u32,
        level_size: (u64, u64),
    ) -> Option<Self> {
        let clamp = |v: i64, max: u64| v.clamp(0, max as i64) as u64;
        let x0 = clamp(origin_x, level_size.0);
        let y0 = clamp(origin_y, level_size.1);
        let x1 = clamp(origin_x.saturating_add(i64::from(width)), level_size.0);
        let y1 = clamp(origin_y.saturating_add(i64::from(height)), level_size.1);
        (x0 < x1 && y0 < y1).then_some(Self {
            x0,
            y0,
            x1,
            y1,
            origin_x,
            origin_y,
            out_width: width as usize,
        })
    }

    /// Byte offset in the output of level pixel (`x`, `y`).
    #[inline(always)]
    fn out_index(&self, x: u64, y: u64) -> usize {
        ((y as i64 - self.origin_y) as usize * self.out_width + (x as i64 - self.origin_x) as usize)
            * 4
    }
}

/// A rectangle of level pixels inside one chunk.
struct ChunkPart {
    tx: u32,
    ty: u32,
    /// Level position of the chunk's first pixel.
    left: u64,
    top: u64,
    x0: u64,
    y0: u64,
    x1: u64,
    y1: u64,
}

/// The chunks of `ifd` that `clip` touches, and the part of each.
fn chunk_parts(ifd: &Ifd, clip: &Clip) -> Vec<ChunkPart> {
    let (cw, ch) = (u64::from(ifd.chunk_width), u64::from(ifd.chunk_height));
    let mut parts = Vec::new();
    for ty in (clip.y0 / ch)..(clip.y1 - 1) / ch + 1 {
        for tx in (clip.x0 / cw)..(clip.x1 - 1) / cw + 1 {
            let (left, top) = (tx * cw, ty * ch);
            parts.push(ChunkPart {
                tx: tx as u32,
                ty: ty as u32,
                left,
                top,
                x0: clip.x0.max(left),
                y0: clip.y0.max(top),
                x1: clip.x1.min(left + cw),
                y1: clip.y1.min(top + ch),
            });
        }
    }
    parts
}

/// Adds the colour of `sample` of every pixel of `part` to `out`.
fn accumulate(
    lut: &ChannelLut,
    chunk: &DecodedChunk,
    sample: usize,
    part: &ChunkPart,
    clip: &Clip,
    out: &mut [u8],
) {
    macro_rules! run {
        ($ty:ty, $map:expr) => {{
            const BYTES: usize = std::mem::size_of::<$ty>();
            let spp = usize::from(chunk.samples);
            let width = chunk.width as usize;
            for y in part.y0..part.y1 {
                let row = (y - part.top) as usize * width;
                let mut o = clip.out_index(part.x0, y);
                for x in part.x0..part.x1 {
                    let i = ((row + (x - part.left) as usize) * spp + sample) * BYTES;
                    let mut bytes = [0u8; BYTES];
                    bytes.copy_from_slice(&chunk.data[i..i + BYTES]);
                    let rgb: [u8; 3] = $map(<$ty>::from_ne_bytes(bytes));
                    out[o] = out[o].saturating_add(rgb[0]);
                    out[o + 1] = out[o + 1].saturating_add(rgb[1]);
                    out[o + 2] = out[o + 2].saturating_add(rgb[2]);
                    o += 4;
                }
            }
        }};
    }
    match chunk.sample_type {
        SampleType::U8 => run!(u8, |v: u8| lut.lookup_index(i64::from(v))),
        SampleType::I8 => run!(i8, |v: i8| lut.lookup_index(i64::from(v))),
        SampleType::U16 => run!(u16, |v: u16| lut.lookup_index(i64::from(v))),
        SampleType::I16 => run!(i16, |v: i16| lut.lookup_index(i64::from(v))),
        SampleType::U32 => run!(u32, |v: u32| lut.lookup_value(f64::from(v))),
        SampleType::I32 => run!(i32, |v: i32| lut.lookup_value(f64::from(v))),
        SampleType::F32 => run!(f32, |v: f32| lut.lookup_value(f64::from(v))),
        SampleType::F64 => run!(f64, |v: f64| lut.lookup_value(v)),
    }
}

/// Visits every sample `sample` of the pixels of `chunk` that lie inside
/// the image (tiles at the edges are padded).
fn for_each_value(
    chunk: &DecodedChunk,
    sample: usize,
    valid: (usize, usize),
    mut f: impl FnMut(f64, i64),
) {
    macro_rules! run {
        ($ty:ty, $float:expr, $int:expr) => {{
            const BYTES: usize = std::mem::size_of::<$ty>();
            let spp = usize::from(chunk.samples);
            for y in 0..valid.1.min(chunk.rows as usize) {
                let row = y * chunk.width as usize;
                for x in 0..valid.0.min(chunk.width as usize) {
                    let i = ((row + x) * spp + sample) * BYTES;
                    let mut bytes = [0u8; BYTES];
                    bytes.copy_from_slice(&chunk.data[i..i + BYTES]);
                    let v = <$ty>::from_ne_bytes(bytes);
                    f($float(v), $int(v));
                }
            }
        }};
    }
    match chunk.sample_type {
        SampleType::U8 => run!(u8, f64::from, i64::from),
        SampleType::I8 => run!(i8, f64::from, i64::from),
        SampleType::U16 => run!(u16, f64::from, i64::from),
        SampleType::I16 => run!(i16, f64::from, i64::from),
        SampleType::U32 => run!(u32, f64::from, i64::from),
        SampleType::I32 => run!(i32, f64::from, i64::from),
        SampleType::F32 => run!(f32, f64::from, |_| 0),
        SampleType::F64 => run!(f64, |v| v, |_| 0),
    }
}

impl ChunkReader {
    /// Statistics of one channel at `level`: every chunk when the level is
    /// small, else chunks spread over it, up to [`STATS_PIXELS`] samples.
    fn channel_stats(&self, level: &Level, plane: &PlaneRef) -> ChannelStats {
        let ifd = &plane.ifd;
        let (across, down) = (ifd.chunks_across(), ifd.chunks_down());
        let chunk_pixels = u64::from(ifd.chunk_width) * u64::from(ifd.chunk_height);
        let budget = (STATS_PIXELS / chunk_pixels.max(1)).max(1);
        let total = u64::from(across) * u64::from(down);
        let picks: Vec<(u32, u32)> = if total <= budget {
            (0..down)
                .flat_map(|ty| (0..across).map(move |tx| (tx, ty)))
                .collect()
        } else {
            // A grid of chunks spread over the level.
            let aspect = f64::from(across) / f64::from(down);
            let gx = ((budget as f64 * aspect).sqrt().round() as u32).clamp(1, across);
            let gy = ((budget / u64::from(gx)) as u32).clamp(1, down);
            (0..gy)
                .flat_map(|j| {
                    (0..gx).map(move |i| {
                        (
                            ((f64::from(i) + 0.5) * f64::from(across) / f64::from(gx)) as u32,
                            ((f64::from(j) + 0.5) * f64::from(down) / f64::from(gy)) as u32,
                        )
                    })
                })
                .collect()
        };
        let sample = match ifd.planar {
            Planar::Chunky => usize::from(plane.sample),
            Planar::Separate => 0,
        };
        let valid = |tx: u32, ty: u32| {
            (
                (level.width - u64::from(tx) * u64::from(ifd.chunk_width))
                    .min(u64::from(ifd.chunk_width)) as usize,
                (level.height - u64::from(ty) * u64::from(ifd.chunk_height))
                    .min(u64::from(ifd.chunk_height)) as usize,
            )
        };
        let chunks = picks.into_iter().filter_map(|(tx, ty)| {
            self.chunk(ifd, tx, ty, plane.sample)
                .ok()
                .filter(|chunk| !chunk.missing)
                .map(|chunk| (chunk, valid(tx, ty)))
        });
        match ifd.sample_type {
            SampleType::U8 | SampleType::I8 | SampleType::U16 | SampleType::I16 => {
                let (offset, len) = match ifd.sample_type {
                    SampleType::U8 => (0, 256),
                    SampleType::I8 => (-128, 256),
                    SampleType::U16 => (0, 65_536),
                    _ => (-32_768, 65_536),
                };
                let mut counts = vec![0u64; len];
                for (chunk, valid) in chunks {
                    for_each_value(&chunk, sample, valid, |_, v| {
                        counts[(v - offset) as usize] += 1;
                    });
                }
                let unsigned = matches!(ifd.sample_type, SampleType::U8 | SampleType::U16);
                ChannelStats::from_counts(&counts, offset as f64, unsigned)
            }
            _ => {
                let mut values = Vec::new();
                for (chunk, valid) in chunks {
                    for_each_value(&chunk, sample, valid, |v, _| values.push(v));
                }
                ChannelStats::from_values(&mut values)
            }
        }
    }
}

/// Validated micrometres per pixel.
fn mpp(length: Option<&xml::Length>) -> Option<f64> {
    length
        .and_then(xml::Length::micrometres)
        .filter(|um| *um <= 10_000.0)
}

/// Micrometres per pixel from the TIFF resolution tags, for files whose
/// OME-XML has no physical size. Only centimetre units count: writers
/// default to 72 pixels per inch, which is not a calibration.
fn resolution_mpp(ifd: &Ifd) -> (Option<f64>, Option<f64>) {
    let Some((x, y, 3)) = ifd.resolution else {
        return (None, None);
    };
    let convert = |per_cm: f64| {
        (per_cm.is_finite() && per_cm > 0.0)
            .then(|| 10_000.0 / per_cm)
            .filter(|um| *um <= 100.0)
    };
    (convert(x), convert(y))
}

fn channel_infos(image: &OmeImage, planes: &ImagePlanes) -> Vec<ChannelInfo> {
    if planes.from_color_samples {
        return ["Red", "Green", "Blue"]
            .into_iter()
            .zip([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
            .enumerate()
            .map(|(index, (name, color))| ChannelInfo {
                index,
                name: name.to_string(),
                color: Some(color),
                fluor: None,
                emission_nm: None,
                excitation_nm: None,
                stats: ChannelStats::default(),
            })
            .collect();
    }
    (0..planes.planes.len())
        .map(|index| {
            let ome = image.pixels.channels.get(index);
            let name = ome
                .and_then(|c| c.name.clone().or_else(|| c.fluor.clone()))
                .unwrap_or_else(|| format!("Channel {}", index + 1));
            ChannelInfo {
                index,
                name,
                color: ome
                    .and_then(|c| c.color)
                    .filter(|rgba| *rgba != [0, 0, 0, 0])
                    .map(|[r, g, b, _]| [r, g, b]),
                fluor: ome.and_then(|c| c.fluor.clone()),
                emission_nm: ome
                    .and_then(|c| c.emission.as_ref())
                    .and_then(xml::Length::nanometres),
                excitation_nm: ome
                    .and_then(|c| c.excitation.as_ref())
                    .and_then(xml::Length::nanometres),
                stats: ChannelStats::default(),
            }
        })
        .collect()
}

/// Properties shown in the Slide Information panel and published to
/// plugins: OpenSlide's standard `openslide.*` keys, so code written for
/// OpenSlide properties keeps working, plus the OME metadata as `ome.*`.
#[allow(clippy::too_many_arguments)]
fn property_map(
    doc: &OmeDocument,
    image: &OmeImage,
    first: &Ifd,
    levels: &[Level],
    channels: Option<&[ChannelInfo]>,
    properties: &WsiProperties,
    pyramid: PyramidKind,
    associated: &[Associated],
) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut set = |key: String, value: String| {
        if !value.is_empty() {
            map.insert(key, value);
        }
    };
    set("openslide.vendor".into(), VENDOR.into());
    set("openslide.level-count".into(), levels.len().to_string());
    for (index, level) in levels.iter().enumerate() {
        let ifd = &level.planes[0].ifd;
        let prefix = format!("openslide.level[{index}]");
        set(format!("{prefix}.width"), level.width.to_string());
        set(format!("{prefix}.height"), level.height.to_string());
        set(
            format!("{prefix}.downsample"),
            format!("{}", level.downsample),
        );
        if ifd.tiled {
            set(format!("{prefix}.tile-width"), ifd.chunk_width.to_string());
            set(
                format!("{prefix}.tile-height"),
                ifd.chunk_height.to_string(),
            );
        }
    }
    if let Some(x) = properties.mpp_x {
        set("openslide.mpp-x".into(), format!("{x}"));
    }
    if let Some(y) = properties.mpp_y {
        set("openslide.mpp-y".into(), format!("{y}"));
    }
    if let Some(power) = properties.objective_power {
        set("openslide.objective-power".into(), format!("{power}"));
    }
    let [r, g, b] = properties.background_color;
    set(
        "openslide.background-color".into(),
        format!("{r:02X}{g:02X}{b:02X}"),
    );

    let text = |value: &Option<String>| value.clone().unwrap_or_default();
    set("ome.creator".into(), text(&doc.creator));
    set("ome.uuid".into(), text(&doc.uuid));
    set("ome.image-count".into(), doc.images.len().to_string());
    for (index, other) in doc.images.iter().enumerate() {
        set(
            format!("ome.image[{index}].name"),
            other.name.clone().unwrap_or_else(|| other.id.clone()),
        );
        set(
            format!("ome.image[{index}].size"),
            format!("{} x {}", other.pixels.size_x, other.pixels.size_y),
        );
    }
    set("ome.image.id".into(), image.id.clone());
    set("ome.image.name".into(), text(&image.name));
    set(
        "ome.image.acquisition-date".into(),
        text(&image.acquisition_date),
    );
    set("ome.image.description".into(), text(&image.description));
    let pixels = &image.pixels;
    set("ome.pixels.type".into(), text(&pixels.pixel_type));
    set(
        "ome.pixels.dimension-order".into(),
        pixels.dimension_order.name().into(),
    );
    for (key, value) in [
        ("size-x", pixels.size_x),
        ("size-y", pixels.size_y),
        ("size-z", pixels.size_z),
        ("size-c", pixels.size_c),
        ("size-t", pixels.size_t),
    ] {
        set(format!("ome.pixels.{key}"), value.to_string());
    }
    for (key, length) in [
        ("physical-size-x", &pixels.physical_size_x),
        ("physical-size-y", &pixels.physical_size_y),
        ("physical-size-z", &pixels.physical_size_z),
    ] {
        if let Some(length) = length {
            set(format!("ome.pixels.{key}"), length.display());
        }
    }
    if let Some(bits) = pixels.significant_bits {
        set("ome.pixels.significant-bits".into(), bits.to_string());
    }
    set("ome.pyramid".into(), pyramid.name().into());
    if let Some(objective) = doc.objective_of(image) {
        let number = |v: Option<f64>| v.map(|v| format!("{v}")).unwrap_or_default();
        set(
            "ome.objective.nominal-magnification".into(),
            number(objective.nominal_magnification),
        );
        set(
            "ome.objective.calibrated-magnification".into(),
            number(objective.calibrated_magnification),
        );
        set("ome.objective.lens-na".into(), number(objective.lens_na));
        set("ome.objective.model".into(), text(&objective.model));
        set(
            "ome.objective.manufacturer".into(),
            text(&objective.manufacturer),
        );
        set("ome.objective.immersion".into(), text(&objective.immersion));
    }
    if let Some(channels) = channels {
        set("ome.channel-count".into(), channels.len().to_string());
        for (index, channel) in channels.iter().enumerate() {
            let prefix = format!("ome.channel[{index}]");
            set(format!("{prefix}.name"), channel.name.clone());
            if let Some([r, g, b]) = channel.color {
                set(format!("{prefix}.color"), format!("#{r:02X}{g:02X}{b:02X}"));
            }
            set(format!("{prefix}.fluor"), text(&channel.fluor));
            if let Some(nm) = channel.emission_nm {
                set(format!("{prefix}.emission-wavelength"), format!("{nm} nm"));
            }
            if let Some(nm) = channel.excitation_nm {
                set(
                    format!("{prefix}.excitation-wavelength"),
                    format!("{nm} nm"),
                );
            }
            if let Some(ome) = pixels.channels.get(index) {
                set(format!("{prefix}.id"), text(&ome.id));
                set(
                    format!("{prefix}.contrast-method"),
                    text(&ome.contrast_method),
                );
                set(
                    format!("{prefix}.illumination-type"),
                    text(&ome.illumination_type),
                );
                set(
                    format!("{prefix}.acquisition-mode"),
                    text(&ome.acquisition_mode),
                );
            }
        }
    }
    for image in associated {
        set(
            format!("openslide.associated.{}.width", image.name),
            image.ifd.width.to_string(),
        );
        set(
            format!("openslide.associated.{}.height", image.name),
            image.ifd.height.to_string(),
        );
    }
    set(
        "tiff.Compression".into(),
        tiff::compression_name(first.compression),
    );
    set("tiff.Software".into(), text(&first.software));
    set("tiff.DateTime".into(), text(&first.date_time));
    set("tiff.Make".into(), text(&first.make));
    set("tiff.Model".into(), text(&first.model));
    map
}

/// Opens an OME-TIFF.
pub(crate) fn open(path: &Path) -> Result<Opened> {
    let tiff = TiffFile::open(path)?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut first = tiff.ifd(tiff.first_ifd)?;
    let description = first
        .description
        .take()
        .filter(|text| xml::looks_like_ome(text))
        .ok_or_else(|| {
            Error::UnsupportedFormat(format!(
                "{file_name}: the first ImageDescription is not OME-XML"
            ))
        })?;
    let doc = xml::parse(&description).map_err(|error| tiff.invalid(error))?;
    drop(description);
    let chain = tiff.chain()?;
    let mut table = IfdTable {
        tiff: &tiff,
        chain,
        parsed: HashMap::new(),
    };
    let first = Arc::new(first);
    table.parsed.insert(first.offset, Arc::clone(&first));
    let own = (file_name.as_str(), doc.uuid.as_deref());

    let main = main_image(&doc);
    let image = &doc.images[main];
    let base = image_planes(&mut table, &image.pixels, own.0, own.1, (0, 0))?;
    check_decodable(&base.planes)?;
    let (sub_levels, pyramid_kind) = pyramid(&mut table, &doc, main, &base, own, (0, 0));

    let base_ifd = Arc::clone(&base.planes[0].ifd);
    let (w0, h0) = (u64::from(base_ifd.width), u64::from(base_ifd.height));
    if (w0, h0) != (image.pixels.size_x, image.pixels.size_y) {
        warn!(
            "{file_name}: OME-XML says {} x {}, the image is {w0} x {h0}",
            image.pixels.size_x, image.pixels.size_y
        );
    }
    let mut levels = vec![Level {
        width: w0,
        height: h0,
        downsample: 1.0,
        planes: base.planes.clone(),
    }];
    for planes in sub_levels {
        let (w, h) = (
            u64::from(planes[0].ifd.width),
            u64::from(planes[0].ifd.height),
        );
        let downsample = (w0 as f64 / w as f64 + h0 as f64 / h as f64) / 2.0;
        levels.push(Level {
            width: w,
            height: h,
            downsample,
            planes,
        });
    }

    // The other focal planes and time points: the same levels, from their
    // own IFDs. A plane that is stored elsewhere or does not match stops the
    // focal planes there, or the time series at the last complete time point.
    let mut stack_at = |z: u64, t: u64| -> std::result::Result<Vec<Vec<PlaneRef>>, String> {
        let base_zt = image_planes(&mut table, &image.pixels, own.0, own.1, (z, t))
            .and_then(|base_zt| {
                check_decodable(&base_zt.planes)?;
                Ok(base_zt)
            })
            .map_err(|error| error.to_string())?;
        if base_zt.layout != base.layout {
            return Err("it is stored differently from the first".to_string());
        }
        let (sub, _) = pyramid(&mut table, &doc, main, &base_zt, own, (z, t));
        let mut by_level = vec![base_zt.planes];
        by_level.extend(sub);
        let matches = by_level.len() == levels.len()
            && by_level.iter().zip(&levels).all(|(planes, level)| {
                planes.len() == level.planes.len()
                    && planes.iter().all(|plane| {
                        (u64::from(plane.ifd.width), u64::from(plane.ifd.height))
                            == (level.width, level.height)
                    })
            });
        if matches {
            Ok(by_level)
        } else {
            Err("its levels differ from the first plane's".to_string())
        }
    };
    let mut stacks: Vec<Vec<Vec<PlaneRef>>> =
        vec![levels.iter().map(|level| level.planes.clone()).collect()];
    for z in 1..image.pixels.size_z.clamp(1, MAX_FOCAL_PLANES) {
        match stack_at(z, 0) {
            Ok(stack) => stacks.push(stack),
            Err(reason) => {
                warn!(
                    "{file_name}: focal plane {} cannot be shown ({reason}); showing {z} of {} focal planes",
                    z + 1,
                    image.pixels.size_z
                );
                break;
            }
        }
    }
    let size_z = stacks.len();
    let declared_t = image.pixels.size_t.max(1);
    let size_t = declared_t.min((MAX_PLANE_STACKS / size_z as u64).max(1));
    if size_t < declared_t {
        warn!("{file_name}: showing {size_t} of {declared_t} time points");
    }
    'time: for t in 1..size_t {
        let mut time_point = Vec::with_capacity(size_z);
        for z in 0..size_z as u64 {
            match stack_at(z, t) {
                Ok(stack) => time_point.push(stack),
                Err(reason) => {
                    warn!(
                        "{file_name}: time point {} cannot be shown ({reason} at focal plane {}); showing {t} of {declared_t} time points",
                        t + 1,
                        z + 1
                    );
                    break 'time;
                }
            }
        }
        stacks.extend(time_point);
    }

    let mut associated = Vec::new();
    for other in &doc.images {
        let Some(name) = associated_name(other) else {
            continue;
        };
        if associated.iter().any(|a: &Associated| a.name == name) {
            continue;
        }
        let Ok(planes) = image_planes(&mut table, &other.pixels, own.0, own.1, (0, 0)) else {
            continue;
        };
        let ifd = Arc::clone(&planes.planes[0].ifd);
        let pixels = u64::from(ifd.width) * u64::from(ifd.height);
        if ifd.sample_type == SampleType::U8
            && (ifd.samples_per_pixel == 1 || ifd.is_color())
            && codec::unsupported(&ifd).is_none()
            && pixels <= MAX_ASSOCIATED_IMAGE_PIXELS
        {
            associated.push(Associated {
                name: name.to_string(),
                ifd,
            });
        }
    }
    associated.sort_by(|a, b| a.name.cmp(&b.name));

    let reader = ChunkReader::new(tiff);
    let content = match base.layout {
        Layout::Rgb => Content::Rgb,
        Layout::Channels => {
            let mut channels = channel_infos(image, &base);
            // The finest level small enough to read whole.
            let stats_level = levels
                .iter()
                .find(|level| level.width * level.height <= STATS_PIXELS)
                .unwrap_or_else(|| levels.last().expect("level 0"));
            let stats: Vec<ChannelStats> = stats_level
                .planes
                .par_iter()
                .map(|plane| reader.channel_stats(stats_level, plane))
                .collect();
            for (channel, stats) in channels.iter_mut().zip(stats) {
                channel.stats = stats;
            }
            let set = ChannelSet {
                channels,
                sample_type: base_ifd.sample_type,
                significant_bits: image.pixels.significant_bits,
                size_z: image.pixels.size_z,
                size_t: image.pixels.size_t,
            };
            let defaults = set.default_display();
            let compositor = Compositor::new(defaults.clone(), set.sample_type, None);
            Content::Channels {
                set,
                defaults,
                display: RwLock::new(Arc::new(compositor)),
            }
        }
    };
    let background = match content {
        Content::Rgb => [255, 255, 255],
        Content::Channels { .. } => [0, 0, 0],
    };

    let pixels = &image.pixels;
    let properties = WsiProperties {
        path: path.to_path_buf(),
        filename: file_name.clone(),
        vendor: Some(VENDOR.to_string()),
        mpp_x: mpp(pixels.physical_size_x.as_ref()).or(resolution_mpp(&base_ifd).0),
        mpp_y: mpp(pixels.physical_size_y.as_ref()).or(resolution_mpp(&base_ifd).1),
        objective_power: doc.objective_of(image).and_then(|objective| {
            objective
                .nominal_magnification
                .or(objective.calibrated_magnification)
                .filter(|power| *power <= 1_000.0)
        }),
        scan_date: image
            .acquisition_date
            .clone()
            .or_else(|| base_ifd.date_time.clone()),
        stain: None,
        levels: levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                let ifd = &level.planes[0].ifd;
                WsiLevel {
                    level: index as u32,
                    width: level.width,
                    height: level.height,
                    downsample: level.downsample,
                    tile_width: ifd.tiled.then_some(ifd.chunk_width),
                    tile_height: ifd.tiled.then_some(ifd.chunk_height),
                    synthetic: false,
                }
            })
            .collect(),
        width: w0,
        height: h0,
        bounds: None,
        regions: Vec::new(),
        background_color: background,
        quickhash1: None,
        comment: image.description.clone(),
        icc_profile_size: match content {
            Content::Rgb => base_ifd.icc_profile.map(|(_, len)| len),
            Content::Channels { .. } => None,
        },
    };
    let channel_list = match &content {
        Content::Channels { set, .. } => Some(set.channels.as_slice()),
        Content::Rgb => None,
    };
    let property_map = property_map(
        &doc,
        image,
        &base_ifd,
        &levels,
        channel_list,
        &properties,
        pyramid_kind,
        &associated,
    );
    let associated_images = associated
        .iter()
        .map(|image| AssociatedImage {
            name: image.name.clone(),
            width: u64::from(image.ifd.width),
            height: u64::from(image.ifd.height),
            icc_profile_size: image.ifd.icc_profile.map(|(_, len)| len),
        })
        .collect();
    debug!(
        "{file_name}: OME-TIFF, {} levels ({}), {}",
        levels.len(),
        pyramid_kind.name(),
        match &content {
            Content::Rgb => "RGB".to_string(),
            Content::Channels { set, .. } =>
                format!("{} {} channels", set.len(), set.sample_type.name()),
        }
    );
    Ok(Opened {
        source: Box::new(OmeTiffSource {
            reader,
            levels,
            stacks,
            size_z,
            focal_plane: AtomicUsize::new(0),
            time_point: AtomicUsize::new(0),
            content,
            background,
            associated,
        }),
        properties,
        property_map,
        associated_images,
    })
}

impl OmeTiffSource {
    /// The planes of `level` at focal plane `z` and time point `t` (clamped
    /// to the planes there are).
    fn planes_at(&self, level: usize, z: usize, t: usize) -> &[PlaneRef] {
        let z = z.min(self.size_z - 1);
        let t = t.min(self.stacks.len() / self.size_z - 1);
        &self.stacks[t * self.size_z + z][level]
    }

    /// Reads `region` at the current plane and display, or at `pin`'s.
    fn read_with(
        &self,
        region: Region,
        mode: AlphaMode,
        out: &mut [u8],
        pin: Option<&Pin>,
    ) -> Result<()> {
        let level = &self.levels[region.level as usize];
        let no_data = match mode {
            AlphaMode::Composited => {
                let [r, g, b] = self.background;
                [r, g, b, 255]
            }
            AlphaMode::Straight | AlphaMode::Premultiplied => [0, 0, 0, 0],
        };
        for px in out.as_chunks_mut::<4>().0 {
            *px = no_data;
        }
        let origin_x = level_coordinate(region.x, level.downsample);
        let origin_y = level_coordinate(region.y, level.downsample);
        let Some(clip) = Clip::new(
            origin_x,
            origin_y,
            region.width,
            region.height,
            (level.width, level.height),
        ) else {
            return Ok(());
        };
        let (z, t) = match pin {
            Some(pin) => (pin.focal_plane as usize, pin.time_point as usize),
            None => (
                self.focal_plane.load(Ordering::Relaxed),
                self.time_point.load(Ordering::Relaxed),
            ),
        };
        let planes = self.planes_at(region.level as usize, z, t);
        let result = match &self.content {
            Content::Rgb => self.copy_color(&planes[0].ifd, &clip, out),
            Content::Channels { display, .. } => {
                let compositor = pin
                    .and_then(|pin| pin.compositor.clone())
                    .unwrap_or_else(|| Arc::clone(&display.read()));
                self.composite(planes, &compositor, &clip, out)
            }
        };
        if matches!(mode, AlphaMode::Composited) {
            // Chunks the writer left out are transparent: show background.
            for px in out.as_chunks_mut::<4>().0 {
                if px[3] == 0 {
                    *px = no_data;
                }
            }
        }
        result.map_err(|error| self.region_error(error, region))
    }

    fn region_error(&self, error: ChunkError, region: Region) -> Error {
        match error {
            ChunkError::Io(error) => Error::Io(error),
            ChunkError::Data(message) => Error::CorruptRegion {
                level: region.level,
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                message,
            },
        }
    }

    /// Copies colour (or grey) pixels of `ifd` inside `clip` to `out`;
    /// chunks the writer left out stay as they are (no data).
    fn copy_color(
        &self,
        ifd: &Ifd,
        clip: &Clip,
        out: &mut [u8],
    ) -> std::result::Result<(), ChunkError> {
        let color = ifd.samples_per_pixel >= 3;
        for part in chunk_parts(ifd, clip) {
            let samples: Vec<(Arc<DecodedChunk>, usize)> = if ifd.planar == Planar::Chunky {
                let chunk = self.reader.chunk(ifd, part.tx, part.ty, 0)?;
                (0..if color { 3 } else { 1 })
                    .map(|s| (Arc::clone(&chunk), s))
                    .collect()
            } else {
                (0..if color { 3 } else { 1 })
                    .map(|s| Ok((self.reader.chunk(ifd, part.tx, part.ty, s as u16)?, 0)))
                    .collect::<std::result::Result<_, ChunkError>>()?
            };
            if samples.iter().any(|(chunk, _)| chunk.missing) {
                continue;
            }
            for y in part.y0..part.y1 {
                let mut o = clip.out_index(part.x0, y);
                for x in part.x0..part.x1 {
                    for (channel, (chunk, sample)) in samples.iter().enumerate() {
                        let spp = usize::from(chunk.samples);
                        let i = ((y - part.top) as usize * chunk.width as usize
                            + (x - part.left) as usize)
                            * spp
                            + sample;
                        out[o + channel] = chunk.data[i];
                    }
                    if !color {
                        out[o + 1] = out[o];
                        out[o + 2] = out[o];
                    }
                    out[o + 3] = 255;
                    o += 4;
                }
            }
        }
        Ok(())
    }

    fn composite(
        &self,
        planes: &[PlaneRef],
        compositor: &Compositor,
        clip: &Clip,
        out: &mut [u8],
    ) -> std::result::Result<(), ChunkError> {
        for y in clip.y0..clip.y1 {
            let start = clip.out_index(clip.x0, y);
            let end = clip.out_index(clip.x1 - 1, y) + 4;
            for px in out[start..end].as_chunks_mut::<4>().0 {
                *px = [0, 0, 0, 255];
            }
        }
        for (channel, lut) in compositor.luts.iter().enumerate() {
            let Some(lut) = lut else {
                continue;
            };
            let plane = &planes[channel];
            let sample = match plane.ifd.planar {
                Planar::Chunky => usize::from(plane.sample),
                Planar::Separate => 0,
            };
            for part in chunk_parts(&plane.ifd, clip) {
                let chunk = self
                    .reader
                    .chunk(&plane.ifd, part.tx, part.ty, plane.sample)?;
                if !chunk.missing {
                    accumulate(lut, &chunk, sample, &part, clip, out);
                }
            }
        }
        Ok(())
    }
}

impl SlideSource for OmeTiffSource {
    fn read_region_into(&self, region: Region, mode: AlphaMode, out: &mut [u8]) -> Result<()> {
        self.read_with(region, mode, out, None)
    }

    fn pin(&self) -> Option<Pin> {
        let compositor = match &self.content {
            Content::Channels { display, .. } => Some(Arc::clone(&display.read())),
            Content::Rgb => None,
        };
        if compositor.is_none() && self.stacks.len() == 1 {
            return None;
        }
        Some(Pin {
            focal_plane: self.focal_plane(),
            time_point: self.time_point(),
            compositor,
        })
    }

    fn read_pinned_into(
        &self,
        region: Region,
        mode: AlphaMode,
        out: &mut [u8],
        pin: &Pin,
    ) -> Result<()> {
        self.read_with(region, mode, out, Some(pin))
    }

    fn read_associated_image(&self, name: &str, _mode: AlphaMode) -> Result<RgbaImageData> {
        let image = self
            .associated
            .iter()
            .find(|image| image.name == name)
            .ok_or_else(|| Error::AssociatedImageNotFound(name.to_string()))?;
        let ifd = &image.ifd;
        let (width, height) = (ifd.width, ifd.height);
        let mut out = pixels::try_alloc(width as usize * height as usize * 4)?;
        let clip = Clip::new(0, 0, width, height, (u64::from(width), u64::from(height)))
            .expect("non-empty image");
        self.copy_color(ifd, &clip, &mut out)
            .map_err(|error| match error {
                ChunkError::Io(error) => Error::Io(error),
                ChunkError::Data(message) => Error::InvalidFormat(format!(
                    "associated image {name:?} is unreadable: {message}"
                )),
            })?;
        for px in out.as_chunks_mut::<4>().0 {
            if px[3] == 0 {
                *px = [255, 255, 255, 255];
            }
        }
        Ok(RgbaImageData {
            width: width as usize,
            height: height as usize,
            pixels: out,
        })
    }

    fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        match self.content {
            Content::Rgb => self.reader.tiff.icc_profile(&self.levels[0].planes[0].ifd),
            Content::Channels { .. } => Ok(None),
        }
    }

    fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let image = self
            .associated
            .iter()
            .find(|image| image.name == name)
            .ok_or_else(|| Error::AssociatedImageNotFound(name.to_string()))?;
        self.reader.tiff.icc_profile(&image.ifd)
    }

    fn health(&self) -> SlideHealth {
        SlideHealth {
            state: if self.reader.failing.load(Ordering::Relaxed) {
                HealthState::Failing
            } else {
                HealthState::Healthy
            },
            bad_regions: self.reader.bad.lock().len(),
            last_error: self.reader.last_error.lock().clone(),
            ..SlideHealth::default()
        }
    }

    fn clear_bad_regions(&self) {
        self.reader.bad.lock().clear();
    }

    fn channels(&self) -> Option<&ChannelSet> {
        match &self.content {
            Content::Channels { set, .. } => Some(set),
            Content::Rgb => None,
        }
    }

    fn channel_display(&self) -> Option<ChannelDisplay> {
        match &self.content {
            Content::Channels { display, .. } => Some(display.read().display.clone()),
            Content::Rgb => None,
        }
    }

    fn default_channel_display(&self) -> Option<ChannelDisplay> {
        match &self.content {
            Content::Channels { defaults, .. } => Some(defaults.clone()),
            Content::Rgb => None,
        }
    }

    fn focal_planes(&self) -> u32 {
        self.size_z as u32
    }

    fn focal_plane(&self) -> u32 {
        self.focal_plane.load(Ordering::Relaxed) as u32
    }

    fn set_focal_plane(&self, z: u32) -> Result<()> {
        if z as usize >= self.size_z {
            return Err(Error::InvalidArgument(format!(
                "focal plane {z} (the slide has {})",
                self.size_z
            )));
        }
        self.focal_plane.store(z as usize, Ordering::Relaxed);
        Ok(())
    }

    fn time_points(&self) -> u32 {
        (self.stacks.len() / self.size_z) as u32
    }

    fn time_point(&self) -> u32 {
        self.time_point.load(Ordering::Relaxed) as u32
    }

    fn set_time_point(&self, t: u32) -> Result<()> {
        let count = self.stacks.len() / self.size_z;
        if t as usize >= count {
            return Err(Error::InvalidArgument(format!(
                "time point {t} (the slide has {count})"
            )));
        }
        self.time_point.store(t as usize, Ordering::Relaxed);
        Ok(())
    }

    fn set_channel_display(&self, new: ChannelDisplay) -> Result<()> {
        let Content::Channels { set, display, .. } = &self.content else {
            return Err(Error::InvalidArgument(
                "the slide has no channels to display".to_string(),
            ));
        };
        let new = new.validated(set)?;
        let previous = Arc::clone(&display.read());
        let compositor = Compositor::new(new, set.sample_type, Some(&previous));
        *display.write() = Arc::new(compositor);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
