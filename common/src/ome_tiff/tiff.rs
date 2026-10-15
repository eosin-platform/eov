//! A small TIFF and BigTIFF container reader.
//!
//! The OME-TIFF reader needs random access to the tiles of SubIFDs, which
//! the `tiff` crate's decoder cannot reach, and positional reads, so that
//! any number of threads decode tiles of one file without sharing a cursor
//! or a lock. This module parses the directory structure and the tags the
//! reader uses. Every offset, count and size comes from the file, so each
//! is checked against the file's length and fixed limits before anything
//! is allocated: a malformed file is an error, never a panic or an abort.

use crate::{Error, Result};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Directory entries accepted in one IFD (real files have a few dozen).
const MAX_IFD_ENTRIES: u64 = 4096;
/// Largest tag value read, in bytes: the offsets of a huge level, or a
/// large OME-XML description.
pub(crate) const MAX_TAG_BYTES: u64 = 256 << 20;
/// Most IFDs followed in the main chain (planes of very long time series).
pub(crate) const MAX_CHAIN_IFDS: usize = 1 << 20;
/// Most SubIFDs (pyramid levels) accepted per IFD.
const MAX_SUB_IFDS: usize = 64;
/// Largest image side accepted (TIFF allows 2^32 - 1, nothing real is near).
const MAX_SIDE: u32 = 1 << 24;

pub(crate) mod tag {
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const MAKE: u16 = 271;
    pub const MODEL: u16 = 272;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const PLANAR_CONFIGURATION: u16 = 284;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const SOFTWARE: u16 = 305;
    pub const DATE_TIME: u16 = 306;
    pub const PREDICTOR: u16 = 317;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SUB_IFDS: u16 = 330;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const JPEG_TABLES: u16 = 347;
    pub const ICC_PROFILE: u16 = 34675;
}

/// Byte order of a TIFF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    pub(crate) fn u16(self, b: &[u8]) -> u16 {
        let b = [b[0], b[1]];
        match self {
            Self::Little => u16::from_le_bytes(b),
            Self::Big => u16::from_be_bytes(b),
        }
    }

    pub(crate) fn u32(self, b: &[u8]) -> u32 {
        let b = [b[0], b[1], b[2], b[3]];
        match self {
            Self::Little => u32::from_le_bytes(b),
            Self::Big => u32::from_be_bytes(b),
        }
    }

    pub(crate) fn u64(self, b: &[u8]) -> u64 {
        let b = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        match self {
            Self::Little => u64::from_le_bytes(b),
            Self::Big => u64::from_be_bytes(b),
        }
    }

    /// Whether multi-byte samples need swapping to native order.
    pub(crate) fn is_native(self) -> bool {
        match self {
            Self::Little => cfg!(target_endian = "little"),
            Self::Big => cfg!(target_endian = "big"),
        }
    }
}

/// A file read with positional reads only, so that threads never share a
/// cursor.
pub(crate) struct PositionalFile {
    file: File,
    #[cfg(not(any(unix, windows)))]
    lock: parking_lot::Mutex<()>,
    len: u64,
}

impl PositionalFile {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            file,
            #[cfg(not(any(unix, windows)))]
            lock: parking_lot::Mutex::new(()),
            len,
        })
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Fills `buf` from `offset`; a read past the end of the file is an
    /// error.
    pub(crate) fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        if offset
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "{} bytes at offset {offset} lie past the end of the file ({} bytes)",
                    buf.len(),
                    self.len
                ),
            ));
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset)
        }
        #[cfg(windows)]
        {
            let mut done = 0;
            while done < buf.len() {
                let read = std::os::windows::fs::FileExt::seek_read(
                    &self.file,
                    &mut buf[done..],
                    offset + done as u64,
                )?;
                if read == 0 {
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
                done += read;
            }
            Ok(())
        }
        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let _guard = self.lock.lock();
            let mut file = &self.file;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(buf)
        }
    }

    /// Reads `len` bytes at `offset` into a new buffer, reporting an
    /// allocation failure as an error.
    pub(crate) fn read_vec_at(&self, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        // Check the range before allocating: a damaged length must not
        // allocate more than the file holds.
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "{len} bytes at offset {offset} lie past the end of the file ({} bytes)",
                    self.len
                ),
            ));
        }
        let len = usize::try_from(len)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "read too large"))?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::OutOfMemory, "read too large"))?;
        buf.resize(len, 0);
        self.read_exact_at(&mut buf, offset)?;
        Ok(buf)
    }
}

/// How the samples of a pixel are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Planar {
    /// Interleaved (`RGBRGB…`); one chunk holds every sample.
    Chunky,
    /// One plane per sample; chunk `s * chunks_per_plane + i` holds sample
    /// `s` of tile or strip `i`.
    Separate,
}

/// The numeric type of a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

impl SampleType {
    pub(crate) fn from_tags(bits: u16, format: u16) -> Option<Self> {
        Some(match (bits, format) {
            (8, 1) => Self::U8,
            (8, 2) => Self::I8,
            (16, 1) => Self::U16,
            (16, 2) => Self::I16,
            (32, 1) => Self::U32,
            (32, 2) => Self::I32,
            (32, 3) => Self::F32,
            (64, 3) => Self::F64,
            _ => return None,
        })
    }

    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// The OME `Pixels@Type` name.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::U16 => "uint16",
            Self::I16 => "int16",
            Self::U32 => "uint32",
            Self::I32 => "int32",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    /// The range of values the type can hold (for floats, the range a
    /// display window defaults to before statistics are known).
    pub fn range(self) -> (f64, f64) {
        match self {
            Self::U8 => (0.0, 255.0),
            Self::I8 => (-128.0, 127.0),
            Self::U16 => (0.0, 65_535.0),
            Self::I16 => (-32_768.0, 32_767.0),
            Self::U32 => (0.0, f64::from(u32::MAX)),
            Self::I32 => (f64::from(i32::MIN), f64::from(i32::MAX)),
            Self::F32 | Self::F64 => (0.0, 1.0),
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

/// A TIFF compression scheme, by its tag value.
pub(crate) fn compression_name(code: u16) -> String {
    let name = match code {
        1 => "none",
        2 => "CCITT modified Huffman",
        3 => "CCITT Group 3 fax",
        4 => "CCITT Group 4 fax",
        5 => "LZW",
        6 => "old-style JPEG",
        7 => "JPEG",
        8 | 32946 => "Deflate",
        32773 => "PackBits",
        33003 | 33005 | 34712 => "JPEG 2000",
        34887 => "LERC",
        34925 => "LZMA",
        50000 => "Zstandard",
        50001 => "WebP",
        50002 => "JPEG XL",
        _ => return format!("compression {code}"),
    };
    format!("{name} ({code})")
}

/// One image file directory: an image (a plane, or one pyramid level of
/// it) with the tags the reader uses.
#[derive(Debug)]
pub(crate) struct Ifd {
    /// File offset of the directory; identifies it.
    pub offset: u64,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u16,
    pub sample_type: SampleType,
    pub photometric: u16,
    pub compression: u16,
    pub predictor: u16,
    pub planar: Planar,
    pub new_subfile_type: u32,
    /// Tile size, or for strips the image width and the rows per strip.
    pub chunk_width: u32,
    pub chunk_height: u32,
    pub tiled: bool,
    pub chunk_offsets: Vec<u64>,
    pub chunk_byte_counts: Vec<u64>,
    pub jpeg_tables: Option<Arc<[u8]>>,
    pub sub_ifds: Vec<u64>,
    pub description: Option<String>,
    pub software: Option<String>,
    pub date_time: Option<String>,
    pub make: Option<String>,
    pub model: Option<String>,
    /// File offset and length of the ICC profile.
    pub icc_profile: Option<(u64, u64)>,
    /// X and Y resolution with `ResolutionUnit` (2 = inch, 3 = cm).
    pub resolution: Option<(f64, f64, u16)>,
}

impl Ifd {
    /// Tiles (or strips) across and down one sample plane.
    pub(crate) fn chunks_across(&self) -> u32 {
        self.width.div_ceil(self.chunk_width)
    }

    pub(crate) fn chunks_down(&self) -> u32 {
        self.height.div_ceil(self.chunk_height)
    }

    pub(crate) fn chunks_per_plane(&self) -> u32 {
        self.chunks_across() * self.chunks_down()
    }

    /// Samples per pixel stored in one chunk.
    pub(crate) fn samples_per_chunk(&self) -> u16 {
        match self.planar {
            Planar::Chunky => self.samples_per_pixel,
            Planar::Separate => 1,
        }
    }

    /// The chunk holding sample `sample` of tile (`tx`, `ty`).
    pub(crate) fn chunk_index(&self, tx: u32, ty: u32, sample: u16) -> u32 {
        let base = ty * self.chunks_across() + tx;
        match self.planar {
            Planar::Chunky => base,
            Planar::Separate => u32::from(sample) * self.chunks_per_plane() + base,
        }
    }

    /// Rows of image data in chunk row `ty` (strips at the bottom are cut
    /// short; tiles are always full and padded).
    pub(crate) fn chunk_rows(&self, ty: u32) -> u32 {
        if self.tiled {
            self.chunk_height
        } else {
            (self.height - ty * self.chunk_height).min(self.chunk_height)
        }
    }

    /// Bytes of a decoded chunk of chunk row `ty`.
    pub(crate) fn decoded_chunk_bytes(&self, ty: u32) -> u64 {
        u64::from(self.chunk_width)
            * u64::from(self.chunk_rows(ty))
            * u64::from(self.samples_per_chunk())
            * self.sample_type.bytes() as u64
    }

    /// Whether pixels are RGB colour (rather than samples of one value).
    pub(crate) fn is_color(&self) -> bool {
        matches!(self.photometric, 2 | 6) && self.samples_per_pixel >= 3
    }
}

/// An open TIFF or BigTIFF file.
pub(crate) struct TiffFile {
    pub path: PathBuf,
    pub file: PositionalFile,
    pub order: ByteOrder,
    pub big: bool,
    pub first_ifd: u64,
}

/// A raw directory entry.
struct Entry {
    tag: u16,
    ty: u16,
    count: u64,
    /// The value field (4 bytes in TIFF, 8 in BigTIFF).
    field: [u8; 8],
}

/// Bytes per value of a TIFF field type.
fn type_size(ty: u16) -> Option<u64> {
    Some(match ty {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => return None,
    })
}

fn invalid(path: &Path, message: impl std::fmt::Display) -> Error {
    Error::InvalidFormat(format!(
        "{}: {message}",
        path.file_name().unwrap_or_default().to_string_lossy()
    ))
}

impl TiffFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = PositionalFile::open(path)?;
        let mut head = [0u8; 16];
        let available = file.len().min(16) as usize;
        file.read_exact_at(&mut head[..available], 0)?;
        let order = match &head[0..2] {
            b"II" => ByteOrder::Little,
            b"MM" => ByteOrder::Big,
            _ => return Err(invalid(path, "not a TIFF file")),
        };
        if available < 8 {
            return Err(invalid(path, "the TIFF header is truncated"));
        }
        let (big, first_ifd) = match order.u16(&head[2..4]) {
            42 => (false, u64::from(order.u32(&head[4..8]))),
            43 => {
                if available < 16 || order.u16(&head[4..6]) != 8 || order.u16(&head[6..8]) != 0 {
                    return Err(invalid(path, "malformed BigTIFF header"));
                }
                (true, order.u64(&head[8..16]))
            }
            other => return Err(invalid(path, format!("bad TIFF version {other}"))),
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
            order,
            big,
            first_ifd,
        })
    }

    pub(crate) fn invalid(&self, message: impl std::fmt::Display) -> Error {
        invalid(&self.path, message)
    }

    fn offset_size(&self) -> u64 {
        if self.big { 8 } else { 4 }
    }

    /// Reads the entry table of the IFD at `offset` and the offset of the
    /// next IFD in the chain (0 at the end).
    fn read_entries(&self, offset: u64) -> Result<(Vec<Entry>, u64)> {
        let (count_size, entry_size) = if self.big { (8, 20) } else { (2, 12) };
        if offset < 8 || offset >= self.file.len() {
            return Err(self.invalid(format!("IFD offset {offset} is outside the file")));
        }
        let mut count_bytes = [0u8; 8];
        self.file
            .read_exact_at(&mut count_bytes[..count_size as usize], offset)
            .map_err(|e| self.invalid(format!("unreadable IFD at {offset}: {e}")))?;
        let count = if self.big {
            self.order.u64(&count_bytes)
        } else {
            u64::from(self.order.u16(&count_bytes))
        };
        if count == 0 || count > MAX_IFD_ENTRIES {
            return Err(self.invalid(format!("IFD at {offset} has {count} entries")));
        }
        let table_len = count * entry_size + self.offset_size();
        let table = self
            .file
            .read_vec_at(offset + count_size, table_len)
            .map_err(|e| self.invalid(format!("unreadable IFD at {offset}: {e}")))?;
        let entries = table[..(count * entry_size) as usize]
            .chunks_exact(entry_size as usize)
            .map(|raw| {
                let mut field = [0u8; 8];
                let (count, value) = if self.big {
                    (self.order.u64(&raw[4..12]), &raw[12..20])
                } else {
                    (u64::from(self.order.u32(&raw[4..8])), &raw[8..12])
                };
                field[..value.len()].copy_from_slice(value);
                Entry {
                    tag: self.order.u16(&raw[0..2]),
                    ty: self.order.u16(&raw[2..4]),
                    count,
                    field,
                }
            })
            .collect();
        let next_at = (count * entry_size) as usize;
        let next = if self.big {
            self.order.u64(&table[next_at..next_at + 8])
        } else {
            u64::from(self.order.u32(&table[next_at..next_at + 4]))
        };
        Ok((entries, next))
    }

    /// The offsets of the IFDs of the main chain, in order.
    pub(crate) fn chain(&self) -> Result<Vec<u64>> {
        let mut offsets = Vec::new();
        let mut seen = HashSet::new();
        let mut next = self.first_ifd;
        while next != 0 {
            if !seen.insert(next) {
                return Err(self.invalid("the IFD chain loops"));
            }
            if offsets.len() >= MAX_CHAIN_IFDS {
                return Err(self.invalid(format!("more than {MAX_CHAIN_IFDS} IFDs")));
            }
            offsets.push(next);
            next = self.next_ifd(next)?;
        }
        if offsets.is_empty() {
            return Err(self.invalid("the file has no IFD"));
        }
        Ok(offsets)
    }

    /// The offset of the IFD after the one at `offset` (0 at the end),
    /// reading only the entry count and the next pointer.
    fn next_ifd(&self, offset: u64) -> Result<u64> {
        let (count_size, entry_size) = if self.big { (8u64, 20u64) } else { (2, 12) };
        let mut count_bytes = [0u8; 8];
        self.file
            .read_exact_at(&mut count_bytes[..count_size as usize], offset)
            .map_err(|e| self.invalid(format!("unreadable IFD at {offset}: {e}")))?;
        let count = if self.big {
            self.order.u64(&count_bytes)
        } else {
            u64::from(self.order.u16(&count_bytes))
        };
        if count > MAX_IFD_ENTRIES {
            return Err(self.invalid(format!("IFD at {offset} has {count} entries")));
        }
        let at = offset + count_size + count * entry_size;
        let mut next = [0u8; 8];
        let size = self.offset_size() as usize;
        self.file
            .read_exact_at(&mut next[..size], at)
            .map_err(|e| self.invalid(format!("unreadable IFD at {offset}: {e}")))?;
        Ok(if self.big {
            self.order.u64(&next)
        } else {
            u64::from(self.order.u32(&next))
        })
    }

    /// The bytes of an entry's value.
    fn value_bytes(&self, entry: &Entry) -> Result<Vec<u8>> {
        let size = type_size(entry.ty).ok_or_else(|| {
            self.invalid(format!("tag {} has unknown type {}", entry.tag, entry.ty))
        })?;
        let len = entry
            .count
            .checked_mul(size)
            .filter(|&len| len <= MAX_TAG_BYTES)
            .ok_or_else(|| self.invalid(format!("tag {} is too large", entry.tag)))?;
        if len <= self.offset_size() {
            return Ok(entry.field[..len as usize].to_vec());
        }
        let offset = self.field_offset(entry);
        self.file
            .read_vec_at(offset, len)
            .map_err(|e| self.invalid(format!("tag {} is unreadable: {e}", entry.tag)))
    }

    /// The file range of an entry's value (for values read later, like
    /// ICC profiles).
    fn value_range(&self, entry: &Entry) -> Result<(u64, u64)> {
        let size = type_size(entry.ty).ok_or_else(|| {
            self.invalid(format!("tag {} has unknown type {}", entry.tag, entry.ty))
        })?;
        let len = entry
            .count
            .checked_mul(size)
            .filter(|&len| len <= MAX_TAG_BYTES)
            .ok_or_else(|| self.invalid(format!("tag {} is too large", entry.tag)))?;
        if len <= self.offset_size() {
            return Err(self.invalid(format!("tag {} is stored inline", entry.tag)));
        }
        let offset = self.field_offset(entry);
        if offset
            .checked_add(len)
            .is_none_or(|end| end > self.file.len())
        {
            return Err(self.invalid(format!("tag {} lies past the end of the file", entry.tag)));
        }
        Ok((offset, len))
    }

    /// An entry's values as unsigned integers.
    fn unsigned(&self, entry: &Entry) -> Result<Vec<u64>> {
        let bytes = self.value_bytes(entry)?;
        let order = self.order;
        let values: Vec<u64> = match entry.ty {
            1 | 7 => bytes.iter().map(|&b| u64::from(b)).collect(),
            3 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u64::from(order.u16(b)))
                .collect(),
            4 | 13 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u64::from(order.u32(b)))
                .collect(),
            16 | 18 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| order.u64(b))
                .collect(),
            _ => {
                return Err(self.invalid(format!(
                    "tag {} has type {}, expected an unsigned integer",
                    entry.tag, entry.ty
                )));
            }
        };
        Ok(values)
    }

    fn first_unsigned(&self, entry: &Entry) -> Result<u64> {
        self.unsigned(entry)?
            .first()
            .copied()
            .ok_or_else(|| self.invalid(format!("tag {} is empty", entry.tag)))
    }

    fn ascii(&self, entry: &Entry) -> Result<String> {
        let bytes = self.value_bytes(entry)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    fn rational(&self, entry: &Entry) -> Option<f64> {
        if entry.ty != 5 {
            return None;
        }
        let bytes = self.value_bytes(entry).ok()?;
        let num = self.order.u32(bytes.get(0..4)?);
        let den = self.order.u32(bytes.get(4..8)?);
        (den != 0).then(|| f64::from(num) / f64::from(den))
    }

    /// The file offset an out-of-line entry's value starts at.
    fn field_offset(&self, entry: &Entry) -> u64 {
        if self.big {
            self.order.u64(&entry.field)
        } else {
            u64::from(self.order.u32(&entry.field))
        }
    }

    /// The start of the ImageDescription of the first IFD: at most `limit`
    /// bytes, enough to tell what the file is without reading a large
    /// OME-XML document.
    pub(crate) fn first_description_start(&self, limit: u64) -> Result<Option<String>> {
        let (entries, _) = self.read_entries(self.first_ifd)?;
        let Some(entry) = entries
            .iter()
            .find(|e| e.tag == tag::IMAGE_DESCRIPTION && e.ty == 2)
        else {
            return Ok(None);
        };
        let bytes = if entry.count <= self.offset_size() {
            entry.field[..entry.count as usize].to_vec()
        } else {
            let offset = self.field_offset(entry);
            let len = entry
                .count
                .min(limit)
                .min(self.file.len().saturating_sub(offset));
            self.file.read_vec_at(offset, len)?
        };
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(Some(String::from_utf8_lossy(&bytes[..end]).into_owned()))
    }

    /// Parses the IFD at `offset`.
    pub(crate) fn ifd(&self, offset: u64) -> Result<Ifd> {
        let (entries, _) = self.read_entries(offset)?;
        let find = |tag: u16| entries.iter().find(|e| e.tag == tag);
        let required = |tag: u16, name: &str| {
            find(tag).ok_or_else(|| self.invalid(format!("IFD at {offset} has no {name}")))
        };
        let width = self.first_unsigned(required(tag::IMAGE_WIDTH, "ImageWidth")?)?;
        let height = self.first_unsigned(required(tag::IMAGE_LENGTH, "ImageLength")?)?;
        if width == 0 || height == 0 || width > u64::from(MAX_SIDE) || height > u64::from(MAX_SIDE)
        {
            return Err(self.invalid(format!("IFD at {offset} is {width} x {height} pixels")));
        }
        let (width, height) = (width as u32, height as u32);
        let samples_per_pixel = match find(tag::SAMPLES_PER_PIXEL) {
            Some(entry) => self.first_unsigned(entry)?,
            None => 1,
        };
        if samples_per_pixel == 0 || samples_per_pixel > 1024 {
            return Err(self.invalid(format!(
                "IFD at {offset} has {samples_per_pixel} samples per pixel"
            )));
        }
        let samples_per_pixel = samples_per_pixel as u16;
        let bits = match find(tag::BITS_PER_SAMPLE) {
            Some(entry) => self.unsigned(entry)?,
            None => vec![1],
        };
        let formats = match find(tag::SAMPLE_FORMAT) {
            Some(entry) => self.unsigned(entry)?,
            None => vec![1],
        };
        if bits.is_empty() || bits.iter().any(|&b| b != bits[0]) {
            return Err(Error::Unsupported(format!(
                "samples of different bit depths in one image ({bits:?})"
            )));
        }
        if formats.is_empty() || formats.iter().any(|&f| f != formats[0]) {
            return Err(Error::Unsupported(
                "samples of different types in one image".to_string(),
            ));
        }
        let sample_type =
            SampleType::from_tags(bits[0] as u16, formats[0] as u16).ok_or_else(|| {
                Error::Unsupported(format!(
                    "{}-bit {} samples",
                    bits[0],
                    match formats[0] {
                        1 => "unsigned",
                        2 => "signed",
                        3 => "floating-point",
                        _ => "untyped",
                    }
                ))
            })?;
        let small = |tag: u16, default: u64| -> Result<u64> {
            Ok(match find(tag) {
                Some(entry) => self.first_unsigned(entry)?,
                None => default,
            })
        };
        let photometric = small(tag::PHOTOMETRIC, 1)? as u16;
        let compression = small(tag::COMPRESSION, 1)? as u16;
        let predictor = small(tag::PREDICTOR, 1)? as u16;
        let planar = match small(tag::PLANAR_CONFIGURATION, 1)? {
            2 if samples_per_pixel > 1 => Planar::Separate,
            _ => Planar::Chunky,
        };
        let new_subfile_type = small(tag::NEW_SUBFILE_TYPE, 0)? as u32;

        let tiled = find(tag::TILE_WIDTH).is_some();
        let (chunk_width, chunk_height, offsets, counts) = if tiled {
            let tile_width = self.first_unsigned(required(tag::TILE_WIDTH, "TileWidth")?)?;
            let tile_height = self.first_unsigned(required(tag::TILE_LENGTH, "TileLength")?)?;
            if tile_width == 0
                || tile_height == 0
                || tile_width > u64::from(MAX_SIDE)
                || tile_height > u64::from(MAX_SIDE)
            {
                return Err(self.invalid(format!(
                    "IFD at {offset} has {tile_width} x {tile_height} tiles"
                )));
            }
            let offsets = self.unsigned(required(tag::TILE_OFFSETS, "TileOffsets")?)?;
            let counts = match find(tag::TILE_BYTE_COUNTS) {
                Some(entry) => self.unsigned(entry)?,
                None => Vec::new(),
            };
            (tile_width as u32, tile_height as u32, offsets, counts)
        } else {
            let rows = small(tag::ROWS_PER_STRIP, u64::from(u32::MAX))?.clamp(1, u64::from(height));
            let offsets = self.unsigned(required(tag::STRIP_OFFSETS, "StripOffsets")?)?;
            let counts = match find(tag::STRIP_BYTE_COUNTS) {
                Some(entry) => self.unsigned(entry)?,
                None => Vec::new(),
            };
            (width, rows as u32, offsets, counts)
        };

        let jpeg_tables = match find(tag::JPEG_TABLES) {
            Some(entry) if compression == 7 => Some(Arc::from(self.value_bytes(entry)?)),
            _ => None,
        };
        let sub_ifds = match find(tag::SUB_IFDS) {
            Some(entry) => {
                let mut values = self.unsigned(entry)?;
                values.truncate(MAX_SUB_IFDS);
                values
            }
            None => Vec::new(),
        };
        let text = |tag: u16| -> Option<String> {
            find(tag)
                .filter(|entry| entry.ty == 2)
                .and_then(|entry| self.ascii(entry).ok())
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
        };
        let icc_profile = find(tag::ICC_PROFILE).and_then(|entry| self.value_range(entry).ok());
        let resolution = match (find(tag::X_RESOLUTION), find(tag::Y_RESOLUTION)) {
            (Some(x), Some(y)) => self.rational(x).zip(self.rational(y)).map(|(x, y)| {
                let unit = small(tag::RESOLUTION_UNIT, 2).unwrap_or(2) as u16;
                (x, y, unit)
            }),
            _ => None,
        };

        let mut ifd = Ifd {
            offset,
            width,
            height,
            samples_per_pixel,
            sample_type,
            photometric,
            compression,
            predictor,
            planar,
            new_subfile_type,
            chunk_width,
            chunk_height,
            tiled,
            chunk_offsets: offsets,
            chunk_byte_counts: counts,
            jpeg_tables,
            sub_ifds,
            description: text(tag::IMAGE_DESCRIPTION),
            software: text(tag::SOFTWARE),
            date_time: text(tag::DATE_TIME),
            make: text(tag::MAKE),
            model: text(tag::MODEL),
            icc_profile,
            resolution,
        };
        self.validate_chunks(&mut ifd)?;
        Ok(ifd)
    }

    /// Checks the chunk tables against the image layout and the file.
    fn validate_chunks(&self, ifd: &mut Ifd) -> Result<()> {
        let planes = match ifd.planar {
            Planar::Chunky => 1,
            Planar::Separate => u64::from(ifd.samples_per_pixel),
        };
        let expected = u64::from(ifd.chunks_per_plane()) * planes;
        if (ifd.chunk_offsets.len() as u64) < expected {
            return Err(self.invalid(format!(
                "IFD at {} lists {} of its {expected} tiles or strips",
                ifd.offset,
                ifd.chunk_offsets.len()
            )));
        }
        ifd.chunk_offsets.truncate(expected as usize);
        if ifd.chunk_byte_counts.is_empty() && ifd.compression == 1 && expected == 1 {
            // Some writers leave out StripByteCounts of a single
            // uncompressed strip; its size follows from the layout.
            ifd.chunk_byte_counts.push(ifd.decoded_chunk_bytes(0));
        }
        if (ifd.chunk_byte_counts.len() as u64) < expected {
            return Err(self.invalid(format!(
                "IFD at {} has byte counts for {} of its {expected} tiles or strips",
                ifd.offset,
                ifd.chunk_byte_counts.len()
            )));
        }
        ifd.chunk_byte_counts.truncate(expected as usize);
        Ok(())
    }

    /// Reads the compressed bytes of chunk `index` of `ifd`; `None` for a
    /// chunk the writer left out (offset or byte count 0).
    ///
    /// Damaged chunk tables (data past the end of the file, as in a
    /// truncated copy) are `InvalidData` or `UnexpectedEof` errors: the
    /// chunk is unreadable, the rest of the file may be fine.
    pub(crate) fn read_chunk(&self, ifd: &Ifd, index: u32) -> std::io::Result<Option<Vec<u8>>> {
        let bad = |message: String| std::io::Error::new(std::io::ErrorKind::InvalidData, message);
        let (Some(&offset), Some(&count)) = (
            ifd.chunk_offsets.get(index as usize),
            ifd.chunk_byte_counts.get(index as usize),
        ) else {
            return Err(bad(format!("no tile {index} in the IFD at {}", ifd.offset)));
        };
        if offset == 0 || count == 0 {
            return Ok(None);
        }
        if count > MAX_TAG_BYTES {
            return Err(bad(format!("a tile of {count} bytes")));
        }
        if offset
            .checked_add(count)
            .is_none_or(|end| end > self.file.len())
        {
            return Err(bad(format!(
                "tile data at {offset} ({count} bytes) lies past the end of the file"
            )));
        }
        self.file.read_vec_at(offset, count).map(Some)
    }

    /// The bytes of the ICC profile of `ifd`.
    pub(crate) fn icc_profile(&self, ifd: &Ifd) -> Result<Option<Vec<u8>>> {
        let Some((offset, len)) = ifd.icc_profile else {
            return Ok(None);
        };
        Ok(Some(self.file.read_vec_at(offset, len)?))
    }
}
