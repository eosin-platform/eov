//! A TIFF file writer for images whose pixel data is streamed first and
//! whose directories are written last.
//!
//! [`TiffFile::append`] writes tile or strip data anywhere in the file and
//! returns its offset; [`TiffFile::finish`] writes the image file
//! directories (IFDs) with their out-of-line values, links them, and only
//! then picks the container: classic TIFF when every offset fits in 32
//! bits, BigTIFF otherwise (or when asked). That works because the header
//! slot reserved at the start is large enough for either header, and pixel
//! data offsets are absolute positions that do not depend on the choice.
//!
//! Directories form the main chain in order; an [`Ifd`] may carry
//! [`sub_ifds`](Ifd::sub_ifds) (OME-TIFF pyramids), which are written
//! before their parent and listed in its `SubIFDs` tag (330).

use std::collections::BTreeMap;
use std::io::{self, Seek, SeekFrom, Write};

/// Bytes reserved at the start of the file for the header (BigTIFF's is
/// 16; a classic header uses the first 8).
const HEADER_SLOT: u64 = 16;

/// Tags this writer fills in itself.
pub(crate) const TAG_SUB_IFDS: u16 = 330;

/// A tag value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Short(Vec<u16>),
    Long(Vec<u32>),
    /// Numerator / denominator pairs.
    Rational(Vec<(u32, u32)>),
    /// 7-bit text without the terminating NUL (added when written).
    Ascii(Vec<u8>),
    Undefined(Vec<u8>),
    /// File offsets or byte counts: `LONG` in classic TIFF, `LONG8` in
    /// BigTIFF.
    Offsets(Vec<u64>),
}

impl Value {
    /// TIFF field type and value count.
    fn type_and_count(&self, big: bool) -> (u16, u64) {
        match self {
            Value::Short(v) => (3, v.len() as u64),
            Value::Long(v) => (4, v.len() as u64),
            Value::Rational(v) => (5, v.len() as u64),
            Value::Ascii(v) => (2, v.len() as u64 + 1),
            Value::Undefined(v) => (7, v.len() as u64),
            Value::Offsets(v) => (if big { 16 } else { 4 }, v.len() as u64),
        }
    }

    /// The value's bytes (little-endian).
    fn bytes(&self, big: bool) -> io::Result<Vec<u8>> {
        Ok(match self {
            Value::Short(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Long(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Rational(v) => v
                .iter()
                .flat_map(|(n, d)| n.to_le_bytes().into_iter().chain(d.to_le_bytes()))
                .collect(),
            Value::Ascii(v) => {
                let mut bytes = v.clone();
                bytes.push(0);
                bytes
            }
            Value::Undefined(v) => v.clone(),
            Value::Offsets(v) if big => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Offsets(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for &x in v {
                    let x = u32::try_from(x).map_err(|_| too_big())?;
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                bytes
            }
        })
    }
}

fn too_big() -> io::Error {
    io::Error::other("offset does not fit a classic TIFF")
}

/// One image file directory.
#[derive(Debug, Clone, Default)]
pub(crate) struct Ifd {
    pub entries: BTreeMap<u16, Value>,
    /// Directories listed in this one's `SubIFDs` tag.
    pub sub_ifds: Vec<Ifd>,
}

impl Ifd {
    pub(crate) fn set(&mut self, tag: u16, value: Value) {
        self.entries.insert(tag, value);
    }
}

/// A TIFF file being written.
pub(crate) struct TiffFile<W: Write + Seek> {
    out: W,
    /// End of the data written so far.
    end: u64,
}

impl<W: Write + Seek> TiffFile<W> {
    /// Starts a file: reserves the header slot.
    pub(crate) fn new(mut out: W) -> io::Result<Self> {
        out.write_all(&[0u8; HEADER_SLOT as usize])?;
        Ok(Self {
            out,
            end: HEADER_SLOT,
        })
    }

    /// Appends a block of pixel data and returns its offset. Blocks start
    /// on even offsets, as TIFF requires.
    pub(crate) fn append(&mut self, bytes: &[u8]) -> io::Result<u64> {
        let offset = self.end;
        self.out.write_all(bytes)?;
        self.end += bytes.len() as u64;
        if self.end % 2 == 1 {
            self.out.write_all(&[0])?;
            self.end += 1;
        }
        Ok(offset)
    }

    /// Writes `chain` (the main IFD chain, first directory first) and the
    /// header, and returns the writer and whether the file is a BigTIFF.
    pub(crate) fn finish(mut self, chain: &[Ifd], force_big: bool) -> io::Result<(W, bool)> {
        if chain.is_empty() {
            return Err(io::Error::other("a TIFF file needs at least one image"));
        }
        let base = self.end;
        let classic = if force_big {
            None
        } else {
            layout(chain, base, false)
                .ok()
                .filter(|(tail, _)| base + tail.len() as u64 <= u64::from(u32::MAX))
        };
        let (big, (tail, first)) = match classic {
            Some(laid_out) => (false, laid_out),
            None => (true, layout(chain, base, true)?),
        };
        self.out.write_all(&tail)?;
        self.end += tail.len() as u64;
        self.out.seek(SeekFrom::Start(0))?;
        if big {
            self.out.write_all(b"II")?;
            self.out.write_all(&43u16.to_le_bytes())?;
            self.out.write_all(&8u16.to_le_bytes())?;
            self.out.write_all(&0u16.to_le_bytes())?;
            self.out.write_all(&first.to_le_bytes())?;
        } else {
            self.out.write_all(b"II")?;
            self.out.write_all(&42u16.to_le_bytes())?;
            self.out.write_all(&(first as u32).to_le_bytes())?;
        }
        self.out.seek(SeekFrom::Start(self.end))?;
        self.out.flush()?;
        Ok((self.out, big))
    }
}

/// Lays out `chain` as bytes that will start at file offset `base`, and
/// returns them with the offset of the first directory.
fn layout(chain: &[Ifd], base: u64, big: bool) -> io::Result<(Vec<u8>, u64)> {
    let mut tail = Vec::new();
    // The main chain is written last to first, so every directory knows
    // the offset of the next one.
    let mut next = 0u64;
    for ifd in chain.iter().rev() {
        next = write_ifd(&mut tail, base, ifd, next, big)?;
    }
    Ok((tail, next))
}

fn align(tail: &mut Vec<u8>, base: u64) {
    while !(base + tail.len() as u64).is_multiple_of(8) {
        tail.push(0);
    }
}

/// Appends `ifd` (its sub-IFDs and out-of-line values first) and returns
/// its offset.
fn write_ifd(tail: &mut Vec<u8>, base: u64, ifd: &Ifd, next: u64, big: bool) -> io::Result<u64> {
    let mut entries = ifd.entries.clone();
    if !ifd.sub_ifds.is_empty() {
        let mut offsets = Vec::with_capacity(ifd.sub_ifds.len());
        for sub in &ifd.sub_ifds {
            offsets.push(write_ifd(tail, base, sub, 0, big)?);
        }
        entries.insert(TAG_SUB_IFDS, Value::Offsets(offsets));
    }
    let inline = if big { 8 } else { 4 };
    let mut fields = Vec::with_capacity(entries.len());
    for (&tag, value) in &entries {
        let (ty, count) = value.type_and_count(big);
        let bytes = value.bytes(big)?;
        let mut field = vec![0u8; inline];
        if bytes.len() <= inline {
            field[..bytes.len()].copy_from_slice(&bytes);
        } else {
            align(tail, base);
            let at = base + tail.len() as u64;
            tail.extend_from_slice(&bytes);
            if big {
                field.copy_from_slice(&at.to_le_bytes());
            } else {
                let at = u32::try_from(at).map_err(|_| too_big())?;
                field.copy_from_slice(&at.to_le_bytes());
            }
        }
        fields.push((tag, ty, count, field));
    }
    align(tail, base);
    let at = base + tail.len() as u64;
    if big {
        tail.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    } else {
        tail.extend_from_slice(&(fields.len() as u16).to_le_bytes());
    }
    for (tag, ty, count, field) in fields {
        tail.extend_from_slice(&tag.to_le_bytes());
        tail.extend_from_slice(&ty.to_le_bytes());
        if big {
            tail.extend_from_slice(&count.to_le_bytes());
        } else {
            let count = u32::try_from(count).map_err(|_| too_big())?;
            tail.extend_from_slice(&count.to_le_bytes());
        }
        tail.extend_from_slice(&field);
    }
    if big {
        tail.extend_from_slice(&next.to_le_bytes());
    } else {
        let next = u32::try_from(next).map_err(|_| too_big())?;
        tail.extend_from_slice(&next.to_le_bytes());
    }
    if !big && base + tail.len() as u64 > u64::from(u32::MAX) {
        return Err(too_big());
    }
    Ok(at)
}

#[cfg(all(test, target_os = "linux"))]
impl<W: Write + Seek> TiffFile<W> {
    /// Leaves `bytes` unwritten (a hole, in a sparse file).
    fn skip(&mut self, bytes: u64) -> io::Result<()> {
        self.end += bytes;
        self.out.seek(SeekFrom::Start(self.end))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A file whose data reaches past 4 GiB is written as a BigTIFF, and
    /// the image stored there reads back (a sparse file: nothing near 4 GiB
    /// is actually written).
    #[cfg(target_os = "linux")]
    #[test]
    fn files_past_4_gib_become_bigtiff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sparse.tif");
        let mut file = TiffFile::new(std::fs::File::create(&path).unwrap()).unwrap();
        let pixels: Vec<u8> = (0..4 * 3 * 3).map(|i| i as u8 + 1).collect();
        let low = file.append(&pixels).unwrap();
        file.skip(u64::from(u32::MAX)).unwrap();
        let high = file.append(&pixels).unwrap();
        assert!(high > u64::from(u32::MAX));
        let chain = vec![
            strip_image(4, 3, low, pixels.len() as u64),
            strip_image(4, 3, high, pixels.len() as u64),
        ];
        let (_, big) = file.finish(&chain, false).unwrap();
        assert!(big);
        let mut decoder = tiff::decoder::Decoder::new(std::fs::File::open(&path).unwrap()).unwrap();
        decoder.next_image().unwrap();
        let tiff::decoder::DecodingResult::U8(far) = decoder.read_image().unwrap() else {
            panic!("8-bit image expected");
        };
        assert_eq!(far, pixels);
    }

    fn strip_image(width: u32, height: u32, data_at: u64, len: u64) -> Ifd {
        let mut ifd = Ifd::default();
        ifd.set(256, Value::Long(vec![width]));
        ifd.set(257, Value::Long(vec![height]));
        ifd.set(258, Value::Short(vec![8, 8, 8]));
        ifd.set(259, Value::Short(vec![1]));
        ifd.set(262, Value::Short(vec![2]));
        ifd.set(
            270,
            Value::Ascii(b"a description longer than eight bytes".to_vec()),
        );
        ifd.set(273, Value::Offsets(vec![data_at]));
        ifd.set(277, Value::Short(vec![3]));
        ifd.set(278, Value::Long(vec![height]));
        ifd.set(279, Value::Offsets(vec![len]));
        ifd
    }

    fn write(force_big: bool, sub: bool) -> (Vec<u8>, bool) {
        let mut file = TiffFile::new(Cursor::new(Vec::new())).unwrap();
        let pixels: Vec<u8> = (0..4 * 3 * 3).map(|i| i as u8 * 7).collect();
        let at = file.append(&pixels).unwrap();
        let small: Vec<u8> = (0..2 * 2 * 3).map(|i| 200 - i as u8).collect();
        let small_at = file.append(&small).unwrap();
        let mut first = strip_image(4, 3, at, pixels.len() as u64);
        let second = strip_image(2, 2, small_at, small.len() as u64);
        let chain = if sub {
            first.sub_ifds.push(second);
            vec![first]
        } else {
            vec![first, second]
        };
        let (cursor, big) = file.finish(&chain, force_big).unwrap();
        (cursor.into_inner(), big)
    }

    #[test]
    fn classic_and_big_files_decode() {
        for force_big in [false, true] {
            let (bytes, big) = write(force_big, false);
            assert_eq!(big, force_big);
            assert_eq!(bytes[2], if big { 43 } else { 42 });
            let mut decoder = tiff::decoder::Decoder::new(Cursor::new(&bytes)).unwrap();
            assert_eq!(decoder.dimensions().unwrap(), (4, 3));
            let tiff::decoder::DecodingResult::U8(first) = decoder.read_image().unwrap() else {
                panic!("8-bit image expected");
            };
            assert_eq!(first[..3], [0, 7, 14]);
            assert!(decoder.more_images());
            decoder.next_image().unwrap();
            assert_eq!(decoder.dimensions().unwrap(), (2, 2));
            let tiff::decoder::DecodingResult::U8(second) = decoder.read_image().unwrap() else {
                panic!("8-bit image expected");
            };
            assert_eq!(second[0], 200);
            assert!(!decoder.more_images());
        }
    }

    #[test]
    fn sub_ifds_are_listed_in_their_parent() {
        for force_big in [false, true] {
            let (bytes, _) = write(force_big, true);
            let mut decoder = tiff::decoder::Decoder::new(Cursor::new(&bytes)).unwrap();
            assert!(!decoder.more_images(), "sub-IFDs are not in the main chain");
            let offsets = decoder
                .get_tag(tiff::tags::Tag::SubIfd)
                .unwrap()
                .into_u64_vec()
                .unwrap();
            assert_eq!(offsets.len(), 1);
            decoder.seek_to_image(0).unwrap();
            assert_eq!(decoder.dimensions().unwrap(), (4, 3));
        }
    }
}
