//! A streaming, strip-based RGB TIFF writer.
//!
//! Strips are compressed and written as rows arrive, so memory use is one
//! strip. Supports Deflate with the horizontal-differencing predictor (or
//! no compression), resolution tags, `ImageDescription` and `Software`, and
//! switches to BigTIFF when the file could exceed 4 GiB.

use std::io::{Seek, SeekFrom, Write};

use flate2::Compression as DeflateLevel;
use flate2::write::ZlibEncoder;

/// TIFF compression for exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TiffCompression {
    None,
    /// Adobe Deflate (zlib) with the horizontal predictor; lossless.
    Deflate,
}

/// Writer settings.
#[derive(Debug, Clone)]
pub struct TiffOptions {
    pub compression: TiffCompression,
    /// Pixels per inch for the resolution tags.
    pub dpi: Option<f64>,
    pub description: Option<String>,
    pub software: Option<String>,
    pub rows_per_strip: u32,
    /// Force BigTIFF (otherwise chosen automatically).
    pub big: bool,
}

impl Default for TiffOptions {
    fn default() -> Self {
        Self {
            compression: TiffCompression::Deflate,
            dpi: None,
            description: None,
            software: None,
            rows_per_strip: 64,
            big: false,
        }
    }
}

const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const ASCII: u16 = 2;
const LONG8: u16 = 16;

/// Accented Latin letters and their unaccented ASCII letter.
const LATIN: &[(&str, char)] = &[
    ("ÀÁÂÃÄÅĀĂĄ", 'A'),
    ("àáâãäåāăą", 'a'),
    ("ÇĆĈĊČ", 'C'),
    ("çćĉċč", 'c'),
    ("ĎĐ", 'D'),
    ("ďđ", 'd'),
    ("ÈÉÊËĒĖĘĚ", 'E'),
    ("èéêëēėęě", 'e'),
    ("ĞĢ", 'G'),
    ("ğģ", 'g'),
    ("ÌÍÎÏĪĮİ", 'I'),
    ("ìíîïīįı", 'i'),
    ("Ķ", 'K'),
    ("ķ", 'k'),
    ("ĹĻĽŁ", 'L'),
    ("ĺļľł", 'l'),
    ("ÑŃŅŇ", 'N'),
    ("ñńņň", 'n'),
    ("ÒÓÔÕÖØŌŐ", 'O'),
    ("òóôõöøōő", 'o'),
    ("ŔŘ", 'R'),
    ("ŕř", 'r'),
    ("ŚŞŠȘ", 'S'),
    ("śşšș", 's'),
    ("ŢŤȚ", 'T'),
    ("ţťț", 't'),
    ("ÙÚÛÜŪŮŰŲ", 'U'),
    ("ùúûüūůűų", 'u'),
    ("ÝŸ", 'Y'),
    ("ýÿ", 'y'),
    ("ŹŻŽ", 'Z'),
    ("źżž", 'z'),
];

/// Text as a NUL-terminated TIFF `ASCII` value. The type is 7-bit ASCII,
/// so the typographic characters eov's descriptions use (`µ`, `×`, `·`,
/// `°`, dashes, quotes) and common accented Latin letters (from file names)
/// are transliterated, anything else becomes `?`, and NULs are dropped.
pub(crate) fn ascii_value(text: &str) -> Vec<u8> {
    let mut out = String::with_capacity(text.len() + 1);
    for c in text.chars() {
        if c.is_ascii() {
            if c != '\0' {
                out.push(c);
            }
            continue;
        }
        let replacement = match c {
            'µ' | 'μ' => "u",
            '×' => "x",
            '·' | '•' | '∙' => "-",
            '°' => " deg",
            '–' | '—' | '−' | '‐' | '‑' => "-",
            '‘' | '’' | '′' => "'",
            '“' | '”' | '″' => "\"",
            '…' => "...",
            '≈' | '∼' => "~",
            '≤' => "<=",
            '≥' => ">=",
            '±' => "+/-",
            '→' => "->",
            '\u{a0}' | '\u{2009}' | '\u{202f}' => " ",
            'γ' => "gamma",
            'Δ' => "Delta",
            'ß' => "ss",
            'Æ' => "AE",
            'æ' => "ae",
            'Œ' => "OE",
            'œ' => "oe",
            _ => match LATIN.iter().find(|(from, _)| from.contains(c)) {
                Some((_, to)) => {
                    out.push(*to);
                    continue;
                }
                None => "?",
            },
        };
        out.push_str(replacement);
    }
    let mut bytes = out.into_bytes();
    bytes.push(0);
    bytes
}

/// Streaming TIFF writer: feed RGB rows with [`write_rows`](Self::write_rows)
/// and call [`finish`](Self::finish).
pub struct TiffWriter<W: Write + Seek> {
    out: W,
    width: u32,
    height: u32,
    options: TiffOptions,
    big: bool,
    rows_per_strip: u32,
    strip_offsets: Vec<u64>,
    strip_counts: Vec<u64>,
    pending: Vec<u8>,
    pending_rows: u32,
    rows_written: u32,
}

impl<W: Write + Seek> TiffWriter<W> {
    pub fn new(mut out: W, width: u32, height: u32, options: TiffOptions) -> std::io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "TIFF images need at least one pixel",
            ));
        }
        let raw = width as u64 * height as u64 * 3;
        let big = options.big || raw > 3_900_000_000;
        if big {
            out.write_all(b"II")?;
            out.write_all(&43u16.to_le_bytes())?;
            out.write_all(&8u16.to_le_bytes())?;
            out.write_all(&0u16.to_le_bytes())?;
            out.write_all(&0u64.to_le_bytes())?;
        } else {
            out.write_all(b"II")?;
            out.write_all(&42u16.to_le_bytes())?;
            out.write_all(&0u32.to_le_bytes())?;
        }
        let rows_per_strip = options.rows_per_strip.clamp(1, height);
        Ok(Self {
            out,
            width,
            height,
            big,
            rows_per_strip,
            options,
            strip_offsets: Vec::new(),
            strip_counts: Vec::new(),
            pending: Vec::new(),
            pending_rows: 0,
            rows_written: 0,
        })
    }

    /// Appends `rows` rows of packed RGB (`width × 3` bytes each).
    pub fn write_rows(&mut self, rgb: &[u8], rows: u32) -> std::io::Result<()> {
        let stride = self.width as usize * 3;
        assert!(rgb.len() >= rows as usize * stride, "RGB data is too short");
        if self.rows_written + rows > self.height {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "more rows than the image height",
            ));
        }
        for row in rgb[..rows as usize * stride].chunks_exact(stride) {
            self.pending.extend_from_slice(row);
            self.pending_rows += 1;
            self.rows_written += 1;
            if self.pending_rows == self.rows_per_strip {
                self.flush_strip()?;
            }
        }
        Ok(())
    }

    fn flush_strip(&mut self) -> std::io::Result<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let offset = self.out.stream_position()?;
        let bytes = match self.options.compression {
            TiffCompression::None => std::mem::take(&mut self.pending),
            TiffCompression::Deflate => {
                let stride = self.width as usize * 3;
                for row in self.pending.chunks_exact_mut(stride) {
                    for i in (3..row.len()).rev() {
                        row[i] = row[i].wrapping_sub(row[i - 3]);
                    }
                }
                let mut encoder = ZlibEncoder::new(Vec::new(), DeflateLevel::new(6));
                encoder.write_all(&self.pending)?;
                self.pending.clear();
                encoder.finish()?
            }
        };
        self.out.write_all(&bytes)?;
        if bytes.len() % 2 == 1 {
            self.out.write_all(&[0])?;
        }
        self.strip_offsets.push(offset);
        self.strip_counts.push(bytes.len() as u64);
        self.pending.clear();
        self.pending_rows = 0;
        Ok(())
    }

    /// Writes the last strip and the IFD.
    pub fn finish(mut self) -> std::io::Result<W> {
        if self.rows_written != self.height {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} of {} rows written", self.rows_written, self.height),
            ));
        }
        self.flush_strip()?;
        let big = self.big;
        let inline = if big { 8 } else { 4 };

        // (tag, type, count, payload bytes)
        let mut entries: Vec<(u16, u16, u64, Vec<u8>)> = Vec::new();
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let offsets_type = if big { LONG8 } else { LONG };
        let offsets_bytes = |v: &[u64]| -> Vec<u8> {
            if big {
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            } else {
                v.iter().flat_map(|x| (*x as u32).to_le_bytes()).collect()
            }
        };
        let ascii = ascii_value;
        entries.push((256, LONG, 1, u32s(&[self.width])));
        entries.push((257, LONG, 1, u32s(&[self.height])));
        entries.push((258, SHORT, 3, u16s(&[8, 8, 8])));
        let compression = match self.options.compression {
            TiffCompression::None => 1,
            TiffCompression::Deflate => 8,
        };
        entries.push((259, SHORT, 1, u16s(&[compression])));
        entries.push((262, SHORT, 1, u16s(&[2])));
        if let Some(text) = &self.options.description {
            let bytes = ascii(text);
            entries.push((270, ASCII, bytes.len() as u64, bytes));
        }
        let strips = self.strip_offsets.len() as u64;
        entries.push((
            273,
            offsets_type,
            strips,
            offsets_bytes(&self.strip_offsets),
        ));
        entries.push((277, SHORT, 1, u16s(&[3])));
        entries.push((278, LONG, 1, u32s(&[self.rows_per_strip])));
        entries.push((279, offsets_type, strips, offsets_bytes(&self.strip_counts)));
        if let Some(dpi) = self.options.dpi.filter(|d| d.is_finite() && *d > 0.0) {
            let n = (dpi * 1000.0).round().min(u32::MAX as f64) as u32;
            let rational = u32s(&[n, 1000]);
            entries.push((282, RATIONAL, 1, rational.clone()));
            entries.push((283, RATIONAL, 1, rational));
        }
        entries.push((284, SHORT, 1, u16s(&[1])));
        if self.options.dpi.is_some() {
            entries.push((296, SHORT, 1, u16s(&[2])));
        }
        if let Some(text) = &self.options.software {
            let bytes = ascii(text);
            entries.push((305, ASCII, bytes.len() as u64, bytes));
        }
        if self.options.compression == TiffCompression::Deflate {
            entries.push((317, SHORT, 1, u16s(&[2])));
        }
        entries.sort_by_key(|e| e.0);

        // Out-of-line payloads first.
        let mut value_offsets = Vec::with_capacity(entries.len());
        for (_, _, _, payload) in &entries {
            if payload.len() > inline {
                let at = self.out.stream_position()?;
                self.out.write_all(payload)?;
                if payload.len() % 2 == 1 {
                    self.out.write_all(&[0])?;
                }
                value_offsets.push(Some(at));
            } else {
                value_offsets.push(None);
            }
        }
        let ifd_at = self.out.stream_position()?;
        if big {
            self.out.write_all(&(entries.len() as u64).to_le_bytes())?;
        } else {
            self.out.write_all(&(entries.len() as u16).to_le_bytes())?;
        }
        for ((tag, ty, count, payload), offset) in entries.iter().zip(&value_offsets) {
            self.out.write_all(&tag.to_le_bytes())?;
            self.out.write_all(&ty.to_le_bytes())?;
            if big {
                self.out.write_all(&count.to_le_bytes())?;
            } else {
                self.out.write_all(&(*count as u32).to_le_bytes())?;
            }
            let mut field = vec![0u8; inline];
            match offset {
                Some(at) if big => field.copy_from_slice(&at.to_le_bytes()),
                Some(at) => field.copy_from_slice(&(*at as u32).to_le_bytes()),
                None => field[..payload.len()].copy_from_slice(payload),
            }
            self.out.write_all(&field)?;
        }
        if big {
            self.out.write_all(&0u64.to_le_bytes())?;
        } else {
            self.out.write_all(&0u32.to_le_bytes())?;
        }
        if !big && ifd_at > u32::MAX as u64 {
            return Err(std::io::Error::other("TIFF exceeds 4 GiB; use BigTIFF"));
        }
        let header_pointer = if big { 8 } else { 4 };
        self.out.seek(SeekFrom::Start(header_pointer))?;
        if big {
            self.out.write_all(&ifd_at.to_le_bytes())?;
        } else {
            self.out.write_all(&(ifd_at as u32).to_le_bytes())?;
        }
        self.out.seek(SeekFrom::End(0))?;
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn image(w: u32, h: u32) -> Vec<u8> {
        (0..w * h * 3)
            .map(|i| ((i * 7 + i / 3) % 256) as u8)
            .collect()
    }

    fn encode(rgb: &[u8], w: u32, h: u32, options: TiffOptions) -> Vec<u8> {
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), w, h, options).unwrap();
        let stride = (w * 3) as usize;
        for chunk in rgb.chunks(stride * 7) {
            writer
                .write_rows(chunk, (chunk.len() / stride) as u32)
                .unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn decode(bytes: &[u8]) -> (u32, u32, Vec<u8>, tiff::decoder::Decoder<Cursor<&[u8]>>) {
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes)).unwrap();
        let (w, h) = decoder.dimensions().unwrap();
        let pixels = match decoder.read_image().unwrap() {
            tiff::decoder::DecodingResult::U8(v) => v,
            other => panic!("unexpected sample type {other:?}"),
        };
        (w, h, pixels, decoder)
    }

    #[test]
    fn deflate_and_uncompressed_round_trip() {
        let (w, h) = (53u32, 41u32);
        let rgb = image(w, h);
        for compression in [TiffCompression::Deflate, TiffCompression::None] {
            for big in [false, true] {
                let bytes = encode(
                    &rgb,
                    w,
                    h,
                    TiffOptions {
                        compression,
                        big,
                        rows_per_strip: 16,
                        ..TiffOptions::default()
                    },
                );
                let (dw, dh, pixels, _) = decode(&bytes);
                assert_eq!((dw, dh), (w, h));
                assert_eq!(pixels, rgb, "{compression:?} big={big}");
            }
        }
    }

    #[test]
    fn metadata_tags_are_written() {
        let rgb = image(8, 8);
        let bytes = encode(
            &rgb,
            8,
            8,
            TiffOptions {
                dpi: Some(300.0),
                description: Some("Pixel size: 0.25 × 0.25 µm\nMPP = 0.2500".into()),
                software: Some("eov 1.0.0".into()),
                ..TiffOptions::default()
            },
        );
        let (_, _, _, mut decoder) = decode(&bytes);
        use tiff::tags::Tag;
        let description = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap();
        assert_eq!(description, "Pixel size: 0.25 x 0.25 um\nMPP = 0.2500");
        let software = decoder.get_tag_ascii_string(Tag::Software).unwrap();
        assert_eq!(software, "eov 1.0.0");
        let unit = decoder.get_tag_u32(Tag::ResolutionUnit).unwrap();
        assert_eq!(unit, 2);
        let xres = decoder.get_tag_u32_vec(Tag::XResolution).unwrap();
        assert_eq!(xres, vec![300_000, 1000]);
    }

    #[test]
    fn ascii_values_are_seven_bit() {
        let value = ascii_value("0.25 × 0.25 µm · 12.5° – Øresund Příbram 日本\0x");
        assert_eq!(
            value,
            b"0.25 x 0.25 um - 12.5 deg - Oresund Pribram ??x\0".to_vec()
        );
        assert!(value.iter().all(u8::is_ascii));
        for (from, to) in LATIN {
            assert!(
                to.is_ascii() && from.chars().all(|c| !c.is_ascii()),
                "{from}"
            );
        }
    }
}
