//! A streaming baseline JPEG encoder (ITU-T T.81).
//!
//! Rows are encoded as soon as one MCU row (8 or 16 pixel rows) is
//! buffered, so memory use is independent of the image height. Writes a
//! JFIF header with the pixel density in dots per inch, an optional comment
//! (COM) segment, quality-scaled Annex K quantisation tables and the
//! Annex K Huffman tables. Chroma is either full resolution (4:4:4) or
//! subsampled 2×2 (4:2:0).
//!
//! [`JpegTileEncoder`] writes the same data the way JPEG-in-TIFF needs it
//! (TIFF Technical Note 2, compression 7): one tables-only stream for the
//! `JPEGTables` tag and an abbreviated stream (frame and scan only, no
//! JFIF marker) per tile.

use std::io::Write;

use rayon::prelude::*;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Annex K.1 luminance quantisation table (natural order).
const LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Annex K.2 chrominance quantisation table (natural order).
const CHROMA_Q: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// A Huffman table as `(code, length)` per symbol.
struct Huffman {
    code: [u16; 256],
    size: [u8; 256],
}

impl Huffman {
    fn new(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut code = [0u16; 256];
        let mut size = [0u8; 256];
        let mut next = 0u16;
        let mut k = 0usize;
        for (length, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                let symbol = values[k] as usize;
                code[symbol] = next;
                size[symbol] = length as u8 + 1;
                next += 1;
                k += 1;
            }
            next <<= 1;
        }
        Self { code, size }
    }
}

/// Chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// Full-resolution chroma.
    None,
    /// Chroma averaged over 2×2 pixels.
    Half,
}

/// Encoder settings.
#[derive(Debug, Clone)]
pub struct JpegOptions {
    /// 1–100 (IJG scale; 75 is the libjpeg default).
    pub quality: u8,
    pub subsampling: Subsampling,
    /// Pixel density written to the JFIF header (dots per inch).
    pub dpi: Option<u16>,
    /// Text for COM segments (split at 65 533 bytes).
    pub comment: Option<String>,
    /// ICC profile for APP2 `ICC_PROFILE` segments (split at 65 519 bytes).
    pub icc_profile: Option<Vec<u8>>,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            subsampling: Subsampling::None,
            dpi: None,
            comment: None,
            icc_profile: None,
        }
    }
}

/// Bytes of profile per APP2 segment (65 535 less the length, the
/// signature and the two sequence bytes).
const ICC_CHUNK: usize = 65_519;

fn scaled_table(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    base.map(|v| ((v as u32 * scale + 50) / 100).clamp(1, 255) as u16)
}

/// DQT with both quantisation tables (zigzag order).
fn write_dqt(
    out: &mut impl Write,
    q_luma: &[u16; 64],
    q_chroma: &[u16; 64],
) -> std::io::Result<()> {
    out.write_all(&[0xFF, 0xDB, 0x00, 132])?;
    for (id, table) in [(0u8, q_luma), (1u8, q_chroma)] {
        out.write_all(&[id])?;
        for &natural in &ZIGZAG {
            out.write_all(&[table[natural] as u8])?;
        }
    }
    Ok(())
}

/// DHT with the four Annex K Huffman tables.
fn write_dht(out: &mut impl Write) -> std::io::Result<()> {
    let tables: [(u8, &[u8; 16], &[u8]); 4] = [
        (0x00, &DC_LUMA_BITS, &DC_VALUES),
        (0x10, &AC_LUMA_BITS, &AC_LUMA_VALUES),
        (0x01, &DC_CHROMA_BITS, &DC_VALUES),
        (0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALUES),
    ];
    let length: usize = 2 + tables.iter().map(|t| 17 + t.2.len()).sum::<usize>();
    out.write_all(&[0xFF, 0xC4])?;
    out.write_all(&(length as u16).to_be_bytes())?;
    for (class_id, bits, values) in tables {
        out.write_all(&[class_id])?;
        out.write_all(bits)?;
        out.write_all(values)?;
    }
    Ok(())
}

/// SOF0 for a three-component (Y, Cb, Cr) baseline frame.
fn write_sof0(
    out: &mut impl Write,
    width: u32,
    height: u32,
    subsampling: Subsampling,
) -> std::io::Result<()> {
    let y_sampling = match subsampling {
        Subsampling::None => 0x11,
        Subsampling::Half => 0x22,
    };
    out.write_all(&[0xFF, 0xC0, 0x00, 17, 8])?;
    out.write_all(&(height as u16).to_be_bytes())?;
    out.write_all(&(width as u16).to_be_bytes())?;
    out.write_all(&[3, 1, y_sampling, 0, 2, 0x11, 1, 3, 0x11, 1])
}

/// SOS for the interleaved scan of all three components.
fn write_sos(out: &mut impl Write) -> std::io::Result<()> {
    out.write_all(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0])
}

/// Which segments a [`JpegWriter`] stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    /// A complete JFIF file: APP0, COM, DQT, SOF0, DHT, SOS.
    Interchange,
    /// SOF0 and SOS only; the decoder gets the tables from a separate
    /// tables-only stream (TIFF `JPEGTables`).
    Abbreviated,
}

struct BitWriter {
    acc: u64,
    bits: u32,
    out: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            acc: 0,
            bits: 0,
            out: Vec::with_capacity(1 << 16),
        }
    }

    #[inline]
    fn put(&mut self, value: u32, length: u8) {
        if length == 0 {
            return;
        }
        self.acc = (self.acc << length) | (value as u64 & ((1u64 << length) - 1));
        self.bits += length as u32;
        while self.bits >= 8 {
            let byte = (self.acc >> (self.bits - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.bits -= 8;
        }
    }

    fn flush_padding(&mut self) {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.put((1 << pad) - 1, pad as u8);
        }
    }
}

#[inline]
fn bit_length(v: i32) -> u8 {
    (32 - v.unsigned_abs().leading_zeros()) as u8
}

/// Float DCT basis: `COS[u][x] = C(u)/2 · cos((2x+1)uπ/16)`.
fn dct_basis() -> [[f32; 8]; 8] {
    let mut basis = [[0.0f32; 8]; 8];
    for (u, row) in basis.iter_mut().enumerate() {
        let c = if u == 0 {
            std::f64::consts::FRAC_1_SQRT_2
        } else {
            1.0
        };
        for (x, value) in row.iter_mut().enumerate() {
            *value =
                (0.5 * c * ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0).cos())
                    as f32;
        }
    }
    basis
}

/// Forward DCT of a level-shifted 8×8 block and quantisation, in zigzag
/// order.
fn fdct_quantize(block: &[f32; 64], basis: &[[f32; 8]; 8], q: &[u16; 64]) -> [i16; 64] {
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0.0;
            for x in 0..8 {
                s += basis[u][x] * block[y * 8 + x];
            }
            tmp[y * 8 + u] = s;
        }
    }
    let mut out = [0i16; 64];
    for (zz, &natural) in ZIGZAG.iter().enumerate() {
        let v = natural / 8;
        let u = natural % 8;
        let mut s = 0.0;
        for y in 0..8 {
            s += basis[v][y] * tmp[y * 8 + u];
        }
        out[zz] = (s / q[natural] as f32).round() as i16;
    }
    out
}

/// Streaming JPEG writer. Feed RGB rows in order with
/// [`write_rows`](Self::write_rows), then call [`finish`](Self::finish).
pub struct JpegWriter<W: Write> {
    out: W,
    width: u32,
    height: u32,
    rows_written: u32,
    mcu_rows: u32,
    subsampling: Subsampling,
    q_luma: [u16; 64],
    q_chroma: [u16; 64],
    dc_luma: Huffman,
    ac_luma: Huffman,
    dc_chroma: Huffman,
    ac_chroma: Huffman,
    basis: [[f32; 8]; 8],
    /// RGB rows of the MCU row being collected.
    pending: Vec<u8>,
    pending_rows: u32,
    prev_dc: [i32; 3],
    bits: BitWriter,
    /// Transform the MCUs of a row on the rayon pool (whole images), or on
    /// the calling thread (tiles, which are encoded in parallel already).
    parallel: bool,
}

impl<W: Write> JpegWriter<W> {
    /// Writes the headers and returns a writer expecting `height` rows of
    /// `width` RGB pixels.
    pub fn new(out: W, width: u32, height: u32, options: &JpegOptions) -> std::io::Result<Self> {
        Self::with_kind(out, width, height, options, StreamKind::Interchange)
    }

    fn with_kind(
        mut out: W,
        width: u32,
        height: u32,
        options: &JpegOptions,
        kind: StreamKind,
    ) -> std::io::Result<Self> {
        if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("JPEG cannot store a {width}×{height} image (limit 65535×65535)"),
            ));
        }
        let q_luma = scaled_table(&LUMA_Q, options.quality);
        let q_chroma = scaled_table(&CHROMA_Q, options.quality);
        out.write_all(&[0xFF, 0xD8])?;
        if kind == StreamKind::Interchange {
            // JFIF APP0.
            let (units, density) = match options.dpi {
                Some(dpi) if dpi > 0 => (1u8, dpi),
                _ => (0u8, 1u16),
            };
            out.write_all(&[
                0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, units,
            ])?;
            out.write_all(&density.to_be_bytes())?;
            out.write_all(&density.to_be_bytes())?;
            out.write_all(&[0x00, 0x00])?;
            // ICC profile, as the ICC specification's annex B embeds it.
            if let Some(profile) = options.icc_profile.as_deref().filter(|p| !p.is_empty()) {
                let count = profile.len().div_ceil(ICC_CHUNK);
                if count > 255 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "the ICC profile is too large for a JPEG",
                    ));
                }
                for (index, chunk) in profile.chunks(ICC_CHUNK).enumerate() {
                    out.write_all(&[0xFF, 0xE2])?;
                    out.write_all(&((chunk.len() + 16) as u16).to_be_bytes())?;
                    out.write_all(b"ICC_PROFILE\0")?;
                    out.write_all(&[(index + 1) as u8, count as u8])?;
                    out.write_all(chunk)?;
                }
            }
            if let Some(comment) = options.comment.as_deref().filter(|c| !c.is_empty()) {
                for chunk in comment.as_bytes().chunks(65_533) {
                    out.write_all(&[0xFF, 0xFE])?;
                    out.write_all(&((chunk.len() + 2) as u16).to_be_bytes())?;
                    out.write_all(chunk)?;
                }
            }
            write_dqt(&mut out, &q_luma, &q_chroma)?;
            write_sof0(&mut out, width, height, options.subsampling)?;
            write_dht(&mut out)?;
        } else {
            write_sof0(&mut out, width, height, options.subsampling)?;
        }
        write_sos(&mut out)?;
        let mcu_rows = match options.subsampling {
            Subsampling::None => 8,
            Subsampling::Half => 16,
        };
        Ok(Self {
            out,
            width,
            height,
            rows_written: 0,
            mcu_rows,
            subsampling: options.subsampling,
            q_luma,
            q_chroma,
            dc_luma: Huffman::new(&DC_LUMA_BITS, &DC_VALUES),
            ac_luma: Huffman::new(&AC_LUMA_BITS, &AC_LUMA_VALUES),
            dc_chroma: Huffman::new(&DC_CHROMA_BITS, &DC_VALUES),
            ac_chroma: Huffman::new(&AC_CHROMA_BITS, &AC_CHROMA_VALUES),
            basis: dct_basis(),
            pending: Vec::with_capacity(width as usize * 3 * mcu_rows as usize),
            pending_rows: 0,
            prev_dc: [0; 3],
            bits: BitWriter::new(),
            parallel: kind == StreamKind::Interchange,
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
            if self.pending_rows == self.mcu_rows {
                self.encode_mcu_row()?;
            }
        }
        Ok(())
    }

    /// Pads and encodes the last MCU row and writes EOI.
    pub fn finish(mut self) -> std::io::Result<W> {
        if self.rows_written != self.height {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} of {} rows written", self.rows_written, self.height),
            ));
        }
        if self.pending_rows > 0 {
            self.encode_mcu_row()?;
        }
        self.bits.flush_padding();
        self.out.write_all(&self.bits.out)?;
        self.bits.out.clear();
        self.out.write_all(&[0xFF, 0xD9])?;
        self.out.flush()?;
        Ok(self.out)
    }

    /// Converts the buffered rows (padded to the MCU height by repeating
    /// the last row) into quantised blocks in parallel, then entropy-codes
    /// them in order.
    fn encode_mcu_row(&mut self) -> std::io::Result<()> {
        let stride = self.width as usize * 3;
        while self.pending_rows < self.mcu_rows {
            let start = self.pending.len() - stride;
            self.pending.extend_from_within(start..start + stride);
            self.pending_rows += 1;
        }
        let mcu_w = self.mcu_rows as usize; // 8 or 16: MCUs are square
        let mcus = (self.width as usize).div_ceil(mcu_w);
        let width = self.width as usize;
        let pending = &self.pending;
        let pixel = |x: usize, y: usize| -> [f32; 3] {
            let x = x.min(width - 1);
            let i = y * stride + x * 3;
            let (r, g, b) = (
                pending[i] as f32,
                pending[i + 1] as f32,
                pending[i + 2] as f32,
            );
            [
                0.299 * r + 0.587 * g + 0.114 * b - 128.0,
                -0.168_736 * r - 0.331_264 * g + 0.5 * b,
                0.5 * r - 0.418_688 * g - 0.081_312 * b,
            ]
        };
        let (basis, q_luma, q_chroma, subsampling) =
            (&self.basis, &self.q_luma, &self.q_chroma, self.subsampling);
        // Blocks per MCU in scan order: Y blocks, then Cb, then Cr.
        let transform = |m: usize| {
            let x0 = m * mcu_w;
            let mut out = Vec::with_capacity(6);
            match subsampling {
                Subsampling::None => {
                    let mut planes = [[0.0f32; 64]; 3];
                    for y in 0..8 {
                        for x in 0..8 {
                            let p = pixel(x0 + x, y);
                            for c in 0..3 {
                                planes[c][y * 8 + x] = p[c];
                            }
                        }
                    }
                    out.push(fdct_quantize(&planes[0], basis, q_luma));
                    out.push(fdct_quantize(&planes[1], basis, q_chroma));
                    out.push(fdct_quantize(&planes[2], basis, q_chroma));
                }
                Subsampling::Half => {
                    for (by, bx) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
                        let mut plane = [0.0f32; 64];
                        for y in 0..8 {
                            for x in 0..8 {
                                plane[y * 8 + x] = pixel(x0 + bx + x, by + y)[0];
                            }
                        }
                        out.push(fdct_quantize(&plane, basis, q_luma));
                    }
                    let mut cb = [0.0f32; 64];
                    let mut cr = [0.0f32; 64];
                    for y in 0..8 {
                        for x in 0..8 {
                            let mut acc = [0.0f32; 3];
                            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                                let p = pixel(x0 + 2 * x + dx, 2 * y + dy);
                                acc[1] += p[1];
                                acc[2] += p[2];
                            }
                            cb[y * 8 + x] = acc[1] / 4.0;
                            cr[y * 8 + x] = acc[2] / 4.0;
                        }
                    }
                    out.push(fdct_quantize(&cb, basis, q_chroma));
                    out.push(fdct_quantize(&cr, basis, q_chroma));
                }
            }
            out
        };
        let blocks: Vec<Vec<[i16; 64]>> = if self.parallel {
            (0..mcus).into_par_iter().map(transform).collect()
        } else {
            (0..mcus).map(transform).collect()
        };
        let luma_blocks = match self.subsampling {
            Subsampling::None => 1,
            Subsampling::Half => 4,
        };
        for mcu in &blocks {
            for (i, block) in mcu.iter().enumerate() {
                let component = if i < luma_blocks {
                    0
                } else {
                    i - luma_blocks + 1
                };
                self.encode_block(block, component);
            }
        }
        self.out.write_all(&self.bits.out)?;
        self.bits.out.clear();
        self.pending.clear();
        self.pending_rows = 0;
        Ok(())
    }

    fn encode_block(&mut self, block: &[i16; 64], component: usize) {
        let (dc, ac) = if component == 0 {
            (&self.dc_luma, &self.ac_luma)
        } else {
            (&self.dc_chroma, &self.ac_chroma)
        };
        let dc_value = block[0] as i32;
        let diff = dc_value - self.prev_dc[component];
        self.prev_dc[component] = dc_value;
        let size = bit_length(diff);
        self.bits
            .put(dc.code[size as usize] as u32, dc.size[size as usize]);
        if size > 0 {
            let bits = if diff < 0 { diff - 1 } else { diff };
            self.bits.put(bits as u32, size);
        }
        let mut run = 0u32;
        for &coefficient in &block[1..] {
            if coefficient == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                self.bits.put(ac.code[0xF0] as u32, ac.size[0xF0]);
                run -= 16;
            }
            let value = coefficient as i32;
            let size = bit_length(value);
            let symbol = ((run << 4) | size as u32) as usize;
            self.bits.put(ac.code[symbol] as u32, ac.size[symbol]);
            let bits = if value < 0 { value - 1 } else { value };
            self.bits.put(bits as u32, size);
            run = 0;
        }
        if run > 0 {
            self.bits.put(ac.code[0x00] as u32, ac.size[0x00]);
        }
    }
}

/// JPEG compression for TIFF tiles ("new-style" JPEG, compression 7).
///
/// [`tables`](Self::tables) is the tables-only stream (SOI, DQT, DHT, EOI)
/// for the `JPEGTables` tag; [`encode`](Self::encode) returns one tile as
/// an abbreviated stream (SOI, SOF0, SOS, scan, EOI) that refers to those
/// tables, as libtiff writes them. The samples are YCbCr (component ids 1,
/// 2 and 3, no JFIF or Adobe marker): store the tiles with
/// `PhotometricInterpretation` YCbCr (6) and `YCbCrSubSampling`
/// [`ycbcr_subsampling`](Self::ycbcr_subsampling).
#[derive(Debug, Clone)]
pub struct JpegTileEncoder {
    options: JpegOptions,
}

impl JpegTileEncoder {
    /// `quality` 1–100 (IJG scale).
    pub fn new(quality: u8, subsampling: Subsampling) -> Self {
        Self {
            options: JpegOptions {
                quality: quality.clamp(1, 100),
                subsampling,
                dpi: None,
                comment: None,
                // Tiles carry no ICC segment; the TIFF's tag does.
                icc_profile: None,
            },
        }
    }

    pub fn quality(&self) -> u8 {
        self.options.quality
    }

    pub fn subsampling(&self) -> Subsampling {
        self.options.subsampling
    }

    /// The `YCbCrSubSampling` tag value matching the tiles.
    pub fn ycbcr_subsampling(&self) -> [u16; 2] {
        match self.options.subsampling {
            Subsampling::None => [1, 1],
            Subsampling::Half => [2, 2],
        }
    }

    /// The tables-only stream for the `JPEGTables` tag.
    pub fn tables(&self) -> Vec<u8> {
        let q_luma = scaled_table(&LUMA_Q, self.options.quality);
        let q_chroma = scaled_table(&CHROMA_Q, self.options.quality);
        let mut out = vec![0xFF, 0xD8];
        // Writing to a Vec cannot fail.
        let _ = write_dqt(&mut out, &q_luma, &q_chroma);
        let _ = write_dht(&mut out);
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    /// Encodes `width × height` packed RGB pixels (a whole tile, padding
    /// included) as an abbreviated stream. Runs on the calling thread.
    pub fn encode(&self, rgb: &[u8], width: u32, height: u32) -> std::io::Result<Vec<u8>> {
        let stride = width as usize * 3;
        if rgb.len() < stride * height as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "RGB data is too short",
            ));
        }
        let mut writer = JpegWriter::with_kind(
            Vec::with_capacity(stride * height as usize / 8),
            width,
            height,
            &self.options,
            StreamKind::Abbreviated,
        )?;
        writer.write_rows(rgb, height)?;
        writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: u32, h: u32) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 255 / w.max(1)) as u8;
                let g = (y * 255 / h.max(1)) as u8;
                let b = if (x / 7 + y / 5) % 2 == 0 { 200 } else { 60 };
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
        rgb
    }

    fn encode(rgb: &[u8], w: u32, h: u32, options: &JpegOptions, chunk_rows: u32) -> Vec<u8> {
        let mut writer = JpegWriter::new(Vec::new(), w, h, options).unwrap();
        let stride = (w * 3) as usize;
        for chunk in rgb.chunks(stride * chunk_rows as usize) {
            writer
                .write_rows(chunk, (chunk.len() / stride) as u32)
                .unwrap();
        }
        writer.finish().unwrap()
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
    }

    #[test]
    fn output_decodes_with_high_fidelity() {
        for (w, h) in [(64, 48), (37, 23), (1, 1), (200, 17)] {
            let rgb = test_image(w, h);
            for subsampling in [Subsampling::None, Subsampling::Half] {
                let options = JpegOptions {
                    quality: 95,
                    subsampling,
                    ..JpegOptions::default()
                };
                let bytes = encode(&rgb, w, h, &options, 5);
                let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
                    .unwrap_or_else(|e| panic!("{w}×{h} {subsampling:?}: {e}"))
                    .to_rgb8();
                assert_eq!(decoded.dimensions(), (w, h));
                let p = psnr(decoded.as_raw(), &rgb);
                // The test pattern has hard 5–7 px colour edges, which 2×2
                // chroma averaging cannot keep; placement is checked by
                // `subsampled_chroma_is_placed_correctly`.
                let floor = if subsampling == Subsampling::None {
                    34.0
                } else {
                    20.0
                };
                assert!(p > floor, "{w}×{h} {subsampling:?}: PSNR {p:.1}");
            }
        }
    }

    #[test]
    fn subsampled_chroma_is_placed_correctly() {
        // Smooth colour gradients survive 4:2:0 almost unchanged; a
        // misplaced chroma block would not.
        let (w, h) = (96u32, 80u32);
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[(x * 2) as u8, (y * 3) as u8, (255 - x - y) as u8]);
            }
        }
        let options = JpegOptions {
            quality: 95,
            subsampling: Subsampling::Half,
            ..JpegOptions::default()
        };
        let bytes = encode(&rgb, w, h, &options, 3);
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        let p = psnr(decoded.as_raw(), &rgb);
        assert!(p > 38.0, "PSNR {p:.1}");
    }

    #[test]
    fn quality_controls_size_and_fidelity() {
        let (w, h) = (128, 96);
        let rgb = test_image(w, h);
        let low = encode(
            &rgb,
            w,
            h,
            &JpegOptions {
                quality: 30,
                ..JpegOptions::default()
            },
            16,
        );
        let high = encode(
            &rgb,
            w,
            h,
            &JpegOptions {
                quality: 95,
                ..JpegOptions::default()
            },
            16,
        );
        assert!(low.len() < high.len());
        let decode = |b: &[u8]| image::load_from_memory(b).unwrap().to_rgb8().into_raw();
        assert!(psnr(&decode(&high), &rgb) > psnr(&decode(&low), &rgb));
    }

    #[test]
    fn density_and_comment_are_written() {
        let rgb = test_image(16, 16);
        let options = JpegOptions {
            dpi: Some(300),
            comment: Some("eov: 0.2500 µm/px".into()),
            ..JpegOptions::default()
        };
        let bytes = encode(&rgb, 16, 16, &options, 16);
        // APP0: units = 1 (dpi), density 300 × 300.
        assert_eq!(&bytes[6..11], b"JFIF\0");
        assert_eq!(bytes[13], 1);
        assert_eq!(u16::from_be_bytes([bytes[14], bytes[15]]), 300);
        assert_eq!(u16::from_be_bytes([bytes[16], bytes[17]]), 300);
        // COM follows APP0.
        assert_eq!(&bytes[20..22], &[0xFF, 0xFE]);
        let len = u16::from_be_bytes([bytes[22], bytes[23]]) as usize;
        assert_eq!(
            std::str::from_utf8(&bytes[24..22 + len]).unwrap(),
            "eov: 0.2500 µm/px"
        );
        assert!(image::load_from_memory(&bytes).is_ok());
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let (w, h) = (45, 39);
        let rgb = test_image(w, h);
        let options = JpegOptions::default();
        let a = encode(&rgb, w, h, &options, 1);
        let b = encode(&rgb, w, h, &options, 39);
        assert_eq!(a, b);
    }

    #[test]
    fn row_count_is_checked() {
        let writer = JpegWriter::new(Vec::new(), 8, 8, &JpegOptions::default()).unwrap();
        assert!(writer.finish().is_err());
        assert!(JpegWriter::new(Vec::new(), 70_000, 8, &JpegOptions::default()).is_err());
    }

    #[test]
    fn tiff_tiles_decode_with_their_tables() {
        let (w, h) = (64, 48);
        let rgb = test_image(w, h);
        for subsampling in [Subsampling::None, Subsampling::Half] {
            let encoder = JpegTileEncoder::new(95, subsampling);
            let tables = encoder.tables();
            assert_eq!(&tables[..4], &[0xFF, 0xD8, 0xFF, 0xDB]);
            assert_eq!(&tables[tables.len() - 2..], &[0xFF, 0xD9]);
            let tile = encoder.encode(&rgb, w, h).unwrap();
            // SOI then SOF0: no JFIF marker, no tables of its own.
            assert_eq!(&tile[..4], &[0xFF, 0xD8, 0xFF, 0xC0]);
            assert!(
                !tile
                    .windows(2)
                    .any(|m| m == [0xFF, 0xDB] || m == [0xFF, 0xC4])
            );
            assert_eq!(&tile[tile.len() - 2..], &[0xFF, 0xD9]);
            // Spliced as libtiff and the tiff crate do: the tables without
            // EOI, then the tile without SOI.
            let mut joined = tables[..tables.len() - 2].to_vec();
            joined.extend_from_slice(&tile[2..]);
            let decoded = image::load_from_memory_with_format(&joined, image::ImageFormat::Jpeg)
                .unwrap()
                .to_rgb8();
            assert_eq!(decoded.dimensions(), (w, h));
            let floor = if subsampling == Subsampling::None {
                34.0
            } else {
                20.0
            };
            let p = psnr(decoded.as_raw(), &rgb);
            assert!(p > floor, "{subsampling:?}: PSNR {p:.1}");
            assert_eq!(
                encoder.ycbcr_subsampling(),
                if subsampling == Subsampling::None {
                    [1, 1]
                } else {
                    [2, 2]
                }
            );
        }
        assert!(
            JpegTileEncoder::new(90, Subsampling::None)
                .encode(&rgb[..10], w, h)
                .is_err()
        );
    }
}
