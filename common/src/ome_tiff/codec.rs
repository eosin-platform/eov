//! Decoding of TIFF tiles and strips into native-endian samples.
//!
//! Supported: uncompressed, LZW, Deflate (Adobe and old), PackBits,
//! Zstandard and JPEG (8-bit, with or without shared `JPEGTables`), with
//! the horizontal and floating-point predictors. [`unsupported`] names
//! anything else when a file is opened, so a slide never fails tile by
//! tile for a reason that was known up front.

use super::tiff::{ByteOrder, Ifd, SampleType, compression_name};

/// Largest decoded tile or strip accepted (bytes). A whole image stored as
/// one strip is refused rather than decoded for every read.
pub(crate) const MAX_CHUNK_BYTES: u64 = 128 << 20;

/// A decoded tile or strip.
#[derive(Debug)]
pub(crate) struct DecodedChunk {
    /// Width of the chunk (tile width, or image width for strips).
    pub width: u32,
    /// Rows decoded.
    pub rows: u32,
    /// Samples per pixel held.
    pub samples: u16,
    pub sample_type: SampleType,
    /// Row-major, interleaved, native-endian samples (empty when
    /// `missing`).
    pub data: Vec<u8>,
    /// The writer left the chunk out (offset or byte count 0): no data.
    pub missing: bool,
}

impl DecodedChunk {
    /// A chunk the writer left out.
    pub(crate) fn missing(width: u32, rows: u32, samples: u16, sample_type: SampleType) -> Self {
        Self {
            width,
            rows,
            samples,
            sample_type,
            data: Vec::new(),
            missing: true,
        }
    }

    /// Bytes the chunk occupies in a cache.
    pub(crate) fn byte_len(&self) -> usize {
        self.data.len().max(64)
    }
}

/// Why chunks of `ifd` cannot be decoded, if they cannot.
pub(crate) fn unsupported(ifd: &Ifd) -> Option<String> {
    match ifd.compression {
        1 | 5 | 8 | 32946 | 32773 | 50000 => {}
        7 => {
            if ifd.sample_type != SampleType::U8 {
                return Some(format!(
                    "JPEG-compressed {} samples (only 8-bit JPEG is supported)",
                    ifd.sample_type.name()
                ));
            }
            if !matches!(ifd.samples_per_chunk(), 1 | 3) {
                return Some(format!(
                    "JPEG-compressed images with {} samples per pixel",
                    ifd.samples_per_chunk()
                ));
            }
            if ifd.predictor != 1 {
                return Some("JPEG combined with a TIFF predictor".to_string());
            }
        }
        other => return Some(format!("{} compression", compression_name(other))),
    }
    match ifd.predictor {
        1 | 2 => {}
        3 if ifd.sample_type.is_float() => {}
        3 => return Some("the floating-point predictor on integer samples".to_string()),
        other => return Some(format!("TIFF predictor {other}")),
    }
    if ifd.photometric == 6 && ifd.compression != 7 {
        return Some("YCbCr pixels that are not JPEG-compressed".to_string());
    }
    if matches!(ifd.photometric, 3..=5 | 8..) {
        return Some(format!("photometric interpretation {}", ifd.photometric));
    }
    let largest = ifd.decoded_chunk_bytes(0);
    if largest > MAX_CHUNK_BYTES {
        return Some(format!(
            "{} of {} x {} pixels ({} MiB decoded; at most {} MiB): convert it to a tiled OME-TIFF",
            if ifd.tiled { "tiles" } else { "strips" },
            ifd.chunk_width,
            ifd.chunk_height,
            largest >> 20,
            MAX_CHUNK_BYTES >> 20
        ));
    }
    None
}

/// Decodes the compressed bytes of chunk row `ty` of `ifd`.
///
/// The error is a description of what is wrong with the data; the caller
/// turns it into the error of the region read.
pub(crate) fn decode(
    ifd: &Ifd,
    order: ByteOrder,
    ty: u32,
    compressed: &[u8],
) -> std::result::Result<DecodedChunk, String> {
    let rows = ifd.chunk_rows(ty);
    let samples = ifd.samples_per_chunk();
    let expected = ifd.decoded_chunk_bytes(ty) as usize;
    let data = if ifd.compression == 7 {
        decode_jpeg(ifd, compressed, rows)?
    } else {
        let mut data = Vec::new();
        data.try_reserve_exact(expected)
            .map_err(|_| format!("out of memory decoding {expected} bytes"))?;
        data.resize(expected, 0);
        match ifd.compression {
            1 => {
                let available = compressed.len().min(expected);
                data[..available].copy_from_slice(&compressed[..available]);
                if available < expected {
                    return Err(format!(
                        "uncompressed data is {} of {expected} bytes",
                        compressed.len()
                    ));
                }
            }
            5 => decode_lzw(compressed, &mut data)?,
            8 | 32946 => decode_deflate(compressed, &mut data)?,
            32773 => decode_packbits(compressed, &mut data)?,
            50000 => decode_zstd(compressed, &mut data)?,
            other => return Err(format!("{} compression", compression_name(other))),
        }
        let bytes = ifd.sample_type.bytes();
        let row_bytes = ifd.chunk_width as usize * samples as usize * bytes;
        match ifd.predictor {
            3 => undo_float_predictor(&mut data, row_bytes, bytes, samples as usize),
            predictor => {
                if !order.is_native() {
                    swap_bytes(&mut data, bytes);
                }
                if predictor == 2 {
                    undo_horizontal_predictor(&mut data, row_bytes, bytes, samples as usize);
                }
            }
        }
        data
    };
    Ok(DecodedChunk {
        width: ifd.chunk_width,
        rows,
        samples,
        sample_type: ifd.sample_type,
        data,
        missing: false,
    })
}

fn decode_lzw(input: &[u8], out: &mut [u8]) -> std::result::Result<(), String> {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let mut written = 0;
    let mut input = input;
    while written < out.len() {
        let result = decoder.decode_bytes(input, &mut out[written..]);
        written += result.consumed_out;
        input = &input[result.consumed_in..];
        match result.status {
            Ok(weezl::LzwStatus::Ok) if result.consumed_in + result.consumed_out > 0 => {}
            Ok(_) => break,
            Err(error) => return Err(format!("LZW: {error}")),
        }
    }
    if written < out.len() {
        return Err(format!(
            "LZW data ends after {written} of {} bytes",
            out.len()
        ));
    }
    Ok(())
}

fn decode_deflate(input: &[u8], out: &mut [u8]) -> std::result::Result<(), String> {
    // Adobe Deflate is zlib-wrapped; a few writers emit raw Deflate.
    let inflate = |zlib: bool, out: &mut [u8]| {
        let mut inflater = flate2::Decompress::new(zlib);
        let result = inflater.decompress(input, out, flate2::FlushDecompress::Finish);
        (result, inflater.total_out() as usize)
    };
    let (result, written) = match inflate(true, out) {
        (Err(_), 0) => inflate(false, out),
        other => other,
    };
    if let Err(error) = result {
        return Err(format!("Deflate: {error}"));
    }
    if written < out.len() {
        return Err(format!(
            "Deflate data ends after {written} of {} bytes",
            out.len()
        ));
    }
    Ok(())
}

fn decode_packbits(input: &[u8], out: &mut [u8]) -> std::result::Result<(), String> {
    let mut i = 0;
    let mut o = 0;
    while o < out.len() {
        let Some(&header) = input.get(i) else {
            return Err(format!(
                "PackBits data ends after {o} of {} bytes",
                out.len()
            ));
        };
        i += 1;
        let header = header as i8;
        if header >= 0 {
            let count = (header as usize + 1).min(out.len() - o);
            let literal = input
                .get(i..i + count)
                .ok_or_else(|| "PackBits literal run past the end of the data".to_string())?;
            out[o..o + count].copy_from_slice(literal);
            i += header as usize + 1;
            o += count;
        } else if header != -128 {
            let count = ((1 - i16::from(header)) as usize).min(out.len() - o);
            let &value = input
                .get(i)
                .ok_or_else(|| "PackBits repeat past the end of the data".to_string())?;
            out[o..o + count].fill(value);
            i += 1;
            o += count;
        }
    }
    Ok(())
}

fn decode_zstd(input: &[u8], out: &mut [u8]) -> std::result::Result<(), String> {
    let mut decoder = zstd::stream::read::Decoder::with_buffer(input)
        .map_err(|error| format!("Zstandard: {error}"))?;
    std::io::Read::read_exact(&mut decoder, out).map_err(|error| format!("Zstandard: {error}"))
}

/// Joins shared `JPEGTables` (an abbreviated stream of quantisation and
/// Huffman tables) with an abbreviated tile stream into one JPEG.
fn with_tables<'a>(tables: Option<&[u8]>, data: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
    const SOI: [u8; 2] = [0xFF, 0xD8];
    const EOI: [u8; 2] = [0xFF, 0xD9];
    match tables {
        Some(tables) if tables.len() >= 4 && tables[..2] == SOI && data.starts_with(&SOI) => {
            let tables = tables.strip_suffix(&EOI).unwrap_or(tables);
            let mut joined = Vec::with_capacity(tables.len() + data.len());
            joined.extend_from_slice(tables);
            joined.extend_from_slice(&data[2..]);
            std::borrow::Cow::Owned(joined)
        }
        _ => std::borrow::Cow::Borrowed(data),
    }
}

fn decode_jpeg(ifd: &Ifd, data: &[u8], rows: u32) -> std::result::Result<Vec<u8>, String> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let stream = with_tables(ifd.jpeg_tables.as_deref(), data);
    let samples = ifd.samples_per_chunk();
    let options = DecoderOptions::default()
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX));
    // The colour transform follows the TIFF photometric interpretation,
    // not the JPEG's own markers: Photometric RGB means the tile holds
    // untransformed RGB, whatever the JPEG decoder would assume.
    let mut probe = JpegDecoder::new_with_options(ZCursor::new(&stream[..]), options);
    probe
        .decode_headers()
        .map_err(|error| format!("JPEG: {error:?}"))?;
    let input = probe
        .input_colorspace()
        .ok_or_else(|| "JPEG: no colour space".to_string())?;
    let output = match (samples, ifd.photometric, input) {
        (1, _, _) => ColorSpace::Luma,
        (3, 2, ColorSpace::RGB) => ColorSpace::RGB,
        (3, 2, ColorSpace::YCbCr) => ColorSpace::YCbCr,
        (3, 6, ColorSpace::YCbCr | ColorSpace::RGB) => ColorSpace::RGB,
        (_, _, other) => {
            return Err(format!(
                "JPEG tile in {other:?} for photometric interpretation {}",
                ifd.photometric
            ));
        }
    };
    let mut decoder = JpegDecoder::new_with_options(
        ZCursor::new(&stream[..]),
        options.jpeg_set_out_colorspace(output),
    );
    decoder
        .decode_headers()
        .map_err(|error| format!("JPEG: {error:?}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| "JPEG: no image information".to_string())?;
    let (jpeg_width, jpeg_height) = (u32::from(info.width), u32::from(info.height));
    if jpeg_width < ifd.chunk_width.min(ifd.width) || jpeg_height < rows.min(ifd.height) {
        return Err(format!(
            "JPEG tile is {jpeg_width} x {jpeg_height}, expected {} x {rows}",
            ifd.chunk_width
        ));
    }
    let size = decoder
        .output_buffer_size()
        .ok_or_else(|| "JPEG: image too large".to_string())?;
    let mut pixels = vec![0u8; size];
    decoder
        .decode_into(&mut pixels)
        .map_err(|error| format!("JPEG: {error:?}"))?;
    let components = output.num_components();
    if components != usize::from(samples) {
        return Err(format!(
            "JPEG has {components} components, expected {samples}"
        ));
    }
    // Copy into the chunk's layout (a JPEG strip or edge tile can be
    // smaller or larger than the chunk; missing pixels stay 0).
    let chunk_row = ifd.chunk_width as usize * components;
    let jpeg_row = jpeg_width as usize * components;
    let copy = chunk_row.min(jpeg_row);
    let mut out = vec![0u8; chunk_row * rows as usize];
    for (dst, src) in out
        .chunks_exact_mut(chunk_row)
        .zip(pixels.chunks_exact(jpeg_row))
    {
        dst[..copy].copy_from_slice(&src[..copy]);
    }
    Ok(out)
}

/// Converts multi-byte samples from the file's byte order to native order.
fn swap_bytes(data: &mut [u8], bytes: usize) {
    if bytes > 1 {
        for sample in data.chunks_exact_mut(bytes) {
            sample.reverse();
        }
    }
}

/// Undoes horizontal differencing (predictor 2) on native-endian samples.
fn undo_horizontal_predictor(data: &mut [u8], row_bytes: usize, bytes: usize, samples: usize) {
    if row_bytes == 0 {
        return;
    }
    for row in data.chunks_exact_mut(row_bytes) {
        match bytes {
            1 => {
                for i in samples..row.len() {
                    row[i] = row[i].wrapping_add(row[i - samples]);
                }
            }
            2 => {
                let stride = samples * 2;
                for i in (stride..row.len()).step_by(2) {
                    let value = u16::from_ne_bytes([row[i], row[i + 1]]);
                    let previous = u16::from_ne_bytes([row[i - stride], row[i - stride + 1]]);
                    row[i..i + 2].copy_from_slice(&value.wrapping_add(previous).to_ne_bytes());
                }
            }
            4 => {
                let stride = samples * 4;
                for i in (stride..row.len()).step_by(4) {
                    let value = u32::from_ne_bytes(row[i..i + 4].try_into().expect("4 bytes"));
                    let previous = u32::from_ne_bytes(
                        row[i - stride..i - stride + 4].try_into().expect("4 bytes"),
                    );
                    row[i..i + 4].copy_from_slice(&value.wrapping_add(previous).to_ne_bytes());
                }
            }
            _ => {
                let stride = samples * 8;
                for i in (stride..row.len()).step_by(8) {
                    let value = u64::from_ne_bytes(row[i..i + 8].try_into().expect("8 bytes"));
                    let previous = u64::from_ne_bytes(
                        row[i - stride..i - stride + 8].try_into().expect("8 bytes"),
                    );
                    row[i..i + 8].copy_from_slice(&value.wrapping_add(previous).to_ne_bytes());
                }
            }
        }
    }
}

/// Undoes the floating-point predictor (predictor 3, Adobe TIFF Technical
/// Note 3): per row, bytes were split into planes, most significant first,
/// and differenced. The result is native-endian.
fn undo_float_predictor(data: &mut [u8], row_bytes: usize, bytes: usize, samples: usize) {
    if row_bytes == 0 {
        return;
    }
    let mut planes = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for i in samples..row.len() {
            row[i] = row[i].wrapping_add(row[i - samples]);
        }
        planes.copy_from_slice(row);
        let values = row_bytes / bytes;
        for (value, out) in row.chunks_exact_mut(bytes).enumerate() {
            for (k, byte) in out.iter_mut().enumerate() {
                // Plane 0 holds the most significant bytes.
                let plane = if cfg!(target_endian = "big") {
                    k
                } else {
                    bytes - 1 - k
                };
                *byte = planes[plane * values + value];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_round_trips_literal_and_repeat_runs() {
        // The TIFF 6.0 specification's example.
        let packed = [
            0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7,
            0xAA,
        ];
        let expected = [
            0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ];
        let mut out = [0u8; 24];
        decode_packbits(&packed, &mut out).unwrap();
        assert_eq!(out, expected);
        // Truncated input is an error, not a panic.
        let mut out = [0u8; 24];
        assert!(decode_packbits(&packed[..9], &mut out).is_err());
        assert!(decode_packbits(&[0x05, 1, 2], &mut out).is_err());
    }

    #[test]
    fn predictors_undo_differencing() {
        // Two 16-bit samples per pixel, three pixels per row, two rows.
        let values: [u16; 12] = [10, 1000, 12, 999, 15, 1001, 7, 7, 7, 8, 9, 65535];
        let mut diff = values;
        for row in diff.as_chunks_mut::<6>().0 {
            for i in (2..6).rev() {
                row[i] = row[i].wrapping_sub(row[i - 2]);
            }
        }
        let mut bytes: Vec<u8> = diff.iter().flat_map(|v| v.to_ne_bytes()).collect();
        undo_horizontal_predictor(&mut bytes, 12, 2, 2);
        let decoded: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        assert_eq!(decoded, values);

        // Floating point: shuffle into big-endian byte planes, difference.
        let floats = [1.5f32, -2.25, 3.0e5, 0.0];
        let n = floats.len();
        let mut planes = vec![0u8; n * 4];
        for (i, f) in floats.iter().enumerate() {
            for (k, b) in f.to_be_bytes().iter().enumerate() {
                planes[k * n + i] = *b;
            }
        }
        for i in (1..planes.len()).rev() {
            planes[i] = planes[i].wrapping_sub(planes[i - 1]);
        }
        undo_float_predictor(&mut planes, n * 4, 4, 1);
        let decoded: Vec<f32> = planes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        assert_eq!(decoded, floats);
    }

    #[test]
    fn jpeg_tables_are_joined_with_abbreviated_tiles() {
        let tables = [0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x02, 0xFF, 0xD9];
        let tile = [0xFF, 0xD8, 0xFF, 0xDA, 0x01, 0xFF, 0xD9];
        assert_eq!(
            &*with_tables(Some(&tables), &tile),
            [
                0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x02, 0xFF, 0xDA, 0x01, 0xFF, 0xD9
            ]
        );
        assert_eq!(&*with_tables(None, &tile), tile);
        assert_eq!(&*with_tables(Some(&[1, 2]), &tile), tile);
    }

    #[test]
    fn corrupt_streams_are_errors() {
        let mut out = vec![0u8; 64];
        assert!(decode_deflate(&[1, 2, 3, 4, 5], &mut out).is_err());
        assert!(decode_lzw(&[0xFF; 8], &mut out).is_err());
        assert!(decode_zstd(&[0x28, 0xB5, 0x2F, 0xFD, 0, 0], &mut out).is_err());
        let ok = zstd::bulk::compress(&[3u8; 32], 3).unwrap();
        assert!(decode_zstd(&ok, &mut out).is_err(), "short data");
        let mut exact = vec![0u8; 32];
        decode_zstd(&ok, &mut exact).unwrap();
        assert_eq!(exact, [3u8; 32]);
    }
}
