//! Pixel-format conversion and resampling for slide reads.

use crate::{Error, Result};

/// How the alpha channel of slide pixels is delivered.
///
/// OpenSlide returns **premultiplied** ARGB. Regions with no image data (gaps
/// between scanned regions in MIRAX, Ventana BIF, Philips TIFF and sparse
/// DICOM slides, and everything outside the slide) are fully transparent, and
/// edges of coarse levels are partially transparent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum AlphaMode {
    /// Opaque, straight RGBA: pixels composited over the slide's background
    /// colour (`openslide.background-color`, white when absent). Alpha is
    /// always 255. This is what [`WsiFile::read_region`](crate::WsiFile::read_region)
    /// and tile loads return, and what viewers, exports and datasets want.
    #[default]
    Composited,
    /// Straight (un-premultiplied) RGBA with OpenSlide's alpha preserved;
    /// alpha 0 marks areas without image data. Colour channels of fully
    /// transparent pixels are 0.
    Straight,
    /// Premultiplied RGBA: OpenSlide's native representation in RGBA byte
    /// order.
    Premultiplied,
}

#[inline(always)]
fn div255(value: u32) -> u32 {
    // Exact round(value / 255) for value <= 255 * 255 + 255.
    (value + 128 + ((value + 128) >> 8)) >> 8
}

/// Converts one premultiplied native-endian ARGB value to RGBA bytes.
#[inline(always)]
pub(crate) fn convert_pixel(argb: u32, mode: AlphaMode, background: [u8; 3]) -> [u8; 4] {
    let a = argb >> 24;
    let r = (argb >> 16) & 0xff;
    let g = (argb >> 8) & 0xff;
    let b = argb & 0xff;
    match mode {
        AlphaMode::Premultiplied => [r as u8, g as u8, b as u8, a as u8],
        _ if a == 255 => [r as u8, g as u8, b as u8, 255],
        AlphaMode::Composited => {
            let inv = 255 - a;
            let mix = |c: u32, bg: u8| (c + div255(inv * u32::from(bg))).min(255) as u8;
            [
                mix(r, background[0]),
                mix(g, background[1]),
                mix(b, background[2]),
                255,
            ]
        }
        AlphaMode::Straight => {
            if a == 0 {
                [0, 0, 0, 0]
            } else {
                let un = |c: u32| ((c * 255 + a / 2) / a).min(255) as u8;
                [un(r), un(g), un(b), a as u8]
            }
        }
    }
}

/// Converts a buffer that holds native-endian ARGB `u32` values (as raw
/// bytes) into RGBA bytes in place.
pub(crate) fn argb_bytes_to_rgba_in_place(buf: &mut [u8], mode: AlphaMode, background: [u8; 3]) {
    for chunk in buf.as_chunks_mut::<4>().0 {
        let argb = u32::from_ne_bytes(*chunk);
        *chunk = convert_pixel(argb, mode, background);
    }
}

/// Converts ARGB `u32` values into an RGBA byte buffer.
pub(crate) fn argb_to_rgba(src: &[u32], dst: &mut [u8], mode: AlphaMode, background: [u8; 3]) {
    debug_assert_eq!(src.len() * 4, dst.len());
    for (chunk, &argb) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
        *chunk = convert_pixel(argb, mode, background);
    }
}

/// Allocates a zeroed byte buffer, reporting allocation failure as an error
/// instead of aborting.
pub(crate) fn try_alloc(len: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory(len))?;
    buf.resize(len, 0);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argb(a: u32, r: u32, g: u32, b: u32) -> u32 {
        a << 24 | r << 16 | g << 8 | b
    }

    #[test]
    fn transparent_pixels_become_background_when_composited() {
        let bg = [255, 255, 255];
        assert_eq!(convert_pixel(0, AlphaMode::Composited, bg), [255, 255, 255, 255]);
        assert_eq!(
            convert_pixel(0, AlphaMode::Composited, [10, 20, 30]),
            [10, 20, 30, 255]
        );
        // Opaque pixels are untouched in every mode.
        let opaque = argb(255, 12, 34, 56);
        for mode in [AlphaMode::Composited, AlphaMode::Straight, AlphaMode::Premultiplied] {
            assert_eq!(convert_pixel(opaque, mode, bg), [12, 34, 56, 255]);
        }
    }

    #[test]
    fn premultiplied_edges_composite_and_unpremultiply_correctly() {
        // OpenSlide's 0x3a373737: alpha 58, premultiplied grey 55.
        let edge = 0x3a37_3737;
        // Over white: 55 + (255 - 58) * 255 / 255 = 252.
        assert_eq!(
            convert_pixel(edge, AlphaMode::Composited, [255, 255, 255]),
            [252, 252, 252, 255]
        );
        // Over black the premultiplied value is the answer.
        assert_eq!(convert_pixel(edge, AlphaMode::Composited, [0, 0, 0]), [55, 55, 55, 255]);
        // Straight: 55 * 255 / 58 = 241.8 -> 242.
        assert_eq!(
            convert_pixel(edge, AlphaMode::Straight, [255, 255, 255]),
            [242, 242, 242, 58]
        );
        assert_eq!(
            convert_pixel(edge, AlphaMode::Premultiplied, [255, 255, 255]),
            [55, 55, 55, 58]
        );
        assert_eq!(convert_pixel(0, AlphaMode::Straight, [255; 3]), [0, 0, 0, 0]);
    }

    #[test]
    fn composite_matches_exact_rounding_for_every_alpha() {
        for a in 0..=255u32 {
            for c in [0, a / 3, a] {
                let expected =
                    (c as f64 + (255 - a) as f64 * 200.0 / 255.0).round().min(255.0) as u8;
                let [r, ..] = convert_pixel(argb(a, c, 0, 0), AlphaMode::Composited, [200, 0, 0]);
                assert!(
                    (i32::from(r) - i32::from(expected)).abs() <= 1,
                    "a={a} c={c}: {r} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn in_place_conversion_handles_native_endianness() {
        let pixels = [argb(255, 1, 2, 3), 0];
        let mut bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_ne_bytes()).collect();
        argb_bytes_to_rgba_in_place(&mut bytes, AlphaMode::Composited, [9, 8, 7]);
        assert_eq!(bytes, [1, 2, 3, 255, 9, 8, 7, 255]);

        let mut out = vec![0u8; 8];
        argb_to_rgba(&pixels, &mut out, AlphaMode::Straight, [9, 8, 7]);
        assert_eq!(out, [1, 2, 3, 255, 0, 0, 0, 0]);
    }
}
