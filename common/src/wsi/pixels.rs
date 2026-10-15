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

/// Per-axis weights of an area-averaging (box) resampler: output pixel `i`
/// covers source interval `[i * scale, (i + 1) * scale)` and every source
/// pixel contributes in proportion to its overlap.
pub(crate) struct AreaAxis {
    /// `(first source index, weights)` per output pixel; weights sum to 1.
    spans: Vec<(usize, Vec<f32>)>,
}

impl AreaAxis {
    pub(crate) fn new(src_len: usize, dst_len: usize) -> Self {
        let scale = src_len as f64 / dst_len.max(1) as f64;
        let spans = (0..dst_len)
            .map(|i| {
                let start = i as f64 * scale;
                let end = ((i + 1) as f64 * scale).min(src_len as f64);
                let first = (start.floor() as usize).min(src_len.saturating_sub(1));
                let last = (end.ceil() as usize).clamp(first + 1, src_len.max(first + 1));
                let total = (end - start).max(f64::MIN_POSITIVE);
                let weights = (first..last)
                    .map(|s| {
                        let lo = (s as f64).max(start);
                        let hi = ((s + 1) as f64).min(end);
                        ((hi - lo).max(0.0) / total) as f32
                    })
                    .collect();
                (first, weights)
            })
            .collect();
        Self { spans }
    }
}

/// Resizes straight RGBA `src` (`sw` x `sh`) to `dw` x `dh` by area
/// averaging. Intended for downscaling; upscaling degenerates to
/// nearest-neighbour.
pub(crate) fn resize_area(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), sw * sh * 4);
    if dw == sw && dh == sh {
        return src.to_vec();
    }
    let xs = AreaAxis::new(sw, dw);
    let ys = AreaAxis::new(sh, dh);

    // Horizontal pass into f32 rows.
    let mut horizontal = vec![0f32; sh * dw * 4];
    for y in 0..sh {
        let row = &src[y * sw * 4..(y + 1) * sw * 4];
        let out = &mut horizontal[y * dw * 4..(y + 1) * dw * 4];
        for (x, (first, weights)) in xs.spans.iter().enumerate() {
            let mut acc = [0f32; 4];
            for (k, w) in weights.iter().enumerate() {
                let p = &row[(first + k) * 4..(first + k) * 4 + 4];
                for c in 0..4 {
                    acc[c] += f32::from(p[c]) * w;
                }
            }
            out[x * 4..x * 4 + 4].copy_from_slice(&acc);
        }
    }

    // Vertical pass.
    let mut dst = vec![0u8; dw * dh * 4];
    for (y, (first, weights)) in ys.spans.iter().enumerate() {
        let out = &mut dst[y * dw * 4..(y + 1) * dw * 4];
        for x in 0..dw {
            let mut acc = [0f32; 4];
            for (k, w) in weights.iter().enumerate() {
                let p = &horizontal[((first + k) * dw + x) * 4..((first + k) * dw + x) * 4 + 4];
                for c in 0..4 {
                    acc[c] += p[c] * w;
                }
            }
            for c in 0..4 {
                out[x * 4 + c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// Averages `factor` x `factor` blocks of straight RGBA `src` (`sw` x `sh`,
/// both multiples of `factor`) into a `sw / factor` x `sh / factor` image.
pub(crate) fn downsample_blocks(src: &[u8], sw: usize, sh: usize, factor: usize) -> Vec<u8> {
    debug_assert!(factor > 0 && sw.is_multiple_of(factor) && sh.is_multiple_of(factor));
    let (dw, dh) = (sw / factor, sh / factor);
    let area = (factor * factor) as u32;
    let mut sums = vec![0u32; dw * 4];
    let mut dst = vec![0u8; dw * dh * 4];
    for oy in 0..dh {
        sums.fill(0);
        for y in oy * factor..(oy + 1) * factor {
            let row = &src[y * sw * 4..(y + 1) * sw * 4];
            for (x, px) in row.as_chunks::<4>().0.iter().enumerate() {
                let o = (x / factor) * 4;
                sums[o] += u32::from(px[0]);
                sums[o + 1] += u32::from(px[1]);
                sums[o + 2] += u32::from(px[2]);
                sums[o + 3] += u32::from(px[3]);
            }
        }
        let out = &mut dst[oy * dw * 4..(oy + 1) * dw * 4];
        for (o, sum) in out.iter_mut().zip(&sums) {
            *o = ((sum + area / 2) / area) as u8;
        }
    }
    dst
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
        assert_eq!(
            convert_pixel(0, AlphaMode::Composited, bg),
            [255, 255, 255, 255]
        );
        assert_eq!(
            convert_pixel(0, AlphaMode::Composited, [10, 20, 30]),
            [10, 20, 30, 255]
        );
        // Opaque pixels are untouched in every mode.
        let opaque = argb(255, 12, 34, 56);
        for mode in [
            AlphaMode::Composited,
            AlphaMode::Straight,
            AlphaMode::Premultiplied,
        ] {
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
        assert_eq!(
            convert_pixel(edge, AlphaMode::Composited, [0, 0, 0]),
            [55, 55, 55, 255]
        );
        // Straight: 55 * 255 / 58 = 241.8 -> 242.
        assert_eq!(
            convert_pixel(edge, AlphaMode::Straight, [255, 255, 255]),
            [242, 242, 242, 58]
        );
        assert_eq!(
            convert_pixel(edge, AlphaMode::Premultiplied, [255, 255, 255]),
            [55, 55, 55, 58]
        );
        assert_eq!(
            convert_pixel(0, AlphaMode::Straight, [255; 3]),
            [0, 0, 0, 0]
        );
    }

    #[test]
    fn composite_matches_exact_rounding_for_every_alpha() {
        for a in 0..=255u32 {
            for c in [0, a / 3, a] {
                let expected = (c as f64 + (255 - a) as f64 * 200.0 / 255.0)
                    .round()
                    .min(255.0) as u8;
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

    #[test]
    fn area_resize_averages_and_preserves_flat_colours() {
        // 4x1 -> 2x1 averages pairs.
        let src = [
            0, 0, 0, 255, 100, 100, 100, 255, 200, 0, 0, 255, 0, 0, 200, 255,
        ];
        assert_eq!(
            resize_area(&src, 4, 1, 2, 1),
            [50, 50, 50, 255, 100, 0, 100, 255]
        );
        // Non-integer ratios keep a flat image flat.
        let flat: Vec<u8> = std::iter::repeat_n([7u8, 77, 177, 255], 7 * 5)
            .flatten()
            .collect();
        let small = resize_area(&flat, 7, 5, 3, 2);
        assert!(
            small
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [7, 77, 177, 255])
        );
    }

    #[test]
    fn block_downsample_rounds_to_nearest() {
        let src = [
            0, 0, 0, 255, 1, 1, 1, 255, //
            1, 1, 1, 255, 1, 1, 1, 255,
        ];
        assert_eq!(downsample_blocks(&src, 2, 2, 2), [1, 1, 1, 255]);
    }
}
