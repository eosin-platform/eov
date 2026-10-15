//! Tone adjustments and sharpening for RGBA buffers.
//!
//! These are stages 3 and 6 of the canonical display pipeline described in
//! [`crate::processing`]; use [`crate::processing::ProcessingPipeline`] to run
//! the whole pipeline in the right order. All operations round to nearest
//! (CLR-5: the pre-1.0 code truncated, so the CPU output was up to one level
//! darker than the GPU's).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const CHUNK_BYTES: usize = 4 * (1 << 14);

/// Gamma, brightness and contrast.
///
/// For a normalised channel value `x ∈ [0, 1]`:
///
/// ```text
/// tone(x) = clamp((x^(1/γ) + brightness − 0.5) · contrast + 0.5, 0, 1)
/// ```
///
/// (`1/γ` is 1 when `γ ≤ 0.001`). This is exactly what the WGSL shaders
/// compute.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToneAdjustments {
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
}

impl Default for ToneAdjustments {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
        }
    }
}

impl ToneAdjustments {
    pub fn new(gamma: f32, brightness: f32, contrast: f32) -> Self {
        Self {
            gamma,
            brightness,
            contrast,
        }
    }

    /// `true` when every control is at its neutral value (within 0.001,
    /// the tolerance used by the HUD).
    pub fn is_identity(&self) -> bool {
        (self.gamma - 1.0).abs() <= 0.001
            && self.brightness.abs() <= 0.001
            && (self.contrast - 1.0).abs() <= 0.001
    }

    /// `1/γ`, or 1 for degenerate gamma.
    pub fn inv_gamma(&self) -> f32 {
        if self.gamma > 0.001 {
            1.0 / self.gamma
        } else {
            1.0
        }
    }

    /// The tone curve for one normalised channel value.
    #[inline(always)]
    pub fn apply_unit(&self, x: f32) -> f32 {
        let inv_gamma = self.inv_gamma();
        let x = x.clamp(0.0, 1.0);
        let g = if inv_gamma == 1.0 {
            x
        } else {
            x.powf(inv_gamma)
        };
        ((g + self.brightness - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0)
    }

    /// 256-entry lookup table, rounded to nearest.
    pub fn lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = quantize(self.apply_unit(i as f32 / 255.0));
        }
        lut
    }
}

/// Rounds a normalised value to the nearest 8-bit level.
#[inline(always)]
pub fn quantize(x: f32) -> u8 {
    (x * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Applies tone adjustments to an RGBA buffer, skipping pixels with alpha 0
/// (outside the slide). Alpha is unchanged.
pub fn apply_tone_rgba(buffer: &mut [u8], tone: &ToneAdjustments) {
    if tone.is_identity() {
        return;
    }
    let lut = tone.lut();
    buffer.par_chunks_mut(CHUNK_BYTES).for_each(|chunk| {
        for px in chunk.as_chunks_mut::<4>().0 {
            if px[3] == 0 {
                continue;
            }
            px[0] = lut[px[0] as usize];
            px[1] = lut[px[1] as usize];
            px[2] = lut[px[2] as usize];
        }
    });
}

/// Applies gamma, brightness and contrast to every pixel of an RGBA buffer
/// (pre-1.0 entry point; see [`ToneAdjustments`] for the curve). Now rounds
/// to nearest instead of truncating.
pub fn apply_adjustments(buffer: &mut [u8], gamma: f32, brightness: f32, contrast: f32) {
    let lut = ToneAdjustments::new(gamma, brightness, contrast).lut();
    buffer.par_chunks_mut(CHUNK_BYTES).for_each(|chunk| {
        for px in chunk.as_chunks_mut::<4>().0 {
            px[0] = lut[px[0] as usize];
            px[1] = lut[px[1] as usize];
            px[2] = lut[px[2] as usize];
        }
    });
}

/// One output channel of the 3×3 Laplacian unsharp mask:
/// `clamp(c + k·(4c − n − s − w − e))`, rounded. Shared by the CPU passes;
/// the WGSL post pass computes the same in `[0, 1]` units.
#[inline(always)]
fn sharpen_value(c: u8, n: u8, s: u8, w: u8, e: u8, amount: f32) -> u8 {
    let c = c as f32;
    let detail = 4.0 * c - (n as f32 + s as f32 + w as f32 + e as f32);
    (c + amount * detail).round().clamp(0.0, 255.0) as u8
}

/// Sharpens `height` rows of an RGBA image in screen space.
///
/// `src` holds `height + 2` rows of `width` pixels: row 0 is the row above
/// the first output row and row `height + 1` the row below the last (at the
/// image border, pass a copy of the edge row: clamp-to-edge). Columns are
/// clamped to the edge. Alpha is copied from the centre pixel.
pub fn sharpen_rows_into(src: &[u8], width: u32, height: u32, amount: f32, dst: &mut [u8]) {
    let w = width as usize;
    let h = height as usize;
    let stride = w * 4;
    assert!(
        src.len() >= (h + 2) * stride,
        "source must include the halo rows"
    );
    assert!(dst.len() >= h * stride, "destination is too small");
    if w == 0 || h == 0 {
        return;
    }
    dst[..h * stride]
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, out)| {
            let up = &src[y * stride..(y + 1) * stride];
            let row = &src[(y + 1) * stride..(y + 2) * stride];
            let down = &src[(y + 2) * stride..(y + 3) * stride];
            for x in 0..w {
                let i = x * 4;
                let l = x.saturating_sub(1) * 4;
                let r = (x + 1).min(w - 1) * 4;
                for c in 0..3 {
                    out[i + c] = sharpen_value(
                        row[i + c],
                        up[i + c],
                        down[i + c],
                        row[l + c],
                        row[r + c],
                        amount,
                    );
                }
                out[i + 3] = row[i + 3];
            }
        });
}

/// Sharpens a whole RGBA image in place (screen space, clamp-to-edge,
/// rounded). No-op for `amount ≤ 0.001`.
pub fn sharpen_rgba(buffer: &mut [u8], width: u32, height: u32, amount: f32) {
    let w = width as usize;
    let h = height as usize;
    let stride = w * 4;
    if amount <= 0.001 || w == 0 || h == 0 || buffer.len() < h * stride {
        return;
    }
    let mut padded = Vec::with_capacity((h + 2) * stride);
    padded.extend_from_slice(&buffer[..stride]);
    padded.extend_from_slice(&buffer[..h * stride]);
    padded.extend_from_slice(&buffer[(h - 1) * stride..h * stride]);
    sharpen_rows_into(&padded, width, height, amount, buffer);
}

/// Pre-1.0 name of [`sharpen_rgba`]. Edge pixels are now sharpened too
/// (clamp-to-edge), matching the GPU, and results are rounded.
pub fn apply_sharpening(buffer: &mut [u8], width: u32, height: u32, sharpness: f32) {
    sharpen_rgba(buffer, width, height, sharpness);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent f64 reference of the tone curve.
    fn tone_ref(x: f64, gamma: f64, brightness: f64, contrast: f64) -> u8 {
        let inv = if gamma > 0.001 { 1.0 / gamma } else { 1.0 };
        let v = ((x.powf(inv) + brightness - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
        (v * 255.0).round() as u8
    }

    #[test]
    fn identity_tone_is_exact() {
        let lut = ToneAdjustments::default().lut();
        for (i, v) in lut.iter().enumerate() {
            assert_eq!(*v as usize, i);
        }
    }

    #[test]
    fn tiny_brightness_rounds_instead_of_truncating() {
        // CLR-5: brightness +0.002 used to make 255 of 256 entries darker.
        let lut = ToneAdjustments::new(1.0, 0.002, 1.0).lut();
        let unchanged_or_brighter = lut
            .iter()
            .enumerate()
            .filter(|(i, v)| **v as usize >= *i)
            .count();
        assert_eq!(unchanged_or_brighter, 256);
    }

    #[test]
    fn tone_matches_an_independent_reference() {
        for (gamma, brightness, contrast) in [(1.0, 0.0, 1.0), (2.2, 0.05, 1.3), (0.5, -0.1, 0.8)] {
            let lut = ToneAdjustments::new(gamma as f32, brightness as f32, contrast as f32).lut();
            for (i, &got) in lut.iter().enumerate() {
                let expected = tone_ref(i as f64 / 255.0, gamma, brightness, contrast);
                assert!(
                    (got as i32 - expected as i32).abs() <= 1,
                    "γ={gamma} b={brightness} c={contrast} i={i}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn tone_skips_pixels_outside_the_slide() {
        let mut buf = vec![100, 100, 100, 0, 100, 100, 100, 255];
        apply_tone_rgba(&mut buf, &ToneAdjustments::new(1.0, 0.2, 1.0));
        assert_eq!(&buf[..4], &[100, 100, 100, 0]);
        assert_eq!(buf[4], (100.0f32 + 0.2 * 255.0).round() as u8);
    }

    #[test]
    fn sharpening_golden_values() {
        // 3×3 image with a bright centre; amount 0.5.
        #[rustfmt::skip]
        let grey = [
            10, 10, 10,
            10, 50, 10,
            10, 10, 10,
        ];
        let mut buf: Vec<u8> = grey.iter().flat_map(|&v| [v, v, v, 255]).collect();
        sharpen_rgba(&mut buf, 3, 3, 0.5);
        let out: Vec<u8> = buf.chunks(4).map(|p| p[0]).collect();
        // Centre: 50 + 0.5·(200 − 40) = 130.
        // Edge-centre (0,1): neighbours 10 (clamped up), 10 (down), 10 (left,
        // clamped), 50 (right): 10 + 0.5·(40 − 80) = −10 → 0.
        // Corner (0,0): neighbours 10,10,10,10 → 10.
        #[rustfmt::skip]
        let expected = [
            10, 0, 10,
            0, 130, 0,
            10, 0, 10,
        ];
        assert_eq!(out, expected);
        assert!(buf.chunks(4).all(|p| p[3] == 255));
    }

    #[test]
    fn flat_images_are_unchanged_by_sharpening() {
        let mut buf = vec![77u8; 8 * 5 * 4];
        sharpen_rgba(&mut buf, 8, 5, 1.0);
        assert!(buf.iter().all(|&v| v == 77));
    }

    #[test]
    fn band_sharpening_matches_full_image() {
        let (w, h) = (17u32, 23u32);
        let img: Vec<u8> = (0..w * h * 4)
            .map(|i| ((i * 37 + (i / 4) * 11) % 251) as u8)
            .collect();
        let mut full = img.clone();
        sharpen_rgba(&mut full, w, h, 0.7);
        let stride = (w * 4) as usize;
        // Sharpen rows 5..12 using a halo from the original image.
        let (y0, y1) = (5usize, 12usize);
        let src = &img[(y0 - 1) * stride..(y1 + 1) * stride];
        let mut band = vec![0u8; (y1 - y0) * stride];
        sharpen_rows_into(src, w, (y1 - y0) as u32, 0.7, &mut band);
        assert_eq!(&band[..], &full[y0 * stride..y1 * stride]);
    }
}
