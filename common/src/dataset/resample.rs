//! Antialiased resampling of patches to the target resolution.
//!
//! A separable Lanczos-3 filter whose support widens with the
//! downsampling factor (as PIL's `LANCZOS` resize does), so minified
//! patches are low-pass filtered instead of aliased. Deterministic.

use rayon::prelude::*;

fn lanczos3(x: f64) -> f64 {
    let x = x.abs();
    if x < 1e-9 {
        return 1.0;
    }
    if x >= 3.0 {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
}

/// Taps (first source index, weights) for every output position.
fn weights(src: u32, dst: u32) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    (0..dst)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            let lo = ((centre - support).floor().max(0.0) as usize).min(src as usize - 1);
            let hi = ((centre + support).ceil() as usize).min(src as usize);
            let mut w: Vec<f64> = (lo..hi)
                .map(|t| lanczos3((t as f64 + 0.5 - centre) / filter_scale))
                .collect();
            let sum: f64 = w.iter().sum();
            if sum.abs() > 1e-12 {
                w.iter_mut().for_each(|v| *v /= sum);
            }
            (lo, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

/// Resamples an RGBA image from `sw × sh` to `dw × dh`.
pub fn resample_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    assert_eq!(
        src.len(),
        sw as usize * sh as usize * 4,
        "source size mismatch"
    );
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let wx = weights(sw, dw);
    let wy = weights(sh, dh);
    let (sw, dw) = (sw as usize, dw as usize);
    // Horizontal pass: sh rows × dw columns, f32.
    let mut tmp = vec![0f32; sh as usize * dw * 4];
    tmp.par_chunks_mut(dw * 4).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * sw * 4..(y + 1) * sw * 4];
        for (x, (lo, w)) in wx.iter().enumerate() {
            let mut acc = [0f32; 4];
            for (k, weight) in w.iter().enumerate() {
                let p = &src_row[(lo + k) * 4..(lo + k) * 4 + 4];
                for c in 0..4 {
                    acc[c] += weight * p[c] as f32;
                }
            }
            row[x * 4..x * 4 + 4].copy_from_slice(&acc);
        }
    });
    // Vertical pass.
    let mut out = vec![0u8; dh as usize * dw * 4];
    out.par_chunks_mut(dw * 4).enumerate().for_each(|(y, row)| {
        let (lo, w) = &wy[y];
        for x in 0..dw {
            let mut acc = [0f32; 4];
            for (k, weight) in w.iter().enumerate() {
                let p = &tmp[((lo + k) * dw + x) * 4..((lo + k) * dw + x) * 4 + 4];
                for c in 0..4 {
                    acc[c] += weight * p[c];
                }
            }
            for c in 0..4 {
                row[x * 4 + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_images_stay_uniform() {
        let src: Vec<u8> = std::iter::repeat_n([120u8, 60, 200, 255], 100 * 80)
            .flatten()
            .collect();
        let out = resample_rgba(&src, 100, 80, 37, 29);
        assert!(out.chunks(4).all(|p| p == [120, 60, 200, 255]));
    }

    #[test]
    fn downsampling_averages_fine_detail() {
        // A 1-px checkerboard minified 4× must become mid-grey, not alias.
        let (w, h) = (64u32, 64u32);
        let src: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let v = if (i % w + i / w) % 2 == 0 { 0 } else { 255 };
                [v, v, v, 255]
            })
            .collect();
        let out = resample_rgba(&src, w, h, 16, 16);
        for p in out.chunks(4).skip(17).take(10) {
            assert!((p[0] as i32 - 128).abs() <= 3, "{p:?}");
        }
    }

    #[test]
    fn identity_size_is_a_copy() {
        let src: Vec<u8> = (0..16 * 4).map(|v| v as u8).collect();
        assert_eq!(resample_rgba(&src, 4, 4, 4, 4), src);
    }
}
