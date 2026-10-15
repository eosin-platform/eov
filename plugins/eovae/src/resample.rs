//! Resampling a region read from the slide onto the model's pixel grid
//! (VAE-15), per axis so that non-square slide pixels are handled.
//!
//! Pixels are composited onto the contract's background first (what the
//! model was trained to see for transparency), then filtered: an exact area
//! (box) filter when shrinking, so no detail aliases into the input, and
//! bilinear interpolation in the rare case that the slide is coarser than
//! the model. The result is opaque RGBA for [`crate::contract::prepare`].

use crate::tiling::ReadPlan;

/// Source pixels contributing to one output pixel, with their weights.
#[derive(Clone, Debug, PartialEq)]
struct Tap {
    first: usize,
    weights: Vec<f32>,
}

/// Taps along one axis: output pixel `a` covers source
/// `[origin + a·step, origin + (a + 1)·step)` of `len` source pixels.
fn axis_taps(origin: f64, step: f64, count: usize, len: usize) -> Vec<Tap> {
    let last = len.saturating_sub(1);
    let clamp = |k: f64| (k.max(0.0) as usize).min(last);
    (0..count)
        .map(|a| {
            let mut pairs: Vec<(usize, f64)> = Vec::new();
            if step >= 1.0 {
                let lo = origin + a as f64 * step;
                let hi = lo + step;
                let mut k = lo.floor();
                while k < hi {
                    let overlap = (k + 1.0).min(hi) - k.max(lo);
                    if overlap > 0.0 {
                        pairs.push((clamp(k), overlap));
                    }
                    k += 1.0;
                }
            } else {
                // Bilinear at the output pixel's centre (source pixel k's
                // centre is k + 0.5).
                let centre = origin + (a as f64 + 0.5) * step - 0.5;
                let k = centre.floor();
                let t = centre - k;
                pairs.push((clamp(k), 1.0 - t));
                pairs.push((clamp(k + 1.0), t));
            }
            let first = pairs.iter().map(|p| p.0).min().unwrap_or(0);
            let end = pairs.iter().map(|p| p.0).max().unwrap_or(0) + 1;
            let mut weights = vec![0f64; end - first];
            for (index, weight) in pairs {
                weights[index - first] += weight;
            }
            let total: f64 = weights.iter().sum();
            Tap {
                first,
                weights: weights
                    .into_iter()
                    .map(|w| if total > 0.0 { (w / total) as f32 } else { 0.0 })
                    .collect(),
            }
        })
        .collect()
}

/// Resamples `rgba` (the region read for `plan`, straight alpha) onto a
/// `tile`×`tile` window of opaque RGBA, composited onto `background`
/// (RGB, 0–255).
pub fn resample(
    rgba: &[u8],
    plan: &ReadPlan,
    tile: u32,
    background: [u8; 3],
) -> Result<Vec<u8>, String> {
    let (width, height) = (plan.width as usize, plan.height as usize);
    if width == 0 || height == 0 || rgba.len() != width * height * 4 {
        return Err(format!(
            "the host returned {} bytes for a {width}×{height} region",
            rgba.len()
        ));
    }
    let side = tile as usize;
    let pixels = width * height;
    // Composited planes, 0–255.
    let mut source = vec![0f32; 3 * pixels];
    for (index, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
        let alpha = f32::from(px[3]) / 255.0;
        for channel in 0..3 {
            source[channel * pixels + index] =
                f32::from(px[channel]) * alpha + f32::from(background[channel]) * (1.0 - alpha);
        }
    }
    let taps_x = axis_taps(plan.origin_x, plan.step_x, side, width);
    let taps_y = axis_taps(plan.origin_y, plan.step_y, side, height);
    // Horizontal pass: 3 × height × side.
    let mut rows = vec![0f32; 3 * height * side];
    for channel in 0..3 {
        let plane = &source[channel * pixels..(channel + 1) * pixels];
        let out = &mut rows[channel * height * side..(channel + 1) * height * side];
        for y in 0..height {
            let line = &plane[y * width..(y + 1) * width];
            for (a, tap) in taps_x.iter().enumerate() {
                out[y * side + a] = tap
                    .weights
                    .iter()
                    .zip(&line[tap.first..])
                    .map(|(w, v)| w * v)
                    .sum();
            }
        }
    }
    // Vertical pass into interleaved RGBA.
    let mut window = vec![255u8; side * side * 4];
    for channel in 0..3 {
        let plane = &rows[channel * height * side..(channel + 1) * height * side];
        for (b, tap) in taps_y.iter().enumerate() {
            for a in 0..side {
                let value: f32 = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * plane[(tap.first + k) * side + a])
                    .sum();
                window[(b * side + a) * 4 + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(width: u32, height: u32, origin: (f64, f64), step: (f64, f64)) -> ReadPlan {
        ReadPlan {
            x: 0,
            y: 0,
            width,
            height,
            origin_x: origin.0,
            origin_y: origin.1,
            step_x: step.0,
            step_y: step.1,
        }
    }

    fn image(width: usize, height: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&f(x, y));
            }
        }
        data
    }

    #[test]
    fn shrinking_averages_the_area_each_model_pixel_covers() {
        // A 4×4 checkerboard of 0 and 200 halves to its mean, 100.
        let rgba = image(4, 4, |x, y| {
            let v = if (x + y) % 2 == 0 { 0 } else { 200 };
            [v, v, v, 255]
        });
        let window = resample(&rgba, &plan(4, 4, (0.0, 0.0), (2.0, 2.0)), 2, [255; 3]).unwrap();
        assert!(
            window
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [100, 100, 100, 255])
        );
        // A fractional step weighs partial pixels: columns 0, 100, 200 with
        // step 1.5 give (0 + 0.5·100)/1.5 and (0.5·100 + 200)/1.5.
        let rgba = image(3, 1, |x, _| [(x * 100) as u8, 0, 0, 255]);
        let window = resample(&rgba, &plan(3, 1, (0.0, 0.0), (1.5, 1.0)), 2, [255; 3]);
        let window = window.unwrap_or_default();
        // The second row repeats the only source row (clamped).
        assert_eq!(window[0], 33);
        assert_eq!(window[4], 167);
    }

    #[test]
    fn axes_are_resampled_independently() {
        // Columns vary, rows do not: shrinking y by 3 keeps x as it is.
        let rgba = image(2, 6, |x, _| [if x == 0 { 10 } else { 250 }, 0, 0, 255]);
        let window = resample(&rgba, &plan(2, 6, (0.0, 0.0), (1.0, 3.0)), 2, [0; 3]).unwrap();
        assert_eq!(
            [window[0], window[4], window[8], window[12]],
            [10, 250, 10, 250]
        );
    }

    #[test]
    fn upsampling_interpolates_and_transparency_becomes_background() {
        let rgba = image(2, 1, |x, _| {
            if x == 0 {
                [0, 0, 0, 255]
            } else {
                [200, 200, 200, 255]
            }
        });
        // Four model pixels across two source pixels: 0, 50, 150, 200.
        let window = resample(&rgba, &plan(2, 1, (0.0, 0.0), (0.5, 0.5)), 4, [255; 3]).unwrap();
        let row: Vec<u8> = window
            .as_chunks::<4>()
            .0
            .iter()
            .take(4)
            .map(|px| px[0])
            .collect();
        assert_eq!(row, [0, 50, 150, 200]);
        // Transparent pixels are composited onto the background.
        let clear = vec![0u8; 4 * 4];
        let window = resample(&clear, &plan(2, 2, (0.0, 0.0), (1.0, 1.0)), 2, [255; 3]).unwrap();
        assert!(
            window
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 255, 255, 255])
        );
        // Wrong sizes are refused.
        assert!(resample(&clear, &plan(3, 3, (0.0, 0.0), (1.0, 1.0)), 2, [255; 3]).is_err());
    }
}
