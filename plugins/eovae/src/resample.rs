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

/// Resamples a region onto a `tile`×`tile` model window as its rows
/// arrive, so that a large region can be read in strips with bounded
/// memory (one row and the window's sums are kept, not the region).
pub struct Resampler {
    width: usize,
    height: usize,
    side: usize,
    taps_x: Vec<Tap>,
    /// For every region row, the window rows it contributes to, and how
    /// much.
    contributions: Vec<Vec<(usize, f32)>>,
    background: [f32; 3],
    /// Window sums, planar (3 × side × side).
    sums: Vec<f32>,
    /// One region row after the horizontal pass (3 × side).
    line: Vec<f32>,
    /// One region row composited onto the background (3 × width).
    composited: Vec<f32>,
    added: usize,
}

impl Resampler {
    /// `background`: what transparent pixels become (RGB, 0–255).
    pub fn new(plan: &ReadPlan, tile: u32, background: [u8; 3]) -> Self {
        let (width, height, side) = (plan.width as usize, plan.height as usize, tile as usize);
        let taps_x = axis_taps(plan.origin_x, plan.step_x, side, width);
        let taps_y = axis_taps(plan.origin_y, plan.step_y, side, height);
        let mut contributions = vec![Vec::new(); height];
        for (b, tap) in taps_y.iter().enumerate() {
            for (k, weight) in tap.weights.iter().enumerate() {
                if *weight != 0.0 {
                    contributions[tap.first + k].push((b, *weight));
                }
            }
        }
        Self {
            width,
            height,
            side,
            taps_x,
            contributions,
            background: background.map(f32::from),
            sums: vec![0.0; 3 * side * side],
            line: vec![0.0; 3 * side],
            composited: vec![0.0; 3 * width],
            added: 0,
        }
    }

    /// Adds the region's next whole rows (straight-alpha RGBA).
    pub fn add_rows(&mut self, rgba: &[u8]) -> Result<(), String> {
        let stride = self.width * 4;
        if stride == 0
            || !rgba.len().is_multiple_of(stride)
            || self.added + rgba.len() / stride > self.height
        {
            return Err(format!(
                "the host returned {} bytes for rows of a {}×{} region",
                rgba.len(),
                self.width,
                self.height
            ));
        }
        let (width, side) = (self.width, self.side);
        for row in rgba.chunks_exact(stride) {
            let y = self.added;
            self.added += 1;
            if self.contributions[y].is_empty() {
                continue;
            }
            for (x, px) in row.as_chunks::<4>().0.iter().enumerate() {
                let alpha = f32::from(px[3]) / 255.0;
                for (channel, (value, background)) in
                    px[..3].iter().zip(self.background).enumerate()
                {
                    self.composited[channel * width + x] =
                        f32::from(*value) * alpha + background * (1.0 - alpha);
                }
            }
            for channel in 0..3 {
                let plane = &self.composited[channel * width..(channel + 1) * width];
                for (a, tap) in self.taps_x.iter().enumerate() {
                    self.line[channel * side + a] = tap
                        .weights
                        .iter()
                        .zip(&plane[tap.first..])
                        .map(|(w, v)| w * v)
                        .sum();
                }
            }
            for &(b, weight) in &self.contributions[y] {
                for channel in 0..3 {
                    let start = (channel * side + b) * side;
                    let line = &self.line[channel * side..(channel + 1) * side];
                    for (sum, value) in self.sums[start..start + side].iter_mut().zip(line) {
                        *sum += weight * value;
                    }
                }
            }
        }
        Ok(())
    }

    /// The window, once every row is in: opaque RGBA.
    pub fn finish(self) -> Result<Vec<u8>, String> {
        if self.added != self.height {
            return Err(format!("{} of {} rows were read", self.added, self.height));
        }
        let side = self.side;
        let mut window = vec![255u8; side * side * 4];
        for channel in 0..3 {
            for b in 0..side {
                for a in 0..side {
                    window[(b * side + a) * 4 + channel] =
                        self.sums[(channel * side + b) * side + a]
                            .round()
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(window)
    }
}

/// Resamples a whole region read for `plan` (straight-alpha RGBA) onto a
/// `tile`×`tile` window of opaque RGBA, composited onto `background`.
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
    let mut resampler = Resampler::new(plan, tile, background);
    resampler.add_rows(rgba)?;
    resampler.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(width: u32, height: u32, origin: (f64, f64), step: (f64, f64)) -> ReadPlan {
        ReadPlan {
            x: 0,
            y: 0,
            row: 0,
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
    fn rows_added_in_strips_give_the_same_window() {
        let rgba = image(37, 29, |x, y| {
            [(x * 7) as u8, (y * 9) as u8, ((x + y) * 3) as u8, 255]
        });
        let plan = plan(37, 29, (0.3, 0.7), (4.1, 3.3));
        let whole = resample(&rgba, &plan, 8, [255; 3]).unwrap();
        let mut strips = Resampler::new(&plan, 8, [255; 3]);
        let stride = 37 * 4;
        for chunk in rgba.chunks(stride * 5) {
            strips.add_rows(chunk).unwrap();
        }
        assert_eq!(strips.finish().unwrap(), whole);
        // Rows must all arrive, and whole.
        let mut short = Resampler::new(&plan, 8, [255; 3]);
        short.add_rows(&rgba[..stride * 3]).unwrap();
        assert!(short.finish().is_err());
        assert!(
            Resampler::new(&plan, 8, [255; 3])
                .add_rows(&rgba[..5])
                .is_err()
        );
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
