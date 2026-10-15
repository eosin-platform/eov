//! Optical density (OD) conversion and background (I₀) estimation.
//!
//! The Beer–Lambert model used throughout this crate defines the optical
//! density of an 8-bit channel value `v` relative to the background
//! (incident-light) intensity `I₀` as
//!
//! ```text
//! OD = ln(I₀) − ln(max(v, 1))
//! ```
//!
//! (natural logarithm; `v = 0` is treated as 1 to keep OD finite). `I₀` is
//! estimated per channel from the brightest, near-neutral pixels of a slide —
//! the glass — rather than assumed to be 255 (CLR-2): on scanners whose
//! background is 225–240, assuming 255 makes glass look like faint tissue.

use serde::{Deserialize, Serialize};

/// Per-channel `−ln` lookup table for 8-bit intensities (CLR-7).
#[derive(Clone)]
pub struct OdLut {
    table: Box<[[f32; 256]; 3]>,
    log_i0: [f32; 3],
}

impl std::fmt::Debug for OdLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OdLut")
            .field("log_i0", &self.log_i0)
            .finish_non_exhaustive()
    }
}

impl OdLut {
    /// Builds the table for background intensities `i0` (each clamped to
    /// `[1, 255]`).
    pub fn new(i0: [f32; 3]) -> Self {
        let log_i0 = i0.map(|v| (v.clamp(1.0, 255.0) as f64).ln() as f32);
        let mut table = Box::new([[0.0f32; 256]; 3]);
        for (channel, row) in table.iter_mut().enumerate() {
            let log_i0 = (i0[channel].clamp(1.0, 255.0) as f64).ln();
            for (v, entry) in row.iter_mut().enumerate() {
                *entry = (log_i0 - (v.max(1) as f64).ln()) as f32;
            }
        }
        Self { table, log_i0 }
    }

    /// OD of an 8-bit RGB triple.
    #[inline(always)]
    pub fn od(&self, rgb: [u8; 3]) -> [f32; 3] {
        [
            self.table[0][rgb[0] as usize],
            self.table[1][rgb[1] as usize],
            self.table[2][rgb[2] as usize],
        ]
    }

    /// `ln(I₀)` per channel.
    pub fn log_i0(&self) -> [f32; 3] {
        self.log_i0
    }
}

/// OD for a normalised `[0, 1]` intensity, using exactly the same definition
/// as [`OdLut`] (and the WGSL shaders): `ln(I₀) − ln(max(x·255, 1))`.
#[inline(always)]
pub fn od_from_unit(x: [f32; 3], log_i0: [f32; 3]) -> [f32; 3] {
    [
        log_i0[0] - (x[0] * 255.0).max(1.0).ln(),
        log_i0[1] - (x[1] * 255.0).max(1.0).ln(),
        log_i0[2] - (x[2] * 255.0).max(1.0).ln(),
    ]
}

/// Settings for estimating the background intensity `I₀`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BackgroundOptions {
    /// Fraction of the brightest pixels (by luminance) treated as background
    /// candidates.
    pub bright_fraction: f32,
    /// The background must be at least this bright (luminance, 0–255);
    /// otherwise no glass was visible and `I₀ = 255` is used.
    pub min_luminance: f32,
    /// The background colour's channels may differ by at most this much
    /// (glass is near-neutral); a tinted "background" means pale tissue.
    pub max_chroma: f32,
}

impl Default for BackgroundOptions {
    fn default() -> Self {
        Self {
            bright_fraction: 0.05,
            min_luminance: 200.0,
            max_chroma: 24.0,
        }
    }
}

/// Result of [`estimate_background`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BackgroundEstimate {
    /// Estimated per-channel background intensity (255 when not found).
    pub i0: [f32; 3],
    /// `false` if no plausible glass was found and `I₀ = 255` was assumed.
    pub found: bool,
}

impl Default for BackgroundEstimate {
    fn default() -> Self {
        Self {
            i0: [255.0; 3],
            found: false,
        }
    }
}

/// Integer Rec. 709 luma used for background detection.
#[inline(always)]
pub(crate) fn luminance(rgb: [u8; 3]) -> u8 {
    ((54 * rgb[0] as u32 + 183 * rgb[1] as u32 + 19 * rgb[2] as u32 + 128) >> 8) as u8
}

/// Estimates the background intensity from opaque RGB pixels.
///
/// The brightest `bright_fraction` of pixels (by luminance) are taken as
/// glass candidates and `I₀` is their per-channel median. If those pixels are
/// too dark or too strongly tinted to be glass, `I₀ = 255` is returned with
/// `found = false`. Deterministic; runs in two histogram passes.
pub fn estimate_background<'a, I>(pixels: I, options: &BackgroundOptions) -> BackgroundEstimate
where
    I: Iterator<Item = [u8; 3]> + Clone + 'a,
{
    let mut luma_hist = [0u64; 256];
    let mut total = 0u64;
    for px in pixels.clone() {
        luma_hist[luminance(px) as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return BackgroundEstimate::default();
    }
    let wanted =
        ((total as f64 * options.bright_fraction.clamp(0.001, 1.0) as f64).ceil() as u64).max(1);
    let mut threshold = 255usize;
    let mut acc = 0u64;
    for level in (0..256).rev() {
        acc += luma_hist[level];
        threshold = level;
        if acc >= wanted {
            break;
        }
    }
    let mut channel_hist = [[0u64; 256]; 3];
    let mut count = 0u64;
    for px in pixels {
        if luminance(px) as usize >= threshold {
            for c in 0..3 {
                channel_hist[c][px[c] as usize] += 1;
            }
            count += 1;
        }
    }
    let median = |hist: &[u64; 256]| -> f32 {
        let half = count.div_ceil(2);
        let mut acc = 0u64;
        for (v, &n) in hist.iter().enumerate() {
            acc += n;
            if acc >= half {
                return v as f32;
            }
        }
        255.0
    };
    let i0 = [
        median(&channel_hist[0]),
        median(&channel_hist[1]),
        median(&channel_hist[2]),
    ];
    let luma = luminance(i0.map(|v| v as u8)) as f32;
    let chroma =
        i0.iter().cloned().fold(f32::MIN, f32::max) - i0.iter().cloned().fold(f32::MAX, f32::min);
    if luma < options.min_luminance || chroma > options.max_chroma {
        return BackgroundEstimate::default();
    }
    BackgroundEstimate {
        i0: i0.map(|v| v.max(1.0)),
        found: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_unit_formula() {
        let i0 = [240.0, 236.0, 244.0];
        let lut = OdLut::new(i0);
        for v in [0u8, 1, 2, 17, 128, 236, 240, 255] {
            let a = lut.od([v, v, v]);
            let b = od_from_unit([v as f32 / 255.0; 3], lut.log_i0());
            for c in 0..3 {
                assert!(
                    (a[c] - b[c]).abs() < 1e-5,
                    "v={v} c={c}: {} vs {}",
                    a[c],
                    b[c]
                );
            }
        }
        // The background itself has zero OD.
        assert!(lut.od([240, 236, 244]).iter().all(|v| v.abs() < 1e-6));
        // Zero is clamped to one, keeping OD finite.
        assert!(lut.od([0, 0, 0]).iter().all(|v| v.is_finite()));
    }

    fn glass_and_tissue(glass: [u8; 3], glass_fraction: f32) -> Vec<[u8; 3]> {
        let n = 10_000;
        let glass_n = (n as f32 * glass_fraction) as usize;
        let mut pixels = Vec::with_capacity(n);
        for i in 0..n {
            if i < glass_n {
                let jitter = (i % 5) as u8;
                pixels.push([glass[0] - jitter, glass[1] - jitter, glass[2] - jitter]);
            } else {
                pixels.push([180 - (i % 40) as u8, 90 + (i % 30) as u8, 170]);
            }
        }
        pixels
    }

    #[test]
    fn background_is_estimated_from_glass() {
        let pixels = glass_and_tissue([232, 228, 236], 0.4);
        let est = estimate_background(pixels.iter().copied(), &BackgroundOptions::default());
        assert!(est.found);
        for (got, want) in est.i0.iter().zip([232.0, 228.0, 236.0]) {
            assert!((got - want).abs() <= 4.0, "{:?}", est.i0);
        }
    }

    #[test]
    fn tissue_only_images_fall_back_to_white() {
        let pixels = glass_and_tissue([232, 228, 236], 0.0);
        let est = estimate_background(pixels.iter().copied(), &BackgroundOptions::default());
        assert!(!est.found);
        assert_eq!(est.i0, [255.0; 3]);
    }

    #[test]
    fn pale_tinted_tissue_is_not_mistaken_for_glass() {
        let pixels: Vec<[u8; 3]> = (0..1000).map(|i| [235, 195 + (i % 3) as u8, 225]).collect();
        let est = estimate_background(pixels.iter().copied(), &BackgroundOptions::default());
        assert!(!est.found, "{est:?}");
    }
}
