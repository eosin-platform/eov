//! Stain quantification over a region: positive-pixel analysis and an
//! H-score helper for brightfield IHC (e.g. DAB with the H-DAB preset).
//!
//! Concentrations come from the same colour-deconvolution model as the
//! display pipeline (`c = S⁺ · OD`, see [`super::transform`]), so the
//! numbers match what an isolated-channel view shows. Only pixels that pass
//! the tissue [`PixelFilter`] are analysed; transparent pixels are ignored.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::basis::StainBasis;
use super::estimate::PixelFilter;
use super::linalg;
use super::od::OdLut;

/// Concentration thresholds for weak (1+), moderate (2+) and strong (3+)
/// staining, ascending.
pub type IntensityThresholds = [f32; 3];

/// Result of [`quantify_stain`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StainQuantification {
    /// Index of the quantified stain in the basis.
    pub stain: usize,
    /// Tissue pixels analysed.
    pub analysed_pixels: u64,
    /// Pixels whose concentration is at least the weak threshold.
    pub positive_pixels: u64,
    /// `positive_pixels / analysed_pixels`.
    pub positive_fraction: f32,
    /// Mean concentration over analysed pixels.
    pub mean_concentration: f32,
    /// Pixels in the weak, moderate and strong bins.
    pub bins: [u64; 3],
    /// `100 × (1·weak + 2·moderate + 3·strong) / analysed`, in `[0, 300]`.
    pub h_score: f32,
}

/// Quantifies stain `stain` of `basis` over RGBA pixels with background
/// `i0`.
pub fn quantify_stain(
    rgba: &[u8],
    basis: &StainBasis,
    i0: [f32; 3],
    stain: usize,
    thresholds: IntensityThresholds,
    filter: &PixelFilter,
) -> Option<StainQuantification> {
    if stain >= basis.len() || !(thresholds[0] <= thresholds[1] && thresholds[1] <= thresholds[2]) {
        return None;
    }
    let pinv = basis.pseudo_inverse()[stain];
    let lut = OdLut::new(i0);
    // Per-chunk partial sums combined in order: deterministic.
    let partials: Vec<(u64, [u64; 3], f64)> = rgba
        .par_chunks(4 * 16_384)
        .map(|chunk| {
            let mut analysed = 0u64;
            let mut bins = [0u64; 3];
            let mut sum = 0f64;
            for p in chunk.as_chunks::<4>().0 {
                if p[3] != 255 {
                    continue;
                }
                let rgb = [p[0], p[1], p[2]];
                let od = lut.od(rgb);
                // The usual tissue mask, except that dark pixels count:
                // strongly stained IHC is legitimately dark.
                if od.iter().any(|&v| v < filter.beta)
                    || (filter.exclude_green_dominant && rgb[1] > rgb[0] && rgb[1] > rgb[2])
                {
                    continue;
                }
                analysed += 1;
                let c = linalg::dot(pinv, od).max(0.0);
                sum += c as f64;
                for (bin, &t) in bins.iter_mut().zip(&thresholds).rev() {
                    if c >= t {
                        *bin += 1;
                        break;
                    }
                }
            }
            (analysed, bins, sum)
        })
        .collect();
    let mut analysed = 0u64;
    let mut bins = [0u64; 3];
    let mut sum = 0f64;
    for (a, b, s) in partials {
        analysed += a;
        for k in 0..3 {
            bins[k] += b[k];
        }
        sum += s;
    }
    let positive: u64 = bins.iter().sum();
    let fraction = |n: u64| {
        if analysed == 0 {
            0.0
        } else {
            n as f32 / analysed as f32
        }
    };
    Some(StainQuantification {
        stain,
        analysed_pixels: analysed,
        positive_pixels: positive,
        positive_fraction: fraction(positive),
        mean_concentration: if analysed == 0 {
            0.0
        } else {
            (sum / analysed as f64) as f32
        },
        bins,
        h_score: 100.0 * (fraction(bins[0]) + 2.0 * fraction(bins[1]) + 3.0 * fraction(bins[2])),
    })
}

#[cfg(test)]
mod tests {
    use super::super::basis::StainPreset;
    use super::*;

    fn pixel(basis: &StainBasis, conc: &[f32]) -> [u8; 4] {
        let v = basis.vectors();
        let mut out = [255u8; 4];
        for c in 0..3 {
            let od: f32 = v.iter().zip(conc).map(|(s, k)| s[c] * k).sum();
            out[c] = (255.0 * (-od).exp()).round() as u8;
        }
        out
    }

    #[test]
    fn dab_positive_fraction_and_h_score() {
        let basis = StainPreset::HematoxylinDab.basis();
        let mut rgba = Vec::new();
        // 50 negative nuclei (H only), 30 weak, 15 moderate, 5 strong DAB.
        for _ in 0..50 {
            rgba.extend_from_slice(&pixel(&basis, &[0.8, 0.0]));
        }
        for (n, dab) in [(30, 0.3), (15, 0.7), (5, 1.4)] {
            for _ in 0..n {
                rgba.extend_from_slice(&pixel(&basis, &[0.3, dab]));
            }
        }
        // Glass and a transparent pixel are not analysed.
        rgba.extend_from_slice(&[245, 245, 245, 255, 0, 0, 0, 0]);
        let q = quantify_stain(
            &rgba,
            &basis,
            [255.0; 3],
            1,
            [0.15, 0.5, 1.0],
            &PixelFilter::default(),
        )
        .unwrap();
        assert_eq!(q.analysed_pixels, 100);
        assert_eq!(q.bins, [30, 15, 5]);
        assert!((q.positive_fraction - 0.5).abs() < 1e-6);
        assert!(
            (q.h_score - (30.0 + 30.0 + 15.0)).abs() < 1e-3,
            "{}",
            q.h_score
        );
        assert!(
            quantify_stain(
                &rgba,
                &basis,
                [255.0; 3],
                2,
                [0.1, 0.2, 0.3],
                &PixelFilter::default()
            )
            .is_none()
        );
    }
}
