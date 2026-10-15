//! The fused stain transform: normalisation and colour deconvolution in one
//! pass, computed from the raw pixels (CLR-3, CLR-4).
//!
//! # Model
//!
//! For a pixel with optical density `OD` (see [`super::od`]) and a source
//! stain basis `S` (3×n, columns `sᵢ`):
//!
//! ```text
//! c   = S⁺ · OD                     least-squares concentrations
//! c⁺  = max(c, 0)                   physical (non-negative) part
//! mᵢ  = scaleᵢ · kᵢ · visibleᵢ      normalisation scale × user intensity
//! OD′ = OD − S·c⁺ + T·(m ∘ c⁺)      replace the stain component, keep the rest
//! ```
//!
//! `T` is the normalisation target basis (`T = S` and `scale = 1` when only
//! deconvolution adjustments are active). Everything the stain basis cannot
//! explain — the out-of-plane residual `OD − S·c` and negative concentration
//! parts (red blood cells, pigment, ink, a third stain) — passes through
//! unchanged, so with all multipliers at 1 the transform is exactly the
//! identity, and turning hematoxylin up by 1 % changes only the hematoxylin
//! component of each pixel.
//!
//! Because `OD′ − OD = W · c⁺` with the precomputed 3×n matrix
//! `W = T·diag(m) − S`, the output intensity is simply
//!
//! ```text
//! I′ = I · exp(−W · c⁺)
//! ```
//!
//! which needs the background `I₀` only to compute `c`. In *isolated* mode a
//! single stain is rendered as a greyscale absorbance image
//! `255 · exp(−mᵢ · cᵢ⁺)`.
//!
//! Pixels with alpha 0 are outside the slide and are left untouched.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::basis::{StainBasis, StainPreset};
use super::estimate::{FitChecks, StainEstimate};
use super::linalg::{self, Vec3};
use super::od::OdLut;

/// 99th-percentile reference concentrations that belong to Macenko's
/// reference H&E matrix ([`StainPreset::HematoxylinEosinMacenko`]).
pub const MACENKO_REFERENCE_MAX_C: [f32; 2] = [1.9705, 1.0308];

const CHUNK_BYTES: usize = 4 * (1 << 14);

/// The stain appearance that normalisation maps every slide onto.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizationTarget {
    /// Target stain vectors (same stain order as the source fit).
    pub basis: StainBasis,
    /// Target robust maximum concentrations, one per stain.
    pub max_concentrations: Vec<f32>,
}

impl Default for NormalizationTarget {
    fn default() -> Self {
        Self::macenko_reference()
    }
}

impl NormalizationTarget {
    /// Macenko et al.'s reference H&E appearance (`HERef`, `maxCRef`), the
    /// default target.
    pub fn macenko_reference() -> Self {
        Self {
            basis: StainPreset::HematoxylinEosinMacenko.basis(),
            max_concentrations: MACENKO_REFERENCE_MAX_C.to_vec(),
        }
    }

    /// Uses another slide's (or region's) fitted appearance as the target:
    /// "normalise to reference slide".
    pub fn from_estimate(estimate: &StainEstimate) -> Self {
        Self {
            basis: estimate.basis.clone(),
            max_concentrations: estimate.max_concentrations.clone(),
        }
    }
}

/// Per-stain user controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StainChannel {
    /// Concentration multiplier (1.0 = unchanged).
    pub intensity: f32,
    /// Hidden stains are removed from the image (multiplier 0).
    pub visible: bool,
}

impl Default for StainChannel {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            visible: true,
        }
    }
}

impl StainChannel {
    /// `true` for the neutral setting (visible, intensity 1 within 0.001).
    pub fn is_neutral(&self) -> bool {
        self.visible && (self.intensity - 1.0).abs() <= 0.001
    }

    fn multiplier(&self) -> f32 {
        if self.visible {
            self.intensity.max(0.0)
        } else {
            0.0
        }
    }
}

/// Normalisation part of a [`StainTransformSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Normalization {
    /// Basis the normalised concentrations are reconstructed with.
    pub target: StainBasis,
    /// Per-stain concentration scale (target / source maximum).
    pub scales: Vec<f32>,
}

/// Error building a stain transform.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StainTransformError {
    #[error("the normalisation target has {target} stains but the source has {source_stains}")]
    StainCountMismatch { source_stains: usize, target: usize },
    #[error("isolated stain {index} does not exist (the basis has {stains} stains)")]
    NoSuchStain { index: usize, stains: usize },
}

/// Uncompiled description of a stain transform. Plain data: compare it to
/// detect changes, then [`compile`](Self::compile) it once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StainTransformSpec {
    /// Basis used to decompose the raw pixels.
    pub source: StainBasis,
    /// Background intensity of the source pixels.
    pub i0: [f32; 3],
    /// Normalisation onto a target appearance, if enabled.
    pub normalization: Option<Normalization>,
    /// Per-stain controls (missing entries are neutral).
    pub channels: Vec<StainChannel>,
    /// Render only this stain, as greyscale absorbance.
    pub isolated: Option<usize>,
}

impl StainTransformSpec {
    /// Deconvolution adjustments only (no normalisation), `I₀ = 255`.
    pub fn deconvolution(source: StainBasis) -> Self {
        let n = source.len();
        Self {
            source,
            i0: [255.0; 3],
            normalization: None,
            channels: vec![StainChannel::default(); n],
            isolated: None,
        }
    }

    /// Normalisation of a fitted slide onto `target`. Scale factors are
    /// clamped to `checks`' range (CLR-6).
    pub fn normalization(
        estimate: &StainEstimate,
        target: &NormalizationTarget,
        checks: &FitChecks,
    ) -> Result<Self, StainTransformError> {
        let n = estimate.basis.len();
        if target.basis.len() != n || target.max_concentrations.len() != n {
            return Err(StainTransformError::StainCountMismatch {
                source_stains: n,
                target: target.basis.len(),
            });
        }
        let lo = checks.min_scale.max(1e-3);
        let hi = checks.max_scale.max(lo);
        let scales = (0..n)
            .map(|i| {
                let src = estimate.max_concentrations.get(i).copied().unwrap_or(1.0);
                (target.max_concentrations[i] / src.max(1e-3)).clamp(lo, hi)
            })
            .collect();
        Ok(Self {
            source: estimate.basis.clone(),
            i0: estimate.i0,
            normalization: Some(Normalization {
                target: target.basis.clone(),
                scales,
            }),
            channels: vec![StainChannel::default(); n],
            isolated: None,
        })
    }

    pub fn with_background(mut self, i0: [f32; 3]) -> Self {
        self.i0 = i0;
        self
    }

    /// Sets the per-stain controls (extra entries are ignored).
    pub fn with_channels(mut self, channels: &[StainChannel]) -> Self {
        let n = self.source.len();
        self.channels = (0..n)
            .map(|i| channels.get(i).copied().unwrap_or_default())
            .collect();
        self
    }

    pub fn with_isolated(mut self, isolated: Option<usize>) -> Self {
        self.isolated = isolated;
        self
    }

    /// `true` when the transform leaves every pixel unchanged.
    pub fn is_identity(&self) -> bool {
        self.normalization.is_none()
            && self.isolated.is_none()
            && self.channels.iter().all(StainChannel::is_neutral)
    }

    /// Precomputes the per-pixel matrices.
    pub fn compile(&self) -> Result<StainTransform, StainTransformError> {
        StainTransform::new(self)
    }
}

/// A compiled, ready-to-apply stain transform. Cheap to clone.
#[derive(Clone, Debug)]
pub struct StainTransform {
    enabled: bool,
    n: usize,
    pinv: [[f32; 3]; 3],
    /// Columns of `W = T·diag(m) − S`.
    w: [[f32; 3]; 3],
    isolated: Option<(usize, f32)>,
    lut: std::sync::Arc<OdLut>,
}

impl Default for StainTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl StainTransform {
    /// The identity transform.
    pub fn identity() -> Self {
        Self {
            enabled: false,
            n: 0,
            pinv: [[0.0; 3]; 3],
            w: [[0.0; 3]; 3],
            isolated: None,
            lut: std::sync::Arc::new(OdLut::new([255.0; 3])),
        }
    }

    pub fn new(spec: &StainTransformSpec) -> Result<Self, StainTransformError> {
        let n = spec.source.len();
        if let Some(index) = spec.isolated
            && index >= n
        {
            return Err(StainTransformError::NoSuchStain { index, stains: n });
        }
        if let Some(norm) = &spec.normalization
            && (norm.target.len() != n || norm.scales.len() != n)
        {
            return Err(StainTransformError::StainCountMismatch {
                source_stains: n,
                target: norm.target.len(),
            });
        }
        if spec.is_identity() {
            return Ok(Self::identity());
        }
        let source = spec.source.vectors();
        let target: Vec<Vec3> = match &spec.normalization {
            Some(norm) => norm.target.vectors(),
            None => source.clone(),
        };
        let multipliers: Vec<f32> = (0..n)
            .map(|i| {
                let scale = spec
                    .normalization
                    .as_ref()
                    .map(|norm| norm.scales[i])
                    .unwrap_or(1.0);
                scale
                    * spec
                        .channels
                        .get(i)
                        .copied()
                        .unwrap_or_default()
                        .multiplier()
            })
            .collect();
        let mut w = [[0.0f32; 3]; 3];
        for i in 0..n {
            for c in 0..3 {
                w[i][c] = multipliers[i] * target[i][c] - source[i][c];
            }
        }
        Ok(Self {
            enabled: true,
            n,
            pinv: spec.source.pseudo_inverse(),
            w,
            isolated: spec.isolated.map(|i| (i, multipliers[i])),
            lut: std::sync::Arc::new(OdLut::new(spec.i0)),
        })
    }

    pub fn is_identity(&self) -> bool {
        !self.enabled
    }

    #[inline(always)]
    fn concentrations(&self, od: [f32; 3]) -> [f32; 3] {
        let mut c = [0.0f32; 3];
        for (i, value) in c.iter_mut().enumerate().take(self.n) {
            *value = linalg::dot(self.pinv[i], od).max(0.0);
        }
        c
    }

    /// Output intensity (0–255 scale, unrounded) for an 8-bit input pixel.
    /// This is the CPU reference implementation.
    #[inline(always)]
    pub fn apply_pixel_unrounded(&self, rgb: [u8; 3]) -> [f32; 3] {
        let input = rgb.map(|v| v as f32);
        if !self.enabled {
            return input;
        }
        let c = self.concentrations(self.lut.od(rgb));
        self.finish(input, c)
    }

    #[inline(always)]
    fn finish(&self, input: [f32; 3], c: [f32; 3]) -> [f32; 3] {
        if let Some((index, multiplier)) = self.isolated {
            let grey = 255.0 * (-multiplier * c[index]).exp();
            return [grey; 3];
        }
        let mut out = [0.0f32; 3];
        for (ch, value) in out.iter_mut().enumerate() {
            let delta = c[0] * self.w[0][ch] + c[1] * self.w[1][ch] + c[2] * self.w[2][ch];
            *value = (input[ch] * (-delta).exp()).clamp(0.0, 255.0);
        }
        out
    }

    /// Applies the transform to one 8-bit pixel, rounding to nearest.
    #[inline(always)]
    pub fn apply_pixel(&self, rgb: [u8; 3]) -> [u8; 3] {
        self.apply_pixel_unrounded(rgb)
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
    }

    /// The same transform for a normalised `[0, 1]` input that is not on the
    /// 8-bit grid (e.g. a GPU-resampled value). Mirrors the WGSL shader.
    pub fn apply_unit(&self, rgb: [f32; 3]) -> [f32; 3] {
        if !self.enabled {
            return rgb;
        }
        let od = super::od::od_from_unit(rgb, self.lut.log_i0());
        let c = self.concentrations(od);
        self.finish(rgb.map(|v| v * 255.0), c).map(|v| v / 255.0)
    }

    /// Applies the transform to an RGBA buffer in place (parallel). Pixels
    /// with alpha 0 are not image content and are left untouched.
    pub fn apply_rgba(&self, buffer: &mut [u8]) {
        if !self.enabled {
            return;
        }
        buffer.par_chunks_mut(CHUNK_BYTES).for_each(|chunk| {
            for px in chunk.as_chunks_mut::<4>().0 {
                if px[3] == 0 {
                    continue;
                }
                let out = self.apply_pixel([px[0], px[1], px[2]]);
                px[..3].copy_from_slice(&out);
            }
        });
    }

    /// Packed uniform parameters for the WGSL implementation of this
    /// transform (see [`crate::processing::wgsl`]).
    pub fn gpu_params(&self) -> StainGpuParams {
        if !self.enabled {
            return StainGpuParams::default();
        }
        let li = self.lut.log_i0();
        let pad = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        let (iso_index, iso_mult) = self
            .isolated
            .map(|(i, m)| ((i + 1) as f32, m))
            .unwrap_or((0.0, 0.0));
        StainGpuParams {
            mode: [1.0, iso_index, iso_mult, self.n as f32],
            log_i0: [li[0], li[1], li[2], 0.0],
            pinv: [pad(self.pinv[0]), pad(self.pinv[1]), pad(self.pinv[2])],
            w: [pad(self.w[0]), pad(self.w[1]), pad(self.w[2])],
        }
    }
}

/// GPU uniform block for the stain transform (std140-compatible: every
/// member is a `vec4<f32>`).
///
/// * `mode` — `x`: enabled (0/1); `y`: isolated stain index + 1 (0 = blended);
///   `z`: isolated stain multiplier; `w`: number of stains.
/// * `log_i0` — `ln(I₀)` per channel (`xyz`).
/// * `pinv[i]` — row `i` of the source pseudo-inverse (`xyz`).
/// * `w[i]` — column `i` of `W = T·diag(m) − S` (`xyz`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StainGpuParams {
    pub mode: [f32; 4],
    pub log_i0: [f32; 4],
    pub pinv: [[f32; 4]; 3],
    pub w: [[f32; 4]; 3],
}

#[cfg(test)]
mod tests {
    use super::super::estimate::tests::{noise, synth_pixel};
    use super::*;

    fn he() -> StainBasis {
        StainPreset::HematoxylinEosin.basis()
    }

    /// Converts concentrations to an RGB pixel exactly (no noise).
    fn pixel(basis: &StainBasis, conc: &[f32]) -> [u8; 3] {
        let p = synth_pixel(&basis.vectors(), conc, [255.0; 3], 0);
        // synth_pixel adds a tiny deterministic noise; recompute exactly.
        let mut out = [0u8; 3];
        for c in 0..3 {
            let od: f32 = basis
                .vectors()
                .iter()
                .zip(conc)
                .map(|(v, k)| v[c] * k)
                .sum();
            out[c] = (255.0 * (-od).exp()).round() as u8;
        }
        let _ = p;
        out
    }

    #[test]
    fn neutral_settings_are_the_identity() {
        let spec = StainTransformSpec::deconvolution(he());
        assert!(spec.is_identity());
        let t = spec.compile().unwrap();
        assert!(t.is_identity());
        for rgb in [
            [200, 60, 70],
            [150, 100, 60],
            [40, 40, 45],
            [255, 255, 255],
            [0, 0, 0],
        ] {
            assert_eq!(t.apply_pixel(rgb), rgb);
        }
    }

    #[test]
    fn non_stain_colours_are_preserved_by_small_intensity_changes() {
        // CLR-4: with H × 1.01, RBCs, pigment/DAB and ink must stay put.
        let mut channels = vec![StainChannel::default(); 2];
        channels[0].intensity = 1.01;
        let t = StainTransformSpec::deconvolution(he())
            .with_channels(&channels)
            .compile()
            .unwrap();
        for rgb in [
            [200u8, 60, 70],
            [150, 100, 60],
            [40, 40, 45],
            [189, 139, 115],
        ] {
            let out = t.apply_pixel(rgb);
            for c in 0..3 {
                assert!(
                    (out[c] as i32 - rgb[c] as i32).abs() <= 2,
                    "{rgb:?} -> {out:?}"
                );
            }
        }
    }

    #[test]
    fn intensity_scales_only_the_selected_stain() {
        let basis = he();
        let original = pixel(&basis, &[0.8, 0.5]);
        let mut channels = vec![StainChannel::default(); 2];
        channels[0].intensity = 2.0;
        let t = StainTransformSpec::deconvolution(basis.clone())
            .with_channels(&channels)
            .compile()
            .unwrap();
        let expected = pixel(&basis, &[1.6, 0.5]);
        let out = t.apply_pixel(original);
        for c in 0..3 {
            assert!(
                (out[c] as i32 - expected[c] as i32).abs() <= 1,
                "{out:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn hidden_stain_is_removed() {
        let basis = he();
        let original = pixel(&basis, &[0.8, 0.5]);
        let channels = [
            StainChannel::default(),
            StainChannel {
                intensity: 1.0,
                visible: false,
            },
        ];
        let t = StainTransformSpec::deconvolution(basis.clone())
            .with_channels(&channels)
            .compile()
            .unwrap();
        let expected = pixel(&basis, &[0.8, 0.0]);
        let out = t.apply_pixel(original);
        for c in 0..3 {
            assert!(
                (out[c] as i32 - expected[c] as i32).abs() <= 1,
                "{out:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn isolated_channel_honours_intensity_and_visibility() {
        let basis = he();
        let rgb = pixel(&basis, &[1.2, 0.3]);
        let mut channels = vec![StainChannel::default(); 2];
        channels[0].intensity = 2.0;
        let t = StainTransformSpec::deconvolution(basis.clone())
            .with_channels(&channels)
            .with_isolated(Some(0))
            .compile()
            .unwrap();
        let grey = t.apply_pixel(rgb);
        let expected = (255.0 * (-2.4f32).exp()).round() as u8;
        assert!(
            (grey[0] as i32 - expected as i32).abs() <= 1,
            "{grey:?} vs {expected}"
        );
        assert_eq!(grey[0], grey[1]);
        assert_eq!(grey[1], grey[2]);
        channels[0].visible = false;
        let hidden = StainTransformSpec::deconvolution(basis)
            .with_channels(&channels)
            .with_isolated(Some(0))
            .compile()
            .unwrap();
        assert_eq!(hidden.apply_pixel(rgb), [255, 255, 255]);
    }

    #[test]
    fn eosin_pixel_reads_white_in_isolated_hematoxylin() {
        // CLR-3: a pure-eosin pixel must not leak into the H channel.
        let basis = he();
        let rgb = pixel(&basis, &[0.0, 0.9]);
        let t = StainTransformSpec::deconvolution(basis)
            .with_isolated(Some(0))
            .compile()
            .unwrap();
        assert!(t.apply_pixel(rgb)[0] >= 252, "{:?}", t.apply_pixel(rgb));
    }

    #[test]
    fn normalisation_maps_source_stains_onto_the_target() {
        let source = he();
        let target = NormalizationTarget::macenko_reference();
        let estimate = StainEstimate {
            format_version: 1,
            method: super::super::StainFitMethod::Macenko,
            basis: source.clone(),
            max_concentrations: vec![1.5, 0.8],
            i0: [255.0; 3],
            background_found: true,
            examined_pixels: 1,
            tissue_pixels: 1,
            tissue_fraction: 1.0,
            samples: 1,
            min_angle_deg: 35.0,
            condition_number: 3.0,
            fallback: None,
            source: super::super::EstimateSource::Pixels,
            slide_identity: None,
        };
        let t = StainTransformSpec::normalization(&estimate, &target, &FitChecks::default())
            .unwrap()
            .compile()
            .unwrap();
        let conc = [0.9f32, 0.4];
        let rgb = pixel(&source, &conc);
        let scaled = [conc[0] * 1.9705 / 1.5, conc[1] * 1.0308 / 0.8];
        let expected = pixel(&target.basis, &scaled);
        let out = t.apply_pixel(rgb);
        for c in 0..3 {
            assert!(
                (out[c] as i32 - expected[c] as i32).abs() <= 2,
                "{out:?} vs {expected:?}"
            );
        }
        // Glass stays glass.
        assert_eq!(t.apply_pixel([255, 255, 255]), [255, 255, 255]);
    }

    #[test]
    fn residual_preserving_normalisation_keeps_red_cells_red() {
        // A red blood cell is mostly outside the H-E plane; the classic
        // two-vector reconstruction turned it magenta, the fused transform
        // keeps its out-of-plane component.
        let est_basis = he();
        let estimate_like = Normalization {
            target: StainPreset::HematoxylinEosinMacenko.basis(),
            scales: vec![1.0, 1.0],
        };
        let spec = StainTransformSpec {
            source: est_basis,
            i0: [255.0; 3],
            normalization: Some(estimate_like),
            channels: vec![StainChannel::default(); 2],
            isolated: None,
        };
        let t = spec.compile().unwrap();
        let rbc = [200u8, 60, 70];
        let out = t.apply_pixel(rbc);
        // Still clearly red: R well above G and B.
        assert!(out[0] > 150 && out[1] < 110 && out[2] < 120, "{out:?}");
    }

    #[test]
    fn transparent_pixels_are_untouched() {
        let mut channels = vec![StainChannel::default(); 2];
        channels[0].intensity = 1.5;
        let t = StainTransformSpec::deconvolution(he())
            .with_channels(&channels)
            .compile()
            .unwrap();
        let mut buf = vec![120, 80, 160, 0, 120, 80, 160, 255];
        t.apply_rgba(&mut buf);
        assert_eq!(&buf[..4], &[120, 80, 160, 0]);
        assert_ne!(&buf[4..7], &[120, 80, 160]);
    }

    #[test]
    fn unit_path_matches_the_byte_path() {
        let mut channels = vec![StainChannel::default(); 2];
        channels[0].intensity = 1.7;
        channels[1].intensity = 0.6;
        let t = StainTransformSpec::deconvolution(he())
            .with_background([240.0, 238.0, 242.0])
            .with_channels(&channels)
            .compile()
            .unwrap();
        for i in 0..2000u64 {
            let rgb = [
                (128.0 + 127.0 * noise(i)) as u8,
                (128.0 + 127.0 * noise(i + 7)) as u8,
                (128.0 + 127.0 * noise(i + 13)) as u8,
            ];
            let a = t.apply_pixel(rgb);
            let b = t
                .apply_unit(rgb.map(|v| v as f32 / 255.0))
                .map(|v| (v * 255.0).round() as u8);
            for c in 0..3 {
                assert!(
                    (a[c] as i32 - b[c] as i32).abs() <= 1,
                    "{rgb:?}: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn three_stain_bases_are_supported() {
        let basis = StainPreset::HematoxylinEosinDab.basis();
        let rgb = pixel(&basis, &[0.5, 0.3, 0.7]);
        let channels = [
            StainChannel::default(),
            StainChannel::default(),
            StainChannel {
                intensity: 0.0,
                visible: true,
            },
        ];
        let t = StainTransformSpec::deconvolution(basis.clone())
            .with_channels(&channels)
            .compile()
            .unwrap();
        let expected = pixel(&basis, &[0.5, 0.3, 0.0]);
        let out = t.apply_pixel(rgb);
        for c in 0..3 {
            assert!(
                (out[c] as i32 - expected[c] as i32).abs() <= 1,
                "{out:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let spec = StainTransformSpec::deconvolution(he()).with_isolated(Some(2));
        assert!(matches!(
            spec.compile(),
            Err(StainTransformError::NoSuchStain { .. })
        ));
    }
}
