//! Pre-1.0 stain API, kept source-compatible for existing callers.
//!
//! These functions are thin wrappers over the new model and inherit its
//! fixes (per-channel tissue mask, fit sanity checks, residual-preserving
//! reconstruction, rounding). They still assume a background of 255 and the
//! default normalisation target, and they still split normalisation and
//! deconvolution into two passes; new code should build a
//! [`ProcessingPipeline`](crate::processing::ProcessingPipeline) from a
//! slide-level [`StainEstimate`] instead.

use super::basis::{Stain, StainBasis, StainPreset};
use super::estimate::{
    FitChecks, StainEstimate, StainEstimateOptions, StainFitMethod, estimate_stain_from_rgba,
};
use super::linalg;
use super::od::OdLut;
use super::transform::{
    Normalization, NormalizationTarget, StainChannel, StainGpuParams, StainTransform,
    StainTransformSpec,
};
use crate::StainNormalization;
use rayon::prelude::*;

/// Legacy marker for "artificial dark fill": the CPU viewport fills the
/// pane with (30,30,30) before blitting tiles, and the legacy passes must not
/// recolour it. Pixels whose OD sum (relative to 255) is at least this are
/// skipped in blended mode, as in eov 0.4.
const LEGACY_DARK_FILL_OD_SUM: f32 = 6.0;

// ─── Stain normalisation parameters ──────────────────────────────────────────

/// Normalisation parameters in the pre-1.0 packed layout.
///
/// `inv_stain_rN = [P[N][0], P[N][1], P[N][2], scale_N]` where `P` is the
/// pseudo-inverse of the *source* stain matrix and `scale_N` the
/// concentration scale onto the default target
/// ([`NormalizationTarget::macenko_reference`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StainNormParams {
    pub enabled: bool,
    /// `[inv[0][0], inv[0][1], inv[0][2], scale_h]`
    pub inv_stain_r0: [f32; 4],
    /// `[inv[1][0], inv[1][1], inv[1][2], scale_e]`
    pub inv_stain_r1: [f32; 4],
}

impl Default for StainNormParams {
    fn default() -> Self {
        Self {
            enabled: false,
            inv_stain_r0: [0.0; 4],
            inv_stain_r1: [0.0; 4],
        }
    }
}

impl StainNormParams {
    /// Packs a (slide-level) two-stain estimate into the legacy layout, for
    /// callers that still use the legacy render path. Scales are clamped with
    /// the default [`FitChecks`]. The background is not representable in this
    /// layout; use [`StainTransformSpec::normalization`] to keep it.
    pub fn from_estimate(estimate: &StainEstimate) -> Self {
        if estimate.basis.len() != 2 {
            return Self::default();
        }
        let target = NormalizationTarget::macenko_reference();
        let Ok(spec) = StainTransformSpec::normalization(estimate, &target, &FitChecks::default())
        else {
            return Self::default();
        };
        let scales = spec
            .normalization
            .as_ref()
            .map(|n| n.scales.clone())
            .unwrap_or_else(|| vec![1.0, 1.0]);
        let p = estimate.basis.pseudo_inverse();
        Self {
            enabled: true,
            inv_stain_r0: [p[0][0], p[0][1], p[0][2], scales[0]],
            inv_stain_r1: [p[1][0], p[1][1], p[1][2], scales[1]],
        }
    }

    /// Recovers the source stain basis from the packed pseudo-inverse.
    fn source_basis(&self) -> Option<StainBasis> {
        let rows = [
            [
                self.inv_stain_r0[0],
                self.inv_stain_r0[1],
                self.inv_stain_r0[2],
            ],
            [
                self.inv_stain_r1[0],
                self.inv_stain_r1[1],
                self.inv_stain_r1[2],
            ],
        ];
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let cols = linalg::matrix_from_pseudo_inverse(&rows)?;
        StainBasis::new(vec![
            Stain::new("Hematoxylin", cols[0]),
            Stain::new("Eosin", cols[1]),
        ])
        .ok()
    }

    fn scales(&self) -> Vec<f32> {
        let clamp = |v: f32| {
            if v.is_finite() && v > 0.0 { v } else { 1.0 }
        };
        vec![clamp(self.inv_stain_r0[3]), clamp(self.inv_stain_r1[3])]
    }

    /// The normalisation as a transform spec (background 255, default target).
    pub(crate) fn spec(&self) -> Option<StainTransformSpec> {
        if !self.enabled {
            return None;
        }
        let source = self.source_basis()?;
        Some(StainTransformSpec {
            source,
            i0: [255.0; 3],
            normalization: Some(Normalization {
                target: NormalizationTarget::macenko_reference().basis,
                scales: self.scales(),
            }),
            channels: vec![StainChannel::default(); 2],
            isolated: None,
        })
    }
}

/// Options matching the pre-1.0 behaviour of fitting the visible tiles:
/// background fixed at 255 (the legacy GPU path cannot carry `I₀`), and a
/// stricter tissue-fraction requirement because a single view is often
/// mostly glass.
fn legacy_view_options() -> StainEstimateOptions {
    StainEstimateOptions {
        max_samples: 10_000,
        background: Some([255.0; 3]),
        min_tissue_fraction: 0.05,
        min_tissue_pixels: 500,
        ..StainEstimateOptions::default()
    }
}

/// Fits stain vectors to the given RGBA tile data (the tiles currently on
/// screen) and returns legacy normalisation parameters.
///
/// Returns disabled parameters when the tiles contain too little tissue.
#[deprecated(
    since = "1.0.0",
    note = "re-fitting from the visible tiles makes colours change as the user pans (CLR-1); \
            use `stain::estimate_slide_stain` once per slide and build a \
            `processing::ProcessingPipeline` (or `StainNormParams::from_estimate`)"
)]
pub fn compute_cpu_stain_params(
    method: StainNormalization,
    tile_data: &[&[u8]],
) -> StainNormParams {
    let Some(method) = StainFitMethod::from_normalization(method) else {
        return StainNormParams::default();
    };
    let estimate = estimate_stain_from_rgba(tile_data, method, &legacy_view_options());
    if matches!(
        estimate.fallback,
        Some(super::estimate::FitFallback::InsufficientTissue { .. })
    ) {
        return StainNormParams::default();
    }
    StainNormParams::from_estimate(&estimate)
}

/// Applies legacy normalisation parameters to an RGBA buffer.
#[deprecated(
    since = "1.0.0",
    note = "use `processing::ProcessingPipeline::apply_rgba` (or `StainTransform::apply_rgba`)"
)]
pub fn apply_stain_params_to_buffer(buffer: &mut [u8], params: &StainNormParams) {
    #[allow(deprecated)]
    apply_normalization_to_buffer(buffer, params);
}

/// Applies legacy normalisation parameters to an RGBA buffer.
#[deprecated(
    since = "1.0.0",
    note = "use `processing::ProcessingPipeline::apply_rgba` (or `StainTransform::apply_rgba`)"
)]
pub fn apply_normalization_to_buffer(buffer: &mut [u8], params: &StainNormParams) {
    let Some(spec) = params.spec() else {
        return;
    };
    if let Ok(transform) = spec.compile() {
        apply_skipping_dark_fill(&transform, buffer, false);
    }
}

/// Fits the given tiles and normalises `buffer` in one call.
#[deprecated(
    since = "1.0.0",
    note = "use `stain::estimate_slide_stain` and `processing::ProcessingPipeline`"
)]
pub fn normalize_buffer(buffer: &mut [u8], method: StainNormalization, tile_data: &[&[u8]]) {
    #[allow(deprecated)]
    let params = compute_cpu_stain_params(method, tile_data);
    #[allow(deprecated)]
    apply_normalization_to_buffer(buffer, &params);
}

fn apply_skipping_dark_fill(transform: &StainTransform, buffer: &mut [u8], isolated: bool) {
    if transform.is_identity() {
        return;
    }
    let white = OdLut::new([255.0; 3]);
    buffer.par_chunks_mut(4 * 16_384).for_each(|chunk| {
        for px in chunk.as_chunks_mut::<4>().0 {
            if px[3] == 0 {
                continue;
            }
            let rgb = [px[0], px[1], px[2]];
            if !isolated {
                let od = white.od(rgb);
                if od[0] + od[1] + od[2] >= LEGACY_DARK_FILL_OD_SUM {
                    continue;
                }
            }
            px[..3].copy_from_slice(&transform.apply_pixel(rgb));
        }
    });
}

// ─── Colour deconvolution parameters ─────────────────────────────────────────

/// Default H&E deconvolution vectors (Ruifrok & Johnston; ImageJ "H&E").
/// Row 0 = hematoxylin, row 1 = eosin.
pub const DEFAULT_DECONV_MATRIX: [[f32; 3]; 2] = [
    [0.644211, 0.716556, 0.266844],
    [0.092789, 0.954111, 0.283111],
];

/// Deconvolution parameters in the pre-1.0 packed layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorDeconvParams {
    /// `[P[0][0], P[0][1], P[0][2], h_intensity]` (pseudo-inverse row of the
    /// decomposition basis).
    pub inv_row0: [f32; 4],
    /// `[P[1][0], P[1][1], P[1][2], e_intensity]`
    pub inv_row1: [f32; 4],
    /// Hematoxylin OD vector of the decomposition basis + intensity.
    pub stain_h: [f32; 4],
    /// Eosin OD vector of the decomposition basis + intensity.
    pub stain_e: [f32; 4],
    /// `[h_visible, e_visible, 0, 0]` as 0.0/1.0.
    pub visibility: [f32; 4],
    /// 0 = blended, 1 = hematoxylin isolated, 2 = eosin isolated.
    pub isolated_mode: f32,
    /// Any parameter differs from the identity.
    pub enabled: bool,
}

impl Default for ColorDeconvParams {
    fn default() -> Self {
        Self {
            inv_row0: [0.0; 4],
            inv_row1: [0.0; 4],
            stain_h: [0.0; 4],
            stain_e: [0.0; 4],
            visibility: [1.0, 1.0, 0.0, 0.0],
            isolated_mode: 0.0,
            enabled: false,
        }
    }
}

impl ColorDeconvParams {
    fn channels(&self) -> [StainChannel; 2] {
        [
            StainChannel {
                intensity: self.inv_row0[3],
                visible: self.visibility[0] > 0.5,
            },
            StainChannel {
                intensity: self.inv_row1[3],
                visible: self.visibility[1] > 0.5,
            },
        ]
    }

    fn isolated(&self) -> Option<usize> {
        if self.isolated_mode > 1.5 {
            Some(1)
        } else if self.isolated_mode > 0.5 {
            Some(0)
        } else {
            None
        }
    }

    fn basis(&self) -> Option<StainBasis> {
        StainBasis::new(vec![
            Stain::new(
                "Hematoxylin",
                [self.stain_h[0], self.stain_h[1], self.stain_h[2]],
            ),
            Stain::new("Eosin", [self.stain_e[0], self.stain_e[1], self.stain_e[2]]),
        ])
        .ok()
    }

    pub(crate) fn spec(&self) -> Option<StainTransformSpec> {
        if !self.enabled {
            return None;
        }
        Some(
            StainTransformSpec::deconvolution(self.basis()?)
                .with_channels(&self.channels())
                .with_isolated(self.isolated()),
        )
    }
}

/// Builds legacy deconvolution parameters from the HUD controls.
///
/// When `stain_norm_params` is enabled the buffer passed to
/// [`apply_color_deconvolution`] has already been normalised, i.e. its stain
/// content is expressed in the *target* basis, so the target basis is used
/// for the decomposition (CLR-3). Otherwise the default H&E vectors are used.
#[deprecated(
    since = "1.0.0",
    note = "use `processing::ColorSettings` / `ProcessingPipeline`, which fuse normalisation and \
            deconvolution into one pass"
)]
pub fn build_deconv_params(
    h_intensity: f32,
    h_visible: bool,
    e_intensity: f32,
    e_visible: bool,
    isolated_channel: u8, // 0=none, 1=hematoxylin, 2=eosin
    stain_norm_params: Option<&StainNormParams>,
) -> ColorDeconvParams {
    let is_default = h_visible
        && e_visible
        && (h_intensity - 1.0).abs() < 0.001
        && (e_intensity - 1.0).abs() < 0.001
        && isolated_channel == 0;
    if is_default {
        return ColorDeconvParams::default();
    }
    let normalized = stain_norm_params.is_some_and(|p| p.enabled);
    let basis = if normalized {
        NormalizationTarget::macenko_reference().basis
    } else {
        StainPreset::HematoxylinEosin.basis()
    };
    let p = basis.pseudo_inverse();
    let v = basis.vectors();
    ColorDeconvParams {
        inv_row0: [p[0][0], p[0][1], p[0][2], h_intensity],
        inv_row1: [p[1][0], p[1][1], p[1][2], e_intensity],
        stain_h: [v[0][0], v[0][1], v[0][2], h_intensity],
        stain_e: [v[1][0], v[1][1], v[1][2], e_intensity],
        visibility: [
            if h_visible { 1.0 } else { 0.0 },
            if e_visible { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        isolated_mode: isolated_channel.min(2) as f32,
        enabled: true,
    }
}

/// Applies legacy deconvolution parameters to an RGBA buffer.
///
/// Uses the residual-preserving transform: with all intensities at 1 the
/// output equals the input, and colours outside the H&E plane (RBCs,
/// pigment, DAB, ink) are not recoloured by intensity changes (CLR-4).
#[deprecated(
    since = "1.0.0",
    note = "use `processing::ProcessingPipeline::apply_rgba` (or `StainTransform::apply_rgba`)"
)]
pub fn apply_color_deconvolution(buffer: &mut [u8], params: &ColorDeconvParams) {
    let Some(spec) = params.spec() else {
        return;
    };
    let isolated = spec.isolated.is_some();
    if let Ok(transform) = spec.compile() {
        apply_skipping_dark_fill(&transform, buffer, isolated);
    }
}

impl StainGpuParams {
    /// Fuses legacy normalisation and deconvolution parameters (as carried
    /// by the pre-1.0 render path) into one GPU transform: decompose with the
    /// fitted source basis, scale and apply the user's intensities in
    /// concentration space, reconstruct once with the target basis.
    pub fn from_legacy(
        stain_norm_enabled: bool,
        inv_stain_r0: [f32; 4],
        inv_stain_r1: [f32; 4],
        deconv: &ColorDeconvParams,
    ) -> Self {
        let norm = StainNormParams {
            enabled: stain_norm_enabled,
            inv_stain_r0,
            inv_stain_r1,
        };
        let spec = match (norm.spec(), deconv.enabled) {
            (Some(spec), true) => Some(
                spec.with_channels(&deconv.channels())
                    .with_isolated(deconv.isolated()),
            ),
            (Some(spec), false) => Some(spec),
            (None, true) => deconv.spec(),
            (None, false) => None,
        };
        spec.and_then(|s| s.compile().ok())
            .map(|t| t.gpu_params())
            .unwrap_or_default()
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::super::estimate::tests::synthetic_field;
    use super::*;

    #[test]
    fn default_deconv_matrix_is_the_he_preset() {
        let preset = StainPreset::HematoxylinEosin.basis().vectors();
        for (row, v) in DEFAULT_DECONV_MATRIX.iter().zip(&preset) {
            for k in 0..3 {
                assert!((row[k] - v[k]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn legacy_view_fit_produces_finite_params() {
        let pixels = synthetic_field(
            &StainPreset::HematoxylinEosin.basis().vectors(),
            [255.0; 3],
            4000,
            1000,
        );
        let params = compute_cpu_stain_params(StainNormalization::Macenko, &[&pixels]);
        assert!(params.enabled);
        assert!(
            params
                .inv_stain_r0
                .iter()
                .chain(&params.inv_stain_r1)
                .all(|v| v.is_finite())
        );
        assert!(params.inv_stain_r0[3] > 0.0 && params.inv_stain_r1[3] > 0.0);
    }

    #[test]
    fn legacy_view_fit_is_disabled_without_tissue() {
        let glass: Vec<u8> = (0..4000).flat_map(|_| [240u8, 238, 242, 255]).collect();
        let params = compute_cpu_stain_params(StainNormalization::Macenko, &[&glass]);
        assert!(!params.enabled);
    }

    #[test]
    fn legacy_normalisation_preserves_background_and_dark_fill() {
        let pixels = synthetic_field(
            &StainPreset::HematoxylinEosin.basis().vectors(),
            [255.0; 3],
            4000,
            1000,
        );
        let params = compute_cpu_stain_params(StainNormalization::Macenko, &[&pixels]);
        let mut buffer = vec![255u8, 255, 255, 255, 30, 30, 30, 255];
        apply_stain_params_to_buffer(&mut buffer, &params);
        assert_eq!(&buffer[..4], &[255, 255, 255, 255]);
        assert_eq!(
            &buffer[4..],
            &[30, 30, 30, 255],
            "the viewport fill must not be recoloured"
        );
    }

    #[test]
    fn deconvolution_after_normalisation_uses_the_target_basis() {
        // CLR-3: a pixel H = 1.0, E = 0.5 *in the target basis* must
        // decompose as exactly that after normalisation.
        let norm = StainNormParams {
            enabled: true,
            inv_stain_r0: [1.0, 0.0, 0.0, 1.0],
            inv_stain_r1: [0.0, 1.0, 0.0, 1.0],
        };
        let params = build_deconv_params(1.0, true, 1.0, true, 1, Some(&norm));
        let target = NormalizationTarget::macenko_reference().basis.vectors();
        let p = [
            [params.inv_row0[0], params.inv_row0[1], params.inv_row0[2]],
            [params.inv_row1[0], params.inv_row1[1], params.inv_row1[2]],
        ];
        let od: Vec<f32> = (0..3).map(|c| target[0][c] + 0.5 * target[1][c]).collect();
        let c_h = linalg::dot(p[0], [od[0], od[1], od[2]]);
        let c_e = linalg::dot(p[1], [od[0], od[1], od[2]]);
        assert!(
            (c_h - 1.0).abs() < 1e-4 && (c_e - 0.5).abs() < 1e-4,
            "{c_h} {c_e}"
        );
    }

    #[test]
    fn legacy_deconvolution_keeps_non_stain_colours() {
        // The audit's CLR-4 table: H × 1.01 used to turn RBCs, DAB and ink
        // blue-magenta.
        let params = build_deconv_params(1.01, true, 1.0, true, 0, None);
        let mut buffer = vec![200, 60, 70, 255, 150, 100, 60, 255, 40, 40, 45, 255];
        let original = buffer.clone();
        apply_color_deconvolution(&mut buffer, &params);
        for (a, b) in buffer.iter().zip(&original) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 2,
                "{buffer:?} vs {original:?}"
            );
        }
    }

    #[test]
    fn legacy_gpu_params_fuse_both_stages() {
        assert_eq!(
            StainGpuParams::from_legacy(false, [0.0; 4], [0.0; 4], &ColorDeconvParams::default()),
            StainGpuParams::default()
        );
        let deconv = build_deconv_params(1.5, true, 1.0, false, 0, None);
        let gpu = StainGpuParams::from_legacy(false, [0.0; 4], [0.0; 4], &deconv);
        assert_eq!(gpu.mode[0], 1.0);
        assert_eq!(gpu.mode[3], 2.0);
        let iso = build_deconv_params(2.0, true, 1.0, true, 1, None);
        let gpu = StainGpuParams::from_legacy(false, [0.0; 4], [0.0; 4], &iso);
        assert_eq!(gpu.mode[1], 1.0);
        assert!((gpu.mode[2] - 2.0).abs() < 1e-6);
    }
}
