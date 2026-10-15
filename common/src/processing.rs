//! The canonical display-processing pipeline (CLR-5) and colour-provenance
//! helpers (CLR-9).
//!
//! # Order of operations
//!
//! Every renderer — the CPU viewport, the GPU shaders, the export renderer
//! and dataset extraction — applies the user's processing in this order, per
//! output (screen or export) pixel:
//!
//! 1. **Resample** the slide (bilinear / trilinear / Lanczos, pixel-centre
//!    convention) — `blitter` on the CPU, the texture sampler on the GPU.
//! 2. **Stain transform**: fused stain normalisation + colour deconvolution
//!    ([`StainTransform`]), computed from the resampled value.
//! 3. **Tone**: gamma → brightness → contrast ([`ToneAdjustments`]).
//! 4. **Quantise** to 8 bits, rounding to nearest. Stages 2–3 are evaluated
//!    in floating point without intermediate rounding.
//! 5. **Composite** pixels outside the slide (alpha 0) onto the background.
//! 6. **Sharpen** in *screen space*: a 3×3 Laplacian unsharp mask
//!    `c + k·(4c − n − s − w − e)` on the quantised, composited frame, with
//!    clamp-to-edge borders, rounded. Sharpening therefore does not depend on
//!    the zoom level or the pyramid level being displayed.
//! 7. **Overlays** (annotations, measurements, scale bar, badges).
//!
//! The CPU implementation in this module is the reference. The WGSL in
//! [`wgsl`] implements stages 2–3 identically; [`wgsl::emulate_color`] is a
//! line-by-line Rust transcription of that WGSL used by the golden tests.
//! On the GPU, stage 6 belongs in a separate full-screen pass over the
//! composited frame ([`wgsl::SHARPEN_POST_PASS`]).

use serde::{Deserialize, Serialize};

use crate::postprocess::{self, ToneAdjustments, quantize};
use crate::stain::{
    FitChecks, NormalizationTarget, StainBasis, StainChannel, StainEstimate, StainFitMethod,
    StainGpuParams, StainPreset, StainTransform, StainTransformError, StainTransformSpec,
    stain_model_warning,
};

/// Which stain vectors deconvolution adjustments use when the slide is not
/// being normalised.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StainModel {
    Preset(StainPreset),
    Custom(StainBasis),
}

impl Default for StainModel {
    fn default() -> Self {
        Self::Preset(StainPreset::HematoxylinEosin)
    }
}

impl StainModel {
    pub fn basis(&self) -> StainBasis {
        match self {
            Self::Preset(preset) => preset.basis(),
            Self::Custom(basis) => basis.clone(),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Preset(preset) => preset.label().to_string(),
            Self::Custom(basis) => format!("custom ({})", basis.names().join(", ")),
        }
    }

    /// `true` for the H&E presets (the only model normalisation supports).
    pub fn is_hematoxylin_eosin(&self) -> bool {
        matches!(self, Self::Preset(p) if p.is_hematoxylin_eosin())
    }
}

/// Everything the user can change about how a slide's colours are shown.
/// Plain data (the HUD and export dialog state); resolve it into a
/// [`ProcessingPipeline`] with [`ColorSettings::pipeline`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorSettings {
    /// Stain normalisation method (`None` = off).
    pub normalization: Option<StainFitMethod>,
    /// Normalisation target; `None` = Macenko's reference H&E appearance.
    pub target: Option<NormalizationTarget>,
    /// Stain vectors for deconvolution adjustments without normalisation.
    pub stain_model: StainModel,
    /// Per-stain intensity and visibility (missing entries are neutral).
    pub channels: Vec<StainChannel>,
    /// Show only this stain, as greyscale absorbance.
    pub isolated: Option<usize>,
    pub tone: ToneAdjustments,
    /// Screen-space sharpening amount (0 = off).
    pub sharpness: f32,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            normalization: None,
            target: None,
            stain_model: StainModel::default(),
            channels: Vec::new(),
            isolated: None,
            tone: ToneAdjustments::default(),
            sharpness: 0.0,
        }
    }
}

impl ColorSettings {
    /// Builds settings from the pre-1.0 HUD/export fields (H&E model).
    #[allow(clippy::too_many_arguments)]
    pub fn from_legacy(
        normalization: crate::StainNormalization,
        h_intensity: f32,
        h_visible: bool,
        e_intensity: f32,
        e_visible: bool,
        isolated_channel: u8,
        tone: ToneAdjustments,
        sharpness: f32,
    ) -> Self {
        Self {
            normalization: StainFitMethod::from_normalization(normalization),
            target: None,
            stain_model: StainModel::default(),
            channels: vec![
                StainChannel {
                    intensity: h_intensity,
                    visible: h_visible,
                },
                StainChannel {
                    intensity: e_intensity,
                    visible: e_visible,
                },
            ],
            isolated: match isolated_channel {
                1 => Some(0),
                2 => Some(1),
                _ => None,
            },
            tone,
            sharpness,
        }
    }

    fn channels_are_neutral(&self) -> bool {
        self.channels.iter().all(StainChannel::is_neutral)
    }

    /// The fit this configuration needs (`None` when not normalising or when
    /// the stain model does not support normalisation).
    pub fn required_estimate(&self) -> Option<StainFitMethod> {
        self.normalization
            .filter(|_| self.stain_model.is_hematoxylin_eosin())
    }

    /// `true` if stain normalisation, deconvolution adjustments or channel
    /// isolation change the colours.
    pub fn alters_stain_colors(&self) -> bool {
        self.required_estimate().is_some()
            || self.isolated.is_some()
            || !self.channels_are_neutral()
    }

    /// `true` if *any* processing is active (use this for the HUD
    /// "Adjustments" indicator, CLR-9).
    pub fn is_modified(&self) -> bool {
        self.alters_stain_colors() || !self.tone.is_identity() || self.sharpness > 0.001
    }

    /// Problems with this configuration for a slide with the given stain
    /// label, as user-facing messages.
    pub fn warnings(&self, stain_label: Option<&str>) -> Vec<String> {
        let mut out = Vec::new();
        if self.normalization.is_some() && !self.stain_model.is_hematoxylin_eosin() {
            out.push(format!(
                "Stain normalisation only supports H&E; it is disabled for the {} stain model.",
                self.stain_model.label()
            ));
        }
        let uses_he = self.stain_model.is_hematoxylin_eosin() && self.alters_stain_colors();
        if let Some(warning) = stain_model_warning(stain_label, uses_he) {
            out.push(warning);
        }
        out.extend(
            self.plan(None)
                .1
                .not_applied
                .into_iter()
                .map(|item| format!("Not applied: {item}.")),
        );
        out
    }

    /// Short label for an on-screen / burned-in badge, e.g.
    /// `"Colour-normalised (Macenko)"`; `None` when nothing is altered.
    pub fn badge(&self) -> Option<String> {
        self.plan(None).1.badge()
    }

    /// Resolves the stain stage these settings ask for and describes what
    /// will actually be applied. Requests that cannot be honoured (a
    /// normalisation target with another number of stains, isolating a stain
    /// the model does not have, adjusting stains beyond the model) are left
    /// out of the stage and listed in [`ProcessingSummary::not_applied`], so
    /// the summary never claims processing the pixels did not get.
    fn plan(
        &self,
        estimate: Option<&StainEstimate>,
    ) -> (Option<StainTransformSpec>, ProcessingSummary) {
        let mut summary = ProcessingSummary::default();
        let required = self.required_estimate();
        let mut spec = None;
        match (required, estimate) {
            (Some(method), None) => {
                summary.normalization = Some(method.label().to_string());
                summary.normalization_pending = true;
            }
            (Some(method), Some(est)) => {
                let target = self.target.clone().unwrap_or_default();
                match StainTransformSpec::normalization(est, &target, &FitChecks::default()) {
                    Ok(s) => {
                        summary.normalization = Some(method.label().to_string());
                        summary.normalization_fallback =
                            est.fallback.as_ref().map(|f| f.describe());
                        summary.estimate = Some(est.summary());
                        spec = Some(s);
                    }
                    Err(e) => summary
                        .not_applied
                        .push(format!("{} stain normalisation ({e})", method.label())),
                }
            }
            (None, _) => {}
        }
        if spec.is_none() && (self.isolated.is_some() || !self.channels_are_neutral()) {
            // Deconvolution adjustments without normalisation. While an H&E
            // normalisation is pending (or failed) they use the H&E preset.
            let (basis, label) = if required.is_some() {
                let preset = StainPreset::HematoxylinEosin;
                (preset.basis(), preset.label().to_string())
            } else {
                (self.stain_model.basis(), self.stain_model.label())
            };
            let i0 = estimate.map(|e| e.i0).unwrap_or([255.0; 3]);
            summary.stain_model = Some(label);
            spec = Some(StainTransformSpec::deconvolution(basis).with_background(i0));
        }
        let spec = spec.and_then(|spec| {
            let names: Vec<String> = spec.source.names().iter().map(|s| s.to_string()).collect();
            let isolated = match self.isolated {
                Some(index) if index >= names.len() => {
                    summary.not_applied.push(format!(
                        "isolating stain {} (the stain model has {} stains)",
                        index + 1,
                        names.len()
                    ));
                    None
                }
                other => other,
            };
            if self
                .channels
                .iter()
                .skip(names.len())
                .any(|c| !c.is_neutral())
            {
                summary.not_applied.push(format!(
                    "adjustments of stains beyond the {} of the stain model",
                    names.len()
                ));
            }
            let spec = spec.with_channels(&self.channels).with_isolated(isolated);
            if spec.is_identity() {
                // Only requests that cannot be applied were left.
                summary.stain_model = None;
                return None;
            }
            summary.describe_channels(&names, &spec.channels, spec.isolated);
            Some(spec)
        });
        summary.describe_tone(self.tone, self.sharpness);
        (spec, summary)
    }

    /// Resolves the settings into a ready-to-apply pipeline.
    ///
    /// `estimate` is the slide-level fit for [`required_estimate`](Self::required_estimate);
    /// pass `None` while it is still being computed (normalisation is then
    /// skipped and the summary reports it as pending). When an estimate is
    /// given its background `I₀` is also used for deconvolution-only
    /// settings. Requests that cannot be applied are skipped and listed in
    /// the summary's [`not_applied`](ProcessingSummary::not_applied).
    pub fn pipeline(&self, estimate: Option<&StainEstimate>) -> ProcessingPipeline {
        let (spec, mut summary) = self.plan(estimate);
        let compiled = spec.as_ref().map(StainTransformSpec::compile);
        let (stain, spec) = match compiled {
            Some(Ok(transform)) => (transform, spec),
            Some(Err(e)) => {
                // `plan` validates everything `compile` checks; never claim
                // a stage that is not applied all the same.
                summary.clear_stain();
                summary.not_applied.push(e.to_string());
                (StainTransform::identity(), None)
            }
            None => (StainTransform::identity(), None),
        };
        ProcessingPipeline::from_parts(stain, spec, self.tone, self.sharpness, summary)
    }
}

/// What a pipeline does, for badges and provenance (CLR-9, EXP-5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessingSummary {
    /// Normalisation method label, e.g. `"Macenko"`.
    pub normalization: Option<String>,
    /// Normalisation was requested but the slide fit is not available yet.
    pub normalization_pending: bool,
    /// Why the fit fell back to reference vectors, if it did.
    pub normalization_fallback: Option<String>,
    /// One-line description of the stain estimate.
    pub estimate: Option<String>,
    /// Stain model used for deconvolution adjustments (when not normalising).
    pub stain_model: Option<String>,
    /// e.g. `"Hematoxylin ×1.20"`, `"Eosin hidden"`.
    pub stain_adjustments: Vec<String>,
    /// Name of the isolated stain.
    pub isolated: Option<String>,
    /// e.g. `"gamma 1.20"`.
    pub tone: Vec<String>,
    pub sharpness: Option<f32>,
    /// Requested processing that could not be applied, with the reason,
    /// e.g. `"isolating stain 3 (the stain model has 2 stains)"`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_applied: Vec<String>,
}

impl ProcessingSummary {
    /// Lists the per-stain adjustments and the isolated stain of an applied
    /// stain stage whose stains are called `names`.
    fn describe_channels(
        &mut self,
        names: &[String],
        channels: &[StainChannel],
        isolated: Option<usize>,
    ) {
        for (name, channel) in names.iter().zip(channels) {
            if !channel.visible {
                self.stain_adjustments.push(format!("{name} hidden"));
            } else if (channel.intensity - 1.0).abs() > 0.001 {
                self.stain_adjustments
                    .push(format!("{name} ×{:.2}", channel.intensity));
            }
        }
        self.isolated = isolated.and_then(|i| names.get(i).cloned());
    }

    fn describe_tone(&mut self, tone: ToneAdjustments, sharpness: f32) {
        if (tone.gamma - 1.0).abs() > 0.001 {
            self.tone.push(format!("gamma {:.2}", tone.gamma));
        }
        if tone.brightness.abs() > 0.001 {
            self.tone
                .push(format!("brightness {:+.3}", tone.brightness));
        }
        if (tone.contrast - 1.0).abs() > 0.001 {
            self.tone.push(format!("contrast ×{:.2}", tone.contrast));
        }
        if sharpness > 0.001 {
            self.sharpness = Some(sharpness);
        }
    }

    /// Forgets the stain stage (it was not applied).
    fn clear_stain(&mut self) {
        self.normalization = None;
        self.normalization_pending = false;
        self.normalization_fallback = None;
        self.estimate = None;
        self.stain_model = None;
        self.stain_adjustments.clear();
        self.isolated = None;
    }

    /// Describes an explicitly built stain stage (see
    /// [`ProcessingPipeline::new`]).
    fn for_spec(spec: Option<&StainTransformSpec>) -> Self {
        let mut summary = Self::default();
        if let Some(spec) = spec.filter(|s| !s.is_identity()) {
            let names: Vec<String> = spec.source.names().iter().map(|s| s.to_string()).collect();
            if spec.normalization.is_some() {
                summary.normalization = Some("custom".to_string());
            } else {
                summary.stain_model = Some(format!("custom ({})", names.join(", ")));
            }
            summary.describe_channels(&names, &spec.channels, spec.isolated);
        }
        summary
    }

    /// Stain normalisation, deconvolution or isolation is active.
    pub fn alters_stain_colors(&self) -> bool {
        (self.normalization.is_some() && !self.normalization_pending)
            || !self.stain_adjustments.is_empty()
            || self.isolated.is_some()
    }

    /// Any processing is active.
    pub fn is_modified(&self) -> bool {
        self.alters_stain_colors()
            || self.normalization_pending
            || !self.tone.is_empty()
            || self.sharpness.is_some()
    }

    /// Short badge text, e.g. `"Colour-normalised (Macenko) · Sharpened"`.
    pub fn badge(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(method) = &self.normalization {
            if self.normalization_pending {
                parts.push(format!("Colour normalisation pending ({method})"));
            } else if self.normalization_fallback.is_some() {
                parts.push(format!("Colour-normalised ({method}, reference vectors)"));
            } else {
                parts.push(format!("Colour-normalised ({method})"));
            }
        }
        if let Some(stain) = &self.isolated {
            parts.push(format!("Isolated stain: {stain}"));
        }
        if !self.stain_adjustments.is_empty() {
            parts.push(format!(
                "Stain-adjusted ({})",
                self.stain_adjustments.join(", ")
            ));
        }
        if !self.tone.is_empty() {
            parts.push("Display-adjusted".to_string());
        }
        if self.sharpness.is_some() {
            parts.push("Sharpened".to_string());
        }
        if !self.not_applied.is_empty() {
            parts.push("Some adjustments not applied".to_string());
        }
        (!parts.is_empty()).then(|| parts.join(" · "))
    }

    /// Full provenance text for export metadata and burned-in captions.
    pub fn provenance(&self) -> String {
        let not_applied = (!self.not_applied.is_empty())
            .then(|| format!("Not applied: {}.", self.not_applied.join("; ")));
        if !self.is_modified() {
            let scanned = "Colours as scanned (no processing).".to_string();
            return match not_applied {
                Some(line) => format!("{scanned} {line}"),
                None => scanned,
            };
        }
        let mut lines = Vec::new();
        if let Some(method) = &self.normalization {
            if self.normalization_pending {
                lines.push(format!(
                    "{method} stain normalisation requested but not applied (fit pending)."
                ));
            } else {
                let mut line = format!("{method} stain normalisation");
                if let Some(estimate) = &self.estimate {
                    line.push_str(&format!(": {estimate}"));
                }
                line.push('.');
                lines.push(line);
            }
        }
        if let Some(model) = &self.stain_model {
            lines.push(format!(
                "Colour deconvolution with the {model} stain model."
            ));
        }
        if !self.stain_adjustments.is_empty() {
            lines.push(format!(
                "Stain adjustments: {}.",
                self.stain_adjustments.join(", ")
            ));
        }
        if let Some(stain) = &self.isolated {
            lines.push(format!(
                "Showing the {stain} channel only (greyscale absorbance)."
            ));
        }
        if !self.tone.is_empty() {
            lines.push(format!("Tone: {}.", self.tone.join(", ")));
        }
        if let Some(amount) = self.sharpness {
            lines.push(format!(
                "Sharpened (screen-space Laplacian, amount {amount:.2})."
            ));
        }
        lines.extend(not_applied);
        lines.join(" ")
    }
}

/// A resolved pipeline: stain transform + tone + sharpening, in the
/// canonical order. Cheap to clone; safe to share across threads.
#[derive(Clone, Debug)]
pub struct ProcessingPipeline {
    stain: StainTransform,
    stain_spec: Option<StainTransformSpec>,
    tone: ToneAdjustments,
    tone_lut: [u8; 256],
    sharpness: f32,
    summary: ProcessingSummary,
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self::identity()
    }
}

impl ProcessingPipeline {
    pub fn identity() -> Self {
        Self {
            stain: StainTransform::identity(),
            stain_spec: None,
            tone: ToneAdjustments::default(),
            tone_lut: ToneAdjustments::default().lut(),
            sharpness: 0.0,
            summary: ProcessingSummary::default(),
        }
    }

    /// Builds a pipeline from explicit stages. The summary describes the
    /// stain stage only generically (a custom normalisation or stain model);
    /// prefer [`ColorSettings::pipeline`], which names the method and the
    /// stain model.
    pub fn new(
        stain: Option<&StainTransformSpec>,
        tone: ToneAdjustments,
        sharpness: f32,
    ) -> Result<Self, StainTransformError> {
        let transform = match stain {
            Some(spec) => spec.compile()?,
            None => StainTransform::identity(),
        };
        let mut summary = ProcessingSummary::for_spec(stain);
        summary.describe_tone(tone, sharpness);
        Ok(Self::from_parts(
            transform,
            stain.cloned(),
            tone,
            sharpness,
            summary,
        ))
    }

    fn from_parts(
        stain: StainTransform,
        stain_spec: Option<StainTransformSpec>,
        tone: ToneAdjustments,
        sharpness: f32,
        summary: ProcessingSummary,
    ) -> Self {
        Self {
            stain,
            stain_spec,
            tone,
            tone_lut: tone.lut(),
            sharpness: if sharpness > 0.001 { sharpness } else { 0.0 },
            summary,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.stain.is_identity() && self.tone.is_identity() && self.sharpness == 0.0
    }

    pub fn stain_transform(&self) -> &StainTransform {
        &self.stain
    }

    pub fn stain_spec(&self) -> Option<&StainTransformSpec> {
        self.stain_spec.as_ref()
    }

    pub fn tone(&self) -> ToneAdjustments {
        self.tone
    }

    /// Effective sharpening amount (0 when off).
    pub fn sharpness(&self) -> f32 {
        self.sharpness
    }

    pub fn summary(&self) -> &ProcessingSummary {
        &self.summary
    }

    /// Stages 2–4 for one pixel (stain → tone → round).
    #[inline(always)]
    pub fn color_pixel(&self, rgb: [u8; 3]) -> [u8; 3] {
        if self.stain.is_identity() {
            return rgb.map(|v| self.tone_lut[v as usize]);
        }
        let y = self.stain.apply_pixel_unrounded(rgb);
        if self.tone.is_identity() {
            y.map(|v| v.round().clamp(0.0, 255.0) as u8)
        } else {
            y.map(|v| quantize(self.tone.apply_unit(v / 255.0)))
        }
    }

    /// Stages 2–4 on an RGBA buffer (parallel); pixels with alpha 0 are
    /// outside the slide and untouched.
    pub fn apply_color(&self, buffer: &mut [u8]) {
        use rayon::prelude::*;
        if self.stain.is_identity() && self.tone.is_identity() {
            return;
        }
        buffer.par_chunks_mut(4 * 16_384).for_each(|chunk| {
            for px in chunk.as_chunks_mut::<4>().0 {
                if px[3] == 0 {
                    continue;
                }
                let out = self.color_pixel([px[0], px[1], px[2]]);
                px[..3].copy_from_slice(&out);
            }
        });
    }

    /// Stage 6 on a full frame.
    pub fn sharpen(&self, buffer: &mut [u8], width: u32, height: u32) {
        postprocess::sharpen_rgba(buffer, width, height, self.sharpness);
    }

    /// Runs stages 2–6 on a rendered frame. Pixels with alpha 0 are
    /// composited onto `background` (stage 5); partially transparent pixels
    /// are blended (straight alpha).
    pub fn apply_rgba(&self, buffer: &mut [u8], width: u32, height: u32, background: [u8; 3]) {
        self.apply_color(buffer);
        composite_onto(buffer, background);
        self.sharpen(buffer, width, height);
    }

    /// Uniform block for the WGSL in [`wgsl`].
    pub fn gpu_uniform(&self) -> GpuColorUniform {
        GpuColorUniform {
            tone: [
                self.tone.inv_gamma(),
                self.tone.brightness,
                self.tone.contrast,
                self.sharpness,
            ],
            stain: self.stain.gpu_params(),
        }
    }
}

/// Composites non-opaque pixels onto an opaque background (straight alpha)
/// and makes every pixel opaque.
pub fn composite_onto(buffer: &mut [u8], background: [u8; 3]) {
    use rayon::prelude::*;
    buffer.par_chunks_mut(4 * 16_384).for_each(|chunk| {
        for px in chunk.as_chunks_mut::<4>().0 {
            match px[3] {
                255 => {}
                0 => {
                    px[..3].copy_from_slice(&background);
                    px[3] = 255;
                }
                a => {
                    let a = a as u32;
                    for c in 0..3 {
                        px[c] = ((px[c] as u32 * a + background[c] as u32 * (255 - a) + 127) / 255)
                            as u8;
                    }
                    px[3] = 255;
                }
            }
        }
    });
}

/// GPU uniform block for stages 2–3 (and the sharpening amount): 9 × vec4
/// (144 bytes), std140-compatible. Matches `EovColorParams` in
/// [`wgsl::COLOR_PARAMS_STRUCT`].
///
/// * `tone` — `x`: 1/γ, `y`: brightness, `z`: contrast, `w`: sharpening
///   amount.
/// * `stain` — see [`StainGpuParams`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpuColorUniform {
    pub tone: [f32; 4],
    pub stain: StainGpuParams,
}

impl GpuColorUniform {
    pub const SIZE: usize = 144;

    /// The 36 floats in declaration order (for `bytemuck::cast_slice`).
    pub fn to_array(&self) -> [f32; 36] {
        let mut out = [0.0f32; 36];
        let rows = [
            self.tone,
            self.stain.mode,
            self.stain.log_i0,
            self.stain.pinv[0],
            self.stain.pinv[1],
            self.stain.pinv[2],
            self.stain.w[0],
            self.stain.w[1],
            self.stain.w[2],
        ];
        for (i, row) in rows.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(row);
        }
        out
    }

    /// Builds the uniform from the pre-1.0 render-path parameters (see
    /// [`StainGpuParams::from_legacy`]).
    #[allow(clippy::too_many_arguments)]
    pub fn from_legacy(
        gamma: f32,
        brightness: f32,
        contrast: f32,
        sharpness: f32,
        stain_norm_enabled: bool,
        inv_stain_r0: [f32; 4],
        inv_stain_r1: [f32; 4],
        deconv: &crate::stain::ColorDeconvParams,
    ) -> Self {
        let tone = ToneAdjustments::new(gamma, brightness, contrast);
        Self {
            tone: [
                tone.inv_gamma(),
                brightness,
                contrast,
                if sharpness > 0.001 { sharpness } else { 0.0 },
            ],
            stain: StainGpuParams::from_legacy(
                stain_norm_enabled,
                inv_stain_r0,
                inv_stain_r1,
                deconv,
            ),
        }
    }
}

/// Canonical WGSL for the pipeline, and a Rust transcription of it.
pub mod wgsl {
    use super::GpuColorUniform;

    /// The uniform struct (binds [`GpuColorUniform`]).
    pub const COLOR_PARAMS_STRUCT: &str = r#"
struct EovColorParams {
    // x: 1/gamma, y: brightness, z: contrast, w: sharpening amount
    tone: vec4<f32>,
    // x: stain transform enabled, y: isolated stain index + 1 (0 = blended),
    // z: isolated stain multiplier, w: number of stains
    stain_mode: vec4<f32>,
    // xyz: ln(I0) per channel
    stain_log_i0: vec4<f32>,
    // rows of the source pseudo-inverse S+
    stain_pinv0: vec4<f32>,
    stain_pinv1: vec4<f32>,
    stain_pinv2: vec4<f32>,
    // columns of W = T * diag(m) - S
    stain_w0: vec4<f32>,
    stain_w1: vec4<f32>,
    stain_w2: vec4<f32>,
};
"#;

    /// Stage 2 (`eov_stain`), stage 3 (`eov_tone`) and both (`eov_color`).
    /// Expects a uniform variable `color_params: EovColorParams` in scope.
    pub const COLOR_FUNCTIONS: &str = r#"
// Fused stain normalisation + deconvolution (eov_common::stain::transform).
fn eov_stain(rgb: vec3<f32>) -> vec3<f32> {
    if (color_params.stain_mode.x < 0.5) {
        return rgb;
    }
    let od = color_params.stain_log_i0.xyz - log(max(rgb * 255.0, vec3<f32>(1.0)));
    let c = max(vec3<f32>(
        dot(color_params.stain_pinv0.xyz, od),
        dot(color_params.stain_pinv1.xyz, od),
        dot(color_params.stain_pinv2.xyz, od),
    ), vec3<f32>(0.0));
    let iso = i32(color_params.stain_mode.y + 0.5);
    if (iso > 0) {
        var ci = c.x;
        if (iso == 2) {
            ci = c.y;
        }
        if (iso == 3) {
            ci = c.z;
        }
        let g = exp(-color_params.stain_mode.z * ci);
        return vec3<f32>(g, g, g);
    }
    let delta = color_params.stain_w0.xyz * c.x
        + color_params.stain_w1.xyz * c.y
        + color_params.stain_w2.xyz * c.z;
    return clamp(rgb * exp(-delta), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Gamma, brightness, contrast (eov_common::postprocess::ToneAdjustments).
fn eov_tone(rgb: vec3<f32>) -> vec3<f32> {
    let x = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var g = x;
    if (abs(color_params.tone.x - 1.0) > 0.000001) {
        g = pow(x, vec3<f32>(color_params.tone.x));
    }
    let c = (g + vec3<f32>(color_params.tone.y) - vec3<f32>(0.5)) * color_params.tone.z + vec3<f32>(0.5);
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn eov_color(rgb: vec3<f32>) -> vec3<f32> {
    return eov_tone(eov_stain(rgb));
}
"#;

    /// Stage 6 as a full-screen pass over the composited 8-bit frame:
    /// bind the frame as `eov_frame` (binding 0) and a 16-byte uniform whose
    /// first float is the sharpening amount (binding 1); draw 3 vertices.
    pub const SHARPEN_POST_PASS: &str = r#"
struct EovSharpenParams {
    amount: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var eov_frame: texture_2d<f32>;
@group(0) @binding(1) var<uniform> eov_sharpen: EovSharpenParams;

struct EovSharpenVertex {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_eov_sharpen(@builtin(vertex_index) index: u32) -> EovSharpenVertex {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: EovSharpenVertex;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    return out;
}

fn eov_frame_at(p: vec2<i32>, size: vec2<i32>) -> vec3<f32> {
    return textureLoad(eov_frame, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

@fragment
fn fs_eov_sharpen(input: EovSharpenVertex) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(eov_frame, 0));
    let p = vec2<i32>(input.position.xy);
    let centre = textureLoad(eov_frame, p, 0);
    if (eov_sharpen.amount <= 0.001) {
        return centre;
    }
    let n = eov_frame_at(p + vec2<i32>(0, -1), size);
    let s = eov_frame_at(p + vec2<i32>(0, 1), size);
    let w = eov_frame_at(p + vec2<i32>(-1, 0), size);
    let e = eov_frame_at(p + vec2<i32>(1, 0), size);
    let v = centre.rgb + eov_sharpen.amount * (4.0 * centre.rgb - (n + s + w + e));
    return vec4<f32>(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), centre.a);
}
"#;

    /// Line-by-line Rust transcription of `eov_color` for a
    /// [`GpuColorUniform`]: what the shader computes for an unquantised
    /// `[0, 1]` input. Used by golden tests to check the uniform contract
    /// and CPU/GPU parity.
    pub fn emulate_color(params: &GpuColorUniform, rgb: [f32; 3]) -> [f32; 3] {
        let stained = emulate_stain(params, rgb);
        emulate_tone(params, stained)
    }

    fn dot3(a: [f32; 4], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn emulate_stain(params: &GpuColorUniform, rgb: [f32; 3]) -> [f32; 3] {
        let s = &params.stain;
        if s.mode[0] < 0.5 {
            return rgb;
        }
        let od = [
            s.log_i0[0] - (rgb[0] * 255.0).max(1.0).ln(),
            s.log_i0[1] - (rgb[1] * 255.0).max(1.0).ln(),
            s.log_i0[2] - (rgb[2] * 255.0).max(1.0).ln(),
        ];
        let c = [
            dot3(s.pinv[0], od).max(0.0),
            dot3(s.pinv[1], od).max(0.0),
            dot3(s.pinv[2], od).max(0.0),
        ];
        let iso = (s.mode[1] + 0.5) as i32;
        if iso > 0 {
            let ci = match iso {
                2 => c[1],
                3 => c[2],
                _ => c[0],
            };
            let g = (-s.mode[2] * ci).exp();
            return [g, g, g];
        }
        let mut out = [0.0f32; 3];
        for (ch, value) in out.iter_mut().enumerate() {
            let delta = s.w[0][ch] * c[0] + s.w[1][ch] * c[1] + s.w[2][ch] * c[2];
            *value = (rgb[ch] * (-delta).exp()).clamp(0.0, 1.0);
        }
        out
    }

    fn emulate_tone(params: &GpuColorUniform, rgb: [f32; 3]) -> [f32; 3] {
        let t = params.tone;
        rgb.map(|v| {
            let x = v.clamp(0.0, 1.0);
            let g = if (t[0] - 1.0).abs() > 0.000_001 {
                x.powf(t[0])
            } else {
                x
            };
            ((g + t[1] - 0.5) * t[2] + 0.5).clamp(0.0, 1.0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stain::{EstimateSource, StainPreset};

    fn estimate() -> StainEstimate {
        StainEstimate {
            format_version: 1,
            method: StainFitMethod::Macenko,
            basis: StainPreset::HematoxylinEosin.basis(),
            max_concentrations: vec![1.6, 0.9],
            i0: [242.0, 240.0, 244.0],
            background_found: true,
            examined_pixels: 100,
            tissue_pixels: 50,
            tissue_fraction: 0.5,
            samples: 50,
            min_angle_deg: 35.0,
            condition_number: 3.2,
            fallback: None,
            source: EstimateSource::Slide {
                level: 2,
                downsample: 16.0,
                sparse: false,
            },
            slide_identity: None,
        }
    }

    fn sample_pixels() -> Vec<[u8; 3]> {
        let mut out = Vec::new();
        for r in (0..=255u32).step_by(17) {
            for g in (0..=255u32).step_by(51) {
                for b in (0..=255u32).step_by(85) {
                    out.push([r as u8, g as u8, b as u8]);
                }
            }
        }
        out.extend([
            [200, 60, 70],
            [115, 86, 175],
            [189, 139, 115],
            [236, 234, 239],
        ]);
        out
    }

    fn settings_variants() -> Vec<ColorSettings> {
        let tone = ToneAdjustments::new(1.3, 0.04, 1.2);
        vec![
            ColorSettings::default(),
            ColorSettings {
                tone,
                ..ColorSettings::default()
            },
            ColorSettings::from_legacy(
                crate::StainNormalization::None,
                1.5,
                true,
                0.7,
                true,
                0,
                tone,
                0.0,
            ),
            ColorSettings::from_legacy(
                crate::StainNormalization::Macenko,
                1.0,
                true,
                1.0,
                true,
                0,
                ToneAdjustments::default(),
                0.0,
            ),
            ColorSettings::from_legacy(
                crate::StainNormalization::Macenko,
                1.2,
                true,
                1.0,
                false,
                0,
                tone,
                0.0,
            ),
            ColorSettings::from_legacy(
                crate::StainNormalization::None,
                2.0,
                true,
                1.0,
                true,
                1,
                ToneAdjustments::default(),
                0.0,
            ),
            ColorSettings {
                stain_model: StainModel::Preset(StainPreset::HematoxylinEosinDab),
                channels: vec![
                    StainChannel::default(),
                    StainChannel::default(),
                    StainChannel {
                        intensity: 1.8,
                        visible: true,
                    },
                ],
                ..ColorSettings::default()
            },
        ]
    }

    #[test]
    fn gpu_emulation_matches_the_cpu_reference() {
        // CLR-5 golden test: the WGSL math (transcribed) and the CPU
        // reference agree to within one 8-bit level for every setting.
        let est = estimate();
        for settings in settings_variants() {
            let pipeline = settings.pipeline(Some(&est));
            let uniform = pipeline.gpu_uniform();
            for rgb in sample_pixels() {
                let cpu = pipeline.color_pixel(rgb);
                let gpu =
                    wgsl::emulate_color(&uniform, rgb.map(|v| v as f32 / 255.0)).map(quantize);
                for c in 0..3 {
                    assert!(
                        (cpu[c] as i32 - gpu[c] as i32).abs() <= 1,
                        "{settings:?}\n{rgb:?}: cpu {cpu:?} gpu {gpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn neutral_settings_are_exactly_the_identity() {
        let pipeline = ColorSettings::default().pipeline(Some(&estimate()));
        assert!(pipeline.is_identity());
        for rgb in sample_pixels() {
            assert_eq!(pipeline.color_pixel(rgb), rgb);
        }
        assert_eq!(pipeline.gpu_uniform().stain.mode[0], 0.0);
    }

    #[test]
    fn golden_values_for_tone_and_isolation() {
        // Independent expected values, computed in f64.
        let tone = ToneAdjustments::new(2.0, 0.0, 1.0);
        let pipeline = ColorSettings {
            tone,
            ..ColorSettings::default()
        }
        .pipeline(None);
        // sqrt(64/255) * 255 = 127.75 → 128.
        assert_eq!(pipeline.color_pixel([64, 64, 64]), [128, 128, 128]);

        // Isolated hematoxylin at 2× on a synthetic pixel with c_H = 0.5.
        let he = StainPreset::HematoxylinEosin.basis().vectors();
        let rgb = [0, 1, 2].map(|c| (255.0f64 * (-(he[0][c] as f64) * 0.5).exp()).round() as u8);
        let iso = ColorSettings::from_legacy(
            crate::StainNormalization::None,
            2.0,
            true,
            1.0,
            true,
            1,
            ToneAdjustments::default(),
            0.0,
        )
        .pipeline(None);
        let expected = (255.0f64 * (-1.0f64).exp()).round() as u8; // 94
        let got = iso.color_pixel(rgb);
        assert!(
            (got[0] as i32 - expected as i32).abs() <= 1,
            "{got:?} vs {expected}"
        );
    }

    #[test]
    fn apply_rgba_composites_then_sharpens() {
        let pipeline = ProcessingPipeline::new(None, ToneAdjustments::default(), 1.0).unwrap();
        // 3×1 image: slide pixel between two "outside" pixels.
        let mut buf = vec![0, 0, 0, 0, 100, 100, 100, 255, 0, 0, 0, 0];
        pipeline.apply_rgba(&mut buf, 3, 1, [255, 255, 255]);
        // Outside pixels became background; then the screen-space sharpen
        // ran on the composited frame: 100 + (400 − 100 − 100 − 255 − 255) < 0.
        assert_eq!(&buf[4..8], &[0, 0, 0, 255]);
        assert_eq!(buf[3], 255);
    }

    #[test]
    fn badges_describe_every_colour_altering_setting() {
        assert_eq!(ColorSettings::default().badge(), None);
        assert!(!ColorSettings::default().is_modified());
        let norm = ColorSettings::from_legacy(
            crate::StainNormalization::Macenko,
            1.0,
            true,
            1.0,
            true,
            0,
            ToneAdjustments::default(),
            0.0,
        );
        assert!(norm.alters_stain_colors());
        let pipeline = norm.pipeline(Some(&estimate()));
        assert_eq!(
            pipeline.summary().badge().as_deref(),
            Some("Colour-normalised (Macenko)")
        );
        assert!(
            pipeline
                .summary()
                .provenance()
                .contains("Macenko stain normalisation")
        );
        let pending = norm.pipeline(None);
        assert!(pending.summary().badge().unwrap().contains("pending"));
        let deconv = ColorSettings::from_legacy(
            crate::StainNormalization::None,
            1.2,
            true,
            1.0,
            false,
            0,
            ToneAdjustments::default(),
            0.5,
        );
        let badge = deconv.badge().unwrap();
        assert!(
            badge.contains("Hematoxylin ×1.20")
                && badge.contains("Eosin hidden")
                && badge.contains("Sharpened"),
            "{badge}"
        );
        let tone_only = ColorSettings {
            tone: ToneAdjustments::new(1.1, 0.0, 1.0),
            ..ColorSettings::default()
        };
        assert!(tone_only.is_modified() && !tone_only.alters_stain_colors());
        assert_eq!(tone_only.badge().as_deref(), Some("Display-adjusted"));
        let mut fallback = estimate();
        fallback.fallback = Some(crate::stain::FitFallback::FitFailed);
        assert!(
            norm.pipeline(Some(&fallback))
                .summary()
                .badge()
                .unwrap()
                .contains("reference vectors")
        );
    }

    #[test]
    fn warnings_flag_ihc_slides_and_unsupported_normalisation() {
        let norm = ColorSettings::from_legacy(
            crate::StainNormalization::Macenko,
            1.0,
            true,
            1.0,
            true,
            0,
            ToneAdjustments::default(),
            0.0,
        );
        assert_eq!(norm.warnings(Some("IHC")).len(), 1);
        assert!(norm.warnings(Some("H&E")).is_empty());
        let dab = ColorSettings {
            normalization: Some(StainFitMethod::Macenko),
            stain_model: StainModel::Preset(StainPreset::HematoxylinDab),
            ..ColorSettings::default()
        };
        assert!(dab.required_estimate().is_none());
        assert!(
            dab.warnings(Some("IHC"))
                .iter()
                .any(|w| w.contains("only supports H&E"))
        );
        let invalid = ColorSettings {
            isolated: Some(4),
            ..ColorSettings::default()
        };
        assert_eq!(
            invalid.warnings(None),
            vec!["Not applied: isolating stain 5 (the stain model has 2 stains).".to_string()]
        );
    }

    #[test]
    fn summaries_only_describe_what_is_applied() {
        // Isolating a stain the model does not have: the other adjustments
        // still apply, and the isolation is reported as not applied.
        let settings = ColorSettings {
            stain_model: StainModel::Preset(StainPreset::HematoxylinDab),
            channels: vec![StainChannel {
                intensity: 1.5,
                visible: true,
            }],
            isolated: Some(2),
            ..ColorSettings::default()
        };
        let pipeline = settings.pipeline(None);
        assert!(!pipeline.stain_transform().is_identity());
        let summary = pipeline.summary();
        assert_eq!(summary.isolated, None);
        assert_eq!(summary.not_applied.len(), 1, "{summary:?}");
        let badge = summary.badge().unwrap();
        assert!(
            badge.contains("Hematoxylin ×1.50")
                && badge.contains("not applied")
                && !badge.contains("Isolated"),
            "{badge}"
        );
        assert!(
            summary
                .provenance()
                .contains("Not applied: isolating stain 3 (the stain model has 2 stains)."),
            "{}",
            summary.provenance()
        );

        // Nothing valid left: no stain stage, and no claim of one.
        let only_invalid = ColorSettings {
            isolated: Some(5),
            ..ColorSettings::default()
        };
        let pipeline = only_invalid.pipeline(Some(&estimate()));
        assert!(pipeline.is_identity());
        assert!(pipeline.summary().stain_model.is_none());
        assert!(!pipeline.summary().alters_stain_colors());
        assert!(
            pipeline
                .summary()
                .provenance()
                .starts_with("Colours as scanned (no processing). Not applied:")
        );

        // A normalisation target with another number of stains: the slide
        // is not normalised and the summary says so, but the channel
        // adjustments still apply (with the H&E preset).
        let mut norm = ColorSettings::from_legacy(
            crate::StainNormalization::Macenko,
            1.3,
            true,
            1.0,
            true,
            0,
            ToneAdjustments::default(),
            0.0,
        );
        norm.target = Some(NormalizationTarget {
            basis: StainPreset::HematoxylinEosinDab.basis(),
            max_concentrations: vec![1.0; 3],
        });
        let pipeline = norm.pipeline(Some(&estimate()));
        let summary = pipeline.summary();
        assert_eq!(summary.normalization, None);
        assert!(
            summary.not_applied[0].starts_with("Macenko stain normalisation ("),
            "{summary:?}"
        );
        let badge = summary.badge().unwrap();
        assert!(
            !badge.contains("Colour-normalised") && badge.contains("Hematoxylin ×1.30"),
            "{badge}"
        );
        assert!(!pipeline.stain_transform().is_identity());
        assert_eq!(
            pipeline.stain_spec().unwrap().source,
            StainPreset::HematoxylinEosin.basis()
        );

        // Adjusting a stain the model does not have changes nothing.
        let extra = ColorSettings {
            channels: vec![
                StainChannel::default(),
                StainChannel::default(),
                StainChannel {
                    intensity: 2.0,
                    visible: true,
                },
            ],
            ..ColorSettings::default()
        };
        let pipeline = extra.pipeline(None);
        assert!(pipeline.is_identity());
        assert!(pipeline.summary().stain_adjustments.is_empty());
        assert_eq!(pipeline.summary().not_applied.len(), 1);
    }

    #[test]
    fn explicit_pipelines_describe_their_stain_stage() {
        let spec = StainTransformSpec::deconvolution(StainPreset::HematoxylinDab.basis())
            .with_channels(&[
                StainChannel::default(),
                StainChannel {
                    intensity: 0.0,
                    visible: false,
                },
            ]);
        let pipeline =
            ProcessingPipeline::new(Some(&spec), ToneAdjustments::default(), 0.0).unwrap();
        let summary = pipeline.summary();
        assert_eq!(summary.stain_adjustments, vec!["DAB hidden".to_string()]);
        assert!(
            summary
                .stain_model
                .as_deref()
                .unwrap()
                .starts_with("custom (")
        );
        assert!(summary.alters_stain_colors());
    }

    #[test]
    fn uniform_layout_is_144_bytes() {
        assert_eq!(
            std::mem::size_of::<GpuColorUniform>(),
            GpuColorUniform::SIZE
        );
        let u = ColorSettings::from_legacy(
            crate::StainNormalization::None,
            1.5,
            true,
            1.0,
            true,
            0,
            ToneAdjustments::new(2.0, 0.1, 1.1),
            0.3,
        )
        .pipeline(None)
        .gpu_uniform();
        let a = u.to_array();
        assert_eq!(&a[0..4], &[0.5, 0.1, 1.1, 0.3]);
        assert_eq!(a[4], 1.0);
    }

    #[test]
    fn legacy_uniform_matches_the_new_pipeline_without_normalisation() {
        #[allow(deprecated)]
        let deconv = crate::stain::build_deconv_params(1.4, true, 0.8, true, 0, None);
        let legacy =
            GpuColorUniform::from_legacy(1.2, 0.0, 1.0, 0.0, false, [0.0; 4], [0.0; 4], &deconv);
        let modern = ColorSettings::from_legacy(
            crate::StainNormalization::None,
            1.4,
            true,
            0.8,
            true,
            0,
            ToneAdjustments::new(1.2, 0.0, 1.0),
            0.0,
        )
        .pipeline(None)
        .gpu_uniform();
        let (a, b) = (legacy.to_array(), modern.to_array());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{a:?}\n{b:?}");
        }
    }

    #[test]
    fn canonical_wgsl_validates() {
        let color_module = format!(
            "{}\n@group(0) @binding(0) var<uniform> color_params: EovColorParams;\n{}\n\
             @fragment fn fs_test(@location(0) rgb: vec3<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(eov_color(rgb), 1.0); }}",
            wgsl::COLOR_PARAMS_STRUCT,
            wgsl::COLOR_FUNCTIONS
        );
        for (name, source) in [
            ("color", color_module.as_str()),
            ("sharpen", wgsl::SHARPEN_POST_PASS),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
