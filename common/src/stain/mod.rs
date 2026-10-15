//! Stain science for brightfield histology: stain vectors and presets,
//! slide-level stain estimation (Macenko, Vahadane), and a fused
//! normalisation + colour-deconvolution transform.
//!
//! # Typical use
//!
//! ```no_run
//! use eov_common::stain::{
//!     NormalizationTarget, StainEstimateOptions, StainFitMethod, StainTransformSpec,
//!     estimate_slide_stain,
//! };
//! # fn main() -> eov_common::Result<()> {
//! let wsi = eov_common::WsiFile::open("slide.svs")?;
//! // Once per slide, on a worker thread; cache the result per slide + method.
//! let options = StainEstimateOptions::default();
//! let estimate = estimate_slide_stain(&wsi, StainFitMethod::Macenko, &options)?;
//! let transform = StainTransformSpec::normalization(
//!     &estimate,
//!     &NormalizationTarget::default(),
//!     &options.checks,
//! )
//! .expect("H&E estimate and H&E target")
//! .compile()
//! .expect("valid transform");
//! let mut rgba = wsi.read_region(0, 0, 0, 256, 256)?;
//! transform.apply_rgba(&mut rgba);
//! # Ok(()) }
//! ```
//!
//! For display and export, combine the stain transform with tone
//! adjustments and sharpening in the canonical order with
//! [`crate::processing::ProcessingPipeline`].
//!
//! # Model summary
//!
//! * Optical density `OD = ln(I₀) − ln(max(v, 1))` per channel, with the
//!   background `I₀` estimated from the glass ([`od`]).
//! * Tissue mask: a pixel is used for fitting only if the OD of every
//!   channel is ≥ β (0.15), it is not near-black, and green is not its
//!   brightest channel ([`PixelFilter`]); a minimum tissue fraction and
//!   pixel count are required ([`StainEstimateOptions`]).
//! * Fits are rejected when the stain vectors are too close together, the
//!   matrix is ill-conditioned or a vector is implausibly far from the
//!   reference ([`FitChecks`]); the reference vectors are then used and the
//!   reason recorded ([`FitFallback`]).
//! * Normalisation, deconvolution intensities and channel visibility are
//!   applied together in concentration space while the out-of-plane residual
//!   is kept ([`transform`]): with every control at its neutral value the
//!   transform is exactly the identity.

mod basis;
mod estimate;
mod legacy;
mod linalg;
pub mod od;
mod slide;
pub mod transform;

pub use basis::{
    Stain, StainBasis, StainBasisError, StainLabelKind, StainPreset, classify_stain_label,
    stain_model_warning, suggested_preset,
};
pub use estimate::{
    EstimateSource, FitChecks, FitFallback, PixelFilter, STAIN_ESTIMATE_FORMAT_VERSION,
    StainEstimate, StainEstimateOptions, StainFitMethod, estimate_stain_from_rgba,
};
#[allow(deprecated)]
pub use legacy::{
    ColorDeconvParams, DEFAULT_DECONV_MATRIX, StainNormParams, apply_color_deconvolution,
    apply_normalization_to_buffer, apply_stain_params_to_buffer, build_deconv_params,
    compute_cpu_stain_params, normalize_buffer,
};
pub use od::{BackgroundEstimate, BackgroundOptions, OdLut};
pub use slide::{
    Level0Rect, StainCacheKey, StainEstimateCache, estimate_region_stain, estimate_slide_stain,
    file_identity, global_stain_cache,
};
pub use transform::{
    MACENKO_REFERENCE_MAX_C, Normalization, NormalizationTarget, StainChannel, StainGpuParams,
    StainTransform, StainTransformError, StainTransformSpec,
};
