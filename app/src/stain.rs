//! Stain normalization and color deconvolution - re-exports from common crate.

pub use eov_common::stain::{
    ColorDeconvParams, StainNormParams, apply_color_deconvolution, apply_stain_params_to_buffer,
    build_deconv_params, compute_cpu_stain_params,
};
