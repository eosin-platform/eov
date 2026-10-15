//! Image export: banded, cancellable rendering of a view or region at any
//! resolution, with the canonical processing pipeline, overlays (including a
//! calibrated scale bar and burned-in provenance) and streaming PNG / JPEG /
//! TIFF output with resolution metadata.
//!
//! # Typical use (on a worker thread)
//!
//! ```no_run
//! use eov_common::export::{ExportFormat, ExportRequest, ExportSource, ImageMetadata, export_to_file};
//! use eov_common::job::JobControl;
//! # fn demo(tile_manager: &eov_common::TileManager, viewport: eov_common::Viewport) -> Result<(), eov_common::export::ExportError> {
//! let mut request = ExportRequest::new(viewport, 300.0);
//! // request.processing = settings.pipeline(Some(&slide_estimate));
//! // request.overlays = ...;   // in request.export_viewport() coordinates
//! let props = tile_manager.wsi().properties();
//! let metadata = ImageMetadata::for_export(&request, 300.0, props.mpp_x, props.mpp_y,
//!     Some(request.processing.summary().provenance()));
//! let format = ExportFormat::Jpeg { quality: 92 };
//! let path = format.ensure_extension(std::path::Path::new("figure"));
//! let control = JobControl::new(); // share via Arc; poll progress(), call cancel()
//! let report = export_to_file(
//!     &ExportSource { tile_manager, tile_cache: None },
//!     &request, &path, format, metadata, &control,
//! )?;
//! if let Some(warning) = report.shortfall_message() { eprintln!("{warning}"); }
//! # Ok(()) }
//! ```

mod encode;
mod jpeg;
mod render;
pub(crate) mod tiff;

pub use encode::{
    ColorSpace, ExportFormat, FileSink, ImageMetadata, export_to_file, software_name,
};
pub use jpeg::{JpegOptions, JpegTileEncoder, JpegWriter, Subsampling};
pub use render::{
    BandSink, ExportError, ExportReport, ExportRequest, ExportSource, ImageSink, MAX_EXPORT_SIDE,
    render_export_bands, render_export_image, render_export_preview,
};
pub use tiff::{TiffCompression, TiffOptions, TiffWriter};

use crate::job::JobControl;
use crate::render::FilteringMode;
use crate::stain;
use crate::{RgbaImageData, StainNormalization, TileCache, TileManager, Viewport};

/// Export settings in the pre-1.0 layout (as edited by the export dialog).
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Output DPI (96 = screen resolution).
    pub dpi: u32,
    /// Filtering mode for tile resampling.
    pub filtering_mode: FilteringMode,
    /// Stain normalization method.
    pub stain_normalization: StainNormalization,
    /// Sharpening amount (0.0 = off).
    pub sharpness: f32,
    /// Gamma correction (1.0 = identity).
    pub gamma: f32,
    /// Brightness offset (0.0 = identity).
    pub brightness: f32,
    /// Contrast multiplier (1.0 = identity).
    pub contrast: f32,
    /// Background colour (RGBA; alpha is ignored, exports are opaque).
    pub background_rgba: [u8; 4],
    /// Color deconvolution: hematoxylin intensity (1.0 = identity).
    pub deconv_h_intensity: f32,
    /// Color deconvolution: hematoxylin channel visible.
    pub deconv_h_visible: bool,
    /// Color deconvolution: eosin intensity (1.0 = identity).
    pub deconv_e_intensity: f32,
    /// Color deconvolution: eosin channel visible.
    pub deconv_e_visible: bool,
    /// Color deconvolution: isolated channel (0=none, 1=hematoxylin, 2=eosin).
    pub deconv_isolated_channel: u8,
}

impl ExportSettings {
    /// The colour-processing part of these settings.
    pub fn color_settings(&self) -> crate::processing::ColorSettings {
        crate::processing::ColorSettings::from_legacy(
            self.stain_normalization,
            self.deconv_h_intensity,
            self.deconv_h_visible,
            self.deconv_e_intensity,
            self.deconv_e_visible,
            self.deconv_isolated_channel,
            crate::postprocess::ToneAdjustments::new(self.gamma, self.brightness, self.contrast),
            self.sharpness,
        )
    }

    /// An [`ExportRequest`] for `viewport` with these settings. Stain
    /// normalisation uses the slide-level fit from the process-wide cache
    /// (computed on first use), so the preview and the final export agree.
    pub fn request(&self, tile_manager: &TileManager, viewport: &Viewport) -> ExportRequest {
        let color = self.color_settings();
        let wsi = tile_manager.wsi();
        let estimate = color.required_estimate().and_then(|method| {
            let options = stain::StainEstimateOptions::default();
            let key = stain::StainCacheKey::for_file(&wsi.properties().path, method, &options);
            stain::global_stain_cache()
                .get_or_try_insert_with(key, || stain::estimate_slide_stain(wsi, method, &options))
                .ok()
        });
        let mut request = ExportRequest::new(viewport.clone(), self.dpi.max(1) as f64);
        request.filtering = self.filtering_mode;
        request.processing = color.pipeline(estimate.as_deref());
        request.background = [
            self.background_rgba[0],
            self.background_rgba[1],
            self.background_rgba[2],
        ];
        request
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            dpi: 96,
            filtering_mode: FilteringMode::Trilinear,
            stain_normalization: StainNormalization::None,
            sharpness: 0.0,
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
            background_rgba: [255, 255, 255, 255],
            deconv_h_intensity: 1.0,
            deconv_h_visible: true,
            deconv_e_intensity: 1.0,
            deconv_e_visible: true,
            deconv_isolated_channel: 0,
        }
    }
}

/// The viewport used for export rendering at the given DPI (maps level-0
/// coordinates to export pixels for overlays).
pub fn export_viewport(viewport: &Viewport, dpi: u32) -> Viewport {
    ExportRequest::new(viewport.clone(), dpi.max(1) as f64).export_viewport()
}

/// Renders the viewport at the requested DPI into memory.
///
/// Now uncapped (every tile is loaded) and processed in the canonical order
/// with one slide-level stain fit, but still synchronous and holding the
/// whole image in memory.
#[deprecated(
    since = "1.0.0",
    note = "use `ExportSettings::request` with `export::render_export_image`, `render_export_preview` \
            or `export_to_file` on a worker thread (progress, cancellation, bounded memory)"
)]
pub fn render_export(
    tile_manager: &TileManager,
    tile_cache: &TileCache,
    viewport: &Viewport,
    settings: &ExportSettings,
) -> Option<RgbaImageData> {
    let request = settings.request(tile_manager, viewport);
    let source = ExportSource {
        tile_manager,
        tile_cache: Some(tile_cache),
    };
    match render_export_image(&source, &request, &JobControl::new()) {
        Ok((image, report)) => {
            if let Some(warning) = report.shortfall_message() {
                tracing::warn!("export: {warning}");
            }
            Some(image)
        }
        Err(err) => {
            tracing::warn!("export failed: {err}");
            None
        }
    }
}

#[cfg(test)]
mod tests;
