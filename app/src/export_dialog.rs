//! Image export and viewport copies (EXP-1 to EXP-5).
//!
//! The export dialog edits [`SlintExportSettings`] for the pane and slide it
//! was opened for. Previews (longest side [`PREVIEW_MAX_SIDE`]) and exports
//! render on worker threads from the slide's stored pyramid levels with the
//! viewer's colour processing. Exports stream bands to the encoder, run as
//! status-bar jobs with progress and cancel, and record DPI and µm/px in the
//! file. "Copy viewport image" renders the same way at screen resolution,
//! with every overlay as on screen.
//!
//! Stain normalisation uses the viewer's fit of the slide
//! ([`crate::color::slide_fit`], asked for when the export is set up):
//! previews show the slide un-normalised while it runs, exports and copies
//! wait for it on their thread, and a fit that failed is not tried again.
//!
//! A slide with a usable ICC profile is exported colour-managed (converted
//! to sRGB, the file says so) like the viewer shows it, unless the dialog's
//! "Convert to sRGB" is off: then its colours are exported as scanned and
//! its profile is embedded in the file untouched.
//!
//! Overlays, bottom to top: overlay layers (the viewport's rasteriser) and
//! their labels, then the host's ROI and measurements in the dialog's
//! styles, the scale bar and the colour-processing note.

use crate::color::SlideFit;
use crate::host_ui::with_host;
use crate::interaction::{Calibration, HostMeasurement, measure};
use crate::overlay::Owner;
use crate::overlay::labels::{self, LabelAnchor};
use crate::overlay::raster::{self, FrameScene, ViewTransform};
use crate::overlay::store::Layer;
use crate::state::{AppState, IsolatedChannel, PaneId};
use crate::{
    AppWindow, CapStyle as SlintCapStyle, ExportFilteringMode as SlintExportFilteringMode,
    ExportFormat as SlintExportFormat, ExportPreviewInfo as SlintExportPreviewInfo,
    ExportSettings as SlintExportSettings, IsolatedChannel as SlintIsolatedChannel,
    StainNormalization as SlintStainNormalization, StrokeStyle as SlintStrokeStyle,
};
use eov_common::color_management::DisplayTransform;
use eov_common::export::ColorSpace;
use eov_common::export::{
    BandSink, ExportError, ExportFormat, ExportReport, ExportRequest, ExportSource, FileSink,
    ImageMetadata, ImageSink, TiffCompression, render_export_bands,
};
use eov_common::job::JobControl;
use eov_common::overlay::{
    Canvas, CapStyle, Corner, FontArc, LabelStyle, OverlayColor, OverlayItem, ScaleBarStyle,
    StrokeStyle, embedded_font,
};
use eov_common::stain::StainFitMethod;
use eov_common::{
    ExportSettings, FilteringMode, MeasurementUnit, RgbaImageData, StainNormalization, TileCache,
    TileManager, Viewport, WsiFile,
};
use eov_plugin_api::protocol::{NotifyLevel, Rect};
use parking_lot::RwLock;
use rfd::FileDialog;
use slint::{ComponentHandle, SharedString, Timer, TimerMode};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Longest side of export previews, in pixels.
const PREVIEW_MAX_SIDE: u32 = 512;

/// How often running exports update their status-bar progress.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Default)]
struct Dialog {
    /// Pane and slide the open dialog exports (EXP-5).
    target: Option<(PaneId, i32)>,
    /// Settings last used per slide; colour settings are re-read from the
    /// pane's HUD whenever the dialog opens (EXP-5).
    remembered: HashMap<i32, SlintExportSettings>,
    /// Preview requests so far, and whether a preview worker is running.
    previews_wanted: u64,
    preview_running: bool,
    /// Progress timers of running exports, by job id.
    progress_timers: HashMap<String, Timer>,
    /// The slide's stain fit (or colour profile) finished since the preview
    /// was rendered.
    preview_stale: bool,
}

thread_local! {
    static DIALOG: RefCell<Dialog> = RefCell::new(Dialog::default());
}

/// Everything an export needs from the UI thread, captured at once.
#[derive(Clone)]
struct Capture {
    file_id: i32,
    /// Stored levels only: exports never draw synthetic ones.
    wsi: WsiFile,
    file_name: String,
    viewport: Viewport,
    tile_cache: Arc<TileCache>,
    roi: Option<Rect>,
    measurements: Vec<HostMeasurement>,
    calibration: Calibration,
    unit: MeasurementUnit,
    /// Overlay layers drawn with the viewport's rasteriser.
    layers: Vec<Arc<Layer>>,
    /// The slide's stain fit for the settings' normalisation, as the viewer
    /// has it (see [`build_request`]).
    fit: Option<(StainFitMethod, SlideFit)>,
    /// The slide's colour profile, converted when the settings say so.
    color_profile: Option<Arc<DisplayTransform>>,
    /// The viewer's pixel view: exports above 1:1 show pixels as squares.
    pixel_view: bool,
}

/// Captures the slide shown in `pane` for an export with `settings`. With
/// `expected_file`, returns `None` when the pane shows another slide now.
/// `host_layers` includes the host's ROI and measurement layers as drawn on
/// screen; otherwise the export draws those from the dialog's settings.
/// Heat maps are always included.
fn capture(
    state: &AppState,
    tile_cache: &Arc<TileCache>,
    pane: PaneId,
    expected_file: Option<i32>,
    host_layers: bool,
    settings: &SlintExportSettings,
) -> Option<Capture> {
    let file_id = state.active_file_id_for_pane(pane)?;
    if expected_file.is_some_and(|expected| expected != file_id) {
        return None;
    }
    let file = state.get_file(file_id)?;
    let pane_state = file.pane_state(pane)?;
    let pane_uid = state.pane_uid(pane);
    let layers = with_host(|host| host.store.file_layers(i64::from(file_id)).layers)
        .unwrap_or_default()
        .into_iter()
        .filter(|layer| {
            host_layers
                || layer.key.owner != Owner::Host
                || crate::host_ui::is_heatmap_layer(&layer.key.id)
        })
        .filter(|layer| pane_uid.is_none_or(|uid| layer.shown_in(uid)))
        .collect();
    // Starts the viewer's fit of the slide if nobody has.
    let fit = legacy_settings(settings)
        .color_settings()
        .required_estimate()
        .map(|method| (method, crate::color::slide_fit(file_id, &file.wsi, method)));
    let color_profile = match crate::color::slide_profile(file_id, &file.wsi) {
        crate::color::SlideProfile::Ready(transform) => Some(transform),
        _ => None,
    };
    Some(Capture {
        file_id,
        wsi: file.wsi.clone(),
        file_name: file.filename.clone(),
        viewport: pane_state.viewport.viewport.clone(),
        tile_cache: Arc::clone(tile_cache),
        roi: file.roi,
        measurements: file.measurements.clone(),
        calibration: crate::host_ui::calibration(file),
        unit: pane_state.hud.measurement_unit,
        layers,
        fit,
        color_profile,
        pixel_view: state.pixel_view,
    })
}

/// What the exported pixels' colours mean: sRGB when converted, the
/// slide's profile (read here, embedded untouched) when not (worker
/// thread).
fn export_color_space(capture: &Capture, request: &ExportRequest) -> ColorSpace {
    if request.processing.color_management().is_some() {
        return ColorSpace::Srgb;
    }
    match capture.wsi.icc_profile() {
        Ok(Some(profile)) => ColorSpace::Profile(Arc::new(profile)),
        _ => ColorSpace::Unspecified,
    }
}

/// What the export does with the slide's colour profile, if it has one.
fn color_management_note(capture: &Capture, request: &ExportRequest) -> Option<String> {
    let profile = capture.color_profile.as_ref()?;
    Some(if request.processing.color_management().is_some() {
        format!(
            "Colour-managed: \"{}\" converted to sRGB",
            profile.description()
        )
    } else {
        "Colours as scanned; the slide's colour profile is embedded".to_string()
    })
}

/// Why the stain normalisation the settings ask for is not applied, if it
/// cannot be.
fn fit_note(capture: &Capture) -> Option<String> {
    match &capture.fit {
        Some((_, SlideFit::Failed(error))) => Some(format!(
            "Stain normalisation is unavailable for this slide: {error}"
        )),
        _ => None,
    }
}

// -- settings ---------------------------------------------------------------

/// The dialog's settings in the export renderer's terms.
fn legacy_settings(s: &SlintExportSettings) -> ExportSettings {
    let filtering_mode = match s.filtering_mode {
        SlintExportFilteringMode::Bilinear => FilteringMode::Bilinear,
        SlintExportFilteringMode::Trilinear => FilteringMode::Trilinear,
        SlintExportFilteringMode::Lanczos => FilteringMode::Lanczos3,
    };
    let stain_normalization = match s.stain_normalization {
        SlintStainNormalization::None => StainNormalization::None,
        SlintStainNormalization::Macenko => StainNormalization::Macenko,
        SlintStainNormalization::Vahadane => StainNormalization::Vahadane,
    };
    let deconv_isolated_channel = match s.deconv_isolated_channel {
        SlintIsolatedChannel::Hematoxylin => 1,
        SlintIsolatedChannel::Eosin => 2,
        _ => 0,
    };
    ExportSettings {
        dpi: s.dpi.max(1) as u32,
        filtering_mode,
        stain_normalization,
        sharpness: s.sharpness,
        gamma: s.gamma,
        brightness: s.brightness,
        contrast: s.contrast,
        background_rgba: [255, 255, 255, 255],
        deconv_h_intensity: s.deconv_hematoxylin_intensity,
        deconv_h_visible: s.deconv_hematoxylin_visible,
        deconv_e_intensity: s.deconv_eosin_intensity,
        deconv_e_visible: s.deconv_eosin_visible,
        deconv_isolated_channel,
    }
}

fn export_format(s: &SlintExportSettings) -> ExportFormat {
    match s.format {
        SlintExportFormat::Png => ExportFormat::Png,
        SlintExportFormat::Jpeg => ExportFormat::Jpeg {
            quality: s.jpeg_quality.clamp(1, 100) as u8,
        },
        SlintExportFormat::Tiff => ExportFormat::Tiff {
            compression: TiffCompression::Deflate,
        },
    }
}

/// Copies the pane's colour adjustments, and the slide's colour
/// management, into `settings`.
fn apply_hud(settings: &mut SlintExportSettings, state: &AppState, pane: PaneId) {
    let file = state
        .active_file_id_for_pane(pane)
        .and_then(|id| state.get_file(id));
    let profile = file.is_some_and(|file| {
        matches!(
            crate::color::slide_profile(file.id, &file.wsi),
            crate::color::SlideProfile::Ready(_)
        )
    });
    settings.has_color_profile = profile;
    settings.color_management = profile && file.is_some_and(|file| file.color_management);
    let hud = file
        .and_then(|file| file.pane_state(pane))
        .map(|pane_state| pane_state.hud)
        .unwrap_or_default();
    settings.sharpness = hud.sharpness;
    settings.gamma = hud.gamma;
    settings.brightness = hud.brightness;
    settings.contrast = hud.contrast;
    settings.stain_normalization = match hud.stain_normalization {
        StainNormalization::None => SlintStainNormalization::None,
        StainNormalization::Macenko => SlintStainNormalization::Macenko,
        StainNormalization::Vahadane => SlintStainNormalization::Vahadane,
    };
    settings.deconv_hematoxylin_intensity = hud.deconv_hematoxylin_intensity;
    settings.deconv_hematoxylin_visible = hud.deconv_hematoxylin_visible;
    settings.deconv_eosin_intensity = hud.deconv_eosin_intensity;
    settings.deconv_eosin_visible = hud.deconv_eosin_visible;
    settings.deconv_isolated_channel = match hud.deconv_isolated_channel {
        IsolatedChannel::Hematoxylin => SlintIsolatedChannel::Hematoxylin,
        IsolatedChannel::Eosin => SlintIsolatedChannel::Eosin,
        IsolatedChannel::None => SlintIsolatedChannel::None,
    };
}

/// Refreshes what the slide has (measurements, ROI) and drops overlays
/// that no longer exist.
fn apply_slide(settings: &mut SlintExportSettings, state: &AppState, pane: PaneId) {
    let file = state
        .active_file_id_for_pane(pane)
        .and_then(|id| state.get_file(id));
    settings.has_measurement = file.is_some_and(|f| !f.measurements.is_empty());
    settings.has_roi = file.is_some_and(|f| f.roi.is_some());
    if !settings.has_measurement {
        settings.show_measurement = false;
    }
    if !settings.has_roi {
        settings.show_roi_outline = false;
        settings.roi_outside_overlay = false;
    }
}

/// Default settings for the slide in `pane`.
pub(crate) fn default_settings(state: &AppState, pane: PaneId) -> SlintExportSettings {
    let file = state
        .active_file_id_for_pane(pane)
        .and_then(|id| state.get_file(id));
    let calibrated = file.is_some_and(|f| crate::host_ui::calibration(f).average().is_some());
    let mut settings = SlintExportSettings {
        dpi: 150,
        filtering_mode: SlintExportFilteringMode::Trilinear,
        show_measurement: true,
        measurement_color: slint::Color::from_argb_u8(255, 46, 204, 113),
        measurement_opacity: 100.0,
        measurement_thickness: 2.0,
        measurement_stroke_style: SlintStrokeStyle::Solid,
        measurement_cap_style: SlintCapStyle::Round,
        measurement_dash_length: 8.0,
        measurement_dash_gap: 4.0,
        measurement_dot_spacing: 4.0,
        measurement_font_size: 12.0,
        show_roi_outline: true,
        roi_color: slint::Color::from_argb_u8(255, 241, 196, 15),
        roi_opacity: 100.0,
        roi_thickness: 2.0,
        roi_stroke_style: SlintStrokeStyle::Solid,
        roi_cap_style: SlintCapStyle::Round,
        roi_dash_length: 8.0,
        roi_dash_gap: 4.0,
        roi_dot_spacing: 4.0,
        roi_outside_overlay: false,
        roi_outside_color: slint::Color::from_argb_u8(255, 0, 0, 0),
        roi_outside_opacity: 50.0,
        show_scale_bar: calibrated,
        format: SlintExportFormat::Png,
        jpeg_quality: 92,
        ..Default::default()
    };
    apply_hud(&mut settings, state, pane);
    // Label processed colours by default, so a figure says what was done.
    settings.show_processing_note = legacy_settings(&settings).color_settings().is_modified();
    apply_slide(&mut settings, state, pane);
    settings
}

// -- overlays ---------------------------------------------------------------

fn overlay_color(color: slint::Color, opacity_pct: f32) -> OverlayColor {
    let alpha = (f32::from(color.alpha()) * opacity_pct / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    OverlayColor::new(color.red(), color.green(), color.blue(), alpha)
}

fn stroke_style(style: SlintStrokeStyle, dash: f32, gap: f32, dot: f32, k: f32) -> StrokeStyle {
    match style {
        SlintStrokeStyle::Solid => StrokeStyle::Solid,
        SlintStrokeStyle::Dashed => StrokeStyle::Dashed {
            length: dash * k,
            gap: gap * k,
        },
        SlintStrokeStyle::Dotted => StrokeStyle::Dotted { spacing: dot * k },
    }
}

fn cap_style(cap: SlintCapStyle) -> CapStyle {
    match cap {
        SlintCapStyle::Round => CapStyle::Round,
        SlintCapStyle::Square => CapStyle::Square,
        SlintCapStyle::Flat => CapStyle::Flat,
    }
}

/// Overlays and notes for one rendering of `request`.
struct Prepared {
    scene: FrameScene,
    view: ViewTransform,
    /// Drawn after the rasterised layers, in order.
    items: Vec<OverlayItem>,
    /// Things the user should know about the result.
    notes: Vec<String>,
}

/// Maps the request's view to output pixels for the rasteriser: logical
/// pixels are the viewport's, `scale` is output pixels per logical pixel.
fn export_view(request: &ExportRequest, calibration: &Calibration) -> ViewTransform {
    let viewport = &request.viewport;
    let bounds = viewport.bounds();
    ViewTransform {
        left: bounds.left,
        top: bounds.top,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        scale: request.scale,
        mpp: calibration.average(),
        orientation: viewport.orientation,
    }
}

/// Labels of the rasterised layers as glyph items (on screen Slint draws
/// them).
fn layer_label_items(view: &ViewTransform, scene: &FrameScene) -> Vec<OverlayItem> {
    let k = view.scale as f32;
    labels::collect_labels(view, scene)
        .into_iter()
        .map(|label| {
            let color =
                OverlayColor::new(label.color.0, label.color.1, label.color.2, label.color.3);
            let size = label.font_size * k;
            let (x, y) = (label.x * k, label.y * k);
            match label.anchor {
                LabelAnchor::Above if label.backdrop => OverlayItem::Label {
                    anchor: [x, y],
                    text: label.text,
                    size,
                    scale: k,
                    style: LabelStyle {
                        border: None,
                        text: color,
                        ..LabelStyle::default()
                    },
                },
                LabelAnchor::Above => OverlayItem::Text {
                    origin: [x, y - size * 1.2],
                    text: label.text,
                    color,
                    size,
                },
                LabelAnchor::Right => OverlayItem::Text {
                    origin: [x, y - size * 0.6],
                    text: label.text,
                    color,
                    size,
                },
                LabelAnchor::TopLeft => OverlayItem::Text {
                    origin: [x, y],
                    text: label.text,
                    color,
                    size,
                },
            }
        })
        .collect()
}

/// The host's ROI and measurements in the dialog's styles, the scale bar
/// and the processing note, in output pixels of `request`.
fn host_items(
    capture: &Capture,
    settings: &SlintExportSettings,
    request: &ExportRequest,
    items: &mut Vec<OverlayItem>,
    notes: &mut Vec<String>,
) {
    let viewport = request.export_viewport();
    // Output pixels per screen pixel: overlay sizes follow the DPI.
    let k = request.scale as f32;
    let to_px = |x: f64, y: f64| {
        let p = viewport.image_to_screen(x, y);
        [p.x as f32, p.y as f32]
    };

    if let Some(roi) = capture.roi {
        // Opposite corners; a turned view may swap them.
        let a = to_px(roi.x, roi.y);
        let b = to_px(roi.x + roi.width, roi.y + roi.height);
        let rect = [
            a[0].min(b[0]),
            a[1].min(b[1]),
            (a[0] - b[0]).abs(),
            (a[1] - b[1]).abs(),
        ];
        if settings.roi_outside_overlay {
            items.push(OverlayItem::FillOutsideRect {
                rect,
                color: overlay_color(settings.roi_outside_color, settings.roi_outside_opacity),
            });
        }
        if settings.show_roi_outline {
            items.push(OverlayItem::RectOutline {
                rect,
                color: overlay_color(settings.roi_color, settings.roi_opacity),
                thickness: settings.roi_thickness * k,
                stroke: stroke_style(
                    settings.roi_stroke_style,
                    settings.roi_dash_length,
                    settings.roi_dash_gap,
                    settings.roi_dot_spacing,
                    k,
                ),
                cap: cap_style(settings.roi_cap_style),
            });
        }
    }

    if settings.show_measurement {
        let color = overlay_color(settings.measurement_color, settings.measurement_opacity);
        let thickness = settings.measurement_thickness * k;
        let stroke = stroke_style(
            settings.measurement_stroke_style,
            settings.measurement_dash_length,
            settings.measurement_dash_gap,
            settings.measurement_dot_spacing,
            k,
        );
        let cap = cap_style(settings.measurement_cap_style);
        let label_style = LabelStyle::default().with_border(color);
        for m in &capture.measurements {
            let (outline, closed) = crate::overlay::geometry::outline(&m.shape);
            let points: Vec<[f32; 2]> = outline.iter().map(|p| to_px(p[0], p[1])).collect();
            if !closed {
                // Endpoint dots with a white ring, as on screen.
                let radius = (thickness + 1.0) * 0.5 + k;
                let ring = OverlayColor::new(255, 255, 255, color.a);
                for center in [points.first(), points.last()].into_iter().flatten() {
                    items.push(OverlayItem::FilledCircle {
                        center: *center,
                        radius: radius + 1.5 * k,
                        color: ring,
                    });
                }
            }
            items.push(OverlayItem::Polyline {
                points: points.clone(),
                closed,
                color,
                thickness,
                stroke,
                cap,
            });
            if !closed {
                let radius = (thickness + 1.0) * 0.5 + k;
                for center in [points.first(), points.last()].into_iter().flatten() {
                    items.push(OverlayItem::FilledCircle {
                        center: *center,
                        radius,
                        color,
                    });
                }
            }
            // Calibrated per axis, in the pane's unit (EXP-5).
            let value = measure::measure(m.mode, &m.shape, &capture.calibration);
            let text = measure::label(m.mode, &value, capture.unit);
            let (anchor, _) = labels::label_anchor(&m.shape);
            items.push(OverlayItem::Label {
                anchor: to_px(anchor[0], anchor[1]),
                text,
                size: settings.measurement_font_size * k,
                scale: k,
                style: label_style,
            });
        }
    }

    if settings.show_scale_bar {
        let (um_per_px, _) =
            request.output_pixel_size_um(capture.calibration.mpp_x, capture.calibration.mpp_y);
        match um_per_px {
            Some(um_per_px) => items.push(OverlayItem::ScaleBar {
                um_per_px,
                style: ScaleBarStyle {
                    unit: capture.unit,
                    ..ScaleBarStyle::default()
                }
                .scaled(k),
            }),
            None => notes.push("No scale bar: the slide has no pixel size.".to_string()),
        }
    }

    let summary = request.processing.summary();
    if settings.show_processing_note {
        let text = summary.badge().unwrap_or_else(|| summary.provenance());
        items.push(OverlayItem::TextBlock {
            corner: Corner::BottomLeft,
            lines: vec![text],
            size: 12.0 * k,
            scale: k,
            margin: 12.0 * k,
            style: LabelStyle {
                border: None,
                ..LabelStyle::default()
            },
        });
    }
    if let Some(badge) = summary.badge() {
        notes.push(badge);
    }
    for item in &summary.not_applied {
        notes.push(format!("Not applied: {item}"));
    }
}

/// Builds the request for `capture` at `dpi` with `settings`' processing.
/// A stain fit still running is waited for when `wait_for_fit` says how to
/// cancel (exports, copies); previews (`None`) show the slide un-normalised
/// meanwhile, and say so.
fn build_request(
    capture: &Capture,
    settings: &SlintExportSettings,
    dpi: f64,
    wait_for_fit: Option<&dyn Fn() -> bool>,
) -> ExportRequest {
    let legacy = legacy_settings(settings);
    let mut color = legacy.color_settings();
    let fit = match &capture.fit {
        Some((method, SlideFit::Pending)) => match wait_for_fit {
            Some(cancelled) => crate::color::await_slide_fit(&capture.wsi, *method, cancelled),
            None => SlideFit::Pending,
        },
        Some((_, fit)) => fit.clone(),
        None => SlideFit::Pending,
    };
    let estimate = match fit {
        SlideFit::Ready(estimate) => Some(estimate),
        SlideFit::Failed(_) => {
            // As the viewer shows it: not normalised (see `fit_note`).
            color.normalization = None;
            None
        }
        SlideFit::Pending => None,
    };
    // Fractional DPI (screen scale factors) is kept exactly.
    let mut request = ExportRequest::new(capture.viewport.clone(), dpi);
    request.filtering = legacy.filtering_mode;
    request.pixel_view = capture.pixel_view;
    let managed = settings
        .color_management
        .then(|| capture.color_profile.clone())
        .flatten();
    request.processing = color
        .pipeline(estimate.as_deref())
        .with_color_management(managed);
    request.background = [
        legacy.background_rgba[0],
        legacy.background_rgba[1],
        legacy.background_rgba[2],
    ];
    request
}

/// Overlays for one rendering; `dialog` draws the host's ROI and
/// measurements from the dialog's settings (otherwise they come with the
/// rasterised layers).
fn prepare(
    capture: &Capture,
    settings: &SlintExportSettings,
    request: &ExportRequest,
    dialog: bool,
) -> Prepared {
    let view = export_view(request, &capture.calibration);
    let scene = FrameScene {
        layers: capture.layers.clone(),
        ..FrameScene::default()
    };
    let mut items = layer_label_items(&view, &scene);
    let mut notes = Vec::new();
    if dialog {
        host_items(capture, settings, request, &mut items, &mut notes);
    }
    Prepared {
        scene,
        view,
        items,
        notes,
    }
}

/// Draws the prepared overlays onto each band before passing it on.
struct OverlaySink<'a, S> {
    inner: S,
    prepared: &'a Prepared,
    font: &'a FontArc,
    width: u32,
    height: u32,
    row: u32,
    band: Vec<u8>,
}

impl<S: BandSink> BandSink for OverlaySink<'_, S> {
    fn begin(&mut self, width: u32, height: u32) -> Result<(), ExportError> {
        self.width = width;
        self.height = height;
        self.row = 0;
        self.inner.begin(width, height)
    }

    fn write_rows(&mut self, rgba: &[u8], rows: u32) -> Result<(), ExportError> {
        self.band.clear();
        self.band.extend_from_slice(rgba);
        let view = self.prepared.view;
        if !self.prepared.scene.is_empty()
            && let Some(mut pixmap) = tiny_skia::Pixmap::new(self.width, rows)
        {
            let band_view = view.band(self.row, rows);
            raster::render_into(&mut pixmap, &band_view, &self.prepared.scene);
            raster::composite_over_straight(&mut self.band, pixmap.data());
        }
        if !self.prepared.items.is_empty() {
            let mut canvas = Canvas::band(&mut self.band, self.width, rows, i64::from(self.row));
            for item in &self.prepared.items {
                item.draw(&mut canvas, (self.width, self.height), self.font);
            }
        }
        self.row += rows;
        self.inner.write_rows(&self.band, rows)
    }

    fn finish(&mut self) -> Result<(), ExportError> {
        self.inner.finish()
    }
}

/// Renders `request` with the prepared overlays into `inner`, returning
/// the sink so the caller can take the image or abort a file.
fn render_with_overlays<S: BandSink>(
    source: &ExportSource,
    request: &ExportRequest,
    prepared: &Prepared,
    control: &JobControl,
    inner: S,
) -> (Result<ExportReport, ExportError>, S) {
    let mut sink = OverlaySink {
        inner,
        prepared,
        font: embedded_font(),
        width: 0,
        height: 0,
        row: 0,
        band: Vec::new(),
    };
    let result = render_export_bands(source, request, control, &mut sink);
    (result, sink.inner)
}

/// A preview of the export with notes for the dialog (worker thread).
fn render_preview(
    capture: &Capture,
    settings: &SlintExportSettings,
) -> Result<(RgbaImageData, Vec<String>), ExportError> {
    let manager = TileManager::new(capture.wsi.clone(), capture.file_id);
    let full = build_request(capture, settings, f64::from(settings.dpi.max(1)), None);
    let (width, height) = full.output_size();
    let factor = (f64::from(PREVIEW_MAX_SIDE) / f64::from(width.max(height))).min(1.0);
    let mut request = full;
    request.scale *= factor;
    let mut prepared = prepare(capture, settings, &request, true);
    prepared.notes.extend(fit_note(capture));
    prepared
        .notes
        .extend(color_management_note(capture, &request));
    let source = ExportSource {
        tile_manager: &manager,
        tile_cache: Some(&capture.tile_cache),
    };
    let (result, sink) = render_with_overlays(
        &source,
        &request,
        &prepared,
        &JobControl::new(),
        ImageSink::default(),
    );
    result.map(|_| (sink.into_image(), prepared.notes))
}

/// Renders the export to `path` (worker thread).
fn run_export(
    capture: &Capture,
    settings: &SlintExportSettings,
    path: &Path,
    format: ExportFormat,
    control: &JobControl,
) -> Result<ExportReport, ExportError> {
    let manager = TileManager::new(capture.wsi.clone(), capture.file_id);
    let dpi = f64::from(settings.dpi.max(1));
    let request = build_request(capture, settings, dpi, Some(&|| control.is_cancelled()));
    if control.is_cancelled() {
        return Err(ExportError::Cancelled);
    }
    let prepared = prepare(capture, settings, &request, true);
    let mut provenance = request.processing.summary().provenance();
    if let Some(note) = fit_note(capture) {
        provenance.push(' ');
        provenance.push_str(&note);
        provenance.push('.');
    }
    let color_space = export_color_space(capture, &request);
    if matches!(color_space, ColorSpace::Profile(_)) {
        provenance.push_str(" The slide's colour profile is embedded.");
    }
    let metadata = ImageMetadata::for_export(
        &request,
        dpi,
        capture.calibration.mpp_x,
        capture.calibration.mpp_y,
        Some(provenance),
    )
    .with_color_space(color_space);
    let source = ExportSource {
        tile_manager: &manager,
        tile_cache: Some(&capture.tile_cache),
    };
    let (result, mut sink) = render_with_overlays(
        &source,
        &request,
        &prepared,
        control,
        FileSink::new(path, format, metadata),
    );
    if result.is_err() {
        sink.abort();
    }
    result
}

// -- scripted exports (`export.view`) ------------------------------------------

/// What a script's export (`export.view`, [`crate::scripting`]) changes
/// from the dialog's defaults for the slide; `None` keeps a default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScriptedOptions {
    pub(crate) format: ExportFormat,
    pub(crate) dpi: Option<u32>,
    pub(crate) scale_bar: Option<bool>,
    pub(crate) roi: Option<bool>,
    pub(crate) measurements: Option<bool>,
    pub(crate) processing_note: Option<bool>,
}

/// A pane's view to export for a script, captured on the UI thread with
/// the dialog's default settings (the pane's colour processing, its ROI,
/// measurements and overlay layers); [`Self::run`] renders it on a worker
/// as the dialog's Export does.
pub(crate) struct ScriptedExport {
    capture: Capture,
    settings: SlintExportSettings,
    format: ExportFormat,
}

impl ScriptedExport {
    /// The slide shown in `pane`, or `None` when it shows none.
    pub(crate) fn capture(
        state: &AppState,
        tile_cache: &Arc<TileCache>,
        pane: PaneId,
        options: &ScriptedOptions,
    ) -> Option<Self> {
        let mut settings = default_settings(state, pane);
        if let Some(dpi) = options.dpi {
            settings.dpi = i32::try_from(dpi).unwrap_or(i32::MAX);
        }
        if let Some(show) = options.scale_bar {
            settings.show_scale_bar = show;
        }
        if let Some(show) = options.roi {
            settings.show_roi_outline = show && settings.has_roi;
        }
        if let Some(show) = options.measurements {
            settings.show_measurement = show && settings.has_measurement;
        }
        if let Some(show) = options.processing_note {
            settings.show_processing_note = show;
        }
        let capture = capture(state, tile_cache, pane, None, false, &settings)?;
        Some(Self {
            capture,
            settings,
            format: options.format,
        })
    }

    /// The image's size in pixels.
    pub(crate) fn output_size(&self) -> (u32, u32) {
        ExportRequest::new(
            self.capture.viewport.clone(),
            f64::from(self.settings.dpi.max(1)),
        )
        .output_size()
    }

    /// The exported slide's file name.
    pub(crate) fn file_name(&self) -> &str {
        &self.capture.file_name
    }

    /// The exported slide's file id.
    pub(crate) fn file_id(&self) -> i32 {
        self.capture.file_id
    }

    /// Renders the export to `path` (a worker thread); a failed or
    /// cancelled export leaves no file.
    pub(crate) fn run(
        &self,
        path: &Path,
        control: &JobControl,
    ) -> Result<ExportReport, ExportError> {
        run_export(&self.capture, &self.settings, path, self.format, control)
    }
}

// -- the dialog (UI thread) ---------------------------------------------------

/// Opens the export dialog for the slide shown in `pane`.
pub(crate) fn open(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    pane: PaneId,
) {
    let (settings, file_id, viewport) = {
        let state = state.read();
        let Some(file_id) = state.active_file_id_for_pane(pane) else {
            ui.set_status_text(SharedString::from("Open a slide to export an image"));
            return;
        };
        let viewport = state
            .get_file(file_id)
            .and_then(|file| file.pane_state(pane))
            .map(|pane_state| pane_state.viewport.viewport.clone());
        let remembered = DIALOG.with(|dialog| dialog.borrow().remembered.get(&file_id).cloned());
        let settings = match remembered {
            Some(mut settings) => {
                apply_hud(&mut settings, &state, pane);
                apply_slide(&mut settings, &state, pane);
                settings
            }
            None => default_settings(&state, pane),
        };
        (settings, file_id, viewport)
    };
    DIALOG.with(|dialog| dialog.borrow_mut().target = Some((pane, file_id)));
    ui.set_export_settings(settings);
    let (width, height) = viewport
        .map(|viewport| (viewport.width as f32, viewport.height as f32))
        .unwrap_or((0.0, 0.0));
    ui.set_export_preview_info(SlintExportPreviewInfo {
        preview_image: slint::Image::default(),
        estimated_size_mb: 0.0,
        width_px: 0,
        height_px: 0,
        viewport_width: width,
        viewport_height: height,
        note: SharedString::new(),
        rendering: false,
    });
    ui.set_export_dialog_visible(true);
    request_preview(ui, state, tile_cache);
}

/// Slides whose stain fit or colour profile has just finished: the preview
/// of the dialog's slide renders again (see [`refresh_stale_preview`]).
pub(crate) fn slide_colors_changed(file_ids: &[i32]) {
    DIALOG.with(|dialog| {
        let mut dialog = dialog.borrow_mut();
        if dialog
            .target
            .is_some_and(|(_, file)| file_ids.contains(&file))
        {
            dialog.preview_stale = true;
        }
    });
}

/// Renders the preview again after [`slide_colors_changed`] (UI thread,
/// no state lock held).
pub(crate) fn refresh_stale_preview(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) {
    let stale = DIALOG.with(|dialog| std::mem::take(&mut dialog.borrow_mut().preview_stale));
    if stale && ui.get_export_dialog_visible() {
        request_preview(ui, state, tile_cache);
    }
}

/// Remembers edited settings for the dialog's slide.
pub(crate) fn settings_changed(settings: SlintExportSettings) {
    DIALOG.with(|dialog| {
        let mut dialog = dialog.borrow_mut();
        if let Some((_, file_id)) = dialog.target {
            dialog.remembered.insert(file_id, settings);
        }
    });
}

/// Restores the defaults for the dialog's slide.
pub(crate) fn reset(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tile_cache: &Arc<TileCache>) {
    let Some((pane, file_id)) = DIALOG.with(|dialog| dialog.borrow().target) else {
        return;
    };
    DIALOG.with(|dialog| dialog.borrow_mut().remembered.remove(&file_id));
    ui.set_export_settings(default_settings(&state.read(), pane));
    request_preview(ui, state, tile_cache);
}

/// Closes the dialog, keeping its settings for the slide.
pub(crate) fn close(ui: &AppWindow) {
    settings_changed(ui.get_export_settings());
    DIALOG.with(|dialog| dialog.borrow_mut().target = None);
    ui.set_export_dialog_visible(false);
}

/// Renders a new preview on a worker; requests made while one renders are
/// coalesced into one more render with the latest settings.
pub(crate) fn request_preview(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) {
    let start = DIALOG.with(|dialog| {
        let mut dialog = dialog.borrow_mut();
        dialog.previews_wanted += 1;
        !std::mem::replace(&mut dialog.preview_running, true)
    });
    if start {
        spawn_preview(ui, state, tile_cache);
    }
}

fn spawn_preview(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tile_cache: &Arc<TileCache>) {
    let (target, generation) =
        DIALOG.with(|dialog| (dialog.borrow().target, dialog.borrow().previews_wanted));
    let settings = ui.get_export_settings();
    let captured = target.and_then(|(pane, file_id)| {
        capture(
            &state.read(),
            tile_cache,
            pane,
            Some(file_id),
            false,
            &settings,
        )
    });
    let Some(captured) = captured else {
        DIALOG.with(|dialog| dialog.borrow_mut().preview_running = false);
        return;
    };
    let mut info = ui.get_export_preview_info();
    info.rendering = true;
    ui.set_export_preview_info(info);

    let ui_weak = ui.as_weak();
    let state = Arc::clone(state);
    let tile_cache = Arc::clone(tile_cache);
    let spawned = std::thread::Builder::new()
        .name("eov-export-preview".to_string())
        .spawn(move || {
            let result = render_preview(&captured, &settings);
            let _ = slint::invoke_from_event_loop(move || {
                preview_done(ui_weak, state, tile_cache, generation, result)
            });
        });
    if spawned.is_err() {
        DIALOG.with(|dialog| dialog.borrow_mut().preview_running = false);
    }
}

fn preview_done(
    ui_weak: slint::Weak<AppWindow>,
    state: Arc<RwLock<AppState>>,
    tile_cache: Arc<TileCache>,
    generation: u64,
    result: Result<(RgbaImageData, Vec<String>), ExportError>,
) {
    let Some(ui) = ui_weak.upgrade() else {
        return;
    };
    let open = ui.get_export_dialog_visible();
    let again = DIALOG.with(|dialog| {
        let mut dialog = dialog.borrow_mut();
        let again = open && dialog.previews_wanted != generation;
        dialog.preview_running = again;
        again
    });
    if open {
        let mut info = ui.get_export_preview_info();
        info.rendering = again;
        match result {
            Ok((image, notes)) => {
                if let Some(buffer) = crate::blitter::create_image_buffer(
                    &image.pixels,
                    image.width as u32,
                    image.height as u32,
                ) {
                    info.preview_image = slint::Image::from_rgba8(buffer);
                }
                info.note = notes.join("  ·  ").into();
            }
            Err(err) => {
                info.note = format!("Preview failed: {err}").into();
            }
        }
        ui.set_export_preview_info(info);
    }
    if again {
        spawn_preview(&ui, &state, &tile_cache);
    }
}

/// Asks for a file name and starts the export as a status-bar job.
pub(crate) fn export(ui: &AppWindow, state: &Arc<RwLock<AppState>>, tile_cache: &Arc<TileCache>) {
    let Some((pane, file_id)) = DIALOG.with(|dialog| dialog.borrow().target) else {
        return;
    };
    let settings = ui.get_export_settings();
    let Some(captured) = capture(
        &state.read(),
        tile_cache,
        pane,
        Some(file_id),
        false,
        &settings,
    ) else {
        ui.set_status_text(SharedString::from(
            "The slide is no longer shown in this pane",
        ));
        return;
    };
    let format = export_format(&settings);
    let (filter_name, extensions) = match format {
        ExportFormat::Png => ("PNG image", ["png"].as_slice()),
        ExportFormat::Jpeg { .. } => ("JPEG image", ["jpg", "jpeg"].as_slice()),
        ExportFormat::Tiff { .. } => ("TIFF image", ["tif", "tiff"].as_slice()),
    };
    let stem = Path::new(&captured.file_name)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "export".to_string());
    let Some(path) = FileDialog::new()
        .add_filter(filter_name, extensions)
        .set_file_name(format!("{stem}-export.{}", format.extension()))
        .save_file()
    else {
        return;
    };
    // The format chosen in the dialog wins over a typed extension (EXP-3).
    let path = format.ensure_extension(&path);
    close(ui);

    let control = Arc::new(JobControl::new());
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cancel = {
        let control = Arc::clone(&control);
        Box::new(move || control.cancel()) as Box<dyn Fn()>
    };
    let Some(job_id) =
        with_host(|host| host.start_host_job(&format!("Exporting {name}"), Some(cancel)))
    else {
        return;
    };
    let timer = Timer::default();
    {
        let control = Arc::clone(&control);
        let job_id = job_id.clone();
        timer.start(TimerMode::Repeated, PROGRESS_INTERVAL, move || {
            let progress = control.progress();
            with_host(|host| {
                host.update_host_job(
                    &job_id,
                    (progress.total > 0).then(|| progress.fraction()),
                    None,
                )
            });
        });
    }
    DIALOG.with(|dialog| {
        dialog
            .borrow_mut()
            .progress_timers
            .insert(job_id.clone(), timer)
    });

    let spawned = std::thread::Builder::new()
        .name("eov-export".to_string())
        .spawn({
            let job_id = job_id.clone();
            let path = path.clone();
            move || {
                let result = run_export(&captured, &settings, &path, format, &control);
                let _ = slint::invoke_from_event_loop(move || export_done(job_id, path, result));
            }
        });
    if let Err(err) = spawned {
        export_done(job_id, path, Err(ExportError::Io(err)));
    }
}

fn export_done(job_id: String, path: PathBuf, result: Result<ExportReport, ExportError>) {
    DIALOG.with(|dialog| dialog.borrow_mut().progress_timers.remove(&job_id));
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    with_host(|host| {
        host.finish_host_job(&job_id);
        match result {
            Ok(report) => {
                let detail = format!(
                    "{} × {} px, saved to {}",
                    report.width,
                    report.height,
                    path.display()
                );
                host.notify(
                    NotifyLevel::Success,
                    &format!("Exported {name}"),
                    Some(&detail),
                    None,
                );
                // Never silently (EXP-1).
                if let Some(warning) = report.shortfall_message() {
                    host.notify(
                        NotifyLevel::Warning,
                        &format!("{name} is incomplete"),
                        Some(&warning),
                        None,
                    );
                }
            }
            Err(ExportError::Cancelled) => host.status("Export cancelled"),
            Err(err) => host.notify(
                NotifyLevel::Error,
                &format!("Could not export {name}"),
                Some(&err.to_string()),
                None,
            ),
        }
    });
}

// -- viewport copies ------------------------------------------------------------

/// Copies the pane's view to the clipboard as shown: screen resolution,
/// colour processing and every overlay layer.
pub(crate) fn copy_view(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    pane: PaneId,
) {
    let (captured, mut settings) = {
        let state = state.read();
        let settings = default_settings(&state, pane);
        (
            capture(&state, tile_cache, pane, None, true, &settings),
            settings,
        )
    };
    let Some(captured) = captured else {
        ui.set_status_text(SharedString::from("No slide to copy in this pane"));
        return;
    };
    settings.show_measurement = false;
    settings.show_roi_outline = false;
    settings.roi_outside_overlay = false;
    settings.show_scale_bar = false;
    settings.show_processing_note = false;
    let dpi = 96.0 * f64::from(ui.window().scale_factor());
    ui.set_status_text(SharedString::from("Copying the view…"));
    let spawned = std::thread::Builder::new()
        .name("eov-copy-view".to_string())
        .spawn(move || {
            let manager = TileManager::new(captured.wsi.clone(), captured.file_id);
            let request = build_request(&captured, &settings, dpi, Some(&|| false));
            let prepared = prepare(&captured, &settings, &request, false);
            let source = ExportSource {
                tile_manager: &manager,
                tile_cache: Some(&captured.tile_cache),
            };
            let (result, sink) = render_with_overlays(
                &source,
                &request,
                &prepared,
                &JobControl::new(),
                ImageSink::default(),
            );
            match result {
                Ok(_) => {
                    let image = crop_to_slide(sink.into_image(), &request);
                    crate::clipboard::copy_image(image, copied_notice);
                }
                Err(err) => {
                    // Fall back to the frame on screen (without overlays).
                    let message = err.to_string();
                    let _ = slint::invoke_from_event_loop(move || {
                        match crate::clipboard::capture_pane_clipboard_image(pane) {
                            Some(image) => crate::clipboard::copy_image(image, copied_notice),
                            None => {
                                with_host(|host| {
                                    host.notify(
                                        NotifyLevel::Error,
                                        "Could not copy the view",
                                        Some(&message),
                                        None,
                                    )
                                });
                            }
                        }
                    });
                }
            }
        });
    if spawned.is_err() {
        ui.set_status_text(SharedString::from("Could not copy the view"));
    }
}

/// Tells the user how a copy went (UI thread).
fn copied_notice(copied: bool) {
    with_host(|host| {
        if copied {
            host.notify(
                NotifyLevel::Success,
                "Viewport image copied to the clipboard",
                None,
                None,
            );
        } else {
            host.notify(
                NotifyLevel::Error,
                "Could not copy the image to the clipboard",
                None,
                None,
            );
        }
    });
}

/// Crops the rendered view to the slide, dropping the background around it.
fn crop_to_slide(image: RgbaImageData, request: &ExportRequest) -> RgbaImageData {
    let viewport = request.export_viewport();
    // Opposite corners of the slide; a turned view may swap them.
    let a = viewport.image_to_screen(0.0, 0.0);
    let b = viewport.image_to_screen(viewport.image_width, viewport.image_height);
    let clamp_x = |v: f64| v.round().clamp(0.0, image.width as f64) as usize;
    let clamp_y = |v: f64| v.round().clamp(0.0, image.height as f64) as usize;
    let (x0, x1) = (clamp_x(a.x.min(b.x)), clamp_x(a.x.max(b.x)));
    let (y0, y1) = (clamp_y(a.y.min(b.y)), clamp_y(a.y.max(b.y)));
    if x1 <= x0 || y1 <= y0 || (x0 == 0 && y0 == 0 && x1 == image.width && y1 == image.height) {
        return image;
    }
    let width = x1 - x0;
    let mut pixels = Vec::with_capacity(width * (y1 - y0) * 4);
    for y in y0..y1 {
        let start = (y * image.width + x0) * 4;
        pixels.extend_from_slice(&image.pixels[start..start + width * 4]);
    }
    RgbaImageData {
        width,
        height: y1 - y0,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(width: f64, height: f64, zoom: f64, center: (f64, f64)) -> ExportRequest {
        let mut viewport = Viewport::new(width, height, 1000.0, 500.0);
        viewport.zoom = zoom;
        viewport.center.x = center.0;
        viewport.center.y = center.1;
        ExportRequest::new(viewport, 96.0)
    }

    #[test]
    fn copies_are_cropped_to_the_slide() {
        // A 1000x500 slide shown at zoom 0.5 in a 800x400 view: 500x250 px
        // of slide centred in the view.
        let request = request(800.0, 400.0, 0.5, (500.0, 250.0));
        let (w, h) = request.output_size();
        let image = RgbaImageData {
            width: w as usize,
            height: h as usize,
            pixels: vec![255; (w * h * 4) as usize],
        };
        let cropped = crop_to_slide(image, &request);
        assert_eq!((cropped.width, cropped.height), (500, 250));
    }

    #[test]
    fn views_inside_the_slide_are_not_cropped() {
        let request = request(200.0, 100.0, 1.0, (500.0, 250.0));
        let (w, h) = request.output_size();
        let image = RgbaImageData {
            width: w as usize,
            height: h as usize,
            pixels: vec![255; (w * h * 4) as usize],
        };
        let cropped = crop_to_slide(image, &request);
        assert_eq!((cropped.width, cropped.height), (200, 100));
    }

    #[test]
    fn exports_render_slide_layers_roi_and_scale_bar_to_a_file() {
        use crate::overlay::OverlayStore;
        use eov_plugin_api::protocol::{LayerSpec, OverlayObject, Rgba, Shape, Style};

        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("slide.tif");
        eov_common::test_support::write_tiled_tiff(&slide, 512, 512, 256, |_, _| [200, 100, 150])
            .unwrap();
        let wsi = WsiFile::open(&slide).unwrap();

        // A plugin layer with a filled square in the top-left quarter.
        let store = OverlayStore::new();
        let owner = Owner::plugin("example");
        let style = Style {
            fill: Rgba(255, 0, 0, 255),
            stroke: Rgba(255, 0, 0, 255),
            ..Style::default()
        };
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("marks-1", 1).with_style(style),
            )
            .unwrap();
        let square = Shape::Rectangle {
            x: 32.0,
            y: 32.0,
            width: 128.0,
            height: 128.0,
        };
        store
            .upsert(
                &owner,
                "marks-1",
                vec![OverlayObject::new("a", square)],
                false,
            )
            .unwrap();

        // The whole slide in a 256 x 256 view (zoom 0.5).
        let viewport = Viewport::new(256.0, 256.0, 512.0, 512.0);
        let captured = Capture {
            file_id: 1,
            wsi,
            file_name: "slide.tif".to_string(),
            viewport,
            tile_cache: Arc::new(TileCache::with_limits(64, 64 << 20)),
            roi: Some(Rect {
                x: 256.0,
                y: 256.0,
                width: 200.0,
                height: 200.0,
            }),
            measurements: Vec::new(),
            calibration: Calibration::new(Some(0.5), Some(0.5)),
            unit: MeasurementUnit::Um,
            layers: store.file_layers(1).layers,
            fit: None,
            color_profile: None,
            pixel_view: false,
        };
        let settings = SlintExportSettings {
            dpi: 192,
            // Neutral colour processing.
            gamma: 1.0,
            contrast: 1.0,
            deconv_hematoxylin_intensity: 1.0,
            deconv_hematoxylin_visible: true,
            deconv_eosin_intensity: 1.0,
            deconv_eosin_visible: true,
            show_roi_outline: true,
            has_roi: true,
            roi_color: slint::Color::from_argb_u8(255, 241, 196, 15),
            roi_opacity: 100.0,
            roi_thickness: 2.0,
            show_scale_bar: true,
            format: SlintExportFormat::Png,
            ..Default::default()
        };
        let path = dir.path().join("figure.png");
        let control = JobControl::new();
        let report = run_export(&captured, &settings, &path, ExportFormat::Png, &control).unwrap();
        // 192 DPI doubles the view: 512 x 512 output pixels.
        assert_eq!((report.width, report.height), (512, 512));
        assert_eq!(report.tiles_failed, 0);
        assert_eq!(control.progress().done, 512);

        let image = image::open(&path).unwrap().to_rgba8();
        assert_eq!(image.dimensions(), (512, 512));
        let px = |x: u32, y: u32| image.get_pixel(x, y).0;
        // The slide colour, the plugin's square, and the ROI outline.
        assert_eq!(px(400, 100)[..3], [200, 100, 150]);
        assert_eq!(px(96, 96)[..3], [255, 0, 0]);
        let roi_edge = px(256, 330);
        assert!(
            roi_edge[0] > 200 && roi_edge[1] > 150 && roi_edge[2] < 80,
            "{roi_edge:?}"
        );
        // The scale bar is drawn in black near the bottom-right corner.
        let dark = (380..512)
            .flat_map(|x| (440..512).map(move |y| (x, y)))
            .filter(|&(x, y)| px(x, y)[..3].iter().all(|&c| c < 40))
            .count();
        assert!(dark > 50, "no scale bar ({dark} dark pixels)");
    }

    /// The export uses the viewer's fit of the slide: a failed one is not
    /// tried again (the export is not normalised, and says why), a pending
    /// one is waited for by exports and not by previews.
    #[test]
    fn exports_use_the_viewer_s_stain_fit() {
        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("slide.tif");
        eov_common::test_support::write_tiled_tiff(&slide, 256, 256, 256, |_, _| [200, 100, 150])
            .unwrap();
        let mut captured = Capture {
            file_id: 1,
            wsi: WsiFile::open(&slide).unwrap(),
            file_name: "slide.tif".to_string(),
            viewport: Viewport::new(128.0, 128.0, 256.0, 256.0),
            tile_cache: Arc::new(TileCache::with_limits(16, 16 << 20)),
            roi: None,
            measurements: Vec::new(),
            calibration: Calibration::new(None, None),
            unit: MeasurementUnit::Um,
            layers: Vec::new(),
            fit: Some((
                StainFitMethod::Macenko,
                SlideFit::Failed("no tissue to fit".into()),
            )),
            color_profile: None,
            pixel_view: false,
        };
        let settings = SlintExportSettings {
            dpi: 96,
            gamma: 1.0,
            contrast: 1.0,
            deconv_hematoxylin_intensity: 1.0,
            deconv_hematoxylin_visible: true,
            deconv_eosin_intensity: 1.0,
            deconv_eosin_visible: true,
            stain_normalization: SlintStainNormalization::Macenko,
            ..Default::default()
        };
        let request = build_request(&captured, &settings, 96.0, Some(&|| false));
        assert!(request.processing.summary().normalization.is_none());
        assert!(fit_note(&captured).unwrap().contains("no tissue to fit"));
        // A preview does not wait for a running fit: it says it is pending.
        captured.fit = Some((StainFitMethod::Macenko, SlideFit::Pending));
        let preview = build_request(&captured, &settings, 96.0, None);
        assert!(preview.processing.summary().normalization_pending);
    }

    /// A slide with a profile exports converted to sRGB (and says so), or,
    /// with the conversion off, as scanned with its profile embedded.
    #[test]
    fn exports_are_colour_managed_or_carry_the_profile() {
        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("slide.svs");
        let profile = eov_common::color_management::test_profiles::adobe_rgb();
        let mut spec = eov_common::test_support::TiffSpec::aperio(128, 128, 64, 4);
        spec.icc_profile = Some(profile.clone());
        eov_common::test_support::write_tiff(&slide, &spec, |_, _| [60, 170, 90]).unwrap();
        let transform = Arc::new(DisplayTransform::from_icc(&profile).unwrap());
        let captured = Capture {
            file_id: 1,
            wsi: WsiFile::open(&slide).unwrap(),
            file_name: "slide.svs".to_string(),
            viewport: Viewport::new(64.0, 64.0, 128.0, 128.0),
            tile_cache: Arc::new(TileCache::with_limits(16, 16 << 20)),
            roi: None,
            measurements: Vec::new(),
            calibration: Calibration::new(None, None),
            unit: MeasurementUnit::Um,
            layers: Vec::new(),
            fit: None,
            color_profile: Some(Arc::clone(&transform)),
            pixel_view: false,
        };
        let mut settings = SlintExportSettings {
            dpi: 96,
            gamma: 1.0,
            contrast: 1.0,
            deconv_hematoxylin_intensity: 1.0,
            deconv_hematoxylin_visible: true,
            deconv_eosin_intensity: 1.0,
            deconv_eosin_visible: true,
            has_color_profile: true,
            color_management: true,
            format: SlintExportFormat::Png,
            ..Default::default()
        };
        let export = |settings: &SlintExportSettings, name: &str| {
            let path = dir.path().join(name);
            run_export(
                &captured,
                settings,
                &path,
                ExportFormat::Png,
                &JobControl::new(),
            )
            .unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let pixel = image::load_from_memory(&bytes)
                .unwrap()
                .to_rgb8()
                .get_pixel(32, 32)
                .0;
            (bytes, pixel)
        };
        let has = |bytes: &[u8], chunk: &[u8]| bytes.windows(4).any(|w| w == chunk);

        let (managed, pixel) = export(&settings, "managed.png");
        assert_eq!(pixel, transform.apply_pixel([60, 170, 90]));
        assert!(has(&managed, b"sRGB") && !has(&managed, b"iCCP"));
        let request = build_request(&captured, &settings, 96.0, None);
        assert!(
            color_management_note(&captured, &request)
                .unwrap()
                .contains("sRGB")
        );

        settings.color_management = false;
        let (as_scanned, pixel) = export(&settings, "scanned.png");
        assert_eq!(pixel, [60, 170, 90]);
        assert!(has(&as_scanned, b"iCCP") && !has(&as_scanned, b"sRGB"));
    }

    /// A turned view exports as shown: slide, plugin layers and the ROI
    /// turn with it.
    #[test]
    fn turned_views_export_turned() {
        use crate::overlay::OverlayStore;
        use eov_plugin_api::protocol::{LayerSpec, OverlayObject, Rgba, Shape, Style};

        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("slide.tif");
        eov_common::test_support::write_tiled_tiff(&slide, 512, 512, 256, |_, _| [200, 100, 150])
            .unwrap();
        let wsi = WsiFile::open(&slide).unwrap();
        let store = OverlayStore::new();
        let owner = Owner::plugin("example");
        let red = Style {
            fill: Rgba(255, 0, 0, 255),
            stroke: Rgba(255, 0, 0, 255),
            ..Style::default()
        };
        store
            .create_layer(owner.clone(), LayerSpec::new("marks-1", 1).with_style(red))
            .unwrap();
        let square = Shape::Rectangle {
            x: 32.0,
            y: 32.0,
            width: 128.0,
            height: 128.0,
        };
        store
            .upsert(
                &owner,
                "marks-1",
                vec![OverlayObject::new("a", square)],
                false,
            )
            .unwrap();
        let mut viewport = Viewport::new(256.0, 256.0, 512.0, 512.0);
        viewport.orientation = eov_common::viewport::Orientation::new(1, false);
        let captured = Capture {
            file_id: 1,
            wsi,
            file_name: "slide.tif".to_string(),
            viewport,
            tile_cache: Arc::new(TileCache::with_limits(64, 64 << 20)),
            roi: Some(Rect {
                x: 256.0,
                y: 256.0,
                width: 200.0,
                height: 100.0,
            }),
            measurements: Vec::new(),
            calibration: Calibration::new(Some(0.5), Some(0.5)),
            unit: MeasurementUnit::Um,
            layers: store.file_layers(1).layers,
            fit: None,
            color_profile: None,
            pixel_view: false,
        };
        let settings = SlintExportSettings {
            dpi: 192,
            gamma: 1.0,
            contrast: 1.0,
            deconv_hematoxylin_intensity: 1.0,
            deconv_hematoxylin_visible: true,
            deconv_eosin_intensity: 1.0,
            deconv_eosin_visible: true,
            show_roi_outline: true,
            has_roi: true,
            roi_color: slint::Color::from_argb_u8(255, 241, 196, 15),
            roi_opacity: 100.0,
            roi_thickness: 2.0,
            format: SlintExportFormat::Png,
            ..Default::default()
        };
        let path = dir.path().join("turned.png");
        run_export(
            &captured,
            &settings,
            &path,
            ExportFormat::Png,
            &JobControl::new(),
        )
        .unwrap();
        let image = image::open(&path).unwrap().to_rgba8();
        let px = |x: u32, y: u32| image.get_pixel(x, y).0;
        // Turned clockwise, the slide's top-left square shows top-right.
        assert_eq!(px(416, 96)[..3], [255, 0, 0]);
        assert_eq!(px(96, 96)[..3], [200, 100, 150]);
        // The ROI's top edge (level-0 y = 256) is the column x = 256 now,
        // running down over its width (output y 256 to 456).
        let edge = px(256, 400);
        assert!(edge[0] > 200 && edge[1] > 150 && edge[2] < 80, "{edge:?}");
        let inside = px(200, 400);
        assert_eq!(inside[..3], [200, 100, 150], "inside the ROI");
    }

    #[test]
    fn the_dialog_format_wins_over_the_typed_extension() {
        let settings = SlintExportSettings {
            format: SlintExportFormat::Tiff,
            ..Default::default()
        };
        let format = export_format(&settings);
        assert_eq!(
            format.ensure_extension(Path::new("figure.png")),
            PathBuf::from("figure.tif")
        );
        let jpeg = SlintExportSettings {
            format: SlintExportFormat::Jpeg,
            jpeg_quality: 250,
            ..Default::default()
        };
        assert_eq!(export_format(&jpeg), ExportFormat::Jpeg { quality: 100 });
    }
}
