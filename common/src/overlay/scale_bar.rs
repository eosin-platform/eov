//! A burned-in, calibrated scale bar.
//!
//! The bar length is the largest "nice" value (1, 2 or 5 × 10ⁿ **in the
//! selected unit**) that fits in a fraction of the image width, so the label
//! is always exact ("25 µm", "0.05 mm"), never a rounded approximation of a
//! micrometre value (UI-6). A horizontal bar is scaled with the horizontal
//! pixel size only.

use ab_glyph::{Font, FontArc, ScaleFont};
use serde::{Deserialize, Serialize};

use super::items::Corner;
use super::measure::{format_significant, unit_symbol, unit_um};
use super::{Canvas, OverlayColor, measure_text};
use crate::MeasurementUnit;

/// Appearance and placement of a scale bar. Sizes are in output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScaleBarStyle {
    pub corner: Corner,
    #[serde(with = "super::measure::unit_serde")]
    pub unit: MeasurementUnit,
    /// The bar is at most this fraction of the image width.
    pub max_fraction: f32,
    /// Use exactly this length (µm) instead of choosing one.
    pub fixed_length_um: Option<f64>,
    pub bar_height: f32,
    pub font_size: f32,
    pub margin: f32,
    pub padding: f32,
    pub bar_color: OverlayColor,
    pub text_color: OverlayColor,
    /// Box behind the bar and label (`None` for none).
    pub background: Option<OverlayColor>,
}

impl Default for ScaleBarStyle {
    fn default() -> Self {
        Self {
            corner: Corner::BottomRight,
            unit: MeasurementUnit::Um,
            max_fraction: 0.25,
            fixed_length_um: None,
            bar_height: 6.0,
            font_size: 14.0,
            margin: 16.0,
            padding: 8.0,
            bar_color: OverlayColor::BLACK,
            text_color: OverlayColor::BLACK,
            background: Some(OverlayColor::new(255, 255, 255, 210)),
        }
    }
}

impl ScaleBarStyle {
    /// Scales every size (e.g. by DPI / 96).
    pub fn scaled(mut self, factor: f32) -> Self {
        self.bar_height *= factor;
        self.font_size *= factor;
        self.margin *= factor;
        self.padding *= factor;
        self
    }
}

/// Where everything of a scale bar goes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleBarLayout {
    pub length_um: f64,
    pub length_px: f32,
    pub label: String,
    /// Background box `[x, y, w, h]`.
    pub box_rect: [f32; 4],
    /// The bar `[x, y, w, h]`.
    pub bar_rect: [f32; 4],
    /// Top-left of the label text.
    pub text_origin: [f32; 2],
}

/// The largest 1-2-5 × 10ⁿ length in `unit` that is at most `max_um`
/// micrometres.
pub fn nice_scale_length_um(max_um: f64, unit: MeasurementUnit) -> Option<f64> {
    if !(max_um.is_finite() && max_um > 0.0) {
        return None;
    }
    let per_unit = unit_um(unit);
    let max = max_um / per_unit;
    let exponent = max.log10().floor() as i32;
    for mantissa in [5.0, 2.0, 1.0] {
        let value = mantissa * 10f64.powi(exponent);
        if value <= max * (1.0 + 1e-9) {
            return Some(value * per_unit);
        }
    }
    Some(10f64.powi(exponent - 1) * 5.0 * per_unit)
}

/// Lays out a scale bar for an `image_width × image_height` image whose
/// pixels are `um_per_px` micrometres wide. Returns `None` when uncalibrated
/// or when no bar fits.
pub fn layout_scale_bar(
    image_width: u32,
    image_height: u32,
    um_per_px: f64,
    style: &ScaleBarStyle,
    font: &FontArc,
) -> Option<ScaleBarLayout> {
    if !(um_per_px.is_finite() && um_per_px > 0.0) {
        return None;
    }
    let max_px = (image_width as f32 * style.max_fraction.clamp(0.01, 1.0)) as f64;
    let length_um = match style.fixed_length_um {
        Some(fixed) if fixed > 0.0 => fixed,
        _ => nice_scale_length_um(max_px * um_per_px, style.unit)?,
    };
    let length_px = (length_um / um_per_px) as f32;
    if length_px < 2.0 || length_px > image_width as f32 {
        return None;
    }
    let label = format!(
        "{} {}",
        format_significant(length_um / unit_um(style.unit), 3, true),
        unit_symbol(style.unit)
    );
    let scaled = font.as_scaled(style.font_size);
    let text_h = scaled.ascent() - scaled.descent();
    let text_w = measure_text(&label, font, style.font_size);
    let gap = style.font_size * 0.25;
    let content_w = length_px.max(text_w);
    let content_h = text_h + gap + style.bar_height;
    let box_w = content_w + 2.0 * style.padding;
    let box_h = content_h + 2.0 * style.padding;
    let (box_x, box_y) = match style.corner {
        Corner::TopLeft => (style.margin, style.margin),
        Corner::TopRight => (image_width as f32 - style.margin - box_w, style.margin),
        Corner::BottomLeft => (style.margin, image_height as f32 - style.margin - box_h),
        Corner::BottomRight => (
            image_width as f32 - style.margin - box_w,
            image_height as f32 - style.margin - box_h,
        ),
    };
    let inner_x = box_x + style.padding;
    let inner_y = box_y + style.padding;
    Some(ScaleBarLayout {
        length_um,
        length_px,
        label,
        box_rect: [box_x, box_y, box_w, box_h],
        bar_rect: [
            inner_x + (content_w - length_px) / 2.0,
            inner_y + text_h + gap,
            length_px,
            style.bar_height,
        ],
        text_origin: [inner_x + (content_w - text_w) / 2.0, inner_y],
    })
}

/// A scale bar for an on-screen view: the nicest length (1, 2 or 5 × 10ⁿ in
/// the unit shown) that fits in `max_px` pixels of `um_per_px` µm each, as
/// `(length in pixels, label)`. In µm mode, lengths of a millimetre or
/// more are shown in mm. `None` when uncalibrated.
pub fn scale_bar_for_view(
    um_per_px: f64,
    max_px: f64,
    unit: MeasurementUnit,
) -> Option<(f64, String)> {
    if !(um_per_px.is_finite() && um_per_px > 0.0 && max_px > 0.0) {
        return None;
    }
    let max_um = max_px * um_per_px;
    let unit = match unit {
        MeasurementUnit::Um if max_um >= 1000.0 => MeasurementUnit::Mm,
        unit => unit,
    };
    let length_um = nice_scale_length_um(max_um, unit)?;
    let label = format!(
        "{} {}",
        format_significant(length_um / unit_um(unit), 3, true),
        unit_symbol(unit)
    );
    Some((length_um / um_per_px, label))
}

/// Draws a laid-out scale bar.
pub(crate) fn draw_scale_bar(
    canvas: &mut Canvas,
    layout: &ScaleBarLayout,
    style: &ScaleBarStyle,
    font: &FontArc,
) {
    if let Some(bg) = style.background {
        let [x, y, w, h] = layout.box_rect;
        canvas.fill_rect(x, y, w, h, bg);
    }
    let [x, y, w, h] = layout.bar_rect;
    canvas.fill_rect(x, y, w, h, style.bar_color);
    canvas.draw_text(
        layout.text_origin[0],
        layout.text_origin[1],
        &layout.label,
        style.text_color,
        font,
        style.font_size,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_scale_bars_label_exact_values() {
        // 0.25 µm/px at native 40x, up to 110 px: 25 µm (not "0.03 mm").
        let (px, label) = scale_bar_for_view(0.25, 110.0, MeasurementUnit::Mm).unwrap();
        assert_eq!(label, "0.02 mm");
        assert!((px - 80.0).abs() < 1e-9);
        let (_, label) = scale_bar_for_view(0.25, 110.0, MeasurementUnit::Um).unwrap();
        assert_eq!(label, "20 µm");
        // Sub-micrometre bars keep their digits.
        let (_, label) = scale_bar_for_view(0.01, 110.0, MeasurementUnit::Um).unwrap();
        assert_eq!(label, "1 µm");
        let (_, label) = scale_bar_for_view(0.004, 110.0, MeasurementUnit::Um).unwrap();
        assert_eq!(label, "0.2 µm");
        // Long bars switch to mm in µm mode.
        let (_, label) = scale_bar_for_view(20.0, 110.0, MeasurementUnit::Um).unwrap();
        assert_eq!(label, "2 mm");
        assert_eq!(scale_bar_for_view(0.0, 110.0, MeasurementUnit::Um), None);
    }
    use crate::overlay::embedded_font;

    #[test]
    fn nice_lengths_are_exact_in_the_selected_unit() {
        assert_eq!(nice_scale_length_um(37.0, MeasurementUnit::Um), Some(20.0));
        assert_eq!(nice_scale_length_um(60.0, MeasurementUnit::Um), Some(50.0));
        assert_eq!(nice_scale_length_um(0.3, MeasurementUnit::Um), Some(0.2));
        // 25 µm cannot be "0.03 mm": in mm the bar snaps to 0.02 mm.
        let mm = nice_scale_length_um(25.0, MeasurementUnit::Mm).unwrap();
        assert!((mm - 20.0).abs() < 1e-9, "{mm}");
        let inch = nice_scale_length_um(30_000.0, MeasurementUnit::Inches).unwrap();
        assert!((inch - 25_400.0).abs() < 1e-6, "{inch}");
        assert_eq!(nice_scale_length_um(0.0, MeasurementUnit::Um), None);
    }

    #[test]
    fn layout_is_calibrated_and_labelled_exactly() {
        let style = ScaleBarStyle {
            unit: MeasurementUnit::Mm,
            ..ScaleBarStyle::default()
        };
        // 0.25 µm/px, 800 px wide: max 200 px = 50 µm → 0.05 mm = 200 px.
        let layout = layout_scale_bar(800, 600, 0.25, &style, embedded_font()).unwrap();
        assert_eq!(layout.label, "0.05 mm");
        assert!((layout.length_px - 200.0).abs() < 1e-3);
        assert!(layout.box_rect[0] + layout.box_rect[2] <= 800.0 - style.margin + 1e-3);
        assert!(layout_scale_bar(800, 600, f64::NAN, &style, embedded_font()).is_none());
    }

    #[test]
    fn fixed_length_bars() {
        let style = ScaleBarStyle {
            fixed_length_um: Some(100.0),
            ..ScaleBarStyle::default()
        };
        let layout = layout_scale_bar(1000, 500, 0.5, &style, embedded_font()).unwrap();
        assert_eq!(layout.label, "100 µm");
        assert!((layout.length_px - 200.0).abs() < 1e-3);
    }
}
