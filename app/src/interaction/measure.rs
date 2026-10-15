//! Host measurements and the region of interest: kinds, physical values and
//! label formatting.

use crate::overlay::geometry;
use eov_common::MeasurementUnit;
use eov_plugin_api::protocol::{Dash, Rect, Rgba, Shape, Style};
use serde::{Deserialize, Serialize};

/// Host layer holding the region of interest (below plugin layers).
pub const ROI_LAYER: &str = "roi";
pub const ROI_OBJECT: &str = "roi";
pub const ROI_Z: i32 = -50;
/// Host layer holding measurements (above plugin layers).
pub const MEASUREMENT_LAYER: &str = "measurements";
pub const MEASUREMENT_Z: i32 = 80;

/// What the Measure tool draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasureMode {
    /// A straight line (click-click or drag).
    #[default]
    Line,
    /// A path of segments (click vertices, double-click or Enter to finish).
    Polyline,
    /// A closed outline with area and perimeter.
    Polygon,
    Rectangle,
    Ellipse,
    /// Three clicks: first arm, vertex, second arm.
    Angle,
}

impl MeasureMode {
    pub const ALL: [MeasureMode; 6] = [
        MeasureMode::Line,
        MeasureMode::Polyline,
        MeasureMode::Polygon,
        MeasureMode::Rectangle,
        MeasureMode::Ellipse,
        MeasureMode::Angle,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MeasureMode::Line => "Distance",
            MeasureMode::Polyline => "Path length",
            MeasureMode::Polygon => "Polygon area",
            MeasureMode::Rectangle => "Rectangle area",
            MeasureMode::Ellipse => "Ellipse area",
            MeasureMode::Angle => "Angle",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            MeasureMode::Line => "line",
            MeasureMode::Polyline => "polyline",
            MeasureMode::Polygon => "polygon",
            MeasureMode::Rectangle => "rectangle",
            MeasureMode::Ellipse => "ellipse",
            MeasureMode::Angle => "angle",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.id() == id)
    }
}

/// A measurement stored with an open file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostMeasurement {
    pub id: String,
    pub mode: MeasureMode,
    pub shape: Shape,
}

/// Pixel spacing of a slide.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Calibration {
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
}

impl Calibration {
    pub fn new(mpp_x: Option<f64>, mpp_y: Option<f64>) -> Self {
        let valid = |v: Option<f64>| v.filter(|v| v.is_finite() && *v > 0.0);
        Self {
            mpp_x: valid(mpp_x),
            mpp_y: valid(mpp_y),
        }
    }

    /// `(mpp_x, mpp_y, approximate)`. With one axis missing the other is
    /// assumed for both (and the value is flagged approximate); with both
    /// missing there is no physical spacing.
    pub fn spacing(&self) -> Option<(f64, f64, bool)> {
        match (self.mpp_x, self.mpp_y) {
            (Some(x), Some(y)) => Some((x, y, false)),
            (Some(v), None) | (None, Some(v)) => Some((v, v, true)),
            (None, None) => None,
        }
    }

    /// Average spacing, for sizes that are not direction-dependent.
    pub fn average(&self) -> Option<f64> {
        self.spacing().map(|(x, y, _)| (x + y) / 2.0)
    }
}

/// The physical value of a measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeasurementValue {
    pub length_px: f64,
    pub area_px2: f64,
    pub length_um: Option<f64>,
    pub area_um2: Option<f64>,
    pub angle_deg: Option<f64>,
    /// Spacing was only partly known.
    pub approximate: bool,
}

pub fn measure(mode: MeasureMode, shape: &Shape, calibration: &Calibration) -> MeasurementValue {
    let spacing = calibration.spacing();
    if mode == MeasureMode::Angle {
        let angle = match shape {
            Shape::Polyline { points } if points.len() == 3 => {
                let (mx, my) = spacing.map(|(x, y, _)| (x, y)).unwrap_or((1.0, 1.0));
                Some(geometry::angle_deg(points[0], points[1], points[2], mx, my))
            }
            _ => None,
        };
        return MeasurementValue {
            angle_deg: angle,
            approximate: spacing.is_some_and(|(_, _, approx)| approx),
            ..Default::default()
        };
    }
    let closed = matches!(
        mode,
        MeasureMode::Polygon | MeasureMode::Rectangle | MeasureMode::Ellipse
    );
    MeasurementValue {
        length_px: geometry::shape_length(shape),
        area_px2: if closed { geometry::shape_area(shape) } else { 0.0 },
        length_um: spacing.map(|(x, y, _)| geometry::shape_length_um(shape, x, y)),
        area_um2: closed
            .then(|| spacing.map(|(x, y, _)| geometry::shape_area_um2(shape, x, y)))
            .flatten(),
        angle_deg: None,
        approximate: spacing.is_some_and(|(_, _, approx)| approx),
    }
}

/// Rounds to `digits` significant digits.
fn significant(value: f64, digits: i32) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let factor = 10f64.powi(digits - 1 - magnitude);
    (value * factor).round() / factor
}

/// Formats with three significant digits, but never in scientific notation
/// and without trailing zeros beyond what precision needs.
fn format_sig(value: f64) -> String {
    let rounded = significant(value, 3);
    let abs = rounded.abs();
    let decimals = if abs == 0.0 || abs >= 100.0 {
        0
    } else if abs >= 10.0 {
        1
    } else if abs >= 1.0 {
        2
    } else {
        // 0.0123 → 4 decimals for three significant digits.
        (2 - abs.log10().floor() as i32).clamp(0, 6) as usize
    };
    format!("{rounded:.decimals$}")
}

/// A length for display in the user's unit. In µm mode values from 1 mm
/// upwards are shown in mm.
pub fn format_length(um: f64, unit: MeasurementUnit) -> String {
    match unit {
        MeasurementUnit::Um if um.abs() >= 1000.0 => format!("{} mm", format_sig(um / 1000.0)),
        MeasurementUnit::Um => format!("{} µm", format_sig(um)),
        MeasurementUnit::Mm => format!("{} mm", format_sig(um / 1000.0)),
        MeasurementUnit::Inches => format!("{} in", format_sig(um / 25_400.0)),
    }
}

pub fn format_area(um2: f64, unit: MeasurementUnit) -> String {
    match unit {
        MeasurementUnit::Um if um2.abs() >= 1_000_000.0 => {
            format!("{} mm²", format_sig(um2 / 1_000_000.0))
        }
        MeasurementUnit::Um => format!("{} µm²", format_sig(um2)),
        MeasurementUnit::Mm => format!("{} mm²", format_sig(um2 / 1_000_000.0)),
        MeasurementUnit::Inches => format!("{} in²", format_sig(um2 / 645_160_000.0)),
    }
}

/// The text shown next to a measurement.
pub fn label(mode: MeasureMode, value: &MeasurementValue, unit: MeasurementUnit) -> String {
    let prefix = if value.approximate { "≈ " } else { "" };
    if mode == MeasureMode::Angle {
        return value
            .angle_deg
            .map(|deg| format!("{prefix}{deg:.1}°"))
            .unwrap_or_default();
    }
    let length = match value.length_um {
        Some(um) => format_length(um, unit),
        None => format!("{} px", format_sig(value.length_px)),
    };
    match mode {
        MeasureMode::Line | MeasureMode::Polyline => format!("{prefix}{length}"),
        _ => {
            let area = match value.area_um2 {
                Some(um2) => format_area(um2, unit),
                None => format!("{} px²", format_sig(value.area_px2)),
            };
            format!("{prefix}{area} · {length}")
        }
    }
}

/// Style of committed measurements.
pub fn measurement_style() -> Style {
    Style {
        stroke: Rgba(0x2E, 0xCC, 0x71, 0xFF),
        stroke_width: 2.0,
        fill: Rgba(0x2E, 0xCC, 0x71, 0x18),
        point_radius: 3.5,
        ..Style::default()
    }
}

/// Style of measurements and shapes being drawn.
pub fn preview_style() -> Style {
    Style {
        stroke: Rgba(0xF1, 0xC4, 0x0F, 0xFF),
        stroke_width: 2.0,
        fill: Rgba(0xF1, 0xC4, 0x0F, 0x14),
        dash: Dash::Solid,
        point_radius: 3.5,
        ..Style::default()
    }
}

/// Style of edit handles.
pub fn handle_style() -> Style {
    Style {
        fill: Rgba(0xFF, 0xFF, 0xFF, 0xFF),
        stroke: Rgba(0x1E, 0x90, 0xFF, 0xFF),
        stroke_width: 1.5,
        point_radius: 4.0,
        ..Style::default()
    }
}

/// The ROI as a rectangle shape.
pub fn roi_shape(rect: &Rect) -> Shape {
    Shape::Rectangle {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(len: f64) -> Shape {
        Shape::Polyline {
            points: vec![[0.0, 0.0], [len, 0.0]],
        }
    }

    #[test]
    fn missing_spacing_falls_back_to_pixels() {
        let value = measure(MeasureMode::Line, &line(250.0), &Calibration::default());
        assert_eq!(value.length_um, None);
        assert_eq!(label(MeasureMode::Line, &value, MeasurementUnit::Um), "250 px");
    }

    #[test]
    fn partial_spacing_is_flagged_approximate() {
        let value = measure(MeasureMode::Line, &line(100.0), &Calibration::new(Some(0.5), None));
        assert!(value.approximate);
        assert_eq!(label(MeasureMode::Line, &value, MeasurementUnit::Um), "≈ 50.0 µm");
    }

    #[test]
    fn anisotropic_spacing_is_honoured() {
        let shape = Shape::Polyline {
            points: vec![[0.0, 0.0], [0.0, 100.0]],
        };
        let value = measure(MeasureMode::Line, &shape, &Calibration::new(Some(0.25), Some(0.5)));
        assert_eq!(value.length_um, Some(50.0));
    }

    #[test]
    fn lengths_are_formatted_with_sensible_precision() {
        assert_eq!(format_length(0.253_7, MeasurementUnit::Um), "0.254 µm");
        assert_eq!(format_length(12.345, MeasurementUnit::Um), "12.3 µm");
        assert_eq!(format_length(999.4, MeasurementUnit::Um), "999 µm");
        assert_eq!(format_length(1234.0, MeasurementUnit::Um), "1.23 mm");
        assert_eq!(format_length(250.0, MeasurementUnit::Mm), "0.250 mm");
        assert_eq!(format_length(25_400.0, MeasurementUnit::Inches), "1.00 in");
        assert_eq!(format_area(2_500_000.0, MeasurementUnit::Um), "2.50 mm²");
    }

    #[test]
    fn areas_and_angles() {
        let rect = Shape::Rectangle {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        };
        let value = measure(MeasureMode::Rectangle, &rect, &Calibration::new(Some(0.5), Some(0.5)));
        assert_eq!(value.area_um2, Some(1250.0));
        assert_eq!(
            label(MeasureMode::Rectangle, &value, MeasurementUnit::Um),
            "1250 µm² · 150 µm"
        );
        let angle = Shape::Polyline {
            points: vec![[10.0, 0.0], [0.0, 0.0], [0.0, 10.0]],
        };
        let value = measure(MeasureMode::Angle, &angle, &Calibration::new(Some(0.5), Some(0.5)));
        assert_eq!(label(MeasureMode::Angle, &value, MeasurementUnit::Um), "90.0°");
    }
}
