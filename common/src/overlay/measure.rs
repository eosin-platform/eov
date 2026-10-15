//! Physical lengths from level-0 pixel distances and their display (EXP-5).
//!
//! Lengths are computed per axis — `hypot(dx·mpp_x, dy·mpp_y)` — so
//! anisotropic pixels are measured correctly; with one axis calibrated,
//! square pixels are assumed and the label is marked approximate; with none,
//! the length is reported in **level-0** pixels (never in screen or export
//! pixels, which change with zoom and DPI) and labelled as uncalibrated.

use crate::MeasurementUnit;

/// How a [`Length`] was calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Calibration {
    /// Both axes have a pixel size.
    Calibrated,
    /// Only one axis has a pixel size; square pixels were assumed.
    AssumedSquare,
    /// No pixel size; only the level-0 pixel length is known.
    Uncalibrated,
}

/// A measured distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Length {
    /// Euclidean length in level-0 pixels.
    pub level0_px: f64,
    /// Physical length in micrometres, when any axis is calibrated.
    pub micrometres: Option<f64>,
    pub calibration: Calibration,
}

fn valid_mpp(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0)
}

/// Measures the segment between two level-0 points.
pub fn measure_segment(
    start: [f64; 2],
    end: [f64; 2],
    mpp_x: Option<f64>,
    mpp_y: Option<f64>,
) -> Length {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let level0_px = dx.hypot(dy);
    let (micrometres, calibration) = match (valid_mpp(mpp_x), valid_mpp(mpp_y)) {
        (Some(mx), Some(my)) => (Some((dx * mx).hypot(dy * my)), Calibration::Calibrated),
        (Some(m), None) | (None, Some(m)) => (Some(level0_px * m), Calibration::AssumedSquare),
        (None, None) => (None, Calibration::Uncalibrated),
    };
    Length {
        level0_px,
        micrometres,
        calibration,
    }
}

/// Serde adapter for [`MeasurementUnit`] (`"um"`, `"mm"`, `"in"`).
pub mod unit_serde {
    use crate::MeasurementUnit;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(unit: &MeasurementUnit, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match unit {
            MeasurementUnit::Um => "um",
            MeasurementUnit::Mm => "mm",
            MeasurementUnit::Inches => "in",
        })
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MeasurementUnit, D::Error> {
        let text = String::deserialize(d)?;
        match text.as_str() {
            "um" | "µm" | "micrometre" | "micrometer" => Ok(MeasurementUnit::Um),
            "mm" | "millimetre" | "millimeter" => Ok(MeasurementUnit::Mm),
            "in" | "inch" | "inches" => Ok(MeasurementUnit::Inches),
            other => Err(serde::de::Error::custom(format!("unknown unit {other:?}"))),
        }
    }
}

/// Micrometres per unit.
pub(crate) fn unit_um(unit: MeasurementUnit) -> f64 {
    match unit {
        MeasurementUnit::Um => 1.0,
        MeasurementUnit::Mm => 1000.0,
        MeasurementUnit::Inches => 25_400.0,
    }
}

/// Display symbol of a unit.
pub fn unit_symbol(unit: MeasurementUnit) -> &'static str {
    match unit {
        MeasurementUnit::Um => "µm",
        MeasurementUnit::Mm => "mm",
        MeasurementUnit::Inches => "in",
    }
}

/// Formats `value` with at least `digits` significant digits (integers are
/// never rounded off), optionally trimming trailing zeros.
pub fn format_significant(value: f64, digits: u32, trim_zeros: bool) -> String {
    if !value.is_finite() {
        return "–".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let digits = digits.max(1) as i32;
    let magnitude = value.abs().log10().floor() as i32;
    if magnitude >= digits - 1 {
        // Large values keep every integer digit.
        return format!("{:.0}", value.round());
    }
    // Round to `digits` significant digits first, then decide decimals from
    // the rounded value (so 9.996 → "10.0", not "10.00").
    let scale = 10f64.powi(digits - 1 - magnitude);
    let rounded = (value * scale).round() / scale;
    let magnitude = if rounded == 0.0 {
        magnitude
    } else {
        rounded.abs().log10().floor() as i32
    };
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    let mut text = format!("{rounded:.decimals$}");
    if trim_zeros && text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

/// Formats a physical length (µm) in the given unit with three significant
/// digits, e.g. `"12.3 µm"`, `"0.0250 mm"`.
pub fn format_physical_length(micrometres: f64, unit: MeasurementUnit) -> String {
    format!(
        "{} {}",
        format_significant(micrometres / unit_um(unit), 3, false),
        unit_symbol(unit)
    )
}

/// Formats a measured length for a label, honouring the unit setting.
///
/// * calibrated: `"12.3 µm"`;
/// * one axis calibrated: `"≈12.3 µm"`;
/// * uncalibrated: `"1234 px (uncalibrated)"`, in level-0 pixels.
pub fn format_length(length: &Length, unit: MeasurementUnit) -> String {
    match (length.micrometres, length.calibration) {
        (Some(um), Calibration::Calibrated) => format_physical_length(um, unit),
        (Some(um), _) => format!("≈{}", format_physical_length(um, unit)),
        (None, _) => format!(
            "{} px (uncalibrated)",
            format_significant(length.level0_px, 3, true)
        ),
    }
}

/// µm below one millimetre, mm above, deciding on the *rounded* value.
pub(crate) fn auto_unit(micrometres: f64) -> MeasurementUnit {
    let rounded: f64 = format_significant(micrometres, 3, false)
        .parse()
        .unwrap_or(micrometres);
    if rounded.abs() >= 1000.0 {
        MeasurementUnit::Mm
    } else {
        MeasurementUnit::Um
    }
}

/// Pre-1.0 label formatter: µm below 1 mm, mm above.
#[deprecated(
    since = "1.0.0",
    note = "use `overlay::measure_segment` + `overlay::format_length`, which honour per-axis pixel \
            size and the unit setting"
)]
pub fn format_measurement_label(distance_um: f64) -> String {
    if distance_um.is_nan() || distance_um <= 0.0 {
        return String::new();
    }
    format_physical_length(distance_um, auto_unit(distance_um))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anisotropic_pixels_are_measured_per_axis() {
        let len = measure_segment([0.0, 0.0], [300.0, 400.0], Some(0.25), Some(0.5));
        assert_eq!(len.calibration, Calibration::Calibrated);
        assert!((len.level0_px - 500.0).abs() < 1e-9);
        // hypot(75, 200) = 213.6, not 500 × mean(0.25, 0.5) = 187.5.
        assert!((len.micrometres.unwrap() - 75f64.hypot(200.0)).abs() < 1e-9);
    }

    #[test]
    fn partial_and_missing_calibration_are_labelled_honestly() {
        let partial = measure_segment([0.0, 0.0], [100.0, 0.0], None, Some(0.5));
        assert_eq!(partial.calibration, Calibration::AssumedSquare);
        assert_eq!(format_length(&partial, MeasurementUnit::Um), "≈50.0 µm");
        let none = measure_segment([0.0, 0.0], [1000.0, 0.0], None, Some(f64::NAN));
        assert_eq!(none.calibration, Calibration::Uncalibrated);
        assert_eq!(
            format_length(&none, MeasurementUnit::Mm),
            "1000 px (uncalibrated)"
        );
    }

    #[test]
    fn units_follow_the_setting_with_three_significant_digits() {
        let len = measure_segment([0.0, 0.0], [100.0, 0.0], Some(0.25), Some(0.25));
        assert_eq!(format_length(&len, MeasurementUnit::Um), "25.0 µm");
        assert_eq!(format_length(&len, MeasurementUnit::Mm), "0.0250 mm");
        assert_eq!(format_length(&len, MeasurementUnit::Inches), "0.000984 in");
        assert_eq!(
            format_physical_length(0.25, MeasurementUnit::Um),
            "0.250 µm"
        );
        assert_eq!(
            format_physical_length(12_345.0, MeasurementUnit::Um),
            "12345 µm"
        );
    }

    #[test]
    fn significant_digit_formatting_handles_rounding_boundaries() {
        assert_eq!(format_significant(9.996, 3, false), "10.0");
        assert_eq!(format_significant(0.025, 3, true), "0.025");
        assert_eq!(format_significant(1500.0, 3, true), "1500");
        assert_eq!(format_significant(-2.5, 3, false), "-2.50");
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_label_switches_units_after_rounding() {
        assert_eq!(format_measurement_label(999.96), "1.00 mm");
        assert_eq!(format_measurement_label(12.34), "12.3 µm");
        assert_eq!(format_measurement_label(0.0), "");
    }
}
