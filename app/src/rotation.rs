//! View rotation by any angle (AUDIT §14.1): the fine steps of the
//! keyboard, angles as people type and read them, and angles in saved
//! files.
//!
//! A view turns about the pane centre by any angle (kept to a millionth of
//! a degree, see [`Orientation`]); quarter turns stay exact. Saved views
//! (bookmarks, the session) write whole angles as integers, as before
//! views could turn freely, so older files and versions keep reading them.

use eov_common::viewport::Orientation;
use serde::Serializer;

/// Degrees one fine rotation step turns the view (`{` and `}` by default).
pub(crate) const FINE_STEP_DEGREES: f64 = 5.0;

/// An angle typed into the HUD's angle field: degrees clockwise, with an
/// optional `°` or `deg`, a decimal point or comma. `None` for anything
/// else (and for angles that are not finite).
pub(crate) fn parse_degrees(text: &str) -> Option<f64> {
    let text = text.trim();
    let text = text
        .strip_suffix('°')
        .or_else(|| text.strip_suffix("deg"))
        .unwrap_or(text)
        .trim();
    let text = text.replace(',', ".").replace('−', "-");
    text.parse::<f64>()
        .ok()
        .filter(|degrees| degrees.is_finite())
}

/// An angle in `0..360` as the HUD shows it: at most two decimals, no
/// trailing zeros ("90", "37.5", "12.35").
pub(crate) fn degrees_label(degrees: f64) -> String {
    trim_decimals(format!("{:.2}", round_to(degrees, 100.0)))
}

/// An angle with every decimal it has (a millionth of a degree at most),
/// for editing.
pub(crate) fn degrees_text(degrees: f64) -> String {
    trim_decimals(format!("{:.6}", round_to(degrees, 1e6)))
}

/// `degrees` rounded to `1 / per_degree`, with a full turn as 0.
fn round_to(degrees: f64, per_degree: f64) -> f64 {
    let rounded = (degrees * per_degree).round() / per_degree;
    if rounded >= 360.0 { 0.0 } else { rounded }
}

fn trim_decimals(text: String) -> String {
    if !text.contains('.') {
        return text;
    }
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The orientation `degrees` and `flipped` describe in a saved view: any
/// finite angle, upright for anything else.
pub(crate) fn saved_orientation(degrees: f64, flipped: bool) -> Orientation {
    Orientation::from_degrees(degrees, flipped).unwrap_or(Orientation::new(0, flipped))
}

/// `skip_serializing_if` for an upright view's angle.
pub(crate) fn is_upright(degrees: &f64) -> bool {
    *degrees == 0.0
}

/// Writes a whole angle as an integer (`rotation = 90`, readable by
/// versions that only knew quarter turns) and others as decimals.
pub(crate) fn serialize_degrees<S: Serializer>(
    degrees: &f64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if degrees.fract() == 0.0 && degrees.abs() < 1e9 {
        serializer.serialize_i64(*degrees as i64)
    } else {
        serializer.serialize_f64(*degrees)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_angles_parse_and_angles_print_briefly() {
        assert_eq!(parse_degrees("37.5"), Some(37.5));
        assert_eq!(parse_degrees(" 37,5° "), Some(37.5));
        assert_eq!(parse_degrees("-30 deg"), Some(-30.0));
        assert_eq!(parse_degrees("−12"), Some(-12.0));
        assert_eq!(parse_degrees("90°"), Some(90.0));
        assert_eq!(parse_degrees(""), None);
        assert_eq!(parse_degrees("north"), None);
        assert_eq!(parse_degrees("inf"), None);
        assert_eq!(parse_degrees("NaN"), None);
        assert_eq!(degrees_label(90.0), "90");
        assert_eq!(degrees_label(37.5), "37.5");
        assert_eq!(degrees_label(12.345678), "12.35");
        assert_eq!(degrees_label(359.999), "0");
        assert_eq!(degrees_text(12.345678), "12.345678");
        assert_eq!(degrees_text(0.0), "0");
        assert_eq!(degrees_text(270.0), "270");
    }

    #[test]
    fn saved_angles_read_back_and_bad_ones_are_upright() {
        assert_eq!(saved_orientation(37.5, true).degrees(), 37.5);
        assert!(saved_orientation(37.5, true).flipped());
        assert_eq!(saved_orientation(f64::NAN, true), Orientation::new(0, true));
        assert_eq!(
            saved_orientation(f64::INFINITY, false),
            Orientation::IDENTITY
        );
    }
}
