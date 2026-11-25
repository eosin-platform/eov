use crate::ui_update::localized_number;
use eov_common::{format_decimal, format_significant};
use eov_plugin_api::i18n::{parse_decimal, tr};

const WHOLE_NUMBER_EPSILON: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoomDisplayMode {
    Optical { objective_power: f64 },
    Percent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoomDisplayInfo {
    pub text: String,
    pub input_text: String,
    pub tooltip: String,
}

pub fn zoom_display_mode(objective_power: Option<f64>) -> ZoomDisplayMode {
    reliable_objective_power(objective_power)
        .map(|objective_power| ZoomDisplayMode::Optical { objective_power })
        .unwrap_or(ZoomDisplayMode::Percent)
}

pub fn zoom_display_info(digital_zoom: f64, objective_power: Option<f64>) -> ZoomDisplayInfo {
    match zoom_display_mode(objective_power) {
        ZoomDisplayMode::Optical { objective_power } => {
            let optical_zoom = objective_power * digital_zoom;
            let optical_text = format_optical_value(optical_zoom);
            let objective_text = format_optical_value(objective_power);
            let digital_zoom_text = format_percent_text(digital_zoom);
            ZoomDisplayInfo {
                text: format!("{optical_text}x"),
                input_text: optical_text.clone(),
                tooltip: tr!(
                    // Translators: "20×" is a magnification (as by a
                    // microscope objective); the digital zoom is a percentage.
                    "{optical}× optical-equivalent zoom; objective power {objective}×; digital zoom {digital}",
                    optical = optical_text,
                    objective = objective_text,
                    digital = digital_zoom_text
                ),
            }
        }
        ZoomDisplayMode::Percent => {
            let percent_text = format_percent_value(digital_zoom);
            ZoomDisplayInfo {
                text: format!("{percent_text}%"),
                input_text: percent_text,
                tooltip: tr!("100% = 1 image pixel per display pixel"),
            }
        }
    }
}

/// The zoom typed into the zoom field, as a number in the interface's
/// language ([`parse_decimal`]). A number that reads as one with a
/// thousands separator (`1,600` where the decimal separator is not `,`,
/// `1.600` where it is) is refused rather than taken for 1.6: the field
/// never shows thousands separators, and zooming to 1.6 % for 1,600 % would
/// be far off.
pub fn parse_zoom_input(input: &str, objective_power: Option<f64>) -> Option<f64> {
    match zoom_display_mode(objective_power) {
        ZoomDisplayMode::Optical { objective_power } => {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                return None;
            }
            let value = parse_decimal(trimmed.strip_suffix(['x', 'X']).unwrap_or(trimmed))?;
            if !value.is_finite() || value <= 0.0 {
                return None;
            }
            Some(value / objective_power)
        }
        ZoomDisplayMode::Percent => {
            let value = parse_decimal(input)?;
            if !value.is_finite() || value <= 0.0 {
                return None;
            }
            Some(value / 100.0)
        }
    }
}

pub fn reliable_objective_power(objective_power: Option<f64>) -> Option<f64> {
    objective_power.filter(|value| value.is_finite() && *value > 0.0)
}

fn format_optical_value(value: f64) -> String {
    format_small_value(value)
}

fn format_percent_text(digital_zoom: f64) -> String {
    format!("{}%", format_percent_value(digital_zoom))
}

fn format_percent_value(digital_zoom: f64) -> String {
    if digital_zoom >= 1.0 {
        (digital_zoom * 100.0).round().to_string()
    } else {
        format_small_value(digital_zoom * 100.0)
    }
}

/// One decimal, or two significant digits below 1, so that small values
/// never read as 0 (RND-10); with the decimal separator of the interface's
/// language, as typed back into the zoom field.
fn format_small_value(value: f64) -> String {
    if value > 0.0 && value < 0.95 {
        localized_number(&format_significant(value, 2))
    } else {
        localized_number(&format_one_decimal_clean(value))
    }
}

fn format_one_decimal_clean(value: f64) -> String {
    let rounded = (value * 10.0).round() / 10.0;
    if (rounded - rounded.round()).abs() < WHOLE_NUMBER_EPSILON {
        rounded.round().to_string()
    } else {
        format_decimal(rounded)
    }
}

#[cfg(test)]
mod tests {
    use super::{ZoomDisplayMode, *};
    use eov_common::WsiFile;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn uses_optical_mode_for_valid_objective_power() {
        assert_eq!(
            zoom_display_mode(Some(40.0)),
            ZoomDisplayMode::Optical {
                objective_power: 40.0,
            }
        );
    }

    #[test]
    fn uses_percent_mode_when_objective_power_is_missing_or_invalid() {
        assert_eq!(zoom_display_mode(None), ZoomDisplayMode::Percent);
        assert_eq!(zoom_display_mode(Some(0.0)), ZoomDisplayMode::Percent);
        assert_eq!(zoom_display_mode(Some(-20.0)), ZoomDisplayMode::Percent);
        assert_eq!(zoom_display_mode(Some(f64::NAN)), ZoomDisplayMode::Percent);
    }

    #[test]
    fn formats_40x_slide_at_expected_digital_zooms() {
        assert_eq!(zoom_display_info(1.0, Some(40.0)).text, "40x");
        assert_eq!(zoom_display_info(0.5, Some(40.0)).text, "20x");
        assert_eq!(zoom_display_info(0.25, Some(40.0)).text, "10x");
    }

    #[test]
    fn formats_20x_slide_at_expected_digital_zooms() {
        assert_eq!(zoom_display_info(1.0, Some(20.0)).text, "20x");
        assert_eq!(zoom_display_info(0.5, Some(20.0)).text, "10x");
    }

    #[test]
    fn low_magnifications_never_read_as_zero() {
        assert_eq!(zoom_display_info(0.001, Some(40.0)).text, "0.04x");
        assert_eq!(zoom_display_info(0.001, Some(20.0)).text, "0.02x");
        assert_eq!(zoom_display_info(0.015, Some(40.0)).text, "0.6x");
        assert_eq!(zoom_display_info(0.03, Some(40.0)).text, "1.2x");
        assert_eq!(zoom_display_info(0.001, None).text, "0.1%");
        assert_eq!(zoom_display_info(0.125, None).text, "12.5%");
        // What is shown parses back to the same zoom.
        let shown = zoom_display_info(0.001, Some(40.0)).input_text;
        assert_eq!(parse_zoom_input(&shown, Some(40.0)), Some(0.001));
    }

    #[test]
    fn keeps_percent_display_when_objective_power_is_missing() {
        let info = zoom_display_info(0.5, None);
        assert_eq!(info.text, "50%");
        assert_eq!(info.input_text, "50");
        assert_eq!(info.tooltip, "100% = 1 image pixel per display pixel");
    }

    #[test]
    fn keeps_percent_display_when_objective_power_is_invalid() {
        let info = zoom_display_info(0.5, Some(0.0));
        assert_eq!(info.text, "50%");
        assert_eq!(info.input_text, "50");
    }

    #[test]
    fn optical_tooltip_includes_objective_and_digital_zoom() {
        let info = zoom_display_info(0.5, Some(40.0));
        assert_eq!(
            info.tooltip,
            "20× optical-equivalent zoom; objective power 40×; digital zoom 50%"
        );
    }

    #[test]
    fn optical_mode_parses_bare_numbers_and_x_suffix() {
        assert_eq!(parse_zoom_input("20", Some(40.0)), Some(0.5));
        assert_eq!(parse_zoom_input("20x", Some(40.0)), Some(0.5));
        assert_eq!(parse_zoom_input("2.5x", Some(40.0)), Some(0.0625));
    }

    #[test]
    fn fallback_mode_preserves_percent_input_behavior() {
        assert_eq!(parse_zoom_input("50", None), Some(0.5));
        assert_eq!(parse_zoom_input("12.5", None), Some(0.125));
    }

    #[test]
    fn rejects_invalid_zoom_input() {
        assert_eq!(parse_zoom_input("", None), None);
        assert_eq!(parse_zoom_input("abc", Some(40.0)), None);
        assert_eq!(parse_zoom_input("0", Some(40.0)), None);
        assert_eq!(parse_zoom_input("-20", None), None);
    }

    /// A separator between thousands is refused rather than read as the
    /// decimal one; a decimal separator the language does not use is still
    /// read where it cannot be one between thousands.
    #[test]
    fn thousands_are_not_read_as_decimals() {
        use eov_plugin_api::i18n;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        assert_eq!(parse_zoom_input("1,600", None), None);
        assert_eq!(parse_zoom_input("10,000", None), None);
        assert_eq!(parse_zoom_input(" -2,500 ", None), None);
        assert_eq!(parse_zoom_input("1,000x", Some(40.0)), None);
        assert_eq!(parse_zoom_input("2.500", None), Some(0.025));
        assert_eq!(parse_zoom_input("0,250", None), Some(0.0025));
        assert_eq!(parse_zoom_input("2,5", None), Some(0.025));
        assert_eq!(parse_zoom_input("1250,5", None), Some(12.505));
        assert_eq!(parse_zoom_input("2,5x", Some(40.0)), Some(0.0625));

        let _english = English;
        i18n::set_catalog(i18n::by_tag("de").unwrap(), None);
        assert_eq!(parse_zoom_input("2,500", None), Some(0.025));
        assert_eq!(parse_zoom_input("2.500", None), None);
        assert_eq!(parse_zoom_input("12.5", None), Some(0.125));
        // What the field shows reads back.
        let shown = zoom_display_info(0.125, None).input_text;
        assert_eq!(shown, "12,5");
        assert_eq!(parse_zoom_input(&shown, None), Some(0.125));

        // Persian digits, with the Arabic decimal sign.
        i18n::set_catalog(i18n::by_tag("fa").unwrap(), None);
        assert_eq!(parse_zoom_input("۲,۵۰۰", None), None);
        assert_eq!(parse_zoom_input("۲\u{66b}۵", None), Some(0.025));
    }

    #[test]
    #[ignore = "uses large local fixtures"]
    fn fixture_with_objective_metadata_uses_optical_mode() {
        let path = fixture_path("C3L-00088-22.svs");
        if !path.exists() {
            eprintln!("Skipping test: fixture file not found at {:?}", path);
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI fixture");
        assert_eq!(wsi.properties().objective_power, Some(20.0));
        assert_eq!(
            zoom_display_mode(wsi.properties().objective_power),
            ZoomDisplayMode::Optical {
                objective_power: 20.0,
            }
        );
    }

    #[test]
    #[ignore = "uses large local fixtures"]
    fn fixture_without_objective_metadata_stays_in_percent_mode() {
        let path = fixture_path("patient_198_node_0.tif");
        if !path.exists() {
            eprintln!("Skipping test: fixture file not found at {:?}", path);
            return;
        }

        let wsi = WsiFile::open(&path).expect("Failed to open WSI fixture");
        assert_eq!(wsi.properties().objective_power, None);
        assert_eq!(
            zoom_display_mode(wsi.properties().objective_power),
            ZoomDisplayMode::Percent
        );
        assert_eq!(
            zoom_display_info(0.5, wsi.properties().objective_power).text,
            "50%"
        );
    }
}
