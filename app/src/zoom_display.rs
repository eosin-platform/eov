use common::format_decimal;

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
                tooltip: format!(
                    "{}x optical-equivalent zoom; objective power {}x; digital zoom {digital_zoom_text}",
                    optical_text, objective_text
                ),
            }
        }
        ZoomDisplayMode::Percent => {
            let percent_text = format_percent_value(digital_zoom);
            ZoomDisplayInfo {
                text: format!("{percent_text}%"),
                input_text: percent_text,
                tooltip: "100% = 1 image pixel per display pixel".to_string(),
            }
        }
    }
}

pub fn parse_zoom_input(input: &str, objective_power: Option<f64>) -> Option<f64> {
    match zoom_display_mode(objective_power) {
        ZoomDisplayMode::Optical { objective_power } => {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                return None;
            }
            let value = trimmed
                .strip_suffix(['x', 'X'])
                .unwrap_or(trimmed)
                .trim()
                .parse::<f64>()
                .ok()?;
            if !value.is_finite() || value <= 0.0 {
                return None;
            }
            Some(value / objective_power)
        }
        ZoomDisplayMode::Percent => {
            let value = input.trim().parse::<f64>().ok()?;
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
    format_one_decimal_clean(value)
}

fn format_percent_text(digital_zoom: f64) -> String {
    format!("{}%", format_percent_value(digital_zoom))
}

fn format_percent_value(digital_zoom: f64) -> String {
    if digital_zoom >= 1.0 {
        (digital_zoom * 100.0).round().to_string()
    } else {
        format_one_decimal_clean((digital_zoom * 1000.0).round() / 10.0)
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
    use common::WsiFile;
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
            "20x optical-equivalent zoom; objective power 40x; digital zoom 50%"
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
        assert_eq!(zoom_display_mode(wsi.properties().objective_power), ZoomDisplayMode::Percent);
        assert_eq!(zoom_display_info(0.5, wsi.properties().objective_power).text, "50%");
    }
}