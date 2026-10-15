//! Number formatting for user-facing labels.

fn trim_trailing_zeros(formatted: &mut String) {
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
}

/// Formats a number for display, trimming trailing zeros: two decimals for
/// magnitudes of at least 1 (`12.5`, `1234.57`), four significant digits
/// below 1 so small calibrations keep their precision (`0.4942` µm/px used
/// to be shown as `0.49`).
pub fn format_decimal(value: f64) -> String {
    if value != 0.0 && value.is_finite() && value.abs() < 1.0 {
        return format_significant(value, 4);
    }
    let mut formatted = format!("{value:.2}");
    trim_trailing_zeros(&mut formatted);
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    formatted
}

pub fn format_optional_decimal(value: Option<f64>) -> String {
    value
        .map(format_decimal)
        .unwrap_or_else(|| "unknown".to_string())
}

/// Formats `value` with `digits` significant digits (at least 1), trimming
/// trailing zeros after the decimal point. Integer digits are never
/// dropped: `format_significant(123456.0, 2)` is `"123456"`.
pub fn format_significant(value: f64, digits: u32) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let decimals = significant_decimals(value, digits);
    let mut formatted = format!("{value:.decimals$}");
    trim_trailing_zeros(&mut formatted);
    formatted
}

/// Decimals needed to show `value` with `digits` significant digits.
fn significant_decimals(value: f64, digits: u32) -> usize {
    let digits = digits.clamp(1, 17) as i32;
    let magnitude = value.abs().log10().floor() as i32;
    (digits - 1 - magnitude).clamp(0, 17) as usize
}

/// Formats microns per pixel with four significant digits (`0.4942`,
/// `0.2325`, `0.25`).
pub fn format_mpp(mpp: f64) -> String {
    format_significant(mpp, 4)
}

/// Formats a length given in micrometres for measurement labels and scale
/// bars: three significant digits below 100 µm (`0.250 µm`, `2.50 µm`,
/// `25.0 µm`), one decimal up to 1 mm (`123.5 µm`), and millimetres with
/// two decimals from there on (`1.23 mm`).
///
/// The unit is chosen **after** rounding, so 999.96 µm reads `1.00 mm`
/// rather than `1000.0 µm`.
pub fn format_length_um(um: f64) -> String {
    if !um.is_finite() {
        return "–".to_string();
    }
    if um == 0.0 {
        return "0 µm".to_string();
    }
    let abs = um.abs();
    let decimals = if abs < 100.0 {
        significant_decimals(um, 3)
    } else {
        1
    };
    let scale = 10f64.powi(decimals as i32);
    let rounded = (um * scale).round() / scale;
    if rounded.abs() < 1000.0 {
        return format!("{rounded:.decimals$} µm");
    }
    let mm = (um / 10.0).round() / 100.0;
    format!("{mm:.2} mm")
}

pub fn format_u64(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().rev().enumerate() {
        if index != 0 && index % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted.chars().rev().collect()
}

pub fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let bytes = bytes as f64;
    if bytes >= GB {
        format!("{} GB", format_decimal(bytes / GB))
    } else if bytes >= MB {
        format!("{} MB", format_decimal(bytes / MB))
    } else {
        format!("{} KB", format_decimal(bytes / KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_keep_precision_for_small_values() {
        assert_eq!(format_decimal(0.4942), "0.4942");
        assert_eq!(format_decimal(0.2325), "0.2325");
        assert_eq!(format_decimal(0.25), "0.25");
        assert_eq!(format_decimal(0.000_123_41), "0.0001234");
        assert_eq!(format_decimal(-0.5), "-0.5");
        assert_eq!(format_decimal(12.5), "12.5");
        assert_eq!(format_decimal(20.0), "20");
        assert_eq!(format_decimal(1234.5678), "1234.57");
        assert_eq!(format_decimal(0.0), "0");
        assert_eq!(format_decimal(-0.001), "-0.001");
        assert_eq!(format_optional_decimal(None), "unknown");
        assert_eq!(format_optional_decimal(Some(2.5)), "2.5");
    }

    #[test]
    fn significant_digits() {
        assert_eq!(format_significant(0.4942, 2), "0.49");
        assert_eq!(format_significant(123_456.0, 2), "123456");
        assert_eq!(format_significant(9.99, 2), "10");
        assert_eq!(format_significant(f64::NAN, 3), "NaN");
        assert_eq!(format_mpp("0.49419999999999997".parse().unwrap()), "0.4942");
        assert_eq!(format_mpp(0.465), "0.465");
    }

    #[test]
    fn lengths_switch_units_after_rounding() {
        assert_eq!(format_length_um(999.96), "1.00 mm");
        assert_eq!(format_length_um(999.94), "999.9 µm");
        assert_eq!(format_length_um(1234.5), "1.23 mm");
        assert_eq!(format_length_um(25.0), "25.0 µm");
        assert_eq!(format_length_um(2.5), "2.50 µm");
        assert_eq!(format_length_um(0.25), "0.250 µm");
        assert_eq!(format_length_um(123.45), "123.5 µm");
        assert_eq!(format_length_um(0.0), "0 µm");
        assert_eq!(format_length_um(f64::INFINITY), "–");
    }

    #[test]
    fn integers_and_sizes() {
        assert_eq!(format_u64(1_234_567), "1,234,567");
        assert_eq!(format_u64(12), "12");
        assert_eq!(format_file_size(1536 * 1024), "1.5 MB");
        assert_eq!(format_file_size(3 * 1024 * 1024 * 1024), "3 GB");
    }
}
