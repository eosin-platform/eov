//! Interpretation of OpenSlide properties into [`WsiProperties`] fields.

use super::SlideBounds;
use std::collections::BTreeMap;

pub(crate) type PropertyMap = BTreeMap<String, String>;

/// Upper bound accepted for microns per pixel. Real scans are 0.05-50 µm/px;
/// anything above this is a unit error in the file, not a calibration.
const MAX_PLAUSIBLE_MPP: f64 = 10_000.0;

/// Upper bound accepted for an objective power.
const MAX_PLAUSIBLE_OBJECTIVE: f64 = 1_000.0;

/// Parses a finite, strictly positive number not above `max`.
fn parse_positive(value: &str, max: f64) -> Option<f64> {
    let parsed: f64 = value.trim().parse().ok()?;
    (parsed.is_finite() && parsed > 0.0 && parsed <= max).then_some(parsed)
}

/// Microns per pixel along one axis, validated.
pub(crate) fn parse_mpp(map: &PropertyMap, key: &str) -> Option<f64> {
    map.get(key)
        .and_then(|value| parse_positive(value, MAX_PLAUSIBLE_MPP))
}

/// Objective power as a float, so 1.25×, 2.5× and 6.3× objectives survive.
pub(crate) fn parse_objective_power(map: &PropertyMap) -> Option<f64> {
    map.get("openslide.objective-power")
        .and_then(|value| parse_positive(value, MAX_PLAUSIBLE_OBJECTIVE))
}

pub(crate) fn clean_metadata_value(value: &str) -> Option<String> {
    let cleaned = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = cleaned.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("unknown")
        || trimmed.eq_ignore_ascii_case("n/a")
        || trimmed.eq_ignore_ascii_case("na")
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Scan or acquisition date, from the vendor-specific property that holds it.
///
/// The value is returned as written by the scanner (formats differ by
/// vendor); `tiff.DateTime` is the generic fallback for TIFF-based formats.
pub(crate) fn parse_scan_date(map: &PropertyMap) -> Option<String> {
    let get = |key: &str| map.get(key).and_then(|value| clean_metadata_value(value));

    // Aperio splits date and time into two fields.
    if let Some(date) = get("aperio.Date") {
        return Some(match get("aperio.Time") {
            Some(time) => format!("{date} {time}"),
            None => date,
        });
    }

    // DICOM: prefer the combined acquisition timestamp.
    if let Some(value) = get("dicom.AcquisitionDateTime") {
        return Some(value);
    }
    for (date_key, time_key) in [
        ("dicom.AcquisitionDate", "dicom.AcquisitionTime"),
        ("dicom.ContentDate", "dicom.ContentTime"),
        ("dicom.SeriesDate", "dicom.SeriesTime"),
        ("dicom.StudyDate", "dicom.StudyTime"),
    ] {
        if let Some(date) = get(date_key) {
            return Some(match get(time_key) {
                Some(time) => format!("{date} {time}"),
                None => date,
            });
        }
    }

    for key in [
        "mirax.GENERAL.SLIDE_CREATIONDATETIME",
        "philips.DICOM_ACQUISITION_DATETIME",
        "leica.creation-date",
        "sakura.Date",
        "hamamatsu.Created",
        "zeiss.Information.Document.CreationDate",
        "zeiss.Information.Image.AcquisitionDateAndTime",
        "tiff.DateTime",
    ] {
        if let Some(value) = get(key) {
            return Some(value);
        }
    }
    None
}

/// `openslide.background-color` (an `RRGGBB` hex triplet), white if absent
/// or malformed.
pub(crate) fn parse_background_color(map: &PropertyMap) -> [u8; 3] {
    map.get("openslide.background-color")
        .and_then(|value| parse_hex_rgb(value.trim()))
        .unwrap_or([255, 255, 255])
}

fn parse_hex_rgb(value: &str) -> Option<[u8; 3]> {
    let value = value.strip_prefix('#').unwrap_or(value);
    if value.len() != 6 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let channel = |i: usize| u8::from_str_radix(&value[i..i + 2], 16).ok();
    Some([channel(0)?, channel(2)?, channel(4)?])
}

fn parse_rect(map: &PropertyMap, prefix: &str, sep: &str) -> Option<SlideBounds> {
    let int = |name: &str| -> Option<i64> {
        map.get(&format!("{prefix}{sep}{name}"))?
            .trim()
            .parse()
            .ok()
    };
    let x = int("x")?;
    let y = int("y")?;
    let width = int("width")?;
    let height = int("height")?;
    (width > 0 && height > 0).then_some(SlideBounds {
        x,
        y,
        width: width as u64,
        height: height as u64,
    })
}

/// `openslide.bounds-{x,y,width,height}`, clipped to the level-0 image.
pub(crate) fn parse_bounds(map: &PropertyMap, width: u64, height: u64) -> Option<SlideBounds> {
    parse_rect(map, "openslide.bounds", "-")
        .and_then(|bounds| bounds.intersect(&SlideBounds::full(width, height)))
}

/// `openslide.region[i].{x,y,width,height}`, clipped to the level-0 image.
pub(crate) fn parse_regions(map: &PropertyMap, width: u64, height: u64) -> Vec<SlideBounds> {
    let full = SlideBounds::full(width, height);
    (0..)
        .map_while(|i| {
            let prefix = format!("openslide.region[{i}]");
            // Stop at the first index without any region property.
            map.contains_key(&format!("{prefix}.x"))
                .then(|| parse_rect(map, &prefix, "."))
        })
        .flatten()
        .filter_map(|region| region.intersect(&full))
        .collect()
}

/// Whether `needle` occurs in `haystack` delimited by non-alphanumeric
/// characters (or the string ends), e.g. "h&e" in "case 12 h&e" but not in
/// "smith&evans".
fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut search_from = 0;
    while let Some(found) = haystack[search_from..].find(needle) {
        let start = search_from + found;
        let end = start + needle.len();
        let before_ok = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let after_ok = haystack[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        search_from = start + needle.chars().next().map_or(1, char::len_utf8);
    }
    false
}

pub(crate) fn parse_stain_value_from_token(token: &str) -> Option<String> {
    let trimmed = token.trim();
    let lower = trimmed.to_lowercase();
    for separator in [':', '='] {
        if let Some((key, value)) = trimmed.split_once(separator) {
            let normalized_key = key
                .trim()
                .to_ascii_lowercase()
                .replace([' ', '-', '.'], "_");
            if matches!(
                normalized_key.as_str(),
                "stain" | "stain_name" | "stain_type" | "primary_stain" | "histologic_stain"
            ) {
                return clean_metadata_value(value);
            }
        }
    }

    const HE_PHRASES: [&str; 4] = [
        "hematoxylin and eosin",
        "hematoxylin & eosin",
        "haematoxylin and eosin",
        "haematoxylin & eosin",
    ];
    const HE_TOKENS: [&str; 4] = ["h&e", "h & e", "h+e", "he stain"];
    if HE_PHRASES
        .iter()
        .any(|phrase| contains_word(&lower, phrase))
        || HE_TOKENS.iter().any(|token| contains_word(&lower, token))
    {
        return Some("H&E".to_string());
    }

    if contains_word(&lower, "immunohistochemistry") || contains_word(&lower, "ihc") {
        return Some("IHC".to_string());
    }

    None
}

pub(crate) fn extract_stain_from_text(text: &str) -> Option<String> {
    text.split(['|', ';', '\n'])
        .find_map(parse_stain_value_from_token)
}

pub(crate) fn extract_stain_metadata(map: &PropertyMap) -> Option<String> {
    for (name, value) in map {
        if name.to_ascii_lowercase().contains("stain")
            && let Some(stain) = clean_metadata_value(value)
        {
            return Some(stain);
        }
    }
    for key in ["openslide.comment", "tiff.ImageDescription", "aperio.Title"] {
        if let Some(value) = map.get(key)
            && let Some(stain) = extract_stain_from_text(value)
        {
            return Some(stain);
        }
    }
    None
}

/// Native tile size of `level`, from `openslide.level[N].tile-{width,height}`.
pub(crate) fn level_tile_size(map: &PropertyMap, level: u32) -> (Option<u32>, Option<u32>) {
    let get = |dim: &str| {
        map.get(&format!("openslide.level[{level}].tile-{dim}"))
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|&v| v > 0)
    };
    (get("width"), get("height"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> PropertyMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parses_fractional_objectives_and_rejects_nonsense() {
        assert_eq!(
            parse_objective_power(&map(&[("openslide.objective-power", "2.5")])),
            Some(2.5)
        );
        assert_eq!(
            parse_objective_power(&map(&[("openslide.objective-power", "1.25")])),
            Some(1.25)
        );
        for bad in ["0", "-20", "NaN", "inf", "twenty", "1e9"] {
            assert_eq!(
                parse_objective_power(&map(&[("openslide.objective-power", bad)])),
                None,
                "{bad}"
            );
        }
    }

    #[test]
    fn parses_mpp_as_validated_f64() {
        let m = map(&[
            ("openslide.mpp-x", "0.49419999999999997"),
            ("openslide.mpp-y", "-1"),
        ]);
        assert_eq!(parse_mpp(&m, "openslide.mpp-x"), Some(0.4942));
        assert_eq!(parse_mpp(&m, "openslide.mpp-y"), None);
        assert_eq!(
            parse_mpp(&map(&[("openslide.mpp-x", "nan")]), "openslide.mpp-x"),
            None
        );
        assert_eq!(
            parse_mpp(&map(&[("openslide.mpp-x", "0")]), "openslide.mpp-x"),
            None
        );
    }

    #[test]
    fn reads_scan_dates_for_each_vendor() {
        let cases: &[(&[(&str, &str)], &str)] = &[
            (
                &[("aperio.Date", "08/11/16"), ("aperio.Time", "10:18:38")],
                "08/11/16 10:18:38",
            ),
            (
                &[("tiff.DateTime", "2012-10-29 21:12:59")],
                "2012-10-29 21:12:59",
            ),
            (
                &[(
                    "mirax.GENERAL.SLIDE_CREATIONDATETIME",
                    "17/06/2014 13:02:04",
                )],
                "17/06/2014 13:02:04",
            ),
            (
                &[(
                    "philips.DICOM_ACQUISITION_DATETIME",
                    "20160118142425.000000",
                )],
                "20160118142425.000000",
            ),
            (
                &[("leica.creation-date", "2010-09-24T12:43:41")],
                "2010-09-24T12:43:41",
            ),
            (
                &[
                    ("dicom.ContentDate", "20230101"),
                    ("dicom.ContentTime", "101500"),
                ],
                "20230101 101500",
            ),
            (&[("sakura.Date", "2013-11-12")], "2013-11-12"),
        ];
        for (pairs, expected) in cases {
            assert_eq!(parse_scan_date(&map(pairs)).as_deref(), Some(*expected));
        }
        assert_eq!(parse_scan_date(&map(&[("tiff.DateTime", "  ")])), None);
        // The vendor-specific value wins over the generic TIFF one.
        assert_eq!(
            parse_scan_date(&map(&[
                ("tiff.DateTime", "2000:01:01 00:00:00"),
                ("leica.creation-date", "2010-09-24"),
            ]))
            .as_deref(),
            Some("2010-09-24")
        );
    }

    #[test]
    fn parses_background_bounds_and_regions() {
        let m = map(&[
            ("openslide.background-color", "F0E0D0"),
            ("openslide.bounds-x", "100"),
            ("openslide.bounds-y", "-20"),
            ("openslide.bounds-width", "5000"),
            ("openslide.bounds-height", "400"),
            ("openslide.region[0].x", "0"),
            ("openslide.region[0].y", "34994"),
            ("openslide.region[0].width", "10136"),
            ("openslide.region[0].height", "21357"),
            ("openslide.region[1].x", "27333"),
            ("openslide.region[1].y", "0"),
            ("openslide.region[1].width", "86669"),
            ("openslide.region[1].height", "76349"),
        ]);
        assert_eq!(parse_background_color(&m), [0xf0, 0xe0, 0xd0]);
        assert_eq!(parse_background_color(&map(&[])), [255, 255, 255]);
        assert_eq!(
            parse_background_color(&map(&[("openslide.background-color", "zzzzzz")])),
            [255, 255, 255]
        );
        // Clipped to the 1000x300 image.
        assert_eq!(
            parse_bounds(&m, 1000, 300),
            Some(SlideBounds {
                x: 100,
                y: 0,
                width: 900,
                height: 300
            })
        );
        let regions = parse_regions(&m, 114_943, 76_349);
        assert_eq!(regions.len(), 2);
        assert_eq!(
            regions[1],
            SlideBounds {
                x: 27_333,
                y: 0,
                width: 86_669,
                height: 76_349
            }
        );
        assert!(parse_bounds(&map(&[("openslide.bounds-x", "1")]), 10, 10).is_none());
    }

    #[test]
    fn stain_detection_respects_word_boundaries() {
        assert_eq!(
            extract_stain_from_text("Case 12 | H&E | archived").as_deref(),
            Some("H&E")
        );
        assert_eq!(
            extract_stain_from_text("hematoxylin and eosin section").as_deref(),
            Some("H&E")
        );
        assert_eq!(extract_stain_from_text("H&E").as_deref(), Some("H&E"));
        assert_eq!(
            extract_stain_from_text("block A1 (h & e)").as_deref(),
            Some("H&E")
        );
        assert_eq!(extract_stain_from_text("Smith&Evans 2019"), None);
        assert_eq!(extract_stain_from_text("PATH&ENDO_case12"), None);
        assert_eq!(extract_stain_from_text("Thursday"), None);
        assert_eq!(extract_stain_from_text("IHC Ki-67").as_deref(), Some("IHC"));
        assert_eq!(extract_stain_from_text("vehicle"), None);
        assert_eq!(
            parse_stain_value_from_token("Stain = Masson trichrome").as_deref(),
            Some("Masson trichrome")
        );
        assert_eq!(
            parse_stain_value_from_token("stain_name: PAS").as_deref(),
            Some("PAS")
        );
    }

    #[test]
    fn stain_properties_take_precedence_over_free_text() {
        let m = map(&[
            ("openslide.comment", "H&E"),
            ("ventana.StainName", "  CD3  "),
        ]);
        assert_eq!(extract_stain_metadata(&m).as_deref(), Some("CD3"));
    }

    #[test]
    fn reads_native_tile_sizes() {
        let m = map(&[
            ("openslide.level[1].tile-width", "240"),
            ("openslide.level[1].tile-height", "240"),
        ]);
        assert_eq!(level_tile_size(&m, 1), (Some(240), Some(240)));
        assert_eq!(level_tile_size(&m, 0), (None, None));
    }
}
