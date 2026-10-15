//! What a copy records about its slide: the `ImageDescription` text, the
//! OME-XML and the TIFF tags derived from them.
//!
//! De-identified copies are built from an **allow-list of fields**, never
//! by removing known-bad properties: the pixel size, the objective power,
//! the stain and, only when asked for, the scanner vendor and model. No
//! other property, date, file name, path, barcode, user name, comment,
//! slide or device id reaches the file, whatever the vendor called it.
//! Free-text fields that are kept (stain, scanner model) are refused when
//! they look like they carry an identifier (long digit runs, dates, e-mail
//! addresses, several lines). Copies that are not de-identified record the
//! source's file name, region and every vendor property as well.

use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::wsi::SlideBounds;

/// OpenSlide properties that describe the source file's encoding or
/// geometry, which the copy does not share (or records in its own way).
fn is_structural(key: &str) -> bool {
    const PREFIXES: [&str; 6] = [
        "openslide.level[",
        "openslide.associated.",
        "openslide.bounds-",
        "openslide.region[",
        "openslide.level-count",
        "openslide.icc-size",
    ];
    const KEYS: [&str; 12] = [
        "openslide.quickhash-1",
        "openslide.mpp-x",
        "openslide.mpp-y",
        "openslide.objective-power",
        "openslide.vendor",
        "openslide.comment",
        "openslide.background-color",
        "tiff.XResolution",
        "tiff.YResolution",
        "tiff.ResolutionUnit",
        "tiff.XPosition",
        "tiff.YPosition",
    ];
    PREFIXES.iter().any(|prefix| key.starts_with(prefix)) || KEYS.contains(&key)
}

/// Properties that name the scanner's manufacturer, first match wins.
const MAKE_KEYS: [&str; 4] = [
    "dicom.Manufacturer",
    "philips.DICOM_MANUFACTURER",
    "leica.device-vendor",
    "tiff.Make",
];

/// Properties that name the scanner model, first match wins.
const MODEL_KEYS: [&str; 6] = [
    "dicom.ManufacturerModelName",
    "philips.DICOM_MANUFACTURERS_MODEL_NAME",
    "leica.device-model",
    "ventana.ScannerModel",
    "hamamatsu.Product",
    "tiff.Model",
];

/// What a kept free-text field may look like in a de-identified copy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextScreen {
    pub max_chars: usize,
    /// Longest run of digits (accession, record and serial numbers are
    /// longer).
    pub max_digit_run: usize,
    /// Refuse anything that looks like a date (`12/03`, `03-12`, `2024-`).
    pub no_dates: bool,
}

/// Stain names: `H&E`, `Ki-67`, `PD-L1 (22C3)`.
pub(crate) const STAIN_SCREEN: TextScreen = TextScreen {
    max_chars: 64,
    max_digit_run: 4,
    no_dates: true,
};

/// Scanner makes and models: `Aperio AT2`, `C13220-01`.
pub(crate) const SCANNER_SCREEN: TextScreen = TextScreen {
    max_chars: 64,
    max_digit_run: 6,
    no_dates: false,
};

/// A free-text value kept in a de-identified copy, or `None` when it could
/// carry an identifier: too long, several lines, a long run of digits, a
/// date (with [`TextScreen::no_dates`]) or an e-mail address.
pub(crate) fn screened_text(value: &str, screen: TextScreen) -> Option<String> {
    let cleaned = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty()
        || cleaned.chars().count() > screen.max_chars
        || value.trim().contains(['\n', '\r'])
        || cleaned.contains('@')
    {
        return None;
    }
    let mut run = 0;
    for c in cleaned.chars() {
        run = if c.is_ascii_digit() { run + 1 } else { 0 };
        if run > screen.max_digit_run {
            return None;
        }
    }
    let looks_like_date = cleaned.as_bytes().windows(5).any(|w| {
        let digit = |b: u8| b.is_ascii_digit();
        (digit(w[0])
            && digit(w[1])
            && matches!(w[2], b'/' | b'-' | b'.')
            && digit(w[3])
            && digit(w[4]))
            || (digit(w[0]) && digit(w[1]) && digit(w[2]) && digit(w[3]) && w[4] == b'-')
    });
    if screen.no_dates && looks_like_date {
        return None;
    }
    Some(cleaned)
}

/// Today's date (UTC) as `YYYY-MM-DD`.
pub(crate) fn utc_date(now: SystemTime) -> String {
    let days = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0) as i64;
    // Civil-from-days (H. Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// A random (version 4) UUID, from the standard library's per-process
/// random hash keys and the clock. Not derived from anything about the
/// slide.
pub(crate) fn random_uuid() -> String {
    use std::hash::{BuildHasher, Hasher};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut bytes = [0u8; 16];
    for (i, half) in bytes.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_usize(i);
        hasher.write_u128(nanos);
        hasher.write_u32(std::process::id());
        *half = hasher.finish().to_le_bytes();
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// Shortest decimal for a measurement (`0.2527`, `20`, `2.5`).
fn number(value: f64) -> String {
    let text = format!("{value:.6}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() || text == "-" {
        "0".to_string()
    } else {
        text.to_string()
    }
}

/// How the copy's full-resolution level was made from the slide.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SourceSampling {
    /// A stored level at its native pixels.
    Level { level: u32, downsample: f64 },
    /// Resampled to `downsample` level-0 pixels per output pixel.
    Resampled { downsample: f64 },
}

/// Everything a copy records about its slide.
#[derive(Debug, Clone, Default)]
pub(crate) struct CopyMetadata {
    pub deidentified: bool,
    /// `YYYY-MM-DD` (UTC).
    pub created: String,
    /// `eov <version>`.
    pub software: String,
    /// µm per pixel of the copy's full-resolution level.
    pub mpp: Option<(f64, f64)>,
    /// Objective-equivalent magnification of the full-resolution level.
    pub objective_power: Option<f64>,
    pub stain: Option<String>,
    /// OpenSlide's format name (`aperio`, `hamamatsu`, ...).
    pub vendor: Option<String>,
    pub make: Option<String>,
    pub model: Option<String>,
    // Copies that are not de-identified only:
    pub source_name: Option<String>,
    pub source_region: Option<SlideBounds>,
    pub source_sampling: Option<SourceSampling>,
    /// Every non-structural property of the slide, by name.
    pub source_properties: Vec<(String, String)>,
}

/// The fields of a slide's metadata a copy may record.
pub(crate) struct SlideFacts<'a> {
    pub properties: &'a std::collections::BTreeMap<String, String>,
    pub file_name: &'a str,
    pub stain: Option<&'a str>,
    pub vendor: Option<&'a str>,
}

impl CopyMetadata {
    /// The metadata for a copy of a slide described by `facts`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        facts: &SlideFacts,
        deidentify: bool,
        keep_scanner: bool,
        mpp: Option<(f64, f64)>,
        objective_power: Option<f64>,
        region: SlideBounds,
        sampling: SourceSampling,
        now: SystemTime,
    ) -> Self {
        let lookup = |keys: &[&str]| {
            keys.iter()
                .find_map(|key| facts.properties.get(*key))
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let screen = |value: Option<String>, screen: TextScreen| {
            if deidentify {
                value.and_then(|v| screened_text(&v, screen))
            } else {
                value
            }
        };
        let scanner = !deidentify || keep_scanner;
        let mut metadata = Self {
            deidentified: deidentify,
            created: utc_date(now),
            software: crate::export::software_name(),
            mpp,
            objective_power,
            stain: screen(facts.stain.map(str::to_string), STAIN_SCREEN),
            vendor: scanner
                .then(|| facts.vendor.map(str::to_string))
                .flatten()
                .filter(|v| v.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')),
            make: if scanner {
                screen(lookup(&MAKE_KEYS), SCANNER_SCREEN)
            } else {
                None
            },
            model: if scanner {
                screen(lookup(&MODEL_KEYS), SCANNER_SCREEN)
            } else {
                None
            },
            ..Self::default()
        };
        if !deidentify {
            metadata.source_name = Some(facts.file_name.to_string());
            metadata.source_region = Some(region);
            metadata.source_sampling = Some(sampling);
            metadata.source_properties = facts
                .properties
                .iter()
                .filter(|(key, _)| !is_structural(key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
        }
        metadata
    }

    /// The first line of the description.
    fn headline(&self) -> String {
        let kind = if self.deidentified {
            "De-identified slide copy"
        } else {
            "Slide copy"
        };
        format!("{kind} made by {} on {}", self.software, self.created)
    }

    /// `key = value` lines shared by the TIFF description and OME-XML.
    fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = Vec::new();
        fields.push((
            "De-identified",
            if self.deidentified { "yes" } else { "no" }.to_string(),
        ));
        if let Some((x, y)) = self.mpp {
            let value = if (x - y).abs() <= 1e-9 * x.abs().max(1.0) {
                number(x)
            } else {
                format!("{} x {}", number(x), number(y))
            };
            fields.push(("MPP", value));
        }
        if let Some(power) = self.objective_power {
            fields.push(("Objective power", number(power)));
        }
        if let Some(stain) = &self.stain {
            fields.push(("Stain", stain.clone()));
        }
        if let Some(vendor) = &self.vendor {
            fields.push(("Source format", vendor.clone()));
        }
        match (&self.make, &self.model) {
            (Some(make), Some(model)) if model.starts_with(make.as_str()) => {
                fields.push(("Scanner", model.clone()))
            }
            (Some(make), Some(model)) => fields.push(("Scanner", format!("{make} {model}"))),
            (Some(one), None) | (None, Some(one)) => fields.push(("Scanner", one.clone())),
            (None, None) => {}
        }
        if let Some(name) = &self.source_name {
            fields.push(("Source file", name.clone()));
        }
        if let Some(region) = &self.source_region {
            fields.push((
                "Source region",
                format!(
                    "{},{},{},{} (level-0 pixels)",
                    region.x, region.y, region.width, region.height
                ),
            ));
        }
        match &self.source_sampling {
            Some(SourceSampling::Level { level, downsample }) => fields.push((
                "Source level",
                format!("{level} (downsample {})", number(*downsample)),
            )),
            Some(SourceSampling::Resampled { downsample }) => fields.push((
                "Source sampling",
                format!("resampled, downsample {}", number(*downsample)),
            )),
            None => {}
        }
        fields
    }

    /// The `ImageDescription` of a plain TIFF copy (UTF-8; transliterated
    /// to ASCII when written). Starts with the headline, never with a
    /// vendor's signature, so OpenSlide reads the copy as a generic TIFF.
    pub(crate) fn description(&self) -> String {
        let mut text = self.headline();
        for (key, value) in self.fields() {
            let _ = write!(text, "\n{key} = {value}");
        }
        if !self.source_properties.is_empty() {
            text.push_str("\nSource properties:");
            for (key, value) in &self.source_properties {
                let _ = write!(text, "\n{key} = {}", one_line(value));
            }
        }
        text
    }

    /// The OME-XML for an OME-TIFF copy. `images` are the image names and
    /// sizes in IFD order (the slide first, then associated images).
    pub(crate) fn ome_xml(&self, images: &[(String, u32, u32)], uuid: &str) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        let _ = write!(
            xml,
            "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" \
             xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
             xsi:schemaLocation=\"http://www.openmicroscopy.org/Schemas/OME/2016-06 \
             http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd\" \
             UUID=\"urn:uuid:{uuid}\" Creator=\"{}\">",
            xml_escape(&self.software)
        );
        let has_instrument =
            self.objective_power.is_some() || self.make.is_some() || self.model.is_some();
        if has_instrument {
            xml.push_str("<Instrument ID=\"Instrument:0\">");
            if self.make.is_some() || self.model.is_some() {
                xml.push_str("<Microscope");
                if let Some(make) = &self.make {
                    let _ = write!(xml, " Manufacturer=\"{}\"", xml_escape(make));
                }
                if let Some(model) = &self.model {
                    let _ = write!(xml, " Model=\"{}\"", xml_escape(model));
                }
                xml.push_str("/>");
            }
            if let Some(power) = self.objective_power {
                let _ = write!(
                    xml,
                    "<Objective ID=\"Objective:0\" NominalMagnification=\"{}\"/>",
                    number(power)
                );
            }
            xml.push_str("</Instrument>");
        }
        let mut description = self.headline();
        for (key, value) in self.fields() {
            let _ = write!(description, "\n{key} = {value}");
        }
        for (index, (name, width, height)) in images.iter().enumerate() {
            let _ = write!(
                xml,
                "<Image ID=\"Image:{index}\" Name=\"{}\">",
                xml_escape(name)
            );
            if index == 0 {
                let _ = write!(
                    xml,
                    "<Description>{}</Description>",
                    xml_escape(&description)
                );
                if has_instrument {
                    xml.push_str("<InstrumentRef ID=\"Instrument:0\"/>");
                }
                if self.objective_power.is_some() {
                    xml.push_str("<ObjectiveSettings ID=\"Objective:0\"/>");
                }
            }
            let _ = write!(
                xml,
                "<Pixels ID=\"Pixels:{index}\" DimensionOrder=\"XYCZT\" Type=\"uint8\" \
                 SizeX=\"{width}\" SizeY=\"{height}\" SizeZ=\"1\" SizeC=\"3\" SizeT=\"1\" \
                 Interleaved=\"true\" SignificantBits=\"8\""
            );
            if index == 0
                && let Some((x, y)) = self.mpp
            {
                let _ = write!(
                    xml,
                    " PhysicalSizeX=\"{}\" PhysicalSizeXUnit=\"&#xB5;m\" \
                     PhysicalSizeY=\"{}\" PhysicalSizeYUnit=\"&#xB5;m\"",
                    number(x),
                    number(y)
                );
            }
            let _ = write!(
                xml,
                "><Channel ID=\"Channel:{index}:0\" SamplesPerPixel=\"3\"><LightPath/></Channel>\
                 <TiffData IFD=\"{index}\" PlaneCount=\"1\"/></Pixels>"
            );
            xml.push_str("</Image>");
        }
        if !self.source_properties.is_empty() {
            // Bio-Formats' convention for a source's own metadata, which it
            // shows as "original metadata".
            xml.push_str("<StructuredAnnotations>");
            for (index, (key, value)) in self.source_properties.iter().enumerate() {
                let _ = write!(
                    xml,
                    "<XMLAnnotation ID=\"Annotation:{index}\" \
                     Namespace=\"openmicroscopy.org/OriginalMetadata\"><Value><OriginalMetadata>\
                     <Key>{}</Key><Value>{}</Value></OriginalMetadata></Value></XMLAnnotation>",
                    xml_escape(key),
                    xml_escape(value)
                );
            }
            xml.push_str("</StructuredAnnotations>");
        }
        xml.push_str("</OME>");
        xml
    }
}

/// A property value on one line (`\` and line breaks escaped).
fn one_line(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// Text for XML content and attribute values, in plain ASCII: markup
/// characters and everything outside ASCII become character references
/// (lossless), and control characters XML cannot hold are dropped.
pub(crate) fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\n' => out.push_str("&#10;"),
            '\t' => out.push_str("&#9;"),
            '\r' => out.push_str("&#13;"),
            c if (c as u32) < 0x20 => {}
            c if c.is_ascii() => out.push(c),
            c => {
                let _ = write!(out, "&#x{:X};", c as u32);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn properties(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn day(days: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(days * 86_400 + 3_600)
    }

    #[test]
    fn dates_are_civil_utc() {
        assert_eq!(utc_date(day(0)), "1970-01-01");
        assert_eq!(utc_date(day(19_723)), "2024-01-01");
        assert_eq!(utc_date(day(19_782)), "2024-02-29");
        assert_eq!(utc_date(day(20_741)), "2026-10-15");
        assert_eq!(utc_date(day(11_016)), "2000-02-29");
    }

    #[test]
    fn uuids_are_version_4_and_differ() {
        let a = random_uuid();
        let b = random_uuid();
        assert_ne!(a, b);
        assert_eq!(a.len(), 36);
        assert_eq!(&a[14..15], "4");
        assert!(matches!(&a[19..20], "8" | "9" | "a" | "b"));
    }

    #[test]
    fn free_text_with_identifiers_is_refused() {
        assert_eq!(
            screened_text("  H&E ", STAIN_SCREEN).as_deref(),
            Some("H&E")
        );
        assert_eq!(
            screened_text("Ki-67", STAIN_SCREEN).as_deref(),
            Some("Ki-67")
        );
        assert_eq!(
            screened_text("PD-L1 (22C3)", STAIN_SCREEN).as_deref(),
            Some("PD-L1 (22C3)")
        );
        assert_eq!(
            screened_text("NanoZoomer S360 C13220-01", SCANNER_SCREEN).as_deref(),
            Some("NanoZoomer S360 C13220-01")
        );
        let long = "x".repeat(65);
        for bad in [
            "S12-34567",
            "case 2024-03-01",
            "12/03/2024",
            "dr.who@example.org",
            "two\nlines",
            long.as_str(),
        ] {
            assert_eq!(screened_text(bad, STAIN_SCREEN), None, "{bad}");
        }
        assert_eq!(screened_text("SN 12345678", SCANNER_SCREEN), None);
    }

    #[test]
    fn deidentified_metadata_keeps_only_allowed_fields() {
        let props = properties(&[
            ("aperio.Barcode", "ABC-1234567"),
            ("aperio.Patient", "DOE^JANE"),
            ("aperio.Date", "08/11/16"),
            ("tiff.Make", "Aperio"),
            ("tiff.Model", "AT2"),
            ("openslide.vendor", "aperio"),
        ]);
        let facts = SlideFacts {
            properties: &props,
            file_name: "S12-34567 A1.svs",
            stain: Some("H&E"),
            vendor: Some("aperio"),
        };
        let region = SlideBounds::full(100, 100);
        let sampling = SourceSampling::Level {
            level: 0,
            downsample: 1.0,
        };
        let meta = CopyMetadata::new(
            &facts,
            true,
            false,
            Some((0.25, 0.25)),
            Some(40.0),
            region,
            sampling.clone(),
            day(20_741),
        );
        let text = meta.description();
        assert!(text.starts_with("De-identified slide copy made by eov "));
        assert!(text.contains("on 2026-10-15"));
        assert!(text.contains("MPP = 0.25\n") && text.contains("Objective power = 40"));
        assert!(text.contains("Stain = H&E"));
        for leaked in [
            "ABC", "DOE", "08/11", "S12", "Aperio", "AT2", "aperio", "Source",
        ] {
            assert!(!text.contains(leaked), "{leaked} leaked into {text}");
        }
        let xml = meta.ome_xml(&[("slide".into(), 100, 100)], "u");
        for leaked in ["ABC", "DOE", "08/11", "S12", "AT2", "Annotation"] {
            assert!(!xml.contains(leaked), "{leaked} leaked into {xml}");
        }

        let scanner = CopyMetadata::new(
            &facts,
            true,
            true,
            None,
            None,
            region,
            sampling.clone(),
            day(20_741),
        );
        assert!(scanner.description().contains("Scanner = Aperio AT2"));
        assert!(scanner.description().contains("Source format = aperio"));
        assert!(!scanner.description().contains("MPP"));

        let full = CopyMetadata::new(
            &facts,
            false,
            false,
            Some((0.25, 0.26)),
            None,
            region,
            sampling,
            day(20_741),
        );
        let text = full.description();
        assert!(text.starts_with("Slide copy made by eov"));
        assert!(text.contains("MPP = 0.25 x 0.26"));
        assert!(text.contains("Source file = S12-34567 A1.svs"));
        assert!(text.contains("aperio.Barcode = ABC-1234567"));
        assert!(!text.contains("openslide.vendor ="));
    }

    #[test]
    fn ome_xml_is_ascii_and_escaped() {
        let props = properties(&[("x.note", "a < b & \"c\" µ\nnext")]);
        let facts = SlideFacts {
            properties: &props,
            file_name: "Ünïcode.tif",
            stain: Some("H&E"),
            vendor: Some("generic-tiff"),
        };
        let meta = CopyMetadata::new(
            &facts,
            false,
            false,
            Some((0.5, 0.5)),
            Some(20.0),
            SlideBounds::full(10, 10),
            SourceSampling::Resampled { downsample: 2.0 },
            day(1),
        );
        let xml = meta.ome_xml(&[("Ünïcode".into(), 10, 10), ("label".into(), 4, 2)], "abc");
        assert!(xml.is_ascii());
        assert!(xml.contains(
            "<Key>x.note</Key><Value>a &lt; b &amp; &quot;c&quot; &#xB5;&#10;next</Value>"
        ));
        assert!(xml.contains("Name=\"&#xDC;n&#xEF;code\""));
        assert!(xml.contains("<TiffData IFD=\"1\" PlaneCount=\"1\"/>"));
        assert!(xml.contains("PhysicalSizeX=\"0.5\""));
        assert!(xml.contains("NominalMagnification=\"20\""));
        assert!(xml.contains("Stain = H&amp;E"));
    }
}
