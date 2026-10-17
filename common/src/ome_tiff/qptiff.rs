//! Akoya (formerly PerkinElmer) QPTIFF: the slides of the Vectra, Polaris
//! and PhenoImager scanners, multiplex immunofluorescence (one 8- or
//! 16-bit image per channel, 32-bit float for unmixed components) or
//! brightfield (RGB).
//!
//! Every IFD of the main chain describes itself in PerkinElmer-QPI XML in
//! its ImageDescription: `ImageType` (`FullResolution`, `ReducedResolution`,
//! `Thumbnail`, `Overview` or `Label`), the channel's `Name` (the fluor, or
//! the marker), `Biomarker` (the marker, when the name is the fluor),
//! `Color` (`R,G,B`) and the `Objective`. The pixel size is in the TIFF
//! resolution tags.
//!
//! [`document`] turns the descriptions into the [`OmeDocument`] the OME-TIFF
//! reader works from: an image of the full-resolution IFDs (one channel
//! each), one image per reduced level (its IFDs in channel order, grouped by
//! size), and the thumbnail, the overview (as `macro`) and the label. The
//! rest of the reader (pyramid, channel compositing, associated images) is
//! shared.

use super::IfdTable;
use super::tiff::{Ifd, SampleType};
use super::xml::{DimensionOrder, Objective, OmeChannel, OmeDocument, OmeImage, Pixels, TiffData};
use crate::{Error, Result};
use std::sync::Arc;
use tracing::warn;

/// The root element of the descriptions.
const ROOT: &str = "PerkinElmer-QPI-ImageDescription";
/// Top-level fields read from one description at most.
const MAX_FIELDS: usize = 1024;
/// IFDs of the main chain looked at at most.
const MAX_IFDS: usize = 8192;

/// File name endings of QPTIFF slides.
pub(crate) const SUFFIXES: &[&str] = &[".qptiff"];

/// Whether an ImageDescription (or its start) is PerkinElmer-QPI XML.
pub(crate) fn looks_like_qptiff(description: &str) -> bool {
    description.contains(ROOT)
}

/// The top-level fields of one IFD's description, in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Description {
    pub fields: Vec<(String, String)>,
}

impl Description {
    /// The fields of the description's root element, or `None` when it is
    /// not PerkinElmer-QPI XML. Nested elements are kept as their text.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let start = text.find(&format!("<{ROOT}"))?;
        let body = &text[start..];
        let mut rest = &body[body.find('>')? + 1..];
        let mut fields = Vec::new();
        while fields.len() < MAX_FIELDS {
            let Some(open) = rest.find('<') else {
                break;
            };
            rest = &rest[open..];
            if rest.starts_with("</") {
                // The root element ends.
                break;
            }
            if let Some(after) = rest.strip_prefix("<!--") {
                rest = after.find("-->").map_or("", |end| &after[end + 3..]);
                continue;
            }
            if rest.starts_with("<?") || rest.starts_with("<!") {
                rest = rest.find('>').map_or("", |end| &rest[end + 1..]);
                continue;
            }
            let tag_end = rest.find('>')?;
            let name: String = rest[1..tag_end]
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '/')
                .collect();
            if name.is_empty() {
                break;
            }
            if rest[..tag_end].ends_with('/') {
                fields.push((name, String::new()));
                rest = &rest[tag_end + 1..];
                continue;
            }
            let content = &rest[tag_end + 1..];
            let close = format!("</{name}>");
            let Some(end) = content.find(&close) else {
                break;
            };
            fields.push((name, unescape(content[..end].trim())));
            rest = &content[end + close.len()..];
        }
        Some(Self { fields })
    }

    /// The first field called `key` (case-insensitively), if it has text.
    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(key))
            .map(|(_, value)| value.as_str())
            .filter(|value| !value.is_empty())
    }

    fn image_type(&self) -> &str {
        self.get("ImageType").unwrap_or_default()
    }

    /// `Color` as written: `R,G,B` (0–255).
    fn color(&self) -> Option<[u8; 3]> {
        let parts: Vec<u8> = self
            .get("Color")?
            .split(',')
            .map(|part| part.trim().parse::<u8>())
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        match parts.as_slice() {
            [r, g, b] | [r, g, b, _] => Some([*r, *g, *b]),
            _ => None,
        }
    }

    /// The objective's magnification: `20x`, `20X` or `20`.
    fn objective(&self) -> Option<f64> {
        let text = self.get("Objective")?.trim();
        let number = text.trim_end_matches(['x', 'X']).trim();
        number
            .parse::<f64>()
            .ok()
            .filter(|power| power.is_finite() && *power > 0.0 && *power <= 1000.0)
    }

    /// The channel's name: the marker when there is one, else the name.
    fn channel_name(&self) -> Option<&str> {
        self.get("Biomarker")
            .filter(|marker| !matches!(*marker, "--" | "None" | "none"))
            .or_else(|| self.get("Name"))
    }

    /// The fluor, when the name is one and the marker names the channel.
    fn fluor(&self) -> Option<&str> {
        let name = self.get("Name")?;
        (self.channel_name() != Some(name)).then_some(name)
    }
}

/// `&lt;` and the other entities XML text may hold.
fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let entity = &rest[at..];
        let Some(end) = entity.find(';').filter(|end| *end <= 10) else {
            out.push('&');
            rest = &entity[1..];
            continue;
        };
        let decoded = match &entity[1..end] {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            code => code
                .strip_prefix("#x")
                .or_else(|| code.strip_prefix("#X"))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| code.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &entity[end + 1..];
            }
            None => {
                out.push('&');
                rest = &entity[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// One IFD of the main chain with its description.
struct Entry {
    index: u64,
    ifd: Arc<Ifd>,
    description: Description,
}

fn pixel_type(sample_type: SampleType) -> &'static str {
    match sample_type {
        SampleType::U8 => "uint8",
        SampleType::U16 => "uint16",
        SampleType::U32 => "uint32",
        SampleType::I8 => "int8",
        SampleType::I16 => "int16",
        SampleType::I32 => "int32",
        SampleType::F32 => "float",
        SampleType::F64 => "double",
    }
}

/// An OME image of `entries`, one channel each (or one RGB channel).
fn image(id: &str, name: &str, entries: &[&Entry], channels: &[OmeChannel]) -> OmeImage {
    let first = &entries[0].ifd;
    let rgb = first.samples_per_pixel >= 3;
    OmeImage {
        id: id.to_string(),
        name: Some(name.to_string()),
        acquisition_date: None,
        description: None,
        objective_settings: None,
        instrument: None,
        pixels: Pixels {
            size_x: u64::from(first.width),
            size_y: u64::from(first.height),
            size_z: 1,
            size_c: if rgb { 3 } else { entries.len() as u64 },
            size_t: 1,
            pixel_type: Some(pixel_type(first.sample_type).to_string()),
            dimension_order: DimensionOrder::Xyczt,
            physical_size_x: None,
            physical_size_y: None,
            physical_size_z: None,
            significant_bits: None,
            channels: channels.to_vec(),
            tiff_data: entries
                .iter()
                .enumerate()
                .map(|(c, entry)| TiffData {
                    ifd: Some(entry.index),
                    plane_count: Some(1),
                    first_c: Some(c as u64),
                    first_z: Some(0),
                    first_t: Some(0),
                    ..TiffData::default()
                })
                .collect(),
        },
    }
}

/// The OME document of a QPTIFF and the description of its first
/// full-resolution image.
pub(super) fn document(
    table: &mut IfdTable<'_>,
    file_name: &str,
) -> Result<(OmeDocument, Description)> {
    let mut entries = Vec::new();
    for index in 0..table.chain.len().min(MAX_IFDS) as u64 {
        let ifd = table.by_index(index)?;
        let Some(description) = ifd.description.as_deref().and_then(Description::parse) else {
            continue;
        };
        entries.push(Entry {
            index,
            ifd,
            description,
        });
    }
    let of_type = |kind: &str| -> Vec<&Entry> {
        entries
            .iter()
            .filter(|entry| entry.description.image_type().eq_ignore_ascii_case(kind))
            .collect()
    };
    let mut full = of_type("FullResolution");
    let Some(first) = full.first().map(|entry| Arc::clone(&entry.ifd)) else {
        return Err(Error::UnsupportedFormat(format!(
            "{file_name}: the QPTIFF has no full-resolution image"
        )));
    };
    let rgb = first.samples_per_pixel >= 3;
    let same = |ifd: &Ifd| {
        (
            ifd.width,
            ifd.height,
            ifd.sample_type,
            ifd.samples_per_pixel,
        ) == (
            first.width,
            first.height,
            first.sample_type,
            first.samples_per_pixel,
        )
    };
    if let Some(odd) = full.iter().position(|entry| !same(&entry.ifd)) {
        warn!(
            "{file_name}: full-resolution image {} differs from the first; showing {odd} channels",
            odd + 1
        );
        full.truncate(odd);
    }
    if rgb {
        full.truncate(1);
    }
    let channel_count = full.len();
    let channels: Vec<OmeChannel> = full
        .iter()
        .enumerate()
        .map(|(c, entry)| OmeChannel {
            id: Some(format!("Channel:0:{c}")),
            name: Some(
                entry
                    .description
                    .channel_name()
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        if rgb {
                            "Brightfield".to_string()
                        } else {
                            format!("Channel {}", c + 1)
                        }
                    }),
            ),
            color: entry.description.color().map(|[r, g, b]| [r, g, b, 255]),
            samples_per_pixel: Some(if rgb { 3 } else { 1 }),
            fluor: entry.description.fluor().map(str::to_string),
            ..OmeChannel::default()
        })
        .collect();

    let mut images = vec![image("Image:0", "Full resolution", &full, &channels)];
    let objective = full[0].description.objective();
    if objective.is_some() {
        images[0].objective_settings = Some("Objective:0".to_string());
    }

    // Reduced levels: runs of one IFD per channel, the same size within a
    // run, each run smaller than the one before.
    let reduced = of_type("ReducedResolution");
    let mut previous = (first.width, first.height);
    for (level, group) in (1..).zip(reduced.chunks(channel_count)) {
        let size = (group[0].ifd.width, group[0].ifd.height);
        let consistent = group.len() == channel_count
            && group.iter().all(|entry| {
                (
                    entry.ifd.width,
                    entry.ifd.height,
                    entry.ifd.sample_type,
                    entry.ifd.samples_per_pixel,
                ) == (size.0, size.1, first.sample_type, first.samples_per_pixel)
            });
        if !consistent || size.0 >= previous.0 || size.1 >= previous.1 {
            warn!("{file_name}: stopping the pyramid at reduced level {level}");
            break;
        }
        images.push(image(
            &format!("Image:{level}"),
            &format!("Reduced level {level}"),
            group,
            &channels,
        ));
        previous = size;
    }

    // The thumbnail, overview and label, when they are RGB.
    for (kind, name) in [
        ("Thumbnail", "thumbnail"),
        ("Overview", "macro"),
        ("Label", "label"),
    ] {
        if let Some(entry) = of_type(kind)
            .into_iter()
            .find(|entry| entry.ifd.samples_per_pixel >= 3)
        {
            let rgb_channel = OmeChannel {
                samples_per_pixel: Some(3),
                ..OmeChannel::default()
            };
            let id = format!("Image:{}", images.len());
            images.push(image(&id, name, &[entry], &[rgb_channel]));
        }
    }

    let description = full[0].description.clone();
    let doc = OmeDocument {
        creator: description.get("AcquisitionSoftware").map(str::to_string),
        uuid: None,
        images,
        objectives: objective
            .map(|power| {
                vec![Objective {
                    id: "Objective:0".to_string(),
                    nominal_magnification: Some(power),
                    ..Objective::default()
                }]
            })
            .unwrap_or_default(),
    };
    Ok((doc, description))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_parse_their_fields() {
        let text = "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n\
            <PerkinElmer-QPI-ImageDescription>\r\n\
              <DescriptionVersion>2</DescriptionVersion>\r\n\
              <!-- a comment -->\r\n\
              <ImageType>FullResolution</ImageType>\r\n\
              <Name>Opal 520</Name>\r\n\
              <Biomarker>CD8</Biomarker>\r\n\
              <Color>0,255,0</Color>\r\n\
              <Objective>20x</Objective>\r\n\
              <Empty/>\r\n\
              <ScanProfile><root><x>1</x></root></ScanProfile>\r\n\
              <Notes>a &lt;b&gt; &amp; &#233;</Notes>\r\n\
            </PerkinElmer-QPI-ImageDescription>";
        assert!(looks_like_qptiff(text));
        let description = Description::parse(text).unwrap();
        assert_eq!(description.image_type(), "FullResolution");
        assert_eq!(description.channel_name(), Some("CD8"));
        assert_eq!(description.fluor(), Some("Opal 520"));
        assert_eq!(description.color(), Some([0, 255, 0]));
        assert_eq!(description.objective(), Some(20.0));
        assert_eq!(description.get("Notes"), Some("a <b> & é"));
        assert_eq!(
            description.get("ScanProfile"),
            Some("<root><x>1</x></root>")
        );
        assert_eq!(description.get("Empty"), None);
        assert_eq!(description.get("DescriptionVersion"), Some("2"));

        // Without a marker the name is the channel's, and not a fluor.
        let plain = Description::parse(
            "<PerkinElmer-QPI-ImageDescription><Name>DAPI</Name>\
             <Biomarker>--</Biomarker></PerkinElmer-QPI-ImageDescription>",
        )
        .unwrap();
        assert_eq!(plain.channel_name(), Some("DAPI"));
        assert_eq!(plain.fluor(), None);
        assert!(Description::parse("<OME/>").is_none());
        // Truncated or odd text ends the fields without panicking.
        for cut in 0..text.len() {
            if text.is_char_boundary(cut) {
                let _ = Description::parse(&text[..cut]);
            }
        }
        assert_eq!(unescape("&bogus; & &#xZZ;"), "&bogus; & &#xZZ;");
    }
}
