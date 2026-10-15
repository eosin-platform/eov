//! OME-XML, the metadata an OME-TIFF stores in its first ImageDescription.
//!
//! Only what the reader uses is parsed: images and their pixel geometry,
//! channels, the `TiffData` plane-to-IFD mapping, physical pixel sizes and
//! objectives. Elements are matched by local name, so the schema versions
//! since 2010 (which share this structure) all parse. The parser refuses
//! DTDs and bounds the node count, since the XML is untrusted input.

use std::collections::HashMap;

/// Most XML nodes parsed (a few hundred thousand cover the largest
/// cyclic-immunofluorescence files).
const MAX_XML_NODES: u32 = 4_000_000;

/// A length with its unit as written in the file.
#[derive(Debug, Clone, PartialEq)]
pub struct Length {
    pub value: f64,
    pub unit: String,
}

impl Length {
    /// The length in micrometres, when its unit is a length unit.
    pub fn micrometres(&self) -> Option<f64> {
        let factor = match self.unit.trim() {
            "m" => 1e6,
            "dm" => 1e5,
            "cm" => 1e4,
            "mm" => 1e3,
            // Micro sign (U+00B5) and Greek mu (U+03BC).
            "\u{b5}m" | "\u{3bc}m" | "um" | "micron" | "microns" | "micrometer" | "micrometre" => {
                1.0
            }
            "nm" => 1e-3,
            "pm" => 1e-6,
            // Latin capital A with ring (U+00C5) and the angstrom sign
            // (U+212B).
            "\u{c5}" | "\u{212b}" => 1e-4,
            "in" => 25_400.0,
            "thou" | "mil" => 25.4,
            "ft" => 304_800.0,
            _ => return None,
        };
        let um = self.value * factor;
        (um.is_finite() && um > 0.0).then_some(um)
    }

    /// The length in nanometres (wavelengths).
    pub fn nanometres(&self) -> Option<f64> {
        self.micrometres().map(|um| um * 1e3)
    }

    /// As people read it: `"0.325 µm"`.
    pub fn display(&self) -> String {
        format!("{} {}", crate::format_significant(self.value, 6), self.unit)
    }
}

/// The order of the Z, C and T dimensions of planes, fastest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DimensionOrder {
    Xyzct,
    Xyztc,
    Xyctz,
    Xyczt,
    Xytcz,
    Xytzc,
}

impl DimensionOrder {
    fn parse(text: &str) -> Option<Self> {
        Some(match text.trim() {
            "XYZCT" => Self::Xyzct,
            "XYZTC" => Self::Xyztc,
            "XYCTZ" => Self::Xyctz,
            "XYCZT" => Self::Xyczt,
            "XYTCZ" => Self::Xytcz,
            "XYTZC" => Self::Xytzc,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Xyzct => "XYZCT",
            Self::Xyztc => "XYZTC",
            Self::Xyctz => "XYCTZ",
            Self::Xyczt => "XYCZT",
            Self::Xytcz => "XYTCZ",
            Self::Xytzc => "XYTZC",
        }
    }

    /// The index of plane (`c`, `z`, `t`) among `sizes` = (C, Z, T).
    pub(crate) fn plane_index(
        self,
        (c, z, t): (u64, u64, u64),
        (sc, sz, st): (u64, u64, u64),
    ) -> u64 {
        // The fastest dimension and its size, the middle one and its size,
        // and the slowest.
        let (a, size_a, b, size_b, d) = match self {
            Self::Xyzct => (z, sz, c, sc, t),
            Self::Xyztc => (z, sz, t, st, c),
            Self::Xyctz => (c, sc, t, st, z),
            Self::Xyczt => (c, sc, z, sz, t),
            Self::Xytcz => (t, st, c, sc, z),
            Self::Xytzc => (t, st, z, sz, c),
        };
        // Sizes come from the file: saturate rather than overflow.
        a.saturating_add(size_a.saturating_mul(b.saturating_add(size_b.saturating_mul(d))))
    }
}

/// An `Objective` of an `Instrument`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Objective {
    pub id: String,
    pub nominal_magnification: Option<f64>,
    pub calibrated_magnification: Option<f64>,
    pub model: Option<String>,
    pub manufacturer: Option<String>,
    pub lens_na: Option<f64>,
    pub immersion: Option<String>,
    /// The `Instrument` it belongs to.
    pub instrument: Option<String>,
}

/// A `Channel` of `Pixels`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OmeChannel {
    pub id: Option<String>,
    pub name: Option<String>,
    /// `Color` as RGBA.
    pub color: Option<[u8; 4]>,
    pub samples_per_pixel: Option<u32>,
    pub fluor: Option<String>,
    pub emission: Option<Length>,
    pub excitation: Option<Length>,
    pub contrast_method: Option<String>,
    pub illumination_type: Option<String>,
    pub acquisition_mode: Option<String>,
}

/// A `TiffData` element: which IFDs hold which planes.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TiffData {
    pub ifd: Option<u64>,
    pub plane_count: Option<u64>,
    pub first_c: Option<u64>,
    pub first_z: Option<u64>,
    pub first_t: Option<u64>,
    /// `UUID@FileName`: the planes are in this file of a multi-file set.
    pub file_name: Option<String>,
    pub uuid: Option<String>,
}

/// The `Pixels` of an image.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Pixels {
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
    pub size_c: u64,
    pub size_t: u64,
    pub pixel_type: Option<String>,
    pub dimension_order: DimensionOrder,
    pub physical_size_x: Option<Length>,
    pub physical_size_y: Option<Length>,
    pub physical_size_z: Option<Length>,
    pub significant_bits: Option<u32>,
    pub channels: Vec<OmeChannel>,
    pub tiff_data: Vec<TiffData>,
}

impl Pixels {
    /// Planes per (Z, T): the number of `Channel` elements, or `SizeC`
    /// divided by the samples of an RGB channel.
    pub(crate) fn effective_size_c(&self) -> u64 {
        if self.channels.is_empty() {
            return self.size_c.max(1);
        }
        let samples = u64::from(self.channels[0].samples_per_pixel.unwrap_or(1).max(1));
        if self.channels.len() as u64 * samples == self.size_c || samples > 1 {
            self.channels.len() as u64
        } else {
            self.size_c.max(self.channels.len() as u64)
        }
    }

    /// Total planes (`C × Z × T` with the effective C).
    pub(crate) fn plane_count(&self) -> u64 {
        self.effective_size_c()
            .saturating_mul(self.size_z.max(1))
            .saturating_mul(self.size_t.max(1))
    }

    /// Plane index → IFD index (of the main chain) for the planes stored
    /// in this file. `chain_len` is the number of IFDs in the file;
    /// `own_file` and `own_uuid` identify it in multi-file sets.
    ///
    /// Planes that `TiffData` places in another file are left out.
    pub(crate) fn plane_ifds(
        &self,
        chain_len: u64,
        own_file: &str,
        own_uuid: Option<&str>,
    ) -> HashMap<u64, u64> {
        let planes = self.plane_count();
        let sizes = (
            self.effective_size_c(),
            self.size_z.max(1),
            self.size_t.max(1),
        );
        let mut map = HashMap::new();
        if self.tiff_data.is_empty() {
            for plane in 0..planes.min(chain_len) {
                map.insert(plane, plane);
            }
            return map;
        }
        for data in &self.tiff_data {
            let elsewhere = data.file_name.as_deref().is_some_and(|name| {
                let name = std::path::Path::new(name)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                !name.eq_ignore_ascii_case(own_file)
                    && own_uuid.is_none_or(|uuid| data.uuid.as_deref() != Some(uuid))
            });
            if elsewhere {
                continue;
            }
            let first_ifd = data.ifd.unwrap_or(0);
            let count = data
                .plane_count
                .unwrap_or(if data.ifd.is_some() { 1 } else { chain_len });
            let start = self.dimension_order.plane_index(
                (
                    data.first_c.unwrap_or(0),
                    data.first_z.unwrap_or(0),
                    data.first_t.unwrap_or(0),
                ),
                sizes,
            );
            for k in 0..count.min(planes) {
                let (plane, ifd) = (start.saturating_add(k), first_ifd.saturating_add(k));
                if plane >= planes || ifd >= chain_len {
                    break;
                }
                map.entry(plane).or_insert(ifd);
            }
        }
        map
    }
}

/// An `Image`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OmeImage {
    pub id: String,
    pub name: Option<String>,
    pub acquisition_date: Option<String>,
    pub description: Option<String>,
    /// `ObjectiveSettings@ID`: the objective used.
    pub objective_settings: Option<String>,
    pub instrument: Option<String>,
    pub pixels: Pixels,
}

/// A parsed OME-XML document.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OmeDocument {
    pub creator: Option<String>,
    pub uuid: Option<String>,
    pub images: Vec<OmeImage>,
    pub objectives: Vec<Objective>,
}

impl OmeDocument {
    /// The objective used for `image`: the one its `ObjectiveSettings`
    /// names, else the only one of its instrument (or of the document).
    pub(crate) fn objective_of(&self, image: &OmeImage) -> Option<&Objective> {
        if let Some(id) = &image.objective_settings
            && let Some(objective) = self.objectives.iter().find(|o| &o.id == id)
        {
            return Some(objective);
        }
        let of_instrument: Vec<&Objective> = match &image.instrument {
            Some(instrument) => self
                .objectives
                .iter()
                .filter(|o| o.instrument.as_ref() == Some(instrument))
                .collect(),
            None => self.objectives.iter().collect(),
        };
        match of_instrument.as_slice() {
            [only] => Some(only),
            _ => None,
        }
    }
}

/// Whether the start of an ImageDescription is OME-XML: an `OME` root
/// element, which the `<?xml` declaration and comments may precede.
pub(crate) fn looks_like_ome(description_start: &str) -> bool {
    let text = description_start
        .trim_start_matches('\u{feff}')
        .trim_start();
    if !text.starts_with('<') {
        return false;
    }
    // The root element's name: the first tag that is not a declaration,
    // comment or processing instruction.
    let mut rest = text;
    loop {
        let Some(start) = rest.find('<') else {
            return false;
        };
        rest = &rest[start + 1..];
        if rest.starts_with('?') || rest.starts_with('!') {
            continue;
        }
        let name: String = rest
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '>' && *c != '/')
            .collect();
        let local = name.rsplit(':').next().unwrap_or("");
        return local == "OME";
    }
}

fn attr<'a>(node: roxmltree::Node<'a, '_>, name: &str) -> Option<&'a str> {
    node.attribute(name)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn text_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Option<String> {
    attr(node, name).map(str::to_string)
}

fn number<T: std::str::FromStr>(node: roxmltree::Node<'_, '_>, name: &str) -> Option<T> {
    attr(node, name)?.parse().ok()
}

fn positive_f64(node: roxmltree::Node<'_, '_>, name: &str) -> Option<f64> {
    number::<f64>(node, name).filter(|v| v.is_finite() && *v > 0.0)
}

fn length(node: roxmltree::Node<'_, '_>, name: &str, default_unit: &str) -> Option<Length> {
    let value = positive_f64(node, name)?;
    let unit = attr(node, &format!("{name}Unit")).unwrap_or(default_unit);
    Some(Length {
        value,
        unit: unit.to_string(),
    })
}

fn children<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    name: &'static str,
) -> impl Iterator<Item = roxmltree::Node<'a, 'input>> + 'a {
    node.children()
        .filter(move |child| child.is_element() && child.tag_name().name() == name)
}

fn child<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    name: &'static str,
) -> Option<roxmltree::Node<'a, 'input>> {
    children(node, name).next()
}

fn child_text(node: roxmltree::Node<'_, '_>, name: &'static str) -> Option<String> {
    child(node, name)
        .and_then(|c| c.text())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

/// `Color` (a signed 32-bit RGBA integer) as RGBA bytes.
fn parse_color(text: &str) -> Option<[u8; 4]> {
    let value: i64 = text.trim().parse().ok()?;
    if !(i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&value) {
        return None;
    }
    Some((value as u32).to_be_bytes())
}

fn parse_channel(node: roxmltree::Node<'_, '_>) -> OmeChannel {
    OmeChannel {
        id: text_attr(node, "ID"),
        name: text_attr(node, "Name"),
        color: attr(node, "Color").and_then(parse_color),
        samples_per_pixel: number(node, "SamplesPerPixel"),
        fluor: text_attr(node, "Fluor"),
        emission: length(node, "EmissionWavelength", "nm"),
        excitation: length(node, "ExcitationWavelength", "nm"),
        contrast_method: text_attr(node, "ContrastMethod"),
        illumination_type: text_attr(node, "IlluminationType"),
        acquisition_mode: text_attr(node, "AcquisitionMode"),
    }
}

fn parse_tiff_data(node: roxmltree::Node<'_, '_>) -> TiffData {
    let uuid = child(node, "UUID");
    TiffData {
        ifd: number(node, "IFD"),
        plane_count: number(node, "PlaneCount"),
        first_c: number(node, "FirstC"),
        first_z: number(node, "FirstZ"),
        first_t: number(node, "FirstT"),
        file_name: uuid.and_then(|u| text_attr(u, "FileName")),
        uuid: uuid
            .and_then(|u| u.text())
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string),
    }
}

fn parse_pixels(node: roxmltree::Node<'_, '_>) -> Result<Pixels, String> {
    let size = |name: &str| -> Result<u64, String> {
        number::<u64>(node, name)
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("Pixels has no valid {name}"))
    };
    let optional_size = |name: &str| number::<u64>(node, name).filter(|&v| v > 0).unwrap_or(1);
    Ok(Pixels {
        size_x: size("SizeX")?,
        size_y: size("SizeY")?,
        size_z: optional_size("SizeZ"),
        size_c: optional_size("SizeC"),
        size_t: optional_size("SizeT"),
        pixel_type: text_attr(node, "Type"),
        dimension_order: attr(node, "DimensionOrder")
            .and_then(DimensionOrder::parse)
            .unwrap_or(DimensionOrder::Xyczt),
        physical_size_x: length(node, "PhysicalSizeX", "µm"),
        physical_size_y: length(node, "PhysicalSizeY", "µm"),
        physical_size_z: length(node, "PhysicalSizeZ", "µm"),
        significant_bits: number(node, "SignificantBits"),
        channels: children(node, "Channel").map(parse_channel).collect(),
        tiff_data: children(node, "TiffData").map(parse_tiff_data).collect(),
    })
}

/// Parses an OME-XML document.
pub(crate) fn parse(text: &str) -> Result<OmeDocument, String> {
    let options = roxmltree::ParsingOptions {
        allow_dtd: false,
        nodes_limit: MAX_XML_NODES,
        ..Default::default()
    };
    let document = roxmltree::Document::parse_with_options(text, options)
        .map_err(|error| format!("invalid OME-XML: {error}"))?;
    let root = document.root_element();
    if root.tag_name().name() != "OME" {
        return Err(format!(
            "the description is XML, but its root is <{}>, not <OME>",
            root.tag_name().name()
        ));
    }
    let mut objectives = Vec::new();
    for instrument in children(root, "Instrument") {
        let instrument_id = text_attr(instrument, "ID");
        for objective in children(instrument, "Objective") {
            objectives.push(Objective {
                id: text_attr(objective, "ID").unwrap_or_default(),
                nominal_magnification: positive_f64(objective, "NominalMagnification"),
                calibrated_magnification: positive_f64(objective, "CalibratedMagnification"),
                model: text_attr(objective, "Model"),
                manufacturer: text_attr(objective, "Manufacturer"),
                lens_na: positive_f64(objective, "LensNA"),
                immersion: text_attr(objective, "Immersion"),
                instrument: instrument_id.clone(),
            });
        }
    }
    let mut images = Vec::new();
    for image in children(root, "Image") {
        let Some(pixels) = child(image, "Pixels") else {
            continue;
        };
        images.push(OmeImage {
            id: text_attr(image, "ID").unwrap_or_default(),
            name: text_attr(image, "Name"),
            acquisition_date: child_text(image, "AcquisitionDate"),
            description: child_text(image, "Description"),
            objective_settings: child(image, "ObjectiveSettings").and_then(|o| text_attr(o, "ID")),
            instrument: child(image, "InstrumentRef").and_then(|i| text_attr(i, "ID")),
            pixels: parse_pixels(pixels)?,
        });
    }
    if images.is_empty() {
        return Err("the OME-XML describes no image".to_string());
    }
    Ok(OmeDocument {
        creator: text_attr(root, "Creator"),
        uuid: text_attr(root, "UUID"),
        images,
        objectives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!-- Written by a test -->
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     Creator="test 1.0" UUID="urn:uuid:1234">
  <Instrument ID="Instrument:0">
    <Objective ID="Objective:0:0" NominalMagnification="20.0" LensNA="0.75" Model="Plan-Apo"/>
  </Instrument>
  <Image ID="Image:0" Name="slide">
    <AcquisitionDate>2024-05-06T07:08:09</AcquisitionDate>
    <InstrumentRef ID="Instrument:0"/>
    <ObjectiveSettings ID="Objective:0:0"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="300" SizeY="200"
            SizeZ="1" SizeC="3" SizeT="1" PhysicalSizeX="325" PhysicalSizeXUnit="nm"
            PhysicalSizeY="0.325">
      <Channel ID="Channel:0:0" Name="DAPI" Color="65535" SamplesPerPixel="1"
               EmissionWavelength="461" Fluor="DAPI"/>
      <Channel ID="Channel:0:1" Name="CD45" Color="-16776961" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:2" Name="Pan-CK" SamplesPerPixel="1"/>
      <TiffData IFD="0" PlaneCount="3"/>
    </Pixels>
  </Image>
</OME>"#;

    #[test]
    fn parses_images_channels_sizes_and_objectives() {
        assert!(looks_like_ome(SAMPLE));
        let doc = parse(SAMPLE).unwrap();
        assert_eq!(doc.creator.as_deref(), Some("test 1.0"));
        let image = &doc.images[0];
        assert_eq!(image.name.as_deref(), Some("slide"));
        assert_eq!(
            image.acquisition_date.as_deref(),
            Some("2024-05-06T07:08:09")
        );
        let pixels = &image.pixels;
        assert_eq!((pixels.size_x, pixels.size_y, pixels.size_c), (300, 200, 3));
        assert_eq!(pixels.pixel_type.as_deref(), Some("uint16"));
        let x = pixels.physical_size_x.as_ref().unwrap();
        assert!((x.micrometres().unwrap() - 0.325).abs() < 1e-12);
        // No unit: the schema's default, µm.
        assert_eq!(
            pixels.physical_size_y.as_ref().unwrap().micrometres(),
            Some(0.325)
        );
        assert_eq!(pixels.channels.len(), 3);
        assert_eq!(pixels.channels[0].color, Some([0, 0, 255, 255]));
        assert_eq!(pixels.channels[1].color, Some([255, 0, 0, 255]));
        assert_eq!(pixels.channels[2].color, None);
        assert_eq!(
            pixels.channels[0].emission.as_ref().unwrap().nanometres(),
            Some(461.0)
        );
        let objective = doc.objective_of(image).unwrap();
        assert_eq!(objective.nominal_magnification, Some(20.0));
        assert_eq!(pixels.plane_ifds(3, "x.ome.tif", None).len(), 3);
    }

    #[test]
    fn plane_mapping_follows_tiff_data_and_dimension_order() {
        let mut pixels = parse(SAMPLE).unwrap().images.remove(0).pixels;
        pixels.size_z = 2;
        pixels.dimension_order = DimensionOrder::Xyzct;
        // No TiffData: plane i is IFD i.
        pixels.tiff_data.clear();
        let map = pixels.plane_ifds(6, "x", None);
        assert_eq!(map.len(), 6);
        // XYZCT: (c=1, z=0) is plane 2.
        let order = pixels.dimension_order;
        assert_eq!(order.plane_index((1, 0, 0), (3, 2, 1)), 2);
        assert_eq!(DimensionOrder::Xyczt.plane_index((1, 0, 0), (3, 2, 1)), 1);
        assert_eq!(DimensionOrder::Xyczt.plane_index((2, 1, 0), (3, 2, 1)), 5);
        assert_eq!(DimensionOrder::Xytcz.plane_index((1, 1, 0), (3, 2, 1)), 4);

        // One TiffData per plane, the second channel in another file.
        pixels.size_z = 1;
        pixels.tiff_data = vec![
            TiffData {
                ifd: Some(4),
                first_c: Some(0),
                ..TiffData::default()
            },
            TiffData {
                ifd: Some(0),
                first_c: Some(1),
                file_name: Some("other.ome.tif".into()),
                ..TiffData::default()
            },
            TiffData {
                ifd: Some(5),
                first_c: Some(2),
                file_name: Some("dir/X.OME.TIF".into()),
                ..TiffData::default()
            },
        ];
        let map = pixels.plane_ifds(6, "x.ome.tif", None);
        assert_eq!(map.get(&0), Some(&4));
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), Some(&5));
        // IFDs beyond the file are ignored.
        pixels.tiff_data[0].ifd = Some(99);
        assert_eq!(pixels.plane_ifds(6, "x.ome.tif", None).get(&0), None);
    }

    #[test]
    fn units_convert_to_micrometres() {
        let len = |value: f64, unit: &str| Length {
            value,
            unit: unit.to_string(),
        };
        assert_eq!(len(0.5, "µm").micrometres(), Some(0.5));
        assert_eq!(len(0.5, "\u{3bc}m").micrometres(), Some(0.5));
        assert_eq!(len(500.0, "nm").micrometres(), Some(0.5));
        assert_eq!(len(0.0005, "mm").micrometres(), Some(0.5));
        assert_eq!(len(5000.0, "\u{c5}").micrometres(), Some(0.5));
        assert_eq!(len(5000.0, "\u{212b}").micrometres(), Some(0.5));
        assert_eq!(len(1.0, "pixel").micrometres(), None);
        assert_eq!(len(1.0, "parsec?").micrometres(), None);
    }

    #[test]
    fn rejects_documents_that_are_not_ome() {
        assert!(!looks_like_ome("Aperio Image Library v12"));
        assert!(!looks_like_ome(
            "<?xml version=\"1.0\"?><DataObject ObjectType=\"DPUfsImport\">"
        ));
        assert!(looks_like_ome(
            "<?xml version=\"1.0\"?><ome:OME xmlns:ome=\"x\">"
        ));
        assert!(parse("<DataObject/>").is_err());
        assert!(parse("<OME>").is_err(), "unclosed");
        assert!(parse("<OME/>").is_err(), "no image");
        assert!(
            parse(r#"<OME><Image ID="i"><Pixels SizeX="0" SizeY="1"/></Image></OME>"#).is_err()
        );
        // DTDs (entity expansion) are refused.
        let dtd = r#"<?xml version="1.0"?><!DOCTYPE OME [<!ENTITY a "aaaa">]><OME>&a;</OME>"#;
        assert!(parse(dtd).is_err());
    }
}
