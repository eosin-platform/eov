//! A writer of small synthetic OME-TIFF files, for tests of the OME-TIFF
//! reader and of everything built on multichannel slides.
//!
//! [`write_ome_tiff`] writes the layouts real files use: tiled or stripped
//! planes, one per channel or RGB, pyramids as SubIFDs, as separate images
//! or as reduced-resolution IFDs in the main chain, TIFF or BigTIFF in
//! either byte order, and every compression the reader decodes. Values come
//! from a function of (channel, x, y) at level 0; lower levels sample it at
//! the matching level-0 pixel, so tests can predict every pixel.

use std::io::Write;
use std::path::Path;

/// The numeric type of the samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmeSample {
    U8,
    U16,
    I16,
    F32,
    /// Three 8-bit samples per pixel, photometric RGB (or YCbCr for JPEG).
    Rgb8,
}

impl OmeSample {
    fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::Rgb8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::F32 => 4,
        }
    }

    fn ome_type(self) -> &'static str {
        match self {
            Self::U8 | Self::Rgb8 => "uint8",
            Self::U16 => "uint16",
            Self::I16 => "int16",
            Self::F32 => "float",
        }
    }

    fn sample_format(self) -> u16 {
        match self {
            Self::U8 | Self::U16 | Self::Rgb8 => 1,
            Self::I16 => 2,
            Self::F32 => 3,
        }
    }
}

/// The compression of tiles and strips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmeCompression {
    None,
    Lzw,
    Deflate,
    Zstd,
    PackBits,
    /// Self-contained JPEG streams per tile.
    Jpeg,
    /// Abbreviated JPEG streams with the tables in `JPEGTables`.
    JpegTables,
    /// A code the reader does not decode (JPEG 2000).
    Jpeg2000,
}

/// Where the pyramid levels go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmePyramid {
    SubIfds,
    /// One OME `Image` per level.
    Images,
    /// Reduced-resolution IFDs (NewSubfileType 1) in the main chain.
    ReducedIfds,
}

/// One channel.
#[derive(Debug, Clone, PartialEq)]
pub struct OmeChannelSpec {
    pub name: String,
    /// `Channel@Color` as RGB (alpha 255).
    pub color: Option<[u8; 3]>,
    pub emission_nm: Option<f64>,
}

impl OmeChannelSpec {
    pub fn new(name: &str, color: Option<[u8; 3]>) -> Self {
        Self {
            name: name.to_string(),
            color,
            emission_nm: None,
        }
    }
}

/// An extra RGB image stored with the slide (`label`, `macro`).
#[derive(Debug, Clone, PartialEq)]
pub struct OmeExtraImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub color: [u8; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct OmeTiffSpec {
    pub width: u32,
    pub height: u32,
    /// Tile size; `None` writes strips of `rows_per_strip` rows.
    pub tile: Option<u32>,
    pub rows_per_strip: u32,
    /// Levels including level 0, each half the size of the previous.
    pub levels: u32,
    pub pyramid: OmePyramid,
    pub sample: OmeSample,
    /// Channels; ignored for [`OmeSample::Rgb8`].
    pub channels: Vec<OmeChannelSpec>,
    pub compression: OmeCompression,
    /// Horizontal differencing for integers, the floating-point predictor
    /// for floats.
    pub predictor: bool,
    /// RGB samples as separate planes (PlanarConfiguration 2).
    pub planar: bool,
    pub big_tiff: bool,
    pub big_endian: bool,
    /// `PhysicalSizeX`/`Y` and their unit.
    pub physical_size: Option<(f64, String)>,
    pub objective: Option<f64>,
    pub extra_images: Vec<OmeExtraImage>,
    /// Tiles left out (offset 0) as (level, channel, tile x, tile y).
    pub missing_tiles: Vec<(u32, usize, u32, u32)>,
    /// Tiles whose data is garbage, so decoding fails.
    pub corrupt_tiles: Vec<(u32, usize, u32, u32)>,
    /// Write no `TiffData` (planes are then IFDs 0, 1, …).
    pub omit_tiff_data: bool,
    /// The OME-XML to write instead of the generated one.
    pub xml: Option<String>,
}

impl OmeTiffSpec {
    /// A tiled `width` x `height` pyramid of `levels` levels with SubIFDs,
    /// uint16 channels named and coloured as given, 0.5 µm/px and a 20x
    /// objective.
    pub fn channels(
        width: u32,
        height: u32,
        tile: u32,
        levels: u32,
        channels: &[(&str, [u8; 3])],
    ) -> Self {
        Self {
            width,
            height,
            tile: Some(tile),
            rows_per_strip: 16,
            levels,
            pyramid: OmePyramid::SubIfds,
            sample: OmeSample::U16,
            channels: channels
                .iter()
                .map(|(name, color)| OmeChannelSpec::new(name, Some(*color)))
                .collect(),
            compression: OmeCompression::None,
            predictor: false,
            planar: false,
            big_tiff: false,
            big_endian: false,
            physical_size: Some((0.5, "µm".to_string())),
            objective: Some(20.0),
            extra_images: Vec::new(),
            missing_tiles: Vec::new(),
            corrupt_tiles: Vec::new(),
            omit_tiff_data: false,
            xml: None,
        }
    }

    /// A tiled RGB pyramid with SubIFDs.
    pub fn rgb(width: u32, height: u32, tile: u32, levels: u32) -> Self {
        Self {
            sample: OmeSample::Rgb8,
            channels: Vec::new(),
            ..Self::channels(width, height, tile, levels, &[])
        }
    }

    fn channel_count(&self) -> usize {
        match self.sample {
            OmeSample::Rgb8 => 3,
            _ => self.channels.len(),
        }
    }

    /// Size of level `level`.
    pub fn level_size(&self, level: u32) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }
}

/// The level-0 pixel that level pixel (`x`, `y`) of a `level_size` level
/// samples (pixel centres).
pub fn ome_level0_pixel(spec: &OmeTiffSpec, level: u32, x: u32, y: u32) -> (u32, u32) {
    let (w, h) = spec.level_size(level);
    let x0 = ((f64::from(x) + 0.5) * f64::from(spec.width) / f64::from(w)) as u32;
    let y0 = ((f64::from(y) + 0.5) * f64::from(spec.height) / f64::from(h)) as u32;
    (x0.min(spec.width - 1), y0.min(spec.height - 1))
}

#[derive(Clone)]
enum Value {
    Short(Vec<u16>),
    Long(Vec<u32>),
    Long8(Vec<u64>),
    Ifd(Vec<u64>),
    Ascii(String),
    Undefined(Vec<u8>),
}

struct Writer {
    buf: Vec<u8>,
    big: bool,
    big_endian: bool,
}

impl Writer {
    fn u16(&self, v: u16) -> [u8; 2] {
        if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn u32(&self, v: u32) -> [u8; 4] {
        if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn u64(&self, v: u64) -> [u8; 8] {
        if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn offset_bytes(&self, v: u64) -> Vec<u8> {
        if self.big {
            self.u64(v).to_vec()
        } else {
            self.u32(v as u32).to_vec()
        }
    }

    fn align(&mut self) {
        if self.buf.len() % 2 == 1 {
            self.buf.push(0);
        }
    }

    /// An offset array: LONG in TIFF, LONG8 in BigTIFF.
    fn offsets(&self, values: Vec<u64>) -> Value {
        if self.big {
            Value::Long8(values)
        } else {
            Value::Long(values.into_iter().map(|v| v as u32).collect())
        }
    }

    fn encode(&self, value: &Value) -> (u16, u64, Vec<u8>) {
        match value {
            Value::Short(v) => (
                3,
                v.len() as u64,
                v.iter().flat_map(|x| self.u16(*x)).collect(),
            ),
            Value::Long(v) => (
                4,
                v.len() as u64,
                v.iter().flat_map(|x| self.u32(*x)).collect(),
            ),
            Value::Long8(v) => (
                16,
                v.len() as u64,
                v.iter().flat_map(|x| self.u64(*x)).collect(),
            ),
            Value::Ifd(v) => {
                if self.big {
                    (
                        18,
                        v.len() as u64,
                        v.iter().flat_map(|x| self.u64(*x)).collect(),
                    )
                } else {
                    (
                        13,
                        v.len() as u64,
                        v.iter().flat_map(|x| self.u32(*x as u32)).collect(),
                    )
                }
            }
            Value::Ascii(s) => (
                2,
                s.len() as u64 + 1,
                s.bytes().chain(std::iter::once(0)).collect(),
            ),
            Value::Undefined(v) => (7, v.len() as u64, v.clone()),
        }
    }

    /// Writes an IFD; returns its offset and the position of its "next
    /// IFD" field.
    fn ifd(&mut self, mut tags: Vec<(u16, Value)>) -> (u64, usize) {
        tags.sort_by_key(|(tag, _)| *tag);
        let inline = if self.big { 8 } else { 4 };
        // Out-of-line values first, then the table pointing at them.
        let mut fields = Vec::new();
        for (tag, value) in &tags {
            let (ty, count, bytes) = self.encode(value);
            let field = if bytes.len() <= inline {
                let mut field = bytes.clone();
                field.resize(inline, 0);
                field
            } else {
                self.align();
                let at = self.buf.len() as u64;
                self.buf.extend_from_slice(&bytes);
                self.offset_bytes(at)
            };
            fields.push((*tag, ty, count, field));
        }
        self.align();
        let offset = self.buf.len() as u64;
        if self.big {
            let count = self.u64(fields.len() as u64);
            self.buf.extend_from_slice(&count);
        } else {
            let count = self.u16(fields.len() as u16);
            self.buf.extend_from_slice(&count);
        }
        for (tag, ty, count, field) in fields {
            let (tag, ty) = (self.u16(tag), self.u16(ty));
            self.buf.extend_from_slice(&tag);
            self.buf.extend_from_slice(&ty);
            let count = self.offset_bytes(count);
            self.buf.extend_from_slice(&count);
            self.buf.extend_from_slice(&field);
        }
        let next = self.buf.len();
        let zero = self.offset_bytes(0);
        self.buf.extend_from_slice(&zero);
        (offset, next)
    }

    fn patch_offset(&mut self, at: usize, value: u64) {
        let bytes = self.offset_bytes(value);
        self.buf[at..at + bytes.len()].copy_from_slice(&bytes);
    }
}

fn compress(spec: &OmeTiffSpec, raw: &[u8], width: u32, rows: u32, samples: usize) -> Vec<u8> {
    match spec.compression {
        OmeCompression::None | OmeCompression::Jpeg2000 => raw.to_vec(),
        OmeCompression::Lzw => {
            weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .encode(raw)
                .expect("LZW")
        }
        OmeCompression::Deflate => {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(raw).expect("deflate");
            encoder.finish().expect("deflate")
        }
        OmeCompression::Zstd => zstd::bulk::compress(raw, 3).expect("zstd"),
        OmeCompression::PackBits => packbits(raw),
        OmeCompression::Jpeg | OmeCompression::JpegTables => {
            let mut out = Vec::new();
            let color = if samples == 3 {
                image::ExtendedColorType::Rgb8
            } else {
                image::ExtendedColorType::L8
            };
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
                .encode(raw, width, rows, color)
                .expect("JPEG");
            out
        }
    }
}

/// PackBits with literal runs and repeats of up to 128 bytes.
fn packbits(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let mut run = 1;
        while i + run < raw.len() && raw[i + run] == raw[i] && run < 128 {
            run += 1;
        }
        if run >= 3 {
            out.push((1 - run as i16) as i8 as u8);
            out.push(raw[i]);
            i += run;
        } else {
            let start = i;
            while i < raw.len() && i - start < 128 {
                if i + 2 < raw.len() && raw[i] == raw[i + 1] && raw[i] == raw[i + 2] {
                    break;
                }
                i += 1;
            }
            out.push((i - start - 1) as u8);
            out.extend_from_slice(&raw[start..i]);
        }
    }
    out
}

/// Splits a JPEG into its quantisation and Huffman tables (as a tables-only
/// stream) and the abbreviated image stream.
fn split_jpeg_tables(jpeg: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut tables = vec![0xFF, 0xD8];
    let mut image = vec![0xFF, 0xD8];
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        let marker = jpeg[i + 1];
        if marker == 0xDA {
            // Start of scan: the entropy-coded data follows; keep the rest.
            image.extend_from_slice(&jpeg[i..]);
            break;
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        let segment = &jpeg[i..i + 2 + len];
        if matches!(marker, 0xDB | 0xC4) {
            tables.extend_from_slice(segment);
        } else {
            image.extend_from_slice(segment);
        }
        i += 2 + len;
    }
    tables.extend_from_slice(&[0xFF, 0xD9]);
    (tables, image)
}

/// The raw bytes of one chunk of plane `channel` of `level`, in the file's
/// byte order, with the predictor applied.
#[allow(clippy::too_many_arguments)]
fn chunk_bytes(
    spec: &OmeTiffSpec,
    level: u32,
    channels: &[usize],
    left: u32,
    top: u32,
    chunk_width: u32,
    rows: u32,
    value: &dyn Fn(usize, u32, u32) -> f64,
) -> Vec<u8> {
    let (lw, lh) = spec.level_size(level);
    let samples = channels.len();
    let bytes = spec.sample.bytes();
    let mut values = vec![0f64; chunk_width as usize * rows as usize * samples];
    for dy in 0..rows {
        for dx in 0..chunk_width {
            let (x, y) = (left + dx, top + dy);
            if x >= lw || y >= lh {
                continue;
            }
            let (x0, y0) = ome_level0_pixel(spec, level, x, y);
            for (s, &channel) in channels.iter().enumerate() {
                values[(dy as usize * chunk_width as usize + dx as usize) * samples + s] =
                    value(channel, x0, y0);
            }
        }
    }
    let row_len = chunk_width as usize * samples;
    let jpeg = matches!(
        spec.compression,
        OmeCompression::Jpeg | OmeCompression::JpegTables
    );
    if spec.sample == OmeSample::F32 && spec.predictor {
        // Floating-point predictor: big-endian byte planes, differenced.
        let mut out = Vec::with_capacity(values.len() * 4);
        for row in values.chunks_exact(row_len) {
            let n = row.len();
            let mut planes = vec![0u8; n * 4];
            for (i, v) in row.iter().enumerate() {
                for (k, b) in (*v as f32).to_be_bytes().iter().enumerate() {
                    planes[k * n + i] = *b;
                }
            }
            for i in (samples..planes.len()).rev() {
                planes[i] = planes[i].wrapping_sub(planes[i - samples]);
            }
            out.extend_from_slice(&planes);
        }
        return out;
    }
    let mut ints: Vec<i64> = values
        .iter()
        .map(|&v| match spec.sample {
            OmeSample::U8 | OmeSample::Rgb8 => v.clamp(0.0, 255.0) as i64,
            OmeSample::U16 => v.clamp(0.0, 65_535.0) as i64,
            OmeSample::I16 => v.clamp(-32_768.0, 32_767.0) as i64,
            OmeSample::F32 => i64::from((v as f32).to_bits()),
        })
        .collect();
    if spec.predictor && !jpeg {
        let modulus = 1i64 << (bytes * 8);
        for row in ints.chunks_exact_mut(row_len) {
            for i in (samples..row.len()).rev() {
                row[i] = (row[i] - row[i - samples]).rem_euclid(modulus);
            }
        }
    }
    let mut out = Vec::with_capacity(ints.len() * bytes);
    for v in ints {
        let le = (v as u64).to_le_bytes();
        let sample = &le[..bytes];
        if spec.big_endian {
            out.extend(sample.iter().rev());
        } else {
            out.extend_from_slice(sample);
        }
    }
    out
}

/// One IFD to write: a plane of a level.
struct PlaneIfd {
    level: u32,
    /// The channels it holds, as samples.
    channels: Vec<usize>,
    /// The main-chain channel index for missing/corrupt tile lookups.
    channel: usize,
    reduced: bool,
}

/// Writes the chunks of a plane; returns its tags without the
/// description.
fn write_plane(
    w: &mut Writer,
    spec: &OmeTiffSpec,
    plane: &PlaneIfd,
    value: &dyn Fn(usize, u32, u32) -> f64,
    jpeg_tables: &mut Option<Vec<u8>>,
) -> Vec<(u16, Value)> {
    let (lw, lh) = spec.level_size(plane.level);
    let rgb = spec.sample == OmeSample::Rgb8;
    let planar = rgb && spec.planar;
    let (chunk_width, chunk_height) = match spec.tile {
        Some(tile) => (tile, tile),
        None => (lw, spec.rows_per_strip.min(lh)),
    };
    let across = lw.div_ceil(chunk_width);
    let down = lh.div_ceil(chunk_height);
    let sample_sets: Vec<Vec<usize>> = if planar {
        plane.channels.iter().map(|&c| vec![c]).collect()
    } else {
        vec![plane.channels.clone()]
    };
    let mut offsets = Vec::new();
    let mut counts = Vec::new();
    for set in &sample_sets {
        for ty in 0..down {
            for tx in 0..across {
                let key = (plane.level, plane.channel, tx, ty);
                if spec.missing_tiles.contains(&key) {
                    offsets.push(0);
                    counts.push(0);
                    continue;
                }
                let rows = if spec.tile.is_some() {
                    chunk_height
                } else {
                    (lh - ty * chunk_height).min(chunk_height)
                };
                let raw = chunk_bytes(
                    spec,
                    plane.level,
                    set,
                    tx * chunk_width,
                    ty * chunk_height,
                    chunk_width,
                    rows,
                    value,
                );
                let mut data = compress(spec, &raw, chunk_width, rows, set.len());
                if spec.compression == OmeCompression::JpegTables {
                    let (tables, image) = split_jpeg_tables(&data);
                    jpeg_tables.get_or_insert(tables);
                    data = image;
                }
                if spec.corrupt_tiles.contains(&key) {
                    data = vec![0xA5; data.len().clamp(16, 64)];
                }
                w.align();
                offsets.push(w.buf.len() as u64);
                counts.push(data.len() as u64);
                w.buf.extend_from_slice(&data);
            }
        }
    }
    let samples = plane.channels.len() as u16;
    let jpeg = matches!(
        spec.compression,
        OmeCompression::Jpeg | OmeCompression::JpegTables
    );
    let compression = match spec.compression {
        OmeCompression::None => 1,
        OmeCompression::Lzw => 5,
        OmeCompression::Deflate => 8,
        OmeCompression::Zstd => 50_000,
        OmeCompression::PackBits => 32_773,
        OmeCompression::Jpeg | OmeCompression::JpegTables => 7,
        OmeCompression::Jpeg2000 => 33_003,
    };
    let photometric = match (rgb, jpeg) {
        (true, true) => 6,
        (true, false) => 2,
        _ => 1,
    };
    let bits = (spec.sample.bytes() * 8) as u16;
    let mut tags = vec![
        (256, Value::Long(vec![lw])),
        (257, Value::Long(vec![lh])),
        (258, Value::Short(vec![bits; samples as usize])),
        (259, Value::Short(vec![compression])),
        (262, Value::Short(vec![photometric])),
        (277, Value::Short(vec![samples])),
        (284, Value::Short(vec![if planar { 2 } else { 1 }])),
        (
            339,
            Value::Short(vec![spec.sample.sample_format(); samples as usize]),
        ),
    ];
    if plane.reduced {
        tags.push((254, Value::Long(vec![1])));
    }
    if spec.predictor && !jpeg {
        tags.push((
            317,
            Value::Short(vec![if spec.sample == OmeSample::F32 { 3 } else { 2 }]),
        ));
    }
    if spec.tile.is_some() {
        tags.push((322, Value::Long(vec![chunk_width])));
        tags.push((323, Value::Long(vec![chunk_height])));
        tags.push((324, w.offsets(offsets)));
        tags.push((325, w.offsets(counts)));
    } else {
        tags.push((273, w.offsets(offsets)));
        tags.push((278, Value::Long(vec![chunk_height])));
        tags.push((279, w.offsets(counts)));
    }
    if let Some(tables) = jpeg_tables {
        tags.push((347, Value::Undefined(tables.clone())));
    }
    tags
}

fn color_attr(color: [u8; 3]) -> String {
    let [r, g, b] = color;
    let packed = u32::from_be_bytes([r, g, b, 255]);
    (packed as i32).to_string()
}

/// The generated OME-XML. `ifd_of` gives the main-chain IFD of the first
/// plane of each image (main, pyramid images, extra images).
fn ome_xml(spec: &OmeTiffSpec, image_ifds: &[(String, u32, u32, u64, usize)]) -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" \
         Creator=\"eov test\" UUID=\"urn:uuid:00000000-0000-0000-0000-000000000001\">\n",
    );
    if let Some(power) = spec.objective {
        xml.push_str(&format!(
            "  <Instrument ID=\"Instrument:0\"><Objective ID=\"Objective:0:0\" \
             NominalMagnification=\"{power}\"/></Instrument>\n"
        ));
    }
    for (index, (name, width, height, first_ifd, planes)) in image_ifds.iter().enumerate() {
        let extra = index > 0 && spec.extra_images.iter().any(|e| &e.name == name);
        let rgb = extra || spec.sample == OmeSample::Rgb8;
        let size_c = if rgb { 3 } else { spec.channels.len() };
        let ty = if extra {
            "uint8"
        } else {
            spec.sample.ome_type()
        };
        xml.push_str(&format!("  <Image ID=\"Image:{index}\" Name=\"{name}\">\n"));
        if index == 0 {
            xml.push_str("    <AcquisitionDate>2024-01-02T03:04:05</AcquisitionDate>\n");
            if spec.objective.is_some() {
                xml.push_str(
                    "    <InstrumentRef ID=\"Instrument:0\"/>\n    \
                     <ObjectiveSettings ID=\"Objective:0:0\"/>\n",
                );
            }
        }
        let physical = match (&spec.physical_size, extra) {
            (Some((value, unit)), false) => {
                let scale = f64::from(spec.width) / f64::from(*width);
                let v = value * scale;
                format!(
                    " PhysicalSizeX=\"{v}\" PhysicalSizeXUnit=\"{unit}\" PhysicalSizeY=\"{v}\" \
                     PhysicalSizeYUnit=\"{unit}\""
                )
            }
            _ => String::new(),
        };
        xml.push_str(&format!(
            "    <Pixels ID=\"Pixels:{index}\" DimensionOrder=\"XYCZT\" Type=\"{ty}\" \
             SizeX=\"{width}\" SizeY=\"{height}\" SizeZ=\"1\" SizeC=\"{size_c}\" SizeT=\"1\"{physical}>\n"
        ));
        if rgb {
            xml.push_str(&format!(
                "      <Channel ID=\"Channel:{index}:0\" SamplesPerPixel=\"3\"/>\n"
            ));
        } else {
            for (c, channel) in spec.channels.iter().enumerate() {
                let color = channel
                    .color
                    .map(|rgb| format!(" Color=\"{}\"", color_attr(rgb)))
                    .unwrap_or_default();
                let emission = channel
                    .emission_nm
                    .map(|nm| format!(" EmissionWavelength=\"{nm}\" EmissionWavelengthUnit=\"nm\""))
                    .unwrap_or_default();
                xml.push_str(&format!(
                    "      <Channel ID=\"Channel:{index}:{c}\" Name=\"{}\"{color}{emission} \
                     SamplesPerPixel=\"1\"/>\n",
                    channel.name
                ));
            }
        }
        if !(spec.omit_tiff_data && index == 0) {
            xml.push_str(&format!(
                "      <TiffData IFD=\"{first_ifd}\" PlaneCount=\"{planes}\"/>\n"
            ));
        }
        xml.push_str("    </Pixels>\n  </Image>\n");
    }
    xml.push_str("</OME>\n");
    xml
}

/// Writes `spec` to `path`. `value(channel, x, y)` is the level-0 sample of
/// a channel (for RGB, channel 0..3 is red, green, blue).
pub fn write_ome_tiff(
    path: &Path,
    spec: &OmeTiffSpec,
    value: impl Fn(usize, u32, u32) -> f64,
) -> std::io::Result<()> {
    let rgb = spec.sample == OmeSample::Rgb8;
    let channel_count = spec.channel_count();
    let mut w = Writer {
        buf: Vec::new(),
        big: spec.big_tiff,
        big_endian: spec.big_endian,
    };
    // Header, patched at the end.
    let magic: &[u8] = if spec.big_endian { b"MM" } else { b"II" };
    w.buf.extend_from_slice(magic);
    if spec.big_tiff {
        let version = w.u16(43);
        let size = w.u16(8);
        let zero = w.u16(0);
        w.buf.extend_from_slice(&version);
        w.buf.extend_from_slice(&size);
        w.buf.extend_from_slice(&zero);
        w.buf.extend_from_slice(&[0; 8]);
    } else {
        let version = w.u16(42);
        w.buf.extend_from_slice(&version);
        w.buf.extend_from_slice(&[0; 4]);
    }
    let first_ifd_field = if spec.big_tiff { 8 } else { 4 };

    // The planes of the main image: one per channel, or one RGB plane.
    let base_planes: Vec<Vec<usize>> = if rgb {
        vec![vec![0, 1, 2]]
    } else {
        (0..channel_count).map(|c| vec![c]).collect()
    };
    let value: &dyn Fn(usize, u32, u32) -> f64 = &value;
    let mut jpeg_tables = None;

    // Main-chain IFDs: (tags, sub-IFD offsets), in chain order.
    let mut chain: Vec<Vec<(u16, Value)>> = Vec::new();
    let mut image_ifds: Vec<(String, u32, u32, u64, usize)> = Vec::new();
    image_ifds.push((
        "test".to_string(),
        spec.width,
        spec.height,
        0,
        base_planes.len(),
    ));
    for (index, channels) in base_planes.iter().enumerate() {
        let mut tags = write_plane(
            &mut w,
            spec,
            &PlaneIfd {
                level: 0,
                channels: channels.clone(),
                channel: index,
                reduced: false,
            },
            value,
            &mut jpeg_tables,
        );
        if spec.pyramid == OmePyramid::SubIfds && spec.levels > 1 {
            let mut subs = Vec::new();
            for level in 1..spec.levels {
                let sub_tags = write_plane(
                    &mut w,
                    spec,
                    &PlaneIfd {
                        level,
                        channels: channels.clone(),
                        channel: index,
                        reduced: true,
                    },
                    value,
                    &mut jpeg_tables,
                );
                let (offset, _) = w.ifd(sub_tags);
                subs.push(offset);
            }
            tags.push((330, Value::Ifd(subs)));
        }
        chain.push(tags);
    }
    if spec.pyramid != OmePyramid::SubIfds {
        for level in 1..spec.levels {
            let (lw, lh) = spec.level_size(level);
            if spec.pyramid == OmePyramid::Images {
                image_ifds.push((
                    format!("test level {level}"),
                    lw,
                    lh,
                    chain.len() as u64,
                    base_planes.len(),
                ));
            }
            for (index, channels) in base_planes.iter().enumerate() {
                chain.push(write_plane(
                    &mut w,
                    spec,
                    &PlaneIfd {
                        level,
                        channels: channels.clone(),
                        channel: index,
                        reduced: spec.pyramid == OmePyramid::ReducedIfds,
                    },
                    value,
                    &mut jpeg_tables,
                ));
            }
        }
    }
    for extra in &spec.extra_images {
        image_ifds.push((
            extra.name.clone(),
            extra.width,
            extra.height,
            chain.len() as u64,
            1,
        ));
        let pixels: Vec<u8> = (0..extra.width * extra.height)
            .flat_map(|_| extra.color)
            .collect();
        w.align();
        let offset = w.buf.len() as u64;
        w.buf.extend_from_slice(&pixels);
        chain.push(vec![
            (256, Value::Long(vec![extra.width])),
            (257, Value::Long(vec![extra.height])),
            (258, Value::Short(vec![8, 8, 8])),
            (259, Value::Short(vec![1])),
            (262, Value::Short(vec![2])),
            (273, w.offsets(vec![offset])),
            (277, Value::Short(vec![3])),
            (278, Value::Long(vec![extra.height])),
            (279, w.offsets(vec![pixels.len() as u64])),
        ]);
    }

    let xml = spec
        .xml
        .clone()
        .unwrap_or_else(|| ome_xml(spec, &image_ifds));
    chain[0].push((270, Value::Ascii(xml)));
    let mut previous_next: Option<usize> = None;
    for tags in chain {
        let (offset, next) = w.ifd(tags);
        match previous_next {
            Some(at) => w.patch_offset(at, offset),
            None => w.patch_offset(first_ifd_field, offset),
        }
        previous_next = Some(next);
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(&w.buf)?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_and_jpeg_table_splitting_round_trip() {
        let raw: Vec<u8> = (0..300u32)
            .map(|i| if i % 50 < 20 { 7 } else { (i % 13) as u8 })
            .collect();
        let packed = packbits(&raw);
        // Decoded with the reader's decoder in the reader's tests; here,
        // only check that runs were found.
        assert!(packed.len() < raw.len());
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
            .encode(&[128u8; 16 * 16], 16, 16, image::ExtendedColorType::L8)
            .unwrap();
        let (tables, image) = split_jpeg_tables(&jpeg);
        assert_eq!(&tables[..2], [0xFF, 0xD8]);
        assert!(tables.windows(2).any(|w| w == [0xFF, 0xDB]));
        assert!(!image.windows(2).any(|w| w == [0xFF, 0xDB]));
        assert_eq!(&image[image.len() - 2..], [0xFF, 0xD9]);
    }
}
