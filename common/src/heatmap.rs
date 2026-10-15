//! Heat maps made by other tools (AUDIT §14.2): an image whose pixels cover
//! a slide, such as a model's per-patch probabilities or an attention map,
//! decoded, placed over the slide, coloured and cut into tiles for drawing.
//!
//! Single-channel images (and colour images whose pixels are all grey) are
//! *values*: 8-bit pixels are divided by 255, 16-bit ones by 65 535, and
//! floating-point ones are used as they are when they all lie in 0–1, or
//! scaled from their minimum to their maximum otherwise. Values are drawn
//! through a colour map; pixels whose alpha is zero, NaN values and values
//! below the threshold are left transparent. Colour images are drawn with
//! their own colours.
//!
//! A PNG can say where it lies on the slide in `tEXt` chunks:
//! `eov:level0_px_per_pixel` (`"sx,sy"`, or one number for both axes) and,
//! optionally, `eov:origin` (`"x,y"`, level-0 pixels, default `0,0`).
//! eovae's `eovae:level0_px_per_pixel` is read too. Such maps are placed
//! [`Placement::FromFile`] by default.

use std::path::Path;

/// Largest heat map read, in pixels.
pub const MAX_PIXELS: u64 = 64 * 1024 * 1024;

/// Edge of the tiles a heat map is drawn in, in heat-map pixels.
pub const TILE: u32 = 1024;

/// What a heat map's pixels hold.
#[derive(Debug, Clone, PartialEq)]
pub enum HeatValues {
    /// One value per pixel in 0–1; NaN where there is no data.
    Scalar(Vec<f32>),
    /// Straight (not premultiplied) RGBA8.
    Color(Vec<u8>),
}

/// Where a map says it lies: its first pixel's level-0 corner and the
/// level-0 pixels each map pixel covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FileGeometry {
    pub x: f64,
    pub y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

/// A decoded heat map.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatMap {
    pub width: u32,
    pub height: u32,
    pub values: HeatValues,
    /// How the file's pixels became values, for the user.
    pub scaling: String,
    /// The placement the file records, if any.
    pub geometry: Option<FileGeometry>,
}

impl HeatMap {
    pub fn is_scalar(&self) -> bool {
        matches!(self.values, HeatValues::Scalar(_))
    }
}

/// Reads a PNG, JPEG or TIFF heat map (the first image of a TIFF).
pub fn decode(path: &Path) -> Result<HeatMap, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("cannot read the file: {error}"))?;
    decode_bytes(&bytes)
}

pub fn decode_bytes(bytes: &[u8]) -> Result<HeatMap, String> {
    if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        decode_tiff(bytes)
    } else {
        let mut map = decode_image(bytes)?;
        map.geometry = png_geometry(bytes);
        Ok(map)
    }
}

/// The `tEXt` entries of a PNG (keyword, text), wherever they are.
fn png_text(bytes: &[u8]) -> Vec<(String, String)> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    let mut entries = Vec::new();
    if !bytes.starts_with(SIGNATURE) {
        return entries;
    }
    let mut at = SIGNATURE.len();
    while at + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind = &bytes[at + 4..at + 8];
        let Some(data) = bytes.get(at + 8..at + 8 + length) else {
            break;
        };
        if kind == b"tEXt"
            && let Some(split) = data.iter().position(|b| *b == 0)
        {
            // Latin-1: each byte is its code point.
            let latin1 = |b: &[u8]| b.iter().map(|&c| char::from(c)).collect::<String>();
            entries.push((latin1(&data[..split]), latin1(&data[split + 1..])));
        }
        if kind == b"IEND" {
            break;
        }
        at += 12 + length;
    }
    entries
}

fn pair(text: &str) -> Option<(f64, f64)> {
    let mut parts = text.split(',').map(|part| part.trim().parse::<f64>().ok());
    match (parts.next().flatten(), parts.next(), parts.next()) {
        (Some(a), None, None) => Some((a, a)),
        (Some(a), Some(Some(b)), None) => Some((a, b)),
        _ => None,
    }
}

fn png_geometry(bytes: &[u8]) -> Option<FileGeometry> {
    let text = png_text(bytes);
    let get = |key: &str| {
        text.iter()
            .find(|(keyword, _)| keyword == key)
            .map(|(_, value)| value.as_str())
    };
    let (scale_x, scale_y) = get("eov:level0_px_per_pixel")
        .or_else(|| get("eovae:level0_px_per_pixel"))
        .and_then(pair)
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *x > 0.0 && *y > 0.0)?;
    let (x, y) = get("eov:origin")
        .and_then(pair)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .unwrap_or((0.0, 0.0));
    Some(FileGeometry {
        x,
        y,
        scale_x,
        scale_y,
    })
}

fn check_size(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("the image is empty".to_string());
    }
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!(
            "the image is {width} × {height} pixels; heat maps of at most {} megapixels are supported",
            MAX_PIXELS / (1024 * 1024)
        ));
    }
    Ok(())
}

fn decode_image(bytes: &[u8]) -> Result<HeatMap, String> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| error.to_string())?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| format!("not a PNG, JPEG or TIFF image ({error})"))?;
    check_size(width, height)?;
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("cannot decode the image: {error}"))?;
    use image::DynamicImage as D;
    Ok(match image {
        D::ImageLuma8(gray) => from_gray8(width, height, gray.as_raw(), None),
        D::ImageLumaA8(gray) => {
            let raw = gray.as_raw();
            let values: Vec<u8> = raw.iter().step_by(2).copied().collect();
            let alpha: Vec<u8> = raw.iter().skip(1).step_by(2).copied().collect();
            from_gray8(width, height, &values, Some(&alpha))
        }
        D::ImageLuma16(gray) => from_gray16(width, height, gray.as_raw(), None),
        D::ImageLumaA16(gray) => {
            let raw = gray.as_raw();
            let values: Vec<u16> = raw.iter().step_by(2).copied().collect();
            let alpha: Vec<u16> = raw.iter().skip(1).step_by(2).copied().collect();
            from_gray16(width, height, &values, Some(&alpha))
        }
        other => from_rgba8(width, height, other.to_rgba8().into_raw()),
    })
}

fn decode_tiff(bytes: &[u8]) -> Result<HeatMap, String> {
    use tiff::ColorType as C;
    use tiff::decoder::{Decoder, DecodingResult as R};
    let error = |error: tiff::TiffError| format!("cannot decode the TIFF: {error}");
    let mut decoder = Decoder::new(std::io::Cursor::new(bytes)).map_err(error)?;
    let (width, height) = decoder.dimensions().map_err(error)?;
    check_size(width, height)?;
    let color = decoder.colortype().map_err(error)?;
    let data = decoder.read_image().map_err(error)?;
    let pixels = width as usize * height as usize;
    let channels = match color {
        C::Gray(_) => 1,
        C::GrayA(_) => 2,
        C::RGB(_) => 3,
        C::RGBA(_) => 4,
        other => return Err(format!("{other:?} TIFFs are not supported as heat maps")),
    };
    // Floating-point and wide integer samples as f64, for scalar maps.
    let samples: Option<Vec<f64>> = match &data {
        R::F32(v) => Some(v.iter().map(|&s| f64::from(s)).collect()),
        R::F64(v) => Some(v.clone()),
        R::U32(v) => Some(v.iter().map(|&s| f64::from(s)).collect()),
        R::I8(v) => Some(v.iter().map(|&s| f64::from(s)).collect()),
        R::I16(v) => Some(v.iter().map(|&s| f64::from(s)).collect()),
        R::I32(v) => Some(v.iter().map(|&s| f64::from(s)).collect()),
        _ => None,
    };
    if data_len(&data) < pixels * channels {
        return Err("the TIFF holds fewer samples than its size needs".to_string());
    }
    match (data, channels) {
        (R::U8(v), 1) => Ok(from_gray8(width, height, &v[..pixels], None)),
        (R::U8(v), 2) => {
            let values: Vec<u8> = v.iter().step_by(2).take(pixels).copied().collect();
            let alpha: Vec<u8> = v.iter().skip(1).step_by(2).take(pixels).copied().collect();
            Ok(from_gray8(width, height, &values, Some(&alpha)))
        }
        (R::U8(v), 3) => Ok(from_rgba8(
            width,
            height,
            v.as_chunks::<3>()
                .0
                .iter()
                .take(pixels)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        )),
        (R::U8(v), _) => Ok(from_rgba8(width, height, v[..pixels * 4].to_vec())),
        (R::U16(v), 1) => Ok(from_gray16(width, height, &v[..pixels], None)),
        (R::U16(v), 2) => {
            let values: Vec<u16> = v.iter().step_by(2).take(pixels).copied().collect();
            let alpha: Vec<u16> = v.iter().skip(1).step_by(2).take(pixels).copied().collect();
            Ok(from_gray16(width, height, &values, Some(&alpha)))
        }
        (R::U16(v), channels) => Ok(from_rgba8(
            width,
            height,
            v.chunks_exact(channels)
                .take(pixels)
                .flat_map(|p| {
                    let c = |s: u16| (s >> 8) as u8;
                    [
                        c(p[0]),
                        c(p[1]),
                        c(p[2]),
                        if channels == 4 { c(p[3]) } else { 255 },
                    ]
                })
                .collect(),
        )),
        (_, 1) => match samples {
            Some(samples) => Ok(from_float(width, height, &samples[..pixels])),
            None => Err("this TIFF sample format is not supported".to_string()),
        },
        _ => Err("only single-channel TIFFs may use floating-point or signed samples".to_string()),
    }
}

fn data_len(data: &tiff::decoder::DecodingResult) -> usize {
    use tiff::decoder::DecodingResult as R;
    match data {
        R::U8(v) => v.len(),
        R::U16(v) => v.len(),
        R::U32(v) => v.len(),
        R::U64(v) => v.len(),
        R::F16(v) => v.len(),
        R::F32(v) => v.len(),
        R::F64(v) => v.len(),
        R::I8(v) => v.len(),
        R::I16(v) => v.len(),
        R::I32(v) => v.len(),
        R::I64(v) => v.len(),
    }
}

fn from_gray8(width: u32, height: u32, values: &[u8], alpha: Option<&[u8]>) -> HeatMap {
    let values = values
        .iter()
        .enumerate()
        .map(|(i, &v)| match alpha {
            Some(alpha) if alpha[i] == 0 => f32::NAN,
            _ => f32::from(v) / 255.0,
        })
        .collect();
    HeatMap {
        width,
        height,
        values: HeatValues::Scalar(values),
        scaling: "8-bit values, 0–255 shown as 0–1".to_string(),
        geometry: None,
    }
}

fn from_gray16(width: u32, height: u32, values: &[u16], alpha: Option<&[u16]>) -> HeatMap {
    let values = values
        .iter()
        .enumerate()
        .map(|(i, &v)| match alpha {
            Some(alpha) if alpha[i] == 0 => f32::NAN,
            _ => f32::from(v) / 65_535.0,
        })
        .collect();
    HeatMap {
        width,
        height,
        values: HeatValues::Scalar(values),
        scaling: "16-bit values, 0–65535 shown as 0–1".to_string(),
        geometry: None,
    }
}

fn from_float(width: u32, height: u32, samples: &[f64]) -> HeatMap {
    let finite = samples.iter().copied().filter(|v| v.is_finite());
    let (min, max) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let (offset, scale, scaling) = if !min.is_finite() {
        (0.0, 1.0, "no finite values".to_string())
    } else if min >= 0.0 && max <= 1.0 {
        (0.0, 1.0, "values 0–1".to_string())
    } else if max > min {
        (
            min,
            1.0 / (max - min),
            format!("values {} to {} shown as 0–1", trim(min), trim(max)),
        )
    } else {
        (min - 1.0, 1.0, format!("every value is {}", trim(min)))
    };
    let values = samples
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - offset) * scale).clamp(0.0, 1.0) as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    HeatMap {
        width,
        height,
        values: HeatValues::Scalar(values),
        scaling,
        geometry: None,
    }
}

fn trim(value: f64) -> String {
    let text = format!("{value:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Colour images whose pixels are all grey hold values (many tools save
/// grey maps as RGB).
fn from_rgba8(width: u32, height: u32, rgba: Vec<u8>) -> HeatMap {
    let grey = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| p[0] == p[1] && p[1] == p[2]);
    if grey {
        let values: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        let alpha: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
        let mut map = from_gray8(width, height, &values, Some(&alpha));
        map.scaling = "grey pixels, 0–255 shown as 0–1".to_string();
        return map;
    }
    HeatMap {
        width,
        height,
        values: HeatValues::Color(rgba),
        scaling: "colour image, drawn as it is".to_string(),
        geometry: None,
    }
}

/// How values are coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMap {
    #[default]
    Viridis,
    Inferno,
    Jet,
    Gray,
}

impl ColorMap {
    pub const ALL: [ColorMap; 4] = [Self::Viridis, Self::Inferno, Self::Jet, Self::Gray];

    pub fn id(self) -> &'static str {
        match self {
            Self::Viridis => "viridis",
            Self::Inferno => "inferno",
            Self::Jet => "jet",
            Self::Gray => "gray",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno",
            Self::Jet => "Jet",
            Self::Gray => "Grey",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|map| map.id() == id)
    }

    /// The colour of value `t` (0–1).
    pub fn color(self, t: f32) -> [u8; 3] {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Self::Viridis => interpolate(&VIRIDIS, t),
            Self::Inferno => interpolate(&INFERNO, t),
            Self::Jet => {
                let channel = |offset: f32| {
                    ((1.5 - (4.0 * t - offset).abs()).clamp(0.0, 1.0) * 255.0).round() as u8
                };
                [channel(3.0), channel(2.0), channel(1.0)]
            }
            Self::Gray => {
                let v = (t * 255.0).round() as u8;
                [v, v, v]
            }
        }
    }
}

/// Matplotlib's viridis at nine evenly spaced points.
const VIRIDIS: [[u8; 3]; 9] = [
    [0x44, 0x01, 0x54],
    [0x47, 0x2d, 0x7b],
    [0x3b, 0x52, 0x8b],
    [0x2c, 0x72, 0x8e],
    [0x21, 0x91, 0x8c],
    [0x28, 0xae, 0x80],
    [0x5e, 0xc9, 0x62],
    [0xad, 0xdc, 0x30],
    [0xfd, 0xe7, 0x25],
];

/// Matplotlib's inferno at ten evenly spaced points.
const INFERNO: [[u8; 3]; 10] = [
    [0x00, 0x00, 0x04],
    [0x1b, 0x0c, 0x41],
    [0x4a, 0x0c, 0x6b],
    [0x78, 0x1c, 0x6d],
    [0xa5, 0x2c, 0x60],
    [0xcf, 0x44, 0x46],
    [0xed, 0x69, 0x25],
    [0xfb, 0x9b, 0x06],
    [0xf7, 0xd1, 0x3d],
    [0xfc, 0xff, 0xa4],
];

fn interpolate(stops: &[[u8; 3]], t: f32) -> [u8; 3] {
    let position = t * (stops.len() - 1) as f32;
    let index = (position.floor() as usize).min(stops.len() - 2);
    let fraction = position - index as f32;
    let (a, b) = (stops[index], stops[index + 1]);
    std::array::from_fn(|c| {
        (f32::from(a[c]) + (f32::from(b[c]) - f32::from(a[c])) * fraction).round() as u8
    })
}

/// How a heat map is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatStyle {
    pub color_map: ColorMap,
    /// 0–1.
    pub opacity: f32,
    /// Values below this (0–1) are not drawn.
    pub threshold: f32,
}

impl Default for HeatStyle {
    fn default() -> Self {
        Self {
            color_map: ColorMap::Viridis,
            opacity: 0.5,
            threshold: 0.0,
        }
    }
}

/// Premultiplied RGBA8 of a `width` × `height` block of `map` starting at
/// (`x0`, `y0`).
pub fn render_block(
    map: &HeatMap,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    style: &HeatStyle,
) -> Vec<u8> {
    let opacity = style.opacity.clamp(0.0, 1.0);
    let mut out = vec![0u8; width as usize * height as usize * 4];
    for row in 0..height {
        let source_row = (y0 + row) as usize * map.width as usize + x0 as usize;
        for column in 0..width as usize {
            let source = source_row + column;
            let (rgb, alpha) = match &map.values {
                HeatValues::Scalar(values) => {
                    let v = values[source];
                    // At zero threshold, exact zeros (background) are not drawn.
                    if !v.is_finite() || v < style.threshold || (style.threshold <= 0.0 && v <= 0.0)
                    {
                        continue;
                    }
                    (style.color_map.color(v), opacity)
                }
                HeatValues::Color(rgba) => {
                    let p = &rgba[source * 4..source * 4 + 4];
                    if p[3] == 0 {
                        continue;
                    }
                    ([p[0], p[1], p[2]], opacity * f32::from(p[3]) / 255.0)
                }
            };
            let target = (row as usize * width as usize + column) * 4;
            for c in 0..3 {
                out[target + c] = (f32::from(rgb[c]) * alpha).round() as u8;
            }
            out[target + 3] = (alpha * 255.0).round() as u8;
        }
    }
    out
}

/// A rectangle in level-0 slide pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlideRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Where a heat map is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    /// Stretched over the slide's level-0 image.
    FitSlide,
    /// Stretched over the scanned region (`openslide.bounds-*`).
    FitBounds,
    /// From the slide's top-left corner, one map pixel per this many
    /// level-0 pixels.
    Downsample(f64),
    /// From the top-left corner, one map pixel per this many micrometres.
    Micrometres(f64),
    /// Where the file says it lies ([`HeatMap::geometry`]).
    FromFile,
}

impl Placement {
    pub fn id(&self) -> &'static str {
        match self {
            Self::FitSlide => "fit-slide",
            Self::FitBounds => "fit-bounds",
            Self::Downsample(_) => "downsample",
            Self::Micrometres(_) => "micrometres",
            Self::FromFile => "from-file",
        }
    }
}

/// The level-0 rectangle the map covers. `file` is what the map records
/// about its placement ([`HeatMap::geometry`]).
pub fn placement_rect(
    map_width: u32,
    map_height: u32,
    placement: Placement,
    slide: (u64, u64),
    bounds: Option<SlideRect>,
    mpp: Option<(f64, f64)>,
    file: Option<FileGeometry>,
) -> Result<SlideRect, String> {
    let (w, h) = (f64::from(map_width), f64::from(map_height));
    let rect = match placement {
        Placement::FromFile => {
            let file = file.ok_or_else(|| "the file does not say where it lies".to_string())?;
            SlideRect {
                x: file.x,
                y: file.y,
                width: w * file.scale_x,
                height: h * file.scale_y,
            }
        }
        Placement::FitSlide => SlideRect {
            x: 0.0,
            y: 0.0,
            width: slide.0 as f64,
            height: slide.1 as f64,
        },
        Placement::FitBounds => {
            bounds.ok_or_else(|| "the slide does not report a scanned region".to_string())?
        }
        Placement::Downsample(factor) => {
            if !(factor.is_finite() && factor > 0.0) {
                return Err("the downsample must be a positive number".to_string());
            }
            SlideRect {
                x: 0.0,
                y: 0.0,
                width: w * factor,
                height: h * factor,
            }
        }
        Placement::Micrometres(um) => {
            if !(um.is_finite() && um > 0.0) {
                return Err("the pixel size must be a positive number".to_string());
            }
            let (mpp_x, mpp_y) = mpp.ok_or_else(|| {
                "the slide has no pixel size; set it in Slide Information, or place the map another way"
                    .to_string()
            })?;
            SlideRect {
                x: 0.0,
                y: 0.0,
                width: w * um / mpp_x,
                height: h * um / mpp_y,
            }
        }
    };
    if rect.width > 0.0 && rect.height > 0.0 {
        Ok(rect)
    } else {
        Err("the heat map would cover nothing".to_string())
    }
}

/// The tiles of a `width` × `height` map: (x0, y0, w, h) in map pixels.
pub fn tiles(width: u32, height: u32) -> Vec<(u32, u32, u32, u32)> {
    let mut tiles = Vec::new();
    for y0 in (0..height).step_by(TILE as usize) {
        for x0 in (0..width).step_by(TILE as usize) {
            tiles.push((x0, y0, TILE.min(width - x0), TILE.min(height - y0)));
        }
    }
    tiles
}

/// The level-0 rectangle of a tile of a map covering `rect`.
pub fn tile_rect(
    rect: &SlideRect,
    map_width: u32,
    map_height: u32,
    tile: (u32, u32, u32, u32),
) -> SlideRect {
    let sx = rect.width / f64::from(map_width);
    let sy = rect.height / f64::from(map_height);
    SlideRect {
        x: rect.x + f64::from(tile.0) * sx,
        y: rect.y + f64::from(tile.1) * sy,
        width: f64::from(tile.2) * sx,
        height: f64::from(tile.3) * sy,
    }
}

/// A `width` × 1 strip of the colour map, straight RGBA8, for a legend.
pub fn legend(color_map: ColorMap, width: u32) -> Vec<u8> {
    (0..width)
        .flat_map(|x| {
            let [r, g, b] = color_map.color(x as f32 / (width.max(2) - 1) as f32);
            [r, g, b, 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(color: image::ColorType, width: u32, height: u32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut bytes),
            data,
            width,
            height,
            color,
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes
    }

    #[test]
    fn grey_pngs_are_values_and_colour_ones_keep_their_colours() {
        let map = decode_bytes(&png(image::ColorType::L8, 2, 1, &[0, 255])).unwrap();
        assert_eq!(map.values, HeatValues::Scalar(vec![0.0, 1.0]));
        // Grey stored as RGB is still a value map.
        let map = decode_bytes(&png(image::ColorType::Rgb8, 1, 1, &[51, 51, 51])).unwrap();
        assert_eq!(map.values, HeatValues::Scalar(vec![0.2]));
        let map = decode_bytes(&png(image::ColorType::Rgba8, 1, 1, &[255, 0, 0, 128])).unwrap();
        assert_eq!(map.values, HeatValues::Color(vec![255, 0, 0, 128]));
        // Transparent grey pixels have no value.
        let map = decode_bytes(&png(image::ColorType::La8, 1, 1, &[200, 0])).unwrap();
        assert!(matches!(&map.values, HeatValues::Scalar(v) if v[0].is_nan()));
    }

    #[test]
    fn float_tiffs_are_scaled_from_their_range() {
        let mut bytes = Vec::new();
        {
            let mut encoder =
                tiff::encoder::TiffEncoder::new(std::io::Cursor::new(&mut bytes)).unwrap();
            encoder
                .write_image::<tiff::encoder::colortype::Gray32Float>(3, 1, &[-1.0, 1.0, 3.0])
                .unwrap();
        }
        let map = decode_bytes(&bytes).unwrap();
        assert_eq!(map.values, HeatValues::Scalar(vec![0.0, 0.5, 1.0]));
        assert!(map.scaling.contains("-1 to 3"), "{}", map.scaling);

        let mut bytes = Vec::new();
        {
            let mut encoder =
                tiff::encoder::TiffEncoder::new(std::io::Cursor::new(&mut bytes)).unwrap();
            encoder
                .write_image::<tiff::encoder::colortype::Gray32Float>(2, 1, &[0.25, f32::NAN])
                .unwrap();
        }
        let map = decode_bytes(&bytes).unwrap();
        let HeatValues::Scalar(values) = &map.values else {
            panic!("a value map")
        };
        assert_eq!(values[0], 0.25, "probabilities are kept as they are");
        assert!(values[1].is_nan());

        let mut bytes = Vec::new();
        {
            let mut encoder =
                tiff::encoder::TiffEncoder::new(std::io::Cursor::new(&mut bytes)).unwrap();
            encoder
                .write_image::<tiff::encoder::colortype::Gray16>(2, 1, &[0, 65535])
                .unwrap();
        }
        assert_eq!(
            decode_bytes(&bytes).unwrap().values,
            HeatValues::Scalar(vec![0.0, 1.0])
        );
    }

    #[test]
    fn pngs_can_say_where_they_lie() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(std::io::Cursor::new(&mut bytes), 2, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .add_text_chunk("eov:level0_px_per_pixel".into(), "256,128".into())
                .unwrap();
            encoder
                .add_text_chunk("eov:origin".into(), "1000, 2000".into())
                .unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[10, 200]).unwrap();
        }
        let map = decode_bytes(&bytes).unwrap();
        assert_eq!(
            map.geometry,
            Some(FileGeometry {
                x: 1000.0,
                y: 2000.0,
                scale_x: 256.0,
                scale_y: 128.0
            })
        );
        let rect =
            placement_rect(2, 1, Placement::FromFile, (1, 1), None, None, map.geometry).unwrap();
        assert_eq!(
            (rect.x, rect.y, rect.width, rect.height),
            (1000.0, 2000.0, 512.0, 128.0)
        );
        // eovae's key, one number for both axes; no origin means 0, 0.
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(std::io::Cursor::new(&mut bytes), 1, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder
                .add_text_chunk("eovae:level0_px_per_pixel".into(), "512".into())
                .unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[1]).unwrap();
        }
        let geometry = decode_bytes(&bytes).unwrap().geometry.unwrap();
        assert_eq!(
            (geometry.x, geometry.scale_x, geometry.scale_y),
            (0.0, 512.0, 512.0)
        );
        assert!(
            placement_rect(1, 1, Placement::FromFile, (1, 1), None, None, None).is_err(),
            "a map without a placement cannot be placed from the file"
        );
    }

    #[test]
    fn oversized_and_unreadable_files_are_refused() {
        assert!(decode_bytes(b"not an image").is_err());
        assert!(check_size(8192, 8192).is_ok());
        assert!(check_size(8193, 8192).unwrap_err().contains("megapixels"));
    }

    #[test]
    fn colour_maps_run_from_dark_to_light() {
        assert_eq!(ColorMap::Viridis.color(0.0), [0x44, 0x01, 0x54]);
        assert_eq!(ColorMap::Viridis.color(1.0), [0xfd, 0xe7, 0x25]);
        assert_eq!(ColorMap::Inferno.color(1.0), [0xfc, 0xff, 0xa4]);
        assert_eq!(ColorMap::Gray.color(0.5), [128, 128, 128]);
        assert_eq!(ColorMap::Jet.color(0.0), [0, 0, 128]);
        assert_eq!(ColorMap::Jet.color(1.0), [128, 0, 0]);
        let luminance =
            |c: [u8; 3]| 0.3 * f32::from(c[0]) + 0.59 * f32::from(c[1]) + 0.11 * f32::from(c[2]);
        for map in [ColorMap::Viridis, ColorMap::Inferno] {
            let steps: Vec<f32> = (0..=10)
                .map(|i| luminance(map.color(i as f32 / 10.0)))
                .collect();
            assert!(
                steps.windows(2).all(|w| w[1] > w[0]),
                "{map:?} is monotonic: {steps:?}"
            );
        }
        assert_eq!(ColorMap::from_id("inferno"), Some(ColorMap::Inferno));
    }

    #[test]
    fn blocks_skip_background_and_premultiply() {
        let map = HeatMap {
            width: 3,
            height: 1,
            values: HeatValues::Scalar(vec![0.0, 0.4, 1.0]),
            scaling: String::new(),
            geometry: None,
        };
        let style = HeatStyle {
            color_map: ColorMap::Gray,
            opacity: 0.5,
            threshold: 0.0,
        };
        let block = render_block(&map, 0, 0, 3, 1, &style);
        assert_eq!(&block[0..4], &[0, 0, 0, 0], "zero is background");
        assert_eq!(&block[8..12], &[128, 128, 128, 128]);
        let above = render_block(
            &map,
            1,
            0,
            2,
            1,
            &HeatStyle {
                threshold: 0.5,
                ..style
            },
        );
        assert_eq!(&above[0..4], &[0, 0, 0, 0], "below the threshold");
        assert_eq!(above[7], 128);
    }

    #[test]
    fn maps_are_placed_and_tiled_in_slide_pixels() {
        let fit = placement_rect(
            100,
            50,
            Placement::FitSlide,
            (40_000, 20_000),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!((fit.width, fit.height), (40_000.0, 20_000.0));
        let down = placement_rect(
            100,
            50,
            Placement::Downsample(256.0),
            (1, 1),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!((down.width, down.height), (25_600.0, 12_800.0));
        let um = placement_rect(
            100,
            50,
            Placement::Micrometres(64.0),
            (1, 1),
            None,
            Some((0.25, 0.5)),
            None,
        )
        .unwrap();
        assert_eq!((um.width, um.height), (25_600.0, 6_400.0));
        assert!(
            placement_rect(1, 1, Placement::Micrometres(1.0), (1, 1), None, None, None).is_err()
        );
        assert!(placement_rect(1, 1, Placement::FitBounds, (1, 1), None, None, None).is_err());

        let tiles = tiles(2500, 1000);
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[2], (2048, 0, 452, 1000));
        let rect = SlideRect {
            x: 100.0,
            y: 0.0,
            width: 25_000.0,
            height: 10_000.0,
        };
        let last = tile_rect(&rect, 2500, 1000, tiles[2]);
        assert_eq!((last.x, last.width), (100.0 + 20_480.0, 4_520.0));
    }
}
