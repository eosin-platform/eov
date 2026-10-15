//! Export file formats, metadata and streaming file output (EXP-3).
//!
//! Every format records the print resolution in its standard field (PNG
//! `pHYs`, JFIF density, TIFF `X/YResolution`) and the specimen pixel size
//! (µm per output pixel) plus processing provenance as text (PNG `tEXt` /
//! `iTXt`, JPEG `COM`, TIFF `ImageDescription`). The JPEG quality setting is
//! honoured. Files are written to a temporary name and renamed when
//! complete, so a failed or cancelled export never leaves a truncated file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::jpeg::{JpegOptions, JpegWriter, Subsampling};
use super::render::{
    BandSink, ExportError, ExportReport, ExportRequest, ExportSource, render_export_bands,
};
use super::tiff::{TiffCompression, TiffOptions, TiffWriter};
use crate::job::JobControl;

/// Output file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "kebab-case")]
pub enum ExportFormat {
    Png,
    /// Baseline JPEG; `quality` 1–100. Chroma is not subsampled at 90 and
    /// above.
    Jpeg {
        quality: u8,
    },
    Tiff {
        compression: TiffCompression,
    },
}

impl ExportFormat {
    /// Preferred file extension (without the dot).
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpg",
            Self::Tiff { .. } => "tif",
        }
    }

    /// Every extension of the format (for file-dialog filters).
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Png => &["png"],
            Self::Jpeg { .. } => &["jpg", "jpeg"],
            Self::Tiff { .. } => &["tif", "tiff"],
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Png => "PNG image",
            Self::Jpeg { .. } => "JPEG image",
            Self::Tiff { .. } => "TIFF image",
        }
    }

    /// Makes `path` end in this format's extension: a missing extension is
    /// appended, another image format's extension is replaced, and any other
    /// extension is kept and the image extension appended (`fig.v2` →
    /// `fig.v2.png`). The written format is always `self`, never inferred
    /// from the name.
    pub fn ensure_extension(&self, path: &Path) -> PathBuf {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext {
            Some(e) if self.extensions().contains(&e.as_str()) => path.to_path_buf(),
            Some(e) if is_image_extension(&e) => path.with_extension(self.extension()),
            Some(_) => {
                let mut name = path.as_os_str().to_os_string();
                name.push(".");
                name.push(self.extension());
                PathBuf::from(name)
            }
            None => path.with_extension(self.extension()),
        }
    }
}

fn is_image_extension(ext: &str) -> bool {
    matches!(
        ext,
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "gif" | "webp"
    )
}

/// Metadata written into exported files.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Print resolution (pixels per inch).
    pub dpi: Option<f64>,
    /// Specimen size of one output pixel, µm (x, y).
    pub um_per_px: (Option<f64>, Option<f64>),
    /// Human-readable provenance (slide, region, processing).
    pub description: Option<String>,
    pub software: Option<String>,
    /// The colour space of the pixels: the slide's own ICC profile, to
    /// embed untouched (colours as scanned), or sRGB (colour-managed).
    #[serde(skip)]
    pub color_space: ColorSpace,
}

/// What exported pixels' colours mean.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ColorSpace {
    /// Unknown (no profile is written).
    #[default]
    Unspecified,
    /// sRGB: PNGs get an sRGB chunk.
    Srgb,
    /// The slide's colour space, given by its ICC profile (embedded: PNG
    /// iCCP, JPEG APP2, TIFF tag 34675).
    Profile(std::sync::Arc<Vec<u8>>),
}

impl ImageMetadata {
    /// Metadata for `request` on a slide with the given level-0 pixel size.
    /// `provenance` should come from the processing pipeline's summary (and
    /// may name the slide; leave names out for de-identified exports).
    pub fn for_export(
        request: &ExportRequest,
        dpi: f64,
        mpp_x: Option<f64>,
        mpp_y: Option<f64>,
        provenance: Option<String>,
    ) -> Self {
        Self {
            dpi: (dpi.is_finite() && dpi > 0.0).then_some(dpi),
            um_per_px: request.output_pixel_size_um(mpp_x, mpp_y),
            description: provenance,
            software: Some(software_name()),
            color_space: ColorSpace::Unspecified,
        }
    }

    /// The same metadata with the pixels' colour space.
    pub fn with_color_space(mut self, color_space: ColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    fn icc_profile(&self) -> Option<&[u8]> {
        match &self.color_space {
            ColorSpace::Profile(bytes) if !bytes.is_empty() => Some(bytes),
            _ => None,
        }
    }

    /// The pixel-size sentence used in text fields.
    pub fn pixel_size_text(&self) -> Option<String> {
        match self.um_per_px {
            (Some(x), Some(y)) => Some(format!("Pixel size: {x:.6} × {y:.6} µm")),
            (Some(v), None) | (None, Some(v)) => Some(format!("Pixel size: {v:.6} µm")),
            (None, None) => None,
        }
    }

    /// Description and pixel size as one text block.
    pub fn text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(d) = &self.description {
            parts.push(d.clone());
        }
        if let Some(p) = self.pixel_size_text() {
            parts.push(p);
        }
        parts.join("\n")
    }
}

/// `"eov <version>"`.
pub fn software_name() -> String {
    format!("eov {}", env!("CARGO_PKG_VERSION"))
}

fn encode_err(e: impl std::fmt::Display) -> ExportError {
    ExportError::Encode(e.to_string())
}

/// A buffered file that remembers the first I/O error, including errors
/// from the final flush in `Drop` (the PNG encoder writes its trailer and
/// flushes while being dropped, where errors would otherwise be lost).
struct TrackedFile {
    inner: Option<BufWriter<File>>,
    error: std::sync::Arc<std::sync::Mutex<Option<std::io::Error>>>,
}

impl TrackedFile {
    fn record(&self, err: &std::io::Error) {
        let mut slot = self.error.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(std::io::Error::new(err.kind(), err.to_string()));
        }
    }
}

impl Write for TrackedFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let result = self.inner.as_mut().expect("open").write(buf);
        if let Err(e) = &result {
            self.record(e);
        }
        result
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let result = self.inner.as_mut().expect("open").flush();
        if let Err(e) = &result {
            self.record(e);
        }
        result
    }
}

impl Drop for TrackedFile {
    fn drop(&mut self) {
        if let Some(mut writer) = self.inner.take()
            && let Err(e) = writer.flush()
        {
            self.record(&e);
        }
    }
}

enum Encoder {
    Png(Box<png::StreamWriter<'static, TrackedFile>>),
    Jpeg(Box<JpegWriter<BufWriter<File>>>),
    Tiff(Box<TiffWriter<BufWriter<File>>>),
}

/// Streams bands into a file of the given format.
pub struct FileSink {
    path: PathBuf,
    temp: PathBuf,
    format: ExportFormat,
    metadata: ImageMetadata,
    encoder: Option<Encoder>,
    png_error: std::sync::Arc<std::sync::Mutex<Option<std::io::Error>>>,
    rgb: Vec<u8>,
    width: u32,
}

impl FileSink {
    /// Writes to `path` (via a temporary file in the same directory).
    pub fn new(path: &Path, format: ExportFormat, metadata: ImageMetadata) -> Self {
        let mut temp = path.as_os_str().to_os_string();
        temp.push(".part");
        Self {
            path: path.to_path_buf(),
            temp: PathBuf::from(temp),
            format,
            metadata,
            encoder: None,
            png_error: Default::default(),
            rgb: Vec::new(),
            width: 0,
        }
    }

    /// Removes the partial file (after an error or cancellation).
    pub fn abort(&mut self) {
        self.encoder = None;
        let _ = std::fs::remove_file(&self.temp);
    }
}

impl Drop for FileSink {
    fn drop(&mut self) {
        if self.encoder.is_some() {
            self.abort();
        }
    }
}

impl BandSink for FileSink {
    fn begin(&mut self, width: u32, height: u32) -> Result<(), ExportError> {
        self.width = width;
        let file = BufWriter::with_capacity(1 << 20, File::create(&self.temp)?);
        let text = self.metadata.text();
        self.encoder = Some(match self.format {
            ExportFormat::Png => {
                let file = TrackedFile {
                    inner: Some(file),
                    error: self.png_error.clone(),
                };
                let mut info = png::Info::with_size(width, height);
                if let Some(profile) = self.metadata.icc_profile() {
                    info.icc_profile = Some(std::borrow::Cow::Owned(profile.to_vec()));
                }
                let mut encoder = png::Encoder::with_info(file, info).map_err(encode_err)?;
                if self.metadata.color_space == ColorSpace::Srgb {
                    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
                }
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                encoder.set_compression(png::Compression::Balanced);
                if let Some(dpi) = self.metadata.dpi {
                    let ppm = (dpi / 0.0254).round() as u32;
                    encoder.set_pixel_dims(Some(png::PixelDimensions {
                        xppu: ppm,
                        yppu: ppm,
                        unit: png::Unit::Meter,
                    }));
                }
                if let Some(software) = &self.metadata.software {
                    encoder
                        .add_text_chunk("Software".into(), software.clone())
                        .map_err(encode_err)?;
                }
                if !text.is_empty() {
                    encoder
                        .add_itxt_chunk("Description".into(), text.clone())
                        .map_err(encode_err)?;
                }
                if let Some(x) = self.metadata.um_per_px.0 {
                    encoder
                        .add_text_chunk("eov.mpp-x".into(), format!("{x:.6}"))
                        .map_err(encode_err)?;
                }
                if let Some(y) = self.metadata.um_per_px.1 {
                    encoder
                        .add_text_chunk("eov.mpp-y".into(), format!("{y:.6}"))
                        .map_err(encode_err)?;
                }
                let writer = encoder.write_header().map_err(encode_err)?;
                Encoder::Png(Box::new(writer.into_stream_writer().map_err(encode_err)?))
            }
            ExportFormat::Jpeg { quality } => {
                let options = JpegOptions {
                    quality: quality.clamp(1, 100),
                    subsampling: if quality >= 90 {
                        Subsampling::None
                    } else {
                        Subsampling::Half
                    },
                    dpi: self
                        .metadata
                        .dpi
                        .map(|d| d.round().clamp(1.0, u16::MAX as f64) as u16),
                    comment: (!text.is_empty()).then(|| match &self.metadata.software {
                        Some(s) => format!("{s}\n{text}"),
                        None => text.clone(),
                    }),
                    icc_profile: self.metadata.icc_profile().map(<[u8]>::to_vec),
                };
                Encoder::Jpeg(Box::new(JpegWriter::new(file, width, height, &options)?))
            }
            ExportFormat::Tiff { compression } => {
                let mut description = text.clone();
                if let Some(x) = self.metadata.um_per_px.0 {
                    // Aperio-style key, read by several WSI tools.
                    description.push_str(&format!("\nMPP = {x:.6}"));
                }
                let options = TiffOptions {
                    compression,
                    dpi: self.metadata.dpi,
                    description: (!description.trim().is_empty()).then_some(description),
                    software: self.metadata.software.clone(),
                    icc_profile: self.metadata.icc_profile().map(<[u8]>::to_vec),
                    ..TiffOptions::default()
                };
                Encoder::Tiff(Box::new(TiffWriter::new(file, width, height, options)?))
            }
        });
        Ok(())
    }

    fn write_rows(&mut self, rgba: &[u8], rows: u32) -> Result<(), ExportError> {
        self.rgb.clear();
        self.rgb.reserve(rows as usize * self.width as usize * 3);
        for px in rgba.as_chunks::<4>().0 {
            self.rgb.extend_from_slice(&px[..3]);
        }
        match self.encoder.as_mut() {
            Some(Encoder::Png(w)) => w.write_all(&self.rgb)?,
            Some(Encoder::Jpeg(w)) => w.write_rows(&self.rgb, rows)?,
            Some(Encoder::Tiff(w)) => w.write_rows(&self.rgb, rows)?,
            None => return Err(ExportError::Encode("encoder not started".into())),
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ExportError> {
        match self.encoder.take() {
            Some(Encoder::Png(w)) => {
                // Writes the remaining data and the IEND chunk, then drops
                // (and flushes) the file.
                w.finish().map_err(encode_err)?;
                if let Some(err) = self
                    .png_error
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                {
                    return Err(ExportError::Io(err));
                }
            }
            Some(Encoder::Jpeg(w)) => {
                w.finish()?.flush()?;
            }
            Some(Encoder::Tiff(w)) => {
                w.finish()?.flush()?;
            }
            None => return Err(ExportError::Encode("encoder not started".into())),
        }
        std::fs::rename(&self.temp, &self.path)?;
        Ok(())
    }
}

/// Renders `request` and writes it to `path` in `format` (with `metadata`).
/// The path is used as given; call [`ExportFormat::ensure_extension`] first
/// to fix the extension. On error or cancellation no file is left behind.
pub fn export_to_file(
    source: &ExportSource,
    request: &ExportRequest,
    path: &Path,
    format: ExportFormat,
    metadata: ImageMetadata,
    control: &JobControl,
) -> Result<ExportReport, ExportError> {
    let mut sink = FileSink::new(path, format, metadata);
    match render_export_bands(source, request, control, &mut sink) {
        Ok(report) => Ok(report),
        Err(err) => {
            sink.abort();
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_are_fixed_without_trusting_the_name() {
        let png = ExportFormat::Png;
        let jpg = ExportFormat::Jpeg { quality: 90 };
        assert_eq!(
            png.ensure_extension(Path::new("/x/fig")),
            PathBuf::from("/x/fig.png")
        );
        assert_eq!(
            png.ensure_extension(Path::new("/x/fig.PNG")),
            PathBuf::from("/x/fig.PNG")
        );
        assert_eq!(
            jpg.ensure_extension(Path::new("/x/fig.png")),
            PathBuf::from("/x/fig.jpg")
        );
        assert_eq!(
            jpg.ensure_extension(Path::new("/x/fig.jpeg")),
            PathBuf::from("/x/fig.jpeg")
        );
        assert_eq!(
            png.ensure_extension(Path::new("/x/fig.v2")),
            PathBuf::from("/x/fig.v2.png")
        );
        let tif = ExportFormat::Tiff {
            compression: TiffCompression::Deflate,
        };
        assert_eq!(
            tif.ensure_extension(Path::new("a.jpg")),
            PathBuf::from("a.tif")
        );
    }

    #[test]
    fn metadata_text_includes_pixel_size() {
        let meta = ImageMetadata {
            dpi: Some(300.0),
            um_per_px: (Some(0.25), Some(0.25)),
            description: Some("Colours as scanned.".into()),
            software: Some(software_name()),
            color_space: ColorSpace::Unspecified,
        };
        let text = meta.text();
        assert!(text.contains("Colours as scanned.") && text.contains("0.250000 × 0.250000 µm"));
    }

    /// Writes a 4×2 image in `format` with `color_space`; returns the file.
    fn write_with(format: ExportFormat, color_space: ColorSpace) -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("out.{}", format.extension()));
        let metadata = ImageMetadata {
            software: Some(software_name()),
            ..ImageMetadata::default()
        }
        .with_color_space(color_space);
        let mut sink = FileSink::new(&path, format, metadata);
        sink.begin(4, 2).unwrap();
        sink.write_rows(&[200u8, 100, 150, 255].repeat(8), 2)
            .unwrap();
        sink.finish().unwrap();
        std::fs::read(&path).unwrap()
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Colours exported as scanned carry the slide's profile untouched;
    /// converted ones are labelled sRGB (PNG) or carry no profile.
    #[test]
    fn exports_carry_their_colour_space() {
        let profile: Vec<u8> = (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let embedded = ColorSpace::Profile(std::sync::Arc::new(profile.clone()));

        // PNG: iCCP holds the (compressed) profile; the decoder gives it back.
        let png = write_with(ExportFormat::Png, embedded.clone());
        let reader = png::Decoder::new(std::io::Cursor::new(&png))
            .read_info()
            .unwrap();
        assert_eq!(
            reader.info().icc_profile.as_deref(),
            Some(profile.as_slice())
        );
        assert!(find(&png, b"sRGB").is_none());
        let srgb = write_with(ExportFormat::Png, ColorSpace::Srgb);
        let reader = png::Decoder::new(std::io::Cursor::new(&srgb))
            .read_info()
            .unwrap();
        assert!(reader.info().srgb.is_some());
        assert!(reader.info().icc_profile.is_none());

        // JPEG: APP2 ICC_PROFILE segments, in order, rejoin to the profile.
        let jpeg = write_with(ExportFormat::Jpeg { quality: 90 }, embedded.clone());
        let mut rejoined = Vec::new();
        let mut at = 2;
        while at + 4 <= jpeg.len() && jpeg[at] == 0xFF && jpeg[at + 1] != 0xDA {
            let len = u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]) as usize;
            let body = &jpeg[at + 4..at + 2 + len];
            if jpeg[at + 1] == 0xE2 && body.starts_with(b"ICC_PROFILE\0") {
                assert_eq!(usize::from(body[13]), profile.len().div_ceil(65_519));
                rejoined.extend_from_slice(&body[14..]);
            }
            at += 2 + len;
        }
        assert_eq!(rejoined, profile);
        assert!(image::load_from_memory(&jpeg).is_ok(), "still a valid JPEG");
        assert!(
            find(
                &write_with(ExportFormat::Jpeg { quality: 90 }, ColorSpace::Srgb),
                b"ICC_PROFILE"
            )
            .is_none()
        );

        // TIFF: tag 34675.
        let tiff = write_with(
            ExportFormat::Tiff {
                compression: super::super::TiffCompression::Deflate,
            },
            embedded,
        );
        let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(&tiff)).unwrap();
        let tag = decoder
            .get_tag_u8_vec(tiff::tags::Tag::Unknown(34675))
            .unwrap();
        assert_eq!(tag, profile);
    }
}
