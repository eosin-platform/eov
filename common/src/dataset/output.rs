//! Output paths, slide keys, pixel handling and tile encoding.

use std::path::{Path, PathBuf};

use super::config::PatchImageFormat;
use super::discovery::SlideInput;

/// Deterministic tile file name, `<key>_x{X:06}_y{Y:06}_s{tile_size}.<ext>`
/// (level-0 origin, zero-padded for lexical sorting). Tiles live under
/// `slides/<key>/`.
pub fn tile_filename_with_extension(
    slide_key: &str,
    x: u64,
    y: u64,
    tile_size: u32,
    extension: &str,
) -> String {
    format!("{slide_key}_x{x:06}_y{y:06}_s{tile_size}.{extension}")
}

/// Mask path relative to the dataset root (always PNG).
pub fn mask_relative_path(slide_key: &str, x: u64, y: u64, tile_size: u32) -> String {
    format!(
        "masks/{slide_key}/{}",
        tile_filename_with_extension(slide_key, x, y, tile_size, "png")
    )
}

/// Composites premultiplied RGBA (OpenSlide's format) onto white in place
/// and returns the number of pixels that were not opaque. Transparent "no
/// data" regions become glass-white instead of black (DS-5).
pub fn composite_onto_white(rgba: &mut [u8]) -> u64 {
    let mut non_opaque = 0;
    for p in rgba.as_chunks_mut::<4>().0 {
        if p[3] != 255 {
            let add = 255 - p[3];
            p[0] = p[0].saturating_add(add);
            p[1] = p[1].saturating_add(add);
            p[2] = p[2].saturating_add(add);
            p[3] = 255;
            non_opaque += 1;
        }
    }
    non_opaque
}

/// Channels at or above this are "white".
pub const WHITE_CHANNEL_MIN: u8 = 220;

/// Fraction of pixels that are nearly white (all channels ≥ 220) or not
/// opaque (alpha < 255 counts as background).
pub fn white_fraction(rgba: &[u8]) -> f32 {
    let total = rgba.len() / 4;
    if total == 0 {
        return 1.0;
    }
    let white = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| {
            p[3] < 255
                || (p[0] >= WHITE_CHANNEL_MIN
                    && p[1] >= WHITE_CHANNEL_MIN
                    && p[2] >= WHITE_CHANNEL_MIN)
        })
        .count();
    white as f32 / total as f32
}

fn png_bytes(
    data: &[u8],
    width: u32,
    height: u32,
    color: png::ColorType,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() / 2);
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        writer
            .write_image_data(data)
            .map_err(std::io::Error::other)?;
        writer.finish().map_err(std::io::Error::other)?;
    }
    Ok(out)
}

/// Encodes an opaque RGBA tile as an RGB image file (no alpha, DS-5).
pub fn encode_tile(
    rgba: &[u8],
    width: u32,
    height: u32,
    format: PatchImageFormat,
) -> std::io::Result<Vec<u8>> {
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    match format {
        PatchImageFormat::Png => png_bytes(&rgb, width, height, png::ColorType::Rgb),
        PatchImageFormat::Jpeg { quality } => {
            let options = crate::export::JpegOptions {
                quality,
                subsampling: if quality >= 90 {
                    crate::export::Subsampling::None
                } else {
                    crate::export::Subsampling::Half
                },
                dpi: None,
                comment: None,
                icc_profile: None,
            };
            let mut writer = crate::export::JpegWriter::new(Vec::new(), width, height, &options)?;
            writer.write_rows(&rgb, height)?;
            writer.finish()
        }
    }
}

/// Encodes an 8-bit single-channel mask as PNG.
pub fn encode_mask(mask: &[u8], width: u32, height: u32) -> std::io::Result<Vec<u8>> {
    png_bytes(mask, width, height, png::ColorType::Grayscale)
}

/// Writes `bytes` to `path` via a temporary file and a rename, so readers
/// (and resumed runs) never see a partial file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".part");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// A filesystem-safe stem: non-UTF-8 parts are converted lossily and
/// unsafe characters replaced with `_`; `"slide"` for empty stems.
pub fn slide_stem(path: &Path) -> String {
    sanitize(
        &path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )
}

fn sanitize(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\u{FFFD}' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let trimmed = sanitized.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if trimmed.is_empty() {
        "slide".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The readable base name of a slide: the file stem, or the directory name
/// for a grouped DICOM series (whose instance names are meaningless).
pub fn slide_base_name(input: &SlideInput) -> String {
    if input.series_uid.is_some()
        && let Some(dir) = input.path.parent().and_then(|p| p.file_name())
    {
        return sanitize(&dir.to_string_lossy());
    }
    slide_stem(&input.path)
}

fn unique_keys(names: Vec<String>, paths: &[PathBuf]) -> Vec<String> {
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in &names {
        *counts.entry(name.to_lowercase()).or_default() += 1;
    }
    names
        .into_iter()
        .zip(paths)
        .map(|(name, path)| {
            if counts.get(&name.to_lowercase()).copied().unwrap_or(0) <= 1 {
                return name;
            }
            let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
            let short: String = digest.iter().take(4).map(|b| format!("{b:02x}")).collect();
            format!("{name}_{short}")
        })
        .collect()
}

/// An opaque, filesystem-safe key for a de-identified dataset:
/// `slide-<12 hex digits>` from the slide's content identity, so the same
/// slide gets the same key in every run and at every site. A slide that
/// cannot be opened is keyed by its path's hash instead. Keys already in
/// `used` (two copies of one slide) get a `-2`, `-3`… suffix.
pub fn deidentified_key(
    wsi: Option<&crate::WsiFile>,
    path: &Path,
    used: &mut std::collections::HashSet<String>,
) -> String {
    use sha2::{Digest, Sha256};
    let digest = match wsi.and_then(|wsi| wsi.identity().ok()) {
        Some(identity) => Sha256::digest(identity.key().as_bytes()),
        None => {
            let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            Sha256::digest(canonical.to_string_lossy().as_bytes())
        }
    };
    let short: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let base = format!("slide-{short}");
    let mut key = base.clone();
    let mut n = 2;
    while !used.insert(key.clone()) {
        key = format!("{base}-{n}");
        n += 1;
    }
    key
}

/// A unique, filesystem-safe key per slide: its base name (the file stem,
/// or the directory of a DICOM series), with a short hash of the canonical
/// path appended when several slides share a name (compared
/// case-insensitively), so their tiles never overwrite one another (DS-2).
pub fn unique_slide_keys_for(inputs: &[SlideInput]) -> Vec<String> {
    let paths: Vec<PathBuf> = inputs.iter().map(|i| i.path.clone()).collect();
    unique_keys(inputs.iter().map(slide_base_name).collect(), &paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_filename() {
        assert_eq!(
            tile_filename_with_extension("C3L-00004-21", 0, 512, 512, "png"),
            "C3L-00004-21_x000000_y000512_s512.png"
        );
        assert_eq!(
            mask_relative_path("s", 0, 0, 8),
            "masks/s/s_x000000_y000000_s8.png"
        );
    }

    #[test]
    fn transparent_pixels_become_white_background() {
        // Fully transparent, half-covered premultiplied, and opaque pixels.
        let mut px = vec![0, 0, 0, 0, 50, 40, 30, 128, 10, 20, 30, 255];
        assert!(
            white_fraction(&px) > 0.6,
            "transparent counts as background"
        );
        assert_eq!(composite_onto_white(&mut px), 2);
        assert_eq!(&px[0..4], &[255, 255, 255, 255]);
        assert_eq!(&px[4..8], &[177, 167, 157, 255]);
        assert_eq!(&px[8..12], &[10, 20, 30, 255]);
        assert_eq!(white_fraction(&[0, 0, 0, 0]), 1.0);
    }

    #[test]
    fn tiles_are_written_as_rgb() {
        let rgba: Vec<u8> = (0..8 * 8).flat_map(|i| [i as u8, 100, 200, 255]).collect();
        let png = encode_tile(&rgba, 8, 8, PatchImageFormat::Png).unwrap();
        let img = image::load_from_memory(&png).unwrap();
        assert_eq!(img.color(), image::ColorType::Rgb8);
        assert_eq!(img.to_rgb8().get_pixel(3, 0).0, [3, 100, 200]);
        let jpg = encode_tile(&rgba, 8, 8, PatchImageFormat::Jpeg { quality: 95 }).unwrap();
        assert_eq!(
            image::load_from_memory(&jpg).unwrap().color(),
            image::ColorType::Rgb8
        );
        let mask = encode_mask(&[0, 255, 1, 2], 2, 2).unwrap();
        assert_eq!(
            image::load_from_memory(&mask).unwrap().color(),
            image::ColorType::L8
        );
    }

    #[test]
    fn slide_stems_and_keys() {
        assert_eq!(
            slide_stem(Path::new("/data/slides/C3L-00004-21.svs")),
            "C3L-00004-21"
        );
        assert_eq!(slide_stem(Path::new("/x/a:b*c.svs")), "a_b_c");
        assert_eq!(slide_stem(Path::new("/x/..tif")), "slide");
        let inputs: Vec<SlideInput> = [
            "/cases/001/HE.svs",
            "/cases/002/HE.svs",
            "/cases/002/he.ndpi",
            "/cases/003/IHC.svs",
        ]
        .iter()
        .map(|p| SlideInput {
            path: PathBuf::from(p),
            members: vec![],
            series_uid: None,
        })
        .collect();
        let keys = unique_slide_keys_for(&inputs);
        assert_eq!(keys[3], "IHC");
        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), keys.len());
        let dicom = SlideInput {
            path: PathBuf::from("/cases/case7/1.2.3.4.dcm"),
            members: vec![],
            series_uid: Some("1.2.3".into()),
        };
        assert_eq!(slide_base_name(&dicom), "case7");
    }
}
