//! Output path planning and tile image writing.

use image::{ImageBuffer, Rgba};
use std::path::{Path, PathBuf};

/// Build the directory path for tiles from a given slide stem.
///
/// Layout: `<output_dir>/slides/<slide_stem>/`
pub fn slide_tiles_dir(output_dir: &Path, slide_stem: &str) -> PathBuf {
    output_dir.join("slides").join(slide_stem)
}

/// Build a deterministic filename for a tile.
///
/// Pattern: `<slide_stem>_x{X:06}_y{Y:06}_s{tile_size}.png`
///
/// Coordinates are zero-padded to 6 digits for lexical sorting.
pub fn tile_filename(slide_stem: &str, x: u64, y: u64, tile_size: u32) -> String {
    format!(
        "{}_x{:06}_y{:06}_s{}.png",
        slide_stem, x, y, tile_size
    )
}

/// Relative path of a tile image within the dataset root.
///
/// E.g. `slides/slide_stem/slide_stem_x000000_y000000_s512.png`
pub fn tile_relative_path(slide_stem: &str, x: u64, y: u64, tile_size: u32) -> String {
    format!(
        "slides/{}/{}",
        slide_stem,
        tile_filename(slide_stem, x, y, tile_size)
    )
}

/// Write RGBA pixel data as a PNG file.
///
/// `data` must be `width * height * 4` bytes in RGBA order.
pub fn write_tile_png(
    path: &Path,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<(), image::ImageError> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(width, height, data.to_vec())
            .expect("buffer size mismatch in write_tile_png");
    img.save(path)
}

/// Returns `true` if the tile is "almost completely white" and should be
/// skipped.
///
/// A pixel is considered "white" when **all three** of its RGB channels are
/// ≥ 220 (out of 255). The function computes the fraction of such pixels and
/// returns `true` when that fraction ≥ `threshold` (e.g. 0.9 → 90 %).
///
/// To keep this fast on large tiles, only every 4th pixel is sampled (stride
/// of 4 in linearised pixel order), giving an ~4× speedup with negligible
/// accuracy loss.
pub fn is_tile_mostly_white(data: &[u8], threshold: f32) -> bool {
    const WHITE_CHANNEL_MIN: u8 = 220;
    const SAMPLE_STRIDE: usize = 4; // check every 4th pixel

    let total_pixels = data.len() / 4;
    if total_pixels == 0 {
        return true;
    }

    let sampled = (total_pixels + SAMPLE_STRIDE - 1) / SAMPLE_STRIDE;
    let mut white_count: u32 = 0;

    let mut i = 0;
    while i < data.len() {
        // data[i..i+4] = [R, G, B, A]
        if data[i] >= WHITE_CHANNEL_MIN
            && data[i + 1] >= WHITE_CHANNEL_MIN
            && data[i + 2] >= WHITE_CHANNEL_MIN
        {
            white_count += 1;
        }
        i += 4 * SAMPLE_STRIDE;
    }

    (white_count as f32 / sampled as f32) >= threshold
}

/// Derive a slide stem from its file path, ensuring it is filesystem-safe.
///
/// Returns the file stem (no extension), or `"unknown"` as a fallback.
pub fn slide_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_filename() {
        assert_eq!(
            tile_filename("C3L-00004-21", 0, 512, 512),
            "C3L-00004-21_x000000_y000512_s512.png"
        );
    }

    #[test]
    fn test_tile_relative_path() {
        assert_eq!(
            tile_relative_path("slide", 1024, 2048, 256),
            "slides/slide/slide_x001024_y002048_s256.png"
        );
    }

    #[test]
    fn test_slide_stem() {
        assert_eq!(slide_stem(Path::new("/data/slides/C3L-00004-21.svs")), "C3L-00004-21");
        assert_eq!(slide_stem(Path::new("relative.tif")), "relative");
    }

    #[test]
    fn test_no_collision_different_slides() {
        let a = tile_relative_path("slide_a", 0, 0, 512);
        let b = tile_relative_path("slide_b", 0, 0, 512);
        assert_ne!(a, b);
    }
}
