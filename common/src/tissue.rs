//! Tissue detection for brightfield slides.
//!
//! Tissue is coloured and glass is near-neutral, so pixels are classified by
//! their HSV saturation `S = (max − min) / max`: a pixel is tissue when `S`
//! exceeds a threshold and it is not near-black (ink, folds, out-of-slide
//! fill). The threshold is either fixed or chosen per slide with Otsu's
//! method on a low-resolution level (as CLAM and similar tools do).
//! Transparent ("no data") pixels are never tissue.
//!
//! [`TissueMask`] holds the low-resolution classification with an integral
//! image, so the tissue fraction of any level-0 rectangle is an O(1) query.

use serde::{Deserialize, Serialize};

use crate::WsiFile;

/// How the tissue threshold is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case")]
pub enum TissueMethod {
    /// Otsu's threshold on the saturation histogram of the slide (clamped to
    /// `[0.04, 0.5]` so that empty or uniform slides behave sensibly).
    #[default]
    Otsu,
    /// A fixed saturation threshold in `(0, 1)`.
    Saturation { threshold: f32 },
}

/// Lower and upper bounds for an Otsu threshold.
const OTSU_RANGE: (f32, f32) = (0.04, 0.5);
/// Pixels whose brightest channel is below this are not tissue.
const MIN_VALUE: u8 = 20;

/// Saturation of an RGB pixel, 0–255.
#[inline(always)]
pub fn saturation(rgb: [u8; 3]) -> u8 {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    if max == 0 {
        return 0;
    }
    (((max - min) as u32 * 255 + max as u32 / 2) / max as u32) as u8
}

/// Whether one opaque pixel is tissue for a saturation threshold (0–1).
#[inline(always)]
pub fn is_tissue_pixel(rgb: [u8; 3], threshold: f32) -> bool {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    max >= MIN_VALUE && saturation(rgb) as f32 > threshold * 255.0
}

/// Fraction of tissue pixels in an RGBA buffer (alpha < 255 counts as
/// background).
pub fn tissue_fraction_rgba(rgba: &[u8], threshold: f32) -> f32 {
    let pixels = rgba.len() / 4;
    if pixels == 0 {
        return 0.0;
    }
    let tissue = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[3] == 255 && is_tissue_pixel([p[0], p[1], p[2]], threshold))
        .count();
    tissue as f32 / pixels as f32
}

/// Otsu's threshold of a 256-bin histogram, as a fraction of 255.
pub fn otsu_threshold(hist: &[u64; 256]) -> f32 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return OTSU_RANGE.0;
    }
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &n)| i as f64 * n as f64)
        .sum();
    let (mut w_b, mut sum_b) = (0f64, 0f64);
    let (mut best, mut best_t) = (-1f64, 0usize);
    for (t, &n) in hist.iter().enumerate() {
        w_b += n as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total as f64 - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * n as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            best_t = t;
        }
    }
    // Pixels with saturation > best_t are foreground.
    ((best_t as f32 + 0.5) / 255.0).clamp(OTSU_RANGE.0, OTSU_RANGE.1)
}

/// A low-resolution tissue mask of a slide.
#[derive(Debug, Clone)]
pub struct TissueMask {
    width: u32,
    height: u32,
    /// Level-0 pixels per mask pixel.
    scale_x: f64,
    scale_y: f64,
    /// Saturation threshold used (0–1).
    threshold: f32,
    /// Summed-area table, `(width + 1) × (height + 1)`.
    integral: Vec<u32>,
}

impl TissueMask {
    /// Builds a mask from RGBA pixels covering the whole slide, where one
    /// pixel spans `scale_x × scale_y` level-0 pixels.
    pub fn from_rgba(
        rgba: &[u8],
        width: u32,
        height: u32,
        scale_x: f64,
        scale_y: f64,
        method: TissueMethod,
    ) -> Self {
        let threshold = match method {
            TissueMethod::Saturation { threshold } => threshold.clamp(0.0, 1.0),
            TissueMethod::Otsu => {
                let mut hist = [0u64; 256];
                for p in rgba.as_chunks::<4>().0 {
                    if p[3] == 255 && p[0].max(p[1]).max(p[2]) >= MIN_VALUE {
                        hist[saturation([p[0], p[1], p[2]]) as usize] += 1;
                    }
                }
                otsu_threshold(&hist)
            }
        };
        let (w, h) = (width as usize, height as usize);
        let mut integral = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                let i = (y * w + x) * 4;
                let p = &rgba[i..i + 4];
                if p[3] == 255 && is_tissue_pixel([p[0], p[1], p[2]], threshold) {
                    row += 1;
                }
                integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row;
            }
        }
        Self {
            width,
            height,
            scale_x,
            scale_y,
            threshold,
            integral,
        }
    }

    /// Reads the finest level with at most `max_pixels` pixels (or the
    /// coarsest level, if it has at most four times that) and builds a mask.
    /// Returns `Ok(None)` for slides without a small enough level (callers
    /// then evaluate tissue on the patches themselves).
    pub fn compute(
        wsi: &WsiFile,
        method: TissueMethod,
        max_pixels: u64,
    ) -> crate::Result<Option<Self>> {
        let props = wsi.properties();
        let levels: Vec<(u32, crate::WsiLevel)> = (0..wsi.level_count())
            .filter_map(|l| wsi.level(l).map(|info| (l, info.clone())))
            .collect();
        let pixels = |info: &crate::WsiLevel| info.width.saturating_mul(info.height);
        let chosen = levels
            .iter()
            .find(|(_, info)| pixels(info) <= max_pixels)
            .or_else(|| {
                levels
                    .last()
                    .filter(|(_, info)| pixels(info) <= max_pixels.saturating_mul(4))
            })
            .cloned();
        let Some((level, info)) = chosen else {
            return Ok(None);
        };
        let (w, h) = (info.width as u32, info.height as u32);
        if w == 0 || h == 0 {
            return Ok(None);
        }
        let mut rgba = vec![0u8; w as usize * h as usize * 4];
        const CHUNK: u32 = 1024;
        for y0 in (0..h).step_by(CHUNK as usize) {
            for x0 in (0..w).step_by(CHUNK as usize) {
                let cw = CHUNK.min(w - x0);
                let ch = CHUNK.min(h - y0);
                let data = wsi.read_region(
                    (x0 as f64 * info.downsample).round() as i64,
                    (y0 as f64 * info.downsample).round() as i64,
                    level,
                    cw,
                    ch,
                )?;
                for row in 0..ch as usize {
                    let src = &data[row * cw as usize * 4..(row + 1) * cw as usize * 4];
                    let dst = ((y0 as usize + row) * w as usize + x0 as usize) * 4;
                    rgba[dst..dst + src.len()].copy_from_slice(src);
                }
            }
        }
        Ok(Some(Self::from_rgba(
            &rgba,
            w,
            h,
            props.width as f64 / w as f64,
            props.height as f64 / h as f64,
            method,
        )))
    }

    /// Saturation threshold (0–1) the mask was built with; use it to judge
    /// patches at full resolution consistently.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        let w = self.width as usize + 1;
        let a = self.integral[y1 * w + x1] as i64;
        let b = self.integral[y0 * w + x1] as i64;
        let c = self.integral[y1 * w + x0] as i64;
        let d = self.integral[y0 * w + x0] as i64;
        (a - b - c + d).max(0) as u64
    }

    /// Tissue fraction of a level-0 rectangle (mask pixels it touches).
    pub fn fraction(&self, x: f64, y: f64, width: f64, height: f64) -> f32 {
        let x0 = ((x / self.scale_x).floor().max(0.0) as usize).min(self.width as usize);
        let y0 = ((y / self.scale_y).floor().max(0.0) as usize).min(self.height as usize);
        let x1 = (((x + width) / self.scale_x).ceil().max(0.0) as usize).min(self.width as usize);
        let y1 = (((y + height) / self.scale_y).ceil().max(0.0) as usize).min(self.height as usize);
        if x1 <= x0 || y1 <= y0 {
            return 0.0;
        }
        self.sum(x0, y0, x1, y1) as f32 / ((x1 - x0) * (y1 - y0)) as f32
    }

    /// Tissue fraction of the whole slide.
    pub fn slide_fraction(&self) -> f32 {
        let total = self.width as u64 * self.height as u64;
        if total == 0 {
            0.0
        } else {
            self.sum(0, 0, self.width as usize, self.height as usize) as f32 / total as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32, tissue: impl Fn(u32, u32) -> bool) -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if tissue(x, y) {
                    out.extend_from_slice(&[190, 110, 170, 255]);
                } else {
                    out.extend_from_slice(&[238, 236, 240, 255]);
                }
            }
        }
        out
    }

    #[test]
    fn saturation_separates_tissue_from_glass() {
        assert!(is_tissue_pixel([190, 110, 170], 0.1));
        assert!(!is_tissue_pixel([238, 236, 240], 0.1));
        assert!(
            !is_tissue_pixel([10, 2, 5], 0.1),
            "near-black is not tissue"
        );
        assert_eq!(saturation([0, 0, 0]), 0);
        let rgba = [190u8, 110, 170, 255, 238, 236, 240, 255, 190, 110, 170, 0];
        assert!((tissue_fraction_rgba(&rgba, 0.1) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn otsu_finds_a_threshold_between_the_modes() {
        let rgba = image(100, 100, |x, _| x < 30);
        let mask = TissueMask::from_rgba(&rgba, 100, 100, 10.0, 10.0, TissueMethod::Otsu);
        let s_tissue = saturation([190, 110, 170]) as f32 / 255.0;
        let s_glass = saturation([238, 236, 240]) as f32 / 255.0;
        assert!(
            mask.threshold() > s_glass && mask.threshold() < s_tissue,
            "{}",
            mask.threshold()
        );
        assert!((mask.slide_fraction() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn rectangle_queries_use_level_zero_coordinates() {
        let rgba = image(100, 100, |x, y| x < 50 && y < 50);
        let mask = TissueMask::from_rgba(
            &rgba,
            100,
            100,
            16.0,
            16.0,
            TissueMethod::Saturation { threshold: 0.1 },
        );
        assert_eq!(mask.fraction(0.0, 0.0, 800.0, 800.0), 1.0);
        assert_eq!(mask.fraction(800.0, 800.0, 800.0, 800.0), 0.0);
        assert!((mask.fraction(0.0, 0.0, 1600.0, 1600.0) - 0.25).abs() < 1e-6);
        assert_eq!(
            mask.fraction(5000.0, 0.0, 10.0, 10.0),
            0.0,
            "outside the slide"
        );
    }
}
