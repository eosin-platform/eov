//! Tissue quality control for brightfield slides (AUDIT §14.5): pen marks,
//! possible tissue folds and out-of-focus areas, judged cell by cell so
//! they can be shown over the slide, summarised per slide, and left out of
//! datasets.
//!
//! The slide is cut into square cells of [`QcOptions::cell_um`] (or
//! [`CELL_PX_WITHOUT_MPP`] level-0 pixels when the slide has no pixel size).
//! Two passes:
//!
//! 1. **A low-resolution pass** over a level of at most [`OVERVIEW_PIXELS`]
//!    pixels classifies each pixel as glass, tissue (saturation, as
//!    [`crate::tissue`] does, with Otsu's threshold) or ink: green and
//!    blue-to-cyan marker colours, which H&E and IHC stains do not produce,
//!    and black. A cell is tissue when at least [`TISSUE_FRACTION`] of it is
//!    tissue, and marked when at least [`PEN_FRACTION`] of it is ink. A tissue
//!    cell is a *possible fold* when its darkness (the optical density of
//!    each pixel's brightest channel, so bright red blood does not count) is
//!    at least [`FOLD_RATIO`] times the median of the slide's tissue cells
//!    and at least [`FOLD_MIN_DARKNESS`]: folded tissue is twice as thick.
//! 2. **A sharpness pass** reads the cells that are mostly tissue (at least
//!    [`SHARPNESS_TISSUE_FRACTION`], so borders with glass are not judged) at
//!    about [`QcOptions::sharpness_mpp`] and measures the variance of the
//!    Laplacian of the grey image relative to its variance: blur takes away
//!    fine detail but not overall contrast, so the ratio is low where the
//!    scan is out of focus whatever the tissue's contrast. A cell is
//!    *possibly out of focus* when its ratio is below [`BLUR_RATIO`] of the
//!    slide's median. Tissue without fine detail (fibrin, mucus, necrosis)
//!    can be marked too.
//!
//! The thresholds are relative to the slide, so they adapt to stains and
//! scanners, and they are heuristics: QC flags are for review, not
//! measurements. Pen detection is skipped for stains that are green or
//! blue by design (trichrome, light green, alcian blue).

use crate::WsiFile;
use crate::tissue::{is_tissue_pixel, otsu_threshold, saturation};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest level read for the low-resolution pass, in pixels.
pub const OVERVIEW_PIXELS: u64 = 16 * 1024 * 1024;
/// Cell edge when the slide has no pixel size, in level-0 pixels.
pub const CELL_PX_WITHOUT_MPP: f64 = 512.0;
/// Share of a cell that must be tissue for it to count as tissue.
pub const TISSUE_FRACTION: f32 = 0.25;
/// Share of a cell that must be ink for it to be marked as pen.
pub const PEN_FRACTION: f32 = 0.15;
/// Darkness, relative to the slide's median tissue cell, from which a cell
/// is a possible fold.
pub const FOLD_RATIO: f32 = 2.5;
/// Darkness (optical density of the brightest channel) a possible fold
/// reaches at least.
pub const FOLD_MIN_DARKNESS: f32 = 0.25;
/// Share of tissue a cell needs for its sharpness to be judged.
pub const SHARPNESS_TISSUE_FRACTION: f32 = 0.8;
/// Sharpness, relative to the slide's median tissue cell, below which a
/// cell is out of focus.
pub const BLUR_RATIO: f32 = 0.35;

/// A cell holds tissue.
pub const TISSUE: u8 = 1;
/// A cell is covered by pen ink.
pub const PEN: u8 = 2;
/// A cell's tissue is much denser than the slide's (likely folded).
pub const FOLD: u8 = 4;
/// A cell's tissue is out of focus.
pub const BLUR: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QcOptions {
    /// Cell edge in micrometres.
    pub cell_um: f64,
    /// Pixel size the sharpness is measured at, in micrometres.
    pub sharpness_mpp: f64,
    /// Measure sharpness (the slower pass).
    pub sharpness: bool,
}

impl Default for QcOptions {
    fn default() -> Self {
        Self {
            cell_um: 128.0,
            sharpness_mpp: 2.0,
            sharpness: true,
        }
    }
}

/// Per-slide shares of the tissue that is marked, 0–1.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct QcSummary {
    pub cells: u32,
    pub tissue_cells: u32,
    /// Tissue area in mm², when the slide has a pixel size.
    pub tissue_mm2: Option<f64>,
    pub pen: f32,
    pub fold: f32,
    /// `None` when sharpness was not measured.
    pub blur: Option<f32>,
    /// Pen detection was skipped (the stain is green or blue by design).
    pub pen_skipped: bool,
}

/// The cells of a slide and what was found in each.
#[derive(Debug, Clone, PartialEq)]
pub struct QcResult {
    pub columns: u32,
    pub rows: u32,
    /// Cell size in level-0 pixels.
    pub cell_width: f64,
    pub cell_height: f64,
    /// [`TISSUE`], [`PEN`], [`FOLD`], [`BLUR`] bits per cell, row by row.
    pub flags: Vec<u8>,
    /// Sharpness ratio per cell (NaN where not measured).
    pub sharpness: Vec<f32>,
    /// Share of each cell that is tissue, 0–1.
    pub tissue_share: Vec<f32>,
    pub summary: QcSummary,
}

impl QcResult {
    /// The marks of the cell holding level-0 point (`x`, `y`).
    pub fn flags_at(&self, x: f64, y: f64) -> u8 {
        if x < 0.0 || y < 0.0 {
            return 0;
        }
        let (column, row) = ((x / self.cell_width) as u32, (y / self.cell_height) as u32);
        if column >= self.columns || row >= self.rows {
            return 0;
        }
        self.flags[(row * self.columns + column) as usize]
    }

    /// Whether any cell overlapping a level-0 rectangle carries one of
    /// `marks` (e.g. `PEN | FOLD`).
    pub fn rect_marked(&self, x: f64, y: f64, width: f64, height: f64, marks: u8) -> bool {
        let first_column = (x / self.cell_width).floor().max(0.0) as u32;
        let first_row = (y / self.cell_height).floor().max(0.0) as u32;
        let last_column = (((x + width) / self.cell_width).ceil() as u32).min(self.columns);
        let last_row = (((y + height) / self.cell_height).ceil() as u32).min(self.rows);
        (first_row..last_row).any(|row| {
            (first_column..last_column)
                .any(|column| self.flags[(row * self.columns + column) as usize] & marks != 0)
        })
    }
}

/// Stains whose own colours look like marker ink.
pub fn pen_detection_applies(stain: Option<&str>) -> bool {
    let Some(stain) = stain else {
        return true;
    };
    let stain = stain.to_ascii_lowercase();
    ![
        "trichrome",
        "light green",
        "alcian",
        "giemsa",
        "elastic",
        "fast green",
    ]
    .iter()
    .any(|name| stain.contains(name))
}

/// Hue (degrees), saturation and value (0–1) of a pixel.
fn hsv(rgb: [u8; 3]) -> (f32, f32, f32) {
    let [r, g, b] = rgb.map(|c| f32::from(c) / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let saturation = if max > 0.0 { delta / max } else { 0.0 };
    let hue = if delta <= f32::EPSILON {
        0.0
    } else if max == r {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    (hue, saturation, max)
}

/// Whether a pixel has a marker-ink colour: green, blue to cyan, or black.
pub fn is_ink(rgb: [u8; 3]) -> bool {
    let (hue, saturation, value) = hsv(rgb);
    let green = (75.0..=170.0).contains(&hue) && saturation > 0.25 && value > 0.12;
    let blue = (170.0..230.0).contains(&hue) && saturation > 0.35 && value > 0.12;
    let black = value < 0.16;
    green || blue || black
}

/// Darkness: the optical density of the brightest channel (glass and
/// bright red blood are near 0, dense purple tissue is not).
fn darkness(rgb: [u8; 3]) -> f32 {
    let brightest = rgb[0].max(rgb[1]).max(rgb[2]).max(1);
    -(f32::from(brightest) / 255.0).log10()
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

/// Per-cell counts of the low-resolution pass.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    pixels: u32,
    tissue: u32,
    ink: u32,
    darkness_sum: f32,
}

/// The low-resolution pass on an RGBA overview covering the whole slide
/// (`rgba` is `width` × `height`, each pixel `scale_x` × `scale_y` level-0
/// pixels). Returns the flags and the pass's cell geometry.
pub fn classify_overview(
    rgba: &[u8],
    width: u32,
    height: u32,
    scale: (f64, f64),
    cell: (f64, f64),
    slide: (u64, u64),
    pen: bool,
) -> QcResult {
    let columns = ((slide.0 as f64 / cell.0).ceil() as u32).max(1);
    let rows = ((slide.1 as f64 / cell.1).ceil() as u32).max(1);
    let pixels = rgba.as_chunks::<4>().0;
    // Otsu's threshold on the saturation of opaque, non-ink pixels.
    let mut histogram = [0u64; 256];
    for p in pixels {
        let rgb = [p[0], p[1], p[2]];
        if p[3] == 255 && !(pen && is_ink(rgb)) {
            histogram[saturation(rgb) as usize] += 1;
        }
    }
    let threshold = otsu_threshold(&histogram);
    let mut tallies = vec![Tally::default(); (columns * rows) as usize];
    for y in 0..height {
        let row = (((f64::from(y) + 0.5) * scale.1 / cell.1) as u32).min(rows - 1);
        for x in 0..width {
            let column = (((f64::from(x) + 0.5) * scale.0 / cell.0) as u32).min(columns - 1);
            let p = &pixels[(y * width + x) as usize];
            let tally = &mut tallies[(row * columns + column) as usize];
            tally.pixels += 1;
            if p[3] != 255 {
                continue;
            }
            let rgb = [p[0], p[1], p[2]];
            if pen && is_ink(rgb) {
                tally.ink += 1;
            } else if is_tissue_pixel(rgb, threshold) {
                tally.tissue += 1;
                tally.darkness_sum += darkness(rgb);
            }
        }
    }
    let mut flags = vec![0u8; tallies.len()];
    let mut tissue_share = vec![0.0f32; tallies.len()];
    let mut darknesses = Vec::new();
    for ((flag, share), tally) in flags.iter_mut().zip(&mut tissue_share).zip(&tallies) {
        if tally.pixels == 0 {
            continue;
        }
        let pixels = tally.pixels as f32;
        *share = tally.tissue as f32 / pixels;
        if tally.ink as f32 >= PEN_FRACTION * pixels {
            *flag |= PEN;
        }
        if *share >= TISSUE_FRACTION {
            *flag |= TISSUE;
            darknesses.push(tally.darkness_sum / tally.tissue as f32);
        }
    }
    if let Some(typical) = median(&mut darknesses) {
        let threshold = (FOLD_RATIO * typical).max(FOLD_MIN_DARKNESS);
        for (flag, tally) in flags.iter_mut().zip(&tallies) {
            if *flag & TISSUE != 0 && tally.darkness_sum / tally.tissue as f32 >= threshold {
                *flag |= FOLD;
            }
        }
    }
    QcResult {
        columns,
        rows,
        cell_width: cell.0,
        cell_height: cell.1,
        sharpness: vec![f32::NAN; flags.len()],
        tissue_share,
        flags,
        summary: QcSummary {
            pen_skipped: !pen,
            ..QcSummary::default()
        },
    }
}

/// Fine detail relative to contrast: the variance of the 4-neighbour
/// Laplacian of the grey image divided by the variance of the image.
/// `None` for flat or tiny images.
pub fn sharpness(grey: &[f32], width: usize, height: usize) -> Option<f32> {
    if width < 3 || height < 3 || grey.len() < width * height {
        return None;
    }
    let n = (width * height) as f64;
    let mean = grey.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let variance = grey
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    if variance < 1e-6 {
        return None;
    }
    let (mut sum, mut sum_sq, mut count) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            let laplacian = f64::from(
                grey[i - 1] + grey[i + 1] + grey[i - width] + grey[i + width] - 4.0 * grey[i],
            );
            sum += laplacian;
            sum_sq += laplacian * laplacian;
            count += 1.0;
        }
    }
    let laplacian_variance = sum_sq / count - (sum / count).powi(2);
    Some((laplacian_variance / variance) as f32)
}

/// Marks the tissue cells whose sharpness is below [`BLUR_RATIO`] of the
/// slide's median, given each cell's sharpness.
pub fn mark_blur(result: &mut QcResult) {
    let mut measured: Vec<f32> = result
        .flags
        .iter()
        .zip(&result.sharpness)
        .filter(|(flag, s)| **flag & TISSUE != 0 && **flag & PEN == 0 && s.is_finite())
        .map(|(_, s)| *s)
        .collect();
    let Some(typical) = median(&mut measured) else {
        return;
    };
    for (flag, s) in result.flags.iter_mut().zip(&result.sharpness) {
        if *flag & TISSUE != 0 && s.is_finite() && *s < BLUR_RATIO * typical {
            *flag |= BLUR;
        }
    }
}

fn summarise(result: &mut QcResult, mpp: Option<(f64, f64)>, sharpness: bool) {
    let tissue_cells = result.flags.iter().filter(|f| **f & TISSUE != 0).count() as u32;
    let share = |bit: u8| {
        if tissue_cells == 0 {
            0.0
        } else {
            result
                .flags
                .iter()
                .filter(|f| **f & TISSUE != 0 && **f & bit != 0)
                .count() as f32
                / tissue_cells as f32
        }
    };
    result.summary.cells = result.flags.len() as u32;
    result.summary.tissue_cells = tissue_cells;
    result.summary.tissue_mm2 = mpp.map(|(x, y)| {
        f64::from(tissue_cells) * result.cell_width * x * result.cell_height * y / 1e6
    });
    // Ink may lie on glass: its share is of tissue plus marked cells.
    let pen_cells = result.flags.iter().filter(|f| **f & PEN != 0).count() as f32;
    let judged = result
        .flags
        .iter()
        .filter(|f| **f & (TISSUE | PEN) != 0)
        .count() as f32;
    result.summary.pen = if judged > 0.0 {
        pen_cells / judged
    } else {
        0.0
    };
    result.summary.fold = share(FOLD);
    result.summary.blur = sharpness.then(|| share(BLUR));
}

/// Runs both passes on a slide (worker thread). `progress` gets 0–1;
/// returns `Ok(None)` when cancelled.
pub fn run(
    wsi: &WsiFile,
    options: &QcOptions,
    progress: &dyn Fn(f32),
    cancel: &AtomicBool,
) -> crate::Result<Option<QcResult>> {
    if wsi.is_multichannel() {
        return Err(crate::Error::Unsupported(
            "tissue QC is for brightfield slides".to_string(),
        ));
    }
    let wsi = wsi.without_synthetic_levels();
    let props = wsi.properties();
    let mpp = match (props.mpp_x, props.mpp_y) {
        (Some(x), Some(y)) if x > 0.0 && y > 0.0 => Some((x, y)),
        (Some(v), None) | (None, Some(v)) if v > 0.0 => Some((v, v)),
        _ => None,
    };
    let cell = match mpp {
        Some((x, y)) => (options.cell_um / x, options.cell_um / y),
        None => (CELL_PX_WITHOUT_MPP, CELL_PX_WITHOUT_MPP),
    };
    let levels: Vec<(u32, crate::WsiLevel)> = (0..wsi.level_count())
        .filter_map(|l| wsi.level(l).map(|info| (l, info.clone())))
        .collect();
    let Some((level, overview)) = levels
        .iter()
        .find(|(_, info)| info.width.saturating_mul(info.height) <= OVERVIEW_PIXELS)
        .or_else(|| levels.last())
        .cloned()
    else {
        return Err(crate::Error::Unsupported(
            "the slide has no levels".to_string(),
        ));
    };
    let (w, h) = (overview.width as u32, overview.height as u32);
    if u64::from(w) * u64::from(h) > OVERVIEW_PIXELS * 4 {
        return Err(crate::Error::Unsupported(
            "the slide has no low-resolution level for tissue QC".to_string(),
        ));
    }
    let rgba = read_level(&wsi, level, overview.downsample, w, h, cancel)?;
    let Some(rgba) = rgba else {
        return Ok(None);
    };
    progress(if options.sharpness { 0.1 } else { 0.9 });
    let pen = pen_detection_applies(props.stain.as_deref());
    let mut result = classify_overview(
        &rgba,
        w,
        h,
        (
            props.width as f64 / f64::from(w),
            props.height as f64 / f64::from(h),
        ),
        cell,
        (props.width, props.height),
        pen,
    );
    if options.sharpness
        && !measure_sharpness(&wsi, &mut result, &levels, mpp, options, progress, cancel)?
    {
        return Ok(None);
    }
    if options.sharpness {
        mark_blur(&mut result);
    }
    summarise(&mut result, mpp, options.sharpness);
    progress(1.0);
    Ok(Some(result))
}

fn read_level(
    wsi: &WsiFile,
    level: u32,
    downsample: f64,
    width: u32,
    height: u32,
    cancel: &AtomicBool,
) -> crate::Result<Option<Vec<u8>>> {
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    const CHUNK: u32 = 1024;
    for y0 in (0..height).step_by(CHUNK as usize) {
        for x0 in (0..width).step_by(CHUNK as usize) {
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let cw = CHUNK.min(width - x0);
            let ch = CHUNK.min(height - y0);
            let data = wsi.read_region(
                (f64::from(x0) * downsample).round() as i64,
                (f64::from(y0) * downsample).round() as i64,
                level,
                cw,
                ch,
            )?;
            for row in 0..ch as usize {
                let src = &data[row * cw as usize * 4..(row + 1) * cw as usize * 4];
                let dst = ((y0 as usize + row) * width as usize + x0 as usize) * 4;
                rgba[dst..dst + src.len()].copy_from_slice(src);
            }
        }
    }
    Ok(Some(rgba))
}

/// The sharpness pass: reads blocks of tissue cells at about
/// `sharpness_mpp` (the coarsest level at least that fine) and fills
/// `result.sharpness`. Returns `false` when cancelled.
fn measure_sharpness(
    wsi: &WsiFile,
    result: &mut QcResult,
    levels: &[(u32, crate::WsiLevel)],
    mpp: Option<(f64, f64)>,
    options: &QcOptions,
    progress: &dyn Fn(f32),
    cancel: &AtomicBool,
) -> crate::Result<bool> {
    // Without a pixel size, aim for 4 level-0 pixels per sample.
    let target = match mpp {
        Some((x, y)) => options.sharpness_mpp / x.max(y),
        None => 4.0,
    };
    let (level, downsample) = levels
        .iter()
        .filter(|(_, info)| info.downsample <= target * 1.01)
        .map(|(level, info)| (*level, info.downsample))
        .next_back()
        .unwrap_or((0, 1.0));
    // Blocks of cells read together, at most about 2048 px on a side.
    let cell_px = (result.cell_width / downsample).max(1.0);
    let per_block = ((2048.0 / cell_px).floor() as u32).max(1);
    let block_rows = result.rows.div_ceil(per_block);
    let block_columns = result.columns.div_ceil(per_block);
    let total = (block_rows * block_columns).max(1) as f32;
    let mut done = 0.0;
    for block_row in 0..block_rows {
        for block_column in 0..block_columns {
            done += 1.0;
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let rows = block_row * per_block..((block_row + 1) * per_block).min(result.rows);
            let columns =
                block_column * per_block..((block_column + 1) * per_block).min(result.columns);
            let judged = |index: usize| {
                result.flags[index] & (TISSUE | PEN) == TISSUE
                    && result.tissue_share[index] >= SHARPNESS_TISSUE_FRACTION
            };
            let wanted = rows.clone().any(|row| {
                columns
                    .clone()
                    .any(|column| judged((row * result.columns + column) as usize))
            });
            if !wanted {
                continue;
            }
            let x0 = f64::from(columns.start) * result.cell_width;
            let y0 = f64::from(rows.start) * result.cell_height;
            let x1 =
                (f64::from(columns.end) * result.cell_width).min(wsi.properties().width as f64);
            let y1 = (f64::from(rows.end) * result.cell_height).min(wsi.properties().height as f64);
            let width = ((x1 - x0) / downsample).floor().max(1.0) as u32;
            let height = ((y1 - y0) / downsample).floor().max(1.0) as u32;
            let rgba =
                wsi.read_region(x0.round() as i64, y0.round() as i64, level, width, height)?;
            let grey: Vec<f32> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])
                })
                .collect();
            for row in rows.clone() {
                for column in columns.clone() {
                    let index = (row * result.columns + column) as usize;
                    if !judged(index) {
                        continue;
                    }
                    // The cell's pixels in the block.
                    let cx0 = ((f64::from(column) * result.cell_width - x0) / downsample) as usize;
                    let cy0 = ((f64::from(row) * result.cell_height - y0) / downsample) as usize;
                    let cx1 = (((f64::from(column + 1) * result.cell_width - x0) / downsample)
                        as usize)
                        .min(width as usize);
                    let cy1 = (((f64::from(row + 1) * result.cell_height - y0) / downsample)
                        as usize)
                        .min(height as usize);
                    if cx1 <= cx0 + 2 || cy1 <= cy0 + 2 {
                        continue;
                    }
                    let cell: Vec<f32> = (cy0..cy1)
                        .flat_map(|y| {
                            grey[y * width as usize + cx0..y * width as usize + cx1]
                                .iter()
                                .copied()
                        })
                        .collect();
                    if let Some(value) = sharpness(&cell, cx1 - cx0, cy1 - cy0) {
                        result.sharpness[index] = value;
                    }
                }
            }
            progress(0.1 + 0.9 * done / total);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLASS: [u8; 4] = [244, 244, 242, 255];
    const TISSUE_PINK: [u8; 4] = [214, 140, 190, 255];
    const GREEN_INK: [u8; 4] = [40, 150, 70, 255];
    const BLUE_INK: [u8; 4] = [30, 120, 220, 255];

    #[test]
    fn ink_colours_are_told_from_stains() {
        assert!(is_ink([40, 150, 70]), "green marker");
        assert!(is_ink([30, 120, 220]), "blue marker");
        assert!(is_ink([20, 20, 25]), "black marker");
        assert!(!is_ink([214, 140, 190]), "eosin pink");
        assert!(!is_ink([110, 80, 160]), "haematoxylin purple");
        assert!(!is_ink([150, 100, 60]), "DAB brown");
        assert!(!is_ink([244, 244, 242]), "glass");
        assert!(pen_detection_applies(Some("H&E")));
        assert!(!pen_detection_applies(Some("Masson Trichrome")));
    }

    /// A 40 × 20 overview of a 4000 × 2000 slide: the left half tissue,
    /// a band of green ink, a dense (folded) patch, glass on the right.
    fn overview() -> Vec<u8> {
        let mut rgba = Vec::new();
        for y in 0..20 {
            for x in 0..40 {
                let pixel = if x >= 30 {
                    GLASS
                } else if (10..14).contains(&x) {
                    GREEN_INK
                } else if x < 5 && y < 5 {
                    [120, 40, 100, 255] // dense, twice the density of the pink
                } else if y >= 15 && (20..25).contains(&x) {
                    BLUE_INK
                } else if y < 5 && (25..30).contains(&x) {
                    [200, 30, 40, 255] // blood
                } else {
                    TISSUE_PINK
                };
                rgba.extend_from_slice(&pixel);
            }
        }
        rgba
    }

    #[test]
    fn the_overview_marks_tissue_ink_and_dense_tissue() {
        let result = classify_overview(
            &overview(),
            40,
            20,
            (100.0, 100.0),
            (500.0, 500.0),
            (4000, 2000),
            true,
        );
        assert_eq!((result.columns, result.rows), (8, 4));
        let at = |x: f64, y: f64| result.flags_at(x, y);
        assert_eq!(at(3500.0, 500.0), 0, "glass");
        assert_eq!(at(700.0, 1200.0) & TISSUE, TISSUE);
        assert_ne!(at(1200.0, 500.0) & PEN, 0, "green ink");
        assert_ne!(at(2200.0, 1700.0) & PEN, 0, "blue ink");
        assert_ne!(at(200.0, 200.0) & FOLD, 0, "the dense patch");
        assert_eq!(
            at(2700.0, 200.0) & FOLD,
            0,
            "bright red blood is dense, not dark"
        );
        assert_eq!(at(700.0, 1200.0) & FOLD, 0, "ordinary tissue");
        assert!(result.rect_marked(1000.0, 0.0, 400.0, 400.0, PEN));
        assert!(!result.rect_marked(2500.0, 0.0, 400.0, 400.0, PEN | FOLD));

        let without_pen = classify_overview(
            &overview(),
            40,
            20,
            (100.0, 100.0),
            (500.0, 500.0),
            (4000, 2000),
            false,
        );
        assert!(without_pen.flags.iter().all(|f| f & PEN == 0));
        assert!(without_pen.summary.pen_skipped);
    }

    #[test]
    fn sharpness_drops_with_blur_but_not_with_contrast() {
        let (w, h) = (64, 64);
        // Detail at every scale (noise), as tissue has.
        let pattern = |contrast: f32| -> Vec<f32> {
            let mut seed = 12345u32;
            (0..w * h)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    128.0 + contrast * (((seed >> 16) & 0xff) as f32 / 255.0 - 0.5)
                })
                .collect()
        };
        let blur = |image: &[f32]| -> Vec<f32> {
            let mut out = image.to_vec();
            for _ in 0..3 {
                let copy = out.clone();
                for y in 1..h - 1 {
                    for x in 1..w - 1 {
                        let i = y * w + x;
                        out[i] =
                            (copy[i] * 4.0 + copy[i - 1] + copy[i + 1] + copy[i - w] + copy[i + w])
                                / 8.0;
                    }
                }
            }
            out
        };
        let sharp = sharpness(&pattern(40.0), w, h).unwrap();
        let faint = sharpness(&pattern(10.0), w, h).unwrap();
        let blurred = sharpness(&blur(&pattern(40.0)), w, h).unwrap();
        assert!(
            (sharp - faint).abs() / sharp < 0.01,
            "independent of contrast"
        );
        assert!(blurred < 0.35 * sharp, "{blurred} vs {sharp}");
        assert_eq!(sharpness(&vec![7.0; w * h], w, h), None, "flat");
    }

    #[test]
    fn blur_is_relative_to_the_slide_and_summarised() {
        let mut result = QcResult {
            columns: 4,
            rows: 1,
            cell_width: 100.0,
            cell_height: 100.0,
            flags: vec![TISSUE, TISSUE, TISSUE, PEN],
            sharpness: vec![1.0, 1.1, 0.2, f32::NAN],
            tissue_share: vec![1.0; 4],
            summary: QcSummary::default(),
        };
        mark_blur(&mut result);
        assert_eq!(result.flags[2] & BLUR, BLUR);
        assert_eq!(result.flags[0] & BLUR, 0);
        summarise(&mut result, Some((0.5, 0.5)), true);
        assert_eq!(result.summary.tissue_cells, 3);
        assert_eq!(result.summary.blur, Some(1.0 / 3.0));
        assert_eq!(result.summary.pen, 0.25, "one marked cell of four judged");
        assert!((result.summary.tissue_mm2.unwrap() - 3.0 * 2500.0 / 1e6).abs() < 1e-12);
    }

    #[test]
    fn whole_slides_are_checked_with_both_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qc.tiff");
        // 2048 × 1024: tissue with fine detail on the left, a blurred
        // (smooth) tissue band in the middle, green ink, glass on the right.
        crate::test_support::write_tiled_tiff(&path, 2048, 1024, 256, |x, y| {
            if x >= 1536 {
                [244, 244, 242]
            } else if (1024..1280).contains(&x) {
                [40, 150, 70]
            } else if x >= 512 {
                // Smooth gradient: tissue colour, little fine detail.
                let t = ((x as f32 - 512.0) / 512.0 * 20.0) as u8;
                [200 - t, 130, 180]
            } else if (x / 2 + y / 2) % 2 == 0 {
                [190, 110, 170]
            } else {
                [230, 160, 205]
            }
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let cancel = AtomicBool::new(false);
        let options = QcOptions {
            sharpness_mpp: 1.0,
            ..QcOptions::default()
        };
        let result = run(&wsi, &options, &|_| {}, &cancel).unwrap().unwrap();
        // No pixel size: 512 px cells.
        assert_eq!((result.columns, result.rows), (4, 2));
        assert_ne!(result.flags_at(100.0, 100.0) & TISSUE, 0);
        assert_eq!(
            result.flags_at(100.0, 100.0) & BLUR,
            0,
            "the detailed tissue is sharp"
        );
        assert_ne!(
            result.flags_at(700.0, 100.0) & BLUR,
            0,
            "the smooth band is not"
        );
        assert_ne!(result.flags_at(1100.0, 100.0) & PEN, 0);
        assert_eq!(result.flags_at(1800.0, 100.0), 0);
        assert!(result.summary.blur.is_some_and(|b| b > 0.0));

        cancel.store(true, Ordering::Relaxed);
        assert!(run(&wsi, &options, &|_| {}, &cancel).unwrap().is_none());
    }
}
