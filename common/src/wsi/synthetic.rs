//! Synthetic low-resolution levels for slides without a (deep enough)
//! pyramid.
//!
//! A single-level 100k x 100k TIFF has no level a renderer can draw at fit
//! zoom: it would need 150k level-0 tiles. [`WsiFile::with_synthetic_levels`]
//! returns a view of the slide with extra levels appended after the coarsest
//! real one, each [`SYNTHETIC_LEVEL_RATIO`] times coarser, until the longest
//! side is at most [`SYNTHETIC_LEVEL_MAX_SIZE`] pixels. Real level indices
//! are unchanged.
//!
//! Reading a synthetic level box-filters the coarsest real level ("base"):
//!
//! * exactly, streamed in bands, when that needs at most
//!   [`EXACT_SYNTHESIS_BUDGET`] base pixels;
//! * otherwise from a bounded number of point samples (approximate).
//!
//! A [`TileManager`](crate::TileManager) with a tile cache attached builds
//! coarse synthetic tiles exactly from cached tiles of the next finer level
//! instead, so the approximation is only a fallback for direct reads.

use super::{AlphaMode, WsiFile, WsiLevel, pixels};
use crate::{Error, Result};

/// Each synthetic level is this many times coarser than the previous one.
pub const SYNTHETIC_LEVEL_RATIO: u32 = 4;

/// Synthetic levels are added until the longest side is at most this.
pub const SYNTHETIC_LEVEL_MAX_SIZE: u64 = 2048;

/// Most base-level pixels an exact synthetic read decodes (2^28: e.g. a
/// 256-px tile 64x coarser than the base).
pub const EXACT_SYNTHESIS_BUDGET: u64 = 1 << 28;

/// Point samples an approximate synthetic read takes at most.
pub const SPARSE_SYNTHESIS_SAMPLES: u64 = 1 << 16;

/// Base pixels read per band of an exact synthetic read.
const BAND_PIXELS: u64 = 1 << 22;

/// The synthetic levels to append to `levels`.
pub(crate) fn synthetic_levels(levels: &[WsiLevel]) -> Vec<WsiLevel> {
    let Some(base) = levels.last() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut factor = 1u64;
    loop {
        let width = base.width.div_ceil(factor);
        let height = base.height.div_ceil(factor);
        if width.max(height) <= SYNTHETIC_LEVEL_MAX_SIZE || factor > (1 << 40) {
            break;
        }
        factor *= u64::from(SYNTHETIC_LEVEL_RATIO);
        out.push(WsiLevel {
            level: (levels.len() + out.len()) as u32,
            width: base.width.div_ceil(factor).max(1),
            height: base.height.div_ceil(factor).max(1),
            downsample: base.downsample * factor as f64,
            tile_width: None,
            tile_height: None,
            synthetic: true,
        });
    }
    out
}

/// A base-level pixel coordinate in level-0 pixels (what OpenSlide reads
/// take, whatever the level).
#[inline]
fn to_level0(base_px: i64, base_downsample: f64) -> i64 {
    (base_px as f64 * base_downsample).round() as i64
}

/// Packs premultiplied RGBA back into OpenSlide's ARGB for conversion.
#[inline]
fn to_argb(p: [u32; 4]) -> u32 {
    p[3] << 24 | p[0] << 16 | p[1] << 8 | p[2]
}

impl WsiFile {
    /// Index of the coarsest real level, from which synthetic levels are
    /// built.
    fn base_level(&self) -> &WsiLevel {
        let real = &self.shared.properties.levels;
        &real[real.len() - 1]
    }

    /// How many base-level pixels an exact read of `width` x `height`
    /// pixels of synthetic `level` decodes, or `None` for real levels.
    pub fn synthesis_cost(&self, level: u32, width: u32, height: u32) -> Option<u64> {
        let info = self.level(level).filter(|info| info.synthetic)?;
        let factor = (info.downsample / self.base_level().downsample).round() as u64;
        Some((u64::from(width) * factor).saturating_mul(u64::from(height) * factor))
    }

    /// Reads a region of a synthetic level; see the module documentation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_synthetic_into(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
        mode: AlphaMode,
        out: &mut [u8],
    ) -> Result<()> {
        let info = self
            .level(level)
            .filter(|info| info.synthetic)
            .ok_or(Error::InvalidLevel(
                level,
                self.level_count().saturating_sub(1),
            ))?;
        let base = self.base_level();
        let factor = (info.downsample / base.downsample).round().max(1.0) as u64;
        // Origin in base-level pixels.
        let bx = (x as f64 / base.downsample).round() as i64;
        let by = (y as f64 / base.downsample).round() as i64;
        let background = self.shared.properties.background_color;
        let cost = self
            .synthesis_cost(level, width, height)
            .unwrap_or(u64::MAX);
        if cost <= EXACT_SYNTHESIS_BUDGET {
            self.synthesize_exact(
                (base.level, base.downsample),
                (bx, by),
                factor,
                width,
                height,
                mode,
                out,
                background,
            )
        } else {
            self.synthesize_sparse(
                (base.level, base.downsample),
                (bx, by),
                factor,
                width,
                height,
                mode,
                out,
                background,
            )
        }
    }

    /// `(bx, by)` is in base-level pixels; reads take level-0 coordinates.
    #[allow(clippy::too_many_arguments)]
    fn synthesize_exact(
        &self,
        (base_level, base_downsample): (u32, f64),
        (bx, by): (i64, i64),
        factor: u64,
        width: u32,
        height: u32,
        mode: AlphaMode,
        out: &mut [u8],
        background: [u8; 3],
    ) -> Result<()> {
        let w = width as usize;
        let f = factor as usize;
        let row_pixels = u64::from(width) * factor * factor;
        let rows_per_band = (BAND_PIXELS / row_pixels.max(1)).clamp(1, u64::from(height)) as u32;
        let band_w =
            u32::try_from(u64::from(width) * factor).map_err(|_| Error::RegionTooLarge {
                width: u64::from(width) * factor,
                height: u64::from(height) * factor,
                max_pixels: EXACT_SYNTHESIS_BUDGET,
            })?;
        let area = (factor * factor) as u32;
        let mut sums = vec![0u32; w * 4];
        let mut row0 = 0u32;
        while row0 < height {
            let rows = rows_per_band.min(height - row0);
            let band_h = rows * factor as u32;
            let band = self.read_region_with(
                to_level0(bx, base_downsample),
                to_level0(by + i64::from(row0) * factor as i64, base_downsample),
                base_level,
                band_w,
                band_h,
                AlphaMode::Premultiplied,
            )?;
            for r in 0..rows as usize {
                sums.fill(0);
                for sy in 0..f {
                    let line = &band[(r * f + sy) * band_w as usize * 4..][..band_w as usize * 4];
                    for (sx, px) in line.as_chunks::<4>().0.iter().enumerate() {
                        let o = (sx / f) * 4;
                        sums[o] += u32::from(px[0]);
                        sums[o + 1] += u32::from(px[1]);
                        sums[o + 2] += u32::from(px[2]);
                        sums[o + 3] += u32::from(px[3]);
                    }
                }
                let out_row = &mut out[(row0 as usize + r) * w * 4..][..w * 4];
                for (px, sum) in out_row
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(sums.as_chunks::<4>().0)
                {
                    let avg = sum.map(|s| (s + area / 2) / area);
                    *px = pixels::convert_pixel(to_argb(avg), mode, background);
                }
            }
            row0 += rows;
        }
        Ok(())
    }

    /// `(bx, by)` is in base-level pixels; reads take level-0 coordinates.
    #[allow(clippy::too_many_arguments)]
    fn synthesize_sparse(
        &self,
        (base_level, base_downsample): (u32, f64),
        (bx, by): (i64, i64),
        factor: u64,
        width: u32,
        height: u32,
        mode: AlphaMode,
        out: &mut [u8],
        background: [u8; 3],
    ) -> Result<()> {
        let pixels_out = u64::from(width) * u64::from(height);
        let (gw, gh) = if pixels_out <= SPARSE_SYNTHESIS_SAMPLES {
            (width, height)
        } else {
            let scale = (SPARSE_SYNTHESIS_SAMPLES as f64 / pixels_out as f64).sqrt();
            (
                ((f64::from(width) * scale) as u32).max(1),
                ((f64::from(height) * scale) as u32).max(1),
            )
        };
        let span_x = u64::from(width) * factor;
        let span_y = u64::from(height) * factor;
        let mut grid = vec![[0u8; 4]; gw as usize * gh as usize];
        for j in 0..gh {
            for i in 0..gw {
                // Centre of the sample's footprint, in base pixels; average
                // the 2x2 pixels around it.
                let cx = bx + ((f64::from(i) + 0.5) * span_x as f64 / f64::from(gw)) as i64;
                let cy = by + ((f64::from(j) + 0.5) * span_y as f64 / f64::from(gh)) as i64;
                let sample = self
                    .read_region_with(
                        to_level0(cx - 1, base_downsample),
                        to_level0(cy - 1, base_downsample),
                        base_level,
                        2,
                        2,
                        AlphaMode::Premultiplied,
                    )
                    .map(|p| {
                        let mut sum = [0u32; 4];
                        for px in p.as_chunks::<4>().0 {
                            for c in 0..4 {
                                sum[c] += u32::from(px[c]);
                            }
                        }
                        sum.map(|s| (s + 2) / 4)
                    })
                    .unwrap_or([0; 4]);
                grid[(j * gw + i) as usize] =
                    pixels::convert_pixel(to_argb(sample), mode, background);
            }
        }
        // Nearest-neighbour upscale of the sample grid to the output.
        for y in 0..height {
            let gy = (u64::from(y) * u64::from(gh) / u64::from(height)) as usize;
            for x in 0..width {
                let gx = (u64::from(x) * u64::from(gw) / u64::from(width)) as usize;
                let o = (y as usize * width as usize + x as usize) * 4;
                out[o..o + 4].copy_from_slice(&grid[gy * gw as usize + gx]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffSpec, write_tiff};

    fn level(width: u64, height: u64, downsample: f64) -> WsiLevel {
        WsiLevel {
            level: 0,
            width,
            height,
            downsample,
            tile_width: None,
            tile_height: None,
            synthetic: false,
        }
    }

    #[test]
    fn appends_levels_until_the_slide_is_small() {
        let levels = [level(100_000, 80_000, 1.0)];
        let synthetic = synthetic_levels(&levels);
        let sizes: Vec<_> = synthetic.iter().map(|l| (l.width, l.downsample)).collect();
        assert_eq!(sizes, [(25_000, 4.0), (6_250, 16.0), (1_563, 64.0)]);
        assert!(synthetic.iter().all(|l| l.synthetic));
        assert_eq!(synthetic[0].level, 1);
        // Two real levels: synthetic ones continue from the coarser one.
        let levels = [level(100_000, 80_000, 1.0), level(25_000, 20_000, 4.0)];
        let synthetic = synthetic_levels(&levels);
        assert_eq!(synthetic[0].level, 2);
        assert_eq!(synthetic[0].downsample, 16.0);
        // Deep pyramids get none.
        assert!(synthetic_levels(&[level(2048, 1024, 1.0)]).is_empty());
        assert!(synthetic_levels(&[]).is_empty());
    }

    #[test]
    fn synthetic_regions_average_the_base_level() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat.tiff");
        // 4096 x 64: alternating 2-px columns of 0 and 200, so every 4x4
        // block averages to 100.
        write_tiff(&path, &TiffSpec::single(4096, 64, 64), |x, _| {
            if (x / 2) % 2 == 0 {
                [0, 0, 0]
            } else {
                [200, 100, 50]
            }
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap().with_synthetic_levels();
        assert!(wsi.has_synthetic_levels());
        assert_eq!(wsi.real_level_count(), 1);
        assert_eq!(wsi.level_count(), 2);
        let level = wsi.level(1).unwrap();
        assert!(level.synthetic);
        assert_eq!(
            (level.width, level.height, level.downsample),
            (1024, 16, 4.0)
        );
        assert_eq!(wsi.synthesis_cost(1, 10, 10), Some(1600));
        assert_eq!(wsi.synthesis_cost(0, 10, 10), None);

        let pixels = wsi.read_region(0, 0, 1, 1024, 16).unwrap();
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [100, 50, 25, 255])
        );
        // Areas past the slide edge are transparent at the base, so edge
        // blocks blend with the background (white).
        let edge = wsi.read_region(4096 - 8, 0, 1, 4, 1).unwrap();
        assert_eq!(&edge[..4], [100, 50, 25, 255]);
        assert_eq!(&edge[8..12], [255, 255, 255, 255]);
        let straight = wsi
            .read_region_with(4096 - 8, 0, 1, 4, 1, AlphaMode::Straight)
            .unwrap();
        assert_eq!(straight[11], 0);
    }

    #[test]
    fn oversized_synthetic_reads_fall_back_to_sampling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat.tiff");
        write_tiff(&path, &TiffSpec::single(4096, 64, 64), |x, _| {
            if x < 2048 { [255, 0, 0] } else { [0, 0, 255] }
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap().with_synthetic_levels();
        let mut out = vec![0u8; 1024 * 16 * 4];
        wsi.synthesize_sparse(
            (0, 1.0),
            (0, 0),
            4,
            1024,
            16,
            AlphaMode::Composited,
            &mut out,
            [255; 3],
        )
        .unwrap();
        assert_eq!(&out[..4], [255, 0, 0, 255]);
        assert_eq!(&out[out.len() - 4..], [0, 0, 255, 255]);
    }

    #[test]
    fn synthetic_levels_of_pyramids_read_the_right_place() {
        // Two real levels (downsample 1 and 2): the synthetic level is built
        // from level 1, whose pixel coordinates differ from level 0's. (With
        // base-level coordinates passed as level-0 ones, x = 12000 read the
        // red half.)
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pyramid.tiff");
        write_tiff(&path, &TiffSpec::pyramid(16384, 256, 64, 2), |x, _| {
            if x < 8192 { [255, 0, 0] } else { [0, 0, 255] }
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap().with_synthetic_levels();
        assert_eq!(wsi.real_level_count(), 2);
        let synthetic = wsi.level(2).unwrap();
        assert!(synthetic.synthetic);
        assert_eq!(synthetic.downsample, 8.0);

        // Exact path: level-0 x = 12000 is in the blue half.
        let exact = wsi.read_region(12_000, 0, 2, 4, 1).unwrap();
        assert!(
            exact
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [0, 0, 255, 255])
        );
        let exact = wsi.read_region(1_000, 0, 2, 4, 1).unwrap();
        assert!(
            exact
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [255, 0, 0, 255])
        );

        // Sampling path: base-level x = 6000 is level-0 x = 12000.
        let mut out = vec![0u8; 4 * 4];
        wsi.synthesize_sparse(
            (1, 2.0),
            (6000, 0),
            4,
            4,
            1,
            AlphaMode::Composited,
            &mut out,
            [255; 3],
        )
        .unwrap();
        assert!(
            out.as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [0, 0, 255, 255])
        );
    }
}
