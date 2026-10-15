//! Full-resolution tiles of the copy, read from the slide.
//!
//! The copy's full-resolution level is either a stored pyramid level at
//! its native pixels, or a resampled one (a target pixel size between
//! stored levels): then every block of output pixels is filtered with the
//! viewer's separable Lanczos-3 (widened when minifying, so it does not
//! alias) from the next finer stored level. Reads never reach outside the
//! requested region: what an edge tile needs beyond it repeats the
//! region's edge, so nothing outside a region of interest ends up in the
//! file, not even in the padding of edge tiles.

use super::ConvertError;
use super::pyramid::{TileRead, TileSource, pad_edges};
use crate::Error;
use crate::WsiFile;
use crate::blitter::{BlitRect, TileSrc, blit_tile_lanczos3, lanczos_border_for_scale};

/// Output pixels resampled per block (bounds the source window to about
/// 4 MB at 4× minification).
const RESAMPLE_BLOCK: u32 = 256;

/// How the full-resolution level relates to the slide.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Mapping {
    /// Pixels of stored level `level`; `origin` is the region's top-left
    /// in that level's pixels.
    Native {
        level: u32,
        downsample: f64,
        origin: (i64, i64),
    },
    /// Resampled from stored level `level`: output pixel `(x, y)` covers
    /// level pixels `origin + (x, y) * scale` (continuous, top-left
    /// corners); `extent` is the region in level pixels
    /// (`x0, y0, x1, y1`, exclusive end).
    Resampled {
        level: u32,
        downsample: f64,
        scale: (f64, f64),
        origin: (f64, f64),
        extent: (i64, i64, i64, i64),
    },
}

impl Mapping {
    pub(crate) fn level(&self) -> u32 {
        match self {
            Mapping::Native { level, .. } | Mapping::Resampled { level, .. } => *level,
        }
    }
}

/// Reads tiles of the copy's full-resolution level from a slide.
pub(crate) struct SlideTiles {
    pub wsi: WsiFile,
    pub mapping: Mapping,
    pub tile: u32,
    /// Size of the full-resolution level.
    pub width: u32,
    pub height: u32,
    pub background: [u8; 3],
}

/// Attempts at a read that fails with a transient error (a poisoned
/// OpenSlide handle that is being replaced, or a region that failed once
/// and is held back for a few seconds until it can be re-checked); the
/// waits add up to about 12 s.
const READ_ATTEMPTS: u32 = 7;

/// Reads a region as RGBA, retrying transient errors; `None` if the slide
/// data there is damaged.
fn read_rgba(
    wsi: &WsiFile,
    x: i64,
    y: i64,
    level: u32,
    width: u32,
    height: u32,
) -> Result<Option<Vec<u8>>, ConvertError> {
    let mut attempt = 0;
    loop {
        match wsi.read_region(x, y, level, width, height) {
            Ok(rgba) => return Ok(Some(rgba)),
            Err(Error::CorruptRegion { .. }) => return Ok(None),
            Err(err) if err.is_transient() && attempt + 1 < READ_ATTEMPTS => {
                std::thread::sleep(std::time::Duration::from_millis(250 << attempt.min(4)));
                attempt += 1;
            }
            Err(err) => return Err(err.into()),
        }
    }
}

impl SlideTiles {
    fn fill_background(&self, out: &mut [u8]) {
        for pixel in out.as_chunks_mut::<3>().0 {
            *pixel = self.background;
        }
    }

    /// Copies a `width × height` RGBA block into `out` (stride `tile`) at
    /// `(x, y)`.
    fn place(&self, rgba: &[u8], width: usize, height: usize, out: &mut [u8], x: usize, y: usize) {
        let tile = self.tile as usize;
        for row in 0..height {
            let src = &rgba[row * width * 4..][..width * 4];
            let dst = &mut out[((y + row) * tile + x) * 3..][..width * 3];
            for (d, s) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0)
            {
                *d = [s[0], s[1], s[2]];
            }
        }
    }

    /// Reads a block of `width × height` output pixels at output
    /// `(x, y)` into `rgba` by resampling; `false` if the slide data is
    /// damaged.
    #[allow(clippy::too_many_arguments)]
    fn resample_block(
        &self,
        level: u32,
        downsample: f64,
        scale: (f64, f64),
        origin: (f64, f64),
        extent: (i64, i64, i64, i64),
        (x, y, width, height): (u32, u32, u32, u32),
        rgba: &mut Vec<u8>,
    ) -> Result<bool, ConvertError> {
        let (sx, sy) = scale;
        let border = i64::from(lanczos_border_for_scale(sx.max(sy)));
        // The block in continuous level pixels.
        let (u0, u1) = (
            origin.0 + f64::from(x) * sx,
            origin.0 + f64::from(x + width) * sx,
        );
        let (v0, v1) = (
            origin.1 + f64::from(y) * sy,
            origin.1 + f64::from(y + height) * sy,
        );
        // The source window, inner area plus a border for the filter.
        let (inner_x, inner_y) = (u0.floor() as i64, v0.floor() as i64);
        let wx0 = inner_x - border;
        let wy0 = inner_y - border;
        let wx1 = u1.ceil() as i64 + border;
        let wy1 = v1.ceil() as i64 + border;
        let (ww, wh) = ((wx1 - wx0) as usize, (wy1 - wy0) as usize);
        // Only the region is read; the rest of the window repeats its edge.
        let (ex0, ey0, ex1, ey1) = extent;
        let rx0 = wx0.clamp(ex0, ex1 - 1);
        let ry0 = wy0.clamp(ey0, ey1 - 1);
        let rx1 = wx1.clamp(rx0 + 1, ex1);
        let ry1 = wy1.clamp(ry0 + 1, ey1);
        let (rw, rh) = ((rx1 - rx0) as u32, (ry1 - ry0) as u32);
        let read = read_rgba(
            &self.wsi,
            (rx0 as f64 * downsample).round() as i64,
            (ry0 as f64 * downsample).round() as i64,
            level,
            rw,
            rh,
        )?;
        let Some(read) = read else {
            return Ok(false);
        };
        let mut window = vec![0u8; ww * wh * 4];
        for wy in 0..wh {
            let sy_idx = ((wy0 + wy as i64).clamp(ry0, ry1 - 1) - ry0) as usize;
            let src_row = &read[sy_idx * rw as usize * 4..][..rw as usize * 4];
            let dst_row = &mut window[wy * ww * 4..][..ww * 4];
            for wx in 0..ww {
                let sx_idx = ((wx0 + wx as i64).clamp(rx0, rx1 - 1) - rx0) as usize;
                dst_row[wx * 4..wx * 4 + 4].copy_from_slice(&src_row[sx_idx * 4..sx_idx * 4 + 4]);
            }
        }
        let inner_w = (ww - 2 * border as usize) as u32;
        let inner_h = (wh - 2 * border as usize) as u32;
        let exact_x = (inner_x as f64 - u0) / sx;
        let exact_y = (inner_y as f64 - v0) / sy;
        let exact_width = f64::from(inner_w) / sx;
        let exact_height = f64::from(inner_h) / sy;
        let rect_x = exact_x.floor();
        let rect_y = exact_y.floor();
        let rect = BlitRect {
            x: rect_x as i32,
            y: rect_y as i32,
            width: ((exact_x + exact_width).ceil() - rect_x) as i32,
            height: ((exact_y + exact_height).ceil() - rect_y) as i32,
            exact_x,
            exact_y,
            exact_width,
            exact_height,
        };
        rgba.clear();
        rgba.resize(width as usize * height as usize * 4, 0);
        blit_tile_lanczos3(
            rgba,
            width,
            height,
            TileSrc {
                data: &window,
                width: inner_w,
                height: inner_h,
                border: border as u32,
            },
            rect,
        );
        Ok(true)
    }
}

impl TileSource for SlideTiles {
    fn read_tile(&self, col: u32, row: u32, out: &mut [u8]) -> Result<TileRead, ConvertError> {
        let tile = self.tile;
        let (x, y) = (col * tile, row * tile);
        let width = (self.width - x).min(tile);
        let height = (self.height - y).min(tile);
        let mut status = TileRead::Read;
        match &self.mapping {
            Mapping::Native {
                level,
                downsample,
                origin,
            } => {
                let lx = origin.0 + i64::from(x);
                let ly = origin.1 + i64::from(y);
                match read_rgba(
                    &self.wsi,
                    (lx as f64 * downsample).round() as i64,
                    (ly as f64 * downsample).round() as i64,
                    *level,
                    width,
                    height,
                )? {
                    Some(rgba) => {
                        self.place(&rgba, width as usize, height as usize, out, 0, 0);
                    }
                    None => {
                        self.fill_background(out);
                        status = TileRead::Unreadable;
                    }
                }
            }
            Mapping::Resampled {
                level,
                downsample,
                scale,
                origin,
                extent,
            } => {
                let mut rgba = Vec::new();
                'blocks: for by in (0..height).step_by(RESAMPLE_BLOCK as usize) {
                    for bx in (0..width).step_by(RESAMPLE_BLOCK as usize) {
                        let bw = (width - bx).min(RESAMPLE_BLOCK);
                        let bh = (height - by).min(RESAMPLE_BLOCK);
                        let ok = self.resample_block(
                            *level,
                            *downsample,
                            *scale,
                            *origin,
                            *extent,
                            (x + bx, y + by, bw, bh),
                            &mut rgba,
                        )?;
                        if !ok {
                            self.fill_background(out);
                            status = TileRead::Unreadable;
                            break 'blocks;
                        }
                        self.place(
                            &rgba,
                            bw as usize,
                            bh as usize,
                            out,
                            bx as usize,
                            by as usize,
                        );
                    }
                }
            }
        }
        if status == TileRead::Read {
            pad_edges(out, tile as usize, width as usize, height as usize);
        }
        Ok(status)
    }
}
