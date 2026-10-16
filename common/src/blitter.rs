//! Low-level tile blitting functions for CPU rendering
//!
//! This module contains optimized functions for copying and scaling tile data
//! to an RGBA output buffer. All functions operate on raw `[u8]` RGBA buffers
//! and have no GUI framework dependency.
//!
//! # Sampling convention
//!
//! All samplers use **pixel centres**, like the GPU path (`uv * size - 0.5`)
//! and the overlay geometry: destination pixel `d` covers `[d, d + 1)` and
//! is sampled at the source position `(d + 0.5 - origin) * scale - 0.5` (in
//! source pixel indices), where `origin` is the tile's exact screen-space
//! left edge and `scale` the number of source pixels per screen pixel.
//! Fixed-point arithmetic rounds instead of truncating, so CPU output
//! matches the GPU's within one grey level. Nearest-neighbour sampling
//! ([`blit_tile_nearest`], the pixel view above 1:1) takes the source pixel
//! whose area holds that position, index `floor(position + 0.5)`, as a GPU
//! sampler with nearest filtering does.
//!
//! Tile borders (see [`TileData::border`](crate::TileData::border)) are
//! used for every tap that falls outside the tile's inner area; taps beyond
//! the border repeat the outermost border pixel. One border pixel makes
//! bilinear and trilinear sampling seamless; Lanczos needs
//! [`lanczos_border_for_scale`] pixels.

use rayon::prelude::*;

mod turned;
pub use turned::{
    FrameMap, blit_tile_lanczos3_turned, blit_tile_nearest_turned, blit_tile_trilinear_turned,
    blit_tile_turned, reproject_frame_mapped,
};

#[derive(Clone, Copy)]
struct LinearColSample {
    x0_4: usize,
    x1_4: usize,
    w0: u32,
    w1: u32,
}

#[derive(Clone, Copy)]
struct LinearRowSample {
    row0: usize,
    row1: usize,
    w0: u32,
    w1: u32,
}

/// Borrowed reference to tile source data for blitting.
pub struct TileSrc<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub border: u32,
}

#[derive(Clone, Copy)]
pub struct BlitRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    /// Exact floating-point origin and size of the tile in screen pixels.
    /// When `exact_width > 0`, the bilinear/lanczos samplers derive source
    /// UV mapping from the exact coordinates instead of `x`/`y`/`width`/
    /// `height`, keeping UV coordinates stable as the integer rect
    /// oscillates ±1 px during smooth zoom.
    pub exact_x: f64,
    pub exact_y: f64,
    pub exact_width: f64,
    pub exact_height: f64,
}

#[derive(Clone, Copy)]
struct AxisMapping {
    start: u32,
    end: u32,
    dest_origin: i32,
    exact_origin: f64,
    exact_size: f64,
    scaled_size: i32,
}

impl AxisMapping {
    /// Screen-space origin and size of the tile along this axis.
    #[inline(always)]
    fn origin_and_size(self) -> (f64, f64) {
        if self.exact_size > 0.0 {
            (self.exact_origin, self.exact_size)
        } else {
            (
                f64::from(self.dest_origin),
                f64::from(self.scaled_size.max(1)),
            )
        }
    }

    /// Source pixels per destination pixel.
    #[inline(always)]
    fn scale(self, src_range: f64) -> f64 {
        src_range / self.origin_and_size().1
    }

    /// Continuous source *index* sampled by destination pixel `dest`
    /// (pixel-centre convention), relative to the tile's inner origin.
    #[inline(always)]
    fn src_index(self, dest: u32, src_offset: f64, src_range: f64) -> f64 {
        let (origin, size) = self.origin_and_size();
        src_offset + (f64::from(dest) + 0.5 - origin) * (src_range / size) - 0.5
    }
}

/// Coarse tile source for fused trilinear blitting.
/// Maps a sub-region of a coarse-level tile onto the same screen rect
/// as a fine tile, blending the two per-pixel to avoid a full-screen
/// intermediate buffer and blend pass.
pub struct CoarseSrc<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub border: u32,
    /// UV sub-region of the coarse tile that covers this fine tile:
    /// (u_min, v_min, u_max, v_max) in 0..=1 normalised coordinates.
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Mip blend factor: 0.0 = 100 % fine, 1.0 = 100 % coarse.
    pub blend: f32,
}

/// Splits a continuous data-space position into a clamped pair of indices
/// and an 8-bit weight for the second, rounding the position to 1/256 px.
#[inline(always)]
fn linear_taps(data_pos: f64, max_index: u32) -> (u32, u32, u32) {
    let fp = (data_pos * 256.0).round();
    let fp = if fp.is_finite() { fp as i64 } else { 0 };
    let i0 = fp >> 8;
    let frac = (fp & 0xff) as u32;
    let max = i64::from(max_index);
    let a = i0.clamp(0, max) as u32;
    let b = (i0 + 1).clamp(0, max) as u32;
    (a, b, frac)
}

#[inline(always)]
fn compute_linear_col_samples(
    axis: AxisMapping,
    src_offset: f64,
    src_range: f64,
    border: u32,
    max_x: u32,
) -> Vec<LinearColSample> {
    (axis.start..axis.end)
        .map(|x| {
            let pos = axis.src_index(x, src_offset, src_range) + f64::from(border);
            let (x0, x1, frac) = linear_taps(pos, max_x);
            LinearColSample {
                x0_4: x0 as usize * 4,
                x1_4: x1 as usize * 4,
                w0: 256 - frac,
                w1: frac,
            }
        })
        .collect()
}

#[inline(always)]
fn compute_linear_row_samples(
    axis: AxisMapping,
    src_offset: f64,
    src_range: f64,
    border: u32,
    max_y: u32,
    stride: usize,
) -> Vec<LinearRowSample> {
    (axis.start..axis.end)
        .map(|y| {
            let pos = axis.src_index(y, src_offset, src_range) + f64::from(border);
            let (y0, y1, frac) = linear_taps(pos, max_y);
            LinearRowSample {
                row0: y0 as usize * stride,
                row1: y1 as usize * stride,
                w0: 256 - frac,
                w1: frac,
            }
        })
        .collect()
}

/// Fill an RGBA buffer with a single colour. Trailing bytes that do not
/// form a whole pixel are left untouched.
#[inline(always)]
pub fn fast_fill_rgba(buffer: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    // Writing whole `[u8; 4]` pixels keeps the channel order correct for any
    // slice alignment; LLVM vectorises the loop.
    let pixel = [r, g, b, a];
    for chunk in buffer.as_chunks_mut::<4>().0 {
        *chunk = pixel;
    }
}

/// Clipped destination bounds `[start_x, end_x) x [start_y, end_y)` of a
/// blit, or `None` if nothing is drawn.
fn clip(rect: &BlitRect, dest_width: u32, dest_height: u32) -> Option<(u32, u32, u32, u32)> {
    if rect.width <= 0 || rect.height <= 0 {
        return None;
    }
    let right = i64::from(rect.x) + i64::from(rect.width);
    let bottom = i64::from(rect.y) + i64::from(rect.height);
    let start_x = i64::from(rect.x).max(0);
    let start_y = i64::from(rect.y).max(0);
    let end_x = right.min(i64::from(dest_width));
    let end_y = bottom.min(i64::from(dest_height));
    (start_x < end_x && start_y < end_y).then_some((
        start_x as u32,
        start_y as u32,
        end_x as u32,
        end_y as u32,
    ))
}

/// Whether the tile maps 1:1 onto whole destination pixels, so it can be
/// copied without resampling.
fn is_identity(rect: &BlitRect, src_width: u32, src_height: u32) -> bool {
    if rect.width != src_width as i32 || rect.height != src_height as i32 {
        return false;
    }
    if rect.exact_width <= 0.0 {
        return true;
    }
    let aligned = |origin: f64, int: i32| (origin - f64::from(int)).abs() < 1e-6;
    (rect.exact_width - f64::from(src_width)).abs() < 1e-6
        && (rect.exact_height - f64::from(src_height)).abs() < 1e-6
        && aligned(rect.exact_x, rect.x)
        && aligned(rect.exact_y, rect.y)
}

#[allow(clippy::too_many_arguments)]
fn copy_identity(
    dest: &mut [u8],
    dest_width: u32,
    src: &[u8],
    src_width: u32,
    border: u32,
    rect: &BlitRect,
    (start_x, start_y, end_x, end_y): (u32, u32, u32, u32),
) {
    let data_width = (src_width + 2 * border) as usize;
    let dest_stride = dest_width as usize * 4;
    let src_stride = data_width * 4;
    let copy_width = (end_x - start_x) as usize * 4;
    let src_x_offset = (start_x as i64 - i64::from(rect.x)) as usize * 4 + border as usize * 4;
    for y in start_y..end_y {
        let src_y = (y as i64 - i64::from(rect.y)) as usize + border as usize;
        let src_off = src_y * src_stride + src_x_offset;
        let dest_off = y as usize * dest_stride + start_x as usize * 4;
        if src_off + copy_width <= src.len() && dest_off + copy_width <= dest.len() {
            dest[dest_off..dest_off + copy_width]
                .copy_from_slice(&src[src_off..src_off + copy_width]);
        }
    }
}

/// Whether `data` can hold a `width` x `height` tile with `border`.
fn tile_data_fits(data: &[u8], width: u32, height: u32, border: u32) -> bool {
    let w = u64::from(width) + 2 * u64::from(border);
    let h = u64::from(height) + 2 * u64::from(border);
    width > 0 && height > 0 && (data.len() as u64) >= w * h * 4
}

fn dest_fits(dest: &[u8], dest_width: u32, end_y: u32) -> bool {
    (dest.len() as u64) >= u64::from(dest_width) * u64::from(end_y) * 4
}

#[inline(always)]
fn round_shift16(value: u32) -> u8 {
    ((value + (1 << 15)) >> 16) as u8
}

/// Optimized bilinear tile blitter with cache-friendly access patterns
/// Uses fixed-point arithmetic and minimizes bounds checking.
///
/// `border`: number of padding pixels on each side of the source data.
/// When border > 0, `src` is `(src_width + 2*border) × (src_height + 2*border) × 4`
/// bytes and the bilinear filter can sample the border region at tile edges,
/// eliminating visible seams between adjacent tiles.
#[inline(always)]
pub fn blit_tile(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
) {
    let src = tile.data;
    let src_width = tile.width;
    let src_height = tile.height;
    let border = tile.border;
    let Some(bounds) = clip(&rect, dest_width, dest_height) else {
        return;
    };
    let (start_x, start_y, end_x, end_y) = bounds;
    if !tile_data_fits(src, src_width, src_height, border) || !dest_fits(dest, dest_width, end_y) {
        return;
    }

    if is_identity(&rect, src_width, src_height) {
        copy_identity(dest, dest_width, src, src_width, border, &rect, bounds);
        return;
    }

    let data_width = src_width + 2 * border;
    let data_height = src_height + 2 * border;
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (data_width * 4) as usize;

    let x_axis = AxisMapping {
        start: start_x,
        end: end_x,
        dest_origin: rect.x,
        exact_origin: rect.exact_x,
        exact_size: rect.exact_width,
        scaled_size: rect.width,
    };
    let y_axis = AxisMapping {
        start: start_y,
        end: end_y,
        dest_origin: rect.y,
        exact_origin: rect.exact_y,
        exact_size: rect.exact_height,
        scaled_size: rect.height,
    };

    let x_samples =
        compute_linear_col_samples(x_axis, 0.0, src_width as f64, border, data_width - 1);
    let y_samples = compute_linear_row_samples(
        y_axis,
        0.0,
        src_height as f64,
        border,
        data_height - 1,
        src_stride,
    );

    for (row_index, row_sample) in y_samples.iter().enumerate() {
        let y = start_y as usize + row_index;
        let dest_row = y * dest_stride;

        for (col_index, col_sample) in x_samples.iter().enumerate() {
            let dest_idx = dest_row + (start_x as usize + col_index) * 4;

            let w00 = col_sample.w0 * row_sample.w0;
            let w10 = col_sample.w1 * row_sample.w0;
            let w01 = col_sample.w0 * row_sample.w1;
            let w11 = col_sample.w1 * row_sample.w1;

            // SAFETY: tap indices are clamped to the data extent, which
            // `tile_data_fits` checked against `src.len()`; `dest_idx` lies in
            // the clipped rectangle, which `dest_fits` checked.
            unsafe {
                let s00 = src.get_unchecked(
                    row_sample.row0 + col_sample.x0_4..row_sample.row0 + col_sample.x0_4 + 4,
                );
                let s10 = src.get_unchecked(
                    row_sample.row0 + col_sample.x1_4..row_sample.row0 + col_sample.x1_4 + 4,
                );
                let s01 = src.get_unchecked(
                    row_sample.row1 + col_sample.x0_4..row_sample.row1 + col_sample.x0_4 + 4,
                );
                let s11 = src.get_unchecked(
                    row_sample.row1 + col_sample.x1_4..row_sample.row1 + col_sample.x1_4 + 4,
                );
                let d = dest.get_unchecked_mut(dest_idx..dest_idx + 4);

                for c in 0..4 {
                    d[c] = round_shift16(
                        s00[c] as u32 * w00
                            + s10[c] as u32 * w10
                            + s01[c] as u32 * w01
                            + s11[c] as u32 * w11,
                    );
                }
            }
        }
    }
}

/// Nearest-neighbour tile blitter: every destination pixel takes the
/// source pixel its centre falls in (pixel-centre convention), so a
/// magnified slide shows its pixels as sharp squares. Positions beyond the
/// inner area read the border, like the other samplers.
pub fn blit_tile_nearest(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
) {
    let src = tile.data;
    let (src_width, src_height, border) = (tile.width, tile.height, tile.border);
    let Some(bounds) = clip(&rect, dest_width, dest_height) else {
        return;
    };
    let (start_x, start_y, end_x, end_y) = bounds;
    if !tile_data_fits(src, src_width, src_height, border) || !dest_fits(dest, dest_width, end_y) {
        return;
    }
    if is_identity(&rect, src_width, src_height) {
        copy_identity(dest, dest_width, src, src_width, border, &rect, bounds);
        return;
    }
    let data_width = src_width + 2 * border;
    let data_height = src_height + 2 * border;
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (data_width * 4) as usize;
    let nearest = |axis: AxisMapping, src_range: u32, max_index: u32| -> Vec<usize> {
        (axis.start..axis.end)
            .map(|d| {
                let pos = axis.src_index(d, 0.0, f64::from(src_range)) + f64::from(border);
                let index = (pos + 0.5).floor();
                let index = if index.is_finite() { index as i64 } else { 0 };
                index.clamp(0, i64::from(max_index)) as usize
            })
            .collect()
    };
    let columns = nearest(
        AxisMapping {
            start: start_x,
            end: end_x,
            dest_origin: rect.x,
            exact_origin: rect.exact_x,
            exact_size: rect.exact_width,
            scaled_size: rect.width,
        },
        src_width,
        data_width - 1,
    );
    let rows = nearest(
        AxisMapping {
            start: start_y,
            end: end_y,
            dest_origin: rect.y,
            exact_origin: rect.exact_y,
            exact_size: rect.exact_height,
            scaled_size: rect.height,
        },
        src_height,
        data_height - 1,
    );
    for (row_index, &src_row) in rows.iter().enumerate() {
        let dest_row = (start_y as usize + row_index) * dest_stride + start_x as usize * 4;
        let src_row = src_row * src_stride;
        for (col_index, &src_col) in columns.iter().enumerate() {
            let s = src_row + src_col * 4;
            let d = dest_row + col_index * 4;
            dest[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
}

/// Lanczos kernel evaluation: sinc(x) * sinc(x/a)
#[inline(always)]
pub fn lanczos_weight(x: f64, a: f64) -> f64 {
    if x.abs() < 1e-8 {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    let pxa = px / a;
    (px.sin() / px) * (pxa.sin() / pxa)
}

/// Lanczos lobes (`a`).
const LANCZOS_A: f64 = 3.0;

/// Border (in source pixels) a tile needs for seamless Lanczos-3 at
/// `scale` source pixels per destination pixel: the kernel's support is
/// widened by `scale` when minifying (`scale > 1`), so it reaches
/// `ceil(3 * max(1, scale))` pixels beyond the sample.
pub fn lanczos_border_for_scale(scale: f64) -> u32 {
    let scale = if scale.is_finite() {
        scale.max(1.0)
    } else {
        1.0
    };
    (LANCZOS_A * scale).ceil() as u32 + 1
}

/// Per-output-pixel Lanczos taps along one axis, flattened.
struct LanczosAxis {
    /// First data index of each output pixel's taps.
    first: Vec<u32>,
    /// `offsets[i]..offsets[i + 1]` indexes `weights` for output pixel `i`.
    offsets: Vec<usize>,
    weights: Vec<f32>,
}

impl LanczosAxis {
    /// `max_index` is the last valid data index; taps beyond it (or below
    /// 0) repeat the edge, like the GPU's clamped sampler.
    fn new(axis: AxisMapping, src_range: f64, border: u32, max_index: u32) -> Self {
        let scale = axis.scale(src_range);
        // Widen the kernel when minifying so it low-pass filters.
        let stretch = if scale.is_finite() {
            scale.max(1.0)
        } else {
            1.0
        };
        let support = LANCZOS_A * stretch;
        let count = (axis.end - axis.start) as usize;
        let mut first = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count + 1);
        let mut weights = Vec::new();
        offsets.push(0);
        for d in axis.start..axis.end {
            let center = axis.src_index(d, 0.0, src_range) + f64::from(border);
            let lo = (center - support).floor() as i64 + 1;
            let hi = (center + support).ceil() as i64 - 1;
            let start = weights.len();
            let mut sum = 0.0f64;
            let mut taps = Vec::with_capacity((hi - lo + 1).max(1) as usize);
            for i in lo..=hi.max(lo) {
                let w = lanczos_weight((i as f64 - center) / stretch, LANCZOS_A);
                taps.push((i, w));
                sum += w;
            }
            if sum.abs() < 1e-9 {
                sum = 1.0;
            }
            // Clamp taps into the data, merging repeated edge taps so each
            // output pixel's taps are contiguous from `first`.
            let lo_clamped = lo.clamp(0, i64::from(max_index)) as u32;
            let hi_clamped = hi.max(lo).clamp(0, i64::from(max_index)) as u32;
            weights.resize(start + (hi_clamped - lo_clamped + 1) as usize, 0.0);
            for (i, w) in taps {
                let idx = i.clamp(0, i64::from(max_index)) as u32;
                weights[start + (idx - lo_clamped) as usize] += (w / sum) as f32;
            }
            first.push(lo_clamped);
            offsets.push(weights.len());
        }
        Self {
            first,
            offsets,
            weights,
        }
    }

    #[inline(always)]
    fn taps(&self, i: usize) -> (u32, &[f32]) {
        (
            self.first[i],
            &self.weights[self.offsets[i]..self.offsets[i + 1]],
        )
    }
}

/// Lanczos-3 tile blitter.
///
/// Separable (a horizontal then a vertical pass, f32 weights). When the
/// tile is minified the kernel is widened by the minification factor so it
/// anti-aliases instead of skipping source pixels. Taps outside the tile's
/// inner area read its border; for seamless results at tile edges the tile
/// needs [`lanczos_border_for_scale`] border pixels (narrower borders
/// repeat the outermost border pixel, like the GPU path).
#[inline(always)]
pub fn blit_tile_lanczos3(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
) {
    let src = tile.data;
    let src_width = tile.width;
    let src_height = tile.height;
    let border = tile.border;
    let Some(bounds) = clip(&rect, dest_width, dest_height) else {
        return;
    };
    let (start_x, start_y, end_x, end_y) = bounds;
    if !tile_data_fits(src, src_width, src_height, border) || !dest_fits(dest, dest_width, end_y) {
        return;
    }

    if is_identity(&rect, src_width, src_height) {
        copy_identity(dest, dest_width, src, src_width, border, &rect, bounds);
        return;
    }

    let data_width = src_width + 2 * border;
    let data_height = src_height + 2 * border;
    let dest_stride = dest_width as usize * 4;
    let src_stride = data_width as usize * 4;
    let x_axis = AxisMapping {
        start: start_x,
        end: end_x,
        dest_origin: rect.x,
        exact_origin: rect.exact_x,
        exact_size: rect.exact_width,
        scaled_size: rect.width,
    };
    let y_axis = AxisMapping {
        start: start_y,
        end: end_y,
        dest_origin: rect.y,
        exact_origin: rect.exact_y,
        exact_size: rect.exact_height,
        scaled_size: rect.height,
    };
    let cols = LanczosAxis::new(x_axis, f64::from(src_width), border, data_width - 1);
    let rows = LanczosAxis::new(y_axis, f64::from(src_height), border, data_height - 1);

    // Source rows the vertical pass needs.
    let row_count = (end_y - start_y) as usize;
    let first_row = rows.first.iter().copied().min().unwrap_or(0);
    let last_row = (0..row_count)
        .map(|i| {
            let (first, taps) = rows.taps(i);
            first + taps.len() as u32
        })
        .max()
        .unwrap_or(first_row);
    let col_count = (end_x - start_x) as usize;

    // Horizontal pass: filtered source rows x destination columns.
    let mut horizontal = vec![0f32; (last_row - first_row) as usize * col_count * 4];
    for (r, out_row) in horizontal.chunks_exact_mut(col_count * 4).enumerate() {
        let src_row = &src[(first_row as usize + r) * src_stride..][..src_stride];
        for (c, out) in out_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (first, taps) = cols.taps(c);
            let mut acc = [0f32; 4];
            for (k, &w) in taps.iter().enumerate() {
                let p = &src_row[(first as usize + k) * 4..][..4];
                acc[0] += f32::from(p[0]) * w;
                acc[1] += f32::from(p[1]) * w;
                acc[2] += f32::from(p[2]) * w;
                acc[3] += f32::from(p[3]) * w;
            }
            *out = acc;
        }
    }

    // Vertical pass.
    let h_stride = col_count * 4;
    for i in 0..row_count {
        let (first, taps) = rows.taps(i);
        let base = (first - first_row) as usize;
        let dest_row = &mut dest[(start_y as usize + i) * dest_stride + start_x as usize * 4..]
            [..col_count * 4];
        for c in 0..col_count {
            let mut acc = [0f32; 4];
            for (k, &w) in taps.iter().enumerate() {
                let p = &horizontal[(base + k) * h_stride + c * 4..][..4];
                acc[0] += p[0] * w;
                acc[1] += p[1] * w;
                acc[2] += p[2] * w;
                acc[3] += p[3] * w;
            }
            for ch in 0..4 {
                dest_row[c * 4 + ch] = (acc[ch] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Alpha-blend two RGBA buffers: dest = fine * (1-blend) + coarse * blend
/// Both buffers must be the same size (width * height * 4 bytes). `blend`
/// is clamped to `0..=1` (NaN counts as 0).
pub fn blend_buffers(dest: &mut [u8], coarse: &[u8], blend: f64) {
    let blend = if blend.is_nan() {
        0.0
    } else {
        blend.clamp(0.0, 1.0)
    };
    let b = (blend * 256.0).round() as u32;
    let inv_b = 256 - b;
    for (d, c) in dest
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(coarse.as_chunks::<4>().0)
    {
        for ch in 0..4 {
            d[ch] = ((d[ch] as u32 * inv_b + c[ch] as u32 * b + 128) >> 8) as u8;
        }
    }
}

/// Fused bilinear + trilinear blit: sample both fine and coarse tiles
/// per-pixel and blend in a single pass, eliminating one full-screen
/// allocation and two full-frame traversals.
#[inline(always)]
pub fn blit_tile_trilinear(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    coarse: &CoarseSrc,
    rect: BlitRect,
) {
    let fine_src = tile.data;
    let fine_w = tile.width;
    let fine_h = tile.height;
    let fine_border = tile.border;
    let Some((start_x, start_y, end_x, end_y)) = clip(&rect, dest_width, dest_height) else {
        return;
    };
    if !tile_data_fits(fine_src, fine_w, fine_h, fine_border) || !dest_fits(dest, dest_width, end_y)
    {
        return;
    }
    if !tile_data_fits(coarse.data, coarse.width, coarse.height, coarse.border) {
        // Fall back to fine-only if the coarse data is unusable.
        blit_tile(dest, dest_width, dest_height, tile, rect);
        return;
    }

    let blend = if coarse.blend.is_nan() {
        0.0
    } else {
        coarse.blend.clamp(0.0, 1.0)
    };
    let blend_i = (blend * 256.0).round() as u32;
    let inv_blend_i = 256 - blend_i;

    // --- Fine tile setup ---
    let fine_data_w = fine_w + 2 * fine_border;
    let fine_data_h = fine_h + 2 * fine_border;
    let dest_stride = (dest_width * 4) as usize;
    let fine_stride = (fine_data_w * 4) as usize;

    // --- Coarse tile setup ---
    let coarse_src = coarse.data;
    let coarse_w = coarse.width;
    let coarse_h = coarse.height;
    let coarse_border = coarse.border;
    let coarse_data_w = coarse_w + 2 * coarse_border;
    let coarse_data_h = coarse_h + 2 * coarse_border;
    let coarse_stride = (coarse_data_w * 4) as usize;

    // UV range in coarse tile pixels (content, excl. border)
    let cu_min = coarse.uv_min[0] as f64 * coarse_w as f64;
    let cv_min = coarse.uv_min[1] as f64 * coarse_h as f64;
    let cu_range = (coarse.uv_max[0] - coarse.uv_min[0]) as f64 * coarse_w as f64;
    let cv_range = (coarse.uv_max[1] - coarse.uv_min[1]) as f64 * coarse_h as f64;
    let x_axis = AxisMapping {
        start: start_x,
        end: end_x,
        dest_origin: rect.x,
        exact_origin: rect.exact_x,
        exact_size: rect.exact_width,
        scaled_size: rect.width,
    };
    let y_axis = AxisMapping {
        start: start_y,
        end: end_y,
        dest_origin: rect.y,
        exact_origin: rect.exact_y,
        exact_size: rect.exact_height,
        scaled_size: rect.height,
    };
    let fine_x_samples =
        compute_linear_col_samples(x_axis, 0.0, fine_w as f64, fine_border, fine_data_w - 1);
    let fine_y_samples = compute_linear_row_samples(
        y_axis,
        0.0,
        fine_h as f64,
        fine_border,
        fine_data_h - 1,
        fine_stride,
    );

    let coarse_x_samples =
        compute_linear_col_samples(x_axis, cu_min, cu_range, coarse_border, coarse_data_w - 1);
    let coarse_y_samples = compute_linear_row_samples(
        y_axis,
        cv_min,
        cv_range,
        coarse_border,
        coarse_data_h - 1,
        coarse_stride,
    );

    for row_index in 0..fine_y_samples.len() {
        let fine_row = fine_y_samples[row_index];
        let coarse_row = coarse_y_samples[row_index];
        let dest_row = (start_y as usize + row_index) * dest_stride;

        for col_index in 0..fine_x_samples.len() {
            let fine_col = fine_x_samples[col_index];
            let coarse_col = coarse_x_samples[col_index];
            let fw00 = fine_col.w0 * fine_row.w0;
            let fw10 = fine_col.w1 * fine_row.w0;
            let fw01 = fine_col.w0 * fine_row.w1;
            let fw11 = fine_col.w1 * fine_row.w1;

            let cw00 = coarse_col.w0 * coarse_row.w0;
            let cw10 = coarse_col.w1 * coarse_row.w0;
            let cw01 = coarse_col.w0 * coarse_row.w1;
            let cw11 = coarse_col.w1 * coarse_row.w1;

            let dest_idx = dest_row + (start_x as usize + col_index) * 4;

            // SAFETY: as in `blit_tile`, all tap indices are clamped to the
            // validated data extents and `dest_idx` to the validated rect.
            unsafe {
                let fs00 = fine_src.get_unchecked(
                    fine_row.row0 + fine_col.x0_4..fine_row.row0 + fine_col.x0_4 + 4,
                );
                let fs10 = fine_src.get_unchecked(
                    fine_row.row0 + fine_col.x1_4..fine_row.row0 + fine_col.x1_4 + 4,
                );
                let fs01 = fine_src.get_unchecked(
                    fine_row.row1 + fine_col.x0_4..fine_row.row1 + fine_col.x0_4 + 4,
                );
                let fs11 = fine_src.get_unchecked(
                    fine_row.row1 + fine_col.x1_4..fine_row.row1 + fine_col.x1_4 + 4,
                );

                let cs00 = coarse_src.get_unchecked(
                    coarse_row.row0 + coarse_col.x0_4..coarse_row.row0 + coarse_col.x0_4 + 4,
                );
                let cs10 = coarse_src.get_unchecked(
                    coarse_row.row0 + coarse_col.x1_4..coarse_row.row0 + coarse_col.x1_4 + 4,
                );
                let cs01 = coarse_src.get_unchecked(
                    coarse_row.row1 + coarse_col.x0_4..coarse_row.row1 + coarse_col.x0_4 + 4,
                );
                let cs11 = coarse_src.get_unchecked(
                    coarse_row.row1 + coarse_col.x1_4..coarse_row.row1 + coarse_col.x1_4 + 4,
                );

                let d = dest.get_unchecked_mut(dest_idx..dest_idx + 4);

                for c in 0..4 {
                    // Each sum is at most 255 * 2^16; blending two of them
                    // with 8-bit weights stays below 2^32.
                    let fine_val = fs00[c] as u32 * fw00
                        + fs10[c] as u32 * fw10
                        + fs01[c] as u32 * fw01
                        + fs11[c] as u32 * fw11;
                    let coarse_val = cs00[c] as u32 * cw00
                        + cs10[c] as u32 * cw10
                        + cs01[c] as u32 * cw01
                        + cs11[c] as u32 * cw11;
                    d[c] =
                        ((fine_val * inv_blend_i + coarse_val * blend_i + (1 << 23)) >> 24) as u8;
                }
            }
        }
    }
}

/// Size, position and zoom of a rendered frame, for [`try_reproject_frame`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameGeometry {
    /// Frame size in pixels.
    pub width: u32,
    pub height: u32,
    /// Level-0 coordinate of the frame's top-left corner.
    pub bounds_left: f64,
    pub bounds_top: f64,
    /// Screen pixels per level-0 pixel.
    pub zoom: f64,
}

/// Why [`try_reproject_frame`] refused to run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BlitError {
    #[error("destination holds {actual} bytes, {needed} needed")]
    DestinationTooSmall { needed: u64, actual: u64 },
    #[error("source holds {actual} bytes, {needed} needed")]
    SourceTooSmall { needed: u64, actual: u64 },
    #[error("zoom must be finite and positive")]
    InvalidZoom,
}

/// Copies `src`, a `width × height` RGBA frame drawn axis-aligned with the
/// image, into `dst` as `orientation` shows it on screen: `dst` is
/// `orientation.swap(width, height)` pixels. Exact (a permutation), for
/// quarter turns; other angles turn by their nearest quarter turn (see
/// [`FrameMap`] for drawing those).
pub fn orient_frame(
    src: &[u8],
    width: u32,
    height: u32,
    orientation: crate::viewport::Orientation,
    dst: &mut [u8],
) {
    let orientation = orientation.quarter_part();
    let need = width as usize * height as usize * 4;
    if src.len() < need || dst.len() < need || need == 0 {
        return;
    }
    if orientation.is_identity() {
        dst[..need].copy_from_slice(&src[..need]);
        return;
    }
    let (dst_width, _) = orientation.swap(width, height);
    let w = i64::from(width);
    // One destination column to the right is this step in the source.
    let step = orientation.to_image(glam::DVec2::new(1.0, 0.0));
    let (step_x, step_y) = (step.x as i64, step.y as i64);
    dst[..need]
        .par_chunks_mut(dst_width as usize * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let (mut sx, mut sy) = orientation.aligned_pixel(width, height, 0, y as i64);
            for px in row.as_chunks_mut::<4>().0 {
                let i = ((sy * w + sx) * 4) as usize;
                px.copy_from_slice(&src[i..i + 4]);
                sx += step_x;
                sy += step_y;
            }
        });
}

/// Bilinear reproject one RGBA frame onto another at a different viewport.
///
/// Inconsistent sizes (a buffer shorter than its stated dimensions) make
/// this a no-op; use [`try_reproject_frame`] to get the reason.
#[allow(clippy::too_many_arguments)]
pub fn reproject_frame(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src_pixels: &[u8],
    src_width: u32,
    src_height: u32,
    src_bounds_left: f64,
    src_bounds_top: f64,
    src_zoom: f64,
    dest_bounds_left: f64,
    dest_bounds_top: f64,
    dest_zoom: f64,
    clear_rgba: [u8; 4],
) {
    let result = try_reproject_frame(
        dest,
        FrameGeometry {
            width: dest_width,
            height: dest_height,
            bounds_left: dest_bounds_left,
            bounds_top: dest_bounds_top,
            zoom: dest_zoom,
        },
        src_pixels,
        FrameGeometry {
            width: src_width,
            height: src_height,
            bounds_left: src_bounds_left,
            bounds_top: src_bounds_top,
            zoom: src_zoom,
        },
        clear_rgba,
    );
    if let Err(err) = result {
        tracing::debug!("reproject_frame skipped: {err}");
    }
}

/// Bilinearly resamples the frame `src` (rendered with `src_geom`) into
/// `dest` (with `dest_geom`), filling uncovered pixels with `clear_rgba`.
/// Pixel-centre convention; alpha is interpolated like the colour channels.
pub fn try_reproject_frame(
    dest: &mut [u8],
    dest_geom: FrameGeometry,
    src_pixels: &[u8],
    src_geom: FrameGeometry,
    clear_rgba: [u8; 4],
) -> Result<(), BlitError> {
    let needed_dest = u64::from(dest_geom.width) * u64::from(dest_geom.height) * 4;
    if (dest.len() as u64) < needed_dest {
        return Err(BlitError::DestinationTooSmall {
            needed: needed_dest,
            actual: dest.len() as u64,
        });
    }
    let needed_src = u64::from(src_geom.width) * u64::from(src_geom.height) * 4;
    if (src_pixels.len() as u64) < needed_src {
        return Err(BlitError::SourceTooSmall {
            needed: needed_src,
            actual: src_pixels.len() as u64,
        });
    }
    let valid_zoom = |z: f64| z.is_finite() && z > 0.0;
    if !valid_zoom(src_geom.zoom) || !valid_zoom(dest_geom.zoom) {
        return Err(BlitError::InvalidZoom);
    }
    if dest_geom.width == 0 || dest_geom.height == 0 {
        return Ok(());
    }
    let dest = &mut dest[..needed_dest as usize];
    let [r, g, b, a] = clear_rgba;
    fast_fill_rgba(dest, r, g, b, a);
    if src_geom.width == 0 || src_geom.height == 0 {
        return Ok(());
    }

    let src_stride = (src_geom.width * 4) as usize;
    let dest_stride = (dest_geom.width * 4) as usize;
    let src_w = f64::from(src_geom.width);
    let src_h = f64::from(src_geom.height);

    // Continuous source index of a destination pixel centre, or None when the
    // centre falls outside the source frame.
    let map = |d: u32, dest_origin: f64, src_origin: f64, extent: f64| -> Option<f64> {
        let image = dest_origin + (f64::from(d) + 0.5) / dest_geom.zoom;
        let src = (image - src_origin) * src_geom.zoom;
        (0.0..extent).contains(&src).then_some(src - 0.5)
    };

    let x_samples: Vec<Option<LinearColSample>> = (0..dest_geom.width)
        .map(|x| {
            let pos = map(x, dest_geom.bounds_left, src_geom.bounds_left, src_w)?;
            let (x0, x1, frac) = linear_taps(pos, src_geom.width - 1);
            Some(LinearColSample {
                x0_4: x0 as usize * 4,
                x1_4: x1 as usize * 4,
                w0: 256 - frac,
                w1: frac,
            })
        })
        .collect();

    dest.par_chunks_mut(dest_stride)
        .enumerate()
        .for_each(|(y, dest_row)| {
            let Some(pos) = map(y as u32, dest_geom.bounds_top, src_geom.bounds_top, src_h) else {
                return;
            };
            let (y0, y1, frac_y) = linear_taps(pos, src_geom.height - 1);
            let row0 = &src_pixels[y0 as usize * src_stride..][..src_stride];
            let row1 = &src_pixels[y1 as usize * src_stride..][..src_stride];
            let (wy0, wy1) = (256 - frac_y, frac_y);
            for (x, col_sample) in x_samples.iter().enumerate() {
                let Some(s) = col_sample else {
                    continue;
                };
                let w00 = s.w0 * wy0;
                let w10 = s.w1 * wy0;
                let w01 = s.w0 * wy1;
                let w11 = s.w1 * wy1;
                let d = &mut dest_row[x * 4..x * 4 + 4];
                for c in 0..4 {
                    d[c] = round_shift16(
                        row0[s.x0_4 + c] as u32 * w00
                            + row0[s.x1_4 + c] as u32 * w10
                            + row1[s.x0_4 + c] as u32 * w01
                            + row1[s.x1_4 + c] as u32 * w11,
                    );
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bordered RGBA tile from a grey-level function of inner pixel
    /// coordinates; the border repeats the edge like `create_bordered_tile`.
    fn tile(width: u32, height: u32, border: u32, grey: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let (dw, dh) = (width + 2 * border, height + 2 * border);
        let mut data = Vec::with_capacity((dw * dh * 4) as usize);
        for y in 0..dh {
            for x in 0..dw {
                let ix = x.saturating_sub(border).min(width - 1);
                let iy = y.saturating_sub(border).min(height - 1);
                let v = grey(ix, iy);
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        data
    }

    fn rect(x: f64, y: f64, w: f64, h: f64) -> BlitRect {
        BlitRect {
            x: x.floor() as i32,
            y: y.floor() as i32,
            width: ((x + w).floor() - x.floor()) as i32,
            height: ((y + h).floor() - y.floor()) as i32,
            exact_x: x,
            exact_y: y,
            exact_width: w,
            exact_height: h,
        }
    }

    fn row_greys(dest: &[u8], width: usize, row: usize) -> Vec<u8> {
        (0..width).map(|x| dest[(row * width + x) * 4]).collect()
    }

    /// Exact pixel-centre bilinear reference for a 1-D grey row with
    /// edge clamping.
    fn reference_bilinear(values: &[f64], dest: u32, origin: f64, scale: f64) -> f64 {
        let p = (f64::from(dest) + 0.5 - origin) * scale - 0.5;
        let i0 = p.floor();
        let t = p - i0;
        let at = |i: f64| values[(i.max(0.0) as usize).min(values.len() - 1)];
        at(i0) * (1.0 - t) + at(i0 + 1.0) * t
    }

    #[test]
    fn bilinear_uses_pixel_centres_and_rounds() {
        // The audit's case: a black|white 2-px tile blitted at 4x used to
        // give [0, 63, 127, 191, 255, ...] (top-left convention, truncation).
        let src = tile(2, 2, 1, |x, _| if x == 0 { 0 } else { 255 });
        let mut dest = vec![0u8; 8 * 8 * 4];
        blit_tile(
            &mut dest,
            8,
            8,
            TileSrc {
                data: &src,
                width: 2,
                height: 2,
                border: 1,
            },
            rect(0.0, 0.0, 8.0, 8.0),
        );
        assert_eq!(row_greys(&dest, 8, 3), [0, 0, 32, 96, 159, 223, 255, 255]);
        assert!(dest.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
    }

    #[test]
    fn bilinear_matches_the_exact_reference_within_one_level() {
        let values: Vec<f64> = (0..32).map(|x| ((x * 37) % 256) as f64).collect();
        let src = tile(32, 4, 1, |x, _| values[x as usize] as u8);
        for (origin, width) in [(0.0, 77.0), (3.3, 50.5), (-7.6, 20.25), (1.0, 13.0)] {
            let dest_w = 100u32;
            let mut dest = vec![0u8; dest_w as usize * 4 * 4];
            blit_tile(
                &mut dest,
                dest_w,
                4,
                TileSrc {
                    data: &src,
                    width: 32,
                    height: 4,
                    border: 1,
                },
                rect(origin, 0.0, width, 4.0),
            );
            let r = rect(origin, 0.0, width, 4.0);
            for d in r.x.max(0)..(r.x + r.width).min(dest_w as i32) {
                let expected = reference_bilinear(&values, d as u32, origin, 32.0 / width);
                let got = f64::from(dest[d as usize * 4]);
                assert!(
                    (got - expected).abs() <= 1.0,
                    "origin {origin} width {width} d {d}: {got} vs {expected:.2}"
                );
            }
        }
    }

    #[test]
    fn adjacent_tiles_meet_without_seams() {
        // One 8-px ramp split into two bordered 4-px tiles must render like
        // the unsplit ramp.
        let ramp = |x: u32| (x * 30) as u8;
        let whole = tile(8, 1, 1, |x, _| ramp(x));
        let mut left = tile(4, 1, 1, |x, _| ramp(x));
        let mut right = tile(4, 1, 1, |x, _| ramp(x + 4));
        // Real borders hold the neighbour's pixels (data rows are 6 px).
        for row in 0..3 {
            left[(row * 6 + 5) * 4..(row * 6 + 5) * 4 + 3].fill(ramp(4));
            right[row * 6 * 4..row * 6 * 4 + 3].fill(ramp(3));
        }
        let mut expected = vec![0u8; 24 * 4];
        let mut split = vec![0u8; 24 * 4];
        fn src(data: &[u8], width: u32) -> TileSrc<'_> {
            TileSrc {
                data,
                width,
                height: 1,
                border: 1,
            }
        }
        blit_tile(
            &mut expected,
            24,
            1,
            src(&whole, 8),
            rect(0.0, 0.0, 24.0, 1.0),
        );
        blit_tile(&mut split, 24, 1, src(&left, 4), rect(0.0, 0.0, 12.0, 1.0));
        blit_tile(
            &mut split,
            24,
            1,
            src(&right, 4),
            rect(12.0, 0.0, 12.0, 1.0),
        );
        assert_eq!(row_greys(&split, 24, 0), row_greys(&expected, 24, 0));
    }

    #[test]
    fn identity_blits_copy_only_when_pixel_aligned() {
        let src = tile(4, 4, 1, |x, y| (x * 40 + y) as u8);
        let mut aligned = vec![0u8; 8 * 8 * 4];
        blit_tile(
            &mut aligned,
            8,
            8,
            TileSrc {
                data: &src,
                width: 4,
                height: 4,
                border: 1,
            },
            rect(2.0, 2.0, 4.0, 4.0),
        );
        assert_eq!(aligned[(2 * 8 + 2) * 4], 0);
        assert_eq!(aligned[(2 * 8 + 3) * 4], 40);
        // Half a pixel off: interpolated, not copied.
        let mut shifted = vec![0u8; 8 * 8 * 4];
        blit_tile(
            &mut shifted,
            8,
            8,
            TileSrc {
                data: &src,
                width: 4,
                height: 4,
                border: 1,
            },
            BlitRect {
                exact_x: 2.5,
                ..rect(2.0, 2.0, 4.0, 4.0)
            },
        );
        assert_eq!(shifted[(2 * 8 + 3) * 4], 20);
    }

    #[test]
    fn trilinear_blends_with_rounding() {
        let fine = tile(2, 2, 1, |_, _| 100);
        let coarse = tile(2, 2, 1, |_, _| 201);
        let mut dest = vec![0u8; 4 * 4 * 4];
        blit_tile_trilinear(
            &mut dest,
            4,
            4,
            TileSrc {
                data: &fine,
                width: 2,
                height: 2,
                border: 1,
            },
            &CoarseSrc {
                data: &coarse,
                width: 2,
                height: 2,
                border: 1,
                uv_min: [0.0, 0.0],
                uv_max: [0.5, 0.5],
                blend: 0.5,
            },
            rect(0.0, 0.0, 4.0, 4.0),
        );
        // (100 + 201) / 2 = 150.5 -> 151 with rounding (truncation gave 150).
        assert!(
            dest.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 151 && p[3] == 255)
        );
    }

    /// The pixel view: magnified, every output pixel is the source pixel
    /// its centre falls in (sharp squares, no blending), seamless at tile
    /// edges through the border.
    #[test]
    fn nearest_blits_show_the_source_pixels() {
        let src = tile(4, 3, 1, |x, y| (x * 60 + y * 7) as u8);
        let (w, h) = (4.0 * 2.5, 3.0 * 2.5);
        let mut dest = vec![0u8; 16 * 16 * 4];
        let r = rect(1.3, 2.2, w, h);
        blit_tile_nearest(
            &mut dest,
            16,
            16,
            TileSrc {
                data: &src,
                width: 4,
                height: 3,
                border: 1,
            },
            r,
        );
        for y in r.y.max(0)..(r.y + r.height).min(16) {
            for x in r.x.max(0)..(r.x + r.width).min(16) {
                // Continuous source position of the pixel's centre.
                let sx = (f64::from(x) + 0.5 - 1.3) / 2.5;
                let sy = (f64::from(y) + 0.5 - 2.2) / 2.5;
                let (ix, iy) = (
                    sx.floor().clamp(0.0, 3.0) as u32,
                    sy.floor().clamp(0.0, 2.0) as u32,
                );
                let expected = (ix * 60 + iy * 7) as u8;
                assert_eq!(dest[((y * 16 + x) * 4) as usize], expected, "at {x},{y}");
            }
        }
        // Only source values appear: nothing is blended.
        let values: std::collections::HashSet<u8> = (0..4)
            .flat_map(|x| (0..3).map(move |y| (x * 60 + y * 7) as u8))
            .collect();
        for y in r.y.max(0)..(r.y + r.height).min(16) {
            for x in r.x.max(0)..(r.x + r.width).min(16) {
                assert!(values.contains(&dest[((y * 16 + x) * 4) as usize]));
            }
        }
    }

    #[test]
    fn lanczos_preserves_flat_areas_and_matches_bilinear_geometry() {
        let src = tile(16, 16, 3, |_, _| 77);
        for (w, h) in [(64.0, 64.0), (5.0, 5.0), (16.0, 16.0), (23.5, 9.25)] {
            let mut dest = vec![0u8; 64 * 64 * 4];
            blit_tile_lanczos3(
                &mut dest,
                64,
                64,
                TileSrc {
                    data: &src,
                    width: 16,
                    height: 16,
                    border: 3,
                },
                rect(0.3, 0.7, w, h),
            );
            let r = rect(0.3, 0.7, w, h);
            for y in r.y.max(0)..(r.y + r.height).min(64) {
                for x in r.x.max(0)..(r.x + r.width).min(64) {
                    assert_eq!(dest[((y * 64 + x) * 4) as usize], 77, "{w}x{h} at {x},{y}");
                }
            }
        }
        // A step edge is centred where bilinear centres it.
        let step = tile(8, 1, 3, |x, _| if x < 4 { 0 } else { 255 });
        let mut dest = vec![0u8; 32 * 4];
        blit_tile_lanczos3(
            &mut dest,
            32,
            1,
            TileSrc {
                data: &step,
                width: 8,
                height: 1,
                border: 3,
            },
            rect(0.0, 0.0, 32.0, 1.0),
        );
        let greys = row_greys(&dest, 32, 0);
        // Symmetric around the edge between dest pixels 15 and 16.
        for i in 0..16 {
            let a = i32::from(greys[15 - i]);
            let b = i32::from(greys[16 + i]);
            assert!((a + b - 255).abs() <= 1, "{greys:?}");
        }
    }

    #[test]
    fn lanczos_minification_low_passes_fine_detail() {
        // A 1-px checkerboard minified 4x must come out as flat mid-grey,
        // not alias into stripes as fixed 6-tap Lanczos did.
        let src = tile(64, 64, 16, |x, y| if (x + y) % 2 == 0 { 0 } else { 255 });
        let mut dest = vec![0u8; 16 * 16 * 4];
        blit_tile_lanczos3(
            &mut dest,
            16,
            16,
            TileSrc {
                data: &src,
                width: 64,
                height: 64,
                border: 16,
            },
            rect(0.0, 0.0, 16.0, 16.0),
        );
        for p in dest.as_chunks::<4>().0 {
            assert!((i32::from(p[0]) - 128).abs() <= 3, "{}", p[0]);
        }
        assert_eq!(lanczos_border_for_scale(4.0), 13);
        assert_eq!(lanczos_border_for_scale(0.5), 4);
        assert_eq!(lanczos_border_for_scale(f64::NAN), 4);
    }

    #[test]
    fn blitters_ignore_inconsistent_buffers() {
        let src = tile(4, 4, 1, |_, _| 9);
        let mut small_dest = vec![0u8; 10];
        for f in [blit_tile, blit_tile_lanczos3] {
            f(
                &mut small_dest,
                8,
                8,
                TileSrc {
                    data: &src,
                    width: 4,
                    height: 4,
                    border: 1,
                },
                rect(0.0, 0.0, 8.0, 8.0),
            );
            // Tile data shorter than its declared size.
            let mut dest = vec![0u8; 8 * 8 * 4];
            f(
                &mut dest,
                8,
                8,
                TileSrc {
                    data: &src[..20],
                    width: 4,
                    height: 4,
                    border: 1,
                },
                rect(0.0, 0.0, 8.0, 8.0),
            );
            assert!(dest.iter().all(|&b| b == 0));
        }
        assert!(small_dest.iter().all(|&b| b == 0));
    }

    #[test]
    fn reproject_validates_sizes_and_uses_pixel_centres() {
        let src_geom = FrameGeometry {
            width: 4,
            height: 1,
            bounds_left: 0.0,
            bounds_top: 0.0,
            zoom: 1.0,
        };
        let src: Vec<u8> = [0u8, 100, 200, 250]
            .iter()
            .flat_map(|&v| [v, v, v, 128])
            .collect();
        // Identity mapping reproduces the frame exactly (alpha included).
        let mut dest = vec![0u8; 16];
        try_reproject_frame(&mut dest, src_geom, &src, src_geom, [1, 2, 3, 4]).unwrap();
        assert_eq!(dest, src);

        // 2x zoom-in around the left edge.
        let dest_geom = FrameGeometry {
            width: 4,
            zoom: 2.0,
            ..src_geom
        };
        let mut dest = vec![0u8; 16];
        try_reproject_frame(&mut dest, dest_geom, &src, src_geom, [1, 2, 3, 4]).unwrap();
        assert_eq!(row_greys(&dest, 4, 0), [0, 25, 75, 125]);

        // Uncovered pixels get the clear colour.
        let shifted = FrameGeometry {
            bounds_left: -2.0,
            ..src_geom
        };
        let mut dest = vec![0u8; 16];
        try_reproject_frame(&mut dest, shifted, &src, src_geom, [1, 2, 3, 4]).unwrap();
        assert_eq!(&dest[..4], [1, 2, 3, 4]);
        assert_eq!(&dest[8..12], [0, 0, 0, 128]);

        // Inconsistent sizes are refused instead of reading out of bounds.
        let mut short = vec![0u8; 15];
        assert!(matches!(
            try_reproject_frame(&mut short, src_geom, &src, src_geom, [0; 4]),
            Err(BlitError::DestinationTooSmall { .. })
        ));
        let mut dest = vec![7u8; 16];
        let big = FrameGeometry {
            width: 400,
            ..src_geom
        };
        assert!(matches!(
            try_reproject_frame(&mut dest, src_geom, &src, big, [0; 4]),
            Err(BlitError::SourceTooSmall { .. })
        ));
        let zero_zoom = FrameGeometry {
            zoom: 0.0,
            ..src_geom
        };
        assert_eq!(
            try_reproject_frame(&mut dest, src_geom, &src, zero_zoom, [0; 4]),
            Err(BlitError::InvalidZoom)
        );
        // The legacy entry point is a no-op on bad input.
        reproject_frame(
            &mut dest, 4, 1, &src, 400, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, [0; 4],
        );
        assert!(dest.iter().all(|&b| b == 7));
    }

    #[test]
    fn fills_and_blends_handle_awkward_input() {
        // Misaligned slice start: the old u32 fill wrote pixels out of phase.
        let mut backing = [0u8; 4 * 5 + 3];
        fast_fill_rgba(&mut backing[1..], 10, 20, 30, 40);
        assert_eq!(&backing[1..5], [10, 20, 30, 40]);
        assert_eq!(&backing[17..21], [10, 20, 30, 40]);
        assert_eq!(backing[21], 0, "partial trailing pixel untouched");

        let coarse = [200u8; 8];
        for blend in [1.5, 4.0] {
            let mut fine = [100u8; 8];
            // Used to underflow `256 - b` and panic (debug) or wrap.
            blend_buffers(&mut fine, &coarse, blend);
            assert_eq!(fine, [200; 8]);
        }
        let mut fine = [100u8; 8];
        blend_buffers(&mut fine, &coarse, f64::NAN);
        assert_eq!(fine, [100; 8]);
        let mut fine = [101u8; 4];
        blend_buffers(&mut fine, &[200; 4], 0.5);
        assert_eq!(fine, [151; 4]);
    }
}

#[cfg(test)]
mod orient_tests {
    use super::orient_frame;
    use crate::viewport::Orientation;

    fn frame(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| [i as u8, (i >> 8) as u8, 7, 255])
            .collect()
    }

    #[test]
    fn frames_turn_exactly_and_back() {
        let src = frame(3, 2);
        let mut turned = vec![0; src.len()];
        orient_frame(&src, 3, 2, Orientation::new(1, false), &mut turned);
        // Clockwise: the bottom-left pixel (index 3) comes to the top-left.
        let first: Vec<u8> = turned.chunks(4).map(|px| px[0]).collect();
        assert_eq!(first, vec![3, 0, 4, 1, 5, 2]);
        for orientation in Orientation::all() {
            let src = frame(7, 4);
            let (w, h) = orientation.swap(7, 4);
            let mut turned = vec![0; src.len()];
            orient_frame(&src, 7, 4, orientation, &mut turned);
            let mut back = vec![0; src.len()];
            orient_frame(&turned, w, h, orientation.inverse(), &mut back);
            assert_eq!(back, src, "{orientation:?}");
        }
    }
}
