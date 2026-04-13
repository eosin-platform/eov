//! Low-level tile blitting functions for CPU rendering
//!
//! This module contains optimized functions for copying and scaling tile data
//! to the viewport buffer.

use slint::{Rgba8Pixel, SharedPixelBuffer};

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

#[derive(Clone, Copy)]
struct LanczosColSample {
    offsets: [usize; 6],
    weights: [f32; 6],
}

#[derive(Clone, Copy)]
struct LanczosRowSample {
    rows: [usize; 6],
    weights: [f32; 6],
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

#[inline(always)]
fn compute_linear_col_samples(
    start_x: u32,
    end_x: u32,
    dest_x: i32,
    exact_x: f64,
    exact_width: f64,
    scaled_width: i32,
    src_width: u32,
    border: u32,
    max_x: u32,
) -> Vec<LinearColSample> {
    let use_exact = exact_width > 0.0;
    let mapped_width = if use_exact {
        exact_width.max(1.0)
    } else {
        scaled_width.max(1) as f64
    };
    let scale = src_width as f64 / mapped_width;

    (start_x..end_x)
        .map(|x| {
            let src_xf = if use_exact {
                (x as f64 - exact_x) * scale
            } else {
                (x as i32 - dest_x) as f64 * scale
            };
            let src_x_fp = (src_xf * 65536.0).max(0.0) as u32;
            let x0_inner = src_x_fp >> 16;
            let frac = (src_x_fp & 0xFFFF) >> 8;
            let x0 = (x0_inner + border).min(max_x);
            let x1 = (x0_inner + border + 1).min(max_x);
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
fn compute_linear_col_samples_subregion(
    start_x: u32,
    end_x: u32,
    dest_x: i32,
    exact_x: f64,
    exact_width: f64,
    scaled_width: i32,
    src_offset: f64,
    src_range: f64,
    border: u32,
    max_x: u32,
) -> Vec<LinearColSample> {
    let use_exact = exact_width > 0.0;
    let mapped_width = if use_exact {
        exact_width.max(1.0)
    } else {
        scaled_width.max(1) as f64
    };
    let scale = src_range / mapped_width;

    (start_x..end_x)
        .map(|x| {
            let src_xf = if use_exact {
                src_offset + (x as f64 - exact_x) * scale
            } else {
                src_offset + (x as i32 - dest_x) as f64 * scale
            };
            let src_x_fp = (src_xf * 65536.0).max(0.0) as u32;
            let x0_inner = src_x_fp >> 16;
            let frac = (src_x_fp & 0xFFFF) >> 8;
            let x0 = (x0_inner + border).min(max_x);
            let x1 = (x0_inner + border + 1).min(max_x);
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
    start_y: u32,
    end_y: u32,
    dest_y: i32,
    exact_y: f64,
    exact_height: f64,
    scaled_height: i32,
    src_height: u32,
    border: u32,
    max_y: u32,
    stride: usize,
) -> Vec<LinearRowSample> {
    let use_exact = exact_height > 0.0;
    let mapped_height = if use_exact {
        exact_height.max(1.0)
    } else {
        scaled_height.max(1) as f64
    };
    let scale = src_height as f64 / mapped_height;

    (start_y..end_y)
        .map(|y| {
            let src_yf = if use_exact {
                (y as f64 - exact_y) * scale
            } else {
                (y as i32 - dest_y) as f64 * scale
            };
            let src_y_fp = (src_yf * 65536.0).max(0.0) as u32;
            let y0_inner = src_y_fp >> 16;
            let frac = (src_y_fp & 0xFFFF) >> 8;
            let y0 = (y0_inner + border).min(max_y);
            let y1 = (y0_inner + border + 1).min(max_y);
            LinearRowSample {
                row0: y0 as usize * stride,
                row1: y1 as usize * stride,
                w0: 256 - frac,
                w1: frac,
            }
        })
        .collect()
}

#[inline(always)]
fn compute_linear_row_samples_subregion(
    start_y: u32,
    end_y: u32,
    dest_y: i32,
    exact_y: f64,
    exact_height: f64,
    scaled_height: i32,
    src_offset: f64,
    src_range: f64,
    border: u32,
    max_y: u32,
    stride: usize,
) -> Vec<LinearRowSample> {
    let use_exact = exact_height > 0.0;
    let mapped_height = if use_exact {
        exact_height.max(1.0)
    } else {
        scaled_height.max(1) as f64
    };
    let scale = src_range / mapped_height;

    (start_y..end_y)
        .map(|y| {
            let src_yf = if use_exact {
                src_offset + (y as f64 - exact_y) * scale
            } else {
                src_offset + (y as i32 - dest_y) as f64 * scale
            };
            let src_y_fp = (src_yf * 65536.0).max(0.0) as u32;
            let y0_inner = src_y_fp >> 16;
            let frac = (src_y_fp & 0xFFFF) >> 8;
            let y0 = (y0_inner + border).min(max_y);
            let y1 = (y0_inner + border + 1).min(max_y);
            LinearRowSample {
                row0: y0 as usize * stride,
                row1: y1 as usize * stride,
                w0: 256 - frac,
                w1: frac,
            }
        })
        .collect()
}

#[inline(always)]
fn normalize_lanczos_weights(weights: &mut [f32; 6]) {
    let sum = weights.iter().copied().sum::<f32>();
    if sum.abs() > 1e-6 {
        for weight in weights.iter_mut() {
            *weight /= sum;
        }
    }
}

#[inline(always)]
fn compute_lanczos_col_samples(
    start_x: u32,
    end_x: u32,
    dest_x: i32,
    exact_x: f64,
    exact_width: f64,
    scaled_width: i32,
    src_width: u32,
    border: u32,
    max_x: i32,
) -> Vec<LanczosColSample> {
    let use_exact = exact_width > 0.0;
    let mapped_width = if use_exact {
        exact_width.max(1.0)
    } else {
        scaled_width.max(1) as f64
    };
    let scale = src_width as f64 / mapped_width;

    (start_x..end_x)
        .map(|x| {
            let src_xf = if use_exact {
                (x as f64 - exact_x) * scale
            } else {
                (x as i32 - dest_x) as f64 * scale
            }
            .max(0.0);
            let center_x = src_xf.floor() as i32;
            let frac_x = src_xf - center_x as f64;
            let mut offsets = [0usize; 6];
            let mut weights = [0.0f32; 6];
            for (slot, tap) in (-2..=3_i32).enumerate() {
                let sx = (center_x + tap + border as i32).clamp(0, max_x) as usize;
                offsets[slot] = sx * 4;
                weights[slot] = lanczos_weight(tap as f64 - frac_x, 3.0) as f32;
            }
            normalize_lanczos_weights(&mut weights);
            LanczosColSample { offsets, weights }
        })
        .collect()
}

#[inline(always)]
fn compute_lanczos_row_samples(
    start_y: u32,
    end_y: u32,
    dest_y: i32,
    exact_y: f64,
    exact_height: f64,
    scaled_height: i32,
    src_height: u32,
    border: u32,
    max_y: i32,
    stride: usize,
) -> Vec<LanczosRowSample> {
    let use_exact = exact_height > 0.0;
    let mapped_height = if use_exact {
        exact_height.max(1.0)
    } else {
        scaled_height.max(1) as f64
    };
    let scale = src_height as f64 / mapped_height;

    (start_y..end_y)
        .map(|y| {
            let src_yf = if use_exact {
                (y as f64 - exact_y) * scale
            } else {
                (y as i32 - dest_y) as f64 * scale
            }
            .max(0.0);
            let center_y = src_yf.floor() as i32;
            let frac_y = src_yf - center_y as f64;
            let mut rows = [0usize; 6];
            let mut weights = [0.0f32; 6];
            for (slot, tap) in (-2..=3_i32).enumerate() {
                let sy = (center_y + tap + border as i32).clamp(0, max_y) as usize;
                rows[slot] = sy * stride;
                weights[slot] = lanczos_weight(tap as f64 - frac_y, 3.0) as f32;
            }
            normalize_lanczos_weights(&mut weights);
            LanczosRowSample { rows, weights }
        })
        .collect()
}

/// Fast fill RGBA buffer with a single color using u32 writes
/// This is ~4x faster than byte-by-byte writes on most architectures
#[inline(always)]
pub fn fast_fill_rgba(buffer: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    let pixel = u32::from_ne_bytes([r, g, b, a]);

    if buffer.len() >= 4 && buffer.len().is_multiple_of(4) {
        let (prefix, pixels, suffix) = unsafe { buffer.align_to_mut::<u32>() };

        for chunk in prefix.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }

        pixels.fill(pixel);

        for chunk in suffix.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }
    } else {
        for chunk in buffer.chunks_exact_mut(4) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
            chunk[3] = a;
        }
    }
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
    let BlitRect {
        x: dest_x,
        y: dest_y,
        width: scaled_width,
        height: scaled_height,
        ..
    } = rect;
    if scaled_width <= 0 || scaled_height <= 0 {
        return;
    }

    if dest_x + scaled_width <= 0 || dest_y + scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }

    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + scaled_height).min(dest_height as i32) as u32;

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    // Fast path: 1:1 mapping (tile pixels map exactly to screen pixels).
    // Bilinear interpolation with zero fractional parts is equivalent to
    // direct copy, so row-by-row memcpy produces identical output.
    if scaled_width == src_width as i32 && scaled_height == src_height as i32 {
        let data_width = (src_width + 2 * border) as usize;
        let dest_stride = (dest_width * 4) as usize;
        let src_stride = data_width * 4;
        let copy_width = (end_x - start_x) as usize * 4;
        let src_x_offset = (start_x as i32 - dest_x) as usize * 4 + border as usize * 4;
        for y in start_y..end_y {
            let src_y = (y as i32 - dest_y) as usize + border as usize;
            let src_off = src_y * src_stride + src_x_offset;
            let dest_off = y as usize * dest_stride + start_x as usize * 4;
            if src_off + copy_width <= src.len() && dest_off + copy_width <= dest.len() {
                dest[dest_off..dest_off + copy_width]
                    .copy_from_slice(&src[src_off..src_off + copy_width]);
            }
        }
        return;
    }

    let data_width = src_width + 2 * border;
    let data_height = src_height + 2 * border;
    let data_w_minus_1 = data_width.saturating_sub(1);
    let data_h_minus_1 = data_height.saturating_sub(1);

    // Use exact floating-point tile size for UV mapping when available.
    // This keeps the src→dest mapping stable even when the integer rect
    // oscillates ±1 px during smooth zoom.
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (data_width * 4) as usize;

    let src_max_idx = data_h_minus_1 as usize * src_stride + data_w_minus_1 as usize * 4 + 3;
    if src.len() <= src_max_idx {
        return;
    }

    let dest_max_idx = (end_y - 1) as usize * dest_stride + (end_x - 1) as usize * 4 + 3;
    if dest.len() <= dest_max_idx {
        return;
    }

    let x_samples = compute_linear_col_samples(
        start_x,
        end_x,
        dest_x,
        rect.exact_x,
        rect.exact_width,
        scaled_width,
        src_width,
        border,
        data_w_minus_1,
    );
    let y_samples = compute_linear_row_samples(
        start_y,
        end_y,
        dest_y,
        rect.exact_y,
        rect.exact_height,
        scaled_height,
        src_height,
        border,
        data_h_minus_1,
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

                d[0] = ((s00[0] as u32 * w00
                    + s10[0] as u32 * w10
                    + s01[0] as u32 * w01
                    + s11[0] as u32 * w11)
                    >> 16) as u8;
                d[1] = ((s00[1] as u32 * w00
                    + s10[1] as u32 * w10
                    + s01[1] as u32 * w01
                    + s11[1] as u32 * w11)
                    >> 16) as u8;
                d[2] = ((s00[2] as u32 * w00
                    + s10[2] as u32 * w10
                    + s01[2] as u32 * w01
                    + s11[2] as u32 * w11)
                    >> 16) as u8;
                d[3] = ((s00[3] as u32 * w00
                    + s10[3] as u32 * w10
                    + s01[3] as u32 * w01
                    + s11[3] as u32 * w11)
                    >> 16) as u8;
            }
        }
    }
}

/// Lanczos kernel evaluation: sinc(x) * sinc(x/a)
#[inline(always)]
fn lanczos_weight(x: f64, a: f64) -> f64 {
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

/// Lanczos-3 tile blitter (a=3, 6x6 kernel)
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
    let BlitRect {
        x: dest_x,
        y: dest_y,
        width: scaled_width,
        height: scaled_height,
        ..
    } = rect;
    if scaled_width <= 0 || scaled_height <= 0 {
        return;
    }
    if dest_x + scaled_width <= 0 || dest_y + scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }

    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + scaled_height).min(dest_height as i32) as u32;
    if start_x >= end_x || start_y >= end_y {
        return;
    }

    // 1:1 fast path
    if scaled_width == src_width as i32 && scaled_height == src_height as i32 {
        let data_width = (src_width + 2 * border) as usize;
        let dest_stride = (dest_width * 4) as usize;
        let src_stride = data_width * 4;
        let copy_width = (end_x - start_x) as usize * 4;
        let src_x_offset = (start_x as i32 - dest_x) as usize * 4 + border as usize * 4;
        for y in start_y..end_y {
            let src_y = (y as i32 - dest_y) as usize + border as usize;
            let src_off = src_y * src_stride + src_x_offset;
            let dest_off = y as usize * dest_stride + start_x as usize * 4;
            if src_off + copy_width <= src.len() && dest_off + copy_width <= dest.len() {
                dest[dest_off..dest_off + copy_width]
                    .copy_from_slice(&src[src_off..src_off + copy_width]);
            }
        }
        return;
    }

    let data_width = src_width + 2 * border;
    let data_height = src_height + 2 * border;

    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (data_width * 4) as usize;
    let data_w = data_width as i32;
    let data_h = data_height as i32;
    let x_samples = compute_lanczos_col_samples(
        start_x,
        end_x,
        dest_x,
        rect.exact_x,
        rect.exact_width,
        scaled_width,
        src_width,
        border,
        data_w - 1,
    );
    let y_samples = compute_lanczos_row_samples(
        start_y,
        end_y,
        dest_y,
        rect.exact_y,
        rect.exact_height,
        scaled_height,
        src_height,
        border,
        data_h - 1,
        src_stride,
    );

    for (row_index, row_sample) in y_samples.iter().enumerate() {
        let y = start_y as usize + row_index;
        let dest_row = y * dest_stride;

        for (col_index, col_sample) in x_samples.iter().enumerate() {

            let mut r = 0.0_f64;
            let mut g = 0.0_f64;
            let mut bl = 0.0_f64;
            let mut aa = 0.0_f64;

            for (row_slot, row_offset) in row_sample.rows.iter().enumerate() {
                let wy = row_sample.weights[row_slot] as f64;
                for (col_slot, col_offset) in col_sample.offsets.iter().enumerate() {
                    let w = wy * col_sample.weights[col_slot] as f64;
                    let idx = row_offset + col_offset;
                    if idx + 3 < src.len() {
                        r += src[idx] as f64 * w;
                        g += src[idx + 1] as f64 * w;
                        bl += src[idx + 2] as f64 * w;
                        aa += src[idx + 3] as f64 * w;
                    }
                }
            }

            let dest_idx = dest_row + (start_x as usize + col_index) * 4;
            if dest_idx + 3 < dest.len() {
                dest[dest_idx] = r.clamp(0.0, 255.0) as u8;
                dest[dest_idx + 1] = g.clamp(0.0, 255.0) as u8;
                dest[dest_idx + 2] = bl.clamp(0.0, 255.0) as u8;
                dest[dest_idx + 3] = aa.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Alpha-blend two RGBA buffers: dest = fine * (1-blend) + coarse * blend
/// Both buffers must be the same size (width * height * 4 bytes).
#[allow(dead_code)]
pub fn blend_buffers(dest: &mut [u8], coarse: &[u8], blend: f64) {
    let b = (blend * 256.0).round() as u32;
    let inv_b = 256 - b;
    for (d, c) in dest.chunks_exact_mut(4).zip(coarse.chunks_exact(4)) {
        d[0] = ((d[0] as u32 * inv_b + c[0] as u32 * b) >> 8) as u8;
        d[1] = ((d[1] as u32 * inv_b + c[1] as u32 * b) >> 8) as u8;
        d[2] = ((d[2] as u32 * inv_b + c[2] as u32 * b) >> 8) as u8;
        d[3] = ((d[3] as u32 * inv_b + c[3] as u32 * b) >> 8) as u8;
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
    /// (u_min, v_min, u_max, v_max) in [0,1] normalised coordinates.
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Mip blend factor: 0.0 = 100 % fine, 1.0 = 100 % coarse.
    pub blend: f32,
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
    let BlitRect {
        x: dest_x,
        y: dest_y,
        width: scaled_width,
        height: scaled_height,
        ..
    } = rect;
    if scaled_width <= 0 || scaled_height <= 0 {
        return;
    }
    if dest_x + scaled_width <= 0 || dest_y + scaled_height <= 0 {
        return;
    }
    if dest_x >= dest_width as i32 || dest_y >= dest_height as i32 {
        return;
    }

    let start_x = dest_x.max(0) as u32;
    let start_y = dest_y.max(0) as u32;
    let end_x = (dest_x + scaled_width).min(dest_width as i32) as u32;
    let end_y = (dest_y + scaled_height).min(dest_height as i32) as u32;
    if start_x >= end_x || start_y >= end_y {
        return;
    }

    let blend_i = (coarse.blend * 256.0).round() as u32;
    let inv_blend_i = 256u32.saturating_sub(blend_i);

    // --- Fine tile setup ---
    let fine_data_w = fine_w + 2 * fine_border;
    let fine_data_h = fine_h + 2 * fine_border;
    let fine_dw1 = fine_data_w.saturating_sub(1);
    let fine_dh1 = fine_data_h.saturating_sub(1);
    let dest_stride = (dest_width * 4) as usize;
    let fine_stride = (fine_data_w * 4) as usize;

    let fine_max_idx = fine_dh1 as usize * fine_stride + fine_dw1 as usize * 4 + 3;
    if fine_src.len() <= fine_max_idx {
        return;
    }
    let dest_max_idx = (end_y - 1) as usize * dest_stride + (end_x - 1) as usize * 4 + 3;
    if dest.len() <= dest_max_idx {
        return;
    }

    // --- Coarse tile setup ---
    let coarse_src = coarse.data;
    let coarse_w = coarse.width;
    let coarse_h = coarse.height;
    let coarse_border = coarse.border;
    let coarse_data_w = coarse_w + 2 * coarse_border;
    let coarse_data_h = coarse_h + 2 * coarse_border;
    let coarse_dw1 = coarse_data_w.saturating_sub(1);
    let coarse_dh1 = coarse_data_h.saturating_sub(1);
    let coarse_stride = (coarse_data_w * 4) as usize;
    let coarse_max_idx = coarse_dh1 as usize * coarse_stride + coarse_dw1 as usize * 4 + 3;
    if coarse_src.len() <= coarse_max_idx {
        // Fall back to fine-only if coarse data bad
        blit_tile(dest, dest_width, dest_height, tile, rect);
        return;
    }

    // UV range in coarse tile pixels (content, excl. border)
    let cu_min = coarse.uv_min[0] as f64 * coarse_w as f64;
    let cv_min = coarse.uv_min[1] as f64 * coarse_h as f64;
    let cu_range = (coarse.uv_max[0] - coarse.uv_min[0]) as f64 * coarse_w as f64;
    let cv_range = (coarse.uv_max[1] - coarse.uv_min[1]) as f64 * coarse_h as f64;
    let fine_x_samples = compute_linear_col_samples(
        start_x,
        end_x,
        dest_x,
        rect.exact_x,
        rect.exact_width,
        scaled_width,
        fine_w,
        fine_border,
        fine_dw1,
    );
    let fine_y_samples = compute_linear_row_samples(
        start_y,
        end_y,
        dest_y,
        rect.exact_y,
        rect.exact_height,
        scaled_height,
        fine_h,
        fine_border,
        fine_dh1,
        fine_stride,
    );

    let coarse_x_samples = compute_linear_col_samples_subregion(
        start_x,
        end_x,
        dest_x,
        rect.exact_x,
        rect.exact_width,
        scaled_width,
        cu_min,
        cu_range,
        coarse_border,
        coarse_dw1,
    );
    let coarse_y_samples = compute_linear_row_samples_subregion(
        start_y,
        end_y,
        dest_y,
        rect.exact_y,
        rect.exact_height,
        scaled_height,
        cv_min,
        cv_range,
        coarse_border,
        coarse_dh1,
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
                    let fine_val = (fs00[c] as u32 * fw00
                        + fs10[c] as u32 * fw10
                        + fs01[c] as u32 * fw01
                        + fs11[c] as u32 * fw11)
                        >> 16;
                    let coarse_val = (cs00[c] as u32 * cw00
                        + cs10[c] as u32 * cw10
                        + cs01[c] as u32 * cw01
                        + cs11[c] as u32 * cw11)
                        >> 16;
                    d[c] = ((fine_val * inv_blend_i + coarse_val * blend_i) >> 8) as u8;
                }
            }
        }
    }
}

pub fn reproject_frame(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    src_viewport: &crate::render_pool::CachedCpuFrame,
    dest_viewport: &common::Viewport,
    clear_rgba: [u8; 4],
) {
    if dest_width == 0 || dest_height == 0 || src_width == 0 || src_height == 0 {
        return;
    }

    fast_fill_rgba(dest, clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3]);

    let src_bounds = src_viewport.viewport.bounds();
    let dest_bounds = dest_viewport.bounds();
    let src_stride = (src_width * 4) as usize;
    let dest_stride = (dest_width * 4) as usize;
    let max_x = src_width.saturating_sub(1) as f64;
    let max_y = src_height.saturating_sub(1) as f64;

    let x_samples: Vec<Option<LinearColSample>> = (0..dest_width)
        .map(|x| {
            let image_x = dest_bounds.left + x as f64 / dest_viewport.zoom;
            let src_x = (image_x - src_bounds.left) * src_viewport.viewport.zoom;
            if src_x < 0.0 || src_x > max_x {
                return None;
            }
            let src_x_fp = (src_x * 65536.0).max(0.0) as u32;
            let x0 = (src_x_fp >> 16).min(src_width.saturating_sub(1));
            let frac = (src_x_fp & 0xFFFF) >> 8;
            let x1 = (x0 + 1).min(src_width.saturating_sub(1));
            Some(LinearColSample {
                x0_4: x0 as usize * 4,
                x1_4: x1 as usize * 4,
                w0: 256 - frac,
                w1: frac,
            })
        })
        .collect();

    for y in 0..dest_height {
        let image_y = dest_bounds.top + y as f64 / dest_viewport.zoom;
        let src_y = (image_y - src_bounds.top) * src_viewport.viewport.zoom;
        if src_y < 0.0 || src_y > max_y {
            continue;
        }
        let src_y_fp = (src_y * 65536.0).max(0.0) as u32;
        let y0 = (src_y_fp >> 16).min(src_height.saturating_sub(1));
        let frac_y = (src_y_fp & 0xFFFF) >> 8;
        let y1 = (y0 + 1).min(src_height.saturating_sub(1));
        let row_sample = LinearRowSample {
            row0: y0 as usize * src_stride,
            row1: y1 as usize * src_stride,
            w0: 256 - frac_y,
            w1: frac_y,
        };
        let dest_row = y as usize * dest_stride;

        for (x, col_sample) in x_samples.iter().enumerate() {
            let Some(col_sample) = col_sample else {
                continue;
            };

            let w00 = col_sample.w0 * row_sample.w0;
            let w10 = col_sample.w1 * row_sample.w0;
            let w01 = col_sample.w0 * row_sample.w1;
            let w11 = col_sample.w1 * row_sample.w1;
            let dest_idx = dest_row + x * 4;

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

                d[0] = ((s00[0] as u32 * w00
                    + s10[0] as u32 * w10
                    + s01[0] as u32 * w01
                    + s11[0] as u32 * w11)
                    >> 16) as u8;
                d[1] = ((s00[1] as u32 * w00
                    + s10[1] as u32 * w10
                    + s01[1] as u32 * w01
                    + s11[1] as u32 * w11)
                    >> 16) as u8;
                d[2] = ((s00[2] as u32 * w00
                    + s10[2] as u32 * w10
                    + s01[2] as u32 * w01
                    + s11[2] as u32 * w11)
                    >> 16) as u8;
                d[3] = 255;
            }
        }
    }
}

/// Create a SharedPixelBuffer from raw RGBA data
pub fn create_image_buffer(
    data: &[u8],
    width: u32,
    height: u32,
) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let expected_len = (width * height * 4) as usize;
    if data.len() < expected_len {
        return None;
    }

    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    buffer
        .make_mut_bytes()
        .copy_from_slice(&data[..expected_len]);
    Some(buffer)
}
