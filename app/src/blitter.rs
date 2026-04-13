//! Low-level tile blitting functions for CPU rendering
//!
//! This module contains optimized functions for copying and scaling tile data
//! to the viewport buffer.

use slint::{Rgba8Pixel, SharedPixelBuffer};

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
    src: &[u8],
    src_width: u32,
    src_height: u32,
    border: u32,
    rect: BlitRect,
) {
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
    let use_exact = rect.exact_width > 0.0;
    let exact_x = rect.exact_x;
    let exact_y = rect.exact_y;
    let exact_w = if use_exact {
        rect.exact_width
    } else {
        scaled_width as f64
    };
    let exact_h = if use_exact {
        rect.exact_height
    } else {
        scaled_height as f64
    };
    let scale_x_f = src_width as f64 / exact_w;
    let scale_y_f = src_height as f64 / exact_h;

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

    for y in start_y..end_y {
        // Map dest pixel to source coord using exact float origin/size.
        // This produces the same source coordinate for a given dest pixel
        // regardless of integer rect rounding: src = (dest_px - exact_origin) * scale
        let src_yf = if use_exact {
            (y as f64 - exact_y) * scale_y_f
        } else {
            (y as i32 - dest_y) as f64 * scale_y_f
        };
        let src_y_fp = (src_yf * 65536.0).max(0.0) as u32;
        let y0_inner = src_y_fp >> 16;
        let fy = (src_y_fp & 0xFFFF) >> 8;
        let inv_fy = 256 - fy;
        // Offset by border into the padded data; clamp to data bounds.
        let y0 = (y0_inner + border).min(data_h_minus_1);
        let y1 = (y0_inner + border + 1).min(data_h_minus_1);

        let dest_row = y as usize * dest_stride;
        let src_row0 = y0 as usize * src_stride;
        let src_row1 = y1 as usize * src_stride;

        for x in start_x..end_x {
            let src_xf = if use_exact {
                (x as f64 - exact_x) * scale_x_f
            } else {
                (x as i32 - dest_x) as f64 * scale_x_f
            };
            let src_x_fp = (src_xf * 65536.0).max(0.0) as u32;
            let x0_inner = src_x_fp >> 16;
            let fx = (src_x_fp & 0xFFFF) >> 8;
            let inv_fx = 256 - fx;
            let x0 = (x0_inner + border).min(data_w_minus_1);
            let x1 = (x0_inner + border + 1).min(data_w_minus_1);

            let x0_4 = x0 as usize * 4;
            let x1_4 = x1 as usize * 4;
            let dest_idx = dest_row + x as usize * 4;

            let w00 = inv_fx * inv_fy;
            let w10 = fx * inv_fy;
            let w01 = inv_fx * fy;
            let w11 = fx * fy;

            unsafe {
                let s00 = src.get_unchecked(src_row0 + x0_4..src_row0 + x0_4 + 4);
                let s10 = src.get_unchecked(src_row0 + x1_4..src_row0 + x1_4 + 4);
                let s01 = src.get_unchecked(src_row1 + x0_4..src_row1 + x0_4 + 4);
                let s11 = src.get_unchecked(src_row1 + x1_4..src_row1 + x1_4 + 4);
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
    src: &[u8],
    src_width: u32,
    src_height: u32,
    border: u32,
    rect: BlitRect,
) {
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

    let use_exact = rect.exact_width > 0.0;
    let exact_x = rect.exact_x;
    let exact_y = rect.exact_y;
    let exact_w = if use_exact {
        rect.exact_width
    } else {
        scaled_width.max(1) as f64
    };
    let exact_h = if use_exact {
        rect.exact_height
    } else {
        scaled_height.max(1) as f64
    };
    let scale_x = src_width as f64 / exact_w;
    let scale_y = src_height as f64 / exact_h;
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (data_width * 4) as usize;
    let data_w = data_width as i32;
    let data_h = data_height as i32;
    let b = border as i32;
    let a = 3.0_f64;

    for y in start_y..end_y {
        let src_yf = if use_exact {
            (y as f64 - exact_y) * scale_y
        } else {
            (y as i32 - dest_y) as f64 * scale_y
        };
        let src_yf = src_yf.max(0.0);
        let center_y = src_yf.floor() as i32;
        let frac_y = src_yf - center_y as f64;

        let dest_row = y as usize * dest_stride;

        for x in start_x..end_x {
            let src_xf = if use_exact {
                (x as f64 - exact_x) * scale_x
            } else {
                (x as i32 - dest_x) as f64 * scale_x
            };
            let src_xf = src_xf.max(0.0);
            let center_x = src_xf.floor() as i32;
            let frac_x = src_xf - center_x as f64;

            let mut r = 0.0_f64;
            let mut g = 0.0_f64;
            let mut bl = 0.0_f64;
            let mut aa = 0.0_f64;
            let mut w_sum = 0.0_f64;

            for j in -2..=3_i32 {
                let sy = (center_y + j + b).clamp(0, data_h - 1) as usize;
                let wy = lanczos_weight(j as f64 - frac_y, a);
                let row = sy * src_stride;
                for i in -2..=3_i32 {
                    let sx = (center_x + i + b).clamp(0, data_w - 1) as usize;
                    let wx = lanczos_weight(i as f64 - frac_x, a);
                    let w = wx * wy;
                    let idx = row + sx * 4;
                    if idx + 3 < src.len() {
                        r += src[idx] as f64 * w;
                        g += src[idx + 1] as f64 * w;
                        bl += src[idx + 2] as f64 * w;
                        aa += src[idx + 3] as f64 * w;
                        w_sum += w;
                    }
                }
            }

            let dest_idx = dest_row + x as usize * 4;
            if dest_idx + 3 < dest.len() && w_sum.abs() > 1e-8 {
                dest[dest_idx] = (r / w_sum).clamp(0.0, 255.0) as u8;
                dest[dest_idx + 1] = (g / w_sum).clamp(0.0, 255.0) as u8;
                dest[dest_idx + 2] = (bl / w_sum).clamp(0.0, 255.0) as u8;
                dest[dest_idx + 3] = (aa / w_sum).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Alpha-blend two RGBA buffers: dest = fine * (1-blend) + coarse * blend
/// Both buffers must be the same size (width * height * 4 bytes).
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
