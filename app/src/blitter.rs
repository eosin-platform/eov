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
/// Uses fixed-point arithmetic and minimizes bounds checking
#[inline(always)]
pub fn blit_tile(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    rect: BlitRect,
) {
    let BlitRect {
        x: dest_x,
        y: dest_y,
        width: scaled_width,
        height: scaled_height,
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
        let dest_stride = (dest_width * 4) as usize;
        let src_stride = (src_width * 4) as usize;
        let copy_width = (end_x - start_x) as usize * 4;
        let src_x_offset = (start_x as i32 - dest_x) as usize * 4;
        for y in start_y..end_y {
            let src_y = (y as i32 - dest_y) as usize;
            let src_off = src_y * src_stride + src_x_offset;
            let dest_off = y as usize * dest_stride + start_x as usize * 4;
            if src_off + copy_width <= src.len() && dest_off + copy_width <= dest.len() {
                dest[dest_off..dest_off + copy_width]
                    .copy_from_slice(&src[src_off..src_off + copy_width]);
            }
        }
        return;
    }

    let scale_x_fp = ((src_width as u64) << 16) / (scaled_width as u64).max(1);
    let scale_y_fp = ((src_height as u64) << 16) / (scaled_height as u64).max(1);

    let src_width_minus_1 = src_width.saturating_sub(1);
    let src_height_minus_1 = src_height.saturating_sub(1);
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (src_width * 4) as usize;

    let src_max_idx = src_height_minus_1 as usize * src_stride + src_width_minus_1 as usize * 4 + 3;
    if src.len() <= src_max_idx {
        return;
    }

    let dest_max_idx = (end_y - 1) as usize * dest_stride + (end_x - 1) as usize * 4 + 3;
    if dest.len() <= dest_max_idx {
        return;
    }

    for y in start_y..end_y {
        let local_y = (y as i32 - dest_y) as u64;
        let src_y_fp = (local_y * scale_y_fp) as u32;
        let y0 = (src_y_fp >> 16).min(src_height_minus_1);
        let y1 = (y0 + 1).min(src_height_minus_1);
        let fy = (src_y_fp & 0xFFFF) >> 8;
        let inv_fy = 256 - fy;

        let dest_row = y as usize * dest_stride;
        let src_row0 = y0 as usize * src_stride;
        let src_row1 = y1 as usize * src_stride;

        for x in start_x..end_x {
            let local_x = (x as i32 - dest_x) as u64;
            let src_x_fp = (local_x * scale_x_fp) as u32;
            let x0 = (src_x_fp >> 16).min(src_width_minus_1);
            let x1 = (x0 + 1).min(src_width_minus_1);
            let fx = (src_x_fp & 0xFFFF) >> 8;
            let inv_fx = 256 - fx;

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
    rect: BlitRect,
) {
    let BlitRect {
        x: dest_x,
        y: dest_y,
        width: scaled_width,
        height: scaled_height,
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
        let dest_stride = (dest_width * 4) as usize;
        let src_stride = (src_width * 4) as usize;
        let copy_width = (end_x - start_x) as usize * 4;
        let src_x_offset = (start_x as i32 - dest_x) as usize * 4;
        for y in start_y..end_y {
            let src_y = (y as i32 - dest_y) as usize;
            let src_off = src_y * src_stride + src_x_offset;
            let dest_off = y as usize * dest_stride + start_x as usize * 4;
            if src_off + copy_width <= src.len() && dest_off + copy_width <= dest.len() {
                dest[dest_off..dest_off + copy_width]
                    .copy_from_slice(&src[src_off..src_off + copy_width]);
            }
        }
        return;
    }

    let scale_x = src_width as f64 / scaled_width.max(1) as f64;
    let scale_y = src_height as f64 / scaled_height.max(1) as f64;
    let dest_stride = (dest_width * 4) as usize;
    let src_stride = (src_width * 4) as usize;
    let src_w = src_width as i32;
    let src_h = src_height as i32;
    let a = 3.0_f64;

    for y in start_y..end_y {
        let local_y = (y as i32 - dest_y) as f64;
        let src_yf = local_y * scale_y;
        let center_y = src_yf.floor() as i32;
        let frac_y = src_yf - center_y as f64;

        let dest_row = y as usize * dest_stride;

        for x in start_x..end_x {
            let local_x = (x as i32 - dest_x) as f64;
            let src_xf = local_x * scale_x;
            let center_x = src_xf.floor() as i32;
            let frac_x = src_xf - center_x as f64;

            let mut r = 0.0_f64;
            let mut g = 0.0_f64;
            let mut b = 0.0_f64;
            let mut aa = 0.0_f64;
            let mut w_sum = 0.0_f64;

            for j in -2..=3_i32 {
                let sy = (center_y + j).clamp(0, src_h - 1) as usize;
                let wy = lanczos_weight(j as f64 - frac_y, a);
                let row = sy * src_stride;
                for i in -2..=3_i32 {
                    let sx = (center_x + i).clamp(0, src_w - 1) as usize;
                    let wx = lanczos_weight(i as f64 - frac_x, a);
                    let w = wx * wy;
                    let idx = row + sx * 4;
                    if idx + 3 < src.len() {
                        r += src[idx] as f64 * w;
                        g += src[idx + 1] as f64 * w;
                        b += src[idx + 2] as f64 * w;
                        aa += src[idx + 3] as f64 * w;
                        w_sum += w;
                    }
                }
            }

            let dest_idx = dest_row + x as usize * 4;
            if dest_idx + 3 < dest.len() && w_sum.abs() > 1e-8 {
                dest[dest_idx] = (r / w_sum).clamp(0.0, 255.0) as u8;
                dest[dest_idx + 1] = (g / w_sum).clamp(0.0, 255.0) as u8;
                dest[dest_idx + 2] = (b / w_sum).clamp(0.0, 255.0) as u8;
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
