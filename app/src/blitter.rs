//! Low-level tile blitting functions for CPU rendering
//!
//! This module contains optimized functions for copying and scaling tile data
//! to the viewport buffer.

use slint::{Rgba8Pixel, SharedPixelBuffer};

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
    let use_exact = rect.exact_width > 0.0;
    let exact_x = rect.exact_x;
    let exact_y = rect.exact_y;
    let exact_w = if use_exact { rect.exact_width } else { scaled_width as f64 };
    let exact_h = if use_exact { rect.exact_height } else { scaled_height as f64 };
    let fine_sx = fine_w as f64 / exact_w;
    let fine_sy = fine_h as f64 / exact_h;
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
    let coarse_sx = cu_range / exact_w;
    let coarse_sy = cv_range / exact_h;

    for y in start_y..end_y {
        // --- Fine Y ---
        let fyf = if use_exact {
            (y as f64 - exact_y) * fine_sy
        } else {
            (y as i32 - dest_y) as f64 * fine_sy
        };
        let fyf_fp = (fyf * 65536.0).max(0.0) as u32;
        let fy0i = fyf_fp >> 16;
        let ffy = (fyf_fp & 0xFFFF) >> 8;
        let ify = 256 - ffy;
        let fy0 = (fy0i + fine_border).min(fine_dh1);
        let fy1 = (fy0i + fine_border + 1).min(fine_dh1);
        let frow0 = fy0 as usize * fine_stride;
        let frow1 = fy1 as usize * fine_stride;

        // --- Coarse Y ---
        let cyf = if use_exact {
            cv_min + (y as f64 - exact_y) * coarse_sy
        } else {
            cv_min + (y as i32 - dest_y) as f64 * coarse_sy
        };
        let cyf_fp = (cyf * 65536.0).max(0.0) as u32;
        let cy0i = cyf_fp >> 16;
        let fcy = (cyf_fp & 0xFFFF) >> 8;
        let icy = 256 - fcy;
        let cy0 = (cy0i + coarse_border).min(coarse_dh1);
        let cy1 = (cy0i + coarse_border + 1).min(coarse_dh1);
        let crow0 = cy0 as usize * coarse_stride;
        let crow1 = cy1 as usize * coarse_stride;

        let dest_row = y as usize * dest_stride;

        for x in start_x..end_x {
            // --- Fine pixel ---
            let fxf = if use_exact {
                (x as f64 - exact_x) * fine_sx
            } else {
                (x as i32 - dest_x) as f64 * fine_sx
            };
            let fxf_fp = (fxf * 65536.0).max(0.0) as u32;
            let fx0i = fxf_fp >> 16;
            let ffx = (fxf_fp & 0xFFFF) >> 8;
            let ifx = 256 - ffx;
            let fx0 = (fx0i + fine_border).min(fine_dw1);
            let fx1 = (fx0i + fine_border + 1).min(fine_dw1);
            let fx04 = fx0 as usize * 4;
            let fx14 = fx1 as usize * 4;

            let fw00 = ifx * ify;
            let fw10 = ffx * ify;
            let fw01 = ifx * ffy;
            let fw11 = ffx * ffy;

            // --- Coarse pixel ---
            let cxf = if use_exact {
                cu_min + (x as f64 - exact_x) * coarse_sx
            } else {
                cu_min + (x as i32 - dest_x) as f64 * coarse_sx
            };
            let cxf_fp = (cxf * 65536.0).max(0.0) as u32;
            let cx0i = cxf_fp >> 16;
            let fcx = (cxf_fp & 0xFFFF) >> 8;
            let icx = 256 - fcx;
            let cx0 = (cx0i + coarse_border).min(coarse_dw1);
            let cx1 = (cx0i + coarse_border + 1).min(coarse_dw1);
            let cx04 = cx0 as usize * 4;
            let cx14 = cx1 as usize * 4;

            let cw00 = icx * icy;
            let cw10 = fcx * icy;
            let cw01 = icx * fcy;
            let cw11 = fcx * fcy;

            let dest_idx = dest_row + x as usize * 4;

            unsafe {
                let fs00 = fine_src.get_unchecked(frow0 + fx04..frow0 + fx04 + 4);
                let fs10 = fine_src.get_unchecked(frow0 + fx14..frow0 + fx14 + 4);
                let fs01 = fine_src.get_unchecked(frow1 + fx04..frow1 + fx04 + 4);
                let fs11 = fine_src.get_unchecked(frow1 + fx14..frow1 + fx14 + 4);

                let cs00 = coarse_src.get_unchecked(crow0 + cx04..crow0 + cx04 + 4);
                let cs10 = coarse_src.get_unchecked(crow0 + cx14..crow0 + cx14 + 4);
                let cs01 = coarse_src.get_unchecked(crow1 + cx04..crow1 + cx04 + 4);
                let cs11 = coarse_src.get_unchecked(crow1 + cx14..crow1 + cx14 + 4);

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
