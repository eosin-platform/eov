//! Blitting into frames turned by angles other than quarter turns.
//!
//! Quarter turns are drawn axis-aligned with the slide and permuted onto
//! the screen ([`orient_frame`](super::orient_frame), exact). Any other
//! angle draws every tile straight into the turned frame: each destination
//! pixel whose centre falls inside the tile (half-open, the rule GPU
//! rasterisers use for the tile's turned quad) samples the tile at the
//! position its centre maps to, with the same pixel-centre convention,
//! taps and rounding as the axis-aligned samplers. The slide is resampled
//! once, as on the GPU, and the output matches the GPU's within the
//! rounding of its interpolators.
//!
//! The blits keep their layout in the frame drawn axis-aligned with the
//! slide (`Viewport::aligned`, which is the turned view's bounding box); a
//! [`FrameMap`] says where each destination pixel falls in it.

use super::{
    BlitRect, CoarseSrc, LANCZOS_A, TileSrc, dest_fits, fast_fill_rgba, lanczos_weight,
    linear_taps, round_shift16, tile_data_fits,
};
use crate::Viewport;
use glam::DVec2;
use rayon::prelude::*;
use std::sync::OnceLock;

/// An affine map from the pixels of a destination frame into the
/// continuous pixel coordinates of another frame (pixel `i` spans
/// `i..i + 1`): the centre of destination pixel `(x, y)` lies at
/// `origin + x · step_x + y · step_y`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameMap {
    pub origin: [f64; 2],
    pub step_x: [f64; 2],
    pub step_y: [f64; 2],
}

impl FrameMap {
    /// Where the pixels of `viewport`'s turned frame (`width × height`, the
    /// screen) fall in its aligned layout: the frame drawn axis-aligned
    /// with the slide, its top-left corner at `viewport.bounds()`'s, one
    /// pixel per screen pixel. Tile rects computed against
    /// `viewport.aligned()` are in that layout.
    pub fn aligned_layout(viewport: &Viewport) -> Self {
        let orientation = viewport.orientation;
        let (aligned_width, aligned_height) =
            orientation.turned_extent(viewport.width, viewport.height);
        let half = DVec2::new(aligned_width / 2.0, aligned_height / 2.0);
        let screen_half = DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        let origin = half + orientation.to_image(DVec2::new(0.5, 0.5) - screen_half);
        let step_x = orientation.to_image(DVec2::new(1.0, 0.0));
        let step_y = orientation.to_image(DVec2::new(0.0, 1.0));
        Self {
            origin: origin.to_array(),
            step_x: step_x.to_array(),
            step_y: step_y.to_array(),
        }
    }

    /// The map `position(x, y)` gives for destination pixel centres
    /// `(x, y)` (affine; sampled at the corners of a `width × height`
    /// frame for precision).
    pub fn sampled(width: u32, height: u32, position: impl Fn(f64, f64) -> DVec2) -> Self {
        let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
        let origin = position(0.5, 0.5);
        let step_x = (position(w + 0.5, 0.5) - origin) / w;
        let step_y = (position(0.5, h + 0.5) - origin) / h;
        Self {
            origin: origin.to_array(),
            step_x: step_x.to_array(),
            step_y: step_y.to_array(),
        }
    }

    /// The same map for the frame's rows from `row` on (a band of it).
    pub fn from_row(self, row: u32) -> Self {
        let row = f64::from(row);
        Self {
            origin: [
                self.origin[0] + row * self.step_y[0],
                self.origin[1] + row * self.step_y[1],
            ],
            ..self
        }
    }

    /// The position of destination pixel `(x, y)`'s centre.
    #[inline(always)]
    pub fn at(&self, x: f64, y: f64) -> [f64; 2] {
        [
            self.origin[0] + x * self.step_x[0] + y * self.step_y[0],
            self.origin[1] + x * self.step_x[1] + y * self.step_y[1],
        ]
    }

    /// The destination point (continuous, not a pixel centre) that maps to
    /// `p`, or `None` for a degenerate map.
    fn inverse_at(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let [a, c] = self.step_x;
        let [b, d] = self.step_y;
        let det = a * d - b * c;
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        // p − origin = x·step_x + y·step_y, for pixel centres (x + 0.5).
        let (px, py) = (p[0] - self.origin[0], p[1] - self.origin[1]);
        Some([(d * px - b * py) / det + 0.5, (a * py - c * px) / det + 0.5])
    }
}

/// How far (aligned pixels) a tile's edges reach out, so that tiles that
/// share an edge leave no pixel uncovered between them whatever the
/// rounding of their exact rects.
const EDGE_SLACK: f64 = 1e-6;

/// The tile's half-open rectangle in the aligned layout:
/// `(left, top, right, bottom)`.
fn rect_extent(rect: &BlitRect) -> (f64, f64, f64, f64) {
    if rect.exact_width > 0.0 && rect.exact_height > 0.0 {
        (
            rect.exact_x,
            rect.exact_y,
            rect.exact_x + rect.exact_width,
            rect.exact_y + rect.exact_height,
        )
    } else {
        (
            f64::from(rect.x),
            f64::from(rect.y),
            f64::from(rect.x) + f64::from(rect.width.max(0)),
            f64::from(rect.y) + f64::from(rect.height.max(0)),
        )
    }
}

/// Columns `x` (real) where `base + x · slope` lies in `lo..hi`.
fn axis_span(base: f64, slope: f64, lo: f64, hi: f64) -> (f64, f64) {
    if slope.abs() < 1e-15 {
        if base >= lo && base < hi {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (f64::INFINITY, f64::NEG_INFINITY)
        }
    } else {
        let (a, b) = ((lo - base) / slope, (hi - base) / slope);
        (a.min(b), a.max(b))
    }
}

/// Calls `sample(pixel, position)` for every pixel of `dest` (`dest_width`
/// × `dest_height`, tightly packed RGBA) whose centre maps (through `map`)
/// into `rect`, with that position in the aligned layout.
fn for_each_covered_pixel(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    rect: &BlitRect,
    map: &FrameMap,
    mut sample: impl FnMut(&mut [u8; 4], [f64; 2]),
) {
    let (left, top, right, bottom) = rect_extent(rect);
    if !(right > left && bottom > top) || dest_width == 0 || dest_height == 0 {
        return;
    }
    let (left, top) = (left - EDGE_SLACK, top - EDGE_SLACK);
    let (right, bottom) = (right + EDGE_SLACK, bottom + EDGE_SLACK);
    // Destination rows the rect can touch: its corners mapped back.
    let Some(corners) = [[left, top], [right, top], [right, bottom], [left, bottom]]
        .iter()
        .map(|&p| map.inverse_at(p))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let min_y = corners.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = corners
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let first_row = (min_y.floor() - 1.0).max(0.0);
    let last_row = (max_y.ceil() + 1.0).min(f64::from(dest_height));
    if first_row >= last_row {
        return;
    }
    if !dest_fits(dest, dest_width, last_row as u32) {
        return;
    }
    let stride = dest_width as usize * 4;
    for y in first_row as u32..last_row as u32 {
        let base = map.at(0.0, f64::from(y));
        let (ax0, ax1) = axis_span(base[0], map.step_x[0], left, right);
        let (ay0, ay1) = axis_span(base[1], map.step_x[1], top, bottom);
        let (lo, hi) = (ax0.max(ay0), ax1.min(ay1));
        if lo > hi {
            continue;
        }
        let x_start = (lo.floor() - 1.0).max(0.0);
        let x_end = (hi.ceil() + 1.0).min(f64::from(dest_width));
        if x_start >= x_end {
            continue;
        }
        let row = &mut dest[y as usize * stride..(y as usize + 1) * stride];
        let pixels = row.as_chunks_mut::<4>().0;
        for x in x_start as u32..x_end as u32 {
            let xf = f64::from(x);
            let p = [base[0] + xf * map.step_x[0], base[1] + xf * map.step_x[1]];
            // The exact test: the span above only bounds it.
            if p[0] >= left && p[0] < right && p[1] >= top && p[1] < bottom {
                sample(&mut pixels[x as usize], p);
            }
        }
    }
}

/// Maps aligned positions to source data positions (pixel-centre
/// convention, border included) of one blit.
#[derive(Clone, Copy)]
struct SourceMap {
    scale: [f64; 2],
    offset: [f64; 2],
}

impl SourceMap {
    /// The tile's inner area `src_offset .. src_offset + src_range` (source
    /// pixels, per axis) over the rect, with `border` pixels around it.
    fn new(rect: &BlitRect, src_offset: [f64; 2], src_range: [f64; 2], border: u32) -> Self {
        let (left, top, right, bottom) = rect_extent(rect);
        let scale = [
            src_range[0] / (right - left).max(1e-12),
            src_range[1] / (bottom - top).max(1e-12),
        ];
        let border = f64::from(border);
        Self {
            scale,
            offset: [
                src_offset[0] - left * scale[0] - 0.5 + border,
                src_offset[1] - top * scale[1] - 0.5 + border,
            ],
        }
    }

    #[inline(always)]
    fn at(&self, p: [f64; 2]) -> [f64; 2] {
        [
            p[0] * self.scale[0] + self.offset[0],
            p[1] * self.scale[1] + self.offset[1],
        ]
    }
}

/// Bilinear taps at data position `s`: the four texel offsets (bytes) and
/// their 16-bit weights.
#[inline(always)]
fn bilinear_taps(s: [f64; 2], max_x: u32, max_y: u32, stride: usize) -> ([usize; 4], [u32; 4]) {
    let (x0, x1, fx) = linear_taps(s[0], max_x);
    let (y0, y1, fy) = linear_taps(s[1], max_y);
    let (wx0, wx1, wy0, wy1) = (256 - fx, fx, 256 - fy, fy);
    let (r0, r1) = (y0 as usize * stride, y1 as usize * stride);
    let (c0, c1) = (x0 as usize * 4, x1 as usize * 4);
    (
        [r0 + c0, r0 + c1, r1 + c0, r1 + c1],
        [wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1],
    )
}

/// Weighted sum of the taps of one channel (at most 255 · 2^16).
#[inline(always)]
fn weighted(data: &[u8], taps: &[usize; 4], weights: &[u32; 4], channel: usize) -> u32 {
    u32::from(data[taps[0] + channel]) * weights[0]
        + u32::from(data[taps[1] + channel]) * weights[1]
        + u32::from(data[taps[2] + channel]) * weights[2]
        + u32::from(data[taps[3] + channel]) * weights[3]
}

/// Bilinear blit into a turned frame (see the module docs).
pub fn blit_tile_turned(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
    map: &FrameMap,
) {
    if !tile_data_fits(tile.data, tile.width, tile.height, tile.border) {
        return;
    }
    let data_width = tile.width + 2 * tile.border;
    let data_height = tile.height + 2 * tile.border;
    let stride = data_width as usize * 4;
    let source = SourceMap::new(
        &rect,
        [0.0, 0.0],
        [f64::from(tile.width), f64::from(tile.height)],
        tile.border,
    );
    let data = tile.data;
    for_each_covered_pixel(dest, dest_width, dest_height, &rect, map, |pixel, p| {
        let (taps, weights) = bilinear_taps(source.at(p), data_width - 1, data_height - 1, stride);
        for (channel, value) in pixel.iter_mut().enumerate() {
            *value = round_shift16(weighted(data, &taps, &weights, channel));
        }
    });
}

/// Nearest-neighbour blit into a turned frame: every pixel takes the
/// source pixel its centre falls in (the pixel view; squares turned with
/// the slide).
pub fn blit_tile_nearest_turned(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
    map: &FrameMap,
) {
    if !tile_data_fits(tile.data, tile.width, tile.height, tile.border) {
        return;
    }
    let data_width = tile.width + 2 * tile.border;
    let data_height = tile.height + 2 * tile.border;
    let stride = data_width as usize * 4;
    let source = SourceMap::new(
        &rect,
        [0.0, 0.0],
        [f64::from(tile.width), f64::from(tile.height)],
        tile.border,
    );
    let data = tile.data;
    let index = |position: f64, max: u32| -> usize {
        let i = (position + 0.5).floor();
        let i = if i.is_finite() { i as i64 } else { 0 };
        i.clamp(0, i64::from(max)) as usize
    };
    for_each_covered_pixel(dest, dest_width, dest_height, &rect, map, |pixel, p| {
        let s = source.at(p);
        let at = index(s[1], data_height - 1) * stride + index(s[0], data_width - 1) * 4;
        pixel.copy_from_slice(&data[at..at + 4]);
    });
}

/// Fused bilinear + trilinear blit into a turned frame: the fine and the
/// coarse tile sampled per pixel and blended, as
/// [`blit_tile_trilinear`](super::blit_tile_trilinear) does.
pub fn blit_tile_trilinear_turned(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    coarse: &CoarseSrc,
    rect: BlitRect,
    map: &FrameMap,
) {
    if !tile_data_fits(tile.data, tile.width, tile.height, tile.border) {
        return;
    }
    if !tile_data_fits(coarse.data, coarse.width, coarse.height, coarse.border) {
        blit_tile_turned(dest, dest_width, dest_height, tile, rect, map);
        return;
    }
    let blend = if coarse.blend.is_nan() {
        0.0
    } else {
        coarse.blend.clamp(0.0, 1.0)
    };
    let blend = (blend * 256.0).round() as u32;
    let inv_blend = 256 - blend;
    let fine_w = tile.width + 2 * tile.border;
    let fine_h = tile.height + 2 * tile.border;
    let fine_stride = fine_w as usize * 4;
    let fine_source = SourceMap::new(
        &rect,
        [0.0, 0.0],
        [f64::from(tile.width), f64::from(tile.height)],
        tile.border,
    );
    let coarse_w = coarse.width + 2 * coarse.border;
    let coarse_h = coarse.height + 2 * coarse.border;
    let coarse_stride = coarse_w as usize * 4;
    let (cw, ch) = (f64::from(coarse.width), f64::from(coarse.height));
    let coarse_source = SourceMap::new(
        &rect,
        [
            f64::from(coarse.uv_min[0]) * cw,
            f64::from(coarse.uv_min[1]) * ch,
        ],
        [
            f64::from(coarse.uv_max[0] - coarse.uv_min[0]) * cw,
            f64::from(coarse.uv_max[1] - coarse.uv_min[1]) * ch,
        ],
        coarse.border,
    );
    let (fine, coarse_data) = (tile.data, coarse.data);
    for_each_covered_pixel(dest, dest_width, dest_height, &rect, map, |pixel, p| {
        let (fine_taps, fine_weights) =
            bilinear_taps(fine_source.at(p), fine_w - 1, fine_h - 1, fine_stride);
        let (coarse_taps, coarse_weights) = bilinear_taps(
            coarse_source.at(p),
            coarse_w - 1,
            coarse_h - 1,
            coarse_stride,
        );
        for (channel, value) in pixel.iter_mut().enumerate() {
            let f = weighted(fine, &fine_taps, &fine_weights, channel);
            let c = weighted(coarse_data, &coarse_taps, &coarse_weights, channel);
            *value = ((f * inv_blend + c * blend + (1 << 23)) >> 24) as u8;
        }
    });
}

/// Samples of the Lanczos-3 kernel per unit of its argument.
const LANCZOS_TABLE_STEPS: usize = 1024;

/// The Lanczos-3 kernel at `0, 1/STEPS, …, 3`, for the turned sampler's
/// many taps.
fn lanczos_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=LANCZOS_TABLE_STEPS * LANCZOS_A as usize + 1)
            .map(|i| lanczos_weight(i as f64 / LANCZOS_TABLE_STEPS as f64, LANCZOS_A) as f32)
            .collect()
    })
}

/// The kernel at `x` (interpolated from the table).
#[inline(always)]
fn lanczos_at(table: &[f32], x: f64) -> f32 {
    let t = x.abs() * LANCZOS_TABLE_STEPS as f64;
    let i = t.floor();
    let index = i as usize;
    if index + 1 >= table.len() {
        return 0.0;
    }
    let frac = (t - i) as f32;
    table[index] + (table[index + 1] - table[index]) * frac
}

/// The taps of one axis at data position `center`: the first index and the
/// weights of consecutive indices (normalised, edge taps merged into the
/// clamped index).
#[inline(always)]
fn lanczos_taps(
    table: &[f32],
    center: f64,
    stretch: f64,
    max_index: u32,
    weights: &mut Vec<f32>,
) -> u32 {
    let support = LANCZOS_A * stretch;
    let lo = (center - support).floor() as i64 + 1;
    let hi = ((center + support).ceil() as i64 - 1).max(lo);
    let max = i64::from(max_index);
    let first = lo.clamp(0, max);
    let last = hi.clamp(0, max);
    weights.clear();
    weights.resize((last - first + 1) as usize, 0.0);
    let mut sum = 0.0f32;
    for i in lo..=hi {
        let w = lanczos_at(table, (i as f64 - center) / stretch);
        weights[(i.clamp(0, max) - first) as usize] += w;
        sum += w;
    }
    if sum.abs() > 1e-6 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    first as u32
}

/// Lanczos-3 blit into a turned frame: the kernel is laid along the
/// source's axes at each pixel's position (as the GPU's Lanczos shader
/// does) and, like [`blit_tile_lanczos3`](super::blit_tile_lanczos3),
/// widened when the tile is minified. Tiles need
/// [`lanczos_border_for_scale`](super::lanczos_border_for_scale) border
/// pixels to be seamless.
pub fn blit_tile_lanczos3_turned(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    tile: TileSrc,
    rect: BlitRect,
    map: &FrameMap,
) {
    if !tile_data_fits(tile.data, tile.width, tile.height, tile.border) {
        return;
    }
    let data_width = tile.width + 2 * tile.border;
    let data_height = tile.height + 2 * tile.border;
    let stride = data_width as usize * 4;
    let source = SourceMap::new(
        &rect,
        [0.0, 0.0],
        [f64::from(tile.width), f64::from(tile.height)],
        tile.border,
    );
    // Source pixels per destination pixel (the same along both axes for a
    // turn; the larger if they differ).
    let step = |v: [f64; 2]| DVec2::new(v[0] * source.scale[0], v[1] * source.scale[1]).length();
    let scale = step(map.step_x).max(step(map.step_y));
    let stretch = if scale.is_finite() {
        scale.max(1.0)
    } else {
        1.0
    };
    let table = lanczos_table();
    let data = tile.data;
    let (mut wx, mut wy) = (Vec::new(), Vec::new());
    for_each_covered_pixel(dest, dest_width, dest_height, &rect, map, |pixel, p| {
        let s = source.at(p);
        let first_x = lanczos_taps(table, s[0], stretch, data_width - 1, &mut wx) as usize;
        let first_y = lanczos_taps(table, s[1], stretch, data_height - 1, &mut wy) as usize;
        let mut acc = [0f32; 4];
        for (j, &w_row) in wy.iter().enumerate() {
            if w_row == 0.0 {
                continue;
            }
            let row = &data[(first_y + j) * stride + first_x * 4..];
            let mut line = [0f32; 4];
            for (i, &w) in wx.iter().enumerate() {
                let texel = &row[i * 4..i * 4 + 4];
                line[0] += f32::from(texel[0]) * w;
                line[1] += f32::from(texel[1]) * w;
                line[2] += f32::from(texel[2]) * w;
                line[3] += f32::from(texel[3]) * w;
            }
            for c in 0..4 {
                acc[c] += line[c] * w_row;
            }
        }
        for c in 0..4 {
            pixel[c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
        }
    });
}

/// Bilinearly resamples the frame `src` (`src_width × src_height`) into
/// `dest`, where `map` gives each destination pixel's position in `src`;
/// pixels whose centre falls outside `src` get `clear_rgba`. The turned
/// counterpart of [`try_reproject_frame`](super::try_reproject_frame) (same
/// pixel-centre convention), for any change of view.
#[allow(clippy::too_many_arguments)]
pub fn reproject_frame_mapped(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    map: &FrameMap,
    clear_rgba: [u8; 4],
) {
    let needed = dest_width as usize * dest_height as usize * 4;
    if dest.len() < needed || needed == 0 {
        return;
    }
    let dest = &mut dest[..needed];
    let [r, g, b, a] = clear_rgba;
    fast_fill_rgba(dest, r, g, b, a);
    if src_width == 0
        || src_height == 0
        || (src.len() as u64) < u64::from(src_width) * u64::from(src_height) * 4
    {
        return;
    }
    let (w, h) = (f64::from(src_width), f64::from(src_height));
    let stride = src_width as usize * 4;
    dest.par_chunks_mut(dest_width as usize * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let base = map.at(0.0, y as f64);
            for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let xf = x as f64;
                let p = [base[0] + xf * map.step_x[0], base[1] + xf * map.step_x[1]];
                if !(p[0] >= 0.0 && p[0] < w && p[1] >= 0.0 && p[1] < h) {
                    continue;
                }
                let (taps, weights) = bilinear_taps(
                    [p[0] - 0.5, p[1] - 0.5],
                    src_width - 1,
                    src_height - 1,
                    stride,
                );
                for (channel, value) in pixel.iter_mut().enumerate() {
                    *value = round_shift16(weighted(src, &taps, &weights, channel));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::super::{blit_tile, orient_frame};
    use super::*;
    use crate::viewport::Orientation;

    /// A bordered tile from a grey-level function of inner coordinates (the
    /// border repeats the edge).
    fn tile(width: u32, height: u32, border: u32, grey: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let (dw, dh) = (width + 2 * border, height + 2 * border);
        let mut data = Vec::with_capacity((dw * dh * 4) as usize);
        for y in 0..dh {
            for x in 0..dw {
                let ix = x.saturating_sub(border).min(width - 1);
                let iy = y.saturating_sub(border).min(height - 1);
                let v = grey(ix, iy);
                data.extend_from_slice(&[v, v / 2, 255 - v, 255]);
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

    fn src(data: &[u8], width: u32, height: u32, border: u32) -> TileSrc<'_> {
        TileSrc {
            data,
            width,
            height,
            border,
        }
    }

    /// A view of `width × height` at `degrees` and its map.
    fn view(width: f64, height: f64, degrees: f64) -> (Viewport, FrameMap) {
        let mut viewport = Viewport::new(width, height, 1000.0, 1000.0);
        viewport.zoom = 1.0;
        viewport.center = DVec2::new(500.0, 500.0);
        viewport.orientation = Orientation::from_degrees(degrees, false).unwrap();
        let map = FrameMap::aligned_layout(&viewport);
        (viewport, map)
    }

    fn max_difference(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    }

    /// The map agrees with the viewport: a pixel centre's aligned position
    /// is its level-0 point relative to the bounds, times the zoom.
    #[test]
    fn maps_follow_the_viewport() {
        for degrees in [0.0, 30.0, 90.0, 213.5] {
            let (viewport, map) = view(64.0, 40.0, degrees);
            let bounds = viewport.bounds();
            for (x, y) in [(0.0, 0.0), (63.0, 0.0), (10.0, 39.0)] {
                let p = viewport.screen_to_image(x + 0.5, y + 0.5);
                let expected = [
                    (p.x - bounds.left) * viewport.zoom,
                    (p.y - bounds.top) * viewport.zoom,
                ];
                let got = map.at(x, y);
                assert!((got[0] - expected[0]).abs() < 1e-9 && (got[1] - expected[1]).abs() < 1e-9);
                let back = map.inverse_at(got).unwrap();
                assert!((back[0] - x - 0.5).abs() < 1e-9 && (back[1] - y - 0.5).abs() < 1e-9);
            }
            let band = map.from_row(7);
            assert_eq!(band.at(3.0, 0.0), map.at(3.0, 7.0));
        }
    }

    /// Through a quarter turn, the turned blitter samples where the exact
    /// path (aligned blit, then permutation) does.
    #[test]
    fn quarter_turns_match_the_exact_path() {
        let data = tile(16, 12, 1, |x, y| (x * 13 + y * 7) as u8);
        for turns in 0..4 {
            let (viewport, map) = view(30.0, 24.0, f64::from(turns) * 90.0);
            let (aw, ah) = viewport.orientation.swap(30u32, 24u32);
            let r = rect(2.0, 3.0, 20.3, 15.3);
            let mut aligned = vec![0u8; (aw * ah * 4) as usize];
            blit_tile(&mut aligned, aw, ah, src(&data, 16, 12, 1), r);
            let mut expected = vec![0u8; aligned.len()];
            orient_frame(&aligned, aw, ah, viewport.orientation, &mut expected);
            let mut turned = vec![0u8; aligned.len()];
            blit_tile_turned(&mut turned, 30, 24, src(&data, 16, 12, 1), r, &map);
            // Covered where the exact path covers whole pixel centres.
            for y in 0..24usize {
                for x in 0..30usize {
                    let i = (y * 30 + x) * 4;
                    let a = map.at(x as f64, y as f64);
                    let inside = a[0] >= 2.0 && a[0] < 22.3 && a[1] >= 3.0 && a[1] < 18.3;
                    if inside {
                        assert!(
                            max_difference(&turned[i..i + 4], &expected[i..i + 4]) <= 1,
                            "{turns}: {x},{y}"
                        );
                    } else {
                        assert_eq!(turned[i + 3], 0, "{turns}: {x},{y} is outside");
                    }
                }
            }
        }
    }

    /// Two tiles side by side draw as one tile does: no seam, no gap, at
    /// any angle (the borders hold the neighbour's pixels).
    #[test]
    fn adjacent_turned_tiles_meet_without_seams() {
        let grey = |x: u32, y: u32| ((x * 9 + y * 5) % 251) as u8;
        let whole = tile(32, 16, 1, grey);
        let mut left = tile(16, 16, 1, grey);
        let mut right = tile(16, 16, 1, |x, y| grey(x + 16, y));
        // Real borders: the neighbour's column.
        for row in 0..18usize {
            let y = (row as u32).saturating_sub(1).min(15);
            let at = |col: usize| (row * 18 + col) * 4;
            let v = grey(16, y);
            left[at(17)..at(17) + 4].copy_from_slice(&[v, v / 2, 255 - v, 255]);
            let v = grey(15, y);
            right[at(0)..at(0) + 4].copy_from_slice(&[v, v / 2, 255 - v, 255]);
        }
        for degrees in [30.0, 45.0, 161.0] {
            let (_, map) = view(60.0, 60.0, degrees);
            let (x0, y0, scale) = (7.3, 11.9, 1.37);
            let mut expected = vec![0u8; 60 * 60 * 4];
            blit_tile_turned(
                &mut expected,
                60,
                60,
                src(&whole, 32, 16, 1),
                rect(x0, y0, 32.0 * scale, 16.0 * scale),
                &map,
            );
            let mut split = vec![0u8; 60 * 60 * 4];
            let x_mid = x0 + 16.0 * scale;
            blit_tile_turned(
                &mut split,
                60,
                60,
                src(&left, 16, 16, 1),
                rect(x0, y0, x_mid - x0, 16.0 * scale),
                &map,
            );
            blit_tile_turned(
                &mut split,
                60,
                60,
                src(&right, 16, 16, 1),
                rect(x_mid, y0, x0 + 32.0 * scale - x_mid, 16.0 * scale),
                &map,
            );
            assert!(max_difference(&split, &expected) <= 1, "{degrees}°");
            let covered = |frame: &[u8]| frame.chunks(4).filter(|p| p[3] == 255).count();
            assert_eq!(covered(&split), covered(&expected), "{degrees}°: no gap");
            assert!(covered(&expected) > 500);
        }
    }

    /// The pixel view turned: every pixel is the source pixel its centre
    /// falls in, and nothing is blended.
    #[test]
    fn nearest_turned_blits_pick_the_source_pixel_under_each_centre() {
        let data = tile(8, 6, 1, |x, y| (x * 30 + y) as u8);
        let (_, map) = view(48.0, 48.0, 27.0);
        let r = rect(5.25, 6.5, 8.0 * 3.3, 6.0 * 3.3);
        let mut dest = vec![0u8; 48 * 48 * 4];
        blit_tile_nearest_turned(&mut dest, 48, 48, src(&data, 8, 6, 1), r, &map);
        let mut drawn = 0;
        for y in 0..48usize {
            for x in 0..48usize {
                let a = map.at(x as f64, y as f64);
                let (sx, sy) = ((a[0] - 5.25) / 3.3, (a[1] - 6.5) / 3.3);
                let i = (y * 48 + x) * 4;
                if !(0.0..8.0).contains(&sx) || !(0.0..6.0).contains(&sy) {
                    assert_eq!(dest[i + 3], 0);
                    continue;
                }
                drawn += 1;
                let expected = (sx.floor() as u32 * 30 + sy.floor() as u32) as u8;
                assert_eq!(dest[i], expected, "at {x},{y}");
            }
        }
        assert!(drawn > 300);
    }

    #[test]
    fn turned_lanczos_and_trilinear_keep_flat_areas_and_blend() {
        let flat = tile(16, 16, 4, |_, _| 90);
        let (viewport, map) = view(40.0, 40.0, 33.0);
        // Rects about the middle of the aligned layout show on screen.
        let (aw, _) = viewport.orientation.turned_extent(40.0, 40.0);
        let centred = |size: f64| rect(aw / 2.0 - size / 2.0, aw / 2.0 - size / 2.0, size, size);
        for size in [8.0, 16.0, 30.0] {
            let mut dest = vec![0u8; 40 * 40 * 4];
            blit_tile_lanczos3_turned(
                &mut dest,
                40,
                40,
                src(&flat, 16, 16, 4),
                centred(size),
                &map,
            );
            let drawn: Vec<_> = dest.chunks(4).filter(|p| p[3] == 255).collect();
            assert!(!drawn.is_empty());
            assert!(drawn.iter().all(|p| p[0] == 90), "{size}");
        }
        let fine = tile(4, 4, 1, |_, _| 100);
        let coarse = tile(4, 4, 1, |_, _| 202);
        let mut dest = vec![0u8; 40 * 40 * 4];
        blit_tile_trilinear_turned(
            &mut dest,
            40,
            40,
            src(&fine, 4, 4, 1),
            &CoarseSrc {
                data: &coarse,
                width: 4,
                height: 4,
                border: 1,
                uv_min: [0.0, 0.0],
                uv_max: [0.5, 0.5],
                blend: 0.5,
            },
            centred(20.0),
            &map,
        );
        let drawn: Vec<_> = dest.chunks(4).filter(|p| p[3] == 255).collect();
        assert!(drawn.len() > 200);
        assert!(drawn.iter().all(|p| p[0] == 151), "(100 + 202) / 2");
    }

    /// Reprojecting a frame onto itself reproduces it; turned 90° it is the
    /// permutation; outside the source is the clear colour.
    #[test]
    fn mapped_reprojection_resamples_any_view_change() {
        let frame: Vec<u8> = (0..12 * 8)
            .flat_map(|i| [(i * 2) as u8, (i % 7) as u8, 9, 255])
            .collect();
        let identity = FrameMap {
            origin: [0.5, 0.5],
            step_x: [1.0, 0.0],
            step_y: [0.0, 1.0],
        };
        let mut dest = vec![0u8; frame.len()];
        reproject_frame_mapped(&mut dest, 12, 8, &frame, 12, 8, &identity, [1, 2, 3, 4]);
        assert_eq!(dest, frame);
        // Clockwise: destination (x, y) shows source (y, 7 - x)... as
        // orient_frame turns.
        let turned = FrameMap {
            origin: [0.5, 7.5],
            step_x: [0.0, -1.0],
            step_y: [1.0, 0.0],
        };
        let mut dest = vec![0u8; frame.len()];
        reproject_frame_mapped(&mut dest, 8, 12, &frame, 12, 8, &turned, [1, 2, 3, 4]);
        let mut expected = vec![0u8; frame.len()];
        orient_frame(&frame, 12, 8, Orientation::new(1, false), &mut expected);
        assert_eq!(dest, expected);
        let shifted = FrameMap {
            origin: [-3.5, 0.5],
            ..identity
        };
        let mut dest = vec![0u8; frame.len()];
        reproject_frame_mapped(&mut dest, 12, 8, &frame, 12, 8, &shifted, [1, 2, 3, 4]);
        assert_eq!(&dest[..4], &[1, 2, 3, 4]);
        assert_eq!(&dest[4 * 4..4 * 4 + 4], &frame[..4]);
    }
}
