//! CPU overlay drawing for exported images: anti-aliased lines, circles,
//! rectangles, polygons and text, a calibrated scale bar, measurement
//! labels, and a display list ([`OverlayItem`]) that a banded renderer can
//! draw one strip at a time.
//!
//! All drawing goes through a [`Canvas`], which may cover a whole image or
//! a horizontal band of one; coordinates are always in whole-image pixels
//! (pixel `(x, y)` covers `[x, x+1) × [y, y+1)`), so splitting an image into
//! bands does not change the result. Text uses an embedded font
//! ([`embedded_font`]), never system font paths.

mod font;
mod items;
pub mod measure;
mod scale_bar;

pub use ab_glyph::FontArc;
pub use font::embedded_font;
#[allow(deprecated)]
pub use font::load_system_font;
pub use items::{Corner, OverlayItem, measurement_items};
#[allow(deprecated)]
pub use measure::format_measurement_label;
pub use measure::{Calibration, Length, format_length, format_physical_length, measure_segment};
pub use scale_bar::{ScaleBarLayout, ScaleBarStyle, layout_scale_bar, nice_scale_length_um};

use serde::{Deserialize, Serialize};

/// RGBA colour (straight alpha) for overlay drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl OverlayColor {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
}

/// Stroke pattern for lines and outlines.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StrokeStyle {
    Solid,
    Dashed { length: f32, gap: f32 },
    Dotted { spacing: f32 },
}

/// Cap style for line endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapStyle {
    Round,
    Square,
    Flat,
}

/// Appearance of a text label pill (measurement labels, badges, captions).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LabelStyle {
    pub background: OverlayColor,
    /// Border colour, or `None` for no border.
    pub border: Option<OverlayColor>,
    pub text: OverlayColor,
    /// Horizontal / vertical padding around the text, in pixels at scale 1.
    pub padding: [f32; 2],
    /// Border width in pixels at scale 1.
    pub border_width: f32,
}

impl Default for LabelStyle {
    /// The viewport's measurement-label look: dark pill, white text, a
    /// translucent green border.
    fn default() -> Self {
        Self {
            background: OverlayColor::new(26, 26, 26, 220),
            border: Some(OverlayColor::new(46, 204, 113, 128)),
            text: OverlayColor::WHITE,
            padding: [7.0, 3.0],
            border_width: 1.0,
        }
    }
}

impl LabelStyle {
    /// The default style with the border in `accent` (e.g. the measurement
    /// line's colour) instead of the fixed green.
    pub fn with_border(mut self, accent: OverlayColor) -> Self {
        self.border = Some(OverlayColor {
            a: accent.a.min(160),
            ..accent
        });
        self
    }
}

/// A mutable RGBA image, or a horizontal band of one, to draw on.
pub struct Canvas<'a> {
    pixels: &'a mut [u8],
    width: u32,
    rows: u32,
    origin_y: i64,
}

impl<'a> Canvas<'a> {
    /// A whole `width × height` image.
    pub fn new(pixels: &'a mut [u8], width: u32, height: u32) -> Self {
        Self::band(pixels, width, height, 0)
    }

    /// `rows` rows of an image starting at image row `origin_y`.
    pub fn band(pixels: &'a mut [u8], width: u32, rows: u32, origin_y: i64) -> Self {
        let rows = rows.min((pixels.len() / (width.max(1) as usize * 4)) as u32);
        Self {
            pixels,
            width,
            rows,
            origin_y,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn origin_y(&self) -> i64 {
        self.origin_y
    }

    /// Image rows covered by this canvas, `[first, end)`.
    pub fn row_range(&self) -> (i64, i64) {
        (self.origin_y, self.origin_y + self.rows as i64)
    }

    /// Clamps an image-space y range to canvas rows, as image rows.
    fn clip_rows(&self, y0: f32, y1: f32) -> Option<(i64, i64)> {
        let (first, end) = self.row_range();
        let lo = (y0.floor() as i64).max(first);
        let hi = (y1.ceil() as i64).min(end - 1);
        (hi >= lo).then_some((lo, hi))
    }

    fn clip_cols(&self, x0: f32, x1: f32) -> Option<(i64, i64)> {
        let lo = (x0.floor() as i64).max(0);
        let hi = (x1.ceil() as i64).min(self.width as i64 - 1);
        (hi >= lo).then_some((lo, hi))
    }

    /// Alpha-blends `color` onto image pixel `(x, y)` with the given
    /// coverage (src-over, rounded).
    #[inline]
    pub fn blend(&mut self, x: i64, y: i64, color: OverlayColor, coverage: f32) {
        if x < 0 || x >= self.width as i64 || y < self.origin_y {
            return;
        }
        let row = y - self.origin_y;
        if row >= self.rows as i64 {
            return;
        }
        let idx = (row as usize * self.width as usize + x as usize) * 4;
        let sa = (color.a as f32 / 255.0) * coverage.clamp(0.0, 1.0);
        if sa <= 0.0 {
            return;
        }
        let inv = 1.0 - sa;
        let px = &mut self.pixels[idx..idx + 4];
        px[0] = (color.r as f32 * sa + px[0] as f32 * inv)
            .round()
            .min(255.0) as u8;
        px[1] = (color.g as f32 * sa + px[1] as f32 * inv)
            .round()
            .min(255.0) as u8;
        px[2] = (color.b as f32 * sa + px[2] as f32 * inv)
            .round()
            .min(255.0) as u8;
        let da = px[3] as f32 / 255.0;
        px[3] = ((sa + da * inv) * 255.0).round().min(255.0) as u8;
    }

    /// Thick anti-aliased line segment.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_line(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    ) {
        self.draw_segment(x1, y1, x2, y2, color, thickness, stroke, cap, 0.0);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_segment(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
        pattern_offset: f32,
    ) {
        let half_t = thickness / 2.0;
        let cap_extend = match cap {
            CapStyle::Round | CapStyle::Square => half_t,
            CapStyle::Flat => 0.0,
        };
        let extend = half_t + 1.5 + cap_extend;
        let Some((ry0, ry1)) = self.clip_rows(y1.min(y2) - extend, y1.max(y2) + extend) else {
            return;
        };
        let Some((cx0, cx1)) = self.clip_cols(x1.min(x2) - extend, x1.max(x2) + extend) else {
            return;
        };
        for py in ry0..=ry1 {
            for px in cx0..=cx1 {
                let cov = line_pixel_coverage(
                    px as f32 + 0.5,
                    py as f32 + 0.5,
                    [x1, y1, x2, y2],
                    half_t,
                    &stroke,
                    &cap,
                    pattern_offset,
                );
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// Open or closed polyline; dash patterns continue across vertices.
    pub fn draw_polyline(
        &mut self,
        points: &[[f32; 2]],
        closed: bool,
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    ) {
        if points.len() < 2 {
            if let Some(p) = points.first() {
                self.fill_circle(p[0], p[1], thickness / 2.0, color);
            }
            return;
        }
        let mut offset = 0.0f32;
        let count = if closed {
            points.len()
        } else {
            points.len() - 1
        };
        for i in 0..count {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            self.draw_segment(
                a[0], a[1], b[0], b[1], color, thickness, stroke, cap, offset,
            );
            offset += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        }
    }

    /// Filled anti-aliased circle.
    pub fn fill_circle(&mut self, cx: f32, cy: f32, radius: f32, color: OverlayColor) {
        let extend = radius + 1.0;
        let Some((ry0, ry1)) = self.clip_rows(cy - extend, cy + extend) else {
            return;
        };
        let Some((cx0, cx1)) = self.clip_cols(cx - extend, cx + extend) else {
            return;
        };
        for py in ry0..=ry1 {
            for px in cx0..=cx1 {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                let cov = (radius + 0.5 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// Rectangle outline (four lines).
    #[allow(clippy::too_many_arguments)]
    pub fn stroke_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    ) {
        let (x2, y2) = (x + width, y + height);
        self.draw_line(x, y, x2, y, color, thickness, stroke, cap);
        self.draw_line(x, y2, x2, y2, color, thickness, stroke, cap);
        self.draw_line(x, y, x, y2, color, thickness, stroke, cap);
        self.draw_line(x2, y, x2, y2, color, thickness, stroke, cap);
    }

    /// Filled rectangle with fractional (anti-aliased) edges.
    pub fn fill_rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: OverlayColor) {
        if width <= 0.0 || height <= 0.0 {
            return;
        }
        let Some((ry0, ry1)) = self.clip_rows(y, y + height) else {
            return;
        };
        let Some((cx0, cx1)) = self.clip_cols(x, x + width) else {
            return;
        };
        // Exact area of the pixel covered by the rectangle, per axis.
        let overlap = |p: i64, lo: f32, hi: f32| -> f32 {
            ((p as f32 + 1.0).min(hi) - (p as f32).max(lo)).clamp(0.0, 1.0)
        };
        for py in ry0..=ry1 {
            let vertical = overlap(py, y, y + height);
            if vertical <= 0.0 {
                continue;
            }
            for px in cx0..=cx1 {
                let cov = overlap(px, x, x + width) * vertical;
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// Fills everything outside a rectangle within a `image_width ×
    /// image_height` image (the ROI "dim outside" effect).
    #[allow(clippy::too_many_arguments)]
    pub fn fill_outside_rect(
        &mut self,
        roi_x: f32,
        roi_y: f32,
        roi_width: f32,
        roi_height: f32,
        image_width: f32,
        image_height: f32,
        color: OverlayColor,
    ) {
        let bottom = roi_y + roi_height;
        let right = roi_x + roi_width;
        if roi_y > 0.0 {
            self.fill_rect(0.0, 0.0, image_width, roi_y, color);
        }
        if bottom < image_height {
            self.fill_rect(0.0, bottom, image_width, image_height - bottom, color);
        }
        if roi_x > 0.0 {
            self.fill_rect(0.0, roi_y, roi_x, roi_height, color);
        }
        if right < image_width {
            self.fill_rect(right, roi_y, image_width - right, roi_height, color);
        }
    }

    /// Fills polygons (outer rings and holes) with the even-odd rule,
    /// anti-aliased with four sub-scanlines per row and exact horizontal
    /// coverage.
    pub fn fill_polygon(&mut self, rings: &[Vec<[f32; 2]>], color: OverlayColor) {
        let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);
        let (mut x_min, mut x_max) = (f32::MAX, f32::MIN);
        for ring in rings {
            for p in ring {
                y_min = y_min.min(p[1]);
                y_max = y_max.max(p[1]);
                x_min = x_min.min(p[0]);
                x_max = x_max.max(p[0]);
            }
        }
        let Some((ry0, ry1)) = self.clip_rows(y_min, y_max) else {
            return;
        };
        let Some((cx0, cx1)) = self.clip_cols(x_min, x_max) else {
            return;
        };
        let span = (cx1 - cx0 + 1) as usize;
        let mut coverage = vec![0.0f32; span + 1];
        let mut crossings: Vec<f32> = Vec::new();
        const SUB: usize = 4;
        for py in ry0..=ry1 {
            coverage.iter_mut().for_each(|c| *c = 0.0);
            for k in 0..SUB {
                let sy = py as f32 + (k as f32 + 0.5) / SUB as f32;
                crossings.clear();
                for ring in rings {
                    if ring.len() < 3 {
                        continue;
                    }
                    for i in 0..ring.len() {
                        let a = ring[i];
                        let b = ring[(i + 1) % ring.len()];
                        if (a[1] <= sy) != (b[1] <= sy) {
                            let t = (sy - a[1]) / (b[1] - a[1]);
                            crossings.push(a[0] + t * (b[0] - a[0]));
                        }
                    }
                }
                crossings.sort_unstable_by(f32::total_cmp);
                for pair in crossings.as_chunks::<2>().0 {
                    accumulate_span(
                        &mut coverage,
                        cx0 as f32,
                        pair[0],
                        pair[1],
                        1.0 / SUB as f32,
                    );
                }
            }
            for (i, &cov) in coverage.iter().take(span).enumerate() {
                if cov > 0.0 {
                    self.blend(cx0 + i as i64, py, color, cov.min(1.0));
                }
            }
        }
    }

    /// Draws `text` with its top-left corner at `(x, y)`, with kerning.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text(
        &mut self,
        x: f32,
        y: f32,
        text: &str,
        color: OverlayColor,
        font: &FontArc,
        size_px: f32,
    ) {
        use ab_glyph::{Font, ScaleFont, point};
        let scaled = font.as_scaled(size_px);
        let ascent = scaled.ascent();
        let height = ascent - scaled.descent();
        let (first, end) = self.row_range();
        if y + height + 2.0 < first as f32 || y - 2.0 > end as f32 {
            return;
        }
        let mut cursor_x = x;
        let mut previous = None;
        for ch in text.chars() {
            let glyph_id = font.glyph_id(ch);
            if let Some(prev) = previous {
                cursor_x += scaled.kern(prev, glyph_id);
            }
            let glyph = ab_glyph::Glyph {
                id: glyph_id,
                scale: ab_glyph::PxScale::from(size_px),
                position: point(cursor_x, y + ascent),
            };
            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                outlined.draw(|gx, gy, cov| {
                    let px = gx as i64 + bounds.min.x as i64;
                    let py = gy as i64 + bounds.min.y as i64;
                    if cov > 0.0 {
                        self.blend(px, py, color, cov);
                    }
                });
            }
            cursor_x += scaled.h_advance(glyph_id);
            previous = Some(glyph_id);
        }
    }

    /// Draws a label pill centred horizontally on `cx`, with its bottom edge
    /// `6·scale` pixels above `cy` (the measurement-label placement).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_label(
        &mut self,
        cx: f32,
        cy: f32,
        text: &str,
        font: &FontArc,
        size_px: f32,
        scale: f32,
        style: &LabelStyle,
    ) {
        if text.is_empty() {
            return;
        }
        let (w, h) = label_size(text, font, size_px, scale, style);
        let x = cx - w / 2.0;
        let y = cy - h - 6.0 * scale;
        self.draw_label_at(x, y, text, font, size_px, scale, style);
    }

    /// Draws a label pill with its top-left corner at `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_label_at(
        &mut self,
        x: f32,
        y: f32,
        text: &str,
        font: &FontArc,
        size_px: f32,
        scale: f32,
        style: &LabelStyle,
    ) {
        let (w, h) = label_size(text, font, size_px, scale, style);
        self.fill_rect(x, y, w, h, style.background);
        if let Some(border) = style.border {
            self.stroke_rect(
                x,
                y,
                w,
                h,
                border,
                style.border_width * scale,
                StrokeStyle::Solid,
                CapStyle::Flat,
            );
        }
        self.draw_text(
            x + style.padding[0] * scale,
            y + style.padding[1] * scale,
            text,
            style.text,
            font,
            size_px,
        );
    }
}

/// Adds the horizontal coverage of span `[x0, x1)` to `coverage` (indexed
/// from `origin`), weighted by `weight`.
fn accumulate_span(coverage: &mut [f32], origin: f32, x0: f32, x1: f32, weight: f32) {
    let (x0, x1) = (x0 - origin, x1 - origin);
    let n = coverage.len() as f32 - 1.0;
    let x0 = x0.clamp(0.0, n);
    let x1 = x1.clamp(0.0, n);
    if x1 <= x0 {
        return;
    }
    let i0 = x0.floor() as usize;
    let i1 = x1.floor() as usize;
    if i0 == i1 {
        coverage[i0] += (x1 - x0) * weight;
        return;
    }
    coverage[i0] += (i0 as f32 + 1.0 - x0) * weight;
    for c in coverage.iter_mut().take(i1).skip(i0 + 1) {
        *c += weight;
    }
    if i1 < coverage.len() {
        coverage[i1] += (x1 - i1 as f32) * weight;
    }
}

/// Width and height of a label pill.
pub fn label_size(
    text: &str,
    font: &FontArc,
    size_px: f32,
    scale: f32,
    style: &LabelStyle,
) -> (f32, f32) {
    use ab_glyph::{Font, ScaleFont};
    let scaled = font.as_scaled(size_px);
    let text_h = scaled.ascent() - scaled.descent();
    (
        measure_text(text, font, size_px) + 2.0 * style.padding[0] * scale,
        text_h + 2.0 * style.padding[1] * scale,
    )
}

/// Coverage of pixel centre `(px, py)` by a thick line segment.
fn line_pixel_coverage(
    px: f32,
    py: f32,
    [x1, y1, x2, y2]: [f32; 4],
    half_t: f32,
    stroke: &StrokeStyle,
    cap: &CapStyle,
    pattern_offset: f32,
) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-6 {
        let dist = ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
        return (half_t + 0.5 - dist).clamp(0.0, 1.0);
    }
    let len = len_sq.sqrt();
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let dist = match cap {
        CapStyle::Round => {
            let tc = t.clamp(0.0, 1.0);
            ((px - (x1 + tc * dx)).powi(2) + (py - (y1 + tc * dy)).powi(2)).sqrt()
        }
        CapStyle::Flat => {
            if !(0.0..=1.0).contains(&t) {
                return 0.0;
            }
            ((px - (x1 + t * dx)).powi(2) + (py - (y1 + t * dy)).powi(2)).sqrt()
        }
        CapStyle::Square => {
            let extend = half_t / len;
            if t < -extend || t > 1.0 + extend {
                return 0.0;
            }
            ((px - (x1 + t * dx)).powi(2) + (py - (y1 + t * dy)).powi(2)).sqrt()
        }
    };
    let edge_alpha = (half_t + 0.5 - dist).clamp(0.0, 1.0);
    if edge_alpha <= 0.0 {
        return 0.0;
    }
    let along = (t * len).clamp(0.0, len) + pattern_offset;
    let pattern_alpha = match stroke {
        StrokeStyle::Solid => 1.0,
        StrokeStyle::Dashed { length, gap } => {
            let period = length + gap;
            if period <= 0.0 || (along % period) < *length {
                1.0
            } else {
                0.0
            }
        }
        StrokeStyle::Dotted { spacing } => {
            if *spacing <= 0.0 {
                return edge_alpha;
            }
            let nearest = (along / spacing).round() * spacing;
            (half_t + 0.5 - (along - nearest).abs()).clamp(0.0, 1.0)
        }
    };
    edge_alpha * pattern_alpha
}

/// Pixel width of `text` at `size_px`, including kerning.
pub fn measure_text(text: &str, font: &FontArc, size_px: f32) -> f32 {
    use ab_glyph::{Font, ScaleFont};
    let scaled = font.as_scaled(size_px);
    let mut width = 0.0;
    let mut previous = None;
    for ch in text.chars() {
        let id = font.glyph_id(ch);
        if let Some(prev) = previous {
            width += scaled.kern(prev, id);
        }
        width += scaled.h_advance(id);
        previous = Some(id);
    }
    width
}

// ─── Whole-image convenience functions (pre-1.0 API) ─────────────────────────

/// Draws a thick anti-aliased line on an RGBA buffer.
#[allow(clippy::too_many_arguments)]
pub fn draw_line(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: OverlayColor,
    thickness: f32,
    stroke: StrokeStyle,
    cap: CapStyle,
) {
    Canvas::new(buffer, buf_width, buf_height)
        .draw_line(x1, y1, x2, y2, color, thickness, stroke, cap);
}

/// Draws a filled anti-aliased circle on an RGBA buffer.
pub fn draw_filled_circle(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: OverlayColor,
) {
    Canvas::new(buffer, buf_width, buf_height).fill_circle(cx, cy, radius, color);
}

/// Draws a rectangle outline on an RGBA buffer.
#[allow(clippy::too_many_arguments)]
pub fn draw_rect_outline(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: OverlayColor,
    thickness: f32,
    stroke: StrokeStyle,
    cap: CapStyle,
) {
    Canvas::new(buffer, buf_width, buf_height)
        .stroke_rect(x, y, width, height, color, thickness, stroke, cap);
}

/// Fills a rectangle with alpha blending (anti-aliased edges).
#[allow(clippy::too_many_arguments)]
pub fn fill_rect(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: OverlayColor,
) {
    Canvas::new(buffer, buf_width, buf_height).fill_rect(x, y, width, height, color);
}

/// Fills the area outside a rectangle.
#[allow(clippy::too_many_arguments)]
pub fn fill_outside_rect(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    roi_x: f32,
    roi_y: f32,
    roi_width: f32,
    roi_height: f32,
    color: OverlayColor,
) {
    Canvas::new(buffer, buf_width, buf_height).fill_outside_rect(
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        buf_width as f32,
        buf_height as f32,
        color,
    );
}

/// Draws text with its top-left corner at `(x, y)`.
#[allow(clippy::too_many_arguments)]
pub fn draw_text(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    x: f32,
    y: f32,
    text: &str,
    color: OverlayColor,
    font: &FontArc,
    size_px: f32,
) {
    Canvas::new(buffer, buf_width, buf_height).draw_text(x, y, text, color, font, size_px);
}

/// Draws a measurement label pill (default [`LabelStyle`]) centred above
/// `(cx, cy)`.
#[allow(clippy::too_many_arguments)]
pub fn draw_measurement_label(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    cx: f32,
    cy: f32,
    label: &str,
    font: &FontArc,
    font_size_px: f32,
    dpi_scale: f32,
) {
    Canvas::new(buffer, buf_width, buf_height).draw_label(
        cx,
        cy,
        label,
        font,
        font_size_px,
        dpi_scale,
        &LabelStyle::default(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    fn alpha_at(buf: &[u8], w: u32, x: u32, y: u32) -> u8 {
        buf[((y * w + x) * 4 + 3) as usize]
    }

    #[test]
    fn banded_drawing_matches_whole_image_drawing() {
        let (w, h) = (64u32, 48u32);
        let font = embedded_font();
        let draw = |canvas: &mut Canvas| {
            canvas.draw_line(
                3.0,
                5.0,
                60.0,
                40.0,
                OverlayColor::new(255, 0, 0, 255),
                3.0,
                StrokeStyle::Dashed {
                    length: 6.0,
                    gap: 3.0,
                },
                CapStyle::Round,
            );
            canvas.fill_circle(30.0, 24.0, 7.5, OverlayColor::new(0, 255, 0, 200));
            canvas.fill_polygon(
                &[vec![[5.0, 30.0], [25.0, 44.0], [2.0, 46.0]]],
                OverlayColor::new(0, 0, 255, 160),
            );
            canvas.draw_label(
                32.0,
                30.0,
                "12.5 µm",
                font,
                10.0,
                1.0,
                &LabelStyle::default(),
            );
        };
        let mut whole = blank(w, h);
        draw(&mut Canvas::new(&mut whole, w, h));
        let mut banded = blank(w, h);
        let stride = (w * 4) as usize;
        for (i, band) in banded.chunks_mut(stride * 7).enumerate() {
            let rows = (band.len() / stride) as u32;
            draw(&mut Canvas::band(band, w, rows, (i * 7) as i64));
        }
        assert_eq!(whole, banded);
    }

    #[test]
    fn polygon_fill_respects_holes_and_coverage() {
        let (w, h) = (20u32, 20u32);
        let mut buf = blank(w, h);
        let outer = vec![[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]];
        let hole = vec![[8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0]];
        Canvas::new(&mut buf, w, h).fill_polygon(&[outer, hole], OverlayColor::WHITE);
        assert_eq!(alpha_at(&buf, w, 4, 4), 255);
        assert_eq!(alpha_at(&buf, w, 10, 10), 0, "hole stays empty");
        assert_eq!(alpha_at(&buf, w, 0, 0), 0);
        // Half-covered pixel on a fractional edge.
        let mut buf = blank(w, h);
        Canvas::new(&mut buf, w, h).fill_polygon(
            &[vec![[2.5, 2.0], [10.0, 2.0], [10.0, 10.0], [2.5, 10.0]]],
            OverlayColor::WHITE,
        );
        let a = alpha_at(&buf, w, 2, 5);
        assert!((120..=135).contains(&a), "{a}");
    }

    #[test]
    fn text_is_rendered_with_the_embedded_font() {
        let (w, h) = (80u32, 24u32);
        let mut buf = blank(w, h);
        Canvas::new(&mut buf, w, h).draw_text(
            2.0,
            2.0,
            "25 µm",
            OverlayColor::WHITE,
            embedded_font(),
            14.0,
        );
        let inked = buf.chunks(4).filter(|p| p[3] > 0).count();
        assert!(inked > 40, "{inked}");
        let width = measure_text("25 µm", embedded_font(), 14.0);
        assert!(width > 20.0 && width < 60.0, "{width}");
    }

    #[test]
    fn label_border_follows_the_accent_colour() {
        let style = LabelStyle::default().with_border(OverlayColor::new(255, 0, 0, 255));
        assert_eq!(style.border.unwrap().r, 255);
        assert_eq!(style.border.unwrap().g, 0);
    }

    #[test]
    fn blending_rounds() {
        let mut buf = vec![0u8, 0, 0, 255];
        Canvas::new(&mut buf, 1, 1).blend(0, 0, OverlayColor::new(255, 255, 255, 255), 0.5);
        assert_eq!(buf[0], 128);
    }
}
