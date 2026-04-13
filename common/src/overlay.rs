//! CPU overlay drawing primitives for export rendering.
//!
//! Provides anti-aliased line, circle, rectangle outline, and fill
//! operations on RGBA pixel buffers.

/// RGBA color for overlay drawing.
#[derive(Debug, Clone, Copy)]
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
}

/// Stroke pattern for lines and outlines.
#[derive(Debug, Clone, Copy)]
pub enum StrokeStyle {
    Solid,
    Dashed { length: f32, gap: f32 },
    Dotted { spacing: f32 },
}

/// Cap style for line endpoints.
#[derive(Debug, Clone, Copy)]
pub enum CapStyle {
    Round,
    Square,
    Flat,
}

// ── pixel blending ──────────────────────────────────────────────────────────

/// Alpha-blend a single pixel onto the buffer (src-over compositing).
#[inline]
fn blend_pixel(
    buffer: &mut [u8],
    buf_width: u32,
    x: u32,
    y: u32,
    color: OverlayColor,
    coverage: f32,
) {
    let idx = (y as usize * buf_width as usize + x as usize) * 4;
    if idx + 3 >= buffer.len() {
        return;
    }
    let sa = (color.a as f32 / 255.0) * coverage;
    if sa <= 0.0 {
        return;
    }
    let inv = 1.0 - sa;
    buffer[idx] = (color.r as f32 * sa + buffer[idx] as f32 * inv).min(255.0) as u8;
    buffer[idx + 1] = (color.g as f32 * sa + buffer[idx + 1] as f32 * inv).min(255.0) as u8;
    buffer[idx + 2] = (color.b as f32 * sa + buffer[idx + 2] as f32 * inv).min(255.0) as u8;
    let da = buffer[idx + 3] as f32 / 255.0;
    buffer[idx + 3] = ((sa + da * inv) * 255.0).min(255.0) as u8;
}

// ── SDF line coverage ───────────────────────────────────────────────────────

/// Compute the coverage (0.0–1.0) of a pixel at (px, py) for a thick line.
#[allow(clippy::too_many_arguments)]
fn line_pixel_coverage(
    px: f32,
    py: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    half_t: f32,
    stroke: &StrokeStyle,
    cap: &CapStyle,
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

    // Distance to the line segment, considering cap style.
    let dist = match cap {
        CapStyle::Round => {
            let tc = t.clamp(0.0, 1.0);
            let cx = x1 + tc * dx;
            let cy = y1 + tc * dy;
            ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
        }
        CapStyle::Flat => {
            if !(0.0..=1.0).contains(&t) {
                return 0.0;
            }
            let lx = x1 + t * dx;
            let ly = y1 + t * dy;
            ((px - lx).powi(2) + (py - ly).powi(2)).sqrt()
        }
        CapStyle::Square => {
            let extend = half_t / len;
            if t < -extend || t > 1.0 + extend {
                return 0.0;
            }
            // For the square-cap region beyond the segment endpoints,
            // compute perpendicular distance to the infinite line.
            let lx = x1 + t * dx;
            let ly = y1 + t * dy;
            ((px - lx).powi(2) + (py - ly).powi(2)).sqrt()
        }
    };

    let edge_alpha = (half_t + 0.5 - dist).clamp(0.0, 1.0);
    if edge_alpha <= 0.0 {
        return 0.0;
    }

    // Check stroke pattern.
    let along = (t * len).clamp(0.0, len);
    let pattern_alpha = match stroke {
        StrokeStyle::Solid => 1.0,
        StrokeStyle::Dashed { length, gap } => {
            let period = length + gap;
            if period <= 0.0 {
                1.0
            } else {
                let pos = along % period;
                if pos < *length {
                    1.0
                } else {
                    0.0
                }
            }
        }
        StrokeStyle::Dotted { spacing } => {
            if *spacing <= 0.0 {
                return edge_alpha;
            }
            let nearest = (along / spacing).round() * spacing;
            let dot_dist = (along - nearest).abs();
            let dot_r = half_t;
            (dot_r + 0.5 - dot_dist).clamp(0.0, 1.0)
        }
    };

    edge_alpha * pattern_alpha
}

// ── public drawing API ──────────────────────────────────────────────────────

/// Draw a thick anti-aliased line on an RGBA buffer.
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
    let half_t = thickness / 2.0;
    let cap_extend = match cap {
        CapStyle::Round | CapStyle::Square => half_t,
        CapStyle::Flat => 0.0,
    };
    let extend = half_t + 1.5 + cap_extend;

    let min_x = (x1.min(x2) - extend).floor().max(0.0) as u32;
    let min_y = (y1.min(y2) - extend).floor().max(0.0) as u32;
    let max_x = (x1.max(x2) + extend)
        .ceil()
        .min(buf_width as f32 - 1.0)
        .max(0.0) as u32;
    let max_y = (y1.max(y2) + extend)
        .ceil()
        .min(buf_height as f32 - 1.0)
        .max(0.0) as u32;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let cov = line_pixel_coverage(
                px as f32 + 0.5,
                py as f32 + 0.5,
                x1,
                y1,
                x2,
                y2,
                half_t,
                &stroke,
                &cap,
            );
            if cov > 0.0 {
                blend_pixel(buffer, buf_width, px, py, color, cov);
            }
        }
    }
}

/// Draw a filled anti-aliased circle on an RGBA buffer.
pub fn draw_filled_circle(
    buffer: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: OverlayColor,
) {
    let extend = radius + 1.0;
    let min_x = (cx - extend).floor().max(0.0) as u32;
    let min_y = (cy - extend).floor().max(0.0) as u32;
    let max_x = (cx + extend)
        .ceil()
        .min(buf_width as f32 - 1.0)
        .max(0.0) as u32;
    let max_y = (cy + extend)
        .ceil()
        .min(buf_height as f32 - 1.0)
        .max(0.0) as u32;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let cov = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if cov > 0.0 {
                blend_pixel(buffer, buf_width, px, py, color, cov);
            }
        }
    }
}

/// Draw a rectangle outline on an RGBA buffer (4 lines).
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
    let x2 = x + width;
    let y2 = y + height;
    // Top
    draw_line(buffer, buf_width, buf_height, x, y, x2, y, color, thickness, stroke, cap);
    // Bottom
    draw_line(buffer, buf_width, buf_height, x, y2, x2, y2, color, thickness, stroke, cap);
    // Left
    draw_line(buffer, buf_width, buf_height, x, y, x, y2, color, thickness, stroke, cap);
    // Right
    draw_line(buffer, buf_width, buf_height, x2, y, x2, y2, color, thickness, stroke, cap);
}

/// Fill a rectangle region with alpha blending (sub-pixel AA on edges).
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
    if width <= 0.0 || height <= 0.0 {
        return;
    }
    let x1 = x.max(0.0).floor() as u32;
    let y1 = y.max(0.0).floor() as u32;
    let x2 = (x + width).min(buf_width as f32).ceil().min(buf_width as f32) as u32;
    let y2 = (y + height).min(buf_height as f32).ceil().min(buf_height as f32) as u32;

    for py in y1..y2 {
        for px in x1..x2 {
            let left_cov = ((px as f32 + 1.0) - x).clamp(0.0, 1.0);
            let right_cov = ((x + width) - px as f32).clamp(0.0, 1.0);
            let top_cov = ((py as f32 + 1.0) - y).clamp(0.0, 1.0);
            let bottom_cov = ((y + height) - py as f32).clamp(0.0, 1.0);
            let cov = left_cov * right_cov * top_cov * bottom_cov;
            if cov > 0.0 {
                blend_pixel(buffer, buf_width, px, py, color, cov);
            }
        }
    }
}

/// Fill the area outside a rectangle (L-frame of 4 rectangles).
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
    let bw = buf_width as f32;
    let bh = buf_height as f32;

    // Top strip (full width, above ROI)
    if roi_y > 0.0 {
        fill_rect(buffer, buf_width, buf_height, 0.0, 0.0, bw, roi_y, color);
    }
    // Bottom strip (full width, below ROI)
    let roi_bottom = roi_y + roi_height;
    if roi_bottom < bh {
        fill_rect(buffer, buf_width, buf_height, 0.0, roi_bottom, bw, bh - roi_bottom, color);
    }
    // Left strip (between top and bottom)
    if roi_x > 0.0 {
        fill_rect(buffer, buf_width, buf_height, 0.0, roi_y, roi_x, roi_height, color);
    }
    // Right strip (between top and bottom)
    let roi_right = roi_x + roi_width;
    if roi_right < bw {
        fill_rect(buffer, buf_width, buf_height, roi_right, roi_y, bw - roi_right, roi_height, color);
    }
}

// ── text rendering (ab_glyph) ───────────────────────────────────────────────

pub use ab_glyph::FontArc;

/// Try to load a system font suitable for overlay text rendering.
///
/// Searches common system font paths across Linux, macOS, and Windows.
/// Returns `None` if no usable font is found.
pub fn load_system_font() -> Option<FontArc> {
    let candidates = [
        // Linux (Debian/Ubuntu)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        // Linux (Arch)
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        // Linux (Fedora)
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        // Linux (Noto fallback)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        // macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        // Windows
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];
    for path in &candidates {
        if let Ok(data) = std::fs::read(path)
            && let Ok(font) = FontArc::try_from_vec(data)
        {
            return Some(font);
        }
    }
    None
}

/// Measure the pixel width of a text string rendered at `size_px`.
pub fn measure_text(text: &str, font: &FontArc, size_px: f32) -> f32 {
    use ab_glyph::{Font, ScaleFont};
    let scaled = font.as_scaled(size_px);
    text.chars()
        .map(|ch| scaled.h_advance(font.glyph_id(ch)))
        .sum()
}

/// Draw a text string at `(x, y)` (top-left) onto an RGBA buffer.
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
    use ab_glyph::{Font, ScaleFont, point};
    let scaled = font.as_scaled(size_px);
    let ascent = scaled.ascent();
    let mut cursor_x = x;

    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        let glyph = ab_glyph::Glyph {
            id: glyph_id,
            scale: ab_glyph::PxScale::from(size_px),
            position: point(cursor_x, y + ascent),
        };
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, cov| {
                let px = gx as i32 + bounds.min.x as i32;
                let py = gy as i32 + bounds.min.y as i32;
                if px >= 0
                    && py >= 0
                    && (px as u32) < buf_width
                    && (py as u32) < buf_height
                    && cov > 0.0
                {
                    blend_pixel(buffer, buf_width, px as u32, py as u32, color, cov);
                }
            });
        }
        cursor_x += scaled.h_advance(glyph_id);
    }
}

/// Format a measurement distance (in micrometres) as a human-readable label.
pub fn format_measurement_label(distance_um: f64) -> String {
    if distance_um <= 0.0 {
        return String::new();
    }
    if distance_um >= 1000.0 {
        let mm = distance_um / 1000.0;
        format!("{:.2} mm", (mm * 100.0).round() / 100.0)
    } else {
        // Show as µm for sub-millimetre distances
        format!("{:.1} µm", (distance_um * 10.0).round() / 10.0)
    }
}

/// Draw a measurement label pill (dark background + white text) centred
/// above the midpoint between two endpoints.
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
    if label.is_empty() {
        return;
    }
    use ab_glyph::{Font, ScaleFont};
    let scaled = font.as_scaled(font_size_px);
    let text_w = measure_text(label, font, font_size_px);
    let text_h = scaled.ascent() - scaled.descent();
    let pad_x = 7.0 * dpi_scale;
    let pad_y = 3.0 * dpi_scale;
    let pill_w = text_w + pad_x * 2.0;
    let pill_h = text_h + pad_y * 2.0;
    let pill_x = cx - pill_w / 2.0;
    let pill_y = cy - pill_h - 6.0 * dpi_scale;

    // Dark semi-transparent background
    let bg = OverlayColor::new(26, 26, 26, 220);
    fill_rect(buffer, buf_width, buf_height, pill_x, pill_y, pill_w, pill_h, bg);

    // Border
    let border = OverlayColor::new(46, 204, 113, 128);
    let bw = 1.0 * dpi_scale;
    draw_rect_outline(
        buffer, buf_width, buf_height,
        pill_x, pill_y, pill_w, pill_h,
        border, bw, StrokeStyle::Solid, CapStyle::Flat,
    );

    // White text
    let white = OverlayColor::new(255, 255, 255, 255);
    let text_x = pill_x + pad_x;
    let text_y = pill_y + pad_y;
    draw_text(buffer, buf_width, buf_height, text_x, text_y, label, white, font, font_size_px);
}
