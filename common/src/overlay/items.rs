//! A display list of overlay primitives in output-image coordinates.
//!
//! Exports are rendered in horizontal bands; overlays are therefore
//! described as data ([`OverlayItem`]) and drawn onto each band
//! ([`OverlayItem::draw`]), which gives exactly the same pixels as drawing
//! onto the whole image.

use ab_glyph::{Font, FontArc, ScaleFont};
use serde::{Deserialize, Serialize};

use super::scale_bar::{ScaleBarStyle, draw_scale_bar, layout_scale_bar};
use super::{Canvas, CapStyle, LabelStyle, OverlayColor, StrokeStyle, label_size};

/// An image corner, for anchored items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// One overlay primitive. Coordinates and sizes are output pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum OverlayItem {
    Line {
        from: [f32; 2],
        to: [f32; 2],
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    },
    Polyline {
        points: Vec<[f32; 2]>,
        closed: bool,
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    },
    /// Filled polygons (rings; even-odd rule, so inner rings are holes).
    Polygon {
        rings: Vec<Vec<[f32; 2]>>,
        color: OverlayColor,
    },
    FilledCircle {
        center: [f32; 2],
        radius: f32,
        color: OverlayColor,
    },
    RectOutline {
        rect: [f32; 4],
        color: OverlayColor,
        thickness: f32,
        stroke: StrokeStyle,
        cap: CapStyle,
    },
    FillRect {
        rect: [f32; 4],
        color: OverlayColor,
    },
    /// Fills the whole image outside `rect`.
    FillOutsideRect {
        rect: [f32; 4],
        color: OverlayColor,
    },
    /// Text with its top-left corner at `origin`.
    Text {
        origin: [f32; 2],
        text: String,
        color: OverlayColor,
        size: f32,
    },
    /// A label pill centred above `anchor` (measurement labels).
    Label {
        anchor: [f32; 2],
        text: String,
        size: f32,
        scale: f32,
        style: LabelStyle,
    },
    /// Lines of text in a pill anchored to an image corner (colour badge,
    /// provenance caption).
    TextBlock {
        corner: Corner,
        lines: Vec<String>,
        size: f32,
        scale: f32,
        margin: f32,
        style: LabelStyle,
    },
    /// A calibrated scale bar; `um_per_px` is the output pixel width.
    ScaleBar {
        um_per_px: f64,
        style: ScaleBarStyle,
    },
}

impl OverlayItem {
    /// Draws the item onto a canvas (whole image or band) of an
    /// `image_width × image_height` image.
    pub fn draw(&self, canvas: &mut Canvas, image_size: (u32, u32), font: &FontArc) {
        let (iw, ih) = image_size;
        match self {
            Self::Line {
                from,
                to,
                color,
                thickness,
                stroke,
                cap,
            } => canvas.draw_line(
                from[0], from[1], to[0], to[1], *color, *thickness, *stroke, *cap,
            ),
            Self::Polyline {
                points,
                closed,
                color,
                thickness,
                stroke,
                cap,
            } => canvas.draw_polyline(points, *closed, *color, *thickness, *stroke, *cap),
            Self::Polygon { rings, color } => canvas.fill_polygon(rings, *color),
            Self::FilledCircle {
                center,
                radius,
                color,
            } => canvas.fill_circle(center[0], center[1], *radius, *color),
            Self::RectOutline {
                rect,
                color,
                thickness,
                stroke,
                cap,
            } => canvas.stroke_rect(
                rect[0], rect[1], rect[2], rect[3], *color, *thickness, *stroke, *cap,
            ),
            Self::FillRect { rect, color } => {
                canvas.fill_rect(rect[0], rect[1], rect[2], rect[3], *color)
            }
            Self::FillOutsideRect { rect, color } => canvas.fill_outside_rect(
                rect[0], rect[1], rect[2], rect[3], iw as f32, ih as f32, *color,
            ),
            Self::Text {
                origin,
                text,
                color,
                size,
            } => canvas.draw_text(origin[0], origin[1], text, *color, font, *size),
            Self::Label {
                anchor,
                text,
                size,
                scale,
                style,
            } => canvas.draw_label(anchor[0], anchor[1], text, font, *size, *scale, style),
            Self::TextBlock {
                corner,
                lines,
                size,
                scale,
                margin,
                style,
            } => {
                let scaled = font.as_scaled(*size);
                let line_h = (scaled.ascent() - scaled.descent() + scaled.line_gap()).ceil();
                let mut sizes: Vec<(f32, f32)> = lines
                    .iter()
                    .map(|l| label_size(l, font, *size, *scale, style))
                    .collect();
                if sizes.is_empty() {
                    return;
                }
                let width = sizes.iter().map(|s| s.0).fold(0.0f32, f32::max);
                for s in &mut sizes {
                    s.0 = width;
                }
                let total_h = sizes[0].1 + line_h * (lines.len() as f32 - 1.0);
                let x = match corner {
                    Corner::TopLeft | Corner::BottomLeft => *margin,
                    Corner::TopRight | Corner::BottomRight => iw as f32 - margin - width,
                };
                let y = match corner {
                    Corner::TopLeft | Corner::TopRight => *margin,
                    Corner::BottomLeft | Corner::BottomRight => ih as f32 - margin - total_h,
                };
                canvas.fill_rect(x, y, width, total_h, style.background);
                if let Some(border) = style.border {
                    canvas.stroke_rect(
                        x,
                        y,
                        width,
                        total_h,
                        border,
                        style.border_width * scale,
                        StrokeStyle::Solid,
                        CapStyle::Flat,
                    );
                }
                for (i, line) in lines.iter().enumerate() {
                    canvas.draw_text(
                        x + style.padding[0] * scale,
                        y + style.padding[1] * scale + i as f32 * line_h,
                        line,
                        style.text,
                        font,
                        *size,
                    );
                }
            }
            Self::ScaleBar { um_per_px, style } => {
                if let Some(layout) = layout_scale_bar(iw, ih, *um_per_px, style, font) {
                    draw_scale_bar(canvas, &layout, style, font);
                }
            }
        }
    }

    /// The item with every coordinate and size multiplied by `factor`
    /// (e.g. for a downscaled preview).
    pub fn scaled(&self, factor: f32) -> Self {
        let p = |v: [f32; 2]| [v[0] * factor, v[1] * factor];
        let r = |v: [f32; 4]| v.map(|c| c * factor);
        let s = |stroke: StrokeStyle| match stroke {
            StrokeStyle::Solid => StrokeStyle::Solid,
            StrokeStyle::Dashed { length, gap } => StrokeStyle::Dashed {
                length: length * factor,
                gap: gap * factor,
            },
            StrokeStyle::Dotted { spacing } => StrokeStyle::Dotted {
                spacing: spacing * factor,
            },
        };
        match self.clone() {
            Self::Line {
                from,
                to,
                color,
                thickness,
                stroke,
                cap,
            } => Self::Line {
                from: p(from),
                to: p(to),
                color,
                thickness: thickness * factor,
                stroke: s(stroke),
                cap,
            },
            Self::Polyline {
                points,
                closed,
                color,
                thickness,
                stroke,
                cap,
            } => Self::Polyline {
                points: points.into_iter().map(p).collect(),
                closed,
                color,
                thickness: thickness * factor,
                stroke: s(stroke),
                cap,
            },
            Self::Polygon { rings, color } => Self::Polygon {
                rings: rings
                    .into_iter()
                    .map(|ring| ring.into_iter().map(p).collect())
                    .collect(),
                color,
            },
            Self::FilledCircle {
                center,
                radius,
                color,
            } => Self::FilledCircle {
                center: p(center),
                radius: radius * factor,
                color,
            },
            Self::RectOutline {
                rect,
                color,
                thickness,
                stroke,
                cap,
            } => Self::RectOutline {
                rect: r(rect),
                color,
                thickness: thickness * factor,
                stroke: s(stroke),
                cap,
            },
            Self::FillRect { rect, color } => Self::FillRect {
                rect: r(rect),
                color,
            },
            Self::FillOutsideRect { rect, color } => Self::FillOutsideRect {
                rect: r(rect),
                color,
            },
            Self::Text {
                origin,
                text,
                color,
                size,
            } => Self::Text {
                origin: p(origin),
                text,
                color,
                size: size * factor,
            },
            Self::Label {
                anchor,
                text,
                size,
                scale,
                style,
            } => Self::Label {
                anchor: p(anchor),
                text,
                size: size * factor,
                scale: scale * factor,
                style,
            },
            Self::TextBlock {
                corner,
                lines,
                size,
                scale,
                margin,
                style,
            } => Self::TextBlock {
                corner,
                lines,
                size: size * factor,
                scale: scale * factor,
                margin: margin * factor,
                style,
            },
            Self::ScaleBar { um_per_px, style } => Self::ScaleBar {
                um_per_px: um_per_px / factor as f64,
                style: style.scaled(factor),
            },
        }
    }
}

/// The viewport's measurement look as overlay items: the line, white-ringed
/// endpoints and (optionally) a label pill above the midpoint. `scale` is
/// the output DPI / 96.
#[allow(clippy::too_many_arguments)]
pub fn measurement_items(
    start: [f32; 2],
    end: [f32; 2],
    label: Option<String>,
    color: OverlayColor,
    thickness: f32,
    stroke: StrokeStyle,
    cap: CapStyle,
    font_size: f32,
    scale: f32,
    label_style: LabelStyle,
) -> Vec<OverlayItem> {
    let mut items = vec![OverlayItem::Line {
        from: start,
        to: end,
        color,
        thickness,
        stroke,
        cap,
    }];
    let endpoint_r = (thickness + 1.0) * 0.5 + scale;
    let ring = OverlayColor {
        a: color.a,
        ..OverlayColor::WHITE
    };
    for p in [start, end] {
        items.push(OverlayItem::FilledCircle {
            center: p,
            radius: endpoint_r + 1.5 * scale,
            color: ring,
        });
        items.push(OverlayItem::FilledCircle {
            center: p,
            radius: endpoint_r,
            color,
        });
    }
    if let Some(text) = label.filter(|t| !t.is_empty()) {
        items.push(OverlayItem::Label {
            anchor: [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0],
            text,
            size: font_size,
            scale,
            style: label_style,
        });
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeasurementUnit;
    use crate::overlay::embedded_font;

    fn render(items: &[OverlayItem], w: u32, h: u32, band: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let stride = (w * 4) as usize;
        for (i, chunk) in buf.chunks_mut(stride * band as usize).enumerate() {
            let rows = (chunk.len() / stride) as u32;
            let mut canvas = Canvas::band(chunk, w, rows, i as i64 * band as i64);
            for item in items {
                item.draw(&mut canvas, (w, h), embedded_font());
            }
        }
        buf
    }

    #[test]
    fn every_item_kind_draws_identically_in_bands() {
        let items = vec![
            OverlayItem::FillOutsideRect {
                rect: [10.0, 10.0, 80.0, 50.0],
                color: OverlayColor::new(0, 0, 0, 128),
            },
            OverlayItem::RectOutline {
                rect: [10.0, 10.0, 80.0, 50.0],
                color: OverlayColor::new(241, 196, 15, 255),
                thickness: 2.0,
                stroke: StrokeStyle::Dotted { spacing: 4.0 },
                cap: CapStyle::Round,
            },
            OverlayItem::Polyline {
                points: vec![[5.0, 70.0], [40.0, 30.0], [90.0, 75.0]],
                closed: true,
                color: OverlayColor::new(0, 128, 255, 255),
                thickness: 1.5,
                stroke: StrokeStyle::Dashed {
                    length: 5.0,
                    gap: 2.0,
                },
                cap: CapStyle::Flat,
            },
            OverlayItem::TextBlock {
                corner: Corner::TopLeft,
                lines: vec!["Colour-normalised (Macenko)".into(), "eov 1.0".into()],
                size: 9.0,
                scale: 1.0,
                margin: 2.0,
                style: LabelStyle::default(),
            },
            OverlayItem::ScaleBar {
                um_per_px: 0.5,
                style: ScaleBarStyle {
                    unit: MeasurementUnit::Um,
                    ..ScaleBarStyle::default().scaled(0.5)
                },
            },
        ]
        .into_iter()
        .chain(measurement_items(
            [20.0, 60.0],
            [70.0, 40.0],
            Some("12.5 µm".into()),
            OverlayColor::new(46, 204, 113, 255),
            2.0,
            StrokeStyle::Solid,
            CapStyle::Round,
            10.0,
            1.0,
            LabelStyle::default(),
        ))
        .collect::<Vec<_>>();
        let whole = render(&items, 100, 80, 80);
        assert!(whole.iter().any(|&v| v != 0));
        for band in [1, 3, 16, 33] {
            assert_eq!(render(&items, 100, 80, band), whole, "band height {band}");
        }
    }

    #[test]
    fn scaling_keeps_the_scale_bar_calibrated() {
        let item = OverlayItem::ScaleBar {
            um_per_px: 0.5,
            style: ScaleBarStyle::default(),
        };
        match item.scaled(0.25) {
            OverlayItem::ScaleBar { um_per_px, style } => {
                assert!((um_per_px - 2.0).abs() < 1e-12);
                assert!((style.font_size - 3.5).abs() < 1e-6);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn items_round_trip_through_json() {
        let item = OverlayItem::ScaleBar {
            um_per_px: 0.25,
            style: ScaleBarStyle::default(),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert_eq!(serde_json::from_str::<OverlayItem>(&json).unwrap(), item);
    }
}
