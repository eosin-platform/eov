//! Text labels of overlay objects.
//!
//! On screen, labels are drawn by Slint (crisp text in the UI font at any
//! scale factor); exports draw the same list with glyphs. Placement is
//! computed here so both agree.

use super::geometry::{self, Point};
use super::raster::{FrameScene, ViewTransform, resolve_style};
use super::store::ObjectKey;
use eov_plugin_api::protocol::{Rgba, Shape};

/// Upper bound on labels per pane, to keep the UI responsive.
pub const MAX_LABELS: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelAnchor {
    /// `(x, y)` is the label's top-left corner.
    TopLeft,
    /// `(x, y)` is the bottom-centre of the label (drawn above a point).
    Above,
    /// `(x, y)` is the left-middle of the label.
    Right,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Logical pixels relative to the target's top-left corner.
    pub x: f32,
    pub y: f32,
    pub anchor: LabelAnchor,
    pub text: String,
    pub color: Rgba,
    /// Draw a rounded backdrop behind the text.
    pub backdrop: bool,
    pub font_size: f32,
}

/// Where a shape's label goes (level-0 point and anchor).
pub fn label_anchor(shape: &Shape) -> (Point, LabelAnchor) {
    match shape {
        Shape::Point { x, y } => ([*x, *y], LabelAnchor::Right),
        Shape::Text { x, y, .. } => ([*x, *y], LabelAnchor::TopLeft),
        Shape::Polyline { points } => (polyline_midpoint(points), LabelAnchor::Above),
        _ => {
            let b = shape.bounds();
            ([b.x + b.width / 2.0, b.y], LabelAnchor::Above)
        }
    }
}

/// The point halfway along a polyline.
pub fn polyline_midpoint(points: &[Point]) -> Point {
    let total = geometry::polyline_length(points);
    let mut remaining = total / 2.0;
    for w in points.windows(2) {
        let len = geometry::distance(w[0], w[1]);
        if len >= remaining && len > 0.0 {
            let t = remaining / len;
            return [
                w[0][0] + t * (w[1][0] - w[0][0]),
                w[0][1] + t * (w[1][1] - w[0][1]),
            ];
        }
        remaining -= len;
    }
    points.first().copied().unwrap_or([0.0, 0.0])
}

fn to_logical(view: &ViewTransform, p: Point) -> (f32, f32) {
    (
        ((p[0] - view.left) * view.zoom) as f32,
        ((p[1] - view.top) * view.zoom) as f32,
    )
}

/// Collects the labels of everything visible in `scene`, in draw order.
pub fn collect_labels(view: &ViewTransform, scene: &FrameScene) -> Vec<Label> {
    let visible = view.visible_rect();
    let mut labels = Vec::new();
    let in_view = |x: f32, y: f32| {
        x >= -200.0
            && y >= -50.0
            && f64::from(x) <= view.width + 200.0
            && f64::from(y) <= view.height + 50.0
    };
    for layer in &scene.layers {
        if !layer.spec.visible || layer.spec.min_zoom.is_some_and(|min| view.zoom < min) {
            continue;
        }
        for object in layer.visible_objects(&visible, view.zoom) {
            let shape_has_text = matches!(object.object.shape, Shape::Text { .. });
            if object.object.label.is_none() && !shape_has_text {
                continue;
            }
            let key = ObjectKey {
                layer: layer.key.clone(),
                object_id: object.object.object_id.clone(),
            };
            let shape = scene.overrides.get(&key).unwrap_or(&object.object.shape);
            let style = resolve_style(layer, &object, scene.hovered.as_ref() == Some(&key));
            let (anchor_point, anchor) = label_anchor(shape);
            let (mut x, y) = to_logical(view, anchor_point);
            if anchor == LabelAnchor::Right {
                x += style.point_radius + 4.0;
            }
            if !in_view(x, y) {
                continue;
            }
            if let Shape::Text { text, .. } = shape {
                labels.push(Label {
                    x,
                    y,
                    anchor,
                    text: text.clone(),
                    color: style.text_color,
                    backdrop: false,
                    font_size: style.font_size,
                });
            }
            if let Some(text) = &object.object.label {
                labels.push(Label {
                    x,
                    y: if anchor == LabelAnchor::Above {
                        y - 4.0
                    } else {
                        y
                    },
                    anchor: if shape_has_text {
                        LabelAnchor::TopLeft
                    } else {
                        anchor
                    },
                    text: text.clone(),
                    color: style.text_color,
                    backdrop: true,
                    font_size: style.font_size,
                });
            }
            if labels.len() >= MAX_LABELS {
                return labels;
            }
        }
    }
    for transient in &scene.transients {
        let Some(text) = &transient.label else {
            continue;
        };
        let (anchor_point, anchor) = label_anchor(&transient.shape);
        let (x, y) = to_logical(view, anchor_point);
        labels.push(Label {
            x,
            y: y - 4.0,
            anchor,
            text: text.clone(),
            color: transient.style.text_color,
            backdrop: true,
            font_size: transient.style.font_size,
        });
    }
    labels.truncate(MAX_LABELS);
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::raster::Transient;
    use crate::overlay::store::{OverlayStore, Owner};
    use eov_plugin_api::protocol::{LayerSpec, OverlayObject, Style};

    fn view() -> ViewTransform {
        ViewTransform {
            left: 0.0,
            top: 0.0,
            zoom: 2.0,
            width: 200.0,
            height: 200.0,
            scale: 1.0,
            mpp: None,
        }
    }

    #[test]
    fn midpoints_follow_arc_length() {
        let mid = polyline_midpoint(&[[0.0, 0.0], [10.0, 0.0], [10.0, 30.0]]);
        assert_eq!(mid, [10.0, 10.0]);
    }

    #[test]
    fn labels_are_collected_for_visible_objects() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("p");
        store
            .create_layer(owner.clone(), LayerSpec::new("l", 1))
            .unwrap();
        store
            .upsert(
                &owner,
                "l",
                vec![
                    OverlayObject::new("a", Shape::Point { x: 10.0, y: 10.0 }).with_label("Tumour"),
                    OverlayObject::new("b", Shape::Point { x: 5000.0, y: 10.0 }).with_label("Far"),
                    OverlayObject::new(
                        "t",
                        Shape::Text {
                            x: 20.0,
                            y: 20.0,
                            text: "Note".into(),
                        },
                    ),
                ],
                false,
            )
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            transients: vec![Transient {
                label: Some("12.5 µm".into()),
                ..Transient::new(
                    Shape::Polyline {
                        points: vec![[0.0, 50.0], [20.0, 50.0]],
                    },
                    Style::default(),
                )
            }],
            ..Default::default()
        };
        let labels = collect_labels(&view(), &scene);
        let texts: Vec<_> = labels.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["Tumour", "Note", "12.5 µm"]);
        // Point labels sit to the right of the marker, in logical pixels.
        assert_eq!(labels[0].anchor, LabelAnchor::Right);
        assert!(labels[0].x > 20.0);
        assert_eq!((labels[2].x, labels[2].anchor), (20.0, LabelAnchor::Above));
    }
}
