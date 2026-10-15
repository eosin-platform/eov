//! Geometry helpers shared by overlay rendering, hit-testing, editing and
//! measurement. All coordinates are level-0 pixels unless noted.

use eov_plugin_api::protocol::{Rect, Shape};

pub type Point = [f64; 2];

pub fn distance(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Distance from `p` to the segment `a`–`b`.
pub fn distance_to_segment(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len_sq = dx * dx + dy * dy;
    if len_sq <= f64::EPSILON {
        return distance(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0);
    distance(p, [a[0] + t * dx, a[1] + t * dy])
}

/// Even-odd point-in-ring test.
pub fn point_in_ring(p: Point, ring: &[Point]) -> bool {
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i][0], ring[i][1]);
        let (xj, yj) = (ring[j][0], ring[j][1]);
        if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Point-in-polygon with holes (even-odd across all rings).
pub fn point_in_polygon(p: Point, exterior: &[Point], holes: &[Vec<Point>]) -> bool {
    point_in_ring(p, exterior) && !holes.iter().any(|hole| point_in_ring(p, hole))
}

/// Signed shoelace area of a ring (positive for clockwise in y-down space).
pub fn ring_signed_area(ring: &[Point]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Length of an open polyline.
pub fn polyline_length(points: &[Point]) -> f64 {
    points.windows(2).map(|w| distance(w[0], w[1])).sum()
}

/// Perimeter of a closed ring.
pub fn ring_perimeter(ring: &[Point]) -> f64 {
    if ring.len() < 2 {
        return 0.0;
    }
    polyline_length(ring) + distance(ring[ring.len() - 1], ring[0])
}

/// Ramanujan's approximation of an ellipse's circumference.
pub fn ellipse_perimeter(rx: f64, ry: f64) -> f64 {
    let (a, b) = (rx.abs(), ry.abs());
    if a + b <= 0.0 {
        return 0.0;
    }
    let h = ((a - b) / (a + b)).powi(2);
    std::f64::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()))
}

/// Area of a shape in level-0 pixels² (0 for open shapes).
pub fn shape_area(shape: &Shape) -> f64 {
    match shape {
        Shape::Polygon { exterior, holes } => {
            let outer = ring_signed_area(exterior).abs();
            let inner: f64 = holes.iter().map(|h| ring_signed_area(h).abs()).sum();
            (outer - inner).max(0.0)
        }
        Shape::Rectangle { width, height, .. } => width.abs() * height.abs(),
        Shape::Ellipse { rx, ry, .. } => std::f64::consts::PI * rx.abs() * ry.abs(),
        _ => 0.0,
    }
}

/// Length of open shapes, perimeter of closed ones, in level-0 pixels.
pub fn shape_length(shape: &Shape) -> f64 {
    match shape {
        Shape::Polyline { points } => polyline_length(points),
        Shape::Polygon { exterior, .. } => ring_perimeter(exterior),
        Shape::Rectangle { width, height, .. } => 2.0 * (width.abs() + height.abs()),
        Shape::Ellipse { rx, ry, .. } => ellipse_perimeter(*rx, *ry),
        _ => 0.0,
    }
}

/// Converts a length measured in level-0 pixels along direction `(dx, dy)`
/// to micrometres using possibly anisotropic pixel spacing.
pub fn segment_length_um(a: Point, b: Point, mpp_x: f64, mpp_y: f64) -> f64 {
    ((b[0] - a[0]) * mpp_x).hypot((b[1] - a[1]) * mpp_y)
}

/// Physical length of a shape's outline, honouring anisotropic spacing.
pub fn shape_length_um(shape: &Shape, mpp_x: f64, mpp_y: f64) -> f64 {
    let open = |points: &[Point]| -> f64 {
        points
            .windows(2)
            .map(|w| segment_length_um(w[0], w[1], mpp_x, mpp_y))
            .sum()
    };
    match shape {
        Shape::Polyline { points } => open(points),
        Shape::Polygon { exterior, .. } => {
            let closing = match (exterior.first(), exterior.last()) {
                (Some(first), Some(last)) if exterior.len() > 1 => {
                    segment_length_um(*last, *first, mpp_x, mpp_y)
                }
                _ => 0.0,
            };
            open(exterior) + closing
        }
        Shape::Rectangle { width, height, .. } => {
            2.0 * (width.abs() * mpp_x + height.abs() * mpp_y)
        }
        Shape::Ellipse { rx, ry, .. } => ellipse_perimeter(rx * mpp_x, ry * mpp_y),
        _ => 0.0,
    }
}

/// Physical area of a closed shape in µm² (anisotropic spacing scales areas
/// by `mpp_x·mpp_y`).
pub fn shape_area_um2(shape: &Shape, mpp_x: f64, mpp_y: f64) -> f64 {
    shape_area(shape) * mpp_x * mpp_y
}

/// The angle at `vertex` between the rays to `a` and `b`, in degrees
/// (0–180), honouring anisotropic spacing.
pub fn angle_deg(a: Point, vertex: Point, b: Point, mpp_x: f64, mpp_y: f64) -> f64 {
    let u = [(a[0] - vertex[0]) * mpp_x, (a[1] - vertex[1]) * mpp_y];
    let v = [(b[0] - vertex[0]) * mpp_x, (b[1] - vertex[1]) * mpp_y];
    let (lu, lv) = (u[0].hypot(u[1]), v[0].hypot(v[1]));
    if lu <= f64::EPSILON || lv <= f64::EPSILON {
        return 0.0;
    }
    let cos = ((u[0] * v[0] + u[1] * v[1]) / (lu * lv)).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

/// A closed polygon approximating an ellipse.
pub fn ellipse_ring(cx: f64, cy: f64, rx: f64, ry: f64, segments: usize) -> Vec<Point> {
    let segments = segments.max(8);
    (0..segments)
        .map(|i| {
            let t = i as f64 / segments as f64 * std::f64::consts::TAU;
            [cx + rx * t.cos(), cy + ry * t.sin()]
        })
        .collect()
}

/// The outline of a shape as points, and whether it is closed.
pub fn outline(shape: &Shape) -> (Vec<Point>, bool) {
    match shape {
        Shape::Point { x, y } | Shape::Text { x, y, .. } => (vec![[*x, *y]], false),
        Shape::Polyline { points } => (points.clone(), false),
        Shape::Polygon { exterior, .. } => (exterior.clone(), true),
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => (
            vec![
                [*x, *y],
                [x + width, *y],
                [x + width, y + height],
                [*x, y + height],
            ],
            true,
        ),
        Shape::Ellipse { cx, cy, rx, ry } => (ellipse_ring(*cx, *cy, *rx, *ry, 96), true),
    }
}

/// Douglas–Peucker simplification of an open polyline.
pub fn simplify(points: &[Point], tolerance: f64) -> Vec<Point> {
    if points.len() <= 2 || tolerance <= 0.0 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];
    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let (mut max_d, mut max_i) = (0.0, start);
        for i in start + 1..end {
            let d = distance_to_segment(points[i], points[start], points[end]);
            if d > max_d {
                max_d = d;
                max_i = i;
            }
        }
        if max_d > tolerance {
            keep[max_i] = true;
            stack.push((start, max_i));
            stack.push((max_i, end));
        }
    }
    points
        .iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}

/// Whether a closed outline may be drawn as a polygon (at least three
/// distinct vertices and a non-zero area).
pub fn is_usable_ring(ring: &[Point]) -> bool {
    ring.len() >= 3 && ring_signed_area(ring).abs() > f64::EPSILON
}

pub fn rect_intersects(a: &Rect, b: &Rect) -> bool {
    a.x <= b.x + b.width && b.x <= a.x + a.width && a.y <= b.y + b.height && b.y <= a.y + a.height
}

pub fn expand_rect(r: &Rect, margin: f64) -> Rect {
    Rect {
        x: r.x - margin,
        y: r.y - margin,
        width: r.width + 2.0 * margin,
        height: r.height + 2.0 * margin,
    }
}

/// Normalises a rectangle given by two corners.
pub fn rect_from_corners(a: Point, b: Point) -> Rect {
    Rect {
        x: a[0].min(b[0]),
        y: a[1].min(b[1]),
        width: (a[0] - b[0]).abs(),
        height: (a[1] - b[1]).abs(),
    }
}

/// Sutherland–Hodgman clipping of a closed ring against an axis-aligned
/// rectangle (keeps far-away vertices out of the rasteriser's f32 range).
pub fn clip_ring_to_rect(ring: &[Point], rect: &Rect) -> Vec<Point> {
    let (x0, y0, x1, y1) = (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
    let mut output: Vec<Point> = ring.to_vec();
    // (inside test, intersection) per edge.
    type Edge = (fn(Point, f64) -> bool, fn(Point, Point, f64) -> Point, f64);
    let edges: [Edge; 4] = [
        (|p, x| p[0] >= x, |a, b, x| lerp_x(a, b, x), x0),
        (|p, x| p[0] <= x, |a, b, x| lerp_x(a, b, x), x1),
        (|p, y| p[1] >= y, |a, b, y| lerp_y(a, b, y), y0),
        (|p, y| p[1] <= y, |a, b, y| lerp_y(a, b, y), y1),
    ];
    for (inside, intersect, bound) in edges {
        if output.is_empty() {
            break;
        }
        let input = std::mem::take(&mut output);
        let mut prev = *input.last().expect("non-empty");
        for &current in &input {
            let (cur_in, prev_in) = (inside(current, bound), inside(prev, bound));
            if cur_in {
                if !prev_in {
                    output.push(intersect(prev, current, bound));
                }
                output.push(current);
            } else if prev_in {
                output.push(intersect(prev, current, bound));
            }
            prev = current;
        }
    }
    output
}

fn lerp_x(a: Point, b: Point, x: f64) -> Point {
    let t = if (b[0] - a[0]).abs() <= f64::EPSILON {
        0.0
    } else {
        (x - a[0]) / (b[0] - a[0])
    };
    [x, a[1] + t * (b[1] - a[1])]
}

fn lerp_y(a: Point, b: Point, y: f64) -> Point {
    let t = if (b[1] - a[1]).abs() <= f64::EPSILON {
        0.0
    } else {
        (y - a[1]) / (b[1] - a[1])
    };
    [a[0] + t * (b[0] - a[0]), y]
}

/// Liang–Barsky clipping of a segment against a rectangle.
pub fn clip_segment_to_rect(a: Point, b: Point, rect: &Rect) -> Option<(Point, Point)> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    let checks = [
        (-dx, a[0] - rect.x),
        (dx, rect.x + rect.width - a[0]),
        (-dy, a[1] - rect.y),
        (dy, rect.y + rect.height - a[1]),
    ];
    for (p, q) in checks {
        if p.abs() <= f64::EPSILON {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return None;
            }
            t1 = t1.min(r);
        }
    }
    Some((
        [a[0] + t0 * dx, a[1] + t0 * dy],
        [a[0] + t1 * dx, a[1] + t1 * dy],
    ))
}

/// Translates a shape by `(dx, dy)`.
pub fn translate_shape(shape: &Shape, dx: f64, dy: f64) -> Shape {
    let mv = |p: &Point| [p[0] + dx, p[1] + dy];
    match shape {
        Shape::Point { x, y } => Shape::Point {
            x: x + dx,
            y: y + dy,
        },
        Shape::Text { x, y, text } => Shape::Text {
            x: x + dx,
            y: y + dy,
            text: text.clone(),
        },
        Shape::Polyline { points } => Shape::Polyline {
            points: points.iter().map(mv).collect(),
        },
        Shape::Polygon { exterior, holes } => Shape::Polygon {
            exterior: exterior.iter().map(mv).collect(),
            holes: holes
                .iter()
                .map(|ring| ring.iter().map(mv).collect())
                .collect(),
        },
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => Shape::Rectangle {
            x: x + dx,
            y: y + dy,
            width: *width,
            height: *height,
        },
        Shape::Ellipse { cx, cy, rx, ry } => Shape::Ellipse {
            cx: cx + dx,
            cy: cy + dy,
            rx: *rx,
            ry: *ry,
        },
    }
}

/// The editable vertices of a shape: polygon exterior / polyline points,
/// the four corners of a rectangle, the four axis extremes of an ellipse.
pub fn shape_handles(shape: &Shape) -> Vec<Point> {
    match shape {
        Shape::Point { x, y } => vec![[*x, *y]],
        Shape::Polyline { points } => points.clone(),
        Shape::Polygon { exterior, .. } => exterior.clone(),
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => vec![
            [*x, *y],
            [x + width, *y],
            [x + width, y + height],
            [*x, y + height],
        ],
        Shape::Ellipse { cx, cy, rx, ry } => vec![
            [cx + rx, *cy],
            [*cx, cy + ry],
            [cx - rx, *cy],
            [*cx, cy - ry],
        ],
        Shape::Text { .. } => Vec::new(),
    }
}

/// Moves handle `index` (see [`shape_handles`]) to `to`.
pub fn move_handle(shape: &Shape, index: usize, to: Point) -> Shape {
    let mut shape = shape.clone();
    match &mut shape {
        Shape::Point { x, y } => {
            *x = to[0];
            *y = to[1];
        }
        Shape::Polyline { points } => {
            if let Some(p) = points.get_mut(index) {
                *p = to;
            }
        }
        Shape::Polygon { exterior, .. } => {
            if let Some(p) = exterior.get_mut(index) {
                *p = to;
            }
        }
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => {
            // The opposite corner stays fixed.
            let corners = [
                [*x, *y],
                [*x + *width, *y],
                [*x + *width, *y + *height],
                [*x, *y + *height],
            ];
            let anchor = corners[(index + 2) % 4];
            let r = rect_from_corners(anchor, to);
            *x = r.x;
            *y = r.y;
            *width = r.width;
            *height = r.height;
        }
        Shape::Ellipse { cx, cy, rx, ry } => match index {
            0 | 2 => *rx = (to[0] - *cx).abs(),
            _ => *ry = (to[1] - *cy).abs(),
        },
        Shape::Text { .. } => {}
    }
    shape
}

/// Inserts a vertex after `index` (polylines and polygons).
pub fn insert_vertex(shape: &Shape, index: usize, at: Point) -> Shape {
    let mut shape = shape.clone();
    match &mut shape {
        Shape::Polyline { points } => points.insert((index + 1).min(points.len()), at),
        Shape::Polygon { exterior, .. } => exterior.insert((index + 1).min(exterior.len()), at),
        _ => {}
    }
    shape
}

/// Removes vertex `index` when enough vertices remain (3 for polygons, 2 for
/// polylines); returns `None` otherwise.
pub fn remove_vertex(shape: &Shape, index: usize) -> Option<Shape> {
    let mut shape = shape.clone();
    match &mut shape {
        Shape::Polyline { points } if points.len() > 2 && index < points.len() => {
            points.remove(index);
        }
        Shape::Polygon { exterior, .. } if exterior.len() > 3 && index < exterior.len() => {
            exterior.remove(index);
        }
        _ => return None,
    }
    Some(shape)
}

/// The segments a vertex can be inserted into: `(index, a, b)` where the
/// new vertex goes after `index`.
pub fn shape_edges(shape: &Shape) -> Vec<(usize, Point, Point)> {
    match shape {
        Shape::Polyline { points } => points
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i, w[0], w[1]))
            .collect(),
        Shape::Polygon { exterior, .. } if exterior.len() >= 2 => (0..exterior.len())
            .map(|i| (i, exterior[i], exterior[(i + 1) % exterior.len()]))
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: [Point; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

    #[test]
    fn polygon_containment_honours_holes() {
        let hole = vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]];
        assert!(point_in_polygon(
            [2.0, 2.0],
            &SQUARE,
            std::slice::from_ref(&hole)
        ));
        assert!(!point_in_polygon([5.0, 5.0], &SQUARE, &[hole]));
        assert!(!point_in_polygon([11.0, 5.0], &SQUARE, &[]));
    }

    #[test]
    fn areas_and_lengths() {
        let poly = Shape::Polygon {
            exterior: SQUARE.to_vec(),
            holes: vec![vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]],
        };
        assert_eq!(shape_area(&poly), 96.0);
        assert_eq!(shape_length(&poly), 40.0);
        let line = Shape::Polyline {
            points: vec![[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]],
        };
        assert_eq!(shape_length(&line), 11.0);
        assert!((ellipse_perimeter(1.0, 1.0) - std::f64::consts::TAU).abs() < 1e-9);
    }

    #[test]
    fn anisotropic_lengths_use_both_axes() {
        // 10 px right and 10 px down with 0.5 µm/px in x and 0.25 µm/px in y.
        let um = segment_length_um([0.0, 0.0], [10.0, 10.0], 0.5, 0.25);
        assert!((um - (25.0f64 + 6.25).sqrt()).abs() < 1e-9);
        let rect = Shape::Rectangle {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 4.0,
        };
        assert!((shape_area_um2(&rect, 0.5, 0.25) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn angles() {
        let a = angle_deg([10.0, 0.0], [0.0, 0.0], [0.0, 10.0], 1.0, 1.0);
        assert!((a - 90.0).abs() < 1e-9);
        let flat = angle_deg([10.0, 0.0], [0.0, 0.0], [-10.0, 0.0], 1.0, 1.0);
        assert!((flat - 180.0).abs() < 1e-9);
    }

    #[test]
    fn simplification_keeps_corners() {
        let points: Vec<Point> = (0..=10)
            .map(|i| [i as f64, 0.0])
            .chain((1..=10).map(|i| [10.0, i as f64]))
            .collect();
        assert_eq!(
            simplify(&points, 0.1),
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
        );
    }

    #[test]
    fn ring_clipping() {
        let clip = Rect {
            x: 5.0,
            y: 5.0,
            width: 100.0,
            height: 100.0,
        };
        let clipped = clip_ring_to_rect(&SQUARE, &clip);
        assert!((ring_signed_area(&clipped).abs() - 25.0).abs() < 1e-9);
        let outside = Rect {
            x: 50.0,
            y: 50.0,
            width: 1.0,
            height: 1.0,
        };
        assert!(clip_ring_to_rect(&SQUARE, &outside).is_empty());
    }

    #[test]
    fn segment_clipping() {
        let r = Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let (a, b) = clip_segment_to_rect([-10.0, 5.0], [20.0, 5.0], &r).unwrap();
        assert_eq!((a, b), ([0.0, 5.0], [10.0, 5.0]));
        assert!(clip_segment_to_rect([-10.0, -5.0], [20.0, -5.0], &r).is_none());
    }

    #[test]
    fn handles_edit_rectangles_about_the_opposite_corner() {
        let rect = Shape::Rectangle {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let moved = move_handle(&rect, 2, [20.0, 15.0]);
        assert_eq!(
            moved,
            Shape::Rectangle {
                x: 0.0,
                y: 0.0,
                width: 20.0,
                height: 15.0
            }
        );
        // Dragging past the anchor flips without negative sizes.
        let flipped = move_handle(&rect, 2, [-5.0, -5.0]);
        assert_eq!(
            flipped,
            Shape::Rectangle {
                x: -5.0,
                y: -5.0,
                width: 5.0,
                height: 5.0
            }
        );
    }

    #[test]
    fn vertex_insertion_and_removal_limits() {
        let tri = Shape::Polygon {
            exterior: vec![[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]],
            holes: vec![],
        };
        assert!(remove_vertex(&tri, 0).is_none());
        let quad = insert_vertex(&tri, 0, [5.0, -1.0]);
        assert_eq!(shape_handles(&quad).len(), 4);
        assert_eq!(shape_handles(&quad)[1], [5.0, -1.0]);
        assert_eq!(shape_handles(&remove_vertex(&quad, 1).unwrap()).len(), 3);
    }
}
