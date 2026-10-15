//! Rasterisation of overlay layers with tiny-skia.
//!
//! The same code draws the viewport overlay (on a worker thread, at the
//! pane's physical resolution) and the overlays of exported images, so what
//! is exported matches what is on screen.

use super::geometry::{self, Point};
use super::store::{HostDecoration, Layer, ObjectKey, StoredObject, StoredRaster, size_to_screen};
use eov_common::viewport::Orientation;
use eov_plugin_api::protocol::{Dash, Rect, Rgba, Shape, SizeUnit, Style};
use std::collections::HashMap;
use std::sync::Arc;
use tiny_skia::{
    BlendMode, Color, FillRule, FilterQuality, LineCap, LineJoin, Paint, PathBuilder, Pixmap,
    PixmapPaint, PixmapRef, Stroke, StrokeDash, Transform,
};

/// Objects larger than this on screen (physical px) are clipped to the view
/// before rasterising, keeping coordinates well inside f32 precision.
const CLIP_THRESHOLD_PX: f64 = 16_384.0;
/// Margin kept around the view when clipping, so clip edges stay off screen.
const CLIP_MARGIN_PX: f64 = 64.0;
/// Below this many physical pixels an object is drawn as a dot.
const LOD_DOT_PX: f64 = 1.5;
/// With more visible point markers than this in one layer, markers are drawn
/// as squares (no anti-aliasing) to keep frames fast.
const FAST_POINTS_THRESHOLD: usize = 20_000;

/// Maps level-0 coordinates to the pixels of a raster target. A turned
/// target (`orientation`) maps through the target drawn axis-aligned with
/// the slide, turned about the centres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    /// Level-0 coordinates of the top-left corner of the target drawn
    /// axis-aligned with the slide (the visible bounding box).
    pub left: f64,
    pub top: f64,
    /// Logical pixels per level-0 pixel.
    pub zoom: f64,
    /// Logical size of the target.
    pub width: f64,
    pub height: f64,
    /// Physical pixels per logical pixel.
    pub scale: f64,
    /// Average µm per level-0 pixel, for µm-sized strokes and markers.
    pub mpp: Option<f64>,
    /// How the slide is turned on the target.
    pub orientation: Orientation,
}

impl ViewTransform {
    /// Level-0 rectangle covered by the target (its bounding box).
    pub fn visible_rect(self) -> Rect {
        let zoom = self.zoom.max(1e-12);
        let (width, height) = self.orientation.swap(self.width, self.height);
        Rect {
            x: self.left,
            y: self.top,
            width: width / zoom,
            height: height / zoom,
        }
    }

    /// Physical pixel position of level-0 point `p`.
    fn px(self, p: Point) -> (f64, f64) {
        let k = self.px_per_image();
        let (ax, ay) = ((p[0] - self.left) * k, (p[1] - self.top) * k);
        if self.orientation.is_identity() {
            return (ax, ay);
        }
        let (w, h) = (self.width * self.scale, self.height * self.scale);
        let (aw, ah) = self.orientation.swap(w, h);
        let (x, y) = self.orientation.to_screen_xy(ax - aw / 2.0, ay - ah / 2.0);
        (x + w / 2.0, y + h / 2.0)
    }

    /// Logical position of level-0 point `p` on the target.
    pub fn to_logical(self, p: Point) -> (f64, f64) {
        let (x, y) = self.px(p);
        (x / self.scale, y / self.scale)
    }

    /// The level-0 point at logical position `(x, y)` of the target.
    pub fn image_at_logical(self, x: f64, y: f64) -> Point {
        let k = self.px_per_image().max(1e-12);
        let (px, py) = (x * self.scale, y * self.scale);
        let (ax, ay) = if self.orientation.is_identity() {
            (px, py)
        } else {
            let (w, h) = (self.width * self.scale, self.height * self.scale);
            let (aw, ah) = self.orientation.swap(w, h);
            let (dx, dy) = self.orientation.to_image_xy(px - w / 2.0, py - h / 2.0);
            (dx + aw / 2.0, dy + ah / 2.0)
        };
        [self.left + ax / k, self.top + ay / k]
    }

    /// Level-0 → physical pixel as an affine transform (for drawing
    /// rasters), composed after `inner` (source pixel → level-0).
    fn affine_after(self, inner: Transform) -> Transform {
        let origin = self.px([0.0, 0.0]);
        let x_axis = self.px([1.0, 0.0]);
        let y_axis = self.px([0.0, 1.0]);
        let outer = Transform::from_row(
            (x_axis.0 - origin.0) as f32,
            (x_axis.1 - origin.1) as f32,
            (y_axis.0 - origin.0) as f32,
            (y_axis.1 - origin.1) as f32,
            origin.0 as f32,
            origin.1 as f32,
        );
        inner.post_concat(outer)
    }

    /// The part of the target showing physical rows `row..row + rows`, as a
    /// target of its own (exports draw overlays band by band).
    pub fn band(self, row: u32, rows: u32) -> ViewTransform {
        let k = self.px_per_image();
        let (width, height) = self.physical_size();
        let (aligned_width, aligned_height) = self.orientation.swap(width, height);
        let (x, y, _, _) =
            self.orientation
                .aligned_rect(aligned_width, aligned_height, 0, row, width, rows);
        ViewTransform {
            left: self.left + f64::from(x) / k,
            top: self.top + f64::from(y) / k,
            width: f64::from(width) / self.scale,
            height: f64::from(rows) / self.scale,
            ..self
        }
    }

    pub fn physical_size(self) -> (u32, u32) {
        (
            (self.width * self.scale).round().clamp(1.0, 16_384.0) as u32,
            (self.height * self.scale).round().clamp(1.0, 16_384.0) as u32,
        )
    }

    /// Physical pixels per level-0 pixel.
    pub fn px_per_image(self) -> f64 {
        self.zoom * self.scale
    }

    pub fn to_px(self, p: Point) -> (f32, f32) {
        let (x, y) = self.px(p);
        (x as f32, y as f32)
    }

    /// Style size → physical pixels.
    fn size_px(self, value: f32, unit: SizeUnit) -> f32 {
        (size_to_screen(value, unit, self.zoom, self.mpp) * self.scale) as f32
    }
}

/// Something drawn on top of the layers that is not stored (tool previews,
/// edit handles, rubber bands).
#[derive(Clone, Debug, PartialEq)]
pub struct Transient {
    pub shape: Shape,
    pub style: Style,
    pub decoration: HostDecoration,
    /// Live text shown with the shape (e.g. a measurement in progress).
    pub label: Option<String>,
}

impl Transient {
    pub fn new(shape: Shape, style: Style) -> Self {
        Self {
            shape,
            style,
            decoration: HostDecoration::default(),
            label: None,
        }
    }
}

/// Everything needed to draw one frame of one pane.
#[derive(Clone, Debug, Default)]
pub struct FrameScene {
    pub layers: Vec<Arc<Layer>>,
    pub hovered: Option<ObjectKey>,
    /// Shapes shown instead of the stored ones (edits in progress).
    pub overrides: HashMap<ObjectKey, Shape>,
    pub transients: Vec<Transient>,
}

impl FrameScene {
    pub fn is_empty(&self) -> bool {
        self.transients.is_empty()
            && self
                .layers
                .iter()
                .all(|layer| !layer.spec.visible || layer.is_empty())
    }
}

/// Renders `scene` into a new transparent pixmap of the view's physical
/// size. Returns `None` if the size is invalid.
pub fn render(view: &ViewTransform, scene: &FrameScene) -> Option<Pixmap> {
    let (width, height) = view.physical_size();
    let mut pixmap = Pixmap::new(width, height)?;
    render_into(&mut pixmap, view, scene);
    Some(pixmap)
}

/// Renders `scene` onto `target` (premultiplied), which must have the view's
/// physical size.
pub fn render_into(target: &mut Pixmap, view: &ViewTransform, scene: &FrameScene) {
    let mut scratch: Option<Pixmap> = None;
    let visible = view.visible_rect();
    for layer in &scene.layers {
        if !layer.spec.visible
            || layer.spec.opacity <= 0.0
            || layer.spec.min_zoom.is_some_and(|min| view.zoom < min)
            || layer.is_empty()
        {
            continue;
        }
        if layer.spec.opacity >= 0.999 {
            draw_layer(target, view, &visible, layer, scene);
        } else {
            let buffer = match &mut scratch {
                Some(buffer) => {
                    buffer.fill(Color::TRANSPARENT);
                    buffer
                }
                None => match Pixmap::new(target.width(), target.height()) {
                    Some(buffer) => scratch.insert(buffer),
                    None => continue,
                },
            };
            draw_layer(buffer, view, &visible, layer, scene);
            target.draw_pixmap(
                0,
                0,
                buffer.as_ref(),
                &PixmapPaint {
                    opacity: layer.spec.opacity,
                    blend_mode: BlendMode::SourceOver,
                    quality: FilterQuality::Nearest,
                },
                Transform::identity(),
                None,
            );
        }
    }
    for transient in &scene.transients {
        draw_shape(
            target,
            view,
            &transient.shape,
            &transient.style,
            transient.decoration,
            false,
        );
    }
}

fn draw_layer(
    target: &mut Pixmap,
    view: &ViewTransform,
    visible: &Rect,
    layer: &Layer,
    scene: &FrameScene,
) {
    for raster in layer.visible_rasters(visible) {
        draw_raster(target, view, &raster);
    }
    let objects = layer.visible_objects(visible, view.zoom);
    let fast_points = objects
        .iter()
        .filter(|o| matches!(o.object.shape, Shape::Point { .. }))
        .count()
        > FAST_POINTS_THRESHOLD;
    for object in objects {
        let key = ObjectKey {
            layer: layer.key.clone(),
            object_id: object.object.object_id.clone(),
        };
        let style = resolve_style(layer, &object, scene.hovered.as_ref() == Some(&key));
        let shape = scene.overrides.get(&key).unwrap_or(&object.object.shape);
        draw_object(target, view, shape, &style, &object, fast_points);
    }
}

/// The style an object is drawn with, including host selection and hover
/// highlighting.
pub fn resolve_style(layer: &Layer, object: &StoredObject, hovered: bool) -> Style {
    let base = object
        .object
        .style
        .clone()
        .unwrap_or_else(|| layer.spec.style.clone());
    if layer.is_selected(&object.object.object_id) {
        layer
            .spec
            .selected_style
            .clone()
            .unwrap_or_else(|| selected_variant(&base))
    } else if hovered || layer.highlighted.contains(&object.object.object_id) {
        layer
            .spec
            .hover_style
            .clone()
            .unwrap_or_else(|| hover_variant(&base))
    } else {
        base
    }
}

/// Default selection highlight: an amber outline, one pixel wider.
pub fn selected_variant(base: &Style) -> Style {
    Style {
        stroke: Rgba(0xFF, 0xD6, 0x0A, 0xFF),
        stroke_width: base.stroke_width + 1.0,
        fill: if base.fill.3 == 0 {
            base.fill
        } else {
            base.fill.with_alpha(base.fill.3.saturating_add(0x30))
        },
        ..base.clone()
    }
}

/// Default hover highlight: a lighter, wider outline.
pub fn hover_variant(base: &Style) -> Style {
    let lighten = |c: u8| c.saturating_add((255 - c) / 2);
    Style {
        stroke: Rgba(
            lighten(base.stroke.0),
            lighten(base.stroke.1),
            lighten(base.stroke.2),
            base.stroke.3.max(0xC0),
        ),
        stroke_width: base.stroke_width + 1.0,
        ..base.clone()
    }
}

fn draw_object(
    target: &mut Pixmap,
    view: &ViewTransform,
    shape: &Shape,
    style: &Style,
    object: &StoredObject,
    fast_points: bool,
) {
    if fast_points && let Shape::Point { x, y } = shape {
        let (px, py) = view.to_px([*x, *y]);
        let r = view.size_px(style.point_radius, style.size_unit).max(1.0);
        let color = if style.stroke.3 > 0 {
            style.stroke
        } else {
            style.fill
        };
        fill_rect_px(target, px - r, py - r, 2.0 * r, 2.0 * r, color);
        return;
    }
    draw_shape(target, view, shape, style, object.decoration, true);
}

fn paint_for(color: Rgba) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.0, color.1, color.2, color.3);
    paint.anti_alias = true;
    paint
}

fn fill_rect_px(target: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, color: Rgba) {
    if color.3 == 0 {
        return;
    }
    if let Some(rect) = tiny_skia::Rect::from_xywh(x, y, w.max(1.0), h.max(1.0)) {
        let mut paint = paint_for(color);
        paint.anti_alias = false;
        target.fill_rect(rect, &paint, Transform::identity(), None);
    }
}

fn stroke_for(style: &Style, width_px: f32) -> Stroke {
    let width = width_px.max(0.5);
    let dash = match style.dash {
        Dash::Solid => None,
        Dash::Dashed => StrokeDash::new(vec![width * 4.0, width * 3.0], 0.0),
        Dash::Dotted => StrokeDash::new(vec![width.max(1.0), width * 2.0], 0.0),
    };
    Stroke {
        width,
        line_cap: if style.dash == Dash::Dotted {
            LineCap::Round
        } else {
            LineCap::Butt
        },
        line_join: LineJoin::Round,
        dash,
        ..Stroke::default()
    }
}

/// Expanded view rectangle (level-0) used to clip very large shapes.
fn clip_rect(view: &ViewTransform) -> Rect {
    let margin = CLIP_MARGIN_PX / view.px_per_image().max(1e-12);
    geometry::expand_rect(&view.visible_rect(), margin)
}

fn needs_clipping(view: &ViewTransform, bounds: &Rect) -> bool {
    let k = view.px_per_image();
    bounds.width * k > CLIP_THRESHOLD_PX || bounds.height * k > CLIP_THRESHOLD_PX
}

fn push_ring(builder: &mut PathBuilder, view: &ViewTransform, ring: &[Point]) {
    let mut iter = ring.iter();
    let Some(first) = iter.next() else {
        return;
    };
    let (x, y) = view.to_px(*first);
    builder.move_to(x, y);
    for p in iter {
        let (x, y) = view.to_px(*p);
        builder.line_to(x, y);
    }
    builder.close();
}

/// Draws any shape with `style`. `lod` allows tiny objects to collapse to
/// dots.
pub fn draw_shape(
    target: &mut Pixmap,
    view: &ViewTransform,
    shape: &Shape,
    style: &Style,
    decoration: HostDecoration,
    lod: bool,
) {
    let bounds = shape.bounds();
    let k = view.px_per_image();
    let stroke_px = view.size_px(style.stroke_width, style.size_unit);

    if decoration.dim_outside {
        dim_outside(target, view, &bounds);
    }

    match shape {
        Shape::Point { x, y } => {
            let (px, py) = view.to_px([*x, *y]);
            let r = view.size_px(style.point_radius, style.size_unit).max(1.0);
            let Some(circle) = PathBuilder::from_circle(px, py, r) else {
                return;
            };
            if style.fill.3 > 0 {
                target.fill_path(
                    &circle,
                    &paint_for(style.fill),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
            if style.stroke.3 > 0 {
                // A dark halo keeps markers visible on any stain.
                let halo = Stroke {
                    width: stroke_px.max(1.0) + 2.0 * view.scale as f32,
                    ..Stroke::default()
                };
                target.stroke_path(
                    &circle,
                    &paint_for(Rgba(0x11, 0x13, 0x18, 0xC8)),
                    &halo,
                    Transform::identity(),
                    None,
                );
                target.stroke_path(
                    &circle,
                    &paint_for(style.stroke),
                    &stroke_for(style, stroke_px.max(1.0)),
                    Transform::identity(),
                    None,
                );
            }
        }
        // Text is laid out by the caller (Slint labels on screen, glyphs in
        // exports).
        Shape::Text { .. } => {}
        _ if lod && bounds.width * k < LOD_DOT_PX && bounds.height * k < LOD_DOT_PX => {
            let (px, py) = view.to_px([
                bounds.x + bounds.width / 2.0,
                bounds.y + bounds.height / 2.0,
            ]);
            let color = if style.stroke.3 > 0 {
                style.stroke
            } else {
                style.fill
            };
            let size = view.scale.max(1.0) as f32;
            fill_rect_px(target, px - size / 2.0, py - size / 2.0, size, size, color);
        }
        Shape::Polyline { points } => {
            if style.stroke.3 == 0 {
                return;
            }
            let mut builder = PathBuilder::new();
            if needs_clipping(view, &bounds) {
                let clip = clip_rect(view);
                for w in points.windows(2) {
                    if let Some((a, b)) = geometry::clip_segment_to_rect(w[0], w[1], &clip) {
                        let (ax, ay) = view.to_px(a);
                        let (bx, by) = view.to_px(b);
                        builder.move_to(ax, ay);
                        builder.line_to(bx, by);
                    }
                }
            } else {
                let mut iter = points.iter();
                if let Some(first) = iter.next() {
                    let (x, y) = view.to_px(*first);
                    builder.move_to(x, y);
                    for p in iter {
                        let (x, y) = view.to_px(*p);
                        builder.line_to(x, y);
                    }
                }
            }
            if let Some(path) = builder.finish() {
                let mut stroke = stroke_for(style, stroke_px);
                stroke.line_cap = LineCap::Round;
                target.stroke_path(
                    &path,
                    &paint_for(style.stroke),
                    &stroke,
                    Transform::identity(),
                    None,
                );
            }
        }
        Shape::Polygon { exterior, holes } => {
            let mut builder = PathBuilder::new();
            if needs_clipping(view, &bounds) {
                let clip = clip_rect(view);
                for ring in std::iter::once(exterior).chain(holes.iter()) {
                    let clipped = geometry::clip_ring_to_rect(ring, &clip);
                    if clipped.len() >= 3 {
                        push_ring(&mut builder, view, &clipped);
                    }
                }
            } else {
                for ring in std::iter::once(exterior).chain(holes.iter()) {
                    push_ring(&mut builder, view, ring);
                }
            }
            if let Some(path) = builder.finish() {
                fill_and_stroke(target, &path, style, stroke_px, decoration);
            }
        }
        Shape::Rectangle {
            x,
            y,
            width,
            height,
        } => {
            let ring = [
                [*x, *y],
                [x + width, *y],
                [x + width, y + height],
                [*x, y + height],
            ];
            let ring = if needs_clipping(view, &bounds) {
                geometry::clip_ring_to_rect(&ring, &clip_rect(view))
            } else {
                ring.to_vec()
            };
            if ring.len() < 3 {
                return;
            }
            let mut builder = PathBuilder::new();
            push_ring(&mut builder, view, &ring);
            if let Some(path) = builder.finish() {
                fill_and_stroke(target, &path, style, stroke_px, decoration);
            }
        }
        Shape::Ellipse { cx, cy, rx, ry } => {
            let path = if needs_clipping(view, &bounds) {
                // Enough segments that chords stay within a pixel.
                let segments = ((rx.max(*ry) * k).sqrt() * 4.0).clamp(64.0, 4096.0) as usize;
                let ring = geometry::ellipse_ring(*cx, *cy, *rx, *ry, segments);
                let clipped = geometry::clip_ring_to_rect(&ring, &clip_rect(view));
                if clipped.len() < 3 {
                    return;
                }
                let mut builder = PathBuilder::new();
                push_ring(&mut builder, view, &clipped);
                builder.finish()
            } else {
                // A quarter turn keeps the ellipse axis-aligned on screen.
                let (ax, ay) = view.to_px([cx - rx, cy - ry]);
                let (bx, by) = view.to_px([cx + rx, cy + ry]);
                let (x0, y0) = (ax.min(bx), ay.min(by));
                let (x1, y1) = (ax.max(bx), ay.max(by));
                tiny_skia::Rect::from_ltrb(x0, y0, x1.max(x0 + 0.01), y1.max(y0 + 0.01))
                    .and_then(PathBuilder::from_oval)
            };
            if let Some(path) = path {
                fill_and_stroke(target, &path, style, stroke_px, decoration);
            }
        }
    }
}

fn fill_and_stroke(
    target: &mut Pixmap,
    path: &tiny_skia::Path,
    style: &Style,
    stroke_px: f32,
    decoration: HostDecoration,
) {
    if style.fill.3 > 0 {
        target.fill_path(
            path,
            &paint_for(style.fill),
            FillRule::EvenOdd,
            Transform::identity(),
            None,
        );
    }
    if decoration.contrast_border {
        let width = stroke_px.max(1.0);
        let solid = Stroke {
            width: width * 3.0,
            line_join: LineJoin::Miter,
            ..Stroke::default()
        };
        target.stroke_path(
            path,
            &paint_for(Rgba(0, 0, 0, 0xFF)),
            &solid,
            Transform::identity(),
            None,
        );
        let inner = Stroke {
            width,
            line_join: LineJoin::Miter,
            ..Stroke::default()
        };
        target.stroke_path(
            path,
            &paint_for(Rgba(0xFF, 0xFF, 0xFF, 0xFF)),
            &inner,
            Transform::identity(),
            None,
        );
        let dashed = Stroke {
            width,
            line_join: LineJoin::Miter,
            dash: StrokeDash::new(vec![width * 6.0, width * 6.0], 0.0),
            ..Stroke::default()
        };
        target.stroke_path(
            path,
            &paint_for(Rgba(0, 0, 0, 0xFF)),
            &dashed,
            Transform::identity(),
            None,
        );
        return;
    }
    if style.stroke.3 > 0 && stroke_px > 0.0 {
        target.stroke_path(
            path,
            &paint_for(style.stroke),
            &stroke_for(style, stroke_px),
            Transform::identity(),
            None,
        );
    }
}

/// Darkens the whole target except `bounds` (the host ROI).
fn dim_outside(target: &mut Pixmap, view: &ViewTransform, bounds: &Rect) {
    let (w, h) = (target.width() as f32, target.height() as f32);
    let clip = clip_rect(view);
    let ring = [
        [bounds.x, bounds.y],
        [bounds.x + bounds.width, bounds.y],
        [bounds.x + bounds.width, bounds.y + bounds.height],
        [bounds.x, bounds.y + bounds.height],
    ];
    let ring = geometry::clip_ring_to_rect(&ring, &clip);
    let mut builder = PathBuilder::new();
    if let Some(full) = tiny_skia::Rect::from_xywh(0.0, 0.0, w, h) {
        builder.push_rect(full);
    }
    if ring.len() >= 3 {
        push_ring(&mut builder, view, &ring);
    }
    if let Some(path) = builder.finish() {
        let mut paint = paint_for(Rgba(0, 0, 0, 0x80));
        paint.anti_alias = false;
        target.fill_path(
            &path,
            &paint,
            FillRule::EvenOdd,
            Transform::identity(),
            None,
        );
    }
}

fn draw_raster(target: &mut Pixmap, view: &ViewTransform, raster: &StoredRaster) {
    let Some(source) = PixmapRef::from_bytes(&raster.premultiplied, raster.width, raster.height)
    else {
        return;
    };
    let k = view.px_per_image();
    let sx = raster.rect.width * k / f64::from(raster.width.max(1));
    let sy = raster.rect.height * k / f64::from(raster.height.max(1));
    let quality = if raster.smooth {
        FilterQuality::Bilinear
    } else {
        FilterQuality::Nearest
    };
    let paint = PixmapPaint {
        opacity: 1.0,
        blend_mode: BlendMode::SourceOver,
        quality,
    };
    // Source pixel → level-0 for the raster's pixels.
    let su = raster.rect.width / f64::from(raster.width.max(1));
    let sv = raster.rect.height / f64::from(raster.height.max(1));
    if sx <= 1.0 && sy <= 1.0 {
        let inner = Transform::from_row(
            su as f32,
            0.0,
            0.0,
            sv as f32,
            raster.rect.x as f32,
            raster.rect.y as f32,
        );
        target.draw_pixmap(0, 0, source, &paint, view.affine_after(inner), None);
        return;
    }
    // Magnified: crop to the visible source pixels so the transform stays
    // small and the copy cheap.
    let visible = view.visible_rect();
    let to_src_x = |x: f64| (x - raster.rect.x) / raster.rect.width * f64::from(raster.width);
    let to_src_y = |y: f64| (y - raster.rect.y) / raster.rect.height * f64::from(raster.height);
    let x0 = to_src_x(visible.x).floor().max(0.0) as u32;
    let y0 = to_src_y(visible.y).floor().max(0.0) as u32;
    let x1 = (to_src_x(visible.x + visible.width).ceil() + 1.0).min(f64::from(raster.width)) as u32;
    let y1 =
        (to_src_y(visible.y + visible.height).ceil() + 1.0).min(f64::from(raster.height)) as u32;
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let Some(mut crop) = Pixmap::new(cw, ch) else {
        return;
    };
    let stride = raster.width as usize * 4;
    let row_bytes = cw as usize * 4;
    for (row, chunk) in crop.data_mut().chunks_exact_mut(row_bytes).enumerate() {
        let start = (y0 as usize + row) * stride + x0 as usize * 4;
        chunk.copy_from_slice(&raster.premultiplied[start..start + row_bytes]);
    }
    let origin = [
        raster.rect.x + f64::from(x0) * raster.rect.width / f64::from(raster.width),
        raster.rect.y + f64::from(y0) * raster.rect.height / f64::from(raster.height),
    ];
    let inner = Transform::from_row(
        su as f32,
        0.0,
        0.0,
        sv as f32,
        origin[0] as f32,
        origin[1] as f32,
    );
    target.draw_pixmap(0, 0, crop.as_ref(), &paint, view.affine_after(inner), None);
}

/// Converts straight-alpha RGBA to premultiplied RGBA (for raster uploads).
pub fn premultiply(rgba: &mut [u8]) {
    for px in rgba.as_chunks_mut::<4>().0 {
        let a = u16::from(px[3]);
        if a == 255 {
            continue;
        }
        for c in &mut px[..3] {
            *c = ((u16::from(*c) * a + 127) / 255) as u8;
        }
    }
}

/// Composites a premultiplied overlay onto a straight-alpha RGBA image of the
/// same size (source-over); used to burn overlays into exported images.
#[cfg_attr(not(test), allow(dead_code))]
pub fn composite_over_straight(dst: &mut [u8], overlay_premultiplied: &[u8]) {
    for (d, s) in dst
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(overlay_premultiplied.as_chunks::<4>().0)
    {
        let sa = u32::from(s[3]);
        if sa == 0 {
            continue;
        }
        let da = u32::from(d[3]);
        let inv = 255 - sa;
        // out_a = sa + da·(1 − sa)
        let out_a = sa + (da * inv + 127) / 255;
        if out_a == 0 {
            continue;
        }
        for i in 0..3 {
            // Premultiplied result, then un-premultiply.
            let dp = u32::from(d[i]) * da / 255;
            let out_p = u32::from(s[i]) + (dp * inv + 127) / 255;
            d[i] = ((out_p * 255 + out_a / 2) / out_a).min(255) as u8;
        }
        d[3] = out_a.min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::store::{OverlayStore, Owner};
    use eov_plugin_api::protocol::{LayerSpec, OverlayObject};

    fn view(width: f64, height: f64) -> ViewTransform {
        ViewTransform {
            left: 0.0,
            top: 0.0,
            zoom: 1.0,
            width,
            height,
            scale: 1.0,
            mpp: None,
            orientation: Default::default(),
        }
    }

    fn pixel(pixmap: &Pixmap, x: u32, y: u32) -> [u8; 4] {
        let c = pixmap.pixel(x, y).unwrap();
        [c.red(), c.green(), c.blue(), c.alpha()]
    }

    fn filled(color: Rgba) -> Style {
        Style {
            fill: color,
            stroke: Rgba::TRANSPARENT,
            ..Style::default()
        }
    }

    #[test]
    fn polygons_fill_with_holes() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("p");
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("l", 1).with_style(filled(Rgba(255, 0, 0, 255))),
            )
            .unwrap();
        store
            .upsert(
                &owner,
                "l",
                vec![OverlayObject::new(
                    "ring",
                    Shape::Polygon {
                        exterior: vec![[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]],
                        holes: vec![vec![[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]],
                    },
                )],
                false,
            )
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..Default::default()
        };
        let pixmap = render(&view(50.0, 50.0), &scene).unwrap();
        assert_eq!(pixel(&pixmap, 5, 5), [255, 0, 0, 255]);
        assert_eq!(pixel(&pixmap, 20, 20)[3], 0, "hole stays transparent");
        assert_eq!(pixel(&pixmap, 45, 45)[3], 0);
    }

    #[test]
    fn layer_opacity_and_order() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("p");
        store
            .create_layer(
                owner.clone(),
                LayerSpec {
                    opacity: 0.5,
                    ..LayerSpec::new("top", 1)
                        .at_z(1)
                        .with_style(filled(Rgba(0, 0, 255, 255)))
                },
            )
            .unwrap();
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("bottom", 1).with_style(filled(Rgba(255, 0, 0, 255))),
            )
            .unwrap();
        let rect = |id: &str| {
            OverlayObject::new(
                id,
                Shape::Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            )
        };
        store.upsert(&owner, "top", vec![rect("a")], false).unwrap();
        store
            .upsert(&owner, "bottom", vec![rect("b")], false)
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..Default::default()
        };
        let pixmap = render(&view(10.0, 10.0), &scene).unwrap();
        let [r, _, b, a] = pixel(&pixmap, 5, 5);
        assert_eq!(a, 255);
        assert!(
            (120..=135).contains(&r) && (120..=135).contains(&b),
            "{r} {b}"
        );
    }

    #[test]
    fn tiny_objects_become_dots_and_huge_ones_are_clipped() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("p");
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("l", 1).with_style(filled(Rgba(0, 255, 0, 255))),
            )
            .unwrap();
        store
            .upsert(
                &owner,
                "l",
                vec![
                    OverlayObject::new(
                        "tiny",
                        Shape::Rectangle {
                            x: 1000.0,
                            y: 1000.0,
                            width: 10.0,
                            height: 10.0,
                        },
                    ),
                    OverlayObject::new(
                        "huge",
                        Shape::Polygon {
                            exterior: vec![[-1e9, -1e9], [1e9, -1e9], [1e9, 1e9], [-1e9, 1e9]],
                            holes: vec![],
                        },
                    ),
                ],
                false,
            )
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..Default::default()
        };
        // Zoomed out: the 10 px square is 0.1 px wide and drawn as one dot.
        let zoomed_out = ViewTransform {
            zoom: 0.01,
            ..view(20.0, 20.0)
        };
        let pixmap = render(&zoomed_out, &scene).unwrap();
        assert_eq!(pixel(&pixmap, 10, 10)[1], 255);
        // Zoomed in on the huge polygon: fully covered, no precision gaps.
        let zoomed_in = ViewTransform {
            left: 5e8,
            top: 5e8,
            zoom: 100.0,
            ..view(32.0, 32.0)
        };
        let pixmap = render(&zoomed_in, &scene).unwrap();
        assert_eq!(pixel(&pixmap, 0, 0), [0, 255, 0, 255]);
        assert_eq!(pixel(&pixmap, 31, 31), [0, 255, 0, 255]);
    }

    #[test]
    fn rasters_are_stretched_over_their_rectangle() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("p");
        store
            .create_layer(owner.clone(), LayerSpec::new("heat", 1))
            .unwrap();
        // 2×1 raster: red, blue; covers level-0 (0,0)-(100,50).
        let mut data = vec![255, 0, 0, 255, 0, 0, 255, 255];
        premultiply(&mut data);
        store
            .put_raster(
                &owner,
                "heat",
                StoredRaster {
                    object_id: "r".into(),
                    rect: Rect {
                        x: 0.0,
                        y: 0.0,
                        width: 100.0,
                        height: 50.0,
                    },
                    width: 2,
                    height: 1,
                    premultiplied: Arc::new(data),
                    smooth: false,
                    seq: 0,
                },
            )
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..Default::default()
        };
        let pixmap = render(&view(100.0, 50.0), &scene).unwrap();
        assert_eq!(pixel(&pixmap, 10, 25), [255, 0, 0, 255]);
        assert_eq!(pixel(&pixmap, 90, 25), [0, 0, 255, 255]);
        // Magnified path (crop) gives the same colours.
        let magnified = ViewTransform {
            left: 60.0,
            top: 10.0,
            zoom: 4.0,
            ..view(40.0, 40.0)
        };
        let pixmap = render(&magnified, &scene).unwrap();
        assert_eq!(pixel(&pixmap, 20, 20), [0, 0, 255, 255]);
    }

    #[test]
    fn host_roi_dims_outside() {
        let mut target = Pixmap::new(20, 20).unwrap();
        draw_shape(
            &mut target,
            &view(20.0, 20.0),
            &Shape::Rectangle {
                x: 5.0,
                y: 5.0,
                width: 10.0,
                height: 10.0,
            },
            &Style {
                fill: Rgba::TRANSPARENT,
                ..Style::default()
            },
            HostDecoration {
                dim_outside: true,
                contrast_border: true,
            },
            false,
        );
        assert_eq!(pixel(&target, 1, 1)[3], 0x80);
        assert_eq!(pixel(&target, 10, 10)[3], 0);
    }

    #[test]
    fn compositing_matches_source_over() {
        let mut dst = vec![0, 0, 255, 255];
        // 50 % red, premultiplied.
        composite_over_straight(&mut dst, &[128, 0, 0, 128]);
        assert_eq!(dst[3], 255);
        assert!((126..=129).contains(&dst[0]) && (126..=129).contains(&dst[2]));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::overlay::store::{OverlayStore, Owner};
    use eov_plugin_api::protocol::{LayerSpec, OverlayObject};
    use std::time::Instant;

    /// `cargo test --release -p eov --bin eov overlay_render_budget -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn overlay_render_budget() {
        let store = OverlayStore::new();
        let owner = Owner::plugin("cells");
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("nuclei", 1).with_style(Style {
                    fill: Rgba(0x2E, 0xCC, 0x71, 0x40),
                    ..Style::default()
                }),
            )
            .unwrap();
        // 50k 16-gon nuclei of radius 6 px on a 100k × 100k slide.
        let objects: Vec<_> = (0..50_000)
            .map(|i| {
                let (cx, cy) = (
                    ((i * 7919) % 100_000) as f64,
                    ((i * 104_729) % 100_000) as f64,
                );
                OverlayObject::new(
                    format!("n{i}"),
                    Shape::Polygon {
                        exterior: geometry::ellipse_ring(cx, cy, 6.0, 5.0, 16),
                        holes: vec![],
                    },
                )
            })
            .collect();
        let t = Instant::now();
        store.upsert(&owner, "nuclei", objects, true).unwrap();
        println!("bulk upsert 50k: {:?}", t.elapsed());
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..Default::default()
        };
        for (label, zoom) in [
            ("whole slide", 0.0192),
            ("mid zoom", 0.25),
            ("cell level", 2.0),
        ] {
            let view = ViewTransform {
                left: 30_000.0,
                top: 30_000.0,
                zoom,
                width: 1920.0,
                height: 1080.0,
                scale: 1.0,
                mpp: None,
                orientation: Default::default(),
            };
            let view = if label == "whole slide" {
                ViewTransform {
                    left: 0.0,
                    top: 0.0,
                    ..view
                }
            } else {
                view
            };
            let t = Instant::now();
            let runs = 5;
            for _ in 0..runs {
                std::hint::black_box(render(&view, &scene));
            }
            println!("{label}: {:?} per frame", t.elapsed() / runs);
        }
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use eov_common::Viewport;

    fn view_of(viewport: &Viewport, scale: f64) -> ViewTransform {
        let bounds = viewport.bounds();
        ViewTransform {
            left: bounds.left,
            top: bounds.top,
            zoom: viewport.zoom,
            width: viewport.width,
            height: viewport.height,
            scale,
            mpp: None,
            orientation: viewport.orientation,
        }
    }

    /// Overlays land where the turned slide shows the same level-0 point.
    #[test]
    fn turned_overlays_follow_the_view() {
        let mut viewport = Viewport::new(40.0, 30.0, 400.0, 300.0);
        viewport.zoom = 0.5;
        viewport.center.x = 100.0;
        viewport.center.y = 80.0;
        for orientation in Orientation::all() {
            viewport.orientation = orientation;
            let view = view_of(&viewport, 2.0);
            for p in [[90.0, 70.0], [130.0, 95.0], [100.0, 80.0]] {
                let screen = viewport.image_to_screen(p[0], p[1]);
                let (x, y) = view.to_logical(p);
                assert!((x - screen.x).abs() < 1e-9 && (y - screen.y).abs() < 1e-9);
                let (px, py) = view.to_px(p);
                assert!((f64::from(px) - screen.x * 2.0).abs() < 1e-3);
                assert!((f64::from(py) - screen.y * 2.0).abs() < 1e-3);
                let back = view.image_at_logical(x, y);
                assert!((back[0] - p[0]).abs() < 1e-9 && (back[1] - p[1]).abs() < 1e-9);
            }
            let visible = view.visible_rect();
            let bounds = viewport.bounds();
            assert!((visible.width - bounds.width()).abs() < 1e-9);
            assert!((visible.height - bounds.height()).abs() < 1e-9);
        }
    }

    /// Drawing band by band gives the frame drawn at once, turned or not.
    #[test]
    fn turned_bands_compose_the_whole_frame() {
        use crate::overlay::store::{OverlayStore, Owner};
        use eov_plugin_api::protocol::{LayerSpec, OverlayObject};
        let store = OverlayStore::new();
        let owner = Owner::plugin("test");
        store
            .create_layer(owner.clone(), LayerSpec::new("shapes", 1))
            .unwrap();
        store
            .upsert(
                &owner,
                "shapes",
                vec![
                    OverlayObject::new(
                        "line",
                        Shape::Polyline {
                            points: vec![[80.0, 60.0], [125.0, 100.0]],
                        },
                    ),
                    OverlayObject::new(
                        "box",
                        Shape::Rectangle {
                            x: 95.0,
                            y: 72.0,
                            width: 20.0,
                            height: 9.0,
                        },
                    ),
                ],
                false,
            )
            .unwrap();
        let scene = FrameScene {
            layers: store.file_layers(1).layers,
            ..FrameScene::default()
        };
        let mut viewport = Viewport::new(40.0, 30.0, 400.0, 300.0);
        viewport.zoom = 0.75;
        viewport.center.x = 100.0;
        viewport.center.y = 80.0;
        for orientation in Orientation::all() {
            viewport.orientation = orientation;
            let view = view_of(&viewport, 1.0);
            let whole = render(&view, &scene).unwrap();
            let (width, height) = view.physical_size();
            let mut bands = Vec::new();
            for row in (0..height).step_by(7) {
                let rows = 7.min(height - row);
                let band = view.band(row, rows);
                let mut pixmap = Pixmap::new(width, rows).unwrap();
                render_into(&mut pixmap, &band, &scene);
                bands.extend_from_slice(pixmap.data());
            }
            let differing = whole
                .data()
                .iter()
                .zip(&bands)
                .filter(|(a, b)| a.abs_diff(**b) > 8)
                .count();
            assert!(differing < 16, "{orientation:?}: {differing} bytes differ");
            assert!(whole.data().iter().any(|&b| b > 0), "something is drawn");
        }
    }
}
