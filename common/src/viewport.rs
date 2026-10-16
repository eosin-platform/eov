//! Viewport management for WSI rendering
//!
//! Handles viewport state, transformations, pan/zoom calculations,
//! and smooth animations for navigation.

use crate::wsi::SlideBounds;
use glam::DVec2;
use std::time::{Duration, Instant};
use tracing::trace;

/// Minimum zoom level (zoomed out)
pub const MIN_ZOOM: f64 = 0.001;

/// Maximum zoom level (zoomed in)
pub const MAX_ZOOM: f64 = 100.0;

/// Default zoom factor per mouse wheel tick
pub const ZOOM_FACTOR: f64 = 1.025;

/// Animation duration for smooth zoom transitions (300ms)
pub const ANIMATION_DURATION_MS: u64 = 300;

/// Animation duration for smooth framing transitions (300ms)
pub const NAVIGATION_DURATION_MS: u64 = 300;

/// Pan inertia duration (500ms for more perceptible glide)
pub const INERTIA_DURATION_MS: u64 = 500;

/// Recent drag history window used to estimate release velocity.
pub const INERTIA_SAMPLE_WINDOW_MS: u64 = 75;

/// Minimum time span used when converting recent drag motion into a velocity.
pub const INERTIA_MIN_SAMPLE_SPAN_MS: u64 = 8;

/// Idle time after the last drag movement over which release velocity decays to zero.
pub const INERTIA_RELEASE_IDLE_MS: u64 = 40;

/// Upper bound for pan flick velocity in screen pixels per second.
pub const INERTIA_MAX_VELOCITY: f64 = 8_000.0;

/// Minimum drag displacement (in screen pixels) required to trigger inertia.
/// Prevents tiny mouse jitter during a quick catch-click from restarting momentum.
pub const INERTIA_MIN_DISPLACEMENT: f64 = 5.0;

/// Pan margin ratio (can pan past edges by this percentage)
pub const PAN_MARGIN_RATIO: f64 = 0.5;

/// Millionths of a degree in a full turn: angles are kept to a millionth of
/// a degree, so quarter turns are exact and angles read back as written.
const FULL_TURN: u32 = 360_000_000;
/// Millionths of a degree in a quarter turn.
const QUARTER_TURN: u32 = 90_000_000;

/// How a view turns the slide: a clockwise rotation by any angle, then an
/// optional mirror across the screen's vertical axis. Level-0 coordinates
/// are the same whatever the orientation; only their mapping to the screen
/// changes.
///
/// The angle is kept to a millionth of a degree, in `0..360`. Quarter turns
/// are exact: their mappings are permutations, and renderers turn frames
/// of them without resampling ([`is_quarter_turn`](Self::is_quarter_turn)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Orientation {
    /// Clockwise rotation in millionths of a degree, `0..FULL_TURN`.
    micro_degrees: u32,
    flipped: bool,
}

impl Orientation {
    pub const IDENTITY: Self = Self {
        micro_degrees: 0,
        flipped: false,
    };

    /// `quarter_turns` clockwise (any integer), then mirrored left to right
    /// if `flipped`.
    pub fn new(quarter_turns: i32, flipped: bool) -> Self {
        Self {
            micro_degrees: quarter_turns.rem_euclid(4) as u32 * QUARTER_TURN,
            flipped,
        }
    }

    /// From a clockwise angle in degrees: any finite angle (kept to a
    /// millionth of a degree), `None` otherwise.
    pub fn from_degrees(degrees: f64, flipped: bool) -> Option<Self> {
        degrees.is_finite().then(|| Self {
            micro_degrees: micro_degrees_of(degrees),
            flipped,
        })
    }

    /// Whole quarter turns, when the rotation is one.
    pub fn quarter_turns(self) -> Option<u8> {
        self.micro_degrees
            .is_multiple_of(QUARTER_TURN)
            .then_some((self.micro_degrees / QUARTER_TURN) as u8)
    }

    /// Whether the rotation is a whole number of quarter turns: the image's
    /// axes stay along the screen's.
    pub fn is_quarter_turn(self) -> bool {
        self.quarter_turns().is_some()
    }

    /// The clockwise rotation (applied before the mirror), in degrees,
    /// `0..360`.
    pub fn degrees(self) -> f64 {
        f64::from(self.micro_degrees) / 1e6
    }

    pub fn flipped(self) -> bool {
        self.flipped
    }

    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }

    /// The same mirror, upright: what "reset rotation" shows.
    pub fn upright(self) -> Self {
        Self {
            micro_degrees: 0,
            ..self
        }
    }

    /// The same mirror with the rotation set to `degrees` (`None` when it
    /// is not finite).
    pub fn with_degrees(self, degrees: f64) -> Option<Self> {
        Self::from_degrees(degrees, self.flipped)
    }

    /// The orientation that turns this one back.
    pub fn inverse(self) -> Self {
        if self.flipped {
            // A mirrored turn is its own inverse.
            self
        } else {
            Self {
                micro_degrees: (FULL_TURN - self.micro_degrees) % FULL_TURN,
                flipped: false,
            }
        }
    }

    /// This orientation followed by `next` (`next ∘ self`).
    pub fn then(self, next: Self) -> Self {
        // Mirror_b·R(b)·Mirror_a·R(a): past a mirror, R(b)·M = M·R(−b).
        let micro_degrees = if self.flipped {
            (self.micro_degrees + FULL_TURN - next.micro_degrees) % FULL_TURN
        } else {
            (self.micro_degrees + next.micro_degrees) % FULL_TURN
        };
        Self {
            micro_degrees,
            flipped: self.flipped != next.flipped,
        }
    }

    /// The eight quarter-turn orientations.
    pub fn all() -> [Self; 8] {
        let mut all = [Self::IDENTITY; 8];
        for (i, orientation) in all.iter_mut().enumerate() {
            *orientation = Self::new(i as i32 % 4, i >= 4);
        }
        all
    }

    /// The quarter turn nearest to this orientation (with its mirror); the
    /// orientation itself for quarter turns. The minimap shows the slide
    /// turned by it.
    pub fn quarter_part(self) -> Self {
        let quarter_turns = (self.micro_degrees + QUARTER_TURN / 2) / QUARTER_TURN;
        Self::new(quarter_turns as i32, self.flipped)
    }

    /// Whether the image's x axis is vertical on screen (odd quarter turns;
    /// for other angles, of the nearest quarter turn).
    pub fn swaps_axes(self) -> bool {
        (self.quarter_part().micro_degrees / QUARTER_TURN) % 2 == 1
    }

    /// This orientation turned a further `quarter_turns` clockwise as seen
    /// on screen (a mirror reverses the sense of the stored rotation).
    pub fn turned(self, quarter_turns: i32) -> Self {
        self.then(Self::new(quarter_turns, false))
    }

    /// This orientation turned a further `degrees` clockwise as seen on
    /// screen (counterclockwise when negative; a mirror reverses the sense
    /// of the stored rotation); unchanged when `degrees` is not finite.
    pub fn rotated(self, degrees: f64) -> Self {
        match Self::from_degrees(degrees, false) {
            Some(turn) => self.then(turn),
            None => self,
        }
    }

    /// This orientation mirrored left to right as seen on screen.
    pub fn mirrored(self) -> Self {
        Self {
            flipped: !self.flipped,
            ..self
        }
    }

    /// `(sin, cos)` of the rotation, exact for quarter turns.
    fn sin_cos(self) -> (f64, f64) {
        match self.quarter_turns() {
            Some(0) => (0.0, 1.0),
            Some(1) => (1.0, 0.0),
            Some(2) => (0.0, -1.0),
            Some(3) => (-1.0, 0.0),
            _ => self.degrees().to_radians().sin_cos(),
        }
    }

    /// An image-space direction as it points on screen (y down).
    pub fn to_screen(self, v: DVec2) -> DVec2 {
        let turned = match self.quarter_turns() {
            Some(0) => v,
            Some(1) => DVec2::new(-v.y, v.x),
            Some(2) => -v,
            Some(3) => DVec2::new(v.y, -v.x),
            _ => {
                let (sin, cos) = self.sin_cos();
                DVec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
            }
        };
        if self.flipped {
            DVec2::new(-turned.x, turned.y)
        } else {
            turned
        }
    }

    /// A screen direction in image space (the inverse of
    /// [`to_screen`](Self::to_screen)).
    pub fn to_image(self, v: DVec2) -> DVec2 {
        let unmirrored = if self.flipped {
            DVec2::new(-v.x, v.y)
        } else {
            v
        };
        match self.quarter_turns() {
            Some(0) => unmirrored,
            Some(1) => DVec2::new(unmirrored.y, -unmirrored.x),
            Some(2) => -unmirrored,
            Some(3) => DVec2::new(-unmirrored.y, unmirrored.x),
            _ => {
                let (sin, cos) = self.sin_cos();
                let (x, y) = (unmirrored.x, unmirrored.y);
                DVec2::new(x * cos + y * sin, y * cos - x * sin)
            }
        }
    }

    /// [`to_screen`](Self::to_screen) for a pair.
    pub fn to_screen_xy(self, x: f64, y: f64) -> (f64, f64) {
        let p = self.to_screen(DVec2::new(x, y));
        (p.x, p.y)
    }

    /// [`to_image`](Self::to_image) for a pair.
    pub fn to_image_xy(self, x: f64, y: f64) -> (f64, f64) {
        let p = self.to_image(DVec2::new(x, y));
        (p.x, p.y)
    }

    /// The axis-aligned extent of a `width × height` rectangle turned by
    /// this orientation (or back: the same): the on-screen bounding box of
    /// an image-space extent, and the image-space bounding box of an
    /// on-screen size. Width and height swap for odd quarter turns.
    pub fn turned_extent(self, width: f64, height: f64) -> (f64, f64) {
        match self.quarter_turns() {
            Some(turns) if turns % 2 == 1 => (height, width),
            Some(_) => (width, height),
            None => {
                let (sin, cos) = self.sin_cos();
                let (sin, cos) = (sin.abs(), cos.abs());
                (width * cos + height * sin, width * sin + height * cos)
            }
        }
    }

    /// `(a, b)` swapped when the axes are: the on-screen size of an
    /// image-space `a × b` extent, and the image-space extent of an
    /// on-screen `a × b` size. Exact for quarter turns; other angles swap
    /// as their nearest quarter turn does (see
    /// [`turned_extent`](Self::turned_extent) for their extents).
    pub fn swap<T>(self, a: T, b: T) -> (T, T) {
        if self.swaps_axes() { (b, a) } else { (a, b) }
    }

    /// Normalised coordinates (`0..=1` per axis) of an image region as they
    /// fall on the same region shown turned by [`quarter_part`]
    /// (the minimap).
    ///
    /// [`quarter_part`]: Self::quarter_part
    pub fn to_screen_normalized(self, u: f64, v: f64) -> (f64, f64) {
        let p = self.quarter_part().to_screen(DVec2::new(u - 0.5, v - 0.5));
        (p.x + 0.5, p.y + 0.5)
    }

    /// The inverse of [`to_screen_normalized`](Self::to_screen_normalized).
    pub fn to_image_normalized(self, u: f64, v: f64) -> (f64, f64) {
        let p = self.quarter_part().to_image(DVec2::new(u - 0.5, v - 0.5));
        (p.x + 0.5, p.y + 0.5)
    }

    /// The pixel of a frame drawn axis-aligned with the image
    /// (`aligned_width × aligned_height`) that shows as pixel `(x, y)` of
    /// the frame turned to this orientation. For quarter turns (others map
    /// as their nearest quarter turn).
    pub fn aligned_pixel(
        self,
        aligned_width: u32,
        aligned_height: u32,
        x: i64,
        y: i64,
    ) -> (i64, i64) {
        let quarter = self.quarter_part();
        let (w, h) = (i64::from(aligned_width), i64::from(aligned_height));
        let (sw, sh) = quarter.swap(w, h);
        // Doubled coordinates of the pixel centre, relative to the frame
        // centre: integers, so the mapping is exact.
        let d = quarter.to_image(DVec2::new((2 * x + 1 - sw) as f64, (2 * y + 1 - sh) as f64));
        ((d.x as i64 + w - 1) / 2, (d.y as i64 + h - 1) / 2)
    }

    /// The rectangle of the aligned frame (`aligned_width ×
    /// aligned_height`) that shows as the rectangle `x, y, width, height` of
    /// the turned frame, as `(x, y, width, height)`. For quarter turns
    /// (others map as their nearest quarter turn).
    pub fn aligned_rect(
        self,
        aligned_width: u32,
        aligned_height: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> (u32, u32, u32, u32) {
        if width == 0 || height == 0 {
            return (0, 0, 0, 0);
        }
        let (ax, ay) =
            self.aligned_pixel(aligned_width, aligned_height, i64::from(x), i64::from(y));
        let (bx, by) = self.aligned_pixel(
            aligned_width,
            aligned_height,
            i64::from(x + width - 1),
            i64::from(y + height - 1),
        );
        (
            ax.min(bx).max(0) as u32,
            ay.min(by).max(0) as u32,
            ((ax - bx).abs() + 1) as u32,
            ((ay - by).abs() + 1) as u32,
        )
    }
}

/// A finite angle in degrees as millionths of a degree, `0..FULL_TURN`.
fn micro_degrees_of(degrees: f64) -> u32 {
    // Reduced first (exactly), so that huge angles cannot overflow.
    let micro = (degrees.rem_euclid(360.0) * 1e6).round();
    (micro as i64).rem_euclid(i64::from(FULL_TURN)) as u32
}

/// Viewport representing the visible area of a WSI
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Current center position in level 0 coordinates
    pub center: DVec2,
    /// Current zoom level (1.0 = native resolution)
    pub zoom: f64,
    /// Viewport width in screen pixels
    pub width: f64,
    /// Viewport height in screen pixels
    pub height: f64,
    /// Image width at level 0
    pub image_width: f64,
    /// Image height at level 0
    pub image_height: f64,
    /// How the slide is turned on screen.
    pub orientation: Orientation,
}

impl Viewport {
    /// Create a new viewport
    pub fn new(width: f64, height: f64, image_width: f64, image_height: f64) -> Self {
        let mut viewport = Self {
            center: DVec2::new(image_width / 2.0, image_height / 2.0),
            zoom: 1.0,
            width,
            height,
            image_width,
            image_height,
            orientation: Orientation::IDENTITY,
        };
        viewport.fit_to_view();
        viewport
    }

    /// The same view drawn axis-aligned with the image: the layout renderers
    /// place tiles in (relative to [`bounds`](Self::bounds)) before turning
    /// them to `orientation`. Its size is the bounding box of the turned
    /// view (width and height swapped for odd quarter turns; larger, and
    /// fractional, for other angles).
    pub fn aligned(&self) -> Viewport {
        let (width, height) = self.orientation.turned_extent(self.width, self.height);
        Viewport {
            width,
            height,
            orientation: Orientation::IDENTITY,
            ..self.clone()
        }
    }

    /// Get the visible bounds in level 0 coordinates (the bounding box of
    /// the visible area, whatever the orientation)
    pub fn bounds(&self) -> ViewportBounds {
        let (width, height) = self.orientation.turned_extent(self.width, self.height);
        let half_width = (width / self.zoom) / 2.0;
        let half_height = (height / self.zoom) / 2.0;

        ViewportBounds {
            left: self.center.x - half_width,
            right: self.center.x + half_width,
            top: self.center.y - half_height,
            bottom: self.center.y + half_height,
        }
    }

    /// The level-0 corners of the visible area: the pane's top-left,
    /// top-right, bottom-right and bottom-left corners (a turned rectangle;
    /// [`bounds`](Self::bounds) is its bounding box).
    pub fn visible_corners(&self) -> [DVec2; 4] {
        [
            self.screen_to_image(0.0, 0.0),
            self.screen_to_image(self.width, 0.0),
            self.screen_to_image(self.width, self.height),
            self.screen_to_image(0.0, self.height),
        ]
    }

    /// Fit the entire image in the viewport
    pub fn fit_to_view(&mut self) {
        let (width, height) = self
            .orientation
            .turned_extent(self.image_width, self.image_height);
        let zoom_x = self.width / width;
        let zoom_y = self.height / height;
        self.zoom = zoom_x.min(zoom_y).clamp(MIN_ZOOM, MAX_ZOOM);
        self.center = DVec2::new(self.image_width / 2.0, self.image_height / 2.0);
    }

    /// The zoom at which a `width` x `height` level-0 rectangle fits the
    /// viewport (turned as the view is: its turned bounding box fits).
    pub fn fit_zoom_for(&self, width: f64, height: f64) -> f64 {
        if width <= 0.0 || height <= 0.0 {
            return self.zoom;
        }
        let (width, height) = self.orientation.turned_extent(width, height);
        (self.width / width)
            .min(self.height / height)
            .clamp(MIN_ZOOM, MAX_ZOOM)
    }

    /// The level-0 displacement of one screen pixel along the screen's x
    /// axis (`1 / zoom` long, pointing where the turn puts it).
    pub fn screen_x_in_image(&self) -> DVec2 {
        self.orientation.to_image(DVec2::new(1.0, 0.0)) / self.zoom
    }

    /// Micrometres per screen pixel along the screen's x axis, from the
    /// slide's pixel size per image axis (the length a horizontal scale bar
    /// measures, whatever the turn).
    pub fn screen_x_um_per_px(&self, mpp_x: f64, mpp_y: f64) -> f64 {
        let step = self.screen_x_in_image();
        (step.x * mpp_x).hypot(step.y * mpp_y)
    }

    /// Centre and fit a level-0 rectangle (e.g. the slide's
    /// [content bounds](crate::WsiProperties::content_bounds)).
    pub fn fit_to_rect(&mut self, x: f64, y: f64, width: f64, height: f64) {
        if width <= 0.0 || height <= 0.0 {
            return;
        }
        self.zoom = self.fit_zoom_for(width, height);
        self.center = DVec2::new(x + width / 2.0, y + height / 2.0);
    }

    /// [`fit_to_rect`](Self::fit_to_rect) for [`SlideBounds`].
    pub fn fit_to_bounds(&mut self, bounds: &SlideBounds) {
        self.fit_to_rect(
            bounds.x as f64,
            bounds.y as f64,
            bounds.width as f64,
            bounds.height as f64,
        );
    }

    /// Keeps the centre within `region` plus `margin_ratio` of its size on
    /// every side (the image-wide equivalent uses [`PAN_MARGIN_RATIO`]).
    pub fn clamp_center_to(&mut self, region: &SlideBounds, margin_ratio: f64) {
        let margin_x = region.width as f64 * margin_ratio;
        let margin_y = region.height as f64 * margin_ratio;
        let (left, top) = (region.x as f64, region.y as f64);
        self.center.x = self
            .center
            .x
            .clamp(left - margin_x, left + region.width as f64 + margin_x);
        self.center.y = self
            .center
            .y
            .clamp(top - margin_y, top + region.height as f64 + margin_y);
    }

    /// Convert screen coordinates to image coordinates
    pub fn screen_to_image(&self, screen_x: f64, screen_y: f64) -> DVec2 {
        let offset = DVec2::new(screen_x - self.width / 2.0, screen_y - self.height / 2.0);
        self.center + self.orientation.to_image(offset) / self.zoom
    }

    /// Convert image coordinates to screen coordinates
    pub fn image_to_screen(&self, image_x: f64, image_y: f64) -> DVec2 {
        let offset = self
            .orientation
            .to_screen(DVec2::new(image_x, image_y) - self.center)
            * self.zoom;
        DVec2::new(self.width / 2.0 + offset.x, self.height / 2.0 + offset.y)
    }

    /// Pan the viewport by the given screen pixel delta
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.center -= self.orientation.to_image(DVec2::new(dx, dy)) / self.zoom;
        self.clamp_position();
    }

    /// Position the viewport so that `image_pos` appears at `screen_pos`.
    /// Computes center directly from the anchor constraint to avoid
    /// floating-point error accumulation.
    pub fn anchor_at(&mut self, image_pos: DVec2, screen_pos: DVec2) {
        let offset = DVec2::new(
            screen_pos.x - self.width / 2.0,
            screen_pos.y - self.height / 2.0,
        );
        self.center = image_pos - self.orientation.to_image(offset) / self.zoom;
        self.clamp_position();
    }

    /// Zoom by the given factor around the specified screen point
    pub fn zoom_at(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        // Get image position under cursor before zoom
        let image_pos = self.screen_to_image(screen_x, screen_y);

        // Apply zoom
        let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.zoom = new_zoom;

        // Adjust center so the image point stays under the cursor
        self.anchor_at(image_pos, DVec2::new(screen_x, screen_y));
    }

    /// Zoom by the given factor around the center
    pub fn zoom_center(&mut self, factor: f64) {
        self.zoom_at(factor, self.width / 2.0, self.height / 2.0);
    }

    /// Set the viewport size
    pub fn set_size(&mut self, width: f64, height: f64) {
        self.width = width;
        self.height = height;
        self.clamp_position();
    }

    /// Clamp the viewport position to within valid bounds (with margin)
    fn clamp_position(&mut self) {
        let margin_x = self.image_width * PAN_MARGIN_RATIO;
        let margin_y = self.image_height * PAN_MARGIN_RATIO;

        self.center.x = self
            .center
            .x
            .max(-margin_x)
            .min(self.image_width + margin_x);
        self.center.y = self
            .center
            .y
            .max(-margin_y)
            .min(self.image_height + margin_y);
    }

    /// Get the effective downsample factor for tile selection
    pub fn effective_downsample(&self) -> f64 {
        1.0 / self.zoom
    }

    /// Calculate the minimap rectangle (0-1 normalized coordinates)
    ///
    /// The visible area is intersected with the image first, so the
    /// rectangle stays inside the minimap when the view crosses an edge.
    pub fn minimap_rect(&self) -> MinimapRect {
        self.minimap_rect_within(&SlideBounds {
            x: 0,
            y: 0,
            width: self.image_width.max(0.0) as u64,
            height: self.image_height.max(0.0) as u64,
        })
    }

    /// The visible part of `region`, normalised to `region` (for a minimap
    /// that shows the slide's content bounds rather than the whole canvas),
    /// as it falls on `region` turned like the minimap (by the view's
    /// [nearest quarter turn](Orientation::quarter_part)). For other angles
    /// it is the bounding box of [`minimap_outline_within`](Self::minimap_outline_within).
    pub fn minimap_rect_within(&self, region: &SlideBounds) -> MinimapRect {
        let (rw, rh) = (region.width as f64, region.height as f64);
        if rw <= 0.0 || rh <= 0.0 {
            return MinimapRect::default();
        }
        if !self.orientation.is_quarter_turn() {
            let outline = self.minimap_outline_within(region);
            let Some(first) = outline.first() else {
                return MinimapRect::default();
            };
            let (mut x0, mut y0, mut x1, mut y1) = (first[0], first[1], first[0], first[1]);
            for p in &outline {
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
            return MinimapRect {
                x: x0 as f32,
                y: y0 as f32,
                width: (x1 - x0) as f32,
                height: (y1 - y0) as f32,
            };
        }
        let bounds = self.bounds();
        let axis = |lo: f64, hi: f64, origin: f64, size: f64| -> (f64, f64) {
            (
                ((lo - origin) / size).clamp(0.0, 1.0),
                ((hi - origin) / size).clamp(0.0, 1.0),
            )
        };
        let (x0, x1) = axis(bounds.left, bounds.right, region.x as f64, rw);
        let (y0, y1) = axis(bounds.top, bounds.bottom, region.y as f64, rh);
        let (ax, ay) = self.orientation.to_screen_normalized(x0, y0);
        let (bx, by) = self.orientation.to_screen_normalized(x1, y1);
        MinimapRect {
            x: ax.min(bx) as f32,
            y: ay.min(by) as f32,
            width: (ax - bx).abs() as f32,
            height: (ay - by).abs() as f32,
        }
    }

    /// The outline of the visible area on a minimap of `region`: the
    /// pane's corners normalised to `region`, turned like the minimap (by
    /// the view's [nearest quarter turn](Orientation::quarter_part), so the
    /// rest of the turn tilts the outline), and clipped to the minimap. A
    /// turned rectangle, or a polygon of up to eight points where it
    /// crosses the edges; empty when nothing of `region` is visible.
    pub fn minimap_outline_within(&self, region: &SlideBounds) -> Vec<[f64; 2]> {
        let (rw, rh) = (region.width as f64, region.height as f64);
        if rw <= 0.0 || rh <= 0.0 {
            return Vec::new();
        }
        let corners = self.visible_corners().map(|p| {
            let (u, v) = self
                .orientation
                .to_screen_normalized((p.x - region.x as f64) / rw, (p.y - region.y as f64) / rh);
            [u, v]
        });
        clip_to_unit_square(&corners)
    }
}

/// `polygon` (convex) clipped to `0..=1` on both axes (Sutherland–Hodgman).
fn clip_to_unit_square(polygon: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut points = polygon.to_vec();
    // Each edge of the square: the axis, its value, and which side is kept.
    for (axis, edge, keep_below) in [
        (0, 0.0, false),
        (0, 1.0, true),
        (1, 0.0, false),
        (1, 1.0, true),
    ] {
        let inside = |p: &[f64; 2]| {
            if keep_below {
                p[axis] <= edge
            } else {
                p[axis] >= edge
            }
        };
        let mut clipped = Vec::with_capacity(points.len() + 2);
        for (i, current) in points.iter().enumerate() {
            let previous = &points[(i + points.len() - 1) % points.len()];
            let (now_in, before_in) = (inside(current), inside(previous));
            if now_in != before_in {
                let t = (edge - previous[axis]) / (current[axis] - previous[axis]);
                let mut crossing = [
                    previous[0] + (current[0] - previous[0]) * t,
                    previous[1] + (current[1] - previous[1]) * t,
                ];
                crossing[axis] = edge;
                clipped.push(crossing);
            }
            if now_in {
                clipped.push(*current);
            }
        }
        points = clipped;
        if points.is_empty() {
            break;
        }
    }
    points
}

/// Visible area bounds in image coordinates
#[derive(Debug, Clone, Copy)]
pub struct ViewportBounds {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

impl ViewportBounds {
    pub fn width(&self) -> f64 {
        self.right - self.left
    }

    pub fn height(&self) -> f64 {
        self.bottom - self.top
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.left && x <= self.right && y >= self.top && y <= self.bottom
    }
}

/// Minimap rectangle in normalized (0-1) coordinates
#[derive(Debug, Clone, Copy, Default)]
pub struct MinimapRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Viewport state with smooth animations for zoom and pan
#[derive(Debug, Clone)]
pub struct ViewportState {
    /// Base viewport
    pub viewport: Viewport,

    // --- Pan inertia animation ---
    /// Is currently dragging
    is_dragging: bool,
    /// Whether the active drag was started by `start_external_drag`.
    external_drag: bool,
    /// Last drag position
    drag_start: DVec2,
    /// Recent absolute drag samples used to estimate release velocity.
    drag_samples: Vec<(DVec2, Instant)>,
    /// Inertia animation: initial velocity when drag ended
    inertia_start_velocity: DVec2,
    /// Inertia animation: start time
    inertia_start_time: Option<Instant>,

    // --- Zoom animation ---
    /// Target zoom level (for smooth zoom)
    target_zoom: f64,
    /// Zoom start value (for animation)
    zoom_start: f64,
    /// Screen position to zoom around
    zoom_anchor_screen: DVec2,
    /// Image position to keep under cursor during zoom
    zoom_anchor_image: DVec2,
    /// Zoom animation start time (timed eased animation for discrete zooms)
    zoom_start_time: Option<Instant>,
    /// Duration of the current zoom animation in milliseconds
    zoom_duration_ms: u64,
    /// Use a smoother ease-in-out profile for larger discrete zoom jumps
    zoom_use_ease_in_out: bool,
    /// Exponential-smoothing chase mode (for scroll wheel zoom)
    zoom_chasing: bool,

    // --- Framing/navigation animation ---
    /// Navigation start center (for smooth frame-to-view / frame-to-ROI)
    navigation_start_center: DVec2,
    /// Navigation target center
    navigation_target_center: DVec2,
    /// Navigation start zoom
    navigation_start_zoom: f64,
    /// Navigation target zoom
    navigation_target_zoom: f64,
    /// Navigation animation start time
    navigation_start_time: Option<Instant>,

    /// Last update time for physics
    last_update: Instant,

    /// Area with image data; when set, fitting and pan limits use it
    /// instead of the full canvas.
    content_bounds: Option<SlideBounds>,
}

impl ViewportState {
    /// Create a new viewport state
    pub fn new(width: f64, height: f64, image_width: f64, image_height: f64) -> Self {
        let viewport = Viewport::new(width, height, image_width, image_height);
        let initial_zoom = viewport.zoom;
        let initial_center = viewport.center;
        Self {
            viewport,
            is_dragging: false,
            external_drag: false,
            drag_start: DVec2::ZERO,
            drag_samples: Vec::with_capacity(8),
            inertia_start_velocity: DVec2::ZERO,
            inertia_start_time: None,
            target_zoom: initial_zoom,
            zoom_start: initial_zoom,
            zoom_anchor_screen: DVec2::ZERO,
            zoom_anchor_image: DVec2::ZERO,
            zoom_start_time: None,
            zoom_duration_ms: ANIMATION_DURATION_MS,
            zoom_use_ease_in_out: false,
            zoom_chasing: false,
            navigation_start_center: initial_center,
            navigation_target_center: initial_center,
            navigation_start_zoom: initial_zoom,
            navigation_target_zoom: initial_zoom,
            navigation_start_time: None,
            last_update: Instant::now(),
            content_bounds: None,
        }
    }

    /// A viewport state that fits and pans within `content_bounds` (see
    /// [`set_content_bounds`](Self::set_content_bounds)).
    pub fn with_content_bounds(
        width: f64,
        height: f64,
        image_width: f64,
        image_height: f64,
        content_bounds: Option<SlideBounds>,
    ) -> Self {
        let mut state = Self::new(width, height, image_width, image_height);
        state.set_content_bounds(content_bounds);
        state.fit_to_view();
        state
    }

    /// Restricts fitting and panning to the slide's content bounds
    /// ([`WsiProperties::content_bounds`](crate::WsiProperties::content_bounds)),
    /// so MIRAX-style slides whose tissue is a small island in a large
    /// canvas open framed on the tissue. `None` restores the full canvas.
    /// Bounds equal to the whole canvas are treated as `None`.
    pub fn set_content_bounds(&mut self, bounds: Option<SlideBounds>) {
        let full = SlideBounds {
            x: 0,
            y: 0,
            width: self.viewport.image_width.max(0.0) as u64,
            height: self.viewport.image_height.max(0.0) as u64,
        };
        self.content_bounds = bounds.filter(|b| !b.is_empty() && *b != full);
        self.clamp_to_content();
    }

    /// The content bounds set with [`set_content_bounds`](Self::set_content_bounds).
    pub fn content_bounds(&self) -> Option<SlideBounds> {
        self.content_bounds
    }

    /// The minimap rectangle relative to the content bounds when set
    /// (pair it with a thumbnail of the same region), else to the canvas.
    pub fn minimap_rect(&self) -> MinimapRect {
        match &self.content_bounds {
            Some(bounds) => self.viewport.minimap_rect_within(bounds),
            None => self.viewport.minimap_rect(),
        }
    }

    /// [`Viewport::minimap_outline_within`] the content bounds when set,
    /// else the canvas.
    pub fn minimap_outline(&self) -> Vec<[f64; 2]> {
        let canvas = SlideBounds {
            x: 0,
            y: 0,
            width: self.viewport.image_width.max(0.0) as u64,
            height: self.viewport.image_height.max(0.0) as u64,
        };
        self.viewport
            .minimap_outline_within(self.content_bounds.as_ref().unwrap_or(&canvas))
    }

    fn clamp_to_content(&mut self) {
        if let Some(bounds) = self.content_bounds {
            self.viewport.clamp_center_to(&bounds, PAN_MARGIN_RATIO);
        }
    }

    /// The rectangle fitting targets: the content bounds or the canvas.
    fn fit_target(&self) -> (f64, f64, f64, f64) {
        match &self.content_bounds {
            Some(b) => (b.x as f64, b.y as f64, b.width as f64, b.height as f64),
            None => (
                0.0,
                0.0,
                self.viewport.image_width,
                self.viewport.image_height,
            ),
        }
    }

    /// The centre (x, y) and zoom of [`fit_to_view`](Self::fit_to_view),
    /// without moving the view.
    pub fn fit_view(&self) -> (f64, f64, f64) {
        let (x, y, width, height) = self.fit_target();
        (
            x + width / 2.0,
            y + height / 2.0,
            self.viewport.fit_zoom_for(width, height),
        )
    }

    /// Start a drag operation
    pub fn start_drag(&mut self, x: f64, y: f64) {
        let now = Instant::now();
        self.is_dragging = true;
        self.external_drag = false;
        self.drag_start = DVec2::new(x, y);
        self.drag_samples.clear();
        self.drag_samples.push((self.drag_start, now));
        self.inertia_start_time = None; // Cancel any ongoing inertia
        self.zoom_start_time = None; // Cancel any ongoing zoom animation to prevent snap-back
        self.navigation_start_time = None; // Cancel any framing animation
        self.last_update = now;
    }

    /// Continue a drag operation
    pub fn drag_to(&mut self, x: f64, y: f64) {
        if !self.is_dragging {
            return;
        }

        let now = Instant::now();
        let current = DVec2::new(x, y);
        let delta = current - self.drag_start;

        if self
            .drag_samples
            .last()
            .map(|(position, _)| *position != current)
            .unwrap_or(true)
        {
            self.drag_samples.push((current, now));
            let cutoff = now - Duration::from_millis(INERTIA_SAMPLE_WINDOW_MS);
            self.drag_samples
                .retain(|(_, sample_time)| *sample_time >= cutoff);
        }

        // Pan viewport
        self.viewport.pan(delta.x, delta.y);
        self.clamp_to_content();

        // Keep zoom anchor in sync so the chase doesn't snap back
        if self.zoom_chasing || self.zoom_start_time.is_some() {
            self.zoom_anchor_image = self
                .viewport
                .screen_to_image(self.zoom_anchor_screen.x, self.zoom_anchor_screen.y);
        }

        self.drag_start = current;
        self.last_update = now;
    }

    /// End a drag operation - start inertia animation
    pub fn end_drag(&mut self) {
        if !self.is_dragging {
            return;
        }

        let now = Instant::now();
        self.is_dragging = false;
        self.external_drag = false;

        if let (Some((first_position, first_time)), Some((last_position, last_time))) = (
            self.drag_samples.first().copied(),
            self.drag_samples.last().copied(),
        ) {
            let idle_secs = now.duration_since(last_time).as_secs_f64();
            let idle_decay = (1.0
                - idle_secs / Duration::from_millis(INERTIA_RELEASE_IDLE_MS).as_secs_f64())
            .clamp(0.0, 1.0);
            let displacement = last_position - first_position;
            let sample_span = last_time.duration_since(first_time).as_secs_f64();
            let effective_span =
                sample_span.max(Duration::from_millis(INERTIA_MIN_SAMPLE_SPAN_MS).as_secs_f64());

            if displacement.length() >= INERTIA_MIN_DISPLACEMENT && idle_decay > 0.0 {
                let mut velocity = displacement / effective_span;
                let speed = velocity.length();
                if speed > INERTIA_MAX_VELOCITY {
                    velocity = velocity / speed * INERTIA_MAX_VELOCITY;
                }
                velocity *= idle_decay;

                if velocity.length() > 20.0 {
                    self.inertia_start_velocity = velocity;
                    self.inertia_start_time = Some(now);
                    trace!(
                        "Starting pan inertia: velocity={:?} length={:.1} idle_decay={:.2}",
                        velocity,
                        velocity.length(),
                        idle_decay
                    );
                }
            }
        }

        self.drag_samples.clear();
        self.last_update = now;
    }

    /// Start an external drag sequence, such as minimap scrubbing.
    pub fn start_external_drag(&mut self) {
        self.stop();
        self.is_dragging = true;
        self.external_drag = true;
        self.drag_samples.clear();
        self.last_update = Instant::now();
    }

    /// Update the viewport center directly while an external drag is active.
    pub fn drag_center_to(&mut self, center_x: f64, center_y: f64) {
        if !self.is_dragging {
            self.start_external_drag();
        }
        self.viewport.center = DVec2::new(center_x, center_y);
        self.viewport.clamp_position();
        self.clamp_to_content();
        self.last_update = Instant::now();
    }

    /// End an external drag sequence without starting inertia.
    pub fn end_external_drag(&mut self) {
        if !self.is_dragging || !self.external_drag {
            return;
        }
        self.is_dragging = false;
        self.external_drag = false;
        self.drag_samples.clear();
        self.last_update = Instant::now();
    }

    /// Update animations (call every frame). Returns true if still animating.
    pub fn update(&mut self) -> bool {
        if self.is_dragging {
            return false;
        }

        let now = Instant::now();
        let zoom_duration = Duration::from_millis(self.zoom_duration_ms.max(1));
        let navigation_duration = Duration::from_millis(NAVIGATION_DURATION_MS);
        let inertia_duration = Duration::from_millis(INERTIA_DURATION_MS);
        let mut is_animating = false;

        // --- Framing/navigation animation ---
        if let Some(start_time) = self.navigation_start_time {
            let elapsed = now.duration_since(start_time);

            if elapsed < navigation_duration {
                let t = elapsed.as_secs_f64() / navigation_duration.as_secs_f64();
                let ease_out = 1.0 - (1.0 - t).powi(3);
                let next_center = self.navigation_start_center
                    + (self.navigation_target_center - self.navigation_start_center) * ease_out;
                let log_start = self.navigation_start_zoom.ln();
                let log_target = self.navigation_target_zoom.ln();
                let next_zoom = (log_start + (log_target - log_start) * ease_out)
                    .exp()
                    .clamp(MIN_ZOOM, MAX_ZOOM);

                self.viewport.center = next_center;
                self.viewport.zoom = next_zoom;
                self.viewport.clamp_position();
                is_animating = true;
            } else {
                self.viewport.center = self.navigation_target_center;
                self.viewport.zoom = self.navigation_target_zoom;
                self.viewport.clamp_position();
                self.navigation_start_time = None;
            }
        }

        // --- Pan inertia animation ---
        if let Some(start_time) = self.inertia_start_time {
            let elapsed = now.duration_since(start_time);

            if elapsed < inertia_duration {
                // Ease-out: velocity decreases over time (quadratic for smoother feel)
                let t = elapsed.as_secs_f64() / inertia_duration.as_secs_f64();
                let ease_out = (1.0 - t) * (1.0 - t); // Quadratic ease-out

                let dt = now.duration_since(self.last_update).as_secs_f64();
                let current_velocity = self.inertia_start_velocity * ease_out;

                // Apply panning - velocity was measured in screen-space drag direction
                // which matches pan() semantics (positive = image moves right)
                self.viewport
                    .pan(current_velocity.x * dt, current_velocity.y * dt);

                // If a zoom animation is also running, update its anchor so the
                // zoom tracks the inertia-displaced viewport instead of fighting it.
                if self.zoom_start_time.is_some() || self.zoom_chasing {
                    self.zoom_anchor_image = self
                        .viewport
                        .screen_to_image(self.zoom_anchor_screen.x, self.zoom_anchor_screen.y);
                }

                is_animating = true;
                trace!("Pan inertia: t={:.2}, velocity={:?}", t, current_velocity);
            } else {
                // Animation complete
                self.inertia_start_time = None;
            }
        }

        // --- Scroll wheel zoom chase (exponential smoothing) ---
        if self.zoom_chasing {
            let log_current = self.viewport.zoom.ln();
            let log_target = self.target_zoom.ln();
            let diff = (log_target - log_current).abs();

            if diff < 0.0001 {
                // Close enough — snap to target
                self.viewport.zoom = self.target_zoom;
                self.viewport
                    .anchor_at(self.zoom_anchor_image, self.zoom_anchor_screen);
                self.zoom_chasing = false;
            } else {
                let dt = now.duration_since(self.last_update).as_secs_f64();
                // Time constant: 80ms gives snappy response with smooth tail
                let tau = 0.08;
                let alpha = 1.0 - (-dt / tau).exp();
                let new_log = log_current + (log_target - log_current) * alpha;
                let new_zoom = new_log.exp().clamp(MIN_ZOOM, MAX_ZOOM);

                self.viewport.zoom = new_zoom;
                self.viewport
                    .anchor_at(self.zoom_anchor_image, self.zoom_anchor_screen);
                is_animating = true;
                trace!(
                    "Zoom chase: zoom={:.4}, target={:.4}",
                    new_zoom, self.target_zoom
                );
            }
        }

        // --- Timed zoom animation (discrete/button zooms) ---
        if let Some(start_time) = self.zoom_start_time {
            let elapsed = now.duration_since(start_time);

            if elapsed < zoom_duration {
                let t = elapsed.as_secs_f64() / zoom_duration.as_secs_f64();
                let ease = if self.zoom_use_ease_in_out {
                    t * t * (3.0 - 2.0 * t)
                } else {
                    1.0 - (1.0 - t).powi(3)
                };

                // Interpolate zoom in log space for perceptually linear zoom
                let log_start = self.zoom_start.ln();
                let log_target = self.target_zoom.ln();
                let log_current = log_start + (log_target - log_start) * ease;
                let new_zoom = log_current.exp().clamp(MIN_ZOOM, MAX_ZOOM);

                // Apply zoom while keeping anchor point fixed
                self.viewport.zoom = new_zoom;
                self.viewport
                    .anchor_at(self.zoom_anchor_image, self.zoom_anchor_screen);

                is_animating = true;
                trace!("Zoom animation: t={:.2}, zoom={:.4}", t, new_zoom);
            } else {
                // Animation complete - snap to target
                self.viewport.zoom = self.target_zoom;
                self.viewport
                    .anchor_at(self.zoom_anchor_image, self.zoom_anchor_screen);

                self.zoom_start_time = None;
            }
        }

        self.clamp_to_content();
        self.last_update = now;
        is_animating
    }

    /// Stop all movement and animations
    pub fn stop(&mut self) {
        self.inertia_start_time = None;
        self.zoom_start_time = None;
        self.zoom_chasing = false;
        self.navigation_start_time = None;
        self.is_dragging = false;
        self.target_zoom = self.viewport.zoom;
        self.zoom_use_ease_in_out = false;
        self.navigation_target_zoom = self.viewport.zoom;
        self.navigation_start_zoom = self.viewport.zoom;
        self.navigation_start_center = self.viewport.center;
        self.navigation_target_center = self.viewport.center;
    }

    /// Immediately set the viewport center and zoom, cancelling any active animation.
    pub fn set_center_zoom(&mut self, center_x: f64, center_y: f64, zoom: f64) {
        self.stop();
        self.viewport.center = DVec2::new(center_x, center_y);
        self.viewport.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.viewport.fit_to_view();
        self.viewport.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.viewport.center = DVec2::new(center_x, center_y);
        self.viewport.clamp_position();
        self.clamp_to_content();
        self.target_zoom = self.viewport.zoom;
    }

    /// Smoothly animate to an absolute center and zoom.
    pub fn smooth_set_center_zoom(&mut self, center_x: f64, center_y: f64, zoom: f64) {
        self.animate_to(DVec2::new(center_x, center_y), zoom);
    }

    /// Zoom at screen position with smooth exponential chase.
    /// Ideal for scroll wheel input — each tick compounds onto the target
    /// while the viewport smoothly converges via exponential smoothing.
    pub fn zoom_at(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        // A wheel zoom always wins over a dangling external drag (for
        // example a minimap drag whose release was never delivered), which
        // would otherwise freeze `update()` forever.
        self.end_external_drag();
        self.navigation_start_time = None;
        self.zoom_start_time = None; // cancel any timed animation
        if !self.zoom_chasing {
            // Start the chase from "now"; otherwise the first frame after an
            // idle period sees a multi-second dt and jumps to the target.
            self.last_update = Instant::now();
        }

        let anchor_screen = DVec2::new(screen_x, screen_y);
        let anchor_image = self.viewport.screen_to_image(screen_x, screen_y);

        // Compound onto existing target when already chasing
        let base_zoom = if self.zoom_chasing {
            self.target_zoom
        } else {
            self.viewport.zoom
        };

        self.zoom_anchor_screen = anchor_screen;
        self.zoom_anchor_image = anchor_image;
        self.target_zoom = (base_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.zoom_chasing = true;
    }

    /// Zoom at screen position with smooth animation and an explicit duration.
    pub fn zoom_at_with_duration(
        &mut self,
        factor: f64,
        screen_x: f64,
        screen_y: f64,
        duration_ms: u64,
    ) {
        self.start_zoom_animation(factor, screen_x, screen_y, duration_ms, false);
    }

    /// Zoom at screen position with a smoother discrete-action easing profile.
    pub fn zoom_at_discrete(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        self.start_zoom_animation(factor, screen_x, screen_y, ANIMATION_DURATION_MS, true);
    }

    /// Smoothly zoom to an absolute zoom level around the viewport center.
    pub fn zoom_to(&mut self, zoom: f64) {
        self.zoom_to_with_duration(zoom, ANIMATION_DURATION_MS);
    }

    /// Smoothly zoom to an absolute zoom level around the viewport center with an explicit duration.
    pub fn zoom_to_with_duration(&mut self, zoom: f64, duration_ms: u64) {
        let clamped_zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        let anchor_x = self.viewport.width / 2.0;
        let anchor_y = self.viewport.height / 2.0;

        self.navigation_start_time = None;
        self.zoom_chasing = false;
        self.zoom_use_ease_in_out = false;
        self.zoom_duration_ms = duration_ms.max(1);
        self.zoom_start = self.viewport.zoom;
        self.zoom_anchor_screen = DVec2::new(anchor_x, anchor_y);
        self.zoom_anchor_image = self.viewport.screen_to_image(anchor_x, anchor_y);
        self.zoom_start_time = Some(Instant::now());
        self.target_zoom = clamped_zoom;
        self.inertia_start_time = None;
        self.is_dragging = false;
    }

    /// Fit the entire image (or the content bounds, when set) in view
    /// (immediate, no animation)
    pub fn fit_to_view(&mut self) {
        self.stop();
        let (x, y, w, h) = self.fit_target();
        self.viewport.fit_to_rect(x, y, w, h);
        self.target_zoom = self.viewport.zoom;
        self.navigation_target_zoom = self.viewport.zoom;
        self.navigation_target_center = self.viewport.center;
    }

    /// Smoothly frame the entire image (or the content bounds, when set).
    pub fn smooth_fit_to_view(&mut self) {
        let (x, y, w, h) = self.fit_target();
        let target_zoom = self.viewport.fit_zoom_for(w, h);
        let target_center = DVec2::new(x + w / 2.0, y + h / 2.0);
        self.animate_to(target_center, target_zoom);
    }

    /// Smoothly frame a rectangle in image coordinates with a small margin.
    pub fn smooth_frame_rect(&mut self, x: f64, y: f64, width: f64, height: f64) {
        if width <= 0.0 || height <= 0.0 {
            self.smooth_fit_to_view();
            return;
        }

        let padding_factor = 1.1;
        let framed_width = (width * padding_factor).max(1.0);
        let framed_height = (height * padding_factor).max(1.0);
        let target_zoom = self.viewport.fit_zoom_for(framed_width, framed_height);
        let target_center = DVec2::new(x + width / 2.0, y + height / 2.0);
        self.animate_to(target_center, target_zoom);
    }

    fn animate_to(&mut self, center: DVec2, zoom: f64) {
        self.is_dragging = false;
        self.inertia_start_time = None;
        self.zoom_start_time = None;
        self.zoom_use_ease_in_out = false;
        self.navigation_start_center = self.viewport.center;
        self.navigation_target_center = center;
        self.navigation_start_zoom = self.viewport.zoom;
        self.navigation_target_zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.navigation_start_time = Some(Instant::now());
        self.target_zoom = self.navigation_target_zoom;
        self.last_update = Instant::now();
    }

    fn start_zoom_animation(
        &mut self,
        factor: f64,
        screen_x: f64,
        screen_y: f64,
        duration_ms: u64,
        ease_in_out: bool,
    ) {
        self.navigation_start_time = None;
        self.zoom_chasing = false; // cancel any chase
        self.zoom_duration_ms = duration_ms.max(1);
        self.zoom_use_ease_in_out = ease_in_out;

        let anchor_screen = DVec2::new(screen_x, screen_y);
        let anchor_image = self.viewport.screen_to_image(screen_x, screen_y);

        // When a zoom animation is already running, compound the new factor onto
        // the existing target instead of the current (partially-animated) zoom.
        // This keeps rapid scroll-wheel ticks from resetting progress and
        // feeling sluggish.
        let base_zoom = if self.zoom_start_time.is_some() {
            self.target_zoom
        } else {
            self.viewport.zoom
        };

        self.zoom_start = self.viewport.zoom;
        self.zoom_anchor_screen = anchor_screen;
        self.zoom_anchor_image = anchor_image;
        self.zoom_start_time = Some(Instant::now());
        self.target_zoom = (base_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);

        trace!(
            "Zoom requested: factor={}, target={}",
            factor, self.target_zoom
        );
    }

    /// Set viewport size
    pub fn set_size(&mut self, width: f64, height: f64) {
        self.viewport.set_size(width, height);
        self.clamp_to_content();
    }

    /// Turns the view (about its centre, keeping the zoom). A view that was
    /// fitted is fitted again to the turned slide.
    pub fn set_orientation(&mut self, orientation: Orientation) {
        if self.viewport.orientation == orientation {
            return;
        }
        let fitted = self.is_fitted();
        self.stop();
        self.viewport.orientation = orientation;
        if fitted {
            self.fit_to_view();
        }
    }

    /// Whether the view shows the fitted slide (or content bounds).
    fn is_fitted(&self) -> bool {
        let (x, y, w, h) = self.fit_target();
        let fit_zoom = self.viewport.fit_zoom_for(w, h);
        let center = DVec2::new(x + w / 2.0, y + h / 2.0);
        (self.viewport.zoom / fit_zoom - 1.0).abs() < 0.01
            && (self.viewport.center - center).length() * self.viewport.zoom < 2.0
    }

    /// Convert screen coordinates to image coordinates
    pub fn screen_to_image(&self, screen_x: f64, screen_y: f64) -> (f64, f64) {
        let result = self.viewport.screen_to_image(screen_x, screen_y);
        (result.x, result.y)
    }

    /// Check if the viewport is currently animating
    pub fn is_moving(&self) -> bool {
        self.is_dragging
            || self.inertia_start_time.is_some()
            || self.zoom_start_time.is_some()
            || self.zoom_chasing
            || self.navigation_start_time.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec2, b: DVec2) -> bool {
        (a - b).length() < 1e-9
    }

    #[test]
    fn orientations_map_directions_and_back() {
        let right = DVec2::new(1.0, 0.0);
        let down = DVec2::new(0.0, 1.0);
        // A clockwise quarter turn: the image's x axis points down, its y
        // axis left; mirroring then swaps left and right.
        let turned = Orientation::new(1, false);
        assert!(close(turned.to_screen(right), down));
        assert!(close(turned.to_screen(down), -right));
        let mirrored = Orientation::new(0, true);
        assert!(close(mirrored.to_screen(right), -right));
        for orientation in Orientation::all() {
            let v = DVec2::new(3.0, -7.0);
            assert!(close(orientation.to_image(orientation.to_screen(v)), v));
            let inverse = orientation.inverse();
            assert!(close(inverse.to_screen(orientation.to_screen(v)), v));
        }
        assert_eq!(
            Orientation::from_degrees(-90.0, false),
            Some(Orientation::new(3, false))
        );
        assert_eq!(
            Orientation::from_degrees(450.0, true),
            Some(Orientation::new(1, true))
        );
        assert_eq!(Orientation::from_degrees(f64::NAN, false), None);
        assert_eq!(Orientation::from_degrees(f64::INFINITY, false), None);
    }

    /// Any angle: kept to a millionth of a degree, read back as written,
    /// with quarter turns exact.
    #[test]
    fn angles_are_kept_exactly_and_quarter_turns_stay_exact() {
        let turned = |degrees: f64| Orientation::from_degrees(degrees, false).unwrap();
        assert_eq!(turned(37.5).degrees(), 37.5);
        assert_eq!(turned(-5.0).degrees(), 355.0);
        assert_eq!(turned(720.25).degrees(), 0.25);
        assert_eq!(turned(12.345678).degrees(), 12.345678);
        assert_eq!(turned(359.9999999).degrees(), 0.0, "rounds to a full turn");
        assert_eq!(turned(1e300).degrees(), turned(1e300).degrees());
        assert!(turned(1e300).degrees() < 360.0);
        // Steps add up to an exact quarter turn.
        let mut stepped = Orientation::IDENTITY;
        for _ in 0..18 {
            stepped = stepped.rotated(5.0);
        }
        assert_eq!(stepped, Orientation::new(1, false));
        assert!(stepped.is_quarter_turn());
        assert_eq!(
            stepped.to_screen(DVec2::new(1.0, 0.0)),
            DVec2::new(0.0, 1.0)
        );
        assert_eq!(turned(30.0).quarter_turns(), None);
        assert_eq!(turned(270.0).quarter_turns(), Some(3));
        // The nearest quarter turn (the minimap's).
        assert_eq!(turned(30.0).quarter_part(), Orientation::IDENTITY);
        assert_eq!(turned(50.0).quarter_part(), Orientation::new(1, false));
        assert_eq!(turned(316.0).quarter_part(), Orientation::IDENTITY);
        assert_eq!(
            Orientation::from_degrees(100.0, true)
                .unwrap()
                .quarter_part(),
            Orientation::new(1, true)
        );
        // Resetting the rotation keeps the mirror.
        let mirrored = Orientation::from_degrees(30.0, true).unwrap();
        assert_eq!(mirrored.upright(), Orientation::new(0, true));
        assert_eq!(mirrored.with_degrees(75.0).unwrap().degrees(), 75.0);
        assert!(mirrored.with_degrees(75.0).unwrap().flipped());
    }

    /// Mapping, composing and inverting at angles other than quarter turns.
    #[test]
    fn free_angles_map_compose_and_invert() {
        let angles = [0.0, 5.0, 30.0, 45.0, 90.0, 123.456, 200.0, 315.0, 359.5];
        let clockwise = |degrees: f64, p: DVec2| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            DVec2::new(p.x * cos - p.y * sin, p.x * sin + p.y * cos)
        };
        let v = DVec2::new(3.0, -7.0);
        for &a in &angles {
            for flipped in [false, true] {
                let o = Orientation::from_degrees(a, flipped).unwrap();
                let expected = clockwise(a, v);
                let expected = if flipped {
                    DVec2::new(-expected.x, expected.y)
                } else {
                    expected
                };
                assert!((o.to_screen(v) - expected).length() < 1e-9, "{a} {flipped}");
                assert!((o.to_image(o.to_screen(v)) - v).length() < 1e-9);
                assert!((o.inverse().to_screen(o.to_screen(v)) - v).length() < 1e-9);
                assert!(o.then(o.inverse()).is_identity());
                // Turning on screen, as the keys do.
                let screen = o.to_screen(v);
                assert!((o.rotated(17.0).to_screen(v) - clockwise(17.0, screen)).length() < 1e-9);
                for &b in &angles {
                    for flip_b in [false, true] {
                        let p = Orientation::from_degrees(b, flip_b).unwrap();
                        let composed = o.then(p);
                        assert!(
                            (composed.to_screen(v) - p.to_screen(o.to_screen(v))).length() < 1e-9,
                            "{a} {flipped} then {b} {flip_b}"
                        );
                    }
                }
                // The turned extent bounds the turned rectangle.
                let (w, h) = o.turned_extent(40.0, 10.0);
                for corner in [
                    DVec2::new(20.0, 5.0),
                    DVec2::new(-20.0, 5.0),
                    DVec2::new(20.0, -5.0),
                ] {
                    let p = o.to_screen(corner);
                    assert!(p.x.abs() <= w / 2.0 + 1e-9 && p.y.abs() <= h / 2.0 + 1e-9);
                }
            }
        }
        let (w, h) = Orientation::from_degrees(45.0, false)
            .unwrap()
            .turned_extent(100.0, 100.0);
        assert!((w - 100.0 * 2f64.sqrt()).abs() < 1e-9 && (h - w).abs() < 1e-9);
        assert_eq!(
            Orientation::new(1, false).turned_extent(3.0, 2.0),
            (2.0, 3.0)
        );
    }

    #[test]
    fn turning_and_mirroring_act_on_what_the_screen_shows() {
        let v = DVec2::new(2.0, 1.0);
        let clockwise = |p: DVec2| DVec2::new(-p.y, p.x);
        let mirror = |p: DVec2| DVec2::new(-p.x, p.y);
        for orientation in Orientation::all() {
            assert!(close(
                orientation.turned(1).to_screen(v),
                clockwise(orientation.to_screen(v))
            ));
            assert!(close(
                orientation.mirrored().to_screen(v),
                mirror(orientation.to_screen(v))
            ));
        }
        assert_eq!(Orientation::IDENTITY.turned(4), Orientation::IDENTITY);
        assert_eq!(Orientation::IDENTITY.turned(1).degrees(), 90.0);
        for a in Orientation::all() {
            assert_eq!(a.then(a.inverse()), Orientation::IDENTITY);
            for b in Orientation::all() {
                let composed = a.then(b);
                assert!(close(composed.to_screen(v), b.to_screen(a.to_screen(v))));
            }
        }
    }

    /// The quarter turns and some free angles, mirrored or not.
    fn test_orientations() -> Vec<Orientation> {
        let mut all = Orientation::all().to_vec();
        for degrees in [5.0, 30.0, 45.0, 137.25, 300.0] {
            for flipped in [false, true] {
                all.push(Orientation::from_degrees(degrees, flipped).unwrap());
            }
        }
        all
    }

    #[test]
    fn turned_views_map_screen_and_image_consistently() {
        let mut viewport = Viewport::new(800.0, 600.0, 10_000.0, 8_000.0);
        viewport.zoom = 0.5;
        viewport.center = DVec2::new(4000.0, 3000.0);
        for orientation in test_orientations() {
            viewport.orientation = orientation;
            let p = viewport.screen_to_image(123.0, 456.0);
            assert!(close(
                viewport.image_to_screen(p.x, p.y),
                DVec2::new(123.0, 456.0)
            ));
            // The centre stays the centre; the bounds hold every corner.
            assert!(close(
                viewport.screen_to_image(400.0, 300.0),
                viewport.center
            ));
            let bounds = viewport.bounds();
            for (x, y) in [(0.0, 0.0), (800.0, 0.0), (0.0, 600.0), (800.0, 600.0)] {
                let corner = viewport.screen_to_image(x, y);
                assert!(corner.x >= bounds.left - 1e-6 && corner.x <= bounds.right + 1e-6);
                assert!(corner.y >= bounds.top - 1e-6 && corner.y <= bounds.bottom + 1e-6);
            }
            let aligned = viewport.aligned();
            assert_eq!(aligned.bounds().left, bounds.left);
            assert_eq!(aligned.bounds().bottom, bounds.bottom);
            // Dragging moves the image with the pointer.
            let mut panned = viewport.clone();
            let before = panned.image_to_screen(5000.0, 3500.0);
            panned.pan(30.0, -20.0);
            let after = panned.image_to_screen(5000.0, 3500.0);
            assert!(close(after - before, DVec2::new(30.0, -20.0)));
            // Zooming keeps the point under the pointer.
            let mut zoomed = viewport.clone();
            let under = zoomed.screen_to_image(100.0, 50.0);
            zoomed.zoom_at(2.0, 100.0, 50.0);
            assert!(close(zoomed.screen_to_image(100.0, 50.0), under));
        }
    }

    #[test]
    fn fitting_and_the_minimap_follow_the_turn() {
        let mut viewport = Viewport::new(800.0, 400.0, 2000.0, 1000.0);
        assert!((viewport.zoom - 0.4).abs() < 1e-9);
        viewport.orientation = Orientation::new(1, false);
        viewport.fit_to_view();
        // The 2000 px wide slide stands upright in the 400 px high pane.
        assert!((viewport.zoom - 0.2).abs() < 1e-9);
        viewport.orientation = Orientation::IDENTITY;
        viewport.zoom = 0.8;
        viewport.center = DVec2::new(500.0, 500.0);
        let plain = viewport.minimap_rect();
        assert!((plain.x, plain.y, plain.width, plain.height) == (0.0, 0.25, 0.5, 0.5));
        // Turned clockwise, the pane shows the slide's full height across
        // and x from 250 to 750 down: on the upright (turned) thumbnail
        // that is all of its width and 1/8 to 3/8 of its height.
        viewport.orientation = Orientation::new(1, false);
        let turned = viewport.minimap_rect();
        let expected = [0.0, 0.125, 1.0, 0.25];
        let got = [turned.x, turned.y, turned.width, turned.height];
        assert!(
            got.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6),
            "{got:?}"
        );
        let (u, v) = viewport.orientation.to_screen_normalized(0.1, 0.2);
        let (x, y) = viewport.orientation.to_image_normalized(u, v);
        assert!((x - 0.1).abs() < 1e-12 && (y - 0.2).abs() < 1e-12);
    }

    #[test]
    fn turning_a_fitted_view_fits_it_again() {
        let mut state = ViewportState::new(800.0, 400.0, 2000.0, 1000.0);
        state.fit_to_view();
        state.set_orientation(Orientation::new(1, false));
        assert!((state.viewport.zoom - 0.2).abs() < 1e-9);
        // A view that was not fitted keeps its centre and zoom.
        state.set_center_zoom(300.0, 200.0, 1.5);
        state.set_orientation(Orientation::new(2, true));
        assert_eq!(state.viewport.zoom, 1.5);
        assert!(close(state.viewport.center, DVec2::new(300.0, 200.0)));
        // Any angle turns about the pane centre: the point there stays.
        let pane_centre = state.viewport.screen_to_image(400.0, 200.0);
        state.set_orientation(Orientation::from_degrees(33.0, true).unwrap());
        assert!(close(
            state.viewport.screen_to_image(400.0, 200.0),
            pane_centre
        ));
        assert_eq!(state.viewport.zoom, 1.5);
    }

    /// The bounding box of a view at a free angle holds the turned pane
    /// exactly (every side touched by a corner); a fitted slide's corners
    /// all lie in the pane.
    #[test]
    fn free_angle_bounds_and_fits_hold_the_turned_rectangles() {
        let mut viewport = Viewport::new(800.0, 400.0, 2000.0, 1000.0);
        for degrees in [10.0, 30.0, 45.0, 100.0, 222.0] {
            viewport.orientation = Orientation::from_degrees(degrees, false).unwrap();
            viewport.zoom = 0.7;
            viewport.center = DVec2::new(900.0, 400.0);
            let bounds = viewport.bounds();
            let corners = viewport.visible_corners();
            let touches = |value: f64, pick: fn(&DVec2) -> f64| {
                corners.iter().any(|c| (pick(c) - value).abs() < 1e-6)
            };
            assert!(touches(bounds.left, |c| c.x) && touches(bounds.right, |c| c.x));
            assert!(touches(bounds.top, |c| c.y) && touches(bounds.bottom, |c| c.y));
            viewport.fit_to_view();
            let mut on_edge = false;
            for (x, y) in [(0.0, 0.0), (2000.0, 0.0), (2000.0, 1000.0), (0.0, 1000.0)] {
                let p = viewport.image_to_screen(x, y);
                assert!(p.x >= -1e-6 && p.x <= 800.0 + 1e-6, "{degrees}: {p:?}");
                assert!(p.y >= -1e-6 && p.y <= 400.0 + 1e-6, "{degrees}: {p:?}");
                on_edge |= p.x.abs() < 1e-6
                    || (p.x - 800.0).abs() < 1e-6
                    || p.y.abs() < 1e-6
                    || (p.y - 400.0).abs() < 1e-6;
            }
            assert!(on_edge, "{degrees}: fitted, not just inside");
        }
        // A horizontal scale bar measures along the turned x axis.
        viewport.orientation = Orientation::from_degrees(90.0, false).unwrap();
        viewport.zoom = 2.0;
        assert!((viewport.screen_x_um_per_px(0.25, 0.5) - 0.25).abs() < 1e-12);
        viewport.orientation = Orientation::from_degrees(60.0, false).unwrap();
        let expected = (0.5f64 * 0.25 * 0.5).hypot(3f64.sqrt() / 2.0 * 0.5 * 0.5);
        assert!((viewport.screen_x_um_per_px(0.25, 0.5) - expected).abs() < 1e-12);
    }

    /// At a free angle the minimap keeps the nearest quarter turn and the
    /// outline of the view is the pane's turned rectangle, clipped.
    #[test]
    fn minimap_outlines_tilt_with_the_rest_of_the_turn() {
        let mut viewport = Viewport::new(400.0, 200.0, 4000.0, 2000.0);
        viewport.zoom = 0.25;
        viewport.center = DVec2::new(2000.0, 1000.0);
        viewport.orientation = Orientation::from_degrees(30.0, false).unwrap();
        let outline = viewport.minimap_outline_within(&SlideBounds::full(4000, 2000));
        assert_eq!(outline.len(), 4, "the view is inside the slide");
        // Each corner is the pane's corner normalised to the (upright)
        // thumbnail.
        for (p, corner) in outline.iter().zip(viewport.visible_corners()) {
            assert!((p[0] - corner.x / 4000.0).abs() < 1e-9);
            assert!((p[1] - corner.y / 2000.0).abs() < 1e-9);
        }
        // The sides are tilted 30° (the y axis points down).
        let (dx, dy) = (outline[1][0] - outline[0][0], outline[1][1] - outline[0][1]);
        let angle = (dy * 2000.0).atan2(dx * 4000.0).to_degrees();
        assert!((angle + 30.0).abs() < 1e-9, "{angle}");
        let rect = viewport.minimap_rect();
        let xs = outline.iter().map(|p| p[0]);
        assert!((f64::from(rect.x) - xs.fold(f64::INFINITY, f64::min)).abs() < 1e-6);
        // Over the edge, the outline is clipped to the thumbnail.
        viewport.center = DVec2::new(100.0, 100.0);
        let clipped = viewport.minimap_outline_within(&SlideBounds::full(4000, 2000));
        assert!(clipped.len() >= 3);
        assert!(
            clipped
                .iter()
                .all(|p| (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]))
        );
        viewport.center = DVec2::new(-9000.0, 100.0);
        assert!(
            viewport
                .minimap_outline_within(&SlideBounds::full(4000, 2000))
                .is_empty()
        );
        assert_eq!(viewport.minimap_rect().width, 0.0);
        // Past 45° the thumbnail turns a quarter and the outline tilts back.
        viewport.center = DVec2::new(2000.0, 1000.0);
        viewport.orientation = Orientation::from_degrees(80.0, false).unwrap();
        let outline = viewport.minimap_outline_within(&SlideBounds::full(4000, 2000));
        let quarter = Orientation::new(1, false);
        for (p, corner) in outline.iter().zip(viewport.visible_corners()) {
            let (u, v) = quarter.to_screen_normalized(corner.x / 4000.0, corner.y / 2000.0);
            assert!((p[0] - u).abs() < 1e-9 && (p[1] - v).abs() < 1e-9);
        }
    }

    #[test]
    fn aligned_pixels_and_rects_map_exactly() {
        // A 3 × 2 frame turned clockwise is 2 × 3; its top-left pixel shows
        // the aligned frame's bottom-left one.
        let turned = Orientation::new(1, false);
        assert_eq!(turned.aligned_pixel(3, 2, 0, 0), (0, 1));
        assert_eq!(turned.aligned_pixel(3, 2, 1, 2), (2, 0));
        // The turned frame's middle row is the aligned frame's middle column.
        assert_eq!(turned.aligned_rect(3, 2, 0, 1, 2, 1), (1, 0, 1, 2));
        for orientation in Orientation::all() {
            let (w, h) = orientation.swap(5u32, 3u32);
            assert_eq!(orientation.aligned_rect(5, 3, 0, 0, w, h), (0, 0, 5, 3));
        }
    }

    #[test]
    fn test_viewport_creation() {
        let viewport = Viewport::new(800.0, 600.0, 10000.0, 10000.0);
        assert!(viewport.zoom > 0.0);
        assert!(viewport.center.x > 0.0);
    }

    #[test]
    fn test_screen_to_image_conversion() {
        let mut viewport = Viewport::new(800.0, 600.0, 10000.0, 10000.0);
        viewport.zoom = 1.0;
        viewport.center = DVec2::new(5000.0, 5000.0);

        // Center of screen should map to center of image
        let center = viewport.screen_to_image(400.0, 300.0);
        assert!((center.x - 5000.0).abs() < 0.1);
        assert!((center.y - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_zoom_at_point() {
        let mut viewport = Viewport::new(800.0, 600.0, 10000.0, 10000.0);
        viewport.zoom = 1.0;
        viewport.center = DVec2::new(5000.0, 5000.0);

        // Get initial image position under cursor
        let before = viewport.screen_to_image(200.0, 150.0);

        // Zoom in
        viewport.zoom_at(2.0, 200.0, 150.0);

        // Position under cursor should be preserved
        let after = viewport.screen_to_image(200.0, 150.0);
        assert!((before.x - after.x).abs() < 1.0);
        assert!((before.y - after.y).abs() < 1.0);
    }

    #[test]
    fn test_fit_to_view() {
        let mut viewport = Viewport::new(800.0, 600.0, 10000.0, 10000.0);
        viewport.fit_to_view();

        // After fit, bounds should include entire image
        let bounds = viewport.bounds();
        assert!(bounds.left <= 0.0);
        assert!(bounds.top <= 0.0);
        assert!(bounds.right >= viewport.image_width);
        assert!(bounds.bottom >= viewport.image_height);
    }

    #[test]
    fn test_pan_with_margin() {
        let mut viewport = Viewport::new(800.0, 600.0, 1000.0, 1000.0);
        viewport.zoom = 1.0;

        // Pan way past the edge
        viewport.pan(-10000.0, -10000.0);

        // Should be clamped within margin
        let margin_x = viewport.image_width * PAN_MARGIN_RATIO;
        let margin_y = viewport.image_height * PAN_MARGIN_RATIO;
        assert!(viewport.center.x <= viewport.image_width + margin_x);
        assert!(viewport.center.y <= viewport.image_height + margin_y);
    }

    #[test]
    fn test_minimap_rect() {
        let mut viewport = Viewport::new(800.0, 600.0, 10000.0, 10000.0);
        viewport.fit_to_view();

        let rect = viewport.minimap_rect();
        // When fit to view, minimap rect should be close to full image
        assert!(rect.x >= 0.0 && rect.x <= 1.0);
        assert!(rect.y >= 0.0 && rect.y <= 1.0);
        assert!(rect.width > 0.0 && rect.width <= 1.0);
        assert!(rect.height > 0.0 && rect.height <= 1.0);
    }

    #[test]
    fn test_viewport_state_drag() {
        let mut state = ViewportState::new(800.0, 600.0, 10000.0, 10000.0);
        let initial_center = state.viewport.center;

        state.start_drag(400.0, 300.0);
        state.drag_to(500.0, 400.0);
        state.end_drag();

        // Center should have moved
        assert!((state.viewport.center - initial_center).length() > 0.1);
    }

    #[test]
    fn test_fast_release_starts_pan_inertia() {
        let mut state = ViewportState::new(800.0, 600.0, 10000.0, 10000.0);
        let now = Instant::now();

        state.is_dragging = true;
        state.drag_samples = vec![
            (DVec2::new(400.0, 300.0), now - Duration::from_millis(18)),
            (DVec2::new(520.0, 300.0), now - Duration::from_millis(4)),
        ];
        state.end_drag();

        assert!(state.inertia_start_time.is_some());
    }

    #[test]
    fn test_stopping_before_release_cancels_pan_inertia() {
        let mut state = ViewportState::new(800.0, 600.0, 10000.0, 10000.0);
        let now = Instant::now();

        state.is_dragging = true;
        state.drag_samples = vec![
            (
                DVec2::new(400.0, 300.0),
                now - Duration::from_millis(INERTIA_RELEASE_IDLE_MS + 28),
            ),
            (
                DVec2::new(520.0, 300.0),
                now - Duration::from_millis(INERTIA_RELEASE_IDLE_MS + 12),
            ),
        ];
        state.end_drag();

        assert!(state.inertia_start_time.is_none());
    }

    #[test]
    fn test_pan_inertia_velocity_is_bounded() {
        let mut state = ViewportState::new(800.0, 600.0, 10000.0, 10000.0);
        let now = Instant::now();

        state.is_dragging = true;
        state.drag_samples = vec![
            (DVec2::new(400.0, 300.0), now - Duration::from_millis(1)),
            (DVec2::new(1000.0, 300.0), now),
        ];
        state.end_drag();

        assert!(state.inertia_start_time.is_some());
        assert!(state.inertia_start_velocity.length() <= INERTIA_MAX_VELOCITY + 0.1);
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    #[test]
    fn minimap_rect_is_clipped_to_the_image() {
        let mut viewport = Viewport::new(500.0, 500.0, 1000.0, 1000.0);
        viewport.zoom = 1.0;
        // Visible: x -200..300, y 800..1300.
        viewport.center = DVec2::new(50.0, 1050.0);
        let rect = viewport.minimap_rect();
        assert_eq!((rect.x, rect.width), (0.0, 0.3));
        assert!((rect.y - 0.8).abs() < 1e-6 && (rect.height - 0.2).abs() < 1e-6);
        // Entirely outside: an empty rectangle at the edge.
        viewport.center = DVec2::new(-5000.0, 500.0);
        assert_eq!(viewport.minimap_rect().width, 0.0);
    }

    #[test]
    fn fits_and_clamps_to_content_bounds() {
        let island = SlideBounds {
            x: 60_000,
            y: 20_000,
            width: 10_000,
            height: 5_000,
        };
        let mut viewport = Viewport::new(1000.0, 500.0, 100_000.0, 80_000.0);
        viewport.fit_to_bounds(&island);
        assert_eq!(viewport.center, DVec2::new(65_000.0, 22_500.0));
        assert!((viewport.zoom - 0.1).abs() < 1e-12);
        let minimap = viewport.minimap_rect_within(&island);
        assert!(minimap.x.abs() < 1e-6 && (minimap.width - 1.0).abs() < 1e-6);

        let mut state =
            ViewportState::with_content_bounds(1000.0, 500.0, 100_000.0, 80_000.0, Some(island));
        assert_eq!(state.viewport.center, DVec2::new(65_000.0, 22_500.0));
        assert_eq!(state.content_bounds(), Some(island));
        assert_eq!(
            state.fit_view(),
            (65_000.0, 22_500.0, state.viewport.zoom),
            "fit_view describes the fitted view"
        );
        // Panning far away stops half a bounds-size outside the island.
        state.start_drag(500.0, 250.0);
        state.drag_to(500.0 + 1e7, 250.0 + 1e7);
        state.end_drag();
        assert_eq!(state.viewport.center, DVec2::new(55_000.0, 17_500.0));
        // Refit, smoothly.
        state.smooth_fit_to_view();
        while state.update() {}
        assert!((state.viewport.center - DVec2::new(65_000.0, 22_500.0)).length() < 1e-6);
        let rect = state.minimap_rect();
        assert!(rect.width > 0.99 && rect.x < 0.01);

        // Bounds covering the canvas are ignored.
        state.set_content_bounds(Some(SlideBounds::full(100_000, 80_000)));
        assert_eq!(state.content_bounds(), None);
        state.fit_to_view();
        assert_eq!(state.viewport.center, DVec2::new(50_000.0, 40_000.0));
    }
}

#[cfg(test)]
mod external_drag_tests {
    use super::*;

    #[test]
    fn wheel_zoom_ends_a_dangling_external_drag_but_not_a_mouse_pan() {
        let mut state = ViewportState::new(800.0, 600.0, 10_000.0, 8_000.0);
        state.start_external_drag();
        state.zoom_at(1.5, 400.0, 300.0);
        assert!(
            state.update() || state.is_moving(),
            "zoom must animate after a dangling drag"
        );

        let mut panning = ViewportState::new(800.0, 600.0, 10_000.0, 8_000.0);
        panning.start_drag(10.0, 10.0);
        panning.zoom_at(1.5, 400.0, 300.0);
        assert!(
            !panning.update(),
            "a mouse pan keeps priority over the zoom chase"
        );
    }
}
