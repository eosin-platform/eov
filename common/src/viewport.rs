//! Viewport management for WSI rendering
//!
//! Handles viewport state, transformations, pan/zoom calculations,
//! and smooth animations for navigation.

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
        };
        viewport.fit_to_view();
        viewport
    }

    /// Get the visible bounds in level 0 coordinates
    pub fn bounds(&self) -> ViewportBounds {
        let half_width = (self.width / self.zoom) / 2.0;
        let half_height = (self.height / self.zoom) / 2.0;

        ViewportBounds {
            left: self.center.x - half_width,
            right: self.center.x + half_width,
            top: self.center.y - half_height,
            bottom: self.center.y + half_height,
        }
    }

    /// Fit the entire image in the viewport
    pub fn fit_to_view(&mut self) {
        let zoom_x = self.width / self.image_width;
        let zoom_y = self.height / self.image_height;
        self.zoom = zoom_x.min(zoom_y).clamp(MIN_ZOOM, MAX_ZOOM);
        self.center = DVec2::new(self.image_width / 2.0, self.image_height / 2.0);
    }

    /// Convert screen coordinates to image coordinates
    pub fn screen_to_image(&self, screen_x: f64, screen_y: f64) -> DVec2 {
        let offset_x = (screen_x - self.width / 2.0) / self.zoom;
        let offset_y = (screen_y - self.height / 2.0) / self.zoom;
        DVec2::new(self.center.x + offset_x, self.center.y + offset_y)
    }

    /// Convert image coordinates to screen coordinates
    pub fn image_to_screen(&self, image_x: f64, image_y: f64) -> DVec2 {
        let offset_x = (image_x - self.center.x) * self.zoom;
        let offset_y = (image_y - self.center.y) * self.zoom;
        DVec2::new(self.width / 2.0 + offset_x, self.height / 2.0 + offset_y)
    }

    /// Pan the viewport by the given screen pixel delta
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let image_dx = dx / self.zoom;
        let image_dy = dy / self.zoom;
        self.center.x -= image_dx;
        self.center.y -= image_dy;
        self.clamp_position();
    }

    /// Position the viewport so that `image_pos` appears at `screen_pos`.
    /// Computes center directly from the anchor constraint to avoid
    /// floating-point error accumulation.
    pub fn anchor_at(&mut self, image_pos: DVec2, screen_pos: DVec2) {
        self.center.x = image_pos.x - (screen_pos.x - self.width / 2.0) / self.zoom;
        self.center.y = image_pos.y - (screen_pos.y - self.height / 2.0) / self.zoom;
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
    pub fn minimap_rect(&self) -> MinimapRect {
        let bounds = self.bounds();
        MinimapRect {
            x: (bounds.left / self.image_width).clamp(0.0, 1.0) as f32,
            y: (bounds.top / self.image_height).clamp(0.0, 1.0) as f32,
            width: ((bounds.right - bounds.left) / self.image_width).clamp(0.0, 1.0) as f32,
            height: ((bounds.bottom - bounds.top) / self.image_height).clamp(0.0, 1.0) as f32,
        }
    }
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
        }
    }

    /// Start a drag operation
    pub fn start_drag(&mut self, x: f64, y: f64) {
        let now = Instant::now();
        self.is_dragging = true;
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
        self.navigation_start_time = None;
        self.zoom_start_time = None; // cancel any timed animation

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

    /// Fit entire image in view (immediate, no animation)
    pub fn fit_to_view(&mut self) {
        self.stop();
        self.viewport.fit_to_view();
        self.target_zoom = self.viewport.zoom;
    }

    /// Smoothly frame the entire image in view.
    pub fn smooth_fit_to_view(&mut self) {
        let zoom_x = self.viewport.width / self.viewport.image_width;
        let zoom_y = self.viewport.height / self.viewport.image_height;
        let target_zoom = zoom_x.min(zoom_y).clamp(MIN_ZOOM, MAX_ZOOM);
        let target_center = DVec2::new(
            self.viewport.image_width / 2.0,
            self.viewport.image_height / 2.0,
        );
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
        let target_zoom = (self.viewport.width / framed_width)
            .min(self.viewport.height / framed_height)
            .clamp(MIN_ZOOM, MAX_ZOOM);
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
