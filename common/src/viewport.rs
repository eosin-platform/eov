//! Viewport management for WSI rendering
//!
//! Handles viewport state, transformations, pan/zoom calculations,
//! and smooth inertia for navigation.

use glam::DVec2;
use std::time::Instant;
use tracing::trace;

/// Minimum zoom level (zoomed out)
pub const MIN_ZOOM: f64 = 0.001;

/// Maximum zoom level (zoomed in)
pub const MAX_ZOOM: f64 = 100.0;

/// Default zoom factor per mouse wheel tick
pub const ZOOM_FACTOR: f64 = 1.15;

/// Inertia decay factor (0-1, lower = faster decay)
pub const INERTIA_DECAY: f64 = 0.92;

/// Minimum velocity before stopping movement
pub const MIN_VELOCITY: f64 = 0.1;

/// Velocity scale factor for pan
pub const VELOCITY_SCALE: f64 = 0.1;

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
        self.zoom = zoom_x.min(zoom_y).max(MIN_ZOOM).min(MAX_ZOOM);
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

    /// Zoom by the given factor around the specified screen point
    pub fn zoom_at(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        // Get image position under cursor before zoom
        let image_pos = self.screen_to_image(screen_x, screen_y);

        // Apply zoom
        let new_zoom = (self.zoom * factor).max(MIN_ZOOM).min(MAX_ZOOM);
        self.zoom = new_zoom;

        // Adjust center so the image point stays under the cursor
        let new_screen_pos = self.image_to_screen(image_pos.x, image_pos.y);
        let screen_dx = screen_x - new_screen_pos.x;
        let screen_dy = screen_y - new_screen_pos.y;
        self.pan(-screen_dx, -screen_dy);
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

        self.center.x = self.center.x.max(-margin_x).min(self.image_width + margin_x);
        self.center.y = self.center.y.max(-margin_y).min(self.image_height + margin_y);
    }

    /// Get the effective downsample factor for tile selection
    pub fn effective_downsample(&self) -> f64 {
        1.0 / self.zoom
    }

    /// Calculate the minimap rectangle (0-1 normalized coordinates)
    pub fn minimap_rect(&self) -> MinimapRect {
        let bounds = self.bounds();
        MinimapRect {
            x: (bounds.left / self.image_width).max(0.0).min(1.0) as f32,
            y: (bounds.top / self.image_height).max(0.0).min(1.0) as f32,
            width: ((bounds.right - bounds.left) / self.image_width).max(0.0).min(1.0) as f32,
            height: ((bounds.bottom - bounds.top) / self.image_height).max(0.0).min(1.0) as f32,
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

/// Viewport state with velocity for smooth movement
#[derive(Debug, Clone)]
pub struct ViewportState {
    /// Base viewport
    pub viewport: Viewport,
    /// Current velocity in screen pixels per second
    velocity: DVec2,
    /// Last update time for physics
    last_update: Instant,
    /// Is currently dragging
    is_dragging: bool,
    /// Last drag position
    drag_start: DVec2,
    /// Velocity samples for averaging
    velocity_samples: Vec<DVec2>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl ViewportState {
    /// Create a new viewport state
    pub fn new(width: f64, height: f64, image_width: f64, image_height: f64) -> Self {
        Self {
            viewport: Viewport::new(width, height, image_width, image_height),
            velocity: DVec2::ZERO,
            last_update: Instant::now(),
            is_dragging: false,
            drag_start: DVec2::ZERO,
            velocity_samples: Vec::with_capacity(10),
            max_samples: 5,
        }
    }

    /// Start a drag operation
    pub fn start_drag(&mut self, x: f64, y: f64) {
        self.is_dragging = true;
        self.drag_start = DVec2::new(x, y);
        self.velocity = DVec2::ZERO;
        self.velocity_samples.clear();
        self.last_update = Instant::now();
    }

    /// Continue a drag operation
    pub fn drag_to(&mut self, x: f64, y: f64) {
        if !self.is_dragging {
            return;
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        
        let current = DVec2::new(x, y);
        let delta = current - self.drag_start;
        
        // Calculate instantaneous velocity
        if dt > 0.0 {
            let instant_velocity = delta / dt;
            self.velocity_samples.push(instant_velocity);
            if self.velocity_samples.len() > self.max_samples {
                self.velocity_samples.remove(0);
            }
        }

        // Pan viewport
        self.viewport.pan(delta.x, delta.y);
        
        self.drag_start = current;
        self.last_update = now;
    }

    /// End a drag operation
    pub fn end_drag(&mut self) {
        if !self.is_dragging {
            return;
        }

        self.is_dragging = false;
        
        // Calculate average velocity from samples
        if !self.velocity_samples.is_empty() {
            let sum: DVec2 = self.velocity_samples.iter().copied().sum();
            self.velocity = sum / self.velocity_samples.len() as f64;
            self.velocity *= VELOCITY_SCALE;
        }

        self.last_update = Instant::now();
    }

    /// Update physics (call every frame)
    pub fn update(&mut self) -> bool {
        if self.is_dragging {
            return false;
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        // Apply velocity with inertia
        if self.velocity.length() > MIN_VELOCITY {
            self.viewport.pan(-self.velocity.x * dt, -self.velocity.y * dt);
            self.velocity *= INERTIA_DECAY.powf(dt * 60.0); // Normalize to ~60fps
            
            trace!("Inertia velocity: {:?}", self.velocity);
            return true; // Still moving
        } else {
            self.velocity = DVec2::ZERO;
            return false; // Stopped
        }
    }

    /// Stop all movement
    pub fn stop(&mut self) {
        self.velocity = DVec2::ZERO;
        self.is_dragging = false;
    }

    /// Zoom at screen position
    pub fn zoom_at(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        self.stop();
        self.viewport.zoom_at(factor, screen_x, screen_y);
    }

    /// Fit entire image in view
    pub fn fit_to_view(&mut self) {
        self.stop();
        self.viewport.fit_to_view();
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

    /// Check if the viewport is currently moving
    pub fn is_moving(&self) -> bool {
        self.is_dragging || self.velocity.length() > MIN_VELOCITY
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
}
