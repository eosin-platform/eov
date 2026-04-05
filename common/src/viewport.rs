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
pub const ZOOM_FACTOR: f64 = 1.15;

/// Animation duration for smooth transitions (300ms)
pub const ANIMATION_DURATION_MS: u64 = 300;

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
        // screen_dx = where we want it - where it actually is
        // positive dx means image drifted left, need to pan right (positive)
        let new_screen_pos = self.image_to_screen(image_pos.x, image_pos.y);
        let screen_dx = screen_x - new_screen_pos.x;
        let screen_dy = screen_y - new_screen_pos.y;
        self.pan(screen_dx, screen_dy);
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
    /// Velocity samples for averaging (screen pixels per second)
    velocity_samples: Vec<(DVec2, Instant)>,
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
    /// Zoom animation start time
    zoom_start_time: Option<Instant>,
    
    /// Last update time for physics
    last_update: Instant,
}

impl ViewportState {
    /// Create a new viewport state
    pub fn new(width: f64, height: f64, image_width: f64, image_height: f64) -> Self {
        let viewport = Viewport::new(width, height, image_width, image_height);
        let initial_zoom = viewport.zoom;
        Self {
            viewport,
            is_dragging: false,
            drag_start: DVec2::ZERO,
            velocity_samples: Vec::with_capacity(10),
            inertia_start_velocity: DVec2::ZERO,
            inertia_start_time: None,
            target_zoom: initial_zoom,
            zoom_start: initial_zoom,
            zoom_anchor_screen: DVec2::ZERO,
            zoom_anchor_image: DVec2::ZERO,
            zoom_start_time: None,
            last_update: Instant::now(),
        }
    }

    /// Start a drag operation
    pub fn start_drag(&mut self, x: f64, y: f64) {
        self.is_dragging = true;
        self.drag_start = DVec2::new(x, y);
        self.velocity_samples.clear();
        self.inertia_start_time = None; // Cancel any ongoing inertia
        self.zoom_start_time = None; // Cancel any ongoing zoom animation to prevent snap-back
        self.last_update = Instant::now();
    }

    /// Continue a drag operation
    pub fn drag_to(&mut self, x: f64, y: f64) {
        if !self.is_dragging {
            return;
        }

        let now = Instant::now();
        let current = DVec2::new(x, y);
        let delta = current - self.drag_start;
        
        // Store sample with timestamp for velocity calculation
        self.velocity_samples.push((delta, now));
        // Keep only recent samples (last 100ms)
        let cutoff = now - Duration::from_millis(100);
        self.velocity_samples.retain(|(_, t)| *t > cutoff);

        // Pan viewport
        self.viewport.pan(delta.x, delta.y);
        
        self.drag_start = current;
        self.last_update = now;
    }

    /// End a drag operation - start inertia animation
    pub fn end_drag(&mut self) {
        if !self.is_dragging {
            return;
        }

        self.is_dragging = false;
        
        // Calculate average velocity from recent samples
        if self.velocity_samples.len() >= 2 {
            let now = Instant::now();
            let mut total_delta = DVec2::ZERO;
            let mut total_time = 0.0;
            
            for (delta, _) in &self.velocity_samples {
                total_delta += *delta;
            }
            
            // Calculate time span of samples
            if let (Some((_, first_time)), Some((_, last_time))) = 
                (self.velocity_samples.first(), self.velocity_samples.last()) 
            {
                total_time = last_time.duration_since(*first_time).as_secs_f64();
            }
            
            if total_time > 0.001 {
                // Velocity in screen pixels per second
                let velocity = total_delta / total_time;
                // Only apply inertia if velocity is significant
                if velocity.length() > 50.0 {
                    self.inertia_start_velocity = velocity;
                    self.inertia_start_time = Some(now);
                    trace!("Starting pan inertia: velocity={:?}", velocity);
                }
            }
        }
        
        self.velocity_samples.clear();
        self.last_update = Instant::now();
    }

    /// Update animations (call every frame). Returns true if still animating.
    pub fn update(&mut self) -> bool {
        if self.is_dragging {
            return false;
        }

        let now = Instant::now();
        let animation_duration = Duration::from_millis(ANIMATION_DURATION_MS);
        let mut is_animating = false;

        // --- Pan inertia animation ---
        if let Some(start_time) = self.inertia_start_time {
            let elapsed = now.duration_since(start_time);
            
            if elapsed < animation_duration {
                // Ease-out: velocity decreases linearly to zero
                let t = elapsed.as_secs_f64() / animation_duration.as_secs_f64();
                let ease_out = 1.0 - t; // Linear ease-out
                
                let dt = now.duration_since(self.last_update).as_secs_f64();
                let current_velocity = self.inertia_start_velocity * ease_out;
                
                // Apply panning - velocity was measured in screen-space drag direction
                // which matches pan() semantics (positive = image moves right)
                self.viewport.pan(current_velocity.x * dt, current_velocity.y * dt);
                
                is_animating = true;
                trace!("Pan inertia: t={:.2}, velocity={:?}", t, current_velocity);
            } else {
                // Animation complete
                self.inertia_start_time = None;
            }
        }

        // --- Zoom animation ---
        if let Some(start_time) = self.zoom_start_time {
            let elapsed = now.duration_since(start_time);
            
            if elapsed < animation_duration {
                // Ease-out cubic: fast start, slow end
                let t = elapsed.as_secs_f64() / animation_duration.as_secs_f64();
                let ease_out = 1.0 - (1.0 - t).powi(3);
                
                // Interpolate zoom in log space for perceptually linear zoom
                let log_start = self.zoom_start.ln();
                let log_target = self.target_zoom.ln();
                let log_current = log_start + (log_target - log_start) * ease_out;
                let new_zoom = log_current.exp().max(MIN_ZOOM).min(MAX_ZOOM);
                
                // Apply zoom while keeping anchor point fixed
                self.viewport.zoom = new_zoom;
                
                // Adjust center to keep the image anchor under the screen anchor
                // screen_dx = where we want it - where it actually is
                // positive dx means image drifted left, need to pan right (positive)
                let new_screen_pos = self.viewport.image_to_screen(
                    self.zoom_anchor_image.x, 
                    self.zoom_anchor_image.y
                );
                let screen_dx = self.zoom_anchor_screen.x - new_screen_pos.x;
                let screen_dy = self.zoom_anchor_screen.y - new_screen_pos.y;
                self.viewport.pan(screen_dx, screen_dy);
                
                is_animating = true;
                trace!("Zoom animation: t={:.2}, zoom={:.4}", t, new_zoom);
            } else {
                // Animation complete - snap to target
                self.viewport.zoom = self.target_zoom;
                
                // Final adjustment to keep anchor point
                let new_screen_pos = self.viewport.image_to_screen(
                    self.zoom_anchor_image.x, 
                    self.zoom_anchor_image.y
                );
                let screen_dx = self.zoom_anchor_screen.x - new_screen_pos.x;
                let screen_dy = self.zoom_anchor_screen.y - new_screen_pos.y;
                self.viewport.pan(screen_dx, screen_dy);
                
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
        self.is_dragging = false;
        self.target_zoom = self.viewport.zoom;
    }

    /// Zoom at screen position with smooth animation
    pub fn zoom_at(&mut self, factor: f64, screen_x: f64, screen_y: f64) {
        // Calculate new target zoom
        let new_target = (self.target_zoom * factor).max(MIN_ZOOM).min(MAX_ZOOM);
        
        // If already animating, update the target but keep the same anchor
        // If not animating, start a new animation
        if self.zoom_start_time.is_none() {
            self.zoom_start = self.viewport.zoom;
            self.zoom_anchor_screen = DVec2::new(screen_x, screen_y);
            self.zoom_anchor_image = self.viewport.screen_to_image(screen_x, screen_y);
            self.zoom_start_time = Some(Instant::now());
        } else {
            // Update start to current animated position for smooth chaining
            self.zoom_start = self.viewport.zoom;
            // Keep the same screen anchor but update image anchor for current zoom
            self.zoom_anchor_image = self.viewport.screen_to_image(
                self.zoom_anchor_screen.x, 
                self.zoom_anchor_screen.y
            );
            self.zoom_start_time = Some(Instant::now());
        }
        
        self.target_zoom = new_target;
        trace!("Zoom requested: factor={}, target={}", factor, new_target);
    }

    /// Fit entire image in view (immediate, no animation)
    pub fn fit_to_view(&mut self) {
        self.stop();
        self.viewport.fit_to_view();
        self.target_zoom = self.viewport.zoom;
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
        self.is_dragging || self.inertia_start_time.is_some() || self.zoom_start_time.is_some()
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
