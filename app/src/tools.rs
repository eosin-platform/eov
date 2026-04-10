//! Tool handling functions for viewport interactions
//!
//! This module contains functions for handling user interactions with tools
//! like the region of interest selector and distance measurement tool.

use crate::state::{
    AppState, ImagePoint, Measurement, OpenFile, PaneId, RegionOfInterest, Tool,
    ToolInteractionState,
};
use crate::{MeasurementLine, ROIRect, ToolType, ViewportState};
/// Get the viewport state for a specific pane from an open file
pub fn pane_viewport_state(file: &OpenFile, pane: PaneId) -> Option<&ViewportState> {
    file.pane_state(pane).map(|pane_state| &pane_state.viewport)
}

/// Update the UI to reflect the current tool state
pub fn update_tool_state(ui: &crate::AppWindow, state: &AppState) {
    let tool_type = match state.current_tool {
        Tool::Navigate => ToolType::Navigate,
        Tool::RegionOfInterest => ToolType::RegionOfInterest,
        Tool::MeasureDistance => ToolType::MeasureDistance,
    };
    ui.set_current_tool(tool_type);
}

/// Create a hidden ROI rect (not visible)
pub fn hidden_roi_rect(ant_offset: f32) -> ROIRect {
    ROIRect {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: 0.0,
        visible: false,
        ant_offset,
    }
}

/// Create a hidden measurement line (not visible)
pub fn hidden_measurement_line() -> MeasurementLine {
    MeasurementLine {
        x1: 0.0,
        y1: 0.0,
        x2: 0.0,
        y2: 0.0,
        distance_um: 0.0,
        visible: false,
    }
}

/// Calculate overlay data for a pane (ROI rect, measurements, candidate measurement)
pub fn pane_overlay_data(
    state: &AppState,
    pane: PaneId,
) -> (ROIRect, Vec<MeasurementLine>, MeasurementLine) {
    let ant_offset = state.ant_offset;
    let mut roi_rect = hidden_roi_rect(ant_offset);
    let mut measurement_lines = Vec::new();
    let mut candidate_measurement = hidden_measurement_line();

    let Some(file_id) = state.active_file_id_for_pane(pane) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };
    let Some(file) = state.get_file(file_id) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };
    let Some(viewport_state) = pane_viewport_state(file, pane) else {
        return (roi_rect, measurement_lines, candidate_measurement);
    };

    let vp = &viewport_state.viewport;
    let bounds = vp.bounds();
    let mpp = file
        .wsi
        .properties()
        .mpp_x
        .zip(file.wsi.properties().mpp_y)
        .map(|(mx, my)| (mx + my) / 2.0)
        .unwrap_or(0.0);

    let committed_roi = file.roi.filter(|roi| roi.pane == pane);
    let roi_to_display =
        if state.focused_pane == pane && state.current_tool == Tool::RegionOfInterest {
            if let ToolInteractionState::Dragging(start) = state.tool_state {
                state
                    .candidate_point
                    .map(|end| RegionOfInterest::from_points(start, end, pane))
                    .or(committed_roi)
            } else {
                committed_roi
            }
        } else {
            committed_roi
        };

    if let Some(roi) = roi_to_display {
        roi_rect = ROIRect {
            x: ((roi.x - bounds.left) * vp.zoom) as f32,
            y: ((roi.y - bounds.top) * vp.zoom) as f32,
            width: (roi.width * vp.zoom) as f32,
            height: (roi.height * vp.zoom) as f32,
            visible: true,
            ant_offset,
        };
    }

    measurement_lines = file
        .measurements
        .iter()
        .filter(|measurement| measurement.pane == pane)
        .map(|measurement| {
            let p1 = vp.image_to_screen(measurement.start.x, measurement.start.y);
            let p2 = vp.image_to_screen(measurement.end.x, measurement.end.y);
            MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um: (measurement.distance() * mpp) as f32,
                visible: true,
            }
        })
        .collect();

    if state.focused_pane == pane && state.current_tool == Tool::MeasureDistance {
        if let (
            ToolInteractionState::Dragging(start) | ToolInteractionState::FirstPointPlaced(start),
            Some(end),
        ) = (state.tool_state, state.candidate_point)
        {
            let p1 = vp.image_to_screen(start.x, start.y);
            let p2 = vp.image_to_screen(end.x, end.y);
            let candidate = Measurement { pane, start, end };
            candidate_measurement = MeasurementLine {
                x1: p1.x as f32,
                y1: p1.y as f32,
                x2: p2.x as f32,
                y2: p2.y as f32,
                distance_um: (candidate.distance() * mpp) as f32,
                visible: true,
            };
        }
    }

    (roi_rect, measurement_lines, candidate_measurement)
}

/// Check if there's an active ROI overlay being drawn or displayed
pub fn has_active_roi_overlay(state: &AppState) -> bool {
    if state.current_tool == Tool::RegionOfInterest
        && matches!(state.tool_state, ToolInteractionState::Dragging(_))
        && state.candidate_point.is_some()
    {
        return true;
    }

    state.panes.iter().enumerate().any(|(pane_index, _)| {
        let pane = PaneId(pane_index);
        state
            .active_file_id_for_pane(pane)
            .and_then(|id| state.get_file(id))
            .and_then(|file| file.roi)
            .map(|roi| roi.pane == pane)
            .unwrap_or(false)
    })
}

/// Update tool overlays in the UI (currently a no-op placeholder)
pub fn update_tool_overlays(_ui: &crate::AppWindow, _state: &AppState) {}

/// Handle mouse down event for the current tool
pub fn handle_tool_mouse_down(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    match state.current_tool {
        Tool::Navigate => {
            // Should not happen - Navigate uses LMB for panning
        }
        Tool::RegionOfInterest => {
            state.tool_state = ToolInteractionState::Dragging(point);
            state.candidate_point = Some(point);
        }
        Tool::MeasureDistance => {
            // If we already have a first point placed (click-click mode),
            // this second click commits the measurement.
            if let ToolInteractionState::FirstPointPlaced(start) = state.tool_state {
                let measurement = Measurement {
                    pane: focused_pane,
                    start,
                    end: point,
                };
                file.measurements.clear();
                file.measurements.push(measurement);
                state.tool_state = ToolInteractionState::Idle;
                state.candidate_point = None;
            } else {
                // Start a new measurement — clear any previous one
                file.measurements.clear();
                state.tool_state = ToolInteractionState::Dragging(point);
                state.candidate_point = Some(point);
            }
        }
    }
}

/// Handle mouse move event for the current tool
pub fn handle_tool_mouse_move(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;

    let Some(file) = state.open_files.iter().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let point = ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    // Update candidate point during drag or first-point-placed
    match state.tool_state {
        ToolInteractionState::Dragging(_) | ToolInteractionState::FirstPointPlaced(_) => {
            state.candidate_point = Some(point);
        }
        _ => {}
    }
}

/// Handle mouse up event for the current tool
pub fn handle_tool_mouse_up(state: &mut AppState, screen_x: f64, screen_y: f64) {
    let Some(file_id) = state.active_file_id else {
        return;
    };

    let focused_pane = state.focused_pane;
    let current_tool = state.current_tool;

    let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
        return;
    };

    // Get the active viewport for coordinate conversion
    let Some(viewport_state) = pane_viewport_state(file, focused_pane) else {
        return;
    };

    // Convert screen coordinates to image coordinates
    let image_point = viewport_state.screen_to_image(screen_x, screen_y);
    let end_point = ImagePoint {
        x: image_point.0,
        y: image_point.1,
    };

    if let ToolInteractionState::Dragging(start) = state.tool_state {
        match current_tool {
            Tool::Navigate => {}
            Tool::RegionOfInterest => {
                // Create ROI from the two points
                let roi = RegionOfInterest::from_points(start, end_point, focused_pane);
                if roi.is_valid() {
                    file.roi = Some(roi);
                }
                state.tool_state = ToolInteractionState::Idle;
                state.candidate_point = None;
            }
            Tool::MeasureDistance => {
                // Check if the mouse moved enough to be a drag
                let dx = end_point.x - start.x;
                let dy = end_point.y - start.y;
                let dist_sq = dx * dx + dy * dy;
                // Threshold: 5 pixels in image space
                if dist_sq > 25.0 {
                    // Drag completed — commit measurement
                    let measurement = Measurement {
                        pane: focused_pane,
                        start,
                        end: end_point,
                    };
                    file.measurements.clear();
                    file.measurements.push(measurement);
                    state.tool_state = ToolInteractionState::Idle;
                    state.candidate_point = None;
                } else {
                    // Click (mouse didn't move much) — enter click-click mode
                    state.tool_state = ToolInteractionState::FirstPointPlaced(start);
                    // Keep candidate_point so the preview line follows the mouse
                }
            }
        }
    } else {
        // For FirstPointPlaced, mouse-up is ignored (commit happens on next mouse-down)
        // Reset for other states
        if !matches!(state.tool_state, ToolInteractionState::FirstPointPlaced(_)) {
            state.tool_state = ToolInteractionState::Idle;
            state.candidate_point = None;
        }
    }
}
