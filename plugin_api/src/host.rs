//! Shared host-facing types for plugin access to application state.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostToolMode {
    Navigate,
    RegionOfInterest,
    MeasureDistance,
    PointAnnotation,
    PolygonAnnotation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SidebarRequest {
    pub button_id: Option<String>,
    pub width_px: u32,
    pub ui_path: String,
    pub component: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveSidebar {
    pub plugin_id: String,
    pub button_id: Option<String>,
    pub width_px: u32,
    pub ui_path: String,
    pub component: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfirmationDialogRequest {
    pub title: String,
    pub message: String,
    pub confirm_label: String,
    pub cancel_label: String,
    pub confirm_callback: Option<String>,
    pub confirm_args_json: Option<String>,
    pub cancel_callback: Option<String>,
    pub cancel_args_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HostSnapshot {
    pub app_name: String,
    pub app_version: String,
    pub render_backend: String,
    pub filtering_mode: String,
    pub split_enabled: bool,
    pub focused_pane: u32,
    pub open_files: Vec<OpenFileInfo>,
    pub active_file: Option<OpenFileInfo>,
    pub active_viewport: Option<ViewportSnapshot>,
    pub recent_files: Vec<String>,
    pub active_sidebar: Option<ActiveSidebar>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenFileInfo {
    pub file_id: i32,
    pub path: String,
    pub filename: String,
    pub width: u64,
    pub height: u64,
    pub level_count: u32,
    pub vendor: Option<String>,
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    pub objective_power: Option<f64>,
    pub scan_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViewportSnapshot {
    pub pane_index: u32,
    pub file_id: i32,
    pub file_path: String,
    pub filename: String,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub width: f64,
    pub height: f64,
    pub image_width: f64,
    pub image_height: f64,
    pub bounds_left: f64,
    pub bounds_top: f64,
    pub bounds_right: f64,
    pub bounds_bottom: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}
