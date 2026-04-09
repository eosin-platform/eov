//! Application state management

use crate::tile_loader::TileLoader;
use common::{TileManager, ViewportState, WsiFile};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderBackend {
    #[default]
    Cpu,
    Gpu,
}

/// Maximum number of recently opened files to track
const MAX_RECENT_FILES: usize = 10;

/// Recently opened file entry
#[derive(Debug, Clone)]
pub struct RecentFile {
    /// Full path to the file
    pub path: PathBuf,
    /// Display name (filename)
    pub name: String,
    /// Last opened timestamp
    #[allow(dead_code)]
    pub last_opened: std::time::SystemTime,
}

impl RecentFile {
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        Self {
            path,
            name,
            last_opened: std::time::SystemTime::now(),
        }
    }
}

/// Available tools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    /// Navigate/Pan tool (default) - LMB pans viewport
    #[default]
    Navigate,
    /// Region of Interest tool - click and drag to define ROI
    RegionOfInterest,
    /// Measure Distance tool - measure distance between two points
    MeasureDistance,
}

/// Point in image coordinates
#[derive(Debug, Clone, Copy, Default)]
pub struct ImagePoint {
    pub x: f64,
    pub y: f64,
}

/// Region of Interest
#[derive(Debug, Clone, Copy, Default)]
pub struct RegionOfInterest {
    /// Pane where the ROI was created
    pub pane: PaneId,
    /// Top-left corner in image coordinates
    pub x: f64,
    pub y: f64,
    /// Width in image coordinates  
    pub width: f64,
    /// Height in image coordinates
    pub height: f64,
}

impl RegionOfInterest {
    pub fn is_valid(&self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }

    pub fn from_points(p1: ImagePoint, p2: ImagePoint, pane: PaneId) -> Self {
        let x = p1.x.min(p2.x);
        let y = p1.y.min(p2.y);
        let width = (p1.x - p2.x).abs();
        let height = (p1.y - p2.y).abs();
        Self {
            pane,
            x,
            y,
            width,
            height,
        }
    }
}

/// Measurement between two points
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Measurement {
    pub pane: PaneId,
    pub start: ImagePoint,
    pub end: ImagePoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileRequestSignature {
    pub level: u32,
    pub margin_tiles: i32,
    pub start_x: u64,
    pub start_y: u64,
    pub end_x: u64,
    pub end_y: u64,
    pub tile_size: u32,
}

#[allow(dead_code)]
impl Measurement {
    /// Calculate distance in image pixels
    pub fn distance(&self) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Tool interaction state
#[derive(Debug, Clone, Copy, Default)]
pub enum ToolInteractionState {
    #[default]
    Idle,
    /// First point placed for ROI or measurement
    #[allow(dead_code)]
    FirstPointPlaced(ImagePoint),
    /// Currently dragging (for ROI or drag-style measurement)
    Dragging(ImagePoint),
}

/// A single open file with its associated state
pub struct OpenFile {
    /// Unique identifier
    pub id: i32,
    /// File path
    pub path: PathBuf,
    /// Display filename
    pub filename: String,
    /// WSI file handle
    pub wsi: WsiFile,
    /// Tile manager for this file
    pub tile_manager: Arc<TileManager>,
    /// Background tile loader
    pub tile_loader: TileLoader,
    /// Primary viewport state
    pub viewport: ViewportState,
    /// Secondary viewport state (for split view)
    pub secondary_viewport: Option<ViewportState>,
    /// Thumbnail for minimap (RGBA data)
    pub thumbnail: Option<Vec<u8>>,
    /// Whether thumbnail has been uploaded to UI
    #[allow(dead_code)]
    pub thumbnail_set: bool,
    /// Region of interest (if set)
    pub roi: Option<RegionOfInterest>,
    /// Measurements
    pub measurements: Vec<Measurement>,
    /// Reusable render buffer to avoid per-frame allocation
    pub render_buffer: Vec<u8>,
    /// Last rendered viewport state (for dirty checking)
    pub last_render_zoom: f64,
    pub last_render_center_x: f64,
    pub last_render_center_y: f64,
    pub last_render_width: f64,
    pub last_render_height: f64,
    /// Last rendered pyramid level (for detecting level changes)
    pub last_render_level: u32,
    /// Number of tiles loaded since last render (at current level)
    pub tiles_loaded_since_render: u32,
    /// Frame counter for dirty tracking (0 means never rendered)
    pub frame_count: u32,
    /// Last render timestamp (for throttling tile update renders)
    pub last_render_time: std::time::Instant,
    /// Tile loader epoch observed by the primary viewport render path
    pub last_seen_tile_epoch: u64,
    /// Last tile request signature submitted for the primary viewport
    pub last_primary_request: Option<TileRequestSignature>,
    /// Last rendered secondary viewport state (for dirty checking)
    pub last_secondary_zoom: f64,
    pub last_secondary_center_x: f64,
    pub last_secondary_center_y: f64,
    pub last_secondary_width: f64,
    pub last_secondary_height: f64,
    /// Tile loader epoch observed by the secondary viewport render path
    pub last_seen_secondary_tile_epoch: u64,
    /// Last tile request signature submitted for the secondary viewport
    pub last_secondary_request: Option<TileRequestSignature>,
    /// Reusable render buffer for secondary viewport
    pub secondary_render_buffer: Vec<u8>,
}

impl OpenFile {
    pub fn invalidate_render_state(&mut self) {
        self.frame_count = 0;
        self.last_render_zoom = 0.0;
        self.last_render_center_x = 0.0;
        self.last_render_center_y = 0.0;
        self.last_render_width = 0.0;
        self.last_render_height = 0.0;
        self.last_render_level = u32::MAX;
        self.tiles_loaded_since_render = 0;
        self.last_seen_tile_epoch = 0;
        self.last_primary_request = None;
        self.last_secondary_zoom = 0.0;
        self.last_secondary_center_x = 0.0;
        self.last_secondary_center_y = 0.0;
        self.last_secondary_width = 0.0;
        self.last_secondary_height = 0.0;
        self.last_seen_secondary_tile_epoch = 0;
        self.last_secondary_request = None;
    }
}

/// Pane identifier (left/primary or right/secondary)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaneId {
    #[default]
    Primary,
    Secondary,
}

impl PaneId {
    pub fn as_index(self) -> i32 {
        match self {
            Self::Primary => 0,
            Self::Secondary => 1,
        }
    }
}

/// Application state
pub struct AppState {
    /// Currently open files
    pub open_files: Vec<OpenFile>,
    /// Active tab ID for the currently focused pane
    pub active_file_id: Option<i32>,
    /// Active tab ID in the primary pane
    pub primary_active_tab_id: Option<i32>,
    /// Active tab ID in the secondary pane
    pub secondary_active_tab_id: Option<i32>,
    /// Active file ID
    /// Ordered tab IDs owned by the primary pane
    pub primary_tabs: Vec<i32>,
    /// Ordered tab IDs owned by the secondary pane
    pub secondary_tabs: Vec<i32>,
    /// Next file ID
    next_id: i32,
    /// Whether split view is enabled
    pub split_enabled: bool,
    /// Split position (0.0-1.0, percentage from left)
    pub split_position: f32,
    /// Currently focused pane
    pub focused_pane: PaneId,
    /// Current active tool
    pub current_tool: Tool,
    /// Current tool interaction state
    pub tool_state: ToolInteractionState,
    /// Candidate point (for visual feedback during tool use)
    pub candidate_point: Option<ImagePoint>,
    /// Animation offset for marching ants (wraps at 16)
    pub ant_offset: f32,
    /// Debug mode enabled (--debug flag)
    #[allow(dead_code)]
    pub debug_mode: bool,
    /// Frame timestamps for FPS calculation
    pub frame_times: Vec<std::time::Instant>,
    /// Current FPS value
    pub current_fps: f32,
    /// Recently opened files (most recent first)
    pub recent_files: Vec<RecentFile>,
    /// IDs of tabs that are "home" tabs (no file open)
    pub home_tabs: Vec<i32>,
    /// Last file ID rendered into the primary surface
    pub last_primary_rendered_file_id: Option<i32>,
    /// Last file ID rendered into the secondary surface
    pub last_secondary_rendered_file_id: Option<i32>,
    /// The selected rendering backend
    pub render_backend: RenderBackend,
    /// Whether GPU rendering is available on this system
    pub gpu_backend_available: bool,
    /// Whether a new frame should be rendered as soon as possible
    pub needs_render: bool,
    /// Whether the render loop timer is currently running
    pub render_loop_running: bool,
}

impl AppState {
    /// Create a new application state
    pub fn new(debug_mode: bool) -> Self {
        let recent_files = Self::load_recent_files();
        Self {
            open_files: Vec::new(),
            active_file_id: None,
            primary_active_tab_id: None,
            secondary_active_tab_id: None,
            primary_tabs: Vec::new(),
            secondary_tabs: Vec::new(),
            next_id: 1,
            split_enabled: false,
            split_position: 0.5,
            focused_pane: PaneId::Primary,
            current_tool: Tool::Navigate,
            tool_state: ToolInteractionState::Idle,
            candidate_point: None,
            ant_offset: 0.0,
            debug_mode,
            frame_times: Vec::with_capacity(60),
            current_fps: 0.0,
            recent_files,
            home_tabs: Vec::new(),
            last_primary_rendered_file_id: None,
            last_secondary_rendered_file_id: None,
            render_backend: RenderBackend::Cpu,
            gpu_backend_available: false,
            needs_render: true,
            render_loop_running: false,
        }
    }

    /// Get the config directory for storing recent files
    fn config_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("eosmol"))
    }

    /// Get the path to the recent files store
    fn recent_files_path() -> Option<PathBuf> {
        Self::config_dir().map(|p| p.join("recent_files.txt"))
    }

    /// Load recently opened files from disk
    fn load_recent_files() -> Vec<RecentFile> {
        let Some(path) = Self::recent_files_path() else {
            return Vec::new();
        };

        let Ok(content) = fs::read_to_string(&path) else {
            return Vec::new();
        };

        content
            .lines()
            .filter_map(|line| {
                let path = PathBuf::from(line.trim());
                if path.exists() {
                    Some(RecentFile::new(path))
                } else {
                    None
                }
            })
            .take(MAX_RECENT_FILES)
            .collect()
    }

    /// Save recently opened files to disk
    fn save_recent_files(&self) {
        let Some(dir) = Self::config_dir() else {
            return;
        };

        if !dir.exists() && fs::create_dir_all(&dir).is_err() {
            return;
        }

        let Some(path) = Self::recent_files_path() else {
            return;
        };

        let content: String = self
            .recent_files
            .iter()
            .map(|f| f.path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let _ = fs::write(path, content);
    }

    /// Add a file to the recently opened list
    pub fn add_to_recent(&mut self, path: &PathBuf) {
        // Remove if already exists (we'll re-add at top)
        self.recent_files.retain(|f| f.path != *path);

        // Add to front
        self.recent_files.insert(0, RecentFile::new(path.clone()));

        // Trim to max size
        self.recent_files.truncate(MAX_RECENT_FILES);

        // Save to disk
        self.save_recent_files();
    }

    fn tab_ids_for_pane(&self, pane: PaneId) -> &[i32] {
        match pane {
            PaneId::Primary => &self.primary_tabs,
            PaneId::Secondary => &self.secondary_tabs,
        }
    }

    fn tab_ids_for_pane_mut(&mut self, pane: PaneId) -> &mut Vec<i32> {
        match pane {
            PaneId::Primary => &mut self.primary_tabs,
            PaneId::Secondary => &mut self.secondary_tabs,
        }
    }

    fn active_tab_id_for_pane_mut(&mut self, pane: PaneId) -> &mut Option<i32> {
        match pane {
            PaneId::Primary => &mut self.primary_active_tab_id,
            PaneId::Secondary => &mut self.secondary_active_tab_id,
        }
    }

    fn next_tab_after_removal(tabs: &[i32], removed_index: usize) -> Option<i32> {
        if tabs.is_empty() {
            None
        } else if removed_index < tabs.len() {
            Some(tabs[removed_index])
        } else {
            tabs.last().copied()
        }
    }

    fn sync_active_file_id(&mut self) {
        self.active_file_id = self.active_tab_id_for_pane(self.focused_pane);
    }

    pub fn active_tab_id_for_pane(&self, pane: PaneId) -> Option<i32> {
        match pane {
            PaneId::Primary => self.primary_active_tab_id,
            PaneId::Secondary => self.secondary_active_tab_id,
        }
    }

    pub fn active_file_id_for_pane(&self, pane: PaneId) -> Option<i32> {
        self.active_tab_id_for_pane(pane)
            .filter(|id| !self.is_home_tab(*id))
    }

    pub fn tabs_for_pane(&self, pane: PaneId) -> &[i32] {
        self.tab_ids_for_pane(pane)
    }

    pub fn is_home_tab_active_in_pane(&self, pane: PaneId) -> bool {
        self.active_tab_id_for_pane(pane)
            .map(|id| self.is_home_tab(id))
            .unwrap_or(false)
    }

    pub fn set_focused_pane(&mut self, pane: PaneId) {
        self.focused_pane = pane;
        self.sync_active_file_id();
    }

    fn ensure_secondary_viewport_for_file(&mut self, id: i32) {
        if let Some(file) = self.get_file_mut(id)
            && file.secondary_viewport.is_none()
        {
            file.secondary_viewport = Some(file.viewport.clone());
        }
    }

    fn add_tab_to_pane_if_missing(&mut self, pane: PaneId, id: i32) {
        let tabs = self.tab_ids_for_pane_mut(pane);
        if !tabs.contains(&id) {
            tabs.push(id);
        }
        if pane == PaneId::Secondary {
            self.ensure_secondary_viewport_for_file(id);
        }
    }

    fn remove_tab_from_pane(&mut self, pane: PaneId, id: i32) {
        let removed_index = {
            let tabs = self.tab_ids_for_pane_mut(pane);
            tabs.iter()
                .position(|&tab_id| tab_id == id)
                .inspect(|&index| {
                    tabs.remove(index);
                })
        };

        if let Some(index) = removed_index {
            if self.active_tab_id_for_pane(pane) == Some(id) {
                let replacement = Self::next_tab_after_removal(self.tab_ids_for_pane(pane), index);
                *self.active_tab_id_for_pane_mut(pane) = replacement;
            }
            self.normalize_split_after_tab_change();
            self.sync_active_file_id();
        }
    }

    fn normalize_split_after_tab_change(&mut self) {
        if !self.split_enabled {
            return;
        }

        if self.primary_tabs.is_empty() && self.secondary_tabs.is_empty() {
            self.split_enabled = false;
            self.primary_active_tab_id = None;
            self.secondary_active_tab_id = None;
            self.set_focused_pane(PaneId::Primary);
            return;
        }

        if self.primary_tabs.is_empty() {
            self.primary_tabs = self.secondary_tabs.clone();
            self.primary_active_tab_id = self.secondary_active_tab_id;
            self.secondary_tabs.clear();
            self.secondary_active_tab_id = None;
            self.split_enabled = false;
            self.set_focused_pane(PaneId::Primary);
            self.needs_render = true;
            return;
        }

        if self.secondary_tabs.is_empty() {
            self.split_enabled = false;
            self.secondary_active_tab_id = None;
            self.set_focused_pane(PaneId::Primary);
            self.needs_render = true;
        }
    }

    fn is_known_tab(&self, id: i32) -> bool {
        self.is_home_tab(id) || self.open_files.iter().any(|file| file.id == id)
    }

    pub fn duplicate_tab_to_pane(&mut self, id: i32, pane: PaneId) {
        if !self.is_known_tab(id) {
            return;
        }
        self.add_tab_to_pane_if_missing(pane, id);
        *self.active_tab_id_for_pane_mut(pane) = Some(id);
        self.needs_render = true;
        self.sync_active_file_id();
    }

    pub fn move_tab_between_panes(&mut self, id: i32, from: PaneId, to: PaneId) {
        if from == to || !self.is_known_tab(id) {
            return;
        }
        self.remove_tab_from_pane(from, id);
        self.add_tab_to_pane_if_missing(to, id);
        *self.active_tab_id_for_pane_mut(to) = Some(id);
        if self.focused_pane == from {
            self.set_focused_pane(to);
        } else {
            self.sync_active_file_id();
        }
        self.normalize_split_after_tab_change();
        self.needs_render = true;
    }

    /// Reorder a tab within the same pane by moving it to a new index position.
    pub fn reorder_tab(&mut self, pane: PaneId, id: i32, new_index: i32) {
        let tabs = self.tab_ids_for_pane_mut(pane);
        if let Some(current_pos) = tabs.iter().position(|&tab_id| tab_id == id) {
            let new_pos = (new_index as usize).min(tabs.len().saturating_sub(1));
            if current_pos != new_pos {
                let tab_id = tabs.remove(current_pos);
                // Adjust insertion index if removing affected position
                let insert_pos = if current_pos < new_pos {
                    new_pos.min(tabs.len())
                } else {
                    new_pos.min(tabs.len())
                };
                tabs.insert(insert_pos, tab_id);
            }
        }
    }

    pub fn activate_tab_in_pane(&mut self, pane: PaneId, id: i32) {
        if !self.is_known_tab(id) {
            return;
        }
        self.add_tab_to_pane_if_missing(pane, id);
        *self.active_tab_id_for_pane_mut(pane) = Some(id);
        if pane == PaneId::Secondary {
            self.ensure_secondary_viewport_for_file(id);
        }
        self.set_focused_pane(pane);
        self.needs_render = true;
    }

    /// Create a new home tab and return its ID.
    /// If the pane already has a home tab, switch to it instead.
    pub fn create_home_tab(&mut self) -> i32 {
        let pane = if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::Primary
        };
        // If the pane already contains a home tab, just activate it
        let tabs = self.tabs_for_pane(pane);
        for &tab_id in tabs {
            if self.is_home_tab(tab_id) {
                self.activate_tab_in_pane(pane, tab_id);
                return tab_id;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.home_tabs.push(id);
        self.activate_tab_in_pane(pane, id);
        id
    }

    /// Close a home tab
    pub fn close_home_tab(&mut self, id: i32) {
        self.home_tabs.retain(|&tab_id| tab_id != id);

        self.remove_tab_from_pane(PaneId::Primary, id);
        self.remove_tab_from_pane(PaneId::Secondary, id);
        self.needs_render = true;
    }

    pub fn close_tab_in_pane(&mut self, pane: PaneId, id: i32) {
        if !self.is_known_tab(id) {
            return;
        }

        self.remove_tab_from_pane(pane, id);

        let still_open_elsewhere =
            self.primary_tabs.contains(&id) || self.secondary_tabs.contains(&id);
        if still_open_elsewhere {
            self.needs_render = true;
            self.sync_active_file_id();
            return;
        }

        if self.is_home_tab(id) {
            self.home_tabs.retain(|&tab_id| tab_id != id);
            self.needs_render = true;
            self.sync_active_file_id();
            return;
        }

        self.close_file(id);
    }

    /// Check if an ID is a home tab
    pub fn is_home_tab(&self, id: i32) -> bool {
        self.home_tabs.contains(&id)
    }

    /// Update FPS counter (call once per frame)
    pub fn update_fps(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push(now);

        // Keep only timestamps from the last second
        let one_second_ago = now - std::time::Duration::from_secs(1);
        self.frame_times.retain(|t| *t > one_second_ago);

        // FPS = number of frames in the last second
        self.current_fps = self.frame_times.len() as f32;
    }

    /// Add a new open file
    pub fn add_file(
        &mut self,
        path: PathBuf,
        wsi: WsiFile,
        tile_manager: Arc<TileManager>,
        tile_loader: TileLoader,
        viewport: ViewportState,
        thumbnail: Option<Vec<u8>>,
    ) -> i32 {
        // Add to recent files
        self.add_to_recent(&path);

        let id = self.next_id;
        self.next_id += 1;

        let filename = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        self.open_files.push(OpenFile {
            id,
            path,
            filename,
            wsi,
            tile_manager,
            tile_loader,
            viewport,
            secondary_viewport: None,
            thumbnail,
            thumbnail_set: false,
            roi: None,
            measurements: Vec::new(),
            render_buffer: Vec::new(),
            last_render_zoom: 0.0,
            last_render_center_x: 0.0,
            last_render_center_y: 0.0,
            last_render_width: 0.0,
            last_render_height: 0.0,
            last_render_level: u32::MAX,
            tiles_loaded_since_render: 0,
            frame_count: 0,
            last_render_time: std::time::Instant::now(),
            last_seen_tile_epoch: 0,
            last_primary_request: None,
            last_secondary_zoom: 0.0,
            last_secondary_center_x: 0.0,
            last_secondary_center_y: 0.0,
            last_secondary_width: 0.0,
            last_secondary_height: 0.0,
            last_seen_secondary_tile_epoch: 0,
            last_secondary_request: None,
            secondary_render_buffer: Vec::new(),
        });

        let target_pane = if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::Primary
        };
        self.activate_tab_in_pane(target_pane, id);
        id
    }

    /// Set the current tool and reset interaction state
    pub fn set_tool(&mut self, tool: Tool) {
        self.current_tool = tool;
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
        self.needs_render = true;
        // Clear ROI and measurements when switching tools
        if let Some(file) = self
            .open_files
            .iter_mut()
            .find(|f| Some(f.id) == self.active_file_id)
        {
            file.roi = None;
            file.measurements.clear();
        }
    }

    /// Cancel current tool operation and return to Navigate
    pub fn cancel_tool(&mut self) {
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
        self.current_tool = Tool::Navigate;
        self.needs_render = true;
        // Clear ROI and measurements when cancelling
        if let Some(file) = self
            .open_files
            .iter_mut()
            .find(|f| Some(f.id) == self.active_file_id)
        {
            file.roi = None;
            file.measurements.clear();
        }
    }

    /// Enable split view for the current file
    pub fn enable_split(&mut self) {
        self.split_enabled = true;
        if self.secondary_tabs.is_empty()
            && let Some(primary_id) = self.primary_active_tab_id
        {
            self.duplicate_tab_to_pane(primary_id, PaneId::Secondary);
        }
        self.set_focused_pane(PaneId::Secondary);
        self.needs_render = true;
    }

    /// Disable split view
    #[allow(dead_code)]
    pub fn disable_split(&mut self) {
        for id in self.secondary_tabs.clone() {
            if !self.primary_tabs.contains(&id) {
                self.primary_tabs.push(id);
            }
        }
        if self.focused_pane == PaneId::Secondary
            && let Some(id) = self.secondary_active_tab_id
        {
            self.primary_active_tab_id = Some(id);
        }
        self.split_enabled = false;
        self.secondary_tabs.clear();
        self.secondary_active_tab_id = None;
        self.set_focused_pane(PaneId::Primary);
        self.needs_render = true;
    }

    /// Get the viewport for the current focused pane
    #[allow(dead_code)]
    pub fn focused_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let active_id = self.active_file_id_for_pane(self.focused_pane)?;
        self.open_files
            .iter_mut()
            .find(|f| f.id == active_id)
            .and_then(|f| match self.focused_pane {
                PaneId::Primary => Some(&mut f.viewport),
                PaneId::Secondary => f.secondary_viewport.as_mut(),
            })
    }

    /// Activate a file by ID
    #[allow(dead_code)]
    pub fn activate_file(&mut self, id: i32) {
        self.activate_tab_in_pane(self.focused_pane, id);
    }

    /// Close a file by ID
    pub fn close_file(&mut self, id: i32) {
        let index = self.open_files.iter().position(|f| f.id == id);

        if let Some(idx) = index {
            self.open_files.remove(idx);

            self.remove_tab_from_pane(PaneId::Primary, id);
            self.remove_tab_from_pane(PaneId::Secondary, id);

            if self.primary_tabs.is_empty() {
                self.primary_active_tab_id = None;
            }
            if self.secondary_tabs.is_empty() {
                self.secondary_active_tab_id = None;
            }

            if self.primary_active_tab_id.is_none() {
                self.primary_active_tab_id = self.primary_tabs.first().copied();
            }
            if self.secondary_active_tab_id.is_none() {
                self.secondary_active_tab_id = self.secondary_tabs.first().copied();
            }

            self.last_primary_rendered_file_id = self
                .last_primary_rendered_file_id
                .filter(|&file_id| file_id != id);
            self.last_secondary_rendered_file_id = self
                .last_secondary_rendered_file_id
                .filter(|&file_id| file_id != id);

            self.needs_render = true;
            self.sync_active_file_id();
        }
    }

    pub fn request_render(&mut self) {
        self.needs_render = true;
    }

    pub fn select_render_backend(&mut self, backend: RenderBackend) {
        let next_backend = match backend {
            RenderBackend::Gpu if !self.gpu_backend_available => RenderBackend::Cpu,
            other => other,
        };
        if self.render_backend != next_backend {
            self.render_backend = next_backend;
            for file in &mut self.open_files {
                file.invalidate_render_state();
            }
        }
        self.needs_render = true;
    }

    /// Get a file by ID
    pub fn get_file(&self, id: i32) -> Option<&OpenFile> {
        self.open_files.iter().find(|f| f.id == id)
    }

    /// Get a mutable file by ID
    pub fn get_file_mut(&mut self, id: i32) -> Option<&mut OpenFile> {
        self.open_files.iter_mut().find(|f| f.id == id)
    }

    /// Get the active file
    #[allow(dead_code)]
    pub fn active_file(&self) -> Option<&OpenFile> {
        self.active_file_id
            .filter(|id| !self.is_home_tab(*id))
            .and_then(|id| self.get_file(id))
    }

    /// Get the active file mutably
    #[allow(dead_code)]
    pub fn active_file_mut(&mut self) -> Option<&mut OpenFile> {
        self.active_file_id
            .filter(|id| !self.is_home_tab(*id))
            .and_then(|id| self.get_file_mut(id))
    }

    /// Get the active viewport
    #[allow(dead_code)]
    pub fn active_viewport(&self) -> Option<&ViewportState> {
        let pane = if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::Primary
        };
        let active_id = self.active_file_id_for_pane(pane)?;
        self.get_file(active_id).and_then(|file| match pane {
            PaneId::Primary => Some(&file.viewport),
            PaneId::Secondary => file.secondary_viewport.as_ref(),
        })
    }

    /// Get the active viewport mutably (respects focused pane in split view)
    pub fn active_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let pane = self.focused_pane;
        let effective_pane = if self.split_enabled {
            pane
        } else {
            PaneId::Primary
        };
        let active_id = self.active_file_id_for_pane(effective_pane)?;
        self.open_files
            .iter_mut()
            .find(|f| f.id == active_id)
            .and_then(|f| match effective_pane {
                PaneId::Primary => Some(&mut f.viewport),
                PaneId::Secondary => f.secondary_viewport.as_mut(),
            })
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(false)
    }
}
