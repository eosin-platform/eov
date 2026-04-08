//! Application state management

use common::{TileCache, TileManager, ViewportState, WsiFile};
use crate::tile_loader::TileLoader;
use std::path::PathBuf;
use std::sync::Arc;
use std::fs;

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
    
    pub fn from_points(p1: ImagePoint, p2: ImagePoint) -> Self {
        let x = p1.x.min(p2.x);
        let y = p1.y.min(p2.y);
        let width = (p1.x - p2.x).abs();
        let height = (p1.y - p2.y).abs();
        Self { x, y, width, height }
    }
}

/// Measurement between two points
#[derive(Debug, Clone, Copy, Default)]
pub struct Measurement {
    pub start: ImagePoint,
    pub end: ImagePoint,
}

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
    /// Last rendered secondary viewport state (for dirty checking)
    pub last_secondary_zoom: f64,
    pub last_secondary_center_x: f64,
    pub last_secondary_center_y: f64,
    pub last_secondary_width: f64,
    pub last_secondary_height: f64,
}

/// Pane identifier (left/primary or right/secondary)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneId {
    Primary,
    Secondary,
}

/// Application state
pub struct AppState {
    /// Currently open files
    pub open_files: Vec<OpenFile>,
    /// Active file ID
    pub active_file_id: Option<i32>,
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
    pub debug_mode: bool,
    /// Frame timestamps for FPS calculation
    pub frame_times: Vec<std::time::Instant>,
    /// Current FPS value
    pub current_fps: f32,
    /// Recently opened files (most recent first)
    pub recent_files: Vec<RecentFile>,
    /// IDs of tabs that are "home" tabs (no file open)
    pub home_tabs: Vec<i32>,
    /// Last file ID that was rendered (to detect tab switches)
    pub last_rendered_file_id: Option<i32>,
}

impl AppState {
    /// Create a new application state
    pub fn new(debug_mode: bool) -> Self {
        let recent_files = Self::load_recent_files();
        Self {
            open_files: Vec::new(),
            active_file_id: None,
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
            last_rendered_file_id: None,
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
        
        if !dir.exists() {
            if fs::create_dir_all(&dir).is_err() {
                return;
            }
        }
        
        let Some(path) = Self::recent_files_path() else {
            return;
        };
        
        let content: String = self.recent_files
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
    
    /// Create a new home tab and return its ID
    pub fn create_home_tab(&mut self) -> i32 {
        let id = self.next_id;
        self.next_id += 1;
        self.home_tabs.push(id);
        self.active_file_id = Some(id);
        id
    }
    
    /// Close a home tab
    pub fn close_home_tab(&mut self, id: i32) {
        self.home_tabs.retain(|&tab_id| tab_id != id);
        
        // Update active file if this was active
        if self.active_file_id == Some(id) {
            // Find next tab to activate
            self.active_file_id = self.open_files.first().map(|f| f.id)
                .or_else(|| self.home_tabs.first().copied());
        }
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
            last_secondary_zoom: 0.0,
            last_secondary_center_x: 0.0,
            last_secondary_center_y: 0.0,
            last_secondary_width: 0.0,
            last_secondary_height: 0.0,
        });

        self.active_file_id = Some(id);
        id
    }
    
    /// Set the current tool and reset interaction state
    pub fn set_tool(&mut self, tool: Tool) {
        self.current_tool = tool;
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
        // Clear ROI when switching tools
        if let Some(file) = self.open_files.iter_mut().find(|f| Some(f.id) == self.active_file_id) {
            file.roi = None;
        }
    }
    
    /// Cancel current tool operation and return to Navigate
    pub fn cancel_tool(&mut self) {
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
        self.current_tool = Tool::Navigate;
        // Clear ROI when cancelling
        if let Some(file) = self.open_files.iter_mut().find(|f| Some(f.id) == self.active_file_id) {
            file.roi = None;
        }
    }

    /// Enable split view for the current file
    pub fn enable_split(&mut self) {
        if let Some(file) = self.open_files.iter_mut().find(|f| Some(f.id) == self.active_file_id) {
            if file.secondary_viewport.is_none() {
                // Clone the primary viewport for the secondary view
                let secondary = file.viewport.clone();
                file.secondary_viewport = Some(secondary);
            }
            self.split_enabled = true;
            self.focused_pane = PaneId::Secondary;
        }
    }

    /// Disable split view
    pub fn disable_split(&mut self) {
        self.split_enabled = false;
        self.focused_pane = PaneId::Primary;
    }

    /// Get the viewport for the current focused pane
    pub fn focused_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let pane = self.focused_pane;
        self.open_files.iter_mut()
            .find(|f| Some(f.id) == self.active_file_id)
            .and_then(|f| match pane {
                PaneId::Primary => Some(&mut f.viewport),
                PaneId::Secondary => f.secondary_viewport.as_mut(),
            })
    }

    /// Activate a file by ID
    pub fn activate_file(&mut self, id: i32) {
        if self.open_files.iter().any(|f| f.id == id) {
            self.active_file_id = Some(id);
        }
    }

    /// Close a file by ID
    pub fn close_file(&mut self, id: i32) {
        let index = self.open_files.iter().position(|f| f.id == id);
        
        if let Some(idx) = index {
            self.open_files.remove(idx);
            
            // Update active file
            if self.active_file_id == Some(id) {
                // Try to select another file tab first
                self.active_file_id = self.open_files.get(idx.saturating_sub(1))
                    .or_else(|| self.open_files.first())
                    .map(|f| f.id)
                    // Fall back to a home tab if no files remain
                    .or_else(|| self.home_tabs.first().copied());
            }
        }
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
    pub fn active_file(&self) -> Option<&OpenFile> {
        self.active_file_id.and_then(|id| self.get_file(id))
    }

    /// Get the active file mutably
    pub fn active_file_mut(&mut self) -> Option<&mut OpenFile> {
        self.active_file_id.and_then(|id| self.get_file_mut(id))
    }

    /// Get the active viewport
    pub fn active_viewport(&self) -> Option<&ViewportState> {
        self.active_file().map(|f| &f.viewport)
    }

    /// Get the active viewport mutably (respects focused pane in split view)
    pub fn active_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let pane = self.focused_pane;
        let active_id = self.active_file_id;
        let split_enabled = self.split_enabled;
        self.open_files.iter_mut()
            .find(|f| Some(f.id) == active_id)
            .and_then(|f| {
                if split_enabled {
                    match pane {
                        PaneId::Primary => Some(&mut f.viewport),
                        PaneId::Secondary => f.secondary_viewport.as_mut(),
                    }
                } else {
                    Some(&mut f.viewport)
                }
            })
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(false)
    }
}
