//! Application state management

use common::{TileManager, ViewportState, WsiFile};
use std::path::PathBuf;

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
    pub tile_manager: TileManager,
    /// Primary viewport state
    pub viewport: ViewportState,
    /// Secondary viewport state (for split view)
    pub secondary_viewport: Option<ViewportState>,
    /// Thumbnail for minimap (RGBA data)
    pub thumbnail: Option<Vec<u8>>,
    /// Region of interest (if set)
    pub roi: Option<RegionOfInterest>,
    /// Measurements
    pub measurements: Vec<Measurement>,
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
}

impl AppState {
    /// Create a new application state
    pub fn new() -> Self {
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
        }
    }

    /// Add a new open file
    pub fn add_file(
        &mut self,
        path: PathBuf,
        wsi: WsiFile,
        tile_manager: TileManager,
        viewport: ViewportState,
        thumbnail: Option<Vec<u8>>,
    ) -> i32 {
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
            viewport,
            secondary_viewport: None,
            thumbnail,
            roi: None,
            measurements: Vec::new(),
        });

        self.active_file_id = Some(id);
        id
    }
    
    /// Set the current tool and reset interaction state
    pub fn set_tool(&mut self, tool: Tool) {
        self.current_tool = tool;
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
    }
    
    /// Cancel current tool operation and return to Navigate
    pub fn cancel_tool(&mut self) {
        self.tool_state = ToolInteractionState::Idle;
        self.candidate_point = None;
        self.current_tool = Tool::Navigate;
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
                self.active_file_id = self.open_files.get(idx.saturating_sub(1))
                    .or_else(|| self.open_files.first())
                    .map(|f| f.id);
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
        Self::new()
    }
}
