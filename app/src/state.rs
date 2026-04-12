//! Application state management

use crate::tile_loader::TileLoader;
use common::{TileManager, ViewportState, WsiFile};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct PaneState {
    pub tabs: Vec<i32>,
    pub active_tab_id: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderBackend {
    #[default]
    Cpu,
    Gpu,
}

/// Filtering mode for tile rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilteringMode {
    /// Bilinear interpolation within a single mip level
    Bilinear,
    /// Trilinear: bilinear + mip-level blending
    #[default]
    Trilinear,
    /// Lanczos-3 (a=3) resampling kernel
    Lanczos3,
}

/// Measurement unit for scale bar and measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeasurementUnit {
    /// Micrometers (µm)
    #[default]
    Um,
    /// Millimeters (mm)
    Mm,
    /// Inches
    Inches,
}

/// Stain normalization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StainNormalization {
    #[default]
    None,
    Macenko,
    Vahadane,
}

/// Per-tab HUD settings
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudSettings {
    pub show_scale_bar: bool,
    pub show_hud_toolbar: bool,
    pub hud_dropdown_open: bool,
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub measurement_unit: MeasurementUnit,
    pub stain_normalization: StainNormalization,
}

impl Default for HudSettings {
    fn default() -> Self {
        Self {
            show_scale_bar: true,
            show_hud_toolbar: true,
            hud_dropdown_open: false,
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
            measurement_unit: MeasurementUnit::Um,
            stain_normalization: StainNormalization::None,
        }
    }
}

impl HudSettings {
    pub fn reset_adjustments(&mut self) {
        self.gamma = 1.0;
        self.brightness = 0.0;
        self.contrast = 1.0;
        self.stain_normalization = StainNormalization::None;
    }

    #[allow(dead_code)]
    pub fn has_adjustments(&self) -> bool {
        (self.gamma - 1.0).abs() > 0.001
            || self.brightness.abs() > 0.001
            || (self.contrast - 1.0).abs() > 0.001
            || self.stain_normalization != StainNormalization::None
    }
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

fn normalize_recent_path(path: &PathBuf) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.clone())
}

fn dedupe_recent_files(paths: impl IntoIterator<Item = PathBuf>) -> Vec<RecentFile> {
    let mut seen_paths = HashSet::new();
    let mut recent_files = Vec::new();

    for path in paths {
        let normalized_path = normalize_recent_path(&path);
        if seen_paths.insert(normalized_path.clone()) {
            recent_files.push(RecentFile::new(normalized_path));
        }

        if recent_files.len() >= MAX_RECENT_FILES {
            break;
        }
    }

    recent_files
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
    /// Per-pane viewport/render state indexed by pane position
    pub pane_states: Vec<Option<FilePaneState>>,
    /// Thumbnail for minimap (RGBA data)
    pub thumbnail: Option<Vec<u8>>,
    /// Whether thumbnail has been uploaded to UI
    #[allow(dead_code)]
    pub thumbnail_set: bool,
    /// Region of interest (if set)
    pub roi: Option<RegionOfInterest>,
    /// Measurements
    pub measurements: Vec<Measurement>,
}

impl OpenFile {
    pub fn invalidate_render_state(&mut self) {
        for pane_state in &mut self.pane_states {
            if let Some(pane_state) = pane_state.as_mut() {
                pane_state.invalidate();
            }
        }
    }

    pub fn ensure_pane_capacity(&mut self, pane_count: usize) {
        if self.pane_states.len() < pane_count {
            self.pane_states.resize_with(pane_count, || None);
        }
    }

    pub fn insert_pane_slot(&mut self, index: usize) {
        if index <= self.pane_states.len() {
            self.pane_states.insert(index, None);
        } else {
            self.ensure_pane_capacity(index + 1);
        }
    }

    pub fn remove_pane_slot(&mut self, index: usize) {
        if index < self.pane_states.len() {
            self.pane_states.remove(index);
        }
        if self.pane_states.is_empty() {
            self.pane_states.push(None);
        }
    }

    pub fn pane_state(&self, pane: PaneId) -> Option<&FilePaneState> {
        self.pane_states
            .get(pane.0)
            .and_then(|state| state.as_ref())
    }

    pub fn pane_state_mut(&mut self, pane: PaneId) -> Option<&mut FilePaneState> {
        self.pane_states
            .get_mut(pane.0)
            .and_then(|state| state.as_mut())
    }

    pub fn ensure_pane_state_from(&mut self, pane: PaneId, source_pane: PaneId) {
        self.ensure_pane_capacity(pane.0 + 1);
        if self.pane_state(pane).is_some() {
            return;
        }

        let viewport = self
            .pane_state(source_pane)
            .map(|source| source.viewport.clone())
            .or_else(|| {
                self.pane_states
                    .iter()
                    .flatten()
                    .next()
                    .map(|state| state.viewport.clone())
            })
            .unwrap_or_else(|| self.viewport.clone());
        self.pane_states[pane.0] = Some(FilePaneState::new(viewport));
    }
}

#[derive(Clone)]
pub struct FilePaneState {
    pub viewport: ViewportState,
    #[allow(dead_code)]
    pub render_buffer: Vec<u8>,
    pub last_render_zoom: f64,
    pub last_render_center_x: f64,
    pub last_render_center_y: f64,
    pub last_render_width: f64,
    pub last_render_height: f64,
    pub last_render_level: u32,
    pub tiles_loaded_since_render: u32,
    pub frame_count: u32,
    pub last_render_time: std::time::Instant,
    pub last_seen_tile_epoch: u64,
    pub last_request: Option<TileRequestSignature>,
    pub last_render_gamma: f32,
    pub last_render_brightness: f32,
    pub last_render_contrast: f32,
    pub last_render_stain_normalization: StainNormalization,
    pub hud: HudSettings,
}

impl FilePaneState {
    pub fn new(viewport: ViewportState) -> Self {
        Self {
            viewport,
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
            last_request: None,
            last_render_gamma: 1.0,
            last_render_brightness: 0.0,
            last_render_contrast: 1.0,
            last_render_stain_normalization: StainNormalization::None,
            hud: HudSettings::default(),
        }
    }

    pub fn invalidate(&mut self) {
        self.last_render_zoom = 0.0;
        self.last_render_center_x = 0.0;
        self.last_render_center_y = 0.0;
        self.last_render_width = 0.0;
        self.last_render_height = 0.0;
        self.last_render_level = u32::MAX;
        self.tiles_loaded_since_render = 0;
        self.frame_count = 0;
        self.last_seen_tile_epoch = 0;
        self.last_request = None;
    }
}

/// Pane identifier (0-based pane index)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaneId(pub usize);

impl PaneId {
    pub const PRIMARY: Self = Self(0);
    pub const SECONDARY: Self = Self(1);

    pub fn as_index(self) -> i32 {
        self.0 as i32
    }
}

/// Application state
pub struct AppState {
    /// Currently open files
    pub open_files: Vec<OpenFile>,
    /// Active tab ID for the currently focused pane
    pub active_file_id: Option<i32>,
    /// Ordered pane state from left to right
    pub panes: Vec<PaneState>,
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
    /// Duplicate tab instances mapped to their backing file ID
    tab_aliases: HashMap<i32, i32>,
    /// Per-tab viewport/render snapshots for duplicated file tabs
    tab_instance_states: HashMap<i32, FilePaneState>,
    /// Last file ID rendered into each pane
    pub last_rendered_file_ids: Vec<Option<i32>>,
    /// The selected rendering backend
    pub render_backend: RenderBackend,
    /// Whether GPU rendering is available on this system
    pub gpu_backend_available: bool,
    /// The selected texture filtering mode
    pub filtering_mode: FilteringMode,
    /// Whether the minimap/zoom controls are shown in viewports
    pub show_minimap: bool,
    /// Whether the metadata HUD is shown in viewports
    pub show_metadata: bool,
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
            panes: vec![PaneState::default()],
            next_id: 1,
            split_enabled: false,
            split_position: 0.5,
            focused_pane: PaneId::PRIMARY,
            current_tool: Tool::Navigate,
            tool_state: ToolInteractionState::Idle,
            candidate_point: None,
            ant_offset: 0.0,
            debug_mode,
            frame_times: Vec::with_capacity(60),
            current_fps: 0.0,
            recent_files,
            home_tabs: Vec::new(),
            tab_aliases: HashMap::new(),
            tab_instance_states: HashMap::new(),
            last_rendered_file_ids: vec![None],
            render_backend: RenderBackend::Cpu,
            gpu_backend_available: false,
            filtering_mode: FilteringMode::default(),
            show_minimap: true,
            show_metadata: false,
            needs_render: true,
            render_loop_running: false,
        }
    }

    /// Get the config directory for storing recent files
    fn config_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("eov"))
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

        dedupe_recent_files(content.lines().filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }

            let path = PathBuf::from(trimmed);
            path.exists().then_some(path)
        }))
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
        let normalized_path = normalize_recent_path(path);

        // Remove if already exists (we'll re-add at top)
        self.recent_files.retain(|f| f.path != normalized_path);

        // Add to front
        self.recent_files
            .insert(0, RecentFile::new(normalized_path));

        // Trim to max size
        self.recent_files.truncate(MAX_RECENT_FILES);

        // Save to disk
        self.save_recent_files();
    }

    fn tab_ids_for_pane(&self, pane: PaneId) -> &[i32] {
        self.panes
            .get(pane.0)
            .map(|pane_state| pane_state.tabs.as_slice())
            .unwrap_or(&[])
    }

    fn tab_ids_for_pane_mut(&mut self, pane: PaneId) -> &mut Vec<i32> {
        &mut self.panes[pane.0].tabs
    }

    fn active_tab_id_for_pane_mut(&mut self, pane: PaneId) -> &mut Option<i32> {
        &mut self.panes[pane.0].active_tab_id
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

    fn resolve_tab_file_id(&self, id: i32) -> i32 {
        self.tab_aliases.get(&id).copied().unwrap_or(id)
    }

    fn create_duplicate_tab_id(&mut self, id: i32) -> i32 {
        if self.is_home_tab(id) {
            return id;
        }

        let file_id = self.resolve_tab_file_id(id);
        let tab_id = self.next_id;
        self.next_id += 1;
        self.tab_aliases.insert(tab_id, file_id);
        tab_id
    }

    fn has_tab_reference_to_file(&self, file_id: i32, excluding_tab_id: Option<i32>) -> bool {
        self.panes.iter().any(|pane_state| {
            pane_state.tabs.iter().copied().any(|tab_id| {
                Some(tab_id) != excluding_tab_id && self.resolve_tab_file_id(tab_id) == file_id
            })
        })
    }

    fn save_tab_instance_state(&mut self, tab_id: i32, pane: PaneId) {
        if self.is_home_tab(tab_id) {
            return;
        }

        let Some(snapshot) = self
            .get_file(self.resolve_tab_file_id(tab_id))
            .and_then(|file| file.pane_state(pane).cloned())
        else {
            return;
        };

        self.tab_instance_states.insert(tab_id, snapshot);
    }

    fn restore_tab_instance_state(&mut self, tab_id: i32, pane: PaneId) {
        let Some(snapshot) = self.tab_instance_states.get(&tab_id).cloned() else {
            return;
        };

        let file_id = self.resolve_tab_file_id(tab_id);
        if let Some(file) = self.get_file_mut(file_id) {
            file.ensure_pane_capacity(pane.0 + 1);
            file.pane_states[pane.0] = Some(snapshot);
        }
    }

    fn select_tab_in_pane(&mut self, pane: PaneId, id: i32) {
        if let Some(current_tab_id) = self.active_tab_id_for_pane(pane)
            && current_tab_id != id
        {
            self.save_tab_instance_state(current_tab_id, pane);
        }

        self.restore_tab_instance_state(id, pane);
        *self.active_tab_id_for_pane_mut(pane) = Some(id);
    }

    fn sync_active_file_id(&mut self) {
        self.active_file_id = self
            .active_tab_id_for_pane(self.focused_pane)
            .filter(|id| !self.is_home_tab(*id))
            .map(|id| self.resolve_tab_file_id(id));
    }

    pub fn active_tab_id_for_pane(&self, pane: PaneId) -> Option<i32> {
        self.panes
            .get(pane.0)
            .and_then(|pane_state| pane_state.active_tab_id)
    }

    pub fn active_file_id_for_pane(&self, pane: PaneId) -> Option<i32> {
        self.active_tab_id_for_pane(pane)
            .filter(|id| !self.is_home_tab(*id))
            .map(|id| self.resolve_tab_file_id(id))
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
        if pane.0 >= self.panes.len() {
            return;
        }
        self.focused_pane = pane;
        self.sync_active_file_id();
    }

    fn ensure_file_pane_state(&mut self, id: i32, pane: PaneId, source_pane: PaneId) {
        let file_id = self.resolve_tab_file_id(id);
        if let Some(file) = self.get_file_mut(file_id) {
            file.ensure_pane_state_from(pane, source_pane);
        }
    }

    pub fn insert_pane(&mut self, index: usize) -> PaneId {
        let insert_index = index.min(self.panes.len());
        self.panes.insert(insert_index, PaneState::default());
        self.last_rendered_file_ids.insert(insert_index, None);
        for file in &mut self.open_files {
            file.insert_pane_slot(insert_index);
        }
        if self.focused_pane.0 >= insert_index {
            self.focused_pane = PaneId(self.focused_pane.0 + 1);
        }
        self.split_enabled = self.panes.len() > 1;
        self.sync_active_file_id();
        PaneId(insert_index)
    }

    fn remove_pane(&mut self, index: usize) {
        if self.panes.len() <= 1 || index >= self.panes.len() {
            return;
        }

        self.panes.remove(index);
        if index < self.last_rendered_file_ids.len() {
            self.last_rendered_file_ids.remove(index);
        }
        for file in &mut self.open_files {
            file.remove_pane_slot(index);
        }

        if self.focused_pane.0 > index {
            self.focused_pane = PaneId(self.focused_pane.0 - 1);
        } else if self.focused_pane.0 >= self.panes.len() {
            self.focused_pane = PaneId(self.panes.len().saturating_sub(1));
        }
    }

    pub fn reset_to_single_pane(&mut self) {
        self.panes = vec![PaneState::default()];
        self.last_rendered_file_ids = vec![None];
        self.focused_pane = PaneId::PRIMARY;
        for file in &mut self.open_files {
            if file.pane_states.is_empty() {
                file.pane_states.push(None);
            } else {
                file.pane_states.truncate(1);
            }
        }
        self.split_enabled = false;
        self.sync_active_file_id();
    }

    fn add_tab_to_pane_if_missing(&mut self, pane: PaneId, id: i32) {
        let tabs = self.tab_ids_for_pane_mut(pane);
        if !tabs.contains(&id) {
            tabs.push(id);
        }
        self.ensure_file_pane_state(id, pane, PaneId::PRIMARY);
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
            self.sync_active_file_id();
        }
    }

    fn normalize_split_after_tab_change(&mut self) {
        if self.focused_pane.0 >= self.panes.len() {
            self.focused_pane = PaneId(self.panes.len().saturating_sub(1));
        }
        self.split_enabled = self.panes.len() > 1;
    }

    fn collapse_empty_panes(&mut self) {
        let mut pane_index = 0;
        while self.panes.len() > 1 && pane_index < self.panes.len() {
            if self.panes[pane_index].tabs.is_empty() {
                self.remove_pane(pane_index);
            } else {
                pane_index += 1;
            }
        }
        self.normalize_split_after_tab_change();
        self.sync_active_file_id();
    }

    fn is_known_tab(&self, id: i32) -> bool {
        self.is_home_tab(id)
            || self.tab_aliases.contains_key(&id)
            || self
                .open_files
                .iter()
                .any(|file| file.id == self.resolve_tab_file_id(id))
    }

    pub fn duplicate_tab_to_pane(&mut self, id: i32, pane: PaneId) -> i32 {
        if !self.is_known_tab(id) {
            return id;
        }
        let source_pane = self
            .panes
            .iter()
            .enumerate()
            .find(|(_, pane_state)| pane_state.tabs.contains(&id))
            .map(|(index, _)| PaneId(index))
            .unwrap_or(PaneId::PRIMARY);
        let duplicated_id = self.create_duplicate_tab_id(id);
        self.save_tab_instance_state(id, source_pane);
        if let Some(snapshot) = self.tab_instance_states.get(&id).cloned() {
            self.tab_instance_states.insert(duplicated_id, snapshot);
        }
        self.add_tab_to_pane_if_missing(pane, duplicated_id);
        self.ensure_file_pane_state(duplicated_id, pane, source_pane);
        self.select_tab_in_pane(pane, duplicated_id);
        self.needs_render = true;
        self.sync_active_file_id();
        duplicated_id
    }

    pub fn move_tab_between_panes(&mut self, id: i32, from: PaneId, to: PaneId) {
        if from == to || !self.is_known_tab(id) {
            return;
        }

        let file_id = self.resolve_tab_file_id(id);
        if let Some(existing_target_tab_id) = self
            .tabs_for_pane(to)
            .iter()
            .copied()
            .find(|tab_id| *tab_id != id && self.resolve_tab_file_id(*tab_id) == file_id)
        {
            self.save_tab_instance_state(existing_target_tab_id, to);
        }
        self.save_tab_instance_state(id, from);

        self.remove_tab_from_pane(from, id);
        self.add_tab_to_pane_if_missing(to, id);
        self.ensure_file_pane_state(id, to, from);
        self.select_tab_in_pane(to, id);
        if self.focused_pane == from {
            self.set_focused_pane(to);
        } else {
            self.sync_active_file_id();
        }
        self.collapse_empty_panes();
        self.normalize_split_after_tab_change();
        self.needs_render = true;
    }

    pub fn split_tab_to_new_pane(&mut self, id: i32, source_pane: PaneId, insert_at: usize) {
        if !self.is_known_tab(id) || source_pane.0 >= self.panes.len() {
            return;
        }

        let target_pane = self.insert_pane(insert_at);
        self.move_tab_between_panes(id, source_pane, target_pane);
        self.set_focused_pane(target_pane);
        self.needs_render = true;
    }

    /// Reorder a tab within the same pane by moving it to a new index position.
    pub fn reorder_tab(&mut self, pane: PaneId, id: i32, new_index: i32) {
        let tabs = self.tab_ids_for_pane_mut(pane);
        if let Some(current_pos) = tabs.iter().position(|&tab_id| tab_id == id) {
            let slot = (new_index.max(0) as usize).min(tabs.len());
            if slot == current_pos || slot == current_pos + 1 {
                return;
            }

            let tab_id = tabs.remove(current_pos);
            let insert_pos = if slot > current_pos { slot - 1 } else { slot };
            tabs.insert(insert_pos.min(tabs.len()), tab_id);
        }
    }

    pub fn activate_tab_in_pane(&mut self, pane: PaneId, id: i32) {
        if !self.is_known_tab(id) {
            return;
        }
        self.add_tab_to_pane_if_missing(pane, id);
        self.select_tab_in_pane(pane, id);
        self.set_focused_pane(pane);
        self.needs_render = true;
    }

    /// Create a new home tab and return its ID.
    /// If the pane already has a home tab, switch to it instead.
    pub fn create_home_tab(&mut self) -> i32 {
        let pane = if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::PRIMARY
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
        for pane_index in 0..self.panes.len() {
            self.remove_tab_from_pane(PaneId(pane_index), id);
        }
        self.collapse_empty_panes();
        self.needs_render = true;
    }

    pub fn close_tab_in_pane(&mut self, pane: PaneId, id: i32) {
        if !self.is_known_tab(id) {
            return;
        }

        let file_id = self.resolve_tab_file_id(id);

        self.remove_tab_from_pane(pane, id);
        self.collapse_empty_panes();

        self.tab_aliases.remove(&id);
        self.tab_instance_states.remove(&id);

        let still_open_elsewhere = self.has_tab_reference_to_file(file_id, None);
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

        self.close_file(file_id);
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
            viewport: viewport.clone(),
            pane_states: vec![Some(FilePaneState::new(viewport))],
            thumbnail,
            thumbnail_set: false,
            roi: None,
            measurements: Vec::new(),
        });

        let target_pane = if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::PRIMARY
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

    /// Disable split view
    #[allow(dead_code)]
    pub fn disable_split(&mut self) {
        if self.panes.len() <= 1 {
            return;
        }

        let mut merged_tabs = Vec::new();
        let mut merged_active = None;
        for pane_index in 0..self.panes.len() {
            let pane_state = &self.panes[pane_index];
            for &tab_id in &pane_state.tabs {
                if !merged_tabs.contains(&tab_id) {
                    merged_tabs.push(tab_id);
                }
            }
            if pane_index == self.focused_pane.0 {
                merged_active = pane_state.active_tab_id.or(merged_active);
            }
        }

        self.panes = vec![PaneState {
            tabs: merged_tabs,
            active_tab_id: merged_active,
        }];
        self.last_rendered_file_ids = vec![None];
        self.focused_pane = PaneId::PRIMARY;
        for file in &mut self.open_files {
            if let Some(first_state) = file.pane_states.iter().flatten().next().cloned() {
                file.pane_states = vec![Some(first_state)];
            } else {
                file.pane_states = vec![None];
            }
        }
        self.split_enabled = false;
        self.needs_render = true;
    }

    /// Get the viewport for the current focused pane
    #[allow(dead_code)]
    pub fn focused_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let active_id = self.active_file_id_for_pane(self.focused_pane)?;
        self.open_files
            .iter_mut()
            .find(|f| f.id == active_id)
            .and_then(|f| {
                f.pane_state_mut(self.focused_pane)
                    .map(|pane_state| &mut pane_state.viewport)
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

            for pane_index in 0..self.panes.len() {
                let pane = PaneId(pane_index);
                let tab_ids_to_remove: Vec<i32> = self
                    .tabs_for_pane(pane)
                    .iter()
                    .copied()
                    .filter(|tab_id| self.resolve_tab_file_id(*tab_id) == id)
                    .collect();
                for tab_id in tab_ids_to_remove {
                    self.remove_tab_from_pane(pane, tab_id);
                }
                if self.active_tab_id_for_pane(PaneId(pane_index)).is_none() {
                    let replacement = self.tabs_for_pane(PaneId(pane_index)).first().copied();
                    *self.active_tab_id_for_pane_mut(PaneId(pane_index)) = replacement;
                }
            }

            self.tab_aliases.retain(|_, file_id| *file_id != id);
            let tab_instance_ids_to_remove: Vec<i32> = self
                .tab_instance_states
                .keys()
                .copied()
                .filter(|tab_id| self.resolve_tab_file_id(*tab_id) == id)
                .collect();
            for tab_id in tab_instance_ids_to_remove {
                self.tab_instance_states.remove(&tab_id);
            }

            self.collapse_empty_panes();

            for rendered_id in &mut self.last_rendered_file_ids {
                *rendered_id = rendered_id.filter(|&file_id| file_id != id);
            }

            self.needs_render = true;
            self.sync_active_file_id();
        }
    }

    pub fn close_all_tabs(&mut self) {
        self.open_files.clear();
        self.home_tabs.clear();
        self.tab_aliases.clear();
        self.tab_instance_states.clear();
        self.reset_to_single_pane();
        self.active_file_id = None;
        self.request_render();
    }

    pub fn last_rendered_file_id(&self, pane: PaneId) -> Option<i32> {
        self.last_rendered_file_ids.get(pane.0).copied().flatten()
    }

    pub fn set_last_rendered_file_id(&mut self, pane: PaneId, id: Option<i32>) {
        if pane.0 >= self.last_rendered_file_ids.len() {
            self.last_rendered_file_ids.resize(pane.0 + 1, None);
        }
        self.last_rendered_file_ids[pane.0] = id;
    }

    pub fn request_render(&mut self) {
        self.needs_render = true;
    }

    pub fn toggle_minimap(&mut self) {
        self.show_minimap = !self.show_minimap;
        self.needs_render = true;
    }

    pub fn toggle_metadata(&mut self) {
        self.show_metadata = !self.show_metadata;
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

    pub fn select_filtering_mode(&mut self, mode: FilteringMode) {
        if self.filtering_mode != mode {
            self.filtering_mode = mode;
            for file in &mut self.open_files {
                file.invalidate_render_state();
            }
            self.needs_render = true;
        }
    }

    /// Get a file by ID
    pub fn get_file(&self, id: i32) -> Option<&OpenFile> {
        let file_id = self.resolve_tab_file_id(id);
        self.open_files.iter().find(|f| f.id == file_id)
    }

    /// Get a mutable file by ID
    pub fn get_file_mut(&mut self, id: i32) -> Option<&mut OpenFile> {
        let file_id = self.resolve_tab_file_id(id);
        self.open_files.iter_mut().find(|f| f.id == file_id)
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
            PaneId::PRIMARY
        };
        let active_id = self.active_file_id_for_pane(pane)?;
        self.get_file(active_id)
            .and_then(|file| file.pane_state(pane).map(|pane_state| &pane_state.viewport))
    }

    /// Get the active viewport mutably (respects focused pane in split view)
    pub fn active_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        let pane = self.focused_pane;
        let effective_pane = if self.split_enabled {
            pane
        } else {
            PaneId::PRIMARY
        };
        let active_id = self.active_file_id_for_pane(effective_pane)?;
        self.open_files
            .iter_mut()
            .find(|f| f.id == active_id)
            .and_then(|f| {
                f.pane_state_mut(effective_pane)
                    .map(|pane_state| &mut pane_state.viewport)
            })
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::dedupe_recent_files;
    use std::path::PathBuf;

    #[test]
    fn dedupe_recent_files_preserves_first_occurrence_order() {
        let recent_files = dedupe_recent_files([
            PathBuf::from("/tmp/slide-c.svs"),
            PathBuf::from("/tmp/slide-b.svs"),
            PathBuf::from("/tmp/slide-c.svs"),
            PathBuf::from("/tmp/slide-a.svs"),
            PathBuf::from("/tmp/slide-b.svs"),
        ]);

        let paths: Vec<_> = recent_files.into_iter().map(|file| file.path).collect();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("/tmp/slide-c.svs"),
                PathBuf::from("/tmp/slide-b.svs"),
                PathBuf::from("/tmp/slide-a.svs"),
            ]
        );
    }
}
