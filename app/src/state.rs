//! Application state management

use crate::interaction::HostMeasurement;
use crate::plugin_runtime::filters::SharedFilterChain;
use crate::tile_loader::TileLoader;
use eov_common::{
    FilteringMode, MeasurementUnit, RenderBackend, SlideBounds, StainNormalization, TileManager,
    ViewportState, WsiFile,
};
use eov_plugin_api::protocol::Rect;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

fn normalize_series_entry_path(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

static NEXT_PANE_UID: AtomicU32 = AtomicU32::new(1);

/// Pane position → stable id, readable without the `AppState` lock (by
/// render workers that tell plugin filters which pane they filter).
static PANE_UIDS: parking_lot::RwLock<Vec<u32>> = parking_lot::RwLock::new(Vec::new());

/// The stable id of the pane at `index`, as last published.
pub fn pane_uid_for_index(index: usize) -> Option<u32> {
    PANE_UIDS.read().get(index).copied()
}

#[derive(Debug, Clone)]
pub struct PaneState {
    /// Stable identity, unaffected by panes inserted or removed around it
    /// (what plugins see as the pane id).
    pub uid: u32,
    pub tabs: Vec<i32>,
    pub active_tab_id: Option<i32>,
}

impl PaneState {
    pub fn new() -> Self {
        Self {
            uid: NEXT_PANE_UID.fetch_add(1, Ordering::Relaxed),
            tabs: Vec::new(),
            active_tab_id: None,
        }
    }
}

impl Default for PaneState {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-tab HUD settings
/// Which stain channel is being viewed in grayscale isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolatedChannel {
    #[default]
    None,
    Hematoxylin,
    Eosin,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudSettings {
    pub show_hud_toolbar: bool,
    pub hud_dropdown_open: bool,
    pub sharpness: f32,
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub measurement_unit: MeasurementUnit,
    pub stain_normalization: StainNormalization,
    // Color deconvolution state
    pub deconv_hematoxylin_intensity: f32,
    pub deconv_hematoxylin_visible: bool,
    pub deconv_eosin_intensity: f32,
    pub deconv_eosin_visible: bool,
    pub deconv_isolated_channel: IsolatedChannel,
}

impl Default for HudSettings {
    fn default() -> Self {
        Self {
            show_hud_toolbar: true,
            hud_dropdown_open: false,
            sharpness: 0.0,
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
            measurement_unit: MeasurementUnit::Um,
            stain_normalization: StainNormalization::None,
            deconv_hematoxylin_intensity: 1.0,
            deconv_hematoxylin_visible: true,
            deconv_eosin_intensity: 1.0,
            deconv_eosin_visible: true,
            deconv_isolated_channel: IsolatedChannel::None,
        }
    }
}

impl HudSettings {
    pub fn reset_adjustments(&mut self) {
        self.sharpness = 0.0;
        self.gamma = 1.0;
        self.brightness = 0.0;
        self.contrast = 1.0;
        self.stain_normalization = StainNormalization::None;
        self.deconv_hematoxylin_intensity = 1.0;
        self.deconv_hematoxylin_visible = true;
        self.deconv_eosin_intensity = 1.0;
        self.deconv_eosin_visible = true;
        self.deconv_isolated_channel = IsolatedChannel::None;
    }
}

/// Maximum number of recently opened files to track
const MAX_RECENT_FILES: usize = 10;
pub const BOTTOM_PANEL_HEIGHT_PX: f32 = 232.0;

/// Recently opened file entry
#[derive(Debug, Clone)]
pub struct RecentFile {
    /// Full path to the file
    pub path: PathBuf,
    /// Display name (filename)
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BottomPanelKind {
    #[default]
    None,
    Series,
}

#[derive(Debug, Clone, Default)]
pub struct SeriesThumbnail {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// The minimap image of a slide: its real dimensions and the level-0
/// rectangle it shows (the content bounds of sparse slides), so the
/// minimap never has to guess either (RND-10).
#[derive(Debug, Clone)]
pub struct MinimapThumbnail {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub region: SlideBounds,
}

#[derive(Debug, Clone, Default)]
pub struct SeriesEntry {
    pub path: PathBuf,
    pub filename: String,
    pub is_directory: bool,
    pub metadata_tooltip: String,
    pub objective_label: Option<String>,
    pub stain_label: Option<String>,
    pub thumbnail: Option<SeriesThumbnail>,
    pub thumbnail_loading: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesNavigationMode {
    Initialize,
    Replace,
    Push,
    Back,
    Forward,
}

#[derive(Debug, Clone, Default)]
pub struct OpenedSeries {
    pub current_path: PathBuf,
    pub entries: Vec<SeriesEntry>,
    pub back_history: Vec<PathBuf>,
    pub forward_history: Vec<PathBuf>,
}

impl RecentFile {
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        Self { path, name }
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
    pub tile_loader: Arc<TileLoader>,
    /// Primary viewport state
    pub viewport: ViewportState,
    /// Per-pane viewport/render state indexed by pane position
    pub pane_states: Vec<Option<FilePaneState>>,
    /// Minimap thumbnail.
    pub thumbnail: Option<MinimapThumbnail>,
    /// Region of interest in level-0 pixels (if set); shown in every pane
    /// with this slide.
    pub roi: Option<Rect>,
    /// Host measurements.
    pub measurements: Vec<HostMeasurement>,
}

pub struct NewOpenFile {
    pub id: i32,
    pub path: PathBuf,
    pub wsi: WsiFile,
    pub tile_manager: Arc<TileManager>,
    pub tile_loader: Arc<TileLoader>,
    pub viewport: ViewportState,
    pub thumbnail: Option<MinimapThumbnail>,
    /// Pane that shows the new tab (the focused pane when it no longer
    /// exists).
    pub target_pane: PaneId,
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
    pub last_render_sharpness: f32,
    pub last_render_stain_normalization: StainNormalization,
    pub last_render_deconv_h_intensity: f32,
    pub last_render_deconv_h_visible: bool,
    pub last_render_deconv_e_intensity: f32,
    pub last_render_deconv_e_visible: bool,
    pub last_render_deconv_isolated: IsolatedChannel,
    pub hud: HudSettings,
    /// Cached stain normalization parameters.
    pub cached_stain_params: Option<crate::stain::StainNormParams>,
    /// Tile epoch at which stain params were last computed.
    pub stain_params_epoch: u64,
    /// Stain normalization method for which cached params were computed.
    pub stain_params_method: StainNormalization,
    /// ID of the currently pending async CPU render job for this pane.
    pub pending_cpu_job_id: Option<u64>,
    /// Whether the pane is showing an interaction preview and still needs a
    /// settled, full-quality CPU render.
    pub needs_settled_cpu_render: bool,
    /// Filter revision last incorporated into a settled pane render.
    pub last_render_filter_revision: u64,
    /// Whether the GPU surface still matches this pane's current viewport.
    pub gpu_surface_valid: bool,
}

impl FilePaneState {
    pub fn new(viewport: ViewportState) -> Self {
        Self {
            viewport,
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
            last_render_sharpness: 0.0,
            last_render_stain_normalization: StainNormalization::None,
            last_render_deconv_h_intensity: 1.0,
            last_render_deconv_h_visible: true,
            last_render_deconv_e_intensity: 1.0,
            last_render_deconv_e_visible: true,
            last_render_deconv_isolated: IsolatedChannel::None,
            hud: HudSettings::default(),
            cached_stain_params: None,
            stain_params_epoch: 0,
            stain_params_method: StainNormalization::None,
            pending_cpu_job_id: None,
            needs_settled_cpu_render: false,
            last_render_filter_revision: 0,
            gpu_surface_valid: false,
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
        self.pending_cpu_job_id = None;
        self.needs_settled_cpu_render = false;
        self.last_render_filter_revision = 0;
        self.gpu_surface_valid = false;
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
    /// Whether scale bars are shown in all viewports.
    pub show_scale_bar: bool,
    /// Whether a bottom panel is currently visible.
    pub bottom_panel_visible: bool,
    /// Which bottom-panel content is active.
    pub bottom_panel_kind: BottomPanelKind,
    /// Currently opened folder-backed series content.
    pub opened_series: Option<OpenedSeries>,
    /// Currently selected entry in the series bar, if any.
    pub series_selected_path: Option<PathBuf>,
    /// Monotonic revision used to reject stale background thumbnail work.
    pub series_revision: u64,
    /// Monotonic revision for visible series entry content changes.
    pub series_content_revision: u64,
    /// Whether viewport navigation is mirrored across panes showing the same file.
    pub viewport_lock_enabled: bool,
    /// Whether a new frame should be rendered as soon as possible
    pub needs_render: bool,
    /// Whether the render loop timer is currently running
    pub render_loop_running: bool,
    /// Plugin viewport filters, read lock-free by the renderer.
    pub filter_chain: SharedFilterChain,
    /// Monotonic revision for viewport filter enable-state changes.
    pub filter_revision: u64,
}

impl AppState {
    /// Create a new application state
    pub fn new() -> Self {
        let recent_files = Self::load_recent_files();
        Self {
            open_files: Vec::new(),
            active_file_id: None,
            panes: vec![PaneState::new()],
            next_id: 1,
            split_enabled: false,
            split_position: 0.5,
            focused_pane: PaneId::PRIMARY,
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
            show_scale_bar: true,
            bottom_panel_visible: false,
            bottom_panel_kind: BottomPanelKind::None,
            opened_series: None,
            series_selected_path: None,
            series_revision: 0,
            series_content_revision: 0,
            viewport_lock_enabled: false,
            needs_render: true,
            render_loop_running: false,
            filter_chain: Arc::new(crate::plugin_runtime::filters::FilterChain::default()),
            filter_revision: 0,
        }
    }

    /// Get the config directory for storing recent files
    fn config_dir() -> Option<PathBuf> {
        crate::paths::legacy_config_dir()
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

        self.recent_files.retain(|f| f.path != normalized_path);
        self.recent_files
            .insert(0, RecentFile::new(normalized_path));
        self.recent_files.truncate(MAX_RECENT_FILES);
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
        self.sync_series_selection_to_active_file();
    }

    fn sync_series_selection_to_active_file(&mut self) {
        let Some(active_path) = self
            .active_file_id
            .and_then(|file_id| self.get_file(file_id))
            .map(|file| normalize_series_entry_path(&file.path))
        else {
            return;
        };

        let Some(series) = self.opened_series.as_ref() else {
            return;
        };

        if let Some(entry) = series
            .entries
            .iter()
            .find(|entry| normalize_series_entry_path(&entry.path) == active_path)
        {
            self.series_selected_path = Some(entry.path.clone());
        }
    }

    fn reconcile_series_selection(&mut self) {
        let Some(series) = self.opened_series.as_ref() else {
            self.series_selected_path = None;
            return;
        };

        if series.entries.is_empty() {
            self.series_selected_path = None;
            return;
        }

        let current_selection = self
            .series_selected_path
            .as_ref()
            .map(|path| normalize_series_entry_path(path));
        let active_file_path = self
            .active_file_id
            .and_then(|file_id| self.get_file(file_id))
            .map(|file| normalize_series_entry_path(&file.path));

        let next_selection = current_selection
            .as_ref()
            .and_then(|selected| {
                series
                    .entries
                    .iter()
                    .find(|entry| normalize_series_entry_path(&entry.path) == *selected)
            })
            .or_else(|| {
                active_file_path.as_ref().and_then(|active| {
                    series
                        .entries
                        .iter()
                        .find(|entry| normalize_series_entry_path(&entry.path) == *active)
                })
            })
            .or_else(|| series.entries.first())
            .map(|entry| entry.path.clone());

        self.series_selected_path = next_selection;
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

        let previous_pane = self.active_viewport_pane();
        let previous_file_id = self.active_file_id_for_pane(previous_pane);
        self.focused_pane = pane;
        self.sync_active_file_id();

        if self.viewport_lock_enabled {
            let active_pane = self.active_viewport_pane();
            let active_file_id = self.active_file_id_for_pane(active_pane);
            if active_pane != previous_pane || active_file_id != previous_file_id {
                self.sync_locked_viewports_for_active_file(true);
            }
        }
    }

    pub fn cycle_focused_pane(&mut self, offset: i32) {
        let pane_count = self.panes.len();
        if pane_count <= 1 || offset == 0 {
            return;
        }

        let current = self.focused_pane.0 as i32;
        let next = (current + offset).rem_euclid(pane_count as i32) as usize;
        self.set_focused_pane(PaneId(next));
    }

    /// Publishes the pane ids for [`pane_uid_for_index`].
    pub fn publish_pane_uids(&self) {
        *PANE_UIDS.write() = self.panes.iter().map(|p| p.uid).collect();
    }

    /// The stable id of the pane at `pane`.
    pub fn pane_uid(&self, pane: PaneId) -> Option<u32> {
        self.panes.get(pane.0).map(|p| p.uid)
    }

    /// The current position of the pane with stable id `uid`.
    pub fn pane_by_uid(&self, uid: u32) -> Option<PaneId> {
        self.panes.iter().position(|p| p.uid == uid).map(PaneId)
    }

    /// The measurement unit of the first pane showing `file_id`.
    pub fn file_pane_unit(&self, file_id: i32) -> Option<MeasurementUnit> {
        let file = self.get_file(file_id)?;
        (0..self.panes.len())
            .map(PaneId)
            .filter(|pane| self.active_file_id_for_pane(*pane) == Some(file_id))
            .find_map(|pane| file.pane_state(pane).map(|ps| ps.hud.measurement_unit))
            .or_else(|| {
                file.pane_states
                    .iter()
                    .flatten()
                    .next()
                    .map(|ps| ps.hud.measurement_unit)
            })
    }

    fn ensure_file_pane_state(&mut self, id: i32, pane: PaneId, source_pane: PaneId) {
        let file_id = self.resolve_tab_file_id(id);
        if let Some(file) = self.get_file_mut(file_id) {
            file.ensure_pane_state_from(pane, source_pane);
        }
    }

    pub fn insert_pane(&mut self, index: usize) -> PaneId {
        let insert_index = index.min(self.panes.len());
        self.panes.insert(insert_index, PaneState::new());
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
        self.panes = vec![PaneState::new()];
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

    pub fn find_tab_by_path(&self, path: &std::path::Path) -> Option<(PaneId, i32)> {
        let normalized_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.panes
            .iter()
            .enumerate()
            .find_map(|(pane_index, pane_state)| {
                pane_state.tabs.iter().copied().find_map(|tab_id| {
                    if self.is_home_tab(tab_id) {
                        return None;
                    }

                    let file_id = self.resolve_tab_file_id(tab_id);
                    let file = self.get_file(file_id)?;
                    (file.path == normalized_path).then_some((PaneId(pane_index), tab_id))
                })
            })
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

    pub fn cycle_tab_in_focused_pane(&mut self, offset: i32) {
        if offset == 0 {
            return;
        }

        let pane = self.focused_pane;
        let tabs = self.tabs_for_pane(pane).to_vec();
        if tabs.is_empty() {
            return;
        }

        let current_index = self
            .active_tab_id_for_pane(pane)
            .and_then(|active_id| tabs.iter().position(|tab_id| *tab_id == active_id))
            .unwrap_or(0) as i32;
        let next_index = (current_index + offset).rem_euclid(tabs.len() as i32) as usize;
        self.activate_tab_in_pane(pane, tabs[next_index]);
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
            if self.viewport_lock_enabled {
                self.sync_locked_viewports_for_active_file(true);
            }
            return;
        }

        if self.is_home_tab(id) {
            self.home_tabs.retain(|&tab_id| tab_id != id);
            self.needs_render = true;
            self.sync_active_file_id();
            if self.viewport_lock_enabled {
                self.sync_locked_viewports_for_active_file(true);
            }
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

    /// Allocate the next file identifier without registering a file.
    pub fn allocate_file_id(&mut self) -> i32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add a new open file with a pre-allocated identifier.
    pub fn add_file(&mut self, new_file: NewOpenFile) -> i32 {
        let NewOpenFile {
            id,
            path,
            wsi,
            tile_manager,
            tile_loader,
            viewport,
            thumbnail,
            target_pane,
        } = new_file;

        let normalized_path = normalize_recent_path(&path);

        // Add to recent files
        self.add_to_recent(&normalized_path);

        let filename = normalized_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        self.open_files.push(OpenFile {
            id,
            path: normalized_path,
            filename,
            wsi,
            tile_manager,
            tile_loader,
            viewport: viewport.clone(),
            pane_states: vec![Some(FilePaneState::new(viewport))],
            thumbnail,
            roi: None,
            measurements: Vec::new(),
        });

        let target_pane = if target_pane.0 < self.panes.len() {
            target_pane
        } else if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::PRIMARY
        };
        self.activate_tab_in_pane(target_pane, id);
        id
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
            if self.viewport_lock_enabled {
                self.sync_locked_viewports_for_active_file(true);
            }
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

    pub fn bump_filter_revision(&mut self) {
        self.filter_revision = self.filter_revision.wrapping_add(1);
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

    pub fn toggle_scale_bar(&mut self) {
        self.show_scale_bar = !self.show_scale_bar;
        self.needs_render = true;
    }

    pub fn set_bottom_panel_visible(&mut self, visible: bool, kind: BottomPanelKind) {
        self.bottom_panel_visible = visible;
        self.bottom_panel_kind = if visible { kind } else { BottomPanelKind::None };
        self.needs_render = true;
    }

    pub fn toggle_series_bar(&mut self) {
        let next_visible =
            !(self.bottom_panel_visible && self.bottom_panel_kind == BottomPanelKind::Series);
        self.set_bottom_panel_visible(next_visible, BottomPanelKind::Series);
    }

    pub fn bottom_panel_height_px(&self) -> f32 {
        if self.bottom_panel_visible {
            BOTTOM_PANEL_HEIGHT_PX
        } else {
            0.0
        }
    }

    pub fn current_series_path(&self) -> Option<&std::path::Path> {
        self.opened_series
            .as_ref()
            .map(|series| series.current_path.as_path())
    }

    pub fn selected_series_path(&self) -> Option<&Path> {
        self.series_selected_path.as_deref()
    }

    pub fn cycle_series_selection(&mut self, offset: i32) -> Option<PathBuf> {
        let Some(series) = self.opened_series.as_ref() else {
            self.series_selected_path = None;
            return None;
        };
        if series.entries.is_empty() {
            self.series_selected_path = None;
            return None;
        }

        let selected_path = self
            .series_selected_path
            .as_ref()
            .map(|path| normalize_series_entry_path(path));
        let active_file_path = self
            .active_file_id
            .and_then(|file_id| self.get_file(file_id))
            .map(|file| normalize_series_entry_path(&file.path));

        let current_index = selected_path
            .as_ref()
            .and_then(|selected| {
                series
                    .entries
                    .iter()
                    .position(|entry| normalize_series_entry_path(&entry.path) == *selected)
            })
            .or_else(|| {
                active_file_path.as_ref().and_then(|active| {
                    series
                        .entries
                        .iter()
                        .position(|entry| normalize_series_entry_path(&entry.path) == *active)
                })
            })
            .unwrap_or(0) as i32;

        let next_index = (current_index + offset).rem_euclid(series.entries.len() as i32) as usize;
        let next_path = series.entries[next_index].path.clone();
        self.series_selected_path = Some(next_path.clone());
        self.needs_render = true;
        Some(next_path)
    }

    pub fn selected_series_entry_path(&self) -> Option<PathBuf> {
        let selected_path = self
            .series_selected_path
            .as_ref()
            .map(|path| normalize_series_entry_path(path))?;
        let series = self.opened_series.as_ref()?;
        series
            .entries
            .iter()
            .find(|entry| normalize_series_entry_path(&entry.path) == selected_path)
            .map(|entry| entry.path.clone())
    }

    pub fn series_can_go_back(&self) -> bool {
        self.opened_series
            .as_ref()
            .map(|series| !series.back_history.is_empty())
            .unwrap_or(false)
    }

    pub fn series_can_go_forward(&self) -> bool {
        self.opened_series
            .as_ref()
            .map(|series| !series.forward_history.is_empty())
            .unwrap_or(false)
    }

    pub fn series_can_go_up(&self) -> bool {
        self.current_series_path()
            .and_then(|path| path.parent())
            .is_some()
    }

    pub fn series_back_target(&self) -> Option<PathBuf> {
        self.opened_series
            .as_ref()
            .and_then(|series| series.back_history.last().cloned())
    }

    pub fn series_forward_target(&self) -> Option<PathBuf> {
        self.opened_series
            .as_ref()
            .and_then(|series| series.forward_history.last().cloned())
    }

    pub fn series_up_target(&self) -> Option<PathBuf> {
        self.current_series_path()
            .and_then(|path| path.parent())
            .map(|path| path.to_path_buf())
    }

    pub fn set_opened_series(
        &mut self,
        folder_path: PathBuf,
        entries: Vec<SeriesEntry>,
        navigation: SeriesNavigationMode,
    ) -> u64 {
        self.series_revision = self.series_revision.wrapping_add(1);
        self.series_content_revision = self.series_content_revision.wrapping_add(1);

        match self.opened_series.as_mut() {
            Some(series) => {
                let current_path = series.current_path.clone();
                match navigation {
                    SeriesNavigationMode::Initialize | SeriesNavigationMode::Replace => {}
                    SeriesNavigationMode::Push => {
                        if current_path != folder_path {
                            series.back_history.push(current_path);
                            series.forward_history.clear();
                        }
                    }
                    SeriesNavigationMode::Back => {
                        if let Some(previous_path) = series.back_history.pop()
                            && previous_path == folder_path
                        {
                            series.forward_history.push(current_path);
                        }
                    }
                    SeriesNavigationMode::Forward => {
                        if let Some(next_path) = series.forward_history.pop()
                            && next_path == folder_path
                        {
                            series.back_history.push(current_path);
                        }
                    }
                }
                series.current_path = folder_path;
                series.entries = entries;
            }
            None => {
                self.opened_series = Some(OpenedSeries {
                    current_path: folder_path,
                    entries,
                    back_history: Vec::new(),
                    forward_history: Vec::new(),
                });
            }
        }

        self.reconcile_series_selection();

        self.needs_render = true;
        self.series_revision
    }

    pub fn update_series_entry_thumbnail(
        &mut self,
        revision: u64,
        path: &std::path::Path,
        metadata_tooltip: String,
        objective_label: Option<String>,
        stain_label: Option<String>,
        thumbnail: Option<SeriesThumbnail>,
    ) -> bool {
        if self.series_revision != revision {
            return false;
        }

        let normalized_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let Some(series) = self.opened_series.as_mut() else {
            return false;
        };

        let Some(entry) = series.entries.iter_mut().find(|entry| {
            fs::canonicalize(&entry.path).unwrap_or_else(|_| entry.path.clone()) == normalized_path
        }) else {
            return false;
        };

        entry.metadata_tooltip = metadata_tooltip;
        entry.objective_label = objective_label;
        entry.stain_label = stain_label;
        entry.thumbnail = thumbnail;
        entry.thumbnail_loading = false;
        self.series_content_revision = self.series_content_revision.wrapping_add(1);
        self.needs_render = true;
        true
    }

    pub fn replace_opened_series_entries(
        &mut self,
        revision: u64,
        folder_path: &std::path::Path,
        entries: Vec<SeriesEntry>,
    ) -> bool {
        if self.series_revision != revision {
            return false;
        }

        let normalized_path =
            fs::canonicalize(folder_path).unwrap_or_else(|_| folder_path.to_path_buf());
        let Some(series) = self.opened_series.as_mut() else {
            return false;
        };

        if series.current_path != normalized_path {
            return false;
        }

        series.entries = entries;
        self.reconcile_series_selection();
        self.series_content_revision = self.series_content_revision.wrapping_add(1);
        self.needs_render = true;
        true
    }

    pub fn toggle_viewport_lock(&mut self) {
        self.viewport_lock_enabled = !self.viewport_lock_enabled;
        if self.viewport_lock_enabled {
            self.sync_locked_viewports_for_active_file(true);
        }
        self.needs_render = true;
    }

    pub fn mirror_locked_viewports_from_active(&mut self) -> bool {
        if !self.viewport_lock_enabled {
            return false;
        }
        self.sync_locked_viewports_for_active_file(false)
    }

    fn active_viewport_pane(&self) -> PaneId {
        if self.split_enabled {
            self.focused_pane
        } else {
            PaneId::PRIMARY
        }
    }

    fn sync_locked_viewports_for_active_file(&mut self, animated: bool) -> bool {
        let source_pane = self.active_viewport_pane();
        let Some(file_id) = self.active_file_id_for_pane(source_pane) else {
            return false;
        };
        self.sync_viewports_for_file_from_pane(file_id, source_pane, animated)
    }

    fn sync_viewports_for_file_from_pane(
        &mut self,
        file_id: i32,
        source_pane: PaneId,
        animated: bool,
    ) -> bool {
        let Some(source_viewport) = self
            .get_file(file_id)
            .and_then(|file| file.pane_state(source_pane))
            .map(|pane_state| pane_state.viewport.clone())
        else {
            return false;
        };

        let target_panes: Vec<_> = self
            .panes
            .iter()
            .enumerate()
            .filter_map(|(pane_index, _)| {
                let pane = PaneId(pane_index);
                (pane != source_pane && self.active_file_id_for_pane(pane) == Some(file_id))
                    .then_some(pane)
            })
            .collect();

        if target_panes.is_empty() {
            return false;
        }

        let mut changed = false;
        if let Some(file) = self.get_file_mut(file_id) {
            for target_pane in target_panes {
                file.ensure_pane_state_from(target_pane, source_pane);
                if let Some(target_state) = file.pane_state_mut(target_pane) {
                    if animated {
                        let center = source_viewport.viewport.center;
                        target_state.viewport.smooth_set_center_zoom(
                            center.x,
                            center.y,
                            source_viewport.viewport.zoom,
                        );
                    } else {
                        let width = target_state.viewport.viewport.width;
                        let height = target_state.viewport.viewport.height;
                        let mut synced_viewport = source_viewport.clone();
                        synced_viewport.viewport.width = width;
                        synced_viewport.viewport.height = height;
                        target_state.viewport = synced_viewport;
                    }
                    changed = true;
                }
            }
        }

        if changed {
            self.request_render();
        }

        changed
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
        Self::new()
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
