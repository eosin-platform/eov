use crate::analysis::{AnalysisConfig, HotRegion, TileCacheEntry};
use crate::model::LoadedModel;
use crate::stats::{ErrorHistogramBin, ErrorStats, build_error_histogram, summarize_errors};
use abi_stable::std_types::RString;
use plugin_api::ffi::{HostApiVTable, HostLogLevelFFI};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

fn default_true() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedConfig {
    pub default_model_path: Option<String>,
    pub model_section_expanded: bool,
    pub model_io_section_expanded: bool,
    pub analysis_section_expanded: bool,
    pub results_section_expanded: bool,
    pub analysis_threads: Option<usize>,
    pub gpu_batch_size: Option<usize>,
}

impl Default for PersistedConfig {
    fn default() -> Self {
        Self {
            default_model_path: None,
            model_section_expanded: default_true(),
            model_io_section_expanded: default_true(),
            analysis_section_expanded: default_true(),
            results_section_expanded: default_true(),
            analysis_threads: None,
            gpu_batch_size: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualizationMode {
    Original,
    Reconstruction,
    ErrorMap,
    Difference,
}

impl VisualizationMode {
    pub fn from_index(index: usize) -> Self {
        match index {
            1 => Self::Reconstruction,
            2 => Self::ErrorMap,
            3 => Self::Difference,
            _ => Self::Original,
        }
    }

    pub fn to_index(self) -> i32 {
        match self {
            Self::Original => 0,
            Self::Reconstruction => 1,
            Self::ErrorMap => 2,
            Self::Difference => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Viewport,
    WholeSlide,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalysisPhase {
    #[default]
    Idle,
    Running,
    Completed,
    Error,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct RunningJob {
    pub cancel: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub struct SidebarRegion {
    pub id: String,
    pub label: String,
    pub score: String,
}

pub struct PluginState {
    pub model: Option<LoadedModel>,
    pub model_path: String,
    pub model_status: String,
    pub model_load_generation: u64,
    pub analysis_run_generation: u64,
    pub model_load_in_progress: bool,
    pub visualization_mode: VisualizationMode,
    pub config: AnalysisConfig,
    pub cache_namespace: String,
    pub cache: HashMap<String, TileCacheEntry>,
    pub hot_regions: Vec<HotRegion>,
    pub sidebar_regions: Vec<SidebarRegion>,
    pub error_stats: ErrorStats,
    pub error_histogram: Vec<ErrorHistogramBin>,
    pub progress_value: f32,
    pub job_status: String,
    pub job: Option<RunningJob>,
    pub analysis_phase: AnalysisPhase,
    pub analysis_started_at: Option<Instant>,
    pub analysis_elapsed: Option<Duration>,
    pub analysis_error_message: Option<String>,
    pub model_section_expanded: bool,
    pub model_io_section_expanded: bool,
    pub analysis_section_expanded: bool,
    pub results_section_expanded: bool,
    pub hovered_region_id: Option<String>,
    pub pulsing_region_id: Option<String>,
    pub pulsing_region_started_at: Option<Instant>,
    pub grid_enabled: bool,
    pub mip_combo_hovered: bool,
    pub mip_dropdown_open: bool,
    pub mip_preview_mip_level: Option<u32>,
}

impl Default for PluginState {
    fn default() -> Self {
        Self {
            model: None,
            model_path: String::new(),
            model_status: "No ONNX model loaded.".to_string(),
            model_load_generation: 0,
            analysis_run_generation: 0,
            model_load_in_progress: false,
            visualization_mode: VisualizationMode::Original,
            config: AnalysisConfig::default(),
            cache_namespace: String::new(),
            cache: HashMap::new(),
            hot_regions: Vec::new(),
            sidebar_regions: Vec::new(),
            error_stats: ErrorStats::default(),
            error_histogram: Vec::new(),
            progress_value: 0.0,
            job_status: "Idle".to_string(),
            job: None,
            analysis_phase: AnalysisPhase::Idle,
            analysis_started_at: None,
            analysis_elapsed: None,
            analysis_error_message: None,
            model_section_expanded: true,
            model_io_section_expanded: true,
            analysis_section_expanded: true,
            results_section_expanded: true,
            hovered_region_id: None,
            pulsing_region_id: None,
            pulsing_region_started_at: None,
            grid_enabled: false,
            mip_combo_hovered: false,
            mip_dropdown_open: false,
            mip_preview_mip_level: None,
        }
    }
}

static HOST_API: OnceLock<Mutex<Option<HostApiVTable>>> = OnceLock::new();
static PLUGIN_STATE: OnceLock<Mutex<PluginState>> = OnceLock::new();

pub fn set_host_api(host_api: HostApiVTable) {
    *host_api_cell().lock().unwrap() = Some(host_api);
}

pub fn host_api() -> Option<HostApiVTable> {
    *host_api_cell().lock().unwrap()
}

pub fn plugin_state() -> &'static Mutex<PluginState> {
    PLUGIN_STATE.get_or_init(|| Mutex::new(PluginState::default()))
}

pub fn log_message(level: HostLogLevelFFI, message: impl AsRef<str>) {
    if let Some(host_api) = host_api() {
        (host_api.log_message)(host_api.context, level, RString::from(message.as_ref()));
    }
}

pub fn request_render_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.request_render)(host_api.context);
    }
}

pub fn set_hud_toolbar_button_active_if_available(button_id: &str, active: bool) {
    if let Some(host_api) = host_api() {
        let _ = (host_api.set_hud_toolbar_button_active)(host_api.context, button_id.into(), active);
    }
}

pub fn refresh_sidebar_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.refresh_sidebar)(host_api.context);
    }
}

pub fn clear_cache_for_namespace(namespace: String) {
    let mut state = plugin_state().lock().unwrap();
    let had_grid_enabled = state.grid_enabled;
    state.cache_namespace = namespace;
    state.cache.clear();
    state.hot_regions.clear();
    state.sidebar_regions.clear();
    state.error_stats = ErrorStats::default();
    state.error_histogram.clear();
    state.progress_value = 0.0;
    state.job_status = "Idle".to_string();
    state.analysis_phase = AnalysisPhase::Idle;
    state.analysis_started_at = None;
    state.analysis_elapsed = None;
    state.analysis_error_message = None;
    state.hovered_region_id = None;
    state.pulsing_region_id = None;
    state.pulsing_region_started_at = None;
    state.grid_enabled = false;
    state.mip_combo_hovered = false;
    state.mip_dropdown_open = false;
    state.mip_preview_mip_level = None;
    drop(state);

    if had_grid_enabled {
        set_hud_toolbar_button_active_if_available("toggle_grid", false);
    }
}

pub fn max_analysis_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .max(1)
}

pub fn clamp_analysis_threads(value: usize) -> usize {
    value.clamp(1, max_analysis_threads())
}

pub fn clamp_gpu_batch_size(value: usize) -> usize {
    value.clamp(1, 1024)
}

pub fn rebuild_sidebar_statistics(state: &mut PluginState) {
    let entries = state
        .cache
        .values()
        .filter(|entry| entry.namespace == state.cache_namespace)
        .collect::<Vec<_>>();
    state.error_stats =
        summarize_errors(entries.iter().map(|entry| entry.tile.mean_absolute_error));
    state.error_histogram = build_error_histogram(
        entries.iter().map(|entry| entry.tile.mean_absolute_error),
        12,
    );
    state.hot_regions = entries
        .iter()
        .map(|entry| HotRegion {
            id: entry.tile.id(),
            x: entry.tile.x,
            y: entry.tile.y,
            width: entry.tile.width,
            height: entry.tile.height,
            mean_absolute_error: entry.tile.mean_absolute_error,
        })
        .collect::<Vec<_>>();
    state.hot_regions.sort_by(|left, right| {
        right
            .mean_absolute_error
            .partial_cmp(&left.mean_absolute_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    state.hot_regions.truncate(12);
    state.sidebar_regions = state
        .hot_regions
        .iter()
        .map(|region| SidebarRegion {
            id: region.id.clone(),
            label: format!(
                "x={} y={} size={}x{}",
                region.x, region.y, region.width, region.height
            ),
            score: format!("MAE {:.4}", region.mean_absolute_error),
        })
        .collect();
}

pub fn cancel_running_job() -> bool {
    let mut state = plugin_state().lock().unwrap();
    if let Some(job) = &state.job {
        job.cancel.store(true, Ordering::Relaxed);
        if state.analysis_phase == AnalysisPhase::Running {
            state.job_status = "Cancelling analysis...".to_string();
        }
        return true;
    }
    false
}

fn host_api_cell() -> &'static Mutex<Option<HostApiVTable>> {
    HOST_API.get_or_init(|| Mutex::new(None))
}

pub fn load_persisted_config() -> Result<PersistedConfig, String> {
    let path = persisted_config_path()?;
    if !path.exists() {
        return Ok(PersistedConfig::default());
    }
    let json = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read eovae config '{}': {err}", path.display()))?;
    serde_json::from_str::<PersistedConfig>(&json)
        .map_err(|err| format!("failed to parse eovae config '{}': {err}", path.display()))
}

pub fn save_persisted_config_field(
    update: impl FnOnce(&mut PersistedConfig),
) -> Result<(), String> {
    let path = persisted_config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create eovae config directory '{}': {err}",
                parent.display()
            )
        })?;
    }

    let mut config = load_persisted_config()?;
    update(&mut config);

    let json = serde_json::to_string_pretty(&config)
        .map_err(|err| format!("failed to serialize eovae config: {err}"))?;
    fs::write(&path, json)
        .map_err(|err| format!("failed to write eovae config '{}': {err}", path.display()))
}

pub fn save_persisted_model_path(model_path: Option<&str>) -> Result<(), String> {
    save_persisted_config_field(|config| {
        config.default_model_path = model_path.map(str::to_string);
    })
}

fn persisted_config_path() -> Result<PathBuf, String> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| "could not determine config directory for eovae".to_string())?
        .join("eov");
    Ok(config_dir.join("eovae.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clears_cache_when_namespace_changes() {
        let mut state = PluginState {
            cache_namespace: "a".to_string(),
            ..Default::default()
        };
        state.cache.insert(
            "key".to_string(),
            TileCacheEntry {
                namespace: "a".to_string(),
                tile: crate::analysis::AnalyzedTile::dummy(0, 0, 64, 64, 0.2),
            },
        );

        state.cache_namespace = "b".to_string();
        state
            .cache
            .retain(|_, entry| entry.namespace == state.cache_namespace);
        assert!(state.cache.is_empty());
    }
}
