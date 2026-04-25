use crate::analysis::{AnalysisConfig, HotRegion, TileCacheEntry};
use crate::model::LoadedModel;
use crate::stats::{ErrorStats, summarize_errors};
use abi_stable::std_types::RString;
use plugin_api::ffi::{HostApiVTable, HostLogLevelFFI};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    pub visualization_mode: VisualizationMode,
    pub config: AnalysisConfig,
    pub cache_namespace: String,
    pub cache: HashMap<String, TileCacheEntry>,
    pub hot_regions: Vec<HotRegion>,
    pub sidebar_regions: Vec<SidebarRegion>,
    pub error_stats: ErrorStats,
    pub progress_value: f32,
    pub job_status: String,
    pub job: Option<RunningJob>,
    pub last_auto_update: Option<Instant>,
    pub last_auto_viewport_key: Option<String>,
}

impl Default for PluginState {
    fn default() -> Self {
        Self {
            model: None,
            model_path: String::new(),
            model_status: "No ONNX model loaded.".to_string(),
            visualization_mode: VisualizationMode::Original,
            config: AnalysisConfig::default(),
            cache_namespace: String::new(),
            cache: HashMap::new(),
            hot_regions: Vec::new(),
            sidebar_regions: Vec::new(),
            error_stats: ErrorStats::default(),
            progress_value: 0.0,
            job_status: "Idle".to_string(),
            job: None,
            last_auto_update: None,
            last_auto_viewport_key: None,
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

pub fn refresh_sidebar_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.refresh_sidebar)(host_api.context);
    }
}

pub fn clear_cache_for_namespace(namespace: String) {
    let mut state = plugin_state().lock().unwrap();
    state.cache_namespace = namespace;
    state.cache.clear();
    state.hot_regions.clear();
    state.sidebar_regions.clear();
    state.error_stats = ErrorStats::default();
    state.progress_value = 0.0;
}

pub fn rebuild_sidebar_statistics(state: &mut PluginState) {
    let entries = state
        .cache
        .values()
        .filter(|entry| entry.namespace == state.cache_namespace)
        .collect::<Vec<_>>();
    state.error_stats = summarize_errors(entries.iter().map(|entry| entry.tile.mean_absolute_error));
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
            label: format!("x={} y={} size={}x{}", region.x, region.y, region.width, region.height),
            score: format!("MAE {:.4}", region.mean_absolute_error),
        })
        .collect();
}

pub fn should_auto_update(viewport_key: &str) -> bool {
    let state = plugin_state().lock().unwrap();
    if !state.config.auto_update_viewport || state.job.is_some() || state.model.is_none() {
        return false;
    }
    let now = Instant::now();
    if state.last_auto_viewport_key.as_deref() == Some(viewport_key)
        && state
            .last_auto_update
            .is_some_and(|last| now.duration_since(last) < Duration::from_millis(350))
    {
        return false;
    }
    true
}

pub fn mark_auto_update(viewport_key: String) {
    let mut state = plugin_state().lock().unwrap();
    state.last_auto_viewport_key = Some(viewport_key);
    state.last_auto_update = Some(Instant::now());
}

pub fn cancel_running_job() -> bool {
    let state = plugin_state().lock().unwrap();
    if let Some(job) = &state.job {
        job.cancel.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

fn host_api_cell() -> &'static Mutex<Option<HostApiVTable>> {
    HOST_API.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clears_cache_when_namespace_changes() {
        let mut state = PluginState::default();
        state.cache_namespace = "a".to_string();
        state.cache.insert(
            "key".to_string(),
            TileCacheEntry {
                namespace: "a".to_string(),
                tile: crate::analysis::AnalyzedTile::dummy(0, 0, 64, 64, 0.2),
            },
        );

        state.cache_namespace = "b".to_string();
        state.cache.retain(|_, entry| entry.namespace == state.cache_namespace);
        assert!(state.cache.is_empty());
    }
}