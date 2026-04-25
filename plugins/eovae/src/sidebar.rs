use crate::analysis::{start_viewport_analysis, start_whole_slide_analysis};
use crate::model::{LoadedModel, load_model};
use crate::state::{
    VisualizationMode, cancel_running_job, clear_cache_for_namespace, host_api,
    load_persisted_model_path, log_message, plugin_state, refresh_sidebar_if_available,
    request_render_if_available, save_persisted_model_path,
};
use abi_stable::std_types::{RString, RVec};
use plugin_api::ffi::{HostLogLevelFFI, UiPropertyFFI};
use serde_json::json;
use std::thread;

const TOOLBAR_BUTTON_ID: &str = "toggle_eovae";

fn plugin_trace(message: impl AsRef<str>) {
    if std::env::var_os("EOV_PLUGIN_TRACE").is_some() {
        eprintln!("[eovae/sidebar] {}", message.as_ref());
    }
}

fn analysis_namespace(base_namespace: &str, mip_level: u32) -> String {
    format!("{base_namespace}|mip{mip_level}")
}

pub fn get_sidebar_properties() -> RVec<UiPropertyFFI> {
    let state = plugin_state().lock().unwrap();
    let summary = state.model.as_ref().map(|model| &model.summary);
    let input_rows = summary
        .map(|summary| {
            summary
                .inputs
                .iter()
                .map(|tensor| {
                    json!({
                        "name": tensor.name,
                        "dtype": tensor.dtype,
                        "shape": tensor.shape_label(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let output_rows = summary
        .map(|summary| {
            summary
                .outputs
                .iter()
                .map(|tensor| {
                    json!({
                        "name": tensor.name,
                        "dtype": tensor.dtype,
                        "shape": tensor.shape_label(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let input_options = summary
        .map(|summary| summary.inputs.iter().map(|tensor| tensor.name.clone()).collect::<Vec<_>>())
        .unwrap_or_default();
    let output_options = summary
        .map(|summary| summary.outputs.iter().map(|tensor| tensor.name.clone()).collect::<Vec<_>>())
        .unwrap_or_default();
    let hot_regions = state
        .sidebar_regions
        .iter()
        .map(|region| json!({ "id": region.id, "label": region.label, "score": region.score }))
        .collect::<Vec<_>>();
    let stats_summary = if state.error_stats.count == 0 {
        "No analysis yet.".to_string()
    } else {
        format!(
            "{} tiles analyzed. mean {:.4}, median {:.4}, p95 {:.4}, max {:.4}",
            state.error_stats.count,
            state.error_stats.mean,
            state.error_stats.median,
            state.error_stats.p95,
            state.error_stats.max,
        )
    };

    let properties = vec![
        property("model-path", json!(state.model_path)),
        property("model-status", json!(state.model_status)),
        property("model-loaded", json!(state.model.is_some())),
        property("input-rows", json!(input_rows)),
        property("output-rows", json!(output_rows)),
        property("input-options", json!(input_options)),
        property("output-options", json!(output_options)),
        property(
            "selected-input-index",
            json!(summary.map(|summary| summary.selected_input as i32).unwrap_or(0)),
        ),
        property(
            "selected-output-index",
            json!(summary.map(|summary| summary.selected_output as i32).unwrap_or(0)),
        ),
        property("selected-mode-index", json!(state.visualization_mode.to_index())),
        property("mip-options", json!(["1x", "2x", "4x", "8x"])),
        property("selected-mip-index", json!(state.config.mip_level as i32)),
        property("job-status", json!(state.job_status)),
        property("progress-value", json!(state.progress_value)),
        property("job-running", json!(state.job.is_some())),
        property("use-gpu", json!(state.config.use_gpu)),
        property("auto-update", json!(state.config.auto_update_viewport)),
        property("stats-summary", json!(stats_summary)),
        property("hot-regions", json!(hot_regions)),
    ];
    RVec::from(properties)
}

pub fn on_sidebar_callback(callback_name: &str, args_json: &str) {
    let (refresh_sidebar, request_render) = match callback_name {
        "load-model" => {
            load_model_from_dialog();
            (true, true)
        }
        "clear-model" => {
            clear_model();
            (true, true)
        }
        "analyze-viewport" => {
            analyze_viewport();
            (true, true)
        }
        "analyze-slide" => {
            analyze_whole_slide();
            (true, true)
        }
        "cancel-job" => (cancel_running_job(), false),
        "jump-to-region" => {
            jump_to_region(args_json);
            (false, false)
        }
        "mode-changed" => (false, update_mode(args_json)),
        "mip-changed" => {
            let changed = update_mip_level(args_json);
            (changed, changed)
        }
        "input-changed" => {
            let changed = update_selected_tensor(args_json, true);
            (changed, changed)
        }
        "output-changed" => {
            let changed = update_selected_tensor(args_json, false);
            (changed, changed)
        }
        "use-gpu-changed" => {
            let changed = update_boolean(args_json, |state, value| state.config.use_gpu = value);
            (changed, false)
        }
        "auto-update-changed" => {
            let changed = update_boolean(args_json, |state, value| state.config.auto_update_viewport = value);
            (changed, false)
        }
        _ => (false, false),
    };
    if refresh_sidebar {
        refresh_sidebar_if_available();
    }
    if request_render {
        request_render_if_available();
    }
}

pub fn initialize_from_config() {
    let saved_path = match load_persisted_model_path() {
        Ok(path) => path,
        Err(error) => {
            log_message(HostLogLevelFFI::Error, error);
            return;
        }
    };
    let Some(saved_path) = saved_path else {
        return;
    };

    let should_start = {
        let state = plugin_state().lock().unwrap();
        state.model.is_none() && !state.model_load_in_progress && state.model_path.is_empty()
    };
    if should_start {
        start_model_load(saved_path, false);
    }
}

fn load_model_from_dialog() {
    let Some(host) = host_api() else {
        log_message(HostLogLevelFFI::Error, "host API is not available");
        return;
    };

    plugin_trace("load_model dialog worker spawn");
    thread::spawn(move || {
        plugin_trace("load_model dialog worker opening dialog");
        let path = match (host.open_file_dialog)(
            host.context,
            RString::from("ONNX"),
            RString::from("onnx"),
        )
        .into_result()
        {
            Ok(path) if !path.is_empty() => path.to_string(),
            Ok(_) => {
                plugin_trace("load_model dialog canceled");
                return;
            }
            Err(error) => {
                log_message(HostLogLevelFFI::Error, error.to_string());
                plugin_trace(format!("load_model dialog error err={}", error));
                return;
            }
        };

        start_model_load(path, true);
    });
}

fn start_model_load(path: String, persist_on_success: bool) {
    let (load_generation, use_gpu) = {
        let mut state = plugin_state().lock().unwrap();
        state.model_load_generation = state.model_load_generation.wrapping_add(1);
        state.model_load_in_progress = true;
        state.model = None;
        state.model_path = path.clone();
        state.model_status = "Loading ONNX model...".to_string();
        state.job_status = "Idle".to_string();
        state.cache_namespace.clear();
        state.cache.clear();
        state.hot_regions.clear();
        state.sidebar_regions.clear();
        state.error_stats = crate::stats::ErrorStats::default();
        state.progress_value = 0.0;
        (state.model_load_generation, state.config.use_gpu)
    };
    refresh_sidebar_if_available();
    request_render_if_available();

    plugin_trace(format!(
        "load_model spawn path={} generation={} use_gpu={}",
        path, load_generation, use_gpu
    ));
    thread::spawn(move || {
        let result = load_model(&path, use_gpu);
        plugin_trace(format!(
            "load_model finished path={} generation={} success={}",
            path,
            load_generation,
            result.is_ok()
        ));
        finish_model_load(load_generation, path, result, persist_on_success);
    });
}

fn finish_model_load(
    load_generation: u64,
    path: String,
    result: Result<LoadedModel, String>,
    persist_on_success: bool,
) {
    let mut log_error = None;
    let mut model_to_persist = None;

    {
        let mut state = plugin_state().lock().unwrap();
        if state.model_load_generation != load_generation {
            plugin_trace(format!(
                "load_model stale result ignored path={} generation={} current_generation={}",
                path, load_generation, state.model_load_generation
            ));
            return;
        }

        state.model_load_in_progress = false;
        state.job_status = "Idle".to_string();

        match result {
            Ok(model) => {
                let namespace = analysis_namespace(&model.summary.identity(), state.config.mip_level);
                model_to_persist = persist_on_success.then(|| model.summary.path.clone());
                state.model_path = model.summary.path.clone();
                state.model_status = if model.summary.warnings.is_empty() {
                    format!(
                        "Loaded {} inputs / {} outputs. Layout {}. Tile {}.",
                        model.summary.inputs.len(),
                        model.summary.outputs.len(),
                        model.summary.layout,
                        model.summary.tile_size,
                    )
                } else {
                    model.summary.warnings.join(" ")
                };
                state.model = Some(model);
                state.cache_namespace = namespace;
                state.cache.clear();
                state.hot_regions.clear();
                state.sidebar_regions.clear();
                state.error_stats = crate::stats::ErrorStats::default();
                state.progress_value = 0.0;
            }
            Err(error) => {
                state.model = None;
                state.model_path = path.clone();
                state.model_status = error.clone();
                state.cache_namespace.clear();
                state.cache.clear();
                state.hot_regions.clear();
                state.sidebar_regions.clear();
                state.error_stats = crate::stats::ErrorStats::default();
                state.progress_value = 0.0;
                log_error = Some(error);
            }
        }
    }

    if let Some(error) = log_error {
        log_message(HostLogLevelFFI::Error, error);
    }
    if let Some(model_path) = model_to_persist
        && let Err(error) = save_persisted_model_path(Some(&model_path))
    {
        log_message(HostLogLevelFFI::Error, error);
    }
    refresh_sidebar_if_available();
    request_render_if_available();
}

fn clear_model() {
    let mut state = plugin_state().lock().unwrap();
    state.model_load_generation = state.model_load_generation.wrapping_add(1);
    state.model_load_in_progress = false;
    state.model = None;
    state.model_path.clear();
    state.model_status = "No ONNX model loaded.".to_string();
    state.job_status = "Idle".to_string();
    state.cache_namespace.clear();
    state.cache.clear();
    state.sidebar_regions.clear();
    state.hot_regions.clear();
    if let Err(error) = save_persisted_model_path(None) {
        log_message(HostLogLevelFFI::Error, error);
    }
}

fn analyze_viewport() {
    let host = match host_api() {
        Some(host) => host,
        None => return,
    };
    let snapshot = (host.get_snapshot)(host.context);
    let viewport = match snapshot.active_viewport.into_option() {
        Some(viewport) => viewport,
        None => {
            log_message(HostLogLevelFFI::Error, "no active viewport is available");
            return;
        }
    };
    let (model, namespace, mip_level) = {
        let state = plugin_state().lock().unwrap();
        let model = state.model.clone();
        let namespace = model
            .as_ref()
            .map(|model| analysis_namespace(&model.summary.identity(), state.config.mip_level))
            .unwrap_or_default();
        (model, namespace, state.config.mip_level)
    };
    if let Some(model) = model {
        start_viewport_analysis(model, viewport, namespace, mip_level);
    }
}

fn analyze_whole_slide() {
    let host = match host_api() {
        Some(host) => host,
        None => return,
    };
    let snapshot = (host.get_snapshot)(host.context);
    let active_file = match snapshot.active_file.into_option() {
        Some(file) => file,
        None => {
            log_message(HostLogLevelFFI::Error, "no active slide is open");
            return;
        }
    };
    let (model, namespace, mip_level) = {
        let state = plugin_state().lock().unwrap();
        let model = state.model.clone();
        let namespace = model
            .as_ref()
            .map(|model| analysis_namespace(&model.summary.identity(), state.config.mip_level))
            .unwrap_or_default();
        (model, namespace, state.config.mip_level)
    };
    if let Some(model) = model {
        start_whole_slide_analysis(
            model,
            active_file.file_id,
            active_file.path.to_string(),
            active_file.width,
            active_file.height,
            namespace,
            mip_level,
        );
    }
}

fn update_mode(args_json: &str) -> bool {
    let value = serde_json::from_str::<String>(args_json).unwrap_or_default();
    let index = match value.as_str() {
        "Reconstruction" => 1,
        "Error Map" => 2,
        "Difference" => 3,
        _ => 0,
    };
    let mut state = plugin_state().lock().unwrap();
    let new_mode = VisualizationMode::from_index(index);
    if state.visualization_mode == new_mode {
        return false;
    }
    state.visualization_mode = new_mode;
    true
}

fn update_mip_level(args_json: &str) -> bool {
    let value = serde_json::from_str::<String>(args_json).unwrap_or_default();
    let mip_level = match value.as_str() {
        "2x" => 1,
        "4x" => 2,
        "8x" => 3,
        _ => 0,
    };
    let namespace = {
        let mut state = plugin_state().lock().unwrap();
        if state.config.mip_level == mip_level {
            return false;
        }
        state.config.mip_level = mip_level;
        let Some(model) = state.model.as_ref() else {
            state.cache_namespace.clear();
            state.cache.clear();
            state.hot_regions.clear();
            state.sidebar_regions.clear();
            state.error_stats = crate::stats::ErrorStats::default();
            state.progress_value = 0.0;
            return true;
        };
        analysis_namespace(&model.summary.identity(), mip_level)
    };
    clear_cache_for_namespace(namespace);
    true
}

fn update_selected_tensor(args_json: &str, is_input: bool) -> bool {
    let value = serde_json::from_str::<String>(args_json).unwrap_or_default();
    let mut state = plugin_state().lock().unwrap();
    let Some(model) = state.model.as_mut() else {
        return false;
    };
    let changed = if is_input {
        model.summary.inputs.iter().position(|tensor| tensor.name == value).is_some_and(|index| {
            if model.summary.selected_input == index {
                false
            } else {
                model.summary.selected_input = index;
                true
            }
        })
    } else {
        model.summary.outputs.iter().position(|tensor| tensor.name == value).is_some_and(|index| {
            if model.summary.selected_output == index {
                false
            } else {
                model.summary.selected_output = index;
                true
            }
        })
    };
    if !changed {
        return false;
    }
    let namespace = analysis_namespace(&model.summary.identity(), state.config.mip_level);
    drop(state);
    clear_cache_for_namespace(namespace);
    true
}

fn update_boolean<F>(args_json: &str, update: F) -> bool
where
    F: FnOnce(&mut crate::state::PluginState, bool),
{
    let value = serde_json::from_str::<bool>(args_json).unwrap_or(false);
    let mut state = plugin_state().lock().unwrap();
    let previous = serde_json::to_value(json!({
        "use_gpu": state.config.use_gpu,
        "auto_update_viewport": state.config.auto_update_viewport,
    }))
    .ok();
    update(&mut state, value);
    let current = serde_json::to_value(json!({
        "use_gpu": state.config.use_gpu,
        "auto_update_viewport": state.config.auto_update_viewport,
    }))
    .ok();
    previous != current
}

fn jump_to_region(args_json: &str) {
    let region_id = serde_json::from_str::<String>(args_json)
        .ok()
        .or_else(|| {
            serde_json::from_str::<Vec<String>>(args_json)
                .ok()
                .and_then(|args| args.into_iter().next())
        })
        .unwrap_or_default();
    let parts = region_id.split(':').collect::<Vec<_>>();
    if parts.len() != 4 {
        return;
    }
    let parsed = (
        parts[0].parse::<f64>(),
        parts[1].parse::<f64>(),
        parts[2].parse::<f64>(),
        parts[3].parse::<f64>(),
    );
    let (Ok(x), Ok(y), Ok(width), Ok(height)) = parsed else {
        return;
    };
    if let Some(host) = host_api() {
        let center_x = x + width * 0.5;
        let center_y = y + height * 0.5;
        let _ = (host.set_active_viewport)(host.context, center_x, center_y, 1.0);
    }
}

pub fn show_sidebar() {
    let Some(host) = host_api() else {
        return;
    };
    let _ = (host.show_sidebar)(
        host.context,
        RString::from(TOOLBAR_BUTTON_ID),
        340,
        RString::from("ui/eovae-sidebar.slint"),
        RString::from("EovaeSidebar"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_sidebar_test_state() {
        let mut state = plugin_state().lock().unwrap();
        *state = crate::state::PluginState::default();
    }

    #[test]
    fn ignores_noop_mode_change() {
        reset_sidebar_test_state();
        assert!(!update_mode("\"Original\""));
    }

    #[test]
    fn ignores_noop_mip_change() {
        reset_sidebar_test_state();
        assert!(!update_mip_level("\"1x\""));
    }

    #[test]
    fn jump_to_region_accepts_single_string_array_payload() {
        let parsed = serde_json::from_str::<Vec<String>>("[\"10:20:30:40\"]")
            .ok()
            .and_then(|args| args.into_iter().next())
            .unwrap();
        assert_eq!(parsed, "10:20:30:40");
    }

    #[test]
    fn ignores_noop_tensor_selection_during_initialization() {
        reset_sidebar_test_state();
        assert!(!update_selected_tensor("\"input\"", true));
        assert!(!update_selected_tensor("\"reconstruction\"", false));
    }

    #[test]
    fn ignores_noop_boolean_updates() {
        reset_sidebar_test_state();
        assert!(!update_boolean("false", |state, value| state.config.use_gpu = value));
        assert!(!update_boolean("false", |state, value| state.config.auto_update_viewport = value));
    }
}

fn property(name: &str, value: serde_json::Value) -> UiPropertyFFI {
    UiPropertyFFI {
        name: name.into(),
        json_value: value.to_string().into(),
    }
}