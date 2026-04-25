use crate::analysis::{start_viewport_analysis, start_whole_slide_analysis};
use crate::model::{LoadedModel, load_model};
use crate::state::{
    VisualizationMode, cancel_running_job, clear_cache_for_namespace, host_api, log_message,
    plugin_state, refresh_sidebar_if_available, request_render_if_available,
};
use abi_stable::std_types::{RString, RVec};
use plugin_api::ffi::{HostLogLevelFFI, UiPropertyFFI};
use serde_json::json;

const TOOLBAR_BUTTON_ID: &str = "toggle_eovae";

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
    match callback_name {
        "load-model" => load_model_from_dialog(),
        "clear-model" => clear_model(),
        "analyze-viewport" => analyze_viewport(),
        "analyze-slide" => analyze_whole_slide(),
        "cancel-job" => {
            cancel_running_job();
        }
        "jump-to-region" => jump_to_region(args_json),
        "mode-changed" => update_mode(args_json),
        "input-changed" => update_selected_tensor(args_json, true),
        "output-changed" => update_selected_tensor(args_json, false),
        "use-gpu-changed" => update_boolean(args_json, |state, value| state.config.use_gpu = value),
        "auto-update-changed" => update_boolean(args_json, |state, value| state.config.auto_update_viewport = value),
        _ => {}
    }
    refresh_sidebar_if_available();
    request_render_if_available();
}

fn load_model_from_dialog() {
    let Some(host) = host_api() else {
        log_message(HostLogLevelFFI::Error, "host API is not available");
        return;
    };
    let path = match (host.open_file_dialog)(host.context, RString::from("ONNX"), RString::from("onnx")).into_result() {
        Ok(path) if !path.is_empty() => path.to_string(),
        Ok(_) => return,
        Err(error) => {
            log_message(HostLogLevelFFI::Error, error.to_string());
            return;
        }
    };

    match load_model(&path, plugin_state().lock().unwrap().config.use_gpu) {
        Ok(model) => set_loaded_model(model),
        Err(error) => {
            let mut state = plugin_state().lock().unwrap();
            state.model = None;
            state.model_path = path;
            state.model_status = error.clone();
            log_message(HostLogLevelFFI::Error, error);
        }
    }
}

fn set_loaded_model(model: LoadedModel) {
    let namespace = model.summary.identity();
    {
        let mut state = plugin_state().lock().unwrap();
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
        state.job_status = "Idle".to_string();
    }
    clear_cache_for_namespace(namespace);
}

fn clear_model() {
    let mut state = plugin_state().lock().unwrap();
    state.model = None;
    state.model_path.clear();
    state.model_status = "No ONNX model loaded.".to_string();
    state.job_status = "Idle".to_string();
    state.cache.clear();
    state.sidebar_regions.clear();
    state.hot_regions.clear();
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
    let (model, namespace) = {
        let state = plugin_state().lock().unwrap();
        (state.model.clone(), state.cache_namespace.clone())
    };
    if let Some(model) = model {
        start_viewport_analysis(model, viewport, namespace);
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
    let (model, namespace) = {
        let state = plugin_state().lock().unwrap();
        (state.model.clone(), state.cache_namespace.clone())
    };
    if let Some(model) = model {
        start_whole_slide_analysis(
            model,
            active_file.file_id,
            active_file.path.to_string(),
            active_file.width,
            active_file.height,
            namespace,
        );
    }
}

fn update_mode(args_json: &str) {
    let value = serde_json::from_str::<String>(args_json).unwrap_or_default();
    let index = match value.as_str() {
        "Reconstruction" => 1,
        "Error Map" => 2,
        "Difference" => 3,
        _ => 0,
    };
    let mut state = plugin_state().lock().unwrap();
    state.visualization_mode = VisualizationMode::from_index(index);
}

fn update_selected_tensor(args_json: &str, is_input: bool) {
    let value = serde_json::from_str::<String>(args_json).unwrap_or_default();
    let mut state = plugin_state().lock().unwrap();
    let Some(model) = state.model.as_mut() else {
        return;
    };
    if is_input {
        if let Some(index) = model.summary.inputs.iter().position(|tensor| tensor.name == value) {
            model.summary.selected_input = index;
        }
    } else {
        if let Some(index) = model.summary.outputs.iter().position(|tensor| tensor.name == value) {
            model.summary.selected_output = index;
        }
    }
    let namespace = model.summary.identity();
    drop(state);
    clear_cache_for_namespace(namespace);
}

fn update_boolean<F>(args_json: &str, update: F)
where
    F: FnOnce(&mut crate::state::PluginState, bool),
{
    let value = serde_json::from_str::<bool>(args_json).unwrap_or(false);
    let mut state = plugin_state().lock().unwrap();
    update(&mut state, value);
}

fn jump_to_region(args_json: &str) {
    let region_id = serde_json::from_str::<String>(args_json).unwrap_or_default();
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
        let _ = (host.frame_active_rect)(host.context, x, y, width, height);
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

fn property(name: &str, value: serde_json::Value) -> UiPropertyFFI {
    UiPropertyFFI {
        name: name.into(),
        json_value: value.to_string().into(),
    }
}