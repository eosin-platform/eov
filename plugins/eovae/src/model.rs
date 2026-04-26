use crate::state::{clamp_analysis_threads, host_api, plugin_state};
use ndarray::{Array4, ArrayViewD, Axis, Ix4};
use onnx_extractor::{DataType as OnnxDataType, OnnxModel, OnnxOperation, OnnxTensor};
use ort::{
    ep,
    logging::LogLevel,
    session::{OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::env;
use std::fmt;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const GPU_SESSION_BUILD_TIMEOUT: Duration = Duration::from_secs(20);
const CPU_SESSION_BUILD_TIMEOUT: Duration = Duration::from_secs(5);
const PLACEMENT_SENSITIVE_OPS: &[&str] = &[
    "Shape",
    "Gather",
    "Slice",
    "Resize",
    "InstanceNormalization",
    "Clip",
    "Reshape",
    "Constant",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Layout {
    Nchw,
    Nhwc,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nchw => write!(f, "NCHW"),
            Self::Nhwc => write!(f, "NHWC"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub is_image_candidate: bool,
}

impl TensorSummary {
    pub fn shape_label(&self) -> String {
        let dims = self
            .shape
            .iter()
            .map(|dimension| {
                if *dimension < 0 {
                    "?".to_string()
                } else {
                    dimension.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" x ");
        format!("[{dims}]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelSummary {
    pub path: String,
    pub inputs: Vec<TensorSummary>,
    pub outputs: Vec<TensorSummary>,
    pub selected_input: usize,
    pub selected_output: usize,
    pub layout: Layout,
    pub tile_size: u32,
    pub warnings: Vec<String>,
}

impl ModelSummary {
    pub fn identity(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.path, self.selected_input, self.selected_output, self.layout, self.tile_size
        )
    }
}

#[derive(Clone)]
pub struct LoadedModel {
    pub summary: ModelSummary,
    pub session: Arc<Mutex<Option<CachedSession>>>,
    prefer_gpu: bool,
    session_threads_override: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SessionSettings {
    prefer_gpu: bool,
    analysis_threads: usize,
    allow_cpu_fallback: bool,
    opt_level: GraphOptimizationLevel,
    opt_level_name: String,
    profile_path: Option<PathBuf>,
}

pub struct CachedSession {
    session: Session,
    settings: SessionSettings,
    profile_finished: bool,
}

pub struct ReconstructionResult {
    pub rgb: Vec<u8>,
}

pub struct BatchReconstructionInput<'a> {
    pub rgba: &'a [u8],
    pub width: u32,
    pub height: u32,
}

pub fn load_model(path: &str, prefer_gpu: bool) -> Result<LoadedModel, String> {
    if !path.ends_with(".onnx") {
        return Err("only .onnx models are supported".to_string());
    }
    if !Path::new(path).exists() {
        return Err(format!("model does not exist: {path}"));
    }

    let onnx_model = OnnxModel::load_from_file(path).map_err(|error| error.to_string())?;
    log_model_operation_summary(&onnx_model);
    let inputs = onnx_model
        .get_input_tensors()
        .into_iter()
        .map(tensor_to_summary)
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = onnx_model
        .get_output_tensors()
        .into_iter()
        .map(tensor_to_summary)
        .collect::<Result<Vec<_>, _>>()?;

    if inputs.is_empty() {
        return Err("model has no inputs".to_string());
    }
    if outputs.is_empty() {
        return Err("model has no outputs".to_string());
    }

    let inferred = infer_model_contract(path, &inputs, &outputs)?;
    log_selected_tensors(&inferred);
    log_selected_output_subgraph(&onnx_model, &inferred);
    Ok(LoadedModel {
        summary: inferred,
        session: Arc::new(Mutex::new(None)),
        prefer_gpu,
        session_threads_override: None,
    })
}

impl LoadedModel {
    pub fn clone_for_analysis_worker(&self, session_threads: usize) -> Self {
        Self {
            summary: self.summary.clone(),
            session: Arc::new(Mutex::new(None)),
            prefer_gpu: self.prefer_gpu,
            session_threads_override: Some(session_threads.max(1)),
        }
    }
}

fn build_inference_session(
    path: &str,
    prefer_gpu: bool,
    analysis_threads: usize,
    layout: Layout,
) -> Result<Session, String> {
    let settings = session_settings(prefer_gpu, analysis_threads);
    build_session_with_watchdog(path, &settings, layout)
}

fn build_inference_session_inner(
    path: &str,
    settings: &SessionSettings,
    layout: Layout,
) -> Result<Session, String> {
    let builder = Session::builder()
        .map_err(|error| error.to_string())?
        .with_optimization_level(settings.opt_level)
        .map_err(|error| error.to_string())?
        .with_parallel_execution(settings.analysis_threads > 1)
        .map_err(|error| error.to_string())?
        .with_intra_threads(settings.analysis_threads)
        .map_err(|error| error.to_string())?;
    let mut builder = if ort_debug_enabled() {
        let mut builder = builder
            .with_log_level(LogLevel::Verbose)
            .map_err(|error| error.to_string())?
            .with_log_verbosity(1)
            .map_err(|error| error.to_string())?
            .with_log_id("eovae-ort")
            .map_err(|error| error.to_string())?;
        if settings.prefer_gpu {
            let optimized_model_path = diagnostic_output_path(path, "optimized", "onnx");
            debug_timing(&format!(
                "writing optimized gpu model to {}",
                optimized_model_path.display()
            ));
            builder = builder
                .with_optimized_model_path(&optimized_model_path)
                .map_err(|error| error.to_string())?;
        }
        builder
    } else {
        builder
    };
    if settings.prefer_gpu && settings.allow_cpu_fallback {
        debug_timing(
            "DIAGNOSTIC ONLY: CPU fallback enabled to identify CUDA placement failures. This does not validate GPU execution.",
        );
    }
    if let Some(profile_path) = &settings.profile_path {
        debug_timing(&format!("enabling ORT profiling at {}", profile_path.display()));
        builder = builder
            .with_profiling(profile_path)
            .map_err(|error| error.to_string())?;
    }
    builder = if settings.prefer_gpu {
        // GPU analysis uses variable batch sizes; memory pattern caching can work against that.
        let builder = builder
            .with_memory_pattern(false)
            .map_err(|error| error.to_string())?;
        if settings.allow_cpu_fallback {
            builder
        } else {
            builder
                .with_disable_cpu_fallback()
                .map_err(|error| error.to_string())?
        }
    } else {
        builder
    };
    if settings.prefer_gpu {
        debug_timing("configuring gpu session options");
        let cuda = ep::CUDA::default()
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_cuda_graph(false)
            .with_tf32(true)
            .with_prefer_nhwc(matches!(layout, Layout::Nhwc))
            .build();
        builder = builder
            .with_execution_providers([cuda])
            .map_err(|error| error.to_string())?;
    }

    let timeout = if settings.prefer_gpu {
        GPU_SESSION_BUILD_TIMEOUT
    } else {
        CPU_SESSION_BUILD_TIMEOUT
    };
    commit_session_with_timeout(
        builder,
        path,
        timeout,
        if settings.prefer_gpu { "gpu" } else { "cpu" },
        settings,
    )
}

fn build_session_with_watchdog(
    path: &str,
    settings: &SessionSettings,
    layout: Layout,
) -> Result<Session, String> {
    let timeout = if settings.prefer_gpu {
        GPU_SESSION_BUILD_TIMEOUT
    } else {
        CPU_SESSION_BUILD_TIMEOUT
    };
    let label = if settings.prefer_gpu { "gpu" } else { "cpu" };
    let (sender, receiver) = mpsc::channel();
    let path = path.to_string();
    let worker_settings = settings.clone();

    thread::spawn(move || {
        let result = build_inference_session_inner(&path, &worker_settings, layout);
        let _ = sender.send(result);
    });

    debug_timing(&format!(
        "waiting up to {:?} for {label} session build thread opt_level={} cpu_fallback={} profiling={}",
        timeout,
        settings.opt_level_name,
        settings.allow_cpu_fallback,
        settings.profile_path.is_some()
    ));
    let result = match receiver.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "{label} session build timed out after {:?}",
            timeout
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(format!("{label} session build thread disconnected"))
        }
    };
    if settings.prefer_gpu && !settings.allow_cpu_fallback {
        match &result {
            Ok(_) => debug_timing(&format!(
                "strict CUDA session build with opt_level={} succeeded",
                settings.opt_level_name
            )),
            Err(error) => debug_timing(&format!(
                "strict CUDA session build with opt_level={} failed: {}",
                settings.opt_level_name, error
            )),
        }
    }
    result
}

fn commit_session_with_timeout(
    mut builder: ort::session::builder::SessionBuilder,
    path: &str,
    timeout: Duration,
    label: &str,
    settings: &SessionSettings,
) -> Result<Session, String> {
    let canceler = builder.canceler();
    let start = Instant::now();
    let timeout_label = label.to_string();
    let (watchdog_tx, watchdog_rx) = mpsc::channel();
    thread::spawn(move || {
        match watchdog_rx.recv_timeout(timeout) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                debug_timing(&format!(
                    "{timeout_label} session build exceeded {:?}; requesting cancellation",
                    timeout
                ));
                let _ = canceler.cancel();
            }
        }
    });

    debug_timing(&format!(
        "starting {label} session build opt_level={} cpu_fallback={} profiling={}",
        settings.opt_level_name,
        settings.allow_cpu_fallback,
        settings.profile_path.is_some()
    ));
    let result = builder
        .commit_from_file(path)
        .map_err(|error| error.to_string());
    let _ = watchdog_tx.send(());
    debug_timing(&format!(
        "{label} session build finished in {:?}",
        start.elapsed()
    ));
    result.map_err(|error| format!("{label} session build failed: {error}"))
}

fn tensor_to_summary(tensor: &OnnxTensor) -> Result<TensorSummary, String> {
    let dtype = tensor.data_type();
    Ok(TensorSummary {
        name: tensor.name().to_string(),
        dtype: format!("{dtype:?}"),
        is_image_candidate: is_image_tensor(dtype, tensor.shape()),
        shape: tensor.shape().to_vec(),
    })
}

fn is_image_tensor(dtype: OnnxDataType, shape: &[i64]) -> bool {
    if dtype != OnnxDataType::Float && dtype != OnnxDataType::Float16 {
        return false;
    }
    is_image_shape(shape)
}

fn is_image_shape(shape: &[i64]) -> bool {
    if shape.len() != 4 {
        return false;
    }
    let channels_second = shape.get(1).copied().unwrap_or(-1);
    let channels_last = shape.get(3).copied().unwrap_or(-1);
    matches!(channels_second, 1 | 3 | 4 | -1) || matches!(channels_last, 1 | 3 | 4 | -1)
}

pub fn infer_model_contract(
    path: &str,
    inputs: &[TensorSummary],
    outputs: &[TensorSummary],
) -> Result<ModelSummary, String> {
    let input_candidates = inputs
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.is_image_candidate)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let output_candidates = outputs
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.is_image_candidate)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    let selected_input = *input_candidates
        .first()
        .ok_or_else(|| "no image-like tensor input was found".to_string())?;
    let selected_output = select_output(outputs, selected_input, inputs)?;
    let layout = infer_layout(&inputs[selected_input], &outputs[selected_output]);
    let tile_size = infer_tile_size(&inputs[selected_input], layout)
        .or_else(|| infer_tile_size(&outputs[selected_output], layout))
        .unwrap_or(256);

    let mut warnings = Vec::new();
    if input_candidates.len() > 1 {
        warnings
            .push("multiple image-like inputs were found; the first one was selected".to_string());
    }
    if output_candidates.len() > 1 {
        warnings.push("multiple image-like outputs were found; the current output can be changed from the sidebar".to_string());
    }

    Ok(ModelSummary {
        path: path.to_string(),
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
        selected_input,
        selected_output,
        layout,
        tile_size,
        warnings,
    })
}

fn select_output(
    outputs: &[TensorSummary],
    selected_input: usize,
    inputs: &[TensorSummary],
) -> Result<usize, String> {
    let input_shape = &inputs[selected_input].shape;
    let exact_match = outputs
        .iter()
        .enumerate()
        .find(|(_, output)| output.shape == *input_shape)
        .map(|(index, _)| index);
    if let Some(index) = exact_match {
        return Ok(index);
    }

    let named_match = outputs
        .iter()
        .enumerate()
        .find(|(_, output)| {
            let lowered = output.name.to_ascii_lowercase();
            lowered.contains("recon") || lowered.contains("decode") || lowered.contains("output")
        })
        .map(|(index, _)| index);

    named_match.ok_or_else(|| "no plausible reconstruction output tensor was found".to_string())
}

fn infer_layout(input: &TensorSummary, output: &TensorSummary) -> Layout {
    for tensor in [input, output] {
        if tensor.shape.len() == 4 {
            let second = tensor.shape[1];
            let last = tensor.shape[3];
            if matches!(last, 1 | 3 | 4) {
                return Layout::Nhwc;
            }
            if matches!(second, 1 | 3 | 4) {
                return Layout::Nchw;
            }
        }
    }
    Layout::Nchw
}

fn infer_tile_size(tensor: &TensorSummary, layout: Layout) -> Option<u32> {
    if tensor.shape.len() != 4 {
        return None;
    }
    let (height, width) = match layout {
        Layout::Nchw => (tensor.shape[2], tensor.shape[3]),
        Layout::Nhwc => (tensor.shape[1], tensor.shape[2]),
    };
    if height > 0 && width > 0 && height == width {
        return Some(height as u32);
    }
    None
}

pub fn run_reconstruction(
    model: &LoadedModel,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<ReconstructionResult, String> {
    let started = Instant::now();
    let input = preprocess_rgba(rgba, width, height, model.summary.layout)?;
    debug_timing(&format!("preprocess completed in {:?}", started.elapsed()));

    let session_started = Instant::now();
    let mut session = ensure_inference_session(model)?;
    debug_timing(&format!(
        "ensure_inference_session completed in {:?}",
        session_started.elapsed()
    ));

    let session = session
        .as_mut()
        .ok_or_else(|| "inference session is unavailable".to_string())?;
    let selected_output_name = model
        .summary
        .outputs
        .get(model.summary.selected_output)
        .map(|tensor| tensor.name.as_str())
        .ok_or_else(|| "selected output index is out of bounds".to_string())?;
    let run_options = RunOptions::new()
        .map_err(|error| error.to_string())?
        .with_outputs(OutputSelector::no_default().with(selected_output_name));

    let run_started = Instant::now();
    let reconstruction = {
        let outputs = session
            .session
            .run_with_options(
                ort::inputs![TensorRef::from_array_view(input.view()).map_err(|error| error.to_string())?],
                &run_options,
            )
            .map_err(|error| error.to_string())?;
        debug_timing(&format!(
            "session.run completed in {:?}",
            run_started.elapsed()
        ));

        let output_value = &outputs[selected_output_name];
        let output = output_value
            .try_extract_array::<f32>()
            .map_err(|error| error.to_string())?;
        postprocess_reconstruction(output, model.summary.layout)
    };
    maybe_finalize_ort_profile(session);
    reconstruction
}

pub fn run_reconstruction_batch(
    model: &LoadedModel,
    inputs: &[BatchReconstructionInput<'_>],
) -> Result<Vec<ReconstructionResult>, String> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let width = inputs[0].width;
    let height = inputs[0].height;
    debug_timing(&format!(
        "run_reconstruction_batch batch_size={} tile={}x{} layout={}",
        inputs.len(),
        width,
        height,
        model.summary.layout
    ));
    if inputs
        .iter()
        .any(|input| input.width != width || input.height != height)
    {
        return Err("batched reconstruction requires all tiles to have the same dimensions".to_string());
    }

    let started = Instant::now();
    let input = preprocess_rgba_batch(inputs, model.summary.layout)?;
    debug_timing(&format!("batch preprocess completed in {:?}", started.elapsed()));

    let session_started = Instant::now();
    let mut session = ensure_inference_session(model)?;
    debug_timing(&format!(
        "batch ensure_inference_session completed in {:?}",
        session_started.elapsed()
    ));

    let session = session
        .as_mut()
        .ok_or_else(|| "inference session is unavailable".to_string())?;
    let selected_output_name = model
        .summary
        .outputs
        .get(model.summary.selected_output)
        .map(|tensor| tensor.name.as_str())
        .ok_or_else(|| "selected output index is out of bounds".to_string())?;
    let run_options = RunOptions::new()
        .map_err(|error| error.to_string())?
        .with_outputs(OutputSelector::no_default().with(selected_output_name));

    let run_started = Instant::now();
    let reconstruction = {
        let outputs = session
            .session
            .run_with_options(
                ort::inputs![TensorRef::from_array_view(input.view()).map_err(|error| error.to_string())?],
                &run_options,
            )
            .map_err(|error| error.to_string())?;
        debug_timing(&format!(
            "batch session.run completed in {:?}",
            run_started.elapsed()
        ));

        let output_value = &outputs[selected_output_name];
        let output = output_value
            .try_extract_array::<f32>()
            .map_err(|error| error.to_string())?;
        postprocess_reconstruction_batch(output, model.summary.layout)
    };
    maybe_finalize_ort_profile(session);
    reconstruction
}

fn ensure_inference_session(
    model: &LoadedModel,
) -> Result<std::sync::MutexGuard<'_, Option<CachedSession>>, String> {
    let settings = desired_execution_settings(model);
    {
        let session = model
            .session
            .lock()
            .map_err(|_| "model session lock is poisoned".to_string())?;
        if session
            .as_ref()
            .is_some_and(|session| session.settings == settings)
        {
            debug_timing("reusing cached inference session");
            return Ok(session);
        }
    }

    debug_timing(&format!(
        "creating new {} inference session with {} thread(s) opt_level={} cpu_fallback={} profiling={}",
        if settings.prefer_gpu { "gpu" } else { "cpu" },
        settings.analysis_threads,
        settings.opt_level_name,
        settings.allow_cpu_fallback,
        settings.profile_path.is_some()
    ));
    let session = match build_inference_session(
        &model.summary.path,
        settings.prefer_gpu,
        settings.analysis_threads,
        model.summary.layout,
    ) {
        Ok(session) => session,
        Err(error) => return Err(error),
    };

    let mut guard = model
        .session
        .lock()
        .map_err(|_| "model session lock is poisoned".to_string())?;
    *guard = Some(CachedSession {
        session,
        settings,
        profile_finished: false,
    });
    Ok(guard)
}

fn desired_execution_settings(model: &LoadedModel) -> SessionSettings {
    let prefer_gpu = host_api()
        .map(|host| (host.get_snapshot)(host.context))
        .map(|snapshot| snapshot.render_backend.to_ascii_lowercase() == "gpu")
        .unwrap_or(model.prefer_gpu);
    let analysis_threads = model.session_threads_override.unwrap_or_else(|| {
        clamp_analysis_threads(plugin_state().lock().unwrap().config.analysis_threads)
    });
    session_settings(prefer_gpu, analysis_threads)
}

fn session_settings(prefer_gpu: bool, analysis_threads: usize) -> SessionSettings {
    let (opt_level, opt_level_name) = graph_optimization_level_from_env();
    SessionSettings {
        prefer_gpu,
        analysis_threads,
        allow_cpu_fallback: prefer_gpu && env_flag_enabled("EOVAE_DIAG_ALLOW_CPU_FALLBACK"),
        opt_level,
        opt_level_name,
        profile_path: ort_profile_path(),
    }
}

fn debug_timing(message: &str) {
    if std::env::var_os("EOVAE_DEBUG_TIMING").is_some() {
        eprintln!("[eovae] {message}");
    }
}

fn ort_debug_enabled() -> bool {
    env::var_os("EOVAE_DEBUG_TIMING").is_some() || env::var_os("EOV_PLUGIN_TRACE").is_some()
}

fn env_flag_enabled(name: &str) -> bool {
    env::var_os(name).is_some_and(|value| !value.is_empty())
}

fn graph_optimization_level_from_env() -> (GraphOptimizationLevel, String) {
    parse_graph_optimization_level_value(env::var("EOVAE_ORT_OPT_LEVEL").ok().as_deref())
}

fn parse_graph_optimization_level_value(
    value: Option<&str>,
) -> (GraphOptimizationLevel, String) {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => (GraphOptimizationLevel::Level3, "level3(default)".to_string()),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "disable" => (GraphOptimizationLevel::Disable, "disable".to_string()),
            "basic" => (GraphOptimizationLevel::Level1, "basic".to_string()),
            "extended" => (GraphOptimizationLevel::Level2, "extended".to_string()),
            "all" => (GraphOptimizationLevel::All, "all".to_string()),
            other => {
                debug_timing(&format!(
                    "invalid EOVAE_ORT_OPT_LEVEL={other}; using level3(default)"
                ));
                (GraphOptimizationLevel::Level3, format!("invalid({other}) -> level3(default)"))
            }
        },
    }
}

fn ort_profile_path() -> Option<PathBuf> {
    env_flag_enabled("EOVAE_ORT_PROFILE")
        .then(|| env::temp_dir().join(format!("eovae-ort-profile-{}.json", process::id())))
}

fn diagnostic_output_path(model_path: &str, kind: &str, extension: &str) -> PathBuf {
    let model_stem = Path::new(model_path)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("model");
    env::temp_dir().join(format!("eovae-{model_stem}-{kind}.{extension}"))
}

fn log_model_operation_summary(onnx_model: &OnnxModel) {
    if !ort_debug_enabled() {
        return;
    }

    let mut counts = onnx_model.count_operations_by_type().into_iter().collect::<Vec<_>>();
    counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let summary = counts
        .iter()
        .map(|(op_type, count)| format!("{op_type}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    debug_timing(&format!(
        "model operations: total_nodes={} unique_ops={} {summary}",
        onnx_model.operations.len(),
        counts.len()
    ));
}

fn log_selected_tensors(summary: &ModelSummary) {
    if !ort_debug_enabled() {
        return;
    }

    let input_name = summary
        .inputs
        .get(summary.selected_input)
        .map(|tensor| tensor.name.as_str())
        .unwrap_or("<missing>");
    let output_name = summary
        .outputs
        .get(summary.selected_output)
        .map(|tensor| tensor.name.as_str())
        .unwrap_or("<missing>");
    let all_outputs = summary
        .outputs
        .iter()
        .map(|tensor| format!("{} {}", tensor.name, tensor.shape_label()))
        .collect::<Vec<_>>()
        .join(", ");
    debug_timing(&format!(
        "selected tensors: input={} output={} available_outputs=[{}]",
        input_name, output_name, all_outputs
    ));
}

fn log_selected_output_subgraph(onnx_model: &OnnxModel, summary: &ModelSummary) {
    if !ort_debug_enabled() {
        return;
    }

    let Some(selected_output_name) = summary
        .outputs
        .get(summary.selected_output)
        .map(|tensor| tensor.name.as_str())
    else {
        return;
    };

    let branch_operations = collect_selected_branch_operations(onnx_model, selected_output_name);
    for (branch_index, (_, operation)) in branch_operations.iter().enumerate() {
        debug_timing(&format!(
            "selected branch node[{branch_index:03}]: name=\"{}\" op={} inputs=[{}] outputs=[{}]",
            display_node_name(operation),
            operation.op_type,
            format_tensor_names(&operation.inputs),
            format_tensor_names(&operation.outputs)
        ));
    }

    log_placement_sensitive_branch_nodes(&branch_operations);

    let mut counts = branch_operations
        .iter()
        .fold(HashMap::<String, usize>::new(), |mut counts, (_, operation)| {
            *counts.entry(operation.op_type.clone()).or_insert(0) += 1;
            counts
        })
        .into_iter()
        .collect::<Vec<_>>();
    counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let summary_text = counts
        .iter()
        .map(|(op_type, count)| format!("{op_type}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    debug_timing(&format!(
        "selected output subgraph: output={} total_nodes={} unique_ops={} {}",
        selected_output_name,
        branch_operations.len(),
        counts.len(),
        summary_text
    ));
}

fn collect_selected_branch_operations<'a>(
    onnx_model: &'a OnnxModel,
    selected_output_name: &str,
) -> Vec<(usize, &'a OnnxOperation)> {
    let mut producers = HashMap::new();
    for (index, operation) in onnx_model.operations.iter().enumerate() {
        for output in &operation.outputs {
            if !output.is_empty() {
                producers.insert(output.as_str(), index);
            }
        }
    }

    let mut pending_tensors = vec![selected_output_name];
    let mut visited_tensors = HashSet::new();
    let mut visited_ops = HashSet::new();

    while let Some(tensor_name) = pending_tensors.pop() {
        if !visited_tensors.insert(tensor_name.to_string()) {
            continue;
        }
        let Some(&operation_index) = producers.get(tensor_name) else {
            continue;
        };
        let operation = &onnx_model.operations[operation_index];
        if visited_ops.insert(operation_index) {
            for input in &operation.inputs {
                if !input.is_empty() {
                    pending_tensors.push(input);
                }
            }
        }
    }

    let mut operation_indices = visited_ops.into_iter().collect::<Vec<_>>();
    operation_indices.sort_unstable();
    operation_indices
        .into_iter()
        .map(|index| (index, &onnx_model.operations[index]))
        .collect()
}

fn log_placement_sensitive_branch_nodes(branch_operations: &[(usize, &OnnxOperation)]) {
    let mut logged_header = false;
    for op_type in PLACEMENT_SENSITIVE_OPS {
        let matching = branch_operations
            .iter()
            .filter_map(|(_, operation)| (operation.op_type == *op_type).then_some(*operation))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        if !logged_header {
            debug_timing("possible placement-sensitive nodes:");
            logged_header = true;
        }
        debug_timing(&format!("  {}: {}", op_type, matching.len()));
        for operation in matching {
            debug_timing(&format!(
                "    name=\"{}\" op={} inputs=[{}] outputs=[{}]",
                display_node_name(operation),
                operation.op_type,
                format_tensor_names(&operation.inputs),
                format_tensor_names(&operation.outputs)
            ));
        }
    }
}

fn display_node_name(operation: &OnnxOperation) -> &str {
    if operation.name.is_empty() {
        "<unnamed>"
    } else {
        operation.name.as_str()
    }
}

fn format_tensor_names(values: &[String]) -> String {
    values
        .iter()
        .map(|value| if value.is_empty() { "<empty>" } else { value.as_str() })
        .collect::<Vec<_>>()
        .join(", ")
}

fn maybe_finalize_ort_profile(cached_session: &mut CachedSession) {
    let Some(requested_path) = cached_session.settings.profile_path.clone() else {
        return;
    };
    if cached_session.profile_finished {
        return;
    }

    match cached_session.session.end_profiling() {
        Ok(path) => {
            cached_session.profile_finished = true;
            let actual_path = if path.is_empty() {
                requested_path
            } else {
                PathBuf::from(path)
            };
            debug_timing(&format!("ORT profile written to {}", actual_path.display()));
            if let Err(error) = summarize_ort_profile(&actual_path) {
                debug_timing(&format!(
                    "failed to summarize ORT profile {}: {}",
                    actual_path.display(),
                    error
                ));
            }
        }
        Err(error) => debug_timing(&format!(
            "failed to end ORT profiling for {}: {}",
            requested_path.display(),
            error
        )),
    }
}

fn summarize_ort_profile(path: &Path) -> Result<(), String> {
    let profile_text = fs::read_to_string(path)
        .map_err(|error| format!("read failed: {error}"))?;
    let profile_json: Value =
        serde_json::from_str(&profile_text).map_err(|error| format!("parse failed: {error}"))?;
    let events = match &profile_json {
        Value::Array(events) => events.iter().collect::<Vec<_>>(),
        Value::Object(object) => object
            .get("traceEvents")
            .and_then(Value::as_array)
            .map(|events| events.iter().collect::<Vec<_>>())
            .unwrap_or_default(),
        _ => Vec::new(),
    };

    if events.is_empty() {
        return Err("profile did not contain any events".to_string());
    }

    let mut provider_counts = HashMap::<String, usize>::new();
    let mut cpu_nodes = HashSet::<(String, String)>::new();
    for event in events {
        let Some(event_object) = event.as_object() else {
            continue;
        };
        let args = event_object.get("args").and_then(Value::as_object);
        let provider = args
            .and_then(|args| args.get("provider"))
            .and_then(Value::as_str)
            .or_else(|| {
                args.and_then(|args| args.get("execution_provider"))
                    .and_then(Value::as_str)
            });
        let Some(provider) = provider else {
            continue;
        };
        *provider_counts.entry(provider.to_string()).or_insert(0) += 1;

        if provider == "CPUExecutionProvider" {
            let node_name = args
                .and_then(|args| args.get("node_name"))
                .and_then(Value::as_str)
                .or_else(|| event_object.get("name").and_then(Value::as_str))
                .unwrap_or("<unknown>");
            let op_name = args
                .and_then(|args| args.get("op_name"))
                .and_then(Value::as_str)
                .or_else(|| args.and_then(|args| args.get("op_type")).and_then(Value::as_str))
                .unwrap_or("<unknown>");
            cpu_nodes.insert((node_name.to_string(), op_name.to_string()));
        }
    }

    let mut provider_counts = provider_counts.into_iter().collect::<Vec<_>>();
    provider_counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    debug_timing("ORT provider summary:");
    for (provider, count) in provider_counts {
        debug_timing(&format!("  {}: {} events", provider, count));
    }

    let mut cpu_nodes = cpu_nodes.into_iter().collect::<Vec<_>>();
    cpu_nodes.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    if cpu_nodes.is_empty() {
        debug_timing("CPU-assigned nodes: none found in ORT profile");
    } else {
        debug_timing("CPU-assigned nodes:");
        for (node_name, op_name) in cpu_nodes {
            debug_timing(&format!("  node=\"{}\" op={}", node_name, op_name));
        }
    }

    Ok(())
}

fn preprocess_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    layout: Layout,
) -> Result<Array4<f32>, String> {
    if rgba.len() != (width as usize) * (height as usize) * 4 {
        return Err("tile byte size does not match RGBA dimensions".to_string());
    }
    let mut array = match layout {
        Layout::Nchw => Array4::<f32>::zeros((1, 3, height as usize, width as usize)),
        Layout::Nhwc => Array4::<f32>::zeros((1, height as usize, width as usize, 3)),
    };

    for y in 0..height as usize {
        for x in 0..width as usize {
            let offset = (y * width as usize + x) * 4;
            let r = rgba[offset] as f32 / 255.0;
            let g = rgba[offset + 1] as f32 / 255.0;
            let b = rgba[offset + 2] as f32 / 255.0;
            match layout {
                Layout::Nchw => {
                    array[[0, 0, y, x]] = r;
                    array[[0, 1, y, x]] = g;
                    array[[0, 2, y, x]] = b;
                }
                Layout::Nhwc => {
                    array[[0, y, x, 0]] = r;
                    array[[0, y, x, 1]] = g;
                    array[[0, y, x, 2]] = b;
                }
            }
        }
    }

    Ok(array)
}

fn preprocess_rgba_batch(
    inputs: &[BatchReconstructionInput<'_>],
    layout: Layout,
) -> Result<Array4<f32>, String> {
    let width = inputs[0].width as usize;
    let height = inputs[0].height as usize;
    let batch = inputs.len();
    let mut array = match layout {
        Layout::Nchw => Array4::<f32>::zeros((batch, 3, height, width)),
        Layout::Nhwc => Array4::<f32>::zeros((batch, height, width, 3)),
    };

    for (batch_index, input) in inputs.iter().enumerate() {
        if input.rgba.len() != width * height * 4 {
            return Err("tile byte size does not match RGBA dimensions".to_string());
        }
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                let r = input.rgba[offset] as f32 / 255.0;
                let g = input.rgba[offset + 1] as f32 / 255.0;
                let b = input.rgba[offset + 2] as f32 / 255.0;
                match layout {
                    Layout::Nchw => {
                        array[[batch_index, 0, y, x]] = r;
                        array[[batch_index, 1, y, x]] = g;
                        array[[batch_index, 2, y, x]] = b;
                    }
                    Layout::Nhwc => {
                        array[[batch_index, y, x, 0]] = r;
                        array[[batch_index, y, x, 1]] = g;
                        array[[batch_index, y, x, 2]] = b;
                    }
                }
            }
        }
    }

    Ok(array)
}

fn postprocess_reconstruction(
    output: ArrayViewD<'_, f32>,
    layout: Layout,
) -> Result<ReconstructionResult, String> {
    postprocess_reconstruction_batch(output, layout)?
        .into_iter()
        .next()
        .ok_or_else(|| "reconstruction batch output was empty".to_string())
}

fn postprocess_reconstruction_batch(
    output: ArrayViewD<'_, f32>,
    layout: Layout,
) -> Result<Vec<ReconstructionResult>, String> {
    let output = output
        .into_dimensionality::<Ix4>()
        .map_err(|_| "expected a 4D reconstruction tensor".to_string())?;

    let batch_size = output.shape()[0];
    let mut reconstructions = Vec::with_capacity(batch_size);
    for batch_index in 0..batch_size {
        let output = output.index_axis(Axis(0), batch_index);

        let (height, width, channels) = match layout {
            Layout::Nchw => (output.shape()[1], output.shape()[2], output.shape()[0]),
            Layout::Nhwc => (output.shape()[0], output.shape()[1], output.shape()[2]),
        };
        if channels < 3 {
            return Err("reconstruction output must have at least 3 channels".to_string());
        }

        let max_value = output
            .iter()
            .fold(f32::MIN, |current, value| current.max(*value));
        let scale = if max_value <= 1.5 { 255.0 } else { 1.0 };

        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = match layout {
                    Layout::Nchw => (output[[0, y, x]], output[[1, y, x]], output[[2, y, x]]),
                    Layout::Nhwc => (output[[y, x, 0]], output[[y, x, 1]], output[[y, x, 2]]),
                };
                let offset = (y * width + x) * 3;
                rgb[offset] = (r * scale).clamp(0.0, 255.0) as u8;
                rgb[offset + 1] = (g * scale).clamp(0.0, 255.0) as u8;
                rgb[offset + 2] = (b * scale).clamp(0.0, 255.0) as u8;
            }
        }

        reconstructions.push(ReconstructionResult { rgb });
    }

    Ok(reconstructions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::WsiFile;
    use std::path::PathBuf;

    fn tensor(name: &str, shape: &[i64]) -> TensorSummary {
        TensorSummary {
            name: name.to_string(),
            dtype: "Float32".to_string(),
            shape: shape.to_vec(),
            is_image_candidate: true,
        }
    }

    #[test]
    fn infers_nhwc_layout_from_last_channel() {
        let inputs = vec![tensor("input", &[-1, 256, 256, 3])];
        let outputs = vec![tensor("reconstruction", &[-1, 256, 256, 3])];
        let summary = infer_model_contract("model.onnx", &inputs, &outputs).unwrap();
        assert_eq!(summary.layout, Layout::Nhwc);
        assert_eq!(summary.tile_size, 256);
    }

    #[test]
    fn infers_nchw_layout_from_second_channel() {
        let inputs = vec![tensor("input", &[-1, 3, 128, 128])];
        let outputs = vec![tensor("reconstruction", &[-1, 3, 128, 128])];
        let summary = infer_model_contract("model.onnx", &inputs, &outputs).unwrap();
        assert_eq!(summary.layout, Layout::Nchw);
        assert_eq!(summary.selected_output, 0);
    }

    #[test]
    fn prefers_named_reconstruction_output_when_shape_is_ambiguous() {
        let inputs = vec![tensor("image", &[-1, 3, 64, 64])];
        let outputs = vec![
            tensor("latent", &[-1, 16]),
            tensor("decoded_reconstruction", &[-1, 3, 64, 64]),
        ];
        let summary = infer_model_contract("model.onnx", &inputs, &outputs).unwrap();
        assert_eq!(summary.selected_output, 1);
    }

    #[test]
    #[ignore = "manual fixture-backed runtime validation that requires ONNX runtime GPU support"]
    fn runs_reconstruction_with_fixture_model_and_slide_tile() {
        let model_path = fixture_path("HistoVAE_converted.onnx");
        println!("model path: {}", model_path.display());
        let slide_path = fixture_path("patient_198_node_0.tif");
        println!("slide path: {}", slide_path.display());
        println!("loading model and slide...");
        let model = load_model(model_path.to_str().unwrap(), true).unwrap();
        println!("model loaded, opening WSI file...");
        let slide = WsiFile::open(&slide_path).unwrap();
        println!("model summary: {:#?}", model.summary);
        let tile_size = model
            .summary
            .tile_size
            .min(slide.properties().width.min(slide.properties().height) as u32)
            .max(1);
        let x = ((slide.properties().width - tile_size as u64) / 2) as i64;
        let y = ((slide.properties().height - tile_size as u64) / 2) as i64;
        println!(
            "reading tile at x={}, y={} with size {}x{}",
            x, y, tile_size, tile_size
        );
        let rgba = slide.read_region(x, y, 0, tile_size, tile_size).unwrap();
        println!("tile read, running reconstruction...");
        let reconstruction = run_reconstruction(&model, &rgba, tile_size, tile_size).unwrap();
        println!("reconstruction completed, validating output...");
        assert_eq!(rgba.len(), tile_size as usize * tile_size as usize * 4);
        assert_eq!(
            reconstruction.rgb.len(),
            tile_size as usize * tile_size as usize * 3
        );
        assert!(reconstruction.rgb.iter().any(|value| *value != 0));
    }

    #[test]
    #[ignore = "manual fixture-backed runtime validation that requires ONNX runtime GPU support"]
    fn fixture_model_gpu_session_build_reproduces_cpu_ep_assignment_failure() {
        let model_path = fixture_path("HistoVAE_converted.onnx");
        let model = load_model(model_path.to_str().unwrap(), true).unwrap();

        let error = build_inference_session(
            model.summary.path.as_str(),
            true,
            crate::state::clamp_analysis_threads(32),
            model.summary.layout,
        )
        .unwrap_err();

        assert!(
            error.contains("assigned to the default CPU EP"),
            "unexpected gpu session build error: {error}"
        );
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(name)
    }
}
