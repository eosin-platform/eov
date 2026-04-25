use crate::state::{clamp_analysis_threads, host_api, plugin_state};
use ndarray::{Array4, ArrayViewD, Axis, Ix4};
use onnx_extractor::{DataType as OnnxDataType, OnnxModel, OnnxTensor};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use std::fmt;
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const GPU_SESSION_BUILD_TIMEOUT: Duration = Duration::from_secs(20);
const CPU_SESSION_BUILD_TIMEOUT: Duration = Duration::from_secs(5);

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

pub struct CachedSession {
    session: Session,
    prefer_gpu: bool,
    analysis_threads: usize,
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
    build_session_with_watchdog(path, prefer_gpu, analysis_threads, layout)
}

fn build_inference_session_inner(
    path: &str,
    prefer_gpu: bool,
    analysis_threads: usize,
    layout: Layout,
) -> Result<Session, String> {
    let mut builder = Session::builder()
        .map_err(|error| error.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| error.to_string())?
        .with_parallel_execution(analysis_threads > 1)
        .map_err(|error| error.to_string())?
        .with_intra_threads(analysis_threads)
        .map_err(|error| error.to_string())?;
    if prefer_gpu {
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

    let timeout = if prefer_gpu {
        GPU_SESSION_BUILD_TIMEOUT
    } else {
        CPU_SESSION_BUILD_TIMEOUT
    };
    commit_session_with_timeout(
        builder,
        path,
        timeout,
        if prefer_gpu { "gpu" } else { "cpu" },
    )
}

fn build_session_with_watchdog(
    path: &str,
    prefer_gpu: bool,
    analysis_threads: usize,
    layout: Layout,
) -> Result<Session, String> {
    let timeout = if prefer_gpu {
        GPU_SESSION_BUILD_TIMEOUT
    } else {
        CPU_SESSION_BUILD_TIMEOUT
    };
    let label = if prefer_gpu { "gpu" } else { "cpu" };
    let (sender, receiver) = mpsc::channel();
    let path = path.to_string();

    thread::spawn(move || {
        let result = build_inference_session_inner(&path, prefer_gpu, analysis_threads, layout);
        let _ = sender.send(result);
    });

    debug_timing(&format!(
        "waiting up to {:?} for {label} session build thread",
        timeout
    ));
    match receiver.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "{label} session build timed out after {:?}",
            timeout
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(format!("{label} session build thread disconnected"))
        }
    }
}

fn commit_session_with_timeout(
    mut builder: ort::session::builder::SessionBuilder,
    path: &str,
    timeout: Duration,
    label: &str,
) -> Result<Session, String> {
    let canceler = builder.canceler();
    let start = Instant::now();
    let timeout_label = label.to_string();
    thread::spawn(move || {
        thread::sleep(timeout);
        debug_timing(&format!(
            "{timeout_label} session build exceeded {:?}; requesting cancellation",
            timeout
        ));
        let _ = canceler.cancel();
    });

    debug_timing(&format!("starting {label} session build"));
    let result = builder
        .commit_from_file(path)
        .map_err(|error| error.to_string());
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

    let run_started = Instant::now();
    let outputs = session
        .session
        .run(ort::inputs![
            TensorRef::from_array_view(input.view()).map_err(|error| error.to_string())?
        ])
        .map_err(|error| error.to_string())?;
    debug_timing(&format!(
        "session.run completed in {:?}",
        run_started.elapsed()
    ));

    let output_value = &outputs[model.summary.selected_output];
    let output = output_value
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?;
    postprocess_reconstruction(output, model.summary.layout)
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

    let run_started = Instant::now();
    let outputs = session
        .session
        .run(ort::inputs![
            TensorRef::from_array_view(input.view()).map_err(|error| error.to_string())?
        ])
        .map_err(|error| error.to_string())?;
    debug_timing(&format!(
        "batch session.run completed in {:?}",
        run_started.elapsed()
    ));

    let output_value = &outputs[model.summary.selected_output];
    let output = output_value
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?;
    postprocess_reconstruction_batch(output, model.summary.layout)
}

fn ensure_inference_session(
    model: &LoadedModel,
) -> Result<std::sync::MutexGuard<'_, Option<CachedSession>>, String> {
    let (prefer_gpu, analysis_threads) = desired_execution_settings(model);
    {
        let session = model
            .session
            .lock()
            .map_err(|_| "model session lock is poisoned".to_string())?;
        if session.as_ref().is_some_and(|session| {
            session.prefer_gpu == prefer_gpu && session.analysis_threads == analysis_threads
        }) {
            debug_timing("reusing cached inference session");
            return Ok(session);
        }
    }

    debug_timing(&format!(
        "creating new {} inference session with {} thread(s)",
        if prefer_gpu { "gpu" } else { "cpu" },
        analysis_threads
    ));
    let session = match build_inference_session(
        &model.summary.path,
        prefer_gpu,
        analysis_threads,
        model.summary.layout,
    ) {
        Ok(session) => session,
        Err(error) if prefer_gpu => {
            debug_timing(&format!(
                "gpu inference session unavailable; falling back to cpu: {error}"
            ));
            build_inference_session(
                &model.summary.path,
                false,
                analysis_threads,
                model.summary.layout,
            )?
        }
        Err(error) => return Err(error),
    };

    let mut guard = model
        .session
        .lock()
        .map_err(|_| "model session lock is poisoned".to_string())?;
    *guard = Some(CachedSession {
        session,
        prefer_gpu,
        analysis_threads,
    });
    Ok(guard)
}

fn desired_execution_settings(model: &LoadedModel) -> (bool, usize) {
    let prefer_gpu = host_api()
        .map(|host| (host.get_snapshot)(host.context))
        .map(|snapshot| snapshot.render_backend.to_ascii_lowercase() == "gpu")
        .unwrap_or(model.prefer_gpu);
    let analysis_threads = model.session_threads_override.unwrap_or_else(|| {
        clamp_analysis_threads(plugin_state().lock().unwrap().config.analysis_threads)
    });
    (prefer_gpu, analysis_threads)
}

fn debug_timing(message: &str) {
    if std::env::var_os("EOVAE_DEBUG_TIMING").is_some() {
        eprintln!("[eovae] {message}");
    }
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

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(name)
    }
}
