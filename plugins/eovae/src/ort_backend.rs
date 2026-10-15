//! ONNX Runtime engines.
//!
//! - ONNX Runtime is opened at run time (`ort`'s `load-dynamic`), searched
//!   for portably: `EOVAE_ORT_LIB`, `ORT_DYLIB_PATH`, the configured path,
//!   the copy bundled in the plugin package, then the system library search
//!   path. No `/proc/self/maps`, no Unix-only loader calls (VAE-14).
//! - CUDA is optional and located lazily, only when a CUDA session is
//!   requested: nothing GPU-related is loaded for CPU inference. With
//!   `Auto`, any CUDA problem (no CUDA in the runtime, missing libraries,
//!   registration or warm-up failure) falls back to the CPU and says why
//!   (VAE-8).
//! - Sessions are built on worker threads with a watchdog, validated with a
//!   real warm-up run and cached by the caller (VAE-11, VAE-15). Inputs are
//!   fed by name; float16 models are supported.

use crate::engine::{
    Backend, Device, DevicePreference, EngineError, EngineInfo, EngineOutput, EngineRequest,
    InferenceEngine,
};
use crate::model::{ModelInfo, TensorType};
use ort::ep::{self, ExecutionProvider};
use ort::session::{RunOptions, Session, builder::GraphOptimizationLevel};
use ort::value::{DynValue, TensorRef};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Environment variable naming the ONNX Runtime library to use.
pub const ORT_LIB_ENV: &str = "EOVAE_ORT_LIB";
/// Directories holding the CUDA runtime and cuDNN libraries (optional).
pub const CUDA_LIB_ENV: &str = "EOVAE_CUDA_LIB_DIR";
pub const CUDNN_LIB_ENV: &str = "EOVAE_CUDNN_LIB_DIR";

const CPU_BUILD_TIMEOUT: Duration = Duration::from_secs(90);
const CUDA_BUILD_TIMEOUT: Duration = Duration::from_secs(180);

/// The platform's ONNX Runtime library names, preferred first.
fn library_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so", "libonnxruntime.so.1"]
    }
}

/// Whether `name` is a versioned ONNX Runtime library
/// (`libonnxruntime.so.1.22.0`, `libonnxruntime.1.22.0.dylib`).
fn is_versioned_library(name: &str) -> bool {
    if cfg!(target_os = "windows") {
        false
    } else if cfg!(target_os = "macos") {
        name.starts_with("libonnxruntime.")
            && name.ends_with(".dylib")
            && !name.contains("providers")
    } else {
        name.starts_with("libonnxruntime.so.")
    }
}

/// ONNX Runtime libraries inside `dir`.
fn libraries_in(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = library_names()
        .iter()
        .map(|name| dir.join(name))
        .filter(|path| path.is_file())
        .collect();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut versioned: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(is_versioned_library)
            })
            .collect();
        versioned.sort();
        found.extend(versioned.into_iter().rev());
    }
    found
}

/// Every place ONNX Runtime is looked for, in order.
pub fn runtime_candidates(plugin_root: &Path, configured: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in [ORT_LIB_ENV, "ORT_DYLIB_PATH"] {
        if let Some(value) = std::env::var_os(variable).filter(|v| !v.is_empty()) {
            candidates.push(PathBuf::from(value));
        }
    }
    if let Some(path) = configured {
        if path.is_dir() {
            candidates.extend(libraries_in(path));
        } else {
            candidates.push(path.to_path_buf());
        }
    }
    for dir in [plugin_root.join("onnxruntime"), plugin_root.to_path_buf()] {
        candidates.extend(libraries_in(&dir));
    }
    // Bare names: the system's library search path.
    candidates.extend(library_names().iter().map(PathBuf::from));
    candidates.dedup();
    candidates
}

/// The loaded runtime.
#[derive(Clone, Debug)]
struct Runtime {
    path: PathBuf,
    /// "1.28.3", from the runtime's build information.
    version: Option<String>,
}

/// The oldest ONNX Runtime eovae runs with (its minor version; 1.x).
const MIN_RUNTIME_MINOR: u32 = 22;

/// Where `ort` looks for `path`: a relative name next to the executable
/// first, as `ort` does.
fn resolve_like_ort(path: &Path) -> PathBuf {
    match std::env::current_exe() {
        Ok(exe) if path.is_relative() => exe
            .parent()
            .map(|dir| dir.join(path))
            .filter(|p| p.exists())
            .unwrap_or_else(|| path.to_path_buf()),
        _ => path.to_path_buf(),
    }
}

/// Opens a candidate without `ort` and checks that it is ONNX Runtime and
/// new enough: the resolved path and its version. `ort` 2.0.0-rc.13 can
/// load only once per process and keeps a failed load as if it had
/// succeeded (with an empty library), so it is only ever given a library
/// that passed this check.
fn check_runtime(path: &Path) -> Result<(PathBuf, Option<String>), String> {
    let path = resolve_like_ort(path);
    // SAFETY: opening a library runs its initialisers, as loading it would;
    // `OrtGetApiBase` and `GetVersionString` are ONNX Runtime's stable C API,
    // take no arguments, and the version string is static and copied before
    // the handle is released.
    let version = unsafe {
        let library =
            libloading::Library::new(&path).map_err(|e| format!("it could not be opened: {e}"))?;
        let get_api_base: libloading::Symbol<
            unsafe extern "system" fn() -> *const ort::sys::OrtApiBase,
        > = library
            .get(b"OrtGetApiBase")
            .map_err(|_| "it is not ONNX Runtime (it has no OrtGetApiBase)".to_string())?;
        let base = get_api_base();
        if base.is_null() {
            return Err("it is not ONNX Runtime (OrtGetApiBase returned nothing)".to_string());
        }
        let version = ((*base).GetVersionString)();
        (!version.is_null()).then(|| {
            std::ffi::CStr::from_ptr(version)
                .to_string_lossy()
                .into_owned()
        })
    };
    if let Some(version) = &version
        && runtime_too_old(version)
    {
        return Err(format!(
            "it is ONNX Runtime {version}, older than 1.{MIN_RUNTIME_MINOR}"
        ));
    }
    Ok((path, version))
}

fn runtime_too_old(version: &str) -> bool {
    let mut parts = version.split('.').map(|part| part.parse::<u32>().ok());
    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(1), Some(minor)) => minor < MIN_RUNTIME_MINOR,
        (Some(0), _) => true,
        _ => false,
    }
}

/// The runtime's own version (`OrtGetApiBase()->GetVersionString()`), read
/// from the library ort has just loaded from `path`.
fn library_version(path: &Path) -> Option<String> {
    // SAFETY: `path` names the ONNX Runtime library that is already loaded,
    // so this opens the same library again; `OrtGetApiBase` and
    // `GetVersionString` are its stable C API, and the version string is
    // static and copied before the handle is released.
    unsafe {
        let library = libloading::Library::new(path).ok()?;
        let get_api_base: libloading::Symbol<
            unsafe extern "system" fn() -> *const ort::sys::OrtApiBase,
        > = library.get(b"OrtGetApiBase").ok()?;
        let base = get_api_base();
        if base.is_null() {
            return None;
        }
        let version = ((*base).GetVersionString)();
        if version.is_null() {
            return None;
        }
        Some(
            std::ffi::CStr::from_ptr(version)
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// The version in ONNX Runtime's build information
/// (`ORT Build Info: git-branch=rel-1.28.3, git-commit-id=…`): the release
/// branch's number, or the branch and commit for other builds.
fn version_of(build_info: &str) -> Option<String> {
    let field = |name: &str| {
        build_info
            .split(',')
            .filter_map(|part| part.split_once('='))
            .find(|(key, _)| key.trim().ends_with(name))
            .map(|(_, value)| value.trim().to_string())
    };
    let branch = field("git-branch")?;
    match branch.strip_prefix("rel-") {
        Some(version) => Some(version.to_string()),
        None => Some(match field("git-commit-id") {
            Some(commit) => format!("{branch} ({commit})"),
            None => branch,
        }),
    }
}

pub struct OrtBackend {
    plugin_root: PathBuf,
    runtime: Mutex<Option<Runtime>>,
    /// Why `ort` itself failed to load a checked library: it cannot try
    /// again in this process.
    spent: Mutex<Option<String>>,
    cuda_notes: Mutex<Option<Vec<String>>>,
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown error".to_string())
}

impl OrtBackend {
    /// Opens ONNX Runtime once per process.
    fn ensure_runtime(&self, configured: Option<&Path>) -> Result<Runtime, String> {
        let mut slot = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(runtime) = slot.as_ref() {
            return Ok(runtime.clone());
        }
        let mut spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(reason) = spent.as_ref() {
            return Err(reason.clone());
        }
        let mut attempts = Vec::new();
        for candidate in runtime_candidates(&self.plugin_root, configured) {
            let (path, checked_version) =
                match catch_unwind(AssertUnwindSafe(|| check_runtime(&candidate))) {
                    Ok(Ok(checked)) => checked,
                    Ok(Err(reason)) => {
                        attempts.push(format!("{}: {reason}", candidate.display()));
                        continue;
                    }
                    Err(payload) => {
                        attempts.push(format!(
                            "{}: opening it crashed ({})",
                            candidate.display(),
                            panic_text(payload.as_ref())
                        ));
                        continue;
                    }
                };
            // The one try `ort` gets in this process.
            let result = catch_unwind(AssertUnwindSafe(|| ort::init_from(&path)));
            let failure = match result {
                Ok(Ok(builder)) => {
                    builder.with_name("eovae").commit();
                    let version = checked_version
                        .or_else(|| library_version(&path))
                        .or_else(|| catch_unwind(ort::info).ok().and_then(version_of));
                    let runtime = Runtime {
                        path: candidate,
                        version,
                    };
                    *slot = Some(runtime.clone());
                    return Ok(runtime);
                }
                Ok(Err(error)) => error.to_string(),
                Err(payload) => format!("loading it crashed ({})", panic_text(payload.as_ref())),
            };
            let reason = format!(
                "ONNX Runtime at {} could not be loaded: {failure}. Restart eov to try again.",
                path.display()
            );
            *spent = Some(reason.clone());
            return Err(reason);
        }
        Err(format!(
            "ONNX Runtime (1.22 or newer) was not found. Install it, set {ORT_LIB_ENV} to the library, or choose it under Model → Runtime. Tried: {}",
            attempts
                .iter()
                .rev()
                .take(4)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ))
    }

    /// Preloads CUDA and cuDNN from configured directories, once. Missing
    /// directories are fine: the system loader may still find the libraries.
    fn prepare_cuda(&self) -> Vec<String> {
        let mut notes = self.cuda_notes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(notes) = notes.as_ref() {
            return notes.clone();
        }
        let mut found = Vec::new();
        let mut dirs: Vec<(PathBuf, &[&str])> = Vec::new();
        if let Some(dir) = std::env::var_os(CUDA_LIB_ENV) {
            dirs.push((PathBuf::from(dir), ep::cuda::CUDA_DYLIBS));
        } else if let Some(home) =
            std::env::var_os("CUDA_PATH").or_else(|| std::env::var_os("CUDA_HOME"))
        {
            let home = PathBuf::from(home);
            let sub = if cfg!(target_os = "windows") {
                "bin"
            } else {
                "lib64"
            };
            dirs.push((home.join(sub), ep::cuda::CUDA_DYLIBS));
        }
        if let Some(dir) = std::env::var_os(CUDNN_LIB_ENV) {
            dirs.push((PathBuf::from(dir), ep::cuda::CUDNN_DYLIBS));
        }
        for (dir, names) in dirs {
            for name in names {
                let path = dir.join(name);
                if path.is_file() {
                    match ort::util::preload_dylib(&path) {
                        Ok(()) => found.push(format!("loaded {}", path.display())),
                        Err(error) => found.push(format!("{}: {error}", path.display())),
                    }
                }
            }
        }
        *notes = Some(found.clone());
        found
    }

    fn build_session(
        &self,
        model: &ModelInfo,
        device: Device,
        threads: usize,
        tf32: bool,
        stop: &[Arc<AtomicBool>],
    ) -> Result<Session, String> {
        let error = |e: ort::Error<_>| e.to_string();
        let mut builder = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(error)?
            .with_intra_threads(threads.max(1))
            .map_err(error)?;
        if device == Device::Cuda {
            let cuda = ep::CUDA::default()
                .with_tf32(tf32)
                // Exhaustive search took 28 s on the first run (VAE-13).
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
                .with_prefer_nhwc(model.layout == crate::contract::Layout::Nhwc);
            builder = builder
                .with_execution_providers([cuda.build().error_on_failure()])
                .map_err(error)?;
        }
        let canceler = builder.canceler();
        let timeout = if device == Device::Cuda {
            CUDA_BUILD_TIMEOUT
        } else {
            CPU_BUILD_TIMEOUT
        };
        // The build can be cancelled: on a timeout, or when the user or eov
        // gives up (shutdown, another model).
        let (done, wait) = mpsc::channel::<()>();
        let flags: Vec<Arc<AtomicBool>> = stop.to_vec();
        let started = std::time::Instant::now();
        let watchdog = std::thread::spawn(move || {
            loop {
                match wait.recv_timeout(Duration::from_millis(50)) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    _ => return,
                }
                if started.elapsed() >= timeout || any_set(&flags) {
                    let _ = canceler.cancel();
                    return;
                }
            }
        });
        let result = builder
            .commit_from_file(&model.path)
            .map_err(|e| e.to_string());
        let _ = done.send(());
        let _ = watchdog.join();
        result.map_err(|e| {
            if any_set(stop) {
                "cancelled".to_string()
            } else if e.to_ascii_lowercase().contains("cancel") {
                format!(
                    "building the session took longer than {} s",
                    timeout.as_secs()
                )
            } else {
                e
            }
        })
    }

    fn engine_for(
        &self,
        request: &EngineRequest,
        device: Device,
        status: &dyn Fn(&str),
        stop: &[Arc<AtomicBool>],
    ) -> Result<OrtEngine, String> {
        let model = &request.model;
        status(match device {
            Device::Cpu => "Building CPU session…",
            Device::Cuda => "Building CUDA session…",
        });
        let session = self.build_session(model, device, request.threads, request.tf32, stop)?;
        if !session
            .inputs()
            .iter()
            .any(|outlet| outlet.name() == model.input.name)
        {
            return Err(format!(
                "ONNX Runtime does not see input '{}'",
                model.input.name
            ));
        }
        let output = model
            .output(&request.output)
            .ok_or_else(|| format!("unknown output '{}'", request.output))?
            .clone();
        if !session.outputs().iter().any(|o| o.name() == output.name) {
            return Err(format!(
                "ONNX Runtime does not see output '{}'",
                output.name
            ));
        }
        let latent = request.latent.as_ref().and_then(|name| {
            let latent = model.latent.as_ref().filter(|l| &l.name == name)?;
            session
                .outputs()
                .iter()
                .any(|o| o.name() == latent.name)
                .then(|| latent.clone())
        });
        let tile = model.tile_size as usize;
        let item_dims = match model.layout {
            crate::contract::Layout::Nchw => [3, tile, tile],
            crate::contract::Layout::Nhwc => [tile, tile, 3],
        };
        let precision = match (device, model.input_elem, request.tf32) {
            (_, TensorType::F16, _) => "fp16",
            (Device::Cuda, TensorType::F32, true) => "tf32",
            _ => "fp32",
        };
        let label = match device {
            Device::Cpu => format!("CPU · {} threads", request.threads.max(1)),
            Device::Cuda if precision == "tf32" => "CUDA · TF32".to_string(),
            Device::Cuda => "CUDA".to_string(),
        };
        let mut engine = OrtEngine {
            session,
            info: EngineInfo {
                device,
                precision: precision.to_string(),
                label,
                fallback: None,
                fixed_batch: model.fixed_batch,
                item_len: model.item_len(),
                latent_dim: latent.as_ref().map(|l| l.dim),
            },
            input_name: model.input.name.clone(),
            input_elem: model.input_elem,
            item_dims,
            output_name: output.name,
            output_elem: output.elem,
            latent: latent.map(|l| (l.name, l.elem)),
        };
        // A real run validates shapes and types and warms up the device
        // (cuDNN algorithm selection happens here, not in the first job).
        status(match device {
            Device::Cpu => "Checking the model…",
            Device::Cuda => "Warming up CUDA…",
        });
        let batch = model.fixed_batch.unwrap_or(request.batch_size.clamp(1, 4));
        let probe = vec![0.25f32; batch * model.item_len()];
        engine
            .run_watched(&probe, batch, stop)
            .map_err(|e| format!("test run failed: {e}"))
            .and_then(|output| {
                if output.recon.len() == batch * model.item_len() {
                    Ok(())
                } else {
                    Err(format!(
                        "the model returned {} values for {batch} tiles of {} values",
                        output.recon.len(),
                        model.item_len()
                    ))
                }
            })?;
        Ok(engine)
    }
}

impl Backend for OrtBackend {
    fn new(plugin_root: &Path) -> Self {
        Self {
            plugin_root: plugin_root.to_path_buf(),
            runtime: Mutex::new(None),
            spent: Mutex::new(None),
            cuda_notes: Mutex::new(None),
        }
    }

    fn create(
        &self,
        request: &EngineRequest,
        status: &dyn Fn(&str),
        stop: &[Arc<AtomicBool>],
    ) -> Result<Box<dyn InferenceEngine>, String> {
        status("Loading ONNX Runtime…");
        self.ensure_runtime(request.runtime_path.as_deref())?;
        let wants_cuda = request.device != DevicePreference::Cpu;
        let mut fallback = None;
        if wants_cuda {
            let cuda_problem =
                match catch_unwind(AssertUnwindSafe(|| ep::CUDA::default().is_available())) {
                    Ok(Ok(true)) => None,
                    Ok(Ok(false)) => Some(
                        "this ONNX Runtime has no CUDA support (use the GPU build; see the README)"
                            .to_string(),
                    ),
                    Ok(Err(error)) => Some(error.to_string()),
                    Err(payload) => Some(panic_text(payload.as_ref())),
                };
            let attempt = match cuda_problem {
                Some(problem) => Err(problem),
                None => {
                    let notes = self.prepare_cuda();
                    catch_unwind(AssertUnwindSafe(|| {
                        self.engine_for(request, Device::Cuda, status, stop)
                    }))
                    .unwrap_or_else(|payload| Err(panic_text(payload.as_ref())))
                    .map_err(|error| {
                        if notes.is_empty() {
                            error
                        } else {
                            format!("{error} ({})", notes.join(", "))
                        }
                    })
                }
            };
            match attempt {
                Ok(engine) => return Ok(Box::new(engine)),
                Err(error) if request.device == DevicePreference::Cuda => {
                    return Err(format!("CUDA is unavailable: {error}"));
                }
                Err(error) if any_set(stop) => return Err(error),
                Err(error) => fallback = Some(error),
            }
        }
        let mut engine = catch_unwind(AssertUnwindSafe(|| {
            self.engine_for(request, Device::Cpu, status, stop)
        }))
        .unwrap_or_else(|payload| Err(panic_text(payload.as_ref())))?;
        engine.info.fallback = fallback;
        Ok(Box::new(engine))
    }

    fn runtime_label(&self) -> String {
        match self
            .runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            Some(runtime) => format!("ONNX Runtime: {}", runtime.path.display()),
            None => "ONNX Runtime is loaded with the first model.".to_string(),
        }
    }

    fn runtime_version(&self) -> Option<String> {
        self.runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|runtime| runtime.version.clone())
    }
}

fn any_set(flags: &[Arc<AtomicBool>]) -> bool {
    flags.iter().any(|flag| flag.load(Ordering::SeqCst))
}

/// A session on one device.
struct OrtEngine {
    session: Session,
    info: EngineInfo,
    input_name: String,
    input_elem: TensorType,
    item_dims: [usize; 3],
    output_name: String,
    output_elem: TensorType,
    latent: Option<(String, TensorType)>,
}

/// Terminates a run when any stop flag is set.
struct Watcher {
    done: mpsc::Sender<()>,
    handle: std::thread::JoinHandle<()>,
}

impl Watcher {
    fn start(stop: Vec<Arc<AtomicBool>>, options: Arc<RunOptions>) -> Self {
        let (done, wait) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            loop {
                if any_set(&stop) {
                    let _ = options.terminate();
                    return;
                }
                match wait.recv_timeout(Duration::from_millis(20)) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    _ => return,
                }
            }
        });
        Self { done, handle }
    }

    fn finish(self) {
        let _ = self.done.send(());
        let _ = self.handle.join();
    }
}

fn extract(value: &DynValue, elem: TensorType) -> Result<Vec<f32>, String> {
    match elem {
        TensorType::F32 => value
            .try_extract_tensor::<f32>()
            .map(|(_, data)| data.to_vec())
            .map_err(|e| e.to_string()),
        TensorType::F16 => value
            .try_extract_tensor::<half::f16>()
            .map(|(_, data)| data.iter().map(|v| v.to_f32()).collect())
            .map_err(|e| e.to_string()),
    }
}

impl InferenceEngine for OrtEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn run(
        &mut self,
        input: &[f32],
        batch: usize,
        stop: &Arc<AtomicBool>,
    ) -> Result<EngineOutput, EngineError> {
        self.run_watched(input, batch, std::slice::from_ref(stop))
    }
}

impl OrtEngine {
    fn run_watched(
        &mut self,
        input: &[f32],
        batch: usize,
        stop: &[Arc<AtomicBool>],
    ) -> Result<EngineOutput, EngineError> {
        if any_set(stop) {
            return Err(EngineError::Stopped);
        }
        let failed = |e: String| {
            if any_set(stop) {
                EngineError::Stopped
            } else {
                EngineError::Failed(e)
            }
        };
        let Self {
            session,
            input_name,
            input_elem,
            item_dims,
            output_name,
            output_elem,
            latent,
            ..
        } = self;
        let shape = [
            batch as i64,
            item_dims[0] as i64,
            item_dims[1] as i64,
            item_dims[2] as i64,
        ];
        let options = Arc::new(RunOptions::new().map_err(|e| failed(e.to_string()))?);
        let watcher = Watcher::start(stop.to_vec(), Arc::clone(&options));
        let collect_outputs = |outputs: ort::session::SessionOutputs<'_>| {
            collect(&outputs, output_name, *output_elem, latent.as_ref())
        };
        let result = match input_elem {
            TensorType::F32 => TensorRef::from_array_view((shape, input))
                .map_err(|e| e.to_string())
                .and_then(|tensor| {
                    session
                        .run_with_options(ort::inputs![input_name.as_str() => tensor], &options)
                        .map_err(|e| e.to_string())
                        .and_then(collect_outputs)
                }),
            TensorType::F16 => {
                let converted: Vec<half::f16> =
                    input.iter().map(|v| half::f16::from_f32(*v)).collect();
                TensorRef::from_array_view((shape, converted.as_slice()))
                    .map_err(|e| e.to_string())
                    .and_then(|tensor| {
                        session
                            .run_with_options(ort::inputs![input_name.as_str() => tensor], &options)
                            .map_err(|e| e.to_string())
                            .and_then(collect_outputs)
                    })
            }
        };
        watcher.finish();
        result.map_err(failed)
    }
}

fn collect(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    output_elem: TensorType,
    latent: Option<&(String, TensorType)>,
) -> Result<EngineOutput, String> {
    let recon = outputs
        .get(output_name)
        .ok_or_else(|| format!("the model produced no '{output_name}'"))?;
    let recon = extract(recon, output_elem)?;
    let latent = latent.and_then(|(name, elem)| {
        outputs
            .get(name)
            .and_then(|value| extract(value, *elem).ok())
    });
    Ok(EngineOutput { recon, latent })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_information_names_the_version() {
        assert_eq!(
            version_of(
                "ORT Build Info: git-branch=rel-1.19.0, git-commit-id=26250ae, build type=Release"
            )
            .as_deref(),
            Some("1.19.0")
        );
        assert_eq!(
            version_of(
                "ORT Build Info: git-branch=HEAD, git-commit-id=0d68ff6, build type=Release"
            )
            .as_deref(),
            Some("HEAD (0d68ff6)")
        );
        assert_eq!(version_of("something else"), None);
        // A file that is not ONNX Runtime has no version.
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join(library_names()[0]);
        std::fs::write(&fake, b"not a library").unwrap();
        assert_eq!(library_version(&fake), None);
    }

    #[test]
    fn runtimes_are_checked_before_ort_sees_them() {
        let dir = tempfile::tempdir().unwrap();
        let not_a_library = dir.path().join("libonnxruntime.so");
        std::fs::write(&not_a_library, b"not a library").unwrap();
        let reason = check_runtime(&not_a_library).unwrap_err();
        assert!(reason.contains("could not be opened"), "{reason}");
        assert!(check_runtime(&dir.path().join("missing.so")).is_err());
        assert!(runtime_too_old("1.21.1"));
        assert!(!runtime_too_old("1.22.0"));
        assert!(!runtime_too_old("1.28.3"));
        assert!(
            !runtime_too_old("2.0.0"),
            "a later major version is not refused"
        );
        assert!(!runtime_too_old("HEAD"), "an unknown version is tried");
    }

    #[test]
    fn candidates_prefer_explicit_paths_then_the_package_then_the_system() {
        let dir = tempfile::tempdir().unwrap();
        let bundled = dir.path().join("onnxruntime");
        std::fs::create_dir_all(&bundled).unwrap();
        let name = library_names()[0];
        std::fs::write(bundled.join(name), b"").unwrap();
        let configured = dir.path().join("custom").join(name);
        let candidates = runtime_candidates(dir.path(), Some(&configured));
        let configured_at = candidates.iter().position(|c| *c == configured).unwrap();
        let bundled_at = candidates
            .iter()
            .position(|c| *c == bundled.join(name))
            .unwrap();
        let system_at = candidates
            .iter()
            .position(|c| c.as_path() == Path::new(name))
            .unwrap();
        assert!(configured_at < bundled_at && bundled_at < system_at);
    }

    /// Compares the full pipeline with Python onnxruntime on real tiles.
    /// Needs `EOVAE_ORT_LIB` and `EOVAE_REFERENCE_TILES` (a folder written by
    /// the reference script described in the README), so it is opt-in.
    #[test]
    #[ignore = "needs ONNX Runtime and reference tiles"]
    fn matches_the_python_reference() {
        use crate::contract::{is_background, prepare, score_output, write_input};
        let Some(dir) = std::env::var_os("EOVAE_REFERENCE_TILES").map(PathBuf::from) else {
            return;
        };
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("HistoVAE/pretrained/HistoVae_reconstruct.onnx");
        let model = Arc::new(crate::model::inspect(&model_path, &AtomicBool::new(false)).unwrap());
        let reference: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
        let backend = OrtBackend::new(Path::new("."));
        let request = EngineRequest {
            output: model.default_output_name().to_string(),
            latent: None,
            model: Arc::clone(&model),
            device: DevicePreference::Cpu,
            threads: 8,
            tf32: false,
            batch_size: 32,
            runtime_path: None,
        };
        let mut engine = backend.create(&request, &|_| {}, &[]).unwrap();
        let contract = model.contract.clone();
        let mut input = vec![0f32; 32 * model.item_len()];
        let mut tiles = Vec::new();
        for (i, slot) in input.chunks_exact_mut(model.item_len()).enumerate() {
            let rgba = std::fs::read(dir.join(format!("tile_{i:02}.rgba"))).unwrap();
            let tile = prepare(&rgba, 256, (256, 256), &contract).unwrap();
            assert!(!is_background(&tile), "tile {i} is tissue");
            write_input(&tile, &contract, model.layout, slot);
            tiles.push(tile);
        }
        let output = engine
            .run(&input, 32, &Arc::new(AtomicBool::new(false)))
            .unwrap();
        let mut worst = 0f64;
        for (i, tile) in tiles.iter().enumerate() {
            let scored = score_output(
                tile,
                &output.recon[i * model.item_len()..(i + 1) * model.item_len()],
                &contract,
                model.layout,
            );
            let expected = reference[i]["mae"].as_f64().unwrap();
            worst = worst.max((f64::from(scored.mae) - expected).abs() / expected);
        }
        assert!(worst < 1e-3, "worst relative MAE difference {worst}");
        println!(
            "{}: worst relative difference {worst:.2e}",
            backend.runtime_label()
        );
    }

    #[test]
    fn versioned_libraries_are_recognised() {
        if cfg!(target_os = "linux") {
            assert!(is_versioned_library("libonnxruntime.so.1.22.0"));
            assert!(!is_versioned_library("libonnxruntime_providers_cuda.so"));
        }
        if cfg!(target_os = "macos") {
            assert!(is_versioned_library("libonnxruntime.1.22.0.dylib"));
        }
    }
}
