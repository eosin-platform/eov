//! Inference engines, behind a trait so the analysis pipeline and its tests
//! do not depend on ONNX Runtime.
//!
//! The inference device is chosen independently of the viewer's render
//! backend (VAE-8): [`DevicePreference::Auto`] tries CUDA and falls back to
//! the CPU, saying why; `Cuda` insists on CUDA; `Cpu` never touches a GPU.

use crate::model::ModelInfo;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Which device the user wants inference on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevicePreference {
    /// CUDA when it works, otherwise the CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl DevicePreference {
    pub const ALL: [DevicePreference; 3] = [Self::Auto, Self::Cpu, Self::Cuda];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|d| d.label().eq_ignore_ascii_case(label.trim()))
    }
}

/// The device an engine actually runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn key(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }
}

/// What an engine is.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineInfo {
    pub device: Device,
    /// Numeric precision of the computation (`fp32`, `tf32`): part of the
    /// cache key because it changes results slightly.
    pub precision: String,
    /// Human-readable description ("CUDA · TF32", "CPU · 8 threads").
    pub label: String,
    /// Why `Auto` did not get CUDA.
    pub fallback: Option<String>,
    /// The batch size the model's graph requires, if fixed.
    pub fixed_batch: Option<usize>,
    /// Values per batch item of the input and output tensors.
    pub item_len: usize,
    /// Values per batch item of the latent output, if one is read.
    pub latent_dim: Option<usize>,
}

/// One batch of results.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineOutput {
    /// `batch · item_len` reconstruction values, in the input's layout.
    pub recon: Vec<f32>,
    /// `batch · latent_dim` latent values, if requested.
    pub latent: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    /// The run was stopped through the stop flag.
    Stopped,
    Failed(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("stopped"),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

/// A loaded model on a device.
pub trait InferenceEngine: Send {
    fn info(&self) -> &EngineInfo;

    /// Runs `batch` items (`input.len() == batch · item_len`). Must return
    /// promptly with [`EngineError::Stopped`] once `stop` is set.
    fn run(
        &mut self,
        input: &[f32],
        batch: usize,
        stop: &Arc<AtomicBool>,
    ) -> Result<EngineOutput, EngineError>;
}

/// What to build an engine for.
#[derive(Clone, Debug)]
pub struct EngineRequest {
    pub model: Arc<ModelInfo>,
    /// The output to read.
    pub output: String,
    /// The latent output to read, if any.
    pub latent: Option<String>,
    pub device: DevicePreference,
    pub threads: usize,
    /// Allow TF32 math on CUDA (faster, slightly different results).
    pub tf32: bool,
    /// Batch size used for dynamic-batch models (and the warm-up run).
    pub batch_size: usize,
    /// A user-chosen ONNX Runtime library (or folder), tried first. The
    /// runtime is loaded once per process.
    pub runtime_path: Option<PathBuf>,
}

impl EngineRequest {
    /// The configuration an engine was built for; engines are reused while
    /// it is unchanged.
    pub fn signature(&self) -> String {
        format!(
            "{}|{}|{}|{:?}|{}|{}|{}",
            self.model.sha256,
            self.output,
            self.latent.as_deref().unwrap_or(""),
            self.device,
            self.threads,
            self.tf32,
            if self.model.fixed_batch.is_some() {
                0
            } else {
                self.batch_size
            }
        )
    }
}

/// Creates engines. Implemented by the ONNX Runtime backend and by test
/// doubles.
pub trait Backend: Send + Sync + 'static {
    /// `plugin_root` is the extracted package (bundled runtimes live there).
    fn new(plugin_root: &Path) -> Self
    where
        Self: Sized;

    /// Builds (and warms up) an engine. Runs on a worker thread; `status`
    /// reports progress to the user ("Loading ONNX Runtime", "Warming up
    /// CUDA"). Gives up soon after any `stop` flag is set.
    fn create(
        &self,
        request: &EngineRequest,
        status: &dyn Fn(&str),
        stop: &[Arc<AtomicBool>],
    ) -> Result<Box<dyn InferenceEngine>, String>;

    /// Which inference runtime is in use, for the sidebar.
    fn runtime_label(&self) -> String;
}

#[cfg(test)]
pub(crate) mod testing {
    //! Engines for tests.

    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// How a test engine transforms its input.
    #[derive(Clone, Copy, Debug)]
    pub enum Behaviour {
        /// Output = input (zero error).
        Identity,
        /// Output = input + offset (in tensor units).
        Offset(f32),
        /// Fails on the n-th run (0-based).
        FailOn(usize),
        /// Waits until stopped.
        Hang,
    }

    pub struct TestEngine {
        pub info: EngineInfo,
        pub behaviour: Behaviour,
        pub runs: Arc<AtomicUsize>,
        pub batches: Arc<Mutex<Vec<usize>>>,
    }

    impl TestEngine {
        pub fn new(item_len: usize, fixed_batch: Option<usize>, behaviour: Behaviour) -> Self {
            Self {
                info: EngineInfo {
                    device: Device::Cpu,
                    precision: "fp32".to_string(),
                    label: "Test engine".to_string(),
                    fallback: None,
                    fixed_batch,
                    item_len,
                    latent_dim: Some(2),
                },
                behaviour,
                runs: Arc::new(AtomicUsize::new(0)),
                batches: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl InferenceEngine for TestEngine {
        fn info(&self) -> &EngineInfo {
            &self.info
        }

        fn run(
            &mut self,
            input: &[f32],
            batch: usize,
            stop: &Arc<AtomicBool>,
        ) -> Result<EngineOutput, EngineError> {
            assert_eq!(input.len(), batch * self.info.item_len, "input size");
            if let Some(fixed) = self.info.fixed_batch {
                assert_eq!(batch, fixed, "fixed-batch models get full batches");
            }
            let run = self.runs.fetch_add(1, Ordering::SeqCst);
            self.batches.lock().unwrap().push(batch);
            let recon = match self.behaviour {
                Behaviour::Identity => input.to_vec(),
                Behaviour::Offset(offset) => input.iter().map(|v| v + offset).collect(),
                Behaviour::FailOn(n) if n == run => {
                    return Err(EngineError::Failed(
                        "model output has the wrong shape".into(),
                    ));
                }
                Behaviour::FailOn(_) => input.to_vec(),
                Behaviour::Hang => loop {
                    if stop.load(Ordering::SeqCst) {
                        return Err(EngineError::Stopped);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                },
            };
            Ok(EngineOutput {
                recon,
                latent: Some((0..batch * 2).map(|i| i as f32).collect()),
            })
        }
    }

    /// A backend producing [`TestEngine`]s.
    pub struct TestBackend;

    impl Backend for TestBackend {
        fn new(_plugin_root: &Path) -> Self {
            TestBackend
        }

        fn create(
            &self,
            request: &EngineRequest,
            status: &dyn Fn(&str),
            _stop: &[Arc<AtomicBool>],
        ) -> Result<Box<dyn InferenceEngine>, String> {
            status("Building test engine");
            let tile = request.model.tile_size as usize;
            // The model's file name picks the behaviour.
            let name = &request.model.file_name;
            let behaviour = if name.contains("hang") {
                Behaviour::Hang
            } else if name.contains("fail") {
                Behaviour::FailOn(0)
            } else {
                Behaviour::Offset(0.02)
            };
            let mut engine = TestEngine::new(tile * tile * 3, request.model.fixed_batch, behaviour);
            if request.device == DevicePreference::Cuda {
                return Err("CUDA is not available in tests".to_string());
            }
            if request.device == DevicePreference::Auto {
                engine.info.fallback = Some("no CUDA in tests".to_string());
            }
            Ok(Box::new(engine))
        }

        fn runtime_label(&self) -> String {
            "test runtime".to_string()
        }
    }
}
