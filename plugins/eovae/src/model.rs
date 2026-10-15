//! Inspecting, validating and identifying models (VAE-15, VAE-12).
//!
//! A model is accepted only if eovae can feed it: exactly one 4-D RGB
//! float32/float16 input, fed by name, and at least one image output with the
//! input's layout and size. The model is identified by a SHA-256 over the
//! `.onnx` file *and* every external-data file holding its weights, so
//! changed weights never reuse cached results.

use crate::contract::{Contract, ContractSource, Layout};
use crate::onnx_scan::{self, Dim, ElemType, OnnxSummary, TensorInfo};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// Element type of a tensor eovae can feed or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
}

impl TensorType {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
        }
    }
}

/// An image output the model offers.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageOutput {
    pub name: String,
    pub elem: TensorType,
}

/// A non-image output read as an embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentOutput {
    pub name: String,
    pub elem: TensorType,
    pub dim: usize,
}

/// A model eovae has inspected and accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub file_name: String,
    /// Hex SHA-256 over the model and its external weights.
    pub sha256: String,
    pub summary: OnnxSummary,
    pub input: TensorInfo,
    pub input_elem: TensorType,
    pub layout: Layout,
    /// Input edge in pixels.
    pub tile_size: u32,
    /// The batch size baked into the graph, if any.
    pub fixed_batch: Option<usize>,
    pub outputs: Vec<ImageOutput>,
    /// Index into `outputs` of the default output.
    pub default_output: usize,
    pub latent: Option<LatentOutput>,
    /// The declared (or default) contract, before user overrides.
    pub contract: Contract,
    pub contract_source: ContractSource,
    /// Name of a model eovae recognises, if any.
    pub card: Option<&'static str>,
    pub warnings: Vec<String>,
    /// Bytes of the model and its weights.
    pub bytes: u64,
}

impl ModelInfo {
    pub fn output(&self, name: &str) -> Option<&ImageOutput> {
        self.outputs.iter().find(|output| output.name == name)
    }

    pub fn default_output_name(&self) -> &str {
        &self.outputs[self.default_output].name
    }

    pub fn item_len(&self) -> usize {
        self.tile_size as usize * self.tile_size as usize * 3
    }
}

/// A model eovae knows by its weights.
struct ModelCard {
    sha256: &'static str,
    name: &'static str,
    /// Applied on top of HistoVAE's defaults.
    declarations: &'static [(&'static str, &'static str)],
}

/// The reference model shipped with HistoVAE
/// (`pretrained/HistoVae_reconstruct.onnx` + `.onnx.data`), trained on
/// CAMELYON17 level 0 (about 0.23 µm/px).
const CARDS: &[ModelCard] = &[ModelCard {
    sha256: "cc2e3e5bb22ea87b2d30b00503b97bc1115b2b89aa30b708970054464a47c94c",
    name: "HistoVAE reference (CAMELYON17)",
    declarations: &[("mpp", "0.227")],
}];

/// Splits a 4-D input shape into (layout, height, width, batch).
fn classify_image(dims: &[Dim]) -> Result<(Layout, Dim, Dim, Dim), String> {
    if dims.len() != 4 {
        return Err(format!(
            "it has {} dimensions; an image tensor has 4 (batch, channels, height, width)",
            dims.len()
        ));
    }
    let channels_first = dims[1].fixed();
    let channels_last = dims[3].fixed();
    match (channels_first, channels_last) {
        (Some(3), _) => Ok((
            Layout::Nchw,
            dims[2].clone(),
            dims[3].clone(),
            dims[0].clone(),
        )),
        (_, Some(3)) => Ok((
            Layout::Nhwc,
            dims[1].clone(),
            dims[2].clone(),
            dims[0].clone(),
        )),
        (Some(c), _) if c == 1 || c == 4 => {
            Err(format!("it has {c} channels; eovae feeds RGB (3 channels)"))
        }
        _ => Err("its channel dimension is not 3 (RGB)".to_string()),
    }
}

fn tensor_type(elem: Option<ElemType>) -> Option<TensorType> {
    match elem? {
        ElemType::Float => Some(TensorType::F32),
        ElemType::Float16 => Some(TensorType::F16),
        _ => None,
    }
}

fn same_dim(a: &Dim, b: &Dim) -> bool {
    match (a.fixed(), b.fixed()) {
        (Some(x), Some(y)) => x == y,
        // A symbolic dimension can match anything.
        _ => true,
    }
}

/// Checks a scanned model and works out how to feed it.
/// How a scanned model is fed: its input, layout and outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedPlan {
    pub input: TensorInfo,
    pub input_elem: TensorType,
    pub layout: Layout,
    pub tile_size: u32,
    pub fixed_batch: Option<usize>,
    pub outputs: Vec<ImageOutput>,
    pub default_output: usize,
    pub latent: Option<LatentOutput>,
    pub warnings: Vec<String>,
}

/// Checks a scanned model and works out how to feed it.
pub fn analyse_summary(
    summary: &OnnxSummary,
    tile_size_hint: Option<u32>,
) -> Result<FeedPlan, String> {
    let mut warnings = Vec::new();
    let input = match summary.inputs.as_slice() {
        [single] => single.clone(),
        [] => return Err("the model has no inputs".to_string()),
        several => {
            return Err(format!(
                "the model needs {} inputs ({}); eovae feeds exactly one image input",
                several.len(),
                several
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };
    let input_elem = tensor_type(input.elem_type).ok_or_else(|| {
        format!(
            "input '{}' is {}; eovae feeds float32 or float16",
            input.name,
            input.type_label()
        )
    })?;
    let dims = input
        .shape
        .as_ref()
        .ok_or_else(|| format!("input '{}' does not declare its shape", input.name))?;
    let (layout, height, width, batch) =
        classify_image(dims).map_err(|e| format!("input '{}' is not an image: {e}", input.name))?;
    let tile_size = match (height.fixed(), width.fixed()) {
        (Some(h), Some(w)) if h == w => h as u32,
        (Some(h), Some(w)) => {
            return Err(format!(
                "input '{}' is {h}×{w}; eovae analyses square tiles",
                input.name
            ));
        }
        (None, None) => {
            let size = tile_size_hint.unwrap_or(256);
            warnings.push(format!(
                "The input size is dynamic; analysing {size}×{size} tiles{}.",
                if tile_size_hint.is_some() {
                    " as the model declares"
                } else {
                    " (declare eovae.tile_size to change)"
                }
            ));
            size
        }
        (Some(size), None) | (None, Some(size)) => size as u32,
    };
    if !(16..=2048).contains(&tile_size) {
        return Err(format!(
            "tiles of {tile_size} px are not supported (16–2048)"
        ));
    }
    let fixed_batch = batch.fixed().map(|b| b as usize);
    if let Some(batch) = fixed_batch
        && batch > 1
    {
        warnings.push(format!(
            "The model has a fixed batch of {batch}: tiles are analysed {batch} at a time and the last batch of each run is padded. Export with a dynamic batch axis for small viewport runs."
        ));
    }
    if input_elem == TensorType::F16 {
        warnings.push("Float16 model: inputs are converted from float32.".to_string());
    }

    let mut outputs = Vec::new();
    let mut rejected = Vec::new();
    for output in &summary.outputs {
        let Some(elem) = tensor_type(output.elem_type) else {
            rejected.push(format!("{} ({})", output.name, output.type_label()));
            continue;
        };
        let Some(dims) = &output.shape else {
            rejected.push(format!("{} (no shape)", output.name));
            continue;
        };
        match classify_image(dims) {
            Ok((out_layout, out_h, out_w, out_batch))
                if out_layout == layout
                    && same_dim(&out_h, &height)
                    && same_dim(&out_w, &width)
                    && same_dim(&out_batch, &batch)
                    && out_h.fixed().is_none_or(|h| h as u32 == tile_size) =>
            {
                outputs.push(ImageOutput {
                    name: output.name.clone(),
                    elem,
                });
            }
            _ => rejected.push(format!("{} {}", output.name, output.shape_label())),
        }
    }
    if outputs.is_empty() {
        let names: Vec<String> = summary
            .outputs
            .iter()
            .map(|o| o.name.to_ascii_lowercase())
            .collect();
        let encoder = names.iter().any(|n| n == "mu") && names.iter().any(|n| n == "logvar");
        return Err(if encoder {
            "the model only encodes (outputs mu and logvar); export the reconstruction model (HistoVAE: create_onnx --mode reconstruct)".to_string()
        } else {
            format!(
                "no output is an RGB image matching the input {} (outputs: {})",
                input.shape_label(),
                rejected.join(", ")
            )
        });
    }
    let default_output = outputs
        .iter()
        .position(|o| {
            let name = o.name.to_ascii_lowercase();
            name.contains("recon") || name.contains("decod")
        })
        .unwrap_or(0);
    if outputs.len() > 1 {
        warnings.push(format!(
            "{} image outputs; using '{}' (change it in the Model section).",
            outputs.len(),
            outputs[default_output].name
        ));
    }

    let latent = summary
        .outputs
        .iter()
        .filter(|o| !outputs.iter().any(|image| image.name == o.name))
        .filter_map(|o| {
            let elem = tensor_type(o.elem_type)?;
            let dims = o.shape.as_ref()?;
            if dims.len() < 2 {
                return None;
            }
            let dim = dims[1..]
                .iter()
                .try_fold(1usize, |acc, d| d.fixed().map(|v| acc * v as usize))?;
            (dim > 0 && dim <= 1 << 20).then(|| LatentOutput {
                name: o.name.clone(),
                elem,
                dim,
            })
        })
        .max_by_key(|latent| {
            let name = latent.name.to_ascii_lowercase();
            if name == "mu" || name.contains("latent") || name.contains("embed") {
                2
            } else if name.contains("logvar") {
                0
            } else {
                1
            }
        });

    Ok(FeedPlan {
        input,
        input_elem,
        layout,
        tile_size,
        fixed_batch,
        outputs,
        default_output,
        latent,
        warnings,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FileVersion {
    len: u64,
    modified: Option<SystemTime>,
}

fn file_version(path: &Path) -> std::io::Result<FileVersion> {
    let metadata = std::fs::metadata(path)?;
    Ok(FileVersion {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    })
}

type HashCache = Mutex<HashMap<Vec<PathBuf>, (Vec<FileVersion>, String)>>;

fn hash_cache() -> &'static HashCache {
    static CACHE: OnceLock<HashCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// SHA-256 over every file of a model (name, length and content of each),
/// cached while the files' sizes and modification times are unchanged.
pub fn hash_model_files(files: &[PathBuf], cancel: &AtomicBool) -> Result<String, String> {
    let versions = files
        .iter()
        .map(|path| file_version(path).map_err(|e| format!("{}: {e}", path.display())))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some((cached, digest)) = hash_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(files)
        && *cached == versions
    {
        return Ok(digest.clone());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"eovae-model-v1\0");
    let mut buffer = vec![0u8; 1 << 20];
    for (path, version) in files.iter().zip(&versions) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(version.len.to_le_bytes());
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let read = file
                .read(&mut buffer)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    hash_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(files.to_vec(), (versions, digest.clone()));
    Ok(digest)
}

/// The sidecar contract file of a model: `<model>.eovae.json`.
pub fn sidecar_path(model: &Path) -> PathBuf {
    let mut name = model
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".eovae.json");
    model.with_file_name(name)
}

/// Inspects a model file: reads its structure, checks it can be fed,
/// hashes it with its weights and resolves its contract. Slow for large
/// models (hashing reads every weight); call it off the UI thread.
pub fn inspect(path: &Path, cancel: &AtomicBool) -> Result<ModelInfo, String> {
    let is_onnx = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"));
    if !is_onnx {
        return Err("only .onnx models are supported".to_string());
    }
    if !path.is_file() {
        return Err(format!("{} does not exist", path.display()));
    }
    let summary = onnx_scan::scan_file(path).map_err(|e| e.to_string())?;

    // External weights must sit next to the model, as ONNX Runtime expects.
    let directory = path.parent().unwrap_or(Path::new("."));
    let mut files = vec![path.to_path_buf()];
    for location in &summary.external_data {
        if !onnx_scan::is_safe_location(location) {
            return Err(format!(
                "the model's weights point outside its folder ('{location}')"
            ));
        }
        let weights = directory.join(location);
        if !weights.is_file() {
            return Err(format!(
                "the model's weights file '{location}' is missing; keep it next to the .onnx file"
            ));
        }
        files.push(weights);
    }
    let bytes = files
        .iter()
        .map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0))
        .sum();
    let sha256 = hash_model_files(&files, cancel)?;

    // Contract: defaults, then a known model card, metadata, sidecar.
    let mut contract = Contract::histovae();
    let mut source = ContractSource::Default;
    let mut warnings = Vec::new();
    let card = CARDS.iter().find(|card| card.sha256 == sha256);
    if let Some(card) = card {
        let declarations: BTreeMap<String, String> = card
            .declarations
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        contract.apply_declarations(&declarations);
        source = ContractSource::BuiltIn;
    }
    let (applied, problems) =
        contract.apply_declarations(&Contract::metadata_declarations(&summary.metadata));
    if applied {
        source = ContractSource::Metadata;
    }
    warnings.extend(
        problems
            .into_iter()
            .map(|p| format!("Model metadata: {p}.")),
    );
    let sidecar = sidecar_path(path);
    if sidecar.is_file() {
        let parsed = std::fs::read_to_string(&sidecar)
            .map_err(|e| e.to_string())
            .and_then(|text| {
                serde_json::from_str::<serde_json::Value>(&text).map_err(|e| e.to_string())
            })
            .and_then(|json| Contract::sidecar_declarations(&json));
        match parsed {
            Ok(declarations) => {
                let (applied, problems) = contract.apply_declarations(&declarations);
                if applied {
                    source = ContractSource::Sidecar;
                }
                warnings.extend(problems.into_iter().map(|p| format!("Sidecar: {p}.")));
            }
            Err(error) => warnings.push(format!(
                "Ignored {}: {error}.",
                sidecar.file_name().unwrap_or_default().to_string_lossy()
            )),
        }
    }

    let FeedPlan {
        input,
        input_elem,
        layout,
        tile_size,
        fixed_batch,
        outputs,
        default_output,
        latent,
        warnings: mut model_warnings,
    } = analyse_summary(&summary, contract.tile_size)?;
    warnings.append(&mut model_warnings);
    if contract.mpp.is_none() {
        warnings.push(
            "The model does not declare the resolution it was trained at (eovae.mpp); errors depend on the level you choose.".to_string(),
        );
    }
    if source == ContractSource::Default {
        warnings.push(
            "No preprocessing declared: using HistoVAE's ([-1, 1], alpha on white). Check it in Preprocessing.".to_string(),
        );
    }
    Ok(ModelInfo {
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: path.to_path_buf(),
        sha256,
        summary,
        input,
        input_elem,
        layout,
        tile_size,
        fixed_batch,
        outputs,
        default_output,
        latent,
        contract,
        contract_source: source,
        card: card.map(|card| card.name),
        warnings,
        bytes,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::onnx_scan::encode::{ModelSpec, model};

    /// Writes a model (and its external data) into `dir`.
    pub(crate) fn write_model(dir: &Path, name: &str, spec: &ModelSpec) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, model(spec)).unwrap();
        for (_, external, _) in &spec.initializers {
            if let Some(location) = external {
                std::fs::write(dir.join(location), vec![1u8; 64]).unwrap();
            }
        }
        path
    }

    pub(crate) fn small_vae(fixed_batch: i64) -> ModelSpec<'static> {
        ModelSpec {
            inputs: vec![("input", 1, vec![fixed_batch, 3, 32, 32])],
            outputs: vec![("reconstruction", 1, vec![fixed_batch, 3, 32, 32])],
            initializers: vec![("w", Some("vae.onnx.data"), 0)],
            metadata: vec![],
            initializers_as_inputs: false,
        }
    }

    #[test]
    fn accepts_a_reconstruction_model_and_hashes_its_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_model(dir.path(), "vae.onnx", &small_vae(4));
        let cancel = AtomicBool::new(false);
        let info = inspect(&path, &cancel).unwrap();
        assert_eq!(info.tile_size, 32);
        assert_eq!(info.layout, Layout::Nchw);
        assert_eq!(info.fixed_batch, Some(4));
        assert_eq!(info.default_output_name(), "reconstruction");
        assert_eq!(info.contract_source, ContractSource::Default);
        assert!(info.warnings.iter().any(|w| w.contains("fixed batch")));
        // Changing the external weights changes the identity.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.path().join("vae.onnx.data"), vec![2u8; 65]).unwrap();
        let changed = inspect(&path, &cancel).unwrap();
        assert_ne!(changed.sha256, info.sha256);
        // A missing weights file is an error, not a crash at inference.
        std::fs::remove_file(dir.path().join("vae.onnx.data")).unwrap();
        assert!(
            inspect(&path, &cancel)
                .unwrap_err()
                .contains("weights file")
        );
    }

    #[test]
    fn rejects_models_it_cannot_feed() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = AtomicBool::new(false);
        let cases: Vec<(ModelSpec, &str)> = vec![
            (
                ModelSpec {
                    inputs: vec![("a", 1, vec![1, 3, 32, 32]), ("b", 1, vec![1])],
                    ..small_vae(1)
                },
                "exactly one",
            ),
            (
                ModelSpec {
                    inputs: vec![("input", 7, vec![1, 3, 32, 32])],
                    ..small_vae(1)
                },
                "float32 or float16",
            ),
            (
                ModelSpec {
                    inputs: vec![("input", 1, vec![1, 1, 32, 32])],
                    ..small_vae(1)
                },
                "RGB",
            ),
            (
                ModelSpec {
                    outputs: vec![("mu", 1, vec![1, 64]), ("logvar", 1, vec![1, 64])],
                    ..small_vae(1)
                },
                "only encodes",
            ),
            (
                ModelSpec {
                    outputs: vec![("other", 1, vec![1, 3, 16, 16])],
                    ..small_vae(1)
                },
                "no output",
            ),
        ];
        for (index, (spec, expected)) in cases.iter().enumerate() {
            let path = write_model(dir.path(), &format!("m{index}.onnx"), spec);
            let error = inspect(&path, &cancel).unwrap_err();
            assert!(error.contains(expected), "case {index}: {error}");
        }
        assert!(inspect(&dir.path().join("x.pt"), &cancel).is_err());
    }

    #[test]
    fn reads_contracts_from_metadata_and_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = AtomicBool::new(false);
        let spec = ModelSpec {
            inputs: vec![("input", 10, vec![-1, 224, 224, 3])],
            outputs: vec![
                ("out", 10, vec![-1, 224, 224, 3]),
                ("mu", 1, vec![-1, 8, 4, 4]),
            ],
            metadata: vec![("eovae.input_range", "0,1"), ("eovae.mpp", "0.5")],
            ..small_vae(-1)
        };
        let path = write_model(dir.path(), "meta.onnx", &spec);
        let info = inspect(&path, &cancel).unwrap();
        assert_eq!(info.layout, Layout::Nhwc);
        assert_eq!(info.input_elem, TensorType::F16);
        assert_eq!(info.fixed_batch, None);
        assert_eq!(info.contract_source, ContractSource::Metadata);
        assert_eq!(info.contract.mpp, Some(0.5));
        assert_eq!(
            info.latent.as_ref().map(|l| (l.name.as_str(), l.dim)),
            Some(("mu", 128))
        );

        std::fs::write(
            sidecar_path(&path),
            r#"{"mpp": 0.25, "output_range": "0,255"}"#,
        )
        .unwrap();
        let info = inspect(&path, &cancel).unwrap();
        assert_eq!(info.contract_source, ContractSource::Sidecar);
        assert_eq!(info.contract.mpp, Some(0.25));
    }

    #[test]
    fn identifies_the_reference_model_when_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("HistoVAE/pretrained/HistoVae_reconstruct.onnx");
        if !path.exists() {
            return;
        }
        let info = inspect(&path, &AtomicBool::new(false)).unwrap();
        assert_eq!(
            info.card,
            Some("HistoVAE reference (CAMELYON17)"),
            "{}",
            info.sha256
        );
        assert_eq!(info.contract_source, ContractSource::BuiltIn);
        assert_eq!(info.contract.mpp, Some(0.227));
        assert_eq!(info.fixed_batch, Some(32));
        assert_eq!(info.tile_size, 256);
    }
}
