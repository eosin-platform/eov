//! The model's input/output contract: how slide pixels become model inputs
//! and how outputs become pixels again (VAE-4).
//!
//! The defaults are HistoVAE's training pipeline
//! (`HistoVAE/pytorch/src/data.py`, `create_onnx.py`): RGBA is composited
//! onto white, pixels darker than 4 in every channel become white, and RGB is
//! mapped to [-1, 1]; the output is in [-1, 1] too. Models can declare a
//! different contract in their ONNX `metadata_props` or a sidecar file, and
//! the user can override it (see [`ContractOverride`]).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bump whenever the meaning of preprocessing, postprocessing or scoring
/// changes, so cached results computed the old way are not reused.
pub const CONTRACT_VERSION: u32 = 2;

/// A value range a tensor uses for 0–255 pixel values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueRange {
    /// `v / 127.5 - 1`
    MinusOneOne,
    /// `v / 255`
    ZeroOne,
    /// `v`
    Zero255,
}

impl ValueRange {
    pub const ALL: [ValueRange; 3] = [Self::MinusOneOne, Self::ZeroOne, Self::Zero255];

    pub fn label(self) -> &'static str {
        match self {
            Self::MinusOneOne => "[-1, 1]",
            Self::ZeroOne => "[0, 1]",
            Self::Zero255 => "[0, 255]",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        let compact: String = text
            .chars()
            .filter(|c| !c.is_whitespace() && !matches!(c, '[' | ']' | '(' | ')'))
            .collect::<String>()
            .to_ascii_lowercase();
        match compact.as_str() {
            "-1,1" | "minus_one_one" | "tanh" => Some(Self::MinusOneOne),
            "0,1" | "zero_one" | "unit" | "sigmoid" => Some(Self::ZeroOne),
            "0,255" | "zero_255" | "uint8" | "byte" => Some(Self::Zero255),
            _ => None,
        }
    }

    fn encode(self, v255: f32) -> f32 {
        match self {
            Self::MinusOneOne => v255 / 127.5 - 1.0,
            Self::ZeroOne => v255 / 255.0,
            Self::Zero255 => v255,
        }
    }

    fn decode(self, value: f32) -> f32 {
        match self {
            Self::MinusOneOne => (value + 1.0) * 127.5,
            Self::ZeroOne => value * 255.0,
            Self::Zero255 => value,
        }
    }
}

/// How 0–255 channel values map to tensor values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scaling {
    Range {
        range: ValueRange,
    },
    /// `(v / 255 - mean[c]) / std[c]` (e.g. ImageNet statistics).
    MeanStd {
        mean: [f32; 3],
        std: [f32; 3],
    },
}

impl Scaling {
    pub const IMAGENET: Scaling = Scaling::MeanStd {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };

    pub fn range(range: ValueRange) -> Self {
        Self::Range { range }
    }

    #[inline]
    pub fn encode(&self, channel: usize, v255: f32) -> f32 {
        match self {
            Self::Range { range } => range.encode(v255),
            Self::MeanStd { mean, std } => (v255 / 255.0 - mean[channel]) / std[channel],
        }
    }

    #[inline]
    pub fn decode(&self, channel: usize, value: f32) -> f32 {
        match self {
            Self::Range { range } => range.decode(value),
            Self::MeanStd { mean, std } => (value * std[channel] + mean[channel]) * 255.0,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Range { range } => range.label().to_string(),
            Self::MeanStd { .. } if *self == Self::IMAGENET => "ImageNet mean/std".to_string(),
            Self::MeanStd { mean, std } => format!(
                "mean {:.3}/{:.3}/{:.3}, std {:.3}/{:.3}/{:.3}",
                mean[0], mean[1], mean[2], std[0], std[1], std[2]
            ),
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::Range { .. } => true,
            Self::MeanStd { mean, std } => {
                mean.iter().all(|v| v.is_finite()) && std.iter().all(|v| v.is_finite() && *v > 0.0)
            }
        }
    }
}

/// Where a model's contract came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractSource {
    /// Nothing declared: HistoVAE's defaults.
    Default,
    /// A model eovae knows by its hash.
    BuiltIn,
    /// The model's ONNX `metadata_props`.
    Metadata,
    /// `<model>.eovae.json` next to the model.
    Sidecar,
}

impl ContractSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Default => "HistoVAE defaults",
            Self::BuiltIn => "built-in model card",
            Self::Metadata => "model metadata",
            Self::Sidecar => "sidecar file",
        }
    }
}

/// The model contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    pub input: Scaling,
    pub output: Scaling,
    /// Composite transparent pixels onto white (`true`) or black.
    pub composite_on_white: bool,
    /// Pixels darker than this in every channel become white (scanner
    /// artefacts); `0` disables.
    pub black_to_white_below: u8,
    /// Tensors use BGR channel order.
    pub bgr: bool,
    /// µm per pixel the model was trained at.
    pub mpp: Option<f64>,
    /// Input edge for models with dynamic spatial dimensions.
    pub tile_size: Option<u32>,
}

impl Default for Contract {
    fn default() -> Self {
        Self::histovae()
    }
}

/// Keys understood in ONNX `metadata_props` (prefixed with `eovae.`) and in
/// sidecar JSON files (without the prefix).
pub const METADATA_KEYS: &[&str] = &[
    "input_range",
    "input_mean",
    "input_std",
    "output_range",
    "output_mean",
    "output_std",
    "background",
    "black_to_white",
    "channel_order",
    "mpp",
    "tile_size",
];

fn parse_triplet(text: &str) -> Option<[f32; 3]> {
    let values: Vec<f32> = text
        .split([',', ' ', ';'])
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().trim_matches(['[', ']']).parse::<f32>())
        .collect::<Result<_, _>>()
        .ok()?;
    match values.as_slice() {
        [v] => Some([*v; 3]),
        [a, b, c] => Some([*a, *b, *c]),
        _ => None,
    }
}

impl Contract {
    /// HistoVAE's training contract.
    pub fn histovae() -> Self {
        Self {
            input: Scaling::range(ValueRange::MinusOneOne),
            output: Scaling::range(ValueRange::MinusOneOne),
            composite_on_white: true,
            black_to_white_below: 4,
            bgr: false,
            mpp: None,
            tile_size: None,
        }
    }

    /// The `background` of region reads (`0xRRGGBBAA`).
    pub fn background_rgba(&self) -> u32 {
        if self.composite_on_white {
            0xFFFF_FFFF
        } else {
            0x0000_00FF
        }
    }

    fn background_rgb(&self) -> u8 {
        if self.composite_on_white { 255 } else { 0 }
    }

    /// Applies declared values (keys without the `eovae.` prefix). Returns
    /// whether anything was applied, and problems with bad values.
    pub fn apply_declarations(&mut self, values: &BTreeMap<String, String>) -> (bool, Vec<String>) {
        let mut applied = false;
        let mut problems = Vec::new();
        let mut bad = |key: &str, value: &str| problems.push(format!("ignored {key} = '{value}'"));
        let mut input_mean = None;
        let mut input_std = None;
        let mut output_mean = None;
        let mut output_std = None;
        for (key, value) in values {
            let value = value.trim();
            match key.as_str() {
                "input_range" => match ValueRange::parse(value) {
                    Some(range) => {
                        self.input = Scaling::range(range);
                        applied = true;
                    }
                    None => bad(key, value),
                },
                "output_range" => match ValueRange::parse(value) {
                    Some(range) => {
                        self.output = Scaling::range(range);
                        applied = true;
                    }
                    None => bad(key, value),
                },
                "input_mean" => {
                    input_mean = parse_triplet(value).or_else(|| {
                        bad(key, value);
                        None
                    })
                }
                "input_std" => {
                    input_std = parse_triplet(value).or_else(|| {
                        bad(key, value);
                        None
                    })
                }
                "output_mean" => {
                    output_mean = parse_triplet(value).or_else(|| {
                        bad(key, value);
                        None
                    })
                }
                "output_std" => {
                    output_std = parse_triplet(value).or_else(|| {
                        bad(key, value);
                        None
                    })
                }
                "background" => match value.to_ascii_lowercase().as_str() {
                    "white" => {
                        self.composite_on_white = true;
                        applied = true;
                    }
                    "black" => {
                        self.composite_on_white = false;
                        applied = true;
                    }
                    _ => bad(key, value),
                },
                "black_to_white" => match value.to_ascii_lowercase().as_str() {
                    "off" | "false" | "no" | "none" => {
                        self.black_to_white_below = 0;
                        applied = true;
                    }
                    "on" | "true" | "yes" => {
                        self.black_to_white_below = 4;
                        applied = true;
                    }
                    number => match number.parse::<u8>() {
                        Ok(threshold) => {
                            self.black_to_white_below = threshold;
                            applied = true;
                        }
                        Err(_) => bad(key, value),
                    },
                },
                "channel_order" => match value.to_ascii_lowercase().as_str() {
                    "rgb" => {
                        self.bgr = false;
                        applied = true;
                    }
                    "bgr" => {
                        self.bgr = true;
                        applied = true;
                    }
                    _ => bad(key, value),
                },
                "mpp" | "microns_per_pixel" => match value.parse::<f64>() {
                    Ok(mpp) if mpp.is_finite() && mpp > 0.0 && mpp < 1000.0 => {
                        self.mpp = Some(mpp);
                        applied = true;
                    }
                    _ => bad(key, value),
                },
                "tile_size" => match value.parse::<u32>() {
                    Ok(size) if (16..=4096).contains(&size) => {
                        self.tile_size = Some(size);
                        applied = true;
                    }
                    _ => bad(key, value),
                },
                _ => {}
            }
        }
        for (mean, std, target) in [
            (input_mean, input_std, &mut self.input),
            (output_mean, output_std, &mut self.output),
        ] {
            match (mean, std) {
                (Some(mean), Some(std)) => {
                    let scaling = Scaling::MeanStd { mean, std };
                    if scaling.is_valid() {
                        *target = scaling;
                        applied = true;
                    } else {
                        problems.push("ignored mean/std with a non-positive std".to_string());
                    }
                }
                (None, None) => {}
                _ => problems.push("mean and std must be declared together".to_string()),
            }
        }
        (applied, problems)
    }

    /// Declarations from ONNX `metadata_props` (`eovae.*` keys).
    pub fn metadata_declarations(metadata: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        metadata
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix("eovae.")
                    .map(|key| (key.to_string(), value.clone()))
            })
            .collect()
    }

    /// Declarations from a sidecar JSON object.
    pub fn sidecar_declarations(
        json: &serde_json::Value,
    ) -> Result<BTreeMap<String, String>, String> {
        let object = json
            .as_object()
            .ok_or_else(|| "the sidecar must be a JSON object".to_string())?;
        Ok(object
            .iter()
            .map(|(key, value)| {
                let text = match value {
                    serde_json::Value::String(text) => text.clone(),
                    serde_json::Value::Array(items) => items
                        .iter()
                        .map(|item| item.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    other => other.to_string(),
                };
                (key.trim_start_matches("eovae.").to_string(), text)
            })
            .collect())
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![format!(
            "in {} → out {}",
            self.input.label(),
            self.output.label()
        )];
        parts.push(if self.composite_on_white {
            "alpha on white".to_string()
        } else {
            "alpha on black".to_string()
        });
        if self.black_to_white_below > 0 {
            parts.push(format!("black<{} → white", self.black_to_white_below));
        }
        if self.bgr {
            parts.push("BGR".to_string());
        }
        parts.join(", ")
    }
}

/// User overrides of a model's contract (persisted per model hash).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContractOverride {
    pub input: Option<Scaling>,
    pub output: Option<Scaling>,
    pub composite_on_white: Option<bool>,
    pub black_to_white_below: Option<u8>,
}

impl ContractOverride {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn apply(&self, contract: &mut Contract) {
        if let Some(input) = &self.input {
            contract.input = input.clone();
        }
        if let Some(output) = &self.output {
            contract.output = output.clone();
        }
        if let Some(white) = self.composite_on_white {
            contract.composite_on_white = white;
        }
        if let Some(threshold) = self.black_to_white_below {
            contract.black_to_white_below = threshold;
        }
    }
}

/// Tensor memory layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Nchw,
    Nhwc,
}

impl Layout {
    pub fn label(self) -> &'static str {
        match self {
            Self::Nchw => "NCHW",
            Self::Nhwc => "NHWC",
        }
    }
}

/// A tile as the model sees it: RGB after compositing and cleaning, plus
/// the in-slide region.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTile {
    pub size: u32,
    pub valid_w: u32,
    pub valid_h: u32,
    /// `size·size·3` bytes.
    pub rgb: Vec<u8>,
}

/// Turns region pixels (RGBA, `size`×`size`) into the model's view.
///
/// The host composites onto the requested background already; pixels that
/// still have alpha (older hosts) are composited here. Outside the in-slide
/// region the background colour is enforced.
pub fn prepare(
    rgba: &[u8],
    size: u32,
    valid: (u32, u32),
    contract: &Contract,
) -> Option<PreparedTile> {
    let n = size as usize * size as usize;
    if rgba.len() != n * 4 {
        return None;
    }
    let background = f32::from(contract.background_rgb());
    let mut rgb = vec![0u8; n * 3];
    let (valid_w, valid_h) = (valid.0.clamp(1, size), valid.1.clamp(1, size));
    let threshold = contract.black_to_white_below;
    for y in 0..size as usize {
        for x in 0..size as usize {
            let i = y * size as usize + x;
            let out = &mut rgb[i * 3..i * 3 + 3];
            if x >= valid_w as usize || y >= valid_h as usize {
                out.fill(contract.background_rgb());
                continue;
            }
            let px = &rgba[i * 4..i * 4 + 4];
            let alpha = px[3];
            for c in 0..3 {
                out[c] = if alpha == 255 {
                    px[c]
                } else {
                    let a = f32::from(alpha) / 255.0;
                    (f32::from(px[c]) * a + background * (1.0 - a)).round() as u8
                };
            }
            if threshold > 0 && out.iter().all(|v| *v < threshold) {
                out.fill(255);
            }
        }
    }
    Some(PreparedTile {
        size,
        valid_w,
        valid_h,
        rgb,
    })
}

/// Whether a tile has no tissue, by the rule HistoVAE's training used to
/// reject tiles (`has_content` in `data.py`): too flat (std < 5), too dark
/// (mean < 10) or too bright (mean > 245) over the in-slide pixels.
pub fn is_background(tile: &PreparedTile) -> bool {
    let size = tile.size as usize;
    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    let mut count = 0u64;
    for y in 0..tile.valid_h as usize {
        let row = &tile.rgb[y * size * 3..(y * size + tile.valid_w as usize) * 3];
        for value in row {
            let v = f64::from(*value);
            sum += v;
            sum_sq += v * v;
        }
        count += row.len() as u64;
    }
    if count == 0 {
        return true;
    }
    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64 - mean * mean).max(0.0);
    variance.sqrt() < 5.0 || !(10.0..=245.0).contains(&mean)
}

/// Writes one tile into the model's input tensor (`3·size·size` values).
pub fn write_input(tile: &PreparedTile, contract: &Contract, layout: Layout, dst: &mut [f32]) {
    let n = tile.size as usize * tile.size as usize;
    debug_assert_eq!(dst.len(), n * 3);
    let order: [usize; 3] = if contract.bgr { [2, 1, 0] } else { [0, 1, 2] };
    for i in 0..n {
        let px = &tile.rgb[i * 3..i * 3 + 3];
        for (tensor_channel, &source_channel) in order.iter().enumerate() {
            // Mean/std are declared in tensor channel order.
            let value = contract
                .input
                .encode(tensor_channel, f32::from(px[source_channel]));
            match layout {
                Layout::Nchw => dst[tensor_channel * n + i] = value,
                Layout::Nhwc => dst[i * 3 + tensor_channel] = value,
            }
        }
    }
}

/// Encodes a per-pixel error in [0, 1] into a byte, with more precision for
/// the small errors that dominate (square-root encoding).
#[inline]
pub fn encode_error(error: f32) -> u8 {
    (error.clamp(0.0, 1.0).sqrt() * 255.0).round() as u8
}

/// Inverse of [`encode_error`].
#[inline]
pub fn decode_error(code: u8) -> f32 {
    let v = f32::from(code) / 255.0;
    v * v
}

/// What one tile's reconstruction scored.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    /// Mean absolute error over the in-slide pixels and channels, in [0, 1].
    pub mae: f32,
    /// Largest per-pixel (channel-mean) absolute error, in [0, 1].
    pub max: f32,
    /// The reconstruction, `size·size·3` bytes (rounded, not truncated).
    pub recon: Vec<u8>,
    /// Per-pixel error, `size·size` bytes ([`encode_error`]).
    pub error: Vec<u8>,
}

/// Decodes one tile of model output and scores it against the input.
///
/// Errors are computed in floating point from the model output before it is
/// quantised to bytes (VAE-17), against exactly the pixels the model saw.
pub fn score_output(
    tile: &PreparedTile,
    output: &[f32],
    contract: &Contract,
    layout: Layout,
) -> Scored {
    let size = tile.size as usize;
    let n = size * size;
    debug_assert_eq!(output.len(), n * 3);
    let order: [usize; 3] = if contract.bgr { [2, 1, 0] } else { [0, 1, 2] };
    let mut recon = vec![0u8; n * 3];
    let mut error = vec![0u8; n];
    let mut sum = 0f64;
    let mut max = 0f32;
    let mut count = 0u64;
    for i in 0..n {
        let x = i % size;
        let y = i / size;
        let inside = x < tile.valid_w as usize && y < tile.valid_h as usize;
        let mut pixel_error = 0f32;
        for (tensor_channel, &channel) in order.iter().enumerate() {
            let raw = match layout {
                Layout::Nchw => output[tensor_channel * n + i],
                Layout::Nhwc => output[i * 3 + tensor_channel],
            };
            let value = if raw.is_finite() {
                contract
                    .output
                    .decode(tensor_channel, raw)
                    .clamp(0.0, 255.0)
            } else {
                0.0
            };
            recon[i * 3 + channel] = value.round() as u8;
            pixel_error += (value - f32::from(tile.rgb[i * 3 + channel])).abs();
        }
        let pixel_error = pixel_error / (3.0 * 255.0);
        if inside {
            sum += f64::from(pixel_error);
            max = max.max(pixel_error);
            count += 1;
            error[i] = encode_error(pixel_error);
        }
    }
    Scored {
        mae: if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        },
        max,
        recon,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(size: u32, rgba: [u8; 4]) -> Vec<u8> {
        rgba.repeat(size as usize * size as usize)
    }

    #[test]
    fn histovae_defaults_match_training() {
        let contract = Contract::histovae();
        // RGB 255 → 1, 0 → -1.
        assert_eq!(contract.input.encode(0, 255.0), 1.0);
        assert_eq!(contract.input.encode(0, 0.0), -1.0);
        // Output tanh range back to bytes: (x + 1)·127.5.
        assert!((contract.output.decode(0, 0.0) - 127.5).abs() < 1e-6);
        assert_eq!(contract.background_rgba(), 0xFFFF_FFFF);
    }

    #[test]
    fn preparation_composites_cleans_and_pads() {
        let contract = Contract::histovae();
        let mut rgba = solid(4, [200, 100, 50, 255]);
        // A transparent pixel, a near-black pixel.
        rgba[0..4].copy_from_slice(&[10, 10, 10, 0]);
        rgba[4..8].copy_from_slice(&[3, 2, 1, 255]);
        let tile = prepare(&rgba, 4, (3, 4), &contract).unwrap();
        assert_eq!(&tile.rgb[0..3], &[255, 255, 255], "transparent → white");
        assert_eq!(&tile.rgb[3..6], &[255, 255, 255], "near black → white");
        assert_eq!(&tile.rgb[6..9], &[200, 100, 50]);
        assert_eq!(&tile.rgb[9..12], &[255, 255, 255], "outside the slide");
        assert!(prepare(&rgba[..10], 4, (4, 4), &contract).is_none());
    }

    #[test]
    fn background_rule_matches_has_content() {
        let contract = Contract::histovae();
        let white = prepare(&solid(8, [250, 250, 250, 255]), 8, (8, 8), &contract).unwrap();
        assert!(is_background(&white));
        // `has_content` takes the std over every channel value, so only grey
        // (not merely uniform) tiles are flat.
        let flat = prepare(&solid(8, [128, 128, 128, 255]), 8, (8, 8), &contract).unwrap();
        assert!(is_background(&flat), "flat grey tiles carry no content");
        let dark = prepare(&solid(8, [6, 5, 9, 255]), 8, (8, 8), &contract).unwrap();
        assert!(is_background(&dark), "near-black becomes white, i.e. glass");
        let mut tissue = solid(8, [200, 120, 180, 255]);
        for (i, px) in tissue.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if i % 3 == 0 {
                px[..3].copy_from_slice(&[90, 40, 120]);
            }
        }
        let tissue = prepare(&tissue, 8, (8, 8), &contract).unwrap();
        assert!(!is_background(&tissue));
    }

    #[test]
    fn scoring_uses_float_outputs_and_the_valid_region() {
        let contract = Contract::histovae();
        let tile = prepare(&solid(2, [100, 100, 100, 255]), 2, (1, 2), &contract).unwrap();
        // Output exactly 101.4 everywhere (in [-1, 1] units).
        let value = 101.4 / 127.5 - 1.0;
        let output = vec![value; 12];
        let scored = score_output(&tile, &output, &contract, Layout::Nchw);
        // Only the left column is inside: |100 - 101.4| / 255.
        assert!((scored.mae - 1.4 / 255.0).abs() < 1e-5, "{}", scored.mae);
        // Truncation would give 101; rounding keeps 101 here, 102 for .6.
        assert_eq!(scored.recon[0], 101);
        assert_eq!(scored.error[1], 0, "outside pixels carry no error");
        assert!((decode_error(scored.error[0]) - 1.4 / 255.0).abs() < 2e-4);
    }

    #[test]
    fn layouts_and_channel_order_round_trip() {
        let contract = Contract {
            input: Scaling::range(ValueRange::Zero255),
            output: Scaling::range(ValueRange::Zero255),
            bgr: true,
            ..Contract::histovae()
        };
        let mut rgba = Vec::new();
        for i in 0..4u8 {
            rgba.extend_from_slice(&[10 + i, 100 + i, 200 + i, 255]);
        }
        let tile = prepare(&rgba, 2, (2, 2), &contract).unwrap();
        for layout in [Layout::Nchw, Layout::Nhwc] {
            let mut input = vec![0f32; 12];
            write_input(&tile, &contract, layout, &mut input);
            // BGR: the first tensor channel is blue.
            let first = match layout {
                Layout::Nchw => input[0],
                Layout::Nhwc => input[0],
            };
            assert_eq!(first, 200.0);
            // An identity model reproduces the tile exactly.
            let scored = score_output(&tile, &input, &contract, layout);
            assert_eq!(scored.mae, 0.0);
            assert_eq!(scored.recon, tile.rgb);
        }
    }

    #[test]
    fn imagenet_scaling_inverts() {
        let s = Scaling::IMAGENET;
        for c in 0..3 {
            let encoded = s.encode(c, 77.0);
            assert!((s.decode(c, encoded) - 77.0).abs() < 1e-3);
        }
    }

    #[test]
    fn declarations_parse_and_report_problems() {
        let mut contract = Contract::histovae();
        let values: BTreeMap<String, String> = [
            ("input_range", "[0, 1]"),
            ("output_mean", "0.5"),
            ("output_std", "0.5,0.5,0.5"),
            ("background", "black"),
            ("black_to_white", "off"),
            ("mpp", "0.5"),
            ("tile_size", "224"),
            ("channel_order", "sideways"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let (applied, problems) = contract.apply_declarations(&values);
        assert!(applied);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert_eq!(contract.input, Scaling::range(ValueRange::ZeroOne));
        assert_eq!(
            contract.output,
            Scaling::MeanStd {
                mean: [0.5; 3],
                std: [0.5; 3]
            }
        );
        assert!(!contract.composite_on_white);
        assert_eq!(contract.black_to_white_below, 0);
        assert_eq!(contract.mpp, Some(0.5));
        assert_eq!(contract.tile_size, Some(224));

        let metadata: BTreeMap<String, String> = [
            ("eovae.mpp".to_string(), "0.25".to_string()),
            ("other".into(), "x".into()),
        ]
        .into_iter()
        .collect();
        assert_eq!(Contract::metadata_declarations(&metadata).len(), 1);
        let sidecar = serde_json::json!({"mpp": 0.23, "input_mean": [0.1, 0.2, 0.3]});
        let declared = Contract::sidecar_declarations(&sidecar).unwrap();
        assert_eq!(declared["input_mean"], "0.1,0.2,0.3");
    }

    #[test]
    fn overrides_apply_on_top() {
        let mut contract = Contract::histovae();
        ContractOverride {
            input: Some(Scaling::IMAGENET),
            black_to_white_below: Some(0),
            ..Default::default()
        }
        .apply(&mut contract);
        assert_eq!(contract.input, Scaling::IMAGENET);
        assert_eq!(contract.black_to_white_below, 0);
        assert!(contract.composite_on_white);
    }
}
