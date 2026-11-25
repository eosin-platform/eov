//! eovae's settings, kept in the host's per-plugin settings store (no files
//! in the user's config directory, VAE-16).
//!
//! Stored as one `config` object. Values are validated when read, so a
//! damaged or hand-edited value falls back to its default instead of
//! blocking later saves (VAE-17), and writes are debounced and made from
//! the presenter thread, never on the UI thread for every keystroke.

use crate::contract::ContractOverride;
use crate::engine::DevicePreference;
use crate::model::ModelInfo;
use crate::tiling::Resolution;
use eov_plugin_api::i18n::{self, tr};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The settings key.
pub const SETTINGS_KEY: &str = "config";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sections {
    pub model: bool,
    pub analysis: bool,
    pub results: bool,
    pub cache: bool,
}

impl Default for Sections {
    fn default() -> Self {
        Self {
            model: true,
            analysis: true,
            results: true,
            cache: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The model to load at start-up.
    pub model_path: Option<String>,
    /// A user-chosen ONNX Runtime library or folder.
    pub runtime_path: Option<String>,
    pub device: DevicePreference,
    /// Inference threads; 0 = every core.
    pub threads: usize,
    /// Batch size for models without a fixed batch.
    pub batch_size: usize,
    /// Allow TF32 on CUDA.
    pub tf32: bool,
    pub resolution: Resolution,
    pub skip_background: bool,
    /// Analyse visible tiles automatically while an overlay is shown.
    pub auto_analyze: bool,
    /// Keep reconstructions in the disk cache (large).
    pub keep_pixels: bool,
    pub cache_limit_mb: u64,
    pub memory_limit_mb: u64,
    pub sections: Sections,
    /// Chosen output per model (by SHA-256).
    pub outputs: BTreeMap<String, String>,
    /// Preprocessing overrides per model (by SHA-256).
    pub overrides: BTreeMap<String, ContractOverride>,
    /// Where the site keeps the model "Download" fetches: the `.onnx` file
    /// (the reference model's weights come from the same folder).
    pub download_url: Option<String>,
    /// SHA-256 of the site's own model at `download_url`; `None` downloads
    /// the reference model, checked against its card.
    pub download_sha256: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: None,
            runtime_path: None,
            download_url: None,
            download_sha256: None,
            device: DevicePreference::Auto,
            threads: 0,
            batch_size: 16,
            tf32: true,
            resolution: Resolution::Auto,
            skip_background: true,
            auto_analyze: true,
            keep_pixels: false,
            cache_limit_mb: 2048,
            memory_limit_mb: 512,
            sections: Sections::default(),
            outputs: BTreeMap::new(),
            overrides: BTreeMap::new(),
        }
    }
}

pub fn max_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

impl Config {
    /// Reads the stored settings. Settings that cannot be read fall back to
    /// the defaults, with the reason (serde's text, in English).
    pub fn from_settings(settings: &Map<String, Value>) -> (Self, Option<String>) {
        let Some(value) = settings.get(SETTINGS_KEY) else {
            return (Self::default(), None);
        };
        match serde_json::from_value::<Config>(value.clone()) {
            Ok(config) => (config.sanitized(), None),
            Err(error) => (Self::default(), Some(error.to_string())),
        }
    }

    /// Clamps every value into its valid range.
    pub fn sanitized(mut self) -> Self {
        self.threads = self.threads.min(max_threads());
        self.batch_size = self.batch_size.clamp(1, 256);
        self.cache_limit_mb = self.cache_limit_mb.clamp(64, 1 << 20);
        self.memory_limit_mb = self.memory_limit_mb.clamp(1, 1 << 16);
        if let Resolution::Downsample { value } = self.resolution
            && !(value.is_finite() && (1.0..=1e6).contains(&value))
        {
            self.resolution = Resolution::Auto;
        }
        self.model_path = self.model_path.filter(|p| !p.trim().is_empty());
        self.runtime_path = self.runtime_path.filter(|p| !p.trim().is_empty());
        self.download_url = Self::parse_download_url(self.download_url.as_deref().unwrap_or(""));
        self.download_sha256 = self
            .download_sha256
            .as_deref()
            .and_then(|text| Self::parse_sha256(text).ok())
            .flatten();
        self
    }

    /// A typed download address: trimmed, `None` when empty.
    pub fn parse_download_url(text: &str) -> Option<String> {
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// A typed SHA-256: `Ok(None)` when empty, lower case when valid.
    pub fn parse_sha256(text: &str) -> Result<Option<String>, String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(None);
        }
        if eov_plugin_api::protocol::is_sha256_hex(text) {
            Ok(Some(text.to_ascii_lowercase()))
        } else {
            Err(tr!(
                // Translators: {} is what was typed into the SHA-256 field; it is
                // ignored and the field keeps its previous value.
                "'{}' is not a SHA-256 (64 hex digits) and was not saved.",
                i18n::isolate(text)
            ))
        }
    }

    pub fn effective_threads(&self) -> usize {
        if self.threads == 0 {
            max_threads()
        } else {
            self.threads
        }
    }

    pub fn cache_limit_bytes(&self) -> u64 {
        self.cache_limit_mb * 1024 * 1024
    }

    pub fn memory_limit_bytes(&self) -> usize {
        (self.memory_limit_mb * 1024 * 1024) as usize
    }

    /// The output to read for `model` (the saved choice if still valid).
    pub fn output_for(&self, model: &ModelInfo) -> String {
        self.outputs
            .get(&model.sha256)
            .filter(|name| model.output(name).is_some())
            .cloned()
            .unwrap_or_else(|| model.default_output_name().to_string())
    }

    /// The model's contract with the user's overrides.
    pub fn contract_for(&self, model: &ModelInfo) -> crate::contract::Contract {
        let mut contract = model.contract.clone();
        if let Some(overrides) = self.overrides.get(&model.sha256) {
            overrides.apply(&mut contract);
        }
        contract
    }

    /// Parses a typed thread count (0 or "auto": as many as the CPU has);
    /// empty or invalid text keeps `current` (clearing the field used to
    /// persist 1, VAE-17).
    pub fn parse_threads(text: &str, current: usize) -> usize {
        match text.trim() {
            "" => current,
            "auto" | "Auto" => 0,
            other => typed_count(other)
                .map(|n| n.min(max_threads()))
                .unwrap_or(current),
        }
    }

    /// Parses a typed batch size; invalid text keeps `current`.
    pub fn parse_batch(text: &str, current: usize) -> usize {
        typed_count(text)
            .map(|n| n.clamp(1, 256))
            .unwrap_or(current)
    }
}

/// A whole number a user typed, in Western, Arabic-Indic or Persian digits
/// (as their keyboards type them, see [`i18n::parse_decimal`]); `None` for
/// anything else, such as a fraction or a negative number.
fn typed_count(text: &str) -> Option<usize> {
    let value = i18n::parse_decimal(text)?;
    // Too large a number saturates; the callers clamp it.
    (value >= 0.0 && value.fract() == 0.0).then_some(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_and_damaged_settings_fall_back_to_defaults() {
        let (config, warning) = Config::from_settings(&Map::new());
        assert_eq!(config, Config::default());
        assert!(warning.is_none());
        let mut settings = Map::new();
        settings.insert(SETTINGS_KEY.into(), json!({"device": "warp-drive"}));
        let (config, warning) = Config::from_settings(&settings);
        assert_eq!(config, Config::default());
        assert!(warning.unwrap().contains("warp-drive"), "says why");
    }

    #[test]
    fn values_are_clamped_and_unknown_keys_ignored() {
        let mut settings = Map::new();
        settings.insert(
            SETTINGS_KEY.into(),
            json!({
                "batch_size": 100000,
                "threads": 100000,
                "resolution": {"mode": "downsample", "value": -3.0},
                "model_path": "  ",
                "from_the_future": true
            }),
        );
        let (config, warning) = Config::from_settings(&settings);
        assert!(warning.is_none());
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.threads, max_threads());
        assert_eq!(config.resolution, Resolution::Auto);
        assert_eq!(config.model_path, None);
    }

    #[test]
    fn typed_numbers_never_persist_garbage() {
        assert_eq!(Config::parse_threads("", 6), 6, "clearing keeps the value");
        assert_eq!(Config::parse_threads("abc", 6), 6);
        assert_eq!(Config::parse_threads("auto", 6), 0);
        assert_eq!(Config::parse_threads("0", 6), 0);
        assert_eq!(Config::parse_threads("2", 6), 2.min(max_threads()));
        assert_eq!(Config::parse_threads("2.5", 6), 6);
        assert_eq!(Config::parse_threads("-2", 6), 6);
        assert_eq!(Config::parse_threads("100000", 6), max_threads());
        assert_eq!(Config::parse_batch("0", 16), 1);
        assert_eq!(Config::parse_batch("x", 16), 16);
        assert_eq!(Config::parse_batch(" 64 ", 16), 64);
        assert_eq!(Config::parse_batch("100000", 16), 256);
        // Persian and Arabic-Indic digits, as their keyboards type them.
        assert_eq!(Config::parse_threads("۲", 6), 2.min(max_threads()));
        assert_eq!(Config::parse_threads("۰", 6), 0);
        assert_eq!(Config::parse_batch("۶۴", 16), 64);
        assert_eq!(Config::parse_batch("٣٢", 16), 32);
    }

    #[test]
    fn download_settings_are_checked() {
        let mut settings = Map::new();
        settings.insert(
            SETTINGS_KEY.into(),
            json!({
                "download_url": "  https://models.example.org/vae.onnx ",
                "download_sha256": "not a hash"
            }),
        );
        let (config, _) = Config::from_settings(&settings);
        assert_eq!(
            config.download_url.as_deref(),
            Some("https://models.example.org/vae.onnx")
        );
        assert_eq!(config.download_sha256, None);
        let upper = "A".repeat(64);
        assert_eq!(Config::parse_sha256(&upper), Ok(Some("a".repeat(64))));
        assert_eq!(Config::parse_sha256(" "), Ok(None));
        assert!(Config::parse_sha256("abc").is_err());
        assert_eq!(Config::parse_download_url("  "), None);
    }

    #[test]
    fn round_trips_through_json() {
        let config = Config {
            device: DevicePreference::Cuda,
            resolution: Resolution::Downsample { value: 4.0 },
            ..Default::default()
        };
        let value = serde_json::to_value(&config).unwrap();
        let mut settings = Map::new();
        settings.insert(SETTINGS_KEY.into(), value);
        assert_eq!(Config::from_settings(&settings).0, config);
    }
}
