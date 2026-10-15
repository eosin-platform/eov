//! What an analysis run is: the full configuration that determines its
//! results. Its hash is the cache key (VAE-12) and it is stored with the
//! results as a provenance manifest (§14.5).

use crate::contract::{CONTRACT_VERSION, Contract, Layout};
use crate::engine::EngineInfo;
use crate::model::ModelInfo;
use crate::tiling::TileGrid;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Everything that changes the results of a run. No file paths (PHI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    pub plugin_version: String,
    pub contract_version: u32,
    /// SHA-256 of the model and its external weights.
    pub model_sha256: String,
    /// File name only, for people reading the manifest.
    pub model_file: String,
    pub input: String,
    pub output: String,
    pub layout: Layout,
    pub tile_size: u32,
    pub contract: Contract,
    pub level: u32,
    pub downsample: f64,
    /// µm per pixel of the analysed level, if calibrated (informative).
    pub level_mpp: Option<f64>,
    pub device: String,
    pub precision: String,
}

impl RunSpec {
    pub fn new(
        model: &ModelInfo,
        output: &str,
        contract: &Contract,
        grid: &TileGrid,
        level_mpp: Option<f64>,
        engine: &EngineInfo,
    ) -> Self {
        Self {
            plugin_version: env!("CARGO_PKG_VERSION").to_string(),
            contract_version: CONTRACT_VERSION,
            model_sha256: model.sha256.clone(),
            model_file: model.file_name.clone(),
            input: model.input.name.clone(),
            output: output.to_string(),
            layout: model.layout,
            tile_size: grid.tile,
            contract: contract.clone(),
            level: grid.level,
            downsample: grid.downsample,
            level_mpp,
            device: engine.device.key().to_string(),
            precision: engine.precision.clone(),
        }
    }

    /// The cache key: a hash of everything above except informative fields.
    pub fn key(&self) -> String {
        let keyed = serde_json::json!({
            "plugin": self.plugin_version,
            "contract_version": self.contract_version,
            "model": self.model_sha256,
            "input": self.input,
            "output": self.output,
            "layout": self.layout,
            "tile": self.tile_size,
            "contract": self.contract,
            "level": self.level,
            "downsample": format!("{:.6}", self.downsample),
            "device": self.device,
            "precision": self.precision,
        });
        let digest = Sha256::digest(keyed.to_string().as_bytes());
        digest[..16].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The manifest stored with results and written next to exports.
    pub fn manifest(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Device;
    use crate::tiling::{SlideGeometry, TileGrid};
    use eov_plugin_api::protocol::FileInfo;

    fn spec() -> RunSpec {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width: 1000,
            height: 1000,
            ..Default::default()
        });
        let grid = TileGrid::new(&geometry, &geometry.levels[0], 256);
        RunSpec {
            plugin_version: "1.0.0".into(),
            contract_version: CONTRACT_VERSION,
            model_sha256: "abc".into(),
            model_file: "m.onnx".into(),
            input: "input".into(),
            output: "reconstruction".into(),
            layout: Layout::Nchw,
            tile_size: grid.tile,
            contract: Contract::histovae(),
            level: grid.level,
            downsample: grid.downsample,
            level_mpp: Some(0.25),
            device: Device::Cpu.key().into(),
            precision: "fp32".into(),
        }
    }

    #[test]
    fn keys_cover_everything_that_changes_results() {
        let base = spec();
        let key = base.key();
        assert_eq!(key.len(), 32);
        let variants = [
            RunSpec {
                model_sha256: "def".into(),
                ..base.clone()
            },
            RunSpec {
                output: "other".into(),
                ..base.clone()
            },
            RunSpec {
                level: 1,
                ..base.clone()
            },
            RunSpec {
                device: "cuda".into(),
                ..base.clone()
            },
            RunSpec {
                precision: "tf32".into(),
                ..base.clone()
            },
            RunSpec {
                plugin_version: "1.1.0".into(),
                ..base.clone()
            },
            RunSpec {
                contract: Contract {
                    black_to_white_below: 0,
                    ..Contract::histovae()
                },
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(variant.key(), key, "{variant:?}");
        }
        // Informative fields do not split the cache.
        let renamed = RunSpec {
            model_file: "renamed.onnx".into(),
            level_mpp: None,
            ..base.clone()
        };
        assert_eq!(renamed.key(), key);
        assert!(!base.manifest().contains('/'), "no paths in the manifest");
    }
}
