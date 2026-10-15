//! `dataset.json`: the run manifest (parameters, software, status, totals),
//! used to make resumed runs safe (DS-3).

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::config::{DatasetPatchesConfig, DatasetPatchesOptions};

pub(crate) const MANIFEST_FILE: &str = "dataset.json";
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// State of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Running,
    Complete,
    /// Finished, but some slides or tiles failed.
    Partial,
    Cancelled,
    Failed,
}

/// Everything that determines which tiles a run writes and what they
/// contain. Two runs with equal parameters produce the same files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionParameters {
    pub tile_size: u32,
    pub stride: u32,
    pub white_threshold: Option<f32>,
    pub resolution: Option<super::config::TargetResolution>,
    pub tissue_filter: Option<super::config::TissueFilter>,
    pub stain_normalization: Option<super::config::PatchNormalization>,
    pub annotations: Option<super::config::AnnotationOptions>,
    pub max_tiles_per_slide: Option<u64>,
    pub seed: Option<u64>,
    pub image_format: super::config::PatchImageFormat,
    /// Slides are named by opaque ids; no paths or file names are written.
    #[serde(default)]
    pub deidentified: bool,
}

impl ExtractionParameters {
    pub fn new(config: &DatasetPatchesConfig, options: &DatasetPatchesOptions) -> Self {
        let mut annotations = options.annotations.clone();
        if options.deidentify
            && let Some(annotations) = &mut annotations
        {
            // Where the annotations came from names the slides.
            annotations.source = super::config::AnnotationSource::Sidecar;
        }
        Self {
            tile_size: config.tile_size,
            stride: config.stride,
            white_threshold: config.white_threshold,
            resolution: options.resolution,
            tissue_filter: options.tissue_filter,
            stain_normalization: options.stain_normalization.clone(),
            annotations,
            max_tiles_per_slide: options.max_tiles_per_slide,
            seed: options.seed,
            image_format: options.image_format,
            deidentified: options.deidentify,
        }
    }
}

/// Run totals (also in the report).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunTotals {
    pub slides: u64,
    pub slides_failed: u64,
    pub tiles_written: u64,
    pub tiles_existing: u64,
    pub tiles_skipped_white: u64,
    pub tiles_skipped_tissue: u64,
    pub tiles_skipped_annotation: u64,
    pub tiles_failed: u64,
}

/// Contents of `dataset.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub software: String,
    pub parameters: ExtractionParameters,
    pub inputs: Vec<String>,
    pub status: RunStatus,
    /// Seconds since the Unix epoch.
    pub started: u64,
    pub updated: u64,
    pub totals: RunTotals,
    /// Annotation class indices used in multiclass masks (index = position
    /// + 1; 0 is background).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
}

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl DatasetManifest {
    pub fn load(dir: &Path) -> std::io::Result<Option<Self>> {
        let path = dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        serde_json::from_str(&text).map(Some).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }

    /// Writes the manifest atomically.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        let path = dir.join(MANIFEST_FILE);
        let tmp = dir.join(format!("{MANIFEST_FILE}.part"));
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&tmp, json + "\n")?;
        std::fs::rename(&tmp, &path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn manifests_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let config = DatasetPatchesConfig {
            inputs: vec![PathBuf::from("a.svs")],
            output_dir: dir.path().to_path_buf(),
            tile_size: 256,
            stride: 256,
            metadata_format: None,
            threads: 2,
            white_threshold: Some(0.8),
        };
        let manifest = DatasetManifest {
            format_version: MANIFEST_FORMAT_VERSION,
            software: "eov".into(),
            parameters: ExtractionParameters::new(&config, &DatasetPatchesOptions::default()),
            inputs: vec!["a.svs".into()],
            status: RunStatus::Running,
            started: now(),
            updated: now(),
            totals: RunTotals::default(),
            classes: vec!["Tumor".into()],
        };
        manifest.save(dir.path()).unwrap();
        assert_eq!(DatasetManifest::load(dir.path()).unwrap(), Some(manifest));
        assert_eq!(
            DatasetManifest::load(&dir.path().join("missing")).unwrap(),
            None
        );
    }
}
