//! The stain normalisation target (AUDIT §14.5, CLR-1): what normalised
//! slides are made to look like. By default it is Macenko's reference H&E;
//! a slide or a region of one can become the target ("normalise to this
//! slide"), and a target can be saved to a file, shared and loaded again,
//! for the viewer and for `eov dataset patches --stain-target`.
//!
//! The target in use is kept in `<data>/stain-target.json` and applies to
//! every pane with stain normalisation on. A target file is the
//! [`StainEstimate`] it came from with a name and when it was made:
//!
//! ```json
//! {"format": "eov-stain-target", "version": 1, "name": "Reference H&E (S21-123)",
//!  "created_unix": 1792000000, "estimate": { ...a StainEstimate... }}
//! ```
//!
//! A bare `StainEstimate` file (as `StainEstimate::save_json` writes) loads
//! too, named after its file.

use eov_common::stain::{NormalizationTarget, StainEstimate};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

const FORMAT: &str = "eov-stain-target";
const VERSION: u32 = 1;

/// A named target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StainTarget {
    pub name: String,
    pub created_unix: u64,
    pub estimate: StainEstimate,
}

#[derive(Serialize, Deserialize)]
struct TargetFile {
    format: String,
    version: u32,
    name: String,
    #[serde(default)]
    created_unix: u64,
    estimate: StainEstimate,
}

impl StainTarget {
    pub(crate) fn new(name: impl Into<String>, estimate: StainEstimate) -> Self {
        Self {
            name: name.into(),
            created_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default(),
            estimate,
        }
    }

    /// The appearance normalisation maps slides onto.
    pub(crate) fn normalization_target(&self) -> NormalizationTarget {
        NormalizationTarget::from_estimate(&self.estimate)
    }

    pub(crate) fn to_json(&self) -> String {
        serde_json::to_string_pretty(&TargetFile {
            format: FORMAT.to_string(),
            version: VERSION,
            name: self.name.clone(),
            created_unix: self.created_unix,
            estimate: self.estimate.clone(),
        })
        .expect("stain targets are serialisable")
    }

    /// A target file, or a bare stain estimate named `fallback_name`.
    pub(crate) fn from_json(text: &str, fallback_name: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("not a stain target: {e}"))?;
        if value.get("format").and_then(|f| f.as_str()) == Some(FORMAT) {
            let file: TargetFile =
                serde_json::from_value(value).map_err(|e| format!("not a stain target: {e}"))?;
            if file.version > VERSION {
                return Err(format!(
                    "the stain target was saved by a newer eov (format {})",
                    file.version
                ));
            }
            // Validates the vectors and concentrations.
            let estimate = StainEstimate::from_json(
                &serde_json::to_string(&file.estimate).unwrap_or_default(),
            )
            .map_err(|e| e.to_string())?;
            return Ok(Self {
                name: file.name,
                created_unix: file.created_unix,
                estimate,
            });
        }
        let estimate = StainEstimate::from_json(text).map_err(|e| e.to_string())?;
        Ok(Self::new(fallback_name, estimate))
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, self.to_json()).map_err(|e| e.to_string())?;
        std::fs::rename(&temp, path).map_err(|e| e.to_string())
    }

    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Stain target".to_string());
        Self::from_json(&text, &name)
    }
}

fn store_path() -> PathBuf {
    crate::paths::data_dir().join("stain-target.json")
}

/// The target in use (`None`: Macenko's reference), read once.
static CURRENT: RwLock<Option<Option<Arc<StainTarget>>>> = RwLock::new(None);

/// The target in use, `None` for Macenko's reference H&E.
pub(crate) fn current() -> Option<Arc<StainTarget>> {
    if let Some(current) = CURRENT.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return current.clone();
    }
    let loaded = match StainTarget::load(&store_path()) {
        Ok(target) => Some(Arc::new(target)),
        Err(error) => {
            if store_path().exists() {
                tracing::warn!("The saved stain target cannot be used: {error}");
            }
            None
        }
    };
    *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = Some(loaded.clone());
    loaded
}

/// Makes `target` the one in use (`None`: back to the reference) and
/// remembers it.
pub(crate) fn set(target: Option<StainTarget>) -> Result<(), String> {
    let path = store_path();
    match &target {
        Some(target) => target.save(&path)?,
        None => {
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            }
        }
    }
    *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = Some(target.map(Arc::new));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::stain::{StainEstimateOptions, StainFitMethod, estimate_stain_from_rgba};

    fn estimate() -> StainEstimate {
        // Haematoxylin and eosin pixels on glass.
        let mut rgba = Vec::new();
        for i in 0..20_000u32 {
            let pixel = match i % 3 {
                0 => [90, 60, 150, 255],
                1 => [220, 120, 180, 255],
                _ => [245, 245, 245, 255],
            };
            rgba.extend_from_slice(&pixel);
        }
        estimate_stain_from_rgba(
            &[&rgba],
            StainFitMethod::Macenko,
            &StainEstimateOptions::default(),
        )
    }

    #[test]
    fn targets_round_trip_and_bare_estimates_load() {
        let dir = tempfile::tempdir().unwrap();
        let target = StainTarget::new("Reference H&E", estimate());
        let path = dir.path().join("reference.json");
        target.save(&path).unwrap();
        let loaded = StainTarget::load(&path).unwrap();
        assert_eq!(loaded, target);
        assert_eq!(
            loaded.normalization_target().basis,
            target.estimate.basis,
            "the target is the estimate's appearance"
        );
        // A bare estimate is named after its file.
        let bare = dir.path().join("site-b.json");
        target.estimate.save_json(&bare).unwrap();
        assert_eq!(StainTarget::load(&bare).unwrap().name, "site-b");
        // Newer formats and garbage are refused.
        let newer = target.to_json().replace("\"version\": 1", "\"version\": 9");
        assert!(
            StainTarget::from_json(&newer, "x")
                .unwrap_err()
                .contains("newer")
        );
        assert!(StainTarget::from_json("{}", "x").is_err());
    }

    #[test]
    fn the_target_in_use_is_remembered() {
        crate::paths::isolate_for_tests();
        set(None).unwrap();
        assert!(current().is_none());
        set(Some(StainTarget::new("Mine", estimate()))).unwrap();
        assert_eq!(current().unwrap().name, "Mine");
        // Read back from the file, as at the next start.
        *CURRENT.write().unwrap() = None;
        assert_eq!(current().unwrap().name, "Mine");
        set(None).unwrap();
        assert!(current().is_none());
    }
}
