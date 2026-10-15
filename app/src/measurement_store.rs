//! The region of interest and measurements of each slide (AUDIT §14.1),
//! kept in `<data>/measurements/<slide>.json` and keyed by slide identity,
//! so they come back when the slide is opened again, under any name.
//! `remember_measurements = false` in `config.toml` turns this off.

use crate::interaction::measure::HostMeasurement;
use eov_plugin_api::protocol::Rect;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const VERSION: u32 = 1;

/// What is kept for one slide.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Saved {
    #[serde(default)]
    pub version: u32,
    /// File name when last saved, for people reading the file.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default)]
    pub roi: Option<Rect>,
    #[serde(default)]
    pub measurements: Vec<HostMeasurement>,
}

impl Saved {
    fn is_empty(&self) -> bool {
        self.roi.is_none() && self.measurements.is_empty()
    }
}

fn store_dir() -> PathBuf {
    crate::paths::data_dir().join("measurements")
}

/// A file name for a slide identity key (`eov-slide-1:3f2a…`).
fn file_name(key: &str) -> String {
    let safe: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{safe}.json")
}

/// The saved region and measurements of slide `key`, if any.
pub(crate) fn load(key: &str) -> Option<Saved> {
    load_in(&store_dir(), key)
}

fn load_in(dir: &Path, key: &str) -> Option<Saved> {
    let path = dir.join(file_name(key));
    let text = fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<Saved>(&text) {
        Ok(saved) => Some(saved),
        Err(error) => {
            tracing::warn!("Ignoring {}: {error}", path.display());
            None
        }
    }
}

/// Keeps slide `key`'s region and measurements; with neither, forgets the
/// slide.
pub(crate) fn save(
    key: &str,
    name: &str,
    roi: Option<Rect>,
    measurements: &[HostMeasurement],
) -> Result<(), String> {
    save_in(&store_dir(), key, name, roi, measurements)
}

fn save_in(
    dir: &Path,
    key: &str,
    name: &str,
    roi: Option<Rect>,
    measurements: &[HostMeasurement],
) -> Result<(), String> {
    let path = dir.join(file_name(key));
    let saved = Saved {
        version: VERSION,
        name: name.to_string(),
        roi,
        measurements: measurements.to_vec(),
    };
    if saved.is_empty() {
        return match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.to_string()),
        };
    }
    let text = serde_json::to_string_pretty(&saved).map_err(|error| error.to_string())?;
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, text).map_err(|error| error.to_string())?;
    fs::rename(&temp, &path).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::measure::MeasureMode;
    use eov_plugin_api::protocol::Shape;

    #[test]
    fn regions_and_measurements_round_trip_per_slide() {
        let dir = tempfile::tempdir().unwrap();
        let line = HostMeasurement {
            id: "m1".into(),
            mode: MeasureMode::Line,
            shape: Shape::Polyline {
                points: vec![[0.0, 0.0], [10.0, 0.0]],
            },
        };
        let roi = Some(Rect {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        });
        let key = "eov-slide-1:abc/def";
        save_in(dir.path(), key, "a.svs", roi, std::slice::from_ref(&line)).unwrap();
        let saved = load_in(dir.path(), key).unwrap();
        assert_eq!(saved.roi, roi);
        assert_eq!(saved.measurements, vec![line]);
        assert_eq!(saved.name, "a.svs");
        assert!(load_in(dir.path(), "other").is_none());
        // Nothing left: the slide is forgotten.
        save_in(dir.path(), key, "a.svs", None, &[]).unwrap();
        assert!(load_in(dir.path(), key).is_none());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
