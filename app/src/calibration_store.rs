//! Manual pixel sizes for slides that do not record one, or record a wrong
//! one (AUDIT §14.1), kept in `<data>/calibrations.toml` and keyed by slide
//! identity, so they survive renames and moves.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default)]
    slide: BTreeMap<String, Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    /// Micrometres per level-0 pixel (both axes).
    um_per_px: f64,
    /// File name when it was set, for people reading the file.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    name: String,
}

fn store_path() -> PathBuf {
    crate::paths::calibrations_file()
}

fn read(path: &Path) -> Store {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

fn write(path: &Path, store: &Store) -> Result<(), String> {
    let text = toml::to_string(store).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text).map_err(|error| error.to_string())?;
    fs::rename(&temp, path).map_err(|error| error.to_string())
}

/// The manual pixel size of the slide with identity `key`, if one was set.
pub(crate) fn lookup(key: &str) -> Option<f64> {
    lookup_in(&store_path(), key)
}

fn lookup_in(path: &Path, key: &str) -> Option<f64> {
    read(path)
        .slide
        .get(key)
        .map(|entry| entry.um_per_px)
        .filter(|um| um.is_finite() && *um > 0.0)
}

/// Sets (or with `None` clears) the manual pixel size of slide `key`.
pub(crate) fn store(key: &str, um_per_px: Option<f64>, name: &str) -> Result<(), String> {
    store_in(&store_path(), key, um_per_px, name)
}

fn store_in(path: &Path, key: &str, um_per_px: Option<f64>, name: &str) -> Result<(), String> {
    let mut store = read(path);
    match um_per_px {
        Some(um) if um.is_finite() && um > 0.0 => {
            store.slide.insert(
                key.to_string(),
                Entry {
                    um_per_px: um,
                    name: name.to_string(),
                },
            );
        }
        Some(_) => return Err("the pixel size must be a positive number".to_string()),
        None => {
            store.slide.remove(key);
        }
    }
    write(path, &store)
}

/// Parses a typed pixel size in µm/px ("0.25", "0,25", "0.25 µm").
pub(crate) fn parse_um_per_px(text: &str) -> Option<f64> {
    let cleaned: String = text
        .trim()
        .trim_end_matches("µm/px")
        .trim_end_matches("um/px")
        .trim_end_matches("µm")
        .trim_end_matches("um")
        .trim()
        .replace(',', ".");
    cleaned
        .parse::<f64>()
        .ok()
        .filter(|um| um.is_finite() && *um > 0.0 && *um < 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrations_persist_by_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibrations.toml");
        assert_eq!(lookup_in(&path, "abc"), None);
        store_in(&path, "abc", Some(0.25), "slide.tif").unwrap();
        assert_eq!(lookup_in(&path, "abc"), Some(0.25));
        assert!(store_in(&path, "abc", Some(-1.0), "slide.tif").is_err());
        store_in(&path, "abc", None, "slide.tif").unwrap();
        assert_eq!(lookup_in(&path, "abc"), None);
    }

    #[test]
    fn typed_pixel_sizes_parse_leniently() {
        assert_eq!(parse_um_per_px("0.25"), Some(0.25));
        assert_eq!(parse_um_per_px(" 0,5 µm/px "), Some(0.5));
        assert_eq!(parse_um_per_px("0.2528 um"), Some(0.2528));
        assert_eq!(parse_um_per_px("0"), None);
        assert_eq!(parse_um_per_px("abc"), None);
    }
}
