//! Per-plugin persisted settings (`<data>/plugins/<id>/settings.json`).

use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Total size limit of one plugin's settings, serialised.
pub const MAX_SETTINGS_BYTES: usize = 1024 * 1024;

pub struct SettingsStore {
    /// Plugin id → (file, values).
    plugins: Mutex<HashMap<String, (PathBuf, Map<String, Value>)>>,
}

impl SettingsStore {
    pub fn new() -> Self {
        Self {
            plugins: Mutex::new(HashMap::new()),
        }
    }

    /// Loads (or starts empty) the settings of `plugin_id` stored in `dir`.
    pub fn load(&self, plugin_id: &str, dir: &Path) -> Map<String, Value> {
        let path = dir.join("settings.json");
        let values = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(map)) => Some(map),
                Ok(_) | Err(_) => {
                    tracing::warn!(
                        "Ignoring unreadable plugin settings {}",
                        path.display()
                    );
                    None
                }
            })
            .unwrap_or_default();
        self.plugins
            .lock()
            .insert(plugin_id.to_string(), (path, values.clone()));
        values
    }

    pub fn get(&self, plugin_id: &str, key: &str) -> Value {
        self.plugins
            .lock()
            .get(plugin_id)
            .and_then(|(_, values)| values.get(key).cloned())
            .unwrap_or(Value::Null)
    }

    /// Sets (or with `null` removes) a value and writes the file atomically.
    pub fn set(&self, plugin_id: &str, key: &str, value: Value) -> Result<(), String> {
        if key.is_empty() || key.len() > 256 {
            return Err("setting keys must be 1–256 bytes".to_string());
        }
        let mut plugins = self.plugins.lock();
        let (path, values) = plugins
            .get_mut(plugin_id)
            .ok_or_else(|| format!("no settings store for '{plugin_id}'"))?;
        let previous = if value.is_null() {
            values.remove(key)
        } else {
            values.insert(key.to_string(), value)
        };
        let text = serde_json::to_string_pretty(&Value::Object(values.clone()))
            .map_err(|e| e.to_string())?;
        if text.len() > MAX_SETTINGS_BYTES {
            // Roll back so memory and disk stay in sync.
            match previous {
                Some(old) => {
                    values.insert(key.to_string(), old);
                }
                None => {
                    values.remove(key);
                }
            }
            return Err(format!("plugin settings would exceed {MAX_SETTINGS_BYTES} bytes"));
        }
        let path = path.clone();
        drop(plugins);
        write_atomically(&path, text.as_bytes())
    }
}

impl Default for SettingsStore {
    fn default() -> Self {
        Self::new()
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, bytes).map_err(|e| format!("{}: {e}", temp.display()))?;
    std::fs::rename(&temp, path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_persist_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let store = SettingsStore::new();
        assert!(store.load("p", dir.path()).is_empty());
        store.set("p", "threshold", Value::from(0.25)).unwrap();
        store.set("p", "gone", Value::from(true)).unwrap();
        store.set("p", "gone", Value::Null).unwrap();

        let reloaded = SettingsStore::new();
        let values = reloaded.load("p", dir.path());
        assert_eq!(values.get("threshold"), Some(&Value::from(0.25)));
        assert!(!values.contains_key("gone"));
        assert_eq!(reloaded.get("p", "threshold"), Value::from(0.25));
        assert!(reloaded.set("unknown", "k", Value::from(1)).is_err());
    }

    #[test]
    fn oversized_settings_are_rejected_without_corrupting_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = SettingsStore::new();
        store.load("p", dir.path());
        let huge = Value::from("x".repeat(MAX_SETTINGS_BYTES + 1));
        assert!(store.set("p", "blob", huge).is_err());
        assert_eq!(store.get("p", "blob"), Value::Null);
    }
}
