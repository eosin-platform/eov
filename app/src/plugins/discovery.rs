//! Plugin directory scanning.
//!
//! Discovers plugin directories by looking for `plugin.toml` manifests in
//! immediate subdirectories of the configured plugin directory.

use plugin_api::{PluginDescriptor, PluginManifest, PluginResult};
use std::path::Path;
use tracing::{debug, info, warn};

/// Scan `plugin_dir` for plugin subdirectories containing a `plugin.toml`.
///
/// Returns descriptors sorted by plugin id for deterministic ordering.
/// Invalid plugins are skipped with a warning; they do not prevent other
/// plugins from being discovered.
pub fn discover_plugins(plugin_dir: &Path) -> Vec<PluginDescriptor> {
    if !plugin_dir.is_dir() {
        info!(
            "Plugin directory does not exist, skipping discovery: {}",
            plugin_dir.display()
        );
        return Vec::new();
    }

    let read_dir = match std::fs::read_dir(plugin_dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!(
                "Failed to read plugin directory {}: {e}",
                plugin_dir.display()
            );
            return Vec::new();
        }
    };

    let mut descriptors: Vec<PluginDescriptor> = Vec::new();

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Error reading plugin directory entry: {e}");
                continue;
            }
        };

        let path = entry.path();
        if !path.is_dir() {
            debug!("Skipping non-directory entry: {}", path.display());
            continue;
        }

        match try_load_descriptor(&path) {
            Ok(desc) => {
                info!("Discovered plugin '{}' at {}", desc.manifest.id, path.display());
                descriptors.push(desc);
            }
            Err(e) => {
                warn!("Skipping invalid plugin at {}: {e}", path.display());
            }
        }
    }

    // Sort by id for deterministic ordering
    descriptors.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    descriptors
}

/// Try to load a single plugin descriptor from a directory.
fn try_load_descriptor(plugin_root: &Path) -> PluginResult<PluginDescriptor> {
    let manifest_path = plugin_root.join(plugin_api::manifest::MANIFEST_FILENAME);
    let manifest = PluginManifest::from_file(&manifest_path)?;
    Ok(PluginDescriptor {
        root: plugin_root.to_path_buf(),
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_valid_manifest(dir: &Path, id: &str) {
        let manifest = format!(
            r#"
id = "{id}"
name = "Test Plugin {id}"
version = "0.1.0"
entry_ui = "ui/panel.slint"
entry_component = "Panel"
"#
        );
        fs::write(dir.join("plugin.toml"), manifest).unwrap();
        fs::create_dir_all(dir.join("ui")).unwrap();
        fs::write(dir.join("ui/panel.slint"), "export component Panel inherits Window {}").unwrap();
    }

    #[test]
    fn discover_from_nonexistent_dir() {
        let result = discover_plugins(Path::new("/nonexistent/plugin/dir/12345xyz"));
        assert!(result.is_empty());
    }

    #[test]
    fn discover_from_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = discover_plugins(tmp.path());
        assert!(result.is_empty());
    }

    #[test]
    fn discover_valid_plugins() {
        let tmp = tempfile::tempdir().unwrap();

        let plugin_a = tmp.path().join("plugin_a");
        fs::create_dir(&plugin_a).unwrap();
        write_valid_manifest(&plugin_a, "alpha");

        let plugin_b = tmp.path().join("plugin_b");
        fs::create_dir(&plugin_b).unwrap();
        write_valid_manifest(&plugin_b, "beta");

        let result = discover_plugins(tmp.path());
        assert_eq!(result.len(), 2);
        // Sorted by id
        assert_eq!(result[0].manifest.id, "alpha");
        assert_eq!(result[1].manifest.id, "beta");
    }

    #[test]
    fn skip_invalid_continue_valid() {
        let tmp = tempfile::tempdir().unwrap();

        // Valid plugin
        let good = tmp.path().join("good_plugin");
        fs::create_dir(&good).unwrap();
        write_valid_manifest(&good, "good");

        // Invalid plugin (empty manifest)
        let bad = tmp.path().join("bad_plugin");
        fs::create_dir(&bad).unwrap();
        fs::write(bad.join("plugin.toml"), "not valid toml {{{{").unwrap();

        // Directory without manifest
        let no_manifest = tmp.path().join("no_manifest");
        fs::create_dir(&no_manifest).unwrap();

        let result = discover_plugins(tmp.path());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].manifest.id, "good");
    }

    #[test]
    fn deterministic_ordering() {
        let tmp = tempfile::tempdir().unwrap();

        for id in ["zulu", "alpha", "mango"] {
            let dir = tmp.path().join(id);
            fs::create_dir(&dir).unwrap();
            write_valid_manifest(&dir, id);
        }

        let result = discover_plugins(tmp.path());
        let ids: Vec<&str> = result.iter().map(|d| d.manifest.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mango", "zulu"]);
    }
}
