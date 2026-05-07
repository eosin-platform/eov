//! Plugin manifest parsing and validation.

use crate::{IconDescriptor, PluginError, PluginResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A toolbar button declared in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestToolbarButton {
    pub button_id: String,
    pub tooltip: String,
    pub action_id: String,
    /// Inline SVG icon data. If omitted, the plugin's top-level icon is used.
    pub icon_svg: Option<String>,
}

/// The parsed contents of a `plugin.toml` manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    /// Unique, stable identifier for the plugin (e.g. `"example_plugin"`).
    pub id: String,
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Relative path to the `.slint` UI file (from the plugin root).
    /// Optional — plugins that are pure viewport filters may omit this.
    #[serde(default)]
    pub entry_ui: Option<String>,
    /// Name of the exported Slint component inside `entry_ui`.
    /// Optional — plugins that are pure viewport filters may omit this.
    #[serde(default)]
    pub entry_component: Option<String>,
    /// Optional icon for the toolbar button.
    pub icon: Option<IconDescriptor>,
    /// Toolbar buttons declared in the manifest.
    #[serde(default)]
    pub toolbar_buttons: Vec<ManifestToolbarButton>,
}

/// Name of the manifest file inside each plugin directory.
pub const MANIFEST_FILENAME: &str = "plugin.toml";

impl PluginManifest {
    /// Parse a manifest from a TOML string, validating required fields.
    pub fn from_toml(toml_str: &str, plugin_id_hint: &str) -> PluginResult<Self> {
        let manifest: Self = toml::from_str(toml_str).map_err(|e| PluginError::Manifest {
            plugin_id: plugin_id_hint.to_string(),
            message: format!("TOML parse error: {e}"),
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Load a manifest from a file path.
    pub fn from_file(path: &Path) -> PluginResult<Self> {
        let dir_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");
        let contents = std::fs::read_to_string(path).map_err(|e| PluginError::Manifest {
            plugin_id: dir_name.to_string(),
            message: format!("failed to read {}: {e}", path.display()),
        })?;
        Self::from_toml(&contents, dir_name)
    }

    /// Validate semantic constraints beyond TOML structure.
    fn validate(&self) -> PluginResult<()> {
        if self.id.is_empty() {
            return Err(PluginError::Manifest {
                plugin_id: self.id.clone(),
                message: "'id' must not be empty".into(),
            });
        }
        if self.name.is_empty() {
            return Err(PluginError::Manifest {
                plugin_id: self.id.clone(),
                message: "'name' must not be empty".into(),
            });
        }
        if self.version.is_empty() {
            return Err(PluginError::Manifest {
                plugin_id: self.id.clone(),
                message: "'version' must not be empty".into(),
            });
        }
        if let Some(ref entry_ui) = self.entry_ui {
            if entry_ui.is_empty() {
                return Err(PluginError::Manifest {
                    plugin_id: self.id.clone(),
                    message: "'entry_ui' must not be empty when specified".into(),
                });
            }
            // entry_ui must be a relative path
            if Path::new(entry_ui).is_absolute() {
                return Err(PluginError::Manifest {
                    plugin_id: self.id.clone(),
                    message: format!("'entry_ui' must be a relative path, got '{entry_ui}'"),
                });
            }
            // Reject path traversal
            if entry_ui.contains("..") {
                return Err(PluginError::Manifest {
                    plugin_id: self.id.clone(),
                    message: format!("'entry_ui' must not contain '..', got '{entry_ui}'"),
                });
            }
        }
        if let Some(ref entry_component) = self.entry_component
            && entry_component.is_empty()
        {
            return Err(PluginError::Manifest {
                plugin_id: self.id.clone(),
                message: "'entry_component' must not be empty when specified".into(),
            });
        }
        // Validate icon file path if present
        if let Some(IconDescriptor::File { path }) = &self.icon {
            if path.is_absolute() {
                return Err(PluginError::Manifest {
                    plugin_id: self.id.clone(),
                    message: format!("icon file path must be relative, got '{}'", path.display()),
                });
            }
            if path.to_string_lossy().contains("..") {
                return Err(PluginError::Manifest {
                    plugin_id: self.id.clone(),
                    message: format!(
                        "icon file path must not contain '..', got '{}'",
                        path.display()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Resolve the `entry_ui` to an absolute path given the plugin root.
    /// Returns `None` if `entry_ui` is not set.
    pub fn resolve_entry_ui(&self, plugin_root: &Path) -> Option<PathBuf> {
        self.entry_ui.as_ref().map(|ui| plugin_root.join(ui))
    }

    /// Validate that referenced files actually exist on disk.
    pub fn validate_files(&self, plugin_root: &Path) -> PluginResult<()> {
        if let Some(ui_path) = self.resolve_entry_ui(plugin_root)
            && !ui_path.exists()
        {
            return Err(PluginError::MissingFile {
                plugin_id: self.id.clone(),
                path: ui_path,
            });
        }
        if let Some(IconDescriptor::File { path }) = &self.icon {
            let icon_path = plugin_root.join(path);
            if !icon_path.exists() {
                return Err(PluginError::MissingFile {
                    plugin_id: self.id.clone(),
                    path: icon_path,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_TOML: &str = r#"
id = "test_plugin"
name = "Test Plugin"
version = "1.0.0"
entry_ui = "ui/panel.slint"
entry_component = "Panel"

[icon]
kind = "svg"
data = "<svg/>"
"#;

    #[test]
    fn parse_valid_manifest() {
        let m = PluginManifest::from_toml(VALID_TOML, "test").unwrap();
        assert_eq!(m.id, "test_plugin");
        assert_eq!(m.name, "Test Plugin");
        assert_eq!(m.version, "1.0.0");
        assert_eq!(m.entry_ui.as_deref(), Some("ui/panel.slint"));
        assert_eq!(m.entry_component.as_deref(), Some("Panel"));
        assert_eq!(
            m.icon,
            Some(IconDescriptor::Svg {
                data: "<svg/>".into()
            })
        );
    }

    #[test]
    fn reject_missing_id() {
        let toml = r#"
name = "Test"
version = "1.0.0"
entry_ui = "ui/p.slint"
entry_component = "P"
"#;
        let err = PluginManifest::from_toml(toml, "hint").unwrap_err();
        assert!(err.to_string().contains("TOML parse error"));
    }

    #[test]
    fn reject_empty_id() {
        let toml = r#"
id = ""
name = "Test"
version = "1.0.0"
entry_ui = "ui/p.slint"
entry_component = "P"
"#;
        let err = PluginManifest::from_toml(toml, "hint").unwrap_err();
        assert!(err.to_string().contains("'id' must not be empty"));
    }

    #[test]
    fn reject_absolute_entry_ui() {
        let toml = r#"
id = "abs"
name = "Test"
version = "1.0.0"
entry_ui = "/etc/evil.slint"
entry_component = "Evil"
"#;
        let err = PluginManifest::from_toml(toml, "hint").unwrap_err();
        assert!(err.to_string().contains("relative path"));
    }

    #[test]
    fn reject_path_traversal_in_entry_ui() {
        let toml = r#"
id = "trav"
name = "Test"
version = "1.0.0"
entry_ui = "../escape/evil.slint"
entry_component = "Evil"
"#;
        let err = PluginManifest::from_toml(toml, "hint").unwrap_err();
        assert!(err.to_string().contains(".."));
    }

    #[test]
    fn resolve_entry_ui_relative_to_root() {
        let m = PluginManifest::from_toml(VALID_TOML, "test").unwrap();
        let resolved = m.resolve_entry_ui(Path::new("/plugins/test_plugin"));
        assert_eq!(
            resolved,
            Some(PathBuf::from("/plugins/test_plugin/ui/panel.slint"))
        );
    }

    #[test]
    fn validate_files_missing_ui() {
        let m = PluginManifest::from_toml(VALID_TOML, "test").unwrap();
        let err = m
            .validate_files(Path::new("/nonexistent/plugin/root"))
            .unwrap_err();
        match err {
            PluginError::MissingFile { plugin_id, .. } => {
                assert_eq!(plugin_id, "test_plugin");
            }
            other => panic!("expected MissingFile, got {other:?}"),
        }
    }

    #[test]
    fn parse_manifest_without_icon() {
        let toml = r#"
id = "no_icon"
name = "No Icon Plugin"
version = "0.1.0"
entry_ui = "ui/panel.slint"
entry_component = "Panel"
"#;
        let m = PluginManifest::from_toml(toml, "no_icon").unwrap();
        assert!(m.icon.is_none());
    }

    #[test]
    fn reject_absolute_icon_path() {
        let toml = r#"
id = "bad_icon"
name = "Test"
version = "1.0.0"
entry_ui = "ui/p.slint"
entry_component = "P"

[icon]
kind = "file"
path = "/etc/icon.png"
"#;
        let err = PluginManifest::from_toml(toml, "hint").unwrap_err();
        assert!(err.to_string().contains("icon file path must be relative"));
    }
}
