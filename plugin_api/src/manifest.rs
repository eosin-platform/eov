//! Plugin manifest (`plugin.toml`) parsing and validation.
//!
//! ```toml
//! id = "example_plugin"              # ^[a-z0-9][a-z0-9_]{0,63}$
//! name = "Example Plugin"
//! version = "1.0.0"                  # semantic version
//! description = "Optional description shown in the Plugins panel"
//! repository = "https://github.com/example/example-plugin"
//! api = ">=1.0, <2"                  # optional eov-plugin-api requirement
//!
//! [environment]                      # optional; defaults to any eov version
//! version = ">=1.0.0, <2.0.0"        # eov application version requirement
//!
//! [icon]                             # optional, for the Plugins panel
//! kind = "svg"
//! data = "<svg xmlns='http://www.w3.org/2000/svg'/>"
//!
//! [ui]                               # optional
//! slint = ">=1.17"                   # Slint version the .slint files need
//! ```
//!
//! The plugin library must be named after the id (`lib<id>.so`,
//! `lib<id>.dylib` or `<id>.dll`) and sit next to `plugin.toml`. Unknown keys
//! are ignored so that manifests written for newer hosts still load.

use crate::{IconDescriptor, PluginError, PluginResult};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};

/// Requirements on the host application.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginEnvironment {
    /// eov version requirement (semver range).
    pub version: String,
}

impl Default for PluginEnvironment {
    /// Manifests without an `[environment]` table accept any eov version; the
    /// plugin-API check at load time still applies.
    fn default() -> Self {
        Self {
            version: "*".to_string(),
        }
    }
}

/// UI requirements.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginUi {
    /// Slint version requirement for the plugin's `.slint` surfaces.
    #[serde(default)]
    pub slint: Option<String>,
}

/// The parsed contents of a `plugin.toml` manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    /// Unique, stable identifier (`^[a-z0-9][a-z0-9_]{0,63}$`).
    pub id: String,
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Semantic version of the plugin.
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub repository: Option<String>,
    /// eov-plugin-api requirement, e.g. `">=1.0, <2"`. When absent the API
    /// version embedded in the library is the only check.
    #[serde(default)]
    pub api: Option<String>,
    /// Icon for the Plugins panel.
    #[serde(default)]
    pub icon: Option<IconDescriptor>,
    /// Host application requirements.
    #[serde(default)]
    pub environment: PluginEnvironment,
    /// Whether the manifest had an `[environment]` table.
    #[serde(skip)]
    pub declares_environment: bool,
    #[serde(default)]
    pub ui: PluginUi,
}

/// Name of the manifest file inside each plugin directory.
pub const MANIFEST_FILENAME: &str = "plugin.toml";

/// Whether `id` is a valid plugin id: lowercase ASCII letters, digits and
/// `_`, starting with a letter or digit, at most 64 characters.
pub fn is_valid_plugin_id(id: &str) -> bool {
    let mut chars = id.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit())
        && id.len() <= 64
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Whether `path` is relative and stays inside its base directory
/// (no root, drive prefix or `..` component).
pub fn is_contained_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

impl PluginManifest {
    /// Parses and validates a manifest. `plugin_id_hint` names the package
    /// in error messages when the id itself is unusable.
    pub fn from_toml(toml_str: &str, plugin_id_hint: &str) -> PluginResult<Self> {
        let mut manifest: Self = toml::from_str(toml_str).map_err(|e| PluginError::Manifest {
            plugin_id: plugin_id_hint.to_string(),
            message: format!("TOML parse error: {e}"),
        })?;
        manifest.declares_environment = toml_str
            .parse::<toml::Table>()
            .map(|table| table.contains_key("environment"))
            .unwrap_or(false);
        manifest.validate()?;
        Ok(manifest)
    }

    /// Loads a manifest from a file path.
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

    fn error(&self, message: impl Into<String>) -> PluginError {
        PluginError::Manifest {
            plugin_id: self.id.clone(),
            message: message.into(),
        }
    }

    fn validate(&self) -> PluginResult<()> {
        if !is_valid_plugin_id(&self.id) {
            return Err(self.error(format!(
                "'id' must be 1–64 lowercase letters, digits or '_' and start with a letter or digit, got '{}'",
                self.id
            )));
        }
        if self.name.trim().is_empty() || self.name.len() > 100 {
            return Err(self.error("'name' must be 1–100 characters"));
        }
        Version::parse(&self.version)
            .map_err(|e| self.error(format!("'version' must be a semantic version: {e}")))?;
        VersionReq::parse(&self.environment.version).map_err(|e| {
            self.error(format!(
                "'environment.version' must be a semver requirement: {e}"
            ))
        })?;
        if let Some(api) = &self.api {
            VersionReq::parse(api)
                .map_err(|e| self.error(format!("'api' must be a semver requirement: {e}")))?;
        }
        if let Some(slint) = &self.ui.slint {
            VersionReq::parse(slint)
                .map_err(|e| self.error(format!("'ui.slint' must be a semver requirement: {e}")))?;
        }
        if self.description.as_ref().is_some_and(|d| d.len() > 2000) {
            return Err(self.error("'description' must be at most 2000 characters"));
        }
        match &self.icon {
            Some(IconDescriptor::File { path }) if !is_contained_relative_path(path) => {
                return Err(self.error(format!(
                    "icon file path must be relative and stay inside the package, got '{}'",
                    path.display()
                )));
            }
            Some(IconDescriptor::Svg { data }) if data.len() > 64 * 1024 => {
                return Err(self.error("inline icon SVG must be at most 64 KiB"));
            }
            _ => {}
        }
        Ok(())
    }

    /// Whether the host's plugin-API version satisfies `api`.
    pub fn accepts_api(&self, api_version: &Version) -> bool {
        self.api
            .as_deref()
            .and_then(|req| VersionReq::parse(req).ok())
            .is_none_or(|req| req.matches(api_version))
    }

    /// Whether the eov version satisfies `environment.version`.
    pub fn accepts_host(&self, host_version: &Version) -> bool {
        VersionReq::parse(&self.environment.version)
            .map(|req| req.matches(host_version))
            .unwrap_or(false)
    }

    /// Whether the host's Slint version satisfies `ui.slint`.
    pub fn accepts_slint(&self, slint_version: &Version) -> bool {
        self.ui
            .slint
            .as_deref()
            .and_then(|req| VersionReq::parse(req).ok())
            .is_none_or(|req| req.matches(slint_version))
    }

    /// Checks that referenced files exist.
    pub fn validate_files(&self, plugin_root: &Path) -> PluginResult<()> {
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
description = "A test plugin"
repository = "https://github.com/example/test-plugin"
api = ">=1.0, <2"

[environment]
version = ">=1.0.0, <2.0.0"

[icon]
kind = "svg"
data = "<svg/>"

[ui]
slint = ">=1.17"
"#;

    fn parse(toml: &str) -> PluginResult<PluginManifest> {
        PluginManifest::from_toml(toml, "hint")
    }

    #[test]
    fn parse_valid_manifest() {
        let m = parse(VALID_TOML).unwrap();
        assert_eq!(m.id, "test_plugin");
        assert_eq!(m.name, "Test Plugin");
        assert_eq!(m.description.as_deref(), Some("A test plugin"));
        assert!(m.declares_environment);
        assert!(m.accepts_api(&Version::new(1, 3, 0)));
        assert!(!m.accepts_api(&Version::new(2, 0, 0)));
        assert!(m.accepts_host(&Version::new(1, 0, 0)));
        assert!(m.accepts_slint(&Version::new(1, 17, 1)));
        assert!(!m.accepts_slint(&Version::new(1, 16, 0)));
    }

    #[test]
    fn environment_is_optional() {
        // PLG-4: cellaris shipped without [environment].
        let m = parse(
            r#"
id = "cellaris"
name = "Cellaris"
version = "1.0.0"
"#,
        )
        .unwrap();
        assert!(!m.declares_environment);
        assert!(m.accepts_host(&Version::new(1, 2, 3)));
        assert!(m.accepts_api(&Version::new(1, 0, 0)));
    }

    #[test]
    fn manifests_of_older_hosts_still_parse() {
        // Fields from the 0.x format are ignored.
        let m = parse(
            r#"
id = "legacy"
name = "Legacy"
version = "0.4.9"
entry_ui = "ui/panel.slint"
entry_component = "Panel"

[[toolbar_buttons]]
button_id = "x"
tooltip = "X"
action_id = "x"

[environment]
version = ">=0.4.1"
"#,
        )
        .unwrap();
        assert_eq!(m.id, "legacy");
        // The host range admits 1.0; the ABI check at load time rejects it.
        assert!(m.accepts_host(&Version::new(1, 0, 0)));
    }

    #[test]
    fn ids_are_validated() {
        for bad in [
            "",
            "Upper",
            "has-dash",
            "has:colon",
            "C:evil",
            "a/b",
            "../x",
            "_lead",
            &"x".repeat(65),
        ] {
            let toml = format!("id = \"{bad}\"\nname = \"N\"\nversion = \"1.0.0\"\n");
            assert!(parse(&toml).is_err(), "accepted '{bad}'");
        }
        for good in ["a", "eovae", "gamepad_2", "0day"] {
            assert!(is_valid_plugin_id(good), "rejected '{good}'");
        }
    }

    #[test]
    fn versions_and_requirements_are_validated() {
        assert!(parse("id = \"a\"\nname = \"A\"\nversion = \"one\"\n").is_err());
        assert!(
            parse("id = \"a\"\nname = \"A\"\nversion = \"1.0.0\"\napi = \"not a req\"\n").is_err()
        );
        assert!(
            parse(
                "id = \"a\"\nname = \"A\"\nversion = \"1.0.0\"\n[environment]\nversion = \"??\"\n"
            )
            .is_err()
        );
        assert!(parse("id = \"a\"\nname = \"\"\nversion = \"1.0.0\"\n").is_err());
    }

    #[test]
    fn icon_paths_must_stay_inside_the_package() {
        for bad in ["/etc/icon.png", "../icon.png", "a/../../icon.png"] {
            let toml = format!(
                "id = \"a\"\nname = \"A\"\nversion = \"1.0.0\"\n[icon]\nkind = \"file\"\npath = \"{bad}\"\n"
            );
            assert!(parse(&toml).is_err(), "accepted '{bad}'");
        }
        let toml = "id = \"a\"\nname = \"A\"\nversion = \"1.0.0\"\n[icon]\nkind = \"file\"\npath = \"assets/icon.png\"\n";
        let m = parse(toml).unwrap();
        assert!(matches!(
            m.validate_files(Path::new("/nonexistent")),
            Err(PluginError::MissingFile { .. })
        ));
    }

    #[test]
    fn readme_example_parses() {
        // Keep the documented format honest (PLG-4).
        let docs = include_str!("manifest.rs");
        let example: String = docs
            .lines()
            .skip_while(|line| !line.starts_with("//! ```toml"))
            .skip(1)
            .take_while(|line| !line.starts_with("//! ```"))
            .map(|line| line.trim_start_matches("//!").trim_start_matches(' '))
            .map(|line| format!("{line}\n"))
            .collect();
        let m = parse(&example).unwrap();
        assert_eq!(m.id, "example_plugin");
        assert!(m.declares_environment);
    }
}
