//! Shared plugin API for EOV.
//!
//! This crate defines the traits, structs, and manifest types that both the
//! host application and individual plugins depend on. It intentionally avoids
//! any dependency on Slint or other UI frameworks so that plugin non-UI logic
//! can be tested independently.
//!
//! # Adding a new plugin
//!
//! 1. Create a directory under the plugin directory (default `~/.eov/plugins/`).
//! 2. Add a `plugin.toml` manifest (see [`PluginManifest`]).
//! 3. Place your `.slint` UI file relative to the plugin root.
//! 4. Implement the [`Plugin`] trait and register it via the plugin registry.
//!
//! # Manifest format (`plugin.toml`)
//!
//! ```toml
//! id = "example_plugin"
//! name = "Example Plugin"
//! version = "0.1.0"
//! entry_ui = "ui/my_panel.slint"
//! entry_component = "MyPanel"
//!
//! [icon]
//! kind = "svg"
//! data = "<svg>...</svg>"
//! ```

pub mod ffi;
pub mod manifest;

pub use manifest::PluginManifest;

use std::path::{Path, PathBuf};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("plugin manifest error in '{plugin_id}': {message}")]
    Manifest {
        plugin_id: String,
        message: String,
    },
    #[error("plugin '{plugin_id}' missing required file: {path}")]
    MissingFile {
        plugin_id: String,
        path: PathBuf,
    },
    #[error("duplicate plugin id: '{0}'")]
    DuplicateId(String),
    #[error("duplicate toolbar button id: '{0}'")]
    DuplicateButtonId(String),
    #[error("plugin activation error in '{plugin_id}': {message}")]
    Activation {
        plugin_id: String,
        message: String,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type PluginResult<T> = Result<T, PluginError>;

// ---------------------------------------------------------------------------
// Icon descriptor
// ---------------------------------------------------------------------------

/// Describes how a plugin icon is provided.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IconDescriptor {
    /// Inline SVG string.
    Svg { data: String },
    /// Path to an image file, relative to the plugin root.
    File { path: PathBuf },
}

// ---------------------------------------------------------------------------
// Toolbar button registration
// ---------------------------------------------------------------------------

/// Registration data for a plugin toolbar button.
///
/// The host renders the actual button; the plugin only provides metadata and
/// an action identifier.
#[derive(Debug, Clone)]
pub struct ToolbarButtonRegistration {
    /// Owning plugin id.
    pub plugin_id: String,
    /// Unique identifier for this button (scoped to the plugin).
    pub button_id: String,
    /// Tooltip / accessible label shown on hover.
    pub tooltip: String,
    /// Icon to display.
    pub icon: IconDescriptor,
    /// Opaque action identifier dispatched back to the plugin on click.
    pub action_id: String,
}

// ---------------------------------------------------------------------------
// Host context — the API surface a plugin can call during activation
// ---------------------------------------------------------------------------

/// Trait implemented by the host and passed to plugins during activation.
///
/// Plugins call methods on the host context to register toolbar buttons and
/// request windows. The trait is object-safe so it can be used with `dyn`.
pub trait HostContext {
    /// Register a toolbar button. The button is appended after all built-in
    /// toolbar items.
    fn add_toolbar_button(&mut self, button: ToolbarButtonRegistration) -> PluginResult<()>;

    /// Request the host to open a plugin window.
    ///
    /// `ui_path` is an absolute path to the `.slint` file.
    /// `component` is the exported component name within that file.
    fn open_plugin_window(
        &mut self,
        plugin_id: &str,
        ui_path: &Path,
        component: &str,
    ) -> PluginResult<()>;
}

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// The core trait that every plugin must implement.
pub trait Plugin: Send + Sync {
    /// Return the parsed manifest for this plugin.
    fn manifest(&self) -> &PluginManifest;

    /// Called once during startup. The plugin should register toolbar buttons
    /// and any other contributions via `host`. `plugin_root` is the absolute
    /// path to the plugin directory on disk.
    fn activate(&self, host: &mut dyn HostContext, plugin_root: &Path) -> PluginResult<()>;

    /// Called when a toolbar button registered by this plugin is clicked.
    /// `action_id` corresponds to `ToolbarButtonRegistration::action_id`.
    /// `plugin_root` is the absolute path to the plugin directory on disk.
    fn on_action(
        &self,
        action_id: &str,
        host: &mut dyn HostContext,
        plugin_root: &Path,
    ) -> PluginResult<()>;
}

// ---------------------------------------------------------------------------
// Plugin descriptor — returned by discovery, before the plugin is loaded
// ---------------------------------------------------------------------------

/// Lightweight descriptor produced by directory scanning before the plugin is
/// fully loaded.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Absolute path to the plugin root directory.
    pub root: PathBuf,
    /// Parsed manifest.
    pub manifest: PluginManifest,
}

impl PluginDescriptor {
    /// Resolve the `entry_ui` path against the plugin root.
    pub fn resolve_ui_path(&self) -> PathBuf {
        self.root.join(&self.manifest.entry_ui)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbar_button_fields_preserved() {
        let reg = ToolbarButtonRegistration {
            plugin_id: "test".into(),
            button_id: "btn1".into(),
            tooltip: "Test Button".into(),
            icon: IconDescriptor::Svg {
                data: "<svg/>".into(),
            },
            action_id: "do_thing".into(),
        };
        assert_eq!(reg.plugin_id, "test");
        assert_eq!(reg.button_id, "btn1");
        assert_eq!(reg.tooltip, "Test Button");
        assert_eq!(reg.action_id, "do_thing");
    }

    #[test]
    fn descriptor_resolves_ui_path() {
        let desc = PluginDescriptor {
            root: PathBuf::from("/plugins/my_plugin"),
            manifest: PluginManifest {
                id: "my_plugin".into(),
                name: "My Plugin".into(),
                version: "0.1.0".into(),
                entry_ui: "ui/panel.slint".into(),
                entry_component: "Panel".into(),
                icon: None,
            },
        };
        assert_eq!(
            desc.resolve_ui_path(),
            PathBuf::from("/plugins/my_plugin/ui/panel.slint")
        );
    }
}
