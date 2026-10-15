//! The eov plugin API.
//!
//! An eov plugin is a native library (`cdylib`) that implements
//! [`sdk::Plugin`] and exports it with [`export_plugin!`]. The host talks to
//! it through a small, layout-checked ABI ([`abi`]) carrying JSON messages
//! defined in [`protocol`]; the SDK hides both behind safe Rust types.
//!
//! ```no_run
//! use eov_plugin_api::prelude::*;
//!
//! struct Counter {
//!     host: Host,
//! }
//!
//! impl Plugin for Counter {
//!     fn create(host: Host, _init: PluginInit) -> Result<Self, String> {
//!         Ok(Self { host })
//!     }
//!
//!     fn on_event(&self, event: HostEvent) -> Result<(), String> {
//!         if let HostEvent::FileOpened { file } = event {
//!             self.host.info(format!("{} opened", file.filename));
//!         }
//!         Ok(())
//!     }
//! }
//!
//! eov_plugin_api::export_plugin!(Counter);
//! ```
//!
//! A plugin package (`.eop`) is a tar archive holding `plugin.toml` (see
//! [`manifest`]), the library and any `.slint` files and assets. See the
//! crate README and `example-plugin/` for a complete walkthrough, and
//! [`testing`] for unit-testing plugins without eov.

pub mod abi;
pub mod manifest;
pub mod protocol;
pub mod sdk;
pub mod testing;

pub use manifest::PluginManifest;

/// Everything a plugin usually needs: `use eov_plugin_api::prelude::*;`.
pub mod prelude {
    pub use crate::abi::{FfiErrorKind, FilterView, RegionRequest};
    pub use crate::protocol::*;
    pub use crate::sdk::{Frame, Host, HostError, HostResult, Image, Job, Layer, Plugin, Surface};
    pub use serde_json::{Map, Value, json};
}

/// Re-exports used by [`export_plugin!`]; not part of the public API.
#[doc(hidden)]
pub mod __private {
    pub use abi_stable::library::{
        __call_root_module_loader as call_root_module_loader, CheckTypeLayout, LibHeader,
        RootModuleResult,
    };
}

use std::path::PathBuf;
use thiserror::Error;

/// Errors raised while reading plugin packages and manifests.
#[derive(Debug, Error)]
pub enum PluginError {
    #[error("plugin manifest error in '{plugin_id}': {message}")]
    Manifest { plugin_id: String, message: String },
    #[error("plugin '{plugin_id}' missing required file: {path}")]
    MissingFile { plugin_id: String, path: PathBuf },
    #[error("duplicate plugin id: '{0}'")]
    DuplicateId(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type PluginResult<T> = Result<T, PluginError>;

/// How a plugin's icon is provided in its manifest.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IconDescriptor {
    /// Inline SVG.
    Svg { data: String },
    /// Path to an image, relative to the plugin root.
    File { path: PathBuf },
}

/// A discovered plugin package: its extracted root and parsed manifest.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Absolute path to the extracted plugin root.
    pub root: PathBuf,
    pub manifest: PluginManifest,
}
