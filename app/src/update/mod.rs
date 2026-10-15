//! Application and plugin update infrastructure.
//!
//! Release metadata parsing and network access live here so CLI handlers do not
//! embed HTTP or loosely typed TOML logic.

mod app;
pub mod compat;
mod homebrew;
pub mod keys;
pub mod manifest;
pub mod network;
pub mod platform;
pub mod plugins;
pub mod policy;
pub mod prompt;
pub mod registry;
pub mod signature;
pub mod transport;

pub use plugins::OutputFormat as PluginOutputFormat;
/// Integration points for the plugin host: plugins the user disabled with
/// `eov plugin disable` must not be loaded.
#[allow(unused_imports)]
pub use registry::{disabled_plugins, is_plugin_disabled};

#[derive(Debug, Clone)]
pub struct ApplicationUpdateOptions {
    pub release: Option<String>,
    pub app_only: bool,
    pub yes: bool,
    pub allow_network: bool,
}

pub fn application_update(
    plugin_dir: &std::path::Path,
    options: ApplicationUpdateOptions,
) -> anyhow::Result<()> {
    app::application_update(plugin_dir, options)
}
