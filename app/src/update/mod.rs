//! Application and plugin update infrastructure.
//!
//! Release metadata parsing and network access live here so CLI handlers do not
//! embed HTTP or loosely typed TOML logic.

mod app;
mod homebrew;
pub mod manifest;
pub mod network;
mod plugins;
mod prompt;

pub use plugins::{
    OutputFormat as PluginOutputFormat, info as plugin_info, install as plugin_install,
    list as plugin_list, remove as plugin_remove,
};

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
