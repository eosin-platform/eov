use anyhow::{Context, Result};
use eov_common::{FilteringMode, RenderBackend};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

static CONFIG_PATH_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConfigRenderBackend {
    Cpu,
    Gpu,
}

impl From<ConfigRenderBackend> for RenderBackend {
    fn from(value: ConfigRenderBackend) -> Self {
        match value {
            ConfigRenderBackend::Cpu => RenderBackend::Cpu,
            ConfigRenderBackend::Gpu => RenderBackend::Gpu,
        }
    }
}

impl From<RenderBackend> for ConfigRenderBackend {
    fn from(value: RenderBackend) -> Self {
        match value {
            RenderBackend::Cpu => ConfigRenderBackend::Cpu,
            RenderBackend::Gpu => ConfigRenderBackend::Gpu,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConfigFilteringMode {
    Bilinear,
    Trilinear,
    #[serde(rename = "lanczos3")]
    Lanczos3,
}

impl From<ConfigFilteringMode> for FilteringMode {
    fn from(value: ConfigFilteringMode) -> Self {
        match value {
            ConfigFilteringMode::Bilinear => FilteringMode::Bilinear,
            ConfigFilteringMode::Trilinear => FilteringMode::Trilinear,
            ConfigFilteringMode::Lanczos3 => FilteringMode::Lanczos3,
        }
    }
}

impl From<FilteringMode> for ConfigFilteringMode {
    fn from(value: FilteringMode) -> Self {
        match value {
            FilteringMode::Bilinear => ConfigFilteringMode::Bilinear,
            FilteringMode::Trilinear => ConfigFilteringMode::Trilinear,
            FilteringMode::Lanczos3 => ConfigFilteringMode::Lanczos3,
        }
    }
}

/// The `[active_sidebar]` table written by eov < 1.0; only its plugin id is
/// still meaningful.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct LegacyActiveSidebar {
    #[serde(default)]
    plugin_id: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AppConfig {
    render_backend: Option<ConfigRenderBackend>,
    filtering_mode: Option<ConfigFilteringMode>,
    /// Plugin whose sidebar was open at the last exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sidebar_plugin: Option<String>,
    #[serde(default, skip_serializing)]
    active_sidebar: Option<LegacyActiveSidebar>,
    /// Plugins the user turned off in the Plugins panel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    disabled_plugins: Vec<String>,
    /// The Measure tool's mode (`line`, `polyline`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    measure_mode: Option<String>,
    /// Plain scrolling pans instead of zooming (trackpad users).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scroll_pans: Option<bool>,
}

pub fn set_config_path_override(path: PathBuf) -> Result<()> {
    if let Some(existing) = CONFIG_PATH_OVERRIDE.get() {
        if existing == &path {
            return Ok(());
        }

        anyhow::bail!(
            "config path override already set to {}; cannot replace with {}",
            existing.display(),
            path.display()
        );
    }

    CONFIG_PATH_OVERRIDE
        .set(path)
        .map_err(|_| anyhow::anyhow!("failed to initialize config path override"))
}

pub fn resolve_config_path() -> Result<PathBuf> {
    if let Some(path) = CONFIG_PATH_OVERRIDE.get() {
        return Ok(path.clone());
    }

    Ok(crate::paths::config_file())
}

pub fn load_render_backend() -> Result<Option<RenderBackend>> {
    let path = resolve_config_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file at {}", path.display()))?;
    Ok(config.render_backend.map(RenderBackend::from))
}

pub fn load_filtering_mode() -> Result<Option<FilteringMode>> {
    let path = resolve_config_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file at {}", path.display()))?;
    Ok(config.filtering_mode.map(FilteringMode::from))
}

pub fn save_render_backend(backend: RenderBackend) -> Result<()> {
    save_config_field(|config| {
        config.render_backend = Some(ConfigRenderBackend::from(backend));
    })
}

pub fn save_filtering_mode(mode: FilteringMode) -> Result<()> {
    save_config_field(|config| {
        config.filtering_mode = Some(ConfigFilteringMode::from(mode));
    })
}

fn load_config() -> Result<AppConfig> {
    let path = resolve_config_path()?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file at {}", path.display()))
}

/// The plugin whose sidebar should be restored at startup.
pub fn load_sidebar_plugin() -> Result<Option<String>> {
    let config = load_config()?;
    Ok(config.sidebar_plugin.or_else(|| {
        config
            .active_sidebar
            .map(|legacy| legacy.plugin_id)
            .filter(|id| !id.is_empty())
    }))
}

pub fn save_sidebar_plugin(plugin_id: Option<&str>) -> Result<()> {
    save_config_field(|config| {
        config.sidebar_plugin = plugin_id.map(str::to_string);
        config.active_sidebar = None;
    })
}

pub fn load_disabled_plugins() -> Result<Vec<String>> {
    Ok(load_config()?.disabled_plugins)
}

pub fn set_plugin_disabled(plugin_id: &str, disabled: bool) -> Result<()> {
    save_config_field(|config| {
        config.disabled_plugins.retain(|id| id != plugin_id);
        if disabled {
            config.disabled_plugins.push(plugin_id.to_string());
            config.disabled_plugins.sort();
        }
    })
}

pub fn load_measure_mode() -> Result<Option<String>> {
    Ok(load_config()?.measure_mode)
}

pub fn save_measure_mode(mode: &str) -> Result<()> {
    save_config_field(|config| config.measure_mode = Some(mode.to_string()))
}

pub fn load_scroll_pans() -> Result<bool> {
    Ok(load_config()?.scroll_pans.unwrap_or(false))
}

fn save_config_field(update: impl FnOnce(&mut AppConfig)) -> Result<()> {
    let path = resolve_config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    let mut config = if path.exists() {
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file at {}", path.display()))?;
        toml::from_str(&contents)
            .with_context(|| format!("failed to parse config file at {}", path.display()))?
    } else {
        AppConfig::default()
    };

    update(&mut config);

    let contents =
        toml::to_string_pretty(&config).context("failed to serialize EOV configuration")?;
    fs::write(&path, contents)
        .with_context(|| format!("failed to write config file at {}", path.display()))?;
    Ok(())
}
