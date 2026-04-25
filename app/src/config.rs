use anyhow::{Context, Result};
use common::{FilteringMode, RenderBackend};
use plugin_api::ActiveSidebar;
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

#[derive(Debug, Default, Serialize, Deserialize)]
struct AppConfig {
    render_backend: Option<ConfigRenderBackend>,
    filtering_mode: Option<ConfigFilteringMode>,
    extension_host_port: Option<u16>,
    active_sidebar: Option<ActiveSidebar>,
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

    if let Some(path) = std::env::var_os("EOV_CONFIG") {
        return Ok(PathBuf::from(path));
    }

    let home = dirs::home_dir().context("failed to determine the home directory for EOV config")?;
    Ok(home.join(".eov").join("config.toml"))
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

pub fn load_extension_host_port() -> Result<Option<u16>> {
    let path = resolve_config_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file at {}", path.display()))?;
    Ok(config.extension_host_port)
}

pub fn load_active_sidebar() -> Result<Option<ActiveSidebar>> {
    let path = resolve_config_path()?;
    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file at {}", path.display()))?;
    Ok(config.active_sidebar)
}

pub fn save_active_sidebar(sidebar: Option<&ActiveSidebar>) -> Result<()> {
    save_config_field(|config| {
        config.active_sidebar = sidebar.cloned();
    })
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
