use crate::state::RenderBackend;
use anyhow::{Context, Result};
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

#[derive(Debug, Default, Serialize, Deserialize)]
struct AppConfig {
    render_backend: Option<ConfigRenderBackend>,
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

    let home =
        dirs::home_dir().context("failed to determine the home directory for EOV config")?;
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

pub fn save_render_backend(backend: RenderBackend) -> Result<()> {
    let path = resolve_config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    let config = AppConfig {
        render_backend: Some(ConfigRenderBackend::from(backend)),
    };
    let contents =
        toml::to_string_pretty(&config).context("failed to serialize EOV configuration")?;
    fs::write(&path, contents)
        .with_context(|| format!("failed to write config file at {}", path.display()))?;
    Ok(())
}
