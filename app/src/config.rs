//! The user configuration file (`~/.eov/config.toml`, `$EOV_CONFIG` or
//! `--config`).
//!
//! The file is read once and edited in place with `toml_edit`, so keys this
//! version does not know (written by a newer eov or by hand) and comments
//! survive saving. Saving re-reads the file first, so changes made by
//! another running eov are kept, and writes atomically. A file that cannot
//! be parsed is set aside as `config.toml.broken` with a warning and eov
//! starts with defaults instead of refusing to start; unknown values of
//! known keys are ignored with a warning (SH-3).

use anyhow::{Context, Result};
use eov_common::{FilteringMode, RenderBackend};
use parking_lot::Mutex;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use toml_edit::{DocumentMut, value};
use tracing::warn;

static CONFIG_PATH_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// The parsed file, cached per path.
struct Loaded {
    path: PathBuf,
    document: DocumentMut,
}

fn loaded() -> &'static Mutex<Option<Loaded>> {
    static LOADED: OnceLock<Mutex<Option<Loaded>>> = OnceLock::new();
    LOADED.get_or_init(|| Mutex::new(None))
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

/// Reads and parses the file. `None` when it is missing; a file that
/// cannot be parsed is moved aside (when `quarantine`) and also gives
/// `None`.
fn read_document(path: &Path, quarantine: bool) -> Option<DocumentMut> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warn!("Could not read {}: {error}; using defaults", path.display());
            return None;
        }
    };
    match text.parse::<DocumentMut>() {
        Ok(document) => Some(document),
        Err(error) => {
            if quarantine {
                let backup = path.with_extension("toml.broken");
                match fs::rename(path, &backup) {
                    Ok(()) => warn!(
                        "{} is not valid TOML ({error}); it was moved to {} and defaults are used",
                        path.display(),
                        backup.display()
                    ),
                    Err(rename_error) => warn!(
                        "{} is not valid TOML ({error}) and could not be moved aside: \
                         {rename_error}; defaults are used",
                        path.display()
                    ),
                }
            }
            None
        }
    }
}

fn with_document<R>(read: impl FnOnce(&Path, &DocumentMut) -> R) -> Result<R> {
    let path = resolve_config_path()?;
    let mut slot = loaded().lock();
    if slot.as_ref().is_none_or(|loaded| loaded.path != path) {
        let document = read_document(&path, true).unwrap_or_default();
        *slot = Some(Loaded {
            path: path.clone(),
            document,
        });
    }
    let loaded = slot.as_ref().expect("just loaded");
    Ok(read(&loaded.path, &loaded.document))
}

/// Applies `update` to the file as it is on disk now (keeping changes made
/// by other eov instances) and writes it atomically.
fn save(update: impl FnOnce(&mut DocumentMut)) -> Result<()> {
    let path = resolve_config_path()?;
    let mut slot = loaded().lock();
    let mut document = read_document(&path, false)
        .or_else(|| {
            slot.as_ref()
                .filter(|loaded| loaded.path == path)
                .map(|loaded| loaded.document.clone())
        })
        .unwrap_or_default();
    update(&mut document);
    write_atomically(&path, &document.to_string())?;
    *slot = Some(Loaded { path, document });
    Ok(())
}

fn write_atomically(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text)
        .with_context(|| format!("failed to write config file at {}", temp.display()))?;
    fs::rename(&temp, path)
        .with_context(|| format!("failed to replace config file at {}", path.display()))
}

fn string(document: &DocumentMut, key: &str) -> Option<String> {
    document.get(key)?.as_str().map(str::to_string)
}

fn ignore_value(path: &Path, key: &str, found: &str) {
    warn!("Ignoring unknown {key} = {found:?} in {}", path.display());
}

pub fn load_render_backend() -> Result<Option<RenderBackend>> {
    with_document(|path, document| {
        let found = string(document, "render_backend")?;
        match found.to_ascii_lowercase().as_str() {
            "cpu" => Some(RenderBackend::Cpu),
            "gpu" => Some(RenderBackend::Gpu),
            _ => {
                ignore_value(path, "render_backend", &found);
                None
            }
        }
    })
}

pub fn save_render_backend(backend: RenderBackend) -> Result<()> {
    let name = match backend {
        RenderBackend::Cpu => "cpu",
        RenderBackend::Gpu => "gpu",
    };
    save(|document| document["render_backend"] = value(name))
}

/// Parses a filtering mode as written in the file or on the command line
/// (`lanczos` and `lanczos3` are the same).
fn parse_filtering_mode(text: &str) -> Option<FilteringMode> {
    match text.to_ascii_lowercase().as_str() {
        "bilinear" => Some(FilteringMode::Bilinear),
        "trilinear" => Some(FilteringMode::Trilinear),
        "lanczos" | "lanczos3" => Some(FilteringMode::Lanczos3),
        _ => None,
    }
}

pub fn load_filtering_mode() -> Result<Option<FilteringMode>> {
    with_document(|path, document| {
        let found = string(document, "filtering_mode")?;
        let mode = parse_filtering_mode(&found);
        if mode.is_none() {
            ignore_value(path, "filtering_mode", &found);
        }
        mode
    })
}

pub fn save_filtering_mode(mode: FilteringMode) -> Result<()> {
    let name = match mode {
        FilteringMode::Bilinear => "bilinear",
        FilteringMode::Trilinear => "trilinear",
        FilteringMode::Lanczos3 => "lanczos3",
    };
    save(|document| document["filtering_mode"] = value(name))
}

/// The plugin whose sidebar should be restored at startup.
pub fn load_sidebar_plugin() -> Result<Option<String>> {
    with_document(|_, document| {
        string(document, "sidebar_plugin").or_else(|| {
            // The `[active_sidebar]` table written by eov < 1.0.
            document
                .get("active_sidebar")?
                .get("plugin_id")?
                .as_str()
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        })
    })
}

pub fn save_sidebar_plugin(plugin_id: Option<&str>) -> Result<()> {
    save(|document| {
        match plugin_id {
            Some(id) => document["sidebar_plugin"] = value(id),
            None => {
                document.remove("sidebar_plugin");
            }
        }
        document.remove("active_sidebar");
    })
}

pub fn load_measure_mode() -> Result<Option<String>> {
    with_document(|_, document| string(document, "measure_mode"))
}

pub fn save_measure_mode(mode: &str) -> Result<()> {
    save(|document| document["measure_mode"] = value(mode))
}

/// Whether recently opened files are remembered (`remember_recent_files`,
/// on by default).
pub fn load_remember_recent_files() -> bool {
    with_document(|_, document| {
        document
            .get("remember_recent_files")
            .and_then(|item| item.as_bool())
            .unwrap_or(true)
    })
    .unwrap_or(true)
}

/// Whether slide labels (and macro photos, which show them) are hidden in
/// the Slide Information panel (`hide_slide_label`).
pub fn load_hide_slide_label() -> bool {
    with_document(|_, document| {
        document
            .get("hide_slide_label")
            .and_then(|item| item.as_bool())
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

pub fn save_hide_slide_label(hidden: bool) -> Result<()> {
    save(|document| document["hide_slide_label"] = value(hidden))
}

pub fn load_scroll_pans() -> Result<bool> {
    with_document(|_, document| {
        document
            .get("scroll_pans")
            .and_then(|item| item.as_bool())
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_keep_unknown_keys_and_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "# my settings\nfuture_key = 3\nfiltering_mode = \"lanczos\"\n",
        )
        .unwrap();
        let mut document = read_document(&path, true).unwrap();
        assert_eq!(
            string(&document, "filtering_mode")
                .as_deref()
                .and_then(parse_filtering_mode),
            Some(FilteringMode::Lanczos3)
        );
        document["render_backend"] = value("gpu");
        write_atomically(&path, &document.to_string()).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("# my settings"));
        assert!(text.contains("future_key = 3"));
        assert!(text.contains("render_backend = \"gpu\""));
        assert!(!dir.path().join("config.toml.tmp").exists());
    }

    #[test]
    fn broken_files_are_set_aside() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "render_backend = = \"gpu\"").unwrap();
        assert!(read_document(&path, true).is_none());
        assert!(!path.exists());
        assert!(dir.path().join("config.toml.broken").exists());
    }

    #[test]
    fn unknown_values_are_not_fatal() {
        assert_eq!(parse_filtering_mode("bicubic"), None);
        assert_eq!(
            parse_filtering_mode("Trilinear"),
            Some(FilteringMode::Trilinear)
        );
    }
}
