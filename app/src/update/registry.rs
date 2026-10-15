//! The managed plugin registry (`<plugin-dir>/.eov-managed.toml`) and the
//! plugin-directory lock.
//!
//! The registry records the provenance of every package eov installed
//! (official catalog, GitHub repository, or local file) together with the
//! SHA-256 that was verified at install time, and the set of plugins the
//! user disabled. Unmanaged packages (copied into the directory by hand) have
//! no provenance record but can still be disabled.
//!
//! The format stays at schema 1 and only gains optional fields, so older eov
//! versions (for example after `eov update --release 0.4.9`) can still read
//! it; they ignore, and on their next write drop, the `disabled` list.
//!
//! Every mutation of the plugin directory (install, remove, update, enable,
//! disable, sync, and the post-exit update helper) holds an exclusive lock on
//! `<plugin-dir>/.eov-plugins.lock`, so concurrent eov processes cannot
//! corrupt the directory or the registry.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const REGISTRY_SCHEMA: u32 = 1;
pub const REGISTRY_FILENAME: &str = ".eov-managed.toml";
pub const LOCK_FILENAME: &str = ".eov-plugins.lock";
const LOCK_TIMEOUT: Duration = Duration::from_secs(300);

/// Where a managed package came from.
pub const SOURCE_OFFICIAL: &str = "official";
pub const SOURCE_GITHUB: &str = "github";
pub const SOURCE_LOCAL: &str = "local";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedPlugin {
    /// `official`, `github` or `local`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_tag: Option<String>,
    pub version: String,
    pub sha256: String,
    pub file: String,
    /// Unix time of the installation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_at: Option<u64>,
}

impl ManagedPlugin {
    pub fn is_official(&self) -> bool {
        self.source == SOURCE_OFFICIAL
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedRegistry {
    pub schema: u32,
    /// Plugin ids that must not be loaded (managed or not).
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub disabled: BTreeSet<String>,
    #[serde(default)]
    pub plugins: BTreeMap<String, ManagedPlugin>,
}

impl Default for ManagedRegistry {
    fn default() -> Self {
        Self {
            schema: REGISTRY_SCHEMA,
            disabled: BTreeSet::new(),
            plugins: BTreeMap::new(),
        }
    }
}

impl ManagedRegistry {
    pub fn is_disabled(&self, id: &str) -> bool {
        self.disabled.contains(id)
    }
}

/// Reads the registry; a missing file is an empty registry.
pub fn load(plugin_dir: &Path) -> Result<ManagedRegistry> {
    let path = plugin_dir.join(REGISTRY_FILENAME);
    let path = if path.exists() {
        path
    } else {
        // eov < 1.0 renamed the registry away while writing on Windows; an
        // interrupted write left only such a backup.
        let mut backups = fs::read_dir(plugin_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&format!("{REGISTRY_FILENAME}.backup-")))
            })
            .collect::<Vec<_>>();
        backups.sort();
        backups.pop().unwrap_or(path)
    };
    if !path.exists() {
        return Ok(ManagedRegistry::default());
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("could not read plugin registry {}", path.display()))?;
    let registry: ManagedRegistry = toml::from_str(&content)
        .with_context(|| format!("could not parse plugin registry {}", path.display()))?;
    if registry.schema != REGISTRY_SCHEMA {
        bail!(
            "plugin registry {} uses schema {}, which this eov does not support",
            path.display(),
            registry.schema
        );
    }
    Ok(registry)
}

/// Writes the registry atomically (temporary file, fsync, rename).
pub fn save(plugin_dir: &Path, registry: &ManagedRegistry) -> Result<()> {
    let target = plugin_dir.join(REGISTRY_FILENAME);
    let staged = stage(plugin_dir, registry)?;
    staged
        .persist(&target)
        .map_err(|error| error.error)
        .with_context(|| format!("could not write plugin registry {}", target.display()))?;
    sync_directory(plugin_dir);
    Ok(())
}

/// Writes the registry to a temporary file next to its final location, for
/// transactions that activate it later (the update helper). The file is
/// deleted if the returned handle is dropped without being persisted.
pub fn stage(plugin_dir: &Path, registry: &ManagedRegistry) -> Result<tempfile::NamedTempFile> {
    fs::create_dir_all(plugin_dir)
        .with_context(|| format!("could not create {}", plugin_dir.display()))?;
    let mut file = tempfile::Builder::new()
        .prefix(".eov-managed-")
        .suffix(".tmp")
        .tempfile_in(plugin_dir)?;
    file.write_all(toml::to_string_pretty(registry)?.as_bytes())?;
    file.as_file().sync_all()?;
    Ok(file)
}

/// The ids the user disabled. Errors are logged and treated as "none".
///
/// This is the function the plugin host calls before loading plugins.
#[allow(dead_code)] // integration point for the plugin host
pub fn disabled_plugins(plugin_dir: &Path) -> BTreeSet<String> {
    match load(plugin_dir) {
        Ok(registry) => registry.disabled,
        Err(error) => {
            tracing::warn!("{error:#}");
            BTreeSet::new()
        }
    }
}

/// Whether the user disabled `id` in `plugin_dir`.
#[allow(dead_code)] // integration point for the plugin host
pub fn is_plugin_disabled(plugin_dir: &Path, id: &str) -> bool {
    disabled_plugins(plugin_dir).contains(id)
}

/// Exclusive ownership of a plugin directory for the guard's lifetime.
#[derive(Debug)]
pub struct PluginDirLock {
    file: File,
    #[cfg_attr(not(test), allow(dead_code))]
    path: PathBuf,
}

impl PluginDirLock {
    /// Locks `plugin_dir`, waiting (with a notice) while another eov process
    /// is modifying it.
    pub fn acquire(plugin_dir: &Path) -> Result<Self> {
        acquire_lock_file(&plugin_dir.join(LOCK_FILENAME), "plugins in")
            .map(|(file, path)| Self { file, path })
    }

    #[cfg(test)]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PluginDirLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Opens and exclusively locks `path` (creating it), waiting up to five
/// minutes for another process to release it.
pub fn acquire_lock_file(path: &Path, what: &str) -> Result<(File, PathBuf)> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("could not open lock file {}", path.display()))?;
    let start = Instant::now();
    let mut announced = false;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok((file, path.to_path_buf())),
            Err(fs::TryLockError::WouldBlock) => {
                if !announced {
                    eprintln!(
                        "Waiting for another eov process that is modifying {what} {}...",
                        path.parent().unwrap_or(path).display()
                    );
                    announced = true;
                }
                if start.elapsed() > LOCK_TIMEOUT {
                    bail!(
                        "another eov process has been holding {} for over {} minutes; try again later",
                        path.display(),
                        LOCK_TIMEOUT.as_secs() / 60
                    );
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(fs::TryLockError::Error(error)) => {
                return Err(error).with_context(|| format!("could not lock {}", path.display()));
            }
        }
    }
}

/// Flushes directory metadata (renames) to disk where the platform allows it.
pub fn sync_directory(directory: &Path) {
    #[cfg(unix)]
    if let Ok(handle) = File::open(directory) {
        let _ = handle.sync_all();
    }
    #[cfg(not(unix))]
    let _ = directory;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> ManagedPlugin {
        ManagedPlugin {
            source: SOURCE_OFFICIAL.into(),
            repository: Some("https://github.com/eosin-platform/eov-gamepad-plugin".into()),
            release_tag: Some("v1.0.0".into()),
            version: "1.0.0".into(),
            sha256: "a".repeat(64),
            file: "gamepad.eop".into(),
            installed_at: Some(1_760_000_000),
        }
    }

    #[test]
    fn registry_round_trips_with_disabled_plugins() {
        let temp = tempfile::tempdir().unwrap();
        let mut registry = ManagedRegistry::default();
        registry.plugins.insert("gamepad".into(), entry());
        registry.disabled.insert("annotations".into());
        save(temp.path(), &registry).unwrap();
        assert_eq!(load(temp.path()).unwrap(), registry);
        assert!(is_plugin_disabled(temp.path(), "annotations"));
        assert!(!is_plugin_disabled(temp.path(), "gamepad"));
        // Nothing but the registry is left behind.
        let names: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec![REGISTRY_FILENAME.to_string()]);
    }

    #[test]
    fn registries_written_by_older_eov_versions_load() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join(REGISTRY_FILENAME),
            "schema = 1\n\n[plugins.gamepad]\nsource = \"official\"\nversion = \"0.2.3\"\nsha256 = \"aa\"\nfile = \"gamepad.eop\"\n",
        )
        .unwrap();
        let registry = load(temp.path()).unwrap();
        assert!(registry.disabled.is_empty());
        assert_eq!(registry.plugins["gamepad"].version, "0.2.3");
        // The written form stays readable by older versions (schema 1).
        save(temp.path(), &registry).unwrap();
        let text = fs::read_to_string(temp.path().join(REGISTRY_FILENAME)).unwrap();
        assert!(text.starts_with("schema = 1"), "{text}");
    }

    #[test]
    fn unsupported_schema_is_an_error() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join(REGISTRY_FILENAME), "schema = 7\n").unwrap();
        assert!(load(temp.path()).is_err());
        assert!(disabled_plugins(temp.path()).is_empty());
    }

    #[test]
    fn plugin_dir_lock_is_exclusive_across_handles() {
        let temp = tempfile::tempdir().unwrap();
        let guard = PluginDirLock::acquire(temp.path()).unwrap();
        let other = OpenOptions::new()
            .read(true)
            .write(true)
            .open(guard.path())
            .unwrap();
        assert!(matches!(
            other.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        drop(guard);
        other.try_lock().unwrap();
    }
}
