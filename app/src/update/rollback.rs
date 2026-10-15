//! What `eov update --rollback` restores.
//!
//! Every update that replaces eov keeps the previous installation until the
//! next update replaces it, and records how to get it back in
//! `<updates>/rollback.toml`:
//!
//! - **AppImage**: the hidden `.<name>.previous` image next to the AppImage.
//! - **Windows portable / manual macOS bundle**: the previous tree, kept by
//!   the update helper next to the installation (`.eov-previous-*`).
//! - **Cargo**: the previous version, reinstalled with `cargo install`.
//! - **Homebrew**: the previous version's `eov@X.Y.Z` cask.
//! - **Plugins**: the replaced packages, kept in `<updates>/rollback/plugins/`
//!   together with their registry records.
//!
//! A rollback swaps the current and the recorded state, so running it again
//! returns to the newer version.

use super::registry::ManagedPlugin;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const RECORD_FILE: &str = "rollback.toml";
pub const RECORD_SCHEMA: u32 = 1;

/// How the replaced application can be restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum AppRollback {
    AppImage { path: PathBuf, previous: PathBuf },
    Tree { path: PathBuf, previous: PathBuf },
    Cargo { root: PathBuf, version: String },
    Homebrew { tap: String, version: String },
}

/// A replaced plugin package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginRollback {
    pub id: String,
    /// The kept package, or `None` when the plugin was newly installed (a
    /// rollback then removes it).
    #[serde(default)]
    pub package: Option<PathBuf>,
    /// The registry record of the kept package.
    #[serde(default)]
    pub entry: Option<ManagedPlugin>,
}

/// The state a rollback returns to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackRecord {
    pub schema: u32,
    pub created_at: u64,
    /// The version a rollback restores.
    pub previous_version: String,
    /// The version that is installed now.
    pub current_version: String,
    #[serde(default)]
    pub app: Option<AppRollback>,
    #[serde(default)]
    pub plugins: Vec<PluginRollback>,
}

/// The directory that keeps replaced plugin packages.
pub fn plugin_store(updates_dir: &Path) -> PathBuf {
    updates_dir.join("rollback").join("plugins")
}

pub fn load(updates_dir: &Path) -> Result<Option<RollbackRecord>> {
    let path = updates_dir.join(RECORD_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("could not read {}", path.display()));
        }
    };
    let record: RollbackRecord =
        toml::from_str(&text).with_context(|| format!("could not parse {}", path.display()))?;
    if record.schema != RECORD_SCHEMA {
        anyhow::bail!("unsupported rollback record schema {}", record.schema);
    }
    Ok(Some(record))
}

/// Renders a record (the update helper writes it once its swap succeeded).
pub fn render(record: &RollbackRecord) -> Result<String> {
    Ok(toml::to_string_pretty(record)?)
}

pub fn save(updates_dir: &Path, record: &RollbackRecord) -> Result<()> {
    fs::create_dir_all(updates_dir)?;
    let mut file = tempfile::NamedTempFile::new_in(updates_dir)?;
    file.write_all(render(record)?.as_bytes())?;
    file.as_file().sync_all()?;
    file.persist(updates_dir.join(RECORD_FILE))
        .map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let record = RollbackRecord {
            schema: RECORD_SCHEMA,
            created_at: 1,
            previous_version: "1.0.0".into(),
            current_version: "1.0.1".into(),
            app: Some(AppRollback::AppImage {
                path: "/apps/eov.AppImage".into(),
                previous: "/apps/.eov.AppImage.previous".into(),
            }),
            plugins: vec![PluginRollback {
                id: "gamepad".into(),
                package: Some(plugin_store(temp.path()).join("gamepad.eop")),
                entry: None,
            }],
        };
        save(temp.path(), &record).unwrap();
        assert_eq!(load(temp.path()).unwrap(), Some(record));
        fs::remove_file(temp.path().join(RECORD_FILE)).unwrap();
        assert_eq!(load(temp.path()).unwrap(), None);
    }
}
