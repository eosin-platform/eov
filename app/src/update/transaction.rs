//! The hand-off between eov and `eov-update-helper`.
//!
//! This file is compiled into both binaries (the helper includes it with
//! `#[path]`), so the transaction format can never drift between them. eov
//! writes a fully resolved [`Transaction`] (local paths only: no URLs, no
//! version selection) into a private temporary directory, starts the helper
//! and exits; the helper waits for eov to exit, applies the transaction and
//! writes a [`HelperReport`] that the next `eov update` (or the GUI) shows.

#![allow(dead_code)] // each binary uses a different half of this module

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Version of the transaction format.
pub const TRANSACTION_SCHEMA: u32 = 2;
/// Version of the report format.
pub const REPORT_SCHEMA: u32 = 1;
/// File name of the helper's result report inside the updates directory.
pub const REPORT_FILE: &str = "last-update.toml";
/// Marker written into every installation tree the helper creates, so it
/// only ever deletes directories that are provably eov installations.
pub const TREE_MARKER: &str = ".eov-installation";

/// What the helper does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// Replace the installation tree with a staged one (Windows portable,
    /// manually installed macOS bundle). The old tree is kept as the
    /// rollback copy.
    ReplaceTree,
    /// Swap the installation tree with the kept previous tree (rollback).
    SwapTree,
    /// Run `cargo install` for an exact version (Windows, where the running
    /// executable cannot be overwritten).
    CargoInstall,
}

/// How to start eov again once the transaction is applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relaunch {
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
}

/// A plugin package to activate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginReplacement {
    /// The verified package, staged inside the plugin directory.
    pub staged: PathBuf,
    /// `<plugin-dir>/<id>.eop`.
    pub target: PathBuf,
    /// Where the replaced `target` is moved (kept for rollback).
    pub backup: PathBuf,
    /// Other packages with the same id that are superseded.
    #[serde(default)]
    pub remove_after: Vec<PathBuf>,
    /// Where to keep the replaced package for `eov update --rollback` once
    /// the transaction succeeded (best effort; it is deleted otherwise).
    #[serde(default)]
    pub keep_as: Option<PathBuf>,
}

/// The plugin registry to activate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryReplacement {
    pub staged: PathBuf,
    pub target: PathBuf,
    pub backup: PathBuf,
}

/// A fully resolved, local-only update transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transaction {
    pub schema: u32,
    /// Random identifier; also used in backup file names.
    pub nonce: String,
    /// The eov process to wait for.
    pub pid: u32,
    pub mode: Mode,
    /// The versions involved, for the report.
    #[serde(default)]
    pub from_version: Option<String>,
    #[serde(default)]
    pub to_version: Option<String>,
    /// The installation tree (the folder containing `eov.exe`, or `eov.app`).
    #[serde(default)]
    pub current_tree: Option<PathBuf>,
    /// The verified, extracted replacement tree (replace-tree), or the kept
    /// previous tree (swap-tree). Must be on the same volume as the current
    /// tree.
    #[serde(default)]
    pub staged_tree: Option<PathBuf>,
    /// Where the current tree is moved and kept for `eov update --rollback`.
    #[serde(default)]
    pub previous_tree: Option<PathBuf>,
    /// An older rollback copy to delete once the update succeeded. It is
    /// only deleted when it carries [`TREE_MARKER`].
    #[serde(default)]
    pub stale_previous: Option<PathBuf>,
    #[serde(default)]
    pub cargo_version: Option<String>,
    #[serde(default)]
    pub cargo_root: Option<PathBuf>,
    #[serde(default)]
    pub relaunch: Option<Relaunch>,
    #[serde(default)]
    pub plugins: Vec<PluginReplacement>,
    #[serde(default)]
    pub registry: Option<RegistryReplacement>,
    /// The plugin-directory lock file to hold while plugins are replaced.
    #[serde(default)]
    pub plugin_lock: Option<PathBuf>,
    /// Where to write the [`HelperReport`].
    pub report: PathBuf,
    /// The rollback record to write (pre-rendered by eov) once the
    /// transaction succeeded, and where.
    #[serde(default)]
    pub rollback_record: Option<String>,
    #[serde(default)]
    pub rollback_path: Option<PathBuf>,
    /// Staging leftovers to delete when the helper finishes.
    #[serde(default)]
    pub cleanup: Vec<PathBuf>,
}

impl Transaction {
    /// Structural validation shared by eov (before handing over) and the
    /// helper (before acting).
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != TRANSACTION_SCHEMA {
            return Err(format!(
                "unsupported updater transaction schema {}",
                self.schema
            ));
        }
        if self.nonce.is_empty()
            || !self
                .nonce
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
        {
            return Err("invalid updater transaction nonce".into());
        }
        let mut paths: Vec<&Path> = vec![self.report.as_path()];
        match self.mode {
            Mode::ReplaceTree | Mode::SwapTree => {
                let current = self
                    .current_tree
                    .as_deref()
                    .ok_or("the transaction has no current tree")?;
                let staged = self
                    .staged_tree
                    .as_deref()
                    .ok_or("the transaction has no staged tree")?;
                let previous = self
                    .previous_tree
                    .as_deref()
                    .ok_or("the transaction has no previous-tree location")?;
                if current == staged || current == previous || staged == previous {
                    return Err("updater transaction paths must be distinct".into());
                }
                // Renames only work within one volume: the previous tree is
                // a sibling and the staged tree lives inside the same parent.
                let parent = current.parent().ok_or("the installation has no parent")?;
                if previous.parent() != Some(parent) || !staged.starts_with(parent) {
                    return Err(
                        "the staged and previous trees must sit next to the installation".into(),
                    );
                }
                paths.extend([current, staged, previous]);
            }
            Mode::CargoInstall => {
                let version = self
                    .cargo_version
                    .as_deref()
                    .ok_or("the cargo-install transaction has no exact target version")?;
                if semver::Version::parse(version).is_err() {
                    return Err("the cargo-install target is not a SemVer version".into());
                }
                if let Some(root) = &self.cargo_root {
                    paths.push(root.as_path());
                }
            }
        }
        if let Some(previous) = &self.stale_previous {
            paths.push(previous.as_path());
        }
        if let Some(relaunch) = &self.relaunch {
            paths.push(relaunch.program.as_path());
        }
        if let Some(registry) = &self.registry {
            paths.extend([
                registry.staged.as_path(),
                registry.target.as_path(),
                registry.backup.as_path(),
            ]);
        }
        if let Some(lock) = &self.plugin_lock {
            paths.push(lock.as_path());
        }
        if let Some(path) = &self.rollback_path {
            paths.push(path.as_path());
        }
        for plugin in &self.plugins {
            if plugin.staged == plugin.target || plugin.target == plugin.backup {
                return Err("plugin transaction paths must be distinct".into());
            }
            for path in &plugin.remove_after {
                if path == &plugin.target || path == &plugin.backup || path == &plugin.staged {
                    return Err("a plugin cleanup path overlaps an active transaction path".into());
                }
                paths.push(path.as_path());
            }
            paths.extend([
                plugin.staged.as_path(),
                plugin.target.as_path(),
                plugin.backup.as_path(),
            ]);
            if let Some(keep_as) = &plugin.keep_as {
                paths.push(keep_as.as_path());
            }
        }
        paths.extend(self.cleanup.iter().map(PathBuf::as_path));
        for path in paths {
            validate_absolute_path(path)?;
        }
        Ok(())
    }
}

/// Transaction paths must be absolute and free of `..`.
pub fn validate_absolute_path(path: &Path) -> Result<(), String> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(format!(
            "updater transaction contains an unsafe path: {}",
            path.display()
        ));
    }
    Ok(())
}

/// The outcome of a helper run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Outcome {
    /// The transaction was applied.
    Applied,
    /// Applying failed and everything was restored.
    RolledBack,
    /// Applying failed and the installation could not be fully restored;
    /// `message` says where the files are.
    Failed,
}

/// What the helper reports to the next eov launch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperReport {
    pub schema: u32,
    pub nonce: String,
    pub mode: Mode,
    pub outcome: Outcome,
    #[serde(default)]
    pub from_version: Option<String>,
    #[serde(default)]
    pub to_version: Option<String>,
    pub finished_at: u64,
    pub message: String,
    /// Where the (possibly restored) installation is.
    #[serde(default)]
    pub installation: Option<PathBuf>,
    /// Where the rollback copy is.
    #[serde(default)]
    pub previous: Option<PathBuf>,
}

impl HelperReport {
    /// Whether the update was applied (otherwise it failed, rolled back or
    /// not).
    pub fn applied(&self) -> bool {
        matches!(self.outcome, Outcome::Applied)
    }

    /// Whether a failed update left the installation partly replaced.
    pub fn left_damage(&self) -> bool {
        matches!(self.outcome, Outcome::Failed)
    }

    /// A one-paragraph summary for people.
    pub fn summary(&self) -> String {
        let versions = match (&self.from_version, &self.to_version) {
            (Some(from), Some(to)) => format!(" ({from} -> {to})"),
            _ => String::new(),
        };
        match self.outcome {
            Outcome::Applied => {
                format!("The last eov update{versions} completed. {}", self.message)
            }
            Outcome::RolledBack => format!(
                "The last eov update{versions} failed and was rolled back: {}",
                self.message
            ),
            Outcome::Failed => format!(
                "The last eov update{versions} failed and could not be fully rolled back: {}",
                self.message
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\Tools")
        } else {
            PathBuf::from("/opt/tools")
        }
    }

    pub(crate) fn sample() -> Transaction {
        let root = root();
        Transaction {
            schema: TRANSACTION_SCHEMA,
            nonce: "abc-123".into(),
            pid: 42,
            mode: Mode::ReplaceTree,
            from_version: Some("1.0.0".into()),
            to_version: Some("1.0.1".into()),
            current_tree: Some(root.join("eov")),
            staged_tree: Some(root.join(".eov-update-stage-abc")),
            previous_tree: Some(root.join(".eov-previous-abc")),
            stale_previous: None,
            cargo_version: None,
            cargo_root: None,
            relaunch: None,
            plugins: Vec::new(),
            registry: None,
            plugin_lock: None,
            report: root.join("last-update.toml"),
            rollback_record: None,
            rollback_path: None,
            cleanup: Vec::new(),
        }
    }

    #[test]
    fn round_trips_through_toml_and_validates() {
        let transaction = sample();
        transaction.validate().unwrap();
        let text = toml::to_string_pretty(&transaction).unwrap();
        let parsed: Transaction = toml::from_str(&text).unwrap();
        assert_eq!(parsed, transaction);
    }

    #[test]
    fn rejects_unknown_fields_relative_paths_and_bad_nonces() {
        let text = toml::to_string(&sample()).unwrap() + "url = \"https://example.invalid\"\n";
        assert!(toml::from_str::<Transaction>(&text).is_err());

        let mut relative = sample();
        relative.report = PathBuf::from("last-update.toml");
        assert!(relative.validate().is_err());

        let mut traversal = sample();
        traversal.cleanup = vec![root().join("..").join("x")];
        assert!(traversal.validate().is_err());

        let mut nonce = sample();
        nonce.nonce = "../evil".into();
        assert!(nonce.validate().is_err());
    }

    #[test]
    fn trees_must_be_distinct_siblings() {
        let mut same = sample();
        same.previous_tree = same.current_tree.clone();
        assert!(same.validate().is_err());
        let mut elsewhere = sample();
        elsewhere.staged_tree = Some(root().parent().unwrap().join("elsewhere").join("stage"));
        assert!(elsewhere.validate().is_err());
        let mut nested = sample();
        nested.staged_tree = Some(root().join(".eov-update-stage-abc").join("eov"));
        nested.validate().unwrap();
        let mut previous_elsewhere = sample();
        previous_elsewhere.previous_tree = Some(root().join("sub").join(".eov-previous"));
        assert!(previous_elsewhere.validate().is_err());
    }
}
