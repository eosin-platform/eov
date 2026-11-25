//! The hand-off between eov and `eov-update-helper`.
//!
//! This file is compiled into both binaries (the helper includes it with
//! `#[path]`), so the transaction format can never drift between them. eov
//! writes a fully resolved [`Transaction`] (local paths only: no URLs, no
//! version selection) into a private temporary directory, starts the helper
//! and exits; the helper waits for eov to exit, applies the transaction and
//! writes a [`HelperReport`] that the next `eov update` (or the GUI) shows.

#![allow(dead_code)] // each binary uses a different half of this module

use eov_plugin_api::i18n::{isolate, join_sentences, tr};
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
    /// What happened, in the helper's English words (the audit log, the
    /// command line; the window words it from the other fields).
    pub message: String,
    /// Where the (possibly restored) installation is.
    #[serde(default)]
    pub installation: Option<PathBuf>,
    /// Where the rollback copy is: the version before the update (also when
    /// a failed update left the installation damaged; older helpers left it
    /// out then).
    #[serde(default)]
    pub previous: Option<PathBuf>,
    /// When the helper could not put the previous version back: the folder
    /// eov was installed in, which `previous` must be renamed to (older
    /// helpers left it out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restore_to: Option<PathBuf>,
}

/// What the helper tells people when a failed update left the new version
/// at `current` and the previous one at `previous` (in English: the helper
/// has no translations; the window words it from the report's fields).
pub fn new_version_kept_message(current: &Path, previous: &Path) -> String {
    format!(
        "The new version is still at {} and the previous one is at {}. Close every program that uses eov, then move {} away and rename {} to {}.",
        current.display(),
        previous.display(),
        current.display(),
        previous.display(),
        current.display()
    )
}

/// How [`not_restored_message`] starts; why follows.
const NOT_RESTORED_START: &str = "eov could not be restored automatically (";

/// What the helper tells people when it could not rename the previous
/// version at `previous` back to `current`, the folder eov was installed in
/// (`error`: why; in English, as [`new_version_kept_message`]).
pub fn not_restored_message(
    error: impl std::fmt::Display,
    previous: &Path,
    current: &Path,
) -> String {
    format!(
        "{NOT_RESTORED_START}{error}). Your previous installation is at {}; rename it to {}.",
        previous.display(),
        current.display()
    )
}

/// Why the helper could not restore eov, when `line` is its
/// [`not_restored_message`] for these folders.
fn not_restored_reason<'a>(line: &'a str, previous: &Path, current: &Path) -> Option<&'a str> {
    let message = not_restored_message("", previous, current);
    let (start, end) = message.split_at(NOT_RESTORED_START.len());
    line.strip_prefix(start)?.strip_suffix(end)
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

    /// A one-paragraph summary in English, with the helper's words: for the
    /// command line and the helper's own message (it has no translations).
    /// The window shows [`Self::localized_summary`].
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

    /// The summary in the language of eov's interface, for the window. It
    /// is worded from what the report records, as the helper (which has no
    /// translations) may be of another eov version: what happened and, when
    /// the installation was left damaged, what to do. What only the helper's
    /// English words say (why the update failed) follows on lines of its
    /// own.
    pub fn localized_summary(&self) -> String {
        let (text, detail) = match (self.outcome, &self.installation, &self.previous) {
            (Outcome::Applied, _, Some(previous)) => (
                join_sentences([
                    self.outcome_sentence(),
                    // Translators: after "The last eov update completed."; {path} is a folder. "eov update --rollback" is a command: keep it as it is.
                    tr!(
                        "The previous version is kept at {path} for `eov update --rollback`.",
                        path = isolate(previous.display())
                    ),
                ]),
                String::new(),
            ),
            (Outcome::Applied, _, None) => (self.outcome_sentence(), String::new()),
            (Outcome::RolledBack, _, _) => (self.outcome_sentence(), self.message.clone()),
            // The new version is in place, the previous one beside it.
            (Outcome::Failed, Some(current), Some(previous)) if current != previous => (
                join_sentences([
                    self.outcome_sentence(),
                    // Translators: after "The last eov update failed and could not be fully rolled back."; {new} and {previous} are folders.
                    tr!(
                        "The new version is still at {new} and the previous one is at {previous}. Close every program that uses eov, then move {new} away and rename {previous} to {new}.",
                        new = isolate(current.display()),
                        previous = isolate(previous.display())
                    ),
                ]),
                // Without the steps, which the helper also wrote.
                self.message
                    .strip_suffix(&new_version_kept_message(current, previous))
                    .map(|detail| detail.trim_end_matches('\n'))
                    .unwrap_or(&self.message)
                    .to_string(),
            ),
            // eov is no longer where it was installed.
            (Outcome::Failed, _, Some(previous)) => match &self.restore_to {
                Some(current) => (
                    join_sentences([
                        self.outcome_sentence(),
                        // Translators: after "The last eov update failed and could not be fully rolled back."; {path} is the folder the previous version is in, {folder} the folder eov was installed in.
                        tr!(
                            "eov could not be restored automatically. Your previous installation is at {path}; rename it to {folder}.",
                            path = isolate(previous.display()),
                            folder = isolate(current.display())
                        ),
                    ]),
                    // Without the steps, which the helper also wrote; why
                    // it could not restore eov stays.
                    self.message
                        .lines()
                        .map(|line| not_restored_reason(line, previous, current).unwrap_or(line))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                // A helper that did not say where eov was installed.
                None => (
                    join_sentences([
                        self.outcome_sentence(),
                        // Translators: after "The last eov update failed and could not be fully rolled back."; {path} is a folder.
                        tr!(
                            "eov could not be restored automatically. Your previous installation is at {path}; rename it back to the folder eov was installed in.",
                            path = isolate(previous.display())
                        ),
                    ]),
                    self.message.clone(),
                ),
            },
            // A helper that did not say where things are.
            (Outcome::Failed, _, None) => (self.outcome_sentence(), self.message.clone()),
        };
        let detail = detail.trim();
        if detail.is_empty() {
            return text;
        }
        let detail: Vec<String> = detail.lines().map(isolate).collect();
        format!(
            "{text}\n{}",
            // Translators: followed by what the update helper wrote, in English (it has no translations).
            tr!("The update helper reported: {}", detail.join("\n"))
        )
    }

    /// The first sentence of the window's summary: how the update ended,
    /// with the versions when the report has them.
    fn outcome_sentence(&self) -> String {
        let versions = self.from_version.as_deref().zip(self.to_version.as_deref());
        match (self.outcome, versions) {
            // Translators: {from} and {to} are eov versions ("1.0.0", "1.1.0").
            (Outcome::Applied, Some((from, to))) => tr!(
                "The last eov update ({from} -> {to}) completed.",
                from = isolate(from),
                to = isolate(to)
            ),
            (Outcome::Applied, None) => tr!("The last eov update completed."),
            // Translators: {from} and {to} are eov versions ("1.0.0", "1.1.0").
            (Outcome::RolledBack, Some((from, to))) => tr!(
                "The last eov update ({from} -> {to}) failed and was rolled back.",
                from = isolate(from),
                to = isolate(to)
            ),
            (Outcome::RolledBack, None) => tr!("The last eov update failed and was rolled back."),
            // Translators: {from} and {to} are eov versions ("1.0.0", "1.1.0").
            (Outcome::Failed, Some((from, to))) => tr!(
                "The last eov update ({from} -> {to}) failed and could not be fully rolled back.",
                from = isolate(from),
                to = isolate(to)
            ),
            (Outcome::Failed, None) => {
                tr!("The last eov update failed and could not be fully rolled back.")
            }
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

    fn report(outcome: Outcome, message: &str) -> HelperReport {
        HelperReport {
            schema: REPORT_SCHEMA,
            nonce: "abc-123".into(),
            mode: Mode::ReplaceTree,
            outcome,
            from_version: Some("1.0.0".into()),
            to_version: Some("1.0.1".into()),
            finished_at: 1,
            message: message.into(),
            installation: Some(root().join("eov")),
            previous: None,
            restore_to: None,
        }
    }

    /// The window words the report from its fields, in the interface's
    /// language; the helper's English words only follow as the detail of a
    /// failure. The command line keeps the helper's summary.
    #[test]
    fn reports_are_worded_from_their_fields() {
        let (current, previous) = (root().join("eov"), root().join(".eov-previous-abc"));
        let mut applied = report(
            Outcome::Applied,
            &format!(
                "The previous version is kept at {} for `eov update --rollback`.",
                previous.display()
            ),
        );
        applied.previous = Some(previous.clone());
        assert_eq!(applied.localized_summary(), applied.summary());
        let mut cargo = report(Outcome::Applied, "The update was applied.");
        cargo.mode = Mode::CargoInstall;
        cargo.installation = None;
        assert_eq!(
            cargo.localized_summary(),
            "The last eov update (1.0.0 -> 1.0.1) completed."
        );
        assert_eq!(
            cargo.summary(),
            "The last eov update (1.0.0 -> 1.0.1) completed. The update was applied."
        );
        // A report without both versions names none.
        let mut unversioned = cargo.clone();
        unversioned.to_version = None;
        assert_eq!(
            unversioned.localized_summary(),
            "The last eov update completed."
        );

        let rolled_back = report(Outcome::RolledBack, "a file was locked");
        assert_eq!(
            rolled_back.localized_summary(),
            "The last eov update (1.0.0 -> 1.0.1) failed and was rolled back.\n\
             The update helper reported: a file was locked"
        );
        assert_eq!(
            rolled_back.summary(),
            "The last eov update (1.0.0 -> 1.0.1) failed and was rolled back: a file was locked"
        );

        // The new version stayed: the steps are worded once, the helper's
        // reason follows.
        let steps = new_version_kept_message(&current, &previous);
        let mut kept = report(
            Outcome::Failed,
            &format!("could not activate the staged plugin registry\n{steps}"),
        );
        kept.previous = Some(previous.clone());
        let shown = kept.localized_summary();
        assert!(
            shown.starts_with(&format!(
                "The last eov update (1.0.0 -> 1.0.1) failed and could not be fully rolled back. {steps}\n"
            )),
            "{shown}"
        );
        assert!(shown.ends_with(
            "The update helper reported: could not activate the staged plugin registry"
        ));
        assert_eq!(shown.matches("Close every program").count(), 1);

        // eov is gone from its folder: an older helper did not say where
        // that was.
        let mut gone = report(
            Outcome::Failed,
            "x\neov could not be restored automatically",
        );
        gone.installation = Some(previous.clone());
        gone.previous = Some(previous.clone());
        assert!(gone.localized_summary().contains(&format!(
            "Your previous installation is at {}; rename it back to the folder eov was installed in.",
            previous.display()
        )));
        // This one does: the window names it, and leaves out the steps the
        // helper wrote, but not why it could not restore eov.
        let mut gone = report(
            Outcome::Failed,
            &format!(
                "could not activate the staged plugin registry\n{}",
                not_restored_message("access denied", &previous, &current)
            ),
        );
        gone.installation = Some(previous.clone());
        gone.previous = Some(previous.clone());
        gone.restore_to = Some(current.clone());
        assert_eq!(
            gone.localized_summary(),
            format!(
                "The last eov update (1.0.0 -> 1.0.1) failed and could not be fully rolled back. \
                 eov could not be restored automatically. Your previous installation is at {}; \
                 rename it to {}.\nThe update helper reported: could not activate the staged \
                 plugin registry\naccess denied",
                previous.display(),
                current.display()
            )
        );
        // The command line keeps the helper's words, and `eov update
        // --json` has the folder too.
        assert_eq!(
            gone.summary(),
            format!(
                "The last eov update (1.0.0 -> 1.0.1) failed and could not be fully rolled back: \
                 could not activate the staged plugin registry\neov could not be restored \
                 automatically (access denied). Your previous installation is at {}; rename it \
                 to {}.",
                previous.display(),
                current.display()
            )
        );
        let json = serde_json::to_value(&gone).unwrap();
        assert_eq!(json["restore_to"], current.display().to_string());
        assert!(
            serde_json::to_value(report(Outcome::RolledBack, "x"))
                .unwrap()
                .get("restore_to")
                .is_none(),
            "only when there is one"
        );
        // A helper that did not say where things are: its words.
        let old = report(Outcome::Failed, "something happened");
        assert_eq!(
            old.localized_summary(),
            "The last eov update (1.0.0 -> 1.0.1) failed and could not be fully rolled back.\n\
             The update helper reported: something happened"
        );
    }

    /// In another language the window's summary is that language's, with
    /// only the helper's own reason in English, isolated in right-to-left
    /// languages.
    #[test]
    fn reports_are_shown_in_the_interface_s_language() {
        use eov_plugin_api::i18n::{self, Catalog};
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let _english = English;
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 2] = [
            (
                None,
                "The last eov update ({from} -> {to}) failed and was rolled back.",
                &[
                    "Das letzte eov-Update ({from} -> {to}) ist fehlgeschlagen und wurde zurückgenommen.",
                ],
            ),
            (
                None,
                "The update helper reported: {}",
                &["Das Aktualisierungsprogramm meldete: {}"],
            ),
        ];
        i18n::set_catalog(
            german,
            Some(Catalog::from_messages(german, "", messages).unwrap()),
        );
        let rolled_back = report(Outcome::RolledBack, "a file was locked");
        assert_eq!(
            rolled_back.localized_summary(),
            "Das letzte eov-Update (1.0.0 -> 1.0.1) ist fehlgeschlagen und wurde \
             zurückgenommen.\nDas Aktualisierungsprogramm meldete: a file was locked"
        );
        i18n::set_catalog(i18n::by_tag("he").unwrap(), None);
        assert!(
            rolled_back
                .localized_summary()
                .ends_with("\u{2068}a file was locked\u{2069}")
        );
        // The command line's (and the audit log's) words do not change.
        assert_eq!(
            rolled_back.summary(),
            "The last eov update (1.0.0 -> 1.0.1) failed and was rolled back: a file was locked"
        );
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
