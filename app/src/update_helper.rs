//! Local post-exit updater helper.
//!
//! This binary intentionally has no release discovery, HTTP, telemetry, or
//! version-selection logic. It applies only a fully resolved local transaction
//! prepared by the main EOV process.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const TRANSACTION_SCHEMA: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Transaction {
    schema: u32,
    nonce: String,
    pid: u32,
    mode: String,
    #[serde(default)]
    current_tree: Option<PathBuf>,
    #[serde(default)]
    staged_tree: Option<PathBuf>,
    #[serde(default)]
    backup_tree: Option<PathBuf>,
    #[serde(default)]
    cargo_version: Option<String>,
    #[serde(default)]
    relaunch: Option<PathBuf>,
    #[serde(default)]
    plugins: Vec<PluginReplacement>,
    #[serde(default)]
    registry: Option<RegistryReplacement>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginReplacement {
    staged: PathBuf,
    target: PathBuf,
    backup: PathBuf,
    #[serde(default)]
    remove_after: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryReplacement {
    staged: PathBuf,
    target: PathBuf,
    backup: PathBuf,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let mode = args.next().context("missing helper mode")?;
    if mode != "replace-tree" && mode != "cargo-install" {
        bail!("unsupported updater helper mode");
    }
    let flag = args.next().context("missing transaction flag")?;
    if flag != "--transaction" {
        bail!("expected --transaction <path>");
    }
    let transaction_path = PathBuf::from(args.next().context("missing transaction path")?);
    if args.next().is_some() {
        bail!("unexpected updater helper arguments");
    }
    let transaction: Transaction = toml::from_str(
        &fs::read_to_string(&transaction_path)
            .with_context(|| format!("could not read {}", transaction_path.display()))?,
    )
    .context("could not parse updater transaction")?;
    validate_transaction(&transaction)?;
    wait_for_process(transaction.pid)?;
    match transaction.mode.as_str() {
        "replace-tree" => replace_tree(&transaction)?,
        "cargo-install" => cargo_install(&transaction)?,
        _ => bail!("unsupported updater transaction mode"),
    }
    let _ = fs::remove_file(&transaction_path);
    if let Some(relaunch) = transaction.relaunch {
        Command::new(relaunch)
            .spawn()
            .context("could not relaunch EOV after update")?;
    }
    Ok(())
}

fn validate_transaction(transaction: &Transaction) -> Result<()> {
    if transaction.schema != TRANSACTION_SCHEMA {
        bail!("unsupported updater transaction schema");
    }
    if transaction.nonce.is_empty()
        || transaction.nonce.contains('/')
        || transaction.nonce.contains('\\')
    {
        bail!("invalid updater transaction nonce");
    }
    match transaction.mode.as_str() {
        "replace-tree" => {
            let current_tree = transaction
                .current_tree
                .as_ref()
                .context("replace-tree transaction has no current tree")?;
            let staged_tree = transaction
                .staged_tree
                .as_ref()
                .context("replace-tree transaction has no staged tree")?;
            let backup_tree = transaction
                .backup_tree
                .as_ref()
                .context("replace-tree transaction has no backup tree")?;
            for path in [current_tree, staged_tree, backup_tree] {
                validate_absolute_path(path)?;
            }
            if current_tree == staged_tree
                || current_tree == backup_tree
                || staged_tree == backup_tree
            {
                bail!("updater transaction paths must be distinct");
            }
            if !staged_tree.is_dir() {
                bail!("updater staging tree does not exist");
            }
        }
        "cargo-install" => {
            if cfg!(not(target_os = "windows")) {
                bail!("cargo-install helper mode is only available on Windows");
            }
            let version = transaction
                .cargo_version
                .as_deref()
                .context("cargo-install transaction has no exact target version")?;
            semver::Version::parse(version).context("cargo-install target is not SemVer")?;
        }
        _ => bail!("unsupported updater transaction mode"),
    }
    if let Some(registry) = &transaction.registry {
        for path in [&registry.staged, &registry.target, &registry.backup] {
            validate_absolute_path(path)?;
        }
        if !registry.staged.is_file() {
            bail!("staged plugin registry does not exist");
        }
    }
    for plugin in &transaction.plugins {
        validate_absolute_path(&plugin.staged)?;
        validate_absolute_path(&plugin.target)?;
        validate_absolute_path(&plugin.backup)?;
        if plugin.staged == plugin.target || plugin.target == plugin.backup {
            bail!("plugin transaction paths must be distinct");
        }
        if !plugin.staged.is_file() {
            bail!("staged plugin package does not exist");
        }
        for path in &plugin.remove_after {
            validate_absolute_path(path)?;
            if path == &plugin.target || path == &plugin.backup || path == &plugin.staged {
                bail!("plugin cleanup path overlaps an active transaction path");
            }
        }
    }
    Ok(())
}

fn validate_absolute_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        bail!(
            "updater transaction contains an unsafe path: {}",
            path.display()
        );
    }
    Ok(())
}

fn wait_for_process(pid: u32) -> Result<()> {
    for _ in 0..600 {
        if !process_is_running(pid) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!("timed out waiting for EOV process {pid} to exit")
}

fn process_is_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if result == 0 {
            return true;
        }
        return std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle == 0 {
            return false;
        }
        unsafe { CloseHandle(handle) };
        return true;
    }
    #[allow(unreachable_code)]
    false
}

fn replace_tree(transaction: &Transaction) -> Result<()> {
    let current_tree = transaction
        .current_tree
        .as_ref()
        .context("missing current tree")?;
    let staged_tree = transaction
        .staged_tree
        .as_ref()
        .context("missing staged tree")?;
    let backup_tree = transaction
        .backup_tree
        .as_ref()
        .context("missing backup tree")?;
    if !current_tree.exists() {
        bail!("current EOV installation tree does not exist");
    }
    if backup_tree.exists() {
        fs::remove_dir_all(backup_tree).context("could not remove stale updater backup")?;
    }
    fs::rename(current_tree, backup_tree)
        .context("could not move current EOV installation to backup")?;
    if let Err(error) = fs::rename(staged_tree, current_tree) {
        let _ = fs::rename(backup_tree, current_tree);
        return Err(error).context("could not activate staged EOV installation");
    }

    let mut applied_plugins = Vec::new();
    for plugin in &transaction.plugins {
        if let Err(error) = apply_plugin(plugin) {
            let _ = fs::remove_dir_all(current_tree);
            let _ = fs::rename(backup_tree, current_tree);
            rollback_plugins(applied_plugins);
            return Err(error);
        }
        applied_plugins.push(plugin);
    }

    let mut cleanup_backups = Vec::new();
    for plugin in &transaction.plugins {
        for path in &plugin.remove_after {
            if path.exists() {
                let backup = path.with_file_name(format!(
                    ".{}.cleanup-{}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("plugin"),
                    transaction.nonce
                ));
                if let Err(error) = fs::rename(path, &backup) {
                    restore_cleanup_backups(&cleanup_backups);
                    let _ = fs::remove_dir_all(current_tree);
                    let _ = fs::rename(backup_tree, current_tree);
                    rollback_plugins(applied_plugins);
                    return Err(error)
                        .context("could not stage a superseded plugin package removal");
                }
                cleanup_backups.push((path.clone(), backup));
            }
        }
    }

    if let Err(error) = apply_registry(transaction.registry.as_ref()) {
        restore_cleanup_backups(&cleanup_backups);
        let _ = fs::remove_dir_all(current_tree);
        let _ = fs::rename(backup_tree, current_tree);
        rollback_plugins(applied_plugins);
        return Err(error);
    }
    remove_cleanup_backups(&cleanup_backups);
    cleanup_plugin_backups(&transaction.plugins);
    if let Some(registry) = &transaction.registry {
        let _ = fs::remove_file(&registry.backup);
    }
    let _ = fs::remove_dir_all(backup_tree);
    Ok(())
}

fn restore_cleanup_backups(backups: &[(PathBuf, PathBuf)]) {
    for (original, backup) in backups.iter().rev() {
        if backup.exists() {
            let _ = fs::rename(backup, original);
        }
    }
}

fn cargo_install(transaction: &Transaction) -> Result<()> {
    let version = transaction
        .cargo_version
        .as_deref()
        .context("missing cargo target")?;
    let status = Command::new("cargo")
        .args([
            "install",
            "eov",
            "--version",
            version,
            "--locked",
            "--force",
        ])
        .status()
        .context("could not invoke Cargo for the EOV update")?;
    if !status.success() {
        bail!("cargo install failed with status {status}");
    }
    let applied = apply_plugins(&transaction.plugins)?;
    let cleanup_backups = match stage_cleanup_paths(&transaction.plugins) {
        Ok(backups) => backups,
        Err(error) => {
            rollback_plugins(applied);
            return Err(error);
        }
    };
    if let Some(registry) = &transaction.registry
        && let Err(error) = apply_registry(Some(registry))
    {
        restore_cleanup_backups(&cleanup_backups);
        rollback_plugins(applied);
        return Err(error);
    }
    remove_cleanup_backups(&cleanup_backups);
    cleanup_plugin_backups(&transaction.plugins);
    if let Some(registry) = &transaction.registry {
        let _ = fs::remove_file(&registry.backup);
    }
    Ok(())
}

fn apply_plugins(plugins: &[PluginReplacement]) -> Result<Vec<&PluginReplacement>> {
    let mut applied = Vec::new();
    for plugin in plugins {
        if let Err(error) = apply_plugin(plugin) {
            rollback_plugins(applied);
            return Err(error);
        }
        applied.push(plugin);
    }
    Ok(applied)
}

fn stage_cleanup_paths(plugins: &[PluginReplacement]) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut backups = Vec::new();
    for plugin in plugins {
        for path in &plugin.remove_after {
            if !path.exists() {
                continue;
            }
            let backup = path.with_file_name(format!(
                ".{}.cleanup-cargo",
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("plugin")
            ));
            if let Err(error) = fs::rename(path, &backup) {
                restore_cleanup_backups(&backups);
                return Err(error).context("could not stage a superseded plugin package removal");
            }
            backups.push((path.clone(), backup));
        }
    }
    Ok(backups)
}

fn remove_cleanup_backups(backups: &[(PathBuf, PathBuf)]) {
    for (_, backup) in backups {
        let _ = fs::remove_file(backup);
    }
}

fn cleanup_plugin_backups(plugins: &[PluginReplacement]) {
    for plugin in plugins {
        let _ = fs::remove_file(&plugin.backup);
    }
}

fn apply_registry(registry: Option<&RegistryReplacement>) -> Result<()> {
    let Some(registry) = registry else {
        return Ok(());
    };
    if registry.backup.exists() {
        fs::remove_file(&registry.backup)?;
    }
    if registry.target.exists() {
        fs::rename(&registry.target, &registry.backup)?;
    }
    if let Err(error) = fs::rename(&registry.staged, &registry.target) {
        if registry.backup.exists() {
            let _ = fs::rename(&registry.backup, &registry.target);
        }
        return Err(error).context("could not activate staged plugin registry");
    }
    Ok(())
}

fn apply_plugin(plugin: &PluginReplacement) -> Result<()> {
    if plugin.backup.exists() {
        fs::remove_file(&plugin.backup)?;
    }
    if plugin.target.exists() {
        fs::rename(&plugin.target, &plugin.backup)
            .context("could not back up an installed plugin package")?;
    }
    if let Err(error) = fs::rename(&plugin.staged, &plugin.target) {
        if plugin.backup.exists() {
            let _ = fs::rename(&plugin.backup, &plugin.target);
        }
        return Err(error).context("could not activate staged plugin package");
    }
    Ok(())
}

fn rollback_plugins(plugins: Vec<&PluginReplacement>) {
    for plugin in plugins.into_iter().rev() {
        let _ = fs::remove_file(&plugin.target);
        if plugin.backup.exists() {
            let _ = fs::rename(&plugin.backup, &plugin.target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn rejects_remote_or_unsafe_transaction_values() {
        let temp = tempdir().unwrap();
        let staged = temp.path().join("staged");
        fs::create_dir(&staged).unwrap();
        let transaction = Transaction {
            schema: TRANSACTION_SCHEMA,
            nonce: "nonce".into(),
            pid: 0,
            mode: "replace-tree".into(),
            current_tree: Some(temp.path().join("current")),
            staged_tree: Some(staged),
            backup_tree: Some(temp.path().join("backup")),
            cargo_version: None,
            relaunch: None,
            plugins: Vec::new(),
            registry: None,
        };
        assert!(validate_transaction(&transaction).is_ok());
        let mut unsafe_transaction = transaction;
        unsafe_transaction.nonce = "https://github.com/example".into();
        assert!(validate_transaction(&unsafe_transaction).is_err());
    }

    #[test]
    fn replace_tree_swaps_staged_directory() {
        let temp = tempdir().unwrap();
        let current = temp.path().join("current");
        let staged = temp.path().join("staged");
        let backup = temp.path().join("backup");
        fs::create_dir(&current).unwrap();
        fs::create_dir(&staged).unwrap();
        fs::write(current.join("version"), "old").unwrap();
        fs::write(staged.join("version"), "new").unwrap();
        let transaction = Transaction {
            schema: TRANSACTION_SCHEMA,
            nonce: "nonce".into(),
            pid: 0,
            mode: "replace-tree".into(),
            current_tree: Some(current.clone()),
            staged_tree: Some(staged),
            backup_tree: Some(backup),
            cargo_version: None,
            relaunch: None,
            plugins: Vec::new(),
            registry: None,
        };
        replace_tree(&transaction).unwrap();
        assert_eq!(fs::read_to_string(current.join("version")).unwrap(), "new");
    }

    #[test]
    fn plugin_replacement_is_local_only() {
        let temp = tempdir().unwrap();
        let staged = temp.path().join("new.eop");
        let target = temp.path().join("plugin.eop");
        let backup = temp.path().join("plugin.backup");
        File::create(&staged).unwrap();
        fs::write(&target, b"old").unwrap();
        let replacement = PluginReplacement {
            staged,
            target: target.clone(),
            backup,
            remove_after: Vec::new(),
        };
        apply_plugin(&replacement).unwrap();
        assert!(replacement.target.exists());
        assert_eq!(fs::read(&replacement.target).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn transaction_rejects_unknown_url_fields() {
        let transaction = r#"
schema = 1
nonce = "nonce"
pid = 0
mode = "replace-tree"
current_tree = "/tmp/current"
staged_tree = "/tmp/staged"
backup_tree = "/tmp/backup"
url = "https://example.invalid/update"
"#;
        assert!(toml::from_str::<Transaction>(transaction).is_err());
    }
}
