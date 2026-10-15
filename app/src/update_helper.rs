//! Local post-exit updater helper.
//!
//! This binary intentionally has no release discovery, HTTP, telemetry, or
//! version-selection logic. It applies only a fully resolved local
//! transaction prepared by the main eov process (see
//! `update/transaction.rs`, which both binaries compile):
//!
//! 1. Wait until eov has exited: on an inherited pipe (Unix) or process
//!    handle (Windows), so a recycled PID can never be mistaken for eov.
//! 2. Swap the installation tree and activate plugin packages and the
//!    registry, retrying renames that antivirus scanners or indexers block
//!    for a moment, and restoring everything if a step fails.
//! 3. Keep the replaced tree next to the installation as the rollback copy
//!    (`eov update --rollback`); only directories marked as eov
//!    installations are ever deleted.
//! 4. Write a report for the next eov launch, tell the user about failures
//!    (a message box on Windows, an alert on macOS), and relaunch eov.

#[path = "update/transaction.rs"]
mod transaction;

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use transaction::{
    HelperReport, Mode, Outcome, PluginReplacement, REPORT_SCHEMA, RegistryReplacement,
    TREE_MARKER, Transaction,
};

/// How long to wait for eov to exit before giving up.
const PARENT_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// How long to keep retrying a rename that fails with a transient error.
const RENAME_RETRY_BUDGET: Duration = Duration::from_secs(30);

/// How the helper learns that eov exited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitFor {
    /// End of file on an inherited pipe whose write end only eov holds.
    Fd(i32),
    /// An inherited, synchronisable handle to the eov process.
    Handle(usize),
    /// The process id (fallback only).
    Pid,
}

#[derive(Debug, PartialEq, Eq)]
enum Invocation {
    SelfTest,
    Apply { transaction: PathBuf, wait: WaitFor },
}

fn parse_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Invocation> {
    let mut args = args.into_iter().skip(1);
    let mode = args.next().context("missing helper mode")?;
    match mode.to_str() {
        Some("--self-test") => return Ok(Invocation::SelfTest),
        // `replace-tree` and `cargo-install` are accepted for compatibility.
        Some("apply" | "replace-tree" | "cargo-install") => {}
        _ => bail!("unsupported updater helper mode"),
    }
    let mut transaction = None;
    let mut wait = WaitFor::Pid;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("{} needs a value", flag.to_string_lossy()))?;
        match flag.to_str() {
            Some("--transaction") => transaction = Some(PathBuf::from(value)),
            Some("--wait-fd") => {
                wait = WaitFor::Fd(
                    value
                        .to_str()
                        .and_then(|value| value.parse().ok())
                        .context("invalid --wait-fd")?,
                );
            }
            Some("--wait-handle") => {
                wait = WaitFor::Handle(
                    value
                        .to_str()
                        .and_then(|value| value.parse().ok())
                        .context("invalid --wait-handle")?,
                );
            }
            _ => bail!("unexpected updater helper argument"),
        }
    }
    Ok(Invocation::Apply {
        transaction: transaction.context("expected --transaction <path>")?,
        wait,
    })
}

fn main() -> ExitCode {
    let invocation = match parse_args(std::env::args_os()) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("eov-update-helper: {error:#}");
            return ExitCode::from(2);
        }
    };
    let (transaction_path, wait) = match invocation {
        Invocation::SelfTest => {
            // eov runs this before it exits, to prove the helper can start
            // from its temporary location (for example that dyld finds every
            // library on macOS).
            println!("eov-update-helper {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Invocation::Apply { transaction, wait } => (transaction, wait),
    };
    let transaction = match read_transaction(&transaction_path) {
        Ok(transaction) => transaction,
        Err(error) => {
            notify_failure(&format!(
                "The eov update could not start: {error:#}\nNothing was changed."
            ));
            return ExitCode::FAILURE;
        }
    };
    let report = run(&transaction, wait);
    if report.outcome == Outcome::Applied
        && let (Some(record), Some(path)) =
            (&transaction.rollback_record, &transaction.rollback_path)
    {
        let _ = write_atomically(path, record);
    }
    let _ = write_report(&transaction.report, &report);
    let _ = fs::remove_file(&transaction_path);
    for path in &transaction.cleanup {
        remove_any(path);
    }
    if report.outcome != Outcome::Applied {
        notify_failure(&report.summary());
    }
    if report.outcome != Outcome::Failed
        && let Some(relaunch) = &transaction.relaunch
        && let Err(error) = relaunch_eov(&relaunch.program, &relaunch.args)
    {
        eprintln!("eov-update-helper: could not relaunch eov: {error:#}");
    }
    match report.outcome {
        Outcome::Applied => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

fn read_transaction(path: &Path) -> Result<Transaction> {
    let text =
        fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    let transaction: Transaction =
        toml::from_str(&text).context("could not parse the updater transaction")?;
    transaction.validate().map_err(anyhow::Error::msg)?;
    Ok(transaction)
}

fn run(transaction: &Transaction, wait: WaitFor) -> HelperReport {
    let mut report = HelperReport {
        schema: REPORT_SCHEMA,
        nonce: transaction.nonce.clone(),
        mode: transaction.mode,
        outcome: Outcome::RolledBack,
        from_version: transaction.from_version.clone(),
        to_version: transaction.to_version.clone(),
        finished_at: 0,
        message: String::new(),
        installation: transaction.current_tree.clone(),
        previous: None,
    };
    let result = (|| -> Result<Outcome> {
        wait_for_parent(wait, transaction.pid, PARENT_TIMEOUT)?;
        let _lock = match &transaction.plugin_lock {
            Some(path) => Some(lock_file(path)?),
            None => None,
        };
        match transaction.mode {
            Mode::ReplaceTree | Mode::SwapTree => apply_tree(transaction, &mut report),
            Mode::CargoInstall => apply_cargo(transaction),
        }
    })();
    report.finished_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    match result {
        Ok(outcome) => {
            report.outcome = outcome;
            if report.message.is_empty() {
                report.message = "The update was applied.".into();
            }
        }
        Err(error) => {
            report.outcome = if report.outcome == Outcome::Failed {
                Outcome::Failed
            } else {
                Outcome::RolledBack
            };
            let detail = format!("{error:#}");
            report.message = if report.message.is_empty() {
                detail
            } else {
                format!("{detail}\n{}", report.message)
            };
        }
    }
    report
}

/// Replaces (or, for a rollback, swaps) the installation tree, then the
/// plugins and registry. Returns `Applied`, or an error after restoring.
fn apply_tree(transaction: &Transaction, report: &mut HelperReport) -> Result<Outcome> {
    let current = transaction
        .current_tree
        .as_deref()
        .context("no current tree")?;
    let staged = transaction
        .staged_tree
        .as_deref()
        .context("no staged tree")?;
    let previous = transaction
        .previous_tree
        .as_deref()
        .context("no previous-tree location")?;
    if !current.is_dir() {
        bail!("the eov installation {} does not exist", current.display());
    }
    if !staged.is_dir() {
        bail!(
            "the staged installation {} does not exist",
            staged.display()
        );
    }
    if previous.exists() {
        bail!(
            "{} already exists; refusing to overwrite it",
            previous.display()
        );
    }

    // 1. Move the running installation aside. Nothing changed if this fails.
    rename_with_retry(current, previous).with_context(|| {
        format!(
            "could not move {} aside (is a file in it open, for example a terminal in that folder?)",
            current.display()
        )
    })?;
    // 2. Activate the staged (or kept previous) tree.
    if let Err(error) = rename_with_retry(staged, current) {
        return Err(restore_tree(current, previous, None, report, error.into()));
    }
    let _ = fs::write(
        current.join(TREE_MARKER),
        transaction.to_version.as_deref().unwrap_or_default(),
    );
    let _ = fs::write(
        previous.join(TREE_MARKER),
        transaction.from_version.as_deref().unwrap_or_default(),
    );

    // 3. Plugins and registry.
    if let Err(error) = apply_plugins_and_registry(transaction) {
        return Err(restore_tree(current, previous, Some(staged), report, error));
    }

    // 4. The update stands: drop the stale rollback copy, keep this one.
    if let Some(stale) = &transaction.stale_previous
        && stale != previous
        && stale != current
    {
        remove_installation_tree(stale);
    }
    report.previous = Some(previous.to_path_buf());
    report.message = format!(
        "The previous version is kept at {} for `eov update --rollback`.",
        previous.display()
    );
    Ok(Outcome::Applied)
}

/// Puts the previous tree back at `current`. When `parked` is given, the
/// new tree is first moved there (it was activated already).
fn restore_tree(
    current: &Path,
    previous: &Path,
    parked: Option<&Path>,
    report: &mut HelperReport,
    error: anyhow::Error,
) -> anyhow::Error {
    if let Some(parked) = parked
        && current.exists()
        && rename_with_retry(current, parked).is_err()
    {
        report.outcome = Outcome::Failed;
        report.message = format!(
            "The new version is still at {} and the previous one is at {}. Close every program that uses eov, then move {} away and rename {} to {}.",
            current.display(),
            previous.display(),
            current.display(),
            previous.display(),
            current.display()
        );
        return error;
    }
    if let Err(restore_error) = rename_with_retry(previous, current) {
        report.outcome = Outcome::Failed;
        report.message = format!(
            "eov could not be restored automatically ({restore_error}). Your previous installation is at {}; rename it to {}.",
            previous.display(),
            current.display()
        );
        report.installation = Some(previous.to_path_buf());
    }
    error
}

fn apply_plugins_and_registry(transaction: &Transaction) -> Result<()> {
    let mut applied: Vec<&PluginReplacement> = Vec::new();
    for plugin in &transaction.plugins {
        if let Err(error) = apply_plugin(plugin) {
            rollback_plugins(&applied);
            return Err(error);
        }
        applied.push(plugin);
    }
    let mut cleanup = Vec::new();
    for plugin in &transaction.plugins {
        for path in &plugin.remove_after {
            if !path.exists() {
                continue;
            }
            let parked = sibling(path, &format!("cleanup-{}", transaction.nonce));
            if let Err(error) = rename_with_retry(path, &parked) {
                restore_parked(&cleanup);
                rollback_plugins(&applied);
                return Err(error).context("could not remove a superseded plugin package");
            }
            cleanup.push((path.clone(), parked));
        }
    }
    if let Err(error) = apply_registry(transaction.registry.as_ref()) {
        restore_parked(&cleanup);
        rollback_plugins(&applied);
        return Err(error);
    }
    for (_, parked) in cleanup {
        let _ = fs::remove_file(parked);
    }
    for plugin in &transaction.plugins {
        let kept = plugin.keep_as.as_ref().is_some_and(|keep_as| {
            plugin.backup.exists()
                && keep_as
                    .parent()
                    .is_none_or(|dir| fs::create_dir_all(dir).is_ok())
                && fs::rename(&plugin.backup, keep_as).is_ok()
        });
        if !kept {
            let _ = fs::remove_file(&plugin.backup);
        }
    }
    if let Some(registry) = &transaction.registry {
        let _ = fs::remove_file(&registry.backup);
    }
    Ok(())
}

fn apply_cargo(transaction: &Transaction) -> Result<Outcome> {
    let version = transaction
        .cargo_version
        .as_deref()
        .context("no cargo target version")?;
    let mut command = Command::new("cargo");
    command.args([
        "install",
        "eov",
        "--version",
        version,
        "--locked",
        "--force",
    ]);
    if let Some(root) = &transaction.cargo_root {
        command.arg("--root").arg(root);
    }
    let status = command
        .status()
        .context("could not run cargo for the eov update")?;
    if !status.success() {
        bail!("cargo install failed with {status}; eov was not changed");
    }
    apply_plugins_and_registry(transaction)?;
    Ok(Outcome::Applied)
}

fn apply_registry(registry: Option<&RegistryReplacement>) -> Result<()> {
    let Some(registry) = registry else {
        return Ok(());
    };
    if registry.backup.exists() {
        fs::remove_file(&registry.backup)?;
    }
    if registry.target.exists() {
        rename_with_retry(&registry.target, &registry.backup)
            .context("could not back up the plugin registry")?;
    }
    if let Err(error) = rename_with_retry(&registry.staged, &registry.target) {
        if registry.backup.exists() {
            let _ = rename_with_retry(&registry.backup, &registry.target);
        }
        return Err(error).context("could not activate the staged plugin registry");
    }
    Ok(())
}

fn apply_plugin(plugin: &PluginReplacement) -> Result<()> {
    if plugin.backup.exists() {
        fs::remove_file(&plugin.backup)?;
    }
    if plugin.target.exists() {
        rename_with_retry(&plugin.target, &plugin.backup)
            .context("could not back up an installed plugin package")?;
    }
    if let Err(error) = rename_with_retry(&plugin.staged, &plugin.target) {
        if plugin.backup.exists() {
            let _ = rename_with_retry(&plugin.backup, &plugin.target);
        }
        return Err(error).context("could not activate a staged plugin package");
    }
    Ok(())
}

fn rollback_plugins(plugins: &[&PluginReplacement]) {
    for plugin in plugins.iter().rev() {
        // Put the new package back where eov staged it, then the old one.
        let _ = rename_with_retry(&plugin.target, &plugin.staged);
        if plugin.backup.exists() {
            let _ = rename_with_retry(&plugin.backup, &plugin.target);
        }
    }
}

fn restore_parked(parked: &[(PathBuf, PathBuf)]) {
    for (original, parked) in parked.iter().rev() {
        if parked.exists() {
            let _ = rename_with_retry(parked, original);
        }
    }
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "item".into());
    path.with_file_name(format!(".{name}.{suffix}"))
}

/// Deletes a rollback copy, but only if it is provably an eov installation.
fn remove_installation_tree(path: &Path) {
    let marked = path.join(TREE_MARKER).is_file();
    let legacy = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(".eov-backup-") || name.starts_with(".eov-previous-"))
        && (path.join("eov.exe").is_file() || path.join("Contents/MacOS/eov").is_file());
    if (marked || legacy) && fs::remove_dir_all(path).is_err() {
        eprintln!(
            "eov-update-helper: could not remove the old rollback copy {}",
            path.display()
        );
    }
}

fn remove_any(path: &Path) {
    if path.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else {
        let _ = fs::remove_file(path);
    }
}

/// Renames, retrying for a while when another process (an antivirus
/// scanner, the search indexer, a terminal sitting in the folder) briefly
/// holds a file open.
fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let start = Instant::now();
    let mut delay = Duration::from_millis(50);
    loop {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(error) if is_transient(&error) && start.elapsed() < RENAME_RETRY_BUDGET => {
                thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_secs(2));
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_transient(error: &std::io::Error) -> bool {
    if cfg!(windows) {
        // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
        matches!(error.raw_os_error(), Some(5 | 32 | 33))
    } else {
        // EBUSY, ETXTBSY
        matches!(error.raw_os_error(), Some(16 | 26))
    }
}

fn lock_file(path: &Path) -> Result<fs::File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(fs::TryLockError::WouldBlock) if start.elapsed() < PARENT_TIMEOUT => {
                thread::sleep(Duration::from_millis(250));
            }
            Err(fs::TryLockError::WouldBlock) => {
                bail!("another eov process kept {} locked", path.display())
            }
            Err(fs::TryLockError::Error(error)) => {
                return Err(error).with_context(|| format!("could not lock {}", path.display()));
            }
        }
    }
}

fn write_report(path: &Path, report: &HelperReport) -> Result<()> {
    write_atomically(path, &toml::to_string_pretty(report)?)
}

fn write_atomically(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn relaunch_eov(program: &Path, args: &[String]) -> Result<()> {
    let mut command = if cfg!(target_os = "macos")
        && let Some(bundle) = macos_bundle_of(program)
    {
        // LaunchServices starts the bundle like a double-click would.
        let mut command = Command::new("/usr/bin/open");
        command.arg("-n").arg(bundle);
        if !args.is_empty() {
            command.arg("--args").args(args);
        }
        command
    } else {
        let mut command = Command::new(program);
        command.args(args);
        command
    };
    if let Some(parent) = program.parent() {
        command.current_dir(parent);
    }
    command.spawn().context("could not start eov")?;
    Ok(())
}

fn macos_bundle_of(program: &Path) -> Option<PathBuf> {
    let macos = program.parent()?;
    let contents = macos.parent()?;
    let bundle = contents.parent()?;
    (macos.file_name()? == "MacOS"
        && contents.file_name()? == "Contents"
        && bundle.extension()? == "app")
        .then(|| bundle.to_path_buf())
}

/// Shows a failure where a person will see it: the helper has no console.
fn notify_failure(message: &str) {
    eprintln!("eov-update-helper: {message}");
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW,
        };
        let wide = |text: &str| {
            text.encode_utf16()
                .chain(std::iter::once(0))
                .collect::<Vec<u16>>()
        };
        let text = wide(message);
        let title = wide("eov update");
        // SAFETY: both strings are NUL-terminated UTF-16 buffers that live
        // across the call; a null owner window is allowed.
        unsafe {
            MessageBoxW(
                0,
                text.as_ptr(),
                title.as_ptr(),
                MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
            );
        }
    }
    #[cfg(target_os = "macos")]
    {
        let escape = |text: &str| text.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(
            "display alert \"eov update\" message \"{}\" as critical",
            escape(message)
        );
        let _ = Command::new("/usr/bin/osascript")
            .args(["-e", &script])
            .status();
    }
}

/// Blocks until the eov process that started the helper has exited.
fn wait_for_parent(wait: WaitFor, pid: u32, timeout: Duration) -> Result<()> {
    match wait {
        WaitFor::Fd(fd) => wait_for_pipe_eof(fd, timeout),
        WaitFor::Handle(handle) => wait_for_handle(handle, timeout),
        WaitFor::Pid => wait_for_pid(pid, timeout),
    }
}

#[cfg(unix)]
fn wait_for_pipe_eof(fd: i32, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    let mut buffer = [0u8; 64];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("timed out waiting for eov to exit");
        }
        let mut poll_fd = libc::pollfd {
            fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let millis = remaining.as_millis().min(i32::MAX as u128) as i32;
        // SAFETY: `poll_fd` is a valid pollfd for the duration of the call.
        let ready = unsafe { libc::poll(&mut poll_fd, 1, millis) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error).context("could not wait for eov to exit");
        }
        if ready == 0 {
            continue;
        }
        // SAFETY: reading into a stack buffer of the given length.
        let read = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if read == 0 || (read < 0 && poll_fd.revents & libc::POLLHUP != 0) {
            // SAFETY: the descriptor was inherited for this purpose only.
            unsafe { libc::close(fd) };
            return Ok(());
        }
        if read < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error).context("could not wait for eov to exit");
            }
        }
    }
}

#[cfg(not(unix))]
fn wait_for_pipe_eof(_fd: i32, _timeout: Duration) -> Result<()> {
    bail!("--wait-fd is only supported on Unix")
}

#[cfg(windows)]
fn wait_for_handle(handle: usize, timeout: Duration) -> Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let millis = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
    // SAFETY: the handle was duplicated for this process by eov and is only
    // used here.
    let result = unsafe { WaitForSingleObject(handle as _, millis) };
    unsafe { CloseHandle(handle as _) };
    if result != WAIT_OBJECT_0 {
        bail!("timed out waiting for eov to exit");
    }
    Ok(())
}

#[cfg(not(windows))]
fn wait_for_handle(_handle: usize, _timeout: Duration) -> Result<()> {
    bail!("--wait-handle is only supported on Windows")
}

fn wait_for_pid(pid: u32, timeout: Duration) -> Result<()> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::SYNCHRONIZE;
        use windows_sys::Win32::System::Threading::OpenProcess;
        // SAFETY: plain Win32 calls; a null handle means the process is gone.
        let handle = unsafe { OpenProcess(SYNCHRONIZE, 0, pid) };
        if handle == 0 {
            return Ok(());
        }
        // `wait_for_handle` closes the handle.
        wait_for_handle(handle as usize, timeout)
    }
    #[cfg(unix)]
    {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            // SAFETY: signal 0 only checks whether the process exists.
            let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            if !alive {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(100));
        }
        bail!("timed out waiting for eov process {pid} to exit")
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (pid, timeout);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transaction::TRANSACTION_SCHEMA;

    fn tree(path: &Path, version: &str) {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("eov.exe"), version).unwrap();
    }

    fn transaction(root: &Path) -> Transaction {
        Transaction {
            schema: TRANSACTION_SCHEMA,
            nonce: "nonce".into(),
            pid: std::process::id(),
            mode: Mode::ReplaceTree,
            from_version: Some("1.0.0".into()),
            to_version: Some("1.0.1".into()),
            current_tree: Some(root.join("eov")),
            staged_tree: Some(root.join(".eov-update-stage")),
            previous_tree: Some(root.join(".eov-previous-nonce")),
            stale_previous: None,
            cargo_version: None,
            cargo_root: None,
            relaunch: None,
            plugins: Vec::new(),
            registry: None,
            plugin_lock: None,
            report: root.join("report.toml"),
            rollback_record: None,
            rollback_path: None,
            cleanup: Vec::new(),
        }
    }

    fn report_for(transaction: &Transaction) -> HelperReport {
        HelperReport {
            schema: REPORT_SCHEMA,
            nonce: transaction.nonce.clone(),
            mode: transaction.mode,
            outcome: Outcome::RolledBack,
            from_version: None,
            to_version: None,
            finished_at: 0,
            message: String::new(),
            installation: None,
            previous: None,
        }
    }

    #[test]
    fn parses_arguments() {
        let args = |list: &[&str]| list.iter().map(|arg| arg.into()).collect::<Vec<_>>();
        assert_eq!(
            parse_args(args(&["helper", "--self-test"])).unwrap(),
            Invocation::SelfTest
        );
        assert_eq!(
            parse_args(args(&[
                "helper",
                "apply",
                "--transaction",
                "/t",
                "--wait-fd",
                "7"
            ]))
            .unwrap(),
            Invocation::Apply {
                transaction: "/t".into(),
                wait: WaitFor::Fd(7)
            }
        );
        assert!(parse_args(args(&["helper", "apply", "--bogus", "1"])).is_err());
        assert!(parse_args(args(&["helper", "download"])).is_err());
    }

    #[test]
    fn replace_tree_keeps_the_previous_version_and_marks_both_trees() {
        let temp = tempfile::tempdir().unwrap();
        let transaction = transaction(temp.path());
        tree(transaction.current_tree.as_ref().unwrap(), "old");
        tree(transaction.staged_tree.as_ref().unwrap(), "new");
        let mut report = report_for(&transaction);
        assert_eq!(
            apply_tree(&transaction, &mut report).unwrap(),
            Outcome::Applied
        );
        let current = transaction.current_tree.as_ref().unwrap();
        let previous = transaction.previous_tree.as_ref().unwrap();
        assert_eq!(fs::read_to_string(current.join("eov.exe")).unwrap(), "new");
        assert_eq!(fs::read_to_string(previous.join("eov.exe")).unwrap(), "old");
        assert!(current.join(TREE_MARKER).is_file() && previous.join(TREE_MARKER).is_file());
    }

    #[test]
    fn a_failed_plugin_step_restores_the_previous_tree() {
        let temp = tempfile::tempdir().unwrap();
        let mut transaction = transaction(temp.path());
        tree(transaction.current_tree.as_ref().unwrap(), "old");
        tree(transaction.staged_tree.as_ref().unwrap(), "new");
        transaction.plugins.push(PluginReplacement {
            staged: temp.path().join("missing.eop"),
            target: temp.path().join("plugins").join("x.eop"),
            backup: temp.path().join("plugins").join(".x.eop.backup"),
            remove_after: Vec::new(),
            keep_as: None,
        });
        let mut report = report_for(&transaction);
        assert!(apply_tree(&transaction, &mut report).is_err());
        let current = transaction.current_tree.as_ref().unwrap();
        assert_eq!(fs::read_to_string(current.join("eov.exe")).unwrap(), "old");
        assert!(!transaction.previous_tree.as_ref().unwrap().exists());
        assert_ne!(report.outcome, Outcome::Failed);
    }

    #[test]
    fn only_marked_rollback_copies_are_deleted() {
        let temp = tempfile::tempdir().unwrap();
        let unmarked = temp.path().join("Downloads");
        tree(&unmarked, "x");
        remove_installation_tree(&unmarked);
        assert!(unmarked.exists(), "an unproven directory was deleted");
        let marked = temp.path().join(".eov-previous-old");
        tree(&marked, "x");
        fs::write(marked.join(TREE_MARKER), "0.9.0").unwrap();
        remove_installation_tree(&marked);
        assert!(!marked.exists());
    }

    #[test]
    fn plugin_replacement_and_registry_are_applied_together() {
        let temp = tempfile::tempdir().unwrap();
        let plugins = temp.path().join("plugins");
        fs::create_dir_all(&plugins).unwrap();
        fs::write(plugins.join("x.eop"), b"old").unwrap();
        fs::write(plugins.join(".stage.part"), b"new").unwrap();
        fs::write(plugins.join("x-0.9.eop"), b"older").unwrap();
        fs::write(plugins.join(".eov-managed.toml"), b"old registry").unwrap();
        fs::write(plugins.join(".registry.tmp"), b"new registry").unwrap();
        let mut transaction = transaction(temp.path());
        transaction.mode = Mode::CargoInstall;
        transaction.plugins.push(PluginReplacement {
            staged: plugins.join(".stage.part"),
            target: plugins.join("x.eop"),
            backup: plugins.join(".x.eop.backup"),
            remove_after: vec![plugins.join("x-0.9.eop")],
            keep_as: Some(temp.path().join("store").join("x.eop")),
        });
        transaction.registry = Some(RegistryReplacement {
            staged: plugins.join(".registry.tmp"),
            target: plugins.join(".eov-managed.toml"),
            backup: plugins.join(".eov-managed.backup"),
        });
        apply_plugins_and_registry(&transaction).unwrap();
        assert_eq!(fs::read(plugins.join("x.eop")).unwrap(), b"new");
        assert!(!plugins.join("x-0.9.eop").exists());
        assert_eq!(
            fs::read(plugins.join(".eov-managed.toml")).unwrap(),
            b"new registry"
        );
        let mut names: Vec<_> = fs::read_dir(&plugins)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, vec![".eov-managed.toml", "x.eop"]);
        // The replaced package was kept for a rollback.
        assert_eq!(
            fs::read(temp.path().join("store").join("x.eop")).unwrap(),
            b"old"
        );
    }

    #[cfg(unix)]
    #[test]
    fn waits_for_the_write_end_of_the_pipe_to_close() {
        let mut fds = [0; 2];
        // SAFETY: creating a pipe into a local array.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let writer = fds[1];
        let closer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            // SAFETY: closing our own descriptor.
            unsafe { libc::close(writer) };
        });
        let start = Instant::now();
        wait_for_pipe_eof(fds[0], Duration::from_secs(10)).unwrap();
        assert!(start.elapsed() >= Duration::from_millis(90));
        closer.join().unwrap();
    }

    #[test]
    fn report_round_trips() {
        let temp = tempfile::tempdir().unwrap();
        let transaction = transaction(temp.path());
        let mut report = report_for(&transaction);
        report.outcome = Outcome::Applied;
        write_report(&transaction.report, &report).unwrap();
        let parsed: HelperReport =
            toml::from_str(&fs::read_to_string(&transaction.report).unwrap()).unwrap();
        assert_eq!(parsed, report);
    }
}
