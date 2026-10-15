//! Starting `eov-update-helper` for updates that must happen after eov
//! exits (Windows portable folders, manually installed macOS bundles, and
//! `cargo install` on Windows).
//!
//! The helper is copied out of the installation it replaces into a fresh
//! private temporary directory with a random name (never a predictable path
//! another user could pre-create), smoke-tested there (`--self-test`, which
//! proves for example that dyld can load it on macOS) and started:
//!
//! - with its working directory in that temporary directory, so it never
//!   holds the installation folder open (a Windows sharing violation);
//! - without a console window on Windows (`CREATE_NO_WINDOW`);
//! - with a way to learn that eov exited that cannot be fooled by PID reuse:
//!   the read end of a pipe whose write end only eov holds (Unix), or an
//!   inherited handle to the eov process (Windows).

use super::transaction::Transaction;
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

/// Prefix of the helper's temporary directories.
pub const HELPER_DIR_PREFIX: &str = "eov-update-";
const TRANSACTION_FILE: &str = "transaction.toml";

/// A helper copied into its private temporary directory.
#[derive(Debug)]
pub struct PreparedHelper {
    pub dir: PathBuf,
    pub executable: PathBuf,
}

/// Copies `helper` (from the current installation) into a new private
/// temporary directory and checks that it starts there.
pub fn prepare(helper: &Path) -> Result<PreparedHelper> {
    if !helper.is_file() {
        bail!(
            "this installation has no update helper ({} is missing); reinstall eov from its release archive",
            helper.display()
        );
    }
    let mut builder = tempfile::Builder::new();
    builder.prefix(HELPER_DIR_PREFIX);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(fs::Permissions::from_mode(0o700));
    }
    let dir = builder
        .tempdir()
        .context("could not create a temporary directory for the update helper")?;
    let executable = dir.path().join(
        helper
            .file_name()
            .context("the update helper path has no file name")?,
    );
    fs::copy(helper, &executable).with_context(|| {
        format!(
            "could not copy the update helper out of the installation ({})",
            helper.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))?;
    }
    let mut command = Command::new(&executable);
    command
        .arg("--self-test")
        .current_dir(dir.path())
        .stdin(Stdio::null());
    hide_console(&mut command);
    let output = command
        .output()
        .with_context(|| format!("the update helper cannot run from {}", dir.path().display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || !stdout.starts_with("eov-update-helper") {
        bail!(
            "the update helper does not run from {} ({}): {}{}",
            dir.path().display(),
            output.status,
            stdout.trim(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(PreparedHelper {
        dir: dir.keep(),
        executable,
    })
}

/// Writes `transaction` next to the prepared helper and starts it. The
/// helper waits for this process to exit before it changes anything.
pub fn launch(helper: &PreparedHelper, transaction: &Transaction) -> Result<()> {
    transaction.validate().map_err(anyhow::Error::msg)?;
    let transaction_path = helper.dir.join(TRANSACTION_FILE);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&transaction_path)
        .with_context(|| format!("could not write {}", transaction_path.display()))?;
    std::io::Write::write_all(&mut file, toml::to_string_pretty(transaction)?.as_bytes())?;
    file.sync_all()?;
    drop(file);

    let mut command = Command::new(&helper.executable);
    command
        .arg("apply")
        .arg("--transaction")
        .arg(&transaction_path)
        .current_dir(&helper.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_console(&mut command);
    spawn_waiting_for_exit(command).context("could not start the eov update helper")
}

#[cfg(windows)]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_command: &mut Command) {}

#[cfg(unix)]
fn spawn_waiting_for_exit(mut command: Command) -> Result<()> {
    use std::os::unix::process::CommandExt;
    use std::sync::OnceLock;
    // The write end stays open (close-on-exec, so no other child inherits
    // it) until this process exits; the helper then reads end-of-file.
    static WAIT_PIPE_WRITER: OnceLock<i32> = OnceLock::new();

    let mut fds = [0; 2];
    // SAFETY: creating a pipe into a local array.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("could not create a pipe");
    }
    let [read, write] = fds;
    for fd in fds {
        // SAFETY: setting a descriptor flag on descriptors we own.
        unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    command.arg("--wait-fd").arg(read.to_string());
    // SAFETY: only async-signal-safe fcntl runs between fork and exec; it
    // makes the read end (and only it) survive exec in the helper.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(read, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let spawned = command.spawn();
    // SAFETY: the parent no longer needs the read end.
    unsafe { libc::close(read) };
    match spawned {
        Ok(_) => {
            // Never closed: it must stay open until this process exits. If a
            // helper was already started, this descriptor stays open too.
            let _ = WAIT_PIPE_WRITER.set(write);
            Ok(())
        }
        Err(error) => {
            // SAFETY: closing our own descriptor after a failed spawn.
            unsafe { libc::close(write) };
            Err(error.into())
        }
    }
}

#[cfg(windows)]
fn spawn_waiting_for_exit(mut command: Command) -> Result<()> {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut handle: HANDLE = 0;
    // SAFETY: duplicating the pseudo handle of the current process into an
    // inheritable real handle; Rust's process spawning inherits it.
    let ok = unsafe {
        let process = GetCurrentProcess();
        DuplicateHandle(
            process,
            process,
            process,
            &mut handle,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error())
            .context("could not create a process handle for the update helper");
    }
    command
        .arg("--wait-handle")
        .arg((handle as usize).to_string());
    command.spawn()?;
    // The handle stays open until this process exits.
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn spawn_waiting_for_exit(mut command: Command) -> Result<()> {
    command.spawn()?;
    Ok(())
}

/// Removes helper directories left by earlier updates (the helper cannot
/// delete its own executable on Windows).
pub fn cleanup_stale(temp_root: &Path, older_than: Duration) {
    let Ok(entries) = fs::read_dir(temp_root) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(HELPER_DIR_PREFIX) {
            continue;
        }
        let path = entry.path();
        let old = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= older_than);
        // Only directories that look like ours: a helper and nothing else
        // unexpected.
        let ours = fs::read_dir(&path).is_ok_and(|children| {
            children.flatten().all(|child| {
                let name = child.file_name().to_string_lossy().into_owned();
                name.starts_with("eov-update-helper") || name.starts_with("transaction")
            })
        });
        if old && ours && path.is_dir() {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_helper_is_reported_before_anything_happens() {
        let temp = tempfile::tempdir().unwrap();
        let error = prepare(&temp.path().join("eov-update-helper")).unwrap_err();
        assert!(error.to_string().contains("no update helper"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn a_helper_that_cannot_run_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let helper = temp.path().join("eov-update-helper");
        fs::write(&helper, "#!/bin/sh\necho broken; exit 3\n").unwrap();
        let error = prepare(&helper).unwrap_err();
        assert!(error.to_string().contains("does not run"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn a_working_helper_is_copied_into_a_private_directory() {
        let temp = tempfile::tempdir().unwrap();
        let helper = temp.path().join("eov-update-helper");
        fs::write(&helper, "#!/bin/sh\necho eov-update-helper 1.0.0\n").unwrap();
        let prepared = prepare(&helper).unwrap();
        assert!(prepared.executable.starts_with(&prepared.dir));
        assert!(
            prepared
                .dir
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(HELPER_DIR_PREFIX)
        );
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&prepared.dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "the helper directory must be private");
        fs::remove_dir_all(&prepared.dir).unwrap();
    }

    #[test]
    fn stale_cleanup_only_touches_helper_directories() {
        let temp = tempfile::tempdir().unwrap();
        let ours = temp.path().join("eov-update-abc");
        fs::create_dir(&ours).unwrap();
        fs::write(ours.join("eov-update-helper"), b"x").unwrap();
        let foreign = temp.path().join("eov-update-user-data");
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join("notes.txt"), b"keep").unwrap();
        cleanup_stale(temp.path(), Duration::ZERO);
        assert!(!ours.exists());
        assert!(foreign.exists());
    }
}
