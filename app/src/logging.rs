//! Logging and crash reports (SH-16).
//!
//! Logs go to stderr, so command output (`eov plugin info -o toml > file`)
//! stays clean, at the level chosen with `--log-level` or `RUST_LOG`
//! (default: warnings). GUI sessions also write `eov.log` in the log folder
//! at info level and keep the previous sessions' logs (`eov.1.log` …).
//! Messages at info level name files, not full paths, which can carry
//! patient identifiers; full paths are logged at debug level.
//!
//! A panic hook writes `crash-<time>.txt` (message, location, backtrace)
//! to the log folder; the next start tells the user about it.

use crate::cli::CliLogLevel;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Session logs kept besides the current one.
const KEPT_LOGS: usize = 5;

/// The folder with logs and crash reports.
pub(crate) fn log_dir() -> PathBuf {
    crate::paths::log_dir()
}

fn stderr_filter(log_level: Option<CliLogLevel>) -> EnvFilter {
    let filter = match log_level {
        Some(level) => EnvFilter::new(level.as_filter()),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
    };
    quiet_noisy_crates(filter)
}

fn quiet_noisy_crates(filter: EnvFilter) -> EnvFilter {
    filter.add_directive(
        "sctk_adwaita::buttons=error"
            .parse()
            .expect("valid sctk_adwaita buttons log directive"),
    )
}

/// Renames `eov.log` to `eov.1.log` and so on, dropping the oldest.
fn rotate(dir: &Path) {
    let name = |index: usize| {
        if index == 0 {
            dir.join("eov.log")
        } else {
            dir.join(format!("eov.{index}.log"))
        }
    };
    let _ = fs::remove_file(name(KEPT_LOGS));
    for index in (0..KEPT_LOGS).rev() {
        let _ = fs::rename(name(index), name(index + 1));
    }
}

/// Opens this session's log file, rotating older ones.
fn session_log() -> Option<File> {
    let dir = log_dir();
    fs::create_dir_all(&dir).ok()?;
    rotate(&dir);
    File::create(dir.join("eov.log")).ok()
}

/// Sets up logging; `gui` adds the session log file.
pub(crate) fn init(log_level: Option<CliLogLevel>, gui: bool) {
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(stderr_filter(log_level));
    let file = gui.then(session_log).flatten().map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(Mutex::new(file))
            .with_filter(quiet_noisy_crates(EnvFilter::new("info")))
    });
    let _ = tracing_subscriber::registry()
        .with(stderr)
        .with(file)
        .try_init();
}

/// Writes a crash report when the process panics, then runs the default
/// hook (which prints the panic).
pub(crate) fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let thread = std::thread::current();
        let report = format!(
            "eov {} crashed\nthread: {}\n{info}\n\nbacktrace:\n{backtrace}\n",
            crate::version::BUILD_VERSION,
            thread.name().unwrap_or("<unnamed>"),
        );
        tracing::error!("{info}");
        let dir = log_dir();
        if fs::create_dir_all(&dir).is_ok() {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default();
            if let Ok(mut file) = File::create(dir.join(format!("crash-{stamp}.txt"))) {
                let _ = file.write_all(report.as_bytes());
            }
        }
        default(info);
    }));
}

/// The newest crash report not shown yet; it is marked as shown.
pub(crate) fn take_crash_report() -> Option<PathBuf> {
    let dir = log_dir();
    let newest = fs::read_dir(&dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("crash-") && name.ends_with(".txt"))
        })
        .max()?;
    let marker = dir.join("last-crash-shown");
    let shown = fs::read_to_string(&marker).unwrap_or_default();
    let name = newest.file_name()?.to_string_lossy().into_owned();
    if shown.trim() == name {
        return None;
    }
    let _ = fs::write(&marker, &name);
    Some(newest)
}

/// Opens the log folder in the file manager (on a worker; a failure is
/// reported in the window).
pub(crate) fn open_log_folder() -> Result<(), String> {
    let dir = log_dir();
    fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    crate::desktop::open_folder(dir, "Could not open the log folder");
    Ok(())
}

/// A file's name for info-level messages (full paths only at debug level).
pub(crate) fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<slide>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_rotate_and_keep_a_bounded_history() {
        let dir = tempfile::tempdir().unwrap();
        for session in 0..(KEPT_LOGS + 3) {
            rotate(dir.path());
            fs::write(dir.path().join("eov.log"), format!("{session}")).unwrap();
        }
        let count = fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(count, KEPT_LOGS + 1);
        assert_eq!(
            fs::read_to_string(dir.path().join("eov.log")).unwrap(),
            format!("{}", KEPT_LOGS + 2)
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("eov.1.log")).unwrap(),
            format!("{}", KEPT_LOGS + 1)
        );
    }

    #[test]
    fn info_messages_name_files_only() {
        assert_eq!(file_label(Path::new("/cases/2024-12345/HE.svs")), "HE.svs");
    }
}
