//! Hand-offs to the desktop: showing a file in the file manager, opening a
//! folder, and the environment of the programs eov starts for that (SH-15).
//!
//! - **Reveal.** "Open Containing Folder" opens the folder with the file
//!   selected: `explorer /select,` on Windows, `open -R` on macOS, and the
//!   `org.freedesktop.FileManager1.ShowItems` D-Bus call (Nautilus, Dolphin,
//!   Nemo, Caja, Thunar, …) on Linux, which falls back to opening the folder
//!   with `xdg-open`. Both can block (D-Bus activation, a slow mount), so
//!   [`reveal`] and [`open_folder`] run on a worker and report failures in
//!   the window.
//! - **Environment.** Programs eov starts get the environment eov was
//!   started with, not the one it runs with:
//!   - the AppImage runtime's `APPIMAGE`, `APPDIR`, `ARGV0` and `OWD`
//!     describe eov's image; AppImage-aware programs (Electron's updater, for
//!     example) would take them as their own;
//!   - variables that steer library loading and data lookup are reset to
//!     their values at launch, in case a plugin or a library changed them
//!     in-process;
//!   - the startup-notification tokens belong to eov's own launch.
//!
//!   `packaging/appimage/AppRun` changes nothing today (bundled libraries
//!   are found through RUNPATH, not `LD_LIBRARY_PATH`). A variable it ever
//!   has to set must first be saved with its `save_env`, which records the
//!   launch value as `EOV_ORIG_<NAME>` and the name in `EOV_SAVED_ENV`;
//!   children get the saved value back (or the variable removed, when it
//!   was not set).

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// Variables that change how programs load libraries or find their data.
const LAUNCH_VARIABLES: &[&str] = &[
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "PYTHONHOME",
    "PYTHONPATH",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_MODULE_DIR",
    "GSETTINGS_SCHEMA_DIR",
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_DATA_PREFIX",
    "QT_PLUGIN_PATH",
    "XDG_DATA_DIRS",
];

/// Set by the AppImage runtime for eov's own image.
const APPIMAGE_VARIABLES: &[&str] = &["APPIMAGE", "APPDIR", "ARGV0", "OWD"];

/// Startup-notification tokens of eov's own launch.
const LAUNCH_TOKENS: &[&str] = &["XDG_ACTIVATION_TOKEN", "DESKTOP_STARTUP_ID"];

/// Names of the variables AppRun saved (space-separated).
const SAVED_LIST: &str = "EOV_SAVED_ENV";
/// Prefix of a saved launch value.
const SAVED_PREFIX: &str = "EOV_ORIG_";

static LAUNCH: OnceLock<Vec<(String, Option<OsString>)>> = OnceLock::new();

/// Records the launch values of the library-loading variables. Call first
/// thing in `main`, before plugins or libraries can change them.
pub(crate) fn remember_launch_environment() {
    LAUNCH.get_or_init(|| snapshot(|name| std::env::var_os(name)));
}

fn snapshot(lookup: impl Fn(&str) -> Option<OsString>) -> Vec<(String, Option<OsString>)> {
    LAUNCH_VARIABLES
        .iter()
        .map(|name| (name.to_string(), lookup(name)))
        .collect()
}

/// A variable name AppRun may have saved (no `=`, no NUL, not empty).
fn is_variable_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// The changes a child's environment needs: `Some(value)` sets, `None`
/// removes. `current` looks up eov's environment now.
fn child_changes(
    current: impl Fn(&str) -> Option<OsString>,
    launch: &[(String, Option<OsString>)],
) -> BTreeMap<String, Option<OsString>> {
    let mut changes = BTreeMap::new();
    for (name, value) in launch {
        if current(name) != *value {
            changes.insert(name.clone(), value.clone());
        }
    }
    if let Some(list) = current(SAVED_LIST) {
        for name in list.to_string_lossy().split_whitespace() {
            if !is_variable_name(name) {
                continue;
            }
            let saved = format!("{SAVED_PREFIX}{name}");
            changes.insert(name.to_string(), current(&saved));
            changes.insert(saved, None);
        }
        changes.insert(SAVED_LIST.to_string(), None);
    }
    if current("APPIMAGE").is_some() || current("APPDIR").is_some() {
        for name in APPIMAGE_VARIABLES {
            changes.insert(name.to_string(), None);
        }
    }
    for name in LAUNCH_TOKENS {
        if current(name).is_some() {
            changes.insert(name.to_string(), None);
        }
    }
    changes
}

/// Gives `command` the environment eov was started with (see the module
/// documentation).
pub(crate) fn apply_launch_environment(command: &mut Command) {
    let launch = LAUNCH.get().map(Vec::as_slice).unwrap_or_default();
    for (name, value) in child_changes(|name| std::env::var_os(name), launch) {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
}

/// A command for a desktop program, with eov's launch environment and no
/// standard input or output.
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
pub(crate) fn command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    apply_launch_environment(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

/// Opens a folder (or any file) with the desktop's default program.
/// Blocking.
pub(crate) fn open_blocking(target: &Path) -> Result<(), String> {
    let mut last_error = None;
    for mut launcher in open::commands(target) {
        apply_launch_environment(&mut launcher);
        launcher
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let program = launcher.get_program().to_string_lossy().into_owned();
        match launcher.status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => last_error = Some(format!("{program} failed ({status})")),
            Err(error) => last_error = Some(format!("{program}: {error}")),
        }
    }
    Err(last_error.unwrap_or_else(|| "no program to open folders was found".to_string()))
}

/// Shows `path` selected in the file manager. Blocking.
pub(crate) fn reveal_blocking(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("{} no longer exists", path.display()));
    }
    let path = crate::paths::simplified(path);
    reveal_platform(&path)
}

#[cfg(target_os = "windows")]
fn reveal_platform(path: &Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    // Explorer parses its command line itself: the path must follow
    // `/select,` quoted as one piece. Windows paths cannot contain `"`.
    let explorer = std::env::var_os("SystemRoot")
        .map(|root| PathBuf::from(root).join("explorer.exe"))
        .unwrap_or_else(|| PathBuf::from("explorer.exe"));
    let mut explorer = command(explorer);
    explorer.raw_arg(format!("/select,\"{}\"", path.display()));
    // Explorer's exit status means nothing (it is 1 on success).
    explorer
        .spawn()
        .map(drop)
        .map_err(|error| format!("explorer: {error}"))
}

#[cfg(target_os = "macos")]
fn reveal_platform(path: &Path) -> Result<(), String> {
    let status = command("/usr/bin/open")
        .arg("-R")
        .arg(path)
        .status()
        .map_err(|error| format!("open -R: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("open -R failed ({status})"))
    }
}

#[cfg(target_os = "linux")]
fn reveal_platform(path: &Path) -> Result<(), String> {
    let uri = url::Url::from_file_path(path)
        .map_err(|()| format!("{} is not an absolute path", path.display()))?;
    match show_items(uri.as_str()) {
        Ok(()) => Ok(()),
        Err(dbus_error) => {
            tracing::debug!("FileManager1.ShowItems failed ({dbus_error}); opening the folder");
            open_parent(path).map_err(|open_error| format!("{dbus_error}; {open_error}"))
        }
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn reveal_platform(path: &Path) -> Result<(), String> {
    open_parent(path)
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn open_parent(path: &Path) -> Result<(), String> {
    let folder = path
        .parent()
        .ok_or_else(|| format!("{} has no parent folder", path.display()))?;
    open_blocking(folder)
}

/// `org.freedesktop.FileManager1.ShowItems`: the file manager opens the
/// folder and selects the file (the Flatpak may call it, see its
/// manifest).
#[cfg(target_os = "linux")]
fn show_items(uri: &str) -> Result<(), String> {
    let connection = zbus::blocking::Connection::session().map_err(|error| error.to_string())?;
    connection
        .call_method(
            Some("org.freedesktop.FileManager1"),
            "/org/freedesktop/FileManager1",
            Some("org.freedesktop.FileManager1"),
            "ShowItems",
            &(&[uri][..], ""),
        )
        .map(drop)
        .map_err(|error| error.to_string())
}

/// Tells the user, in the window, that a hand-off failed (UI thread or
/// not).
fn report_failure(title: String, error: String) {
    let _ = slint::invoke_from_event_loop(move || {
        crate::host_ui::with_host(|host| {
            host.notify(
                eov_plugin_api::protocol::NotifyLevel::Error,
                &title,
                Some(&error),
                None,
            )
        });
    });
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Shows `path` selected in the file manager, on a worker; a failure is
/// reported in the window.
pub(crate) fn reveal(path: PathBuf) {
    let spawned = std::thread::Builder::new()
        .name("eov-reveal".to_string())
        .spawn(move || {
            if let Err(error) = reveal_blocking(&path) {
                tracing::warn!(
                    "Could not reveal {}: {error}",
                    crate::logging::file_label(&path)
                );
                report_failure(
                    format!("Could not show {} in the file manager", display_name(&path)),
                    error,
                );
            }
        });
    if let Err(error) = spawned {
        report_failure(
            "Could not open the file manager".to_string(),
            error.to_string(),
        );
    }
}

/// Opens `folder` in the file manager, on a worker; a failure is reported
/// in the window under `title`.
pub(crate) fn open_folder(folder: PathBuf, title: &str) {
    let title = title.to_string();
    let spawned = std::thread::Builder::new()
        .name("eov-open-folder".to_string())
        .spawn(move || {
            if let Err(error) = open_blocking(&folder) {
                tracing::warn!("Could not open {}: {error}", folder.display());
                report_failure(title, error);
            }
        });
    if let Err(error) = spawned {
        report_failure(
            "Could not open the file manager".to_string(),
            error.to_string(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, OsString> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), OsString::from(value)))
            .collect()
    }

    #[test]
    fn appimage_variables_and_launch_tokens_are_not_inherited() {
        let current = env(&[
            ("APPIMAGE", "/home/u/eov.AppImage"),
            ("APPDIR", "/tmp/.mount_eovXYZ"),
            ("ARGV0", "./eov"),
            ("OWD", "/home/u"),
            ("XDG_ACTIVATION_TOKEN", "token"),
            ("HOME", "/home/u"),
        ]);
        let changes = child_changes(|name| current.get(name).cloned(), &[]);
        for name in ["APPIMAGE", "APPDIR", "ARGV0", "OWD", "XDG_ACTIVATION_TOKEN"] {
            assert_eq!(changes.get(name), Some(&None), "{name} must be removed");
        }
        assert!(!changes.contains_key("HOME"));
        assert!(!changes.contains_key("DESKTOP_STARTUP_ID"));
    }

    #[test]
    fn library_paths_go_back_to_their_launch_values() {
        let launch = snapshot(|name| (name == "PATH").then(|| OsString::from("/usr/bin")));
        // A plugin prepended its own directories in-process.
        let current = env(&[
            ("PATH", "/opt/cuda/bin:/usr/bin"),
            ("LD_LIBRARY_PATH", "/opt/cuda/lib64"),
        ]);
        let changes = child_changes(|name| current.get(name).cloned(), &launch);
        assert_eq!(changes.get("PATH"), Some(&Some(OsString::from("/usr/bin"))));
        assert_eq!(changes.get("LD_LIBRARY_PATH"), Some(&None));
        assert!(!changes.contains_key("XDG_DATA_DIRS"));
    }

    #[test]
    fn values_saved_by_apprun_are_restored() {
        let current = env(&[
            ("EOV_SAVED_ENV", "XDG_DATA_DIRS GTK_PATH bad=name"),
            ("EOV_ORIG_XDG_DATA_DIRS", "/usr/share"),
            ("XDG_DATA_DIRS", "/tmp/.mount_eov/usr/share:/usr/share"),
            ("GTK_PATH", "/tmp/.mount_eov/usr/lib/gtk"),
        ]);
        let changes = child_changes(|name| current.get(name).cloned(), &[]);
        assert_eq!(
            changes.get("XDG_DATA_DIRS"),
            Some(&Some(OsString::from("/usr/share")))
        );
        // Not set before AppRun: removed.
        assert_eq!(changes.get("GTK_PATH"), Some(&None));
        assert_eq!(changes.get("EOV_ORIG_XDG_DATA_DIRS"), Some(&None));
        assert_eq!(changes.get("EOV_SAVED_ENV"), Some(&None));
        assert!(!changes.keys().any(|name| name.contains('=')));
    }

    #[test]
    fn missing_files_are_reported_before_anything_starts() {
        let dir = tempfile::tempdir().unwrap();
        let error = reveal_blocking(&dir.path().join("gone.svs")).unwrap_err();
        assert!(error.contains("no longer exists"), "{error}");
    }
}
