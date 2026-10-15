//! App-side access to on-disk locations.
//!
//! Thin wrappers over [`eov_common::paths`]. In test builds every accessor
//! first isolates the process into a temporary root, so no unit test can read
//! or modify the developer's real configuration, caches or plugin data.

// Several accessors are consumed by the session, recent-files and plugin
// data work that follows; keep the full set together.
#![allow(dead_code)]

use serde::Serialize;
use std::path::{Path, PathBuf};

#[inline]
fn ensure_isolation() {
    #[cfg(test)]
    {
        eov_common::paths::isolate_for_tests();
    }
}

/// Isolates the test process explicitly (for tests that drive plugin code
/// which resolves its own paths through `eov_common::paths`).
#[cfg(test)]
pub(crate) fn isolate_for_tests() -> &'static Path {
    eov_common::paths::isolate_for_tests()
}

pub(crate) fn home_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::home_dir()
}

pub(crate) fn config_file() -> PathBuf {
    ensure_isolation();
    eov_common::paths::config_file()
}

pub(crate) fn recent_files_file() -> PathBuf {
    ensure_isolation();
    eov_common::paths::recent_files_file()
}

pub(crate) fn session_file() -> PathBuf {
    ensure_isolation();
    eov_common::paths::session_file()
}

pub(crate) fn plugin_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::plugin_dir()
}

pub(crate) fn updates_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::updates_dir()
}

pub(crate) fn data_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::data_dir()
}

pub(crate) fn cache_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::cache_dir()
}

pub(crate) fn log_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::log_dir()
}

pub(crate) fn plugin_package_cache_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::plugin_package_cache_dir()
}

pub(crate) fn thumbnail_cache_dir() -> PathBuf {
    ensure_isolation();
    eov_common::paths::thumbnail_cache_dir()
}

pub(crate) fn plugin_data_dir(plugin_id: &str) -> PathBuf {
    ensure_isolation();
    eov_common::paths::plugin_data_dir(plugin_id)
}

pub(crate) fn plugin_cache_dir(plugin_id: &str) -> PathBuf {
    ensure_isolation();
    eov_common::paths::plugin_cache_dir(plugin_id)
}

pub(crate) fn legacy_config_dir() -> Option<PathBuf> {
    ensure_isolation();
    eov_common::paths::legacy_config_dir()
}

/// A folder private to the user where a running eov is found by the next
/// one (single instance, SH-8): `$EOV_RUNTIME_DIR`; else
/// `$XDG_RUNTIME_DIR/eov` on Linux (`$XDG_RUNTIME_DIR/app/<id>` in the
/// Flatpak, the only part shared between its instances), the per-user
/// `$TMPDIR/eov` on macOS, and `<data>/run` elsewhere or without those.
pub(crate) fn runtime_dir() -> PathBuf {
    ensure_isolation();
    if let Some(root) = eov_common::paths::root_override() {
        return root.join("run");
    }
    let absolute_var = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    if let Some(dir) = absolute_var("EOV_RUNTIME_DIR") {
        return dir;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    if let Some(runtime) = absolute_var("XDG_RUNTIME_DIR") {
        return match std::env::var_os("FLATPAK_ID").filter(|id| !id.is_empty()) {
            Some(id) => runtime.join("app").join(id),
            None => runtime.join("eov"),
        };
    }
    #[cfg(target_os = "macos")]
    if let Some(temp) = absolute_var("TMPDIR") {
        return temp.join("eov");
    }
    data_dir().join("run")
}

/// Named views of slides (`<data>/bookmarks.toml`).
pub(crate) fn bookmarks_file() -> PathBuf {
    data_dir().join("bookmarks.toml")
}

/// Manual pixel sizes of slides (`<data>/calibrations.toml`).
pub(crate) fn calibrations_file() -> PathBuf {
    data_dir().join("calibrations.toml")
}

/// What is kept at a [`StorageLocation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum StorageKind {
    File,
    Folder,
}

/// A place eov keeps data, as `eov paths` lists it (SH-17).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StorageLocation {
    /// A stable name for scripts.
    pub(crate) id: String,
    pub(crate) description: String,
    pub(crate) path: PathBuf,
    pub(crate) kind: StorageKind,
}

/// Every place eov and its plugins keep data. `config_file` and
/// `plugin_dir` are the ones in use (after `--config` and `--plugin-dir`).
pub(crate) fn storage_locations(
    config_file: &Path,
    plugin_dir: &Path,
    policy_file: Option<&Path>,
    managed_plugin_dirs: &[PathBuf],
) -> Vec<StorageLocation> {
    use StorageKind::{File, Folder};
    let mut locations = Vec::new();
    let mut add = |id: &str, description: &str, path: PathBuf, kind: StorageKind| {
        locations.push(StorageLocation {
            id: id.to_string(),
            description: description.to_string(),
            path,
            kind,
        });
    };
    add("config", "Settings", config_file.to_path_buf(), File);
    add("home", "eov's home folder", home_dir(), Folder);
    add(
        "recent-files",
        "Recently opened slides and folders",
        recent_files_file(),
        File,
    );
    add(
        "session",
        "Slides, panes and views of the last session",
        session_file(),
        File,
    );
    add(
        "plugins",
        "Installed plugin packages and their registry",
        plugin_dir.to_path_buf(),
        Folder,
    );
    for (index, dir) in managed_plugin_dirs.iter().enumerate() {
        add(
            &format!("managed-plugins-{index}"),
            "Plugin packages from the administrator (read-only)",
            dir.clone(),
            Folder,
        );
    }
    add(
        "updates",
        "Downloaded updates and what `eov update --rollback` restores",
        updates_dir(),
        Folder,
    );
    add("data", "Application data", data_dir(), Folder);
    add("bookmarks", "Bookmarked views", bookmarks_file(), File);
    add(
        "calibrations",
        "Pixel sizes entered for slides",
        calibrations_file(),
        File,
    );
    add("logs", "Logs and crash reports", log_dir(), Folder);
    let plugin_data = data_dir().join("plugins");
    add(
        "plugin-data",
        "Plugin data (annotation databases, settings)",
        plugin_data.clone(),
        Folder,
    );
    add("cache", "Caches (safe to delete)", cache_dir(), Folder);
    add(
        "plugin-packages-cache",
        "Extracted plugin packages (pruned by eov, or `eov plugin gc`)",
        plugin_package_cache_dir(),
        Folder,
    );
    add(
        "thumbnail-cache",
        "Series-bar thumbnails (not pruned; safe to delete)",
        thumbnail_cache_dir(),
        Folder,
    );
    add(
        "plugin-caches",
        "Plugin caches",
        cache_dir().join("plugins"),
        Folder,
    );
    add(
        "instance",
        "Where the running eov is found by the next one",
        runtime_dir(),
        Folder,
    );
    add(
        "script-endpoint",
        "How scripts reach the running eov: port, process id and secret (only you can read it)",
        crate::instance::Endpoint::for_setup(config_file, plugin_dir).info_file(),
        File,
    );
    if let Some(policy_file) = policy_file {
        add(
            "policy",
            "The administrator's policy (read-only)",
            policy_file.to_path_buf(),
            File,
        );
    }
    if let Some(legacy) = legacy_config_dir() {
        add(
            "legacy-config",
            "Files of eov < 1.0, read once to migrate them",
            legacy,
            Folder,
        );
    }
    locations
}

/// Bytes used below `path` (symbolic links are not followed), or `None`
/// when it does not exist.
pub(crate) fn disk_usage(path: &Path) -> Option<u64> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    if !metadata.is_dir() {
        return Some(metadata.len());
    }
    let mut total = 0;
    let mut pending = vec![path.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                pending.push(entry.path());
            } else {
                total += metadata.len();
            }
        }
    }
    Some(total)
}

pub(crate) fn expand_tilde(path: &Path) -> PathBuf {
    eov_common::paths::expand_tilde(path)
}

/// A slide or folder path as eov shows, compares and stores it: absolute,
/// with symbolic links resolved, and on Windows without the `\\?\` prefix
/// that `std::fs::canonicalize` adds (SH-15). Explorer, other programs and
/// people reading a copied path do not understand `\\?\C:\…` or
/// `\\?\UNC\server\…`. A path that cannot be resolved (missing, or on a
/// drive that is not mounted) comes back as given, only simplified.
pub(crate) fn canonical(path: &Path) -> PathBuf {
    dunce::canonicalize(path).unwrap_or_else(|_| simplified(path))
}

/// `path` without a `\\?\` prefix that can be dropped safely, for paths
/// stored by older versions (Windows; unchanged elsewhere).
pub(crate) fn simplified(path: &Path) -> PathBuf {
    dunce::simplified(path).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_locations_cover_everything_and_stay_in_the_test_root() {
        let root = isolate_for_tests();
        let config = root.join("custom.toml");
        let plugins = root.join("my-plugins");
        let locations = storage_locations(&config, &plugins, None, &[]);
        let ids: Vec<&str> = locations
            .iter()
            .map(|location| location.id.as_str())
            .collect();
        for id in [
            "config",
            "recent-files",
            "session",
            "plugins",
            "updates",
            "bookmarks",
            "calibrations",
            "logs",
            "plugin-data",
            "plugin-packages-cache",
            "thumbnail-cache",
            "plugin-caches",
            "script-endpoint",
        ] {
            assert!(ids.contains(&id), "{id} is not listed");
        }
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "ids repeat: {ids:?}");
        for location in &locations {
            assert!(
                location.path.starts_with(root),
                "{} escapes the test root",
                location.path.display()
            );
        }
        assert_eq!(locations[0].path, config);
    }

    #[test]
    fn disk_usage_adds_up_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), [0; 10]).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b"), [0; 5]).unwrap();
        assert_eq!(disk_usage(dir.path()), Some(15));
        assert_eq!(disk_usage(&dir.path().join("a")), Some(10));
        assert_eq!(disk_usage(&dir.path().join("missing")), None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_lose_the_verbatim_prefix() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!canonical(dir.path()).to_string_lossy().starts_with(r"\\?\"));
        assert_eq!(
            simplified(Path::new(r"\\?\C:\slides\a.svs")),
            PathBuf::from(r"C:\slides\a.svs")
        );
        assert_eq!(
            canonical(Path::new(r"\\?\C:\no such eov folder\a.svs")),
            PathBuf::from(r"C:\no such eov folder\a.svs")
        );
    }
}
