//! Central resolution of every on-disk location used by eov and its plugins.
//!
//! All code that reads or writes files outside of the slides themselves must
//! resolve its location through this module so that:
//!
//! - users and administrators can redirect state with environment variables
//!   (`EOV_HOME`, `EOV_CONFIG`, `EOV_DATA_DIR`, `EOV_CACHE_DIR`,
//!   `EOV_PLUGIN_DIR`), for example for portable or managed installs, and
//! - test suites never touch the developer's real configuration, caches or
//!   plugin databases ([`isolate_for_tests`]).
//!
//! Layout (defaults):
//!
//! | Location            | Default                                   | Contents |
//! |---------------------|-------------------------------------------|----------|
//! | [`home_dir`]        | `~/.eov`                                  | `config.toml`, `recent_files.toml`, `session.toml`, `plugins/`, `updates/` |
//! | [`data_dir`]        | platform local data dir + `/eov`          | plugin data (`plugins/<id>/`), logs |
//! | [`cache_dir`]       | platform cache dir + `/eov`               | extracted plugin packages, thumbnails, plugin caches |
//! | [`legacy_config_dir`] | platform config dir + `/eov`            | files written by eov < 1.0 (read for migration only) |

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ROOT_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Redirects every location below `root`.
///
/// Environment overrides are ignored while a root override is active. Returns
/// `false` when a different override was already installed.
pub fn set_root_override(root: PathBuf) -> bool {
    match ROOT_OVERRIDE.get() {
        Some(existing) => existing == &root,
        None => ROOT_OVERRIDE.set(root).is_ok() || ROOT_OVERRIDE.get().is_some(),
    }
}

/// Returns the active root override, if any.
pub fn root_override() -> Option<&'static Path> {
    ROOT_OVERRIDE.get().map(PathBuf::as_path)
}

/// Isolates the current process for tests: every location resolves inside a
/// fresh temporary directory that is unique to this process.
///
/// Idempotent; returns the isolation root.
pub fn isolate_for_tests() -> &'static Path {
    ROOT_OVERRIDE.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let dir = std::env::temp_dir().join(format!("eov-test-{}-{nanos}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
}

fn env_path(var: &str) -> Option<PathBuf> {
    if ROOT_OVERRIDE.get().is_some() {
        return None;
    }
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn user_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// The eov home directory (`~/.eov`, or `$EOV_HOME`).
pub fn home_dir() -> PathBuf {
    if let Some(root) = ROOT_OVERRIDE.get() {
        return root.join("home");
    }
    env_path("EOV_HOME").unwrap_or_else(|| user_home().join(".eov"))
}

/// The main configuration file (`$EOV_CONFIG`, else `<home>/config.toml`).
pub fn config_file() -> PathBuf {
    env_path("EOV_CONFIG").unwrap_or_else(|| home_dir().join("config.toml"))
}

/// Recently opened files (`<home>/recent_files.toml`).
pub fn recent_files_file() -> PathBuf {
    home_dir().join("recent_files.toml")
}

/// Persisted session: open tabs, panes and viewports (`<home>/session.toml`).
pub fn session_file() -> PathBuf {
    home_dir().join("session.toml")
}

/// The directory scanned for `.eop` plugin packages
/// (`$EOV_PLUGIN_DIR`, else `<home>/plugins`).
pub fn plugin_dir() -> PathBuf {
    env_path("EOV_PLUGIN_DIR").unwrap_or_else(|| home_dir().join("plugins"))
}

/// Where downloaded update bundles are staged (`<home>/updates`).
pub fn updates_dir() -> PathBuf {
    home_dir().join("updates")
}

/// Durable application data (`$EOV_DATA_DIR`, else the platform's local data
/// directory + `/eov`). Local, not roaming, on Windows.
pub fn data_dir() -> PathBuf {
    if let Some(root) = ROOT_OVERRIDE.get() {
        return root.join("data");
    }
    env_path("EOV_DATA_DIR")
        .or_else(|| dirs::data_local_dir().map(|dir| dir.join("eov")))
        .unwrap_or_else(|| home_dir().join("data"))
}

/// Disposable caches (`$EOV_CACHE_DIR`, else the platform cache dir + `/eov`).
pub fn cache_dir() -> PathBuf {
    if let Some(root) = ROOT_OVERRIDE.get() {
        return root.join("cache");
    }
    env_path("EOV_CACHE_DIR")
        .or_else(|| dirs::cache_dir().map(|dir| dir.join("eov")))
        .unwrap_or_else(|| home_dir().join("cache"))
}

/// Log files (`<data>/logs`).
pub fn log_dir() -> PathBuf {
    data_dir().join("logs")
}

/// Extracted plugin packages (`<cache>/plugin-packages`).
pub fn plugin_package_cache_dir() -> PathBuf {
    cache_dir().join("plugin-packages")
}

/// Series-bar thumbnail cache (`<cache>/series_thumbnails`).
pub fn thumbnail_cache_dir() -> PathBuf {
    cache_dir().join("series_thumbnails")
}

/// Durable per-plugin data (`<data>/plugins/<plugin_id>`).
pub fn plugin_data_dir(plugin_id: &str) -> PathBuf {
    data_dir().join("plugins").join(plugin_id)
}

/// Disposable per-plugin cache (`<cache>/plugins/<plugin_id>`).
pub fn plugin_cache_dir(plugin_id: &str) -> PathBuf {
    cache_dir().join("plugins").join(plugin_id)
}

/// The configuration directory used by eov < 1.0 (platform config dir +
/// `/eov`). Only read to migrate old files.
pub fn legacy_config_dir() -> Option<PathBuf> {
    if let Some(root) = ROOT_OVERRIDE.get() {
        return Some(root.join("legacy-config"));
    }
    dirs::config_dir().map(|dir| dir.join("eov"))
}

/// Expands a leading `~` (or `~/`) to the user's home directory.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if text == "~" {
        return user_home();
    }
    if let Some(rest) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) {
        return user_home().join(rest);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_redirects_every_location() {
        let root = isolate_for_tests().to_path_buf();
        for path in [
            home_dir(),
            config_file(),
            recent_files_file(),
            session_file(),
            plugin_dir(),
            updates_dir(),
            data_dir(),
            cache_dir(),
            log_dir(),
            plugin_package_cache_dir(),
            thumbnail_cache_dir(),
            plugin_data_dir("example"),
            plugin_cache_dir("example"),
            legacy_config_dir().unwrap(),
        ] {
            assert!(
                path.starts_with(&root),
                "{} escapes {}",
                path.display(),
                root.display()
            );
        }
    }

    #[test]
    fn tilde_expansion() {
        let home = user_home();
        assert_eq!(expand_tilde(Path::new("~")), home);
        assert_eq!(expand_tilde(Path::new("~/a/b")), home.join("a/b"));
        assert_eq!(
            expand_tilde(Path::new("/abs/~/x")),
            PathBuf::from("/abs/~/x")
        );
    }
}
