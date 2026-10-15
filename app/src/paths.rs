//! App-side access to on-disk locations.
//!
//! Thin wrappers over [`eov_common::paths`]. In test builds every accessor
//! first isolates the process into a temporary root, so no unit test can read
//! or modify the developer's real configuration, caches or plugin data.

// Several accessors are consumed by the session, recent-files and plugin
// data work that follows; keep the full set together.
#![allow(dead_code)]

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

pub(crate) fn expand_tilde(path: &Path) -> PathBuf {
    eov_common::paths::expand_tilde(path)
}
