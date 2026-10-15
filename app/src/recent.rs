//! Recently opened slides and folders (SH-14).
//!
//! Stored in `<eov home>/recent_files.toml`, next to `config.toml`, as
//! `[[entry]]` tables. Paths that are not valid UTF-8 are kept exactly (as
//! hex of the platform's encoding) instead of being mangled. Saving merges
//! with the file on disk, so two running eovs do not erase each other's
//! entries, and writes atomically. Nothing is checked on disk when the list
//! is loaded: entries on a drive that is not mounted right now stay, and a
//! hung network mount cannot block start-up. `remember_recent_files =
//! false` in `config.toml` turns the history off; it can also be cleared
//! from the File menu.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Entries kept.
pub const MAX_RECENT_FILES: usize = 10;

/// A recently opened slide or folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecentFile {
    pub path: PathBuf,
    /// Display name (the file name).
    pub name: String,
    /// Opened as a folder (series).
    pub is_dir: bool,
}

impl RecentFile {
    pub fn new(path: PathBuf, is_dir: bool) -> Self {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Self { path, name, is_dir }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default, rename = "entry")]
    entries: Vec<Entry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    /// The path in the platform's encoding, as hex, when it is not UTF-8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    os_path: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    dir: bool,
}

impl Entry {
    fn from_recent(recent: &RecentFile) -> Self {
        match recent.path.to_str() {
            Some(path) => Self {
                path: Some(path.to_string()),
                os_path: None,
                dir: recent.is_dir,
            },
            None => Self {
                path: None,
                os_path: Some(encode_os(&recent.path)),
                dir: recent.is_dir,
            },
        }
    }

    fn to_recent(&self) -> Option<RecentFile> {
        let path = match (&self.path, &self.os_path) {
            (_, Some(hex)) => decode_os(hex)?,
            (Some(path), None) => PathBuf::from(path),
            (None, None) => return None,
        };
        // Older versions stored `\\?\` paths on Windows (SH-15).
        Some(RecentFile::new(crate::paths::simplified(&path), self.dir))
    }
}

#[cfg(unix)]
pub(crate) fn encode_os(path: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str()
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(unix)]
pub(crate) fn decode_os(hex: &str) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    Some(PathBuf::from(std::ffi::OsString::from_vec(bytes)))
}

#[cfg(windows)]
pub(crate) fn encode_os(path: &Path) -> String {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .map(|unit| format!("{unit:04x}"))
        .collect()
}

#[cfg(windows)]
pub(crate) fn decode_os(hex: &str) -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    let units = (0..hex.len())
        .step_by(4)
        .map(|i| u16::from_str_radix(hex.get(i..i + 4)?, 16).ok())
        .collect::<Option<Vec<u16>>>()?;
    Some(PathBuf::from(std::ffi::OsString::from_wide(&units)))
}

fn store_path() -> PathBuf {
    crate::paths::recent_files_file()
}

fn read_store(path: &Path) -> Option<Vec<RecentFile>> {
    let text = fs::read_to_string(path).ok()?;
    match toml::from_str::<Store>(&text) {
        Ok(store) => Some(store.entries.iter().filter_map(Entry::to_recent).collect()),
        Err(error) => {
            warn!("Ignoring unreadable {}: {error}", path.display());
            None
        }
    }
}

/// The list written by eov < 1.0 (one path per line), for migration.
fn read_legacy() -> Vec<RecentFile> {
    let Some(dir) = crate::paths::legacy_config_dir() else {
        return Vec::new();
    };
    fs::read_to_string(dir.join("recent_files.txt"))
        .map(|text| {
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| RecentFile::new(PathBuf::from(line), false))
                .collect()
        })
        .unwrap_or_default()
}

fn dedupe(entries: impl IntoIterator<Item = RecentFile>) -> Vec<RecentFile> {
    let mut seen = std::collections::HashSet::new();
    entries
        .into_iter()
        .filter(|entry| seen.insert(entry.path.clone()))
        .take(MAX_RECENT_FILES)
        .collect()
}

/// The saved history (empty when it is turned off).
pub fn load() -> Vec<RecentFile> {
    if !crate::config::load_remember_recent_files() {
        return Vec::new();
    }
    let path = store_path();
    let entries = read_store(&path).unwrap_or_else(read_legacy);
    dedupe(entries)
}

/// Puts `path` first. Returns whether the list changed.
pub fn record(list: &mut Vec<RecentFile>, path: &Path, is_dir: bool) -> bool {
    if !crate::config::load_remember_recent_files() {
        return false;
    }
    if list.first().is_some_and(|first| first.path == path) {
        return false;
    }
    list.retain(|entry| entry.path != path);
    list.insert(0, RecentFile::new(path.to_path_buf(), is_dir));
    list.truncate(MAX_RECENT_FILES);
    true
}

/// Saves `list`, keeping entries another eov added since this one loaded.
pub fn save(list: &[RecentFile]) {
    if !crate::config::load_remember_recent_files() {
        return;
    }
    let path = store_path();
    let on_disk = read_store(&path).unwrap_or_default();
    let merged = dedupe(list.iter().cloned().chain(on_disk));
    if let Err(error) = write(&path, &merged) {
        warn!("Could not save the recent files: {error}");
    }
}

/// Forgets the history, here and on disk.
pub fn clear(list: &mut Vec<RecentFile>) {
    list.clear();
    if let Err(error) = write(&store_path(), &[]) {
        warn!("Could not clear the recent files: {error}");
    }
}

fn write(path: &Path, entries: &[RecentFile]) -> std::io::Result<()> {
    let store = Store {
        entries: entries.iter().map(Entry::from_recent).collect(),
    };
    let text = toml::to_string(&store).map_err(std::io::Error::other)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text)?;
    fs::rename(&temp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip_including_odd_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recent_files.toml");
        let mut entries = vec![
            RecentFile::new(PathBuf::from("/slides/ case 1 .svs"), false),
            RecentFile::new(PathBuf::from("/slides/line\nbreak.svs"), false),
            RecentFile::new(PathBuf::from("/slides/series"), true),
        ];
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            entries.push(RecentFile::new(
                PathBuf::from(std::ffi::OsString::from_vec(b"/slides/\xff.svs".to_vec())),
                false,
            ));
        }
        write(&path, &entries).unwrap();
        assert_eq!(read_store(&path).unwrap(), entries);
    }

    #[test]
    fn recording_moves_entries_to_the_front_once() {
        let mut list = vec![
            RecentFile::new(PathBuf::from("/a.svs"), false),
            RecentFile::new(PathBuf::from("/b.svs"), false),
        ];
        assert!(!record(&mut list, Path::new("/a.svs"), false));
        assert!(record(&mut list, Path::new("/b.svs"), false));
        let paths: Vec<_> = list.iter().map(|e| e.path.clone()).collect();
        assert_eq!(paths, [PathBuf::from("/b.svs"), PathBuf::from("/a.svs")]);
    }

    #[test]
    fn duplicates_are_dropped_and_the_list_is_bounded() {
        let entries =
            (0..30).map(|i| RecentFile::new(PathBuf::from(format!("/{}.svs", i % 15)), false));
        let deduped = dedupe(entries);
        assert_eq!(deduped.len(), MAX_RECENT_FILES);
        assert_eq!(deduped[0].path, PathBuf::from("/0.svs"));
    }
}
