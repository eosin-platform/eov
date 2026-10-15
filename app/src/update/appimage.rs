//! In-place AppImage replacement.
//!
//! The running AppImage keeps its (FUSE-mounted) old file open, so the file
//! can be replaced while eov runs. The new image is downloaded next to it,
//! made executable and renamed over the old path in one atomic step; the old
//! image is kept as a hidden `.<name>.previous` hard link (or copy) for
//! `eov update --rollback`, and the directory is fsynced so the rename
//! survives a crash.

use super::registry::sync_directory;
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Where the rollback copy of `current` lives.
pub fn previous_path(current: &Path) -> PathBuf {
    let name = current
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "eov.AppImage".into());
    current.with_file_name(format!(".{name}.previous"))
}

/// Replaces `current` with the verified `staged` image (which must be in
/// the same directory). Returns the rollback copy, if one could be kept.
pub fn replace(current: &Path, staged: tempfile::TempPath) -> Result<Option<PathBuf>> {
    set_executable(&staged)?;
    let previous = previous_path(current);
    let _ = fs::remove_file(&previous);
    let kept = fs::hard_link(current, &previous)
        .or_else(|_| fs::copy(current, &previous).map(|_| ()))
        .is_ok();
    staged
        .persist(current)
        .map_err(|error| error.error)
        .with_context(|| format!("could not replace {}", current.display()))?;
    if let Some(parent) = current.parent() {
        sync_directory(parent);
    }
    Ok(kept.then_some(previous))
}

/// Swaps `current` and `previous` (rollback, and rollback of a rollback).
pub fn swap(current: &Path, previous: &Path) -> Result<()> {
    let parked = current.with_file_name(format!(
        ".{}.swap",
        current
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    let _ = fs::remove_file(&parked);
    fs::hard_link(current, &parked)
        .or_else(|_| fs::copy(current, &parked).map(|_| ()))
        .with_context(|| format!("could not keep a copy of {}", current.display()))?;
    if let Err(error) = fs::rename(previous, current) {
        let _ = fs::remove_file(&parked);
        return Err(error).with_context(|| format!("could not restore {}", previous.display()));
    }
    fs::rename(&parked, previous)
        .with_context(|| format!("could not keep {} for a later rollback", current.display()))?;
    if let Some(parent) = current.parent() {
        sync_directory(parent);
    }
    Ok(())
}

pub fn set_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn staged(dir: &Path, contents: &[u8]) -> tempfile::TempPath {
        let file = tempfile::Builder::new()
            .prefix(".eov-download-")
            .tempfile_in(dir)
            .unwrap();
        fs::write(file.path(), contents).unwrap();
        file.into_temp_path()
    }

    #[test]
    fn replacement_is_atomic_keeps_the_previous_image_and_is_executable() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("eov.AppImage");
        fs::write(&current, b"old").unwrap();
        let previous = replace(&current, staged(temp.path(), b"new"))
            .unwrap()
            .unwrap();
        assert_eq!(fs::read(&current).unwrap(), b"new");
        assert_eq!(fs::read(&previous).unwrap(), b"old");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&current).unwrap().permissions().mode();
            assert_ne!(mode & 0o111, 0);
        }
        // Only the image and its rollback copy remain.
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
    }

    #[test]
    fn swap_rolls_back_and_forward() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("eov.AppImage");
        fs::write(&current, b"old").unwrap();
        let previous = replace(&current, staged(temp.path(), b"new"))
            .unwrap()
            .unwrap();
        swap(&current, &previous).unwrap();
        assert_eq!(fs::read(&current).unwrap(), b"old");
        assert_eq!(fs::read(&previous).unwrap(), b"new");
        swap(&current, &previous).unwrap();
        assert_eq!(fs::read(&current).unwrap(), b"new");
    }

    #[test]
    fn a_failed_replacement_leaves_the_current_image() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("missing-dir").join("eov.AppImage");
        assert!(replace(&current, staged(temp.path(), b"new")).is_err());
    }
}
