//! Cancellable slide hashing over every member file (DS-6).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

/// Hex SHA-256 identifying a slide's content.
///
/// For a single-file slide this is the plain SHA-256 of the file (the value
/// pre-1.0 datasets recorded). For multi-file slides (MIRAX, VMS, DICOM
/// series) it is the SHA-256 over, for each member in sorted order, the
/// member's file name, its length (u64 little-endian) and its bytes.
/// Returns `Ok(None)` if `cancel` was set while hashing.
pub fn slide_sha256(members: &[PathBuf], cancel: &AtomicBool) -> std::io::Result<Option<String>> {
    let mut members = members.to_vec();
    members.sort();
    let mut hasher = Sha256::new();
    let multi = members.len() > 1;
    let mut buffer = vec![0u8; 4 << 20];
    for path in &members {
        if multi {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            hasher.update(name.as_bytes());
            hasher.update(std::fs::metadata(path)?.len().to_le_bytes());
        }
        if !hash_file(path, &mut hasher, &mut buffer, cancel)? {
            return Ok(None);
        }
    }
    Ok(Some(crate::file_id::hex_digest(&hasher.finalize().into())))
}

fn hash_file(
    path: &Path,
    hasher: &mut Sha256,
    buffer: &mut [u8],
    cancel: &AtomicBool,
) -> std::io::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let n = file.read(buffer)?;
        if n == 0 {
            return Ok(true);
        }
        hasher.update(&buffer[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_files_hash_like_sha256sum() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.svs");
        std::fs::write(&path, b"abc").unwrap();
        let hash = slide_sha256(&[path], &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn member_files_change_the_hash_and_cancellation_stops() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("s.mrxs");
        let data = dir.path().join("Data0000.dat");
        std::fs::write(&index, b"index").unwrap();
        std::fs::write(&data, b"pixels-1").unwrap();
        let members = vec![index.clone(), data.clone()];
        let a = slide_sha256(&members, &AtomicBool::new(false)).unwrap();
        std::fs::write(&data, b"pixels-2").unwrap();
        let b = slide_sha256(&members, &AtomicBool::new(false)).unwrap();
        assert_ne!(a, b, "re-scanned data must change the identity");
        assert_eq!(
            slide_sha256(&members, &AtomicBool::new(true)).unwrap(),
            None
        );
    }
}
