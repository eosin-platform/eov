//! Slide identity for result keys (VAE-5).
//!
//! The host's `FileInfo::identity` (derived from `openslide.quickhash-1`) is
//! the preferred key. Hosts that do not provide it yet get a fallback
//! fingerprint: SHA-256 over the file length, the pyramid geometry and 64
//! evenly spaced 4 KiB chunks of the file. It reads 256 KiB instead of the
//! whole slide (the full-file hash took minutes for multi-GB slides on
//! network shares, VAE-13) and runs off the UI thread. Both keys are stored
//! so results move to the host identity once it appears (see
//! [`crate::db`]).

use crate::db::SlideKeys;
use eov_plugin_api::protocol::FileInfo;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const CHUNKS: u64 = 64;
const CHUNK: usize = 4096;

/// A slide's keys, plus the text key used in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlideIdentity {
    pub keys: SlideKeys,
    /// The host identity if known, else the fingerprint, else a key that is
    /// valid for this session only (results are then not persisted).
    pub key: String,
    pub persistent: bool,
}

/// Fingerprint of a slide file and its geometry.
pub fn fingerprint(path: &Path, file: &FileInfo) -> io::Result<String> {
    let mut handle = File::open(path)?;
    let len = handle.metadata()?.len();
    let mut hasher = Sha256::new();
    hasher.update(b"eovae-fp1\0");
    hasher.update(len.to_le_bytes());
    hasher.update(file.width.to_le_bytes());
    hasher.update(file.height.to_le_bytes());
    for level in &file.levels {
        hasher.update(level.width.to_le_bytes());
        hasher.update(level.height.to_le_bytes());
    }
    let mut chunk = vec![0u8; CHUNK];
    if len <= CHUNKS * CHUNK as u64 {
        let mut all = Vec::new();
        handle.read_to_end(&mut all)?;
        hasher.update(&all);
    } else {
        let last = len - CHUNK as u64;
        for index in 0..CHUNKS {
            let offset = last * index / (CHUNKS - 1);
            handle.seek(SeekFrom::Start(offset))?;
            handle.read_exact(&mut chunk)?;
            hasher.update(&chunk);
        }
    }
    let digest = hasher.finalize();
    Ok(format!(
        "fp1:{}",
        digest[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    ))
}

/// Resolves a slide's identity. Reads the slide file when the host gives no
/// identity; call it off the UI thread.
pub fn resolve(file: &FileInfo) -> SlideIdentity {
    let identity = file.identity.clone().filter(|id| !id.trim().is_empty());
    // The fingerprint is computed even with a host identity, so results
    // stored under it by older hosts are found and migrated.
    let fingerprint = fingerprint(Path::new(&file.path), file).ok();
    let key = identity
        .clone()
        .or_else(|| fingerprint.clone())
        .unwrap_or_else(|| format!("session:{}", file.file_id));
    SlideIdentity {
        persistent: identity.is_some() || fingerprint.is_some(),
        keys: SlideKeys {
            identity,
            fingerprint,
            width: file.width,
            height: file.height,
        },
        key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_info(path: &Path) -> FileInfo {
        FileInfo {
            file_id: 3,
            path: path.to_string_lossy().into_owned(),
            width: 1000,
            height: 500,
            ..Default::default()
        }
    }

    #[test]
    fn fingerprints_follow_content_not_names() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.svs");
        let mut data = vec![0u8; 1 << 20];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i * 31 % 251) as u8;
        }
        std::fs::File::create(&a).unwrap().write_all(&data).unwrap();
        let b = dir.path().join("renamed.svs");
        std::fs::copy(&a, &b).unwrap();
        let fa = fingerprint(&a, &file_info(&a)).unwrap();
        assert_eq!(fa, fingerprint(&b, &file_info(&b)).unwrap(), "copies match");
        // Changing a sampled byte changes the fingerprint.
        data[0] ^= 0xFF;
        std::fs::write(&b, &data).unwrap();
        assert_ne!(fa, fingerprint(&b, &file_info(&b)).unwrap());
        // So does different geometry for the same bytes.
        let mut other = file_info(&a);
        other.width = 999;
        assert_ne!(fa, fingerprint(&a, &other).unwrap());
    }

    #[test]
    fn host_identity_wins_and_unreadable_files_stay_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.tif");
        std::fs::write(&path, b"tiny").unwrap();
        let mut info = file_info(&path);
        info.identity = Some("quickhash1:abc".to_string());
        let resolved = resolve(&info);
        assert_eq!(resolved.key, "quickhash1:abc");
        assert!(resolved.keys.fingerprint.is_some(), "kept for migration");
        info.identity = None;
        let resolved = resolve(&info);
        assert!(resolved.key.starts_with("fp1:"));
        info.path = dir.path().join("missing").to_string_lossy().into_owned();
        let resolved = resolve(&info);
        assert_eq!(resolved.key, "session:3");
        assert!(!resolved.persistent);
    }
}
