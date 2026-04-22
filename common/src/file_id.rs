use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub const FINGERPRINT_CHUNK_COUNT: u64 = 256;
pub const FINGERPRINT_CHUNK_SIZE: usize = 1024;
pub const MIN_FINGERPRINT_FILE_SIZE: u64 = FINGERPRINT_CHUNK_COUNT * FINGERPRINT_CHUNK_SIZE as u64;

static SHA256_CACHE: OnceLock<Mutex<HashMap<PathBuf, [u8; 32]>>> = OnceLock::new();

fn sha256_cache() -> &'static Mutex<HashMap<PathBuf, [u8; 32]>> {
    SHA256_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn normalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn compute_sha256(path: &Path) -> io::Result<[u8; 32]> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hasher.finalize().into())
}

pub fn cached_sha256(path: &Path) -> io::Result<[u8; 32]> {
    let path_key = normalize_path(path);
    if let Some(digest) = sha256_cache().lock().unwrap().get(&path_key).copied() {
        return Ok(digest);
    }

    let digest = compute_sha256(path)?;
    sha256_cache().lock().unwrap().insert(path_key, digest);
    Ok(digest)
}

pub fn compute_fingerprint(path: &Path) -> io::Result<[u8; 32]> {
    let len = std::fs::metadata(path)?.len();
    if len < MIN_FINGERPRINT_FILE_SIZE {
        return compute_sha256(path);
    }

    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut chunk = [0_u8; FINGERPRINT_CHUNK_SIZE];
    let max_offset = len - FINGERPRINT_CHUNK_SIZE as u64;

    for index in 0..FINGERPRINT_CHUNK_COUNT {
        let offset = if index == 0 {
            0
        } else {
            ((index as u128 * max_offset as u128) / (FINGERPRINT_CHUNK_COUNT - 1) as u128) as u64
        };
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut chunk)?;
        hasher.update(chunk);
    }

    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_files_fingerprint_as_full_sha() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.bin");
        std::fs::write(&path, vec![0x5a; 32 * 1024]).unwrap();

        assert_eq!(
            compute_fingerprint(&path).unwrap(),
            compute_sha256(&path).unwrap()
        );
    }

    #[test]
    fn fingerprint_uses_header_and_even_distribution() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");
        let mut bytes = vec![0_u8; MIN_FINGERPRINT_FILE_SIZE as usize + 8192];
        let last_chunk_start = bytes.len() - FINGERPRINT_CHUNK_SIZE;
        bytes[..FINGERPRINT_CHUNK_SIZE].fill(0x11);
        bytes[last_chunk_start..].fill(0x77);
        std::fs::write(&path, bytes).unwrap();

        let digest = compute_fingerprint(&path).unwrap();
        assert_ne!(digest, [0_u8; 32]);
        assert_ne!(digest, compute_sha256(&path).unwrap());
    }
}
