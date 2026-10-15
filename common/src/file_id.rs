//! File and slide identities.
//!
//! * [`SlideIdentity`] is the stable key for data attached to a slide
//!   (annotations, analysis results, calibration). It combines OpenSlide's
//!   `openslide.quickhash-1`, the pyramid geometry and a format-aware hash
//!   over every file the slide consists of, and records the dimensions and
//!   MPP. It does not depend on the file name, so renamed or copied slides
//!   keep their key; it changes when the image data changes.
//! * [`compute_fingerprint`] / [`cached_sha256`] are the older per-file
//!   hashes, kept so existing databases keyed by them keep working.
//!   Migrate by storing both keys and looking up by
//!   [`SlideIdentity::key`] first.

use crate::WsiFile;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

pub const FINGERPRINT_CHUNK_COUNT: u64 = 256;
pub const FINGERPRINT_CHUNK_SIZE: usize = 1024;
pub const MIN_FINGERPRINT_FILE_SIZE: u64 = FINGERPRINT_CHUNK_COUNT * FINGERPRINT_CHUNK_SIZE as u64;

/// Sampled chunks per member when a slide has many member files.
const MEMBER_CHUNK_COUNT_MANY: u64 = 64;

/// Version prefix of [`SlideIdentity::key`].
pub const SLIDE_IDENTITY_VERSION: &str = "eov-slide-1";

#[derive(Clone, Copy, PartialEq, Eq)]
struct FileVersion {
    len: u64,
    modified: Option<SystemTime>,
}

impl FileVersion {
    fn of(path: &Path) -> io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

type Sha256Cache = Mutex<HashMap<PathBuf, (FileVersion, [u8; 32])>>;

static SHA256_CACHE: OnceLock<Sha256Cache> = OnceLock::new();

fn sha256_cache() -> &'static Sha256Cache {
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
    let mut buffer = vec![0_u8; 1024 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hasher.finalize().into())
}

/// SHA-256 of a file's contents, cached per path. The cached value is
/// reused only while the file's size and modification time are unchanged,
/// so a file replaced in place is hashed again.
pub fn cached_sha256(path: &Path) -> io::Result<[u8; 32]> {
    let path_key = normalize_path(path);
    let version = FileVersion::of(path)?;
    if let Some((cached_version, digest)) = sha256_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&path_key)
        .copied()
        && cached_version == version
    {
        return Ok(digest);
    }

    let digest = compute_sha256(path)?;
    sha256_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(path_key, (version, digest));
    Ok(digest)
}

/// SHA-256 over `chunk_count` evenly spaced 1 KiB chunks of a file (the
/// whole file when it is smaller than that). Legacy per-file key: see the
/// [module documentation](self).
pub fn compute_fingerprint(path: &Path) -> io::Result<[u8; 32]> {
    sampled_digest(path, FINGERPRINT_CHUNK_COUNT, false)
}

/// Digest of `chunk_count` evenly spaced chunks, optionally prefixed with
/// the file length.
fn sampled_digest(path: &Path, chunk_count: u64, with_length: bool) -> io::Result<[u8; 32]> {
    let len = std::fs::metadata(path)?.len();
    let min_size = chunk_count * FINGERPRINT_CHUNK_SIZE as u64;
    let mut hasher = Sha256::new();
    if with_length {
        hasher.update(len.to_le_bytes());
    }
    if len < min_size {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        hasher.update(&buffer);
        return Ok(hasher.finalize().into());
    }

    let mut file = File::open(path)?;
    let mut chunk = [0_u8; FINGERPRINT_CHUNK_SIZE];
    let max_offset = len - FINGERPRINT_CHUNK_SIZE as u64;

    for index in 0..chunk_count {
        let offset = if index == 0 {
            0
        } else {
            ((index as u128 * max_offset as u128) / (chunk_count - 1) as u128) as u64
        };
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut chunk)?;
        hasher.update(chunk);
    }

    Ok(hasher.finalize().into())
}

/// The files a slide consists of, for formats that spread a slide over
/// several files:
///
/// * MIRAX (`.mrxs`): the index file plus every file in the directory of the
///   same name (`Slidedat.ini`, `Index.dat`, `Data*.dat`);
/// * Hamamatsu VMS/VMU: the index file plus the `ImageFile*`, `MapFile` and
///   `OptimisationFile` it references;
/// * DICOM: none — a series spans many instance files and OpenSlide's
///   quickhash already identifies the series (its Series Instance UID), so
///   the identity must not depend on which instance was opened;
/// * everything else: the file itself.
pub fn slide_member_files(path: &Path, vendor: Option<&str>) -> io::Result<Vec<PathBuf>> {
    let extension = path
        .extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase());
    match (vendor, extension.as_deref()) {
        (Some("dicom"), _) => Ok(Vec::new()),
        (Some("mirax"), _) | (None, Some("mrxs")) => {
            let mut members = vec![path.to_path_buf()];
            let data_dir = path.with_extension("");
            if data_dir.is_dir() {
                let mut files: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|p| p.is_file())
                    .collect();
                files.sort();
                members.extend(files);
            }
            Ok(members)
        }
        (_, Some("vms" | "vmu")) => {
            let mut members = vec![path.to_path_buf()];
            let dir = path.parent().unwrap_or(Path::new("."));
            let text = std::fs::read(path)?;
            for line in String::from_utf8_lossy(&text).lines() {
                let Some((key, value)) = line.split_once('=') else {
                    continue;
                };
                let key = key.trim().to_ascii_lowercase();
                if key.starts_with("imagefile")
                    || key.starts_with("mapfile")
                    || key.starts_with("optimisationfile")
                {
                    let member = dir.join(value.trim());
                    if member.is_file() && !members.contains(&member) {
                        members.push(member);
                    }
                }
            }
            Ok(members)
        }
        _ => Ok(vec![path.to_path_buf()]),
    }
}

/// One file of a slide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlideMember {
    pub path: PathBuf,
    pub len: u64,
}

/// Content hash over member files: each member contributes a digest of its
/// length and sampled contents; digests are sorted so the result depends on
/// neither file names nor directory order.
pub fn members_content_hash(members: &[PathBuf]) -> io::Result<[u8; 32]> {
    let chunks = if members.len() > 4 {
        MEMBER_CHUNK_COUNT_MANY
    } else {
        FINGERPRINT_CHUNK_COUNT
    };
    let mut digests = members
        .iter()
        .map(|member| sampled_digest(member, chunks, true))
        .collect::<io::Result<Vec<_>>>()?;
    digests.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update((digests.len() as u64).to_le_bytes());
    for digest in &digests {
        hasher.update(digest);
    }
    Ok(hasher.finalize().into())
}

/// A stable identity of a slide; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct SlideIdentity {
    /// `openslide.quickhash-1`, when OpenSlide could compute it (it cannot
    /// for TIFF-like files whose smallest level exceeds 5 MiB).
    pub quickhash1: Option<String>,
    /// Hex SHA-256 over the member files (see [`members_content_hash`]).
    pub content_hash: String,
    /// Files covered by `content_hash`.
    pub members: Vec<SlideMember>,
    pub width: u64,
    pub height: u64,
    pub level_count: u32,
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    pub vendor: Option<String>,
    /// DICOM Series Instance UID, when the slide is a DICOM series.
    pub series_uid: Option<String>,
    key: String,
}

impl SlideIdentity {
    /// Computes the identity of an open slide. Reads samples of every
    /// member file, so call it off the UI thread.
    pub fn compute(wsi: &WsiFile) -> io::Result<Self> {
        let props = wsi.real_properties();
        let member_paths = slide_member_files(&props.path, props.vendor.as_deref())?;
        let members = member_paths
            .iter()
            .map(|path| {
                Ok(SlideMember {
                    path: path.clone(),
                    len: std::fs::metadata(path)?.len(),
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let content_hash = hex_digest(&members_content_hash(&member_paths)?);
        let series_uid = wsi
            .property("dicom.SeriesInstanceUID")
            .map(|uid| uid.trim().to_string());

        let mut hasher = Sha256::new();
        let mut field = |name: &str, value: &str| {
            hasher.update(name.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\n");
        };
        field("version", SLIDE_IDENTITY_VERSION);
        field("quickhash1", props.quickhash1.as_deref().unwrap_or(""));
        field("series", series_uid.as_deref().unwrap_or(""));
        field("dims", &format!("{}x{}", props.width, props.height));
        let levels: String = props
            .levels
            .iter()
            .map(|level| format!("{}x{};", level.width, level.height))
            .collect();
        field("levels", &levels);
        field("content", &content_hash);
        let key = format!(
            "{SLIDE_IDENTITY_VERSION}:{}",
            hex_digest(&hasher.finalize().into())
        );

        Ok(Self {
            quickhash1: props.quickhash1.clone(),
            content_hash,
            members,
            width: props.width,
            height: props.height,
            level_count: props.levels.len() as u32,
            mpp_x: props.mpp_x,
            mpp_y: props.mpp_y,
            vendor: props.vendor.clone(),
            series_uid,
            key,
        })
    }

    /// The identity as a stable string, e.g. `eov-slide-1:3f2a…`: the key
    /// to store annotations and results under.
    pub fn key(&self) -> &str {
        &self.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffSpec, write_tiff};

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

    #[test]
    fn fingerprint_keeps_the_legacy_algorithm() {
        // Existing annotation and eovae databases are keyed by this value.
        fn legacy(path: &Path) -> [u8; 32] {
            let len = std::fs::metadata(path).unwrap().len();
            let mut file = File::open(path).unwrap();
            let mut hasher = Sha256::new();
            let mut chunk = [0_u8; FINGERPRINT_CHUNK_SIZE];
            let max_offset = len - FINGERPRINT_CHUNK_SIZE as u64;
            for index in 0..FINGERPRINT_CHUNK_COUNT {
                let offset = if index == 0 {
                    0
                } else {
                    ((index as u128 * max_offset as u128) / (FINGERPRINT_CHUNK_COUNT - 1) as u128)
                        as u64
                };
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.read_exact(&mut chunk).unwrap();
                hasher.update(chunk);
            }
            hasher.finalize().into()
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");
        let bytes: Vec<u8> = (0..MIN_FINGERPRINT_FILE_SIZE as usize * 3 + 777)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(compute_fingerprint(&path).unwrap(), legacy(&path));
    }

    #[test]
    fn cached_sha256_notices_replaced_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.bin");
        std::fs::write(&path, b"first version").unwrap();
        let first = cached_sha256(&path).unwrap();
        assert_eq!(first, cached_sha256(&path).unwrap());
        std::fs::write(&path, b"second, longer version").unwrap();
        let second = cached_sha256(&path).unwrap();
        assert_ne!(first, second);
        assert_eq!(second, compute_sha256(&path).unwrap());
    }

    #[test]
    fn identity_is_stable_name_independent_and_content_sensitive() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.tiff");
        write_tiff(&a, &TiffSpec::pyramid(256, 192, 64, 2), |x, y| [x as u8, y as u8, 1]).unwrap();
        let copy = dir.path().join("renamed copy.tiff");
        std::fs::copy(&a, &copy).unwrap();
        let other = dir.path().join("other.tiff");
        write_tiff(&other, &TiffSpec::pyramid(256, 192, 64, 2), |x, y| [x as u8, y as u8, 2])
            .unwrap();

        let id_a = SlideIdentity::compute(&WsiFile::open(&a).unwrap()).unwrap();
        let id_again = WsiFile::open(&a).unwrap().identity().unwrap();
        let id_copy = SlideIdentity::compute(&WsiFile::open(&copy).unwrap()).unwrap();
        let id_other = SlideIdentity::compute(&WsiFile::open(&other).unwrap()).unwrap();

        assert!(id_a.key().starts_with("eov-slide-1:"));
        assert_eq!(id_a.key(), id_again.key());
        assert_eq!(id_a.key(), id_copy.key(), "renaming keeps the identity");
        assert_ne!(id_a.key(), id_other.key(), "different pixels, different identity");
        assert!(id_a.quickhash1.is_some());
        assert_eq!((id_a.width, id_a.height, id_a.level_count), (256, 192, 2));
        assert_eq!(id_a.members.len(), 1);
        assert_eq!(id_a.members[0].len, std::fs::metadata(&a).unwrap().len());
    }

    #[test]
    fn multi_file_formats_hash_every_member() {
        let dir = tempfile::tempdir().unwrap();
        // A MIRAX-style layout: index file plus a data directory.
        let mrxs = dir.path().join("slide.mrxs");
        std::fs::write(&mrxs, b"[GENERAL]\n").unwrap();
        let data = dir.path().join("slide");
        std::fs::create_dir(&data).unwrap();
        std::fs::write(data.join("Slidedat.ini"), b"[HIERARCHICAL]\n").unwrap();
        std::fs::write(data.join("Data0000.dat"), vec![1u8; 5000]).unwrap();
        std::fs::write(data.join("Data0001.dat"), vec![2u8; 5000]).unwrap();

        let members = slide_member_files(&mrxs, Some("mirax")).unwrap();
        assert_eq!(members.len(), 4);
        let before = members_content_hash(&members).unwrap();
        // A re-scan that keeps the .mrxs but rewrites a data file changes it.
        std::fs::write(data.join("Data0001.dat"), vec![3u8; 5000]).unwrap();
        assert_ne!(members_content_hash(&members).unwrap(), before);
        // Order does not matter.
        let mut reversed = members.clone();
        reversed.reverse();
        assert_eq!(
            members_content_hash(&reversed).unwrap(),
            members_content_hash(&members).unwrap()
        );

        // Hamamatsu VMS lists its image files.
        let vms = dir.path().join("scan.vms");
        std::fs::write(dir.path().join("scan_0.jpg"), b"jpeg").unwrap();
        std::fs::write(dir.path().join("scan.map"), b"map").unwrap();
        std::fs::write(
            &vms,
            b"[Virtual Microscope Specimen]\nImageFile=scan_0.jpg\nMapFile = scan.map\n\
              ImageFile(1,0)=missing.jpg\n",
        )
        .unwrap();
        let members = slide_member_files(&vms, Some("hamamatsu")).unwrap();
        assert_eq!(
            members,
            [vms.clone(), dir.path().join("scan_0.jpg"), dir.path().join("scan.map")]
        );

        // DICOM identity comes from the series, not from the opened file.
        assert!(slide_member_files(&dir.path().join("x.dcm"), Some("dicom")).unwrap().is_empty());
        assert_eq!(
            slide_member_files(&dir.path().join("x.svs"), Some("aperio")).unwrap(),
            [dir.path().join("x.svs")]
        );
    }
}
