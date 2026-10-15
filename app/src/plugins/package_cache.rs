//! Content-addressed cache of extracted plugin packages (`.eop`).
//!
//! A plugin package is a tar archive, optionally compressed with zstd
//! (detected by its magic bytes). Before eov can read a package's manifest or
//! load its native library, the package is extracted into
//! `<cache>/plugin-packages/<sha256>/`, where `<sha256>` is the SHA-256 of the
//! package file itself. This gives the cache three properties the previous
//! path/mtime-keyed layout lacked:
//!
//! - **Exact keys.** Replacing or removing a package never deletes another
//!   package's extraction, and identical packages (for example a staged
//!   download and the installed `<id>.eop`) share one extraction, so the
//!   extraction done at install time is reused by the next launch.
//! - **Safe garbage collection.** Every process that uses an extraction holds
//!   a shared lock on `locks/<sha256>.lock` for its whole lifetime. The
//!   collector only removes entries that no index record references, that are
//!   older than a grace period, and whose lock it can take exclusively, so an
//!   extraction in use by a running eov (for example lazily loaded `.slint`
//!   files, or a DLL on Windows) is never deleted underneath it.
//! - **Cheap re-use.** `index.json` maps each package path and file identity
//!   (size, mtime, inode or creation time) to its SHA-256, so unchanged
//!   packages are not re-hashed on every launch or CLI command.
//!
//! Extraction is hardened: symlinks, hardlinks, devices, FIFOs, absolute,
//! drive-prefixed and `..` paths, duplicate manifests and duplicate files are
//! rejected; permissions are masked with `0o022`; and the total size and
//! entry count are bounded.

use eov_plugin_api::{PluginError, PluginResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

/// File extension of plugin packages.
pub(crate) const PACKAGE_EXTENSION: &str = "eop";
/// The zstd frame magic number (`0xFD2FB528`, little endian).
pub(crate) const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

const INDEX_FILE: &str = "index.json";
const INDEX_LOCK: &str = ".index.lock";
const GC_LOCK: &str = ".gc.lock";
const LOCKS_DIR: &str = "locks";
const COMPLETE_MARKER: &str = ".eov-extracted.json";
const PRIVATE_PREFIX: &str = ".private-";
const CACHE_FORMAT: u32 = 1;
const MAX_INDEX_RECORDS: usize = 4096;

/// Entries younger than this are never collected, which covers the short
/// window between finishing an extraction and taking the shared lock.
#[cfg_attr(test, allow(dead_code))]
pub(crate) const DEFAULT_GC_GRACE: Duration = Duration::from_secs(60 * 60);
/// Extractions in the pre-1.0 `<stem>-<len>-<mtime>-<hash>` layout carry no
/// lock, so they are only collected once they are clearly stale.
const LEGACY_ENTRY_GRACE: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Upper bound on the bytes a single package may expand to.
const MAX_EXTRACTED_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Upper bound on the number of archive entries in a single package.
const MAX_ARCHIVE_ENTRIES: u64 = 500_000;
/// How long to wait for another process that is extracting the same package.
const EXCLUSIVE_LOCK_TIMEOUT: Duration = Duration::from_secs(600);

/// Compression of a plugin package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackageFormat {
    /// A plain (uncompressed) tar archive.
    Tar,
    /// A zstd-compressed tar archive.
    ZstdTar,
}

/// A package that has been extracted into the cache.
#[derive(Debug, Clone)]
pub(crate) struct ExtractedPackage {
    /// SHA-256 of the package file (lower-case hex).
    pub(crate) sha256: String,
    /// Directory that contains the package's `plugin.toml`.
    pub(crate) root: PathBuf,
}

/// What a garbage-collection pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct GcReport {
    pub(crate) removed: Vec<PathBuf>,
    pub(crate) kept_in_use: usize,
    pub(crate) kept_referenced: usize,
    pub(crate) kept_recent: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct FileIdentity {
    len: u64,
    modified_secs: u64,
    modified_nanos: u32,
    /// Inode and device on Unix; creation time on Windows. A package that is
    /// replaced by renaming another file over it always changes this value.
    file_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexRecord {
    path: String,
    #[serde(flatten)]
    identity: FileIdentity,
    sha256: String,
    last_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Index {
    format: u32,
    #[serde(default)]
    records: Vec<IndexRecord>,
}

impl Default for Index {
    fn default() -> Self {
        Self {
            format: CACHE_FORMAT,
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CompleteMarker {
    format: u32,
    sha256: String,
    /// Path of the manifest directory relative to the entry (may be empty).
    root: String,
    extracted_at: u64,
}

/// Locks held by this process on extractions it uses, keyed by lock path.
/// They are released when the process exits.
fn in_use_locks() -> &'static Mutex<HashMap<PathBuf, File>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, File>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The content-addressed package cache rooted at one directory.
#[derive(Debug, Clone)]
pub(crate) struct PackageCache {
    root: PathBuf,
}

impl Default for PackageCache {
    fn default() -> Self {
        Self::at(crate::paths::plugin_package_cache_dir())
    }
}

impl PackageCache {
    /// A cache rooted at `root` (the directory is created on demand).
    pub(crate) fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the SHA-256 of `package`, re-hashing only when the file's
    /// identity changed since it was last seen.
    pub(crate) fn sha256(&self, package: &Path) -> PluginResult<String> {
        let metadata = fs::metadata(package)?;
        if !metadata.is_file() {
            return Err(PluginError::Other(format!(
                "plugin package is not a regular file: {}",
                package.display()
            )));
        }
        let identity = identity_of(&metadata);
        let key = index_key(package);
        if let Some(key) = key.as_deref()
            && let Some(sha256) = self.lookup(key, &identity)
        {
            return Ok(sha256);
        }
        let sha256 = sha256_file(package)?;
        // The file must not have changed while it was hashed.
        if identity_of(&fs::metadata(package)?) != identity {
            return Err(PluginError::Other(format!(
                "plugin package changed while it was being read: {}",
                package.display()
            )));
        }
        if let Some(key) = key {
            self.record(&key, identity, &sha256);
        }
        Ok(sha256)
    }

    /// Extracts `package` (if this content has not been extracted yet) and
    /// marks the extraction as in use by this process.
    pub(crate) fn ensure_extracted(&self, package: &Path) -> PluginResult<ExtractedPackage> {
        let sha256 = self.sha256(package)?;
        let entry = self.entry_dir(&sha256);
        let lock_path = self.lock_path(&sha256);

        // This process already uses the entry: its shared lock would make an
        // exclusive lock wait for ourselves, so never take one here.
        let held = in_use_locks()
            .lock()
            .map(|locks| locks.contains_key(&lock_path))
            .unwrap_or(false);
        if held {
            return match read_complete_marker(&entry, &sha256) {
                Some(root) => Ok(ExtractedPackage { sha256, root }),
                None => self.extract_private(package, sha256),
            };
        }

        let lock = self.open_lock(&sha256)?;
        lock.lock_shared()?;
        if let Some(root) = read_complete_marker(&entry, &sha256) {
            self.keep_in_use(lock_path, lock);
            return Ok(ExtractedPackage { sha256, root });
        }
        // Missing or incomplete: extract while holding the exclusive lock.
        lock.unlock()?;
        match lock_exclusive_with_timeout(&lock, EXCLUSIVE_LOCK_TIMEOUT) {
            Ok(()) => {
                let result = match read_complete_marker(&entry, &sha256) {
                    // Another process finished the extraction while we waited.
                    Some(root) => Ok(root),
                    None => self.extract_entry(package, &sha256, &entry),
                };
                lock.unlock()?;
                result?;
            }
            Err(error) => {
                // Another process keeps a damaged entry locked. Extract a
                // private copy instead of waiting for it to exit.
                warn!(
                    "Plugin cache entry {} is locked ({error}); extracting a private copy",
                    entry.display()
                );
                return self.extract_private(package, sha256);
            }
        }
        lock.lock_shared()?;
        // A collector may have run between `unlock` and `lock_shared`; the
        // grace period makes this practically impossible, but never hand out
        // an entry that is not complete.
        match read_complete_marker(&entry, &sha256) {
            Some(root) => {
                self.keep_in_use(lock_path, lock);
                Ok(ExtractedPackage { sha256, root })
            }
            None => {
                lock.unlock()?;
                Err(PluginError::Other(format!(
                    "plugin cache entry {} disappeared during extraction; retry the operation",
                    entry.display()
                )))
            }
        }
    }

    /// Records that `new_path` now holds the content previously indexed for
    /// `old_path` (a rename), so the next access does not re-hash it.
    pub(crate) fn record_rename(&self, old_path: &Path, new_path: &Path) {
        let (Some(old_key), Some(new_key)) = (index_key(old_path), index_key(new_path)) else {
            return;
        };
        let Ok(metadata) = fs::metadata(new_path) else {
            return;
        };
        let identity = identity_of(&metadata);
        let sha256 = {
            let index = self.load_index();
            index
                .records
                .iter()
                .find(|record| record.path == old_key)
                .map(|record| record.sha256.clone())
        };
        if let Some(sha256) = sha256 {
            self.update_index(|index| {
                index.records.retain(|record| record.path != old_key);
                upsert(index, &new_key, identity, &sha256);
            });
        }
    }

    /// Forgets the index record of a removed or replaced package. The
    /// extraction itself is left for the collector, which knows whether a
    /// running instance still uses it.
    pub(crate) fn forget(&self, package: &Path) {
        if let Some(key) = index_key(package) {
            self.update_index(|index| index.records.retain(|record| record.path != key));
        }
    }

    /// Removes extractions that no existing package references, that are
    /// older than `grace`, and that no running process has locked.
    pub(crate) fn collect_garbage(&self, grace: Duration) -> PluginResult<GcReport> {
        let mut report = GcReport::default();
        if !self.root.is_dir() {
            return Ok(report);
        }
        let gc_lock = open_lock_file(&self.root.join(GC_LOCK))?;
        if gc_lock.try_lock().is_err() {
            debug!("Another process is collecting the plugin cache; skipping");
            return Ok(report);
        }

        // Drop records whose package is gone or changed, then collect the
        // set of content hashes that are still referenced.
        let mut referenced = HashSet::new();
        self.update_index(|index| {
            index.records.retain(|record| {
                fs::metadata(&record.path)
                    .ok()
                    .filter(|metadata| identity_of(metadata) == record.identity)
                    .is_some()
            });
            for record in &index.records {
                referenced.insert(record.sha256.clone());
            }
        });

        let now = SystemTime::now();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let path = entry.path();
            if !entry.file_type()?.is_dir() || name == LOCKS_DIR {
                continue;
            }
            let age = entry_age(&path, now);
            if is_content_key(name) {
                if referenced.contains(name) {
                    report.kept_referenced += 1;
                    continue;
                }
                if age < grace {
                    report.kept_recent += 1;
                    continue;
                }
                // Lock files are never deleted: removing one while another
                // process has it open would let two processes "own" the
                // same entry. They are empty, so keeping them is free.
                let lock = self.open_lock(name)?;
                if lock.try_lock().is_err() {
                    report.kept_in_use += 1;
                    continue;
                }
                if remove_tree(&self.root, &path).is_ok() {
                    report.removed.push(path);
                }
                let _ = lock.unlock();
            } else if name.starts_with(PRIVATE_PREFIX) {
                // A private extraction is in use while its creator lives.
                let lock = open_lock_file(&self.root.join(LOCKS_DIR).join(format!("{name}.lock")))?;
                if age < grace || lock.try_lock().is_err() {
                    report.kept_in_use += 1;
                    continue;
                }
                if remove_tree(&self.root, &path).is_ok() {
                    report.removed.push(path);
                }
                let _ = lock.unlock();
            } else if is_scratch_dir(name) {
                if age >= grace && remove_tree(&self.root, &path).is_ok() {
                    report.removed.push(path);
                }
            } else if age >= grace.max(LEGACY_ENTRY_GRACE) {
                // Legacy pre-1.0 extraction or an unknown leftover.
                if remove_tree(&self.root, &path).is_ok() {
                    report.removed.push(path);
                }
            } else {
                report.kept_recent += 1;
            }
        }
        let _ = gc_lock.unlock();
        Ok(report)
    }

    fn entry_dir(&self, sha256: &str) -> PathBuf {
        self.root.join(sha256)
    }

    fn lock_path(&self, sha256: &str) -> PathBuf {
        self.root.join(LOCKS_DIR).join(format!("{sha256}.lock"))
    }

    fn open_lock(&self, sha256: &str) -> io::Result<File> {
        open_lock_file(&self.lock_path(sha256))
    }

    fn keep_in_use(&self, lock_path: PathBuf, lock: File) {
        if let Ok(mut locks) = in_use_locks().lock() {
            locks.entry(lock_path).or_insert(lock);
        }
    }

    fn extract_entry(&self, package: &Path, sha256: &str, entry: &Path) -> PluginResult<PathBuf> {
        fs::create_dir_all(&self.root)?;
        let staging = tempfile::Builder::new()
            .prefix(&format!(".extract-{}-", &sha256[..12]))
            .tempdir_in(&self.root)?;
        extract_package(package, staging.path())?;
        let relative_root = find_manifest_root_relative(staging.path())?;
        write_complete_marker(staging.path(), sha256, &relative_root)?;
        // A stale, incomplete entry may exist; we hold the exclusive lock.
        if entry.exists() {
            remove_tree(&self.root, entry)?;
        }
        let staging_path = staging.keep();
        if let Err(error) = fs::rename(&staging_path, entry) {
            let _ = fs::remove_dir_all(&staging_path);
            return Err(error.into());
        }
        Ok(join_root(entry, &relative_root))
    }

    fn extract_private(&self, package: &Path, sha256: String) -> PluginResult<ExtractedPackage> {
        fs::create_dir_all(&self.root)?;
        let private = tempfile::Builder::new()
            .prefix(&format!("{PRIVATE_PREFIX}{}-", &sha256[..12]))
            .tempdir_in(&self.root)?
            .keep();
        let name = private
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_string();
        let lock_path = self.root.join(LOCKS_DIR).join(format!("{name}.lock"));
        let lock = open_lock_file(&lock_path)?;
        lock.lock_shared()?;
        let relative_root = match extract_package(package, &private)
            .and_then(|()| find_manifest_root_relative(&private))
        {
            Ok(relative_root) => relative_root,
            Err(error) => {
                let _ = fs::remove_dir_all(&private);
                return Err(error);
            }
        };
        self.keep_in_use(lock_path, lock);
        Ok(ExtractedPackage {
            root: join_root(&private, &relative_root),
            sha256,
        })
    }

    fn lookup(&self, key: &str, identity: &FileIdentity) -> Option<String> {
        self.load_index()
            .records
            .into_iter()
            .find(|record| record.path == key && &record.identity == identity)
            .map(|record| record.sha256)
    }

    fn record(&self, key: &str, identity: FileIdentity, sha256: &str) {
        self.update_index(|index| upsert(index, key, identity, sha256));
    }

    fn load_index(&self) -> Index {
        match fs::read_to_string(self.root.join(INDEX_FILE)) {
            Ok(text) => match serde_json::from_str::<Index>(&text) {
                Ok(index) if index.format == CACHE_FORMAT => index,
                Ok(_) | Err(_) => Index::default(),
            },
            Err(_) => Index::default(),
        }
    }

    /// Read-modify-write of the index under an exclusive lock. Failures are
    /// logged and ignored: the index is only an optimisation.
    fn update_index(&self, change: impl FnOnce(&mut Index)) {
        let result = (|| -> io::Result<()> {
            fs::create_dir_all(&self.root)?;
            let lock = open_lock_file(&self.root.join(INDEX_LOCK))?;
            lock.lock()?;
            let mut index = self.load_index();
            change(&mut index);
            if index.records.len() > MAX_INDEX_RECORDS {
                index
                    .records
                    .sort_by_key(|record| std::cmp::Reverse(record.last_seen));
                index.records.truncate(MAX_INDEX_RECORDS);
            }
            let text = serde_json::to_string_pretty(&index).map_err(io::Error::other)?;
            let mut temporary = tempfile::Builder::new()
                .prefix(".index-")
                .suffix(".tmp")
                .tempfile_in(&self.root)?;
            temporary.write_all(text.as_bytes())?;
            temporary.as_file().sync_all()?;
            temporary
                .persist(self.root.join(INDEX_FILE))
                .map_err(|error| error.error)?;
            lock.unlock()
        })();
        if let Err(error) = result {
            debug!(
                "Could not update the plugin package index in {}: {error}",
                self.root.display()
            );
        }
    }
}

fn upsert(index: &mut Index, key: &str, identity: FileIdentity, sha256: &str) {
    let now = unix_now();
    index.records.retain(|record| record.path != key);
    index.records.push(IndexRecord {
        path: key.to_string(),
        identity,
        sha256: sha256.to_string(),
        last_seen: now,
    });
}

/// Extracts the default-cache entry for `package`.
pub(crate) fn ensure_extracted(package: &Path) -> PluginResult<ExtractedPackage> {
    PackageCache::default().ensure_extracted(package)
}

/// Starts one background garbage-collection pass per process. Called when
/// the GUI discovers plugins at startup; CLI commands do not collect.
pub(crate) fn schedule_startup_gc() {
    #[cfg(not(test))]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let spawned = std::thread::Builder::new()
                .name("eov-plugin-cache-gc".into())
                .spawn(|| {
                    // Let startup I/O (extraction, library loading) go first.
                    std::thread::sleep(Duration::from_secs(15));
                    match PackageCache::default().collect_garbage(DEFAULT_GC_GRACE) {
                        Ok(report) if !report.removed.is_empty() => tracing::info!(
                            "Removed {} unused plugin package extraction(s)",
                            report.removed.len()
                        ),
                        Ok(_) => {}
                        Err(error) => debug!("Plugin cache garbage collection failed: {error}"),
                    }
                });
            if let Err(error) = spawned {
                debug!("Could not start plugin cache garbage collection: {error}");
            }
        });
    }
}

/// Detects whether a package is a plain or zstd-compressed tar archive.
pub(crate) fn package_format(package: &Path) -> io::Result<PackageFormat> {
    let mut magic = [0u8; 4];
    let mut file = File::open(package)?;
    let mut read = 0;
    while read < magic.len() {
        let n = file.read(&mut magic[read..])?;
        if n == 0 {
            break;
        }
        read += n;
    }
    Ok(if read == magic.len() && magic == ZSTD_MAGIC {
        PackageFormat::ZstdTar
    } else {
        PackageFormat::Tar
    })
}

fn open_package_reader(package: &Path) -> PluginResult<Box<dyn Read>> {
    let file = BufReader::with_capacity(1 << 20, File::open(package)?);
    Ok(match package_format(package)? {
        PackageFormat::Tar => Box::new(file),
        PackageFormat::ZstdTar => Box::new(
            zstd::stream::read::Decoder::with_buffer(file).map_err(|error| {
                PluginError::Other(format!(
                    "plugin package '{}' is not a valid zstd stream: {error}",
                    package.display()
                ))
            })?,
        ),
    })
}

/// Extracts a package archive into `destination` (which must be empty).
pub(crate) fn extract_package(package: &Path, destination: &Path) -> PluginResult<()> {
    let mut archive = tar::Archive::new(open_package_reader(package)?);
    archive.set_mask(0o022);
    archive.set_preserve_permissions(false);
    archive.set_unpack_xattrs(false);
    archive.set_overwrite(false);
    let mut manifest_paths = HashSet::new();
    let mut entries = 0u64;
    let mut total_bytes = 0u64;

    for entry in archive.entries()? {
        let mut entry = entry.map_err(|error| {
            PluginError::Other(format!(
                "plugin package '{}' is not a valid archive: {error}",
                package.display()
            ))
        })?;
        entries += 1;
        if entries > MAX_ARCHIVE_ENTRIES {
            return Err(PluginError::Other(format!(
                "plugin package '{}' has more than {MAX_ARCHIVE_ENTRIES} entries",
                package.display()
            )));
        }
        let archive_path = entry.path()?.into_owned();
        validate_archive_path(package, &archive_path)?;
        let entry_type = entry.header().entry_type();
        if !(entry_type.is_file() || entry_type.is_dir()) {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains unsupported archive entry '{}'",
                package.display(),
                archive_path.display()
            )));
        }
        total_bytes = total_bytes.saturating_add(entry.size());
        if total_bytes > MAX_EXTRACTED_BYTES {
            return Err(PluginError::Other(format!(
                "plugin package '{}' expands to more than {} GiB",
                package.display(),
                MAX_EXTRACTED_BYTES >> 30
            )));
        }
        let normalized = normalize_archive_path(&archive_path);
        if normalized.file_name() == Some(OsStr::new(eov_plugin_api::manifest::MANIFEST_FILENAME))
            && entry_type.is_file()
            && !manifest_paths.insert(normalized)
        {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains duplicate {} entries",
                package.display(),
                eov_plugin_api::manifest::MANIFEST_FILENAME
            )));
        }
        let unpacked = entry.unpack_in(destination).map_err(|error| {
            PluginError::Other(format!(
                "could not extract '{}' from plugin package '{}': {error}",
                archive_path.display(),
                package.display()
            ))
        })?;
        if !unpacked {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains an invalid path '{}'",
                package.display(),
                archive_path.display()
            )));
        }
    }
    Ok(())
}

fn normalize_archive_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, Component::CurDir))
        .collect()
}

/// Rejects archive paths that could escape the extraction directory.
pub(crate) fn validate_archive_path(package_path: &Path, archive_path: &Path) -> PluginResult<()> {
    if archive_path.as_os_str().is_empty() {
        return Err(PluginError::Other(format!(
            "plugin package '{}' contains an empty path",
            package_path.display()
        )));
    }

    let raw_path = archive_path.to_string_lossy();
    let has_windows_prefix = raw_path.len() >= 2 && raw_path.as_bytes().get(1) == Some(&b':')
        || raw_path.starts_with("\\\\")
        || raw_path.starts_with('\\')
        || raw_path.contains(':');
    if has_windows_prefix {
        return Err(PluginError::Other(format!(
            "plugin package '{}' contains an unsafe Windows path '{}'",
            package_path.display(),
            archive_path.display()
        )));
    }

    for component in archive_path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(PluginError::Other(format!(
                    "plugin package '{}' contains an unsafe path '{}'",
                    package_path.display(),
                    archive_path.display()
                )));
            }
        }
    }

    Ok(())
}

/// SHA-256 of a file, streamed.
pub(crate) fn sha256_file(path: &Path) -> PluginResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn join_root(base: &Path, relative_root: &Path) -> PathBuf {
    if relative_root.as_os_str().is_empty() {
        base.to_path_buf()
    } else {
        base.join(relative_root)
    }
}

fn find_manifest_root_relative(extracted_dir: &Path) -> PluginResult<PathBuf> {
    let mut dirs_to_visit = vec![extracted_dir.to_path_buf()];
    let mut manifest_roots = Vec::new();

    while let Some(dir) = dirs_to_visit.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                dirs_to_visit.push(path);
                continue;
            }

            if path.file_name() == Some(OsStr::new(eov_plugin_api::manifest::MANIFEST_FILENAME)) {
                manifest_roots.push(
                    path.parent()
                        .unwrap_or(extracted_dir)
                        .strip_prefix(extracted_dir)
                        .map(Path::to_path_buf)
                        .map_err(|err| {
                            PluginError::Other(format!(
                                "failed to resolve plugin manifest root in {}: {err}",
                                extracted_dir.display()
                            ))
                        })?,
                );
            }
        }
    }

    match manifest_roots.len() {
        0 => Err(PluginError::Other(format!(
            "plugin package extraction '{}' does not contain {}",
            extracted_dir.display(),
            eov_plugin_api::manifest::MANIFEST_FILENAME
        ))),
        1 => Ok(manifest_roots.pop().unwrap_or_default()),
        _ => Err(PluginError::Other(format!(
            "plugin package extraction '{}' contains multiple {} files",
            extracted_dir.display(),
            eov_plugin_api::manifest::MANIFEST_FILENAME
        ))),
    }
}

fn write_complete_marker(entry: &Path, sha256: &str, relative_root: &Path) -> PluginResult<()> {
    let root = relative_root
        .to_str()
        .ok_or_else(|| PluginError::Other("plugin manifest root is not valid UTF-8".into()))?
        .replace('\\', "/");
    let marker = CompleteMarker {
        format: CACHE_FORMAT,
        sha256: sha256.to_string(),
        root,
        extracted_at: unix_now(),
    };
    let text = serde_json::to_string_pretty(&marker)
        .map_err(|error| PluginError::Other(error.to_string()))?;
    let mut file = File::create(entry.join(COMPLETE_MARKER))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

/// Returns the manifest root of a complete entry, or `None` when the entry
/// is missing, incomplete, or does not belong to `sha256`.
fn read_complete_marker(entry: &Path, sha256: &str) -> Option<PathBuf> {
    let text = fs::read_to_string(entry.join(COMPLETE_MARKER)).ok()?;
    let marker: CompleteMarker = serde_json::from_str(&text).ok()?;
    if marker.format != CACHE_FORMAT || marker.sha256 != sha256 {
        return None;
    }
    let relative = Path::new(&marker.root);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return None;
    }
    let root = join_root(entry, relative);
    root.join(eov_plugin_api::manifest::MANIFEST_FILENAME)
        .is_file()
        .then_some(root)
}

fn open_lock_file(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

fn lock_exclusive_with_timeout(file: &File, timeout: Duration) -> io::Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    let mut delay = Duration::from_millis(20);
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(()),
            Err(std::fs::TryLockError::WouldBlock) => {
                if std::time::Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "timed out waiting for the plugin cache lock",
                    ));
                }
                std::thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_millis(500));
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error),
        }
    }
}

/// Renames `path` out of the way before deleting it, so an interrupted
/// deletion never leaves something that looks like a valid entry.
fn remove_tree(cache_root: &Path, path: &Path) -> io::Result<()> {
    let trash = cache_root.join(format!(
        ".trash-{}-{}",
        std::process::id(),
        unix_now_nanos()
    ));
    match fs::rename(path, &trash) {
        Ok(()) => fs::remove_dir_all(&trash),
        Err(_) => fs::remove_dir_all(path),
    }
}

fn is_content_key(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_scratch_dir(name: &str) -> bool {
    name.starts_with(".extract-") || name.starts_with(".trash-") || name.contains(".tmp-")
}

fn entry_age(path: &Path, now: SystemTime) -> Duration {
    let modified = fs::metadata(path.join(COMPLETE_MARKER))
        .or_else(|_| fs::metadata(path))
        .and_then(|metadata| metadata.modified())
        .unwrap_or(UNIX_EPOCH);
    now.duration_since(modified).unwrap_or_default()
}

/// The stable index key of a package path: its canonical parent directory
/// joined with its file name, which also works after the file was removed.
fn index_key(path: &Path) -> Option<String> {
    let absolute = std::path::absolute(path).ok()?;
    let file_name = absolute.file_name()?;
    let parent = absolute.parent()?;
    let parent = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    parent.join(file_name).to_str().map(str::to_string)
}

fn identity_of(metadata: &fs::Metadata) -> FileIdentity {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    FileIdentity {
        len: metadata.len(),
        modified_secs: modified.as_secs(),
        modified_nanos: modified.subsec_nanos(),
        file_id: file_id(metadata),
    }
}

#[cfg(unix)]
fn file_id(metadata: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.ino() ^ metadata.dev().rotate_left(32)
}

#[cfg(not(unix))]
fn file_id(metadata: &fs::Metadata) -> u64 {
    metadata
        .created()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn unix_now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod test_packages {
    //! Builders for package archives used by tests across the crate.

    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// A minimal valid manifest for `id`/`version` with the given EOV range.
    pub(crate) fn manifest(id: &str, version: &str, environment: &str) -> String {
        format!(
            "id = \"{id}\"\nname = \"{}\"\nversion = \"{version}\"\ndescription = \"Test package\"\nrepository = \"https://github.com/example/{id}\"\n\n[environment]\nversion = \"{environment}\"\n",
            crate::update::manifest::title_case_id(id)
        )
    }

    /// Builds a tar (optionally zstd-compressed) from `(path, contents)` pairs.
    pub(crate) fn write_archive(path: &Path, files: &[(&str, &[u8])], zstd: bool) -> PathBuf {
        let mut tar_bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_bytes);
            for (name, contents) in files {
                let mut header = tar::Header::new_gnu();
                header.set_entry_type(tar::EntryType::file());
                header.set_size(contents.len() as u64);
                header.set_mode(0o666);
                header.set_cksum();
                builder.append_data(&mut header, name, *contents).unwrap();
            }
            builder.finish().unwrap();
        }
        let bytes = if zstd {
            zstd::stream::encode_all(&tar_bytes[..], 3).unwrap()
        } else {
            tar_bytes
        };
        let mut file = File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
        path.to_path_buf()
    }

    /// Writes a package containing only a manifest.
    pub(crate) fn write_package(path: &Path, id: &str, version: &str) -> PathBuf {
        let manifest = manifest(id, version, ">=1.0.0, <2.0.0");
        write_archive(path, &[("plugin.toml", manifest.as_bytes())], false)
    }
}

#[cfg(test)]
mod tests {
    use super::test_packages::{manifest, write_archive};
    use super::*;

    fn cache_in(dir: &Path) -> PackageCache {
        PackageCache::at(dir.join("cache"))
    }

    #[test]
    fn plain_and_zstd_packages_extract_to_the_same_layout() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let manifest = manifest("example", "1.0.0", ">=1.0.0, <2.0.0");
        let files: &[(&str, &[u8])] = &[
            ("bundle/plugin.toml", manifest.as_bytes()),
            ("bundle/ui/panel.slint", b"export component Panel {}"),
        ];
        let plain = write_archive(&temp.path().join("plain.eop"), files, false);
        let compressed = write_archive(&temp.path().join("compressed.eop"), files, true);
        assert_eq!(package_format(&plain).unwrap(), PackageFormat::Tar);
        assert_eq!(package_format(&compressed).unwrap(), PackageFormat::ZstdTar);

        for package in [&plain, &compressed] {
            let extracted = cache.ensure_extracted(package).unwrap();
            assert!(extracted.root.ends_with("bundle"));
            assert!(extracted.root.join("ui/panel.slint").is_file());
            assert_eq!(extracted.sha256, sha256_file(package).unwrap());
        }
    }

    #[test]
    fn identical_packages_share_one_extraction() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let manifest = manifest("example", "1.0.0", ">=1.0.0, <2.0.0");
        let files: &[(&str, &[u8])] = &[("plugin.toml", manifest.as_bytes())];
        let first = write_archive(&temp.path().join("a.eop"), files, false);
        let second = write_archive(&temp.path().join("b.eop"), files, false);
        let a = cache.ensure_extracted(&first).unwrap();
        let b = cache.ensure_extracted(&second).unwrap();
        assert_eq!(a.root, b.root);
    }

    #[test]
    fn extraction_masks_group_and_world_write_permissions() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let manifest = manifest("example", "1.0.0", ">=1.0.0, <2.0.0");
        let package = write_archive(
            &temp.path().join("perm.eop"),
            &[("plugin.toml", manifest.as_bytes()), ("data.bin", b"x")],
            false,
        );
        let extracted = cache.ensure_extracted(&package).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(extracted.root.join("data.bin"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o022, 0, "mode {mode:o} is group/world writable");
        }
        #[cfg(not(unix))]
        assert!(extracted.root.join("data.bin").is_file());
    }

    #[test]
    fn rejects_unsafe_archive_paths() {
        let temp = tempfile::tempdir().unwrap();
        for path in [
            "../escape",
            "/escape",
            "C:evil",
            "C:\\evil",
            "\\\\server\\share",
            "a:b",
        ] {
            let error = validate_archive_path(temp.path(), Path::new(path)).unwrap_err();
            assert!(error.to_string().contains("unsafe"), "{path}: {error}");
        }
        validate_archive_path(temp.path(), Path::new("./ui/panel.slint")).unwrap();
    }

    #[test]
    fn duplicate_manifests_are_rejected_even_with_dot_prefixes() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let manifest = manifest("example", "1.0.0", ">=1.0.0, <2.0.0");
        let package = write_archive(
            &temp.path().join("dup.eop"),
            &[
                ("plugin.toml", manifest.as_bytes()),
                ("./plugin.toml", manifest.as_bytes()),
            ],
            false,
        );
        assert!(cache.ensure_extracted(&package).is_err());
        // The failed extraction left no staging directory behind.
        let leftovers: Vec<_> = fs::read_dir(cache.root())
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(".extract-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn unchanged_packages_are_not_rehashed() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let package = super::test_packages::write_package(
            &temp.path().join("example.eop"),
            "example",
            "1.0.0",
        );
        let first = cache.sha256(&package).unwrap();
        // Corrupt the index entry: a lookup that trusts the index returns it.
        let mut index = cache.load_index();
        index.records[0].sha256 = "f".repeat(64);
        fs::write(
            cache.root().join(INDEX_FILE),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        assert_eq!(cache.sha256(&package).unwrap(), "f".repeat(64));
        // Rewriting the package changes its identity and forces a re-hash.
        std::thread::sleep(Duration::from_millis(20));
        super::test_packages::write_package(&package, "example", "1.0.1");
        let second = cache.sha256(&package).unwrap();
        assert_ne!(second, first);
        assert_eq!(second, sha256_file(&package).unwrap());
    }

    #[test]
    fn garbage_collection_keeps_referenced_recent_and_locked_entries() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let kept =
            super::test_packages::write_package(&temp.path().join("kept.eop"), "kept", "1.0.0");
        let removed =
            super::test_packages::write_package(&temp.path().join("gone.eop"), "gone", "1.0.0");
        let in_use =
            super::test_packages::write_package(&temp.path().join("busy.eop"), "busy", "1.0.0");
        let kept_entry = cache.ensure_extracted(&kept).unwrap();
        let removed_entry = cache.ensure_extracted(&removed).unwrap();
        let busy_entry = cache.ensure_extracted(&in_use).unwrap();

        // This process holds shared locks on every entry it extracted; drop
        // them to simulate another process, except for the busy entry.
        let busy_lock = cache.lock_path(&busy_entry.sha256);
        let ours = |path: &Path| path.starts_with(cache.root());
        in_use_locks()
            .lock()
            .unwrap()
            .retain(|path, _| !ours(path) || path == &busy_lock);

        fs::remove_file(&removed).unwrap();
        fs::remove_file(&in_use).unwrap();

        // A recent unreferenced entry survives a collection with a grace period.
        let report = cache.collect_garbage(Duration::from_secs(3600)).unwrap();
        assert!(report.removed.is_empty());
        assert!(removed_entry.root.exists());

        let report = cache.collect_garbage(Duration::ZERO).unwrap();
        assert!(kept_entry.root.exists(), "referenced entry was removed");
        assert!(!removed_entry.root.exists(), "unreferenced entry survived");
        assert!(busy_entry.root.exists(), "locked entry was removed");
        assert_eq!(report.kept_in_use, 1);
        assert_eq!(report.kept_referenced, 1);
        in_use_locks()
            .lock()
            .unwrap()
            .retain(|path, _| path != &busy_lock);
        let report = cache.collect_garbage(Duration::ZERO).unwrap();
        assert_eq!(report.removed.len(), 1);
        assert!(!busy_entry.root.exists());
    }

    #[test]
    fn garbage_collection_removes_stale_legacy_and_scratch_directories() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        fs::create_dir_all(
            cache
                .root()
                .join("gamepad-123-1700000000-0-0123456789abcdef"),
        )
        .unwrap();
        fs::create_dir_all(cache.root().join(".extract-abc")).unwrap();
        // Fresh legacy entries are kept (an old eov may still use them).
        let report = cache.collect_garbage(Duration::ZERO).unwrap();
        assert_eq!(report.removed.len(), 1, "{report:?}");
        assert!(
            cache
                .root()
                .join("gamepad-123-1700000000-0-0123456789abcdef")
                .exists()
        );
    }

    #[test]
    fn forget_and_rename_keep_the_index_exact() {
        let temp = tempfile::tempdir().unwrap();
        let cache = cache_in(temp.path());
        let staged = super::test_packages::write_package(
            &temp.path().join(".stage.part"),
            "example",
            "1.0.0",
        );
        let sha = cache.sha256(&staged).unwrap();
        let installed = temp.path().join("example.eop");
        fs::rename(&staged, &installed).unwrap();
        cache.record_rename(&staged, &installed);
        let index = cache.load_index();
        assert_eq!(index.records.len(), 1);
        assert!(index.records[0].path.ends_with("example.eop"));
        assert_eq!(index.records[0].sha256, sha);
        cache.forget(&installed);
        assert!(cache.load_index().records.is_empty());
    }
}
