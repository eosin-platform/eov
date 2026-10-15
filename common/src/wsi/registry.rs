//! Process-wide sharing of slide handles and of one OpenSlide tile cache.
//!
//! Opening a slide re-parses its metadata (expensive for MIRAX and DICOM)
//! and, by default, gives every OpenSlide handle a private ~32 MiB tile cache
//! that no budget accounts for. eov therefore:
//!
//! * shares one handle per slide: [`WsiFile::open`](crate::WsiFile::open)
//!   returns a clone of an already open slide when the same file (same
//!   canonical path, size and modification time) is still open anywhere in
//!   the process. OpenSlide handles are thread-safe, so all threads can read
//!   through the shared handle concurrently;
//! * attaches every handle to one global `openslide_cache_t`, sized with
//!   [`set_openslide_cache_capacity`], which also sizes the decoded-tile
//!   cache of eov's own OME-TIFF reader.

use super::SharedSlide;
use super::health::HandleCell;
use crate::openslide::{OpenSlide, OpenSlideCache};
use crate::{Error, Result};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Weak};
use std::time::SystemTime;

/// Default capacity of the global OpenSlide tile cache (64 MiB).
pub const DEFAULT_OPENSLIDE_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Suggested OpenSlide cache size for a given eov tile-cache budget.
///
/// eov caches its own (bordered, resampled) tiles; OpenSlide's cache only
/// needs to hold the native tiles shared by neighbouring eov tiles and the
/// working set of synthetic-level reads, so one eighth of the tile budget,
/// clamped to 32-512 MiB, is plenty.
pub fn recommended_openslide_cache_bytes(tile_cache_bytes: usize) -> usize {
    (tile_cache_bytes / 8).clamp(32 * 1024 * 1024, 512 * 1024 * 1024)
}

struct GlobalCache {
    cache: Option<Arc<OpenSlideCache>>,
    capacity: usize,
}

static GLOBAL_CACHE: LazyLock<Mutex<GlobalCache>> = LazyLock::new(|| {
    Mutex::new(GlobalCache {
        cache: None,
        capacity: DEFAULT_OPENSLIDE_CACHE_BYTES,
    })
});

/// Every live shared slide, for handle reuse.
static SLIDES: LazyLock<Mutex<Vec<Weak<SharedSlide>>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Every live OpenSlide handle cell, to move them to a new global cache.
static OPENSLIDE_HANDLES: LazyLock<Mutex<Vec<Weak<HandleCell>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Remembers an OpenSlide handle cell for [`set_openslide_cache_capacity`].
pub(crate) fn register_openslide(handle: &Arc<HandleCell>) {
    let mut handles = OPENSLIDE_HANDLES.lock();
    handles.retain(|weak| weak.strong_count() > 0);
    handles.push(Arc::downgrade(handle));
}

fn global_cache() -> Option<Arc<OpenSlideCache>> {
    let mut global = GLOBAL_CACHE.lock();
    if global.cache.is_none() {
        match OpenSlideCache::new(global.capacity) {
            Ok(cache) => global.cache = Some(Arc::new(cache)),
            Err(err) => tracing::warn!("could not create the OpenSlide cache: {err}"),
        }
    }
    global.cache.clone()
}

/// Attaches the global cache to a newly opened handle.
pub(crate) fn attach_global_cache(slide: &OpenSlide) {
    if let Some(cache) = global_cache() {
        slide.set_cache(&cache);
    }
}

/// Sets the capacity of the process-wide OpenSlide tile cache shared by all
/// slides, replacing the current cache. Open slides are moved to the new
/// cache immediately. The OME-TIFF reader's cache of decoded tiles follows
/// (twice this, as raw multichannel samples take more room, within
/// 128 MiB-1 GiB).
///
/// Call this at startup with a value derived from the user's memory budget,
/// e.g. [`recommended_openslide_cache_bytes`].
pub fn set_openslide_cache_capacity(capacity_bytes: usize) -> Result<()> {
    let cache = Arc::new(OpenSlideCache::new(capacity_bytes)?);
    {
        let mut global = GLOBAL_CACHE.lock();
        global.capacity = capacity_bytes;
        global.cache = Some(Arc::clone(&cache));
    }
    let handles: Vec<Arc<HandleCell>> = {
        let mut handles = OPENSLIDE_HANDLES.lock();
        handles.retain(|weak| weak.strong_count() > 0);
        handles.iter().filter_map(Weak::upgrade).collect()
    };
    for handle in handles {
        handle.current().slide.set_cache(&cache);
    }
    crate::ome_tiff::cache::CHUNKS
        .set_capacity(crate::ome_tiff::cache::capacity_for(capacity_bytes));
    Ok(())
}

/// The capacity of the process-wide OpenSlide tile cache.
pub fn openslide_cache_capacity() -> usize {
    GLOBAL_CACHE.lock().capacity
}

/// Identity of the file behind a path, used to decide whether an open slide
/// can be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
}

impl FileStamp {
    pub(crate) fn of(path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            dev: std::os::unix::fs::MetadataExt::dev(&metadata),
            #[cfg(unix)]
            ino: std::os::unix::fs::MetadataExt::ino(&metadata),
        })
    }
}

pub(crate) fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn live_slides() -> Vec<Arc<SharedSlide>> {
    let mut slides = SLIDES.lock();
    slides.retain(|weak| weak.strong_count() > 0);
    slides.iter().filter_map(Weak::upgrade).collect()
}

/// Returns an open, shareable slide for `canonical_path` whose file still
/// matches `stamp`.
pub(crate) fn find_shared(canonical_path: &Path, stamp: &FileStamp) -> Option<Arc<SharedSlide>> {
    live_slides().into_iter().find(|slide| {
        slide.shareable && slide.canonical_path == canonical_path && slide.stamp == *stamp
    })
}

/// Registers a newly opened slide. If another thread registered an equal
/// shareable slide meanwhile, that one is returned instead.
pub(crate) fn register(slide: Arc<SharedSlide>) -> Arc<SharedSlide> {
    let mut slides = SLIDES.lock();
    slides.retain(|weak| weak.strong_count() > 0);
    if slide.shareable
        && let Some(existing) = slides.iter().filter_map(Weak::upgrade).find(|other| {
            other.shareable
                && other.canonical_path == slide.canonical_path
                && other.stamp == slide.stamp
        })
    {
        return existing;
    }
    slides.push(Arc::downgrade(&slide));
    slide
}

/// Number of distinct open slides (for tests and diagnostics).
pub fn open_slide_count() -> usize {
    live_slides().len()
}

pub(crate) fn not_found(path: &Path) -> Error {
    Error::FileNotFound(path.display().to_string())
}
