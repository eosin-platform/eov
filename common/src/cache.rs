//! Tile caching system for WSI rendering
//!
//! [`TileCache`] is a concurrent, byte-budgeted cache of decoded tiles with
//! least-recently-*used* eviction:
//!
//! * **Every lookup counts as a use.** [`get`](TileCache::get),
//!   [`peek`](TileCache::peek) and [`touch`](TileCache::touch) store the
//!   cache's current clock value in the entry (one relaxed atomic store under
//!   a shard read lock), so tiles the renderer draws every frame stay
//!   resident and eviction order is by last use, not by insertion.
//!   [`peek_untracked`](TileCache::peek_untracked) reads without touching.
//! * **Frames.** A renderer that calls [`begin_frame`](TileCache::begin_frame)
//!   once per frame protects every tile used (or inserted) during the
//!   current frame from eviction, so a working set larger than the budget
//!   overshoots temporarily instead of thrashing. Protection is bounded by a
//!   hard limit of twice the budget.
//! * **Pinning.** [`set_pinned_levels`](TileCache::set_pinned_levels) keeps
//!   the coarse fallback levels of a file resident; [`pin`](TileCache::pin)
//!   pins individual tiles.
//! * **Files.** Tiles of files marked hidden with
//!   [`set_file_visible`](TileCache::set_file_visible) are evicted first, and
//!   [`remove_file`](TileCache::remove_file) drops a closed file's tiles at
//!   once.
//!
//! There is a single map and all counters are atomics, so the old
//! LRU/map desynchronisation and the `stats`/`current_bytes` lock-order
//! deadlock cannot happen. Nothing is preallocated, so a large `max_tiles`
//! costs nothing up front.

use crate::tile::{TileCoord, TileData};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, trace, warn};

/// Default maximum cache size in bytes (768 MiB).
///
/// About three times the tile working set of a 1080p pane; prefer
/// [`CacheBudget::for_display`] to size the cache for the actual display.
pub const DEFAULT_CACHE_SIZE_BYTES: usize = 768 * 1024 * 1024;

/// Default maximum number of tiles to cache.
pub const DEFAULT_MAX_TILES: usize = 4096;

/// Upper bound accepted for `max_tiles`.
pub const MAX_TILES_LIMIT: usize = 1 << 24;

/// Evict below the limits by this fraction (1/8) so eviction runs in
/// batches rather than on every insert.
const HEADROOM_DIVISOR: usize = 8;

/// Frame-protected tiles may push the cache to this multiple of the budget
/// before even they are evicted.
const HARD_LIMIT_FACTOR: usize = 2;

/// Minimum spacing of "working set exceeds the budget" warnings.
const OVERSHOOT_WARN_INTERVAL: Duration = Duration::from_secs(10);

/// Sentinel frame stamp: no frame has begun, nothing is protected.
const NO_FRAME: u64 = u64::MAX;

struct CacheEntry {
    tile: Arc<TileData>,
    size_bytes: usize,
    /// Cache clock at the last use.
    last_used: AtomicU64,
    /// Explicit pin count.
    pins: AtomicU32,
    /// Distinguishes a re-inserted entry from the one eviction inspected.
    id: u64,
}

/// Statistics for cache monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_tiles: usize,
    pub current_bytes: usize,
    /// Successful inserts.
    pub insertions: u64,
    /// Configured byte budget.
    pub max_bytes: usize,
    /// Configured tile-count limit.
    pub max_tiles: usize,
    /// Eviction passes run (diagnostic).
    pub eviction_scans: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// A cache budget, typically derived from the display with
/// [`CacheBudget::for_display`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBudget {
    pub max_bytes: usize,
    pub max_tiles: usize,
}

impl Default for CacheBudget {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_CACHE_SIZE_BYTES,
            max_tiles: DEFAULT_MAX_TILES,
        }
    }
}

impl CacheBudget {
    /// Smallest budget [`for_display`](Self::for_display) returns.
    pub const MIN_BYTES: usize = 256 * 1024 * 1024;
    /// Largest budget [`for_display`](Self::for_display) returns; clamp
    /// further to the machine's memory if needed.
    pub const MAX_BYTES: usize = 4 * 1024 * 1024 * 1024;

    /// Worst-case bytes of tiles one frame of a `width` x `height` pane
    /// (physical pixels) needs.
    ///
    /// With trilinear filtering on a 4x pyramid the finer level is sampled
    /// at up to 4 level pixels per screen pixel (16x the pane area) and the
    /// coarser one at up to 1x; tiles straddling the edges and the one-tile
    /// pan margin add a ring of 256-px tiles on every side. Tiles are RGBA.
    pub fn frame_working_set_bytes(width: u32, height: u32) -> u64 {
        const RING: u64 = 2 * 256;
        const LEVEL_FACTOR: u64 = 16 + 1;
        (u64::from(width) + RING) * (u64::from(height) + RING) * LEVEL_FACTOR * 4
    }

    /// A budget for `panes` panes of `width` x `height` physical pixels:
    /// two worst-case frames (about three typical ones), clamped to
    /// [`MIN_BYTES`](Self::MIN_BYTES)..=[`MAX_BYTES`](Self::MAX_BYTES).
    /// The tile limit is sized so that it never binds before the byte budget
    /// for 256-px tiles.
    pub fn for_display(width: u32, height: u32, panes: u32) -> Self {
        let frame = Self::frame_working_set_bytes(width, height);
        let total = frame
            .saturating_mul(2)
            .saturating_mul(u64::from(panes.max(1)));
        let max_bytes = total.clamp(Self::MIN_BYTES as u64, Self::MAX_BYTES as u64) as usize;
        // A 256-px tile with a 1-px border: 258 * 258 * 4 bytes.
        let max_tiles = (max_bytes / (258 * 258 * 4)).clamp(DEFAULT_MAX_TILES, MAX_TILES_LIMIT);
        Self {
            max_bytes,
            max_tiles,
        }
    }
}

#[derive(Default)]
struct Policy {
    /// Levels of a file that are never evicted (coarse fallbacks).
    pinned_levels: HashMap<i32, HashSet<u32>>,
    /// Files whose tiles are evicted before those of visible files.
    hidden_files: HashSet<i32>,
}

/// Thread-safe tile cache with LRU eviction; see the
/// [module documentation](self).
pub struct TileCache {
    map: DashMap<TileCoord, CacheEntry>,
    max_bytes: AtomicUsize,
    max_tiles: AtomicUsize,
    current_bytes: AtomicUsize,
    clock: AtomicU64,
    frame_start: AtomicU64,
    next_id: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    insertions: AtomicU64,
    eviction_scans: AtomicU64,
    policy: RwLock<Policy>,
    /// Bumped whenever pins or file visibility change.
    policy_version: AtomicU64,
    /// After a pass that could not get below the limits (everything left was
    /// protected or pinned), further passes are skipped until the cache
    /// grows past these marks or the frame or policy changes.
    futile_until_bytes: AtomicUsize,
    futile_until_tiles: AtomicUsize,
    futile_frame: AtomicU64,
    futile_policy: AtomicU64,
    evicting: Mutex<()>,
    last_overshoot_warning: Mutex<Option<Instant>>,
}

impl TileCache {
    /// Create a new tile cache with default settings
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_TILES, DEFAULT_CACHE_SIZE_BYTES)
    }

    /// Create a new tile cache with custom limits. Nothing is preallocated;
    /// `max_tiles` is clamped to [`MAX_TILES_LIMIT`].
    pub fn with_limits(max_tiles: usize, max_bytes: usize) -> Self {
        Self {
            map: DashMap::new(),
            max_bytes: AtomicUsize::new(max_bytes),
            max_tiles: AtomicUsize::new(max_tiles.clamp(1, MAX_TILES_LIMIT)),
            current_bytes: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            frame_start: AtomicU64::new(NO_FRAME),
            next_id: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            eviction_scans: AtomicU64::new(0),
            policy: RwLock::new(Policy::default()),
            policy_version: AtomicU64::new(0),
            futile_until_bytes: AtomicUsize::new(0),
            futile_until_tiles: AtomicUsize::new(0),
            futile_frame: AtomicU64::new(NO_FRAME),
            futile_policy: AtomicU64::new(0),
            evicting: Mutex::new(()),
            last_overshoot_warning: Mutex::new(None),
        }
    }

    /// Create a cache with a [`CacheBudget`].
    pub fn with_budget(budget: CacheBudget) -> Self {
        Self::with_limits(budget.max_tiles, budget.max_bytes)
    }

    /// Changes the limits at runtime (e.g. when the window is resized) and
    /// evicts down to them if needed.
    pub fn set_limits(&self, max_tiles: usize, max_bytes: usize) {
        self.max_tiles
            .store(max_tiles.clamp(1, MAX_TILES_LIMIT), Ordering::Relaxed);
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
        self.policy_changed();
        self.evict_if_needed();
    }

    fn policy_changed(&self) {
        self.policy_version.fetch_add(1, Ordering::Relaxed);
    }

    /// The byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes.load(Ordering::Relaxed)
    }

    /// The tile-count limit.
    pub fn max_tiles(&self) -> usize {
        self.max_tiles.load(Ordering::Relaxed)
    }

    /// Bytes currently cached.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Number of cached tiles.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline]
    fn now(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
    }

    /// Starts a new render frame and returns its stamp.
    ///
    /// Tiles used or inserted from now until the next `begin_frame` are
    /// protected from eviction. Call it once per rendered frame, before the
    /// frame's lookups, for all panes together. Caches that never call it
    /// behave as a plain LRU.
    pub fn begin_frame(&self) -> u64 {
        let stamp = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        self.frame_start.store(stamp, Ordering::Relaxed);
        stamp
    }

    /// Get a tile from the cache, counting a hit or miss and marking it as
    /// used.
    pub fn get(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        let tile = self.peek(coord);
        if tile.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            trace!("Cache hit for {:?}", coord);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            trace!("Cache miss for {:?}", coord);
        }
        tile
    }

    /// Read a tile and mark it as used, without updating statistics.
    ///
    /// The "use" is a single relaxed atomic store, so this is cheap enough
    /// for every render-loop lookup — and it must be used for them: tiles
    /// that are drawn but never touched age out as if unused.
    pub fn peek(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        let now = self.now();
        self.map.get(coord).map(|entry| {
            entry.last_used.store(now, Ordering::Relaxed);
            Arc::clone(&entry.tile)
        })
    }

    /// Read a tile without marking it as used (for diagnostics and
    /// speculative probes).
    pub fn peek_untracked(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        self.map.get(coord).map(|entry| Arc::clone(&entry.tile))
    }

    /// Marks a cached tile as used. Returns whether it was cached.
    pub fn touch(&self, coord: &TileCoord) -> bool {
        let now = self.now();
        self.map
            .get(coord)
            .map(|entry| entry.last_used.store(now, Ordering::Relaxed))
            .is_some()
    }

    /// Marks every cached tile in `coords` as used; returns how many were
    /// cached.
    pub fn touch_all<'a>(&self, coords: impl IntoIterator<Item = &'a TileCoord>) -> usize {
        coords.into_iter().filter(|coord| self.touch(coord)).count()
    }

    /// Insert a tile into the cache
    pub fn insert(&self, tile: TileData) {
        self.insert_arc(Arc::new(tile));
    }

    /// Insert a shared tile into the cache, replacing any previous tile at
    /// the same coordinate.
    pub fn insert_arc(&self, tile: Arc<TileData>) {
        let coord = tile.coord;
        let size_bytes = tile.data.len();
        // Stamp with the pre-increment value: tiles touched after this insert
        // see the incremented clock and so count as more recently used.
        let stamp = self.clock.fetch_add(1, Ordering::Relaxed);
        let entry = CacheEntry {
            tile,
            size_bytes,
            last_used: AtomicU64::new(stamp),
            pins: AtomicU32::new(0),
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
        };
        self.current_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        if let Some(old) = self.map.insert(coord, entry) {
            self.current_bytes
                .fetch_sub(old.size_bytes, Ordering::Relaxed);
            // Keep explicit pins across a replacement.
            let pins = old.pins.load(Ordering::Relaxed);
            if pins > 0
                && let Some(new) = self.map.get(&coord)
            {
                new.pins.fetch_add(pins, Ordering::Relaxed);
            }
        }
        self.insertions.fetch_add(1, Ordering::Relaxed);
        trace!("Cached tile {:?} ({} bytes)", coord, size_bytes);
        self.evict_if_needed();
    }

    /// Check if a tile is in the cache (does not mark it as used).
    pub fn contains(&self, coord: &TileCoord) -> bool {
        self.map.contains_key(coord)
    }

    /// Remove a tile from the cache
    pub fn remove(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        self.map.remove(coord).map(|(_, entry)| {
            self.current_bytes
                .fetch_sub(entry.size_bytes, Ordering::Relaxed);
            entry.tile
        })
    }

    /// Removes every tile for which `keep` returns false; returns how many
    /// were removed.
    pub fn retain(&self, mut keep: impl FnMut(&TileCoord, &TileData) -> bool) -> usize {
        let mut removed = 0;
        self.map.retain(|coord, entry| {
            let keep = keep(coord, &entry.tile);
            if !keep {
                self.current_bytes
                    .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                removed += 1;
            }
            keep
        });
        removed
    }

    /// Drops every tile of a file (call it when the file is closed; a
    /// reopened file gets a new id, so its old tiles would never be used
    /// again) and forgets the file's pins and visibility.
    pub fn remove_file(&self, file_id: i32) -> usize {
        {
            let mut policy = self.policy.write();
            policy.pinned_levels.remove(&file_id);
            policy.hidden_files.remove(&file_id);
        }
        self.policy_changed();
        let removed = self.retain(|coord, _| coord.file_id != file_id);
        debug!("Dropped {removed} cached tiles of closed file {file_id}");
        removed
    }

    /// Clear all tiles from the cache
    pub fn clear(&self) {
        self.retain(|_, _| false);
        debug!("Cache cleared");
    }

    /// Pins a cached tile so it is never evicted; returns whether it was
    /// cached. Pins nest; release with [`unpin`](Self::unpin).
    pub fn pin(&self, coord: &TileCoord) -> bool {
        self.map
            .get(coord)
            .map(|entry| entry.pins.fetch_add(1, Ordering::Relaxed))
            .is_some()
    }

    /// Releases one [`pin`](Self::pin) of a tile.
    pub fn unpin(&self, coord: &TileCoord) {
        if let Some(entry) = self.map.get(coord) {
            let _ = entry
                .pins
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |pins| {
                    pins.checked_sub(1)
                });
            self.policy_changed();
        }
    }

    /// Keeps the given levels of a file resident (typically its coarsest
    /// levels, drawn as fallback under every frame). Replaces any previous
    /// set for that file; an empty slice clears it.
    pub fn set_pinned_levels(&self, file_id: i32, levels: &[u32]) {
        let mut policy = self.policy.write();
        if levels.is_empty() {
            policy.pinned_levels.remove(&file_id);
        } else {
            policy
                .pinned_levels
                .insert(file_id, levels.iter().copied().collect());
        }
        self.policy_changed();
    }

    /// Marks a file visible (default) or hidden (no pane shows it); tiles of
    /// hidden files are evicted before any tile of a visible file.
    pub fn set_file_visible(&self, file_id: i32, visible: bool) {
        let mut policy = self.policy.write();
        if visible {
            policy.hidden_files.remove(&file_id);
        } else {
            policy.hidden_files.insert(file_id);
        }
        self.policy_changed();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            current_tiles: self.map.len(),
            current_bytes: self.current_bytes(),
            insertions: self.insertions.load(Ordering::Relaxed),
            max_bytes: self.max_bytes(),
            max_tiles: self.max_tiles(),
            eviction_scans: self.eviction_scans.load(Ordering::Relaxed),
        }
    }

    fn over(&self, max_bytes: usize, max_tiles: usize) -> bool {
        self.current_bytes() > max_bytes || self.map.len() > max_tiles
    }

    /// Evicts least-recently-used tiles once a limit is exceeded, down to
    /// 7/8 of the limits.
    fn evict_if_needed(&self) {
        let max_bytes = self.max_bytes();
        let max_tiles = self.max_tiles();
        if !self.over(max_bytes, max_tiles) {
            return;
        }
        let frame_start = self.frame_start.load(Ordering::Relaxed);
        let policy_version = self.policy_version.load(Ordering::Relaxed);
        let hard_bytes = max_bytes.saturating_mul(HARD_LIMIT_FACTOR);
        let hard_tiles = max_tiles.saturating_mul(HARD_LIMIT_FACTOR);
        let futile_bytes = self.futile_until_bytes.load(Ordering::Relaxed);
        if futile_bytes != 0
            && self.futile_frame.load(Ordering::Relaxed) == frame_start
            && self.futile_policy.load(Ordering::Relaxed) == policy_version
            && self.current_bytes() < futile_bytes
            && self.map.len() < self.futile_until_tiles.load(Ordering::Relaxed)
            && !self.over(hard_bytes, hard_tiles)
        {
            // The last pass could not free anything that is not protected;
            // scanning again now would only cost time.
            return;
        }
        // One evictor at a time; concurrent inserters just add to the batch.
        let Some(_guard) = self.evicting.try_lock() else {
            return;
        };
        self.eviction_scans.fetch_add(1, Ordering::Relaxed);
        let target_bytes = max_bytes - max_bytes / HEADROOM_DIVISOR;
        let target_tiles = max_tiles - max_tiles / HEADROOM_DIVISOR;

        // Snapshot candidates: (class, last_used, coord, id, size, protected).
        // class 0 = hidden file, 1 = visible file.
        let mut candidates = Vec::with_capacity(self.map.len());
        {
            let policy = self.policy.read();
            for item in self.map.iter() {
                let coord = *item.key();
                let entry = item.value();
                if entry.pins.load(Ordering::Relaxed) > 0
                    || policy
                        .pinned_levels
                        .get(&coord.file_id)
                        .is_some_and(|levels| levels.contains(&coord.level))
                {
                    continue;
                }
                let last_used = entry.last_used.load(Ordering::Relaxed);
                let class = u8::from(!policy.hidden_files.contains(&coord.file_id));
                let protected = frame_start != NO_FRAME && last_used >= frame_start;
                candidates.push((
                    class,
                    last_used,
                    coord,
                    entry.id,
                    entry.size_bytes,
                    protected,
                ));
            }
        }
        candidates.sort_unstable_by_key(|&(class, last_used, ..)| (class, last_used));

        let mut evicted = 0u64;
        let mut stopped_at_protected = false;
        // First pass: unprotected tiles down to the target. Second pass:
        // protected tiles, only while above the hard limit.
        for allow_protected in [false, true] {
            for &(_, last_used, coord, id, _, protected) in &candidates {
                let done = if allow_protected {
                    !self.over(hard_bytes, hard_tiles)
                } else {
                    self.current_bytes() <= target_bytes && self.map.len() <= target_tiles
                };
                if done {
                    break;
                }
                if protected != allow_protected {
                    if protected {
                        stopped_at_protected = true;
                    }
                    continue;
                }
                // Skip entries that were replaced or used since the snapshot.
                if let Some((_, entry)) = self.map.remove_if(&coord, |_, entry| {
                    entry.id == id
                        && entry.pins.load(Ordering::Relaxed) == 0
                        && entry.last_used.load(Ordering::Relaxed) == last_used
                }) {
                    self.current_bytes
                        .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                    evicted += 1;
                    trace!("Evicted tile {:?}", coord);
                }
            }
        }
        self.evictions.fetch_add(evicted, Ordering::Relaxed);

        if self.over(max_bytes, max_tiles) {
            // Everything left is protected or pinned: skip passes until the
            // cache grows by another 1/16 of its limits (or the frame or the
            // policy changes).
            self.futile_until_bytes.store(
                (self.current_bytes() + max_bytes / 16).max(1),
                Ordering::Relaxed,
            );
            self.futile_until_tiles
                .store(self.map.len() + max_tiles / 16 + 1, Ordering::Relaxed);
            self.futile_frame.store(frame_start, Ordering::Relaxed);
            self.futile_policy.store(policy_version, Ordering::Relaxed);
            if stopped_at_protected {
                self.warn_overshoot(max_bytes);
            }
        } else {
            self.futile_until_bytes.store(0, Ordering::Relaxed);
        }
    }

    fn warn_overshoot(&self, max_bytes: usize) {
        let now = Instant::now();
        let mut last = self.last_overshoot_warning.lock();
        if last.is_none_or(|at| now.duration_since(at) >= OVERSHOOT_WARN_INTERVAL) {
            *last = Some(now);
            warn!(
                "Tile cache budget of {} MiB is smaller than the current frame's working set \
                 ({} MiB in use); raise the cache size to avoid reloading tiles",
                max_bytes / (1024 * 1024),
                self.current_bytes() / (1024 * 1024)
            );
        }
    }
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILE_BYTES: usize = 64 * 64 * 4;

    fn tile(file_id: i32, level: u32, x: u64) -> TileData {
        TileData::placeholder(TileCoord::new(file_id, level, x, 0, 64), 64)
    }

    fn coord(file_id: i32, level: u32, x: u64) -> TileCoord {
        TileCoord::new(file_id, level, x, 0, 64)
    }

    /// A cache that holds `tiles` 64x64 tiles by bytes.
    fn cache_for(tiles: usize) -> TileCache {
        TileCache::with_limits(1_000_000, tiles * TILE_BYTES)
    }

    #[test]
    fn test_cache_basic_operations() {
        let cache = TileCache::new();
        let coord = TileCoord::new(0, 0, 0, 0, 256);
        let tile = TileData::placeholder(coord, 256);

        cache.insert(tile.clone());
        assert!(cache.contains(&coord));

        let retrieved = cache.get(&coord).unwrap();
        assert_eq!(retrieved.coord, coord);
        assert_eq!(retrieved.data.len(), tile.data.len());
        assert_eq!(cache.current_bytes(), tile.data.len());
        assert_eq!(cache.remove(&coord).unwrap().coord, coord);
        assert_eq!(cache.current_bytes(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let tile_size = 256 * 256 * 4;
        let cache = TileCache::with_limits(2, tile_size * 2 + 1);
        for x in 0..3 {
            cache.insert(TileData::placeholder(TileCoord::new(0, 0, x, 0, 256), 256));
        }
        assert!(cache.stats().current_tiles <= 2);
        assert!(!cache.contains(&TileCoord::new(0, 0, 0, 0, 256)));
        assert!(cache.contains(&TileCoord::new(0, 0, 2, 0, 256)));
    }

    #[test]
    fn test_cache_stats() {
        let cache = TileCache::new();
        let coord = TileCoord::new(0, 0, 0, 0, 256);

        cache.get(&coord);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        cache.insert(TileData::placeholder(coord, 256));
        cache.get(&coord);

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.insertions, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn peeked_tiles_survive_eviction_like_an_lru() {
        // The audit's reproduction: in a 3-tile cache, peeking tile 0 and
        // then inserting a fourth tile used to evict tile 0 (FIFO).
        let cache = TileCache::with_limits(3, usize::MAX);
        for x in 0..3 {
            cache.insert(tile(0, 0, x));
        }
        for _ in 0..100 {
            assert!(cache.peek(&coord(0, 0, 0)).is_some());
        }
        cache.insert(tile(0, 0, 3));
        assert!(
            cache.contains(&coord(0, 0, 0)),
            "recently used tile was evicted"
        );
        assert!(
            !cache.contains(&coord(0, 0, 1)),
            "least recently used tile kept"
        );

        // peek_untracked does not count as a use.
        let cache = TileCache::with_limits(3, usize::MAX);
        for x in 0..3 {
            cache.insert(tile(0, 0, x));
        }
        cache.peek_untracked(&coord(0, 0, 0));
        cache.insert(tile(0, 0, 3));
        assert!(!cache.contains(&coord(0, 0, 0)));
    }

    #[test]
    fn touch_marks_tiles_as_used() {
        let cache = cache_for(4);
        for x in 0..4 {
            cache.insert(tile(0, 0, x));
        }
        assert_eq!(
            cache.touch_all(&[coord(0, 0, 0), coord(0, 0, 1), coord(0, 0, 9)]),
            2
        );
        assert!(!cache.touch(&coord(0, 0, 9)));
        cache.insert(tile(0, 0, 4));
        assert!(cache.contains(&coord(0, 0, 0)));
        assert!(cache.contains(&coord(0, 0, 1)));
        assert!(!cache.contains(&coord(0, 0, 2)));
    }

    #[test]
    fn the_current_frame_is_protected_up_to_the_hard_limit() {
        let cache = cache_for(4);
        cache.begin_frame();
        for x in 0..6 {
            cache.insert(tile(0, 0, x));
        }
        // All six were used this frame: kept although over the budget.
        assert_eq!(cache.len(), 6);

        // Next frame only uses two of them; the rest become evictable.
        cache.begin_frame();
        cache.touch(&coord(0, 0, 4));
        cache.touch(&coord(0, 0, 5));
        cache.insert(tile(0, 0, 6));
        assert!(cache.current_bytes() <= 4 * TILE_BYTES);
        for x in 4..7 {
            assert!(
                cache.contains(&coord(0, 0, x)),
                "tile {x} of this frame evicted"
            );
        }

        // Beyond twice the budget even this frame's tiles go.
        cache.begin_frame();
        for x in 10..30 {
            cache.insert(tile(0, 0, x));
        }
        assert!(cache.current_bytes() <= 2 * 4 * TILE_BYTES);
    }

    #[test]
    fn futile_eviction_passes_are_skipped() {
        let cache = cache_for(64);
        cache.begin_frame();
        // A frame whose working set is 1.5x the budget: everything is
        // protected, so after a futile pass inserts only rescan once the
        // cache has grown by another 1/16 of the budget (4 tiles here).
        for x in 0..96 {
            cache.insert(tile(0, 0, x));
        }
        let scans = cache.stats().eviction_scans;
        assert!(scans <= 9, "{scans} scans for 32 inserts over budget");
        assert_eq!(cache.len(), 96);
        // A new frame makes the old tiles evictable again.
        cache.begin_frame();
        cache.insert(tile(0, 0, 999));
        assert!(cache.current_bytes() <= 64 * TILE_BYTES);
    }

    #[test]
    fn pinned_levels_and_tiles_are_never_evicted() {
        let cache = cache_for(2);
        cache.set_pinned_levels(7, &[5]);
        cache.insert(tile(7, 5, 0));
        cache.insert(tile(7, 0, 0));
        assert!(cache.pin(&coord(7, 0, 0)));
        for x in 1..10 {
            cache.insert(tile(7, 0, x));
        }
        assert!(cache.contains(&coord(7, 5, 0)));
        assert!(cache.contains(&coord(7, 0, 0)));

        cache.unpin(&coord(7, 0, 0));
        cache.unpin(&coord(7, 0, 0)); // extra unpins are harmless
        cache.set_pinned_levels(7, &[]);
        for x in 10..20 {
            cache.insert(tile(7, 0, x));
        }
        assert!(!cache.contains(&coord(7, 5, 0)));
        assert!(!cache.contains(&coord(7, 0, 0)));
    }

    #[test]
    fn hidden_files_are_evicted_first_and_closed_files_at_once() {
        let cache = cache_for(4);
        cache.insert(tile(1, 0, 0));
        cache.insert(tile(1, 0, 1));
        cache.insert(tile(2, 0, 0));
        cache.insert(tile(2, 0, 1));
        cache.set_file_visible(2, false);
        // File 2's tiles are newer, but hidden: they go first.
        cache.insert(tile(1, 0, 2));
        assert!(cache.contains(&coord(1, 0, 0)));
        assert!(!cache.contains(&coord(2, 0, 0)));

        assert_eq!(cache.remove_file(1), 3);
        assert!(cache.iter_coords_for_test().iter().all(|c| c.file_id != 1));
        assert_eq!(
            cache.current_bytes(),
            cache.len() * TILE_BYTES,
            "byte accounting matches the map"
        );
    }

    #[test]
    fn huge_tile_limits_allocate_nothing_up_front() {
        // `--max-tiles 1e10` used to preallocate the LRU table and abort.
        let cache = TileCache::with_limits(usize::MAX, 1 << 20);
        assert_eq!(cache.max_tiles(), MAX_TILES_LIMIT);
        cache.insert(tile(0, 0, 0));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn concurrent_inserts_and_stats_keep_accounting_consistent() {
        // Previously `stats()` and the eviction loop took two mutexes in
        // opposite orders and deadlocked; the map and the LRU could also
        // disagree, orphaning tiles.
        let cache = Arc::new(cache_for(32));
        std::thread::scope(|scope| {
            for t in 0..8 {
                let cache = Arc::clone(&cache);
                scope.spawn(move || {
                    for i in 0..400u64 {
                        cache.insert(tile(t, 0, i % 50));
                        cache.peek(&coord(t, 0, (i * 7) % 50));
                    }
                });
            }
            for _ in 0..4 {
                let cache = Arc::clone(&cache);
                scope.spawn(move || {
                    for _ in 0..2000 {
                        let stats = cache.stats();
                        assert!(stats.current_tiles <= 8 * 50);
                    }
                });
            }
        });
        assert_eq!(cache.current_bytes(), cache.len() * TILE_BYTES);
        // Concurrent inserters may skip eviction while another thread runs
        // it; the next eviction catches up.
        cache.set_limits(1_000_000, 32 * TILE_BYTES);
        assert!(cache.current_bytes() <= 32 * TILE_BYTES);
        cache.clear();
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn set_limits_shrinks_the_cache() {
        let cache = cache_for(10);
        for x in 0..10 {
            cache.insert(tile(0, 0, x));
        }
        cache.set_limits(1_000_000, 2 * TILE_BYTES);
        assert!(cache.len() <= 2);
        assert_eq!(cache.max_bytes(), 2 * TILE_BYTES);
    }

    #[test]
    fn display_budgets_cover_several_frames() {
        let hd = CacheBudget::for_display(1920, 1080, 1);
        let frame = CacheBudget::frame_working_set_bytes(1920, 1080) as usize;
        assert!(hd.max_bytes >= 2 * frame);
        assert!(hd.max_bytes > 256 * 1024 * 1024);
        let uhd = CacheBudget::for_display(3840, 2160, 1);
        assert!(uhd.max_bytes > hd.max_bytes);
        let split = CacheBudget::for_display(3840, 2160, 4);
        assert_eq!(split.max_bytes, CacheBudget::MAX_BYTES);
        let tiny = CacheBudget::for_display(100, 100, 1);
        assert_eq!(tiny.max_bytes, CacheBudget::MIN_BYTES);
        assert!(hd.max_tiles * 258 * 258 * 4 >= hd.max_bytes);
        assert_eq!(CacheBudget::default().max_bytes, DEFAULT_CACHE_SIZE_BYTES);
    }

    impl TileCache {
        fn iter_coords_for_test(&self) -> Vec<TileCoord> {
            self.map.iter().map(|item| *item.key()).collect()
        }
    }
}
