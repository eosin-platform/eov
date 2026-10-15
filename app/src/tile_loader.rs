//! Background tile loading.
//!
//! One worker pool serves every open slide (RND-8). Each frame the renderer
//! replaces a slide's request queue with the tiles it wants now, in priority
//! order (centre first, the displayed level before fallbacks), so requests
//! for tiles that scrolled away are dropped without being loaded (RND-5).
//! Workers sleep on a condition variable instead of polling, and closing a
//! slide never joins threads on the UI thread.
//!
//! Failures: regions the WSI core reports as corrupt are not retried for a
//! long time (the renderer draws coarser levels there); transient failures
//! (a recovering OpenSlide handle, I/O) are retried after a few frames.

use eov_common::{Error, TileCache, TileCoord, TileManager};
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tracing::trace;

/// Frames to wait before retrying a tile after a transient failure.
const TRANSIENT_RETRY_GENERATIONS: u64 = 12;
/// Frames to wait before retrying a tile in a corrupt region (~5 min at 60 fps).
const CORRUPT_RETRY_GENERATIONS: u64 = 60 * 300;

fn loader_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().clamp(4, 16))
        .unwrap_or(4)
}

fn max_queued_tiles(worker_count: usize) -> usize {
    (worker_count * 24).clamp(64, 256)
}

struct FileQueue {
    manager: Arc<TileManager>,
    cache: Arc<TileCache>,
    queue: VecDeque<TileCoord>,
    in_flight: HashSet<TileCoord>,
    /// Tile → generation after which it may be retried.
    failed: HashMap<TileCoord, u64>,
    generation: u64,
    pending: Arc<AtomicUsize>,
    loaded_epoch: Arc<AtomicU64>,
}

impl FileQueue {
    fn update_pending(&self) {
        self.pending
            .store(self.queue.len() + self.in_flight.len(), Ordering::Relaxed);
    }
}

#[derive(Default)]
struct PoolState {
    files: HashMap<u64, FileQueue>,
    /// Round-robin position, so one slide cannot starve another.
    turn: usize,
}

struct Pool {
    state: Mutex<PoolState>,
    work: Condvar,
    max_queued: usize,
    next_id: AtomicU64,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = loader_worker_count();
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(PoolState::default()),
            work: Condvar::new(),
            max_queued: max_queued_tiles(workers),
            next_id: AtomicU64::new(1),
        }));
        for index in 0..workers {
            std::thread::Builder::new()
                .name(format!("eov-tile-loader-{index}"))
                .spawn(move || worker_loop(pool))
                .expect("spawn tile loader");
        }
        pool
    })
}

/// A slide's handle on the shared loader pool. Dropping it (when the slide
/// is closed) cancels its queue and frees its cached tiles.
pub struct TileLoader {
    id: u64,
    file_id: i32,
    cache: Arc<TileCache>,
    pending: Arc<AtomicUsize>,
    loaded_epoch: Arc<AtomicU64>,
}

impl TileLoader {
    pub fn new(tile_manager: Arc<TileManager>, cache: Arc<TileCache>) -> Self {
        let pool = pool();
        let id = pool.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::new(AtomicUsize::new(0));
        let loaded_epoch = Arc::new(AtomicU64::new(0));
        let file_id = tile_manager.file_id();
        pool.state.lock().files.insert(
            id,
            FileQueue {
                manager: tile_manager,
                cache: Arc::clone(&cache),
                queue: VecDeque::new(),
                in_flight: HashSet::new(),
                failed: HashMap::new(),
                generation: 0,
                pending: Arc::clone(&pending),
                loaded_epoch: Arc::clone(&loaded_epoch),
            },
        );
        Self {
            id,
            file_id,
            cache,
            pending,
            loaded_epoch,
        }
    }

    /// Replaces the slide's queue with `tiles`, most important first.
    pub fn set_wanted_tiles(&self, tiles: Vec<TileCoord>) {
        let pool = pool();
        let mut state = pool.state.lock();
        let Some(file) = state.files.get_mut(&self.id) else {
            return;
        };
        file.generation += 1;
        let generation = file.generation;
        if generation.is_multiple_of(64) {
            file.failed.retain(|_, retry| *retry >= generation);
        }
        let mut seen = HashSet::new();
        let queue: VecDeque<TileCoord> = tiles
            .into_iter()
            .filter(|coord| seen.insert(*coord))
            .filter(|coord| !file.in_flight.contains(coord))
            .filter(|coord| {
                file.failed
                    .get(coord)
                    .is_none_or(|retry| *retry < generation)
            })
            .filter(|coord| !file.cache.contains(coord))
            .take(pool.max_queued)
            .collect();
        let has_work = !queue.is_empty();
        file.queue = queue;
        file.update_pending();
        drop(state);
        if has_work {
            pool.work.notify_all();
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    pub fn loaded_epoch(&self) -> u64 {
        self.loaded_epoch.load(Ordering::Relaxed)
    }

    /// Stops loading for this slide and frees its cached tiles now, even
    /// while render jobs still hold the loader. Retire the tile manager
    /// first ([`TileManager::retire`]), so that loads already running cannot
    /// cache tiles again afterwards. Used when a new loader with a new tile
    /// file id takes over (a multichannel slide's display changed).
    pub fn retire(&self) {
        pool().state.lock().files.remove(&self.id);
        self.cache.remove_file(self.file_id);
    }
}

impl Drop for TileLoader {
    fn drop(&mut self) {
        // In-flight loads finish on their worker and are discarded.
        self.retire();
    }
}

fn worker_loop(pool: &'static Pool) {
    loop {
        let (id, coord, manager) = {
            let mut state = pool.state.lock();
            loop {
                let ids: Vec<u64> = {
                    let mut ids: Vec<u64> = state
                        .files
                        .iter()
                        .filter(|(_, file)| !file.queue.is_empty())
                        .map(|(id, _)| *id)
                        .collect();
                    ids.sort_unstable();
                    ids
                };
                if let Some(id) = (!ids.is_empty()).then(|| ids[state.turn % ids.len()]) {
                    state.turn = state.turn.wrapping_add(1);
                    let file = state.files.get_mut(&id).expect("listed");
                    let coord = file.queue.pop_front().expect("non-empty");
                    file.in_flight.insert(coord);
                    break (id, coord, Arc::clone(&file.manager));
                }
                pool.work.wait(&mut state);
            }
        };

        let result = manager.load_tile_cached(coord);

        let mut state = pool.state.lock();
        let Some(file) = state.files.get_mut(&id) else {
            continue;
        };
        file.in_flight.remove(&coord);
        match result {
            Ok(tile) => {
                // `load_tile_cached` already inserted it when the manager
                // has the cache attached (and must not when it is retired).
                if !file.cache.contains(&coord) && !file.manager.is_retired() {
                    file.cache.insert_arc(tile);
                }
                file.loaded_epoch.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                let wait = if error.is_transient() {
                    TRANSIENT_RETRY_GENERATIONS
                } else if matches!(error, Error::CorruptRegion { .. }) {
                    CORRUPT_RETRY_GENERATIONS
                } else {
                    TRANSIENT_RETRY_GENERATIONS * 4
                };
                trace!("Tile {coord:?} failed: {error}");
                file.failed.insert(coord, file.generation + wait);
                // A failed tile still changes what the renderer can show
                // (it falls back to coarser levels).
                file.loaded_epoch.fetch_add(1, Ordering::Relaxed);
            }
        }
        file.update_pending();
    }
}

/// Calculate tiles needed for the current viewport, most important first:
/// the displayed level centre-out, then a few tiles of each coarser level.
pub fn calculate_wanted_tiles(
    tile_manager: &TileManager,
    level: u32,
    bounds_left: f64,
    bounds_top: f64,
    bounds_right: f64,
    bounds_bottom: f64,
    margin_tiles: i32,
) -> Vec<TileCoord> {
    let mut wanted = Vec::new();
    let level_count = tile_manager.wsi().level_count();

    // The tile manager returns visible tiles nearest the view centre first.
    let visible = tile_manager.visible_tiles_with_margin(
        level,
        bounds_left,
        bounds_top,
        bounds_right,
        bounds_bottom,
        margin_tiles,
    );
    wanted.extend(visible.iter().take(500).copied());

    // Coarser levels give quick feedback while the displayed level loads.
    for fallback_level in (level + 1)..level_count {
        let fallback_tiles = tile_manager.visible_tiles_with_margin(
            fallback_level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
            margin_tiles.max(0),
        );
        // Closer levels get more tiles, as they look better.
        let priority_boost = level_count.saturating_sub(fallback_level);
        let tiles_at_this_level =
            (fallback_tiles.len().min(20) + priority_boost as usize * 10).min(50);
        wanted.extend(fallback_tiles.iter().take(tiles_at_this_level).copied());
    }

    wanted
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::WsiFile;
    use std::time::{Duration, Instant};

    fn write_slide(path: &std::path::Path) {
        // A minimal tiled TIFF that OpenSlide's generic-tiff driver opens.
        eov_common::test_support::write_tiled_tiff(path, 512, 512, 256, |x, y| {
            [(x % 256) as u8, (y % 256) as u8, 128]
        })
        .unwrap();
    }

    #[test]
    fn tiles_load_in_the_background_and_stale_requests_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tif");
        write_slide(&path);
        let wsi = WsiFile::open(&path).unwrap();
        let cache = Arc::new(TileCache::with_limits(256, 64 << 20));
        let manager = Arc::new(TileManager::new(wsi, 1).with_tile_cache(Arc::clone(&cache)));
        let loader = TileLoader::new(Arc::clone(&manager), Arc::clone(&cache));
        let level = 0;
        let wanted = calculate_wanted_tiles(&manager, level, 0.0, 0.0, 512.0, 512.0, 0);
        assert!(!wanted.is_empty());
        loader.set_wanted_tiles(wanted.clone());
        let deadline = Instant::now() + Duration::from_secs(10);
        while wanted.iter().any(|coord| !cache.contains(coord)) {
            assert!(Instant::now() < deadline, "tiles did not load");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(loader.loaded_epoch() >= 1);
        // Requesting cached tiles again queues nothing.
        loader.set_wanted_tiles(wanted.clone());
        assert_eq!(loader.pending_count(), 0);
        // Closing the slide frees its tiles.
        drop(loader);
        assert!(wanted.iter().all(|coord| !cache.contains(coord)));
    }
}
