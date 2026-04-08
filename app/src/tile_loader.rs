//! Background tile loading system with automatic cancellation
//!
//! This module provides asynchronous tile loading that automatically discards
//! requests for tiles that are no longer visible.

use common::{TileCache, TileCoord, TileData, TileManager, WsiFile};
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::trace;

/// Number of background worker threads for tile loading
const WORKER_COUNT: usize = 4;

/// Maximum tiles to send to workers per frame
const MAX_TILES_PER_FRAME: usize = 32;

/// Background tile loader with automatic cancellation of stale requests
pub struct TileLoader {
    /// Channel to send tile requests to workers
    request_tx: Sender<TileCoord>,
    /// Set of tiles that failed to load (to avoid repeated requests)
    failed: Arc<Mutex<HashSet<TileCoord>>>,
    /// Current generation - tiles from old generations are skipped
    generation: Arc<AtomicU64>,
    /// Tiles currently queued or loading, used to deduplicate work
    pending: Arc<Mutex<HashSet<TileCoord>>>,
    /// Number of tiles currently queued or loading
    pending_count: Arc<AtomicUsize>,
    /// Monotonically increasing epoch when a new tile becomes available
    loaded_epoch: Arc<AtomicU64>,
    /// Worker handles
    workers: Vec<thread::JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Reference to cache
    cache: Arc<TileCache>,
}

impl TileLoader {
    /// Create a new tile loader
    pub fn new(tile_manager: Arc<TileManager>, cache: Arc<TileCache>) -> Self {
        // Bounded channel for tile requests - sized for good throughput
        let (request_tx, request_rx) = bounded::<TileCoord>(64);
        let failed = Arc::new(Mutex::new(HashSet::new()));
        let generation = Arc::new(AtomicU64::new(0));
        let pending = Arc::new(Mutex::new(HashSet::new()));
        let pending_count = Arc::new(AtomicUsize::new(0));
        let loaded_epoch = Arc::new(AtomicU64::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        
        let mut workers = Vec::with_capacity(WORKER_COUNT);
        
        for i in 0..WORKER_COUNT {
            let request_rx = request_rx.clone();
            let tile_manager = Arc::clone(&tile_manager);
            let worker_path = tile_manager.wsi().properties().path.clone();
            let cache = Arc::clone(&cache);
            let failed = Arc::clone(&failed);
            let pending = Arc::clone(&pending);
            let pending_count = Arc::clone(&pending_count);
            let loaded_epoch = Arc::clone(&loaded_epoch);
            let shutdown = Arc::clone(&shutdown);
            
            let handle = thread::spawn(move || {
                let tile_manager = match WsiFile::open(&worker_path) {
                    Ok(worker_wsi) => Arc::new(TileManager::new(worker_wsi)),
                    Err(_) => tile_manager,
                };

                worker_loop(
                    i,
                    request_rx,
                    tile_manager,
                    cache,
                    failed,
                    pending,
                    pending_count,
                    loaded_epoch,
                    shutdown,
                );
            });
            
            workers.push(handle);
        }
        
        Self {
            request_tx,
            failed,
            generation,
            pending,
            pending_count,
            loaded_epoch,
            workers,
            shutdown,
            cache,
        }
    }
    
    /// Update the set of wanted tiles for this frame
    /// Sends tiles to workers, skipping those already cached or failed
    pub fn set_wanted_tiles(&self, tiles: Vec<TileCoord>) {
        // Bump generation to invalidate any stale in-flight requests
        self.generation.fetch_add(1, Ordering::SeqCst);
        
        let mut sent = 0;
        
        // Send tiles to workers (limited to avoid flooding)
        for coord in tiles {
            if sent >= MAX_TILES_PER_FRAME {
                break;
            }
            
            // Skip if already cached
            if self.cache.contains(&coord) {
                continue;
            }

            // Skip if previously failed (brief lock)
            if self.failed.lock().contains(&coord) {
                continue;
            }

            // Skip if already queued or loading
            {
                let mut pending = self.pending.lock();
                if pending.contains(&coord) {
                    continue;
                }
                pending.insert(coord);
            }
            
            // Try to send to workers (non-blocking)
            match self.request_tx.try_send(coord) {
                Ok(_) => sent += 1,
                Err(TrySendError::Full(coord)) => {
                    self.pending.lock().remove(&coord);
                    break;
                }
                Err(TrySendError::Disconnected(coord)) => {
                    self.pending.lock().remove(&coord);
                    break;
                }
            }

            self.pending_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Clear failed tiles set (call when view changes significantly)
    pub fn clear_failed(&self) {
        self.failed.lock().clear();
    }

    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::Relaxed)
    }

    pub fn loaded_epoch(&self) -> u64 {
        self.loaded_epoch.load(Ordering::Relaxed)
    }
}

impl Drop for TileLoader {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Drop the sender to unblock workers
        drop(self.request_tx.clone());
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Worker thread loop
fn worker_loop(
    id: usize,
    request_rx: Receiver<TileCoord>,
    tile_manager: Arc<TileManager>,
    cache: Arc<TileCache>,
    failed: Arc<Mutex<HashSet<TileCoord>>>,
    pending: Arc<Mutex<HashSet<TileCoord>>>,
    pending_count: Arc<AtomicUsize>,
    loaded_epoch: Arc<AtomicU64>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) {
    trace!("Tile loader worker {} starting", id);
    
    while !shutdown.load(Ordering::Relaxed) {
        // Wait for a tile request with timeout
        let coord = match request_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(coord) => coord,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        
        // Skip if already in cache
        if cache.contains(&coord) {
            pending.lock().remove(&coord);
            pending_count.fetch_sub(1, Ordering::Relaxed);
            continue;
        }
        
        // Load the tile
        match tile_manager.load_tile_sync(coord) {
            Ok(tile) => {
                cache.insert(tile);
                loaded_epoch.fetch_add(1, Ordering::Relaxed);
                trace!("Worker {} loaded tile {:?}", id, coord);
            }
            Err(e) => {
                trace!("Worker {} failed to load tile {:?}: {}", id, coord, e);
                failed.lock().insert(coord);
            }
        }

        pending.lock().remove(&coord);
        pending_count.fetch_sub(1, Ordering::Relaxed);
    }
    
    trace!("Tile loader worker {} shutting down", id);
}

/// Find the best fallback tile for a given target tile coordinate
/// 
/// This searches lower-resolution (higher mip) levels for a tile that covers
/// the same area as the target tile. Returns the covering tile if found in cache.
#[allow(dead_code)]
pub fn find_fallback_tile(
    cache: &TileCache,
    wsi: &WsiFile,
    target: TileCoord,
    tile_size: u32,
) -> Option<(Arc<TileData>, u32)> {
    let level_count = wsi.level_count();
    
    // Get target tile's position in level-0 (image) coordinates
    let target_level_info = wsi.level(target.level)?;
    let target_image_x = target.x as f64 * tile_size as f64 * target_level_info.downsample;
    let target_image_y = target.y as f64 * tile_size as f64 * target_level_info.downsample;
    
    // Search lower-resolution levels (higher indices = lower resolution)
    for fallback_level in (target.level + 1)..level_count {
        let fallback_level_info = wsi.level(fallback_level)?;
        
        // Calculate which tile at the fallback level contains this image position
        // Each fallback tile covers (tile_size * downsample) pixels in image coordinates
        let fallback_tile_image_size = tile_size as f64 * fallback_level_info.downsample;
        let fallback_x = (target_image_x / fallback_tile_image_size).floor() as u64;
        let fallback_y = (target_image_y / fallback_tile_image_size).floor() as u64;
        
        let fallback_coord = TileCoord::new(fallback_level, fallback_x, fallback_y);
        
        if let Some(tile) = cache.get(&fallback_coord) {
            return Some((tile, fallback_level));
        }
    }
    
    None
}

/// Calculate the sub-region of a fallback tile that corresponds to the target tile
#[allow(dead_code)]
pub struct FallbackRegion {
    /// Source X offset within the fallback tile (0.0-1.0)
    pub src_x: f64,
    /// Source Y offset within the fallback tile (0.0-1.0)
    pub src_y: f64,
    /// Source width fraction (0.0-1.0)
    pub src_w: f64,
    /// Source height fraction (0.0-1.0)
    pub src_h: f64,
}

#[allow(dead_code)]
pub fn calculate_fallback_region(
    wsi: &WsiFile,
    target: TileCoord,
    fallback_level: u32,
    tile_size: u32,
) -> Option<FallbackRegion> {
    let target_level_info = wsi.level(target.level)?;
    let fallback_level_info = wsi.level(fallback_level)?;
    
    // Get target tile's position in level-0 (image) coordinates
    let target_image_x = target.x as f64 * tile_size as f64 * target_level_info.downsample;
    let target_image_y = target.y as f64 * tile_size as f64 * target_level_info.downsample;
    let target_image_w = tile_size as f64 * target_level_info.downsample;
    let target_image_h = tile_size as f64 * target_level_info.downsample;
    
    // Calculate the fallback tile's position in image coordinates
    let fallback_tile_image_size = tile_size as f64 * fallback_level_info.downsample;
    let fallback_x = (target_image_x / fallback_tile_image_size).floor() as u64;
    let fallback_y = (target_image_y / fallback_tile_image_size).floor() as u64;
    let fallback_image_x = fallback_x as f64 * fallback_tile_image_size;
    let fallback_image_y = fallback_y as f64 * fallback_tile_image_size;
    
    // Calculate the fractional position within the fallback tile (0.0-1.0)
    let frac_x = (target_image_x - fallback_image_x) / fallback_tile_image_size;
    let frac_y = (target_image_y - fallback_image_y) / fallback_tile_image_size;
    let frac_w = (target_image_w / fallback_tile_image_size).min(1.0 - frac_x);
    let frac_h = (target_image_h / fallback_tile_image_size).min(1.0 - frac_y);
    
    Some(FallbackRegion {
        src_x: frac_x,
        src_y: frac_y,
        src_w: frac_w,
        src_h: frac_h,
    })
}

/// Calculate tiles needed for the current viewport
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
    let center_x = (bounds_left + bounds_right) * 0.5;
    let center_y = (bounds_top + bounds_bottom) * 0.5;
    let tile_size = tile_manager.tile_size() as f64;

    let sort_by_center = |tiles: &mut Vec<TileCoord>, downsample: f64| {
        tiles.sort_by(|a, b| {
            let ax = ((a.x as f64 + 0.5) * tile_size * downsample - center_x).abs();
            let ay = ((a.y as f64 + 0.5) * tile_size * downsample - center_y).abs();
            let bx = ((b.x as f64 + 0.5) * tile_size * downsample - center_x).abs();
            let by = ((b.y as f64 + 0.5) * tile_size * downsample - center_y).abs();
            let da = ax + ay;
            let db = bx + by;
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    };
    
    // FIRST: Add current level tiles - these are the primary display tiles
    // Prioritize these over fallbacks so user sees sharp content ASAP
    let mut visible = tile_manager.visible_tiles_with_margin(
        level,
        bounds_left,
        bounds_top,
        bounds_right,
        bounds_bottom,
        margin_tiles,
    );
    if let Some(level_info) = tile_manager.wsi().level(level) {
        sort_by_center(&mut visible, level_info.downsample);
    }
    wanted.extend(visible.iter().take(500).copied());
    
    // THEN: Add lower-resolution tiles for fallback display (while high-res loads)
    // These provide quick visual feedback but are lower priority
    for fallback_level in (level + 1)..level_count {
        let mut fallback_tiles = tile_manager.visible_tiles_with_margin(
            fallback_level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
            margin_tiles.max(0),
        );
        if let Some(level_info) = tile_manager.wsi().level(fallback_level) {
            sort_by_center(&mut fallback_tiles, level_info.downsample);
        }
        
        // Closer levels (smaller fallback_level - level) get more tiles loaded
        // as they provide better fallback quality
        let priority_boost = level_count.saturating_sub(fallback_level);
        let tiles_at_this_level = (fallback_tiles.len().min(20) + priority_boost as usize * 10).min(50);
        wanted.extend(fallback_tiles.iter().take(tiles_at_this_level).copied());
    }
    
    wanted
}
