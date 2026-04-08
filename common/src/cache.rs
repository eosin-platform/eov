//! Tile caching system for WSI rendering
//!
//! Implements a multi-level cache with LRU eviction for efficient
//! tile management during WSI viewing.

use crate::tile::{TileCoord, TileData, TileManager};
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

/// Default maximum cache size in bytes (256MB)
pub const DEFAULT_CACHE_SIZE_BYTES: usize = 256 * 1024 * 1024;

/// Default maximum number of tiles to cache
pub const DEFAULT_MAX_TILES: usize = 2048;

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry {
    tile: Arc<TileData>,
    last_access: Instant,
    size_bytes: usize,
}

impl CacheEntry {
    fn new(tile: TileData) -> Self {
        let size_bytes = tile.data.len();
        Self {
            tile: Arc::new(tile),
            last_access: Instant::now(),
            size_bytes,
        }
    }
}

/// Statistics for cache monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_tiles: usize,
    pub current_bytes: usize,
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

/// Thread-safe tile cache with LRU eviction
pub struct TileCache {
    /// Primary cache using DashMap for concurrent access
    cache: DashMap<TileCoord, CacheEntry>,
    /// LRU tracking for eviction
    lru: Mutex<LruCache<TileCoord, ()>>,
    /// Maximum cache size in bytes
    max_bytes: usize,
    /// Maximum number of tiles
    max_tiles: usize,
    /// Current size in bytes
    current_bytes: Mutex<usize>,
    /// Statistics
    stats: Mutex<CacheStats>,
}

impl TileCache {
    /// Create a new tile cache with default settings
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_TILES, DEFAULT_CACHE_SIZE_BYTES)
    }

    /// Create a new tile cache with custom limits
    pub fn with_limits(max_tiles: usize, max_bytes: usize) -> Self {
        Self {
            cache: DashMap::new(),
            lru: Mutex::new(LruCache::new(
                NonZeroUsize::new(max_tiles).unwrap_or(NonZeroUsize::new(1).unwrap())
            )),
            max_bytes,
            max_tiles,
            current_bytes: Mutex::new(0),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Get a tile from the cache
    pub fn get(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        // Get the tile and release DashMap lock immediately
        let tile = {
            if let Some(mut entry) = self.cache.get_mut(coord) {
                entry.last_access = Instant::now();
                Some(Arc::clone(&entry.tile))
            } else {
                None
            }
        }; // DashMap lock released here
        
        if tile.is_some() {
            // Update LRU (now safe - not holding DashMap lock)
            {
                let mut lru = self.lru.lock();
                lru.get(coord);
            }
            
            // Update stats
            {
                let mut stats = self.stats.lock();
                stats.hits += 1;
            }
            
            trace!("Cache hit for {:?}", coord);
        } else {
            {
                let mut stats = self.stats.lock();
                stats.misses += 1;
            }
            trace!("Cache miss for {:?}", coord);
        }
        
        tile
    }

    /// Insert a tile into the cache
    pub fn insert(&self, tile: TileData) {
        let coord = tile.coord;
        let entry = CacheEntry::new(tile);
        let entry_size = entry.size_bytes;

        // Evict if necessary
        self.ensure_capacity(entry_size);

        // Insert into cache
        if let Some(old) = self.cache.insert(coord, entry) {
            // Update byte count (replacing existing)
            let mut bytes = self.current_bytes.lock();
            *bytes = bytes.saturating_sub(old.size_bytes);
            *bytes += entry_size;
        } else {
            // New entry
            let mut bytes = self.current_bytes.lock();
            *bytes += entry_size;
            
            // Add to LRU
            let mut lru = self.lru.lock();
            lru.put(coord, ());
        }

        trace!("Cached tile {:?} ({} bytes)", coord, entry_size);
    }

    /// Check if a tile is in the cache
    pub fn contains(&self, coord: &TileCoord) -> bool {
        self.cache.contains_key(coord)
    }

    /// Remove a tile from the cache
    pub fn remove(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        if let Some((_, entry)) = self.cache.remove(coord) {
            let mut bytes = self.current_bytes.lock();
            *bytes = bytes.saturating_sub(entry.size_bytes);
            
            let mut lru = self.lru.lock();
            lru.pop(coord);
            
            Some(entry.tile)
        } else {
            None
        }
    }

    /// Clear all tiles from the cache
    pub fn clear(&self) {
        self.cache.clear();
        *self.current_bytes.lock() = 0;
        self.lru.lock().clear();
        
        debug!("Cache cleared");
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let stats = self.stats.lock();
        let bytes = self.current_bytes.lock();
        CacheStats {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            current_tiles: self.cache.len(),
            current_bytes: *bytes,
        }
    }

    /// Ensure there's capacity for a new entry
    fn ensure_capacity(&self, required_bytes: usize) {
        // First, determine which tiles to evict while holding locks
        let coords_to_evict: Vec<TileCoord> = {
            let bytes = self.current_bytes.lock();
            let mut lru = self.lru.lock();
            
            let mut coords = Vec::new();
            let mut projected_bytes = *bytes;
            let mut projected_count = self.cache.len();
            
            // Collect coords to evict
            while (projected_bytes + required_bytes > self.max_bytes || projected_count >= self.max_tiles)
                && projected_count > 0
            {
                if let Some((coord, _)) = lru.pop_lru() {
                    let exact_size = self.cache.get(&coord)
                        .map(|entry| entry.size_bytes)
                        .unwrap_or(0);
                    projected_bytes = projected_bytes.saturating_sub(exact_size);
                    projected_count = projected_count.saturating_sub(1);
                    coords.push(coord);
                } else {
                    break;
                }
            }
            coords
        }; // Release bytes and lru locks here
        
        // Now evict tiles without holding the LRU lock
        for coord in coords_to_evict {
            if let Some((_, entry)) = self.cache.remove(&coord) {
                let mut bytes = self.current_bytes.lock();
                *bytes = bytes.saturating_sub(entry.size_bytes);
                
                let mut stats = self.stats.lock();
                stats.evictions += 1;
                trace!("Evicted tile {:?}", coord);
            }
        }
    }

    /// Get tiles that should be evicted (for manual eviction control)
    pub fn tiles_to_evict(&self, count: usize) -> Vec<TileCoord> {
        let mut lru = self.lru.lock();
        let mut coords = Vec::with_capacity(count);
        
        // Peek at LRU entries without removing
        for _ in 0..count {
            if let Some((coord, _)) = lru.pop_lru() {
                coords.push(coord);
                lru.put(coord, ()); // Re-insert to maintain order
            } else {
                break;
            }
        }
        
        coords
    }
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Async tile loader that manages background loading and caching
pub struct AsyncTileLoader {
    /// Tile manager
    manager: Arc<TileManager>,
    /// Tile cache
    cache: Arc<TileCache>,
    /// Request channel
    request_tx: mpsc::Sender<TileCoord>,
    /// Currently loading tiles (to avoid duplicate requests)
    loading: DashMap<TileCoord, ()>,
}

impl AsyncTileLoader {
    /// Create a new async tile loader
    pub fn new(
        manager: TileManager,
        cache: Arc<TileCache>,
        worker_count: usize,
    ) -> (Self, mpsc::Receiver<TileData>) {
        let manager = Arc::new(manager);
        let (request_tx, request_rx) = mpsc::channel::<TileCoord>(1024);
        let (response_tx, response_rx) = mpsc::channel::<TileData>(1024);
        let loading = DashMap::new();

        // Spawn worker tasks
        let request_rx = Arc::new(tokio::sync::Mutex::new(request_rx));
        
        for i in 0..worker_count {
            let manager = Arc::clone(&manager);
            let cache = Arc::clone(&cache);
            let request_rx = Arc::clone(&request_rx);
            let response_tx = response_tx.clone();
            let loading = loading.clone();

            tokio::spawn(async move {
                loop {
                    let coord = {
                        let mut rx = request_rx.lock().await;
                        match rx.recv().await {
                            Some(coord) => coord,
                            None => break,
                        }
                    };

                    // Skip if already cached
                    if cache.contains(&coord) {
                        loading.remove(&coord);
                        continue;
                    }

                    // Load tile
                    match manager.load_tile_sync(coord) {
                        Ok(tile) => {
                            cache.insert(tile.clone());
                            if response_tx.send(tile).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load tile {:?}: {}", coord, e);
                        }
                    }

                    loading.remove(&coord);
                }

                debug!("Tile loader worker {} shutting down", i);
            });
        }

        (
            Self {
                manager,
                cache,
                request_tx,
                loading,
            },
            response_rx,
        )
    }

    /// Request tiles to be loaded
    pub async fn request_tiles(&self, coords: &[TileCoord]) {
        for coord in coords {
            // Skip if already cached or loading
            if self.cache.contains(coord) || self.loading.contains_key(coord) {
                continue;
            }

            self.loading.insert(*coord, ());
            
            if self.request_tx.send(*coord).await.is_err() {
                warn!("Tile loader channel closed");
                break;
            }
        }
    }

    /// Get a tile from cache, or return None if not available
    pub fn get_cached(&self, coord: &TileCoord) -> Option<Arc<TileData>> {
        self.cache.get(coord)
    }

    /// Check if a tile is currently being loaded
    pub fn is_loading(&self, coord: &TileCoord) -> bool {
        self.loading.contains_key(coord)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = TileCache::new();
        let coord = TileCoord::new(0, 0, 0);
        let tile = TileData::placeholder(coord, 256);

        // Insert and retrieve
        cache.insert(tile.clone());
        assert!(cache.contains(&coord));
        
        let retrieved = cache.get(&coord).unwrap();
        assert_eq!(retrieved.coord, coord);
        assert_eq!(retrieved.data.len(), tile.data.len());
    }

    #[test]
    fn test_cache_eviction() {
        // Small cache that can only hold 2 tiles
        let tile_size = 256 * 256 * 4; // ~256KB per tile
        let cache = TileCache::with_limits(2, tile_size * 2 + 1);

        let tile1 = TileData::placeholder(TileCoord::new(0, 0, 0), 256);
        let tile2 = TileData::placeholder(TileCoord::new(0, 1, 0), 256);
        let tile3 = TileData::placeholder(TileCoord::new(0, 2, 0), 256);

        cache.insert(tile1);
        cache.insert(tile2);
        assert_eq!(cache.stats().current_tiles, 2);

        // Adding third tile should evict the first (LRU)
        cache.insert(tile3);
        assert!(cache.stats().current_tiles <= 2);
    }

    #[test]
    fn test_cache_stats() {
        let cache = TileCache::new();
        let coord = TileCoord::new(0, 0, 0);

        // Miss
        cache.get(&coord);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Insert and hit
        cache.insert(TileData::placeholder(coord, 256));
        cache.get(&coord);
        
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }
}
