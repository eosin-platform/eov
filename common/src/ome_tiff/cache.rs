//! A process-wide, byte-bounded cache of decoded tiles and strips.
//!
//! eov's own tiles carry a one-pixel border, so a tile read touches the
//! file's neighbouring tiles too; and every channel of a multichannel read
//! is decoded separately. Without a cache each native tile would be
//! decompressed several times per frame. Raw channel values are cached,
//! not composited colour, so changing a channel's display re-composites
//! from here without decoding again.
//!
//! The cache is sharded to keep lock hold times short; each shard evicts
//! its least recently used chunks. Capacity follows the OpenSlide cache
//! setting ([`crate::wsi::set_openslide_cache_capacity`]).

use super::codec::DecodedChunk;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

const SHARDS: usize = 16;

/// Smallest and largest capacity, whatever the OpenSlide cache is set to.
pub(crate) const MIN_CAPACITY: usize = 128 << 20;
pub(crate) const MAX_CAPACITY: usize = 1 << 30;

/// Identifies a chunk: the slide (a per-open id), the IFD (by offset) and
/// the chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ChunkKey {
    pub slide: u64,
    pub ifd: u64,
    pub chunk: u32,
}

struct Entry {
    chunk: Arc<DecodedChunk>,
    last_used: u64,
}

#[derive(Default)]
struct Shard {
    map: HashMap<ChunkKey, Entry>,
    bytes: usize,
}

pub(crate) struct ChunkCache {
    shards: Vec<Mutex<Shard>>,
    capacity: AtomicUsize,
    clock: AtomicU64,
}

impl ChunkCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            capacity: AtomicUsize::new(capacity),
            clock: AtomicU64::new(0),
        }
    }

    fn shard(&self, key: &ChunkKey) -> &Mutex<Shard> {
        // Chunks of one region spread over the shards.
        let hash = key
            .slide
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(key.ifd.rotate_left(17))
            .wrapping_add(u64::from(key.chunk).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    fn shard_capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed) / SHARDS
    }

    pub(crate) fn get(&self, key: &ChunkKey) -> Option<Arc<DecodedChunk>> {
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut shard = self.shard(key).lock();
        shard.map.get_mut(key).map(|entry| {
            entry.last_used = now;
            Arc::clone(&entry.chunk)
        })
    }

    /// Inserts a chunk (unless it alone exceeds a shard's share) and
    /// evicts the shard's least recently used chunks beyond its share.
    pub(crate) fn insert(&self, key: ChunkKey, chunk: Arc<DecodedChunk>) {
        let size = chunk.byte_len();
        let limit = self.shard_capacity();
        if size > limit {
            return;
        }
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut shard = self.shard(&key).lock();
        if let Some(old) = shard.map.insert(
            key,
            Entry {
                chunk,
                last_used: now,
            },
        ) {
            shard.bytes -= old.chunk.byte_len();
        }
        shard.bytes += size;
        evict(&mut shard, limit);
    }

    /// Drops every chunk of a slide (when it closes).
    pub(crate) fn remove_slide(&self, slide: u64) {
        for shard in &self.shards {
            let mut shard = shard.lock();
            let Shard { map, bytes } = &mut *shard;
            map.retain(|key, entry| {
                let keep = key.slide != slide;
                if !keep {
                    *bytes -= entry.chunk.byte_len();
                }
                keep
            });
        }
    }

    pub(crate) fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::Relaxed);
        let limit = self.shard_capacity();
        for shard in &self.shards {
            evict(&mut shard.lock(), limit);
        }
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.shards.iter().map(|s| s.lock().bytes).sum()
    }
}

fn evict(shard: &mut Shard, limit: usize) {
    if shard.bytes <= limit {
        return;
    }
    let mut by_age: Vec<(u64, ChunkKey)> = shard
        .map
        .iter()
        .map(|(key, entry)| (entry.last_used, *key))
        .collect();
    by_age.sort_unstable_by_key(|(last_used, _)| *last_used);
    // Down to 7/8 of the share, so eviction runs in batches.
    let target = limit - limit / 8;
    for (_, key) in by_age {
        if shard.bytes <= target {
            break;
        }
        if let Some(entry) = shard.map.remove(&key) {
            shard.bytes -= entry.chunk.byte_len();
        }
    }
}

/// The cache size for an OpenSlide cache of `openslide_bytes`: raw
/// multichannel samples take more room than OpenSlide's RGB tiles.
pub(crate) fn capacity_for(openslide_bytes: usize) -> usize {
    openslide_bytes
        .saturating_mul(2)
        .clamp(MIN_CAPACITY, MAX_CAPACITY)
}

pub(crate) static CHUNKS: LazyLock<ChunkCache> =
    LazyLock::new(|| ChunkCache::new(capacity_for(crate::wsi::openslide_cache_capacity())));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ome_tiff::tiff::SampleType;

    fn chunk(bytes: usize) -> Arc<DecodedChunk> {
        Arc::new(DecodedChunk {
            width: bytes as u32,
            rows: 1,
            samples: 1,
            sample_type: SampleType::U8,
            data: vec![0; bytes],
            missing: false,
        })
    }

    fn key(slide: u64, chunk: u32) -> ChunkKey {
        ChunkKey {
            slide,
            ifd: 8,
            chunk,
        }
    }

    #[test]
    fn evicts_least_recently_used_chunks_within_the_capacity() {
        let cache = ChunkCache::new(SHARDS * 1000);
        for i in 0..200 {
            cache.insert(key(1, i), chunk(100));
            // Chunk 0 stays in use.
            assert!(cache.get(&key(1, 0)).is_some(), "chunk 0 evicted at {i}");
        }
        assert!(cache.bytes() <= SHARDS * 1000);
        // Too large for a shard: not cached at all.
        cache.insert(key(1, 999), chunk(5000));
        assert!(cache.get(&key(1, 999)).is_none());
        cache.insert(key(2, 0), chunk(100));
        cache.remove_slide(1);
        assert!(cache.get(&key(1, 0)).is_none());
        assert!(cache.get(&key(2, 0)).is_some());
        assert_eq!(cache.bytes(), 100);
        cache.set_capacity(0);
        assert_eq!(cache.bytes(), 0);
        assert_eq!(capacity_for(64 << 20), MIN_CAPACITY);
        assert_eq!(capacity_for(usize::MAX), MAX_CAPACITY);
    }
}
