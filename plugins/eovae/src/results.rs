//! Analysis results in memory, keyed by slide and run (VAE-5).
//!
//! Every tile of a run has a small [`TileState`] (its score, or that it is
//! background or failed). The pixels needed to draw reconstructions and
//! error maps ([`TilePayload`], ~256 KiB per 256-px tile) live in a
//! byte-budgeted LRU cache (VAE-6), so memory stays bounded however much of a
//! slide is analysed; evicted payloads are recomputed (or reloaded from the
//! disk cache) when a view needs them again.

use crate::contract::decode_error;
use crate::tiling::{TileCoord, TileGrid, TileRange};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Revisions are unique across all runs, so a view cached from a run that
/// was dropped and rebuilt (a slide closed and reopened) never matches the
/// rebuilt run by accident.
static REVISION: AtomicU64 = AtomicU64::new(0);

fn next_revision() -> u64 {
    REVISION.fetch_add(1, Ordering::Relaxed) + 1
}

/// Identifies one run's results on one slide.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResultKey {
    /// [`crate::slide::SlideKey`] text.
    pub slide: Arc<str>,
    /// [`crate::run::RunSpec::key`].
    pub run: Arc<str>,
}

impl ResultKey {
    pub fn new(slide: &str, run: &str) -> Self {
        Self {
            slide: Arc::from(slide),
            run: Arc::from(run),
        }
    }
}

/// What is known about one tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TileState {
    /// Scored: mean and largest per-pixel absolute error in [0, 1].
    Analysed { mae: f32, max: f32 },
    /// No tissue (skipped, never inferred).
    Background,
    /// Could not be read.
    Failed,
}

impl TileState {
    pub fn code(self) -> i64 {
        match self {
            Self::Analysed { .. } => 0,
            Self::Background => 1,
            Self::Failed => 2,
        }
    }

    pub fn mae(self) -> Option<f32> {
        match self {
            Self::Analysed { mae, .. } => Some(mae),
            _ => None,
        }
    }
}

/// The pixels of one analysed tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePayload {
    pub size: u32,
    pub valid_w: u32,
    pub valid_h: u32,
    /// `size·size·3` reconstruction bytes.
    pub recon: Vec<u8>,
    /// `size·size` per-pixel errors ([`crate::contract::encode_error`]).
    pub error: Vec<u8>,
}

impl TilePayload {
    pub fn bytes(&self) -> usize {
        self.recon.len() + self.error.len() + std::mem::size_of::<Self>()
    }

    pub fn error_at(&self, x: u32, y: u32) -> f32 {
        decode_error(self.error[(y * self.size + x) as usize])
    }

    /// Compressed form for the disk cache.
    pub fn encode(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(12 + self.recon.len() + self.error.len());
        raw.extend_from_slice(&self.size.to_le_bytes());
        raw.extend_from_slice(&self.valid_w.to_le_bytes());
        raw.extend_from_slice(&self.valid_h.to_le_bytes());
        raw.extend_from_slice(&self.recon);
        raw.extend_from_slice(&self.error);
        miniz_oxide::deflate::compress_to_vec(&raw, 3)
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let raw = miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, 64 << 20).ok()?;
        let read = |at: usize| -> Option<u32> {
            Some(u32::from_le_bytes(raw.get(at..at + 4)?.try_into().ok()?))
        };
        let (size, valid_w, valid_h) = (read(0)?, read(4)?, read(8)?);
        let n = size as usize * size as usize;
        if raw.len() != 12 + n * 4 || valid_w > size || valid_h > size {
            return None;
        }
        Some(Self {
            size,
            valid_w,
            valid_h,
            recon: raw[12..12 + n * 3].to_vec(),
            error: raw[12 + n * 3..].to_vec(),
        })
    }
}

/// Tile counts of a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub analysed: usize,
    pub background: usize,
    pub failed: usize,
}

impl Counts {
    pub fn total(&self) -> usize {
        self.analysed + self.background + self.failed
    }

    fn add(&mut self, state: TileState, sign: isize) {
        let slot = match state {
            TileState::Analysed { .. } => &mut self.analysed,
            TileState::Background => &mut self.background,
            TileState::Failed => &mut self.failed,
        };
        *slot = slot.saturating_add_signed(sign);
    }
}

/// One run's results on one slide.
#[derive(Clone, Debug)]
pub struct RunResults {
    pub grid: TileGrid,
    tiles: HashMap<u64, TileState>,
    counts: Counts,
    /// Increases with every change (unique across runs).
    pub revision: u64,
    /// Whether the disk cache was consulted for this run.
    pub loaded_from_disk: bool,
}

impl RunResults {
    fn new(grid: TileGrid) -> Self {
        Self {
            grid,
            tiles: HashMap::new(),
            counts: Counts::default(),
            revision: next_revision(),
            loaded_from_disk: false,
        }
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }

    pub fn state(&self, coord: TileCoord) -> Option<TileState> {
        self.tiles.get(&coord.key()).copied()
    }

    /// Every tile's state.
    pub fn iter(&self) -> impl Iterator<Item = (TileCoord, TileState)> + '_ {
        self.tiles
            .iter()
            .map(|(key, state)| (TileCoord::from_key(*key), *state))
    }

    fn set(&mut self, coord: TileCoord, state: TileState) {
        if let Some(previous) = self.tiles.insert(coord.key(), state) {
            self.counts.add(previous, -1);
        }
        self.counts.add(state, 1);
        self.revision = next_revision();
    }
}

type PayloadKey = (ResultKey, u64);

/// A byte-budgeted LRU cache of tile payloads.
#[derive(Debug)]
struct PayloadCache {
    budget: usize,
    bytes: usize,
    tick: u64,
    entries: HashMap<PayloadKey, (u64, Arc<TilePayload>)>,
    order: BTreeMap<u64, PayloadKey>,
}

impl PayloadCache {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            tick: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn insert(&mut self, key: PayloadKey, payload: Arc<TilePayload>) {
        let tick = self.next_tick();
        self.bytes += payload.bytes();
        if let Some((old_tick, old)) = self.entries.insert(key.clone(), (tick, payload)) {
            self.order.remove(&old_tick);
            self.bytes -= old.bytes();
        }
        self.order.insert(tick, key);
        self.evict();
    }

    fn get(&mut self, key: &PayloadKey) -> Option<Arc<TilePayload>> {
        let tick = self.next_tick();
        let (entry_tick, payload) = self.entries.get_mut(key)?;
        self.order.remove(entry_tick);
        *entry_tick = tick;
        self.order.insert(tick, key.clone());
        Some(Arc::clone(payload))
    }

    fn contains(&self, key: &PayloadKey) -> bool {
        self.entries.contains_key(key)
    }

    fn evict(&mut self) {
        while self.bytes > self.budget {
            let Some((_, key)) = self.order.pop_first() else {
                break;
            };
            if let Some((_, payload)) = self.entries.remove(&key) {
                self.bytes -= payload.bytes();
            }
        }
    }

    fn retain(&mut self, keep: impl Fn(&ResultKey) -> bool) {
        let removed: Vec<PayloadKey> = self
            .entries
            .keys()
            .filter(|(result, _)| !keep(result))
            .cloned()
            .collect();
        for key in removed {
            if let Some((tick, payload)) = self.entries.remove(&key) {
                self.order.remove(&tick);
                self.bytes -= payload.bytes();
            }
        }
    }

    /// Moves a slide's payloads to another slide key, keeping their place in
    /// the LRU order. A payload already stored under the new key is kept.
    fn rename_slide(&mut self, from: &str, to: &Arc<str>) {
        let moved: Vec<PayloadKey> = self
            .entries
            .keys()
            .filter(|(result, _)| &*result.slide == from)
            .cloned()
            .collect();
        for key in moved {
            let Some((tick, payload)) = self.entries.remove(&key) else {
                continue;
            };
            let renamed = (
                ResultKey {
                    slide: Arc::clone(to),
                    run: key.0.run,
                },
                key.1,
            );
            if self.entries.contains_key(&renamed) {
                self.order.remove(&tick);
                self.bytes -= payload.bytes();
            } else {
                self.order.insert(tick, renamed.clone());
                self.entries.insert(renamed, (tick, payload));
            }
        }
    }
}

/// All results in memory.
#[derive(Debug)]
pub struct ResultStore {
    runs: HashMap<ResultKey, RunResults>,
    payloads: PayloadCache,
    /// Slide keys replaced by [`ResultStore::rename_slide`] and their
    /// current key: work started under the old key (a running job, a load
    /// from disk) lands where the results are shown.
    aliases: HashMap<Arc<str>, Arc<str>>,
}

/// A tile a job must (re)compute, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// Never analysed (or background analysis was not skipped).
    Score,
    /// Scored, but its pixels are needed and not in memory.
    Payload,
}

impl ResultStore {
    pub fn new(payload_budget: usize) -> Self {
        Self {
            runs: HashMap::new(),
            payloads: PayloadCache::new(payload_budget),
            aliases: HashMap::new(),
        }
    }

    /// `key` with a renamed slide replaced by its current key.
    fn resolve<'k>(&self, key: &'k ResultKey) -> Cow<'k, ResultKey> {
        match self.aliases.get(&key.slide) {
            Some(slide) => Cow::Owned(ResultKey {
                slide: Arc::clone(slide),
                run: Arc::clone(&key.run),
            }),
            None => Cow::Borrowed(key),
        }
    }

    pub fn set_payload_budget(&mut self, budget: usize) {
        self.payloads.budget = budget;
        self.payloads.evict();
    }

    pub fn payload_bytes(&self) -> usize {
        self.payloads.bytes
    }

    pub fn run(&self, key: &ResultKey) -> Option<&RunResults> {
        self.runs.get(&*self.resolve(key))
    }

    /// Whether results are stored under exactly `key` (no renamed slides).
    pub fn holds(&self, key: &ResultKey) -> bool {
        self.runs.contains_key(key)
    }

    /// The run's results, created empty for `grid` if new (or if the grid
    /// changed).
    pub fn ensure_run(&mut self, key: &ResultKey, grid: TileGrid) -> &mut RunResults {
        let key = self.resolve(key).into_owned();
        let replace = self.runs.get(&key).is_some_and(|run| run.grid != grid);
        if replace {
            self.runs.remove(&key);
            self.payloads.retain(|k| *k != key);
        }
        self.runs
            .entry(key)
            .or_insert_with(|| RunResults::new(grid))
    }

    /// Records a tile; a payload goes to the LRU cache.
    pub fn record(
        &mut self,
        key: &ResultKey,
        grid: TileGrid,
        coord: TileCoord,
        state: TileState,
        payload: Option<Arc<TilePayload>>,
    ) {
        let key = self.resolve(key).into_owned();
        let run = self.ensure_run(&key, grid);
        if !run.grid.contains(coord) {
            return;
        }
        run.set(coord, state);
        if let Some(payload) = payload {
            self.payloads.insert((key, coord.key()), payload);
        }
    }

    /// Records tiles loaded from the disk cache without disturbing tiles
    /// already known (in-memory results are at least as fresh).
    pub fn merge_loaded(
        &mut self,
        key: &ResultKey,
        grid: TileGrid,
        tiles: impl IntoIterator<Item = (TileCoord, TileState)>,
    ) {
        let run = self.ensure_run(key, grid);
        for (coord, state) in tiles {
            if run.grid.contains(coord) && !run.tiles.contains_key(&coord.key()) {
                run.set(coord, state);
            }
        }
        run.loaded_from_disk = true;
    }

    pub fn payload(&mut self, key: &ResultKey, coord: TileCoord) -> Option<Arc<TilePayload>> {
        let key = self.resolve(key).into_owned();
        self.payloads.get(&(key, coord.key()))
    }

    pub fn has_payload(&self, key: &ResultKey, coord: TileCoord) -> bool {
        let key = self.resolve(key).into_owned();
        self.payloads.contains(&(key, coord.key()))
    }

    /// What a job over `tiles` must compute. Background tiles count as done
    /// only when background is skipped (VAE-10); failed tiles are retried.
    pub fn needs(
        &self,
        key: &ResultKey,
        tiles: impl IntoIterator<Item = TileCoord>,
        need_payloads: bool,
        skip_background: bool,
    ) -> Vec<(TileCoord, Need)> {
        let key = self.resolve(key);
        let run = self.runs.get(&*key);
        tiles
            .into_iter()
            .filter_map(|coord| match run.and_then(|run| run.state(coord)) {
                None | Some(TileState::Failed) => Some((coord, Need::Score)),
                Some(TileState::Background) if !skip_background => Some((coord, Need::Score)),
                Some(TileState::Background) => None,
                Some(TileState::Analysed { .. }) => (need_payloads
                    && !self
                        .payloads
                        .contains(&(key.clone().into_owned(), coord.key())))
                .then_some((coord, Need::Payload)),
            })
            .collect()
    }

    /// Scores of a run's analysed tiles inside `range` (all when `None`).
    pub fn scores(&self, key: &ResultKey, range: Option<TileRange>) -> Vec<(f32, TileCoord)> {
        let Some(run) = self.run(key) else {
            return Vec::new();
        };
        run.iter()
            .filter(|(coord, _)| range.is_none_or(|range| range.contains(*coord)))
            .filter_map(|(coord, state)| state.mae().map(|mae| (mae, coord)))
            .collect()
    }

    /// Forgets a slide's results (it was closed; the disk cache keeps them).
    pub fn drop_slide(&mut self, slide: &str) {
        self.runs.retain(|key, _| &*key.slide != slide);
        self.payloads.retain(|key| &*key.slide != slide);
        self.aliases
            .retain(|from, to| &**from != slide && &**to != slide);
    }

    /// Forgets one run.
    pub fn drop_run(&mut self, key: &ResultKey) {
        let key = self.resolve(key).into_owned();
        self.runs.remove(&key);
        self.payloads.retain(|k| *k != key);
    }

    /// Re-keys a slide's results (its identity became known). Results the new
    /// key already has win over the old key's; later work under the old key
    /// is recorded under the new one.
    pub fn rename_slide(&mut self, from: &str, to: &str) {
        if from == to {
            return;
        }
        let to: Arc<str> = Arc::from(to);
        let keys: Vec<ResultKey> = self
            .runs
            .keys()
            .filter(|key| &*key.slide == from)
            .cloned()
            .collect();
        for key in keys {
            let Some(run) = self.runs.remove(&key) else {
                continue;
            };
            let renamed = ResultKey {
                slide: Arc::clone(&to),
                run: key.run,
            };
            match self.runs.get_mut(&renamed) {
                Some(existing) if existing.grid == run.grid => {
                    for (coord, state) in run.iter() {
                        if !existing.tiles.contains_key(&coord.key()) {
                            existing.set(coord, state);
                        }
                    }
                    existing.loaded_from_disk |= run.loaded_from_disk;
                }
                Some(_) => {}
                None => {
                    self.runs.insert(renamed, run);
                }
            }
        }
        self.payloads.rename_slide(from, &to);
        // Single-step aliases: everything that pointed at `from` now points
        // at `to`, and `to` itself is current.
        for target in self.aliases.values_mut() {
            if &**target == from {
                *target = Arc::clone(&to);
            }
        }
        self.aliases.remove(&to);
        self.aliases.insert(Arc::from(from), to);
    }

    pub fn clear(&mut self) {
        self.runs.clear();
        self.payloads.retain(|_| false);
        self.aliases.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::SlideGeometry;
    use eov_plugin_api::protocol::FileInfo;

    fn grid() -> TileGrid {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width: 2560,
            height: 2560,
            ..Default::default()
        });
        TileGrid::new(&geometry, &geometry.levels[0], 256)
    }

    fn payload(fill: u8) -> Arc<TilePayload> {
        Arc::new(TilePayload {
            size: 16,
            valid_w: 16,
            valid_h: 16,
            recon: vec![fill; 16 * 16 * 3],
            error: vec![fill; 16 * 16],
        })
    }

    #[test]
    fn results_are_separated_by_slide_and_run() {
        let mut store = ResultStore::new(1 << 20);
        let a = ResultKey::new("slide-a", "run-1");
        let b = ResultKey::new("slide-b", "run-1");
        let c = TileCoord::new(1, 1);
        store.record(
            &a,
            grid(),
            c,
            TileState::Analysed { mae: 0.1, max: 0.5 },
            Some(payload(1)),
        );
        assert!(
            store.run(&b).is_none(),
            "slide B never sees slide A's tiles"
        );
        assert_eq!(store.needs(&b, [c], true, true), vec![(c, Need::Score)]);
        assert!(store.needs(&a, [c], true, true).is_empty());
        store.drop_slide("slide-a");
        assert!(store.run(&a).is_none());
        assert_eq!(store.payload_bytes(), 0);
    }

    #[test]
    fn payloads_are_evicted_by_bytes_but_scores_stay() {
        let one = payload(0).bytes();
        let mut store = ResultStore::new(one * 3);
        let key = ResultKey::new("s", "r");
        for col in 0..5 {
            store.record(
                &key,
                grid(),
                TileCoord::new(col, 0),
                TileState::Analysed {
                    mae: col as f32,
                    max: 1.0,
                },
                Some(payload(col as u8)),
            );
        }
        assert!(store.payload_bytes() <= one * 3);
        // The oldest payloads went; every score is kept.
        assert!(!store.has_payload(&key, TileCoord::new(0, 0)));
        assert!(store.has_payload(&key, TileCoord::new(4, 0)));
        assert_eq!(store.run(&key).unwrap().counts().analysed, 5);
        // Touching an entry protects it from the next eviction.
        assert!(store.payload(&key, TileCoord::new(2, 0)).is_some());
        store.record(
            &key,
            grid(),
            TileCoord::new(5, 0),
            TileState::Analysed { mae: 0.0, max: 0.0 },
            Some(payload(9)),
        );
        assert!(store.has_payload(&key, TileCoord::new(2, 0)));
        assert!(!store.has_payload(&key, TileCoord::new(3, 0)));
        // A view that needs pixels asks for them again.
        let needs = store.needs(&key, [TileCoord::new(0, 0)], true, true);
        assert_eq!(needs, vec![(TileCoord::new(0, 0), Need::Payload)]);
        assert!(
            store
                .needs(&key, [TileCoord::new(0, 0)], false, true)
                .is_empty()
        );
    }

    #[test]
    fn background_and_failed_tiles_are_not_pending_forever() {
        let mut store = ResultStore::new(1 << 20);
        let key = ResultKey::new("s", "r");
        let bg = TileCoord::new(0, 0);
        let failed = TileCoord::new(1, 0);
        store.record(&key, grid(), bg, TileState::Background, None);
        store.record(&key, grid(), failed, TileState::Failed, None);
        // Background is done when skipped, failed tiles are retried.
        assert_eq!(
            store.needs(&key, [bg, failed], true, true),
            vec![(failed, Need::Score)]
        );
        // With skipping off, background tiles are analysed after all.
        assert_eq!(
            store.needs(&key, [bg], false, false),
            vec![(bg, Need::Score)]
        );
        let counts = store.run(&key).unwrap().counts();
        assert_eq!((counts.background, counts.failed), (1, 1));
        // Upgrading a tile adjusts the counts.
        store.record(
            &key,
            grid(),
            bg,
            TileState::Analysed { mae: 0.2, max: 0.3 },
            None,
        );
        let counts = store.run(&key).unwrap().counts();
        assert_eq!((counts.analysed, counts.background), (1, 0));
    }

    #[test]
    fn loaded_tiles_never_overwrite_fresh_ones() {
        let mut store = ResultStore::new(1 << 20);
        let key = ResultKey::new("s", "r");
        let c = TileCoord::new(3, 3);
        store.record(
            &key,
            grid(),
            c,
            TileState::Analysed { mae: 0.5, max: 0.9 },
            None,
        );
        store.merge_loaded(
            &key,
            grid(),
            [
                (c, TileState::Background),
                (TileCoord::new(4, 4), TileState::Background),
            ],
        );
        let run = store.run(&key).unwrap();
        assert_eq!(
            run.state(c),
            Some(TileState::Analysed { mae: 0.5, max: 0.9 })
        );
        assert_eq!(run.counts().total(), 2);
        assert!(run.loaded_from_disk);
        // Tiles outside the grid are ignored.
        store.record(
            &key,
            grid(),
            TileCoord::new(99, 0),
            TileState::Background,
            None,
        );
        assert_eq!(store.run(&key).unwrap().counts().total(), 2);
    }

    #[test]
    fn slides_can_be_renamed_when_their_identity_arrives() {
        let mut store = ResultStore::new(1 << 20);
        let old = ResultKey::new("fp1:abc", "r");
        let new = ResultKey::new("quickhash1:def", "r");
        store.record(
            &old,
            grid(),
            TileCoord::new(0, 0),
            TileState::Background,
            None,
        );
        let analysed = TileState::Analysed { mae: 0.1, max: 0.5 };
        store.record(
            &old,
            grid(),
            TileCoord::new(1, 0),
            analysed,
            Some(payload(1)),
        );
        // The new key already knows one tile (e.g. a second pane's job).
        let fresh = TileState::Analysed { mae: 0.2, max: 0.6 };
        store.record(&new, grid(), TileCoord::new(1, 0), fresh, None);
        let bytes = store.payload_bytes();

        store.rename_slide("fp1:abc", "quickhash1:def");
        assert!(store.runs.keys().all(|key| &*key.slide == "quickhash1:def"));
        let run = store.run(&new).unwrap();
        assert_eq!(run.counts().total(), 2);
        assert_eq!(run.state(TileCoord::new(1, 0)), Some(fresh));
        // Pixels survive the rename, in the same LRU budget.
        assert!(store.has_payload(&new, TileCoord::new(1, 0)));
        assert_eq!(store.payload_bytes(), bytes);

        // A job started under the old key records where results are shown.
        store.record(&old, grid(), TileCoord::new(2, 0), analysed, None);
        assert_eq!(store.run(&new).unwrap().counts().total(), 3);
        assert!(store.runs.keys().all(|key| &*key.slide == "quickhash1:def"));

        // A second rename keeps the first alias pointing at the current key.
        store.rename_slide("quickhash1:def", "quickhash1:ghi");
        let latest = ResultKey::new("quickhash1:ghi", "r");
        store.record(&old, grid(), TileCoord::new(3, 0), analysed, None);
        assert_eq!(store.run(&latest).unwrap().counts().total(), 4);
        assert_eq!(store.runs.len(), 1);

        store.drop_slide("quickhash1:ghi");
        assert!(store.aliases.is_empty());
        assert!(store.run(&old).is_none());
    }

    #[test]
    fn payloads_round_trip_through_compression() {
        let original = TilePayload {
            size: 8,
            valid_w: 5,
            valid_h: 8,
            recon: (0..192).map(|i| i as u8).collect(),
            error: (0..64).map(|i| (i * 3) as u8).collect(),
        };
        let decoded = TilePayload::decode(&original.encode()).unwrap();
        assert_eq!(decoded, original);
        assert!(TilePayload::decode(b"garbage").is_none());
    }
}
