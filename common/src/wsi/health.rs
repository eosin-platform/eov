//! Health tracking and recovery of the shared OpenSlide handle.
//!
//! OpenSlide errors are sticky: the first decode or I/O error moves a handle
//! permanently into an error state in which every later call fails. Because
//! all [`WsiFile`](crate::WsiFile) clones of a slide share one handle, a
//! single corrupt tile would otherwise disable the slide for every thread.
//!
//! [`HandleCell`] runs every OpenSlide operation and, when the handle turns
//! out to be poisoned:
//!
//! 1. serialises recovery behind a mutex, so one thread reopens and the
//!    others wait for (or reuse) its result;
//! 2. if the operation itself failed, re-runs it on a fresh, private handle
//!    (a *probe*). Success means the failure was collateral (another
//!    thread's bad read, a transient I/O error); failing again means the
//!    data is bad: the region is remembered (reads of it fail fast with
//!    [`Error::CorruptRegion`]) and the caller gets that error;
//! 3. always replaces the poisoned shared handle, so one bad tile never
//!    disables the slide;
//! 4. limits probes with a token bucket. Without a token the region is
//!    remembered *provisionally*: reads of it fail fast with the transient
//!    [`Error::HandlePoisoned`] for a period that doubles with every
//!    unverified failure (5 s, 10 s, ... up to 5 min), so a bad region cannot
//!    re-poison the handle in a loop, while a region that was only caught up
//!    in someone else's failure reads normally again afterwards. Only a probe
//!    confirms a region as bad;
//! 5. backs off exponentially after failed opens, so a slide on a dead
//!    network share is not hammered; while backing off, reads fail with
//!    [`Error::HandlePoisoned`].

use crate::openslide::OpenSlide;
use crate::{Error, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Probes allowed in a burst (more than a screenful of bad tiles).
const PROBE_BURST: f64 = 32.0;
/// Sustained probe rate (tokens per second).
const PROBE_RATE_PER_SEC: f64 = 16.0;
/// Backoff after the first failed open; doubles per consecutive failure.
const REOPEN_BACKOFF_BASE: Duration = Duration::from_millis(250);
const REOPEN_BACKOFF_MAX: Duration = Duration::from_secs(30);
/// How long a confirmed-bad region fails fast before it is retried.
pub(crate) const BAD_REGION_TTL: Duration = Duration::from_secs(300);
/// Fail-fast period after the first unverified failure of a region; it
/// doubles with every further one, up to [`BAD_REGION_TTL`].
#[cfg(not(test))]
const PROVISIONAL_BASE: Duration = Duration::from_secs(5);
#[cfg(test)]
const PROVISIONAL_BASE: Duration = Duration::from_millis(50);
/// How long a provisional entry's strike count outlives its fail-fast
/// period.
const STRIKE_MEMORY: Duration = Duration::from_secs(60);
/// Bound on remembered bad regions per slide.
const MAX_BAD_REGIONS: usize = 4096;
/// Recovery rounds a single operation may go through.
const MAX_RECOVERY_ROUNDS: usize = 8;
/// Minimum spacing of user-visible warnings per slide.
const WARN_INTERVAL: Duration = Duration::from_secs(5);

/// Identifies an operation that can fail persistently.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum BadKey {
    Region {
        level: u32,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
    },
    AssociatedImage(String),
    IccProfile(Option<String>),
}

impl BadKey {
    fn into_error(self, message: String) -> Error {
        match self {
            BadKey::Region {
                level,
                x,
                y,
                width,
                height,
            } => Error::CorruptRegion {
                level,
                x,
                y,
                width,
                height,
                message,
            },
            BadKey::AssociatedImage(name) => Error::OpenSlide(format!(
                "associated image {name:?} is unreadable: {message}"
            )),
            BadKey::IccProfile(name) => Error::OpenSlide(format!(
                "ICC profile{} is unreadable: {message}",
                name.map(|n| format!(" of {n:?}")).unwrap_or_default()
            )),
        }
    }
}

/// Coarse state of a slide's handle, for status displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealthState {
    /// The handle is usable.
    #[default]
    Healthy,
    /// The handle is poisoned and a reopen is pending or rate-limited.
    Recovering,
    /// Reopening failed (for example the file disappeared or the network
    /// share is down); reads fail until a later reopen succeeds.
    Failing,
}

/// A snapshot of a slide's handle health.
///
/// Cheap to obtain ([`WsiFile::health`](crate::WsiFile::health)); a UI should
/// poll it and tell the user when `poison_events` or `bad_regions` grow.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlideHealth {
    pub state: HealthState,
    /// How often an operation failed and poisoned the shared handle.
    pub poison_events: u64,
    /// Successful reopens (published handles).
    pub reopens: u64,
    /// Reopen attempts that failed to open the slide.
    pub failed_reopens: u64,
    /// Regions (or associated images) whose reads currently fail fast,
    /// confirmed or provisional.
    pub bad_regions: usize,
    /// The most recent OpenSlide error message.
    pub last_error: Option<String>,
    /// Incremented every time a fresh handle is published.
    pub generation: u64,
}

pub(crate) struct HandleSlot {
    pub slide: OpenSlide,
    pub generation: u64,
}

struct RecoveryState {
    tokens: f64,
    refilled_at: Instant,
    consecutive_failures: u32,
    retry_at: Option<Instant>,
}

impl RecoveryState {
    fn new() -> Self {
        Self {
            tokens: PROBE_BURST,
            refilled_at: Instant::now(),
            consecutive_failures: 0,
            retry_at: None,
        }
    }

    /// Takes a probe token, or returns how long until one is available.
    fn acquire(&mut self, now: Instant) -> std::result::Result<(), Duration> {
        let elapsed = now
            .saturating_duration_since(self.refilled_at)
            .as_secs_f64();
        self.tokens = (self.tokens + elapsed * PROBE_RATE_PER_SEC).min(PROBE_BURST);
        self.refilled_at = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            Err(Duration::from_secs_f64(
                (1.0 - self.tokens) / PROBE_RATE_PER_SEC,
            ))
        }
    }

    fn note_open_failure(&mut self, now: Instant) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let factor = 1u32 << self.consecutive_failures.saturating_sub(1).min(16);
        let backoff = REOPEN_BACKOFF_BASE
            .saturating_mul(factor)
            .min(REOPEN_BACKOFF_MAX);
        self.retry_at = Some(now + backoff);
    }

    fn note_open_success(&mut self) {
        self.consecutive_failures = 0;
        self.retry_at = None;
    }

    /// Whether the failure backoff allows an open now.
    fn backoff_elapsed(&self, now: Instant) -> bool {
        self.retry_at.is_none_or(|retry_at| now >= retry_at)
    }
}

#[derive(Default)]
struct Counters {
    poison_events: AtomicU64,
    reopens: AtomicU64,
    failed_reopens: AtomicU64,
}

/// A remembered failure of a region or associated image.
struct BadEntry {
    since: Instant,
    message: String,
    /// Failed on a probe too (permanent), as opposed to provisional.
    confirmed: bool,
    /// Unverified failures so far (provisional entries).
    strikes: u32,
}

impl BadEntry {
    fn fail_fast_period(&self) -> Duration {
        if self.confirmed {
            BAD_REGION_TTL
        } else {
            let factor = 1u32 << self.strikes.saturating_sub(1).min(16);
            PROVISIONAL_BASE.saturating_mul(factor).min(BAD_REGION_TTL)
        }
    }

    /// Whether reads of the region currently fail fast.
    fn failing_fast(&self, now: Instant) -> bool {
        now.duration_since(self.since) < self.fail_fast_period()
    }

    /// Whether the entry can be dropped.
    fn forgotten(&self, now: Instant) -> bool {
        let keep = if self.confirmed {
            BAD_REGION_TTL
        } else {
            self.fail_fast_period() + STRIKE_MEMORY
        };
        now.duration_since(self.since) >= keep
    }
}

/// Shape of the slide, used to reject a reopened handle if the file was
/// replaced by a different slide.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Geometry {
    pub levels: Vec<(u64, u64)>,
}

impl Geometry {
    pub(crate) fn of(slide: &OpenSlide) -> Result<Self> {
        let count = slide.level_count()?;
        let levels = (0..count)
            .map(|level| slide.level_dimensions(level))
            .collect::<Result<_>>()?;
        Ok(Self { levels })
    }
}

/// The shared handle of one slide, with recovery.
pub(crate) struct HandleCell {
    path: PathBuf,
    geometry: Geometry,
    current: RwLock<Arc<HandleSlot>>,
    recovery: Mutex<RecoveryState>,
    counters: Counters,
    bad: Mutex<HashMap<BadKey, BadEntry>>,
    /// Whether `bad` may hold provisional entries, so successful reads only
    /// lock it when there is something to clear.
    has_provisional: AtomicBool,
    last_error: Mutex<Option<String>>,
    last_warning: Mutex<Option<Instant>>,
    failing: AtomicBool,
    /// Called on every handle this cell opens (attaches the shared cache).
    on_open: fn(&OpenSlide),
}

enum Recovered<T> {
    Done(T),
    Retry(Arc<HandleSlot>),
}

impl HandleCell {
    pub(crate) fn new(
        path: PathBuf,
        slide: OpenSlide,
        geometry: Geometry,
        on_open: fn(&OpenSlide),
    ) -> Self {
        on_open(&slide);
        Self {
            path,
            geometry,
            current: RwLock::new(Arc::new(HandleSlot {
                slide,
                generation: 0,
            })),
            recovery: Mutex::new(RecoveryState::new()),
            counters: Counters::default(),
            bad: Mutex::new(HashMap::new()),
            has_provisional: AtomicBool::new(false),
            last_error: Mutex::new(None),
            last_warning: Mutex::new(None),
            failing: AtomicBool::new(false),
            on_open,
        }
    }

    /// The currently published handle.
    pub(crate) fn current(&self) -> Arc<HandleSlot> {
        Arc::clone(&self.current.read())
    }

    pub(crate) fn health(&self) -> SlideHealth {
        let current = self.current();
        let failing = self.failing.load(Ordering::Relaxed);
        SlideHealth {
            state: if failing {
                HealthState::Failing
            } else if current.slide.is_poisoned() {
                HealthState::Recovering
            } else {
                HealthState::Healthy
            },
            poison_events: self.counters.poison_events.load(Ordering::Relaxed),
            reopens: self.counters.reopens.load(Ordering::Relaxed),
            failed_reopens: self.counters.failed_reopens.load(Ordering::Relaxed),
            bad_regions: self.bad_count(),
            last_error: self.last_error.lock().clone(),
            generation: current.generation,
        }
    }

    fn bad_count(&self) -> usize {
        let now = Instant::now();
        let mut bad = self.bad.lock();
        bad.retain(|_, entry| !entry.forgotten(now));
        bad.values().filter(|entry| entry.failing_fast(now)).count()
    }

    pub(crate) fn clear_bad_regions(&self) {
        self.bad.lock().clear();
        self.has_provisional.store(false, Ordering::Relaxed);
    }

    /// The error to fail fast with if `key` is remembered as bad.
    fn remembered_failure(&self, key: &BadKey) -> Option<Error> {
        let now = Instant::now();
        let mut bad = self.bad.lock();
        let entry = bad.get(key)?;
        if entry.forgotten(now) {
            bad.remove(key);
            return None;
        }
        if !entry.failing_fast(now) {
            return None;
        }
        Some(if entry.confirmed {
            key.clone().into_error(entry.message.clone())
        } else {
            Error::HandlePoisoned(format!(
                "{} (failed recently; retrying later)",
                entry.message
            ))
        })
    }

    /// Records a failure of `key`: confirmed (a probe failed too) or one
    /// more unverified strike.
    fn remember(&self, key: &Option<BadKey>, message: &str, confirmed: bool) {
        let Some(key) = key else {
            return;
        };
        let now = Instant::now();
        let mut bad = self.bad.lock();
        let strikes = match bad.get(key) {
            Some(entry) if !entry.confirmed && !entry.forgotten(now) => entry.strikes + 1,
            _ => 1,
        };
        if bad.len() >= MAX_BAD_REGIONS
            && !bad.contains_key(key)
            && let Some(oldest) = bad
                .iter()
                .min_by_key(|(_, entry)| entry.since)
                .map(|(key, _)| key.clone())
        {
            bad.remove(&oldest);
        }
        bad.insert(
            key.clone(),
            BadEntry {
                since: now,
                message: message.to_string(),
                confirmed,
                strikes,
            },
        );
        if !confirmed {
            self.has_provisional.store(true, Ordering::Relaxed);
        }
    }

    /// After a successful operation: forget unverified strikes against it
    /// (it only failed because someone else poisoned the handle).
    fn clear_provisional(&self, key: &Option<BadKey>) {
        let Some(key) = key else {
            return;
        };
        if !self.has_provisional.load(Ordering::Relaxed) {
            return;
        }
        let mut bad = self.bad.lock();
        if bad.get(key).is_some_and(|entry| !entry.confirmed) {
            bad.remove(key);
        }
        if bad.values().all(|entry| entry.confirmed) {
            self.has_provisional.store(false, Ordering::Relaxed);
        }
    }

    fn corrupt_error(key: &Option<BadKey>, message: String) -> Error {
        match key {
            Some(key) => key.clone().into_error(message),
            None => Error::OpenSlide(message),
        }
    }

    fn deferred_error(&self, message: &str) -> Error {
        Error::HandlePoisoned(format!(
            "{}: read failed ({message}); retrying later",
            self.path.display()
        ))
    }

    fn record_error(&self, message: &str) {
        *self.last_error.lock() = Some(message.to_string());
        // Log the file name only: full paths often contain accession numbers.
        let name = self
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let now = Instant::now();
        let mut last = self.last_warning.lock();
        if last.is_none_or(|at| now.duration_since(at) >= WARN_INTERVAL) {
            *last = Some(now);
            warn!("OpenSlide reported an error for {name}: {message} (recovering)");
        } else {
            debug!("OpenSlide error for {name}: {message}");
        }
    }

    fn open_fresh(&self) -> Result<OpenSlide> {
        let slide = OpenSlide::open(&self.path)?;
        let geometry = Geometry::of(&slide)?;
        if geometry != self.geometry {
            return Err(Error::OpenFile(format!(
                "{} changed on disk since it was opened; reopen the slide",
                self.path.display()
            )));
        }
        (self.on_open)(&slide);
        Ok(slide)
    }

    /// Opens a fresh handle for recovery, recording failures for backoff.
    fn open_for_recovery(&self, recovery: &mut RecoveryState, now: Instant) -> Result<OpenSlide> {
        match self.open_fresh() {
            Ok(slide) => {
                recovery.note_open_success();
                Ok(slide)
            }
            Err(err) => {
                recovery.note_open_failure(now);
                self.counters.failed_reopens.fetch_add(1, Ordering::Relaxed);
                self.failing.store(true, Ordering::Relaxed);
                let message = format!("reopening failed: {err}");
                self.record_error(&message);
                Err(Error::HandlePoisoned(format!(
                    "{}: {message}",
                    self.path.display()
                )))
            }
        }
    }

    fn publish(&self, slide: OpenSlide) -> Arc<HandleSlot> {
        let mut current = self.current.write();
        let slot = Arc::new(HandleSlot {
            slide,
            generation: current.generation + 1,
        });
        *current = Arc::clone(&slot);
        self.counters.reopens.fetch_add(1, Ordering::Relaxed);
        self.failing.store(false, Ordering::Relaxed);
        slot
    }

    /// Runs `op` on the shared handle, recovering from a poisoned handle as
    /// described in the [module documentation](self).
    ///
    /// `op` must report a poisoned handle as [`Error::HandlePoisoned`] (all
    /// [`OpenSlide`] methods do) and must be safe to run more than once.
    pub(crate) fn run<T>(
        &self,
        key: Option<BadKey>,
        mut op: impl FnMut(&OpenSlide) -> Result<T>,
    ) -> Result<T> {
        if let Some(key) = &key
            && let Some(err) = self.remembered_failure(key)
        {
            return Err(err);
        }

        let mut slot = self.current();
        // Whether `op` itself failed on `slot`, as opposed to finding it
        // already poisoned by someone else.
        let mut attempted = false;
        for _ in 0..MAX_RECOVERY_ROUNDS {
            if !attempted && !slot.slide.is_poisoned() {
                match op(&slot.slide) {
                    Ok(value) => {
                        self.clear_provisional(&key);
                        return Ok(value);
                    }
                    Err(Error::HandlePoisoned(message)) => {
                        self.counters.poison_events.fetch_add(1, Ordering::Relaxed);
                        self.record_error(&message);
                        attempted = true;
                    }
                    Err(other) => return Err(other),
                }
            }

            match self.recover(&slot, attempted, &key, &mut op)? {
                Recovered::Done(value) => {
                    self.clear_provisional(&key);
                    return Ok(value);
                }
                Recovered::Retry(fresh) => {
                    slot = fresh;
                    attempted = false;
                }
            }
        }
        Err(Error::HandlePoisoned(format!(
            "{}: the slide handle keeps failing",
            self.path.display()
        )))
    }

    fn recover<T>(
        &self,
        failed: &Arc<HandleSlot>,
        attempted: bool,
        key: &Option<BadKey>,
        op: &mut impl FnMut(&OpenSlide) -> Result<T>,
    ) -> Result<Recovered<T>> {
        let mut recovery = self.recovery.lock();
        let now = Instant::now();

        // Another thread may have replaced the handle while we waited.
        let current = self.current();
        if current.generation != failed.generation && !current.slide.is_poisoned() {
            if !attempted {
                return Ok(Recovered::Retry(current));
            }
            // Our own operation failed, but someone else already published a
            // healthy handle. Probe on a private handle: if the data is bad,
            // retrying on the shared one would poison it again.
            if !recovery.backoff_elapsed(now) {
                return Err(self.deferred_error("reopening is backing off"));
            }
            if recovery.acquire(now).is_err() {
                self.remember(key, "probe deferred", false);
                return Err(self.deferred_error("probe deferred"));
            }
            let probe = match self.open_fresh() {
                Ok(probe) => probe,
                Err(err) => {
                    recovery.note_open_failure(now);
                    self.counters.failed_reopens.fetch_add(1, Ordering::Relaxed);
                    return Err(self.deferred_error(&format!("reopening failed: {err}")));
                }
            };
            recovery.note_open_success();
            drop(recovery);
            return match op(&probe) {
                Ok(value) => Ok(Recovered::Done(value)),
                Err(Error::HandlePoisoned(message)) => {
                    self.remember(key, &message, true);
                    Err(Self::corrupt_error(key, message))
                }
                Err(other) => Err(other),
            };
        }

        // The shared handle is poisoned; every path below replaces it,
        // unless opens are backing off after failures.
        if !recovery.backoff_elapsed(now) {
            return Err(Error::HandlePoisoned(format!(
                "{}: reopening is backing off after failures",
                self.path.display()
            )));
        }

        if !attempted {
            // Our operation never ran; publish and let the caller run it.
            let fresh = self.open_for_recovery(&mut recovery, now)?;
            return Ok(Recovered::Retry(self.publish(fresh)));
        }

        if recovery.acquire(now).is_err() {
            // No probe budget: remember the region provisionally so it cannot
            // re-poison the handle in a loop, and restore the handle for
            // everyone else.
            self.remember(key, "probe deferred", false);
            let fresh = self.open_for_recovery(&mut recovery, now)?;
            self.publish(fresh);
            return Err(self.deferred_error("probe deferred"));
        }

        // Probe: run the failed operation on the fresh, unpublished handle.
        let fresh = self.open_for_recovery(&mut recovery, now)?;
        match op(&fresh) {
            Ok(value) => {
                self.publish(fresh);
                Ok(Recovered::Done(value))
            }
            Err(Error::HandlePoisoned(message)) => {
                // It fails on a fresh handle too: the data is bad.
                drop(fresh);
                self.remember(key, &message, true);
                if let Ok(clean) = self.open_for_recovery(&mut recovery, Instant::now()) {
                    self.publish(clean);
                }
                Err(Self::corrupt_error(key, message))
            }
            Err(other) => {
                // A non-OpenSlide failure on a healthy fresh handle.
                if !fresh.is_poisoned() {
                    self.publish(fresh);
                }
                Err(other)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffSpec, write_tiff};

    #[test]
    fn token_bucket_limits_bursts_and_refills() {
        let mut state = RecoveryState::new();
        let start = Instant::now();
        for _ in 0..32 {
            state.acquire(start).unwrap();
        }
        let wait = state.acquire(start).unwrap_err();
        assert!(wait <= Duration::from_millis(63), "{wait:?}");
        state.acquire(start + Duration::from_millis(70)).unwrap();
    }

    #[test]
    fn failed_opens_back_off_exponentially() {
        let mut state = RecoveryState::new();
        let t0 = Instant::now();
        state.note_open_failure(t0);
        assert!(!state.backoff_elapsed(t0 + Duration::from_millis(100)));
        assert!(state.backoff_elapsed(t0 + Duration::from_millis(300)));
        state.note_open_failure(t0);
        state.note_open_failure(t0);
        // Third failure: 1 s.
        assert!(!state.backoff_elapsed(t0 + Duration::from_millis(900)));
        state.note_open_success();
        assert!(state.backoff_elapsed(t0 + Duration::from_millis(901)));
    }

    #[test]
    fn provisional_strikes_back_off_and_successes_clear_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tiff");
        write_tiff(&path, &TiffSpec::single(64, 64, 32), |_, _| [1, 2, 3]).unwrap();
        let slide = OpenSlide::open(&path).unwrap();
        let geometry = Geometry::of(&slide).unwrap();
        let cell = HandleCell::new(path, slide, geometry, |_| {});
        let key = Some(BadKey::Region {
            level: 0,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });

        cell.remember(&key, "first", false);
        cell.remember(&key, "second", false);
        {
            let bad = cell.bad.lock();
            let entry = bad.get(key.as_ref().unwrap()).unwrap();
            assert_eq!(entry.strikes, 2);
            assert_eq!(entry.fail_fast_period(), PROVISIONAL_BASE * 2);
        }
        assert!(matches!(
            cell.remembered_failure(key.as_ref().unwrap()),
            Some(Error::HandlePoisoned(_))
        ));
        assert_eq!(cell.health().bad_regions, 1);

        // A successful read clears an unverified entry...
        cell.clear_provisional(&key);
        assert!(cell.remembered_failure(key.as_ref().unwrap()).is_none());
        assert!(!cell.has_provisional.load(Ordering::Relaxed));
        // ...but never a confirmed one.
        cell.remember(&key, "probe failed", true);
        cell.clear_provisional(&key);
        assert!(matches!(
            cell.remembered_failure(key.as_ref().unwrap()),
            Some(Error::CorruptRegion { .. })
        ));
    }
}
