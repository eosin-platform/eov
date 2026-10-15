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
//! 2. opens a *fresh* handle and, if the failing operation had actually run,
//!    re-runs it on that handle **before publishing it**. Success means the
//!    failure was collateral (another thread's bad read, a transient I/O
//!    error) and the fresh handle replaces the poisoned one. A second
//!    failure means the data itself is bad: the region is remembered in a
//!    bad-region list, a clean handle is published, and the caller gets
//!    [`Error::CorruptRegion`] — the shared handle never stays poisoned
//!    because of one bad tile;
//! 3. rate-limits reopening with a token bucket plus exponential backoff
//!    after failed opens, so a slide on a dead network share is not hammered.
//!    While a reopen is not allowed callers get the transient
//!    [`Error::HandlePoisoned`].

use crate::openslide::OpenSlide;
use crate::{Error, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Reopen attempts allowed in a burst.
const REOPEN_BURST: f64 = 4.0;
/// Sustained reopen rate (tokens per second).
const REOPEN_RATE_PER_SEC: f64 = 4.0;
/// Backoff after the first failed open; doubles per consecutive failure.
const REOPEN_BACKOFF_BASE: Duration = Duration::from_millis(250);
const REOPEN_BACKOFF_MAX: Duration = Duration::from_secs(30);
/// How long a confirmed-bad region fails fast before it is retried.
pub(crate) const BAD_REGION_TTL: Duration = Duration::from_secs(300);
/// Bound on remembered bad regions per slide.
const MAX_BAD_REGIONS: usize = 4096;
/// Minimum spacing of user-visible warnings per slide.
const WARN_INTERVAL: Duration = Duration::from_secs(5);

/// Identifies an operation that failed on two handles in a row.
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
            BadKey::AssociatedImage(name) => {
                Error::OpenSlide(format!("associated image {name:?} is unreadable: {message}"))
            }
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
    /// How often the shared handle entered OpenSlide's error state.
    pub poison_events: u64,
    /// Successful reopens.
    pub reopens: u64,
    /// Reopen attempts that failed to open the slide.
    pub failed_reopens: u64,
    /// Regions (or associated images) currently known to be unreadable.
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
            tokens: REOPEN_BURST,
            refilled_at: Instant::now(),
            consecutive_failures: 0,
            retry_at: None,
        }
    }

    /// Takes a reopen token, or returns how long to wait.
    fn acquire(&mut self, now: Instant) -> std::result::Result<(), Duration> {
        if let Some(retry_at) = self.retry_at
            && now < retry_at
        {
            return Err(retry_at - now);
        }
        let elapsed = now.saturating_duration_since(self.refilled_at).as_secs_f64();
        self.tokens = (self.tokens + elapsed * REOPEN_RATE_PER_SEC).min(REOPEN_BURST);
        self.refilled_at = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            Err(Duration::from_secs_f64(
                (1.0 - self.tokens) / REOPEN_RATE_PER_SEC,
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
}

#[derive(Default)]
struct Counters {
    poison_events: AtomicU64,
    reopens: AtomicU64,
    failed_reopens: AtomicU64,
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
    bad: Mutex<HashMap<BadKey, (Instant, String)>>,
    last_error: Mutex<Option<String>>,
    last_warning: Mutex<Option<Instant>>,
    failing: std::sync::atomic::AtomicBool,
    /// Called on every handle this cell opens (attaches the shared cache).
    on_open: fn(&OpenSlide),
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
            last_error: Mutex::new(None),
            last_warning: Mutex::new(None),
            failing: std::sync::atomic::AtomicBool::new(false),
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
        bad.retain(|_, (since, _)| now.duration_since(*since) < BAD_REGION_TTL);
        bad.len()
    }

    pub(crate) fn clear_bad_regions(&self) {
        self.bad.lock().clear();
    }

    fn bad_message(&self, key: &BadKey) -> Option<String> {
        let mut bad = self.bad.lock();
        match bad.get(key) {
            Some((since, message)) if since.elapsed() < BAD_REGION_TTL => Some(message.clone()),
            Some(_) => {
                bad.remove(key);
                None
            }
            None => None,
        }
    }

    fn mark_bad(&self, key: BadKey, message: String) {
        let mut bad = self.bad.lock();
        if bad.len() >= MAX_BAD_REGIONS
            && let Some(oldest) = bad
                .iter()
                .min_by_key(|(_, (since, _))| *since)
                .map(|(key, _)| key.clone())
        {
            bad.remove(&oldest);
        }
        bad.insert(key, (Instant::now(), message));
    }

    fn record_error(&self, message: &str) {
        *self.last_error.lock() = Some(message.to_string());
        let now = Instant::now();
        let mut last = self.last_warning.lock();
        if last.is_none_or(|at| now.duration_since(at) >= WARN_INTERVAL) {
            *last = Some(now);
            warn!(
                "OpenSlide reported an error for {}: {message} (recovering)",
                self.path.display()
            );
        } else {
            debug!("OpenSlide error for {}: {message}", self.path.display());
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
    /// [`OpenSlide`] methods do) and must be safe to run twice.
    pub(crate) fn run<T>(
        &self,
        key: Option<BadKey>,
        mut op: impl FnMut(&OpenSlide) -> Result<T>,
    ) -> Result<T> {
        if let Some(key) = &key
            && let Some(message) = self.bad_message(key)
        {
            return Err(key.clone().into_error(message));
        }

        let mut slot = self.current();
        // `attempted`: whether `op` already failed on `slot` itself (as
        // opposed to finding the handle poisoned by someone else).
        let mut attempted = false;
        for _ in 0..3 {
            if !attempted {
                if slot.slide.is_poisoned() {
                    // Collateral: poisoned by another operation.
                } else {
                    match op(&slot.slide) {
                        Ok(value) => return Ok(value),
                        Err(Error::HandlePoisoned(message)) => {
                            self.counters.poison_events.fetch_add(1, Ordering::Relaxed);
                            self.record_error(&message);
                            attempted = true;
                        }
                        Err(other) => return Err(other),
                    }
                }
            }

            match self.recover(&slot, attempted, &key, &mut op)? {
                Recovered::Done(value) => return Ok(value),
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

        // Another thread may have replaced the handle while we waited.
        let current = self.current();
        let replaced = current.generation != failed.generation && !current.slide.is_poisoned();
        if replaced && !attempted {
            return Ok(Recovered::Retry(current));
        }

        let now = Instant::now();
        if let Err(wait) = recovery.acquire(now) {
            return Err(Error::HandlePoisoned(format!(
                "{}: reopening is rate-limited, retry in {} ms",
                self.path.display(),
                wait.as_millis().max(1)
            )));
        }

        if replaced {
            // Our own operation failed, but someone else already published a
            // healthy handle. Re-check on a private handle: if the data is
            // bad, retrying on the shared one would poison it again.
            let probe = match self.open_fresh() {
                Ok(probe) => probe,
                Err(err) => {
                    recovery.note_open_failure(now);
                    self.counters.failed_reopens.fetch_add(1, Ordering::Relaxed);
                    return Err(Error::HandlePoisoned(format!(
                        "{}: reopening failed: {err}",
                        self.path.display()
                    )));
                }
            };
            recovery.note_open_success();
            drop(recovery);
            return match op(&probe) {
                Ok(value) => Ok(Recovered::Done(value)),
                Err(Error::HandlePoisoned(message)) => {
                    if let Some(key) = key {
                        self.mark_bad(key.clone(), message.clone());
                    }
                    Err(match key {
                        Some(key) => key.clone().into_error(message),
                        None => Error::OpenSlide(message),
                    })
                }
                Err(other) => Err(other),
            };
        }

        let fresh = match self.open_fresh() {
            Ok(fresh) => fresh,
            Err(err) => {
                recovery.note_open_failure(now);
                self.counters.failed_reopens.fetch_add(1, Ordering::Relaxed);
                self.failing.store(true, Ordering::Relaxed);
                let message = format!("reopening failed: {err}");
                self.record_error(&message);
                return Err(Error::HandlePoisoned(format!(
                    "{}: {message}",
                    self.path.display()
                )));
            }
        };
        recovery.note_open_success();

        if !attempted {
            // Our operation never ran; publish and let the caller run it.
            return Ok(Recovered::Retry(self.publish(fresh)));
        }

        // Probe: run the failed operation on the fresh, unpublished handle.
        match op(&fresh) {
            Ok(value) => {
                self.publish(fresh);
                Ok(Recovered::Done(value))
            }
            Err(Error::HandlePoisoned(message)) => {
                // It fails on a fresh handle too: the data is bad.
                drop(fresh);
                if let Some(key) = key {
                    self.mark_bad(key.clone(), message.clone());
                }
                // Replace the poisoned shared handle with a clean one.
                if recovery.acquire(Instant::now()).is_ok() {
                    match self.open_fresh() {
                        Ok(clean) => {
                            self.publish(clean);
                        }
                        Err(err) => {
                            recovery.note_open_failure(Instant::now());
                            self.counters.failed_reopens.fetch_add(1, Ordering::Relaxed);
                            self.record_error(&format!("reopening failed: {err}"));
                        }
                    }
                }
                Err(match key {
                    Some(key) => key.clone().into_error(message),
                    None => Error::OpenSlide(message),
                })
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

enum Recovered<T> {
    Done(T),
    Retry(Arc<HandleSlot>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bucket_limits_bursts_and_refills() {
        let mut state = RecoveryState::new();
        let start = Instant::now();
        for _ in 0..4 {
            state.acquire(start).unwrap();
        }
        let wait = state.acquire(start).unwrap_err();
        assert!(wait <= Duration::from_millis(250), "{wait:?}");
        state.acquire(start + Duration::from_millis(260)).unwrap();
    }

    #[test]
    fn failed_opens_back_off_exponentially() {
        let mut state = RecoveryState::new();
        let t0 = Instant::now();
        state.note_open_failure(t0);
        assert!(state.acquire(t0 + Duration::from_millis(100)).is_err());
        state.acquire(t0 + Duration::from_millis(300)).unwrap();
        state.note_open_failure(t0);
        state.note_open_failure(t0);
        // Third failure: 1 s.
        assert!(state.acquire(t0 + Duration::from_millis(900)).is_err());
        state.note_open_success();
        state.acquire(t0 + Duration::from_millis(901)).unwrap();
    }
}
