//! Progress reporting and cancellation for long-running jobs (export,
//! dataset extraction) that run on worker threads.
//!
//! A [`JobControl`] is shared (usually via `Arc`) between the worker and
//! the UI: the worker advances the counters and polls
//! [`is_cancelled`](JobControl::is_cancelled); the UI reads
//! [`progress`](JobControl::progress) and may call
//! [`cancel`](JobControl::cancel). All operations are lock-free.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Cancellation flag plus `done / total` work counters.
#[derive(Debug, Default)]
pub struct JobControl {
    cancelled: AtomicBool,
    done: AtomicU64,
    total: AtomicU64,
}

/// A consistent-enough snapshot of a job's progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JobProgress {
    pub done: u64,
    pub total: u64,
}

impl JobProgress {
    /// Completed fraction in `[0, 1]` (0 while the total is unknown).
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32
        }
    }
}

impl JobControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. The worker stops at its next check point.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Sets the amount of work (in arbitrary units, e.g. rows).
    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Records `amount` units of completed work.
    pub fn advance(&self, amount: u64) {
        self.done.fetch_add(amount, Ordering::Relaxed);
    }

    /// Resets the counters (not the cancellation flag).
    pub fn reset(&self) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
    }

    pub fn progress(&self) -> JobProgress {
        JobProgress {
            done: self.done.load(Ordering::Relaxed),
            total: self.total.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_cancellation() {
        let job = JobControl::new();
        assert_eq!(job.progress().fraction(), 0.0);
        job.set_total(10);
        job.advance(4);
        assert_eq!(job.progress(), JobProgress { done: 4, total: 10 });
        assert!((job.progress().fraction() - 0.4).abs() < 1e-6);
        assert!(!job.is_cancelled());
        job.cancel();
        assert!(job.is_cancelled());
    }
}
