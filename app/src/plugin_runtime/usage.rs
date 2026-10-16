//! What each plugin costs (AUDIT §14.4: the Plugins panel shows resource
//! use): the time its event handlers take on the UI thread, the time its
//! viewport filters take on render workers, its host calls and the slide
//! pixels it reads, and, from the overlay store, the layers, objects and
//! raster memory it keeps. Counters run from when the plugin started.
//!
//! Memory and threads a plugin allocates itself cannot be told apart from
//! eov's own in one process, so they are not shown.

use crate::overlay::store::OwnerUsage;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A plugin's counters, updated from the UI thread, render workers and the
/// plugin's own threads.
#[derive(Debug)]
pub struct PluginUsage {
    started: Instant,
    events: AtomicU64,
    event_nanos: AtomicU64,
    /// The slowest event handler so far and the event it handled.
    slowest: Mutex<Option<(Duration, String)>>,
    filter_calls: AtomicU64,
    filter_nanos: AtomicU64,
    host_calls: AtomicU64,
    region_reads: AtomicU64,
    region_bytes: AtomicU64,
}

impl Default for PluginUsage {
    fn default() -> Self {
        Self::new()
    }
}

fn nanos(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

impl PluginUsage {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            events: AtomicU64::new(0),
            event_nanos: AtomicU64::new(0),
            slowest: Mutex::new(None),
            filter_calls: AtomicU64::new(0),
            filter_nanos: AtomicU64::new(0),
            host_calls: AtomicU64::new(0),
            region_reads: AtomicU64::new(0),
            region_bytes: AtomicU64::new(0),
        }
    }

    /// The plugin handled `event` on the UI thread in `elapsed`.
    pub fn record_event(&self, event: &str, elapsed: Duration) {
        self.events.fetch_add(1, Ordering::Relaxed);
        self.event_nanos
            .fetch_add(nanos(elapsed), Ordering::Relaxed);
        let mut slowest = self.slowest.lock();
        if slowest.as_ref().is_none_or(|(time, _)| elapsed > *time) {
            *slowest = Some((elapsed, event.to_string()));
        }
    }

    /// One of the plugin's filters ran over a frame in `elapsed`.
    pub fn record_filter(&self, elapsed: Duration) {
        self.filter_calls.fetch_add(1, Ordering::Relaxed);
        self.filter_nanos
            .fetch_add(nanos(elapsed), Ordering::Relaxed);
    }

    /// The plugin called a host method.
    pub fn record_host_call(&self) {
        self.host_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// The plugin read `bytes` of slide pixels.
    pub fn record_region_read(&self, bytes: u64) {
        self.region_reads.fetch_add(1, Ordering::Relaxed);
        self.region_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        UsageSnapshot {
            running_for: self.started.elapsed(),
            events: self.events.load(Ordering::Relaxed),
            event_time: Duration::from_nanos(self.event_nanos.load(Ordering::Relaxed)),
            slowest_event: self.slowest.lock().clone(),
            filter_calls: self.filter_calls.load(Ordering::Relaxed),
            filter_time: Duration::from_nanos(self.filter_nanos.load(Ordering::Relaxed)),
            host_calls: self.host_calls.load(Ordering::Relaxed),
            region_reads: self.region_reads.load(Ordering::Relaxed),
            region_bytes: self.region_bytes.load(Ordering::Relaxed),
        }
    }
}

/// [`PluginUsage`] at one moment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UsageSnapshot {
    pub running_for: Duration,
    pub events: u64,
    pub event_time: Duration,
    pub slowest_event: Option<(Duration, String)>,
    pub filter_calls: u64,
    pub filter_time: Duration,
    pub host_calls: u64,
    pub region_reads: u64,
    pub region_bytes: u64,
}

impl UsageSnapshot {
    /// One line for the Plugins panel, e.g. "UI thread 1.2 s in 3,410
    /// events (0.4 %; slowest 230 ms, view_changed) · filters 3.4 s over
    /// 1,200 frames · read 1.2 GB of slide pixels · 12,345 objects in 3
    /// layers, 45 MB of rasters".
    pub fn summary(&self, overlays: &OwnerUsage) -> String {
        let mut parts = Vec::new();
        if self.events > 0 {
            let share = if self.running_for.is_zero() {
                0.0
            } else {
                100.0 * self.event_time.as_secs_f64() / self.running_for.as_secs_f64()
            };
            let mut part = format!(
                "UI thread {} in {} {} ({}",
                duration(self.event_time),
                count(self.events),
                plural(self.events, "event", "events"),
                percent(share)
            );
            if let Some((time, event)) = &self.slowest_event
                && *time >= Duration::from_millis(10)
            {
                part.push_str(&format!("; slowest {}, {event}", duration(*time)));
            }
            part.push(')');
            parts.push(part);
        }
        if self.filter_calls > 0 {
            parts.push(format!(
                "filters {} over {} {}",
                duration(self.filter_time),
                count(self.filter_calls),
                plural(self.filter_calls, "frame", "frames")
            ));
        }
        if self.region_reads > 0 {
            parts.push(format!("read {} of slide pixels", bytes(self.region_bytes)));
        }
        if overlays.objects > 0 || overlays.rasters > 0 {
            let mut part = format!(
                "{} {} in {} {}",
                count(overlays.objects as u64),
                plural(overlays.objects as u64, "object", "objects"),
                count(overlays.layers as u64),
                plural(overlays.layers as u64, "layer", "layers"),
            );
            if overlays.raster_bytes > 0 {
                part.push_str(&format!(", {} of rasters", bytes(overlays.raster_bytes)));
            }
            parts.push(part);
        }
        if parts.is_empty() {
            "Nothing measured yet".to_string()
        } else {
            parts.join(" · ")
        }
    }
}

fn plural<'a>(n: u64, one: &'a str, many: &'a str) -> &'a str {
    if n == 1 { one } else { many }
}

/// "1,234,567".
fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// "850 µs", "12 ms", "1.2 s", "3 min 20 s".
fn duration(time: Duration) -> String {
    let secs = time.as_secs_f64();
    if secs < 0.001 {
        format!("{} µs", time.as_micros())
    } else if secs < 1.0 {
        format!("{} ms", time.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1} s")
    } else {
        let whole = time.as_secs();
        format!("{} min {} s", whole / 60, whole % 60)
    }
}

/// "0.4 %", "<0.1 %", "12 %".
fn percent(share: f64) -> String {
    if share > 0.0 && share < 0.1 {
        "<0.1 %".to_string()
    } else if share < 10.0 {
        format!("{share:.1} %")
    } else {
        format!("{share:.0} %")
    }
}

/// "512 KB", "45 MB", "1.2 GB" (decimal units).
fn bytes(n: u64) -> String {
    let n = n as f64;
    if n < 1e6 {
        format!("{:.0} KB", (n / 1e3).max(1.0))
    } else if n < 1e9 {
        format!("{:.0} MB", n / 1e6)
    } else {
        format!("{:.1} GB", n / 1e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_add_up_and_the_slowest_handler_is_kept() {
        let usage = PluginUsage::new();
        usage.record_event("view_changed", Duration::from_millis(4));
        usage.record_event("file_opened", Duration::from_millis(230));
        usage.record_event("view_changed", Duration::from_millis(6));
        usage.record_filter(Duration::from_millis(30));
        usage.record_filter(Duration::from_millis(10));
        usage.record_host_call();
        usage.record_region_read(3_000_000);
        let snapshot = usage.snapshot();
        assert_eq!(snapshot.events, 3);
        assert_eq!(snapshot.event_time, Duration::from_millis(240));
        assert_eq!(
            snapshot.slowest_event,
            Some((Duration::from_millis(230), "file_opened".to_string()))
        );
        assert_eq!(
            (snapshot.filter_calls, snapshot.filter_time),
            (2, Duration::from_millis(40))
        );
        assert_eq!(snapshot.host_calls, 1);
        assert_eq!(
            (snapshot.region_reads, snapshot.region_bytes),
            (1, 3_000_000)
        );
    }

    #[test]
    fn the_summary_names_what_was_used() {
        let snapshot = UsageSnapshot {
            running_for: Duration::from_secs(300),
            events: 3410,
            event_time: Duration::from_millis(1200),
            slowest_event: Some((Duration::from_millis(230), "view_changed".to_string())),
            filter_calls: 1200,
            filter_time: Duration::from_millis(3400),
            host_calls: 50,
            region_reads: 10,
            region_bytes: 1_234_000_000,
        };
        let overlays = OwnerUsage {
            layers: 3,
            objects: 12_345,
            rasters: 2,
            raster_bytes: 45_000_000,
        };
        assert_eq!(
            snapshot.summary(&overlays),
            "UI thread 1.2 s in 3,410 events (0.4 %; slowest 230 ms, view_changed) · \
             filters 3.4 s over 1,200 frames · read 1.2 GB of slide pixels · \
             12,345 objects in 3 layers, 45 MB of rasters"
        );
        // Fast handlers do not name the slowest; idle plugins say so.
        let quick = UsageSnapshot {
            running_for: Duration::from_secs(10),
            events: 1,
            event_time: Duration::from_micros(300),
            slowest_event: Some((Duration::from_micros(300), "file_opened".to_string())),
            ..UsageSnapshot::default()
        };
        assert_eq!(
            quick.summary(&OwnerUsage::default()),
            "UI thread 300 µs in 1 event (<0.1 %)"
        );
        assert_eq!(
            UsageSnapshot::default().summary(&OwnerUsage::default()),
            "Nothing measured yet"
        );
    }
}
