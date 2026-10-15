//! Plugin viewport filters as seen by the renderer.
//!
//! The UI thread publishes the ordered list of enabled filters; render
//! workers read it lock-free and call the plugins without holding any host
//! lock (PLG-3, PLG-12). Failures are collected for the UI thread, which
//! disables filters that keep failing and quarantines plugins that panic.

use super::contributions::Contributions;
use abi_stable::std_types::{RSliceMut, RStr};
use arc_swap::ArcSwap;
use eov_plugin_api::abi::{FfiError, FfiErrorKind, FilterFrame, FilterView, PluginBox};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Consecutive failures after which a filter is switched off.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;

pub struct ActiveFilter {
    pub plugin_id: Arc<str>,
    pub filter_id: String,
    instance: Arc<PluginBox>,
    failures: AtomicU32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterFailure {
    pub plugin_id: Arc<str>,
    pub filter_id: String,
    pub kind: FfiErrorKind,
    pub message: String,
    /// The filter has failed often enough to be disabled.
    pub give_up: bool,
}

/// The enabled filters in execution order.
#[derive(Default)]
pub struct ChainSnapshot {
    filters: Vec<ActiveFilter>,
    sink: Arc<Mutex<Vec<FilterFailure>>>,
}

impl ChainSnapshot {
    pub fn has_enabled_cpu_filters(&self) -> bool {
        !self.filters.is_empty()
    }

    /// Ids (`plugin:filter`) in execution order.
    #[cfg(test)]
    pub fn ids(&self) -> Vec<String> {
        self.filters
            .iter()
            .map(|f| format!("{}:{}", f.plugin_id, f.filter_id))
            .collect()
    }

    /// Runs every filter over a tightly packed RGBA frame, in order. A
    /// failing filter leaves the frame as it was before that filter ran.
    pub fn apply_cpu(&self, pixels: &mut [u8], width: u32, height: u32, view: Option<&FilterView>) {
        let stride = width * 4;
        let expected = stride as usize * height as usize;
        if pixels.len() < expected || width == 0 || height == 0 {
            return;
        }
        let view = view.copied().unwrap_or_default();
        let mut backup: Option<Vec<u8>> = None;
        for filter in &self.filters {
            if filter.failures.load(Ordering::Relaxed) >= MAX_CONSECUTIVE_FAILURES {
                continue;
            }
            // Keep a copy only when a previous run of this filter failed, so
            // the common path stays copy-free.
            let guard_copy = filter.failures.load(Ordering::Relaxed) > 0;
            if guard_copy {
                backup.get_or_insert_with(Vec::new).clear();
                backup
                    .as_mut()
                    .expect("backup")
                    .extend_from_slice(&pixels[..expected]);
            }
            let result = filter.instance.apply_filter(
                RStr::from_str(&filter.filter_id),
                FilterFrame {
                    pixels: RSliceMut::from_mut_slice(&mut pixels[..expected]),
                    width,
                    height,
                    stride,
                    view,
                },
            );
            match result.into_result() {
                Ok(()) => {
                    filter.failures.store(0, Ordering::Relaxed);
                }
                Err(error) => {
                    if let Some(copy) = &backup
                        && guard_copy
                    {
                        pixels[..expected].copy_from_slice(copy);
                    }
                    let count = filter.failures.fetch_add(1, Ordering::Relaxed) + 1;
                    self.report(filter, error, count);
                }
            }
        }
    }

    fn report(&self, filter: &ActiveFilter, error: FfiError, count: u32) {
        let kind = error.kind();
        self.sink.lock().push(FilterFailure {
            plugin_id: Arc::clone(&filter.plugin_id),
            filter_id: filter.filter_id.clone(),
            kind,
            message: error.message.into_string(),
            give_up: kind == FfiErrorKind::Panicked || count >= MAX_CONSECUTIVE_FAILURES,
        });
    }
}

/// The renderer's handle to the current filter list.
pub struct FilterChain {
    current: ArcSwap<ChainSnapshot>,
    sink: Arc<Mutex<Vec<FilterFailure>>>,
}

impl Default for FilterChain {
    fn default() -> Self {
        let sink = Arc::new(Mutex::new(Vec::new()));
        Self {
            current: ArcSwap::from_pointee(ChainSnapshot {
                filters: Vec::new(),
                sink: Arc::clone(&sink),
            }),
            sink,
        }
    }
}

pub type SharedFilterChain = Arc<FilterChain>;

impl FilterChain {
    /// The current filters; never blocks.
    pub fn read(&self) -> Arc<ChainSnapshot> {
        self.current.load_full()
    }

    /// Rebuilds the list from the registrations (UI thread).
    pub fn rebuild(
        &self,
        contributions: &Contributions,
        instances: &HashMap<Arc<str>, Arc<PluginBox>>,
    ) {
        let filters = contributions
            .filters()
            .into_iter()
            .filter(|entry| entry.spec.enabled)
            .filter_map(|entry| {
                instances
                    .get(&entry.plugin_id)
                    .map(|instance| ActiveFilter {
                        plugin_id: Arc::clone(&entry.plugin_id),
                        filter_id: entry.spec.filter_id.clone(),
                        instance: Arc::clone(instance),
                        failures: AtomicU32::new(0),
                    })
            })
            .collect();
        self.current.store(Arc::new(ChainSnapshot {
            filters,
            sink: Arc::clone(&self.sink),
        }));
    }

    pub fn take_failures(&self) -> Vec<FilterFailure> {
        std::mem::take(&mut *self.sink.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::{FilterSpec, PluginInit};
    use eov_plugin_api::sdk::{Frame, Host, Plugin, create_plugin};
    use eov_plugin_api::testing::{MockHost, test_init};

    struct Filters;

    impl Plugin for Filters {
        fn create(_host: Host, _init: PluginInit) -> Result<Self, String> {
            Ok(Self)
        }

        fn apply_filter(&self, filter_id: &str, frame: &mut Frame<'_>) -> Result<(), String> {
            match filter_id {
                "invert" => {
                    frame.for_each_pixel(|_, _, px| {
                        px[0] = 255 - px[0];
                    });
                    Ok(())
                }
                "plus-one" => {
                    frame.for_each_pixel(|_, _, px| px[0] = px[0].wrapping_add(1));
                    Ok(())
                }
                "scribble-then-fail" => {
                    frame.for_each_pixel(|_, _, px| px[0] = 7);
                    Err("broken".into())
                }
                "panic" => panic!("filter bug"),
                _ => Ok(()),
            }
        }
    }

    fn chain_with(ids: &[(&str, i32)]) -> FilterChain {
        let host = MockHost::new();
        let init = serde_json::to_string(&test_init("f")).unwrap();
        let instance = Arc::new(create_plugin::<Filters>(host.to_ffi(), &init).unwrap());
        let mut contributions = Contributions::default();
        let plugin: Arc<str> = Arc::from("f");
        for (id, priority) in ids {
            contributions
                .register_filter(
                    &plugin,
                    FilterSpec {
                        filter_id: id.to_string(),
                        name: id.to_string(),
                        priority: *priority,
                        enabled: true,
                    },
                )
                .unwrap();
        }
        let chain = FilterChain::default();
        chain.rebuild(&contributions, &HashMap::from([(plugin, instance)]));
        chain
    }

    #[test]
    fn filters_run_in_priority_order() {
        let chain = chain_with(&[("plus-one", 1), ("invert", 0)]);
        let mut pixels = vec![10, 0, 0, 255];
        chain.read().apply_cpu(&mut pixels, 1, 1, None);
        // invert first (245), then +1.
        assert_eq!(pixels[0], 246);
        assert_eq!(chain.read().ids(), ["f:invert", "f:plus-one"]);
    }

    #[test]
    fn repeated_failures_disable_a_filter_and_restore_pixels() {
        let chain = chain_with(&[("scribble-then-fail", 0)]);
        let snapshot = chain.read();
        let mut pixels = vec![10, 0, 0, 255];
        snapshot.apply_cpu(&mut pixels, 1, 1, None);
        // The first failure is not guarded (no copy taken yet).
        assert_eq!(pixels[0], 7);
        let mut pixels = vec![10, 0, 0, 255];
        snapshot.apply_cpu(&mut pixels, 1, 1, None);
        assert_eq!(pixels[0], 10, "later failures leave the frame untouched");
        snapshot.apply_cpu(&mut pixels, 1, 1, None);
        let failures = chain.take_failures();
        assert_eq!(failures.len(), 3);
        assert!(failures[2].give_up);
        // Disabled now.
        snapshot.apply_cpu(&mut pixels, 1, 1, None);
        assert!(chain.take_failures().is_empty());
    }

    #[test]
    fn panics_are_reported_for_quarantine() {
        let chain = chain_with(&[("panic", 0)]);
        let mut pixels = vec![0, 0, 0, 255];
        chain.read().apply_cpu(&mut pixels, 1, 1, None);
        let failures = chain.take_failures();
        assert_eq!(failures[0].kind, FfiErrorKind::Panicked);
        assert!(failures[0].give_up);
    }

    #[test]
    fn short_buffers_are_skipped() {
        let chain = chain_with(&[("invert", 0)]);
        let mut pixels = vec![0, 0, 0];
        chain.read().apply_cpu(&mut pixels, 1, 1, None);
        assert_eq!(pixels, vec![0, 0, 0]);
    }
}
