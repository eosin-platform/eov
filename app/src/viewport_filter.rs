//! Viewport filter chain — manages and applies post-processing filters.
//!
//! The filter chain collects both in-process (FFI/Rust) and remote (gRPC)
//! filters and applies them sequentially to rendered frames.

use parking_lot::RwLock;
use plugin_api::ffi::{PluginVTable, ViewportFilterFFI, ViewportSnapshotFFI};
use plugin_api::viewport_filter::{CpuFrameBuffer, ViewportFilter};
use std::sync::Arc;

use abi_stable::std_types::RString;

/// A registered filter with its source information.
struct FilterEntry {
    /// Unique identifier.
    id: String,
    /// The filter implementation (in-process only).
    filter: Box<dyn ViewportFilter>,
}

/// Manages the chain of viewport filters applied after the render pipeline.
pub(crate) struct ViewportFilterChain {
    filters: Vec<FilterEntry>,
}

impl ViewportFilterChain {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Register an in-process viewport filter.
    pub fn register(&mut self, id: String, filter: Box<dyn ViewportFilter>) {
        self.filters.push(FilterEntry { id, filter });
    }

    /// Toggle a filter on or off by id.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> bool {
        if let Some(entry) = self.filters.iter_mut().find(|f| f.id == id) {
            entry.filter.set_enabled(enabled);
            true
        } else {
            false
        }
    }

    /// Apply all enabled CPU filters to a frame buffer in-place.
    pub fn apply_cpu(
        &self,
        data: &mut [u8],
        width: u32,
        height: u32,
        viewport: Option<&ViewportSnapshotFFI>,
    ) {
        let mut frame = CpuFrameBuffer {
            data,
            width,
            height,
            viewport: viewport.cloned(),
        };
        for entry in &self.filters {
            if entry.filter.enabled() && entry.filter.supports_cpu() {
                entry.filter.apply_cpu(&mut frame);
            }
        }
    }

    /// Check if any filters are enabled and support CPU.
    pub fn has_enabled_cpu_filters(&self) -> bool {
        self.filters
            .iter()
            .any(|f| f.filter.enabled() && f.filter.supports_cpu())
    }

    /// Check if any filters are enabled, support CPU, and require CPU fallback.
    pub fn has_enabled_cpu_only_filters(&self) -> bool {
        self.filters
            .iter()
            .any(|f| f.filter.enabled() && f.filter.supports_cpu() && !f.filter.supports_gpu())
    }

    /// Check if any filters are enabled and support GPU.
    pub fn has_enabled_gpu_filters(&self) -> bool {
        self.filters
            .iter()
            .any(|f| f.filter.enabled() && f.filter.supports_gpu())
    }
}

/// Thread-safe handle to the filter chain.
pub(crate) type SharedFilterChain = Arc<RwLock<ViewportFilterChain>>;

pub(crate) fn new_shared_filter_chain() -> SharedFilterChain {
    Arc::new(RwLock::new(ViewportFilterChain::new()))
}

// ---------------------------------------------------------------------------
// FFI adapter — wraps a PluginVTable into a ViewportFilter
// ---------------------------------------------------------------------------

/// Adapts an FFI plugin vtable into the [`ViewportFilter`] trait so it can be
/// registered in the shared filter chain and called by the render pipeline.
pub(crate) struct FfiViewportFilter {
    /// Copy of the vtable (all function pointers, `Copy`-safe).
    vtable: PluginVTable,
    /// The filter_id this wrapper represents.
    filter_id: String,
    /// Human-readable name.
    filter_name: String,
    /// Cached enabled state (polled from the plugin).
    enabled: bool,
    supports_cpu: bool,
    supports_gpu: bool,
}

impl FfiViewportFilter {
    pub fn new(vtable: PluginVTable, desc: &ViewportFilterFFI) -> Self {
        Self {
            vtable,
            filter_id: desc.filter_id.to_string(),
            filter_name: desc.name.to_string(),
            enabled: desc.enabled,
            supports_cpu: desc.supports_cpu,
            supports_gpu: desc.supports_gpu,
        }
    }
}

// SAFETY: PluginVTable contains only extern "C" fn pointers which are Send+Sync.
unsafe impl Send for FfiViewportFilter {}
unsafe impl Sync for FfiViewportFilter {}

impl ViewportFilter for FfiViewportFilter {
    fn name(&self) -> &str {
        &self.filter_name
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        (self.vtable.set_filter_enabled)(RString::from(&*self.filter_id), enabled);
    }

    fn supports_cpu(&self) -> bool {
        self.supports_cpu
    }

    fn supports_gpu(&self) -> bool {
        self.supports_gpu
    }

    fn apply_cpu(&self, frame: &mut CpuFrameBuffer<'_>) {
        let len = frame.width * frame.height * 4;
        let viewport = frame
            .viewport
            .as_ref()
            .map(|viewport| viewport as *const ViewportSnapshotFFI)
            .unwrap_or(std::ptr::null());
        (self.vtable.apply_filter_cpu)(
            RString::from(&*self.filter_id),
            frame.data.as_mut_ptr(),
            len,
            frame.width,
            frame.height,
            viewport,
        );
    }
}
