//! Viewport filter trait for post-processing rendered frames.
//!
//! Plugins that implement [`ViewportFilter`] can transform the viewport image
//! after the main render pipeline (tile compositing, adjustments, stain norm)
//! has produced a frame. Filters are applied in registration order.
//!
//! Two rendering paths are supported:
//!
//! - **CPU**: The filter receives and returns an RGBA8 pixel buffer.
//! - **GPU**: The filter receives raw Vulkan handles for zero-copy access
//!   to GPU memory. In-process (FFI) plugins get the VkDevice and VkImage
//!   directly; out-of-process (gRPC) plugins receive a DMA-BUF file
//!   descriptor passed over a Unix domain socket.
//!
//! In-process Rust plugins implement this trait directly via `abi_stable`.
//! Out-of-process plugins (Python, etc.) use the gRPC `ExtensionHost` service,
//! where the host acts as a bridge between the render pipeline and the remote
//! filter.

#[cfg(unix)]
use std::os::fd::RawFd;

#[cfg(not(unix))]
type RawFd = std::os::raw::c_int;

/// Describes what an RGBA8 pixel buffer looks like in memory.
pub struct CpuFrameBuffer<'a> {
    /// Raw RGBA8 pixel data, row-major, no padding.
    pub data: &'a mut [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Snapshot of the pane currently being filtered, when available.
    pub viewport: Option<crate::ffi::ViewportSnapshotFFI>,
}

/// Raw Vulkan handles for GPU filter execution (in-process FFI path).
///
/// The host populates this with the Vulkan device and a VkImage backed by
/// exportable memory. The plugin creates its own pipeline on the shared
/// device and submits commands that operate on `vk_image`. The plugin must
/// signal `vk_done_fence` when its GPU work is complete so the host can
/// safely read the result.
#[repr(C)]
pub struct GpuFilterContext {
    /// `VkDevice` handle (as `u64`).
    pub vk_device: u64,
    /// `VkPhysicalDevice` handle (as `u64`).
    pub vk_physical_device: u64,
    /// `VkQueue` handle the plugin may submit to.
    pub vk_queue: u64,
    /// Queue family index for `vk_queue`.
    pub queue_family_index: u32,
    /// `VkImage` handle for the RGBA8 frame.
    pub vk_image: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `VkFormat` value (e.g. `VK_FORMAT_R8G8B8A8_UNORM` = 37).
    pub vk_format: u32,
    /// `VkFence` handle — the plugin **must** signal this when its GPU
    /// commands have finished executing on `vk_image`.
    pub vk_done_fence: u64,
}

/// DMA-BUF descriptor for cross-process GPU memory sharing.
///
/// The host exports the filter texture's backing memory as a DMA-BUF file
/// descriptor and passes it (along with a sync fence fd) to the plugin
/// process over a Unix domain socket. The plugin imports the DMA-BUF,
/// wraps it as a Vulkan image, runs its shader, and signals completion
/// by writing to the done-fence fd.
#[repr(C)]
pub struct DmaBufDescriptor {
    /// DMA-BUF file descriptor for the image memory.
    pub dma_buf_fd: RawFd,
    /// Total size of the DMA-BUF allocation in bytes.
    pub dma_buf_size: u64,
    /// Row stride in bytes.
    pub stride: u32,
    /// DRM pixel format (e.g. `DRM_FORMAT_ABGR8888`).
    pub drm_format: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Sync fence fd — signaled by the host when the image is ready to read.
    /// The plugin must wait on this before accessing the DMA-BUF.
    pub ready_fence_fd: RawFd,
    /// Sync fence fd — the plugin must signal this when its processing is
    /// complete. The host waits on this before reading back the result.
    pub done_fence_fd: RawFd,
}

/// Trait for viewport post-processing filters.
///
/// A filter can support CPU processing, GPU processing, or both. The host
/// calls whichever variant matches the active render backend.
pub trait ViewportFilter: Send + Sync {
    /// Human-readable name for the UI.
    fn name(&self) -> &str;

    /// Whether this filter is currently enabled.
    fn enabled(&self) -> bool;

    /// Toggle the filter on or off.
    fn set_enabled(&mut self, enabled: bool);

    /// Whether this filter supports CPU pixel processing.
    fn supports_cpu(&self) -> bool;

    /// Whether this filter supports GPU texture processing.
    fn supports_gpu(&self) -> bool;

    /// Apply the filter to a CPU RGBA8 buffer **in place**.
    ///
    /// Called only when `supports_cpu()` returns `true` and the render
    /// backend is CPU. The default implementation is a no-op.
    fn apply_cpu(&self, _frame: &mut CpuFrameBuffer<'_>) {}

    /// Apply the filter on the GPU using raw Vulkan handles (FFI path).
    ///
    /// Called only when `supports_gpu()` returns `true` and the render
    /// backend is GPU. The plugin receives the shared `VkDevice` and
    /// `VkImage`, creates/caches its own pipeline, submits GPU commands,
    /// and **must** signal `ctx.vk_done_fence` on completion.
    ///
    /// The default implementation is a no-op that returns `false`.
    fn apply_gpu(&self, _ctx: &GpuFilterContext) -> bool {
        false
    }
}
