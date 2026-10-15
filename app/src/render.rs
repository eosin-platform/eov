//! Rendering utilities for the WSI viewer
//!
//! This module contains helper functions for rendering tiles and compositing
//! the viewport image.
//!
//! Each backend keeps to its own path: the GPU backend composites on the
//! GPU even when plugin filters (CPU-only) are enabled — the finished frame
//! is read back and filtered on a worker (RND-6) — and the CPU backend
//! composites, processes and filters frames on the render pool. Fallback
//! tiles that finer tiles hide completely are not drawn.

use crate::AppWindow;
use crate::blitter;
use crate::gpu::{QueuedFrame, ReadbackFrame, ReadbackTag, SurfaceSlot, TileDraw};
use crate::render_pool::{
    CachedCpuFrame, CpuBlitCommand, CpuBlitKind, CpuRenderJob, CpuRenderPostProcess,
    CpuRenderResult, FrameFilters,
};
use crate::state::{AppState, OpenFile, PaneId};
use crate::tile_loader::{TileLoader, calculate_wanted_tiles};
use eov_common::render::single_level_trilinear;
use eov_common::{
    FilteringMode, RenderBackend, StainNormalization, TileCache, TileManager, TrilinearLevels,
    Viewport, calculate_trilinear_levels,
};
use parking_lot::RwLock;
use slint::{ComponentHandle, Image};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn trilinear_lod_bias_enabled(filtering_mode: FilteringMode) -> bool {
    filtering_mode == FilteringMode::Trilinear
}

/// Adaptive Lanczos-to-Trilinear blending weight based on zoom level.
fn lanczos_adaptive_weight(zoom: f64) -> f64 {
    const LANCZOS_FULL: f64 = 0.12;
    const TRILINEAR_FULL: f64 = 0.07;
    if zoom >= LANCZOS_FULL {
        1.0
    } else if zoom <= TRILINEAR_FULL {
        0.0
    } else {
        (zoom - TRILINEAR_FULL) / (LANCZOS_FULL - TRILINEAR_FULL)
    }
}

type CoarseBlendData = (Arc<eov_common::TileData>, [f32; 2], [f32; 2], f32);

#[derive(Default)]
struct PaneRenderOutcome {
    image: Option<Image>,
    keep_running: bool,
    rendered: bool,
}

struct RenderPaneRequest<'a> {
    ui: &'a AppWindow,
    tile_cache: &'a Arc<TileCache>,
    target_width: f64,
    target_height: f64,
    force_render: bool,
    filtering_mode: FilteringMode,
    filter_chain: &'a crate::plugin_runtime::filters::SharedFilterChain,
    /// Stable id of the pane (what plugins and render jobs know it by).
    pane_uid: u32,
    /// `AppState::filter_revision` of this frame.
    filter_revision: u64,
    /// Physical pixels to render per logical pixel (RND-3).
    scale: f64,
}

#[derive(Clone)]
struct CpuPaneSnapshot {
    pane: PaneId,
    /// Stable id of the pane (what plugins and render jobs know it by).
    pane_uid: u32,
    file_id: i32,
    file_switched: bool,
    content_missing: bool,
    minimap_missing: bool,
    minimap_image: Option<Image>,
    tile_manager: Arc<TileManager>,
    tile_loader: Arc<TileLoader>,
    viewport_state: eov_common::ViewportState,
    /// The view in physical pixels, which frames are rendered at (RND-3).
    render_viewport: Viewport,
    /// Physical pixels per logical pixel.
    scale: f64,
    frame_count: u32,
    last_render_zoom: f64,
    last_render_center_x: f64,
    last_render_center_y: f64,
    last_render_width: f64,
    last_render_height: f64,
    last_render_level: u32,
    last_render_orientation: eov_common::viewport::Orientation,
    previous_tiles_loaded: u32,
    last_seen_tile_epoch: u64,
    hud_sharpness: f32,
    hud_gamma: f32,
    hud_brightness: f32,
    hud_contrast: f32,
    hud_stain_normalization: StainNormalization,
    hud_deconv_h_intensity: f32,
    hud_deconv_h_visible: bool,
    hud_deconv_e_intensity: f32,
    hud_deconv_e_visible: bool,
    hud_deconv_isolated: crate::state::IsolatedChannel,
    last_render_sharpness: f32,
    last_render_gamma: f32,
    last_render_brightness: f32,
    last_render_contrast: f32,
    last_render_stain_normalization: StainNormalization,
    last_render_deconv_h_intensity: f32,
    last_render_deconv_h_visible: bool,
    last_render_deconv_e_intensity: f32,
    last_render_deconv_e_visible: bool,
    last_render_deconv_isolated: crate::state::IsolatedChannel,
    pending_cpu_job_id: Option<u64>,
    needs_settled_cpu_render: bool,
    /// The pane's colour processing, resolved when the snapshot was taken.
    color: crate::color::PaneColor,
    filtering_mode: FilteringMode,
    filter_revision: u64,
    last_render_filter_revision: u64,
}

struct CpuFrameSnapshot {
    pane_count: usize,
    render_requested: bool,
    cpu_jobs_pending: bool,
    panes: Vec<Option<CpuPaneSnapshot>>,
}

struct CpuRenderCommit {
    frame_update: Option<CpuFrameStateUpdate>,
    pending_cpu_job_id: Option<Option<u64>>,
    needs_settled_cpu_render: Option<bool>,
    /// The filter revision the pane now shows (refiltered cached frame).
    last_render_filter_revision: Option<u64>,
}

struct CpuFrameStateUpdate {
    frame_count: u32,
    last_render_time: std::time::Instant,
    last_render_zoom: f64,
    last_render_center_x: f64,
    last_render_center_y: f64,
    last_render_width: f64,
    last_render_height: f64,
    last_render_level: u32,
    last_render_orientation: eov_common::viewport::Orientation,
    tiles_loaded_since_render: u32,
    last_seen_tile_epoch: u64,
    last_render_gamma: f32,
    last_render_brightness: f32,
    last_render_contrast: f32,
    last_render_sharpness: f32,
    last_render_stain_normalization: StainNormalization,
    last_render_deconv_h_intensity: f32,
    last_render_deconv_h_visible: bool,
    last_render_deconv_e_intensity: f32,
    last_render_deconv_e_visible: bool,
    last_render_deconv_isolated: crate::state::IsolatedChannel,
    last_render_filter_revision: u64,
    gpu_surface_valid: bool,
}

struct CpuPaneExecution {
    pane: PaneId,
    file_id: i32,
    content_missing: bool,
    outcome: PaneRenderOutcome,
    commit: CpuRenderCommit,
}

// ─── Physical-resolution frames (RND-3) ─────────────────────────────────────

/// Largest edge of a CPU-rendered frame, in pixels.
const MAX_CPU_FRAME_DIMENSION: u32 = 16384;

/// The view at the display's physical resolution: the same image region
/// over `scale` times as many pixels in each direction. Frames, levels and
/// tile requests use it; interaction, overlays and what the HUD shows stay
/// in logical pixels.
fn physical_viewport(viewport: &Viewport, scale: f64) -> Viewport {
    Viewport {
        center: viewport.center,
        zoom: viewport.zoom * scale,
        width: (viewport.width * scale).round().max(1.0),
        height: (viewport.height * scale).round().max(1.0),
        image_width: viewport.image_width,
        image_height: viewport.image_height,
        orientation: viewport.orientation,
    }
}

/// Physical pixels per logical pixel to render a `width × height` logical
/// pane with: the window's scale factor, less if the frame would be larger
/// than `max_dimension`.
fn render_scale(scale_factor: f32, width: f64, height: f64, max_dimension: u32) -> f64 {
    let scale = f64::from(scale_factor);
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    scale.min(f64::from(max_dimension) / width.max(height).max(1.0))
}

static FILTER_FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a plugin filter is told about the frame it filters: the view the
/// frame shows (which may be older than the pane's current view), in logical
/// pixels, and the frame's physical pixels per logical pixel.
fn filter_view(
    file_id: i32,
    pane_uid: u32,
    viewport: &Viewport,
    scale: f64,
) -> eov_plugin_api::abi::FilterView {
    let bounds = viewport.bounds();
    eov_plugin_api::abi::FilterView {
        pane: pane_uid,
        file_id: i64::from(file_id),
        center_x: viewport.center.x,
        center_y: viewport.center.y,
        zoom: viewport.zoom / scale,
        width: viewport.width / scale,
        height: viewport.height / scale,
        scale_factor: scale,
        visible_x: bounds.left,
        visible_y: bounds.top,
        visible_width: bounds.width(),
        visible_height: bounds.height(),
        frame: FILTER_FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    }
}

/// The enabled plugin filters for a frame of `viewport` (physical pixels,
/// `scale` per logical pixel), if any.
fn frame_filters(
    filter_chain: &crate::plugin_runtime::filters::SharedFilterChain,
    file_id: i32,
    pane_uid: u32,
    viewport: &Viewport,
    scale: f64,
) -> Option<FrameFilters> {
    let chain = filter_chain.read();
    chain.has_enabled_cpu_filters().then(|| FrameFilters {
        chain,
        view: filter_view(file_id, pane_uid, viewport, scale),
    })
}

fn cached_cpu_frame_matches(
    cpu_frame: &CachedCpuFrame,
    file_id: i32,
    viewport: &Viewport,
    render_width: u32,
    render_height: u32,
) -> bool {
    cpu_frame.file_id == file_id
        && cpu_frame.width == render_width
        && cpu_frame.height == render_height
        && (cpu_frame.viewport.center.x - viewport.center.x).abs() <= 1e-6
        && (cpu_frame.viewport.center.y - viewport.center.y).abs() <= 1e-6
        && (cpu_frame.viewport.zoom - viewport.zoom).abs() <= 1e-6
        && (cpu_frame.viewport.width - viewport.width).abs() <= 1e-6
        && (cpu_frame.viewport.height - viewport.height).abs() <= 1e-6
        && (cpu_frame.viewport.image_width - viewport.image_width).abs() <= 1e-6
        && (cpu_frame.viewport.image_height - viewport.image_height).abs() <= 1e-6
}

/// Runs the (changed) plugin filters again over the pane's last frame,
/// when that frame still shows `viewport`. `None` otherwise.
#[allow(clippy::too_many_arguments)]
fn refilter_cached_frame(
    pane: PaneId,
    pane_uid: u32,
    file_id: i32,
    viewport: &Viewport,
    render_width: u32,
    render_height: u32,
    filter_chain: &crate::plugin_runtime::filters::SharedFilterChain,
    scale: f64,
) -> Option<Image> {
    crate::with_pane_render_cache(pane.0 + 1, |cache| {
        let entry = cache.get_mut(pane.0)?;
        let cpu_frame = entry.cpu_frame.as_ref()?;
        if !cached_cpu_frame_matches(cpu_frame, file_id, viewport, render_width, render_height) {
            return None;
        }
        let needed = (render_width as usize) * (render_height as usize) * 4;
        let mut buffer =
            slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(render_width, render_height);
        let pixels = buffer.make_mut_bytes();
        pixels.copy_from_slice(&cpu_frame.pixels[..needed]);
        if let Some(filters) = frame_filters(filter_chain, file_id, pane_uid, viewport, scale) {
            filters.apply(pixels, render_width, render_height);
        }
        Some(Image::from_rgba8(buffer))
    })
}

// ─── CPU filters on GPU frames (RND-6) ──────────────────────────────────────

/// GPU panes with plugin filters: their frames are read back and filtered on
/// the render pool, one at a time per pane; the pane shows the newest
/// filtered frame.
mod gpu_filters {
    use super::*;

    #[derive(Default)]
    struct PaneFilterState {
        /// Generation of the readback on its way from the GPU, if any.
        awaiting_readback: Option<u64>,
        /// A filter job is running for the pane.
        filtering: bool,
        /// The pane's frame changed while busy: read it back again after.
        dirty: bool,
        /// Newest generation shown; older results are dropped.
        shown: u64,
    }

    struct FilterPanes {
        panes: HashMap<u32, PaneFilterState>,
        next_generation: u64,
        results: (
            crossbeam_channel::Sender<FilteredFrame>,
            crossbeam_channel::Receiver<FilteredFrame>,
        ),
    }

    impl Default for FilterPanes {
        fn default() -> Self {
            Self {
                panes: HashMap::new(),
                next_generation: 0,
                results: crossbeam_channel::unbounded(),
            }
        }
    }

    /// A filtered frame, back from the pool.
    pub(super) struct FilteredFrame {
        pub(super) tag: ReadbackTag,
        pub(super) width: u32,
        pub(super) height: u32,
        /// The frame before filtering.
        pub(super) unfiltered: Vec<u8>,
        pub(super) display: slint::SharedPixelBuffer<slint::Rgba8Pixel>,
    }

    thread_local! {
        // UI thread only.
        static PANES: RefCell<FilterPanes> = RefCell::new(FilterPanes::default());
    }

    /// A readback tag for the pane's next frame, or `None` (the frame is
    /// remembered as dirty) while the pane's previous frame is still being
    /// read back or filtered.
    pub(super) fn request(
        pane_uid: u32,
        file_id: i32,
        viewport: &Viewport,
        scale: f64,
    ) -> Option<ReadbackTag> {
        PANES.with(|panes| {
            let mut panes = panes.borrow_mut();
            panes.next_generation += 1;
            let generation = panes.next_generation;
            let pane = panes.panes.entry(pane_uid).or_default();
            if pane.awaiting_readback.is_some() || pane.filtering {
                pane.dirty = true;
                return None;
            }
            pane.awaiting_readback = Some(generation);
            Some(ReadbackTag {
                pane_uid,
                file_id,
                viewport: viewport.clone(),
                scale,
                generation,
            })
        })
    }

    /// Hands read-back frames to the render pool for filtering. Readbacks
    /// the renderer lost (`readbacks_pending == false` while one is
    /// awaited) are forgotten; returns whether there were any.
    pub(super) fn dispatch(
        frames: Vec<ReadbackFrame>,
        readbacks_pending: bool,
        filter_chain: &crate::plugin_runtime::filters::SharedFilterChain,
    ) -> bool {
        let mut lost = false;
        PANES.with(|panes| {
            let mut panes = panes.borrow_mut();
            let sender = panes.results.0.clone();
            for frame in frames {
                let Some(pane) = panes.panes.get_mut(&frame.tag.pane_uid) else {
                    continue;
                };
                pane.awaiting_readback = None;
                pane.filtering = true;
                let filters = frame_filters(
                    filter_chain,
                    frame.tag.file_id,
                    frame.tag.pane_uid,
                    &frame.tag.viewport,
                    frame.tag.scale,
                );
                let sender = sender.clone();
                let job = move || {
                    let mut display = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(
                        frame.width,
                        frame.height,
                    );
                    let bytes = display.make_mut_bytes();
                    bytes.copy_from_slice(&frame.pixels);
                    if let Some(filters) = filters {
                        filters.apply(bytes, frame.width, frame.height);
                    }
                    let _ = sender.send(FilteredFrame {
                        tag: frame.tag,
                        width: frame.width,
                        height: frame.height,
                        unfiltered: frame.pixels,
                        display,
                    });
                };
                match crate::render_pool::global() {
                    Some(pool) => pool.spawn(job),
                    None => job(),
                }
            }
            if !readbacks_pending {
                // Readbacks the renderer dropped (a reset) never arrive.
                for pane in panes.panes.values_mut() {
                    if pane.awaiting_readback.take().is_some() {
                        pane.dirty = true;
                        lost = true;
                    }
                }
            }
        });
        lost
    }

    /// The filtered frames that are ready, newest per pane, and whether a
    /// pane must render again (its frame changed while it was filtered).
    pub(super) fn receive() -> (Vec<FilteredFrame>, bool) {
        PANES.with(|panes| {
            let mut panes = panes.borrow_mut();
            let mut again = false;
            let mut ready: Vec<FilteredFrame> = Vec::new();
            while let Ok(frame) = panes.results.1.try_recv() {
                let Some(pane) = panes.panes.get_mut(&frame.tag.pane_uid) else {
                    continue;
                };
                pane.filtering = false;
                again |= std::mem::take(&mut pane.dirty);
                if frame.tag.generation > pane.shown {
                    pane.shown = frame.tag.generation;
                    ready.retain(|older| older.tag.pane_uid != frame.tag.pane_uid);
                    ready.push(frame);
                }
            }
            (ready, again)
        })
    }

    /// Shows the filtered frames that are ready. Returns whether one was
    /// shown, and whether a pane must render again.
    pub(super) fn apply(state: &mut AppState) -> (bool, bool) {
        let (frames, again) = receive();
        let mut shown = false;
        for frame in frames {
            let Some(pane) = state.pane_by_uid(frame.tag.pane_uid) else {
                continue;
            };
            if state.active_file_id_for_pane(pane) != Some(frame.tag.file_id) {
                continue;
            }
            crate::set_cached_pane_cpu_result(
                pane,
                Image::from_rgba8(frame.display),
                CachedCpuFrame {
                    file_id: frame.tag.file_id,
                    width: frame.width,
                    height: frame.height,
                    viewport: frame.tag.viewport,
                    pixels: frame.unfiltered,
                    source: Vec::new(),
                },
            );
            state.set_last_rendered_file_id(pane, Some(frame.tag.file_id));
            shown = true;
        }
        (shown, again)
    }

    /// Whether any pane is waiting for a readback or a filter job.
    pub(super) fn busy() -> bool {
        PANES.with(|panes| {
            panes
                .borrow()
                .panes
                .values()
                .any(|pane| pane.awaiting_readback.is_some() || pane.filtering)
        })
    }

    /// Forgets panes that no longer exist.
    pub(super) fn retain_panes(live: &[u32]) {
        PANES.with(|panes| panes.borrow_mut().panes.retain(|uid, _| live.contains(uid)));
    }
}

// ─── Fallback culling ───────────────────────────────────────────────────────

/// Most fallback tiles drawn of one level; they only show while the fine
/// tiles load.
const MAX_FALLBACK_TILES_PER_LEVEL: usize = 1024;

/// The view's coverage by the fine tiles drawn this frame. A fallback tile
/// that fine tiles hide completely is not drawn: on the CPU it would only
/// cost compositing time, on the GPU texture layers (RND-2, RND-6).
struct FineCoverage {
    cached: HashSet<(u64, u64)>,
    /// Level-0 size of a fine tile.
    tile_extent: f64,
    tiles_x: u64,
    tiles_y: u64,
    view: eov_common::viewport::ViewportBounds,
}

impl FineCoverage {
    fn new(
        tile_manager: &TileManager,
        level: u32,
        fine_tiles: &[(eov_common::TileCoord, Arc<eov_common::TileData>)],
        view: eov_common::viewport::ViewportBounds,
    ) -> Option<Self> {
        let info = tile_manager.wsi().level(level)?;
        let tile_size = tile_manager.tile_size_for_level(level);
        Some(Self {
            cached: fine_tiles
                .iter()
                .filter(|(coord, _)| coord.level == level)
                .map(|(coord, _)| (coord.x, coord.y))
                .collect(),
            tile_extent: f64::from(tile_size) * info.downsample,
            tiles_x: info.tiles_x(tile_size),
            tiles_y: info.tiles_y(tile_size),
            view,
        })
    }

    /// Whether the visible part of a tile of a level with downsample
    /// `downsample` lies entirely under cached fine tiles.
    fn covers_tile(
        &self,
        coord: &eov_common::TileCoord,
        tile: &eov_common::TileData,
        downsample: f64,
    ) -> bool {
        let left = coord.x as f64 * coord.tile_size as f64 * downsample;
        let top = coord.y as f64 * coord.tile_size as f64 * downsample;
        self.covers(
            left,
            top,
            left + tile.width as f64 * downsample,
            top + tile.height as f64 * downsample,
        )
    }

    /// Whether the visible part of a level-0 rectangle lies entirely under
    /// cached fine tiles.
    fn covers(&self, left: f64, top: f64, right: f64, bottom: f64) -> bool {
        let l = left.max(self.view.left);
        let t = top.max(self.view.top);
        let r = right.min(self.view.right);
        let b = bottom.min(self.view.bottom);
        if r <= l || b <= t {
            return true;
        }
        if self.tile_extent <= 0.0 {
            return false;
        }
        let x0 = (l / self.tile_extent).floor().max(0.0) as u64;
        let y0 = (t / self.tile_extent).floor().max(0.0) as u64;
        let x1 = ((r / self.tile_extent).ceil().max(0.0) as u64).min(self.tiles_x);
        let y1 = ((b / self.tile_extent).ceil().max(0.0) as u64).min(self.tiles_y);
        if x1 <= x0 || y1 <= y0 {
            return false;
        }
        (y0..y1).all(|y| (x0..x1).all(|x| self.cached.contains(&(x, y))))
    }
}

#[derive(Clone, Copy)]
struct TileProjection<'a> {
    viewport: &'a Viewport,
    bounds_left: f64,
    bounds_top: f64,
    downsample: f64,
}

struct CachedVisibleTiles {
    fine_tiles: Vec<(eov_common::TileCoord, Arc<eov_common::TileData>)>,
    coarse_tiles: Vec<(eov_common::TileCoord, Arc<eov_common::TileData>)>,
}

#[derive(Clone, Copy)]
struct RenderAdjustments {
    sharpness: f32,
    gamma: f32,
    brightness: f32,
    contrast: f32,
    stain_normalization: StainNormalization,
    deconv_h_intensity: f32,
    deconv_h_visible: bool,
    deconv_e_intensity: f32,
    deconv_e_visible: bool,
    deconv_isolated: crate::state::IsolatedChannel,
}

impl RenderAdjustments {
    fn from_hud(hud: &crate::state::HudSettings) -> Self {
        Self {
            sharpness: hud.sharpness,
            gamma: hud.gamma,
            brightness: hud.brightness,
            contrast: hud.contrast,
            stain_normalization: hud.stain_normalization,
            deconv_h_intensity: hud.deconv_hematoxylin_intensity,
            deconv_h_visible: hud.deconv_hematoxylin_visible,
            deconv_e_intensity: hud.deconv_eosin_intensity,
            deconv_e_visible: hud.deconv_eosin_visible,
            deconv_isolated: hud.deconv_isolated_channel,
        }
    }

    /// The adjustments of the pane's last rendered frame.
    fn last_rendered(pane: &crate::state::FilePaneState) -> Self {
        Self {
            sharpness: pane.last_render_sharpness,
            gamma: pane.last_render_gamma,
            brightness: pane.last_render_brightness,
            contrast: pane.last_render_contrast,
            stain_normalization: pane.last_render_stain_normalization,
            deconv_h_intensity: pane.last_render_deconv_h_intensity,
            deconv_h_visible: pane.last_render_deconv_h_visible,
            deconv_e_intensity: pane.last_render_deconv_e_intensity,
            deconv_e_visible: pane.last_render_deconv_e_visible,
            deconv_isolated: pane.last_render_deconv_isolated,
        }
    }

    /// Records these as the adjustments of the pane's last rendered frame.
    fn record(self, pane: &mut crate::state::FilePaneState) {
        pane.last_render_sharpness = self.sharpness;
        pane.last_render_gamma = self.gamma;
        pane.last_render_brightness = self.brightness;
        pane.last_render_contrast = self.contrast;
        pane.last_render_stain_normalization = self.stain_normalization;
        pane.last_render_deconv_h_intensity = self.deconv_h_intensity;
        pane.last_render_deconv_h_visible = self.deconv_h_visible;
        pane.last_render_deconv_e_intensity = self.deconv_e_intensity;
        pane.last_render_deconv_e_visible = self.deconv_e_visible;
        pane.last_render_deconv_isolated = self.deconv_isolated;
    }

    fn changed_from(self, previous: Self) -> bool {
        (self.sharpness - previous.sharpness).abs() > 0.001
            || (self.gamma - previous.gamma).abs() > 0.001
            || (self.brightness - previous.brightness).abs() > 0.001
            || (self.contrast - previous.contrast).abs() > 0.001
            || self.stain_normalization != previous.stain_normalization
            || (self.deconv_h_intensity - previous.deconv_h_intensity).abs() > 0.001
            || self.deconv_h_visible != previous.deconv_h_visible
            || (self.deconv_e_intensity - previous.deconv_e_intensity).abs() > 0.001
            || self.deconv_e_visible != previous.deconv_e_visible
            || self.deconv_isolated != previous.deconv_isolated
    }
}

fn requested_trilinear_blend(filtering_mode: FilteringMode, zoom: f64) -> (f64, bool) {
    let lanczos_weight = if filtering_mode == FilteringMode::Lanczos3 {
        lanczos_adaptive_weight(zoom)
    } else {
        1.0
    };
    let use_trilinear_blend = filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0);
    (lanczos_weight, use_trilinear_blend)
}

fn effective_cpu_filtering_mode(
    filtering_mode: FilteringMode,
    use_trilinear_blend: bool,
    is_moving: bool,
) -> FilteringMode {
    if is_moving {
        if use_trilinear_blend {
            FilteringMode::Trilinear
        } else {
            FilteringMode::Bilinear
        }
    } else {
        filtering_mode
    }
}

fn filter_uses_trilinear(filtering_mode: FilteringMode, lanczos_weight: f64) -> bool {
    filtering_mode == FilteringMode::Trilinear
        || (filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0)
}

fn collect_cached_visible_tiles(
    tile_manager: &TileManager,
    tile_cache: &Arc<TileCache>,
    viewport: &Viewport,
    trilinear: TrilinearLevels,
    margin_tiles: i32,
    fetch_coarse_tiles: bool,
) -> CachedVisibleTiles {
    let bounds = viewport.bounds();
    let fine_tiles = tile_manager
        .visible_tiles_with_margin(
            trilinear.level_fine,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
            margin_tiles,
        )
        .into_iter()
        .take(500)
        .filter_map(|coord| tile_cache.peek(&coord).map(|data| (coord, data)))
        .collect();

    let coarse_tiles = if fetch_coarse_tiles
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
    {
        tile_manager
            .visible_tiles_with_margin(
                trilinear.level_coarse,
                bounds.left,
                bounds.top,
                bounds.right,
                bounds.bottom,
                margin_tiles,
            )
            .into_iter()
            .filter_map(|coord| tile_cache.peek(&coord).map(|data| (coord, data)))
            .collect()
    } else {
        Vec::new()
    };

    CachedVisibleTiles {
        fine_tiles,
        coarse_tiles,
    }
}

/// One wanted-tile list for a slide from its panes' lists (each most
/// important first): taken in turns, so every pane's centre comes before any
/// pane's margins and fallbacks, without duplicates.
fn merge_wanted_tiles(lists: Vec<Vec<eov_common::TileCoord>>) -> Vec<eov_common::TileCoord> {
    if lists.len() == 1 {
        return lists.into_iter().next().unwrap_or_default();
    }
    let total = lists.iter().map(Vec::len).sum();
    let mut seen = HashSet::with_capacity(total);
    let mut merged = Vec::with_capacity(total);
    let mut lists: Vec<_> = lists.into_iter().map(Vec::into_iter).collect();
    while !lists.is_empty() {
        lists.retain_mut(|list| match list.next() {
            Some(coord) => {
                if seen.insert(coord) {
                    merged.push(coord);
                }
                true
            }
            None => false,
        });
    }
    merged
}

/// The slide's minimap thumbnail turned like the pane's view.
pub(crate) fn thumbnail_image_for_file(
    file: &OpenFile,
    orientation: eov_common::viewport::Orientation,
) -> Option<Image> {
    let thumbnail = file.thumbnail.as_ref()?;
    let (width, height) = orientation.swap(thumbnail.width, thumbnail.height);
    let needed = width as usize * height as usize * 4;
    if thumbnail.rgba.len() < needed {
        return None;
    }
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    eov_common::blitter::orient_frame(
        &thumbnail.rgba,
        thumbnail.width,
        thumbnail.height,
        orientation,
        buffer.make_mut_bytes(),
    );
    Some(Image::from_rgba8(buffer))
}

/// Whether the pane's minimap needs its thumbnail (again): none yet, or
/// turned differently from the view.
fn minimap_stale(
    pane: PaneId,
    pane_count: usize,
    orientation: eov_common::viewport::Orientation,
) -> bool {
    crate::with_pane_render_cache(pane_count, |cache| {
        cache.get(pane.0).is_none_or(|entry| {
            entry.minimap_thumbnail.is_none() || entry.minimap_orientation != orientation
        })
    })
}

/// Per-frame tile-cache bookkeeping: protects this frame's working set from
/// eviction, sizes the cache for the window, and lets the tiles of slides
/// shown in no pane go first (WSI-4).
fn begin_cache_frame(ui: &AppWindow, state: &AppState, tile_cache: &TileCache) {
    tile_cache.begin_frame();

    let visible_panes = if state.split_enabled {
        state.panes.len()
    } else {
        1
    };
    let size = ui.window().size();
    crate::cache_budget::update(tile_cache, size.width, size.height, visible_panes as u32);

    thread_local! {
        /// File id → visibility last told to the cache.
        static VISIBILITY: RefCell<HashMap<i32, bool>> = RefCell::new(HashMap::new());
    }
    let visible: HashSet<i32> = (0..visible_panes)
        .filter_map(|pane| state.active_file_id_for_pane(PaneId(pane)))
        .collect();
    VISIBILITY.with(|known| {
        let mut known = known.borrow_mut();
        known.retain(|id, _| state.get_file(*id).is_some());
        for file in &state.open_files {
            let shown = visible.contains(&file.id);
            if known.insert(file.id, shown) != Some(shown) {
                tile_cache.set_file_visible(file.id, shown);
            }
        }
    });
}

/// Per-frame colour bookkeeping: panes of slides whose stain fit has just
/// finished render again (now normalised, or with the reason it is not),
/// and the colour state of closed files is dropped.
fn begin_color_frame(state: &mut AppState) {
    let open: HashSet<i32> = state.open_files.iter().map(|file| file.id).collect();
    crate::color::retain_files(&open);
    // Render jobs and filter state of panes that are gone (RND-9).
    let live: Vec<u32> = state.panes.iter().map(|pane| pane.uid).collect();
    if let Some(pool) = crate::render_pool::global() {
        pool.retain_panes(&live);
    }
    gpu_filters::retain_panes(&live);
    // GPU surfaces of closed panes (RND-7).
    crate::with_gpu_renderer(|renderer| renderer.borrow_mut().retain_slots(live.len()));
    let refitted = crate::color::files_with_new_fits();
    if refitted.is_empty() {
        return;
    }
    for file in state
        .open_files
        .iter_mut()
        .filter(|file| refitted.contains(&file.id))
    {
        file.invalidate_render_state();
    }
    state.request_render();
}

/// Message shown when the GPU renderer fails while running.
const GPU_FAILED_MESSAGE: &str = "The GPU renderer failed; slides are drawn by the CPU renderer";

/// The GPU failed (a wgpu error or a lost device): slides are drawn by the
/// CPU renderer from now on (RND-7).
fn use_cpu_after_gpu_failure(ui: &AppWindow, state: &mut AppState, reason: &str) {
    tracing::warn!("{GPU_FAILED_MESSAGE}: {reason}");
    crate::with_gpu_renderer(|renderer| renderer.borrow_mut().disable());
    state.gpu_backend_available = false;
    state.select_render_backend(RenderBackend::Cpu);
    crate::ui_update::update_render_backend(ui, state);
    ui.set_status_text(GPU_FAILED_MESSAGE.into());
}

pub(crate) fn update_and_render(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
) -> bool {
    if let Some(reason) = crate::gpu::take_failure() {
        use_cpu_after_gpu_failure(ui, &mut state.write(), &reason);
        crate::host_ui::with_host(|host| {
            host.notify(
                eov_plugin_api::protocol::NotifyLevel::Warning,
                GPU_FAILED_MESSAGE,
                Some(&reason),
                None,
            )
        });
    }
    let (completed_cpu_frame, cpu_jobs_pending, render_backend) = {
        let mut state = state.write();
        begin_color_frame(&mut state);
        let (completed_cpu_frame, cpu_jobs_pending) = apply_completed_cpu_renders(&mut state);
        let filtered_frame = apply_gpu_filter_results(&mut state);
        begin_cache_frame(ui, &state, tile_cache);
        (
            completed_cpu_frame || filtered_frame,
            cpu_jobs_pending,
            state.render_backend,
        )
    };

    if render_backend == RenderBackend::Cpu {
        return update_and_render_cpu(ui, state, tile_cache, completed_cpu_frame, cpu_jobs_pending);
    }

    let mut state = state.write();
    let filtering_mode = state.filtering_mode;
    let pane_count = state.panes.len();
    let active_file_ids: Vec<Option<i32>> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| state.active_file_id_for_pane(PaneId(pane_index)))
        .collect();
    if active_file_ids.iter().all(Option::is_none) {
        for pane_index in 0..pane_count {
            crate::clear_cached_pane(PaneId(pane_index));
            state.set_last_rendered_file_id(PaneId(pane_index), None);
        }
        crate::update_tabs(ui, &state);
        return false;
    }

    let render_requested = std::mem::take(&mut state.needs_render) || completed_cpu_frame;

    // The panes' real size (as Slint lays them out), never zero.
    let content_width = (ui.get_content_area_width() as f64).max(1.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(1.0);
    let pane_gap = crate::state::PANE_GAP;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(1.0);

    // Physical pixels per logical pixel of the frames (RND-3).
    let scale = render_scale(
        ui.window().scale_factor(),
        pane_width,
        content_height,
        crate::with_gpu_renderer(|renderer| renderer.borrow().max_surface_dimension())
            .flatten()
            .unwrap_or(MAX_CPU_FRAME_DIMENSION),
    );

    // Each pane's wanted tiles, most important first, per file.
    let mut wanted_tiles_by_file: HashMap<i32, Vec<Vec<eov_common::TileCoord>>> = HashMap::new();
    for (pane_index, file_id) in active_file_ids.iter().copied().enumerate() {
        let Some(file_id) = file_id else {
            continue;
        };
        let pane = PaneId(pane_index);
        let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
            continue;
        };
        let Some(pane_state) = file.pane_state_mut(pane) else {
            continue;
        };
        pane_state.viewport.set_size(pane_width, content_height);
        let viewport_state = pane_state.viewport.clone();
        let vp = &physical_viewport(&viewport_state.viewport, scale);
        let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
        let trilinear = calculate_trilinear_levels(
            file.tile_manager.wsi(),
            vp.effective_downsample(),
            trilinear_lod_bias_enabled(filtering_mode),
        );
        let bounds = vp.bounds();
        wanted_tiles_by_file
            .entry(file_id)
            .or_default()
            .push(calculate_wanted_tiles(
                &file.tile_manager,
                trilinear.level_fine,
                bounds.left,
                bounds.top,
                bounds.right,
                bounds.bottom,
                margin_tiles,
            ));
    }

    for (file_id, wanted_tiles) in wanted_tiles_by_file {
        if let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) {
            file.tile_loader
                .set_wanted_tiles(merge_wanted_tiles(wanted_tiles));
        }
    }

    let mut keep_running =
        render_requested || cpu_jobs_pending || crate::color::fits_pending() || gpu_filters::busy();
    let mut rendered_frame = completed_cpu_frame;
    let filter_chain = state.filter_chain.clone();
    let filter_revision = state.filter_revision;

    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let pane_uid = state.pane_uid(pane).unwrap_or(pane_index as u32);
        let file_switched = state.last_rendered_file_id(pane) != file_id;
        keep_running |= file_switched;

        let Some(file_id) = file_id else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let Some(file) = state.open_files.iter_mut().find(|f| f.id == file_id) else {
            crate::clear_cached_pane(pane);
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        let orientation = file
            .pane_state(pane)
            .map(|pane_state| pane_state.viewport.viewport.orientation)
            .unwrap_or_default();
        let minimap_missing = minimap_stale(pane, pane_count, orientation);
        let content_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.content.as_ref())
                .is_none()
        });
        let force_render = file_switched || content_missing || render_requested;

        if file_switched && let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.frame_count = 0;
        }
        if file_switched || minimap_missing {
            crate::set_cached_pane_minimap(
                pane,
                thumbnail_image_for_file(file, orientation),
                orientation,
            );
        }

        let outcome = render_pane_gpu(
            file,
            pane,
            RenderPaneRequest {
                ui,
                tile_cache,
                target_width: pane_width,
                target_height: content_height,
                force_render,
                filtering_mode,
                filter_chain: &filter_chain,
                pane_uid,
                filter_revision,
                scale,
            },
        );

        keep_running |= outcome.keep_running;
        rendered_frame |= outcome.rendered;
        if let Some(image) = outcome.image {
            crate::set_cached_pane_content(pane, image);
            state.set_last_rendered_file_id(pane, Some(file_id));
        } else {
            state.set_last_rendered_file_id(pane, Some(file_id));
            keep_running |= content_missing;
        }
    }

    crate::update_tabs(ui, &state);
    keep_running |= crate::host_ui::with_host(|host| {
        host.after_frame(&state);
        host.overlay_frames_pending()
    })
    .unwrap_or(false);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

fn update_and_render_cpu(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    completed_cpu_frame: bool,
    cpu_jobs_pending: bool,
) -> bool {
    let Some(frame) = collect_cpu_frame_snapshot(ui, state, completed_cpu_frame, cpu_jobs_pending)
    else {
        return false;
    };

    let filter_chain = {
        let state = state.read();
        state.filter_chain.clone()
    };

    let mut wanted_tiles_by_file: HashMap<i32, (Arc<TileLoader>, Vec<Vec<eov_common::TileCoord>>)> =
        HashMap::new();

    for snapshot in frame.panes.iter().flatten() {
        let vp = &snapshot.render_viewport;
        let margin_tiles = if snapshot.viewport_state.is_moving() {
            1
        } else {
            0
        };
        let trilinear = calculate_trilinear_levels(
            snapshot.tile_manager.wsi(),
            vp.effective_downsample(),
            trilinear_lod_bias_enabled(snapshot.filtering_mode),
        );
        let wanted = calculate_wanted_tiles(
            &snapshot.tile_manager,
            trilinear.level_fine,
            vp.bounds().left,
            vp.bounds().top,
            vp.bounds().right,
            vp.bounds().bottom,
            margin_tiles,
        );

        wanted_tiles_by_file
            .entry(snapshot.file_id)
            .or_insert_with(|| (Arc::clone(&snapshot.tile_loader), Vec::new()))
            .1
            .push(wanted);
    }

    for (_, (tile_loader, wanted_tiles)) in wanted_tiles_by_file {
        tile_loader.set_wanted_tiles(merge_wanted_tiles(wanted_tiles));
    }

    let mut keep_running =
        frame.render_requested || frame.cpu_jobs_pending || crate::color::fits_pending();
    let mut rendered_frame = completed_cpu_frame;

    for pane_index in 0..frame.pane_count {
        let pane = PaneId(pane_index);
        let Some(snapshot) = frame
            .panes
            .get(pane_index)
            .and_then(|snapshot| snapshot.clone())
        else {
            crate::clear_cached_pane(pane);
            let mut state = state.write();
            state.set_last_rendered_file_id(pane, None);
            continue;
        };

        if snapshot.file_switched || snapshot.minimap_missing {
            crate::set_cached_pane_minimap(
                snapshot.pane,
                snapshot.minimap_image.clone(),
                snapshot.viewport_state.viewport.orientation,
            );
        }

        let execution = render_cpu_pane_from_snapshot(&snapshot, tile_cache, &filter_chain);
        keep_running |= execution.outcome.keep_running || snapshot.file_switched;
        rendered_frame |= execution.outcome.rendered;

        if let Some(image) = execution.outcome.image.clone() {
            crate::set_cached_pane_content(snapshot.pane, image);
        }

        apply_cpu_render_commit(state, execution);
    }

    let mut state = state.write();
    crate::update_tabs(ui, &state);
    keep_running |= crate::host_ui::with_host(|host| {
        host.after_frame(&state);
        host.overlay_frames_pending()
    })
    .unwrap_or(false);

    if rendered_frame {
        state.update_fps();
        ui.set_fps(state.current_fps);
    }

    keep_running
}

fn collect_cpu_frame_snapshot(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    completed_cpu_frame: bool,
    cpu_jobs_pending: bool,
) -> Option<CpuFrameSnapshot> {
    let mut state = state.write();
    let pane_count = state.panes.len();
    let active_file_ids: Vec<Option<i32>> = state
        .panes
        .iter()
        .enumerate()
        .map(|(pane_index, _)| state.active_file_id_for_pane(PaneId(pane_index)))
        .collect();

    if active_file_ids.iter().all(Option::is_none) {
        for pane_index in 0..pane_count {
            crate::clear_cached_pane(PaneId(pane_index));
            state.set_last_rendered_file_id(PaneId(pane_index), None);
        }
        crate::update_tabs(ui, &state);
        return None;
    }

    let render_requested = std::mem::take(&mut state.needs_render) || completed_cpu_frame;
    let filter_revision = state.filter_revision;

    // The panes' real size (as Slint lays them out), never zero.
    let content_width = (ui.get_content_area_width() as f64).max(1.0);
    let content_height = (ui.get_content_area_height() as f64 - 35.0).max(1.0);
    let pane_gap = crate::state::PANE_GAP;
    let pane_width = ((content_width - pane_gap * (pane_count.saturating_sub(1) as f64))
        / pane_count.max(1) as f64)
        .max(1.0);
    let scale = render_scale(
        ui.window().scale_factor(),
        pane_width,
        content_height,
        MAX_CPU_FRAME_DIMENSION,
    );

    let mut panes = Vec::with_capacity(pane_count);
    for (pane_index, file_id) in active_file_ids.into_iter().enumerate() {
        let pane = PaneId(pane_index);
        let Some(file_id) = file_id else {
            panes.push(None);
            continue;
        };

        let pane_uid = state.pane_uid(pane).unwrap_or(pane_index as u32);
        let file_switched = state.last_rendered_file_id(pane) != Some(file_id);
        let content_missing = crate::with_pane_render_cache(pane_count, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.content.as_ref())
                .is_none()
        });

        let filtering_mode = state.filtering_mode;
        let Some(file) = state.open_files.iter_mut().find(|file| file.id == file_id) else {
            panes.push(None);
            continue;
        };
        let orientation = file
            .pane_state(pane)
            .map(|pane_state| pane_state.viewport.viewport.orientation)
            .unwrap_or_default();
        let minimap_missing = minimap_stale(pane, pane_count, orientation);
        let tile_manager = Arc::clone(&file.tile_manager);
        let tile_loader = Arc::clone(&file.tile_loader);
        let minimap_image = if file_switched || minimap_missing {
            thumbnail_image_for_file(file, orientation)
        } else {
            None
        };

        let Some((
            viewport_state,
            frame_count,
            last_render_zoom,
            last_render_center_x,
            last_render_center_y,
            last_render_width,
            last_render_height,
            last_render_level,
            last_render_orientation,
            previous_tiles_loaded,
            last_seen_tile_epoch,
            hud_sharpness,
            hud_gamma,
            hud_brightness,
            hud_contrast,
            hud_stain_normalization,
            hud_deconv_h_intensity,
            hud_deconv_h_visible,
            hud_deconv_e_intensity,
            hud_deconv_e_visible,
            hud_deconv_isolated,
            last_render_sharpness,
            last_render_gamma,
            last_render_brightness,
            last_render_contrast,
            last_render_stain_normalization,
            last_render_deconv_h_intensity,
            last_render_deconv_h_visible,
            last_render_deconv_e_intensity,
            last_render_deconv_e_visible,
            last_render_deconv_isolated,
            last_render_filter_revision,
            pending_cpu_job_id,
            needs_settled_cpu_render,
        )) = (match file.pane_state_mut(pane) {
            Some(pane_state) => {
                if file_switched {
                    pane_state.frame_count = 0;
                }
                let _ = pane_state.viewport.update();
                pane_state.viewport.set_size(pane_width, content_height);
                Some((
                    pane_state.viewport.clone(),
                    pane_state.frame_count,
                    pane_state.last_render_zoom,
                    pane_state.last_render_center_x,
                    pane_state.last_render_center_y,
                    pane_state.last_render_width,
                    pane_state.last_render_height,
                    pane_state.last_render_level,
                    pane_state.last_render_orientation,
                    pane_state.tiles_loaded_since_render,
                    pane_state.last_seen_tile_epoch,
                    pane_state.hud.sharpness,
                    pane_state.hud.gamma,
                    pane_state.hud.brightness,
                    pane_state.hud.contrast,
                    pane_state.hud.stain_normalization,
                    pane_state.hud.deconv_hematoxylin_intensity,
                    pane_state.hud.deconv_hematoxylin_visible,
                    pane_state.hud.deconv_eosin_intensity,
                    pane_state.hud.deconv_eosin_visible,
                    pane_state.hud.deconv_isolated_channel,
                    pane_state.last_render_sharpness,
                    pane_state.last_render_gamma,
                    pane_state.last_render_brightness,
                    pane_state.last_render_contrast,
                    pane_state.last_render_stain_normalization,
                    pane_state.last_render_deconv_h_intensity,
                    pane_state.last_render_deconv_h_visible,
                    pane_state.last_render_deconv_e_intensity,
                    pane_state.last_render_deconv_e_visible,
                    pane_state.last_render_deconv_isolated,
                    pane_state.last_render_filter_revision,
                    pane_state.pending_cpu_job_id,
                    pane_state.needs_settled_cpu_render,
                ))
            }
            None => None,
        })
        else {
            panes.push(None);
            continue;
        };

        let color = crate::color::pane_color(file, pane);
        let render_viewport = physical_viewport(&viewport_state.viewport, scale);

        panes.push(Some(CpuPaneSnapshot {
            pane,
            pane_uid,
            file_id,
            file_switched,
            content_missing,
            minimap_missing,
            minimap_image,
            tile_manager,
            tile_loader,
            viewport_state,
            render_viewport,
            scale,
            frame_count,
            last_render_zoom,
            last_render_center_x,
            last_render_center_y,
            last_render_width,
            last_render_height,
            last_render_level,
            last_render_orientation,
            previous_tiles_loaded,
            last_seen_tile_epoch,
            hud_sharpness,
            hud_gamma,
            hud_brightness,
            hud_contrast,
            hud_stain_normalization,
            hud_deconv_h_intensity,
            hud_deconv_h_visible,
            hud_deconv_e_intensity,
            hud_deconv_e_visible,
            hud_deconv_isolated,
            last_render_sharpness,
            last_render_gamma,
            last_render_brightness,
            last_render_contrast,
            last_render_stain_normalization,
            last_render_deconv_h_intensity,
            last_render_deconv_h_visible,
            last_render_deconv_e_intensity,
            last_render_deconv_e_visible,
            last_render_deconv_isolated,
            last_render_filter_revision,
            pending_cpu_job_id,
            needs_settled_cpu_render,
            color,
            filtering_mode,
            filter_revision,
        }));
    }

    Some(CpuFrameSnapshot {
        pane_count,
        render_requested,
        cpu_jobs_pending,
        panes,
    })
}

fn apply_cpu_render_commit(state: &Arc<RwLock<AppState>>, execution: CpuPaneExecution) {
    let mut state = state.write();
    let pane = execution.pane;
    let file_id = execution.file_id;

    if let Some(file) = state.open_files.iter_mut().find(|file| file.id == file_id)
        && let Some(pane_state) = file.pane_state_mut(pane)
    {
        if let Some(frame_update) = execution.commit.frame_update {
            pane_state.frame_count = frame_update.frame_count;
            pane_state.last_render_time = frame_update.last_render_time;
            pane_state.last_render_zoom = frame_update.last_render_zoom;
            pane_state.last_render_center_x = frame_update.last_render_center_x;
            pane_state.last_render_center_y = frame_update.last_render_center_y;
            pane_state.last_render_width = frame_update.last_render_width;
            pane_state.last_render_height = frame_update.last_render_height;
            pane_state.last_render_level = frame_update.last_render_level;
            pane_state.last_render_orientation = frame_update.last_render_orientation;
            pane_state.tiles_loaded_since_render = frame_update.tiles_loaded_since_render;
            pane_state.last_seen_tile_epoch = frame_update.last_seen_tile_epoch;
            pane_state.last_render_gamma = frame_update.last_render_gamma;
            pane_state.last_render_brightness = frame_update.last_render_brightness;
            pane_state.last_render_contrast = frame_update.last_render_contrast;
            pane_state.last_render_sharpness = frame_update.last_render_sharpness;
            pane_state.last_render_stain_normalization =
                frame_update.last_render_stain_normalization;
            pane_state.last_render_deconv_h_intensity = frame_update.last_render_deconv_h_intensity;
            pane_state.last_render_deconv_h_visible = frame_update.last_render_deconv_h_visible;
            pane_state.last_render_deconv_e_intensity = frame_update.last_render_deconv_e_intensity;
            pane_state.last_render_deconv_e_visible = frame_update.last_render_deconv_e_visible;
            pane_state.last_render_deconv_isolated = frame_update.last_render_deconv_isolated;
            pane_state.last_render_filter_revision = frame_update.last_render_filter_revision;
            pane_state.gpu_surface_valid = frame_update.gpu_surface_valid;
        }

        if let Some(pending_cpu_job_id) = execution.commit.pending_cpu_job_id {
            pane_state.pending_cpu_job_id = pending_cpu_job_id;
        }
        if let Some(needs_settled_cpu_render) = execution.commit.needs_settled_cpu_render {
            pane_state.needs_settled_cpu_render = needs_settled_cpu_render;
        }
        if let Some(revision) = execution.commit.last_render_filter_revision {
            pane_state.last_render_filter_revision = revision;
        }
    }

    if execution.outcome.image.is_some() || execution.content_missing {
        state.set_last_rendered_file_id(pane, Some(file_id));
    }
}

/// Hands GPU frames read back for plugin filters to the render pool and
/// shows the filtered frames that came back. Returns whether one was shown.
fn apply_gpu_filter_results(state: &mut AppState) -> bool {
    let (frames, pending) = crate::with_gpu_renderer(|renderer| {
        let mut renderer = renderer.borrow_mut();
        let frames = renderer.poll_readbacks();
        (frames, renderer.readbacks_pending())
    })
    .unwrap_or_default();
    let lost = gpu_filters::dispatch(frames, pending, &state.filter_chain);
    let (shown, again) = gpu_filters::apply(state);
    if again || lost {
        // A pane's frame changed while it was being filtered.
        state.request_render();
    }
    shown
}

fn apply_completed_cpu_renders(state: &mut AppState) -> (bool, bool) {
    let Some(pool) = crate::render_pool::global() else {
        return (false, false);
    };

    let mut applied = false;

    let recycle = |result: CpuRenderResult| {
        pool.recycle_buffer(result.pixels);
        pool.recycle_buffer(result.source);
    };

    while let Some(result) = pool.try_recv() {
        // The pane may have moved since the job started (RND-9).
        let Some(pane) = state.pane_by_uid(result.pane_uid) else {
            recycle(result);
            continue;
        };
        let is_active = state.active_file_id_for_pane(pane) == Some(result.file_id);
        let Some(file) = state
            .open_files
            .iter_mut()
            .find(|file| file.id == result.file_id)
        else {
            recycle(result);
            continue;
        };
        let Some(pane_state) = file.pane_state_mut(pane) else {
            recycle(result);
            continue;
        };

        if pane_state.pending_cpu_job_id != Some(result.job_id) {
            recycle(result);
            continue;
        }

        pane_state.pending_cpu_job_id = None;
        pane_state.needs_settled_cpu_render = !result.settled_quality;

        if !is_active {
            recycle(result);
            continue;
        }

        // Composited, processed and filtered on the worker: only the image
        // swap is left for the UI thread (RND-6).
        crate::set_cached_pane_cpu_result(
            pane,
            Image::from_rgba8(result.display),
            CachedCpuFrame {
                file_id: result.file_id,
                width: result.width,
                height: result.height,
                viewport: result.viewport,
                pixels: result.pixels,
                source: result.source,
            },
        );
        state.set_last_rendered_file_id(pane, Some(result.file_id));
        applied = true;
    }

    let pending = state
        .open_files
        .iter()
        .flat_map(|file| file.pane_states.iter().flatten())
        .any(|pane_state| pane_state.pending_cpu_job_id.is_some());

    if applied {
        state.request_render();
    }

    (applied, pending)
}

fn render_cpu_pane_from_snapshot(
    snapshot: &CpuPaneSnapshot,
    tile_cache: &Arc<TileCache>,
    filter_chain: &crate::plugin_runtime::filters::SharedFilterChain,
) -> CpuPaneExecution {
    let vp = &snapshot.render_viewport;
    // Tiles are composited axis-aligned with the slide and the frame turned
    // to the view's orientation afterwards.
    let aligned = vp.aligned();
    let vp_zoom = vp.zoom;
    let vp_center_x = vp.center.x;
    let vp_center_y = vp.center.y;
    let vp_width = vp.width;
    let vp_height = vp.height;
    let is_first_frame = snapshot.frame_count == 0;
    let is_moving = snapshot.viewport_state.is_moving();
    let viewport_changed = is_moving
        || (snapshot.last_render_zoom - vp_zoom).abs() > 0.001
        || (snapshot.last_render_center_x - vp_center_x).abs() > 1.0
        || (snapshot.last_render_center_y - vp_center_y).abs() > 1.0
        || (snapshot.last_render_width - vp_width).abs() > 1.0
        || (snapshot.last_render_height - vp_height).abs() > 1.0
        || snapshot.last_render_orientation != vp.orientation;

    let trilinear = calculate_trilinear_levels(
        snapshot.tile_manager.wsi(),
        vp.effective_downsample(),
        trilinear_lod_bias_enabled(snapshot.filtering_mode),
    );
    let level = trilinear.level_fine;
    let margin_tiles = if is_moving { 1 } else { 0 };
    let level_changed = level != snapshot.last_render_level;
    let bounds = vp.bounds();
    let (lanczos_weight, use_trilinear_blend) =
        requested_trilinear_blend(snapshot.filtering_mode, vp_zoom);
    let cached_tiles = collect_cached_visible_tiles(
        &snapshot.tile_manager,
        tile_cache,
        &aligned,
        trilinear,
        margin_tiles,
        use_trilinear_blend,
    );
    let cached_count = cached_tiles.fine_tiles.len() as u32;

    let loaded_tile_epoch = snapshot.tile_loader.loaded_epoch();
    let tile_epoch_advanced = loaded_tile_epoch > snapshot.last_seen_tile_epoch;
    // Newly arrived tiles (fine or coarse) always advance the loader epoch;
    // merely *having* coarse tiles cached must not keep the loop running.
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            snapshot.previous_tiles_loaded
        }
        || tile_epoch_advanced;
    let tiles_pending = snapshot.tile_loader.pending_count() > 0;
    let keep_running = is_moving || new_tiles_loaded || tiles_pending;
    let filters_changed = snapshot.filter_revision != snapshot.last_render_filter_revision;

    let adjustments_changed = RenderAdjustments {
        sharpness: snapshot.hud_sharpness,
        gamma: snapshot.hud_gamma,
        brightness: snapshot.hud_brightness,
        contrast: snapshot.hud_contrast,
        stain_normalization: snapshot.hud_stain_normalization,
        deconv_h_intensity: snapshot.hud_deconv_h_intensity,
        deconv_h_visible: snapshot.hud_deconv_h_visible,
        deconv_e_intensity: snapshot.hud_deconv_e_intensity,
        deconv_e_visible: snapshot.hud_deconv_e_visible,
        deconv_isolated: snapshot.hud_deconv_isolated,
    }
    .changed_from(RenderAdjustments {
        sharpness: snapshot.last_render_sharpness,
        gamma: snapshot.last_render_gamma,
        brightness: snapshot.last_render_brightness,
        contrast: snapshot.last_render_contrast,
        stain_normalization: snapshot.last_render_stain_normalization,
        deconv_h_intensity: snapshot.last_render_deconv_h_intensity,
        deconv_h_visible: snapshot.last_render_deconv_h_visible,
        deconv_e_intensity: snapshot.last_render_deconv_e_intensity,
        deconv_e_visible: snapshot.last_render_deconv_e_visible,
        deconv_isolated: snapshot.last_render_deconv_isolated,
    });

    let mut commit = CpuRenderCommit {
        frame_update: None,
        pending_cpu_job_id: None,
        needs_settled_cpu_render: None,
        last_render_filter_revision: None,
    };

    if !snapshot.file_switched
        && !snapshot.content_missing
        && !is_first_frame
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
        && !filters_changed
        && !snapshot.needs_settled_cpu_render
    {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running: keep_running || snapshot.pending_cpu_job_id.is_some(),
                rendered: false,
            },
            commit,
        };
    }

    let render_width = vp_width as u32;
    let render_height = vp_height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running,
                rendered: false,
            },
            commit,
        };
    }

    if filters_changed
        && !snapshot.file_switched
        && !snapshot.content_missing
        && !viewport_changed
        && !level_changed
        && !new_tiles_loaded
        && !adjustments_changed
        && !snapshot.needs_settled_cpu_render
        && let Some(image) = refilter_cached_frame(
            snapshot.pane,
            snapshot.pane_uid,
            snapshot.file_id,
            vp,
            render_width,
            render_height,
            filter_chain,
            snapshot.scale,
        )
    {
        commit.last_render_filter_revision = Some(snapshot.filter_revision);
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: Some(image),
                keep_running: keep_running || snapshot.pending_cpu_job_id.is_some(),
                rendered: true,
            },
            commit,
        };
    }

    commit.frame_update = Some(CpuFrameStateUpdate {
        frame_count: snapshot.frame_count + 1,
        last_render_time: std::time::Instant::now(),
        last_render_zoom: vp_zoom,
        last_render_center_x: vp_center_x,
        last_render_center_y: vp_center_y,
        last_render_width: vp_width,
        last_render_height: vp_height,
        last_render_level: level,
        last_render_orientation: vp.orientation,
        tiles_loaded_since_render: cached_count,
        last_seen_tile_epoch: loaded_tile_epoch,
        last_render_gamma: snapshot.hud_gamma,
        last_render_brightness: snapshot.hud_brightness,
        last_render_contrast: snapshot.hud_contrast,
        last_render_sharpness: snapshot.hud_sharpness,
        last_render_stain_normalization: snapshot.hud_stain_normalization,
        last_render_deconv_h_intensity: snapshot.hud_deconv_h_intensity,
        last_render_deconv_h_visible: snapshot.hud_deconv_h_visible,
        last_render_deconv_e_intensity: snapshot.hud_deconv_e_intensity,
        last_render_deconv_e_visible: snapshot.hud_deconv_e_visible,
        last_render_deconv_isolated: snapshot.hud_deconv_isolated,
        last_render_filter_revision: snapshot.filter_revision,
        gpu_surface_valid: false,
    });

    let Some(level_info) = snapshot.tile_manager.wsi().level(level).cloned() else {
        return CpuPaneExecution {
            pane: snapshot.pane,
            file_id: snapshot.file_id,
            content_missing: snapshot.content_missing,
            outcome: PaneRenderOutcome {
                image: None,
                keep_running: keep_running || snapshot.pending_cpu_job_id.is_some(),
                rendered: false,
            },
            commit,
        };
    };

    let mut pending_cpu_job_id = snapshot.pending_cpu_job_id;
    if pending_cpu_job_id.is_some()
        && (viewport_changed || level_changed || adjustments_changed || filters_changed)
    {
        commit.pending_cpu_job_id = Some(None);
        pending_cpu_job_id = None;
    }
    if is_moving {
        commit.needs_settled_cpu_render = Some(true);
    }

    let filtering_mode =
        effective_cpu_filtering_mode(snapshot.filtering_mode, use_trilinear_blend, is_moving);
    let is_adaptive_lanczos = filtering_mode == FilteringMode::Lanczos3;
    let effective_lanczos_weight = if is_adaptive_lanczos {
        lanczos_weight
    } else {
        1.0
    };
    let effective_trilinear = filter_uses_trilinear(filtering_mode, effective_lanczos_weight);

    let coverage = FineCoverage::new(
        &snapshot.tile_manager,
        level,
        &cached_tiles.fine_tiles,
        bounds,
    );
    let fallback_commands = fallback_blits(
        &snapshot.tile_manager,
        tile_cache,
        &aligned,
        level,
        coverage.as_ref(),
    );
    let fine_commands = fine_blits(
        &snapshot.tile_manager,
        &aligned,
        &level_info,
        trilinear,
        &cached_tiles,
        effective_trilinear,
        is_adaptive_lanczos && effective_lanczos_weight >= 1.0,
    );

    let postprocess = CpuRenderPostProcess {
        pipeline: Arc::clone(&snapshot.color.pipeline),
        background: crate::color::VIEWPORT_BACKGROUND,
    };

    let preview_image = if is_moving {
        render_cached_preview(RenderCachedPreview {
            pane: snapshot.pane,
            pane_uid: snapshot.pane_uid,
            file_id: snapshot.file_id,
            viewport: vp,
            scale: snapshot.scale,
            render_width,
            render_height,
            fallback_commands: &fallback_commands,
            fine_commands: &fine_commands,
            postprocess: &postprocess,
            filter_chain,
        })
    } else {
        None
    };

    if pending_cpu_job_id.is_none()
        && let Some(pool) = crate::render_pool::global()
    {
        let job_id = pool.next_job_id();
        pool.submit(CpuRenderJob {
            pane_uid: snapshot.pane_uid,
            file_id: snapshot.file_id,
            job_id,
            width: render_width,
            height: render_height,
            viewport: vp.clone(),
            fallback_blits: fallback_commands,
            fine_blits: fine_commands,
            postprocess,
            filters: frame_filters(
                filter_chain,
                snapshot.file_id,
                snapshot.pane_uid,
                vp,
                snapshot.scale,
            ),
            settled_quality: !is_moving,
        });
        commit.pending_cpu_job_id = Some(Some(job_id));
        commit.needs_settled_cpu_render = Some(is_moving);
        pending_cpu_job_id = Some(job_id);
    }

    let outcome = if let Some(image) = preview_image {
        PaneRenderOutcome {
            image: Some(image),
            keep_running: true,
            rendered: true,
        }
    } else {
        PaneRenderOutcome {
            image: None,
            keep_running: keep_running
                || pending_cpu_job_id.is_some()
                || snapshot.needs_settled_cpu_render,
            rendered: false,
        }
    };

    CpuPaneExecution {
        pane: snapshot.pane,
        file_id: snapshot.file_id,
        content_missing: snapshot.content_missing,
        outcome,
        commit,
    }
}

/// Screen rectangle of a tile of a level with downsample `downsample`.
fn tile_blit_rect(
    viewport: &Viewport,
    coord: &eov_common::TileCoord,
    tile: &eov_common::TileData,
    downsample: f64,
) -> blitter::BlitRect {
    let bounds = viewport.bounds();
    let origin_x = coord.x as f64 * coord.tile_size as f64;
    let origin_y = coord.y as f64 * coord.tile_size as f64;
    let exact_sx = (origin_x * downsample - bounds.left) * viewport.zoom;
    let exact_sy = (origin_y * downsample - bounds.top) * viewport.zoom;
    let exact_sx_end = ((origin_x + tile.width as f64) * downsample - bounds.left) * viewport.zoom;
    let exact_sy_end = ((origin_y + tile.height as f64) * downsample - bounds.top) * viewport.zoom;
    let screen_x = exact_sx.floor() as i32;
    let screen_y = exact_sy.floor() as i32;
    blitter::BlitRect {
        x: screen_x,
        y: screen_y,
        width: exact_sx_end.floor() as i32 - screen_x,
        height: exact_sy_end.floor() as i32 - screen_y,
        exact_x: exact_sx,
        exact_y: exact_sy,
        exact_width: exact_sx_end - exact_sx,
        exact_height: exact_sy_end - exact_sy,
    }
}

/// Cached tiles of the levels coarser than `level`, coarsest first, drawn
/// under the fine tiles while those load; tiles the fine tiles cover
/// completely are left out.
fn fallback_blits(
    tile_manager: &TileManager,
    tile_cache: &TileCache,
    viewport: &Viewport,
    level: u32,
    coverage: Option<&FineCoverage>,
) -> Vec<CpuBlitCommand> {
    let bounds = viewport.bounds();
    let mut commands = Vec::new();
    for fallback_level in (level + 1..tile_manager.wsi().level_count()).rev() {
        let Some(info) = tile_manager.wsi().level(fallback_level).cloned() else {
            continue;
        };
        let tiles = tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );
        let uncovered = tiles
            .iter()
            .filter_map(|coord| Some((coord, tile_cache.peek(coord)?)))
            .filter(|(coord, tile)| {
                !coverage.is_some_and(|coverage| coverage.covers_tile(coord, tile, info.downsample))
            })
            .take(MAX_FALLBACK_TILES_PER_LEVEL);
        for (coord, tile) in uncovered {
            let rect = tile_blit_rect(viewport, coord, &tile, info.downsample);
            if rect.width <= 0 || rect.height <= 0 {
                continue;
            }
            commands.push(CpuBlitCommand {
                tile,
                rect,
                kind: CpuBlitKind::Bilinear,
            });
        }
    }
    commands
}

/// The fine tiles, with their coarse partners when blending trilinearly.
fn fine_blits(
    tile_manager: &TileManager,
    viewport: &Viewport,
    level_info: &eov_common::WsiLevel,
    trilinear: TrilinearLevels,
    cached_tiles: &CachedVisibleTiles,
    use_trilinear: bool,
    use_lanczos: bool,
) -> Vec<CpuBlitCommand> {
    let coarse_tiles: HashMap<_, _> = cached_tiles.coarse_tiles.iter().cloned().collect();
    let blend_coarse = use_trilinear
        && trilinear.level_fine != trilinear.level_coarse
        && trilinear.blend > 0.01
        && !coarse_tiles.is_empty();
    let mut commands = Vec::with_capacity(cached_tiles.fine_tiles.len());
    for (coord, tile) in &cached_tiles.fine_tiles {
        let rect = tile_blit_rect(viewport, coord, tile, level_info.downsample);
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }
        let coarse = blend_coarse
            .then(|| {
                coarse_blend_for_tile(
                    tile_manager,
                    &coarse_tiles,
                    trilinear,
                    level_info.downsample,
                    *coord,
                    tile,
                )
            })
            .flatten();
        let kind = if let Some((coarse_tile, uv_min, uv_max, blend)) = coarse {
            CpuBlitKind::Trilinear {
                coarse_tile,
                uv_min,
                uv_max,
                blend,
            }
        } else if use_lanczos {
            CpuBlitKind::Lanczos3
        } else {
            CpuBlitKind::Bilinear
        };
        commands.push(CpuBlitCommand {
            tile: Arc::clone(tile),
            rect,
            kind,
        });
    }
    commands
}

fn render_pane_gpu(
    file: &mut OpenFile,
    pane: PaneId,
    request: RenderPaneRequest<'_>,
) -> PaneRenderOutcome {
    let RenderPaneRequest {
        ui,
        tile_cache,
        target_width,
        target_height,
        force_render,
        filtering_mode,
        filter_chain,
        pane_uid,
        filter_revision,
        scale,
    } = request;

    let Some(pane_state) = file.pane_state_mut(pane) else {
        return PaneRenderOutcome::default();
    };
    let animating = pane_state.viewport.update();
    pane_state.viewport.set_size(target_width, target_height);
    let last = pane_state.clone();

    let viewport_state = &last.viewport;
    // Frames, levels and change detection are in physical pixels; tiles are
    // drawn axis-aligned with the slide (`aligned`) and the frame turned to
    // the view's orientation as it is composited.
    let physical = physical_viewport(&viewport_state.viewport, scale);
    let vp = &physical;
    let aligned = vp.aligned();
    let is_first_frame = last.frame_count == 0;
    let viewport_changed = animating
        || (last.last_render_zoom - vp.zoom).abs() > 0.001
        || (last.last_render_center_x - vp.center.x).abs() > 1.0
        || (last.last_render_center_y - vp.center.y).abs() > 1.0
        || (last.last_render_width - vp.width).abs() > 1.0
        || (last.last_render_height - vp.height).abs() > 1.0
        || last.last_render_orientation != vp.orientation;

    let trilinear = calculate_trilinear_levels(
        file.tile_manager.wsi(),
        vp.effective_downsample(),
        trilinear_lod_bias_enabled(filtering_mode),
    );
    let level = trilinear.level_fine;
    let margin_tiles = if viewport_state.is_moving() { 1 } else { 0 };
    let level_changed = level != last.last_render_level;
    let (lanczos_weight, use_trilinear_blend) = requested_trilinear_blend(filtering_mode, vp.zoom);
    let cached_tiles = collect_cached_visible_tiles(
        &file.tile_manager,
        tile_cache,
        &aligned,
        trilinear,
        margin_tiles,
        use_trilinear_blend,
    );
    let cached_count = cached_tiles.fine_tiles.len() as u32;
    let loaded_tile_epoch = file.tile_loader.loaded_epoch();
    let new_tiles_loaded = cached_count
        > if level_changed {
            0
        } else {
            last.tiles_loaded_since_render
        }
        || loaded_tile_epoch > last.last_seen_tile_epoch;
    let tiles_pending = file.tile_loader.pending_count() > 0;
    let keep_running = animating || new_tiles_loaded || tiles_pending;
    let adjustments = RenderAdjustments::from_hud(&last.hud);
    let adjustments_changed = adjustments.changed_from(RenderAdjustments::last_rendered(&last));
    let filters_changed = filter_revision != last.last_render_filter_revision;
    let frame_changed = is_first_frame
        || viewport_changed
        || level_changed
        || new_tiles_loaded
        || adjustments_changed;

    if !force_render && !frame_changed && !filters_changed {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    let render_width = vp.width as u32;
    let render_height = vp.height.max(1.0) as u32;
    if render_width == 0 || render_height == 0 {
        return PaneRenderOutcome {
            image: None,
            keep_running,
            rendered: false,
        };
    }

    if let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.frame_count += 1;
        pane_state.last_render_time = std::time::Instant::now();
        pane_state.last_render_zoom = vp.zoom;
        pane_state.last_render_center_x = vp.center.x;
        pane_state.last_render_center_y = vp.center.y;
        pane_state.last_render_width = vp.width;
        pane_state.last_render_height = vp.height;
        pane_state.last_render_level = level;
        pane_state.last_render_orientation = vp.orientation;
        pane_state.tiles_loaded_since_render = cached_count;
        pane_state.last_seen_tile_epoch = loaded_tile_epoch;
        pane_state.last_render_filter_revision = filter_revision;
        adjustments.record(pane_state);
    }

    // Adaptive Lanczos: at low zoom, switch to trilinear on GPU
    let gpu_filtering = if filtering_mode == FilteringMode::Lanczos3 && lanczos_weight < 1.0 {
        FilteringMode::Trilinear
    } else {
        filtering_mode
    };
    let gpu_trilinear = if gpu_filtering == FilteringMode::Trilinear {
        trilinear
    } else {
        single_level_trilinear(trilinear.level_fine)
    };
    let draws = collect_tile_draws_from_cached(
        file,
        tile_cache,
        &aligned,
        gpu_trilinear,
        gpu_filtering,
        &cached_tiles.fine_tiles,
        &cached_tiles.coarse_tiles,
    );
    let color = crate::color::pane_color(file, pane);
    let slot = SurfaceSlot(pane.0);

    // Plugin filters are CPU-only. The frame is still composited on the GPU;
    // it is read back and filtered on the render pool (also when only the
    // filters changed), and the pane shows the newest filtered frame
    // meanwhile (RND-6).
    let filtered = filter_chain.read().has_enabled_cpu_filters();
    let readback = if filtered {
        let current = crate::with_pane_render_cache(pane.0 + 1, |cache| {
            cache
                .get(pane.0)
                .and_then(|entry| entry.cpu_frame.as_ref())
                .is_some_and(|frame| {
                    cached_cpu_frame_matches(frame, file.id, vp, render_width, render_height)
                })
        });
        (frame_changed || filters_changed || !current)
            .then(|| gpu_filters::request(pane_uid, file.id, vp, scale))
            .flatten()
    } else {
        None
    };
    let image = crate::with_gpu_renderer(|renderer| {
        renderer.borrow_mut().queue_frame(
            slot,
            QueuedFrame {
                width: render_width,
                height: render_height,
                orientation: vp.orientation,
                draws,
                color: color.pipeline.gpu_uniform(),
                sharpness: color.pipeline.sharpness(),
                readback,
            },
        )
    })
    .flatten();

    if image.is_some() {
        if let Some(pane_state) = file.pane_state_mut(pane) {
            pane_state.gpu_surface_valid = true;
        }
        ui.window().request_redraw();
    } else if let Some(pane_state) = file.pane_state_mut(pane) {
        pane_state.frame_count = 0;
        pane_state.last_render_level = u32::MAX;
        pane_state.tiles_loaded_since_render = 0;
        pane_state.gpu_surface_valid = false;
    }
    let rendered = image.is_some();

    PaneRenderOutcome {
        // A filtered pane keeps showing its last filtered frame until the
        // new one is ready.
        image: if filtered { None } else { image },
        keep_running: keep_running || !rendered,
        rendered,
    }
}

struct RenderCachedPreview<'a> {
    pane: PaneId,
    pane_uid: u32,
    file_id: i32,
    viewport: &'a Viewport,
    scale: f64,
    render_width: u32,
    render_height: u32,
    fallback_commands: &'a [CpuBlitCommand],
    fine_commands: &'a [CpuBlitCommand],
    postprocess: &'a CpuRenderPostProcess,
    filter_chain: &'a crate::plugin_runtime::filters::SharedFilterChain,
}

/// The frame shown while the view moves: the pane's last unprocessed frame
/// reprojected to the new view, with the cached tiles composited over it,
/// colour-processed and filtered. `None` when there is no usable last frame.
fn render_cached_preview(input: RenderCachedPreview<'_>) -> Option<Image> {
    let RenderCachedPreview {
        pane,
        pane_uid,
        file_id,
        viewport,
        scale,
        render_width,
        render_height,
        fallback_commands,
        fine_commands,
        postprocess,
        filter_chain,
    } = input;
    crate::with_pane_render_cache(pane.0 + 1, |cache| {
        let entry = cache.get_mut(pane.0)?;
        let cpu_frame = entry.cpu_frame.as_ref()?;
        let source_len = (cpu_frame.width as usize) * (cpu_frame.height as usize) * 4;
        if cpu_frame.file_id != file_id
            || cpu_frame.source.len() < source_len
            || cpu_frame.viewport.orientation != viewport.orientation
        {
            return None;
        }
        // Reprojection and compositing happen axis-aligned with the slide;
        // the result is turned to the view's orientation.
        let orientation = viewport.orientation;
        let (source_width, source_height) = orientation.swap(cpu_frame.width, cpu_frame.height);
        let (aligned_width, aligned_height) = orientation.swap(render_width, render_height);
        let aligned_zoom = viewport.zoom;

        let src_bounds = cpu_frame.viewport.bounds();
        let dest_bounds = viewport.bounds();
        let overlap_left = src_bounds.left.max(dest_bounds.left);
        let overlap_top = src_bounds.top.max(dest_bounds.top);
        let overlap_right = src_bounds.right.min(dest_bounds.right);
        let overlap_bottom = src_bounds.bottom.min(dest_bounds.bottom);
        if overlap_right <= overlap_left || overlap_bottom <= overlap_top {
            // A stale preview source with no visible overlap would reproject to
            // an all-background frame. Keep the last displayed pane image
            // instead of flashing black while a fresher CPU job is in flight.
            return None;
        }

        let needed = (render_width as usize) * (render_height as usize) * 4;
        if entry.preview_buffer.len() < needed {
            entry.preview_buffer.resize(needed, 0);
        }
        let preview = &mut entry.preview_buffer[..needed];
        let source = &cpu_frame.source[..source_len];
        let reproject = |band: &mut [u8], band_height: u32, first_row: u32| {
            eov_common::blitter::reproject_frame(
                band,
                aligned_width,
                band_height,
                source,
                source_width,
                source_height,
                src_bounds.left,
                src_bounds.top,
                cpu_frame.viewport.zoom,
                dest_bounds.left,
                dest_bounds.top + f64::from(first_row) / aligned_zoom,
                aligned_zoom,
                [0, 0, 0, 0],
            );
        };
        crate::render_pool::composite_preview(
            preview,
            aligned_width,
            aligned_height,
            &reproject,
            fallback_commands,
            fine_commands,
        );
        let mut frame =
            slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(render_width, render_height);
        let pixels = frame.make_mut_bytes();
        eov_common::blitter::orient_frame(
            preview,
            aligned_width,
            aligned_height,
            orientation,
            pixels,
        );
        crate::render_pool::apply_postprocess(pixels, render_width, render_height, postprocess);
        if let Some(filters) = frame_filters(filter_chain, file_id, pane_uid, viewport, scale) {
            filters.apply(pixels, render_width, render_height);
        }
        Some(Image::from_rgba8(frame))
    })
}

fn collect_tile_draws_from_cached(
    file: &OpenFile,
    tile_cache: &Arc<TileCache>,
    vp: &Viewport,
    trilinear: TrilinearLevels,
    filtering_mode: FilteringMode,
    cached_tiles: &[(eov_common::TileCoord, Arc<eov_common::TileData>)],
    cached_coarse_tiles: &[(eov_common::TileCoord, Arc<eov_common::TileData>)],
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    let bounds = vp.bounds();
    let base_projection = TileProjection {
        viewport: vp,
        bounds_left: bounds.left,
        bounds_top: bounds.top,
        downsample: 1.0,
    };
    let level_count = file.tile_manager.wsi().level_count();
    let coarse_tile_map: HashMap<eov_common::TileCoord, Arc<eov_common::TileData>> =
        if filtering_mode == FilteringMode::Trilinear
            && trilinear.level_fine != trilinear.level_coarse
            && trilinear.blend > 0.01
        {
            cached_coarse_tiles.iter().cloned().collect()
        } else {
            HashMap::new()
        };

    let coverage = FineCoverage::new(
        &file.tile_manager,
        trilinear.level_fine,
        cached_tiles,
        bounds,
    );
    for fallback_level in (0..level_count).rev() {
        if fallback_level <= trilinear.level_fine {
            continue;
        }

        let Some(fallback_level_info) = file.tile_manager.wsi().level(fallback_level) else {
            continue;
        };

        let fallback_tiles = file.tile_manager.visible_tiles(
            fallback_level,
            bounds.left,
            bounds.top,
            bounds.right,
            bounds.bottom,
        );

        let downsample = fallback_level_info.downsample;
        let uncovered = fallback_tiles
            .iter()
            .filter_map(|coord| Some((coord, tile_cache.peek(coord)?)))
            .filter(|(coord, tile)| {
                !coverage
                    .as_ref()
                    .is_some_and(|coverage| coverage.covers_tile(coord, tile, downsample))
            })
            .take(MAX_FALLBACK_TILES_PER_LEVEL);
        for (coord, tile_data) in uncovered {
            let projection = TileProjection {
                downsample: fallback_level_info.downsample,
                ..base_projection
            };
            if let Some(draw) =
                tile_draw_from_tile(projection, *coord, tile_data, None, filtering_mode)
            {
                draws.push(draw);
            }
        }
    }

    let Some(level_info) = file.tile_manager.wsi().level(trilinear.level_fine) else {
        return draws;
    };

    for (coord, tile_data) in cached_tiles.iter() {
        let projection = TileProjection {
            downsample: level_info.downsample,
            ..base_projection
        };
        let coarse_blend = coarse_blend_for_tile(
            &file.tile_manager,
            &coarse_tile_map,
            trilinear,
            level_info.downsample,
            *coord,
            tile_data,
        );
        if let Some(draw) = tile_draw_from_tile(
            projection,
            *coord,
            Arc::clone(tile_data),
            coarse_blend,
            filtering_mode,
        ) {
            draws.push(draw);
        }
    }

    draws
}

fn coarse_blend_for_tile(
    tile_manager: &TileManager,
    coarse_tiles: &HashMap<eov_common::TileCoord, Arc<eov_common::TileData>>,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine_coord: eov_common::TileCoord,
    fine_tile: &Arc<eov_common::TileData>,
) -> Option<CoarseBlendData> {
    const COARSE_BOUNDARY_EPSILON: f64 = 1e-3;

    if trilinear.level_fine == trilinear.level_coarse || trilinear.blend <= 0.01 {
        return None;
    }

    let coarse_info = tile_manager.wsi().level(trilinear.level_coarse)?;
    let image_x = fine_coord.x as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let image_y = fine_coord.y as f64 * fine_coord.tile_size as f64 * fine_downsample;
    let fine_image_w = fine_tile.width as f64 * fine_downsample;
    let fine_image_h = fine_tile.height as f64 * fine_downsample;
    let coarse_tile_size = tile_manager.tile_size_for_level(trilinear.level_coarse) as f64;
    let image_x_end = image_x + fine_image_w;
    let image_y_end = image_y + fine_image_h;

    let coarse_tile_x = image_x / coarse_info.downsample;
    let coarse_tile_y = image_y / coarse_info.downsample;
    let coarse_tile_x_end =
        ((image_x_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_x);
    let coarse_tile_y_end =
        ((image_y_end - COARSE_BOUNDARY_EPSILON) / coarse_info.downsample).max(coarse_tile_y);
    let coarse_start_tile_x = (coarse_tile_x / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_start_tile_y = (coarse_tile_y / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_x = (coarse_tile_x_end / coarse_tile_size).floor().max(0.0) as u64;
    let coarse_end_tile_y = (coarse_tile_y_end / coarse_tile_size).floor().max(0.0) as u64;

    if coarse_start_tile_x != coarse_end_tile_x || coarse_start_tile_y != coarse_end_tile_y {
        return None;
    }

    let coarse_coord = eov_common::TileCoord::new(
        fine_coord.file_id,
        trilinear.level_coarse,
        coarse_start_tile_x,
        coarse_start_tile_y,
        coarse_tile_size as u32,
    );

    let coarse_tile = coarse_tiles.get(&coarse_coord)?.clone();
    let coarse_origin_x = coarse_coord.x as f64 * coarse_coord.tile_size as f64;
    let coarse_origin_y = coarse_coord.y as f64 * coarse_coord.tile_size as f64;
    let coarse_src_x = (coarse_tile_x - coarse_origin_x).max(0.0);
    let coarse_src_y = (coarse_tile_y - coarse_origin_y).max(0.0);
    let coarse_src_w = (fine_image_w / coarse_info.downsample).max(0.0);
    let coarse_src_h = (fine_image_h / coarse_info.downsample).max(0.0);
    let coarse_src_x_end = coarse_src_x + coarse_src_w;
    let coarse_src_y_end = coarse_src_y + coarse_src_h;

    if coarse_src_x_end <= coarse_src_x
        || coarse_src_y_end <= coarse_src_y
        || coarse_src_x_end > coarse_tile.width as f64 + COARSE_BOUNDARY_EPSILON
        || coarse_src_y_end > coarse_tile.height as f64 + COARSE_BOUNDARY_EPSILON
    {
        return None;
    }

    let coarse_width = coarse_tile.width as f64;
    let coarse_height = coarse_tile.height as f64;

    Some((
        coarse_tile,
        [
            (coarse_src_x / coarse_width) as f32,
            (coarse_src_y / coarse_height) as f32,
        ],
        [
            (coarse_src_x_end / coarse_width) as f32,
            (coarse_src_y_end / coarse_height) as f32,
        ],
        trilinear.blend as f32,
    ))
}

fn tile_draw_from_tile(
    projection: TileProjection<'_>,
    coord: eov_common::TileCoord,
    tile_data: Arc<eov_common::TileData>,
    coarse_blend: Option<CoarseBlendData>,
    filtering_mode: FilteringMode,
) -> Option<TileDraw> {
    let TileProjection {
        viewport: vp,
        bounds_left,
        bounds_top,
        downsample,
    } = projection;

    let origin_x = coord.x as f64 * coord.tile_size as f64;
    let origin_y = coord.y as f64 * coord.tile_size as f64;
    let image_x = origin_x * downsample;
    let image_y = origin_y * downsample;
    let image_x_end = (origin_x + tile_data.width as f64) * downsample;
    let image_y_end = (origin_y + tile_data.height as f64) * downsample;

    let screen_x = ((image_x - bounds_left) * vp.zoom) as f32;
    let screen_y = ((image_y - bounds_top) * vp.zoom) as f32;
    let screen_x_end = ((image_x_end - bounds_left) * vp.zoom) as f32;
    let screen_y_end = ((image_y_end - bounds_top) * vp.zoom) as f32;
    let screen_w = screen_x_end - screen_x;
    let screen_h = screen_y_end - screen_y;

    if screen_w <= 0.0 || screen_h <= 0.0 {
        return None;
    }

    let (coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend) = coarse_blend
        .map(|(coarse_tile, coarse_uv_min, coarse_uv_max, mip_blend)| {
            (Some(coarse_tile), coarse_uv_min, coarse_uv_max, mip_blend)
        })
        .unwrap_or((None, [0.0, 0.0], [1.0, 1.0], 0.0));

    Some(TileDraw {
        tile: tile_data,
        coarse_tile,
        screen_x,
        screen_y,
        screen_w,
        screen_h,
        coarse_uv_min,
        coarse_uv_max,
        mip_blend,
        filtering_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::TileCoord;
    use std::time::{Duration, Instant};

    fn tile(coord: TileCoord) -> Arc<eov_common::TileData> {
        Arc::new(eov_common::TileData::new(coord, vec![0; 4], 1, 1, 0))
    }

    fn bounds(
        left: f64,
        top: f64,
        right: f64,
        bottom: f64,
    ) -> eov_common::viewport::ViewportBounds {
        eov_common::viewport::ViewportBounds {
            left,
            right,
            top,
            bottom,
        }
    }

    #[test]
    fn fallback_tiles_hidden_by_fine_tiles_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tif");
        eov_common::test_support::write_tiled_tiff(&path, 512, 512, 256, |_, _| [200, 200, 200])
            .unwrap();
        let manager = TileManager::new(eov_common::WsiFile::open(&path).unwrap(), 1);
        // Level 0 is 2 × 2 tiles; the bottom-right one is not loaded.
        let fine: Vec<_> = [(0, 0), (1, 0), (0, 1)]
            .into_iter()
            .map(|(x, y)| {
                let coord = TileCoord::new(1, 0, x, y, 256);
                (coord, tile(coord))
            })
            .collect();
        let whole = FineCoverage::new(&manager, 0, &fine, bounds(0.0, 0.0, 512.0, 512.0)).unwrap();
        assert!(
            whole.covers(0.0, 0.0, 512.0, 256.0),
            "the top row is loaded"
        );
        assert!(
            whole.covers(0.0, 0.0, 256.0, 512.0),
            "so is the left column"
        );
        assert!(!whole.covers(0.0, 0.0, 512.0, 512.0));
        assert!(!whole.covers(300.0, 300.0, 400.0, 400.0));

        // Only the visible part of a fallback tile counts.
        let coarse = TileCoord::new(1, 1, 0, 0, 256);
        let coarse_tile = Arc::new(eov_common::TileData::new(coarse, vec![0; 4], 256, 256, 0));
        assert!(!whole.covers_tile(&coarse, &coarse_tile, 2.0));
        let top = FineCoverage::new(&manager, 0, &fine, bounds(0.0, 0.0, 512.0, 250.0)).unwrap();
        assert!(top.covers_tile(&coarse, &coarse_tile, 2.0));
        let corner = FineCoverage::new(&manager, 0, &fine, bounds(0.0, 0.0, 256.0, 256.0)).unwrap();
        assert!(
            corner.covers(300.0, 300.0, 400.0, 400.0),
            "nothing of it is visible"
        );
    }

    /// RND-3: frames cover the pane's region at physical resolution, levels
    /// follow the physical zoom, and filters still see logical sizes.
    #[test]
    fn frames_are_rendered_at_physical_resolution() {
        let mut logical = Viewport::new(800.0, 600.0, 10_000.0, 8_000.0);
        logical.zoom = 0.25;
        logical.center.x = 3000.0;
        logical.center.y = 2000.0;
        let physical = physical_viewport(&logical, 2.0);
        assert_eq!((physical.width, physical.height), (1600.0, 1200.0));
        assert_eq!(physical.effective_downsample(), 2.0);
        let (a, b) = (logical.bounds(), physical.bounds());
        assert_eq!(
            (a.left, a.top, a.right, a.bottom),
            (b.left, b.top, b.right, b.bottom),
            "the same region"
        );

        let view = filter_view(1, 7, &physical, 2.0);
        assert_eq!(
            (view.zoom, view.width, view.height, view.scale_factor),
            (0.25, 800.0, 600.0, 2.0)
        );
        assert_eq!((view.visible_x, view.visible_width), (a.left, a.width()));

        // Frames stay within the GPU's texture limit.
        assert_eq!(render_scale(2.0, 800.0, 600.0, 8192), 2.0);
        assert_eq!(render_scale(2.0, 6000.0, 600.0, 8192), 8192.0 / 6000.0);
        assert_eq!(render_scale(f32::NAN, 800.0, 600.0, 8192), 1.0);
    }

    #[test]
    fn panes_share_a_slide_s_tile_requests_in_turns() {
        let coord = |x| TileCoord::new(1, 0, x, 0, 256);
        let merged = merge_wanted_tiles(vec![
            vec![coord(1), coord(2), coord(3), coord(4)],
            vec![coord(7), coord(2), coord(8)],
        ]);
        let xs: Vec<u64> = merged.iter().map(|c| c.x).collect();
        assert_eq!(xs, vec![1, 7, 2, 3, 8, 4], "in turns, without duplicates");
        let single = merge_wanted_tiles(vec![vec![coord(3), coord(1), coord(2)]]);
        assert_eq!(
            single.iter().map(|c| c.x).collect::<Vec<_>>(),
            vec![3, 1, 2],
            "one pane's order is kept"
        );
    }

    /// RND-1: once the view is loaded and rendered the render loop stops,
    /// also between pyramid levels in Trilinear mode, where coarse tiles
    /// are cached too.
    #[test]
    fn the_render_loop_idles_once_the_view_is_loaded() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            crate::render_pool::init_global().unwrap();
            crate::clear_cached_pane(PaneId::PRIMARY);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("pyramid.tif");
            eov_common::test_support::write_tiff(
                &path,
                &eov_common::test_support::TiffSpec::pyramid(1024, 1024, 256, 3),
                |x, y| [(x % 251) as u8, (y % 241) as u8, 128],
            )
            .unwrap();

            let ui = AppWindow::new().unwrap();
            let tile_cache = Arc::new(TileCache::with_limits(256, 256 << 20));
            let state = Arc::new(RwLock::new(AppState::new()));
            {
                let mut state = state.write();
                state.render_backend = RenderBackend::Cpu;
                state.filtering_mode = FilteringMode::Trilinear;
                let id = state.allocate_file_id();
                let wsi = eov_common::WsiFile::open(&path).unwrap();
                let tile_manager = Arc::new(
                    TileManager::new(wsi.with_synthetic_levels(), id)
                        .with_tile_cache(Arc::clone(&tile_cache)),
                );
                let tile_loader = Arc::new(TileLoader::new(
                    Arc::clone(&tile_manager),
                    Arc::clone(&tile_cache),
                ));
                let mut viewport = eov_common::ViewportState::new(400.0, 300.0, 1024.0, 1024.0);
                // Between levels 0 and 1.
                viewport.set_center_zoom(512.0, 512.0, 0.7);
                state.add_file(crate::state::NewOpenFile {
                    id,
                    path: path.clone(),
                    wsi,
                    tile_manager,
                    tile_loader,
                    viewport,
                    thumbnail: None,
                    target_pane: PaneId::PRIMARY,
                    activation: crate::state::TabActivation::Focus,
                });
            }

            let deadline = Instant::now() + Duration::from_secs(30);
            let mut ticks = 0;
            while update_and_render(&ui, &state, &tile_cache) {
                ticks += 1;
                assert!(
                    Instant::now() < deadline,
                    "the render loop still runs after {ticks} ticks"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            for _ in 0..5 {
                std::thread::sleep(Duration::from_millis(20));
                assert!(
                    !update_and_render(&ui, &state, &tile_cache),
                    "and stays idle"
                );
            }

            // The case RND-1 was about: the view blends with cached coarse
            // tiles, and the pane shows a frame.
            let state = state.read();
            let file = &state.open_files[0];
            let vp = &file.pane_state(PaneId::PRIMARY).unwrap().viewport.viewport;
            let trilinear = calculate_trilinear_levels(
                file.tile_manager.wsi(),
                vp.effective_downsample(),
                true,
            );
            assert_ne!(trilinear.level_fine, trilinear.level_coarse);
            let cached = collect_cached_visible_tiles(
                &file.tile_manager,
                &tile_cache,
                vp,
                trilinear,
                0,
                true,
            );
            assert!(!cached.coarse_tiles.is_empty());
            assert!(!cached.fine_tiles.is_empty());
            let shown = crate::with_pane_render_cache(1, |cache| cache[0].content.is_some());
            assert!(shown);
        });
    }

    /// RND-7: a GPU error switches the slides to the CPU renderer instead
    /// of crashing, and the GPU stays off.
    #[test]
    fn a_failed_gpu_hands_the_slides_to_the_cpu_renderer() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = AppWindow::new().unwrap();
            let tile_cache = Arc::new(TileCache::with_limits(16, 16 << 20));
            let state = Arc::new(RwLock::new(AppState::new()));
            {
                let mut state = state.write();
                state.gpu_backend_available = true;
                state.select_render_backend(RenderBackend::Gpu);
            }
            crate::gpu::record_failure("GPU error: test".to_string());
            update_and_render(&ui, &state, &tile_cache);
            let state = state.read();
            assert_eq!(state.render_backend, RenderBackend::Cpu);
            assert!(!state.gpu_backend_available);
            assert!(!ui.get_gpu_rendering_available());
            assert_eq!(ui.get_status_text(), GPU_FAILED_MESSAGE);
            assert!(crate::gpu::take_failure().is_none(), "reported once");
        });
    }

    fn receive_filtered() -> (Vec<gpu_filters::FilteredFrame>, bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (frames, again) = gpu_filters::receive();
            if !frames.is_empty() || Instant::now() > deadline {
                return (frames, again);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn gpu_frames_are_filtered_one_at_a_time_per_pane() {
        let viewport = Viewport::new(4.0, 2.0, 100.0, 100.0);
        let chain = crate::plugin_runtime::filters::SharedFilterChain::default();
        let first = gpu_filters::request(5, 1, &viewport, 1.0).expect("an idle pane reads back");
        assert!(
            gpu_filters::request(5, 1, &viewport, 1.0).is_none(),
            "one frame per pane at a time"
        );
        assert!(
            gpu_filters::request(6, 1, &viewport, 1.0).is_some(),
            "panes are independent"
        );
        assert!(gpu_filters::busy());

        let pixels: Vec<u8> = (0..32).collect();
        let frame = ReadbackFrame {
            tag: first,
            width: 4,
            height: 2,
            pixels: pixels.clone(),
        };
        assert!(!gpu_filters::dispatch(vec![frame], true, &chain));
        let (frames, again) = receive_filtered();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].tag.pane_uid, 5);
        assert_eq!(frames[0].display.as_bytes(), &pixels[..]);
        assert_eq!(frames[0].unfiltered, pixels);
        assert!(again, "pane 5 changed while its frame was filtered");

        // The renderer lost pane 6's readback: the pane renders again.
        assert!(gpu_filters::dispatch(Vec::new(), false, &chain));
        assert!(!gpu_filters::busy());
        assert!(gpu_filters::request(6, 1, &viewport, 1.0).is_some());
    }
}
