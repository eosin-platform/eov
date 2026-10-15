//! CPU frame rendering on a worker pool.
//!
//! A job composites a pane's tiles, colour-processes the frame, runs the
//! plugin filters and builds the pixel buffer Slint displays — all on the
//! pool, so the UI thread only swaps images in (RND-6). Jobs are keyed by
//! the pane's stable id, not its position (RND-9).

use crate::blitter;
use crate::plugin_runtime::filters::ChainSnapshot;
use eov_common::processing::ProcessingPipeline;
use eov_common::{TileData, Viewport};
use eov_plugin_api::abi::FilterView;
use parking_lot::Mutex;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use slint::{Rgba8Pixel, SharedPixelBuffer};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Recycled frame buffers kept for reuse; more are freed.
const MAX_RECYCLED_BUFFERS: usize = 8;

#[derive(Debug)]
pub struct CachedCpuFrame {
    pub file_id: i32,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    /// The frame as displayed before plugin filters: colour-processed and
    /// composited onto the background (filters run again on this one when
    /// only they change).
    pub pixels: Vec<u8>,
    /// The same frame before colour processing, transparent outside the
    /// slide, and axis-aligned with the slide (`viewport.aligned()`, not
    /// turned). Interaction previews reproject this one, so that no pixel is
    /// colour-processed twice. Empty for frames rendered on the GPU.
    pub source: Vec<u8>,
}

/// Colour processing of a CPU frame.
#[derive(Clone)]
pub struct CpuRenderPostProcess {
    /// Stages 2–6 of `eov_common::processing`: stain transform, tone,
    /// quantisation, compositing and screen-space sharpening.
    pub pipeline: Arc<ProcessingPipeline>,
    /// Colour of the pixels outside the slide.
    pub background: [u8; 3],
}

impl CpuRenderPostProcess {
    /// Processes a frame whose pixels outside the slide are transparent.
    pub fn apply(&self, buffer: &mut [u8], width: u32, height: u32) {
        self.pipeline
            .apply_rgba(buffer, width, height, self.background);
    }
}

/// The plugin filters to run on a finished frame, and what they are told
/// about the view the frame shows.
#[derive(Clone)]
pub struct FrameFilters {
    pub chain: Arc<ChainSnapshot>,
    pub view: FilterView,
}

impl FrameFilters {
    /// Runs the filters over a tightly packed RGBA frame.
    pub fn apply(&self, pixels: &mut [u8], width: u32, height: u32) {
        self.chain
            .apply_cpu(pixels, width, height, Some(&self.view));
    }
}

#[derive(Clone)]
pub enum CpuBlitKind {
    Bilinear,
    Lanczos3,
    Trilinear {
        coarse_tile: Arc<TileData>,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        blend: f32,
    },
}

#[derive(Clone)]
pub struct CpuBlitCommand {
    pub tile: Arc<TileData>,
    pub rect: blitter::BlitRect,
    pub kind: CpuBlitKind,
}

pub struct CpuRenderJob {
    /// Stable id of the pane (`PaneState::uid`): pane positions change when
    /// panes are inserted or removed while a job runs (RND-9).
    pub pane_uid: u32,
    pub file_id: i32,
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    pub fallback_blits: Vec<CpuBlitCommand>,
    pub fine_blits: Vec<CpuBlitCommand>,
    pub postprocess: CpuRenderPostProcess,
    /// Plugin filters, run on the worker for this frame's own view.
    pub filters: Option<FrameFilters>,
    pub settled_quality: bool,
}

pub struct CpuRenderResult {
    /// Stable id of the pane the frame was rendered for.
    pub pane_uid: u32,
    pub file_id: i32,
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    /// The processed frame before plugin filters (see
    /// [`CachedCpuFrame::pixels`]).
    pub pixels: Vec<u8>,
    /// The frame before colour processing (see [`CachedCpuFrame::source`]).
    pub source: Vec<u8>,
    /// What the pane displays: the filtered frame, ready for
    /// `slint::Image::from_rgba8` without another copy.
    pub display: SharedPixelBuffer<Rgba8Pixel>,
    pub settled_quality: bool,
}

pub struct RenderWorkerPool {
    pool: ThreadPool,
    results_tx: crossbeam_channel::Sender<CpuRenderResult>,
    results_rx: crossbeam_channel::Receiver<CpuRenderResult>,
    next_job_id: AtomicU64,
    recycled_buffers: Mutex<Vec<Vec<u8>>>,
    /// Pane uid → its newest job; older jobs of the pane stop early.
    latest_jobs: Arc<Mutex<HashMap<u32, u64>>>,
}

static GLOBAL_RENDER_POOL: OnceLock<Arc<RenderWorkerPool>> = OnceLock::new();

impl RenderWorkerPool {
    pub fn new() -> anyhow::Result<Self> {
        let thread_count = std::thread::available_parallelism()
            .map(|count| count.get().clamp(4, 32))
            .unwrap_or(4);
        let pool = ThreadPoolBuilder::new()
            .thread_name(|index| format!("cpu-render-{index}"))
            .num_threads(thread_count)
            .build()?;
        let (results_tx, results_rx) = crossbeam_channel::unbounded();

        Ok(Self {
            pool,
            results_tx,
            results_rx,
            next_job_id: AtomicU64::new(1),
            recycled_buffers: Mutex::new(Vec::new()),
            latest_jobs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn next_job_id(&self) -> u64 {
        self.next_job_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn submit(&self, job: CpuRenderJob) {
        let results_tx = self.results_tx.clone();
        self.latest_jobs.lock().insert(job.pane_uid, job.job_id);
        let latest_jobs = Arc::clone(&self.latest_jobs);
        let needed = (job.width as usize) * (job.height as usize) * 4;
        let mut buffer = self.acquire_buffer(needed);
        let mut source = self.acquire_buffer(needed);

        self.pool.spawn_fifo(move || {
            let Some(display) = render_job(&job, &latest_jobs, &mut buffer, &mut source) else {
                return;
            };
            let _ = results_tx.send(CpuRenderResult {
                pane_uid: job.pane_uid,
                file_id: job.file_id,
                job_id: job.job_id,
                width: job.width,
                height: job.height,
                viewport: job.viewport,
                pixels: buffer,
                source,
                display,
                settled_quality: job.settled_quality,
            });
        });
    }

    /// Stops the jobs of panes that no longer exist.
    pub fn retain_panes(&self, live: &[u32]) {
        self.latest_jobs.lock().retain(|uid, _| live.contains(uid));
    }

    /// Runs `f` on the pool (for work that must not block the UI thread).
    pub fn spawn(&self, f: impl FnOnce() + Send + 'static) {
        self.pool.spawn_fifo(f);
    }

    pub fn try_recv(&self) -> Option<CpuRenderResult> {
        self.results_rx.try_recv().ok()
    }

    pub fn recycle_buffer(&self, mut buffer: Vec<u8>) {
        if buffer.capacity() == 0 {
            return;
        }
        buffer.clear();
        let mut buffers = self.recycled_buffers.lock();
        buffers.push(buffer);
        if buffers.len() > MAX_RECYCLED_BUFFERS {
            // Keep the largest ones: they serve any smaller frame too.
            buffers.sort_unstable_by_key(|b| std::cmp::Reverse(b.capacity()));
            buffers.truncate(MAX_RECYCLED_BUFFERS);
        }
    }

    fn acquire_buffer(&self, needed: usize) -> Vec<u8> {
        let mut buffers = self.recycled_buffers.lock();
        // The smallest buffer that fits, so large ones stay for large frames.
        let best = buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= needed)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index);
        if let Some(index) = best {
            let mut buffer = buffers.swap_remove(index);
            buffer.resize(needed, 0);
            buffer
        } else {
            vec![0; needed]
        }
    }

    #[cfg(test)]
    fn recycled_count(&self) -> usize {
        self.recycled_buffers.lock().len()
    }
}

fn job_is_current(latest_jobs: &Mutex<HashMap<u32, u64>>, pane_uid: u32, job_id: u64) -> bool {
    latest_jobs
        .lock()
        .get(&pane_uid)
        .is_some_and(|current_job_id| *current_job_id == job_id)
}

pub fn init_global() -> anyhow::Result<()> {
    if GLOBAL_RENDER_POOL.get().is_none() {
        let pool = Arc::new(RenderWorkerPool::new()?);
        let _ = GLOBAL_RENDER_POOL.set(pool);
    }
    Ok(())
}

pub fn global() -> Option<&'static Arc<RenderWorkerPool>> {
    GLOBAL_RENDER_POOL.get()
}

/// Rows per parallel band of a `height`-row frame.
fn rows_per_band(height: u32) -> usize {
    ((height as usize) / rayon::current_num_threads()).max(32)
}

/// Composites `fallback` then `fine` blits into `buffer` (cleared to
/// transparent first) in parallel row bands. Stops early, leaving the
/// buffer incomplete, once `cancelled()` returns `true`.
pub fn composite_frame(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    fallback: &[CpuBlitCommand],
    fine: &[CpuBlitCommand],
    cancelled: &(dyn Fn() -> bool + Sync),
) {
    let stride = (width as usize) * 4;
    if stride == 0 || height == 0 {
        return;
    }
    let rows = rows_per_band(height);
    buffer
        .par_chunks_mut(rows * stride)
        .enumerate()
        .for_each(|(band, chunk)| {
            if cancelled() {
                return;
            }
            let chunk_height = (chunk.len() / stride) as u32;
            let row_offset = (band * rows) as i32;
            // Transparent until a tile covers it: the pipeline composites
            // the rest onto the background after colour processing.
            chunk.fill(0);
            for command in fallback.iter().chain(fine) {
                if cancelled() {
                    return;
                }
                draw_command_into_chunk(chunk, width, chunk_height, row_offset, command);
            }
        });
}

/// Fills each row band of `buffer` with `reproject(band, band_height,
/// first_row)` (an earlier frame moved to this view), then composites the
/// blits over it, in parallel.
pub fn composite_preview(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    reproject: &(dyn Fn(&mut [u8], u32, u32) + Sync),
    fallback: &[CpuBlitCommand],
    fine: &[CpuBlitCommand],
) {
    let stride = (width as usize) * 4;
    if stride == 0 || height == 0 {
        return;
    }
    let rows = rows_per_band(height);
    buffer
        .par_chunks_mut(rows * stride)
        .enumerate()
        .for_each(|(band, chunk)| {
            let chunk_height = (chunk.len() / stride) as u32;
            let first_row = band * rows;
            reproject(chunk, chunk_height, first_row as u32);
            for command in fallback.iter().chain(fine) {
                draw_command_into_chunk(chunk, width, chunk_height, first_row as i32, command);
            }
        });
}

/// Composites the job's tiles into `source` (axis-aligned with the slide,
/// transparent outside it, unprocessed), turns that frame to the view's
/// orientation into `buffer`, colour-processes `buffer`, and returns the
/// filtered frame to display. `None` when a newer job for the pane
/// superseded this one.
fn render_job(
    job: &CpuRenderJob,
    latest_jobs: &Mutex<HashMap<u32, u64>>,
    buffer: &mut [u8],
    source: &mut [u8],
) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let current = || job_is_current(latest_jobs, job.pane_uid, job.job_id);
    if !current() {
        return None;
    }
    let cancelled = || !current();
    // The blits are laid out axis-aligned with the slide; the frame is
    // turned to the view's orientation after compositing.
    let orientation = job.viewport.orientation;
    let (aligned_width, aligned_height) = orientation.swap(job.width, job.height);
    composite_frame(
        source,
        aligned_width,
        aligned_height,
        &job.fallback_blits,
        &job.fine_blits,
        &cancelled,
    );
    if !current() {
        return None;
    }

    eov_common::blitter::orient_frame(source, aligned_width, aligned_height, orientation, buffer);
    job.postprocess.apply(buffer, job.width, job.height);
    if !current() {
        return None;
    }

    let mut display = SharedPixelBuffer::<Rgba8Pixel>::new(job.width, job.height);
    let bytes = display.make_mut_bytes();
    let len = bytes.len().min(buffer.len());
    bytes[..len].copy_from_slice(&buffer[..len]);
    if let Some(filters) = &job.filters {
        filters.apply(bytes, job.width, job.height);
    }
    current().then_some(display)
}

/// Colour-processes an interaction preview (the synchronous counterpart of
/// a render job): `buffer` must be transparent outside the slide.
pub fn apply_postprocess(buffer: &mut [u8], width: u32, height: u32, pp: &CpuRenderPostProcess) {
    pp.apply(buffer, width, height);
}

fn draw_command_into_chunk(
    chunk: &mut [u8],
    width: u32,
    chunk_height: u32,
    row_offset: i32,
    command: &CpuBlitCommand,
) {
    let mut rect = command.rect;
    rect.y -= row_offset;
    rect.exact_y -= row_offset as f64;

    let tile = blitter::TileSrc {
        data: &command.tile.data,
        width: command.tile.width,
        height: command.tile.height,
        border: command.tile.border,
    };
    match &command.kind {
        CpuBlitKind::Bilinear => blitter::blit_tile(chunk, width, chunk_height, tile, rect),
        CpuBlitKind::Lanczos3 => {
            blitter::blit_tile_lanczos3(chunk, width, chunk_height, tile, rect)
        }
        CpuBlitKind::Trilinear {
            coarse_tile,
            uv_min,
            uv_max,
            blend,
        } => blitter::blit_tile_trilinear(
            chunk,
            width,
            chunk_height,
            tile,
            &blitter::CoarseSrc {
                data: &coarse_tile.data,
                width: coarse_tile.width,
                height: coarse_tile.height,
                border: coarse_tile.border,
                uv_min: *uv_min,
                uv_max: *uv_max,
                blend: *blend,
            },
            rect,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::TileCoord;
    use eov_common::postprocess::ToneAdjustments;

    fn job_with_tile(pipeline: ProcessingPipeline) -> CpuRenderJob {
        // A 2×2 tile at the left of a 4×2 frame; the right half is outside
        // the slide.
        let tile = Arc::new(TileData::new(
            TileCoord::new(1, 0, 0, 0, 2),
            [100u8, 150, 200, 255].repeat(4),
            2,
            2,
            0,
        ));
        CpuRenderJob {
            pane_uid: 3,
            file_id: 1,
            job_id: 7,
            width: 4,
            height: 2,
            viewport: Viewport::new(4.0, 2.0, 2.0, 2.0),
            fallback_blits: Vec::new(),
            fine_blits: vec![CpuBlitCommand {
                tile,
                rect: blitter::BlitRect {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                    exact_x: 0.0,
                    exact_y: 0.0,
                    exact_width: 2.0,
                    exact_height: 2.0,
                },
                kind: CpuBlitKind::Bilinear,
            }],
            postprocess: CpuRenderPostProcess {
                pipeline: Arc::new(pipeline),
                background: [30, 30, 30],
            },
            filters: None,
            settled_quality: true,
        }
    }

    #[test]
    fn slide_pixels_are_processed_once_and_the_rest_is_background() {
        let pipeline =
            ProcessingPipeline::new(None, ToneAdjustments::new(2.0, 0.1, 1.2), 0.0).unwrap();
        let expected = pipeline.color_pixel([100, 150, 200]);
        let job = job_with_tile(pipeline);
        let latest = Mutex::new(HashMap::from([(3, 7)]));
        let mut pixels = vec![0u8; 4 * 2 * 4];
        let mut source = vec![0u8; 4 * 2 * 4];
        let display = render_job(&job, &latest, &mut pixels, &mut source).unwrap();
        for row in pixels.chunks(16) {
            assert_eq!(&row[..3], &expected);
            assert_eq!(row[3], 255);
            // The background is not tone-mapped (stage 5 follows stage 4).
            assert_eq!(&row[8..16], &[30, 30, 30, 255, 30, 30, 30, 255]);
        }
        // Without filters the displayed frame is the processed one.
        assert_eq!(display.as_bytes(), pixels.as_slice());
        // The unprocessed frame keeps the transparency outside the slide.
        for row in source.chunks(16) {
            assert_eq!(&row[..8], &[100, 150, 200, 255, 100, 150, 200, 255]);
            assert_eq!(&row[8..16], &[0; 8]);
        }
        // The synchronous preview path gives the same result.
        let mut preview = source.clone();
        apply_postprocess(&mut preview, 4, 2, &job.postprocess);
        assert_eq!(preview, pixels);
    }

    /// A turned view's frame is the aligned composite turned (exactly), and
    /// its unprocessed source stays aligned for reprojection.
    #[test]
    fn turned_jobs_turn_the_composited_frame() {
        let plain = job_with_tile(ProcessingPipeline::identity());
        let latest = Mutex::new(HashMap::from([(3, 7)]));
        let (mut aligned, mut aligned_source) = (vec![0u8; 32], vec![0u8; 32]);
        render_job(&plain, &latest, &mut aligned, &mut aligned_source).unwrap();
        for orientation in eov_common::viewport::Orientation::all() {
            let mut job = job_with_tile(ProcessingPipeline::identity());
            let (w, h) = orientation.swap(4, 2);
            job.width = w;
            job.height = h;
            job.viewport.orientation = orientation;
            let (mut pixels, mut source) = (vec![0u8; 32], vec![0u8; 32]);
            render_job(&job, &latest, &mut pixels, &mut source).unwrap();
            let mut expected = vec![0u8; 32];
            eov_common::blitter::orient_frame(&aligned, 4, 2, orientation, &mut expected);
            assert_eq!(pixels, expected, "{orientation:?}");
            assert_eq!(source, aligned_source, "{orientation:?}");
        }
    }

    #[test]
    fn superseded_jobs_are_dropped() {
        let job = job_with_tile(ProcessingPipeline::identity());
        let latest = Mutex::new(HashMap::from([(3, 8)]));
        let mut pixels = vec![0u8; 32];
        let mut source = vec![0u8; 32];
        assert!(render_job(&job, &latest, &mut pixels, &mut source).is_none());
    }

    #[test]
    fn jobs_follow_their_pane_not_its_position() {
        let pool = RenderWorkerPool::new().unwrap();
        // Two panes each have a job in flight; which result belongs to
        // which pane does not depend on where the panes are.
        let mut first = job_with_tile(ProcessingPipeline::identity());
        first.pane_uid = 9;
        first.job_id = pool.next_job_id();
        let mut second = job_with_tile(ProcessingPipeline::identity());
        second.job_id = pool.next_job_id();
        pool.submit(first);
        pool.submit(second);
        let mut uids = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while uids.len() < 2 && std::time::Instant::now() < deadline {
            match pool.try_recv() {
                Some(result) => uids.push(result.pane_uid),
                None => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
        uids.sort_unstable();
        assert_eq!(uids, vec![3, 9]);
        // A removed pane's jobs stop counting as current.
        pool.retain_panes(&[3]);
        assert!(!job_is_current(&pool.latest_jobs, 9, 1));
    }

    #[test]
    fn recycled_buffers_are_bounded_and_reused_by_size() {
        let pool = RenderWorkerPool::new().unwrap();
        for size in 1..=20 {
            pool.recycle_buffer(vec![0; size * 100]);
        }
        assert_eq!(pool.recycled_count(), MAX_RECYCLED_BUFFERS);
        // The smallest fitting buffer is handed out.
        let buffer = pool.acquire_buffer(1_350);
        assert_eq!(buffer.capacity(), 1_400);
        assert_eq!(buffer.len(), 1_350);
    }
}
