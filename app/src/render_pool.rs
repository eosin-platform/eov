use crate::blitter;
use eov_common::processing::ProcessingPipeline;
use eov_common::{TileData, Viewport};
use parking_lot::Mutex;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct CachedCpuFrame {
    pub file_id: i32,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    /// The displayed frame: colour-processed and composited onto the
    /// background (before plugin filters).
    pub pixels: Vec<u8>,
    /// The same frame before colour processing, transparent outside the
    /// slide. Interaction previews reproject this one, so that no pixel is
    /// colour-processed twice.
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
    pub pane_index: usize,
    pub file_id: i32,
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    pub fallback_blits: Vec<CpuBlitCommand>,
    pub fine_blits: Vec<CpuBlitCommand>,
    pub postprocess: CpuRenderPostProcess,
    pub settled_quality: bool,
}

pub struct CpuRenderResult {
    pub pane_index: usize,
    pub file_id: i32,
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    /// The processed frame.
    pub pixels: Vec<u8>,
    /// The frame before colour processing (see [`CachedCpuFrame::source`]).
    pub source: Vec<u8>,
    pub settled_quality: bool,
}

pub struct RenderWorkerPool {
    pool: ThreadPool,
    results_tx: crossbeam_channel::Sender<CpuRenderResult>,
    results_rx: crossbeam_channel::Receiver<CpuRenderResult>,
    next_job_id: AtomicU64,
    recycled_buffers: Mutex<Vec<Vec<u8>>>,
    latest_jobs: Arc<Mutex<HashMap<usize, u64>>>,
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
        self.latest_jobs.lock().insert(job.pane_index, job.job_id);
        let latest_jobs = Arc::clone(&self.latest_jobs);
        let needed = (job.width as usize) * (job.height as usize) * 4;
        let mut buffer = self.acquire_buffer(needed);
        let mut source = self.acquire_buffer(needed);

        self.pool.spawn_fifo(move || {
            if !render_job_into_buffer(&job, &latest_jobs, &mut buffer, &mut source) {
                return;
            }
            let _ = results_tx.send(CpuRenderResult {
                pane_index: job.pane_index,
                file_id: job.file_id,
                job_id: job.job_id,
                width: job.width,
                height: job.height,
                viewport: job.viewport,
                pixels: buffer,
                source,
                settled_quality: job.settled_quality,
            });
        });
    }

    pub fn try_recv(&self) -> Option<CpuRenderResult> {
        self.results_rx.try_recv().ok()
    }

    pub fn recycle_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        self.recycled_buffers.lock().push(buffer);
    }

    fn acquire_buffer(&self, needed: usize) -> Vec<u8> {
        let mut buffers = self.recycled_buffers.lock();
        if let Some(index) = buffers
            .iter()
            .position(|buffer| buffer.capacity() >= needed)
        {
            let mut buffer = buffers.swap_remove(index);
            buffer.resize(needed, 0);
            buffer
        } else {
            vec![0; needed]
        }
    }
}

fn job_is_current(
    latest_jobs: &Mutex<HashMap<usize, u64>>,
    pane_index: usize,
    job_id: u64,
) -> bool {
    latest_jobs
        .lock()
        .get(&pane_index)
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

/// Composites the job's tiles into `buffer` (transparent outside the slide),
/// keeps that unprocessed frame in `source`, then colour-processes `buffer`.
/// Returns `false` when a newer job for the pane superseded this one.
fn render_job_into_buffer(
    job: &CpuRenderJob,
    latest_jobs: &Mutex<HashMap<usize, u64>>,
    buffer: &mut [u8],
    source: &mut [u8],
) -> bool {
    if !job_is_current(latest_jobs, job.pane_index, job.job_id) {
        return false;
    }

    let width = job.width;
    let height = job.height;
    let stride = (width as usize) * 4;
    let rows_per_chunk = ((height as usize) / rayon::current_num_threads()).max(32);

    buffer
        .par_chunks_mut(rows_per_chunk * stride)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            if !job_is_current(latest_jobs, job.pane_index, job.job_id) {
                return;
            }

            let start_row = chunk_index * rows_per_chunk;
            let chunk_rows = chunk.len() / stride;
            let chunk_height = chunk_rows as u32;
            let row_offset = start_row as i32;

            // Transparent until a tile covers it: the pipeline composites
            // the rest onto the background after colour processing.
            chunk.fill(0);

            for command in &job.fallback_blits {
                if !job_is_current(latest_jobs, job.pane_index, job.job_id) {
                    return;
                }
                draw_command_into_chunk(chunk, width, chunk_height, row_offset, command);
            }
            for command in &job.fine_blits {
                if !job_is_current(latest_jobs, job.pane_index, job.job_id) {
                    return;
                }
                draw_command_into_chunk(chunk, width, chunk_height, row_offset, command);
            }
        });

    if !job_is_current(latest_jobs, job.pane_index, job.job_id) {
        return false;
    }

    let len = buffer.len().min(source.len());
    source[..len].copy_from_slice(&buffer[..len]);
    job.postprocess.apply(buffer, width, height);

    job_is_current(latest_jobs, job.pane_index, job.job_id)
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

    match &command.kind {
        CpuBlitKind::Bilinear => blitter::blit_tile(
            chunk,
            width,
            chunk_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
            rect,
        ),
        CpuBlitKind::Lanczos3 => blitter::blit_tile_lanczos3(
            chunk,
            width,
            chunk_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
            rect,
        ),
        CpuBlitKind::Trilinear {
            coarse_tile,
            uv_min,
            uv_max,
            blend,
        } => blitter::blit_tile_trilinear(
            chunk,
            width,
            chunk_height,
            blitter::TileSrc {
                data: &command.tile.data,
                width: command.tile.width,
                height: command.tile.height,
                border: command.tile.border,
            },
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
            pane_index: 0,
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
            settled_quality: true,
        }
    }

    #[test]
    fn slide_pixels_are_processed_once_and_the_rest_is_background() {
        let pipeline =
            ProcessingPipeline::new(None, ToneAdjustments::new(2.0, 0.1, 1.2), 0.0).unwrap();
        let expected = pipeline.color_pixel([100, 150, 200]);
        let job = job_with_tile(pipeline);
        let latest = Mutex::new(HashMap::from([(0, 7)]));
        let mut pixels = vec![0u8; 4 * 2 * 4];
        let mut source = vec![0u8; 4 * 2 * 4];
        assert!(render_job_into_buffer(
            &job,
            &latest,
            &mut pixels,
            &mut source
        ));
        for row in pixels.chunks(16) {
            assert_eq!(&row[..3], &expected);
            assert_eq!(row[3], 255);
            // The background is not tone-mapped (stage 5 follows stage 4).
            assert_eq!(&row[8..16], &[30, 30, 30, 255, 30, 30, 30, 255]);
        }
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

    #[test]
    fn superseded_jobs_are_dropped() {
        let job = job_with_tile(ProcessingPipeline::identity());
        let latest = Mutex::new(HashMap::from([(0, 8)]));
        let mut pixels = vec![0u8; 32];
        let mut source = vec![0u8; 32];
        assert!(!render_job_into_buffer(
            &job,
            &latest,
            &mut pixels,
            &mut source
        ));
    }
}
