use crate::blitter;
use crate::stain;
use common::{TileData, Viewport};
use parking_lot::Mutex;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct CachedCpuFrame {
    pub file_id: i32,
    pub width: u32,
    pub height: u32,
    pub viewport: Viewport,
    pub pixels: Vec<u8>,
}

#[derive(Clone)]
pub struct CpuRenderPostProcess {
    pub stain_params: Option<crate::stain::StainNormParams>,
    pub sharpness: f32,
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
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
    pub background_rgba: [u8; 4],
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
    pub pixels: Vec<u8>,
    pub settled_quality: bool,
}

pub struct RenderWorkerPool {
    pool: ThreadPool,
    results_tx: crossbeam_channel::Sender<CpuRenderResult>,
    results_rx: crossbeam_channel::Receiver<CpuRenderResult>,
    next_job_id: AtomicU64,
    recycled_buffers: Mutex<Vec<Vec<u8>>>,
}

static GLOBAL_RENDER_POOL: OnceLock<Arc<RenderWorkerPool>> = OnceLock::new();

impl RenderWorkerPool {
    pub fn new() -> anyhow::Result<Self> {
        let thread_count = std::thread::available_parallelism()
            .map(|count| count.get().clamp(2, 16))
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
        })
    }

    pub fn next_job_id(&self) -> u64 {
        self.next_job_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn submit(&self, job: CpuRenderJob) {
        let results_tx = self.results_tx.clone();
        let mut buffer = self.acquire_buffer((job.width as usize) * (job.height as usize) * 4);

        self.pool.spawn_fifo(move || {
            render_job_into_buffer(&job, &mut buffer);
            let _ = results_tx.send(CpuRenderResult {
                pane_index: job.pane_index,
                file_id: job.file_id,
                job_id: job.job_id,
                width: job.width,
                height: job.height,
                viewport: job.viewport,
                pixels: buffer,
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
        if let Some(index) = buffers.iter().position(|buffer| buffer.capacity() >= needed) {
            let mut buffer = buffers.swap_remove(index);
            buffer.resize(needed, 0);
            buffer
        } else {
            vec![0; needed]
        }
    }
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

fn render_job_into_buffer(job: &CpuRenderJob, buffer: &mut [u8]) {
    let width = job.width;
    let height = job.height;
    let stride = (width as usize) * 4;
    let rows_per_chunk = ((height as usize) / rayon::current_num_threads()).max(32);

    buffer
        .par_chunks_mut(rows_per_chunk * stride)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start_row = chunk_index * rows_per_chunk;
            let chunk_rows = chunk.len() / stride;
            let chunk_height = chunk_rows as u32;
            let row_offset = start_row as i32;

            blitter::fast_fill_rgba(
                chunk,
                job.background_rgba[0],
                job.background_rgba[1],
                job.background_rgba[2],
                job.background_rgba[3],
            );

            for command in &job.fallback_blits {
                draw_command_into_chunk(chunk, width, chunk_height, row_offset, command);
            }
            for command in &job.fine_blits {
                draw_command_into_chunk(chunk, width, chunk_height, row_offset, command);
            }
        });

    if let Some(ref params) = job.postprocess.stain_params {
        stain::apply_stain_params_to_buffer(buffer, params);
    }

    if job.postprocess.sharpness > 0.001 {
        apply_sharpening(buffer, width, height, job.postprocess.sharpness);
    }

    let has_adjustments = (job.postprocess.gamma - 1.0).abs() > 0.001
        || job.postprocess.brightness.abs() > 0.001
        || (job.postprocess.contrast - 1.0).abs() > 0.001;
    if has_adjustments {
        apply_adjustments(
            buffer,
            job.postprocess.gamma,
            job.postprocess.brightness,
            job.postprocess.contrast,
        );
    }
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

fn apply_adjustments(buffer: &mut [u8], gamma: f32, brightness: f32, contrast: f32) {
    let inv_gamma = if gamma > 0.001 { 1.0 / gamma } else { 1.0 };
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let normalized = i as f32 / 255.0;
        let g = normalized.powf(inv_gamma);
        let b = g + brightness;
        let c = (b - 0.5) * contrast + 0.5;
        *entry = (c * 255.0).clamp(0.0, 255.0) as u8;
    }
    buffer.par_chunks_mut(4).for_each(|chunk| {
        chunk[0] = lut[chunk[0] as usize];
        chunk[1] = lut[chunk[1] as usize];
        chunk[2] = lut[chunk[2] as usize];
    });
}

fn apply_sharpening(buffer: &mut [u8], width: u32, height: u32, sharpness: f32) {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return;
    }
    let src = buffer.to_vec();
    let stride = w * 4;

    buffer
        .par_chunks_mut(stride)
        .enumerate()
        .skip(1)
        .take(h.saturating_sub(2))
        .for_each(|(y, row)| {
            for x in 1..w - 1 {
                let idx = y * stride + x * 4;
                let row_idx = x * 4;
                let n = idx - stride;
                let s = idx + stride;
                let we = idx - 4;
                let e = idx + 4;
                for c in 0..3 {
                    let center = src[idx + c] as f32;
                    let neighbors = src[n + c] as f32
                        + src[s + c] as f32
                        + src[we + c] as f32
                        + src[e + c] as f32;
                    let detail = center * 4.0 - neighbors;
                    let sharpened = center + sharpness * detail;
                    row[row_idx + c] = sharpened.clamp(0.0, 255.0) as u8;
                }
            }
        });
}