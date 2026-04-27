use crate::db::{AsyncLatentCacheWriter, LatentCacheKey, load_tiles_for_positions};
use crate::model::{
    BatchReconstructionInput, LoadedModel, ReconstructionResult, run_reconstruction,
    run_reconstruction_batch,
};
use crate::state::{
    AnalysisPhase, JobKind, RunningJob, VisualizationMode, host_api, log_message, plugin_state,
    rebuild_sidebar_statistics, refresh_sidebar_if_available, request_render_if_available,
};
use common::file_id::{cached_sha256, hex_digest};
use plugin_api::ffi::{HostLogLevelFFI, ViewportSnapshotFFI};
use serde::Serialize;
use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;

const GPU_BATCH_FILL_WAIT: Duration = Duration::from_millis(3);
const BACKGROUND_SAMPLE_STRIDE: usize = 8;

fn cpu_worker_layout(requested_threads: usize, total_tiles: usize) -> (usize, usize) {
    let requested_threads = requested_threads.max(1);
    let total_tiles = total_tiles.max(1);
    if requested_threads == 1 || total_tiles == 1 {
        return (0, requested_threads);
    }
    let loader_workers = requested_threads.saturating_sub(1).min(2).min(total_tiles);
    let session_threads = requested_threads.saturating_sub(loader_workers).max(1);
    (loader_workers, session_threads)
}

fn cpu_prefetch_capacity(
    requested_threads: usize,
    total_tiles: usize,
    loader_workers: usize,
) -> usize {
    if loader_workers == 0 {
        return 0;
    }
    requested_threads
        .max(loader_workers * 4)
        .clamp(8, 32)
        .min(total_tiles)
}

fn gpu_prefetch_capacity(total_tiles: usize, loader_workers: usize, batch_size: usize) -> usize {
    loader_workers
        .saturating_mul(4)
        .max(batch_size.saturating_mul(2))
        .clamp(8, 256)
        .min(total_tiles.max(1))
}

fn gpu_completed_batch_capacity(total_tiles: usize, batch_size: usize) -> usize {
    total_tiles.div_ceil(batch_size.max(1)).clamp(1, 4)
}

fn cpu_read_status_message(scheduled: usize, loaded: usize, done: usize, total: usize) -> String {
    format!(
        "Reading {scheduled}/{total} | queued {} | done {done}/{total}",
        loaded.saturating_sub(done)
    )
}

fn cpu_inference_status_message(
    done: usize,
    loaded: usize,
    total: usize,
    elapsed_secs: Option<u64>,
) -> String {
    let current_tile = if loaded > done {
        (done + 1).min(total)
    } else {
        done.min(total)
    };
    let queued = loaded.saturating_sub(done);
    let mut message =
        format!("Infer {current_tile}/{total} | done {done}/{total} | queued {queued}");
    if let Some(elapsed_secs) = elapsed_secs {
        message.push_str(&format!(" | {elapsed_secs}s"));
    }
    message
}

#[derive(Clone, Debug)]
pub struct AnalysisConfig {
    pub analysis_threads: usize,
    pub gpu_batch_size: usize,
    pub skip_background: bool,
    pub background_threshold: u8,
    pub mip_level: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_threads: crate::state::max_analysis_threads(),
            gpu_batch_size: crate::state::clamp_gpu_batch_size(64),
            skip_background: true,
            background_threshold: 242,
            mip_level: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalyzedTile {
    pub x: u64,
    pub y: u64,
    pub width: u32,
    pub height: u32,
    pub sample_width: u32,
    pub sample_height: u32,
    pub reconstruction_rgb: Vec<u8>,
    pub difference_rgb: Vec<u8>,
    pub error_map_luma: Vec<u8>,
    pub mean_absolute_error: f64,
    pub max_error: u8,
}

impl AnalyzedTile {
    pub fn id(&self) -> String {
        format!("{}:{}:{}:{}", self.x, self.y, self.width, self.height)
    }

    #[cfg(test)]
    pub fn dummy(x: u64, y: u64, width: u32, height: u32, mae: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            sample_width: width,
            sample_height: height,
            reconstruction_rgb: vec![0; width as usize * height as usize * 3],
            difference_rgb: vec![0; width as usize * height as usize * 3],
            error_map_luma: vec![0; width as usize * height as usize],
            mean_absolute_error: mae,
            max_error: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileCacheEntry {
    pub namespace: String,
    pub tile: AnalyzedTile,
}

#[derive(Clone, Copy, Debug)]
struct TilePlan {
    x: u64,
    y: u64,
    width: u32,
    height: u32,
    level: u32,
    read_width: u32,
    read_height: u32,
}

struct LoadedGpuTile {
    tile_plan: TilePlan,
    bytes: Vec<u8>,
}

struct CompletedGpuBatch {
    tile_plans: Vec<TilePlan>,
    rgba_tiles: Vec<Vec<u8>>,
    reconstructions: Vec<ReconstructionResult>,
}

struct LoadedCpuTile {
    tile_plan: TilePlan,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct TileGridBounds {
    left: u64,
    top: u64,
    right: u64,
    bottom: u64,
    image_width: u64,
    image_height: u64,
}

fn debug_timing(message: &str) {
    if env::var_os("EOVAE_DEBUG_TIMING").is_some() {
        eprintln!("[eovae] {message}");
    }
}

#[derive(Clone, Debug)]
pub struct HotRegion {
    pub id: String,
    pub x: u64,
    pub y: u64,
    pub width: u32,
    pub height: u32,
    pub mean_absolute_error: f64,
}

pub fn start_viewport_analysis(
    model: LoadedModel,
    viewport: ViewportSnapshotFFI,
    namespace: String,
    mip_level: u32,
) {
    let tile_size = model.summary.tile_size;
    start_job(JobKind::Viewport, namespace.clone(), move |cancel| {
        let tiles = viewport_tiles(&viewport, tile_size, mip_level);
        run_tile_plan(model, namespace, tiles, cancel)
    });
}

pub fn start_visible_tile_analysis(
    model: LoadedModel,
    positions: Vec<(u64, u64)>,
    namespace: String,
    mip_level: u32,
) {
    if positions.is_empty() {
        return;
    }

    let tile_size = model.summary.tile_size;
    let downsample = 1u64 << mip_level.min(3);
    let world_size = (tile_size as u64 * downsample) as u32;
    start_job(JobKind::Viewport, namespace.clone(), move |cancel| {
        let tiles = positions
            .iter()
            .copied()
            .map(|(x, y)| TilePlan {
                x,
                y,
                width: world_size,
                height: world_size,
                level: mip_level,
                read_width: tile_size,
                read_height: tile_size,
            })
            .collect::<Vec<_>>();
        run_tile_plan(model, namespace, tiles, cancel)
    });
}

pub fn start_whole_slide_analysis(
    model: LoadedModel,
    file_id: i32,
    file_path: String,
    image_width: u64,
    image_height: u64,
    namespace: String,
    mip_level: u32,
) {
    let tile_size = model.summary.tile_size;
    start_job(JobKind::WholeSlide, namespace.clone(), move |cancel| {
        let tiles = full_slide_tiles(image_width, image_height, tile_size, mip_level);
        run_tile_plan_with_file(model, namespace, tiles, cancel, file_id, file_path)
    });
}

fn start_job<F>(kind: JobKind, namespace: String, worker: F)
where
    F: FnOnce(Arc<AtomicBool>) -> Result<(), String> + Send + 'static,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let analysis_run_generation = {
        let mut state = plugin_state().lock().unwrap();
        state.analysis_run_generation = state.analysis_run_generation.wrapping_add(1);
        state.analysis_run_generation
    };
    {
        let mut state = plugin_state().lock().unwrap();
        state.job = Some(RunningJob {
            cancel: cancel.clone(),
        });
        state.analysis_phase = AnalysisPhase::Running;
        state.analysis_started_at = Some(Instant::now());
        state.analysis_started_wallclock = Some(SystemTime::now());
        state.analysis_elapsed = None;
        state.analysis_error_message = None;
        state.progress_value = 0.0;
        state.job_status = match kind {
            JobKind::Viewport => "Preparing viewport analysis".to_string(),
            JobKind::WholeSlide => "Preparing whole-slide analysis".to_string(),
        };
        state.cache_namespace = namespace;
    }
    refresh_sidebar_if_available();
    start_analysis_elapsed_refresh_loop(analysis_run_generation);

    thread::spawn(move || {
        let result = worker(cancel.clone());
        let mut state = plugin_state().lock().unwrap();
        let elapsed = state
            .analysis_started_at
            .map(|started_at| started_at.elapsed());
        state.job = None;
        state.analysis_started_at = None;
        state.analysis_elapsed = elapsed;
        state.progress_value = if cancel.load(Ordering::Relaxed) {
            0.0
        } else {
            1.0
        };
        state.job_status = if cancel.load(Ordering::Relaxed) {
            state.analysis_phase = AnalysisPhase::Cancelled;
            state.analysis_error_message = None;
            "Analysis cancelled".to_string()
        } else if let Err(error) = &result {
            state.analysis_phase = AnalysisPhase::Error;
            state.analysis_error_message = Some(error.clone());
            error.clone()
        } else {
            state.analysis_phase = AnalysisPhase::Completed;
            state.analysis_error_message = None;
            "Analysis complete".to_string()
        };
        rebuild_sidebar_statistics(&mut state);
        drop(state);
        if let Err(error) = result {
            if cancel.load(Ordering::Relaxed) {
                refresh_sidebar_if_available();
                request_render_if_available();
                return;
            }
            log_message(HostLogLevelFFI::Error, error);
        }
        refresh_sidebar_if_available();
        request_render_if_available();
    });
}

fn start_analysis_elapsed_refresh_loop(analysis_run_generation: u64) {
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(1));
            let should_refresh = {
                let state = plugin_state().lock().unwrap();
                state.analysis_run_generation == analysis_run_generation
                    && state.analysis_phase == AnalysisPhase::Running
            };
            if !should_refresh {
                return;
            }
            refresh_sidebar_if_available();
        }
    });
}

fn run_tile_plan(
    model: LoadedModel,
    namespace: String,
    tiles: Vec<TilePlan>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    let snapshot = host_api()
        .ok_or_else(|| "host API is not available".to_string())?
        .get_snapshot;
    let host = host_api().ok_or_else(|| "host API is not available".to_string())?;
    let current = snapshot(host.context);
    let active_file = current
        .active_file
        .into_option()
        .ok_or_else(|| "no active slide is open".to_string())?;
    run_tile_plan_with_file(
        model,
        namespace,
        tiles,
        cancel,
        active_file.file_id,
        active_file.path.to_string(),
    )
}

fn run_tile_plan_with_file(
    model: LoadedModel,
    namespace: String,
    tiles: Vec<TilePlan>,
    cancel: Arc<AtomicBool>,
    file_id: i32,
    file_path: String,
) -> Result<(), String> {
    let host = host_api().ok_or_else(|| "host API is not available".to_string())?;
    let (skip_background, background_threshold, analysis_threads, gpu_batch_size, prefer_gpu) = {
        let state = plugin_state().lock().unwrap();
        let prefer_gpu = host_api()
            .map(|host| (host.get_snapshot)(host.context))
            .map(|snapshot| snapshot.render_backend.to_ascii_lowercase() == "gpu")
            .unwrap_or(false);
        (
            state.config.skip_background,
            state.config.background_threshold,
            state.config.analysis_threads.max(1),
            state.config.gpu_batch_size.max(1),
            prefer_gpu,
        )
    };

    if tiles.is_empty() {
        return Ok(());
    }

    let total_tiles = tiles.len();
    let (cache_key, cached_tiles, tiles) =
        prepare_cached_tiles(&model, &namespace, &file_path, tiles);
    let cached_count = cached_tiles.len();

    if !cached_tiles.is_empty() {
        report_progress_message(
            cached_count,
            total_tiles,
            format!("Cache {cached_count}/{total_tiles} ready"),
            true,
            true,
        );
    }

    if tiles.is_empty() {
        return Ok(());
    }

    if prefer_gpu {
        return run_tile_plan_with_file_gpu_batched(
            model,
            namespace,
            tiles.as_slice(),
            cancel,
            file_id,
            host,
            skip_background,
            background_threshold,
            analysis_threads,
            gpu_batch_size,
            total_tiles,
            cached_count,
            cache_key,
        );
    }

    let work_tiles = tiles.len();
    let (loader_workers, worker_session_threads) = cpu_worker_layout(analysis_threads, work_tiles);
    let tiles = Arc::new(tiles);
    let next_index = Arc::new(AtomicUsize::new(0));
    let processed = Arc::new(AtomicUsize::new(cached_count));
    let loaded = Arc::new(AtomicUsize::new(cached_count));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let mut persistent_cache = cache_key.clone().and_then(|key| {
        AsyncLatentCacheWriter::open(key)
            .map_err(log_persistent_cache_error)
            .ok()
    });

    if loader_workers == 0 {
        let worker_model = model.clone_for_analysis_worker(worker_session_threads);
        for tile_plan in tiles.iter().copied() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let scheduled = processed.load(Ordering::Relaxed) + 1;
            let done = processed.load(Ordering::Relaxed);
            report_progress_message(
                done,
                total_tiles,
                cpu_read_status_message(scheduled, done, done, total_tiles),
                false,
                true,
            );
            let bytes = (host.read_region)(
                host.context,
                file_id,
                tile_plan.level,
                tile_plan.x as i64,
                tile_plan.y as i64,
                tile_plan.read_width,
                tile_plan.read_height,
            )
            .into_result()
            .map_err(|error| error.to_string())?;
            let loaded_count = loaded.fetch_add(1, Ordering::Relaxed) + 1;
            report_progress_message(
                done,
                total_tiles,
                cpu_inference_status_message(done, loaded_count, total_tiles, None),
                false,
                true,
            );
            let _status_pulse = BlockingStatusPulse::start(
                total_tiles,
                Arc::clone(&processed),
                Arc::clone(&loaded),
            );
            let reconstruction = run_reconstruction(
                &worker_model,
                bytes.as_slice(),
                tile_plan.read_width,
                tile_plan.read_height,
            )?;
            let tile = build_analyzed_tile(tile_plan, bytes.as_slice(), &reconstruction.rgb);
            if let Some(cache) = persistent_cache.as_ref()
                && let Err(error) =
                    cache.enqueue_tile(tile.clone(), reconstruction.embedding.clone())
            {
                log_persistent_cache_error(error);
                persistent_cache = None;
            }
            {
                let mut state = plugin_state().lock().unwrap();
                state.cache.insert(
                    tile.id(),
                    TileCacheEntry {
                        namespace: namespace.clone(),
                        tile,
                    },
                );
            }

            let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
            let refresh_stats = done == total_tiles || done.is_multiple_of(16);
            report_progress(done, total_tiles, "Processing", refresh_stats);
        }
    } else {
        let prefetch_capacity = cpu_prefetch_capacity(analysis_threads, work_tiles, loader_workers);
        let (sender, receiver) = mpsc::sync_channel::<LoadedCpuTile>(prefetch_capacity);
        let mut handles = Vec::with_capacity(loader_workers);

        for _ in 0..loader_workers {
            let worker_tiles = Arc::clone(&tiles);
            let worker_next_index = Arc::clone(&next_index);
            let worker_processed = Arc::clone(&processed);
            let worker_loaded = Arc::clone(&loaded);
            let worker_cancel = Arc::clone(&cancel);
            let worker_error = Arc::clone(&first_error);
            let worker_host = host;
            let worker_sender = sender.clone();

            handles.push(thread::spawn(move || {
                loop {
                    if worker_cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    if worker_error.lock().unwrap().is_some() {
                        return;
                    }

                    let index = worker_next_index.fetch_add(1, Ordering::Relaxed);
                    if index >= worker_tiles.len() {
                        return;
                    }

                    let tile_plan = worker_tiles[index];
                    let result = (|| -> Result<(), String> {
                        let scheduled = worker_next_index.load(Ordering::Relaxed).min(total_tiles);
                        let done = worker_processed.load(Ordering::Relaxed);
                        let loaded_count = worker_loaded.load(Ordering::Relaxed);
                        report_progress_message(
                            done,
                            total_tiles,
                            cpu_read_status_message(scheduled, loaded_count, done, total_tiles),
                            false,
                            true,
                        );
                        let bytes = (worker_host.read_region)(
                            worker_host.context,
                            file_id,
                            tile_plan.level,
                            tile_plan.x as i64,
                            tile_plan.y as i64,
                            tile_plan.read_width,
                            tile_plan.read_height,
                        )
                        .into_result()
                        .map_err(|error| error.to_string())?;
                        worker_sender
                            .send(LoadedCpuTile {
                                tile_plan,
                                bytes: bytes.to_vec(),
                            })
                            .map_err(|_| "cpu tile receiver disconnected".to_string())?;
                        let loaded_count = worker_loaded.fetch_add(1, Ordering::Relaxed) + 1;
                        report_progress_message(
                            done,
                            total_tiles,
                            cpu_read_status_message(scheduled, loaded_count, done, total_tiles),
                            false,
                            true,
                        );
                        Ok(())
                    })();

                    if let Err(error) = result {
                        worker_cancel.store(true, Ordering::Relaxed);
                        let mut slot = worker_error.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(error);
                        }
                        return;
                    }
                }
            }));
        }

        drop(sender);

        let worker_model = model.clone_for_analysis_worker(worker_session_threads);
        while !cancel.load(Ordering::Relaxed) {
            let loaded_tile = match receiver.recv() {
                Ok(loaded_tile) => loaded_tile,
                Err(_) => break,
            };
            let done = processed.load(Ordering::Relaxed);
            let loaded_count = loaded.load(Ordering::Relaxed);
            report_progress_message(
                done,
                total_tiles,
                cpu_inference_status_message(done, loaded_count, total_tiles, None),
                false,
                true,
            );
            let _status_pulse = BlockingStatusPulse::start(
                total_tiles,
                Arc::clone(&processed),
                Arc::clone(&loaded),
            );
            let reconstruction = run_reconstruction(
                &worker_model,
                loaded_tile.bytes.as_slice(),
                loaded_tile.tile_plan.read_width,
                loaded_tile.tile_plan.read_height,
            )?;
            let tile = build_analyzed_tile(
                loaded_tile.tile_plan,
                loaded_tile.bytes.as_slice(),
                &reconstruction.rgb,
            );
            if let Some(cache) = persistent_cache.as_ref()
                && let Err(error) =
                    cache.enqueue_tile(tile.clone(), reconstruction.embedding.clone())
            {
                log_persistent_cache_error(error);
                persistent_cache = None;
            }
            {
                let mut state = plugin_state().lock().unwrap();
                state.cache.insert(
                    tile.id(),
                    TileCacheEntry {
                        namespace: namespace.clone(),
                        tile,
                    },
                );
            }

            let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
            let refresh_stats = done == total_tiles || done.is_multiple_of(16);
            report_progress(done, total_tiles, "Processing", refresh_stats);
        }

        drop(receiver);

        for handle in handles {
            handle
                .join()
                .map_err(|_| "analysis worker thread panicked".to_string())?;
        }

        if let Some(error) = first_error.lock().unwrap().clone() {
            return Err(error);
        }
    }

    if let Some(cache) = persistent_cache.take() {
        cache.finish().map_err(log_persistent_cache_error)?;
    }

    {
        let mut state = plugin_state().lock().unwrap();
        rebuild_sidebar_statistics(&mut state);
    }
    refresh_sidebar_if_available();
    request_render_if_available();
    Ok(())
}

fn report_progress(done: usize, total: usize, label: &str, refresh_stats: bool) {
    report_progress_message(
        done,
        total,
        format!("{label} {done}/{total} tiles"),
        refresh_stats,
        false,
    );
}

fn report_progress_message(
    done: usize,
    total: usize,
    message: String,
    refresh_stats: bool,
    force_refresh: bool,
) {
    if force_refresh {
        debug_timing(&format!("progress force refresh: {message}"));
    }
    let mut state = plugin_state().lock().unwrap();
    state.progress_value = done as f32 / total.max(1) as f32;
    state.job_status = message;
    if refresh_stats {
        rebuild_sidebar_statistics(&mut state);
    }
    drop(state);
    if force_refresh || done == total || done.is_multiple_of(8) {
        refresh_sidebar_if_available();
        request_render_if_available();
    }
}

struct BlockingStatusPulse {
    done_tx: mpsc::Sender<()>,
    handle: Option<thread::JoinHandle<()>>,
}

impl BlockingStatusPulse {
    fn start(total: usize, done: Arc<AtomicUsize>, loaded: Arc<AtomicUsize>) -> Self {
        let (done_tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            loop {
                match done_rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let done_count = done.load(Ordering::Relaxed);
                        let loaded_count = loaded.load(Ordering::Relaxed);
                        if loaded_count <= done_count {
                            return;
                        }
                        report_progress_message(
                            done_count,
                            total,
                            cpu_inference_status_message(
                                done_count,
                                loaded_count,
                                total,
                                Some(started.elapsed().as_secs()),
                            ),
                            false,
                            true,
                        );
                    }
                }
            }
        });
        Self {
            done_tx,
            handle: Some(handle),
        }
    }
}

impl Drop for BlockingStatusPulse {
    fn drop(&mut self) {
        let _ = self.done_tx.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct InferenceRunningGuard {
    flag: Arc<AtomicBool>,
}

impl InferenceRunningGuard {
    fn new(flag: Arc<AtomicBool>) -> Self {
        flag.store(true, Ordering::Relaxed);
        Self { flag }
    }
}

impl Drop for InferenceRunningGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct GpuPipelineStats {
    read_ns: AtomicU64,
    read_tiles: AtomicUsize,
    batch_fill_wait_ns: AtomicU64,
    inference_ns: AtomicU64,
    batches: AtomicUsize,
    inferred_tiles: AtomicUsize,
}

struct GpuPipelineSnapshot {
    read_ns: u64,
    read_tiles: usize,
    batch_fill_wait_ns: u64,
    inference_ns: u64,
    batches: usize,
    inferred_tiles: usize,
}

impl GpuPipelineStats {
    fn record_read(&self, duration: Duration) {
        self.read_ns.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.read_tiles.fetch_add(1, Ordering::Relaxed);
    }

    fn record_batch_fill_wait(&self, duration: Duration) {
        self.batch_fill_wait_ns.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    fn record_inference(&self, duration: Duration, batch_tiles: usize) {
        self.inference_ns.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.inferred_tiles
            .fetch_add(batch_tiles, Ordering::Relaxed);
    }

    fn snapshot(&self) -> GpuPipelineSnapshot {
        GpuPipelineSnapshot {
            read_ns: self.read_ns.load(Ordering::Relaxed),
            read_tiles: self.read_tiles.load(Ordering::Relaxed),
            batch_fill_wait_ns: self.batch_fill_wait_ns.load(Ordering::Relaxed),
            inference_ns: self.inference_ns.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            inferred_tiles: self.inferred_tiles.load(Ordering::Relaxed),
        }
    }
}

fn format_duration_from_ns(total_ns: u64) -> Duration {
    Duration::from_nanos(total_ns)
}

fn format_avg_ms(total_ns: u64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total_ns as f64 / count as f64 / 1_000_000.0
    }
}

#[allow(clippy::too_many_arguments)]
fn run_tile_plan_with_file_gpu_batched(
    model: LoadedModel,
    namespace: String,
    tiles: &[TilePlan],
    cancel: Arc<AtomicBool>,
    file_id: i32,
    host: plugin_api::ffi::HostApiVTable,
    skip_background: bool,
    background_threshold: u8,
    analysis_threads: usize,
    gpu_batch_size: usize,
    total_tiles: usize,
    cached_count: usize,
    cache_key: Option<LatentCacheKey>,
) -> Result<(), String> {
    let work_tiles = tiles.len();
    let worker_count = analysis_threads.min(work_tiles).max(1);
    let model_fixed_batch_size = model.fixed_batch_size();
    let requested_batch_size = crate::state::clamp_gpu_batch_size(gpu_batch_size);
    let batch_size = model_fixed_batch_size
        .unwrap_or(requested_batch_size)
        .min(work_tiles)
        .max(1);
    let queue_capacity = gpu_prefetch_capacity(work_tiles, worker_count, batch_size);
    let completed_queue_capacity = gpu_completed_batch_capacity(work_tiles, batch_size);
    debug_timing(&format!(
        "starting gpu batched analysis tiles={} loader_workers={} configured_batch_size={} model_fixed_batch_size={} batch_size={} queue_capacity={} completed_queue_capacity={}",
        total_tiles,
        worker_count,
        gpu_batch_size,
        model_fixed_batch_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "dynamic".to_string()),
        batch_size,
        queue_capacity,
        completed_queue_capacity
    ));
    let tiles = Arc::new(tiles.to_vec());
    let next_index = Arc::new(AtomicUsize::new(0));
    let processed = Arc::new(AtomicUsize::new(cached_count));
    let scanned = Arc::new(AtomicUsize::new(cached_count));
    let filtered = Arc::new(AtomicUsize::new(0));
    let queued = Arc::new(AtomicUsize::new(0));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let pipeline_stats = Arc::new(GpuPipelineStats::default());
    let (sender, receiver) = mpsc::sync_channel::<LoadedGpuTile>(queue_capacity);
    let (completed_sender, completed_receiver) =
        mpsc::sync_channel::<CompletedGpuBatch>(completed_queue_capacity);
    let mut handles = Vec::with_capacity(worker_count);
    let inference_running = Arc::new(AtomicBool::new(false));
    let gpu_started = Instant::now();

    let postprocess_cancel = Arc::clone(&cancel);
    let postprocess_processed = Arc::clone(&processed);
    let postprocess_namespace = namespace.clone();
    let postprocess_cache_key = cache_key.clone();
    let postprocess_handle = thread::spawn(move || -> Result<(), String> {
        let mut persistent_cache = postprocess_cache_key.and_then(|key| {
            AsyncLatentCacheWriter::open(key)
                .map_err(log_persistent_cache_error)
                .ok()
        });
        while let Ok(completed_batch) = completed_receiver.recv() {
            if postprocess_cancel.load(Ordering::Relaxed) {
                return Ok(());
            }

            let analysis_postprocess_started = Instant::now();
            let CompletedGpuBatch {
                tile_plans,
                rgba_tiles,
                reconstructions,
            } = completed_batch;
            let embeddings = reconstructions
                .iter()
                .map(|reconstruction| reconstruction.embedding.clone())
                .collect::<Vec<_>>();
            let analyzed_tiles =
                build_analyzed_tiles_batch(tile_plans, rgba_tiles, reconstructions);
            debug_timing(&format!(
                "batch analyzed-tile construction completed in {:?}",
                analysis_postprocess_started.elapsed()
            ));

            if let Some(cache) = persistent_cache.as_ref() {
                for (tile, embedding) in analyzed_tiles.iter().zip(embeddings.into_iter()) {
                    if let Err(error) = cache.enqueue_tile(tile.clone(), embedding) {
                        log_persistent_cache_error(error);
                        persistent_cache = None;
                        break;
                    }
                }
            }

            {
                let mut state = plugin_state()
                    .lock()
                    .map_err(|_| "plugin state lock is poisoned".to_string())?;
                for tile in analyzed_tiles.iter() {
                    state.cache.insert(
                        tile.id(),
                        TileCacheEntry {
                            namespace: postprocess_namespace.clone(),
                            tile: tile.clone(),
                        },
                    );
                }
            }

            let completed_batch_size = analyzed_tiles.len();
            let previous_done =
                postprocess_processed.fetch_add(completed_batch_size, Ordering::Relaxed);
            let done = previous_done + completed_batch_size;
            let refresh_stats = done == total_tiles || done / 16 != previous_done / 16;
            report_progress(done, total_tiles, "Processing", refresh_stats);
        }
        if let Some(cache) = persistent_cache.take() {
            cache.finish().map_err(log_persistent_cache_error)?;
        }
        Ok(())
    });

    for _ in 0..worker_count {
        let worker_tiles = Arc::clone(&tiles);
        let worker_next_index = Arc::clone(&next_index);
        let worker_processed = Arc::clone(&processed);
        let worker_scanned = Arc::clone(&scanned);
        let worker_filtered = Arc::clone(&filtered);
        let worker_queued = Arc::clone(&queued);
        let worker_inference_running = Arc::clone(&inference_running);
        let worker_cancel = Arc::clone(&cancel);
        let worker_error = Arc::clone(&first_error);
        let worker_stats = Arc::clone(&pipeline_stats);
        let worker_host = host;
        let worker_sender = sender.clone();

        handles.push(thread::spawn(move || {
            loop {
                if worker_cancel.load(Ordering::Relaxed) {
                    return;
                }
                if worker_error.lock().unwrap().is_some() {
                    return;
                }

                let start = worker_next_index.fetch_add(batch_size, Ordering::Relaxed);
                if start >= worker_tiles.len() {
                    return;
                }

                let end = (start + batch_size).min(worker_tiles.len());
                let result = (|| -> Result<(), String> {
                    for index in start..end {
                        if worker_cancel.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        let tile_plan = worker_tiles[index];
                        let read_started = Instant::now();
                        let bytes = (worker_host.read_region)(
                            worker_host.context,
                            file_id,
                            tile_plan.level,
                            tile_plan.x as i64,
                            tile_plan.y as i64,
                            tile_plan.read_width,
                            tile_plan.read_height,
                        )
                        .into_result()
                        .map_err(|error| error.to_string())?;
                        worker_stats.record_read(read_started.elapsed());
                        let scanned_count = worker_scanned.fetch_add(1, Ordering::Relaxed) + 1;

                        if skip_background
                            && should_skip_background(bytes.as_slice(), background_threshold)
                        {
                            let filtered_count = worker_filtered.fetch_add(1, Ordering::Relaxed) + 1;
                            let done = worker_processed.fetch_add(1, Ordering::Relaxed) + 1;
                            if !worker_inference_running.load(Ordering::Relaxed)
                                && (done == total_tiles || done.is_multiple_of(8))
                            {
                                report_progress_message(
                                    done,
                                    total_tiles,
                                    format!(
                                        "Filter {scanned_count}/{total_tiles} | skipped {filtered_count} | queued {}",
                                        worker_queued.load(Ordering::Relaxed)
                                    ),
                                    false,
                                    false,
                                );
                            }
                            continue;
                        }

                        worker_sender
                            .send(LoadedGpuTile {
                                tile_plan,
                                bytes: bytes.to_vec(),
                            })
                            .map_err(|_| "gpu batch receiver disconnected".to_string())?;
                        let queued_count = worker_queued.fetch_add(1, Ordering::Relaxed) + 1;
                        if !worker_inference_running.load(Ordering::Relaxed)
                            && (scanned_count == total_tiles
                                || scanned_count.is_multiple_of(8))
                        {
                            report_progress_message(
                                worker_processed.load(Ordering::Relaxed),
                                total_tiles,
                                format!(
                                    "Queue {scanned_count}/{total_tiles} | skipped {} | ready {queued_count}",
                                    worker_filtered.load(Ordering::Relaxed)
                                ),
                                false,
                                false,
                            );
                        }
                    }

                    Ok(())
                })();

                if let Err(error) = result {
                    worker_cancel.store(true, Ordering::Relaxed);
                    let mut slot = worker_error.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(error);
                    }
                    return;
                }
            }
        }));
    }

    drop(sender);

    let worker_model = model.clone_for_analysis_worker(analysis_threads);
    let mut pending_tiles = Vec::with_capacity(batch_size);
    let mut pending_bytes = Vec::with_capacity(batch_size);
    let mut sender_disconnected = false;
    let require_full_batch = model_fixed_batch_size.is_some() && total_tiles >= batch_size;
    let mut inference_error = None::<String>;

    while !cancel.load(Ordering::Relaxed) {
        let loaded = match receiver.recv() {
            Ok(loaded) => loaded,
            Err(_) => break,
        };
        pending_tiles.push(loaded.tile_plan);
        pending_bytes.push(loaded.bytes);
        let batch_fill_started = Instant::now();

        while pending_tiles.len() < batch_size {
            let recv_result = if require_full_batch {
                receiver
                    .recv()
                    .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
            } else {
                receiver.recv_timeout(GPU_BATCH_FILL_WAIT)
            };
            match recv_result {
                Ok(loaded) => {
                    pending_tiles.push(loaded.tile_plan);
                    pending_bytes.push(loaded.bytes);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    sender_disconnected = true;
                    break;
                }
            }
        }
        pipeline_stats.record_batch_fill_wait(batch_fill_started.elapsed());

        let target_width = pending_tiles[0].read_width;
        let target_height = pending_tiles[0].read_height;
        let batch_inputs = pending_bytes
            .iter()
            .map(|bytes| BatchReconstructionInput {
                rgba: bytes.as_slice(),
                width: target_width,
                height: target_height,
            })
            .collect::<Vec<_>>();
        let partial_reason = if batch_inputs.len() == batch_size {
            None
        } else if sender_disconnected {
            Some("channel drained")
        } else if require_full_batch {
            Some("awaiting static batch fill")
        } else {
            Some("fill wait expired")
        };
        let scanned_count = scanned.load(Ordering::Relaxed);
        let filtered_count = filtered.load(Ordering::Relaxed);
        debug_timing(&format!(
            "dispatching gpu inference batch actual_batch_size={} configured_batch_size={} tile={}x{} reason={}",
            batch_inputs.len(),
            batch_size,
            target_width,
            target_height,
            partial_reason.unwrap_or("full")
        ));
        report_progress_message(
            processed.load(Ordering::Relaxed),
            total_tiles,
            format!(
                "Batch {} | scanned {scanned_count}/{total_tiles} | skipped {filtered_count}",
                batch_inputs.len()
            ),
            false,
            true,
        );
        let _inference_running = InferenceRunningGuard::new(Arc::clone(&inference_running));
        let inference_started = Instant::now();
        let reconstructions = match run_reconstruction_batch(&worker_model, batch_inputs.as_slice())
        {
            Ok(reconstructions) => reconstructions,
            Err(error) => {
                cancel.store(true, Ordering::Relaxed);
                inference_error = Some(error);
                break;
            }
        };
        pipeline_stats.record_inference(inference_started.elapsed(), batch_inputs.len());

        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let completed_batch = CompletedGpuBatch {
            tile_plans: std::mem::take(&mut pending_tiles),
            rgba_tiles: std::mem::take(&mut pending_bytes),
            reconstructions,
        };
        if completed_sender.send(completed_batch).is_err() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            inference_error = Some("gpu completed batch receiver disconnected".to_string());
            cancel.store(true, Ordering::Relaxed);
            break;
        }
    }

    drop(receiver);
    drop(completed_sender);

    for handle in handles {
        handle
            .join()
            .map_err(|_| "analysis worker thread panicked".to_string())?;
    }

    let postprocess_result = postprocess_handle
        .join()
        .map_err(|_| "gpu postprocess thread panicked".to_string())?;

    if let Some(error) = first_error.lock().unwrap().clone() {
        return Err(error);
    }
    if let Some(error) = inference_error {
        return Err(error);
    }
    postprocess_result?;

    let stats = pipeline_stats.snapshot();
    let elapsed = gpu_started.elapsed();
    debug_timing(&format!(
        "gpu pipeline summary elapsed={elapsed:?} batches={} inferred_tiles={} avg_batch_size={:.2} avg_read_ms_per_tile={:.2} avg_infer_ms_per_batch={:.2} total_fill_wait={:?} total_infer={:?} throughput_tiles_per_sec={:.2}",
        stats.batches,
        stats.inferred_tiles,
        if stats.batches == 0 {
            0.0
        } else {
            stats.inferred_tiles as f64 / stats.batches as f64
        },
        format_avg_ms(stats.read_ns, stats.read_tiles),
        format_avg_ms(stats.inference_ns, stats.batches),
        format_duration_from_ns(stats.batch_fill_wait_ns),
        format_duration_from_ns(stats.inference_ns),
        if elapsed.as_secs_f64() == 0.0 {
            0.0
        } else {
            stats.inferred_tiles as f64 / elapsed.as_secs_f64()
        }
    ));

    {
        let mut state = plugin_state().lock().unwrap();
        rebuild_sidebar_statistics(&mut state);
    }
    refresh_sidebar_if_available();
    request_render_if_available();
    Ok(())
}

fn viewport_tiles(viewport: &ViewportSnapshotFFI, tile_size: u32, mip_level: u32) -> Vec<TilePlan> {
    grid_tiles(
        TileGridBounds {
            left: viewport.bounds_left.max(0.0) as u64,
            top: viewport.bounds_top.max(0.0) as u64,
            right: viewport.bounds_right.max(0.0) as u64,
            bottom: viewport.bounds_bottom.max(0.0) as u64,
            image_width: viewport.image_width.max(0.0) as u64,
            image_height: viewport.image_height.max(0.0) as u64,
        },
        tile_size,
        mip_level,
    )
}

fn full_slide_tiles(
    image_width: u64,
    image_height: u64,
    tile_size: u32,
    mip_level: u32,
) -> Vec<TilePlan> {
    grid_tiles(
        TileGridBounds {
            left: 0,
            top: 0,
            right: image_width,
            bottom: image_height,
            image_width,
            image_height,
        },
        tile_size,
        mip_level,
    )
}

fn grid_tiles(bounds: TileGridBounds, tile_size: u32, mip_level: u32) -> Vec<TilePlan> {
    let mut tiles = Vec::new();
    let downsample = 1u64 << mip_level.min(3);
    let step = tile_size as u64 * downsample;
    let mut y = (bounds.top / step) * step;
    while y < bounds.bottom {
        let mut x = (bounds.left / step) * step;
        while x < bounds.right {
            if x + step <= bounds.image_width && y + step <= bounds.image_height {
                let world_width = step as u32;
                let world_height = step as u32;
                let read_width = tile_size;
                let read_height = tile_size;
                tiles.push(TilePlan {
                    x,
                    y,
                    width: world_width,
                    height: world_height,
                    level: mip_level,
                    read_width,
                    read_height,
                });
            }
            x += step;
        }
        y += step;
    }
    tiles
}

fn should_skip_background(rgba: &[u8], threshold: u8) -> bool {
    if rgba.is_empty() {
        return true;
    }
    let mut bright_pixels = 0usize;
    let mut total_pixels = 0usize;
    for pixel in rgba.chunks_exact(4).step_by(BACKGROUND_SAMPLE_STRIDE) {
        total_pixels += 1;
        if pixel[0] >= threshold && pixel[1] >= threshold && pixel[2] >= threshold {
            bright_pixels += 1;
        }
    }
    bright_pixels * 100 / total_pixels.max(1) > 92
}

fn build_analyzed_tile(
    tile_plan: TilePlan,
    rgba: &[u8],
    reconstruction_rgb: &[u8],
) -> AnalyzedTile {
    let sample_width = tile_plan.read_width;
    let sample_height = tile_plan.read_height;
    let mut difference_rgb = vec![0u8; sample_width as usize * sample_height as usize * 3];
    let mut error_map_luma = vec![0u8; sample_width as usize * sample_height as usize];
    let mut error_sum = 0f64;
    let mut max_error = 0u8;

    for (index, error_value) in error_map_luma.iter_mut().enumerate() {
        let src = index * 4;
        let dst = index * 3;
        let dr = rgba[src].abs_diff(reconstruction_rgb[dst]);
        let dg = rgba[src + 1].abs_diff(reconstruction_rgb[dst + 1]);
        let db = rgba[src + 2].abs_diff(reconstruction_rgb[dst + 2]);
        let mean = ((dr as u16 + dg as u16 + db as u16) / 3) as u8;
        difference_rgb[dst] = ((dr as u16 * dr as u16) / 255) as u8;
        difference_rgb[dst + 1] = ((dg as u16 * dg as u16) / 255) as u8;
        difference_rgb[dst + 2] = ((db as u16 * db as u16) / 255) as u8;
        *error_value = mean;
        error_sum += mean as f64 / 255.0;
        max_error = max_error.max(mean);
    }

    AnalyzedTile {
        x: tile_plan.x,
        y: tile_plan.y,
        width: tile_plan.width,
        height: tile_plan.height,
        sample_width,
        sample_height,
        reconstruction_rgb: reconstruction_rgb.to_vec(),
        difference_rgb,
        error_map_luma,
        mean_absolute_error: error_sum / (sample_width as f64 * sample_height as f64).max(1.0),
        max_error,
    }
}

fn build_analyzed_tiles_batch(
    pending_tiles: Vec<TilePlan>,
    pending_bytes: Vec<Vec<u8>>,
    reconstructions: Vec<ReconstructionResult>,
) -> Vec<AnalyzedTile> {
    let mut batch_items = pending_tiles
        .into_iter()
        .zip(pending_bytes)
        .zip(reconstructions)
        .map(|((tile_plan, bytes), reconstruction)| (tile_plan, bytes, reconstruction))
        .collect::<Vec<_>>();

    let parallelism = analyzed_tile_parallelism(batch_items.len());
    if parallelism <= 1 || batch_items.len() <= 1 {
        return batch_items
            .into_iter()
            .map(
                |(tile_plan, bytes, reconstruction): (TilePlan, Vec<u8>, ReconstructionResult)| {
                    build_analyzed_tile(tile_plan, bytes.as_slice(), &reconstruction.rgb)
                },
            )
            .collect();
    }

    let chunk_size = batch_items.len().div_ceil(parallelism);
    let mut analyzed_tiles = Vec::with_capacity(batch_items.len());
    thread::scope(|scope| {
        let mut handles = Vec::new();
        while !batch_items.is_empty() {
            let split_at = batch_items.len().saturating_sub(chunk_size);
            let chunk = batch_items.split_off(split_at);
            handles.push(scope.spawn(move || {
                chunk
                    .into_iter()
                    .map(
                        |(tile_plan, bytes, reconstruction): (
                            TilePlan,
                            Vec<u8>,
                            ReconstructionResult,
                        )| {
                            build_analyzed_tile(tile_plan, bytes.as_slice(), &reconstruction.rgb)
                        },
                    )
                    .collect::<Vec<_>>()
            }));
        }

        for handle in handles {
            let chunk_tiles: Vec<AnalyzedTile> =
                handle.join().expect("analyzed-tile worker thread panicked");
            analyzed_tiles.extend(chunk_tiles);
        }
    });
    analyzed_tiles
}

fn prepare_cached_tiles(
    model: &LoadedModel,
    namespace: &str,
    file_path: &str,
    tiles: Vec<TilePlan>,
) -> (Option<LatentCacheKey>, Vec<AnalyzedTile>, Vec<TilePlan>) {
    let Some(cache_key) = build_latent_cache_key(model, file_path, tiles.first()) else {
        return (None, Vec::new(), tiles);
    };

    let positions = tiles
        .iter()
        .map(|tile| (tile.x, tile.y))
        .collect::<Vec<_>>();
    let cached_tiles = match load_tiles_for_positions(&cache_key, &positions) {
        Ok(cached_tiles) => cached_tiles,
        Err(error) => {
            log_persistent_cache_error(error);
            return (None, Vec::new(), tiles);
        }
    };
    let (renderable_cached_tiles, incomplete_cached_tiles): (Vec<_>, Vec<_>) = cached_tiles
        .into_iter()
        .partition(|tile| {
            let rgb_len = tile.sample_width as usize * tile.sample_height as usize * 3;
            tile.reconstruction_rgb.len() >= rgb_len && tile.difference_rgb.len() >= rgb_len
        });
    let cached_positions = renderable_cached_tiles
        .iter()
        .map(|tile| (tile.x, tile.y))
        .collect::<HashSet<_>>();
    let missing_tiles = tiles
        .into_iter()
        .filter(|tile| {
            !cached_positions.contains(&(tile.x, tile.y))
                || incomplete_cached_tiles
                    .iter()
                    .any(|cached| cached.x == tile.x && cached.y == tile.y)
        })
        .collect::<Vec<_>>();

    if !renderable_cached_tiles.is_empty() {
        insert_tiles_into_state(namespace, &renderable_cached_tiles);
    }

    (Some(cache_key), renderable_cached_tiles, missing_tiles)
}

fn build_latent_cache_key(
    model: &LoadedModel,
    file_path: &str,
    first_tile: Option<&TilePlan>,
) -> Option<LatentCacheKey> {
    let first_tile = first_tile?;
    let model_sha256 = cached_sha256(Path::new(&model.summary.path))
        .map(|digest| hex_digest(&digest))
        .map_err(log_persistent_cache_error)
        .ok()?;
    let file_sha256 = cached_sha256(Path::new(file_path))
        .map(|digest| hex_digest(&digest))
        .map_err(log_persistent_cache_error)
        .ok()?;

    Some(LatentCacheKey {
        model_sha256,
        model_path: model.summary.path.clone(),
        file_sha256,
        file_path: file_path.to_string(),
        level: first_tile.level,
        tile_size: first_tile.read_width,
        stride: first_tile.width,
        embedding_dim: model.summary.embedding_dim(),
    })
}

fn insert_tiles_into_state(namespace: &str, tiles: &[AnalyzedTile]) {
    let mut state = plugin_state().lock().unwrap();
    for tile in tiles {
        state.cache.insert(
            tile.id(),
            TileCacheEntry {
                namespace: namespace.to_string(),
                tile: tile.clone(),
            },
        );
    }
    rebuild_sidebar_statistics(&mut state);
}

fn log_persistent_cache_error(error: impl ToString) -> String {
    let error = error.to_string();
    log_message(
        HostLogLevelFFI::Error,
        format!("persistent latent cache disabled for this run: {error}"),
    );
    error
}

fn analyzed_tile_parallelism(batch_size: usize) -> usize {
    let configured_threads = crate::state::clamp_analysis_threads(
        plugin_state().lock().unwrap().config.analysis_threads,
    );
    thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(batch_size.max(1))
        .min(configured_threads.max(1))
        .max(1)
}

pub fn should_render_overlay(mode: VisualizationMode) -> bool {
    mode != VisualizationMode::Original
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_worker_layout_scales_with_requested_threads() {
        assert_eq!(cpu_worker_layout(1, 100), (0, 1));
        assert_eq!(cpu_worker_layout(4, 100), (2, 2));
        assert_eq!(cpu_worker_layout(8, 100), (2, 6));
        assert_eq!(cpu_worker_layout(32, 100), (2, 30));
    }

    #[test]
    fn cpu_prefetch_capacity_scales_with_requested_threads() {
        assert_eq!(cpu_prefetch_capacity(1, 100, 0), 0);
        assert_eq!(cpu_prefetch_capacity(4, 100, 2), 8);
        assert_eq!(cpu_prefetch_capacity(8, 100, 2), 8);
        assert_eq!(cpu_prefetch_capacity(32, 100, 2), 32);
    }

    #[test]
    fn gpu_prefetch_capacity_scales_with_batch_size() {
        assert_eq!(gpu_prefetch_capacity(1, 1, 1), 1);
        assert_eq!(gpu_prefetch_capacity(64, 4, 8), 16);
        assert_eq!(gpu_prefetch_capacity(256, 8, 64), 128);
        assert_eq!(gpu_prefetch_capacity(1000, 32, 256), 256);
    }

    #[test]
    fn cpu_inference_status_message_uses_global_completion() {
        assert_eq!(
            cpu_inference_status_message(12, 3, 100, Some(5)),
            "Running inference tile 12/100, 12/100 completed, 0 queued (5s)"
        );
        assert_eq!(
            cpu_inference_status_message(0, 5, 100, Some(5)),
            "Running inference tile 1/100, 0/100 completed, 5 queued (5s)"
        );
    }

    #[test]
    fn sorts_hot_regions_by_error() {
        let mut state = crate::state::PluginState {
            cache_namespace: "ns".to_string(),
            ..Default::default()
        };
        state.cache.insert(
            "a".to_string(),
            TileCacheEntry {
                namespace: "ns".to_string(),
                tile: AnalyzedTile::dummy(0, 0, 64, 64, 0.2),
            },
        );
        state.cache.insert(
            "b".to_string(),
            TileCacheEntry {
                namespace: "ns".to_string(),
                tile: AnalyzedTile::dummy(64, 0, 64, 64, 0.8),
            },
        );
        crate::state::rebuild_sidebar_statistics(&mut state);
        assert_eq!(state.hot_regions.first().unwrap().x, 64);
    }

    #[test]
    fn grid_tiles_expand_world_coverage_for_selected_mip() {
        let tiles = grid_tiles(
            TileGridBounds {
                left: 0,
                top: 0,
                right: 700,
                bottom: 700,
                image_width: 700,
                image_height: 700,
            },
            256,
            1,
        );

        assert_eq!(tiles[0].width, 512);
        assert_eq!(tiles[0].height, 512);
        assert_eq!(tiles[0].read_width, 256);
        assert_eq!(tiles[0].read_height, 256);
        assert_eq!(tiles[0].level, 1);

        let edge = tiles.last().unwrap();
        assert_eq!(edge.x, 0);
        assert_eq!(edge.y, 0);
        assert_eq!(edge.width, 512);
        assert_eq!(edge.height, 512);
        assert_eq!(edge.read_width, 256);
        assert_eq!(edge.read_height, 256);
    }

    #[test]
    fn grid_tiles_skip_partial_right_and_bottom_edges() {
        let tiles = grid_tiles(
            TileGridBounds {
                left: 300,
                top: 300,
                right: 900,
                bottom: 900,
                image_width: 900,
                image_height: 900,
            },
            256,
            0,
        );

        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].x, 256);
        assert_eq!(tiles[0].y, 256);
        assert_eq!(tiles[1].x, 512);
        assert_eq!(tiles[1].y, 256);
        assert_eq!(tiles[2].x, 256);
        assert_eq!(tiles[2].y, 512);
        assert_eq!(tiles[3].x, 512);
        assert_eq!(tiles[3].y, 512);
    }

    #[test]
    fn background_filter_detects_bright_tiles_with_sampling() {
        let rgba = vec![255u8; 256 * 256 * 4];
        assert!(should_skip_background(&rgba, 242));
    }

    #[test]
    fn background_filter_keeps_dark_tiles_with_sampling() {
        let rgba = vec![0u8; 256 * 256 * 4];
        assert!(!should_skip_background(&rgba, 242));
    }
}
