use crate::model::{BatchReconstructionInput, LoadedModel, run_reconstruction, run_reconstruction_batch};
use crate::state::{
    AnalysisPhase, JobKind, RunningJob, VisualizationMode, host_api, log_message, plugin_state,
    rebuild_sidebar_statistics, refresh_sidebar_if_available, request_render_if_available,
};
use plugin_api::ffi::{HostLogLevelFFI, ViewportSnapshotFFI};
use serde::Serialize;
use std::env;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;

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
        let elapsed = state.analysis_started_at.map(|started_at| started_at.elapsed());
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
            log_message(HostLogLevelFFI::Error, error);
        }
        refresh_sidebar_if_available();
        request_render_if_available();
    });
}

fn start_analysis_elapsed_refresh_loop(analysis_run_generation: u64) {
    thread::spawn(move || loop {
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
    _file_path: String,
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
        );
    }

    let total_tiles = tiles.len();
    let worker_count = if prefer_gpu {
        analysis_threads.min(2).min(total_tiles).max(1)
    } else {
        analysis_threads.min(total_tiles).max(1)
    };
    let tiles = Arc::new(tiles);
    let next_index = Arc::new(AtomicUsize::new(0));
    let processed = Arc::new(AtomicUsize::new(0));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let mut handles = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let worker_model = model.clone_for_analysis_worker(1);
        let worker_tiles = Arc::clone(&tiles);
        let worker_next_index = Arc::clone(&next_index);
        let worker_processed = Arc::clone(&processed);
        let worker_cancel = Arc::clone(&cancel);
        let worker_error = Arc::clone(&first_error);
        let worker_namespace = namespace.clone();
        let worker_host = host;

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

                    if false && skip_background
                        && should_skip_background(bytes.as_slice(), background_threshold)
                    {
                        let done = worker_processed.fetch_add(1, Ordering::Relaxed) + 1;
                        report_progress(done, total_tiles, "Skipping background tile", false);
                        return Ok(());
                    }

                    let reconstruction = run_reconstruction(
                        &worker_model,
                        bytes.as_slice(),
                        tile_plan.read_width,
                        tile_plan.read_height,
                    )?;
                    let tile = build_analyzed_tile(tile_plan, bytes.as_slice(), &reconstruction.rgb);
                    {
                        let mut state = plugin_state().lock().unwrap();
                        state.cache.insert(
                            tile.id(),
                            TileCacheEntry {
                                namespace: worker_namespace.clone(),
                                tile,
                            },
                        );
                    }

                    let done = worker_processed.fetch_add(1, Ordering::Relaxed) + 1;
                    let refresh_stats = done == total_tiles || done % 16 == 0;
                    report_progress(done, total_tiles, "Processing", refresh_stats);
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

    for handle in handles {
        handle
            .join()
            .map_err(|_| "analysis worker thread panicked".to_string())?;
    }

    if let Some(error) = first_error.lock().unwrap().clone() {
        return Err(error);
    }

    {
        let mut state = plugin_state().lock().unwrap();
        rebuild_sidebar_statistics(&mut state);
    }
    refresh_sidebar_if_available();
    request_render_if_available();
    return Ok(());
}

fn report_progress(done: usize, total: usize, label: &str, refresh_stats: bool) {
    let mut state = plugin_state().lock().unwrap();
    state.progress_value = done as f32 / total.max(1) as f32;
    state.job_status = format!("{label} {done}/{total} tiles");
    if refresh_stats {
        rebuild_sidebar_statistics(&mut state);
    }
    drop(state);
    if done == total || done % 8 == 0 {
        refresh_sidebar_if_available();
        request_render_if_available();
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
) -> Result<(), String> {
    let total_tiles = tiles.len();
    let worker_count = analysis_threads.min(total_tiles).max(1);
    let batch_size = crate::state::clamp_gpu_batch_size(gpu_batch_size)
        .min(total_tiles)
        .max(1);
    debug_timing(&format!(
        "starting gpu batched analysis tiles={} loader_workers={} configured_batch_size={} batch_size={} queue_capacity={}",
        total_tiles,
        worker_count,
        gpu_batch_size,
        batch_size,
        worker_count * 2
    ));
    let tiles = Arc::new(tiles.to_vec());
    let next_index = Arc::new(AtomicUsize::new(0));
    let processed = Arc::new(AtomicUsize::new(0));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let (sender, receiver) = mpsc::sync_channel::<LoadedGpuTile>(worker_count * 2);
    let mut handles = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let worker_tiles = Arc::clone(&tiles);
        let worker_next_index = Arc::clone(&next_index);
        let worker_processed = Arc::clone(&processed);
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

                        if skip_background
                            && should_skip_background(bytes.as_slice(), background_threshold)
                        {
                            let done = worker_processed.fetch_add(1, Ordering::Relaxed) + 1;
                            report_progress(done, total_tiles, "Skipping background tile", false);
                            continue;
                        }

                        worker_sender
                            .send(LoadedGpuTile {
                                tile_plan,
                                bytes: bytes.to_vec(),
                            })
                            .map_err(|_| "gpu batch receiver disconnected".to_string())?;
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

    while !cancel.load(Ordering::Relaxed) {
        let loaded = match receiver.recv() {
            Ok(loaded) => loaded,
            Err(_) => break,
        };
        pending_tiles.push(loaded.tile_plan);
        pending_bytes.push(loaded.bytes);

        while pending_tiles.len() < batch_size {
            match receiver.try_recv() {
                Ok(loaded) => {
                    pending_tiles.push(loaded.tile_plan);
                    pending_bytes.push(loaded.bytes);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

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
        debug_timing(&format!(
            "dispatching gpu inference batch actual_batch_size={} configured_batch_size={} tile={}x{}",
            batch_inputs.len(),
            batch_size,
            target_width,
            target_height
        ));
        let reconstructions = run_reconstruction_batch(&worker_model, batch_inputs.as_slice())?;

        for ((tile_plan, bytes), reconstruction) in pending_tiles
            .drain(..)
            .zip(pending_bytes.drain(..))
            .zip(reconstructions.into_iter())
        {
            let tile = build_analyzed_tile(tile_plan, bytes.as_slice(), &reconstruction.rgb);
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
            let refresh_stats = done == total_tiles || done % 16 == 0;
            report_progress(done, total_tiles, "Processing", refresh_stats);
        }
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| "analysis worker thread panicked".to_string())?;
    }

    if let Some(error) = first_error.lock().unwrap().clone() {
        return Err(error);
    }

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
    for pixel in rgba.chunks_exact(4) {
        total_pixels += 1;
        if pixel[0] >= threshold && pixel[1] >= threshold && pixel[2] >= threshold {
            bright_pixels += 1;
        }
    }
    bright_pixels * 100 / total_pixels.max(1) > 92
}

fn build_analyzed_tile(tile_plan: TilePlan, rgba: &[u8], reconstruction_rgb: &[u8]) -> AnalyzedTile {
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
        difference_rgb[dst] = dr;
        difference_rgb[dst + 1] = dg;
        difference_rgb[dst + 2] = db;
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

pub fn should_render_overlay(mode: VisualizationMode) -> bool {
    mode != VisualizationMode::Original
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
