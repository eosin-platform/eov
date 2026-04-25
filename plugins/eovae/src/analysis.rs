use crate::model::{LoadedModel, run_reconstruction};
use crate::state::{
    JobKind, RunningJob, VisualizationMode, host_api, log_message, plugin_state,
    rebuild_sidebar_statistics, refresh_sidebar_if_available, request_render_if_available,
};
use plugin_api::ffi::{HostLogLevelFFI, ViewportSnapshotFFI};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

#[derive(Clone, Debug)]
pub struct AnalysisConfig {
    pub use_gpu: bool,
    pub auto_update_viewport: bool,
    pub skip_background: bool,
    pub background_threshold: u8,
    pub mip_level: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            use_gpu: false,
            auto_update_viewport: false,
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
    {
        let mut state = plugin_state().lock().unwrap();
        state.job = Some(RunningJob { cancel: cancel.clone() });
        state.progress_value = 0.0;
        state.job_status = match kind {
            JobKind::Viewport => "Analyzing viewport".to_string(),
            JobKind::WholeSlide => "Analyzing whole slide".to_string(),
        };
        state.cache_namespace = namespace;
    }
    refresh_sidebar_if_available();

    thread::spawn(move || {
        let result = worker(cancel.clone());
        let mut state = plugin_state().lock().unwrap();
        state.job = None;
        state.progress_value = if cancel.load(Ordering::Relaxed) { 0.0 } else { 1.0 };
        state.job_status = if cancel.load(Ordering::Relaxed) {
            "Analysis cancelled".to_string()
        } else if let Err(error) = &result {
            error.clone()
        } else {
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
    let (skip_background, background_threshold) = {
        let state = plugin_state().lock().unwrap();
        (state.config.skip_background, state.config.background_threshold)
    };

    for (index, tile_plan) in tiles.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

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
        if skip_background && should_skip_background(bytes.as_slice(), background_threshold) {
            update_progress(index + 1, tiles.len(), "Skipping background tile");
            continue;
        }

        let reconstruction = run_reconstruction(
            &model,
            bytes.as_slice(),
            tile_plan.read_width,
            tile_plan.read_height,
        )?;
        let tile = build_analyzed_tile(
            tile_plan.x,
            tile_plan.y,
            tile_plan.width,
            tile_plan.height,
            tile_plan.read_width,
            tile_plan.read_height,
            bytes.as_slice(),
            &reconstruction.rgb,
        );
        {
            let mut state = plugin_state().lock().unwrap();
            state.cache.insert(
                tile.id(),
                TileCacheEntry {
                    namespace: namespace.clone(),
                    tile,
                },
            );
            state.progress_value = (index + 1) as f32 / tiles.len().max(1) as f32;
            state.job_status = format!("Processed {} / {} tiles", index + 1, tiles.len());
            rebuild_sidebar_statistics(&mut state);
        }
        refresh_sidebar_if_available();
        request_render_if_available();
    }

    Ok(())
}

fn update_progress(done: usize, total: usize, label: &str) {
    let mut state = plugin_state().lock().unwrap();
    state.progress_value = done as f32 / total.max(1) as f32;
    state.job_status = format!("{label} ({done}/{total})");
}

fn viewport_tiles(viewport: &ViewportSnapshotFFI, tile_size: u32, mip_level: u32) -> Vec<TilePlan> {
    let left = viewport.bounds_left.max(0.0) as u64;
    let top = viewport.bounds_top.max(0.0) as u64;
    let right = viewport.bounds_right.max(0.0) as u64;
    let bottom = viewport.bounds_bottom.max(0.0) as u64;
    grid_tiles(left, top, right, bottom, tile_size, mip_level)
}

fn full_slide_tiles(image_width: u64, image_height: u64, tile_size: u32, mip_level: u32) -> Vec<TilePlan> {
    grid_tiles(0, 0, image_width, image_height, tile_size, mip_level)
}

fn grid_tiles(
    left: u64,
    top: u64,
    right: u64,
    bottom: u64,
    tile_size: u32,
    mip_level: u32,
) -> Vec<TilePlan> {
    let mut tiles = Vec::new();
    let downsample = 1u64 << mip_level.min(3);
    let step = tile_size as u64 * downsample;
    let mut y = top;
    while y < bottom {
        let mut x = left;
        while x < right {
            let width = (right - x).min(step) as u32;
            let height = (bottom - y).min(step) as u32;
            tiles.push(TilePlan {
                x,
                y,
                width,
                height,
                level: mip_level,
                read_width: width.div_ceil(downsample as u32),
                read_height: height.div_ceil(downsample as u32),
            });
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

fn build_analyzed_tile(
    x: u64,
    y: u64,
    width: u32,
    height: u32,
    sample_width: u32,
    sample_height: u32,
    rgba: &[u8],
    reconstruction_rgb: &[u8],
) -> AnalyzedTile {
    let mut difference_rgb = vec![0u8; sample_width as usize * sample_height as usize * 3];
    let mut error_map_luma = vec![0u8; sample_width as usize * sample_height as usize];
    let mut error_sum = 0f64;
    let mut max_error = 0u8;

    for index in 0..(sample_width as usize * sample_height as usize) {
        let src = index * 4;
        let dst = index * 3;
        let dr = rgba[src].abs_diff(reconstruction_rgb[dst]);
        let dg = rgba[src + 1].abs_diff(reconstruction_rgb[dst + 1]);
        let db = rgba[src + 2].abs_diff(reconstruction_rgb[dst + 2]);
        let mean = ((dr as u16 + dg as u16 + db as u16) / 3) as u8;
        difference_rgb[dst] = dr;
        difference_rgb[dst + 1] = dg;
        difference_rgb[dst + 2] = db;
        error_map_luma[index] = mean;
        error_sum += mean as f64 / 255.0;
        max_error = max_error.max(mean);
    }

    AnalyzedTile {
        x,
        y,
        width,
        height,
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
        let mut state = crate::state::PluginState::default();
        state.cache_namespace = "ns".to_string();
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
        let tiles = grid_tiles(0, 0, 700, 700, 256, 1);

        assert_eq!(tiles[0].width, 512);
        assert_eq!(tiles[0].height, 512);
        assert_eq!(tiles[0].read_width, 256);
        assert_eq!(tiles[0].read_height, 256);
        assert_eq!(tiles[0].level, 1);

        let edge = tiles.last().unwrap();
        assert_eq!(edge.width, 188);
        assert_eq!(edge.height, 188);
        assert_eq!(edge.read_width, 94);
        assert_eq!(edge.read_height, 94);
    }
}