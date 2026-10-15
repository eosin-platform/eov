//! Banded export rendering (EXP-1, EXP-2).
//!
//! The export image is produced in horizontal bands. For every band the
//! renderer enumerates — without any cap — the tiles of the chosen pyramid
//! level that intersect it, loads the missing ones (in parallel), blits them
//! with the requested filter, runs the canonical processing pipeline
//! ([`crate::processing`]), composites the background, sharpens with a
//! one-row halo, draws the overlay display list and hands the finished rows
//! to a [`BandSink`] (an encoder or an in-memory image). Tiles above the
//! current band are dropped, so memory is bounded by about two rows of tiles
//! plus one band, whatever the output size. Progress is reported and
//! cancellation honoured per band through a [`JobControl`]; run it on a
//! worker thread.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::blitter::{self, BlitRect, CoarseSrc, TileSrc};
use crate::job::JobControl;
use crate::overlay::{Canvas, OverlayItem, embedded_font};
use crate::postprocess::sharpen_rows_into;
use crate::processing::{ProcessingPipeline, composite_onto};
use crate::render::{FilteringMode, TrilinearLevels, calculate_trilinear_levels};
use crate::viewport::ViewportBounds;
use crate::{RgbaImageData, TileCache, TileCoord, TileData, TileManager, Viewport};

/// Largest export dimension accepted (pixels per side).
pub const MAX_EXPORT_SIDE: u32 = 1 << 20;

/// Why an export did not complete.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("export cancelled")]
    Cancelled,
    #[error("invalid export request: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Slide(#[from] crate::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image encoding failed: {0}")]
    Encode(String),
}

/// Where tiles come from.
pub struct ExportSource<'a> {
    /// Tile access for the slide (a handle owned by the export worker).
    pub tile_manager: &'a TileManager,
    /// The viewer's tile cache, consulted (never filled) to reuse tiles that
    /// are already decoded.
    pub tile_cache: Option<&'a TileCache>,
}

/// What to export.
#[derive(Clone, Debug)]
pub struct ExportRequest {
    /// The view to export, as on screen (logical pixels).
    pub viewport: Viewport,
    /// Output pixels per viewport pixel (`dpi / 96`).
    pub scale: f64,
    pub filtering: FilteringMode,
    /// Colour processing (normally resolved once per slide and shared with
    /// the viewer, see `processing::ColorSettings::pipeline`).
    pub processing: ProcessingPipeline,
    /// Colour outside the slide.
    pub background: [u8; 3],
    /// Overlays in output-pixel coordinates (see [`ExportRequest::export_viewport`]).
    pub overlays: Vec<OverlayItem>,
    /// Rows per band (`None` = automatic, ~48 MB per band).
    pub band_rows: Option<u32>,
}

impl ExportRequest {
    /// Exports `viewport` at `dpi` (96 = one output pixel per screen pixel).
    pub fn new(viewport: Viewport, dpi: f64) -> Self {
        Self {
            viewport,
            scale: dpi / 96.0,
            filtering: FilteringMode::Trilinear,
            processing: ProcessingPipeline::identity(),
            background: [255, 255, 255],
            overlays: Vec::new(),
            band_rows: None,
        }
    }

    /// Exports a level-0 rectangle with one output pixel per `downsample`
    /// level-0 pixels (`downsample = 1` is full resolution).
    pub fn region(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        downsample: f64,
        image_size: (f64, f64),
    ) -> Self {
        let downsample = downsample.max(1e-6);
        let viewport = Viewport {
            center: glam::DVec2::new(x + width / 2.0, y + height / 2.0),
            zoom: 1.0 / downsample,
            width: (width / downsample).round().max(1.0),
            height: (height / downsample).round().max(1.0),
            image_width: image_size.0,
            image_height: image_size.1,
        };
        Self::new(viewport, 96.0)
    }

    /// Output dimensions.
    pub fn output_size(&self) -> (u32, u32) {
        let w = (self.viewport.width * self.scale).round().max(1.0);
        let h = (self.viewport.height * self.scale).round().max(1.0);
        (w.min(u32::MAX as f64) as u32, h.min(u32::MAX as f64) as u32)
    }

    /// The viewport in output pixels (use it to map level-0 coordinates of
    /// overlays to output pixels with `image_to_screen`).
    pub fn export_viewport(&self) -> Viewport {
        let (w, h) = self.output_size();
        Viewport {
            center: self.viewport.center,
            zoom: self.viewport.zoom * self.scale,
            width: w as f64,
            height: h as f64,
            image_width: self.viewport.image_width,
            image_height: self.viewport.image_height,
        }
    }

    /// Level-0 pixels per output pixel.
    pub fn level0_per_output_px(&self) -> f64 {
        1.0 / (self.viewport.zoom * self.scale)
    }

    /// Physical size of an output pixel per axis (µm), from the slide's
    /// level-0 pixel size.
    pub fn output_pixel_size_um(
        &self,
        mpp_x: Option<f64>,
        mpp_y: Option<f64>,
    ) -> (Option<f64>, Option<f64>) {
        let f = self.level0_per_output_px();
        let valid = |v: Option<f64>| v.filter(|m| m.is_finite() && *m > 0.0).map(|m| m * f);
        (valid(mpp_x), valid(mpp_y))
    }

    fn validate(&self) -> Result<(), ExportError> {
        let vp = &self.viewport;
        if !(vp.width > 0.0 && vp.height > 0.0 && vp.zoom > 0.0 && vp.zoom.is_finite()) {
            return Err(ExportError::InvalidRequest("the view has no area".into()));
        }
        if !(self.scale > 0.0 && self.scale.is_finite()) {
            return Err(ExportError::InvalidRequest(format!(
                "invalid scale {}",
                self.scale
            )));
        }
        let (w, h) = self.output_size();
        if w > MAX_EXPORT_SIDE || h > MAX_EXPORT_SIDE {
            return Err(ExportError::InvalidRequest(format!(
                "{w}×{h} px exceeds the {MAX_EXPORT_SIDE} px limit per side"
            )));
        }
        Ok(())
    }
}

/// Receives finished, opaque RGBA rows from top to bottom.
pub trait BandSink {
    fn begin(&mut self, width: u32, height: u32) -> Result<(), ExportError>;
    fn write_rows(&mut self, rgba: &[u8], rows: u32) -> Result<(), ExportError>;
    fn finish(&mut self) -> Result<(), ExportError>;
}

/// Collects the whole image in memory.
#[derive(Default)]
pub struct ImageSink {
    image: RgbaImageData,
}

impl ImageSink {
    pub fn into_image(self) -> RgbaImageData {
        self.image
    }
}

impl BandSink for ImageSink {
    fn begin(&mut self, width: u32, height: u32) -> Result<(), ExportError> {
        let bytes = width as usize * height as usize * 4;
        self.image = RgbaImageData {
            width: width as usize,
            height: height as usize,
            pixels: Vec::with_capacity(bytes),
        };
        Ok(())
    }

    fn write_rows(&mut self, rgba: &[u8], _rows: u32) -> Result<(), ExportError> {
        self.image.pixels.extend_from_slice(rgba);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ExportError> {
        Ok(())
    }
}

/// What an export actually did.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportReport {
    pub width: u32,
    pub height: u32,
    /// Pyramid level the image was rendered from.
    pub level: u32,
    /// Filter actually used (Lanczos falls back to trilinear when
    /// minifying, as in the viewer).
    pub filtering: FilteringMode,
    pub tiles_drawn: u64,
    /// Tiles that could not be read; their area was filled from a coarser
    /// level when possible (EXP-1: never silently).
    pub tiles_failed: u64,
    pub failed_tiles: Vec<TileCoord>,
    pub used_fallback: bool,
    pub elapsed: Duration,
}

impl ExportReport {
    /// `true` when every tile was rendered at the intended resolution.
    pub fn is_complete(&self) -> bool {
        self.tiles_failed == 0
    }

    /// A user-facing description of any shortfall.
    pub fn shortfall_message(&self) -> Option<String> {
        (self.tiles_failed > 0).then(|| {
            format!(
                "{} of {} tiles could not be read{}.",
                self.tiles_failed,
                self.tiles_drawn + self.tiles_failed,
                if self.used_fallback {
                    "; those areas were filled from a lower-resolution level"
                } else {
                    " and are left as background"
                }
            )
        })
    }
}

/// Adaptive Lanczos → trilinear switch, mirroring the viewer: Lanczos-3
/// with fixed taps aliases when minifying (WSI-8).
fn lanczos_adaptive_weight(zoom: f64) -> f64 {
    const LANCZOS_FULL: f64 = 0.12;
    const TRILINEAR_FULL: f64 = 0.07;
    ((zoom - TRILINEAR_FULL) / (LANCZOS_FULL - TRILINEAR_FULL)).clamp(0.0, 1.0)
}

fn effective_filter(requested: FilteringMode, zoom: f64) -> FilteringMode {
    match requested {
        FilteringMode::Lanczos3 if lanczos_adaptive_weight(zoom) < 1.0 => FilteringMode::Trilinear,
        other => other,
    }
}

/// Decoded tiles for the bands in flight. Tiles entirely above the next
/// band are dropped.
struct TileStore<'a> {
    source: &'a ExportSource<'a>,
    tiles: HashMap<TileCoord, Option<Arc<TileData>>>,
}

impl<'a> TileStore<'a> {
    fn new(source: &'a ExportSource<'a>) -> Self {
        Self {
            source,
            tiles: HashMap::new(),
        }
    }

    /// Loads every coordinate that is not cached yet (in parallel).
    fn ensure(&mut self, coords: &[TileCoord]) {
        let missing: Vec<TileCoord> = coords
            .iter()
            .filter(|c| !self.tiles.contains_key(c))
            .copied()
            .collect();
        let manager = self.source.tile_manager;
        let cache = self.source.tile_cache;
        let loaded: Vec<(TileCoord, Option<Arc<TileData>>)> = missing
            .par_iter()
            .map(|coord| {
                if let Some(hit) = cache.and_then(|c| c.peek(coord)) {
                    return (*coord, Some(hit));
                }
                match manager.load_tile_sync(*coord) {
                    Ok(tile) => (*coord, Some(Arc::new(tile))),
                    Err(err) => {
                        tracing::warn!("export: tile {coord:?} could not be read: {err}");
                        (*coord, None)
                    }
                }
            })
            .collect();
        self.tiles.extend(loaded);
    }

    fn get(&self, coord: &TileCoord) -> Option<&Arc<TileData>> {
        self.tiles.get(coord).and_then(Option::as_ref)
    }

    /// Drops tiles whose level-0 bottom edge is above `top`.
    fn evict_above(&mut self, top: f64, manager: &TileManager) {
        self.tiles.retain(|coord, _| {
            let Some(info) = manager.wsi().level(coord.level) else {
                return false;
            };
            let bottom = (coord.y + 1) as f64 * coord.tile_size as f64 * info.downsample;
            bottom > top
        });
    }
}

/// Tiles of `level` intersecting the level-0 rectangle (uncapped, EXP-1),
/// clamped with the level's real tile grid (WSI-5).
fn tiles_in_rect(
    manager: &TileManager,
    level: u32,
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
) -> Vec<TileCoord> {
    let Some(info) = manager.wsi().level(level) else {
        return Vec::new();
    };
    let ts = manager.tile_size_for_level(level);
    let span = ts as f64 * info.downsample;
    let x0 = (left / span).floor().max(0.0) as u64;
    let y0 = (top / span).floor().max(0.0) as u64;
    let x1 = ((right / span).ceil().max(0.0) as u64).min(info.tiles_x(ts));
    let y1 = ((bottom / span).ceil().max(0.0) as u64).min(info.tiles_y(ts));
    let mut out = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            out.push(TileCoord::new(manager.file_id(), level, x, y, ts));
        }
    }
    out
}

#[cfg(test)]
pub(super) fn tiles_in_rect_for_test(
    manager: &TileManager,
    level: u32,
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
) -> usize {
    tiles_in_rect(manager, level, left, top, right, bottom).len()
}

/// Output-pixel rectangle of a tile, relative to output row `row_offset`.
fn tile_rect(
    zoom: f64,
    bounds: &ViewportBounds,
    downsample: f64,
    coord: &TileCoord,
    tile: &TileData,
    row_offset: u32,
) -> BlitRect {
    let origin_x = coord.x as f64 * coord.tile_size as f64;
    let origin_y = coord.y as f64 * coord.tile_size as f64;
    let sx = (origin_x * downsample - bounds.left) * zoom;
    let sy = (origin_y * downsample - bounds.top) * zoom - row_offset as f64;
    let sx_end = ((origin_x + tile.width as f64) * downsample - bounds.left) * zoom;
    let sy_end =
        ((origin_y + tile.height as f64) * downsample - bounds.top) * zoom - row_offset as f64;
    let x = sx.floor() as i32;
    let y = sy.floor() as i32;
    BlitRect {
        x,
        y,
        width: sx_end.floor() as i32 - x,
        height: sy_end.floor() as i32 - y,
        exact_x: sx,
        exact_y: sy,
        exact_width: sx_end - sx,
        exact_height: sy_end - sy,
    }
}

struct CoarseBlend {
    tile: Arc<TileData>,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    blend: f32,
}

/// The coarse tile and UV sub-rectangle under a fine tile, for fused
/// trilinear blending (same rule as the viewer).
fn coarse_blend(
    manager: &TileManager,
    store: &TileStore,
    trilinear: TrilinearLevels,
    fine_downsample: f64,
    fine: &TileCoord,
    fine_tile: &TileData,
) -> Option<CoarseBlend> {
    const EPSILON: f64 = 1e-3;
    let info = manager.wsi().level(trilinear.level_coarse)?;
    let x0 = fine.x as f64 * fine.tile_size as f64 * fine_downsample;
    let y0 = fine.y as f64 * fine.tile_size as f64 * fine_downsample;
    let w0 = fine_tile.width as f64 * fine_downsample;
    let h0 = fine_tile.height as f64 * fine_downsample;
    let ts = manager.tile_size_for_level(trilinear.level_coarse) as f64;
    let cx = x0 / info.downsample;
    let cy = y0 / info.downsample;
    let cx_end = ((x0 + w0 - EPSILON) / info.downsample).max(cx);
    let cy_end = ((y0 + h0 - EPSILON) / info.downsample).max(cy);
    let (tx, ty) = ((cx / ts).floor() as u64, (cy / ts).floor() as u64);
    if tx != (cx_end / ts).floor() as u64 || ty != (cy_end / ts).floor() as u64 {
        return None;
    }
    let coord = TileCoord::new(fine.file_id, trilinear.level_coarse, tx, ty, ts as u32);
    let tile = store.get(&coord)?.clone();
    let sx = (cx - tx as f64 * ts).max(0.0);
    let sy = (cy - ty as f64 * ts).max(0.0);
    let sw = w0 / info.downsample;
    let sh = h0 / info.downsample;
    if sx + sw > tile.width as f64 + EPSILON || sy + sh > tile.height as f64 + EPSILON {
        return None;
    }
    let (cw, ch) = (tile.width as f64, tile.height as f64);
    Some(CoarseBlend {
        uv_min: [(sx / cw) as f32, (sy / ch) as f32],
        uv_max: [((sx + sw) / cw) as f32, ((sy + sh) / ch) as f32],
        blend: trilinear.blend as f32,
        tile,
    })
}

fn blit(
    buffer: &mut [u8],
    width: u32,
    rows: u32,
    tile: &TileData,
    rect: BlitRect,
    filtering: FilteringMode,
    coarse: Option<&CoarseBlend>,
) {
    let src = TileSrc {
        data: &tile.data,
        width: tile.width,
        height: tile.height,
        border: tile.border,
    };
    match (filtering, coarse) {
        (FilteringMode::Lanczos3, _) => blitter::blit_tile_lanczos3(buffer, width, rows, src, rect),
        (FilteringMode::Trilinear, Some(c)) => {
            let csrc = CoarseSrc {
                data: &c.tile.data,
                width: c.tile.width,
                height: c.tile.height,
                border: c.tile.border,
                uv_min: c.uv_min,
                uv_max: c.uv_max,
                blend: c.blend,
            };
            blitter::blit_tile_trilinear(buffer, width, rows, src, &csrc, rect);
        }
        _ => blitter::blit_tile(buffer, width, rows, src, rect),
    }
}

struct Plan {
    width: u32,
    zoom: f64,
    bounds: ViewportBounds,
    level: u32,
    downsample: f64,
    filtering: FilteringMode,
    trilinear: Option<TrilinearLevels>,
}

#[derive(Default)]
struct Stats {
    drawn: std::collections::HashSet<TileCoord>,
    failed: std::collections::HashSet<TileCoord>,
    used_fallback: bool,
}

/// Renders output rows `[row0, row0 + rows)` (transparent where there is no
/// slide) into `buffer`.
fn render_rows(
    source: &ExportSource,
    plan: &Plan,
    store: &mut TileStore,
    buffer: &mut [u8],
    row0: u32,
    rows: u32,
    stats: &mut Stats,
) {
    let manager = source.tile_manager;
    let top = plan.bounds.top + row0 as f64 / plan.zoom;
    let bottom = plan.bounds.top + (row0 + rows) as f64 / plan.zoom;
    let (left, right) = (plan.bounds.left, plan.bounds.right);
    let fine = tiles_in_rect(manager, plan.level, left, top, right, bottom);
    let mut wanted = fine.clone();
    if let Some(tri) = plan.trilinear {
        wanted.extend(tiles_in_rect(
            manager,
            tri.level_coarse,
            left,
            top,
            right,
            bottom,
        ));
    }
    store.ensure(&wanted);

    // Fill the area of unreadable fine tiles from the nearest coarser level
    // that can be read, before drawing the fine tiles on top.
    let failed: Vec<TileCoord> = fine
        .iter()
        .filter(|c| store.get(c).is_none())
        .copied()
        .collect();
    stats.failed.extend(failed.iter().copied());
    if !failed.is_empty() {
        let level_count = manager.wsi().level_count();
        for coord in &failed {
            let span = coord.tile_size as f64 * plan.downsample;
            let (fx0, fy0) = (coord.x as f64 * span, coord.y as f64 * span);
            for fallback in (plan.level + 1)..level_count {
                let Some(info) = manager.wsi().level(fallback) else {
                    continue;
                };
                let covering = tiles_in_rect(
                    manager,
                    fallback,
                    fx0,
                    fy0.max(top),
                    fx0 + span,
                    (fy0 + span).min(bottom),
                );
                store.ensure(&covering);
                if covering.iter().all(|c| store.get(c).is_some()) {
                    for c in &covering {
                        let tile = store.get(c).expect("checked above").clone();
                        let rect =
                            tile_rect(plan.zoom, &plan.bounds, info.downsample, c, &tile, row0);
                        blit(
                            buffer,
                            plan.width,
                            rows,
                            &tile,
                            rect,
                            FilteringMode::Bilinear,
                            None,
                        );
                    }
                    stats.used_fallback = true;
                    break;
                }
            }
        }
    }

    for coord in &fine {
        let Some(tile) = store.get(coord).cloned() else {
            continue;
        };
        let rect = tile_rect(plan.zoom, &plan.bounds, plan.downsample, coord, &tile, row0);
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }
        let coarse = plan
            .trilinear
            .and_then(|tri| coarse_blend(manager, store, tri, plan.downsample, coord, &tile));
        blit(
            buffer,
            plan.width,
            rows,
            &tile,
            rect,
            plan.filtering,
            coarse.as_ref(),
        );
        stats.drawn.insert(*coord);
    }
}

/// Budget for one band's pixels.
const BAND_BUFFER_BYTES: usize = 48 << 20;
/// Budget for the decoded tiles one band needs.
const BAND_TILE_BYTES: f64 = 192.0 * 1024.0 * 1024.0;

/// Band height that keeps both the band buffer and the tiles it touches
/// within budget, whatever the zoom: memory stays bounded for any output
/// size.
fn auto_band_rows(manager: &TileManager, plan: &Plan, width: u32) -> u32 {
    let by_buffer = (BAND_BUFFER_BYTES / (width as usize * 4).max(1)).clamp(8, 512) as f64;
    let ts = manager.tile_size_for_level(plan.level) as f64;
    let span = ts * plan.downsample; // level-0 pixels per tile row
    let across = ((plan.bounds.right - plan.bounds.left) / span)
        .ceil()
        .max(1.0)
        + 1.0;
    // A fine tile row, plus the coarse partner level when blending.
    let mut row_bytes = across * (ts + 2.0) * (ts + 2.0) * 4.0;
    if plan.trilinear.is_some() {
        row_bytes *= 1.25;
    }
    let tile_rows = (BAND_TILE_BYTES / row_bytes).floor().max(1.0);
    // Output rows covered by `tile_rows` tile rows (one extra row of tiles
    // straddles every band edge).
    let by_tiles = ((tile_rows - 1.0).max(1.0) * span * plan.zoom)
        .floor()
        .max(8.0);
    by_buffer.min(by_tiles) as u32
}

/// Renders the export band by band into `sink`.
pub fn render_export_bands(
    source: &ExportSource,
    request: &ExportRequest,
    control: &JobControl,
    sink: &mut dyn BandSink,
) -> Result<ExportReport, ExportError> {
    let started = Instant::now();
    request.validate()?;
    let manager = source.tile_manager;
    let wsi = manager.wsi();
    if wsi.level_count() == 0 {
        return Err(ExportError::InvalidRequest(
            "the slide has no pyramid levels".into(),
        ));
    }
    let evp = request.export_viewport();
    let (width, height) = request.output_size();
    let filtering = effective_filter(request.filtering, evp.zoom);
    let tri =
        calculate_trilinear_levels(wsi, 1.0 / evp.zoom, filtering == FilteringMode::Trilinear);
    let level = tri.level_fine;
    let downsample = wsi.level(level).map(|l| l.downsample).unwrap_or(1.0);
    let plan = Plan {
        width,
        zoom: evp.zoom,
        bounds: evp.bounds(),
        level,
        downsample,
        filtering,
        trilinear: (filtering == FilteringMode::Trilinear
            && tri.level_fine != tri.level_coarse
            && tri.blend > 0.01)
            .then_some(tri),
    };
    let band_rows = request
        .band_rows
        .unwrap_or_else(|| auto_band_rows(manager, &plan, width))
        .max(1);
    let sharpen = request.processing.sharpness();
    let halo = u32::from(sharpen > 0.0);
    let stride = width as usize * 4;

    control.set_total(height as u64);
    sink.begin(width, height)?;
    let mut store = TileStore::new(source);
    let mut stats = Stats::default();
    let mut buffer: Vec<u8> = Vec::new();
    let mut sharpened: Vec<u8> = Vec::new();
    let mut halo_rows: Vec<u8> = Vec::new();
    let font = embedded_font();

    let mut band_start = 0u32;
    while band_start < height {
        if control.is_cancelled() {
            return Err(ExportError::Cancelled);
        }
        let band_end = (band_start + band_rows).min(height);
        let r0 = band_start.saturating_sub(halo);
        let r1 = (band_end + halo).min(height);
        let rows = r1 - r0;
        buffer.clear();
        buffer.resize(rows as usize * stride, 0);
        render_rows(source, &plan, &mut store, &mut buffer, r0, rows, &mut stats);
        request.processing.apply_color(&mut buffer);
        composite_onto(&mut buffer, request.background);

        let out_rows = band_end - band_start;
        let out: &mut [u8] = if halo == 1 {
            // Rows above/below the band (clamp-to-edge at the image border).
            let first = (band_start - r0) as usize;
            halo_rows.clear();
            let above = if band_start > 0 { 0 } else { first };
            halo_rows.extend_from_slice(&buffer[above * stride..(above + 1) * stride]);
            halo_rows
                .extend_from_slice(&buffer[first * stride..(first + out_rows as usize) * stride]);
            let below = if band_end < height {
                first + out_rows as usize
            } else {
                first + out_rows as usize - 1
            };
            halo_rows.extend_from_slice(&buffer[below * stride..(below + 1) * stride]);
            sharpened.clear();
            sharpened.resize(out_rows as usize * stride, 0);
            sharpen_rows_into(&halo_rows, width, out_rows, sharpen, &mut sharpened);
            &mut sharpened
        } else {
            &mut buffer
        };

        if !request.overlays.is_empty() {
            let mut canvas = Canvas::band(out, width, out_rows, band_start as i64);
            for item in &request.overlays {
                item.draw(&mut canvas, (width, height), font);
            }
        }
        sink.write_rows(&out[..out_rows as usize * stride], out_rows)?;
        control.advance(out_rows as u64);

        // Tiles entirely above the next band are never needed again.
        let next_top = plan.bounds.top + band_end.saturating_sub(halo) as f64 / plan.zoom;
        store.evict_above(next_top, manager);
        band_start = band_end;
    }
    sink.finish()?;

    let mut failed_tiles: Vec<TileCoord> = stats.failed.iter().copied().collect();
    failed_tiles.sort_by_key(|c| (c.y, c.x));
    let tiles_failed = failed_tiles.len() as u64;
    failed_tiles.truncate(100);
    Ok(ExportReport {
        width,
        height,
        level,
        filtering,
        tiles_drawn: stats.drawn.len() as u64,
        tiles_failed,
        failed_tiles,
        used_fallback: stats.used_fallback,
        elapsed: started.elapsed(),
    })
}

/// Renders the whole export into memory.
pub fn render_export_image(
    source: &ExportSource,
    request: &ExportRequest,
    control: &JobControl,
) -> Result<(RgbaImageData, ExportReport), ExportError> {
    let mut sink = ImageSink::default();
    let report = render_export_bands(source, request, control, &mut sink)?;
    Ok((sink.into_image(), report))
}

/// Renders a small preview (longest side at most `max_side`, EXP-2): the
/// same request — processing, filtering and overlays — at a reduced scale.
pub fn render_export_preview(
    source: &ExportSource,
    request: &ExportRequest,
    max_side: u32,
) -> Result<RgbaImageData, ExportError> {
    let (w, h) = request.output_size();
    let factor = (max_side.max(16) as f64 / w.max(h) as f64).min(1.0);
    let mut preview = request.clone();
    preview.scale *= factor;
    preview.overlays = request
        .overlays
        .iter()
        .map(|item| item.scaled(factor as f32))
        .collect();
    let control = JobControl::new();
    render_export_image(source, &preview, &control).map(|(image, _)| image)
}
