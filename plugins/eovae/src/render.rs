//! Building the overlay rasters the host draws.
//!
//! Nothing here is drawn per frame by the plugin: the host keeps the
//! rasters in level-0 coordinates and composites them off the UI thread on
//! any render backend, so eovae needs no CPU viewport filter (which forced
//! CPU rendering and ran on render workers every frame, VAE-13, VAE-16).
//!
//! - **Error map**: a heat map of the whole slide with a few pixels per
//!   tile, coloured by tile error, complete at any zoom. It is sent as
//!   chunks of [`HeatMap::chunks`], so new results re-send only the chunks
//!   they touch.
//! - **Reconstruction / Difference**: a mosaic of the view (plus a margin)
//!   at screen resolution, built from tile pixels with a box filter when
//!   zoomed out. Its size is bounded, however much is analysed (VAE-6).
//!   Reconstructions get the pane's display adjustments ([`DisplayTone`]).

use crate::results::{TilePayload, TileState};
use crate::tiling::{TileCoord, TileGrid};
use eov_plugin_api::protocol::{DisplayAdjustments, Rect, ViewInfo};
use eov_plugin_api::sdk::Image;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// What a slide's overlay shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Original,
    Reconstruction,
    ErrorMap,
    Difference,
}

impl Mode {
    pub const ALL: [Mode; 4] = [
        Self::Original,
        Self::Reconstruction,
        Self::ErrorMap,
        Self::Difference,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Reconstruction => "reconstruction",
            Self::ErrorMap => "error_map",
            Self::Difference => "difference",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Original => "Original",
            Self::Reconstruction => "Reconstruction",
            Self::ErrorMap => "Error map",
            Self::Difference => "Difference",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.id() == id)
    }

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|m| *m == self).unwrap_or(0)
    }

    /// Whether the mode draws tile pixels (and so needs payloads).
    pub fn needs_pixels(self) -> bool {
        matches!(self, Self::Reconstruction | Self::Difference)
    }
}

/// Perceptually uniform, colour-blind-safe ramp (viridis) with opacity
/// rising with the value, so low errors keep the tissue visible.
pub fn colormap(t: f32) -> [u8; 4] {
    const STOPS: [[f32; 3]; 8] = [
        [68.0, 1.0, 84.0],
        [70.0, 50.0, 127.0],
        [54.0, 92.0, 141.0],
        [39.0, 127.0, 142.0],
        [31.0, 161.0, 135.0],
        [74.0, 194.0, 109.0],
        [159.0, 218.0, 58.0],
        [253.0, 231.0, 37.0],
    ];
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let position = t * (STOPS.len() - 1) as f32;
    let index = (position.floor() as usize).min(STOPS.len() - 2);
    let fraction = position - index as f32;
    let a = STOPS[index];
    let b = STOPS[index + 1];
    let channel = |c: usize| (a[c] + (b[c] - a[c]) * fraction).round() as u8;
    [
        channel(0),
        channel(1),
        channel(2),
        (110.0 + 110.0 * t).round() as u8,
    ]
}

/// Colours of tile states that have no score.
pub const PENDING: [u8; 3] = [0xF2, 0xD5, 0x4A];
pub const FAILED: [u8; 4] = [0xD9, 0x43, 0x43, 0x70];

/// Largest heat raster, in pixels.
const MAX_HEAT_PIXELS: u64 = 2_000_000;
/// Largest view mosaic, in pixels.
pub const MAX_MOSAIC_PIXELS: f64 = 4_000_000.0;

/// The shape of a run's heat map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatLayout {
    /// Size in heat-map pixels.
    pub width: u32,
    pub height: u32,
    /// Level-0 pixels per heat-map pixel.
    pub pixel0: f64,
}

/// The error map of a run: each tile spans `p×p` pixels (`p` = 1–8, chosen
/// to bound the map; on very large grids several tiles share a pixel and the
/// worst score wins). Pixels outside the slide stay transparent, so partial
/// edge tiles do not spill over the canvas.
pub struct HeatMap {
    layout: HeatLayout,
    data: Vec<u8>,
}

/// One chunk of a [`HeatMap`].
pub struct HeatChunk {
    /// Chunk column and row.
    pub col: u32,
    pub row: u32,
    /// Level-0 rectangle the chunk covers.
    pub rect: Rect,
    /// Its pixels and a hash of them; `None` when fully transparent.
    pub content: Option<(Image, u64)>,
}

impl HeatMap {
    pub fn new(grid: &TileGrid, tiles: &[(TileCoord, TileState)], lo: f32, hi: f32) -> Self {
        let tiles_total = u64::from(grid.cols) * u64::from(grid.rows);
        let per_tile = ((MAX_HEAT_PIXELS as f64 / tiles_total.max(1) as f64)
            .sqrt()
            .floor() as u32)
            .clamp(1, 8);
        let mut group = 1u32;
        while u64::from(grid.cols.div_ceil(group)) * u64::from(grid.rows.div_ceil(group))
            > MAX_HEAT_PIXELS
            || grid.cols.div_ceil(group) > 4096
            || grid.rows.div_ceil(group) > 4096
        {
            group += 1;
        }
        let (per_tile, group) = if group > 1 { (1, group) } else { (per_tile, 1) };
        let width = grid.cols.div_ceil(group) * per_tile;
        let height = grid.rows.div_ceil(group) * per_tile;
        let mut data = vec![0u8; width as usize * height as usize * 4];
        let span = (hi - lo).max(1e-9);
        let pixel0 = grid.step0() * f64::from(group) / f64::from(per_tile);
        for (coord, state) in tiles {
            let color = match state {
                TileState::Analysed { mae, .. } => colormap((mae - lo) / span),
                TileState::Failed => FAILED,
                TileState::Background => continue,
            };
            let gx = coord.col / group;
            let gy = coord.row / group;
            for sy in 0..per_tile {
                let py = gy * per_tile + sy;
                if f64::from(py) * pixel0 >= grid.height0 as f64 {
                    break;
                }
                for sx in 0..per_tile {
                    let px = gx * per_tile + sx;
                    if f64::from(px) * pixel0 >= grid.width0 as f64 {
                        break;
                    }
                    let i = (py as usize * width as usize + px as usize) * 4;
                    // With grouping, keep the strongest colour (highest alpha).
                    if data[i + 3] <= color[3] {
                        data[i..i + 4].copy_from_slice(&color);
                    }
                }
            }
        }
        Self {
            layout: HeatLayout {
                width,
                height,
                pixel0,
            },
            data,
        }
    }

    pub fn layout(&self) -> HeatLayout {
        self.layout
    }

    /// The colour of one heat-map pixel.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        (x < self.layout.width && y < self.layout.height).then(|| {
            let i = (y as usize * self.layout.width as usize + x as usize) * 4;
            [
                self.data[i],
                self.data[i + 1],
                self.data[i + 2],
                self.data[i + 3],
            ]
        })
    }

    /// The map cut into `size`×`size`-pixel chunks (smaller at the right and
    /// bottom edges) that tile it exactly, row by row.
    pub fn chunks(&self, size: u32) -> Vec<HeatChunk> {
        let size = size.max(1);
        let HeatLayout {
            width,
            height,
            pixel0,
        } = self.layout;
        let mut chunks = Vec::new();
        for row in 0..height.div_ceil(size) {
            for col in 0..width.div_ceil(size) {
                let (x0, y0) = (col * size, row * size);
                let (w, h) = ((width - x0).min(size), (height - y0).min(size));
                let mut pixels = Vec::with_capacity(w as usize * h as usize * 4);
                for y in y0..y0 + h {
                    let start = (y as usize * width as usize + x0 as usize) * 4;
                    pixels.extend_from_slice(&self.data[start..start + w as usize * 4]);
                }
                let content = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|px| px[3] != 0)
                    .then(|| {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        (w, h).hash(&mut hasher);
                        pixels.hash(&mut hasher);
                        let hash = hasher.finish();
                        (
                            Image {
                                width: w,
                                height: h,
                                data: pixels,
                            },
                            hash,
                        )
                    });
                chunks.push(HeatChunk {
                    col,
                    row,
                    rect: Rect {
                        x: f64::from(x0) * pixel0,
                        y: f64::from(y0) * pixel0,
                        width: f64::from(w) * pixel0,
                        height: f64::from(h) * pixel0,
                    },
                    content,
                });
            }
        }
        chunks
    }
}

/// One pixel per tile, for heat-map images: the colour ramp at full
/// opacity for scored tiles, red for unreadable ones; glass and tiles not
/// analysed stay transparent. Pixel `(col, row)` covers tile `(col, row)`.
pub fn tile_map(grid: &TileGrid, tiles: &[(TileCoord, TileState)], lo: f32, hi: f32) -> Image {
    let (width, height) = (grid.cols.max(1), grid.rows.max(1));
    let mut data = vec![0u8; width as usize * height as usize * 4];
    let span = (hi - lo).max(1e-9);
    for (coord, state) in tiles {
        if !grid.contains(*coord) {
            continue;
        }
        let color = match state {
            TileState::Analysed { mae, .. } => {
                let [r, g, b, _] = colormap((mae - lo) / span);
                [r, g, b, 255]
            }
            TileState::Failed => [FAILED[0], FAILED[1], FAILED[2], 255],
            TileState::Background => continue,
        };
        let i = (coord.row as usize * width as usize + coord.col as usize) * 4;
        data[i..i + 4].copy_from_slice(&color);
    }
    Image {
        width,
        height,
        data,
    }
}

/// How a pane shows slide pixels, from eov's display pipeline: the tone
/// curve (gamma, brightness, contrast) and screen-space sharpening, applied
/// to reconstructions so that they compare with the slide as shown (VAE-17).
/// Stain normalisation and deconvolution cannot be reproduced here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayTone {
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub sharpness: f32,
}

impl DisplayTone {
    pub fn new(display: &DisplayAdjustments) -> Self {
        Self {
            gamma: display.gamma,
            brightness: display.brightness,
            contrast: display.contrast,
            sharpness: display.sharpness,
        }
    }

    /// Neutral within eov's tolerance (0.001).
    pub fn tone_is_identity(&self) -> bool {
        (self.gamma - 1.0).abs() <= 0.001
            && self.brightness.abs() <= 0.001
            && (self.contrast - 1.0).abs() <= 0.001
    }

    pub fn sharpens(&self) -> bool {
        self.sharpness > 0.001
    }

    pub fn is_identity(&self) -> bool {
        self.tone_is_identity() && !self.sharpens()
    }

    /// eov's tone curve for a channel value in [0, 1]:
    /// `clamp((x^(1/γ) + brightness − 0.5) · contrast + 0.5, 0, 1)`.
    pub fn apply_unit(&self, x: f32) -> f32 {
        let inv_gamma = if self.gamma > 0.001 {
            1.0 / self.gamma
        } else {
            1.0
        };
        let x = x.clamp(0.0, 1.0);
        let g = if inv_gamma == 1.0 {
            x
        } else {
            x.powf(inv_gamma)
        };
        ((g + self.brightness - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0)
    }
}

/// eov's sharpening, `c + k·(4c − n − s − w − e)` rounded, over the
/// reconstruction pixels only: neighbours that are not reconstructions (or
/// outside the image) count as the centre, like eov's clamp-to-edge.
fn sharpen_masked(data: &mut [u8], width: usize, height: usize, mask: &[bool], amount: f32) {
    let source = data.to_vec();
    for y in 0..height {
        for x in 0..width {
            if !mask[y * width + x] {
                continue;
            }
            let i = (y * width + x) * 4;
            for c in 0..3 {
                let centre = f32::from(source[i + c]);
                let at = |nx: usize, ny: usize| {
                    if mask[ny * width + nx] {
                        f32::from(source[(ny * width + nx) * 4 + c])
                    } else {
                        centre
                    }
                };
                let n = if y > 0 { at(x, y - 1) } else { centre };
                let s = if y + 1 < height { at(x, y + 1) } else { centre };
                let w = if x > 0 { at(x - 1, y) } else { centre };
                let e = if x + 1 < width { at(x + 1, y) } else { centre };
                let detail = 4.0 * centre - (n + s + w + e);
                data[i + c] = (centre + amount * detail).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Where and how large a pane's mosaic is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosaicFrame {
    /// Level-0 rectangle covered.
    pub rect: Rect,
    /// Mosaic pixels per level-0 pixel.
    pub scale: f64,
    /// Screen (device) pixels per level-0 pixel of the view.
    pub screen: f64,
    pub width: u32,
    pub height: u32,
}

impl MosaicFrame {
    /// The view plus a margin, at screen resolution, never finer than the
    /// analysed level and never larger than [`MAX_MOSAIC_PIXELS`].
    pub fn for_view(view: &ViewInfo, grid: &TileGrid) -> Option<Self> {
        let margin_x = view.visible.width * 0.15;
        let margin_y = view.visible.height * 0.15;
        let x0 = (view.visible.x - margin_x).max(0.0);
        let y0 = (view.visible.y - margin_y).max(0.0);
        let x1 = (view.visible.x + view.visible.width + margin_x).min(grid.width0 as f64);
        let y1 = (view.visible.y + view.visible.height + margin_y).min(grid.height0 as f64);
        if !(x1 > x0 && y1 > y0) {
            return None;
        }
        let screen = view.zoom * view.scale_factor.max(1.0);
        let mut scale = screen.min(1.0 / grid.downsample);
        let pixels = (x1 - x0) * (y1 - y0) * scale * scale;
        if pixels > MAX_MOSAIC_PIXELS {
            scale *= (MAX_MOSAIC_PIXELS / pixels).sqrt();
        }
        let width = (((x1 - x0) * scale).round() as u32).clamp(1, 4096);
        let height = (((y1 - y0) * scale).round() as u32).clamp(1, 4096);
        Some(Self {
            rect: Rect {
                x: x0,
                y: y0,
                width: f64::from(width) / scale,
                height: f64::from(height) / scale,
            },
            scale,
            screen,
            width,
            height,
        })
    }

    /// Whether mosaic pixels are about screen pixels (so screen-space
    /// sharpening can be reproduced on them).
    pub fn at_screen_resolution(&self) -> bool {
        self.screen > 0.0 && (self.scale / self.screen - 1.0).abs() < 0.25
    }

    /// Whether `other` shows the same thing closely enough to keep this one.
    pub fn covers(&self, other: &MosaicFrame) -> bool {
        let inside = other.rect.x >= self.rect.x - 1.0
            && other.rect.y >= self.rect.y - 1.0
            && other.rect.x + other.rect.width <= self.rect.x + self.rect.width + 1.0
            && other.rect.y + other.rect.height <= self.rect.y + self.rect.height + 1.0;
        inside && (other.scale / self.scale - 1.0).abs() < 0.2
    }
}

/// A tile as the mosaic sees it.
#[derive(Clone, Debug)]
pub struct MosaicTile {
    pub coord: TileCoord,
    pub state: Option<TileState>,
    pub payload: Option<Arc<TilePayload>>,
}

/// Renders a pane's mosaic. `error_hi` is the per-pixel error drawn at the
/// top of the colour ramp in Difference mode; `display` is how the pane
/// shows slide pixels, applied to reconstructions.
pub fn mosaic(
    grid: &TileGrid,
    frame: &MosaicFrame,
    tiles: &[MosaicTile],
    mode: Mode,
    error_hi: f32,
    display: Option<&DisplayTone>,
) -> Image {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut data = vec![0u8; width * height * 4];
    let display = display.filter(|_| mode == Mode::Reconstruction);
    // The tone curve runs on the filtered value, like eov's (resample, then
    // tone, then round once).
    let tone = display.filter(|d| !d.tone_is_identity());
    let mut sharpen = display
        .filter(|d| d.sharpens() && frame.at_screen_resolution())
        .map(|d| (d.sharpness, vec![false; width * height]));
    let step = grid.step0();
    let k = frame.scale;
    // Source (level) pixels per mosaic pixel.
    let ratio = 1.0 / (k * grid.downsample);
    let samples = if ratio <= 1.5 {
        1
    } else {
        (ratio.ceil() as usize).min(4)
    };
    for tile in tiles {
        let footprint = grid.valid_footprint(tile.coord);
        let dx0 = ((footprint.x - frame.rect.x) * k).round().max(0.0) as usize;
        let dy0 = ((footprint.y - frame.rect.y) * k).round().max(0.0) as usize;
        let dx1 = (((footprint.x + footprint.width - frame.rect.x) * k)
            .round()
            .max(0.0) as usize)
            .min(width);
        let dy1 = (((footprint.y + footprint.height - frame.rect.y) * k)
            .round()
            .max(0.0) as usize)
            .min(height);
        if dx1 <= dx0 || dy1 <= dy0 {
            continue;
        }
        let tile_x0 = f64::from(tile.coord.col) * step;
        let tile_y0 = f64::from(tile.coord.row) * step;
        match (&tile.state, &tile.payload) {
            (Some(TileState::Background), _) => {}
            (Some(TileState::Failed), _) => {
                for dy in dy0..dy1 {
                    for dx in dx0..dx1 {
                        let i = (dy * width + dx) * 4;
                        data[i..i + 4].copy_from_slice(&FAILED);
                    }
                }
            }
            (Some(TileState::Analysed { .. }), Some(payload)) => {
                let max_x = payload.valid_w.max(1) - 1;
                let max_y = payload.valid_h.max(1) - 1;
                let to_source = |d: usize, offset: f64, origin: f64, tile_origin: f64| {
                    let level0 = origin + (d as f64 + offset) / k;
                    (level0 - tile_origin) / grid.downsample
                };
                for dy in dy0..dy1 {
                    for dx in dx0..dx1 {
                        let mut acc = [0u32; 4];
                        for sy in 0..samples {
                            let oy = (sy as f64 + 0.5) / samples as f64;
                            let v = to_source(dy, oy, frame.rect.y, tile_y0)
                                .floor()
                                .clamp(0.0, f64::from(max_y))
                                as u32;
                            for sx in 0..samples {
                                let ox = (sx as f64 + 0.5) / samples as f64;
                                let u = to_source(dx, ox, frame.rect.x, tile_x0)
                                    .floor()
                                    .clamp(0.0, f64::from(max_x))
                                    as u32;
                                let px = match mode {
                                    Mode::Difference => {
                                        let e = payload.error_at(u, v);
                                        colormap(e / error_hi.max(1e-6))
                                    }
                                    _ => {
                                        let s = ((v * payload.size + u) * 3) as usize;
                                        [
                                            payload.recon[s],
                                            payload.recon[s + 1],
                                            payload.recon[s + 2],
                                            255,
                                        ]
                                    }
                                };
                                for c in 0..4 {
                                    acc[c] += u32::from(px[c]);
                                }
                            }
                        }
                        let n = (samples * samples) as u32;
                        let i = (dy * width + dx) * 4;
                        match tone {
                            Some(tone) => {
                                let scale = 1.0 / (n as f32 * 255.0);
                                for c in 0..3 {
                                    data[i + c] = (tone.apply_unit(acc[c] as f32 * scale) * 255.0)
                                        .round()
                                        .clamp(0.0, 255.0)
                                        as u8;
                                }
                                data[i + 3] = ((acc[3] + n / 2) / n) as u8;
                            }
                            None => {
                                for c in 0..4 {
                                    data[i + c] = ((acc[c] + n / 2) / n) as u8;
                                }
                            }
                        }
                        if let Some((_, mask)) = &mut sharpen {
                            mask[dy * width + dx] = true;
                        }
                    }
                }
            }
            // Not analysed yet (or its pixels must be recomputed): stripes.
            _ => {
                for dy in dy0..dy1 {
                    for dx in dx0..dx1 {
                        let stripe = ((dx + dy) / 8) % 2 == 0;
                        let i = (dy * width + dx) * 4;
                        data[i..i + 3].copy_from_slice(&PENDING);
                        data[i + 3] = if stripe { 0x50 } else { 0x18 };
                    }
                }
            }
        }
    }
    if let Some((amount, mask)) = &sharpen {
        sharpen_masked(&mut data, width, height, mask, *amount);
    }
    Image {
        width: frame.width,
        height: frame.height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::SlideGeometry;
    use eov_plugin_api::protocol::{FileInfo, LevelInfo};

    fn grid(width: u64, height: u64, downsample: f64) -> TileGrid {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width,
            height,
            levels: vec![LevelInfo {
                level: 0,
                width: (width as f64 / downsample) as u64,
                height: (height as f64 / downsample) as u64,
                downsample,
            }],
            ..Default::default()
        });
        TileGrid::new(&geometry, &geometry.levels[0], 16)
    }

    #[test]
    fn colormap_is_monotonic_in_opacity_and_clamped() {
        assert_eq!(colormap(-1.0), colormap(0.0));
        assert_eq!(colormap(f32::NAN), colormap(0.0));
        assert!(colormap(1.0)[3] > colormap(0.0)[3]);
        assert_eq!(colormap(1.0)[..3], [253, 231, 37]);
    }

    #[test]
    fn heat_maps_cover_the_grid_and_skip_background() {
        let grid = grid(100, 40, 1.0);
        assert_eq!((grid.cols, grid.rows), (7, 3));
        let tiles = vec![
            (
                TileCoord::new(0, 0),
                TileState::Analysed { mae: 0.0, max: 0.0 },
            ),
            (
                TileCoord::new(6, 2),
                TileState::Analysed { mae: 1.0, max: 1.0 },
            ),
            (TileCoord::new(1, 0), TileState::Background),
            (TileCoord::new(2, 0), TileState::Failed),
        ];
        let heat = HeatMap::new(&grid, &tiles, 0.0, 1.0);
        let layout = heat.layout();
        let per_tile = layout.width / grid.cols;
        assert!(per_tile >= 1);
        assert_eq!(layout.pixel0, 16.0 / f64::from(per_tile));
        assert_eq!(heat.pixel(0, 0).unwrap(), colormap(0.0));
        assert_eq!(
            heat.pixel(per_tile, 0).unwrap()[3],
            0,
            "background is transparent"
        );
        assert_eq!(heat.pixel(2 * per_tile, 0).unwrap(), FAILED);
        // The last tile is partial (100 = 6·16 + 4): pixels past the slide
        // edge stay transparent.
        let edge_x = 6 * per_tile + per_tile - 1;
        if per_tile > 1 {
            assert_eq!(heat.pixel(edge_x, 2 * per_tile).unwrap()[3], 0);
        }
        assert_eq!(
            heat.pixel(6 * per_tile, 2 * per_tile).unwrap(),
            colormap(1.0)
        );
    }

    #[test]
    fn huge_grids_group_tiles_to_bound_the_heat_map() {
        let grid = grid(16 * 5000, 16 * 3000, 1.0);
        let layout = HeatMap::new(&grid, &[], 0.0, 1.0).layout();
        assert!(u64::from(layout.width) * u64::from(layout.height) <= MAX_HEAT_PIXELS);
        assert!(layout.width <= 4096);
    }

    #[test]
    fn heat_chunks_tile_the_map_exactly_and_change_only_where_tiles_did() {
        let grid = grid(1000, 600, 1.0);
        let analysed = |col, row| {
            (
                TileCoord::new(col, row),
                TileState::Analysed { mae: 0.5, max: 0.9 },
            )
        };
        let before = HeatMap::new(&grid, &[analysed(0, 0)], 0.0, 1.0);
        let layout = before.layout();
        let chunks = before.chunks(64);
        // Chunks tile the map: no gaps or overlaps, partial at the edges.
        let covered: u64 = chunks
            .iter()
            .map(|c| (c.rect.width / layout.pixel0 * c.rect.height / layout.pixel0) as u64)
            .sum();
        assert_eq!(covered, u64::from(layout.width) * u64::from(layout.height));
        for chunk in &chunks {
            let right = chunks
                .iter()
                .find(|c| c.row == chunk.row && c.col == chunk.col + 1);
            if let Some(right) = right {
                assert_eq!(chunk.rect.x + chunk.rect.width, right.rect.x);
            }
        }
        // Only the chunk with the analysed tile has pixels.
        assert_eq!(chunks.iter().filter(|c| c.content.is_some()).count(), 1);
        // A tile far away changes only its own chunk.
        let after = HeatMap::new(&grid, &[analysed(0, 0), analysed(60, 35)], 0.0, 1.0);
        let changed: Vec<(u32, u32)> = before
            .chunks(64)
            .iter()
            .zip(after.chunks(64))
            .filter(|(a, b)| {
                a.content.as_ref().map(|(_, h)| *h) != b.content.as_ref().map(|(_, h)| *h)
            })
            .map(|(a, _)| (a.col, a.row))
            .collect();
        assert_eq!(changed.len(), 1, "{changed:?}");
        let (col, row) = changed[0];
        assert!(col > 0 && row > 0);
    }

    #[test]
    fn display_tone_follows_eovs_curve() {
        let neutral = DisplayTone::new(&DisplayAdjustments::default());
        assert!(neutral.is_identity());
        for x in [0.0, 0.25, 0.5, 1.0] {
            assert_eq!(neutral.apply_unit(x), x);
        }
        let tone = DisplayTone {
            gamma: 2.0,
            brightness: 0.1,
            contrast: 1.5,
            sharpness: 0.0,
        };
        // clamp((x^(1/γ) + b − 0.5)·c + 0.5)
        let x: f32 = 0.25;
        let expected = ((x.powf(0.5) + 0.1 - 0.5) * 1.5 + 0.5).clamp(0.0, 1.0);
        assert!((tone.apply_unit(x) - expected).abs() < 1e-6);
        assert_eq!(tone.apply_unit(2.0), 1.0);
    }

    fn view(x: f64, y: f64, w: f64, h: f64, zoom: f64) -> ViewInfo {
        ViewInfo {
            zoom,
            width: w * zoom,
            height: h * zoom,
            scale_factor: 1.0,
            visible: Rect {
                x,
                y,
                width: w,
                height: h,
            },
            ..Default::default()
        }
    }

    #[test]
    fn mosaics_are_bounded_and_never_finer_than_the_level() {
        let grid = grid(100_000, 100_000, 4.0);
        // Zoomed far in: limited by the level (1/4 px per level-0 px).
        let frame = MosaicFrame::for_view(&view(1000.0, 1000.0, 200.0, 100.0, 8.0), &grid).unwrap();
        assert!((frame.scale - 0.25).abs() < 1e-9);
        // Zoomed out over a huge area: limited by the pixel budget.
        let frame =
            MosaicFrame::for_view(&view(0.0, 0.0, 90_000.0, 90_000.0, 0.05), &grid).unwrap();
        assert!(f64::from(frame.width) * f64::from(frame.height) <= MAX_MOSAIC_PIXELS * 1.01);
        assert!(frame.covers(&frame));
        // Outside the slide there is nothing to draw.
        assert!(MosaicFrame::for_view(&view(200_000.0, 0.0, 10.0, 10.0, 1.0), &grid).is_none());
    }

    #[test]
    fn mosaics_draw_pixels_stripes_and_nothing_for_background() {
        let grid = grid(48, 16, 1.0);
        let payload = Arc::new(TilePayload {
            size: 16,
            valid_w: 16,
            valid_h: 16,
            recon: [10u8, 20, 30].repeat(256),
            error: vec![crate::contract::encode_error(0.5); 256],
        });
        let tiles = vec![
            MosaicTile {
                coord: TileCoord::new(0, 0),
                state: Some(TileState::Analysed { mae: 0.1, max: 0.2 }),
                payload: Some(payload),
            },
            MosaicTile {
                coord: TileCoord::new(1, 0),
                state: None,
                payload: None,
            },
            MosaicTile {
                coord: TileCoord::new(2, 0),
                state: Some(TileState::Background),
                payload: None,
            },
        ];
        let frame = MosaicFrame {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 48.0,
                height: 16.0,
            },
            scale: 0.5,
            screen: 0.5,
            width: 24,
            height: 8,
        };
        let image = mosaic(&grid, &frame, &tiles, Mode::Reconstruction, 1.0, None);
        assert_eq!(image.pixel(2, 2).unwrap(), [10, 20, 30, 255]);
        assert_eq!(image.pixel(10, 2).unwrap()[..3], PENDING);
        assert_eq!(image.pixel(20, 2).unwrap()[3], 0);
        // With the pane's display adjustments: eov's tone curve, rounded
        // once; markers and the difference map are left alone.
        let tone = DisplayTone {
            gamma: 2.0,
            brightness: 0.0,
            contrast: 1.2,
            sharpness: 0.0,
        };
        let adjusted = mosaic(
            &grid,
            &frame,
            &tiles,
            Mode::Reconstruction,
            1.0,
            Some(&tone),
        );
        let expect = |v: u8| (tone.apply_unit(f32::from(v) / 255.0) * 255.0).round() as u8;
        assert_eq!(
            adjusted.pixel(2, 2).unwrap(),
            [expect(10), expect(20), expect(30), 255]
        );
        assert_eq!(adjusted.pixel(10, 2).unwrap(), image.pixel(10, 2).unwrap());
        // Sharpening keeps flat areas flat and leaves stripes alone.
        let sharp = DisplayTone {
            sharpness: 0.5,
            ..tone
        };
        let sharpened = mosaic(
            &grid,
            &frame,
            &tiles,
            Mode::Reconstruction,
            1.0,
            Some(&sharp),
        );
        assert_eq!(
            sharpened.pixel(2, 2).unwrap(),
            adjusted.pixel(2, 2).unwrap()
        );
        assert_eq!(sharpened.pixel(10, 2).unwrap(), image.pixel(10, 2).unwrap());
        let difference = mosaic(&grid, &frame, &tiles, Mode::Difference, 1.0, Some(&tone));
        let image = mosaic(&grid, &frame, &tiles, Mode::Difference, 1.0, None);
        assert_eq!(difference.data, image.data);
        let expected = colormap(crate::contract::decode_error(
            crate::contract::encode_error(0.5),
        ));
        let got = image.pixel(2, 2).unwrap();
        for c in 0..4 {
            assert!((i32::from(got[c]) - i32::from(expected[c])).abs() <= 1);
        }
    }
}
