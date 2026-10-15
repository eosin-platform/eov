//! Building the overlay rasters the host draws.
//!
//! Nothing here is drawn per frame by the plugin: the host keeps the
//! rasters in level-0 coordinates and composites them off the UI thread on
//! any render backend, so eovae needs no CPU viewport filter (which forced
//! CPU rendering and ran on render workers every frame, VAE-13, VAE-16).
//!
//! - **Error map**: one raster for the whole slide with a few pixels per
//!   tile, coloured by tile error. It is complete at any zoom.
//! - **Reconstruction / Difference**: a mosaic of the view (plus a margin)
//!   at screen resolution, built from tile pixels with a box filter when
//!   zoomed out. Its size is bounded, however much is analysed (VAE-6).

use crate::results::{TilePayload, TileState};
use crate::tiling::{TileCoord, TileGrid};
use eov_plugin_api::protocol::{Rect, ViewInfo};
use eov_plugin_api::sdk::Image;
use serde::{Deserialize, Serialize};
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

/// The error-map raster of a run: `(image, level-0 rect)`.
///
/// Each tile spans `p×p` pixels (`p` = 1–8, chosen to bound the raster);
/// pixels outside the slide stay transparent, so partial edge tiles do not
/// spill over the canvas.
pub fn heat_raster(
    grid: &TileGrid,
    tiles: &[(TileCoord, TileState)],
    lo: f32,
    hi: f32,
) -> (Image, Rect) {
    let tiles_total = u64::from(grid.cols) * u64::from(grid.rows);
    let per_tile = ((MAX_HEAT_PIXELS as f64 / tiles_total.max(1) as f64)
        .sqrt()
        .floor() as u32)
        .clamp(1, 8);
    // Very large grids: several tiles per pixel (worst score wins).
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
    let step = grid.step0();
    let pixel0 = step * f64::from(group) / f64::from(per_tile);
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
    let rect = Rect {
        x: 0.0,
        y: 0.0,
        width: f64::from(width) * pixel0,
        height: f64::from(height) * pixel0,
    };
    (
        Image {
            width,
            height,
            data,
        },
        rect,
    )
}

/// Where and how large a pane's mosaic is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MosaicFrame {
    /// Level-0 rectangle covered.
    pub rect: Rect,
    /// Mosaic pixels per level-0 pixel.
    pub scale: f64,
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
            width,
            height,
        })
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
/// top of the colour ramp in Difference mode.
pub fn mosaic(
    grid: &TileGrid,
    frame: &MosaicFrame,
    tiles: &[MosaicTile],
    mode: Mode,
    error_hi: f32,
) -> Image {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut data = vec![0u8; width * height * 4];
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
                        for c in 0..4 {
                            data[i + c] = ((acc[c] + n / 2) / n) as u8;
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
    fn heat_rasters_cover_the_grid_and_skip_background() {
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
        let (image, rect) = heat_raster(&grid, &tiles, 0.0, 1.0);
        let per_tile = image.width / grid.cols;
        assert!(per_tile >= 1);
        assert_eq!(
            rect.width,
            f64::from(image.width) * 16.0 / f64::from(per_tile)
        );
        assert_eq!(image.pixel(0, 0).unwrap(), colormap(0.0));
        assert_eq!(
            image.pixel(per_tile, 0).unwrap()[3],
            0,
            "background is transparent"
        );
        assert_eq!(image.pixel(2 * per_tile, 0).unwrap(), FAILED);
        // The last tile is partial (100 = 6·16 + 4): pixels past the slide
        // edge stay transparent.
        let edge_x = 6 * per_tile + per_tile - 1;
        if per_tile > 1 {
            assert_eq!(image.pixel(edge_x, 2 * per_tile).unwrap()[3], 0);
        }
        assert_eq!(
            image.pixel(6 * per_tile, 2 * per_tile).unwrap(),
            colormap(1.0)
        );
    }

    #[test]
    fn huge_grids_group_tiles_to_bound_the_raster() {
        let grid = grid(16 * 5000, 16 * 3000, 1.0);
        let (image, _) = heat_raster(&grid, &[], 0.0, 1.0);
        assert!(u64::from(image.width) * u64::from(image.height) <= MAX_HEAT_PIXELS);
        assert!(image.width <= 4096);
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
            width: 24,
            height: 8,
        };
        let image = mosaic(&grid, &frame, &tiles, Mode::Reconstruction, 1.0);
        assert_eq!(image.pixel(2, 2).unwrap(), [10, 20, 30, 255]);
        assert_eq!(image.pixel(10, 2).unwrap()[..3], PENDING);
        assert_eq!(image.pixel(20, 2).unwrap()[3], 0);
        let image = mosaic(&grid, &frame, &tiles, Mode::Difference, 1.0);
        let expected = colormap(crate::contract::decode_error(
            crate::contract::encode_error(0.5),
        ));
        let got = image.pixel(2, 2).unwrap();
        for c in 0..4 {
            assert!((i32::from(got[c]) - i32::from(expected[c])).abs() <= 1);
        }
    }
}
