//! Tile grids on the slide's real pyramid levels.
//!
//! Pyramids do not always halve per level: Aperio slides commonly go 1, 4,
//! 16, and levels can be slightly off (8.002). Every grid here is built from
//! a level's reported downsample (VAE-3), and tiles at the right and bottom
//! edges are part of the grid: they are read with a background colour and
//! only their in-slide part is scored and drawn (VAE-17).

use eov_plugin_api::protocol::{FileInfo, Rect};
use serde::{Deserialize, Serialize};

/// One pyramid level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub index: u32,
    pub width: u64,
    pub height: u64,
    /// Level-0 size divided by this level's size.
    pub downsample: f64,
}

/// Geometry and calibration of a slide.
#[derive(Clone, Debug, PartialEq)]
pub struct SlideGeometry {
    pub width: u64,
    pub height: u64,
    /// Levels by increasing downsample.
    pub levels: Vec<Level>,
    /// Average µm per level-0 pixel, when the slide is calibrated.
    pub mpp: Option<f64>,
}

fn valid_mpp(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0)
}

impl SlideGeometry {
    pub fn from_file_info(file: &FileInfo) -> Self {
        let mut levels: Vec<Level> = file
            .levels
            .iter()
            .filter(|level| level.width > 0 && level.height > 0)
            .map(|level| {
                let derived = file.width as f64 / level.width as f64;
                let downsample = if level.downsample.is_finite() && level.downsample > 0.0 {
                    level.downsample
                } else {
                    derived
                };
                Level {
                    index: level.level,
                    width: level.width,
                    height: level.height,
                    downsample: downsample.max(1e-6),
                }
            })
            .collect();
        levels.sort_by(|a, b| a.downsample.total_cmp(&b.downsample));
        levels.dedup_by_key(|level| level.index);
        if levels.is_empty() && file.width > 0 && file.height > 0 {
            levels.push(Level {
                index: 0,
                width: file.width,
                height: file.height,
                downsample: 1.0,
            });
        }
        let mpp = match (valid_mpp(file.mpp_x), valid_mpp(file.mpp_y)) {
            (Some(x), Some(y)) => Some((x + y) / 2.0),
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        };
        Self {
            width: file.width,
            height: file.height,
            levels,
            mpp,
        }
    }

    pub fn level(&self, index: u32) -> Option<&Level> {
        self.levels.iter().find(|level| level.index == index)
    }

    /// µm per pixel of `level`, if the slide is calibrated.
    pub fn level_mpp(&self, level: &Level) -> Option<f64> {
        self.mpp.map(|mpp| mpp * level.downsample)
    }
}

/// The resolution the user asked to analyse at.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Resolution {
    /// The level closest to the model's training resolution when both the
    /// slide and the model declare one; level 0 otherwise.
    #[default]
    Auto,
    /// The level whose downsample is closest to `value`.
    Downsample { value: f64 },
}

/// The level an analysis runs on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelChoice {
    pub level: Level,
    /// µm per pixel of the chosen level, if calibrated.
    pub mpp: Option<f64>,
    /// Chosen level's µm/px divided by the model's training µm/px.
    pub mismatch: Option<f64>,
}

impl LevelChoice {
    /// Whether the slide resolution is far enough from the model's training
    /// resolution to distort errors (more than 25 % off).
    pub fn mismatched(&self) -> bool {
        self.mismatch
            .is_some_and(|ratio| !(0.8..=1.25).contains(&ratio))
    }
}

fn log_distance(a: f64, b: f64) -> f64 {
    (a.max(1e-12).ln() - b.max(1e-12).ln()).abs()
}

/// Picks the level for `resolution`. Returns `None` for slides without
/// levels.
pub fn choose_level(
    geometry: &SlideGeometry,
    resolution: Resolution,
    model_mpp: Option<f64>,
) -> Option<LevelChoice> {
    let first = *geometry.levels.first()?;
    let level = match (resolution, geometry.mpp, valid_mpp(model_mpp)) {
        (Resolution::Downsample { value }, _, _) if value.is_finite() && value > 0.0 => geometry
            .levels
            .iter()
            .copied()
            .min_by(|a, b| {
                log_distance(a.downsample, value).total_cmp(&log_distance(b.downsample, value))
            })
            .unwrap_or(first),
        (_, Some(slide_mpp), Some(target)) => geometry
            .levels
            .iter()
            .copied()
            .min_by(|a, b| {
                log_distance(slide_mpp * a.downsample, target)
                    .total_cmp(&log_distance(slide_mpp * b.downsample, target))
            })
            .unwrap_or(first),
        _ => first,
    };
    let mpp = geometry.level_mpp(&level);
    let mismatch = match (mpp, valid_mpp(model_mpp)) {
        (Some(level_mpp), Some(target)) => Some(level_mpp / target),
        _ => None,
    };
    Some(LevelChoice {
        level,
        mpp,
        mismatch,
    })
}

/// A tile position in a grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub col: u32,
    pub row: u32,
}

impl TileCoord {
    pub const fn new(col: u32, row: u32) -> Self {
        Self { col, row }
    }

    /// A compact integer key (row-major order).
    pub const fn key(self) -> u64 {
        ((self.row as u64) << 32) | self.col as u64
    }

    pub const fn from_key(key: u64) -> Self {
        Self {
            col: key as u32,
            row: (key >> 32) as u32,
        }
    }

    /// A short, stable id for menus and lists.
    pub fn id(self) -> String {
        format!("{}_{}", self.col, self.row)
    }

    pub fn parse_id(id: &str) -> Option<Self> {
        let (col, row) = id.split_once('_')?;
        Some(Self::new(col.parse().ok()?, row.parse().ok()?))
    }
}

/// A rectangle of tiles, `col0..col1` × `row0..row1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TileRange {
    pub col0: u32,
    pub col1: u32,
    pub row0: u32,
    pub row1: u32,
}

impl TileRange {
    pub fn len(&self) -> u64 {
        u64::from(self.col1.saturating_sub(self.col0))
            * u64::from(self.row1.saturating_sub(self.row0))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, coord: TileCoord) -> bool {
        (self.col0..self.col1).contains(&coord.col) && (self.row0..self.row1).contains(&coord.row)
    }

    /// Row-major iteration.
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        (self.row0..self.row1)
            .flat_map(move |row| (self.col0..self.col1).map(move |col| TileCoord::new(col, row)))
    }

    /// Every tile, nearest to `center` (in tile units) first, so the middle
    /// of the view is analysed before its edges (VAE-10).
    pub fn center_out(&self, center: (f64, f64)) -> Vec<TileCoord> {
        let mut tiles: Vec<TileCoord> = self.iter().collect();
        let distance = |c: &TileCoord| {
            let dx = f64::from(c.col) + 0.5 - center.0;
            let dy = f64::from(c.row) + 0.5 - center.1;
            dx * dx + dy * dy
        };
        tiles.sort_by(|a, b| {
            distance(a)
                .total_cmp(&distance(b))
                .then(a.key().cmp(&b.key()))
        });
        tiles
    }
}

/// A grid of square tiles of `tile` level pixels on one level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileGrid {
    pub level: u32,
    pub downsample: f64,
    /// Tile edge in pixels of `level` (the model's input size).
    pub tile: u32,
    pub cols: u32,
    pub rows: u32,
    pub level_width: u64,
    pub level_height: u64,
    pub width0: u64,
    pub height0: u64,
}

impl TileGrid {
    pub fn new(geometry: &SlideGeometry, level: &Level, tile: u32) -> Self {
        let tile = tile.max(1);
        let cols = level
            .width
            .div_ceil(u64::from(tile))
            .clamp(1, u64::from(u32::MAX)) as u32;
        let rows = level
            .height
            .div_ceil(u64::from(tile))
            .clamp(1, u64::from(u32::MAX)) as u32;
        Self {
            level: level.index,
            downsample: level.downsample,
            tile,
            cols,
            rows,
            level_width: level.width,
            level_height: level.height,
            width0: geometry.width,
            height0: geometry.height,
        }
    }

    /// Level-0 extent of one tile.
    pub fn step0(&self) -> f64 {
        f64::from(self.tile) * self.downsample
    }

    pub fn tile_count(&self) -> u64 {
        u64::from(self.cols) * u64::from(self.rows)
    }

    pub fn contains(&self, coord: TileCoord) -> bool {
        coord.col < self.cols && coord.row < self.rows
    }

    pub fn full_range(&self) -> TileRange {
        TileRange {
            col0: 0,
            col1: self.cols,
            row0: 0,
            row1: self.rows,
        }
    }

    /// Level-0 corner to read `coord` from. Rounded up so that OpenSlide's
    /// division by the (non-integer) downsample never lands one level pixel
    /// short of the tile's first pixel.
    pub fn origin0(&self, coord: TileCoord) -> (i64, i64) {
        let at = |index: u32| {
            let exact = f64::from(index) * f64::from(self.tile) * self.downsample;
            (exact - 1e-6).ceil().max(0.0) as i64
        };
        (at(coord.col), at(coord.row))
    }

    /// The in-slide part of a tile, in level pixels (at least 1×1).
    pub fn valid_size(&self, coord: TileCoord) -> (u32, u32) {
        let span = |index: u32, extent: u64| {
            let start = u64::from(index) * u64::from(self.tile);
            extent.saturating_sub(start).clamp(1, u64::from(self.tile)) as u32
        };
        (
            span(coord.col, self.level_width),
            span(coord.row, self.level_height),
        )
    }

    /// The level-0 rectangle the whole tile covers.
    pub fn footprint(&self, coord: TileCoord) -> Rect {
        let step = self.step0();
        Rect {
            x: f64::from(coord.col) * step,
            y: f64::from(coord.row) * step,
            width: step,
            height: step,
        }
    }

    /// The level-0 rectangle of the tile's in-slide part.
    pub fn valid_footprint(&self, coord: TileCoord) -> Rect {
        let full = self.footprint(coord);
        let (vw, vh) = self.valid_size(coord);
        let width = (f64::from(vw) * self.downsample).min(self.width0 as f64 - full.x);
        let height = (f64::from(vh) * self.downsample).min(self.height0 as f64 - full.y);
        Rect {
            width: width.max(1.0),
            height: height.max(1.0),
            ..full
        }
    }

    /// Tiles intersecting a level-0 rectangle.
    pub fn range_in(&self, rect: &Rect) -> TileRange {
        if !(rect.x.is_finite() && rect.y.is_finite() && rect.width.is_finite())
            || !rect.height.is_finite()
        {
            return TileRange::default();
        }
        let step = self.step0();
        let index =
            |value: f64, limit: u32| (value / step).floor().clamp(0.0, f64::from(limit)) as u32;
        let end =
            |value: f64, limit: u32| (value / step).ceil().clamp(0.0, f64::from(limit)) as u32;
        TileRange {
            col0: index(rect.x, self.cols),
            col1: end(rect.x + rect.width.max(0.0), self.cols),
            row0: index(rect.y, self.rows),
            row1: end(rect.y + rect.height.max(0.0), self.rows),
        }
    }

    /// The tile containing a level-0 point, if any.
    pub fn tile_at(&self, x: f64, y: f64) -> Option<TileCoord> {
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return None;
        }
        let step = self.step0();
        let coord = TileCoord::new((x / step) as u32, (y / step) as u32);
        self.contains(coord).then_some(coord)
    }

    /// Tile position (fractional, in tile units) of a level-0 point.
    pub fn tile_units(&self, x: f64, y: f64) -> (f64, f64) {
        let step = self.step0();
        (x / step, y / step)
    }

    /// Screen pixels one tile spans at `zoom` (screen px per level-0 px).
    pub fn screen_size(&self, zoom: f64) -> f64 {
        self.step0() * zoom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::LevelInfo;

    /// The level geometry of the fixture `C3L-00088-22.svs` (VAE-3).
    pub(crate) fn aperio_like() -> FileInfo {
        FileInfo {
            file_id: 1,
            width: 100_000,
            height: 80_000,
            levels: vec![
                LevelInfo {
                    level: 0,
                    width: 100_000,
                    height: 80_000,
                    downsample: 1.0,
                },
                LevelInfo {
                    level: 1,
                    width: 25_000,
                    height: 20_000,
                    downsample: 4.0,
                },
                LevelInfo {
                    level: 2,
                    width: 6_249,
                    height: 4_999,
                    downsample: 16.004,
                },
            ],
            mpp_x: Some(0.494),
            mpp_y: Some(0.494),
            ..Default::default()
        }
    }

    #[test]
    fn uses_the_reported_downsamples() {
        let geometry = SlideGeometry::from_file_info(&aperio_like());
        let choice = choose_level(&geometry, Resolution::Downsample { value: 2.0 }, None).unwrap();
        // 2x does not exist: the closest real level is 1x or 4x, never a
        // fake 2x read of level 1.
        assert!(choice.level.index == 0 || choice.level.index == 1);
        let choice = choose_level(&geometry, Resolution::Downsample { value: 8.0 }, None).unwrap();
        assert_eq!(choice.level.index, 1);
        let choice = choose_level(&geometry, Resolution::Downsample { value: 16.0 }, None).unwrap();
        assert_eq!(choice.level.index, 2);
        assert!((choice.level.downsample - 16.004).abs() < 1e-9);
        let grid = TileGrid::new(&geometry, &choice.level, 256);
        assert!((grid.step0() - 256.0 * 16.004).abs() < 1e-9);
    }

    #[test]
    fn auto_matches_the_model_resolution_and_flags_mismatches() {
        let geometry = SlideGeometry::from_file_info(&aperio_like());
        // A model trained at 2 µm/px picks the 4x level (1.976 µm/px).
        let choice = choose_level(&geometry, Resolution::Auto, Some(2.0)).unwrap();
        assert_eq!(choice.level.index, 1);
        assert!(!choice.mismatched());
        // HistoVAE (0.227 µm/px) on a 0.494 µm/px slide: level 0, flagged.
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.227)).unwrap();
        assert_eq!(choice.level.index, 0);
        assert!(choice.mismatched());
        // Uncalibrated: level 0, nothing to compare.
        let mut file = aperio_like();
        file.mpp_x = None;
        file.mpp_y = None;
        let geometry = SlideGeometry::from_file_info(&file);
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.25)).unwrap();
        assert_eq!(choice.level.index, 0);
        assert_eq!(choice.mismatch, None);
    }

    #[test]
    fn missing_or_bad_level_info_is_repaired() {
        let file = FileInfo {
            width: 1000,
            height: 500,
            levels: vec![
                LevelInfo {
                    level: 0,
                    width: 1000,
                    height: 500,
                    downsample: f64::NAN,
                },
                LevelInfo {
                    level: 1,
                    width: 250,
                    height: 125,
                    downsample: 0.0,
                },
            ],
            ..Default::default()
        };
        let geometry = SlideGeometry::from_file_info(&file);
        assert_eq!(geometry.levels[0].downsample, 1.0);
        assert_eq!(geometry.levels[1].downsample, 4.0);
        let empty = SlideGeometry::from_file_info(&FileInfo {
            width: 10,
            height: 10,
            ..Default::default()
        });
        assert_eq!(empty.levels.len(), 1);
    }

    #[test]
    fn edge_tiles_are_part_of_the_grid() {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width: 700,
            height: 300,
            ..Default::default()
        });
        let grid = TileGrid::new(&geometry, &geometry.levels[0], 256);
        assert_eq!((grid.cols, grid.rows), (3, 2));
        let edge = TileCoord::new(2, 1);
        assert_eq!(grid.valid_size(edge), (700 - 512, 300 - 256));
        let rect = grid.valid_footprint(edge);
        assert_eq!(
            (rect.x, rect.y, rect.width, rect.height),
            (512.0, 256.0, 188.0, 44.0)
        );
        // An image smaller than one tile still has one (partial) tile.
        let tiny = SlideGeometry::from_file_info(&FileInfo {
            width: 100,
            height: 50,
            ..Default::default()
        });
        let grid = TileGrid::new(&tiny, &tiny.levels[0], 256);
        assert_eq!(grid.tile_count(), 1);
        assert_eq!(grid.valid_size(TileCoord::new(0, 0)), (100, 50));
    }

    #[test]
    fn origins_never_round_below_the_level_pixel() {
        let geometry = SlideGeometry::from_file_info(&aperio_like());
        let level = geometry.levels[2];
        let grid = TileGrid::new(&geometry, &level, 256);
        for col in 0..grid.cols {
            let (x, _) = grid.origin0(TileCoord::new(col, 0));
            // OpenSlide maps level-0 x to level pixel floor(x / downsample).
            let level_x = (x as f64 / level.downsample).floor() as u64;
            assert_eq!(level_x, u64::from(col) * 256, "col {col}");
        }
    }

    #[test]
    fn ranges_cover_rectangles_and_order_center_out() {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width: 2560,
            height: 2560,
            ..Default::default()
        });
        let grid = TileGrid::new(&geometry, &geometry.levels[0], 256);
        let range = grid.range_in(&Rect {
            x: 300.0,
            y: -50.0,
            width: 600.0,
            height: 300.0,
        });
        assert_eq!(
            range,
            TileRange {
                col0: 1,
                col1: 4,
                row0: 0,
                row1: 1
            }
        );
        assert_eq!(range.len(), 3);
        let full = grid.full_range();
        let ordered = full.center_out((5.0, 5.0));
        assert_eq!(ordered.len(), 100);
        let first = ordered[0];
        assert!((4..=5).contains(&first.col) && (4..=5).contains(&first.row));
        let last = *ordered.last().unwrap();
        assert!(last.col == 0 || last.col == 9);
        assert_eq!(grid.tile_at(300.0, 600.0), Some(TileCoord::new(1, 2)));
        assert_eq!(grid.tile_at(-1.0, 0.0), None);
    }

    #[test]
    fn coord_keys_and_ids_round_trip() {
        let coord = TileCoord::new(123, 456);
        assert_eq!(TileCoord::from_key(coord.key()), coord);
        assert_eq!(TileCoord::parse_id(&coord.id()), Some(coord));
        assert_eq!(TileCoord::parse_id("x_1"), None);
    }
}
