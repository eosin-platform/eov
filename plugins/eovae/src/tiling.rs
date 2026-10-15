//! Tile grids on the slide's real pyramid levels.
//!
//! Pyramids do not always halve per level: Aperio slides commonly go 1, 4,
//! 16, and levels can be slightly off (8.002). Every grid here is built from
//! a level's reported downsample (VAE-3), and tiles at the right and bottom
//! edges are part of the grid: they are read with a background colour and
//! only their in-slide part is scored and drawn (VAE-17).
//!
//! When the slide and the model both declare their µm per pixel, tiles have
//! the model's training pixel size exactly (VAE-15): each one covers
//! `tile × model µm/px` per axis, is read from the coarsest level that is
//! still at least as fine, and is resampled to the model's pixels (see
//! [`crate::resample`]); non-square slide pixels are handled per axis.

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
    /// µm per level-0 pixel along x and y (one given value is used for
    /// both).
    pub mpp_xy: Option<(f64, f64)>,
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
        let mpp_xy = match (valid_mpp(file.mpp_x), valid_mpp(file.mpp_y)) {
            (Some(x), Some(y)) => Some((x, y)),
            (Some(v), None) | (None, Some(v)) => Some((v, v)),
            (None, None) => None,
        };
        Self {
            width: file.width,
            height: file.height,
            levels,
            mpp: mpp_xy.map(|(x, y)| (x + y) / 2.0),
            mpp_xy,
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
    /// The model's training µm/px when both the slide and the model declare
    /// one (tiles are resampled to it); level 0 otherwise.
    #[default]
    Auto,
    /// The level whose downsample is closest to `value`, read as it is.
    Downsample { value: f64 },
}

/// How an analysis reads the slide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelChoice {
    /// The level tiles are read from.
    pub level: Level,
    /// µm per pixel fed to the model, if calibrated: the model's training
    /// µm/px when resampled, else the level's.
    pub mpp: Option<f64>,
    /// The detail fed to the model against the model's training µm/px: the
    /// level's µm/px divided by it, or for resampled tiles, how much coarser
    /// the source is (1 when it is fine enough).
    pub mismatch: Option<f64>,
    /// Level-0 pixels per model pixel along x and y when tiles are
    /// resampled to the model's µm/px; `None`: the level's own pixels.
    pub resample: Option<(f64, f64)>,
}

impl LevelChoice {
    /// Whether the input differs enough from the model's training resolution
    /// to distort errors: a level more than 25 % off, or a slide whose
    /// finest level is more than 10 % coarser than the model's pixels.
    pub fn mismatched(&self) -> bool {
        match (self.mismatch, self.resample) {
            (Some(ratio), Some(_)) => ratio > 1.1,
            (Some(ratio), None) => !(0.8..=1.25).contains(&ratio),
            (None, _) => false,
        }
    }

    /// µm/px of the level read.
    pub fn level_mpp(&self, geometry: &SlideGeometry) -> Option<f64> {
        geometry.level_mpp(&self.level)
    }
}

fn log_distance(a: f64, b: f64) -> f64 {
    (a.max(1e-12).ln() - b.max(1e-12).ln()).abs()
}

/// Picks how to read the slide for `resolution`. Returns `None` for slides
/// without levels.
///
/// `Auto` with a calibrated slide and model resamples to the model's µm/px,
/// reading the coarsest level that is still at least as fine on both axes
/// (level 0, upsampled, when even that is coarser); a level within 0.5 % of
/// the model's pixel size is read as it is. An explicit downsample reads
/// that level's own pixels.
pub fn choose_level(
    geometry: &SlideGeometry,
    resolution: Resolution,
    model_mpp: Option<f64>,
) -> Option<LevelChoice> {
    let first = *geometry.levels.first()?;
    let target = valid_mpp(model_mpp);
    let native = |level: Level| {
        let mpp = geometry.level_mpp(&level);
        LevelChoice {
            level,
            mpp,
            mismatch: mpp.zip(target).map(|(mpp, target)| mpp / target),
            resample: None,
        }
    };
    match (resolution, geometry.mpp_xy, target) {
        (Resolution::Downsample { value }, _, _) if value.is_finite() && value > 0.0 => {
            let level = geometry
                .levels
                .iter()
                .copied()
                .min_by(|a, b| {
                    log_distance(a.downsample, value).total_cmp(&log_distance(b.downsample, value))
                })
                .unwrap_or(first);
            Some(native(level))
        }
        (Resolution::Auto, Some((mpp_x, mpp_y)), Some(target)) => {
            let coarser = mpp_x.max(mpp_y);
            let level = geometry
                .levels
                .iter()
                .copied()
                .rfind(|level| level.downsample * coarser <= target * (1.0 + 1e-3))
                .unwrap_or(first);
            let (scale_x, scale_y) = (target / mpp_x, target / mpp_y);
            let same = |scale: f64| (scale / level.downsample - 1.0).abs() < 0.005;
            if same(scale_x) && same(scale_y) {
                return Some(native(level));
            }
            Some(LevelChoice {
                level,
                mpp: Some(target),
                mismatch: Some((level.downsample * coarser / target).max(1.0)),
                resample: Some((scale_x, scale_y)),
            })
        }
        _ => Some(native(first)),
    }
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

/// How to read one tile: the region to ask the host for, and where the
/// tile's model pixels lie in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadPlan {
    /// Level-0 corner of the region.
    pub x: i64,
    pub y: i64,
    /// The level row of the region's first row.
    pub row: i64,
    /// Its size in pixels of the level read.
    pub width: u32,
    pub height: u32,
    /// The tile's left and top edge, in pixels of the region.
    pub origin_x: f64,
    pub origin_y: f64,
    /// Region pixels per model pixel.
    pub step_x: f64,
    pub step_y: f64,
}

impl ReadPlan {
    /// The level-0 `y` to read the region from its row `offset` on
    /// (OpenSlide maps it back to level row `row + offset`), for reads in
    /// strips.
    pub fn row_y(&self, offset: u32, downsample: f64) -> i64 {
        if offset == 0 {
            return self.y;
        }
        ((self.row + i64::from(offset)) as f64 * downsample - 1e-6)
            .ceil()
            .max(0.0) as i64
    }
}

/// A grid of tiles of `tile`×`tile` model pixels, read from one level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileGrid {
    /// The level read.
    pub level: u32,
    pub downsample: f64,
    /// Tile edge in model pixels (the model's input size).
    pub tile: u32,
    pub cols: u32,
    pub rows: u32,
    pub level_width: u64,
    pub level_height: u64,
    pub width0: u64,
    pub height0: u64,
    /// Level-0 pixels per model pixel along x and y: the level's downsample
    /// when tiles are its own pixels, the model's µm/px over the slide's
    /// when they are resampled.
    pub scale_x: f64,
    pub scale_y: f64,
}

impl TileGrid {
    /// Tiles of the level's own pixels.
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
            scale_x: level.downsample,
            scale_y: level.downsample,
        }
    }

    /// The grid `choice` asks for: resampled to the model's pixels, or the
    /// level's own.
    pub fn for_choice(geometry: &SlideGeometry, choice: &LevelChoice, tile: u32) -> Self {
        let native = Self::new(geometry, &choice.level, tile);
        let Some((scale_x, scale_y)) = choice.resample else {
            return native;
        };
        let count = |extent: u64, scale: f64| {
            (extent as f64 / (f64::from(native.tile) * scale))
                .ceil()
                .clamp(1.0, f64::from(u32::MAX)) as u32
        };
        Self {
            cols: count(geometry.width, scale_x),
            rows: count(geometry.height, scale_y),
            scale_x,
            scale_y,
            ..native
        }
    }

    /// Whether tiles are resampled from the level (not its own pixels).
    pub fn resampled(&self) -> bool {
        self.scale_x != self.downsample || self.scale_y != self.downsample
    }

    /// Level-0 width of one tile.
    pub fn step_x(&self) -> f64 {
        f64::from(self.tile) * self.scale_x
    }

    /// Level-0 height of one tile.
    pub fn step_y(&self) -> f64 {
        f64::from(self.tile) * self.scale_y
    }

    /// The smaller level-0 edge of a tile.
    pub fn step_min(&self) -> f64 {
        self.step_x().min(self.step_y())
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

    /// Level-0 corner to read `coord` from, for tiles of the level's own
    /// pixels. Rounded up so that OpenSlide's division by the (non-integer)
    /// downsample never lands one level pixel short of the tile's first
    /// pixel.
    pub fn origin0(&self, coord: TileCoord) -> (i64, i64) {
        let at = |index: u32| {
            let exact = f64::from(index) * f64::from(self.tile) * self.downsample;
            (exact - 1e-6).ceil().max(0.0) as i64
        };
        (at(coord.col), at(coord.row))
    }

    /// The region to read for `coord`. Resampled tiles read the level pixels
    /// they cover (one more on each side when upsampling, for bilinear
    /// interpolation); parts outside the slide come back as background.
    pub fn read_plan(&self, coord: TileCoord) -> ReadPlan {
        if !self.resampled() {
            let (x, y) = self.origin0(coord);
            return ReadPlan {
                x,
                y,
                row: i64::from(coord.row) * i64::from(self.tile),
                width: self.tile,
                height: self.tile,
                origin_x: 0.0,
                origin_y: 0.0,
                step_x: 1.0,
                step_y: 1.0,
            };
        }
        let tile = f64::from(self.tile);
        let axis = |index: u32, scale: f64| {
            let step = scale / self.downsample;
            let start = f64::from(index) * tile * step;
            let margin = if step < 1.0 { 1.0 } else { 0.0 };
            let first = (start - margin).floor().max(0.0);
            let end = (start + tile * step + margin).ceil();
            let count = (end - first).max(1.0) as u32;
            // The level-0 coordinate OpenSlide maps back to level pixel
            // `first` (see `origin0`).
            let corner = (first * self.downsample - 1e-6).ceil().max(0.0) as i64;
            (corner, count, start - first, step, first as i64)
        };
        let (x, width, origin_x, step_x, _) = axis(coord.col, self.scale_x);
        let (y, height, origin_y, step_y, row) = axis(coord.row, self.scale_y);
        ReadPlan {
            x,
            y,
            row,
            width,
            height,
            origin_x,
            origin_y,
            step_x,
            step_y,
        }
    }

    /// The in-slide part of a tile, in model pixels (at least 1×1).
    pub fn valid_size(&self, coord: TileCoord) -> (u32, u32) {
        if !self.resampled() {
            let span = |index: u32, extent: u64| {
                let start = u64::from(index) * u64::from(self.tile);
                extent.saturating_sub(start).clamp(1, u64::from(self.tile)) as u32
            };
            return (
                span(coord.col, self.level_width),
                span(coord.row, self.level_height),
            );
        }
        let span = |index: u32, extent: u64, scale: f64| {
            let start = f64::from(index) * f64::from(self.tile) * scale;
            ((extent as f64 - start) / scale)
                .ceil()
                .clamp(1.0, f64::from(self.tile)) as u32
        };
        (
            span(coord.col, self.width0, self.scale_x),
            span(coord.row, self.height0, self.scale_y),
        )
    }

    /// The level-0 rectangle the whole tile covers.
    pub fn footprint(&self, coord: TileCoord) -> Rect {
        let (step_x, step_y) = (self.step_x(), self.step_y());
        Rect {
            x: f64::from(coord.col) * step_x,
            y: f64::from(coord.row) * step_y,
            width: step_x,
            height: step_y,
        }
    }

    /// The level-0 rectangle of the tile's in-slide part.
    pub fn valid_footprint(&self, coord: TileCoord) -> Rect {
        let full = self.footprint(coord);
        let (vw, vh) = self.valid_size(coord);
        let width = (f64::from(vw) * self.scale_x).min(self.width0 as f64 - full.x);
        let height = (f64::from(vh) * self.scale_y).min(self.height0 as f64 - full.y);
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
        let index = |value: f64, step: f64, limit: u32| {
            (value / step).floor().clamp(0.0, f64::from(limit)) as u32
        };
        let end = |value: f64, step: f64, limit: u32| {
            (value / step).ceil().clamp(0.0, f64::from(limit)) as u32
        };
        let (step_x, step_y) = (self.step_x(), self.step_y());
        TileRange {
            col0: index(rect.x, step_x, self.cols),
            col1: end(rect.x + rect.width.max(0.0), step_x, self.cols),
            row0: index(rect.y, step_y, self.rows),
            row1: end(rect.y + rect.height.max(0.0), step_y, self.rows),
        }
    }

    /// The tile containing a level-0 point, if any.
    pub fn tile_at(&self, x: f64, y: f64) -> Option<TileCoord> {
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return None;
        }
        let coord = TileCoord::new((x / self.step_x()) as u32, (y / self.step_y()) as u32);
        self.contains(coord).then_some(coord)
    }

    /// Tile position (fractional, in tile units) of a level-0 point.
    pub fn tile_units(&self, x: f64, y: f64) -> (f64, f64) {
        (x / self.step_x(), y / self.step_y())
    }

    /// Screen pixels one tile spans at `zoom` (screen px per level-0 px),
    /// along its shorter edge.
    pub fn screen_size(&self, zoom: f64) -> f64 {
        self.step_min() * zoom
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
        assert!((grid.step_x() - 256.0 * 16.004).abs() < 1e-9);
        assert!(!grid.resampled());
    }

    #[test]
    fn auto_resamples_to_the_model_resolution_and_flags_slides_that_cannot_reach_it() {
        let geometry = SlideGeometry::from_file_info(&aperio_like());
        // A model trained at 2 µm/px: the coarsest level at least as fine is
        // the 4x level (1.976 µm/px), resampled to 2 µm/px.
        let choice = choose_level(&geometry, Resolution::Auto, Some(2.0)).unwrap();
        assert_eq!(choice.level.index, 1);
        let (scale_x, scale_y) = choice.resample.unwrap();
        assert!((scale_x - 2.0 / 0.494).abs() < 1e-9 && scale_x == scale_y);
        assert_eq!(choice.mpp, Some(2.0));
        assert!(!choice.mismatched());
        // At 8 µm/px the 16.004x level (7.906 µm/px) is still finer.
        let choice = choose_level(&geometry, Resolution::Auto, Some(8.0)).unwrap();
        assert_eq!(choice.level.index, 2);
        // A level within 0.5 % of the model is read as it is.
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.494 * 4.0)).unwrap();
        assert_eq!((choice.level.index, choice.resample), (1, None));
        // HistoVAE (0.227 µm/px) on a 0.494 µm/px slide: level 0 upsampled,
        // flagged; 5 % coarser is not.
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.227)).unwrap();
        assert_eq!(choice.level.index, 0);
        assert!(choice.resample.is_some() && choice.mismatched());
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.494 / 1.05)).unwrap();
        assert!(choice.resample.is_some() && !choice.mismatched());
        // An explicit level is read as it is, flagged when far off.
        let choice =
            choose_level(&geometry, Resolution::Downsample { value: 1.0 }, Some(2.0)).unwrap();
        assert_eq!((choice.level.index, choice.resample), (0, None));
        assert!(choice.mismatched());
        // Uncalibrated: level 0, nothing to compare.
        let mut file = aperio_like();
        file.mpp_x = None;
        file.mpp_y = None;
        let geometry = SlideGeometry::from_file_info(&file);
        let choice = choose_level(&geometry, Resolution::Auto, Some(0.25)).unwrap();
        assert_eq!((choice.level.index, choice.resample), (0, None));
        assert_eq!(choice.mismatch, None);
    }

    /// A model at 0.8 µm/px: windows of 32 model pixels cover 25.6 µm on
    /// every slide, whatever its pixel size, also when pixels are not
    /// square.
    #[test]
    fn resampled_windows_cover_the_same_physical_extent_on_every_slide() {
        let tile = 32;
        let target = 0.8;
        for (mpp_x, mpp_y) in [(0.25, 0.25), (0.5, 0.5), (0.3, 0.45), (0.9, 0.9)] {
            let mut file = aperio_like();
            file.mpp_x = Some(mpp_x);
            file.mpp_y = Some(mpp_y);
            let geometry = SlideGeometry::from_file_info(&file);
            let choice = choose_level(&geometry, Resolution::Auto, Some(target)).unwrap();
            let grid = TileGrid::for_choice(&geometry, &choice, tile);
            assert!(grid.resampled(), "{mpp_x}/{mpp_y}");
            let um_x = grid.step_x() * mpp_x;
            let um_y = grid.step_y() * mpp_y;
            assert!((um_x - 25.6).abs() < 1e-9 && (um_y - 25.6).abs() < 1e-9);
            // Read from a level at least as fine as the model (except the
            // 0.9 µm/px slide, which is upsampled from level 0).
            let level_mpp = choice.level.downsample * mpp_x.max(mpp_y);
            assert!(level_mpp <= target * 1.001 || choice.level.index == 0);
            // The region read covers the tile, in the level's pixels.
            let coord = TileCoord::new(3, 2);
            let plan = grid.read_plan(coord);
            let footprint = grid.footprint(coord);
            // OpenSlide reads from level pixel floor(x / downsample).
            let level_x = (plan.x as f64 / grid.downsample).floor();
            assert!(level_x <= footprint.x / grid.downsample + 1e-6);
            let read_right = level_x + f64::from(plan.width);
            assert!(read_right >= (footprint.x + footprint.width) / grid.downsample - 1e-6);
            assert!(
                (plan.step_x * f64::from(tile) * grid.downsample - footprint.width).abs() < 1e-6
            );
            assert!(
                (plan.step_y * f64::from(tile) * grid.downsample - footprint.height).abs() < 1e-6
            );
            // Strips start at the region's rows.
            let strip = (plan.row_y(7, grid.downsample) as f64 / grid.downsample).floor();
            assert_eq!(strip as i64, plan.row + 7);
            // Tiles still cover the whole slide.
            assert!(f64::from(grid.cols) * grid.step_x() >= file.width as f64);
            assert!(f64::from(grid.cols - 1) * grid.step_x() < file.width as f64);
        }
    }

    #[test]
    fn resampled_edge_tiles_score_only_their_in_slide_part() {
        let mut file = aperio_like();
        file.width = 1000;
        file.height = 1000;
        file.levels.truncate(1);
        file.levels[0].width = 1000;
        file.levels[0].height = 1000;
        file.mpp_x = Some(0.25);
        file.mpp_y = Some(0.25);
        let geometry = SlideGeometry::from_file_info(&file);
        let choice = choose_level(&geometry, Resolution::Auto, Some(1.0)).unwrap();
        let grid = TileGrid::for_choice(&geometry, &choice, 64);
        // 64 model px of 1 µm = 256 level-0 px: 4 columns, the last one
        // 1000 - 768 = 232 px = 58 model px wide.
        assert_eq!((grid.cols, grid.rows), (4, 4));
        assert_eq!(grid.valid_size(TileCoord::new(3, 0)), (58, 64));
        let rect = grid.valid_footprint(TileCoord::new(3, 3));
        assert_eq!((rect.x, rect.width), (768.0, 232.0));
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
