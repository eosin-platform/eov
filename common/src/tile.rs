//! Tile management for WSI rendering
//!
//! This module handles tile coordinates, tile requests, and tile data management
//! for efficient rendering of whole slide images.

use crate::wsi::WsiLevel;
use crate::{Error, Result, WsiFile};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, trace, warn};

/// Most tiles [`TileManager::visible_tiles_with_margin`] returns for one
/// level. Larger visible sets keep the tiles nearest the view centre.
pub const MAX_VISIBLE_TILES: usize = 4096;

/// Border (in pixels of the tile's level) that tiles carry by default. One
/// pixel is enough for seamless bilinear and trilinear sampling.
pub const DEFAULT_TILE_BORDER: u32 = 1;

/// Border needed for seamless Lanczos-3 magnification (three taps on each
/// side of the sample).
pub const LANCZOS_TILE_BORDER: u32 = 3;

/// Largest border [`TileManager`] accepts.
pub const MAX_TILE_BORDER: u32 = 64;

/// Minimum spacing of "too many visible tiles" warnings.
const CAP_WARN_INTERVAL_MS: u64 = 10_000;

fn millis_since_start() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Describes the expanded read region and how much valid border was available.
struct ReadRegion<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    avail_left: u32,
    avail_top: u32,
}

/// Build a uniformly-bordered RGBA buffer from an expanded-region read.
///
/// The read data has variable border amounts on each side (limited by image
/// bounds).  This function copies it into a buffer that always has exactly
/// `border` pixels on every side, edge-clamping where the image boundary
/// prevented reading real neighbor data.
fn create_bordered_tile(read: ReadRegion, inner_w: u32, inner_h: u32, border: u32) -> Vec<u8> {
    let read_data = read.data;
    let read_w = read.width;
    let read_h = read.height;
    let avail_left = read.avail_left;
    let avail_top = read.avail_top;
    let out_w = (inner_w + 2 * border) as usize;
    let out_h = (inner_h + 2 * border) as usize;
    let dst_stride = out_w * 4;
    let src_stride = read_w as usize * 4;
    let mut out = vec![0u8; out_h * dst_stride];

    // Offset in the output where the read data is placed.
    let dx = (border - avail_left) as usize;
    let dy = (border - avail_top) as usize;

    // Step 1: copy the read data into the output buffer.
    for y in 0..read_h as usize {
        let src_off = y * src_stride;
        let dst_off = (dy + y) * dst_stride + dx * 4;
        out[dst_off..dst_off + src_stride]
            .copy_from_slice(&read_data[src_off..src_off + src_stride]);
    }

    // Step 2: fill missing left-border columns by replicating the first
    // available column in each row that has data.
    if dx > 0 {
        for iy in dy..dy + read_h as usize {
            let src_off = iy * dst_stride + dx * 4;
            let pixel = [
                out[src_off],
                out[src_off + 1],
                out[src_off + 2],
                out[src_off + 3],
            ];
            for ix in 0..dx {
                let off = iy * dst_stride + ix * 4;
                out[off..off + 4].copy_from_slice(&pixel);
            }
        }
    }

    // Step 3: fill missing right-border columns.
    let right_data_end = dx + read_w as usize;
    if right_data_end < out_w {
        for iy in dy..dy + read_h as usize {
            let src_off = iy * dst_stride + (right_data_end - 1) * 4;
            let pixel = [
                out[src_off],
                out[src_off + 1],
                out[src_off + 2],
                out[src_off + 3],
            ];
            for ix in right_data_end..out_w {
                let off = iy * dst_stride + ix * 4;
                out[off..off + 4].copy_from_slice(&pixel);
            }
        }
    }

    // Step 4: fill missing top-border rows (copies full rows including L/R borders).
    if dy > 0 {
        let src_row_start = dy * dst_stride;
        for iy in 0..dy {
            let dst_row_start = iy * dst_stride;
            out.copy_within(src_row_start..src_row_start + dst_stride, dst_row_start);
        }
    }

    // Step 5: fill missing bottom-border rows.
    let bottom_data_end = dy + read_h as usize;
    if bottom_data_end < out_h {
        let src_row_start = (bottom_data_end - 1) * dst_stride;
        for iy in bottom_data_end..out_h {
            let dst_row_start = iy * dst_stride;
            out.copy_within(src_row_start..src_row_start + dst_stride, dst_row_start);
        }
    }

    out
}

/// Tile coordinate identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    /// File identifier (distinguishes tiles from different open files)
    pub file_id: i32,
    /// Level index
    pub level: u32,
    /// X tile index
    pub x: u64,
    /// Y tile index
    pub y: u64,
    /// Tile size used to address this level
    pub tile_size: u32,
}

impl TileCoord {
    pub fn new(file_id: i32, level: u32, x: u64, y: u64, tile_size: u32) -> Self {
        Self {
            file_id,
            level,
            x,
            y,
            tile_size,
        }
    }
}

/// Tile data with RGBA pixels
#[derive(Debug, Clone)]
pub struct TileData {
    /// Tile coordinate
    pub coord: TileCoord,
    /// RGBA pixel data — layout is data_width() × data_height() × 4 bytes.
    pub data: Vec<u8>,
    /// Inner tile width (content, excluding border padding)
    pub width: u32,
    /// Inner tile height (content, excluding border padding)
    pub height: u32,
    /// Border padding in pixels on each side (0 = legacy, 1 = standard).
    /// When border > 0, `data` is `(width + 2*border) × (height + 2*border) × 4` bytes.
    pub border: u32,
}

impl TileData {
    /// Create a new tile data instance
    pub fn new(coord: TileCoord, data: Vec<u8>, width: u32, height: u32, border: u32) -> Self {
        Self {
            coord,
            data,
            width,
            height,
            border,
        }
    }

    /// Full data width including border padding on both sides.
    #[inline]
    pub fn data_width(&self) -> u32 {
        self.width + 2 * self.border
    }

    /// Full data height including border padding on both sides.
    #[inline]
    pub fn data_height(&self) -> u32 {
        self.height + 2 * self.border
    }

    /// Create a placeholder tile (checkerboard pattern for debugging)
    pub fn placeholder(coord: TileCoord, tile_size: u32) -> Self {
        let mut data = vec![0u8; (tile_size * tile_size * 4) as usize];

        // Create checkerboard pattern
        for y in 0..tile_size {
            for x in 0..tile_size {
                let idx = ((y * tile_size + x) * 4) as usize;
                let is_light = ((x / 16) + (y / 16)) % 2 == 0;
                let color = if is_light { 200 } else { 150 };
                data[idx] = color; // R
                data[idx + 1] = color; // G
                data[idx + 2] = color; // B
                data[idx + 3] = 255; // A
            }
        }

        Self {
            coord,
            data,
            width: tile_size,
            height: tile_size,
            border: 0,
        }
    }
}

/// Priority for tile loading
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TilePriority {
    /// Low priority (prefetch)
    Low = 0,
    /// Normal priority (visible)
    Normal = 1,
    /// High priority (center of view)
    High = 2,
}

/// Request to load a tile
#[derive(Debug, Clone)]
pub struct TileRequest {
    pub coord: TileCoord,
    pub priority: TilePriority,
}

/// A rectangle of tile indices `[x_start, x_end) x [y_start, y_end)` of one
/// level, clamped to that level's real tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRange {
    pub level: u32,
    pub tile_size: u32,
    pub x_start: u64,
    pub x_end: u64,
    pub y_start: u64,
    pub y_end: u64,
}

impl TileRange {
    /// Number of tiles in the range.
    pub fn len(&self) -> u64 {
        (self.x_end - self.x_start).saturating_mul(self.y_end - self.y_start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The tiles in row-major order.
    pub fn coords(&self, file_id: i32) -> impl Iterator<Item = TileCoord> + '_ {
        (self.y_start..self.y_end).flat_map(move |y| {
            (self.x_start..self.x_end)
                .map(move |x| TileCoord::new(file_id, self.level, x, y, self.tile_size))
        })
    }
}

/// The tiles of `level` that intersect the level-0 rectangle
/// `left, top, right, bottom`, grown by `margin_tiles` on each side and
/// clamped to the level's grid **for `tile_size`** (the grid of the size
/// the tiles are actually addressed with).
pub fn tile_range_for(
    level: &WsiLevel,
    tile_size: u32,
    bounds: [f64; 4],
    margin_tiles: i32,
) -> TileRange {
    let [left, top, right, bottom] = bounds;
    let tile_size = tile_size.max(1);
    let ts = f64::from(tile_size);
    let tiles_x = level.tiles_x(tile_size);
    let tiles_y = level.tiles_y(tile_size);
    let margin = i64::from(margin_tiles.max(0));
    let ds = level.downsample;
    // Float-to-int `as` saturates (NaN becomes 0), so degenerate bounds
    // produce an empty or clamped range rather than garbage.
    let start = |v: f64, n: u64| ((v / ds / ts).floor() as i64 - margin).clamp(0, n as i64) as u64;
    let end = |v: f64, n: u64| {
        ((v / ds / ts).ceil() as i64)
            .saturating_add(margin)
            .clamp(0, n as i64) as u64
    };
    let x_start = start(left, tiles_x);
    let y_start = start(top, tiles_y);
    TileRange {
        level: level.level,
        tile_size,
        x_start,
        x_end: end(right, tiles_x).max(x_start),
        y_start,
        y_end: end(bottom, tiles_y).max(y_start),
    }
}

/// The tiles of `range`, nearest to the level-0 point `center` first, at
/// most `cap` of them. When the range is larger than `cap`, a window of
/// about `cap` tiles centred on `center` is chosen first, so the result is
/// a central patch — not the top rows of the range.
pub fn select_centered(
    range: &TileRange,
    file_id: i32,
    downsample: f64,
    center: (f64, f64),
    cap: usize,
) -> Vec<TileCoord> {
    if range.is_empty() || cap == 0 {
        return Vec::new();
    }
    let ts = f64::from(range.tile_size) * downsample;
    // Centre in fractional tile units.
    let cx = center.0 / ts;
    let cy = center.1 / ts;

    let mut window = *range;
    let total = range.len();
    if total > cap as u64 {
        let width = range.x_end - range.x_start;
        let height = range.y_end - range.y_start;
        let scale = (cap as f64 / total as f64).sqrt();
        let w = ((width as f64 * scale).floor() as u64).clamp(1, width);
        let h = (cap as u64 / w).clamp(1, height);
        let place = |center: f64, len: u64, lo: u64, hi: u64| -> u64 {
            let start = (center - len as f64 / 2.0).round().max(lo as f64) as u64;
            start.min(hi - len)
        };
        window.x_start = place(cx, w, range.x_start, range.x_end);
        window.x_end = window.x_start + w;
        window.y_start = place(cy, h, range.y_start, range.y_end);
        window.y_end = window.y_start + h;
    }

    let mut tiles: Vec<TileCoord> = window.coords(file_id).collect();
    let distance = |tile: &TileCoord| {
        let dx = tile.x as f64 + 0.5 - cx;
        let dy = tile.y as f64 + 0.5 - cy;
        dx * dx + dy * dy
    };
    tiles.sort_by(|a, b| {
        distance(a)
            .total_cmp(&distance(b))
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });
    tiles.truncate(cap);
    tiles
}

/// Tile manager handles tile loading and coordination
pub struct TileManager {
    /// WSI file reference
    wsi: WsiFile,
    /// File identifier (propagated into every TileCoord this manager creates)
    file_id: i32,
    /// Tile size
    tile_size: u32,
    /// Border loaded around every tile
    border: u32,
    /// Last time the visible-tile cap warning was logged (ms since start).
    last_cap_warning_ms: AtomicU64,
}

impl TileManager {
    /// Create a new tile manager for the given WSI file
    pub fn new(wsi: WsiFile, file_id: i32) -> Self {
        let tile_size = wsi.tile_size();
        Self {
            wsi,
            file_id,
            tile_size,
            border: DEFAULT_TILE_BORDER,
            last_cap_warning_ms: AtomicU64::new(u64::MAX),
        }
    }

    /// Sets the border loaded around every tile (default
    /// [`DEFAULT_TILE_BORDER`]; use [`LANCZOS_TILE_BORDER`] or more for
    /// seamless Lanczos). Clamped to [`MAX_TILE_BORDER`].
    ///
    /// Tiles of different borders share cache keys, so use one border per
    /// cache, or clear the file's tiles when changing it.
    pub fn with_border(mut self, border: u32) -> Self {
        self.border = border.min(MAX_TILE_BORDER);
        self
    }

    /// The border loaded around every tile.
    pub fn border(&self) -> u32 {
        self.border
    }

    /// Get the file identifier
    pub fn file_id(&self) -> i32 {
        self.file_id
    }

    /// Get the tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    pub fn tile_size_for_level(&self, level: u32) -> u32 {
        self.wsi.tile_size_for_level(level)
    }

    /// Get the WSI file reference
    pub fn wsi(&self) -> &WsiFile {
        &self.wsi
    }

    /// Load a tile synchronously with the manager's border (1 px by default)
    /// for seamless bilinear/trilinear filtering at tile boundaries.
    ///
    /// Pixels are opaque RGBA (see [`WsiFile::read_region`]).
    pub fn load_tile_sync(&self, coord: TileCoord) -> Result<TileData> {
        self.load_tile_with_border(coord, self.border)
    }

    /// Load a tile with an explicit border (clamped to
    /// [`MAX_TILE_BORDER`]). Where the image ends, the border repeats the
    /// edge pixels.
    pub fn load_tile_with_border(&self, coord: TileCoord, border: u32) -> Result<TileData> {
        let border = border.min(MAX_TILE_BORDER);
        let level_info = self.wsi.level(coord.level).ok_or(Error::InvalidLevel(
            coord.level,
            self.wsi.level_count().saturating_sub(1),
        ))?;
        if coord.tile_size == 0 {
            return Err(Error::InvalidArgument("tile size 0".to_string()));
        }

        // Inner tile bounds in level coordinates
        let invalid = || Error::InvalidCoordinates {
            x: coord.x.min(i64::MAX as u64) as i64,
            y: coord.y.min(i64::MAX as u64) as i64,
            level: coord.level,
        };
        let tile_start_x = coord
            .x
            .checked_mul(coord.tile_size as u64)
            .ok_or_else(invalid)?;
        let tile_start_y = coord
            .y
            .checked_mul(coord.tile_size as u64)
            .ok_or_else(invalid)?;

        let inner_w =
            (level_info.width.saturating_sub(tile_start_x)).min(coord.tile_size as u64) as u32;
        let inner_h =
            (level_info.height.saturating_sub(tile_start_y)).min(coord.tile_size as u64) as u32;

        if inner_w == 0 || inner_h == 0 {
            return Err(invalid());
        }

        trace!(
            "Loading tile {:?}, inner size: {}x{}",
            coord, inner_w, inner_h
        );

        // Compute how many border pixels are available on each side
        // (clamped to image bounds at this level).
        let avail_left = tile_start_x.min(border as u64) as u32;
        let avail_top = tile_start_y.min(border as u64) as u32;
        let avail_right = (level_info
            .width
            .saturating_sub(tile_start_x + inner_w as u64))
        .min(border as u64) as u32;
        let avail_bottom = (level_info
            .height
            .saturating_sub(tile_start_y + inner_h as u64))
        .min(border as u64) as u32;

        // Expanded read region
        let read_x = tile_start_x - avail_left as u64;
        let read_y = tile_start_y - avail_top as u64;
        let read_w = avail_left + inner_w + avail_right;
        let read_h = avail_top + inner_h + avail_bottom;

        // Convert to level-0 coordinates for read_region
        let read_x0 = (read_x as f64 * level_info.downsample) as i64;
        let read_y0 = (read_y as f64 * level_info.downsample) as i64;

        let read_data = self
            .wsi
            .read_region(read_x0, read_y0, coord.level, read_w, read_h)?;

        // Build the uniformly-bordered output buffer.
        let data = create_bordered_tile(
            ReadRegion {
                data: &read_data,
                width: read_w,
                height: read_h,
                avail_left,
                avail_top,
            },
            inner_w,
            inner_h,
            border,
        );

        Ok(TileData::new(coord, data, inner_w, inner_h, border))
    }

    /// The range of tiles of `level` covering a level-0 rectangle, clamped
    /// to the level's grid for [`tile_size_for_level`](Self::tile_size_for_level).
    pub fn tile_range(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
        margin_tiles: i32,
    ) -> Option<TileRange> {
        let level_info = self.wsi.level(level)?;
        Some(tile_range_for(
            level_info,
            self.tile_size_for_level(level),
            [bounds_left, bounds_top, bounds_right, bounds_bottom],
            margin_tiles,
        ))
    }

    /// Calculate visible tiles for a given viewport
    ///
    /// Arguments:
    /// - level: the resolution level to use
    /// - bounds_left, bounds_top, bounds_right, bounds_bottom: visible area in level 0 coordinates
    pub fn visible_tiles(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
    ) -> Vec<TileCoord> {
        self.visible_tiles_with_margin(
            level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
            1,
        )
    }

    /// The tiles of `level` covering the level-0 rectangle (plus
    /// `margin_tiles` on each side), **nearest to its centre first**.
    ///
    /// At most [`MAX_VISIBLE_TILES`] are returned; larger sets (e.g. a
    /// single-level slide viewed at fit zoom) keep a central patch. Use
    /// [`visible_tiles_uncapped`](Self::visible_tiles_uncapped) for
    /// exhaustive enumeration (export).
    pub fn visible_tiles_with_margin(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
        margin_tiles: i32,
    ) -> Vec<TileCoord> {
        let Some(level_info) = self.wsi.level(level) else {
            return Vec::new();
        };
        let range = tile_range_for(
            level_info,
            self.tile_size_for_level(level),
            [bounds_left, bounds_top, bounds_right, bounds_bottom],
            margin_tiles,
        );
        let total = range.len();
        if total > MAX_VISIBLE_TILES as u64 {
            self.warn_cap(total, level);
        }
        select_centered(
            &range,
            self.file_id,
            level_info.downsample,
            (
                (bounds_left + bounds_right) / 2.0,
                (bounds_top + bounds_bottom) / 2.0,
            ),
            MAX_VISIBLE_TILES,
        )
    }

    /// Every tile of `level` covering the rectangle, in row-major order and
    /// without a cap. Intended for export and batch processing, which must
    /// not silently drop tiles; check [`tile_range`](Self::tile_range)'s
    /// `len()` first to bound the work.
    pub fn visible_tiles_uncapped(
        &self,
        level: u32,
        bounds_left: f64,
        bounds_top: f64,
        bounds_right: f64,
        bounds_bottom: f64,
        margin_tiles: i32,
    ) -> Vec<TileCoord> {
        self.tile_range(
            level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
            margin_tiles,
        )
        .map(|range| range.coords(self.file_id).collect())
        .unwrap_or_default()
    }

    fn warn_cap(&self, total: u64, level: u32) {
        let now = millis_since_start();
        let last = self.last_cap_warning_ms.load(Ordering::Relaxed);
        let due = last == u64::MAX || now.saturating_sub(last) >= CAP_WARN_INTERVAL_MS;
        if due
            && self
                .last_cap_warning_ms
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            warn!(
                "{total} tiles of level {level} are visible; keeping the {MAX_VISIBLE_TILES} \
                 nearest the view centre (the slide has no coarse enough level)"
            );
        } else {
            debug!("visible tiles capped: {total} at level {level}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffSpec, write_tiff};
    use std::path::Path;

    fn gradient(x: u32, y: u32) -> [u8; 3] {
        [(x % 256) as u8, (y % 256) as u8, ((x / 2 + y / 3) % 256) as u8]
    }

    fn pyramid(dir: &Path) -> TileManager {
        let path = dir.join("pyramid.tiff");
        write_tiff(&path, &TiffSpec::pyramid(600, 400, 64, 3), gradient).unwrap();
        TileManager::new(WsiFile::open(&path).unwrap(), 7)
    }

    fn fake_level(width: u64, height: u64, downsample: f64) -> WsiLevel {
        WsiLevel {
            level: 2,
            width,
            height,
            downsample,
            tile_width: None,
            tile_height: None,
        }
    }

    #[test]
    fn test_tile_coord_equality() {
        let a = TileCoord::new(0, 0, 1, 2, 256);
        let b = TileCoord::new(0, 0, 1, 2, 256);
        let c = TileCoord::new(0, 0, 1, 3, 256);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_placeholder_tile() {
        let coord = TileCoord::new(0, 0, 0, 0, 256);
        let tile = TileData::placeholder(coord, 256);

        assert_eq!(tile.width, 256);
        assert_eq!(tile.height, 256);
        assert_eq!(tile.data.len(), 256 * 256 * 4);
    }

    #[test]
    fn loads_bordered_tiles_with_neighbour_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let manager = pyramid(dir.path());
        let coord = TileCoord::new(7, 0, 1, 1, 256);
        let tile = manager.load_tile_sync(coord).unwrap();
        assert_eq!((tile.width, tile.height, tile.border), (256, 144, 1));
        assert_eq!(tile.data.len(), (258 * 146 * 4) as usize);
        // Data pixel (0, 0) is level pixel (255, 255): the neighbour's.
        let [r, g, b] = gradient(255, 255);
        assert_eq!(&tile.data[..4], [r, g, b, 255]);

        // Wider borders, replicated at the image edge.
        let tile = manager
            .load_tile_with_border(TileCoord::new(7, 0, 0, 0, 256), 3)
            .unwrap();
        assert_eq!(tile.border, 3);
        let stride = (256 + 6) * 4;
        let [r, g, b] = gradient(0, 0);
        assert_eq!(&tile.data[..4], [r, g, b, 255], "top-left corner is replicated");
        let [r, g, b] = gradient(256, 0);
        let right_border = 3 * stride + (3 + 256) * 4;
        assert_eq!(&tile.data[right_border..right_border + 4], [r, g, b, 255]);

        assert!(matches!(
            manager.load_tile_sync(TileCoord::new(7, 0, 9, 0, 256)),
            Err(Error::InvalidCoordinates { .. })
        ));
        assert_eq!(manager.with_border(500).border(), MAX_TILE_BORDER);
    }

    #[test]
    fn ranges_are_clamped_with_the_level_tile_size() {
        // The audit's case: a coarse level addressed with 1024-px tiles was
        // clamped with the 256-px grid, producing non-existent tiles.
        let level = fake_level(2240, 2260, 16.0);
        let whole = [0.0, 0.0, 2240.0 * 16.0, 2260.0 * 16.0];
        let range = tile_range_for(&level, 1024, whole, 1);
        assert_eq!((range.x_end, range.y_end), (3, 3));
        assert_eq!(range.len(), 9);
        let range_256 = tile_range_for(&level, 256, whole, 1);
        assert_eq!((range_256.x_end, range_256.y_end), (9, 9));
        // Past the right/bottom edge nothing is invented.
        let beyond = tile_range_for(&level, 1024, [30_000.0, 30_000.0, 90_000.0, 90_000.0], 1);
        assert!(beyond.coords(0).all(|c| c.x < 3 && c.y < 3));
        // Degenerate bounds give an empty range instead of panicking.
        let nan = tile_range_for(&level, 1024, [f64::NAN; 4], 0);
        assert!(nan.is_empty());
        assert!(!tile_range_for(&level, 0, whole, 0).is_empty());
    }

    #[test]
    fn large_visible_sets_keep_the_centre_not_the_top_rows() {
        // A 100k x 100k single-level slide at fit zoom: 152,881 candidates.
        let level = fake_level(100_000, 100_000, 1.0);
        let range = tile_range_for(&level, 256, [0.0, 0.0, 100_000.0, 100_000.0], 1);
        assert_eq!(range.len(), 391 * 391);
        let tiles = select_centered(&range, 0, 1.0, (50_000.0, 50_000.0), MAX_VISIBLE_TILES);
        assert_eq!(tiles.len(), MAX_VISIBLE_TILES);
        assert_eq!((tiles[0].x, tiles[0].y), (195, 195));
        let (min_y, max_y) = tiles
            .iter()
            .fold((u64::MAX, 0), |(lo, hi), t| (lo.min(t.y), hi.max(t.y)));
        assert!(min_y > 150 && max_y < 240, "rows {min_y}..={max_y} are not central");

        // Small sets are complete and centre-first.
        let small = tile_range_for(&level, 256, [0.0, 0.0, 1000.0, 1000.0], 0);
        let tiles = select_centered(&small, 0, 1.0, (500.0, 500.0), MAX_VISIBLE_TILES);
        assert_eq!(tiles.len(), 16);
        assert!([(1, 1), (1, 2), (2, 1), (2, 2)].contains(&(tiles[0].x, tiles[0].y)));
        let last = tiles.last().unwrap();
        assert!([(0, 0), (0, 3), (3, 0), (3, 3)].contains(&(last.x, last.y)));
    }

    #[test]
    fn visible_tiles_are_valid_and_centre_first() {
        let dir = tempfile::tempdir().unwrap();
        let manager = pyramid(dir.path());
        let tiles = manager.visible_tiles(0, 0.0, 0.0, 600.0, 400.0);
        // 600x400 with 256-px tiles: a 3x2 grid, all of it.
        assert_eq!(tiles.len(), 6);
        assert!(tiles.iter().all(|t| manager.load_tile_sync(*t).is_ok()));
        assert_eq!((tiles[0].x, tiles[0].y), (1, 0));

        let uncapped = manager.visible_tiles_uncapped(0, 0.0, 0.0, 600.0, 400.0, 1);
        assert_eq!(uncapped.len(), 6);
        assert_eq!((uncapped[0].x, uncapped[0].y), (0, 0), "row-major");
        assert_eq!(
            manager.tile_range(0, 0.0, 0.0, 600.0, 400.0, 0).unwrap().len(),
            6
        );
        assert!(manager.visible_tiles(9, 0.0, 0.0, 1.0, 1.0).is_empty());
    }
}
