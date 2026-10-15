//! Pyramid geometry and the streaming pyramid builder.
//!
//! Level `k + 1` halves level `k` (rounding up) until a level fits in one
//! tile. Tile `(col, row)` of level `k + 1` is the 2×2 box filter (mean) of
//! tiles `(2col..2col+1, 2row..2row+1)` of level `k`, so the pyramid is a
//! quadtree and can be built depth first: every tile is written as soon as
//! its four children are, and at most one partially built tile per level
//! is held in memory. Level-0 tiles are produced in batches (a subtree of
//! up to 64 tiles, or 256 for small tiles) that are read and compressed in
//! parallel and then written in a fixed order, so output files are
//! deterministic. Memory is bounded by the batch (about 50 MB of pixels)
//! plus one tile per level, whatever the slide's size.
//!
//! Tiles are packed RGB, `tile × tile` pixels; the part of an edge tile
//! beyond the image repeats the last row and column, which keeps JPEG
//! edges clean and makes the box filter clamp at the image edge.

use std::collections::HashMap;
use std::io::{Seek, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use flate2::Compression as DeflateLevel;
use flate2::write::ZlibEncoder;
use parking_lot::Mutex;
use rayon::prelude::*;

use super::ConvertError;
use super::tiff_file::TiffFile;
use crate::export::{JpegTileEncoder, Subsampling};
use crate::job::JobControl;

/// Sizes of the pyramid levels, full resolution first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub tile: u32,
    pub levels: Vec<(u32, u32)>,
}

impl Geometry {
    /// Levels halving down to one that fits in a single tile.
    pub(crate) fn new(width: u32, height: u32, tile: u32) -> Self {
        let mut levels = vec![(width.max(1), height.max(1))];
        while let Some(&(w, h)) = levels.last()
            && (w > tile || h > tile)
        {
            levels.push((w.div_ceil(2), h.div_ceil(2)));
        }
        Self { tile, levels }
    }

    /// Levels halving while both sides stay at least one tile long.
    ///
    /// OpenSlide reads tiles that are not JPEG through libtiff's
    /// `TIFFRGBAImageGet` with a raster of a whole tile, and libtiff puts
    /// an image shorter (or narrower) than that raster at its bottom, so
    /// such levels read shifted and partly transparent (OpenSlide 4.0 with
    /// libtiff 4.5). Lossless copies therefore never have levels smaller
    /// than a tile; `tile` must not exceed either side of level 0.
    pub(crate) fn whole_tiles(width: u32, height: u32, tile: u32) -> Self {
        let mut levels = vec![(width.max(1), height.max(1))];
        while let Some(&(w, h)) = levels.last() {
            let next = (w.div_ceil(2), h.div_ceil(2));
            if next.0.min(next.1) < tile {
                break;
            }
            levels.push(next);
        }
        Self { tile, levels }
    }

    pub(crate) fn cols(&self, level: usize) -> u32 {
        self.levels[level].0.div_ceil(self.tile)
    }

    pub(crate) fn rows(&self, level: usize) -> u32 {
        self.levels[level].1.div_ceil(self.tile)
    }

    pub(crate) fn tiles(&self, level: usize) -> u64 {
        u64::from(self.cols(level)) * u64::from(self.rows(level))
    }

    pub(crate) fn total_tiles(&self) -> u64 {
        (0..self.levels.len()).map(|level| self.tiles(level)).sum()
    }

    /// Pixels of tile `(col, row)` that lie inside the image.
    pub(crate) fn valid(&self, level: usize, col: u32, row: u32) -> (usize, usize) {
        let (w, h) = self.levels[level];
        (
            (w - col * self.tile).min(self.tile) as usize,
            (h - row * self.tile).min(self.tile) as usize,
        )
    }
}

/// Repeats the last valid column and row of a `tile × tile` RGB tile into
/// the part beyond the image.
pub(crate) fn pad_edges(rgb: &mut [u8], tile: usize, valid_w: usize, valid_h: usize) {
    let stride = tile * 3;
    if valid_w == 0 || valid_h == 0 {
        return;
    }
    if valid_w < tile {
        for row in rgb.chunks_exact_mut(stride).take(valid_h) {
            let last = [
                row[(valid_w - 1) * 3],
                row[(valid_w - 1) * 3 + 1],
                row[(valid_w - 1) * 3 + 2],
            ];
            for pixel in row[valid_w * 3..].as_chunks_mut::<3>().0 {
                *pixel = last;
            }
        }
    }
    if valid_h < tile {
        let (top, bottom) = rgb.split_at_mut(valid_h * stride);
        let last = &top[(valid_h - 1) * stride..];
        for row in bottom.chunks_exact_mut(stride) {
            row.copy_from_slice(last);
        }
    }
}

/// Box-filters a child tile into quadrant `(qx, qy)` of its parent.
pub(crate) fn downsample_quadrant(
    child: &[u8],
    parent: &mut [u8],
    tile: usize,
    qx: usize,
    qy: usize,
) {
    let half = tile / 2;
    let stride = tile * 3;
    for y in 0..half {
        let top = &child[2 * y * stride..][..stride];
        let bottom = &child[(2 * y + 1) * stride..][..stride];
        let out = &mut parent[(qy * half + y) * stride + qx * half * 3..][..half * 3];
        for x in 0..half {
            for c in 0..3 {
                let i = x * 6 + c;
                let sum = u16::from(top[i])
                    + u16::from(top[i + 3])
                    + u16::from(bottom[i])
                    + u16::from(bottom[i + 3]);
                out[x * 3 + c] = ((sum + 2) / 4) as u8;
            }
        }
    }
}

/// Tile compression.
pub(crate) enum TileCodec {
    Jpeg(JpegTileEncoder),
    /// Adobe Deflate with the horizontal-differencing predictor.
    Deflate,
}

/// Compresses tiles, reusing the bytes of uniform tiles (glass,
/// background) of a colour already seen.
pub(crate) struct TileEncoder {
    codec: TileCodec,
    tile: u32,
    uniform: Mutex<HashMap<[u8; 3], Arc<Vec<u8>>>>,
}

impl TileEncoder {
    pub(crate) fn new(codec: TileCodec, tile: u32) -> Self {
        Self {
            codec,
            tile,
            uniform: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn codec(&self) -> &TileCodec {
        &self.codec
    }

    /// Compresses one `tile × tile` RGB tile.
    pub(crate) fn encode(&self, rgb: &[u8]) -> std::io::Result<Arc<Vec<u8>>> {
        let first = [rgb[0], rgb[1], rgb[2]];
        let uniform = rgb.as_chunks::<3>().0.iter().all(|pixel| *pixel == first);
        if uniform && let Some(bytes) = self.uniform.lock().get(&first) {
            return Ok(Arc::clone(bytes));
        }
        let bytes = Arc::new(match &self.codec {
            TileCodec::Jpeg(encoder) => encoder.encode(rgb, self.tile, self.tile)?,
            TileCodec::Deflate => deflate_rows(rgb, self.tile as usize)?,
        });
        if uniform {
            let mut cache = self.uniform.lock();
            // A slide has few background colours; never let it grow.
            if cache.len() < 64 {
                cache.insert(first, Arc::clone(&bytes));
            }
        }
        Ok(bytes)
    }
}

/// Zlib-compresses packed RGB rows of `width` pixels after the TIFF
/// horizontal predictor (each sample minus the one a pixel to its left).
pub(crate) fn deflate_rows(rgb: &[u8], width: usize) -> std::io::Result<Vec<u8>> {
    let stride = width * 3;
    let mut predicted = rgb.to_vec();
    for row in predicted.chunks_exact_mut(stride) {
        for i in (3..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - 3]);
        }
    }
    let mut encoder = ZlibEncoder::new(
        Vec::with_capacity(predicted.len() / 2),
        DeflateLevel::new(6),
    );
    encoder.write_all(&predicted)?;
    encoder.finish()
}

/// Whether a source tile could be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TileRead {
    Read,
    /// The slide data there is damaged; the tile holds the background.
    Unreadable,
}

/// Where the full-resolution tiles come from.
pub(crate) trait TileSource: Sync {
    /// Fills `out` (`tile × tile` packed RGB) with full-resolution tile
    /// `(col, row)`, edge-padded beyond the image.
    fn read_tile(&self, col: u32, row: u32, out: &mut [u8]) -> Result<TileRead, ConvertError>;
}

/// Where each level's tiles were written.
#[derive(Debug, Clone, Default)]
pub(crate) struct LevelTiles {
    /// Row-major.
    pub offsets: Vec<u64>,
    pub counts: Vec<u64>,
}

/// The result of [`write_pyramid`].
#[derive(Debug)]
pub(crate) struct WrittenPyramid {
    pub levels: Vec<LevelTiles>,
    pub unreadable_tiles: u64,
}

/// Bytes of RGB pixels a batch may hold.
const BATCH_PIXEL_BYTES: u64 = 64 << 20;

/// The level whose tiles are the roots of the level-0 batches: the
/// smallest subtree with at least two tiles per thread, within the memory
/// bound, and never above the top level.
fn batch_level(geometry: &Geometry, threads: usize) -> usize {
    let tile_bytes = u64::from(geometry.tile) * u64::from(geometry.tile) * 3;
    let mut level = 0usize;
    while level + 1 < geometry.levels.len()
        && (1u64 << (2 * level)) < 2 * threads as u64
        && (1u64 << (2 * (level + 1))) * tile_bytes <= BATCH_PIXEL_BYTES
    {
        level += 1;
    }
    level
}

/// Builds and writes every tile of the pyramid.
pub(crate) fn write_pyramid<W: Write + Seek>(
    file: &mut TiffFile<W>,
    geometry: &Geometry,
    source: &dyn TileSource,
    encoder: &TileEncoder,
    control: &JobControl,
) -> Result<WrittenPyramid, ConvertError> {
    let levels = (0..geometry.levels.len())
        .map(|level| {
            let count = geometry.tiles(level) as usize;
            LevelTiles {
                offsets: vec![0; count],
                counts: vec![0; count],
            }
        })
        .collect();
    let mut builder = Builder {
        file,
        geometry,
        source,
        encoder,
        control,
        batch_level: batch_level(geometry, rayon::current_num_threads()),
        levels,
        unreadable: AtomicU64::new(0),
    };
    // The top level has one tile, or a few (lossless copies stop while
    // levels are still at least a tile long).
    let top = geometry.levels.len() - 1;
    for row in 0..geometry.rows(top) {
        for col in 0..geometry.cols(top) {
            builder.build(top, col, row)?;
        }
    }
    Ok(WrittenPyramid {
        levels: builder.levels,
        unreadable_tiles: builder.unreadable.into_inner(),
    })
}

struct Builder<'a, W: Write + Seek> {
    file: &'a mut TiffFile<W>,
    geometry: &'a Geometry,
    source: &'a dyn TileSource,
    encoder: &'a TileEncoder,
    control: &'a JobControl,
    batch_level: usize,
    levels: Vec<LevelTiles>,
    unreadable: AtomicU64,
}

/// A tile's pixels and compressed bytes, or why it could not be made.
type Produced = Result<(Vec<u8>, Arc<Vec<u8>>), ConvertError>;

/// Children of a tile, in the order they are filtered.
const QUADRANTS: [(u32, u32); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

impl<W: Write + Seek> Builder<'_, W> {
    fn tile_len(&self) -> usize {
        let tile = self.geometry.tile as usize;
        tile * tile * 3
    }

    fn write_tile(
        &mut self,
        level: usize,
        col: u32,
        row: u32,
        bytes: &[u8],
    ) -> Result<(), ConvertError> {
        let offset = self.file.append(bytes)?;
        let index = row as usize * self.geometry.cols(level) as usize + col as usize;
        self.levels[level].offsets[index] = offset;
        self.levels[level].counts[index] = bytes.len() as u64;
        self.control.advance(1);
        Ok(())
    }

    /// Writes tile `(col, row)` of `level` and everything below it, and
    /// returns its pixels.
    fn build(&mut self, level: usize, col: u32, row: u32) -> Result<Vec<u8>, ConvertError> {
        if level <= self.batch_level {
            return self.build_batch(col, row);
        }
        let tile = self.geometry.tile as usize;
        let mut pixels = vec![0u8; self.tile_len()];
        for (dx, dy) in QUADRANTS {
            let (child_col, child_row) = (col * 2 + dx, row * 2 + dy);
            if child_col >= self.geometry.cols(level - 1)
                || child_row >= self.geometry.rows(level - 1)
            {
                continue;
            }
            let child = self.build(level - 1, child_col, child_row)?;
            downsample_quadrant(&child, &mut pixels, tile, dx as usize, dy as usize);
        }
        let (valid_w, valid_h) = self.geometry.valid(level, col, row);
        pad_edges(&mut pixels, tile, valid_w, valid_h);
        let bytes = self.encoder.encode(&pixels)?;
        self.write_tile(level, col, row, &bytes)?;
        Ok(pixels)
    }

    /// Writes the subtree below tile `(col, row)` of the batch level, every
    /// level of it in parallel, and returns that tile's pixels.
    fn build_batch(&mut self, col: u32, row: u32) -> Result<Vec<u8>, ConvertError> {
        if self.control.is_cancelled() {
            return Err(ConvertError::Cancelled);
        }
        let top = self.batch_level;
        let tile = self.geometry.tile as usize;
        let tile_len = self.tile_len();

        // Level 0: read and compress.
        let span = 1u32 << top;
        let (col0, row0) = (col * span, row * span);
        let mut cols = span.min(self.geometry.cols(0) - col0);
        let mut rows = span.min(self.geometry.rows(0) - row0);
        let coords: Vec<(u32, u32)> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (col0 + c, row0 + r)))
            .collect();
        let (source, encoder, control, unreadable) =
            (self.source, self.encoder, self.control, &self.unreadable);
        let produced: Vec<Produced> = coords
            .par_iter()
            .map(|&(c, r)| {
                if control.is_cancelled() {
                    return Err(ConvertError::Cancelled);
                }
                let mut pixels = vec![0u8; tile_len];
                if source.read_tile(c, r, &mut pixels)? == TileRead::Unreadable {
                    unreadable.fetch_add(1, Ordering::Relaxed);
                }
                let bytes = encoder.encode(&pixels)?;
                Ok((pixels, bytes))
            })
            .collect();
        let mut current = Vec::with_capacity(produced.len());
        for (&(c, r), result) in coords.iter().zip(produced) {
            let (pixels, bytes) = result?;
            self.write_tile(0, c, r, &bytes)?;
            current.push(pixels);
        }

        // Levels 1..=top of the subtree: filter and compress.
        let (mut cur_col0, mut cur_row0) = (col0, row0);
        for level in 1..=top {
            if self.control.is_cancelled() {
                return Err(ConvertError::Cancelled);
            }
            let span = 1u32 << (top - level);
            let (lcol0, lrow0) = (col * span, row * span);
            let lcols = span.min(self.geometry.cols(level) - lcol0);
            let lrows = span.min(self.geometry.rows(level) - lrow0);
            let coords: Vec<(u32, u32)> = (0..lrows)
                .flat_map(|r| (0..lcols).map(move |c| (lcol0 + c, lrow0 + r)))
                .collect();
            let geometry = self.geometry;
            let children = &current;
            let (child_cols, child_rows) = (cols, rows);
            let produced: Vec<Produced> = coords
                .par_iter()
                .map(|&(c, r)| {
                    let mut pixels = vec![0u8; tile_len];
                    for (dx, dy) in QUADRANTS {
                        let (child_col, child_row) = (c * 2 + dx, r * 2 + dy);
                        if child_col >= cur_col0 + child_cols || child_row >= cur_row0 + child_rows
                        {
                            continue;
                        }
                        let index =
                            ((child_row - cur_row0) * child_cols + (child_col - cur_col0)) as usize;
                        downsample_quadrant(
                            &children[index],
                            &mut pixels,
                            tile,
                            dx as usize,
                            dy as usize,
                        );
                    }
                    let (valid_w, valid_h) = geometry.valid(level, c, r);
                    pad_edges(&mut pixels, tile, valid_w, valid_h);
                    let bytes = encoder.encode(&pixels)?;
                    Ok((pixels, bytes))
                })
                .collect();
            let mut next = Vec::with_capacity(produced.len());
            for (&(c, r), result) in coords.iter().zip(produced) {
                let (pixels, bytes) = result?;
                self.write_tile(level, c, r, &bytes)?;
                next.push(pixels);
            }
            current = next;
            (cols, rows, cur_col0, cur_row0) = (lcols, lrows, lcol0, lrow0);
        }
        debug_assert_eq!(current.len(), 1);
        Ok(current.pop().unwrap_or_else(|| vec![0u8; tile_len]))
    }
}

/// The JPEG settings for `quality`: chroma at full resolution from 90 up,
/// subsampled 2×2 below (as the image export does).
pub(crate) fn jpeg_codec(quality: u8) -> TileCodec {
    let subsampling = if quality >= 90 {
        Subsampling::None
    } else {
        Subsampling::Half
    };
    TileCodec::Jpeg(JpegTileEncoder::new(quality, subsampling))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn levels_halve_until_one_tile() {
        let g = Geometry::new(1000, 300, 256);
        assert_eq!(g.levels, vec![(1000, 300), (500, 150), (250, 75)]);
        assert_eq!((g.cols(0), g.rows(0)), (4, 2));
        assert_eq!(g.tiles(2), 1);
        assert_eq!(g.total_tiles(), 8 + 2 + 1);
        assert_eq!(g.valid(0, 3, 1), (1000 - 768, 300 - 256));
        assert_eq!(Geometry::new(100, 80, 512).levels, vec![(100, 80)]);
        assert_eq!(Geometry::new(513, 1, 512).levels, vec![(513, 1), (257, 1)]);
        assert_eq!(
            Geometry::whole_tiles(1000, 600, 256).levels,
            vec![(1000, 600), (500, 300)]
        );
        assert_eq!(
            Geometry::whole_tiles(300, 200, 192).levels,
            vec![(300, 200)]
        );
        // Child grids always halve with the levels.
        for (w, h) in [(100_001, 77_777), (4097, 4096), (65_537, 3)] {
            let g = Geometry::new(w, h, 512);
            for level in 1..g.levels.len() {
                assert_eq!(g.cols(level), g.cols(level - 1).div_ceil(2));
                assert_eq!(g.rows(level), g.rows(level - 1).div_ceil(2));
            }
            assert_eq!(g.tiles(g.levels.len() - 1), 1);
        }
    }

    #[test]
    fn padding_repeats_the_edges() {
        let tile = 4;
        let mut rgb = vec![0u8; tile * tile * 3];
        for y in 0..2 {
            for x in 0..3 {
                rgb[(y * tile + x) * 3] = (10 * y + x) as u8 + 1;
            }
        }
        pad_edges(&mut rgb, tile, 3, 2);
        let red: Vec<u8> = rgb.chunks(3).map(|p| p[0]).collect();
        assert_eq!(
            red,
            [1, 2, 3, 3, 11, 12, 13, 13, 11, 12, 13, 13, 11, 12, 13, 13]
        );
    }

    #[test]
    fn box_filter_averages_two_by_two() {
        let tile = 4;
        let child: Vec<u8> = (0..tile * tile)
            .flat_map(|i| {
                let v = (i * 10) as u8;
                [v, 255 - v, 7]
            })
            .collect();
        let mut parent = vec![0u8; tile * tile * 3];
        downsample_quadrant(&child, &mut parent, tile, 1, 0);
        // Parent pixel (2, 0) = mean of child (0,0), (1,0), (0,1), (1,1).
        let expected = (10 + 40 + 50 + 2) / 4;
        assert_eq!(parent[2 * 3], expected as u8);
        assert_eq!(parent[2 * 3 + 2], 7);
        // The left half is untouched.
        assert_eq!(parent[0], 0);
    }

    #[test]
    fn predictor_rows_round_trip_through_zlib() {
        let width = 5;
        let rgb: Vec<u8> = (0..width * 2 * 3).map(|i| (i * 37 % 251) as u8).collect();
        let compressed = deflate_rows(&rgb, width).unwrap();
        let mut decoder = flate2::read::ZlibDecoder::new(&compressed[..]);
        let mut predicted = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut predicted).unwrap();
        for row in predicted.chunks_exact_mut(width * 3) {
            for i in 3..row.len() {
                row[i] = row[i].wrapping_add(row[i - 3]);
            }
        }
        assert_eq!(predicted, rgb);
    }

    struct Gradient {
        geometry: Geometry,
        reads: AtomicU64,
    }

    impl TileSource for Gradient {
        fn read_tile(&self, col: u32, row: u32, out: &mut [u8]) -> Result<TileRead, ConvertError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let tile = self.geometry.tile as usize;
            let (vw, vh) = self.geometry.valid(0, col, row);
            for y in 0..vh {
                for x in 0..vw {
                    let (gx, gy) = (col as usize * tile + x, row as usize * tile + y);
                    let i = (y * tile + x) * 3;
                    out[i..i + 3].copy_from_slice(&[(gx % 256) as u8, (gy % 256) as u8, 99]);
                }
            }
            pad_edges(out, tile, vw, vh);
            Ok(TileRead::Read)
        }
    }

    #[test]
    fn every_tile_is_written_once_and_cancellation_stops_the_build() {
        for geometry in [
            Geometry::new(700, 300, 64),
            Geometry::whole_tiles(700, 300, 64),
            Geometry::whole_tiles(200, 1000, 48),
        ] {
            let source = Gradient {
                geometry: geometry.clone(),
                reads: AtomicU64::new(0),
            };
            let encoder = TileEncoder::new(TileCodec::Deflate, geometry.tile);
            let control = JobControl::new();
            let mut file = TiffFile::new(Cursor::new(Vec::new())).unwrap();
            let written = write_pyramid(&mut file, &geometry, &source, &encoder, &control).unwrap();
            assert_eq!(source.reads.load(Ordering::Relaxed), geometry.tiles(0));
            assert_eq!(control.progress().done, geometry.total_tiles());
            for (level, tiles) in written.levels.iter().enumerate() {
                assert_eq!(tiles.offsets.len() as u64, geometry.tiles(level));
                assert!(
                    tiles.offsets.iter().all(|&o| o >= 16) && tiles.counts.iter().all(|&c| c > 0),
                    "{geometry:?} level {level}"
                );
            }
            // Offsets are distinct: nothing was written twice or skipped.
            let mut all: Vec<u64> = written
                .levels
                .iter()
                .flat_map(|l| l.offsets.clone())
                .collect();
            all.sort_unstable();
            all.dedup();
            assert_eq!(all.len() as u64, geometry.total_tiles());

            let cancelled = JobControl::new();
            cancelled.cancel();
            let mut file = TiffFile::new(Cursor::new(Vec::new())).unwrap();
            assert!(matches!(
                write_pyramid(&mut file, &geometry, &source, &encoder, &cancelled),
                Err(ConvertError::Cancelled)
            ));
        }
    }

    #[test]
    fn batches_stay_within_the_memory_bound() {
        let geometry = Geometry::new(100_000, 80_000, 512);
        let level = batch_level(&geometry, 32);
        assert_eq!(level, 3);
        assert!((1u64 << (2 * level)) * 512 * 512 * 3 <= BATCH_PIXEL_BYTES);
        assert_eq!(batch_level(&geometry, 1), 1);
        assert_eq!(batch_level(&Geometry::new(100, 100, 512), 32), 0);
        assert_eq!(batch_level(&Geometry::new(100_000, 80_000, 256), 64), 4);
    }

    #[test]
    fn uniform_tiles_are_compressed_once() {
        let encoder = TileEncoder::new(jpeg_codec(80), 16);
        let white = vec![255u8; 16 * 16 * 3];
        let a = encoder.encode(&white).unwrap();
        let b = encoder.encode(&white).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let mut varied = white.clone();
        varied[5] = 0;
        assert!(!Arc::ptr_eq(&encoder.encode(&varied).unwrap(), &a));
    }
}
