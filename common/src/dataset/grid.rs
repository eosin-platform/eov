//! Patch grid coordinates, generated lazily (DS-4).

/// A single patch coordinate in the fixed grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchCoord {
    /// X origin in level-0 pixels.
    pub x: u64,
    /// Y origin in level-0 pixels.
    pub y: u64,
}

/// A deterministic, row-major grid of patch origins over a slide.
///
/// Patches have a level-0 footprint of `footprint_x × footprint_y` pixels
/// and origins every `step_x`/`step_y` level-0 pixels (fractional steps are
/// rounded per position, so a grid at a target resolution never drifts).
/// Only full patches are produced. Nothing is allocated up front: use
/// [`PatchGrid::len`] for the count and [`PatchGrid::get`] /
/// [`PatchGrid::iter`] for positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchGrid {
    pub columns: u64,
    pub rows: u64,
    pub step_x: f64,
    pub step_y: f64,
    pub footprint_x: f64,
    pub footprint_y: f64,
}

impl PatchGrid {
    /// A grid over a `width × height` slide (level-0 pixels).
    pub fn new(
        width: u64,
        height: u64,
        footprint_x: f64,
        footprint_y: f64,
        step_x: f64,
        step_y: f64,
    ) -> Self {
        let count = |extent: u64, footprint: f64, step: f64| -> u64 {
            if !(footprint > 0.0 && step > 0.0) || (extent as f64) < footprint {
                0
            } else {
                // Origins i·step with round(i·step) + footprint ≤ extent.
                let last = ((extent as f64 - footprint) / step).floor() as u64;
                let mut n = last + 1;
                while n > 0 && ((n - 1) as f64 * step).round() + footprint > extent as f64 {
                    n -= 1;
                }
                n
            }
        };
        Self {
            columns: count(width, footprint_x, step_x),
            rows: count(height, footprint_y, step_y),
            step_x,
            step_y,
            footprint_x,
            footprint_y,
        }
    }

    /// The level-0 grid of `tile_size` patches every `stride` pixels.
    pub fn level0(width: u64, height: u64, tile_size: u32, stride: u32) -> Self {
        Self::new(
            width,
            height,
            tile_size as f64,
            tile_size as f64,
            stride as f64,
            stride as f64,
        )
    }

    /// Number of positions.
    pub fn len(&self) -> u64 {
        self.columns.saturating_mul(self.rows)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Position `index` in row-major order.
    pub fn get(&self, index: u64) -> PatchCoord {
        let row = index / self.columns.max(1);
        let col = index % self.columns.max(1);
        PatchCoord {
            x: (col as f64 * self.step_x).round() as u64,
            y: (row as f64 * self.step_y).round() as u64,
        }
    }

    /// All positions, row-major (Y outer, X inner).
    pub fn iter(&self) -> impl Iterator<Item = PatchCoord> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

/// Generates the level-0 grid as a vector (pre-1.0 API). Prefer
/// [`PatchGrid`], which does not allocate.
///
/// Returns an empty `Vec` when the slide is smaller than `tile_size` or the
/// stride is zero.
pub fn generate_patch_coords(
    slide_width: u64,
    slide_height: u64,
    tile_size: u32,
    stride: u32,
) -> Vec<PatchCoord> {
    if stride == 0 {
        return Vec::new();
    }
    PatchGrid::level0(slide_width, slide_height, tile_size, stride)
        .iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_grid() {
        let coords = generate_patch_coords(1024, 1024, 512, 512);
        assert_eq!(coords.len(), 4);
        assert_eq!(coords[0], PatchCoord { x: 0, y: 0 });
        assert_eq!(coords[1], PatchCoord { x: 512, y: 0 });
        assert_eq!(coords[2], PatchCoord { x: 0, y: 512 });
        assert_eq!(coords[3], PatchCoord { x: 512, y: 512 });
    }

    #[test]
    fn test_stride_smaller_than_tile() {
        let coords = generate_patch_coords(1024, 512, 512, 256);
        assert_eq!(
            coords,
            vec![
                PatchCoord { x: 0, y: 0 },
                PatchCoord { x: 256, y: 0 },
                PatchCoord { x: 512, y: 0 }
            ]
        );
    }

    #[test]
    fn test_slide_smaller_than_tile() {
        assert!(generate_patch_coords(256, 256, 512, 512).is_empty());
    }

    #[test]
    fn test_edge_tiles_skipped() {
        assert_eq!(
            generate_patch_coords(1000, 512, 512, 512),
            vec![PatchCoord { x: 0, y: 0 }]
        );
    }

    #[test]
    fn test_zero_stride() {
        assert!(generate_patch_coords(1024, 1024, 512, 0).is_empty());
    }

    #[test]
    fn huge_grids_are_not_materialised() {
        // A 100k × 80k slide with stride 1 has ~7.9e9 positions (DS-4);
        // the grid knows its size without allocating.
        let grid = PatchGrid::level0(100_000, 80_000, 512, 1);
        assert_eq!(grid.len(), (100_000 - 512 + 1) * (80_000 - 512 + 1));
        assert_eq!(
            grid.get(grid.len() - 1),
            PatchCoord {
                x: 99_488,
                y: 79_488
            }
        );
    }

    #[test]
    fn fractional_steps_round_per_position_and_stay_inside() {
        // 0.25 µm/px slide at 0.5 µm/px with 256-px tiles: footprint 512.
        let grid = PatchGrid::new(1300, 600, 512.0, 512.0, 409.6, 409.6);
        let coords: Vec<_> = grid.iter().collect();
        assert_eq!(grid.columns, 2);
        assert_eq!(coords[1], PatchCoord { x: 410, y: 0 });
        for c in coords {
            assert!(c.x as f64 + 512.0 <= 1300.0 && c.y as f64 + 512.0 <= 600.0);
        }
    }
}
