//! Patch grid coordinate generation.

/// A single patch coordinate in the fixed grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchCoord {
    /// X origin in level-0 pixels.
    pub x: u64,
    /// Y origin in level-0 pixels.
    pub y: u64,
}

/// Generate deterministic fixed-grid patch coordinates over a slide.
///
/// The grid starts at `(0, 0)` and steps by `stride` in both axes (row-major
/// order: Y outer, X inner). Only **full tiles** are emitted—patches whose
/// extent would exceed the slide bounds are skipped. This makes the output
/// reproducible and avoids partial-tile edge artifacts commonly undesirable
/// in ML datasets.
///
/// Returns an empty `Vec` (without error) when the slide is smaller than
/// `tile_size` in either dimension.
pub fn generate_patch_coords(
    slide_width: u64,
    slide_height: u64,
    tile_size: u32,
    stride: u32,
) -> Vec<PatchCoord> {
    let ts = tile_size as u64;
    let st = stride as u64;

    if slide_width < ts || slide_height < ts || st == 0 {
        return Vec::new();
    }

    // Last valid origin in each axis such that origin + tile_size <= dimension.
    let max_x = slide_width - ts;
    let max_y = slide_height - ts;

    let nx = max_x / st + 1;
    let ny = max_y / st + 1;

    let mut coords = Vec::with_capacity((nx * ny) as usize);

    // Row-major traversal: Y outer, X inner.
    let mut y = 0u64;
    while y <= max_y {
        let mut x = 0u64;
        while x <= max_x {
            coords.push(PatchCoord { x, y });
            x += st;
        }
        y += st;
    }

    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_grid() {
        let coords = generate_patch_coords(1024, 1024, 512, 512);
        // 2×2 grid
        assert_eq!(coords.len(), 4);
        assert_eq!(coords[0], PatchCoord { x: 0, y: 0 });
        assert_eq!(coords[1], PatchCoord { x: 512, y: 0 });
        assert_eq!(coords[2], PatchCoord { x: 0, y: 512 });
        assert_eq!(coords[3], PatchCoord { x: 512, y: 512 });
    }

    #[test]
    fn test_stride_smaller_than_tile() {
        let coords = generate_patch_coords(1024, 512, 512, 256);
        // X: 0, 256, 512 → 3 positions; Y: 0 → 1 position
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0], PatchCoord { x: 0, y: 0 });
        assert_eq!(coords[1], PatchCoord { x: 256, y: 0 });
        assert_eq!(coords[2], PatchCoord { x: 512, y: 0 });
    }

    #[test]
    fn test_slide_smaller_than_tile() {
        let coords = generate_patch_coords(256, 256, 512, 512);
        assert!(coords.is_empty());
    }

    #[test]
    fn test_edge_tiles_skipped() {
        // 1000 wide, tile_size=512, stride=512 → only x=0 fits (0+512 ≤ 1000, 512+512 > 1000)
        let coords = generate_patch_coords(1000, 512, 512, 512);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0], PatchCoord { x: 0, y: 0 });
    }

    #[test]
    fn test_single_tile_exact() {
        let coords = generate_patch_coords(512, 512, 512, 512);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0], PatchCoord { x: 0, y: 0 });
    }

    #[test]
    fn test_zero_stride() {
        let coords = generate_patch_coords(1024, 1024, 512, 0);
        assert!(coords.is_empty());
    }
}
