//! Test-only helpers shared by unit tests in this crate.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Writes a minimal uncompressed, tiled, 8-bit RGB TIFF that OpenSlide opens
/// with its generic-tiff driver. `pixel(x, y)` supplies the colour of every
/// pixel; `tile` must be a multiple of 16.
pub(crate) fn write_tiled_tiff(
    path: &Path,
    width: u32,
    height: u32,
    tile: u32,
    pixel: impl Fn(u32, u32) -> [u8; 3],
) -> std::io::Result<()> {
    assert!(tile.is_multiple_of(16), "TIFF tile sizes must be multiples of 16");
    let tiles_x = width.div_ceil(tile);
    let tiles_y = height.div_ceil(tile);
    let tile_count = (tiles_x * tiles_y) as usize;
    let tile_bytes = (tile * tile * 3) as usize;

    // Layout: header | tile data | bits-per-sample | offsets | counts | IFD
    let header_len = 8u32;
    let data_len = (tile_bytes * tile_count) as u32;
    let bps_offset = header_len + data_len;
    let offsets_offset = bps_offset + 6 + 2; // keep word alignment
    let counts_offset = offsets_offset + 4 * tile_count as u32;
    let ifd_offset = counts_offset + 4 * tile_count as u32;

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(b"II")?;
    out.write_all(&42u16.to_le_bytes())?;
    out.write_all(&ifd_offset.to_le_bytes())?;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut buf = vec![0u8; tile_bytes];
            for dy in 0..tile {
                for dx in 0..tile {
                    let (x, y) = (tx * tile + dx, ty * tile + dy);
                    let rgb = if x < width && y < height { pixel(x, y) } else { [0, 0, 0] };
                    let i = ((dy * tile + dx) * 3) as usize;
                    buf[i..i + 3].copy_from_slice(&rgb);
                }
            }
            out.write_all(&buf)?;
        }
    }

    for _ in 0..3 {
        out.write_all(&8u16.to_le_bytes())?;
    }
    out.write_all(&[0, 0])?;
    for i in 0..tile_count as u32 {
        out.write_all(&(header_len + i * tile_bytes as u32).to_le_bytes())?;
    }
    for _ in 0..tile_count {
        out.write_all(&(tile_bytes as u32).to_le_bytes())?;
    }

    // (tag, type, count, value)  type 3 = SHORT, 4 = LONG
    let entries: [(u16, u16, u32, u32); 11] = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 3, bps_offset),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (277, 3, 1, 3),
        (284, 3, 1, 1),
        (322, 4, 1, tile),
        (323, 4, 1, tile),
        (324, 4, tile_count as u32, if tile_count == 1 { header_len } else { offsets_offset }),
        (325, 4, tile_count as u32, if tile_count == 1 { tile_bytes as u32 } else { counts_offset }),
    ];
    out.write_all(&(entries.len() as u16).to_le_bytes())?;
    for (tag, ty, count, value) in entries {
        out.write_all(&tag.to_le_bytes())?;
        out.write_all(&ty.to_le_bytes())?;
        out.write_all(&count.to_le_bytes())?;
        if ty == 3 && count == 1 {
            out.write_all(&(value as u16).to_le_bytes())?;
            out.write_all(&[0, 0])?;
        } else {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.write_all(&0u32.to_le_bytes())?;
    out.flush()
}
