//! Synthetic multi-level slides for colour, export and dataset tests.
//!
//! [`write_pyramid_tiff`] writes an uncompressed, tiled, pyramidal RGB TIFF
//! (one IFD per level) that OpenSlide opens with its generic-tiff driver,
//! optionally with a pixel size (resolution tags in centimetres, which
//! OpenSlide reports as `openslide.mpp-x/y`) and an `ImageDescription`.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Parameters of a synthetic pyramid.
#[derive(Clone, Debug)]
pub(crate) struct PyramidSpec {
    pub width: u32,
    pub height: u32,
    /// Tile size (a multiple of 16).
    pub tile: u32,
    /// Integer downsample of every level, starting with 1.
    pub downsamples: Vec<u32>,
    /// Level-0 pixel size in µm.
    pub mpp: Option<f64>,
    pub description: Option<String>,
}

impl PyramidSpec {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            tile: 64,
            downsamples: vec![1],
            mpp: None,
            description: None,
        }
    }

    pub(crate) fn levels(mut self, downsamples: &[u32]) -> Self {
        self.downsamples = downsamples.to_vec();
        self
    }

    pub(crate) fn mpp(mut self, mpp: f64) -> Self {
        self.mpp = Some(mpp);
        self
    }

    pub(crate) fn description(mut self, text: &str) -> Self {
        self.description = Some(text.to_string());
        self
    }
}

struct Entry {
    tag: u16,
    ty: u16,
    count: u32,
    value: u32,
}

fn write_u16(out: &mut impl Write, v: u16) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

fn write_u32(out: &mut impl Write, v: u32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

/// Writes the pyramid. `pixel(x, y)` gives the level-0 colour; reduced
/// levels sample it at the centre of each block.
pub(crate) fn write_pyramid_tiff(
    path: &Path,
    spec: &PyramidSpec,
    pixel: impl Fn(u32, u32) -> [u8; 3],
) -> io::Result<()> {
    assert!(
        spec.tile.is_multiple_of(16),
        "TIFF tile sizes must be multiples of 16"
    );
    assert_eq!(spec.downsamples.first(), Some(&1));
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(b"II")?;
    write_u16(&mut out, 42)?;
    let mut pending_ifd_pointer: u64 = 4;
    write_u32(&mut out, 0)?; // patched below

    for (level_index, &ds) in spec.downsamples.iter().enumerate() {
        let lw = spec.width.div_ceil(ds);
        let lh = spec.height.div_ceil(ds);
        let tiles_x = lw.div_ceil(spec.tile);
        let tiles_y = lh.div_ceil(spec.tile);
        let tile_bytes = (spec.tile * spec.tile * 3) as usize;
        let mut offsets = Vec::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                offsets.push(out.stream_position()? as u32);
                let mut buf = vec![0u8; tile_bytes];
                for dy in 0..spec.tile {
                    for dx in 0..spec.tile {
                        let (x, y) = (tx * spec.tile + dx, ty * spec.tile + dy);
                        if x >= lw || y >= lh {
                            continue;
                        }
                        let sx = (x * ds + ds / 2).min(spec.width - 1);
                        let sy = (y * ds + ds / 2).min(spec.height - 1);
                        let i = ((dy * spec.tile + dx) * 3) as usize;
                        buf[i..i + 3].copy_from_slice(&pixel(sx, sy));
                    }
                }
                out.write_all(&buf)?;
            }
        }
        let count = offsets.len() as u32;
        let offsets_at = out.stream_position()? as u32;
        for o in &offsets {
            write_u32(&mut out, *o)?;
        }
        let counts_at = out.stream_position()? as u32;
        for _ in &offsets {
            write_u32(&mut out, tile_bytes as u32)?;
        }
        let bps_at = out.stream_position()? as u32;
        for _ in 0..3 {
            write_u16(&mut out, 8)?;
        }
        let mut entries = vec![
            Entry {
                tag: 256,
                ty: 4,
                count: 1,
                value: lw,
            },
            Entry {
                tag: 257,
                ty: 4,
                count: 1,
                value: lh,
            },
            Entry {
                tag: 258,
                ty: 3,
                count: 3,
                value: bps_at,
            },
            Entry {
                tag: 259,
                ty: 3,
                count: 1,
                value: 1,
            },
            Entry {
                tag: 262,
                ty: 3,
                count: 1,
                value: 2,
            },
            Entry {
                tag: 277,
                ty: 3,
                count: 1,
                value: 3,
            },
            Entry {
                tag: 284,
                ty: 3,
                count: 1,
                value: 1,
            },
            Entry {
                tag: 322,
                ty: 4,
                count: 1,
                value: spec.tile,
            },
            Entry {
                tag: 323,
                ty: 4,
                count: 1,
                value: spec.tile,
            },
            Entry {
                tag: 324,
                ty: 4,
                count,
                value: if count == 1 { offsets[0] } else { offsets_at },
            },
            Entry {
                tag: 325,
                ty: 4,
                count,
                value: if count == 1 {
                    tile_bytes as u32
                } else {
                    counts_at
                },
            },
        ];
        if level_index > 0 {
            // NewSubfileType = reduced-resolution image; OpenSlide's
            // generic-tiff driver only treats such IFDs as pyramid levels.
            entries.push(Entry {
                tag: 254,
                ty: 4,
                count: 1,
                value: 1,
            });
        }
        if level_index == 0 {
            if let Some(text) = &spec.description {
                assert!(
                    text.len() > 3,
                    "short descriptions would have to be stored inline"
                );
                let at = out.stream_position()? as u32;
                out.write_all(text.as_bytes())?;
                out.write_all(&[0])?;
                entries.push(Entry {
                    tag: 270,
                    ty: 2,
                    count: text.len() as u32 + 1,
                    value: at,
                });
            }
            if let Some(mpp) = spec.mpp {
                // Pixels per centimetre as a rational with denominator 1000.
                let per_cm = (10_000.0 / mpp * 1000.0).round() as u32;
                let at = out.stream_position()? as u32;
                write_u32(&mut out, per_cm)?;
                write_u32(&mut out, 1000)?;
                write_u32(&mut out, per_cm)?;
                write_u32(&mut out, 1000)?;
                entries.push(Entry {
                    tag: 282,
                    ty: 5,
                    count: 1,
                    value: at,
                });
                entries.push(Entry {
                    tag: 283,
                    ty: 5,
                    count: 1,
                    value: at + 8,
                });
                entries.push(Entry {
                    tag: 296,
                    ty: 3,
                    count: 1,
                    value: 3,
                });
            }
        }
        entries.sort_by_key(|e| e.tag);
        if out.stream_position()? % 2 == 1 {
            out.write_all(&[0])?;
        }
        let ifd_at = out.stream_position()?;
        // Link the previous IFD (or the header) to this one.
        out.seek(SeekFrom::Start(pending_ifd_pointer))?;
        write_u32(&mut out, ifd_at as u32)?;
        out.seek(SeekFrom::Start(ifd_at))?;
        write_u16(&mut out, entries.len() as u16)?;
        for e in &entries {
            write_u16(&mut out, e.tag)?;
            write_u16(&mut out, e.ty)?;
            write_u32(&mut out, e.count)?;
            if e.ty == 3 && e.count == 1 {
                write_u16(&mut out, e.value as u16)?;
                write_u16(&mut out, 0)?;
            } else {
                write_u32(&mut out, e.value)?;
            }
        }
        pending_ifd_pointer = out.stream_position()?;
        write_u32(&mut out, 0)?;
    }
    out.flush()
}

/// A deterministic synthetic "H&E tissue on glass" image: a disc of tissue
/// with nuclei-like hematoxylin-rich spots on an eosin-rich stroma, on a
/// background of `glass`.
pub(crate) fn he_tissue_pixel(x: u32, y: u32, width: u32, height: u32, glass: [u8; 3]) -> [u8; 3] {
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let r = (width.min(height) as f32) * 0.35;
    let dx = x as f32 - cx;
    let dy = y as f32 - cy;
    if dx * dx + dy * dy > r * r {
        return glass;
    }
    let h = [0.644211f32, 0.716556, 0.266844];
    let e = [0.092789f32, 0.954111, 0.283111];
    let nucleus = ((x / 9) * 7 + (y / 9) * 13).is_multiple_of(5);
    let a = ((x * 31 + y * 17) % 11) as f32 / 10.0;
    let b = ((x * 7 + y * 29) % 13) as f32 / 12.0;
    // Nuclei: hematoxylin with a trace of eosin; stroma: eosin with a trace
    // of hematoxylin.
    let (ch, ce) = if nucleus {
        (0.9 + 0.4 * a, 0.08 * b)
    } else {
        (0.25 * a, 0.6 + 0.5 * b)
    };
    let mut out = [0u8; 3];
    for c in 0..3 {
        let od = h[c] * ch + e[c] * ce;
        out[c] = (glass[c] as f32 * (-od).exp()).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WsiFile;

    #[test]
    fn openslide_reads_synthetic_pyramids_with_mpp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pyramid.tif");
        let spec = PyramidSpec::new(512, 384)
            .levels(&[1, 2, 4])
            .mpp(0.5)
            .description("H&E test");
        write_pyramid_tiff(&path, &spec, |x, y| [(x % 256) as u8, (y % 256) as u8, 128]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let props = wsi.properties();
        assert_eq!((props.width, props.height), (512, 384));
        assert_eq!(wsi.level_count(), 3);
        assert_eq!(wsi.level(2).unwrap().width, 128);
        let mpp = props.mpp_x.expect("mpp parsed from TIFF resolution");
        assert!((mpp - 0.5).abs() < 1e-3, "{mpp}");
        let px = wsi.read_region(100, 50, 0, 1, 1).unwrap();
        assert_eq!(&px[..3], &[100, 50, 128]);
        // Level 1 pixel (10, 5) samples level-0 (21, 11).
        let px = wsi.read_region(20, 10, 1, 1, 1).unwrap();
        assert_eq!(&px[..3], &[21, 11, 128]);
    }
}
