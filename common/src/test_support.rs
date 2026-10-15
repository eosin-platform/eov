//! Test-only helpers shared by unit tests in this crate.
//!
//! [`write_tiff`] produces small, uncompressed TIFF files that OpenSlide opens
//! with its generic-tiff driver or, with an `Aperio` description, with its
//! Aperio driver (which adds associated images). Tests use them instead of
//! real slides so they run everywhere.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One tiled pyramid level.
#[derive(Clone, Debug)]
pub struct TiffLevel {
    pub width: u32,
    pub height: u32,
    pub tile: u32,
    /// Tiles whose data offset points past the end of the file, so reading
    /// them is an I/O error that poisons the OpenSlide handle.
    pub broken_tiles: Vec<(u32, u32)>,
    /// Tiles with a byte count of zero, which OpenSlide renders transparent.
    pub missing_tiles: Vec<(u32, u32)>,
}

impl TiffLevel {
    /// `tile` must be a multiple of 32: libtiff 4.5 rejects uncompressed
    /// tiles whose byte size is not a multiple of 1 KiB when the file is not
    /// memory-mapped (as OpenSlide opens it).
    pub fn new(width: u32, height: u32, tile: u32) -> Self {
        assert!(
            tile.is_multiple_of(32),
            "test TIFF tile sizes must be multiples of 32"
        );
        Self {
            width,
            height,
            tile,
            broken_tiles: Vec::new(),
            missing_tiles: Vec::new(),
        }
    }

    pub fn with_broken_tile(mut self, tx: u32, ty: u32) -> Self {
        self.broken_tiles.push((tx, ty));
        self
    }

    pub fn with_missing_tile(mut self, tx: u32, ty: u32) -> Self {
        self.missing_tiles.push((tx, ty));
        self
    }
}

/// A stripped (non-tiled) image, used for Aperio associated images.
#[derive(Clone, Debug)]
pub struct TiffStripped {
    pub width: u32,
    pub height: u32,
    pub description: String,
    pub color: [u8; 3],
}

#[derive(Clone, Debug, Default)]
pub struct TiffSpec {
    /// Tiled levels, largest first.
    pub levels: Vec<TiffLevel>,
    /// ImageDescription of the first directory.
    pub description: Option<String>,
    /// ICC profile of the first directory.
    pub icc_profile: Option<Vec<u8>>,
    /// X/Y resolution in pixels per centimetre (OpenSlide derives MPP).
    pub pixels_per_cm: Option<u32>,
    /// TIFF DateTime of the first directory.
    pub date_time: Option<String>,
    /// Stripped image written as the second directory (Aperio thumbnail).
    pub second_directory: Option<TiffStripped>,
    /// Stripped images written after all levels (Aperio label/macro).
    pub trailing: Vec<TiffStripped>,
}

impl TiffSpec {
    /// A single-level generic TIFF.
    pub fn single(width: u32, height: u32, tile: u32) -> Self {
        Self {
            levels: vec![TiffLevel::new(width, height, tile)],
            ..Self::default()
        }
    }

    /// A generic TIFF pyramid that halves `count - 1` times.
    pub fn pyramid(width: u32, height: u32, tile: u32, count: u32) -> Self {
        let levels = (0..count)
            .map(|i| TiffLevel::new((width >> i).max(1), (height >> i).max(1), tile))
            .collect();
        Self {
            levels,
            ..Self::default()
        }
    }

    /// An Aperio-style SVS with two levels (downsample `downsample`), a
    /// thumbnail, a 24x16 label and a 48x16 macro.
    pub fn aperio(width: u32, height: u32, tile: u32, downsample: u32) -> Self {
        let description = format!(
            "Aperio Image Library v12.0.11\n{width}x{height} ({tile}x{tile}) RAW|AppMag = 2.5\
             |MPP = 0.4942|Date = 08/11/16|Time = 10:18:38|Title = Case 7 | H&E"
        );
        Self {
            levels: vec![
                TiffLevel::new(width, height, tile),
                TiffLevel::new(width / downsample, height / downsample, tile),
            ],
            description: Some(description),
            second_directory: Some(TiffStripped {
                width: (width / 8).max(1),
                height: (height / 8).max(1),
                description: "Aperio Image Library v12.0.11\nthumbnail".to_string(),
                color: [200, 180, 210],
            }),
            trailing: vec![
                TiffStripped {
                    width: 24,
                    height: 16,
                    description: "Aperio Image Library v12.0.11\nlabel 24x16".to_string(),
                    color: [10, 20, 30],
                },
                TiffStripped {
                    width: 48,
                    height: 16,
                    description: "Aperio Image Library v12.0.11\nmacro 48x16".to_string(),
                    color: [40, 50, 60],
                },
            ],
            ..Self::default()
        }
    }
}

/// Writes a minimal uncompressed, tiled, 8-bit RGB TIFF that OpenSlide opens
/// with its generic-tiff driver. `pixel(x, y)` supplies the colour of every
/// pixel; `tile` must be a multiple of 32.
pub fn write_tiled_tiff(
    path: &Path,
    width: u32,
    height: u32,
    tile: u32,
    pixel: impl Fn(u32, u32) -> [u8; 3],
) -> std::io::Result<()> {
    write_tiff(path, &TiffSpec::single(width, height, tile), pixel)
}

enum Value {
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(u32, u32),
    Ascii(String),
    Undefined(Vec<u8>),
}

impl Value {
    fn type_and_count(&self) -> (u16, u32) {
        match self {
            Value::Short(v) => (3, v.len() as u32),
            Value::Long(v) => (4, v.len() as u32),
            Value::Rational(..) => (5, 1),
            Value::Ascii(s) => (2, s.len() as u32 + 1),
            Value::Undefined(v) => (7, v.len() as u32),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Value::Short(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Long(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Rational(n, d) => n.to_le_bytes().into_iter().chain(d.to_le_bytes()).collect(),
            Value::Ascii(s) => s.bytes().chain(std::iter::once(0)).collect(),
            Value::Undefined(v) => v.clone(),
        }
    }
}

/// Writes `spec` to `path`. `pixel(x, y)` gives the level-0 colour; lower
/// levels sample it at the corresponding level-0 position.
pub fn write_tiff(
    path: &Path,
    spec: &TiffSpec,
    pixel: impl Fn(u32, u32) -> [u8; 3],
) -> std::io::Result<()> {
    assert!(!spec.levels.is_empty(), "a TIFF needs at least one level");
    let base_w = spec.levels[0].width;
    let base_h = spec.levels[0].height;

    // Pixel data first, directories at the end.
    let mut data: Vec<u8> = Vec::new();
    let header_len = 8u32;

    struct DirPlan {
        tags: Vec<(u16, Value)>,
    }
    let mut plans: Vec<DirPlan> = Vec::new();
    let mut broken_slots: Vec<(usize, usize)> = Vec::new(); // (dir, tile index)

    let mut level_plan = |index: usize, level: &TiffLevel, data: &mut Vec<u8>| -> DirPlan {
        let tiles_x = level.width.div_ceil(level.tile);
        let tiles_y = level.height.div_ceil(level.tile);
        assert!(
            level.broken_tiles.is_empty() || tiles_x * tiles_y > 1,
            "broken tiles need a level with more than one tile"
        );
        let tile_bytes = (level.tile * level.tile * 3) as usize;
        let sx = base_w as f64 / level.width as f64;
        let sy = base_h as f64 / level.height as f64;
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                if level.missing_tiles.contains(&(tx, ty)) {
                    offsets.push(0);
                    counts.push(0);
                    continue;
                }
                let mut buf = vec![0u8; tile_bytes];
                for dy in 0..level.tile {
                    for dx in 0..level.tile {
                        let (x, y) = (tx * level.tile + dx, ty * level.tile + dy);
                        if x < level.width && y < level.height {
                            let x0 = ((x as f64 + 0.5) * sx) as u32;
                            let y0 = ((y as f64 + 0.5) * sy) as u32;
                            let rgb = pixel(x0.min(base_w - 1), y0.min(base_h - 1));
                            let i = ((dy * level.tile + dx) * 3) as usize;
                            buf[i..i + 3].copy_from_slice(&rgb);
                        }
                    }
                }
                if level.broken_tiles.contains(&(tx, ty)) {
                    broken_slots.push((index, offsets.len()));
                }
                offsets.push(header_len + data.len() as u32);
                counts.push(tile_bytes as u32);
                data.extend_from_slice(&buf);
            }
        }
        let mut tags = vec![
            (254, Value::Long(vec![u32::from(index > 0)])),
            (256, Value::Long(vec![level.width])),
            (257, Value::Long(vec![level.height])),
            (258, Value::Short(vec![8, 8, 8])),
            (259, Value::Short(vec![1])),
            (262, Value::Short(vec![2])),
            (277, Value::Short(vec![3])),
            (284, Value::Short(vec![1])),
            (322, Value::Long(vec![level.tile])),
            (323, Value::Long(vec![level.tile])),
            (324, Value::Long(offsets)),
            (325, Value::Long(counts)),
        ];
        if index == 0 {
            if let Some(description) = &spec.description {
                tags.push((270, Value::Ascii(description.clone())));
            }
            if let Some(ppcm) = spec.pixels_per_cm {
                tags.push((282, Value::Rational(ppcm, 1)));
                tags.push((283, Value::Rational(ppcm, 1)));
                tags.push((296, Value::Short(vec![3])));
            }
            if let Some(date_time) = &spec.date_time {
                tags.push((306, Value::Ascii(date_time.clone())));
            }
            if let Some(icc) = &spec.icc_profile {
                tags.push((34675, Value::Undefined(icc.clone())));
            }
        }
        DirPlan { tags }
    };

    let stripped_plan = |image: &TiffStripped, data: &mut Vec<u8>| -> DirPlan {
        let offset = header_len + data.len() as u32;
        for _ in 0..image.width * image.height {
            data.extend_from_slice(&image.color);
        }
        DirPlan {
            tags: vec![
                (254, Value::Long(vec![0])),
                (256, Value::Long(vec![image.width])),
                (257, Value::Long(vec![image.height])),
                (258, Value::Short(vec![8, 8, 8])),
                (259, Value::Short(vec![1])),
                (262, Value::Short(vec![2])),
                (270, Value::Ascii(image.description.clone())),
                (273, Value::Long(vec![offset])),
                (277, Value::Short(vec![3])),
                (278, Value::Long(vec![image.height])),
                (279, Value::Long(vec![image.width * image.height * 3])),
                (284, Value::Short(vec![1])),
            ],
        }
    };

    for (index, level) in spec.levels.iter().enumerate() {
        if index == 1
            && let Some(second) = &spec.second_directory
        {
            plans.push(stripped_plan(second, &mut data));
        }
        let plan = level_plan(plans.len(), level, &mut data);
        plans.push(plan);
    }
    if spec.levels.len() == 1
        && let Some(second) = &spec.second_directory
    {
        plans.push(stripped_plan(second, &mut data));
    }
    for image in &spec.trailing {
        plans.push(stripped_plan(image, &mut data));
    }
    if data.len() % 2 == 1 {
        data.push(0);
    }

    // Lay out the directories and their out-of-line values.
    let mut tail: Vec<u8> = Vec::new();
    let tail_start = header_len + data.len() as u32;
    let mut dir_offsets = Vec::new();
    let mut patched: Vec<(usize, usize, usize)> = Vec::new(); // (tail pos of offsets array, dir, tile)
    for (dir_index, plan) in plans.iter_mut().enumerate() {
        plan.tags.sort_by_key(|(tag, _)| *tag);
        // Out-of-line values first.
        let mut value_fields: Vec<[u8; 4]> = Vec::new();
        for (tag, value) in &plan.tags {
            let bytes = value.bytes();
            if bytes.len() <= 4 {
                let mut field = [0u8; 4];
                field[..bytes.len()].copy_from_slice(&bytes);
                value_fields.push(field);
            } else {
                let at = tail_start + tail.len() as u32;
                if *tag == 324 {
                    for (dir, tile) in &broken_slots {
                        if *dir == dir_index {
                            patched.push((tail.len() + tile * 4, *dir, *tile));
                        }
                    }
                }
                tail.extend_from_slice(&bytes);
                if tail.len() % 2 == 1 {
                    tail.push(0);
                }
                value_fields.push(at.to_le_bytes());
            }
        }
        dir_offsets.push(tail_start + tail.len() as u32);
        tail.extend_from_slice(&(plan.tags.len() as u16).to_le_bytes());
        for ((tag, value), field) in plan.tags.iter().zip(&value_fields) {
            let (ty, count) = value.type_and_count();
            tail.extend_from_slice(&tag.to_le_bytes());
            tail.extend_from_slice(&ty.to_le_bytes());
            tail.extend_from_slice(&count.to_le_bytes());
            tail.extend_from_slice(field);
        }
        tail.extend_from_slice(&0u32.to_le_bytes()); // next IFD, patched below
    }
    // Chain the directories.
    for i in 0..dir_offsets.len().saturating_sub(1) {
        let next_field = (dir_offsets[i] - tail_start) as usize + 2 + plans[i].tags.len() * 12;
        tail[next_field..next_field + 4].copy_from_slice(&dir_offsets[i + 1].to_le_bytes());
    }
    // Point broken tiles past the end of the file.
    let file_len = tail_start + tail.len() as u32;
    for (pos, _, _) in patched {
        tail[pos..pos + 4].copy_from_slice(&(file_len + 1_000_000).to_le_bytes());
    }

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(b"II")?;
    out.write_all(&42u16.to_le_bytes())?;
    out.write_all(&dir_offsets[0].to_le_bytes())?;
    out.write_all(&data)?;
    out.write_all(&tail)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_files_with_every_feature() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("all.tiff");
        let mut spec = TiffSpec::aperio(64, 64, 32, 2);
        spec.levels[0] = spec.levels[0].clone().with_missing_tile(0, 0);
        spec.icc_profile = Some(vec![1, 2, 3, 4, 5, 6]);
        spec.date_time = Some("2024:01:02 03:04:05".to_string());
        spec.pixels_per_cm = Some(40_000);
        write_tiff(&path, &spec, |x, y| [x as u8, y as u8, 0]).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > 64 * 64 * 3);
    }
}
