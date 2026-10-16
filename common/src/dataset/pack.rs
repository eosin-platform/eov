//! Packing a dataset into one zarr store or HDF5 file (AUDIT §14.5), for
//! training code that reads arrays rather than many files:
//! `eov dataset pack <dir> [--format zarr|hdf5]`.
//!
//! [`pack`] reads the dataset's patch records (`metadata.jsonl`, else
//! `metadata.csv`, else `metadata.json`), decodes every patch and mask, and
//! writes one of:
//!
//! * [`PackFormat::Zarr`] (the default): `<dir>/patches.zarr`, a zarr v2
//!   store with zstd chunks and consolidated metadata (`.zmetadata`),
//!   readable with zarr-python (`zarr.open("patches.zarr")`).
//! * [`PackFormat::Hdf5`]: `<dir>/patches.h5`, written by the HDF5 C
//!   library (1.10 or newer), which eov loads at run time
//!   ([`crate::hdf5`]); readable with h5py (`h5py.File("patches.h5")`).
//!   Chunks are deflated (level 4), `coords` and `slide` with the shuffle
//!   filter first. The file uses the HDF5 1.8 format.
//!
//! Both hold the same arrays:
//!
//! | Array    | Shape          | Type   | Contents |
//! |----------|----------------|--------|----------|
//! | `images` | (N, H, W, 3)   | uint8  | the patches, one chunk each |
//! | `masks`  | (N, H, W)      | uint8  | their masks, when the dataset has them |
//! | `coords` | (N, 2)         | int64  | level-0 x and y of each patch |
//! | `slide`  | (N,)           | int32  | index into the `slides` attribute |
//! | `label`  | (N,)           | uint8  | class index from `classes.json` (0: none) |
//!
//! The one-dimensional arrays (and `coords`) are chunked by 65 536 rows.
//! Patches keep the order of the records. The attributes of the store's
//! root group are the same in both: `slides` (per slide its id, stem, pixel
//! size, read level, downsample and size), `classes` (index → name),
//! `tile_size`, `patches`, `eov_version` and a `description`. HDF5 has no
//! nested attributes, so there `slides` and `classes` are JSON text, UTF-8
//! strings: `json.loads(f.attrs["slides"])`.
//!
//! The output is written under a temporary name (`<name>.<pid>.partial`)
//! and renamed when complete, replacing an earlier one; a cancelled or
//! failed pack leaves an earlier one as it was.

mod hdf5;
mod zarr;

use super::metadata::{CSV_FILE, JSON_FILE, JSONL_FILE, TileRecord, for_each_csv, for_each_jsonl};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The zarr store's directory name.
pub const ZARR_STORE: &str = "patches.zarr";
/// The HDF5 file's name.
pub const HDF5_FILE: &str = "patches.h5";
/// Rows of the one-dimensional arrays per chunk.
const ROW_CHUNK: usize = 65_536;

/// What [`pack`] writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PackFormat {
    /// `patches.zarr`, which needs nothing installed.
    #[default]
    Zarr,
    /// `patches.h5`, which needs the HDF5 library (1.10 or newer).
    Hdf5,
}

/// How [`pack`] packs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackOptions {
    pub format: PackFormat,
    /// The HDF5 library to write with (the file, or a folder with it),
    /// instead of searching for one; without it, `EOV_HDF5_LIBRARY` names
    /// it when set ([`crate::hdf5::LIBRARY_ENV`]).
    pub hdf5_library: Option<PathBuf>,
}

/// What [`pack`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct PackReport {
    /// The store or file written.
    pub store: PathBuf,
    pub patches: usize,
    pub tile_size: u32,
    pub slides: usize,
    pub with_masks: bool,
    /// The HDF5 library that wrote the file: its version and where it was
    /// loaded from.
    pub library: Option<String>,
}

/// Packs the dataset in `dir` as `options` say. Returns `Ok(None)` when
/// cancelled.
pub fn pack(
    dir: &Path,
    options: &PackOptions,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PackReport>> {
    match options.format {
        PackFormat::Zarr => zarr::pack(&Patches::read(dir)?, cancel),
        PackFormat::Hdf5 => {
            let search = crate::hdf5::LibrarySearch::from_env(options.hdf5_library.as_deref());
            hdf5::pack(&Patches::read(dir)?, &search, cancel)
        }
    }
}

/// Packs the dataset in `dir` into `dir/patches.zarr`. Returns `Ok(None)`
/// when cancelled.
pub fn pack_zarr(dir: &Path, cancel: &AtomicBool) -> std::io::Result<Option<PackReport>> {
    pack(dir, &PackOptions::default(), cancel)
}

/// A dataset's patch records, checked, and what every format writes of
/// them.
struct Patches {
    dir: PathBuf,
    records: Vec<TileRecord>,
    width: u32,
    height: u32,
    with_masks: bool,
    /// `classes.json` as name → index (index 0 is no class).
    classes: BTreeMap<String, u8>,
    /// The slides in order of first appearance, and each one's index.
    slides: Vec<Value>,
    slide_index: BTreeMap<String, i32>,
}

/// One patch as stored: its pixels (H, W, 3) and, when the dataset has
/// masks, its mask (H, W), row by row.
struct Patch {
    image: Vec<u8>,
    mask: Option<Vec<u8>>,
}

impl Patches {
    fn read(dir: &Path) -> std::io::Result<Self> {
        let records = read_records(dir)?;
        if records.is_empty() {
            return Err(std::io::Error::other(format!(
                "{} has no patch records (metadata.jsonl, metadata.csv or metadata.json)",
                dir.display()
            )));
        }
        let (width, height) = (records[0].width, records[0].height);
        if let Some(odd) = records
            .iter()
            .find(|r| (r.width, r.height) != (width, height))
        {
            return Err(std::io::Error::other(format!(
                "patches have different sizes ({width}×{height} and {}×{}); pack datasets of one tile size",
                odd.width, odd.height
            )));
        }
        let with_masks = records.iter().all(|r| r.mask_path.is_some());
        let mut slide_index: BTreeMap<String, i32> = BTreeMap::new();
        let mut slides: Vec<Value> = Vec::new();
        for record in &records {
            if !slide_index.contains_key(&record.slide_id) {
                slide_index.insert(record.slide_id.clone(), slides.len() as i32);
                slides.push(json!({
                    "id": record.slide_id,
                    "stem": record.slide_stem,
                    "mpp_x": record.mpp_x,
                    "mpp_y": record.mpp_y,
                    "output_mpp_x": record.output_mpp_x,
                    "output_mpp_y": record.output_mpp_y,
                    "level": record.level,
                    "downsample": record.downsample,
                    "width": record.slide_width,
                    "height": record.slide_height,
                }));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            classes: read_classes(dir),
            records,
            width,
            height,
            with_masks,
            slides,
            slide_index,
        })
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    /// Decodes patch `index` (and its mask), checking its size.
    fn decode(&self, index: usize) -> std::io::Result<Patch> {
        let record = &self.records[index];
        let (width, height) = (self.width, self.height);
        let image = image::open(self.dir.join(&record.tile_path))
            .map_err(|e| std::io::Error::other(format!("{}: {e}", record.tile_path)))?
            .to_rgb8();
        if image.dimensions() != (width, height) {
            return Err(std::io::Error::other(format!(
                "{} is {}×{}, its record says {width}×{height}",
                record.tile_path,
                image.width(),
                image.height()
            )));
        }
        let mask = match (&record.mask_path, self.with_masks) {
            (Some(path), true) => {
                let mask = image::open(self.dir.join(path))
                    .map_err(|e| std::io::Error::other(format!("{path}: {e}")))?
                    .to_luma8();
                if mask.dimensions() != (width, height) {
                    return Err(std::io::Error::other(format!(
                        "{path} is {}×{}, its patch is {width}×{height}",
                        mask.width(),
                        mask.height()
                    )));
                }
                Some(mask.into_raw())
            }
            _ => None,
        };
        Ok(Patch {
            image: image.into_raw(),
            mask,
        })
    }

    /// Level-0 x and y of every patch, in pairs.
    fn coords(&self) -> Vec<i64> {
        self.records
            .iter()
            .flat_map(|r| [r.x as i64, r.y as i64])
            .collect()
    }

    /// Each patch's index into the slides.
    fn slide_indices(&self) -> Vec<i32> {
        self.records
            .iter()
            .map(|r| self.slide_index[&r.slide_id])
            .collect()
    }

    /// Each patch's class index (0: none).
    fn labels(&self) -> Vec<u8> {
        self.records
            .iter()
            .map(|r| {
                r.label
                    .as_ref()
                    .and_then(|label| self.classes.get(label).copied())
                    .unwrap_or(0)
            })
            .collect()
    }

    /// The root group's attributes, `note` ending the description.
    fn attributes(&self, note: &str) -> Map<String, Value> {
        let class_names: BTreeMap<String, String> = self
            .classes
            .iter()
            .map(|(name, index)| (index.to_string(), name.clone()))
            .collect();
        let description = format!(
            "Patches of an eov dataset: images (N, H, W, 3) uint8; coords (N, 2) level-0 x, y; \
             slide (N,) index into slides; label (N,) index into classes (0: none); \
             masks (N, H, W) when present.{note}"
        );
        let Value::Object(attributes) = json!({
            "eov_version": env!("CARGO_PKG_VERSION"),
            "description": description,
            "tile_size": self.width,
            "patches": self.len(),
            "slides": self.slides,
            "classes": class_names,
        }) else {
            unreachable!("an object literal")
        };
        attributes
    }

    fn report(&self, store: PathBuf, library: Option<String>) -> PackReport {
        PackReport {
            store,
            patches: self.len(),
            tile_size: self.width,
            slides: self.slides.len(),
            with_masks: self.with_masks,
            library,
        }
    }
}

fn read_records(dir: &Path) -> std::io::Result<Vec<TileRecord>> {
    let mut records = Vec::new();
    if dir.join(JSONL_FILE).is_file() {
        for_each_jsonl::<TileRecord>(&dir.join(JSONL_FILE), |record| {
            records.push(record);
            Ok(())
        })?;
    } else if dir.join(CSV_FILE).is_file() {
        for_each_csv::<TileRecord>(&dir.join(CSV_FILE), |record| {
            records.push(record);
            Ok(())
        })?;
    } else if dir.join(JSON_FILE).is_file() {
        let text = std::fs::read_to_string(dir.join(JSON_FILE))?;
        records = serde_json::from_str(&text).map_err(std::io::Error::other)?;
    }
    Ok(records)
}

/// `classes.json` as name → index (index 0 is no class).
fn read_classes(dir: &Path) -> BTreeMap<String, u8> {
    let Ok(text) = std::fs::read_to_string(dir.join("classes.json")) else {
        return BTreeMap::new();
    };
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) else {
        return BTreeMap::new();
    };
    map.into_iter()
        .filter_map(|(index, name)| Some((name.as_str()?.to_string(), index.parse::<u8>().ok()?)))
        .collect()
}

/// Writes `dir/name` through a temporary name next to it: `write` fills the
/// path it is given (a file or a folder) and returns `Ok(None)` when
/// cancelled. A complete result replaces an earlier one; anything else is
/// removed, leaving an earlier one as it was.
fn replace_when_complete(
    dir: &Path,
    name: &str,
    write: impl FnOnce(&Path) -> std::io::Result<Option<()>>,
) -> std::io::Result<Option<PathBuf>> {
    let complete = dir.join(name);
    let partial = dir.join(format!("{name}.{}.partial", std::process::id()));
    remove(&partial)?;
    let result = write(&partial);
    if !matches!(result, Ok(Some(()))) {
        let _ = remove(&partial);
        return result.map(|_| None);
    }
    if complete.is_dir() {
        std::fs::remove_dir_all(&complete)?;
    }
    // Replaces a file in one step.
    std::fs::rename(&partial, &complete)?;
    Ok(Some(complete))
}

/// Removes a file or a folder, if there is one.
fn remove(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(path),
        Ok(_) => std::fs::remove_file(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        AnnotationOptions, AnnotationSource, DatasetControl, DatasetPatchesConfig,
        DatasetPatchesOptions, MaskOutput, run_dataset,
    };

    /// A dataset of 8 patches of 64 px from one slide, without masks.
    pub(super) fn small_dataset(root: &Path) -> PathBuf {
        let slide = root.join("s.tif");
        crate::test_support::write_tiled_tiff(&slide, 256, 128, 64, |x, y| {
            [(x % 251) as u8, (y % 251) as u8, 77]
        })
        .unwrap();
        let out = root.join("out");
        let report = run_dataset(
            &DatasetPatchesConfig {
                inputs: vec![slide],
                output_dir: out.clone(),
                tile_size: 64,
                stride: 64,
                metadata_format: None,
                threads: 2,
                white_threshold: None,
            },
            &DatasetPatchesOptions {
                hash_files: false,
                ..DatasetPatchesOptions::default()
            },
            &DatasetControl::new(),
        )
        .unwrap();
        assert_eq!(report.total_tiles, 8);
        out
    }

    /// A dataset of two slides (6 and 4 patches of 32 px), labelled from
    /// annotations with multiclass masks.
    pub(super) fn labelled_dataset(root: &Path) -> PathBuf {
        let mut inputs = Vec::new();
        for (name, width, colour) in [("a", 96, 40), ("b", 64, 200)] {
            let slide = root.join(format!("{name}.tif"));
            crate::test_support::write_tiled_tiff(&slide, width, 64, 32, move |x, y| {
                [colour, (x * 2 % 251) as u8, (y * 3 % 251) as u8]
            })
            .unwrap();
            // Tumor in the left half, Stroma at the bottom right.
            std::fs::write(
                root.join(format!("{name}.geojson")),
                format!(
                    r#"{{"type":"FeatureCollection","features":[
                    {{"type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[0,0],[40,0],[40,64],[0,64],[0,0]]]}},
                     "properties":{{"classification":{{"name":"Tumor"}}}}}},
                    {{"type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[40,32],[{width},32],[{width},64],[40,64],[40,32]]]}},
                     "properties":{{"classification":{{"name":"Stroma"}}}}}}]}}"#
                ),
            )
            .unwrap();
            inputs.push(slide);
        }
        let out = root.join("labelled");
        let report = run_dataset(
            &DatasetPatchesConfig {
                inputs,
                output_dir: out.clone(),
                tile_size: 32,
                stride: 32,
                metadata_format: None,
                threads: 2,
                white_threshold: None,
            },
            &DatasetPatchesOptions {
                hash_files: false,
                annotations: Some(AnnotationOptions {
                    source: AnnotationSource::Sidecar,
                    restrict_to_annotations: false,
                    masks: Some(MaskOutput::Multiclass),
                    ..AnnotationOptions::default()
                }),
                ..DatasetPatchesOptions::default()
            },
            &DatasetControl::new(),
        )
        .unwrap();
        assert_eq!(report.total_tiles, 10);
        out
    }

    #[test]
    fn records_are_read_in_order_with_slides_labels_and_masks() {
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let patches = Patches::read(&out).unwrap();
        assert_eq!(patches.len(), 10);
        assert!(patches.with_masks);
        assert_eq!(patches.slides.len(), 2);
        // background, Tumor and Stroma.
        assert_eq!(patches.classes.len(), 3);
        let slides = patches.slide_indices();
        assert!(slides.contains(&0) && slides.contains(&1));
        // Tumor, Stroma, and none for the patches outside both.
        let labels = patches.labels();
        assert!(labels.iter().all(|&label| label <= 2));
        assert!(labels.contains(&0) && labels.contains(&1) && labels.contains(&2));
        for index in 0..patches.len() {
            let patch = patches.decode(index).unwrap();
            assert_eq!(patch.image.len(), 32 * 32 * 3);
            assert_eq!(patch.mask.unwrap().len(), 32 * 32);
        }
        let coords = patches.coords();
        assert_eq!(coords.len(), 20);
        for (index, record) in patches.records.iter().enumerate() {
            assert_eq!(
                (coords[2 * index], coords[2 * index + 1]),
                (record.x as i64, record.y as i64)
            );
        }
        let attributes = patches.attributes("");
        assert_eq!(attributes["tile_size"], 32);
        assert_eq!(attributes["patches"], 10);
        assert_eq!(attributes["classes"].as_object().unwrap().len(), 3);
    }

    #[test]
    fn a_mask_of_another_size_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let patches = Patches::read(&out).unwrap();
        let mask = out.join(patches.records[3].mask_path.as_ref().unwrap());
        image::GrayImage::new(16, 32).save(&mask).unwrap();
        let error = patches.decode(3).err().unwrap();
        assert!(error.to_string().contains("16×32"), "{error}");
    }

    #[test]
    fn folders_without_records_are_refused() {
        let temp = tempfile::tempdir().unwrap();
        assert!(pack_zarr(temp.path(), &AtomicBool::new(false)).is_err());
        let options = PackOptions {
            format: PackFormat::Hdf5,
            hdf5_library: None,
        };
        assert!(pack(temp.path(), &options, &AtomicBool::new(false)).is_err());
    }
}
