//! Packing a dataset into one zarr v2 store (AUDIT §14.5), for training code
//! that reads arrays rather than many files: `eov dataset pack <dir>`.
//!
//! [`pack_zarr`] reads the dataset's patch records (`metadata.jsonl`, else
//! `metadata.csv`, else `metadata.json`), decodes every patch and mask, and
//! writes `<dir>/patches.zarr`, readable with zarr-python
//! (`zarr.open("patches.zarr")`):
//!
//! | Array    | Shape          | Type   | Contents |
//! |----------|----------------|--------|----------|
//! | `images` | (N, H, W, 3)   | uint8  | the patches, one zstd chunk each |
//! | `masks`  | (N, H, W)      | uint8  | their masks, when the dataset has them |
//! | `coords` | (N, 2)         | int64  | level-0 x and y of each patch |
//! | `slide`  | (N,)           | int32  | index into the `slides` attribute |
//! | `label`  | (N,)           | uint8  | class index from `classes.json` (0: none) |
//!
//! Patches keep the order of the records. The group's attributes list the
//! slides (id, stem, pixel size, read level and downsample), the classes,
//! the tile size and eov's version; `.zmetadata` holds the consolidated
//! metadata. The store is written under a temporary name and renamed when
//! complete, replacing an earlier one.

use super::metadata::{CSV_FILE, JSON_FILE, JSONL_FILE, TileRecord, for_each_csv, for_each_jsonl};
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// The store's directory name.
pub const ZARR_STORE: &str = "patches.zarr";
/// zstd level of the chunks.
const ZSTD_LEVEL: i32 = 3;
/// Rows of the one-dimensional arrays per chunk.
const ROW_CHUNK: usize = 65_536;

/// What [`pack_zarr`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct PackReport {
    pub store: PathBuf,
    pub patches: usize,
    pub tile_size: u32,
    pub slides: usize,
    pub with_masks: bool,
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

fn write_json(path: &Path, value: &Value) -> std::io::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")
}

/// The `.zarray` of an array.
fn zarray(shape: &[usize], chunks: &[usize], dtype: &str) -> Value {
    json!({
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": {"id": "zstd", "level": ZSTD_LEVEL},
        "fill_value": 0,
        "order": "C",
        "filters": null,
        "dimension_separator": ".",
    })
}

fn compress(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::bulk::compress(bytes, ZSTD_LEVEL)
}

/// Writes a one-dimensional (or `(n, columns)`) array of little-endian
/// values in chunks of [`ROW_CHUNK`] rows.
#[allow(clippy::too_many_arguments)]
fn write_rows(
    dir: &Path,
    name: &str,
    rows: usize,
    columns: Option<usize>,
    dtype: &str,
    bytes_per_row: usize,
    data: &[u8],
    meta: &mut Map<String, Value>,
) -> std::io::Result<()> {
    let array = dir.join(name);
    std::fs::create_dir_all(&array)?;
    let chunk_rows = rows.clamp(1, ROW_CHUNK);
    let (shape, chunks): (Vec<usize>, Vec<usize>) = match columns {
        Some(columns) => (vec![rows, columns], vec![chunk_rows, columns]),
        None => (vec![rows], vec![chunk_rows]),
    };
    let zarray = zarray(&shape, &chunks, dtype);
    write_json(&array.join(".zarray"), &zarray)?;
    meta.insert(format!("{name}/.zarray"), zarray);
    for (index, chunk) in data.chunks(chunk_rows * bytes_per_row).enumerate() {
        // Chunks are always full size; the last one is padded.
        let mut full = chunk.to_vec();
        full.resize(chunk_rows * bytes_per_row, 0);
        let key = match columns {
            Some(_) => format!("{index}.0"),
            None => index.to_string(),
        };
        std::fs::write(array.join(key), compress(&full)?)?;
    }
    Ok(())
}

/// Packs the dataset in `dir` into `dir/patches.zarr`. Returns `Ok(None)`
/// when cancelled.
pub fn pack_zarr(dir: &Path, cancel: &AtomicBool) -> std::io::Result<Option<PackReport>> {
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
    let classes = read_classes(dir);

    // Slides in order of first appearance.
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

    let final_store = dir.join(ZARR_STORE);
    let store = dir.join(format!("{ZARR_STORE}.{}.partial", std::process::id()));
    if store.exists() {
        std::fs::remove_dir_all(&store)?;
    }
    std::fs::create_dir_all(store.join("images"))?;
    let result = (|| -> std::io::Result<Option<()>> {
        let mut meta = Map::new();
        let group = json!({"zarr_format": 2});
        write_json(&store.join(".zgroup"), &group)?;
        meta.insert(".zgroup".into(), group);

        let n = records.len();
        let (w, h) = (width as usize, height as usize);
        let images = zarray(&[n, h, w, 3], &[1, h, w, 3], "|u1");
        write_json(&store.join("images").join(".zarray"), &images)?;
        meta.insert("images/.zarray".into(), images);
        if with_masks {
            std::fs::create_dir_all(store.join("masks"))?;
            let masks = zarray(&[n, h, w], &[1, h, w], "|u1");
            write_json(&store.join("masks").join(".zarray"), &masks)?;
            meta.insert("masks/.zarray".into(), masks);
        }

        // Patches and masks: decoded, compressed and written in parallel.
        records
            .par_iter()
            .enumerate()
            .try_for_each(|(index, record)| -> std::io::Result<()> {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let image = image::open(dir.join(&record.tile_path))
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
                std::fs::write(
                    store.join("images").join(format!("{index}.0.0.0")),
                    compress(image.as_raw())?,
                )?;
                if with_masks && let Some(mask) = &record.mask_path {
                    let mask = image::open(dir.join(mask))
                        .map_err(|e| std::io::Error::other(format!("{mask}: {e}")))?
                        .to_luma8();
                    std::fs::write(
                        store.join("masks").join(format!("{index}.0.0")),
                        compress(mask.as_raw())?,
                    )?;
                }
                Ok(())
            })?;
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }

        let coords: Vec<u8> = records
            .iter()
            .flat_map(|r| {
                (r.x as i64)
                    .to_le_bytes()
                    .into_iter()
                    .chain((r.y as i64).to_le_bytes())
            })
            .collect();
        write_rows(&store, "coords", n, Some(2), "<i8", 16, &coords, &mut meta)?;
        let slide: Vec<u8> = records
            .iter()
            .flat_map(|r| slide_index[&r.slide_id].to_le_bytes())
            .collect();
        write_rows(&store, "slide", n, None, "<i4", 4, &slide, &mut meta)?;
        let label: Vec<u8> = records
            .iter()
            .map(|r| {
                r.label
                    .as_ref()
                    .and_then(|label| classes.get(label).copied())
                    .unwrap_or(0)
            })
            .collect();
        write_rows(&store, "label", n, None, "|u1", 1, &label, &mut meta)?;

        let class_names: BTreeMap<String, String> = classes
            .iter()
            .map(|(name, index)| (index.to_string(), name.clone()))
            .collect();
        let attrs = json!({
            "eov_version": env!("CARGO_PKG_VERSION"),
            "description": "Patches of an eov dataset: images (N, H, W, 3) uint8; coords (N, 2) level-0 x, y; slide (N,) index into slides; label (N,) index into classes (0: none); masks (N, H, W) when present.",
            "tile_size": width,
            "patches": n,
            "slides": slides,
            "classes": class_names,
        });
        write_json(&store.join(".zattrs"), &attrs)?;
        meta.insert(".zattrs".into(), attrs);
        write_json(
            &store.join(".zmetadata"),
            &json!({"zarr_consolidated_format": 1, "metadata": meta}),
        )?;
        Ok(Some(()))
    })();
    match result {
        Ok(Some(())) => {
            if final_store.exists() {
                std::fs::remove_dir_all(&final_store)?;
            }
            std::fs::rename(&store, &final_store)?;
            Ok(Some(PackReport {
                store: final_store,
                patches: records.len(),
                tile_size: width,
                slides: slides.len(),
                with_masks,
            }))
        }
        Ok(None) => {
            let _ = std::fs::remove_dir_all(&store);
            Ok(None)
        }
        Err(error) => {
            let _ = std::fs::remove_dir_all(&store);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        DatasetControl, DatasetPatchesConfig, DatasetPatchesOptions, run_dataset,
    };

    fn read_json(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn chunk(path: &Path) -> Vec<u8> {
        zstd::bulk::decompress(&std::fs::read(path).unwrap(), 1 << 24).unwrap()
    }

    #[test]
    fn a_dataset_packs_into_readable_zarr_arrays() {
        let temp = tempfile::tempdir().unwrap();
        let slide = temp.path().join("s.tif");
        crate::test_support::write_tiled_tiff(&slide, 256, 128, 64, |x, y| {
            [(x % 251) as u8, (y % 251) as u8, 77]
        })
        .unwrap();
        let out = temp.path().join("out");
        let config = DatasetPatchesConfig {
            inputs: vec![slide],
            output_dir: out.clone(),
            tile_size: 64,
            stride: 64,
            metadata_format: None,
            threads: 2,
            white_threshold: None,
        };
        let report = run_dataset(
            &config,
            &DatasetPatchesOptions {
                hash_files: false,
                ..DatasetPatchesOptions::default()
            },
            &DatasetControl::new(),
        )
        .unwrap();
        assert_eq!(report.total_tiles, 8);

        let packed = pack_zarr(&out, &AtomicBool::new(false)).unwrap().unwrap();
        assert_eq!(
            (packed.patches, packed.tile_size, packed.slides),
            (8, 64, 1)
        );
        let store = out.join(ZARR_STORE);
        let images = read_json(&store.join("images/.zarray"));
        assert_eq!(images["shape"], json!([8, 64, 64, 3]));
        assert_eq!(images["chunks"], json!([1, 64, 64, 3]));
        assert_eq!(images["compressor"]["id"], "zstd");
        // Patch i's pixels are those of its record's tile image.
        let records = read_records(&out).unwrap();
        for (index, record) in records.iter().enumerate() {
            let pixels = chunk(&store.join(format!("images/{index}.0.0.0")));
            let expected = image::open(out.join(&record.tile_path)).unwrap().to_rgb8();
            assert_eq!(pixels, expected.into_raw(), "patch {index}");
        }
        let coords = chunk(&store.join("coords/0.0"));
        let x = |index: usize| {
            i64::from_le_bytes(coords[index * 16..index * 16 + 8].try_into().unwrap())
        };
        let y = |index: usize| {
            i64::from_le_bytes(coords[index * 16 + 8..index * 16 + 16].try_into().unwrap())
        };
        for (index, record) in records.iter().enumerate() {
            assert_eq!((x(index), y(index)), (record.x as i64, record.y as i64));
        }
        let attrs = read_json(&store.join(".zattrs"));
        assert_eq!(attrs["slides"].as_array().unwrap().len(), 1);
        let consolidated = read_json(&store.join(".zmetadata"));
        assert!(consolidated["metadata"]["coords/.zarray"].is_object());
        assert!(
            !store.join("masks").exists(),
            "no masks without annotations"
        );

        // Packing again replaces the store.
        assert!(pack_zarr(&out, &AtomicBool::new(false)).unwrap().is_some());
        let leftovers: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".partial"))
            .collect();
        assert!(leftovers.is_empty());
        // A cancelled pack leaves the last complete store.
        assert!(pack_zarr(&out, &AtomicBool::new(true)).unwrap().is_none());
        assert!(store.join(".zmetadata").is_file());
    }

    #[test]
    fn folders_without_records_are_refused() {
        let temp = tempfile::tempdir().unwrap();
        assert!(pack_zarr(temp.path(), &AtomicBool::new(false)).is_err());
    }
}
