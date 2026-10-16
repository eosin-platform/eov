//! `patches.zarr`: a zarr v2 store with zstd chunks and consolidated
//! metadata.

use super::{Patches, ROW_CHUNK, ZARR_STORE, replace_when_complete};
use crate::dataset::PackReport;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// zstd level of the chunks.
const ZSTD_LEVEL: i32 = 3;

pub(super) fn pack(patches: &Patches, cancel: &AtomicBool) -> std::io::Result<Option<PackReport>> {
    let store = replace_when_complete(&patches.dir, ZARR_STORE, |store| {
        write_store(patches, store, cancel)
    })?;
    Ok(store.map(|store| patches.report(store, None)))
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

/// Fills the folder `store`; `Ok(None)` when cancelled.
fn write_store(
    patches: &Patches,
    store: &Path,
    cancel: &AtomicBool,
) -> std::io::Result<Option<()>> {
    std::fs::create_dir_all(store.join("images"))?;
    let mut meta = Map::new();
    let group = json!({"zarr_format": 2});
    write_json(&store.join(".zgroup"), &group)?;
    meta.insert(".zgroup".into(), group);

    let n = patches.len();
    let (w, h) = (patches.width as usize, patches.height as usize);
    let images = zarray(&[n, h, w, 3], &[1, h, w, 3], "|u1");
    write_json(&store.join("images").join(".zarray"), &images)?;
    meta.insert("images/.zarray".into(), images);
    if patches.with_masks {
        std::fs::create_dir_all(store.join("masks"))?;
        let masks = zarray(&[n, h, w], &[1, h, w], "|u1");
        write_json(&store.join("masks").join(".zarray"), &masks)?;
        meta.insert("masks/.zarray".into(), masks);
    }

    // Patches and masks: decoded, compressed and written in parallel.
    (0..n)
        .into_par_iter()
        .try_for_each(|index| -> std::io::Result<()> {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let patch = patches.decode(index)?;
            std::fs::write(
                store.join("images").join(format!("{index}.0.0.0")),
                compress(&patch.image)?,
            )?;
            if let Some(mask) = &patch.mask {
                std::fs::write(
                    store.join("masks").join(format!("{index}.0.0")),
                    compress(mask)?,
                )?;
            }
            Ok(())
        })?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let coords: Vec<u8> = patches
        .coords()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    write_rows(store, "coords", n, Some(2), "<i8", 16, &coords, &mut meta)?;
    let slide: Vec<u8> = patches
        .slide_indices()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    write_rows(store, "slide", n, None, "<i4", 4, &slide, &mut meta)?;
    write_rows(
        store,
        "label",
        n,
        None,
        "|u1",
        1,
        &patches.labels(),
        &mut meta,
    )?;

    let attrs = Value::Object(patches.attributes(""));
    write_json(&store.join(".zattrs"), &attrs)?;
    meta.insert(".zattrs".into(), attrs);
    write_json(
        &store.join(".zmetadata"),
        &json!({"zarr_consolidated_format": 1, "metadata": meta}),
    )?;
    Ok(Some(()))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{labelled_dataset, small_dataset};
    use super::super::{pack_zarr, read_records};
    use super::*;

    fn read_json(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn chunk(path: &Path) -> Vec<u8> {
        zstd::bulk::decompress(&std::fs::read(path).unwrap(), 1 << 24).unwrap()
    }

    #[test]
    fn a_dataset_packs_into_readable_zarr_arrays() {
        let temp = tempfile::tempdir().unwrap();
        let out = small_dataset(temp.path());

        let packed = pack_zarr(&out, &AtomicBool::new(false)).unwrap().unwrap();
        assert_eq!(
            (packed.patches, packed.tile_size, packed.slides),
            (8, 64, 1)
        );
        assert_eq!(packed.library, None);
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
    fn masks_slides_and_labels_are_packed() {
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let packed = pack_zarr(&out, &AtomicBool::new(false)).unwrap().unwrap();
        assert!(packed.with_masks);
        assert_eq!((packed.patches, packed.slides), (10, 2));
        let store = out.join(ZARR_STORE);
        let records = read_records(&out).unwrap();
        for (index, record) in records.iter().enumerate() {
            let mask = chunk(&store.join(format!("masks/{index}.0.0")));
            let expected = image::open(out.join(record.mask_path.as_ref().unwrap()))
                .unwrap()
                .to_luma8();
            assert_eq!(mask, expected.into_raw(), "mask {index}");
        }
        let patches = Patches::read(&out).unwrap();
        let slide = chunk(&store.join("slide/0"));
        let expected: Vec<u8> = patches
            .slide_indices()
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        assert_eq!(slide, expected);
        assert_eq!(chunk(&store.join("label/0")), patches.labels());
    }
}
