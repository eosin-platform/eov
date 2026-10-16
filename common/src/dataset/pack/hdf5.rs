//! `patches.h5`: the zarr store's arrays and attributes in one HDF5 file,
//! written by the HDF5 library ([`crate::hdf5`]).
//!
//! Patches are decoded on every core, in batches, and written one chunk
//! each in record order. With HDF5 1.10.3 and later eov also deflates them
//! on every core and stores the chunks with `H5Dwrite_chunk`; older
//! versions deflate them in `H5Dwrite`, one at a time. Either way the
//! datasets declare the deflate filter and HDF5 builds the file.

use super::{HDF5_FILE, Patch, Patches, ROW_CHUNK, replace_when_complete};
use crate::dataset::PackReport;
use crate::hdf5::{AttrValue, Hdf5, Layout, LibrarySearch};
use rayon::prelude::*;
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Deflate level of every chunk (h5py's default for gzip).
const DEFLATE_LEVEL: u32 = 4;
/// Most pixels decoded at once, in bytes.
const BATCH_BYTES: usize = 256 << 20;

/// How the patches' chunks reach the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkWrites {
    /// Deflated by eov on every core and stored as they are
    /// (`H5Dwrite_chunk`, HDF5 1.10.3 and later).
    Direct,
    /// Written with `H5Dwrite`, which deflates them in the library.
    Library,
}

pub(super) fn pack(
    patches: &Patches,
    search: &LibrarySearch,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PackReport>> {
    let hdf5 = Hdf5::open(search).map_err(std::io::Error::other)?;
    let writes = if hdf5.can_write_chunks() {
        ChunkWrites::Direct
    } else {
        ChunkWrites::Library
    };
    write(patches, &hdf5, writes, cancel)
}

fn write(
    patches: &Patches,
    hdf5: &Hdf5,
    writes: ChunkWrites,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PackReport>> {
    let file = replace_when_complete(&patches.dir, HDF5_FILE, |path| {
        write_file(patches, hdf5, writes, path, cancel)
    })?;
    Ok(file.map(|file| patches.report(file, Some(hdf5.describe()))))
}

/// Chunks of `chunk`, deflated (after the shuffle filter with `shuffle`).
pub(super) fn layout(chunk: &[u64], shuffle: bool) -> Layout<'_> {
    Layout {
        chunk,
        shuffle,
        deflate: DEFLATE_LEVEL,
    }
}

/// What HDF5's deflate filter stores: a zlib stream.
fn deflate(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(
        Vec::with_capacity(bytes.len() / 2 + 64),
        flate2::Compression::new(DEFLATE_LEVEL),
    );
    encoder.write_all(bytes)?;
    encoder.finish()
}

/// Patches decoded at once: enough to keep every core busy, within
/// [`BATCH_BYTES`].
fn batch_size(patches: &Patches) -> usize {
    let bytes = patches.width as usize * patches.height as usize * 4;
    (rayon::current_num_threads() * 4)
        .clamp(4, 1024)
        .min(BATCH_BYTES / bytes.max(1))
        .max(1)
}

/// Writes the file at `path`; `Ok(None)` when cancelled.
fn write_file(
    patches: &Patches,
    hdf5: &Hdf5,
    writes: ChunkWrites,
    path: &Path,
    cancel: &AtomicBool,
) -> std::io::Result<Option<()>> {
    let file = hdf5.create(path)?;
    let n = patches.len() as u64;
    let (h, w) = (u64::from(patches.height), u64::from(patches.width));
    let images =
        file.create_dataset::<u8>("images", &[n, h, w, 3], &layout(&[1, h, w, 3], false))?;
    let masks = match patches.with_masks {
        true => Some(file.create_dataset::<u8>("masks", &[n, h, w], &layout(&[1, h, w], false))?),
        false => None,
    };

    let batch = batch_size(patches);
    for first in (0..patches.len()).step_by(batch) {
        let indices = first..(first + batch).min(patches.len());
        let encoded = indices
            .clone()
            .into_par_iter()
            .map(|index| -> std::io::Result<Option<Patch>> {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                let patch = patches.decode(index)?;
                Ok(Some(match writes {
                    ChunkWrites::Direct => Patch {
                        image: deflate(&patch.image)?,
                        mask: patch.mask.as_deref().map(deflate).transpose()?,
                    },
                    ChunkWrites::Library => patch,
                }))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        for (index, patch) in indices.zip(encoded) {
            let Some(patch) = patch else {
                return Ok(None);
            };
            let row = index as u64;
            match writes {
                ChunkWrites::Direct => images.write_chunk(&[row, 0, 0, 0], &patch.image)?,
                ChunkWrites::Library => images.write_rows(row, &patch.image)?,
            }
            if let (Some(masks), Some(mask)) = (&masks, &patch.mask) {
                match writes {
                    ChunkWrites::Direct => masks.write_chunk(&[row, 0, 0], mask)?,
                    ChunkWrites::Library => masks.write_rows(row, mask)?,
                }
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }
    images.close()?;
    if let Some(masks) = masks {
        masks.close()?;
    }

    let rows = (n as usize).clamp(1, ROW_CHUNK) as u64;
    let coords = file.create_dataset::<i64>("coords", &[n, 2], &layout(&[rows, 2], true))?;
    coords.write_all(&patches.coords())?;
    coords.close()?;
    let slide = file.create_dataset::<i32>("slide", &[n], &layout(&[rows], true))?;
    slide.write_all(&patches.slide_indices())?;
    slide.close()?;
    let label = file.create_dataset::<u8>("label", &[n], &layout(&[rows], false))?;
    label.write_all(&patches.labels())?;
    label.close()?;

    for (name, value) in patches.attributes(" The slides and classes attributes are JSON text.") {
        let json;
        let value = match &value {
            Value::String(text) => AttrValue::Text(text),
            Value::Number(number) => match number.as_i64() {
                Some(int) => AttrValue::Int(int),
                None => AttrValue::Float(number.as_f64().unwrap_or(f64::NAN)),
            },
            // Lists and maps (and anything else) as JSON text.
            other => {
                json = other.to_string();
                AttrValue::Text(&json)
            }
        };
        file.set_attribute(&name, &value)?;
    }
    file.close()?;
    Ok(Some(()))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{labelled_dataset, small_dataset};
    use super::super::{PackFormat, PackOptions, ZARR_STORE, pack, pack_zarr};
    use super::*;
    use crate::hdf5::testing::session;
    use std::path::PathBuf;

    const DEFLATE: (i32, u32) = (1, DEFLATE_LEVEL);
    const SHUFFLE: i32 = 2;

    fn hdf5_options() -> PackOptions {
        PackOptions {
            format: PackFormat::Hdf5,
            hdf5_library: None,
        }
    }

    fn partial_files(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|path| path.to_string_lossy().ends_with(".partial"))
            .collect()
    }

    /// The patches' pixels and masks as their image files hold them.
    fn pixels_and_masks(patches: &Patches) -> (Vec<u8>, Vec<u8>) {
        let (mut pixels, mut masks) = (Vec::new(), Vec::new());
        for record in &patches.records {
            let image = image::open(patches.dir.join(&record.tile_path)).unwrap();
            pixels.extend(image.to_rgb8().into_raw());
            if let Some(mask) = &record.mask_path {
                masks.extend(
                    image::open(patches.dir.join(mask))
                        .unwrap()
                        .to_luma8()
                        .into_raw(),
                );
            }
        }
        (pixels, masks)
    }

    #[test]
    fn a_dataset_packs_into_an_hdf5_file_with_the_zarr_stores_contents() {
        let test = "a_dataset_packs_into_an_hdf5_file_with_the_zarr_stores_contents";
        if session(test).is_none() {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let report = pack(&out, &hdf5_options(), &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert_eq!(report.store, out.join(HDF5_FILE));
        assert_eq!(
            (
                report.patches,
                report.tile_size,
                report.slides,
                report.with_masks
            ),
            (10, 32, 2, true)
        );
        assert!(report.library.as_ref().unwrap().starts_with("HDF5 1."));
        assert!(partial_files(&out).is_empty());
        pack_zarr(&out, &AtomicBool::new(false)).unwrap().unwrap();
        let zattrs: Value = serde_json::from_str(
            &std::fs::read_to_string(out.join(ZARR_STORE).join(".zattrs")).unwrap(),
        )
        .unwrap();

        let patches = Patches::read(&out).unwrap();
        let (pixels, masks) = pixels_and_masks(&patches);
        let hdf5 = session(test).unwrap();
        let file = hdf5.open_file(&report.store).unwrap();

        let images = file.stored("images").unwrap();
        assert_eq!(images.shape, [10, 32, 32, 3]);
        assert_eq!(
            (
                images.class,
                images.size,
                images.signed,
                images.little_endian
            ),
            (0, 1, false, true),
            "uint8"
        );
        assert_eq!(images.chunk.as_deref(), Some(&[1, 32, 32, 3][..]));
        assert_eq!(images.filters, [(DEFLATE.0, vec![DEFLATE.1])]);
        assert_eq!(file.read::<u8>("images").unwrap(), pixels);

        let stored = file.stored("masks").unwrap();
        assert_eq!(stored.shape, [10, 32, 32]);
        assert_eq!(stored.chunk.as_deref(), Some(&[1, 32, 32][..]));
        assert_eq!(stored.filters, [(DEFLATE.0, vec![DEFLATE.1])]);
        assert_eq!(file.read::<u8>("masks").unwrap(), masks);
        assert!(masks.contains(&1) && masks.contains(&2), "both classes");

        let stored = file.stored("coords").unwrap();
        assert_eq!(stored.shape, [10, 2]);
        assert_eq!(
            (
                stored.class,
                stored.size,
                stored.signed,
                stored.little_endian
            ),
            (0, 8, true, true),
            "int64"
        );
        assert_eq!(stored.chunk.as_deref(), Some(&[10, 2][..]));
        let filters: Vec<i32> = stored.filters.iter().map(|filter| filter.0).collect();
        assert_eq!(filters, [SHUFFLE, DEFLATE.0]);
        assert_eq!(file.read::<i64>("coords").unwrap(), patches.coords());

        let stored = file.stored("slide").unwrap();
        assert_eq!(
            (stored.shape.as_slice(), stored.size, stored.signed),
            (&[10][..], 4, true)
        );
        assert_eq!(stored.chunk.as_deref(), Some(&[10][..]));
        let filters: Vec<i32> = stored.filters.iter().map(|filter| filter.0).collect();
        assert_eq!(filters, [SHUFFLE, DEFLATE.0]);
        assert_eq!(file.read::<i32>("slide").unwrap(), patches.slide_indices());

        let stored = file.stored("label").unwrap();
        assert_eq!(
            (stored.shape.as_slice(), stored.size, stored.signed),
            (&[10][..], 1, false)
        );
        assert_eq!(stored.filters, [(DEFLATE.0, vec![DEFLATE.1])]);
        assert_eq!(file.read::<u8>("label").unwrap(), patches.labels());

        // The zarr store's attributes; lists and maps as JSON text.
        let json = |name| serde_json::from_str::<Value>(&file.text(None, name).unwrap()).unwrap();
        assert_eq!(json("slides"), zattrs["slides"]);
        assert_eq!(json("classes"), zattrs["classes"]);
        assert_eq!(json("classes").as_object().unwrap().len(), 3);
        assert_eq!(file.numbers::<i64>(None, "tile_size").unwrap(), [32]);
        assert_eq!(file.numbers::<i64>(None, "patches").unwrap(), [10]);
        assert_eq!(
            file.text(None, "eov_version").unwrap(),
            env!("CARGO_PKG_VERSION")
        );
        let description = file.text(None, "description").unwrap();
        assert!(description.starts_with(zattrs["description"].as_str().unwrap()));
        assert!(description.ends_with("JSON text."));
        drop(file);
        assert_eq!(hdf5.open_objects(), 0, "every id is closed");
    }

    #[test]
    fn chunks_deflated_by_eov_or_by_hdf5_read_the_same() {
        let test = "chunks_deflated_by_eov_or_by_hdf5_read_the_same";
        let Some(hdf5) = session(test) else {
            return;
        };
        let temp = tempfile::tempdir().unwrap();
        let out = small_dataset(temp.path());
        let patches = Patches::read(&out).unwrap();
        let (pixels, _) = pixels_and_masks(&patches);
        let file = out.join(HDF5_FILE);
        let read = |hdf5: &Hdf5| hdf5.open_file(&file).unwrap().read::<u8>("images").unwrap();

        let report = write(
            &patches,
            &hdf5,
            ChunkWrites::Library,
            &AtomicBool::new(false),
        )
        .unwrap()
        .unwrap();
        assert_eq!(report.store, file);
        assert!(!report.with_masks);
        assert_eq!(read(&hdf5), pixels);
        assert!(!hdf5.open_file(&file).unwrap().exists("masks"));
        if hdf5.can_write_chunks() {
            write(
                &patches,
                &hdf5,
                ChunkWrites::Direct,
                &AtomicBool::new(false),
            )
            .unwrap()
            .unwrap();
            assert_eq!(read(&hdf5), pixels);
            // Without modification times, a dataset packs into the same bytes.
            let bytes = std::fs::read(&file).unwrap();
            write(
                &patches,
                &hdf5,
                ChunkWrites::Direct,
                &AtomicBool::new(false),
            )
            .unwrap()
            .unwrap();
            assert_eq!(std::fs::read(&file).unwrap(), bytes);
        } else {
            eprintln!("{test}: {} has no H5Dwrite_chunk", hdf5.describe());
        }

        // A cancelled pack leaves the last complete file, and no partial one.
        for writes in [ChunkWrites::Library, ChunkWrites::Direct] {
            let cancelled = write(&patches, &hdf5, writes, &AtomicBool::new(true)).unwrap();
            assert!(cancelled.is_none());
        }
        assert_eq!(read(&hdf5), pixels);
        assert!(partial_files(&out).is_empty());
        assert_eq!(hdf5.open_objects(), 0, "every id is closed");
    }

    #[test]
    fn without_a_library_nothing_is_written() {
        let temp = tempfile::tempdir().unwrap();
        let out = small_dataset(temp.path());
        let not_a_library = temp.path().join("libhdf5.so");
        std::fs::write(&not_a_library, "not a library").unwrap();
        let options = PackOptions {
            hdf5_library: Some(not_a_library.clone()),
            ..hdf5_options()
        };
        let error = pack(&out, &options, &AtomicBool::new(false)).unwrap_err();
        let text = error.to_string();
        assert!(
            text.contains(&not_a_library.display().to_string()),
            "{text}"
        );
        assert!(text.contains("from --hdf5-library"), "{text}");
        assert!(text.contains("--format zarr"), "{text}");
        assert!(!out.join(HDF5_FILE).exists());
        assert!(partial_files(&out).is_empty());
    }
}
