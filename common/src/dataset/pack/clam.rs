//! `clam/<slide id>.h5`: each slide's patch coordinates in the files of
//! CLAM's patching step, for CLAM's feature extraction and the tools that
//! read its coordinates.
//!
//! A file holds `coords`, (N, 2) int64: the level-0 x and y of the slide's
//! patches in the order of the records, with CLAM's attributes on it:
//! `patch_level` 0, `patch_size` the patches' width in level-0 pixels,
//! `downsample` (1, 1), the downsample of level 0, `level_dim` and
//! `downsampled_level_dim` the slide's level-0 width and height, and `name`
//! the slide's id, after which the file is named (the slide's file stem,
//! made unique when two slides share one). Given at level 0, the size
//! covers the patch's area whatever resolution eov extracted at: a
//! 256-pixel patch at 0.5 µm/px of a 0.25 µm/px slide has a `patch_size`
//! of 512, which CLAM reads at level 0 and resizes for its model.

use super::hdf5::layout;
use super::{CLAM_FOLDER, Patches, ROW_CHUNK, replace_when_complete};
use crate::dataset::{PackReport, TileRecord};
use crate::hdf5::{AttrValue, Hdf5, LibrarySearch};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) fn pack(
    patches: &Patches,
    search: &LibrarySearch,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PackReport>> {
    let hdf5 = Hdf5::open(search).map_err(std::io::Error::other)?;
    write(patches, &hdf5, cancel)
}

fn write(
    patches: &Patches,
    hdf5: &Hdf5,
    cancel: &AtomicBool,
) -> std::io::Result<Option<PackReport>> {
    let folder = replace_when_complete(&patches.dir, CLAM_FOLDER, |folder| {
        write_files(patches, hdf5, folder, cancel)
    })?;
    Ok(folder.map(|folder| PackReport {
        // Coordinates only.
        with_masks: false,
        ..patches.report(folder, Some(hdf5.describe()))
    }))
}

/// A patch's width in level-0 pixels (for records older than
/// `level0_width`, its size times the downsample).
fn level0_size(record: &TileRecord) -> i64 {
    let width = if record.level0_width > 0.0 {
        record.level0_width
    } else {
        f64::from(record.tile_size) * record.downsample.max(1.0)
    };
    width.round() as i64
}

/// Writes one file per slide into the folder `folder`; `Ok(None)` when
/// cancelled.
fn write_files(
    patches: &Patches,
    hdf5: &Hdf5,
    folder: &Path,
    cancel: &AtomicBool,
) -> std::io::Result<Option<()>> {
    std::fs::create_dir(folder)?;
    let slide_of = patches.slide_indices();
    for slide in 0..patches.slides.len() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let records: Vec<&TileRecord> = patches
            .records
            .iter()
            .zip(&slide_of)
            .filter(|&(_, &index)| index as usize == slide)
            .map(|(record, _)| record)
            .collect();
        // Slides are listed from the records, so each has one.
        let first = records[0];
        let name = match first.slide_id.as_str() {
            // Records of datasets older than slide ids.
            "" => first.slide_stem.as_str(),
            id => id,
        };
        let coords: Vec<i64> = records
            .iter()
            .flat_map(|record| [record.x as i64, record.y as i64])
            .collect();
        let n = records.len() as u64;
        let file = hdf5.create(&folder.join(format!("{name}.h5")))?;
        let dataset = file.create_dataset::<i64>(
            "coords",
            &[n, 2],
            &layout(&[n.clamp(1, ROW_CHUNK as u64), 2], true),
        )?;
        dataset.write_all(&coords)?;
        let size = [first.slide_width as i64, first.slide_height as i64];
        dataset.set_attribute("patch_level", &AttrValue::Int(0))?;
        dataset.set_attribute("patch_size", &AttrValue::Int(level0_size(first)))?;
        dataset.set_attribute("downsample", &AttrValue::Floats(&[1.0, 1.0]))?;
        dataset.set_attribute("level_dim", &AttrValue::Ints(&size))?;
        dataset.set_attribute("downsampled_level_dim", &AttrValue::Ints(&size))?;
        dataset.set_attribute("name", &AttrValue::Text(name))?;
        dataset.close()?;
        file.close()?;
    }
    Ok(Some(()))
}

#[cfg(test)]
mod tests {
    use super::super::tests::labelled_dataset;
    use super::super::{PackFormat, PackOptions, pack};
    use super::*;
    use crate::hdf5::testing::session;

    #[test]
    fn each_slide_gets_a_file_of_level_0_coordinates() {
        let test = "each_slide_gets_a_file_of_level_0_coordinates";
        if session(test).is_none() {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let options = PackOptions {
            format: PackFormat::Clam,
            hdf5_library: None,
        };
        let report = pack(&out, &options, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert_eq!(report.store, out.join(CLAM_FOLDER));
        assert_eq!((report.patches, report.slides), (10, 2));
        assert!(!report.with_masks && report.library.is_some());

        let patches = Patches::read(&out).unwrap();
        let hdf5 = session(test).unwrap();
        let mut files: Vec<String> = std::fs::read_dir(&report.store)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        assert_eq!(files, ["a.h5", "b.h5"]);
        for (id, width, patch_count) in [("a", 96, 6), ("b", 64, 4)] {
            let file = hdf5
                .open_file(&report.store.join(format!("{id}.h5")))
                .unwrap();
            let coords = file.read::<i64>("coords").unwrap();
            let expected: Vec<i64> = patches
                .records
                .iter()
                .filter(|record| record.slide_id == id)
                .flat_map(|record| [record.x as i64, record.y as i64])
                .collect();
            assert_eq!(coords, expected, "slide {id}");
            assert_eq!(coords.len(), 2 * patch_count);
            let stored = file.stored("coords").unwrap();
            assert_eq!(stored.shape, [patch_count as u64, 2]);
            assert_eq!((stored.size, stored.signed), (8, true));
            let number = |name| file.numbers::<i64>(Some("coords"), name).unwrap();
            assert_eq!(number("patch_level"), [0]);
            assert_eq!(number("patch_size"), [32]);
            assert_eq!(number("level_dim"), [width, 64]);
            assert_eq!(number("downsampled_level_dim"), [width, 64]);
            assert_eq!(
                file.numbers::<f64>(Some("coords"), "downsample").unwrap(),
                [1.0, 1.0]
            );
            assert_eq!(file.text(Some("coords"), "name").unwrap(), id);
        }
        assert_eq!(hdf5.open_objects(), 0, "every id is closed");

        // Packing again replaces the folder; a cancelled pack keeps it.
        std::fs::write(report.store.join("stale.h5"), "").unwrap();
        assert!(
            write(&patches, &hdf5, &AtomicBool::new(false))
                .unwrap()
                .is_some()
        );
        assert!(!report.store.join("stale.h5").exists());
        assert!(
            write(&patches, &hdf5, &AtomicBool::new(true))
                .unwrap()
                .is_none()
        );
        assert!(report.store.join("a.h5").is_file());
        let partial = std::fs::read_dir(&out)
            .unwrap()
            .filter(|entry| {
                let name = entry.as_ref().unwrap().file_name();
                name.to_string_lossy().ends_with(".partial")
            })
            .count();
        assert_eq!(partial, 0);
    }

    #[test]
    fn the_patch_size_is_the_level_0_width() {
        let temp = tempfile::tempdir().unwrap();
        let out = labelled_dataset(temp.path());
        let mut record = Patches::read(&out).unwrap().records.remove(0);
        assert_eq!(level0_size(&record), 32);
        // 32 px at twice the slide's pixel size.
        record.level0_width = 64.0;
        record.downsample = 2.0;
        assert_eq!(level0_size(&record), 64);
        // Older records: the tile size times the downsample.
        record.level0_width = 0.0;
        assert_eq!(level0_size(&record), 64);
        record.downsample = 0.0;
        assert_eq!(level0_size(&record), 32);
    }
}
