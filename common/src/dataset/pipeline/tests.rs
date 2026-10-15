//! Dataset pipeline integration tests on synthetic slides.

use std::path::{Path, PathBuf};

use super::*;
use crate::dataset::config::{
    AnnotationOptions, MetadataFormat, PatchImageFormat, PatchNormalization, TissueFilter,
};
use crate::stain::StainFitMethod;
use crate::test_slides::{PyramidSpec, he_tissue_pixel, write_pyramid_tiff};
use crate::test_support::write_tiled_tiff;

fn config(inputs: Vec<PathBuf>, output_dir: PathBuf) -> DatasetPatchesConfig {
    DatasetPatchesConfig {
        inputs,
        output_dir,
        tile_size: 128,
        stride: 128,
        metadata_format: Some(MetadataFormat::Json),
        threads: 3,
        white_threshold: None,
    }
}

fn options() -> DatasetPatchesOptions {
    DatasetPatchesOptions {
        hash_files: true,
        ..DatasetPatchesOptions::default()
    }
}

fn run(config: &DatasetPatchesConfig, options: &DatasetPatchesOptions) -> DatasetPatchesReport {
    run_dataset(config, options, &DatasetControl::new()).unwrap()
}

fn centre_colour(path: &Path) -> [u8; 3] {
    let image = image::open(path).expect("tile image").to_rgb8();
    image.get_pixel(image.width() / 2, image.height() / 2).0
}

fn files_below(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| d.map(|e| e.unwrap().path()).collect())
        .unwrap_or_default();
    files.sort();
    files
}

fn records(out: &Path) -> Vec<TileRecord> {
    std::fs::read_to_string(out.join("metadata.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn every_slide_is_read_from_its_own_file() {
    // DS-1 regression: worker handles must never outlive their slide.
    let temp = tempfile::tempdir().unwrap();
    let colours = [[200, 40, 40], [40, 40, 200], [40, 160, 40]];
    let mut inputs = Vec::new();
    for (index, colour) in colours.iter().enumerate() {
        let path = temp.path().join(format!("slide{index}.tif"));
        write_tiled_tiff(&path, 512, 384, 64, |_, _| *colour).unwrap();
        inputs.push(path);
    }
    let out = temp.path().join("out");
    let report = run(&config(inputs, out.clone()), &options());
    assert_eq!(report.processed_slides, 3);
    assert_eq!(report.total_tiles, 3 * 4 * 3);
    assert!(!report.has_errors());
    for (index, colour) in colours.iter().enumerate() {
        let tiles = files_below(&out.join("slides").join(format!("slide{index}")));
        assert_eq!(tiles.len(), 12);
        for tile in tiles {
            assert_eq!(&centre_colour(&tile), colour, "{}", tile.display());
        }
    }
    // Records are in deterministic (raster) order per slide.
    let recs = records(&out);
    assert_eq!(recs.len(), 36);
    assert_eq!((recs[0].x, recs[0].y, recs[1].x), (0, 0, 128));
    assert!(
        recs.iter()
            .all(|r| r.file_sha256.len() == 64 && !r.slide_id.is_empty())
    );
    // The JSON array is produced at the end.
    let json: Vec<TileRecord> =
        serde_json::from_str(&std::fs::read_to_string(out.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(json, recs);
}

#[test]
fn slides_with_the_same_stem_do_not_overwrite_each_other() {
    let temp = tempfile::tempdir().unwrap();
    let a = temp.path().join("case1").join("HE.tif");
    let b = temp.path().join("case2").join("HE.tif");
    std::fs::create_dir_all(a.parent().unwrap()).unwrap();
    std::fs::create_dir_all(b.parent().unwrap()).unwrap();
    write_tiled_tiff(&a, 256, 256, 64, |_, _| [220, 30, 30]).unwrap();
    write_tiled_tiff(&b, 256, 256, 64, |_, _| [30, 30, 220]).unwrap();
    let out = temp.path().join("out");
    let report = run(&config(vec![a, b], out.clone()), &options());
    assert_eq!(report.total_tiles, 8);
    let dirs = files_below(&out.join("slides"));
    assert_eq!(
        dirs.len(),
        2,
        "each slide needs its own directory: {dirs:?}"
    );
    let mut seen: Vec<[u8; 3]> = dirs
        .iter()
        .map(|d| centre_colour(&files_below(d)[0]))
        .collect();
    seen.sort();
    assert_eq!(seen, vec![[30, 30, 220], [220, 30, 30]]);
    let ids: HashSet<String> = records(&out).into_iter().map(|r| r.slide_id).collect();
    assert_eq!(ids.len(), 2);
}

#[test]
fn a_broken_slide_does_not_stop_the_run() {
    // DS-3: per-slide isolation.
    let temp = tempfile::tempdir().unwrap();
    let good = temp.path().join("good.tif");
    write_tiled_tiff(&good, 256, 256, 64, |_, _| [200, 100, 150]).unwrap();
    let bad = temp.path().join("bad.svs");
    std::fs::write(&bad, b"this is not a slide").unwrap();
    let out = temp.path().join("out");
    let report = run(&config(vec![bad.clone(), good], out.clone()), &options());
    assert_eq!(report.processed_slides, 1);
    assert_eq!(report.skipped_slides, 1);
    assert!(report.has_errors());
    assert_eq!(report.status(), RunStatus::Partial);
    assert!(matches!(
        report.slides[0].skipped,
        Some(SlideSkipReason::OpenError(_))
    ));
    assert_eq!(report.total_tiles, 4);
    let slides = std::fs::read_to_string(out.join("slides.jsonl")).unwrap();
    assert!(slides.contains("\"status\":\"failed\""), "{slides}");
    let manifest = DatasetManifest::load(&out).unwrap().unwrap();
    assert_eq!(manifest.status, RunStatus::Partial);
    assert_eq!(manifest.totals.tiles_written, 4);
}

#[test]
fn reruns_resume_and_parameter_changes_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 512, 256, 64, |x, _| [(x % 251) as u8, 100, 150]).unwrap();
    let out = temp.path().join("out");
    let cfg = DatasetPatchesConfig {
        metadata_format: Some(MetadataFormat::JsonLines),
        ..config(vec![slide], out.clone())
    };
    let first = run(&cfg, &options());
    assert_eq!(first.total_tiles, 8);

    // Simulate an interrupted run: keep 3 records, delete the other tiles.
    let recs = records(&out);
    let kept: Vec<String> = recs
        .iter()
        .take(3)
        .map(|r| serde_json::to_string(r).unwrap())
        .collect();
    std::fs::write(out.join("metadata.jsonl"), kept.join("\n") + "\n").unwrap();
    for r in &recs[3..] {
        std::fs::remove_file(out.join(&r.tile_path)).unwrap();
    }
    let resumed = run(&cfg, &options());
    assert_eq!(resumed.total_tiles_existing, 3);
    assert_eq!(resumed.total_tiles, 5);
    let after = records(&out);
    assert_eq!(after.len(), 8, "every tile recorded exactly once");
    let unique: HashSet<_> = after.iter().map(|r| r.tile_path.clone()).collect();
    assert_eq!(unique.len(), 8);

    // Same parameters again: nothing to do.
    let again = run(&cfg, &options());
    assert_eq!((again.total_tiles, again.total_tiles_existing), (0, 8));

    // Different parameters are refused...
    let other = DatasetPatchesConfig {
        tile_size: 64,
        stride: 64,
        ..cfg.clone()
    };
    let err = run_dataset(&other, &options(), &DatasetControl::new()).unwrap_err();
    assert!(matches!(err, DatasetError::OutputConflict(_)), "{err}");
    // ...unless the output is overwritten.
    let overwrite = DatasetPatchesOptions {
        existing: ExistingOutput::Overwrite,
        ..options()
    };
    assert_eq!(run(&other, &overwrite).total_tiles, 32);
}

#[test]
fn completed_slides_are_not_processed_again_on_resume() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    // Left half glass, right half tissue: 4 patches written, 4 rejected.
    write_tiled_tiff(&slide, 512, 256, 64, |x, _| {
        if x < 256 {
            [255, 255, 255]
        } else {
            [120, 60, 160]
        }
    })
    .unwrap();
    let out = temp.path().join("out");
    let cfg = DatasetPatchesConfig {
        white_threshold: Some(0.8),
        ..config(vec![slide], out.clone())
    };
    let first = run(&cfg, &options());
    assert_eq!((first.total_tiles, first.total_tiles_skipped_white), (4, 4));

    // Nothing to do: the slide is recognised as complete without visiting
    // its grid again (the glass patches are not re-read and re-rejected).
    let again = run(&cfg, &options());
    let slide_report = &again.slides[0];
    assert_eq!(slide_report.status, SlideStatus::Complete);
    assert_eq!(
        (
            slide_report.tiles_written,
            slide_report.tiles_existing,
            slide_report.tiles_skipped_white
        ),
        (0, 4, 0)
    );
    assert_eq!(again.status(), RunStatus::Complete);

    // A recorded tile that disappeared is written again by a full pass.
    let lost = records(&out)[1].tile_path.clone();
    std::fs::remove_file(out.join(&lost)).unwrap();
    let repaired = run(&cfg, &options());
    assert_eq!(
        (
            repaired.total_tiles,
            repaired.total_tiles_existing,
            repaired.total_tiles_skipped_white
        ),
        (1, 3, 4)
    );
    assert!(out.join(&lost).is_file());

    // The slide files list the slide once, with its latest record.
    let slides: Vec<SlideRecord> = std::fs::read_to_string(out.join("slides.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(slides.len(), 1);
    assert_eq!((slides[0].written, slides[0].existing), (1, 3));
}

#[test]
fn errors_after_the_manifest_is_written_mark_the_run_failed() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 256, 256, 64, |_, _| [100, 50, 200]).unwrap();
    let out = temp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    // Tiles cannot be written: `slides` is a file.
    std::fs::write(out.join("slides"), b"in the way").unwrap();
    let err = run_dataset(
        &config(vec![slide], out.clone()),
        &options(),
        &DatasetControl::new(),
    )
    .unwrap_err();
    assert!(matches!(err, DatasetError::Write { .. }), "{err}");
    assert_eq!(
        DatasetManifest::load(&out).unwrap().unwrap().status,
        RunStatus::Failed
    );
    // The metadata was still finished.
    assert!(out.join("metadata.json").is_file());
}

#[test]
fn the_progress_mirror_stops_when_the_run_panics() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let control = DatasetControl::new();
        let cancel = AtomicBool::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_progress_mirror(&control, &cancel, &|| {}, || -> u32 {
                panic!("simulated failure inside a run")
            })
        }));
        let _ = tx.send(result.is_err());
    });
    // Without the guard the helper thread would spin forever and the scope
    // would never return.
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)), Ok(true));
}

#[test]
fn sampled_chunks_follow_the_remaining_budget() {
    // Before any result: the remaining budget plus a margin.
    assert_eq!(sampled_chunk_size(10, 0, 0, 4), 13);
    // A quarter of the candidates are accepted: 5 more need ~20 (+25 %).
    assert_eq!(sampled_chunk_size(10, 5, 20, 4), 25);
    // At least one candidate per thread.
    assert_eq!(sampled_chunk_size(1, 0, 0, 16), 16);
    assert_eq!(sampled_chunk_size(10, 10, 30, 4), 4);
    // Hardly anything accepted: large, but finite (callers cap it).
    assert_eq!(sampled_chunk_size(1000, 0, 100, 8), 125_000);
}

#[test]
fn cancellation_keeps_consistent_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 256, 256, 64, |_, _| [100, 50, 200]).unwrap();
    let out = temp.path().join("out");
    let control = DatasetControl::new();
    control.cancel();
    let report = run_dataset(&config(vec![slide], out.clone()), &options(), &control).unwrap();
    assert!(report.cancelled);
    assert_eq!(report.status(), RunStatus::Cancelled);
    assert_eq!(report.slides[0].status, SlideStatus::Cancelled);
    assert!(out.join("metadata.json").is_file());
    assert_eq!(
        DatasetManifest::load(&out).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    let progress = control.progress();
    assert_eq!(progress.tiles_done, progress.tiles_total);
}

#[test]
fn progress_totals_are_planned_up_front() {
    let temp = tempfile::tempdir().unwrap();
    let mut inputs = Vec::new();
    for i in 0..2 {
        let path = temp.path().join(format!("s{i}.tif"));
        write_tiled_tiff(&path, 256 + 128 * i, 256, 64, |_, _| [230, 230, 230]).unwrap();
        inputs.push(path);
    }
    let out = temp.path().join("out");
    let cfg = DatasetPatchesConfig {
        white_threshold: Some(0.5),
        ..config(inputs, out)
    };
    let control = DatasetControl::new();
    let report = run_dataset(&cfg, &options(), &control).unwrap();
    let p = control.progress();
    assert_eq!(p.tiles_total, 4 + 6);
    assert_eq!(p.tiles_done, p.tiles_total);
    // "Exported" counts only written tiles; white tiles are counted apart.
    assert_eq!(p.tiles_written, 0);
    assert_eq!(p.tiles_skipped, 10);
    assert_eq!(report.total_tiles_skipped_white, 10);
    assert_eq!(p.phase, DatasetPhase::Done);
}

#[test]
fn legacy_progress_api_reports_a_fixed_total() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 384, 256, 64, |_, _| [90, 60, 160]).unwrap();
    let cfg = config(vec![slide], temp.path().join("out"));
    let (cancel, tiles, current, slides, total) = (
        AtomicBool::new(false),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    );
    let report =
        run_dataset_patches_with_progress(&cfg, &cancel, &tiles, &current, &slides, &total)
            .unwrap();
    assert_eq!(report.total_tiles, 6);
    assert_eq!(total.load(Ordering::Relaxed), 6);
    assert_eq!(tiles.load(Ordering::Relaxed), 6);
    assert_eq!(slides.load(Ordering::Relaxed), 1);
}

#[test]
fn invalid_parameters_and_empty_inputs_are_errors() {
    let temp = tempfile::tempdir().unwrap();
    let mut cfg = config(
        vec![temp.path().join("missing.svs")],
        temp.path().join("out"),
    );
    assert!(matches!(
        run_dataset(&cfg, &options(), &DatasetControl::new()),
        Err(DatasetError::NoSlides { .. })
    ));
    cfg.white_threshold = Some(0.0);
    assert!(matches!(
        run_dataset(&cfg, &options(), &DatasetControl::new()),
        Err(DatasetError::Config(DatasetConfigError::WhiteThreshold(_)))
    ));
}

#[test]
fn huge_grids_are_refused_per_slide() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 512, 512, 64, |_, _| [90, 60, 160]).unwrap();
    let cfg = DatasetPatchesConfig {
        stride: 1,
        ..config(vec![slide], temp.path().join("out"))
    };
    let opts = DatasetPatchesOptions {
        max_candidates_per_slide: 10_000,
        ..options()
    };
    let report = run(&cfg, &opts);
    assert!(matches!(
        report.slides[0].skipped,
        Some(SlideSkipReason::TooManyCandidates {
            candidates: 148_225,
            limit: 10_000
        })
    ));
}

fn mpp_slide(dir: &Path, name: &str, mpp: Option<f64>, colour: [u8; 3]) -> PathBuf {
    let path = dir.join(name);
    let mut spec = PyramidSpec::new(1024, 768).levels(&[1, 2, 4]);
    if let Some(mpp) = mpp {
        spec = spec.mpp(mpp);
    }
    write_pyramid_tiff(&path, &spec, |_, _| colour).unwrap();
    path
}

#[test]
fn patches_are_extracted_at_a_target_resolution() {
    let temp = tempfile::tempdir().unwrap();
    let slide = mpp_slide(temp.path(), "s.tif", Some(0.25), [180, 90, 160]);
    let out = temp.path().join("out");
    // 1.0 µm/px = downsample 4: read level 2 directly.
    let opts = DatasetPatchesOptions {
        resolution: Some(TargetResolution::Mpp(1.0)),
        ..options()
    };
    let cfg = DatasetPatchesConfig {
        tile_size: 64,
        stride: 64,
        ..config(vec![slide.clone()], out.clone())
    };
    let report = run(&cfg, &opts);
    // 1024×768 level 0 → 256×192 at 1 µm/px → 4×3 tiles of 64 px.
    assert_eq!(report.total_tiles, 12);
    let recs = records(&out);
    assert_eq!(recs[0].level, 2);
    assert_eq!((recs[1].x, recs[0].level0_width), (256, 256.0));
    assert!((recs[0].output_mpp_x.unwrap() - 1.0).abs() < 1e-9);
    assert_eq!(centre_colour(&out.join(&recs[0].tile_path)), [180, 90, 160]);

    // 0.75 µm/px = downsample 3: read level 1 (×2) and resample by 1.5.
    let out2 = temp.path().join("out2");
    let opts = DatasetPatchesOptions {
        resolution: Some(TargetResolution::Mpp(0.75)),
        ..options()
    };
    let report = run(
        &DatasetPatchesConfig {
            output_dir: out2.clone(),
            ..cfg.clone()
        },
        &opts,
    );
    assert_eq!(report.total_tiles, 5 * 4);
    let recs = records(&out2);
    assert_eq!(recs[0].level, 1);
    assert!((recs[0].level0_width - 192.0).abs() < 1e-9);
    let img = image::open(out2.join(&recs[0].tile_path)).unwrap();
    assert_eq!((img.width(), img.height()), (64, 64));

    // Magnification needs objective power or MPP; finer than the scan fails.
    let no_mpp = mpp_slide(temp.path(), "nompp.tif", None, [1, 2, 3]);
    let opts = DatasetPatchesOptions {
        resolution: Some(TargetResolution::Mpp(0.1)),
        ..options()
    };
    let report = run(
        &DatasetPatchesConfig {
            output_dir: temp.path().join("out3"),
            inputs: vec![slide, no_mpp],
            ..cfg
        },
        &opts,
    );
    assert!(matches!(
        report.slides[0].skipped,
        Some(SlideSkipReason::ResolutionTooFine { .. })
    ));
    assert!(matches!(
        report.slides[1].skipped,
        Some(SlideSkipReason::NoPixelSize)
    ));
}

#[test]
fn tissue_filter_keeps_only_tissue_patches() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("half.tif");
    write_pyramid_tiff(
        &slide,
        &PyramidSpec::new(512, 256).levels(&[1, 4]),
        |x, _| {
            if x < 256 {
                [190, 110, 170]
            } else {
                [238, 236, 240]
            }
        },
    )
    .unwrap();
    let out = temp.path().join("out");
    let opts = DatasetPatchesOptions {
        tissue_filter: Some(TissueFilter::default()),
        ..options()
    };
    let report = run(&config(vec![slide], out.clone()), &opts);
    assert_eq!(report.total_tiles, 4, "left half only");
    assert_eq!(report.total_tiles_skipped_tissue, 4);
    let recs = records(&out);
    assert!(
        recs.iter()
            .all(|r| r.x < 256 && r.tissue_fraction == Some(1.0))
    );
}

#[test]
fn per_slide_limits_sample_deterministically() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 1024, 1024, 64, |x, y| {
        [(x / 4) as u8, (y / 4) as u8, 100]
    })
    .unwrap();
    let sample = |seed: Option<u64>, out: &str| {
        let opts = DatasetPatchesOptions {
            max_tiles_per_slide: Some(10),
            seed,
            ..options()
        };
        let report = run(&config(vec![slide.clone()], temp.path().join(out)), &opts);
        assert_eq!(report.total_tiles, 10);
        let mut tiles: Vec<(u64, u64)> = records(&temp.path().join(out))
            .iter()
            .map(|r| (r.x, r.y))
            .collect();
        tiles.sort();
        tiles
    };
    let a = sample(Some(7), "a");
    assert_eq!(a, sample(Some(7), "b"), "same seed, same sample");
    assert_ne!(a, sample(Some(8), "c"));
    let spread = sample(None, "d");
    // Evenly spread: both halves of the slide are represented.
    assert!(spread.iter().any(|t| t.1 < 512) && spread.iter().any(|t| t.1 >= 512));
}

#[test]
fn annotation_driven_extraction_with_labels_and_masks() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("case.tif");
    write_tiled_tiff(&slide, 512, 256, 64, |_, _| [150, 100, 180]).unwrap();
    // Tumor covers x ∈ [0, 200), Stroma x ∈ [300, 512).
    std::fs::write(
        temp.path().join("case.geojson"),
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[200,0],[200,256],[0,256],[0,0]]]},
             "properties":{"classification":{"name":"Tumor"}}},
            {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[300,0],[512,0],[512,256],[300,256],[300,0]]]},
             "properties":{"classification":{"name":"Stroma"}}}]}"#,
    )
    .unwrap();
    let out = temp.path().join("out");
    let opts = DatasetPatchesOptions {
        annotations: Some(AnnotationOptions {
            source: AnnotationSource::Sidecar,
            include_classes: vec!["tumor".into()],
            masks: Some(MaskOutput::Binary),
            ..AnnotationOptions::default()
        }),
        ..options()
    };
    let report = run(&config(vec![slide.clone()], out.clone()), &opts);
    // Columns 0 (fully inside) and 1 ([128, 256): 72/128 inside) qualify.
    assert_eq!(report.total_tiles, 4);
    assert_eq!(report.total_tiles_skipped_annotation, 4);
    let recs = records(&out);
    assert!(recs.iter().all(|r| r.label.as_deref() == Some("Tumor")));
    let mask_path = out.join(
        recs.iter()
            .find(|r| r.x == 128)
            .unwrap()
            .mask_path
            .as_ref()
            .unwrap(),
    );
    let mask = image::open(mask_path).unwrap().to_luma8();
    assert_eq!(mask.get_pixel(10, 10).0, [255]);
    assert_eq!(mask.get_pixel(100, 10).0, [0], "beyond x = 200");
    let classes: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("classes.json")).unwrap()).unwrap();
    assert_eq!(classes["1"], "Tumor");

    // Unrestricted: every patch, labelled by majority class.
    let opts = DatasetPatchesOptions {
        annotations: Some(AnnotationOptions {
            source: AnnotationSource::Sidecar,
            restrict_to_annotations: false,
            masks: Some(MaskOutput::Multiclass),
            ..AnnotationOptions::default()
        }),
        ..options()
    };
    let out2 = temp.path().join("out2");
    let report = run(&config(vec![slide], out2.clone()), &opts);
    assert_eq!(report.total_tiles, 8);
    let labels: Vec<Option<String>> = records(&out2)
        .into_iter()
        .filter(|r| r.y == 0)
        .map(|r| r.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            Some("Tumor".into()),
            Some("Tumor".into()),
            Some("Stroma".into()),
            Some("Stroma".into())
        ]
    );
}

#[test]
fn patches_can_be_stain_normalised() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("he.tif");
    let (w, h) = (512u32, 512u32);
    write_pyramid_tiff(&slide, &PyramidSpec::new(w, h).levels(&[1, 4]), |x, y| {
        he_tissue_pixel(x, y, w, h, [236, 234, 238])
    })
    .unwrap();
    let plain = temp.path().join("plain");
    let normalised = temp.path().join("norm");
    let cfg = config(vec![slide], plain.clone());
    run(&cfg, &options());
    let opts = DatasetPatchesOptions {
        stain_normalization: Some(PatchNormalization {
            method: StainFitMethod::Macenko,
            target: None,
        }),
        ..options()
    };
    run(
        &DatasetPatchesConfig {
            output_dir: normalised.clone(),
            ..cfg
        },
        &opts,
    );
    let a = records(&plain);
    let b = records(&normalised);
    let centre = &a
        .iter()
        .find(|r| r.x == 256 && r.y == 256)
        .unwrap()
        .tile_path;
    assert_ne!(
        centre_colour(&plain.join(centre)),
        centre_colour(&normalised.join(centre))
    );
    assert!(b.iter().all(|r| {
        r.stain_normalization
            .as_deref()
            .is_some_and(|s| s.contains("Macenko"))
    }));
}

#[test]
fn dry_runs_plan_without_writing() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 512, 512, 64, |_, _| [90, 60, 160]).unwrap();
    let out = temp.path().join("out");
    let opts = DatasetPatchesOptions {
        dry_run: true,
        ..options()
    };
    let report = run(&config(vec![slide], out.clone()), &opts);
    assert!(report.dry_run);
    assert_eq!(report.planned_tiles, 16);
    assert_eq!(report.total_tiles, 0);
    assert!(!out.exists());
}

#[test]
fn jpeg_patches_are_rgb() {
    let temp = tempfile::tempdir().unwrap();
    let slide = temp.path().join("s.tif");
    write_tiled_tiff(&slide, 256, 128, 64, |_, _| [120, 80, 200]).unwrap();
    let out = temp.path().join("out");
    let opts = DatasetPatchesOptions {
        image_format: PatchImageFormat::Jpeg { quality: 92 },
        ..options()
    };
    run(&config(vec![slide], out.clone()), &opts);
    let recs = records(&out);
    assert!(recs[0].tile_path.ends_with(".jpg"));
    let img = image::open(out.join(&recs[0].tile_path)).unwrap();
    assert_eq!(img.color(), image::ColorType::Rgb8);
}

#[test]
fn deidentified_runs_name_no_slide_and_no_path() {
    let temp = tempfile::tempdir().unwrap();
    let case = temp.path().join("Smith-John-S23-1234");
    std::fs::create_dir_all(&case).unwrap();
    let slide = case.join("S23-1234-A1-HE.tif");
    write_tiled_tiff(&slide, 256, 256, 64, |_, _| [200, 100, 150]).unwrap();
    let broken = case.join("S23-1234-B2-HE.svs");
    std::fs::write(&broken, b"not a slide").unwrap();
    let out = temp.path().join("out");
    let mut config = config(vec![case.clone()], out.clone());
    config.metadata_format = None;
    let deidentify = DatasetPatchesOptions {
        deidentify: true,
        ..options()
    };
    let report = run(&config, &deidentify);
    assert_eq!(report.total_tiles, 4);

    // Nothing written mentions the case folder, a file name or a stem.
    let mut written = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            written.push(path.strip_prefix(&out).unwrap().display().to_string());
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(text) = std::fs::read_to_string(&path) {
                written.push(text);
            }
        }
    }
    let all = written.join("\n");
    for secret in ["Smith", "S23-1234", "A1-HE", "B2-HE"] {
        assert!(!all.contains(secret), "'{secret}' leaked into the dataset");
    }
    assert!(all.contains("slide-"), "slides are named by opaque ids");

    // The same slide gets the same id in a second run (resume).
    let again = run(&config, &deidentify);
    assert_eq!(again.total_tiles_existing, 4);
    let ids: HashSet<String> = records(&out).into_iter().map(|r| r.slide_id).collect();
    assert_eq!(ids.len(), 1);
    assert!(ids.iter().all(|id| id.starts_with("slide-")));
}
