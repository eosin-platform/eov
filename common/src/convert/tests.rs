//! Round trips: synthetic slides are copied and the copies are read back
//! through OpenSlide (`WsiFile`), the `tiff` crate and their raw bytes.

use super::*;
use crate::test_support::{TiffLevel, TiffSpec, write_tiff};
use ::tiff::decoder::{Decoder, DecodingResult};
use ::tiff::tags::Tag;
use std::io::Cursor;

fn pattern(x: u32, y: u32) -> [u8; 3] {
    [
        (x % 256) as u8,
        (y % 256) as u8,
        ((x / 3 + y / 2) % 256) as u8,
    ]
}

/// A smooth pattern for lossy round trips.
fn smooth(x: u32, y: u32) -> [u8; 3] {
    let f = |v: f64| (127.5 + 100.0 * v.sin()) as u8;
    [
        f(x as f64 / 37.0),
        f(y as f64 / 23.0),
        f((x + y) as f64 / 51.0),
    ]
}

/// A 600 × 400 generic TIFF pyramid (0.25 µm/px, 3 levels, 64-px tiles).
fn generic_slide(dir: &Path, name: &str, pixel: fn(u32, u32) -> [u8; 3]) -> PathBuf {
    let path = dir.join(name);
    let mut spec = TiffSpec::pyramid(600, 400, 64, 3);
    spec.pixels_per_cm = Some(40_000);
    spec.description = Some("Stain = H&E".to_string());
    write_tiff(&path, &spec, pixel).unwrap();
    path
}

/// PHI-like values planted in the synthetic Aperio slide.
const PHI: [&str; 11] = [
    "BC-998877",
    "DOE^JANE",
    "Jane Doe",
    "S12-3456",
    "08/11/16",
    "10:18:38",
    "SS1234",
    "jdoe",
    "Case 7",
    "2024:01:02",
    "MRN 4455667",
];

/// An Aperio-style slide with a thumbnail, a label and a macro image and
/// identifiers in its properties and file name.
fn aperio_slide(dir: &Path) -> PathBuf {
    let path = dir.join("S12-3456 A1.svs");
    let mut spec = TiffSpec::aperio(512, 384, 64, 4);
    spec.description = Some(
        "Aperio Image Library v12.0.11\n512x384 (64x64) RAW|AppMag = 20|MPP = 0.4942\
         |Date = 08/11/16|Time = 10:18:38|Title = Case 7 | H&E|Barcode = BC-998877\
         |Patient = DOE^JANE|Patient Name = Jane Doe|Filename = S12-3456|ScanScope ID = SS1234\
         |User = jdoe|Comment = MRN 4455667"
            .to_string(),
    );
    spec.date_time = Some("2024:01:02 03:04:05".to_string());
    write_tiff(&path, &spec, smooth).unwrap();
    path
}

fn convert(source: &Path, output: &Path, options: &ConvertOptions) -> ConvertReport {
    let wsi = WsiFile::open(source).unwrap();
    convert_slide(&wsi, output, options, &JobControl::new()).unwrap()
}

fn deflate(tile_size: u32) -> ConvertOptions {
    ConvertOptions {
        compression: SlideCompression::Deflate,
        tile_size,
        ..ConvertOptions::default()
    }
}

fn rgb_of(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

fn expected(width: u32, height: u32, x0: u32, y0: u32, pixel: fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| pixel(x0 + x, y0 + y)))
        .collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

/// The 2×2 mean pyramid of `level0`, clamping at the edges, as the
/// writer computes it.
fn reference_pyramid(
    level0: Vec<u8>,
    width: u32,
    height: u32,
    levels: usize,
) -> Vec<(u32, u32, Vec<u8>)> {
    let mut out = vec![(width, height, level0)];
    for _ in 1..levels {
        let (w, h, prev) = out.last().unwrap();
        let (w, h) = (*w, *h);
        let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
        let mut next = Vec::with_capacity((nw * nh * 3) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let xs = [2 * x, (2 * x + 1).min(w - 1)];
                let ys = [2 * y, (2 * y + 1).min(h - 1)];
                for c in 0..3 {
                    let mut sum = 0u32;
                    for yy in ys {
                        for xx in xs {
                            sum += u32::from(prev[((yy * w + xx) * 3) as usize + c]);
                        }
                    }
                    next.push(((sum + 2) / 4) as u8);
                }
            }
        }
        out.push((nw, nh, next));
    }
    out
}

fn decoder(bytes: &[u8]) -> Decoder<Cursor<&[u8]>> {
    Decoder::new(Cursor::new(bytes)).unwrap()
}

/// The YCbCr → RGB conversion of JFIF.
fn ycbcr_to_rgb(data: &[u8]) -> Vec<u8> {
    data.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| {
            let (y, cb, cr) = (p[0] as f32, p[1] as f32 - 128.0, p[2] as f32 - 128.0);
            [
                (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8,
                (y - 0.344_136 * cb - 0.714_136 * cr)
                    .round()
                    .clamp(0.0, 255.0) as u8,
                (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

/// Every main-chain image's (width, height, tiled).
fn main_chain(bytes: &[u8]) -> Vec<(u32, u32, bool)> {
    let mut decoder = decoder(bytes);
    let mut images = Vec::new();
    loop {
        let (w, h) = decoder.dimensions().unwrap();
        let tiled = decoder.get_tag_u32(Tag::TileWidth).is_ok();
        images.push((w, h, tiled));
        if !decoder.more_images() {
            break;
        }
        decoder.next_image().unwrap();
    }
    images
}

#[test]
fn deflate_copies_are_exact_pyramids_that_openslide_reads() {
    let dir = tempfile::tempdir().unwrap();
    let source = generic_slide(dir.path(), "source.tif", pattern);
    let output = dir.path().join("copy.tif");
    let report = convert(&source, &output, &deflate(64));
    assert_eq!(report.plan.levels, vec![(600, 400), (300, 200), (150, 100)]);
    assert!(!report.bigtiff);
    assert_eq!(report.unreadable_tiles, 0);
    assert!(!partial_path(&output).exists());

    let copy = WsiFile::open(&output).unwrap();
    let props = copy.properties();
    assert_eq!(props.vendor.as_deref(), Some("generic-tiff"));
    assert_eq!(copy.level_count(), 3);
    assert_eq!((props.width, props.height), (600, 400));
    assert_eq!(copy.level(2).unwrap().width, 150);
    assert!(
        (props.mpp_x.unwrap() - 0.25).abs() < 1e-9,
        "{:?}",
        props.mpp_x
    );
    assert!((props.mpp_y.unwrap() - 0.25).abs() < 1e-9);
    assert_eq!(props.stain.as_deref(), Some("H&E"));
    assert!(props.quickhash1.is_some());
    assert_eq!(copy.property("openslide.level[0].tile-width"), Some("64"));
    let comment = props.comment.as_deref().unwrap();
    assert!(comment.starts_with("Slide copy made by eov "), "{comment}");
    assert!(comment.contains("MPP = 0.25"));
    assert!(comment.contains("Source file = source.tif"));
    assert_eq!(
        copy.property("tiff.Software"),
        Some(crate::export::software_name().as_str())
    );

    let reference = reference_pyramid(expected(600, 400, 0, 0, pattern), 600, 400, 3);
    for (level, (w, h, pixels)) in reference.iter().enumerate() {
        let rgba = copy.read_region(0, 0, level as u32, *w, *h).unwrap();
        assert!(rgb_of(&rgba) == *pixels, "level {level} differs");
    }
}

#[test]
fn many_batches_assemble_the_same_pyramid_as_a_reference() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("big.tif");
    let spec = TiffSpec::single(2000, 1500, 128);
    write_tiff(&source, &spec, pattern).unwrap();
    let output = dir.path().join("copy.tif");
    let report = convert(&source, &output, &deflate(64));
    // Lossless levels stay at least a tile long: 125 × 94 is the last.
    assert_eq!(report.plan.levels.len(), 5);
    assert_eq!(report.plan.levels[4], (125, 94));

    let copy = WsiFile::open(&output).unwrap();
    assert_eq!(copy.level_count(), 5);
    let reference = reference_pyramid(expected(2000, 1500, 0, 0, pattern), 2000, 1500, 5);
    for (level, (w, h, pixels)) in reference.iter().enumerate() {
        let rgba = copy.read_region(0, 0, level as u32, *w, *h).unwrap();
        assert!(rgb_of(&rgba) == *pixels, "level {level} differs");
    }
}

#[test]
fn jpeg_copies_read_through_openslide_and_the_tiff_crate() {
    let dir = tempfile::tempdir().unwrap();
    let source = generic_slide(dir.path(), "smooth.tif", smooth);
    let original = expected(600, 400, 0, 0, smooth);
    for (quality, subsampling, tile) in [(95u8, [1u16, 1], 512u32), (80, [2, 2], 256)] {
        let output = dir.path().join(format!("q{quality}.tif"));
        let options = ConvertOptions {
            compression: SlideCompression::Jpeg { quality },
            tile_size: tile,
            ..ConvertOptions::default()
        };
        let report = convert(&source, &output, &options);
        let copy = WsiFile::open(&output).unwrap();
        assert_eq!(copy.level_count() as usize, report.plan.levels.len());
        let rgba = copy.read_region(0, 0, 0, 600, 400).unwrap();
        let p = psnr(&rgb_of(&rgba), &original);
        assert!(p > 34.0, "quality {quality}: PSNR {p:.1} through OpenSlide");
        // Every level through OpenSlide, the ones smaller than a tile too.
        let reference = reference_pyramid(original.clone(), 600, 400, report.plan.levels.len());
        for (level, (w, h, pixels)) in reference.iter().enumerate() {
            let rgba = copy.read_region(0, 0, level as u32, *w, *h).unwrap();
            let p = psnr(&rgb_of(&rgba), pixels);
            assert!(p > 34.0, "quality {quality}, level {level}: PSNR {p:.1}");
        }

        // Pure-Rust decoding (the tiff crate leaves YCbCr as it is).
        let bytes = std::fs::read(&output).unwrap();
        let mut decoder = decoder(&bytes);
        assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 7);
        assert_eq!(
            decoder.get_tag_u32(Tag::PhotometricInterpretation).unwrap(),
            6
        );
        assert_eq!(
            decoder.get_tag_u32_vec(Tag::Unknown(530)).unwrap(),
            subsampling.map(u32::from).to_vec()
        );
        let tables = decoder.get_tag_u8_vec(Tag::JPEGTables).unwrap();
        assert_eq!((tables[0], tables[1]), (0xFF, 0xD8));
        let DecodingResult::U8(ycbcr) = decoder.read_image().unwrap() else {
            panic!("8-bit samples expected");
        };
        let p = psnr(&ycbcr_to_rgb(&ycbcr), &original);
        assert!(
            p > 34.0,
            "quality {quality}: PSNR {p:.1} through the tiff crate"
        );
    }
}

#[test]
fn ome_tiff_copies_have_a_sub_ifd_pyramid() {
    let dir = tempfile::tempdir().unwrap();
    let source = generic_slide(dir.path(), "source.tif", pattern);
    let output = dir.path().join("copy.ome.tif");
    assert_eq!(SlideFormat::from_path(&output), SlideFormat::OmeTiff);
    let options = ConvertOptions {
        format: SlideFormat::OmeTiff,
        ..deflate(64)
    };
    let report = convert(&source, &output, &options);
    assert_eq!(report.plan.levels.len(), 3);
    let bytes = std::fs::read(&output).unwrap();
    // One image in the main chain; the reduced levels are SubIFDs.
    assert_eq!(main_chain(&bytes), vec![(600, 400, true)]);
    let mut decoder = decoder(&bytes);
    let xml = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap();
    assert!(
        xml.starts_with("<?xml")
            && xml.contains("<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"")
    );
    assert!(xml.contains("SizeX=\"600\" SizeY=\"400\""), "{xml}");
    assert!(xml.contains("PhysicalSizeX=\"0.25\""));
    assert!(xml.contains("<TiffData IFD=\"0\" PlaneCount=\"1\"/>"));
    assert!(xml.is_ascii());
    let sub_ifds = decoder
        .get_tag(Tag::SubIfd)
        .unwrap()
        .into_u64_vec()
        .unwrap();
    assert_eq!(sub_ifds.len(), 2);
    let full = rgb_of(
        &WsiFile::open(&output)
            .unwrap()
            .read_region(0, 0, 0, 600, 400)
            .unwrap(),
    );
    let reference = reference_pyramid(expected(600, 400, 0, 0, pattern), 600, 400, 3);
    assert!(full == reference[0].2);
    for (index, offset) in sub_ifds.iter().enumerate() {
        // The tiff crate reads a SubIFD after pointing it at the offset.
        let mut sub = decoder_at(&bytes, *offset);
        let (w, h) = sub.dimensions().unwrap();
        assert_eq!((w, h), report.plan.levels[index + 1]);
        assert_eq!(sub.get_tag_u32(Tag::NewSubfileType).unwrap(), 1);
        let DecodingResult::U8(pixels) = sub.read_image().unwrap() else {
            panic!("8-bit samples expected");
        };
        assert!(pixels == reference[index + 1].2, "SubIFD {index} differs");
    }
    // eov's own OME-TIFF reader sees the whole pyramid.
    let copy = WsiFile::open(&output).unwrap();
    assert_eq!(copy.properties().vendor.as_deref(), Some("ome-tiff"));
    assert_eq!(copy.level_count(), 3);
    assert!((copy.properties().mpp_x.unwrap() - 0.25).abs() < 1e-9);
    for (level, (w, h, pixels)) in reference.iter().enumerate() {
        let rgba = copy.read_region(0, 0, level as u32, *w, *h).unwrap();
        assert!(rgb_of(&rgba) == *pixels, "level {level} differs");
    }
}

/// OME-TIFF copies with JPEG tiles, associated images and an objective
/// read back through eov's OME-TIFF reader.
#[test]
fn ome_tiff_copies_open_in_eov_with_their_images_and_objective() {
    let dir = tempfile::tempdir().unwrap();
    let source = aperio_slide(dir.path());
    let original = expected(512, 384, 0, 0, smooth);
    let output = dir.path().join("copy.ome.tif");
    let options = ConvertOptions {
        format: SlideFormat::OmeTiff,
        tile_size: 256,
        ..ConvertOptions::default()
    };
    let report = convert(&source, &output, &options);
    assert_eq!(report.plan.levels.len(), 2);
    let copy = WsiFile::open(&output).unwrap();
    let props = copy.properties();
    assert_eq!(props.vendor.as_deref(), Some("ome-tiff"));
    assert_eq!(copy.level_count(), 2);
    assert_eq!(props.objective_power, Some(20.0));
    assert!((props.mpp_x.unwrap() - 0.4942).abs() < 1e-9);
    // YCbCr JPEG tiles decode to the right colours.
    let rgba = copy.read_region(0, 0, 0, 512, 384).unwrap();
    let p = psnr(&rgb_of(&rgba), &original);
    assert!(p > 34.0, "PSNR {p:.1} through eov's OME-TIFF reader");
    // The label, macro and thumbnail are images of their own, by name.
    for name in ["label", "macro", "thumbnail"] {
        assert!(copy.associated_image(name).is_some(), "{name} missing");
    }
    let label = copy.read_associated_image("label").unwrap();
    assert_eq!((label.width, label.height), (24, 16));
    assert_eq!(&label.pixels[..3], &[10, 20, 30]);
}

/// A decoder whose first image is the IFD at `offset` (a TIFF header
/// pointing there, in front of the file's bytes).
fn decoder_at(bytes: &[u8], offset: u64) -> Decoder<Cursor<Vec<u8>>> {
    let mut patched = bytes.to_vec();
    if patched[2] == 43 {
        patched[8..16].copy_from_slice(&offset.to_le_bytes());
    } else {
        patched[4..8].copy_from_slice(&(offset as u32).to_le_bytes());
    }
    Decoder::new(Cursor::new(patched)).unwrap()
}

#[test]
fn regions_levels_and_pixel_sizes() {
    let dir = tempfile::tempdir().unwrap();
    let source = generic_slide(dir.path(), "source.tif", pattern);
    let wsi = WsiFile::open(&source).unwrap();

    // A region at full resolution: exactly those pixels.
    let output = dir.path().join("region.tif");
    let options = ConvertOptions {
        region: Some(SlideBounds {
            x: 100,
            y: 50,
            width: 300,
            height: 200,
        }),
        ..deflate(256)
    };
    let report = convert(&source, &output, &options);
    assert_eq!(report.plan.levels[0], (300, 200));
    let copy = WsiFile::open(&output).unwrap();
    assert!(
        rgb_of(&copy.read_region(0, 0, 0, 300, 200).unwrap())
            == expected(300, 200, 100, 50, pattern)
    );
    assert!(
        copy.properties()
            .comment
            .as_deref()
            .unwrap()
            .contains("Source region = 100,50,300,200")
    );

    // A stored level at its native pixels, with the doubled pixel size.
    let output = dir.path().join("level1.tif");
    let report = convert(
        &source,
        &output,
        &ConvertOptions {
            resolution: OutputResolution::Level(1),
            ..deflate(256)
        },
    );
    assert_eq!(report.plan.levels[0], (300, 200));
    assert!(!report.plan.resampled);
    let copy = WsiFile::open(&output).unwrap();
    assert!((copy.properties().mpp_x.unwrap() - 0.5).abs() < 1e-9);
    let level1 = rgb_of(&wsi.read_region(0, 0, 1, 300, 200).unwrap());
    assert!(rgb_of(&copy.read_region(0, 0, 0, 300, 200).unwrap()) == level1);

    // A pixel size that matches a level uses it; others are resampled.
    let plan = plan_conversion(
        &wsi,
        &ConvertOptions {
            resolution: OutputResolution::Mpp(0.502),
            ..deflate(256)
        },
    )
    .unwrap();
    assert!(!plan.resampled && plan.source_level == 1);
    let output = dir.path().join("resampled.tif");
    let report = convert(
        &source,
        &output,
        &ConvertOptions {
            resolution: OutputResolution::Mpp(0.75),
            ..deflate(256)
        },
    );
    assert!(report.plan.resampled);
    assert_eq!(report.plan.source_level, 1);
    assert_eq!(report.plan.levels[0], (200, 134));
    let copy = WsiFile::open(&output).unwrap();
    assert!((copy.properties().mpp_x.unwrap() - 0.75).abs() < 1e-6);
    // Smooth content survives resampling: compare against the source read
    // at the same scale.
    let smooth_source = generic_slide(dir.path(), "smooth.tif", smooth);
    let output = dir.path().join("smooth-resampled.tif");
    convert(
        &smooth_source,
        &output,
        &ConvertOptions {
            resolution: OutputResolution::Mpp(0.75),
            ..deflate(256)
        },
    );
    let copy = WsiFile::open(&output).unwrap();
    let got = rgb_of(&copy.read_region(0, 0, 0, 200, 133).unwrap());
    let want: Vec<u8> = (0..133)
        .flat_map(|y| (0..200).flat_map(move |x| smooth(x * 3 + 1, y * 3 + 1)))
        .collect();
    let p = psnr(&got, &want);
    assert!(p > 30.0, "resampled PSNR {p:.1}");

    // What cannot work is refused before anything is written.
    for (resolution, needle) in [
        (OutputResolution::Mpp(0.1), "finer"),
        (OutputResolution::Level(7), "no level 7"),
        (OutputResolution::Mpp(-1.0), "pixel size"),
    ] {
        let err = plan_conversion(
            &wsi,
            &ConvertOptions {
                resolution,
                ..ConvertOptions::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains(needle), "{err}");
    }
    let outside = ConvertOptions {
        region: Some(SlideBounds {
            x: 700,
            y: 0,
            width: 10,
            height: 10,
        }),
        ..ConvertOptions::default()
    };
    assert!(
        plan_conversion(&wsi, &outside)
            .unwrap_err()
            .to_string()
            .contains("outside")
    );
    let bad_tile = ConvertOptions {
        tile_size: 100,
        ..ConvertOptions::default()
    };
    assert!(plan_conversion(&wsi, &bad_tile).is_err());

    // A slide without a pixel size cannot be copied at a pixel size.
    let plain = dir.path().join("plain.tif");
    write_tiff(&plain, &TiffSpec::single(64, 64, 32), pattern).unwrap();
    let err = plan_conversion(
        &WsiFile::open(&plain).unwrap(),
        &ConvertOptions {
            resolution: OutputResolution::Mpp(1.0),
            ..ConvertOptions::default()
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("does not record its pixel size"),
        "{err}"
    );
}

#[test]
fn deidentified_copies_contain_no_label_macro_or_phi() {
    let dir = tempfile::tempdir().unwrap();
    let source = aperio_slide(dir.path());
    let wsi = WsiFile::open(&source).unwrap();
    // The source really has them.
    assert_eq!(wsi.properties().vendor.as_deref(), Some("aperio"));
    assert!(wsi.associated_image("label").is_some() && wsi.associated_image("macro").is_some());
    let all: String = wsi
        .property_map()
        .iter()
        .map(|(k, v)| format!("{k}={v}\n"))
        .collect();
    for phi in PHI {
        assert!(all.contains(phi), "the test slide should carry {phi}");
    }

    for (format, keep_scanner, name) in [
        (SlideFormat::Tiff, false, "teaching-01.tif"),
        (SlideFormat::OmeTiff, false, "teaching-01.ome.tif"),
        (SlideFormat::Tiff, true, "teaching-02.tif"),
    ] {
        let output = dir.path().join(name);
        let options = ConvertOptions {
            deidentify: true,
            keep_scanner,
            format,
            ..ConvertOptions::default()
        };
        let report = convert(&source, &output, &options);
        assert!(report.associated_images.is_empty());
        assert!(report.plan.associated_images.is_empty());

        // No associated images: every image in the file is a tiled
        // pyramid level.
        let bytes = std::fs::read(&output).unwrap();
        let chain = main_chain(&bytes);
        let expected_chain = match format {
            SlideFormat::Tiff => report.plan.levels.len(),
            SlideFormat::OmeTiff => 1,
        };
        assert_eq!(chain.len(), expected_chain, "{chain:?}");
        assert!(chain.iter().all(|(_, _, tiled)| *tiled));

        // No identifier anywhere in the file, nor the file name.
        let text = String::from_utf8_lossy(&bytes);
        for phi in PHI
            .iter()
            .chain(["S12-3456 A1", "Barcode", "Patient", "ScanScope"].iter())
        {
            assert!(!text.contains(phi), "{name}: {phi} found in the copy");
        }
        if !keep_scanner {
            assert!(
                !text.to_ascii_lowercase().contains("aperio"),
                "{name}: vendor kept"
            );
        }

        // What it does say.
        let copy = WsiFile::open(&output).unwrap();
        let props = copy.properties();
        let vendor = match format {
            SlideFormat::Tiff => "generic-tiff",
            SlideFormat::OmeTiff => "ome-tiff",
        };
        assert_eq!(props.vendor.as_deref(), Some(vendor));
        assert!(copy.associated_images().is_empty());
        assert!((props.mpp_x.unwrap() - 0.4942).abs() < 1e-6);
        let description = match format {
            SlideFormat::Tiff => props.comment.clone().unwrap(),
            SlideFormat::OmeTiff => decoder(&bytes)
                .get_tag_ascii_string(Tag::ImageDescription)
                .unwrap(),
        };
        let headline = format!(
            "De-identified slide copy made by {} on ",
            crate::export::software_name()
        );
        assert!(description.contains(&headline), "{description}");
        assert!(description.contains("Objective power = 20"));
        assert!(description.contains("Stain = H&"));
        for (key, value) in copy.property_map() {
            for phi in PHI {
                assert!(
                    !value.contains(phi) && !key.contains(phi),
                    "{key} = {value}"
                );
            }
        }
        if keep_scanner {
            assert!(description.contains("Source format = aperio"));
        }
    }
}

#[test]
fn plain_copies_keep_associated_images_and_properties() {
    let dir = tempfile::tempdir().unwrap();
    let source = aperio_slide(dir.path());
    let output = dir.path().join("copy.tif");
    let report = convert(&source, &output, &ConvertOptions::default());
    let mut names = report.associated_images.clone();
    names.sort();
    assert_eq!(names, ["label", "macro", "thumbnail"]);
    let bytes = std::fs::read(&output).unwrap();
    let chain = main_chain(&bytes);
    let stripped: Vec<(u32, u32)> = chain
        .iter()
        .filter(|(_, _, tiled)| !tiled)
        .map(|(w, h, _)| (*w, *h))
        .collect();
    assert!(
        stripped.contains(&(24, 16)) && stripped.contains(&(48, 16)),
        "{chain:?}"
    );
    // The label's pixels, through the tiff crate.
    let mut decoder = decoder(&bytes);
    loop {
        if decoder.dimensions().unwrap() == (24, 16) {
            let DecodingResult::U8(pixels) = decoder.read_image().unwrap() else {
                panic!("8-bit samples expected");
            };
            assert_eq!(&pixels[..3], &[10, 20, 30]);
            break;
        }
        decoder.next_image().unwrap();
    }
    let copy = WsiFile::open(&output).unwrap();
    let comment = copy.properties().comment.clone().unwrap();
    assert!(comment.contains("aperio.Barcode = BC-998877"), "{comment}");
    assert!(comment.contains("Source file = S12-3456 A1.svs"));
    assert!(comment.contains("De-identified = no"));
    assert_eq!(copy.properties().vendor.as_deref(), Some("generic-tiff"));
}

#[test]
fn failed_and_cancelled_conversions_leave_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let source = generic_slide(dir.path(), "source.tif", pattern);
    let wsi = WsiFile::open(&source).unwrap();
    let output = dir.path().join("copy.tif");

    let control = JobControl::new();
    control.cancel();
    let err = convert_slide(&wsi, &output, &deflate(256), &control).unwrap_err();
    assert!(matches!(err, ConvertError::Cancelled));
    assert!(!output.exists() && !partial_path(&output).exists());

    std::fs::write(&output, b"keep me").unwrap();
    let err = convert_slide(&wsi, &output, &deflate(256), &JobControl::new()).unwrap_err();
    assert!(matches!(err, ConvertError::OutputExists(_)));
    assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
    let overwrite = ConvertOptions {
        overwrite: true,
        ..deflate(256)
    };
    convert_slide(&wsi, &output, &overwrite, &JobControl::new()).unwrap();
    assert!(WsiFile::open(&output).is_ok());

    // Never onto the slide itself, even with overwrite.
    let err = convert_slide(&wsi, &source, &overwrite, &JobControl::new()).unwrap_err();
    assert!(err.to_string().contains("slide being copied"), "{err}");
    let err = convert_slide(
        &wsi,
        &dir.path().join("no/such/dir/x.tif"),
        &overwrite,
        &JobControl::new(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        2,
        "only the source and the copy"
    );
}

#[test]
fn bigtiff_and_icc_profiles() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("icc.tif");
    let mut spec = TiffSpec::pyramid(300, 200, 64, 2);
    spec.icc_profile = Some((0..200u32).map(|i| (i * 7 % 256) as u8).collect());
    write_tiff(&source, &spec, pattern).unwrap();

    let output = dir.path().join("big.tif");
    let report = convert(
        &source,
        &output,
        &ConvertOptions {
            bigtiff: true,
            ..deflate(256)
        },
    );
    assert!(report.bigtiff && report.icc_profile);
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(&bytes[..4], &[b'I', b'I', 43, 0]);
    let copy = WsiFile::open(&output).unwrap();
    assert_eq!(copy.icc_profile().unwrap(), spec.icc_profile);
    assert!(
        rgb_of(&copy.read_region(0, 0, 0, 300, 200).unwrap()) == expected(300, 200, 0, 0, pattern)
    );

    let output = dir.path().join("no-icc.tif");
    let report = convert(
        &source,
        &output,
        &ConvertOptions {
            keep_icc_profile: false,
            ..deflate(256)
        },
    );
    assert!(!report.icc_profile && !report.bigtiff);
    assert_eq!(WsiFile::open(&output).unwrap().icc_profile().unwrap(), None);
}

#[test]
fn unreadable_source_tiles_become_background() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("broken.tif");
    // Two levels: OpenSlide hashes the smallest one when it opens a slide.
    let spec = TiffSpec {
        levels: vec![
            TiffLevel::new(256, 128, 64).with_broken_tile(1, 0),
            TiffLevel::new(128, 64, 64),
        ],
        ..TiffSpec::default()
    };
    write_tiff(&source, &spec, pattern).unwrap();
    let output = dir.path().join("copy.tif");
    let report = convert(&source, &output, &deflate(64));
    assert_eq!(report.unreadable_tiles, 1);
    assert!(report.shortfall_message().is_some());
    let copy = WsiFile::open(&output).unwrap();
    let rgba = copy.read_region(64, 0, 0, 64, 64).unwrap();
    assert!(rgba.as_chunks::<4>().0.iter().all(|p| *p == [255; 4]));
    let rgba = copy.read_region(0, 0, 0, 64, 64).unwrap();
    assert!(rgb_of(&rgba) == expected(64, 64, 0, 0, pattern));
}

#[test]
fn partial_files_are_named_next_to_the_output() {
    let partial = partial_path(Path::new("/data/out/slide.tif"));
    assert_eq!(partial.parent(), Some(Path::new("/data/out")));
    let name = partial.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        name.starts_with("slide.tif.") && name.ends_with(".eov-partial"),
        "{name}"
    );
    assert_eq!(
        SlideFormat::from_path(Path::new("a.OME.TIFF")),
        SlideFormat::OmeTiff
    );
    assert_eq!(
        SlideFormat::from_path(Path::new("a.tif")),
        SlideFormat::Tiff
    );
}

#[test]
fn copies_record_their_objective_power_for_readers() {
    let dir = tempfile::tempdir().unwrap();
    let source = aperio_slide(dir.path());
    for deidentify in [false, true] {
        let output = dir.path().join(format!("copy-{deidentify}.tif"));
        let options = ConvertOptions {
            deidentify,
            resolution: OutputResolution::Level(1),
            ..ConvertOptions::default()
        };
        convert(&source, &output, &options);
        let copy = WsiFile::open(&output).unwrap();
        // OpenSlide's generic TIFF driver reports none itself.
        assert_eq!(copy.properties().objective_power, None);
        let comment = copy.properties().comment.clone().unwrap();
        // 20× at level 0; level 1 is 4× smaller.
        assert_eq!(
            objective_power_from_description(&comment),
            Some(5.0),
            "{comment}"
        );
    }
    assert_eq!(
        objective_power_from_description("Objective power = 20"),
        None
    );
    assert_eq!(
        objective_power_from_description(
            "Slide copy made by eov 1.0.0 on 2026-10-15\nSource properties:\nObjective power = 7"
        ),
        None,
        "a source property is not the copy's"
    );
}

/// A multichannel slide is copied as the RGB composite eov displays.
#[test]
fn multichannel_slides_are_copied_as_their_composite() {
    use crate::test_support::{OmeTiffSpec, write_ome_tiff};

    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cycif.ome.tif");
    let spec = OmeTiffSpec::channels(
        256,
        192,
        64,
        2,
        &[("DAPI", [0, 0, 255]), ("CD3", [0, 255, 0])],
    );
    write_ome_tiff(&source, &spec, |channel, x, y| {
        f64::from(if channel == 0 { x * 200 } else { y * 300 })
    })
    .unwrap();
    let wsi = WsiFile::open(&source).unwrap();
    assert!(wsi.is_multichannel());
    let composite = rgb_of(&wsi.read_region(0, 0, 0, 256, 192).unwrap());

    let output = dir.path().join("composite.tif");
    let report = convert(&source, &output, &deflate(64));
    assert_eq!(report.plan.composite_channels, Some(2));
    let copy = WsiFile::open(&output).unwrap();
    assert!(!copy.is_multichannel());
    assert!(rgb_of(&copy.read_region(0, 0, 0, 256, 192).unwrap()) == composite);
    let comment = copy.properties().comment.clone().unwrap();
    assert!(
        comment.contains("Channels = RGB composite of 2 channels, as displayed"),
        "{comment}"
    );
    assert!((copy.properties().mpp_x.unwrap() - 0.5).abs() < 1e-9);
}
