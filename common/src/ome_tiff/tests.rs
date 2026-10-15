//! Tests of the OME-TIFF reader, through [`WsiFile`] as the rest of eov
//! uses it, on files written by [`crate::test_support::write_ome_tiff`].

use super::*;
use crate::channels::{ChannelDisplay, ChannelSettings, Palette};
use crate::test_support::{
    OmeChannelSpec, OmeCompression, OmeExtraImage, OmePyramid, OmeSample, OmeTiffSpec,
    ome_level0_pixel, write_ome_tiff,
};
use crate::{TileManager, WsiFile};
use std::path::PathBuf;

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];

/// Channel values of the standard test slide: ramps in x, in y and along
/// the diagonal, up to about 1000.
fn ramps(channel: usize, x: u32, y: u32) -> f64 {
    f64::from(match channel {
        0 => (x * 3) % 1200,
        1 => (y * 5) % 1200,
        _ => ((x + y) * 2) % 1200,
    })
}

fn three_channels() -> OmeTiffSpec {
    OmeTiffSpec::channels(
        300,
        200,
        64,
        3,
        &[("DAPI", BLUE), ("CD45", GREEN), ("Pan-CK", RED)],
    )
}

fn write(dir: &Path, name: &str, spec: &OmeTiffSpec) -> PathBuf {
    let path = dir.join(name);
    write_ome_tiff(&path, spec, ramps).unwrap();
    path
}

/// The same window, colour and gamma for every channel.
fn display(channels: &[[u8; 3]], min: f64, max: f64, gamma: f64) -> ChannelDisplay {
    ChannelDisplay {
        channels: channels
            .iter()
            .map(|&color| ChannelSettings {
                visible: true,
                color,
                min,
                max,
                gamma,
            })
            .collect(),
    }
}

/// What a pixel with these channel values composites to.
fn composite_of(display: &ChannelDisplay, values: &[f64]) -> [u8; 4] {
    let mut rgb = [0u8; 3];
    for (settings, &value) in display.channels.iter().zip(values) {
        if !settings.visible {
            continue;
        }
        let intensity = settings.intensity(value);
        for (out, color) in rgb.iter_mut().zip(settings.color) {
            *out = out.saturating_add((f64::from(color) * intensity).round() as u8);
        }
    }
    [rgb[0], rgb[1], rgb[2], 255]
}

/// Reads a `width` x `height` region at level pixel (`lx`, `ly`) of
/// `level` the way the tile loader does (level-0 coordinates truncated).
fn read_level(wsi: &WsiFile, level: u32, lx: i64, ly: i64, width: u32, height: u32) -> Vec<u8> {
    let ds = wsi.level(level).unwrap().downsample;
    wsi.read_region(
        (lx as f64 * ds) as i64,
        (ly as f64 * ds) as i64,
        level,
        width,
        height,
    )
    .unwrap()
}

/// Checks a region read against the value function.
fn assert_region(
    wsi: &WsiFile,
    spec: &OmeTiffSpec,
    level: u32,
    (lx, ly, width, height): (i64, i64, u32, u32),
    expected_at: impl Fn(u32, u32) -> [u8; 4],
    outside: [u8; 4],
) {
    let pixels = read_level(wsi, level, lx, ly, width, height);
    let (lw, lh) = spec.level_size(level);
    for j in 0..height {
        for i in 0..width {
            let (x, y) = (lx + i64::from(i), ly + i64::from(j));
            let at = ((j * width + i) * 4) as usize;
            let got = &pixels[at..at + 4];
            let want = if x < 0 || y < 0 || x >= i64::from(lw) || y >= i64::from(lh) {
                outside
            } else {
                expected_at(x as u32, y as u32)
            };
            assert_eq!(got, want, "level {level} pixel ({x}, {y})");
        }
    }
}

fn channel_values(spec: &OmeTiffSpec, level: u32, x: u32, y: u32, channels: usize) -> Vec<f64> {
    let (x0, y0) = ome_level0_pixel(spec, level, x, y);
    (0..channels).map(|c| ramps(c, x0, y0)).collect()
}

#[test]
fn opens_a_multichannel_pyramid_with_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let spec = three_channels();
    let wsi = WsiFile::open(write(dir.path(), "slide.ome.tif", &spec)).unwrap();
    let props = wsi.properties();
    assert_eq!(props.vendor.as_deref(), Some(VENDOR));
    assert_eq!((props.width, props.height), (300, 200));
    let sizes: Vec<_> = props.levels.iter().map(|l| (l.width, l.height)).collect();
    assert_eq!(sizes, [(300, 200), (150, 100), (75, 50)]);
    assert_eq!(props.levels[1].downsample, 2.0);
    assert_eq!(props.levels[2].downsample, 4.0);
    assert!(props.levels.iter().all(|l| l.tile_width == Some(64)));
    assert_eq!(props.mpp_x, Some(0.5));
    assert_eq!(props.mpp_y, Some(0.5));
    assert_eq!(props.mpp(), Some(0.5));
    assert_eq!(props.objective_power, Some(20.0));
    assert_eq!(props.scan_date.as_deref(), Some("2024-01-02T03:04:05"));
    assert_eq!(props.background_color, [0, 0, 0]);
    assert_eq!(props.stain, None);
    assert_eq!(wsi.best_level_for_downsample(3.0), 1);

    assert!(wsi.is_multichannel());
    let set = wsi.channels().unwrap();
    assert_eq!(set.sample_type, SampleType::U16);
    let names: Vec<&str> = set.channels.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["DAPI", "CD45", "Pan-CK"]);
    let colors: Vec<_> = set.channels.iter().map(|c| c.color).collect();
    assert_eq!(colors, [Some(BLUE), Some(GREEN), Some(RED)]);
    // The file's colours are distinct, so they are the defaults.
    let defaults = wsi.default_channel_display().unwrap();
    assert_eq!(
        defaults
            .channels
            .iter()
            .map(|c| c.color)
            .collect::<Vec<_>>(),
        [BLUE, GREEN, RED]
    );
    assert_eq!(wsi.channel_display(), Some(defaults));

    let map = wsi.property_map();
    assert_eq!(map["openslide.vendor"], VENDOR);
    assert_eq!(map["openslide.level-count"], "3");
    assert_eq!(map["openslide.level[2].width"], "75");
    assert_eq!(map["openslide.mpp-x"], "0.5");
    assert_eq!(map["openslide.objective-power"], "20");
    assert_eq!(map["ome.pixels.size-c"], "3");
    assert_eq!(map["ome.pixels.type"], "uint16");
    assert_eq!(map["ome.pixels.physical-size-x"], "0.5 µm");
    assert_eq!(map["ome.channel[1].name"], "CD45");
    assert_eq!(map["ome.channel[2].color"], "#FF0000");
    assert_eq!(map["ome.pyramid"], "SubIFDs");
    assert_eq!(map["ome.objective.nominal-magnification"], "20");
    assert!(wsi.associated_images().is_empty());
    assert_eq!(wsi.icc_profile().unwrap(), None);
    assert_eq!(wsi.health().state, crate::HealthState::Healthy);
}

#[test]
fn region_reads_composite_every_level() {
    let dir = tempfile::tempdir().unwrap();
    let spec = three_channels();
    let wsi = WsiFile::open(write(dir.path(), "slide.ome.tif", &spec)).unwrap();
    let shown = display(&[BLUE, GREEN, RED], 100.0, 900.0, 1.0);
    wsi.set_channel_display(shown.clone()).unwrap();
    for level in 0..3 {
        let (lw, lh) = spec.level_size(level);
        // Across tile borders, past the right and bottom edges, and from
        // negative coordinates.
        for rect in [
            (60, 60, 10, 9),
            (i64::from(lw) - 5, i64::from(lh) - 4, 12, 8),
            (-3, -2, 8, 6),
            (0, 0, lw, lh),
        ] {
            assert_region(
                &wsi,
                &spec,
                level,
                (rect.0, rect.1, rect.2, rect.3),
                |x, y| composite_of(&shown, &channel_values(&spec, level, x, y, 3)),
                [0, 0, 0, 255],
            );
        }
    }
    // Areas without data are transparent in the straight and premultiplied
    // modes, and opaque inside.
    let straight = wsi
        .read_region_with(-2, 0, 0, 4, 1, AlphaMode::Straight)
        .unwrap();
    assert_eq!(&straight[..4], [0, 0, 0, 0]);
    assert_eq!(straight[11], 255);
    assert!(
        wsi.read_region(1000, 1000, 0, 4, 4)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [0, 0, 0, 255])
    );
}

#[test]
fn compositing_follows_colours_windows_gamma_and_visibility() {
    let dir = tempfile::tempdir().unwrap();
    let spec = three_channels();
    let wsi = WsiFile::open(write(dir.path(), "slide.ome.tif", &spec)).unwrap();
    // (100, 50): DAPI 300, CD45 250, Pan-CK 300.
    let at = |wsi: &WsiFile| wsi.read_region(100, 50, 0, 1, 1).unwrap();

    let mut shown = display(&[BLUE, GREEN, RED], 0.0, 600.0, 1.0);
    wsi.set_channel_display(shown.clone()).unwrap();
    assert_eq!(at(&wsi), [128, 106, 128, 255]);

    // One channel alone.
    for (index, channel) in shown.channels.iter_mut().enumerate() {
        channel.visible = index == 1;
    }
    wsi.set_channel_display(shown.clone()).unwrap();
    assert_eq!(at(&wsi), [0, 106, 0, 255]);

    // A narrower window saturates; gamma above 1 brightens.
    shown.channels[1].max = 200.0;
    wsi.set_channel_display(shown.clone()).unwrap();
    assert_eq!(at(&wsi), [0, 255, 0, 255]);
    shown.channels[1].max = 1000.0;
    shown.channels[1].gamma = 2.0;
    wsi.set_channel_display(shown.clone()).unwrap();
    // (250 / 1000)^(1/2) = 0.5.
    assert_eq!(at(&wsi), [0, 128, 0, 255]);

    // Colours mix additively and clamp.
    let mut white = display(&[[255, 255, 255]; 3], 0.0, 300.0, 1.0);
    wsi.set_channel_display(white.clone()).unwrap();
    assert_eq!(at(&wsi), [255, 255, 255, 255]);
    white.apply_palette(Palette::OkabeIto);
    assert_eq!(white.channels[0].color, Palette::OkabeIto.color(0));

    // Nothing visible: black.
    let mut none = display(&[RED, GREEN, BLUE], 0.0, 1.0, 1.0);
    for channel in &mut none.channels {
        channel.visible = false;
    }
    wsi.set_channel_display(none).unwrap();
    assert_eq!(at(&wsi), [0, 0, 0, 255]);

    // The display is checked against the slide.
    assert!(
        wsi.set_channel_display(display(&[RED], 0.0, 1.0, 1.0))
            .is_err()
    );
}

#[test]
fn display_settings_are_shared_by_every_handle_of_the_slide() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "slide.ome.tif", &three_channels());
    let a = WsiFile::open(&path).unwrap();
    let b = WsiFile::open(&path).unwrap().with_synthetic_levels();
    assert!(a.shares_handle_with(&b));
    let shown = display(&[RED, RED, RED], 0.0, 10.0, 1.0);
    a.set_channel_display(shown.clone()).unwrap();
    assert_eq!(b.channel_display(), Some(shown));
    assert_eq!(&b.read_region(100, 50, 0, 1, 1).unwrap(), &[255, 0, 0, 255]);
}

#[test]
fn auto_contrast_comes_from_low_resolution_percentiles() {
    let dir = tempfile::tempdir().unwrap();
    // Channel 0: a dim background of 100 with a bright quarter at 3000;
    // channel 1: zeros (no data) except a ramp in the bottom half.
    let spec = OmeTiffSpec::channels(256, 256, 64, 3, &[("A", RED), ("B", GREEN)]);
    let path = dir.path().join("stats.ome.tif");
    write_ome_tiff(&path, &spec, |c, x, y| match c {
        0 if x < 128 && y < 128 => 3000.0,
        0 => 100.0,
        _ if y >= 128 => f64::from(x) * 10.0 + 1.0,
        _ => 0.0,
    })
    .unwrap();
    let wsi = WsiFile::open(&path).unwrap();
    let set = wsi.channels().unwrap();
    let a = &set.channels[0].stats;
    assert!(a.samples > 0);
    assert_eq!((a.min, a.max), (100.0, 3000.0));
    assert_eq!((a.low, a.high), (100.0, 3000.0));
    let b = &set.channels[1].stats;
    assert_eq!(b.min, 0.0);
    // Zeros are left out of the low percentile.
    assert!(b.low >= 1.0 && b.low < 30.0, "{b:?}");
    assert!(b.high > 2400.0, "{b:?}");
    let defaults = wsi.default_channel_display().unwrap();
    assert_eq!(
        (defaults.channels[0].min, defaults.channels[0].max),
        (100.0, 3000.0)
    );
    let mut changed = defaults.clone();
    changed.channels[0].min = 7.0;
    changed.auto_contrast(set);
    assert_eq!(changed, defaults);
}

#[test]
fn rgb_ome_tiff_reads_as_rgb() {
    let dir = tempfile::tempdir().unwrap();
    let spec = OmeTiffSpec::rgb(200, 120, 32, 3);
    let wsi = WsiFile::open(write(dir.path(), "brightfield.ome.tiff", &spec)).unwrap();
    assert!(!wsi.is_multichannel());
    assert!(wsi.channels().is_none());
    assert!(wsi.channel_display().is_none());
    assert!(
        wsi.set_channel_display(display(&[RED], 0.0, 1.0, 1.0))
            .is_err()
    );
    let props = wsi.properties();
    assert_eq!(props.background_color, [255, 255, 255]);
    assert_eq!(props.levels.len(), 3);
    let rgb = |level: u32, x: u32, y: u32| {
        let (x0, y0) = ome_level0_pixel(&spec, level, x, y);
        let v = |c| ramps(c, x0, y0).min(255.0) as u8;
        [v(0), v(1), v(2), 255]
    };
    for level in 0..3 {
        let (lw, lh) = spec.level_size(level);
        assert_region(
            &wsi,
            &spec,
            level,
            (i64::from(lw) - 40, i64::from(lh) - 30, 50, 40),
            |x, y| rgb(level, x, y),
            [255, 255, 255, 255],
        );
    }
}

#[test]
fn compression_variants_decode_identically() {
    let dir = tempfile::tempdir().unwrap();
    for sample in [
        OmeSample::U8,
        OmeSample::U16,
        OmeSample::I16,
        OmeSample::F32,
    ] {
        let mut base = three_channels();
        base.sample = sample;
        base.levels = 2;
        let reference = WsiFile::open(write(dir.path(), "reference.ome.tif", &base)).unwrap();
        let shown = display(&[BLUE, GREEN, RED], 50.0, 250.0, 1.0);
        reference.set_channel_display(shown.clone()).unwrap();
        let expected = reference.read_region(0, 0, 0, 300, 200).unwrap();
        let expected_1 = read_level(&reference, 1, 0, 0, 150, 100);
        for compression in [
            OmeCompression::Lzw,
            OmeCompression::Deflate,
            OmeCompression::Zstd,
            OmeCompression::PackBits,
        ] {
            for predictor in [false, true] {
                let mut spec = base.clone();
                spec.compression = compression;
                spec.predictor = predictor;
                let name = format!("{sample:?}-{compression:?}-{predictor}.ome.tif");
                let wsi = WsiFile::open(write(dir.path(), &name, &spec)).unwrap();
                wsi.set_channel_display(shown.clone()).unwrap();
                assert_eq!(
                    wsi.read_region(0, 0, 0, 300, 200).unwrap(),
                    expected,
                    "{name}"
                );
                assert_eq!(read_level(&wsi, 1, 0, 0, 150, 100), expected_1, "{name}");
            }
        }
    }
}

/// Largest per-sample difference.
fn max_difference(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn jpeg_tiles_decode_with_and_without_shared_tables() {
    let dir = tempfile::tempdir().unwrap();
    // A smooth image: JPEG keeps it close.
    let smooth = |c: usize, x: u32, y: u32| f64::from(60 + (x + y * 2 + c as u32 * 30) / 4);
    for (rgb, compression) in [
        (false, OmeCompression::Jpeg),
        (false, OmeCompression::JpegTables),
        (true, OmeCompression::Jpeg),
        (true, OmeCompression::JpegTables),
    ] {
        let mut spec = if rgb {
            OmeTiffSpec::rgb(128, 96, 32, 2)
        } else {
            let mut spec = OmeTiffSpec::channels(128, 96, 32, 2, &[("A", RED), ("B", GREEN)]);
            spec.sample = OmeSample::U8;
            spec
        };
        let reference_path = dir.path().join(format!("raw-{rgb}.ome.tif"));
        write_ome_tiff(&reference_path, &spec, smooth).unwrap();
        spec.compression = compression;
        let path = dir.path().join(format!("{rgb}-{compression:?}.ome.tif"));
        write_ome_tiff(&path, &spec, smooth).unwrap();
        let reference = WsiFile::open(&reference_path).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        if !rgb {
            let shown = display(&[RED, GREEN], 0.0, 255.0, 1.0);
            reference.set_channel_display(shown.clone()).unwrap();
            wsi.set_channel_display(shown).unwrap();
        }
        for level in 0..2 {
            let a = read_level(&reference, level, 0, 0, 64, 48);
            let b = read_level(&wsi, level, 0, 0, 64, 48);
            assert!(
                max_difference(&a, &b) <= 12,
                "{rgb} {compression:?} level {level}: {}",
                max_difference(&a, &b)
            );
        }
    }
}

#[test]
fn strips_bigtiff_big_endian_and_planar_layouts_read_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let base = three_channels();
    let reference = WsiFile::open(write(dir.path(), "reference.ome.tif", &base)).unwrap();
    let shown = display(&[BLUE, GREEN, RED], 0.0, 1000.0, 1.0);
    reference.set_channel_display(shown.clone()).unwrap();
    let expected: Vec<Vec<u8>> = (0..3)
        .map(|level| read_level(&reference, level, 3, 5, 70, 60))
        .collect();
    let variants: Vec<(&str, OmeTiffSpec)> = vec![
        (
            "strips",
            OmeTiffSpec {
                tile: None,
                rows_per_strip: 7,
                compression: OmeCompression::Lzw,
                ..base.clone()
            },
        ),
        (
            "bigtiff",
            OmeTiffSpec {
                big_tiff: true,
                compression: OmeCompression::Deflate,
                ..base.clone()
            },
        ),
        (
            "big-endian",
            OmeTiffSpec {
                big_endian: true,
                predictor: true,
                compression: OmeCompression::Lzw,
                ..base.clone()
            },
        ),
        (
            "big-endian-bigtiff",
            OmeTiffSpec {
                big_endian: true,
                big_tiff: true,
                ..base.clone()
            },
        ),
        (
            "no-tiff-data",
            OmeTiffSpec {
                omit_tiff_data: true,
                ..base.clone()
            },
        ),
    ];
    for (name, spec) in variants {
        let wsi = WsiFile::open(write(dir.path(), &format!("{name}.ome.tif"), &spec)).unwrap();
        wsi.set_channel_display(shown.clone()).unwrap();
        assert_eq!(wsi.level_count(), 3, "{name}");
        for (level, expected) in expected.iter().enumerate() {
            assert_eq!(
                &read_level(&wsi, level as u32, 3, 5, 70, 60),
                expected,
                "{name} level {level}"
            );
        }
    }
    let strips = WsiFile::open(dir.path().join("strips.ome.tif")).unwrap();
    assert_eq!(strips.level(0).unwrap().tile_width, None);

    // RGB stored as separate sample planes.
    let rgb = OmeTiffSpec::rgb(90, 70, 32, 2);
    let chunky = WsiFile::open(write(dir.path(), "chunky.ome.tif", &rgb)).unwrap();
    let planar = WsiFile::open(write(
        dir.path(),
        "planar.ome.tif",
        &OmeTiffSpec {
            planar: true,
            ..rgb
        },
    ))
    .unwrap();
    assert_eq!(
        planar.read_region(0, 0, 0, 90, 70).unwrap(),
        chunky.read_region(0, 0, 0, 90, 70).unwrap()
    );
}

#[test]
fn pyramids_stored_as_images_or_reduced_ifds_are_found() {
    let dir = tempfile::tempdir().unwrap();
    let base = three_channels();
    let reference = WsiFile::open(write(dir.path(), "subifds.ome.tif", &base)).unwrap();
    let shown = display(&[BLUE, GREEN, RED], 0.0, 1000.0, 1.0);
    reference.set_channel_display(shown.clone()).unwrap();
    for (pyramid, name) in [
        (OmePyramid::Images, "separate images"),
        (OmePyramid::ReducedIfds, "reduced-resolution IFDs"),
    ] {
        let spec = OmeTiffSpec {
            pyramid,
            ..base.clone()
        };
        let wsi = WsiFile::open(write(dir.path(), &format!("{pyramid:?}.ome.tif"), &spec)).unwrap();
        wsi.set_channel_display(shown.clone()).unwrap();
        assert_eq!(wsi.level_count(), 3, "{pyramid:?}");
        assert_eq!(wsi.property("ome.pyramid"), Some(name));
        assert_eq!(wsi.level(2).unwrap().downsample, 4.0);
        for level in 0..3 {
            assert_eq!(
                read_level(&wsi, level, 0, 0, 40, 30),
                read_level(&reference, level, 0, 0, 40, 30),
                "{pyramid:?} level {level}"
            );
        }
    }
    // A single level is a pyramid of one.
    let flat = WsiFile::open(write(
        dir.path(),
        "flat.ome.tif",
        &OmeTiffSpec { levels: 1, ..base },
    ))
    .unwrap();
    assert_eq!(flat.level_count(), 1);
    assert_eq!(flat.property("ome.pyramid"), Some("none"));
}

#[test]
fn ome_xml_is_recognised_in_plain_tif_files_and_plain_tiffs_still_use_openslide() {
    let dir = tempfile::tempdir().unwrap();
    // OME-XML in a .tif: eov's reader.
    let wsi = WsiFile::open(write(dir.path(), "sniffed.tif", &three_channels())).unwrap();
    assert_eq!(wsi.properties().vendor.as_deref(), Some(VENDOR));
    assert!(is_ome_tiff(&dir.path().join("sniffed.tif")));

    // A generic RGB TIFF keeps going to OpenSlide, even when misnamed.
    let plain = dir.path().join("plain.tif");
    crate::test_support::write_tiled_tiff(&plain, 64, 64, 32, |x, y| [x as u8, y as u8, 0])
        .unwrap();
    assert!(!is_ome_tiff(&plain));
    let wsi = WsiFile::open(&plain).unwrap();
    assert_eq!(wsi.properties().vendor.as_deref(), Some("generic-tiff"));
    let misnamed = dir.path().join("misnamed.ome.tif");
    std::fs::copy(&plain, &misnamed).unwrap();
    let wsi = WsiFile::open(&misnamed).unwrap();
    assert_eq!(wsi.properties().vendor.as_deref(), Some("generic-tiff"));
    assert!(!wsi.is_multichannel());
}

#[test]
fn slide_identity_is_keyed_by_content() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "a.ome.tif", &three_channels());
    let copy = dir.path().join("renamed.ome.tif");
    std::fs::copy(&path, &copy).unwrap();
    let mut other_spec = three_channels();
    other_spec.channels[0].name = "Hoechst".into();
    let other = write(dir.path(), "b.ome.tif", &other_spec);

    let a = WsiFile::open(&path).unwrap().identity().unwrap();
    let renamed = WsiFile::open(&copy).unwrap().identity().unwrap();
    let different = WsiFile::open(&other).unwrap().identity().unwrap();
    assert!(a.key().starts_with("eov-slide-1:"));
    assert_eq!(a.key(), renamed.key(), "renaming keeps the identity");
    assert_ne!(a.key(), different.key());
    assert_eq!(a.quickhash1, None);
    assert_eq!(a.vendor.as_deref(), Some(VENDOR));
    assert_eq!((a.width, a.height, a.level_count), (300, 200, 3));
    assert_eq!(a.mpp_x, Some(0.5));
    assert_eq!(a.members.len(), 1);
}

#[test]
fn thumbnails_tiles_and_synthetic_levels_work_on_composites() {
    let dir = tempfile::tempdir().unwrap();
    // One level, 4096 wide: synthetic levels are added.
    let mut spec = OmeTiffSpec::channels(4096, 64, 256, 1, &[("A", RED), ("B", GREEN)]);
    spec.compression = OmeCompression::Zstd;
    let path = dir.path().join("wide.ome.tif");
    write_ome_tiff(&path, &spec, |c, x, _| {
        if c == 0 {
            f64::from(x % 2) * 1000.0
        } else {
            0.0
        }
    })
    .unwrap();
    let wsi = WsiFile::open(&path).unwrap();
    wsi.set_channel_display(display(&[RED, GREEN], 0.0, 1000.0, 1.0))
        .unwrap();
    let view = wsi.with_synthetic_levels();
    assert_eq!(view.level_count(), 2);
    // Alternating 0 and full red average to half red.
    let synthetic = view.read_region(0, 0, 1, 16, 4).unwrap();
    assert!(
        synthetic
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[0].abs_diff(128) <= 1 && p[1] == 0 && p[3] == 255),
        "{:?}",
        &synthetic[..8]
    );
    let thumbnail = wsi.thumbnail(64).unwrap();
    assert_eq!((thumbnail.width, thumbnail.height), (64, 1));
    assert!(thumbnail.pixels[0].abs_diff(128) <= 2);

    // Tiles with borders, through the tile manager.
    let manager = TileManager::new(view.clone(), 1);
    let coord = crate::TileCoord::new(1, 0, 1, 0, manager.tile_size_for_level(0));
    let tile = manager.load_tile_sync(coord).unwrap();
    assert_eq!(tile.border, 1);
    // Data pixel (0, 0) is the neighbouring tile's last column (odd: red).
    let x = coord.x * u64::from(coord.tile_size) - 1;
    assert_eq!(x % 2, 1);
    assert_eq!(&tile.data[..4], [255, 0, 0, 255]);
}

#[test]
fn label_and_macro_images_are_associated_images() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.extra_images = vec![
        OmeExtraImage {
            name: "label".into(),
            width: 30,
            height: 20,
            color: [10, 20, 30],
        },
        OmeExtraImage {
            name: "macro image".into(),
            width: 60,
            height: 20,
            color: [40, 50, 60],
        },
    ];
    let wsi = WsiFile::open(write(dir.path(), "with-label.ome.tif", &spec)).unwrap();
    let names: Vec<&str> = wsi
        .associated_images()
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    assert_eq!(names, ["label", "macro"]);
    let label = wsi.read_associated_image("label").unwrap();
    assert_eq!((label.width, label.height), (30, 20));
    assert!(
        label
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [10, 20, 30, 255])
    );
    assert_eq!(wsi.associated_image_icc_profile("label").unwrap(), None);
    // The main image is still the largest, with its own levels.
    assert_eq!(wsi.properties().width, 300);
    assert_eq!(wsi.level_count(), 3);
}

#[test]
fn missing_tiles_are_empty_and_corrupt_tiles_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.compression = OmeCompression::Deflate;
    // Channel 0's first tile is missing; channel 1's tile (1, 0) is
    // garbage.
    spec.missing_tiles = vec![(0, 0, 0, 0)];
    spec.corrupt_tiles = vec![(0, 1, 1, 0)];
    let wsi = WsiFile::open(write(dir.path(), "damaged.ome.tif", &spec)).unwrap();
    let shown = display(&[BLUE, GREEN, RED], 0.0, 1000.0, 1.0);
    wsi.set_channel_display(shown.clone()).unwrap();

    // The missing tile contributes nothing.
    let pixel = wsi.read_region(10, 10, 0, 1, 1).unwrap();
    let mut values = channel_values(&spec, 0, 10, 10, 3);
    values[0] = 0.0;
    assert_eq!(pixel, composite_of(&shown, &values));

    let error = wsi.read_region(70, 10, 0, 4, 4).unwrap_err();
    assert!(
        matches!(error, Error::CorruptRegion { level: 0, .. }),
        "{error:?}"
    );
    assert!(!error.is_transient());
    assert_eq!(wsi.health().bad_regions, 1);
    assert!(wsi.health().last_error.is_some());
    // Fails fast, the rest of the slide reads, and hiding the damaged
    // channel makes the region readable.
    assert!(wsi.read_region(70, 10, 0, 4, 4).is_err());
    assert!(wsi.read_region(200, 150, 0, 8, 8).is_ok());
    let mut without = shown;
    without.channels[1].visible = false;
    wsi.set_channel_display(without).unwrap();
    assert!(wsi.read_region(70, 10, 0, 4, 4).is_ok());
    wsi.clear_bad_regions();
    assert_eq!(wsi.health().bad_regions, 0);
}

/// The OME-TIFF reader's error for `path` (without the OpenSlide
/// fallback of `WsiFile::open`).
fn open_error(path: &Path) -> Error {
    match open(path) {
        Ok(_) => panic!("{} opened", path.display()),
        Err(error) => error,
    }
}

#[test]
fn unsupported_and_malformed_files_are_errors_not_panics() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.compression = OmeCompression::Jpeg2000;
    let path = write(dir.path(), "j2k.ome.tif", &spec);
    let error = open_error(&path);
    assert!(
        matches!(&error, Error::Unsupported(message) if message.contains("JPEG 2000")),
        "{error:?}"
    );
    // Through WsiFile too, OpenSlide cannot read it either and the reason
    // is kept.
    assert!(matches!(WsiFile::open(&path), Err(Error::Unsupported(_))));

    let mut spec = three_channels();
    spec.xml = Some("<?xml version=\"1.0\"?><OME><Image ID=\"x\"><Pixels".into());
    let error = open_error(&write(dir.path(), "bad-xml.ome.tif", &spec));
    assert!(matches!(error, Error::InvalidFormat(_)), "{error:?}");

    // Channels the XML promises but the file does not hold.
    let mut spec = three_channels();
    spec.xml = Some(
        "<OME><Image ID=\"i\"><Pixels DimensionOrder=\"XYCZT\" Type=\"uint16\" SizeX=\"300\" \
         SizeY=\"200\" SizeC=\"5\"><TiffData IFD=\"0\" PlaneCount=\"3\"/></Pixels></Image></OME>"
            .into(),
    );
    let error = open_error(&write(dir.path(), "short.ome.tif", &spec));
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");

    // Absurd channel counts are refused before anything is allocated.
    let mut spec = three_channels();
    spec.xml = Some(
        "<OME><Image ID=\"i\"><Pixels Type=\"uint16\" SizeX=\"300\" SizeY=\"200\" \
         SizeC=\"4000000000\"/></Image></OME>"
            .into(),
    );
    let error = open_error(&write(dir.path(), "huge.ome.tif", &spec));
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");

    // Truncated copies: the directories are at the end of these files.
    let path = write(dir.path(), "whole.ome.tif", &three_channels());
    let bytes = std::fs::read(&path).unwrap();
    for keep in [0, 3, 8, 100, bytes.len() / 2, bytes.len() - 10] {
        let truncated = dir.path().join(format!("truncated-{keep}.ome.tif"));
        std::fs::write(&truncated, &bytes[..keep]).unwrap();
        assert!(open(&truncated).is_err(), "kept {keep} bytes");
        // OpenSlide may still read what is left; either way, no panic.
        let _ = WsiFile::open(&truncated);
    }
}

#[test]
fn damaged_bytes_never_panic() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.compression = OmeCompression::Lzw;
    spec.predictor = true;
    let path = write(dir.path(), "source.ome.tif", &spec);
    let bytes = std::fs::read(&path).unwrap();
    // A fixed xorshift sequence: reproducible, no extra dependency.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let damaged = dir.path().join("damaged.ome.tif");
    for round in 0..120 {
        let mut copy = bytes.clone();
        for _ in 0..1 + round % 4 {
            let at = (next() % copy.len() as u64) as usize;
            copy[at] ^= 1 << (next() % 8);
        }
        std::fs::write(&damaged, &copy).unwrap();
        // The reader alone: OpenSlide's handling of damaged files is not
        // under test here.
        let Ok(opened) = open(&damaged) else {
            continue;
        };
        for level in 0..opened.properties.levels.len() as u32 {
            for (x, y, width, height) in [(0, 0, 64, 64), (-10, 150, 300, 80)] {
                let mut out = vec![0u8; width as usize * height as usize * 4];
                let region = Region {
                    x,
                    y,
                    level,
                    width,
                    height,
                };
                let _ = opened
                    .source
                    .read_region_into(region, AlphaMode::Composited, &mut out);
            }
        }
        for image in &opened.associated_images {
            let _ = opened
                .source
                .read_associated_image(&image.name, AlphaMode::Composited);
        }
    }
}

#[test]
fn concurrent_reads_agree() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.compression = OmeCompression::Zstd;
    let wsi = WsiFile::open(write(dir.path(), "shared.ome.tif", &spec)).unwrap();
    let expected = wsi.read_region(0, 0, 0, 300, 200).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let wsi = wsi.clone();
            let expected = &expected;
            scope.spawn(move || {
                for _ in 0..10 {
                    assert_eq!(&wsi.read_region(0, 0, 0, 300, 200).unwrap(), expected);
                }
            });
        }
    });
}

#[test]
fn the_writer_produces_standard_tiff_that_the_tiff_crate_decodes() {
    // Cross-checks the test writer itself against an independent decoder,
    // so the reader is not only tested against its own assumptions.
    let dir = tempfile::tempdir().unwrap();
    // (Not LZW: the `tiff` crate's LZW reader asserts on streams that
    // weezl, which both it and this reader decode with, accepts.)
    for compression in [OmeCompression::None, OmeCompression::Deflate] {
        let mut spec = three_channels();
        spec.compression = compression;
        let path = write(dir.path(), &format!("{compression:?}.ome.tif"), &spec);
        let mut decoder =
            ::tiff::decoder::Decoder::new(std::fs::File::open(&path).unwrap()).unwrap();
        for channel in 0..3 {
            if channel > 0 {
                decoder.next_image().unwrap();
            }
            assert_eq!(decoder.dimensions().unwrap(), (300, 200));
            let ::tiff::decoder::DecodingResult::U16(values) = decoder.read_image().unwrap() else {
                panic!("expected 16-bit samples");
            };
            for (x, y) in [(0, 0), (77, 13), (299, 199)] {
                assert_eq!(
                    f64::from(values[(y * 300 + x) as usize]),
                    ramps(channel, x, y),
                    "{compression:?} channel {channel} ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn reads_files_written_by_an_independent_encoder() {
    // Strips, compressions and the predictor as the `tiff` crate writes
    // them, with OME-XML added: the decoders are not only tested against
    // eov's own writer.
    use ::tiff::encoder::{Compression, DeflateLevel, Predictor, TiffEncoder, colortype};
    use ::tiff::tags::Tag;
    let dir = tempfile::tempdir().unwrap();
    let (width, height) = (120u32, 90u32);
    let xml = format!(
        "<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">\
         <Image ID=\"Image:0\"><Pixels DimensionOrder=\"XYCZT\" Type=\"uint16\" SizeX=\"{width}\" \
         SizeY=\"{height}\" SizeZ=\"1\" SizeC=\"3\" SizeT=\"1\" PhysicalSizeX=\"650\" \
         PhysicalSizeXUnit=\"nm\" PhysicalSizeY=\"0.65\"><Channel ID=\"Channel:0:0\" Name=\"a\"/>\
         <Channel ID=\"Channel:0:1\" Name=\"b\"/><Channel ID=\"Channel:0:2\" Name=\"c\"/>\
         <TiffData/></Pixels></Image></OME>"
    );
    let shown = display(&[BLUE, GREEN, RED], 0.0, 1000.0, 1.0);
    for (name, compression, predictor) in [
        ("none", Compression::Uncompressed, Predictor::None),
        ("lzw", Compression::Lzw, Predictor::Horizontal),
        (
            "deflate",
            Compression::Deflate(DeflateLevel::Balanced),
            Predictor::Horizontal,
        ),
        ("packbits", Compression::Packbits, Predictor::None),
    ] {
        let path = dir.path().join(format!("{name}.ome.tif"));
        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = TiffEncoder::new(file)
            .unwrap()
            .with_compression(compression)
            .with_predictor(predictor);
        for channel in 0..3 {
            let data: Vec<u16> = (0..height)
                .flat_map(|y| (0..width).map(move |x| ramps(channel, x, y) as u16))
                .collect();
            let mut image = encoder
                .new_image::<colortype::Gray16>(width, height)
                .unwrap();
            if channel == 0 {
                image
                    .encoder()
                    .write_tag(Tag::ImageDescription, xml.as_str())
                    .unwrap();
            }
            image.write_data(&data).unwrap();
        }
        let wsi = WsiFile::open(&path).unwrap();
        assert_eq!(wsi.properties().vendor.as_deref(), Some(VENDOR), "{name}");
        assert!((wsi.properties().mpp_x.unwrap() - 0.65).abs() < 1e-12);
        wsi.set_channel_display(shown.clone()).unwrap();
        let pixels = wsi.read_region(0, 0, 0, width, height).unwrap();
        for (x, y) in [(0, 0), (1, 0), (64, 33), (119, 89)] {
            let values: Vec<f64> = (0..3).map(|c| ramps(c, x, y)).collect();
            let at = ((y * width + x) * 4) as usize;
            assert_eq!(
                &pixels[at..at + 4],
                composite_of(&shown, &values),
                "{name} ({x}, {y})"
            );
        }
    }
}

#[test]
fn level_coordinates_round_trip_truncated_and_rounded_positions() {
    for downsample in [1.0, 1.6, 2.0, 2.0004, 3.99, 4.0, 8.002, 16.0, 31.7] {
        for level_x in [0i64, 1, 2, 3, 7, 255, 256, 1023, 99_999] {
            let exact = level_x as f64 * downsample;
            assert_eq!(
                level_coordinate(exact as i64, downsample),
                level_x,
                "truncated {level_x} at {downsample}"
            );
            assert_eq!(
                level_coordinate(exact.round() as i64, downsample),
                level_x,
                "rounded {level_x} at {downsample}"
            );
        }
    }
    assert_eq!(level_coordinate(-10, 1.0), -10);
    assert!(level_coordinate(-10, 2.0) < 0);
}

#[test]
fn ome_extensions_are_recognised() {
    assert!(has_ome_extension(Path::new("/a/b/slide.ome.tif")));
    assert!(has_ome_extension(Path::new("SLIDE.OME.TIFF")));
    assert!(has_ome_extension(Path::new("x.ome.btf")));
    assert!(!has_ome_extension(Path::new("x.tif")));
    assert!(!has_ome_extension(Path::new("ome.tif")));
}

#[test]
fn channel_names_fall_back_and_white_colours_are_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = three_channels();
    spec.channels = vec![
        OmeChannelSpec::new("Hoechst", Some([255, 255, 255])),
        OmeChannelSpec::new("CD3", Some([255, 255, 255])),
        OmeChannelSpec {
            emission_nm: Some(670.0),
            ..OmeChannelSpec::new("CD8", Some([255, 255, 255]))
        },
    ];
    let wsi = WsiFile::open(write(dir.path(), "white.ome.tif", &spec)).unwrap();
    let defaults = wsi.default_channel_display().unwrap();
    let colors: Vec<_> = defaults.channels.iter().map(|c| c.color).collect();
    assert_eq!(colors, [BLUE, GREEN, RED]);
    assert_eq!(wsi.channels().unwrap().channels[2].emission_nm, Some(670.0));
}
