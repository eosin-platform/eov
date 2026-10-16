//! Export integration tests on synthetic slides.

use std::path::Path;

use super::*;
use crate::overlay::{OverlayColor, OverlayItem, ScaleBarStyle};
use crate::postprocess::ToneAdjustments;
use crate::processing::ProcessingPipeline;
use crate::test_slides::{PyramidSpec, write_pyramid_tiff, write_uniform_tiff};
use crate::{TileManager, WsiFile};

fn pattern(x: u32, y: u32) -> [u8; 3] {
    [
        (x * 3 % 256) as u8,
        (y * 5 % 256) as u8,
        ((x / 16 + y / 16) % 2 * 200 + 20) as u8,
    ]
}

fn slide(dir: &Path) -> TileManager {
    let path = dir.join("pattern.tif");
    write_pyramid_tiff(
        &path,
        &PyramidSpec::new(600, 400).levels(&[1, 2, 4]).mpp(0.5),
        pattern,
    )
    .unwrap();
    TileManager::new(WsiFile::open(&path).unwrap(), 7)
}

fn source(manager: &TileManager) -> ExportSource<'_> {
    ExportSource {
        tile_manager: manager,
        tile_cache: None,
    }
}

fn render(manager: &TileManager, request: &ExportRequest) -> (RgbaImageData, ExportReport) {
    render_export_image(&source(manager), request, &JobControl::new()).unwrap()
}

#[test]
fn full_resolution_region_export_is_pixel_exact_in_any_band_size() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let mut request = ExportRequest::region(37.0, 51.0, 300.0, 200.0, 1.0, (600.0, 400.0));
    request.filtering = FilteringMode::Bilinear;
    let (whole, report) = render(&manager, &request);
    assert_eq!((whole.width, whole.height), (300, 200));
    assert!(report.is_complete());
    assert_eq!(report.level, 0);
    for (x, y) in [(0u32, 0u32), (17, 3), (299, 199), (150, 100)] {
        let i = ((y * 300 + x) * 4) as usize;
        assert_eq!(
            &whole.pixels[i..i + 3],
            &pattern(x + 37, y + 51),
            "({x},{y})"
        );
        assert_eq!(whole.pixels[i + 3], 255);
    }
    for band in [1, 7, 64] {
        request.band_rows = Some(band);
        assert_eq!(render(&manager, &request).0, whole, "band {band}");
    }
}

/// A turned view exports as shown: the unturned export, turned, in any band
/// size (bands of a quarter turn are column strips of the slide).
#[test]
fn turned_views_export_turned() {
    use crate::viewport::Orientation;
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let mut request = ExportRequest::region(37.0, 51.0, 300.0, 200.0, 1.0, (600.0, 400.0));
    request.filtering = FilteringMode::Bilinear;
    let (plain, _) = render(&manager, &request);
    for orientation in Orientation::all() {
        let mut turned = request.clone();
        turned.viewport.orientation = orientation;
        let (w, h) = orientation.swap(300.0, 200.0);
        turned.viewport.width = w;
        turned.viewport.height = h;
        let mut expected = vec![0; plain.pixels.len()];
        crate::blitter::orient_frame(&plain.pixels, 300, 200, orientation, &mut expected);
        for band in [None, Some(7), Some(64)] {
            turned.band_rows = band;
            let (image, report) = render(&manager, &turned);
            assert_eq!((image.width, image.height), (w as usize, h as usize));
            assert!(report.is_complete());
            assert!(image.pixels == expected, "{orientation:?}, band {band:?}");
        }
    }
}

/// The pattern bilinearly sampled at level-0 point `(x, y)` (pixel
/// centres at half-integers, edges clamped): what a level-0 render of a
/// view shows at a pixel whose centre maps there.
fn sampled(x: f64, y: f64) -> [f64; 3] {
    let (x, y) = (x - 0.5, y - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |dx: f64, dy: f64| {
        let px = (x0 + dx).clamp(0.0, 599.0) as u32;
        let py = (y0 + dy).clamp(0.0, 399.0) as u32;
        pattern(px, py).map(f64::from)
    };
    let (a, b, c, d) = (at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0), at(1.0, 1.0));
    std::array::from_fn(|i| {
        a[i] * (1.0 - fx) * (1.0 - fy)
            + b[i] * fx * (1.0 - fy)
            + c[i] * (1.0 - fx) * fy
            + d[i] * fx * fy
    })
}

/// A view turned by another angle exports as it shows: every output
/// pixel is the slide at the level-0 point its centre maps to (sampled
/// once), the background outside the slide, in any band size.
#[test]
fn freely_turned_views_export_the_turned_area() {
    use crate::viewport::Orientation;
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    for (degrees, flipped, center) in [
        (30.0, false, (300.0, 200.0)),
        (45.0, true, (310.0, 190.0)),
        (200.5, false, (520.0, 330.0)),
    ] {
        let mut viewport = Viewport::new(240.0, 160.0, 600.0, 400.0);
        viewport.zoom = 1.0;
        viewport.center.x = center.0;
        viewport.center.y = center.1;
        viewport.orientation = Orientation::from_degrees(degrees, flipped).unwrap();
        let mut request = ExportRequest::new(viewport, 96.0);
        request.filtering = FilteringMode::Bilinear;
        request.background = [1, 2, 3];
        let (image, report) = render(&manager, &request);
        assert_eq!((image.width, image.height), (240, 160));
        assert!(report.is_complete());
        assert_eq!(report.level, 0);
        let evp = request.export_viewport();
        let (mut slide_pixels, mut background) = (0, 0);
        for y in 0..160u32 {
            for x in 0..240u32 {
                let p = evp.screen_to_image(f64::from(x) + 0.5, f64::from(y) + 0.5);
                let i = ((y * 240 + x) * 4) as usize;
                let got = &image.pixels[i..i + 4];
                assert_eq!(got[3], 255);
                // Pixel centres within a hair of the slide's edge may go
                // either way.
                let edge = [p.x, 600.0 - p.x, p.y, 400.0 - p.y]
                    .into_iter()
                    .map(f64::abs)
                    .fold(f64::INFINITY, f64::min);
                if edge < 1e-3 {
                    continue;
                }
                if !(0.0..600.0).contains(&p.x) || !(0.0..400.0).contains(&p.y) {
                    assert_eq!(&got[..3], &[1, 2, 3], "{degrees}°: {x},{y} is outside");
                    background += 1;
                    continue;
                }
                slide_pixels += 1;
                let expected = sampled(p.x, p.y);
                // Positions are rounded to 1/256 of a pixel, values to
                // whole levels.
                for c in 0..3 {
                    assert!(
                        (f64::from(got[c]) - expected[c]).abs() < 1.5,
                        "{degrees}°: {x},{y} channel {c}: {} vs {:.2}",
                        got[c],
                        expected[c]
                    );
                }
            }
        }
        assert!(slide_pixels > 10_000, "{degrees}°");
        if center.0 > 400.0 {
            assert!(background > 1000, "{degrees}°: the view crosses the edge");
        }
        for band in [Some(1), Some(7), Some(64)] {
            request.band_rows = band;
            assert_eq!(
                render(&manager, &request).0,
                image,
                "{degrees}°, band {band:?}"
            );
        }
    }
}

/// The scale bar of a turned export measures along the output's x axis.
#[test]
fn turned_exports_size_their_pixels_along_the_output_axes() {
    use crate::viewport::Orientation;
    let mut request = ExportRequest::region(0.0, 0.0, 200.0, 100.0, 2.0, (600.0, 400.0));
    assert_eq!(
        request.output_pixel_size_um(Some(0.25), Some(0.5)),
        (Some(0.5), Some(1.0))
    );
    request.viewport.orientation = Orientation::new(1, false);
    assert_eq!(
        request.output_pixel_size_um(Some(0.25), Some(0.5)),
        (Some(1.0), Some(0.5))
    );
    request.viewport.orientation = Orientation::from_degrees(30.0, false).unwrap();
    let (x, y) = request.output_pixel_size_um(Some(0.25), Some(0.5));
    let (sin, cos) = 30f64.to_radians().sin_cos();
    let expected_x = (cos * 2.0 * 0.25).hypot(sin * 2.0 * 0.5);
    let expected_y = (sin * 2.0 * 0.25).hypot(cos * 2.0 * 0.5);
    assert!((x.unwrap() - expected_x).abs() < 1e-12);
    assert!((y.unwrap() - expected_y).abs() < 1e-12);
    // Isotropic pixels keep their size at any angle.
    let (x, _) = request.output_pixel_size_um(Some(0.5), Some(0.5));
    assert!((x.unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(request.output_pixel_size_um(None, Some(0.5)), (None, None));
}

#[test]
fn processing_sharpening_and_overlays_are_band_independent() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let viewport = Viewport::new(300.0, 200.0, 600.0, 400.0);
    let mut request = ExportRequest::new(viewport, 150.0);
    request.filtering = FilteringMode::Trilinear;
    request.processing =
        ProcessingPipeline::new(None, ToneAdjustments::new(1.3, 0.02, 1.1), 0.6).unwrap();
    request.overlays = vec![
        OverlayItem::Line {
            from: [10.0, 10.0],
            to: [400.0, 280.0],
            color: OverlayColor::new(255, 0, 0, 255),
            thickness: 3.0,
            stroke: crate::overlay::StrokeStyle::Solid,
            cap: crate::overlay::CapStyle::Round,
        },
        OverlayItem::ScaleBar {
            um_per_px: request
                .output_pixel_size_um(Some(0.5), Some(0.5))
                .0
                .unwrap(),
            style: ScaleBarStyle::default(),
        },
    ];
    request.band_rows = Some(10_000);
    let (whole, _) = render(&manager, &request);
    for band in [1, 13, 50] {
        request.band_rows = Some(band);
        assert_eq!(render(&manager, &request).0, whole, "band {band}");
    }
}

#[test]
fn exports_are_not_capped_at_4096_tiles() {
    // EXP-1: the viewer's visible-tile enumeration stops at 4096 tiles; the
    // export enumeration must cover every tile.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.tif");
    write_uniform_tiff(&path, 32_768, 32_768, 256, [200, 100, 50], None, &[1]).unwrap();
    let manager = TileManager::new(WsiFile::open(&path).unwrap(), 1);
    let tiles = render::tiles_in_rect_for_test(&manager, 0, 0.0, 0.0, 32_768.0, 32_768.0);
    assert_eq!(tiles, 128 * 128);
    assert!(manager.visible_tiles(0, 0.0, 0.0, 32_768.0, 32_768.0).len() <= 4096);
    // And a real (small) render of a strip far below the first 4096 tiles.
    let mut request = ExportRequest::region(0.0, 30_000.0, 4096.0, 64.0, 1.0, (32_768.0, 32_768.0));
    request.filtering = FilteringMode::Bilinear;
    let (image, report) = render(&manager, &request);
    assert!(report.is_complete());
    assert!(image.pixels.chunks(4).all(|p| p == [200, 100, 50, 255]));
}

#[test]
fn unreadable_tiles_are_reported_not_hidden() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.tif");
    write_uniform_tiff(&path, 1024, 1024, 256, [10, 20, 30], Some(5), &[1, 4]).unwrap();
    let manager = TileManager::new(WsiFile::open(&path).unwrap(), 1);
    let mut request = ExportRequest::region(0.0, 0.0, 1024.0, 1024.0, 1.0, (1024.0, 1024.0));
    request.filtering = FilteringMode::Bilinear;
    // The export still completes, and says what is missing (EXP-1).
    let (image, report) = render(&manager, &request);
    assert_eq!((image.width, image.height), (1024, 1024));
    assert!(!report.is_complete());
    assert!(report.failed_tiles.iter().all(|c| c.level == 0));
    assert!(
        report
            .shortfall_message()
            .unwrap()
            .contains("could not be read")
    );
}

#[test]
fn cancellation_stops_the_export_and_leaves_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let request = ExportRequest::new(Viewport::new(300.0, 200.0, 600.0, 400.0), 96.0);
    let control = JobControl::new();
    control.cancel();
    let out = dir.path().join("cancelled.png");
    let err = export_to_file(
        &source(&manager),
        &request,
        &out,
        ExportFormat::Png,
        ImageMetadata::default(),
        &control,
    )
    .unwrap_err();
    assert!(matches!(err, ExportError::Cancelled));
    assert!(!out.exists());
    assert!(!dir.path().join("cancelled.png.part").exists());
}

#[test]
fn progress_reaches_the_image_height() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let mut request = ExportRequest::new(Viewport::new(300.0, 200.0, 600.0, 400.0), 192.0);
    request.band_rows = Some(32);
    let control = JobControl::new();
    render_export_image(&source(&manager), &request, &control).unwrap();
    let progress = control.progress();
    assert_eq!(progress.total, 400);
    assert_eq!(progress.done, 400);
}

#[test]
fn preview_is_small_and_matches_the_request() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let request = ExportRequest::new(Viewport::new(300.0, 200.0, 600.0, 400.0), 600.0);
    assert_eq!(request.output_size(), (1875, 1250));
    let preview = render_export_preview(&source(&manager), &request, 512).unwrap();
    assert_eq!((preview.width, preview.height), (512, 341));
}

fn png_metadata(path: &Path) -> (Option<png::PixelDimensions>, Vec<(String, String)>) {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let reader = decoder.read_info().unwrap();
    let info = reader.info();
    let mut texts: Vec<(String, String)> = info
        .uncompressed_latin1_text
        .iter()
        .map(|t| (t.keyword.clone(), t.text.clone()))
        .collect();
    for t in &info.utf8_text {
        texts.push((t.keyword.clone(), t.get_text().unwrap()));
    }
    (info.pixel_dims, texts)
}

#[test]
fn files_carry_resolution_and_provenance_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let manager = slide(dir.path());
    let request = ExportRequest::new(Viewport::new(300.0, 200.0, 600.0, 400.0), 300.0);
    let (image, _) = render(&manager, &request);
    let metadata = ImageMetadata::for_export(
        &request,
        300.0,
        Some(0.5),
        Some(0.5),
        Some("Colours as scanned (no processing).".into()),
    );
    let (mx, _) = metadata.um_per_px;
    // Fit view: 600 level-0 px across 300 screen px; ×300/96 → 937.5 px.
    assert!((mx.unwrap() - 0.5 * 600.0 / 937.5).abs() < 1e-3, "{mx:?}");

    // PNG: pixels, pHYs and text chunks.
    let png_path = dir.path().join("figure.png");
    export_to_file(
        &source(&manager),
        &request,
        &png_path,
        ExportFormat::Png,
        metadata.clone(),
        &JobControl::new(),
    )
    .unwrap();
    let decoded = image::open(&png_path).unwrap().to_rgba8();
    assert_eq!(decoded.as_raw(), &image.pixels);
    let (dims, texts) = png_metadata(&png_path);
    let dims = dims.unwrap();
    assert_eq!(dims.unit, png::Unit::Meter);
    assert_eq!(dims.xppu, (300.0f64 / 0.0254).round() as u32);
    assert!(
        texts
            .iter()
            .any(|(k, v)| k == "Software" && v.starts_with("eov "))
    );
    assert!(
        texts
            .iter()
            .any(|(k, v)| k == "Description" && v.contains("µm"))
    );
    assert!(texts.iter().any(|(k, _)| k == "eov.mpp-x"));

    // JPEG: quality honoured (size grows with quality), density + COM.
    let low = dir.path().join("low.jpg");
    let high = dir.path().join("high.jpg");
    for (path, quality) in [(&low, 40u8), (&high, 95u8)] {
        export_to_file(
            &source(&manager),
            &request,
            path,
            ExportFormat::Jpeg { quality },
            metadata.clone(),
            &JobControl::new(),
        )
        .unwrap();
    }
    let (low_len, high_len) = (
        std::fs::metadata(&low).unwrap().len(),
        std::fs::metadata(&high).unwrap().len(),
    );
    assert!(low_len < high_len, "{low_len} vs {high_len}");
    let bytes = std::fs::read(&high).unwrap();
    assert_eq!(bytes[13], 1, "density in dots per inch");
    assert_eq!(u16::from_be_bytes([bytes[14], bytes[15]]), 300);
    assert!(String::from_utf8_lossy(&bytes).contains("Pixel size"));
    let jpeg = image::open(&high).unwrap().to_rgb8();
    assert_eq!(jpeg.dimensions(), (image.width as u32, image.height as u32));

    // TIFF: lossless pixels and resolution/description tags.
    let tif = dir.path().join("figure.tif");
    export_to_file(
        &source(&manager),
        &request,
        &tif,
        ExportFormat::Tiff {
            compression: TiffCompression::Deflate,
        },
        metadata,
        &JobControl::new(),
    )
    .unwrap();
    let mut decoder = ::tiff::decoder::Decoder::new(std::fs::File::open(&tif).unwrap()).unwrap();
    let description = decoder
        .get_tag_ascii_string(::tiff::tags::Tag::ImageDescription)
        .unwrap();
    assert!(description.contains("MPP = "));
    // TIFF ASCII is 7-bit: "µm" and "×" are transliterated.
    assert!(
        description.is_ascii() && description.contains(" um"),
        "{description}"
    );
    let ::tiff::decoder::DecodingResult::U8(rgb) = decoder.read_image().unwrap() else {
        panic!("8-bit RGB expected");
    };
    let expected: Vec<u8> = image
        .pixels
        .chunks(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    assert_eq!(rgb, expected);
}

#[test]
#[allow(deprecated)]
fn legacy_render_export_uses_one_slide_level_fit() {
    use crate::test_slides::he_tissue_pixel;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("he.tif");
    let (w, h) = (1024u32, 768u32);
    write_pyramid_tiff(&path, &PyramidSpec::new(w, h).levels(&[1, 4]), |x, y| {
        he_tissue_pixel(x, y, w, h, [238, 236, 240])
    })
    .unwrap();
    let manager = TileManager::new(WsiFile::open(&path).unwrap(), 3);
    let cache = crate::TileCache::new();
    let settings = ExportSettings {
        stain_normalization: crate::StainNormalization::Macenko,
        filtering_mode: FilteringMode::Bilinear,
        ..ExportSettings::default()
    };
    // Two different views of the same slide: a tissue-only one and a mostly
    // glass one. With a per-view fit their colours would differ (CLR-1).
    let mut tissue_view = Viewport::new(64.0, 64.0, w as f64, h as f64);
    tissue_view.center = glam::DVec2::new(512.0, 384.0);
    tissue_view.zoom = 1.0;
    let mut edge_view = tissue_view.clone();
    edge_view.width = 512.0;
    edge_view.height = 512.0;
    edge_view.center = glam::DVec2::new(512.0 + 240.0, 384.0);
    let a = render_export(&manager, &cache, &tissue_view, &settings).unwrap();
    let b = render_export(&manager, &cache, &edge_view, &settings).unwrap();
    // The level-0 pixel at the centre of the slide appears in both.
    let pick = |img: &RgbaImageData, vp: &Viewport| {
        let p = vp.image_to_screen(512.5, 384.5);
        let i = ((p.y as usize) * img.width + p.x as usize) * 4;
        [img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]]
    };
    assert_eq!(pick(&a, &tissue_view), pick(&b, &edge_view));
    // And normalisation actually changed the colours.
    let original = he_tissue_pixel(512, 384, w, h, [238, 236, 240]);
    assert_ne!(pick(&a, &tissue_view), original);
}

/// The pixel view: an export above 1:1 shows the slide's pixels as
/// squares; at or below 1:1 the filtering applies as usual.
#[test]
fn pixel_view_exports_show_pixels_above_full_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checker.tif");
    crate::test_support::write_tiled_tiff(&path, 64, 64, 64, |x, y| {
        if (x + y) % 2 == 0 {
            [250, 250, 250]
        } else {
            [10, 10, 10]
        }
    })
    .unwrap();
    let wsi = crate::WsiFile::open(&path).unwrap();
    let manager = crate::TileManager::new(wsi, 1);
    let source = ExportSource {
        tile_manager: &manager,
        tile_cache: None,
    };
    let render = |downsample: f64, pixel_view: bool| {
        let mut request = ExportRequest::region(8.0, 8.0, 16.0, 16.0, downsample, (64.0, 64.0));
        request.filtering = crate::FilteringMode::Bilinear;
        request.pixel_view = pixel_view;
        render_export_image(&source, &request, &crate::job::JobControl::new())
            .unwrap()
            .0
    };
    // 4× magnified: every output pixel is a slide pixel, in 4 × 4 squares.
    let squares = render(0.25, true);
    assert_eq!((squares.width, squares.height), (64, 64));
    for y in 0..64usize {
        for x in 0..64usize {
            let v = squares.pixels[(y * 64 + x) * 4];
            let expected = if ((x / 4) + (y / 4)) % 2 == 0 {
                250
            } else {
                10
            };
            assert_eq!(v, expected, "at {x},{y}");
        }
    }
    // Without the pixel view the same export blends neighbouring pixels.
    let smooth = render(0.25, false);
    assert!(
        smooth
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[0] > 20 && p[0] < 240)
    );
    // At 1:1 the pixel view changes nothing.
    assert_eq!(render(1.0, true).pixels, render(1.0, false).pixels);
}
