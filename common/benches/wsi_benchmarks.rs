//! Benchmarks for WSI file operations
//!
//! Run with: cargo bench -p eov-common
//!
//! Slide benchmarks need the (git-ignored) `fixtures/` directory and are
//! skipped without it; blitter, cache and viewport benchmarks always run.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use eov_common::blitter::{self, BlitRect, CoarseSrc, TileSrc};
use eov_common::{TileCache, TileCoord, TileManager, Viewport, WsiFile};
use std::path::PathBuf;

fn get_svs_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/C3L-00088-22.svs")
}

fn get_tif_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/patient_198_node_0.tif")
}

/// Benchmark opening WSI files
fn bench_open_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_open");

    // `WsiFile::open` reuses open slides, so measure exclusive opens.
    let svs_path = get_svs_fixture();
    if svs_path.exists() {
        group.bench_function("svs_322mb", |b| {
            b.iter(|| WsiFile::open_exclusive(black_box(&svs_path)).unwrap())
        });
    }

    let tif_path = get_tif_fixture();
    if tif_path.exists() {
        group.bench_function("tif_1.8gb", |b| {
            b.iter(|| WsiFile::open_exclusive(black_box(&tif_path)).unwrap())
        });
    }

    group.finish();
}

/// Benchmark tile reading at various zoom levels
fn bench_read_tiles(c: &mut Criterion) {
    let svs_path = get_svs_fixture();
    if !svs_path.exists() {
        eprintln!("Skipping tile benchmarks: SVS fixture not found");
        return;
    }

    let wsi = WsiFile::open(&svs_path).expect("Failed to open SVS file");
    // Clones share the handle; that is how workers should use a slide.
    let manager = TileManager::new(wsi.clone(), 0);

    let mut group = c.benchmark_group("tile_read");
    group.throughput(Throughput::Elements(1));

    // Benchmark tiles at different levels
    for level in 0..wsi.level_count().min(4) {
        let level_info = wsi.level(level).unwrap();
        let tile_size = manager.tile_size_for_level(level);
        let max_x = level_info.tiles_x(tile_size).saturating_sub(1);
        let max_y = level_info.tiles_y(tile_size).saturating_sub(1);

        // Origin tile
        group.bench_with_input(
            BenchmarkId::new("origin", format!("level_{}", level)),
            &(level, 0u64, 0u64),
            |b, &(l, x, y)| {
                b.iter(|| {
                    manager
                        .load_tile_sync(TileCoord::new(0, l, x, y, tile_size))
                        .unwrap()
                })
            },
        );

        // Center tile
        let center_x = max_x / 2;
        let center_y = max_y / 2;
        group.bench_with_input(
            BenchmarkId::new("center", format!("level_{}", level)),
            &(level, center_x, center_y),
            |b, &(l, x, y)| {
                b.iter(|| {
                    manager
                        .load_tile_sync(TileCoord::new(0, l, x, y, tile_size))
                        .unwrap()
                })
            },
        );

        // Edge tile
        group.bench_with_input(
            BenchmarkId::new("edge", format!("level_{}", level)),
            &(level, max_x, max_y),
            |b, &(l, x, y)| {
                b.iter(|| {
                    manager
                        .load_tile_sync(TileCoord::new(0, l, x, y, tile_size))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark batch tile loading
fn bench_batch_tiles(c: &mut Criterion) {
    let svs_path = get_svs_fixture();
    if !svs_path.exists() {
        return;
    }

    let wsi = WsiFile::open(&svs_path).expect("Failed to open SVS file");
    // Clones share the handle; that is how workers should use a slide.
    let manager = TileManager::new(wsi.clone(), 0);

    let mut group = c.benchmark_group("batch_tiles");

    // Simulate viewport: 1920x1080 at various zoom levels
    for level in 0..wsi.level_count().min(4) {
        let tiles = manager.visible_tiles(level, 0.0, 0.0, 1920.0, 1080.0);
        let tile_count = tiles.len().min(20); // Limit to keep benchmark reasonable
        let tiles: Vec<_> = tiles.into_iter().take(tile_count).collect();

        group.throughput(Throughput::Elements(tile_count as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential", format!("level_{}_{}tiles", level, tile_count)),
            &tiles,
            |b, tiles| {
                b.iter(|| {
                    for coord in tiles {
                        let _ = manager.load_tile_sync(*coord);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark tile cache operations
fn bench_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache");

    let cache = TileCache::new();

    // Pre-populate cache
    for i in 0..100u64 {
        let tile = eov_common::TileData::placeholder(TileCoord::new(0, 0, i, 0, 256), 256);
        cache.insert(tile);
    }

    // Benchmark cache hit
    group.bench_function("cache_hit", |b| {
        let coord = TileCoord::new(0, 0, 50, 0, 256);
        b.iter(|| cache.get(black_box(&coord)))
    });

    // Benchmark cache miss
    group.bench_function("cache_miss", |b| {
        let coord = TileCoord::new(0, 0, 9999, 9999, 256);
        b.iter(|| cache.get(black_box(&coord)))
    });

    // Benchmark cache insert
    group.bench_function("cache_insert", |b| {
        let mut i = 1000u64;
        b.iter(|| {
            let tile = eov_common::TileData::placeholder(TileCoord::new(0, 0, i, 0, 256), 256);
            cache.insert(tile);
            i += 1;
        })
    });

    group.finish();
}

/// Benchmark viewport calculations
fn bench_viewport(c: &mut Criterion) {
    let mut group = c.benchmark_group("viewport");

    let mut viewport = Viewport::new(1920.0, 1080.0, 100000.0, 100000.0);

    group.bench_function("screen_to_image", |b| {
        b.iter(|| viewport.screen_to_image(black_box(960.0), black_box(540.0)))
    });

    group.bench_function("image_to_screen", |b| {
        b.iter(|| viewport.image_to_screen(black_box(50000.0), black_box(50000.0)))
    });

    group.bench_function("zoom_at", |b| {
        b.iter(|| {
            viewport.zoom_at(black_box(1.1), black_box(960.0), black_box(540.0));
            viewport.zoom_at(black_box(0.909), black_box(960.0), black_box(540.0));
        })
    });

    group.bench_function("pan", |b| {
        b.iter(|| {
            viewport.pan(black_box(10.0), black_box(10.0));
            viewport.pan(black_box(-10.0), black_box(-10.0));
        })
    });

    group.bench_function("minimap_rect", |b| b.iter(|| viewport.minimap_rect()));

    group.finish();
}

/// Benchmark visible tile calculation
fn bench_visible_tiles(c: &mut Criterion) {
    let svs_path = get_svs_fixture();
    if !svs_path.exists() {
        return;
    }

    let wsi = WsiFile::open(&svs_path).expect("Failed to open SVS file");
    let manager = TileManager::new(wsi, 0);

    let mut group = c.benchmark_group("visible_tiles");

    // Various viewport sizes and positions
    let scenarios = [
        ("small_viewport", 800.0, 600.0, 0.0, 0.0, 1.0),
        ("hd_viewport", 1920.0, 1080.0, 0.0, 0.0, 1.0),
        ("4k_viewport", 3840.0, 2160.0, 0.0, 0.0, 1.0),
        ("zoomed_out", 1920.0, 1080.0, 0.0, 0.0, 0.1),
        ("zoomed_in", 1920.0, 1080.0, 50000.0, 50000.0, 10.0),
    ];

    for (name, w, h, x, y, zoom) in scenarios {
        group.bench_function(name, |b| {
            b.iter(|| {
                manager.visible_tiles(
                    black_box(0),
                    black_box(x),
                    black_box(y),
                    black_box(x + w / zoom),
                    black_box(y + h / zoom),
                )
            })
        });
    }

    group.finish();
}

fn synthetic_tile(size: u32, border: u32) -> Vec<u8> {
    let data = size + 2 * border;
    (0..data * data)
        .flat_map(|i| {
            let v = (i * 7 % 251) as u8;
            [v, v / 2, 255 - v, 255]
        })
        .collect()
}

/// Benchmark the CPU blitters on a 256-px tile (no fixtures needed).
fn bench_blitters(c: &mut Criterion) {
    let mut group = c.benchmark_group("blit_256");
    let tile = synthetic_tile(256, 3);
    let coarse = synthetic_tile(256, 1);
    let mut dest = vec![0u8; 1024 * 1024 * 4];
    for (name, scale) in [("magnify_2.5x", 2.5f64), ("minify_0.6x", 0.6)] {
        let size = 256.0 * scale;
        let rect = BlitRect {
            x: 3,
            y: 3,
            width: size.floor() as i32,
            height: size.floor() as i32,
            exact_x: 3.25,
            exact_y: 3.25,
            exact_width: size,
            exact_height: size,
        };
        let src = || TileSrc {
            data: &tile,
            width: 256,
            height: 256,
            border: 3,
        };
        group.bench_function(BenchmarkId::new("bilinear", name), |b| {
            b.iter(|| blitter::blit_tile(&mut dest, 1024, 1024, src(), black_box(rect)))
        });
        group.bench_function(BenchmarkId::new("trilinear", name), |b| {
            let coarse_src = CoarseSrc {
                data: &coarse,
                width: 256,
                height: 256,
                border: 1,
                uv_min: [0.25, 0.25],
                uv_max: [0.5, 0.5],
                blend: 0.4,
            };
            b.iter(|| {
                blitter::blit_tile_trilinear(
                    &mut dest,
                    1024,
                    1024,
                    src(),
                    &coarse_src,
                    black_box(rect),
                )
            })
        });
        group.bench_function(BenchmarkId::new("lanczos3", name), |b| {
            b.iter(|| blitter::blit_tile_lanczos3(&mut dest, 1024, 1024, src(), black_box(rect)))
        });
    }
    group.finish();
}

/// Benchmark render-loop cache lookups, which now mark tiles as used.
fn bench_cache_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_touch");
    let cache = TileCache::new();
    let coords: Vec<TileCoord> = (0..500u64)
        .map(|i| TileCoord::new(0, 0, i % 25, i / 25, 256))
        .collect();
    for coord in &coords {
        cache.insert(eov_common::TileData::placeholder(*coord, 64));
    }
    group.throughput(Throughput::Elements(coords.len() as u64));
    group.bench_function("peek_500_per_frame", |b| {
        b.iter(|| {
            cache.begin_frame();
            for coord in &coords {
                black_box(cache.peek(coord));
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_open_file,
    bench_read_tiles,
    bench_batch_tiles,
    bench_cache,
    bench_cache_touch,
    bench_blitters,
    bench_viewport,
    bench_visible_tiles,
);
criterion_main!(benches);
