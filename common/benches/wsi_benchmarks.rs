//! Benchmarks for WSI file operations
//!
//! Run with: cargo bench -p common

use common::{TileCache, TileCoord, TileManager, Viewport, WsiFile};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
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

    let svs_path = get_svs_fixture();
    if svs_path.exists() {
        group.bench_function("svs_322mb", |b| {
            b.iter(|| WsiFile::open(black_box(&svs_path)).unwrap())
        });
    }

    let tif_path = get_tif_fixture();
    if tif_path.exists() {
        group.bench_function("tif_1.8gb", |b| {
            b.iter(|| WsiFile::open(black_box(&tif_path)).unwrap())
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
    let manager =
        TileManager::new(WsiFile::open(&svs_path).expect("Failed to open second SVS handle"), 0);

    let mut group = c.benchmark_group("tile_read");
    group.throughput(Throughput::Elements(1));

    // Benchmark tiles at different levels
    for level in 0..wsi.level_count().min(4) {
        let level_info = wsi.level(level).unwrap();
        let max_x = level_info.tiles_x(256).saturating_sub(1);
        let max_y = level_info.tiles_y(256).saturating_sub(1);

        // Origin tile
        group.bench_with_input(
            BenchmarkId::new("origin", format!("level_{}", level)),
            &(level, 0u64, 0u64),
            |b, &(l, x, y)| {
                b.iter(|| {
                    manager
                        .load_tile_sync(TileCoord::new(0, l, x, y, manager.tile_size_for_level(l)))
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
                        .load_tile_sync(TileCoord::new(0, l, x, y, manager.tile_size_for_level(l)))
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
                        .load_tile_sync(TileCoord::new(0, l, x, y, manager.tile_size_for_level(l)))
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
    let manager =
        TileManager::new(WsiFile::open(&svs_path).expect("Failed to open second SVS handle"), 0);

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
        let tile = common::TileData::placeholder(TileCoord::new(0, 0, i, 0, 256), 256);
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
            let tile = common::TileData::placeholder(TileCoord::new(0, 0, i, 0, 256), 256);
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

criterion_group!(
    benches,
    bench_open_file,
    bench_read_tiles,
    bench_batch_tiles,
    bench_cache,
    bench_viewport,
    bench_visible_tiles,
);
criterion_main!(benches);
