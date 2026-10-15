//! Slide-level and region stain estimation, and a per-slide estimate cache
//! (CLR-1, CLR-7).
//!
//! Stain normalisation must not depend on what happens to be on screen, so
//! the stain vectors are fitted **once per slide** on a low-resolution
//! pyramid level (with the tissue mask of [`PixelFilter`](super::PixelFilter)),
//! cached by slide identity and method, and shared by every pane, the export
//! preview and the final export. [`estimate_slide_stain`] is a plain blocking
//! function meant to run on a worker thread; [`StainEstimateCache`] stores the
//! results.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use super::estimate::{EstimateSource, StainEstimate, StainEstimateOptions, StainFitMethod};
use crate::WsiFile;

/// Largest region read with a single `read_region` call.
const READ_CHUNK: u64 = 1024;
/// Block size for sparse sampling of slides without a small pyramid level.
const SPARSE_BLOCK: u64 = 128;

/// A rectangle in level-0 pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level0Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Level0Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Intersects with the slide bounds; `None` if nothing remains.
    fn clamp_to(&self, width: u64, height: u64) -> Option<(u64, u64, u64, u64)> {
        let x0 = self.x.max(0.0).floor() as u64;
        let y0 = self.y.max(0.0).floor() as u64;
        let x1 = ((self.x + self.width).ceil().max(0.0) as u64).min(width);
        let y1 = ((self.y + self.height).ceil().max(0.0) as u64).min(height);
        (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
    }
}

/// Picks the finest level at which the level-0 extent `w0×h0` fits in the
/// pixel budget. Returns `(level, downsample, fits)`; `fits = false` means
/// even the coarsest level is too large.
fn choose_level(wsi: &WsiFile, w0: u64, h0: u64, budget: u64) -> (u32, f64, bool) {
    let count = wsi.level_count();
    for level in 0..count {
        let Some(info) = wsi.level(level) else {
            continue;
        };
        let ds = info.downsample.max(1.0);
        let w = (w0 as f64 / ds).ceil() as u64;
        let h = (h0 as f64 / ds).ceil() as u64;
        if w.saturating_mul(h) <= budget {
            return (level, ds, true);
        }
    }
    // A coarsest level somewhat over budget is still cheaper to read whole
    // than to sample sparsely, and gives a better fit.
    let level = count.saturating_sub(1);
    let ds = wsi
        .level(level)
        .map(|l| l.downsample.max(1.0))
        .unwrap_or(1.0);
    let w = (w0 as f64 / ds).ceil() as u64;
    let h = (h0 as f64 / ds).ceil() as u64;
    (
        level,
        ds,
        w.saturating_mul(h) <= budget.saturating_mul(COARSEST_LEVEL_SLACK),
    )
}

/// How far over the pixel budget the coarsest level may be and still be
/// read in full.
const COARSEST_LEVEL_SLACK: u64 = 4;

/// Reads a level-0 rectangle at `level` in chunks. Coordinates are level-0;
/// the returned buffers are RGBA at the level's resolution.
fn read_rect(
    wsi: &WsiFile,
    level: u32,
    ds: f64,
    (x0, y0, w0, h0): (u64, u64, u64, u64),
) -> crate::Result<Vec<Vec<u8>>> {
    let info = wsi.level(level).ok_or(crate::Error::InvalidLevel(
        level,
        wsi.level_count().saturating_sub(1),
    ))?;
    let lx0 = (x0 as f64 / ds).floor() as u64;
    let ly0 = (y0 as f64 / ds).floor() as u64;
    let lx1 = (((x0 + w0) as f64 / ds).ceil() as u64).min(info.width);
    let ly1 = (((y0 + h0) as f64 / ds).ceil() as u64).min(info.height);
    let mut out = Vec::new();
    let mut ly = ly0;
    while ly < ly1 {
        let h = (ly1 - ly).min(READ_CHUNK);
        let mut lx = lx0;
        while lx < lx1 {
            let w = (lx1 - lx).min(READ_CHUNK);
            let data = wsi.read_region(
                (lx as f64 * ds).round() as i64,
                (ly as f64 * ds).round() as i64,
                level,
                w as u32,
                h as u32,
            )?;
            out.push(data);
            lx += w;
        }
        ly += h;
    }
    Ok(out)
}

/// Reads a regular grid of small blocks spread over a level-0 rectangle.
fn read_sparse(
    wsi: &WsiFile,
    level: u32,
    ds: f64,
    (x0, y0, w0, h0): (u64, u64, u64, u64),
    budget: u64,
) -> crate::Result<Vec<Vec<u8>>> {
    let info = wsi.level(level).ok_or(crate::Error::InvalidLevel(
        level,
        wsi.level_count().saturating_sub(1),
    ))?;
    let per_side = ((budget as f64 / (SPARSE_BLOCK * SPARSE_BLOCK) as f64)
        .sqrt()
        .floor() as u64)
        .max(1);
    let lw = (w0 as f64 / ds).ceil() as u64;
    let lh = (h0 as f64 / ds).ceil() as u64;
    let lx0 = (x0 as f64 / ds).floor() as u64;
    let ly0 = (y0 as f64 / ds).floor() as u64;
    let mut out = Vec::new();
    for j in 0..per_side {
        for i in 0..per_side {
            let cx = lx0 + ((i as f64 + 0.5) * lw as f64 / per_side as f64) as u64;
            let cy = ly0 + ((j as f64 + 0.5) * lh as f64 / per_side as f64) as u64;
            let bx = cx
                .saturating_sub(SPARSE_BLOCK / 2)
                .min(info.width.saturating_sub(1));
            let by = cy
                .saturating_sub(SPARSE_BLOCK / 2)
                .min(info.height.saturating_sub(1));
            let bw = SPARSE_BLOCK.min(info.width - bx);
            let bh = SPARSE_BLOCK.min(info.height - by);
            if bw == 0 || bh == 0 {
                continue;
            }
            out.push(wsi.read_region(
                (bx as f64 * ds).round() as i64,
                (by as f64 * ds).round() as i64,
                level,
                bw as u32,
                bh as u32,
            )?);
        }
    }
    Ok(out)
}

/// Fits stain vectors for a whole slide.
///
/// Reads the finest pyramid level whose image fits in
/// `options.target_pixels` (or a sparse grid of blocks from the coarsest
/// level for slides without one), applies the tissue mask, estimates the
/// background, fits, sanity-checks and — if the fit is unusable — falls back
/// to the reference vectors (see [`StainEstimate::fallback`]).
///
/// Deterministic and blocking: run it on a worker thread and cache the result
/// per slide identity and method ([`StainEstimateCache`]).
pub fn estimate_slide_stain(
    wsi: &WsiFile,
    method: StainFitMethod,
    options: &StainEstimateOptions,
) -> crate::Result<StainEstimate> {
    // Synthetic levels are approximations (possibly sparse samples); fit
    // on stored pixels only.
    let wsi = &wsi.without_synthetic_levels();
    let props = wsi.properties();
    if wsi.level_count() == 0 || props.width == 0 || props.height == 0 {
        return Err(crate::Error::InvalidFormat(
            "slide has no image data".into(),
        ));
    }
    let rect = (0, 0, props.width, props.height);
    let budget = options.target_pixels.max(64 * 64);
    let (level, ds, fits) = choose_level(wsi, props.width, props.height, budget);
    let buffers = if fits {
        read_rect(wsi, level, ds, rect)?
    } else {
        read_sparse(wsi, level, ds, rect, budget)?
    };
    let slices: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
    Ok(super::estimate::estimate_with_source(
        &slices,
        method,
        options,
        EstimateSource::Slide {
            level,
            downsample: ds,
            sparse: !fits,
        },
    ))
}

/// Fits stain vectors to a level-0 region ("fit to ROI", or measuring custom
/// vectors from a region). Use [`StainEstimateOptions::for_region`] for the
/// defaults, and pass the slide-level background in `options.background` to
/// keep `I₀` consistent when the region contains no glass.
pub fn estimate_region_stain(
    wsi: &WsiFile,
    region: Level0Rect,
    method: StainFitMethod,
    options: &StainEstimateOptions,
) -> crate::Result<StainEstimate> {
    let wsi = &wsi.without_synthetic_levels();
    let props = wsi.properties();
    let Some(rect) = region.clamp_to(props.width, props.height) else {
        return Err(crate::Error::InvalidCoordinates {
            x: region.x as i64,
            y: region.y as i64,
            level: 0,
        });
    };
    let budget = options.target_pixels.max(64 * 64);
    let (level, ds, fits) = choose_level(wsi, rect.2, rect.3, budget);
    let buffers = if fits {
        read_rect(wsi, level, ds, rect)?
    } else {
        read_sparse(wsi, level, ds, rect, budget)?
    };
    let slices: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
    Ok(super::estimate::estimate_with_source(
        &slices,
        method,
        options,
        EstimateSource::Region {
            x: rect.0 as i64,
            y: rect.1 as i64,
            width: rect.2,
            height: rect.3,
            level,
            downsample: ds,
        },
    ))
}

// ─── Cache ───────────────────────────────────────────────────────────────────

/// Cache key: slide identity + method + options (+ region for ROI fits).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StainCacheKey {
    /// Caller-defined slide identity, typically the slide fingerprint.
    pub slide: String,
    pub method: StainFitMethod,
    /// Level-0 region (x, y, width, height) for region fits.
    pub region: Option<[i64; 4]>,
    /// Hash of the estimation options.
    pub options: u64,
}

fn options_hash(options: &StainEstimateOptions) -> u64 {
    let json = serde_json::to_string(options).unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    json.hash(&mut hasher);
    hasher.finish()
}

impl StainCacheKey {
    /// Key for a whole-slide fit.
    pub fn slide(
        slide: impl Into<String>,
        method: StainFitMethod,
        options: &StainEstimateOptions,
    ) -> Self {
        Self {
            slide: slide.into(),
            method,
            region: None,
            options: options_hash(options),
        }
    }

    /// Key for a region fit.
    pub fn region(
        slide: impl Into<String>,
        region: Level0Rect,
        method: StainFitMethod,
        options: &StainEstimateOptions,
    ) -> Self {
        Self {
            slide: slide.into(),
            method,
            region: Some([
                region.x.round() as i64,
                region.y.round() as i64,
                region.width.round() as i64,
                region.height.round() as i64,
            ]),
            options: options_hash(options),
        }
    }

    /// Key for a slide identified by its file: canonical path, size and
    /// modification time (so an overwritten file is refitted).
    pub fn for_file(path: &Path, method: StainFitMethod, options: &StainEstimateOptions) -> Self {
        Self::slide(file_identity(path), method, options)
    }
}

/// Canonical path + length + mtime: a cheap identity for cache keys.
pub fn file_identity(path: &Path) -> String {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let meta = std::fs::metadata(path).ok();
    let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}|{len}|{mtime}", canonical.display())
}

/// A small thread-safe LRU of stain estimates, shared across panes, the
/// export preview and the export (CLR-1).
pub struct StainEstimateCache {
    capacity: usize,
    entries: Mutex<VecDeque<(StainCacheKey, Arc<StainEstimate>)>>,
}

impl std::fmt::Debug for StainEstimateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StainEstimateCache")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl StainEstimateCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(VecDeque::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(StainCacheKey, Arc<StainEstimate>)>> {
        self.entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Looks up an estimate and marks it most recently used.
    pub fn get(&self, key: &StainCacheKey) -> Option<Arc<StainEstimate>> {
        let mut entries = self.lock();
        let index = entries.iter().position(|(k, _)| k == key)?;
        let entry = entries.remove(index)?;
        let value = Arc::clone(&entry.1);
        entries.push_front(entry);
        Some(value)
    }

    /// Inserts (or replaces) an estimate, evicting the least recently used.
    pub fn insert(&self, key: StainCacheKey, estimate: StainEstimate) -> Arc<StainEstimate> {
        let value = Arc::new(estimate);
        let mut entries = self.lock();
        entries.retain(|(k, _)| k != &key);
        entries.push_front((key, Arc::clone(&value)));
        entries.truncate(self.capacity);
        value
    }

    /// Returns the cached estimate or computes and caches it. The lock is
    /// not held while `compute` runs, so concurrent misses may compute twice
    /// (the results are identical: estimation is deterministic).
    pub fn get_or_try_insert_with(
        &self,
        key: StainCacheKey,
        compute: impl FnOnce() -> crate::Result<StainEstimate>,
    ) -> crate::Result<Arc<StainEstimate>> {
        if let Some(hit) = self.get(&key) {
            return Ok(hit);
        }
        let estimate = compute()?;
        Ok(self.insert(key, estimate))
    }

    /// Drops every estimate of one slide (e.g. when it is closed).
    pub fn remove_slide(&self, slide: &str) {
        self.lock().retain(|(k, _)| k.slide != slide);
    }

    pub fn clear(&self) {
        self.lock().clear();
    }
}

/// Process-wide cache used by convenience paths that have no app-owned
/// cache (the legacy export entry point).
pub fn global_stain_cache() -> &'static StainEstimateCache {
    static CACHE: OnceLock<StainEstimateCache> = OnceLock::new();
    CACHE.get_or_init(|| StainEstimateCache::new(32))
}

#[cfg(test)]
mod tests {
    use super::super::basis::StainPreset;
    use super::super::linalg;
    use super::*;
    use crate::test_slides::{PyramidSpec, he_tissue_pixel, write_pyramid_tiff};

    fn tissue_slide(dir: &Path, glass: [u8; 3], levels: &[u32]) -> std::path::PathBuf {
        let path = dir.join("he.tif");
        let (w, h) = (1024, 768);
        write_pyramid_tiff(&path, &PyramidSpec::new(w, h).levels(levels), |x, y| {
            he_tissue_pixel(x, y, w, h, glass)
        })
        .unwrap();
        path
    }

    #[test]
    fn slide_fit_uses_a_low_resolution_level_and_finds_the_stains() {
        let dir = tempfile::tempdir().unwrap();
        let path = tissue_slide(dir.path(), [236, 234, 238], &[1, 2, 4]);
        let wsi = WsiFile::open(&path).unwrap();
        let options = StainEstimateOptions {
            target_pixels: 60_000,
            min_tissue_pixels: 1000,
            ..StainEstimateOptions::default()
        };
        let est = estimate_slide_stain(&wsi, StainFitMethod::Macenko, &options).unwrap();
        assert_eq!(
            est.source,
            EstimateSource::Slide {
                level: 2,
                downsample: 4.0,
                sparse: false
            }
        );
        assert!(est.fallback.is_none(), "{}", est.summary());
        assert!(est.background_found);
        assert!((est.i0[0] - 236.0).abs() <= 2.0, "{:?}", est.i0);
        let truth = StainPreset::HematoxylinEosin.basis().vectors();
        for (f, t) in est.basis.vectors().iter().zip(&truth) {
            assert!(linalg::angle_deg(*f, *t) < 5.0, "{f:?} vs {t:?}");
        }
        // The same slide always gives the same estimate.
        let again = estimate_slide_stain(&wsi, StainFitMethod::Macenko, &options).unwrap();
        assert_eq!(est, again);
    }

    #[test]
    fn single_level_slides_are_sampled_sparsely() {
        let dir = tempfile::tempdir().unwrap();
        let path = tissue_slide(dir.path(), [240, 240, 240], &[1]);
        let wsi = WsiFile::open(&path).unwrap();
        let options = StainEstimateOptions {
            target_pixels: 128 * 128 * 9,
            min_tissue_pixels: 1000,
            ..StainEstimateOptions::default()
        };
        let est = estimate_slide_stain(&wsi, StainFitMethod::Macenko, &options).unwrap();
        assert!(matches!(
            est.source,
            EstimateSource::Slide { sparse: true, .. }
        ));
        assert!(est.examined_pixels <= 128 * 128 * 9);
        assert!(est.fallback.is_none(), "{}", est.summary());
    }

    #[test]
    fn region_fit_reads_only_the_region() {
        let dir = tempfile::tempdir().unwrap();
        let path = tissue_slide(dir.path(), [238, 238, 238], &[1, 2]);
        let wsi = WsiFile::open(&path).unwrap();
        // A region in the middle of the tissue disc: all tissue, no glass.
        let est = estimate_region_stain(
            &wsi,
            Level0Rect::new(412.0, 284.0, 200.0, 200.0),
            StainFitMethod::Macenko,
            &StainEstimateOptions::for_region(),
        )
        .unwrap();
        assert!(matches!(
            est.source,
            EstimateSource::Region { level: 0, .. }
        ));
        assert_eq!(est.examined_pixels, 200 * 200);
        assert!(est.tissue_fraction > 0.9);
        // A region off the slide is an error.
        assert!(
            estimate_region_stain(
                &wsi,
                Level0Rect::new(5000.0, 5000.0, 10.0, 10.0),
                StainFitMethod::Macenko,
                &StainEstimateOptions::for_region(),
            )
            .is_err()
        );
    }

    #[test]
    fn glass_only_slides_fall_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("glass.tif");
        write_pyramid_tiff(&path, &PyramidSpec::new(256, 256), |_, _| [235, 233, 237]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let est = estimate_slide_stain(
            &wsi,
            StainFitMethod::Vahadane,
            &StainEstimateOptions::default(),
        )
        .unwrap();
        assert!(est.is_fallback());
    }

    #[test]
    fn cache_is_lru_and_keyed_by_method_and_options() {
        let cache = StainEstimateCache::new(2);
        let options = StainEstimateOptions::default();
        let dummy = || {
            super::super::estimate::estimate_stain_from_rgba(
                &[&[255u8, 255, 255, 255][..]],
                StainFitMethod::Macenko,
                &options,
            )
        };
        let a = StainCacheKey::slide("a", StainFitMethod::Macenko, &options);
        let b = StainCacheKey::slide("a", StainFitMethod::Vahadane, &options);
        let c = StainCacheKey::slide("c", StainFitMethod::Macenko, &options);
        cache.insert(a.clone(), dummy());
        cache.insert(b.clone(), dummy());
        assert!(cache.get(&a).is_some());
        cache.insert(c.clone(), dummy());
        assert!(
            cache.get(&b).is_none(),
            "least recently used entry is evicted"
        );
        assert!(cache.get(&a).is_some() && cache.get(&c).is_some());
        let other_options = StainEstimateOptions {
            max_samples: 7,
            ..options.clone()
        };
        assert_ne!(
            a,
            StainCacheKey::slide("a", StainFitMethod::Macenko, &other_options)
        );
        let mut calls = 0;
        let hit = cache
            .get_or_try_insert_with(a.clone(), || {
                calls += 1;
                Ok(dummy())
            })
            .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(hit.method, StainFitMethod::Macenko);
        cache.remove_slide("a");
        assert!(cache.get(&a).is_none());
    }
}
