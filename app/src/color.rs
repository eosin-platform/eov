//! Colour processing for the viewer: every pane's HUD settings resolved into
//! the canonical pipeline of `eov_common::processing` (CLR-5), with stain
//! normalisation fitted once per slide on a worker thread (CLR-1, CLR-7).
//!
//! Fits go into the process-wide stain cache under the same key the export
//! path uses (`StainCacheKey::for_file` with default options), so panes,
//! the export preview and the export all colour a slide identically.
//!
//! Everything except the fits themselves runs on the UI thread: resolved
//! pipelines are cached per pane and rebuilt only when the settings or the
//! slide's estimate change.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use eov_common::WsiFile;
use eov_common::postprocess::ToneAdjustments;
use eov_common::processing::{ColorSettings, ProcessingPipeline};
use eov_common::stain::{
    StainCacheKey, StainEstimate, StainEstimateOptions, StainFitMethod, estimate_slide_stain,
    file_identity, global_stain_cache,
};
use parking_lot::Mutex;
use tracing::{debug, warn};

use crate::state::{HudSettings, IsolatedChannel, OpenFile, PaneId};

/// Colour of the viewport outside the slide (stage 5 of the pipeline).
pub(crate) const VIEWPORT_BACKGROUND: [u8; 3] = [30, 30, 30];

/// The colour settings a pane's HUD describes.
pub(crate) fn hud_color_settings(hud: &HudSettings) -> ColorSettings {
    ColorSettings::from_legacy(
        hud.stain_normalization,
        hud.deconv_hematoxylin_intensity,
        hud.deconv_hematoxylin_visible,
        hud.deconv_eosin_intensity,
        hud.deconv_eosin_visible,
        match hud.deconv_isolated_channel {
            IsolatedChannel::None => 0,
            IsolatedChannel::Hematoxylin => 1,
            IsolatedChannel::Eosin => 2,
        },
        ToneAdjustments::new(hud.gamma, hud.brightness, hud.contrast),
        hud.sharpness,
    )
}

// ─── Slide-level stain fits ──────────────────────────────────────────────────

/// State of a slide's stain fit for one method.
#[derive(Clone, Debug)]
pub(crate) enum SlideFit {
    Ready(Arc<StainEstimate>),
    /// Being computed on a worker thread.
    Pending,
    /// The slide could not be fitted (not retried for this file version).
    Failed(Arc<str>),
}

#[derive(Default)]
struct FitJobs {
    running: HashSet<StainCacheKey>,
    failed: HashMap<StainCacheKey, Arc<str>>,
    /// Completed fits (successful or not) per slide identity.
    finished: HashMap<String, u64>,
}

static FIT_JOBS: LazyLock<Mutex<FitJobs>> = LazyLock::new(|| Mutex::new(FitJobs::default()));

/// The options every viewer fit uses; the same as the export path's, so
/// both find the same cached estimate.
fn fit_options() -> StainEstimateOptions {
    StainEstimateOptions::default()
}

struct FileEntry {
    /// `stain::file_identity` of the slide (computed once per open file).
    identity: String,
    /// `FitJobs::finished` count last reported by [`files_with_new_fits`].
    seen_fits: u64,
}

thread_local! {
    static FILES: RefCell<HashMap<i32, FileEntry>> = RefCell::new(HashMap::new());
    static PANES: RefCell<HashMap<(i32, usize), CachedPipeline>> = RefCell::new(HashMap::new());
}

fn slide_identity(file_id: i32, wsi: &WsiFile) -> String {
    FILES.with(|files| {
        files
            .borrow_mut()
            .entry(file_id)
            .or_insert_with(|| {
                let identity = file_identity(&wsi.properties().path);
                let seen_fits = FIT_JOBS
                    .lock()
                    .finished
                    .get(&identity)
                    .copied()
                    .unwrap_or(0);
                FileEntry {
                    identity,
                    seen_fits,
                }
            })
            .identity
            .clone()
    })
}

/// The stain fit of the slide open as `file_id` (fitted on its real pyramid
/// levels): from the shared cache, or started on a worker thread and
/// [`SlideFit::Pending`] until it finishes.
pub(crate) fn slide_fit(file_id: i32, wsi: &WsiFile, method: StainFitMethod) -> SlideFit {
    let options = fit_options();
    let key = StainCacheKey::slide(slide_identity(file_id, wsi), method, &options);
    if let Some(estimate) = global_stain_cache().get(&key) {
        return SlideFit::Ready(estimate);
    }
    let mut jobs = FIT_JOBS.lock();
    if let Some(error) = jobs.failed.get(&key) {
        return SlideFit::Failed(Arc::clone(error));
    }
    // Re-check under the lock: a fit that finished since the lookup above
    // has already left `running`.
    if let Some(estimate) = global_stain_cache().get(&key) {
        return SlideFit::Ready(estimate);
    }
    if jobs.running.insert(key.clone()) {
        let wsi = wsi.clone();
        let job_key = key.clone();
        let spawned = std::thread::Builder::new()
            .name("stain-fit".into())
            .spawn(move || run_fit(wsi, method, options, job_key));
        if let Err(err) = spawned {
            jobs.running.remove(&key);
            jobs.failed
                .insert(key, format!("cannot start a worker thread: {err}").into());
        }
    }
    SlideFit::Pending
}

fn run_fit(
    wsi: WsiFile,
    method: StainFitMethod,
    options: StainEstimateOptions,
    key: StainCacheKey,
) {
    let started = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        estimate_slide_stain(&wsi, method, &options)
    }));
    let failure: Option<Arc<str>> = match result {
        Ok(Ok(estimate)) => {
            debug!(
                "stain fit of {} in {:.0} ms: {}",
                wsi.properties().path.display(),
                started.elapsed().as_secs_f64() * 1000.0,
                estimate.summary()
            );
            // Into the cache before leaving `running`, so a concurrent
            // lookup never sees neither.
            global_stain_cache().insert(key.clone(), estimate);
            None
        }
        Ok(Err(err)) => Some(err.to_string().into()),
        Err(_) => Some("the stain fit failed unexpectedly".into()),
    };
    if let Some(error) = &failure {
        warn!(
            "stain normalisation unavailable for {}: {error}",
            wsi.properties().path.display()
        );
    }
    let mut jobs = FIT_JOBS.lock();
    jobs.running.remove(&key);
    if let Some(error) = failure {
        jobs.failed.insert(key.clone(), error);
    }
    *jobs.finished.entry(key.slide).or_default() += 1;
}

/// `true` while any stain fit is running (the render loop keeps polling so
/// panes pick the result up).
pub(crate) fn fits_pending() -> bool {
    !FIT_JOBS.lock().running.is_empty()
}

/// Open files whose slide finished a fit since the last call: their panes
/// must render again with the estimate (or without normalisation, if it
/// failed).
pub(crate) fn files_with_new_fits() -> Vec<i32> {
    let jobs = FIT_JOBS.lock();
    if jobs.finished.is_empty() {
        return Vec::new();
    }
    FILES.with(|files| {
        files
            .borrow_mut()
            .iter_mut()
            .filter_map(|(&file_id, entry)| {
                let finished = jobs.finished.get(&entry.identity).copied().unwrap_or(0);
                (finished != entry.seen_fits).then(|| {
                    entry.seen_fits = finished;
                    file_id
                })
            })
            .collect()
    })
}

/// Forgets per-file and per-pane state of files that are no longer open.
pub(crate) fn retain_files(open: &HashSet<i32>) {
    FILES.with(|files| files.borrow_mut().retain(|id, _| open.contains(id)));
    PANES.with(|panes| panes.borrow_mut().retain(|(id, _), _| open.contains(id)));
}

// ─── Per-pane pipelines ──────────────────────────────────────────────────────

struct CachedPipeline {
    settings: ColorSettings,
    estimate: Option<Arc<StainEstimate>>,
    pipeline: Arc<ProcessingPipeline>,
}

/// A pane's colour processing, resolved.
#[derive(Clone)]
pub(crate) struct PaneColor {
    /// What the pane applies (stages 2–6).
    pub(crate) pipeline: Arc<ProcessingPipeline>,
}

/// Resolves the colour processing of `pane` showing `file`, starting the
/// slide's stain fit if the settings need one.
pub(crate) fn pane_color(file: &OpenFile, pane: PaneId) -> PaneColor {
    let hud = file.pane_state(pane).map(|ps| ps.hud).unwrap_or_default();
    resolve(file.id, &file.wsi, pane, &hud_color_settings(&hud))
}

fn resolve(file_id: i32, wsi: &WsiFile, pane: PaneId, settings: &ColorSettings) -> PaneColor {
    let fit = settings
        .required_estimate()
        .map(|method| slide_fit(file_id, wsi, method));
    let (estimate, fit_error) = match fit {
        Some(SlideFit::Ready(estimate)) => (Some(estimate), None),
        Some(SlideFit::Failed(error)) => (None, Some(error)),
        // While pending, the summary says so ("normalisation pending").
        Some(SlideFit::Pending) | None => (None, None),
    };
    // Without a usable fit the slide is shown un-normalised.
    let applied = if fit_error.is_some() {
        ColorSettings {
            normalization: None,
            ..settings.clone()
        }
    } else {
        settings.clone()
    };
    let pipeline = PANES.with(|panes| {
        let mut panes = panes.borrow_mut();
        let key = (file_id, pane.0);
        if let Some(cached) = panes.get(&key)
            && cached.settings == applied
            && same_estimate(cached.estimate.as_ref(), estimate.as_ref())
        {
            return Arc::clone(&cached.pipeline);
        }
        let pipeline = Arc::new(applied.pipeline(estimate.as_deref()));
        panes.insert(
            key,
            CachedPipeline {
                settings: applied,
                estimate,
                pipeline: Arc::clone(&pipeline),
            },
        );
        pipeline
    });
    PaneColor { pipeline }
}

fn same_estimate(a: Option<&Arc<StainEstimate>>, b: Option<&Arc<StainEstimate>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::StainNormalization;

    fn wait_for_fit(file_id: i32, wsi: &WsiFile, method: StainFitMethod) -> SlideFit {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            match slide_fit(file_id, wsi, method) {
                SlideFit::Pending if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                other => return other,
            }
        }
    }

    fn he_slide(dir: &std::path::Path) -> WsiFile {
        // Beer–Lambert mixtures of hematoxylin and eosin on glass.
        const H: [f32; 3] = [0.644, 0.717, 0.267];
        const E: [f32; 3] = [0.093, 0.954, 0.283];
        let path = dir.join("he.tif");
        eov_common::test_support::write_tiled_tiff(&path, 512, 512, 256, |x, y| {
            if x < 64 || y < 64 {
                return [242, 240, 244];
            }
            let ch = ((x * 7 + y * 3) % 50) as f32 / 50.0 * 1.4;
            let ce = ((x * 3 + y * 11) % 40) as f32 / 40.0 * 0.9 + 0.1;
            [0, 1, 2].map(|c| (242.0 * (-(ch * H[c] + ce * E[c])).exp()).round() as u8)
        })
        .unwrap();
        WsiFile::open(&path).unwrap()
    }

    #[test]
    fn neutral_hud_settings_are_the_identity() {
        let settings = hud_color_settings(&HudSettings::default());
        assert!(!settings.is_modified());
        assert!(settings.pipeline(None).is_identity());
    }

    #[test]
    fn hud_settings_map_onto_the_colour_pipeline() {
        let hud = HudSettings {
            sharpness: 0.4,
            gamma: 1.2,
            stain_normalization: StainNormalization::Vahadane,
            deconv_eosin_visible: false,
            deconv_isolated_channel: IsolatedChannel::Eosin,
            ..HudSettings::default()
        };
        let settings = hud_color_settings(&hud);
        assert_eq!(settings.normalization, Some(StainFitMethod::Vahadane));
        assert_eq!(settings.isolated, Some(1));
        assert!(!settings.channels[1].visible);
        assert!((settings.tone.gamma - 1.2).abs() < 1e-6);
        assert!((settings.sharpness - 0.4).abs() < 1e-6);
    }

    #[test]
    fn slides_are_fitted_once_off_the_calling_thread() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = he_slide(dir.path());
        let file_id = 9001;
        let method = StainFitMethod::Macenko;
        // The first request starts the fit and does not wait for it.
        assert!(matches!(
            slide_fit(file_id, &wsi, method),
            SlideFit::Pending | SlideFit::Ready(_)
        ));
        let SlideFit::Ready(first) = wait_for_fit(file_id, &wsi, method) else {
            panic!("the fit did not succeed");
        };
        // The completion is reported once (it is recorded just after the
        // estimate is cached).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut reported = files_with_new_fits();
        while reported.is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
            reported = files_with_new_fits();
        }
        assert_eq!(reported, vec![file_id]);
        assert!(files_with_new_fits().is_empty(), "reported once");
        assert!(!FIT_JOBS.lock().running.contains(&StainCacheKey::slide(
            file_identity(&wsi.properties().path),
            method,
            &fit_options()
        )));
        // Later requests reuse the cached estimate (the export path's key).
        let SlideFit::Ready(again) = slide_fit(file_id, &wsi, method) else {
            panic!("the estimate is cached");
        };
        assert!(Arc::ptr_eq(&first, &again));
        let export_key = StainCacheKey::for_file(
            &wsi.properties().path,
            method,
            &StainEstimateOptions::default(),
        );
        assert!(Arc::ptr_eq(
            &global_stain_cache().get(&export_key).unwrap(),
            &first
        ));
        retain_files(&HashSet::new());
    }

    #[test]
    fn pane_pipelines_are_rebuilt_only_when_something_changes() {
        let dir = tempfile::tempdir().unwrap();
        let wsi = he_slide(dir.path());
        let file_id = 9002;
        let mut settings = hud_color_settings(&HudSettings {
            gamma: 1.4,
            ..HudSettings::default()
        });
        let a = resolve(file_id, &wsi, PaneId(0), &settings);
        let b = resolve(file_id, &wsi, PaneId(0), &settings);
        assert!(Arc::ptr_eq(&a.pipeline, &b.pipeline));
        settings.tone = ToneAdjustments::new(1.5, 0.0, 1.0);
        let c = resolve(file_id, &wsi, PaneId(0), &settings);
        assert!(!Arc::ptr_eq(&a.pipeline, &c.pipeline));

        // Normalisation is pending until the fit lands, then applied.
        settings.normalization = Some(StainFitMethod::Macenko);
        let first = resolve(file_id, &wsi, PaneId(0), &settings);
        let badge = first.pipeline.summary().badge().unwrap_or_default();
        assert!(
            badge.contains("pending") || badge.contains("Colour-normalised"),
            "{badge}"
        );
        wait_for_fit(file_id, &wsi, StainFitMethod::Macenko);
        let ready = resolve(file_id, &wsi, PaneId(0), &settings);
        assert!(ready.pipeline.summary().estimate.is_some());
        let badge = ready.pipeline.summary().badge().unwrap_or_default();
        assert!(badge.starts_with("Colour-normalised (Macenko"), "{badge}");
        retain_files(&HashSet::new());
    }

    #[test]
    fn failed_fits_show_the_slide_unnormalised() {
        let dir = tempfile::tempdir().unwrap();
        // Uniform glass: no tissue to fit.
        let path = dir.path().join("glass.tif");
        eov_common::test_support::write_tiled_tiff(&path, 256, 256, 256, |_, _| [250, 250, 250])
            .unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let settings = ColorSettings {
            normalization: Some(StainFitMethod::Macenko),
            ..ColorSettings::default()
        };
        // Either the fit fails or it falls back to reference vectors; both
        // must leave a usable, honest pipeline.
        let file_id = 9003;
        let fit = wait_for_fit(file_id, &wsi, StainFitMethod::Macenko);
        let color = resolve(file_id, &wsi, PaneId(0), &settings);
        let badge = color.pipeline.summary().badge().unwrap_or_default();
        match fit {
            SlideFit::Failed(_) => {
                assert!(color.pipeline.is_identity());
                assert!(!badge.contains("Colour-normalised"), "{badge}");
            }
            SlideFit::Ready(estimate) => {
                assert!(estimate.fallback.is_some(), "glass has no tissue to fit");
                assert!(badge.contains("reference vectors"), "{badge}");
            }
            SlideFit::Pending => panic!("the fit did not finish"),
        }
        retain_files(&HashSet::new());
    }
}
