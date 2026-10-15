//! Colour processing for the viewer: every pane's HUD settings resolved into
//! the canonical pipeline of `eov_common::processing` (CLR-5), with stain
//! normalisation fitted once per slide on a worker thread (CLR-1, CLR-7),
//! and each slide's ICC profile read and sampled into its conversion to
//! sRGB on a worker thread too (colour management, AUDIT §14.1). Panes
//! colour-manage a slide that has a usable profile unless the slide's
//! toggle (`OpenFile::color_management`) is off.
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
use eov_common::color_management::DisplayTransform;
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
    /// Colour profiles being read, by slide identity.
    profiles_running: HashSet<String>,
    /// Colour profiles read: their conversion to sRGB, or why there is none.
    profiles: HashMap<String, Result<Arc<DisplayTransform>, Arc<str>>>,
    /// Completed jobs (fits and profile reads, successful or not) per slide
    /// identity.
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

/// `true` while any stain fit or profile read is running (the render loop
/// keeps polling so panes pick the result up).
pub(crate) fn jobs_pending() -> bool {
    let jobs = FIT_JOBS.lock();
    !jobs.running.is_empty() || !jobs.profiles_running.is_empty()
}

// ─── Colour profiles (ICC → sRGB) ────────────────────────────────────────────

/// A slide's colour profile, for colour management.
#[derive(Clone, Debug)]
pub(crate) enum SlideProfile {
    /// The slide has no ICC profile.
    None,
    /// Being read and sampled on a worker thread.
    Pending,
    Ready(Arc<DisplayTransform>),
    /// eov cannot use the profile: the slide is shown as scanned.
    Unsupported(Arc<str>),
}

/// The colour profile of the slide open as `file_id`: from the shared
/// table, or read on a worker thread and [`SlideProfile::Pending`] until it
/// is.
pub(crate) fn slide_profile(file_id: i32, wsi: &WsiFile) -> SlideProfile {
    if wsi.properties().icc_profile_size.is_none() {
        return SlideProfile::None;
    }
    let identity = slide_identity(file_id, wsi);
    let mut jobs = FIT_JOBS.lock();
    match jobs.profiles.get(&identity) {
        Some(Ok(transform)) => return SlideProfile::Ready(Arc::clone(transform)),
        Some(Err(error)) => return SlideProfile::Unsupported(Arc::clone(error)),
        None => {}
    }
    if jobs.profiles_running.insert(identity.clone()) {
        let wsi = wsi.clone();
        let job_identity = identity.clone();
        let spawned = std::thread::Builder::new()
            .name("icc-profile".into())
            .spawn(move || read_profile(wsi, job_identity));
        if let Err(err) = spawned {
            jobs.profiles_running.remove(&identity);
            jobs.profiles.insert(
                identity,
                Err(format!("cannot start a worker thread: {err}").into()),
            );
        }
    }
    SlideProfile::Pending
}

fn read_profile(wsi: WsiFile, identity: String) {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match wsi.icc_profile() {
            Ok(Some(bytes)) => DisplayTransform::from_icc(&bytes)
                .map(Arc::new)
                .map_err(|error| error.to_string()),
            Ok(None) => Err("the slide has no colour profile".to_string()),
            Err(error) => Err(format!("the colour profile cannot be read: {error}")),
        }))
        .unwrap_or_else(|_| Err("reading the colour profile failed unexpectedly".to_string()));
    match &result {
        Ok(transform) => debug!(
            "colour profile of {}: {:?}",
            wsi.properties().path.display(),
            transform
        ),
        Err(error) => warn!(
            "colour management unavailable for {}: {error}",
            wsi.properties().path.display()
        ),
    }
    let mut jobs = FIT_JOBS.lock();
    jobs.profiles_running.remove(&identity);
    *jobs.finished.entry(identity.clone()).or_default() += 1;
    jobs.profiles
        .insert(identity, result.map_err(|error| error.into()));
}

/// Open files whose slide finished a fit or a profile read since the last
/// call: their panes must render again with the estimate or the colour
/// management (or without, if it failed).
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
    transform: Option<Arc<DisplayTransform>>,
    pipeline: Arc<ProcessingPipeline>,
}

/// A pane's colour processing, resolved.
#[derive(Clone)]
pub(crate) struct PaneColor {
    /// What the pane applies (stages 2–6).
    pub(crate) pipeline: Arc<ProcessingPipeline>,
    /// The settings the user chose (they may ask for more than is applied
    /// while a fit is pending or after it failed).
    pub(crate) settings: ColorSettings,
    /// Why stain normalisation is unavailable for this slide, if it is.
    pub(crate) fit_error: Option<Arc<str>>,
    /// The slide's colour profile (whether or not it is applied).
    pub(crate) profile: SlideProfile,
}

/// The conversion a slide's panes apply: its profile's, unless the slide's
/// colour management is off.
pub(crate) fn file_color_management(file: &OpenFile) -> Option<Arc<DisplayTransform>> {
    match slide_profile(file.id, &file.wsi) {
        SlideProfile::Ready(transform) if file.color_management => Some(transform),
        _ => None,
    }
}

/// Resolves the colour processing of `pane` showing `file`, starting the
/// slide's stain fit and profile read if they are needed.
pub(crate) fn pane_color(file: &OpenFile, pane: PaneId) -> PaneColor {
    let hud = file.pane_state(pane).map(|ps| ps.hud).unwrap_or_default();
    let profile = slide_profile(file.id, &file.wsi);
    let transform = match &profile {
        SlideProfile::Ready(transform) if file.color_management => Some(Arc::clone(transform)),
        _ => None,
    };
    let mut color = resolve(
        file.id,
        &file.wsi,
        pane,
        &hud_color_settings(&hud),
        transform,
    );
    color.profile = profile;
    color
}

fn resolve(
    file_id: i32,
    wsi: &WsiFile,
    pane: PaneId,
    settings: &ColorSettings,
    transform: Option<Arc<DisplayTransform>>,
) -> PaneColor {
    let fit = settings
        .required_estimate()
        .map(|method| slide_fit(file_id, wsi, method));
    let (estimate, fit_error) = match fit {
        Some(SlideFit::Ready(estimate)) => (Some(estimate), None),
        Some(SlideFit::Failed(error)) => (None, Some(error)),
        // While pending, the summary says so ("normalisation pending").
        Some(SlideFit::Pending) | None => (None, None),
    };
    // Without a usable fit the slide is shown un-normalised; the HUD says
    // why (see `hud_color_info`).
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
            && same(cached.estimate.as_ref(), estimate.as_ref())
            && same(cached.transform.as_ref(), transform.as_ref())
        {
            return Arc::clone(&cached.pipeline);
        }
        let pipeline = Arc::new(
            applied
                .pipeline(estimate.as_deref())
                .with_color_management(transform.clone()),
        );
        panes.insert(
            key,
            CachedPipeline {
                settings: applied,
                estimate,
                transform,
                pipeline: Arc::clone(&pipeline),
            },
        );
        pipeline
    });
    PaneColor {
        pipeline,
        settings: settings.clone(),
        fit_error,
        profile: SlideProfile::None,
    }
}

fn same<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

// ─── HUD ─────────────────────────────────────────────────────────────────────

/// What the HUD shows about a slide's colour management.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) enum ColorManagementInfo {
    /// The slide has no colour profile.
    #[default]
    NoProfile,
    /// The profile is being read.
    Pending,
    /// Colours converted from the named profile to sRGB.
    On(String),
    /// Colours as scanned; the slide has the named profile.
    Off(String),
    /// The profile cannot be used, and why.
    Unsupported(String),
}

impl ColorManagementInfo {
    /// The HUD indicator's state: 0 none, 1 on, 2 off, 3 pending, 4
    /// unsupported (`HudSettings.color-management`).
    pub(crate) fn code(&self) -> i32 {
        match self {
            Self::NoProfile => 0,
            Self::On(_) => 1,
            Self::Off(_) => 2,
            Self::Pending => 3,
            Self::Unsupported(_) => 4,
        }
    }

    /// The indicator's tooltip.
    pub(crate) fn tooltip(&self) -> String {
        match self {
            Self::NoProfile => String::new(),
            Self::Pending => "Reading the slide's colour profile…".to_string(),
            Self::On(name) => format!(
                "Colour-managed: the slide's profile \"{name}\" is converted to sRGB. Click to show the colours as scanned"
            ),
            Self::Off(name) => format!(
                "Colours as scanned. Click to convert the slide's profile \"{name}\" to sRGB"
            ),
            Self::Unsupported(reason) => {
                format!("Not colour-managed: {reason}")
            }
        }
    }
}

/// What the HUD shows about a pane's colours (CLR-9).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct HudColorInfo {
    /// Any processing is requested (lights the adjustments indicator).
    pub(crate) modified: bool,
    /// Persistent badge, e.g. "Colour-normalised (Macenko) · Sharpened";
    /// empty when the colours are as scanned.
    pub(crate) badge: String,
    /// Problems: stain-label mismatches, unavailable normalisation, …
    pub(crate) warnings: Vec<String>,
    /// The slide's colour management.
    pub(crate) color_management: ColorManagementInfo,
}

/// The HUD's view of `pane` showing `file`.
pub(crate) fn hud_color_info(file: &OpenFile, pane: PaneId) -> HudColorInfo {
    describe(
        &pane_color(file, pane),
        file.wsi.properties().stain.as_deref(),
    )
}

fn describe(color: &PaneColor, stain_label: Option<&str>) -> HudColorInfo {
    let mut warnings = color.settings.warnings(stain_label);
    if let Some(error) = &color.fit_error {
        warnings.push(format!(
            "Colour normalisation is unavailable for this slide: {error}"
        ));
    }
    let color_management = match (&color.profile, color.pipeline.color_management()) {
        (SlideProfile::None, _) => ColorManagementInfo::NoProfile,
        (SlideProfile::Pending, _) => ColorManagementInfo::Pending,
        (SlideProfile::Unsupported(reason), _) => {
            ColorManagementInfo::Unsupported(reason.to_string())
        }
        (SlideProfile::Ready(transform), Some(_)) => {
            ColorManagementInfo::On(transform.description().to_string())
        }
        (SlideProfile::Ready(transform), None) => {
            ColorManagementInfo::Off(transform.description().to_string())
        }
    };
    HudColorInfo {
        modified: color.settings.is_modified(),
        badge: color.pipeline.summary().badge().unwrap_or_default(),
        warnings,
        color_management,
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
        let a = resolve(file_id, &wsi, PaneId(0), &settings, None);
        let b = resolve(file_id, &wsi, PaneId(0), &settings, None);
        assert!(Arc::ptr_eq(&a.pipeline, &b.pipeline));
        settings.tone = ToneAdjustments::new(1.5, 0.0, 1.0);
        let c = resolve(file_id, &wsi, PaneId(0), &settings, None);
        assert!(!Arc::ptr_eq(&a.pipeline, &c.pipeline));

        // Normalisation is pending until the fit lands, then applied.
        settings.normalization = Some(StainFitMethod::Macenko);
        let first = resolve(file_id, &wsi, PaneId(0), &settings, None);
        let info = describe(&first, None);
        assert!(info.modified);
        assert!(
            info.badge.contains("pending") || info.badge.contains("Colour-normalised"),
            "{}",
            info.badge
        );
        wait_for_fit(file_id, &wsi, StainFitMethod::Macenko);
        let ready = resolve(file_id, &wsi, PaneId(0), &settings, None);
        assert!(ready.pipeline.summary().estimate.is_some());
        let info = describe(&ready, Some("H&E"));
        assert!(
            info.badge.starts_with("Colour-normalised (Macenko"),
            "{}",
            info.badge
        );
        assert!(info.warnings.is_empty(), "{:?}", info.warnings);
        // An IHC label warns that the H&E model is being used.
        assert_eq!(describe(&ready, Some("IHC")).warnings.len(), 1);
        retain_files(&HashSet::new());
    }

    #[test]
    fn failed_fits_show_the_slide_unnormalised_and_say_why() {
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
        let color = resolve(file_id, &wsi, PaneId(0), &settings, None);
        let info = describe(&color, None);
        match fit {
            SlideFit::Failed(_) => {
                assert!(color.fit_error.is_some());
                assert!(!info.badge.contains("Colour-normalised"), "{}", info.badge);
                assert!(info.warnings.iter().any(|w| w.contains("unavailable")));
            }
            SlideFit::Ready(estimate) => {
                assert!(estimate.fallback.is_some(), "glass has no tissue to fit");
                assert!(info.badge.contains("reference vectors"), "{}", info.badge);
            }
            SlideFit::Pending => panic!("the fit did not finish"),
        }
        retain_files(&HashSet::new());
    }

    fn wait_for_profile(file_id: i32, wsi: &WsiFile) -> SlideProfile {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            match slide_profile(file_id, wsi) {
                SlideProfile::Pending if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                other => return other,
            }
        }
    }

    /// A slide with an Adobe RGB profile: read once off the calling
    /// thread, then applied to its panes' pipelines, and named on the HUD.
    #[test]
    fn slides_with_a_profile_are_colour_managed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("managed.svs");
        let mut spec = eov_common::test_support::TiffSpec::aperio(256, 256, 64, 4);
        spec.icc_profile = Some(eov_common::color_management::test_profiles::adobe_rgb());
        eov_common::test_support::write_tiff(&path, &spec, |x, y| {
            [200, (x % 200) as u8, (y % 220) as u8]
        })
        .unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let file_id = 9004;
        let SlideProfile::Ready(transform) = wait_for_profile(file_id, &wsi) else {
            panic!("the profile is read");
        };
        assert!(!transform.is_identity());
        // Read once: later lookups share the conversion.
        let SlideProfile::Ready(again) = slide_profile(file_id, &wsi) else {
            panic!("cached");
        };
        assert!(Arc::ptr_eq(&transform, &again));

        let settings = ColorSettings::default();
        let mut on = resolve(
            file_id,
            &wsi,
            PaneId(0),
            &settings,
            Some(Arc::clone(&transform)),
        );
        on.profile = SlideProfile::Ready(Arc::clone(&transform));
        assert!(on.pipeline.color_management().is_some());
        let info = describe(&on, None);
        assert_eq!(info.color_management.code(), 1);
        assert!(!info.modified, "converted colours are not 'processed'");
        assert!(info.badge.is_empty());
        let mut off = resolve(file_id, &wsi, PaneId(0), &settings, None);
        off.profile = SlideProfile::Ready(Arc::clone(&transform));
        assert!(off.pipeline.color_management().is_none());
        assert!(!Arc::ptr_eq(&on.pipeline, &off.pipeline));
        let info = describe(&off, None);
        assert_eq!(info.color_management.code(), 2);
        assert!(
            info.color_management
                .tooltip()
                .contains(transform.description())
        );

        // Slides without a profile have no indicator.
        let plain = he_slide(dir.path());
        assert!(matches!(slide_profile(9005, &plain), SlideProfile::None));
        assert_eq!(ColorManagementInfo::default().code(), 0);
        retain_files(&HashSet::new());
    }

    #[test]
    fn unusable_profiles_are_reported_not_applied() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.svs");
        let mut spec = eov_common::test_support::TiffSpec::aperio(128, 128, 64, 4);
        spec.icc_profile = Some(vec![7; 300]);
        eov_common::test_support::write_tiff(&path, &spec, |_, _| [200, 150, 200]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let SlideProfile::Unsupported(reason) = wait_for_profile(9006, &wsi) else {
            panic!("a broken profile is not used");
        };
        assert!(reason.contains("profile"), "{reason}");
        retain_files(&HashSet::new());
    }
}
