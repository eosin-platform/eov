//! "Save a Copy of the Slide…" (AUDIT §14.3, §14.6).
//!
//! Offered in the tab menu and the Slide Information panel. The dialog
//! chooses the whole slide or its region of interest, TIFF or OME-TIFF,
//! JPEG or lossless tiles, and de-identification (on by default); the copy
//! is written by [`eov_common::convert`] on a worker, as a status-bar job
//! with progress and cancel, using all cores but one so the viewer stays
//! responsive. A pixel size the user set for the slide is recorded in the
//! copy.

use crate::host_ui::with_host;
use crate::state::AppState;
use crate::{AppWindow, SaveCopySettings};
use eov_common::convert::{
    self, ConvertError, ConvertOptions, ConvertPlan, ConvertReport, SlideCompression, SlideFormat,
};
use eov_common::job::JobControl;
use eov_common::{SlideBounds, WsiFile};
use eov_plugin_api::protocol::{NotifyLevel, Rect};
use parking_lot::RwLock;
use slint::{ComponentHandle, SharedString, Timer, TimerMode};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// How often running copies update their status-bar progress.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// The slide the open dialog copies, captured when it opened.
struct Target {
    wsi: WsiFile,
    file_name: String,
    folder: Option<PathBuf>,
    roi: Option<SlideBounds>,
    mpp_override: Option<(f64, f64)>,
}

/// Choices kept from one copy to the next (this session only).
#[derive(Clone, Copy)]
struct Choices {
    format: i32,
    compression: i32,
    deidentify: bool,
}

impl Default for Choices {
    fn default() -> Self {
        Self {
            format: 0,
            compression: 0,
            deidentify: true,
        }
    }
}

#[derive(Default)]
struct Dialog {
    target: Option<Target>,
    choices: Choices,
    /// Progress timers of running copies, by job id.
    timers: HashMap<String, Timer>,
    /// Running copies, by job id, for [`shutdown`].
    running: HashMap<String, (Arc<JobControl>, std::thread::JoinHandle<()>)>,
}

thread_local! {
    static DIALOG: RefCell<Dialog> = RefCell::new(Dialog::default());
}

/// Registers the dialog's callbacks.
pub(crate) fn register(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    {
        let ui_weak = ui.as_weak();
        let state = Arc::clone(state);
        ui.on_save_copy_requested(move |tab_id| {
            if let Some(ui) = ui_weak.upgrade() {
                let file_id = state.read().resolve_tab_file_id(tab_id);
                open(&ui, &state, file_id);
            }
        });
    }
    {
        let ui_weak = ui.as_weak();
        let state = Arc::clone(state);
        ui.on_slide_info_save_copy(move || {
            if let (Some(ui), Some(file_id)) =
                (ui_weak.upgrade(), crate::slide_info::shown_file_id())
            {
                open(&ui, &state, file_id);
            }
        });
    }
    {
        let ui_weak = ui.as_weak();
        ui.on_save_copy_settings_changed(move |settings| {
            if let Some(ui) = ui_weak.upgrade() {
                settings_changed(&ui, settings);
            }
        });
    }
    {
        let ui_weak = ui.as_weak();
        ui.on_save_copy_confirmed(move || {
            if let Some(ui) = ui_weak.upgrade() {
                save(&ui);
            }
        });
    }
}

/// The region of interest as whole level-0 pixels covering it.
fn roi_bounds(roi: &Rect) -> Option<SlideBounds> {
    let (x0, y0) = (roi.x.floor(), roi.y.floor());
    let (x1, y1) = ((roi.x + roi.width).ceil(), (roi.y + roi.height).ceil());
    (x1 > x0 && y1 > y0 && x0.is_finite() && y0.is_finite()).then_some(SlideBounds {
        x: x0 as i64,
        y: y0 as i64,
        width: (x1 - x0) as u64,
        height: (y1 - y0) as u64,
    })
}

/// Threads for a copy made while the viewer runs: all cores but one.
fn background_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(2)
}

fn options_for(settings: &SaveCopySettings, target: &Target) -> ConvertOptions {
    ConvertOptions {
        region: if settings.area == 1 { target.roi } else { None },
        compression: match settings.compression {
            1 => SlideCompression::Jpeg { quality: 80 },
            2 => SlideCompression::Deflate,
            _ => SlideCompression::Jpeg { quality: 90 },
        },
        format: if settings.format == 1 {
            SlideFormat::OmeTiff
        } else {
            SlideFormat::Tiff
        },
        deidentify: settings.deidentify,
        mpp_override: target.mpp_override,
        threads: Some(background_threads()),
        ..ConvertOptions::default()
    }
}

/// What the copy will be, for the dialog.
fn summary(plan: &ConvertPlan) -> String {
    let (width, height) = plan.levels[0];
    let mut text = format!(
        "{width} × {height} px in {} levels of {}-px tiles",
        plan.levels.len(),
        plan.tile_size
    );
    if let Some((x, _)) = plan.mpp {
        text.push_str(&format!(" at {} µm/px", eov_common::format_decimal(x)));
    } else {
        text.push_str("; the slide records no pixel size");
    }
    text.push('.');
    if let Some(channels) = plan.composite_channels {
        text.push_str(&format!(
            " The {channels} channels are copied as the colour composite shown, not as channels."
        ));
    }
    text
}

/// `text` with a capital first letter and a full stop, for the dialog.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    let mut out: String = chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    if !out.ends_with(['.', ')']) {
        out.push('.');
    }
    out
}

/// Re-plans the copy for `settings` and shows the result in the dialog.
fn refresh(ui: &AppWindow, mut settings: SaveCopySettings) {
    let planned = DIALOG.with(|dialog| {
        let dialog = dialog.borrow();
        dialog
            .target
            .as_ref()
            .map(|target| convert::plan_conversion(&target.wsi, &options_for(&settings, target)))
    });
    match planned {
        Some(Ok(plan)) => {
            settings.summary = SharedString::from(summary(&plan));
            settings.error = SharedString::new();
        }
        Some(Err(err)) => {
            settings.summary = SharedString::new();
            settings.error = SharedString::from(sentence(&err.to_string()));
        }
        None => {}
    }
    ui.set_save_copy_settings(settings);
}

fn open(ui: &AppWindow, state: &Arc<RwLock<AppState>>, file_id: i32) {
    let target = {
        let state = state.read();
        let Some(file) = state.get_file(file_id) else {
            return;
        };
        Target {
            wsi: file.wsi.without_synthetic_levels(),
            file_name: file.filename.clone(),
            folder: file.path.parent().map(Path::to_path_buf),
            roi: file.roi.as_ref().and_then(roi_bounds),
            mpp_override: file.calibration_override.map(|um| (um, um)),
        }
    };
    let whole = target.wsi.properties().content_bounds();
    let roi_text = target
        .roi
        .map(|roi| format!("{} × {} px at {}, {}", roi.width, roi.height, roi.x, roi.y))
        .unwrap_or_default();
    let choices = DIALOG.with(|dialog| dialog.borrow().choices);
    let settings = SaveCopySettings {
        title: SharedString::from(target.file_name.as_str()),
        area: 0,
        has_roi: target.roi.is_some(),
        whole_text: SharedString::from(format!("{} × {} px", whole.width, whole.height)),
        roi_text: SharedString::from(roi_text),
        format: choices.format,
        compression: choices.compression,
        deidentify: choices.deidentify,
        summary: SharedString::new(),
        error: SharedString::new(),
    };
    DIALOG.with(|dialog| dialog.borrow_mut().target = Some(target));
    refresh(ui, settings);
    ui.set_save_copy_dialog_visible(true);
}

fn settings_changed(ui: &AppWindow, settings: SaveCopySettings) {
    DIALOG.with(|dialog| {
        dialog.borrow_mut().choices = Choices {
            format: settings.format,
            compression: settings.compression,
            deidentify: settings.deidentify,
        }
    });
    refresh(ui, settings);
}

/// `path` with the format's extension: `.tif`/`.tiff` for TIFF,
/// `.ome.tif`/`.ome.tiff` for OME-TIFF. A TIFF extension of the other
/// kind is replaced; anything else gets the extension appended.
fn with_extension(path: &Path, format: SlideFormat) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lower = name.to_ascii_lowercase();
    let is_ome = [".ome.tif", ".ome.tiff", ".ome.btf"]
        .iter()
        .any(|suffix| lower.ends_with(suffix));
    let is_tiff = !is_ome && (lower.ends_with(".tif") || lower.ends_with(".tiff"));
    match format {
        SlideFormat::Tiff if is_tiff => return path.to_path_buf(),
        SlideFormat::OmeTiff if is_ome => return path.to_path_buf(),
        _ => {}
    }
    let base = [".ome.tiff", ".ome.tif", ".ome.btf", ".tiff", ".tif"]
        .iter()
        .find(|suffix| lower.ends_with(*suffix))
        .map_or(name.as_str(), |suffix| &name[..name.len() - suffix.len()]);
    path.with_file_name(format!("{base}.{}", format.extension()))
}

/// The name the save dialog suggests. A de-identified copy never reuses
/// the slide's file name, which often holds an accession number.
fn suggested_name(file_name: &str, options: &ConvertOptions) -> String {
    let extension = options.format.extension();
    if options.deidentify {
        return format!("deidentified-slide.{extension}");
    }
    let stem = Path::new(file_name)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "slide".to_string());
    format!("{stem}-copy.{extension}")
}

/// Asks for a file name and starts the copy.
fn save(ui: &AppWindow) {
    let mut settings = ui.get_save_copy_settings();
    let prepared = DIALOG.with(|dialog| {
        let dialog = dialog.borrow();
        dialog.target.as_ref().map(|target| {
            (
                target.wsi.clone(),
                target.file_name.clone(),
                target.folder.clone(),
                options_for(&settings, target),
            )
        })
    });
    let Some((wsi, file_name, folder, mut options)) = prepared else {
        return;
    };
    if let Err(err) = convert::plan_conversion(&wsi, &options) {
        settings.error = SharedString::from(sentence(&err.to_string()));
        ui.set_save_copy_settings(settings);
        return;
    }
    // Native dialogs take single extensions; `with_extension` makes the
    // name end in `.ome.tif` for OME-TIFF.
    let mut dialog = rfd::FileDialog::new()
        .add_filter(options.format.label(), &["tif", "tiff"])
        .set_file_name(suggested_name(&file_name, &options));
    if let Some(folder) = folder.filter(|folder| folder.is_dir()) {
        dialog = dialog.set_directory(folder);
    }
    let Some(chosen) = dialog.save_file() else {
        return;
    };
    let path = with_extension(&chosen, options.format);
    // The file dialog asked before replacing the name it returned, not one
    // we changed.
    options.overwrite = path == chosen;
    ui.set_save_copy_dialog_visible(false);
    start(path, wsi, options);
}

fn start(path: PathBuf, wsi: WsiFile, options: ConvertOptions) {
    let control = Arc::new(JobControl::new());
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cancel = {
        let control = Arc::clone(&control);
        Box::new(move || control.cancel()) as Box<dyn Fn()>
    };
    let Some(job_id) =
        with_host(|host| host.start_host_job(&format!("Saving {name}"), Some(cancel)))
    else {
        return;
    };
    let timer = Timer::default();
    {
        let control = Arc::clone(&control);
        let job_id = job_id.clone();
        timer.start(TimerMode::Repeated, PROGRESS_INTERVAL, move || {
            let progress = control.progress();
            let message = (progress.total > 0)
                .then(|| format!("{} of {} tiles", progress.done, progress.total));
            with_host(|host| {
                host.update_host_job(
                    &job_id,
                    (progress.total > 0).then(|| progress.fraction()),
                    message,
                )
            });
        });
    }
    DIALOG.with(|dialog| dialog.borrow_mut().timers.insert(job_id.clone(), timer));

    let deidentified = options.deidentify;
    let spawned = std::thread::Builder::new()
        .name("eov-save-copy".to_string())
        .spawn({
            let job_id = job_id.clone();
            let path = path.clone();
            let control = Arc::clone(&control);
            move || {
                let result = convert::convert_slide(&wsi, &path, &options, &control);
                let _ = slint::invoke_from_event_loop(move || {
                    finished(job_id, path, deidentified, result)
                });
            }
        });
    match spawned {
        // `finished` runs on this (the UI) thread, so after this.
        Ok(handle) => {
            DIALOG.with(|dialog| {
                dialog
                    .borrow_mut()
                    .running
                    .insert(job_id, (control, handle))
            });
        }
        Err(err) => finished(job_id, path, deidentified, Err(ConvertError::Io(err))),
    }
}

/// Cancels the copies still running when eov quits and waits for them to
/// remove their partial files (for at most a few seconds).
pub(crate) fn shutdown() {
    let jobs: Vec<_> = DIALOG.with(|dialog| {
        dialog
            .borrow_mut()
            .running
            .drain()
            .map(|(_, job)| job)
            .collect()
    });
    for (control, _) in &jobs {
        control.cancel();
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    for (_, handle) in jobs {
        while !handle.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if handle.is_finished() {
            let _ = handle.join();
        }
    }
}

fn finished(
    job_id: String,
    path: PathBuf,
    deidentified: bool,
    result: Result<ConvertReport, ConvertError>,
) {
    DIALOG.with(|dialog| {
        let mut dialog = dialog.borrow_mut();
        dialog.timers.remove(&job_id);
        dialog.running.remove(&job_id);
    });
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    with_host(|host| {
        host.finish_host_job(&job_id);
        match result {
            Ok(report) => {
                let (width, height) = report.plan.levels[0];
                let mut facts = vec![
                    format!("{width} × {height} px"),
                    format!("{} levels", report.plan.levels.len()),
                    eov_common::format_file_size(report.bytes),
                ];
                if deidentified {
                    facts.push("de-identified".to_string());
                }
                let detail = format!("{}, saved to {}", facts.join(", "), path.display());
                host.notify(
                    NotifyLevel::Success,
                    &format!("Saved {name}"),
                    Some(&detail),
                    None,
                );
                if let Some(warning) = report.shortfall_message() {
                    host.notify(
                        NotifyLevel::Warning,
                        &format!("{name} is incomplete"),
                        Some(&warning),
                        None,
                    );
                }
                for warning in &report.warnings {
                    host.notify(NotifyLevel::Warning, &name, Some(warning), None);
                }
            }
            Err(ConvertError::Cancelled) => host.status("Copy cancelled; nothing was saved"),
            Err(err) => host.notify(
                NotifyLevel::Error,
                &format!("Could not save {name}"),
                Some(&err.to_string()),
                None,
            ),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opens a 1024 × 768 slide (0.25 µm/px) in a fresh state.
    fn open_slide(dir: &Path) -> (Arc<RwLock<AppState>>, i32) {
        use crate::state::{NewOpenFile, PaneId, TabActivation};
        use crate::tile_loader::TileLoader;
        use eov_common::{TileCache, TileManager, ViewportState};

        let path = dir.join("S12-3456 A1.tif");
        let mut spec = eov_common::test_support::TiffSpec::pyramid(1024, 768, 256, 2);
        spec.pixels_per_cm = Some(40_000);
        eov_common::test_support::write_tiff(&path, &spec, |x, y| {
            [(x % 251) as u8, (y % 241) as u8, 128]
        })
        .unwrap();
        let state = Arc::new(RwLock::new(AppState::new()));
        let tile_cache = Arc::new(TileCache::with_limits(64, 64 << 20));
        let mut guard = state.write();
        let id = guard.allocate_file_id();
        let wsi = WsiFile::open(&path).unwrap();
        let tile_manager = Arc::new(
            TileManager::new(wsi.with_synthetic_levels(), id)
                .with_tile_cache(Arc::clone(&tile_cache)),
        );
        let tile_loader = Arc::new(TileLoader::new(Arc::clone(&tile_manager), tile_cache));
        guard.add_file(NewOpenFile {
            id,
            path,
            wsi,
            tile_manager,
            tile_loader,
            viewport: ViewportState::new(400.0, 300.0, 1024.0, 768.0),
            thumbnail: None,
            target_pane: PaneId::PRIMARY,
            activation: TabActivation::Focus,
        });
        drop(guard);
        (state, id)
    }

    #[test]
    fn the_dialog_offers_the_slide_and_its_region_of_interest() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let dir = tempfile::tempdir().unwrap();
            let (state, id) = open_slide(dir.path());
            let ui = AppWindow::new().unwrap();

            open(&ui, &state, id);
            assert!(ui.get_save_copy_dialog_visible());
            let settings = ui.get_save_copy_settings();
            assert_eq!(settings.title.as_str(), "S12-3456 A1.tif");
            assert!(!settings.has_roi);
            assert_eq!(settings.area, 0);
            assert!(settings.deidentify, "de-identification is on by default");
            assert_eq!(settings.whole_text.as_str(), "1024 × 768 px");
            assert!(
                settings.summary.contains("1024 × 768 px in 2 levels"),
                "{}",
                settings.summary
            );
            assert!(settings.summary.contains("0.25 µm/px"));
            assert!(settings.error.is_empty());

            // With a region of interest, the region can be chosen.
            state.write().get_file_mut(id).unwrap().roi = Some(Rect {
                x: 100.0,
                y: 50.0,
                width: 300.0,
                height: 200.0,
            });
            open(&ui, &state, id);
            let mut settings = ui.get_save_copy_settings();
            assert!(settings.has_roi);
            assert_eq!(settings.roi_text.as_str(), "300 × 200 px at 100, 50");
            settings.area = 1;
            settings.compression = 2;
            settings_changed(&ui, settings);
            let settings = ui.get_save_copy_settings();
            assert!(
                settings.summary.starts_with("300 × 200 px"),
                "{}",
                settings.summary
            );

            // Choices carry over to the next copy.
            open(&ui, &state, id);
            assert_eq!(ui.get_save_copy_settings().compression, 2);

            // A region too small to copy is refused in the dialog.
            state.write().get_file_mut(id).unwrap().roi = Some(Rect {
                x: 0.0,
                y: 0.0,
                width: 8.0,
                height: 8.0,
            });
            open(&ui, &state, id);
            let mut settings = ui.get_save_copy_settings();
            settings.area = 1;
            settings_changed(&ui, settings);
            let settings = ui.get_save_copy_settings();
            assert!(settings.error.contains("at least 16"), "{}", settings.error);
            assert!(settings.summary.is_empty());

            // A manual pixel size is what the copy records.
            state.write().get_file_mut(id).unwrap().calibration_override = Some(0.5);
            open(&ui, &state, id);
            let mut settings = ui.get_save_copy_settings();
            settings.area = 0;
            settings_changed(&ui, settings);
            assert!(
                ui.get_save_copy_settings().summary.contains("0.5 µm/px"),
                "{}",
                ui.get_save_copy_settings().summary
            );
            DIALOG.with(|dialog| *dialog.borrow_mut() = Dialog::default());
        });
    }

    /// The copy runs as a status-bar job and reports its result.
    #[test]
    fn copies_run_as_status_bar_jobs() {
        use slint::Model;

        struct ClearHost;
        impl Drop for ClearHost {
            fn drop(&mut self) {
                crate::host_ui::clear_host();
                DIALOG.with(|dialog| *dialog.borrow_mut() = Dialog::default());
            }
        }

        crate::test_support::run_on_slint_ui_test_thread(|| {
            let _clear = ClearHost;
            let dir = tempfile::tempdir().unwrap();
            let (state, id) = open_slide(dir.path());
            let ui = AppWindow::new().unwrap();
            let tile_cache = Arc::new(eov_common::TileCache::with_limits(64, 64 << 20));
            let plugin_dir = tempfile::tempdir().unwrap();
            let _host = crate::host_ui::init(
                &ui,
                &state,
                &tile_cache,
                &std::rc::Rc::new(Timer::default()),
                crate::host_ui::InitOptions {
                    plugin_dir: plugin_dir.path().to_path_buf(),
                    safe_mode: true,
                },
            );

            open(&ui, &state, id);
            let settings = ui.get_save_copy_settings();
            let (wsi, options) = DIALOG.with(|dialog| {
                let dialog = dialog.borrow();
                let target = dialog.target.as_ref().unwrap();
                (target.wsi.clone(), options_for(&settings, target))
            });
            assert!(options.deidentify && options.threads.is_some());
            let output = dir.path().join("teaching.tif");
            start(output.clone(), wsi, options);
            let jobs = ui.get_jobs();
            assert_eq!(jobs.row_count(), 1);
            let job = jobs.row_data(0).unwrap();
            assert_eq!(job.title.as_str(), "Saving teaching.tif");
            assert!(job.cancellable);

            // Wait for the worker, then let the event loop deliver its result.
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            while DIALOG.with(|dialog| {
                dialog
                    .borrow()
                    .running
                    .values()
                    .any(|(_, handle)| !handle.is_finished())
            }) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the copy never finished"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            slint::invoke_from_event_loop(|| {
                let _ = slint::quit_event_loop();
            })
            .unwrap();
            slint::run_event_loop().unwrap();

            assert_eq!(ui.get_jobs().row_count(), 0, "the job is finished");
            assert!(DIALOG.with(|dialog| dialog.borrow().running.is_empty()));
            let notice = ui.get_notifications().row_data(0).unwrap();
            assert_eq!(notice.message.as_str(), "Saved teaching.tif");
            assert!(notice.detail.contains("de-identified"), "{}", notice.detail);
            let copy = WsiFile::open(&output).unwrap();
            assert_eq!(copy.properties().vendor.as_deref(), Some("generic-tiff"));
            let comment = copy.properties().comment.clone().unwrap_or_default();
            assert!(!comment.contains("S12-3456"), "{comment}");
        });
    }

    #[test]
    fn regions_of_interest_cover_whole_pixels() {
        let roi = Rect {
            x: 10.4,
            y: 20.6,
            width: 100.2,
            height: 0.5,
        };
        assert_eq!(
            roi_bounds(&roi),
            Some(SlideBounds {
                x: 10,
                y: 20,
                width: 101,
                height: 2,
            })
        );
        let empty = Rect {
            x: 5.0,
            y: 5.0,
            width: 0.0,
            height: 3.0,
        };
        assert_eq!(roi_bounds(&empty), None);
    }

    #[test]
    fn file_names_get_the_format_extension() {
        let tiff = SlideFormat::Tiff;
        let ome = SlideFormat::OmeTiff;
        assert_eq!(
            with_extension(Path::new("/a/copy"), tiff),
            Path::new("/a/copy.tif")
        );
        assert_eq!(
            with_extension(Path::new("/a/copy.TIFF"), tiff),
            Path::new("/a/copy.TIFF")
        );
        assert_eq!(
            with_extension(Path::new("/a/copy.tif"), ome),
            Path::new("/a/copy.ome.tif")
        );
        assert_eq!(
            with_extension(Path::new("/a/copy.ome.tiff"), ome),
            Path::new("/a/copy.ome.tiff")
        );
        // A plain TIFF must not be named like an OME-TIFF.
        assert_eq!(
            with_extension(Path::new("/a/copy.ome.tif"), tiff),
            Path::new("/a/copy.tif")
        );
        assert_eq!(
            with_extension(Path::new("/a/x.png"), tiff),
            Path::new("/a/x.png.tif")
        );
    }

    #[test]
    fn deidentified_copies_never_suggest_the_slide_name() {
        let mut options = ConvertOptions {
            deidentify: true,
            ..ConvertOptions::default()
        };
        let name = suggested_name("S12-3456 A1.svs", &options);
        assert!(!name.contains("S12"), "{name}");
        assert_eq!(name, "deidentified-slide.tif");
        options.deidentify = false;
        options.format = SlideFormat::OmeTiff;
        assert_eq!(
            suggested_name("S12-3456 A1.svs", &options),
            "S12-3456 A1-copy.ome.tif"
        );
    }
}
