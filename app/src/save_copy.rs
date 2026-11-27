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
    self, ConvertError, ConvertOptions, ConvertPlan, ConvertProblem, ConvertReport, CopyOmission,
    SlideCompression, SlideFormat,
};
use eov_common::job::JobControl;
use eov_common::{SlideBounds, WsiFile};
use eov_plugin_api::i18n::{isolate, join_sentences, tr};
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
    let levels = plan.levels.len();
    // The numbers isolated (the units are the translation's).
    let (width, height, tile) = (isolate(width), isolate(height), isolate(plan.tile_size));
    let mut text = match plan.mpp {
        // Translators: the size of the copy, its pyramid levels, the size of
        // its tiles and its pixel size.
        Some((x, _)) => tr!(
            "{width} × {height} px in {n} level of {tile}-px tiles at {mpp} µm/px."
                | "{width} × {height} px in {n} levels of {tile}-px tiles at {mpp} µm/px." % levels,
            width = width,
            height = height,
            tile = tile,
            mpp = isolate(crate::ui_update::localized_number(
                &eov_common::format_decimal(x)
            ))
        ),
        None => tr!(
            "{width} × {height} px in {n} level of {tile}-px tiles; the slide records no pixel size."
                | "{width} × {height} px in {n} levels of {tile}-px tiles; the slide records no pixel size."
                    % levels,
            width = width,
            height = height,
            tile = tile
        ),
    };
    if let Some(channels) = plan.composite_channels {
        text = join_sentences([
            text,
            tr!(
                // Translators: the slide's {n} channels are copied as one colour
                // image, the composite the window shows.
                "The {n} channel is copied as the colour composite shown, not as a separate channel."
                    | "The {n} channels are copied as the colour composite shown, not as separate channels."
                        % channels
            ),
        ]);
    }
    text
}

/// An English message (as eov_common and the system word it) as a
/// sentence: with a capital first letter and a full stop. The translated
/// messages are whole sentences already.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    let mut out: String = chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    // Not after the end of a sentence or a bracket, in any script.
    if !out.ends_with(['.', ')', '。', '）', '!', '?', '！', '？', '؟', '।']) {
        out.push('.');
    }
    out
}

/// Why a copy cannot be made, as a sentence in the interface's language;
/// what reading the slide or writing the file says as it comes.
fn convert_error(error: &ConvertError) -> String {
    match error {
        ConvertError::Invalid(problem) => convert_problem(problem),
        // Translators: shown as a sentence on its own; {} is the path of a
        // file.
        ConvertError::OutputExists(path) => tr!("{} already exists.", isolate(path.display())),
        other => sentence(&other.to_string()),
    }
}

/// Why the options of a copy cannot work ([`ConvertProblem`]), as a
/// sentence in the interface's language.
fn convert_problem(problem: &ConvertProblem) -> String {
    match problem {
        // Translators: shown as a sentence on its own: the region of
        // interest chosen for a copy of the slide is empty.
        ConvertProblem::EmptyRegion => tr!("The region is empty."),
        ConvertProblem::RegionOutside {
            region,
            slide_width,
            slide_height,
        } => {
            // Translators: shown as a sentence on its own: the region of
            // interest (its position and size in pixels) is outside the
            // slide (its size in pixels).
            tr!(
                "The region {x},{y},{width},{height} is outside the slide ({slide_width} × {slide_height} pixels)",
                x = region.x,
                y = region.y,
                width = region.width,
                height = region.height,
                slide_width = slide_width,
                slide_height = slide_height
            )
        }
        // Translators: shown as a sentence on its own: the size of the copy
        // of a slide, and the largest it can be.
        ConvertProblem::TooLarge { width, height } => tr!(
            "The copy would be {width} × {height} pixels; at most {max} per side.",
            width = width,
            height = height,
            max = convert::MAX_SIDE
        ),
        // Translators: shown as a sentence on its own: the size of the copy
        // of a slide, and the smallest it can be.
        ConvertProblem::TooSmall { width, height } => tr!(
            "The copy would be {width} × {height} pixels; it must be at least {min} × {min} (choose a larger region or a finer resolution)",
            width = width,
            height = height,
            min = convert::MIN_SIDE
        ),
        ConvertProblem::NotAFileName(path) => {
            // Translators: shown as a sentence on its own; {} is a path the
            // copy of a slide was to be saved to.
            tr!("{} is not a file name.", isolate(path.display()))
        }
        // Translators: shown as a sentence on its own; {} is a path the copy
        // of a slide was to be saved to.
        ConvertProblem::IsAFolder(path) => tr!("{} is a folder.", isolate(path.display())),
        ConvertProblem::NoSuchFolder(path) => {
            // Translators: shown as a sentence on its own; {} is the folder
            // the copy of a slide was to be saved in.
            tr!("The folder {} does not exist.", isolate(path.display()))
        }
        // Translators: shown as a sentence on its own; {} is a path the copy
        // of a slide was to be saved to.
        ConvertProblem::IsTheSlide(path) => tr!(
            "{} is the slide being copied (or one of its files)",
            isolate(path.display())
        ),
        // Translators: shown as a sentence on its own; {} is the temporary
        // file of a copy of a slide.
        ConvertProblem::PartialExists(path) => tr!(
            "{} exists: another conversion is writing this file, or an interrupted one left it behind (delete it and try again)",
            isolate(path.display())
        ),
        // Options the dialog never sets (the command line and scripts do):
        // as eov_common words them.
        ConvertProblem::TileSize(_)
        | ConvertProblem::JpegQuality(_)
        | ConvertProblem::NoSuchLevel { .. }
        | ConvertProblem::NotAPixelSize(_)
        | ConvertProblem::NoPixelSize
        | ConvertProblem::FinerThanSlide { .. } => sentence(&problem.to_string()),
    }
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
            settings.error = SharedString::from(convert_error(&err));
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
            // The plane and channels shown now, whatever the viewer shows
            // while the copy is written.
            wsi: file.wsi.without_synthetic_levels().pinned(),
            file_name: file.filename.clone(),
            folder: file.path.parent().map(Path::to_path_buf),
            roi: file.roi.as_ref().and_then(roi_bounds),
            mpp_override: file.calibration_override.map(|um| (um, um)),
        }
    };
    let whole = target.wsi.properties().content_bounds();
    let roi_text = target.roi.map(|roi| roi_text(&roi)).unwrap_or_default();
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

/// The region of interest, as the dialog offers it.
fn roi_text(roi: &SlideBounds) -> String {
    // Translators: the size of the region of interest, then the position
    // of its top-left corner on the slide.
    tr!(
        "{width} × {height} px at {x}, {y}",
        width = roi.width,
        height = roi.height,
        x = roi.x,
        y = roi.y
    )
}

/// Writes the open dialog's texts again, in the language of the interface
/// (after it changed).
pub(crate) fn refresh_texts(ui: &AppWindow) {
    if !ui.get_save_copy_dialog_visible() {
        return;
    }
    let mut settings = ui.get_save_copy_settings();
    if let Some(roi) = DIALOG.with(|dialog| dialog.borrow().target.as_ref().and_then(|t| t.roi)) {
        settings.roi_text = SharedString::from(roi_text(&roi));
    }
    refresh(ui, settings);
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
        settings.error = SharedString::from(convert_error(&err));
        ui.set_save_copy_settings(settings);
        return;
    }
    // Native dialogs take single extensions; `with_extension` makes the
    // name end in `.ome.tif` for OME-TIFF.
    let mut dialog = rfd::FileDialog::new()
        .add_filter(format_name(options.format), &["tif", "tiff"])
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
        with_host(|host| host.start_host_job(&tr!("Saving {}", isolate(&name)), Some(cancel)))
    else {
        return;
    };
    let timer = Timer::default();
    {
        let control = Arc::clone(&control);
        let job_id = job_id.clone();
        timer.start(TimerMode::Repeated, PROGRESS_INTERVAL, move || {
            let progress = control.progress();
            let message = (progress.total > 0).then(|| {
                tr!(
                    "{done} of {n} tile" | "{done} of {n} tiles" % progress.total,
                    done = progress.done
                )
            });
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
                let levels = report.plan.levels.len();
                let mut facts = vec![
                    format!("{width} × {height} px"),
                    tr!("{n} level" | "{n} levels" % levels),
                    crate::ui_update::localized_number(&eov_common::format_file_size(report.bytes)),
                ];
                if deidentified {
                    // Translators: about a copy of a slide without its label,
                    // macro and thumbnail images and with no properties but the
                    // pixel size, objective power and stain; text or marks
                    // inside the scanned tissue image are kept.
                    facts.push(tr!("de-identified"));
                }
                // Translators: facts about the copy ("1024 × 768 px, 3
                // levels, 12 MB"), then where it was saved.
                let detail = tr!(
                    "{facts}, saved to {path}",
                    facts = crate::processing_text::list(facts),
                    path = isolate(path.display())
                );
                host.notify(
                    NotifyLevel::Success,
                    &tr!("Saved {}", isolate(&name)),
                    Some(&detail),
                    None,
                );
                if let Some(warning) = shortfall(&report) {
                    host.notify(
                        NotifyLevel::Warning,
                        &tr!("{} is incomplete", isolate(&name)),
                        Some(&warning),
                        None,
                    );
                }
                for warning in omissions(&report) {
                    host.notify(NotifyLevel::Warning, &name, Some(&warning), None);
                }
            }
            Err(ConvertError::Cancelled) => host.status(&tr!("Copy cancelled; nothing was saved")),
            Err(err) => host.notify(
                NotifyLevel::Error,
                &tr!("Could not save {}", isolate(&name)),
                Some(&convert_error(&err)),
                None,
            ),
        }
    });
}

/// What a copy could not read ([`ConvertReport::shortfall_message`]), in
/// the interface's language. The plural form follows the tiles that could
/// not be read, which the verb agrees with; the number of tiles is a bare
/// number after them.
fn shortfall(report: &ConvertReport) -> Option<String> {
    (report.unreadable_tiles > 0).then(|| {
        // Translators: {n} tiles of the slide could not be read for a copy
        // of it (the form follows {n}); {total} is the number of tiles of
        // the copy. The copy shows the slide's background colour there.
        tr!(
            "{n} tile of {total} could not be read from the slide and shows its background colour."
                | "{n} tiles of {total} could not be read from the slide and show its background colour."
                % report.unreadable_tiles,
            total = report.plan.tiles
        )
    })
}

/// What a copy left out ([`ConvertReport::warnings`]), in the interface's
/// language; the reasons (errors reading the slide) as they come.
fn omissions(report: &ConvertReport) -> Vec<String> {
    if report.omissions.len() != report.warnings.len() {
        return report.warnings.clone();
    }
    report
        .omissions
        .iter()
        .map(|omission| match omission {
            CopyOmission::AssociatedImage { name, error } => match name.as_str() {
                // Translators: {error} says why the slide's label image was not copied.
                "label" => tr!("the label image was not copied: {error}", error = error),
                // Translators: {error} says why the slide's macro image (a photo of
                // the whole glass) was not copied.
                "macro" => tr!("the macro image was not copied: {error}", error = error),
                // Translators: {error} says why the slide's thumbnail image was not copied.
                "thumbnail" => tr!("the thumbnail image was not copied: {error}", error = error),
                // Translators: an extra image of the slide, by the name the file
                // gives it ({name}, not translated), was not copied; {error} says why.
                _ => tr!(
                    "the extra image {name} was not copied: {error}",
                    name = isolate(name),
                    error = error
                ),
            },
            CopyOmission::IccProfile { error } => {
                // Translators: {error} says why the slide's colour profile
                // was not copied.
                tr!("the ICC profile was not copied: {error}", error = error)
            }
        })
        .collect()
}

/// The name of a copy's file type, for the file dialog.
fn format_name(format: SlideFormat) -> String {
    match format {
        // Translators: a file type: a TIFF file holding the slide at
        // several resolutions.
        SlideFormat::Tiff => tr!("Pyramidal TIFF"),
        SlideFormat::OmeTiff => format.label().to_string(),
    }
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

    /// In right-to-left languages the summary's numbers are isolated, so
    /// that with their units (which the translations isolate with them)
    /// they keep their order: "25895 × 23396 px", not "px 23396 × 25895".
    #[test]
    fn the_summary_isolates_its_numbers_in_right_to_left_languages() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            struct English;
            impl Drop for English {
                fn drop(&mut self) {
                    crate::i18n::activate(eov_plugin_api::i18n::source());
                    DIALOG.with(|dialog| *dialog.borrow_mut() = Dialog::default());
                }
            }
            let _english = English;
            let dir = tempfile::tempdir().unwrap();
            let (state, id) = open_slide(dir.path());
            let ui = AppWindow::new().unwrap();
            for tag in ["ar", "fa", "he"] {
                crate::i18n::activate(eov_plugin_api::i18n::by_tag(tag).unwrap());
                open(&ui, &state, id);
                let summary = ui.get_save_copy_settings().summary;
                let mpp = crate::ui_update::localized_number("0.25");
                for number in ["1024", "768", mpp.as_str()] {
                    assert!(
                        summary.contains(&format!("\u{2068}{number}\u{2069}")),
                        "{tag}: {summary:?}"
                    );
                }
            }
            crate::i18n::activate(eov_plugin_api::i18n::source());
            open(&ui, &state, id);
            let summary = ui.get_save_copy_settings().summary;
            assert!(!summary.contains('\u{2068}'), "{summary:?}");
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

    /// The window says what the copy's report says (in English, word for
    /// word, but where the report's English has "1 tiles").
    #[test]
    fn shortfalls_and_omissions_are_reported_as_the_report_has_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tif");
        let spec = eov_common::test_support::TiffSpec::pyramid(512, 256, 256, 2);
        eov_common::test_support::write_tiff(&path, &spec, |x, y| [x as u8, y as u8, 9]).unwrap();
        let wsi = WsiFile::open(&path).unwrap();
        let options = ConvertOptions {
            tile_size: 64,
            ..ConvertOptions::default()
        };
        let plan = convert::plan_conversion(&wsi, &options).unwrap();
        let left_out = vec![
            CopyOmission::AssociatedImage {
                name: "macro".into(),
                error: "unreadable".into(),
            },
            CopyOmission::IccProfile {
                error: "truncated".into(),
            },
        ];
        let mut report = ConvertReport {
            path: dir.path().join("copy.tif"),
            plan,
            bytes: 0,
            bigtiff: false,
            unreadable_tiles: 0,
            associated_images: Vec::new(),
            icc_profile: false,
            warnings: left_out.iter().map(ToString::to_string).collect(),
            omissions: left_out,
            elapsed: Duration::ZERO,
        };
        assert_eq!(shortfall(&report), None);
        // The plural form follows the tiles that could not be read.
        let total = report.plan.tiles;
        assert!(total > 2);
        report.unreadable_tiles = 1;
        assert_eq!(
            shortfall(&report),
            Some(format!(
                "1 tile of {total} could not be read from the slide and shows its background \
                 colour."
            ))
        );
        report.unreadable_tiles = 2;
        assert_eq!(
            shortfall(&report),
            Some(format!(
                "2 tiles of {total} could not be read from the slide and show its background \
                 colour."
            ))
        );
        assert_eq!(omissions(&report), report.warnings);
        assert_eq!(format_name(SlideFormat::Tiff), SlideFormat::Tiff.label());
        assert_eq!(format_name(SlideFormat::OmeTiff), "OME-TIFF");
    }

    /// In English the dialog says why a copy cannot be made as
    /// `eov_common` does, word for word, as a sentence.
    #[test]
    fn convert_problems_say_what_common_says() {
        let problems = [
            ConvertProblem::EmptyRegion,
            ConvertProblem::RegionOutside {
                region: SlideBounds {
                    x: 700,
                    y: 0,
                    width: 10,
                    height: 10,
                },
                slide_width: 600,
                slide_height: 400,
            },
            ConvertProblem::TooLarge {
                width: 3_000_000_000,
                height: 10,
            },
            ConvertProblem::TooSmall {
                width: 8,
                height: 8,
            },
            ConvertProblem::NotAFileName("/a/..".into()),
            ConvertProblem::IsAFolder("/tmp".into()),
            ConvertProblem::NoSuchFolder("/nowhere".into()),
            ConvertProblem::IsTheSlide("/a/slide.svs".into()),
            ConvertProblem::PartialExists("/a/copy.tif.7.eov-partial".into()),
            ConvertProblem::NoPixelSize,
        ];
        for problem in problems {
            assert_eq!(convert_problem(&problem), sentence(&problem.to_string()));
            let error = ConvertError::Invalid(problem);
            assert_eq!(convert_error(&error), sentence(&error.to_string()));
        }
        let exists = ConvertError::OutputExists("/a/copy.tif".into());
        assert_eq!(convert_error(&exists), "/a/copy.tif already exists.");
        let io = ConvertError::Io(std::io::Error::other("disk full"));
        assert_eq!(convert_error(&io), "Disk full.");
        assert_eq!(
            convert_problem(&ConvertProblem::TooSmall {
                width: 8,
                height: 8
            }),
            "The copy would be 8 × 8 pixels; it must be at least 16 × 16 (choose a larger \
             region or a finer resolution)"
        );
        assert_eq!(
            convert_problem(&ConvertProblem::EmptyRegion),
            "The region is empty."
        );
    }

    /// The translated messages are shown as the translation writes them:
    /// eov neither changes their first letter nor adds a full stop.
    #[test]
    fn translated_problems_are_shown_as_translated() {
        use eov_plugin_api::i18n;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let _english = English;
        let turkish = i18n::by_tag("tr").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 1] =
            [(None, "The region is empty.", &["ilgi alanı boş"])];
        i18n::set_catalog(
            turkish,
            Some(i18n::Catalog::from_messages(turkish, "", messages).unwrap()),
        );
        assert_eq!(
            convert_error(&ConvertError::Invalid(ConvertProblem::EmptyRegion)),
            "ilgi alanı boş"
        );
        let japanese = i18n::by_tag("ja").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 1] =
            [(None, "The region is empty.", &["領域が空です"])];
        i18n::set_catalog(
            japanese,
            Some(i18n::Catalog::from_messages(japanese, "", messages).unwrap()),
        );
        assert_eq!(
            convert_problem(&ConvertProblem::EmptyRegion),
            "領域が空です"
        );
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
