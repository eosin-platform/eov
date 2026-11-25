//! The dataset (patch extraction) dialog (DS-3, DS-4 and DS-6, GUI side).
//!
//! The dialog collects inputs and the basic grid; the run is validated
//! before the dialog closes, then runs on a worker with the shared
//! pipeline (`dataset::run_dataset`) as a status-bar job with progress,
//! counts of written, existing, skipped and failed patches, and cancel.

use crate::host_ui::with_host;
use crate::state::AppState;
use crate::{AppWindow, DatasetExportSettings};
use eov_common::dataset::{
    self, DatasetConfigError, DatasetControl, DatasetError, DatasetPatchesConfig,
    DatasetPatchesOptions, DatasetPatchesProgress, DatasetPatchesReport, DatasetPhase,
    TissueFilter,
};
use eov_plugin_api::i18n::{isolate, join_sentences, tr};
use eov_plugin_api::protocol::NotifyLevel;
use parking_lot::RwLock;
use slint::{Model, ModelRc, SharedString, Timer, TimerMode, VecModel};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

thread_local! {
    /// Progress timers of running extractions, by job id.
    static TIMERS: RefCell<HashMap<String, Timer>> = RefCell::new(HashMap::new());
}

fn inputs_of(settings: &DatasetExportSettings) -> Vec<SharedString> {
    (0..settings.inputs.row_count())
        .filter_map(|i| settings.inputs.row_data(i))
        .collect()
}

fn with_inputs(settings: &mut DatasetExportSettings, inputs: Vec<SharedString>) {
    settings.inputs = ModelRc::from(Rc::new(VecModel::from(inputs)));
}

/// The default output folder (DS-6: from the platform's documents folder,
/// not `$HOME`).
fn default_output_dir() -> PathBuf {
    eov_common::paths::documents_dir()
        .join("EOV")
        .join("Datasets")
}

/// Opens the dialog with every open slide as input.
pub(crate) fn open(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let inputs: Vec<SharedString> = state
        .read()
        .open_files
        .iter()
        .map(|file| SharedString::from(file.path.to_string_lossy().as_ref()))
        .collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let mut settings = DatasetExportSettings {
        output_dir: SharedString::from(default_output_dir().to_string_lossy().as_ref()),
        tile_size: 512,
        stride: 512,
        threads,
        // Same default as the command line: no white filter.
        white_threshold: 0.0,
        tissue_only: false,
        error: SharedString::new(),
        ..Default::default()
    };
    with_inputs(&mut settings, inputs);
    ui.set_dataset_export_settings(settings);
    ui.set_dataset_export_dialog_visible(true);
}

/// Adds slides chosen in a file dialog.
pub(crate) fn add_inputs(ui: &AppWindow) {
    let extensions = dataset::slide_extensions();
    let extensions: Vec<&str> = extensions.iter().map(String::as_str).collect();
    let Some(paths) = rfd::FileDialog::new()
        // Translators: file types in the dialog that adds slides to a
        // dataset.
        .add_filter(tr!("Slides"), &extensions)
        .add_filter(tr!("All files"), &["*"])
        .pick_files()
    else {
        return;
    };
    let mut settings = ui.get_dataset_export_settings();
    let mut inputs = inputs_of(&settings);
    for path in paths {
        let path = SharedString::from(path.to_string_lossy().as_ref());
        if !inputs.contains(&path) {
            inputs.push(path);
        }
    }
    with_inputs(&mut settings, inputs);
    settings.error = SharedString::new();
    ui.set_dataset_export_settings(settings);
}

pub(crate) fn remove_input(ui: &AppWindow, index: i32) {
    let mut settings = ui.get_dataset_export_settings();
    let mut inputs = inputs_of(&settings);
    if let Ok(index) = usize::try_from(index)
        && index < inputs.len()
    {
        inputs.remove(index);
        with_inputs(&mut settings, inputs);
        ui.set_dataset_export_settings(settings);
    }
}

pub(crate) fn choose_output_dir(ui: &AppWindow) {
    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
        let mut settings = ui.get_dataset_export_settings();
        settings.output_dir = SharedString::from(dir.to_string_lossy().as_ref());
        settings.error = SharedString::new();
        ui.set_dataset_export_settings(settings);
    }
}

/// The run described by the dialog.
fn run_spec(settings: &DatasetExportSettings) -> (DatasetPatchesConfig, DatasetPatchesOptions) {
    let config = DatasetPatchesConfig {
        inputs: inputs_of(settings)
            .iter()
            .map(|input| PathBuf::from(input.as_str()))
            .collect(),
        output_dir: PathBuf::from(settings.output_dir.as_str()),
        tile_size: settings.tile_size.max(0) as u32,
        stride: settings.stride.max(0) as u32,
        metadata_format: None,
        threads: settings.threads.max(0) as usize,
        white_threshold: (settings.white_threshold > 0.0).then_some(settings.white_threshold),
    };
    let options = DatasetPatchesOptions {
        tissue_filter: settings.tissue_only.then(TissueFilter::default),
        qc_exclusion: settings.exclude_qc.then(dataset::QcExclusion::all),
        deidentify: settings.deidentify,
        ..DatasetPatchesOptions::default()
    };
    (config, options)
}

/// Why the dialog's run cannot start ([`DatasetConfigError`]), in the
/// interface's language.
fn config_error(error: &DatasetConfigError) -> String {
    match error {
        // Translators: no slides were chosen for a dataset.
        DatasetConfigError::NoInputs => tr!("no input paths were given"),
        // Translators: the size of the dataset's patches is out of range.
        DatasetConfigError::TileSize { value, min, max } => tr!(
            "tile size must be between {min} and {max} pixels (got {value})",
            min = min,
            max = max,
            value = value
        ),
        // Translators: the step between the dataset's patches is out of
        // range.
        DatasetConfigError::Stride { value, min, max } => tr!(
            "stride must be between {min} and {max} pixels (got {value})",
            min = min,
            max = max,
            value = value
        ),
        // Translators: the number of threads (processor cores) to use is out
        // of range.
        DatasetConfigError::Threads { value, min, max } => tr!(
            "threads must be between {min} and {max} (got {value})",
            min = min,
            max = max,
            value = value
        ),
        // Translators: {} is the number given.
        DatasetConfigError::WhiteThreshold(value) => tr!(
            "white threshold must be above 0 and at most 1 (got {})",
            crate::ui_update::localized_number(&value.to_string())
        ),
        // Settings the dialog does not offer (the command line does): as
        // eov_common words them.
        DatasetConfigError::TissueFraction(_)
        | DatasetConfigError::TissueThreshold(_)
        | DatasetConfigError::Resolution(_)
        | DatasetConfigError::MaxTiles
        | DatasetConfigError::JpegQuality(_)
        | DatasetConfigError::Coverage(_)
        | DatasetConfigError::MaxCandidates => error.to_string(),
    }
}

/// Writes the open dialog's error again, in the language of the interface
/// (after it changed).
pub(crate) fn refresh_texts(ui: &AppWindow) {
    if !ui.get_dataset_export_dialog_visible() {
        return;
    }
    let mut settings = ui.get_dataset_export_settings();
    if settings.error.is_empty() {
        return;
    }
    let (config, options) = run_spec(&settings);
    settings.error = config
        .validate(&options)
        .err()
        .map(|error| SharedString::from(config_error(&error)))
        .unwrap_or_default();
    ui.set_dataset_export_settings(settings);
}

/// Validates the dialog and starts the run as a status-bar job.
pub(crate) fn start(ui: &AppWindow) {
    let mut settings = ui.get_dataset_export_settings();
    let (config, options) = run_spec(&settings);
    // Pre-flight (DS-4): refuse bad parameters while the dialog is open.
    if let Err(err) = config.validate(&options) {
        settings.error = SharedString::from(config_error(&err));
        ui.set_dataset_export_settings(settings);
        return;
    }
    ui.set_dataset_export_dialog_visible(false);

    let control = Arc::new(DatasetControl::new());
    let cancel = {
        let control = Arc::clone(&control);
        Box::new(move || control.cancel()) as Box<dyn Fn()>
    };
    let Some(job_id) =
        with_host(|host| host.start_host_job(&tr!("Extracting dataset patches"), Some(cancel)))
    else {
        return;
    };
    let started = Instant::now();
    let timer = Timer::default();
    {
        let control = Arc::clone(&control);
        let job_id = job_id.clone();
        timer.start(TimerMode::Repeated, PROGRESS_INTERVAL, move || {
            let progress = control.progress();
            let fraction = (progress.tiles_total > 0)
                .then(|| (progress.tiles_done as f64 / progress.tiles_total as f64) as f32);
            let message = progress_message(&progress, started.elapsed());
            with_host(|host| host.update_host_job(&job_id, fraction, Some(message)));
        });
    }
    TIMERS.with(|timers| timers.borrow_mut().insert(job_id.clone(), timer));

    let output_dir = config.output_dir.clone();
    let spawned = std::thread::Builder::new()
        .name("eov-dataset".to_string())
        .spawn({
            let job_id = job_id.clone();
            let control = Arc::clone(&control);
            move || {
                let result = dataset::run_dataset(&config, &options, &control);
                let cancelled = control.is_cancelled();
                let _ = slint::invoke_from_event_loop(move || {
                    finished(job_id, output_dir, cancelled, result)
                });
            }
        });
    if spawned.is_err() {
        TIMERS.with(|timers| timers.borrow_mut().remove(&job_id));
        with_host(|host| {
            host.finish_host_job(&job_id);
            host.notify(
                NotifyLevel::Error,
                &tr!("Could not start the dataset extraction"),
                None,
                None,
            );
        });
    }
}

/// The job line in the status bar.
fn progress_message(progress: &DatasetPatchesProgress, elapsed: Duration) -> String {
    let (current, total) = (progress.current_slide.max(1), progress.total_slides);
    let phase = match progress.phase {
        DatasetPhase::Discovering => return tr!("Finding slides…"),
        DatasetPhase::Planning => return tr!("Planning…"),
        // Translators: the dataset extraction is computing the identity of
        // slide {current} of {total}.
        DatasetPhase::Hashing => tr!(
            "Hashing {current}/{total}",
            current = current,
            total = total
        ),
        DatasetPhase::Extracting => {
            // Translators: the dataset extraction is at slide {current} of
            // {total}.
            tr!("Slide {current}/{total}", current = current, total = total)
        }
        DatasetPhase::Finishing | DatasetPhase::Done => return tr!("Finishing…"),
    };
    let mut parts = vec![phase];
    // Translators: patches written so far (as are "kept", "skipped" and
    // "failed" below).
    parts.push(tr!("{n} written" | "{n} written" % progress.tiles_written));
    if progress.tiles_existing > 0 {
        // Translators: patches already there from an earlier run.
        parts.push(tr!("{n} kept" | "{n} kept" % progress.tiles_existing));
    }
    if progress.tiles_skipped > 0 {
        parts.push(tr!("{n} skipped" | "{n} skipped" % progress.tiles_skipped));
    }
    if progress.tiles_failed > 0 {
        parts.push(tr!("{n} failed" | "{n} failed" % progress.tiles_failed));
    }
    if let Some(eta) = remaining(progress, elapsed) {
        // Translators: the time the extraction still needs ("4m 05s left").
        parts.push(tr!("{} left", format_duration(eta)));
    }
    parts.join(" · ")
}

/// Estimated time left from the rate so far.
fn remaining(progress: &DatasetPatchesProgress, elapsed: Duration) -> Option<Duration> {
    if progress.tiles_done == 0 || progress.tiles_total <= progress.tiles_done {
        return None;
    }
    let rate = progress.tiles_done as f64 / elapsed.as_secs_f64().max(1e-3);
    let left = (progress.tiles_total - progress.tiles_done) as f64 / rate;
    Some(Duration::from_secs_f64(left))
}

/// `"1h 02m"`, `"4m 05s"`, `"12s"` (never `"1m60s"`, DS-6), in the
/// interface's language.
fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs_f64().round() as u64;
    let (hours, minutes, seconds) = (total / 3600, (total / 60) % 60, total % 60);
    if hours > 0 {
        // Translators: hours and minutes ("1h 02m").
        tr!(
            "{hours}h {minutes}m",
            hours = hours,
            minutes = format!("{minutes:02}")
        )
    } else if minutes > 0 {
        // Translators: minutes and seconds ("4m 05s").
        tr!(
            "{minutes}m {seconds}s",
            minutes = minutes,
            seconds = format!("{seconds:02}")
        )
    } else {
        // Translators: seconds ("12s").
        tr!("{seconds}s", seconds = seconds)
    }
}

fn finished(
    job_id: String,
    output_dir: PathBuf,
    cancelled: bool,
    result: Result<DatasetPatchesReport, DatasetError>,
) {
    TIMERS.with(|timers| timers.borrow_mut().remove(&job_id));
    with_host(|host| {
        host.finish_host_job(&job_id);
        match result {
            Ok(report) if cancelled => host.notify(
                NotifyLevel::Info,
                &tr!("Dataset extraction cancelled"),
                Some(&tr!(
                    "{n} patch was written to {folder}; run it again to resume."
                        | "{n} patches were written to {folder}; run it again to resume."
                            % report.total_tiles,
                    folder = isolate(output_dir.display())
                )),
                None,
            ),
            Ok(report) => {
                let level = if report.failed_slides > 0 || report.total_tiles_failed > 0 {
                    NotifyLevel::Warning
                } else {
                    NotifyLevel::Success
                };
                host.notify(
                    level,
                    &tr!("Dataset extraction finished"),
                    Some(&summary(&report, &output_dir)),
                    None,
                );
            }
            Err(err) => host.notify(
                NotifyLevel::Error,
                &tr!("Dataset extraction failed"),
                Some(&err.to_string()),
                None,
            ),
        }
    });
}

fn summary(report: &DatasetPatchesReport, output_dir: &std::path::Path) -> String {
    let mut lines = vec![
        tr!(
            "{n} patch written to {folder}."
                | "{n} patches written to {folder}." % report.total_tiles,
            folder = isolate(output_dir.display())
        ),
        tr!("{n} slide processed." | "{n} slides processed." % report.processed_slides),
    ];
    let skipped = report.total_tiles_skipped_white
        + report.total_tiles_skipped_tissue
        + report.total_tiles_skipped_annotation
        + report.total_tiles_skipped_qc;
    if skipped > 0 && report.total_tiles_skipped_qc >= skipped {
        // Translators: tissue QC flagged every patch the filters skipped
        // (pen marks, possible folds or possible blur: flags for review, not
        // certainties).
        lines.push(tr!(
            "{n} patch skipped by the filters: tissue QC flagged it (pen marks, possible folds or blur)."
                | "{n} patches skipped by the filters: tissue QC flagged them (pen marks, possible folds or blur)."
                    % skipped
        ));
    } else if skipped > 0 {
        lines
            .push(tr!("{n} patch skipped by the filters."
                | "{n} patches skipped by the filters." % skipped));
        if report.total_tiles_skipped_qc > 0 {
            // Translators: follows the sentence about the patches skipped
            // by the filters (always more than one here): how many of them
            // tissue QC flagged (flags for review, not certainties).
            lines.push(tr!(
                "{n} of them flagged by tissue QC (pen marks, possible folds or blur)."
                    | "{n} of them flagged by tissue QC (pen marks, possible folds or blur)."
                        % report.total_tiles_skipped_qc
            ));
        }
    }
    if report.total_tiles_existing > 0 {
        lines.push(tr!("{n} patch was already there from an earlier run."
            | "{n} patches were already there from an earlier run."
                % report.total_tiles_existing));
    }
    if report.total_tiles_failed > 0 {
        lines.push(tr!("{n} patch could not be read."
            | "{n} patches could not be read." % report.total_tiles_failed));
    }
    // The slides skipped include the ones that failed.
    if report.skipped_slides > 0 && report.failed_slides >= report.skipped_slides {
        // Translators: every slide that was skipped failed. slides.csv is
        // the name of a file in the dataset; keep it as it is.
        lines.push(tr!("{n} slide failed (see slides.csv)."
            | "{n} slides failed (see slides.csv)." % report.failed_slides));
    } else if report.skipped_slides > 0 {
        // Translators: slides.csv is the name of a file in the dataset;
        // keep it as it is.
        lines.push(tr!("{n} slide was skipped (see slides.csv)."
            | "{n} slides were skipped (see slides.csv)."
                % report.skipped_slides));
        if report.failed_slides > 0 {
            // Translators: follows the sentence about the slides that were
            // skipped (always more than one here): how many of them failed.
            lines.push(tr!(
                "{n} of them failed." | "{n} of them failed." % report.failed_slides
            ));
        }
    }
    join_sentences(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In English the dialog says why a run cannot start as `eov_common`
    /// does, word for word.
    #[test]
    fn config_errors_say_what_common_says() {
        for error in [
            DatasetConfigError::NoInputs,
            DatasetConfigError::TileSize {
                value: 8,
                min: 16,
                max: 8192,
            },
            DatasetConfigError::Stride {
                value: 0,
                min: 1,
                max: 8192,
            },
            DatasetConfigError::Threads {
                value: 0,
                min: 1,
                max: 256,
            },
            DatasetConfigError::WhiteThreshold(1.5),
            DatasetConfigError::MaxTiles,
        ] {
            assert_eq!(config_error(&error), error.to_string());
        }
    }

    /// Each count of the summary has a sentence of its own, with its own
    /// plural; the sentences follow one another as the language writes
    /// them.
    #[test]
    fn the_summary_gives_each_count_a_sentence() {
        use eov_plugin_api::i18n;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let mut report = DatasetPatchesReport {
            input_count: 4,
            discovered_slides: 4,
            processed_slides: 1,
            skipped_slides: 3,
            failed_slides: 1,
            total_tiles: 120,
            total_tiles_skipped_white: 30,
            total_tiles_skipped_tissue: 0,
            total_tiles_skipped_annotation: 0,
            total_tiles_skipped_qc: 1,
            total_tiles_existing: 0,
            total_tiles_failed: 0,
            planned_tiles: 151,
            metadata_paths: Vec::new(),
            manifest_path: None,
            slides: Vec::new(),
            input_errors: Vec::new(),
            skipped_inputs: Vec::new(),
            cancelled: false,
            dry_run: false,
        };
        let folder = std::path::Path::new("/data/patches");
        assert_eq!(
            summary(&report, folder),
            "120 patches written to /data/patches. 1 slide processed. 31 patches skipped by \
             the filters. 1 of them flagged by tissue QC (pen marks, possible folds or blur). \
             3 slides were skipped (see slides.csv). 1 of them failed."
        );
        report.skipped_slides = 1;
        report.failed_slides = 0;
        report.total_tiles_skipped_white = 0;
        report.total_tiles_skipped_qc = 0;
        assert_eq!(
            summary(&report, folder),
            "120 patches written to /data/patches. 1 slide processed. 1 slide was skipped \
             (see slides.csv)."
        );
        // "Of them" is never about one: when every slide skipped failed
        // (or tissue QC flagged every patch skipped), one sentence says so.
        report.failed_slides = 1;
        report.total_tiles_skipped_qc = 1;
        assert_eq!(
            summary(&report, folder),
            "120 patches written to /data/patches. 1 slide processed. 1 patch skipped by the \
             filters: tissue QC flagged it (pen marks, possible folds or blur). 1 slide failed \
             (see slides.csv)."
        );
        report.skipped_slides = 2;
        report.failed_slides = 2;
        assert_eq!(
            summary(&report, folder),
            "120 patches written to /data/patches. 1 slide processed. 1 patch skipped by the \
             filters: tissue QC flagged it (pen marks, possible folds or blur). 2 slides failed \
             (see slides.csv)."
        );
        report.skipped_slides = 1;
        report.failed_slides = 0;
        report.total_tiles_skipped_qc = 0;

        let _english = English;
        let chinese = i18n::by_tag("zh-Hans").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 3] = [
            (
                None,
                "{n} patch written to {folder}.",
                &["已将 {n} 个图块写入 {folder}。"],
            ),
            (None, "{n} slide processed.", &["已处理 {n} 张切片。"]),
            (
                None,
                "{n} slide was skipped (see slides.csv).",
                &["跳过了 {n} 张切片（见 slides.csv）。"],
            ),
        ];
        i18n::set_catalog(
            chinese,
            Some(i18n::Catalog::from_messages(chinese, "", messages).unwrap()),
        );
        assert_eq!(
            summary(&report, folder),
            "已将 120 个图块写入 /data/patches。已处理 1 张切片。跳过了 1 张切片（见 slides.csv）。"
        );
    }

    #[test]
    fn durations_never_show_sixty_seconds() {
        assert_eq!(format_duration(Duration::from_secs_f64(119.6)), "2m 00s");
        assert_eq!(format_duration(Duration::from_secs(59)), "59s");
        assert_eq!(format_duration(Duration::from_secs(3725)), "1h 02m");
    }

    #[test]
    fn progress_lines_show_each_count() {
        let progress = DatasetPatchesProgress {
            phase: DatasetPhase::Extracting,
            current_slide: 2,
            total_slides: 3,
            tiles_done: 50,
            tiles_total: 100,
            tiles_written: 40,
            tiles_existing: 5,
            tiles_skipped: 4,
            tiles_failed: 1,
        };
        let line = progress_message(&progress, Duration::from_secs(10));
        assert_eq!(
            line,
            "Slide 2/3 · 40 written · 5 kept · 4 skipped · 1 failed · 10s left"
        );
    }
}
