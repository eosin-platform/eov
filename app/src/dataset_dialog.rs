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
    self, DatasetControl, DatasetError, DatasetPatchesConfig, DatasetPatchesOptions,
    DatasetPatchesProgress, DatasetPatchesReport, DatasetPhase, TissueFilter,
};
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
    let extensions: Vec<&str> = dataset::SUPPORTED_SLIDE_EXTENSIONS.to_vec();
    let Some(paths) = rfd::FileDialog::new()
        .add_filter("Slides", &extensions)
        .add_filter("All files", &["*"])
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
        ..DatasetPatchesOptions::default()
    };
    (config, options)
}

/// Validates the dialog and starts the run as a status-bar job.
pub(crate) fn start(ui: &AppWindow) {
    let mut settings = ui.get_dataset_export_settings();
    let (config, options) = run_spec(&settings);
    // Pre-flight (DS-4): refuse bad parameters while the dialog is open.
    if let Err(err) = config.validate(&options) {
        settings.error = SharedString::from(err.to_string());
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
        with_host(|host| host.start_host_job("Extracting dataset patches", Some(cancel)))
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
                "Could not start the dataset extraction",
                None,
                None,
            );
        });
    }
}

/// The job line in the status bar.
fn progress_message(progress: &DatasetPatchesProgress, elapsed: Duration) -> String {
    let phase = match progress.phase {
        DatasetPhase::Discovering => return "Finding slides…".to_string(),
        DatasetPhase::Planning => return "Planning…".to_string(),
        DatasetPhase::Hashing => "Hashing",
        DatasetPhase::Extracting => "Slide",
        DatasetPhase::Finishing | DatasetPhase::Done => return "Finishing…".to_string(),
    };
    let mut parts = vec![format!(
        "{phase} {}/{}",
        progress.current_slide.max(1),
        progress.total_slides
    )];
    parts.push(format!("{} written", progress.tiles_written));
    if progress.tiles_existing > 0 {
        parts.push(format!("{} kept", progress.tiles_existing));
    }
    if progress.tiles_skipped > 0 {
        parts.push(format!("{} skipped", progress.tiles_skipped));
    }
    if progress.tiles_failed > 0 {
        parts.push(format!("{} failed", progress.tiles_failed));
    }
    if let Some(eta) = remaining(progress, elapsed) {
        parts.push(format!("{} left", format_duration(eta)));
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

/// `"1h 02m"`, `"4m 05s"`, `"12s"` (never `"1m60s"`, DS-6).
fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs_f64().round() as u64;
    let (hours, minutes, seconds) = (total / 3600, (total / 60) % 60, total % 60);
    if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
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
                "Dataset extraction cancelled",
                Some(&format!(
                    "{} patches were written to {}; run it again to resume.",
                    report.total_tiles,
                    output_dir.display()
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
                    "Dataset extraction finished",
                    Some(&summary(&report, &output_dir)),
                    None,
                );
            }
            Err(err) => host.notify(
                NotifyLevel::Error,
                "Dataset extraction failed",
                Some(&err.to_string()),
                None,
            ),
        }
    });
}

fn summary(report: &DatasetPatchesReport, output_dir: &std::path::Path) -> String {
    let mut lines = vec![format!(
        "{} patches from {} slide(s) in {}.",
        report.total_tiles,
        report.processed_slides,
        output_dir.display()
    )];
    let skipped = report.total_tiles_skipped_white
        + report.total_tiles_skipped_tissue
        + report.total_tiles_skipped_annotation;
    if skipped > 0 {
        lines.push(format!("{skipped} patches skipped by the filters."));
    }
    if report.total_tiles_existing > 0 {
        lines.push(format!(
            "{} patches were already there from an earlier run.",
            report.total_tiles_existing
        ));
    }
    if report.total_tiles_failed > 0 {
        lines.push(format!(
            "{} patches could not be read.",
            report.total_tiles_failed
        ));
    }
    if report.skipped_slides > 0 || report.failed_slides > 0 {
        lines.push(format!(
            "{} slide(s) skipped, {} failed (see slides.csv).",
            report.skipped_slides, report.failed_slides
        ));
    }
    lines.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

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
