//! `eov qc`: tissue quality control of many slides (AUDIT §14.5): the share
//! of each slide's tissue with pen marks, possible folds or out-of-focus
//! areas ([`eov_common::qc`]), as text or JSON, for screening a cohort.

use anyhow::{Result, bail};
use eov_common::WsiFile;
use eov_common::qc::{self, QcOptions, QcSummary};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

pub(crate) struct QcRun {
    pub inputs: Vec<PathBuf>,
    pub json: bool,
    pub sharpness: bool,
    pub cell_um: f64,
}

#[derive(Serialize)]
struct Row {
    slide: String,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    summary: Option<QcSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    seconds: f64,
}

fn percent(share: f32) -> String {
    format!("{:.1} %", share * 100.0)
}

pub(crate) fn run(run: &QcRun) -> Result<()> {
    let discovery = eov_common::dataset::discover_slides(&run.inputs);
    for (path, error) in &discovery.errors {
        eprintln!("{}: {error}", path.display());
    }
    if discovery.slides.is_empty() {
        bail!("no slides found in the inputs");
    }
    let options = QcOptions {
        cell_um: run.cell_um,
        sharpness: run.sharpness,
        ..QcOptions::default()
    };
    let cancel = AtomicBool::new(false);
    let mut rows = Vec::new();
    for slide in &discovery.slides {
        let started = Instant::now();
        let result = WsiFile::open(&slide.path)
            .map_err(|error| error.to_string())
            .and_then(|wsi| {
                qc::run(&wsi, &options, &|_| {}, &cancel).map_err(|error| error.to_string())
            });
        let (summary, error) = match result {
            Ok(Some(result)) => (Some(result.summary), None),
            Ok(None) => (None, Some("cancelled".to_string())),
            Err(error) => (None, Some(error)),
        };
        let row = Row {
            slide: slide.path.display().to_string(),
            summary,
            error,
            seconds: started.elapsed().as_secs_f64(),
        };
        if !run.json {
            print_row(&row);
        }
        rows.push(row);
    }
    if run.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    let failed = rows.iter().filter(|row| row.error.is_some()).count();
    if failed > 0 {
        bail!("{failed} of {} slide(s) could not be checked", rows.len());
    }
    Ok(())
}

fn print_row(row: &Row) {
    let name = std::path::Path::new(&row.slide)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| row.slide.clone());
    match (&row.summary, &row.error) {
        (Some(summary), _) => {
            let tissue = summary
                .tissue_mm2
                .map(|mm2| format!("{mm2:.1} mm²"))
                .unwrap_or_else(|| format!("{} cells", summary.tissue_cells));
            let pen = if summary.pen_skipped {
                "not checked (stain)".to_string()
            } else {
                percent(summary.pen)
            };
            let blur = summary
                .blur
                .map(percent)
                .unwrap_or_else(|| "not checked".to_string());
            println!(
                "{name}: tissue {tissue}, pen {pen}, possible folds {}, possibly out of focus {blur} ({:.1} s)",
                percent(summary.fold),
                row.seconds
            );
        }
        (None, Some(error)) => println!("{name}: {error}"),
        (None, None) => println!("{name}: no result"),
    }
}
