//! Analysis: `quantify` (the Quantify Stain panel's measurement) and
//! `export.view` (the export dialog's Export). Both capture what they need
//! on the UI thread, then work on the connection's thread as a status-bar
//! job the user can cancel, never on the UI thread.

use super::params;
use super::{CallResult, ErrorKind, RpcError, Session};
use crate::export_dialog::{ScriptedExport, ScriptedOptions};
use crate::quantify_panel::{self, Quantification};
use eov_common::export::{ExportError, ExportFormat, TiffCompression};
use eov_common::job::JobControl;
use eov_common::stain::StainPreset;
use eov_plugin_api::protocol::Rect;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Largest image `export.view` writes, in pixels.
pub(crate) const MAX_EXPORT_PIXELS: u64 = 1 << 30;
/// Largest width or height `export.view` writes (JPEG's limit).
pub(crate) const MAX_EXPORT_SIDE: u32 = 65_535;
/// Highest resolution `export.view` renders at.
pub(crate) const MAX_EXPORT_DPI: u32 = 2400;

/// A stain: its index in the model, or its name (`"DAB"`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum StainParam {
    Index(usize),
    Name(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Quantify {
    #[serde(default)]
    slide: Option<i64>,
    /// A stain model (`h-dab`, `he`, `h-e-dab`, `h-aec`, `h-pas`,
    /// `masson-trichrome`); default: suggested from the slide's stain label.
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stain: Option<StainParam>,
    /// Weak, moderate and strong concentrations.
    #[serde(default)]
    thresholds: Option<[f32; 3]>,
    /// The level-0 region; default: the slide's region of interest.
    #[serde(default)]
    region: Option<Rect>,
}

/// The model and stain a script asks for.
fn model_and_stain(
    model: Option<&str>,
    stain: Option<&StainParam>,
    stain_label: Option<&str>,
) -> Result<(StainPreset, usize), RpcError> {
    let preset = match model {
        Some(id) => StainPreset::from_id(id)
            .filter(|preset| quantify_panel::PRESETS.contains(preset))
            .ok_or_else(|| {
                let ids: Vec<&str> = quantify_panel::PRESETS.iter().map(|p| p.id()).collect();
                RpcError::invalid(format!(
                    "{id:?} is not a stain model to quantify with ({})",
                    ids.join(", ")
                ))
            })?,
        None => quantify_panel::default_preset(stain_label),
    };
    let basis = preset.basis();
    let names = basis.names();
    let stain = match stain {
        None => quantify_panel::default_stain(preset),
        Some(StainParam::Index(index)) if *index < names.len() => *index,
        Some(StainParam::Name(name)) => names
            .iter()
            .position(|known| known.eq_ignore_ascii_case(name.trim()))
            .ok_or_else(|| {
                RpcError::invalid(format!(
                    "the {} model has the stains {}, not {name:?}",
                    preset.id(),
                    names.join(", ")
                ))
            })?,
        Some(StainParam::Index(index)) => {
            return Err(RpcError::invalid(format!(
                "the {} model has stains 0 to {}, not {index}",
                preset.id(),
                names.len() - 1
            )));
        }
    };
    Ok((preset, stain))
}

pub(super) fn quantify(session: &Session, params: Quantify) -> CallResult {
    let thresholds = quantify_panel::check_thresholds(
        params
            .thresholds
            .unwrap_or(quantify_panel::DEFAULT_THRESHOLDS),
    )
    .map_err(RpcError::invalid)?;
    let region = params.region.map(params::rect).transpose()?;
    let Quantify {
        slide,
        model,
        stain,
        ..
    } = params;
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&cancel);
    let (job, file_id, job_id) = session.on_ui(move |host| {
        let job = {
            let state = host.state.read();
            let file_id = params::slide(&state, slide)?;
            let file = state
                .get_file(file_id)
                .ok_or_else(|| RpcError::not_found("the slide was closed"))?;
            let region = region.or(file.roi).ok_or_else(|| {
                RpcError::invalid(
                    "the slide has no region of interest: draw one (R), set one with roi.set, \
                     or pass region",
                )
            })?;
            let (preset, stain) = model_and_stain(
                model.as_deref(),
                stain.as_ref(),
                file.wsi.properties().stain.as_deref(),
            )?;
            (
                Quantification::of_file(file, region, preset, stain, thresholds),
                file_id,
            )
        };
        let job_id = host.start_host_job(
            "Quantifying the stain (script)",
            Some(Box::new(move || stop.store(true, Ordering::Relaxed))),
        );
        Ok((job.0, job.1, job_id))
    })?;
    let result = {
        let _permit = super::HEAVY.acquire();
        job.run(&cancel)
    };
    session.post_ui(move |host| host.finish_host_job(&job_id));
    let quantification = match result {
        Ok(Some(quantification)) => quantification,
        Ok(None) => {
            return Err(RpcError::new(
                ErrorKind::Cancelled,
                "the quantification was cancelled",
            ));
        }
        Err(error) => return Err(RpcError::failed(format!("cannot quantify: {error}"))),
    };
    let (level, downsample) = job.level();
    let fraction = |n: u64| {
        if quantification.analysed_pixels == 0 {
            0.0
        } else {
            n as f64 / quantification.analysed_pixels as f64
        }
    };
    let basis = job.preset.basis();
    Ok(json!({
        "file_id": file_id,
        "model": job.preset.id(),
        "model_label": job.preset.label(),
        "stain": job.stain,
        "stain_name": basis.names().get(job.stain),
        "thresholds": job.thresholds,
        "region": job.region,
        "level": level,
        "downsample": downsample,
        "analysed_pixels": quantification.analysed_pixels,
        "positive_pixels": quantification.positive_pixels,
        "positive_fraction": quantification.positive_fraction,
        "bins": quantification.bins,
        "weak_fraction": fraction(quantification.bins[0]),
        "moderate_fraction": fraction(quantification.bins[1]),
        "strong_fraction": fraction(quantification.bins[2]),
        "h_score": quantification.h_score,
        "mean_concentration": quantification.mean_concentration,
    }))
}

/// An image format for `export.view`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum FormatParam {
    Png,
    #[serde(alias = "jpg")]
    Jpeg,
    #[serde(alias = "tif")]
    Tiff,
}

impl FormatParam {
    fn of_extension(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "tif" | "tiff" => Some(Self::Tiff),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Tiff => "tiff",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExportView {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    os_path: Option<String>,
    /// The pane whose view to export; default: the focused one.
    #[serde(default)]
    pane: Option<u32>,
    /// Default: from the file's extension.
    #[serde(default)]
    format: Option<FormatParam>,
    /// Default: 150 (the dialog's); 96 is one pixel per screen pixel.
    #[serde(default)]
    dpi: Option<u32>,
    /// JPEG quality, 1 to 100 (default 92).
    #[serde(default)]
    jpeg_quality: Option<u8>,
    #[serde(default)]
    scale_bar: Option<bool>,
    #[serde(default)]
    roi: Option<bool>,
    #[serde(default)]
    measurements: Option<bool>,
    #[serde(default)]
    processing_note: Option<bool>,
    /// Replace an existing file.
    #[serde(default)]
    overwrite: bool,
}

/// The format of an export to `path`.
fn export_format(
    path: &Path,
    format: Option<FormatParam>,
    jpeg_quality: Option<u8>,
) -> Result<ExportFormat, RpcError> {
    let by_extension = FormatParam::of_extension(path);
    let format = match (format, by_extension) {
        (Some(format), Some(extension)) if format != extension => {
            return Err(RpcError::invalid(format!(
                "the file's extension does not fit the format {}",
                format.name()
            )));
        }
        (Some(format), _) | (None, Some(format)) => format,
        (None, None) => {
            return Err(RpcError::invalid(
                "name a .png, .jpg or .tif file, or give the format",
            ));
        }
    };
    if jpeg_quality.is_some() && format != FormatParam::Jpeg {
        return Err(RpcError::invalid("jpeg_quality is for JPEG exports"));
    }
    Ok(match format {
        FormatParam::Png => ExportFormat::Png,
        FormatParam::Jpeg => {
            let quality = jpeg_quality.unwrap_or(92);
            if !(1..=100).contains(&quality) {
                return Err(RpcError::invalid("jpeg_quality is from 1 to 100"));
            }
            ExportFormat::Jpeg { quality }
        }
        FormatParam::Tiff => ExportFormat::Tiff {
            compression: TiffCompression::Deflate,
        },
    })
}

pub(super) fn export_view(session: &Session, params: ExportView) -> CallResult {
    let path = params::path_from(params.path, params.os_path)?
        .ok_or_else(|| RpcError::invalid("give the path of the image to write"))?;
    let format = export_format(&path, params.format, params.jpeg_quality)?;
    let path = params::output_file(&path, params.overwrite)?;
    if let Some(dpi) = params.dpi
        && !(1..=MAX_EXPORT_DPI).contains(&dpi)
    {
        return Err(RpcError::invalid(format!(
            "dpi is from 1 to {MAX_EXPORT_DPI}"
        )));
    }
    let options = ScriptedOptions {
        format,
        dpi: params.dpi,
        scale_bar: params.scale_bar,
        roi: params.roi,
        measurements: params.measurements,
        processing_note: params.processing_note,
    };
    let pane = params.pane;
    let control = Arc::new(JobControl::new());
    let stop = Arc::clone(&control);
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (job, job_id) = session.on_ui(move |host| {
        let job = {
            let state = host.state.read();
            let pane = params::pane(&state, pane)?;
            ScriptedExport::capture(&state, &host.tile_cache, pane, &options).ok_or_else(|| {
                RpcError::not_found(format!(
                    "pane {} shows no slide",
                    state.pane_uid(pane).unwrap_or_default()
                ))
            })?
        };
        let (width, height) = job.output_size();
        if width > MAX_EXPORT_SIDE
            || height > MAX_EXPORT_SIDE
            || u64::from(width) * u64::from(height) > MAX_EXPORT_PIXELS
        {
            return Err(RpcError::too_large(format!(
                "the image would be {width} × {height} pixels; lower dpi"
            )));
        }
        let job_id = host.start_host_job(
            &format!("Exporting {file_name} (script)"),
            Some(Box::new(move || stop.cancel())),
        );
        Ok((job, job_id))
    })?;

    // Progress in the status bar while the export renders here.
    let done = Arc::new(AtomicBool::new(false));
    let progress = {
        let (done, control, job_id) = (Arc::clone(&done), Arc::clone(&control), job_id.clone());
        let ui = session.service.ui.clone();
        std::thread::Builder::new()
            .name("eov-script-export-progress".to_string())
            .spawn(move || {
                while !done.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(250));
                    let progress = control.progress();
                    let job_id = job_id.clone();
                    ui(Box::new(move || {
                        crate::host_ui::with_host(|host| {
                            host.update_host_job(
                                &job_id,
                                (progress.total > 0).then(|| progress.fraction()),
                                None,
                            )
                        });
                    }));
                }
            })
            .ok()
    };
    let result = {
        let _permit = super::HEAVY.acquire();
        job.run(&path, &control)
    };
    done.store(true, Ordering::Release);
    if let Some(progress) = progress {
        let _ = progress.join();
    }
    session.post_ui(move |host| host.finish_host_job(&job_id));
    let report = match result {
        Ok(report) => report,
        Err(ExportError::Cancelled) => {
            return Err(RpcError::new(
                ErrorKind::Cancelled,
                "the export was cancelled",
            ));
        }
        Err(error) => return Err(RpcError::failed(format!("cannot export: {error}"))),
    };
    let format_name = match format {
        ExportFormat::Png => "png",
        ExportFormat::Jpeg { .. } => "jpeg",
        ExportFormat::Tiff { .. } => "tiff",
    };
    Ok(json!({
        "path": path.to_string_lossy(),
        "file_id": job.file_id(),
        "slide": job.file_name(),
        "format": format_name,
        "width": report.width,
        "height": report.height,
        "level": report.level,
        "tiles_failed": report.tiles_failed,
        "complete": report.is_complete(),
        "warning": report.shortfall_message(),
        "elapsed_ms": report.elapsed.as_millis() as u64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_and_stains_are_named_or_numbered() {
        assert_eq!(
            model_and_stain(None, None, Some("Ki-67 IHC")).unwrap(),
            (StainPreset::HematoxylinDab, 1)
        );
        assert_eq!(
            model_and_stain(Some("he"), None, None).unwrap(),
            (StainPreset::HematoxylinEosin, 1)
        );
        assert_eq!(
            model_and_stain(Some("h-e-dab"), Some(&StainParam::Name("dab".into())), None).unwrap(),
            (StainPreset::HematoxylinEosinDab, 2)
        );
        assert_eq!(
            model_and_stain(Some("h-dab"), Some(&StainParam::Index(0)), None).unwrap(),
            (StainPreset::HematoxylinDab, 0)
        );
        // The Macenko reference normalises; it does not quantify.
        assert!(model_and_stain(Some("he-macenko"), None, None).is_err());
        assert!(model_and_stain(Some("h-dab"), Some(&StainParam::Index(2)), None).is_err());
        assert!(
            model_and_stain(Some("h-dab"), Some(&StainParam::Name("eosin".into())), None).is_err()
        );
    }

    #[test]
    fn export_formats_follow_the_extension() {
        let png = Path::new("/out/figure.png");
        assert_eq!(export_format(png, None, None).unwrap(), ExportFormat::Png);
        assert_eq!(
            export_format(Path::new("/out/figure.JPG"), None, Some(80)).unwrap(),
            ExportFormat::Jpeg { quality: 80 }
        );
        assert!(matches!(
            export_format(Path::new("/out/figure"), Some(FormatParam::Tiff), None).unwrap(),
            ExportFormat::Tiff { .. }
        ));
        assert!(export_format(png, Some(FormatParam::Jpeg), None).is_err());
        assert!(export_format(Path::new("/out/figure.bmp"), None, None).is_err());
        assert!(export_format(png, None, Some(90)).is_err());
        assert!(export_format(Path::new("/out/a.jpg"), None, Some(0)).is_err());
    }
}
