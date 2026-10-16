//! The Quantify Stain panel (AUDIT §14.5): positive-pixel analysis and an
//! H-score of one stain in a slide's region of interest, for brightfield
//! IHC (DAB, AEC) and special stains, with the same colour-deconvolution
//! model as the display ([`eov_common::stain::quantify_stain`]).
//!
//! The region is read at the finest level that keeps it within
//! [`MAX_PIXELS`], on a worker, as a status-bar job that can be cancelled.
//! Its pixels are the slide's own colours, as scanned: neither the panes'
//! adjustments nor the colour management (ICC profile to sRGB) apply to
//! what is measured. Scripts' `quantify` ([`crate::scripting`]) measures
//! the same way, with [`Quantification`].

use crate::state::{AppState, OpenFile};
use crate::{AppWindow, ChoiceItem, ResultRow};
use eov_common::stain::od::estimate_background;
use eov_common::stain::{
    BackgroundOptions, PixelFilter, StainPreset, StainQuantification, quantify_stain,
    suggested_preset,
};
use eov_plugin_api::protocol::{NotifyLevel, Rect};
use parking_lot::RwLock;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest region read, in pixels of the chosen level.
const MAX_PIXELS: u64 = 16_000_000;

/// Largest region read at all, when even the coarsest level holds more
/// than [`MAX_PIXELS`] (256 MB of RGBA).
const MAX_READ_PIXELS: u64 = 4 * MAX_PIXELS;

/// Stain models offered (the Macenko H&E reference is a normalisation
/// target, not a model to measure with).
pub(crate) const PRESETS: [StainPreset; 6] = [
    StainPreset::HematoxylinDab,
    StainPreset::HematoxylinEosin,
    StainPreset::HematoxylinEosinDab,
    StainPreset::HematoxylinAec,
    StainPreset::HematoxylinPas,
    StainPreset::MassonTrichrome,
];

pub(crate) const DEFAULT_THRESHOLDS: [f32; 3] = [0.15, 0.5, 1.0];

/// The model suggested for a slide's stain label, H-DAB when it names
/// none of [`PRESETS`].
pub(crate) fn default_preset(stain_label: Option<&str>) -> StainPreset {
    suggested_preset(stain_label)
        .filter(|preset| PRESETS.contains(preset))
        .unwrap_or(StainPreset::HematoxylinDab)
}

/// Whether a slide's overview shows DAB brown (an IHC slide): brown pixels
/// make up at least 2 % of its tissue. Used when the slide's metadata names
/// no stain.
pub(crate) fn shows_dab(rgba: &[u8]) -> bool {
    let (mut tissue, mut brown) = (0u64, 0u64);
    for p in rgba.as_chunks::<4>().0 {
        if p[3] < 255 {
            continue;
        }
        let [r, g, b] = [p[0], p[1], p[2]].map(|c| f32::from(c) / 255.0);
        let (max, min) = (r.max(g).max(b), r.min(g).min(b));
        let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };
        if saturation < 0.12 || max < 0.15 {
            // Glass or ink.
            continue;
        }
        tissue += 1;
        let d = max - min;
        let hue = if max == r {
            60.0 * ((g - b) / d).rem_euclid(6.0)
        } else if max == g {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        if (15.0..=50.0).contains(&hue) && saturation > 0.3 {
            brown += 1;
        }
    }
    tissue > 0 && brown * 50 >= tissue
}

/// What a quantification needs, captured on the UI thread; [`Self::run`]
/// does the work on a worker. The panel and scripts' `quantify` use it.
pub(crate) struct Quantification {
    /// Stored levels only.
    wsi: eov_common::WsiFile,
    pub(crate) region: Rect,
    pub(crate) preset: StainPreset,
    pub(crate) stain: usize,
    pub(crate) thresholds: [f32; 3],
    /// The background white point.
    i0: [f32; 3],
}

impl Quantification {
    /// Captures what quantifying `region` of `file` needs (UI thread). The
    /// stain index must belong to the preset and the thresholds must pass
    /// [`check_thresholds`].
    pub(crate) fn of_file(
        file: &OpenFile,
        region: Rect,
        preset: StainPreset,
        stain: usize,
        thresholds: [f32; 3],
    ) -> Self {
        Self {
            wsi: file.wsi.without_synthetic_levels().pinned(),
            region,
            preset,
            stain,
            thresholds,
            i0: background_of(file),
        }
    }

    /// The level the region is read at, and its downsample.
    pub(crate) fn level(&self) -> (u32, f64) {
        level_for(&levels(&self.wsi), &self.region)
    }

    /// Reads the region and quantifies it (worker thread); `None` when
    /// cancelled.
    pub(crate) fn run(&self, cancel: &AtomicBool) -> Result<Option<StainQuantification>, String> {
        measure(
            &self.wsi,
            &self.region,
            self.preset,
            self.stain,
            self.thresholds,
            self.i0,
            cancel,
        )
    }
}

/// The background white point from the slide's thumbnail, which shows
/// glass even when the region is all tissue.
fn background_of(file: &OpenFile) -> [f32; 3] {
    file.thumbnail
        .as_ref()
        .map(|thumb| {
            let pixels = thumb
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[3] == 255)
                .map(|p| [p[0], p[1], p[2]]);
            estimate_background(pixels, &BackgroundOptions::default()).i0
        })
        .unwrap_or([255.0; 3])
}

fn levels(wsi: &eov_common::WsiFile) -> Vec<(u32, f64)> {
    (0..wsi.level_count())
        .filter_map(|level| wsi.level(level).map(|info| (level, info.downsample)))
        .collect()
}

/// Thresholds a quantification can use: positive and rising from weak to
/// moderate to strong.
pub(crate) fn check_thresholds(values: [f32; 3]) -> Result<[f32; 3], String> {
    if !values.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("Thresholds must be positive numbers".to_string());
    }
    if values[0] <= values[1] && values[1] <= values[2] {
        Ok(values)
    } else {
        Err("Thresholds must rise from weak to moderate to strong".to_string())
    }
}

/// What the panel is set to.
struct Panel {
    file_id: i32,
    roi: Rect,
    preset: StainPreset,
    stain: usize,
    thresholds: Result<[f32; 3], String>,
    result: Option<(StainQuantification, String)>,
    /// Set to stop a run in progress.
    cancel: Option<Arc<AtomicBool>>,
}

thread_local! {
    static PANEL: RefCell<Option<Panel>> = const { RefCell::new(None) };
}

/// The stain usually measured with a model: DAB or AEC for IHC, else the
/// last stain (eosin, PAS, ponceau).
pub(crate) fn default_stain(preset: StainPreset) -> usize {
    let names = preset
        .basis()
        .names()
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>();
    names
        .iter()
        .position(|name| name == "DAB" || name == "AEC")
        .unwrap_or(names.len().saturating_sub(1))
}

fn parse_thresholds(weak: &str, moderate: &str, strong: &str) -> Result<[f32; 3], String> {
    let parse = |text: &str, what: &str| {
        text.trim()
            .replace(',', ".")
            .parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| format!("The {what} threshold must be a positive number"))
    };
    check_thresholds([
        parse(weak, "weak")?,
        parse(moderate, "moderate")?,
        parse(strong, "strong")?,
    ])
}

/// The level whose region of `roi` is at most [`MAX_PIXELS`]: the finest
/// such level, or the coarsest.
fn level_for(levels: &[(u32, f64)], roi: &Rect) -> (u32, f64) {
    levels
        .iter()
        .copied()
        .find(|(_, downsample)| {
            (roi.width / downsample).ceil() * (roi.height / downsample).ceil() <= MAX_PIXELS as f64
        })
        .or_else(|| levels.last().copied())
        .unwrap_or((0, 1.0))
}

fn result_rows(q: &StainQuantification, stain: &str) -> Vec<(String, String)> {
    let percent = |n: u64| {
        if q.analysed_pixels == 0 {
            "–".to_string()
        } else {
            format!("{:.1} %", 100.0 * n as f64 / q.analysed_pixels as f64)
        }
    };
    vec![
        (
            "Tissue pixels analysed".into(),
            q.analysed_pixels.to_string(),
        ),
        (format!("{stain}-positive"), percent(q.positive_pixels)),
        ("Weak (1+)".into(), percent(q.bins[0])),
        ("Moderate (2+)".into(), percent(q.bins[1])),
        ("Strong (3+)".into(), percent(q.bins[2])),
        ("H-score (0–300)".into(), format!("{:.0}", q.h_score)),
        (
            format!("Mean {stain} concentration"),
            format!("{:.3}", q.mean_concentration),
        ),
    ]
}

/// Opens the panel for the focused pane's slide and its region of
/// interest.
pub(crate) fn open(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let found = {
        let state = state.read();
        let pane = state.active_viewport_pane();
        state
            .active_file_id_for_pane(pane)
            .and_then(|id| state.get_file(id))
            .map(|file| {
                (
                    file.id,
                    file.roi,
                    file.wsi.properties().stain.clone(),
                    file.thumbnail.as_ref().map(|thumb| shows_dab(&thumb.rgba)),
                )
            })
    };
    let notify = |text: &str| {
        crate::host_ui::with_host(|host| host.status(text));
    };
    let Some((file_id, roi, stain_label, dab)) = found else {
        notify("Open a slide to quantify a stain");
        return;
    };
    let Some(roi) = roi else {
        notify("Draw a region of interest first (R), then quantify the stain in it");
        return;
    };
    // The model and stain chosen last for this slide, else one suggested by
    // its stain label, else by whether its overview shows DAB.
    let previous = PANEL.with(|panel| {
        panel
            .borrow()
            .as_ref()
            .filter(|p| p.file_id == file_id && PRESETS.contains(&p.preset))
            .map(|p| (p.preset, p.stain))
    });
    let preset = previous
        .map(|(preset, _)| preset)
        .unwrap_or_else(|| match (&stain_label, dab) {
            (None, Some(false)) => StainPreset::HematoxylinEosin,
            _ => default_preset(stain_label.as_deref()),
        });
    let stain = previous
        .map(|(_, stain)| stain)
        .filter(|stain| *stain < preset.basis().len())
        .unwrap_or_else(|| default_stain(preset));
    PANEL.with(|panel| {
        *panel.borrow_mut() = Some(Panel {
            file_id,
            roi,
            preset,
            stain,
            thresholds: Ok(DEFAULT_THRESHOLDS),
            result: None,
            cancel: None,
        })
    });
    let [weak, moderate, strong] = DEFAULT_THRESHOLDS;
    ui.set_quantify_weak(format!("{weak}").into());
    ui.set_quantify_moderate(format!("{moderate}").into());
    ui.set_quantify_strong(format!("{strong}").into());
    refresh(ui, state);
    ui.set_quantify_visible(true);
}

/// Shows the panel's state.
fn refresh(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    PANEL.with(|panel| {
        let panel = panel.borrow();
        let Some(panel) = panel.as_ref() else {
            return;
        };
        let state = state.read();
        let Some(file) = state.get_file(panel.file_id) else {
            return;
        };
        let presets: Vec<ChoiceItem> = PRESETS
            .iter()
            .map(|preset| ChoiceItem {
                id: preset.id().into(),
                label: preset.label().into(),
            })
            .collect();
        let basis = panel.preset.basis();
        let stains: Vec<ChoiceItem> = basis
            .names()
            .iter()
            .enumerate()
            .map(|(index, name)| ChoiceItem {
                id: index.to_string().into(),
                label: (*name).into(),
            })
            .collect();
        let (level, downsample) =
            level_for(&levels(&file.wsi.without_synthetic_levels()), &panel.roi);
        let mpp = crate::host_ui::calibration(file)
            .average()
            .map(|um| format!(", {:.2} µm/px", um * downsample))
            .unwrap_or_default();
        ui.set_quantify_title(format!("Quantify Stain · {}", file.filename).into());
        ui.set_quantify_region_text(
            format!(
                "The region of interest, {} × {} px, read at level {level} ({:.0} × {:.0} px{mpp}).",
                panel.roi.width.ceil(),
                panel.roi.height.ceil(),
                (panel.roi.width / downsample).ceil(),
                (panel.roi.height / downsample).ceil(),
            )
            .into(),
        );
        ui.set_quantify_presets(ModelRc::from(Rc::new(VecModel::from(presets))));
        ui.set_quantify_preset(panel.preset.id().into());
        ui.set_quantify_stains(ModelRc::from(Rc::new(VecModel::from(stains))));
        ui.set_quantify_stain(panel.stain.to_string().into());
        ui.set_quantify_running(panel.cancel.is_some());
        let rows: Vec<ResultRow> = panel
            .result
            .as_ref()
            .map(|(q, stain)| result_rows(q, stain))
            .unwrap_or_default()
            .into_iter()
            .map(|(label, value)| ResultRow {
                label: label.into(),
                value: value.into(),
            })
            .collect();
        ui.set_quantify_results(ModelRc::from(Rc::new(VecModel::from(rows))));
        let note = match &panel.thresholds {
            Err(problem) => problem.clone(),
            Ok(_) if panel.cancel.is_some() => "Reading the region…".to_string(),
            Ok(_) => String::new(),
        };
        ui.set_quantify_note(note.into());
    });
}

pub(crate) fn preset_chosen(ui: &AppWindow, state: &Arc<RwLock<AppState>>, id: &str) {
    PANEL.with(|panel| {
        if let Some(panel) = panel.borrow_mut().as_mut()
            && let Some(preset) = PRESETS.iter().copied().find(|p| p.id() == id)
        {
            panel.preset = preset;
            panel.stain = default_stain(preset);
            panel.result = None;
        }
    });
    refresh(ui, state);
}

pub(crate) fn stain_chosen(ui: &AppWindow, state: &Arc<RwLock<AppState>>, id: &str) {
    PANEL.with(|panel| {
        if let Some(panel) = panel.borrow_mut().as_mut()
            && let Ok(index) = id.parse::<usize>()
            && index < panel.preset.basis().len()
        {
            panel.stain = index;
            panel.result = None;
        }
    });
    refresh(ui, state);
}

pub(crate) fn thresholds_edited(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    weak: &str,
    moderate: &str,
    strong: &str,
) {
    PANEL.with(|panel| {
        if let Some(panel) = panel.borrow_mut().as_mut() {
            panel.thresholds = parse_thresholds(weak, moderate, strong);
        }
    });
    refresh(ui, state);
}

/// Quantify: reads the region on a worker and measures it.
pub(crate) fn run(ui: &AppWindow, state: &Arc<RwLock<AppState>>) {
    let job = PANEL.with(|panel| {
        let mut panel = panel.borrow_mut();
        let panel = panel.as_mut()?;
        if panel.cancel.is_some() {
            return None;
        }
        let thresholds = panel.thresholds.clone().ok()?;
        let state = state.read();
        let file = state.get_file(panel.file_id)?;
        let job = Quantification::of_file(file, panel.roi, panel.preset, panel.stain, thresholds);
        let cancel = Arc::new(AtomicBool::new(false));
        panel.cancel = Some(Arc::clone(&cancel));
        panel.result = None;
        Some((job, cancel))
    });
    let Some((job, cancel)) = job else {
        return;
    };
    let (preset, stain) = (job.preset, job.stain);
    refresh(ui, state);
    let stop = Arc::clone(&cancel);
    let job_id = crate::host_ui::with_host(|host| {
        host.start_host_job(
            "Quantifying the stain",
            Some(Box::new(move || stop.store(true, Ordering::Relaxed))),
        )
    })
    .unwrap_or_default();
    let ui_weak = ui.as_weak();
    let worker_state = Arc::clone(state);
    let spawned = std::thread::Builder::new()
        .name("eov-quantify".to_string())
        .spawn(move || {
            let result = job.run(&cancel);
            let _ = slint::invoke_from_event_loop(move || {
                crate::host_ui::with_host(|host| host.finish_host_job(&job_id));
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };
                finished(&ui, &worker_state, preset, stain, result);
            });
        });
    if let Err(error) = spawned {
        finished(ui, state, preset, stain, Err(error.to_string()));
    }
}

/// Reads the region in chunks and quantifies it (worker thread).
fn measure(
    wsi: &eov_common::WsiFile,
    roi: &Rect,
    preset: StainPreset,
    stain: usize,
    thresholds: [f32; 3],
    i0: [f32; 3],
    cancel: &AtomicBool,
) -> Result<Option<StainQuantification>, String> {
    let (level, downsample) = level_for(&levels(wsi), roi);
    let (width, height) = (
        (roi.width / downsample).ceil(),
        (roi.height / downsample).ceil(),
    );
    // Even the coarsest level of a shallow pyramid: never allocate without
    // bound (a whole-slide region of a single-level slide).
    let pixels = width * height;
    if !pixels.is_finite() || pixels > MAX_READ_PIXELS as f64 {
        return Err(format!(
            "the region is {width:.0} × {height:.0} pixels at the slide's coarsest level \
             {level}; quantify a smaller region"
        ));
    }
    let width = width.max(1.0) as u32;
    let height = height.max(1.0) as u32;
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    const CHUNK: u32 = 2048;
    for y0 in (0..height).step_by(CHUNK as usize) {
        for x0 in (0..width).step_by(CHUNK as usize) {
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let cw = CHUNK.min(width - x0);
            let ch = CHUNK.min(height - y0);
            let data = wsi
                .read_region(
                    (roi.x + f64::from(x0) * downsample).round() as i64,
                    (roi.y + f64::from(y0) * downsample).round() as i64,
                    level,
                    cw,
                    ch,
                )
                .map_err(|error| error.to_string())?;
            for row in 0..ch as usize {
                let src = &data[row * cw as usize * 4..(row + 1) * cw as usize * 4];
                let dst = ((y0 as usize + row) * width as usize + x0 as usize) * 4;
                rgba[dst..dst + src.len()].copy_from_slice(src);
            }
        }
    }
    let basis = preset.basis();
    Ok(quantify_stain(
        &rgba,
        &basis,
        i0,
        stain,
        thresholds,
        &PixelFilter::default(),
    ))
}

fn finished(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    preset: StainPreset,
    stain: usize,
    result: Result<Option<StainQuantification>, String>,
) {
    let stain_name = preset
        .basis()
        .names()
        .get(stain)
        .map(|name| name.to_string())
        .unwrap_or_default();
    PANEL.with(|panel| {
        if let Some(panel) = panel.borrow_mut().as_mut() {
            panel.cancel = None;
            if let Ok(Some(q)) = &result {
                panel.result = Some((*q, stain_name.clone()));
            }
        }
    });
    if let Err(error) = &result {
        crate::host_ui::with_host(|host| {
            host.notify(
                NotifyLevel::Error,
                "Could not quantify the stain",
                Some(error),
                None,
            )
        });
    }
    refresh(ui, state);
}

/// Copies the results as text, with the model and thresholds used.
pub(crate) fn copy(state: &Arc<RwLock<AppState>>) {
    let text = PANEL.with(|panel| {
        let panel = panel.borrow();
        let panel = panel.as_ref()?;
        let (q, stain) = panel.result.as_ref()?;
        let state = state.read();
        let name = state
            .get_file(panel.file_id)
            .map(|file| file.filename.clone())
            .unwrap_or_default();
        let thresholds = panel.thresholds.clone().unwrap_or(DEFAULT_THRESHOLDS);
        let mut lines = vec![format!(
            "{name}: {stain} in the region of interest ({} model; thresholds {}, {}, {})",
            panel.preset.label(),
            thresholds[0],
            thresholds[1],
            thresholds[2]
        )];
        lines.extend(
            result_rows(q, stain)
                .into_iter()
                .map(|(label, value)| format!("{label}: {value}")),
        );
        Some(lines.join("\n"))
    });
    if let Some(text) = text {
        crate::clipboard::copy_text(text);
        crate::host_ui::with_host(|host| host.status("Copied the results"));
    }
}

/// Close: stops a run in progress.
pub(crate) fn close(ui: &AppWindow) {
    PANEL.with(|panel| {
        if let Some(cancel) = panel.borrow().as_ref().and_then(|p| p.cancel.as_ref()) {
            cancel.store(true, Ordering::Relaxed);
        }
    });
    ui.set_quantify_visible(false);
    ui.invoke_focus_keyboard();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dab_brown_is_told_from_haematoxylin_and_eosin() {
        let pixels = |colours: &[[u8; 3]]| -> Vec<u8> {
            colours
                .iter()
                .cycle()
                .take(4000)
                .flat_map(|c| [c[0], c[1], c[2], 255])
                .collect()
        };
        let he = pixels(&[
            [214, 140, 190],
            [110, 80, 160],
            [244, 244, 242],
            [200, 30, 40],
        ]);
        assert!(!shows_dab(&he), "eosin, haematoxylin, glass and blood");
        let ihc = pixels(&[[150, 100, 60], [110, 80, 160], [244, 244, 242]]);
        assert!(shows_dab(&ihc));
        assert!(!shows_dab(&pixels(&[[244, 244, 242]])), "glass only");
    }

    #[test]
    fn thresholds_must_be_positive_and_rising() {
        assert_eq!(parse_thresholds("0.15", "0,5", " 1 "), Ok([0.15, 0.5, 1.0]));
        assert!(parse_thresholds("0.5", "0.2", "1").is_err());
        assert!(parse_thresholds("-1", "0.2", "1").is_err());
        assert!(parse_thresholds("x", "0.2", "1").is_err());
    }

    #[test]
    fn the_finest_level_within_the_budget_is_read() {
        let levels = [(0, 1.0), (1, 4.0), (2, 16.0)];
        let small = Rect {
            x: 0.0,
            y: 0.0,
            width: 2000.0,
            height: 2000.0,
        };
        assert_eq!(level_for(&levels, &small), (0, 1.0));
        let large = Rect {
            width: 10_000.0,
            height: 10_000.0,
            ..small
        };
        assert_eq!(level_for(&levels, &large), (1, 4.0));
        let huge = Rect {
            width: 1e6,
            height: 1e6,
            ..small
        };
        assert_eq!(level_for(&levels, &huge), (2, 16.0));
    }

    /// An H-DAB slide: the left half moderately DAB-positive, the right
    /// half haematoxylin only.
    #[test]
    fn a_region_is_read_and_quantified() {
        let vectors = StainPreset::HematoxylinDab.basis().vectors();
        let pixel = |conc: [f32; 2]| {
            let mut rgb = [0u8; 3];
            for (c, out) in rgb.iter_mut().enumerate() {
                let od = vectors[0][c] * conc[0] + vectors[1][c] * conc[1];
                *out = (255.0 * (-od).exp()).round() as u8;
            }
            rgb
        };
        let positive = pixel([0.3, 0.7]);
        let negative = pixel([0.8, 0.0]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ihc.tif");
        eov_common::test_support::write_tiled_tiff(&path, 512, 512, 256, |x, _| {
            if x < 256 { positive } else { negative }
        })
        .unwrap();
        let wsi = eov_common::WsiFile::open(&path).unwrap();
        let roi = Rect {
            x: 0.0,
            y: 0.0,
            width: 512.0,
            height: 512.0,
        };
        let q = measure(
            &wsi,
            &roi,
            StainPreset::HematoxylinDab,
            1,
            DEFAULT_THRESHOLDS,
            [255.0; 3],
            &AtomicBool::new(false),
        )
        .unwrap()
        .unwrap();
        assert_eq!(q.analysed_pixels, 512 * 512);
        assert!((q.positive_fraction - 0.5).abs() < 0.01, "{q:?}");
        // All positives are moderate (0.5 ≤ 0.7 < 1.0): H-score 2 × 50.
        assert!((q.h_score - 100.0).abs() < 1.0, "{q:?}");
        // A region no level holds in reasonable memory is refused, never
        // allocated.
        let huge = Rect {
            width: 1e9,
            height: 1e9,
            ..roi
        };
        let refused = measure(
            &wsi,
            &huge,
            StainPreset::HematoxylinDab,
            1,
            DEFAULT_THRESHOLDS,
            [255.0; 3],
            &AtomicBool::new(false),
        );
        assert!(refused.unwrap_err().contains("smaller region"));
        // Cancelled before reading: nothing.
        assert_eq!(
            measure(
                &wsi,
                &roi,
                StainPreset::HematoxylinDab,
                1,
                DEFAULT_THRESHOLDS,
                [255.0; 3],
                &AtomicBool::new(true),
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn ihc_models_measure_their_chromogen() {
        assert_eq!(default_stain(StainPreset::HematoxylinDab), 1);
        assert_eq!(default_stain(StainPreset::HematoxylinEosinDab), 2);
        assert_eq!(default_stain(StainPreset::HematoxylinAec), 1);
        assert_eq!(default_stain(StainPreset::HematoxylinEosin), 1);
    }
}
