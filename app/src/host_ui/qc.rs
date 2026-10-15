//! Tissue QC in the viewer (AUDIT §14.5): **Tissue QC** in the viewport
//! menu checks the slide for pen marks, possible folds and out-of-focus
//! areas ([`eov_common::qc`]) as a status-bar job, then colours the marked
//! cells over the slide (blue pen, orange folds, yellow blur) and says how
//! much of the tissue each covers.

use super::{NotifyLevel, UiHost, with_host};
use crate::overlay::store::{Owner, StoredRaster};
use eov_common::qc::{self, BLUR, FOLD, PEN, QcOptions, QcResult};
use eov_plugin_api::protocol::{FileId, LayerSpec, Rect};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Above heat maps, under the region of interest and annotations.
const QC_Z: i32 = -55;

/// RGBA of a cell by its marks (straight alpha), the first that applies.
const PEN_COLOUR: [u8; 4] = [0x2f, 0x7b, 0xff, 0xb0];
const FOLD_COLOUR: [u8; 4] = [0xff, 0x8c, 0x1a, 0xb0];
const BLUR_COLOUR: [u8; 4] = [0xff, 0xd4, 0x00, 0x99];

fn layer_id(file_id: i32) -> String {
    format!("host-qc-{file_id}")
}

/// A check running or done, per file.
enum Check {
    Running(Arc<AtomicBool>),
    Done(Arc<QcResult>),
}

thread_local! {
    static CHECKS: RefCell<HashMap<i32, Check>> = RefCell::new(HashMap::new());
}

/// The overlay of a result: one premultiplied pixel per cell.
fn overlay_pixels(result: &QcResult) -> Vec<u8> {
    result
        .flags
        .iter()
        .flat_map(|&flag| {
            let [r, g, b, a] = if flag & PEN != 0 {
                PEN_COLOUR
            } else if flag & FOLD != 0 {
                FOLD_COLOUR
            } else if flag & BLUR != 0 {
                BLUR_COLOUR
            } else {
                [0, 0, 0, 0]
            };
            let premultiply = |c: u8| ((u16::from(c) * u16::from(a) + 127) / 255) as u8;
            [premultiply(r), premultiply(g), premultiply(b), a]
        })
        .collect()
}

/// One line for people: what share of the tissue each mark covers.
pub(crate) fn summary_text(result: &QcResult) -> String {
    let s = &result.summary;
    let percent = |share: f32| format!("{:.1} %", share * 100.0);
    let tissue = s
        .tissue_mm2
        .map(|mm2| format!("{mm2:.1} mm² of tissue"))
        .unwrap_or_else(|| format!("{} tissue cells", s.tissue_cells));
    let pen = if s.pen_skipped {
        "pen not checked (the stain looks like ink)".to_string()
    } else {
        format!("pen marks {}", percent(s.pen))
    };
    let blur = s
        .blur
        .map(|share| format!(", possibly out of focus {}", percent(share)))
        .unwrap_or_default();
    format!(
        "Tissue QC: {pen}, possible folds {}{blur} ({tissue})",
        percent(s.fold)
    )
}

impl UiHost {
    /// Whether the viewport menu offers Tissue QC for a slide, and whether
    /// its result is shown.
    pub(super) fn qc_menu_state(&self, file_id: i32) -> (bool, bool) {
        let brightfield = self
            .state
            .read()
            .get_file(file_id)
            .is_some_and(|file| !file.wsi.is_multichannel());
        let shown =
            CHECKS.with(|checks| matches!(checks.borrow().get(&file_id), Some(Check::Done(_))));
        (brightfield, shown)
    }

    /// Tissue QC: checks the slide on a worker.
    pub(super) fn run_tissue_qc(&self, file_id: i32) {
        let Some(wsi) = self
            .state
            .read()
            .get_file(file_id)
            .map(|file| file.wsi.clone())
        else {
            return;
        };
        if CHECKS.with(|checks| matches!(checks.borrow().get(&file_id), Some(Check::Running(_)))) {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        CHECKS.with(|checks| {
            checks
                .borrow_mut()
                .insert(file_id, Check::Running(Arc::clone(&cancel)))
        });
        let stop = Arc::clone(&cancel);
        let job = self.start_host_job(
            "Checking tissue quality",
            Some(Box::new(move || stop.store(true, Ordering::Relaxed))),
        );
        let spawned = std::thread::Builder::new()
            .name("eov-tissue-qc".to_string())
            .spawn({
                let job = job.clone();
                move || {
                    let last = std::cell::Cell::new(Instant::now());
                    let progress = |fraction: f32| {
                        // A few updates a second are enough for the bar.
                        if last.get().elapsed() < Duration::from_millis(200) {
                            return;
                        }
                        last.set(Instant::now());
                        let job = job.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            with_host(|host| host.update_host_job(&job, Some(fraction), None));
                        });
                    };
                    let result = qc::run(&wsi, &QcOptions::default(), &progress, &cancel)
                        .map_err(|error| error.to_string());
                    let _ = slint::invoke_from_event_loop(move || {
                        with_host(|host| host.tissue_qc_done(file_id, &job, result));
                    });
                }
            });
        if let Err(error) = spawned {
            self.tissue_qc_done(file_id, &job, Err(error.to_string()));
        }
    }

    fn tissue_qc_done(&self, file_id: i32, job: &str, result: Result<Option<QcResult>, String>) {
        self.finish_host_job(job);
        CHECKS.with(|checks| checks.borrow_mut().remove(&file_id));
        if self.state.read().get_file(file_id).is_none() {
            return;
        }
        let result = match result {
            Ok(Some(result)) => Arc::new(result),
            Ok(None) => return,
            Err(error) => {
                self.notify(
                    NotifyLevel::Error,
                    "Tissue QC could not check this slide",
                    Some(&error),
                    None,
                );
                return;
            }
        };
        self.show_tissue_qc(file_id, &result);
        self.notify(
            NotifyLevel::Info,
            &summary_text(&result),
            Some(
                "Blue: pen marks. Orange: possible folds (tissue much darker than the rest). \
                 Yellow: possibly out of focus (tissue without fine detail, such as fibrin or \
                 necrosis, can be marked too). The thresholds are relative to this slide; the \
                 marks are for review, not measurements.",
            ),
            None,
        );
        CHECKS.with(|checks| checks.borrow_mut().insert(file_id, Check::Done(result)));
    }

    fn show_tissue_qc(&self, file_id: i32, result: &QcResult) {
        let layer = layer_id(file_id);
        self.store.remove_layer(&Owner::Host, &layer);
        let spec = LayerSpec::new(layer.clone(), FileId::from(file_id))
            .named("Tissue QC")
            .at_z(QC_Z);
        if let Err(error) = self.store.create_layer(Owner::Host, spec) {
            tracing::error!("Cannot show tissue QC: {error}");
            return;
        }
        let raster = StoredRaster {
            object_id: "cells".to_string(),
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: f64::from(result.columns) * result.cell_width,
                height: f64::from(result.rows) * result.cell_height,
            },
            width: result.columns,
            height: result.rows,
            premultiplied: Arc::new(overlay_pixels(result)),
            smooth: false,
            seq: 0,
        };
        if let Err(error) = self.store.put_raster(&Owner::Host, &layer, raster) {
            tracing::error!("Cannot show tissue QC: {error}");
        }
        self.request_render();
    }

    /// Hide Tissue QC: removes the overlay and forgets the result.
    pub(super) fn hide_tissue_qc(&self, file_id: i32) {
        CHECKS.with(|checks| checks.borrow_mut().remove(&file_id));
        self.store.remove_layer(&Owner::Host, &layer_id(file_id));
        self.request_render();
    }

    /// Copy Tissue QC Summary.
    pub(super) fn copy_tissue_qc(&self, file_id: i32) {
        let text = CHECKS.with(|checks| match checks.borrow().get(&file_id) {
            Some(Check::Done(result)) => Some(summary_text(result)),
            _ => None,
        });
        if let Some(text) = text {
            crate::clipboard::copy_text(text);
        }
    }

    /// The file closed (its layer went with it).
    pub(crate) fn tissue_qc_file_closed(&self, file_id: i32) {
        CHECKS.with(|checks| {
            if let Some(Check::Running(cancel)) = checks.borrow_mut().remove(&file_id) {
                cancel.store(true, Ordering::Relaxed);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::qc::{QcSummary, TISSUE};

    #[test]
    fn marks_are_coloured_in_order_and_summarised() {
        let result = QcResult {
            columns: 4,
            rows: 1,
            cell_width: 256.0,
            cell_height: 256.0,
            flags: vec![
                TISSUE,
                TISSUE | PEN | FOLD,
                TISSUE | FOLD | BLUR,
                TISSUE | BLUR,
            ],
            sharpness: vec![f32::NAN; 4],
            tissue_share: vec![1.0; 4],
            summary: QcSummary {
                cells: 4,
                tissue_cells: 4,
                tissue_mm2: Some(1.25),
                pen: 0.25,
                fold: 0.5,
                blur: Some(0.5),
                pen_skipped: false,
            },
        };
        let pixels = overlay_pixels(&result);
        assert_eq!(&pixels[0..4], &[0, 0, 0, 0], "clean tissue is not coloured");
        assert_eq!(pixels[7], PEN_COLOUR[3], "pen first");
        assert_eq!(
            u32::from(pixels[4]),
            (0x2f * 0xb0 + 127) / 255,
            "premultiplied"
        );
        assert_eq!(
            u32::from(pixels[8]),
            (0xff * 0xb0 + 127) / 255,
            "then folds"
        );
        assert_eq!(pixels[15], BLUR_COLOUR[3]);
        let text = summary_text(&result);
        assert_eq!(
            text,
            "Tissue QC: pen marks 25.0 %, possible folds 50.0 %, possibly out of focus 50.0 % (1.2 mm² of tissue)"
        );
    }
}
