//! The Measurements panel (AUDIT §14.1): a slide's region of interest and
//! measurements with their values, to go to, copy (one, or all as CSV) or
//! delete. Edits go through the undo history like any other.

use super::UiHost;
use super::annotations::HostEdit;
use crate::MeasurementRow;
use crate::interaction::measure::{self, Calibration, MeasureMode};
use eov_common::MeasurementUnit;
use eov_plugin_api::protocol::{FileId, NotifyLevel, Rect};
use slint::{ModelRc, VecModel};
use std::rc::Rc;

/// The row id of the region of interest.
const ROI_ID: &str = "roi";

/// A region of interest's size as people read it.
fn roi_value(rect: &Rect, calibration: &Calibration, unit: MeasurementUnit) -> String {
    let value = measure::measure(
        MeasureMode::Rectangle,
        &measure::roi_shape(rect),
        calibration,
    );
    match (calibration.spacing(), value.area_um2) {
        (Some((x, y, approximate)), Some(area)) => format!(
            "{}{} × {} · {}",
            if approximate { "≈ " } else { "" },
            measure::format_length(rect.width * x, unit),
            measure::format_length(rect.height * y, unit),
            measure::format_area(area, unit)
        ),
        _ => format!("{} × {} px", rect.width.round(), rect.height.round()),
    }
}

/// A CSV field, quoted when it needs to be.
fn csv(field: &str) -> String {
    if field.contains([',', '"', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn number(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.3}"))
        .unwrap_or_default()
}

/// A rectangle to frame for `bounds`: lines and points get a size.
fn framing(bounds: Rect) -> Rect {
    let side = bounds.width.max(bounds.height).max(64.0);
    let width = bounds.width.max(side * 0.5);
    let height = bounds.height.max(side * 0.5);
    Rect {
        x: bounds.x + bounds.width / 2.0 - width / 2.0,
        y: bounds.y + bounds.height / 2.0 - height / 2.0,
        width,
        height,
    }
}

impl UiHost {
    /// Lists the focused slide's region and measurements.
    pub(crate) fn open_measurements_panel(&self) {
        let file_id = {
            let state = self.state.read();
            state.active_file_id_for_pane(state.active_viewport_pane())
        };
        let Some(file_id) = file_id else {
            self.status("Open a slide to see its measurements");
            return;
        };
        self.measurements_panel.set(Some(FileId::from(file_id)));
        if let Some(ui) = self.ui() {
            ui.set_measurements_panel_visible(true);
        }
        self.refresh_measurements_panel();
    }

    /// Shows the panel's slide as it is now (after edits, undo or restore).
    pub(super) fn refresh_measurements_panel(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let Some(file_id) = self.measurements_panel.get() else {
            return;
        };
        if !ui.get_measurements_panel_visible() {
            return;
        }
        let state = self.state.read();
        let Some(file) = state.get_file(file_id as i32) else {
            // The slide was closed.
            drop(state);
            self.measurements_panel.set(None);
            ui.set_measurements_panel_visible(false);
            return;
        };
        let unit = state.file_pane_unit(file.id).unwrap_or_default();
        let calibration = super::publish::calibration(file);
        let mut rows = Vec::new();
        if let Some(rect) = &file.roi {
            rows.push(MeasurementRow {
                id: ROI_ID.into(),
                kind: "Region of interest".into(),
                value: roi_value(rect, &calibration, unit).into(),
            });
        }
        for m in &file.measurements {
            let value = measure::measure(m.mode, &m.shape, &calibration);
            rows.push(MeasurementRow {
                id: m.id.clone().into(),
                kind: m.mode.label().into(),
                value: measure::label(m.mode, &value, unit).into(),
            });
        }
        ui.set_measurements_title(format!("Measurements · {}", file.filename).into());
        let key = crate::shortcuts::current().label(crate::shortcuts::Action::ToolMeasure);
        ui.set_measurements_empty_text(
            if key.is_empty() {
                "No measurements yet: use the Measure tool.".to_string()
            } else {
                format!("No measurements yet: use the Measure tool ({key}).")
            }
            .into(),
        );
        ui.set_measurement_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
    }

    /// Frames a measurement (or the region) in the focused pane.
    pub(crate) fn measurement_go(&self, id: &str) {
        let Some(file_id) = self.measurements_panel.get() else {
            return;
        };
        let framed = {
            let mut state = self.state.write();
            let bounds = state.get_file(file_id as i32).and_then(|file| {
                if id == ROI_ID {
                    file.roi
                } else {
                    file.measurements
                        .iter()
                        .find(|m| m.id == id)
                        .map(|m| m.shape.bounds())
                }
            });
            let shown =
                state.active_file_id_for_pane(state.active_viewport_pane()) == Some(file_id as i32);
            match bounds {
                Some(bounds) if shown => {
                    let rect = framing(bounds);
                    if let Some(viewport) = state.active_viewport_mut() {
                        viewport.smooth_frame_rect(rect.x, rect.y, rect.width, rect.height);
                    }
                    state.mirror_locked_viewports_from_active();
                    state.request_render();
                    true
                }
                _ => false,
            }
        };
        if !framed {
            self.status("That slide is not shown in the focused pane");
            return;
        }
        if let Some(ui) = self.ui() {
            ui.set_measurements_panel_visible(false);
            ui.invoke_focus_keyboard();
        }
        self.request_render();
    }

    pub(crate) fn measurement_copy(&self, id: &str) {
        let Some(file_id) = self.measurements_panel.get() else {
            return;
        };
        let text = if id == ROI_ID {
            let state = self.state.read();
            state.get_file(file_id as i32).and_then(|file| {
                let unit = state.file_pane_unit(file.id).unwrap_or_default();
                file.roi.map(|rect| {
                    format!(
                        "Region of interest: {}",
                        roi_value(&rect, &super::publish::calibration(file), unit)
                    )
                })
            })
        } else {
            self.measurement_text(file_id, id)
        };
        if let Some(text) = text {
            crate::clipboard::copy_text(text.clone());
            if let Some(ui) = self.ui() {
                super::notices::show_toast(&ui, &format!("Copied {text}"), NotifyLevel::Info);
            }
        }
    }

    pub(crate) fn measurement_delete(&self, id: &str) {
        let Some(file_id) = self.measurements_panel.get() else {
            return;
        };
        let edit = if id == ROI_ID {
            HostEdit::SetRoi(None)
        } else {
            HostEdit::Remove(vec![id.to_string()])
        };
        self.apply_host_edit(file_id, edit, true);
    }

    pub(crate) fn measurements_clear(&self) {
        if let Some(file_id) = self.measurements_panel.get() {
            self.apply_host_edit(file_id, HostEdit::ClearMeasurements, true);
        }
    }

    /// Copies every row as CSV: values in µm, µm² and degrees, plus pixels
    /// and the level-0 bounds.
    pub(crate) fn measurements_copy_all(&self) {
        let Some(file_id) = self.measurements_panel.get() else {
            return;
        };
        let text = {
            let state = self.state.read();
            let Some(file) = state.get_file(file_id as i32) else {
                return;
            };
            let unit = state.file_pane_unit(file.id).unwrap_or_default();
            let calibration = super::publish::calibration(file);
            let mut lines = vec![
                "slide,kind,value,length_um,area_um2,angle_deg,length_px,area_px2,approximate,x,y,width,height"
                    .to_string(),
            ];
            let mut row = |kind: &str, label: String, value: measure::MeasurementValue, b: Rect| {
                lines.push(
                    [
                        csv(&file.filename),
                        csv(kind),
                        csv(&label),
                        number(value.length_um),
                        number(value.area_um2),
                        number(value.angle_deg),
                        number(Some(value.length_px)),
                        number(Some(value.area_px2)),
                        value.approximate.to_string(),
                        number(Some(b.x)),
                        number(Some(b.y)),
                        number(Some(b.width)),
                        number(Some(b.height)),
                    ]
                    .join(","),
                );
            };
            if let Some(rect) = file.roi {
                let shape = measure::roi_shape(&rect);
                let value = measure::measure(MeasureMode::Rectangle, &shape, &calibration);
                row(
                    "Region of interest",
                    roi_value(&rect, &calibration, unit),
                    value,
                    rect,
                );
            }
            for m in &file.measurements {
                let value = measure::measure(m.mode, &m.shape, &calibration);
                let label = measure::label(m.mode, &value, unit);
                row(m.mode.label(), label, value, m.shape.bounds());
            }
            lines.join("\n") + "\n"
        };
        let count = text.lines().count().saturating_sub(1);
        crate::clipboard::copy_text(text);
        if let Some(ui) = self.ui() {
            super::notices::show_toast(
                &ui,
                &format!("Copied {count} rows as CSV"),
                NotifyLevel::Info,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_fields_are_quoted_when_needed() {
        assert_eq!(csv("a.svs"), "a.svs");
        assert_eq!(csv("Case 1, H&E.svs"), "\"Case 1, H&E.svs\"");
        assert_eq!(csv("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn lines_and_points_are_framed_with_some_room() {
        let line = framing(Rect {
            x: 100.0,
            y: 50.0,
            width: 400.0,
            height: 0.0,
        });
        assert_eq!(line.width, 400.0);
        assert_eq!(line.height, 200.0);
        assert_eq!(line.y, -50.0);
        let point = framing(Rect::default());
        assert!(point.width > 0.0 && point.height > 0.0);
    }

    #[test]
    fn regions_read_in_physical_units() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            width: 2000.0,
            height: 1000.0,
        };
        let calibrated = Calibration::new(Some(0.5), Some(0.5));
        assert_eq!(
            roi_value(&rect, &calibrated, MeasurementUnit::Um),
            "1.00 mm × 500 µm · 0.500 mm²"
        );
        assert_eq!(
            roi_value(&rect, &Calibration::default(), MeasurementUnit::Um),
            "2000 × 1000 px"
        );
    }
}
