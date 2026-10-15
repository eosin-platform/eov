//! The host's own annotations: the region of interest and measurements.
//!
//! They are stored with the open file (`OpenFile::roi`,
//! `OpenFile::measurements`), drawn through host overlay layers, editable
//! with the ROI and Measure tools, visible to plugins, and undoable (UI-13).

use super::UiHost;
use crate::interaction::measure::{
    self, Calibration, HostMeasurement, MEASUREMENT_Z, MeasureMode, ROI_OBJECT, ROI_Z,
};
use crate::overlay::Owner;
use crate::overlay::store::HostDecoration;
use crate::plugin_runtime::EventTarget;
use crate::plugin_runtime::undo::{HostChange, UndoAction, UndoEntry, UndoStep};
use crate::state::PaneId;
use eov_common::MeasurementUnit;
use eov_plugin_api::protocol::{FileId, HostEvent, LayerSpec, OverlayObject, Rect, Style};

/// A change to the host annotations requested by input.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HostEdit {
    SetRoi(Option<Rect>),
    Add(HostMeasurement),
    Update {
        id: String,
        shape: eov_plugin_api::protocol::Shape,
    },
    Remove(Vec<String>),
    ClearMeasurements,
}

impl UiHost {
    /// Creates the host layers of a newly opened file.
    pub(super) fn ensure_host_layers(&self, file_id: FileId) {
        let roi = LayerSpec::new(measure::roi_layer_id(file_id), file_id)
            .named("Region of interest")
            .at_z(ROI_Z)
            .with_style(Style {
                fill: eov_plugin_api::protocol::Rgba::TRANSPARENT,
                stroke: eov_plugin_api::protocol::Rgba(0xFF, 0xFF, 0xFF, 0xFF),
                stroke_width: 1.0,
                ..Style::default()
            });
        let measurements = LayerSpec::new(measure::measurement_layer_id(file_id), file_id)
            .named("Measurements")
            .at_z(MEASUREMENT_Z)
            .interactive(true)
            .editable(true)
            .with_style(measure::measurement_style());
        for spec in [roi, measurements] {
            if let Err(error) = self.store.create_layer(Owner::Host, spec) {
                tracing::error!("Cannot create host layer: {error}");
            }
        }
    }

    /// Mirrors a file's ROI and measurements into its host layers.
    pub(super) fn sync_host_layers(
        &self,
        file_id: FileId,
        roi: Option<Rect>,
        measurements: &[HostMeasurement],
        calibration: &Calibration,
        unit: MeasurementUnit,
    ) {
        let roi_layer = measure::roi_layer_id(file_id);
        if self.store.layer(&Owner::Host, &roi_layer).is_none() {
            self.ensure_host_layers(file_id);
        }
        let _ = self.store.clear_layer(&Owner::Host, &roi_layer);
        if let Some(rect) = roi {
            let object = OverlayObject::new(ROI_OBJECT, measure::roi_shape(&rect))
                .with_tooltip("Region of interest");
            if let Err(error) = self.store.upsert_decorated(
                &roi_layer,
                object,
                HostDecoration {
                    dim_outside: true,
                    contrast_border: true,
                },
            ) {
                tracing::error!("Cannot draw the region of interest: {error}");
            }
        }
        let objects = measurements
            .iter()
            .map(|m| {
                let value = measure::measure(m.mode, &m.shape, calibration);
                let label = measure::label(m.mode, &value, unit);
                OverlayObject::new(m.id.clone(), m.shape.clone())
                    .with_label(label.clone())
                    .with_tooltip(format!("{}: {label}", m.mode.label()))
            })
            .collect();
        if let Err(error) = self.store.upsert(
            &Owner::Host,
            &measure::measurement_layer_id(file_id),
            objects,
            true,
        ) {
            tracing::error!("Cannot draw measurements: {error}");
        }
    }

    /// Re-syncs every open file's host layers (e.g. after a unit change).
    pub(crate) fn resync_all_host_layers(&self) {
        let files: Vec<_> = {
            let state = self.state.read();
            state
                .open_files
                .iter()
                .map(|file| {
                    let unit = (0..state.panes.len())
                        .map(PaneId)
                        .filter(|pane| state.active_file_id_for_pane(*pane) == Some(file.id))
                        .find_map(|pane| file.pane_state(pane).map(|ps| ps.hud.measurement_unit))
                        .unwrap_or_default();
                    (
                        FileId::from(file.id),
                        file.roi,
                        file.measurements.clone(),
                        super::publish::calibration(file),
                        unit,
                    )
                })
                .collect()
        };
        for (file_id, roi, measurements, calibration, unit) in files {
            self.sync_host_layers(file_id, roi, &measurements, &calibration, unit);
        }
        self.redraw_overlays();
    }

    /// Applies an edit, records it for undo and redraws.
    pub(crate) fn apply_host_edit(&self, file_id: FileId, edit: HostEdit, record: bool) {
        let applied = {
            let mut state = self.state.write();
            let unit = state.file_pane_unit(file_id as i32).unwrap_or_default();
            let Some(file) = state.get_file_mut(file_id as i32) else {
                return;
            };
            let change = match edit {
                HostEdit::SetRoi(rect) => {
                    let rect = rect.filter(|r| r.width > 0.0 && r.height > 0.0);
                    if file.roi == rect {
                        None
                    } else {
                        let before = file.roi;
                        file.roi = rect;
                        Some((
                            if rect.is_some() {
                                "Set region of interest"
                            } else {
                                "Clear region of interest"
                            },
                            HostChange::Roi {
                                before,
                                after: rect,
                            },
                        ))
                    }
                }
                HostEdit::Add(measurement) => {
                    file.measurements.push(measurement.clone());
                    Some(("Add measurement", HostChange::MeasurementAdded(measurement)))
                }
                HostEdit::Update { id, shape } => {
                    match file.measurements.iter_mut().find(|m| m.id == id) {
                        Some(existing) if existing.shape != shape => {
                            let before = existing.clone();
                            existing.shape = shape;
                            Some((
                                "Edit measurement",
                                HostChange::MeasurementChanged {
                                    before,
                                    after: existing.clone(),
                                },
                            ))
                        }
                        _ => None,
                    }
                }
                HostEdit::Remove(ids) => {
                    let removed: Vec<HostMeasurement> = file
                        .measurements
                        .iter()
                        .filter(|m| ids.contains(&m.id))
                        .cloned()
                        .collect();
                    file.measurements.retain(|m| !ids.contains(&m.id));
                    let label = if removed.len() == 1 {
                        "Delete measurement"
                    } else {
                        "Delete measurements"
                    };
                    (!removed.is_empty())
                        .then_some((label, HostChange::MeasurementsRemoved(removed)))
                }
                HostEdit::ClearMeasurements => {
                    let removed = std::mem::take(&mut file.measurements);
                    (!removed.is_empty()).then_some((
                        "Clear measurements",
                        HostChange::MeasurementsRemoved(removed),
                    ))
                }
            };
            change.map(|(label, change)| {
                (
                    label,
                    change,
                    file.roi,
                    file.measurements.clone(),
                    super::publish::calibration(file),
                    unit,
                )
            })
        };
        let Some((label, change, roi, measurements, calibration, unit)) = applied else {
            return;
        };
        self.sync_host_layers(file_id, roi, &measurements, &calibration, unit);
        if record {
            self.undo.borrow_mut().push(UndoEntry {
                file_id: Some(file_id),
                label: label.to_string(),
                action: UndoAction::Host(change),
            });
        }
        self.refresh_undo_ui();
        self.redraw_overlays();
    }

    fn apply_host_change(&self, file_id: FileId, change: HostChange) {
        let edit = match change {
            HostChange::Roi { after, .. } => HostEdit::SetRoi(after),
            HostChange::MeasurementAdded(m) => HostEdit::Add(m),
            HostChange::MeasurementsRemoved(ms) => {
                HostEdit::Remove(ms.into_iter().map(|m| m.id).collect())
            }
            HostChange::MeasurementChanged { after, .. } => HostEdit::Update {
                id: after.id,
                shape: after.shape,
            },
        };
        self.apply_host_edit(file_id, edit, false);
    }

    /// Ctrl/⌘+Z: undoes the newest change to the slide in the focused pane.
    pub(crate) fn undo(&self) {
        let file_id = self.focused_file_id();
        let step = self.undo.borrow_mut().undo(file_id);
        self.perform_undo_step(step);
    }

    pub(crate) fn redo(&self) {
        let file_id = self.focused_file_id();
        let step = self.undo.borrow_mut().redo(file_id);
        self.perform_undo_step(step);
    }

    fn perform_undo_step(&self, step: Option<UndoStep>) {
        match step {
            Some(UndoStep::Host {
                file_id: Some(file_id),
                change,
            }) => self.apply_host_change(file_id, change),
            Some(UndoStep::Host { file_id: None, .. }) => {}
            Some(UndoStep::Plugin {
                plugin_id,
                entry_id,
                redo,
            }) => {
                let event = if redo {
                    HostEvent::Redo { entry_id }
                } else {
                    HostEvent::Undo { entry_id }
                };
                self.post(EventTarget::Plugin(plugin_id), event);
            }
            None => self.status("Nothing to undo"),
        }
        self.refresh_undo_ui();
    }

    pub(crate) fn focused_file_id(&self) -> Option<FileId> {
        let state = self.state.read();
        state
            .active_file_id_for_pane(state.focused_pane)
            .map(FileId::from)
    }

    /// Updates the toolbar's undo/redo buttons for the focused slide.
    pub(crate) fn refresh_undo_ui(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let Some(file_id) = self.state.try_read().map(|state| {
            state
                .active_file_id_for_pane(state.focused_pane)
                .map(FileId::from)
        }) else {
            // Called while the renderer holds the state: try again shortly.
            self.refresh_undo_ui_deferred();
            return;
        };
        let undo = self.undo.borrow();
        ui.set_undo_enabled(undo.can_undo(file_id));
        ui.set_redo_enabled(undo.can_redo(file_id));
        ui.set_undo_label(undo.undo_label(file_id).unwrap_or_default().into());
        ui.set_redo_label(undo.redo_label(file_id).unwrap_or_default().into());
    }

    pub(super) fn refresh_undo_ui_deferred(&self) {
        slint::Timer::single_shot(std::time::Duration::ZERO, || {
            super::with_host(|host| host.refresh_undo_ui());
        });
    }

    /// The measurement mode the Measure tool uses.
    pub(crate) fn measure_mode(&self) -> MeasureMode {
        self.measure_mode.get()
    }
}
