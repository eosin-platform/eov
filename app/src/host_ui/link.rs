//! Linked navigation in the host UI: the viewport menu's alignment
//! commands and the landmark markers (see `crate::slide_link`).

use super::UiHost;
use crate::overlay::store::Owner;
use crate::state::PaneId;
use eov_plugin_api::protocol::{FileId, LayerSpec, OverlayObject, Rgba, Shape, SizeUnit, Style};

/// Above measurements, below plugin overlays drawn on top.
const LANDMARK_Z: i32 = 90;

fn landmark_layer_id(file_id: i32) -> String {
    format!("host-link-landmarks-{file_id}")
}

impl UiHost {
    /// Handles the viewport menu's `link-…` commands for the menu opened on
    /// `pane` at level-0 `point`. Returns whether it did.
    pub(super) fn link_command(&self, command: &str, pane: usize, point: [f64; 2]) -> bool {
        let Some(action) = command.strip_prefix("link-") else {
            return false;
        };
        let Some(ui) = self.ui() else {
            return true;
        };
        match action {
            "landmark-1" | "landmark-2" => {
                let index = usize::from(action == "landmark-2");
                {
                    let mut state = self.state.write();
                    if let Some(file_id) = state.active_file_id_for_pane(PaneId(pane)) {
                        crate::slide_link::set_landmark(&mut state, file_id, index, point);
                    }
                }
                self.status(&format!(
                    "Landmark {} placed; place both landmarks on the other slide too, then Align from Landmarks",
                    index + 1
                ));
            }
            "align" => {
                self.state.write().set_focused_pane(PaneId(pane));
                crate::callbacks::align_linked_panes(&ui, &self.state);
            }
            "clear-landmarks" => {
                crate::slide_link::clear_landmarks(&mut self.state.write());
                self.status("Landmarks cleared");
            }
            "reset" => {
                self.state.write().set_focused_pane(PaneId(pane));
                crate::callbacks::reset_linked_alignment(&ui, &self.state);
            }
            _ => {}
        }
        self.sync_landmark_layers();
        self.request_render();
        true
    }

    /// Shows each slide's alignment landmarks (numbered markers).
    pub(crate) fn sync_landmark_layers(&self) {
        let landmarks: Vec<(i32, [Option<[f64; 2]>; 2])> = {
            let state = self.state.read();
            let mut marks: Vec<_> = state
                .slide_link
                .landmarks
                .iter()
                .map(|(file, marks)| (*file, *marks))
                .collect();
            // Layers of slides without landmarks are removed below.
            for file in &state.open_files {
                if !state.slide_link.landmarks.contains_key(&file.id) {
                    marks.push((file.id, [None, None]));
                }
            }
            marks
        };
        for (file_id, marks) in landmarks {
            let layer_id = landmark_layer_id(file_id);
            if marks.iter().all(Option::is_none) {
                self.store.remove_layer(&Owner::Host, &layer_id);
                continue;
            }
            if self.store.layer(&Owner::Host, &layer_id).is_none() {
                let spec = LayerSpec::new(layer_id.clone(), FileId::from(file_id))
                    .named("Alignment landmarks")
                    .at_z(LANDMARK_Z)
                    .with_style(Style {
                        fill: Rgba(0x00, 0xE5, 0xFF, 0xFF),
                        stroke: Rgba(0x00, 0x30, 0x40, 0xFF),
                        stroke_width: 2.0,
                        point_radius: 6.0,
                        size_unit: SizeUnit::Screen,
                        text_color: Rgba(0x00, 0xE5, 0xFF, 0xFF),
                        ..Style::default()
                    });
                if let Err(error) = self.store.create_layer(Owner::Host, spec) {
                    tracing::error!("Cannot show alignment landmarks: {error}");
                    continue;
                }
            }
            let objects = marks
                .iter()
                .enumerate()
                .filter_map(|(index, mark)| {
                    mark.map(|[x, y]| {
                        OverlayObject::new(format!("landmark-{}", index + 1), Shape::Point { x, y })
                            .with_label(format!("{}", index + 1))
                            .with_tooltip(format!("Alignment landmark {}", index + 1))
                    })
                })
                .collect();
            if let Err(error) = self.store.upsert(&Owner::Host, &layer_id, objects, true) {
                tracing::error!("Cannot show alignment landmarks: {error}");
            }
        }
    }
}
