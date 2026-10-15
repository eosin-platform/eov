//! The viewport context menu and the Measure tool's mode menu.

use super::UiHost;
use super::annotations::HostEdit;
use crate::ContextMenuItem;
use crate::interaction::ToolChoice;
use crate::interaction::measure::{self, MeasureMode};
use crate::overlay::geometry::Point;
use crate::overlay::{ObjectKey, Owner};
use crate::plugin_runtime::EventTarget;
use crate::shortcuts::Action;
use crate::state::PaneId;
use eov_plugin_api::protocol::{
    ContextMenuResponse, FileId, HostEvent, MenuContext, MenuItem, ObjectRef,
};
use slint::ComponentHandle;
use slint::{ModelRc, VecModel};
use std::rc::Rc;
use std::sync::Arc;

/// Items a plugin may add to one menu.
const MAX_PLUGIN_ITEMS: usize = 20;

pub(super) struct OpenMenu {
    contexts: Vec<(Arc<str>, MenuContext)>,
    file_id: Option<FileId>,
    /// The pane the menu was opened on, and the level-0 point under it.
    pane: usize,
    image: Point,
}

fn item(id: impl Into<String>, label: &str, icon: &str) -> ContextMenuItem {
    ContextMenuItem {
        id: id.into().into(),
        label: label.into(),
        icon: icon.into(),
        shortcut: "".into(),
        enabled: true,
        separator_after: false,
    }
}

fn plugin_item(plugin_id: &str, menu_item: &MenuItem) -> ContextMenuItem {
    ContextMenuItem {
        id: format!("plugin:{plugin_id}:{}", menu_item.item_id).into(),
        label: menu_item.label.clone().into(),
        icon: if menu_item.checked == Some(true) {
            "check".into()
        } else {
            menu_item.icon.clone().into()
        },
        shortcut: menu_item.shortcut.clone().into(),
        enabled: menu_item.enabled,
        separator_after: menu_item.separator_after,
    }
}

fn mark_last_separator(items: &mut [ContextMenuItem]) {
    if let Some(last) = items.last_mut() {
        last.separator_after = true;
    }
}

impl UiHost {
    pub(super) fn open_viewport_menu(
        &self,
        pane: usize,
        window_x: f64,
        window_y: f64,
        image: Point,
        hits: Vec<ObjectKey>,
    ) {
        let Some(ui) = self.ui() else {
            return;
        };
        let (linked, can_align, has_landmarks) = {
            let state = self.state.read();
            (
                state.slide_link.enabled && state.split_enabled,
                crate::slide_link::can_align_from_landmarks(&state, PaneId(pane)),
                !state.slide_link.landmarks.is_empty(),
            )
        };
        let normalizing = {
            let state = self.state.read();
            state
                .active_file_id_for_pane(PaneId(pane))
                .and_then(|id| state.get_file(id))
                .and_then(|file| file.pane_state(PaneId(pane)))
                .is_some_and(|ps| {
                    ps.hud.stain_normalization != eov_common::StainNormalization::None
                })
        };
        let (file_id, view, has_roi, has_measurements, turned, multichannel, color_managed) = {
            let state = self.state.read();
            let file = state
                .active_file_id_for_pane(PaneId(pane))
                .and_then(|id| state.get_file(id));
            let scale = f64::from(ui.window().scale_factor());
            (
                file.map(|f| FileId::from(f.id)),
                super::publish::view_info(&state, PaneId(pane), scale),
                file.is_some_and(|f| f.roi.is_some()),
                file.is_some_and(|f| !f.measurements.is_empty()),
                file.and_then(|f| f.pane_state(PaneId(pane)))
                    .is_some_and(|ps| !ps.viewport.viewport.orientation.is_identity()),
                file.is_some_and(|f| f.wsi.is_multichannel()),
                // Whether the slide is colour-managed, for slides with a
                // usable profile.
                file.and_then(|f| {
                    matches!(
                        crate::color::slide_profile(f.id, &f.wsi),
                        crate::color::SlideProfile::Ready(_)
                    )
                    .then_some(f.color_management)
                }),
            )
        };
        let keys = crate::shortcuts::current();
        let with_key = |mut item: ContextMenuItem, action: Action| {
            item.shortcut = keys.label(action).into();
            item
        };
        let mut items = vec![
            with_key(
                item("viewport-copy-image", "Copy Image", "copy"),
                Action::CopyView,
            ),
            with_key(
                item("viewport-export-image", "Export Image…", "export"),
                Action::ExportView,
            ),
        ];
        mark_last_separator(&mut items);
        if multichannel {
            items.push(with_key(
                item("host:channels", "Channels…", ""),
                Action::Channels,
            ));
            mark_last_separator(&mut items);
        }
        if file_id.is_some() {
            items.push(with_key(
                item("host:view-rotate-right", "Rotate Right 90°", ""),
                Action::RotateRight,
            ));
            items.push(with_key(
                item("host:view-rotate-left", "Rotate Left 90°", ""),
                Action::RotateLeft,
            ));
            items.push(with_key(
                item("host:view-flip", "Flip Horizontally", ""),
                Action::Flip,
            ));
            if turned {
                items.push(item("host:view-reset-orientation", "Reset Rotation", ""));
            }
            mark_last_separator(&mut items);
        }
        if let Some(on) = color_managed {
            items.push(item(
                "host:color-management",
                if on {
                    "Show Colours as Scanned"
                } else {
                    "Convert Colours to sRGB"
                },
                "",
            ));
            mark_last_separator(&mut items);
        }
        // The stain normalisation target (AUDIT §14.5), where the pane
        // normalises.
        if normalizing && file_id.is_some() {
            items.push(item(
                "host:stain-target-slide",
                "Use This Slide as Stain Target",
                "",
            ));
            if has_roi {
                items.push(item(
                    "host:stain-target-region",
                    "Use the Region as Stain Target",
                    "",
                ));
            }
            items.push(item("host:stain-target-load", "Load Stain Target…", ""));
            if crate::stain_target::current().is_some() {
                items.push(item("host:stain-target-save", "Save Stain Target…", ""));
                items.push(item(
                    "host:stain-target-reset",
                    "Use the Reference Stain Target",
                    "",
                ));
            }
            mark_last_separator(&mut items);
        }
        // Linked navigation: align this slide to the others (Phase 3 #5).
        if linked && file_id.is_some() {
            items.push(item(
                "host:link-landmark-1",
                "Place Alignment Landmark 1",
                "",
            ));
            items.push(item(
                "host:link-landmark-2",
                "Place Alignment Landmark 2",
                "",
            ));
            if can_align {
                items.push(item("host:link-align", "Align from Landmarks", ""));
            }
            if has_landmarks {
                items.push(item("host:link-clear-landmarks", "Clear Landmarks", ""));
            }
            items.push(item("host:link-reset", "Reset Alignment", ""));
            mark_last_separator(&mut items);
        }
        if let Some(file_id) = file_id {
            items.push(with_key(
                item("host:bookmark-add", "Add Bookmark…", ""),
                Action::BookmarkAdd,
            ));
            if self.slide_has_bookmarks(file_id as i32) {
                items.push(with_key(
                    item("host:bookmarks", "Bookmarks…", ""),
                    Action::Bookmarks,
                ));
            }
            items.push(item(
                "host:heatmap",
                if self.has_heatmap(file_id as i32) {
                    "Heat Map…"
                } else {
                    "Show Heat Map…"
                },
                "",
            ));
            let (brightfield, qc_shown) = self.qc_menu_state(file_id as i32);
            if brightfield {
                items.push(item(
                    "host:qc",
                    if qc_shown {
                        "Check Tissue Quality Again"
                    } else {
                        "Check Tissue Quality"
                    },
                    "",
                ));
                if qc_shown {
                    items.push(item("host:qc-copy", "Copy Tissue QC Summary", "copy"));
                    items.push(item("host:qc-hide", "Hide Tissue QC", ""));
                }
            }
            mark_last_separator(&mut items);
        }

        let host_hit = hits.iter().find(|key| key.layer.owner == Owner::Host);
        let mut host_items = Vec::new();
        if let Some(key) = host_hit
            && measure::is_measurement_layer(&key.layer.id)
        {
            host_items.push(item(
                format!("host:measurement-copy:{}", key.object_id),
                "Copy Measurement",
                "copy",
            ));
            host_items.push(item(
                format!("host:measurement-delete:{}", key.object_id),
                "Delete Measurement",
                "delete",
            ));
        }
        if has_roi {
            host_items.push(item(
                "host:roi-frame",
                "Zoom to Region of Interest",
                "measure",
            ));
            host_items.push(item("host:roi-clear", "Clear Region of Interest", "delete"));
        }
        if has_roi || has_measurements {
            host_items.push(with_key(
                item("host:measurements-panel", "Measurements…", "measure"),
                Action::Measurements,
            ));
        }
        if has_roi {
            host_items.push(with_key(
                item("host:quantify", "Quantify Stain in Region…", ""),
                Action::Quantify,
            ));
        }
        if has_measurements {
            host_items.push(item(
                "host:measurements-clear",
                "Clear All Measurements",
                "delete",
            ));
        }
        if !host_items.is_empty() {
            mark_last_separator(&mut host_items);
            items.extend(host_items);
        }

        let mut contexts = Vec::new();
        if let Some(view) = view {
            let plugins: Vec<Arc<str>> = self
                .runtime
                .borrow()
                .slots()
                .iter()
                .filter(|slot| slot.is_active())
                .map(|slot| Arc::from(slot.info.id.as_str()))
                .collect();
            for plugin_id in plugins {
                if !self.shared.wants(&plugin_id, "context_menu_requested") {
                    continue;
                }
                let owner = Owner::Plugin(Arc::clone(&plugin_id));
                let context = MenuContext {
                    pane: view.pane,
                    view,
                    image_x: image[0],
                    image_y: image[1],
                    hit: hits
                        .iter()
                        .find(|key| key.layer.owner == owner)
                        .map(|key| ObjectRef {
                            layer_id: key.layer.id.clone(),
                            object_id: key.object_id.clone(),
                        }),
                    selection: self
                        .store
                        .selection_of(&owner)
                        .into_iter()
                        .map(|key| ObjectRef {
                            layer_id: key.layer.id,
                            object_id: key.object_id,
                        })
                        .collect(),
                };
                let reply = self.request(
                    &plugin_id,
                    &HostEvent::ContextMenuRequested(context.clone()),
                );
                let response: ContextMenuResponse = reply
                    .and_then(|value| serde_json::from_value(value).ok())
                    .unwrap_or_default();
                let plugin_items: Vec<ContextMenuItem> = response
                    .items
                    .iter()
                    .filter(|i| !i.item_id.is_empty() && !i.item_id.contains(':'))
                    .take(MAX_PLUGIN_ITEMS)
                    .map(|i| plugin_item(&plugin_id, i))
                    .collect();
                if !plugin_items.is_empty() {
                    mark_last_separator(&mut items);
                    items.extend(plugin_items);
                }
                contexts.push((plugin_id, context));
            }
        }
        if let Some(last) = items.last_mut() {
            last.separator_after = false;
        }
        *self.menu.borrow_mut() = Some(OpenMenu {
            contexts,
            file_id,
            pane,
            image,
        });

        ui.set_context_menu_tab_id(-1);
        ui.set_context_menu_series_path("".into());
        ui.set_drag_source_pane(pane as i32);
        ui.set_context_menu_x(window_x as f32);
        ui.set_context_menu_y(window_y as f32);
        ui.set_context_menu_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_context_menu_visible(true);
    }

    /// Handles menu commands owned by the host UI; returns whether it did.
    pub(crate) fn context_menu_command(&self, command: &str) -> bool {
        if let Some(rest) = command.strip_prefix("plugin:") {
            let Some((plugin_id, item_id)) = rest.split_once(':') else {
                return true;
            };
            let context = self.menu.borrow().as_ref().and_then(|menu| {
                menu.contexts
                    .iter()
                    .find(|(owner, _)| &**owner == plugin_id)
                    .map(|(_, context)| context.clone())
            });
            if let Some(context) = context {
                self.post(
                    EventTarget::Plugin(Arc::from(plugin_id)),
                    HostEvent::ContextMenuItemSelected {
                        item_id: item_id.to_string(),
                        context,
                    },
                );
            }
            return true;
        }
        if let Some(mode) = command.strip_prefix("measure-mode:") {
            if let Some(mode) = MeasureMode::from_id(mode) {
                self.set_measure_mode(mode);
            }
            return true;
        }
        let Some(host_command) = command.strip_prefix("host:") else {
            return false;
        };
        if self.bookmark_command(host_command) {
            return true;
        }
        let opened_at = self
            .menu
            .borrow()
            .as_ref()
            .map(|menu| (menu.pane, menu.image));
        if let Some((pane, image)) = opened_at
            && self.link_command(host_command, pane, image)
        {
            return true;
        }
        if host_command == "measurements-panel" {
            self.open_measurements_panel();
            return true;
        }
        if host_command == "quantify" {
            if let Some(ui) = self.ui() {
                crate::quantify_panel::open(&ui, &self.state);
            }
            return true;
        }
        if host_command == "color-management" {
            let pane = self.menu.borrow().as_ref().map(|menu| menu.pane);
            if let (Some(pane), Some(ui)) = (pane, self.ui()) {
                self.state.write().set_focused_pane(PaneId(pane));
                crate::callbacks::toggle_color_management(&ui, &self.state);
                self.request_render();
            }
            return true;
        }
        if let Some(action) = host_command.strip_prefix("view-") {
            let pane = self.menu.borrow().as_ref().map(|menu| menu.pane);
            if let Some(pane) = pane {
                {
                    let mut state = self.state.write();
                    state.set_focused_pane(PaneId(pane));
                    let changed = match action {
                        "rotate-right" => crate::callbacks::rotate_active_viewport(&mut state, 1),
                        "rotate-left" => crate::callbacks::rotate_active_viewport(&mut state, -1),
                        "flip" => crate::callbacks::flip_active_viewport(&mut state),
                        _ => crate::callbacks::reset_active_orientation(&mut state),
                    };
                    if changed {
                        state.mirror_locked_viewports_from_active();
                        state.request_render();
                    }
                }
                self.request_render();
            }
            return true;
        }
        let file_id = self.menu.borrow().as_ref().and_then(|menu| menu.file_id);
        let Some(file_id) = file_id else {
            return true;
        };
        if host_command == "channels" {
            crate::channels::open(self, file_id as i32);
            return true;
        }
        if host_command == "heatmap" {
            self.open_heatmap_panel_for(file_id as i32);
            return true;
        }
        if host_command.starts_with("stain-target-") {
            let pane = self.menu.borrow().as_ref().map(|menu| menu.pane);
            if let Some(pane) = pane {
                self.stain_target_command(host_command, file_id as i32, pane);
            }
            return true;
        }
        match host_command {
            "qc" => {
                self.run_tissue_qc(file_id as i32);
                return true;
            }
            "qc-copy" => {
                self.copy_tissue_qc(file_id as i32);
                return true;
            }
            "qc-hide" => {
                self.hide_tissue_qc(file_id as i32);
                return true;
            }
            _ => {}
        }
        if let Some(id) = host_command.strip_prefix("measurement-delete:") {
            self.apply_host_edit(file_id, HostEdit::Remove(vec![id.to_string()]), true);
        } else if let Some(id) = host_command.strip_prefix("measurement-copy:") {
            let text = self.measurement_text(file_id, id);
            if let Some(text) = text {
                crate::clipboard::copy_text(text.clone());
                if let Some(ui) = self.ui() {
                    super::notices::show_toast(
                        &ui,
                        &format!("Copied {text}"),
                        eov_plugin_api::protocol::NotifyLevel::Info,
                    );
                }
            }
        } else if host_command == "roi-clear" {
            self.apply_host_edit(file_id, HostEdit::SetRoi(None), true);
        } else if host_command == "measurements-clear" {
            self.apply_host_edit(file_id, HostEdit::ClearMeasurements, true);
        } else if host_command == "roi-frame" {
            {
                let mut state = self.state.write();
                let roi = state.get_file(file_id as i32).and_then(|f| f.roi);
                if let (Some(roi), Some(viewport)) = (roi, state.active_viewport_mut()) {
                    viewport.smooth_frame_rect(roi.x, roi.y, roi.width, roi.height);
                    state.request_render();
                }
            }
            self.request_render();
        }
        true
    }

    pub(super) fn measurement_text(&self, file_id: FileId, id: &str) -> Option<String> {
        let state = self.state.read();
        let file = state.get_file(file_id as i32)?;
        let m = file.measurements.iter().find(|m| m.id == id)?;
        let unit = state.file_pane_unit(file.id).unwrap_or_default();
        let value = measure::measure(m.mode, &m.shape, &super::publish::calibration(file));
        Some(measure::label(m.mode, &value, unit))
    }

    pub(crate) fn open_measure_mode_menu(&self, x: f32, y: f32) {
        let Some(ui) = self.ui() else {
            return;
        };
        let current = self.measure_mode();
        let mut items: Vec<ContextMenuItem> = MeasureMode::ALL
            .iter()
            .map(|mode| ContextMenuItem {
                id: format!("measure-mode:{}", mode.id()).into(),
                label: mode.label().into(),
                icon: if *mode == current {
                    "check".into()
                } else {
                    "".into()
                },
                shortcut: "".into(),
                enabled: true,
                separator_after: false,
            })
            .collect();
        mark_last_separator(&mut items);
        let mut list = item("host:measurements-panel", "Show Measurements…", "");
        list.shortcut = crate::shortcuts::current()
            .label(Action::Measurements)
            .into();
        items.push(list);
        *self.menu.borrow_mut() = None;
        ui.set_context_menu_tab_id(-1);
        ui.set_context_menu_series_path("".into());
        ui.set_context_menu_x(x);
        ui.set_context_menu_y(y);
        ui.set_context_menu_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_context_menu_visible(true);
    }

    pub(crate) fn set_measure_mode(&self, mode: MeasureMode) {
        self.measure_mode.set(mode);
        if let Err(error) = crate::config::save_measure_mode(mode.id()) {
            tracing::warn!("Could not save the measure mode: {error}");
        }
        if let Some(ui) = self.ui() {
            ui.set_measure_mode_label(mode.label().into());
        }
        super::input::set_tool(self, ToolChoice::Measure(mode));
    }
}

#[cfg(test)]
mod tests {
    use slint::Model;

    /// Multichannel slides offer their Channels panel in the viewport menu.
    #[test]
    fn multichannel_slides_get_a_channels_item() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let setup = crate::test_support::HostWithSlide::new();
            setup
                .host
                .open_viewport_menu(0, 10.0, 10.0, [0.0, 0.0], Vec::new());
            let items: Vec<String> = setup
                .ui
                .get_context_menu_items()
                .iter()
                .map(|item| item.id.to_string())
                .collect();
            assert!(items.contains(&"host:channels".to_string()), "{items:?}");
            assert!(setup.host.context_menu_command("host:channels"));
            assert!(setup.ui.get_channels_panel_visible());
        });
    }
}
