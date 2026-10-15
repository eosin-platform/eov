//! The viewport context menu and the Measure tool's mode menu.

use super::UiHost;
use super::annotations::HostEdit;
use crate::ContextMenuItem;
use crate::interaction::ToolChoice;
use crate::interaction::measure::{self, MeasureMode};
use crate::overlay::geometry::Point;
use crate::overlay::{ObjectKey, Owner};
use crate::plugin_runtime::EventTarget;
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
    /// The pane the menu was opened on.
    pane: usize,
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

/// A primary-modifier shortcut as the platform writes it.
fn shortcut(key: &str, shift: bool) -> String {
    if cfg!(target_os = "macos") {
        format!("{}⌘{key}", if shift { "⇧" } else { "" })
    } else {
        format!("Ctrl+{}{key}", if shift { "Shift+" } else { "" })
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
        let (file_id, view, has_roi, has_measurements, turned) = {
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
            )
        };
        let mut items = vec![
            item("viewport-copy-image", "Copy Image", "copy"),
            item("viewport-export-image", "Export Image…", "export"),
        ];
        mark_last_separator(&mut items);
        if file_id.is_some() {
            items.push(item("host:view-rotate-right", "Rotate Right 90°", ""));
            items.push(item("host:view-rotate-left", "Rotate Left 90°", ""));
            items.push(item("host:view-flip", "Flip Horizontally", ""));
            if turned {
                items.push(item("host:view-reset-orientation", "Reset Rotation", ""));
            }
            mark_last_separator(&mut items);
        }
        if let Some(file_id) = file_id {
            let mut add = item("host:bookmark-add", "Add Bookmark…", "");
            add.shortcut = shortcut("B", false).into();
            items.push(add);
            if self.slide_has_bookmarks(file_id as i32) {
                let mut list = item("host:bookmarks", "Bookmarks…", "");
                list.shortcut = shortcut("B", true).into();
                items.push(list);
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

    fn measurement_text(&self, file_id: FileId, id: &str) -> Option<String> {
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
        let items: Vec<ContextMenuItem> = MeasureMode::ALL
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
