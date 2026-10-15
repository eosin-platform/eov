//! Slint callbacks owned by the host UI, and the plugin toolbar model.

use super::{UiHost, with_host};
use crate::plugin_runtime::EventTarget;
use crate::{AppWindow, PluginButtonData};
use eov_plugin_api::protocol::{HostEvent, ViewInfo};
use slint::{ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

thread_local! {
    static SVG_CACHE: RefCell<HashMap<String, Image>> = RefCell::new(HashMap::new());
}

/// Renders (and caches) an inline SVG icon.
pub(crate) fn svg_image(svg: &str) -> Image {
    SVG_CACHE.with(|cache| {
        if let Some(image) = cache.borrow().get(svg) {
            return image.clone();
        }
        let image = if svg.trim().is_empty() {
            None
        } else {
            Image::load_from_svg_data(svg.as_bytes()).ok()
        }
        .unwrap_or_else(|| {
            Image::from_rgba8_premultiplied(SharedPixelBuffer::<Rgba8Pixel>::new(1, 1))
        });
        let mut cache = cache.borrow_mut();
        if cache.len() > 512 {
            cache.clear();
        }
        cache.insert(svg.to_string(), image.clone());
        image
    })
}

impl UiHost {
    /// Rebuilds the plugin part of the toolbar (buttons and tools).
    pub(crate) fn refresh_toolbar(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let active_tool = self
            .controller
            .borrow()
            .tool()
            .plugin_tool()
            .map(|(owner, spec)| (owner.to_string(), spec.tool_id.clone()));
        let contributions = self.shared.contributions.lock();
        let mut buttons = Vec::new();
        let mut last_plugin: Option<String> = None;
        let tools = contributions.tools();
        let toolbar = contributions.toolbar();
        // Tools first (they act like the host tools), then buttons; each
        // plugin's group is separated.
        let mut entries: Vec<(String, PluginButtonData)> = Vec::new();
        for tool in tools {
            let active = active_tool
                .as_ref()
                .is_some_and(|(p, t)| **p == *tool.plugin_id && *t == tool.spec.tool_id);
            let hotkey = tool
                .spec
                .hotkey
                .as_ref()
                .map(|h| format!(" ({})", h.key.to_uppercase()))
                .unwrap_or_default();
            entries.push((
                tool.plugin_id.to_string(),
                PluginButtonData {
                    plugin_id: tool.plugin_id.to_string().into(),
                    button_id: tool.spec.tool_id.clone().into(),
                    tooltip: format!("{}{hotkey}", tool.spec.label).into(),
                    icon: svg_image(&tool.spec.icon_svg),
                    active,
                    enabled: true,
                    is_tool: true,
                    show_left_separator: false,
                },
            ));
        }
        for button in toolbar {
            let icon = if button.spec.active {
                button
                    .spec
                    .toggled_icon_svg
                    .as_deref()
                    .unwrap_or(&button.spec.icon_svg)
            } else {
                &button.spec.icon_svg
            };
            let hotkey = button
                .spec
                .hotkey
                .as_ref()
                .map(|h| format!(" ({})", h.key.to_uppercase()))
                .unwrap_or_default();
            entries.push((
                button.plugin_id.to_string(),
                PluginButtonData {
                    plugin_id: button.plugin_id.to_string().into(),
                    button_id: button.spec.button_id.clone().into(),
                    tooltip: format!("{}{hotkey}", button.spec.tooltip).into(),
                    icon: svg_image(icon),
                    active: button.spec.active,
                    enabled: button.spec.enabled,
                    is_tool: false,
                    show_left_separator: false,
                },
            ));
        }
        drop(contributions);
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (plugin, mut data) in entries {
            data.show_left_separator = last_plugin.as_deref() != Some(plugin.as_str());
            last_plugin = Some(plugin);
            buttons.push(data);
        }
        ui.set_plugin_buttons(ModelRc::from(Rc::new(VecModel::from(buttons))));
    }

    fn hud_click(&self, pane: usize, plugin_id: &str, button_id: &str, item_id: &str) {
        let (uid, view) = {
            let state = self.state.read();
            let pane_id = crate::state::PaneId(pane);
            let scale = self
                .ui()
                .map(|ui| f64::from(ui.window().scale_factor()))
                .unwrap_or(1.0);
            (
                state.pane_uid(pane_id).unwrap_or_default(),
                super::publish::view_info(&state, pane_id, scale),
            )
        };
        let view: Option<ViewInfo> = view;
        self.post(
            EventTarget::Plugin(Arc::from(plugin_id)),
            HostEvent::HudButtonClicked {
                button_id: button_id.to_string(),
                item_id: (!item_id.is_empty()).then(|| item_id.to_string()),
                pane: uid,
                view,
            },
        );
    }
}

pub(super) fn install(ui: &AppWindow, host: &Rc<UiHost>) {
    ui.on_viewport_pointer(
        |pane, kind, button, x, y, wx, wy, shift, control, alt, meta| {
            with_host(|host| {
                host.on_pointer(
                    pane.max(0) as usize,
                    kind,
                    button,
                    x,
                    y,
                    wx,
                    wy,
                    super::modifiers_from_slint(shift, control, alt, meta),
                )
            });
        },
    );
    ui.on_viewport_wheel(|pane, dx, dy, x, y, shift, control, alt, meta| {
        with_host(|host| {
            host.on_wheel(
                pane.max(0) as usize,
                dx,
                dy,
                x,
                y,
                super::modifiers_from_slint(shift, control, alt, meta),
            )
        });
    });
    ui.on_viewport_pinch(|pane, scale, x, y, active| {
        with_host(|host| host.on_pinch(pane.max(0) as usize, scale, x, y, active));
    });
    ui.on_viewport_key(|pane, text, shift, control, alt, meta, pressed, repeat| {
        with_host(|host| {
            host.on_viewport_key(
                pane.max(0) as usize,
                &text,
                super::modifiers_from_slint(shift, control, alt, meta),
                pressed,
                repeat,
            )
        })
        .unwrap_or(false)
    });
    ui.on_global_key(|text, shift, control, alt, meta, pressed, repeat| {
        with_host(|host| {
            host.on_global_key(
                &text,
                super::modifiers_from_slint(shift, control, alt, meta),
                pressed,
                repeat,
            )
        })
        .unwrap_or(false)
    });
    ui.on_tool_selected(|tool| {
        with_host(|host| host.select_host_tool(tool));
    });
    ui.on_plugin_button_clicked(|plugin_id, button_id, is_tool| {
        with_host(|host| {
            if is_tool {
                host.toggle_plugin_tool(&plugin_id, &button_id);
            } else {
                host.post(
                    EventTarget::Plugin(Arc::from(plugin_id.as_str())),
                    HostEvent::ToolbarButtonClicked {
                        button_id: button_id.to_string(),
                    },
                );
            }
        });
    });
    ui.on_hud_plugin_button_clicked(|pane, plugin_id, button_id, item_id| {
        with_host(|host| host.hud_click(pane.max(0) as usize, &plugin_id, &button_id, &item_id));
    });
    ui.on_undo_requested(|| {
        with_host(|host| host.undo());
    });
    ui.on_redo_requested(|| {
        with_host(|host| host.redo());
    });
    ui.on_measure_mode_menu_requested(|x, y| {
        with_host(|host| host.open_measure_mode_menu(x, y));
    });
    ui.on_shortcuts_copy_config_path(|| {
        with_host(|host| host.copy_config_path());
    });
    let (ui_weak, state) = (ui.as_weak(), Arc::clone(&host.state));
    ui.on_quantify_preset_chosen(move |id| {
        if let Some(ui) = ui_weak.upgrade() {
            crate::quantify_panel::preset_chosen(&ui, &state, &id);
        }
    });
    let (ui_weak, state) = (ui.as_weak(), Arc::clone(&host.state));
    ui.on_quantify_stain_chosen(move |id| {
        if let Some(ui) = ui_weak.upgrade() {
            crate::quantify_panel::stain_chosen(&ui, &state, &id);
        }
    });
    let (ui_weak, state) = (ui.as_weak(), Arc::clone(&host.state));
    ui.on_quantify_thresholds_edited(move |weak, moderate, strong| {
        if let Some(ui) = ui_weak.upgrade() {
            crate::quantify_panel::thresholds_edited(&ui, &state, &weak, &moderate, &strong);
        }
    });
    let (ui_weak, state) = (ui.as_weak(), Arc::clone(&host.state));
    ui.on_quantify_run(move || {
        if let Some(ui) = ui_weak.upgrade() {
            crate::quantify_panel::run(&ui, &state);
        }
    });
    let state = Arc::clone(&host.state);
    ui.on_quantify_copy(move || crate::quantify_panel::copy(&state));
    let ui_weak = ui.as_weak();
    ui.on_quantify_close(move || {
        if let Some(ui) = ui_weak.upgrade() {
            crate::quantify_panel::close(&ui);
        }
    });
    ui.on_measurement_go(|id| {
        with_host(|host| host.measurement_go(&id));
    });
    ui.on_measurement_copy(|id| {
        with_host(|host| host.measurement_copy(&id));
    });
    ui.on_measurement_delete(|id| {
        with_host(|host| host.measurement_delete(&id));
    });
    ui.on_measurements_copy_all(|| {
        with_host(|host| host.measurements_copy_all());
    });
    ui.on_measurements_clear(|| {
        with_host(|host| host.measurements_clear());
    });
    let ui_weak = ui.as_weak();
    ui.on_bookmark_dialog_accepted(move |name| {
        with_host(|host| host.finish_bookmark(&name));
        if let Some(ui) = ui_weak.upgrade() {
            ui.invoke_focus_keyboard();
        }
    });
    let ui_weak = ui.as_weak();
    ui.on_bookmark_dialog_cancelled(move || {
        with_host(|host| host.cancel_bookmark());
        if let Some(ui) = ui_weak.upgrade() {
            ui.invoke_focus_keyboard();
        }
    });
    ui.on_plugin_confirm_accepted(|| {
        with_host(|host| host.answer_plugin_confirm(true));
    });
    ui.on_plugin_confirm_cancelled(|| {
        with_host(|host| host.answer_plugin_confirm(false));
    });
    let ui_weak = ui.as_weak();
    ui.on_plugin_modal_dismiss_requested(move || {
        with_host(|host| host.dismiss_plugin_modal());
        if let Some(ui) = ui_weak.upgrade() {
            ui.invoke_focus_keyboard();
        }
    });
    let ui_weak = ui.as_weak();
    ui.on_plugins_panel_requested(move || {
        with_host(|host| host.refresh_plugin_panel());
        if let Some(ui) = ui_weak.upgrade() {
            ui.set_plugins_panel_visible(true);
        }
    });
    ui.on_plugin_enable_requested(|plugin_id, enable| {
        with_host(|host| host.set_plugin_enabled(&plugin_id, enable));
    });
    ui.on_plugins_check_updates(|| {
        with_host(|host| host.check_plugin_updates());
    });
    ui.on_plugin_update_requested(|plugin_id| {
        with_host(|host| host.update_plugin(&plugin_id));
    });
    ui.on_plugin_remove_requested(|plugin_id| {
        with_host(|host| host.remove_plugin(&plugin_id));
    });
    ui.on_plugins_folder_requested(|| {
        with_host(|host| host.open_plugin_folder());
    });
    ui.on_job_cancel_requested(|job_id| {
        with_host(|host| {
            if host.cancel_host_job(&job_id) {
                return;
            }
            let owner = host.notices.borrow().job_owner(&job_id);
            if let Some(owner) = owner {
                host.post(
                    EventTarget::Plugin(owner),
                    HostEvent::JobCancelRequested {
                        job_id: job_id.to_string(),
                    },
                );
            }
        });
    });
    ui.on_notifications_opened(|| {
        with_host(|host| {
            host.notices.borrow_mut().mark_read();
            host.refresh_notices();
        });
    });
    ui.on_log_folder_requested(|| {
        if let Err(error) = crate::logging::open_log_folder() {
            with_host(|host| {
                host.notify(
                    eov_plugin_api::protocol::NotifyLevel::Error,
                    "Could not open the log folder",
                    Some(&error),
                    None,
                )
            });
        }
    });
    ui.on_notifications_cleared(|| {
        with_host(|host| {
            host.notices.borrow_mut().clear();
            host.refresh_notices();
        });
    });
}
