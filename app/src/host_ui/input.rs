//! Viewport input, tool selection and keyboard shortcuts.
//!
//! Slint forwards raw input; the [`Controller`](crate::interaction::Controller)
//! interprets it; [`UiHost::apply_effects`] carries the effects out, taking
//! `AppState` only briefly and queueing plugin events.
//!
//! All keyboard shortcuts live in one table ([`UiHost::shortcut`]) reached
//! through Slint's focus chain, so a focused text field or dialog consumes
//! keys first (UI-7, UI-8), and plugin hotkeys are validated against it.

use super::UiHost;
use super::annotations::HostEdit;
use crate::interaction::controller::{
    BROADCAST, Effect, KeyInput, PointerInput, PointerKind, keys,
};
use crate::interaction::{PaneView, ToolChoice};
use crate::plugin_runtime::EventTarget;
use crate::plugin_runtime::contributions::HotkeyTarget;
use crate::state::PaneId;
use crate::{PaneCursor, ToolType};
use eov_plugin_api::protocol::{Cursor, FileId, HostEvent, Modifiers, PointerButton};
use slint::ComponentHandle;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A held hotkey that switches tools only while pressed.
pub(super) struct Momentary {
    key: String,
    previous: ToolChoice,
    pressed_at: Instant,
    saw_repeat: bool,
}

/// Holding a tool hotkey longer than this makes it momentary.
const MOMENTARY_HOLD: Duration = Duration::from_millis(250);

/// Converts Slint modifiers. In Slint `control` is ⌘ on macOS and `meta` is
/// the Control key there (UI-9).
pub(crate) fn modifiers_from_slint(shift: bool, control: bool, alt: bool, meta: bool) -> Modifiers {
    if cfg!(target_os = "macos") {
        Modifiers {
            shift,
            control: meta,
            alt,
            meta: control,
            primary: control,
        }
    } else {
        Modifiers {
            shift,
            control,
            alt,
            meta,
            primary: control,
        }
    }
}

/// A stable name for a key, as plugin hotkeys spell it.
pub(crate) fn key_name(text: &str) -> String {
    let mut chars = text.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else {
        return text.to_lowercase();
    };
    match c {
        '\u{8}' => "Backspace".into(),
        '\t' => "Tab".into(),
        '\n' | '\r' => "Enter".into(),
        '\u{1b}' => "Escape".into(),
        '\u{7f}' => "Delete".into(),
        ' ' => "Space".into(),
        '\u{F700}' => "ArrowUp".into(),
        '\u{F701}' => "ArrowDown".into(),
        '\u{F702}' => "ArrowLeft".into(),
        '\u{F703}' => "ArrowRight".into(),
        '\u{F704}'..='\u{F71B}' => format!("F{}", c as u32 - 0xF704 + 1),
        '\u{F729}' => "Home".into(),
        '\u{F72B}' => "End".into(),
        '\u{F72C}' => "PageUp".into(),
        '\u{F72D}' => "PageDown".into(),
        c => c.to_lowercase().collect(),
    }
}

fn cursor_of(cursor: Cursor) -> PaneCursor {
    match cursor {
        Cursor::Default => PaneCursor::Default,
        Cursor::Crosshair => PaneCursor::Crosshair,
        Cursor::Pointer => PaneCursor::Pointer,
        Cursor::Move => PaneCursor::Move,
        Cursor::Text => PaneCursor::Text,
        Cursor::NotAllowed => PaneCursor::NotAllowed,
    }
}

/// Switches tools and updates the UI.
pub(super) fn set_tool(host: &UiHost, tool: ToolChoice) {
    let effects = host.controller.borrow_mut().set_tool(tool);
    host.apply_effects(effects);
    host.after_tool_change();
}

impl UiHost {
    fn pane_view(&self, index: usize) -> Option<PaneView> {
        let state = self.state.read();
        let pane = PaneId(index);
        let uid = state.pane_uid(pane)?;
        let file_id = state.active_file_id_for_pane(pane);
        let (viewport, calibration, unit) = match file_id.and_then(|id| state.get_file(id)) {
            Some(file) => {
                let pane_state = file.pane_state(pane)?;
                (
                    pane_state.viewport.viewport.clone(),
                    super::publish::calibration(file),
                    pane_state.hud.measurement_unit,
                )
            }
            None => (
                eov_common::Viewport::new(1.0, 1.0, 1.0, 1.0),
                Default::default(),
                Default::default(),
            ),
        };
        Some(PaneView {
            index,
            uid,
            file_id: file_id.map(FileId::from),
            viewport,
            calibration,
            unit,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn on_pointer(
        &self,
        index: usize,
        kind: i32,
        button: i32,
        x: f32,
        y: f32,
        window_x: f32,
        window_y: f32,
        modifiers: Modifiers,
    ) {
        let Some(view) = self.pane_view(index) else {
            return;
        };
        let input = PointerInput {
            kind: match kind {
                0 => PointerKind::Down,
                1 => PointerKind::Up,
                3 => PointerKind::Leave,
                _ => PointerKind::Move,
            },
            button: match button {
                1 => PointerButton::Left,
                2 => PointerButton::Right,
                3 => PointerButton::Middle,
                _ => PointerButton::None,
            },
            x: f64::from(x),
            y: f64::from(y),
            window_x: f64::from(window_x),
            window_y: f64::from(window_y),
            modifiers,
        };
        if input.kind == PointerKind::Down
            && let Some(ui) = self.ui()
        {
            // Any press dismisses open menus and popovers.
            ui.set_context_menu_visible(false);
            ui.set_notifications_visible(false);
        }
        let effects = self
            .controller
            .borrow_mut()
            .pointer(&view, input, Instant::now());
        self.apply_effects(effects);
    }

    pub(crate) fn on_wheel(
        &self,
        index: usize,
        dx: f32,
        dy: f32,
        x: f32,
        y: f32,
        modifiers: Modifiers,
    ) {
        let Some(view) = self.pane_view(index) else {
            return;
        };
        let effects = self.controller.borrow_mut().wheel(
            &view,
            f64::from(x),
            f64::from(y),
            f64::from(dx),
            f64::from(dy),
            modifiers,
        );
        self.apply_effects(effects);
    }

    pub(crate) fn on_pinch(&self, index: usize, scale: f32, x: f32, y: f32, active: bool) {
        let Some(view) = self.pane_view(index) else {
            return;
        };
        let effects = self.controller.borrow_mut().pinch(
            &view,
            f64::from(scale),
            f64::from(x),
            f64::from(y),
            active,
        );
        self.apply_effects(effects);
    }

    /// A key in a focused viewport: tools first, then shortcuts.
    pub(crate) fn on_viewport_key(
        &self,
        index: usize,
        text: &str,
        modifiers: Modifiers,
        pressed: bool,
        repeat: bool,
    ) -> bool {
        let view = self.pane_view(index);
        let input = KeyInput {
            text: text.to_string(),
            modifiers,
            pressed,
            repeat,
        };
        let (effects, consumed) = self.controller.borrow_mut().key(view.as_ref(), &input);
        self.apply_effects(effects);
        consumed || self.shortcut(text, modifiers, pressed, repeat)
    }

    /// A key nothing focused consumed.
    pub(crate) fn on_global_key(
        &self,
        text: &str,
        modifiers: Modifiers,
        pressed: bool,
        repeat: bool,
    ) -> bool {
        if pressed && matches!(text, keys::ESCAPE | keys::RETURN | keys::DELETE) {
            let index = self.state.read().focused_pane.0;
            return self.on_viewport_key(index, text, modifiers, pressed, repeat);
        }
        self.shortcut(text, modifiers, pressed, repeat)
    }

    /// The host shortcut table and plugin hotkeys.
    fn shortcut(&self, text: &str, modifiers: Modifiers, pressed: bool, repeat: bool) -> bool {
        let name = key_name(text);
        if !pressed {
            return self.release_momentary(&name);
        }
        let Some(ui) = self.ui() else {
            return false;
        };
        let primary = modifiers.primary;
        let shift = modifiers.shift;
        match (primary, name.as_str()) {
            (true, "z") if shift => {
                self.redo();
                return true;
            }
            (true, "z") => {
                self.undo();
                return true;
            }
            (true, "y") => {
                self.redo();
                return true;
            }
            (true, "w") if shift => {
                let _ = ui.hide();
                let _ = slint::quit_event_loop();
                return true;
            }
            (true, "w") => {
                ui.invoke_close_active_tab();
                return true;
            }
            (true, "o") => {
                ui.invoke_open_file_requested();
                return true;
            }
            (true, "Tab") => {
                self.state
                    .write()
                    .cycle_tab_in_focused_pane(if shift { -1 } else { 1 });
                self.refresh_tabs();
                return true;
            }
            _ => {}
        }
        if !(primary || modifiers.alt) {
            let handled = match name.as_str() {
                "f" => {
                    self.with_focused_viewport(crate::callbacks::frame_active_viewport);
                    true
                }
                "m" => {
                    crate::callbacks::toggle_minimap_visibility(&ui, &self.state);
                    true
                }
                "+" | "=" => {
                    self.with_focused_viewport(|state| {
                        crate::callbacks::zoom_active_viewport(
                            state,
                            crate::callbacks::ACTION_ZOOM_FACTOR,
                        )
                    });
                    true
                }
                "-" => {
                    self.with_focused_viewport(|state| {
                        crate::callbacks::zoom_active_viewport(
                            state,
                            1.0 / crate::callbacks::ACTION_ZOOM_FACTOR,
                        )
                    });
                    true
                }
                "n" => {
                    set_tool(self, ToolChoice::Navigate);
                    true
                }
                "r" => {
                    set_tool(self, ToolChoice::Roi);
                    true
                }
                "d" => {
                    set_tool(self, ToolChoice::Measure(self.measure_mode()));
                    true
                }
                "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown" => {
                    let step = if shift { 0.5 } else { 0.15 };
                    self.pan_focused_by_fraction(match name.as_str() {
                        "ArrowLeft" => (-step, 0.0),
                        "ArrowRight" => (step, 0.0),
                        "ArrowUp" => (0.0, -step),
                        _ => (0.0, step),
                    });
                    true
                }
                "i" => {
                    let pane = self.state.read().focused_pane;
                    crate::slide_info::open_for_pane(&ui, &self.state, pane);
                    true
                }
                "F11" => {
                    let window = ui.window();
                    window.set_fullscreen(!window.is_fullscreen());
                    true
                }
                "PageDown" => {
                    crate::callbacks::cycle_series(
                        &ui,
                        &self.state,
                        &self.tile_cache,
                        &self.render_timer,
                        1,
                    );
                    true
                }
                "PageUp" => {
                    crate::callbacks::cycle_series(
                        &ui,
                        &self.state,
                        &self.tile_cache,
                        &self.render_timer,
                        -1,
                    );
                    true
                }
                _ => false,
            };
            if handled {
                return true;
            }
        }
        if repeat {
            if let Some(entry) = self
                .momentary
                .borrow_mut()
                .iter_mut()
                .find(|m| m.key == name)
            {
                entry.saw_repeat = true;
                return true;
            }
            return false;
        }
        let target =
            self.shared
                .contributions
                .lock()
                .hotkey_target(&name, primary, shift, modifiers.alt);
        match target {
            Some(HotkeyTarget::Tool {
                plugin_id,
                tool_id,
                momentary: _,
            }) => {
                let previous = self.controller.borrow().tool().clone();
                let already = previous
                    .plugin_tool()
                    .is_some_and(|(owner, spec)| *owner == plugin_id && spec.tool_id == tool_id);
                if already {
                    set_tool(self, ToolChoice::Navigate);
                } else {
                    self.momentary.borrow_mut().push(Momentary {
                        key: name,
                        previous,
                        pressed_at: Instant::now(),
                        saw_repeat: false,
                    });
                    self.select_plugin_tool(&plugin_id, &tool_id);
                }
                true
            }
            Some(HotkeyTarget::Button {
                plugin_id,
                button_id,
            }) => {
                self.post(
                    EventTarget::Plugin(plugin_id),
                    HostEvent::ToolbarButtonClicked { button_id },
                );
                true
            }
            None => !primary && !modifiers.alt && self.magnification_key(&name),
        }
    }

    /// Digit keys zoom to objective-equivalent magnifications (1.25×, 2.5×,
    /// 5×, 10×, 20×, 40×; 0 = 1:1) unless a plugin claims them.
    fn magnification_key(&self, name: &str) -> bool {
        let magnification = match name {
            "1" => Some(1.25),
            "2" => Some(2.5),
            "3" => Some(5.0),
            "4" => Some(10.0),
            "5" => Some(20.0),
            "6" => Some(40.0),
            "0" => None,
            _ => return false,
        };
        let result = {
            let mut state = self.state.write();
            let result = crate::callbacks::magnify_active_viewport(&mut state, magnification);
            if matches!(result, Ok(true)) {
                state.mirror_locked_viewports_from_active();
                state.request_render();
            }
            result
        };
        match result {
            Ok(_) => self.request_render(),
            Err(message) => self.status(&message),
        }
        true
    }

    /// Restores the previous tool after a held tool hotkey is released.
    fn release_momentary(&self, name: &str) -> bool {
        let entry = {
            let mut momentary = self.momentary.borrow_mut();
            let Some(index) = momentary.iter().position(|m| m.key == name) else {
                return false;
            };
            momentary.remove(index)
        };
        let held = entry.saw_repeat || entry.pressed_at.elapsed() >= MOMENTARY_HOLD;
        if held {
            set_tool(self, entry.previous);
        }
        true
    }

    fn with_focused_viewport(&self, f: impl FnOnce(&mut crate::state::AppState) -> bool) {
        {
            let mut state = self.state.write();
            if f(&mut state) {
                state.mirror_locked_viewports_from_active();
                state.request_render();
            }
        }
        self.request_render();
    }

    fn pan_focused_by_fraction(&self, (fx, fy): (f64, f64)) {
        {
            let mut state = self.state.write();
            if let Some(viewport) = state.active_viewport_mut() {
                let (w, h) = (viewport.viewport.width, viewport.viewport.height);
                viewport.viewport.pan(-fx * w, -fy * h);
                state.mirror_locked_viewports_from_active();
                state.request_render();
            }
        }
        self.request_render();
    }

    pub(super) fn refresh_tabs(&self) {
        if let Some(ui) = self.ui() {
            let state = self.state.read();
            crate::refresh_tab_ui(&ui, &state);
        }
        self.request_render();
    }

    // -- tools -------------------------------------------------------------------

    pub(crate) fn select_host_tool(&self, tool: ToolType) {
        let choice = match tool {
            ToolType::Navigate => ToolChoice::Navigate,
            ToolType::RegionOfInterest => ToolChoice::Roi,
            ToolType::Measure => ToolChoice::Measure(self.measure_mode()),
            ToolType::Plugin => return,
        };
        // Clicking the active tool's button again returns to Navigate.
        let current = self.controller.borrow().tool().clone();
        let same = std::mem::discriminant(&current) == std::mem::discriminant(&choice);
        set_tool(self, if same { ToolChoice::Navigate } else { choice });
    }

    pub(crate) fn select_plugin_tool(&self, plugin_id: &str, tool_id: &str) {
        let spec = self
            .shared
            .contributions
            .lock()
            .tool(plugin_id, tool_id)
            .map(|entry| entry.spec.clone());
        let Some(spec) = spec else {
            return;
        };
        set_tool(
            self,
            ToolChoice::Plugin {
                plugin_id: Arc::from(plugin_id),
                spec,
            },
        );
    }

    /// Activates or toggles a plugin tool from its toolbar button.
    pub(crate) fn toggle_plugin_tool(&self, plugin_id: &str, tool_id: &str) {
        let active = self
            .controller
            .borrow()
            .tool()
            .plugin_tool()
            .is_some_and(|(owner, spec)| &**owner == plugin_id && spec.tool_id == tool_id);
        if active {
            set_tool(self, ToolChoice::Navigate);
        } else {
            self.select_plugin_tool(plugin_id, tool_id);
        }
    }

    pub(super) fn after_tool_change(&self) {
        if let Some(ui) = self.ui() {
            let tool = match self.controller.borrow().tool() {
                ToolChoice::Navigate => ToolType::Navigate,
                ToolChoice::Roi => ToolType::RegionOfInterest,
                ToolChoice::Measure(_) => ToolType::Measure,
                ToolChoice::Plugin { .. } => ToolType::Plugin,
            };
            ui.set_current_tool(tool);
        }
        self.refresh_toolbar();
        self.redraw_overlays();
    }

    // -- effects -------------------------------------------------------------------

    pub(crate) fn apply_effects(&self, effects: Vec<Effect>) {
        let mut render = false;
        for effect in effects {
            match effect {
                Effect::StartPan { pane, x, y } => {
                    self.with_pane_viewport(pane, |vs| vs.start_drag(x, y));
                    self.overlays
                        .borrow_mut()
                        .set_cursor(pane, PaneCursor::Grabbing);
                    render = true;
                }
                Effect::Pan { pane, x, y } => {
                    self.with_pane_viewport(pane, |vs| vs.drag_to(x, y));
                    render = true;
                }
                Effect::EndPan { pane } => {
                    self.with_pane_viewport(pane, |vs| vs.end_drag());
                    let cursor = cursor_of(match self.controller.borrow().tool() {
                        ToolChoice::Navigate => Cursor::Default,
                        ToolChoice::Roi | ToolChoice::Measure(_) => Cursor::Crosshair,
                        ToolChoice::Plugin { spec, .. } => spec.cursor,
                    });
                    self.overlays.borrow_mut().set_cursor(pane, cursor);
                    render = true;
                }
                Effect::PanBy { pane, dx, dy } => {
                    self.with_pane_viewport(pane, |vs| vs.viewport.pan(-dx, -dy));
                    render = true;
                }
                Effect::Zoom { pane, factor, x, y } => {
                    self.with_pane_viewport(pane, |vs| vs.zoom_at(factor, x, y));
                    render = true;
                }
                Effect::ContextMenu {
                    pane,
                    window_x,
                    window_y,
                    image,
                    hits,
                } => self.open_viewport_menu(pane, window_x, window_y, image, hits),
                Effect::Plugin { plugin_id, event } => {
                    let target = if &*plugin_id == BROADCAST {
                        EventTarget::All
                    } else {
                        EventTarget::Plugin(plugin_id)
                    };
                    self.post(target, event);
                }
                Effect::SetRoi { file_id, rect } => {
                    self.apply_host_edit(file_id, HostEdit::SetRoi(rect), true)
                }
                Effect::AddMeasurement {
                    file_id,
                    measurement,
                } => self.apply_host_edit(file_id, HostEdit::Add(measurement), true),
                Effect::UpdateMeasurement { file_id, id, shape } => {
                    self.apply_host_edit(file_id, HostEdit::Update { id, shape }, true)
                }
                Effect::RemoveMeasurements { file_id, ids } => {
                    self.apply_host_edit(file_id, HostEdit::Remove(ids), true)
                }
                Effect::ToolChanged => self.after_tool_change(),
                Effect::Cursor { pane, cursor } => {
                    self.overlays
                        .borrow_mut()
                        .set_cursor(pane, cursor_of(cursor));
                    render = true;
                }
                Effect::Tooltip(request) => {
                    if let Some(ui) = self.ui() {
                        match request {
                            Some(tooltip) => {
                                ui.set_hover_tooltip_text(tooltip.text.into());
                                ui.set_hover_tooltip_x(tooltip.window_x as f32);
                                ui.set_hover_tooltip_y(tooltip.window_y as f32);
                            }
                            None => ui.set_hover_tooltip_text("".into()),
                        }
                    }
                }
                Effect::Redraw => {
                    self.overlays.borrow_mut().bump_scene();
                    render = true;
                }
            }
        }
        if render {
            self.request_render();
        }
    }

    /// Runs `f` on a pane's viewport, mirroring locked panes.
    fn with_pane_viewport(&self, pane: usize, f: impl FnOnce(&mut eov_common::ViewportState)) {
        let mut state = self.state.write();
        let pane_id = PaneId(pane);
        if state.focused_pane != pane_id {
            state.set_focused_pane(pane_id);
        }
        let changed = match state.active_viewport_mut() {
            Some(viewport) => {
                f(viewport);
                true
            }
            None => false,
        };
        let mirrored = state.mirror_locked_viewports_from_active();
        if changed || mirrored {
            state.request_render();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names() {
        assert_eq!(key_name("A"), "a");
        assert_eq!(key_name("\u{1b}"), "Escape");
        assert_eq!(key_name("\u{F705}"), "F2");
        assert_eq!(key_name("\u{F702}"), "ArrowLeft");
        assert_eq!(key_name("+"), "+");
    }

    #[test]
    fn primary_modifier_follows_the_platform() {
        let m = modifiers_from_slint(false, true, false, false);
        assert!(m.primary);
        if cfg!(target_os = "macos") {
            assert!(m.meta && !m.control);
        } else {
            assert!(m.control && !m.meta);
        }
    }
}
