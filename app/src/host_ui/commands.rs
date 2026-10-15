//! Applying plugin commands on the UI thread.

use super::UiHost;
use super::input::set_tool;
use crate::interaction::ToolChoice;
use crate::plugin_runtime::EventTarget;
use crate::plugin_runtime::shared::{Command, ViewCommand};
use crate::plugin_runtime::undo::{UndoAction, UndoEntry};
use crate::state::PaneId;
use eov_plugin_api::protocol::{
    FileDialogArgs, HostEvent, NotifyLevel, PaneId as PaneUid, SeriesAction,
};
use std::sync::Arc;

pub(super) fn apply(host: &UiHost, plugin_id: &Arc<str>, command: Command) {
    let name = host.plugin_name(plugin_id);
    match command {
        Command::Notify(args) => {
            host.notify(
                args.level,
                &args.message,
                args.detail.as_deref(),
                Some(&name),
            );
        }
        Command::Status(text) => {
            host.notices.borrow_mut().set_status(plugin_id, text);
            host.refresh_notices();
        }
        Command::OpenFile(path) => {
            if let Some(ui) = host.ui() {
                crate::open_file(&ui, &host.state, &host.tile_cache, &host.render_timer, path);
            }
        }
        Command::ContributionsChanged => {
            host.runtime.borrow().rebuild_filters();
            host.state.write().bump_filter_revision();
            host.refresh_toolbar();
            host.overlays.borrow_mut().invalidate_all();
            // The active tool may have been unregistered.
            let stale =
                host.controller
                    .borrow()
                    .tool()
                    .plugin_tool()
                    .is_some_and(|(owner, spec)| {
                        host.shared
                            .contributions
                            .lock()
                            .tool(owner, &spec.tool_id)
                            .is_none()
                    });
            if stale {
                set_tool(host, ToolChoice::Navigate);
            }
            host.request_render();
        }
        Command::ActivateTool(tool_id) => host.select_plugin_tool(plugin_id, &tool_id),
        Command::DeactivateTools => {
            let owned = host
                .controller
                .borrow()
                .tool()
                .plugin_tool()
                .is_some_and(|(owner, _)| owner == plugin_id);
            if owned {
                set_tool(host, ToolChoice::Navigate);
            }
        }
        Command::Surface(command) => {
            host.surfaces.borrow_mut().apply(host, plugin_id, command);
        }
        Command::Confirm { request_id, args } => {
            host.surfaces
                .borrow_mut()
                .queue_confirm(host, plugin_id, request_id, args);
        }
        Command::FileDialog {
            request_id,
            args,
            save,
        } => {
            let paths = run_file_dialog(&args, save);
            host.post(
                EventTarget::Plugin(Arc::clone(plugin_id)),
                HostEvent::DialogResult { request_id, paths },
            );
        }
        Command::View(command) => apply_view(host, command),
        Command::CyclePane(offset) => {
            host.state.write().cycle_focused_pane(offset);
            host.refresh_tabs();
        }
        Command::CycleTab(offset) => {
            host.state.write().cycle_tab_in_focused_pane(offset);
            host.refresh_tabs();
        }
        Command::Series(action) => {
            if let Some(ui) = host.ui() {
                match action {
                    SeriesAction::Toggle => crate::callbacks::toggle_series_bar_requested(
                        &ui,
                        &host.state,
                        &host.tile_cache,
                        &host.render_timer,
                    ),
                    SeriesAction::Next | SeriesAction::Previous => {
                        let offset = if action == SeriesAction::Next { 1 } else { -1 };
                        host.state.write().cycle_series_selection(offset);
                        host.refresh_tabs();
                    }
                    SeriesAction::OpenSelected => {
                        let path = host.state.read().selected_series_entry_path();
                        if let Some(path) = path {
                            crate::callbacks::activate_series_entry_path(
                                &ui,
                                &host.state,
                                &host.tile_cache,
                                &host.render_timer,
                                path,
                            );
                        }
                    }
                }
            }
        }
        Command::Render => {
            host.state.write().request_render();
            host.request_render();
        }
        Command::UndoPush(args) => {
            host.undo.borrow_mut().push(UndoEntry {
                file_id: args.file_id,
                label: args.label,
                action: UndoAction::Plugin {
                    plugin_id: Arc::clone(plugin_id),
                    entry_id: args.entry_id,
                },
            });
            host.refresh_undo_ui();
        }
        Command::UndoClear(args) => {
            host.undo.borrow_mut().clear_plugin(plugin_id, args.file_id);
            host.refresh_undo_ui();
        }
        Command::JobStart { job_id, args } => {
            host.notices.borrow_mut().start_job(plugin_id, job_id, args);
            host.refresh_notices();
        }
        Command::JobUpdate(args) => {
            host.notices.borrow_mut().update_job(plugin_id, args);
            host.refresh_notices();
        }
        Command::JobFinish(args) => {
            let failed = host.notices.borrow_mut().finish_job(plugin_id, &args);
            if let (Some(title), Some(error)) = (failed, &args.error) {
                host.notify(
                    NotifyLevel::Error,
                    &format!("{title} failed"),
                    Some(error),
                    Some(&name),
                );
            }
            host.refresh_notices();
        }
        Command::Clipboard(text) => crate::clipboard::copy_text(text),
        Command::Post(token) => {
            host.post(
                EventTarget::Plugin(Arc::clone(plugin_id)),
                HostEvent::Posted { token },
            );
        }
        Command::OverlaysChanged => host.redraw_overlays(),
    }
}

fn run_file_dialog(args: &FileDialogArgs, save: bool) -> Vec<String> {
    let mut dialog = rfd::FileDialog::new();
    if let Some(title) = &args.title {
        dialog = dialog.set_title(title);
    }
    for filter in &args.filters {
        let extensions: Vec<&str> = filter.extensions.iter().map(String::as_str).collect();
        dialog = dialog.add_filter(&filter.name, &extensions);
    }
    let to_strings = |paths: Vec<std::path::PathBuf>| {
        paths
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    };
    if save {
        if let Some(name) = &args.default_name {
            dialog = dialog.set_file_name(name);
        }
        return to_strings(dialog.save_file().into_iter().collect());
    }
    match (args.directory, args.multiple) {
        (true, true) => to_strings(dialog.pick_folders().unwrap_or_default()),
        (true, false) => to_strings(dialog.pick_folder().into_iter().collect()),
        (false, true) => to_strings(dialog.pick_files().unwrap_or_default()),
        (false, false) => to_strings(dialog.pick_file().into_iter().collect()),
    }
}

/// Resolves a plugin pane id (stable uid) to a pane position.
fn resolve_pane(state: &crate::state::AppState, pane: Option<PaneUid>) -> Option<PaneId> {
    match pane {
        Some(uid) => state.pane_by_uid(uid),
        None => Some(state.focused_pane),
    }
}

fn apply_view(host: &UiHost, command: ViewCommand) {
    {
        let mut state = host.state.write();
        let pane = match &command {
            ViewCommand::Set(args) => args.pane,
            ViewCommand::Pan(args) => args.pane,
            ViewCommand::Zoom(args) => args.pane,
            ViewCommand::Fit(pane) => *pane,
            ViewCommand::Frame(args) => args.pane,
        };
        let Some(pane) = resolve_pane(&state, pane) else {
            return;
        };
        let Some(file_id) = state.active_file_id_for_pane(pane) else {
            return;
        };
        let Some(viewport) = state
            .get_file_mut(file_id)
            .and_then(|file| file.pane_state_mut(pane))
            .map(|pane_state| &mut pane_state.viewport)
        else {
            return;
        };
        match command {
            ViewCommand::Set(args) => {
                if args.animate {
                    viewport.smooth_set_center_zoom(args.center_x, args.center_y, args.zoom);
                } else {
                    viewport.set_center_zoom(args.center_x, args.center_y, args.zoom);
                }
            }
            ViewCommand::Pan(args) => viewport.viewport.pan(-args.dx, -args.dy),
            ViewCommand::Zoom(args) => {
                let (w, h) = (viewport.viewport.width, viewport.viewport.height);
                let x = args.anchor_x.unwrap_or(w / 2.0);
                let y = args.anchor_y.unwrap_or(h / 2.0);
                if args.animate {
                    viewport.zoom_at(args.factor, x, y);
                } else {
                    viewport.viewport.zoom_at(args.factor, x, y);
                }
            }
            ViewCommand::Fit(_) => viewport.smooth_fit_to_view(),
            ViewCommand::Frame(args) => {
                let padding = args.padding.unwrap_or(0.1).clamp(0.0, 0.45);
                let r = args.rect;
                let (px, py) = (r.width * padding, r.height * padding);
                viewport.smooth_frame_rect(
                    r.x - px,
                    r.y - py,
                    r.width + 2.0 * px,
                    r.height + 2.0 * py,
                );
            }
        }
        if state.focused_pane == pane {
            state.mirror_locked_viewports_from_active();
        }
        state.request_render();
    }
    host.request_render();
}
