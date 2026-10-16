//! The viewer: `version`, `slides`, `panes`, `open`, `close`, `view.get`
//! and `view.set`. Results use the plugin API's vocabulary
//! ([`eov_plugin_api::protocol::FileInfo`], `ViewInfo`).

use super::params::{self, CoordinateParam, ZoomParam};
use super::{CallResult, Claim, ErrorKind, NoParams, RpcError, Session};
use crate::host_ui::UiHost;
use crate::launch::ViewRequest;
use crate::plugin_runtime::shared::Published;
use crate::state::{AppState, PaneId};
use eov_plugin_api::protocol::{FileInfo, ViewInfo};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

pub(super) fn version(_: NoParams) -> CallResult {
    Ok(json!({
        "eov": crate::version::BUILD_VERSION,
        "protocol": super::PROTOCOL_VERSION,
        "plugin_api": eov_plugin_api::abi::API_VERSION,
        "methods": super::METHODS,
        "limits": {
            "max_call_bytes": crate::instance::MAX_SCRIPT_REQUEST_BYTES,
            "max_answer_bytes": crate::instance::MAX_SCRIPT_REPLY_BYTES,
            "max_region_pixels": super::pixels::MAX_REGION_PIXELS,
            "max_region_side": super::pixels::MAX_REGION_SIDE,
            "max_scripts": super::MAX_SESSIONS,
            "max_heavy_calls": super::HEAVY_CALLS,
        },
    }))
}

/// Shows the state's panes and tabs in the window and renders (UI thread,
/// no lock held).
pub(super) fn refresh_window(host: &UiHost) {
    if let Some(ui) = host.ui() {
        let state = host.state.read();
        ui.set_split_enabled(state.split_enabled);
        ui.set_focused_pane(state.focused_pane.as_index());
        crate::update_tabs(&ui, &state);
    }
    host.request_render();
}

/// The objective-equivalent magnification of a slide's full resolution:
/// its objective, or about 10 µm ÷ its pixel size.
pub(super) fn native_magnification(info: &FileInfo) -> Option<f64> {
    crate::launch::native_magnification(
        info.objective_power,
        crate::interaction::Calibration::new(info.mpp_x, info.mpp_y).average(),
    )
}

/// A pane's view: the plugin API's `ViewInfo` plus the objective-equivalent
/// `magnification` (`null` when the slide records neither its objective
/// nor its pixel size).
fn view_json(published: &Published, view: &ViewInfo) -> Value {
    let magnification = published
        .files
        .get(&view.file_id)
        .and_then(|file| native_magnification(&file.info))
        .map(|native| native * view.zoom);
    let mut value = serde_json::to_value(view).unwrap_or_default();
    value["magnification"] = json!(magnification);
    value
}

/// The view of pane `pane` (default: the focused one), as last published.
fn published_view(published: &Published, pane: Option<u32>) -> CallResult {
    let uid = pane
        .or(published.snapshot.focused_pane)
        .ok_or_else(|| RpcError::not_found("there is no pane"))?;
    let info = published
        .snapshot
        .panes
        .iter()
        .find(|info| info.pane == uid)
        .ok_or_else(|| RpcError::not_found(format!("there is no pane {uid} (see panes)")))?;
    let view = info
        .view
        .as_ref()
        .ok_or_else(|| RpcError::not_found(format!("pane {uid} shows no slide")))?;
    Ok(view_json(published, view))
}

pub(super) fn slides(session: &Session, _: NoParams) -> CallResult {
    let published = session.published();
    let mut files: Vec<_> = published.files.values().collect();
    files.sort_by_key(|file| file.info.file_id);
    let slides = files
        .into_iter()
        .map(|file| {
            let mut info = file.info.clone();
            if info.identity.is_none() {
                // Not known to the viewer yet: computing it samples the
                // slide's files, here rather than on the UI thread.
                info.identity = session
                    .open_slide(info.file_id)
                    .ok()
                    .and_then(|wsi| wsi.identity().ok())
                    .map(|identity| identity.key().to_string());
            }
            let panes: Vec<u32> = published
                .snapshot
                .panes
                .iter()
                .filter(|pane| pane.file_id == Some(info.file_id))
                .map(|pane| pane.pane)
                .collect();
            let mut value = serde_json::to_value(&info).unwrap_or_default();
            value["file_size"] = json!(file.file_size);
            value["native_magnification"] = json!(native_magnification(&info));
            value["mpp_set_by_hand"] = json!(file.mpp_set_by_hand);
            value["panes"] = json!(panes);
            if file.path.to_str().is_none() {
                value["os_path"] = json!(crate::recent::encode_os(&file.path));
            }
            value
        })
        .collect();
    Ok(Value::Array(slides))
}

pub(super) fn panes(session: &Session, _: NoParams) -> CallResult {
    let published = session.published();
    let snapshot = &published.snapshot;
    let panes = snapshot
        .panes
        .iter()
        .map(|pane| {
            json!({
                "pane": pane.pane,
                "index": pane.index,
                "focused": Some(pane.pane) == snapshot.focused_pane,
                "file_id": pane.file_id,
                "tabs": published.tabs.get(&pane.pane).cloned().unwrap_or_default(),
                "view": pane.view.as_ref().map(|view| view_json(&published, view)),
            })
        })
        .collect();
    Ok(Value::Array(panes))
}

/// Where a slide opens (`open`'s `view`).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ViewParams {
    #[serde(default)]
    x: Option<CoordinateParam>,
    #[serde(default)]
    y: Option<CoordinateParam>,
    #[serde(default)]
    zoom: Option<ZoomParam>,
    #[serde(default)]
    magnification: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Open {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    os_path: Option<String>,
    /// The pane to show it in (default: the focused one).
    #[serde(default)]
    pane: Option<u32>,
    /// Show it in a new pane on the right.
    #[serde(default)]
    new_pane: bool,
    #[serde(default)]
    view: Option<ViewParams>,
}

pub(super) fn open(session: &Session, params: Open) -> CallResult {
    let path = params::path_from(params.path, params.os_path)?
        .ok_or_else(|| RpcError::invalid("give the slide's path"))?;
    // Off the UI thread: a hung network mount must not freeze the window.
    let path = params::slide_file(&path)?;
    if params.new_pane && params.pane.is_some() {
        return Err(RpcError::invalid("give pane or new_pane, not both"));
    }
    let view = match &params.view {
        Some(view) => params::view_request(
            view.x.as_ref(),
            view.y.as_ref(),
            view.zoom.as_ref(),
            view.magnification,
        )?,
        None => ViewRequest::default(),
    };
    let (pane, new_pane) = (params.pane, params.new_pane);
    let (done, opened) = mpsc::sync_channel(1);
    // A slide that opens after the script gave up opens, but its view is
    // left alone.
    let claim = Claim::default();
    let waiting = claim.clone();
    session.on_ui(move |host| start_open(host, path, pane, new_pane, view, waiting, done))?;
    let closing = || RpcError::new(ErrorKind::Unavailable, "eov is closing");
    match opened.recv_timeout(session.timing().open) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) if claim.abandon() => Err(RpcError::new(
            ErrorKind::Busy,
            format!(
                "the slide did not open within {} s (it may still open, as asked, but its \
                 view will not be set)",
                session.timing().open.as_secs()
            ),
        )),
        // It is being shown right now.
        Err(mpsc::RecvTimeoutError::Timeout) => opened.recv().unwrap_or_else(|_| Err(closing())),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(closing()),
    }
}

/// Shows `path` in a pane (UI thread); `done` hears once it is shown,
/// unless the script gave up (`claim`).
fn start_open(
    host: &UiHost,
    path: PathBuf,
    pane: Option<u32>,
    new_pane: bool,
    view: ViewRequest,
    claim: Claim,
    done: mpsc::SyncSender<CallResult>,
) -> Result<(), RpcError> {
    let ui = host
        .ui()
        .ok_or_else(|| RpcError::new(ErrorKind::Unavailable, "the window is closed"))?;
    let target = {
        let mut state = host.state.write();
        let target = if new_pane {
            let index = state.panes.len();
            let source = state.pane_uid(PaneId(index - 1));
            let created = state
                .insert_pane(index)
                .ok_or_else(|| RpcError::refused(crate::state::MAX_PANES_MESSAGE))?;
            crate::insert_pane_ui_state(state.pane_uid(created), source);
            created
        } else {
            params::pane(&state, pane)?
        };
        // Slides open in the focused pane.
        state.set_focused_pane(target);
        state.pane_uid(target).unwrap_or_default()
    };
    let existing = host.state.read().find_tab_by_path(&path);
    if let Some((source, tab_id)) = existing {
        {
            let mut state = host.state.write();
            let target_pane = state.pane_by_uid(target).unwrap_or(source);
            if source == target_pane || (pane.is_none() && !new_pane) {
                // Its own tab, where it is.
                state.activate_tab_in_pane(source, tab_id);
            } else {
                let copy = state.duplicate_tab_to_pane(tab_id, target_pane);
                state.activate_tab_in_pane(target_pane, copy);
            }
        }
        refresh_window(host);
        if claim.take() {
            let _ = done.send(shown(host, &path, target, &view));
        }
        return Ok(());
    }
    let shown_path = path.clone();
    crate::file_ops::when_open(
        &path,
        Box::new(move |opened| {
            if !claim.take() {
                return;
            }
            let result = opened
                .map_err(|error| RpcError::failed(format!("could not open the slide: {error}")))
                .and_then(|_| {
                    crate::host_ui::with_host(|host| shown(host, &shown_path, target, &view))
                        .unwrap_or_else(|| {
                            Err(RpcError::new(ErrorKind::Unavailable, "eov is closing"))
                        })
                });
            let _ = done.send(result);
        }),
    );
    crate::open_file(&ui, &host.state, &host.tile_cache, &host.render_timer, path);
    refresh_window(host);
    Ok(())
}

/// The answer to `open` once the slide is shown: moves the view where
/// asked (UI thread).
fn shown(host: &UiHost, path: &Path, target: u32, view: &ViewRequest) -> CallResult {
    let (pane, file_id, warnings) = {
        let mut state = host.state.write();
        let (_, tab_id) = state
            .find_tab_by_path(path)
            .ok_or_else(|| RpcError::failed("the slide was closed while it opened"))?;
        let file_id = state.resolve_tab_file_id(tab_id);
        let pane = state
            .pane_by_uid(target)
            .filter(|pane| state.active_file_id_for_pane(*pane) == Some(file_id))
            .or_else(|| params::pane_showing(&state, file_id))
            .ok_or_else(|| RpcError::failed("the slide is not shown in a pane"))?;
        let warnings = if view.is_empty() {
            Vec::new()
        } else {
            apply_view(&mut state, pane, view, None, None, false)?
        };
        (state.pane_uid(pane), file_id, warnings)
    };
    host.request_render();
    // Reads that follow see the slide.
    host.publish_now();
    Ok(json!({
        "file_id": file_id,
        "pane": pane,
        "view": published_view(&host.shared.published(), pane).ok(),
        "warnings": warnings,
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Close {
    slide: i64,
}

pub(super) fn close(session: &Session, params: Close) -> CallResult {
    session.on_ui(move |host| {
        let file_id = {
            let mut state = host.state.write();
            let file_id = params::slide(&state, Some(params.slide))?;
            state.close_file(file_id);
            if state.panes.iter().all(|pane| pane.tabs.is_empty()) {
                state.create_home_tab();
            }
            file_id
        };
        refresh_window(host);
        host.publish_now();
        Ok(json!({ "closed": file_id }))
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ViewGet {
    #[serde(default)]
    pane: Option<u32>,
}

pub(super) fn view_get(session: &Session, params: ViewGet) -> CallResult {
    published_view(&session.published(), params.pane)
}

fn shows_no_slide(state: &AppState, pane: PaneId) -> RpcError {
    RpcError::not_found(format!(
        "pane {} shows no slide",
        state.pane_uid(pane).unwrap_or_default()
    ))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ViewSet {
    #[serde(default)]
    pane: Option<u32>,
    #[serde(default)]
    x: Option<CoordinateParam>,
    #[serde(default)]
    y: Option<CoordinateParam>,
    #[serde(default)]
    zoom: Option<ZoomParam>,
    #[serde(default)]
    magnification: Option<f64>,
    /// Degrees clockwise, any angle.
    #[serde(default)]
    rotation: Option<f64>,
    #[serde(default)]
    flipped: Option<bool>,
    /// Glide there as the viewer does (the answer shows where it starts).
    #[serde(default)]
    animate: bool,
}

pub(super) fn view_set(session: &Session, params: ViewSet) -> CallResult {
    let request = params::view_request(
        params.x.as_ref(),
        params.y.as_ref(),
        params.zoom.as_ref(),
        params.magnification,
    )?;
    if request.is_empty() && params.rotation.is_none() && params.flipped.is_none() {
        return Err(RpcError::invalid(
            "nothing to set: give x and y, zoom or magnification, rotation or flipped",
        ));
    }
    let (rotation, flipped, animate) = (params.rotation, params.flipped, params.animate);
    session.on_ui(move |host| {
        let (pane, warnings) = {
            let mut state = host.state.write();
            let pane = params::pane(&state, params.pane)?;
            let warnings = apply_view(&mut state, pane, &request, rotation, flipped, animate)?;
            (state.pane_uid(pane), warnings)
        };
        host.request_render();
        // Reads that follow see the new view.
        host.publish_now();
        let mut view = published_view(&host.shared.published(), pane)?;
        view["warnings"] = json!(warnings);
        Ok(view)
    })
}

/// Moves `pane`'s view to `request`, turned as asked (UI thread). Returns
/// what could not be applied (a position in µm on a slide without a pixel
/// size, ...).
pub(super) fn apply_view(
    state: &mut AppState,
    pane: PaneId,
    request: &ViewRequest,
    rotation: Option<f64>,
    flipped: Option<bool>,
    animate: bool,
) -> Result<Vec<String>, RpcError> {
    let file_id = state
        .active_file_id_for_pane(pane)
        .ok_or_else(|| shows_no_slide(state, pane))?;
    let file = state
        .get_file_mut(file_id)
        .ok_or_else(|| RpcError::not_found("the slide was closed"))?;
    let scale = crate::launch::SlideScale::of_open_file(file);
    let viewport = &mut file
        .pane_state_mut(pane)
        .ok_or_else(|| RpcError::not_found("the pane has no view yet"))?
        .viewport;
    let orientation = params::orientation(rotation, flipped, viewport.viewport.orientation)?;
    viewport.set_orientation(orientation);
    let mut warnings = Vec::new();
    if !request.is_empty() {
        let current = (
            viewport.viewport.center.x,
            viewport.viewport.center.y,
            viewport.viewport.zoom,
        );
        let view = crate::launch::resolve_view(request, &scale, current, viewport.fit_view());
        if animate {
            viewport.smooth_set_center_zoom(view.center_x, view.center_y, view.zoom);
        } else {
            viewport.set_center_zoom(view.center_x, view.center_y, view.zoom);
        }
        warnings = view.warnings;
    }
    if pane == state.active_viewport_pane() {
        state.mirror_locked_viewports_from_active();
    }
    state.request_render();
    Ok(warnings)
}
