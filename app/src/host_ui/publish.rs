//! Publishing host state to plugins and turning state changes into events.
//!
//! The renderer calls [`UiHost::after_frame`] with `AppState` locked; it only
//! compares, publishes and queues events, so it is cheap and never calls a
//! plugin.

use super::UiHost;
use crate::interaction::{Calibration, measure};
use crate::plugin_runtime::EventTarget;
use crate::plugin_runtime::shared::{Published, PublishedFile, SlideReader};
use crate::state::{AppState, OpenFile, PaneId};
use eov_common::WsiFile;
use eov_common::wsi::AlphaMode;
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::protocol::{
    FileId, FileInfo, HostEvent, HostSnapshot, LevelInfo, MeasurementInfo, PaneInfo, PropertyMap,
    Rect, ToolRef, ViewInfo,
};
use slint::ComponentHandle;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A view must be still this long before `ViewChanged` is sent.
const VIEW_SETTLE: Duration = Duration::from_millis(150);

struct ViewTrack {
    view: ViewInfo,
    changed_at: Instant,
    reported: bool,
}

#[derive(Default)]
pub(super) struct Tracker {
    files: BTreeMap<FileId, PathBuf>,
    active: (Option<u32>, Option<FileId>),
    views: HashMap<u32, ViewTrack>,
    properties: HashMap<FileId, Arc<PropertyMap>>,
    window_focused: Option<bool>,
    settle_timer_armed: bool,
    /// The slide each pane showed at the last frame (UI-10).
    pane_files: HashMap<u32, Option<FileId>>,
    pane_count: usize,
    /// Slide identities computed off the UI thread (`None`: unavailable).
    identities: HashMap<FileId, Option<String>>,
    /// Opened files whose `FileOpened` waits for their identity.
    pending_open: Vec<FileId>,
    /// Each file's region of interest at the last frame (`RoiChanged`).
    rois: HashMap<FileId, Option<Rect>>,
    /// Each pane's display settings at the last frame (`DisplayChanged`).
    displays: HashMap<u32, eov_plugin_api::protocol::DisplayAdjustments>,
}

/// How long `FileOpened` may wait for a slide identity.
const IDENTITY_WAIT: Duration = Duration::from_secs(5);

/// Reads plugin pixel requests on the plugin's thread. `WsiFile` clones
/// share the slide's self-healing handle, so no host lock is involved and a
/// bad region read by a plugin cannot poison the viewer (PLG-11, WSI-1).
struct FileReader {
    wsi: WsiFile,
}

impl SlideReader for FileReader {
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String> {
        let mode = if request.background & 0xFF == 0 {
            AlphaMode::Straight
        } else {
            AlphaMode::Premultiplied
        };
        let mut pixels = self
            .wsi
            .read_region_with(
                request.x,
                request.y,
                request.level,
                request.width,
                request.height,
                mode,
            )
            .map_err(|e| e.to_string())?;
        if mode == AlphaMode::Premultiplied {
            finish_alpha(&mut pixels, request.background);
        }
        Ok(pixels)
    }

    fn associated_image(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String> {
        let image = self
            .wsi
            .read_associated_image(name)
            .map_err(|e| e.to_string())?;
        Ok((image.width as u32, image.height as u32, image.pixels))
    }
}

/// Converts premultiplied RGBA to the plugin convention: straight alpha, or
/// composited onto `background` (`0xRRGGBBAA`, alpha ≠ 0).
fn finish_alpha(pixels: &mut [u8], background: u32) {
    let bg = [
        (background >> 24) as u8,
        (background >> 16) as u8,
        (background >> 8) as u8,
        background as u8,
    ];
    for px in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(px[3]);
        if bg[3] != 0 {
            let inv = 255 - a;
            for (channel, background) in px[..3].iter_mut().zip(bg) {
                *channel = (u32::from(*channel) + (u32::from(background) * inv + 127) / 255)
                    .min(255) as u8;
            }
            px[3] = 255;
        } else if a != 0 && a != 255 {
            for channel in &mut px[..3] {
                *channel = ((u32::from(*channel) * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
}

pub(crate) fn file_info(file: &OpenFile, identity: Option<&str>) -> FileInfo {
    let props = file.wsi.properties();
    FileInfo {
        file_id: FileId::from(file.id),
        path: file.path.to_string_lossy().into_owned(),
        filename: file.filename.clone(),
        width: props.width,
        height: props.height,
        levels: props
            .levels
            .iter()
            .map(|level| LevelInfo {
                level: level.level,
                width: level.width,
                height: level.height,
                downsample: level.downsample,
            })
            .collect(),
        mpp_x: calibration(file).mpp_x,
        mpp_y: calibration(file).mpp_y,
        objective_power: props.objective_power,
        vendor: props.vendor.clone(),
        scan_date: props.scan_date.clone(),
        bounds: props.bounds.map(|b| Rect {
            x: b.x as f64,
            y: b.y as f64,
            width: b.width as f64,
            height: b.height as f64,
        }),
        identity: identity.map(str::to_string),
        associated_images: file
            .wsi
            .associated_images()
            .iter()
            .map(|image| image.name.clone())
            .collect(),
        has_icc_profile: props.icc_profile_size.is_some(),
        stain: props.stain.clone(),
    }
}

/// The calibration of a slide: a manual pixel size when the user set one,
/// else what the file records.
pub(crate) fn calibration(file: &OpenFile) -> Calibration {
    if let Some(um) = file.calibration_override {
        return Calibration::new(Some(um), Some(um));
    }
    let props = file.wsi.properties();
    Calibration::new(props.mpp_x, props.mpp_y)
}

pub(crate) fn view_info(state: &AppState, pane: PaneId, scale_factor: f64) -> Option<ViewInfo> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let file = state.get_file(file_id)?;
    let viewport = &file.pane_state(pane)?.viewport.viewport;
    let bounds = viewport.bounds();
    Some(ViewInfo {
        pane: state.pane_uid(pane)?,
        file_id: FileId::from(file_id),
        center_x: viewport.center.x,
        center_y: viewport.center.y,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        scale_factor,
        rotation_deg: f64::from(viewport.orientation.degrees()),
        flipped: viewport.orientation.flipped(),
        visible: Rect {
            x: bounds.left,
            y: bounds.top,
            width: bounds.width(),
            height: bounds.height(),
        },
    })
}

fn measurement_infos(file: &OpenFile) -> Vec<MeasurementInfo> {
    let calibration = calibration(file);
    file.measurements
        .iter()
        .map(|m| {
            let value = measure::measure(m.mode, &m.shape, &calibration);
            MeasurementInfo {
                measurement_id: m.id.clone(),
                file_id: FileId::from(file.id),
                shape: m.shape.clone(),
                length_um: value.length_um,
                area_um2: value.area_um2,
                angle_deg: value.angle_deg,
            }
        })
        .collect()
}

impl UiHost {
    /// The slide identity key of an open file, once it is known.
    pub(crate) fn identity_key(&self, file_id: FileId) -> Option<String> {
        self.tracker
            .borrow()
            .identities
            .get(&file_id)
            .cloned()
            .flatten()
    }

    /// A slide's identity is known: publish it and announce the file.
    fn identity_ready(&self, file_id: FileId, identity: Option<String>) {
        // A manual pixel size saved for this slide applies from now on.
        if let Some(um) = identity
            .as_deref()
            .and_then(crate::calibration_store::lookup)
            && let Ok(id) = i32::try_from(file_id)
        {
            let mut state = self.state.write();
            if let Some(file) = state.get_file_mut(id) {
                file.calibration_override = Some(um);
                state.request_render();
            }
        }
        self.tracker
            .borrow_mut()
            .identities
            .insert(file_id, identity.clone());
        if let Some(key) = &identity {
            self.restore_host_annotations(file_id, key);
        }
        self.publish_now();
        self.announce_open(file_id);
    }

    /// Sends `FileOpened` once per file, when its identity is known or the
    /// wait timed out.
    fn announce_open(&self, file_id: FileId) {
        let waiting = {
            let mut tracker = self.tracker.borrow_mut();
            let before = tracker.pending_open.len();
            tracker.pending_open.retain(|id| *id != file_id);
            tracker.pending_open.len() != before
        };
        if !waiting {
            return;
        }
        let info = self
            .shared
            .published()
            .files
            .get(&file_id)
            .map(|file| file.info.clone());
        if let Some(file) = info {
            self.post(EventTarget::All, HostEvent::FileOpened { file });
        }
    }

    /// Publishes immediately (startup, after commands).
    pub(crate) fn publish_now(&self) {
        let state = self.state.read();
        self.after_frame(&state);
    }

    /// Publishes the state plugins see and queues change events. Called by
    /// the renderer with `AppState` locked.
    pub(crate) fn after_frame(&self, state: &AppState) {
        let scale_factor = self
            .ui()
            .map(|ui| f64::from(ui.window().scale_factor()))
            .unwrap_or(1.0);
        let mut tracker = self.tracker.borrow_mut();
        let mut events: Vec<(EventTarget, HostEvent)> = Vec::new();

        // Files opened and closed.
        let mut files = HashMap::new();
        let mut current: BTreeMap<FileId, PathBuf> = BTreeMap::new();
        for file in &state.open_files {
            let id = FileId::from(file.id);
            current.insert(id, file.path.clone());
            let properties = tracker
                .properties
                .entry(id)
                .or_insert_with(|| Arc::new(file.wsi.property_map().clone()))
                .clone();
            let identity = tracker.identities.get(&id).cloned().flatten();
            let info = file_info(file, identity.as_deref());
            match tracker.rois.insert(id, file.roi) {
                Some(previous) if previous != file.roi => events.push((
                    EventTarget::All,
                    HostEvent::RoiChanged {
                        file_id: id,
                        roi: file.roi,
                    },
                )),
                _ => {}
            }
            files.insert(
                id,
                PublishedFile {
                    info,
                    properties,
                    roi: file.roi,
                    measurements: measurement_infos(file),
                },
            );
        }
        let opened: Vec<FileId> = current
            .keys()
            .filter(|id| !tracker.files.contains_key(id))
            .copied()
            .collect();
        let closed: Vec<FileId> = tracker
            .files
            .keys()
            .filter(|id| !current.contains_key(id))
            .copied()
            .collect();
        for id in &opened {
            let Some(file) = state.get_file(*id as i32) else {
                continue;
            };
            self.shared.set_reader(
                *id,
                Arc::new(FileReader {
                    wsi: file.wsi.clone(),
                }),
            );
            self.ensure_host_layers(*id);
            // The identity samples the slide files: compute it on a worker
            // and announce the file once it is known.
            tracker.pending_open.push(*id);
            let wsi = file.wsi.clone();
            let file_id = *id;
            let spawned = std::thread::Builder::new()
                .name("eov-slide-identity".into())
                .spawn(move || {
                    let identity = match wsi.identity() {
                        Ok(identity) => Some(identity.key().to_string()),
                        Err(error) => {
                            tracing::warn!("No slide identity: {error}");
                            None
                        }
                    };
                    let _ = slint::invoke_from_event_loop(move || {
                        super::with_host(|host| host.identity_ready(file_id, identity));
                    });
                });
            if let Err(error) = spawned {
                tracing::warn!("Cannot compute the slide identity: {error}");
            }
            slint::Timer::single_shot(IDENTITY_WAIT, move || {
                super::with_host(|host| host.announce_open(file_id));
            });
        }
        for id in &closed {
            tracker.identities.remove(id);
            tracker.rois.remove(id);
            tracker.pending_open.retain(|pending| pending != id);
            self.shared.remove_reader(*id);
            self.store.remove_file(*id);
            self.undo.borrow_mut().clear_file(*id);
            tracker.properties.remove(id);
            events.push((EventTarget::All, HostEvent::FileClosed { file_id: *id }));
        }
        if !closed.is_empty() {
            self.refresh_undo_ui_deferred();
        }
        tracker.files = current;

        // Panes and views.
        let mut panes = Vec::new();
        let now = Instant::now();
        let mut unsettled = false;
        for index in 0..state.panes.len() {
            let pane = PaneId(index);
            let Some(uid) = state.pane_uid(pane) else {
                continue;
            };
            let view = view_info(state, pane, scale_factor);
            if let Some(view) = view {
                let track = tracker.views.entry(uid).or_insert(ViewTrack {
                    view,
                    changed_at: now,
                    reported: false,
                });
                if track.view != view {
                    track.view = view;
                    track.changed_at = now;
                    track.reported = false;
                }
                if !track.reported {
                    if now.duration_since(track.changed_at) >= VIEW_SETTLE {
                        track.reported = true;
                        self.history.borrow_mut().settled(
                            uid,
                            crate::history::Visit {
                                file_id: view.file_id as i32,
                                center_x: view.center_x,
                                center_y: view.center_y,
                                zoom: view.zoom,
                                width: view.width,
                                height: view.height,
                            },
                        );
                        events.push((EventTarget::All, HostEvent::ViewChanged { view }));
                    } else {
                        unsettled = true;
                    }
                }
            } else {
                tracker.views.remove(&uid);
            }
            panes.push(PaneInfo {
                pane: uid,
                index: index as u32,
                file_id: state.active_file_id_for_pane(pane).map(FileId::from),
                view,
            });
        }
        let live: Vec<u32> = panes.iter().map(|p| p.pane).collect();
        tracker.views.retain(|uid, _| live.contains(uid));
        self.history.borrow_mut().retain_panes(&live);

        // A gesture in progress must never continue on another slide or pane
        // layout (UI-10).
        let mut cancel_all = tracker.pane_count != 0 && tracker.pane_count != panes.len();
        tracker.pane_count = panes.len();
        let mut changed_panes = Vec::new();
        for info in &panes {
            let previous = tracker.pane_files.insert(info.pane, info.file_id);
            if previous.is_some_and(|previous| previous != info.file_id) {
                changed_panes.push(info.index as usize);
            }
        }
        tracker.pane_files.retain(|uid, _| live.contains(uid));
        if changed_panes.len() > 1 {
            cancel_all = true;
        }

        let focused_uid = state.pane_uid(state.focused_pane);
        let active = (
            focused_uid,
            state
                .active_file_id_for_pane(state.focused_pane)
                .map(FileId::from),
        );
        if active != tracker.active {
            tracker.active = active;
            events.push((
                EventTarget::All,
                HostEvent::ActiveChanged {
                    pane: active.0,
                    file_id: active.1,
                },
            ));
        }

        // The main window or a plugin window.
        let window_focused = self.ui().is_some_and(|ui| {
            use slint::winit_030::WinitWindowAccessor;
            ui.window()
                .with_winit_window(|window| window.has_focus())
                .unwrap_or(true)
        }) || self
            .surfaces
            .try_borrow()
            .is_ok_and(|surfaces| surfaces.window_focused());
        if tracker.window_focused != Some(window_focused) {
            let first = tracker.window_focused.is_none();
            tracker.window_focused = Some(window_focused);
            if !first {
                events.push((
                    EventTarget::All,
                    HostEvent::WindowFocusChanged {
                        focused: window_focused,
                    },
                ));
            }
        }

        let active_tool = self.controller.try_borrow().ok().and_then(|controller| {
            controller
                .tool()
                .plugin_tool()
                .map(|(plugin_id, spec)| ToolRef {
                    plugin_id: Some(plugin_id.to_string()),
                    tool_id: spec.tool_id.clone(),
                })
        });
        let snapshot = HostSnapshot {
            host_version: crate::version::BUILD_VERSION.to_string(),
            window_focused,
            render_backend: format!("{:?}", state.render_backend).to_lowercase(),
            filtering_mode: format!("{:?}", state.filtering_mode).to_lowercase(),
            focused_pane: focused_uid,
            panes,
            files: {
                let mut infos: Vec<FileInfo> = files.values().map(|f| f.info.clone()).collect();
                infos.sort_by_key(|f| f.file_id);
                infos
            },
            active_tool,
        };
        // How each pane shows its slide, for `view.display`.
        let displays = (0..state.panes.len())
            .filter_map(|index| {
                let pane = PaneId(index);
                let uid = state.pane_uid(pane)?;
                let file = state.get_file(state.active_file_id_for_pane(pane)?)?;
                let hud = &file.pane_state(pane)?.hud;
                let color = crate::color::hud_color_info(file, pane);
                Some((
                    uid,
                    eov_plugin_api::protocol::DisplayAdjustments {
                        gamma: hud.gamma,
                        brightness: hud.brightness,
                        contrast: hud.contrast,
                        sharpness: hud.sharpness,
                        stain_normalization: match hud.stain_normalization {
                            eov_common::StainNormalization::None => None,
                            eov_common::StainNormalization::Macenko => Some("macenko".into()),
                            eov_common::StainNormalization::Vahadane => Some("vahadane".into()),
                        },
                        modified: color.modified,
                        badge: color.badge,
                    },
                ))
            })
            .collect::<HashMap<u32, eov_plugin_api::protocol::DisplayAdjustments>>();
        for (pane, display) in &displays {
            let previous = tracker.displays.insert(*pane, display.clone());
            if previous.is_some_and(|previous| previous != *display) {
                events.push((
                    EventTarget::All,
                    HostEvent::DisplayChanged {
                        pane: *pane,
                        display: display.clone(),
                    },
                ));
            }
        }
        tracker
            .displays
            .retain(|pane, _| displays.contains_key(pane));
        self.shared.publish(Published {
            snapshot,
            files,
            displays,
        });

        let arm_timer = unsettled && !tracker.settle_timer_armed;
        if arm_timer {
            tracker.settle_timer_armed = true;
        }
        drop(tracker);
        for (target, event) in events {
            self.post(target, event);
        }
        if cancel_all || !changed_panes.is_empty() {
            // Effects lock `AppState`, which the renderer holds right now.
            slint::Timer::single_shot(Duration::ZERO, move || {
                super::with_host(|host| {
                    let effects = if cancel_all {
                        host.controller.borrow_mut().cancel()
                    } else {
                        let mut controller = host.controller.borrow_mut();
                        changed_panes
                            .iter()
                            .flat_map(|pane| controller.pane_changed(*pane))
                            .collect()
                    };
                    host.apply_effects(effects);
                    host.refresh_undo_ui();
                });
            });
        }
        if arm_timer {
            // Report the view once it settles even if no frame follows.
            slint::Timer::single_shot(VIEW_SETTLE + Duration::from_millis(20), || {
                super::with_host(|host| {
                    host.tracker.borrow_mut().settle_timer_armed = false;
                    host.publish_now();
                });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plugins' screen math (`ViewInfo::image_to_screen`) agrees with the
    /// viewer's for every turn.
    #[test]
    fn published_views_map_points_like_the_viewer() {
        let mut viewport = eov_common::Viewport::new(640.0, 480.0, 5000.0, 3000.0);
        viewport.zoom = 0.3;
        viewport.center.x = 2100.0;
        viewport.center.y = 1400.0;
        for orientation in eov_common::viewport::Orientation::all() {
            viewport.orientation = orientation;
            let view = ViewInfo {
                center_x: viewport.center.x,
                center_y: viewport.center.y,
                zoom: viewport.zoom,
                width: viewport.width,
                height: viewport.height,
                rotation_deg: f64::from(orientation.degrees()),
                flipped: orientation.flipped(),
                ..Default::default()
            };
            let (x, y) = view.image_to_screen(2300.0, 1250.0);
            let expected = viewport.image_to_screen(2300.0, 1250.0);
            assert!((x - expected.x).abs() < 1e-6 && (y - expected.y).abs() < 1e-6);
        }
    }

    #[test]
    fn alpha_is_unpremultiplied_or_composited() {
        // 50 % red, premultiplied.
        let mut straight = vec![128, 0, 0, 128];
        finish_alpha(&mut straight, 0);
        assert_eq!(straight, vec![255, 0, 0, 128]);
        let mut on_white = vec![128, 0, 0, 128];
        finish_alpha(&mut on_white, 0xFFFF_FFFF);
        assert_eq!(on_white[3], 255);
        assert!(on_white[0] >= 254 && (126..=128).contains(&on_white[1]));
        let mut transparent = vec![0, 0, 0, 0];
        finish_alpha(&mut transparent, 0);
        assert_eq!(transparent, vec![0, 0, 0, 0]);
    }
}
