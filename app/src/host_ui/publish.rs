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
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::protocol::{
    FileId, FileInfo, HostEvent, HostSnapshot, LevelInfo, MeasurementInfo, PaneInfo, PropertyMap,
    Rect, ToolRef, ViewInfo,
};
use parking_lot::Mutex;
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
}

/// Slide metadata the plugin API exposes, from the WSI core.
trait WsiMetadata {
    fn property_map_snapshot(&self) -> PropertyMap;
    fn associated_image_rgba(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String>;
}

impl WsiMetadata for WsiFile {
    fn property_map_snapshot(&self) -> PropertyMap {
        PropertyMap::new()
    }

    fn associated_image_rgba(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String> {
        Err(format!("associated image '{name}' is not available"))
    }
}

/// Reads plugin pixel requests through a dedicated slide handle, so a bad
/// region read by a plugin never affects the viewer's own handle.
struct FileReader {
    path: PathBuf,
    handle: Mutex<Option<WsiFile>>,
}

impl FileReader {
    fn with_handle<T>(&self, f: impl FnOnce(&WsiFile) -> Result<T, String>) -> Result<T, String> {
        let mut handle = self.handle.lock();
        if handle.is_none() {
            *handle = Some(WsiFile::open(&self.path).map_err(|e| e.to_string())?);
        }
        f(handle.as_ref().expect("opened"))
    }
}

impl SlideReader for FileReader {
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String> {
        self.with_handle(|wsi| {
            let mut pixels = wsi
                .read_region(
                    request.x,
                    request.y,
                    request.level,
                    request.width,
                    request.height,
                )
                .map_err(|e| e.to_string())?;
            finish_alpha(&mut pixels, request.background);
            Ok(pixels)
        })
    }

    fn associated_image(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String> {
        self.with_handle(|wsi| {
            let image = wsi.associated_image_rgba(name).map_err(|e| e.to_string())?;
            Ok(image)
        })
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

pub(crate) fn file_info(file: &OpenFile, properties: &PropertyMap) -> FileInfo {
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
        mpp_x: props.mpp_x,
        mpp_y: props.mpp_y,
        objective_power: props.objective_power,
        vendor: props.vendor.clone(),
        scan_date: props.scan_date.clone(),
        bounds: None,
        identity: properties
            .get("openslide.quickhash-1")
            .map(|hash| format!("quickhash1:{hash}")),
        associated_images: Vec::new(),
        has_icc_profile: false,
        stain: props.stain.clone(),
    }
}

/// The calibration of a slide.
pub(crate) fn calibration(file: &OpenFile) -> Calibration {
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
        rotation_deg: 0.0,
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
    /// Publishes immediately (startup, after commands).
    pub(super) fn publish_now(&self) {
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
        state.publish_pane_uids();
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
                .or_insert_with(|| Arc::new(file.wsi.property_map_snapshot()))
                .clone();
            let info = file_info(file, &properties);
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
            self.shared.set_reader(
                *id,
                Arc::new(FileReader {
                    path: current[id].clone(),
                    handle: Mutex::new(None),
                }),
            );
            self.ensure_host_layers(*id);
            if let Some(file) = files.get(id) {
                events.push((
                    EventTarget::All,
                    HostEvent::FileOpened {
                        file: file.info.clone(),
                    },
                ));
            }
        }
        for id in &closed {
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

        let window_focused = self.ui().is_some_and(|ui| {
            use slint::winit_030::WinitWindowAccessor;
            ui.window()
                .with_winit_window(|window| window.has_focus())
                .unwrap_or(true)
        });
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
        self.shared.publish(Published { snapshot, files });

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
