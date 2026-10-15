//! Per-pane overlay frames: what each pane row of the UI shows on top of the
//! slide.
//!
//! Each frame the renderer asks [`UiHost::pane_overlay`] for a pane's
//! overlay fields. When the view, the layers or the interaction scene
//! changed, a render job goes to the worker; until it finishes, the last
//! frame is reprojected (moved and scaled) to the current view, so panning
//! stays smooth however many objects there are.

use super::UiHost;
use crate::interaction::{Calibration, PaneView};
use crate::overlay::labels::{self, LabelAnchor};
use crate::overlay::raster::{FrameScene, ViewTransform};
use crate::overlay::worker::OverlayJob;
use crate::{ContextMenuItem, HudToolbarButtonData, OverlayLabel, PaneCursor};
use eov_common::{MeasurementUnit, Viewport};
use eov_plugin_api::protocol::FileId;
use slint::{Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::collections::HashMap;
use std::rc::Rc;

/// What a pane row needs from the overlay system.
pub(crate) struct PaneOverlayFields {
    pub image: Image,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub labels: ModelRc<OverlayLabel>,
    pub cursor: PaneCursor,
    pub hud_buttons: ModelRc<HudToolbarButtonData>,
}

impl PaneOverlayFields {
    /// No overlay (before the host UI exists).
    pub(crate) fn empty() -> Self {
        Self {
            image: Image::default(),
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            labels: ModelRc::default(),
            cursor: PaneCursor::Default,
            hud_buttons: ModelRc::default(),
        }
    }
}

#[derive(Clone, PartialEq)]
struct FrameKey {
    file_id: FileId,
    view: ViewTransform,
    store_revision: u64,
    scene_generation: u64,
}

struct Displayed {
    view: ViewTransform,
    image: Image,
}

struct PaneOverlay {
    requested: Option<FrameKey>,
    ticket: u64,
    displayed_ticket: u64,
    displayed: Option<Displayed>,
    labels: Rc<VecModel<OverlayLabel>>,
    hud: Rc<VecModel<HudToolbarButtonData>>,
    hud_revision: Option<u64>,
}

impl Default for PaneOverlay {
    fn default() -> Self {
        Self {
            requested: None,
            ticket: 0,
            displayed_ticket: 0,
            displayed: None,
            labels: Rc::new(VecModel::default()),
            hud: Rc::new(VecModel::default()),
            hud_revision: None,
        }
    }
}

#[derive(Default)]
pub(super) struct OverlayPanes {
    panes: HashMap<u32, PaneOverlay>,
    /// Bumped whenever hover, previews or handles change.
    scene_generation: u64,
    cursors: HashMap<usize, PaneCursor>,
}

impl OverlayPanes {
    pub(super) fn bump_scene(&mut self) {
        self.scene_generation += 1;
    }

    pub(super) fn invalidate_all(&mut self) {
        self.scene_generation += 1;
        for pane in self.panes.values_mut() {
            pane.requested = None;
            pane.hud_revision = None;
        }
    }

    pub(super) fn set_cursor(&mut self, pane_index: usize, cursor: PaneCursor) {
        self.cursors.insert(pane_index, cursor);
    }

    /// Whether some pane still waits for a frame.
    pub(super) fn pending(&self) -> bool {
        self.panes.values().any(|pane| {
            pane.requested.as_ref().is_some_and(|key| {
                pane.displayed
                    .as_ref()
                    .is_none_or(|displayed| displayed.view != key.view)
            })
        })
    }
}

fn view_transform(viewport: &Viewport, scale: f64, calibration: &Calibration) -> ViewTransform {
    let bounds = viewport.bounds();
    ViewTransform {
        left: bounds.left,
        top: bounds.top,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        scale,
        mpp: calibration.average(),
    }
}

fn to_image(width: u32, height: u32, pixels: &[u8]) -> Option<Image> {
    if pixels.len() != width as usize * height as usize * 4 {
        return None;
    }
    let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(pixels, width, height);
    Some(Image::from_rgba8_premultiplied(buffer))
}

fn color_of(c: eov_plugin_api::protocol::Rgba) -> slint::Color {
    slint::Color::from_argb_u8(c.3, c.0, c.1, c.2)
}

fn label_struct(label: &labels::Label) -> OverlayLabel {
    OverlayLabel {
        x: label.x,
        y: label.y,
        anchor: match label.anchor {
            LabelAnchor::TopLeft => 0,
            LabelAnchor::Above => 1,
            LabelAnchor::Right => 2,
        },
        text: label.text.clone().into(),
        color: color_of(label.color),
        backdrop: label.backdrop,
        font_size: label.font_size.clamp(6.0, 48.0),
    }
}

fn sync_model<T: Clone + PartialEq + 'static>(model: &VecModel<T>, items: Vec<T>) {
    let unchanged = model.row_count() == items.len()
        && items
            .iter()
            .enumerate()
            .all(|(i, item)| model.row_data(i).as_ref() == Some(item));
    if !unchanged {
        model.set_vec(items);
    }
}

impl UiHost {
    /// The overlay fields of pane `index` for this frame. Called while the
    /// renderer holds `AppState`; never calls plugins.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pane_overlay(
        &self,
        index: usize,
        uid: u32,
        file_id: Option<FileId>,
        viewport: Option<&Viewport>,
        calibration: Calibration,
        unit: MeasurementUnit,
        scale: f64,
    ) -> PaneOverlayFields {
        let mut panes = self.overlays.borrow_mut();
        let scene_generation = panes.scene_generation;
        let cursor = panes
            .cursors
            .get(&index)
            .copied()
            .unwrap_or(PaneCursor::Default);
        let pane = panes.panes.entry(uid).or_default();

        let hud_revision = self.shared.contributions.lock().revision();
        if pane.hud_revision != Some(hud_revision) {
            pane.hud_revision = Some(hud_revision);
            let contributions = self.shared.contributions.lock();
            let buttons: Vec<HudToolbarButtonData> = contributions
                .hud()
                .into_iter()
                .map(|entry| HudToolbarButtonData {
                    plugin_id: entry.plugin_id.to_string().into(),
                    button_id: entry.spec.button_id.clone().into(),
                    tooltip: entry.spec.tooltip.clone().into(),
                    icon: super::callbacks::svg_image(if entry.active_in(uid) {
                        entry
                            .spec
                            .toggled_icon_svg
                            .as_deref()
                            .unwrap_or(&entry.spec.icon_svg)
                    } else {
                        &entry.spec.icon_svg
                    }),
                    active: entry.active_in(uid),
                    menu: ModelRc::from(Rc::new(VecModel::from(
                        entry
                            .menu_in(uid)
                            .iter()
                            .map(|item| ContextMenuItem {
                                id: item.item_id.clone().into(),
                                label: item.label.clone().into(),
                                icon: if item.checked == Some(true) {
                                    "check".into()
                                } else {
                                    item.icon.clone().into()
                                },
                                shortcut: item.shortcut.clone().into(),
                                enabled: item.enabled,
                                separator_after: item.separator_after,
                            })
                            .collect::<Vec<_>>(),
                    ))),
                })
                .collect();
            drop(contributions);
            sync_model(&pane.hud, buttons);
        }

        let empty = |pane: &PaneOverlay| PaneOverlayFields {
            image: Image::default(),
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            labels: ModelRc::from(Rc::clone(&pane.labels)),
            cursor,
            hud_buttons: ModelRc::from(Rc::clone(&pane.hud)),
        };
        let (Some(file_id), Some(viewport)) = (file_id, viewport) else {
            pane.displayed = None;
            pane.requested = None;
            sync_model(&pane.labels, Vec::new());
            return empty(pane);
        };
        let view = view_transform(viewport, scale, &calibration);
        let pane_view = PaneView {
            index,
            uid,
            file_id: Some(file_id),
            viewport: viewport.clone(),
            calibration,
            unit,
        };
        let layers = self.store.file_layers(file_id);
        let (hovered, overrides, transients) = match self.controller.try_borrow() {
            Ok(controller) => controller.scene_extras(&pane_view),
            Err(_) => Default::default(),
        };
        let scene = FrameScene {
            layers: layers.layers,
            hovered,
            overrides,
            transients,
        };

        let key = FrameKey {
            file_id,
            view,
            store_revision: layers.revision,
            scene_generation,
        };
        if pane.requested.as_ref() != Some(&key) && !self.worker.is_busy(uid) {
            pane.ticket += 1;
            if scene.is_empty() {
                // Nothing to draw: skip the worker entirely.
                pane.displayed = None;
            } else {
                self.worker.submit(OverlayJob {
                    pane: uid,
                    ticket: pane.ticket,
                    view,
                    scene: scene.clone(),
                });
            }
            pane.requested = Some(key);
        }
        if let Some(frame) = self.worker.take_finished(uid)
            && frame.ticket >= pane.displayed_ticket
        {
            pane.displayed_ticket = frame.ticket;
            pane.displayed = frame.pixels.and_then(|pixels| {
                to_image(frame.width, frame.height, &pixels).map(|image| Displayed {
                    view: frame.view,
                    image,
                })
            });
        }

        let labels: Vec<OverlayLabel> = labels::collect_labels(&view, &scene)
            .iter()
            .map(label_struct)
            .collect();
        sync_model(&pane.labels, labels);

        let Some(displayed) = &pane.displayed else {
            return empty(pane);
        };
        // Reproject the displayed frame onto the current view.
        let ratio = view.zoom / displayed.view.zoom.max(1e-12);
        PaneOverlayFields {
            image: displayed.image.clone(),
            x: ((displayed.view.left - view.left) * view.zoom) as f32,
            y: ((displayed.view.top - view.top) * view.zoom) as f32,
            width: (displayed.view.width * ratio) as f32,
            height: (displayed.view.height * ratio) as f32,
            labels: ModelRc::from(Rc::clone(&pane.labels)),
            cursor,
            hud_buttons: ModelRc::from(Rc::clone(&pane.hud)),
        }
    }

    /// Whether the render loop must keep running for overlays.
    pub(crate) fn overlay_frames_pending(&self) -> bool {
        self.overlays.borrow().pending()
    }

    /// Forgets overlay state of panes that no longer exist.
    pub(crate) fn retain_panes(&self, live: &[u32]) {
        let mut overlays = self.overlays.borrow_mut();
        let gone: Vec<u32> = overlays
            .panes
            .keys()
            .filter(|uid| !live.contains(uid))
            .copied()
            .collect();
        for uid in gone {
            overlays.panes.remove(&uid);
            self.worker.forget(uid);
            self.shared.contributions.lock().forget_pane(uid);
        }
    }

    pub(super) fn redraw_overlays(&self) {
        self.overlays.borrow_mut().bump_scene();
        self.request_render();
    }
}
