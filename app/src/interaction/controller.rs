//! The viewport interaction controller.
//!
//! Slint forwards raw pointer, wheel and key input for every pane; this
//! controller turns it into navigation, host tool actions (ROI and
//! measurements), plugin tool gestures, selection, hover and editing. It is
//! pure logic: it reads the overlay store, never touches `AppState` or Slint,
//! and reports what should happen as [`Effect`]s, which the app applies with
//! the appropriate locks and then delivers plugin events with none held.

use super::measure::{
    self, Calibration, HostMeasurement, MeasureMode, is_measurement_layer, is_roi_layer,
};
use crate::overlay::geometry::{self, Point};
use crate::overlay::raster::Transient;
use crate::overlay::store::{Layer, LayerKey, ObjectKey, OverlayStore, Owner, PickParams};
use eov_common::MeasurementUnit;
use eov_plugin_api::protocol::{
    Cursor, FileId, HostEvent, Modifiers, ObjectRef, PointerButton, PointerEvent,
    PointerKind as ProtoPointerKind, Rect, Shape, Style, ToolKind, ToolSpec,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Movement (logical px) that turns a press into a drag.
pub const DRAG_THRESHOLD_PX: f64 = 4.0;
/// Maximum interval and distance between the clicks of a double click.
pub const DOUBLE_CLICK_INTERVAL: Duration = Duration::from_millis(400);
pub const DOUBLE_CLICK_DISTANCE_PX: f64 = 6.0;
/// Radius (logical px) within which a vertex handle or the first polygon
/// vertex is grabbed.
pub const HANDLE_RADIUS_PX: f64 = 7.0;
/// Outline pick tolerance (logical px).
pub const PICK_TOLERANCE_PX: f64 = 4.0;
/// Mouse-wheel zoom factor per notch (Slint reports 60 px per line).
pub const WHEEL_ZOOM_PER_NOTCH: f64 = 1.3225;
const WHEEL_PX_PER_NOTCH: f64 = 60.0;
/// Minimum on-screen size (logical px) of a dragged rectangle or line.
const MIN_DRAG_SHAPE_PX: f64 = 3.0;
/// At most this many edit handles are drawn.
const MAX_HANDLES: usize = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    Down,
    Move,
    Up,
    Leave,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerInput {
    pub kind: PointerKind,
    pub button: PointerButton,
    /// Logical pixels relative to the pane.
    pub x: f64,
    pub y: f64,
    /// Logical pixels relative to the window (for menus and tooltips).
    pub window_x: f64,
    pub window_y: f64,
    pub modifiers: Modifiers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyInput {
    /// Slint key text (a character, or a special key code point).
    pub text: String,
    pub modifiers: Modifiers,
    pub pressed: bool,
    pub repeat: bool,
}

/// Slint's special key code points (see `slint::platform::Key`).
pub mod keys {
    pub const BACKSPACE: &str = "\u{8}";
    pub const TAB: &str = "\u{9}";
    pub const RETURN: &str = "\u{a}";
    pub const ESCAPE: &str = "\u{1b}";
    pub const DELETE: &str = "\u{7f}";
}

/// What the controller needs to know about a pane for one event.
#[derive(Clone, Debug)]
pub struct PaneView {
    /// Position of the pane (index into the pane list).
    pub index: usize,
    /// Stable pane id exposed to plugins.
    pub uid: u32,
    pub file_id: Option<FileId>,
    pub viewport: eov_common::Viewport,
    pub calibration: Calibration,
    pub unit: MeasurementUnit,
}

impl PaneView {
    pub fn to_image(&self, x: f64, y: f64) -> Point {
        let p = self.viewport.screen_to_image(x, y);
        [p.x, p.y]
    }

    pub fn zoom(&self) -> f64 {
        self.viewport.zoom.max(1e-12)
    }

    fn screen_distance(&self, a: Point, b: Point) -> f64 {
        geometry::distance(a, b) * self.zoom()
    }
}

/// The active tool.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    Navigate,
    Roi,
    Measure(MeasureMode),
    Plugin { plugin_id: Arc<str>, spec: ToolSpec },
}

impl ToolChoice {
    pub fn plugin_tool(&self) -> Option<(&Arc<str>, &ToolSpec)> {
        match self {
            ToolChoice::Plugin { plugin_id, spec } => Some((plugin_id, spec)),
            _ => None,
        }
    }

    fn is_custom(&self) -> bool {
        matches!(self, ToolChoice::Plugin { spec, .. } if spec.kind == ToolKind::Custom)
    }
}

/// Something the app must do in response to input.
// Effects live in short vectors for one event; boxing the plugin event
// would only add allocations.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    StartPan {
        pane: usize,
        x: f64,
        y: f64,
    },
    Pan {
        pane: usize,
        x: f64,
        y: f64,
    },
    EndPan {
        pane: usize,
    },
    /// Pan by logical pixels (wheel/trackpad scrolling).
    PanBy {
        pane: usize,
        dx: f64,
        dy: f64,
    },
    Zoom {
        pane: usize,
        factor: f64,
        x: f64,
        y: f64,
    },
    ContextMenu {
        pane: usize,
        window_x: f64,
        window_y: f64,
        image: Point,
        /// The topmost hit of each owner (host first if any).
        hits: Vec<ObjectKey>,
    },
    Plugin {
        plugin_id: Arc<str>,
        event: HostEvent,
    },
    SetRoi {
        file_id: FileId,
        rect: Option<Rect>,
    },
    AddMeasurement {
        file_id: FileId,
        measurement: HostMeasurement,
    },
    UpdateMeasurement {
        file_id: FileId,
        id: String,
        shape: Shape,
    },
    RemoveMeasurements {
        file_id: FileId,
        ids: Vec<String>,
    },
    /// The active tool changed (e.g. Escape returned to Navigate).
    ToolChanged,
    Cursor {
        pane: usize,
        cursor: Cursor,
    },
    /// Show (or with `None` hide) the hover tooltip.
    Tooltip(Option<TooltipRequest>),
    /// The overlay scene (preview, hover, handles) changed.
    Redraw,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TooltipRequest {
    pub pane: usize,
    pub window_x: f64,
    pub window_y: f64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
enum PressTarget {
    Empty,
    /// An interactive object (click selects; drag behaves like empty).
    Object(ObjectKey),
    /// A vertex/corner handle of an editable object.
    Handle {
        key: ObjectKey,
        index: usize,
    },
    /// The outline of an editable polygon/polyline away from vertices.
    Edge {
        key: ObjectKey,
        after: usize,
        at: Point,
    },
    /// The body of an editable object.
    Body(ObjectKey),
}

impl PressTarget {
    fn key(&self) -> Option<&ObjectKey> {
        match self {
            PressTarget::Empty => None,
            PressTarget::Object(key)
            | PressTarget::Body(key)
            | PressTarget::Handle { key, .. }
            | PressTarget::Edge { key, .. } => Some(key),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum DragShape {
    Roi,
    MeasureLine,
    MeasureRect,
    MeasureEllipse,
    PluginRect,
    PluginEllipse,
    Marquee,
}

#[derive(Clone, Debug, PartialEq)]
enum Grab {
    Handle(usize),
    Body,
}

#[derive(Clone, Debug, PartialEq)]
enum Gesture {
    Idle,
    Pressed {
        pane: usize,
        button: PointerButton,
        start: (f64, f64),
        start_image: Point,
        modifiers: Modifiers,
        target: PressTarget,
        file_id: Option<FileId>,
    },
    Panning {
        pane: usize,
    },
    DragShape {
        pane: usize,
        file_id: FileId,
        kind: DragShape,
        start: Point,
        current: Point,
        modifiers: Modifiers,
    },
    Freehand {
        pane: usize,
        file_id: FileId,
        points: Vec<Point>,
        modifiers: Modifiers,
    },
    Editing {
        pane: usize,
        file_id: FileId,
        key: ObjectKey,
        original: Shape,
        current: Shape,
        grab: Grab,
        start_image: Point,
    },
    /// The left button is down on a plugin custom tool.
    Custom {
        pane: usize,
        file_id: FileId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawKind {
    MeasureLine,
    MeasurePolyline,
    MeasurePolygon,
    MeasureAngle,
    PluginPolyline,
    PluginPolygon,
}

impl DrawKind {
    fn closes(self) -> bool {
        matches!(self, DrawKind::MeasurePolygon | DrawKind::PluginPolygon)
    }

    fn min_points(self) -> usize {
        match self {
            DrawKind::MeasureLine | DrawKind::MeasurePolyline | DrawKind::PluginPolyline => 2,
            DrawKind::MeasurePolygon | DrawKind::PluginPolygon | DrawKind::MeasureAngle => 3,
        }
    }
}

/// A click-by-click shape in progress.
#[derive(Clone, Debug, PartialEq)]
struct Drawing {
    pane: usize,
    file_id: FileId,
    kind: DrawKind,
    points: Vec<Point>,
    hover: Option<Point>,
}

#[derive(Clone, Debug, PartialEq)]
struct Hover {
    pane: usize,
    key: ObjectKey,
    tooltip: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct Click {
    at: Instant,
    pane: usize,
    x: f64,
    y: f64,
}

/// The interaction state machine. Owned by the UI thread.
pub struct Controller {
    store: Arc<OverlayStore>,
    tool: ToolChoice,
    gesture: Gesture,
    drawing: Option<Drawing>,
    hover: Option<Hover>,
    hover_handle: Option<(ObjectKey, usize)>,
    /// A tooltip shown for an object of a non-interactive layer.
    passive_tooltip: Option<String>,
    last_click: Option<Click>,
    cursor: HashMap<usize, Cursor>,
    /// Plain scrolling pans (trackpad mode); Ctrl/⌘+scroll zooms.
    pub scroll_pans: bool,
    pinch_scale: Option<f64>,
    measurement_counter: u64,
}

impl Controller {
    pub fn new(store: Arc<OverlayStore>) -> Self {
        Self {
            store,
            tool: ToolChoice::Navigate,
            gesture: Gesture::Idle,
            drawing: None,
            hover: None,
            hover_handle: None,
            passive_tooltip: None,
            last_click: None,
            cursor: HashMap::new(),
            scroll_pans: false,
            pinch_scale: None,
            measurement_counter: 0,
        }
    }

    pub fn tool(&self) -> &ToolChoice {
        &self.tool
    }

    /// Whether a gesture or click-by-click shape is in progress.
    pub fn is_busy(&self) -> bool {
        self.gesture != Gesture::Idle || self.drawing.is_some()
    }

    /// Switches tools, cancelling anything in progress, and tells plugins
    /// about tools they lose or gain.
    pub fn set_tool(&mut self, tool: ToolChoice) -> Vec<Effect> {
        if tool == self.tool {
            return Vec::new();
        }
        let mut effects = self.cancel();
        if let ToolChoice::Plugin { plugin_id, spec } = &self.tool {
            effects.push(Effect::Plugin {
                plugin_id: Arc::clone(plugin_id),
                event: HostEvent::ToolDeactivated {
                    tool_id: spec.tool_id.clone(),
                },
            });
        }
        if let ToolChoice::Plugin { plugin_id, spec } = &tool {
            effects.push(Effect::Plugin {
                plugin_id: Arc::clone(plugin_id),
                event: HostEvent::ToolActivated {
                    tool_id: spec.tool_id.clone(),
                },
            });
        }
        self.tool = tool;
        effects.push(Effect::ToolChanged);
        let cursor = self.idle_cursor();
        for (pane, current) in self.cursor.iter_mut() {
            if *current != cursor {
                *current = cursor;
                effects.push(Effect::Cursor {
                    pane: *pane,
                    cursor,
                });
            }
        }
        effects
    }

    /// Cancels gestures and shapes in progress (tool, tab, pane or file
    /// changes must never let one gesture mix coordinates of two slides).
    pub fn cancel(&mut self) -> Vec<Effect> {
        let mut effects = Vec::new();
        match std::mem::replace(&mut self.gesture, Gesture::Idle) {
            Gesture::Panning { pane } => effects.push(Effect::EndPan { pane }),
            Gesture::Custom { pane, file_id } => {
                if let Some((plugin_id, spec)) = self.tool.plugin_tool() {
                    effects.push(Effect::Plugin {
                        plugin_id: Arc::clone(plugin_id),
                        event: HostEvent::Pointer {
                            tool_id: spec.tool_id.clone(),
                            pane: pane as u32,
                            file_id,
                            event: PointerEvent {
                                kind: ProtoPointerKind::Leave,
                                button: PointerButton::None,
                                image_x: 0.0,
                                image_y: 0.0,
                                screen_x: 0.0,
                                screen_y: 0.0,
                                modifiers: Modifiers::default(),
                                wheel_delta: 0.0,
                            },
                        },
                    });
                }
            }
            Gesture::Idle => {}
            _ => effects.push(Effect::Redraw),
        }
        if self.drawing.take().is_some() {
            effects.push(Effect::Redraw);
        }
        self.pinch_scale = None;
        effects
    }

    /// Cancels work tied to a pane whose file changed or that was removed.
    pub fn pane_changed(&mut self, pane: usize) -> Vec<Effect> {
        let affected = self.drawing.as_ref().is_some_and(|d| d.pane == pane)
            || match &self.gesture {
                Gesture::Idle => false,
                Gesture::Pressed { pane: p, .. }
                | Gesture::Panning { pane: p }
                | Gesture::DragShape { pane: p, .. }
                | Gesture::Freehand { pane: p, .. }
                | Gesture::Editing { pane: p, .. }
                | Gesture::Custom { pane: p, .. } => *p == pane,
            };
        let mut effects = if affected { self.cancel() } else { Vec::new() };
        if self.hover.as_ref().is_some_and(|h| h.pane == pane) {
            self.hover = None;
            self.hover_handle = None;
            effects.push(Effect::Tooltip(None));
            effects.push(Effect::Redraw);
        }
        effects
    }

    fn idle_cursor(&self) -> Cursor {
        match &self.tool {
            ToolChoice::Navigate => Cursor::Default,
            ToolChoice::Roi | ToolChoice::Measure(_) => Cursor::Crosshair,
            ToolChoice::Plugin { spec, .. } => spec.cursor,
        }
    }

    fn set_cursor(&mut self, pane: usize, cursor: Cursor, effects: &mut Vec<Effect>) {
        if self.cursor.get(&pane) != Some(&cursor) {
            self.cursor.insert(pane, cursor);
            effects.push(Effect::Cursor { pane, cursor });
        }
    }

    // -- layer permissions ---------------------------------------------------

    /// Whether the active tool may reshape objects of `layer`.
    fn can_edit(&self, layer: &Layer) -> bool {
        match (&self.tool, &layer.key.owner) {
            (ToolChoice::Roi, Owner::Host) => is_roi_layer(&layer.key.id),
            (ToolChoice::Measure(_), Owner::Host) => is_measurement_layer(&layer.key.id),
            (ToolChoice::Plugin { plugin_id, spec }, Owner::Plugin(owner)) => {
                spec.kind != ToolKind::Custom
                    && owner == plugin_id
                    && layer.spec.editable
                    && (spec.edit_layers.is_empty() || spec.edit_layers.contains(&layer.key.id))
            }
            _ => false,
        }
    }

    /// Whether clicks and hovers consider `layer`.
    fn can_pick(&self, layer: &Layer) -> bool {
        if self.can_edit(layer) {
            return true;
        }
        match &layer.key.owner {
            Owner::Host => match &self.tool {
                ToolChoice::Navigate => is_measurement_layer(&layer.key.id),
                ToolChoice::Roi => is_roi_layer(&layer.key.id),
                ToolChoice::Measure(_) => is_measurement_layer(&layer.key.id),
                ToolChoice::Plugin { .. } => false,
            },
            Owner::Plugin(owner) => {
                layer.spec.interactive
                    && match &self.tool {
                        ToolChoice::Navigate => true,
                        ToolChoice::Plugin { plugin_id, spec } => {
                            spec.kind == ToolKind::Select || owner == plugin_id
                        }
                        _ => false,
                    }
            }
        }
    }

    fn pick_params(&self, view: &PaneView, file_id: FileId, point: Point) -> PickParams {
        PickParams {
            pane: Some(view.uid),
            file_id,
            point,
            zoom: view.zoom(),
            mpp: view.calibration.average(),
            tolerance_px: PICK_TOLERANCE_PX,
        }
    }

    /// Editable objects whose handles are shown: the hovered one and the
    /// selected ones, in layers the tool may edit.
    fn handle_owners(&self, file_id: FileId) -> Vec<(ObjectKey, Shape)> {
        let layers = self.store.file_layers(file_id);
        let mut out = Vec::new();
        for layer in &layers.layers {
            if !layer.spec.visible || !self.can_edit(layer) {
                continue;
            }
            for object_id in layer.selection() {
                if let Some(object) = layer.object(object_id)
                    && layer.object_is_editable(object)
                {
                    out.push((
                        ObjectKey {
                            layer: layer.key.clone(),
                            object_id: object_id.clone(),
                        },
                        object.object.shape.clone(),
                    ));
                }
            }
            if let Some(hover) = &self.hover
                && hover.key.layer == layer.key
                && !layer.is_selected(&hover.key.object_id)
                && let Some(object) = layer.object(&hover.key.object_id)
                && layer.object_is_editable(object)
            {
                out.push((hover.key.clone(), object.object.shape.clone()));
            }
        }
        out
    }

    /// Finds what a left press at `point` would grab.
    fn press_target(&self, view: &PaneView, file_id: FileId, point: Point) -> PressTarget {
        let zoom = view.zoom();
        // Handles of hovered/selected editable objects take priority.
        for (key, shape) in self.handle_owners(file_id) {
            if matches!(shape, Shape::Point { .. }) {
                continue;
            }
            for (index, handle) in geometry::shape_handles(&shape).into_iter().enumerate() {
                if geometry::distance(handle, point) * zoom <= HANDLE_RADIUS_PX {
                    return PressTarget::Handle { key, index };
                }
            }
        }
        let hits = self
            .store
            .hit_test(self.pick_params(view, file_id, point), |layer| {
                self.can_pick(layer)
            });
        let Some(hit) = hits.into_iter().next() else {
            return PressTarget::Empty;
        };
        let Some(layer) = self.store.layer(&hit.key.layer.owner, &hit.key.layer.id) else {
            return PressTarget::Empty;
        };
        let Some(object) = layer.object(&hit.key.object_id) else {
            return PressTarget::Empty;
        };
        if !(self.can_edit(&layer) && layer.object_is_editable(object)) {
            return PressTarget::Object(hit.key);
        }
        let shape = &object.object.shape;
        if hit.on_outline
            && let Some((after, a, b)) = geometry::shape_edges(shape).into_iter().min_by(|x, y| {
                geometry::distance_to_segment(point, x.1, x.2)
                    .total_cmp(&geometry::distance_to_segment(point, y.1, y.2))
            })
            && geometry::distance_to_segment(point, a, b) * zoom <= PICK_TOLERANCE_PX + 2.0
        {
            return PressTarget::Edge {
                key: hit.key,
                after,
                at: point,
            };
        }
        PressTarget::Body(hit.key)
    }

    // -- selection -------------------------------------------------------------

    fn notify_selection(&self, owners: impl IntoIterator<Item = Owner>, effects: &mut Vec<Effect>) {
        let mut seen = Vec::new();
        for owner in owners {
            if seen.contains(&owner) {
                continue;
            }
            seen.push(owner.clone());
            if let Owner::Plugin(plugin_id) = &owner {
                let objects = self
                    .store
                    .selection_of(&owner)
                    .into_iter()
                    .map(|key| ObjectRef {
                        layer_id: key.layer.id,
                        object_id: key.object_id,
                    })
                    .collect();
                effects.push(Effect::Plugin {
                    plugin_id: Arc::clone(plugin_id),
                    event: HostEvent::SelectionChanged { objects },
                });
            }
        }
    }

    /// Selects `key` (replacing the selection, or toggling with Ctrl/⌘ or
    /// Shift).
    fn select(&mut self, key: &ObjectKey, modifiers: Modifiers, effects: &mut Vec<Effect>) {
        let additive = modifiers.primary || modifiers.shift;
        let mut changed = Vec::new();
        if additive {
            let layer = self.store.layer(&key.layer.owner, &key.layer.id);
            let mut ids: Vec<String> = layer
                .map(|l| l.selection().cloned().collect())
                .unwrap_or_default();
            if let Some(pos) = ids.iter().position(|id| id == &key.object_id) {
                ids.remove(pos);
            } else {
                ids.push(key.object_id.clone());
            }
            if self
                .store
                .select(&key.layer.owner, &key.layer.id, &ids)
                .unwrap_or(false)
            {
                changed.push(key.layer.owner.clone());
            }
        } else {
            changed.extend(self.store.clear_selection(None));
            if self
                .store
                .select(
                    &key.layer.owner,
                    &key.layer.id,
                    std::slice::from_ref(&key.object_id),
                )
                .unwrap_or(false)
            {
                changed.push(key.layer.owner.clone());
            }
        }
        if !changed.is_empty() {
            effects.push(Effect::Redraw);
        }
        self.notify_selection(changed, effects);
    }

    fn clear_selection(&mut self, effects: &mut Vec<Effect>) -> bool {
        let changed = self.store.clear_selection(None);
        if changed.is_empty() {
            return false;
        }
        effects.push(Effect::Redraw);
        self.notify_selection(changed, effects);
        true
    }

    // -- pointer input -----------------------------------------------------------

    pub fn pointer(&mut self, view: &PaneView, input: PointerInput, now: Instant) -> Vec<Effect> {
        let mut effects = Vec::new();
        match input.kind {
            PointerKind::Down => self.on_down(view, input, now, &mut effects),
            PointerKind::Move => self.on_move(view, input, &mut effects),
            PointerKind::Up => self.on_up(view, input, now, &mut effects),
            PointerKind::Leave => self.on_leave(view, &mut effects),
        }
        effects
    }

    fn custom_event(
        &self,
        view: &PaneView,
        file_id: FileId,
        kind: ProtoPointerKind,
        input: &PointerInput,
        wheel_delta: f64,
    ) -> Option<Effect> {
        let (plugin_id, spec) = self.tool.plugin_tool()?;
        let image = view.to_image(input.x, input.y);
        Some(Effect::Plugin {
            plugin_id: Arc::clone(plugin_id),
            event: HostEvent::Pointer {
                tool_id: spec.tool_id.clone(),
                pane: view.uid,
                file_id,
                event: PointerEvent {
                    kind,
                    button: input.button,
                    image_x: image[0],
                    image_y: image[1],
                    screen_x: input.x,
                    screen_y: input.y,
                    modifiers: input.modifiers,
                    wheel_delta,
                },
            },
        })
    }

    fn on_down(
        &mut self,
        view: &PaneView,
        input: PointerInput,
        now: Instant,
        effects: &mut Vec<Effect>,
    ) {
        if self.gesture != Gesture::Idle {
            // A second button during a gesture is ignored.
            return;
        }
        self.pinch_scale = None;
        // A shape in progress belongs to one pane and slide.
        if let Some(drawing) = &self.drawing
            && (drawing.pane != view.index || Some(drawing.file_id) != view.file_id)
        {
            effects.extend(self.cancel());
        }
        let point = view.to_image(input.x, input.y);
        let target = match (input.button, view.file_id) {
            (PointerButton::Left, Some(file_id)) => {
                if self.tool.is_custom() {
                    if let Some(effect) =
                        self.custom_event(view, file_id, ProtoPointerKind::Down, &input, 0.0)
                    {
                        effects.push(effect);
                    }
                    // Double clicks are reported to custom tools too.
                    if self.is_double_click(view.index, input.x, input.y, now) {
                        self.last_click = None;
                        if let Some(effect) = self.custom_event(
                            view,
                            file_id,
                            ProtoPointerKind::DoubleClick,
                            &input,
                            0.0,
                        ) {
                            effects.push(effect);
                        }
                    } else {
                        self.last_click = Some(Click {
                            at: now,
                            pane: view.index,
                            x: input.x,
                            y: input.y,
                        });
                    }
                    self.gesture = Gesture::Custom {
                        pane: view.index,
                        file_id,
                    };
                    return;
                }
                if self.drawing.is_some() {
                    PressTarget::Empty
                } else {
                    self.press_target(view, file_id, point)
                }
            }
            _ => PressTarget::Empty,
        };
        self.gesture = Gesture::Pressed {
            pane: view.index,
            button: input.button,
            start: (input.x, input.y),
            start_image: point,
            modifiers: input.modifiers,
            target,
            file_id: view.file_id,
        };
    }

    fn drag_kind(&self) -> Option<DragShape> {
        Some(match &self.tool {
            ToolChoice::Navigate => return None,
            ToolChoice::Roi => DragShape::Roi,
            ToolChoice::Measure(MeasureMode::Line) => {
                if self.drawing.is_some() {
                    return None;
                }
                DragShape::MeasureLine
            }
            ToolChoice::Measure(MeasureMode::Rectangle) => DragShape::MeasureRect,
            ToolChoice::Measure(MeasureMode::Ellipse) => DragShape::MeasureEllipse,
            ToolChoice::Measure(_) => return None,
            ToolChoice::Plugin { spec, .. } => match spec.kind {
                ToolKind::Rectangle => DragShape::PluginRect,
                ToolKind::Ellipse => DragShape::PluginEllipse,
                ToolKind::Select => DragShape::Marquee,
                _ => return None,
            },
        })
    }

    fn on_move(&mut self, view: &PaneView, input: PointerInput, effects: &mut Vec<Effect>) {
        let point = view.to_image(input.x, input.y);
        match &mut self.gesture {
            Gesture::Idle => {
                // Custom tools see the pointer while it hovers too (cursor
                // reticles, previews); the host coalesces these moves.
                if self.tool.is_custom()
                    && let Some(file_id) = view.file_id
                    && let Some(effect) =
                        self.custom_event(view, file_id, ProtoPointerKind::Move, &input, 0.0)
                {
                    effects.push(effect);
                }
                self.update_hover(view, input, point, effects);
                if let Some(drawing) = &mut self.drawing
                    && drawing.pane == view.index
                {
                    drawing.hover = Some(point);
                    effects.push(Effect::Redraw);
                }
            }
            Gesture::Pressed {
                pane,
                button,
                start,
                start_image,
                modifiers,
                target,
                file_id,
            } => {
                let (dx, dy) = (input.x - start.0, input.y - start.1);
                if dx.hypot(dy) < DRAG_THRESHOLD_PX {
                    return;
                }
                let (pane, button, start, start_image, modifiers, target, file_id) = (
                    *pane,
                    *button,
                    *start,
                    *start_image,
                    *modifiers,
                    target.clone(),
                    *file_id,
                );
                self.begin_drag(
                    view,
                    pane,
                    button,
                    start,
                    start_image,
                    modifiers,
                    target,
                    file_id,
                    effects,
                );
                // Apply the first movement immediately.
                self.on_move(view, input, effects);
            }
            Gesture::Panning { pane } => {
                effects.push(Effect::Pan {
                    pane: *pane,
                    x: input.x,
                    y: input.y,
                });
            }
            Gesture::DragShape { current, .. } => {
                *current = point;
                effects.push(Effect::Redraw);
            }
            Gesture::Freehand { points, .. } => {
                let far_enough = points
                    .last()
                    .is_none_or(|last| geometry::distance(*last, point) * view.zoom() >= 2.0);
                if far_enough {
                    points.push(point);
                    effects.push(Effect::Redraw);
                }
            }
            Gesture::Editing {
                original,
                current,
                grab,
                start_image,
                ..
            } => {
                *current = match grab {
                    Grab::Handle(index) => geometry::move_handle(original, *index, point),
                    Grab::Body => geometry::translate_shape(
                        original,
                        point[0] - start_image[0],
                        point[1] - start_image[1],
                    ),
                };
                effects.push(Effect::Redraw);
            }
            Gesture::Custom { file_id, .. } => {
                let file_id = *file_id;
                if let Some(effect) =
                    self.custom_event(view, file_id, ProtoPointerKind::Move, &input, 0.0)
                {
                    effects.push(effect);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_drag(
        &mut self,
        view: &PaneView,
        pane: usize,
        button: PointerButton,
        start: (f64, f64),
        start_image: Point,
        modifiers: Modifiers,
        target: PressTarget,
        file_id: Option<FileId>,
        effects: &mut Vec<Effect>,
    ) {
        let pan = |effects: &mut Vec<Effect>| {
            effects.push(Effect::StartPan {
                pane,
                x: start.0,
                y: start.1,
            });
            Gesture::Panning { pane }
        };
        if button != PointerButton::Left {
            self.gesture = pan(effects);
            return;
        }
        let Some(file_id) = file_id else {
            self.gesture = pan(effects);
            return;
        };
        let edit = |key: ObjectKey, grab: Grab, store: &OverlayStore| {
            let object = store.object(&key)?;
            Some((key, object.object.shape.clone(), grab))
        };
        let editing = match &target {
            PressTarget::Handle { key, index } => {
                edit(key.clone(), Grab::Handle(*index), &self.store)
            }
            PressTarget::Edge { key, after, at } => {
                // Insert a vertex at the press position and drag it.
                self.store.object(key).map(|object| {
                    let shape = geometry::insert_vertex(&object.object.shape, *after, *at);
                    (key.clone(), shape, Grab::Handle(after + 1))
                })
            }
            PressTarget::Body(key) => edit(key.clone(), Grab::Body, &self.store),
            _ => None,
        };
        if let Some((key, original, grab)) = editing {
            self.gesture = Gesture::Editing {
                pane,
                file_id,
                key,
                current: original.clone(),
                original,
                grab,
                start_image,
            };
            self.set_cursor(pane, Cursor::Move, effects);
            return;
        }
        if let ToolChoice::Plugin { spec, .. } = &self.tool
            && spec.kind == ToolKind::Freehand
        {
            self.gesture = Gesture::Freehand {
                pane,
                file_id,
                points: vec![start_image],
                modifiers,
            };
            return;
        }
        match self.drag_kind() {
            Some(kind) => {
                self.gesture = Gesture::DragShape {
                    pane,
                    file_id,
                    kind,
                    start: start_image,
                    current: start_image,
                    modifiers,
                };
            }
            None => self.gesture = pan(effects),
        }
        let _ = view;
    }

    fn on_up(
        &mut self,
        view: &PaneView,
        input: PointerInput,
        now: Instant,
        effects: &mut Vec<Effect>,
    ) {
        let point = view.to_image(input.x, input.y);
        match std::mem::replace(&mut self.gesture, Gesture::Idle) {
            Gesture::Idle => {}
            Gesture::Pressed {
                pane,
                button,
                start_image,
                modifiers,
                target,
                file_id,
                ..
            } => match button {
                PointerButton::Right => {
                    let hits = file_id
                        .map(|file_id| self.context_hits(view, file_id, start_image))
                        .unwrap_or_default();
                    effects.push(Effect::ContextMenu {
                        pane,
                        window_x: input.window_x,
                        window_y: input.window_y,
                        image: start_image,
                        hits,
                    });
                }
                PointerButton::Left => {
                    if let Some(file_id) = file_id {
                        let double = self.is_double_click(pane, input.x, input.y, now);
                        self.last_click = if double {
                            None
                        } else {
                            Some(Click {
                                at: now,
                                pane,
                                x: input.x,
                                y: input.y,
                            })
                        };
                        self.click(
                            view,
                            file_id,
                            start_image,
                            target,
                            modifiers,
                            double,
                            effects,
                        );
                    }
                }
                _ => {}
            },
            Gesture::Panning { pane } => effects.push(Effect::EndPan { pane }),
            Gesture::DragShape {
                file_id,
                kind,
                start,
                current,
                modifiers,
                ..
            } => {
                let current = if view.file_id == Some(file_id) {
                    point
                } else {
                    current
                };
                self.finish_drag_shape(view, file_id, kind, start, current, modifiers, effects);
                effects.push(Effect::Redraw);
            }
            Gesture::Freehand {
                file_id,
                mut points,
                modifiers,
                ..
            } => {
                if view.file_id == Some(file_id) {
                    points.push(point);
                }
                let tolerance = self
                    .tool
                    .plugin_tool()
                    .map(|(_, spec)| f64::from(spec.freehand_tolerance_px))
                    .unwrap_or(1.5)
                    / view.zoom();
                let ring = geometry::simplify(&points, tolerance);
                if geometry::is_usable_ring(&ring) {
                    self.emit_shape(
                        view,
                        file_id,
                        Shape::Polygon {
                            exterior: ring,
                            holes: Vec::new(),
                        },
                        modifiers,
                        effects,
                    );
                }
                effects.push(Effect::Redraw);
            }
            Gesture::Editing {
                pane,
                file_id,
                key,
                original,
                current,
                ..
            } => {
                let committed = self.store.object(&key).map(|o| o.object.shape.clone());
                // An inserted vertex counts as an edit even without movement.
                if Some(&current) != committed.as_ref() || current != original {
                    self.commit_edit(file_id, &key, current, effects);
                }
                let cursor = self.idle_cursor();
                self.set_cursor(pane, cursor, effects);
                effects.push(Effect::Redraw);
            }
            Gesture::Custom { file_id, .. } => {
                if let Some(effect) =
                    self.custom_event(view, file_id, ProtoPointerKind::Up, &input, 0.0)
                {
                    effects.push(effect);
                }
            }
        }
    }

    fn on_leave(&mut self, view: &PaneView, effects: &mut Vec<Effect>) {
        if self.hover.as_ref().is_some_and(|h| h.pane == view.index) {
            let old = self.hover.take();
            self.hover_handle = None;
            effects.push(Effect::Tooltip(None));
            effects.push(Effect::Redraw);
            if let Some(Hover {
                key:
                    ObjectKey {
                        layer:
                            LayerKey {
                                owner: Owner::Plugin(plugin_id),
                                ..
                            },
                        ..
                    },
                ..
            }) = old
            {
                effects.push(Effect::Plugin {
                    plugin_id,
                    event: HostEvent::ObjectHovered {
                        object: None,
                        pane: view.uid,
                    },
                });
            }
        }
        if let Some(drawing) = &mut self.drawing
            && drawing.pane == view.index
            && drawing.hover.take().is_some()
        {
            effects.push(Effect::Redraw);
        }
        if self.tool.is_custom()
            && let Some(file_id) = view.file_id
            && let Some(effect) = self.custom_event(
                view,
                file_id,
                ProtoPointerKind::Leave,
                &PointerInput {
                    kind: PointerKind::Leave,
                    button: PointerButton::None,
                    x: 0.0,
                    y: 0.0,
                    window_x: 0.0,
                    window_y: 0.0,
                    modifiers: Modifiers::default(),
                },
                0.0,
            )
        {
            effects.push(effect);
        }
    }

    fn is_double_click(&self, pane: usize, x: f64, y: f64, now: Instant) -> bool {
        self.last_click.is_some_and(|click| {
            click.pane == pane
                && now.duration_since(click.at) <= DOUBLE_CLICK_INTERVAL
                && (click.x - x).hypot(click.y - y) <= DOUBLE_CLICK_DISTANCE_PX
        })
    }

    fn update_hover(
        &mut self,
        view: &PaneView,
        input: PointerInput,
        point: Point,
        effects: &mut Vec<Effect>,
    ) {
        let Some(file_id) = view.file_id else {
            return;
        };
        if self.tool.is_custom() {
            return;
        }
        let target = if self.drawing.is_some() {
            PressTarget::Empty
        } else {
            self.press_target(view, file_id, point)
        };
        let handle = match &target {
            PressTarget::Handle { key, index } => Some((key.clone(), *index)),
            _ => None,
        };
        if handle != self.hover_handle {
            self.hover_handle = handle;
            effects.push(Effect::Redraw);
        }
        let cursor = match &target {
            PressTarget::Handle { .. } | PressTarget::Body(_) => Cursor::Move,
            PressTarget::Edge { .. } => Cursor::Crosshair,
            PressTarget::Object(_) => Cursor::Pointer,
            PressTarget::Empty => self.idle_cursor(),
        };
        self.set_cursor(view.index, cursor, effects);

        let new_key = target.key().cloned();
        let old_key = self.hover.as_ref().map(|h| h.key.clone());
        // Objects of layers that cannot be picked still show their
        // tooltips (heat maps, predictions), without hover events.
        if new_key.is_none() && old_key.is_none() {
            let passive = self.passive_tooltip_at(view, file_id, point);
            if let Some(text) = &passive {
                effects.push(Effect::Tooltip(Some(TooltipRequest {
                    pane: view.index,
                    window_x: input.window_x,
                    window_y: input.window_y,
                    text: text.clone(),
                })));
            } else if self.passive_tooltip.is_some() {
                effects.push(Effect::Tooltip(None));
            }
            self.passive_tooltip = passive;
            return;
        }
        self.passive_tooltip = None;
        if new_key == old_key {
            if let (Some(hover), Some(text)) = (
                &self.hover,
                self.hover.as_ref().and_then(|h| h.tooltip.clone()),
            ) {
                effects.push(Effect::Tooltip(Some(TooltipRequest {
                    pane: hover.pane,
                    window_x: input.window_x,
                    window_y: input.window_y,
                    text,
                })));
            }
            return;
        }
        effects.push(Effect::Redraw);
        let tooltip = new_key
            .as_ref()
            .and_then(|key| self.store.object(key))
            .and_then(|object| object.object.tooltip.clone());
        effects.push(Effect::Tooltip(tooltip.clone().map(|text| {
            TooltipRequest {
                pane: view.index,
                window_x: input.window_x,
                window_y: input.window_y,
                text,
            }
        })));
        // Owners hear about hover changes on their interactive layers.
        let plugin_of = |key: &Option<ObjectKey>| match key {
            Some(ObjectKey {
                layer:
                    LayerKey {
                        owner: Owner::Plugin(id),
                        ..
                    },
                ..
            }) => Some(Arc::clone(id)),
            _ => None,
        };
        let (old_plugin, new_plugin) = (plugin_of(&old_key), plugin_of(&new_key));
        if let Some(plugin_id) = &old_plugin
            && old_plugin != new_plugin
        {
            effects.push(Effect::Plugin {
                plugin_id: Arc::clone(plugin_id),
                event: HostEvent::ObjectHovered {
                    object: None,
                    pane: view.uid,
                },
            });
        }
        if let (Some(plugin_id), Some(key)) = (&new_plugin, &new_key) {
            effects.push(Effect::Plugin {
                plugin_id: Arc::clone(plugin_id),
                event: HostEvent::ObjectHovered {
                    object: Some(ObjectRef {
                        layer_id: key.layer.id.clone(),
                        object_id: key.object_id.clone(),
                    }),
                    pane: view.uid,
                },
            });
        }
        self.hover = new_key.map(|key| Hover {
            pane: view.index,
            key,
            tooltip,
        });
    }

    /// The tooltip of the topmost object under `point` in a visible layer
    /// that cannot be picked (no interaction, just information).
    fn passive_tooltip_at(&self, view: &PaneView, file_id: FileId, point: Point) -> Option<String> {
        let params = self.pick_params(view, file_id, point);
        self.store
            .hit_test(params, |layer| !self.can_pick(layer))
            .into_iter()
            .find_map(|hit| {
                self.store
                    .object(&hit.key)
                    .and_then(|object| object.object.tooltip.clone())
            })
    }

    /// The topmost pickable object of each owner under `point`.
    fn context_hits(&self, view: &PaneView, file_id: FileId, point: Point) -> Vec<ObjectKey> {
        let params = self.pick_params(view, file_id, point);
        let hits = self.store.hit_test(params, |layer| {
            layer.key.owner == Owner::Host || layer.spec.interactive
        });
        let mut out: Vec<ObjectKey> = Vec::new();
        for hit in hits {
            if !out.iter().any(|k| k.layer.owner == hit.key.layer.owner) {
                out.push(hit.key);
            }
        }
        out.sort_by_key(|key| key.layer.owner != Owner::Host);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn click(
        &mut self,
        view: &PaneView,
        file_id: FileId,
        point: Point,
        target: PressTarget,
        modifiers: Modifiers,
        double: bool,
        effects: &mut Vec<Effect>,
    ) {
        // Click-by-click shapes consume clicks first.
        if self.drawing.is_some() {
            self.drawing_click(view, file_id, point, double, modifiers, effects);
            return;
        }
        // Double-clicking a vertex of an editable polygon/polyline deletes it.
        if double
            && let PressTarget::Handle { key, index } = &target
            && let Some(object) = self.store.object(key)
            && let Some(shape) = geometry::remove_vertex(&object.object.shape, *index)
        {
            self.commit_edit(file_id, key, shape, effects);
            effects.push(Effect::Redraw);
            return;
        }
        let tool = self.tool.clone();
        match (&tool, target.key().cloned()) {
            // Clicking an existing object selects it in every tool except
            // where clicks place new geometry over it.
            (ToolChoice::Plugin { spec, .. }, None) if spec.kind == ToolKind::Point => {
                self.clear_selection(effects);
                self.emit_shape(
                    view,
                    file_id,
                    Shape::Point {
                        x: point[0],
                        y: point[1],
                    },
                    modifiers,
                    effects,
                );
            }
            (ToolChoice::Plugin { spec, .. }, Some(key))
                if spec.kind == ToolKind::Point && matches!(target, PressTarget::Object(_)) =>
            {
                // A non-editable object under a point tool does not block
                // placing a point.
                let _ = key;
                self.emit_shape(
                    view,
                    file_id,
                    Shape::Point {
                        x: point[0],
                        y: point[1],
                    },
                    modifiers,
                    effects,
                );
            }
            (ToolChoice::Plugin { spec, .. }, None)
                if matches!(spec.kind, ToolKind::Polygon | ToolKind::Polyline) =>
            {
                let kind = if spec.kind == ToolKind::Polygon {
                    DrawKind::PluginPolygon
                } else {
                    DrawKind::PluginPolyline
                };
                self.start_drawing(view, file_id, kind, point, effects);
            }
            (ToolChoice::Measure(mode), None) => {
                let kind = match mode {
                    MeasureMode::Line => Some(DrawKind::MeasureLine),
                    MeasureMode::Polyline => Some(DrawKind::MeasurePolyline),
                    MeasureMode::Polygon => Some(DrawKind::MeasurePolygon),
                    MeasureMode::Angle => Some(DrawKind::MeasureAngle),
                    MeasureMode::Rectangle | MeasureMode::Ellipse => None,
                };
                if !self.clear_selection(effects)
                    && let Some(kind) = kind
                {
                    self.start_drawing(view, file_id, kind, point, effects);
                }
            }
            (_, Some(key)) => {
                self.select(&key, modifiers, effects);
                if let Owner::Plugin(plugin_id) = &key.layer.owner {
                    effects.push(Effect::Plugin {
                        plugin_id: Arc::clone(plugin_id),
                        event: HostEvent::ObjectClicked {
                            object: ObjectRef {
                                layer_id: key.layer.id.clone(),
                                object_id: key.object_id.clone(),
                            },
                            pane: view.uid,
                            button: PointerButton::Left,
                            modifiers,
                            click_count: if double { 2 } else { 1 },
                        },
                    });
                }
            }
            (_, None) => {
                self.clear_selection(effects);
                effects.push(Effect::Plugin {
                    // Broadcast: the app delivers it to every plugin.
                    plugin_id: Arc::from(BROADCAST),
                    event: HostEvent::EmptyAreaClicked {
                        pane: view.uid,
                        file_id: Some(file_id),
                        modifiers,
                    },
                });
            }
        }
    }

    fn start_drawing(
        &mut self,
        view: &PaneView,
        file_id: FileId,
        kind: DrawKind,
        point: Point,
        effects: &mut Vec<Effect>,
    ) {
        self.drawing = Some(Drawing {
            pane: view.index,
            file_id,
            kind,
            points: vec![point],
            hover: Some(point),
        });
        effects.push(Effect::Redraw);
    }

    fn drawing_click(
        &mut self,
        view: &PaneView,
        file_id: FileId,
        point: Point,
        double: bool,
        modifiers: Modifiers,
        effects: &mut Vec<Effect>,
    ) {
        let Some(mut drawing) = self.drawing.take() else {
            return;
        };
        if drawing.file_id != file_id || drawing.pane != view.index {
            effects.push(Effect::Redraw);
            return;
        }
        let closes_on_first = drawing.kind.closes()
            && drawing.points.len() >= 3
            && view.screen_distance(drawing.points[0], point) <= HANDLE_RADIUS_PX;
        if double || closes_on_first {
            // The first click of a double click already placed the final
            // vertex; the second one only finishes the shape.
            self.finish_drawing(view, drawing, modifiers, effects);
            return;
        }
        drawing.points.push(point);
        let complete = match drawing.kind {
            DrawKind::MeasureLine => drawing.points.len() == 2,
            DrawKind::MeasureAngle => drawing.points.len() == 3,
            _ => false,
        };
        if complete {
            self.finish_drawing(view, drawing, modifiers, effects);
        } else {
            self.drawing = Some(drawing);
            effects.push(Effect::Redraw);
        }
    }

    /// Completes the click-by-click shape (Enter, double click, closing click).
    fn finish_drawing(
        &mut self,
        view: &PaneView,
        mut drawing: Drawing,
        modifiers: Modifiers,
        effects: &mut Vec<Effect>,
    ) {
        effects.push(Effect::Redraw);
        // Drop consecutive duplicates (double clicks, accidental taps).
        drawing
            .points
            .dedup_by(|a, b| view.screen_distance(*a, *b) < 1.0);
        if drawing.points.len() < drawing.kind.min_points() {
            return;
        }
        let file_id = drawing.file_id;
        match drawing.kind {
            DrawKind::MeasureLine | DrawKind::MeasurePolyline | DrawKind::MeasureAngle => {
                let mode = match drawing.kind {
                    DrawKind::MeasureLine => MeasureMode::Line,
                    DrawKind::MeasureAngle => MeasureMode::Angle,
                    _ => MeasureMode::Polyline,
                };
                self.add_measurement(
                    file_id,
                    mode,
                    Shape::Polyline {
                        points: drawing.points,
                    },
                    effects,
                );
            }
            DrawKind::MeasurePolygon => {
                if geometry::is_usable_ring(&drawing.points) {
                    self.add_measurement(
                        file_id,
                        MeasureMode::Polygon,
                        Shape::Polygon {
                            exterior: drawing.points,
                            holes: Vec::new(),
                        },
                        effects,
                    );
                }
            }
            DrawKind::PluginPolyline => {
                self.emit_shape(
                    view,
                    file_id,
                    Shape::Polyline {
                        points: drawing.points,
                    },
                    modifiers,
                    effects,
                );
            }
            DrawKind::PluginPolygon => {
                if geometry::is_usable_ring(&drawing.points) {
                    self.emit_shape(
                        view,
                        file_id,
                        Shape::Polygon {
                            exterior: drawing.points,
                            holes: Vec::new(),
                        },
                        modifiers,
                        effects,
                    );
                }
            }
        }
    }

    fn add_measurement(
        &mut self,
        file_id: FileId,
        mode: MeasureMode,
        shape: Shape,
        effects: &mut Vec<Effect>,
    ) {
        self.measurement_counter += 1;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        effects.push(Effect::AddMeasurement {
            file_id,
            measurement: HostMeasurement {
                // The mode prefix lets edits recompute the right value.
                id: format!("{}:{:x}-{}", mode.id(), nanos, self.measurement_counter),
                mode,
                shape,
            },
        });
    }

    fn emit_shape(
        &self,
        view: &PaneView,
        file_id: FileId,
        shape: Shape,
        modifiers: Modifiers,
        effects: &mut Vec<Effect>,
    ) {
        if let Some((plugin_id, spec)) = self.tool.plugin_tool() {
            effects.push(Effect::Plugin {
                plugin_id: Arc::clone(plugin_id),
                event: HostEvent::ShapeDrawn {
                    tool_id: spec.tool_id.clone(),
                    pane: view.uid,
                    file_id,
                    shape,
                    modifiers,
                },
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_drag_shape(
        &mut self,
        view: &PaneView,
        file_id: FileId,
        kind: DragShape,
        start: Point,
        end: Point,
        modifiers: Modifiers,
        effects: &mut Vec<Effect>,
    ) {
        let rect = geometry::rect_from_corners(start, end);
        let zoom = view.zoom();
        let big_enough =
            rect.width * zoom >= MIN_DRAG_SHAPE_PX || rect.height * zoom >= MIN_DRAG_SHAPE_PX;
        let area_shape = |ellipse: bool| {
            if ellipse {
                Shape::Ellipse {
                    cx: rect.x + rect.width / 2.0,
                    cy: rect.y + rect.height / 2.0,
                    rx: rect.width / 2.0,
                    ry: rect.height / 2.0,
                }
            } else {
                measure::roi_shape(&rect)
            }
        };
        match kind {
            DragShape::Roi => {
                if rect.width * zoom >= MIN_DRAG_SHAPE_PX && rect.height * zoom >= MIN_DRAG_SHAPE_PX
                {
                    effects.push(Effect::SetRoi {
                        file_id,
                        rect: Some(rect),
                    });
                }
            }
            DragShape::MeasureLine => {
                if view.screen_distance(start, end) >= MIN_DRAG_SHAPE_PX {
                    self.add_measurement(
                        file_id,
                        MeasureMode::Line,
                        Shape::Polyline {
                            points: vec![start, end],
                        },
                        effects,
                    );
                }
            }
            DragShape::MeasureRect | DragShape::MeasureEllipse => {
                if big_enough {
                    let ellipse = kind == DragShape::MeasureEllipse;
                    self.add_measurement(
                        file_id,
                        if ellipse {
                            MeasureMode::Ellipse
                        } else {
                            MeasureMode::Rectangle
                        },
                        area_shape(ellipse),
                        effects,
                    );
                }
            }
            DragShape::PluginRect | DragShape::PluginEllipse => {
                if big_enough {
                    self.emit_shape(
                        view,
                        file_id,
                        area_shape(kind == DragShape::PluginEllipse),
                        modifiers,
                        effects,
                    );
                }
            }
            DragShape::Marquee => self.marquee_select(file_id, &rect, modifiers, effects),
        }
    }

    fn marquee_select(
        &mut self,
        file_id: FileId,
        rect: &Rect,
        modifiers: Modifiers,
        effects: &mut Vec<Effect>,
    ) {
        let Some((plugin_id, _)) = self.tool.plugin_tool() else {
            return;
        };
        let owner = Owner::Plugin(Arc::clone(plugin_id));
        let mut changed = Vec::new();
        if !(modifiers.primary || modifiers.shift) {
            changed.extend(self.store.clear_selection(None));
        }
        for layer in self.store.file_layers(file_id).layers {
            if layer.key.owner != owner || !layer.spec.visible || !self.can_pick(&layer) {
                continue;
            }
            let mut ids: Vec<String> = layer.selection().cloned().collect();
            for object in layer.visible_objects(rect, 1e12) {
                let b = &object.bounds;
                let inside = b.x >= rect.x
                    && b.y >= rect.y
                    && b.x + b.width <= rect.x + rect.width
                    && b.y + b.height <= rect.y + rect.height;
                if inside && !ids.contains(&object.object.object_id) {
                    ids.push(object.object.object_id.clone());
                }
            }
            if self
                .store
                .select(&owner, &layer.key.id, &ids)
                .unwrap_or(false)
            {
                changed.push(owner.clone());
            }
        }
        if !changed.is_empty() {
            effects.push(Effect::Redraw);
        }
        self.notify_selection(changed, effects);
    }

    /// Applies an edit of an editable object: host objects become host
    /// edits (undoable), plugin objects are updated optimistically and the
    /// owner is told.
    fn commit_edit(
        &mut self,
        file_id: FileId,
        key: &ObjectKey,
        shape: Shape,
        effects: &mut Vec<Effect>,
    ) {
        match &key.layer.owner {
            Owner::Host if is_roi_layer(&key.layer.id) => {
                effects.push(Effect::SetRoi {
                    file_id,
                    rect: Some(shape.bounds()),
                });
            }
            Owner::Host => effects.push(Effect::UpdateMeasurement {
                file_id,
                id: key.object_id.clone(),
                shape,
            }),
            Owner::Plugin(plugin_id) => {
                if let Some(object) = self.store.object(key) {
                    let mut updated = object.object.clone();
                    updated.shape = shape.clone();
                    if let Err(err) =
                        self.store
                            .upsert(&key.layer.owner, &key.layer.id, vec![updated], false)
                    {
                        tracing::warn!(
                            "edit of {}/{} rejected: {err}",
                            key.layer.id,
                            key.object_id
                        );
                        return;
                    }
                }
                effects.push(Effect::Plugin {
                    plugin_id: Arc::clone(plugin_id),
                    event: HostEvent::ObjectEdited {
                        object: ObjectRef {
                            layer_id: key.layer.id.clone(),
                            object_id: key.object_id.clone(),
                        },
                        shape,
                    },
                });
            }
        }
    }

    // -- wheel and gestures --------------------------------------------------------

    pub fn wheel(
        &mut self,
        view: &PaneView,
        x: f64,
        y: f64,
        dx: f64,
        dy: f64,
        modifiers: Modifiers,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let zoom_modifier = modifiers.primary || modifiers.control;
        let horizontal_only = dy == 0.0 && dx != 0.0;
        if modifiers.shift && !zoom_modifier {
            // Shift+wheel scrolls sideways.
            let delta = if dx != 0.0 { dx } else { dy };
            effects.push(Effect::PanBy {
                pane: view.index,
                dx: delta,
                dy: 0.0,
            });
        } else if horizontal_only || (self.scroll_pans && !zoom_modifier) {
            effects.push(Effect::PanBy {
                pane: view.index,
                dx,
                dy,
            });
        } else if dy != 0.0 {
            let notches = (dy / WHEEL_PX_PER_NOTCH).clamp(-5.0, 5.0);
            effects.push(Effect::Zoom {
                pane: view.index,
                factor: WHEEL_ZOOM_PER_NOTCH.powf(notches),
                x,
                y,
            });
        }
        if self.tool.is_custom()
            && let Some(file_id) = view.file_id
            && let Some(effect) = self.custom_event(
                view,
                file_id,
                ProtoPointerKind::Wheel,
                &PointerInput {
                    kind: PointerKind::Move,
                    button: PointerButton::None,
                    x,
                    y,
                    window_x: x,
                    window_y: y,
                    modifiers,
                },
                dy,
            )
        {
            effects.push(effect);
        }
        effects
    }

    /// Pinch gestures: `scale` is cumulative since the gesture started.
    pub fn pinch(
        &mut self,
        view: &PaneView,
        scale: f64,
        x: f64,
        y: f64,
        active: bool,
    ) -> Vec<Effect> {
        if !active || !scale.is_finite() || scale <= 0.0 {
            self.pinch_scale = None;
            return Vec::new();
        }
        let previous = self.pinch_scale.replace(scale).unwrap_or(1.0);
        let factor = scale / previous;
        if (factor - 1.0).abs() < 1e-4 {
            return Vec::new();
        }
        vec![Effect::Zoom {
            pane: view.index,
            factor,
            x,
            y,
        }]
    }

    // -- keyboard ----------------------------------------------------------------

    /// Handles a key in the viewport. Returns the effects and whether the key
    /// was consumed.
    pub fn key(&mut self, view: Option<&PaneView>, input: &KeyInput) -> (Vec<Effect>, bool) {
        let mut effects = Vec::new();
        if self.tool.is_custom() {
            if let Some((plugin_id, spec)) = self.tool.plugin_tool() {
                effects.push(Effect::Plugin {
                    plugin_id: Arc::clone(plugin_id),
                    event: HostEvent::Key {
                        tool_id: Some(spec.tool_id.clone()),
                        pane: view.map(|v| v.uid),
                        key: input.text.clone(),
                        modifiers: input.modifiers,
                        pressed: input.pressed,
                        repeat: input.repeat,
                    },
                });
            }
            // The tool sees every key, but Tab and the app's Ctrl/⌘
            // shortcuts (undo, close, open, …) still work while it is active.
            return (effects, input.text != keys::TAB && !input.modifiers.primary);
        }
        if !input.pressed {
            return (effects, false);
        }
        match input.text.as_str() {
            keys::ESCAPE => {
                if self.is_busy() {
                    effects.extend(self.cancel());
                } else if !self.clear_selection(&mut effects) && self.tool != ToolChoice::Navigate {
                    effects.extend(self.set_tool(ToolChoice::Navigate));
                }
                (effects, true)
            }
            keys::RETURN => {
                if let (Some(drawing), Some(view)) = (self.drawing.take(), view) {
                    self.finish_drawing(view, drawing, input.modifiers, &mut effects);
                    (effects, true)
                } else {
                    (effects, false)
                }
            }
            keys::BACKSPACE if self.drawing.is_some() => {
                if let Some(drawing) = &mut self.drawing {
                    drawing.points.pop();
                    if drawing.points.is_empty() {
                        self.drawing = None;
                    }
                }
                effects.push(Effect::Redraw);
                (effects, true)
            }
            keys::DELETE | keys::BACKSPACE => {
                let consumed = self.delete_selection(view.and_then(|v| v.file_id), &mut effects);
                (effects, consumed)
            }
            _ => (effects, false),
        }
    }

    /// Deletes the selection: host objects directly, plugin objects by asking
    /// their owners.
    pub fn delete_selection(&mut self, file_id: Option<FileId>, effects: &mut Vec<Effect>) -> bool {
        let mut consumed = false;
        let host = self.store.selection_of(&Owner::Host);
        if let Some(file_id) = file_id {
            let measurements: Vec<String> = host
                .iter()
                .filter(|key| key.layer.id == measure::measurement_layer_id(file_id))
                .map(|key| key.object_id.clone())
                .collect();
            if !measurements.is_empty() {
                effects.push(Effect::RemoveMeasurements {
                    file_id,
                    ids: measurements,
                });
                consumed = true;
            }
            if host
                .iter()
                .any(|key| key.layer.id == measure::roi_layer_id(file_id))
            {
                effects.push(Effect::SetRoi {
                    file_id,
                    rect: None,
                });
                consumed = true;
            }
        }
        let mut by_plugin: HashMap<Arc<str>, Vec<ObjectRef>> = HashMap::new();
        for owner in self.plugin_owners_with_selection() {
            if let Owner::Plugin(plugin_id) = &owner {
                let refs = self
                    .store
                    .selection_of(&owner)
                    .into_iter()
                    .map(|key| ObjectRef {
                        layer_id: key.layer.id,
                        object_id: key.object_id,
                    })
                    .collect::<Vec<_>>();
                if !refs.is_empty() {
                    by_plugin.insert(Arc::clone(plugin_id), refs);
                }
            }
        }
        let mut plugins: Vec<_> = by_plugin.into_iter().collect();
        plugins.sort_by(|a, b| a.0.cmp(&b.0));
        for (plugin_id, objects) in plugins {
            consumed = true;
            effects.push(Effect::Plugin {
                plugin_id,
                event: HostEvent::ObjectsDeleteRequested { objects },
            });
        }
        consumed
    }

    fn plugin_owners_with_selection(&self) -> Vec<Owner> {
        // Every plugin that owns a layer with a selection.
        let mut owners: Vec<Owner> = Vec::new();
        for owner in self.store_owners() {
            if matches!(owner, Owner::Plugin(_)) && !self.store.selection_of(&owner).is_empty() {
                owners.push(owner);
            }
        }
        owners
    }

    fn store_owners(&self) -> Vec<Owner> {
        let mut owners = Vec::new();
        for key in self.store.selected_layer_keys() {
            if !owners.contains(&key.owner) {
                owners.push(key.owner);
            }
        }
        owners
    }

    // -- scene for rendering -------------------------------------------------------

    /// The hovered object, edit previews and transient shapes to draw in
    /// `pane` (showing `file_id`).
    pub fn scene_extras(
        &self,
        view: &PaneView,
    ) -> (Option<ObjectKey>, HashMap<ObjectKey, Shape>, Vec<Transient>) {
        let mut overrides = HashMap::new();
        let mut transients = Vec::new();
        let Some(file_id) = view.file_id else {
            return (None, overrides, transients);
        };
        let hovered = self
            .hover
            .as_ref()
            .filter(|h| h.pane == view.index)
            .map(|h| h.key.clone());
        let preview = measure::preview_style();

        match &self.gesture {
            Gesture::DragShape {
                pane,
                file_id: f,
                kind,
                start,
                current,
                ..
            } if *pane == view.index && *f == file_id => {
                let rect = geometry::rect_from_corners(*start, *current);
                let (shape, label_mode) = match kind {
                    DragShape::MeasureLine => (
                        Shape::Polyline {
                            points: vec![*start, *current],
                        },
                        Some(MeasureMode::Line),
                    ),
                    DragShape::MeasureEllipse | DragShape::PluginEllipse => (
                        Shape::Ellipse {
                            cx: rect.x + rect.width / 2.0,
                            cy: rect.y + rect.height / 2.0,
                            rx: rect.width / 2.0,
                            ry: rect.height / 2.0,
                        },
                        (*kind == DragShape::MeasureEllipse).then_some(MeasureMode::Ellipse),
                    ),
                    DragShape::MeasureRect => {
                        (measure::roi_shape(&rect), Some(MeasureMode::Rectangle))
                    }
                    DragShape::Roi | DragShape::PluginRect | DragShape::Marquee => {
                        (measure::roi_shape(&rect), None)
                    }
                };
                let style = match kind {
                    DragShape::Marquee => Style {
                        fill: eov_plugin_api::protocol::Rgba(0x1E, 0x90, 0xFF, 0x20),
                        stroke: eov_plugin_api::protocol::Rgba(0x1E, 0x90, 0xFF, 0xFF),
                        stroke_width: 1.0,
                        ..Style::default()
                    },
                    DragShape::PluginRect | DragShape::PluginEllipse => self.plugin_preview_style(),
                    _ => preview.clone(),
                };
                let mut transient = Transient::new(shape.clone(), style);
                if *kind == DragShape::Roi {
                    transient.decoration.dim_outside = true;
                    transient.decoration.contrast_border = true;
                }
                if let Some(mode) = label_mode {
                    let value = measure::measure(mode, &shape, &view.calibration);
                    transient.label = Some(measure::label(mode, &value, view.unit));
                }
                transients.push(transient);
            }
            Gesture::Freehand {
                pane,
                file_id: f,
                points,
                ..
            } if *pane == view.index && *f == file_id && points.len() >= 2 => {
                transients.push(Transient::new(
                    Shape::Polyline {
                        points: points.clone(),
                    },
                    self.plugin_preview_style(),
                ));
            }
            Gesture::Editing {
                key,
                current,
                file_id: f,
                ..
            } if *f == file_id => {
                overrides.insert(key.clone(), current.clone());
                if key.layer.owner == Owner::Host && is_measurement_layer(&key.layer.id) {
                    // Live value while dragging a measurement handle.
                    let mode = self.measurement_mode(file_id, &key.object_id);
                    let value = measure::measure(mode, current, &view.calibration);
                    let mut transient = Transient::new(current.clone(), preview.clone());
                    transient.label = Some(measure::label(mode, &value, view.unit));
                    transients.push(transient);
                }
            }
            _ => {}
        }

        if let Some(drawing) = &self.drawing
            && drawing.pane == view.index
            && drawing.file_id == file_id
        {
            let mut points = drawing.points.clone();
            if let Some(hover) = drawing.hover {
                points.push(hover);
            }
            let style = match drawing.kind {
                DrawKind::PluginPolygon | DrawKind::PluginPolyline => self.plugin_preview_style(),
                _ => preview.clone(),
            };
            if points.len() >= 2 {
                let closes = drawing.kind.closes() && points.len() >= 3;
                let shape = if closes {
                    Shape::Polygon {
                        exterior: points.clone(),
                        holes: Vec::new(),
                    }
                } else {
                    Shape::Polyline {
                        points: points.clone(),
                    }
                };
                let mut transient = Transient::new(shape.clone(), style.clone());
                let mode = match drawing.kind {
                    DrawKind::MeasureLine => Some(MeasureMode::Line),
                    DrawKind::MeasurePolyline => Some(MeasureMode::Polyline),
                    DrawKind::MeasurePolygon => Some(MeasureMode::Polygon),
                    DrawKind::MeasureAngle if points.len() == 3 => Some(MeasureMode::Angle),
                    _ => None,
                };
                if let Some(mode) = mode {
                    let value = measure::measure(mode, &shape, &view.calibration);
                    transient.label = Some(measure::label(mode, &value, view.unit));
                }
                transients.push(transient);
            }
            for (i, p) in drawing.points.iter().enumerate() {
                let mut handle = measure::handle_style();
                if i == 0 && drawing.kind.closes() && drawing.points.len() >= 3 {
                    // The first vertex closes the polygon.
                    handle.fill = eov_plugin_api::protocol::Rgba(0xF1, 0xC4, 0x0F, 0xFF);
                }
                transients.push(Transient::new(Shape::Point { x: p[0], y: p[1] }, handle));
            }
        }

        // Handles of editable objects.
        let mut handles = 0;
        for (key, shape) in self.handle_owners(file_id) {
            let shape = overrides.get(&key).unwrap_or(&shape);
            if matches!(shape, Shape::Point { .. } | Shape::Text { .. }) {
                continue;
            }
            for (index, p) in geometry::shape_handles(shape).into_iter().enumerate() {
                if handles >= MAX_HANDLES {
                    break;
                }
                handles += 1;
                let mut style = measure::handle_style();
                if self.hover_handle.as_ref() == Some(&(key.clone(), index)) {
                    style.fill = eov_plugin_api::protocol::Rgba(0x1E, 0x90, 0xFF, 0xFF);
                    style.point_radius += 1.0;
                }
                transients.push(Transient::new(Shape::Point { x: p[0], y: p[1] }, style));
            }
        }
        (hovered, overrides, transients)
    }

    fn plugin_preview_style(&self) -> Style {
        self.tool
            .plugin_tool()
            .and_then(|(_, spec)| spec.preview_style.clone())
            .unwrap_or_else(measure::preview_style)
    }

    fn measurement_mode(&self, _file_id: FileId, id: &str) -> MeasureMode {
        measurement_mode_of(id)
    }
}

/// `Effect::Plugin` target meaning "every plugin".
pub const BROADCAST: &str = "*";

/// The mode of a host measurement, from its id prefix.
pub fn measurement_mode_of(id: &str) -> MeasureMode {
    id.split_once(':')
        .and_then(|(mode, _)| MeasureMode::from_id(mode))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::{LayerSpec, OverlayObject};

    #[allow(non_snake_case)]
    fn T0() -> Instant {
        Instant::now()
    }

    fn view() -> PaneView {
        let mut viewport = eov_common::Viewport::new(1000.0, 800.0, 100_000.0, 80_000.0);
        viewport.zoom = 1.0;
        viewport.center.x = 500.0;
        viewport.center.y = 400.0;
        PaneView {
            index: 0,
            uid: 7,
            file_id: Some(1),
            viewport,
            calibration: Calibration::new(Some(0.5), Some(0.5)),
            unit: MeasurementUnit::Um,
        }
    }

    fn input(kind: PointerKind, button: PointerButton, x: f64, y: f64) -> PointerInput {
        PointerInput {
            kind,
            button,
            x,
            y,
            window_x: x,
            window_y: y,
            modifiers: Modifiers::default(),
        }
    }

    fn click(c: &mut Controller, v: &PaneView, x: f64, y: f64, at: Instant) -> Vec<Effect> {
        let mut effects = c.pointer(v, input(PointerKind::Down, PointerButton::Left, x, y), at);
        effects.extend(c.pointer(v, input(PointerKind::Up, PointerButton::Left, x, y), at));
        effects
    }

    fn drag(
        c: &mut Controller,
        v: &PaneView,
        button: PointerButton,
        from: (f64, f64),
        to: (f64, f64),
    ) -> Vec<Effect> {
        let now = T0();
        let mut effects = c.pointer(v, input(PointerKind::Down, button, from.0, from.1), now);
        effects.extend(c.pointer(
            v,
            input(
                PointerKind::Move,
                button,
                (from.0 + to.0) / 2.0,
                (from.1 + to.1) / 2.0,
            ),
            now,
        ));
        effects.extend(c.pointer(v, input(PointerKind::Move, button, to.0, to.1), now));
        effects.extend(c.pointer(v, input(PointerKind::Up, button, to.0, to.1), now));
        effects
    }

    fn plugin_tool(kind: ToolKind) -> ToolChoice {
        ToolChoice::Plugin {
            plugin_id: Arc::from("anno"),
            spec: ToolSpec::new("t", "Tool", "<svg/>", kind),
        }
    }

    fn plugin_events(effects: &[Effect]) -> Vec<&HostEvent> {
        effects
            .iter()
            .filter_map(|e| match e {
                Effect::Plugin { event, .. } => Some(event),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn navigate_drag_pans_and_right_click_opens_the_menu() {
        let store = Arc::new(OverlayStore::new());
        let mut c = Controller::new(store);
        let v = view();
        let effects = drag(&mut c, &v, PointerButton::Left, (10.0, 10.0), (100.0, 50.0));
        assert!(matches!(effects[0], Effect::StartPan { x, y, .. } if x == 10.0 && y == 10.0));
        assert!(effects.iter().any(|e| matches!(e, Effect::EndPan { .. })));

        let now = T0();
        c.pointer(
            &v,
            input(PointerKind::Down, PointerButton::Right, 20.0, 20.0),
            now,
        );
        let effects = c.pointer(
            &v,
            input(PointerKind::Up, PointerButton::Right, 20.0, 20.0),
            now,
        );
        assert!(matches!(effects[0], Effect::ContextMenu { .. }));
    }

    #[test]
    fn wheel_zoom_scales_with_the_delta_and_horizontal_scroll_pans() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        let v = view();
        let effects = c.wheel(&v, 10.0, 10.0, 0.0, 60.0, Modifiers::default());
        assert!(
            matches!(effects[0], Effect::Zoom { factor, .. } if (factor - WHEEL_ZOOM_PER_NOTCH).abs() < 1e-9)
        );
        let effects = c.wheel(&v, 10.0, 10.0, 0.0, 6.0, Modifiers::default());
        assert!(matches!(effects[0], Effect::Zoom { factor, .. } if factor < 1.05 && factor > 1.0));
        let effects = c.wheel(&v, 10.0, 10.0, 30.0, 0.0, Modifiers::default());
        assert!(matches!(effects[0], Effect::PanBy { dx, .. } if dx == 30.0));
    }

    #[test]
    fn measure_line_by_drag_and_by_two_clicks() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(ToolChoice::Measure(MeasureMode::Line));
        let v = view();
        let effects = drag(&mut c, &v, PointerButton::Left, (0.0, 0.0), (100.0, 0.0));
        let Some(Effect::AddMeasurement { measurement, .. }) = effects
            .iter()
            .find(|e| matches!(e, Effect::AddMeasurement { .. }))
        else {
            panic!("no measurement: {effects:?}");
        };
        assert_eq!(measurement.mode, MeasureMode::Line);
        assert_eq!(geometry::shape_length(&measurement.shape), 100.0);

        let t = T0();
        click(&mut c, &v, 0.0, 0.0, t);
        let effects = click(&mut c, &v, 0.0, 50.0, t + Duration::from_secs(1));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::AddMeasurement { .. }))
        );
    }

    #[test]
    fn polygon_tool_closes_on_the_first_vertex_and_reports_the_shape() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(plugin_tool(ToolKind::Polygon));
        let v = view();
        let t = T0();
        let s = Duration::from_secs(1);
        click(&mut c, &v, 0.0, 0.0, t);
        click(&mut c, &v, 100.0, 0.0, t + s);
        click(&mut c, &v, 100.0, 100.0, t + 2 * s);
        let effects = click(&mut c, &v, 2.0, 1.0, t + 3 * s);
        let events = plugin_events(&effects);
        let Some(HostEvent::ShapeDrawn {
            shape: Shape::Polygon { exterior, .. },
            pane,
            ..
        }) = events.first()
        else {
            panic!("no polygon: {effects:?}");
        };
        assert_eq!(exterior.len(), 3);
        assert_eq!(*pane, 7, "events carry the stable pane id");
        assert!(!c.is_busy());
    }

    #[test]
    fn double_click_finishes_a_polyline_without_a_duplicate_vertex() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(plugin_tool(ToolKind::Polyline));
        let v = view();
        let t = T0();
        click(&mut c, &v, 0.0, 0.0, t);
        click(&mut c, &v, 50.0, 0.0, t + Duration::from_secs(1));
        let t2 = t + Duration::from_secs(2);
        click(&mut c, &v, 50.0, 50.0, t2);
        let effects = click(&mut c, &v, 50.0, 50.0, t2 + Duration::from_millis(100));
        let Some(HostEvent::ShapeDrawn {
            shape: Shape::Polyline { points },
            ..
        }) = plugin_events(&effects).first().copied()
        else {
            panic!("no polyline: {effects:?}");
        };
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn escape_cancels_then_deselects_then_returns_to_navigate() {
        let store = Arc::new(OverlayStore::new());
        let mut c = Controller::new(Arc::clone(&store));
        c.set_tool(plugin_tool(ToolKind::Polygon));
        let v = view();
        click(&mut c, &v, 0.0, 0.0, T0());
        assert!(c.is_busy());
        let escape = KeyInput {
            text: keys::ESCAPE.into(),
            modifiers: Modifiers::default(),
            pressed: true,
            repeat: false,
        };
        c.key(Some(&v), &escape);
        assert!(!c.is_busy());
        assert!(matches!(c.tool(), ToolChoice::Plugin { .. }));
        let (effects, consumed) = c.key(Some(&v), &escape);
        assert!(consumed);
        assert_eq!(c.tool(), &ToolChoice::Navigate);
        assert!(
            plugin_events(&effects)
                .iter()
                .any(|e| matches!(e, HostEvent::ToolDeactivated { .. }))
        );
    }

    #[test]
    fn clicking_interactive_objects_selects_and_notifies_the_owner() {
        let store = Arc::new(OverlayStore::new());
        let owner = Owner::plugin("anno");
        store
            .create_layer(owner.clone(), LayerSpec::new("pts", 1).interactive(true))
            .unwrap();
        // Screen (100, 100) is image (100, 100) at zoom 1 centred on (500, 400).
        let v = view();
        let image = v.to_image(100.0, 100.0);
        store
            .upsert(
                &owner,
                "pts",
                vec![OverlayObject::new(
                    "p1",
                    Shape::Point {
                        x: image[0],
                        y: image[1],
                    },
                )],
                false,
            )
            .unwrap();
        let mut c = Controller::new(Arc::clone(&store));
        let effects = click(&mut c, &v, 101.0, 101.0, T0());
        let events = plugin_events(&effects);
        assert!(
            events.iter().any(
                |e| matches!(e, HostEvent::SelectionChanged { objects } if objects.len() == 1)
            )
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, HostEvent::ObjectClicked { click_count: 1, .. }))
        );
        assert_eq!(store.selection_of(&owner).len(), 1);

        // Delete asks the owner to delete.
        let mut effects = Vec::new();
        assert!(c.delete_selection(Some(1), &mut effects));
        assert!(plugin_events(&effects).iter().any(
            |e| matches!(e, HostEvent::ObjectsDeleteRequested { objects } if objects.len() == 1)
        ));

        // Clicking empty space clears the selection and broadcasts.
        let effects = click(&mut c, &v, 600.0, 600.0, T0() + Duration::from_secs(5));
        assert!(store.selection_of(&owner).is_empty());
        assert!(effects
            .iter()
            .any(|e| matches!(e, Effect::Plugin { plugin_id, event: HostEvent::EmptyAreaClicked { .. } } if &**plugin_id == BROADCAST)));
    }

    #[test]
    fn editable_vertices_can_be_dragged_with_the_owning_tool() {
        let store = Arc::new(OverlayStore::new());
        let owner = Owner::plugin("anno");
        store
            .create_layer(
                owner.clone(),
                LayerSpec::new("polys", 1).interactive(true).editable(true),
            )
            .unwrap();
        let v = view();
        let a = v.to_image(100.0, 100.0);
        let b = v.to_image(200.0, 100.0);
        let d = v.to_image(150.0, 200.0);
        store
            .upsert(
                &owner,
                "polys",
                vec![OverlayObject::new(
                    "tri",
                    Shape::Polygon {
                        exterior: vec![a, b, d],
                        holes: vec![],
                    },
                )],
                false,
            )
            .unwrap();
        let mut c = Controller::new(Arc::clone(&store));
        c.set_tool(plugin_tool(ToolKind::Polygon));
        // Hover the polygon so its handles appear, then drag vertex 1.
        c.pointer(
            &v,
            input(PointerKind::Move, PointerButton::None, 150.0, 130.0),
            T0(),
        );
        let effects = drag(
            &mut c,
            &v,
            PointerButton::Left,
            (200.0, 100.0),
            (250.0, 120.0),
        );
        let edited = plugin_events(&effects)
            .into_iter()
            .find_map(|e| match e {
                HostEvent::ObjectEdited {
                    shape: Shape::Polygon { exterior, .. },
                    ..
                } => Some(exterior.clone()),
                _ => None,
            })
            .expect("edited");
        assert_eq!(edited[1], v.to_image(250.0, 120.0));
        // The store shows the new geometry right away.
        let stored = store
            .object(&ObjectKey {
                layer: LayerKey::new(owner, "polys"),
                object_id: "tri".into(),
            })
            .unwrap();
        assert_eq!(
            stored.object.shape,
            Shape::Polygon {
                exterior: edited,
                holes: vec![]
            }
        );
    }

    #[test]
    fn a_shape_in_progress_does_not_survive_a_file_change() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(ToolChoice::Measure(MeasureMode::Polyline));
        let v = view();
        click(&mut c, &v, 0.0, 0.0, T0());
        assert!(c.is_busy());
        let other = PaneView {
            file_id: Some(2),
            ..view()
        };
        // Clicking in the same pane after the tab switched to another slide
        // starts afresh instead of mixing coordinates.
        c.pointer(
            &other,
            input(PointerKind::Down, PointerButton::Left, 5.0, 5.0),
            T0(),
        );
        assert!(c.drawing.is_none());
        c.cancel();
        click(&mut c, &v, 0.0, 0.0, T0());
        c.pane_changed(0);
        assert!(!c.is_busy());
    }

    #[test]
    fn roi_drag_sets_the_roi_and_shows_a_preview() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(ToolChoice::Roi);
        let v = view();
        let now = T0();
        c.pointer(
            &v,
            input(PointerKind::Down, PointerButton::Left, 10.0, 10.0),
            now,
        );
        c.pointer(
            &v,
            input(PointerKind::Move, PointerButton::Left, 60.0, 80.0),
            now,
        );
        let (_, _, transients) = c.scene_extras(&v);
        assert_eq!(transients.len(), 1);
        assert!(transients[0].decoration.dim_outside);
        let effects = c.pointer(
            &v,
            input(PointerKind::Up, PointerButton::Left, 60.0, 80.0),
            now,
        );
        let Some(Effect::SetRoi {
            rect: Some(rect), ..
        }) = effects.iter().find(|e| matches!(e, Effect::SetRoi { .. }))
        else {
            panic!("{effects:?}");
        };
        assert_eq!((rect.width, rect.height), (50.0, 70.0));
    }

    #[test]
    fn custom_tools_receive_raw_pointer_events() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(plugin_tool(ToolKind::Custom));
        let v = view();
        let effects = drag(&mut c, &v, PointerButton::Left, (0.0, 0.0), (30.0, 0.0));
        let kinds: Vec<_> = plugin_events(&effects)
            .into_iter()
            .filter_map(|e| match e {
                HostEvent::Pointer { event, .. } => Some(event.kind),
                _ => None,
            })
            .collect();
        assert_eq!(
            kinds,
            [
                ProtoPointerKind::Down,
                ProtoPointerKind::Move,
                ProtoPointerKind::Move,
                ProtoPointerKind::Up
            ]
        );
    }

    #[test]
    fn freehand_produces_a_simplified_polygon() {
        let mut c = Controller::new(Arc::new(OverlayStore::new()));
        c.set_tool(plugin_tool(ToolKind::Freehand));
        let v = view();
        let now = T0();
        c.pointer(
            &v,
            input(PointerKind::Down, PointerButton::Left, 0.0, 0.0),
            now,
        );
        for i in 1..=20 {
            c.pointer(
                &v,
                input(PointerKind::Move, PointerButton::Left, i as f64 * 5.0, 0.0),
                now,
            );
        }
        for i in 1..=20 {
            c.pointer(
                &v,
                input(
                    PointerKind::Move,
                    PointerButton::Left,
                    100.0,
                    i as f64 * 5.0,
                ),
                now,
            );
        }
        let effects = c.pointer(
            &v,
            input(PointerKind::Up, PointerButton::Left, 0.0, 100.0),
            now,
        );
        let Some(HostEvent::ShapeDrawn {
            shape: Shape::Polygon { exterior, .. },
            ..
        }) = plugin_events(&effects).first().copied()
        else {
            panic!("{effects:?}");
        };
        assert!(exterior.len() <= 5, "{exterior:?}");
    }
}
