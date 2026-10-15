//! The message protocol spoken between the eov host and plugins.
//!
//! The binary ABI ([`crate::abi`]) is deliberately tiny: almost every
//! interaction is a JSON message whose schema is defined here as serde types
//! shared by both sides. This keeps the ABI stable while the protocol grows:
//!
//! - new optional fields are added with `#[serde(default)]`, so older peers
//!   ignore them and newer peers tolerate their absence;
//! - new event variants deserialize as [`HostEvent::Unknown`] in older plugins;
//! - new host methods are answered with an `unknown method` error by older
//!   hosts, so plugins can feature-detect them.
//!
//! # Coordinates
//!
//! All image coordinates are **level-0 pixels** of the slide (`x` to the right,
//! `y` down, origin at the top-left of the level-0 canvas as OpenSlide reports
//! it). Screen coordinates are logical pixels relative to the top-left corner of
//! the pane. `zoom` is screen pixels per level-0 pixel.
//!
//! # Threading contract
//!
//! - [`HostEvent`]s are delivered on the host's UI thread, one at a time, with
//!   no host locks held. Handlers must return quickly (aim for < 5 ms); long
//!   work belongs on a plugin-owned thread.
//! - Every host method may be called from any thread at any time. Calls never
//!   block on the UI thread: queries answer from a recent published snapshot
//!   and mutations are queued and applied on the UI thread in call order.
//! - CPU filters may run on any render worker thread, concurrently for
//!   different panes.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Host-assigned identifier of an open slide, stable for the session.
pub type FileId = i64;

/// Host-assigned, stable identifier of a viewer pane (not its position).
pub type PaneId = u32;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Information handed to a plugin when it is created.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginInit {
    /// The plugin id from its manifest.
    pub plugin_id: String,
    /// The plugin version from its manifest.
    pub plugin_version: String,
    /// Host application name (`"eov"`).
    pub host_name: String,
    /// Host application version.
    pub host_version: String,
    /// The `eov-plugin-api` version the host was built with.
    pub api_version: String,
    /// Root of the extracted plugin package (read-only).
    pub plugin_root: String,
    /// Durable, writable per-plugin data directory (created by the host).
    pub data_dir: String,
    /// Disposable, writable per-plugin cache directory (created by the host).
    pub cache_dir: String,
    /// Persisted per-plugin settings (see `settings.*` methods).
    #[serde(default)]
    pub settings: Map<String, Value>,
    /// Where eov before 1.0 kept per-user files (e.g. `~/.config/eov`), for
    /// migrating data a plugin stored there. Read it; don't write it.
    #[serde(default)]
    pub legacy_config_dir: Option<String>,
    /// The plugin runs without a window, for `eov batch` (see
    /// [`methods::BATCH_REGISTER`]): toolbar buttons, tools, layers and
    /// surfaces are accepted but not shown, so skip expensive UI set-up.
    #[serde(default)]
    pub headless: bool,
}

// ---------------------------------------------------------------------------
// Shared value types
// ---------------------------------------------------------------------------

/// An axis-aligned rectangle in level-0 pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// One pyramid level of a slide.
///
/// `downsample` is the level-0 size divided by this level's size as the
/// format reports it. It is **not** always a power of two (Aperio pyramids
/// commonly go 1, 4, 16).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelInfo {
    pub level: u32,
    pub width: u64,
    pub height: u64,
    pub downsample: f64,
}

/// Metadata of an open slide.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FileInfo {
    pub file_id: FileId,
    /// Path as text (lossy for non-UTF-8 paths; read pixels through the host).
    pub path: String,
    pub filename: String,
    /// Level-0 dimensions.
    pub width: u64,
    pub height: u64,
    #[serde(default)]
    pub levels: Vec<LevelInfo>,
    #[serde(default)]
    pub mpp_x: Option<f64>,
    #[serde(default)]
    pub mpp_y: Option<f64>,
    #[serde(default)]
    pub objective_power: Option<f64>,
    #[serde(default)]
    pub vendor: Option<String>,
    #[serde(default)]
    pub scan_date: Option<String>,
    /// Bounding box of the scanned area (`openslide.bounds-*`), if reported.
    #[serde(default)]
    pub bounds: Option<Rect>,
    /// A stable identity of the slide content (e.g. derived from
    /// `openslide.quickhash-1`), suitable as a database key.
    #[serde(default)]
    pub identity: Option<String>,
    /// Names of associated images (`label`, `macro`, `thumbnail`, …).
    #[serde(default)]
    pub associated_images: Vec<String>,
    #[serde(default)]
    pub has_icc_profile: bool,
    /// Stain label parsed from slide metadata (e.g. `"H&E"`), if any.
    #[serde(default)]
    pub stain: Option<String>,
}

/// The visible state of a pane showing a slide.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ViewInfo {
    pub pane: PaneId,
    pub file_id: FileId,
    /// Level-0 coordinates of the pane centre.
    pub center_x: f64,
    pub center_y: f64,
    /// Screen (logical) pixels per level-0 pixel.
    pub zoom: f64,
    /// Pane size in logical pixels.
    pub width: f64,
    pub height: f64,
    /// Physical pixels per logical pixel of the window.
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,
    /// Clockwise view rotation in degrees (eov turns views in steps of 90).
    #[serde(default)]
    pub rotation_deg: f64,
    /// The view is mirrored left to right, after the rotation.
    #[serde(default)]
    pub flipped: bool,
    /// Level-0 bounding box of the visible area.
    pub visible: Rect,
}

fn default_scale_factor() -> f64 {
    1.0
}

impl ViewInfo {
    /// An image-space direction as it points on the pane (rotation, then
    /// mirror).
    fn turn(&self, x: f64, y: f64) -> (f64, f64) {
        let (sin, cos) = self.rotation_deg.to_radians().sin_cos();
        let (tx, ty) = (x * cos - y * sin, x * sin + y * cos);
        if self.flipped { (-tx, ty) } else { (tx, ty) }
    }

    /// Logical pane position of level-0 point `(x, y)`, accounting for the
    /// view's rotation and mirror.
    pub fn image_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = self.turn(x - self.center_x, y - self.center_y);
        (
            self.width / 2.0 + dx * self.zoom,
            self.height / 2.0 + dy * self.zoom,
        )
    }

    /// The level-0 point at logical pane position `(x, y)`.
    pub fn screen_to_image(&self, x: f64, y: f64) -> (f64, f64) {
        let zoom = if self.zoom > 0.0 { self.zoom } else { 1.0 };
        let (mut dx, dy) = (
            (x - self.width / 2.0) / zoom,
            (y - self.height / 2.0) / zoom,
        );
        if self.flipped {
            dx = -dx;
        }
        let (sin, cos) = (-self.rotation_deg).to_radians().sin_cos();
        (
            self.center_x + dx * cos - dy * sin,
            self.center_y + dx * sin + dy * cos,
        )
    }
}

/// One pane in display order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PaneInfo {
    pub pane: PaneId,
    /// Position from the left, starting at 0.
    pub index: u32,
    #[serde(default)]
    pub file_id: Option<FileId>,
    #[serde(default)]
    pub view: Option<ViewInfo>,
}

/// Reference to a tool: the owning plugin (`None` for host tools) and its id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolRef {
    #[serde(default)]
    pub plugin_id: Option<String>,
    pub tool_id: String,
}

/// A consistent, recent view of the host state.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HostSnapshot {
    pub host_version: String,
    /// Whether an eov window has keyboard focus.
    #[serde(default)]
    pub window_focused: bool,
    #[serde(default)]
    pub render_backend: String,
    #[serde(default)]
    pub filtering_mode: String,
    #[serde(default)]
    pub focused_pane: Option<PaneId>,
    #[serde(default)]
    pub panes: Vec<PaneInfo>,
    #[serde(default)]
    pub files: Vec<FileInfo>,
    #[serde(default)]
    pub active_tool: Option<ToolRef>,
}

impl HostSnapshot {
    /// The view of the focused pane, if it shows a slide.
    pub fn focused_view(&self) -> Option<&ViewInfo> {
        let pane = self.focused_pane?;
        self.panes.iter().find(|p| p.pane == pane)?.view.as_ref()
    }

    /// Metadata of an open file.
    pub fn file(&self, file_id: FileId) -> Option<&FileInfo> {
        self.files.iter().find(|f| f.file_id == file_id)
    }
}

/// Keyboard modifiers held during an input event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Modifiers {
    #[serde(default)]
    pub shift: bool,
    /// The physical Control key.
    #[serde(default)]
    pub control: bool,
    #[serde(default)]
    pub alt: bool,
    /// Command (macOS) / Windows / Super key.
    #[serde(default)]
    pub meta: bool,
    /// The platform's command modifier: ⌘ on macOS, Control elsewhere.
    #[serde(default)]
    pub primary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerButton {
    #[default]
    None,
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerKind {
    Down,
    Move,
    Up,
    /// Second press of a double click (delivered after the matching `Down`).
    DoubleClick,
    Wheel,
    /// The pointer left the pane.
    Leave,
}

/// A pointer event for a plugin-owned custom tool.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointerEvent {
    pub kind: PointerKind,
    #[serde(default)]
    pub button: PointerButton,
    pub image_x: f64,
    pub image_y: f64,
    pub screen_x: f64,
    pub screen_y: f64,
    #[serde(default)]
    pub modifiers: Modifiers,
    /// Wheel delta in logical pixels (positive = away from the user).
    #[serde(default)]
    pub wheel_delta: f64,
}

/// A geometric shape in level-0 pixel coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    Point {
        x: f64,
        y: f64,
    },
    Polyline {
        points: Vec<[f64; 2]>,
    },
    Polygon {
        exterior: Vec<[f64; 2]>,
        #[serde(default)]
        holes: Vec<Vec<[f64; 2]>>,
    },
    Rectangle {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    },
    Ellipse {
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
    },
    /// Text anchored at its top-left corner; drawn at a constant screen size
    /// (`Style::font_size`, `Style::text_color`).
    Text {
        x: f64,
        y: f64,
        text: String,
    },
}

impl Shape {
    /// Whether every coordinate is finite and the shape has enough vertices
    /// (polylines ≥ 2, polygon rings ≥ 3) and non-negative extents.
    pub fn is_valid(&self) -> bool {
        let finite = |p: &[f64; 2]| p[0].is_finite() && p[1].is_finite();
        match self {
            Shape::Point { x, y } | Shape::Text { x, y, .. } => x.is_finite() && y.is_finite(),
            Shape::Polyline { points } => points.len() >= 2 && points.iter().all(finite),
            Shape::Polygon { exterior, holes } => {
                exterior.len() >= 3
                    && exterior.iter().all(finite)
                    && holes
                        .iter()
                        .all(|ring| ring.len() >= 3 && ring.iter().all(finite))
            }
            Shape::Rectangle {
                x,
                y,
                width,
                height,
            } => {
                [x, y, width, height].iter().all(|v| v.is_finite())
                    && *width >= 0.0
                    && *height >= 0.0
            }
            Shape::Ellipse { cx, cy, rx, ry } => {
                [cx, cy, rx, ry].iter().all(|v| v.is_finite()) && *rx >= 0.0 && *ry >= 0.0
            }
        }
    }

    /// Level-0 bounding box. Text has a zero-size box at its anchor.
    pub fn bounds(&self) -> Rect {
        fn of_points<'a>(points: impl Iterator<Item = &'a [f64; 2]>) -> Rect {
            let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for p in points {
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
            if x0 > x1 {
                return Rect::default();
            }
            Rect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            }
        }
        match self {
            Shape::Point { x, y } | Shape::Text { x, y, .. } => Rect {
                x: *x,
                y: *y,
                width: 0.0,
                height: 0.0,
            },
            Shape::Polyline { points } => of_points(points.iter()),
            Shape::Polygon { exterior, .. } => of_points(exterior.iter()),
            Shape::Rectangle {
                x,
                y,
                width,
                height,
            } => Rect {
                x: *x,
                y: *y,
                width: *width,
                height: *height,
            },
            Shape::Ellipse { cx, cy, rx, ry } => Rect {
                x: cx - rx,
                y: cy - ry,
                width: rx * 2.0,
                height: ry * 2.0,
            },
        }
    }
}

/// Identifies an overlay object: a plugin-scoped layer id plus an object id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectRef {
    pub layer_id: String,
    pub object_id: String,
}

/// A vector object in an overlay layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayObject {
    /// Unique within the layer; upserting an existing id replaces it.
    pub object_id: String,
    pub shape: Shape,
    /// Overrides the layer style for this object.
    #[serde(default)]
    pub style: Option<Style>,
    /// Text drawn next to the object (constant screen size).
    #[serde(default)]
    pub label: Option<String>,
    /// Shown when the pointer rests on the object.
    #[serde(default)]
    pub tooltip: Option<String>,
    /// Overrides the layer's `editable` flag for this object.
    #[serde(default)]
    pub editable: Option<bool>,
}

impl OverlayObject {
    pub fn new(object_id: impl Into<String>, shape: Shape) -> Self {
        Self {
            object_id: object_id.into(),
            shape,
            style: None,
            label: None,
            tooltip: None,
            editable: None,
        }
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = Some(style);
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = Some(tooltip.into());
        self
    }

    pub fn with_editable(mut self, editable: bool) -> Self {
        self.editable = Some(editable);
        self
    }
}

/// A measurement made with the host's measurement tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementInfo {
    pub measurement_id: String,
    pub file_id: FileId,
    pub shape: Shape,
    /// Length (lines/polylines) or perimeter (closed shapes) in µm, when the
    /// slide is calibrated.
    #[serde(default)]
    pub length_um: Option<f64>,
    /// Area in µm² for closed shapes, when calibrated.
    #[serde(default)]
    pub area_um2: Option<f64>,
    /// Angle in degrees for angle measurements.
    #[serde(default)]
    pub angle_deg: Option<f64>,
}

/// An entry of a menu (context menu, HUD drop-down).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MenuItem {
    pub item_id: String,
    pub label: String,
    /// Built-in icon name (`"copy"`, `"export"`, `"delete"`, `"point"`,
    /// `"measure"`, `"plugin"`, `"open"`) or empty; other names show none.
    #[serde(default)]
    pub icon: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Draw a check mark (for toggles / radio groups).
    #[serde(default)]
    pub checked: Option<bool>,
    /// Display-only shortcut text (e.g. `"Del"`).
    #[serde(default)]
    pub shortcut: String,
    #[serde(default)]
    pub separator_after: bool,
}

fn default_true() -> bool {
    true
}

/// The name of a key as [`Hotkey::key`] and [`ToolSpec::keys`] write it,
/// from the raw text of [`HostEvent::Key`]: a character in lower case
/// (`"a"`, `"+"`) or a named key (`"Escape"`, `"Enter"`, `"Tab"`,
/// `"Backspace"`, `"Delete"`, `"Space"`, `"ArrowLeft"`, `"F2"`, `"Home"`,
/// `"PageUp"`, …).
pub fn key_name(text: &str) -> String {
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

/// A hotkey binding. `key` is a single character (case-insensitive) or a
/// named key such as `"F2"`, `"Delete"`, `"Escape"` (see [`key_name`]).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Hotkey {
    pub key: String,
    #[serde(default)]
    pub modifiers: Modifiers,
    /// Hold-to-use: the tool is active only while the key is held.
    #[serde(default)]
    pub momentary: bool,
}

/// A plugin key binding (see [`methods::HOTKEY_REGISTER`]).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HotkeySpec {
    pub hotkey_id: String,
    pub hotkey: Hotkey,
    /// What the key does, for listings of key bindings.
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

// ---------------------------------------------------------------------------
// Overlay styles and layers
// ---------------------------------------------------------------------------

/// An RGBA colour, each channel 0–255.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    pub const TRANSPARENT: Self = Self(0, 0, 0, 0);

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(r, g, b, 255)
    }

    pub const fn with_alpha(self, alpha: u8) -> Self {
        Self(self.0, self.1, self.2, alpha)
    }

    /// Packs into `0xRRGGBBAA`.
    pub const fn to_u32(self) -> u32 {
        ((self.0 as u32) << 24) | ((self.1 as u32) << 16) | ((self.2 as u32) << 8) | self.3 as u32
    }

    /// Unpacks `0xRRGGBBAA`.
    pub const fn from_u32(value: u32) -> Self {
        Self(
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        )
    }

    /// Parses `#RRGGBB` or `#RRGGBBAA`.
    pub fn from_hex(text: &str) -> Option<Self> {
        let hex = text.strip_prefix('#')?;
        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
        match hex.len() {
            6 => Some(Self(byte(0)?, byte(2)?, byte(4)?, 255)),
            8 => Some(Self(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
            _ => None,
        }
    }

    pub fn to_hex(self) -> String {
        if self.3 == 255 {
            format!("#{:02X}{:02X}{:02X}", self.0, self.1, self.2)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.0, self.1, self.2, self.3)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dash {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

/// The unit an object's stroke width or point size is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizeUnit {
    /// Logical screen pixels (constant on screen at every zoom).
    #[default]
    Screen,
    /// Micrometres on the slide (scales with zoom; needs MPP).
    Micrometre,
    /// Level-0 pixels (scales with zoom).
    Image,
}

/// Visual style of overlay objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Style {
    #[serde(default)]
    pub fill: Rgba,
    #[serde(default = "default_stroke")]
    pub stroke: Rgba,
    #[serde(default = "default_stroke_width")]
    pub stroke_width: f32,
    #[serde(default)]
    pub dash: Dash,
    /// Radius of point markers.
    #[serde(default = "default_point_radius")]
    pub point_radius: f32,
    /// Unit of `stroke_width` and `point_radius`.
    #[serde(default)]
    pub size_unit: SizeUnit,
    /// Text colour for labels and text objects.
    #[serde(default = "default_text_color")]
    pub text_color: Rgba,
    #[serde(default = "default_font_size")]
    pub font_size: f32,
}

fn default_stroke() -> Rgba {
    Rgba::rgb(0x2E, 0xCC, 0x71)
}
fn default_stroke_width() -> f32 {
    1.5
}
fn default_point_radius() -> f32 {
    5.0
}
fn default_text_color() -> Rgba {
    Rgba::rgb(255, 255, 255)
}
fn default_font_size() -> f32 {
    12.0
}

impl Default for Style {
    fn default() -> Self {
        Self {
            fill: Rgba::TRANSPARENT,
            stroke: default_stroke(),
            stroke_width: default_stroke_width(),
            dash: Dash::Solid,
            point_radius: default_point_radius(),
            size_unit: SizeUnit::Screen,
            text_color: default_text_color(),
            font_size: default_font_size(),
        }
    }
}

/// Declares an overlay layer. Layers belong to one open file and are drawn
/// only in panes showing it; they are discarded when the file closes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Plugin-scoped layer id. Ids are unique per plugin across all open
    /// slides, so include the file id when a layer exists for each slide
    /// (e.g. `format!("cells-{file_id}")`).
    pub layer_id: String,
    pub file_id: FileId,
    #[serde(default)]
    pub name: String,
    /// Draw order; higher is on top. Host layers use the range -100..=100.
    #[serde(default)]
    pub z: i32,
    #[serde(default = "default_true")]
    pub visible: bool,
    /// 0.0–1.0, applied to the whole layer.
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    /// Objects can be hovered and clicked (`ObjectClicked`, `ObjectHovered`).
    #[serde(default)]
    pub interactive: bool,
    /// Objects flagged editable can be reshaped with the host's editing
    /// gestures (`ObjectEdited`).
    #[serde(default)]
    pub editable: bool,
    #[serde(default)]
    pub style: Style,
    /// Override used for selected objects (default: a highlight of `style`).
    #[serde(default)]
    pub selected_style: Option<Style>,
    /// Override used for the hovered object.
    #[serde(default)]
    pub hover_style: Option<Style>,
    /// Hide the layer below this zoom (screen px per level-0 px).
    #[serde(default)]
    pub min_zoom: Option<f64>,
}

fn default_opacity() -> f32 {
    1.0
}

impl LayerSpec {
    /// A visible, non-interactive layer with the default style.
    pub fn new(layer_id: impl Into<String>, file_id: FileId) -> Self {
        Self {
            layer_id: layer_id.into(),
            file_id,
            name: String::new(),
            z: 0,
            visible: true,
            opacity: 1.0,
            interactive: false,
            editable: false,
            style: Style::default(),
            selected_style: None,
            hover_style: None,
            min_zoom: None,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn interactive(mut self, interactive: bool) -> Self {
        self.interactive = interactive;
        self
    }

    pub fn editable(mut self, editable: bool) -> Self {
        self.editable = editable;
        self
    }

    pub fn at_z(mut self, z: i32) -> Self {
        self.z = z;
        self
    }
}

/// Partial update of a layer; `None` leaves a field unchanged.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LayerUpdate {
    pub layer_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub z: Option<i32>,
    #[serde(default)]
    pub visible: Option<bool>,
    #[serde(default)]
    pub opacity: Option<f32>,
    #[serde(default)]
    pub interactive: Option<bool>,
    #[serde(default)]
    pub editable: Option<bool>,
    #[serde(default)]
    pub style: Option<Style>,
    #[serde(default)]
    pub selected_style: Option<Style>,
    #[serde(default)]
    pub hover_style: Option<Style>,
}

// ---------------------------------------------------------------------------
// UI contributions
// ---------------------------------------------------------------------------

/// A button in the main window toolbar.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolbarButtonSpec {
    pub button_id: String,
    pub tooltip: String,
    /// Inline SVG.
    pub icon_svg: String,
    #[serde(default)]
    pub toggled_icon_svg: Option<String>,
    /// Render as active/pressed.
    #[serde(default)]
    pub active: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional hotkey that clicks the button.
    #[serde(default)]
    pub hotkey: Option<Hotkey>,
    /// Ordering among this plugin's buttons.
    #[serde(default)]
    pub order: i32,
}

/// Partial update of a toolbar or HUD button.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ButtonUpdate {
    pub button_id: String,
    /// Restrict the update to one pane (HUD buttons only).
    #[serde(default)]
    pub pane: Option<PaneId>,
    #[serde(default)]
    pub active: Option<bool>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub tooltip: Option<String>,
    #[serde(default)]
    pub icon_svg: Option<String>,
    #[serde(default)]
    pub menu: Option<Vec<MenuItem>>,
}

/// A button in every viewport's HUD toolbar. When `menu` is non-empty the
/// button opens a drop-down and clicks report the chosen `item_id`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HudButtonSpec {
    pub button_id: String,
    pub tooltip: String,
    pub icon_svg: String,
    #[serde(default)]
    pub toggled_icon_svg: Option<String>,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub menu: Vec<MenuItem>,
    #[serde(default)]
    pub order: i32,
}

/// What the host does while a tool is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Click places a point → `ShapeDrawn(Point)`.
    Point,
    /// Click adds vertices, double-click/Enter finishes → `Polyline`.
    Polyline,
    /// Click adds vertices; clicking the first vertex, double-click or Enter
    /// closes → `Polygon`.
    Polygon,
    /// Drag a rectangle → `Rectangle`.
    Rectangle,
    /// Drag an ellipse → `Ellipse`.
    Ellipse,
    /// Drag a free-hand outline (simplified) → `Polygon`.
    Freehand,
    /// No host gestures: the plugin receives raw `Pointer`/`Key` events.
    Custom,
    /// No drawing; clicks select objects and enable host editing gestures
    /// on the plugin's editable layers.
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cursor {
    #[default]
    Default,
    Crosshair,
    Pointer,
    Move,
    Text,
    NotAllowed,
}

/// A tool contributed by a plugin. It appears in the toolbar's tool group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub tool_id: String,
    pub label: String,
    pub icon_svg: String,
    pub kind: ToolKind,
    #[serde(default)]
    pub hotkey: Option<Hotkey>,
    #[serde(default)]
    pub cursor: Cursor,
    /// Layers (of this plugin) whose editable objects may be reshaped while
    /// the tool is active. Empty = all of the plugin's editable layers.
    #[serde(default)]
    pub edit_layers: Vec<String>,
    /// Style of the in-progress shape preview.
    #[serde(default)]
    pub preview_style: Option<Style>,
    /// Douglas–Peucker tolerance (screen px) for `Freehand`.
    #[serde(default = "default_freehand_tolerance")]
    pub freehand_tolerance_px: f32,
    #[serde(default)]
    pub order: i32,
    /// For a `Custom` tool: the keys it handles, by [`key_name`]
    /// (case-insensitive). Other keys keep working as they do without the
    /// tool (eov's shortcuts, other tools' keys, plugin hotkeys; Escape
    /// leaves the tool). Unset: the tool gets every key but Tab and
    /// Ctrl/⌘ shortcuts.
    #[serde(default)]
    pub keys: Option<Vec<String>>,
}

fn default_freehand_tolerance() -> f32 {
    1.5
}

impl ToolSpec {
    pub fn new(
        tool_id: impl Into<String>,
        label: impl Into<String>,
        icon_svg: impl Into<String>,
        kind: ToolKind,
    ) -> Self {
        Self {
            tool_id: tool_id.into(),
            label: label.into(),
            icon_svg: icon_svg.into(),
            kind,
            hotkey: None,
            cursor: Cursor::Crosshair,
            edit_layers: Vec::new(),
            preview_style: None,
            freehand_tolerance_px: default_freehand_tolerance(),
            order: 0,
            keys: None,
        }
    }

    pub fn with_hotkey(mut self, hotkey: Hotkey) -> Self {
        self.hotkey = Some(hotkey);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// Docked panel at the side of the main window (one visible at a time).
    Sidebar,
    /// Separate top-level window.
    Window,
    /// Modal dialog over the main window.
    Modal,
}

/// A Slint UI surface. `ui_path` is relative to the plugin root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceSpec {
    pub kind: SurfaceKind,
    pub ui_path: String,
    pub component: String,
    /// A window's title, like its minimum size, is its component's: inherit
    /// `Window` and set `title`, `min-width` and `min-height` there. eov 1.x
    /// only passes this to a `title` property the component declares, if any.
    #[serde(default)]
    pub title: Option<String>,
    /// Preferred size in logical pixels (sidebar: width only). Windows are
    /// never opened larger than eov's main window.
    #[serde(default)]
    pub width: Option<f32>,
    #[serde(default)]
    pub height: Option<f32>,
    /// Unused by eov 1.x (see `title`).
    #[serde(default)]
    pub min_width: Option<f32>,
    /// Unused by eov 1.x (see `title`).
    #[serde(default)]
    pub min_height: Option<f32>,
    /// Initial property values (by Slint property name).
    #[serde(default)]
    pub properties: Map<String, Value>,
    /// Keep keyboard shortcuts away from the host while this surface has
    /// keyboard focus (text entry).
    #[serde(default = "default_true")]
    pub capture_keyboard: bool,
}

impl SurfaceSpec {
    /// A surface showing `component` from `ui_path` (relative to the plugin
    /// root).
    pub fn new(
        kind: SurfaceKind,
        ui_path: impl Into<String>,
        component: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            ui_path: ui_path.into(),
            component: component.into(),
            title: None,
            width: None,
            height: None,
            min_width: None,
            min_height: None,
            properties: Map::new(),
            capture_keyboard: true,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_size(mut self, width: f32, height: Option<f32>) -> Self {
        self.width = Some(width);
        self.height = height;
        self
    }

    pub fn with_property(mut self, name: impl Into<String>, value: Value) -> Self {
        self.properties.insert(name.into(), value);
        self
    }
}

impl Hotkey {
    /// A plain key without modifiers.
    pub fn key(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            modifiers: Modifiers::default(),
            momentary: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FileFilter {
    pub name: String,
    /// Extensions without the leading dot.
    pub extensions: Vec<String>,
}

/// A viewport filter (CPU post-processing of rendered frames).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSpec {
    /// Plugin-scoped filter id.
    pub filter_id: String,
    pub name: String,
    /// Lower runs first. Ties are broken by plugin id, then filter id.
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotifyLevel {
    #[default]
    Info,
    Success,
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// Host → plugin events
// ---------------------------------------------------------------------------

/// Messages the host sends to a plugin's `handle_event`.
///
/// Unless documented otherwise the response is ignored (return `null`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HostEvent {
    /// The host finished starting up; every plugin has been created and the
    /// main window exists. Restore surfaces and layers here.
    Ready {
        /// This plugin's sidebar was open when eov last exited; re-open it
        /// to restore the user's layout.
        #[serde(default)]
        restore_sidebar: bool,
    },
    /// The application is exiting. Flush state; no further events follow.
    Shutdown,
    FileOpened {
        file: FileInfo,
    },
    /// The file is closed; its layers were discarded.
    FileClosed {
        file_id: FileId,
    },
    /// The focused pane changed, or the file shown in it.
    ActiveChanged {
        #[serde(default)]
        pane: Option<PaneId>,
        #[serde(default)]
        file_id: Option<FileId>,
    },
    /// A pane's view settled after navigation (debounced).
    ViewChanged {
        view: ViewInfo,
    },
    WindowFocusChanged {
        focused: bool,
    },
    ToolbarButtonClicked {
        button_id: String,
    },
    HudButtonClicked {
        button_id: String,
        #[serde(default)]
        item_id: Option<String>,
        pane: PaneId,
        #[serde(default)]
        view: Option<ViewInfo>,
    },
    /// The context menu is opening. Respond with [`ContextMenuResponse`].
    ContextMenuRequested(MenuContext),
    /// One of the plugin's context-menu items was chosen.
    ContextMenuItemSelected {
        item_id: String,
        context: MenuContext,
    },
    ToolActivated {
        tool_id: String,
    },
    ToolDeactivated {
        tool_id: String,
    },
    /// A host-drawn shape for one of the plugin's tools was completed.
    ShapeDrawn {
        tool_id: String,
        pane: PaneId,
        file_id: FileId,
        shape: Shape,
        #[serde(default)]
        modifiers: Modifiers,
    },
    /// Raw pointer input for an active `Custom` tool.
    Pointer {
        tool_id: String,
        pane: PaneId,
        file_id: FileId,
        event: PointerEvent,
    },
    /// Key input for an active `Custom` tool, or a plugin hotkey. `key` is
    /// the raw key text; [`key_name`] names it.
    Key {
        #[serde(default)]
        tool_id: Option<String>,
        #[serde(default)]
        pane: Option<PaneId>,
        key: String,
        #[serde(default)]
        modifiers: Modifiers,
        pressed: bool,
        #[serde(default)]
        repeat: bool,
    },
    ObjectClicked {
        object: ObjectRef,
        pane: PaneId,
        #[serde(default)]
        button: PointerButton,
        #[serde(default)]
        modifiers: Modifiers,
        /// 1 for a click, 2 for a double click.
        #[serde(default = "default_click_count")]
        click_count: u32,
    },
    ObjectHovered {
        #[serde(default)]
        object: Option<ObjectRef>,
        pane: PaneId,
    },
    /// A host editing gesture reshaped an editable object. The host already
    /// shows the new geometry; persist it (or upsert the old one to revert).
    ObjectEdited {
        object: ObjectRef,
        shape: Shape,
    },
    /// The user asked to delete objects (Delete key or context menu).
    ObjectsDeleteRequested {
        objects: Vec<ObjectRef>,
    },
    /// The host-side selection of the plugin's objects changed (click,
    /// Escape, layer removal). `objects` is the complete new selection.
    SelectionChanged {
        objects: Vec<ObjectRef>,
    },
    /// Delivered on the UI thread in response to the `post` method.
    Posted {
        token: String,
    },
    /// A click in Navigate/Select mode hit none of the plugin's objects.
    EmptyAreaClicked {
        pane: PaneId,
        #[serde(default)]
        file_id: Option<FileId>,
        #[serde(default)]
        modifiers: Modifiers,
    },
    /// Undo the plugin-pushed history entry.
    Undo {
        entry_id: String,
    },
    Redo {
        entry_id: String,
    },
    /// A callback declared in a surface's `.slint` file was invoked.
    UiCallback {
        surface_id: String,
        name: String,
        #[serde(default)]
        args: Vec<Value>,
    },
    /// The user closed a surface (window close button, Escape on a modal).
    SurfaceClosed {
        surface_id: String,
    },
    /// Result of `dialog.open_file` / `dialog.save_file`; empty = cancelled.
    DialogResult {
        request_id: String,
        #[serde(default)]
        paths: Vec<String>,
    },
    ConfirmResult {
        request_id: String,
        confirmed: bool,
    },
    JobCancelRequested {
        job_id: String,
    },
    /// The user toggled one of the plugin's filters in the host UI.
    FilterToggled {
        filter_id: String,
        enabled: bool,
    },
    /// The host dropped some of the plugin's undo entries (the history is
    /// bounded, and a new change discards the redo history of its slide):
    /// they will never be undone or redone, so their data can be freed.
    UndoDropped {
        entry_ids: Vec<String>,
    },
    /// A hotkey registered with [`methods::HOTKEY_REGISTER`] was pressed.
    HotkeyPressed {
        hotkey_id: String,
        /// The focused pane, if any.
        #[serde(default)]
        pane: Option<PaneId>,
    },
    /// Part of a registration was refused, e.g. a hotkey that eov or
    /// another plugin already uses (`what`: `"hotkey"`; `id`: the button,
    /// tool or hotkey id).
    RegistrationRefused {
        what: String,
        id: String,
        reason: String,
    },
    /// How a pane displays its slide changed (see
    /// [`methods::VIEW_DISPLAY`]).
    DisplayChanged {
        pane: PaneId,
        display: DisplayAdjustments,
    },
    /// A slide's region of interest was drawn, changed or removed (`None`).
    RoiChanged {
        file_id: FileId,
        #[serde(default)]
        roi: Option<Rect>,
    },
    /// A headless batch run (`eov batch run`, see
    /// [`methods::BATCH_REGISTER`]) asks for `file` to be analysed with
    /// `command_id`, writing results under `output_dir`. Slides come one at
    /// a time; work may continue on the plugin's own threads, and the run
    /// waits until [`methods::BATCH_FINISH`] names `run_id`.
    BatchRun {
        run_id: String,
        command_id: String,
        file: FileInfo,
        /// An existing, writable folder for this slide's results.
        output_dir: String,
        /// `--param name=value` values (JSON when they parse, else
        /// strings), over the command's defaults.
        #[serde(default)]
        params: Map<String, Value>,
    },
    /// Sent by newer hosts for events this plugin version does not know.
    #[serde(other)]
    Unknown,
}

fn default_click_count() -> u32 {
    1
}

/// Where a context menu was opened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MenuContext {
    pub pane: PaneId,
    pub view: ViewInfo,
    /// Level-0 coordinates under the pointer.
    pub image_x: f64,
    pub image_y: f64,
    /// The plugin's topmost object under the pointer, if any.
    #[serde(default)]
    pub hit: Option<ObjectRef>,
    /// The plugin's currently selected objects.
    #[serde(default)]
    pub selection: Vec<ObjectRef>,
}

/// Response to [`HostEvent::ContextMenuRequested`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ContextMenuResponse {
    #[serde(default)]
    pub items: Vec<MenuItem>,
}

// ---------------------------------------------------------------------------
// Plugin → host methods
// ---------------------------------------------------------------------------

/// Names of the methods a plugin can invoke through `HostApi::call`.
///
/// Arguments and results are JSON; the argument type of each method is noted.
pub mod methods {
    /// `LogArgs` → `null`
    pub const LOG: &str = "log";
    /// `NotifyArgs` → `null`. Shows a toast and records it in the log.
    pub const NOTIFY: &str = "notify";
    /// `StatusArgs` → `null`. Plugin-owned status bar segment.
    pub const STATUS: &str = "status";
    /// `{}` → `HostSnapshot`
    pub const SNAPSHOT: &str = "snapshot";
    /// `FileArgs` → `FileInfo`
    pub const FILE_INFO: &str = "file.info";
    /// `FileArgs` → `{ "<name>": "<value>" }` (all slide properties)
    pub const FILE_PROPERTIES: &str = "file.properties";
    /// `OpenFileArgs` → `null`
    pub const FILE_OPEN: &str = "file.open";

    /// `ToolbarButtonSpec` → `null`
    pub const TOOLBAR_ADD: &str = "toolbar.add";
    /// `ButtonUpdate` → `null`
    pub const TOOLBAR_UPDATE: &str = "toolbar.update";
    /// `IdArgs { id: button_id }` → `null`
    pub const TOOLBAR_REMOVE: &str = "toolbar.remove";
    /// `HudButtonSpec` → `null`
    pub const HUD_ADD: &str = "hud.add";
    /// `ButtonUpdate` → `null`
    pub const HUD_UPDATE: &str = "hud.update";
    /// `IdArgs { id: button_id }` → `null`
    pub const HUD_REMOVE: &str = "hud.remove";

    /// `HotkeySpec` → `null`. A key binding of its own, not tied to a
    /// toolbar button or tool; presses arrive as `HotkeyPressed`.
    pub const HOTKEY_REGISTER: &str = "hotkey.register";
    /// `IdArgs { id: hotkey_id }` → `null`
    pub const HOTKEY_UNREGISTER: &str = "hotkey.unregister";
    /// `ToolSpec` → `null`
    pub const TOOL_REGISTER: &str = "tool.register";
    /// `IdArgs { id: tool_id }` → `null`
    pub const TOOL_UNREGISTER: &str = "tool.unregister";
    /// `IdArgs { id: tool_id }` → `null`
    pub const TOOL_ACTIVATE: &str = "tool.activate";
    /// `{}` → `null`. Returns to Navigate if one of this plugin's tools is active.
    pub const TOOL_DEACTIVATE: &str = "tool.deactivate";

    /// `LayerSpec` → `null` (replaces an existing layer with the same id)
    pub const LAYER_CREATE: &str = "layer.create";
    /// `LayerUpdate` → `null`
    pub const LAYER_UPDATE: &str = "layer.update";
    /// `IdArgs { id: layer_id }` → `null`
    pub const LAYER_REMOVE: &str = "layer.remove";
    /// `SelectArgs` → `null`
    pub const LAYER_SELECT: &str = "layer.select";
    /// `UpsertObjectsArgs` → `null`. Parsed and indexed on the calling
    /// thread; call it from a worker thread for large batches.
    pub const LAYER_UPSERT: &str = "layer.upsert";
    /// `RemoveObjectsArgs` → `null`
    pub const LAYER_REMOVE_OBJECTS: &str = "layer.remove_objects";
    /// `IdArgs { id: layer_id }` → `null`. Removes every object.
    pub const LAYER_CLEAR: &str = "layer.clear";
    /// `BadgeArgs` → `null`. Shows (or with `text: None` removes) a short
    /// notice fixed to the corner of every pane showing a slide, which does
    /// not move with the view (e.g. "Predicted — research use only").
    pub const BADGE_SET: &str = "badge.set";
    /// `SelectArgs` → `null`. Draws these objects with the layer's hover
    /// style (an empty list clears it), e.g. while a list row is hovered.
    pub const LAYER_HIGHLIGHT: &str = "layer.highlight";
    /// `MinZoomArgs` → `null`. Hides the layer below a zoom level.
    pub const LAYER_SET_MIN_ZOOM: &str = "layer.set_min_zoom";
    /// `PaneVisibilityArgs` → `null`. Shows or hides the layer in one pane
    /// (on top of the layer's own visibility).
    pub const LAYER_SET_PANE_VISIBILITY: &str = "layer.set_pane_visibility";
    /// `PaneFilterArgs` → `null`. Shows the layer only in these panes
    /// (`None`: in every pane), including panes opened later.
    pub const LAYER_SET_PANES: &str = "layer.set_panes";

    /// `SurfaceSpec` → `SurfaceOpened`
    pub const SURFACE_OPEN: &str = "surface.open";
    /// `SurfaceProperties` → `null`. Values the surface already shows are left alone.
    pub const SURFACE_SET: &str = "surface.set";
    /// `IdArgs { id: surface_id }` → `null`
    pub const SURFACE_CLOSE: &str = "surface.close";
    /// `IdArgs { id: surface_id }` → `bool`
    pub const SURFACE_IS_OPEN: &str = "surface.is_open";

    /// `ConfirmArgs` → `RequestId` (result arrives as `ConfirmResult`)
    pub const DIALOG_CONFIRM: &str = "dialog.confirm";
    /// `FileDialogArgs` → `RequestId` (result arrives as `DialogResult`)
    pub const DIALOG_OPEN_FILE: &str = "dialog.open_file";
    /// `FileDialogArgs` → `RequestId` (result arrives as `DialogResult`)
    pub const DIALOG_SAVE_FILE: &str = "dialog.save_file";

    /// `ViewSetArgs` → `null`
    pub const VIEW_SET: &str = "view.set";
    /// `ViewPanArgs` → `null`
    pub const VIEW_PAN: &str = "view.pan";
    /// `ViewZoomArgs` → `null`
    pub const VIEW_ZOOM: &str = "view.zoom";
    /// `PaneArgs` → `null`
    pub const VIEW_FIT: &str = "view.fit";
    /// `ViewFrameArgs` → `null`
    pub const VIEW_FRAME: &str = "view.frame";
    /// `ViewOrientationArgs` → `null`: turns a pane's view (the rotation
    /// must be a multiple of 90°).
    pub const VIEW_ORIENTATION: &str = "view.orientation";
    /// `PaneArgs` → `DisplayAdjustments`: how the pane renders the slide,
    /// so overlays derived from slide pixels can match it.
    pub const VIEW_DISPLAY: &str = "view.display";

    /// `BatchCommandSpec` → `null`: offers a command for headless batch
    /// runs over many slides (`eov batch run <plugin>:<command> SLIDES`).
    /// Register it in `create`; slides then arrive as
    /// [`HostEvent::BatchRun`](super::HostEvent::BatchRun).
    pub const BATCH_REGISTER: &str = "batch.register";
    /// `BatchProgressArgs` → `null`: how far a batch run's slide is.
    pub const BATCH_PROGRESS: &str = "batch.progress";
    /// `BatchFinishArgs` → `null`: a batch run's slide is done.
    pub const BATCH_FINISH: &str = "batch.finish";

    /// `OffsetArgs` → `null`
    pub const APP_CYCLE_PANE: &str = "app.cycle_pane";
    /// `OffsetArgs` → `null`
    pub const APP_CYCLE_TAB: &str = "app.cycle_tab";
    /// `SeriesArgs` → `null`
    pub const APP_SERIES: &str = "app.series";
    /// `{}` → `null`
    pub const RENDER_REQUEST: &str = "render.request";

    /// `UndoPushArgs` → `null`
    pub const UNDO_PUSH: &str = "undo.push";
    /// `UndoClearArgs` → `null`
    pub const UNDO_CLEAR: &str = "undo.clear";

    /// `FilterSpec` → `null`
    pub const FILTER_REGISTER: &str = "filter.register";
    /// `FilterEnableArgs` → `null`
    pub const FILTER_SET_ENABLED: &str = "filter.set_enabled";
    /// `IdArgs { id: filter_id }` → `null`
    pub const FILTER_UNREGISTER: &str = "filter.unregister";

    /// `KeyArgs` → value or `null`
    pub const SETTINGS_GET: &str = "settings.get";
    /// `SettingsSetArgs` → `null` (persisted by the host)
    pub const SETTINGS_SET: &str = "settings.set";

    /// `JobStartArgs` → `JobStarted`
    pub const JOB_START: &str = "job.start";
    /// `JobUpdateArgs` → `null`
    pub const JOB_UPDATE: &str = "job.update";
    /// `JobFinishArgs` → `null`
    pub const JOB_FINISH: &str = "job.finish";

    /// `TextArgs` → `null`
    pub const CLIPBOARD_SET_TEXT: &str = "clipboard.set_text";

    /// `FileArgs` → `Option<Rect>` (the host region of interest of a slide)
    pub const ROI_GET: &str = "roi.get";
    /// `FileArgs` → `Vec<MeasurementInfo>`
    pub const MEASUREMENTS_LIST: &str = "measurements.list";

    /// `PostArgs` → `null`. Delivers `Posted { token }` on the UI thread.
    pub const POST: &str = "post";
    /// `SubscribeArgs` → `null`. Restricts the events delivered to the
    /// plugin; by default every event is delivered.
    pub const EVENTS_SUBSCRIBE: &str = "events.subscribe";
}

/// The `type` tags of [`HostEvent`] variants, for `events.subscribe`.
///
/// Lifecycle events (`ready`, `shutdown`) and responses to the plugin's own
/// requests (dialogs, confirmations, UI callbacks, undo/redo, posts) are always
/// delivered.
pub mod event_names {
    pub const FILE_OPENED: &str = "file_opened";
    pub const FILE_CLOSED: &str = "file_closed";
    pub const ACTIVE_CHANGED: &str = "active_changed";
    pub const VIEW_CHANGED: &str = "view_changed";
    pub const WINDOW_FOCUS_CHANGED: &str = "window_focus_changed";
    pub const OBJECT_HOVERED: &str = "object_hovered";
    pub const EMPTY_AREA_CLICKED: &str = "empty_area_clicked";
    pub const CONTEXT_MENU_REQUESTED: &str = "context_menu_requested";
    pub const ROI_CHANGED: &str = "roi_changed";
    pub const HOTKEY_PRESSED: &str = "hotkey_pressed";
    pub const DISPLAY_CHANGED: &str = "display_changed";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogArgs {
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NotifyArgs {
    #[serde(default)]
    pub level: NotifyLevel,
    pub message: String,
    /// Optional longer text (shown on demand, e.g. an error chain).
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StatusArgs {
    /// `None` clears the plugin's status segment.
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FileArgs {
    pub file_id: FileId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenFileArgs {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdArgs {
    pub id: String,
}

/// A command a plugin offers for headless batch runs (see
/// [`methods::BATCH_REGISTER`]).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BatchCommandSpec {
    /// `^[a-z0-9][a-z0-9_-]{0,63}$`
    pub command_id: String,
    /// One line, for `eov batch list`.
    pub description: String,
    /// The parameters it takes, for `--param name=value`.
    #[serde(default)]
    pub params: Vec<BatchParam>,
}

/// A parameter of a [`BatchCommandSpec`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BatchParam {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Used when the run does not set it (`null`: no default).
    #[serde(default)]
    pub default: Value,
}

/// Arguments of [`methods::BATCH_PROGRESS`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BatchProgressArgs {
    pub run_id: String,
    /// 0–1, when known.
    #[serde(default)]
    pub fraction: Option<f32>,
    #[serde(default)]
    pub message: Option<String>,
}

/// Arguments of [`methods::BATCH_FINISH`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BatchFinishArgs {
    pub run_id: String,
    /// `false` when the slide could not be analysed (`message` says why).
    pub ok: bool,
    #[serde(default)]
    pub message: Option<String>,
    /// The files written, relative to the slide's `output_dir`.
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Results and provenance for the run manifest: counts, scores, the
    /// model and its hash, the level or resolution read, the device…
    #[serde(default)]
    pub summary: Value,
}

/// Arguments of [`methods::BADGE_SET`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BadgeArgs {
    pub badge_id: String,
    /// The slide whose panes show it; `None`: every pane with a slide.
    #[serde(default)]
    pub file_id: Option<FileId>,
    /// At most 200 characters; `None` removes the badge.
    #[serde(default)]
    pub text: Option<String>,
}

/// Arguments of [`methods::LAYER_SET_MIN_ZOOM`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MinZoomArgs {
    pub layer_id: String,
    /// Screen pixels per level-0 pixel below which the layer is not drawn;
    /// `None` draws it at every zoom.
    #[serde(default)]
    pub min_zoom: Option<f64>,
}

/// Arguments of [`methods::LAYER_SET_PANE_VISIBILITY`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PaneVisibilityArgs {
    pub layer_id: String,
    pub pane: PaneId,
    pub visible: bool,
}

/// Arguments of [`methods::LAYER_SET_PANES`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PaneFilterArgs {
    pub layer_id: String,
    /// The only panes that show the layer; `None` shows it in every pane.
    #[serde(default)]
    pub panes: Option<Vec<PaneId>>,
}

/// How a pane displays its slide (the answer of [`methods::VIEW_DISPLAY`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplayAdjustments {
    /// 1.0 is unchanged.
    pub gamma: f32,
    /// Added to every channel (0.0 is unchanged; −1…1).
    pub brightness: f32,
    /// 1.0 is unchanged.
    pub contrast: f32,
    /// Sharpening amount (0.0 is off).
    pub sharpness: f32,
    /// Stain normalisation method (`"macenko"`, `"vahadane"`), if any.
    #[serde(default)]
    pub stain_normalization: Option<String>,
    /// Whether any colour processing is applied.
    pub modified: bool,
    /// What the viewer shows as its colour badge (empty when unmodified).
    #[serde(default)]
    pub badge: String,
}

impl Default for DisplayAdjustments {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            brightness: 0.0,
            contrast: 1.0,
            sharpness: 0.0,
            stain_normalization: None,
            modified: false,
            badge: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectArgs {
    pub layer_id: String,
    pub object_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpsertObjectsArgs {
    pub layer_id: String,
    pub objects: Vec<OverlayObject>,
    /// Remove every existing object first (atomic replace).
    #[serde(default)]
    pub replace_all: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoveObjectsArgs {
    pub layer_id: String,
    pub object_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostArgs {
    pub token: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubscribeArgs {
    /// Event `type` tags (see [`event_names`]).
    pub events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceOpened {
    pub surface_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceProperties {
    pub surface_id: String,
    pub properties: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmArgs {
    pub title: String,
    pub message: String,
    #[serde(default = "default_confirm_label")]
    pub confirm_label: String,
    #[serde(default = "default_cancel_label")]
    pub cancel_label: String,
    /// Style the confirm button as destructive.
    #[serde(default)]
    pub destructive: bool,
}

fn default_confirm_label() -> String {
    "OK".to_string()
}
fn default_cancel_label() -> String {
    "Cancel".to_string()
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FileDialogArgs {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub filters: Vec<FileFilter>,
    /// Suggested file name (save dialogs).
    #[serde(default)]
    pub default_name: Option<String>,
    /// Pick directories instead of files (open dialogs).
    #[serde(default)]
    pub directory: bool,
    /// Allow several selections (open dialogs).
    #[serde(default)]
    pub multiple: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestId {
    pub request_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PaneArgs {
    /// Defaults to the focused pane.
    #[serde(default)]
    pub pane: Option<PaneId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewSetArgs {
    #[serde(default)]
    pub pane: Option<PaneId>,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    #[serde(default)]
    pub animate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewPanArgs {
    #[serde(default)]
    pub pane: Option<PaneId>,
    /// Logical screen pixels.
    pub dx: f64,
    pub dy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewZoomArgs {
    #[serde(default)]
    pub pane: Option<PaneId>,
    /// > 1 zooms in.
    pub factor: f64,
    /// Screen anchor; defaults to the pane centre.
    #[serde(default)]
    pub anchor_x: Option<f64>,
    #[serde(default)]
    pub anchor_y: Option<f64>,
    #[serde(default)]
    pub animate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewOrientationArgs {
    #[serde(default)]
    pub pane: Option<PaneId>,
    /// Clockwise rotation in degrees, a multiple of 90.
    pub rotation_deg: f64,
    /// Mirror left to right after the rotation.
    #[serde(default)]
    pub flipped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewFrameArgs {
    #[serde(default)]
    pub pane: Option<PaneId>,
    pub rect: Rect,
    /// Fraction of the pane kept as margin (default 0.1).
    #[serde(default)]
    pub padding: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OffsetArgs {
    pub offset: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesAction {
    Toggle,
    Next,
    Previous,
    OpenSelected,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeriesArgs {
    pub action: SeriesAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UndoPushArgs {
    /// Plugin-chosen id, echoed back in `Undo`/`Redo`.
    pub entry_id: String,
    /// The slide the change applies to (scopes Ctrl+Z to that slide).
    #[serde(default)]
    pub file_id: Option<FileId>,
    /// Shown in menus ("Undo Add Point").
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct UndoClearArgs {
    /// Clear entries of one file; `None` clears all of the plugin's entries.
    #[serde(default)]
    pub file_id: Option<FileId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterEnableArgs {
    pub filter_id: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyArgs {
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingsSetArgs {
    pub key: String,
    /// `null` removes the key.
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobStartArgs {
    pub title: String,
    #[serde(default)]
    pub cancellable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobStarted {
    pub job_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobUpdateArgs {
    pub job_id: String,
    /// 0.0–1.0, or `None` for indeterminate.
    #[serde(default)]
    pub fraction: Option<f32>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobFinishArgs {
    pub job_id: String,
    /// `Some` marks the job failed and shows the error.
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextArgs {
    pub text: String,
}

/// Slide-wide JSON map type used by `file.properties`.
pub type PropertyMap = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_map_screen_and_image_through_their_turn() {
        let mut view = ViewInfo {
            center_x: 100.0,
            center_y: 50.0,
            zoom: 2.0,
            width: 300.0,
            height: 200.0,
            ..Default::default()
        };
        assert_eq!(view.image_to_screen(110.0, 50.0), (170.0, 100.0));
        // Turned clockwise, a point right of the centre shows below it.
        view.rotation_deg = 90.0;
        let (x, y) = view.image_to_screen(110.0, 50.0);
        assert!((x - 150.0).abs() < 1e-9 && (y - 120.0).abs() < 1e-9);
        for rotation in [0.0, 90.0, 180.0, 270.0] {
            for flipped in [false, true] {
                view.rotation_deg = rotation;
                view.flipped = flipped;
                let (sx, sy) = view.image_to_screen(37.0, 81.0);
                let (ix, iy) = view.screen_to_image(sx, sy);
                assert!((ix - 37.0).abs() < 1e-9 && (iy - 81.0).abs() < 1e-9);
            }
        }
        // Views from older hosts read as upright.
        let old: ViewInfo = serde_json::from_str(
            r#"{"pane":1,"file_id":1,"center_x":0,"center_y":0,"zoom":1,"width":1,"height":1,"visible":{"x":0,"y":0,"width":1,"height":1}}"#,
        )
        .unwrap();
        assert_eq!((old.rotation_deg, old.flipped), (0.0, false));
    }

    #[test]
    fn unknown_events_deserialize_to_unknown() {
        let event: HostEvent =
            serde_json::from_str(r#"{"type":"something_from_the_future","x":1}"#).unwrap();
        assert_eq!(event, HostEvent::Unknown);
    }

    #[test]
    fn events_round_trip() {
        let event = HostEvent::ShapeDrawn {
            tool_id: "poly".into(),
            pane: 3,
            file_id: 7,
            shape: Shape::Polygon {
                exterior: vec![[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]],
                holes: vec![],
            },
            modifiers: Modifiers::default(),
        };
        let text = serde_json::to_string(&event).unwrap();
        assert!(text.contains(r#""type":"shape_drawn""#), "{text}");
        assert!(text.contains(r#""kind":"polygon""#), "{text}");
        assert_eq!(serde_json::from_str::<HostEvent>(&text).unwrap(), event);
    }

    #[test]
    fn optional_fields_may_be_absent() {
        let event: HostEvent = serde_json::from_str(
            r#"{"type":"object_clicked","object":{"layer_id":"l","object_id":"o"},"pane":1}"#,
        )
        .unwrap();
        match event {
            HostEvent::ObjectClicked {
                click_count,
                button,
                ..
            } => {
                assert_eq!(click_count, 1);
                assert_eq!(button, PointerButton::None);
            }
            other => panic!("unexpected {other:?}"),
        }
        let spec: LayerSpec = serde_json::from_str(r#"{"layer_id":"a","file_id":1}"#).unwrap();
        assert!(spec.visible);
        assert_eq!(spec.opacity, 1.0);
        assert_eq!(spec.style, Style::default());
    }

    #[test]
    fn rgba_hex_round_trip() {
        assert_eq!(Rgba::from_hex("#FF8000"), Some(Rgba(255, 128, 0, 255)));
        assert_eq!(Rgba::from_hex("#ff800080"), Some(Rgba(255, 128, 0, 128)));
        assert_eq!(Rgba::from_hex("FF8000"), None);
        assert_eq!(Rgba::from_hex("#aé0000"), None);
        assert_eq!(Rgba(1, 2, 3, 255).to_hex(), "#010203");
        assert_eq!(Rgba::from_u32(Rgba(9, 8, 7, 6).to_u32()), Rgba(9, 8, 7, 6));
    }
}
