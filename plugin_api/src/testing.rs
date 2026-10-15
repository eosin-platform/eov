//! Unit-test support for plugins: run a [`Plugin`] against an in-memory
//! mock host.
//!
//! ```
//! use eov_plugin_api::prelude::*;
//! use eov_plugin_api::testing::TestHarness;
//!
//! struct Counter {
//!     host: Host,
//! }
//!
//! impl Plugin for Counter {
//!     fn create(host: Host, _init: PluginInit) -> Result<Self, String> {
//!         Ok(Self { host })
//!     }
//!
//!     fn on_event(&self, event: HostEvent) -> Result<(), String> {
//!         if let HostEvent::FileOpened { file } = event {
//!             self.host.info(format!("opened {}", file.filename));
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let harness = TestHarness::<Counter>::new().unwrap();
//! harness
//!     .send(HostEvent::FileOpened {
//!         file: FileInfo {
//!             file_id: 1,
//!             filename: "a.svs".into(),
//!             ..Default::default()
//!         },
//!     })
//!     .unwrap();
//! assert_eq!(harness.calls_to("log").len(), 1);
//! ```
//!
//! The mock answers every method the real host knows with a plausible
//! result, keeps layers, surfaces, settings and contributions in memory so
//! tests can inspect them, and records every call.

use crate::abi::{
    FfiError, FfiErrorKind, FfiResult, FilterFrame, FilterView, HostFfi, HostFfi_TO, Pixels,
    PluginBox, RasterUpload, RegionRequest,
};
use crate::protocol::{methods, *};
use crate::sdk::{Image, Plugin, create_plugin};
use abi_stable::{
    sabi_trait::TD_Opaque,
    std_types::{RErr, ROk, RSliceMut, RStr, RString},
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// One recorded host call.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedCall {
    pub method: String,
    pub args: Value,
}

/// A layer as the mock host stores it.
#[derive(Debug, Clone, PartialEq)]
pub struct MockLayer {
    pub spec: LayerSpec,
    pub objects: BTreeMap<String, OverlayObject>,
    pub rasters: BTreeMap<String, RasterUpload>,
    pub selection: Vec<String>,
    /// Panes the plugin hid the layer in (`layer.set_pane_visibility`).
    pub hidden_panes: std::collections::BTreeSet<PaneId>,
    /// Objects drawn highlighted (`layer.highlight`).
    pub highlighted: Vec<String>,
}

/// A surface as the mock host stores it.
#[derive(Debug, Clone, PartialEq)]
pub struct MockSurface {
    pub spec: SurfaceSpec,
    pub properties: Map<String, Value>,
    pub open: bool,
}

type RegionReader = dyn Fn(&RegionRequest) -> Result<Image, String> + Send + Sync;

/// Everything the mock host knows. Tests may read and modify it freely.
#[derive(Default)]
pub struct MockState {
    pub calls: Vec<RecordedCall>,
    pub snapshot: HostSnapshot,
    pub properties: BTreeMap<FileId, PropertyMap>,
    pub settings: Map<String, Value>,
    pub layers: BTreeMap<String, MockLayer>,
    pub surfaces: BTreeMap<String, MockSurface>,
    pub toolbar_buttons: BTreeMap<String, ToolbarButtonSpec>,
    pub hud_buttons: BTreeMap<String, HudButtonSpec>,
    pub tools: BTreeMap<String, ToolSpec>,
    pub hotkeys: BTreeMap<String, HotkeySpec>,
    /// Badges by id: (slide, text).
    pub badges: BTreeMap<String, (Option<FileId>, String)>,
    pub active_tool: Option<String>,
    pub filters: BTreeMap<String, FilterSpec>,
    pub undo: Vec<UndoPushArgs>,
    pub notifications: Vec<NotifyArgs>,
    pub status: Option<String>,
    pub clipboard: Option<String>,
    pub roi: BTreeMap<FileId, Rect>,
    pub measurements: Vec<MeasurementInfo>,
    /// What `view.display` answers.
    pub display: DisplayAdjustments,
    pub posted: Vec<String>,
    /// Serves `read_region`; without one a mid-grey opaque image is returned.
    pub region_reader: Option<Box<RegionReader>>,
    /// Methods that fail with [`FfiErrorKind::Unsupported`], to emulate an
    /// older host.
    pub unsupported: Vec<String>,
}

impl MockState {
    /// The arguments of every recorded call to `method`.
    pub fn calls_to(&self, method: &str) -> Vec<Value> {
        self.calls
            .iter()
            .filter(|call| call.method == method)
            .map(|call| call.args.clone())
            .collect()
    }

    /// Objects of a layer, in id order.
    pub fn objects(&self, layer_id: &str) -> Vec<OverlayObject> {
        self.layers
            .get(layer_id)
            .map(|layer| layer.objects.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// The mock host object handed to the plugin.
#[derive(Clone)]
pub struct MockHost {
    state: Arc<Mutex<MockState>>,
    next_id: Arc<AtomicU64>,
}

impl Default for MockHost {
    fn default() -> Self {
        Self::new()
    }
}

impl MockHost {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState::default())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// The shared state (poison-tolerant).
    pub fn state(&self) -> MutexGuard<'_, MockState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The FFI object to pass to a plugin.
    pub fn to_ffi(&self) -> crate::abi::HostBox {
        HostFfi_TO::from_value(self.clone(), TD_Opaque)
    }

    fn fresh_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn handle(&self, method: &str, args: Value) -> Result<Value, FfiError> {
        let mut state = self.state();
        state.calls.push(RecordedCall {
            method: method.to_string(),
            args: args.clone(),
        });
        if state.unsupported.iter().any(|m| m == method) {
            return Err(FfiError::new(
                FfiErrorKind::Unsupported,
                format!("unknown method '{method}'"),
            ));
        }
        fn parse<T: DeserializeOwned>(args: Value) -> Result<T, FfiError> {
            serde_json::from_value(args)
                .map_err(|e| FfiError::new(FfiErrorKind::InvalidArguments, e.to_string()))
        }
        let null = Value::Null;
        Ok(match method {
            methods::LOG | methods::RENDER_REQUEST | methods::EVENTS_SUBSCRIBE => null,
            methods::NOTIFY => {
                let args: NotifyArgs = parse(args)?;
                state.notifications.push(args);
                null
            }
            methods::STATUS => {
                let args: StatusArgs = parse(args)?;
                state.status = args.text;
                null
            }
            methods::SNAPSHOT => json!(state.snapshot),
            methods::FILE_INFO => {
                let args: FileArgs = parse(args)?;
                let file =
                    state.snapshot.file(args.file_id).cloned().ok_or_else(|| {
                        FfiError::failed(format!("no open file {}", args.file_id))
                    })?;
                json!(file)
            }
            methods::FILE_PROPERTIES => {
                let args: FileArgs = parse(args)?;
                json!(
                    state
                        .properties
                        .get(&args.file_id)
                        .cloned()
                        .unwrap_or_default()
                )
            }
            methods::FILE_OPEN => null,
            methods::TOOLBAR_ADD => {
                let spec: ToolbarButtonSpec = parse(args)?;
                state.toolbar_buttons.insert(spec.button_id.clone(), spec);
                null
            }
            methods::TOOLBAR_UPDATE => {
                let update: ButtonUpdate = parse(args)?;
                let button = state
                    .toolbar_buttons
                    .get_mut(&update.button_id)
                    .ok_or_else(|| FfiError::failed("unknown toolbar button"))?;
                if let Some(active) = update.active {
                    button.active = active;
                }
                if let Some(enabled) = update.enabled {
                    button.enabled = enabled;
                }
                if let Some(tooltip) = update.tooltip {
                    button.tooltip = tooltip;
                }
                if let Some(icon) = update.icon_svg {
                    button.icon_svg = icon;
                }
                null
            }
            methods::TOOLBAR_REMOVE => {
                let args: IdArgs = parse(args)?;
                state.toolbar_buttons.remove(&args.id);
                null
            }
            methods::HUD_ADD => {
                let spec: HudButtonSpec = parse(args)?;
                state.hud_buttons.insert(spec.button_id.clone(), spec);
                null
            }
            methods::HUD_UPDATE => {
                let update: ButtonUpdate = parse(args)?;
                let button = state
                    .hud_buttons
                    .get_mut(&update.button_id)
                    .ok_or_else(|| FfiError::failed("unknown HUD button"))?;
                if let Some(active) = update.active {
                    button.active = active;
                }
                if let Some(menu) = update.menu {
                    button.menu = menu;
                }
                null
            }
            methods::HUD_REMOVE => {
                let args: IdArgs = parse(args)?;
                state.hud_buttons.remove(&args.id);
                null
            }
            methods::BADGE_SET => {
                let args: BadgeArgs = parse(args)?;
                match args.text {
                    Some(text) => {
                        state.badges.insert(args.badge_id, (args.file_id, text));
                    }
                    None => {
                        state.badges.remove(&args.badge_id);
                    }
                }
                null
            }
            methods::HOTKEY_REGISTER => {
                let spec: HotkeySpec = parse(args)?;
                state.hotkeys.insert(spec.hotkey_id.clone(), spec);
                null
            }
            methods::HOTKEY_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                state.hotkeys.remove(&args.id);
                null
            }
            methods::TOOL_REGISTER => {
                let spec: ToolSpec = parse(args)?;
                state.tools.insert(spec.tool_id.clone(), spec);
                null
            }
            methods::TOOL_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                state.tools.remove(&args.id);
                if state.active_tool.as_deref() == Some(args.id.as_str()) {
                    state.active_tool = None;
                }
                null
            }
            methods::TOOL_ACTIVATE => {
                let args: IdArgs = parse(args)?;
                if !state.tools.contains_key(&args.id) {
                    return Err(FfiError::failed(format!("unknown tool '{}'", args.id)));
                }
                state.active_tool = Some(args.id);
                null
            }
            methods::TOOL_DEACTIVATE => {
                state.active_tool = None;
                null
            }
            methods::LAYER_CREATE => {
                let spec: LayerSpec = parse(args)?;
                state.layers.insert(
                    spec.layer_id.clone(),
                    MockLayer {
                        spec,
                        objects: BTreeMap::new(),
                        rasters: BTreeMap::new(),
                        selection: Vec::new(),
                        hidden_panes: Default::default(),
                        highlighted: Vec::new(),
                    },
                );
                null
            }
            methods::LAYER_UPDATE => {
                let update: LayerUpdate = parse(args)?;
                let layer = layer_mut(&mut state, &update.layer_id)?;
                if let Some(visible) = update.visible {
                    layer.spec.visible = visible;
                }
                if let Some(opacity) = update.opacity {
                    layer.spec.opacity = opacity;
                }
                if let Some(name) = update.name {
                    layer.spec.name = name;
                }
                if let Some(z) = update.z {
                    layer.spec.z = z;
                }
                if let Some(style) = update.style {
                    layer.spec.style = style;
                }
                null
            }
            methods::LAYER_REMOVE => {
                let args: IdArgs = parse(args)?;
                state.layers.remove(&args.id);
                null
            }
            methods::LAYER_HIGHLIGHT => {
                let args: SelectArgs = parse(args)?;
                layer_mut(&mut state, &args.layer_id)?.highlighted = args.object_ids;
                null
            }
            methods::LAYER_SET_MIN_ZOOM => {
                let args: MinZoomArgs = parse(args)?;
                layer_mut(&mut state, &args.layer_id)?.spec.min_zoom = args.min_zoom;
                null
            }
            methods::LAYER_SET_PANE_VISIBILITY => {
                let args: PaneVisibilityArgs = parse(args)?;
                let layer = layer_mut(&mut state, &args.layer_id)?;
                if args.visible {
                    layer.hidden_panes.remove(&args.pane);
                } else {
                    layer.hidden_panes.insert(args.pane);
                }
                null
            }
            methods::VIEW_DISPLAY => json!(state.display),
            methods::LAYER_SELECT => {
                let args: SelectArgs = parse(args)?;
                layer_mut(&mut state, &args.layer_id)?.selection = args.object_ids;
                null
            }
            methods::LAYER_UPSERT => {
                let args: UpsertObjectsArgs = parse(args)?;
                if let Some(bad) = args.objects.iter().find(|o| !o.shape.is_valid()) {
                    return Err(FfiError::new(
                        FfiErrorKind::InvalidArguments,
                        format!("object '{}' has an invalid shape", bad.object_id),
                    ));
                }
                let layer = layer_mut(&mut state, &args.layer_id)?;
                if args.replace_all {
                    layer.objects.clear();
                    layer.rasters.clear();
                }
                for object in args.objects {
                    layer.objects.insert(object.object_id.clone(), object);
                }
                null
            }
            methods::LAYER_REMOVE_OBJECTS => {
                let args: RemoveObjectsArgs = parse(args)?;
                let layer = layer_mut(&mut state, &args.layer_id)?;
                for id in &args.object_ids {
                    layer.objects.remove(id);
                    layer.rasters.remove(id);
                }
                layer.selection.retain(|id| !args.object_ids.contains(id));
                null
            }
            methods::LAYER_CLEAR => {
                let args: IdArgs = parse(args)?;
                let layer = layer_mut(&mut state, &args.id)?;
                layer.objects.clear();
                layer.rasters.clear();
                layer.selection.clear();
                null
            }
            methods::SURFACE_OPEN => {
                let spec: SurfaceSpec = parse(args)?;
                drop(state);
                let surface_id = self.fresh_id("surface");
                let mut state = self.state();
                if spec.kind == SurfaceKind::Sidebar {
                    // One sidebar at a time, as in the real host.
                    for surface in state.surfaces.values_mut() {
                        if surface.spec.kind == SurfaceKind::Sidebar {
                            surface.open = false;
                        }
                    }
                }
                state.surfaces.insert(
                    surface_id.clone(),
                    MockSurface {
                        properties: spec.properties.clone(),
                        spec,
                        open: true,
                    },
                );
                json!(SurfaceOpened { surface_id })
            }
            methods::SURFACE_SET => {
                let args: SurfaceProperties = parse(args)?;
                let surface = state
                    .surfaces
                    .get_mut(&args.surface_id)
                    .filter(|s| s.open)
                    .ok_or_else(|| FfiError::failed("surface is not open"))?;
                surface.properties.extend(args.properties);
                null
            }
            methods::SURFACE_CLOSE => {
                let args: IdArgs = parse(args)?;
                if let Some(surface) = state.surfaces.get_mut(&args.id) {
                    surface.open = false;
                }
                null
            }
            methods::SURFACE_IS_OPEN => {
                let args: IdArgs = parse(args)?;
                json!(state.surfaces.get(&args.id).is_some_and(|s| s.open))
            }
            methods::DIALOG_CONFIRM | methods::DIALOG_OPEN_FILE | methods::DIALOG_SAVE_FILE => {
                drop(state);
                json!(RequestId {
                    request_id: self.fresh_id("request"),
                })
            }
            methods::VIEW_SET
            | methods::VIEW_PAN
            | methods::VIEW_ZOOM
            | methods::VIEW_FIT
            | methods::VIEW_FRAME
            | methods::APP_CYCLE_PANE
            | methods::APP_CYCLE_TAB
            | methods::APP_SERIES => null,
            methods::UNDO_PUSH => {
                let args: UndoPushArgs = parse(args)?;
                state.undo.push(args);
                null
            }
            methods::UNDO_CLEAR => {
                let args: UndoClearArgs = parse(args)?;
                match args.file_id {
                    Some(file_id) => state.undo.retain(|e| e.file_id != Some(file_id)),
                    None => state.undo.clear(),
                }
                null
            }
            methods::FILTER_REGISTER => {
                let spec: FilterSpec = parse(args)?;
                state.filters.insert(spec.filter_id.clone(), spec);
                null
            }
            methods::FILTER_SET_ENABLED => {
                let args: FilterEnableArgs = parse(args)?;
                state
                    .filters
                    .get_mut(&args.filter_id)
                    .ok_or_else(|| FfiError::failed("unknown filter"))?
                    .enabled = args.enabled;
                null
            }
            methods::FILTER_UNREGISTER => {
                let args: IdArgs = parse(args)?;
                state.filters.remove(&args.id);
                null
            }
            methods::SETTINGS_GET => {
                let args: KeyArgs = parse(args)?;
                state
                    .settings
                    .get(&args.key)
                    .cloned()
                    .unwrap_or(Value::Null)
            }
            methods::SETTINGS_SET => {
                let args: SettingsSetArgs = parse(args)?;
                if args.value.is_null() {
                    state.settings.remove(&args.key);
                } else {
                    state.settings.insert(args.key, args.value);
                }
                null
            }
            methods::JOB_START => {
                drop(state);
                json!(JobStarted {
                    job_id: self.fresh_id("job"),
                })
            }
            methods::JOB_UPDATE | methods::JOB_FINISH => null,
            methods::CLIPBOARD_SET_TEXT => {
                let args: TextArgs = parse(args)?;
                state.clipboard = Some(args.text);
                null
            }
            methods::ROI_GET => {
                let args: FileArgs = parse(args)?;
                json!(state.roi.get(&args.file_id))
            }
            methods::MEASUREMENTS_LIST => {
                let args: FileArgs = parse(args)?;
                json!(
                    state
                        .measurements
                        .iter()
                        .filter(|m| m.file_id == args.file_id)
                        .collect::<Vec<_>>()
                )
            }
            methods::POST => {
                let args: PostArgs = parse(args)?;
                state.posted.push(args.token);
                null
            }
            other => {
                return Err(FfiError::new(
                    FfiErrorKind::Unsupported,
                    format!("unknown method '{other}'"),
                ));
            }
        })
    }
}

fn layer_mut<'a>(state: &'a mut MockState, layer_id: &str) -> Result<&'a mut MockLayer, FfiError> {
    state
        .layers
        .get_mut(layer_id)
        .ok_or_else(|| FfiError::failed(format!("unknown layer '{layer_id}'")))
}

impl HostFfi for MockHost {
    fn call(&self, method: RStr<'_>, args_json: RStr<'_>) -> FfiResult<RString> {
        let args: Value = match serde_json::from_str(args_json.as_str()) {
            Ok(args) => args,
            Err(e) => {
                return RErr(FfiError::new(FfiErrorKind::InvalidArguments, e.to_string()));
            }
        };
        match self.handle(method.as_str(), args) {
            Ok(value) => ROk(RString::from(value.to_string())),
            Err(error) => RErr(error),
        }
    }

    fn read_region(&self, request: RegionRequest) -> FfiResult<Pixels> {
        let pixels = u64::from(request.width) * u64::from(request.height);
        if pixels > crate::abi::MAX_REGION_PIXELS {
            return RErr(FfiError::new(
                FfiErrorKind::InvalidArguments,
                "region too large",
            ));
        }
        let state = self.state();
        let image = match &state.region_reader {
            Some(reader) => match reader(&request) {
                Ok(image) => image,
                Err(e) => return RErr(FfiError::failed(e)),
            },
            None => Image {
                width: request.width,
                height: request.height,
                data: [128, 128, 128, 255].repeat(pixels as usize),
            },
        };
        ROk(Pixels {
            width: image.width,
            height: image.height,
            data: image.data.into(),
        })
    }

    fn read_associated_image(&self, _file_id: i64, name: RStr<'_>) -> FfiResult<Pixels> {
        RErr(FfiError::failed(format!(
            "no associated image '{}'",
            name.as_str()
        )))
    }

    fn put_raster(&self, raster: RasterUpload) -> FfiResult<()> {
        if !raster.pixels.is_well_formed() {
            return RErr(FfiError::new(
                FfiErrorKind::InvalidArguments,
                "raster size does not match its data",
            ));
        }
        let mut state = self.state();
        let layer_id = raster.layer_id.to_string();
        match layer_mut(&mut state, &layer_id) {
            Ok(layer) => {
                layer.rasters.insert(raster.object_id.to_string(), raster);
                ROk(())
            }
            Err(e) => RErr(e),
        }
    }
}

/// Default init for tests: unique, writable temporary directories.
pub fn test_init(plugin_id: &str) -> PluginInit {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "eov-plugin-test-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let data_dir = root.join("data");
    let cache_dir = root.join("cache");
    let _ = std::fs::create_dir_all(&data_dir);
    let _ = std::fs::create_dir_all(&cache_dir);
    PluginInit {
        plugin_id: plugin_id.to_string(),
        plugin_version: "0.0.0".to_string(),
        host_name: "eov".to_string(),
        host_version: crate::abi::API_VERSION.to_string(),
        api_version: crate::abi::API_VERSION.to_string(),
        plugin_root: root.to_string_lossy().into_owned(),
        data_dir: data_dir.to_string_lossy().into_owned(),
        cache_dir: cache_dir.to_string_lossy().into_owned(),
        settings: Map::new(),
        legacy_config_dir: None,
    }
}

/// Drives a plugin through the same FFI adapter the host uses.
pub struct TestHarness<P: Plugin> {
    host: MockHost,
    instance: PluginBox,
    _plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> TestHarness<P> {
    pub fn new() -> Result<Self, String> {
        Self::with_init(test_init("test_plugin"))
    }

    pub fn with_init(init: PluginInit) -> Result<Self, String> {
        Self::with_host(MockHost::new(), init)
    }

    /// Uses a prepared mock (e.g. with a snapshot or region reader set).
    pub fn with_host(host: MockHost, init: PluginInit) -> Result<Self, String> {
        let init_json = serde_json::to_string(&init).map_err(|e| e.to_string())?;
        let instance =
            create_plugin::<P>(host.to_ffi(), &init_json).map_err(|e| e.message.into_string())?;
        Ok(Self {
            host,
            instance,
            _plugin: PhantomData,
        })
    }

    pub fn host(&self) -> &MockHost {
        &self.host
    }

    pub fn state(&self) -> MutexGuard<'_, MockState> {
        self.host.state()
    }

    /// The arguments of every call to `method` so far.
    pub fn calls_to(&self, method: &str) -> Vec<Value> {
        self.state().calls_to(method)
    }

    /// Delivers an event; returns the plugin's JSON response.
    pub fn send(&self, event: HostEvent) -> Result<Value, FfiError> {
        let json = serde_json::to_string(&event).map_err(|e| FfiError::failed(e.to_string()))?;
        let reply = self
            .instance
            .handle_event(RStr::from_str(&json))
            .into_result()?;
        serde_json::from_str(reply.as_str()).map_err(|e| FfiError::failed(e.to_string()))
    }

    /// Requests context-menu items.
    pub fn context_menu(&self, context: MenuContext) -> Result<Vec<MenuItem>, FfiError> {
        let reply = self.send(HostEvent::ContextMenuRequested(context))?;
        let response: ContextMenuResponse =
            serde_json::from_value(reply).map_err(|e| FfiError::failed(e.to_string()))?;
        Ok(response.items)
    }

    /// Invokes a surface callback.
    pub fn ui_callback(
        &self,
        surface_id: &str,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value, FfiError> {
        self.send(HostEvent::UiCallback {
            surface_id: surface_id.to_string(),
            name: name.to_string(),
            args,
        })
    }

    /// Runs a filter over a tightly packed RGBA buffer.
    pub fn apply_filter(
        &self,
        filter_id: &str,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        view: FilterView,
    ) -> Result<(), FfiError> {
        self.instance
            .apply_filter(
                RStr::from_str(filter_id),
                FilterFrame {
                    pixels: RSliceMut::from_mut_slice(pixels),
                    width,
                    height,
                    stride: width * 4,
                    view,
                },
            )
            .into_result()
    }
}

impl<P: Plugin> Drop for TestHarness<P> {
    fn drop(&mut self) {
        self.instance.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::{Frame, Host};

    struct Probe {
        host: Host,
    }

    impl Plugin for Probe {
        fn create(host: Host, init: PluginInit) -> Result<Self, String> {
            if init.plugin_id == "refuse" {
                return Err("refused".into());
            }
            host.add_toolbar_button(ToolbarButtonSpec {
                button_id: "go".into(),
                tooltip: "Go".into(),
                icon_svg: "<svg/>".into(),
                ..Default::default()
            })
            .map_err(String::from)?;
            Ok(Self { host })
        }

        fn on_event(&self, event: HostEvent) -> Result<(), String> {
            match event {
                HostEvent::ToolbarButtonClicked { button_id } if button_id == "panic" => {
                    panic!("boom")
                }
                HostEvent::ToolbarButtonClicked { .. } => {
                    let layer = self
                        .host
                        .create_layer(LayerSpec::new("cells", 1).named("Cells").interactive(true))
                        .map_err(String::from)?;
                    layer
                        .upsert(vec![OverlayObject::new(
                            "c1",
                            Shape::Point { x: 1.0, y: 2.0 },
                        )])
                        .map_err(String::from)?;
                    Ok(())
                }
                HostEvent::FileClosed { .. } => Err("cannot close".into()),
                _ => Ok(()),
            }
        }

        fn context_menu(&self, context: &MenuContext) -> Vec<MenuItem> {
            vec![MenuItem {
                item_id: format!("at-{}", context.image_x),
                label: "Here".into(),
                ..Default::default()
            }]
        }

        fn ui_callback(&self, _surface: &str, name: &str, args: &[Value]) -> Result<Value, String> {
            Ok(json!({ "name": name, "argc": args.len() }))
        }

        fn apply_filter(&self, _filter_id: &str, frame: &mut Frame<'_>) -> Result<(), String> {
            frame.for_each_pixel(|_, _, px| px[0] = 255 - px[0]);
            Ok(())
        }
    }

    fn view() -> ViewInfo {
        ViewInfo {
            pane: 1,
            file_id: 1,
            zoom: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn layers_take_min_zoom_and_pane_visibility() {
        use crate::sdk::{Host, Image};
        let mock = MockHost::new();
        let host = Host::from_ffi(mock.to_ffi());
        host.create_layer(LayerSpec::new("heat", 1)).unwrap();
        let layer = host.layer("heat");
        layer.set_min_zoom(Some(0.25)).unwrap();
        layer.set_visible_in_pane(3, false).unwrap();
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let blank = Image {
            width: 2,
            height: 2,
            data: vec![0; 16],
        };
        layer.put_raster("map", rect, blank, false).unwrap();
        let state = mock.state();
        let mock_layer = &state.layers["heat"];
        assert_eq!(mock_layer.spec.min_zoom, Some(0.25));
        assert!(mock_layer.hidden_panes.contains(&3));
        assert!(mock_layer.rasters.contains_key("map"));
        drop(state);
        assert_eq!(host.display_adjustments(None).unwrap().gamma, 1.0);
    }

    #[test]
    fn hotkeys_badges_and_highlights_reach_the_host() {
        use crate::sdk::Host;
        let mock = MockHost::new();
        let host = Host::from_ffi(mock.to_ffi());
        host.register_hotkey(HotkeySpec {
            hotkey_id: "next".into(),
            hotkey: Hotkey::key("j"),
            description: "Next object".into(),
        })
        .unwrap();
        host.set_badge("predicted", Some(1), Some("Predicted"))
            .unwrap();
        host.create_layer(LayerSpec::new("cells", 1)).unwrap();
        host.layer("cells").highlight(vec!["a".into()]).unwrap();
        {
            let state = mock.state();
            assert!(state.hotkeys.contains_key("next"));
            assert_eq!(
                state.badges["predicted"],
                (Some(1), "Predicted".to_string())
            );
            assert_eq!(state.layers["cells"].highlighted, ["a"]);
        }
        host.set_badge("predicted", None, None).unwrap();
        host.unregister_hotkey("next").unwrap();
        let state = mock.state();
        assert!(state.badges.is_empty() && state.hotkeys.is_empty());
    }

    #[test]
    fn create_registers_contributions() {
        let harness = TestHarness::<Probe>::new().unwrap();
        assert!(harness.state().toolbar_buttons.contains_key("go"));
    }

    #[test]
    fn create_errors_are_reported() {
        let error = TestHarness::<Probe>::with_init(test_init("refuse"))
            .err()
            .unwrap();
        assert_eq!(error, "refused");
    }

    #[test]
    fn events_reach_the_plugin_and_overlays_are_stored() {
        let harness = TestHarness::<Probe>::new().unwrap();
        harness
            .send(HostEvent::ToolbarButtonClicked {
                button_id: "go".into(),
            })
            .unwrap();
        let objects = harness.state().objects("cells");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].shape, Shape::Point { x: 1.0, y: 2.0 });
    }

    #[test]
    fn panics_become_errors() {
        let harness = TestHarness::<Probe>::new().unwrap();
        let error = harness
            .send(HostEvent::ToolbarButtonClicked {
                button_id: "panic".into(),
            })
            .unwrap_err();
        assert_eq!(error.kind(), FfiErrorKind::Panicked);
        assert!(error.message.contains("boom"));
        // The instance is still usable afterwards.
        harness
            .send(HostEvent::Ready {
                restore_sidebar: false,
            })
            .unwrap();
    }

    #[test]
    fn handler_errors_are_failures() {
        let harness = TestHarness::<Probe>::new().unwrap();
        let error = harness
            .send(HostEvent::FileClosed { file_id: 1 })
            .unwrap_err();
        assert_eq!(error.kind(), FfiErrorKind::Failed);
        assert_eq!(error.message.as_str(), "cannot close");
    }

    #[test]
    fn context_menus_and_callbacks_return_values() {
        let harness = TestHarness::<Probe>::new().unwrap();
        let items = harness
            .context_menu(MenuContext {
                pane: 1,
                view: view(),
                image_x: 5.0,
                image_y: 6.0,
                hit: None,
                selection: Vec::new(),
            })
            .unwrap();
        assert_eq!(items[0].item_id, "at-5");
        let reply = harness
            .ui_callback("surface-1", "clicked", vec![json!(1), json!("x")])
            .unwrap();
        assert_eq!(reply, json!({ "name": "clicked", "argc": 2 }));
    }

    #[test]
    fn filters_modify_frames_in_place() {
        let harness = TestHarness::<Probe>::new().unwrap();
        let mut pixels = vec![10, 20, 30, 255, 40, 50, 60, 255];
        harness
            .apply_filter("invert", &mut pixels, 2, 1, FilterView::default())
            .unwrap();
        assert_eq!(pixels, vec![245, 20, 30, 255, 215, 50, 60, 255]);
    }

    #[test]
    fn short_frames_are_rejected() {
        let harness = TestHarness::<Probe>::new().unwrap();
        let mut pixels = vec![0u8; 7];
        let error = harness
            .apply_filter("invert", &mut pixels, 2, 1, FilterView::default())
            .unwrap_err();
        assert_eq!(error.kind(), FfiErrorKind::InvalidArguments);
    }

    #[test]
    fn unknown_methods_are_unsupported() {
        let host = MockHost::new();
        let sdk_host = Host::from_ffi(host.to_ffi());
        let error = sdk_host
            .call::<_, Value>("does.not.exist", &json!({}))
            .unwrap_err();
        assert!(error.is_unsupported());
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        let host = MockHost::new();
        let sdk_host = Host::from_ffi(host.to_ffi());
        let layer = sdk_host.create_layer(LayerSpec::new("l", 1)).unwrap();
        let error = layer
            .upsert(vec![OverlayObject::new(
                "bad",
                Shape::Polyline {
                    points: vec![[0.0, f64::NAN]],
                },
            )])
            .unwrap_err();
        assert_eq!(error.kind, FfiErrorKind::InvalidArguments);
    }

    #[test]
    fn settings_round_trip() {
        let host = MockHost::new();
        let sdk_host = Host::from_ffi(host.to_ffi());
        sdk_host.set_setting("threshold", &0.5).unwrap();
        assert_eq!(sdk_host.setting::<f64>("threshold").unwrap(), Some(0.5));
        sdk_host.remove_setting("threshold").unwrap();
        assert_eq!(sdk_host.setting::<f64>("threshold").unwrap(), None);
    }
}
