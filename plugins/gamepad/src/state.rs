use abi_stable::std_types::RString;
use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};
use plugin_api::ffi::{HostApiVTable, HostLogLevelFFI, UiPropertyFFI};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, SystemTime};

const TOOLBAR_BUTTON_ID: &str = "toggle_gamepad";
const WINDOW_HEARTBEAT_STALE_AFTER: Duration = Duration::from_millis(1200);
const WINDOW_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(250);

const AXIS_ACTION_KEYS: &[&str] = &[
    "none",
    "pan_horizontal",
    "pan_vertical",
    "zoom",
    "zoom_in",
    "zoom_out",
];
const BUTTON_ACTION_KEYS: &[&str] = &[
    "none",
    "previous_pane",
    "next_pane",
    "previous_tab",
    "next_tab",
    "fit_view",
    "toggle_panel",
    "toggle_controller",
    "zoom_in",
    "zoom_out",
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MappingAction {
    None,
    PanHorizontal,
    PanVertical,
    Zoom,
    PreviousPane,
    NextPane,
    PreviousTab,
    NextTab,
    FitView,
    TogglePanel,
    ToggleController,
    ZoomIn,
    ZoomOut,
}

impl MappingAction {
    fn key(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PanHorizontal => "pan_horizontal",
            Self::PanVertical => "pan_vertical",
            Self::Zoom => "zoom",
            Self::PreviousPane => "previous_pane",
            Self::NextPane => "next_pane",
            Self::PreviousTab => "previous_tab",
            Self::NextTab => "next_tab",
            Self::FitView => "fit_view",
            Self::TogglePanel => "toggle_panel",
            Self::ToggleController => "toggle_controller",
            Self::ZoomIn => "zoom_in",
            Self::ZoomOut => "zoom_out",
        }
    }

    fn from_key(key: &str) -> Self {
        match key {
            "pan_horizontal" => Self::PanHorizontal,
            "pan_vertical" => Self::PanVertical,
            "zoom" => Self::Zoom,
            "previous_pane" => Self::PreviousPane,
            "next_pane" => Self::NextPane,
            "previous_tab" => Self::PreviousTab,
            "next_tab" => Self::NextTab,
            "fit_view" => Self::FitView,
            "toggle_panel" => Self::TogglePanel,
            "toggle_controller" => Self::ToggleController,
            "zoom_in" => Self::ZoomIn,
            "zoom_out" => Self::ZoomOut,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ControlDescriptor {
    pub key: &'static str,
    pub label: &'static str,
    pub axis: bool,
}

const CONTROL_DESCRIPTORS: &[ControlDescriptor] = &[
    ControlDescriptor { key: "left_stick_x", label: "Left Stick X", axis: true },
    ControlDescriptor { key: "left_stick_y", label: "Left Stick Y", axis: true },
    ControlDescriptor { key: "right_stick_x", label: "Right Stick X", axis: true },
    ControlDescriptor { key: "right_stick_y", label: "Right Stick Y", axis: true },
    ControlDescriptor { key: "left_trigger", label: "Left Trigger", axis: true },
    ControlDescriptor { key: "right_trigger", label: "Right Trigger", axis: true },
    ControlDescriptor { key: "dpad_left", label: "D-Pad Left", axis: false },
    ControlDescriptor { key: "dpad_right", label: "D-Pad Right", axis: false },
    ControlDescriptor { key: "dpad_up", label: "D-Pad Up", axis: false },
    ControlDescriptor { key: "dpad_down", label: "D-Pad Down", axis: false },
    ControlDescriptor { key: "south", label: "South Button", axis: false },
    ControlDescriptor { key: "east", label: "East Button", axis: false },
    ControlDescriptor { key: "north", label: "North Button", axis: false },
    ControlDescriptor { key: "west", label: "West Button", axis: false },
    ControlDescriptor { key: "left_trigger_button", label: "Left Shoulder", axis: false },
    ControlDescriptor { key: "right_trigger_button", label: "Right Shoulder", axis: false },
    ControlDescriptor { key: "start", label: "Start", axis: false },
    ControlDescriptor { key: "select", label: "Select", axis: false },
];

#[derive(Debug, Clone)]
pub struct DeviceSummary {
    pub key: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedProfile {
    pub left_dead_zone: f32,
    pub right_dead_zone: f32,
    pub trigger_dead_zone: f32,
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub mappings: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct PluginState {
    pub host_api: Option<HostApiVTable>,
    pub controller_enabled: bool,
    pub devices: Vec<DeviceSummary>,
    pub selected_device_key: Option<String>,
    pub left_dead_zone: f32,
    pub right_dead_zone: f32,
    pub trigger_dead_zone: f32,
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub mappings: BTreeMap<String, MappingAction>,
    pub profile_draft_name: String,
    pub available_profiles: Vec<String>,
    pub status_text: String,
}

impl Default for PluginState {
    fn default() -> Self {
        let mut state = Self {
            host_api: None,
            controller_enabled: true,
            devices: Vec::new(),
            selected_device_key: None,
            left_dead_zone: 0.16,
            right_dead_zone: 0.12,
            trigger_dead_zone: 0.08,
            pan_sensitivity: 0.95,
            zoom_sensitivity: 0.78,
            mappings: default_mappings(),
            profile_draft_name: "default".to_string(),
            available_profiles: Vec::new(),
            status_text: "Waiting for a controller".to_string(),
        };
        if let Some(runtime) = load_runtime_state().ok().flatten() {
            apply_persisted_profile(&mut state, runtime);
        }
        state.available_profiles = discover_profile_names();
        state
    }
}

static HOST_API: OnceLock<Mutex<Option<HostApiVTable>>> = OnceLock::new();
static PLUGIN_STATE: OnceLock<Mutex<PluginState>> = OnceLock::new();
static WORKER_ONCE: Once = Once::new();
static WINDOW_WATCHER_ONCE: Once = Once::new();

pub fn plugin_state() -> &'static Mutex<PluginState> {
    PLUGIN_STATE.get_or_init(|| Mutex::new(PluginState::default()))
}

pub fn set_host_api(host_api: HostApiVTable) {
    *HOST_API.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(host_api);
    plugin_state().lock().unwrap().host_api = Some(host_api);
}

pub fn host_api() -> Option<HostApiVTable> {
    HOST_API
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .to_owned()
}

pub fn ensure_worker_started() {
    WORKER_ONCE.call_once(|| {
        std::thread::Builder::new()
            .name("gamepad-plugin-worker".to_string())
            .spawn(worker_loop)
            .expect("failed to spawn gamepad plugin worker");
    });
}

pub fn prepare_window_runtime() {
    sync_runtime_state_from_disk();
    ensure_worker_started();
    if host_api().is_none() {
        ensure_window_watcher_started();
    }
}

pub fn axis_action_options() -> Vec<&'static str> {
    AXIS_ACTION_KEYS.iter().map(|key| action_label(key)).collect()
}

pub fn button_action_options() -> Vec<&'static str> {
    BUTTON_ACTION_KEYS.iter().map(|key| action_label(key)).collect()
}

pub fn action_index(action: MappingAction, axis: bool) -> i32 {
    let options = if axis { AXIS_ACTION_KEYS } else { BUTTON_ACTION_KEYS };
    options
        .iter()
        .position(|candidate| *candidate == action.key())
        .unwrap_or(0) as i32
}

pub fn profile_names() -> Vec<String> {
    plugin_state().lock().unwrap().available_profiles.clone()
}

pub fn profile_selected_index() -> i32 {
    let state = plugin_state().lock().unwrap();
    state
        .available_profiles
        .iter()
        .position(|name| *name == state.profile_draft_name)
        .map(|index| index as i32)
        .unwrap_or(-1)
}

pub fn selected_device_index() -> i32 {
    let state = plugin_state().lock().unwrap();
    state
        .selected_device_key
        .as_ref()
        .and_then(|selected| state.devices.iter().position(|device| &device.key == selected))
        .map(|index| index as i32)
        .unwrap_or(-1)
}

pub fn device_labels() -> Vec<String> {
    plugin_state()
        .lock()
        .unwrap()
        .devices
        .iter()
        .map(|device| device.label.clone())
        .collect()
}

pub fn toggle_window() -> bool {
    prepare_window_runtime();
    let open = window_is_open();
    if open {
        let _ = request_window_close();
        set_toolbar_button_active(false);
        false
    } else {
        clear_close_request();
        set_toolbar_button_active(true);
        true
    }
}

pub fn save_profile(name: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("profile name is required".to_string());
    }
    let state = plugin_state().lock().unwrap().clone();
    let profile = PersistedProfile {
        left_dead_zone: state.left_dead_zone,
        right_dead_zone: state.right_dead_zone,
        trigger_dead_zone: state.trigger_dead_zone,
        pan_sensitivity: state.pan_sensitivity,
        zoom_sensitivity: state.zoom_sensitivity,
        mappings: state
            .mappings
            .iter()
            .map(|(key, action)| (key.clone(), action.key().to_string()))
            .collect(),
    };
    let profile_dir = profiles_dir()?;
    fs::create_dir_all(&profile_dir).map_err(|err| err.to_string())?;
    let path = profile_dir.join(format!("{trimmed}.json"));
    let payload = serde_json::to_string_pretty(&profile).map_err(|err| err.to_string())?;
    fs::write(path, payload).map_err(|err| err.to_string())?;

    let mut state = plugin_state().lock().unwrap();
    state.profile_draft_name = trimmed.to_string();
    state.available_profiles = discover_profile_names();
    persist_runtime_state(&state)?;
    Ok(())
}

pub fn load_profile(name: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("profile name is required".to_string());
    }
    let path = profiles_dir()?.join(format!("{trimmed}.json"));
    let raw = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let persisted: PersistedProfile = serde_json::from_str(&raw).map_err(|err| err.to_string())?;
    let mut state = plugin_state().lock().unwrap();
    state.left_dead_zone = clamp_unit(persisted.left_dead_zone);
    state.right_dead_zone = clamp_unit(persisted.right_dead_zone);
    state.trigger_dead_zone = clamp_unit(persisted.trigger_dead_zone);
    state.pan_sensitivity = clamp_sensitivity(persisted.pan_sensitivity);
    state.zoom_sensitivity = clamp_sensitivity(persisted.zoom_sensitivity);
    state.mappings = persisted
        .mappings
        .into_iter()
        .map(|(key, value)| (key, MappingAction::from_key(&value)))
        .collect();
    for descriptor in CONTROL_DESCRIPTORS {
        state
            .mappings
            .entry(descriptor.key.to_string())
            .or_insert(MappingAction::None);
    }
    state.profile_draft_name = trimmed.to_string();
    state.available_profiles = discover_profile_names();
    persist_runtime_state(&state)?;
    Ok(())
}

pub fn delete_profile(name: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("profile name is required".to_string());
    }
    let path = profiles_dir()?.join(format!("{trimmed}.json"));
    if path.exists() {
        fs::remove_file(path).map_err(|err| err.to_string())?;
    }
    let mut state = plugin_state().lock().unwrap();
    state.available_profiles = discover_profile_names();
    if state.profile_draft_name == trimmed {
        state.profile_draft_name = "default".to_string();
    }
    persist_runtime_state(&state)?;
    Ok(())
}

pub fn log_message(level: HostLogLevelFFI, message: impl Into<String>) {
    if let Some(host) = host_api() {
        (host.log_message)(host.context, level, RString::from(message.into()));
    }
}

pub fn refresh_window_ui_if_available() {
    if let Some(host) = host_api() {
        let _ = (host.refresh_sidebar)(host.context);
    }
}

pub fn property(name: &str, value: serde_json::Value) -> UiPropertyFFI {
    UiPropertyFFI {
        name: name.into(),
        json_value: value.to_string().into(),
    }
}

pub fn set_selected_device_by_label(label: &str) {
    let mut state = plugin_state().lock().unwrap();
    if let Some((device_key, device_label)) = state
        .devices
        .iter()
        .find(|device| device.label == label)
        .map(|device| (device.key.clone(), device.label.clone()))
    {
        state.selected_device_key = Some(device_key);
        state.status_text = format!("Ready: {device_label}");
    }
}

pub fn set_profile_draft_name(name: String) {
    let mut state = plugin_state().lock().unwrap();
    state.profile_draft_name = name;
    let _ = persist_runtime_state(&state);
}

pub fn set_controller_enabled(enabled: bool) {
    let mut state = plugin_state().lock().unwrap();
    state.controller_enabled = enabled;
    if enabled {
        state.status_text = current_ready_status(&state);
    } else {
        state.status_text = current_paused_status(&state);
    }
    let _ = persist_runtime_state(&state);
}

pub fn set_numeric_field(field: &str, value: &str) {
    let Ok(parsed) = value.trim().parse::<f32>() else {
        return;
    };
    let mut state = plugin_state().lock().unwrap();
    match field {
        "left-dead-zone" => state.left_dead_zone = clamp_unit(parsed),
        "right-dead-zone" => state.right_dead_zone = clamp_unit(parsed),
        "trigger-dead-zone" => state.trigger_dead_zone = clamp_unit(parsed),
        "pan-sensitivity" => state.pan_sensitivity = clamp_sensitivity(parsed),
        "zoom-sensitivity" => state.zoom_sensitivity = clamp_sensitivity(parsed),
        _ => {}
    }
    let _ = persist_runtime_state(&state);
}

pub fn set_mapping_by_label(control_key: &str, selected_label: &str, axis: bool) {
    let options = if axis { AXIS_ACTION_KEYS } else { BUTTON_ACTION_KEYS };
    let Some(action_key) = options
        .iter()
        .copied()
        .find(|candidate| action_label(*candidate) == selected_label)
    else {
        return;
    };
    plugin_state()
        .lock()
        .unwrap()
        .mappings
        .insert(control_key.to_string(), MappingAction::from_key(action_key));
    let state = plugin_state().lock().unwrap();
    let _ = persist_runtime_state(&state);
}

fn worker_loop() {
    let mut gilrs = match Gilrs::new() {
        Ok(gilrs) => gilrs,
        Err(err) => {
            plugin_state().lock().unwrap().status_text = format!("Gamepad subsystem unavailable: {err}");
            log_message(HostLogLevelFFI::Warn, format!("gilrs unavailable: {err}"));
            refresh_window_ui_if_available();
            return;
        }
    };

    let mut axis_values: HashMap<String, f32> = HashMap::new();
    let mut pressed_buttons: HashSet<String> = HashSet::new();
    let mut last_visible_devices = Vec::<String>::new();
    let mut last_runtime_sync = SystemTime::UNIX_EPOCH;
    let mut last_window_open_state = false;

    loop {
        if host_api().is_some()
            && last_runtime_sync.elapsed().unwrap_or_default() >= Duration::from_millis(250)
        {
            sync_runtime_state_from_disk();
            last_runtime_sync = SystemTime::now();
        }

        let devices = collect_devices(&gilrs);
        let mut refresh_window = false;
        let selected_device_key = {
            let mut state = plugin_state().lock().unwrap();
            if state.devices.iter().map(|device| device.key.as_str()).collect::<Vec<_>>()
                != devices.iter().map(|device| device.key.as_str()).collect::<Vec<_>>()
            {
                state.devices = devices.clone();
                if state.selected_device_key.as_ref().is_none_or(|selected| {
                    !state.devices.iter().any(|device| &device.key == selected)
                }) {
                    state.selected_device_key = state.devices.first().map(|device| device.key.clone());
                }
                if let Some(selected) = &state.selected_device_key
                    && let Some(device) = state.devices.iter().find(|device| &device.key == selected)
                {
                    state.status_text = if state.controller_enabled {
                        format!("Ready: {}", device.label)
                    } else {
                        format!("Controller paused: {}", device.label)
                    };
                } else {
                    state.status_text = "Waiting for a controller".to_string();
                }
                refresh_window = true;
            }
            state.selected_device_key.clone()
        };

        while let Some(event) = gilrs.next_event() {
            let target_selected = selected_device_key
                .as_ref()
                .is_some_and(|selected| *selected == device_key(event.id, gilrs.gamepad(event.id).name()));

            match event.event {
                EventType::AxisChanged(axis, value, _) if target_selected => {
                    if let Some(key) = axis_key(axis) {
                        axis_values.insert(key.to_string(), value);
                    }
                }
                EventType::ButtonPressed(button, _) if target_selected => {
                    if let Some(key) = button_key(button) {
                        pressed_buttons.insert(key.to_string());
                        dispatch_button_action(key);
                    }
                }
                EventType::ButtonReleased(button, _) if target_selected => {
                    if let Some(key) = button_key(button) {
                        pressed_buttons.remove(key);
                    }
                }
                EventType::Connected | EventType::Disconnected => {
                    refresh_window = true;
                }
                _ => {}
            }
        }

        let current_visible_devices = collect_devices(&gilrs)
            .into_iter()
            .map(|device| device.key)
            .collect::<Vec<_>>();
        if current_visible_devices != last_visible_devices {
            last_visible_devices = current_visible_devices;
            refresh_window = true;
        }

        apply_continuous_mappings(&axis_values, &pressed_buttons);

        let window_open = window_is_open();
        if window_open != last_window_open_state {
            set_toolbar_button_active(window_open);
            last_window_open_state = window_open;
        }

        if refresh_window {
            let state = plugin_state().lock().unwrap();
            let _ = persist_runtime_state(&state);
            refresh_window_ui_if_available();
        }

        std::thread::sleep(Duration::from_millis(12));
    }
}

fn dispatch_button_action(control_key: &str) {
    let (action, host) = {
        let state = plugin_state().lock().unwrap();
        (
            state
                .mappings
                .get(control_key)
                .copied()
                .unwrap_or(MappingAction::None),
            state.host_api,
        )
    };
    let Some(host) = host else {
        return;
    };
    match action {
        MappingAction::PreviousPane => {
            let _ = (host.cycle_focused_pane)(host.context, -1);
        }
        MappingAction::NextPane => {
            let _ = (host.cycle_focused_pane)(host.context, 1);
        }
        MappingAction::PreviousTab => {
            let _ = (host.cycle_active_tab)(host.context, -1);
        }
        MappingAction::NextTab => {
            let _ = (host.cycle_active_tab)(host.context, 1);
        }
        MappingAction::FitView => {
            let _ = (host.fit_active_viewport)(host.context);
        }
        MappingAction::TogglePanel => {
            let _ = toggle_window();
        }
        MappingAction::ToggleController => {
            let enabled = {
                let mut state = plugin_state().lock().unwrap();
                state.controller_enabled = !state.controller_enabled;
                state.controller_enabled
            };
            log_message(
                HostLogLevelFFI::Info,
                if enabled {
                    "gamepad controller enabled"
                } else {
                    "gamepad controller paused"
                },
            );
            let state = plugin_state().lock().unwrap();
            let _ = persist_runtime_state(&state);
            refresh_window_ui_if_available();
        }
        MappingAction::ZoomIn => {
            nudge_zoom(1.14);
        }
        MappingAction::ZoomOut => {
            nudge_zoom(1.0 / 1.14);
        }
        _ => {}
    }
}

fn apply_continuous_mappings(axis_values: &HashMap<String, f32>, _pressed_buttons: &HashSet<String>) {
    let snapshot = {
        let state = plugin_state().lock().unwrap();
        if !state.controller_enabled {
            return;
        }
        let Some(host) = state.host_api else {
            return;
        };
        let host_snapshot = (host.get_snapshot)(host.context);
        let Some(viewport) = host_snapshot.active_viewport.into_option() else {
            return;
        };
        let x_span = (viewport.bounds_right - viewport.bounds_left).max(1.0);
        let y_span = (viewport.bounds_bottom - viewport.bounds_top).max(1.0);
        let mut pan_x = 0.0f64;
        let mut pan_y = 0.0f64;
        let mut zoom_signal = 0.0f64;

        for descriptor in CONTROL_DESCRIPTORS.iter().filter(|descriptor| descriptor.axis) {
            let value = normalized_axis_value(
                descriptor.key,
                axis_values.get(descriptor.key).copied().unwrap_or_default(),
            );
            let filtered = apply_dead_zone(descriptor.key, value, &state);
            if filtered.abs() <= f32::EPSILON {
                continue;
            }
            match state
                .mappings
                .get(descriptor.key)
                .copied()
                .unwrap_or(MappingAction::None)
            {
                MappingAction::PanHorizontal => pan_x += filtered as f64,
                MappingAction::PanVertical => pan_y += filtered as f64,
                MappingAction::Zoom => zoom_signal += -(filtered as f64),
                MappingAction::ZoomIn => zoom_signal += filtered.abs() as f64,
                MappingAction::ZoomOut => zoom_signal -= filtered.abs() as f64,
                _ => {}
            }
        }

        let center_x = viewport.center_x + x_span * pan_x * state.pan_sensitivity as f64 * 0.024;
        let center_y = viewport.center_y + y_span * pan_y * state.pan_sensitivity as f64 * 0.024;
        let zoom = if zoom_signal.abs() > 0.001 {
            viewport.zoom * f64::exp(zoom_signal * state.zoom_sensitivity as f64 * 0.04)
        } else {
            viewport.zoom
        };
        (host, center_x, center_y, zoom, viewport.center_x, viewport.center_y, viewport.zoom)
    };

    let (host, center_x, center_y, zoom, previous_x, previous_y, previous_zoom) = snapshot;
    if (center_x - previous_x).abs() > 0.0001
        || (center_y - previous_y).abs() > 0.0001
        || (zoom - previous_zoom).abs() > 0.0001
    {
        let _ = (host.set_active_viewport)(host.context, center_x, center_y, zoom);
    }
}

fn nudge_zoom(factor: f64) {
    let Some(host) = host_api() else {
        return;
    };
    let snapshot = (host.get_snapshot)(host.context);
    let Some(viewport) = snapshot.active_viewport.into_option() else {
        return;
    };
    let _ = (host.set_active_viewport)(
        host.context,
        viewport.center_x,
        viewport.center_y,
        viewport.zoom * factor,
    );
}

fn collect_devices(gilrs: &Gilrs) -> Vec<DeviceSummary> {
    gilrs
        .gamepads()
        .map(|(id, gamepad)| DeviceSummary {
            key: device_key(id, gamepad.name()),
            label: format!("{} ({:?})", gamepad.name(), gamepad.power_info()),
        })
        .collect()
}

fn device_key(id: GamepadId, name: &str) -> String {
    format!("{}::{}", usize::from(id), name)
}

fn axis_key(axis: Axis) -> Option<&'static str> {
    match axis {
        Axis::LeftStickX => Some("left_stick_x"),
        Axis::LeftStickY => Some("left_stick_y"),
        Axis::RightStickX => Some("right_stick_x"),
        Axis::RightStickY => Some("right_stick_y"),
        Axis::LeftZ => Some("left_trigger"),
        Axis::RightZ => Some("right_trigger"),
        _ => None,
    }
}

fn normalized_axis_value(control_key: &str, value: f32) -> f32 {
    match control_key {
        "left_stick_y" | "right_stick_y" => -value,
        _ => value,
    }
}

fn button_key(button: Button) -> Option<&'static str> {
    match button {
        Button::DPadLeft => Some("dpad_left"),
        Button::DPadRight => Some("dpad_right"),
        Button::DPadUp => Some("dpad_up"),
        Button::DPadDown => Some("dpad_down"),
        Button::South => Some("south"),
        Button::East => Some("east"),
        Button::North => Some("north"),
        Button::West => Some("west"),
        Button::LeftTrigger => Some("left_trigger_button"),
        Button::RightTrigger => Some("right_trigger_button"),
        Button::Start => Some("start"),
        Button::Select => Some("select"),
        _ => None,
    }
}

fn apply_dead_zone(control_key: &str, value: f32, state: &PluginState) -> f32 {
    let threshold = match control_key {
        "left_stick_x" | "left_stick_y" => state.left_dead_zone,
        "right_stick_x" | "right_stick_y" => state.right_dead_zone,
        "left_trigger" | "right_trigger" => state.trigger_dead_zone,
        _ => 0.0,
    };
    if value.abs() < threshold {
        0.0
    } else {
        value
    }
}

fn default_mappings() -> BTreeMap<String, MappingAction> {
    let mut mappings = CONTROL_DESCRIPTORS
        .iter()
        .map(|descriptor| (descriptor.key.to_string(), MappingAction::None))
        .collect::<BTreeMap<_, _>>();
    mappings.insert("left_stick_x".to_string(), MappingAction::PanHorizontal);
    mappings.insert("left_stick_y".to_string(), MappingAction::PanVertical);
    mappings.insert("right_stick_y".to_string(), MappingAction::Zoom);
    mappings.insert("dpad_left".to_string(), MappingAction::PreviousPane);
    mappings.insert("dpad_right".to_string(), MappingAction::NextPane);
    mappings.insert("dpad_up".to_string(), MappingAction::PreviousTab);
    mappings.insert("dpad_down".to_string(), MappingAction::NextTab);
    mappings
}

fn profiles_dir() -> Result<PathBuf, String> {
    let Some(root) = dirs::config_dir() else {
        return Err("config directory is unavailable".to_string());
    };
    Ok(root.join("eov").join("plugins").join("gamepad").join("profiles"))
}

fn runtime_state_path() -> Result<PathBuf, String> {
    let Some(root) = dirs::config_dir() else {
        return Err("config directory is unavailable".to_string());
    };
    Ok(root.join("eov").join("plugins").join("gamepad").join("state.json"))
}

fn window_state_dir() -> Result<PathBuf, String> {
    let Some(root) = dirs::config_dir() else {
        return Err("config directory is unavailable".to_string());
    };
    Ok(root.join("eov").join("plugins").join("gamepad").join("window"))
}

fn window_heartbeat_path() -> Result<PathBuf, String> {
    Ok(window_state_dir()?.join("heartbeat"))
}

fn window_close_request_path() -> Result<PathBuf, String> {
    Ok(window_state_dir()?.join("close-request"))
}

fn persist_runtime_state(state: &PluginState) -> Result<(), String> {
    let path = runtime_state_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let profile = PersistedProfile {
        left_dead_zone: state.left_dead_zone,
        right_dead_zone: state.right_dead_zone,
        trigger_dead_zone: state.trigger_dead_zone,
        pan_sensitivity: state.pan_sensitivity,
        zoom_sensitivity: state.zoom_sensitivity,
        mappings: state
            .mappings
            .iter()
            .map(|(key, action)| (key.clone(), action.key().to_string()))
            .collect(),
    };
    let payload = serde_json::to_string_pretty(&profile).map_err(|err| err.to_string())?;
    fs::write(path, payload).map_err(|err| err.to_string())
}

fn load_runtime_state() -> Result<Option<PersistedProfile>, String> {
    let path = runtime_state_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|err| err.to_string())
}

fn apply_persisted_profile(state: &mut PluginState, persisted: PersistedProfile) {
    state.left_dead_zone = clamp_unit(persisted.left_dead_zone);
    state.right_dead_zone = clamp_unit(persisted.right_dead_zone);
    state.trigger_dead_zone = clamp_unit(persisted.trigger_dead_zone);
    state.pan_sensitivity = clamp_sensitivity(persisted.pan_sensitivity);
    state.zoom_sensitivity = clamp_sensitivity(persisted.zoom_sensitivity);
    state.mappings = persisted
        .mappings
        .into_iter()
        .map(|(key, value)| (key, MappingAction::from_key(&value)))
        .collect();
    for descriptor in CONTROL_DESCRIPTORS {
        state
            .mappings
            .entry(descriptor.key.to_string())
            .or_insert(MappingAction::None);
    }
}

fn sync_runtime_state_from_disk() {
    let Ok(Some(runtime)) = load_runtime_state() else {
        return;
    };
    let mut state = plugin_state().lock().unwrap();
    apply_persisted_profile(&mut state, runtime);
}

fn ensure_window_watcher_started() {
    WINDOW_WATCHER_ONCE.call_once(|| {
        let heartbeat_path = match window_heartbeat_path() {
            Ok(path) => path,
            Err(_) => return,
        };
        let close_request_path = match window_close_request_path() {
            Ok(path) => path,
            Err(_) => return,
        };
        if let Some(parent) = heartbeat_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let initial_close_token = fs::read_to_string(&close_request_path).unwrap_or_default();
        let _ = std::thread::Builder::new()
            .name("gamepad-window-heartbeat".to_string())
            .spawn(move || {
                let mut last_close_token = initial_close_token;
                loop {
                    let _ = fs::write(&heartbeat_path, format!("{}", current_unix_millis()));
                    let current_close_token = fs::read_to_string(&close_request_path).unwrap_or_default();
                    if !current_close_token.is_empty() && current_close_token != last_close_token {
                        let _ = slint::quit_event_loop();
                        break;
                    }
                    last_close_token = current_close_token;
                    std::thread::sleep(WINDOW_HEARTBEAT_INTERVAL);
                }
            });
    });
}

fn clear_close_request() {
    if let Ok(path) = window_close_request_path() {
        let _ = fs::remove_file(path);
    }
}

fn request_window_close() -> Result<(), String> {
    let path = window_close_request_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(path, format!("{}", current_unix_millis())).map_err(|err| err.to_string())
}

fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn window_is_open() -> bool {
    let Ok(path) = window_heartbeat_path() else {
        return false;
    };
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return false;
    };
    modified.elapsed().unwrap_or(WINDOW_HEARTBEAT_STALE_AFTER) < WINDOW_HEARTBEAT_STALE_AFTER
}

fn set_toolbar_button_active(active: bool) {
    if let Some(host) = host_api() {
        let _ = (host.set_toolbar_button_active)(host.context, TOOLBAR_BUTTON_ID.into(), active);
    }
}

fn current_ready_status(state: &PluginState) -> String {
    state
        .selected_device_key
        .as_ref()
        .and_then(|selected| state.devices.iter().find(|device| &device.key == selected))
        .map(|device| format!("Ready: {}", device.label))
        .unwrap_or_else(|| "Waiting for a controller".to_string())
}

fn current_paused_status(state: &PluginState) -> String {
    state
        .selected_device_key
        .as_ref()
        .and_then(|selected| state.devices.iter().find(|device| &device.key == selected))
        .map(|device| format!("Controller paused: {}", device.label))
        .unwrap_or_else(|| "Controller paused".to_string())
}

fn discover_profile_names() -> Vec<String> {
    let Ok(profile_dir) = profiles_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(profile_dir) else {
        return Vec::new();
    };
    let mut profiles = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .path()
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    profiles.sort();
    profiles
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 0.95)
}

fn clamp_sensitivity(value: f32) -> f32 {
    value.clamp(0.05, 4.0)
}

fn action_label(key: &str) -> &'static str {
    match key {
        "pan_horizontal" => "Pan Horizontally",
        "pan_vertical" => "Pan Vertically",
        "zoom" => "Zoom Axis",
        "previous_pane" => "Previous Pane",
        "next_pane" => "Next Pane",
        "previous_tab" => "Previous Tab",
        "next_tab" => "Next Tab",
        "fit_view" => "Fit View",
        "toggle_panel" => "Toggle Panel",
        "toggle_controller" => "Enable / Pause",
        "zoom_in" => "Zoom In",
        "zoom_out" => "Zoom Out",
        _ => "Unassigned",
    }
}

pub fn mapping_rows_json() -> serde_json::Value {
    let state = plugin_state().lock().unwrap();
    json!(CONTROL_DESCRIPTORS.iter().map(|descriptor| {
        json!({
            "control_key": descriptor.key,
            "control_label": descriptor.label,
            "current_index": action_index(state.mappings.get(descriptor.key).copied().unwrap_or(MappingAction::None), descriptor.axis),
            "axis_control": descriptor.axis,
        })
    }).collect::<Vec<_>>())
}

pub fn current_status_text() -> String {
    plugin_state().lock().unwrap().status_text.clone()
}

pub fn current_profile_draft_name() -> String {
    plugin_state().lock().unwrap().profile_draft_name.clone()
}

pub fn controller_enabled() -> bool {
    plugin_state().lock().unwrap().controller_enabled
}

pub fn numeric_value_text(field: &str) -> String {
    let state = plugin_state().lock().unwrap();
    let value = match field {
        "left-dead-zone" => state.left_dead_zone,
        "right-dead-zone" => state.right_dead_zone,
        "trigger-dead-zone" => state.trigger_dead_zone,
        "pan-sensitivity" => state.pan_sensitivity,
        "zoom-sensitivity" => state.zoom_sensitivity,
        _ => 0.0,
    };
    format!("{value:.2}")
}