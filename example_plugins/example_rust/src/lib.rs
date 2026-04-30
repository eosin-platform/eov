//! Example plugin for EOV.
//!
//! Demonstrates how to:
//! - Implement the `Plugin` trait
//! - Register a toolbar button with a smiley icon
//! - Open a runtime-loaded `.slint` window on button click
//! - Provide testable non-UI Rust logic (an event log accumulator)

pub mod event_log;

use event_log::EventLog;
use plugin_api::{
    HostContext, IconDescriptor, Plugin, PluginManifest, PluginResult, ToolbarButtonRegistration,
};
use std::path::Path;
use std::sync::Mutex;

/// Inline SVG smiley face icon.
pub const SMILEY_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>"#;

/// The action id dispatched when the toolbar button is clicked.
pub const ACTION_OPEN_PANEL: &str = "open_panel";

pub struct ExamplePlugin {
    manifest: PluginManifest,
    pub event_log: Mutex<EventLog>,
}

impl ExamplePlugin {
    pub fn new(manifest: PluginManifest) -> Self {
        Self {
            manifest,
            event_log: Mutex::new(EventLog::new()),
        }
    }

    /// Build a manifest suitable for the example plugin installed in a given root.
    pub fn default_manifest() -> PluginManifest {
        PluginManifest {
            id: "example_plugin".into(),
            name: "Example Plugin".into(),
            version: "0.1.0".into(),
            entry_ui: Some("ui/my_panel.slint".into()),
            entry_component: Some("MyPanel".into()),
            icon: Some(IconDescriptor::Svg {
                data: SMILEY_SVG.into(),
            }),
            language: Default::default(),
            entry_script: None,
            toolbar_buttons: Vec::new(),
        }
    }
}

impl Plugin for ExamplePlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn activate(&self, host: &mut dyn HostContext, _plugin_root: &Path) -> PluginResult<()> {
        host.add_toolbar_button(ToolbarButtonRegistration {
            plugin_id: self.manifest.id.clone(),
            button_id: "smiley".into(),
            tooltip: "Example Plugin".into(),
            icon: IconDescriptor::Svg {
                data: SMILEY_SVG.into(),
            },
            hotkey: None,
            action_id: ACTION_OPEN_PANEL.into(),
            tool_mode: None,
            active: false,
            toggled_icon: None,
        })?;
        self.event_log.lock().unwrap().record("plugin_activated");
        Ok(())
    }

    fn on_action(
        &self,
        action_id: &str,
        host: &mut dyn HostContext,
        plugin_root: &Path,
    ) -> PluginResult<()> {
        if action_id == ACTION_OPEN_PANEL {
            self.event_log
                .lock()
                .unwrap()
                .record("open_panel_requested");
            // Ask the host to open our plugin window
            let ui_path = self
                .manifest
                .resolve_entry_ui(plugin_root)
                .expect("example plugin must have entry_ui");
            host.open_plugin_window(
                &self.manifest.id,
                &ui_path,
                self.manifest
                    .entry_component
                    .as_deref()
                    .expect("example plugin must have entry_component"),
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dynamic library FFI exports (loaded by the host via abi_stable RawLibrary)
// ---------------------------------------------------------------------------

use abi_stable::std_types::{ROption, RString, RVec};
use plugin_api::ffi::{
    ActionResponseFFI, HostApiVTable, HostLogLevelFFI, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayComponentRequestFFI, ViewportOverlayPointFFI, ViewportOverlayPolygonFFI,
    ViewportOverlayVertexFFI, ViewportSnapshotFFI,
};

static HOST_API: Mutex<Option<HostApiVTable>> = Mutex::new(None);

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    *HOST_API.lock().unwrap() = Some(host_api);
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: RString::from("smiley"),
        tooltip: RString::from("Example Plugin"),
        icon_svg: RString::from(SMILEY_SVG),
        action_id: RString::from(ACTION_OPEN_PANEL),
        tool_mode: abi_stable::std_types::ROption::RNone,
        hotkey: abi_stable::std_types::ROption::RNone,
    }])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    if let Some(host_api) = *HOST_API.lock().unwrap() {
        let snapshot = (host_api.get_snapshot)(host_api.context);
        let message = match snapshot.active_file.into_option() {
            Some(file) => format!(
                "Example plugin opened from {} at {:.0}x{:.0}",
                file.filename, file.width, file.height
            ),
            None => "Example plugin opened with no active file".to_string(),
        };
        (host_api.log_message)(
            host_api.context,
            HostLogLevelFFI::Info,
            RString::from(message),
        );
    }

    println!(
        "[example_plugin] Button pressed! action_id=\"{}\"",
        action_id.as_str()
    );
    ActionResponseFFI {
        open_window: action_id.as_str() == ACTION_OPEN_PANEL,
    }
}

extern "C" fn on_hud_action_ffi(
    _action_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_ui_callback_ffi(callback_name: RString, _args_json: RString) {
    println!(
        "[example_plugin] UI callback invoked: \"{}\"",
        callback_name.as_str()
    );
}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
    RVec::new()
}

extern "C" fn get_viewport_context_menu_items_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    RVec::new()
}

extern "C" fn on_viewport_context_menu_action_ffi(
    _item_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_overlay_points_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    RVec::new()
}

extern "C" fn get_viewport_overlay_polygons_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPolygonFFI> {
    RVec::new()
}

extern "C" fn get_viewport_overlay_component_ffi() -> ROption<ViewportOverlayComponentRequestFFI> {
    ROption::RNone
}

extern "C" fn get_viewport_overlay_properties_ffi(
    _viewport: ViewportSnapshotFFI,
) -> RVec<UiPropertyFFI> {
    RVec::new()
}

extern "C" fn on_point_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_point_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _x_level0: f64,
    _y_level0: f64,
) {
}

extern "C" fn on_polygon_annotation_placed_ffi(
    _viewport: ViewportSnapshotFFI,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn on_polygon_annotation_moved_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
    _vertices: RVec<ViewportOverlayVertexFFI>,
) {
}

extern "C" fn on_undo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_redo_ffi() -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    RVec::new()
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    _rgba_data: *mut u8,
    _len: u32,
    _width: u32,
    _height: u32,
    _viewport: *const plugin_api::ffi::ViewportSnapshotFFI,
) -> bool {
    false
}

extern "C" fn apply_filter_gpu_ffi(
    _filter_id: RString,
    _ctx: *const plugin_api::ffi::GpuFilterContextFFI,
) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {}

extern "C" fn on_viewport_annotation_selected_ffi(
    _viewport: ViewportSnapshotFFI,
    _annotation_id: RString,
) {
}

/// The factory function exported by this plugin for dynamic loading by the host.
#[cfg_attr(feature = "export-vtable-symbol", unsafe(no_mangle))]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        set_host_api: set_host_api_ffi,
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        get_hud_toolbar_buttons: get_hud_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_hud_action: on_hud_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
        get_sidebar_properties: get_sidebar_properties_ffi,
        get_viewport_context_menu_items: get_viewport_context_menu_items_ffi,
        on_viewport_context_menu_action: on_viewport_context_menu_action_ffi,
        get_viewport_overlay_points: get_viewport_overlay_points_ffi,
        get_viewport_overlay_polygons: get_viewport_overlay_polygons_ffi,
        get_viewport_overlay_component: get_viewport_overlay_component_ffi,
        get_viewport_overlay_properties: get_viewport_overlay_properties_ffi,
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_polygon_annotation_placed: on_polygon_annotation_placed_ffi,
        on_undo: on_undo_ffi,
        on_redo: on_redo_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        on_polygon_annotation_moved: on_polygon_annotation_moved_ffi,
        on_viewport_annotation_selected: on_viewport_annotation_selected_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_manifest_valid() {
        let m = ExamplePlugin::default_manifest();
        assert_eq!(m.id, "example_plugin");
        assert_eq!(m.entry_component.as_deref(), Some("MyPanel"));
        assert!(m.icon.is_some());
    }

    #[test]
    fn smiley_svg_is_valid() {
        assert!(SMILEY_SVG.contains("<svg"));
        assert!(SMILEY_SVG.contains("</svg>"));
    }
}
