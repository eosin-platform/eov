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
            entry_ui: "ui/my_panel.slint".into(),
            entry_component: "MyPanel".into(),
            icon: Some(IconDescriptor::Svg {
                data: SMILEY_SVG.into(),
            }),
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
            action_id: ACTION_OPEN_PANEL.into(),
        })?;
        self.event_log
            .lock()
            .unwrap()
            .record("plugin_activated");
        Ok(())
    }

    fn on_action(&self, action_id: &str, host: &mut dyn HostContext, plugin_root: &Path) -> PluginResult<()> {
        if action_id == ACTION_OPEN_PANEL {
            self.event_log
                .lock()
                .unwrap()
                .record("open_panel_requested");
            // Ask the host to open our plugin window
            let ui_path = self.manifest.resolve_entry_ui(plugin_root);
            host.open_plugin_window(
                &self.manifest.id,
                &ui_path,
                &self.manifest.entry_component,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dynamic library FFI exports (loaded by the host via abi_stable RawLibrary)
// ---------------------------------------------------------------------------

use abi_stable::std_types::{RString, RVec};
use plugin_api::ffi::{ActionResponseFFI, PluginVTable, ToolbarButtonFFI};

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: RString::from("smiley"),
        tooltip: RString::from("Example Plugin"),
        icon_svg: RString::from(SMILEY_SVG),
        action_id: RString::from(ACTION_OPEN_PANEL),
    }])
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    println!(
        "[example_plugin] Button pressed! action_id=\"{}\"",
        action_id.as_str()
    );
    ActionResponseFFI {
        open_window: action_id.as_str() == ACTION_OPEN_PANEL,
    }
}

extern "C" fn on_ui_callback_ffi(callback_name: RString) {
    println!(
        "[example_plugin] UI callback invoked: \"{}\"",
        callback_name.as_str()
    );
}

/// The factory function exported by this plugin for dynamic loading by the host.
#[unsafe(no_mangle)]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_manifest_valid() {
        let m = ExamplePlugin::default_manifest();
        assert_eq!(m.id, "example_plugin");
        assert_eq!(m.entry_component, "MyPanel");
        assert!(m.icon.is_some());
    }

    #[test]
    fn smiley_svg_is_valid() {
        assert!(SMILEY_SVG.contains("<svg"));
        assert!(SMILEY_SVG.contains("</svg>"));
    }
}

