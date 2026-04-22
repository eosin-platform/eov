//! Concrete `HostContext` implementation for activating plugins.
//!
//! `AppHostContext` collects toolbar registrations and window-open requests
//! produced during plugin activation. The host drains these after activation
//! to wire them into the real UI.

use crate::plugins::toolbar::ToolbarManager;
use plugin_api::{HostContext, PluginError, PluginResult, ToolbarButtonRegistration};
use std::path::{Path, PathBuf};

/// A request from a plugin to open a window.
#[derive(Debug, Clone)]
pub struct WindowOpenRequest {
    pub plugin_id: String,
    pub ui_path: PathBuf,
    pub component: String,
}

/// Host context passed to plugins during activation and action handling.
///
/// For activation, this accumulates toolbar registrations.
/// For action handling, this accumulates window-open requests.
pub struct AppHostContext<'a> {
    pub toolbar: &'a mut ToolbarManager,
    pub window_requests: Vec<WindowOpenRequest>,
}

impl<'a> AppHostContext<'a> {
    pub fn new(toolbar: &'a mut ToolbarManager) -> Self {
        Self {
            toolbar,
            window_requests: Vec::new(),
        }
    }
}

impl HostContext for AppHostContext<'_> {
    fn add_toolbar_button(&mut self, button: ToolbarButtonRegistration) -> PluginResult<()> {
        self.toolbar.register(button)
    }

    fn open_plugin_window(
        &mut self,
        plugin_id: &str,
        ui_path: &Path,
        component: &str,
    ) -> PluginResult<()> {
        if !ui_path.exists() {
            return Err(PluginError::MissingFile {
                plugin_id: plugin_id.to_string(),
                path: ui_path.to_path_buf(),
            });
        }
        self.window_requests.push(WindowOpenRequest {
            plugin_id: plugin_id.to_string(),
            ui_path: ui_path.to_path_buf(),
            component: component.to_string(),
        });
        Ok(())
    }
}

/// A mock host context for testing plugins without a UI runtime.
#[cfg(test)]
pub struct MockHostContext {
    pub toolbar: ToolbarManager,
    pub window_requests: Vec<WindowOpenRequest>,
}

#[cfg(test)]
impl MockHostContext {
    pub fn new() -> Self {
        Self {
            toolbar: ToolbarManager::new(),
            window_requests: Vec::new(),
        }
    }
}

#[cfg(test)]
impl HostContext for MockHostContext {
    fn add_toolbar_button(&mut self, button: ToolbarButtonRegistration) -> PluginResult<()> {
        self.toolbar.register(button)
    }

    fn open_plugin_window(
        &mut self,
        plugin_id: &str,
        ui_path: &Path,
        component: &str,
    ) -> PluginResult<()> {
        self.window_requests.push(WindowOpenRequest {
            plugin_id: plugin_id.to_string(),
            ui_path: ui_path.to_path_buf(),
            component: component.to_string(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use example_plugin::ExamplePlugin;
    use plugin_api::Plugin;

    use std::path::PathBuf;

    #[test]
    fn example_plugin_activation_registers_toolbar_button() {
        let mut mock = MockHostContext::new();
        let plugin = ExamplePlugin::new(ExamplePlugin::default_manifest());
        let root = PathBuf::from("/fake/plugin/root");
        plugin.activate(&mut mock, &root).unwrap();

        assert_eq!(mock.toolbar.len(), 1);
        let btn = &mock.toolbar.buttons()[0];
        assert_eq!(btn.plugin_id, "example_plugin");
        assert_eq!(btn.button_id, "smiley");
        assert_eq!(btn.tooltip, "Example Plugin");
        assert_eq!(btn.action_id, "open_panel");
    }

    #[test]
    fn example_plugin_action_requests_window() {
        let mut mock = MockHostContext::new();
        let plugin = ExamplePlugin::new(ExamplePlugin::default_manifest());
        let root = PathBuf::from("/fake/plugin/root");
        plugin.activate(&mut mock, &root).unwrap();

        // Trigger the action
        plugin
            .on_action(example_plugin::ACTION_OPEN_PANEL, &mut mock, &root)
            .unwrap();

        assert_eq!(mock.window_requests.len(), 1);
        let req = &mock.window_requests[0];
        assert_eq!(req.plugin_id, "example_plugin");
        assert_eq!(req.component, "MyPanel");
    }

    #[test]
    fn example_plugin_event_log_tracks_activation_and_action() {
        let mut mock = MockHostContext::new();
        let plugin = ExamplePlugin::new(ExamplePlugin::default_manifest());
        let root = PathBuf::from("/fake/plugin/root");

        plugin.activate(&mut mock, &root).unwrap();
        {
            let log = plugin.event_log.lock().unwrap();
            assert_eq!(log.len(), 1);
            assert_eq!(log.entries()[0].message, "plugin_activated");
        }

        plugin
            .on_action(example_plugin::ACTION_OPEN_PANEL, &mut mock, &root)
            .unwrap();
        {
            let log = plugin.event_log.lock().unwrap();
            assert_eq!(log.len(), 2);
            assert_eq!(log.entries()[1].message, "open_panel_requested");
        }
    }

    #[test]
    fn app_host_context_collects_toolbar_and_windows() {
        let mut toolbar = ToolbarManager::new();
        {
            let mut ctx = AppHostContext::new(&mut toolbar);

            ctx.add_toolbar_button(ToolbarButtonRegistration {
                plugin_id: "p".into(),
                button_id: "b".into(),
                tooltip: "T".into(),
                icon: plugin_api::IconDescriptor::Svg {
                    data: "<svg/>".into(),
                },
                action_id: "a".into(),
                tool_mode: None,
                active: false,
            })
            .unwrap();

            assert!(ctx.window_requests.is_empty());
        }
        assert_eq!(toolbar.len(), 1);
    }
}
