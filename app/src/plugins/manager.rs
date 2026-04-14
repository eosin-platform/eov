//! Plugin manager — orchestrates discovery, validation, and activation.
//!
//! The `PluginManager` is the main entry point for the host plugin system.
//! It discovers plugins on disk, matches them against the plugin registry,
//! validates their files, and activates them against the host API.
//!
//! Plugins are loaded in two ways:
//! 1. **Dynamic** (preferred): the plugin directory contains a shared library
//!    (e.g. `libexample_plugin.so`) that exports an `EOV_PLUGIN_VTABLE` symbol,
//!    loaded at runtime via `abi_stable::library::RawLibrary`.
//! 2. **Static** (fallback, used in tests): the plugin is compiled into the
//!    host binary and registered via `PluginRegistry`.

use crate::plugins::discovery;
use crate::plugins::host_context::{AppHostContext, WindowOpenRequest};
use crate::plugins::registry::PluginRegistry;
use crate::plugins::toolbar::ToolbarManager;
use abi_stable::library::RawLibrary;
use abi_stable::std_types::RString;
use plugin_api::ffi::{self, PluginVTable};
use plugin_api::{IconDescriptor, PluginDescriptor, PluginResult, ToolbarButtonRegistration};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Central manager for the plugin lifecycle.
pub struct PluginManager {
    pub registry: PluginRegistry,
    pub toolbar: ToolbarManager,
    /// Descriptors for all successfully discovered plugins on disk.
    pub descriptors: Vec<PluginDescriptor>,
    /// Plugin root directory.
    plugin_dir: PathBuf,
    /// Dynamically loaded plugin vtables, keyed by plugin id.
    loaded_vtables: HashMap<String, PluginVTable>,
}

impl PluginManager {
    pub fn new(plugin_dir: PathBuf) -> Self {
        Self {
            registry: PluginRegistry::new(),
            toolbar: ToolbarManager::new(),
            descriptors: Vec::new(),
            plugin_dir,
            loaded_vtables: HashMap::new(),
        }
    }

    /// Discover plugins from the plugin directory.
    pub fn discover(&mut self) {
        info!("Discovering plugins in {}", self.plugin_dir.display());
        self.descriptors = discovery::discover_plugins(&self.plugin_dir);
        info!("Discovered {} plugin(s)", self.descriptors.len());
    }

    /// Activate all discovered plugins.
    ///
    /// For each plugin, tries dynamic loading first (looks for a shared library
    /// in the plugin directory). Falls back to the static registry if no library
    /// is found.
    pub fn activate_all(&mut self) -> PluginResult<()> {
        let descriptors = self.descriptors.clone();
        for desc in &descriptors {
            // Try dynamic loading first
            let lib_name = ffi::plugin_library_filename(&desc.manifest.id);
            let lib_path = desc.root.join(&lib_name);
            if lib_path.exists() {
                match self.load_dynamic_plugin(&desc.manifest.id, &lib_path) {
                    Ok(()) => {
                        info!(
                            "Dynamically loaded plugin '{}' from {}",
                            desc.manifest.id,
                            lib_path.display()
                        );
                        continue;
                    }
                    Err(e) => {
                        warn!(
                            "Failed to dynamically load plugin '{}': {e}",
                            desc.manifest.id
                        );
                    }
                }
            }

            // Fall back to static registry
            let Some(plugin) = self.registry.get(&desc.manifest.id) else {
                info!(
                    "Plugin '{}' has no shared library and no static registration; skipping",
                    desc.manifest.id
                );
                continue;
            };

            // Validate referenced files exist
            if let Err(e) = desc.manifest.validate_files(&desc.root) {
                warn!(
                    "Plugin '{}' has missing files, skipping: {e}",
                    desc.manifest.id
                );
                continue;
            }

            let plugin = plugin.clone();
            let mut ctx = AppHostContext::new(&mut self.toolbar);
            match plugin.activate(&mut ctx, &desc.root) {
                Ok(()) => {
                    info!("Activated plugin '{}' (static)", desc.manifest.id);
                }
                Err(e) => {
                    warn!("Failed to activate plugin '{}': {e}", desc.manifest.id);
                }
            }
        }
        Ok(())
    }

    /// Load a plugin's shared library and register its toolbar buttons.
    fn load_dynamic_plugin(&mut self, plugin_id: &str, lib_path: &Path) -> Result<(), String> {
        let raw = RawLibrary::load_at(lib_path).map_err(|e| e.to_string())?;
        // Leak the library handle so the loaded code stays mapped for the
        // lifetime of the process (abi_stable mandates this).
        let raw: &'static RawLibrary = Box::leak(Box::new(raw));

        // SAFETY: The plugin crate and host both compile against the same
        // `plugin_api` crate, so the `PluginVTable` layout is guaranteed to
        // match at compile time via the shared `#[derive(StableAbi)]` type.
        // We load a function pointer (pointer-sized) which returns the vtable.
        let vtable: PluginVTable = unsafe {
            let sym = raw
                .get::<ffi::GetPluginVTableFn>(ffi::PLUGIN_VTABLE_SYMBOL)
                .map_err(|e: abi_stable::library::LibraryError| e.to_string())?;
            (*sym)()
        };

        // Register toolbar buttons from the dynamic module
        let buttons = (vtable.get_toolbar_buttons)();
        for btn in buttons.iter() {
            let registration = ToolbarButtonRegistration {
                plugin_id: plugin_id.to_string(),
                button_id: btn.button_id.to_string(),
                tooltip: btn.tooltip.to_string(),
                icon: IconDescriptor::Svg {
                    data: btn.icon_svg.to_string(),
                },
                action_id: btn.action_id.to_string(),
            };
            if let Err(e) = self.toolbar.register(registration) {
                warn!(
                    "Failed to register toolbar button from '{}': {e}",
                    plugin_id
                );
            }
        }

        self.loaded_vtables.insert(plugin_id.to_string(), vtable);
        Ok(())
    }

    /// Handle a toolbar button action by dispatching to the owning plugin.
    ///
    /// Returns window-open requests together with the plugin's vtable (if it
    /// was dynamically loaded) so the caller can wire UI callbacks.
    pub fn handle_action(
        &mut self,
        plugin_id: &str,
        action_id: &str,
    ) -> PluginResult<(Vec<WindowOpenRequest>, Option<PluginVTable>)> {
        // Check dynamically loaded plugins first
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable; // Copy – PluginVTable is Copy
            let response = (vt.on_action)(RString::from(action_id));
            let mut requests = Vec::new();
            if response.open_window {
                if let Some(desc) = self.descriptor(plugin_id) {
                    let ui_path = desc.manifest.resolve_entry_ui(&desc.root);
                    requests.push(WindowOpenRequest {
                        plugin_id: plugin_id.to_string(),
                        ui_path,
                        component: desc.manifest.entry_component.clone(),
                    });
                }
            }
            return Ok((requests, Some(vt)));
        }

        // Fall back to static registry
        let plugin = self
            .registry
            .get(plugin_id)
            .ok_or_else(|| {
                plugin_api::PluginError::Other(format!("unknown plugin '{plugin_id}'"))
            })?
            .clone();

        let plugin_root = self
            .descriptor(plugin_id)
            .map(|d| d.root.clone())
            .unwrap_or_default();

        let mut ctx = AppHostContext::new(&mut self.toolbar);
        plugin.on_action(action_id, &mut ctx, &plugin_root)?;
        Ok((ctx.take_window_requests(), None))
    }

    /// Find a plugin descriptor by id.
    pub fn descriptor(&self, id: &str) -> Option<&PluginDescriptor> {
        self.descriptors.iter().find(|d| d.manifest.id == id)
    }

    /// Get the plugin directory.
    pub fn plugin_dir(&self) -> &Path {
        &self.plugin_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc;

    fn create_example_plugin_dir(base: &Path) -> PathBuf {
        let plugin_dir = base.join("example_plugin");
        fs::create_dir_all(plugin_dir.join("ui")).unwrap();

        let manifest = r#"
id = "example_plugin"
name = "Example Plugin"
version = "0.1.0"
entry_ui = "ui/my_panel.slint"
entry_component = "MyPanel"

[icon]
kind = "svg"
data = "<svg/>"
"#;
        fs::write(plugin_dir.join("plugin.toml"), manifest).unwrap();
        fs::write(
            plugin_dir.join("ui/my_panel.slint"),
            "export component MyPanel inherits Window {}",
        )
        .unwrap();

        plugin_dir
    }

    #[test]
    fn manager_discovers_and_activates_example_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_dir(tmp.path());

        let mut mgr = PluginManager::new(tmp.path().to_path_buf());

        // Register the example plugin implementation
        let plugin = Arc::new(example_plugin::ExamplePlugin::new(
            example_plugin::ExamplePlugin::default_manifest(),
        ));
        mgr.registry.register(plugin).unwrap();

        // Discover from the temp directory
        mgr.discover();
        assert_eq!(mgr.descriptors.len(), 1);
        assert_eq!(mgr.descriptors[0].manifest.id, "example_plugin");

        // Activate
        mgr.activate_all().unwrap();
        assert_eq!(mgr.toolbar.len(), 1);
        assert_eq!(mgr.toolbar.buttons()[0].tooltip, "Example Plugin");
    }

    #[test]
    fn manager_handle_action_returns_window_requests() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_dir(tmp.path());

        let mut mgr = PluginManager::new(tmp.path().to_path_buf());
        let plugin = Arc::new(example_plugin::ExamplePlugin::new(
            example_plugin::ExamplePlugin::default_manifest(),
        ));
        mgr.registry.register(plugin).unwrap();
        mgr.discover();
        mgr.activate_all().unwrap();

        let requests = mgr.handle_action("example_plugin", "open_panel").unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].component, "MyPanel");
    }

    #[test]
    fn manager_nonexistent_plugin_dir_is_empty() {
        let mut mgr = PluginManager::new(PathBuf::from("/nonexistent/test/dir/123456"));
        mgr.discover();
        assert!(mgr.descriptors.is_empty());
        assert!(mgr.toolbar.is_empty());
    }

    #[test]
    fn manager_skips_unregistered_plugins() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_dir(tmp.path());

        // Don't register any plugin implementation
        let mut mgr = PluginManager::new(tmp.path().to_path_buf());
        mgr.discover();
        assert_eq!(mgr.descriptors.len(), 1);

        mgr.activate_all().unwrap();
        // No toolbar button registered because no implementation
        assert_eq!(mgr.toolbar.len(), 0);
    }

    #[test]
    fn manager_descriptor_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_dir(tmp.path());

        let mut mgr = PluginManager::new(tmp.path().to_path_buf());
        mgr.discover();

        assert!(mgr.descriptor("example_plugin").is_some());
        assert!(mgr.descriptor("nonexistent").is_none());
    }
}
