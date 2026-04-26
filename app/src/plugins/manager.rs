//! Plugin manager — orchestrates discovery, validation, and activation.
//!
//! The `PluginManager` is the main entry point for the host plugin system.
//! It discovers plugins on disk, matches them against the plugin registry,
//! validates their files, and activates them against the host API.
//!
//! Plugins are loaded in three ways:
//! 1. **Dynamic Rust** (preferred): the plugin directory contains a shared
//!    library (e.g. `libexample_plugin.so`) loaded via `abi_stable`.
//! 2. **Python**: the plugin directory contains a Python script that is
//!    spawned as a subprocess. The script uses slint-python for its UI.
//! 3. **Static** (fallback, used in tests): the plugin is compiled into the
//!    host binary and registered via `PluginRegistry`.

use crate::plugins::discovery;
use crate::plugins::host_context::AppHostContext;
use crate::plugins::registry::PluginRegistry;
use crate::plugins::toolbar::ToolbarManager;
use abi_stable::library::RawLibrary;
use abi_stable::std_types::RString;
use plugin_api::ffi::{self, PluginVTable};
use plugin_api::manifest::PluginLanguage;
use plugin_api::{
    HostToolMode, IconDescriptor, PluginDescriptor, PluginResult, PluginUndoRedoState,
    ToolbarButtonRegistration,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

fn viewport_snapshot_to_ffi(
    viewport: &plugin_api::ViewportSnapshot,
) -> plugin_api::ffi::ViewportSnapshotFFI {
    plugin_api::ffi::ViewportSnapshotFFI {
        pane_index: viewport.pane_index,
        file_id: viewport.file_id,
        file_path: viewport.file_path.clone().into(),
        filename: viewport.filename.clone().into(),
        center_x: viewport.center_x,
        center_y: viewport.center_y,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        image_width: viewport.image_width,
        image_height: viewport.image_height,
        bounds_left: viewport.bounds_left,
        bounds_top: viewport.bounds_top,
        bounds_right: viewport.bounds_right,
        bounds_bottom: viewport.bounds_bottom,
    }
}

fn vertices_to_ffi(
    vertices: &[crate::state::ImagePoint],
) -> abi_stable::std_types::RVec<plugin_api::ffi::ViewportOverlayVertexFFI> {
    vertices
        .iter()
        .map(|vertex| plugin_api::ffi::ViewportOverlayVertexFFI {
            x_level0: vertex.x,
            y_level0: vertex.y,
        })
        .collect()
}

/// Outcome of handling a toolbar button action.
pub enum ActionOutcome {
    /// Rust plugin: spawn `eov plugin-window <root>` as a subprocess.
    RustPluginWindow { plugin_root: PathBuf },
    /// Python plugin: spawn the entry script as a subprocess.
    PythonSpawn {
        script_path: PathBuf,
        plugin_root: PathBuf,
    },
    /// Static plugin handled the action internally.
    Handled,
}

/// Central manager for the plugin lifecycle.
pub struct PluginManager {
    pub registry: PluginRegistry,
    pub toolbar: ToolbarManager,
    pub hud_toolbar: ToolbarManager,
    pub undo_redo_states: HashMap<String, PluginUndoRedoState>,
    pub undo_redo_order: Vec<String>,
    /// Descriptors for all successfully discovered plugins on disk.
    pub descriptors: Vec<PluginDescriptor>,
    /// Older `.eop` files skipped because a newer version of the same plugin exists.
    old_plugin_package_files: Vec<PathBuf>,
    /// Plugin root directory.
    plugin_dir: PathBuf,
    /// Dynamically loaded plugin vtables, keyed by plugin id.
    loaded_vtables: HashMap<String, PluginVTable>,
    /// Plugin ids that are Python plugins (spawned as subprocesses).
    python_plugins: HashSet<String>,
    /// Python plugin ids that have already been spawned at least once.
    pub spawned_python_plugins: HashSet<String>,
}

impl PluginManager {
    pub fn new(plugin_dir: PathBuf) -> Self {
        Self {
            registry: PluginRegistry::new(),
            toolbar: ToolbarManager::new(),
            hud_toolbar: ToolbarManager::new(),
            undo_redo_states: HashMap::new(),
            undo_redo_order: Vec::new(),
            descriptors: Vec::new(),
            old_plugin_package_files: Vec::new(),
            plugin_dir,
            loaded_vtables: HashMap::new(),
            python_plugins: HashSet::new(),
            spawned_python_plugins: HashSet::new(),
        }
    }

    /// Discover plugins from the plugin directory.
    pub fn discover(&mut self) {
        info!("Discovering plugins in {}", self.plugin_dir.display());
        let report = discovery::discover_plugins(&self.plugin_dir);
        self.descriptors = report.descriptors;
        self.old_plugin_package_files = report.old_plugin_files;
        info!("Discovered {} plugin(s)", self.descriptors.len());
    }

    pub fn old_plugin_package_files(&self) -> &[PathBuf] {
        &self.old_plugin_package_files
    }

    /// Activate all discovered plugins.
    ///
    /// For each plugin, the language field determines the activation strategy:
    /// - **Rust**: tries dynamic loading (shared library), falls back to static.
    /// - **Python**: registers toolbar buttons from the manifest and records the
    ///   plugin id so that actions spawn its entry script as a subprocess.
    pub fn activate_all(&mut self) -> PluginResult<()> {
        let descriptors = self.descriptors.clone();
        for desc in &descriptors {
            // Handle Python plugins
            if desc.manifest.language == PluginLanguage::Python {
                self.activate_python_plugin(desc);
                continue;
            }

            // Try dynamic loading first (Rust plugins)
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
            let mut ctx = AppHostContext::new(
                &mut self.toolbar,
                &mut self.hud_toolbar,
                &mut self.undo_redo_states,
                &mut self.undo_redo_order,
            );
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
                tool_mode: btn.tool_mode.into_option().map(host_tool_mode_from_ffi),
                hotkey: btn.hotkey.clone().into_option().map(|key| key.to_string()),
                active: false,
            };
            if let Err(e) = self.toolbar.register(registration) {
                warn!(
                    "Failed to register toolbar button from '{}': {e}",
                    plugin_id
                );
            }
        }

        let hud_buttons = (vtable.get_hud_toolbar_buttons)();
        for btn in hud_buttons.iter() {
            let registration = ToolbarButtonRegistration {
                plugin_id: plugin_id.to_string(),
                button_id: btn.button_id.to_string(),
                tooltip: btn.tooltip.to_string(),
                icon: IconDescriptor::Svg {
                    data: btn.icon_svg.to_string(),
                },
                action_id: btn.action_id.to_string(),
                tool_mode: None,
                hotkey: None,
                active: false,
            };
            if let Err(e) = self.hud_toolbar.register(registration) {
                warn!(
                    "Failed to register HUD toolbar button from '{}': {e}",
                    plugin_id
                );
            }
        }

        self.loaded_vtables.insert(plugin_id.to_string(), vtable);
        Ok(())
    }

    /// Handle a toolbar button action by dispatching to the owning plugin.
    pub fn handle_action(
        &mut self,
        plugin_id: &str,
        action_id: &str,
    ) -> PluginResult<ActionOutcome> {
        // Python plugins – spawn the entry script as a subprocess.
        if self.python_plugins.contains(plugin_id) {
            if let Some(desc) = self.descriptor(plugin_id)
                && let Some(script) = &desc.manifest.entry_script
            {
                let script_path = desc.root.join(script);
                return Ok(ActionOutcome::PythonSpawn {
                    script_path,
                    plugin_root: desc.root.clone(),
                });
            }
            return Err(plugin_api::PluginError::Other(format!(
                "Python plugin '{plugin_id}' has no entry_script"
            )));
        }

        // Check dynamically loaded Rust plugins.
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable; // Copy – PluginVTable is Copy
            let response = (vt.on_action)(RString::from(action_id));
            if response.open_window
                && let Some(desc) = self.descriptor(plugin_id)
            {
                return Ok(ActionOutcome::RustPluginWindow {
                    plugin_root: desc.root.clone(),
                });
            }
            return Ok(ActionOutcome::Handled);
        }

        // Fall back to static registry.
        let plugin = self
            .registry
            .get(plugin_id)
            .ok_or_else(|| plugin_api::PluginError::Other(format!("unknown plugin '{plugin_id}'")))?
            .clone();

        let plugin_root = self
            .descriptor(plugin_id)
            .map(|d| d.root.clone())
            .unwrap_or_default();

        let mut ctx = AppHostContext::new(
            &mut self.toolbar,
            &mut self.hud_toolbar,
            &mut self.undo_redo_states,
            &mut self.undo_redo_order,
        );
        plugin.on_action(action_id, &mut ctx, &plugin_root)?;
        Ok(ActionOutcome::Handled)
    }

    pub fn handle_undo(&mut self, plugin_id: &str) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let response = (vt.on_undo)();
            if response.open_window
                && let Some(desc) = self.descriptor(plugin_id)
            {
                return Ok(ActionOutcome::RustPluginWindow {
                    plugin_root: desc.root.clone(),
                });
            }
            return Ok(ActionOutcome::Handled);
        }

        let plugin = self
            .registry
            .get(plugin_id)
            .ok_or_else(|| plugin_api::PluginError::Other(format!("unknown plugin '{plugin_id}'")))?
            .clone();
        let plugin_root = self
            .descriptor(plugin_id)
            .map(|d| d.root.clone())
            .unwrap_or_default();
        let mut ctx = AppHostContext::new(
            &mut self.toolbar,
            &mut self.hud_toolbar,
            &mut self.undo_redo_states,
            &mut self.undo_redo_order,
        );
        plugin.on_undo(&mut ctx, &plugin_root)?;
        Ok(ActionOutcome::Handled)
    }

    pub fn handle_redo(&mut self, plugin_id: &str) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let response = (vt.on_redo)();
            if response.open_window
                && let Some(desc) = self.descriptor(plugin_id)
            {
                return Ok(ActionOutcome::RustPluginWindow {
                    plugin_root: desc.root.clone(),
                });
            }
            return Ok(ActionOutcome::Handled);
        }

        let plugin = self
            .registry
            .get(plugin_id)
            .ok_or_else(|| plugin_api::PluginError::Other(format!("unknown plugin '{plugin_id}'")))?
            .clone();
        let plugin_root = self
            .descriptor(plugin_id)
            .map(|d| d.root.clone())
            .unwrap_or_default();
        let mut ctx = AppHostContext::new(
            &mut self.toolbar,
            &mut self.hud_toolbar,
            &mut self.undo_redo_states,
            &mut self.undo_redo_order,
        );
        plugin.on_redo(&mut ctx, &plugin_root)?;
        Ok(ActionOutcome::Handled)
    }

    /// Handle a viewport-scoped HUD toolbar action.
    pub fn handle_hud_action(
        &mut self,
        plugin_id: &str,
        action_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let viewport = plugin_api::ffi::ViewportSnapshotFFI {
                pane_index: viewport.pane_index,
                file_id: viewport.file_id,
                file_path: viewport.file_path.clone().into(),
                filename: viewport.filename.clone().into(),
                center_x: viewport.center_x,
                center_y: viewport.center_y,
                zoom: viewport.zoom,
                width: viewport.width,
                height: viewport.height,
                image_width: viewport.image_width,
                image_height: viewport.image_height,
                bounds_left: viewport.bounds_left,
                bounds_top: viewport.bounds_top,
                bounds_right: viewport.bounds_right,
                bounds_bottom: viewport.bounds_bottom,
            };
            let _ = (vt.on_hud_action)(RString::from(action_id), viewport);
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown HUD plugin '{plugin_id}'"
        )))
    }

    pub fn handle_viewport_context_menu_action(
        &mut self,
        plugin_id: &str,
        item_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let viewport = plugin_api::ffi::ViewportSnapshotFFI {
                pane_index: viewport.pane_index,
                file_id: viewport.file_id,
                file_path: viewport.file_path.clone().into(),
                filename: viewport.filename.clone().into(),
                center_x: viewport.center_x,
                center_y: viewport.center_y,
                zoom: viewport.zoom,
                width: viewport.width,
                height: viewport.height,
                image_width: viewport.image_width,
                image_height: viewport.image_height,
                bounds_left: viewport.bounds_left,
                bounds_top: viewport.bounds_top,
                bounds_right: viewport.bounds_right,
                bounds_bottom: viewport.bounds_bottom,
            };
            let _ = (vt.on_viewport_context_menu_action)(RString::from(item_id), viewport);
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown viewport menu plugin '{plugin_id}'"
        )))
    }

    pub fn handle_point_annotation_placed(
        &mut self,
        plugin_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
        x_level0: f64,
        y_level0: f64,
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let viewport = viewport_snapshot_to_ffi(viewport);
            (vt.on_point_annotation_placed)(viewport, x_level0, y_level0);
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown point annotation plugin '{plugin_id}'"
        )))
    }

    pub fn handle_point_annotation_moved(
        &mut self,
        plugin_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
        annotation_id: &str,
        x_level0: f64,
        y_level0: f64,
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            let viewport = viewport_snapshot_to_ffi(viewport);
            (vt.on_point_annotation_moved)(viewport, annotation_id.into(), x_level0, y_level0);
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown point annotation plugin '{plugin_id}'"
        )))
    }

    pub fn handle_polygon_annotation_placed(
        &mut self,
        plugin_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
        vertices: &[crate::state::ImagePoint],
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            (vt.on_polygon_annotation_placed)(
                viewport_snapshot_to_ffi(viewport),
                vertices_to_ffi(vertices),
            );
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown polygon annotation plugin '{plugin_id}'"
        )))
    }

    pub fn handle_polygon_annotation_moved(
        &mut self,
        plugin_id: &str,
        viewport: &plugin_api::ViewportSnapshot,
        annotation_id: &str,
        vertices: &[crate::state::ImagePoint],
    ) -> PluginResult<ActionOutcome> {
        if let Some(vtable) = self.loaded_vtables.get(plugin_id) {
            let vt = *vtable;
            (vt.on_polygon_annotation_moved)(
                viewport_snapshot_to_ffi(viewport),
                annotation_id.into(),
                vertices_to_ffi(vertices),
            );
            return Ok(ActionOutcome::Handled);
        }

        Err(plugin_api::PluginError::Other(format!(
            "unknown polygon annotation plugin '{plugin_id}'"
        )))
    }

    /// Activate a Python plugin: register its manifest-declared toolbar buttons
    /// and record it for subprocess spawning.
    fn activate_python_plugin(&mut self, desc: &PluginDescriptor) {
        if let Err(e) = desc.manifest.validate_files(&desc.root) {
            warn!(
                "Python plugin '{}' has missing files, skipping: {e}",
                desc.manifest.id
            );
            return;
        }

        // Register toolbar buttons declared in the manifest.
        let top_icon_svg = match &desc.manifest.icon {
            Some(IconDescriptor::Svg { data }) => Some(data.clone()),
            _ => None,
        };

        for btn in &desc.manifest.toolbar_buttons {
            let svg = btn
                .icon_svg
                .as_deref()
                .or(top_icon_svg.as_deref())
                .unwrap_or("")
                .to_string();
            let registration = ToolbarButtonRegistration {
                plugin_id: desc.manifest.id.clone(),
                button_id: btn.button_id.clone(),
                tooltip: btn.tooltip.clone(),
                icon: IconDescriptor::Svg { data: svg },
                action_id: btn.action_id.clone(),
                tool_mode: None,
                hotkey: None,
                active: false,
            };
            if let Err(e) = self.toolbar.register(registration) {
                warn!(
                    "Failed to register toolbar button for Python plugin '{}': {e}",
                    desc.manifest.id
                );
            }
        }

        self.python_plugins.insert(desc.manifest.id.clone());
        info!(
            "Activated Python plugin '{}' from {}",
            desc.manifest.id,
            desc.root.display()
        );
    }

    /// Find a plugin descriptor by id.
    pub fn descriptor(&self, id: &str) -> Option<&PluginDescriptor> {
        self.descriptors.iter().find(|d| d.manifest.id == id)
    }

    /// Returns an iterator over all dynamically loaded plugin vtables.
    /// Used to register FFI viewport filters into the filter chain.
    pub fn loaded_vtables(&self) -> impl Iterator<Item = (&str, &PluginVTable)> {
        self.loaded_vtables.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Poll all loaded FFI plugins for updated filter enabled states and
    /// sync them into the shared filter chain.
    pub fn sync_filter_states(&self, filter_chain: &crate::viewport_filter::SharedFilterChain) {
        let mut chain = filter_chain.write();
        for vtable in self.loaded_vtables.values() {
            let filters = (vtable.get_viewport_filters)();
            for f in filters.iter() {
                chain.set_enabled(&f.filter_id, f.enabled);
            }
        }
    }
}

fn host_tool_mode_from_ffi(mode: plugin_api::ffi::HostToolModeFFI) -> HostToolMode {
    match mode {
        plugin_api::ffi::HostToolModeFFI::Navigate => HostToolMode::Navigate,
        plugin_api::ffi::HostToolModeFFI::RegionOfInterest => HostToolMode::RegionOfInterest,
        plugin_api::ffi::HostToolModeFFI::MeasureDistance => HostToolMode::MeasureDistance,
        plugin_api::ffi::HostToolModeFFI::PointAnnotation => HostToolMode::PointAnnotation,
        plugin_api::ffi::HostToolModeFFI::PolygonAnnotation => HostToolMode::PolygonAnnotation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppWindow;
    use crate::plugin_host::{build_host_api, init_ui_runtime};
    use crate::state::AppState;
    use common::TileCache;
    use parking_lot::RwLock;
    use slint::Timer;
    use std::cell::RefCell;
    use std::fs;
    use std::fs::File;
    use std::rc::Rc;
    use std::sync::Arc;

    fn append_tree(builder: &mut tar::Builder<File>, source_root: &Path, current: &Path) {
        for entry in fs::read_dir(current).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path.strip_prefix(source_root).unwrap();
            if path.is_dir() {
                builder.append_dir(relative, &path).unwrap();
                append_tree(builder, source_root, &path);
            } else {
                builder.append_path_with_name(&path, relative).unwrap();
            }
        }
    }

    fn create_example_plugin_package(base: &Path) -> PathBuf {
        let source_dir = base.join("example_plugin_src");
        fs::create_dir_all(source_dir.join("ui")).unwrap();

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
        fs::write(source_dir.join("plugin.toml"), manifest).unwrap();
        fs::write(
            source_dir.join("ui/my_panel.slint"),
            "export component MyPanel inherits Window {}",
        )
        .unwrap();

        let package_path = base.join("example_plugin.eop");
        let file = File::create(&package_path).unwrap();
        let mut builder = tar::Builder::new(file);
        append_tree(&mut builder, &source_dir, &source_dir);
        builder.finish().unwrap();

        package_path
    }

    #[test]
    fn manager_discovers_and_activates_example_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_package(tmp.path());

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
        create_example_plugin_package(tmp.path());

        let mut mgr = PluginManager::new(tmp.path().to_path_buf());
        let plugin = Arc::new(example_plugin::ExamplePlugin::new(
            example_plugin::ExamplePlugin::default_manifest(),
        ));
        mgr.registry.register(plugin).unwrap();
        mgr.discover();
        mgr.activate_all().unwrap();

        let result = mgr.handle_action("example_plugin", "open_panel").unwrap();
        assert!(matches!(result, ActionOutcome::Handled));
    }

    #[test]
    fn manager_nonexistent_plugin_dir_is_empty() {
        let mut mgr = PluginManager::new(PathBuf::from("/nonexistent/test/dir/123456"));
        mgr.discover();
        assert!(mgr.descriptors.is_empty());
        assert!(mgr.old_plugin_package_files().is_empty());
        assert!(mgr.toolbar.is_empty());
    }

    #[test]
    fn manager_skips_unregistered_plugins() {
        let tmp = tempfile::tempdir().unwrap();
        create_example_plugin_package(tmp.path());

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
        create_example_plugin_package(tmp.path());

        let mut mgr = PluginManager::new(tmp.path().to_path_buf());
        mgr.discover();

        assert!(mgr.descriptor("example_plugin").is_some());
        assert!(mgr.descriptor("nonexistent").is_none());
    }

    #[test]
    fn manager_handle_action_opens_eovae_sidebar_with_real_vtable() {
        let ui = AppWindow::new().unwrap();
        let state = Arc::new(RwLock::new(AppState::new()));
        let tile_cache = Arc::new(TileCache::new());
        let render_timer = Rc::new(Timer::default());
        init_ui_runtime(&ui, &state, &tile_cache, &render_timer);

        let plugin_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../plugins/eovae");
        let vtable = eovae::eov_get_plugin_vtable();
        let host_api = build_host_api("eovae", &plugin_root, &state, vtable);
        (vtable.set_host_api)(host_api);

        state.write().local_plugin_buttons = vec![ToolbarButtonRegistration {
            plugin_id: "eovae".to_string(),
            button_id: "toggle_eovae".to_string(),
            tooltip: "VAE".to_string(),
            icon: IconDescriptor::Svg {
                data: "<svg xmlns=\"http://www.w3.org/2000/svg\"/>".to_string(),
            },
            action_id: "toggle_eovae".to_string(),
            tool_mode: None,
            hotkey: None,
            active: false,
        }];

        let manager = RefCell::new(PluginManager::new(PathBuf::new()));
        manager
            .borrow_mut()
            .loaded_vtables
            .insert("eovae".to_string(), vtable);

        let result = manager
            .borrow_mut()
            .handle_action("eovae", "toggle_eovae")
            .unwrap();
        assert!(matches!(result, ActionOutcome::Handled));

        let state = state.read();
        let active_sidebar = state
            .active_sidebar
            .as_ref()
            .expect("sidebar should be active");
        assert_eq!(active_sidebar.plugin_id, "eovae");
        assert_eq!(active_sidebar.component, "EovaeSidebar");
    }
}
