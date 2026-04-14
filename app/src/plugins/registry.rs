//! Plugin registry — maps plugin ids to `Plugin` trait objects.
//!
//! In this first-pass architecture, plugins are registered statically at
//! compile time via `register_builtin_plugins()`. Discovery still uses the
//! on-disk manifest and plugin directory so external UI assets are loaded at
//! runtime. A future iteration can replace static registration with dynamic
//! library loading.

use plugin_api::{Plugin, PluginResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Global registry mapping plugin id → Plugin trait object.
///
/// Populated at startup, read-only afterwards.
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<dyn Plugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin by its manifest id.
    pub fn register(&mut self, plugin: Arc<dyn Plugin>) -> PluginResult<()> {
        let id = plugin.manifest().id.clone();
        if self.plugins.contains_key(&id) {
            return Err(plugin_api::PluginError::DuplicateId(id));
        }
        self.plugins.insert(id, plugin);
        Ok(())
    }

    /// Look up a plugin by id.
    pub fn get(&self, id: &str) -> Option<&Arc<dyn Plugin>> {
        self.plugins.get(id)
    }

    /// Iterate over all registered plugins.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<dyn Plugin>)> {
        self.plugins.iter()
    }

    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}
