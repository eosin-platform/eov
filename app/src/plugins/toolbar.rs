//! Toolbar management for plugin-contributed buttons.
//!
//! The `ToolbarManager` holds all registered plugin toolbar buttons in order,
//! ensuring they are appended after built-in toolbar items. It is independent
//! of Slint so it can be tested in pure Rust.

use plugin_api::{PluginError, PluginResult, ToolbarButtonRegistration};
use std::collections::HashSet;

/// Manages the ordered list of plugin toolbar buttons.
#[derive(Debug, Default)]
pub struct ToolbarManager {
    buttons: Vec<ToolbarButtonRegistration>,
    /// Set of `"{plugin_id}:{button_id}"` to detect duplicates.
    registered_ids: HashSet<String>,
}

impl ToolbarManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin toolbar button. Returns an error if the composite
    /// key `plugin_id:button_id` is already registered.
    pub fn register(&mut self, button: ToolbarButtonRegistration) -> PluginResult<()> {
        let key = format!("{}:{}", button.plugin_id, button.button_id);
        if !self.registered_ids.insert(key) {
            return Err(PluginError::DuplicateButtonId(format!(
                "{}:{}",
                button.plugin_id, button.button_id
            )));
        }
        self.buttons.push(button);
        Ok(())
    }

    /// All registered buttons in insertion order.
    pub fn buttons(&self) -> &[ToolbarButtonRegistration] {
        &self.buttons
    }

    /// Number of registered buttons.
    pub fn len(&self) -> usize {
        self.buttons.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.buttons.is_empty()
    }

    /// Find a button by its composite key.
    #[cfg(test)]
    pub fn find_button(
        &self,
        plugin_id: &str,
        button_id: &str,
    ) -> Option<&ToolbarButtonRegistration> {
        self.buttons
            .iter()
            .find(|b| b.plugin_id == plugin_id && b.button_id == button_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plugin_api::IconDescriptor;

    fn make_button(plugin_id: &str, button_id: &str) -> ToolbarButtonRegistration {
        ToolbarButtonRegistration {
            plugin_id: plugin_id.into(),
            button_id: button_id.into(),
            tooltip: format!("{plugin_id} - {button_id}"),
            icon: IconDescriptor::Svg {
                data: "<svg/>".into(),
            },
            action_id: format!("{plugin_id}.{button_id}.action"),
            tool_mode: None,
            active: false,
        }
    }

    #[test]
    fn register_and_retrieve() {
        let mut mgr = ToolbarManager::new();
        mgr.register(make_button("p1", "btn_a")).unwrap();
        mgr.register(make_button("p2", "btn_b")).unwrap();

        assert_eq!(mgr.len(), 2);
        assert_eq!(mgr.buttons()[0].plugin_id, "p1");
        assert_eq!(mgr.buttons()[1].plugin_id, "p2");
    }

    #[test]
    fn duplicate_button_id_rejected() {
        let mut mgr = ToolbarManager::new();
        mgr.register(make_button("p1", "btn")).unwrap();
        let err = mgr.register(make_button("p1", "btn")).unwrap_err();
        assert!(err.to_string().contains("p1:btn"));
    }

    #[test]
    fn same_button_id_different_plugins_ok() {
        let mut mgr = ToolbarManager::new();
        mgr.register(make_button("p1", "btn")).unwrap();
        mgr.register(make_button("p2", "btn")).unwrap();
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn buttons_appended_in_order() {
        let mut mgr = ToolbarManager::new();
        for i in 0..5 {
            mgr.register(make_button(&format!("p{i}"), "main")).unwrap();
        }
        let ids: Vec<&str> = mgr.buttons().iter().map(|b| b.plugin_id.as_str()).collect();
        assert_eq!(ids, vec!["p0", "p1", "p2", "p3", "p4"]);
    }

    #[test]
    fn metadata_preserved() {
        let mut mgr = ToolbarManager::new();
        let btn = ToolbarButtonRegistration {
            plugin_id: "test".into(),
            button_id: "my_btn".into(),
            tooltip: "My Tooltip".into(),
            icon: IconDescriptor::Svg {
                data: "<svg>icon</svg>".into(),
            },
            action_id: "open_panel".into(),
            tool_mode: None,
            active: false,
        };
        mgr.register(btn).unwrap();

        let found = mgr.find_button("test", "my_btn").unwrap();
        assert_eq!(found.tooltip, "My Tooltip");
        assert_eq!(found.action_id, "open_panel");
        assert_eq!(
            found.icon,
            IconDescriptor::Svg {
                data: "<svg>icon</svg>".into()
            }
        );
    }
}
