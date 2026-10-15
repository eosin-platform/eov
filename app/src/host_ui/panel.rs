//! The Plugins panel (§14.4, PLG-14): what is installed, where it came
//! from, whether it runs, and why not.

use super::UiHost;
use crate::PluginRowData;
use crate::plugin_runtime::PluginStatus;
use eov_plugin_api::protocol::NotifyLevel;
use slint::{ModelRc, VecModel};
use std::rc::Rc;

impl UiHost {
    pub(crate) fn refresh_plugin_panel(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let runtime = self.runtime.borrow();
        let rows: Vec<PluginRowData> = runtime
            .slots()
            .iter()
            .map(|slot| {
                let (level, can_enable, can_disable) = match &slot.status {
                    PluginStatus::Active => (0, false, true),
                    PluginStatus::Disabled => (1, !self.safe_mode, false),
                    PluginStatus::Failed(_) => (2, false, false),
                    PluginStatus::Quarantined(_) => (3, true, false),
                };
                PluginRowData {
                    id: slot.info.id.clone().into(),
                    name: slot.info.name.clone().into(),
                    version: slot.info.version.clone().into(),
                    description: slot.info.description.clone().unwrap_or_default().into(),
                    status: slot.status.label().into(),
                    status_level: level,
                    api_version: slot.api_version.clone().unwrap_or_default().into(),
                    source: slot
                        .info
                        .repository
                        .clone()
                        .unwrap_or_else(|| slot.info.root.display().to_string())
                        .into(),
                    can_enable,
                    can_disable,
                }
            })
            .collect();
        let problems = runtime.slots().iter().any(|slot| {
            matches!(
                slot.status,
                PluginStatus::Failed(_) | PluginStatus::Quarantined(_)
            )
        });
        let active = runtime.slots().iter().filter(|s| s.is_active()).count();
        drop(runtime);
        ui.set_plugin_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
        ui.set_plugin_count(active as i32);
        ui.set_plugin_problems(problems);
        ui.set_plugin_dir_text(self.plugin_dir.display().to_string().into());
        ui.set_plugins_panel_note(if self.safe_mode {
            "Safe mode: plugins are not loaded in this session.".into()
        } else {
            "".into()
        });
    }

    /// Turns a plugin on or off from the Plugins panel. Disabling takes
    /// effect immediately and is remembered; enabling takes effect on the
    /// next start (a library cannot be safely reloaded in-process).
    pub(crate) fn set_plugin_enabled(&self, plugin_id: &str, enabled: bool) {
        let name = self.plugin_name(plugin_id);
        if enabled {
            if let Err(error) = crate::config::set_plugin_disabled(plugin_id, false) {
                self.notify(
                    NotifyLevel::Error,
                    "Could not save plugin settings",
                    Some(&error.to_string()),
                    None,
                );
                return;
            }
            self.notify(
                NotifyLevel::Info,
                &format!("{name} will be enabled when eov restarts"),
                None,
                None,
            );
        } else {
            if self.runtime.borrow_mut().disable(plugin_id) {
                self.on_plugin_stopped(plugin_id);
            }
            if let Err(error) = crate::config::set_plugin_disabled(plugin_id, true) {
                self.notify(
                    NotifyLevel::Error,
                    "Could not save plugin settings",
                    Some(&error.to_string()),
                    None,
                );
            }
            self.notify(
                NotifyLevel::Info,
                &format!("{name} was disabled"),
                None,
                None,
            );
        }
        self.refresh_plugin_panel();
    }

    pub(crate) fn open_plugin_folder(&self) {
        let _ = std::fs::create_dir_all(&self.plugin_dir);
        if let Err(error) = open::that(&self.plugin_dir) {
            self.notify(
                NotifyLevel::Error,
                "Could not open the plugin folder",
                Some(&error.to_string()),
                None,
            );
        }
    }
}
