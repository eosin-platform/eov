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
    /// effect immediately; enabling on the next start (a library cannot be
    /// safely reloaded in-process). Both are remembered in the plugin
    /// directory's registry, shared with `eov plugin enable/disable`.
    pub(crate) fn set_plugin_enabled(&self, plugin_id: &str, enabled: bool) {
        let name = self.plugin_name(plugin_id);
        if !enabled && self.runtime.borrow_mut().disable(plugin_id) {
            self.on_plugin_stopped(plugin_id);
        }
        self.refresh_plugin_panel();
        // The registry is locked against concurrent `eov plugin` commands,
        // which can take a while: save on a worker.
        let plugin_dir = self.plugin_dir.clone();
        let plugin_id = plugin_id.to_string();
        let spawned = std::thread::Builder::new()
            .name("eov-plugin-registry".to_string())
            .spawn(move || {
                let result =
                    crate::update::plugins::set_plugin_enabled(&plugin_dir, &plugin_id, enabled)
                        .map_err(|error| format!("{error:#}"));
                let _ = slint::invoke_from_event_loop(move || {
                    super::with_host(|host| match result {
                        Ok(_) if enabled => host.notify(
                            NotifyLevel::Info,
                            &format!("{name} will be enabled when eov restarts"),
                            None,
                            None,
                        ),
                        Ok(_) => host.notify(
                            NotifyLevel::Info,
                            &format!("{name} was disabled"),
                            None,
                            None,
                        ),
                        Err(error) => host.notify(
                            NotifyLevel::Error,
                            "Could not save plugin settings",
                            Some(&error),
                            None,
                        ),
                    });
                });
            });
        if let Err(error) = spawned {
            self.notify(
                NotifyLevel::Error,
                "Could not save plugin settings",
                Some(&error.to_string()),
                None,
            );
        }
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
