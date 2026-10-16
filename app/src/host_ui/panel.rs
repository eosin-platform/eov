//! The Plugins panel (§14.4, PLG-14): what is installed, where it came
//! from, whether it runs, and why not.

use super::UiHost;
use crate::PluginRowData;
use crate::plugin_runtime::PluginStatus;
use eov_plugin_api::protocol::NotifyLevel;
use slint::{Model, ModelRc, VecModel};
use std::collections::HashMap;
use std::rc::Rc;

/// How often the open Plugins panel shows the plugins' resource use again.
const USAGE_REFRESH: std::time::Duration = std::time::Duration::from_secs(2);

impl UiHost {
    /// Keeps the Plugins panel's resource use current while it is open.
    pub(crate) fn watch_plugin_usage(&self) {
        self.plugin_usage_timer
            .start(slint::TimerMode::Repeated, USAGE_REFRESH, || {
                super::with_host(|host| host.refresh_plugin_usage());
            });
    }

    /// Updates the usage line of each row in place (a new model would
    /// reset the list's scroll position).
    fn refresh_plugin_usage(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        if !ui.get_plugins_panel_visible() {
            self.plugin_usage_timer.stop();
            return;
        }
        let rows = ui.get_plugin_rows();
        let runtime = self.runtime.borrow();
        for index in 0..rows.row_count() {
            let Some(mut row) = rows.row_data(index) else {
                continue;
            };
            let usage: slint::SharedString =
                runtime.usage_summary(&row.id).unwrap_or_default().into();
            if row.usage != usage {
                row.usage = usage;
                rows.set_row_data(index, row);
            }
        }
    }

    pub(crate) fn refresh_plugin_panel(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        // Where each package came from and whether it is intact (local and
        // quick: no native code is loaded and nothing goes online).
        let inventory = crate::update::plugins::inventory(
            &self.plugin_dir,
            &crate::update::policy::current().plugin_dirs,
            &crate::update::compat::host_version(),
        )
        .map_err(|error| tracing::warn!("Cannot read the plugin directory: {error:#}"))
        .ok();
        let updates = self.plugin_updates.borrow();
        let runtime = self.runtime.borrow();
        let mut rows: Vec<PluginRowData> = runtime
            .slots()
            .iter()
            .map(|slot| {
                let (level, can_enable, can_disable) = match &slot.status {
                    PluginStatus::Active => (0, false, true),
                    PluginStatus::Disabled => (1, !self.safe_mode, false),
                    PluginStatus::Failed(_) => (2, false, false),
                    PluginStatus::Quarantined(_) => (3, true, false),
                };
                let package = inventory
                    .as_ref()
                    .and_then(|inventory| inventory.representative(&slot.info.id));
                let mut detail = Vec::new();
                if let Some(package) = package {
                    if package.modified {
                        detail.push(
                            "The package changed after it was installed; eov cannot tell whether it is the one that was installed.",
                        );
                    }
                    if package.system {
                        detail.push("Installed by your administrator.");
                    }
                }
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
                    origin: package
                        .map(|package| package.provenance.describe())
                        .unwrap_or_default()
                        .into(),
                    detail: detail.join(" ").into(),
                    update: updates.get(&slot.info.id).cloned().unwrap_or_default().into(),
                    usage: runtime
                        .usage_summary(&slot.info.id)
                        .unwrap_or_default()
                        .into(),
                    can_enable,
                    can_disable,
                    can_remove: package.is_some_and(|package| !package.system),
                }
            })
            .collect();
        // Packages that could not even be read as plugins.
        if let Some(inventory) = &inventory {
            for broken in &inventory.broken {
                let file = broken
                    .package_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                rows.push(PluginRowData {
                    id: file.clone().into(),
                    name: file.into(),
                    status: "Not a usable plugin package".into(),
                    status_level: 2,
                    source: broken.package_path.display().to_string().into(),
                    detail: broken.error.clone().into(),
                    can_remove: true,
                    ..Default::default()
                });
            }
        }
        let problems = runtime.slots().iter().any(|slot| {
            matches!(
                slot.status,
                PluginStatus::Failed(_) | PluginStatus::Quarantined(_)
            )
        }) || inventory
            .as_ref()
            .is_some_and(|inventory| !inventory.broken.is_empty());
        let active = runtime.slots().iter().filter(|s| s.is_active()).count();
        drop(runtime);
        drop(updates);
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

    /// Check for Updates: compares the installed plugins with the official
    /// catalog of this eov release, after the user agreed to go online.
    pub(crate) fn check_plugin_updates(&self) {
        if let Err(error) = crate::update::policy::current().ensure_network_allowed() {
            self.notify(
                NotifyLevel::Warning,
                "Checking for plugin updates is turned off",
                Some(&format!("{error:#}")),
                None,
            );
            return;
        }
        self.confirm(
            "Check for plugin updates?",
            "eov downloads the plugin catalog of this release from GitHub (github.com/eosin-platform/eov) and compares it with the installed plugins. Nothing else is sent, and nothing is installed without asking.",
            "Check",
            false,
            |agreed| {
                if agreed {
                    super::with_host(|host| host.start_plugin_update_check());
                }
            },
        );
    }

    fn start_plugin_update_check(&self) {
        if let Some(ui) = self.ui() {
            ui.set_plugins_checking(true);
            ui.set_plugins_update_note("".into());
        }
        let job = self.start_host_job("Checking for plugin updates", None);
        let failed_job = job.clone();
        let plugin_dir = self.plugin_dir.clone();
        let spawned = std::thread::Builder::new()
            .name("eov-plugin-updates".to_string())
            .spawn(move || {
                let result = crate::update::plugins::PluginContext::for_panel(&plugin_dir)
                    .and_then(|mut ctx| crate::update::plugins::check_for_updates(&mut ctx))
                    .map_err(|error| format!("{error:#}"));
                let _ = slint::invoke_from_event_loop(move || {
                    super::with_host(|host| host.plugin_update_check_done(&job, result));
                });
            });
        if let Err(error) = spawned {
            self.plugin_update_check_done(&failed_job, Err(error.to_string()));
        }
    }

    fn plugin_update_check_done(
        &self,
        job: &str,
        result: Result<Vec<crate::update::plugins::PluginListing>, String>,
    ) {
        self.finish_host_job(job);
        let Some(ui) = self.ui() else {
            return;
        };
        ui.set_plugins_checking(false);
        let note = match result {
            Ok(listings) => {
                let found: HashMap<String, String> = listings
                    .into_iter()
                    .filter(|row| row.status == "update available")
                    .filter_map(|row| Some((row.id, row.available?)))
                    .collect();
                let note = match found.len() {
                    0 => "All plugins are up to date.".to_string(),
                    1 => "An update is available.".to_string(),
                    n => format!("{n} updates are available."),
                };
                *self.plugin_updates.borrow_mut() = found;
                note
            }
            Err(error) => {
                self.notify(
                    NotifyLevel::Warning,
                    "Could not check for plugin updates",
                    Some(&error),
                    None,
                );
                "The check failed (see Messages).".to_string()
            }
        };
        ui.set_plugins_update_note(note.into());
        self.refresh_plugin_panel();
    }

    /// Update: installs the newer version found by Check for Updates. It is
    /// used from the next start; the running plugin is not touched.
    pub(crate) fn update_plugin(&self, plugin_id: &str) {
        let name = self.plugin_name(plugin_id);
        let Some(version) = self.plugin_updates.borrow().get(plugin_id).cloned() else {
            return;
        };
        let plugin_id = plugin_id.to_string();
        self.confirm(
            &format!("Update {name} to {version}?"),
            "eov downloads the package from where the plugin was installed from, checks it and installs it. The new version is used when eov restarts.",
            "Update",
            false,
            move |agreed| {
                if agreed {
                    super::with_host(|host| host.start_plugin_update(plugin_id, name, version));
                }
            },
        );
    }

    fn start_plugin_update(&self, plugin_id: String, name: String, version: String) {
        let job = self.start_host_job(&format!("Updating {name}"), None);
        let failed_job = job.clone();
        let plugin_dir = self.plugin_dir.clone();
        let spawned = std::thread::Builder::new()
            .name("eov-plugin-update".to_string())
            .spawn(move || {
                let result = crate::update::plugins::PluginContext::for_panel(&plugin_dir)
                    .and_then(|mut ctx| {
                        crate::update::plugins::update_from_panel(&mut ctx, &plugin_id)
                    })
                    .map_err(|error| format!("{error:#}"));
                let _ = slint::invoke_from_event_loop(move || {
                    super::with_host(|host| {
                        host.finish_host_job(&job);
                        match result {
                            Ok(()) => {
                                host.plugin_updates.borrow_mut().remove(&plugin_id);
                                host.notify(
                                    NotifyLevel::Info,
                                    &format!("{name} {version} is installed"),
                                    Some("Restart eov to use it."),
                                    None,
                                );
                            }
                            Err(error) => host.notify(
                                NotifyLevel::Error,
                                &format!("Could not update {name}"),
                                Some(&error),
                                None,
                            ),
                        }
                        host.refresh_plugin_panel();
                    });
                });
            });
        if let Err(error) = spawned {
            self.finish_host_job(&failed_job);
            self.notify(
                NotifyLevel::Error,
                "Could not update the plugin",
                Some(&error.to_string()),
                None,
            );
        }
    }

    /// Remove: uninstalls a plugin (or an unreadable package) after asking.
    /// A running plugin stops at once.
    pub(crate) fn remove_plugin(&self, query: &str) {
        let host_version = crate::update::compat::host_version();
        let removal =
            match crate::update::plugins::plan_removal(&self.plugin_dir, &host_version, query) {
                Ok(removal) => removal,
                Err(error) => {
                    self.notify(
                        NotifyLevel::Error,
                        "Cannot remove the plugin",
                        Some(&format!("{error:#}")),
                        None,
                    );
                    return;
                }
            };
        let name = removal
            .id
            .as_deref()
            .map(|id| self.plugin_name(id))
            .unwrap_or_else(|| query.to_string());
        let files = removal
            .paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        self.confirm(
            &format!("Remove {name}?"),
            &format!(
                "This deletes:\n{files}\n\nData the plugin saved (annotations, results, settings) is kept."
            ),
            "Remove",
            true,
            move |agreed| {
                if agreed {
                    super::with_host(|host| host.apply_plugin_removal(removal, name));
                }
            },
        );
    }

    fn apply_plugin_removal(&self, removal: crate::update::plugins::Removal, name: String) {
        if let Some(id) = &removal.id
            && self.runtime.borrow_mut().disable(id)
        {
            self.on_plugin_stopped(id);
        }
        match crate::update::plugins::apply_removal(&self.plugin_dir, &removal) {
            Ok(()) => self.notify(NotifyLevel::Info, &format!("Removed {name}"), None, None),
            Err(error) => self.notify(
                NotifyLevel::Error,
                &format!("Could not remove {name}"),
                Some(&format!("{error:#}")),
                None,
            ),
        }
        if let Some(id) = &removal.id {
            self.plugin_updates.borrow_mut().remove(id);
        }
        self.refresh_plugin_panel();
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
        crate::desktop::open_folder(self.plugin_dir.clone(), "Could not open the plugin folder");
    }
}
