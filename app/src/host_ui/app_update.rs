//! Updating eov from the window (AUDIT §14.4). Check for Updates on the
//! Home screen plans an update the way `eov update` does, shows the plan,
//! and applies it once the user agrees. Every check asks before going
//! online, and the administrator policy applies as on the command line.
//!
//! One worker thread owns the update: it plans, waits for the user's answer
//! and executes, so the plan never leaves the thread that made it.

use super::{NotifyLevel, UiHost, with_host};
use crate::update::registry::shown_error;
use crate::update::{PlanSummary, UpdateContext, UpdateOutcome, UpdateRequest};
use eov_plugin_api::i18n::{isolate, tr};
use std::sync::mpsc;

/// What an update did to one plugin (`PluginOutcome::status`, an id), in
/// the language of the interface.
fn plugin_outcome_text(status: &str) -> String {
    match status {
        "updated" => tr!("Plugin update" => "updated"),
        // Translators: the update helper installs it after eov exits.
        "staged" => tr!("Plugin update" => "staged"),
        "skipped" => tr!("Plugin update" => "skipped"),
        "failed" => tr!("Plugin update" => "failed"),
        other => other.to_string(),
    }
}

/// An update check or update in progress.
pub(super) struct AppUpdate {
    /// Answers the plan the worker is waiting on (`None` once answered).
    answer: Option<mpsc::Sender<bool>>,
}

impl UiHost {
    /// Check for Updates: asks before contacting GitHub.
    pub(crate) fn check_for_app_update(&self) {
        if self.app_update.borrow().is_some() {
            self.notify(
                NotifyLevel::Info,
                &tr!("An update check is already running"),
                None,
                None,
            );
            return;
        }
        if let Err(error) = crate::update::policy::current().ensure_network_allowed() {
            self.notify(
                NotifyLevel::Warning,
                &tr!("Checking for updates is turned off"),
                Some(&format!("{error:#}")),
                None,
            );
            return;
        }
        self.confirm(
            &tr!("Check for eov updates?"),
            &tr!(
                "eov downloads the release information of the latest eov from GitHub (github.com/eosin-platform/eov) and compares it with this installation and its plugins. Nothing else is sent, and nothing is installed without asking."
            ),
            &tr!("Button" => "Check"),
            false,
            |agreed| {
                if agreed {
                    with_host(|host| host.start_app_update_check());
                }
            },
        );
    }

    fn start_app_update_check(&self) {
        if self.app_update.borrow().is_some() {
            return;
        }
        let job = self.start_host_job(&tr!("Checking for eov updates"), None);
        let failed_job = job.clone();
        let (answer, answers) = mpsc::channel::<bool>();
        let plugin_dir = self.plugin_dir.clone();
        let spawned = std::thread::Builder::new()
            .name("eov-app-update".to_string())
            .spawn(move || {
                let planned = UpdateContext::for_window(&plugin_dir).and_then(|mut ctx| {
                    let request = UpdateRequest {
                        allow_network: true,
                        ..UpdateRequest::default()
                    };
                    let planned = crate::update::plan_update(&mut ctx, &request)?;
                    Ok((ctx, planned))
                });
                let (mut ctx, planned) = match planned {
                    Ok(planned) => planned,
                    Err(error) => {
                        let _ = slint::invoke_from_event_loop(move || {
                            // Worded in the interface's language of the
                            // moment.
                            let error = shown_error(&error);
                            with_host(|host| host.app_update_check_failed(&job, &error));
                        });
                        return;
                    }
                };
                let summary = planned.plan.summary();
                let _ = slint::invoke_from_event_loop(move || {
                    with_host(|host| host.app_update_planned(&job, summary));
                });
                // A dropped sender (the window closed, or no question was
                // asked) counts as "no".
                if answers.recv() != Ok(true) {
                    return;
                }
                let result = crate::update::execute_update(&mut ctx, planned);
                let _ = slint::invoke_from_event_loop(move || {
                    let result = result.map_err(|error| shown_error(&error));
                    with_host(|host| host.app_update_finished(result));
                });
            });
        match spawned {
            Ok(_) => {
                *self.app_update.borrow_mut() = Some(AppUpdate {
                    answer: Some(answer),
                })
            }
            Err(error) => self.app_update_check_failed(&failed_job, &error.to_string()),
        }
    }

    fn app_update_check_failed(&self, job: &str, error: &str) {
        self.finish_host_job(job);
        self.app_update.borrow_mut().take();
        self.notify(
            NotifyLevel::Warning,
            &tr!("Could not check for eov updates"),
            Some(error),
            None,
        );
    }

    fn app_update_planned(&self, job: &str, summary: PlanSummary) {
        self.finish_host_job(job);
        if summary.noop {
            self.app_update.borrow_mut().take();
            let (title, detail) = if summary.app_available {
                // Newer, but this installation is updated another way (a
                // package manager, a policy): the plan's notes say how.
                (
                    tr!("eov {} is available", isolate(&summary.target)),
                    summary.text,
                )
            } else {
                (
                    tr!(
                        "eov {} and its plugins are up to date",
                        isolate(&summary.current)
                    ),
                    summary.text,
                )
            };
            self.notify(NotifyLevel::Info, &title, Some(&detail), None);
            return;
        }
        let title = if summary.changes_app {
            tr!("Update eov to {}?", isolate(&summary.target))
        } else {
            tr!("Update a plugin?" | "Update {n} plugins?" % summary.plugin_changes)
        };
        // What happens next, after the plan (one paragraph).
        let next = if summary.exits_for_helper {
            if crate::config::load_restore_session() {
                tr!(
                    "eov closes, the update helper replaces the application, and eov starts again. Open slides are reopened."
                )
            } else {
                tr!("eov closes, the update helper replaces the application, and eov starts again.")
            }
        } else if summary.changes_app {
            tr!("The new version is used when eov restarts.")
        } else {
            tr!("Updated plugins are used when eov restarts.")
        };
        let message = format!("{}\n\n{next}", summary.text);
        let label = if summary.exits_for_helper {
            tr!("Update and Restart")
        } else {
            tr!("Button" => "Update")
        };
        self.confirm(&title, &message, &label, false, |agreed| {
            with_host(|host| host.answer_app_update(agreed));
        });
    }

    fn answer_app_update(&self, agreed: bool) {
        let answer = self
            .app_update
            .borrow_mut()
            .as_mut()
            .and_then(|update| update.answer.take());
        let Some(answer) = answer else {
            return;
        };
        if agreed && answer.send(true).is_ok() {
            let job = self.start_host_job(&tr!("Updating eov"), None);
            self.app_update_job.replace(Some(job));
        } else {
            self.app_update.borrow_mut().take();
        }
    }

    fn app_update_finished(&self, result: Result<UpdateOutcome, String>) {
        if let Some(job) = self.app_update_job.take() {
            self.finish_host_job(&job);
        }
        self.app_update.borrow_mut().take();
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                self.notify(
                    NotifyLevel::Error,
                    &tr!("Could not update eov"),
                    Some(&error),
                    None,
                );
                return;
            }
        };
        let plugins: Vec<String> = outcome
            .plugins
            .iter()
            .map(|plugin| {
                let status = plugin_outcome_text(plugin.status);
                match plugin.message.as_deref() {
                    // Translators: one plugin of an update: its id, what
                    // happened to it ("updated", "failed", …) and why.
                    Some(message) => tr!(
                        "{plugin}: {status} ({message})",
                        plugin = isolate(&plugin.id),
                        status = status,
                        message = message
                    ),
                    // Translators: one plugin of an update: its id and what
                    // happened to it ("updated", "failed", …).
                    None => tr!(
                        "{plugin}: {status}",
                        plugin = isolate(&plugin.id),
                        status = status
                    ),
                }
            })
            .collect();
        let failed = outcome
            .plugins
            .iter()
            .any(|plugin| plugin.status == "failed");
        let mut detail: Vec<String> = outcome.message.iter().cloned().collect();
        detail.extend(plugins);
        let detail = detail.join("\n");
        let detail = (!detail.is_empty()).then_some(detail.as_str());
        let level = if failed {
            NotifyLevel::Warning
        } else {
            NotifyLevel::Info
        };
        let title = match outcome.app {
            "restart-pending" => tr!("eov restarts to finish the update"),
            "updated" => tr!(
                "eov {} is installed; restart eov to use it",
                isolate(outcome.installed_version.as_deref().unwrap_or(""))
            ),
            "downloaded" => tr!("The eov update was downloaded"),
            _ if failed => tr!("Some plugins could not be updated"),
            _ => tr!("Plugins are updated; restart eov to use them"),
        };
        self.notify(level, &title, detail, None);
        self.refresh_plugin_panel();
        if outcome.exit_for_helper {
            // The helper waits for this process to end. Quit as the Quit
            // shortcut does, so the session is saved and plugins shut down.
            if let Some(ui) = self.ui() {
                use slint::ComponentHandle;
                let _ = ui.hide();
            }
            let _ = slint::quit_event_loop();
        }
    }
}
