//! Updating eov from the window (AUDIT §14.4). Check for Updates on the
//! Home screen plans an update the way `eov update` does, shows the plan,
//! and applies it once the user agrees. Every check asks before going
//! online, and the administrator policy applies as on the command line.
//!
//! One worker thread owns the update: it plans, waits for the user's answer
//! and executes, so the plan never leaves the thread that made it.

use super::{NotifyLevel, UiHost, with_host};
use crate::update::{PlanSummary, UpdateContext, UpdateOutcome, UpdateRequest};
use std::sync::mpsc;

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
                "An update check is already running",
                None,
                None,
            );
            return;
        }
        if let Err(error) = crate::update::policy::current().ensure_network_allowed() {
            self.notify(
                NotifyLevel::Warning,
                "Checking for updates is turned off",
                Some(&format!("{error:#}")),
                None,
            );
            return;
        }
        self.confirm(
            "Check for eov updates?",
            "eov downloads the release information of the latest eov from GitHub (github.com/eosin-platform/eov) and compares it with this installation and its plugins. Nothing else is sent, and nothing is installed without asking.",
            "Check",
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
        let job = self.start_host_job("Checking for eov updates", None);
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
                        let error = format!("{error:#}");
                        let _ = slint::invoke_from_event_loop(move || {
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
                let result = crate::update::execute_update(&mut ctx, planned)
                    .map_err(|error| format!("{error:#}"));
                let _ = slint::invoke_from_event_loop(move || {
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
            "Could not check for eov updates",
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
                (format!("eov {} is available", summary.target), summary.text)
            } else {
                (
                    format!("eov {} and its plugins are up to date", summary.current),
                    summary.text,
                )
            };
            self.notify(NotifyLevel::Info, &title, Some(&detail), None);
            return;
        }
        let title = if summary.changes_app {
            format!("Update eov to {}?", summary.target)
        } else if summary.plugin_changes == 1 {
            "Update a plugin?".to_string()
        } else {
            format!("Update {} plugins?", summary.plugin_changes)
        };
        let mut message = summary.text;
        message.push_str(if summary.exits_for_helper {
            "\n\neov closes, the update helper replaces the application, and eov starts again."
        } else if summary.changes_app {
            "\n\nThe new version is used when eov restarts."
        } else {
            "\n\nUpdated plugins are used when eov restarts."
        });
        if summary.exits_for_helper && crate::config::load_restore_session() {
            message.push_str(" Open slides are reopened.");
        }
        let label = if summary.exits_for_helper {
            "Update and Restart"
        } else {
            "Update"
        };
        self.confirm(&title, &message, label, false, |agreed| {
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
            let job = self.start_host_job("Updating eov", None);
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
                    "Could not update eov",
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
                let message = plugin
                    .message
                    .as_deref()
                    .map(|message| format!(" ({message})"))
                    .unwrap_or_default();
                format!("{}: {}{message}", plugin.id, plugin.status)
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
            "restart-pending" => "eov restarts to finish the update".to_string(),
            "updated" => format!(
                "eov {} is installed; restart eov to use it",
                outcome.installed_version.as_deref().unwrap_or("")
            ),
            "downloaded" => "The eov update was downloaded".to_string(),
            _ if failed => "Some plugins could not be updated".to_string(),
            _ => "Plugins are updated; restart eov to use them".to_string(),
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
