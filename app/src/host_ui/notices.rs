//! Messages, toasts, status-bar segments and background jobs.

use crate::{AppWindow, JobData, NotificationData};
use eov_plugin_api::protocol::{JobFinishArgs, JobStartArgs, JobUpdateArgs, NotifyLevel};
use slint::ComponentHandle;
use slint::{ModelRc, Timer, VecModel};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Messages kept in the list.
const MAX_MESSAGES: usize = 200;
/// The same message from the same source is toasted at most this often.
const TOAST_COOLDOWN: Duration = Duration::from_secs(5);

struct Message {
    level: NotifyLevel,
    message: String,
    detail: String,
    source: String,
    at: chrono_like::Clock,
}

/// Minimal wall-clock formatting without a date-time dependency.
mod chrono_like {
    #[derive(Clone, Copy)]
    pub struct Clock {
        pub hours: u64,
        pub minutes: u64,
        pub seconds: u64,
    }

    impl Clock {
        pub fn now() -> Self {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default();
            // UTC; the list only needs relative ordering and a rough time.
            Self {
                hours: (secs / 3600) % 24,
                minutes: (secs / 60) % 60,
                seconds: secs % 60,
            }
        }

        pub fn label(self) -> String {
            format!(
                "{:02}:{:02}:{:02} UTC",
                self.hours, self.minutes, self.seconds
            )
        }
    }
}

struct Job {
    plugin_id: Arc<str>,
    title: String,
    message: String,
    fraction: Option<f32>,
    cancellable: bool,
}

#[derive(Default)]
pub(super) struct Notices {
    messages: Vec<Message>,
    unread: usize,
    unread_errors: bool,
    last_toast: BTreeMap<(String, String), Instant>,
    status: BTreeMap<Arc<str>, String>,
    jobs: BTreeMap<String, Job>,
}

fn level_code(level: NotifyLevel) -> i32 {
    match level {
        NotifyLevel::Info => 0,
        NotifyLevel::Success => 1,
        NotifyLevel::Warning => 2,
        NotifyLevel::Error => 3,
    }
}

impl Notices {
    /// Records a message; returns the toast to show, if any.
    pub(super) fn push(
        &mut self,
        level: NotifyLevel,
        message: &str,
        detail: Option<&str>,
        source: Option<&str>,
    ) -> Option<(String, NotifyLevel)> {
        let source = source.unwrap_or_default().to_string();
        self.messages.push(Message {
            level,
            message: message.to_string(),
            detail: detail.unwrap_or_default().to_string(),
            source: source.clone(),
            at: chrono_like::Clock::now(),
        });
        if self.messages.len() > MAX_MESSAGES {
            self.messages.remove(0);
        }
        if matches!(level, NotifyLevel::Warning | NotifyLevel::Error) {
            self.unread += 1;
            self.unread_errors |= level == NotifyLevel::Error;
        }
        let key = (source.clone(), message.to_string());
        let now = Instant::now();
        let recent = self
            .last_toast
            .get(&key)
            .is_some_and(|at| now.duration_since(*at) < TOAST_COOLDOWN);
        if recent {
            return None;
        }
        self.last_toast.insert(key, now);
        if self.last_toast.len() > 500 {
            self.last_toast.clear();
        }
        let text = if source.is_empty() {
            message.to_string()
        } else {
            format!("{source}: {message}")
        };
        Some((text, level))
    }

    pub(super) fn mark_read(&mut self) {
        self.unread = 0;
        self.unread_errors = false;
    }

    pub(super) fn clear(&mut self) {
        self.messages.clear();
        self.mark_read();
    }

    pub(super) fn set_status(&mut self, plugin_id: &Arc<str>, text: Option<String>) {
        match text.filter(|t| !t.trim().is_empty()) {
            Some(text) => {
                self.status.insert(Arc::clone(plugin_id), text);
            }
            None => {
                self.status.remove(plugin_id);
            }
        }
    }

    pub(super) fn start_job(&mut self, plugin_id: &Arc<str>, job_id: String, args: JobStartArgs) {
        self.jobs.insert(
            job_id,
            Job {
                plugin_id: Arc::clone(plugin_id),
                title: args.title,
                message: String::new(),
                fraction: None,
                cancellable: args.cancellable,
            },
        );
    }

    pub(super) fn update_job(&mut self, plugin_id: &str, args: JobUpdateArgs) {
        if let Some(job) = self.jobs.get_mut(&args.job_id)
            && &*job.plugin_id == plugin_id
        {
            job.fraction = args.fraction;
            if let Some(message) = args.message {
                job.message = message;
            }
        }
    }

    /// Ends a job; returns its title when it failed (for a message).
    pub(super) fn finish_job(&mut self, plugin_id: &str, args: &JobFinishArgs) -> Option<String> {
        let owned = self
            .jobs
            .get(&args.job_id)
            .is_some_and(|job| &*job.plugin_id == plugin_id);
        if !owned {
            return None;
        }
        let job = self.jobs.remove(&args.job_id)?;
        args.error.as_ref().map(|_| job.title)
    }

    pub(super) fn job_owner(&self, job_id: &str) -> Option<Arc<str>> {
        self.jobs.get(job_id).map(|job| Arc::clone(&job.plugin_id))
    }

    pub(super) fn clear_plugin(&mut self, plugin_id: &str) {
        self.status.remove(plugin_id);
        self.jobs.retain(|_, job| &*job.plugin_id != plugin_id);
    }

    pub(super) fn apply(&self, ui: &AppWindow) {
        let items: Vec<NotificationData> = self
            .messages
            .iter()
            .rev()
            .map(|m| NotificationData {
                level: level_code(m.level),
                message: m.message.clone().into(),
                detail: m.detail.clone().into(),
                source: m.source.clone().into(),
                time: m.at.label().into(),
            })
            .collect();
        ui.set_notifications(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_unread_notifications(self.unread as i32);
        ui.set_unread_errors(self.unread_errors);
        let status = self
            .status
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("  ·  ");
        ui.set_plugin_status_text(status.into());
        let jobs: Vec<JobData> = self
            .jobs
            .iter()
            .map(|(id, job)| JobData {
                id: id.clone().into(),
                title: job.title.clone().into(),
                message: job.message.clone().into(),
                fraction: job.fraction.unwrap_or(0.0),
                indeterminate: job.fraction.is_none(),
                cancellable: job.cancellable,
            })
            .collect();
        ui.set_jobs(ModelRc::from(Rc::new(VecModel::from(jobs))));
    }
}

thread_local! {
    static TOAST_TIMER: Timer = Timer::default();
}

/// Shows a toast; errors stay up longer.
pub(crate) fn show_toast(ui: &AppWindow, text: &str, level: NotifyLevel) {
    ui.set_toast_text(text.into());
    ui.set_toast_level(level_code(level));
    ui.set_toast_visible(true);
    let duration = match level {
        NotifyLevel::Error => Duration::from_secs(8),
        NotifyLevel::Warning => Duration::from_secs(5),
        _ => Duration::from_millis(2500),
    };
    let ui_weak = ui.as_weak();
    TOAST_TIMER.with(|timer| {
        timer.start(slint::TimerMode::SingleShot, duration, move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_toast_visible(false);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_messages_are_toasted_once() {
        let mut notices = Notices::default();
        assert!(
            notices
                .push(NotifyLevel::Error, "boom", None, Some("p"))
                .is_some()
        );
        assert!(
            notices
                .push(NotifyLevel::Error, "boom", None, Some("p"))
                .is_none()
        );
        assert!(
            notices
                .push(NotifyLevel::Error, "other", None, Some("p"))
                .is_some()
        );
        assert_eq!(notices.messages.len(), 3);
        assert_eq!(notices.unread, 3);
        assert!(notices.unread_errors);
        notices.mark_read();
        assert_eq!(notices.unread, 0);
    }

    #[test]
    fn jobs_belong_to_their_plugin() {
        let mut notices = Notices::default();
        let plugin: Arc<str> = Arc::from("p");
        notices.start_job(
            &plugin,
            "j1".into(),
            JobStartArgs {
                title: "Work".into(),
                cancellable: true,
            },
        );
        notices.update_job(
            "intruder",
            JobUpdateArgs {
                job_id: "j1".into(),
                fraction: Some(0.9),
                message: None,
            },
        );
        assert_eq!(notices.jobs["j1"].fraction, None);
        let failed = notices.finish_job(
            "p",
            &JobFinishArgs {
                job_id: "j1".into(),
                error: Some("disk full".into()),
            },
        );
        assert_eq!(failed.as_deref(), Some("Work"));
        assert!(notices.jobs.is_empty());
    }
}
