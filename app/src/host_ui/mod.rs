//! The UI-thread side of plugins, host tools and overlays.
//!
//! [`UiHost`] ties the pieces together:
//!
//! - the [`PluginRuntime`] (instances, event delivery),
//! - the interaction [`Controller`] (all viewport input),
//! - the overlay store, render worker and per-pane overlay frames,
//! - plugin surfaces, the undo history, notifications, jobs and the
//!   Plugins panel.
//!
//! Rules that keep it deadlock-free (PLG-2, PLG-3):
//!
//! - plugins are only called from [`UiHost::pump`] and from Slint callbacks
//!   (context menus, surface callbacks), never while `AppState` is locked;
//! - code holding the `AppState` lock only *queues* events
//!   ([`UiHost::post`]) and never borrows the runtime;
//! - host calls from plugins never touch `AppState` (they use the published
//!   snapshot), so plugin threads cannot block on the UI.

mod annotations;
mod bookmarks;
mod callbacks;
mod commands;
mod input;
pub(crate) mod key_help;
mod measurements_panel;
mod menus;
mod notices;
mod overlays;
mod panel;
mod publish;
mod surfaces;

use crate::AppWindow;
use crate::interaction::{Controller, MeasureMode};
use crate::overlay::OverlayStore;
use crate::overlay::worker::OverlayWorker;
use crate::plugin_runtime::filters::SharedFilterChain;
use crate::plugin_runtime::loader::{self, Module};
use crate::plugin_runtime::settings::SettingsStore;
use crate::plugin_runtime::shared::HostShared;
use crate::plugin_runtime::undo::UndoStack;
use crate::plugin_runtime::{EventTarget, PluginInfo, PluginRuntime, PluginStatus};
use crate::state::AppState;
use eov_common::TileCache;
use eov_plugin_api::abi::FfiErrorKind;
use eov_plugin_api::protocol::{HostEvent, NotifyLevel};
use parking_lot::RwLock;
use slint::{ComponentHandle, Timer};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

pub(crate) use input::modifiers_from_slint;
pub(crate) use overlays::PaneOverlayFields;
pub(crate) use publish::calibration;

thread_local! {
    static HOST: RefCell<Option<Rc<UiHost>>> = const { RefCell::new(None) };
}

/// Runs `f` with the UI host, if it has been initialised on this thread.
pub(crate) fn with_host<R>(f: impl FnOnce(&Rc<UiHost>) -> R) -> Option<R> {
    let host = HOST.with(|slot| slot.borrow().clone())?;
    Some(f(&host))
}

/// An eov window gained or lost keyboard focus. Plugins learn of it at once,
/// not with the next frame, which never comes while nothing is open.
pub(crate) fn window_focus_changed() {
    // After winit has delivered the focus events of every window involved.
    Timer::single_shot(std::time::Duration::ZERO, || {
        with_host(|host| host.publish_now());
    });
}

/// Options for [`init`].
pub(crate) struct InitOptions {
    pub plugin_dir: PathBuf,
    /// Start without loading any plugin (`--safe-mode`).
    pub safe_mode: bool,
}

pub(crate) struct UiHost {
    pub(crate) ui: slint::Weak<AppWindow>,
    pub(crate) state: Arc<RwLock<AppState>>,
    pub(crate) tile_cache: Arc<TileCache>,
    pub(crate) render_timer: Rc<Timer>,
    pub(crate) store: Arc<OverlayStore>,
    pub(crate) shared: Arc<HostShared>,
    pub(crate) filters: SharedFilterChain,
    runtime: RefCell<PluginRuntime>,
    pub(crate) controller: RefCell<Controller>,
    worker: OverlayWorker,
    overlays: RefCell<overlays::OverlayPanes>,
    surfaces: RefCell<surfaces::Surfaces>,
    pub(crate) undo: RefCell<UndoStack>,
    notices: RefCell<notices::Notices>,
    events: RefCell<VecDeque<(EventTarget, HostEvent)>>,
    pump_scheduled: Cell<bool>,
    tracker: RefCell<publish::Tracker>,
    menu: RefCell<Option<menus::OpenMenu>>,
    momentary: RefCell<Vec<input::Momentary>>,
    measure_mode: Cell<MeasureMode>,
    plugin_dir: PathBuf,
    safe_mode: bool,
    /// Plugin whose sidebar was open at the last exit.
    restore_sidebar: Option<String>,
    /// Cancel handlers of host jobs (exports, dataset runs), by job id.
    host_job_cancels: RefCell<HashMap<String, Box<dyn Fn()>>>,
    /// Each pane's navigation history (Alt+←/→).
    pub(crate) history: RefCell<crate::history::History>,
    /// The view the Add Bookmark dialog is naming.
    pending_bookmark: RefCell<Option<bookmarks::PendingBookmark>>,
    /// The bookmark menu last opened.
    bookmark_menu: RefCell<Option<bookmarks::BookmarkMenu>>,
    /// The pointer's last window position over a pane, for menus opened
    /// from the keyboard.
    last_pointer: Cell<Option<(f32, f32)>>,
    /// The slide the Measurements panel shows.
    measurements_panel: Cell<Option<eov_plugin_api::protocol::FileId>>,
    /// Newer versions found by the Plugins panel's Check for Updates, by
    /// plugin id.
    plugin_updates: RefCell<HashMap<String, String>>,
}

/// Owner recorded for host jobs in the status bar; no plugin id is empty.
pub(crate) const HOST_JOB_OWNER: &str = "";

/// Creates the UI host, starts every plugin and wires the Slint callbacks.
pub(crate) fn init(
    ui: &AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<TileCache>,
    render_timer: &Rc<Timer>,
    options: InitOptions,
) -> Rc<UiHost> {
    let store = Arc::new(OverlayStore::new());
    let shared = Arc::new(HostShared::new(
        Arc::clone(&store),
        SettingsStore::new(),
        crate::version::BUILD_VERSION,
    ));
    // Plugin threads wake the UI thread to apply what they queued.
    shared.set_wake(|| {
        let _ = slint::invoke_from_event_loop(|| {
            with_host(|host| host.pump());
        });
    });
    let filters = state.read().filter_chain.clone();
    let worker = OverlayWorker::new(2, || {
        let _ = slint::invoke_from_event_loop(|| {
            with_host(|host| host.request_render());
        });
    });
    let mut controller = Controller::new(Arc::clone(&store));
    controller.scroll_pans = crate::config::load_scroll_pans().unwrap_or(false);
    let measure_mode = crate::config::load_measure_mode()
        .ok()
        .flatten()
        .and_then(|id| MeasureMode::from_id(&id))
        .unwrap_or_default();
    let host = Rc::new(UiHost {
        ui: ui.as_weak(),
        state: Arc::clone(state),
        tile_cache: Arc::clone(tile_cache),
        render_timer: Rc::clone(render_timer),
        store,
        runtime: RefCell::new(PluginRuntime::new(Arc::clone(&shared), filters.clone())),
        shared,
        filters,
        controller: RefCell::new(controller),
        worker,
        overlays: RefCell::new(overlays::OverlayPanes::default()),
        surfaces: RefCell::new(surfaces::Surfaces::default()),
        undo: RefCell::new(UndoStack::default()),
        notices: RefCell::new(notices::Notices::default()),
        events: RefCell::new(VecDeque::new()),
        pump_scheduled: Cell::new(false),
        tracker: RefCell::new(publish::Tracker::default()),
        menu: RefCell::new(None),
        momentary: RefCell::new(Vec::new()),
        measure_mode: Cell::new(measure_mode),
        plugin_dir: options.plugin_dir.clone(),
        safe_mode: options.safe_mode,
        restore_sidebar: crate::config::load_sidebar_plugin().ok().flatten(),
        host_job_cancels: RefCell::new(HashMap::new()),
        history: RefCell::new(crate::history::History::default()),
        pending_bookmark: RefCell::new(None),
        bookmark_menu: RefCell::new(None),
        last_pointer: Cell::new(None),
        measurements_panel: Cell::new(None),
        plugin_updates: RefCell::new(HashMap::new()),
    });
    HOST.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&host)));
    host.publish_now();
    host.start_plugins(&options);
    callbacks::install(ui, &host);
    host.refresh_toolbar();
    host.refresh_undo_ui();
    host.refresh_plugin_panel();
    ui.set_measure_mode_label(host.measure_mode.get().label().into());
    host
}

/// Panel information for a package that is not started.
fn inactive_info(package: &crate::update::plugins::InstalledPackage) -> PluginInfo {
    PluginInfo {
        id: package.id.clone(),
        name: package.name.clone(),
        version: package.version.clone(),
        description: package.description.clone(),
        repository: package.provenance.repository.clone(),
        root: PathBuf::new(),
        package: Some(package.package_path.clone()),
    }
}

impl UiHost {
    fn start_plugins(&self, options: &InitOptions) {
        let mut runtime = self.runtime.borrow_mut();
        if options.safe_mode {
            tracing::warn!("Safe mode: plugins are not loaded");
        }
        // The same inventory as `eov plugin ls`: per id the highest
        // compatible version, the user's directory before the policy's
        // read-only ones, with provenance and the shared disabled list.
        let policy = crate::update::policy::current();
        let host_version =
            semver::Version::parse(crate::version::BUILD_VERSION).expect("valid version");
        let inventory = match crate::update::plugins::inventory(
            &options.plugin_dir,
            &policy.plugin_dirs,
            &host_version,
        ) {
            Ok(inventory) => inventory,
            Err(error) => {
                tracing::error!("Could not list plugins: {error:#}");
                self.notices.borrow_mut().push(
                    NotifyLevel::Error,
                    "Plugins could not be loaded",
                    Some(&format!("{error:#}")),
                    None,
                );
                return;
            }
        };
        for broken in &inventory.broken {
            tracing::warn!(
                "Unreadable plugin package {}: {}",
                broken.package_path.display(),
                broken.error
            );
        }
        let mut listed = std::collections::BTreeSet::new();
        for package in inventory.packages.iter().filter(|package| package.selected) {
            listed.insert(package.id.clone());
            let descriptor = match crate::plugins::discovery::load_descriptor(&package.package_path)
            {
                Ok(descriptor) => descriptor,
                Err(error) => {
                    runtime.add_inactive(
                        inactive_info(package),
                        PluginStatus::Failed(format!("could not unpack the package: {error}")),
                    );
                    continue;
                }
            };
            let manifest = &descriptor.manifest;
            let info = PluginInfo {
                id: manifest.id.clone(),
                name: manifest.name.clone(),
                version: manifest.version.clone(),
                description: manifest.description.clone(),
                repository: manifest.repository.clone(),
                root: descriptor.root.clone(),
                package: Some(package.package_path.clone()),
            };
            if options.safe_mode || package.disabled {
                runtime.add_inactive(info, PluginStatus::Disabled);
                continue;
            }
            // Packages from the policy's own directories are provisioned by
            // the administrator; anything else must be official.
            if !package.system
                && package.provenance.kind != "official"
                && let Err(error) = policy.ensure_third_party_allowed("Loading this plugin")
            {
                runtime.add_inactive(info, PluginStatus::Failed(error.to_string()));
                continue;
            }
            if let Some(requirement) = &manifest.ui.slint {
                let slint = semver::Version::parse(surfaces::SLINT_VERSION).expect("valid");
                if !manifest.accepts_slint(&slint) {
                    tracing::warn!(
                        "Plugin '{}' needs Slint {requirement}; this eov uses {slint}",
                        manifest.id
                    );
                    self.notices.borrow_mut().push(
                        NotifyLevel::Warning,
                        &format!(
                            "needs Slint {requirement} but this eov uses {slint}; its panels may not display"
                        ),
                        None,
                        Some(&manifest.name),
                    );
                }
            }
            if !manifest.declares_environment {
                tracing::warn!(
                    "Plugin '{}' does not declare [environment]; assuming it supports this eov",
                    manifest.id
                );
            }
            let api = semver::Version::parse(eov_plugin_api::abi::API_VERSION).expect("valid");
            if !manifest.accepts_api(&api) {
                runtime.add_inactive(
                    info,
                    PluginStatus::Failed(format!(
                        "requires plugin API {}, this eov provides {api}",
                        manifest.api.as_deref().unwrap_or("?")
                    )),
                );
                continue;
            }
            let module = loader::library_path(&descriptor.root, &manifest.id)
                .and_then(|path| loader::load_library(&path));
            let data_dir = crate::paths::plugin_data_dir(&manifest.id);
            let cache_dir = crate::paths::plugin_cache_dir(&manifest.id);
            if let Err(error) = runtime.start(info, module, data_dir, cache_dir) {
                tracing::error!("Plugin '{}' did not start: {error}", manifest.id);
            }
        }
        for package in &inventory.packages {
            if listed.insert(package.id.clone()) {
                let reason = package
                    .incompatibility
                    .clone()
                    .unwrap_or_else(|| "not compatible with this eov".to_string());
                runtime.add_inactive(inactive_info(package), PluginStatus::Failed(reason));
            }
        }
        runtime.rebuild_filters();
        drop(runtime);
        self.report_failed_plugins();
    }

    /// Starts an in-process plugin module (tests, embedded plugins).
    #[allow(dead_code)]
    pub(crate) fn start_module(&self, info: PluginInfo, module: Module) -> Result<(), String> {
        let data_dir = crate::paths::plugin_data_dir(&info.id);
        let cache_dir = crate::paths::plugin_cache_dir(&info.id);
        let result = self
            .runtime
            .borrow_mut()
            .start(info, Ok(module), data_dir, cache_dir);
        self.runtime.borrow().rebuild_filters();
        self.refresh_toolbar();
        self.refresh_plugin_panel();
        result
    }

    fn report_failed_plugins(&self) {
        let failed: Vec<(String, String)> = self
            .runtime
            .borrow()
            .slots()
            .iter()
            .filter_map(|slot| match &slot.status {
                PluginStatus::Failed(reason) => Some((slot.info.name.clone(), reason.clone())),
                _ => None,
            })
            .collect();
        for (name, reason) in failed {
            self.notify(
                NotifyLevel::Error,
                &format!("Plugin {name} could not be started"),
                Some(&reason),
                Some(&name),
            );
        }
    }

    /// Tells the user, once, how an update applied by the helper after the
    /// last exit went (the CLI prints the same on the next `eov update`).
    fn report_last_update(&self) {
        let Some(report) = crate::update::take_helper_report(&crate::paths::updates_dir()) else {
            return;
        };
        let (level, title) = if report.applied() {
            (NotifyLevel::Success, "eov was updated")
        } else if report.left_damage() {
            (NotifyLevel::Error, "The last update failed")
        } else {
            (NotifyLevel::Warning, "The last update was rolled back")
        };
        self.notify(level, title, Some(&report.summary()), None);
    }

    /// Sends `Ready` (after the window is shown).
    pub(crate) fn ready(&self) {
        self.report_last_update();
        self.report_shortcut_problems();
        if let Some(report) = crate::logging::take_crash_report() {
            self.notify(
                NotifyLevel::Error,
                "eov closed unexpectedly last time",
                Some(&format!(
                    "A crash report was saved to {} (Open Log Folder in Messages).",
                    report.display()
                )),
                None,
            );
        }
        let ids: Vec<String> = self
            .runtime
            .borrow()
            .slots()
            .iter()
            .filter(|s| s.is_active())
            .map(|s| s.info.id.clone())
            .collect();
        for id in ids {
            let restore = self.restore_sidebar.as_deref() == Some(id.as_str());
            self.post(
                EventTarget::Plugin(Arc::from(id.as_str())),
                HostEvent::Ready {
                    restore_sidebar: restore,
                },
            );
        }
    }

    /// Delivers `Shutdown`, remembers the open sidebar and stops plugins.
    pub(crate) fn shutdown(&self) {
        let sidebar_owner = self.surfaces.borrow().sidebar_owner();
        if let Err(error) = crate::config::save_sidebar_plugin(sidebar_owner.as_deref()) {
            tracing::warn!("Could not remember the open sidebar: {error}");
        }
        self.surfaces.borrow_mut().close_all(self);
        self.runtime.borrow_mut().shutdown_all();
    }

    pub(crate) fn ui(&self) -> Option<AppWindow> {
        self.ui.upgrade()
    }

    pub(crate) fn request_render(&self) {
        crate::request_render_loop(&self.render_timer, &self.ui, &self.state, &self.tile_cache);
    }

    /// Queues an event; it is delivered by the next [`Self::pump`], which is
    /// scheduled on the event loop. Safe to call while `AppState` is locked.
    pub(crate) fn post(&self, target: EventTarget, event: HostEvent) {
        {
            let mut events = self.events.borrow_mut();
            // Pointer moves queue up while a plugin is busy: keep only the
            // latest one per tool and pane.
            if let HostEvent::Pointer {
                tool_id,
                pane,
                event: new,
                ..
            } = &event
                && new.kind == eov_plugin_api::protocol::PointerKind::Move
                && let Some((
                    last_target,
                    HostEvent::Pointer {
                        tool_id: last_tool,
                        pane: last_pane,
                        event: last,
                        ..
                    },
                )) = events.back()
                && last.kind == eov_plugin_api::protocol::PointerKind::Move
                && last_tool == tool_id
                && last_pane == pane
                && *last_target == target
            {
                events.pop_back();
            }
            events.push_back((target, event));
        }
        self.schedule_pump();
    }

    pub(crate) fn schedule_pump(&self) {
        if self.pump_scheduled.replace(true) {
            return;
        }
        Timer::single_shot(std::time::Duration::ZERO, || {
            with_host(|host| host.pump());
        });
    }

    /// Applies queued plugin commands and delivers queued events, with no
    /// host lock held. Runs on the UI thread.
    pub(crate) fn pump(&self) {
        self.pump_scheduled.set(false);
        for _ in 0..32 {
            let commands = self.shared.take_commands();
            let had_commands = !commands.is_empty();
            for (plugin_id, command) in commands {
                commands::apply(self, &plugin_id, command);
            }
            let events: Vec<_> = self.events.borrow_mut().drain(..).collect();
            if !had_commands && events.is_empty() {
                break;
            }
            for (target, event) in events {
                self.deliver(&target, &event);
            }
        }
        if !self.events.borrow().is_empty() || self.shared.has_pending_commands() {
            // A plugin keeps producing work; continue on a later turn.
            self.schedule_pump();
        }
        self.handle_filter_failures();
    }

    /// Delivers one event now and reports failures. Must not be called while
    /// `AppState` is locked.
    pub(crate) fn deliver(
        &self,
        target: &EventTarget,
        event: &HostEvent,
    ) -> Vec<(Arc<str>, serde_json::Value)> {
        let Ok(mut runtime) = self.runtime.try_borrow_mut() else {
            // Re-entrant delivery (a plugin surface callback during another
            // delivery): try again on the next turn.
            self.events
                .borrow_mut()
                .push_back((target.clone(), event.clone()));
            self.schedule_pump();
            return Vec::new();
        };
        let deliveries = runtime.deliver(target, event);
        drop(runtime);
        let mut replies = Vec::new();
        for delivery in deliveries {
            match delivery.result {
                Ok(value) => replies.push((delivery.plugin_id, value)),
                Err(error) => self.plugin_failed(&delivery.plugin_id, error.kind(), &error.message),
            }
        }
        replies
    }

    /// Sends a request to one plugin and returns its reply.
    pub(crate) fn request(&self, plugin_id: &str, event: &HostEvent) -> Option<serde_json::Value> {
        self.deliver(&EventTarget::Plugin(Arc::from(plugin_id)), event)
            .into_iter()
            .next()
            .map(|(_, value)| value)
    }

    fn plugin_name(&self, plugin_id: &str) -> String {
        self.runtime
            .borrow()
            .slot(plugin_id)
            .map(|slot| slot.info.name.clone())
            .unwrap_or_else(|| plugin_id.to_string())
    }

    fn plugin_failed(&self, plugin_id: &str, kind: FfiErrorKind, message: &str) {
        let name = self.plugin_name(plugin_id);
        match kind {
            FfiErrorKind::Panicked => {
                self.on_plugin_stopped(plugin_id);
                self.notify(
                    NotifyLevel::Error,
                    &format!("Plugin {name} crashed and was stopped"),
                    Some(message),
                    Some(&name),
                );
            }
            // Old plugins ignore new events; not an error.
            FfiErrorKind::Unsupported => {}
            FfiErrorKind::Revoked => {}
            _ => self.notify(NotifyLevel::Warning, message, None, Some(&name)),
        }
    }

    /// Cleans up the UI after a plugin stopped (quarantine or disable).
    fn on_plugin_stopped(&self, plugin_id: &str) {
        self.surfaces.borrow_mut().close_plugin(self, plugin_id);
        self.undo.borrow_mut().clear_plugin(plugin_id, None);
        let tool_owned = self
            .controller
            .borrow()
            .tool()
            .plugin_tool()
            .is_some_and(|(owner, _)| &**owner == plugin_id);
        if tool_owned {
            input::set_tool(self, crate::interaction::ToolChoice::Navigate);
        }
        self.notices.borrow_mut().clear_plugin(plugin_id);
        self.refresh_notices();
        self.refresh_toolbar();
        self.refresh_undo_ui();
        self.refresh_plugin_panel();
        self.overlays.borrow_mut().invalidate_all();
        self.request_render();
    }

    fn handle_filter_failures(&self) {
        let failures = self.filters.take_failures();
        if failures.is_empty() {
            return;
        }
        for failure in failures {
            if failure.kind == FfiErrorKind::Panicked {
                let stopped = self
                    .runtime
                    .borrow_mut()
                    .quarantine(&failure.plugin_id, &failure.message);
                if stopped {
                    self.plugin_failed(
                        &failure.plugin_id,
                        FfiErrorKind::Panicked,
                        &failure.message,
                    );
                }
            } else if failure.give_up {
                let name = self.plugin_name(&failure.plugin_id);
                let changed = self
                    .shared
                    .contributions
                    .lock()
                    .set_filter_enabled(&failure.plugin_id, &failure.filter_id, false)
                    .unwrap_or(false);
                if changed {
                    self.runtime.borrow().rebuild_filters();
                    self.post(
                        EventTarget::Plugin(Arc::clone(&failure.plugin_id)),
                        HostEvent::FilterToggled {
                            filter_id: failure.filter_id.clone(),
                            enabled: false,
                        },
                    );
                }
                self.notify(
                    NotifyLevel::Warning,
                    &format!(
                        "Filter {} was turned off after repeated errors",
                        failure.filter_id
                    ),
                    Some(&failure.message),
                    Some(&name),
                );
            }
        }
        self.request_render();
    }

    // -- notices ---------------------------------------------------------------

    pub(crate) fn notify(
        &self,
        level: NotifyLevel,
        message: &str,
        detail: Option<&str>,
        source: Option<&str>,
    ) {
        let toast = self
            .notices
            .borrow_mut()
            .push(level, message, detail, source);
        if let Some(ui) = self.ui() {
            if let Some((text, level)) = toast {
                notices::show_toast(&ui, &text, level);
            }
            self.notices.borrow().apply(&ui);
        }
    }

    /// Tells plugins which of their undo entries were dropped.
    pub(crate) fn undo_dropped(&self, dropped: Vec<crate::plugin_runtime::undo::UndoEntry>) {
        let mut by_plugin: std::collections::BTreeMap<Arc<str>, Vec<String>> = Default::default();
        for entry in dropped {
            if let crate::plugin_runtime::undo::UndoAction::Plugin {
                plugin_id,
                entry_id,
            } = entry.action
            {
                by_plugin.entry(plugin_id).or_default().push(entry_id);
            }
        }
        for (plugin_id, entry_ids) in by_plugin {
            self.post(
                EventTarget::Plugin(plugin_id),
                HostEvent::UndoDropped { entry_ids },
            );
        }
    }

    /// Starts a host background job (export, dataset run) in the status
    /// bar. `cancel` runs on the UI thread when the user cancels it.
    pub(crate) fn start_host_job(&self, title: &str, cancel: Option<Box<dyn Fn()>>) -> String {
        let job_id = self.shared.fresh_id("eov-job");
        let owner: Arc<str> = Arc::from(HOST_JOB_OWNER);
        self.notices.borrow_mut().start_job(
            &owner,
            job_id.clone(),
            eov_plugin_api::protocol::JobStartArgs {
                title: title.to_string(),
                cancellable: cancel.is_some(),
            },
        );
        if let Some(cancel) = cancel {
            self.host_job_cancels
                .borrow_mut()
                .insert(job_id.clone(), cancel);
        }
        self.refresh_notices();
        job_id
    }

    pub(crate) fn update_host_job(
        &self,
        job_id: &str,
        fraction: Option<f32>,
        message: Option<String>,
    ) {
        self.notices.borrow_mut().update_job(
            HOST_JOB_OWNER,
            eov_plugin_api::protocol::JobUpdateArgs {
                job_id: job_id.to_string(),
                fraction: fraction.map(|f| f.clamp(0.0, 1.0)),
                message,
            },
        );
        self.refresh_notices();
    }

    pub(crate) fn finish_host_job(&self, job_id: &str) {
        self.host_job_cancels.borrow_mut().remove(job_id);
        self.notices.borrow_mut().finish_job(
            HOST_JOB_OWNER,
            &eov_plugin_api::protocol::JobFinishArgs {
                job_id: job_id.to_string(),
                error: None,
            },
        );
        self.refresh_notices();
    }

    /// Runs a host job's cancel handler; `false` if it is not a host job.
    pub(crate) fn cancel_host_job(&self, job_id: &str) -> bool {
        let cancels = self.host_job_cancels.borrow();
        let Some(cancel) = cancels.get(job_id) else {
            return false;
        };
        cancel();
        true
    }

    pub(crate) fn refresh_notices(&self) {
        if let Some(ui) = self.ui() {
            self.notices.borrow().apply(&ui);
        }
    }

    /// Shows a short host status message in the status bar.
    pub(crate) fn status(&self, text: &str) {
        if let Some(ui) = self.ui() {
            ui.set_status_text(text.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolType;
    use crate::plugin_runtime::loader::Module;
    use crate::test_support::run_on_slint_ui_test_thread;
    use slint::Model;

    /// The Plugins panel lists a package eov cannot read, and Remove
    /// deletes it after the user confirms in eov's own dialog.
    #[test]
    fn unreadable_packages_are_listed_and_can_be_removed() {
        run_on_slint_ui_test_thread(|| {
            let ui = AppWindow::new().unwrap();
            let state = Arc::new(RwLock::new(AppState::new()));
            let tile_cache = Arc::new(TileCache::with_limits(64, 64 << 20));
            let timer = Rc::new(Timer::default());
            let plugin_dir = tempfile::tempdir().unwrap();
            let package = plugin_dir.path().join("broken.eop");
            std::fs::write(&package, b"not a tar archive").unwrap();
            let host = init(
                &ui,
                &state,
                &tile_cache,
                &timer,
                InitOptions {
                    plugin_dir: plugin_dir.path().to_path_buf(),
                    safe_mode: false,
                },
            );
            host.refresh_plugin_panel();
            let rows: Vec<_> = ui.get_plugin_rows().iter().collect();
            let row = rows
                .iter()
                .find(|row| row.id == "broken.eop")
                .expect("the broken package is listed");
            assert!(row.can_remove);
            assert!(!row.detail.is_empty());
            assert!(ui.get_plugin_problems());

            host.remove_plugin("broken.eop");
            assert!(ui.get_plugin_confirm_visible(), "removing asks first");
            assert!(ui.get_plugin_confirm_destructive());
            assert!(package.exists());
            host.answer_plugin_confirm(true);
            assert!(!package.exists());
            assert!(!ui.get_plugin_confirm_visible());
            assert!(
                !ui.get_plugin_rows()
                    .iter()
                    .any(|row| row.id == "broken.eop")
            );
        });
    }

    /// The example plugin running in a real (headless) main window.
    #[test]
    fn example_plugin_runs_end_to_end() {
        run_on_slint_ui_test_thread(|| {
            let ui = AppWindow::new().unwrap();
            let state = Arc::new(RwLock::new(AppState::new()));
            let tile_cache = Arc::new(TileCache::with_limits(64, 64 << 20));
            let timer = Rc::new(Timer::default());
            let plugin_dir = tempfile::tempdir().unwrap();
            let host = init(
                &ui,
                &state,
                &tile_cache,
                &timer,
                InitOptions {
                    plugin_dir: plugin_dir.path().to_path_buf(),
                    safe_mode: false,
                },
            );
            let root =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../plugin_api/example-plugin");
            host.start_module(
                PluginInfo {
                    id: "example_plugin".into(),
                    name: "Example".into(),
                    version: "1.0.0".into(),
                    root,
                    ..Default::default()
                },
                Module::from_ref(example_plugin::module()).unwrap(),
            )
            .unwrap();
            host.pump();

            // The toolbar shows the plugin's tool and button.
            let buttons: Vec<(String, bool)> = ui
                .get_plugin_buttons()
                .iter()
                .map(|b| (b.button_id.to_string(), b.is_tool))
                .collect();
            assert!(
                buttons.contains(&("toggle_sidebar".to_string(), false)),
                "{buttons:?}"
            );
            assert!(
                buttons.contains(&("place_marker".to_string(), true)),
                "{buttons:?}"
            );

            // Clicking the button reaches the plugin, which opens its sidebar.
            ui.invoke_plugin_button_clicked(
                "example_plugin".into(),
                "toggle_sidebar".into(),
                false,
            );
            host.pump();
            host.pump();
            assert!(ui.get_show_plugin_sidebar());
            assert_eq!(
                host.surfaces.borrow().sidebar_owner().as_deref(),
                Some("example_plugin")
            );
            let pressed = ui
                .get_plugin_buttons()
                .iter()
                .find(|b| b.button_id == "toggle_sidebar")
                .unwrap()
                .active;
            assert!(pressed, "the plugin marked its button active");

            // Plugin tools behave like host tools.
            ui.invoke_plugin_button_clicked("example_plugin".into(), "place_marker".into(), true);
            assert_eq!(ui.get_current_tool(), ToolType::Plugin);
            ui.invoke_tool_selected(ToolType::Measure);
            assert_eq!(ui.get_current_tool(), ToolType::Measure);

            // The Plugins panel lists it as running.
            host.refresh_plugin_panel();
            let rows = ui.get_plugin_rows();
            assert_eq!(rows.row_count(), 1);
            assert_eq!(rows.row_data(0).unwrap().status_level, 0);
            assert_eq!(ui.get_plugin_count(), 1);

            // Disabling stops it and removes its contributions.
            host.set_plugin_enabled("example_plugin", false);
            assert!(ui.get_plugin_buttons().iter().next().is_none());
            assert!(!ui.get_show_plugin_sidebar());

            host.shutdown();
            HOST.with(|slot| *slot.borrow_mut() = None);
        });
    }
}
