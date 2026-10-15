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
mod callbacks;
mod commands;
mod input;
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
use std::collections::VecDeque;
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

/// Options for [`init`].
pub(crate) struct InitOptions {
    pub plugin_dir: PathBuf,
    /// Start without loading any plugin.
    pub safe_mode: bool,
    /// Plugins the user turned off.
    pub disabled: Vec<String>,
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
}

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

impl UiHost {
    fn start_plugins(&self, options: &InitOptions) {
        let mut runtime = self.runtime.borrow_mut();
        if options.safe_mode {
            tracing::warn!("Safe mode: plugins are not loaded");
        }
        let discovered = crate::plugins::discovery::discover_plugins(&options.plugin_dir);
        for descriptor in discovered.descriptors {
            let manifest = &descriptor.manifest;
            let info = PluginInfo {
                id: manifest.id.clone(),
                name: manifest.name.clone(),
                version: manifest.version.clone(),
                description: manifest.description.clone(),
                repository: manifest.repository.clone(),
                root: descriptor.root.clone(),
                package: None,
            };
            if options.safe_mode {
                runtime.add_inactive(info, PluginStatus::Disabled);
                continue;
            }
            if options.disabled.iter().any(|id| id == &manifest.id) {
                runtime.add_inactive(info, PluginStatus::Disabled);
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
        for old in discovered.old_plugin_files {
            tracing::info!("Older plugin package left in place: {}", old.display());
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

    /// Sends `Ready` (after the window is shown).
    pub(crate) fn ready(&self) {
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
        self.events.borrow_mut().push_back((target, event));
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
                    disabled: Vec::new(),
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
