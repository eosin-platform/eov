//! The plugin runtime (plugin API 1.0).
//!
//! - [`loader`] opens libraries with checked layouts and versions.
//! - [`host_api`] implements the thread-safe, non-blocking host handle.
//! - [`shared`] holds what plugin threads and the UI thread share.
//! - [`PluginRuntime`] owns the instances and delivers events on the UI
//!   thread, never while the host holds a lock, and quarantines plugins that
//!   panic (PLG-3, PLG-7).

pub mod contributions;
pub mod filters;
pub mod host_api;
pub mod loader;
pub mod settings;
pub mod shared;
pub mod undo;

use abi_stable::std_types::RStr;
use eov_plugin_api::abi::{FfiError, FfiErrorKind, PluginBox};
use eov_plugin_api::protocol::{HostEvent, PluginInit, event_names};
use filters::SharedFilterChain;
use host_api::HostApi;
use loader::{LoadError, Module};
use serde_json::Value;
use shared::{HostShared, PluginContext};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Handlers slower than this are reported in the log.
const SLOW_HANDLER: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginStatus {
    Active,
    /// Turned off by the user (or safe mode).
    Disabled,
    /// Could not be loaded or started.
    Failed(String),
    /// Stopped after misbehaving (a panic).
    Quarantined(String),
}

impl PluginStatus {
    pub fn label(&self) -> String {
        match self {
            PluginStatus::Active => "Active".to_string(),
            PluginStatus::Disabled => "Disabled".to_string(),
            PluginStatus::Failed(reason) => format!("Failed: {reason}"),
            PluginStatus::Quarantined(reason) => format!("Stopped: {reason}"),
        }
    }
}

/// Static facts about a plugin, from its manifest and package.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub repository: Option<String>,
    /// The extracted package directory.
    pub root: PathBuf,
    /// The `.eop` the plugin came from, if any.
    pub package: Option<PathBuf>,
}

pub struct PluginSlot {
    pub info: PluginInfo,
    pub status: PluginStatus,
    /// The plugin-API version the plugin was built against.
    pub api_version: Option<String>,
    ctx: Option<Arc<PluginContext>>,
    instance: Option<Arc<PluginBox>>,
}

impl PluginSlot {
    pub fn id(&self) -> &str {
        &self.info.id
    }

    pub fn is_active(&self) -> bool {
        self.status == PluginStatus::Active
    }
}

/// Who an event is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventTarget {
    All,
    Plugin(Arc<str>),
}

/// The outcome of delivering an event to one plugin.
#[derive(Debug)]
pub struct Delivery {
    pub plugin_id: Arc<str>,
    pub result: Result<Value, FfiError>,
}

/// Events delivered only to plugins that did not opt out via
/// `events.subscribe`.
const SUBSCRIBABLE: &[&str] = &[
    event_names::FILE_OPENED,
    event_names::FILE_CLOSED,
    event_names::ACTIVE_CHANGED,
    event_names::VIEW_CHANGED,
    event_names::WINDOW_FOCUS_CHANGED,
    event_names::OBJECT_HOVERED,
    event_names::EMPTY_AREA_CLICKED,
    event_names::CONTEXT_MENU_REQUESTED,
];

pub struct PluginRuntime {
    shared: Arc<HostShared>,
    filters: SharedFilterChain,
    slots: Vec<PluginSlot>,
}

impl PluginRuntime {
    pub fn new(shared: Arc<HostShared>, filters: SharedFilterChain) -> Self {
        Self {
            shared,
            filters,
            slots: Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn shared(&self) -> &Arc<HostShared> {
        &self.shared
    }

    /// Every known plugin, sorted by id.
    pub fn slots(&self) -> &[PluginSlot] {
        &self.slots
    }

    pub fn slot(&self, plugin_id: &str) -> Option<&PluginSlot> {
        self.slots.iter().find(|s| s.id() == plugin_id)
    }

    #[cfg(test)]
    pub fn is_active(&self, plugin_id: &str) -> bool {
        self.slot(plugin_id).is_some_and(PluginSlot::is_active)
    }

    fn insert_slot(&mut self, slot: PluginSlot) {
        self.slots.retain(|s| s.id() != slot.id());
        let index = self
            .slots
            .partition_point(|s| s.info.id.as_str() < slot.info.id.as_str());
        self.slots.insert(index, slot);
    }

    /// Records a plugin that will not run (shown in the Plugins panel).
    pub fn add_inactive(&mut self, info: PluginInfo, status: PluginStatus) {
        self.insert_slot(PluginSlot {
            info,
            status,
            api_version: None,
            ctx: None,
            instance: None,
        });
    }

    /// Starts a plugin from a loaded module.
    pub fn start(
        &mut self,
        info: PluginInfo,
        module: Result<Module, LoadError>,
        data_dir: PathBuf,
        cache_dir: PathBuf,
    ) -> Result<(), String> {
        let module = match module {
            Ok(module) => module,
            Err(error) => {
                let reason = error.to_string();
                self.add_inactive(info, PluginStatus::Failed(reason.clone()));
                return Err(reason);
            }
        };
        for dir in [&data_dir, &cache_dir] {
            if let Err(error) = std::fs::create_dir_all(dir) {
                tracing::warn!("Cannot create {}: {error}", dir.display());
            }
        }
        let ctx = Arc::new(PluginContext::new(&info.id, info.root.clone()));
        let settings = self.shared.settings.load(&info.id, &data_dir);
        let init = PluginInit {
            plugin_id: info.id.clone(),
            plugin_version: info.version.clone(),
            host_name: "eov".to_string(),
            host_version: self.shared.host_version.clone(),
            api_version: eov_plugin_api::abi::API_VERSION.to_string(),
            plugin_root: info.root.to_string_lossy().into_owned(),
            data_dir: data_dir.to_string_lossy().into_owned(),
            cache_dir: cache_dir.to_string_lossy().into_owned(),
            settings,
        };
        let host = HostApi::new(Arc::clone(&ctx), Arc::clone(&self.shared)).into_ffi();
        let started = Instant::now();
        let result = module.instantiate(host, &init);
        let elapsed = started.elapsed();
        if elapsed > Duration::from_millis(500) {
            tracing::warn!("Plugin '{}' took {elapsed:?} to start", info.id);
        }
        match result {
            Ok(instance) => {
                tracing::info!(
                    "Started plugin '{}' {} (plugin API {})",
                    info.id,
                    info.version,
                    module.api_version()
                );
                self.insert_slot(PluginSlot {
                    info,
                    status: PluginStatus::Active,
                    api_version: Some(module.api_version().to_string()),
                    ctx: Some(ctx),
                    instance: Some(Arc::new(instance)),
                });
                Ok(())
            }
            Err(error) => {
                ctx.revoke();
                self.shared.forget_plugin(&info.id);
                let reason = error.to_string();
                tracing::error!("Plugin '{}' failed to start: {reason}", info.id);
                self.insert_slot(PluginSlot {
                    info,
                    status: PluginStatus::Failed(reason.clone()),
                    api_version: Some(module.api_version().to_string()),
                    ctx: None,
                    instance: None,
                });
                Err(reason)
            }
        }
    }

    /// The instances of active plugins (for the filter chain).
    pub fn instances(&self) -> HashMap<Arc<str>, Arc<PluginBox>> {
        self.slots
            .iter()
            .filter(|slot| slot.is_active())
            .filter_map(|slot| {
                Some((
                    Arc::clone(&slot.ctx.as_ref()?.id),
                    Arc::clone(slot.instance.as_ref()?),
                ))
            })
            .collect()
    }

    /// Republishes the enabled filters to the renderer.
    pub fn rebuild_filters(&self) {
        let instances = self.instances();
        let contributions = self.shared.contributions.lock();
        self.filters.rebuild(&contributions, &instances);
    }

    /// Delivers an event to its target(s), in plugin-id order. A plugin that
    /// panics is quarantined; other failures are returned for reporting.
    pub fn deliver(&mut self, target: &EventTarget, event: &HostEvent) -> Vec<Delivery> {
        let json = match serde_json::to_value(event) {
            Ok(json) => json,
            Err(error) => {
                tracing::error!("Cannot encode plugin event: {error}");
                return Vec::new();
            }
        };
        let event_type = json
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let text = json.to_string();
        let mut deliveries = Vec::new();
        let mut to_quarantine = Vec::new();
        for slot in &self.slots {
            if !slot.is_active() {
                continue;
            }
            let (Some(ctx), Some(instance)) = (&slot.ctx, &slot.instance) else {
                continue;
            };
            match target {
                EventTarget::All => {
                    if SUBSCRIBABLE.contains(&event_type.as_str())
                        && !self.shared.wants(&ctx.id, &event_type)
                    {
                        continue;
                    }
                }
                EventTarget::Plugin(id) => {
                    if **id != *ctx.id {
                        continue;
                    }
                }
            }
            let started = Instant::now();
            let reply = instance.handle_event(RStr::from_str(&text)).into_result();
            let elapsed = started.elapsed();
            if elapsed > SLOW_HANDLER {
                tracing::warn!(
                    "Plugin '{}' took {} ms to handle '{event_type}'; slow handlers stall the UI",
                    ctx.id,
                    elapsed.as_millis()
                );
            }
            let result = match reply {
                Ok(reply) => Ok(serde_json::from_str(reply.as_str()).unwrap_or(Value::Null)),
                Err(error) => {
                    if error.kind() == FfiErrorKind::Panicked {
                        to_quarantine.push((Arc::clone(&ctx.id), error.message.to_string()));
                    }
                    Err(error)
                }
            };
            deliveries.push(Delivery {
                plugin_id: Arc::clone(&ctx.id),
                result,
            });
        }
        for (plugin_id, reason) in to_quarantine {
            self.quarantine(&plugin_id, &reason);
        }
        deliveries
    }

    /// Sends an event to one plugin and returns its reply.
    #[cfg(test)]
    pub fn request(&mut self, plugin_id: &str, event: &HostEvent) -> Result<Value, FfiError> {
        self.deliver(&EventTarget::Plugin(Arc::from(plugin_id)), event)
            .into_iter()
            .next()
            .map(|delivery| delivery.result)
            .unwrap_or_else(|| {
                Err(FfiError::new(
                    FfiErrorKind::Revoked,
                    format!("plugin '{plugin_id}' is not running"),
                ))
            })
    }

    fn stop(&mut self, plugin_id: &str, status: PluginStatus) -> bool {
        let Some(slot) = self.slots.iter_mut().find(|s| s.id() == plugin_id) else {
            return false;
        };
        if !slot.is_active() {
            slot.status = status;
            return false;
        }
        if let Some(ctx) = &slot.ctx {
            ctx.revoke();
        }
        slot.status = status;
        let instance = slot.instance.take();
        self.shared.forget_plugin(plugin_id);
        self.rebuild_filters();
        if let Some(instance) = instance {
            // The SDK catches panics in shutdown.
            instance.shutdown();
        }
        true
    }

    /// Stops a misbehaving plugin and removes everything it contributed.
    pub fn quarantine(&mut self, plugin_id: &str, reason: &str) -> bool {
        tracing::error!("Stopping plugin '{plugin_id}': {reason}");
        self.stop(plugin_id, PluginStatus::Quarantined(reason.to_string()))
    }

    /// Stops a plugin at the user's request.
    pub fn disable(&mut self, plugin_id: &str) -> bool {
        self.stop(plugin_id, PluginStatus::Disabled)
    }

    /// Delivers `Shutdown` and releases every plugin.
    pub fn shutdown_all(&mut self) {
        self.deliver(&EventTarget::All, &HostEvent::Shutdown);
        let ids: Vec<String> = self
            .slots
            .iter()
            .filter(|s| s.is_active())
            .map(|s| s.info.id.clone())
            .collect();
        for id in ids {
            self.stop(&id, PluginStatus::Disabled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::OverlayStore;
    use eov_plugin_api::prelude::*;
    use settings::SettingsStore;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Shutdown calls per plugin id (tests run in parallel).
    static SHUTDOWNS: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

    fn shutdowns(id: &str) -> usize {
        SHUTDOWNS
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(id).copied())
            .unwrap_or(0)
    }

    struct Echo {
        host: Host,
        id: String,
    }

    impl Plugin for Echo {
        fn create(host: Host, init: PluginInit) -> Result<Self, String> {
            if init.plugin_id == "broken" {
                return Err("cannot start".into());
            }
            Ok(Self {
                host,
                id: init.plugin_id,
            })
        }

        fn on_event(&self, event: HostEvent) -> Result<(), String> {
            match event {
                HostEvent::ToolbarButtonClicked { button_id } if button_id == "panic" => {
                    panic!("echo panicked")
                }
                HostEvent::ToolbarButtonClicked { button_id } if button_id == "fail" => {
                    Err("nope".into())
                }
                HostEvent::FileOpened { .. } => {
                    self.host.subscribe(&[]).map_err(String::from)?;
                    Ok(())
                }
                _ => Ok(()),
            }
        }

        fn context_menu(&self, _context: &MenuContext) -> Vec<MenuItem> {
            vec![MenuItem {
                item_id: "x".into(),
                label: "X".into(),
                ..Default::default()
            }]
        }

        fn shutdown(&self) {
            *SHUTDOWNS
                .lock()
                .unwrap()
                .get_or_insert_with(HashMap::new)
                .entry(self.id.clone())
                .or_default() += 1;
        }
    }

    fn runtime() -> (tempfile::TempDir, PluginRuntime) {
        let dir = tempfile::tempdir().unwrap();
        let shared = Arc::new(HostShared::new(
            Arc::new(OverlayStore::new()),
            SettingsStore::new(),
            "1.0.0",
        ));
        (
            dir,
            PluginRuntime::new(shared, Arc::new(filters::FilterChain::default())),
        )
    }

    fn start(rt: &mut PluginRuntime, dir: &std::path::Path, id: &str) -> Result<(), String> {
        let module = Module::from_ref(eov_plugin_api::sdk::root_module::<Echo>());
        rt.start(
            PluginInfo {
                id: id.into(),
                name: id.into(),
                version: "1.0.0".into(),
                root: dir.to_path_buf(),
                ..Default::default()
            },
            module,
            dir.join(id).join("data"),
            dir.join(id).join("cache"),
        )
    }

    #[test]
    fn plugins_start_in_id_order_and_failures_are_recorded() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "zeta").unwrap();
        start(&mut rt, dir.path(), "alpha").unwrap();
        assert!(start(&mut rt, dir.path(), "broken").is_err());
        let ids: Vec<_> = rt.slots().iter().map(|s| s.id().to_string()).collect();
        assert_eq!(ids, ["alpha", "broken", "zeta"]);
        assert!(matches!(
            rt.slot("broken").unwrap().status,
            PluginStatus::Failed(_)
        ));
        assert!(dir.path().join("alpha/data").is_dir());
    }

    #[test]
    fn requests_return_replies_and_panics_quarantine() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "echo_q").unwrap();
        let reply = rt
            .request(
                "echo_q",
                &HostEvent::ContextMenuRequested(MenuContext {
                    pane: 1,
                    view: ViewInfo::default(),
                    image_x: 0.0,
                    image_y: 0.0,
                    hit: None,
                    selection: vec![],
                }),
            )
            .unwrap();
        let response: ContextMenuResponse = serde_json::from_value(reply).unwrap();
        assert_eq!(response.items[0].item_id, "x");

        let error = rt
            .request(
                "echo_q",
                &HostEvent::ToolbarButtonClicked {
                    button_id: "fail".into(),
                },
            )
            .unwrap_err();
        assert_eq!(error.kind(), FfiErrorKind::Failed);
        assert!(rt.is_active("echo_q"));

        let before = shutdowns("echo_q");
        let error = rt
            .request(
                "echo_q",
                &HostEvent::ToolbarButtonClicked {
                    button_id: "panic".into(),
                },
            )
            .unwrap_err();
        assert_eq!(error.kind(), FfiErrorKind::Panicked);
        assert!(matches!(
            rt.slot("echo_q").unwrap().status,
            PluginStatus::Quarantined(_)
        ));
        assert_eq!(shutdowns("echo_q"), before + 1);
        // Quarantined plugins receive nothing more.
        assert!(
            rt.request(
                "echo_q",
                &HostEvent::Ready {
                    restore_sidebar: false
                }
            )
            .is_err()
        );
    }

    #[test]
    fn subscriptions_filter_optional_events() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "echo").unwrap();
        // The plugin unsubscribes from everything optional on FileOpened.
        let first = rt.deliver(
            &EventTarget::All,
            &HostEvent::FileOpened {
                file: FileInfo::default(),
            },
        );
        assert_eq!(first.len(), 1);
        let second = rt.deliver(&EventTarget::All, &HostEvent::FileClosed { file_id: 1 });
        assert!(second.is_empty());
        // Lifecycle events still arrive.
        let ready = rt.deliver(
            &EventTarget::All,
            &HostEvent::Ready {
                restore_sidebar: false,
            },
        );
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn disabling_removes_contributions() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "echo").unwrap();
        rt.shared()
            .contributions
            .lock()
            .add_toolbar(
                &Arc::from("echo"),
                ToolbarButtonSpec {
                    button_id: "b".into(),
                    tooltip: "B".into(),
                    icon_svg: "<svg/>".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(rt.disable("echo"));
        assert!(rt.shared().contributions.lock().toolbar().is_empty());
        assert_eq!(rt.slot("echo").unwrap().status, PluginStatus::Disabled);
    }
}
