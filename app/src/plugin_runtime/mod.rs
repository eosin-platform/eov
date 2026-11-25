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
pub mod slide_sources;
pub mod undo;
pub mod usage;

use abi_stable::std_types::RStr;
use eov_plugin_api::abi::{FfiError, FfiErrorKind, PluginBox};
use eov_plugin_api::i18n::tr;
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
    Failed(FailReason),
    /// Stopped after misbehaving (a panic): what the plugin or its SDK said.
    Quarantined(String),
}

impl PluginStatus {
    /// The status as the Plugins panel shows it, in the language of the
    /// interface (made when it is shown, so it follows a change of
    /// language).
    pub fn label(&self) -> String {
        match self {
            PluginStatus::Active => tr!("Plugin status" => "Active"),
            PluginStatus::Disabled => tr!("Plugin status" => "Disabled"),
            PluginStatus::Failed(reason) => tr!("Failed: {}", reason.localized()),
            PluginStatus::Quarantined(reason) => tr!("Stopped: {}", reason),
        }
    }

    /// The status in English, for the audit log.
    pub fn audit_text(&self) -> String {
        match self {
            PluginStatus::Active => "Active".to_string(),
            PluginStatus::Disabled => "Disabled".to_string(),
            PluginStatus::Failed(reason) => format!("Failed: {reason}"),
            PluginStatus::Quarantined(reason) => format!("Stopped: {reason}"),
        }
    }
}

/// Why a plugin could not be loaded or started, kept as data: `Display`
/// says it in English (the audit log, the session log), without the marks
/// that keep text in order in right-to-left languages;
/// [`FailReason::localized`] says it in the interface's language when the
/// Plugins panel or a message shows it. Texts from outside eov (the file
/// system, the library loader, the plugin) stay as they are in both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailReason {
    /// The package could not be unpacked: why (what reading it gave),
    /// worded when it is shown.
    Unpack(crate::plugins::discovery::package_cache::Reason),
    /// The administrator policy does not let this plugin load: it only
    /// permits official plugins, or it cannot be used (`invalid`, why) and
    /// is applied as restrictively as possible. `source` is the policy file,
    /// `contact` the policy's message for users.
    NotAllowed {
        source: Option<PathBuf>,
        invalid: Option<crate::update::policy::PolicyError>,
        contact: Option<String>,
    },
    /// The manifest asks for a plugin API this eov does not provide.
    Api { required: String, provided: String },
    /// No version of the package runs with this eov (`None`: the inventory
    /// did not say why).
    Incompatible(Option<crate::update::compat::Incompatibility>),
    /// The library could not be loaded, or the plugin's `create` failed.
    Load(LoadError),
}

impl std::fmt::Display for FailReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailReason::Unpack(reason) => {
                write!(f, "could not unpack the package: {reason}")
            }
            FailReason::NotAllowed {
                source,
                invalid,
                contact,
            } => {
                let policy = match source {
                    Some(path) => format!("the administrator policy {}", path.display()),
                    None => "the administrator policy".to_string(),
                };
                match invalid {
                    Some(error) => write!(
                        f,
                        "{policy}: {error}. eov applies the most restrictive settings until it is fixed."
                    )?,
                    None => write!(
                        f,
                        "Loading this plugin is not allowed: {policy} only permits official eov plugins."
                    )?,
                }
                match contact {
                    Some(contact) => write!(f, "\n{contact}"),
                    None => Ok(()),
                }
            }
            FailReason::Api { required, provided } => write!(
                f,
                "requires plugin API {required}, this eov provides {provided}"
            ),
            FailReason::Incompatible(Some(reason)) => write!(f, "{reason}"),
            FailReason::Incompatible(None) => f.write_str("not compatible with this eov"),
            FailReason::Load(error) => write!(f, "{error}"),
        }
    }
}

impl FailReason {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        use eov_plugin_api::i18n::isolate;
        match self {
            FailReason::Unpack(reason) => {
                tr!("could not unpack the package: {}", reason.shown())
            }
            FailReason::NotAllowed {
                source,
                invalid,
                contact,
            } => {
                // The administrator policy's own texts
                // (`Policy::ensure_third_party_allowed`).
                let policy = isolate(
                    source
                        .clone()
                        .or_else(crate::update::policy::policy_path)
                        .unwrap_or_default()
                        .display(),
                );
                let text = match invalid {
                    Some(error) => {
                        // Translators: {policy} is the policy file, {error} why it cannot be used.
                        tr!(
                            "the administrator policy {policy}: {error}. eov applies the most restrictive settings until it is fixed.",
                            policy = policy,
                            error = error.localized()
                        )
                    }
                    None => crate::update::policy::third_party_refusal(
                        crate::update::policy::ThirdPartyUse::LoadPlugin,
                        &policy,
                    ),
                };
                match contact {
                    Some(contact) => format!("{text}\n{contact}"),
                    None => text,
                }
            }
            FailReason::Api { required, provided } => tr!(
                "requires plugin API {required}, this eov provides {provided}",
                required = isolate(required),
                provided = isolate(provided),
            ),
            FailReason::Incompatible(Some(reason)) => reason.localized(),
            FailReason::Incompatible(None) => tr!("not compatible with this eov"),
            FailReason::Load(error) => error.localized(),
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
    /// The slide formats its manifest declares (`[[slide_formats]]`).
    pub slide_formats: Vec<eov_plugin_api::manifest::SlideFormatSpec>,
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
    /// Plugins run without a window (`eov batch`).
    headless: bool,
}

impl PluginRuntime {
    pub fn new(shared: Arc<HostShared>, filters: SharedFilterChain) -> Self {
        Self {
            shared,
            filters,
            slots: Vec::new(),
            headless: false,
        }
    }

    /// A runtime for plugins without a window (`eov batch`).
    pub fn headless(shared: Arc<HostShared>, filters: SharedFilterChain) -> Self {
        Self {
            headless: true,
            ..Self::new(shared, filters)
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

    /// Starts a plugin from a loaded module. Why it did not start is
    /// returned in English (for logs and the command line); the plugin's
    /// slot keeps it as data.
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
                self.add_inactive(info, PluginStatus::Failed(FailReason::Load(error)));
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
            legacy_config_dir: crate::paths::legacy_config_dir()
                .map(|dir| dir.to_string_lossy().into_owned()),
            headless: self.headless,
            language: Some(eov_plugin_api::i18n::current().tag.to_string()),
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
                if !info.slide_formats.is_empty() {
                    match module.slide_source(&instance) {
                        Some(Ok(source)) => {
                            slide_sources::restarted(&info.id);
                            slide_sources::register(
                                &info.id,
                                &info.name,
                                &info.slide_formats,
                                source,
                            );
                        }
                        Some(Err(error)) => tracing::warn!(
                            "Plugin '{}' declares slide formats but gave no slide source: {error}",
                            info.id
                        ),
                        None => tracing::warn!(
                            "Plugin '{}' declares slide formats but was built before slide sources were added; they are not available",
                            info.id
                        ),
                    }
                }
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
                    status: PluginStatus::Failed(FailReason::Load(error)),
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
        let usage: HashMap<Arc<str>, Arc<usage::PluginUsage>> = self
            .slots
            .iter()
            .filter_map(|slot| {
                let ctx = slot.ctx.as_ref()?;
                Some((Arc::clone(&ctx.id), Arc::clone(&ctx.usage)))
            })
            .collect();
        let contributions = self.shared.contributions.lock();
        self.filters.rebuild(&contributions, &instances, &usage);
    }

    /// What plugin `id` has cost since it started, in one line, or `None`
    /// when it is not running.
    pub fn usage_summary(&self, id: &str) -> Option<String> {
        let slot = self.slots.iter().find(|slot| slot.id() == id)?;
        let ctx = slot.ctx.as_ref().filter(|_| slot.is_active())?;
        let overlays = self
            .shared
            .overlays
            .owner_usage(&crate::overlay::store::Owner::Plugin(Arc::clone(&ctx.id)));
        Some(ctx.usage.snapshot().summary(&overlays))
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
            ctx.usage.record_event(&event_type, elapsed);
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
        if !slot.info.slide_formats.is_empty() {
            slide_sources::unregister(plugin_id);
        }
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

    /// A plugin reading `.<id>` files: a 600 × 400 gradient (the pixel at
    /// level `l`, (`x`, `y`) is `[x % 256, y % 256, 100·l, 255]`) with two
    /// levels; a read at x = 13 panics.
    struct Synth;

    struct SynthSlide;

    impl SlideReader for SynthSlide {
        fn info(&self) -> SlideInfo {
            SlideInfo {
                levels: vec![
                    SlideLevel {
                        width: 600,
                        height: 400,
                        ..Default::default()
                    },
                    SlideLevel {
                        width: 150,
                        height: 100,
                        ..Default::default()
                    },
                ],
                vendor: Some("synthetic".into()),
                mpp_x: Some(0.25),
                mpp_y: Some(0.25),
                ..Default::default()
            }
        }

        fn read_region(&self, region: &SlideRegion, out: &mut [u8]) -> Result<(), String> {
            if region.x == 13 {
                panic!("the decoder fell over");
            }
            for row in 0..u64::from(region.height) {
                for col in 0..u64::from(region.width) {
                    let at = ((row * u64::from(region.width) + col) * 4) as usize;
                    out[at..at + 4].copy_from_slice(&[
                        ((region.x + col) % 256) as u8,
                        ((region.y + row) % 256) as u8,
                        (region.level * 100) as u8,
                        255,
                    ]);
                }
            }
            Ok(())
        }
    }

    impl Plugin for Synth {
        fn create(_host: Host, _init: PluginInit) -> Result<Self, String> {
            Ok(Self)
        }

        fn open_slide(&self, request: &SlideOpenRequest) -> Result<Box<dyn SlideReader>, String> {
            std::fs::metadata(&request.location).map_err(|e| e.to_string())?;
            Ok(Box::new(SynthSlide))
        }
    }

    /// Starts `Synth` as plugin `id` reading `.<id>` files.
    fn start_synth(rt: &mut PluginRuntime, dir: &std::path::Path, id: &str) {
        let module = Module::from_ref(eov_plugin_api::sdk::root_module::<Synth>());
        rt.start(
            PluginInfo {
                id: id.into(),
                name: format!("Synth {id}"),
                version: "1.0.0".into(),
                root: dir.to_path_buf(),
                slide_formats: vec![eov_plugin_api::manifest::SlideFormatSpec {
                    id: "synthetic".into(),
                    name: "Synthetic slides".into(),
                    extensions: vec![id.to_ascii_uppercase()],
                    schemes: vec![],
                }],
                ..Default::default()
            },
            module,
            dir.join(id).join("data"),
            dir.join(id).join("cache"),
        )
        .unwrap();
    }

    #[test]
    fn slide_formats_of_plugins_open_through_wsi_file() {
        let (dir, mut rt) = runtime();
        let path = dir.path().join("slide.synthone");
        std::fs::write(&path, b"synthetic").unwrap();
        assert!(eov_common::WsiFile::open(&path).is_err(), "unknown before");
        start_synth(&mut rt, dir.path(), "synthone");
        assert!(eov_common::dataset::is_supported_slide_extension(&path));
        assert!(eov_common::dataset::slide_extensions().contains(&"synthone".to_string()));

        let wsi = eov_common::WsiFile::open(&path).unwrap();
        let props = wsi.properties();
        assert_eq!(
            (props.width, props.height, props.levels.len()),
            (600, 400, 2)
        );
        assert_eq!(props.vendor.as_deref(), Some("synthetic"));
        assert_eq!(props.levels[1].downsample, 4.0, "from the sizes");
        assert_eq!(
            wsi.read_region(20, 30, 0, 2, 1).unwrap()[..4],
            [20, 30, 0, 255]
        );
        assert_eq!(
            wsi.read_region(400, 80, 1, 1, 1).unwrap()[..4],
            [100, 20, 100, 255]
        );
        // Render workers read from several threads at once.
        std::thread::scope(|scope| {
            for t in 0..4i64 {
                let wsi = wsi.clone();
                scope.spawn(move || {
                    for i in 0..10i64 {
                        let x = 100 + t * 10 + i;
                        assert_eq!(
                            wsi.read_region(x, 7, 0, 1, 1).unwrap()[..4],
                            [(x % 256) as u8, 7, 0, 255]
                        );
                    }
                });
            }
        });

        // Disabled, the plugin opens nothing, and its slides fail fast.
        rt.disable("synthone");
        assert!(!eov_common::dataset::is_supported_slide_extension(&path));
        assert!(wsi.read_region(20, 30, 0, 2, 1).is_err());
    }

    /// The in-tree example plugin's checkerboard format, declared in its
    /// own plugin.toml, through the host as a window or `eov batch` uses it.
    #[test]
    fn the_example_plugins_format_opens_as_declared() {
        let manifest = eov_plugin_api::PluginManifest::from_toml(
            include_str!("../../../plugin_api/example-plugin/plugin.toml"),
            "example_plugin",
        )
        .unwrap();
        assert_eq!(manifest.slide_formats[0].extensions, ["excheck"]);
        let (dir, mut rt) = runtime();
        rt.start(
            PluginInfo {
                id: manifest.id.clone(),
                name: manifest.name.clone(),
                version: manifest.version.clone(),
                root: dir.path().to_path_buf(),
                slide_formats: manifest.slide_formats.clone(),
                ..Default::default()
            },
            Module::from_ref(example_plugin::module()),
            dir.path().join("data"),
            dir.path().join("cache"),
        )
        .unwrap();
        let path = dir.path().join("board.excheck");
        std::fs::write(&path, format!("{}\n", example_plugin::CHECKER_HEADER)).unwrap();
        let wsi = eov_common::WsiFile::open(&path).unwrap();
        assert_eq!(wsi.properties().levels.len(), 3);
        assert_eq!(wsi.properties().objective_power, Some(20.0));
        let pixels = wsi.read_region(255, 0, 0, 2, 1).unwrap();
        assert_eq!(pixels[..4], example_plugin::CheckerSlide::color(255, 0));
        assert_eq!(pixels[4..8], example_plugin::CheckerSlide::color(256, 0));
        // A file of another kind is refused with the plugin's reason.
        let other = dir.path().join("other.excheck");
        std::fs::write(&other, "not a board").unwrap();
        let error = eov_common::WsiFile::open(&other).unwrap_err().to_string();
        assert!(error.contains(example_plugin::CHECKER_HEADER), "{error}");
        rt.disable(&manifest.id);
    }

    #[test]
    fn a_slide_source_that_panics_is_reported_and_stopped() {
        let (dir, mut rt) = runtime();
        let path = dir.path().join("slide.synthtwo");
        std::fs::write(&path, b"synthetic").unwrap();
        start_synth(&mut rt, dir.path(), "synthtwo");
        let wsi = eov_common::WsiFile::open(&path).unwrap();
        let error = wsi.read_region(13, 0, 0, 1, 1).unwrap_err().to_string();
        assert!(error.contains("panicked"), "{error}");
        let panics = slide_sources::take_panics();
        assert!(
            panics
                .iter()
                .any(|(id, message)| &**id == "synthtwo" && message.contains("fell over")),
            "{panics:?}"
        );
        // Later reads fail at once; the UI thread quarantines the plugin.
        assert!(wsi.read_region(20, 30, 0, 2, 1).is_err());
        assert!(rt.quarantine("synthtwo", "its slide source panicked"));
        assert!(!eov_common::dataset::is_supported_slide_extension(&path));
    }

    /// Loads and starts real plugin libraries (`lib<id>.so` and the like)
    /// named in `EOV_TEST_PLUGIN_LIBS`, paths separated as in `PATH`:
    /// `EOV_TEST_PLUGIN_LIBS=… cargo test -p eov --bin eov real_plugin -- --ignored`.
    /// Libraries built before slide sources were added load without one.
    #[test]
    #[ignore = "needs EOV_TEST_PLUGIN_LIBS"]
    fn real_plugin_libraries_load_and_start() {
        let Some(paths) = std::env::var_os("EOV_TEST_PLUGIN_LIBS") else {
            return;
        };
        for path in std::env::split_paths(&paths) {
            let (dir, mut rt) = runtime();
            let module = loader::load_library(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            eprintln!(
                "{}: plugin API {}, built with slide sources: {}",
                path.display(),
                module.api_version(),
                module.has_slide_source()
            );
            rt.start(
                PluginInfo {
                    id: "real".into(),
                    name: "real".into(),
                    version: "1.0.0".into(),
                    root: dir.path().to_path_buf(),
                    ..Default::default()
                },
                Ok(module),
                dir.path().join("data"),
                dir.path().join("cache"),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert!(rt.is_active("real"));
        }
    }

    #[test]
    fn plugins_start_in_id_order_and_failures_are_recorded() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "zeta").unwrap();
        start(&mut rt, dir.path(), "alpha").unwrap();
        assert_eq!(
            start(&mut rt, dir.path(), "broken").unwrap_err(),
            "the plugin failed to start: cannot start"
        );
        let ids: Vec<_> = rt.slots().iter().map(|s| s.id().to_string()).collect();
        assert_eq!(ids, ["alpha", "broken", "zeta"]);
        assert_eq!(
            rt.slot("broken").unwrap().status,
            PluginStatus::Failed(FailReason::Load(LoadError::Create("cannot start".into())))
        );
        assert!(dir.path().join("alpha/data").is_dir());
    }

    /// Why a plugin failed is kept as data: the Plugins panel says all of it
    /// in the language of the moment (also after the language changed), the
    /// audit log in English, without isolation marks, as eov wrote it
    /// before it was translated.
    #[test]
    fn failure_reasons_are_worded_when_shown() {
        use crate::update::compat::{Incompatibility, RequirementError};
        use eov_plugin_api::i18n::{self, Catalog};
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let api = PluginStatus::Failed(FailReason::Api {
            required: ">=2.0".into(),
            provided: "1.1.0".into(),
        });
        let english = "Failed: requires plugin API >=2.0, this eov provides 1.1.0";
        assert_eq!(api.label(), english);
        assert_eq!(api.audit_text(), english);

        let _english = English;
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 2] = [
            (None, "Failed: {}", &["Fehlgeschlagen: {}"]),
            (
                None,
                "requires plugin API {required}, this eov provides {provided}",
                &["benötigt die Plugin-API {required}, dieses eov bietet {provided}"],
            ),
        ];
        i18n::set_catalog(
            german,
            Some(Catalog::from_messages(german, "", messages).unwrap()),
        );
        assert_eq!(
            api.label(),
            "Fehlgeschlagen: benötigt die Plugin-API >=2.0, dieses eov bietet 1.1.0"
        );
        assert_eq!(api.audit_text(), english);

        // Right to left: the panel keeps the versions in order, the audit
        // log has no marks.
        i18n::set_catalog(i18n::by_tag("ar").unwrap(), None);
        assert!(
            api.label().contains("\u{2068}>=2.0\u{2069}"),
            "{}",
            api.label()
        );
        let failures = [
            api,
            PluginStatus::Failed(FailReason::Incompatible(Some(Incompatibility::Eov {
                required: ">=0.4.1".into(),
                bounded: Some(">=0.4.1, <0.5.0".into()),
                version: "1.0.0".into(),
            }))),
            PluginStatus::Failed(FailReason::Incompatible(Some(Incompatibility::InvalidApi(
                RequirementError::NoLowerBound("*".into()),
            )))),
            PluginStatus::Failed(FailReason::Load(LoadError::Incompatible(
                loader::Incompatible::Api {
                    plugin: "0.4.9".into(),
                    host: "1.1.0".into(),
                },
            ))),
            PluginStatus::Failed(FailReason::Load(LoadError::BadPath(
                loader::BadPath::OutsidePackage("libx.so".into()),
            ))),
            PluginStatus::Failed(FailReason::NotAllowed {
                source: Some(PathBuf::from("/etc/eov/policy.toml")),
                invalid: None,
                contact: Some("Ask the lab's IT.".into()),
            }),
        ];
        let audit: Vec<String> = failures.iter().map(PluginStatus::audit_text).collect();
        assert_eq!(
            audit,
            [
                english,
                "Failed: requires eov >=0.4.1 (bounded to >=0.4.1, <0.5.0); this is eov 1.0.0",
                "Failed: invalid plugin API requirement: '*' has no lower bound; declare a \
                 range such as \">=1.0.0, <2.0.0\"",
                "Failed: incompatible plugin: built for plugin API 0.4.9, this eov provides \
                 1.1.0",
                "Failed: invalid library path: libx.so resolves outside the package",
                "Failed: Loading this plugin is not allowed: the administrator policy \
                 /etc/eov/policy.toml only permits official eov plugins.\nAsk the lab's IT.",
            ]
        );
        for status in &failures {
            assert!(status.label().contains('\u{2068}'), "{}", status.label());
        }
    }

    /// Why a package could not be unpacked is worded when it is shown, like
    /// the other reasons: a reason found while the interface was in English
    /// is in German once the interface is, and stays English, without marks,
    /// for the audit log.
    #[test]
    fn unpack_reasons_follow_a_change_of_language() {
        use crate::plugins::discovery::package_cache::{PackageError, validate_archive_path};
        use eov_plugin_api::i18n::{self, Catalog};
        use std::path::Path;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let error =
            validate_archive_path(Path::new("/plugins/x.eop"), Path::new("../escape")).unwrap_err();
        let status = PluginStatus::Failed(FailReason::Unpack(error.reason()));
        let english = "Failed: could not unpack the package: plugin package '/plugins/x.eop' \
                       contains an unsafe path '../escape'";
        assert_eq!(status.label(), english);

        let _english = English;
        let german = i18n::by_tag("de").unwrap();
        let messages: [(Option<&str>, &str, &[&str]); 3] = [
            (None, "Failed: {}", &["Fehlgeschlagen: {}"]),
            (
                None,
                "could not unpack the package: {}",
                &["das Paket konnte nicht entpackt werden: {}"],
            ),
            (
                None,
                "plugin package '{package}' contains an unsafe path '{path}'",
                &["Plugin-Paket „{package}“ enthält einen unsicheren Pfad „{path}“"],
            ),
        ];
        i18n::set_catalog(
            german,
            Some(Catalog::from_messages(german, "", messages).unwrap()),
        );
        assert_eq!(
            status.label(),
            "Fehlgeschlagen: das Paket konnte nicht entpackt werden: Plugin-Paket \
             „/plugins/x.eop“ enthält einen unsicheren Pfad „../escape“"
        );
        assert_eq!(status.audit_text(), english);
        i18n::set_catalog(i18n::by_tag("ar").unwrap(), None);
        assert!(status.label().contains("\u{2068}../escape\u{2069}"));
        assert_eq!(status.audit_text(), english);
        // What the system said is the same in both.
        let io = PackageError::from(std::io::Error::other("gone"));
        assert_eq!(
            FailReason::Unpack(io.reason()).to_string(),
            "could not unpack the package: IO error: gone"
        );
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
    fn the_time_plugins_take_is_counted() {
        let (dir, mut rt) = runtime();
        start(&mut rt, dir.path(), "echo_usage").unwrap();
        assert_eq!(
            rt.usage_summary("echo_usage").as_deref(),
            Some("No usage recorded yet")
        );
        for _ in 0..3 {
            rt.deliver(
                &EventTarget::All,
                &HostEvent::Ready {
                    restore_sidebar: false,
                },
            );
        }
        let summary = rt.usage_summary("echo_usage").unwrap();
        assert!(summary.starts_with("UI thread "), "{summary}");
        assert!(summary.contains(" in 3 events ("), "{summary}");
        // Plugins that are not running have nothing to show.
        assert!(rt.usage_summary("nobody").is_none());
        rt.disable("echo_usage");
        assert!(rt.usage_summary("echo_usage").is_none());
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
