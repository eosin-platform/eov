//! State shared between the UI thread and every plugin's host handle.
//!
//! Plugins call the host from any thread. Everything here is either
//! lock-free (published snapshots), guarded by short-lived locks that are
//! never held while calling plugin code, or a queue that the UI thread
//! drains. No host call ever waits for the UI thread.

use super::contributions::Contributions;
use super::settings::SettingsStore;
use crate::overlay::OverlayStore;
use arc_swap::ArcSwap;
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::protocol::{
    ConfirmArgs, FileDialogArgs, FileId, FileInfo, HostSnapshot, JobFinishArgs, JobStartArgs,
    JobUpdateArgs, MeasurementInfo, NotifyArgs, PaneId, PropertyMap, Rect, SeriesAction,
    SurfaceSpec, UndoClearArgs, UndoPushArgs, ViewFrameArgs, ViewPanArgs, ViewSetArgs,
    ViewZoomArgs,
};
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Reads pixels of one open slide for plugins.
pub trait SlideReader: Send + Sync {
    /// Returns `width·height·4` bytes of RGBA, honouring
    /// `request.background` (alpha 0 = straight alpha, else composited).
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String>;
    /// Decodes an associated image as opaque RGBA: `(width, height, data)`.
    fn associated_image(&self, name: &str) -> Result<(u32, u32, Vec<u8>), String>;
}

/// What plugins may know about one open file beyond [`FileInfo`].
#[derive(Clone, Default)]
pub struct PublishedFile {
    pub info: FileInfo,
    pub properties: Arc<PropertyMap>,
    pub roi: Option<Rect>,
    pub measurements: Vec<MeasurementInfo>,
}

/// A consistent copy of the host state, replaced wholesale by the UI thread.
#[derive(Clone, Default)]
pub struct Published {
    pub snapshot: HostSnapshot,
    pub files: HashMap<FileId, PublishedFile>,
}

/// A plugin's identity and revocable access.
pub struct PluginContext {
    pub id: Arc<str>,
    /// The extracted package; surfaces and assets resolve inside it.
    pub root: PathBuf,
    revoked: AtomicBool,
}

impl PluginContext {
    pub fn new(id: &str, root: PathBuf) -> Self {
        Self {
            id: Arc::from(id),
            root,
            revoked: AtomicBool::new(false),
        }
    }

    pub fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
    }

    pub fn is_revoked(&self) -> bool {
        self.revoked.load(Ordering::Acquire)
    }
}

/// Requests a plugin's surface makes of the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceCommand {
    Open {
        surface_id: String,
        spec: SurfaceSpec,
    },
    Set {
        surface_id: String,
        properties: Map<String, Value>,
    },
    Close {
        surface_id: String,
    },
}

/// Navigation requests.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewCommand {
    Set(ViewSetArgs),
    Pan(ViewPanArgs),
    Zoom(ViewZoomArgs),
    Fit(Option<PaneId>),
    Frame(ViewFrameArgs),
}

/// A UI mutation queued by a host call, applied on the UI thread in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Notify(NotifyArgs),
    Status(Option<String>),
    OpenFile(PathBuf),
    /// Toolbar, HUD, tool or filter registrations changed.
    ContributionsChanged,
    ActivateTool(String),
    DeactivateTools,
    Surface(SurfaceCommand),
    Confirm {
        request_id: String,
        args: ConfirmArgs,
    },
    FileDialog {
        request_id: String,
        args: FileDialogArgs,
        save: bool,
    },
    View(ViewCommand),
    CyclePane(i32),
    CycleTab(i32),
    Series(SeriesAction),
    Render,
    UndoPush(UndoPushArgs),
    UndoClear(UndoClearArgs),
    JobStart {
        job_id: String,
        args: JobStartArgs,
    },
    JobUpdate(JobUpdateArgs),
    JobFinish(JobFinishArgs),
    Clipboard(String),
    Post(String),
    /// The plugin's overlay layers changed; redraw.
    OverlaysChanged,
}

type Wake = Box<dyn Fn() + Send + Sync>;

/// The host state every plugin handle shares.
pub struct HostShared {
    published: ArcSwap<Published>,
    pub overlays: Arc<OverlayStore>,
    pub contributions: Mutex<Contributions>,
    pub settings: SettingsStore,
    readers: RwLock<HashMap<FileId, Arc<dyn SlideReader>>>,
    /// Open surfaces: id → owning plugin.
    surfaces: Mutex<HashMap<String, Arc<str>>>,
    /// Events each plugin asked for (`None` = all).
    subscriptions: Mutex<HashMap<Arc<str>, Option<HashSet<String>>>>,
    commands: Mutex<VecDeque<(Arc<str>, Command)>>,
    wake: Mutex<Option<Wake>>,
    wake_scheduled: AtomicBool,
    next_id: AtomicU64,
    pub host_version: String,
}

impl HostShared {
    pub fn new(overlays: Arc<OverlayStore>, settings: SettingsStore, host_version: &str) -> Self {
        Self {
            published: ArcSwap::from_pointee(Published::default()),
            overlays,
            contributions: Mutex::new(Contributions::default()),
            settings,
            readers: RwLock::new(HashMap::new()),
            surfaces: Mutex::new(HashMap::new()),
            subscriptions: Mutex::new(HashMap::new()),
            commands: Mutex::new(VecDeque::new()),
            wake: Mutex::new(None),
            wake_scheduled: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            host_version: host_version.to_string(),
        }
    }

    /// Installs the function that schedules [`Self::take_commands`] on the UI
    /// thread (e.g. via `slint::invoke_from_event_loop`).
    pub fn set_wake(&self, wake: impl Fn() + Send + Sync + 'static) {
        *self.wake.lock() = Some(Box::new(wake));
    }

    /// Queues a UI mutation and makes sure a drain is scheduled.
    pub fn push(&self, plugin_id: &Arc<str>, command: Command) {
        self.commands
            .lock()
            .push_back((Arc::clone(plugin_id), command));
        self.wake();
    }

    pub fn wake(&self) {
        if !self.wake_scheduled.swap(true, Ordering::AcqRel)
            && let Some(wake) = self.wake.lock().as_ref()
        {
            wake();
        }
    }

    /// Takes every queued command (UI thread).
    pub fn take_commands(&self) -> Vec<(Arc<str>, Command)> {
        self.wake_scheduled.store(false, Ordering::Release);
        self.commands.lock().drain(..).collect()
    }

    pub fn has_pending_commands(&self) -> bool {
        !self.commands.lock().is_empty()
    }

    pub fn fresh_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    pub fn published(&self) -> Arc<Published> {
        self.published.load_full()
    }

    pub fn publish(&self, published: Published) {
        self.published.store(Arc::new(published));
    }

    pub fn set_reader(&self, file_id: FileId, reader: Arc<dyn SlideReader>) {
        self.readers.write().insert(file_id, reader);
    }

    pub fn remove_reader(&self, file_id: FileId) {
        self.readers.write().remove(&file_id);
    }

    pub fn reader(&self, file_id: FileId) -> Option<Arc<dyn SlideReader>> {
        self.readers.read().get(&file_id).cloned()
    }

    pub fn register_surface(&self, surface_id: &str, plugin_id: &Arc<str>) {
        self.surfaces
            .lock()
            .insert(surface_id.to_string(), Arc::clone(plugin_id));
    }

    pub fn surface_owner(&self, surface_id: &str) -> Option<Arc<str>> {
        self.surfaces.lock().get(surface_id).cloned()
    }

    pub fn unregister_surface(&self, surface_id: &str) -> Option<Arc<str>> {
        self.surfaces.lock().remove(surface_id)
    }

    pub fn set_subscriptions(&self, plugin_id: &Arc<str>, events: Option<HashSet<String>>) {
        self.subscriptions
            .lock()
            .insert(Arc::clone(plugin_id), events);
    }

    /// Whether `plugin_id` wants events tagged `event_type`.
    pub fn wants(&self, plugin_id: &str, event_type: &str) -> bool {
        match self.subscriptions.lock().get(plugin_id) {
            Some(Some(events)) => events.contains(event_type),
            _ => true,
        }
    }

    /// Forgets everything a plugin registered (it was unloaded or disabled).
    pub fn forget_plugin(&self, plugin_id: &str) {
        self.surfaces
            .lock()
            .retain(|_, owner| &**owner != plugin_id);
        self.subscriptions.lock().remove(plugin_id);
        self.commands
            .lock()
            .retain(|(owner, _)| &**owner != plugin_id);
        self.contributions.lock().remove_plugin(plugin_id);
        self.overlays
            .remove_owner(&crate::overlay::Owner::plugin(plugin_id));
    }
}
