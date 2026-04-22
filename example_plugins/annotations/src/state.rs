use abi_stable::std_types::{ROption, RString};
use plugin_api::ffi::{
    HostApiVTable, HostLogLevelFFI, HostSnapshotFFI, OpenFileInfoFFI, ViewportSnapshotFFI,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::model::LoadedFileAnnotations;

#[derive(Default)]
pub(crate) struct PluginState {
    pub(crate) files: HashMap<String, LoadedFileAnnotations>,
    pub(crate) active_file_path: Option<String>,
    pub(crate) active_filename: Option<String>,
    pub(crate) selected_set_by_file: HashMap<String, String>,
    pub(crate) editing_set_by_file: HashMap<String, String>,
    pub(crate) collapsed_sets_by_file: HashMap<String, HashSet<String>>,
    pub(crate) hidden_sets_by_file: HashMap<String, HashSet<String>>,
}

static HOST_API: Mutex<Option<HostApiVTable>> = Mutex::new(None);
static PLUGIN_STATE: OnceLock<Mutex<PluginState>> = OnceLock::new();

pub(crate) fn plugin_state() -> &'static Mutex<PluginState> {
    PLUGIN_STATE.get_or_init(|| Mutex::new(PluginState::default()))
}

pub(crate) fn host_api() -> Option<HostApiVTable> {
    *HOST_API.lock().unwrap()
}

pub(crate) fn set_host_api(host_api: HostApiVTable) {
    *HOST_API.lock().unwrap() = Some(host_api);
}

pub(crate) fn log_message(level: HostLogLevelFFI, message: impl Into<String>) {
    if let Some(host_api) = host_api() {
        (host_api.log_message)(host_api.context, level, RString::from(message.into()));
    }
}

pub(crate) fn host_snapshot() -> Result<HostSnapshotFFI, String> {
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    Ok((host_api.get_snapshot)(host_api.context))
}

pub(crate) fn active_file_from_snapshot(snapshot: &HostSnapshotFFI) -> Option<OpenFileInfoFFI> {
    match &snapshot.active_file {
        ROption::RSome(file) => Some(file.clone()),
        ROption::RNone => None,
    }
}

pub(crate) fn active_viewport_from_snapshot(
    snapshot: &HostSnapshotFFI,
) -> Option<ViewportSnapshotFFI> {
    match &snapshot.active_viewport {
        ROption::RSome(viewport) => Some(viewport.clone()),
        ROption::RNone => None,
    }
}

pub(crate) fn active_file_key(state: &PluginState) -> Option<&str> {
    state.active_file_path.as_deref()
}
