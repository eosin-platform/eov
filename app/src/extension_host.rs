//! gRPC extension host server.
//!
//! When launched with `--extension-host-port`, the app starts a Tonic gRPC
//! server implementing the `ExtensionHost` service. Remote plugins (Python,
//! etc.) connect to this server to register viewport filters and exchange
//! pixel data.

use crate::eov_extension::extension_host_server::{ExtensionHost, ExtensionHostServer};
use crate::eov_extension::{
    ApplyFilterCpuRequest, ApplyFilterCpuResponse, ApplyFilterGpuRequest, ApplyFilterGpuResponse,
    Empty, FrameActiveRectRequest, HideSidebarRequest, HostCommandResponse, HostSnapshot,
    HudToolbarActionRequest, HudToolbarActionStreamRequest, LogLevel, LogMessageRequest,
    OpenFileInfo, OpenFileRequest, ReadRegionRequest, ReadRegionResponse, RegisterFilterRequest,
    RegisterFilterResponse, RegisterHudToolbarButtonRequest, RegisterPluginRequest,
    RegisterPluginResponse, RegisterToolbarButtonRequest, SetActiveViewportRequest,
    SetFilterEnabledRequest, SetFilterEnabledResponse, SetHudToolbarButtonActiveRequest,
    SetToolbarButtonActiveRequest, ShowSidebarRequest, ToolbarActionRequest,
    ToolbarActionStreamRequest, UnregisterFilterRequest, UnregisterFilterResponse,
    UnregisterHudToolbarButtonRequest, UnregisterPluginRequest, UnregisterToolbarButtonRequest,
    ViewportSnapshot,
};
use parking_lot::RwLock;
use plugin_api::PluginError;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info, warn};

const MAX_GRPC_FILTER_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct RemotePlugin {
    pub plugin_id: String,
    pub display_name: String,
    pub version: String,
    pub language: String,
    pub plugin_root: Option<PathBuf>,
    pub filter_ids: HashSet<String>,
}

/// A remotely-registered viewport filter (from a gRPC client).
pub(crate) struct RemoteFilter {
    pub owner_handle: String,
    pub name: String,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub enabled: bool,
    /// Channel to send CPU filter requests to the connected plugin.
    pub cpu_request_tx: Option<mpsc::Sender<CpuFilterRequest>>,
}

/// A CPU filter request sent to a remote plugin via gRPC streaming.
pub(crate) struct CpuFilterRequest {
    pub width: u32,
    pub height: u32,
    pub rgba_data: Vec<u8>,
    pub response_tx: std::sync::mpsc::Sender<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct RemoteToolbarButton {
    pub plugin_handle: String,
    pub plugin_id: String,
    pub button_id: String,
    pub tooltip: String,
    pub icon_svg: String,
    pub action_id: String,
    pub active: bool,
}

#[derive(Clone)]
pub(crate) struct RemoteHudToolbarButton {
    pub plugin_handle: String,
    pub plugin_id: String,
    pub button_id: String,
    pub tooltip: String,
    pub icon_svg: String,
    pub action_id: String,
    pub active: bool,
}

pub(crate) struct RemoteToolbarActionRequest {
    pub plugin_handle: String,
    pub plugin_id: String,
    pub button_id: String,
    pub action_id: String,
}

pub(crate) struct RemoteHudToolbarActionRequest {
    pub plugin_handle: String,
    pub plugin_id: String,
    pub button_id: String,
    pub action_id: String,
    pub viewport: plugin_api::ViewportSnapshot,
}

/// Shared state for the extension host, accessible from both the gRPC server
/// and the render pipeline.
pub(crate) struct ExtensionHostState {
    pub remote_plugins: HashMap<String, RemotePlugin>,
    pub filters: HashMap<String, RemoteFilter>,
    pub toolbar_buttons: Vec<RemoteToolbarButton>,
    pub toolbar_action_txs: HashMap<String, mpsc::Sender<RemoteToolbarActionRequest>>,
    pub hud_toolbar_buttons: Vec<RemoteHudToolbarButton>,
    pub hud_toolbar_action_txs: HashMap<String, mpsc::Sender<RemoteHudToolbarActionRequest>>,
}

impl ExtensionHostState {
    pub fn new() -> Self {
        Self {
            remote_plugins: HashMap::new(),
            filters: HashMap::new(),
            toolbar_buttons: Vec::new(),
            toolbar_action_txs: HashMap::new(),
            hud_toolbar_buttons: Vec::new(),
            hud_toolbar_action_txs: HashMap::new(),
        }
    }

    pub fn register_plugin(
        &mut self,
        plugin_id: String,
        display_name: String,
        version: String,
        language: String,
        plugin_root: Option<PathBuf>,
    ) -> Result<String, String> {
        if self
            .remote_plugins
            .values()
            .any(|plugin| plugin.plugin_id == plugin_id)
        {
            return Err(format!(
                "remote plugin id '{plugin_id}' is already registered"
            ));
        }

        let plugin_handle = uuid::Uuid::new_v4().to_string();
        self.remote_plugins.insert(
            plugin_handle.clone(),
            RemotePlugin {
                plugin_id,
                display_name,
                version,
                language,
                plugin_root,
                filter_ids: HashSet::new(),
            },
        );
        Ok(plugin_handle)
    }

    pub fn plugin(&self, plugin_handle: &str) -> Option<&RemotePlugin> {
        self.remote_plugins.get(plugin_handle)
    }

    pub fn plugin_mut(&mut self, plugin_handle: &str) -> Option<&mut RemotePlugin> {
        self.remote_plugins.get_mut(plugin_handle)
    }

    pub fn unregister_plugin(&mut self, plugin_handle: &str) -> Result<RemotePlugin, String> {
        let plugin = self
            .remote_plugins
            .remove(plugin_handle)
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;

        self.filters
            .retain(|_, filter| filter.owner_handle != plugin_handle);
        self.toolbar_buttons
            .retain(|button| button.plugin_handle != plugin_handle);
        self.hud_toolbar_buttons
            .retain(|button| button.plugin_handle != plugin_handle);
        self.toolbar_action_txs.remove(plugin_handle);
        self.hud_toolbar_action_txs.remove(plugin_handle);

        Ok(plugin)
    }

    /// Toggle the enabled state of all registered remote filters.
    pub fn toggle_all_filters(&mut self) {
        for filter in self.filters.values_mut() {
            filter.enabled = !filter.enabled;
            info!(
                "Toggled remote filter '{}' -> enabled={}",
                filter.name, filter.enabled
            );
        }
    }

    pub fn register_toolbar_button(
        &mut self,
        plugin_handle: &str,
        button_id: String,
        tooltip: String,
        icon_svg: String,
        action_id: String,
    ) -> Result<String, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let duplicate = self.toolbar_buttons.iter().any(|existing| {
            existing.plugin_handle == plugin_handle && existing.button_id == button_id
        });
        if duplicate {
            return Err(
                PluginError::DuplicateButtonId(format!("{plugin_id}:{button_id}")).to_string(),
            );
        }

        self.toolbar_buttons.push(RemoteToolbarButton {
            plugin_handle: plugin_handle.to_string(),
            plugin_id: plugin_id.clone(),
            button_id,
            tooltip,
            icon_svg,
            action_id,
            active: false,
        });
        Ok(plugin_id)
    }

    pub fn register_hud_toolbar_button(
        &mut self,
        plugin_handle: &str,
        button_id: String,
        tooltip: String,
        icon_svg: String,
        action_id: String,
    ) -> Result<String, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let duplicate = self.hud_toolbar_buttons.iter().any(|existing| {
            existing.plugin_handle == plugin_handle && existing.button_id == button_id
        });
        if duplicate {
            return Err(
                PluginError::DuplicateButtonId(format!("{plugin_id}:{button_id}")).to_string(),
            );
        }

        self.hud_toolbar_buttons.push(RemoteHudToolbarButton {
            plugin_handle: plugin_handle.to_string(),
            plugin_id: plugin_id.clone(),
            button_id,
            tooltip,
            icon_svg,
            action_id,
            active: false,
        });
        Ok(plugin_id)
    }

    pub fn set_toolbar_button_active(
        &mut self,
        plugin_handle: &str,
        button_id: &str,
        active: bool,
    ) -> Result<String, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let button = self
            .toolbar_buttons
            .iter_mut()
            .find(|button| button.plugin_handle == plugin_handle && button.button_id == button_id)
            .ok_or_else(|| {
                format!("remote toolbar button '{plugin_id}:{button_id}' is not registered")
            })?;
        button.active = active;
        Ok(plugin_id)
    }

    pub fn set_hud_toolbar_button_active(
        &mut self,
        plugin_handle: &str,
        button_id: &str,
        active: bool,
    ) -> Result<String, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let button = self
            .hud_toolbar_buttons
            .iter_mut()
            .find(|button| button.plugin_handle == plugin_handle && button.button_id == button_id)
            .ok_or_else(|| {
                format!("remote HUD toolbar button '{plugin_id}:{button_id}' is not registered")
            })?;
        button.active = active;
        Ok(plugin_id)
    }

    pub fn unregister_toolbar_button(
        &mut self,
        plugin_handle: &str,
        button_id: &str,
    ) -> Result<Option<String>, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let before = self.toolbar_buttons.len();
        self.toolbar_buttons.retain(|button| {
            !(button.plugin_handle == plugin_handle && button.button_id == button_id)
        });
        Ok((before != self.toolbar_buttons.len()).then_some(plugin_id))
    }

    pub fn unregister_hud_toolbar_button(
        &mut self,
        plugin_handle: &str,
        button_id: &str,
    ) -> Result<Option<String>, String> {
        let plugin_id = self
            .plugin(plugin_handle)
            .map(|plugin| plugin.plugin_id.clone())
            .ok_or_else(|| format!("remote plugin handle '{plugin_handle}' is not registered"))?;
        let before = self.hud_toolbar_buttons.len();
        self.hud_toolbar_buttons.retain(|button| {
            !(button.plugin_handle == plugin_handle && button.button_id == button_id)
        });
        Ok((before != self.hud_toolbar_buttons.len()).then_some(plugin_id))
    }

    pub fn has_toolbar_action(&self, plugin_id: &str, action_id: &str) -> bool {
        self.toolbar_buttons
            .iter()
            .any(|button| button.plugin_id == plugin_id && button.action_id == action_id)
    }

    pub fn dispatch_toolbar_action(
        &self,
        plugin_id: &str,
        action_id: &str,
    ) -> Result<bool, String> {
        let Some(button) = self
            .toolbar_buttons
            .iter()
            .find(|button| button.plugin_id == plugin_id && button.action_id == action_id)
            .cloned()
        else {
            return Ok(false);
        };

        let Some(tx) = self.toolbar_action_txs.get(&button.plugin_handle) else {
            return Err(format!(
                "remote toolbar action stream for plugin '{plugin_id}' is not connected"
            ));
        };

        tx.blocking_send(RemoteToolbarActionRequest {
            plugin_handle: button.plugin_handle,
            plugin_id: button.plugin_id,
            button_id: button.button_id,
            action_id: button.action_id,
        })
        .map_err(|err| format!("failed to dispatch remote toolbar action: {err}"))?;
        Ok(true)
    }

    pub fn has_hud_toolbar_action(&self, plugin_id: &str, action_id: &str) -> bool {
        self.hud_toolbar_buttons
            .iter()
            .any(|button| button.plugin_id == plugin_id && button.action_id == action_id)
    }

    pub fn dispatch_hud_toolbar_action(
        &self,
        plugin_id: &str,
        action_id: &str,
        viewport: plugin_api::ViewportSnapshot,
    ) -> Result<bool, String> {
        let Some(button) = self
            .hud_toolbar_buttons
            .iter()
            .find(|button| button.plugin_id == plugin_id && button.action_id == action_id)
            .cloned()
        else {
            return Ok(false);
        };

        let Some(tx) = self.hud_toolbar_action_txs.get(&button.plugin_handle) else {
            return Err(format!(
                "remote HUD toolbar action stream for plugin '{plugin_id}' is not connected"
            ));
        };

        tx.blocking_send(RemoteHudToolbarActionRequest {
            plugin_handle: button.plugin_handle,
            plugin_id: button.plugin_id,
            button_id: button.button_id,
            action_id: button.action_id,
            viewport,
        })
        .map_err(|err| format!("failed to dispatch remote HUD toolbar action: {err}"))?;
        Ok(true)
    }
}

/// Thread-safe handle to extension host state.
pub(crate) type SharedExtensionHostState = Arc<RwLock<ExtensionHostState>>;

pub(crate) fn new_shared_state() -> SharedExtensionHostState {
    Arc::new(RwLock::new(ExtensionHostState::new()))
}

/// The gRPC service implementation.
pub(crate) struct ExtensionHostService {
    state: SharedExtensionHostState,
    app_state: Arc<RwLock<crate::state::AppState>>,
}

impl ExtensionHostService {
    pub fn new(
        state: SharedExtensionHostState,
        app_state: Arc<RwLock<crate::state::AppState>>,
    ) -> Self {
        Self { state, app_state }
    }
}

#[tonic::async_trait]
impl ExtensionHost for ExtensionHostService {
    async fn register_plugin(
        &self,
        request: Request<RegisterPluginRequest>,
    ) -> Result<Response<RegisterPluginResponse>, Status> {
        let req = request.into_inner();
        if req.plugin_id.trim().is_empty() {
            return Err(Status::invalid_argument("plugin_id is required"));
        }

        let display_name = if req.display_name.trim().is_empty() {
            req.plugin_id.clone()
        } else {
            req.display_name
        };
        let version = req.version.clone();
        let language = req.language.clone();
        let plugin_root =
            (!req.plugin_root.trim().is_empty()).then(|| PathBuf::from(&req.plugin_root));

        let plugin_handle = {
            let mut state = self.state.write();
            state
                .register_plugin(
                    req.plugin_id.clone(),
                    display_name.clone(),
                    version.clone(),
                    language.clone(),
                    plugin_root,
                )
                .map_err(Status::already_exists)?
        };

        info!(
            "Registered remote plugin '{}' ({plugin_handle}) display='{}' version='{}' language='{}'",
            req.plugin_id, display_name, version, language
        );

        Ok(Response::new(RegisterPluginResponse {
            plugin_handle,
            plugin_id: req.plugin_id,
            host_snapshot: Some(host_snapshot_response(&self.app_state)),
        }))
    }

    async fn unregister_plugin(
        &self,
        request: Request<UnregisterPluginRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin = {
            let mut state = self.state.write();
            state
                .unregister_plugin(&req.plugin_handle)
                .map_err(Status::not_found)?
        };

        let _ = crate::plugin_host::hide_sidebar(&plugin.plugin_id);

        crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
        info!(
            "Unregistered remote plugin '{}' ({}) display='{}' version='{}' language='{}'",
            plugin.plugin_id,
            req.plugin_handle,
            plugin.display_name,
            plugin.version,
            plugin.language
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn register_toolbar_button(
        &self,
        request: Request<RegisterToolbarButtonRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let mut state = self.state.write();
            state
                .register_toolbar_button(
                    &req.plugin_handle,
                    req.button_id.clone(),
                    req.tooltip,
                    req.icon_svg,
                    req.action_id,
                )
                .map_err(Status::failed_precondition)?
        };
        crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
        info!(
            "Registered remote toolbar button '{}:{}'",
            plugin_id, req.button_id
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn unregister_toolbar_button(
        &self,
        request: Request<UnregisterToolbarButtonRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let removed = {
            let mut state = self.state.write();
            state
                .unregister_toolbar_button(&req.plugin_handle, &req.button_id)
                .map_err(Status::failed_precondition)?
        };
        if let Some(plugin_id) = removed {
            crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
            info!(
                "Unregistered remote toolbar button '{}:{}'",
                plugin_id, req.button_id
            );
        }
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn set_toolbar_button_active(
        &self,
        request: Request<SetToolbarButtonActiveRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let mut state = self.state.write();
            state
                .set_toolbar_button_active(&req.plugin_handle, &req.button_id, req.active)
                .map_err(Status::failed_precondition)?
        };
        crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
        info!(
            "Updated remote toolbar button active state '{}:{}' -> {}",
            plugin_id, req.button_id, req.active
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn show_sidebar(
        &self,
        request: Request<ShowSidebarRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let (plugin_id, plugin_root) = {
            let state = self.state.read();
            let plugin = state
                .plugin(&req.plugin_handle)
                .ok_or_else(|| Status::not_found("remote plugin handle is not registered"))?;
            (plugin.plugin_id.clone(), plugin.plugin_root.clone())
        };

        crate::plugin_host::show_sidebar(
            &plugin_id,
            plugin_root.as_deref(),
            None,
            plugin_api::SidebarRequest {
                button_id: (!req.button_id.trim().is_empty()).then_some(req.button_id),
                width_px: req.width_px,
                ui_path: req.ui_path,
                component: req.component,
            },
        )
        .map_err(Status::failed_precondition)?;

        info!(
            "Updated remote sidebar '{}': show/toggle request applied",
            plugin_id
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn hide_sidebar(
        &self,
        request: Request<HideSidebarRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let state = self.state.read();
            state
                .plugin(&req.plugin_handle)
                .map(|plugin| plugin.plugin_id.clone())
                .ok_or_else(|| Status::not_found("remote plugin handle is not registered"))?
        };

        crate::plugin_host::hide_sidebar(&plugin_id).map_err(Status::failed_precondition)?;
        info!(
            "Updated remote sidebar '{}': hide request applied",
            plugin_id
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn register_hud_toolbar_button(
        &self,
        request: Request<RegisterHudToolbarButtonRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let mut state = self.state.write();
            state
                .register_hud_toolbar_button(
                    &req.plugin_handle,
                    req.button_id.clone(),
                    req.tooltip,
                    req.icon_svg,
                    req.action_id,
                )
                .map_err(Status::failed_precondition)?
        };
        crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
        info!(
            "Registered remote HUD toolbar button '{}:{}'",
            plugin_id, req.button_id
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn unregister_hud_toolbar_button(
        &self,
        request: Request<UnregisterHudToolbarButtonRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let removed = {
            let mut state = self.state.write();
            state
                .unregister_hud_toolbar_button(&req.plugin_handle, &req.button_id)
                .map_err(Status::failed_precondition)?
        };
        if let Some(plugin_id) = removed {
            crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
            info!(
                "Unregistered remote HUD toolbar button '{}:{}'",
                plugin_id, req.button_id
            );
        }
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn set_hud_toolbar_button_active(
        &self,
        request: Request<SetHudToolbarButtonActiveRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let mut state = self.state.write();
            state
                .set_hud_toolbar_button_active(&req.plugin_handle, &req.button_id, req.active)
                .map_err(Status::failed_precondition)?
        };
        crate::plugin_host::refresh_plugin_buttons().map_err(Status::failed_precondition)?;
        info!(
            "Updated remote HUD toolbar button active state '{}:{}' -> {}",
            plugin_id, req.button_id, req.active
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    type ToolbarActionStreamStream = Pin<
        Box<dyn futures_core::Stream<Item = Result<ToolbarActionRequest, Status>> + Send + 'static>,
    >;

    type HudToolbarActionStreamStream = Pin<
        Box<
            dyn futures_core::Stream<Item = Result<HudToolbarActionRequest, Status>>
                + Send
                + 'static,
        >,
    >;

    async fn toolbar_action_stream(
        &self,
        request: Request<ToolbarActionStreamRequest>,
    ) -> Result<Response<Self::ToolbarActionStreamStream>, Status> {
        let req = request.into_inner();
        let plugin = {
            let state = self.state.read();
            state.plugin(&req.plugin_handle).cloned().ok_or_else(|| {
                Status::not_found(format!(
                    "remote plugin handle '{}' is not registered",
                    req.plugin_handle
                ))
            })?
        };

        let (out_tx, out_rx) = mpsc::channel::<Result<ToolbarActionRequest, Status>>(8);

        let (action_tx, mut action_rx) = mpsc::channel::<RemoteToolbarActionRequest>(8);
        {
            let mut state = self.state.write();
            if state.toolbar_action_txs.contains_key(&req.plugin_handle) {
                return Err(Status::already_exists(
                    "toolbar action stream is already connected for this plugin",
                ));
            }
            state
                .toolbar_action_txs
                .insert(req.plugin_handle.clone(), action_tx);
        }

        let shared_state = self.state.clone();
        let plugin_handle = req.plugin_handle.clone();
        tokio::spawn(async move {
            while let Some(action) = action_rx.recv().await {
                if out_tx
                    .send(Ok(ToolbarActionRequest {
                        plugin_handle: action.plugin_handle,
                        plugin_id: action.plugin_id,
                        button_id: action.button_id,
                        action_id: action.action_id,
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            let mut state = shared_state.write();
            state.toolbar_action_txs.remove(&plugin_handle);
        });

        info!(
            "Remote toolbar action stream connected for '{}' ({}) display='{}' language='{}'",
            plugin.plugin_id, req.plugin_handle, plugin.display_name, plugin.language
        );

        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(out_rx),
        )))
    }

    async fn hud_toolbar_action_stream(
        &self,
        request: Request<HudToolbarActionStreamRequest>,
    ) -> Result<Response<Self::HudToolbarActionStreamStream>, Status> {
        let req = request.into_inner();
        let plugin = {
            let state = self.state.read();
            state.plugin(&req.plugin_handle).cloned().ok_or_else(|| {
                Status::not_found(format!(
                    "remote plugin handle '{}' is not registered",
                    req.plugin_handle
                ))
            })?
        };

        let (out_tx, out_rx) = mpsc::channel::<Result<HudToolbarActionRequest, Status>>(8);
        let (action_tx, mut action_rx) = mpsc::channel::<RemoteHudToolbarActionRequest>(8);
        {
            let mut state = self.state.write();
            if state
                .hud_toolbar_action_txs
                .contains_key(&req.plugin_handle)
            {
                return Err(Status::already_exists(
                    "HUD toolbar action stream is already connected for this plugin",
                ));
            }
            state
                .hud_toolbar_action_txs
                .insert(req.plugin_handle.clone(), action_tx);
        }

        let shared_state = self.state.clone();
        let plugin_handle = req.plugin_handle.clone();
        tokio::spawn(async move {
            while let Some(action) = action_rx.recv().await {
                if out_tx
                    .send(Ok(HudToolbarActionRequest {
                        plugin_handle: action.plugin_handle,
                        plugin_id: action.plugin_id,
                        button_id: action.button_id,
                        action_id: action.action_id,
                        viewport: Some(to_proto_viewport(action.viewport)),
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            let mut state = shared_state.write();
            state.hud_toolbar_action_txs.remove(&plugin_handle);
        });

        info!(
            "Remote HUD toolbar action stream connected for '{}' ({}) display='{}' language='{}'",
            plugin.plugin_id, req.plugin_handle, plugin.display_name, plugin.language
        );

        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(out_rx),
        )))
    }

    async fn get_host_snapshot(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<HostSnapshot>, Status> {
        Ok(Response::new(host_snapshot_response(&self.app_state)))
    }

    async fn read_region(
        &self,
        request: Request<ReadRegionRequest>,
    ) -> Result<Response<ReadRegionResponse>, Status> {
        let req = request.into_inner();
        let data = crate::plugin_host::read_region(
            &self.app_state,
            req.file_id,
            req.level,
            req.x,
            req.y,
            req.width,
            req.height,
        )
        .map_err(Status::failed_precondition)?;
        Ok(Response::new(ReadRegionResponse {
            rgba_data: data,
            width: req.width,
            height: req.height,
        }))
    }

    async fn open_file(
        &self,
        request: Request<OpenFileRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        crate::plugin_host::open_file_path(std::path::PathBuf::from(request.into_inner().path))
            .map_err(Status::failed_precondition)?;
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn set_active_viewport(
        &self,
        request: Request<SetActiveViewportRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        crate::plugin_host::set_active_viewport(req.center_x, req.center_y, req.zoom)
            .map_err(Status::failed_precondition)?;
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn fit_active_viewport(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        crate::plugin_host::fit_active_viewport().map_err(Status::failed_precondition)?;
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn frame_active_rect(
        &self,
        request: Request<FrameActiveRectRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        crate::plugin_host::frame_active_rect(req.x, req.y, req.width, req.height)
            .map_err(Status::failed_precondition)?;
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn log_message(
        &self,
        request: Request<LogMessageRequest>,
    ) -> Result<Response<HostCommandResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let state = self.state.read();
            state
                .plugin(&req.plugin_handle)
                .map(|plugin| plugin.plugin_id.clone())
                .ok_or_else(|| {
                    Status::not_found(format!(
                        "remote plugin handle '{}' is not registered",
                        req.plugin_handle
                    ))
                })?
        };
        crate::plugin_host::log_message(
            &plugin_id,
            from_proto_log_level(req.level()),
            &req.message,
        );
        Ok(Response::new(HostCommandResponse {}))
    }

    async fn register_filter(
        &self,
        request: Request<RegisterFilterRequest>,
    ) -> Result<Response<RegisterFilterResponse>, Status> {
        let req = request.into_inner();
        let plugin_id = {
            let state = self.state.read();
            state
                .plugin(&req.plugin_handle)
                .map(|plugin| plugin.plugin_id.clone())
                .ok_or_else(|| {
                    Status::not_found(format!(
                        "remote plugin handle '{}' is not registered",
                        req.plugin_handle
                    ))
                })?
        };
        let filter_id = uuid::Uuid::new_v4().to_string();

        info!(
            "Registering remote filter '{}' for '{}' (id={}, cpu={}, gpu={})",
            req.name, plugin_id, filter_id, req.supports_cpu, req.supports_gpu
        );

        let filter = RemoteFilter {
            owner_handle: req.plugin_handle.clone(),
            name: req.name,
            supports_cpu: req.supports_cpu,
            supports_gpu: req.supports_gpu,
            enabled: false,
            cpu_request_tx: None,
        };

        {
            let mut state = self.state.write();
            state.filters.insert(filter_id.clone(), filter);
            let plugin = state.plugin_mut(&req.plugin_handle).ok_or_else(|| {
                Status::not_found(format!(
                    "remote plugin handle '{}' is not registered",
                    req.plugin_handle
                ))
            })?;
            plugin.filter_ids.insert(filter_id.clone());
        }

        Ok(Response::new(RegisterFilterResponse {
            filter_id,
            dma_buf_socket_path: String::new(),
        }))
    }

    async fn unregister_filter(
        &self,
        request: Request<UnregisterFilterRequest>,
    ) -> Result<Response<UnregisterFilterResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write();
        let Some(filter) = state.filters.get(&req.filter_id) else {
            return Err(Status::not_found(format!(
                "filter '{}' not found",
                req.filter_id
            )));
        };
        if filter.owner_handle != req.plugin_handle {
            return Err(Status::permission_denied(format!(
                "filter '{}' is not owned by plugin handle '{}'",
                req.filter_id, req.plugin_handle
            )));
        }

        state.filters.remove(&req.filter_id);
        if let Some(plugin) = state.plugin_mut(&req.plugin_handle) {
            plugin.filter_ids.remove(&req.filter_id);
            info!(
                "Unregistered remote filter '{}' for '{}'",
                req.filter_id, plugin.plugin_id
            );
        }
        Ok(Response::new(UnregisterFilterResponse {}))
    }

    async fn set_filter_enabled(
        &self,
        request: Request<SetFilterEnabledRequest>,
    ) -> Result<Response<SetFilterEnabledResponse>, Status> {
        let req = request.into_inner();
        {
            let mut state = self.state.write();
            if let Some(filter) = state.filters.get_mut(&req.filter_id) {
                if filter.owner_handle != req.plugin_handle {
                    return Err(Status::permission_denied(format!(
                        "filter '{}' is not owned by plugin handle '{}'",
                        req.filter_id, req.plugin_handle
                    )));
                }
                filter.enabled = req.enabled;
                info!(
                    "Filter '{}' ({}): enabled={}",
                    filter.name, req.filter_id, req.enabled
                );
            } else {
                return Err(Status::not_found(format!(
                    "filter '{}' not found",
                    req.filter_id
                )));
            }
        }
        crate::plugin_host::request_filter_repaint().map_err(Status::failed_precondition)?;
        Ok(Response::new(SetFilterEnabledResponse {}))
    }

    async fn apply_filter_cpu(
        &self,
        request: Request<ApplyFilterCpuRequest>,
    ) -> Result<Response<ApplyFilterCpuResponse>, Status> {
        let req = request.into_inner();
        let expected_len = (req.width as usize) * (req.height as usize) * 4;
        if req.rgba_data.len() != expected_len {
            return Err(Status::invalid_argument(format!(
                "expected {} bytes for {}x{} RGBA8, got {}",
                expected_len,
                req.width,
                req.height,
                req.rgba_data.len()
            )));
        }

        // For unary RPC, the plugin sends the processed data directly.
        // This is invoked by the *plugin* to return processed data.
        // However, the typical flow is host -> plugin -> host.
        //
        // In the bidirectional streaming model, filters are applied via the
        // stream. For the unary model, we store the result and the host picks
        // it up. We'll return an empty response since the plugin is sending
        // this back to us.
        Ok(Response::new(ApplyFilterCpuResponse {
            rgba_data: req.rgba_data,
            width: req.width,
            height: req.height,
        }))
    }

    type ApplyFilterCpuStreamStream = Pin<
        Box<
            dyn futures_core::Stream<Item = Result<ApplyFilterCpuResponse, Status>>
                + Send
                + 'static,
        >,
    >;

    async fn apply_filter_cpu_stream(
        &self,
        request: Request<Streaming<ApplyFilterCpuRequest>>,
    ) -> Result<Response<Self::ApplyFilterCpuStreamStream>, Status> {
        let mut in_stream = request.into_inner();

        // The plugin's first message identifies the filter via filter_id.
        let first_msg = in_stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("expected initial message with filter_id"))?;
        let filter_id = first_msg.filter_id.clone();

        // Create the channel that the render pipeline will use to push frames.
        let (frame_tx, mut frame_rx) = mpsc::channel::<CpuFilterRequest>(1);

        // Store the sender in the RemoteFilter so apply_remote_cpu_filters()
        // can reach us.
        {
            let mut state = self.state.write();
            if let Some(filter) = state.filters.get_mut(&filter_id) {
                filter.cpu_request_tx = Some(frame_tx);
                info!(
                    "CPU stream connected for filter '{}' ({})",
                    filter.name, filter_id
                );
            } else {
                return Err(Status::not_found(format!(
                    "filter '{}' not found",
                    filter_id
                )));
            }
        }

        // Output channel: frames pushed to plugin via the gRPC response stream.
        let (out_tx, out_rx) = mpsc::channel::<Result<ApplyFilterCpuResponse, Status>>(1);

        let state_clone = self.state.clone();
        let fid = filter_id.clone();

        // Coordinator task: bridge render pipeline ↔ plugin.
        tokio::spawn(async move {
            while let Some(frame_req) = frame_rx.recv().await {
                let response_tx = frame_req.response_tx;

                // Push the raw frame to the plugin via the response stream.
                let resp = ApplyFilterCpuResponse {
                    width: frame_req.width,
                    height: frame_req.height,
                    rgba_data: frame_req.rgba_data,
                };
                if out_tx.send(Ok(resp)).await.is_err() {
                    break; // plugin disconnected
                }

                // Wait for the plugin to return the processed frame.
                match in_stream.message().await {
                    Ok(Some(processed)) => {
                        let _ = response_tx.send(processed.rgba_data);
                    }
                    Ok(None) => break,
                    Err(e) => {
                        warn!("Remote filter stream error: {e}");
                        break;
                    }
                }
            }

            // Clean up when the stream closes.
            let mut state = state_clone.write();
            if let Some(filter) = state.filters.get_mut(&fid) {
                filter.cpu_request_tx = None;
                info!("CPU stream disconnected for filter '{}'", fid);
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(out_rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn apply_filter_gpu(
        &self,
        request: Request<ApplyFilterGpuRequest>,
    ) -> Result<Response<ApplyFilterGpuResponse>, Status> {
        let req = request.into_inner();
        let state = self.state.read();
        if !state.filters.contains_key(&req.filter_id) {
            return Err(Status::not_found(format!(
                "filter '{}' not found",
                req.filter_id
            )));
        }

        // GPU filter processing is handled out-of-band via DMA-BUF fd passing
        // over the Unix domain socket. This RPC serves as the metadata/signaling
        // channel. The actual implementation will be wired in a follow-up once
        // the DMA-BUF socket infrastructure is in place.
        Ok(Response::new(ApplyFilterGpuResponse { success: true }))
    }
}

/// Start the gRPC extension host server on the given port.
///
/// Returns the shared state handle so the render pipeline can query
/// registered filters.
pub(crate) async fn start_extension_host(
    port: u16,
    state: SharedExtensionHostState,
    app_state: Arc<RwLock<crate::state::AppState>>,
) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let service = ExtensionHostService::new(state, app_state);

    info!("Starting extension host gRPC server on {addr}");

    tonic::transport::Server::builder()
        .add_service(
            ExtensionHostServer::new(service)
                .max_decoding_message_size(MAX_GRPC_FILTER_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_FILTER_MESSAGE_BYTES),
        )
        .serve(addr)
        .await?;

    Ok(())
}

pub(crate) fn dispatch_remote_toolbar_action(
    state: &SharedExtensionHostState,
    plugin_id: &str,
    action_id: &str,
) -> Result<bool, String> {
    state.read().dispatch_toolbar_action(plugin_id, action_id)
}

pub(crate) fn has_remote_toolbar_action(
    state: &SharedExtensionHostState,
    plugin_id: &str,
    action_id: &str,
) -> bool {
    state.read().has_toolbar_action(plugin_id, action_id)
}

pub(crate) fn dispatch_remote_hud_toolbar_action(
    state: &SharedExtensionHostState,
    plugin_id: &str,
    action_id: &str,
    viewport: plugin_api::ViewportSnapshot,
) -> Result<bool, String> {
    state
        .read()
        .dispatch_hud_toolbar_action(plugin_id, action_id, viewport)
}

pub(crate) fn has_remote_hud_toolbar_action(
    state: &SharedExtensionHostState,
    plugin_id: &str,
    action_id: &str,
) -> bool {
    state.read().has_hud_toolbar_action(plugin_id, action_id)
}

fn host_snapshot_response(app_state: &Arc<RwLock<crate::state::AppState>>) -> HostSnapshot {
    let snapshot = crate::plugin_host::snapshot(app_state);
    HostSnapshot {
        app_name: snapshot.app_name,
        app_version: snapshot.app_version,
        render_backend: snapshot.render_backend,
        filtering_mode: snapshot.filtering_mode,
        split_enabled: snapshot.split_enabled,
        focused_pane: snapshot.focused_pane,
        open_files: snapshot
            .open_files
            .into_iter()
            .map(to_proto_open_file)
            .collect(),
        active_file: Some(
            snapshot
                .active_file
                .clone()
                .map(to_proto_open_file)
                .unwrap_or_default(),
        ),
        active_viewport: Some(
            snapshot
                .active_viewport
                .clone()
                .map(to_proto_viewport)
                .unwrap_or_default(),
        ),
        recent_files: snapshot.recent_files,
        has_active_file: snapshot.active_file.is_some(),
        has_active_viewport: snapshot.active_viewport.is_some(),
    }
}

fn to_proto_open_file(file: plugin_api::OpenFileInfo) -> OpenFileInfo {
    OpenFileInfo {
        file_id: file.file_id,
        path: file.path,
        filename: file.filename,
        width: file.width,
        height: file.height,
        level_count: file.level_count,
        vendor: file.vendor.clone().unwrap_or_default(),
        mpp_x: file.mpp_x.unwrap_or_default(),
        mpp_y: file.mpp_y.unwrap_or_default(),
        objective_power: file.objective_power.unwrap_or_default(),
        scan_date: file.scan_date.clone().unwrap_or_default(),
        has_vendor: file.vendor.is_some(),
        has_mpp_x: file.mpp_x.is_some(),
        has_mpp_y: file.mpp_y.is_some(),
        has_objective_power: file.objective_power.is_some(),
        has_scan_date: file.scan_date.is_some(),
    }
}

fn to_proto_viewport(viewport: plugin_api::ViewportSnapshot) -> ViewportSnapshot {
    ViewportSnapshot {
        pane_index: viewport.pane_index,
        center_x: viewport.center_x,
        center_y: viewport.center_y,
        zoom: viewport.zoom,
        width: viewport.width,
        height: viewport.height,
        image_width: viewport.image_width,
        image_height: viewport.image_height,
        bounds_left: viewport.bounds_left,
        bounds_top: viewport.bounds_top,
        bounds_right: viewport.bounds_right,
        bounds_bottom: viewport.bounds_bottom,
    }
}

fn from_proto_log_level(level: LogLevel) -> plugin_api::HostLogLevel {
    match level {
        LogLevel::Trace => plugin_api::HostLogLevel::Trace,
        LogLevel::Debug => plugin_api::HostLogLevel::Debug,
        LogLevel::Info => plugin_api::HostLogLevel::Info,
        LogLevel::Warn => plugin_api::HostLogLevel::Warn,
        LogLevel::Error => plugin_api::HostLogLevel::Error,
    }
}

/// Apply all enabled remote CPU filters to a frame buffer.
///
/// This is called from the render pipeline after the main render pass.
/// Filters are applied in insertion order.
pub(crate) fn apply_remote_cpu_filters(
    state: &SharedExtensionHostState,
    rgba_data: &mut [u8],
    width: u32,
    height: u32,
    _runtime: &tokio::runtime::Handle,
) {
    let filters: Vec<(String, String, mpsc::Sender<CpuFilterRequest>)> = {
        let state = state.read();
        state
            .filters
            .iter()
            .filter_map(|(filter_id, filter)| {
                if !filter.enabled || !filter.supports_cpu {
                    return None;
                }
                filter
                    .cpu_request_tx
                    .as_ref()
                    .map(|tx| (filter_id.clone(), filter.name.clone(), tx.clone()))
            })
            .collect()
    };

    for (filter_id, filter_name, tx) in filters {
        let (resp_tx, resp_rx) = std::sync::mpsc::channel();
        let req = CpuFilterRequest {
            width,
            height,
            rgba_data: rgba_data.to_vec(),
            response_tx: resp_tx,
        };
        if tx.blocking_send(req).is_ok() {
            match resp_rx.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(result) => {
                    if result.len() == rgba_data.len() {
                        rgba_data.copy_from_slice(&result);
                    } else {
                        warn!(
                            "Remote filter '{}' ({}) returned wrong buffer size",
                            filter_name, filter_id
                        );
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    warn!(
                        "Timed out waiting for remote filter '{}' ({}) response; skipping frame",
                        filter_name, filter_id
                    );
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    warn!(
                        "Remote filter '{}' ({}) disconnected before responding",
                        filter_name, filter_id
                    );
                }
            }
        }
    }
}

pub(crate) fn has_enabled_remote_cpu_only_filters(state: &SharedExtensionHostState) -> bool {
    state
        .read()
        .filters
        .values()
        .any(|filter| filter.enabled && filter.supports_cpu && !filter.supports_gpu)
}
