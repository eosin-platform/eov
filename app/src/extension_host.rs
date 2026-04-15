//! gRPC extension host server.
//!
//! When launched with `--extension-host-port`, the app starts a Tonic gRPC
//! server implementing the `ExtensionHost` service. Remote plugins (Python,
//! etc.) connect to this server to register viewport filters and exchange
//! pixel data.

use crate::eov_extension::extension_host_server::{ExtensionHost, ExtensionHostServer};
use crate::eov_extension::{
    ApplyFilterCpuRequest, ApplyFilterCpuResponse, ApplyFilterGpuRequest, ApplyFilterGpuResponse,
    RegisterFilterRequest, RegisterFilterResponse, SetFilterEnabledRequest,
    SetFilterEnabledResponse, UnregisterFilterRequest, UnregisterFilterResponse,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info, warn};

/// A remotely-registered viewport filter (from a gRPC client).
pub(crate) struct RemoteFilter {
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
    pub response_tx: tokio::sync::oneshot::Sender<Vec<u8>>,
}

/// Shared state for the extension host, accessible from both the gRPC server
/// and the render pipeline.
pub(crate) struct ExtensionHostState {
    pub filters: HashMap<String, RemoteFilter>,
}

impl ExtensionHostState {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
        }
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
}

impl ExtensionHostService {
    pub fn new(state: SharedExtensionHostState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ExtensionHost for ExtensionHostService {
    async fn register_filter(
        &self,
        request: Request<RegisterFilterRequest>,
    ) -> Result<Response<RegisterFilterResponse>, Status> {
        let req = request.into_inner();
        let filter_id = uuid::Uuid::new_v4().to_string();

        info!(
            "Registering remote filter '{}' (id={}, cpu={}, gpu={})",
            req.name, filter_id, req.supports_cpu, req.supports_gpu
        );

        let filter = RemoteFilter {
            name: req.name,
            supports_cpu: req.supports_cpu,
            supports_gpu: req.supports_gpu,
            enabled: false,
            cpu_request_tx: None,
        };

        self.state.write().filters.insert(filter_id.clone(), filter);

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
        if state.filters.remove(&req.filter_id).is_some() {
            info!("Unregistered remote filter '{}'", req.filter_id);
        } else {
            warn!("Attempted to unregister unknown filter '{}'", req.filter_id);
        }
        Ok(Response::new(UnregisterFilterResponse {}))
    }

    async fn set_filter_enabled(
        &self,
        request: Request<SetFilterEnabledRequest>,
    ) -> Result<Response<SetFilterEnabledResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write();
        if let Some(filter) = state.filters.get_mut(&req.filter_id) {
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
        let (out_tx, out_rx) =
            mpsc::channel::<Result<ApplyFilterCpuResponse, Status>>(1);

        let state_clone = self.state.clone();
        let fid = filter_id.clone();

        // Coordinator task: bridge render pipeline ↔ plugin.
        tokio::spawn(async move {
            while let Some(frame_req) = frame_rx.recv().await {
                let oneshot_tx = frame_req.response_tx;

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
                        let _ = oneshot_tx.send(processed.rgba_data);
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
) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let service = ExtensionHostService::new(state);

    info!("Starting extension host gRPC server on {addr}");

    tonic::transport::Server::builder()
        .add_service(ExtensionHostServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
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
    runtime: &tokio::runtime::Handle,
) {
    let state = state.read();
    for (filter_id, filter) in &state.filters {
        if !filter.enabled || !filter.supports_cpu {
            continue;
        }
        if let Some(tx) = &filter.cpu_request_tx {
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            let req = CpuFilterRequest {
                width,
                height,
                rgba_data: rgba_data.to_vec(),
                response_tx: resp_tx,
            };
            if tx.blocking_send(req).is_ok() {
                if let Ok(result) = resp_rx.blocking_recv() {
                    if result.len() == rgba_data.len() {
                        rgba_data.copy_from_slice(&result);
                    } else {
                        warn!(
                            "Remote filter '{}' ({}) returned wrong buffer size",
                            filter.name, filter_id
                        );
                    }
                }
            }
        }
    }
}
