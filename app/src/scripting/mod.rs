//! The scripting interface (AUDIT §14.5): researchers drive the running eov
//! and read slide regions from Python, shell scripts or any language that
//! can open a local socket, without writing a native plugin.
//!
//! - **Transport.** The single-instance endpoint ([`crate::instance`]): the
//!   same per-user Unix-domain socket or localhost port, and the same proof
//!   of the secret kept in `<name>.info`, which only the user can read,
//!   under the protocol name `eov-rpc1`. Nothing listens on the network.
//!   After the proof, each call is one frame (a 4-byte big-endian length,
//!   then UTF-8 JSON) answered by one frame, in order, for as many calls as
//!   the script likes; a connection that stays silent for two minutes is
//!   closed (scripts reconnect).
//! - **Calls** are JSON-RPC 2.0 style: `{"id": 1, "method": "slides",
//!   "params": {...}}`, answered by `{"id": 1, "result": ...}` or `{"id": 1,
//!   "error": {"code": -32602, "message": "...", "data": {"kind":
//!   "invalid_params"}}}` ([`ErrorKind`]). Every call is answered, batches
//!   are not supported, parameters are named, and unknown parameters are
//!   refused. Calls are at most 1 MiB, answers at most 64 MiB.
//! - **Threads.** Every connection has a thread of its own. Calls that only
//!   read (`slides`, `panes`, `view.get`, `roi.get`, `measurements.list`,
//!   `bookmarks.list`, `region.read`) answer from the state the host
//!   publishes for plugins ([`Published`]), so they never wait for the UI
//!   thread, which may be busy drawing or showing a dialog. Edits run on the
//!   UI thread as a short job ([`Session::on_ui`]) and publish the state
//!   again before they answer, so a read that follows sees them. Slide
//!   reads, identities, quantification and exports run on the connection's
//!   thread, never on the UI thread, at most [`HEAVY_CALLS`] at a time.
//! - **No special trust.** Scripts can do what the user can do in eov and
//!   no more: paths follow the rules of links ([`params`]), edits go
//!   through the same code as the tools (undo, persistence, plugin events),
//!   and every request is bounded. `scripting = false` in `config.toml`
//!   refuses every call.
//!
//! The methods are listed in [`METHODS`] and documented in the README
//! ("Scripting"); `clients/python/eov_client.py` is a client.

mod analysis;
mod annotations;
pub(crate) mod client;
mod params;
mod pixels;
#[cfg(test)]
mod tests;
mod viewer;

use crate::host_ui::UiHost;
use crate::instance::{CallError, MAX_SCRIPT_REPLY_BYTES, ScriptConnection, ScriptHandler};
use crate::plugin_runtime::shared::{HostShared, Published, PublishedFile};
use eov_plugin_api::protocol::FileId;
use parking_lot::{Condvar, Mutex};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

/// The protocol version `version` reports; it grows with new methods.
pub(crate) const PROTOCOL_VERSION: u32 = 1;

/// Every method, in the order the README documents them.
pub(crate) const METHODS: &[&str] = &[
    "version",
    "slides",
    "panes",
    "open",
    "close",
    "view.get",
    "view.set",
    "region.read",
    "roi.get",
    "roi.set",
    "measurements.list",
    "measurements.add",
    "measurements.remove",
    "bookmarks.list",
    "bookmarks.add",
    "quantify",
    "export.view",
];

/// Calls that read or render many pixels (`region.read`, `quantify`,
/// `export.view`) running at once, across all scripts; more wait.
pub(crate) const HEAVY_CALLS: usize = 4;

/// Scripts connected at once; the next one's first call is answered
/// `busy`. (The endpoint serves more connections, so other eovs can always
/// hand files over.)
pub(crate) const MAX_SESSIONS: usize = 16;

/// What kind of error a call met; the `kind` of an error's `data`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    /// The call is not JSON.
    Parse,
    /// The call is JSON but not a call (no method, a batch, a bad id).
    InvalidRequest,
    MethodNotFound,
    InvalidParams,
    /// A fault in eov.
    Internal,
    /// No such slide, pane, measurement or file.
    NotFound,
    /// Not allowed: a path rules out, a file that would be replaced.
    Refused,
    /// A region, a call or an answer beyond the limits.
    TooLarge,
    /// The window did not take an edit in time (a modal dialog may be
    /// open); nothing was done.
    Busy,
    /// The work itself failed: an unreadable slide, a failed export.
    Failed,
    /// eov is starting or closing.
    Unavailable,
    /// The user cancelled it (status bar).
    Cancelled,
    /// `scripting = false` in `config.toml`.
    Disabled,
}

impl ErrorKind {
    /// The JSON-RPC error code: the standard ones, and -32001 down for
    /// eov's own.
    pub(crate) fn code(self) -> i64 {
        match self {
            ErrorKind::Parse => -32700,
            ErrorKind::InvalidRequest => -32600,
            ErrorKind::MethodNotFound => -32601,
            ErrorKind::InvalidParams => -32602,
            ErrorKind::Internal => -32603,
            ErrorKind::NotFound => -32001,
            ErrorKind::Refused => -32002,
            ErrorKind::TooLarge => -32003,
            ErrorKind::Busy => -32004,
            ErrorKind::Failed => -32005,
            ErrorKind::Unavailable => -32006,
            ErrorKind::Cancelled => -32007,
            ErrorKind::Disabled => -32008,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            ErrorKind::Parse => "parse_error",
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::MethodNotFound => "method_not_found",
            ErrorKind::InvalidParams => "invalid_params",
            ErrorKind::Internal => "internal",
            ErrorKind::NotFound => "not_found",
            ErrorKind::Refused => "refused",
            ErrorKind::TooLarge => "too_large",
            ErrorKind::Busy => "busy",
            ErrorKind::Failed => "failed",
            ErrorKind::Unavailable => "unavailable",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Disabled => "disabled",
        }
    }
}

/// A call's error, for the script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RpcError {
    pub(crate) kind: ErrorKind,
    pub(crate) message: String,
}

impl RpcError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidParams, message)
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    pub(crate) fn refused(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Refused, message)
    }

    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::TooLarge, message)
    }

    pub(crate) fn failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Failed, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    fn to_json(&self) -> Value {
        json!({
            "code": self.kind.code(),
            "message": self.message,
            "data": { "kind": self.kind.name() },
        })
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.kind.name())
    }
}

pub(crate) type CallResult = Result<Value, RpcError>;

/// A job for the UI thread.
pub(crate) type UiJob = Box<dyn FnOnce() + Send>;

/// Runs jobs on the UI thread: `slint::invoke_from_event_loop` in eov.
pub(crate) type UiRunner = Arc<dyn Fn(UiJob) + Send + Sync>;

/// How long a call waits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    /// For a job on the UI thread.
    pub(crate) answer: Duration,
    /// For a slide to open.
    pub(crate) open: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            answer: Duration::from_secs(20),
            open: Duration::from_secs(120),
        }
    }
}

/// The scripting interface of this eov.
pub(crate) struct Service {
    /// `scripting` in `config.toml`.
    enabled: bool,
    ui: UiRunner,
    /// What the host publishes for plugins: calls that only read answer
    /// from it, without the UI thread.
    shared: Arc<HostShared>,
    timing: Timing,
    /// Scripts connected now.
    sessions: AtomicUsize,
}

/// The endpoint's handler for scripting connections (main.rs).
pub(crate) fn handler(enabled: bool, shared: Arc<HostShared>) -> Arc<ScriptHandler> {
    let ui: UiRunner = Arc::new(|job| {
        if let Err(error) = slint::invoke_from_event_loop(job) {
            tracing::debug!("A script's call cannot reach the window: {error}");
        }
    });
    let service = Arc::new(Service::new(enabled, ui, shared, Timing::default()));
    Arc::new(move |connection| service.serve(connection))
}

impl Service {
    pub(crate) fn new(
        enabled: bool,
        ui: UiRunner,
        shared: Arc<HostShared>,
        timing: Timing,
    ) -> Self {
        Self {
            enabled,
            ui,
            shared,
            timing,
            sessions: AtomicUsize::new(0),
        }
    }

    /// Answers a connection's calls until the script hangs up.
    pub(crate) fn serve(&self, mut connection: ScriptConnection) {
        struct Connected<'a>(&'a AtomicUsize);
        impl Drop for Connected<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let connected = self.sessions.fetch_add(1, Ordering::AcqRel) + 1;
        let _connected = Connected(&self.sessions);
        if connected > MAX_SESSIONS {
            let error = RpcError::new(
                ErrorKind::Busy,
                format!("{MAX_SESSIONS} scripts are connected to eov; close one and try again"),
            );
            if !matches!(connection.next_call(), Err(CallError::Closed)) {
                let _ = connection.answer(&reply(&Value::Null, Err(error)));
            }
            return;
        }
        tracing::debug!("A script connected");
        let mut session = Session::new(self);
        loop {
            let answer = match connection.next_call() {
                Ok(call) => session.answer(&call),
                Err(CallError::TooLarge(length)) => {
                    let error = RpcError::too_large(format!(
                        "a call of {length} bytes is larger than the {} eov accepts",
                        crate::instance::MAX_SCRIPT_REQUEST_BYTES
                    ));
                    let _ = connection.answer(&reply(&Value::Null, Err(error)));
                    break;
                }
                Err(CallError::Closed) => break,
            };
            if connection.answer(&answer).is_err() || connection.closing() {
                break;
            }
        }
        tracing::debug!("A script disconnected");
    }
}

/// The answer to a call, as JSON bytes (never larger than the protocol
/// allows).
fn reply(id: &Value, result: CallResult) -> Vec<u8> {
    let body = match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error.to_json() }),
    };
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    if bytes.len() <= MAX_SCRIPT_REPLY_BYTES as usize {
        return bytes;
    }
    let error = RpcError::too_large(format!(
        "the answer would be {} bytes, more than the {MAX_SCRIPT_REPLY_BYTES} the protocol \
         allows; ask for less",
        bytes.len()
    ));
    serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": id, "error": error.to_json() }))
        .unwrap_or_default()
}

/// One script's connection.
pub(crate) struct Session<'a> {
    service: &'a Service,
    /// Slides `region.read` opened by path.
    slides: pixels::SlideCache,
}

impl<'a> Session<'a> {
    fn new(service: &'a Service) -> Self {
        Self {
            service,
            slides: pixels::SlideCache::default(),
        }
    }

    /// Answers one call (`body`, JSON).
    pub(crate) fn answer(&mut self, body: &[u8]) -> Vec<u8> {
        let call: Value = match serde_json::from_slice(body) {
            Ok(call) => call,
            Err(error) => {
                return reply(
                    &Value::Null,
                    Err(RpcError::new(
                        ErrorKind::Parse,
                        format!("the call is not JSON: {error}"),
                    )),
                );
            }
        };
        let id = match call.get("id") {
            None => Value::Null,
            Some(id @ (Value::Null | Value::Number(_) | Value::String(_))) => id.clone(),
            Some(_) => {
                return reply(
                    &Value::Null,
                    Err(RpcError::new(
                        ErrorKind::InvalidRequest,
                        "the id must be a number, a string or null",
                    )),
                );
            }
        };
        let result = split_call(call).and_then(|(method, params)| {
            if !self.service.enabled {
                return Err(RpcError::new(
                    ErrorKind::Disabled,
                    "scripting is turned off (scripting = false in config.toml)",
                ));
            }
            tracing::debug!("Script call: {method}");
            // A fault in a method answers this call, and the connection
            // stays usable.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.call(&method, params)))
                .unwrap_or_else(|_| {
                    Err(RpcError::internal(format!(
                        "{method} failed inside eov (see eov's log)"
                    )))
                })
        });
        reply(&id, result)
    }

    fn call(&mut self, method: &str, params: Value) -> CallResult {
        match method {
            "version" => viewer::version(parse(params)?),
            "slides" => viewer::slides(self, parse(params)?),
            "panes" => viewer::panes(self, parse(params)?),
            "open" => viewer::open(self, parse(params)?),
            "close" => viewer::close(self, parse(params)?),
            "view.get" => viewer::view_get(self, parse(params)?),
            "view.set" => viewer::view_set(self, parse(params)?),
            "region.read" => pixels::region_read(self, parse(params)?),
            "roi.get" => annotations::roi_get(self, parse(params)?),
            "roi.set" => annotations::roi_set(self, parse(params)?),
            "measurements.list" => annotations::measurements_list(self, parse(params)?),
            "measurements.add" => annotations::measurements_add(self, parse(params)?),
            "measurements.remove" => annotations::measurements_remove(self, parse(params)?),
            "bookmarks.list" => annotations::bookmarks_list(self, parse(params)?),
            "bookmarks.add" => annotations::bookmarks_add(self, parse(params)?),
            "quantify" => analysis::quantify(self, parse(params)?),
            "export.view" => analysis::export_view(self, parse(params)?),
            _ => Err(RpcError::new(
                ErrorKind::MethodNotFound,
                format!("eov has no method {method:?} (see version)"),
            )),
        }
    }

    /// Runs `work` on the UI thread with the host and waits for its
    /// result. `work` must be quick: no slide reads, no file scans. If the
    /// window does not take it within [`Timing::answer`], it never runs: a
    /// script told `busy` knows nothing was done.
    pub(crate) fn on_ui<T: Send + 'static>(
        &self,
        work: impl FnOnce(&UiHost) -> Result<T, RpcError> + Send + 'static,
    ) -> Result<T, RpcError> {
        let claim = Claim::default();
        let (sender, receiver) = mpsc::sync_channel(1);
        let taken = claim.clone();
        (self.service.ui)(Box::new(move || {
            if !taken.take() {
                return;
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::host_ui::with_host(|host| work(host))
            }));
            let result = match outcome {
                Ok(Some(result)) => result,
                Ok(None) => Err(RpcError::new(
                    ErrorKind::Unavailable,
                    "eov is still starting",
                )),
                Err(_) => Err(RpcError::internal(
                    "the call failed inside eov (see eov's log)",
                )),
            };
            let _ = sender.send(result);
        }));
        let closing = || RpcError::new(ErrorKind::Unavailable, "eov is closing");
        match receiver.recv_timeout(self.service.timing.answer) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if claim.abandon() {
                    return Err(RpcError::new(
                        ErrorKind::Busy,
                        format!(
                            "the window did not answer within {} s, and nothing was done (a \
                             dialog may be open in eov)",
                            self.service.timing.answer.as_secs()
                        ),
                    ));
                }
                // It started just now: it is quick.
                receiver.recv().unwrap_or_else(|_| Err(closing()))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(closing()),
        }
    }

    /// Queues `work` on the UI thread without waiting for it.
    pub(crate) fn post_ui(&self, work: impl FnOnce(&UiHost) + Send + 'static) {
        (self.service.ui)(Box::new(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::host_ui::with_host(|host| work(host))
            }));
        }));
    }

    pub(crate) fn timing(&self) -> Timing {
        self.service.timing
    }

    /// The host state as last published (each frame, and after every edit
    /// a script makes).
    pub(crate) fn published(&self) -> Arc<Published> {
        self.service.shared.published()
    }

    /// The open slide `file_id` itself (stored levels), for reading.
    pub(crate) fn open_slide(&self, file_id: FileId) -> Result<eov_common::WsiFile, RpcError> {
        self.service
            .shared
            .reader(file_id)
            .and_then(|reader| reader.wsi())
            .ok_or_else(|| RpcError::not_found(format!("no open slide has the id {file_id}")))
    }
}

/// The open slide `slide` (a `file_id` of `slides`), or the one the focused
/// pane shows, in the published state.
pub(crate) fn published_slide(
    published: &Published,
    slide: Option<i64>,
) -> Result<(FileId, &PublishedFile), RpcError> {
    let file_id = match slide {
        Some(id) => id,
        None => published
            .snapshot
            .focused_pane
            .and_then(|focused| published.snapshot.panes.iter().find(|p| p.pane == focused))
            .and_then(|pane| pane.file_id)
            .ok_or_else(|| {
                RpcError::not_found("the focused pane shows no slide; pass \"slide\" (see slides)")
            })?,
    };
    published
        .files
        .get(&file_id)
        .map(|file| (file_id, file))
        .ok_or_else(|| {
            RpcError::not_found(format!("no open slide has the id {file_id} (see slides)"))
        })
}

/// Who gets a UI job first: the UI thread, which runs it, or its caller,
/// who gave up waiting (then it never runs).
#[derive(Clone, Default)]
struct Claim(Arc<AtomicU8>);

impl Claim {
    const PENDING: u8 = 0;
    const RUNNING: u8 = 1;
    const ABANDONED: u8 = 2;

    /// The UI thread may run the job (its caller still waits).
    fn take(&self) -> bool {
        self.0
            .compare_exchange(
                Self::PENDING,
                Self::RUNNING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// The caller gives up; `false` when the job has started already.
    fn abandon(&self) -> bool {
        self.0
            .compare_exchange(
                Self::PENDING,
                Self::ABANDONED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

/// A call's method and parameters (`{}` when there are none).
fn split_call(call: Value) -> Result<(String, Value), RpcError> {
    let invalid = |message: &str| RpcError::new(ErrorKind::InvalidRequest, message);
    let Value::Object(mut call) = call else {
        return Err(invalid(
            "a call is an object such as {\"id\": 1, \"method\": \"slides\"}; batches are not \
             supported",
        ));
    };
    match call.get("jsonrpc") {
        None => {}
        Some(Value::String(version)) if version == "2.0" => {}
        Some(_) => return Err(invalid("jsonrpc must be \"2.0\" when given")),
    }
    let method = match call.remove("method") {
        Some(Value::String(method)) => method,
        _ => return Err(invalid("the call names no method")),
    };
    let params = match call.remove("params") {
        None | Some(Value::Null) => Value::Object(Default::default()),
        Some(params @ Value::Object(_)) => params,
        Some(_) => {
            return Err(RpcError::invalid(
                "params must be an object of named parameters",
            ));
        }
    };
    if let Some(key) = call
        .keys()
        .find(|key| !matches!(key.as_str(), "id" | "jsonrpc"))
    {
        return Err(invalid(&format!("unknown member {key:?} in the call")));
    }
    Ok((method, params))
}

/// A method's parameters.
fn parse<T: DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    serde_json::from_value(params).map_err(|error| RpcError::invalid(error.to_string()))
}

/// No parameters.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoParams {}

/// Limits the pixel-heavy calls running at once ([`HEAVY_CALLS`]).
pub(crate) struct Permits {
    used: Mutex<usize>,
    freed: Condvar,
    max: usize,
}

/// A place among the [`Permits`], given back when dropped.
pub(crate) struct Permit<'a>(&'a Permits);

impl Permits {
    pub(crate) const fn new(max: usize) -> Self {
        Self {
            used: Mutex::new(0),
            freed: Condvar::new(),
            max,
        }
    }

    pub(crate) fn acquire(&self) -> Permit<'_> {
        let mut used = self.used.lock();
        while *used >= self.max {
            self.freed.wait(&mut used);
        }
        *used += 1;
        Permit(self)
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.0.used.lock() -= 1;
        self.0.freed.notify_one();
    }
}

/// The places of [`HEAVY_CALLS`].
pub(crate) static HEAVY: Permits = Permits::new(HEAVY_CALLS);

#[cfg(test)]
mod session_tests {
    use super::*;

    fn shared() -> Arc<HostShared> {
        Arc::new(HostShared::new(
            Arc::new(crate::overlay::OverlayStore::new()),
            crate::plugin_runtime::settings::SettingsStore::new(),
            "1.0.0",
        ))
    }

    /// A service whose UI thread never answers (jobs are queued, never run).
    fn service_with(enabled: bool, shared: Arc<HostShared>) -> Service {
        let parked: Arc<Mutex<Vec<UiJob>>> = Arc::default();
        Service::new(
            enabled,
            Arc::new(move |job| parked.lock().push(job)),
            shared,
            Timing {
                answer: Duration::from_millis(200),
                open: Duration::from_millis(200),
            },
        )
    }

    fn service(enabled: bool) -> Service {
        service_with(enabled, shared())
    }

    fn answer(service: &Service, call: &str) -> Value {
        serde_json::from_slice(&Session::new(service).answer(call.as_bytes())).unwrap()
    }

    fn kind(answer: &Value) -> &str {
        answer["error"]["data"]["kind"].as_str().unwrap_or("none")
    }

    #[test]
    fn calls_are_checked_before_they_run() {
        let service = service(true);
        let version = answer(
            &service,
            r#"{"jsonrpc": "2.0", "id": 3, "method": "version"}"#,
        );
        assert_eq!(version["id"], json!(3));
        assert_eq!(version["result"]["methods"], json!(METHODS));
        for (call, expected) in [
            ("{", "parse_error"),
            ("[]", "invalid_request"),
            (r#"{"id": {}, "method": "version"}"#, "invalid_request"),
            (r#"{"id": 1}"#, "invalid_request"),
            (
                r#"{"id": 1, "method": "version", "extra": 1}"#,
                "invalid_request",
            ),
            (
                r#"{"id": 1, "method": "version", "params": 5}"#,
                "invalid_params",
            ),
            (
                r#"{"id": 1, "method": "version", "params": {"x": 1}}"#,
                "invalid_params",
            ),
            (r#"{"id": 1, "method": "nope"}"#, "method_not_found"),
        ] {
            assert_eq!(kind(&answer(&service, call)), expected, "{call}");
        }
        // Codes follow JSON-RPC.
        assert_eq!(answer(&service, "{")["error"]["code"], json!(-32700));
        assert_eq!(
            answer(&service, r#"{"id": 1, "method": "nope"}"#)["error"]["code"],
            json!(-32601)
        );
    }

    #[test]
    fn a_window_that_does_not_answer_is_reported_busy() {
        let service = service(true);
        let started = std::time::Instant::now();
        let edit = r#"{"id": 1, "method": "roi.set", "params": {"roi": null}}"#;
        assert_eq!(kind(&answer(&service, edit)), "busy");
        assert!(started.elapsed() < Duration::from_secs(5));
        // A runner that drops the job without running it: eov is closing.
        let gone = Service::new(true, Arc::new(drop), shared(), Timing::default());
        assert_eq!(kind(&answer(&gone, edit)), "unavailable");
    }

    /// Calls that only read answer from the published state, even while the
    /// UI thread is busy (here: never runs a job).
    #[test]
    fn reads_never_wait_for_the_window() {
        use crate::interaction::MeasureMode;
        use crate::plugin_runtime::shared::PublishedFile;
        use eov_plugin_api::protocol::{
            FileInfo, MeasurementInfo, PaneInfo, Rect, Shape, ViewInfo,
        };
        let shared = shared();
        let view = ViewInfo {
            pane: 4,
            file_id: 7,
            center_x: 100.0,
            center_y: 50.0,
            zoom: 0.5,
            ..Default::default()
        };
        let mut published = Published::default();
        published.snapshot.focused_pane = Some(4);
        published.snapshot.panes = vec![PaneInfo {
            pane: 4,
            index: 0,
            file_id: Some(7),
            view: Some(view),
        }];
        published.tabs.insert(4, vec![7]);
        published.files.insert(
            7,
            PublishedFile {
                info: FileInfo {
                    file_id: 7,
                    filename: "a.svs".into(),
                    mpp_x: Some(0.25),
                    mpp_y: Some(0.25),
                    identity: Some("eov-slide-1:test".into()),
                    ..Default::default()
                },
                roi: Some(Rect {
                    x: 1.0,
                    y: 2.0,
                    width: 3.0,
                    height: 4.0,
                }),
                measurements: vec![MeasurementInfo {
                    measurement_id: "line:1".into(),
                    file_id: 7,
                    shape: Shape::Polyline {
                        points: vec![[0.0, 0.0], [40.0, 0.0]],
                    },
                    length_um: Some(10.0),
                    area_um2: None,
                    angle_deg: None,
                }],
                measurement_modes: vec![MeasureMode::Line],
                ..Default::default()
            },
        );
        shared.publish(published);
        let service = service_with(true, shared);
        let started = std::time::Instant::now();
        let call = |method: &str| {
            answer(&service, &format!(r#"{{"id": 1, "method": "{method}"}}"#))["result"].clone()
        };
        let slides = call("slides");
        assert_eq!(slides[0]["file_id"], json!(7));
        assert_eq!(slides[0]["native_magnification"], json!(40.0));
        assert_eq!(slides[0]["panes"], json!([4]));
        let panes = call("panes");
        assert_eq!(panes[0]["tabs"], json!([7]));
        assert_eq!(panes[0]["focused"], json!(true));
        let view = call("view.get");
        assert_eq!(view["zoom"], json!(0.5));
        assert_eq!(view["magnification"], json!(20.0));
        assert_eq!(call("roi.get")["roi"]["height"], json!(4.0));
        let measurements = call("measurements.list");
        assert_eq!(measurements[0]["length_um"], json!(10.0));
        assert_eq!(measurements[0]["label"], json!("10.0 µm"));
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "nothing waited"
        );
        // An open slide's pixels need its handle, which this test has not.
        let read = answer(
            &service,
            r#"{"id": 1, "method": "region.read", "params": {"x": 0, "y": 0, "width": 1, "height": 1}}"#,
        );
        assert_eq!(kind(&read), "not_found");
    }

    /// Multichannel slides are read as eov draws them: a slide eov has open
    /// (by id or by path) with the display its Channels panel set, other
    /// slides with the defaults they open with.
    #[test]
    fn multichannel_slides_are_read_as_drawn() {
        use crate::host_ui::publish::FileReader;
        use crate::plugin_runtime::shared::PublishedFile;
        use base64::Engine;
        use eov_common::test_support::{OmeTiffSpec, write_ome_tiff};
        use eov_plugin_api::protocol::FileInfo;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multiplex.ome.tif");
        let spec = OmeTiffSpec::channels(
            64,
            32,
            32,
            1,
            &[("DAPI", [0, 0, 255]), ("CD8", [255, 0, 0])],
        );
        write_ome_tiff(&path, &spec, |c, x, _| f64::from(x) * (c + 1) as f64).unwrap();
        // The same slide, not open in eov.
        let other = dir.path().join("copy.ome.tif");
        std::fs::copy(&path, &other).unwrap();
        let wsi = eov_common::WsiFile::open(&path).unwrap();
        let shared = shared();
        shared.set_reader(3, Arc::new(FileReader::new(wsi.clone())));
        let mut published = Published::default();
        let info = FileInfo {
            file_id: 3,
            ..Default::default()
        };
        published.files.insert(
            3,
            PublishedFile {
                info,
                ..Default::default()
            },
        );
        shared.publish(published);
        // As the Channels panel does: CD8 alone, in white.
        let mut display = wsi.channel_display().unwrap();
        display.channels[0].visible = false;
        display.channels[1].color = [255, 255, 255];
        wsi.set_channel_display(display).unwrap();

        let service = service_with(true, shared);
        let read = |source: String| {
            let call = json!({
                "id": 1,
                "method": "region.read",
                "params": {
                    "slide": if source.is_empty() { json!(3) } else { Value::Null },
                    "path": if source.is_empty() { Value::Null } else { json!(source) },
                    "x": 0, "y": 0, "width": 64, "height": 32, "format": "rgba",
                },
            });
            let answer = answer(&service, &call.to_string());
            let data = answer["result"]["data"].as_str().expect("pixels");
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .unwrap()
        };
        let shown = wsi.read_region(0, 0, 0, 64, 32).unwrap();
        let defaults = eov_common::WsiFile::open(&other)
            .unwrap()
            .read_region(0, 0, 0, 64, 32)
            .unwrap();
        assert_ne!(shown, defaults, "the display changed what is drawn");
        assert_eq!(read(String::new()), shown, "an open slide, as shown");
        let by_path = read(path.to_string_lossy().into_owned());
        assert_eq!(by_path, shown, "the open slide by path, as shown");
        let closed = read(other.to_string_lossy().into_owned());
        assert_eq!(closed, defaults, "another slide, as it opens");
    }

    #[test]
    fn scripting_can_be_turned_off() {
        let service = service(false);
        let refused = answer(&service, r#"{"id": 1, "method": "version"}"#);
        assert_eq!(kind(&refused), "disabled");
        assert!(
            refused["error"]["message"]
                .as_str()
                .unwrap()
                .contains("scripting = false")
        );
    }

    #[test]
    fn answers_never_exceed_the_protocol() {
        let huge = "x".repeat(MAX_SCRIPT_REPLY_BYTES as usize);
        let bytes = reply(&json!(1), Ok(Value::String(huge)));
        assert!(bytes.len() < 1024);
        let answer: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(kind(&answer), "too_large");
    }

    /// An endpoint serving `service` (and taking every file handed over).
    fn serving(
        service: Service,
    ) -> (
        tempfile::TempDir,
        crate::instance::Endpoint,
        crate::instance::Server,
    ) {
        use crate::instance::{self, Claim, Endpoint, Handlers, Transport};
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), Transport::Tcp);
        let Claim::Primary(mut server) = instance::claim(
            &endpoint,
            &crate::launch::OpenRequest::default(),
            serving_timing(),
        ) else {
            panic!("the endpoint is free");
        };
        let service = Arc::new(service);
        server.serve(
            serving_timing(),
            Handlers {
                open: Arc::new(|_, answer| {
                    let _ = answer.send(Ok(()));
                }),
                script: Some(Arc::new(move |connection| service.serve(connection))),
            },
        );
        (dir, endpoint, server)
    }

    fn serving_timing() -> crate::instance::Timing {
        crate::instance::Timing {
            io: Duration::from_secs(2),
            ..crate::instance::Timing::default()
        }
    }

    /// `eov call`'s exit statuses: 2 for a call that was wrong (nothing was
    /// done), 1 for one that failed, 3 without an eov.
    #[test]
    fn eov_call_reports_how_a_call_went() {
        use super::client::{self, EXIT_CALL_FAILED, EXIT_USAGE};
        let (dir, endpoint, _server) = serving(service(true));
        assert_eq!(
            client::call(&endpoint, "version", json!({}), None).unwrap()["protocol"],
            json!(1)
        );
        for (method, params, status) in [
            ("no.such.method", json!({}), EXIT_USAGE),
            ("view.set", json!({"zom": 1}), EXIT_USAGE),
            // The window never answers here.
            ("roi.set", json!({"roi": null}), EXIT_CALL_FAILED),
        ] {
            let failure = client::call(&endpoint, method, params, None).unwrap_err();
            assert_eq!(failure.status, status, "{method}: {}", failure.message);
        }
        // --out is checked before anything runs.
        let out = dir.path().join("x.bin");
        let failure =
            client::call(&endpoint, "roi.set", json!({"roi": null}), Some(&out)).unwrap_err();
        assert_eq!(failure.status, EXIT_USAGE);
    }

    /// Scripts beyond [`MAX_SESSIONS`] are told so; hand-overs keep their
    /// places.
    #[test]
    fn scripts_beyond_the_limit_are_answered_busy() {
        use crate::instance::{self, Claim};
        let (_dir, endpoint, _server) = serving(service(true));
        let timing = serving_timing();
        let version = br#"{"id": 1, "method": "version"}"#;
        let call = |client: &mut instance::ScriptClient| -> Value {
            serde_json::from_slice(&client.call(version, Duration::from_secs(5)).unwrap()).unwrap()
        };
        let mut connected: Vec<_> = (0..MAX_SESSIONS)
            .map(|_| instance::connect_script(&endpoint, timing).unwrap())
            .collect();
        for client in &mut connected {
            assert_eq!(call(client)["result"]["protocol"], json!(1));
        }
        let mut extra = instance::connect_script(&endpoint, timing).unwrap();
        let refused = call(&mut extra);
        assert_eq!(kind(&refused), "busy");
        // Files are still handed over.
        assert!(matches!(
            instance::claim(&endpoint, &crate::launch::OpenRequest::default(), timing),
            Claim::Forwarded
        ));
        drop(connected.pop());
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let mut again = instance::connect_script(&endpoint, timing).unwrap();
            if call(&mut again).get("result").is_some() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "no place came free");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_job_its_caller_gave_up_on_never_runs() {
        let abandoned = Claim::default();
        assert!(abandoned.abandon());
        assert!(!abandoned.clone().take(), "the window comes too late");
        let started = Claim::default();
        assert!(started.take());
        assert!(!started.abandon(), "a job that started is waited for");
    }

    #[test]
    fn heavy_calls_wait_for_a_place() {
        let permits = Arc::new(Permits::new(2));
        let first = permits.acquire();
        let _second = permits.acquire();
        let waiting = {
            let permits = Arc::clone(&permits);
            std::thread::spawn(move || {
                let _third = permits.acquire();
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!waiting.is_finished(), "a third waits");
        drop(first);
        waiting.join().unwrap();
    }
}
