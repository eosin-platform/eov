//! One eov per user and setup (SH-8): a second `eov FILES` hands its
//! request to the eov that is running, which opens the files as tabs and
//! comes to the front, and exits.
//!
//! - **Endpoint.** The running eov holds an exclusive lock on
//!   `<runtime dir>/<name>.lock` and listens on `<name>.sock`, a Unix-domain
//!   socket (Linux, macOS), or on a localhost TCP port (Windows). `<name>.info`
//!   holds the port, the process id and a random secret. The runtime dir
//!   ([`crate::paths::runtime_dir`]) is private to the user (0700 on Unix,
//!   the user's local application data on Windows). The name hashes eov's
//!   home, configuration file and plugin folder, so eovs set up differently
//!   (`EOV_HOME`, `--config`, `--plugin-dir`) stay apart.
//! - **Stale endpoints.** A crashed eov leaves its files behind, but the
//!   operating system releases its lock, so the next eov takes the lock and
//!   replaces them: nothing a dead eov left can block start-up.
//! - **Authentication.** Both sides prove they know the secret (HMAC-SHA256
//!   over a fresh challenge from the other side) before any path is sent:
//!   a program squatting on the port or socket learns nothing, and only
//!   programs that can read the user's runtime dir can send requests.
//!   Requests are bounded (1 MiB, 4096 paths) and versioned.
//! - **Handling.** The listener thread never touches the viewer's state: it
//!   hands each request to the UI thread, which opens it from the event
//!   loop with no lock held, and answers. The second eov exits once the
//!   files are taken; if the running eov does not answer in time, the
//!   second one starts on its own instead.
//! - Subcommands never forward, and neither do `--new-instance`,
//!   `single_instance = false` in `config.toml` or options that configure
//!   the process (`--cpu`, `--debug`, ...): see
//!   [`crate::cli::LaunchOptions::may_forward`].

use crate::launch::{OpenRequest, Problem, ViewRequest};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};

/// Protocol name and version: the first bytes a client sends.
const MAGIC: &[u8; 8] = b"eov-ipc1";
const SECRET_BYTES: usize = 32;
const NONCE_BYTES: usize = 32;
const PROOF_BYTES: usize = 32;
const MAX_REQUEST_BYTES: u32 = 1 << 20;
const MAX_REPLY_BYTES: u32 = 64 * 1024;
const MAX_PATHS: usize = 4096;

/// How long the two sides wait for each other.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    /// For each read or write.
    pub(crate) io: Duration,
    /// For the running eov to take the files (it may still be starting).
    pub(crate) answer: Duration,
    /// For a running eov that is starting or exiting to become reachable.
    pub(crate) patience: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            io: Duration::from_secs(5),
            answer: Duration::from_secs(20),
            patience: Duration::from_secs(3),
        }
    }
}

/// How the eovs talk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transport {
    /// A Unix-domain socket in the runtime dir.
    #[cfg(unix)]
    Unix,
    /// A localhost TCP port (Windows; tested on every platform).
    #[cfg_attr(unix, allow(dead_code))]
    Tcp,
}

impl Transport {
    fn native() -> Self {
        #[cfg(unix)]
        {
            Transport::Unix
        }
        #[cfg(not(unix))]
        {
            Transport::Tcp
        }
    }
}

/// Where the running eov of one setup listens.
#[derive(Debug, Clone)]
pub(crate) struct Endpoint {
    dir: PathBuf,
    name: String,
    transport: Transport,
}

impl Endpoint {
    pub(crate) fn new(dir: PathBuf, name: String, transport: Transport) -> Self {
        Self {
            dir,
            name,
            transport,
        }
    }

    /// The endpoint of eovs that use this configuration file and plugin
    /// folder (and the current eov home).
    pub(crate) fn for_setup(config_file: &Path, plugin_dir: &Path) -> Self {
        let mut hasher = Sha256::new();
        for part in [&crate::paths::home_dir(), config_file, plugin_dir] {
            hasher.update(part.as_os_str().as_encoded_bytes());
            hasher.update([0]);
        }
        let name = format!("eov-{}", hex(&hasher.finalize()[..8]));
        Self::new(crate::paths::runtime_dir(), name, Transport::native())
    }

    fn file(&self, extension: &str) -> PathBuf {
        self.dir.join(format!("{}.{extension}", self.name))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex<const N: usize>(text: &str) -> Option<[u8; N]> {
    let mut bytes = [0; N];
    if text.len() != N * 2 {
        return None;
    }
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(index * 2..index * 2 + 2)?, 16).ok()?;
    }
    Some(bytes)
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0; N];
    for chunk in bytes.chunks_mut(16) {
        let random = uuid::Uuid::new_v4().into_bytes();
        chunk.copy_from_slice(&random[..chunk.len()]);
    }
    bytes
}

/// What `<name>.info` says.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Info {
    /// TCP port (0 for a Unix socket).
    port: u16,
    pid: u32,
    secret: [u8; SECRET_BYTES],
}

impl Info {
    fn parse(text: &str) -> Option<Self> {
        let mut fields = text.split_whitespace();
        let port = fields.next()?.parse().ok()?;
        let pid = fields.next()?.parse().ok()?;
        let secret = unhex(fields.next()?)?;
        Some(Self { port, pid, secret })
    }

    fn to_text(&self) -> String {
        format!("{} {} {}\n", self.port, self.pid, hex(&self.secret))
    }
}

fn read_info(endpoint: &Endpoint) -> io::Result<Info> {
    let text = fs::read_to_string(endpoint.file("info"))?;
    Info::parse(&text).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unreadable"))
}

/// Proves knowledge of `secret` for `nonce`, as `role`.
fn proof(secret: &[u8; SECRET_BYTES], role: &[u8], nonce: &[u8]) -> [u8; PROOF_BYTES] {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC takes any key length");
    mac.update(role);
    mac.update(nonce);
    let mut proof = [0; PROOF_BYTES];
    proof.copy_from_slice(&mac.finalize().into_bytes());
    proof
}

fn proof_holds(secret: &[u8; SECRET_BYTES], role: &[u8], nonce: &[u8], claimed: &[u8]) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC takes any key length");
    mac.update(role);
    mac.update(nonce);
    mac.verify_slice(claimed).is_ok()
}

/// A folder only this user can use (Unix: refuses folders that others
/// can enter, which the default locations never are).
fn private_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)?;
        let metadata = fs::metadata(dir)?;
        // SAFETY: plain libc call without arguments.
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("{} belongs to another user", dir.display()),
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("other users can open {}", dir.display()),
            ));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(dir)
    }
}

/// Writes `text` to `path` readable by this user only, atomically.
fn write_private(path: &Path, text: &str) -> io::Result<()> {
    let temp = path.with_extension("info.tmp");
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp, path)
}

// -- connections -------------------------------------------------------------------

enum Connection {
    #[cfg(unix)]
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Connection {
    fn set_timeout(&self, timeout: Duration) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Connection::Unix(stream) => {
                stream.set_read_timeout(Some(timeout))?;
                stream.set_write_timeout(Some(timeout))
            }
            Connection::Tcp(stream) => {
                stream.set_read_timeout(Some(timeout))?;
                stream.set_write_timeout(Some(timeout))
            }
        }
    }
}

impl Read for Connection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Connection::Unix(stream) => stream.read(buffer),
            Connection::Tcp(stream) => stream.read(buffer),
        }
    }
}

impl Write for Connection {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Connection::Unix(stream) => stream.write(bytes),
            Connection::Tcp(stream) => stream.write(bytes),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Connection::Unix(stream) => stream.flush(),
            Connection::Tcp(stream) => stream.flush(),
        }
    }
}

enum Listener {
    #[cfg(unix)]
    Unix(UnixListener),
    Tcp(TcpListener),
}

impl Listener {
    fn accept(&self) -> io::Result<Connection> {
        match self {
            #[cfg(unix)]
            Listener::Unix(listener) => listener
                .accept()
                .map(|(stream, _)| Connection::Unix(stream)),
            Listener::Tcp(listener) => listener.accept().map(|(stream, _)| Connection::Tcp(stream)),
        }
    }
}

fn connect(endpoint: &Endpoint, info: &Info, timeout: Duration) -> io::Result<Connection> {
    match endpoint.transport {
        #[cfg(unix)]
        Transport::Unix => UnixStream::connect(endpoint.file("sock")).map(Connection::Unix),
        Transport::Tcp => {
            TcpStream::connect_timeout(&SocketAddr::from((Ipv4Addr::LOCALHOST, info.port)), timeout)
                .map(Connection::Tcp)
        }
    }
}

fn read_frame(connection: &mut Connection, limit: u32) -> io::Result<Vec<u8>> {
    let mut length = [0; 4];
    connection.read_exact(&mut length)?;
    let length = u32::from_be_bytes(length);
    if length > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a message of {length} bytes is larger than {limit}"),
        ));
    }
    let mut body = vec![0; length as usize];
    connection.read_exact(&mut body)?;
    Ok(body)
}

fn write_frame(connection: &mut Connection, body: &[u8]) -> io::Result<()> {
    let length = u32::try_from(body.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "message too large"))?;
    connection.write_all(&length.to_be_bytes())?;
    connection.write_all(body)?;
    connection.flush()
}

// -- messages ----------------------------------------------------------------------

/// A path that survives JSON: UTF-8 as it is, anything else as hex of the
/// platform's encoding (as in `recent_files.toml`).
#[derive(Debug, Serialize, Deserialize)]
struct WirePath {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    os_path: Option<String>,
}

impl WirePath {
    fn from_path(path: &Path) -> Self {
        match path.to_str() {
            Some(text) => Self {
                path: Some(text.to_string()),
                os_path: None,
            },
            None => Self {
                path: None,
                os_path: Some(crate::recent::encode_os(path)),
            },
        }
    }

    fn to_path(&self) -> Result<PathBuf, String> {
        let path = match (&self.path, &self.os_path) {
            (_, Some(hex)) => crate::recent::decode_os(hex).ok_or("an unreadable path")?,
            (Some(text), None) => PathBuf::from(text),
            (None, None) => return Err("an empty path".to_string()),
        };
        if path.is_absolute() {
            Ok(path)
        } else {
            Err(format!("a relative path ({})", path.display()))
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WireRequest {
    #[serde(default)]
    tabs: Vec<WirePath>,
    #[serde(default)]
    panes: Vec<Vec<WirePath>>,
    #[serde(default)]
    views: Vec<(WirePath, ViewRequest)>,
    #[serde(default)]
    problems: Vec<Problem>,
}

fn encode(request: &OpenRequest) -> Vec<u8> {
    let wire = WireRequest {
        tabs: request
            .tabs
            .iter()
            .map(|path| WirePath::from_path(path))
            .collect(),
        panes: request
            .panes
            .iter()
            .map(|pane| pane.iter().map(|path| WirePath::from_path(path)).collect())
            .collect(),
        views: request
            .views
            .iter()
            .map(|(path, view)| (WirePath::from_path(path), *view))
            .collect(),
        problems: request.problems.clone(),
    };
    serde_json::to_vec(&wire).expect("requests serialise")
}

fn decode(body: &[u8]) -> Result<OpenRequest, String> {
    let wire: WireRequest =
        serde_json::from_slice(body).map_err(|error| format!("an unreadable request: {error}"))?;
    let count = wire.tabs.len() + wire.panes.iter().map(Vec::len).sum::<usize>() + wire.views.len();
    if count > MAX_PATHS {
        return Err(format!("{count} paths, more than {MAX_PATHS}"));
    }
    let paths = |paths: &[WirePath]| -> Result<Vec<PathBuf>, String> {
        paths.iter().map(WirePath::to_path).collect()
    };
    Ok(OpenRequest {
        tabs: paths(&wire.tabs)?,
        panes: wire
            .panes
            .iter()
            .map(|pane| paths(pane))
            .collect::<Result<_, _>>()?,
        views: wire
            .views
            .iter()
            .map(|(path, view)| Ok((path.to_path()?, *view)))
            .collect::<Result<_, String>>()?,
        problems: wire.problems,
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct Reply {
    ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

// -- the running eov ---------------------------------------------------------------

/// How the running eov answers a request: `Ok` once the files are taken.
pub(crate) type Answer = mpsc::Sender<Result<(), String>>;

/// The endpoint of the running eov. Dropping it stops serving, removes the
/// endpoint's files and releases the lock, so the next eov can take over.
pub(crate) struct Server {
    endpoint: Endpoint,
    /// Locked while this eov serves; the operating system unlocks it
    /// however the process ends.
    _lock: File,
    listener: Option<Listener>,
    secret: [u8; SECRET_BYTES],
    closing: Arc<AtomicBool>,
}

impl Server {
    /// Takes the endpoint, unless another eov holds it (`Ok(None)`).
    fn bind(endpoint: &Endpoint) -> io::Result<Option<Self>> {
        private_dir(&endpoint.dir)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(endpoint.file("lock"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => return Ok(None),
            Err(fs::TryLockError::Error(error)) => return Err(error),
        }
        // This eov owns the endpoint now: anything there is stale.
        let secret = random_bytes::<SECRET_BYTES>();
        let (listener, port) = match endpoint.transport {
            #[cfg(unix)]
            Transport::Unix => {
                use std::os::unix::fs::PermissionsExt;
                let socket = endpoint.file("sock");
                match fs::remove_file(&socket) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
                let listener = UnixListener::bind(&socket)?;
                fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
                (Listener::Unix(listener), 0)
            }
            Transport::Tcp => {
                let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
                let port = listener.local_addr()?.port();
                (Listener::Tcp(listener), port)
            }
        };
        let info = Info {
            port,
            pid: std::process::id(),
            secret,
        };
        write_private(&endpoint.file("info"), &info.to_text())?;
        Ok(Some(Self {
            endpoint: endpoint.clone(),
            _lock: lock,
            listener: Some(listener),
            secret,
            closing: Arc::new(AtomicBool::new(false)),
        }))
    }

    /// Serves requests on a thread. `deliver` gets each request with the
    /// channel to answer on; it must not block (it hands the request to
    /// the UI thread).
    pub(crate) fn serve(
        &mut self,
        timing: Timing,
        deliver: impl Fn(OpenRequest, Answer) + Send + 'static,
    ) {
        let Some(listener) = self.listener.take() else {
            return;
        };
        let secret = self.secret;
        let closing = Arc::clone(&self.closing);
        let spawned = std::thread::Builder::new()
            .name("eov-instance".to_string())
            .spawn(move || {
                loop {
                    let connection = match listener.accept() {
                        Ok(connection) => connection,
                        Err(error) => {
                            if closing.load(Ordering::Acquire) {
                                return;
                            }
                            tracing::debug!("Single-instance endpoint: {error}");
                            std::thread::sleep(Duration::from_millis(100));
                            continue;
                        }
                    };
                    if let Err(error) = handle(connection, &secret, timing, &closing, &deliver) {
                        tracing::debug!("Refused a request from another eov: {error}");
                    }
                }
            });
        if let Err(error) = spawned {
            tracing::warn!("Other eovs cannot hand files to this one: {error}");
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.closing.store(true, Ordering::Release);
        let _ = fs::remove_file(self.endpoint.file("info"));
        #[cfg(unix)]
        if self.endpoint.transport == Transport::Unix {
            let _ = fs::remove_file(self.endpoint.file("sock"));
        }
    }
}

/// Serves one connection: mutual proof of the secret, the request, the
/// answer.
fn handle(
    mut connection: Connection,
    secret: &[u8; SECRET_BYTES],
    timing: Timing,
    closing: &AtomicBool,
    deliver: &impl Fn(OpenRequest, Answer),
) -> io::Result<()> {
    connection.set_timeout(timing.io)?;
    let mut magic = [0; MAGIC.len()];
    connection.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not an eov"));
    }
    let mut client_nonce = [0; NONCE_BYTES];
    connection.read_exact(&mut client_nonce)?;
    let server_nonce = random_bytes::<NONCE_BYTES>();
    connection.write_all(&proof(secret, b"server", &client_nonce))?;
    connection.write_all(&server_nonce)?;
    connection.flush()?;
    let mut client_proof = [0; PROOF_BYTES];
    connection.read_exact(&mut client_proof)?;
    if !proof_holds(secret, b"client", &server_nonce, &client_proof) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "the client does not know the secret",
        ));
    }
    let outcome = match read_frame(&mut connection, MAX_REQUEST_BYTES) {
        Err(error) if error.kind() == io::ErrorKind::InvalidData => Err(error.to_string()),
        Err(error) => return Err(error),
        Ok(_) if closing.load(Ordering::Acquire) => Err("eov is closing".to_string()),
        Ok(body) => decode(&body).and_then(|request| {
            let (answer, answered) = mpsc::channel();
            deliver(request, answer);
            match answered.recv_timeout(timing.answer) {
                Ok(outcome) => outcome,
                Err(mpsc::RecvTimeoutError::Timeout) => Err("eov did not respond".to_string()),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err("eov is closing".to_string()),
            }
        }),
    };
    let reply = match outcome {
        Ok(()) => Reply {
            ok: true,
            message: None,
        },
        Err(message) => Reply {
            ok: false,
            message: Some(message),
        },
    };
    write_frame(
        &mut connection,
        &serde_json::to_vec(&reply).expect("replies serialise"),
    )
}

// -- the second eov ---------------------------------------------------------------

/// Why a request was not handed over.
#[derive(Debug)]
enum ForwardError {
    /// No eov answers at the endpoint (yet, or any more).
    Unreachable(io::Error),
    /// The running eov answered but did not take the files.
    Refused(String),
    /// The endpoint is not a trustworthy eov, or the exchange broke off.
    Failed(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::Unreachable(error) => write!(f, "no answer: {error}"),
            ForwardError::Refused(message) | ForwardError::Failed(message) => f.write_str(message),
        }
    }
}

/// Opens an authenticated connection to the running eov: it must prove
/// the secret before this side does, and before any path is sent.
fn open_session(
    endpoint: &Endpoint,
    info: &Info,
    timing: Timing,
) -> Result<Connection, ForwardError> {
    let mut connection = connect(endpoint, info, timing.io).map_err(ForwardError::Unreachable)?;
    connection
        .set_timeout(timing.io)
        .map_err(ForwardError::Unreachable)?;
    let client_nonce = random_bytes::<NONCE_BYTES>();
    let mut server_proof = [0; PROOF_BYTES];
    let mut server_nonce = [0; NONCE_BYTES];
    connection
        .write_all(MAGIC)
        .and_then(|()| connection.write_all(&client_nonce))
        .and_then(|()| connection.flush())
        .and_then(|()| connection.read_exact(&mut server_proof))
        .and_then(|()| connection.read_exact(&mut server_nonce))
        .map_err(ForwardError::Unreachable)?;
    if !proof_holds(&info.secret, b"server", &client_nonce, &server_proof) {
        return Err(ForwardError::Failed(format!(
            "the program listening at {} is not the eov that registered there",
            endpoint.dir.display()
        )));
    }
    connection
        .write_all(&proof(&info.secret, b"client", &server_nonce))
        .map_err(|error| ForwardError::Failed(error.to_string()))?;
    Ok(connection)
}

/// Hands `request` to the running eov and waits until it took it.
fn forward(endpoint: &Endpoint, request: &OpenRequest, timing: Timing) -> Result<(), ForwardError> {
    let info = read_info(endpoint).map_err(ForwardError::Unreachable)?;
    let body = encode(request);
    if body.len() > MAX_REQUEST_BYTES as usize {
        return Err(ForwardError::Failed(
            "too many files to hand to the running eov".to_string(),
        ));
    }
    allow_to_come_forward(info.pid);
    let mut connection = open_session(endpoint, &info, timing)?;
    write_frame(&mut connection, &body).map_err(|error| ForwardError::Failed(error.to_string()))?;
    connection
        .set_timeout(timing.answer + timing.io)
        .map_err(|error| ForwardError::Failed(error.to_string()))?;
    let reply = read_frame(&mut connection, MAX_REPLY_BYTES)
        .map_err(|error| ForwardError::Failed(format!("no answer ({error})")))?;
    let reply: Reply = serde_json::from_slice(&reply)
        .map_err(|error| ForwardError::Failed(format!("an unreadable answer ({error})")))?;
    if reply.ok {
        Ok(())
    } else {
        Err(ForwardError::Refused(
            reply.message.unwrap_or_else(|| "refused".to_string()),
        ))
    }
}

/// Windows only lets the foreground program pass on the foreground: let
/// the running eov raise its window.
fn allow_to_come_forward(pid: u32) {
    #[cfg(windows)]
    // SAFETY: plain Win32 call; failure only means the window stays behind.
    unsafe {
        let _ = windows_sys::Win32::UI::WindowsAndMessaging::AllowSetForegroundWindow(pid);
    }
    #[cfg(not(windows))]
    let _ = pid;
}

/// The outcome of [`claim`].
pub(crate) enum Claim {
    /// No eov was running: this one serves the endpoint (keep the server
    /// while it runs).
    Primary(Server),
    /// The running eov took the request; this one can exit.
    Forwarded,
    /// Start without serving, for this reason.
    Standalone(String),
}

/// Becomes the running eov, or hands `request` to the one that is.
pub(crate) fn claim(endpoint: &Endpoint, request: &OpenRequest, timing: Timing) -> Claim {
    let deadline = Instant::now() + timing.patience;
    loop {
        match Server::bind(endpoint) {
            Ok(Some(server)) => return Claim::Primary(server),
            Ok(None) => {}
            Err(error) => {
                return Claim::Standalone(format!(
                    "cannot look for a running eov in {}: {error}",
                    endpoint.dir.display()
                ));
            }
        }
        match forward(endpoint, request, timing) {
            Ok(()) => return Claim::Forwarded,
            // Starting up or exiting: try again for a moment.
            Err(ForwardError::Unreachable(_) | ForwardError::Refused(_))
                if Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Claim::Standalone(format!(
                    "the running eov did not take the files ({error}); starting another"
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::{Coordinate, Point, Zoom};

    fn transports() -> Vec<Transport> {
        #[cfg(unix)]
        {
            vec![Transport::Unix, Transport::Tcp]
        }
        #[cfg(not(unix))]
        {
            vec![Transport::Tcp]
        }
    }

    fn fast() -> Timing {
        Timing {
            io: Duration::from_millis(500),
            answer: Duration::from_secs(5),
            patience: Duration::from_millis(300),
        }
    }

    fn absolute(name: &str) -> PathBuf {
        std::env::temp_dir().join("slides").join(name)
    }

    fn sample_request() -> OpenRequest {
        OpenRequest {
            tabs: vec![absolute("a.svs"), absolute("Case 12, H&E.svs")],
            panes: vec![vec![absolute("b.ndpi")]],
            views: vec![(
                absolute("a.svs"),
                ViewRequest {
                    center: Some(Point {
                        x: Coordinate::Um(300.0),
                        y: Coordinate::Um(200.0),
                    }),
                    zoom: Some(Zoom::Magnification(20.0)),
                },
            )],
            problems: vec![Problem {
                subject: "gone.svs".to_string(),
                reason: "no such file or folder".to_string(),
            }],
        }
    }

    /// A running eov that records what it is handed.
    fn running(endpoint: &Endpoint) -> (Server, mpsc::Receiver<OpenRequest>) {
        let Claim::Primary(mut server) = claim(endpoint, &OpenRequest::default(), fast()) else {
            panic!("the first eov must serve the endpoint");
        };
        let (sender, received) = mpsc::channel();
        server.serve(fast(), move |request, answer| {
            sender.send(request).unwrap();
            answer.send(Ok(())).unwrap();
        });
        (server, received)
    }

    #[test]
    fn a_second_eov_hands_its_request_to_the_first() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            let (_server, received) = running(&endpoint);
            let request = sample_request();
            assert!(matches!(
                claim(&endpoint, &request, fast()),
                Claim::Forwarded
            ));
            assert_eq!(
                received.recv_timeout(Duration::from_secs(5)).unwrap(),
                request
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_are_handed_over_exactly() {
        use std::os::unix::ffi::OsStringExt;
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), Transport::Unix);
        let (_server, received) = running(&endpoint);
        let odd = PathBuf::from(std::ffi::OsString::from_vec(b"/slides/\xff.svs".to_vec()));
        let request = OpenRequest {
            tabs: vec![odd.clone()],
            ..OpenRequest::default()
        };
        assert!(matches!(
            claim(&endpoint, &request, fast()),
            Claim::Forwarded
        ));
        assert_eq!(
            received.recv_timeout(Duration::from_secs(5)).unwrap().tabs,
            [odd]
        );
    }

    #[test]
    fn a_crashed_eov_does_not_block_start_up() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            private_dir(&endpoint.dir).unwrap();
            // What a crashed eov leaves behind: its files, but no lock.
            fs::write(endpoint.file("lock"), "").unwrap();
            fs::write(
                endpoint.file("info"),
                Info {
                    port: 1,
                    pid: 1,
                    secret: [1; SECRET_BYTES],
                }
                .to_text(),
            )
            .unwrap();
            #[cfg(unix)]
            if transport == Transport::Unix {
                drop(UnixListener::bind(endpoint.file("sock")).unwrap());
                assert!(endpoint.file("sock").exists());
            }
            let Claim::Primary(server) = claim(&endpoint, &OpenRequest::default(), fast()) else {
                panic!("a stale endpoint must be taken over");
            };
            assert_ne!(read_info(&endpoint).unwrap().secret, [1; SECRET_BYTES]);
            drop(server);
            assert!(!endpoint.file("info").exists());
        }
    }

    #[test]
    fn only_one_eov_holds_the_endpoint() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            let first = Server::bind(&endpoint).unwrap().expect("free");
            assert!(Server::bind(&endpoint).unwrap().is_none());
            drop(first);
            assert!(Server::bind(&endpoint).unwrap().is_some());
        }
    }

    #[test]
    fn an_eov_that_does_not_answer_is_given_up_on() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            // Bound but never serving, like a hung eov.
            let _hung = Server::bind(&endpoint).unwrap().expect("free");
            let started = Instant::now();
            assert!(matches!(
                claim(&endpoint, &sample_request(), fast()),
                Claim::Standalone(_)
            ));
            assert!(started.elapsed() < Duration::from_secs(5));
        }
    }

    #[test]
    fn requests_must_prove_the_secret() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            let (_server, received) = running(&endpoint);
            let mut info = read_info(&endpoint).unwrap();
            info.secret = [9; SECRET_BYTES];
            // The server's proof does not hold for the wrong secret: the
            // client stops before sending anything.
            let error = open_session(&endpoint, &info, fast()).err().unwrap();
            assert!(matches!(error, ForwardError::Failed(_)), "{error}");

            // A client that skips the proof is cut off.
            let info = read_info(&endpoint).unwrap();
            let mut connection = connect(&endpoint, &info, fast().io).unwrap();
            connection.set_timeout(fast().io).unwrap();
            connection.write_all(MAGIC).unwrap();
            connection.write_all(&[0; NONCE_BYTES]).unwrap();
            let mut answer = [0; PROOF_BYTES + NONCE_BYTES];
            connection.read_exact(&mut answer).unwrap();
            connection.write_all(&[0; PROOF_BYTES]).unwrap();
            let _ = write_frame(&mut connection, &encode(&sample_request()));
            assert!(read_frame(&mut connection, MAX_REPLY_BYTES).is_err());
            assert!(received.recv_timeout(Duration::from_millis(200)).is_err());
        }
    }

    #[test]
    fn oversized_and_malformed_requests_are_refused() {
        for transport in transports() {
            let dir = tempfile::tempdir().unwrap();
            let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), transport);
            let (_server, received) = running(&endpoint);
            let info = read_info(&endpoint).unwrap();

            let mut connection = open_session(&endpoint, &info, fast()).unwrap();
            connection
                .write_all(&(MAX_REQUEST_BYTES + 1).to_be_bytes())
                .unwrap();
            let reply: Reply =
                serde_json::from_slice(&read_frame(&mut connection, MAX_REPLY_BYTES).unwrap())
                    .unwrap();
            assert!(!reply.ok);

            let mut connection = open_session(&endpoint, &info, fast()).unwrap();
            write_frame(&mut connection, br#"{"tabs":[{"path":"relative.svs"}]}"#).unwrap();
            let reply: Reply =
                serde_json::from_slice(&read_frame(&mut connection, MAX_REPLY_BYTES).unwrap())
                    .unwrap();
            assert!(!reply.ok && reply.message.unwrap().contains("relative"));
            assert!(received.recv_timeout(Duration::from_millis(200)).is_err());
        }
    }

    #[test]
    fn files_are_not_sent_to_an_impostor() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), Transport::Tcp);
        private_dir(&endpoint.dir).unwrap();
        // Another program on the port eov registered, not knowing the
        // secret, while the lock is held.
        let impostor = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let info = Info {
            port: impostor.local_addr().unwrap().port(),
            pid: 1,
            secret: [7; SECRET_BYTES],
        };
        fs::write(endpoint.file("info"), info.to_text()).unwrap();
        let lock = File::create(endpoint.file("lock")).unwrap();
        lock.lock().unwrap();
        let seen = std::thread::spawn(move || {
            let (mut stream, _) = impostor.accept().unwrap();
            stream.write_all(&[0; PROOF_BYTES + NONCE_BYTES]).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut seen = Vec::new();
            let _ = stream.read_to_end(&mut seen);
            seen
        });
        assert!(matches!(
            forward(&endpoint, &sample_request(), fast()),
            Err(ForwardError::Failed(_))
        ));
        assert_eq!(
            seen.join().unwrap().len(),
            MAGIC.len() + NONCE_BYTES,
            "only the challenge may reach an impostor"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_endpoint_folder_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::new(dir.path().join("run"), "eov-test".into(), Transport::Unix);
        let _server = Server::bind(&endpoint).unwrap().unwrap();
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&endpoint.dir), 0o700);
        assert_eq!(mode(&endpoint.file("info")), 0o600);

        let shared = dir.path().join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o755)).unwrap();
        let open = Endpoint::new(shared, "eov-test".into(), Transport::Unix);
        assert!(Server::bind(&open).is_err());
    }

    #[test]
    fn each_setup_has_its_own_endpoint() {
        let a = Endpoint::for_setup(&absolute("a.toml"), &absolute("plugins"));
        let b = Endpoint::for_setup(&absolute("b.toml"), &absolute("plugins"));
        assert_ne!(a.name, b.name);
        assert!(a.dir.starts_with(crate::paths::isolate_for_tests()));
    }
}
