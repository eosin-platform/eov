//! eov's model store (AUDIT §14.5): the model files of AI plugins, kept by
//! content and verified.
//!
//! `<data>/models/<sha256>/` holds one model: its main file (whose SHA-256
//! names the folder), the companion files it needs next to it (the external
//! weights of an ONNX model) and [`RECORD_FILE`], what it is, where it came
//! from, when, and who asked. Downloads stream into
//! `.partial/<sha256>.part` while they are hashed, are checked against the
//! expected size and SHA-256, and only then move into place; a file that
//! does not match is deleted and the error says what was expected and what
//! arrived. An interrupted download resumes with an HTTP `Range` request
//! when the server allows it; a cancelled one starts over. A model is
//! downloaded once for every plugin and every eov of the user, and models
//! are never deleted except by `eov model remove`.
//!
//! The administrator's policy can turn downloads off
//! (`allow_model_downloads`), restrict them to the site's servers
//! (`model_url_prefixes`), give the site's copies of models by SHA-256
//! (`model_urls`, which win over the URL a plugin or the command line
//! gives), and provide read-only stores (`model_dirs`) with the same
//! layout. Downloads go through the updater's transport (proxies, the
//! operating system's certificates) and are recorded in the audit log.
//!
//! `eov model` ([`cli`]) and the plugin methods `model.fetch`,
//! `model.resolve` and `model.list` (`plugin_runtime::host_api`,
//! `host_ui::models`) are built on [`ModelStore`].

pub(crate) mod cli;
mod fetch;
#[cfg(test)]
pub(crate) mod test_server;

pub(crate) use fetch::{FetchError, FetchProgress, ModelClient};

use crate::update::policy::{Policy, redacted_url};
use eov_plugin_api::i18n::{isolate, tr};
use eov_plugin_api::protocol::{
    ModelCompanion, ModelFetchArgs, ModelRecord, check_model_file_name, is_sha256_hex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use url::Url;

/// The record of a stored model, in its folder.
pub(crate) const RECORD_FILE: &str = "eov-model.json";
/// Downloads in progress, below the store's root.
const PARTIAL_DIR: &str = ".partial";
/// Serialises changes to the store's folders between processes.
const STORE_LOCK: &str = ".lock";
const RECORD_FORMAT: u32 = 1;
/// The largest model file accepted when its size is not given.
pub(crate) const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Partial downloads nobody resumed for this long are deleted.
const PARTIAL_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 3600);
/// Who `eov model` records as having asked.
pub(crate) const CLI_REQUESTER: &str = "eov";

fn record_format() -> u32 {
    RECORD_FORMAT
}

/// What [`RECORD_FILE`] says about a stored model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Record {
    #[serde(default = "record_format")]
    pub format: u32,
    /// SHA-256 of the main file (lower-case hex), the folder's name.
    pub sha256: String,
    pub file_name: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Where it was downloaded from, without credentials or query; `None`
    /// for an import.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// When it was stored (UTC, RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fetched_at: Option<String>,
    /// The plugin that asked for it, or `eov` for the command line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_by: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub companions: Vec<CompanionRecord>,
}

/// A companion file in a [`Record`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CompanionRecord {
    pub file_name: String,
    pub sha256: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A model in a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredModel {
    pub record: Record,
    pub dir: PathBuf,
    /// From one of the administrator's read-only `model_dirs`.
    pub read_only: bool,
}

impl StoredModel {
    /// The main file.
    pub fn path(&self) -> PathBuf {
        self.dir.join(&self.record.file_name)
    }

    /// The name to show: the record's name, else the file's.
    pub fn label(&self) -> String {
        self.record
            .name
            .clone()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| self.record.file_name.clone())
    }

    /// Bytes of every file.
    pub fn total_size(&self) -> u64 {
        self.record.size + self.record.companions.iter().map(|c| c.size).sum::<u64>()
    }

    /// Every file: (name, SHA-256, size).
    fn files(&self) -> Vec<(&str, &str, u64)> {
        std::iter::once((
            self.record.file_name.as_str(),
            self.record.sha256.as_str(),
            self.record.size,
        ))
        .chain(
            self.record
                .companions
                .iter()
                .map(|c| (c.file_name.as_str(), c.sha256.as_str(), c.size)),
        )
        .collect()
    }

    /// Whether every file is there with its recorded size (no hashing).
    pub fn files_present(&self) -> bool {
        self.files().iter().all(|(name, _, size)| {
            fs::metadata(self.dir.join(name))
                .is_ok_and(|meta| meta.is_file() && meta.len() == *size)
        })
    }

    /// Whether it holds what `request` asks for: the main file and every
    /// companion, by SHA-256 and name.
    pub fn satisfies(&self, request: &ModelRequest) -> bool {
        self.record.sha256 == request.main.sha256
            && request.companions.iter().all(|wanted| {
                self.record.companions.iter().any(|have| {
                    have.sha256 == wanted.sha256
                        && wanted
                            .file_name
                            .as_deref()
                            .is_none_or(|name| name == have.file_name)
                })
            })
    }

    /// The record as plugins see it.
    pub fn to_protocol(&self) -> ModelRecord {
        let text = |path: PathBuf| path.to_string_lossy().into_owned();
        ModelRecord {
            sha256: self.record.sha256.clone(),
            path: text(self.path()),
            file_name: self.record.file_name.clone(),
            size: self.record.size,
            name: self.record.name.clone(),
            version: self.record.version.clone(),
            source: self.record.source.clone(),
            fetched_at: self.record.fetched_at.clone(),
            requested_by: self.record.requested_by.clone(),
            companions: self
                .record
                .companions
                .iter()
                .map(|c| ModelCompanion {
                    file_name: c.file_name.clone(),
                    sha256: c.sha256.clone(),
                    size: c.size,
                    path: text(self.dir.join(&c.file_name)),
                })
                .collect(),
            read_only: self.read_only,
        }
    }
}

/// One file of a model to fetch, by content.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct FileRequest {
    /// Lower-case hex.
    pub sha256: String,
    pub size: Option<u64>,
    pub file_name: Option<String>,
    /// For companions, relative to the main file's URL.
    pub url: Option<String>,
}

/// A model to fetch: its main file, its companions and what to record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ModelRequest {
    pub main: FileRequest,
    pub companions: Vec<FileRequest>,
    pub name: Option<String>,
    pub version: Option<String>,
    /// A plugin id, or [`CLI_REQUESTER`].
    pub requested_by: String,
}

/// Trims text a plugin or a person gave, keeping at most `max` characters.
fn clean_text(text: Option<&str>, max: usize) -> Option<String> {
    let text = text?.trim();
    (!text.is_empty()).then(|| text.chars().take(max).collect())
}

impl ModelRequest {
    /// A plugin's `model.fetch` arguments, checked.
    pub fn from_args(args: &ModelFetchArgs, requested_by: &str) -> Result<Self, String> {
        args.validate()?;
        let file = |sha256: &str, size, file_name: &Option<String>, url: &Option<String>| {
            Ok::<_, String>(FileRequest {
                sha256: sha256.to_ascii_lowercase(),
                size,
                file_name: file_name.clone(),
                url: clean_text(url.as_deref(), 4096),
            })
        };
        let main = file(&args.sha256, args.size, &args.file_name, &args.url)?;
        let companions = args
            .companions
            .iter()
            .map(|c| file(&c.sha256, c.size, &c.file_name, &c.url))
            .collect::<Result<Vec<_>, _>>()?;
        for name in std::iter::once(&main.file_name)
            .chain(companions.iter().map(|c| &c.file_name))
            .flatten()
        {
            check_store_file_name(name)?;
        }
        Ok(Self {
            main,
            companions,
            name: clean_text(args.name.as_deref(), 200),
            version: clean_text(args.version.as_deref(), 100),
            requested_by: requested_by.to_string(),
        })
    }

    /// What to call it when asking or reporting.
    pub fn label(&self) -> String {
        self.name
            .clone()
            .or_else(|| self.main.file_name.clone())
            .unwrap_or_else(|| {
                // Translators: a model that has no name, by the start of
                // its SHA-256. It takes the place of a model's name in other
                // messages, always after a verb or a preposition: "Download
                // {}?", "Cannot download {}", "no download address is set
                // for {}"; write it so that it fits there as it is.
                tr!("the model {}", &self.main.sha256[..12])
            })
    }
}

/// A file name the store accepts (not its own record); why not in the
/// interface's language.
fn check_store_file_name(name: &str) -> Result<(), String> {
    check_model_file_name(name)?;
    if is_record_file(name) {
        return Err(tr!(
            "'{}' is reserved for eov's record of the model",
            isolate(name)
        ));
    }
    Ok(())
}

/// Whether `name` is the store's own record of a model.
fn is_record_file(name: &str) -> bool {
    name.eq_ignore_ascii_case(RECORD_FILE)
}

/// The last path segment of a URL, percent-decoded.
fn url_file_name(url: &Url) -> Option<String> {
    let segment = url.path_segments()?.next_back()?;
    let decoded = percent_encoding::percent_decode_str(segment)
        .decode_utf8()
        .ok()?
        .into_owned();
    (!decoded.is_empty()).then_some(decoded)
}

/// One file to download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedFile {
    pub url: Url,
    pub sha256: String,
    pub size: Option<u64>,
    pub file_name: String,
    /// The model's main file (else a companion).
    pub main: bool,
    /// The URL is the administrator's copy (`model_urls`).
    pub from_policy: bool,
}

/// What a download will fetch: the files the store lacks, main file first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DownloadPlan {
    pub request: ModelRequest,
    /// The main file's name and URL, even when it is stored already.
    pub main_file_name: String,
    pub main_url: Url,
    pub files: Vec<PlannedFile>,
}

impl DownloadPlan {
    /// Bytes to download, when every size is known.
    pub fn known_size(&self) -> Option<u64> {
        self.files.iter().map(|file| file.size).sum()
    }

    /// The servers it contacts, for questions and messages.
    pub fn hosts(&self) -> Vec<String> {
        let mut hosts: Vec<String> = Vec::new();
        for file in &self.files {
            let host = file.url.host_str().unwrap_or_default().to_string();
            if !hosts.contains(&host) {
                hosts.push(host);
            }
        }
        hosts
    }
}

/// Checks the scheme of a model URL and the policy: `https`, or `http`
/// for this computer or a server the policy names.
pub(crate) fn check_url(url: &Url, policy: &Policy) -> Result<(), FetchError> {
    let refused = |reason: String| FetchError::Refused(reason);
    match url.scheme() {
        "https" => {}
        "http"
            if url
                .host_str()
                .is_some_and(crate::update::transport::is_loopback_host)
                || policy.allows_plain_http(url) => {}
        "http" => {
            // Translators: model_url_prefixes is the name of a setting.
            return Err(refused(tr!(
                "{} uses plain HTTP; models are downloaded over HTTPS (or from a server the administrator's model_url_prefixes names)",
                isolate(redacted_url(url))
            )));
        }
        other => {
            return Err(refused(tr!(
                "{url} is not an http(s) URL ({scheme})",
                url = isolate(redacted_url(url)),
                scheme = other
            )));
        }
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(refused(tr!("{} names no server", isolate(url))));
    }
    policy
        .ensure_model_download_allowed(url)
        .map_err(|error| refused(format!("{error:#}")))
}

/// Characters a file name cannot carry as they are in a URL path segment.
const SEGMENT: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'\\')
    .add(b'^')
    .add(b'|')
    .add(b'[')
    .add(b']');

/// Works out where each missing file of `request` comes from: the
/// administrator's copy (`model_urls`) if there is one, else the URL given;
/// companions without a URL of their own come from the main file's folder.
/// Files keep the names the request gives them (else the given URL's last
/// part). `stored` is what the store has already.
pub(crate) fn plan_download(
    request: &ModelRequest,
    policy: &Policy,
    stored: Option<&StoredModel>,
) -> Result<DownloadPlan, FetchError> {
    let parse = |text: &str| {
        Url::parse(text).map_err(|error| {
            FetchError::Invalid(tr!(
                "'{text}' is not a URL: {error}",
                text = isolate(text),
                error = error
            ))
        })
    };
    let given_main = request.main.url.as_deref().map(parse).transpose()?;
    let (main_url, main_from_policy) = match (policy.model_url(&request.main.sha256), &given_main) {
        (Some(url), _) => (parse(url)?, true),
        (None, Some(url)) => (url.clone(), false),
        (None, None) => {
            return Err(FetchError::NoUrl {
                what: request.label(),
            });
        }
    };
    let name_of = |given: &Option<String>, urls: &[&Url]| -> Result<String, FetchError> {
        let name = match given {
            Some(name) => name.clone(),
            None => urls
                .iter()
                .find_map(|url| url_file_name(url))
                .ok_or_else(|| {
                    FetchError::Invalid(tr!(
                        "{} does not end in a file name; give the file's name",
                        isolate(
                            urls.first()
                                .map(|url| redacted_url(url))
                                .unwrap_or_default()
                        )
                    ))
                })?,
        };
        check_store_file_name(&name).map_err(FetchError::Invalid)?;
        Ok(name)
    };
    let main_file_name = match stored {
        Some(stored) => stored.record.file_name.clone(),
        None => {
            let urls: Vec<&Url> = given_main.iter().chain([&main_url]).collect();
            name_of(&request.main.file_name, &urls)?
        }
    };
    let mut files = Vec::new();
    if stored.is_none() {
        check_url(&main_url, policy)?;
        files.push(PlannedFile {
            url: main_url.clone(),
            sha256: request.main.sha256.clone(),
            size: request.main.size,
            file_name: main_file_name.clone(),
            main: true,
            from_policy: main_from_policy,
        });
    }
    let mut names = vec![main_file_name.to_ascii_lowercase()];
    for companion in &request.companions {
        // Where it would come from without the administrator's copy.
        let reference = match (&companion.url, &companion.file_name) {
            (Some(url), _) => url.clone(),
            (None, Some(name)) => percent_encoding::utf8_percent_encode(name, SEGMENT).to_string(),
            (None, None) => {
                return Err(FetchError::Invalid(tr!(
                    "a companion file needs a URL or a file name"
                )));
            }
        };
        let given = main_url.join(&reference).map_err(|error| {
            FetchError::Invalid(tr!(
                "'{text}' is not a URL: {error}",
                text = isolate(&reference),
                error = error
            ))
        })?;
        let (url, from_policy) = match policy.model_url(&companion.sha256) {
            Some(url) => (parse(url)?, true),
            None => (given.clone(), false),
        };
        let file_name = name_of(&companion.file_name, &[&given, &url])?;
        if names.contains(&file_name.to_ascii_lowercase()) {
            return Err(FetchError::Invalid(tr!(
                "two files of the model are named '{}'",
                isolate(&file_name)
            )));
        }
        names.push(file_name.to_ascii_lowercase());
        let present = stored.is_some_and(|stored| {
            stored
                .record
                .companions
                .iter()
                .any(|c| c.sha256 == companion.sha256 && c.file_name == file_name)
        });
        if present {
            continue;
        }
        check_url(&url, policy)?;
        files.push(PlannedFile {
            url,
            sha256: companion.sha256.clone(),
            size: companion.size,
            file_name,
            main: false,
            from_policy,
        });
    }
    Ok(DownloadPlan {
        request: request.clone(),
        main_file_name,
        main_url,
        files,
    })
}

/// A file ready to move into a model's folder.
struct StagedFile {
    path: PathBuf,
    file_name: String,
    sha256: String,
    size: u64,
    source: Option<String>,
}

/// One model's verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifyReport {
    pub dir: PathBuf,
    /// `None` when the folder holds no readable record.
    pub model: Option<StoredModel>,
    /// What is wrong; empty when every file matches its record.
    pub problems: Vec<String>,
}

/// Locks a file exclusively, waiting while another process holds it;
/// gives up when `cancel` is set.
fn lock_file(path: &Path, cancel: &AtomicBool) -> Result<File, FetchError> {
    let local = |error: std::io::Error| FetchError::Local {
        path: path.to_path_buf(),
        message: error.to_string(),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(local)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(local)?;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(fs::TryLockError::WouldBlock) => {
                if cancel.load(Ordering::Relaxed) {
                    return Err(FetchError::Cancelled);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(fs::TryLockError::Error(error)) => return Err(local(error)),
        }
    }
}

/// SHA-256 and size of a file.
pub(crate) fn hash_file(path: &Path, cancel: &AtomicBool) -> std::io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut size = 0_u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "cancelled",
            ));
        }
        let read = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

/// Writes `bytes` to `path` through a temporary file and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let directory = path.parent().unwrap_or(Path::new("."));
    let mut file = tempfile::Builder::new()
        .prefix(".eov-model-")
        .tempfile_in(directory)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

/// A user's model store and the administrator's read-only ones.
#[derive(Debug, Clone)]
pub(crate) struct ModelStore {
    root: PathBuf,
    read_only: Vec<PathBuf>,
}

impl ModelStore {
    pub fn new(root: PathBuf, read_only: Vec<PathBuf>) -> Self {
        Self { root, read_only }
    }

    /// `<data>/models`, with the policy's `model_dirs`.
    pub fn for_user(policy: &Policy) -> Self {
        Self::new(crate::paths::models_dir(), policy.model_dirs.clone())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn partial_dir(&self) -> PathBuf {
        self.root.join(PARTIAL_DIR)
    }

    /// The model in the folder `dir`, if its record is readable and names it
    /// (why not in English: the log and `eov model verify` say it).
    fn read_entry(dir: &Path, read_only: bool) -> Result<StoredModel, String> {
        let name = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = fs::read_to_string(dir.join(RECORD_FILE))
            .map_err(|error| format!("{RECORD_FILE} cannot be read: {error}"))?;
        let record: Record = serde_json::from_str(&text)
            .map_err(|error| format!("{RECORD_FILE} is not valid: {error}"))?;
        if record.sha256 != name {
            return Err(format!(
                "{RECORD_FILE} describes {}, not {name}",
                record.sha256
            ));
        }
        for file_name in
            std::iter::once(&record.file_name).chain(record.companions.iter().map(|c| &c.file_name))
        {
            check_model_file_name(file_name)?;
            if is_record_file(file_name) {
                return Err(format!(
                    "'{file_name}' is reserved for eov's record of the model"
                ));
            }
        }
        Ok(StoredModel {
            record,
            dir: dir.to_path_buf(),
            read_only,
        })
    }

    fn roots(&self) -> impl Iterator<Item = (&Path, bool)> {
        std::iter::once((self.root.as_path(), false))
            .chain(self.read_only.iter().map(|dir| (dir.as_path(), true)))
    }

    /// Every model folder: (folder, read-only), the user's first.
    fn entry_dirs(&self) -> Vec<(PathBuf, bool)> {
        let mut dirs = Vec::new();
        for (root, read_only) in self.roots() {
            let Ok(entries) = fs::read_dir(root) else {
                continue;
            };
            let mut found: Vec<PathBuf> = entries
                .flatten()
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            is_sha256_hex(name) && name == name.to_ascii_lowercase()
                        })
                })
                .collect();
            found.sort();
            dirs.extend(found.into_iter().map(|dir| (dir, read_only)));
        }
        dirs
    }

    /// Every stored model, the user's first; a model in several stores is
    /// listed once.
    pub fn list(&self) -> Vec<StoredModel> {
        let mut models: Vec<StoredModel> = Vec::new();
        for (dir, read_only) in self.entry_dirs() {
            match Self::read_entry(&dir, read_only) {
                Ok(model) => {
                    if !models
                        .iter()
                        .any(|m| m.record.sha256 == model.record.sha256)
                    {
                        models.push(model);
                    }
                }
                Err(error) => {
                    tracing::warn!("Skipping the model folder {}: {error}", dir.display())
                }
            }
        }
        models
    }

    /// The stored model whose main file has this SHA-256 and whose files
    /// are all there (by size; `eov model verify` hashes them).
    pub fn resolve(&self, sha256: &str) -> Option<StoredModel> {
        let sha256 = sha256.to_ascii_lowercase();
        if !is_sha256_hex(&sha256) {
            return None;
        }
        for (root, read_only) in self.roots() {
            let dir = root.join(&sha256);
            if !dir.is_dir() {
                continue;
            }
            match Self::read_entry(&dir, read_only) {
                Ok(model) if model.files_present() => return Some(model),
                Ok(_) => tracing::warn!(
                    "The model in {} is missing files; fetch or import it again",
                    dir.display()
                ),
                Err(error) => {
                    tracing::warn!("Skipping the model folder {}: {error}", dir.display())
                }
            }
        }
        None
    }

    /// The stored model that holds everything `request` asks for.
    pub fn find_request(&self, request: &ModelRequest) -> Option<StoredModel> {
        self.resolve(&request.main.sha256)
            .filter(|model| model.satisfies(request))
    }

    /// The model in the user's own store with this SHA-256, when its files
    /// are there: what a download adds companions to (see
    /// [`plan_download`]).
    pub fn user_model(&self, sha256: &str) -> Option<StoredModel> {
        Self::read_entry(&self.root.join(sha256.to_ascii_lowercase()), false)
            .ok()
            .filter(StoredModel::files_present)
    }

    /// Models matching `query`: a SHA-256, a unique prefix of one (at least
    /// 6 digits), a name or a file name (case-insensitive).
    pub fn find(&self, query: &str) -> Vec<StoredModel> {
        let query = query.trim();
        let lower = query.to_ascii_lowercase();
        let models = self.list();
        if is_sha256_hex(query) {
            return models
                .into_iter()
                .filter(|m| m.record.sha256 == lower)
                .collect();
        }
        if query.len() >= 6 && query.bytes().all(|b| b.is_ascii_hexdigit()) {
            let by_prefix: Vec<StoredModel> = models
                .iter()
                .filter(|m| m.record.sha256.starts_with(&lower))
                .cloned()
                .collect();
            if !by_prefix.is_empty() {
                return by_prefix;
            }
        }
        models
            .into_iter()
            .filter(|m| {
                m.record
                    .name
                    .as_deref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(query))
                    || m.record.file_name.eq_ignore_ascii_case(query)
            })
            .collect()
    }

    /// Deletes partial downloads nobody resumed for a month.
    fn prune_partials(&self) {
        let Ok(entries) = fs::read_dir(self.partial_dir()) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let stale = path.extension().is_some_and(|ext| ext == "part")
                && entry
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|age| age > PARTIAL_MAX_AGE);
            if stale {
                let _ = fs::remove_file(&path);
            }
        }
    }

    /// Downloads what `plan` lists, verifies it and stores the model.
    /// `progress` is called often (callers throttle what they show);
    /// `cancel` stops the download and deletes what was downloaded.
    pub fn fetch(
        &self,
        client: &ModelClient,
        plan: &DownloadPlan,
        policy: &Policy,
        progress: &dyn Fn(&FetchProgress),
        cancel: &AtomicBool,
    ) -> Result<StoredModel, FetchError> {
        self.prune_partials();
        // Another process may be downloading the same files: wait for it,
        // then use what it stored.
        let mut locks = Vec::new();
        let mut shas: Vec<&str> = plan.files.iter().map(|f| f.sha256.as_str()).collect();
        shas.sort_unstable();
        shas.dedup();
        for sha256 in shas {
            locks.push(lock_file(
                &self.partial_dir().join(format!("{sha256}.lock")),
                cancel,
            )?);
        }
        if let Some(stored) = self.find_request(&plan.request) {
            return Ok(stored);
        }
        let total: Option<u64> = plan.known_size();
        let mut main = None;
        let mut staged = Vec::new();
        let mut done_before = 0_u64;
        let mut resumed = false;
        for (index, file) in plan.files.iter().enumerate() {
            let report = |done: u64, file_total: Option<u64>| {
                progress(&FetchProgress {
                    file_name: file.file_name.clone(),
                    file_index: index,
                    file_count: plan.files.len(),
                    done: done_before + done,
                    total: total.or(if plan.files.len() == 1 {
                        file_total
                    } else {
                        None
                    }),
                })
            };
            // On failure, files downloaded before stay complete in
            // `.partial` and are verified, not downloaded, next time.
            let downloaded =
                fetch::download(client, file, &self.partial_dir(), policy, &report, cancel)?;
            resumed |= downloaded.resumed;
            done_before += downloaded.size;
            let staged_file = StagedFile {
                path: downloaded.path,
                file_name: file.file_name.clone(),
                sha256: file.sha256.clone(),
                size: downloaded.size,
                source: Some(redacted_url(&file.url)),
            };
            if file.main {
                main = Some(staged_file);
            } else {
                staged.push(staged_file);
            }
        }
        let stored = self.assemble(
            &plan.request,
            &plan.main_file_name,
            Some(redacted_url(&plan.main_url)),
            main,
            staged,
        )?;
        drop(locks);
        crate::audit::record(
            "model_fetched",
            serde_json::json!({
                "sha256": stored.record.sha256,
                "name": stored.record.name,
                "version": stored.record.version,
                "file": stored.record.file_name,
                "bytes": stored.total_size(),
                "source": stored.record.source,
                "requested_by": plan.request.requested_by,
                "companions": stored.record.companions.iter().map(|c| serde_json::json!({
                    "file": c.file_name,
                    "sha256": c.sha256,
                    "bytes": c.size,
                })).collect::<Vec<_>>(),
                "resumed": resumed,
            }),
        );
        Ok(stored)
    }

    /// Copies a local file (and its companions) into the store after
    /// hashing it; `expected` is checked when given. For air-gapped sites.
    pub fn import(
        &self,
        file: &Path,
        expected: Option<&str>,
        companions: &[(PathBuf, Option<String>)],
        name: Option<&str>,
        version: Option<&str>,
        cancel: &AtomicBool,
    ) -> Result<StoredModel, FetchError> {
        let partial = self.partial_dir();
        let mut staged = Vec::new();
        for (index, (path, expected)) in
            std::iter::once((file.to_path_buf(), expected.map(str::to_string)))
                .chain(companions.iter().cloned())
                .enumerate()
        {
            let what = if index == 0 { "model" } else { "companion" };
            if let Some(expected) = &expected
                && !is_sha256_hex(expected)
            {
                return Err(FetchError::Invalid(format!(
                    "the {what}'s SHA-256 must be 64 hex digits, not '{expected}'"
                )));
            }
            let file_name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            check_store_file_name(&file_name).map_err(FetchError::Invalid)?;
            if staged
                .iter()
                .any(|s: &StagedFile| s.file_name.eq_ignore_ascii_case(&file_name))
            {
                return Err(FetchError::Invalid(format!(
                    "two files of the model are named '{file_name}'"
                )));
            }
            let staged_file = copy_hashed(&path, &partial, cancel)?;
            if let Some(expected) = expected
                && !staged_file.0.eq_ignore_ascii_case(&expected)
            {
                let _ = fs::remove_file(&staged_file.2);
                return Err(FetchError::Integrity {
                    file: file_name,
                    expected: expected.to_ascii_lowercase(),
                    actual: staged_file.0,
                });
            }
            staged.push(StagedFile {
                path: staged_file.2,
                file_name,
                sha256: staged_file.0,
                size: staged_file.1,
                source: None,
            });
        }
        let main = staged.remove(0);
        let request = ModelRequest {
            main: FileRequest {
                sha256: main.sha256.clone(),
                size: Some(main.size),
                file_name: Some(main.file_name.clone()),
                url: None,
            },
            companions: staged
                .iter()
                .map(|s| FileRequest {
                    sha256: s.sha256.clone(),
                    size: Some(s.size),
                    file_name: Some(s.file_name.clone()),
                    url: None,
                })
                .collect(),
            name: clean_text(name, 200),
            version: clean_text(version, 100),
            requested_by: CLI_REQUESTER.to_string(),
        };
        let main_file_name = main.file_name.clone();
        let stored = self.assemble(&request, &main_file_name, None, Some(main), staged)?;
        crate::audit::record(
            "model_imported",
            serde_json::json!({
                "sha256": stored.record.sha256,
                "name": stored.record.name,
                "version": stored.record.version,
                "file": stored.record.file_name,
                "bytes": stored.total_size(),
                "requested_by": CLI_REQUESTER,
                "companions": stored.record.companions.iter().map(|c| serde_json::json!({
                    "file": c.file_name,
                    "sha256": c.sha256,
                    "bytes": c.size,
                })).collect::<Vec<_>>(),
            }),
        );
        Ok(stored)
    }

    /// Moves verified files into the model's folder: a new folder is built
    /// beside it and renamed into place, and an existing model gains the
    /// companions it lacks (a different file under a name it has is a
    /// conflict).
    fn assemble(
        &self,
        request: &ModelRequest,
        main_file_name: &str,
        source: Option<String>,
        main: Option<StagedFile>,
        companions: Vec<StagedFile>,
    ) -> Result<StoredModel, FetchError> {
        let local = |path: &Path| {
            let path = path.to_path_buf();
            move |error: std::io::Error| FetchError::Local {
                path: path.clone(),
                message: error.to_string(),
            }
        };
        let cleanup = |files: &[&StagedFile]| {
            for file in files {
                let _ = fs::remove_file(&file.path);
            }
        };
        let never = AtomicBool::new(false);
        let _lock = lock_file(&self.root.join(STORE_LOCK), &never)?;
        let target = self.root.join(&request.main.sha256);
        let existing = Self::read_entry(&target, false).ok();
        if let Some(mut model) = existing {
            // Add what is missing to the stored model.
            let mut changed = false;
            if let Some(main) = &main
                && !fs::metadata(model.path()).is_ok_and(|meta| meta.len() == model.record.size)
            {
                fs::rename(&main.path, model.path()).map_err(local(&model.path()))?;
                changed = true;
            }
            for companion in &companions {
                match model
                    .record
                    .companions
                    .iter()
                    .find(|c| c.file_name.eq_ignore_ascii_case(&companion.file_name))
                {
                    Some(have) if have.sha256 == companion.sha256 => {
                        let path = model.dir.join(&have.file_name);
                        if !fs::metadata(&path).is_ok_and(|meta| meta.len() == have.size) {
                            fs::rename(&companion.path, &path).map_err(local(&path))?;
                            changed = true;
                        }
                    }
                    Some(have) => {
                        let all: Vec<&StagedFile> = main.iter().chain(&companions).collect();
                        cleanup(&all);
                        return Err(FetchError::Conflict(tr!(
                            "the stored model {model} has a different {file} (SHA-256 {sha256}); remove it first (eov model remove {id})",
                            model = isolate(&model.record.sha256),
                            file = isolate(&have.file_name),
                            sha256 = isolate(&have.sha256),
                            id = &model.record.sha256[..12]
                        )));
                    }
                    None => {
                        let path = model.dir.join(&companion.file_name);
                        fs::rename(&companion.path, &path).map_err(local(&path))?;
                        model.record.companions.push(CompanionRecord {
                            file_name: companion.file_name.clone(),
                            sha256: companion.sha256.clone(),
                            size: companion.size,
                            source: companion.source.clone(),
                        });
                        changed = true;
                    }
                }
            }
            let leftovers: Vec<&StagedFile> = main.iter().chain(&companions).collect();
            cleanup(&leftovers);
            if changed {
                let record = model.dir.join(RECORD_FILE);
                let text = serde_json::to_vec_pretty(&model.record).map_err(|error| {
                    FetchError::Local {
                        path: record.clone(),
                        message: error.to_string(),
                    }
                })?;
                write_atomic(&record, &text).map_err(local(&record))?;
            }
            return Ok(model);
        }
        let Some(main) = main else {
            let all: Vec<&StagedFile> = companions.iter().collect();
            cleanup(&all);
            return Err(FetchError::Local {
                path: target,
                message: tr!("the model's main file went missing from the store"),
            });
        };
        // A folder without a readable record is a leftover: replace it.
        if target.exists() {
            fs::remove_dir_all(&target).map_err(local(&target))?;
        }
        let staging = tempfile::Builder::new()
            .prefix(".staging-")
            .tempdir_in(&self.root)
            .map_err(local(&self.root))?;
        let record = Record {
            format: RECORD_FORMAT,
            sha256: main.sha256.clone(),
            file_name: main_file_name.to_string(),
            size: main.size,
            name: request.name.clone(),
            version: request.version.clone(),
            source,
            fetched_at: Some(crate::audit::now_rfc3339()),
            requested_by: Some(request.requested_by.clone()),
            companions: companions
                .iter()
                .map(|c| CompanionRecord {
                    file_name: c.file_name.clone(),
                    sha256: c.sha256.clone(),
                    size: c.size,
                    source: c.source.clone(),
                })
                .collect(),
        };
        let main_path = staging.path().join(main_file_name);
        fs::rename(&main.path, &main_path).map_err(local(&main_path))?;
        for file in &companions {
            let path = staging.path().join(&file.file_name);
            fs::rename(&file.path, &path).map_err(local(&path))?;
        }
        let record_path = staging.path().join(RECORD_FILE);
        let text = serde_json::to_vec_pretty(&record).map_err(|error| FetchError::Local {
            path: record_path.clone(),
            message: error.to_string(),
        })?;
        write_atomic(&record_path, &text).map_err(local(&record_path))?;
        let staging = staging.keep();
        if let Err(error) = fs::rename(&staging, &target) {
            let _ = fs::remove_dir_all(&staging);
            return Err(local(&target)(error));
        }
        crate::update::registry::sync_directory(&self.root);
        Ok(StoredModel {
            record,
            dir: target,
            read_only: false,
        })
    }

    /// Hashes every file of every stored model again.
    pub fn verify(&self, cancel: &AtomicBool, progress: &dyn Fn(&str)) -> Vec<VerifyReport> {
        let mut reports = Vec::new();
        for (dir, read_only) in self.entry_dirs() {
            let model = match Self::read_entry(&dir, read_only) {
                Ok(model) => model,
                Err(error) => {
                    reports.push(VerifyReport {
                        dir,
                        model: None,
                        problems: vec![error],
                    });
                    continue;
                }
            };
            progress(&model.label());
            let mut problems = Vec::new();
            for (name, sha256, size) in model.files() {
                if cancel.load(Ordering::Relaxed) {
                    return reports;
                }
                match hash_file(&model.dir.join(name), cancel) {
                    Ok((actual, actual_size)) if actual == sha256 && actual_size == size => {}
                    Ok((actual, actual_size)) => problems.push(format!(
                        "{name}: expected SHA-256 {sha256} ({size} bytes), found {actual} ({actual_size} bytes)"
                    )),
                    Err(error) => problems.push(format!("{name}: {error}")),
                }
            }
            reports.push(VerifyReport {
                dir: dir.clone(),
                model: Some(model),
                problems,
            });
        }
        reports
    }

    /// Removes a model from the user's store.
    pub fn remove(&self, model: &StoredModel) -> Result<(), FetchError> {
        if model.read_only || !model.dir.starts_with(&self.root) {
            return Err(FetchError::Refused(format!(
                "{} is provided by the administrator ({}) and cannot be removed",
                model.label(),
                model.dir.display()
            )));
        }
        let never = AtomicBool::new(false);
        let _lock = lock_file(&self.root.join(STORE_LOCK), &never)?;
        fs::remove_dir_all(&model.dir).map_err(|error| FetchError::Local {
            path: model.dir.clone(),
            message: error.to_string(),
        })?;
        crate::audit::record(
            "model_removed",
            serde_json::json!({
                "sha256": model.record.sha256,
                "name": model.record.name,
                "file": model.record.file_name,
            }),
        );
        Ok(())
    }
}

/// The answer to a plugin's `model.fetch`.
pub(crate) fn fetched_event(
    request_id: String,
    result: &Result<StoredModel, FetchError>,
) -> eov_plugin_api::protocol::HostEvent {
    use eov_plugin_api::protocol::HostEvent;
    match result {
        Ok(model) => HostEvent::ModelFetched {
            request_id,
            path: Some(model.path().to_string_lossy().into_owned()),
            model: Some(model.to_protocol()),
            error: None,
            cancelled: false,
        },
        Err(error) => HostEvent::ModelFetched {
            request_id,
            path: None,
            model: None,
            error: Some(error.to_string()),
            cancelled: error.is_cancelled(),
        },
    }
}

/// The answer to `model.fetch` where nothing is downloaded (batch runs):
/// the stored model, or how to get it.
pub(crate) fn answer_from_store(
    store: &ModelStore,
    request_id: String,
    request: &ModelRequest,
) -> eov_plugin_api::protocol::HostEvent {
    let result = store.find_request(request).ok_or_else(|| {
        FetchError::Invalid(format!(
            "{} is not in eov's model store, and batch runs do not download models: download it in eov's window, or with `eov model fetch <URL> --sha256 {}`, before the run",
            request.label(),
            request.main.sha256
        ))
    });
    fetched_event(request_id, &result)
}

/// Copies `path` into a temporary file in `directory` while hashing it:
/// (SHA-256, size, the copy).
fn copy_hashed(
    path: &Path,
    directory: &Path,
    cancel: &AtomicBool,
) -> Result<(String, u64, PathBuf), FetchError> {
    let local = |path: &Path, error: std::io::Error| FetchError::Local {
        path: path.to_path_buf(),
        message: error.to_string(),
    };
    let mut source = File::open(path).map_err(|error| local(path, error))?;
    if !source
        .metadata()
        .map_err(|error| local(path, error))?
        .is_file()
    {
        return Err(FetchError::Invalid(format!(
            "{} is not a file",
            path.display()
        )));
    }
    fs::create_dir_all(directory).map_err(|error| local(directory, error))?;
    let mut copy = tempfile::Builder::new()
        .prefix("import-")
        .suffix(".part")
        .tempfile_in(directory)
        .map_err(|error| local(directory, error))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut size = 0_u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(FetchError::Cancelled);
        }
        let read = match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(local(path, error)),
        };
        size += read as u64;
        hasher.update(&buffer[..read]);
        copy.write_all(&buffer[..read])
            .map_err(|error| local(copy.path(), error))?;
    }
    copy.as_file()
        .sync_all()
        .map_err(|error| local(copy.path(), error))?;
    let (_, kept) = copy.keep().map_err(|error| local(directory, error.error))?;
    Ok((format!("{:x}", hasher.finalize()), size, kept))
}

#[cfg(test)]
mod tests;
