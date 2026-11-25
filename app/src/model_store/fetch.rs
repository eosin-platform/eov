//! Downloading one model file: redirects checked hop by hop, `Range`
//! resumption, streaming SHA-256, size limits and cancellation.

use super::{MAX_MODEL_BYTES, PlannedFile, check_url};
use crate::update::policy::{Policy, redacted_url};
use crate::update::transport::{HttpResponse, Transport, UreqTransport};
use eov_plugin_api::i18n::{isolate, tr};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use url::Url;

const MAX_REDIRECTS: usize = 10;

/// Why a model could not be fetched or imported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FetchError {
    /// The request, a URL or a file is not usable.
    Invalid(String),
    /// Nothing says where to download it from.
    NoUrl { what: String },
    /// The administrator's policy or eov's rules refuse it.
    Refused(String),
    /// The server answered with an error.
    Http { url: String, status: u16 },
    /// DNS, TCP, TLS, proxy, timeout, or a connection that broke off (the
    /// download resumes from where it stopped next time).
    Network { url: String, message: String },
    /// Larger than its expected size or the limit.
    TooLarge { file: String, limit: u64 },
    /// A complete download of another size than expected.
    Size {
        file: String,
        expected: u64,
        actual: u64,
    },
    /// The file's SHA-256 is not the expected one.
    Integrity {
        file: String,
        expected: String,
        actual: String,
    },
    /// Reading or writing on this computer failed.
    Local { path: PathBuf, message: String },
    /// The store has a different file under the same name.
    Conflict(String),
    /// The user cancelled it.
    Cancelled,
}

impl FetchError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// In the language of the interface: eov's window shows it, and plugins
/// get it with `ModelFetched`.
impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Invalid(message) | Self::Refused(message) | Self::Conflict(message) => {
                return f.write_str(message);
            }
            // Translators: model_urls, policy.toml and `eov model import`
            // are names: keep them as they are.
            Self::NoUrl { what } => tr!(
                "no download address is set for {}: give its URL, ask your administrator to add a download address for it to model_urls in policy.toml, or import the file with `eov model import`",
                isolate(what)
            ),
            Self::Http { url, status } => tr!(
                "{url} answered HTTP {status}",
                url = isolate(url),
                status = status
            ),
            // Translators: HTTPS_PROXY, NO_PROXY and SSL_CERT_FILE are
            // names of settings: keep them as they are.
            Self::Network { url, message } => tr!(
                "could not download {url}: {message}\nCheck the network connection; the download continues where it stopped when tried again. Behind a proxy, set HTTPS_PROXY (and NO_PROXY); behind a TLS-inspecting firewall, install its root certificate in the operating system or set SSL_CERT_FILE.",
                url = isolate(url),
                message = message
            ),
            Self::TooLarge { file, limit } => tr!(
                "{file} is larger than expected ({n} byte); it was deleted"
                    | "{file} is larger than expected ({n} bytes); it was deleted" % *limit,
                file = isolate(file)
            ),
            Self::Size {
                file,
                expected,
                actual,
            } => tr!(
                "{file} has the wrong size: expected {n} byte, got {actual}; it was deleted"
                    | "{file} has the wrong size: expected {n} bytes, got {actual}; it was deleted"
                        % *expected,
                file = isolate(file),
                actual = actual
            ),
            Self::Integrity {
                file,
                expected,
                actual,
            } => tr!(
                "{file} does not match its SHA-256: expected {expected}, got {actual}; it was deleted",
                file = isolate(file),
                expected = expected,
                actual = actual
            ),
            Self::Local { path, message } => return write!(f, "{}: {message}", path.display()),
            Self::Cancelled => tr!("the download was cancelled"),
        };
        f.write_str(&text)
    }
}

impl std::error::Error for FetchError {}

/// How far a download is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FetchProgress {
    /// The file being downloaded.
    pub file_name: String,
    pub file_index: usize,
    pub file_count: usize,
    /// Bytes of the model downloaded (or found complete) so far.
    pub done: u64,
    /// Bytes of the model, when known.
    pub total: Option<u64>,
}

impl FetchProgress {
    pub fn fraction(&self) -> Option<f32> {
        self.total
            .filter(|total| *total > 0)
            .map(|total| (self.done as f64 / total as f64).clamp(0.0, 1.0) as f32)
    }
}

/// Downloads model files over the updater's transport (proxies, the
/// operating system's certificates). Unlike the updater's client it is not
/// limited to GitHub: every hop is checked with [`check_url`] instead.
#[derive(Clone)]
pub(crate) struct ModelClient {
    transport: Arc<dyn Transport>,
}

impl ModelClient {
    pub fn with_transport(transport: Arc<dyn Transport>) -> Self {
        Self { transport }
    }

    /// The client of this eov build.
    pub fn for_build() -> Self {
        Self::with_transport(Arc::new(UreqTransport::new(&format!(
            "eov/{}",
            crate::version::BUILD_VERSION
        ))))
    }

    /// GETs `url` from byte `offset` on, following redirects. A server that
    /// cannot serve the range answers `Ok(None)`.
    fn open(
        &self,
        url: &Url,
        offset: u64,
        policy: &Policy,
    ) -> Result<Option<(Url, HttpResponse)>, FetchError> {
        let mut current = url.clone();
        for _ in 0..=MAX_REDIRECTS {
            check_url(&current, policy)?;
            let mut headers = Vec::new();
            if offset > 0 {
                headers.push(("Range", format!("bytes={offset}-")));
            }
            let response = self
                .transport
                .get(&current, "application/octet-stream", &headers)
                .map_err(|message| FetchError::Network {
                    url: redacted_url(&current),
                    message,
                })?;
            match response.status {
                200..=299 => return Ok(Some((current, response))),
                301 | 302 | 303 | 307 | 308 => {
                    let location = response.location.ok_or_else(|| {
                        FetchError::Refused(tr!(
                            "{} redirects without saying where",
                            isolate(redacted_url(&current))
                        ))
                    })?;
                    current = current.join(&location).map_err(|error| {
                        FetchError::Refused(tr!(
                            "{url} redirects to '{location}', which is not a URL: {error}",
                            url = isolate(redacted_url(&current)),
                            location = isolate(&location),
                            error = error
                        ))
                    })?;
                }
                416 if offset > 0 => return Ok(None),
                status => {
                    return Err(FetchError::Http {
                        url: redacted_url(&current),
                        status,
                    });
                }
            }
        }
        Err(FetchError::Refused(tr!(
            "{url} redirects more than {n} time"
                | "{url} redirects more than {n} times" % MAX_REDIRECTS,
            url = isolate(redacted_url(url))
        )))
    }
}

/// A downloaded, verified file in the partial folder.
pub(super) struct Downloaded {
    pub path: PathBuf,
    pub size: u64,
    /// Continued from an earlier, interrupted download.
    pub resumed: bool,
}

/// The first byte of a `Content-Range: bytes START-END/TOTAL` and the
/// total, if given.
fn parse_content_range(value: &str) -> Option<(u64, Option<u64>)> {
    let range = value.trim().strip_prefix("bytes")?.trim_start();
    let (span, total) = range.split_once('/')?;
    let (start, _) = span.split_once('-')?;
    let total = match total.trim() {
        "*" => None,
        total => Some(total.parse().ok()?),
    };
    Some((start.trim().parse().ok()?, total))
}

/// Hashes the first `len` bytes of `file`.
fn hash_prefix(file: &mut File, len: u64, cancel: &AtomicBool) -> std::io::Result<Sha256> {
    let mut hasher = Sha256::new();
    file.seek(SeekFrom::Start(0))?;
    let mut remaining = len;
    let mut buffer = vec![0_u8; 1 << 20];
    while remaining > 0 {
        if cancel.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "cancelled",
            ));
        }
        let want = remaining.min(buffer.len() as u64) as usize;
        let read = file.read(&mut buffer[..want])?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the partial download is shorter than it was",
            ));
        }
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
    }
    Ok(hasher)
}

/// Downloads `file` into `<partial>/<sha256>.part` (continuing a partial
/// one) and verifies it. A cancelled download, and one that does not
/// match, is deleted; an interrupted one is kept to resume.
pub(super) fn download(
    client: &ModelClient,
    file: &PlannedFile,
    partial: &Path,
    policy: &Policy,
    progress: &dyn Fn(u64, Option<u64>),
    cancel: &AtomicBool,
) -> Result<Downloaded, FetchError> {
    let path = partial.join(format!("{}.part", file.sha256));
    let local = |error: std::io::Error| FetchError::Local {
        path: path.clone(),
        message: error.to_string(),
    };
    let limit = file.size.unwrap_or(MAX_MODEL_BYTES);
    let discard = |error: FetchError| {
        let _ = fs::remove_file(&path);
        error
    };
    fs::create_dir_all(partial).map_err(local)?;
    let mut part = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(local)?;
    let mut offset = part.metadata().map_err(local)?.len();
    if offset > limit {
        offset = 0;
    }
    let resumed = offset > 0;
    // Hash what an earlier download left; a complete one needs no network.
    let mut hasher = match hash_prefix(&mut part, offset, cancel) {
        Ok(hasher) => hasher,
        Err(_) if cancel.load(Ordering::Relaxed) => return Err(discard(FetchError::Cancelled)),
        Err(_) => {
            offset = 0;
            Sha256::new()
        }
    };
    if offset > 0 && Some(offset) == file.size {
        let actual = format!("{:x}", hasher.clone().finalize());
        if actual == file.sha256 {
            progress(offset, Some(offset));
            drop(part);
            return Ok(Downloaded {
                path,
                size: offset,
                resumed: true,
            });
        }
        offset = 0;
        hasher = Sha256::new();
    }
    // Ask for the rest; at most one more request, for the whole file, when
    // the server cannot continue (the loop ends once `offset` is 0).
    let (final_url, response, declared) = loop {
        let Some((final_url, response)) = client.open(&file.url, offset, policy)? else {
            // 416: nothing follows the partial download. It may be whole.
            if format!("{:x}", hasher.clone().finalize()) == file.sha256 {
                progress(offset, Some(offset));
                drop(part);
                return Ok(Downloaded {
                    path,
                    size: offset,
                    resumed: true,
                });
            }
            offset = 0;
            hasher = Sha256::new();
            continue;
        };
        if offset > 0 && response.status == 206 {
            match response
                .content_range
                .as_deref()
                .and_then(parse_content_range)
            {
                Some((start, total)) if start == offset => {
                    // The total size the server declares, if any.
                    let declared = total.or(response.content_length.map(|len| offset + len));
                    break (final_url, response, declared);
                }
                _ => {
                    offset = 0;
                    hasher = Sha256::new();
                    continue;
                }
            }
        }
        // A whole file (from the start, or the server ignored the range).
        if offset > 0 {
            offset = 0;
            hasher = Sha256::new();
        }
        let declared = response.content_length;
        break (final_url, response, declared);
    };
    let shown = redacted_url(&final_url);
    if let Some(declared) = declared
        && declared > limit
    {
        let _ = fs::remove_file(&path);
        return Err(match file.size {
            Some(expected) => FetchError::Size {
                file: file.file_name.clone(),
                expected,
                actual: declared,
            },
            None => FetchError::TooLarge {
                file: file.file_name.clone(),
                limit,
            },
        });
    }
    part.set_len(offset).map_err(local)?;
    part.seek(SeekFrom::Start(offset)).map_err(local)?;
    let total = file.size.or(declared);
    progress(offset, total);
    let mut body = response.body;
    let mut buffer = vec![0_u8; 1 << 20];
    let mut size = offset;
    let mut last_report = std::time::Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(part);
            return Err(discard(FetchError::Cancelled));
        }
        let read = match body.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                // Keep what arrived: the next try resumes from here.
                let _ = part.sync_all();
                return Err(FetchError::Network {
                    url: shown,
                    message: error.to_string(),
                });
            }
        };
        size += read as u64;
        if size > limit {
            drop(part);
            return Err(discard(match file.size {
                Some(expected) => FetchError::Size {
                    file: file.file_name.clone(),
                    expected,
                    actual: size,
                },
                None => FetchError::TooLarge {
                    file: file.file_name.clone(),
                    limit,
                },
            }));
        }
        hasher.update(&buffer[..read]);
        part.write_all(&buffer[..read]).map_err(local)?;
        if last_report.elapsed() >= std::time::Duration::from_millis(50) {
            progress(size, total);
            last_report = std::time::Instant::now();
        }
    }
    part.sync_all().map_err(local)?;
    drop(part);
    progress(size, total);
    if let Some(declared) = declared
        && size < declared
    {
        return Err(FetchError::Network {
            url: shown,
            message: tr!(
                "the connection closed after {size} of {n} byte"
                    | "the connection closed after {size} of {n} bytes" % declared,
                size = size
            ),
        });
    }
    if let Some(expected) = file.size
        && size != expected
    {
        return Err(discard(FetchError::Size {
            file: file.file_name.clone(),
            expected,
            actual: size,
        }));
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != file.sha256 {
        return Err(discard(FetchError::Integrity {
            file: file.file_name.clone(),
            expected: file.sha256.clone(),
            actual,
        }));
    }
    Ok(Downloaded {
        path,
        size,
        resumed: resumed && offset > 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_ranges_are_parsed() {
        assert_eq!(
            parse_content_range("bytes 100-199/200"),
            Some((100, Some(200)))
        );
        assert_eq!(parse_content_range("bytes 5-9/*"), Some((5, None)));
        assert_eq!(parse_content_range("items 1-2/3"), None);
        assert_eq!(parse_content_range("bytes x-9/10"), None);
    }

    #[test]
    fn progress_fractions_are_bounded() {
        let progress = FetchProgress {
            file_name: "m.onnx".into(),
            file_index: 0,
            file_count: 1,
            done: 50,
            total: Some(200),
        };
        assert_eq!(progress.fraction(), Some(0.25));
        assert_eq!(
            FetchProgress {
                total: None,
                ..progress
            }
            .fraction(),
            None
        );
    }
}
