//! Permission-gated HTTPS access for release metadata and artifacts.
//!
//! Every request needs a [`NetworkPermission`], which can only be obtained
//! from an explicit `--allow-network` flag or an interactive "yes" (and never
//! when the deployment policy disables the network). Redirects are followed
//! manually and only to HTTPS GitHub release hosts, responses are size
//! limited, downloads are streamed to disk while being hashed, and failures
//! are typed so callers can tell "not published" (HTTP 404) from "no network".

use super::policy::Policy;
use super::prompt::Interaction;
use super::transport::{HttpResponse, Transport, UreqTransport};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use url::Url;

/// Largest release manifest, signature or other metadata document accepted.
pub const MAX_METADATA_BYTES: u64 = 1024 * 1024;
/// Largest GitHub API response page accepted.
pub const MAX_API_BYTES: u64 = 8 * 1024 * 1024;
/// Largest application or plugin artifact accepted.
pub const MAX_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MAX_REDIRECTS: usize = 10;

/// A per-process proof that the user authorized release-network access.
///
/// The constructor is private. Callers can only obtain this value through an
/// explicit flag or the interactive consent prompt.
#[derive(Debug, Clone, Copy)]
pub struct NetworkPermission {
    _private: (),
}

impl NetworkPermission {
    /// Grants permission for `--allow-network`, or asks. Returns `None` when
    /// the user declines or nobody can answer.
    pub fn acquire(
        allow_network: bool,
        context: &str,
        interaction: &mut Interaction,
        policy: &Policy,
    ) -> anyhow::Result<Option<Self>> {
        policy.ensure_network_allowed()?;
        if allow_network {
            return Ok(Some(Self { _private: () }));
        }
        let question = format!(
            "This command needs to fetch release metadata from GitHub.\n\nEOV never makes network requests without your permission and does not collect telemetry.\n\n{context}\n\nAllow network access?"
        );
        Ok(interaction
            .consent(&question)
            .then_some(Self { _private: () }))
    }

    /// Like [`acquire`](Self::acquire), but declining is an error.
    pub fn require(
        allow_network: bool,
        context: &str,
        interaction: &mut Interaction,
        policy: &Policy,
    ) -> anyhow::Result<Self> {
        match Self::acquire(allow_network, context, interaction, policy)? {
            Some(permission) => Ok(permission),
            None if !interaction.interactive() => anyhow::bail!(
                "{context}\nNetwork access needs consent, but no terminal is attached. Re-run with --allow-network.{}",
                super::prompt::windows_hint()
            ),
            None => anyhow::bail!(
                "Network access declined; the requested operation cannot continue without release metadata."
            ),
        }
    }

    #[cfg(test)]
    pub fn for_test() -> Self {
        Self { _private: () }
    }
}

/// Why a request failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchError {
    /// The server answered HTTP 404: the release or asset does not exist.
    NotFound { url: String },
    /// Any other unexpected HTTP status.
    Http { url: String, status: u16 },
    /// DNS, TCP, TLS, proxy or timeout failure.
    Network { url: String, message: String },
    /// The response exceeded the size limit for its kind.
    TooLarge { url: String, limit: u64 },
    /// The URL or a redirect target is not an allowed HTTPS release host.
    Rejected { url: String, reason: String },
    /// The downloaded bytes do not match the published SHA-256.
    Integrity {
        url: String,
        expected: String,
        actual: String,
    },
    /// Writing the download to disk failed.
    Local { path: PathBuf, message: String },
}

impl FetchError {
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { url } => write!(f, "{url} does not exist (HTTP 404)"),
            Self::Http { url, status } => write!(f, "GitHub returned HTTP {status} for {url}"),
            Self::Network { url, message } => write!(
                f,
                "could not reach {url}: {message}\nCheck the network connection. Behind a proxy, set HTTPS_PROXY (and NO_PROXY); behind a TLS-inspecting firewall, install its root certificate in the operating system or set SSL_CERT_FILE."
            ),
            Self::TooLarge { url, limit } => {
                write!(f, "{url} is larger than the {limit}-byte limit")
            }
            Self::Rejected { url, reason } => write!(f, "refused to fetch {url}: {reason}"),
            Self::Integrity {
                url,
                expected,
                actual,
            } => write!(
                f,
                "SHA-256 mismatch for {url}: expected {expected}, got {actual}"
            ),
            Self::Local { path, message } => {
                write!(f, "could not write {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for FetchError {}

/// A synchronous HTTPS client. Every request requires a [`NetworkPermission`].
#[derive(Clone)]
pub struct ReleaseClient {
    transport: Arc<dyn Transport>,
    github_token: Option<String>,
}

impl ReleaseClient {
    pub fn new(user_agent: &str) -> Self {
        Self::with_transport(Arc::new(UreqTransport::new(user_agent)))
    }

    /// A client over a custom transport (tests, or a future GUI).
    pub fn with_transport(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            // Only an explicit, eov-specific variable is ever sent; a
            // developer's GITHUB_TOKEN is never forwarded implicitly.
            github_token: std::env::var("EOV_GITHUB_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty()),
        }
    }

    /// The default client for this eov build.
    pub fn for_build() -> Self {
        Self::new(&format!("eov/{}", crate::version::BUILD_VERSION))
    }

    pub fn get_text(
        &self,
        _permission: &NetworkPermission,
        url: &str,
    ) -> Result<String, FetchError> {
        let bytes = self.get_limited(url, "text/plain, application/toml", MAX_METADATA_BYTES)?;
        String::from_utf8(bytes).map_err(|_| FetchError::Rejected {
            url: url.to_string(),
            reason: "the response is not valid UTF-8".into(),
        })
    }

    pub fn get_json(
        &self,
        _permission: &NetworkPermission,
        url: &str,
    ) -> Result<Value, FetchError> {
        let bytes = self.get_limited(url, "application/vnd.github+json", MAX_API_BYTES)?;
        serde_json::from_slice(&bytes).map_err(|error| FetchError::Rejected {
            url: url.to_string(),
            reason: format!("invalid JSON: {error}"),
        })
    }

    /// Streams `url` into a temporary file in `directory`, verifying its
    /// SHA-256 against `expected_sha256` before returning it. Dropping the
    /// returned file deletes it, so failed updates never leave it behind.
    pub fn download_verified(
        &self,
        _permission: &NetworkPermission,
        url: &str,
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<tempfile::NamedTempFile, FetchError> {
        let (final_url, response) = self.open(url, "application/octet-stream")?;
        let limit = MAX_ARTIFACT_BYTES;
        if response.content_length.is_some_and(|length| length > limit) {
            return Err(FetchError::TooLarge {
                url: final_url.to_string(),
                limit,
            });
        }
        let local = |error: std::io::Error| FetchError::Local {
            path: directory.to_path_buf(),
            message: error.to_string(),
        };
        std::fs::create_dir_all(directory).map_err(local)?;
        let mut file = tempfile::Builder::new()
            .prefix(".eov-download-")
            .suffix(".part")
            .tempfile_in(directory)
            .map_err(local)?;
        let sha256 =
            copy_hashed(response.body, file.as_file_mut(), limit).map_err(|error| match error {
                CopyError::TooLarge => FetchError::TooLarge {
                    url: final_url.to_string(),
                    limit,
                },
                CopyError::Read(message) => FetchError::Network {
                    url: final_url.to_string(),
                    message,
                },
                CopyError::Write(message) => FetchError::Local {
                    path: file.path().to_path_buf(),
                    message,
                },
            })?;
        file.as_file().sync_all().map_err(local)?;
        if !sha256.eq_ignore_ascii_case(expected_sha256) {
            return Err(FetchError::Integrity {
                url: url.to_string(),
                expected: expected_sha256.to_ascii_lowercase(),
                actual: sha256,
            });
        }
        Ok(file)
    }

    fn get_limited(&self, url: &str, accept: &str, limit: u64) -> Result<Vec<u8>, FetchError> {
        let (final_url, response) = self.open(url, accept)?;
        if response.content_length.is_some_and(|length| length > limit) {
            return Err(FetchError::TooLarge {
                url: final_url.to_string(),
                limit,
            });
        }
        let mut bytes = Vec::new();
        response
            .body
            .take(limit + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| FetchError::Network {
                url: final_url.to_string(),
                message: error.to_string(),
            })?;
        if bytes.len() as u64 > limit {
            return Err(FetchError::TooLarge {
                url: final_url.to_string(),
                limit,
            });
        }
        Ok(bytes)
    }

    /// Follows redirects (validating every hop) until a 2xx response.
    fn open(&self, url: &str, accept: &str) -> Result<(Url, HttpResponse), FetchError> {
        let mut current = Url::parse(url).map_err(|error| FetchError::Rejected {
            url: url.to_string(),
            reason: format!("invalid URL: {error}"),
        })?;
        for _ in 0..=MAX_REDIRECTS {
            validate_request_url(&current)?;
            let headers = self.headers_for(&current);
            let response = self
                .transport
                .get(&current, accept, &headers)
                .map_err(|message| FetchError::Network {
                    url: current.to_string(),
                    message,
                })?;
            match response.status {
                200..=299 => return Ok((current, response)),
                300..=399 => {
                    let location = response.location.ok_or_else(|| FetchError::Rejected {
                        url: current.to_string(),
                        reason: "redirect without a Location header".into(),
                    })?;
                    current = current
                        .join(&location)
                        .map_err(|error| FetchError::Rejected {
                            url: current.to_string(),
                            reason: format!("invalid redirect target '{location}': {error}"),
                        })?;
                }
                404 | 410 => {
                    return Err(FetchError::NotFound {
                        url: current.to_string(),
                    });
                }
                status => {
                    return Err(FetchError::Http {
                        url: current.to_string(),
                        status,
                    });
                }
            }
        }
        Err(FetchError::Rejected {
            url: url.to_string(),
            reason: "too many redirects".into(),
        })
    }

    fn headers_for(&self, url: &Url) -> Vec<(&'static str, String)> {
        let mut headers = Vec::new();
        if url.host_str() == Some("api.github.com") {
            headers.push(("X-GitHub-Api-Version", "2022-11-28".to_string()));
            if let Some(token) = &self.github_token {
                headers.push(("Authorization", format!("Bearer {token}")));
            }
        }
        headers
    }
}

enum CopyError {
    TooLarge,
    Read(String),
    Write(String),
}

fn copy_hashed(
    mut reader: Box<dyn Read + Send>,
    writer: &mut std::fs::File,
    limit: u64,
) -> Result<String, CopyError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut total = 0_u64;
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CopyError::Read(error.to_string())),
        };
        total += read as u64;
        if total > limit {
            return Err(CopyError::TooLarge);
        }
        hasher.update(&buffer[..read]);
        writer
            .write_all(&buffer[..read])
            .map_err(|error| CopyError::Write(error.to_string()))?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Only HTTPS requests to GitHub release hosts are allowed.
pub fn validate_request_url(url: &Url) -> Result<(), FetchError> {
    if url.scheme() != "https" {
        return Err(FetchError::Rejected {
            url: url.to_string(),
            reason: "updater requests must use HTTPS".into(),
        });
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let allowed = host == "github.com"
        || host == "api.github.com"
        || host == "objects.githubusercontent.com"
        || host == "release-assets.githubusercontent.com"
        || host == "github-production-release-asset.s3.amazonaws.com"
        || host == "github-production-user-asset.s3.amazonaws.com";
    if !allowed {
        return Err(FetchError::Rejected {
            url: url.to_string(),
            reason: format!("{host} is not a GitHub release host"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::prompt::ScriptedPrompter;
    use super::super::transport::MockTransport;
    use super::*;

    fn interaction(interactive: bool, answers: &[Option<&'static str>]) -> Interaction {
        Interaction::new(false, Box::new(ScriptedPrompter::new(interactive, answers)))
    }

    #[test]
    fn allow_network_constructs_permission_without_prompt() {
        let mut interaction = interaction(false, &[]);
        assert!(
            NetworkPermission::acquire(true, "test", &mut interaction, &Policy::default())
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn consent_is_asked_and_eof_declines() {
        let mut yes = interaction(true, &[Some("y")]);
        assert!(
            NetworkPermission::acquire(false, "ctx", &mut yes, &Policy::default())
                .unwrap()
                .is_some()
        );
        let mut eof = interaction(true, &[None]);
        assert!(
            NetworkPermission::acquire(false, "ctx", &mut eof, &Policy::default())
                .unwrap()
                .is_none()
        );
        let mut noninteractive = interaction(false, &[]);
        let error =
            NetworkPermission::require(false, "ctx", &mut noninteractive, &Policy::default())
                .unwrap_err()
                .to_string();
        assert!(error.contains("--allow-network"), "{error}");
    }

    #[test]
    fn policy_can_forbid_the_network_even_with_the_flag() {
        let policy = Policy {
            allow_network: false,
            ..Policy::default()
        };
        let mut interaction = interaction(false, &[]);
        assert!(NetworkPermission::acquire(true, "ctx", &mut interaction, &policy).is_err());
    }

    #[test]
    fn http_urls_are_rejected_before_transport() {
        let transport = MockTransport::new();
        let client = ReleaseClient::with_transport(transport.clone());
        let error = client
            .get_text(&NetworkPermission::for_test(), "http://github.com/eov")
            .unwrap_err();
        assert!(matches!(error, FetchError::Rejected { .. }));
        assert!(transport.requested().is_empty());
    }

    #[test]
    fn redirect_hosts_are_limited_to_github_release_domains() {
        assert!(
            validate_request_url(
                &Url::parse("https://objects.githubusercontent.com/release").unwrap()
            )
            .is_ok()
        );
        assert!(
            validate_request_url(
                &Url::parse("https://github-production-release-asset.s3.amazonaws.com/a").unwrap()
            )
            .is_ok()
        );
        assert!(
            validate_request_url(&Url::parse("https://example.invalid/release").unwrap()).is_err()
        );
        assert!(validate_request_url(&Url::parse("http://github.com/release").unwrap()).is_err());

        let transport = MockTransport::new();
        transport.redirect("https://github.com/a/b", "https://evil.example/x");
        let client = ReleaseClient::with_transport(transport);
        let error = client
            .get_text(&NetworkPermission::for_test(), "https://github.com/a/b")
            .unwrap_err();
        assert!(matches!(error, FetchError::Rejected { .. }), "{error}");
    }

    #[test]
    fn not_found_and_network_failures_are_distinguished() {
        let transport = MockTransport::new();
        transport.fail("https://github.com/down", "dns error: no such host");
        let client = ReleaseClient::with_transport(transport);
        let permission = NetworkPermission::for_test();
        assert!(
            client
                .get_text(&permission, "https://github.com/missing")
                .unwrap_err()
                .is_not_found()
        );
        let error = client
            .get_text(&permission, "https://github.com/down")
            .unwrap_err();
        assert!(matches!(error, FetchError::Network { .. }));
        assert!(error.to_string().contains("HTTPS_PROXY"));
    }

    #[test]
    fn metadata_responses_are_size_limited() {
        let transport = MockTransport::new();
        transport.respond(
            "https://github.com/huge",
            200,
            vec![b'a'; (MAX_METADATA_BYTES + 1) as usize],
        );
        let client = ReleaseClient::with_transport(transport);
        let error = client
            .get_text(&NetworkPermission::for_test(), "https://github.com/huge")
            .unwrap_err();
        assert!(matches!(error, FetchError::TooLarge { .. }));
    }

    #[test]
    fn downloads_are_verified_and_removed_on_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        transport.redirect(
            "https://github.com/o/r/releases/download/v1/a.bin",
            "https://objects.githubusercontent.com/a.bin",
        );
        transport.respond(
            "https://objects.githubusercontent.com/a.bin",
            200,
            b"payload".to_vec(),
        );
        let client = ReleaseClient::with_transport(transport);
        let permission = NetworkPermission::for_test();
        let expected = format!("{:x}", Sha256::digest(b"payload"));
        let download = client
            .download_verified(
                &permission,
                "https://github.com/o/r/releases/download/v1/a.bin",
                &expected,
                temp.path(),
            )
            .unwrap();
        assert_eq!(std::fs::read(download.path()).unwrap(), b"payload");
        drop(download);

        let error = client
            .download_verified(
                &permission,
                "https://github.com/o/r/releases/download/v1/a.bin",
                &"0".repeat(64),
                temp.path(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, FetchError::Integrity { .. }));
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
    }

    #[test]
    fn github_token_is_only_sent_to_the_api_host() {
        let transport = MockTransport::new();
        transport.respond(
            "https://api.github.com/repos/o/r/releases",
            200,
            b"[]".to_vec(),
        );
        transport.respond(
            "https://github.com/o/r/releases/download/v1/x",
            200,
            b"x".to_vec(),
        );
        let mut client = ReleaseClient::with_transport(transport.clone());
        client.github_token = Some("secret".into());
        let permission = NetworkPermission::for_test();
        client
            .get_json(&permission, "https://api.github.com/repos/o/r/releases")
            .unwrap();
        client
            .get_text(&permission, "https://github.com/o/r/releases/download/v1/x")
            .unwrap();
        let requests = transport.requests.lock().unwrap();
        assert!(
            requests[0]
                .1
                .iter()
                .any(|(name, _)| *name == "Authorization")
        );
        assert!(
            !requests[1]
                .1
                .iter()
                .any(|(name, _)| *name == "Authorization")
        );
    }
}
