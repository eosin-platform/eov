//! The HTTP transport behind every updater request.
//!
//! [`Transport`] performs exactly one GET without following redirects; the
//! [`ReleaseClient`](super::network::ReleaseClient) layers consent, redirect
//! validation, size limits and hashing on top of it. Production code uses
//! [`UreqTransport`]; tests inject [`MockTransport`] so installs, planning and
//! failure handling can be exercised without a network.
//!
//! Enterprise networks are supported out of the box:
//!
//! - **Proxies** are taken from `HTTPS_PROXY`/`https_proxy` or
//!   `ALL_PROXY`/`all_proxy` (`http://[user:password@]host:port`), and
//!   `NO_PROXY`/`no_proxy` (comma-separated host suffixes, or `*`) is honoured.
//! - **Certificates**: the operating system's certificate store is trusted in
//!   addition to the bundled Mozilla roots, so TLS-intercepting firewalls whose
//!   root is deployed to the OS work. `SSL_CERT_FILE` / `SSL_CERT_DIR` replace
//!   the OS store, for example to point at an institution's CA bundle.

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

/// One HTTP response, with the body still unread.
pub struct HttpResponse {
    pub status: u16,
    pub location: Option<String>,
    pub content_length: Option<u64>,
    pub body: Box<dyn Read + Send>,
}

impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("location", &self.location)
            .field("content_length", &self.content_length)
            .finish_non_exhaustive()
    }
}

/// Performs single HTTP GET requests. Implementations must not follow
/// redirects: the caller validates every hop.
pub trait Transport: Send + Sync {
    /// `Err` is a transport failure (DNS, TCP, TLS, proxy, timeout); HTTP
    /// error statuses are returned as `Ok` responses.
    fn get(
        &self,
        url: &Url,
        accept: &str,
        headers: &[(&'static str, String)],
    ) -> Result<HttpResponse, String>;
}

/// The production transport: ureq with rustls, the OS certificate store, and
/// proxy settings from the environment.
pub struct UreqTransport {
    direct: ureq::Agent,
    proxied: Option<(ureq::Agent, NoProxy)>,
}

impl UreqTransport {
    pub fn new(user_agent: &str) -> Self {
        let tls = Arc::new(tls_config());
        let builder = || {
            ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(20))
                .timeout_read(Duration::from_secs(60))
                .timeout_write(Duration::from_secs(60))
                .redirects(0)
                .user_agent(user_agent)
                .tls_config(Arc::clone(&tls))
        };
        let proxied =
            ProxySettings::from_env().and_then(|settings| match ureq::Proxy::new(&settings.url) {
                Ok(proxy) => Some((builder().proxy(proxy).build(), settings.no_proxy)),
                Err(error) => {
                    tracing::warn!(
                        "Ignoring invalid proxy setting {}: {error}",
                        settings.redacted()
                    );
                    None
                }
            });
        Self {
            direct: builder().build(),
            proxied,
        }
    }

    fn agent_for(&self, url: &Url) -> &ureq::Agent {
        match &self.proxied {
            Some((agent, no_proxy)) if !no_proxy.matches(url.host_str().unwrap_or_default()) => {
                agent
            }
            _ => &self.direct,
        }
    }
}

impl Transport for UreqTransport {
    fn get(
        &self,
        url: &Url,
        accept: &str,
        headers: &[(&'static str, String)],
    ) -> Result<HttpResponse, String> {
        let mut request = self.agent_for(url).get(url.as_str()).set("Accept", accept);
        for (name, value) in headers {
            request = request.set(name, value);
        }
        let response = match request.call() {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => return Err(error.to_string()),
        };
        Ok(HttpResponse {
            status: response.status(),
            location: response.header("Location").map(str::to_string),
            content_length: response
                .header("Content-Length")
                .and_then(|value| value.trim().parse().ok()),
            body: Box::new(response.into_reader()),
        })
    }
}

/// Mozilla's roots plus everything in the OS certificate store.
fn tls_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let native = rustls_native_certs::load_native_certs();
    for error in &native.errors {
        tracing::debug!("Could not load an OS certificate: {error}");
    }
    let (added, ignored) = roots.add_parsable_certificates(native.certs);
    tracing::debug!("Trusting {added} OS certificate(s) ({ignored} unparsable)");
    rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .expect("ring supports the default TLS versions")
        .with_root_certificates(roots)
        .with_no_client_auth()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProxySettings {
    url: String,
    no_proxy: NoProxy,
}

impl ProxySettings {
    fn from_env() -> Option<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let first = |names: &[&str]| {
            names
                .iter()
                .filter_map(|name| lookup(name))
                .map(|value| value.trim().to_string())
                .find(|value| !value.is_empty())
        };
        // Every updater request is HTTPS, so HTTP_PROXY does not apply (the
        // same rule curl uses).
        let url = first(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"])?;
        let url = if url.contains("://") {
            url
        } else {
            format!("http://{url}")
        };
        let no_proxy = NoProxy::parse(first(&["NO_PROXY", "no_proxy"]).as_deref().unwrap_or(""));
        Some(Self { url, no_proxy })
    }

    /// The proxy URL without credentials, for log messages.
    fn redacted(&self) -> String {
        match Url::parse(&self.url) {
            Ok(mut url) => {
                let _ = url.set_password(None);
                let _ = url.set_username("");
                url.to_string()
            }
            Err(_) => "<unparsable proxy URL>".to_string(),
        }
    }
}

/// Hosts that bypass the proxy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct NoProxy {
    all: bool,
    suffixes: Vec<String>,
}

impl NoProxy {
    fn parse(value: &str) -> Self {
        let mut result = Self::default();
        for entry in value
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            if entry == "*" {
                result.all = true;
                continue;
            }
            // Strip a port and a leading wildcard or dot.
            let host = entry.rsplit_once(':').map_or(entry, |(host, port)| {
                if port.chars().all(|ch| ch.is_ascii_digit()) {
                    host
                } else {
                    entry
                }
            });
            let host = host.trim_start_matches("*.").trim_start_matches('.');
            if !host.is_empty() {
                result.suffixes.push(host.to_ascii_lowercase());
            }
        }
        result
    }

    fn matches(&self, host: &str) -> bool {
        if self.all {
            return true;
        }
        let host = host.to_ascii_lowercase();
        self.suffixes
            .iter()
            .any(|suffix| host == *suffix || host.ends_with(&format!(".{suffix}")))
    }
}

/// A request seen by [`MockTransport`]: the URL and the extra headers.
#[cfg(test)]
pub type RecordedRequest = (String, Vec<(&'static str, String)>);

/// A scripted transport for tests: responses are looked up by exact URL and
/// every request is recorded.
#[cfg(test)]
#[derive(Default)]
pub struct MockTransport {
    routes: std::sync::Mutex<std::collections::HashMap<String, MockRoute>>,
    pub requests: std::sync::Mutex<Vec<RecordedRequest>>,
}

#[cfg(test)]
#[derive(Clone)]
enum MockRoute {
    Body { status: u16, body: Vec<u8> },
    Redirect { status: u16, location: String },
    Failure(String),
}

#[cfg(test)]
impl MockTransport {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn respond(&self, url: &str, status: u16, body: impl Into<Vec<u8>>) {
        self.routes.lock().unwrap().insert(
            url.to_string(),
            MockRoute::Body {
                status,
                body: body.into(),
            },
        );
    }

    pub fn redirect(&self, url: &str, location: &str) {
        self.routes.lock().unwrap().insert(
            url.to_string(),
            MockRoute::Redirect {
                status: 302,
                location: location.to_string(),
            },
        );
    }

    pub fn fail(&self, url: &str, message: &str) {
        self.routes
            .lock()
            .unwrap()
            .insert(url.to_string(), MockRoute::Failure(message.to_string()));
    }

    pub fn requested(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|(url, _)| url.clone())
            .collect()
    }
}

#[cfg(test)]
impl Transport for MockTransport {
    fn get(
        &self,
        url: &Url,
        _accept: &str,
        headers: &[(&'static str, String)],
    ) -> Result<HttpResponse, String> {
        self.requests
            .lock()
            .unwrap()
            .push((url.to_string(), headers.to_vec()));
        let route = self.routes.lock().unwrap().get(url.as_str()).cloned();
        match route {
            Some(MockRoute::Body { status, body }) => Ok(HttpResponse {
                status,
                location: None,
                content_length: Some(body.len() as u64),
                body: Box::new(std::io::Cursor::new(body)),
            }),
            Some(MockRoute::Redirect { status, location }) => Ok(HttpResponse {
                status,
                location: Some(location),
                content_length: Some(0),
                body: Box::new(std::io::empty()),
            }),
            Some(MockRoute::Failure(message)) => Err(message),
            None => Ok(HttpResponse {
                status: 404,
                location: None,
                content_length: Some(0),
                body: Box::new(std::io::empty()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(vars: &[(&str, &str)]) -> Option<ProxySettings> {
        let vars: std::collections::HashMap<_, _> = vars
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        ProxySettings::from_lookup(|name| vars.get(name).cloned())
    }

    #[test]
    fn https_proxy_is_preferred_and_http_proxy_is_ignored() {
        assert_eq!(settings(&[("HTTP_PROXY", "http://plain:3128")]), None);
        let proxy = settings(&[
            ("https_proxy", "proxy.example:8080"),
            ("ALL_PROXY", "http://other:1"),
        ])
        .unwrap();
        assert_eq!(proxy.url, "http://proxy.example:8080");
    }

    #[test]
    fn no_proxy_matches_hosts_and_suffixes() {
        let no_proxy = NoProxy::parse("localhost, .github.com,*.example.org:8443, 10.0.0.1");
        assert!(no_proxy.matches("github.com"));
        assert!(no_proxy.matches("api.github.com"));
        assert!(no_proxy.matches("a.example.org"));
        assert!(!no_proxy.matches("githubusercontent.com"));
        assert!(!no_proxy.matches("notgithub.com"));
        assert!(NoProxy::parse("*").matches("anything"));
    }

    #[test]
    fn proxy_credentials_are_redacted_in_logs() {
        let proxy = settings(&[("HTTPS_PROXY", "http://user:secret@proxy:3128")]).unwrap();
        let redacted = proxy.redacted();
        assert!(
            !redacted.contains("secret") && !redacted.contains("user"),
            "{redacted}"
        );
    }

    #[test]
    fn tls_config_trusts_bundled_roots() {
        // Building the configuration must not panic even without OS roots.
        let config = tls_config();
        assert!(config.alpn_protocols.is_empty());
    }
}
