//! Managed-deployment policy.
//!
//! Administrators can restrict what eov does on a workstation with a
//! system-wide TOML file that users cannot edit:
//!
//! | Platform | Default location |
//! |----------|------------------|
//! | Linux    | `/etc/eov/policy.toml` |
//! | macOS    | `/Library/Application Support/eov/policy.toml` |
//! | Windows  | `%ProgramData%\eov\policy.toml` |
//!
//! `EOV_POLICY=<path>` points at a different file (for testing a policy, or
//! for deployments that manage the environment instead of system files).
//!
//! ```toml
//! # Application self-update is managed by IT (software centre, MDM, ...).
//! disable_self_update = true
//! # Only plugins from the official eov catalog may be installed or loaded.
//! allow_third_party_plugins = false
//! # Never contact GitHub; only offline updates (`eov update --from`) work.
//! allow_network = false
//! # Refuse release metadata that is not signed by a trusted eov key.
//! require_signatures = true
//! # Read-only plugin directories provisioned for every user.
//! plugin_dirs = ["/opt/eov/plugins"]
//! # Shown whenever the policy blocks an action.
//! message = "Contact the service desk (x1234) to request changes."
//! # Models for AI plugins (`eov model fetch`, a plugin's "Download model"):
//! # unset follows allow_network; false allows only `eov model import`.
//! allow_model_downloads = true
//! # Only from these servers (required when allow_network = false).
//! model_url_prefixes = ["https://models.example.org/"]
//! # Read-only model stores provisioned for every user.
//! model_dirs = ["/opt/eov/models"]
//! # The site's copies of model files, by SHA-256: used instead of the URL
//! # a plugin or the command line gives.
//! [model_urls]
//! "70a8deef1f01716efbc87feb181ea398425d08b5e4984ad9d4ab2b83fd354f92" = "https://models.example.org/histovae/HistoVae_reconstruct.onnx"
//! ```
//!
//! A file that exists but cannot be read or parsed is treated as the most
//! restrictive policy ("fail closed") and every blocked action reports why.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use url::Url;

/// The effective deployment policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub disable_self_update: bool,
    pub allow_third_party_plugins: bool,
    pub allow_network: bool,
    pub require_signatures: bool,
    pub plugin_dirs: Vec<PathBuf>,
    pub message: Option<String>,
    /// Model downloads; `None` follows `allow_network`.
    pub allow_model_downloads: Option<bool>,
    /// URL prefixes models may be downloaded from (empty: any).
    pub model_url_prefixes: Vec<String>,
    /// The site's copies of model files: lower-case SHA-256 → URL.
    pub model_urls: BTreeMap<String, String>,
    /// Read-only model stores provisioned for every user.
    pub model_dirs: Vec<PathBuf>,
    /// The file the policy came from, if any.
    pub source: Option<PathBuf>,
    /// Keys the file contained that this eov does not understand.
    pub unknown_keys: Vec<String>,
    /// Why the policy file could not be used (the policy is then restrictive).
    pub error: Option<String>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            disable_self_update: false,
            allow_third_party_plugins: true,
            allow_network: true,
            require_signatures: false,
            plugin_dirs: Vec::new(),
            message: None,
            allow_model_downloads: None,
            model_url_prefixes: Vec::new(),
            model_urls: BTreeMap::new(),
            model_dirs: Vec::new(),
            source: None,
            unknown_keys: Vec::new(),
            error: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawPolicy {
    #[serde(default)]
    disable_self_update: Option<bool>,
    #[serde(default)]
    allow_third_party_plugins: Option<bool>,
    #[serde(default)]
    allow_network: Option<bool>,
    #[serde(default)]
    require_signatures: Option<bool>,
    #[serde(default)]
    plugin_dirs: Option<Vec<PathBuf>>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    allow_model_downloads: Option<bool>,
    #[serde(default)]
    model_url_prefixes: Option<Vec<String>>,
    #[serde(default)]
    model_urls: Option<BTreeMap<String, String>>,
    #[serde(default)]
    model_dirs: Option<Vec<PathBuf>>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

impl Policy {
    /// The policy that applies when the policy file is unusable.
    pub fn restrictive(source: &Path, error: String) -> Self {
        Self {
            disable_self_update: true,
            allow_third_party_plugins: false,
            allow_network: false,
            require_signatures: true,
            plugin_dirs: Vec::new(),
            message: None,
            allow_model_downloads: Some(false),
            model_url_prefixes: Vec::new(),
            model_urls: BTreeMap::new(),
            model_dirs: Vec::new(),
            source: Some(source.to_path_buf()),
            unknown_keys: Vec::new(),
            error: Some(error),
        }
    }

    /// Parses policy text.
    pub fn parse(text: &str, source: &Path) -> Result<Self> {
        let raw: RawPolicy = toml::from_str(text)?;
        let defaults = Self::default();
        let mut unknown_keys: Vec<String> = raw.unknown.into_keys().collect();
        // An entry that is not a SHA-256 can never match; say so.
        let mut model_urls = BTreeMap::new();
        for (sha256, url) in raw.model_urls.unwrap_or_default() {
            if eov_plugin_api::protocol::is_sha256_hex(&sha256) {
                model_urls.insert(sha256.to_ascii_lowercase(), url.trim().to_string());
            } else {
                unknown_keys.push(format!("model_urls.{sha256}"));
            }
        }
        Ok(Self {
            disable_self_update: raw
                .disable_self_update
                .unwrap_or(defaults.disable_self_update),
            allow_third_party_plugins: raw
                .allow_third_party_plugins
                .unwrap_or(defaults.allow_third_party_plugins),
            allow_network: raw.allow_network.unwrap_or(defaults.allow_network),
            require_signatures: raw
                .require_signatures
                .unwrap_or(defaults.require_signatures),
            plugin_dirs: raw.plugin_dirs.unwrap_or_default(),
            message: raw.message.filter(|message| !message.trim().is_empty()),
            allow_model_downloads: raw.allow_model_downloads,
            // Kept as written: one that does not parse matches nothing, so
            // a typo narrows the list instead of opening it.
            model_url_prefixes: raw
                .model_url_prefixes
                .unwrap_or_default()
                .into_iter()
                .map(|prefix| prefix.trim().to_string())
                .collect(),
            model_urls,
            model_dirs: raw.model_dirs.unwrap_or_default(),
            source: Some(source.to_path_buf()),
            unknown_keys,
            error: None,
        })
    }

    /// Loads the policy at `path`; a missing file is the default policy.
    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => match Self::parse(&text, path) {
                Ok(policy) => {
                    if !policy.unknown_keys.is_empty() {
                        tracing::warn!(
                            "Policy file {} contains unknown keys: {}",
                            path.display(),
                            policy.unknown_keys.join(", ")
                        );
                    }
                    policy
                }
                Err(error) => Self::restrictive(path, format!("invalid policy file: {error}")),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(error) => Self::restrictive(path, format!("unreadable policy file: {error}")),
        }
    }

    /// Whether a policy file is in effect.
    pub fn is_managed(&self) -> bool {
        self.source.is_some()
    }

    /// Fails with an explanation when the policy file is unusable.
    pub fn ensure_valid(&self) -> Result<()> {
        if let Some(error) = &self.error {
            bail!(
                "{}: {error}. eov applies the most restrictive settings until it is fixed.{}",
                self.describe_source(),
                self.contact()
            );
        }
        Ok(())
    }

    /// Refuses application self-update when the policy disables it.
    pub fn ensure_self_update_allowed(&self) -> Result<()> {
        self.ensure_valid()?;
        if self.disable_self_update {
            bail!(
                "Application self-update is disabled by {}.{}",
                self.describe_source(),
                self.contact()
            );
        }
        Ok(())
    }

    /// Refuses third-party plugin sources when the policy disallows them.
    pub fn ensure_third_party_allowed(&self, what: &str) -> Result<()> {
        self.ensure_valid()?;
        if !self.allow_third_party_plugins {
            bail!(
                "{what} is not allowed: {} only permits official eov plugins.{}",
                self.describe_source(),
                self.contact()
            );
        }
        Ok(())
    }

    /// Refuses network access when the policy disallows it.
    pub fn ensure_network_allowed(&self) -> Result<()> {
        self.ensure_valid()?;
        if !self.allow_network {
            bail!(
                "Network access is disabled by {}. Use an offline update bundle instead (eov update --from <dir>).{}",
                self.describe_source(),
                self.contact()
            );
        }
        Ok(())
    }

    /// Refuses a model download from `url` when the policy does not allow
    /// it (callers check every redirect too).
    pub fn ensure_model_download_allowed(&self, url: &Url) -> Result<()> {
        self.ensure_valid()?;
        const IMPORT: &str = "Copy the model to this computer and add it with `eov model import <FILE> --sha256 <SHA-256>`.";
        match self.allow_model_downloads {
            Some(false) => bail!(
                "Model downloads are disabled by {}. {IMPORT}{}",
                self.describe_source(),
                self.contact()
            ),
            None if !self.allow_network => bail!(
                "Network access is disabled by {}, so models cannot be downloaded. {IMPORT}{}",
                self.describe_source(),
                self.contact()
            ),
            _ => {}
        }
        if !self.allow_network && self.model_url_prefixes.is_empty() {
            bail!(
                "{} disables network access and names no model servers (model_url_prefixes), so models cannot be downloaded. {IMPORT}{}",
                self.describe_source(),
                self.contact()
            );
        }
        if !self.model_url_prefixes.is_empty()
            && !self
                .model_url_prefixes
                .iter()
                .any(|prefix| url_has_prefix(url, prefix))
        {
            bail!(
                "{} is not on a model server that {} allows ({}).{}",
                redacted_url(url),
                self.describe_source(),
                self.model_url_prefixes.join(", "),
                self.contact()
            );
        }
        Ok(())
    }

    /// Whether one of the policy's model URL prefixes names this
    /// `http://` URL (plain HTTP is otherwise refused for models, except
    /// on this computer).
    pub fn allows_plain_http(&self, url: &Url) -> bool {
        url.scheme() == "http"
            && self
                .model_url_prefixes
                .iter()
                .any(|prefix| url_has_prefix(url, prefix))
    }

    /// The administrator's copy of the model file with this SHA-256.
    pub fn model_url(&self, sha256: &str) -> Option<&str> {
        self.model_urls
            .get(&sha256.to_ascii_lowercase())
            .map(String::as_str)
            .filter(|url| !url.is_empty())
    }

    fn describe_source(&self) -> String {
        match &self.source {
            Some(path) => format!("the administrator policy {}", path.display()),
            None => "the administrator policy".to_string(),
        }
    }

    fn contact(&self) -> String {
        self.message
            .as_deref()
            .map(|message| format!("\n{message}"))
            .unwrap_or_default()
    }
}

/// Whether `url` is below `prefix`: the same scheme, host and port, and a
/// path that continues the prefix's at a `/`. A prefix that does not parse
/// matches nothing.
pub fn url_has_prefix(url: &Url, prefix: &str) -> bool {
    let Ok(prefix) = Url::parse(prefix.trim()) else {
        return false;
    };
    if prefix.host_str().is_none()
        || url.scheme() != prefix.scheme()
        || url.host_str().map(str::to_ascii_lowercase)
            != prefix.host_str().map(str::to_ascii_lowercase)
        || url.port_or_known_default() != prefix.port_or_known_default()
    {
        return false;
    }
    let base = prefix.path();
    if base.ends_with('/') {
        url.path().starts_with(base)
    } else {
        url.path() == base || url.path().starts_with(&format!("{base}/"))
    }
}

/// `url` without credentials, query or fragment, for messages and records
/// (signed URLs carry their secret in the query).
pub fn redacted_url(url: &Url) -> String {
    let mut url = url.clone();
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

/// The policy file location for this platform (or `$EOV_POLICY`).
pub fn policy_path() -> Option<PathBuf> {
    if cfg!(test) {
        // Unit tests never read a policy from the machine running them.
        return None;
    }
    if let Some(path) = std::env::var_os("EOV_POLICY").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(path));
    }
    Some(default_policy_path())
}

fn default_policy_path() -> PathBuf {
    if cfg!(target_os = "windows") {
        std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
            .join("eov")
            .join("policy.toml")
    } else if cfg!(target_os = "macos") {
        PathBuf::from("/Library/Application Support/eov/policy.toml")
    } else {
        PathBuf::from("/etc/eov/policy.toml")
    }
}

/// The policy for this process, loaded once.
///
/// The plugin host should consult [`Policy::plugin_dirs`] (extra read-only
/// plugin directories) and [`Policy::allow_third_party_plugins`] (load only
/// plugins whose registry source is `official`) when it discovers plugins.
pub fn current() -> &'static Policy {
    static POLICY: OnceLock<Policy> = OnceLock::new();
    POLICY.get_or_init(|| match policy_path() {
        Some(path) => Policy::load_from(&path),
        None => Policy::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_the_permissive_default() {
        let temp = tempfile::tempdir().unwrap();
        let policy = Policy::load_from(&temp.path().join("policy.toml"));
        assert_eq!(policy, Policy::default());
        assert!(!policy.is_managed());
        policy.ensure_self_update_allowed().unwrap();
        policy.ensure_third_party_allowed("x").unwrap();
        policy.ensure_network_allowed().unwrap();
    }

    #[test]
    fn parses_every_setting_and_reports_unknown_keys() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("policy.toml");
        std::fs::write(
            &path,
            r#"
disable_self_update = true
allow_third_party_plugins = false
allow_network = false
require_signatures = true
plugin_dirs = ["/opt/eov/plugins"]
message = "Call x1234."
disable_self_updates = true
"#,
        )
        .unwrap();
        let policy = Policy::load_from(&path);
        assert!(policy.disable_self_update && policy.require_signatures);
        assert!(!policy.allow_third_party_plugins && !policy.allow_network);
        assert_eq!(policy.plugin_dirs, vec![PathBuf::from("/opt/eov/plugins")]);
        assert_eq!(
            policy.unknown_keys,
            vec!["disable_self_updates".to_string()]
        );
        let error = policy.ensure_self_update_allowed().unwrap_err().to_string();
        assert!(
            error.contains("disabled") && error.contains("Call x1234."),
            "{error}"
        );
        assert!(
            policy
                .ensure_network_allowed()
                .unwrap_err()
                .to_string()
                .contains("--from")
        );
    }

    #[test]
    fn invalid_file_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("policy.toml");
        std::fs::write(&path, "allow_network = \"maybe\"").unwrap();
        let policy = Policy::load_from(&path);
        assert!(policy.error.is_some());
        assert!(policy.disable_self_update && !policy.allow_network);
        assert!(
            policy
                .ensure_valid()
                .unwrap_err()
                .to_string()
                .contains("invalid policy file")
        );
    }

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn model_downloads_follow_the_network_unless_set() {
        let any = url("https://models.example.org/a/m.onnx");
        Policy::default()
            .ensure_model_download_allowed(&any)
            .unwrap();
        let offline = Policy {
            allow_network: false,
            ..Policy::default()
        };
        let error = offline
            .ensure_model_download_allowed(&any)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Network access is disabled") && error.contains("eov model import"),
            "{error}"
        );
        // An offline site that serves its own models allows only those.
        let site = Policy {
            allow_network: false,
            allow_model_downloads: Some(true),
            model_url_prefixes: vec!["https://models.example.org/a".into()],
            ..Policy::default()
        };
        site.ensure_model_download_allowed(&any).unwrap();
        for refused in [
            "https://models.example.org/ab/m.onnx",
            "https://models.example.org.evil.test/a/m.onnx",
            "http://models.example.org/a/m.onnx",
            "https://models.example.org:8443/a/m.onnx",
        ] {
            assert!(
                site.ensure_model_download_allowed(&url(refused)).is_err(),
                "{refused}"
            );
        }
        let no_servers = Policy {
            model_url_prefixes: Vec::new(),
            ..site.clone()
        };
        assert!(
            no_servers
                .ensure_model_download_allowed(&any)
                .unwrap_err()
                .to_string()
                .contains("model_url_prefixes")
        );
        let disabled = Policy {
            allow_model_downloads: Some(false),
            message: Some("Ask IT.".into()),
            ..Policy::default()
        };
        let error = disabled
            .ensure_model_download_allowed(&any)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Model downloads are disabled") && error.contains("Ask IT."),
            "{error}"
        );
    }

    #[test]
    fn parses_model_keys_and_fails_closed_on_typos() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("policy.toml");
        std::fs::write(
            &path,
            r#"
allow_model_downloads = true
model_url_prefixes = ["https://models.example.org/", "not a url"]
model_dirs = ["/opt/eov/models"]

[model_urls]
"70A8DEEF1F01716EFBC87FEB181EA398425D08B5E4984AD9D4AB2B83FD354F92" = "https://models.example.org/h.onnx"
"histovae" = "https://models.example.org/x"
"#,
        )
        .unwrap();
        let policy = Policy::load_from(&path);
        assert_eq!(policy.error, None);
        assert_eq!(policy.allow_model_downloads, Some(true));
        assert_eq!(policy.model_dirs, vec![PathBuf::from("/opt/eov/models")]);
        assert_eq!(
            policy.model_url("70a8deef1f01716efbc87feb181ea398425d08b5e4984ad9d4ab2b83fd354f92"),
            Some("https://models.example.org/h.onnx")
        );
        assert_eq!(policy.unknown_keys, vec!["model_urls.histovae".to_string()]);
        policy
            .ensure_model_download_allowed(&url("https://models.example.org/h.onnx"))
            .unwrap();
        // The unparsable prefix opens nothing.
        let typo = Policy {
            model_url_prefixes: vec!["not a url".into()],
            ..policy.clone()
        };
        assert!(
            typo.ensure_model_download_allowed(&url("https://elsewhere.test/m"))
                .is_err()
        );
        let http = url("http://models.example.org/m.onnx");
        assert!(!policy.allows_plain_http(&http));
        let intranet = Policy {
            model_url_prefixes: vec!["http://models.example.org/".into()],
            ..Policy::default()
        };
        assert!(intranet.allows_plain_http(&http));

        std::fs::write(&path, "allow_model_downloads = \"yes\"").unwrap();
        let broken = Policy::load_from(&path);
        assert!(
            broken
                .ensure_model_download_allowed(&url("https://models.example.org/m"))
                .is_err()
        );
        assert_eq!(broken.allow_model_downloads, Some(false));
    }

    #[test]
    fn urls_are_redacted_for_messages() {
        assert_eq!(
            redacted_url(&url(
                "https://user:secret@s3.example.org/m.onnx?X-Amz-Signature=abc#x"
            )),
            "https://s3.example.org/m.onnx"
        );
    }

    #[test]
    fn tests_never_read_the_machine_policy() {
        assert_eq!(policy_path(), None);
        assert!(!current().is_managed());
    }
}
