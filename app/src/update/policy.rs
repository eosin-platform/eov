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
//! ```
//!
//! A file that exists but cannot be read or parsed is treated as the most
//! restrictive policy ("fail closed") and every blocked action reports why.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The effective deployment policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub disable_self_update: bool,
    pub allow_third_party_plugins: bool,
    pub allow_network: bool,
    pub require_signatures: bool,
    pub plugin_dirs: Vec<PathBuf>,
    pub message: Option<String>,
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
            source: Some(source.to_path_buf()),
            unknown_keys: Vec::new(),
            error: Some(error),
        }
    }

    /// Parses policy text.
    pub fn parse(text: &str, source: &Path) -> Result<Self> {
        let raw: RawPolicy = toml::from_str(text)?;
        let defaults = Self::default();
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
            source: Some(source.to_path_buf()),
            unknown_keys: raw.unknown.into_keys().collect(),
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

    #[test]
    fn tests_never_read_the_machine_policy() {
        assert_eq!(policy_path(), None);
        assert!(!current().is_managed());
    }
}
