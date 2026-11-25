//! Plugin compatibility rules shared by installation, updates and listing.
//!
//! A plugin manifest declares which eov versions it supports in
//! `[environment] version` and, optionally, which plugin API it was compiled
//! against in `[environment] api`:
//!
//! ```toml
//! [environment]
//! version = ">=1.0.0, <2.0.0"   # eov application versions
//! api = ">=1.0, <2"            # eov-plugin-api versions (optional)
//! ```
//!
//! The rules are deliberately stricter than plain SemVer matching:
//!
//! - **Ranges are bounded.** An open-ended requirement such as `>=0.4.1`
//!   cannot know about future ABI breaks, so it is capped at the next
//!   breaking release of its lower bound (`>=0.4.1` means `>=0.4.1, <0.5.0`,
//!   `>=1.2` means `>=1.2, <2.0.0`), exactly like a Cargo caret requirement.
//!   A requirement without any lower bound (`*`) is rejected.
//! - **Prerelease hosts** (`1.1.0-rc.1`) are evaluated as their release
//!   version (`1.1.0`) when the requirement does not mention prereleases, so a
//!   release candidate is not cut off from every plugin.
//! - **`api`**, when present, must match the plugin API this host implements.
//!   The plugin loader performs the authoritative ABI check at load time; this
//!   check only lets management commands refuse obviously incompatible
//!   packages before they are installed.

use anyhow::{Result, bail};
use eov_plugin_api::i18n::{isolate, tr};
use semver::{Comparator, Op, Prerelease, Version, VersionReq};

/// The plugin API version implemented by this host.
pub const HOST_PLUGIN_API_VERSION: &str = "1.1.0";

/// The eov version of this build.
pub fn host_version() -> Version {
    Version::parse(crate::version::BUILD_VERSION).expect("CARGO_PKG_VERSION is valid SemVer")
}

/// The plugin API version of this build.
pub fn host_api_version() -> Version {
    Version::parse(HOST_PLUGIN_API_VERSION).expect("valid plugin API version")
}

/// Why a version requirement of a manifest cannot be used. `Display` says it
/// in English (logs, the audit log, the command line);
/// [`RequirementError::localized`] in the interface's language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementError {
    /// Not a SemVer requirement: the requirement, and the version
    /// library's words.
    Invalid { requirement: String, error: String },
    /// A requirement without a lower bound (`*`, `<2`).
    NoLowerBound(String),
}

impl std::fmt::Display for RequirementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequirementError::Invalid { requirement, error } => {
                write!(
                    f,
                    "'{requirement}' is not a valid SemVer requirement: {error}"
                )
            }
            RequirementError::NoLowerBound(requirement) => write!(
                f,
                "'{requirement}' has no lower bound; declare a range such as \">=1.0.0, <2.0.0\""
            ),
        }
    }
}

impl std::error::Error for RequirementError {}

impl RequirementError {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        match self {
            RequirementError::Invalid { requirement, error } => {
                // Translators: {requirement} is a range of versions such as ">=1.0.0, <2.0.0"; {error} the error of the version library.
                tr!(
                    "'{requirement}' is not a valid SemVer requirement: {error}",
                    requirement = isolate(requirement),
                    error = error
                )
            }
            RequirementError::NoLowerBound(requirement) => {
                // Translators: {} is a range of versions; keep ">=1.0.0, <2.0.0" as it is.
                tr!(
                    "'{}' has no lower bound; declare a range such as \">=1.0.0, <2.0.0\"",
                    isolate(requirement)
                )
            }
        }
    }
}

/// Parses a requirement and adds the implicit upper bound when it has none.
pub fn bounded_requirement(text: &str) -> Result<VersionReq, RequirementError> {
    let mut requirement = VersionReq::parse(text).map_err(|error| RequirementError::Invalid {
        requirement: text.to_string(),
        error: error.to_string(),
    })?;
    if requirement.comparators.iter().any(is_upper_bound) {
        return Ok(requirement);
    }
    let Some(lower) = requirement
        .comparators
        .iter()
        .filter(|comparator| matches!(comparator.op, Op::Greater | Op::GreaterEq))
        .map(|comparator| {
            (
                comparator.major,
                comparator.minor.unwrap_or(0),
                comparator.patch.unwrap_or(0),
            )
        })
        .max()
    else {
        return Err(RequirementError::NoLowerBound(text.to_string()));
    };
    let (major, minor, patch) = lower;
    let cap = if major > 0 {
        (major + 1, 0, 0)
    } else if minor > 0 {
        (0, minor + 1, 0)
    } else {
        (0, 0, patch + 1)
    };
    requirement.comparators.push(Comparator {
        op: Op::Less,
        major: cap.0,
        minor: Some(cap.1),
        patch: Some(cap.2),
        pre: Prerelease::EMPTY,
    });
    Ok(requirement)
}

fn is_upper_bound(comparator: &Comparator) -> bool {
    matches!(
        comparator.op,
        Op::Exact | Op::Less | Op::LessEq | Op::Tilde | Op::Caret | Op::Wildcard
    )
}

/// Whether `host` satisfies the bounded form of `requirement`.
pub fn version_matches(requirement: &VersionReq, host: &Version) -> bool {
    if requirement.matches(host) {
        return true;
    }
    let mentions_prerelease = requirement
        .comparators
        .iter()
        .any(|comparator| !comparator.pre.is_empty());
    !host.pre.is_empty()
        && !mentions_prerelease
        && requirement.matches(&Version::new(host.major, host.minor, host.patch))
}

/// Evaluates an `[environment] version` requirement against an eov version.
pub fn environment_matches(text: &str, host: &Version) -> bool {
    bounded_requirement(text).is_ok_and(|requirement| version_matches(&requirement, host))
}

/// Reads the optional plugin-API requirement from raw manifest text: the
/// top-level `api` key of plugin API 1.0 manifests, or `[environment] api`.
pub fn api_requirement(raw_manifest: &str) -> Option<String> {
    let value: toml::Value = toml::from_str(raw_manifest).ok()?;
    value
        .get("api")
        .or_else(|| value.get("environment")?.get("api"))?
        .as_str()
        .map(str::to_string)
}

/// The outcome of a compatibility check. The reason is in the interface's
/// language ([`Incompatibility::localized`]); [`incompatibility`] gives it
/// as data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    Compatible,
    Incompatible(String),
}

impl Compatibility {
    pub fn is_compatible(&self) -> bool {
        matches!(self, Self::Compatible)
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Compatible => None,
            Self::Incompatible(reason) => Some(reason),
        }
    }
}

/// Why a plugin does not run with this eov, as data. `Display` says it in
/// English (logs, the audit log); [`Incompatibility::localized`] in the
/// interface's language, when it is shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Incompatibility {
    /// The manifest's range of eov versions leaves this eov out; `bounded`
    /// is the range eov takes an open-ended one for.
    Eov {
        required: String,
        bounded: Option<String>,
        version: String,
    },
    /// The manifest's range of eov versions cannot be used.
    InvalidEov(RequirementError),
    /// Built for a plugin API this eov does not provide.
    Api { required: String, provided: String },
    /// The manifest's range of plugin API versions cannot be used.
    InvalidApi(RequirementError),
}

impl std::fmt::Display for Incompatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Incompatibility::Eov {
                required,
                bounded: None,
                version,
            } => write!(f, "requires eov {required}; this is eov {version}"),
            Incompatibility::Eov {
                required,
                bounded: Some(bounded),
                version,
            } => write!(
                f,
                "requires eov {required} (bounded to {bounded}); this is eov {version}"
            ),
            Incompatibility::InvalidEov(error) => write!(f, "invalid eov requirement: {error}"),
            Incompatibility::Api { required, provided } => write!(
                f,
                "built for plugin API {required}; this eov provides plugin API {provided}"
            ),
            Incompatibility::InvalidApi(error) => {
                write!(f, "invalid plugin API requirement: {error}")
            }
        }
    }
}

impl Incompatibility {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        match self {
            Incompatibility::Eov {
                required,
                bounded: None,
                version,
            } => {
                // Translators: why a plugin does not run; {required} is a range of eov versions such as ">=1.0.0, <2.0.0".
                tr!(
                    "requires eov {required}; this is eov {version}",
                    required = isolate(required),
                    version = isolate(version)
                )
            }
            Incompatibility::Eov {
                required,
                bounded: Some(bounded),
                version,
            } => {
                // Translators: why a plugin does not run; {required} is a range of eov versions such as ">=1.1", {bounded} the range eov takes it for (">=1.1, <2.0.0").
                tr!(
                    "requires eov {required} (bounded to {bounded}); this is eov {version}",
                    required = isolate(required),
                    bounded = isolate(bounded),
                    version = isolate(version)
                )
            }
            Incompatibility::InvalidEov(error) => {
                tr!("invalid eov requirement: {}", error.localized())
            }
            Incompatibility::Api { required, provided } => {
                // Translators: why a plugin does not run; {required} is a range of versions of eov's plugin API, {provided} a version.
                tr!(
                    "built for plugin API {required}; this eov provides plugin API {provided}",
                    required = isolate(required),
                    provided = isolate(provided)
                )
            }
            Incompatibility::InvalidApi(error) => {
                tr!("invalid plugin API requirement: {}", error.localized())
            }
        }
    }
}

/// Why a plugin with the declared requirements does not run with an eov
/// version and plugin API version, if it does not.
pub fn incompatibility(
    environment: &str,
    api: Option<&str>,
    eov: &Version,
    plugin_api: &Version,
) -> Option<Incompatibility> {
    match bounded_requirement(environment) {
        Ok(requirement) if version_matches(&requirement, eov) => {}
        Ok(requirement) => {
            let bounded = requirement.to_string();
            return Some(Incompatibility::Eov {
                required: environment.to_string(),
                bounded: (bounded != environment).then_some(bounded),
                version: eov.to_string(),
            });
        }
        Err(error) => return Some(Incompatibility::InvalidEov(error)),
    }
    if let Some(api) = api {
        match bounded_requirement(api) {
            Ok(requirement) if version_matches(&requirement, plugin_api) => {}
            Ok(_) => {
                return Some(Incompatibility::Api {
                    required: api.to_string(),
                    provided: plugin_api.to_string(),
                });
            }
            Err(error) => return Some(Incompatibility::InvalidApi(error)),
        }
    }
    None
}

/// Checks a plugin's declared requirements against an eov version and
/// plugin API version (the reason in the interface's language).
pub fn check(
    environment: &str,
    api: Option<&str>,
    eov: &Version,
    plugin_api: &Version,
) -> Compatibility {
    match incompatibility(environment, api, eov, plugin_api) {
        None => Compatibility::Compatible,
        Some(reason) => Compatibility::Incompatible(reason.localized()),
    }
}

/// Why a package with this `[environment] version` and raw manifest does
/// not run with this host's plugin API and the eov version `eov`, if it
/// does not ([`check_manifest`], as data).
pub fn package_incompatibility(
    environment: &str,
    raw_manifest: &str,
    eov: &Version,
) -> Option<Incompatibility> {
    incompatibility(
        environment,
        api_requirement(raw_manifest).as_deref(),
        eov,
        &host_api_version(),
    )
}

/// Checks an inspected manifest against this host.
pub fn check_manifest(
    manifest: &eov_plugin_api::PluginManifest,
    raw_manifest: &str,
    eov: &Version,
) -> Compatibility {
    match package_incompatibility(&manifest.environment.version, raw_manifest, eov) {
        None => Compatibility::Compatible,
        Some(reason) => Compatibility::Incompatible(reason.localized()),
    }
}

/// Validates a plugin id that eov will use as a file name and library name:
/// `^[a-z0-9][a-z0-9_]{0,63}$`, which also rules out Windows drive prefixes,
/// NTFS alternate data streams (`a:b`) and reserved device names.
pub fn validate_plugin_id(id: &str) -> Result<()> {
    let mut chars = id.chars();
    let valid = match chars.next() {
        Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit() => {
            id.len() <= 64
                && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        }
        _ => false,
    };
    if !valid {
        bail!(
            "{}",
            tr!(
                "plugin ID '{}' is not valid: IDs use 1-64 lowercase letters, digits and '_', starting with a letter or digit",
                isolate(id)
            )
        );
    }
    const RESERVED: &[&str] = &[
        "con", "prn", "aux", "nul", "com0", "com1", "com2", "com3", "com4", "com5", "com6", "com7",
        "com8", "com9", "lpt0", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8",
        "lpt9",
    ];
    if RESERVED.contains(&id) {
        bail!(
            "{}",
            tr!(
                "plugin ID '{}' is a reserved Windows device name",
                isolate(id)
            )
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Version {
        Version::parse(text).unwrap()
    }

    #[test]
    fn host_api_version_tracks_the_plugin_api_crate() {
        let manifest: toml::Value =
            toml::from_str(include_str!("../../../plugin_api/Cargo.toml")).unwrap();
        assert_eq!(
            manifest["package"]["version"].as_str().unwrap(),
            HOST_PLUGIN_API_VERSION,
            "update HOST_PLUGIN_API_VERSION when eov-plugin-api's version changes"
        );
    }

    #[test]
    fn open_ended_requirements_are_capped_at_the_next_breaking_release() {
        assert!(environment_matches(">=0.4.1", &v("0.4.9")));
        assert!(!environment_matches(">=0.4.1", &v("0.5.0")));
        assert!(!environment_matches(">=0.4.1", &v("1.0.0")));
        assert!(environment_matches(">=1.0.0", &v("1.9.3")));
        assert!(!environment_matches(">=1.0.0", &v("2.0.0")));
        assert!(!environment_matches(">0.0.3", &v("0.0.5")));
        assert!(!environment_matches("*", &v("1.0.0")));
        assert!(!environment_matches("not semver", &v("1.0.0")));
    }

    #[test]
    fn explicit_bounds_are_kept() {
        assert!(environment_matches(">=1.0.0, <3.0.0", &v("2.5.0")));
        assert!(environment_matches("^1.2", &v("1.9.0")));
        assert!(environment_matches("1.*", &v("1.4.0")));
        assert!(!environment_matches("~1.2", &v("1.3.0")));
        assert_eq!(
            bounded_requirement(">=1.0.0, <2.0.0").unwrap().to_string(),
            ">=1.0.0, <2.0.0"
        );
    }

    #[test]
    fn prerelease_hosts_are_evaluated_as_their_release() {
        assert!(environment_matches(">=1.0.0, <2.0.0", &v("1.1.0-rc.1")));
        assert!(environment_matches(">=1.1.0, <2.0.0", &v("1.1.0-rc.1")));
        assert!(!environment_matches(">=1.0.0, <1.1.0", &v("1.1.0-rc.1")));
        assert!(!environment_matches(">=1.0.0, <2.0.0", &v("2.0.0-rc.1")));
        // A requirement that names a prerelease keeps strict SemVer rules.
        assert!(environment_matches(">=1.1.0-rc.2, <2", &v("1.1.0-rc.3")));
        assert!(!environment_matches(">=1.1.0-rc.2, <2", &v("1.1.0-rc.1")));
    }

    #[test]
    fn api_requirement_is_read_from_the_manifest_and_checked() {
        let raw = "id = \"x\"\n[environment]\nversion = \">=1.0.0, <2\"\napi = \">=2.0, <3\"\n";
        assert_eq!(api_requirement(raw).as_deref(), Some(">=2.0, <3"));
        // Plugin API 1.0 manifests declare it at the top level.
        let raw = "id = \"x\"\napi = \">=1.0, <2\"\n[environment]\nversion = \">=1.0.0, <2\"\n";
        assert_eq!(api_requirement(raw).as_deref(), Some(">=1.0, <2"));
        let result = check(">=1.0.0, <2", Some(">=2.0, <3"), &v("1.0.0"), &v("1.0.0"));
        assert!(result.reason().unwrap().contains("plugin API"));
        assert!(check(">=1.0.0, <2", Some("^1"), &v("1.0.0"), &v("1.0.0")).is_compatible());
        assert!(check(">=1.0.0, <2", None, &v("1.0.0"), &v("1.0.0")).is_compatible());
    }

    #[test]
    fn incompatibility_explains_the_implicit_bound() {
        let reason = check(">=0.4.1", None, &v("1.0.0"), &v("1.0.0"));
        let reason = reason.reason().unwrap();
        assert!(
            reason.contains("bounded to") && reason.contains("<0.5.0"),
            "{reason}"
        );
    }

    /// Why a plugin does not run is data: English for logs and the audit
    /// log, the interface's language (versions isolated in right-to-left
    /// languages) when it is shown.
    #[test]
    fn incompatibilities_are_worded_when_shown() {
        use eov_plugin_api::i18n;
        struct English;
        impl Drop for English {
            fn drop(&mut self) {
                i18n::set_catalog(i18n::source(), None);
            }
        }
        let reason = incompatibility(">=2.0.0", None, &v("1.0.0"), &v("1.0.0")).unwrap();
        assert_eq!(
            reason,
            Incompatibility::Eov {
                required: ">=2.0.0".into(),
                bounded: Some(">=2.0.0, <3.0.0".into()),
                version: "1.0.0".into(),
            }
        );
        let english = "requires eov >=2.0.0 (bounded to >=2.0.0, <3.0.0); this is eov 1.0.0";
        assert_eq!(reason.to_string(), english);
        assert_eq!(reason.localized(), english);
        let invalid = incompatibility("1.0, <2", Some("nonsense"), &v("1.0.0"), &v("1.0.0"));
        assert!(matches!(
            invalid,
            Some(Incompatibility::InvalidApi(
                RequirementError::Invalid { .. }
            ))
        ));

        let _english = English;
        i18n::set_catalog(i18n::by_tag("fa").unwrap(), None);
        assert_eq!(reason.to_string(), english);
        assert!(!invalid.unwrap().to_string().contains('\u{2068}'));
        let shown = check(">=2.0.0", None, &v("1.0.0"), &v("1.0.0"));
        assert_eq!(shown.reason(), Some(reason.localized().as_str()));
        assert!(
            reason.localized().contains("\u{2068}>=2.0.0\u{2069}"),
            "{}",
            reason.localized()
        );
    }

    #[test]
    fn shared_bounded_table_matches_this_implementation() {
        // The release tooling evaluates the same table
        // (scripts/test_release_infrastructure.py).
        let table: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/semver-cases.json")).unwrap();
        for case in table["bounded_cases"].as_array().unwrap() {
            let requirement = case["requirement"].as_str().unwrap();
            let version = v(case["version"].as_str().unwrap());
            assert_eq!(
                environment_matches(requirement, &version),
                case["compatible"].as_bool().unwrap(),
                "{requirement} on {version}"
            );
        }
    }

    #[test]
    fn plugin_ids_are_safe_file_and_library_names() {
        for id in ["annotations", "eovae", "a", "plugin_2", "0x"] {
            validate_plugin_id(id).unwrap();
        }
        for id in [
            "",
            "C:x",
            "a:b",
            "../x",
            "a/b",
            "a\\b",
            "Upper",
            "with-dash",
            "_lead",
            "con",
            "nul",
            "lpt1",
            "a b",
        ] {
            assert!(validate_plugin_id(id).is_err(), "{id} was accepted");
        }
        assert!(validate_plugin_id(&"a".repeat(65)).is_err());
    }
}
