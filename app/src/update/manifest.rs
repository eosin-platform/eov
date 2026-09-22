//! Typed release manifest parsing and validation.

use anyhow::{Context, Result, bail};
use semver::{Version, VersionReq};
use serde::Deserialize;
use std::collections::BTreeMap;
use url::Url;

/// The first EOV release that publishes the self-update manifest format.
pub const MIN_SELF_UPDATE_RELEASE: &str = "0.4.5";
pub const EOV_REPOSITORY: &str = "https://github.com/eosin-platform/eov";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ManifestMetadata {
    pub schema: u32,
    pub kind: String,
    pub version: String,
    pub repository: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PlatformArtifact {
    pub version: String,
    pub sha256: String,
    pub url: String,
    #[serde(default)]
    pub environment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OfficialPluginCatalogEntry {
    #[serde(default)]
    pub name: Option<String>,
    pub repository: String,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PluginReleaseMetadata {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    pub repository: String,
    #[serde(default)]
    pub environment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EovReleaseManifest {
    pub metadata: Option<ManifestMetadata>,
    pub version: Version,
    pub repository: String,
    pub artifacts: BTreeMap<String, PlatformArtifact>,
    pub plugins: BTreeMap<String, OfficialPluginCatalogEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginReleaseManifest {
    pub metadata: Option<ManifestMetadata>,
    pub plugin: Option<PluginReleaseMetadata>,
    pub version: Version,
    pub repository: String,
    pub artifacts: BTreeMap<String, PlatformArtifact>,
}

impl EovReleaseManifest {
    pub fn parse(text: &str, expected_version: Option<&Version>) -> Result<Self> {
        validate_platform_keys(text)?;
        let raw: RawReleaseManifest = toml::from_str(text).context("invalid release.toml")?;
        let artifacts = raw.platform.into_artifacts();
        let version = infer_version(&artifacts, expected_version)?;
        validate_manifest_metadata(raw.manifest.as_ref(), "eov", &version, EOV_REPOSITORY)?;
        validate_artifacts(&artifacts, EOV_REPOSITORY, &version)?;

        let mut plugins = raw.plugins;
        for (id, entry) in &mut plugins {
            if entry.name.is_none() {
                entry.name = Some(title_case_id(id));
            }
            let plugin_version = Version::parse(&entry.version).with_context(|| {
                format!(
                    "official plugin '{id}' has invalid version {}",
                    entry.version
                )
            })?;
            let _ = plugin_version;
            canonical_repository(&entry.repository)
                .with_context(|| format!("official plugin '{id}' has invalid repository"))?;
        }

        Ok(Self {
            metadata: raw.manifest,
            version,
            repository: EOV_REPOSITORY.to_string(),
            artifacts,
            plugins,
        })
    }

    pub fn artifact(&self, key: &str) -> Result<&PlatformArtifact> {
        self.artifacts
            .get(key)
            .with_context(|| format!("release.toml has no artifact [{key}]"))
    }
}

impl PluginReleaseManifest {
    pub fn parse(text: &str, expected_repository: &str) -> Result<Self> {
        validate_platform_keys(text)?;
        let raw: RawReleaseManifest = toml::from_str(text).context("invalid release.toml")?;
        let artifacts = raw.platform.into_artifacts();
        let version = infer_version(&artifacts, None)?;
        let repository = canonical_repository(expected_repository)?;
        validate_manifest_metadata(raw.manifest.as_ref(), "plugin", &version, &repository)?;
        if raw.manifest.is_some() && raw.plugin.is_none() {
            bail!("schema-1 plugin release.toml is missing its [plugin] metadata");
        }
        validate_artifacts(&artifacts, &repository, &version)?;

        if let Some(plugin) = &raw.plugin {
            let plugin_version = Version::parse(&plugin.version).with_context(|| {
                format!("plugin metadata has invalid version {}", plugin.version)
            })?;
            if plugin_version != version {
                bail!(
                    "plugin metadata version {} does not match artifact version {}",
                    plugin.version,
                    version
                );
            }
            let plugin_repository = canonical_repository(&plugin.repository)?;
            if plugin_repository != repository {
                bail!(
                    "plugin metadata repository {} does not match expected repository {}",
                    plugin.repository,
                    expected_repository
                );
            }
            if let Some(environment) = &plugin.environment {
                VersionReq::parse(environment).with_context(|| {
                    format!("plugin environment is not a valid SemVer requirement: {environment}")
                })?;
            }
        }

        for artifact in artifacts.values() {
            if let Some(environment) = &artifact.environment {
                VersionReq::parse(environment).with_context(|| {
                    format!("plugin environment is not a valid SemVer requirement: {environment}")
                })?;
            }
        }

        if self_environment(&raw.plugin, &artifacts).is_none() {
            bail!("plugin release.toml has no EOV environment requirement");
        }

        Ok(Self {
            metadata: raw.manifest,
            plugin: raw.plugin,
            version,
            repository,
            artifacts,
        })
    }

    pub fn environment(&self) -> Option<&str> {
        self.plugin
            .as_ref()
            .and_then(|plugin| plugin.environment.as_deref())
            .or_else(|| {
                self.artifacts
                    .values()
                    .find_map(|artifact| artifact.environment.as_deref())
            })
    }

    pub fn artifact(&self, key: &str) -> Result<&PlatformArtifact> {
        self.artifacts
            .get(key)
            .with_context(|| format!("plugin release.toml has no artifact [{key}]"))
    }

    pub fn plugin_id(&self) -> Option<&str> {
        self.plugin.as_ref().and_then(|plugin| plugin.id.as_deref())
    }
}

fn self_environment<'a>(
    plugin: &'a Option<PluginReleaseMetadata>,
    artifacts: &'a BTreeMap<String, PlatformArtifact>,
) -> Option<&'a str> {
    plugin
        .as_ref()
        .and_then(|plugin| plugin.environment.as_deref())
        .or_else(|| {
            artifacts
                .values()
                .find_map(|artifact| artifact.environment.as_deref())
        })
}

pub fn normalize_version(value: &str) -> Result<Version> {
    let normalized = value.strip_prefix('v').unwrap_or(value);
    Version::parse(normalized).with_context(|| format!("invalid semantic version '{value}'"))
}

pub fn release_tag(version: &Version) -> String {
    format!("v{version}")
}

pub fn validate_self_update_target(version: &Version) -> Result<()> {
    let minimum = Version::parse(MIN_SELF_UPDATE_RELEASE).expect("valid self-update floor");
    if version < &minimum {
        bail!(
            "EOV self-update metadata is supported beginning with v{minimum}.\nInstall v{version} manually from GitHub Releases."
        );
    }
    Ok(())
}

pub fn canonical_repository(repository: &str) -> Result<String> {
    let url =
        Url::parse(repository).with_context(|| format!("invalid repository URL '{repository}'"))?;
    if url.scheme() != "https" || url.host_str() != Some("github.com") {
        bail!("repository must be an HTTPS GitHub URL: {repository}");
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("repository URL must not contain a query or fragment: {repository}");
    }
    let segments: Vec<_> = url.path_segments().into_iter().flatten().collect();
    if segments.len() != 2 || segments.iter().any(|segment| segment.is_empty()) {
        bail!("repository must identify exactly one GitHub repository: {repository}");
    }
    Ok(format!(
        "https://github.com/{}/{}",
        segments[0], segments[1]
    ))
}

/// Validate an artifact URL against the repository that supplied its manifest.
pub fn validate_artifact_url(url: &str, repository: &str) -> Result<()> {
    let expected_repository = canonical_repository(repository)?;
    let expected = Url::parse(&expected_repository).expect("canonical repository URL");
    let actual = Url::parse(url).with_context(|| format!("invalid artifact URL '{url}'"))?;
    if actual.scheme() != "https" || actual.host_str() != Some("github.com") {
        bail!("artifact URL must use HTTPS GitHub releases: {url}");
    }
    if actual.query().is_some() || actual.fragment().is_some() {
        bail!("artifact URL must not contain a query or fragment: {url}");
    }
    if actual.host_str() != expected.host_str()
        || actual
            .path_segments()
            .map(|segments| segments.take(2).collect::<Vec<_>>())
            != expected
                .path_segments()
                .map(|segments| segments.collect::<Vec<_>>())
    {
        bail!("artifact URL does not belong to repository {expected_repository}: {url}");
    }
    let segments: Vec<_> = actual.path_segments().into_iter().flatten().collect();
    if segments.len() != 6
        || segments[2] != "releases"
        || segments[3] != "download"
        || segments[4].is_empty()
        || segments[5].is_empty()
    {
        bail!("artifact URL is not an immutable GitHub release asset URL: {url}");
    }
    Ok(())
}

pub fn release_manifest_url(repository: &str, tag: Option<&str>) -> Result<String> {
    let repository = canonical_repository(repository)?;
    Ok(match tag {
        Some(tag) => format!("{repository}/releases/download/{tag}/release.toml"),
        None => format!("{repository}/releases/latest/download/release.toml"),
    })
}

pub fn title_case_id(id: &str) -> String {
    id.split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Arm64,
}

impl Architecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Arm64 => "arm64",
        }
    }

    pub fn current() -> Result<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            Ok(Self::X86_64)
        }
        #[cfg(target_arch = "aarch64")]
        {
            Ok(Self::Arm64)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            bail!("unsupported CPU architecture; EOV supports x86_64 and arm64")
        }
    }
}

fn infer_version(
    artifacts: &BTreeMap<String, PlatformArtifact>,
    expected_version: Option<&Version>,
) -> Result<Version> {
    let first = artifacts
        .values()
        .next()
        .context("release.toml contains no platform artifacts")?;
    let version = Version::parse(&first.version)
        .with_context(|| format!("invalid artifact version {}", first.version))?;
    for artifact in artifacts.values() {
        let parsed = Version::parse(&artifact.version)
            .with_context(|| format!("invalid artifact version {}", artifact.version))?;
        if parsed != version {
            bail!("release.toml contains platform artifacts with different versions");
        }
    }
    if let Some(expected) = expected_version
        && expected != &version
    {
        bail!(
            "release.toml version {} does not match requested version {}",
            version,
            expected
        );
    }
    Ok(version)
}

fn validate_manifest_metadata(
    metadata: Option<&ManifestMetadata>,
    expected_kind: &str,
    version: &Version,
    repository: &str,
) -> Result<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    if metadata.schema != 1 {
        bail!("unsupported release manifest schema {}", metadata.schema);
    }
    if metadata.kind != expected_kind {
        bail!(
            "release manifest kind '{}' does not match expected '{}',",
            metadata.kind,
            expected_kind
        );
    }
    if normalize_version(&metadata.version)? != *version {
        bail!("manifest metadata version does not match platform artifacts");
    }
    if canonical_repository(&metadata.repository)? != canonical_repository(repository)? {
        bail!("manifest metadata repository does not match the expected repository");
    }
    Ok(())
}

fn validate_artifacts(
    artifacts: &BTreeMap<String, PlatformArtifact>,
    repository: &str,
    version: &Version,
) -> Result<()> {
    if artifacts.is_empty() {
        bail!("release.toml contains no platform artifacts");
    }
    for (key, artifact) in artifacts {
        if normalize_version(&artifact.version)? != *version {
            bail!("artifact [{key}] has a mismatched version");
        }
        if artifact.sha256.len() != 64 || !artifact.sha256.chars().all(|ch| ch.is_ascii_hexdigit())
        {
            bail!("artifact [{key}] has an invalid SHA-256 digest");
        }
        validate_artifact_url(&artifact.url, repository)
            .with_context(|| format!("invalid URL for artifact [{key}]"))?;
    }
    Ok(())
}

fn collect_arch(output: &mut BTreeMap<String, PlatformArtifact>, prefix: &str, arch: RawArch) {
    if let Some(artifact) = arch.x86_64 {
        output.insert(format!("{prefix}.x86_64"), artifact);
    }
    if let Some(artifact) = arch.arm64 {
        output.insert(format!("{prefix}.arm64"), artifact);
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawPlatform {
    #[serde(default)]
    windows: Option<RawArch>,
    #[serde(default)]
    macos: Option<RawArch>,
    #[serde(default)]
    linux: Option<RawLinux>,
}

#[derive(Debug, Deserialize, Default)]
struct RawLinux {
    #[serde(default)]
    x86_64: Option<PlatformArtifact>,
    #[serde(default)]
    #[serde(alias = "aarch64")]
    arm64: Option<PlatformArtifact>,
    #[serde(default)]
    appimage: Option<RawArch>,
    #[serde(default)]
    flatpak: Option<RawArch>,
}

#[derive(Debug, Deserialize, Default)]
struct RawArch {
    #[serde(default)]
    x86_64: Option<PlatformArtifact>,
    #[serde(default)]
    #[serde(alias = "aarch64")]
    arm64: Option<PlatformArtifact>,
}

impl RawPlatform {
    fn into_artifacts(self) -> BTreeMap<String, PlatformArtifact> {
        let mut output = BTreeMap::new();
        if let Some(windows) = self.windows {
            collect_arch(&mut output, "platform.windows", windows);
        }
        if let Some(macos) = self.macos {
            collect_arch(&mut output, "platform.macos", macos);
        }
        if let Some(linux) = self.linux {
            if let Some(artifact) = linux.x86_64 {
                output.insert("platform.linux.x86_64".into(), artifact);
            }
            if let Some(artifact) = linux.arm64 {
                output.insert("platform.linux.arm64".into(), artifact);
            }
            if let Some(appimage) = linux.appimage {
                collect_arch(&mut output, "platform.linux.appimage", appimage);
            }
            if let Some(flatpak) = linux.flatpak {
                collect_arch(&mut output, "platform.linux.flatpak", flatpak);
            }
        }
        output
    }
}

fn validate_platform_keys(text: &str) -> Result<()> {
    let value: toml::Value = toml::from_str(text).context("invalid release.toml")?;
    let Some(platform) = value.get("platform").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    for (platform_name, platform_value) in platform {
        let Some(table) = platform_value.as_table() else {
            bail!("platform.{platform_name} must be a table");
        };
        let allowed = match platform_name.as_str() {
            "windows" | "macos" => &["x86_64", "arm64", "aarch64"][..],
            "linux" => &["x86_64", "arm64", "aarch64", "appimage", "flatpak"][..],
            other => bail!("unsupported release platform '{other}'"),
        };
        for key in table.keys() {
            if !allowed.contains(&key.as_str()) {
                bail!(
                    "unsupported architecture or platform entry 'platform.{platform_name}.{key}'"
                );
            }
        }
        if platform_name == "linux" {
            for variant in ["appimage", "flatpak"] {
                if let Some(variant_table) = table.get(variant).and_then(toml::Value::as_table) {
                    for key in variant_table.keys() {
                        if !["x86_64", "arm64", "aarch64"].contains(&key.as_str()) {
                            bail!(
                                "unsupported architecture entry 'platform.linux.{variant}.{key}'"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize, Default)]
struct RawReleaseManifest {
    #[serde(default)]
    manifest: Option<ManifestMetadata>,
    #[serde(default)]
    platform: RawPlatform,
    #[serde(default)]
    plugins: BTreeMap<String, OfficialPluginCatalogEntry>,
    #[serde(default)]
    plugin: Option<PluginReleaseMetadata>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(section: &str, filename: &str) -> String {
        format!(
            "[platform.{section}]\nversion = \"0.4.5\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov/releases/download/v0.4.5/{filename}\"\n",
            "a".repeat(64)
        )
    }

    #[test]
    fn parses_legacy_eov_manifest() {
        let text = artifact("windows.x86_64", "eov-v0.4.5-windows-x86_64.zip")
            + &artifact("linux.appimage.x86_64", "eov-v0.4.5-linux-x86_64.AppImage");
        let manifest =
            EovReleaseManifest::parse(&text, Some(&Version::parse("0.4.5").unwrap())).unwrap();
        assert!(manifest.metadata.is_none());
        assert_eq!(manifest.version, Version::parse("0.4.5").unwrap());
        assert!(manifest.artifacts.contains_key("platform.windows.x86_64"));
    }

    #[test]
    fn parses_checked_in_legacy_eov_fixture() {
        let text = include_str!("../../tests/fixtures/eov-v0.4.5-legacy-release.toml");
        let manifest =
            EovReleaseManifest::parse(text, Some(&Version::parse("0.4.5").unwrap())).unwrap();
        assert!(manifest.metadata.is_none());
        assert_eq!(manifest.plugins["gamepad"].version, "0.2.2");
    }

    #[test]
    fn parses_schema_one_and_catalog_name() {
        let text = format!(
            "[manifest]\nschema = 1\nkind = \"eov\"\nversion = \"0.4.9\"\nrepository = \"{EOV_REPOSITORY}\"\n\n[platform.windows.x86_64]\nversion = \"0.4.9\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov/releases/download/v0.4.9/eov.zip\"\n\n[plugins.gamepad]\nrepository = \"https://github.com/eosin-platform/eov-gamepad-plugin\"\nversion = \"0.2.3\"\ndescription = \"Gamepad\"\n",
            "b".repeat(64)
        );
        let manifest = EovReleaseManifest::parse(&text, None).unwrap();
        assert_eq!(manifest.metadata.as_ref().unwrap().schema, 1);
        assert_eq!(manifest.plugins["gamepad"].name.as_deref(), Some("Gamepad"));
    }

    #[test]
    fn rejects_wrong_kind_and_old_self_update_target() {
        let text = format!(
            "[manifest]\nschema = 1\nkind = \"plugin\"\nversion = \"0.4.9\"\nrepository = \"{EOV_REPOSITORY}\"\n\n{}",
            artifact("windows.x86_64", "eov.zip")
        );
        assert!(EovReleaseManifest::parse(&text, None).is_err());
        assert!(validate_self_update_target(&Version::parse("0.4.4").unwrap()).is_err());
    }

    #[test]
    fn rejects_artifact_from_wrong_repository() {
        let text = artifact("windows.x86_64", "eov.zip").replace(
            "https://github.com/eosin-platform/eov/",
            "https://github.com/other/project/",
        );
        assert!(EovReleaseManifest::parse(&text, None).is_err());
    }

    #[test]
    fn parses_legacy_plugin_environment() {
        let text = artifact("linux.x86_64", "gamepad-v0.2.3-linux-x86_64.eop")
            .replace("version = \"0.4.5\"", "version = \"0.2.3\"")
            .replace("eosin-platform/eov/", "eosin-platform/eov-gamepad-plugin/")
            + "environment = \">=0.4.1\"\n";
        let manifest = PluginReleaseManifest::parse(
            &text,
            "https://github.com/eosin-platform/eov-gamepad-plugin",
        )
        .unwrap();
        assert_eq!(manifest.environment(), Some(">=0.4.1"));
    }

    #[test]
    fn accepts_aarch64_as_arm64() {
        let text = artifact("linux.aarch64", "eov-v0.4.5-linux-arm64.AppImage");
        let manifest = EovReleaseManifest::parse(&text, None).unwrap();
        assert!(manifest.artifacts.contains_key("platform.linux.arm64"));
    }

    #[test]
    fn rejects_unsupported_architecture() {
        let text = artifact("linux.riscv64", "eov-riscv64.AppImage");
        let error = EovReleaseManifest::parse(&text, None).unwrap_err();
        assert!(error.to_string().contains("unsupported architecture"));
    }

    #[test]
    fn parses_schema_one_plugin_metadata() {
        let text = format!(
            "[manifest]\nschema = 1\nkind = \"plugin\"\nversion = \"0.2.3\"\nrepository = \"https://github.com/eosin-platform/eov-gamepad-plugin\"\n\n[plugin]\nid = \"gamepad\"\nname = \"Gamepad\"\nversion = \"0.2.3\"\nrepository = \"https://github.com/eosin-platform/eov-gamepad-plugin\"\nenvironment = \">=0.4.1\"\n\n[platform.linux.x86_64]\nversion = \"0.2.3\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov-gamepad-plugin/releases/download/v0.2.3/gamepad.eop\"\nenvironment = \">=0.4.1\"\n",
            "c".repeat(64)
        );
        let manifest = PluginReleaseManifest::parse(
            &text,
            "https://github.com/eosin-platform/eov-gamepad-plugin",
        )
        .unwrap();
        assert_eq!(manifest.plugin_id(), Some("gamepad"));
        assert_eq!(manifest.environment(), Some(">=0.4.1"));
    }

    #[test]
    fn parses_checked_in_schema_fixtures() {
        let eov = EovReleaseManifest::parse(
            include_str!("../../tests/fixtures/eov-schema1-release.toml"),
            None,
        )
        .unwrap();
        let plugin = PluginReleaseManifest::parse(
            include_str!("../../tests/fixtures/plugin-schema1-release.toml"),
            "https://github.com/eosin-platform/eov-gamepad-plugin",
        )
        .unwrap();
        let legacy_plugin = PluginReleaseManifest::parse(
            include_str!("../../tests/fixtures/plugin-legacy-release.toml"),
            "https://github.com/eosin-platform/eov-gamepad-plugin",
        )
        .unwrap();
        assert_eq!(eov.version, Version::parse("0.4.9").unwrap());
        assert_eq!(plugin.plugin_id(), Some("gamepad"));
        assert_eq!(legacy_plugin.environment(), Some(">=0.4.1"));
    }
}
