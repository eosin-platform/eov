//! Plugin installation, updates, removal, inspection, provenance, enable /
//! disable state and lockfiles.
//!
//! Everything here runs through a [`PluginContext`], which carries the plugin
//! directory, the host version and platform, the deployment policy, the
//! signature verifier, the HTTP client and the prompter. The CLI builds one
//! with [`PluginContext::for_cli`]; tests and a future GUI inject their own.
//! The read-only functions ([`inventory`], [`list_plugins`]) never touch the
//! network and never prompt.

use super::compat::{self, Compatibility};
use super::manifest::{
    EOV_REPOSITORY, EovReleaseManifest, MANIFEST_FILE, PluginReleaseManifest, canonical_repository,
    normalize_version, release_manifest_url, release_tag, same_repository, title_case_id,
    validate_artifact_url,
};
use super::network::{FetchError, NetworkPermission, ReleaseClient};
use super::platform::Platform;
use super::policy::{self, Policy};
use super::prompt::Interaction;
use super::registry::{
    self, ManagedPlugin, ManagedRegistry, PluginDirLock, SOURCE_GITHUB, SOURCE_LOCAL,
    SOURCE_OFFICIAL,
};
use super::signature::{ManifestIdentity, SIGNATURE_SUFFIX, SignatureStatus, Verifier};
use crate::plugins::discovery::{
    InspectedPluginPackage, inspect_plugin_package, inspect_staged_plugin_package,
    invalidate_plugin_package_cache, package_cache,
};
use anyhow::{Context, Result, bail};
use semver::Version;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Ids of plugins published by the eov project. A third-party package that
/// declares one of these ids would shadow the official plugin, which needs
/// `--force` even when the official catalog cannot be fetched.
pub const KNOWN_OFFICIAL_IDS: &[&str] = &["annotations", "cellaris", "eovae", "gamepad"];

/// Output format of `eov plugin info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Plain,
    Toml,
    Json,
}

/// Everything plugin management needs from its environment.
pub struct PluginContext {
    pub plugin_dir: PathBuf,
    /// Read-only plugin directories provisioned by the deployment policy.
    pub system_plugin_dirs: Vec<PathBuf>,
    pub host: Version,
    pub platform: Platform,
    pub policy: Policy,
    pub verifier: Verifier,
    pub interaction: Interaction,
    pub client: ReleaseClient,
    /// Print machine-readable JSON instead of text.
    pub json: bool,
}

impl PluginContext {
    /// The context of a CLI invocation.
    pub fn for_cli(plugin_dir: &Path, assume_yes: bool, json: bool) -> Result<Self> {
        let policy = policy::current().clone();
        Ok(Self {
            plugin_dir: plugin_dir.to_path_buf(),
            system_plugin_dirs: policy.plugin_dirs.clone(),
            host: compat::host_version(),
            platform: Platform::current()?,
            verifier: Verifier::for_build(&policy)?,
            policy,
            interaction: Interaction::terminal(assume_yes),
            client: ReleaseClient::for_build(),
            json,
        })
    }

    fn plain(&self) -> bool {
        !self.json
    }
}

// ---------------------------------------------------------------------------
// Inventory (local, read-only)
// ---------------------------------------------------------------------------

/// Where an installed package came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// `official`, `github`, `local`, `unmanaged` or `system`.
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_tag: Option<String>,
}

impl Provenance {
    pub fn describe(&self) -> String {
        match (self.kind.as_str(), &self.repository) {
            (SOURCE_OFFICIAL, _) => "official".to_string(),
            (SOURCE_GITHUB, Some(repository)) => {
                format!(
                    "github ({})",
                    repository.trim_start_matches("https://github.com/")
                )
            }
            (kind, _) => kind.to_string(),
        }
    }

    fn same_origin(&self, other: &Provenance) -> bool {
        if self.kind != other.kind {
            return false;
        }
        match (&self.repository, &other.repository) {
            (Some(left), Some(right)) => same_repository(left, right),
            (None, None) => true,
            _ => self.kind == SOURCE_LOCAL,
        }
    }
}

/// An installed, parseable plugin package.
#[derive(Debug, Clone, Serialize)]
pub struct InstalledPackage {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub package_path: PathBuf,
    pub sha256: String,
    pub provenance: Provenance,
    pub disabled: bool,
    pub compatible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incompatibility: Option<String>,
    /// The package eov loads for this id (the highest compatible version).
    pub selected: bool,
    /// The file differs from what eov installed (edited or replaced by hand).
    pub modified: bool,
    /// From a read-only, policy-provisioned plugin directory.
    pub system: bool,
    #[serde(skip)]
    pub environment: String,
    #[serde(skip)]
    pub raw_manifest: String,
}

/// A `.eop` file that cannot be read as a plugin package.
#[derive(Debug, Clone, Serialize)]
pub struct BrokenPackage {
    pub package_path: PathBuf,
    pub error: String,
}

/// The plugins installed in a plugin directory.
#[derive(Debug, Clone)]
pub struct Inventory {
    pub packages: Vec<InstalledPackage>,
    pub broken: Vec<BrokenPackage>,
    pub registry: ManagedRegistry,
}

impl Inventory {
    /// The package eov loads for `id`, if any.
    pub fn selected(&self, id: &str) -> Option<&InstalledPackage> {
        self.packages
            .iter()
            .find(|package| package.id == id && package.selected)
    }

    /// The package eov loads for `id`, or any package with that id.
    pub fn representative(&self, id: &str) -> Option<&InstalledPackage> {
        self.selected(id)
            .or_else(|| self.packages.iter().find(|package| package.id == id))
    }

    /// Every package with `id` (duplicates included).
    pub fn with_id(&self, id: &str) -> Vec<&InstalledPackage> {
        self.packages
            .iter()
            .filter(|package| package.id == id)
            .collect()
    }

    pub fn ids(&self) -> BTreeSet<String> {
        self.packages
            .iter()
            .map(|package| package.id.clone())
            .collect()
    }
}

fn package_paths(directory: &Path) -> Vec<PathBuf> {
    let mut paths = fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("eop")))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// Inspects every package in `plugin_dir` (and the read-only `system_dirs`)
/// without loading native code or using the network.
pub fn inventory(plugin_dir: &Path, system_dirs: &[PathBuf], host: &Version) -> Result<Inventory> {
    let registry = registry::load(plugin_dir)?;
    let mut packages = Vec::new();
    let mut broken = Vec::new();
    let directories = std::iter::once((plugin_dir, false))
        .chain(system_dirs.iter().map(|dir| (dir.as_path(), true)));
    for (directory, system) in directories {
        for path in package_paths(directory) {
            match inspect_plugin_package(&path, true) {
                Ok(inspected) => {
                    packages.push(describe_package(inspected, &registry, host, system));
                }
                Err(error) => broken.push(BrokenPackage {
                    package_path: path,
                    error: error.to_string(),
                }),
            }
        }
    }
    select_packages(&mut packages);
    Ok(Inventory {
        packages,
        broken,
        registry,
    })
}

fn describe_package(
    inspected: InspectedPluginPackage,
    registry: &ManagedRegistry,
    host: &Version,
    system: bool,
) -> InstalledPackage {
    let id = inspected.manifest.id.clone();
    let sha256 = inspected.sha256.clone().unwrap_or_default();
    let managed = registry.plugins.get(&id).filter(|entry| {
        !system
            && inspected
                .package_path
                .file_name()
                .is_some_and(|name| name == OsStr::new(&entry.file))
    });
    let provenance = match managed {
        Some(entry) => Provenance {
            kind: entry.source.clone(),
            repository: entry.repository.clone(),
            release_tag: entry.release_tag.clone(),
        },
        None => Provenance {
            kind: if system { "system" } else { "unmanaged" }.to_string(),
            repository: inspected.manifest.repository.clone(),
            release_tag: None,
        },
    };
    let compatibility = compat::check_manifest(&inspected.manifest, &inspected.raw_manifest, host);
    InstalledPackage {
        name: inspected.manifest.name.clone(),
        version: inspected.manifest.version.clone(),
        description: inspected.manifest.description.clone(),
        modified: managed.is_some_and(|entry| !entry.sha256.eq_ignore_ascii_case(&sha256)),
        disabled: registry.is_disabled(&id),
        compatible: compatibility.is_compatible(),
        incompatibility: compatibility.reason().map(str::to_string),
        selected: false,
        system,
        environment: inspected.manifest.environment.version.clone(),
        raw_manifest: inspected.raw_manifest,
        package_path: inspected.package_path,
        sha256,
        provenance,
        id,
    }
}

/// Marks, per id, the package discovery loads: the highest compatible
/// version, ties broken by path (the user's directory before system ones).
fn select_packages(packages: &mut [InstalledPackage]) {
    let mut best: BTreeMap<String, usize> = BTreeMap::new();
    for (index, package) in packages.iter().enumerate() {
        if !package.compatible {
            continue;
        }
        match best.get(&package.id) {
            Some(&current) if !is_better(package, &packages[current]) => {}
            _ => {
                best.insert(package.id.clone(), index);
            }
        }
    }
    for index in best.into_values() {
        packages[index].selected = true;
    }
}

fn is_better(candidate: &InstalledPackage, current: &InstalledPackage) -> bool {
    match compare_versions(&candidate.version, &current.version) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            (candidate.system, &candidate.package_path) < (current.system, &current.package_path)
        }
    }
}

pub(crate) fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (Version::parse(left), Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

// ---------------------------------------------------------------------------
// The official catalog
// ---------------------------------------------------------------------------

/// A verified eov release manifest.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub manifest: EovReleaseManifest,
    pub signature: SignatureStatus,
}

/// Fetches and verifies the eov release manifest for `tag` (or the latest
/// release when `tag` is `None`).
pub fn fetch_catalog(
    client: &ReleaseClient,
    permission: &NetworkPermission,
    verifier: &Verifier,
    tag: Option<&str>,
    expected_version: Option<&Version>,
) -> Result<Catalog> {
    let url = release_manifest_url(EOV_REPOSITORY, tag)?;
    let text = client.get_text(permission, &url).map_err(|error| {
        if error.is_not_found() {
            match tag {
                Some(tag) => anyhow::anyhow!(
                    "eov release {tag} publishes no {MANIFEST_FILE} ({url} does not exist)."
                ),
                None => anyhow::anyhow!("the latest eov release publishes no {MANIFEST_FILE}."),
            }
        } else {
            anyhow::Error::new(error)
        }
    })?;
    let signature_url = format!("{url}{SIGNATURE_SUFFIX}");
    let signature_text = match client.get_text(permission, &signature_url) {
        Ok(text) => Some(text),
        Err(error) if error.is_not_found() => None,
        Err(error) => return Err(error.into()),
    };
    catalog_from_text(&text, signature_text.as_deref(), verifier, expected_version)
}

/// Parses and verifies an eov release manifest (also used for offline
/// bundles).
pub fn catalog_from_text(
    text: &str,
    signature_text: Option<&str>,
    verifier: &Verifier,
    expected_version: Option<&Version>,
) -> Result<Catalog> {
    let manifest = EovReleaseManifest::parse(text, expected_version)?;
    let signature = verifier.verify(
        text.as_bytes(),
        signature_text,
        manifest.declares_signature(),
        ManifestIdentity {
            kind: "eov",
            version: &manifest.version,
            repository: EOV_REPOSITORY,
        },
    )?;
    Ok(Catalog {
        manifest,
        signature,
    })
}

/// The catalog matching this host, explaining development builds.
fn fetch_host_catalog(ctx: &PluginContext, permission: &NetworkPermission) -> Result<Catalog> {
    fetch_catalog(
        &ctx.client,
        permission,
        &ctx.verifier,
        Some(&release_tag(&ctx.host)),
        Some(&ctx.host),
    )
    .with_context(|| {
        format!(
            "The official plugin catalog for eov {} is unavailable. If this is a development build, install an explicit release instead, e.g. eov plugin install github.com/eosin-platform/eov-gamepad-plugin@v1.0.0",
            ctx.host
        )
    })
}

// ---------------------------------------------------------------------------
// Sources and resolution
// ---------------------------------------------------------------------------

/// What the user asked to install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginSource {
    Official {
        id: String,
        version: Option<Version>,
    },
    Github {
        repository: String,
        tag: Option<String>,
    },
    Local(PathBuf),
}

/// Parses `gamepad`, `gamepad@1.0.0`, `github.com/o/r[@tag]` or a path.
pub fn parse_source(value: &str) -> Result<PluginSource> {
    let looks_like_path = value.ends_with(".eop")
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with(".\\")
        || value.starts_with("..\\")
        || Path::new(value).is_absolute()
        || Path::new(value).is_file();
    if looks_like_path {
        return Ok(PluginSource::Local(PathBuf::from(value)));
    }
    let repository_value = value.strip_prefix("https://").unwrap_or(value);
    if repository_value.starts_with("github.com/") {
        let (repository, tag) = match repository_value.split_once('@') {
            Some((repository, tag)) if !tag.is_empty() => (repository, Some(tag.to_string())),
            Some(_) => bail!("missing release tag after '@' in '{value}'"),
            None => (repository_value, None),
        };
        return Ok(PluginSource::Github {
            repository: canonical_repository(&format!("https://{repository}"))?,
            tag,
        });
    }
    let (id, version) = match value.split_once('@') {
        Some((id, version)) => (id, Some(normalize_version(version)?)),
        None => (value, None),
    };
    compat::validate_plugin_id(id).with_context(|| format!("invalid plugin source '{value}'"))?;
    Ok(PluginSource::Official {
        id: id.to_string(),
        version,
    })
}

/// Where the bytes of a resolved package come from.
#[derive(Debug, Clone)]
pub enum PackageFetch {
    Download { url: String, sha256: String },
    LocalFile { path: PathBuf, sha256: String },
}

impl PackageFetch {
    fn display(&self) -> String {
        match self {
            Self::Download { url, .. } => url.rsplit('/').next().unwrap_or(url).to_string(),
            Self::LocalFile { path, .. } => path.display().to_string(),
        }
    }
}

/// A package ready to be fetched and installed.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedPackage {
    pub id: String,
    pub name: String,
    pub version: Version,
    pub provenance: Provenance,
    /// The eov requirement the release metadata declares, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Whether the SHA-256 comes from the (signed) eov catalog rather than
    /// from the plugin repository's own release manifest.
    pub pinned_by_catalog: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<SignatureStatus>,
    pub sha256: String,
    pub artifact: String,
    #[serde(skip)]
    pub fetch: PackageFetch,
}

/// Resolves an official plugin through the eov catalog `catalog`.
pub fn resolve_official(
    ctx: &PluginContext,
    permission: &NetworkPermission,
    catalog: &Catalog,
    id: &str,
    version: Option<&Version>,
) -> Result<ResolvedPackage> {
    let Some(entry) = catalog.manifest.plugins.get(id) else {
        let available: Vec<_> = catalog.manifest.plugins.keys().cloned().collect();
        bail!(
            "'{id}' is not an official plugin in the eov {} catalog (official plugins: {}).",
            catalog.manifest.version,
            if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            }
        );
    };
    let catalog_version = normalize_version(&entry.version)?;
    let target = version.cloned().unwrap_or_else(|| catalog_version.clone());
    let repository = canonical_repository(&entry.repository)?;
    let name = entry.name.clone().unwrap_or_else(|| title_case_id(id));
    let key = ctx.platform.plugin_artifact_key();
    if target == catalog_version
        && let Some(artifact) = entry.artifacts.get(&key)
    {
        return Ok(ResolvedPackage {
            id: id.to_string(),
            name,
            version: target,
            provenance: Provenance {
                kind: SOURCE_OFFICIAL.into(),
                repository: Some(repository),
                release_tag: Some(entry.tag()),
            },
            environment: entry.environment.clone(),
            pinned_by_catalog: true,
            signature: Some(catalog.signature.clone()),
            sha256: artifact.sha256.to_ascii_lowercase(),
            artifact: artifact.file_name().unwrap_or("artifact").to_string(),
            fetch: PackageFetch::Download {
                url: artifact.url.clone(),
                sha256: artifact.sha256.to_ascii_lowercase(),
            },
        });
    }
    // Older catalogs (and explicitly requested versions) are resolved through
    // the plugin's own release manifest.
    let tag = if target == catalog_version {
        entry.tag()
    } else {
        release_tag(&target)
    };
    let mut resolved = resolve_github_release(ctx, permission, &repository, Some(&tag), Some(id))?;
    if resolved.version != target {
        bail!(
            "official plugin {id} release {tag} contains version {}, expected {target}",
            resolved.version
        );
    }
    resolved.provenance.kind = SOURCE_OFFICIAL.into();
    resolved.name = name;
    Ok(resolved)
}

/// Resolves a plugin release published in a GitHub repository.
pub fn resolve_github_release(
    ctx: &PluginContext,
    permission: &NetworkPermission,
    repository: &str,
    tag: Option<&str>,
    expected_id: Option<&str>,
) -> Result<ResolvedPackage> {
    let url = release_manifest_url(repository, tag)?;
    let text = ctx.client.get_text(permission, &url).map_err(|error| {
        if error.is_not_found() {
            anyhow::Error::new(error).context(format!(
                "{} has no plugin release manifest for {}",
                repository,
                tag.unwrap_or("its latest release")
            ))
        } else {
            anyhow::Error::new(error)
        }
    })?;
    let manifest = PluginReleaseManifest::parse(&text, repository)?;
    let id = match (manifest.plugin_id(), expected_id) {
        (Some(declared), Some(expected)) if declared != expected => {
            bail!("the release declares plugin id '{declared}', but '{expected}' was expected")
        }
        (Some(declared), _) => declared.to_string(),
        (None, Some(expected)) => expected.to_string(),
        (None, None) => bail!(
            "the release in {repository} does not declare a plugin id. Check which plugin it contains and re-run with --id <id>."
        ),
    };
    compat::validate_plugin_id(&id)?;
    let key = ctx.platform.plugin_artifact_key();
    let artifact = manifest.artifact(&key).with_context(|| {
        format!(
            "{repository} publishes no {} package for this release",
            ctx.platform
        )
    })?;
    let artifact_tag = validate_artifact_url(&artifact.url, repository)?;
    if let Some(tag) = tag
        && artifact_tag != tag
    {
        bail!("the release manifest for {tag} points at an artifact of release {artifact_tag}");
    }
    let name = manifest
        .plugin
        .as_ref()
        .and_then(|plugin| plugin.name.clone())
        .unwrap_or_else(|| title_case_id(&id));
    Ok(ResolvedPackage {
        id,
        name,
        version: manifest.version.clone(),
        provenance: Provenance {
            kind: SOURCE_GITHUB.into(),
            repository: Some(canonical_repository(repository)?),
            release_tag: Some(artifact_tag),
        },
        environment: manifest.environment().map(str::to_string),
        pinned_by_catalog: false,
        signature: None,
        sha256: artifact.sha256.to_ascii_lowercase(),
        artifact: artifact.file_name().unwrap_or("artifact").to_string(),
        fetch: PackageFetch::Download {
            url: artifact.url.clone(),
            sha256: artifact.sha256.to_ascii_lowercase(),
        },
    })
}

/// The newest stable release in `repository` whose eov requirement accepts
/// `target`. The latest release is tried first, so the GitHub API (60
/// unauthenticated requests per hour) is only used when it is incompatible.
pub fn resolve_github_compatible(
    ctx: &PluginContext,
    permission: &NetworkPermission,
    repository: &str,
    expected_id: &str,
    target: &Version,
) -> Result<Option<ResolvedPackage>> {
    let compatible = |resolved: &ResolvedPackage| {
        resolved.version.pre.is_empty()
            && resolved
                .environment
                .as_deref()
                .is_some_and(|requirement| compat::environment_matches(requirement, target))
    };
    match resolve_github_release(ctx, permission, repository, None, Some(expected_id)) {
        Ok(latest) if compatible(&latest) => return Ok(Some(latest)),
        Ok(_) => {}
        Err(error) if is_not_found(&error) => {}
        Err(error) => return Err(error),
    }

    let api = repository.replace("https://github.com/", "https://api.github.com/repos/");
    let mut tags = Vec::new();
    for page in 1..=10 {
        let value = ctx.client.get_json(
            permission,
            &format!("{api}/releases?per_page=100&page={page}"),
        )?;
        let releases = value
            .as_array()
            .with_context(|| format!("GitHub releases response for {repository} is not a list"))?;
        for release in releases {
            let skip = release.get("draft").and_then(|value| value.as_bool()) == Some(true)
                || release.get("prerelease").and_then(|value| value.as_bool()) == Some(true);
            let tag = release.get("tag_name").and_then(|value| value.as_str());
            if let (false, Some(tag)) = (skip, tag) {
                let version = normalize_version(tag.rsplit('/').next().unwrap_or(tag)).ok();
                tags.push((version, tag.to_string()));
            }
        }
        if releases.len() < 100 {
            break;
        }
    }
    // Newest first; tags that are not versions go last.
    tags.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, tag) in tags {
        match resolve_github_release(ctx, permission, repository, Some(&tag), Some(expected_id)) {
            Ok(candidate) if compatible(&candidate) => return Ok(Some(candidate)),
            Ok(_) => {}
            Err(error) => {
                tracing::debug!("skipping {repository} release {tag}: {error:#}");
            }
        }
    }
    Ok(None)
}

pub(crate) fn is_not_found(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<FetchError>()
            .is_some_and(FetchError::is_not_found)
    })
}

/// Inspects a local package for installation.
pub fn resolve_local(path: &Path) -> Result<ResolvedPackage> {
    let path = path
        .canonicalize()
        .with_context(|| format!("local plugin package does not exist: {}", path.display()))?;
    let inspected = inspect_staged_plugin_package(&path, true)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
    compat::validate_plugin_id(&inspected.manifest.id)?;
    let sha256 = inspected.sha256.clone().unwrap_or_default();
    Ok(ResolvedPackage {
        id: inspected.manifest.id.clone(),
        name: inspected.manifest.name.clone(),
        version: normalize_version(&inspected.manifest.version)?,
        provenance: Provenance {
            kind: SOURCE_LOCAL.into(),
            repository: inspected.manifest.repository.clone(),
            release_tag: None,
        },
        environment: Some(inspected.manifest.environment.version.clone()),
        pinned_by_catalog: false,
        signature: None,
        artifact: path.display().to_string(),
        sha256: sha256.clone(),
        fetch: PackageFetch::LocalFile { path, sha256 },
    })
}

// ---------------------------------------------------------------------------
// Staging and activation (also used by `eov update`)
// ---------------------------------------------------------------------------

/// A package copied or downloaded into the plugin directory under a
/// temporary name and verified. Dropping it removes the file.
pub struct StagedPackage {
    pub path: tempfile::TempPath,
    pub inspected: InspectedPluginPackage,
    pub sha256: String,
}

/// Fetches `resolved` into a temporary file in the plugin directory and
/// checks that the package is what the release metadata promised and that
/// it is compatible with `eov` (the version it will run under).
pub fn stage_package(
    ctx: &PluginContext,
    permission: Option<&NetworkPermission>,
    resolved: &ResolvedPackage,
    eov: &Version,
) -> Result<StagedPackage> {
    fs::create_dir_all(&ctx.plugin_dir)
        .with_context(|| format!("could not create {}", ctx.plugin_dir.display()))?;
    let path = match &resolved.fetch {
        PackageFetch::Download { url, sha256 } => {
            let permission = permission.context("network permission is required to download")?;
            ctx.client
                .download_verified(permission, url, sha256, &ctx.plugin_dir)?
                .file
                .into_temp_path()
        }
        PackageFetch::LocalFile { path, sha256 } => copy_verified(path, sha256, &ctx.plugin_dir)?,
    };
    verify_staged(path, resolved, eov)
}

/// Copies a local package into `directory`, checking its SHA-256.
pub fn copy_verified(
    source: &Path,
    expected: &str,
    directory: &Path,
) -> Result<tempfile::TempPath> {
    let temp = tempfile::Builder::new()
        .prefix(".eov-stage-")
        .suffix(".part")
        .tempfile_in(directory)?
        .into_temp_path();
    fs::copy(source, &temp).with_context(|| format!("could not copy {}", source.display()))?;
    let actual = package_cache::sha256_file(&temp)?;
    if !actual.eq_ignore_ascii_case(expected) {
        bail!(
            "{} changed while it was being installed (SHA-256 {actual}, expected {expected})",
            source.display()
        );
    }
    Ok(temp)
}

fn verify_staged(
    path: tempfile::TempPath,
    resolved: &ResolvedPackage,
    eov: &Version,
) -> Result<StagedPackage> {
    let inspected = inspect_staged_plugin_package(&path, true)
        .map_err(|error| anyhow::anyhow!("the downloaded package is not valid: {error}"))?;
    let manifest = &inspected.manifest;
    if manifest.id != resolved.id {
        bail!(
            "the package contains plugin '{}', but the release metadata promised '{}'",
            manifest.id,
            resolved.id
        );
    }
    if normalize_version(&manifest.version)? != resolved.version {
        bail!(
            "the package contains version {}, but the release metadata promised {}",
            manifest.version,
            resolved.version
        );
    }
    if let Some(environment) = &resolved.environment
        && manifest.environment.version != *environment
    {
        bail!(
            "the package requires eov {}, but the release metadata declared {environment}",
            manifest.environment.version
        );
    }
    if let Compatibility::Incompatible(reason) =
        compat::check_manifest(manifest, &inspected.raw_manifest, eov)
    {
        bail!("plugin {} {} {reason}", manifest.id, manifest.version);
    }
    let sha256 = inspected.sha256.clone().unwrap_or_default();
    Ok(StagedPackage {
        path,
        inspected,
        sha256,
    })
}

/// The result of activating a staged package.
#[derive(Debug, Clone, Serialize)]
pub struct Activated {
    pub id: String,
    pub version: String,
    pub package_path: PathBuf,
    /// Packages that were replaced (moved to `previous_store` when given).
    pub replaced: Vec<PathBuf>,
}

/// Moves `staged` to `<plugin-dir>/<id>.eop`, supersedes every other package
/// with the same id and records the provenance, as one transaction: if any
/// step fails, the previous packages and registry are restored. The caller
/// must hold the [`PluginDirLock`].
pub fn activate_package(
    plugin_dir: &Path,
    staged: StagedPackage,
    provenance: &Provenance,
    previous_store: Option<&Path>,
) -> Result<Activated> {
    let id = staged.inspected.manifest.id.clone();
    compat::validate_plugin_id(&id)?;
    let target = plugin_dir.join(format!("{id}.eop"));
    if target.is_file()
        && let Ok(existing) = inspect_plugin_package(&target, false)
        && existing.manifest.id != id
    {
        bail!(
            "{} contains a different plugin ('{}'); rename or remove it before installing '{id}'",
            target.display(),
            existing.manifest.id
        );
    }
    let staged_path = staged.path.to_path_buf();
    let superseded: Vec<PathBuf> = package_paths(plugin_dir)
        .into_iter()
        .filter(|path| path != &staged_path)
        .filter(|path| {
            path == &target
                || inspect_plugin_package(path, false)
                    .is_ok_and(|package| package.manifest.id == id)
        })
        .collect();

    let mut registry = registry::load(plugin_dir)?;
    let backups = move_to_backups(&superseded)?;
    let entry = ManagedPlugin {
        source: provenance.kind.clone(),
        repository: provenance.repository.clone(),
        release_tag: provenance.release_tag.clone(),
        version: staged.inspected.manifest.version.clone(),
        sha256: staged.sha256.clone(),
        file: format!("{id}.eop"),
        installed_at: Some(unix_now()),
    };
    registry.plugins.insert(id.clone(), entry);
    let result = fs::rename(&staged_path, &target)
        .with_context(|| format!("could not move the package to {}", target.display()))
        .and_then(|()| registry::save(plugin_dir, &registry));
    if let Err(error) = result {
        if !staged_path.exists() && target.exists() {
            let _ = fs::rename(&target, &staged_path);
        }
        restore_backups(&backups)?;
        return Err(error);
    }
    // The staged file now lives at `target`; do not let the guard delete it.
    let _ = staged.path.keep();
    registry::sync_directory(plugin_dir);

    package_cache::PackageCache::default().record_rename(&staged_path, &target);
    for path in &superseded {
        if path != &target {
            let _ = invalidate_plugin_package_cache(path);
        }
    }
    let replaced = finish_backups(backups, previous_store);
    Ok(Activated {
        id,
        version: staged.inspected.manifest.version.clone(),
        package_path: target,
        replaced,
    })
}

type Backup = (PathBuf, PathBuf);

fn move_to_backups(packages: &[PathBuf]) -> Result<Vec<Backup>> {
    let mut backups = Vec::new();
    for package in packages {
        let name = package
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("plugin.eop");
        let backup = package.with_file_name(format!(".{name}.backup-{}", random_suffix()));
        if let Err(error) = fs::rename(package, &backup) {
            restore_backups(&backups)?;
            return Err(error)
                .with_context(|| format!("could not move {} aside", package.display()));
        }
        backups.push((package.clone(), backup));
    }
    Ok(backups)
}

fn restore_backups(backups: &[Backup]) -> Result<()> {
    for (original, backup) in backups.iter().rev() {
        if backup.exists() {
            fs::rename(backup, original).with_context(|| {
                format!(
                    "could not restore {} from {}",
                    original.display(),
                    backup.display()
                )
            })?;
        }
    }
    Ok(())
}

/// Deletes backups, or keeps them in `previous_store` for `eov update
/// --rollback`. Returns the original paths.
fn finish_backups(backups: Vec<Backup>, previous_store: Option<&Path>) -> Vec<PathBuf> {
    let mut replaced = Vec::new();
    for (original, backup) in backups {
        let kept = previous_store.is_some_and(|store| {
            fs::create_dir_all(store).is_ok()
                && original
                    .file_name()
                    .is_some_and(|name| fs::rename(&backup, store.join(name)).is_ok())
        });
        if !kept {
            let _ = fs::remove_file(&backup);
        }
        replaced.push(original);
    }
    replaced
}

/// A random, filename-safe suffix.
pub(crate) fn random_suffix() -> String {
    let name = tempfile::Builder::new()
        .prefix("")
        .rand_bytes(10)
        .make(|path| Ok(path.to_path_buf()))
        .map(|named| {
            named
                .path()
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    if name.is_empty() {
        format!(
            "{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        )
    } else {
        name
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Options of `eov plugin install` and `eov plugin update`.
#[derive(Debug, Clone, Default)]
pub struct InstallOptions {
    pub allow_network: bool,
    pub dry_run: bool,
    /// Replace a plugin from a different source, or shadow an official id.
    pub force: bool,
    /// The plugin id a GitHub release must contain.
    pub id: Option<String>,
    /// `update`: the plugin must already be installed.
    pub require_existing: bool,
}

/// What `install` plans to do.
#[derive(Debug, Clone, Serialize)]
pub struct InstallPlan {
    pub package: ResolvedPackage,
    pub action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_provenance: Option<Provenance>,
    pub warnings: Vec<String>,
    /// Needs `--force` or an explicit interactive confirmation.
    pub needs_override: bool,
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
struct InstallReport<'a> {
    plan: &'a InstallPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    installed: Option<&'a Activated>,
}

/// `eov plugin install` / `eov plugin update`.
pub fn install(ctx: &mut PluginContext, source_text: &str, options: &InstallOptions) -> Result<()> {
    ctx.policy.ensure_valid()?;
    let source = parse_source(source_text)?;
    let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host)?;
    let source = if options.require_existing {
        prepare_update_source(source, &inventory)?
    } else {
        source
    };
    match &source {
        PluginSource::Github { .. } => ctx
            .policy
            .ensure_third_party_allowed("Installing plugins from GitHub repositories")?,
        PluginSource::Local(_) => ctx
            .policy
            .ensure_third_party_allowed("Installing local plugin packages")?,
        PluginSource::Official { .. } => {}
    }

    let permission = match &source {
        PluginSource::Local(_) => None,
        _ => Some(NetworkPermission::require(
            options.allow_network,
            "This command needs release metadata and a verified plugin package from GitHub.",
            &mut ctx.interaction,
            &ctx.policy,
        )?),
    };
    let mut official_ids: BTreeSet<String> =
        KNOWN_OFFICIAL_IDS.iter().map(|id| id.to_string()).collect();
    official_ids.extend(
        inventory
            .registry
            .plugins
            .iter()
            .filter(|(_, entry)| entry.is_official())
            .map(|(id, _)| id.clone()),
    );
    let resolved = match (&source, permission.as_ref()) {
        (PluginSource::Official { id, version }, Some(permission)) => {
            let catalog = fetch_host_catalog(ctx, permission)?;
            official_ids.extend(catalog.manifest.plugins.keys().cloned());
            resolve_official(ctx, permission, &catalog, id, version.as_ref())?
        }
        (PluginSource::Github { repository, tag }, Some(permission)) => resolve_github_release(
            ctx,
            permission,
            repository,
            tag.as_deref(),
            options.id.as_deref(),
        )?,
        (PluginSource::Local(path), _) => {
            let resolved = resolve_local(path)?;
            if let Some(expected) = &options.id
                && expected != &resolved.id
            {
                bail!(
                    "{} contains plugin '{}', not '{expected}'",
                    path.display(),
                    resolved.id
                );
            }
            resolved
        }
        (_, None) => unreachable!("remote sources always have a network permission"),
    };
    if options.require_existing && inventory.with_id(&resolved.id).is_empty() {
        bail!("plugin '{}' is not installed", resolved.id);
    }
    if let Some(environment) = &resolved.environment
        && let Compatibility::Incompatible(reason) =
            compat::check(environment, None, &ctx.host, &compat::host_api_version())
    {
        bail!(
            "Plugin {} {} {reason}.{}",
            resolved.id,
            resolved.version,
            if compat::environment_matches(environment, &next_major(&ctx.host)) {
                "\n\nUpdate eov first:\n    eov update"
            } else {
                ""
            }
        );
    }

    let plan = plan_install(&inventory, resolved, &official_ids, options);
    print_install_plan(ctx, &plan);
    if options.dry_run {
        if ctx.json {
            print_json(&InstallReport {
                plan: &plan,
                installed: None,
            })?;
        }
        return Ok(());
    }
    if plan.package.provenance.kind != SOURCE_OFFICIAL && ctx.plain() {
        eprintln!(
            "This is a third-party plugin.\n\nEOV plugins contain native executable code and run with the permissions of EOV itself. Install plugins only from sources you trust.\n"
        );
    }
    if plan.needs_override && !options.force {
        if !ctx.interaction.interactive() {
            bail!(
                "{}\nRe-run with --force to proceed.",
                plan.warnings.join("\n")
            );
        }
        if !ctx
            .interaction
            .consent(&format!("{}\nReplace it anyway?", plan.warnings.join("\n")))
        {
            bail!("Operation cancelled.");
        }
    }
    ctx.interaction.confirm("Install this plugin?")?;

    let _lock = PluginDirLock::acquire(&ctx.plugin_dir)?;
    let staged = stage_package(ctx, permission.as_ref(), &plan.package, &ctx.host)?;
    let activated = activate_package(&ctx.plugin_dir, staged, &plan.package.provenance, None)?;
    if ctx.json {
        print_json(&InstallReport {
            plan: &plan,
            installed: Some(&activated),
        })?;
    } else {
        println!("Installed {} {}.", plan.package.name, activated.version);
    }
    Ok(())
}

pub(crate) fn next_major(version: &Version) -> Version {
    if version.major == 0 {
        Version::new(0, version.minor + 1, 0)
    } else {
        Version::new(version.major + 1, 0, 0)
    }
}

fn plan_install(
    inventory: &Inventory,
    package: ResolvedPackage,
    official_ids: &BTreeSet<String>,
    options: &InstallOptions,
) -> InstallPlan {
    let installed = inventory.representative(&package.id);
    let action = match installed {
        None => "install",
        Some(installed) => {
            match compare_versions(&package.version.to_string(), &installed.version) {
                std::cmp::Ordering::Greater => "upgrade",
                std::cmp::Ordering::Less => "downgrade",
                std::cmp::Ordering::Equal => "reinstall",
            }
        }
    };
    let mut warnings = Vec::new();
    let mut needs_override = false;
    let third_party = package.provenance.kind != SOURCE_OFFICIAL;
    if third_party && official_ids.contains(&package.id) {
        warnings.push(format!(
            "'{}' is the id of an official eov plugin. This package ({}) would {} it.",
            package.id,
            package.provenance.describe(),
            if installed.is_some() {
                "replace"
            } else {
                "shadow"
            }
        ));
        needs_override = true;
    }
    if let Some(installed) = installed
        && !installed.provenance.same_origin(&package.provenance)
        && installed.provenance.kind != "unmanaged"
    {
        warnings.push(format!(
            "This replaces the installed {} {} from {} with a package from {}.",
            installed.id,
            installed.version,
            installed.provenance.describe(),
            package.provenance.describe()
        ));
        needs_override = true;
    }
    if !package.pinned_by_catalog && package.provenance.kind == SOURCE_OFFICIAL {
        warnings.push(
            "This version is not pinned by the eov release catalog; its SHA-256 comes from the plugin repository's own release.toml.".to_string(),
        );
    }
    InstallPlan {
        installed_version: installed.map(|package| package.version.clone()),
        installed_provenance: installed.map(|package| package.provenance.clone()),
        package,
        action,
        warnings,
        needs_override,
        dry_run: options.dry_run,
    }
}

fn print_install_plan(ctx: &PluginContext, plan: &InstallPlan) {
    if !ctx.plain() {
        return;
    }
    let package = &plan.package;
    println!("Plugin:        {} ({})", package.name, package.id);
    println!("Source:        {}", package.provenance.describe());
    if let Some(repository) = &package.provenance.repository {
        println!("Repository:    {repository}");
    }
    if let Some(tag) = &package.provenance.release_tag {
        println!("Release:       {tag}");
    }
    println!(
        "Installed:     {}",
        match (&plan.installed_version, &plan.installed_provenance) {
            (Some(version), Some(provenance)) => format!("{version} ({})", provenance.describe()),
            _ => "-".to_string(),
        }
    );
    println!("Target:        {}", package.version);
    println!("Action:        {}", plan.action);
    println!(
        "Requires eov:  {}",
        package.environment.as_deref().unwrap_or("unknown")
    );
    println!("Artifact:      {}", package.fetch.display());
    println!(
        "SHA-256:       {}{}",
        package.sha256,
        if package.pinned_by_catalog {
            " (pinned by the eov release catalog)"
        } else {
            ""
        }
    );
    if let Some(signature) = &package.signature {
        println!("Signature:     {}", signature.describe());
    }
    for warning in &plan.warnings {
        println!("Warning:       {warning}");
    }
    if plan.dry_run {
        println!("\nDry run: nothing was changed.");
    }
}

fn prepare_update_source(source: PluginSource, inventory: &Inventory) -> Result<PluginSource> {
    match source {
        PluginSource::Official { id, version } => {
            if inventory.with_id(&id).is_empty() {
                bail!("plugin '{id}' is not installed");
            }
            if version.is_some() {
                return Ok(PluginSource::Official { id, version });
            }
            match inventory.registry.plugins.get(&id) {
                Some(entry) if entry.source == SOURCE_GITHUB => Ok(PluginSource::Github {
                    repository: entry
                        .repository
                        .clone()
                        .context("managed GitHub plugin has no repository")?,
                    tag: None,
                }),
                Some(entry) if entry.source == SOURCE_LOCAL => bail!(
                    "Plugin '{id}' was installed from a local package and has no remote update source.\n\nInstall a new package explicitly:\n    eov plugin install /path/to/{id}.eop"
                ),
                _ => Ok(PluginSource::Official { id, version }),
            }
        }
        PluginSource::Github { repository, tag } => {
            let known = inventory.registry.plugins.values().any(|entry| {
                entry.source == SOURCE_GITHUB
                    && entry
                        .repository
                        .as_deref()
                        .is_some_and(|known| same_repository(known, &repository))
            });
            if !known && tag.is_none() {
                bail!("no installed plugin is associated with GitHub repository {repository}");
            }
            Ok(PluginSource::Github { repository, tag })
        }
        PluginSource::Local(path) => bail!(
            "plugin update does not accept a local package as a remote source: {}",
            path.display()
        ),
    }
}

/// One row of `eov plugin ls`.
#[derive(Debug, Clone, Serialize)]
pub struct PluginListing {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available: Option<String>,
    pub source: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    pub disabled: bool,
}

#[derive(Debug, Serialize)]
struct ListReport<'a> {
    plugins: &'a [PluginListing],
    broken: &'a [BrokenPackage],
    catalog_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog_signature: Option<SignatureStatus>,
}

/// Builds the `eov plugin ls` rows (library API for a plugins panel).
pub fn list_plugins(
    inventory: &Inventory,
    catalog: Option<&EovReleaseManifest>,
) -> Vec<PluginListing> {
    let mut ids = inventory.ids();
    if let Some(catalog) = catalog {
        ids.extend(catalog.plugins.keys().cloned());
    }
    ids.into_iter()
        .map(|id| {
            let package = inventory.representative(&id);
            let official = catalog.and_then(|catalog| catalog.plugins.get(&id));
            let duplicates = inventory.with_id(&id).len();
            let status = match (package, official) {
                (None, Some(_)) => "not installed".to_string(),
                (None, None) => "-".to_string(),
                (Some(package), _) if package.disabled => "disabled".to_string(),
                (Some(package), _) if !package.compatible => format!(
                    "incompatible: {}",
                    package
                        .incompatibility
                        .as_deref()
                        .unwrap_or("unknown reason")
                ),
                (Some(package), Some(entry)) if package.provenance.kind == SOURCE_OFFICIAL => {
                    match compare_versions(&package.version, &entry.version) {
                        std::cmp::Ordering::Less => "update available".to_string(),
                        std::cmp::Ordering::Equal => "current".to_string(),
                        std::cmp::Ordering::Greater => "newer than the catalog".to_string(),
                    }
                }
                (Some(package), _) if package.modified => "modified since installation".to_string(),
                (Some(_), _) => "installed".to_string(),
            };
            let status = if duplicates > 1 {
                format!("{status} ({duplicates} packages; the newest compatible one loads)")
            } else {
                status
            };
            PluginListing {
                name: official
                    .and_then(|entry| entry.name.clone())
                    .or_else(|| package.map(|package| package.name.clone()))
                    .unwrap_or_else(|| title_case_id(&id)),
                installed: package.map(|package| package.version.clone()),
                available: official.map(|entry| entry.version.clone()),
                source: package
                    .map(|package| package.provenance.describe())
                    .unwrap_or_else(|| "official catalog".to_string()),
                status,
                description: package
                    .and_then(|package| package.description.clone())
                    .or_else(|| official.and_then(|entry| entry.description.clone())),
                repository: official
                    .map(|entry| entry.repository.clone())
                    .or_else(|| package.and_then(|package| package.provenance.repository.clone())),
                disabled: package.is_some_and(|package| package.disabled),
                id,
            }
        })
        .collect()
}

/// `eov plugin ls`.
pub fn list(ctx: &mut PluginContext, allow_network: bool, offline: bool) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &ctx.system_plugin_dirs, &ctx.host)?;
    let mut catalog = None;
    if !offline && ctx.policy.allow_network && ctx.policy.error.is_none() {
        // Listing must work without a network: only ask when someone can
        // answer, and fall back to the local inventory otherwise.
        let permission = if allow_network || ctx.interaction.interactive() {
            NetworkPermission::acquire(
                allow_network,
                "This command compares installed plugins with the official eov plugin catalog.",
                &mut ctx.interaction,
                &ctx.policy,
            )?
        } else {
            None
        };
        match permission {
            Some(permission) => match fetch_host_catalog(ctx, &permission) {
                Ok(value) => catalog = Some(value),
                Err(error) => eprintln!("warning: {error:#}"),
            },
            None => eprintln!(
                "Showing installed plugins only (pass --allow-network to compare with the official catalog)."
            ),
        }
    }
    let rows = list_plugins(
        &inventory,
        catalog.as_ref().map(|catalog| &catalog.manifest),
    );
    if ctx.json {
        return print_json(&ListReport {
            plugins: &rows,
            broken: &inventory.broken,
            catalog_version: catalog
                .as_ref()
                .map(|catalog| catalog.manifest.version.to_string()),
            catalog_signature: catalog.as_ref().map(|catalog| catalog.signature.clone()),
        });
    }
    if rows.is_empty() && inventory.broken.is_empty() {
        println!("No plugins are installed in {}.", ctx.plugin_dir.display());
        return Ok(());
    }
    let table: Vec<[String; 6]> = rows
        .iter()
        .map(|row| {
            [
                row.id.clone(),
                row.name.clone(),
                row.installed.clone().unwrap_or_else(|| "-".into()),
                row.available.clone().unwrap_or_else(|| "-".into()),
                row.source.clone(),
                row.status.clone(),
            ]
        })
        .collect();
    print_table(
        &["ID", "Name", "Installed", "Available", "Source", "Status"],
        &table,
    );
    for broken in &inventory.broken {
        println!(
            "\nBroken package {}: {}\n  Remove it with: eov plugin remove {}",
            broken.package_path.display(),
            broken.error,
            broken.package_path.display()
        );
    }
    Ok(())
}

fn print_table<const N: usize>(headers: &[&str; N], rows: &[[String; N]]) {
    let mut widths = headers.map(str::len);
    for row in rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }
    let render = |cells: &[String]| {
        cells
            .iter()
            .zip(widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_string()
    };
    println!("{}", render(&headers.map(str::to_string)));
    for row in rows {
        println!("{}", render(row));
    }
}

fn find_matches<'a>(inventory: &'a Inventory, query: &str) -> Vec<&'a InstalledPackage> {
    let by_id: Vec<_> = inventory.with_id(query);
    if !by_id.is_empty() {
        return by_id;
    }
    let query_path = Path::new(query);
    inventory
        .packages
        .iter()
        .filter(|package| {
            package.name.eq_ignore_ascii_case(query)
                || package.package_path == query_path
                || package.package_path.file_name() == Some(OsStr::new(query))
        })
        .collect()
}

fn find_broken<'a>(inventory: &'a Inventory, query: &str) -> Vec<&'a BrokenPackage> {
    let query_path = Path::new(query);
    let canonical = query_path.canonicalize().ok();
    inventory
        .broken
        .iter()
        .filter(|broken| {
            broken.package_path == query_path
                || broken.package_path.file_name() == Some(OsStr::new(query))
                || broken.package_path.file_stem() == Some(OsStr::new(query))
                || canonical.as_ref().is_some_and(|canonical| {
                    broken.package_path.canonicalize().ok().as_ref() == Some(canonical)
                })
        })
        .collect()
}

/// `eov plugin info`.
pub fn info(ctx: &PluginContext, query: &str, format: OutputFormat) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &ctx.system_plugin_dirs, &ctx.host)?;
    let matches = find_matches(&inventory, query);
    let broken = find_broken(&inventory, query);
    if matches.is_empty() && broken.is_empty() {
        bail!("no installed plugin matches '{query}'");
    }
    let ids: BTreeSet<_> = matches.iter().map(|package| package.id.as_str()).collect();
    match format {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct InfoReport<'a> {
                packages: Vec<&'a InstalledPackage>,
                broken: Vec<&'a BrokenPackage>,
            }
            print_json(&InfoReport {
                packages: matches,
                broken,
            })
        }
        OutputFormat::Toml => {
            if ids.len() > 1 {
                bail!(
                    "'{query}' matches several plugins ({}); use the plugin ID",
                    ids.into_iter().collect::<Vec<_>>().join(", ")
                );
            }
            let Some(package) = matches
                .iter()
                .find(|package| package.selected)
                .or_else(|| matches.first())
            else {
                bail!(
                    "{} cannot be parsed: {}",
                    broken[0].package_path.display(),
                    broken[0].error
                );
            };
            if matches.len() > 1 {
                eprintln!(
                    "note: {} packages declare '{}'; showing {}, the one eov loads",
                    matches.len(),
                    package.id,
                    package.package_path.display()
                );
            }
            print!("{}", package.raw_manifest);
            Ok(())
        }
        OutputFormat::Plain => {
            for (index, package) in matches.iter().enumerate() {
                if index > 0 {
                    println!();
                }
                print_package_info(
                    package,
                    matches.len() > 1,
                    inventory.registry.plugins.get(&package.id),
                );
            }
            for broken in broken {
                println!("\nPackage:  {}", broken.package_path.display());
                println!("Status:   broken: {}", broken.error);
                println!(
                    "Remove:   eov plugin remove {}",
                    broken.package_path.display()
                );
            }
            Ok(())
        }
    }
}

fn print_package_info(
    package: &InstalledPackage,
    duplicate: bool,
    managed: Option<&ManagedPlugin>,
) {
    println!("Name:              {}", package.name);
    println!("ID:                {}", package.id);
    println!("Version:           {}", package.version);
    if let Some(description) = &package.description {
        println!("Description:       {description}");
    }
    println!("EOV requirement:   {}", package.environment);
    if let Some(api) = compat::api_requirement(&package.raw_manifest) {
        println!("Plugin API:        {api}");
    }
    println!(
        "Compatible:        {}",
        match &package.incompatibility {
            None => "yes".to_string(),
            Some(reason) => format!("no ({reason})"),
        }
    );
    println!(
        "Enabled:           {}",
        if package.disabled {
            "no (disabled)"
        } else {
            "yes"
        }
    );
    println!("Package path:      {}", package.package_path.display());
    println!("Package SHA-256:   {}", package.sha256);
    println!("Source:            {}", package.provenance.describe());
    if let Some(repository) = &package.provenance.repository {
        println!("Repository:        {repository}");
    }
    if let Some(tag) = &package.provenance.release_tag {
        println!("Release tag:       {tag}");
    }
    if let Some(managed) = managed
        && package.provenance.kind != "unmanaged"
    {
        println!("Installed SHA-256: {}", managed.sha256);
    }
    if package.modified {
        println!("Warning:           the package changed since eov installed it");
    }
    if duplicate {
        println!(
            "Duplicate ID:      yes; {}",
            if package.selected {
                "this package is the one eov loads"
            } else {
                "eov loads another package with this ID"
            }
        );
    }
}

/// Options of `eov plugin remove`.
#[derive(Debug, Clone, Default)]
pub struct RemoveOptions {
    pub dry_run: bool,
}

/// `eov plugin remove`: removes every package of a plugin (or a broken
/// package, by path or file name) and its registry record.
pub fn remove(ctx: &mut PluginContext, query: &str, options: &RemoveOptions) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host)?;
    let matches = find_matches(&inventory, query);
    let ids: BTreeSet<String> = matches.iter().map(|package| package.id.clone()).collect();
    if ids.len() > 1 {
        bail!(
            "'{query}' matches several plugins ({}); use the plugin ID",
            ids.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    let id = ids.into_iter().next();
    let mut paths: Vec<PathBuf> = match &id {
        Some(id) => inventory
            .with_id(id)
            .into_iter()
            .map(|package| package.package_path.clone())
            .collect(),
        None => Vec::new(),
    };
    paths.extend(
        find_broken(&inventory, query)
            .into_iter()
            .map(|broken| broken.package_path.clone()),
    );
    if paths.is_empty() {
        bail!("no installed plugin matches '{query}'");
    }

    #[derive(Serialize)]
    struct RemoveReport<'a> {
        id: Option<&'a str>,
        packages: &'a [PathBuf],
        dry_run: bool,
        removed: bool,
    }
    let mut description = String::new();
    if let Some(id) = &id {
        description.push_str(&format!("Plugin:   {id}\n"));
    }
    for path in &paths {
        description.push_str(&format!("Package:  {}\n", path.display()));
    }
    let description = description.trim_end().to_string();
    if options.dry_run {
        if ctx.json {
            return print_json(&RemoveReport {
                id: id.as_deref(),
                packages: &paths,
                dry_run: true,
                removed: false,
            });
        }
        println!("{description}\n\nDry run: nothing was removed.");
        return Ok(());
    }
    ctx.interaction.confirm(&description)?;

    let _lock = PluginDirLock::acquire(&ctx.plugin_dir)?;
    let mut registry = registry::load(&ctx.plugin_dir)?;
    let backups = move_to_backups(&paths)?;
    if let Some(id) = &id {
        registry.plugins.remove(id);
    }
    if let Err(error) = registry::save(&ctx.plugin_dir, &registry) {
        restore_backups(&backups)?;
        return Err(error);
    }
    finish_backups(backups, None);
    for path in &paths {
        let _ = invalidate_plugin_package_cache(path);
    }
    if ctx.json {
        return print_json(&RemoveReport {
            id: id.as_deref(),
            packages: &paths,
            dry_run: false,
            removed: true,
        });
    }
    match id {
        Some(id) => println!("Removed plugin '{id}'."),
        None => println!("Removed {} broken package(s).", paths.len()),
    }
    Ok(())
}

/// Enables or disables a plugin without uninstalling it. Returns whether the
/// state changed. Library API for the plugin host and a plugins panel.
pub fn set_plugin_enabled(plugin_dir: &Path, id: &str, enabled: bool) -> Result<bool> {
    compat::validate_plugin_id(id)?;
    let _lock = PluginDirLock::acquire(plugin_dir)?;
    let mut registry = registry::load(plugin_dir)?;
    let changed = if enabled {
        registry.disabled.remove(id)
    } else {
        registry.disabled.insert(id.to_string())
    };
    if changed {
        registry::save(plugin_dir, &registry)?;
    }
    Ok(changed)
}

/// `eov plugin enable` / `eov plugin disable`.
pub fn set_enabled(ctx: &PluginContext, id: &str, enabled: bool) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &ctx.system_plugin_dirs, &ctx.host)?;
    let installed = !inventory.with_id(id).is_empty();
    let changed = set_plugin_enabled(&ctx.plugin_dir, id, enabled)?;
    if ctx.json {
        #[derive(Serialize)]
        struct EnableReport<'a> {
            id: &'a str,
            enabled: bool,
            changed: bool,
            installed: bool,
        }
        return print_json(&EnableReport {
            id,
            enabled,
            changed,
            installed,
        });
    }
    let state = if enabled { "enabled" } else { "disabled" };
    if !changed {
        println!("Plugin '{id}' is already {state}.");
    } else {
        println!("Plugin '{id}' is now {state}. The change takes effect the next time eov starts.");
    }
    if !installed {
        eprintln!(
            "note: no installed package declares '{id}'; the setting applies once one is installed"
        );
    }
    Ok(())
}

/// `eov plugin gc`: removes unused plugin package extractions now.
pub fn collect_garbage(ctx: &PluginContext) -> Result<()> {
    let report = package_cache::PackageCache::default()
        .collect_garbage(package_cache::DEFAULT_GC_GRACE)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    if ctx.json {
        #[derive(Serialize)]
        struct GcJson<'a> {
            removed: &'a [PathBuf],
            kept_in_use: usize,
            kept_referenced: usize,
            kept_recent: usize,
        }
        return print_json(&GcJson {
            removed: &report.removed,
            kept_in_use: report.kept_in_use,
            kept_referenced: report.kept_referenced,
            kept_recent: report.kept_recent,
        });
    }
    println!(
        "Removed {} unused extraction(s); kept {} in use, {} referenced and {} recent.",
        report.removed.len(),
        report.kept_in_use,
        report.kept_referenced,
        report.kept_recent
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Lockfiles
// ---------------------------------------------------------------------------

pub const LOCKFILE_SCHEMA: u32 = 1;

/// A reproducible description of a workstation's plugins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lockfile {
    pub schema: u32,
    /// The eov version the lockfile was created with (informational).
    pub eov: String,
    /// The platform of the recorded package hashes, e.g. `linux.x86_64`.
    pub platform: String,
    #[serde(default, rename = "plugins")]
    pub entries: Vec<LockEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockEntry {
    pub id: String,
    pub version: String,
    /// `official`, `github`, `local` or `unmanaged`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_tag: Option<String>,
    pub sha256: String,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
}

fn enabled_by_default() -> bool {
    true
}

/// Builds the lockfile for the packages eov currently loads (or, for ids
/// with no compatible package, their first package).
pub fn lockfile(inventory: &Inventory, host: &Version, platform: &Platform) -> Lockfile {
    let mut entries: Vec<LockEntry> = inventory
        .ids()
        .into_iter()
        .filter_map(|id| inventory.representative(&id))
        .filter(|package| !package.system)
        .map(|package| LockEntry {
            id: package.id.clone(),
            version: package.version.clone(),
            source: package.provenance.kind.clone(),
            repository: package.provenance.repository.clone(),
            release_tag: package.provenance.release_tag.clone(),
            sha256: package.sha256.clone(),
            enabled: !package.disabled,
        })
        .collect();
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    Lockfile {
        schema: LOCKFILE_SCHEMA,
        eov: host.to_string(),
        platform: platform.short_name(),
        entries,
    }
}

/// `eov plugin lock`: prints (or writes) the lockfile.
pub fn lock(ctx: &PluginContext, output: Option<&Path>) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host)?;
    let lockfile = lockfile(&inventory, &ctx.host, &ctx.platform);
    let text = format!(
        "# eov plugin lockfile, created by eov {}.\n# Apply it on another {} workstation with: eov plugin sync <this file>\n{}",
        ctx.host,
        ctx.platform,
        toml::to_string_pretty(&lockfile)?
    );
    match output {
        Some(path) => {
            let directory = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let mut file = tempfile::NamedTempFile::new_in(directory)?;
            std::io::Write::write_all(&mut file, text.as_bytes())?;
            file.persist(path).map_err(|error| error.error)?;
            if ctx.plain() {
                eprintln!(
                    "Wrote {} plugin(s) to {}.",
                    lockfile.entries.len(),
                    path.display()
                );
            }
        }
        None if ctx.json => print_json(&lockfile)?,
        None => print!("{text}"),
    }
    Ok(())
}

/// Options of `eov plugin sync`.
#[derive(Debug, Clone, Default)]
pub struct SyncOptions {
    pub allow_network: bool,
    pub dry_run: bool,
    /// Remove installed plugins that the lockfile does not list.
    pub prune: bool,
    /// Directory with `.eop` packages to take matching packages from instead
    /// of downloading them (air-gapped workstations).
    pub from: Option<PathBuf>,
}

/// What `sync` does for one plugin.
#[derive(Debug, Clone, Serialize)]
pub struct SyncStep {
    pub id: String,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

enum SyncWork<'a> {
    Keep,
    Install(&'a LockEntry),
    Remove(String),
}

/// `eov plugin sync <lockfile>`.
pub fn sync(ctx: &mut PluginContext, lockfile_path: &Path, options: &SyncOptions) -> Result<()> {
    ctx.policy.ensure_valid()?;
    let text = fs::read_to_string(lockfile_path)
        .with_context(|| format!("could not read {}", lockfile_path.display()))?;
    let lock: Lockfile = toml::from_str(&text).with_context(|| {
        format!(
            "{} is not a valid eov plugin lockfile",
            lockfile_path.display()
        )
    })?;
    if lock.schema != LOCKFILE_SCHEMA {
        bail!("unsupported plugin lockfile schema {}", lock.schema);
    }
    if lock.platform != ctx.platform.short_name() {
        bail!(
            "{} records {} packages; this workstation is {}. Create a lockfile on a {} workstation.",
            lockfile_path.display(),
            lock.platform,
            ctx.platform,
            ctx.platform
        );
    }
    let mut seen = BTreeSet::new();
    for entry in &lock.entries {
        compat::validate_plugin_id(&entry.id)?;
        if !seen.insert(entry.id.clone()) {
            bail!(
                "{} lists plugin '{}' twice",
                lockfile_path.display(),
                entry.id
            );
        }
        if entry.source != SOURCE_OFFICIAL {
            ctx.policy
                .ensure_third_party_allowed(&format!("Plugin '{}' ({})", entry.id, entry.source))?;
        }
    }
    let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host)?;
    let local_packages = match &options.from {
        Some(directory) => index_packages_by_hash(directory)?,
        None => BTreeMap::new(),
    };

    let mut plan: Vec<(SyncStep, SyncWork<'_>)> = Vec::new();
    for entry in &lock.entries {
        let installed = inventory.selected(&entry.id);
        let in_sync = installed
            .is_some_and(|package| package.sha256.eq_ignore_ascii_case(&entry.sha256))
            && inventory.with_id(&entry.id).len() == 1;
        let step = SyncStep {
            id: entry.id.clone(),
            action: if in_sync {
                "keep".into()
            } else if installed.is_some() {
                "replace".into()
            } else {
                "install".into()
            },
            detail: Some(format!("{} ({})", entry.version, entry.source)),
            error: None,
        };
        let work = if in_sync {
            SyncWork::Keep
        } else {
            SyncWork::Install(entry)
        };
        plan.push((step, work));
    }
    if options.prune {
        for id in inventory.ids() {
            if !seen.contains(&id) {
                plan.push((
                    SyncStep {
                        id: id.clone(),
                        action: "remove".into(),
                        detail: None,
                        error: None,
                    },
                    SyncWork::Remove(id),
                ));
            }
        }
    }
    let needs_network = plan.iter().any(|(_, work)| {
        matches!(work, SyncWork::Install(entry) if !local_packages.contains_key(&entry.sha256.to_ascii_lowercase()))
    });

    if ctx.plain() {
        println!(
            "Plugin lockfile {} ({} plugin(s))",
            lockfile_path.display(),
            lock.entries.len()
        );
        for (step, _) in &plan {
            println!(
                "  {:<16} {:<8} {}",
                step.id,
                step.action,
                step.detail.as_deref().unwrap_or("")
            );
        }
    }
    let changes = plan.iter().any(|(_, work)| !matches!(work, SyncWork::Keep));
    let enable_changes = lock
        .entries
        .iter()
        .any(|entry| inventory.registry.is_disabled(&entry.id) == entry.enabled);
    if options.dry_run || (!changes && !enable_changes) {
        if ctx.json {
            let steps: Vec<_> = plan.iter().map(|(step, _)| step).collect();
            print_json(&steps)?;
        } else if options.dry_run {
            println!("\nDry run: nothing was changed.");
        } else {
            println!("\nAll plugins already match the lockfile.");
        }
        return Ok(());
    }
    ctx.interaction.confirm("Apply the plugin lockfile?")?;
    let permission = if needs_network {
        Some(NetworkPermission::require(
            options.allow_network,
            "Some locked plugin packages must be downloaded from GitHub.",
            &mut ctx.interaction,
            &ctx.policy,
        )?)
    } else {
        None
    };

    let _lock = PluginDirLock::acquire(&ctx.plugin_dir)?;
    let mut failures = 0;
    for (step, work) in &mut plan {
        let result = match work {
            SyncWork::Keep => Ok(()),
            SyncWork::Remove(id) => remove_locked(ctx, id),
            SyncWork::Install(entry) => {
                sync_entry(ctx, permission.as_ref(), entry, &local_packages).map(|_| ())
            }
        };
        if let Err(error) = result {
            failures += 1;
            step.error = Some(format!("{error:#}"));
        }
    }
    let mut registry = registry::load(&ctx.plugin_dir)?;
    for entry in &lock.entries {
        if entry.enabled {
            registry.disabled.remove(&entry.id);
        } else {
            registry.disabled.insert(entry.id.clone());
        }
    }
    registry::save(&ctx.plugin_dir, &registry)?;

    if ctx.json {
        let steps: Vec<_> = plan.iter().map(|(step, _)| step).collect();
        print_json(&steps)?;
    } else {
        for (step, _) in &plan {
            if let Some(error) = &step.error {
                eprintln!("error: {}: {error}", step.id);
            }
        }
    }
    if failures > 0 {
        bail!("{failures} plugin(s) could not be synchronised with the lockfile");
    }
    if ctx.plain() {
        println!("The installed plugins match the lockfile.");
    }
    Ok(())
}

fn index_packages_by_hash(directory: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut packages = BTreeMap::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in
            fs::read_dir(&dir).with_context(|| format!("could not read {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension() == Some(OsStr::new("eop")) {
                let sha256 = package_cache::PackageCache::default()
                    .sha256(&path)
                    .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
                packages.insert(sha256, path);
            }
        }
    }
    Ok(packages)
}

fn sync_entry(
    ctx: &PluginContext,
    permission: Option<&NetworkPermission>,
    entry: &LockEntry,
    local_packages: &BTreeMap<String, PathBuf>,
) -> Result<Activated> {
    let expected = entry.sha256.to_ascii_lowercase();
    let mut resolved = if let Some(path) = local_packages.get(&expected) {
        resolve_local(path)?
    } else {
        let permission = permission.context("network access is required")?;
        let unfetchable = || {
            anyhow::anyhow!(
                "plugin '{}' is a {} package; pass --from <dir> with a package whose SHA-256 is {expected}",
                entry.id,
                entry.source
            )
        };
        let repository = entry.repository.as_deref().ok_or_else(unfetchable)?;
        match entry.source.as_str() {
            SOURCE_OFFICIAL | SOURCE_GITHUB => resolve_github_release(
                ctx,
                permission,
                repository,
                entry.release_tag.as_deref(),
                Some(&entry.id),
            )?,
            _ => return Err(unfetchable()),
        }
    };
    if resolved.id != entry.id {
        bail!(
            "the package contains plugin '{}', not '{}'",
            resolved.id,
            entry.id
        );
    }
    if !resolved.sha256.eq_ignore_ascii_case(&expected) {
        bail!(
            "the published package has SHA-256 {}, but the lockfile pins {expected}",
            resolved.sha256
        );
    }
    // Keep the provenance the lockfile records.
    resolved.provenance = Provenance {
        kind: if entry.source == "unmanaged" {
            SOURCE_LOCAL.to_string()
        } else {
            entry.source.clone()
        },
        repository: entry.repository.clone(),
        release_tag: entry.release_tag.clone(),
    };
    let staged = stage_package(ctx, permission, &resolved, &ctx.host)?;
    activate_package(&ctx.plugin_dir, staged, &resolved.provenance, None)
}

fn remove_locked(ctx: &PluginContext, id: &str) -> Result<()> {
    let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host)?;
    let paths: Vec<_> = inventory
        .with_id(id)
        .into_iter()
        .map(|package| package.package_path.clone())
        .collect();
    let mut registry = registry::load(&ctx.plugin_dir)?;
    let backups = move_to_backups(&paths)?;
    registry.plugins.remove(id);
    if let Err(error) = registry::save(&ctx.plugin_dir, &registry) {
        restore_backups(&backups)?;
        return Err(error);
    }
    finish_backups(backups, None);
    Ok(())
}

pub(crate) fn print_json<T: Serialize + ?Sized>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::prompt::ScriptedPrompter;
    use super::super::signature::test_keys::TEST_KEY;
    use super::super::transport::MockTransport;
    use super::*;
    use crate::plugins::discovery::package_cache::test_packages::{
        manifest, write_archive, write_package,
    };
    use sha2::{Digest, Sha256};
    use std::sync::Arc;

    pub(crate) const HOST: &str = "1.0.0";
    pub(crate) const GAMEPAD_REPO: &str = "https://github.com/eosin-platform/eov-gamepad-plugin";

    pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    pub(crate) fn context(
        root: &Path,
        transport: Arc<MockTransport>,
        prompter: ScriptedPrompter,
        assume_yes: bool,
    ) -> PluginContext {
        PluginContext {
            plugin_dir: root.join("plugins"),
            system_plugin_dirs: Vec::new(),
            host: Version::parse(HOST).unwrap(),
            platform: Platform::linux_x86_64(),
            policy: Policy::default(),
            verifier: Verifier::new(&[], None, false).unwrap(),
            interaction: Interaction::new(assume_yes, Box::new(prompter)),
            client: ReleaseClient::with_transport(transport),
            json: false,
        }
    }

    /// An eov catalog for `eov_version` pinning one gamepad package.
    pub(crate) fn catalog_text(
        eov_version: &str,
        plugin_version: &str,
        package_sha: &str,
    ) -> String {
        format!(
            r#"[manifest]
schema = 1
kind = "eov"
version = "{eov_version}"
repository = "https://github.com/eosin-platform/eov"

[platform.linux.appimage.x86_64]
version = "{eov_version}"
sha256 = "{app}"
url = "https://github.com/eosin-platform/eov/releases/download/v{eov_version}/eov-v{eov_version}-linux-x86_64.AppImage"

[plugins.gamepad]
name = "Gamepad"
repository = "{GAMEPAD_REPO}"
version = "{plugin_version}"
release_tag = "v{plugin_version}"
environment = ">=1.0.0, <2.0.0"

[plugins.gamepad.platform.linux.x86_64]
version = "{plugin_version}"
sha256 = "{package_sha}"
url = "{GAMEPAD_REPO}/releases/download/v{plugin_version}/gamepad-v{plugin_version}-linux-x86_64.eop"
"#,
            app = "a".repeat(64)
        )
    }

    pub(crate) fn catalog_url(version: &str) -> String {
        format!("https://github.com/eosin-platform/eov/releases/download/v{version}/release.toml")
    }

    pub(crate) fn gamepad_url(version: &str) -> String {
        format!("{GAMEPAD_REPO}/releases/download/v{version}/gamepad-v{version}-linux-x86_64.eop")
    }

    pub(crate) fn package_bytes(root: &Path, id: &str, version: &str) -> Vec<u8> {
        let path = root.join(format!("source-{id}-{version}.eop"));
        write_package(&path, id, version);
        fs::read(path).unwrap()
    }

    /// Serves the eov `eov_version` catalog and the gamepad package it pins.
    pub(crate) fn serve_gamepad(
        transport: &MockTransport,
        root: &Path,
        eov_version: &str,
        plugin_version: &str,
    ) -> Vec<u8> {
        let bytes = package_bytes(root, "gamepad", plugin_version);
        transport.respond(
            &catalog_url(eov_version),
            200,
            catalog_text(eov_version, plugin_version, &sha256_hex(&bytes)),
        );
        transport.respond(&gamepad_url(plugin_version), 200, bytes.clone());
        bytes
    }

    fn serve_official_gamepad(transport: &MockTransport, root: &Path, version: &str) -> Vec<u8> {
        serve_gamepad(transport, root, HOST, version)
    }

    fn options() -> InstallOptions {
        InstallOptions {
            allow_network: true,
            ..InstallOptions::default()
        }
    }

    pub(crate) fn plugin_dir_entries(plugin_dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(plugin_dir)
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                    .filter(|name| name != registry::LOCK_FILENAME)
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    #[test]
    fn parses_official_github_and_local_sources() {
        assert!(matches!(
            parse_source("gamepad@v0.2.2").unwrap(),
            PluginSource::Official {
                version: Some(_),
                ..
            }
        ));
        assert_eq!(
            parse_source("github.com/foo/bar@plugin/v1.0").unwrap(),
            PluginSource::Github {
                repository: "https://github.com/foo/bar".into(),
                tag: Some("plugin/v1.0".into())
            }
        );
        assert!(matches!(
            parse_source("./foo.eop").unwrap(),
            PluginSource::Local(_)
        ));
        assert!(matches!(
            parse_source(".\\foo.eop").unwrap(),
            PluginSource::Local(_)
        ));
        assert!(parse_source("C:evil").is_err());
        assert!(parse_source("a:b").is_err());
    }

    #[test]
    fn local_install_is_offline_and_uses_canonical_filename() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let source = temp.path().join("local.eop");
        write_package(&source, "example", "1.0.0");
        install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        assert!(transport.requested().is_empty());
        assert_eq!(
            plugin_dir_entries(&ctx.plugin_dir),
            vec![".eov-managed.toml", "example.eop"]
        );
        let registry = registry::load(&ctx.plugin_dir).unwrap();
        assert_eq!(registry.plugins["example"].source, SOURCE_LOCAL);
        assert_eq!(
            registry.plugins["example"].sha256,
            sha256_hex(&fs::read(&source).unwrap())
        );
    }

    #[test]
    fn official_install_uses_the_catalog_pinned_hash() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let bytes = serve_official_gamepad(&transport, temp.path(), "1.0.1");
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        install(&mut ctx, "gamepad", &options()).unwrap();
        let registry = registry::load(&ctx.plugin_dir).unwrap();
        let entry = &registry.plugins["gamepad"];
        assert_eq!(entry.source, SOURCE_OFFICIAL);
        assert_eq!(entry.release_tag.as_deref(), Some("v1.0.1"));
        assert_eq!(entry.sha256, sha256_hex(&bytes));
        // The plugin repository's own release.toml was never needed.
        assert!(
            !transport
                .requested()
                .iter()
                .any(|url| url.starts_with(GAMEPAD_REPO) && url.ends_with("release.toml"))
        );
    }

    #[test]
    fn sha_mismatch_changes_nothing_and_leaves_no_staging_files() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_official_gamepad(&transport, temp.path(), "1.0.1");
        // Serve different bytes than the catalog pins.
        transport.respond(
            &gamepad_url("1.0.1"),
            200,
            package_bytes(temp.path(), "gamepad", "1.0.2"),
        );
        let mut ctx = context(
            temp.path(),
            transport,
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let error = install(&mut ctx, "gamepad", &options()).unwrap_err();
        assert!(
            format!("{error:#}").contains("SHA-256 mismatch"),
            "{error:#}"
        );
        assert!(
            plugin_dir_entries(&ctx.plugin_dir).is_empty(),
            "{:?}",
            plugin_dir_entries(&ctx.plugin_dir)
        );
    }

    #[test]
    fn package_contents_must_match_the_release_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        // The catalog pins a package that actually contains a different id.
        let bytes = package_bytes(temp.path(), "notgamepad", "1.0.1");
        transport.respond(
            &catalog_url(HOST),
            200,
            catalog_text(HOST, "1.0.1", &sha256_hex(&bytes)),
        );
        transport.respond(&gamepad_url("1.0.1"), 200, bytes);
        let mut ctx = context(
            temp.path(),
            transport,
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let error = install(&mut ctx, "gamepad", &options()).unwrap_err();
        assert!(
            format!("{error:#}").contains("promised 'gamepad'"),
            "{error:#}"
        );
        assert!(plugin_dir_entries(&ctx.plugin_dir).is_empty());
    }

    #[test]
    fn signed_catalogs_are_verified_before_anything_is_downloaded() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_official_gamepad(&transport, temp.path(), "1.0.1");
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        // This build trusts a key and requires signatures from 1.0.0 on.
        ctx.verifier = Verifier::new(&[TEST_KEY], Some(Version::new(1, 0, 0)), false).unwrap();
        let error = install(&mut ctx, "gamepad", &options()).unwrap_err();
        assert!(format!("{error:#}").contains("not signed"), "{error:#}");
        assert!(
            !transport
                .requested()
                .iter()
                .any(|url| url.ends_with(".eop"))
        );
        // A signature that does not match the catalog is refused too.
        transport.respond(
            &format!("{}{SIGNATURE_SUFFIX}", catalog_url(HOST)),
            200,
            include_str!("../../tests/data/signing/release.toml.minisig"),
        );
        let error = install(&mut ctx, "gamepad", &options()).unwrap_err();
        assert!(
            format!("{error:#}").contains("does not verify"),
            "{error:#}"
        );
        assert!(plugin_dir_entries(&ctx.plugin_dir).is_empty());
    }

    #[test]
    fn network_needs_consent_even_with_yes() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_official_gamepad(&transport, temp.path(), "1.0.1");
        // -y without --allow-network in a non-interactive session.
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let error = install(&mut ctx, "gamepad", &InstallOptions::default()).unwrap_err();
        assert!(error.to_string().contains("--allow-network"), "{error}");
        assert!(transport.requested().is_empty());
        // Interactive consent ("y"), then confirmation (EOF declines).
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(true, &[Some("y"), None]),
            false,
        );
        let error = install(&mut ctx, "gamepad", &InstallOptions::default()).unwrap_err();
        assert_eq!(error.to_string(), "Operation cancelled.");
        assert!(
            !transport
                .requested()
                .iter()
                .any(|url| url.ends_with(".eop"))
        );
        // Consent and confirmation.
        let mut ctx = context(
            temp.path(),
            transport,
            ScriptedPrompter::new(true, &[Some(""), Some("yes")]),
            false,
        );
        install(&mut ctx, "gamepad", &InstallOptions::default()).unwrap();
    }

    pub(crate) fn serve_github_plugin(
        transport: &MockTransport,
        root: &Path,
        repo: &str,
        id: &str,
        declare_id: bool,
    ) {
        let bytes = package_bytes(root, id, "1.5.0");
        let url = format!("{repo}/releases/download/v1.5.0/{id}-linux-x86_64.eop");
        let header = if declare_id {
            format!(
                "[manifest]\nschema = 1\nkind = \"plugin\"\nversion = \"1.5.0\"\nrepository = \"{repo}\"\n\n[plugin]\nid = \"{id}\"\nname = \"Fork\"\nversion = \"1.5.0\"\nrepository = \"{repo}\"\nenvironment = \">=1.0.0, <2.0.0\"\n\n"
            )
        } else {
            String::new()
        };
        transport.respond(
            &format!("{repo}/releases/latest/download/release.toml"),
            200,
            format!(
                "{header}[platform.linux.x86_64]\nversion = \"1.5.0\"\nsha256 = \"{}\"\nurl = \"{url}\"\nenvironment = \">=1.0.0, <2.0.0\"\n",
                sha256_hex(&bytes)
            ),
        );
        transport.respond(&url, 200, bytes);
    }

    #[test]
    fn github_releases_must_declare_or_be_given_an_id() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_github_plugin(
            &transport,
            temp.path(),
            "https://github.com/someone/thing",
            "thing",
            false,
        );
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let error = install(&mut ctx, "github.com/someone/thing", &options()).unwrap_err();
        assert!(error.to_string().contains("--id"), "{error}");
        let mut with_id = options();
        with_id.id = Some("thing".into());
        install(&mut ctx, "github.com/someone/thing", &with_id).unwrap();
        let entry = &registry::load(&ctx.plugin_dir).unwrap().plugins["thing"];
        assert_eq!(entry.source, SOURCE_GITHUB);
        // The real tag from the artifact URL is recorded.
        assert_eq!(entry.release_tag.as_deref(), Some("v1.5.0"));
    }

    #[test]
    fn third_party_packages_cannot_silently_shadow_official_plugins() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_official_gamepad(&transport, temp.path(), "1.0.1");
        let fork = "https://github.com/fork/gamepad";
        serve_github_plugin(&transport, temp.path(), fork, "gamepad", true);
        let mut ctx = context(
            temp.path(),
            transport,
            ScriptedPrompter::new(false, &[]),
            true,
        );
        install(&mut ctx, "gamepad", &options()).unwrap();

        let error = install(&mut ctx, "github.com/fork/gamepad", &options()).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("official eov plugin") && message.contains("--force"),
            "{message}"
        );
        assert_eq!(
            registry::load(&ctx.plugin_dir).unwrap().plugins["gamepad"].source,
            SOURCE_OFFICIAL
        );

        let mut forced = options();
        forced.force = true;
        install(&mut ctx, "github.com/fork/gamepad", &forced).unwrap();
        let entry = &registry::load(&ctx.plugin_dir).unwrap().plugins["gamepad"];
        assert_eq!(entry.source, SOURCE_GITHUB);
        assert_eq!(entry.repository.as_deref(), Some(fork));
    }

    #[test]
    fn third_party_sources_obey_the_policy() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        ctx.policy.allow_third_party_plugins = false;
        let source = temp.path().join("local.eop");
        write_package(&source, "example", "1.0.0");
        let error = install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("only permits official"),
            "{error}"
        );
    }

    #[test]
    fn registry_write_failure_restores_the_previous_package() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let old = temp.path().join("old.eop");
        write_package(&old, "example", "1.0.0");
        install(&mut ctx, old.to_str().unwrap(), &InstallOptions::default()).unwrap();
        let installed = ctx.plugin_dir.join("example.eop");
        let before = fs::read(&installed).unwrap();

        // Make the registry impossible to replace.
        let registry_path = ctx.plugin_dir.join(registry::REGISTRY_FILENAME);
        let saved_registry = fs::read(&registry_path).unwrap();
        fs::remove_file(&registry_path).unwrap();
        fs::create_dir(&registry_path).unwrap();
        fs::write(registry_path.join("blocker"), &saved_registry).unwrap();

        let new = temp.path().join("new.eop");
        write_package(&new, "example", "1.0.1");
        assert!(install(&mut ctx, new.to_str().unwrap(), &InstallOptions::default()).is_err());
        assert_eq!(
            fs::read(&installed).unwrap(),
            before,
            "the old package was not restored"
        );
        let leftovers: Vec<_> = plugin_dir_entries(&ctx.plugin_dir)
            .into_iter()
            .filter(|name| name.starts_with('.') && name != registry::REGISTRY_FILENAME)
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn foreign_package_at_the_canonical_name_is_not_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        fs::create_dir_all(&ctx.plugin_dir).unwrap();
        write_package(&ctx.plugin_dir.join("example.eop"), "other", "1.0.0");
        let source = temp.path().join("example-src.eop");
        write_package(&source, "example", "1.0.0");
        let error = install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("different plugin"), "{error}");
    }

    #[test]
    fn downgrade_replaces_every_package_with_the_id() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        fs::create_dir_all(&ctx.plugin_dir).unwrap();
        write_package(
            &ctx.plugin_dir.join("example-1.0.3.eop"),
            "example",
            "1.0.3",
        );
        let stage = temp.path().join("example-1.0.2.eop");
        write_package(&stage, "example", "1.0.2");
        install(
            &mut ctx,
            stage.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        assert_eq!(
            plugin_dir_entries(&ctx.plugin_dir),
            vec![".eov-managed.toml", "example.eop"]
        );
    }

    #[test]
    fn broken_packages_are_listed_and_removable() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        fs::create_dir_all(&ctx.plugin_dir).unwrap();
        let broken = ctx.plugin_dir.join("broken.eop");
        fs::write(&broken, b"not a tar archive at all, sorry").unwrap();
        let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host).unwrap();
        assert_eq!(inventory.broken.len(), 1);
        remove(&mut ctx, "broken.eop", &RemoveOptions::default()).unwrap();
        assert!(!broken.exists());
    }

    #[test]
    fn disable_and_enable_without_uninstalling() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let source = temp.path().join("local.eop");
        write_package(&source, "example", "1.0.0");
        install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        assert!(set_plugin_enabled(&ctx.plugin_dir, "example", false).unwrap());
        assert!(!set_plugin_enabled(&ctx.plugin_dir, "example", false).unwrap());
        assert!(registry::is_plugin_disabled(&ctx.plugin_dir, "example"));
        let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host).unwrap();
        let rows = list_plugins(&inventory, None);
        assert_eq!(rows[0].status, "disabled");
        // Reinstalling keeps the user's choice.
        install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        assert!(registry::is_plugin_disabled(&ctx.plugin_dir, "example"));
        assert!(set_plugin_enabled(&ctx.plugin_dir, "example", true).unwrap());
        assert!(!registry::is_plugin_disabled(&ctx.plugin_dir, "example"));
    }

    #[test]
    fn listing_works_without_network_in_a_non_interactive_shell() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let source = temp.path().join("local.eop");
        write_package(&source, "example", "1.0.0");
        install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        ctx.interaction.assume_yes = false;
        list(&mut ctx, false, false).unwrap();
        assert!(transport.requested().is_empty());
    }

    #[test]
    fn managed_github_plugins_are_not_reported_as_unmanaged() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_github_plugin(
            &transport,
            temp.path(),
            "https://github.com/someone/thing",
            "thing",
            true,
        );
        let mut ctx = context(
            temp.path(),
            transport,
            ScriptedPrompter::new(false, &[]),
            true,
        );
        install(&mut ctx, "github.com/someone/thing", &options()).unwrap();
        let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host).unwrap();
        let rows = list_plugins(&inventory, None);
        assert_eq!(rows[0].source, "github (someone/thing)");
    }

    #[test]
    fn duplicate_ids_report_which_package_loads() {
        let temp = tempfile::tempdir().unwrap();
        let ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        fs::create_dir_all(&ctx.plugin_dir).unwrap();
        write_package(&ctx.plugin_dir.join("a.eop"), "example", "1.0.0");
        write_package(&ctx.plugin_dir.join("b.eop"), "example", "1.2.0");
        // An incompatible newer version is not selected.
        let text = manifest("example", "9.0.0", ">=9.0.0, <10.0.0");
        write_archive(
            &ctx.plugin_dir.join("c.eop"),
            &[("plugin.toml", text.as_bytes())],
            false,
        );
        let inventory = inventory(&ctx.plugin_dir, &[], &ctx.host).unwrap();
        let selected = inventory.selected("example").unwrap();
        assert_eq!(selected.version, "1.2.0");
        assert_eq!(inventory.with_id("example").len(), 3);
        // `info -o toml` works with duplicates and prints the loaded one.
        info(&ctx, "example", OutputFormat::Toml).unwrap();
        info(&ctx, "example", OutputFormat::Json).unwrap();
    }

    #[test]
    fn incompatible_api_requirement_is_refused_at_install() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let text = format!(
            "{}api = \">=2.0, <3\"\n",
            manifest("future", "1.0.0", ">=1.0.0, <2.0.0")
        );
        let source = write_archive(
            &temp.path().join("future.eop"),
            &[("plugin.toml", text.as_bytes())],
            false,
        );
        let error = install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("plugin API"), "{error:#}");
    }

    #[test]
    fn dry_run_changes_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_official_gamepad(&transport, temp.path(), "1.0.1");
        let mut ctx = context(
            temp.path(),
            transport.clone(),
            ScriptedPrompter::new(false, &[]),
            false,
        );
        ctx.json = true;
        let mut dry = options();
        dry.dry_run = true;
        install(&mut ctx, "gamepad", &dry).unwrap();
        assert!(!ctx.plugin_dir.join("gamepad.eop").exists());
        assert!(
            !transport
                .requested()
                .iter()
                .any(|url| url.ends_with(".eop"))
        );
    }

    #[test]
    fn lockfile_round_trip_reproduces_plugins_offline() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let source = temp.path().join("bundle").join("example.eop");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        write_package(&source, "example", "1.0.0");
        install(
            &mut ctx,
            source.to_str().unwrap(),
            &InstallOptions::default(),
        )
        .unwrap();
        set_plugin_enabled(&ctx.plugin_dir, "example", false).unwrap();

        let inventory_before = inventory(&ctx.plugin_dir, &[], &ctx.host).unwrap();
        let lock = lockfile(&inventory_before, &ctx.host, &ctx.platform);
        assert_eq!(lock.entries.len(), 1);
        assert!(!lock.entries[0].enabled);
        let lock_path = temp.path().join("eov-plugins.lock");
        fs::write(&lock_path, toml::to_string_pretty(&lock).unwrap()).unwrap();

        // A fresh workstation.
        let other = tempfile::tempdir().unwrap();
        let mut fresh = context(
            other.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let sync_options = SyncOptions {
            from: Some(temp.path().join("bundle")),
            ..SyncOptions::default()
        };
        sync(&mut fresh, &lock_path, &sync_options).unwrap();
        let synced = inventory(&fresh.plugin_dir, &[], &fresh.host).unwrap();
        assert_eq!(synced.packages[0].sha256, lock.entries[0].sha256);
        assert!(synced.packages[0].disabled);

        // A package with different bytes is refused.
        let mut tampered = lock.clone();
        tampered.entries[0].sha256 = "0".repeat(64);
        fs::write(&lock_path, toml::to_string_pretty(&tampered).unwrap()).unwrap();
        let third = tempfile::tempdir().unwrap();
        let mut strict = context(
            third.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        assert!(sync(&mut strict, &lock_path, &sync_options).is_err());
        assert!(!third.path().join("plugins").join("example.eop").exists());
    }

    #[test]
    fn lockfiles_from_another_platform_are_refused() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(
            temp.path(),
            MockTransport::new(),
            ScriptedPrompter::new(false, &[]),
            true,
        );
        let lock = Lockfile {
            schema: LOCKFILE_SCHEMA,
            eov: HOST.into(),
            platform: "windows.arm64".into(),
            entries: Vec::new(),
        };
        let path = temp.path().join("lock.toml");
        fs::write(&path, toml::to_string(&lock).unwrap()).unwrap();
        let error = sync(&mut ctx, &path, &SyncOptions::default()).unwrap_err();
        assert!(error.to_string().contains("windows.arm64"), "{error}");
    }
}
