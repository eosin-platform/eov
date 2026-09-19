//! Plugin installation, update, inspection, and provenance management.

use super::manifest::{
    EovReleaseManifest, OfficialPluginCatalogEntry, PlatformArtifact, PluginReleaseManifest,
    canonical_repository, normalize_version, release_manifest_url, release_tag,
};
use super::network::{NetworkPermission, ReleaseClient, require_permission};
use super::prompt::confirm;
use crate::plugins::discovery::{
    InspectedPluginPackage, inspect_plugin_package, inspect_staged_plugin_package,
    invalidate_plugin_package_cache,
};
use crate::version::BUILD_VERSION;
use anyhow::{Context, Result, bail};
use eov_plugin_api::PluginManifest;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const REGISTRY_SCHEMA: u32 = 1;
const REGISTRY_FILENAME: &str = ".eov-managed.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Plain,
    Toml,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PluginSource {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedPlugin {
    pub source: String,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub release_tag: Option<String>,
    pub version: String,
    pub sha256: String,
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedRegistry {
    pub schema: u32,
    #[serde(default)]
    pub plugins: BTreeMap<String, ManagedPlugin>,
}

#[derive(Debug, Clone)]
pub(crate) struct RegistryReplacement {
    pub(crate) id: String,
    pub(crate) source: String,
    pub(crate) repository: Option<String>,
    pub(crate) release_tag: Option<String>,
    pub(crate) version: String,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RegistryStage {
    pub(crate) staged: PathBuf,
    pub(crate) target: PathBuf,
    pub(crate) backup: PathBuf,
}

impl Default for ManagedRegistry {
    fn default() -> Self {
        Self {
            schema: REGISTRY_SCHEMA,
            plugins: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RemotePackagePlan {
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) repository: String,
    pub(crate) release_tag: String,
    pub(crate) version: Version,
    pub(crate) environment: Option<String>,
    pub(crate) artifact: PlatformArtifact,
    pub(crate) official: bool,
}

pub(crate) struct RemotePluginRequest<'a> {
    pub(crate) repository: &'a str,
    pub(crate) tag: Option<&'a str>,
    pub(crate) expected_id: Option<&'a str>,
    pub(crate) catalog_entry: Option<&'a OfficialPluginCatalogEntry>,
    pub(crate) official: bool,
    pub(crate) require_catalog_version: bool,
}

#[derive(Debug, Clone)]
struct InstallPlan {
    source: PluginSource,
    remote: Option<RemotePackagePlan>,
    local: Option<InspectedPluginPackage>,
    installed: Option<InspectedPluginPackage>,
}

pub fn list(plugin_dir: &Path, allow_network: bool) -> Result<()> {
    let local = inspect_packages(plugin_dir, true)?;
    let mut catalog = None;
    match NetworkPermission::acquire(
        allow_network,
        "This command needs the current EOV official plugin catalog.",
    )? {
        Some(permission) => {
            let client = ReleaseClient::new(&format!("eov/{BUILD_VERSION}"));
            match fetch_current_catalog(&client, &permission) {
                Ok(value) => catalog = Some(value),
                Err(error) => eprintln!("warning: could not load official plugin catalog: {error}"),
            }
        }
        None => println!("Network access declined; showing installed plugins only."),
    }
    print_plugin_table(&local, catalog.as_ref(), &load_registry(plugin_dir)?);
    Ok(())
}

pub fn info(plugin_dir: &Path, query: &str, output: OutputFormat) -> Result<()> {
    let packages = inspect_packages(plugin_dir, true)?;
    let matches = select_packages(&packages, query)?;
    if matches.is_empty() {
        bail!("no installed plugin matches '{query}'");
    }
    if output == OutputFormat::Toml {
        if matches.len() > 1 {
            bail!("multiple installed plugins match '{query}'; use the plugin ID");
        }
        print!("{}", matches[0].raw_manifest);
        return Ok(());
    }

    let registry = load_registry(plugin_dir)?;
    for (index, package) in matches.iter().enumerate() {
        if index > 0 {
            println!();
        }
        print_plugin_info(
            package,
            registry.plugins.get(&package.manifest.id),
            matches.len() > 1,
        );
    }
    Ok(())
}

pub fn remove(plugin_dir: &Path, query: &str, yes: bool) -> Result<()> {
    let packages = inspect_packages(plugin_dir, true)?;
    let matches = select_packages(&packages, query)?;
    if matches.is_empty() {
        bail!("no installed plugin matches '{query}'");
    }
    let ids: Vec<_> = matches
        .iter()
        .map(|package| package.manifest.id.clone())
        .collect();
    let id = unique_id(&ids, query)?;
    let selected: Vec<_> = packages
        .iter()
        .filter(|package| package.manifest.id == id)
        .collect();
    let package = selected
        .iter()
        .max_by(|left, right| compare_versions(&left.manifest.version, &right.manifest.version))
        .copied()
        .context("selected plugin has no package")?;

    let mut message = format!(
        "Plugin:  {}\nVersion: {}\nPackage: {}",
        package.manifest.name,
        package.manifest.version,
        package.package_path.display()
    );
    if selected.len() > 1 {
        message.push_str(&format!("\nPackages removed: {}", selected.len()));
    }
    confirm(&message, yes)?;

    let mut registry = load_registry(plugin_dir)?;
    let old_registry_entry = registry.plugins.remove(&id);
    let selected_paths: Vec<_> = selected
        .iter()
        .map(|package| package.package_path.clone())
        .collect();
    let backups = move_packages_to_backups(&selected_paths)?;
    if let Err(error) = write_registry(plugin_dir, &registry) {
        restore_backups(backups)?;
        if let Some(entry) = old_registry_entry {
            registry.plugins.insert(id.clone(), entry);
        }
        return Err(error);
    }
    delete_backups(backups)?;
    for package in selected {
        let _ = invalidate_plugin_package_cache(&package.package_path);
    }
    println!("Removed plugin '{id}'.");
    Ok(())
}

pub fn install(
    plugin_dir: &Path,
    source_text: &str,
    allow_network: bool,
    yes: bool,
    require_existing: bool,
) -> Result<()> {
    let source = parse_source(source_text)?;
    let existing_packages = inspect_packages(plugin_dir, true)?;
    let registry = load_registry(plugin_dir)?;
    let source = if require_existing {
        prepare_update_source(source, &existing_packages, &registry)?
    } else {
        source
    };

    let remote = !matches!(source, PluginSource::Local(_));
    let permission = if remote {
        Some(require_permission(
            allow_network,
            "This command needs release metadata and a verified plugin artifact from GitHub.",
        )?)
    } else {
        None
    };
    let client = permission
        .as_ref()
        .map(|_| ReleaseClient::new(&format!("eov/{BUILD_VERSION}")));
    let plan = resolve_install_plan(
        source,
        existing_packages,
        registry,
        client.as_ref(),
        permission.as_ref(),
        require_existing,
    )?;
    print_install_plan(&plan)?;
    if matches!(plan.source, PluginSource::Local(_))
        || plan.remote.as_ref().is_some_and(|remote| !remote.official)
    {
        print_third_party_warning();
    }
    confirm("Install this plugin?", yes)?;
    execute_install(plugin_dir, plan, permission.as_ref())
}

fn resolve_install_plan(
    source: PluginSource,
    existing_packages: Vec<InspectedPluginPackage>,
    registry: ManagedRegistry,
    client: Option<&ReleaseClient>,
    permission: Option<&NetworkPermission>,
    require_existing: bool,
) -> Result<InstallPlan> {
    match &source {
        PluginSource::Local(path) => {
            let path = path.canonicalize().with_context(|| {
                format!("local plugin package does not exist: {}", path.display())
            })?;
            let local = inspect_plugin_package(&path, true)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let installed = find_installed(&existing_packages, &local.manifest.id);
            if require_existing && installed.is_none() {
                bail!("plugin '{}' is not installed", local.manifest.id);
            }
            ensure_compatible(&local.manifest, &current_eov_version()?)?;
            Ok(InstallPlan {
                source,
                remote: None,
                local: Some(local),
                installed,
            })
        }
        PluginSource::Official { id, version } => {
            let permission = permission.context("missing network permission")?;
            let client = client.context("missing release client")?;
            let catalog = fetch_current_catalog(client, permission)?;
            let catalog_entry = catalog.plugins.get(id).with_context(|| {
                format!(
                    "'{id}' is not an official plugin in the EOV {} catalog",
                    catalog.version
                )
            })?;
            let target_version = version
                .clone()
                .unwrap_or(normalize_version(&catalog_entry.version)?);
            let repository = canonical_repository(&catalog_entry.repository)?;
            let tag = release_tag(&target_version);
            let remote = fetch_remote_plugin_plan(
                client,
                permission,
                RemotePluginRequest {
                    repository: &repository,
                    tag: Some(&tag),
                    expected_id: Some(id),
                    catalog_entry: Some(catalog_entry),
                    official: true,
                    require_catalog_version: version.is_none(),
                },
            )?;
            if remote.version != target_version {
                bail!(
                    "official plugin release version {} does not match requested version {}",
                    remote.version,
                    target_version
                );
            }
            if let Some(environment) = remote.environment.as_deref()
                && !requirement_matches(environment, &current_eov_version()?)
            {
                bail!(
                    "Plugin {} {} requires EOV {}, but this installation is EOV {}.",
                    id,
                    remote.version,
                    environment,
                    BUILD_VERSION
                );
            }
            let installed = find_installed(&existing_packages, id);
            if require_existing && installed.is_none() {
                bail!("plugin '{id}' is not installed");
            }
            Ok(InstallPlan {
                source,
                remote: Some(remote),
                local: None,
                installed,
            })
        }
        PluginSource::Github { repository, tag } => {
            let permission = permission.context("missing network permission")?;
            let client = client.context("missing release client")?;
            let remote = fetch_remote_plugin_plan(
                client,
                permission,
                RemotePluginRequest {
                    repository,
                    tag: tag.as_deref(),
                    expected_id: None,
                    catalog_entry: None,
                    official: false,
                    require_catalog_version: false,
                },
            )?;
            let installed = remote
                .id
                .as_deref()
                .and_then(|id| find_installed(&existing_packages, id))
                .or_else(|| {
                    registry.plugins.iter().find_map(|(id, entry)| {
                        (entry.repository.as_deref() == Some(repository))
                            .then(|| find_installed(&existing_packages, id))
                            .flatten()
                    })
                });
            if let Some(environment) = remote.environment.as_deref()
                && !requirement_matches(environment, &current_eov_version()?)
            {
                bail!(
                    "Plugin {} {} requires EOV {}, but this installation is EOV {}.",
                    remote.id.as_deref().unwrap_or("unknown"),
                    remote.version,
                    environment,
                    BUILD_VERSION
                );
            }
            if require_existing && installed.is_none() {
                bail!("no installed plugin is associated with GitHub repository {repository}");
            }
            Ok(InstallPlan {
                source,
                remote: Some(remote),
                local: None,
                installed,
            })
        }
    }
}

fn execute_install(
    plugin_dir: &Path,
    plan: InstallPlan,
    permission: Option<&NetworkPermission>,
) -> Result<()> {
    fs::create_dir_all(plugin_dir)
        .with_context(|| format!("could not create plugin directory {}", plugin_dir.display()))?;
    let (
        expected_id,
        expected_version,
        expected_environment,
        source_type,
        repository,
        release_tag,
        expected_sha,
        bytes,
    ) = if let Some(remote) = &plan.remote {
        let bytes =
            download_remote_artifact(remote, permission.context("missing network permission")?)?;
        (
            remote.id.clone(),
            remote.version.clone(),
            remote.environment.clone(),
            if remote.official {
                "official"
            } else {
                "github"
            },
            Some(remote.repository.clone()),
            Some(remote.release_tag.clone()),
            Some(remote.artifact.sha256.clone()),
            bytes,
        )
    } else {
        let local = plan.local.as_ref().context("missing local package")?;
        (
            Some(local.manifest.id.clone()),
            normalize_version(&local.manifest.version)?,
            Some(local.manifest.environment.version.clone()),
            "local",
            local.manifest.repository.clone(),
            None,
            None,
            fs::read(&local.package_path)
                .with_context(|| format!("could not read {}", local.package_path.display()))?,
        )
    };

    let actual_sha = sha256_bytes(&bytes);
    if let Some(expected_sha) = expected_sha.as_deref()
        && !expected_sha.eq_ignore_ascii_case(&actual_sha)
    {
        bail!(
            "SHA-256 mismatch for downloaded plugin: expected {}, got {}",
            expected_sha,
            actual_sha
        );
    }
    let stage_path = temporary_package_path(plugin_dir)?;
    write_bytes_and_sync(&stage_path, &bytes)?;

    let inspected = match inspect_staged_plugin_package(&stage_path, true) {
        Ok(inspected) => inspected,
        Err(error) => {
            let _ = fs::remove_file(&stage_path);
            return Err(anyhow::anyhow!(error.to_string()));
        }
    };
    if let Err(error) = validate_downloaded_manifest(
        &inspected.manifest,
        expected_id.as_deref(),
        &expected_version,
        expected_environment.as_deref(),
    ) {
        let _ = fs::remove_file(&stage_path);
        return Err(error);
    }
    let id = inspected.manifest.id.clone();
    let installed = plan.installed;
    let action = classify_action(
        installed
            .as_ref()
            .map(|package| package.manifest.version.as_str()),
        &inspected.manifest.version,
    );
    println!("Action: {action}");
    let mut registry = load_registry(plugin_dir)?;
    let entry = ManagedPlugin {
        source: source_type.to_string(),
        repository,
        release_tag,
        version: inspected.manifest.version.clone(),
        sha256: actual_sha,
        file: format!("{id}.eop"),
    };
    replace_package_atomically(plugin_dir, &id, &stage_path, &mut registry, entry)?;
    println!(
        "Installed {} {}.",
        inspected.manifest.name, inspected.manifest.version
    );
    Ok(())
}

pub(crate) fn install_verified_stage(
    plugin_dir: &Path,
    stage_path: &Path,
    source: &str,
    repository: Option<String>,
    release_tag: Option<String>,
    expected_sha256: &str,
) -> Result<InspectedPluginPackage> {
    let inspected = inspect_staged_plugin_package(stage_path, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let actual_sha256 = inspected
        .sha256
        .as_deref()
        .context("staged package hash was not calculated")?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        bail!(
            "SHA-256 mismatch for staged plugin: expected {}, got {}",
            expected_sha256,
            actual_sha256
        );
    }
    validate_plugin_id_for_filename(&inspected.manifest.id)?;
    let mut registry = load_registry(plugin_dir)?;
    let id = inspected.manifest.id.clone();
    let entry = ManagedPlugin {
        source: source.to_string(),
        repository,
        release_tag,
        version: inspected.manifest.version.clone(),
        sha256: actual_sha256.to_string(),
        file: format!("{id}.eop"),
    };
    replace_package_atomically(plugin_dir, &id, stage_path, &mut registry, entry)?;
    Ok(inspected)
}

fn download_remote_artifact(
    plan: &RemotePackagePlan,
    permission: &NetworkPermission,
) -> Result<Vec<u8>> {
    let client = ReleaseClient::new(&format!("eov/{BUILD_VERSION}"));
    client.get_bytes(permission, &plan.artifact.url)
}

fn fetch_current_catalog(
    client: &ReleaseClient,
    permission: &NetworkPermission,
) -> Result<EovReleaseManifest> {
    let version = current_eov_version()?;
    let url = release_manifest_url(
        super::manifest::EOV_REPOSITORY,
        Some(&release_tag(&version)),
    )?;
    let text = client.get_text(permission, &url).with_context(|| {
        format!(
            "No published release manifest exists for EOV {version}.\nThe official plugin compatibility catalog is therefore unavailable. Install an explicit GitHub plugin release instead if appropriate."
        )
    })?;
    EovReleaseManifest::parse(&text, Some(&version))
}

pub(crate) fn fetch_remote_plugin_plan(
    client: &ReleaseClient,
    permission: &NetworkPermission,
    request: RemotePluginRequest<'_>,
) -> Result<RemotePackagePlan> {
    let RemotePluginRequest {
        repository,
        tag,
        expected_id,
        catalog_entry,
        official,
        require_catalog_version,
    } = request;
    let manifest_url = release_manifest_url(repository, tag)?;
    let text = client.get_text(permission, &manifest_url)?;
    let manifest = PluginReleaseManifest::parse(&text, repository)?;
    if let Some(expected_id) = expected_id
        && manifest.plugin_id().is_some_and(|id| id != expected_id)
    {
        bail!("plugin release manifest ID does not match official plugin '{expected_id}'");
    }
    if require_catalog_version
        && let Some(entry) = catalog_entry
        && manifest.version != normalize_version(&entry.version)?
    {
        bail!(
            "official plugin release version {} does not match catalog version {}",
            manifest.version,
            entry.version
        );
    }
    let key = host_plugin_artifact_key()?;
    let artifact = manifest.artifact(&key)?.clone();
    let id = manifest
        .plugin_id()
        .map(str::to_string)
        .or_else(|| expected_id.map(str::to_string));
    let name = catalog_entry
        .and_then(|entry| entry.name.clone())
        .or_else(|| {
            manifest
                .plugin
                .as_ref()
                .and_then(|plugin| plugin.name.clone())
        })
        .or_else(|| id.as_deref().map(super::manifest::title_case_id));
    let release_tag = tag
        .map(str::to_string)
        .unwrap_or_else(|| release_tag(&manifest.version));
    Ok(RemotePackagePlan {
        id,
        name,
        repository: canonical_repository(repository)?,
        release_tag,
        version: manifest.version.clone(),
        environment: manifest.environment().map(str::to_string),
        artifact,
        official,
    })
}

fn validate_downloaded_manifest(
    manifest: &PluginManifest,
    expected_id: Option<&str>,
    expected_version: &Version,
    expected_environment: Option<&str>,
) -> Result<()> {
    if let Some(expected_id) = expected_id
        && manifest.id != expected_id
    {
        bail!(
            "downloaded plugin ID '{}' does not match expected '{}'",
            manifest.id,
            expected_id
        );
    }
    let actual_version = normalize_version(&manifest.version)?;
    if &actual_version != expected_version {
        bail!(
            "downloaded plugin version {} does not match release metadata {}",
            manifest.version,
            expected_version
        );
    }
    if let Some(expected_environment) = expected_environment
        && manifest.environment.version != expected_environment
    {
        bail!(
            "downloaded plugin environment '{}' does not match release metadata '{}'",
            manifest.environment.version,
            expected_environment
        );
    }
    ensure_compatible(manifest, &current_eov_version()?)
}

fn ensure_compatible(manifest: &PluginManifest, eov_version: &Version) -> Result<()> {
    let requirement = VersionReq::parse(&manifest.environment.version).with_context(|| {
        format!(
            "plugin {} has invalid EOV requirement {}",
            manifest.id, manifest.environment.version
        )
    })?;
    if !requirement.matches(eov_version) {
        if requirement.matches(&Version::new(eov_version.major + 1, 0, 0)) {
            bail!(
                "Plugin {} {} requires EOV {}, but this installation is EOV {}.\n\nRun:\n    eov update",
                manifest.id,
                manifest.version,
                manifest.environment.version,
                eov_version
            );
        }
        bail!(
            "Plugin {} {} requires EOV {}, but this installation is EOV {}.",
            manifest.id,
            manifest.version,
            manifest.environment.version,
            eov_version
        );
    }
    Ok(())
}

fn parse_source(value: &str) -> Result<PluginSource> {
    if value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with('/')
        || value.ends_with(".eop")
    {
        return Ok(PluginSource::Local(PathBuf::from(value)));
    }
    let repository_value = value.strip_prefix("https://").unwrap_or(value);
    if repository_value.starts_with("github.com/") {
        let (repository, tag) = match repository_value.rsplit_once('@') {
            Some((repository, tag)) => (repository, Some(tag.to_string())),
            None => (repository_value, None),
        };
        let repository = format!("https://{repository}");
        Ok(PluginSource::Github {
            repository: canonical_repository(&repository)?,
            tag,
        })
    } else {
        let (id, version) = match value.split_once('@') {
            Some((id, version)) => (id, Some(normalize_version(version)?)),
            None => (value, None),
        };
        if id.is_empty() || id.contains('/') {
            bail!("invalid official plugin source '{value}'");
        }
        Ok(PluginSource::Official {
            id: id.to_string(),
            version,
        })
    }
}

fn prepare_update_source(
    source: PluginSource,
    packages: &[InspectedPluginPackage],
    registry: &ManagedRegistry,
) -> Result<PluginSource> {
    match source {
        PluginSource::Official { id, version } => {
            let installed = find_installed(packages, &id);
            if installed.is_none() {
                bail!("plugin '{id}' is not installed");
            }
            if version.is_some() {
                return Ok(PluginSource::Official { id, version });
            }
            match registry.plugins.get(&id) {
                Some(entry) if entry.source == "official" => {
                    Ok(PluginSource::Official { id, version })
                }
                Some(entry) if entry.source == "github" => Ok(PluginSource::Github {
                    repository: entry
                        .repository
                        .clone()
                        .context("managed GitHub plugin has no repository")?,
                    tag: None,
                }),
                Some(entry) if entry.source == "local" => bail!(
                    "Plugin '{id}' was installed from a local package and has no remote update source.\n\nInstall a new package explicitly:\n    eov plugin install /path/to/{id}.eop"
                ),
                _ => Ok(PluginSource::Official { id, version }),
            }
        }
        PluginSource::Github { repository, tag } => {
            if tag.is_some() {
                return Ok(PluginSource::Github { repository, tag });
            }
            let known = registry.plugins.values().any(|entry| {
                entry.source == "github" && entry.repository.as_deref() == Some(repository.as_str())
            });
            if !known {
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

fn inspect_packages(
    plugin_dir: &Path,
    calculate_sha256: bool,
) -> Result<Vec<InspectedPluginPackage>> {
    if !plugin_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(plugin_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("eop")))
        .collect::<Vec<_>>();
    paths.sort();
    let mut packages = Vec::new();
    for path in paths {
        match inspect_plugin_package(&path, calculate_sha256) {
            Ok(package) => packages.push(package),
            Err(error) => eprintln!(
                "warning: invalid plugin package {}: {error}",
                path.display()
            ),
        }
    }
    Ok(packages)
}

fn select_packages<'a>(
    packages: &'a [InspectedPluginPackage],
    query: &str,
) -> Result<Vec<&'a InspectedPluginPackage>> {
    let id_matches: Vec<_> = packages
        .iter()
        .filter(|package| package.manifest.id == query)
        .collect();
    if !id_matches.is_empty() {
        return Ok(id_matches);
    }
    Ok(packages
        .iter()
        .filter(|package| package.manifest.name.eq_ignore_ascii_case(query))
        .collect())
}

fn unique_id(ids: &[String], query: &str) -> Result<String> {
    let Some(id) = ids.first() else {
        bail!("no installed plugin matches '{query}'");
    };
    if ids.iter().any(|candidate| candidate != id) {
        bail!("plugin name '{query}' matches multiple plugin IDs; use an ID");
    }
    Ok(id.clone())
}

fn find_installed(packages: &[InspectedPluginPackage], id: &str) -> Option<InspectedPluginPackage> {
    packages
        .iter()
        .filter(|package| package.manifest.id == id)
        .max_by(|left, right| compare_versions(&left.manifest.version, &right.manifest.version))
        .cloned()
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (Version::parse(left), Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn requirement_matches(requirement: &str, version: &Version) -> bool {
    VersionReq::parse(requirement)
        .map(|requirement| requirement.matches(version))
        .unwrap_or(false)
}

fn classify_action(installed: Option<&str>, target: &str) -> &'static str {
    let Some(installed) = installed else {
        return "install";
    };
    match compare_versions(target, installed) {
        std::cmp::Ordering::Greater => "upgrade",
        std::cmp::Ordering::Less => "downgrade",
        std::cmp::Ordering::Equal => "reinstall",
    }
}

fn print_install_plan(plan: &InstallPlan) -> Result<()> {
    let (id, name, installed, target, environment, artifact, repository) =
        if let Some(remote) = &plan.remote {
            (
                remote
                    .id
                    .as_deref()
                    .unwrap_or("(discovered after download)")
                    .to_string(),
                remote
                    .name
                    .as_deref()
                    .unwrap_or("Unknown plugin")
                    .to_string(),
                plan.installed
                    .as_ref()
                    .map(|package| package.manifest.version.clone())
                    .unwrap_or_else(|| "-".to_string()),
                remote.version.to_string(),
                remote
                    .environment
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_string(),
                remote
                    .artifact
                    .url
                    .rsplit('/')
                    .next()
                    .unwrap_or("artifact")
                    .to_string(),
                remote.repository.clone(),
            )
        } else {
            let local = plan.local.as_ref().context("missing local plan")?;
            (
                local.manifest.id.clone(),
                local.manifest.name.clone(),
                plan.installed
                    .as_ref()
                    .map(|package| package.manifest.version.clone())
                    .unwrap_or_else(|| "-".to_string()),
                local.manifest.version.clone(),
                local.manifest.environment.version.clone(),
                local.package_path.display().to_string(),
                local
                    .manifest
                    .repository
                    .clone()
                    .unwrap_or_else(|| "local package".to_string()),
            )
        };
    let action = classify_action((installed != "-").then_some(installed.as_str()), &target);
    println!(
        "Plugin:       {name}\nID:            {id}\nInstalled:    {installed}\nTarget:       {target}\nAction:       {action}\nRequires EOV: {environment}\nArtifact:     {artifact}\nRepository:   {repository}"
    );
    if let Some(local) = &plan.local {
        if let Some(sha256) = &local.sha256 {
            println!("SHA-256:      {sha256}");
        }
    } else if let Some(remote) = &plan.remote {
        println!("SHA-256:      {}", remote.artifact.sha256);
    }
    Ok(())
}

fn print_third_party_warning() {
    println!(
        "This is a third-party plugin.\n\nEOV plugins contain native executable code and run with the permissions of EOV itself. Install plugins only from sources you trust."
    );
}

fn print_plugin_info(
    package: &InspectedPluginPackage,
    managed: Option<&ManagedPlugin>,
    duplicate: bool,
) {
    println!("Name:             {}", package.manifest.name);
    println!("ID:               {}", package.manifest.id);
    println!("Version:          {}", package.manifest.version);
    if let Some(description) = &package.manifest.description {
        println!("Description:       {description}");
    }
    println!("EOV requirement:  {}", package.manifest.environment.version);
    println!("Package path:      {}", package.package_path.display());
    println!(
        "Package SHA-256:   {}",
        package.sha256.as_deref().unwrap_or("-")
    );
    if let Some(managed) = managed {
        println!("Managed:           yes");
        println!("Source type:       {}", managed.source);
        println!(
            "Source repository:  {}",
            managed.repository.as_deref().unwrap_or("-")
        );
        println!(
            "Release tag:       {}",
            managed.release_tag.as_deref().unwrap_or("-")
        );
    } else {
        println!("Managed:           no (unmanaged)");
    }
    if duplicate {
        println!(
            "Duplicate ID:      yes; runtime discovery chooses the highest compatible version"
        );
    }
}

fn print_plugin_table(
    local: &[InspectedPluginPackage],
    catalog: Option<&EovReleaseManifest>,
    registry: &ManagedRegistry,
) {
    println!("Name | Available | Installed | Status | Description | Repo");
    let mut ids = BTreeMap::<String, Option<&InspectedPluginPackage>>::new();
    for package in local {
        ids.entry(package.manifest.id.clone())
            .and_modify(|current| {
                if current.as_ref().is_none_or(|current| {
                    compare_versions(&package.manifest.version, &current.manifest.version).is_gt()
                }) {
                    *current = Some(package);
                }
            })
            .or_insert(Some(package));
    }
    if let Some(catalog) = catalog {
        for id in catalog.plugins.keys() {
            ids.entry(id.clone()).or_insert(None);
        }
    }
    for (id, package) in ids {
        let official = catalog.and_then(|catalog| catalog.plugins.get(&id));
        let name = official
            .and_then(|entry| entry.name.as_deref())
            .or_else(|| package.map(|package| package.manifest.name.as_str()))
            .unwrap_or(&id);
        let available = official.map(|entry| entry.version.as_str()).unwrap_or("-");
        let installed = package
            .map(|package| package.manifest.version.as_str())
            .unwrap_or("-");
        let status = match (package, official) {
            (None, Some(_)) => "not installed",
            (Some(package), Some(entry)) => {
                let host_version = Version::parse(BUILD_VERSION).ok();
                if host_version.as_ref().is_none_or(|version| {
                    !requirement_matches(&package.manifest.environment.version, version)
                }) {
                    "incompatible"
                } else {
                    match compare_versions(&package.manifest.version, &entry.version) {
                        std::cmp::Ordering::Less => "update",
                        std::cmp::Ordering::Equal => "current",
                        std::cmp::Ordering::Greater => "downgrade available",
                    }
                }
            }
            (Some(_), None) => match registry.plugins.get(&id) {
                Some(entry) if entry.source == "local" => "local",
                _ => "unmanaged",
            },
            (None, None) => "-",
        };
        let description = official
            .and_then(|entry| entry.description.as_deref())
            .or_else(|| package.and_then(|package| package.manifest.description.as_deref()))
            .unwrap_or("-");
        let repository = official
            .map(|entry| entry.repository.as_str())
            .or_else(|| package.and_then(|package| package.manifest.repository.as_deref()))
            .unwrap_or("-")
            .trim_start_matches("https://");
        println!("{name} | {available} | {installed} | {status} | {description} | {repository}");
    }
}

pub(crate) fn load_registry(plugin_dir: &Path) -> Result<ManagedRegistry> {
    let path = plugin_dir.join(REGISTRY_FILENAME);
    let path = if path.exists() {
        path
    } else {
        let mut backups = fs::read_dir(plugin_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&format!("{REGISTRY_FILENAME}.backup-")))
            })
            .collect::<Vec<_>>();
        backups.sort();
        backups.pop().unwrap_or(path)
    };
    if !path.exists() {
        return Ok(ManagedRegistry::default());
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("could not read plugin registry {}", path.display()))?;
    let registry: ManagedRegistry = toml::from_str(&content)
        .with_context(|| format!("could not parse plugin registry {}", path.display()))?;
    if registry.schema != REGISTRY_SCHEMA {
        bail!("unsupported plugin registry schema {}", registry.schema);
    }
    Ok(registry)
}

pub(crate) fn stage_registry_update(
    plugin_dir: &Path,
    replacements: &[RegistryReplacement],
) -> Result<RegistryStage> {
    let mut registry = load_registry(plugin_dir)?;
    for replacement in replacements {
        registry.plugins.insert(
            replacement.id.clone(),
            ManagedPlugin {
                source: replacement.source.clone(),
                repository: replacement.repository.clone(),
                release_tag: replacement.release_tag.clone(),
                version: replacement.version.clone(),
                sha256: replacement.sha256.clone(),
                file: format!("{}.eop", replacement.id),
            },
        );
    }
    fs::create_dir_all(plugin_dir)?;
    let target = plugin_dir.join(REGISTRY_FILENAME);
    let staged = plugin_dir.join(format!(".eov-managed-stage-{}.toml", unique_suffix()));
    let backup = plugin_dir.join(format!(".eov-managed-backup-{}.toml", unique_suffix()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    if let Err(error) = (|| -> Result<()> {
        file.write_all(toml::to_string_pretty(&registry)?.as_bytes())?;
        file.sync_all()?;
        Ok(())
    })() {
        let _ = fs::remove_file(&staged);
        return Err(error);
    }
    Ok(RegistryStage {
        staged,
        target,
        backup,
    })
}

fn write_registry(plugin_dir: &Path, registry: &ManagedRegistry) -> Result<()> {
    fs::create_dir_all(plugin_dir)?;
    let path = plugin_dir.join(REGISTRY_FILENAME);
    let temp = path.with_file_name(format!(".{REGISTRY_FILENAME}.tmp-{}", unique_suffix()));
    let content = toml::to_string_pretty(registry)?;
    #[cfg(windows)]
    let backup = path.with_file_name(format!("{REGISTRY_FILENAME}.backup-{}", unique_suffix()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        #[cfg(windows)]
        if path.exists() {
            fs::rename(&path, &backup)?;
        }
        fs::rename(&temp, &path)?;
        #[cfg(windows)]
        if backup.exists() {
            fs::remove_file(&backup)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
        #[cfg(windows)]
        if backup.exists() {
            let _ = fs::remove_file(&path);
            let _ = fs::rename(&backup, &path);
        }
    }
    result
}

fn replace_package_atomically(
    plugin_dir: &Path,
    id: &str,
    stage_path: &Path,
    registry: &mut ManagedRegistry,
    entry: ManagedPlugin,
) -> Result<()> {
    validate_plugin_id_for_filename(id)?;
    let packages = inspect_packages(plugin_dir, false)?;
    let old_paths: Vec<_> = packages
        .iter()
        .filter(|package| package.manifest.id == id && package.package_path != stage_path)
        .map(|package| package.package_path.clone())
        .collect();
    let backups = move_packages_to_backups(&old_paths)?;
    let target = plugin_dir.join(format!("{id}.eop"));
    let mut updated = registry.clone();
    updated.plugins.insert(id.to_string(), entry);
    if let Err(error) = (|| -> Result<()> {
        fs::rename(stage_path, &target)?;
        write_registry(plugin_dir, &updated)?;
        Ok(())
    })() {
        let _ = fs::remove_file(&target);
        restore_backups(backups)?;
        return Err(error);
    }
    delete_backups(backups)?;
    *registry = updated;
    for path in old_paths {
        let _ = invalidate_plugin_package_cache(&path);
    }
    let _ = invalidate_plugin_package_cache(&target);
    Ok(())
}

fn move_packages_to_backups(packages: &[PathBuf]) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut backups = Vec::new();
    for package in packages {
        let backup = package.with_file_name(format!(
            ".{}.backup-{}",
            package
                .file_name()
                .and_then(OsStr::to_str)
                .unwrap_or("plugin.eop"),
            unique_suffix()
        ));
        if let Err(error) = fs::rename(package, &backup) {
            restore_backups(backups)?;
            return Err(error.into());
        }
        backups.push((package.clone(), backup));
    }
    Ok(backups)
}

fn restore_backups(backups: Vec<(PathBuf, PathBuf)>) -> Result<()> {
    for (original, backup) in backups {
        if backup.exists() {
            fs::rename(backup, original)?;
        }
    }
    Ok(())
}

fn delete_backups(backups: Vec<(PathBuf, PathBuf)>) -> Result<()> {
    for (_, backup) in backups {
        if backup.exists() {
            fs::remove_file(backup)?;
        }
    }
    Ok(())
}

fn temporary_package_path(plugin_dir: &Path) -> Result<PathBuf> {
    Ok(plugin_dir.join(format!(".eov-stage-{}.part", unique_suffix())))
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

fn write_bytes_and_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    if let Err(error) = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })() {
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn host_plugin_artifact_key() -> Result<String> {
    let architecture = super::manifest::Architecture::current()?;
    let platform = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        bail!("unsupported host platform for native plugins")
    };
    Ok(format!("platform.{platform}.{}", architecture.as_str()))
}

fn current_eov_version() -> Result<Version> {
    normalize_version(BUILD_VERSION)
}

fn validate_plugin_id_for_filename(id: &str) -> Result<()> {
    if id.is_empty()
        || id == "."
        || id == ".."
        || id.contains('/')
        || id.contains('\\')
        || id.contains(std::path::MAIN_SEPARATOR)
    {
        bail!("plugin ID '{id}' cannot be used as a managed filename");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    fn write_package(path: &Path, id: &str, version: &str) {
        let file = File::create(path).unwrap();
        let mut builder = tar::Builder::new(file);
        let content = format!(
            "id = \"{id}\"\nname = \"{}\"\nversion = \"{version}\"\ndescription = \"Test package\"\nrepository = \"https://github.com/example/{id}\"\n\n[environment]\nversion = \">=0.4.1\"\n",
            super::super::manifest::title_case_id(id)
        );
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::file());
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "plugin.toml", content.as_bytes())
            .unwrap();
        builder.finish().unwrap();
    }

    #[test]
    fn parses_official_and_github_sources() {
        assert!(matches!(
            parse_source("gamepad@v0.2.2").unwrap(),
            PluginSource::Official { .. }
        ));
        assert!(matches!(
            parse_source("github.com/foo/bar@my-tag").unwrap(),
            PluginSource::Github { tag: Some(_), .. }
        ));
        assert!(matches!(
            parse_source("./foo.eop").unwrap(),
            PluginSource::Local(_)
        ));
    }

    #[test]
    fn classifies_downgrades_and_reinstalls() {
        assert_eq!(classify_action(Some("1.0.0"), "0.9.0"), "downgrade");
        assert_eq!(classify_action(Some("1.0.0"), "1.0.0"), "reinstall");
        assert_eq!(classify_action(None, "1.0.0"), "install");
    }

    #[test]
    fn registry_round_trips() {
        let registry = ManagedRegistry {
            schema: REGISTRY_SCHEMA,
            plugins: BTreeMap::from([(
                "gamepad".into(),
                ManagedPlugin {
                    source: "official".into(),
                    repository: Some("https://github.com/eosin-platform/eov-gamepad-plugin".into()),
                    release_tag: Some("v0.2.3".into()),
                    version: "0.2.3".into(),
                    sha256: "a".repeat(64),
                    file: "gamepad.eop".into(),
                },
            )]),
        };
        let text = toml::to_string(&registry).unwrap();
        let parsed: ManagedRegistry = toml::from_str(&text).unwrap();
        assert_eq!(parsed, registry);
    }

    #[test]
    fn refuses_unsafe_managed_ids() {
        assert!(validate_plugin_id_for_filename("../escape").is_err());
        assert!(validate_plugin_id_for_filename("gamepad").is_ok());
    }

    #[test]
    fn local_install_is_offline_and_uses_canonical_filename() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("local.eop");
        let plugin_dir = temp.path().join("plugins");
        write_package(&source, "example", "1.0.0");
        install(&plugin_dir, source.to_str().unwrap(), false, true, false).unwrap();
        let installed = plugin_dir.join("example.eop");
        assert!(installed.is_file());
        assert_eq!(
            load_registry(&plugin_dir).unwrap().plugins["example"].source,
            "local"
        );
        let inspected = inspect_plugin_package(&installed, true).unwrap();
        assert!(
            inspected
                .raw_manifest
                .contains("description = \"Test package\"")
        );
    }

    #[test]
    fn managed_replacement_removes_duplicate_ids_for_downgrades() {
        let temp = tempfile::tempdir().unwrap();
        let plugin_dir = temp.path().join("plugins");
        fs::create_dir_all(&plugin_dir).unwrap();
        let old = plugin_dir.join("gamepad-0.2.3.eop");
        let stage = temp.path().join("gamepad-0.2.2.eop");
        write_package(&old, "gamepad", "0.2.3");
        write_package(&stage, "gamepad", "0.2.2");
        let bytes = fs::read(&stage).unwrap();
        let hash = sha256_bytes(&bytes);
        let installed = install_verified_stage(
            &plugin_dir,
            &stage,
            "official",
            Some("https://github.com/eosin-platform/eov-gamepad-plugin".into()),
            Some("v0.2.2".into()),
            &hash,
        )
        .unwrap();
        assert_eq!(installed.manifest.version, "0.2.2");
        assert!(!old.exists());
        assert!(plugin_dir.join("gamepad.eop").is_file());
    }
}
