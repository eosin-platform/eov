//! Application update resolution, planning, staging, and distribution dispatch.

use super::manifest::{
    EovReleaseManifest, PlatformArtifact, PluginReleaseManifest, normalize_version,
    release_manifest_url, release_tag, validate_self_update_target,
};
use super::network::{NetworkPermission, ReleaseClient, require_permission};
use super::plugins::{self, RegistryReplacement, RemotePackagePlan, install_verified_stage};
use super::prompt::confirm;
use crate::distribution::{self, Distribution};
use crate::plugins::discovery::{
    InspectedPluginPackage, inspect_plugin_package, inspect_staged_plugin_package,
};
use crate::version::BUILD_VERSION;
use anyhow::{Context, Result, bail};
use semver::{Version, VersionReq};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateAction {
    Upgrade,
    Downgrade,
    Current,
    Skipped,
    Incompatible,
    Unchanged,
}

impl UpdateAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upgrade => "upgrade",
            Self::Downgrade => "downgrade",
            Self::Current => "current",
            Self::Skipped => "skipped",
            Self::Incompatible => "incompatible",
            Self::Unchanged => "unchanged",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppUpdatePlan {
    pub current_version: Version,
    pub target_version: Version,
    pub action: UpdateAction,
    pub distribution: Distribution,
    pub artifact: Option<PlatformArtifact>,
}

#[derive(Debug, Clone)]
pub struct PluginUpdatePlan {
    pub id: String,
    pub name: String,
    pub current_version: String,
    pub target_version: Option<String>,
    pub action: UpdateAction,
    pub repository: Option<String>,
    pub release_tag: Option<String>,
    pub environment: Option<String>,
    pub artifact: Option<PlatformArtifact>,
    pub official: bool,
}

#[derive(Debug, Clone)]
pub struct UpdatePlan {
    pub app: AppUpdatePlan,
    pub plugins: Vec<PluginUpdatePlan>,
}

#[derive(Debug, Serialize)]
struct HelperTransaction {
    schema: u32,
    nonce: String,
    pid: u32,
    mode: String,
    current_tree: Option<PathBuf>,
    staged_tree: Option<PathBuf>,
    backup_tree: Option<PathBuf>,
    cargo_version: Option<String>,
    relaunch: Option<PathBuf>,
    plugins: Vec<HelperPluginReplacement>,
    registry: Option<HelperRegistryReplacement>,
}

#[derive(Debug, Serialize)]
struct HelperPluginReplacement {
    staged: PathBuf,
    target: PathBuf,
    backup: PathBuf,
    remove_after: Vec<PathBuf>,
}

#[derive(Debug, Serialize)]
struct HelperRegistryReplacement {
    staged: PathBuf,
    target: PathBuf,
    backup: PathBuf,
}

pub fn application_update(
    plugin_dir: &Path,
    options: super::ApplicationUpdateOptions,
) -> Result<()> {
    let current_version = normalize_version(BUILD_VERSION)?;
    let requested_version = match options.release.as_deref() {
        Some(value) => {
            let version = normalize_version(value)?;
            validate_self_update_target(&version)?;
            Some(version)
        }
        None => None,
    };
    let permission = require_permission(
        options.allow_network,
        "This command needs release metadata and verified update artifacts from GitHub.",
    )?;
    let client = ReleaseClient::new(&format!("eov/{BUILD_VERSION}"));
    let manifest_url = release_manifest_url(
        super::manifest::EOV_REPOSITORY,
        requested_version.as_ref().map(release_tag).as_deref(),
    )?;
    let manifest_text = client.get_text(&permission, &manifest_url)?;
    let manifest = EovReleaseManifest::parse(&manifest_text, requested_version.as_ref())?;
    validate_self_update_target(&manifest.version)?;

    let plan = resolve_plan(
        plugin_dir,
        &manifest,
        current_version,
        distribution::distribution(),
        options.app_only,
        &client,
        &permission,
    )?;
    print_plan(&plan, options.app_only);
    confirm(
        if requires_restart(&plan.app.distribution, plan.app.action) {
            "EOV will exit and restart to replace the application."
        } else {
            "EOV will apply the verified update transaction."
        },
        options.yes,
    )?;
    execute_plan(
        plugin_dir,
        &plan,
        &client,
        &permission,
        options.app_only,
        options.release.is_some(),
    )
}

pub fn resolve_plan(
    plugin_dir: &Path,
    manifest: &EovReleaseManifest,
    current_version: Version,
    distribution: Distribution,
    app_only: bool,
    client: &ReleaseClient,
    permission: &NetworkPermission,
) -> Result<UpdatePlan> {
    let app_action = classify_versions(&current_version, &manifest.version);
    let artifact = app_artifact(manifest, distribution)?;
    let local_packages = inspect_packages(plugin_dir)?;
    let registry = plugins::load_registry(plugin_dir)?;
    let mut plugins = Vec::new();
    for package in local_packages {
        let id = package.manifest.id.clone();
        let current = package.manifest.version.clone();
        let Some(catalog) = manifest.plugins.get(&id) else {
            let compatible =
                requirement_matches(&package.manifest.environment.version, &manifest.version);
            if !app_only
                && let Some(managed) = registry.plugins.get(&id)
                && managed.source == "github"
                && managed.repository.is_some()
            {
                if let Some(repository) = managed.repository.as_deref() {
                    match resolve_third_party_target(
                        client,
                        permission,
                        repository,
                        &manifest.version,
                    ) {
                        Ok(Some(remote)) => {
                            plugins.push(PluginUpdatePlan {
                                id,
                                name: remote.name.unwrap_or_else(|| {
                                    super::manifest::title_case_id(&package.manifest.id)
                                }),
                                current_version: current,
                                target_version: Some(remote.version.to_string()),
                                action: classify_versions(
                                    &normalize_version(&package.manifest.version)?,
                                    &remote.version,
                                ),
                                repository: Some(remote.repository),
                                release_tag: Some(remote.release_tag),
                                environment: remote.environment,
                                artifact: Some(remote.artifact),
                                official: false,
                            });
                            continue;
                        }
                        Ok(None) => {}
                        Err(error) => eprintln!(
                            "warning: could not resolve managed plugin {}: {error}",
                            package.manifest.id
                        ),
                    }
                }
            }
            plugins.push(PluginUpdatePlan {
                id,
                name: package.manifest.name,
                current_version: current,
                target_version: None,
                action: if compatible {
                    UpdateAction::Unchanged
                } else {
                    UpdateAction::Incompatible
                },
                repository: package.manifest.repository,
                release_tag: None,
                environment: Some(package.manifest.environment.version),
                artifact: None,
                official: false,
            });
            continue;
        };

        let managed_official = registry.plugins.get(&id).is_some_and(|entry| {
            entry.source == "official"
                && entry.repository.as_deref() == Some(catalog.repository.as_str())
        });
        if !managed_official {
            let compatible =
                requirement_matches(&package.manifest.environment.version, &manifest.version);
            plugins.push(PluginUpdatePlan {
                id,
                name: package.manifest.name,
                current_version: current,
                target_version: None,
                action: if compatible {
                    UpdateAction::Unchanged
                } else {
                    UpdateAction::Incompatible
                },
                repository: package.manifest.repository,
                release_tag: None,
                environment: Some(package.manifest.environment.version),
                artifact: None,
                official: false,
            });
            continue;
        }

        let target_version = normalize_version(&catalog.version)?;
        let mut action = classify_versions(
            &normalize_version(&package.manifest.version)?,
            &target_version,
        );
        let mut environment = Some(package.manifest.environment.version.clone());
        let mut plugin_artifact = None;
        let repository = Some(catalog.repository.clone());
        let release_tag = Some(release_tag(&target_version));
        let installed_compatible =
            requirement_matches(&package.manifest.environment.version, &manifest.version);
        if app_only {
            action = UpdateAction::Skipped;
        } else if action != UpdateAction::Current {
            let plugin_url = release_manifest_url(&catalog.repository, release_tag.as_deref())?;
            let plugin_text = client.get_text(permission, &plugin_url)?;
            let plugin_manifest = PluginReleaseManifest::parse(&plugin_text, &catalog.repository)?;
            if plugin_manifest.version != target_version {
                bail!(
                    "official plugin {} release metadata resolved to {}, expected {}",
                    id,
                    plugin_manifest.version,
                    target_version
                );
            }
            environment = plugin_manifest.environment().map(str::to_string);
            if let Some(requirement) = environment.as_deref()
                && !requirement_matches(requirement, &manifest.version)
            {
                action = UpdateAction::Incompatible;
            } else {
                plugin_artifact = Some(
                    plugin_manifest
                        .artifact(&host_plugin_artifact_key()?)?
                        .clone(),
                );
            }
        } else if !installed_compatible {
            action = UpdateAction::Incompatible;
        }
        plugins.push(PluginUpdatePlan {
            id,
            name: catalog
                .name
                .clone()
                .unwrap_or_else(|| super::manifest::title_case_id(&package.manifest.id)),
            current_version: current,
            target_version: Some(catalog.version.clone()),
            action,
            repository,
            release_tag,
            environment,
            artifact: plugin_artifact,
            official: true,
        });
    }

    Ok(UpdatePlan {
        app: AppUpdatePlan {
            current_version,
            target_version: manifest.version.clone(),
            action: app_action,
            distribution,
            artifact,
        },
        plugins,
    })
}

fn execute_plan(
    plugin_dir: &Path,
    plan: &UpdatePlan,
    client: &ReleaseClient,
    permission: &NetworkPermission,
    app_only: bool,
    exact_target: bool,
) -> Result<()> {
    let mut staged_plugins = stage_plugin_updates(plugin_dir, plan, client, permission, app_only)?;
    let mut app_stage = None;
    let helper_required = plan.app.action != UpdateAction::Current
        && matches!(
            plan.app.distribution,
            Distribution::MacosBundle | Distribution::WindowsPortable
        );
    if plan.app.action != UpdateAction::Current {
        match plan.app.distribution {
            Distribution::AppImage => {
                let artifact = plan
                    .app
                    .artifact
                    .as_ref()
                    .context("release has no AppImage artifact")?;
                let bytes = download_verified(client, permission, artifact)?;
                let path = appimage_stage_path()?;
                write_bytes_and_sync(&path, &bytes)?;
                set_executable(&path)?;
                app_stage = Some(path);
            }
            Distribution::Flatpak => {
                let artifact = plan
                    .app
                    .artifact
                    .as_ref()
                    .context("release has no Flatpak artifact")?;
                let bytes = download_verified(client, permission, artifact)?;
                let path = flatpak_destination(&plan.app.target_version)?;
                write_bytes_and_sync(&path, &bytes)?;
                println!(
                    "The verified Flatpak update was downloaded to {}.\nThis EOV build is distributed as a standalone Flatpak bundle, so the host installation must be updated outside the sandbox.\nInstall it from the host with:\n    flatpak install --user --bundle {}",
                    path.display(),
                    path.display()
                );
            }
            Distribution::Cargo => {
                if !cargo_installation_is_owned()? {
                    discard_staged_plugins(&mut staged_plugins);
                    println!(
                        "EOV cannot safely determine how this application installation is managed. No application files were changed."
                    );
                    return Ok(());
                }
                if cfg!(target_os = "windows") {
                    let helper = std::env::current_exe()?
                        .canonicalize()?
                        .parent()
                        .context("Cargo-installed EOV executable has no parent directory")?
                        .join("eov-update-helper.exe");
                    if !helper.is_file() {
                        discard_staged_plugins(&mut staged_plugins);
                        bail!(
                            "Cargo-managed EOV installation does not contain eov-update-helper.exe"
                        );
                    }
                    let (plugin_replacements, registry) =
                        build_helper_plugin_transaction(plugin_dir, &staged_plugins)?;
                    let registry = registry.map(|stage| HelperRegistryReplacement {
                        staged: stage.staged,
                        target: stage.target,
                        backup: stage.backup,
                    });
                    let transaction_path = write_helper_transaction(HelperTransaction {
                        schema: 1,
                        nonce: unique_suffix(),
                        pid: std::process::id(),
                        mode: "cargo-install".to_string(),
                        current_tree: None,
                        staged_tree: None,
                        backup_tree: None,
                        cargo_version: Some(plan.app.target_version.to_string()),
                        relaunch: Some(std::env::current_exe()?.canonicalize()?),
                        plugins: plugin_replacements,
                        registry,
                    })?;
                    launch_update_helper(&helper, &transaction_path)?;
                    println!("EOV staged a Cargo update and will restart to finish the update.");
                    return Ok(());
                }
                if let Err(error) = run_cargo_install(&plan.app.target_version) {
                    discard_staged_plugins(&mut staged_plugins);
                    return Err(error);
                }
            }
            Distribution::Unknown => {
                println!(
                    "EOV cannot safely determine how this application installation is managed. No application files were changed."
                );
                return Ok(());
            }
            Distribution::MacosBundle | Distribution::WindowsPortable => {
                if plan.app.distribution == Distribution::MacosBundle {
                    if let Some(bundle) = macos_bundle_path().ok() {
                        match super::homebrew::probe(&bundle)? {
                            super::homebrew::MacosInstallation::Homebrew(installation) => {
                                if let Err(error) = super::homebrew::update_owned(
                                    &installation,
                                    exact_target.then_some(&plan.app.target_version),
                                ) {
                                    discard_staged_plugins(&mut staged_plugins);
                                    return Err(error);
                                }
                                apply_staged_plugins(plugin_dir, &mut staged_plugins)?;
                                println!("Homebrew updated EOV to {}.", plan.app.target_version);
                                return Ok(());
                            }
                            super::homebrew::MacosInstallation::Ambiguous(reason) => {
                                discard_staged_plugins(&mut staged_plugins);
                                bail!(
                                    "Homebrew ownership is ambiguous: {reason}. No application files were changed."
                                );
                            }
                            super::homebrew::MacosInstallation::ManualBundle => {}
                        }
                    }
                }
                let artifact = plan
                    .app
                    .artifact
                    .as_ref()
                    .context("release has no portable application artifact")?;
                let bytes = download_verified(client, permission, artifact)?;
                app_stage = Some(stage_archive_update(plan.app.distribution, &bytes)?);
            }
        }
    }

    if helper_required {
        let (current_tree, staged_tree, helper_path, relaunch) = helper_application_paths(
            plan.app.distribution,
            app_stage.as_ref().context("missing staged application")?,
        )?;
        let (plugin_replacements, registry_stage) =
            build_helper_plugin_transaction(plugin_dir, &staged_plugins)?;
        let transaction_path = write_helper_transaction(HelperTransaction {
            schema: 1,
            nonce: unique_suffix(),
            pid: std::process::id(),
            mode: "replace-tree".to_string(),
            current_tree: Some(current_tree),
            staged_tree: Some(staged_tree),
            backup_tree: Some(archive_backup_path(
                plan.app.distribution,
                app_stage.as_ref().unwrap(),
            )?),
            cargo_version: None,
            relaunch: Some(relaunch),
            plugins: plugin_replacements,
            registry: registry_stage.map(|stage| HelperRegistryReplacement {
                staged: stage.staged,
                target: stage.target,
                backup: stage.backup,
            }),
        })?;
        launch_update_helper(&helper_path, &transaction_path)?;
        println!("EOV staged a verified replacement and will restart to finish the update.");
        return Ok(());
    }

    if let Some(stage) = app_stage {
        if plan.app.distribution == Distribution::AppImage {
            apply_appimage_stage(&stage)?;
        }
    }
    apply_staged_plugins(plugin_dir, &mut staged_plugins)?;
    println!("Update applied successfully. Restart EOV if it did not relaunch automatically.");
    Ok(())
}

type StagedPlugin<'a> = (&'a PluginUpdatePlan, PathBuf, String);

fn stage_plugin_updates<'a>(
    plugin_dir: &Path,
    plan: &'a UpdatePlan,
    client: &ReleaseClient,
    permission: &NetworkPermission,
    app_only: bool,
) -> Result<Vec<StagedPlugin<'a>>> {
    let can_update_plugins = matches!(
        plan.app.distribution,
        Distribution::AppImage
            | Distribution::Cargo
            | Distribution::MacosBundle
            | Distribution::WindowsPortable
    ) || (plan.app.action == UpdateAction::Current
        && matches!(
            plan.app.distribution,
            Distribution::Flatpak | Distribution::Unknown
        ));
    if app_only || !can_update_plugins {
        return Ok(Vec::new());
    }
    fs::create_dir_all(plugin_dir)?;
    let mut staged = Vec::new();
    let result = (|| -> Result<()> {
        for plugin in &plan.plugins {
            let Some(artifact) = &plugin.artifact else {
                continue;
            };
            if !matches!(
                plugin.action,
                UpdateAction::Upgrade | UpdateAction::Downgrade
            ) {
                continue;
            }
            let bytes = download_verified(client, permission, artifact)?;
            let stage = plugin_dir.join(format!(
                ".eov-update-{}-{}.part",
                plugin.id,
                unique_suffix()
            ));
            write_bytes_and_sync(&stage, &bytes)?;
            let inspected = match inspect_staged_plugin_package(&stage, true) {
                Ok(inspected) => inspected,
                Err(error) => {
                    let _ = fs::remove_file(&stage);
                    return Err(anyhow::anyhow!(error.to_string()));
                }
            };
            if inspected.manifest.id != plugin.id
                || normalize_version(&inspected.manifest.version)?
                    != normalize_version(plugin.target_version.as_deref().unwrap_or("0.0.0"))?
            {
                let _ = fs::remove_file(&stage);
                bail!("downloaded plugin package does not match its release metadata");
            }
            if let Some(environment) = &plugin.environment
                && inspected.manifest.environment.version != *environment
            {
                let _ = fs::remove_file(&stage);
                bail!("downloaded plugin package has a mismatched EOV requirement");
            }
            if !requirement_matches(
                &inspected.manifest.environment.version,
                &plan.app.target_version,
            ) {
                let _ = fs::remove_file(&stage);
                bail!(
                    "downloaded plugin {} is incompatible with target EOV {}",
                    plugin.id,
                    plan.app.target_version
                );
            }
            staged.push((plugin, stage, artifact.sha256.clone()));
        }
        Ok(())
    })();
    if let Err(error) = result {
        discard_staged_plugins(&mut staged);
        return Err(error);
    }
    Ok(staged)
}

fn apply_staged_plugins(
    plugin_dir: &Path,
    staged_plugins: &mut Vec<StagedPlugin<'_>>,
) -> Result<()> {
    for (plugin, stage, sha256) in staged_plugins.drain(..) {
        let result = install_verified_stage(
            plugin_dir,
            &stage,
            if plugin.official {
                "official"
            } else {
                "github"
            },
            plugin.repository.clone(),
            plugin.release_tag.clone(),
            &sha256,
        );
        if result.is_err() {
            let _ = fs::remove_file(&stage);
        }
        result?;
    }
    Ok(())
}

fn build_helper_plugin_transaction(
    plugin_dir: &Path,
    staged_plugins: &[StagedPlugin<'_>],
) -> Result<(Vec<HelperPluginReplacement>, Option<plugins::RegistryStage>)> {
    let mut plugin_replacements = Vec::new();
    let mut registry_replacements = Vec::new();
    for (plugin, stage, sha256) in staged_plugins {
        let superseded = inspect_packages(plugin_dir)?
            .into_iter()
            .filter(|package| {
                package.manifest.id == plugin.id
                    && package.package_path != plugin_dir.join(format!("{}.eop", plugin.id))
                    && package.package_path != *stage
            })
            .map(|package| package.package_path)
            .collect();
        plugin_replacements.push(HelperPluginReplacement {
            staged: stage.clone(),
            target: plugin_dir.join(format!("{}.eop", plugin.id)),
            backup: plugin_dir.join(format!(".{}.eop.backup-{}", plugin.id, unique_suffix())),
            remove_after: superseded,
        });
        registry_replacements.push(RegistryReplacement {
            id: plugin.id.clone(),
            source: if plugin.official {
                "official"
            } else {
                "github"
            }
            .to_string(),
            repository: plugin.repository.clone(),
            release_tag: plugin.release_tag.clone(),
            version: plugin.target_version.clone().unwrap_or_default(),
            sha256: sha256.clone(),
        });
    }
    let registry = if registry_replacements.is_empty() {
        None
    } else {
        Some(plugins::stage_registry_update(
            plugin_dir,
            &registry_replacements,
        )?)
    };
    Ok((plugin_replacements, registry))
}

fn discard_staged_plugins(staged_plugins: &mut Vec<StagedPlugin<'_>>) {
    for (_, stage, _) in staged_plugins.drain(..) {
        let _ = fs::remove_file(stage);
    }
}

fn app_artifact(
    manifest: &EovReleaseManifest,
    distribution: Distribution,
) -> Result<Option<PlatformArtifact>> {
    let key = match distribution {
        Distribution::AppImage => Some(format!(
            "platform.linux.appimage.{}",
            super::manifest::Architecture::current()?.as_str()
        )),
        Distribution::Flatpak => Some(format!(
            "platform.linux.flatpak.{}",
            super::manifest::Architecture::current()?.as_str()
        )),
        Distribution::WindowsPortable => Some(format!(
            "platform.windows.{}",
            super::manifest::Architecture::current()?.as_str()
        )),
        Distribution::MacosBundle => Some(format!(
            "platform.macos.{}",
            super::manifest::Architecture::current()?.as_str()
        )),
        Distribution::Cargo | Distribution::Unknown => None,
    };
    key.map(|key| manifest.artifact(&key).cloned()).transpose()
}

fn stage_archive_update(distribution: Distribution, bytes: &[u8]) -> Result<PathBuf> {
    let current_tree = match distribution {
        Distribution::MacosBundle => macos_bundle_path()?,
        Distribution::WindowsPortable => std::env::current_exe()?
            .canonicalize()?
            .parent()
            .context("running EOV executable has no installation directory")?
            .to_path_buf(),
        _ => bail!(
            "archive staging is only supported for macOS bundles and Windows portable installs"
        ),
    };
    let parent = current_tree
        .parent()
        .context("application installation has no parent directory")?;
    ensure_directory_writable(&current_tree)?;
    ensure_directory_writable(parent)?;
    let staging_root = parent.join(format!(".eov-update-stage-{}", unique_suffix()));
    fs::create_dir(&staging_root)?;
    if let Err(error) = extract_zip_safely(bytes, &staging_root) {
        let _ = fs::remove_dir_all(&staging_root);
        return Err(error);
    }

    let staged_tree = match distribution {
        Distribution::MacosBundle => {
            let candidate = staging_root.join("eov.app");
            if !candidate.is_dir() || !candidate.join("Contents/MacOS/eov").is_file() {
                let _ = fs::remove_dir_all(&staging_root);
                bail!("staged macOS archive does not contain a valid eov.app bundle");
            }
            candidate
        }
        Distribution::WindowsPortable => {
            if !staging_root.join("eov.exe").is_file()
                || !staging_root.join("eov-update-helper.exe").is_file()
            {
                let _ = fs::remove_dir_all(&staging_root);
                bail!(
                    "staged Windows portable archive does not contain eov.exe and eov-update-helper.exe"
                );
            }
            staging_root
        }
        _ => unreachable!(),
    };
    Ok(staged_tree)
}

fn extract_zip_safely(bytes: &[u8], destination: &Path) -> Result<()> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).context("invalid application ZIP archive")?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let enclosed = entry
            .enclosed_name()
            .context("application ZIP contains an unsafe path")?
            .to_path_buf();
        if enclosed.as_os_str().is_empty() {
            bail!("application ZIP contains an empty path");
        }
        if entry
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            bail!("application ZIP contains a symlink entry");
        }
        let output = destination.join(&enclosed);
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if !entry.is_file() {
            bail!("application ZIP contains an unsupported special entry");
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&output)
            .with_context(|| format!("could not stage {}", output.display()))?;
        std::io::copy(&mut entry, &mut file)?;
        file.sync_all()?;
        #[cfg(unix)]
        if entry.unix_mode().is_some_and(|mode| mode & 0o111 != 0) {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&output)?.permissions();
            permissions.set_mode(mode_to_permissions(entry.unix_mode().unwrap()));
            fs::set_permissions(&output, permissions)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn mode_to_permissions(mode: u32) -> u32 {
    mode & 0o777
}

fn macos_bundle_path() -> Result<PathBuf> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let contents = executable
        .parent()
        .context("EOV executable has no parent")?;
    if contents.file_name().and_then(|name| name.to_str()) != Some("MacOS") {
        bail!("EOV is not running from a macOS application bundle");
    }
    let contents = contents
        .parent()
        .context("bundle has no Contents directory")?;
    if contents.file_name().and_then(|name| name.to_str()) != Some("Contents") {
        bail!("EOV is not running from a valid macOS application bundle");
    }
    Ok(contents
        .parent()
        .context("bundle has no application root")?
        .to_path_buf())
}

fn helper_application_paths(
    distribution: Distribution,
    staged_tree: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    match distribution {
        Distribution::MacosBundle => {
            if !cfg!(target_os = "macos") {
                bail!("macOS bundle updates are only available on macOS");
            }
            let current = macos_bundle_path()?;
            let helper = current.join("Contents/MacOS/eov-update-helper");
            let relaunch = current.join("Contents/MacOS/eov");
            if !helper.is_file() {
                bail!("the current macOS bundle does not contain eov-update-helper");
            }
            Ok((current, staged_tree.to_path_buf(), helper, relaunch))
        }
        Distribution::WindowsPortable => {
            if !cfg!(target_os = "windows") {
                bail!("Windows portable updates are only available on Windows");
            }
            let current = std::env::current_exe()?
                .canonicalize()?
                .parent()
                .context("running EOV executable has no installation directory")?
                .to_path_buf();
            let helper = current.join("eov-update-helper.exe");
            let relaunch = current.join("eov.exe");
            if !helper.is_file() {
                bail!(
                    "the current Windows portable installation does not contain eov-update-helper.exe"
                );
            }
            Ok((current, staged_tree.to_path_buf(), helper, relaunch))
        }
        _ => bail!("unsupported helper distribution"),
    }
}

fn archive_backup_path(distribution: Distribution, _staged_tree: &Path) -> Result<PathBuf> {
    let current = match distribution {
        Distribution::MacosBundle => macos_bundle_path()?,
        Distribution::WindowsPortable => std::env::current_exe()?
            .canonicalize()?
            .parent()
            .context("running EOV executable has no installation directory")?
            .to_path_buf(),
        _ => bail!("unsupported helper distribution"),
    };
    let parent = current
        .parent()
        .context("application installation has no parent")?;
    Ok(parent.join(format!(".eov-backup-{}", unique_suffix())))
}

fn write_helper_transaction(transaction: HelperTransaction) -> Result<PathBuf> {
    let directory = std::env::temp_dir().join("eov-update");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("transaction-{}.toml", transaction.nonce));
    let temporary = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(toml::to_string_pretty(&transaction)?.as_bytes())?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

fn launch_update_helper(helper: &Path, transaction: &Path) -> Result<()> {
    let temporary_helper = std::env::temp_dir().join(format!(
        "eov-update-helper-{}{}",
        unique_suffix(),
        if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        }
    ));
    fs::copy(helper, &temporary_helper).with_context(|| {
        format!(
            "could not stage updater helper outside the installation tree: {}",
            helper.display()
        )
    })?;
    Command::new(&temporary_helper)
        .args(["replace-tree", "--transaction"])
        .arg(transaction)
        .spawn()
        .context("could not launch the EOV updater helper")?;
    Ok(())
}

fn classify_versions(current: &Version, target: &Version) -> UpdateAction {
    match target.cmp(current) {
        std::cmp::Ordering::Greater => UpdateAction::Upgrade,
        std::cmp::Ordering::Less => UpdateAction::Downgrade,
        std::cmp::Ordering::Equal => UpdateAction::Current,
    }
}

fn requirement_matches(requirement: &str, version: &Version) -> bool {
    VersionReq::parse(requirement)
        .map(|requirement| requirement.matches(version))
        .unwrap_or(false)
}

fn resolve_third_party_target(
    client: &ReleaseClient,
    permission: &NetworkPermission,
    repository: &str,
    target_eov: &Version,
) -> Result<Option<RemotePackagePlan>> {
    let mut candidates = Vec::new();
    if let Ok(latest) = plugins::fetch_remote_plugin_plan(
        client, permission, repository, None, None, None, false, false,
    ) {
        candidates.push(latest);
    }

    let releases_url = format!("{repository}/releases");
    let api_url = releases_url.replace("https://github.com/", "https://api.github.com/repos/");
    let releases = client.get_json(permission, &api_url)?;
    let Some(releases) = releases.as_array() else {
        bail!("GitHub releases response for {repository} is not an array");
    };
    for release in releases {
        if release.get("draft").and_then(|value| value.as_bool()) == Some(true)
            || release.get("prerelease").and_then(|value| value.as_bool()) == Some(true)
        {
            continue;
        }
        let Some(tag) = release.get("tag_name").and_then(|value| value.as_str()) else {
            continue;
        };
        if candidates
            .iter()
            .any(|candidate| candidate.release_tag == tag)
        {
            continue;
        }
        if let Ok(candidate) = plugins::fetch_remote_plugin_plan(
            client,
            permission,
            repository,
            Some(tag),
            None,
            None,
            false,
            false,
        ) {
            candidates.push(candidate);
        }
    }
    candidates.retain(|candidate| {
        if !candidate.version.pre.is_empty() {
            return false;
        }
        candidate
            .environment
            .as_deref()
            .is_some_and(|requirement| requirement_matches(requirement, target_eov))
    });
    candidates.sort_by(|left, right| right.version.cmp(&left.version));
    Ok(candidates.into_iter().next())
}

fn print_plan(plan: &UpdatePlan, app_only: bool) {
    println!(
        "EOV\n  {} -> {}\n  Distribution: {:?}\n  Action: {}",
        plan.app.current_version,
        plan.app.target_version,
        plan.app.distribution,
        plan.app.action.as_str()
    );
    if !plan.plugins.is_empty() {
        println!("\nPlugins");
        for plugin in &plan.plugins {
            println!(
                "  {:<14} {} -> {}  {}{}",
                format!("{} ({})", plugin.name, plugin.id),
                plugin.current_version,
                plugin.target_version.as_deref().unwrap_or("-"),
                plugin.action.as_str(),
                if app_only { " (app-only)" } else { "" }
            );
        }
    }
}

fn requires_restart(distribution: &Distribution, action: UpdateAction) -> bool {
    action != UpdateAction::Current
        && matches!(
            distribution,
            Distribution::AppImage | Distribution::MacosBundle | Distribution::WindowsPortable
        )
}

fn inspect_packages(plugin_dir: &Path) -> Result<Vec<InspectedPluginPackage>> {
    if !plugin_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(plugin_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file() && path.extension().and_then(|value| value.to_str()) == Some("eop")
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths
        .into_iter()
        .filter_map(|path| inspect_plugin_package(&path, true).ok())
        .collect())
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

fn download_verified(
    client: &ReleaseClient,
    permission: &NetworkPermission,
    artifact: &PlatformArtifact,
) -> Result<Vec<u8>> {
    let bytes = client.get_bytes(permission, &artifact.url)?;
    let actual = sha256_bytes(&bytes);
    if !actual.eq_ignore_ascii_case(&artifact.sha256) {
        bail!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            artifact.url,
            artifact.sha256,
            actual
        );
    }
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn appimage_stage_path() -> Result<PathBuf> {
    let current = std::env::var_os("APPIMAGE")
        .context("APPIMAGE is not set; EOV cannot locate the running AppImage")?;
    let current = PathBuf::from(current)
        .canonicalize()
        .with_context(|| "APPIMAGE does not resolve to an existing file")?;
    if !current.is_file() {
        bail!(
            "APPIMAGE does not identify a regular file: {}",
            current.display()
        );
    }
    ensure_directory_writable(
        current
            .parent()
            .context("AppImage has no parent directory")?,
    )?;
    Ok(current.with_file_name(format!(".eov.AppImage.update-{}", unique_suffix())))
}

fn apply_appimage_stage(stage: &Path) -> Result<()> {
    let current = std::env::var_os("APPIMAGE")
        .context("APPIMAGE is not set; EOV cannot locate the running AppImage")?;
    let current = PathBuf::from(current)
        .canonicalize()
        .with_context(|| "APPIMAGE does not resolve to an existing file")?;
    atomic_replace_file(&current, stage)
}

fn atomic_replace_file(current: &Path, stage: &Path) -> Result<()> {
    let backup = current.with_file_name(format!(".eov.AppImage.backup-{}", unique_suffix()));
    fs::rename(&current, &backup).with_context(|| {
        format!(
            "EOV cannot replace {} as the current user",
            current.display()
        )
    })?;
    if let Err(error) = fs::rename(stage, &current) {
        let _ = fs::rename(&backup, &current);
        return Err(error.into());
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

fn ensure_directory_writable(directory: &Path) -> Result<()> {
    let probe = directory.join(format!(".eov-write-probe-{}", unique_suffix()));
    let result = OpenOptions::new().create_new(true).write(true).open(&probe);
    match result {
        Ok(_) => {
            let _ = fs::remove_file(probe);
            Ok(())
        }
        Err(error) => bail!(
            "EOV cannot replace files in {} as the current user: {error}. No privilege escalation was attempted.",
            directory.display()
        ),
    }
}

fn flatpak_destination(version: &Version) -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    let directory = home.join(".eov").join("updates");
    fs::create_dir_all(&directory)?;
    Ok(directory.join(format!("eov-v{version}.flatpak")))
}

fn set_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
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

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

fn cargo_installation_is_owned() -> Result<bool> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".cargo")));
    let Some(cargo_home) = cargo_home else {
        return Ok(false);
    };
    let binary = cargo_home.join("bin").join(if cfg!(target_os = "windows") {
        "eov.exe"
    } else {
        "eov"
    });
    Ok(cargo_installation_matches(&executable, &binary))
}

fn cargo_installation_matches(executable: &Path, installed_binary: &Path) -> bool {
    installed_binary.exists() && installed_binary.canonicalize().ok().as_deref() == Some(executable)
}

fn run_cargo_install(version: &Version) -> Result<()> {
    let status = Command::new("cargo")
        .args([
            "install",
            "eov",
            "--version",
            &version.to_string(),
            "--locked",
            "--force",
        ])
        .status()
        .context("failed to invoke cargo for the EOV update")?;
    if !status.success() {
        bail!("cargo install failed with status {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_exact_app_downgrade() {
        assert_eq!(
            classify_versions(
                &Version::parse("0.4.6").unwrap(),
                &Version::parse("0.4.5").unwrap()
            ),
            UpdateAction::Downgrade
        );
    }

    #[test]
    fn appimage_artifact_uses_linux_appimage_namespace() {
        let manifest = EovReleaseManifest {
            metadata: None,
            version: Version::parse("0.4.6").unwrap(),
            repository: super::super::manifest::EOV_REPOSITORY.into(),
            artifacts: std::collections::BTreeMap::from([(
                format!("platform.linux.appimage.{}", super::super::manifest::Architecture::current().unwrap().as_str()),
                PlatformArtifact {
                    version: "0.4.6".into(),
                    sha256: "a".repeat(64),
                    url: "https://github.com/eosin-platform/eov/releases/download/v0.4.6/eov.AppImage".into(),
                    environment: None,
                },
            )]),
            plugins: std::collections::BTreeMap::new(),
        };
        assert!(app_artifact(&manifest, Distribution::AppImage).is_ok());
    }

    #[test]
    fn appimage_swap_is_atomic_and_restores_on_failure() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("eov.AppImage");
        let staged = temp.path().join("staged.AppImage");
        fs::write(&current, b"old").unwrap();
        fs::write(&staged, b"new").unwrap();
        atomic_replace_file(&current, &staged).unwrap();
        assert_eq!(fs::read(&current).unwrap(), b"new");

        let missing_stage = temp.path().join("missing.AppImage");
        assert!(atomic_replace_file(&current, &missing_stage).is_err());
        assert_eq!(fs::read(&current).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn appimage_stage_is_executable() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("eov.AppImage");
        fs::write(&path, b"appimage").unwrap();
        set_executable(&path).unwrap();
        assert_ne!(fs::metadata(path).unwrap().permissions().mode() & 0o111, 0);
    }

    #[test]
    fn application_zip_rejects_traversal() {
        use std::io::Write;
        let mut bytes = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut bytes);
            writer
                .start_file("../escape", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"bad").unwrap();
            writer.finish().unwrap();
        }
        let temp = tempfile::tempdir().unwrap();
        assert!(extract_zip_safely(bytes.get_ref(), temp.path()).is_err());
    }

    #[test]
    fn cargo_ownership_requires_matching_canonical_path() {
        let temp = tempfile::tempdir().unwrap();
        let installed = temp.path().join("eov");
        let other = temp.path().join("other-eov");
        fs::write(&installed, b"eov").unwrap();
        fs::write(&other, b"eov").unwrap();
        assert!(cargo_installation_matches(&installed, &installed));
        assert!(!cargo_installation_matches(&other, &installed));
    }
}
