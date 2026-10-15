//! Application updates: planning, the per-distribution strategies, offline
//! bundles and rollback.
//!
//! [`plan`] resolves everything an update would do (source and signature,
//! how this installation is managed, each plugin's target and why) without
//! changing anything; [`execute`] applies a plan; [`rollback_plan`] and
//! [`rollback`] restore the previous installation. These functions are the
//! library API a GUI updater calls, with its own `Prompter`
//! (`super::prompt`) in the [`UpdateContext`].
//!
//! Strategies:
//!
//! | Distribution | What `eov update` does |
//! |---|---|
//! | AppImage | downloads the verified image next to the running one and renames it over it; the old image is kept as `.<name>.previous` |
//! | Flatpak | downloads the verified bundle to `~/.eov/updates/` and prints the `flatpak install --user`/`--system` command |
//! | Cargo | `cargo install eov --version X --locked --force --root <root>` for the root that owns the binary |
//! | Homebrew-managed macOS bundle | `brew update`, then upgrades or swaps casks, then reads back the installed version |
//! | Manual macOS bundle / Windows portable folder | stages the verified archive next to the installation and hands over to `eov-update-helper` |
//!
//! Plugins are planned against the eov version they will run under. A
//! plugin that cannot be resolved, downloaded or verified is skipped with
//! its reason; only problems with eov's own artifact stop the update.

use super::appimage;
use super::cargo_install::{self, CargoEnvironment};
use super::compat::{self, Compatibility};
use super::flatpak;
use super::handoff;
use super::homebrew::{
    self, BrewEnvironment, CommandRunner, HomebrewInstallation, MacosInstallation, SystemRunner,
};
use super::manifest::{
    Architecture, EOV_REPOSITORY, PlatformArtifact, normalize_version, release_tag,
    validate_self_update_target,
};
use super::network::NetworkPermission;
use super::plugins::{
    self, Catalog, InstalledPackage, Inventory, PluginContext, Provenance, ResolvedPackage,
    StagedPackage, compare_versions,
};
use super::registry::{self, ManagedPlugin, PluginDirLock, SOURCE_GITHUB, SOURCE_OFFICIAL};
use super::rollback::{self, AppRollback, PluginRollback, RollbackRecord};
use super::signature::SignatureStatus;
use super::source::ReleaseSource;
use super::transaction::{
    HelperReport, Mode, PluginReplacement, REPORT_FILE, RegistryReplacement, Relaunch,
    TRANSACTION_SCHEMA, Transaction,
};
use crate::distribution::{self, Distribution};
use anyhow::{Context, Result, bail};
use semver::Version;
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Facts about the running installation (injectable for tests).
#[derive(Debug, Clone)]
pub struct Installation {
    /// The running executable, canonicalised.
    pub executable: PathBuf,
    /// `$APPIMAGE`, set by the AppImage runtime.
    pub appimage: Option<PathBuf>,
    /// The contents of `/.flatpak-info` inside a Flatpak sandbox.
    pub flatpak_info: Option<String>,
    pub home: Option<PathBuf>,
}

impl Installation {
    pub fn detect() -> Result<Self> {
        let executable = std::env::current_exe().context("cannot locate the eov executable")?;
        Ok(Self {
            executable: executable.canonicalize().unwrap_or(executable),
            appimage: std::env::var_os("APPIMAGE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            flatpak_info: fs::read_to_string("/.flatpak-info").ok(),
            home: dirs::home_dir(),
        })
    }
}

/// Everything an update needs from its environment.
pub struct UpdateContext {
    pub plugins: PluginContext,
    pub distribution: Distribution,
    pub updates_dir: PathBuf,
    pub installation: Installation,
    pub brew: Box<dyn CommandRunner>,
    pub brew_env: BrewEnvironment,
    pub cargo_env: CargoEnvironment,
    /// Start eov again after a helper-based update.
    pub relaunch: bool,
}

impl UpdateContext {
    /// The context of a CLI invocation.
    pub fn for_cli(plugin_dir: &Path, assume_yes: bool, json: bool) -> Result<Self> {
        Ok(Self {
            plugins: PluginContext::for_cli(plugin_dir, assume_yes, json)?,
            distribution: distribution::distribution(),
            updates_dir: crate::paths::updates_dir(),
            installation: Installation::detect()?,
            brew: Box::new(SystemRunner),
            brew_env: BrewEnvironment::from_process(),
            cargo_env: CargoEnvironment::from_process(),
            relaunch: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// What the user asked for.
#[derive(Debug, Clone, Default)]
pub struct UpdateRequest {
    /// An exact release (allows downgrades); the latest release otherwise.
    pub release: Option<String>,
    /// Update eov only, not its plugins.
    pub app_only: bool,
    /// An offline bundle (directory or its `release.toml`).
    pub from: Option<PathBuf>,
    /// Trust an offline bundle without a verified signature.
    pub allow_unsigned: bool,
    pub allow_network: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AppAction {
    Upgrade,
    Downgrade,
    Current,
    /// This build is newer than the latest release; nothing to do.
    NewerThanRelease,
}

impl AppAction {
    fn changes_app(self) -> bool {
        matches!(self, Self::Upgrade | Self::Downgrade)
    }
}

/// How this installation is updated.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Strategy {
    AppImage {
        path: PathBuf,
    },
    Flatpak {
        installation: flatpak::Installation,
    },
    Cargo {
        root: PathBuf,
    },
    Homebrew {
        cask: String,
        app_path: PathBuf,
        installed: String,
    },
    /// Windows portable folder or manually installed macOS bundle.
    ReplaceTree {
        tree: PathBuf,
    },
    NotManaged {
        reason: String,
    },
    DisabledByPolicy {
        reason: String,
    },
}

impl Strategy {
    pub fn describe(&self) -> String {
        match self {
            Self::AppImage { path } => format!(
                "replace {} in place (the previous image is kept for --rollback)",
                path.display()
            ),
            Self::Flatpak { .. } => {
                "download the verified Flatpak bundle for installation on the host".to_string()
            }
            Self::Cargo { root } => format!(
                "cargo install eov --locked --force --root {}",
                root.display()
            ),
            Self::Homebrew { cask, app_path, .. } => format!(
                "Homebrew ({cask} owns {}); brew update, then brew changes the cask",
                app_path.display()
            ),
            Self::ReplaceTree { tree } => format!(
                "exit and let eov-update-helper replace {} (the previous version is kept for --rollback)",
                tree.display()
            ),
            Self::NotManaged { reason } => format!("not managed by eov update: {reason}"),
            Self::DisabledByPolicy { reason } => format!("disabled: {reason}"),
        }
    }

    /// Whether executing the plan installs the new eov before plugins are
    /// activated (so plugins must be compatible with the target version).
    fn installs_app(&self) -> bool {
        matches!(
            self,
            Self::AppImage { .. }
                | Self::Cargo { .. }
                | Self::Homebrew { .. }
                | Self::ReplaceTree { .. }
        )
    }

    fn artifact_key(&self, distribution: Distribution, arch: Architecture) -> Option<String> {
        let arch = arch.as_str();
        match self {
            Self::AppImage { .. } => Some(format!("platform.linux.appimage.{arch}")),
            Self::Flatpak { .. } => Some(format!("platform.linux.flatpak.{arch}")),
            Self::ReplaceTree { .. } => Some(match distribution {
                Distribution::WindowsPortable => format!("platform.windows.{arch}"),
                _ => format!("platform.macos.{arch}"),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginGroup {
    Official,
    ThirdParty,
    Unmanaged,
}

impl PluginGroup {
    fn title(self) -> &'static str {
        match self {
            Self::Official => "Official plugins",
            Self::ThirdParty => "Third-party plugins",
            Self::Unmanaged => "Unmanaged plugins",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginAction {
    Upgrade,
    Downgrade,
    Keep,
    Skip,
    Error,
}

impl PluginAction {
    fn changes(self) -> bool {
        matches!(self, Self::Upgrade | Self::Downgrade)
    }
}

/// What happens to one installed plugin.
#[derive(Debug, Clone, Serialize)]
pub struct PluginStep {
    pub id: String,
    pub name: String,
    pub group: PluginGroup,
    pub current: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub action: PluginAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip)]
    package: Option<ResolvedPackage>,
}

/// Everything an update would do.
#[derive(Debug, Clone, Serialize)]
pub struct UpdatePlan {
    pub current_version: Version,
    pub target_version: Version,
    pub action: AppAction,
    pub distribution: &'static str,
    pub strategy: Strategy,
    pub source: String,
    pub signature: SignatureStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_notes: Option<String>,
    pub plugins: Vec<PluginStep>,
    pub notes: Vec<String>,
    /// The version plugins are planned against.
    pub plugin_target: Version,
    /// The administrator policy file in effect, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    #[serde(skip)]
    exact: bool,
    #[serde(skip)]
    artifact: Option<PlatformArtifact>,
    #[serde(skip)]
    homebrew: Option<HomebrewInstallation>,
}

impl UpdatePlan {
    /// Whether executing the plan changes the application.
    pub fn changes_app(&self) -> bool {
        self.action.changes_app()
            && matches!(
                self.strategy,
                Strategy::AppImage { .. }
                    | Strategy::Flatpak { .. }
                    | Strategy::Cargo { .. }
                    | Strategy::Homebrew { .. }
                    | Strategy::ReplaceTree { .. }
            )
    }

    pub fn changes_plugins(&self) -> bool {
        self.plugins.iter().any(|step| step.action.changes())
    }

    /// Nothing to do: no prompt, no "update applied" message.
    pub fn is_noop(&self) -> bool {
        !self.changes_app() && !self.changes_plugins()
    }
}

/// A plan together with the source it was resolved from.
pub struct PlannedUpdate {
    pub plan: UpdatePlan,
    pub source: ReleaseSource,
}

/// Resolves what `request` would do. Never changes anything.
pub fn plan(ctx: &mut UpdateContext, request: &UpdateRequest) -> Result<PlannedUpdate> {
    ctx.plugins.policy.ensure_valid()?;
    let requested = request
        .release
        .as_deref()
        .map(normalize_version)
        .transpose()?;
    if let Some(version) = &requested {
        validate_self_update_target(version)?;
    }
    let source = match &request.from {
        Some(from) => ReleaseSource::offline(from, request.allow_unsigned)?,
        None => ReleaseSource::Network {
            permission: NetworkPermission::require(
                request.allow_network,
                "This command needs release metadata and verified update artifacts from GitHub.",
                &mut ctx.plugins.interaction,
                &ctx.plugins.policy,
            )?,
        },
    };
    let catalog = source.catalog(
        &ctx.plugins.client,
        &ctx.plugins.verifier,
        requested.as_ref(),
    )?;
    let target = catalog.manifest.version.clone();
    validate_self_update_target(&target)?;
    let current = ctx.plugins.host.clone();
    let action = match target.cmp(&current) {
        std::cmp::Ordering::Equal => AppAction::Current,
        std::cmp::Ordering::Greater => AppAction::Upgrade,
        std::cmp::Ordering::Less if requested.is_some() => AppAction::Downgrade,
        std::cmp::Ordering::Less => AppAction::NewerThanRelease,
    };

    let mut notes = Vec::new();
    let (strategy, homebrew) = probe_strategy(ctx);
    if action.changes_app() {
        if let Strategy::Homebrew { .. } = &strategy {
            homebrew::check_target(&target)?;
        }
        if let Strategy::NotManaged { reason } = &strategy {
            notes.push(format!(
                "eov {target} is available, but this installation is not managed by eov update ({reason}). Download it from {EOV_REPOSITORY}/releases/tag/{}.",
                release_tag(&target)
            ));
        }
        if let Strategy::DisabledByPolicy { reason } = &strategy {
            notes.push(format!("eov {target} is available, but {reason}"));
        }
    }
    if action == AppAction::NewerThanRelease {
        notes.push(format!(
            "This build ({current}) is newer than the latest release ({target}); it is not downgraded. Use --release {target} to install that release."
        ));
    }
    let artifact = match strategy.artifact_key(ctx.distribution, ctx.plugins.platform.arch) {
        Some(key) if action.changes_app() => Some(catalog.manifest.artifact(&key)?.clone()),
        _ => None,
    };
    let plugin_target = if action.changes_app() && strategy.installs_app() {
        target.clone()
    } else {
        current.clone()
    };
    if action.changes_app() && !strategy.installs_app() && !request.app_only {
        notes.push(format!(
            "Plugins are kept compatible with the installed eov {current}; run eov update again after installing eov {target}."
        ));
    }

    let plugins = if request.app_only {
        Vec::new()
    } else {
        plan_plugins(ctx, &source, &catalog, &plugin_target)?
    };
    Ok(PlannedUpdate {
        plan: UpdatePlan {
            source: source.describe(&target),
            signature: catalog.signature.clone(),
            release_notes: (!source.is_offline())
                .then(|| format!("{EOV_REPOSITORY}/releases/tag/{}", release_tag(&target))),
            current_version: current,
            target_version: target,
            action,
            distribution: ctx.distribution.display_name(),
            strategy,
            plugins,
            notes,
            plugin_target,
            policy: ctx
                .plugins
                .policy
                .source
                .as_ref()
                .filter(|_| ctx.plugins.policy.is_managed())
                .map(|path| format!("managed by {}", path.display())),
            exact: requested.is_some(),
            artifact,
            homebrew,
        },
        source,
    })
}

fn probe_strategy(ctx: &UpdateContext) -> (Strategy, Option<HomebrewInstallation>) {
    if ctx.plugins.policy.disable_self_update {
        let reason = ctx
            .plugins
            .policy
            .ensure_self_update_allowed()
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "self-update is disabled by policy".into());
        return (Strategy::DisabledByPolicy { reason }, None);
    }
    let executable = &ctx.installation.executable;
    let strategy = match ctx.distribution {
        Distribution::AppImage => match &ctx.installation.appimage {
            Some(path) if path.is_file() => {
                let path = path.canonicalize().unwrap_or_else(|_| path.clone());
                match path.parent().map(ensure_directory_writable) {
                    Some(Ok(())) => Strategy::AppImage { path },
                    Some(Err(error)) => Strategy::NotManaged {
                        reason: error.to_string(),
                    },
                    None => Strategy::NotManaged {
                        reason: "the AppImage has no parent directory".into(),
                    },
                }
            }
            _ => Strategy::NotManaged {
                reason: "APPIMAGE is not set, so eov cannot locate the running AppImage".into(),
            },
        },
        Distribution::Flatpak => Strategy::Flatpak {
            installation: flatpak::detect(
                ctx.installation.flatpak_info.as_deref(),
                ctx.installation.home.as_deref(),
            ),
        },
        Distribution::Cargo => match cargo_install::probe(executable, &ctx.cargo_env) {
            Some(installation) if installation.from_crates_io => Strategy::Cargo {
                root: installation.root,
            },
            Some(installation) => Strategy::NotManaged {
                reason: format!(
                    "eov was installed with cargo from {}, not crates.io; rebuild it the same way",
                    installation.package
                ),
            },
            None => Strategy::NotManaged {
                reason: "no cargo install root (CARGO_INSTALL_ROOT, install.root, CARGO_HOME or ~/.cargo) records this binary".into(),
            },
        },
        Distribution::MacosBundle => match macos_bundle_path(executable) {
            Err(error) => Strategy::NotManaged {
                reason: error.to_string(),
            },
            Ok(bundle) => match homebrew::probe(&*ctx.brew, &ctx.brew_env, &bundle) {
                MacosInstallation::Homebrew(installation) => {
                    let strategy = Strategy::Homebrew {
                        cask: installation.cask.full_token.clone(),
                        app_path: installation.app_path.clone(),
                        installed: installation.cask.installed.clone(),
                    };
                    return (strategy, Some(installation));
                }
                MacosInstallation::ManualBundle => Strategy::ReplaceTree { tree: bundle },
                MacosInstallation::Ambiguous(reason) => Strategy::NotManaged {
                    reason: format!("Homebrew ownership is ambiguous: {reason}"),
                },
            },
        },
        Distribution::WindowsPortable => match executable.parent() {
            Some(tree) if tree.join("eov.exe").is_file() => Strategy::ReplaceTree {
                tree: tree.to_path_buf(),
            },
            _ => Strategy::NotManaged {
                reason: "eov.exe is not in a portable installation folder".into(),
            },
        },
        Distribution::Unknown => Strategy::NotManaged {
            reason: "this build does not record how it was packaged".into(),
        },
    };
    (strategy, None)
}

fn plan_plugins(
    ctx: &UpdateContext,
    source: &ReleaseSource,
    catalog: &Catalog,
    plugin_target: &Version,
) -> Result<Vec<PluginStep>> {
    let inventory = plugins::inventory(&ctx.plugins.plugin_dir, &[], &ctx.plugins.host)?;
    let mut steps = Vec::new();
    for id in inventory.ids() {
        let Some(installed) = inventory.representative(&id) else {
            continue;
        };
        let step = match plan_plugin(ctx, source, catalog, &inventory, installed, plugin_target) {
            Ok(step) => step,
            Err(error) => PluginStep {
                id: id.clone(),
                name: installed.name.clone(),
                group: group_of(installed),
                current: installed.version.clone(),
                target: None,
                action: PluginAction::Error,
                reason: Some(format!("{error:#}")),
                source: Some(installed.provenance.describe()),
                package: None,
            },
        };
        steps.push(step);
    }
    steps.sort_by(|left, right| (left.group as u8, &left.id).cmp(&(right.group as u8, &right.id)));
    Ok(steps)
}

fn group_of(package: &InstalledPackage) -> PluginGroup {
    match package.provenance.kind.as_str() {
        SOURCE_OFFICIAL => PluginGroup::Official,
        SOURCE_GITHUB => PluginGroup::ThirdParty,
        _ => PluginGroup::Unmanaged,
    }
}

fn plan_plugin(
    ctx: &UpdateContext,
    source: &ReleaseSource,
    catalog: &Catalog,
    inventory: &Inventory,
    installed: &InstalledPackage,
    plugin_target: &Version,
) -> Result<PluginStep> {
    let group = group_of(installed);
    let mut step = PluginStep {
        id: installed.id.clone(),
        name: installed.name.clone(),
        group,
        current: installed.version.clone(),
        target: None,
        action: PluginAction::Keep,
        reason: None,
        source: Some(installed.provenance.describe()),
        package: None,
    };
    let installed_compatible = compat::check(
        &installed.environment,
        compat::api_requirement(&installed.raw_manifest).as_deref(),
        plugin_target,
        &compat::host_api_version(),
    );
    let keep_reason = |otherwise: &str| match &installed_compatible {
        Compatibility::Compatible => Some(otherwise.to_string()),
        Compatibility::Incompatible(reason) => {
            Some(format!("will not load with eov {plugin_target}: {reason}"))
        }
    };
    let resolved = match group {
        PluginGroup::Official => {
            let Some(entry) = catalog.manifest.plugins.get(&installed.id) else {
                step.reason = keep_reason(&format!(
                    "not in the eov {} plugin catalog",
                    catalog.manifest.version
                ));
                return Ok(step);
            };
            let pinned = entry
                .artifacts
                .contains_key(&ctx.plugins.platform.plugin_artifact_key());
            let mut resolved = match source.permission() {
                Some(permission) => plugins::resolve_official(
                    &ctx.plugins,
                    permission,
                    catalog,
                    &installed.id,
                    None,
                )?,
                None if pinned => resolve_offline_official(ctx, catalog, &installed.id)?,
                None => {
                    step.action = PluginAction::Skip;
                    step.reason = Some(
                        "the offline bundle's catalog does not pin this platform's package".into(),
                    );
                    return Ok(step);
                }
            };
            source.localize(&mut resolved)?;
            resolved
        }
        PluginGroup::ThirdParty => {
            let Some(permission) = source.permission() else {
                step.action = PluginAction::Skip;
                step.reason = Some("third-party plugins are not part of offline bundles".into());
                return Ok(step);
            };
            let repository = installed
                .provenance
                .repository
                .as_deref()
                .context("the managed GitHub plugin has no repository")?;
            match plugins::resolve_github_compatible(
                &ctx.plugins,
                permission,
                repository,
                &installed.id,
                plugin_target,
            )? {
                Some(resolved) => resolved,
                None if installed_compatible.is_compatible() => {
                    step.reason = Some("no newer compatible release".into());
                    return Ok(step);
                }
                None => {
                    step.action = PluginAction::Skip;
                    step.reason = Some(format!(
                        "no release in {repository} supports eov {plugin_target}; it will not load"
                    ));
                    return Ok(step);
                }
            }
        }
        PluginGroup::Unmanaged => {
            step.reason = keep_reason(if installed.provenance.kind == registry::SOURCE_LOCAL {
                "installed from a local package; install a newer package to update it"
            } else {
                "copied into the plugin directory by hand; update it from its source"
            });
            return Ok(step);
        }
    };
    step.target = Some(resolved.version.to_string());
    if let Some(environment) = &resolved.environment
        && let Compatibility::Incompatible(reason) = compat::check(
            environment,
            None,
            plugin_target,
            &compat::host_api_version(),
        )
    {
        step.action = PluginAction::Skip;
        step.reason = Some(reason);
        return Ok(step);
    }
    let duplicates = inventory.with_id(&installed.id).len() > 1;
    step.action = match compare_versions(&resolved.version.to_string(), &installed.version) {
        std::cmp::Ordering::Greater => PluginAction::Upgrade,
        std::cmp::Ordering::Less => PluginAction::Downgrade,
        std::cmp::Ordering::Equal if duplicates => PluginAction::Upgrade,
        std::cmp::Ordering::Equal => {
            step.reason = Some("current".into());
            PluginAction::Keep
        }
    };
    step.package = Some(resolved);
    Ok(step)
}

/// Resolves an official plugin from an offline catalog's pinned artifact.
fn resolve_offline_official(
    ctx: &UpdateContext,
    catalog: &Catalog,
    id: &str,
) -> Result<ResolvedPackage> {
    let entry = &catalog.manifest.plugins[id];
    let artifact = entry
        .artifacts
        .get(&ctx.plugins.platform.plugin_artifact_key())
        .context("the catalog does not pin this platform's package")?;
    Ok(ResolvedPackage {
        id: id.to_string(),
        name: entry.name.clone().unwrap_or_else(|| id.to_string()),
        version: normalize_version(&entry.version)?,
        provenance: Provenance {
            kind: SOURCE_OFFICIAL.into(),
            repository: Some(entry.repository.clone()),
            release_tag: Some(entry.tag()),
        },
        environment: entry.environment.clone(),
        pinned_by_catalog: true,
        signature: Some(catalog.signature.clone()),
        sha256: artifact.sha256.to_ascii_lowercase(),
        artifact: artifact.file_name().unwrap_or("artifact").to_string(),
        fetch: plugins::PackageFetch::Download {
            url: artifact.url.clone(),
            sha256: artifact.sha256.to_ascii_lowercase(),
        },
    })
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// The result for one plugin.
#[derive(Debug, Clone, Serialize)]
pub struct PluginOutcome {
    pub id: String,
    /// `updated`, `staged` (applied by the helper after eov exits),
    /// `skipped` or `failed`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// What an update did.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateOutcome {
    /// `updated`, `restart-pending` (the helper finishes after eov exits),
    /// `downloaded` (Flatpak), or `unchanged`.
    pub app: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_version: Option<String>,
    pub plugins: Vec<PluginOutcome>,
    /// The process should exit now so the helper can finish.
    pub exit_for_helper: bool,
}

impl UpdateOutcome {
    fn new() -> Self {
        Self {
            app: "unchanged",
            message: None,
            installed_version: None,
            plugins: Vec::new(),
            exit_for_helper: false,
        }
    }

    fn failed_plugins(&self) -> Vec<&PluginOutcome> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.status == "failed")
            .collect()
    }
}

struct Staged {
    step: PluginStep,
    package: StagedPackage,
}

impl Staged {
    fn provenance(&self) -> Provenance {
        self.step
            .package
            .as_ref()
            .map(|package| package.provenance.clone())
            .expect("staged steps have packages")
    }
}

/// Applies a plan. Plugin failures are recorded in the outcome (the rest of
/// the update continues); errors with eov's own artifact are returned.
pub fn execute(ctx: &mut UpdateContext, planned: PlannedUpdate) -> Result<UpdateOutcome> {
    let PlannedUpdate { plan, source } = planned;
    let (_update_lock, _) =
        registry::acquire_lock_file(&ctx.updates_dir.join(".update.lock"), "updates in")?;
    let plugin_lock = PluginDirLock::acquire(&ctx.plugins.plugin_dir)?;
    let mut outcome = UpdateOutcome::new();

    // Stage every plugin first; a failure only affects that plugin.
    let mut staged = Vec::new();
    for step in &plan.plugins {
        if step.action == PluginAction::Error {
            outcome.plugins.push(PluginOutcome {
                id: step.id.clone(),
                status: "failed",
                message: step.reason.clone(),
            });
            continue;
        }
        let Some(package) = step.package.as_ref().filter(|_| step.action.changes()) else {
            continue;
        };
        match plugins::stage_package(
            &ctx.plugins,
            source.permission(),
            package,
            &plan.plugin_target,
        ) {
            Ok(package) => staged.push(Staged {
                step: step.clone(),
                package,
            }),
            Err(error) => outcome.plugins.push(PluginOutcome {
                id: step.id.clone(),
                status: "failed",
                message: Some(format!("{error:#}")),
            }),
        }
    }

    let previous_version = plan.current_version.to_string();
    let mut record = RollbackRecord {
        schema: rollback::RECORD_SCHEMA,
        created_at: unix_now(),
        previous_version: previous_version.clone(),
        current_version: plan.target_version.to_string(),
        app: None,
        plugins: Vec::new(),
    };
    let mut plugin_target = plan.plugin_target.clone();

    if plan.changes_app() {
        let artifact = plan.artifact.as_ref();
        match &plan.strategy {
            Strategy::AppImage { path } => {
                let artifact = artifact.context("the release has no AppImage for this platform")?;
                let directory = path.parent().context("the AppImage has no directory")?;
                let image = source.fetch(&ctx.plugins.client, artifact, directory)?;
                if let Some(previous) = appimage::replace(path, image)? {
                    record.app = Some(AppRollback::AppImage {
                        path: path.clone(),
                        previous,
                    });
                }
                outcome.app = "updated";
                outcome.installed_version = Some(plan.target_version.to_string());
                outcome.message = Some(format!(
                    "{} runs eov {} from its next start.",
                    path.display(),
                    plan.target_version
                ));
            }
            Strategy::Flatpak { installation } => {
                let artifact =
                    artifact.context("the release has no Flatpak bundle for this platform")?;
                let bundle = source.fetch(&ctx.plugins.client, artifact, &ctx.updates_dir)?;
                let destination = ctx.updates_dir.join(
                    artifact
                        .file_name()
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("eov-v{}.flatpak", plan.target_version)),
                );
                // Atomically replaces a bundle left by an earlier run.
                bundle
                    .persist(&destination)
                    .map_err(|error| error.error)
                    .with_context(|| format!("could not write {}", destination.display()))?;
                outcome.app = "downloaded";
                outcome.message = Some(format!(
                    "The verified Flatpak bundle was saved to {}.\nThe sandbox cannot update itself; install it on the host with:\n{}",
                    destination.display(),
                    flatpak::install_instructions(*installation, &destination)
                ));
            }
            Strategy::Cargo { root } => {
                if cfg!(windows) {
                    // The running executable cannot be overwritten.
                    return hand_over(
                        ctx,
                        &plan,
                        staged,
                        plugin_lock,
                        HandOver::Cargo { root: root.clone() },
                        outcome,
                        record,
                    );
                }
                run_cargo_install(root, &plan.target_version)?;
                record.app = Some(AppRollback::Cargo {
                    root: root.clone(),
                    version: previous_version.clone(),
                });
                outcome.app = "updated";
                outcome.installed_version = Some(plan.target_version.to_string());
            }
            Strategy::Homebrew { .. } => {
                let installation = plan
                    .homebrew
                    .as_ref()
                    .context("the Homebrew installation is missing from the plan")?;
                let installed = homebrew::update(
                    &*ctx.brew,
                    &ctx.brew_env,
                    installation,
                    &plan.target_version,
                    plan.exact,
                )?;
                let installed_version = normalize_version(&installed)?;
                if installed_version != plan.target_version {
                    outcome.message = Some(format!(
                        "Homebrew installed eov {installed}, not {} (its tap may not have the release yet). Plugins were aligned with {installed}.",
                        plan.target_version
                    ));
                }
                record.app = Some(AppRollback::Homebrew {
                    tap: installation.tap().to_string(),
                    version: previous_version.clone(),
                });
                record.current_version = installed.clone();
                plugin_target = installed_version;
                outcome.app = "updated";
                outcome.installed_version = Some(installed);
            }
            Strategy::ReplaceTree { tree } => {
                let artifact = artifact
                    .context("the release has no application archive for this platform")?
                    .clone();
                return hand_over(
                    ctx,
                    &plan,
                    staged,
                    plugin_lock,
                    HandOver::Tree {
                        tree: tree.clone(),
                        artifact,
                        source: &source,
                    },
                    outcome,
                    record,
                );
            }
            Strategy::NotManaged { .. } | Strategy::DisabledByPolicy { .. } => {}
        }
    }

    // Activate plugins in this process.
    let store = rollback::plugin_store(&ctx.updates_dir);
    if !staged.is_empty() {
        let _ = fs::remove_dir_all(&store);
    }
    let registry_before = registry::load(&ctx.plugins.plugin_dir).unwrap_or_default();
    for staged in staged {
        let provenance = staged.provenance();
        let Staged { step, package } = staged;
        if let Compatibility::Incompatible(reason) = compat::check_manifest(
            &package.inspected.manifest,
            &package.inspected.raw_manifest,
            &plugin_target,
        ) {
            outcome.plugins.push(PluginOutcome {
                id: step.id.clone(),
                status: "skipped",
                message: Some(reason),
            });
            continue;
        }
        match plugins::activate_package(&ctx.plugins.plugin_dir, package, &provenance, Some(&store))
        {
            Ok(activated) => {
                let kept = activated
                    .replaced
                    .iter()
                    .find(|path| path.file_name() == activated.package_path.file_name())
                    .or_else(|| activated.replaced.first())
                    .and_then(|path| path.file_name())
                    .map(|name| store.join(name))
                    .filter(|path| path.is_file());
                record.plugins.push(PluginRollback {
                    id: step.id.clone(),
                    package: kept,
                    entry: registry_before.plugins.get(&step.id).cloned(),
                });
                outcome.plugins.push(PluginOutcome {
                    id: step.id.clone(),
                    status: "updated",
                    message: Some(format!("{} -> {}", step.current, activated.version)),
                });
            }
            Err(error) => outcome.plugins.push(PluginOutcome {
                id: step.id.clone(),
                status: "failed",
                message: Some(format!("{error:#}")),
            }),
        }
    }
    drop(plugin_lock);
    if record.app.is_some() || !record.plugins.is_empty() {
        rollback::save(&ctx.updates_dir, &record)?;
    }
    Ok(outcome)
}

/// What the update helper does after eov exits.
enum HandOver<'a> {
    Tree {
        tree: PathBuf,
        artifact: PlatformArtifact,
        source: &'a ReleaseSource,
    },
    Cargo {
        root: PathBuf,
    },
}

/// Stages the application archive (for tree swaps), prepares the helper
/// transaction including every staged plugin, and starts the helper.
fn hand_over(
    ctx: &mut UpdateContext,
    plan: &UpdatePlan,
    staged: Vec<Staged>,
    plugin_lock: PluginDirLock,
    work: HandOver<'_>,
    mut outcome: UpdateOutcome,
    mut record: RollbackRecord,
) -> Result<UpdateOutcome> {
    let nonce = new_nonce();
    let mut cleanup = Vec::new();
    let mut transaction = Transaction {
        schema: TRANSACTION_SCHEMA,
        nonce: nonce.clone(),
        pid: std::process::id(),
        mode: Mode::ReplaceTree,
        from_version: Some(plan.current_version.to_string()),
        to_version: Some(plan.target_version.to_string()),
        current_tree: None,
        staged_tree: None,
        previous_tree: None,
        stale_previous: None,
        cargo_version: None,
        cargo_root: None,
        relaunch: ctx.relaunch.then(|| Relaunch {
            program: ctx.installation.executable.clone(),
            args: Vec::new(),
        }),
        plugins: Vec::new(),
        registry: None,
        plugin_lock: Some(ctx.plugins.plugin_dir.join(registry::LOCK_FILENAME)),
        report: ctx.updates_dir.join(REPORT_FILE),
        rollback_record: None,
        rollback_path: Some(ctx.updates_dir.join(rollback::RECORD_FILE)),
        cleanup: Vec::new(),
    };
    let mut staging_dir = None;
    let helper_source = match work {
        HandOver::Tree {
            tree,
            artifact,
            source,
        } => {
            let parent = tree
                .parent()
                .context("the installation has no parent directory")?
                .to_path_buf();
            ensure_directory_writable(&tree)?;
            ensure_directory_writable(&parent)?;
            let archive = source.fetch(&ctx.plugins.client, &artifact, &parent)?;
            let staging = tempfile::Builder::new()
                .prefix(".eov-update-stage-")
                .tempdir_in(&parent)
                .context("could not create a staging directory next to the installation")?;
            extract_zip_safely(&archive, staging.path())?;
            drop(archive);
            let staged_tree = locate_staged_tree(ctx.distribution, staging.path())?;
            if ctx.distribution == Distribution::WindowsPortable {
                verify_portable_folder_is_owned(&tree, &staged_tree)?;
            }
            let suffix = if ctx.distribution == Distribution::MacosBundle {
                ".app"
            } else {
                ""
            };
            let previous_tree = parent.join(format!(".eov-previous-{nonce}{suffix}"));
            if let Ok(Some(RollbackRecord {
                app: Some(AppRollback::Tree { previous, .. }),
                ..
            })) = rollback::load(&ctx.updates_dir)
            {
                transaction.stale_previous = Some(previous);
            }
            record.app = Some(AppRollback::Tree {
                path: tree.clone(),
                previous: previous_tree.clone(),
            });
            cleanup.push(staging.path().to_path_buf());
            let helper = helper_path(ctx.distribution, &tree);
            transaction.current_tree = Some(tree);
            transaction.staged_tree = Some(staged_tree);
            transaction.previous_tree = Some(previous_tree);
            staging_dir = Some(staging);
            helper
        }
        HandOver::Cargo { root } => {
            transaction.mode = Mode::CargoInstall;
            transaction.cargo_version = Some(plan.target_version.to_string());
            transaction.cargo_root = Some(root.clone());
            record.app = Some(AppRollback::Cargo {
                root,
                version: plan.current_version.to_string(),
            });
            ctx.installation
                .executable
                .with_file_name("eov-update-helper.exe")
        }
    };

    // Plugins and registry, applied by the helper.
    let store = rollback::plugin_store(&ctx.updates_dir);
    if !staged.is_empty() {
        let _ = fs::remove_dir_all(&store);
    }
    let mut registry = registry::load(&ctx.plugins.plugin_dir)?;
    let inventory = plugins::inventory(&ctx.plugins.plugin_dir, &[], &ctx.plugins.host)?;
    let mut kept_paths = Vec::new();
    for staged in staged {
        let provenance = staged.provenance();
        let Staged { step, package } = staged;
        let id = step.id.clone();
        let target = ctx.plugins.plugin_dir.join(format!("{id}.eop"));
        record.plugins.push(PluginRollback {
            id: id.clone(),
            package: target.exists().then(|| store.join(format!("{id}.eop"))),
            entry: registry.plugins.get(&id).cloned(),
        });
        let version = package.inspected.manifest.version.clone();
        registry.plugins.insert(
            id.clone(),
            ManagedPlugin {
                source: provenance.kind.clone(),
                repository: provenance.repository.clone(),
                release_tag: provenance.release_tag.clone(),
                version: version.clone(),
                sha256: package.sha256.clone(),
                file: format!("{id}.eop"),
                installed_at: Some(unix_now()),
            },
        );
        let staged_path = package
            .path
            .keep()
            .context("could not keep a staged plugin package")?;
        kept_paths.push(staged_path.clone());
        transaction.plugins.push(PluginReplacement {
            staged: staged_path,
            backup: ctx
                .plugins
                .plugin_dir
                .join(format!(".{id}.eop.backup-{nonce}")),
            remove_after: inventory
                .with_id(&id)
                .into_iter()
                .map(|package| package.package_path.clone())
                .filter(|path| path != &target)
                .collect(),
            keep_as: Some(store.join(format!("{id}.eop"))),
            target,
        });
        outcome.plugins.push(PluginOutcome {
            id,
            status: "staged",
            message: Some(format!("{} -> {version} after eov exits", step.current)),
        });
    }
    if !transaction.plugins.is_empty() {
        let staged_registry = registry::stage(&ctx.plugins.plugin_dir, &registry)?
            .into_temp_path()
            .keep()
            .context("could not keep the staged plugin registry")?;
        kept_paths.push(staged_registry.clone());
        transaction.registry = Some(RegistryReplacement {
            staged: staged_registry,
            target: ctx.plugins.plugin_dir.join(registry::REGISTRY_FILENAME),
            backup: ctx
                .plugins
                .plugin_dir
                .join(format!(".eov-managed-backup-{nonce}.toml")),
        });
    }
    transaction.rollback_record = Some(rollback::render(&record)?);

    let launched = (|| -> Result<()> {
        let helper = handoff::prepare(&helper_source)?;
        cleanup.push(helper.dir.clone());
        transaction.cleanup = cleanup.clone();
        // The helper takes the plugin lock itself once eov has exited.
        drop(plugin_lock);
        handoff::launch(&helper, &transaction)
    })();
    if let Err(error) = launched {
        for path in kept_paths {
            let _ = fs::remove_file(path);
        }
        return Err(error);
    }
    if let Some(staging) = staging_dir {
        let _ = staging.keep();
    }
    outcome.app = "restart-pending";
    outcome.exit_for_helper = true;
    outcome.message = Some(match transaction.mode {
        Mode::CargoInstall => {
            "eov exits now; the update helper runs cargo install and restarts eov.".into()
        }
        _ => "eov exits now; the update helper replaces the installation and restarts eov.".into(),
    });
    Ok(outcome)
}

fn new_nonce() -> String {
    plugins::random_suffix()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

fn helper_path(distribution: Distribution, tree: &Path) -> PathBuf {
    match distribution {
        Distribution::MacosBundle => tree.join("Contents/MacOS/eov-update-helper"),
        _ => tree.join("eov-update-helper.exe"),
    }
}

/// Finds the installation tree inside an extracted release archive.
fn locate_staged_tree(distribution: Distribution, staging: &Path) -> Result<PathBuf> {
    match distribution {
        Distribution::MacosBundle => {
            let bundle = staging.join("eov.app");
            if !bundle.join("Contents/MacOS/eov").is_file() {
                bail!("the downloaded macOS archive does not contain a valid eov.app bundle");
            }
            Ok(bundle)
        }
        Distribution::WindowsPortable => {
            // Releases from 1.0 ship their files inside a top-level `eov/`
            // folder; older archives (down to 0.4.5, which has no helper)
            // stored them at the archive root.
            let nested = staging.join("eov");
            let tree = if nested.join("eov.exe").is_file() {
                nested
            } else {
                staging.to_path_buf()
            };
            if !tree.join("eov.exe").is_file() {
                bail!("the downloaded Windows archive does not contain eov.exe");
            }
            Ok(tree)
        }
        _ => bail!("tree updates are only used for macOS bundles and Windows portable folders"),
    }
}

/// Name of the manifest that lists every file shipped in a Windows portable
/// release. Written by the release workflow next to `eov.exe`.
pub(crate) const PORTABLE_MANIFEST: &str = "eov-portable.json";

/// Refuses to replace a Windows portable installation folder unless every
/// entry in it belongs to eov.
///
/// The portable update swaps the whole folder that contains `eov.exe`. When a
/// user extracted the release straight into a shared folder (Downloads, the
/// Desktop, a tools directory), that folder also holds unrelated files that a
/// swap would move away. An entry is accepted when it is part of the release
/// being installed, is listed in the current installation's
/// [`PORTABLE_MANIFEST`], or is an artifact earlier eov releases shipped.
pub(crate) fn verify_portable_folder_is_owned(current: &Path, staged: &Path) -> Result<()> {
    use std::collections::HashSet;
    use std::ffi::OsString;

    let lower = |name: &std::ffi::OsStr| name.to_string_lossy().to_ascii_lowercase();
    let mut owned: HashSet<String> = HashSet::new();
    for entry in fs::read_dir(staged).context("could not list the staged release")? {
        owned.insert(lower(&entry?.file_name()));
    }
    if let Ok(text) = fs::read_to_string(current.join(PORTABLE_MANIFEST))
        && let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(files) = manifest.get("files").and_then(|files| files.as_array())
    {
        for file in files.iter().filter_map(|file| file.as_str()) {
            if let Some(first) = Path::new(file).components().next() {
                owned.insert(lower(first.as_os_str()));
            }
        }
    }

    let is_legacy_artifact = |name: &str| {
        name == "eov.exe"
            || name == "eov.com"
            || name == "eov-update-helper.exe"
            || name == PORTABLE_MANIFEST
            || name == super::transaction::TREE_MARKER
            || name.ends_with(".dll")
            || name.starts_with("license")
            || name.starts_with("copying")
            || name.starts_with("readme")
            || name.starts_with("third-party")
            || name.starts_with(".eov-")
    };

    if !current.join("eov.exe").is_file() {
        bail!(
            "{} does not look like an eov installation folder (no eov.exe); refusing to replace it",
            current.display()
        );
    }

    let mut foreign: Vec<OsString> = Vec::new();
    for entry in fs::read_dir(current).context("could not list the installation folder")? {
        let name = entry?.file_name();
        let key = lower(&name);
        if !owned.contains(&key) && !is_legacy_artifact(&key) {
            foreign.push(name);
        }
    }
    if !foreign.is_empty() {
        foreign.sort();
        let examples: Vec<String> = foreign
            .iter()
            .take(5)
            .map(|name| name.to_string_lossy().into_owned())
            .collect();
        bail!(
            "{} contains {} item(s) that are not part of eov (for example: {}).\n\
             A portable update replaces this whole folder, so it is refused to protect those files.\n\
             Move eov.exe and its files into a folder of their own (for example C:\\Tools\\eov) and run the update again.",
            current.display(),
            foreign.len(),
            examples.join(", ")
        );
    }
    Ok(())
}

fn extract_zip_safely(archive: &Path, destination: &Path) -> Result<()> {
    let file =
        fs::File::open(archive).with_context(|| format!("could not open {}", archive.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .context("invalid application ZIP archive")?;
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
        if let Some(mode) = entry.unix_mode().filter(|mode| mode & 0o111 != 0) {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&output, fs::Permissions::from_mode(mode & 0o755))?;
        }
    }
    Ok(())
}

fn macos_bundle_path(executable: &Path) -> Result<PathBuf> {
    let macos = executable.parent().context("eov has no parent directory")?;
    if macos.file_name().and_then(|name| name.to_str()) != Some("MacOS") {
        bail!("eov is not running from a macOS application bundle");
    }
    let contents = macos
        .parent()
        .context("the bundle has no Contents directory")?;
    if contents.file_name().and_then(|name| name.to_str()) != Some("Contents") {
        bail!("eov is not running from a valid macOS application bundle");
    }
    Ok(contents
        .parent()
        .context("the bundle has no application root")?
        .to_path_buf())
}

fn ensure_directory_writable(directory: &Path) -> Result<()> {
    tempfile::Builder::new()
        .prefix(".eov-write-probe-")
        .tempfile_in(directory)
        .map(drop)
        .map_err(|error| {
            anyhow::anyhow!(
                "eov cannot replace files in {} as the current user: {error}. No privilege escalation was attempted.",
                directory.display()
            )
        })
}

fn run_cargo_install(root: &Path, version: &Version) -> Result<()> {
    let status = Command::new("cargo")
        .args([
            "install",
            "eov",
            "--version",
            &version.to_string(),
            "--locked",
            "--force",
        ])
        .arg("--root")
        .arg(root)
        .status()
        .context("failed to run cargo for the eov update")?;
    if !status.success() {
        bail!("cargo install failed with {status}; eov was not changed");
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// The report the update helper left, if any (removed once read). The GUI
/// can show it at startup.
pub fn take_helper_report(updates_dir: &Path) -> Option<HelperReport> {
    let path = updates_dir.join(REPORT_FILE);
    let report = toml::from_str(&fs::read_to_string(&path).ok()?).ok();
    let _ = fs::remove_file(&path);
    report
}

// ---------------------------------------------------------------------------
// Rollback
// ---------------------------------------------------------------------------

/// What `eov update --rollback` would restore.
#[derive(Debug, Clone, Serialize)]
pub struct RollbackPlan {
    pub current_version: String,
    pub previous_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<AppRollback>,
    pub plugins: Vec<String>,
    #[serde(skip)]
    record: RollbackRecord,
}

pub fn rollback_plan(ctx: &UpdateContext) -> Result<RollbackPlan> {
    let record = rollback::load(&ctx.updates_dir)?.context(
        "Nothing to roll back: no update replaced eov or its plugins since the rollback state was last used.",
    )?;
    Ok(RollbackPlan {
        current_version: record.current_version.clone(),
        previous_version: record.previous_version.clone(),
        app: record.app.clone(),
        plugins: record
            .plugins
            .iter()
            .map(|plugin| plugin.id.clone())
            .collect(),
        record,
    })
}

/// Restores the recorded previous state (and records the current one, so a
/// second rollback returns to it).
pub fn rollback(ctx: &mut UpdateContext, plan: RollbackPlan) -> Result<UpdateOutcome> {
    let (_update_lock, _) =
        registry::acquire_lock_file(&ctx.updates_dir.join(".update.lock"), "updates in")?;
    let record = plan.record;
    let mut swapped = RollbackRecord {
        schema: rollback::RECORD_SCHEMA,
        created_at: unix_now(),
        previous_version: record.current_version.clone(),
        current_version: record.previous_version.clone(),
        app: record.app.clone(),
        plugins: Vec::new(),
    };
    let mut outcome = UpdateOutcome::new();
    rollback_plugins(ctx, &record, &mut swapped, &mut outcome)?;

    match &record.app {
        None => {}
        Some(AppRollback::AppImage { path, previous }) => {
            appimage::swap(path, previous)?;
            outcome.app = "updated";
            outcome.installed_version = Some(record.previous_version.clone());
        }
        Some(AppRollback::Cargo { root, version }) => {
            if cfg!(windows) {
                bail!(
                    "On Windows, roll back a cargo installation with: cargo install eov --version {version} --locked --force --root {}",
                    root.display()
                );
            }
            run_cargo_install(root, &normalize_version(version)?)?;
            swapped.app = Some(AppRollback::Cargo {
                root: root.clone(),
                version: record.current_version.clone(),
            });
            outcome.app = "updated";
            outcome.installed_version = Some(version.clone());
        }
        Some(AppRollback::Homebrew { tap, version }) => {
            let bundle = macos_bundle_path(&ctx.installation.executable)?;
            let MacosInstallation::Homebrew(installation) =
                homebrew::probe(&*ctx.brew, &ctx.brew_env, &bundle)
            else {
                bail!("eov is no longer managed by Homebrew; install {tap}/eov@{version} manually");
            };
            let installed = homebrew::update(
                &*ctx.brew,
                &ctx.brew_env,
                &installation,
                &normalize_version(version)?,
                true,
            )?;
            swapped.app = Some(AppRollback::Homebrew {
                tap: tap.clone(),
                version: record.current_version.clone(),
            });
            outcome.app = "updated";
            outcome.installed_version = Some(installed);
        }
        Some(AppRollback::Tree { path, previous }) => {
            let parent = path.parent().context("the installation has no parent")?;
            let nonce = new_nonce();
            let suffix = if ctx.distribution == Distribution::MacosBundle {
                ".app"
            } else {
                ""
            };
            let new_previous = parent.join(format!(".eov-previous-{nonce}{suffix}"));
            swapped.app = Some(AppRollback::Tree {
                path: path.clone(),
                previous: new_previous.clone(),
            });
            let helper = handoff::prepare(&helper_path(ctx.distribution, path))?;
            let transaction = Transaction {
                schema: TRANSACTION_SCHEMA,
                nonce,
                pid: std::process::id(),
                mode: Mode::SwapTree,
                from_version: Some(record.current_version.clone()),
                to_version: Some(record.previous_version.clone()),
                current_tree: Some(path.clone()),
                staged_tree: Some(previous.clone()),
                previous_tree: Some(new_previous),
                stale_previous: None,
                cargo_version: None,
                cargo_root: None,
                relaunch: ctx.relaunch.then(|| Relaunch {
                    program: ctx.installation.executable.clone(),
                    args: Vec::new(),
                }),
                plugins: Vec::new(),
                registry: None,
                plugin_lock: None,
                report: ctx.updates_dir.join(REPORT_FILE),
                rollback_record: Some(rollback::render(&swapped)?),
                rollback_path: Some(ctx.updates_dir.join(rollback::RECORD_FILE)),
                cleanup: vec![helper.dir.clone()],
            };
            handoff::launch(&helper, &transaction)?;
            outcome.app = "restart-pending";
            outcome.exit_for_helper = true;
            outcome.message = Some(
                "eov exits now; the update helper restores the previous version and restarts eov."
                    .into(),
            );
            return Ok(outcome);
        }
    }
    rollback::save(&ctx.updates_dir, &swapped)?;
    Ok(outcome)
}

/// Swaps the kept plugin packages with the installed ones.
fn rollback_plugins(
    ctx: &UpdateContext,
    record: &RollbackRecord,
    swapped: &mut RollbackRecord,
    outcome: &mut UpdateOutcome,
) -> Result<()> {
    if record.plugins.is_empty() {
        return Ok(());
    }
    let _lock = PluginDirLock::acquire(&ctx.plugins.plugin_dir)?;
    let store = rollback::plugin_store(&ctx.updates_dir);
    let swap_store = store.with_file_name("plugins-swap");
    let _ = fs::remove_dir_all(&swap_store);
    fs::create_dir_all(&swap_store)?;
    let mut registry = registry::load(&ctx.plugins.plugin_dir)?;
    for plugin in &record.plugins {
        let target = ctx.plugins.plugin_dir.join(format!("{}.eop", plugin.id));
        let current_entry = registry.plugins.get(&plugin.id).cloned();
        let parked = swap_store.join(format!("{}.eop", plugin.id));
        let kept_current = target.exists() && fs::rename(&target, &parked).is_ok();
        let restored = match &plugin.package {
            Some(package) => fs::rename(package, &target).is_ok(),
            None => true,
        };
        if !restored && kept_current {
            // Put the current package back rather than leaving nothing.
            let _ = fs::rename(&parked, &target);
        }
        if restored {
            match &plugin.entry {
                Some(entry) => {
                    registry.plugins.insert(plugin.id.clone(), entry.clone());
                }
                None => {
                    registry.plugins.remove(&plugin.id);
                }
            }
        }
        outcome.plugins.push(PluginOutcome {
            id: plugin.id.clone(),
            status: if restored { "updated" } else { "failed" },
            message: (!restored).then(|| "the kept package is missing".to_string()),
        });
        if restored {
            swapped.plugins.push(PluginRollback {
                id: plugin.id.clone(),
                package: kept_current.then(|| store.join(format!("{}.eop", plugin.id))),
                entry: current_entry,
            });
        }
        let _ = crate::plugins::discovery::invalidate_plugin_package_cache(&target);
    }
    registry::save(&ctx.plugins.plugin_dir, &registry)?;
    let _ = fs::remove_dir_all(&store);
    fs::rename(&swap_store, &store)
        .with_context(|| format!("could not keep the replaced plugins in {}", store.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Options of `eov update`.
#[derive(Debug, Clone, Default)]
pub struct UpdateOptions {
    pub request: UpdateRequest,
    pub yes: bool,
    pub dry_run: bool,
    pub json: bool,
    pub rollback: bool,
    pub relaunch: bool,
}

#[derive(Serialize)]
struct UpdateReport<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_update: Option<&'a HelperReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<&'a UpdatePlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollback: Option<&'a RollbackPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<&'a UpdateOutcome>,
    dry_run: bool,
}

/// `eov update`. Returns whether eov must exit now for the helper.
pub fn run_cli(plugin_dir: &Path, options: &UpdateOptions) -> Result<bool> {
    let mut ctx = UpdateContext::for_cli(plugin_dir, options.yes, options.json)?;
    ctx.relaunch = options.relaunch;
    handoff::cleanup_stale(&std::env::temp_dir(), Duration::from_secs(24 * 60 * 60));
    let previous = take_helper_report(&ctx.updates_dir);
    if let Some(report) = &previous
        && !options.json
    {
        eprintln!("{}\n", report.summary());
    }
    run_with_context(&mut ctx, options, previous.as_ref())
}

pub fn run_with_context(
    ctx: &mut UpdateContext,
    options: &UpdateOptions,
    previous: Option<&HelperReport>,
) -> Result<bool> {
    let json = options.json;
    let report = |plan: Option<&UpdatePlan>,
                  rollback: Option<&RollbackPlan>,
                  outcome: Option<&UpdateOutcome>| {
        plugins::print_json(&UpdateReport {
            previous_update: previous,
            plan,
            rollback,
            outcome,
            dry_run: options.dry_run,
        })
    };
    if options.rollback {
        let plan = rollback_plan(ctx)?;
        if !json {
            print_rollback_plan(&plan);
        }
        if options.dry_run {
            if json {
                report(None, Some(&plan), None)?;
            } else {
                println!("\nDry run: nothing was changed.");
            }
            return Ok(false);
        }
        ctx.plugins.interaction.confirm(&format!(
            "Roll back eov {} to {}?",
            plan.current_version, plan.previous_version
        ))?;
        let shown = plan.clone();
        let outcome = rollback(ctx, plan)?;
        if json {
            report(None, Some(&shown), Some(&outcome))?;
        } else {
            print_outcome(&outcome);
        }
        return Ok(outcome.exit_for_helper);
    }

    let planned = plan(ctx, &options.request)?;
    if !json {
        print_plan(&planned.plan);
    }
    if planned.plan.is_noop() {
        if json {
            report(Some(&planned.plan), None, None)?;
        } else if planned.plan.action.changes_app() {
            println!(
                "\nNothing was changed: eov {} is available, but eov update cannot install it here (see the note above).",
                planned.plan.target_version
            );
        } else {
            println!(
                "\nNothing to update: eov {} and its plugins are up to date.",
                planned.plan.current_version
            );
        }
        return Ok(false);
    }
    if options.dry_run {
        if json {
            report(Some(&planned.plan), None, None)?;
        } else {
            println!("\nDry run: nothing was changed.");
        }
        return Ok(false);
    }
    let restart =
        matches!(planned.plan.strategy, Strategy::ReplaceTree { .. }) && planned.plan.changes_app();
    ctx.plugins.interaction.confirm(if restart {
        "eov will exit and the update helper will replace the application."
    } else {
        "Apply this update?"
    })?;
    let shown = planned.plan.clone();
    let outcome = execute(ctx, planned)?;
    if json {
        report(Some(&shown), None, Some(&outcome))?;
    } else {
        print_outcome(&outcome);
    }
    let failed = outcome.failed_plugins();
    if !failed.is_empty() {
        bail!(
            "{} plugin(s) could not be updated: {}",
            failed.len(),
            failed
                .iter()
                .map(|plugin| plugin.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(outcome.exit_for_helper)
}

fn print_plan(plan: &UpdatePlan) {
    println!("eov");
    println!(
        "  Installed:      {} ({})",
        plan.current_version, plan.distribution
    );
    println!(
        "  Release:        {} from {}",
        plan.target_version, plan.source
    );
    println!("  Signature:      {}", plan.signature.describe());
    println!(
        "  Action:         {}",
        match plan.action {
            AppAction::Upgrade => "upgrade",
            AppAction::Downgrade => "downgrade",
            AppAction::Current => "none (current)",
            AppAction::NewerThanRelease => "none (this build is newer)",
        }
    );
    println!("  Method:         {}", plan.strategy.describe());
    if let Some(policy) = &plan.policy {
        println!("  Policy:         {policy}");
    }
    if let Some(notes) = &plan.release_notes
        && plan.action.changes_app()
    {
        println!("  Release notes:  {notes}");
    }
    for group in [
        PluginGroup::Official,
        PluginGroup::ThirdParty,
        PluginGroup::Unmanaged,
    ] {
        let steps: Vec<_> = plan
            .plugins
            .iter()
            .filter(|step| step.group == group)
            .collect();
        if steps.is_empty() {
            continue;
        }
        println!("\n{}", group.title());
        for step in steps {
            let versions = match &step.target {
                Some(target) if step.action.changes() => format!("{} -> {target}", step.current),
                _ => step.current.clone(),
            };
            let action = match step.action {
                PluginAction::Upgrade => "upgrade",
                PluginAction::Downgrade => "downgrade",
                PluginAction::Keep => "keep",
                PluginAction::Skip => "skip",
                PluginAction::Error => "error",
            };
            let reason = step
                .reason
                .as_deref()
                .map(|reason| format!(": {reason}"))
                .unwrap_or_default();
            println!("  {:<14} {:<18} {action}{reason}", step.id, versions);
        }
    }
    if !plan.notes.is_empty() {
        println!();
        for note in &plan.notes {
            println!("Note: {note}");
        }
    }
}

fn print_rollback_plan(plan: &RollbackPlan) {
    println!(
        "Roll back eov {} to {}",
        plan.current_version, plan.previous_version
    );
    match &plan.app {
        Some(AppRollback::AppImage { previous, .. }) => {
            println!("  Application: restore {}", previous.display())
        }
        Some(AppRollback::Tree { previous, .. }) => println!(
            "  Application: restore {} (the helper swaps the folders)",
            previous.display()
        ),
        Some(AppRollback::Cargo { version, .. }) => {
            println!("  Application: cargo install eov --version {version}")
        }
        Some(AppRollback::Homebrew { tap, version }) => {
            println!("  Application: brew install --cask {tap}/eov@{version}")
        }
        None => println!("  Application: unchanged"),
    }
    if !plan.plugins.is_empty() {
        println!("  Plugins:     {}", plan.plugins.join(", "));
    }
}

fn print_outcome(outcome: &UpdateOutcome) {
    println!();
    if let Some(message) = &outcome.message {
        println!("{message}");
    }
    for plugin in &outcome.plugins {
        println!(
            "Plugin {}: {}{}",
            plugin.id,
            plugin.status,
            plugin
                .message
                .as_deref()
                .map(|message| format!(" ({message})"))
                .unwrap_or_default()
        );
    }
    if outcome.app == "updated" {
        println!("Update applied. Restart eov to use the new version.");
    }
}

#[cfg(test)]
mod tests {
    use super::super::homebrew::fake::FakeBrew;
    use super::super::plugins::tests::{
        HOST, catalog_text, catalog_url, context as plugin_context, gamepad_url, package_bytes,
        serve_gamepad, serve_github_plugin, sha256_hex,
    };
    use super::super::prompt::ScriptedPrompter;
    use super::super::signature::Verifier;
    use super::super::signature::test_keys::TEST_KEY;
    use super::super::transaction::Outcome;
    use super::super::transport::MockTransport;
    use super::*;
    use crate::plugins::discovery::package_cache::test_packages::write_package;
    use std::sync::Arc;

    fn context(
        root: &Path,
        transport: Arc<MockTransport>,
        distribution: Distribution,
    ) -> UpdateContext {
        UpdateContext {
            plugins: plugin_context(root, transport, ScriptedPrompter::new(false, &[]), true),
            distribution,
            updates_dir: root.join("updates"),
            installation: Installation {
                executable: root.join("bin").join("eov"),
                appimage: None,
                flatpak_info: None,
                home: Some(root.to_path_buf()),
            },
            brew: Box::new(FakeBrew::default()),
            brew_env: BrewEnvironment::default(),
            cargo_env: CargoEnvironment {
                cargo_install_root: None,
                cargo_home: Some(root.join("no-cargo").into_os_string()),
                home: None,
            },
            relaunch: false,
        }
    }

    fn request() -> UpdateRequest {
        UpdateRequest {
            allow_network: true,
            ..UpdateRequest::default()
        }
    }

    const LATEST: &str =
        "https://github.com/eosin-platform/eov/releases/latest/download/release.toml";

    /// Serves an eov `version` catalog with an AppImage and a gamepad pin.
    fn serve_release(transport: &MockTransport, root: &Path, version: &str, appimage: &[u8]) {
        let plugin = package_bytes(root, "gamepad", "1.0.5");
        let catalog = catalog_text(version, "1.0.5", &sha256_hex(&plugin))
            .replace(&"a".repeat(64), &sha256_hex(appimage));
        transport.respond(&catalog_url(version), 200, catalog.clone());
        transport.respond(LATEST, 200, catalog);
        transport.respond(
            &format!(
                "https://github.com/eosin-platform/eov/releases/download/v{version}/eov-v{version}-linux-x86_64.AppImage"
            ),
            200,
            appimage.to_vec(),
        );
        transport.respond(&gamepad_url("1.0.5"), 200, plugin);
    }

    fn install_official_gamepad(ctx: &mut UpdateContext, transport: &MockTransport, root: &Path) {
        serve_gamepad(transport, root, HOST, "1.0.1");
        plugins::install(
            &mut ctx.plugins,
            "gamepad",
            &plugins::InstallOptions {
                allow_network: true,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn current_release_is_a_no_op_without_a_prompt() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_release(&transport, temp.path(), HOST, b"image");
        let mut ctx = context(temp.path(), transport, Distribution::AppImage);
        // No -y and no terminal: any prompt would fail the run.
        ctx.plugins.interaction.assume_yes = false;
        let options = UpdateOptions {
            request: request(),
            ..UpdateOptions::default()
        };
        assert!(!run_with_context(&mut ctx, &options, None).unwrap());
    }

    #[test]
    fn a_build_newer_than_the_latest_release_is_not_downgraded() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_release(&transport, temp.path(), "0.9.0", b"image");
        let mut ctx = context(temp.path(), transport, Distribution::AppImage);
        let planned = plan(&mut ctx, &request()).unwrap();
        assert_eq!(planned.plan.action, AppAction::NewerThanRelease);
        assert!(!planned.plan.changes_app());
        // An explicit release does downgrade.
        let mut explicit = request();
        explicit.release = Some("0.9.0".into());
        let planned = plan(&mut ctx, &explicit).unwrap();
        assert_eq!(planned.plan.action, AppAction::Downgrade);
    }

    #[test]
    fn appimage_update_replaces_the_image_updates_plugins_and_can_roll_back() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(temp.path(), transport.clone(), Distribution::AppImage);
        install_official_gamepad(&mut ctx, &transport, temp.path());
        let apps = temp.path().join("apps");
        fs::create_dir_all(&apps).unwrap();
        let image = apps.join("eov.AppImage");
        fs::write(&image, b"old image").unwrap();
        ctx.installation.appimage = Some(image.clone());
        serve_release(&transport, temp.path(), "1.0.1", b"new image");

        let planned = plan(&mut ctx, &request()).unwrap();
        assert_eq!(planned.plan.action, AppAction::Upgrade);
        assert!(matches!(planned.plan.strategy, Strategy::AppImage { .. }));
        let gamepad = &planned.plan.plugins[0];
        assert_eq!(
            (gamepad.group, gamepad.action),
            (PluginGroup::Official, PluginAction::Upgrade)
        );
        let outcome = execute(&mut ctx, planned).unwrap();
        assert_eq!(outcome.app, "updated");
        assert_eq!(fs::read(&image).unwrap(), b"new image");
        assert_eq!(
            registry::load(&ctx.plugins.plugin_dir).unwrap().plugins["gamepad"].version,
            "1.0.5"
        );

        let plan = rollback_plan(&ctx).unwrap();
        assert_eq!(plan.previous_version, HOST);
        rollback(&mut ctx, plan).unwrap();
        assert_eq!(fs::read(&image).unwrap(), b"old image");
        assert_eq!(
            registry::load(&ctx.plugins.plugin_dir).unwrap().plugins["gamepad"].version,
            "1.0.1"
        );
        // Rolling back again returns to the update.
        let plan = rollback_plan(&ctx).unwrap();
        rollback(&mut ctx, plan).unwrap();
        assert_eq!(fs::read(&image).unwrap(), b"new image");
        assert_eq!(
            registry::load(&ctx.plugins.plugin_dir).unwrap().plugins["gamepad"].version,
            "1.0.5"
        );
    }

    #[test]
    fn a_failing_plugin_does_not_block_the_application_update() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(temp.path(), transport.clone(), Distribution::AppImage);
        install_official_gamepad(&mut ctx, &transport, temp.path());
        let image = temp.path().join("eov.AppImage");
        fs::write(&image, b"old").unwrap();
        ctx.installation.appimage = Some(image.clone());
        serve_release(&transport, temp.path(), "1.0.1", b"new");
        // The plugin artifact disappeared from GitHub.
        transport.respond(&gamepad_url("1.0.5"), 404, Vec::new());

        let options = UpdateOptions {
            request: request(),
            yes: true,
            ..UpdateOptions::default()
        };
        let error = run_with_context(&mut ctx, &options, None).unwrap_err();
        assert!(error.to_string().contains("gamepad"), "{error}");
        assert_eq!(fs::read(&image).unwrap(), b"new");
        assert_eq!(
            registry::load(&ctx.plugins.plugin_dir).unwrap().plugins["gamepad"].version,
            "1.0.1"
        );
    }

    #[test]
    fn plugins_are_grouped_with_reasons() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(temp.path(), transport.clone(), Distribution::Unknown);
        install_official_gamepad(&mut ctx, &transport, temp.path());
        serve_github_plugin(
            &transport,
            temp.path(),
            "https://github.com/someone/thing",
            "thing",
            true,
        );
        plugins::install(
            &mut ctx.plugins,
            "github.com/someone/thing",
            &plugins::InstallOptions {
                allow_network: true,
                ..Default::default()
            },
        )
        .unwrap();
        write_package(
            &ctx.plugins.plugin_dir.join("manual.eop"),
            "manual",
            "1.0.0",
        );
        serve_release(&transport, temp.path(), "1.0.1", b"new");

        let planned = plan(&mut ctx, &request()).unwrap();
        let groups: Vec<_> = planned
            .plan
            .plugins
            .iter()
            .map(|step| (step.id.as_str(), step.group, step.action))
            .collect();
        assert_eq!(
            groups,
            vec![
                ("gamepad", PluginGroup::Official, PluginAction::Upgrade),
                ("thing", PluginGroup::ThirdParty, PluginAction::Keep),
                ("manual", PluginGroup::Unmanaged, PluginAction::Keep),
            ]
        );
        // Unknown packaging: eov itself is not updated, and plugins stay
        // aligned with the installed version.
        assert!(matches!(planned.plan.strategy, Strategy::NotManaged { .. }));
        assert_eq!(planned.plan.plugin_target, Version::parse(HOST).unwrap());
        assert!(
            planned
                .plan
                .notes
                .iter()
                .any(|note| note.contains("not managed"))
        );
    }

    #[test]
    fn tampered_signed_release_changes_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        serve_release(&transport, temp.path(), "1.0.1", b"new");
        transport.respond(
            &format!("{LATEST}.minisig"),
            200,
            include_str!("../../tests/data/signing/release.toml.minisig"),
        );
        let mut ctx = context(temp.path(), transport.clone(), Distribution::AppImage);
        ctx.plugins.verifier = Verifier::new(&[TEST_KEY], None, false).unwrap();
        let error = plan(&mut ctx, &request()).err().unwrap();
        assert!(
            format!("{error:#}").contains("does not verify"),
            "{error:#}"
        );
        assert!(
            !transport
                .requested()
                .iter()
                .any(|url| url.ends_with(".AppImage"))
        );
    }

    #[test]
    fn policy_disables_self_update_but_not_plugin_updates() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let mut ctx = context(temp.path(), transport.clone(), Distribution::AppImage);
        install_official_gamepad(&mut ctx, &transport, temp.path());
        serve_release(&transport, temp.path(), "1.0.1", b"new");
        ctx.plugins.policy.disable_self_update = true;
        let planned = plan(&mut ctx, &request()).unwrap();
        assert!(matches!(
            planned.plan.strategy,
            Strategy::DisabledByPolicy { .. }
        ));
        assert!(!planned.plan.changes_app());
        assert!(planned.plan.changes_plugins());
    }

    #[test]
    fn offline_bundles_need_a_signature_or_explicit_trust() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle");
        fs::create_dir_all(bundle.join("plugins")).unwrap();
        let image = b"offline image";
        let plugin = package_bytes(temp.path(), "gamepad", "1.0.5");
        let catalog = catalog_text("1.0.1", "1.0.5", &sha256_hex(&plugin))
            .replace(&"a".repeat(64), &sha256_hex(image));
        fs::write(bundle.join("release.toml"), catalog).unwrap();
        fs::write(bundle.join("eov-v1.0.1-linux-x86_64.AppImage"), image).unwrap();
        fs::write(
            bundle.join("plugins/gamepad-v1.0.5-linux-x86_64.eop"),
            &plugin,
        )
        .unwrap();

        let transport = MockTransport::new();
        let mut ctx = context(temp.path(), transport.clone(), Distribution::AppImage);
        install_official_gamepad(&mut ctx, &transport, temp.path());
        let app = temp.path().join("eov.AppImage");
        fs::write(&app, b"old").unwrap();
        ctx.installation.appimage = Some(app.clone());
        let mut offline = UpdateRequest {
            from: Some(bundle.clone()),
            ..UpdateRequest::default()
        };
        let error = plan(&mut ctx, &offline).err().unwrap();
        assert!(error.to_string().contains("--allow-unsigned"), "{error}");

        let requests_before = transport.requested().len();
        offline.allow_unsigned = true;
        let planned = plan(&mut ctx, &offline).unwrap();
        assert!(planned.plan.source.contains("offline bundle"));
        execute(&mut ctx, planned).unwrap();
        assert_eq!(fs::read(&app).unwrap(), image);
        assert_eq!(
            registry::load(&ctx.plugins.plugin_dir).unwrap().plugins["gamepad"].version,
            "1.0.5"
        );
        assert_eq!(
            transport.requested().len(),
            requests_before,
            "offline updates must not use the network"
        );
    }

    #[test]
    fn flatpak_bundle_is_written_atomically_and_reruns_replace_it() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let bundle_bytes = b"flatpak bundle";
        let catalog = catalog_text("1.0.1", "1.0.5", &"b".repeat(64))
            + &format!(
                "\n[platform.linux.flatpak.x86_64]\nversion = \"1.0.1\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov/releases/download/v1.0.1/eov-v1.0.1-linux-x86_64.flatpak\"\n",
                sha256_hex(bundle_bytes)
            );
        transport.respond(LATEST, 200, catalog);
        transport.respond(
            "https://github.com/eosin-platform/eov/releases/download/v1.0.1/eov-v1.0.1-linux-x86_64.flatpak",
            200,
            bundle_bytes.to_vec(),
        );
        let mut ctx = context(temp.path(), transport, Distribution::Flatpak);
        ctx.installation.flatpak_info = Some(format!(
            "[Instance]\napp-path={}/.local/share/flatpak/app/io.eosin.eov/files\n",
            temp.path().display()
        ));
        for _ in 0..2 {
            let mut request = request();
            request.app_only = true;
            let planned = plan(&mut ctx, &request).unwrap();
            let outcome = execute(&mut ctx, planned).unwrap();
            assert_eq!(outcome.app, "downloaded");
            assert!(outcome.message.unwrap().contains("--user"));
        }
        assert_eq!(
            fs::read(ctx.updates_dir.join("eov-v1.0.1-linux-x86_64.flatpak")).unwrap(),
            bundle_bytes
        );
    }

    #[test]
    fn homebrew_prerelease_targets_are_rejected_during_planning() {
        let temp = tempfile::tempdir().unwrap();
        let transport = MockTransport::new();
        let version = "1.1.0-rc.1";
        let catalog = catalog_text(version, "1.0.5", &"b".repeat(64))
            + &format!(
                "\n[platform.macos.x86_64]\nversion = \"{version}\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov/releases/download/v{version}/eov-v{version}-macos-x86_64.zip\"\n\n[platform.macos.arm64]\nversion = \"{version}\"\nsha256 = \"{}\"\nurl = \"https://github.com/eosin-platform/eov/releases/download/v{version}/eov-v{version}-macos-arm64.zip\"\n",
                "c".repeat(64),
                "d".repeat(64)
            );
        transport.respond(&catalog_url(version), 200, catalog);
        let mut ctx = context(temp.path(), transport, Distribution::MacosBundle);
        ctx.installation.executable = PathBuf::from("/Applications/eov.app/Contents/MacOS/eov");
        let brew = FakeBrew::default();
        brew.answer(
            "info --cask --json=v2 --installed",
            0,
            include_str!("../../tests/data/homebrew/info-installed.json"),
        );
        brew.answer(
            "list --cask eosin-platform/eov/eov",
            0,
            include_str!("../../tests/data/homebrew/list-cask-eov.txt"),
        );
        ctx.brew = Box::new(brew);
        // A brew executable must exist for ownership detection.
        let prefix = temp.path().join("homebrew");
        fs::create_dir_all(prefix.join("bin")).unwrap();
        fs::write(prefix.join("bin/brew"), b"").unwrap();
        ctx.brew_env.homebrew_prefix = Some(prefix.into_os_string());
        let mut request = request();
        request.release = Some(version.into());
        let error = plan(&mut ctx, &request).err().unwrap();
        assert!(error.to_string().contains("prerelease"), "{error}");
    }

    #[test]
    fn manual_macos_bundle_without_brew_is_a_tree_update() {
        let temp = tempfile::tempdir().unwrap();
        let mut ctx = context(temp.path(), MockTransport::new(), Distribution::MacosBundle);
        ctx.installation.executable = PathBuf::from("/Users/me/Apps/eov.app/Contents/MacOS/eov");
        ctx.brew_env = BrewEnvironment {
            path: Some("/nonexistent".into()),
            homebrew_prefix: Some("/nonexistent".into()),
            ..BrewEnvironment::default()
        };
        if homebrew::locate_brew(&ctx.brew_env).is_none() {
            let (strategy, _) = probe_strategy(&ctx);
            assert!(
                matches!(strategy, Strategy::ReplaceTree { .. }),
                "{strategy:?}"
            );
        }
    }

    #[test]
    fn helper_reports_are_shown_once() {
        let temp = tempfile::tempdir().unwrap();
        let updates = temp.path().join("updates");
        fs::create_dir_all(&updates).unwrap();
        let report = HelperReport {
            schema: super::super::transaction::REPORT_SCHEMA,
            nonce: "n".into(),
            mode: Mode::ReplaceTree,
            outcome: Outcome::RolledBack,
            from_version: Some("1.0.0".into()),
            to_version: Some("1.0.1".into()),
            finished_at: 1,
            message: "a file was locked".into(),
            installation: None,
            previous: None,
        };
        fs::write(updates.join(REPORT_FILE), toml::to_string(&report).unwrap()).unwrap();
        let taken = take_helper_report(&updates).unwrap();
        assert!(taken.summary().contains("rolled back"));
        assert!(take_helper_report(&updates).is_none());
    }

    #[test]
    fn staged_windows_archives_may_be_old_releases_without_a_helper() {
        let temp = tempfile::tempdir().unwrap();
        // v0.4.5 layout: files at the archive root, no helper.
        fs::write(temp.path().join("eov.exe"), b"exe").unwrap();
        fs::write(temp.path().join("libopenslide-1.dll"), b"dll").unwrap();
        let tree = locate_staged_tree(Distribution::WindowsPortable, temp.path()).unwrap();
        assert_eq!(tree, temp.path());
        // v1.0 layout: a top-level eov folder.
        let nested = tempfile::tempdir().unwrap();
        fs::create_dir(nested.path().join("eov")).unwrap();
        fs::write(nested.path().join("eov/eov.exe"), b"exe").unwrap();
        assert_eq!(
            locate_staged_tree(Distribution::WindowsPortable, nested.path()).unwrap(),
            nested.path().join("eov")
        );
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"x").unwrap();
    }

    #[test]
    fn portable_guard_accepts_a_dedicated_eov_folder() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("eov");
        let staged = temp.path().join("stage");
        for name in [
            "eov.exe",
            "eov-update-helper.exe",
            "libopenslide-1.dll",
            "LICENSE-MIT",
        ] {
            touch(&current.join(name));
        }
        for name in [
            "eov.exe",
            "eov-update-helper.exe",
            "libopenslide-1.dll",
            "zlib1.dll",
        ] {
            touch(&staged.join(name));
        }
        verify_portable_folder_is_owned(&current, &staged).unwrap();
    }

    #[test]
    fn portable_guard_refuses_a_shared_folder() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("Downloads");
        let staged = temp.path().join("stage");
        touch(&current.join("eov.exe"));
        touch(&current.join("tax-return.pdf"));
        touch(&current.join("holiday").join("photo.jpg"));
        touch(&staged.join("eov.exe"));
        let error = verify_portable_folder_is_owned(&current, &staged)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not part of eov"), "{error}");
        assert!(error.contains("tax-return.pdf"), "{error}");
    }

    #[test]
    fn portable_guard_honours_the_shipped_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("eov");
        let staged = temp.path().join("stage");
        touch(&current.join("eov.exe"));
        touch(&current.join("plugins-readme.md"));
        fs::write(
            current.join(PORTABLE_MANIFEST),
            r#"{"version":"1.0.0","files":["eov.exe","plugins-readme.md","eov-portable.json"]}"#,
        )
        .unwrap();
        touch(&staged.join("eov.exe"));
        verify_portable_folder_is_owned(&current, &staged).unwrap();
    }

    #[test]
    fn portable_guard_requires_eov_exe() {
        let temp = tempfile::tempdir().unwrap();
        let current = temp.path().join("somewhere");
        let staged = temp.path().join("stage");
        fs::create_dir_all(&current).unwrap();
        touch(&staged.join("eov.exe"));
        assert!(verify_portable_folder_is_owned(&current, &staged).is_err());
    }

    #[test]
    fn application_zip_rejects_traversal() {
        use std::io::Write;
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("eov.zip");
        {
            let mut writer = zip::ZipWriter::new(fs::File::create(&archive).unwrap());
            writer
                .start_file("../escape", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"bad").unwrap();
            writer.finish().unwrap();
        }
        let destination = temp.path().join("out");
        fs::create_dir(&destination).unwrap();
        assert!(extract_zip_safely(&archive, &destination).is_err());
    }
}
