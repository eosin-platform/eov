//! Homebrew ownership detection and delegation for macOS bundles.
//!
//! eov is distributed through the `eosin-platform/eov` tap, whose casks are
//! named `eov` (latest) and `eov@X.Y.Z` (one per release). When the running
//! `eov.app` belongs to an installed cask, Homebrew stays its owner: eov asks
//! brew to change the installation and verifies the result instead of
//! swapping the bundle behind brew's back.
//!
//! Ownership is derived from `brew info --cask --json=v2 --installed`
//! (structured, and it names the tap-qualified `full_token`) together with
//! the app artifact's install location (`brew list --cask <token>`, the
//! `--appdir` in `HOMEBREW_CASK_OPTS`, `/Applications` and `~/Applications`).
//! brew is found on `PATH` or in its default prefixes (`/opt/homebrew`,
//! `/usr/local`), because apps started from the Dock do not inherit the
//! shell's `PATH`; a Mac without Homebrew is simply a manual installation.

use anyhow::{Context, Result, bail};
use semver::Version;
use serde::Deserialize;
use serde_json::Value;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

/// Runs brew (injectable for tests).
pub trait CommandRunner {
    fn output(&self, program: &Path, args: &[&str]) -> io::Result<Output>;
    fn status(&self, program: &Path, args: &[&str]) -> io::Result<ExitStatus>;
}

/// Runs real processes. Queries never trigger brew's auto-update or
/// analytics.
pub struct SystemRunner;

impl CommandRunner for SystemRunner {
    fn output(&self, program: &Path, args: &[&str]) -> io::Result<Output> {
        Command::new(program)
            .env("HOMEBREW_NO_AUTO_UPDATE", "1")
            .env("HOMEBREW_NO_ANALYTICS", "1")
            .env("HOMEBREW_NO_ENV_HINTS", "1")
            .args(args)
            .output()
    }

    fn status(&self, program: &Path, args: &[&str]) -> io::Result<ExitStatus> {
        Command::new(program)
            .env("HOMEBREW_NO_AUTO_UPDATE", "1")
            .env("HOMEBREW_NO_ANALYTICS", "1")
            .env("HOMEBREW_NO_ENV_HINTS", "1")
            .args(args)
            .status()
    }
}

/// Environment that influences where brew and cask apps live.
#[derive(Debug, Clone, Default)]
pub struct BrewEnvironment {
    pub path: Option<OsString>,
    pub homebrew_prefix: Option<OsString>,
    pub cask_opts: Option<String>,
    pub home: Option<PathBuf>,
}

impl BrewEnvironment {
    pub fn from_process() -> Self {
        Self {
            path: std::env::var_os("PATH"),
            homebrew_prefix: std::env::var_os("HOMEBREW_PREFIX"),
            cask_opts: std::env::var("HOMEBREW_CASK_OPTS").ok(),
            home: dirs::home_dir(),
        }
    }
}

/// Finds the brew executable: `$HOMEBREW_PREFIX/bin/brew`, `PATH`, then the
/// default prefixes of Apple-silicon and Intel installations.
pub fn locate_brew(env: &BrewEnvironment) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(prefix) = &env.homebrew_prefix {
        candidates.push(PathBuf::from(prefix).join("bin/brew"));
    }
    if let Some(path) = &env.path {
        candidates.extend(std::env::split_paths(path).map(|dir| dir.join("brew")));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin/brew"));
    candidates.push(PathBuf::from("/usr/local/bin/brew"));
    candidates.into_iter().find(|candidate| candidate.is_file())
}

/// An installed eov cask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaskRecord {
    /// `eov` or `eov@1.0.0`.
    pub token: String,
    /// `eosin-platform/eov/eov`.
    pub full_token: String,
    /// `eosin-platform/eov`.
    pub tap: Option<String>,
    /// The installed version.
    pub installed: String,
    /// The newest version the (local copy of the) tap knows.
    pub latest: String,
    /// App bundle names the cask installs (after `target:` renames).
    pub apps: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct InfoV2 {
    #[serde(default)]
    casks: Vec<CaskInfo>,
}

#[derive(Debug, Deserialize)]
struct CaskInfo {
    token: String,
    #[serde(default)]
    full_token: Option<String>,
    #[serde(default)]
    tap: Option<String>,
    version: String,
    #[serde(default)]
    installed: Option<String>,
    #[serde(default)]
    artifacts: Vec<Value>,
}

/// Whether a cask token is one of eov's (`eov`, `eov@X.Y.Z`).
pub fn is_eov_token(token: &str) -> bool {
    token == "eov"
        || token
            .strip_prefix("eov@")
            .is_some_and(|version| !version.is_empty())
}

/// Parses `brew info --cask --json=v2` output, keeping installed eov casks.
pub fn parse_info_v2(json: &str) -> Result<Vec<CaskRecord>> {
    let info: InfoV2 = serde_json::from_str(json).context("brew returned invalid JSON")?;
    Ok(info
        .casks
        .into_iter()
        .filter(|cask| is_eov_token(&cask.token))
        .filter_map(|cask| {
            let installed = cask.installed?;
            Some(CaskRecord {
                full_token: cask.full_token.unwrap_or_else(|| cask.token.clone()),
                apps: app_names(&cask.artifacts),
                token: cask.token,
                tap: cask.tap,
                installed,
                latest: cask.version,
            })
        })
        .collect())
}

/// The installed names of a cask's `app` artifacts: `{"app": ["eov.app"]}`
/// or `{"app": ["eov.app", {"target": "Other.app"}]}`.
fn app_names(artifacts: &[Value]) -> Vec<String> {
    let mut names = Vec::new();
    for artifact in artifacts {
        let Some(app) = artifact.get("app").and_then(Value::as_array) else {
            continue;
        };
        let source = app.iter().find_map(Value::as_str);
        let target = app
            .iter()
            .filter_map(Value::as_object)
            .find_map(|options| options.get("target").and_then(Value::as_str));
        if let Some(name) = target.or(source) {
            let name = Path::new(name)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| name.to_string());
            names.push(name);
        }
    }
    names
}

/// App bundle paths from `brew list --cask <token>` output, whose app lines
/// look like `/Applications/eov.app (2,345 files, 48.2MB)`.
pub fn parse_list_app_paths(output: &str) -> Vec<PathBuf> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('/') && !line.contains(" -> "))
        .map(|line| match line.rfind(" (") {
            Some(index) if line.ends_with(')') => &line[..index],
            _ => line,
        })
        .filter(|path| path.ends_with(".app"))
        .map(PathBuf::from)
        .collect()
}

/// Directories brew installs app artifacts into.
pub fn app_dirs(env: &BrewEnvironment) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(opts) = &env.cask_opts {
        let mut words = opts.split_whitespace();
        while let Some(word) = words.next() {
            let value = match word.strip_prefix("--appdir=") {
                Some(value) => Some(value.to_string()),
                None if word == "--appdir" => words.next().map(str::to_string),
                None => None,
            };
            if let Some(value) = value {
                let value = value.trim_matches(|ch| ch == '"' || ch == '\'');
                let path = match (value.strip_prefix("~/"), &env.home) {
                    (Some(rest), Some(home)) => home.join(rest),
                    _ => PathBuf::from(value),
                };
                dirs.push(path);
            }
        }
    }
    dirs.push(PathBuf::from("/Applications"));
    if let Some(home) = &env.home {
        dirs.push(home.join("Applications"));
    }
    dirs
}

/// A Homebrew-owned installation of the running bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomebrewInstallation {
    pub brew: PathBuf,
    pub cask: CaskRecord,
    pub app_path: PathBuf,
}

impl HomebrewInstallation {
    /// The tap that provides eov's casks.
    pub fn tap(&self) -> &str {
        self.cask.tap.as_deref().unwrap_or("eosin-platform/eov")
    }
}

/// Who manages the running macOS bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacosInstallation {
    Homebrew(HomebrewInstallation),
    ManualBundle,
    Ambiguous(String),
}

fn same_path(left: &Path, right: &Path) -> bool {
    let canonical = |path: &Path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    canonical(left) == canonical(right)
}

/// Determines whether Homebrew owns `running_bundle`.
pub fn probe(
    runner: &dyn CommandRunner,
    env: &BrewEnvironment,
    running_bundle: &Path,
) -> MacosInstallation {
    let Some(brew) = locate_brew(env) else {
        return MacosInstallation::ManualBundle;
    };
    probe_with(runner, &brew, env, running_bundle)
}

/// [`probe`] with an explicit brew executable.
pub fn probe_with(
    runner: &dyn CommandRunner,
    brew: &Path,
    env: &BrewEnvironment,
    running_bundle: &Path,
) -> MacosInstallation {
    let records = match runner.output(brew, &["info", "--cask", "--json=v2", "--installed"]) {
        Ok(output) if output.status.success() => {
            match parse_info_v2(&String::from_utf8_lossy(&output.stdout)) {
                Ok(records) => records,
                Err(error) => return MacosInstallation::Ambiguous(error.to_string()),
            }
        }
        Ok(_) | Err(_) if running_bundle.to_string_lossy().contains("/Caskroom/") => {
            return MacosInstallation::Ambiguous(
                "brew could not list installed casks, but eov runs from the Caskroom".into(),
            );
        }
        // No usable brew: nothing can own the bundle.
        Ok(_) | Err(_) => return MacosInstallation::ManualBundle,
    };
    let prefix = brew.parent().and_then(Path::parent).map(Path::to_path_buf);
    let mut matches = Vec::new();
    for record in records {
        let mut candidates = Vec::new();
        if let Ok(output) = runner.output(brew, &["list", "--cask", &record.full_token])
            && output.status.success()
        {
            candidates.extend(parse_list_app_paths(&String::from_utf8_lossy(
                &output.stdout,
            )));
        }
        for app in &record.apps {
            candidates.extend(app_dirs(env).into_iter().map(|dir| dir.join(app)));
            if let Some(prefix) = &prefix {
                candidates.push(
                    prefix
                        .join("Caskroom")
                        .join(&record.token)
                        .join(&record.installed)
                        .join(app),
                );
            }
        }
        if let Some(app_path) = candidates
            .into_iter()
            .find(|candidate| same_path(candidate, running_bundle))
        {
            matches.push(HomebrewInstallation {
                brew: brew.to_path_buf(),
                cask: record,
                app_path,
            });
        }
    }
    match matches.len() {
        0 => MacosInstallation::ManualBundle,
        1 => MacosInstallation::Homebrew(matches.remove(0)),
        _ => MacosInstallation::Ambiguous(format!(
            "several Homebrew casks claim {}: {}",
            running_bundle.display(),
            matches
                .iter()
                .map(|entry| entry.cask.full_token.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Refuses targets Homebrew cannot deliver, before anything is changed.
pub fn check_target(target: &Version) -> Result<()> {
    if !target.pre.is_empty() || !target.build.is_empty() {
        bail!(
            "Homebrew casks are only published for stable eov releases; {target} is a prerelease. Install it manually from GitHub Releases, or uninstall the cask first."
        );
    }
    Ok(())
}

fn run(runner: &dyn CommandRunner, brew: &Path, args: &[&str]) -> Result<()> {
    let status = runner
        .status(brew, args)
        .with_context(|| format!("could not run brew {}", args.join(" ")))?;
    if !status.success() {
        bail!("brew {} failed with {status}", args.join(" "));
    }
    Ok(())
}

/// The installed version of the cask that owns `app_path` now.
pub fn installed_version(
    runner: &dyn CommandRunner,
    installation: &HomebrewInstallation,
    env: &BrewEnvironment,
) -> Result<Option<String>> {
    match probe_with(runner, &installation.brew, env, &installation.app_path) {
        MacosInstallation::Homebrew(now) => Ok(Some(now.cask.installed)),
        MacosInstallation::ManualBundle => Ok(None),
        MacosInstallation::Ambiguous(reason) => bail!("{reason}"),
    }
}

/// Asks Homebrew to install eov `target` (the latest cask when `exact` is
/// false) and returns the version that is installed afterwards, which the
/// caller must compare with `target` before aligning plugins.
pub fn update(
    runner: &dyn CommandRunner,
    env: &BrewEnvironment,
    installation: &HomebrewInstallation,
    target: &Version,
    exact: bool,
) -> Result<String> {
    check_target(target)?;
    // Refresh the tap: without it, brew does not know a release published
    // since its last auto-update.
    if let Err(error) = run(runner, &installation.brew, &["update"]) {
        eprintln!("warning: brew update failed ({error:#}); using the taps brew already has");
    }
    let tap = installation.tap().to_string();
    let target_token = if exact {
        format!("eov@{target}")
    } else {
        "eov".to_string()
    };
    let target_full = format!("{tap}/{target_token}");
    if installation.cask.token == target_token {
        if !exact {
            run(
                runner,
                &installation.brew,
                &["upgrade", "--cask", &installation.cask.full_token],
            )?;
        }
    } else {
        // Make sure the target exists before uninstalling anything.
        let exists = runner
            .output(
                &installation.brew,
                &["info", "--cask", "--json=v2", &target_full],
            )
            .is_ok_and(|output| output.status.success());
        if !exists {
            bail!(
                "Homebrew has no cask {target_full}. Nothing was changed. (Run `brew update`, or check that eov {target} was released.)"
            );
        }
        run(
            runner,
            &installation.brew,
            &["uninstall", "--cask", &installation.cask.full_token],
        )?;
        if let Err(error) = run(
            runner,
            &installation.brew,
            &["install", "--cask", &target_full],
        ) {
            let restored = run(
                runner,
                &installation.brew,
                &["install", "--cask", &installation.cask.full_token],
            );
            return Err(error).context(match restored {
                Ok(()) => format!(
                    "Homebrew could not install {target_full}; {} was reinstalled",
                    installation.cask.full_token
                ),
                Err(_) => format!(
                    "Homebrew could not install {target_full}, and reinstalling {} failed too; run `brew install --cask {}`",
                    installation.cask.full_token, installation.cask.full_token
                ),
            });
        }
    }
    let mut after = installation.clone();
    if installation.cask.token != target_token {
        after.cask.token = target_token;
        after.cask.full_token = target_full;
    }
    installed_version(runner, &after, env)?
        .context("Homebrew no longer reports an eov cask for this bundle")
}

/// Helpers for tests elsewhere in the updater.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// A brew stand-in: answers commands from a table and records them.
    #[derive(Default)]
    pub(crate) struct FakeBrew {
        pub(crate) outputs: RefCell<HashMap<String, (i32, String)>>,
        pub(crate) calls: RefCell<Vec<String>>,
    }

    impl FakeBrew {
        pub(crate) fn answer(&self, args: &str, code: i32, stdout: &str) {
            self.outputs
                .borrow_mut()
                .insert(args.to_string(), (code, stdout.to_string()));
        }
    }

    #[cfg(unix)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    #[cfg(windows)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(code as u32)
    }

    impl CommandRunner for FakeBrew {
        fn output(&self, _program: &Path, args: &[&str]) -> io::Result<Output> {
            let key = args.join(" ");
            self.calls.borrow_mut().push(key.clone());
            let (code, stdout) = self
                .outputs
                .borrow()
                .get(&key)
                .cloned()
                .unwrap_or((1, String::new()));
            Ok(Output {
                status: exit_status(code),
                stdout: stdout.into_bytes(),
                stderr: Vec::new(),
            })
        }

        fn status(&self, program: &Path, args: &[&str]) -> io::Result<ExitStatus> {
            self.output(program, args).map(|output| output.status)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeBrew;
    use super::*;

    const INFO: &str = include_str!("../../tests/data/homebrew/info-installed.json");
    const INFO_VERSIONED: &str =
        include_str!("../../tests/data/homebrew/info-installed-versioned.json");
    const LIST: &str = include_str!("../../tests/data/homebrew/list-cask-eov.txt");

    fn env() -> BrewEnvironment {
        BrewEnvironment {
            home: Some(PathBuf::from("/Users/me")),
            ..BrewEnvironment::default()
        }
    }

    #[test]
    fn parses_tap_qualified_casks_from_json_v2() {
        let records = parse_info_v2(INFO).unwrap();
        assert_eq!(records.len(), 1, "firefox must be ignored");
        let eov = &records[0];
        assert_eq!(eov.full_token, "eosin-platform/eov/eov");
        assert_eq!(eov.tap.as_deref(), Some("eosin-platform/eov"));
        assert_eq!(eov.installed, "1.0.0");
        assert_eq!(eov.latest, "1.0.1");
        assert_eq!(eov.apps, vec!["eov.app".to_string()]);

        let versioned = parse_info_v2(INFO_VERSIONED).unwrap();
        assert_eq!(versioned[0].token, "eov@0.4.9");
        assert_eq!(versioned[0].apps, vec!["eov 0.4.9.app".to_string()]);
    }

    #[test]
    fn parses_app_paths_from_brew_list() {
        assert_eq!(
            parse_list_app_paths(LIST),
            vec![PathBuf::from("/Applications/eov.app")]
        );
    }

    #[test]
    fn appdir_from_cask_opts_is_searched_first() {
        let env = BrewEnvironment {
            cask_opts: Some("--no-quarantine --appdir=~/Apps".into()),
            home: Some(PathBuf::from("/Users/me")),
            ..BrewEnvironment::default()
        };
        assert_eq!(
            app_dirs(&env),
            vec![
                PathBuf::from("/Users/me/Apps"),
                PathBuf::from("/Applications"),
                PathBuf::from("/Users/me/Applications")
            ]
        );
    }

    #[test]
    fn running_cask_bundle_is_owned_by_homebrew() {
        let brew = FakeBrew::default();
        brew.answer("info --cask --json=v2 --installed", 0, INFO);
        brew.answer("list --cask eosin-platform/eov/eov", 0, LIST);
        let result = probe_with(
            &brew,
            Path::new("/opt/homebrew/bin/brew"),
            &env(),
            Path::new("/Applications/eov.app"),
        );
        let MacosInstallation::Homebrew(installation) = result else {
            panic!("expected Homebrew ownership, got {result:?}");
        };
        assert_eq!(installation.cask.full_token, "eosin-platform/eov/eov");
        assert_eq!(installation.tap(), "eosin-platform/eov");
    }

    #[test]
    fn other_bundles_are_manual_installations() {
        let brew = FakeBrew::default();
        brew.answer("info --cask --json=v2 --installed", 0, INFO);
        brew.answer("list --cask eosin-platform/eov/eov", 0, LIST);
        assert_eq!(
            probe_with(
                &brew,
                Path::new("/opt/homebrew/bin/brew"),
                &env(),
                Path::new("/Users/me/Downloads/eov.app"),
            ),
            MacosInstallation::ManualBundle
        );
    }

    #[test]
    fn missing_or_broken_brew_means_manual_bundle() {
        let env = BrewEnvironment {
            path: Some(OsString::from("/nonexistent/bin")),
            homebrew_prefix: Some(OsString::from("/nonexistent")),
            ..BrewEnvironment::default()
        };
        // On the CI machines running these tests brew may exist in a default
        // prefix; only assert when it does not.
        if locate_brew(&env).is_none() {
            assert_eq!(
                probe(
                    &FakeBrew::default(),
                    &env,
                    Path::new("/Applications/eov.app")
                ),
                MacosInstallation::ManualBundle
            );
        }
        let brew = FakeBrew::default();
        assert_eq!(
            probe_with(
                &brew,
                Path::new("/opt/homebrew/bin/brew"),
                &env,
                Path::new("/Applications/eov.app")
            ),
            MacosInstallation::ManualBundle
        );
    }

    #[test]
    fn prereleases_are_rejected_before_anything_runs() {
        let brew = FakeBrew::default();
        let installation = HomebrewInstallation {
            brew: PathBuf::from("/opt/homebrew/bin/brew"),
            cask: parse_info_v2(INFO).unwrap().remove(0),
            app_path: PathBuf::from("/Applications/eov.app"),
        };
        let target = Version::parse("1.1.0-rc.1").unwrap();
        assert!(update(&brew, &env(), &installation, &target, true).is_err());
        assert!(brew.calls.borrow().is_empty());
    }

    #[test]
    fn exact_targets_are_verified_before_uninstalling() {
        let brew = FakeBrew::default();
        brew.answer("update", 0, "");
        let installation = HomebrewInstallation {
            brew: PathBuf::from("/opt/homebrew/bin/brew"),
            cask: parse_info_v2(INFO).unwrap().remove(0),
            app_path: PathBuf::from("/Applications/eov.app"),
        };
        let target = Version::parse("0.4.9").unwrap();
        let error = update(&brew, &env(), &installation, &target, true).unwrap_err();
        assert!(error.to_string().contains("no cask"), "{error}");
        assert!(
            !brew
                .calls
                .borrow()
                .iter()
                .any(|call| call.starts_with("uninstall"))
        );
    }

    #[test]
    fn latest_upgrade_refreshes_the_tap_and_reads_back_the_version() {
        let brew = FakeBrew::default();
        brew.answer("update", 0, "");
        brew.answer("upgrade --cask eosin-platform/eov/eov", 0, "");
        // After the upgrade brew reports 1.0.1 as installed.
        brew.answer(
            "info --cask --json=v2 --installed",
            0,
            &INFO.replace("\"installed\": \"1.0.0\"", "\"installed\": \"1.0.1\""),
        );
        brew.answer("list --cask eosin-platform/eov/eov", 0, LIST);
        let installation = HomebrewInstallation {
            brew: PathBuf::from("/opt/homebrew/bin/brew"),
            cask: parse_info_v2(INFO).unwrap().remove(0),
            app_path: PathBuf::from("/Applications/eov.app"),
        };
        let installed = update(
            &brew,
            &env(),
            &installation,
            &Version::parse("1.0.1").unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(installed, "1.0.1");
        assert_eq!(brew.calls.borrow()[0], "update");
    }
}
