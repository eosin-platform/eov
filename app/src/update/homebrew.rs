//! Local Homebrew ownership detection for macOS bundles.

use anyhow::{Context, Result, bail};
use semver::Version;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomebrewInstallation {
    pub cask: String,
    pub version: Version,
    pub app_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacosInstallation {
    Homebrew(HomebrewInstallation),
    ManualBundle,
    Ambiguous(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaskListing {
    cask: String,
    version: Version,
    app_path: PathBuf,
}

pub fn probe(running_bundle: &Path) -> Result<MacosInstallation> {
    if !cfg!(target_os = "macos") {
        return Ok(MacosInstallation::ManualBundle);
    }
    let mut installed = Vec::new();
    for cask in installed_cask_tokens()? {
        let Some(version) = installed_cask_version(&cask)? else {
            continue;
        };
        let output = brew_output(&["list", "--cask", "--full-name", &cask])?;
        let Some(app_path) = parse_app_path(&output, "eov.app") else {
            return Ok(MacosInstallation::Ambiguous(format!(
                "Homebrew cask {cask} is installed but its eov.app path could not be resolved"
            )));
        };
        installed.push(CaskListing {
            cask,
            version,
            app_path,
        });
    }
    if installed.is_empty() {
        return Ok(MacosInstallation::ManualBundle);
    }
    match classify_installation(running_bundle, &installed) {
        MacosInstallation::Homebrew(match_entry) => {
            Ok(MacosInstallation::Homebrew(HomebrewInstallation {
                cask: match_entry.cask,
                version: match_entry.version,
                app_path: match_entry.app_path,
            }))
        }
        MacosInstallation::Ambiguous(reason) => Ok(MacosInstallation::Ambiguous(reason)),
        MacosInstallation::ManualBundle => Ok(MacosInstallation::ManualBundle),
    }
}

fn classify_installation(running_bundle: &Path, installed: &[CaskListing]) -> MacosInstallation {
    let running = running_bundle
        .canonicalize()
        .unwrap_or_else(|_| running_bundle.to_path_buf());
    let matches: Vec<_> = installed
        .iter()
        .filter(|entry| {
            entry
                .app_path
                .canonicalize()
                .unwrap_or_else(|_| entry.app_path.clone())
                == running
        })
        .collect();
    match matches.as_slice() {
        [entry] => MacosInstallation::Homebrew(HomebrewInstallation {
            cask: entry.cask.clone(),
            version: entry.version.clone(),
            app_path: entry.app_path.clone(),
        }),
        [] => MacosInstallation::ManualBundle,
        _ => MacosInstallation::Ambiguous(
            "multiple Homebrew EOV casks claim the running bundle".into(),
        ),
    }
}

pub fn update_owned(installation: &HomebrewInstallation, target: Option<&Version>) -> Result<()> {
    if !cfg!(target_os = "macos") {
        bail!("Homebrew updates are only available on macOS");
    }
    let target_cask = target
        .map(|version| format!("eov@{version}"))
        .unwrap_or_else(|| "eov".to_string());
    if installation.cask == target_cask {
        run_brew(&["upgrade", "--cask", &target_cask])
    } else {
        run_brew(&["uninstall", "--cask", &installation.cask])?;
        if let Err(error) = run_brew(&["install", "--cask", &target_cask]) {
            let _ = run_brew(&["install", "--cask", &installation.cask]);
            return Err(error)
                .context("Homebrew failed to install the target EOV cask; rollback was attempted");
        }
        Ok(())
    }
}

fn installed_cask_tokens() -> Result<Vec<String>> {
    let output = Command::new("brew")
        .env("HOMEBREW_NO_AUTO_UPDATE", "1")
        .env("HOMEBREW_NO_ANALYTICS", "1")
        .args(["list", "--cask", "--full-name"])
        .output()
        .with_context(|| "could not invoke Homebrew for local ownership detection")?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|token| *token == "eov" || token.starts_with("eov@"))
        .map(str::to_string)
        .collect())
}

fn installed_cask_version(cask: &str) -> Result<Option<Version>> {
    let output = Command::new("brew")
        .env("HOMEBREW_NO_AUTO_UPDATE", "1")
        .env("HOMEBREW_NO_ANALYTICS", "1")
        .args(["list", "--cask", "--versions", cask])
        .output()
        .with_context(|| "could not invoke Homebrew for local ownership detection")?;
    if !output.status.success() {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let version = text.split_whitespace().nth(1);
    version
        .map(|version| Version::parse(version).map_err(anyhow::Error::from))
        .transpose()
        .with_context(|| format!("Homebrew returned an invalid version for cask {cask}"))
}

fn brew_output(args: &[&str]) -> Result<String> {
    let output = Command::new("brew")
        .env("HOMEBREW_NO_AUTO_UPDATE", "1")
        .env("HOMEBREW_NO_ANALYTICS", "1")
        .args(args)
        .output()
        .with_context(|| "could not invoke Homebrew for local ownership detection")?;
    if !output.status.success() {
        bail!("Homebrew ownership query failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn run_brew(args: &[&str]) -> Result<()> {
    let status = Command::new("brew")
        .env("HOMEBREW_NO_ANALYTICS", "1")
        .args(args)
        .status()
        .with_context(|| "could not invoke Homebrew")?;
    if !status.success() {
        bail!("Homebrew command failed: brew {}", args.join(" "));
    }
    Ok(())
}

fn parse_app_path(output: &str, app_name: &str) -> Option<PathBuf> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some(app_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cask_listing_path() {
        let path = parse_app_path(
            "/opt/homebrew/Caskroom/eov/0.4.5/eov.app\n/opt/homebrew/Caskroom/eov/0.4.5/eov.app/Contents/MacOS/eov\n",
            "eov.app",
        )
        .unwrap();
        assert!(path.ends_with("eov.app"));
    }

    #[test]
    fn ignores_other_files() {
        assert!(parse_app_path("/tmp/eov.txt\n", "eov.app").is_none());
    }

    fn listing(cask: &str, app_path: &str) -> CaskListing {
        CaskListing {
            cask: cask.into(),
            version: Version::parse("0.4.5").unwrap(),
            app_path: PathBuf::from(app_path),
        }
    }

    #[test]
    fn classifies_matching_unversioned_cask_as_homebrew() {
        let result =
            classify_installation(Path::new("/tmp/eov.app"), &[listing("eov", "/tmp/eov.app")]);
        assert!(matches!(result, MacosInstallation::Homebrew(_)));
    }

    #[test]
    fn classifies_matching_versioned_cask_as_homebrew() {
        let result = classify_installation(
            Path::new("/tmp/eov.app"),
            &[listing("eov@0.4.5", "/tmp/eov.app")],
        );
        assert!(matches!(result, MacosInstallation::Homebrew(_)));
    }

    #[test]
    fn treats_other_bundle_path_as_manual() {
        let result = classify_installation(
            Path::new("/Applications/eov.app"),
            &[listing("eov", "/opt/homebrew/Caskroom/eov/0.4.5/eov.app")],
        );
        assert_eq!(result, MacosInstallation::ManualBundle);
    }

    #[test]
    fn reports_ambiguous_matching_casks() {
        let result = classify_installation(
            Path::new("/tmp/eov.app"),
            &[
                listing("eov", "/tmp/eov.app"),
                listing("eov@0.4.5", "/tmp/eov.app"),
            ],
        );
        assert!(matches!(result, MacosInstallation::Ambiguous(_)));
    }
}
