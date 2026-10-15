//! Flatpak bundle updates.
//!
//! The standalone `.flatpak` bundle cannot replace itself from inside the
//! sandbox, so eov downloads and verifies the new bundle and tells the user
//! the exact host command, for the installation (per-user or system-wide)
//! that the running app came from.

use serde::Serialize;
use std::path::Path;

/// Which Flatpak installation the running app belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Installation {
    User,
    System,
    Unknown,
}

/// Reads `/.flatpak-info` (an INI file the sandbox provides): the
/// `[Instance] app-path` lies below `~/.local/share/flatpak` for per-user
/// installations and below `/var/lib/flatpak` (or another system
/// installation) otherwise.
pub fn detect(flatpak_info: Option<&str>, home: Option<&Path>) -> Installation {
    let Some(info) = flatpak_info else {
        return Installation::Unknown;
    };
    let mut in_instance = false;
    for line in info.lines().map(str::trim) {
        if line.starts_with('[') {
            in_instance = line == "[Instance]";
            continue;
        }
        if !in_instance {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "installation" if value.trim() == "user" => return Installation::User,
            "app-path" => {
                let value = value.trim();
                let user_root = home.map(|home| home.join(".local/share/flatpak"));
                if user_root.is_some_and(|root| Path::new(value).starts_with(root)) {
                    return Installation::User;
                }
                if value.starts_with('/') {
                    return Installation::System;
                }
            }
            _ => {}
        }
    }
    Installation::Unknown
}

/// The host command that installs `bundle`.
pub fn install_instructions(installation: Installation, bundle: &Path) -> String {
    let bundle = bundle.display();
    match installation {
        Installation::User => format!("    flatpak install --user --bundle {bundle}"),
        Installation::System => format!(
            "    flatpak install --system --bundle {bundle}\n(system-wide installations need administrator rights)"
        ),
        Installation::Unknown => format!(
            "    flatpak install --user --bundle {bundle}\nor, for a system-wide installation:\n    flatpak install --system --bundle {bundle}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: &str = "[Application]\nname=io.eosin.eov\n\n[Instance]\ninstance-id=123\napp-path=/home/me/.local/share/flatpak/app/io.eosin.eov/x86_64/stable/abc/files\nbranch=stable\n";
    const SYSTEM: &str = "[Application]\nname=io.eosin.eov\n\n[Instance]\napp-path=/var/lib/flatpak/app/io.eosin.eov/x86_64/stable/abc/files\n";

    #[test]
    fn detects_user_and_system_installations() {
        let home = Path::new("/home/me");
        assert_eq!(detect(Some(USER), Some(home)), Installation::User);
        assert_eq!(detect(Some(SYSTEM), Some(home)), Installation::System);
        assert_eq!(detect(None, Some(home)), Installation::Unknown);
        assert_eq!(
            detect(Some("[Instance]\ninstallation=user\n"), None),
            Installation::User
        );
    }

    #[test]
    fn instructions_name_the_right_installation() {
        let bundle = Path::new("/home/me/.eov/updates/eov.flatpak");
        assert!(install_instructions(Installation::User, bundle).contains("--user"));
        assert!(!install_instructions(Installation::User, bundle).contains("--system"));
        assert!(install_instructions(Installation::System, bundle).contains("--system"));
        let unknown = install_instructions(Installation::Unknown, bundle);
        assert!(unknown.contains("--user") && unknown.contains("--system"));
    }
}
