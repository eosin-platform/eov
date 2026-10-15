//! Detection of `cargo install` ownership.
//!
//! An eov binary is treated as Cargo-managed only when an install root's
//! metadata (`.crates2.json`, or the older `.crates.toml`) records the `eov`
//! package from crates.io *and* that root's `bin/eov` is the running
//! executable. Install roots are searched like Cargo does: `$CARGO_INSTALL_ROOT`,
//! `install.root` in `$CARGO_HOME/config.toml`, `$CARGO_HOME`, `~/.cargo`, and
//! the parent of the directory holding the executable (for `--root` installs).

use serde::Deserialize;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// crates.io source ids as Cargo records them.
const CRATES_IO_SOURCES: &[&str] = &[
    "(registry+https://github.com/rust-lang/crates.io-index)",
    "(sparse+https://index.crates.io/)",
];

/// A Cargo installation of eov.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CargoInstallation {
    /// The install root (pass to `cargo install --root`).
    pub root: PathBuf,
    /// The package id, e.g. `eov 1.0.0 (registry+https://...)`.
    pub package: String,
    /// Whether the package came from crates.io (and so can be updated by
    /// version). Path and git installs cannot.
    pub from_crates_io: bool,
}

/// Environment lookups (injectable for tests).
pub struct CargoEnvironment {
    pub cargo_install_root: Option<OsString>,
    pub cargo_home: Option<OsString>,
    pub home: Option<PathBuf>,
}

impl CargoEnvironment {
    pub fn from_process() -> Self {
        Self {
            cargo_install_root: std::env::var_os("CARGO_INSTALL_ROOT"),
            cargo_home: std::env::var_os("CARGO_HOME"),
            home: dirs::home_dir(),
        }
    }
}

fn cargo_home(env: &CargoEnvironment) -> Option<PathBuf> {
    env.cargo_home
        .as_ref()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| env.home.as_ref().map(|home| home.join(".cargo")))
}

fn config_install_root(cargo_home: &Path) -> Option<PathBuf> {
    for name in ["config.toml", "config"] {
        let Ok(text) = std::fs::read_to_string(cargo_home.join(name)) else {
            continue;
        };
        let value: toml::Value = toml::from_str(&text).ok()?;
        if let Some(root) = value
            .get("install")
            .and_then(|install| install.get("root"))
            .and_then(toml::Value::as_str)
        {
            return Some(PathBuf::from(root));
        }
    }
    None
}

/// Candidate install roots, in Cargo's precedence order.
pub fn install_roots(executable: &Path, env: &CargoEnvironment) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = env
        .cargo_install_root
        .as_ref()
        .filter(|value| !value.is_empty())
    {
        roots.push(PathBuf::from(root));
    }
    let home = cargo_home(env);
    if let Some(home) = &home {
        if let Some(root) = config_install_root(home) {
            roots.push(root);
        }
        roots.push(home.clone());
    }
    if let Some(bin) = executable.parent()
        && bin.file_name().is_some_and(|name| name == "bin")
        && let Some(root) = bin.parent()
    {
        roots.push(root.to_path_buf());
    }
    roots.dedup();
    roots
}

#[derive(Deserialize)]
struct Crates2 {
    #[serde(default)]
    installs: BTreeMap<String, Crates2Entry>,
}

#[derive(Deserialize)]
struct Crates2Entry {
    #[serde(default)]
    bins: Vec<String>,
}

/// The package id that installed binary `bin` in `root`, if any.
fn installed_package(root: &Path, bin: &str) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(root.join(".crates2.json"))
        && let Ok(crates) = serde_json::from_str::<Crates2>(&text)
    {
        return crates
            .installs
            .into_iter()
            .find(|(package, entry)| {
                package.starts_with("eov ") && entry.bins.iter().any(|name| name == bin)
            })
            .map(|(package, _)| package);
    }
    let text = std::fs::read_to_string(root.join(".crates.toml")).ok()?;
    let value: toml::Value = toml::from_str(&text).ok()?;
    value
        .get("v1")?
        .as_table()?
        .iter()
        .find(|(package, bins)| {
            package.starts_with("eov ")
                && bins
                    .as_array()
                    .is_some_and(|bins| bins.iter().any(|name| name.as_str() == Some(bin)))
        })
        .map(|(package, _)| package.clone())
}

/// Finds the Cargo installation the running `executable` belongs to.
pub fn probe(executable: &Path, env: &CargoEnvironment) -> Option<CargoInstallation> {
    let executable = executable
        .canonicalize()
        .unwrap_or_else(|_| executable.to_path_buf());
    let bin_name = if cfg!(windows) { "eov.exe" } else { "eov" };
    for root in install_roots(&executable, env) {
        let installed = root.join("bin").join(bin_name);
        let matches = installed
            .canonicalize()
            .is_ok_and(|installed| installed == executable);
        if !matches {
            continue;
        }
        // `.crates2.json` records binary names without the extension.
        if let Some(package) = installed_package(&root, "eov") {
            let from_crates_io = CRATES_IO_SOURCES
                .iter()
                .any(|source| package.ends_with(source));
            return Some(CargoInstallation {
                root,
                package,
                from_crates_io,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn install(root: &Path, crates2: &str) -> PathBuf {
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let exe = bin.join(if cfg!(windows) { "eov.exe" } else { "eov" });
        fs::write(&exe, b"eov").unwrap();
        fs::write(root.join(".crates2.json"), crates2).unwrap();
        exe
    }

    const REGISTRY: &str = r#"{"installs":{"eov 1.0.0 (registry+https://github.com/rust-lang/crates.io-index)":{"version_req":null,"bins":["eov","eov-update-helper"],"features":["distribution-cargo"],"all_features":false,"no_default_features":false,"profile":"release","target":"x86_64-unknown-linux-gnu","rustc":"rustc 1.98.1"}}}"#;

    fn env(home: &Path) -> CargoEnvironment {
        CargoEnvironment {
            cargo_install_root: None,
            cargo_home: None,
            home: Some(home.to_path_buf()),
        }
    }

    #[test]
    fn default_cargo_home_install_is_detected() {
        let temp = tempfile::tempdir().unwrap();
        let exe = install(&temp.path().join(".cargo"), REGISTRY);
        let installation = probe(&exe, &env(temp.path())).unwrap();
        assert!(installation.from_crates_io);
        assert_eq!(
            installation.root.canonicalize().unwrap(),
            temp.path().join(".cargo").canonicalize().unwrap()
        );
    }

    #[test]
    fn custom_install_roots_are_honoured() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("opt-eov");
        let exe = install(&root, REGISTRY);
        let mut environment = env(temp.path());
        environment.cargo_install_root = Some(root.clone().into_os_string());
        assert!(probe(&exe, &environment).is_some());
        // Found via the executable's location even without the variable.
        assert!(probe(&exe, &env(temp.path())).is_some());
    }

    #[test]
    fn config_install_root_is_honoured() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("tools");
        let cargo = temp.path().join(".cargo");
        fs::create_dir_all(&cargo).unwrap();
        fs::write(
            cargo.join("config.toml"),
            format!("[install]\nroot = {:?}\n", root.to_string_lossy()),
        )
        .unwrap();
        let roots = install_roots(&root.join("bin/eov"), &env(temp.path()));
        assert_eq!(roots[0], root);
    }

    #[test]
    fn path_installs_and_foreign_binaries_are_not_updatable() {
        let temp = tempfile::tempdir().unwrap();
        let exe = install(
            &temp.path().join(".cargo"),
            r#"{"installs":{"eov 1.0.0 (path+file:///src/eov/app)":{"bins":["eov"]}}}"#,
        );
        assert!(!probe(&exe, &env(temp.path())).unwrap().from_crates_io);

        let other = temp.path().join("downloads/eov");
        fs::create_dir_all(other.parent().unwrap()).unwrap();
        fs::write(&other, b"eov").unwrap();
        assert!(probe(&other, &env(temp.path())).is_none());
    }

    #[test]
    fn legacy_crates_toml_is_read() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join(".cargo");
        let exe = install(&root, "");
        fs::remove_file(root.join(".crates2.json")).unwrap();
        fs::write(
            root.join(".crates.toml"),
            "[v1]\n\"eov 0.4.9 (registry+https://github.com/rust-lang/crates.io-index)\" = [\"eov\", \"eov-update-helper\"]\n",
        )
        .unwrap();
        assert!(probe(&exe, &env(temp.path())).unwrap().from_crates_io);
    }
}
