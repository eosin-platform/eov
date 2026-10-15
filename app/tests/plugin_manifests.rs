//! Every plugin manifest in the repository must load in this eov: it must
//! parse with the host's real manifest parser, use an id the installer
//! accepts, and declare a bounded SemVer requirement that admits the current
//! eov version. (cellaris once shipped a manifest the parser rejected, so it
//! could never load; AUDIT PLG-4.)
//!
//! Submodule plugins are checked when they are checked out. CI sets
//! `EOV_REQUIRE_PLUGIN_MANIFESTS=1`, which turns a missing checkout into a
//! failure instead of a skip.

use semver::{Op, Version, VersionReq};
use std::path::Path;

const PLUGINS: &[&str] = &["annotations", "cellaris", "eovae", "gamepad"];

fn valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit())
        && id.len() <= 64
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn check(directory: &str, text: &str, host: &Version) -> Result<(), String> {
    let manifest = eov_plugin_api::PluginManifest::from_toml(text, directory)
        .map_err(|error| format!("rejected by the host parser: {error}"))?;
    if !valid_id(&manifest.id) {
        return Err(format!("id {:?} is not a valid plugin id", manifest.id));
    }
    if manifest.id != directory {
        return Err(format!("id {:?} differs from its directory", manifest.id));
    }
    Version::parse(&manifest.version)
        .map_err(|error| format!("version {:?}: {error}", manifest.version))?;
    let requirement = &manifest.environment.version;
    let parsed = VersionReq::parse(requirement)
        .map_err(|error| format!("[environment] version {requirement:?}: {error}"))?;
    let bounded = parsed.comparators.iter().any(|comparator| {
        matches!(
            comparator.op,
            Op::Exact | Op::Less | Op::LessEq | Op::Tilde | Op::Caret | Op::Wildcard
        )
    });
    if !bounded {
        return Err(format!(
            "[environment] version {requirement:?} has no upper bound; declare e.g. \">=1.0.0, <2.0.0\""
        ));
    }
    let release = Version::new(host.major, host.minor, host.patch);
    if !parsed.matches(host) && !parsed.matches(&release) {
        return Err(format!(
            "[environment] version {requirement:?} excludes eov {host}"
        ));
    }
    Ok(())
}

#[test]
fn repository_plugin_manifests_load_in_this_eov() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../plugins");
    let host = Version::parse(env!("CARGO_PKG_VERSION")).unwrap();
    let required = std::env::var("EOV_REQUIRE_PLUGIN_MANIFESTS").is_ok_and(|value| value == "1");
    let mut failures = Vec::new();
    let mut checked = 0;
    for directory in PLUGINS {
        let path = root.join(directory).join("plugin.toml");
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                checked += 1;
                if let Err(error) = check(directory, &text, &host) {
                    failures.push(format!("{}: {error}", path.display()));
                }
            }
            Err(error) if required => failures.push(format!("{}: {error}", path.display())),
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        checked > 0,
        "no plugin manifests found under {}",
        root.display()
    );
}

#[test]
fn the_checks_reject_what_the_installer_rejects() {
    let host = Version::parse("1.0.0").unwrap();
    let manifest = |id: &str, environment: &str| {
        format!(
            "id = \"{id}\"\nname = \"Example\"\nversion = \"1.0.0\"\n\n[environment]\nversion = \"{environment}\"\n"
        )
    };
    check("example", &manifest("example", ">=1.0.0, <2.0.0"), &host).unwrap();
    assert!(check("example", &manifest("example", ">=1.0.0"), &host).is_err());
    assert!(check("example", &manifest("example", ">=2.0.0, <3.0.0"), &host).is_err());
    assert!(check("Example", &manifest("Example", "^1"), &host).is_err());
    assert!(check("example", &manifest("other", "^1"), &host).is_err());
    assert!(
        check(
            "example",
            "id = \"example\"\nname = \"x\"\nversion = \"1.0.0\"\n",
            &host
        )
        .is_err()
    );
}
