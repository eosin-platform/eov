//! Keeps the shared SemVer table honest: every case in
//! `tests/data/semver-cases.json` must match what the `semver` crate eov
//! uses actually does. The release tooling (`scripts/release_infrastructure.py`)
//! is tested against the same table, so the two can never disagree about
//! which plugin releases a given eov accepts.

use semver::{Version, VersionReq};
use serde_json::Value;

#[test]
fn shared_semver_table_matches_the_semver_crate() {
    let table: Value =
        serde_json::from_str(include_str!("data/semver-cases.json")).expect("valid JSON");
    let versions: Vec<Version> = table["versions"]
        .as_array()
        .expect("versions")
        .iter()
        .map(|version| Version::parse(version.as_str().unwrap()).unwrap())
        .collect();
    let mut failures = Vec::new();
    for case in table["cases"].as_array().expect("cases") {
        let requirement = case["requirement"].as_str().unwrap();
        let valid = case["valid"].as_bool().unwrap();
        match (VersionReq::parse(requirement), valid) {
            (Err(_), false) => {}
            (Ok(_), false) => failures.push(format!("{requirement:?} parses, table says invalid")),
            (Err(error), true) => {
                failures.push(format!("{requirement:?} fails to parse ({error})"))
            }
            (Ok(parsed), true) => {
                let actual: Vec<String> = versions
                    .iter()
                    .filter(|version| parsed.matches(version))
                    .map(Version::to_string)
                    .collect();
                let expected: Vec<String> = case["matches"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|version| version.as_str().unwrap().to_string())
                    .collect();
                if actual != expected {
                    failures.push(format!(
                        "{requirement:?} matches {actual:?}, table says {expected:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
