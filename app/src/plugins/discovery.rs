//! Plugin packages: inspection and extraction.
//!
//! `.eop` packages (plain or zstd-compressed tar archives) are extracted into
//! the content-addressed package cache ([`package_cache`]) and their
//! manifests read from the extracted plugin root. Which packages are loaded
//! is decided by the plugin inventory (`update::plugins::inventory`), shared
//! by the GUI and `eov plugin`.

#[path = "package_cache.rs"]
pub(crate) mod package_cache;

use eov_plugin_api::{PluginDescriptor, PluginError, PluginManifest, PluginResult};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

const PLUGIN_PACKAGE_EXTENSION: &str = package_cache::PACKAGE_EXTENSION;

/// Parsed package information for management operations.
///
/// Inspection only extracts and parses the manifest. It never loads the
/// package's native library or executes plugin code.
#[derive(Debug, Clone)]
pub struct InspectedPluginPackage {
    pub package_path: PathBuf,
    pub manifest: PluginManifest,
    pub raw_manifest: String,
    pub sha256: Option<String>,
    #[allow(dead_code)]
    pub extracted_root: PathBuf,
}

/// Inspect one `.eop` package regardless of host compatibility.
pub fn inspect_plugin_package(
    package_path: &Path,
    calculate_sha256: bool,
) -> PluginResult<InspectedPluginPackage> {
    if !is_plugin_package(package_path) {
        return Err(PluginError::Other(format!(
            "plugin package must have a .eop extension: {}",
            package_path.display()
        )));
    }
    inspect_plugin_archive(package_path, calculate_sha256)
}

/// Inspect a package staged outside the discoverable `.eop` namespace.
pub(crate) fn inspect_staged_plugin_package(
    package_path: &Path,
    calculate_sha256: bool,
) -> PluginResult<InspectedPluginPackage> {
    inspect_plugin_archive(package_path, calculate_sha256)
}

fn inspect_plugin_archive(
    package_path: &Path,
    calculate_sha256: bool,
) -> PluginResult<InspectedPluginPackage> {
    // Extraction is content-addressed, so the package hash is always known.
    let extracted = package_cache::ensure_extracted(package_path)?;
    let manifest_path = extracted
        .root
        .join(eov_plugin_api::manifest::MANIFEST_FILENAME);
    let raw_manifest = fs::read_to_string(&manifest_path)?;
    let manifest = PluginManifest::from_toml(&raw_manifest, "<package>")?;
    Ok(InspectedPluginPackage {
        package_path: package_path.to_path_buf(),
        manifest,
        raw_manifest,
        sha256: calculate_sha256.then_some(extracted.sha256),
        extracted_root: extracted.root,
    })
}

/// Forget cached metadata for a package that was replaced or removed.
///
/// Extractions are keyed by content and may still be used by a running eov
/// instance, so nothing is deleted here; the package cache's garbage
/// collector removes unreferenced, unlocked extractions at a later startup.
pub fn invalidate_plugin_package_cache(package_path: &Path) -> PluginResult<()> {
    package_cache::PackageCache::default().forget(package_path);
    Ok(())
}

/// Scan `plugin_dir` for plugin package tarballs.
///
/// Returns descriptors sorted by plugin id for deterministic ordering.
/// Invalid plugins are skipped with a warning; they do not prevent other
/// plugins from being discovered.
fn is_plugin_package(path: &Path) -> bool {
    path.is_file() && path.extension().and_then(OsStr::to_str) == Some(PLUGIN_PACKAGE_EXTENSION)
}

fn plugin_cache_dir(_plugin_dir: &Path) -> PathBuf {
    // Discovery runs once at GUI startup: collect unused extractions in the
    // background (entries in use by any running eov stay locked and are kept).
    package_cache::schedule_startup_gc();
    package_cache::PackageCache::default().root().to_path_buf()
}

/// Extracts (or reuses the extraction of) one package and reads its
/// manifest, for the GUI's start-up; also schedules the cache's start-up
/// garbage collection.
pub fn load_descriptor(package_path: &Path) -> PluginResult<PluginDescriptor> {
    try_load_descriptor_from_package(package_path, &plugin_cache_dir(package_path))
}

/// Try to load a single plugin descriptor from a plugin package tarball.
fn try_load_descriptor_from_package(
    package_path: &Path,
    cache_dir: &Path,
) -> PluginResult<PluginDescriptor> {
    let plugin_root = ensure_extracted_plugin_root(package_path, cache_dir)?;
    let manifest_path = plugin_root.join(eov_plugin_api::manifest::MANIFEST_FILENAME);
    let manifest = PluginManifest::from_file(&manifest_path)?;
    Ok(PluginDescriptor {
        root: plugin_root,
        manifest,
    })
}

fn ensure_extracted_plugin_root(package_path: &Path, cache_dir: &Path) -> PluginResult<PathBuf> {
    package_cache::PackageCache::at(cache_dir)
        .ensure_extracted(package_path)
        .map(|extracted| extracted.root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};

    fn write_valid_manifest(dir: &Path, id: &str) {
        let manifest = format!(
            r#"
id = "{id}"
name = "Test Plugin {id}"
version = "0.1.0"
entry_ui = "ui/panel.slint"
entry_component = "Panel"

[environment]
version = ">=0.4.1"
"#
        );
        fs::write(dir.join("plugin.toml"), manifest).unwrap();
        fs::create_dir_all(dir.join("ui")).unwrap();
        fs::write(
            dir.join("ui/panel.slint"),
            "export component Panel inherits Window {}",
        )
        .unwrap();
    }

    fn append_tree(
        builder: &mut tar::Builder<File>,
        source_root: &Path,
        current: &Path,
        archive_prefix: &Path,
    ) {
        for entry in fs::read_dir(current).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path.strip_prefix(source_root).unwrap();
            let archive_path = if archive_prefix.as_os_str().is_empty() {
                relative.to_path_buf()
            } else {
                archive_prefix.join(relative)
            };

            if path.is_dir() {
                builder.append_dir(&archive_path, &path).unwrap();
                append_tree(builder, source_root, &path, archive_prefix);
            } else {
                builder.append_path_with_name(&path, &archive_path).unwrap();
            }
        }
    }

    fn write_plugin_package(
        base: &Path,
        package_name: &str,
        id: &str,
        archive_prefix: Option<&str>,
    ) -> PathBuf {
        let source_dir = base.join(format!("{package_name}_src"));
        fs::create_dir_all(&source_dir).unwrap();
        write_valid_manifest(&source_dir, id);

        let package_path = base.join(format!("{package_name}.eop"));
        let file = File::create(&package_path).unwrap();
        let mut builder = tar::Builder::new(file);
        append_tree(
            &mut builder,
            &source_dir,
            &source_dir,
            Path::new(archive_prefix.unwrap_or("")),
        );
        builder.finish().unwrap();
        package_path
    }

    fn write_plugin_package_with_dot_entries(base: &Path, package_name: &str, id: &str) -> PathBuf {
        let source_dir = base.join(format!("{package_name}_src"));
        fs::create_dir_all(&source_dir).unwrap();
        write_valid_manifest(&source_dir, id);

        let package_path = base.join(format!("{package_name}.eop"));
        let file = File::create(&package_path).unwrap();
        let mut builder = tar::Builder::new(file);
        builder.append_dir(".", &source_dir).unwrap();
        builder
            .append_path_with_name(source_dir.join("plugin.toml"), "./plugin.toml")
            .unwrap();
        builder.append_dir("./ui", source_dir.join("ui")).unwrap();
        builder
            .append_path_with_name(source_dir.join("ui/panel.slint"), "./ui/panel.slint")
            .unwrap();
        builder.finish().unwrap();
        package_path
    }

    fn write_unsafe_package(
        base: &Path,
        package_name: &str,
        path: &str,
        entry_type: tar::EntryType,
    ) -> PathBuf {
        let package_path = base.join(format!("{package_name}.eop"));
        let file = File::create(&package_path).unwrap();
        let mut builder = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(entry_type);
        header.set_size(0);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, path, std::io::empty())
            .unwrap();
        builder.finish().unwrap();
        package_path
    }

    #[test]
    fn packages_load_from_their_extracted_root() {
        let tmp = tempfile::tempdir().unwrap();
        let alpha = write_plugin_package(tmp.path(), "plugin_a", "alpha", None);
        let descriptor = load_descriptor(&alpha).unwrap();
        assert_eq!(descriptor.manifest.id, "alpha");
        assert!(descriptor.root.join("plugin.toml").exists());
    }

    #[test]
    fn packages_with_a_nested_root_load() {
        let tmp = tempfile::tempdir().unwrap();
        let package = write_plugin_package(
            tmp.path(),
            "wrapped_plugin",
            "wrapped",
            Some("plugin_bundle"),
        );
        let descriptor = load_descriptor(&package).unwrap();
        assert_eq!(descriptor.manifest.id, "wrapped");
        assert!(descriptor.root.ends_with("plugin_bundle"));
    }

    #[test]
    fn packages_with_dot_prefixed_entries_load() {
        let tmp = tempfile::tempdir().unwrap();
        let package =
            write_plugin_package_with_dot_entries(tmp.path(), "dot_prefixed", "dot_prefixed");
        let descriptor = load_descriptor(&package).unwrap();
        assert_eq!(descriptor.manifest.id, "dot_prefixed");
        assert!(descriptor.root.join("plugin.toml").exists());
    }

    #[test]
    fn invalid_manifests_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let bad_package = tmp.path().join("bad_plugin.eop");
        let bad_file = File::create(&bad_package).unwrap();
        let mut bad_builder = tar::Builder::new(bad_file);
        let bad_source = tmp.path().join("bad_source");
        fs::create_dir_all(&bad_source).unwrap();
        fs::write(bad_source.join("plugin.toml"), "not valid toml {{{{").unwrap();
        bad_builder
            .append_path_with_name(bad_source.join("plugin.toml"), "plugin.toml")
            .unwrap();
        bad_builder.finish().unwrap();
        assert!(load_descriptor(&bad_package).is_err());
    }

    #[test]
    fn reject_traversal_and_absolute_archive_paths() {
        let tmp = tempfile::tempdir().unwrap();
        for path in [Path::new("../escape"), Path::new("/escape")] {
            let error = package_cache::validate_archive_path(tmp.path(), path).unwrap_err();
            assert!(error.to_string().contains("unsafe"));
        }
    }

    #[test]
    fn reject_symlink_hardlink_and_special_entries() {
        let tmp = tempfile::tempdir().unwrap();
        for (name, entry_type) in [
            ("symlink", tar::EntryType::symlink()),
            ("hardlink", tar::EntryType::hard_link()),
            ("block", tar::EntryType::block_special()),
            ("char", tar::EntryType::character_special()),
            ("fifo", tar::EntryType::fifo()),
        ] {
            let package = write_unsafe_package(tmp.path(), name, "plugin.toml", entry_type);
            assert!(
                inspect_plugin_package(&package, false).is_err(),
                "{name} was accepted"
            );
        }
    }

    #[test]
    fn reject_duplicate_manifest_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let package = tmp.path().join("duplicate.eop");
        let file = File::create(&package).unwrap();
        let mut builder = tar::Builder::new(file);
        for _ in 0..2 {
            let mut header = tar::Header::new_gnu();
            let contents = b"id = \"duplicate\"\nname = \"Duplicate\"\nversion = \"1.0.0\"\n[environment]\nversion = \">=0.4.1\"\n";
            header.set_entry_type(tar::EntryType::file());
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "plugin.toml", &contents[..])
                .unwrap();
        }
        builder.finish().unwrap();
        assert!(inspect_plugin_package(&package, false).is_err());
    }

    fn write_plugin_package_with_version_and_requirement(
        base: &Path,
        package_name: &str,
        id: &str,
        version: &str,
        host_requirement: &str,
    ) -> PathBuf {
        let source_dir = base.join(format!("{package_name}_src"));
        fs::create_dir_all(source_dir.join("ui")).unwrap();
        fs::write(
            source_dir.join("plugin.toml"),
            format!(
                r#"
id = "{id}"
name = "Test Plugin {id}"
version = "{version}"
entry_ui = "ui/panel.slint"
entry_component = "Panel"

[environment]
version = "{host_requirement}"
"#
            ),
        )
        .unwrap();
        fs::write(
            source_dir.join("ui/panel.slint"),
            "export component Panel inherits Window {}",
        )
        .unwrap();

        let package_path = base.join(format!("{package_name}.eop"));
        let file = File::create(&package_path).unwrap();
        let mut builder = tar::Builder::new(file);
        append_tree(&mut builder, &source_dir, &source_dir, Path::new(""));
        builder.finish().unwrap();
        package_path
    }

    #[test]
    fn the_inventory_loads_the_newest_compatible_version() {
        let tmp = tempfile::tempdir().unwrap();
        let requirement = ">=1.0.0, <2";
        for (package, version) in [("example_v1", "1.2.0"), ("example_v2", "1.10.0")] {
            write_plugin_package_with_version_and_requirement(
                tmp.path(),
                package,
                "example",
                version,
                requirement,
            );
        }
        write_plugin_package_with_version_and_requirement(
            tmp.path(),
            "future",
            "future",
            "1.0.0",
            ">=9.0.0",
        );
        let host = semver::Version::parse("1.0.0").unwrap();
        let inventory = crate::update::plugins::inventory(tmp.path(), &[], &host).unwrap();
        assert_eq!(inventory.selected("example").unwrap().version, "1.10.0");
        assert!(inventory.selected("future").is_none());
        assert!(inventory.with_id("future")[0].incompatibility.is_some());
        let selected = inventory.selected("example").unwrap();
        let descriptor = load_descriptor(&selected.package_path).unwrap();
        assert_eq!(descriptor.manifest.version, "1.10.0");
    }
}
