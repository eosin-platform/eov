//! Plugin package scanning.
//!
//! Discovers `.eop` tarballs, extracts them into a cache directory, and loads
//! plugin manifests from the extracted plugin root.

use eov_plugin_api::{PluginDescriptor, PluginError, PluginManifest, PluginResult};
use semver::{Version, VersionReq};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::UNIX_EPOCH;
use tracing::{debug, info, warn};

const PLUGIN_PACKAGE_EXTENSION: &str = "eop";
static EXTRACT_ATTEMPT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct PluginDiscoveryResult {
    pub descriptors: Vec<PluginDescriptor>,
    pub old_plugin_files: Vec<PathBuf>,
}

struct ScannedPluginPackage {
    descriptor: PluginDescriptor,
    package_path: PathBuf,
}

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
    let cache_dir = plugin_cache_dir(package_path.parent().unwrap_or_else(|| Path::new(".")));
    fs::create_dir_all(&cache_dir)?;
    let extracted_root = ensure_extracted_plugin_root(package_path, &cache_dir)?;
    let manifest_path = extracted_root.join(eov_plugin_api::manifest::MANIFEST_FILENAME);
    let raw_manifest = fs::read_to_string(&manifest_path)?;
    let manifest = PluginManifest::from_toml(&raw_manifest, "<package>")?;
    let sha256 = calculate_sha256
        .then(|| sha256_file(package_path))
        .transpose()?;
    Ok(InspectedPluginPackage {
        package_path: package_path.to_path_buf(),
        manifest,
        raw_manifest,
        sha256,
        extracted_root,
    })
}

/// Remove cached extraction data for one package when it is replaced or
/// removed. A later inspection will rebuild only this package's extraction.
pub fn invalidate_plugin_package_cache(package_path: &Path) -> PluginResult<()> {
    let cache_dir = plugin_cache_dir(package_path.parent().unwrap_or_else(|| Path::new(".")));
    if !cache_dir.is_dir() {
        return Ok(());
    }
    let prefix = package_path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(sanitize_cache_component)
        .unwrap_or_else(|| "plugin".to_string());
    for entry in fs::read_dir(cache_dir)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(&format!("{prefix}-"))
        {
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            }
        }
    }
    Ok(())
}

/// Scan `plugin_dir` for plugin package tarballs.
///
/// Returns descriptors sorted by plugin id for deterministic ordering.
/// Invalid plugins are skipped with a warning; they do not prevent other
/// plugins from being discovered.
pub fn discover_plugins(plugin_dir: &Path) -> PluginDiscoveryResult {
    let host_version = Version::parse(crate::version::BUILD_VERSION)
        .expect("CARGO_PKG_VERSION must be a valid semantic version");
    discover_plugins_for_host_version(plugin_dir, &host_version)
}

fn discover_plugins_for_host_version(
    plugin_dir: &Path,
    host_version: &Version,
) -> PluginDiscoveryResult {
    if !plugin_dir.is_dir() {
        info!(
            "Plugin directory does not exist, skipping discovery: {}",
            plugin_dir.display()
        );
        return PluginDiscoveryResult {
            descriptors: Vec::new(),
            old_plugin_files: Vec::new(),
        };
    }

    let read_dir = match std::fs::read_dir(plugin_dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!(
                "Failed to read plugin directory {}: {e}",
                plugin_dir.display()
            );
            return PluginDiscoveryResult {
                descriptors: Vec::new(),
                old_plugin_files: Vec::new(),
            };
        }
    };

    let cache_dir = plugin_cache_dir(plugin_dir);
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        warn!(
            "Failed to create plugin cache directory {}: {e}",
            cache_dir.display()
        );
        return PluginDiscoveryResult {
            descriptors: Vec::new(),
            old_plugin_files: Vec::new(),
        };
    }

    let mut packages = Vec::new();

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Error reading plugin directory entry: {e}");
                continue;
            }
        };

        let path = entry.path();
        if !is_plugin_package(&path) {
            debug!("Skipping non-plugin package entry: {}", path.display());
            continue;
        }

        match try_load_descriptor_from_package(&path, &cache_dir) {
            Ok(desc) => match host_version_satisfies_requirement(
                &desc.manifest.environment.version,
                host_version,
            ) {
                Ok(true) => {
                    info!(
                        "Discovered compatible plugin '{}' from {}",
                        desc.manifest.id,
                        path.display()
                    );
                    packages.push(ScannedPluginPackage {
                        descriptor: desc,
                        package_path: path,
                    });
                }
                Ok(false) => {
                    warn!(
                        "Skipping incompatible plugin '{}' from {}: requires host version '{}', running {}",
                        desc.manifest.id,
                        path.display(),
                        desc.manifest.environment.version,
                        host_version
                    );
                }
                Err(err) => {
                    warn!(
                        "Skipping plugin '{}' from {}: invalid host version requirement '{}': {}",
                        desc.manifest.id,
                        path.display(),
                        desc.manifest.environment.version,
                        err
                    );
                }
            },
            Err(e) => {
                warn!("Skipping invalid plugin at {}: {e}", path.display());
            }
        }
    }

    dedupe_plugin_versions(packages)
}

fn host_version_satisfies_requirement(
    requirement: &str,
    host_version: &Version,
) -> Result<bool, semver::Error> {
    Ok(VersionReq::parse(requirement)?.matches(host_version))
}

fn dedupe_plugin_versions(packages: Vec<ScannedPluginPackage>) -> PluginDiscoveryResult {
    let mut grouped = BTreeMap::<String, Vec<ScannedPluginPackage>>::new();
    for package in packages {
        grouped
            .entry(package.descriptor.manifest.id.clone())
            .or_default()
            .push(package);
    }

    let mut descriptors = Vec::new();
    let mut old_plugin_files = Vec::new();

    for (plugin_id, mut group) in grouped {
        group.sort_by(|left, right| {
            compare_plugin_versions(
                &right.descriptor.manifest.version,
                &left.descriptor.manifest.version,
            )
            .then_with(|| left.package_path.cmp(&right.package_path))
        });

        let winner_version = group[0].descriptor.manifest.version.clone();

        for package in group.drain(1..) {
            match compare_plugin_versions(&winner_version, &package.descriptor.manifest.version) {
                Ordering::Greater => {
                    warn!(
                        "Skipping older plugin '{}' version {} from {} in favor of version {}",
                        plugin_id,
                        package.descriptor.manifest.version,
                        package.package_path.display(),
                        winner_version
                    );
                    old_plugin_files.push(package.package_path);
                }
                Ordering::Equal => {
                    warn!(
                        "Skipping duplicate plugin '{}' version {} from {}",
                        plugin_id,
                        package.descriptor.manifest.version,
                        package.package_path.display()
                    );
                }
                Ordering::Less => {}
            }
        }

        let winner = group
            .into_iter()
            .next()
            .map(|package| package.descriptor)
            .expect("plugin discovery group must contain a winner");
        descriptors.push(winner);
    }

    descriptors.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    old_plugin_files.sort();

    PluginDiscoveryResult {
        descriptors,
        old_plugin_files,
    }
}

fn compare_plugin_versions(left: &str, right: &str) -> Ordering {
    match (Version::parse(left), Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn is_plugin_package(path: &Path) -> bool {
    path.is_file() && path.extension().and_then(OsStr::to_str) == Some(PLUGIN_PACKAGE_EXTENSION)
}

fn plugin_cache_dir(plugin_dir: &Path) -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| plugin_dir.join(".cache"))
        .join("eov")
        .join("plugin-packages")
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
    let extract_dir = cache_dir.join(package_cache_key(package_path)?);

    if extract_dir.exists() {
        match find_manifest_root(&extract_dir) {
            Ok(root) => return Ok(root),
            Err(err) => {
                warn!(
                    "Discarding invalid cached plugin extraction {}: {err}",
                    extract_dir.display()
                );
                fs::remove_dir_all(&extract_dir)?;
            }
        }
    }

    let tmp_dir = extract_dir.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        EXTRACT_ATTEMPT_COUNTER.fetch_add(1, AtomicOrdering::Relaxed)
    ));
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir_all(&tmp_dir)?;

    if let Err(err) = extract_plugin_package(package_path, &tmp_dir) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err(err);
    }

    let relative_root = match find_manifest_root_relative(&tmp_dir) {
        Ok(root) => root,
        Err(err) => {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(err);
        }
    };

    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::rename(&tmp_dir, &extract_dir)?;

    Ok(join_root(&extract_dir, &relative_root))
}

fn package_cache_key(package_path: &Path) -> PluginResult<String> {
    let metadata = fs::metadata(package_path)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    let stem = package_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("plugin");
    let path_hash = hash_path_for_cache(package_path);

    Ok(format!(
        "{}-{}-{}-{}-{:016x}",
        sanitize_cache_component(stem),
        metadata.len(),
        modified.as_secs(),
        modified.subsec_nanos(),
        path_hash
    ))
}

fn hash_path_for_cache(package_path: &Path) -> u64 {
    let stable_path = package_path
        .canonicalize()
        .unwrap_or_else(|_| package_path.to_path_buf());
    let mut hasher = DefaultHasher::new();
    stable_path.hash(&mut hasher);
    hasher.finish()
}

fn sanitize_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn extract_plugin_package(package_path: &Path, destination: &Path) -> PluginResult<()> {
    let file = File::open(package_path)?;
    let mut archive = tar::Archive::new(file);
    let mut manifest_paths = std::collections::HashSet::new();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let archive_path = entry.path()?.into_owned();
        validate_archive_path(package_path, &archive_path)?;
        let entry_type = entry.header().entry_type();
        if !(entry_type.is_file() || entry_type.is_dir()) {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains unsupported archive entry '{}'",
                package_path.display(),
                archive_path.display()
            )));
        }
        if archive_path.file_name() == Some(OsStr::new(eov_plugin_api::manifest::MANIFEST_FILENAME))
            && !manifest_paths.insert(archive_path.clone())
        {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains duplicate {} entries",
                package_path.display(),
                eov_plugin_api::manifest::MANIFEST_FILENAME
            )));
        }
        let unpacked = entry.unpack_in(destination)?;
        if !unpacked {
            return Err(PluginError::Other(format!(
                "plugin package '{}' contains an invalid path '{}'",
                package_path.display(),
                archive_path.display()
            )));
        }
    }

    Ok(())
}

fn validate_archive_path(package_path: &Path, archive_path: &Path) -> PluginResult<()> {
    if archive_path.as_os_str().is_empty() {
        return Err(PluginError::Other(format!(
            "plugin package '{}' contains an empty path",
            package_path.display()
        )));
    }

    let raw_path = archive_path.to_string_lossy();
    let has_windows_prefix = raw_path.len() >= 2 && raw_path.as_bytes().get(1) == Some(&b':')
        || raw_path.starts_with("\\\\")
        || raw_path.starts_with('\\');
    if has_windows_prefix {
        return Err(PluginError::Other(format!(
            "plugin package '{}' contains an unsafe Windows path '{}'",
            package_path.display(),
            archive_path.display()
        )));
    }

    for component in archive_path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(PluginError::Other(format!(
                    "plugin package '{}' contains an unsafe path '{}'",
                    package_path.display(),
                    archive_path.display()
                )));
            }
        }
    }

    Ok(())
}

fn sha256_file(path: &Path) -> PluginResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn find_manifest_root(extracted_dir: &Path) -> PluginResult<PathBuf> {
    let relative_root = find_manifest_root_relative(extracted_dir)?;
    Ok(join_root(extracted_dir, &relative_root))
}

fn join_root(base: &Path, relative_root: &Path) -> PathBuf {
    if relative_root.as_os_str().is_empty() {
        base.to_path_buf()
    } else {
        base.join(relative_root)
    }
}

fn find_manifest_root_relative(extracted_dir: &Path) -> PluginResult<PathBuf> {
    let mut dirs_to_visit = vec![extracted_dir.to_path_buf()];
    let mut manifest_roots = Vec::new();

    while let Some(dir) = dirs_to_visit.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                dirs_to_visit.push(path);
                continue;
            }

            if path.file_name() == Some(OsStr::new(eov_plugin_api::manifest::MANIFEST_FILENAME)) {
                manifest_roots.push(
                    path.parent()
                        .unwrap_or(extracted_dir)
                        .strip_prefix(extracted_dir)
                        .map(Path::to_path_buf)
                        .map_err(|err| {
                            PluginError::Other(format!(
                                "failed to resolve plugin manifest root in {}: {err}",
                                extracted_dir.display()
                            ))
                        })?,
                );
            }
        }
    }

    match manifest_roots.len() {
        0 => Err(PluginError::Other(format!(
            "plugin package extraction '{}' does not contain {}",
            extracted_dir.display(),
            eov_plugin_api::manifest::MANIFEST_FILENAME
        ))),
        1 => Ok(manifest_roots.pop().unwrap_or_default()),
        _ => Err(PluginError::Other(format!(
            "plugin package extraction '{}' contains multiple {} files",
            extracted_dir.display(),
            eov_plugin_api::manifest::MANIFEST_FILENAME
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
    fn discover_from_nonexistent_dir() {
        let result = discover_plugins(Path::new("/nonexistent/plugin/dir/12345xyz"));
        assert!(result.descriptors.is_empty());
        assert!(result.old_plugin_files.is_empty());
    }

    #[test]
    fn discover_from_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = discover_plugins(tmp.path());
        assert!(result.descriptors.is_empty());
        assert!(result.old_plugin_files.is_empty());
    }

    #[test]
    fn discover_valid_plugin_packages() {
        let tmp = tempfile::tempdir().unwrap();

        write_plugin_package(tmp.path(), "plugin_a", "alpha", None);
        write_plugin_package(tmp.path(), "plugin_b", "beta", None);

        let result = discover_plugins(tmp.path());
        assert_eq!(result.descriptors.len(), 2);
        // Sorted by id
        assert_eq!(result.descriptors[0].manifest.id, "alpha");
        assert_eq!(result.descriptors[1].manifest.id, "beta");
        assert!(result.descriptors[0].root.join("plugin.toml").exists());
        assert!(result.old_plugin_files.is_empty());
    }

    #[test]
    fn discover_package_with_nested_root() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin_package(
            tmp.path(),
            "wrapped_plugin",
            "wrapped",
            Some("plugin_bundle"),
        );

        let result = discover_plugins(tmp.path());
        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].manifest.id, "wrapped");
        assert!(result.descriptors[0].root.ends_with("plugin_bundle"));
    }

    #[test]
    fn discover_package_with_dot_prefixed_entries() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin_package_with_dot_entries(tmp.path(), "dot_prefixed", "dot_prefixed");

        let result = discover_plugins(tmp.path());
        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].manifest.id, "dot_prefixed");
        assert!(result.descriptors[0].root.join("plugin.toml").exists());
    }

    #[test]
    fn skip_invalid_continue_valid() {
        let tmp = tempfile::tempdir().unwrap();

        write_plugin_package(tmp.path(), "good_plugin", "good", None);

        fs::write(tmp.path().join("ignore-me.txt"), "not a plugin").unwrap();
        let legacy_dir = tmp.path().join("legacy_plugin_dir");
        fs::create_dir(&legacy_dir).unwrap();
        write_valid_manifest(&legacy_dir, "legacy");

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

        let result = discover_plugins(tmp.path());
        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].manifest.id, "good");
    }

    #[test]
    fn reject_traversal_and_absolute_archive_paths() {
        let tmp = tempfile::tempdir().unwrap();
        for path in [Path::new("../escape"), Path::new("/escape")] {
            let error = validate_archive_path(tmp.path(), path).unwrap_err();
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

    #[test]
    fn deterministic_ordering() {
        let tmp = tempfile::tempdir().unwrap();

        for id in ["zulu", "alpha", "mango"] {
            write_plugin_package(tmp.path(), id, id, None);
        }

        let result = discover_plugins(tmp.path());
        let ids: Vec<&str> = result
            .descriptors
            .iter()
            .map(|d| d.manifest.id.as_str())
            .collect();
        assert_eq!(ids, vec!["alpha", "mango", "zulu"]);
    }

    fn write_plugin_package_with_version(
        base: &Path,
        package_name: &str,
        id: &str,
        version: &str,
    ) -> PathBuf {
        write_plugin_package_with_version_and_requirement(
            base,
            package_name,
            id,
            version,
            ">=0.4.1",
        )
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
    fn semver_compatibility_supports_standard_requirements() {
        let host_version = Version::parse("0.3.5").unwrap();

        for requirement in [
            ">0.3.4",
            ">=0.3.4",
            "^0.3.4",
            "~0.3.4",
            ">=0.3.4, <0.4.0",
            "0.3.*",
            "*",
        ] {
            assert!(
                host_version_satisfies_requirement(requirement, &host_version).unwrap(),
                "expected host version {host_version} to satisfy {requirement}"
            );
        }

        assert!(host_version_satisfies_requirement("0.3.4", &host_version).unwrap());
        assert!(!host_version_satisfies_requirement("=0.3.4", &host_version).unwrap());
        assert!(!host_version_satisfies_requirement(">=0.4.0", &host_version).unwrap());
        assert!(host_version_satisfies_requirement("0.3.5", &host_version).unwrap());
        assert!(host_version_satisfies_requirement("^0.3.0", &host_version).unwrap());
    }

    #[test]
    fn discover_skips_incompatible_and_invalid_requirements() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin_package_with_version_and_requirement(
            tmp.path(),
            "compatible",
            "compatible",
            "1.0.0",
            ">=0.3.4, <0.4.0",
        );
        write_plugin_package_with_version_and_requirement(
            tmp.path(),
            "incompatible",
            "incompatible",
            "1.0.0",
            ">=0.4.0",
        );
        write_plugin_package_with_version_and_requirement(
            tmp.path(),
            "invalid",
            "invalid",
            "1.0.0",
            "not a semver requirement",
        );

        let host_version = Version::parse("0.3.5").unwrap();
        let result = discover_plugins_for_host_version(tmp.path(), &host_version);

        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].manifest.id, "compatible");
        assert!(result.old_plugin_files.is_empty());
    }

    #[test]
    fn discover_keeps_latest_plugin_version_and_reports_old_files() {
        let tmp = tempfile::tempdir().unwrap();
        let old_path =
            write_plugin_package_with_version(tmp.path(), "example_v1", "example", "1.2.0");
        let new_path =
            write_plugin_package_with_version(tmp.path(), "example_v2", "example", "1.10.0");
        write_plugin_package_with_version(tmp.path(), "other_v1", "other", "0.1.0");

        let result = discover_plugins(tmp.path());

        assert_eq!(result.descriptors.len(), 2);
        let example = result
            .descriptors
            .iter()
            .find(|descriptor| descriptor.manifest.id == "example")
            .unwrap();
        assert_eq!(example.manifest.version, "1.10.0");
        assert_eq!(result.old_plugin_files, vec![old_path]);
        assert_ne!(example.root.join("plugin.toml"), new_path);
    }

    #[test]
    fn discover_skips_same_version_duplicates_without_prune_warning() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin_package_with_version(tmp.path(), "example_a", "example", "1.2.0");
        write_plugin_package_with_version(tmp.path(), "example_b", "example", "1.2.0");

        let result = discover_plugins(tmp.path());

        assert_eq!(result.descriptors.len(), 1);
        assert!(result.old_plugin_files.is_empty());
    }
}
