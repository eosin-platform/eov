//! Plugin package scanning.
//!
//! Discovers `.eop` tarballs, extracts them into a cache directory, and loads
//! plugin manifests from the extracted plugin root.

use plugin_api::{PluginDescriptor, PluginError, PluginManifest, PluginResult};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;
use tracing::{debug, info, warn};

const PLUGIN_PACKAGE_EXTENSION: &str = "eop";

/// Scan `plugin_dir` for plugin package tarballs.
///
/// Returns descriptors sorted by plugin id for deterministic ordering.
/// Invalid plugins are skipped with a warning; they do not prevent other
/// plugins from being discovered.
pub fn discover_plugins(plugin_dir: &Path) -> Vec<PluginDescriptor> {
    if !plugin_dir.is_dir() {
        info!(
            "Plugin directory does not exist, skipping discovery: {}",
            plugin_dir.display()
        );
        return Vec::new();
    }

    let read_dir = match std::fs::read_dir(plugin_dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!(
                "Failed to read plugin directory {}: {e}",
                plugin_dir.display()
            );
            return Vec::new();
        }
    };

    let cache_dir = plugin_cache_dir(plugin_dir);
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        warn!(
            "Failed to create plugin cache directory {}: {e}",
            cache_dir.display()
        );
        return Vec::new();
    }

    let mut descriptors = Vec::new();

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
            Ok(desc) => {
                info!(
                    "Discovered plugin '{}' from {}",
                    desc.manifest.id,
                    path.display()
                );
                descriptors.push(desc);
            }
            Err(e) => {
                warn!("Skipping invalid plugin at {}: {e}", path.display());
            }
        }
    }

    // Sort by id for deterministic ordering
    descriptors.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    descriptors
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
    let manifest_path = plugin_root.join(plugin_api::manifest::MANIFEST_FILENAME);
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

    let tmp_dir = extract_dir.with_extension(format!("tmp-{}", std::process::id()));
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

    Ok(format!(
        "{}-{}-{}-{}",
        sanitize_cache_component(stem),
        metadata.len(),
        modified.as_secs(),
        modified.subsec_nanos()
    ))
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

    for entry in archive.entries()? {
        let mut entry = entry?;
        let archive_path = entry.path()?.into_owned();
        validate_archive_path(package_path, &archive_path)?;
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

            if path.file_name() == Some(OsStr::new(plugin_api::manifest::MANIFEST_FILENAME)) {
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
            plugin_api::manifest::MANIFEST_FILENAME
        ))),
        1 => Ok(manifest_roots.pop().unwrap_or_default()),
        _ => Err(PluginError::Other(format!(
            "plugin package extraction '{}' contains multiple {} files",
            extracted_dir.display(),
            plugin_api::manifest::MANIFEST_FILENAME
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

    #[test]
    fn discover_from_nonexistent_dir() {
        let result = discover_plugins(Path::new("/nonexistent/plugin/dir/12345xyz"));
        assert!(result.is_empty());
    }

    #[test]
    fn discover_from_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = discover_plugins(tmp.path());
        assert!(result.is_empty());
    }

    #[test]
    fn discover_valid_plugin_packages() {
        let tmp = tempfile::tempdir().unwrap();

        write_plugin_package(tmp.path(), "plugin_a", "alpha", None);
        write_plugin_package(tmp.path(), "plugin_b", "beta", None);

        let result = discover_plugins(tmp.path());
        assert_eq!(result.len(), 2);
        // Sorted by id
        assert_eq!(result[0].manifest.id, "alpha");
        assert_eq!(result[1].manifest.id, "beta");
        assert!(result[0].root.join("plugin.toml").exists());
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
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].manifest.id, "wrapped");
        assert!(result[0].root.ends_with("plugin_bundle"));
    }

    #[test]
    fn discover_package_with_dot_prefixed_entries() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin_package_with_dot_entries(tmp.path(), "dot_prefixed", "dot_prefixed");

        let result = discover_plugins(tmp.path());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].manifest.id, "dot_prefixed");
        assert!(result[0].root.join("plugin.toml").exists());
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
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].manifest.id, "good");
    }

    #[test]
    fn deterministic_ordering() {
        let tmp = tempfile::tempdir().unwrap();

        for id in ["zulu", "alpha", "mango"] {
            write_plugin_package(tmp.path(), id, id, None);
        }

        let result = discover_plugins(tmp.path());
        let ids: Vec<&str> = result.iter().map(|d| d.manifest.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mango", "zulu"]);
    }
}
