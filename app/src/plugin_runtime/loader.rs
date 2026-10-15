//! Loading plugin libraries and creating plugin instances.
//!
//! Each library is opened with `abi_stable::library::lib_header_from_path`
//! and its root module initialised with `LibHeader::init_root_module`, which
//! checks the `eov-plugin-api` major version and the layout of every type in
//! the interface before any plugin code runs (PLG-1). Unlike
//! `RootModule::load_from_file`, this keeps one module per library, so any
//! number of plugins can be loaded.

use eov_plugin_api::abi::{self, HostBox, PluginBox, PluginModuleRef};
use eov_plugin_api::protocol::PluginInit;
use abi_stable::std_types::RStr;
use semver::Version;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The library file is missing or outside the package.
    BadPath(String),
    /// The library could not be opened or is not an eov plugin.
    Open(String),
    /// Built against an incompatible plugin API or ABI.
    Incompatible(String),
    /// The plugin's `create` failed.
    Create(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::BadPath(m) => write!(f, "invalid library path: {m}"),
            LoadError::Open(m) => write!(f, "could not load the library: {m}"),
            LoadError::Incompatible(m) => write!(f, "incompatible plugin: {m}"),
            LoadError::Create(m) => write!(f, "the plugin failed to start: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A loaded, version-checked root module.
#[derive(Clone, Copy)]
pub struct Module {
    module: PluginModuleRef,
}

impl Module {
    /// Wraps a module obtained in-process (tests, statically linked plugins).
    pub fn from_ref(module: PluginModuleRef) -> Result<Self, LoadError> {
        check_versions(module.abi_version(), module.api_version().as_str())?;
        Ok(Self { module })
    }

    pub fn api_version(&self) -> &'static str {
        self.module.api_version().as_str()
    }

    /// Calls the plugin's `create`.
    pub fn instantiate(&self, host: HostBox, init: &PluginInit) -> Result<PluginBox, LoadError> {
        let init_json = serde_json::to_string(init).map_err(|e| LoadError::Create(e.to_string()))?;
        let create = self.module.create();
        create(host, RStr::from_str(&init_json))
            .into_result()
            .map_err(|e| LoadError::Create(e.message.into_string()))
    }
}

/// Checks the explicit handshake on top of abi_stable's own checks.
pub fn check_versions(abi_version: u32, api_version: &str) -> Result<(), LoadError> {
    if abi_version != abi::ABI_VERSION {
        return Err(LoadError::Incompatible(format!(
            "plugin ABI {abi_version}, this eov speaks ABI {}",
            abi::ABI_VERSION
        )));
    }
    let plugin = Version::parse(api_version)
        .map_err(|e| LoadError::Incompatible(format!("bad API version '{api_version}': {e}")))?;
    let host = Version::parse(abi::API_VERSION).expect("valid crate version");
    if plugin.major != host.major {
        return Err(LoadError::Incompatible(format!(
            "built for plugin API {plugin}, this eov provides {host}"
        )));
    }
    if plugin.minor > host.minor {
        // Allowed (newer methods answer "unsupported"), but worth knowing.
        tracing::info!(
            "plugin built against plugin API {plugin}; this eov provides {host}, newer calls will be unsupported"
        );
    }
    Ok(())
}

/// The library of plugin `id` inside `package_root`, refusing paths that
/// escape the package (PLG-13).
pub fn library_path(package_root: &Path, id: &str) -> Result<PathBuf, LoadError> {
    let file_name = abi::plugin_library_filename(id);
    if file_name.contains(['/', '\\', ':']) {
        return Err(LoadError::BadPath(format!("plugin id '{id}' is not a file name")));
    }
    let root = dunce::canonicalize(package_root)
        .map_err(|e| LoadError::BadPath(format!("{}: {e}", package_root.display())))?;
    let path = dunce::canonicalize(root.join(&file_name))
        .map_err(|e| LoadError::BadPath(format!("{file_name}: {e}")))?;
    if !path.starts_with(&root) {
        return Err(LoadError::BadPath(format!("{file_name} resolves outside the package")));
    }
    Ok(path)
}

/// Opens a plugin library and initialises its root module.
pub fn load_library(path: &Path) -> Result<Module, LoadError> {
    let header = abi_stable::library::lib_header_from_path(path)
        .map_err(|e| LoadError::Open(e.to_string()))?;
    let module = header
        .init_root_module::<PluginModuleRef>()
        .map_err(|e| match e {
            abi_stable::library::LibraryError::IncompatibleVersionNumber { .. }
            | abi_stable::library::LibraryError::AbiInstability(_) => {
                LoadError::Incompatible(e.to_string())
            }
            other => LoadError::Open(other.to_string()),
        })?;
    Module::from_ref(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_handshake() {
        assert!(check_versions(abi::ABI_VERSION, abi::API_VERSION).is_ok());
        assert!(matches!(
            check_versions(abi::ABI_VERSION + 1, abi::API_VERSION),
            Err(LoadError::Incompatible(_))
        ));
        assert!(matches!(
            check_versions(abi::ABI_VERSION, "0.4.9"),
            Err(LoadError::Incompatible(_))
        ));
        assert!(check_versions(abi::ABI_VERSION, "1.99.0").is_ok());
        assert!(check_versions(abi::ABI_VERSION, "not-a-version").is_err());
    }

    #[test]
    fn library_paths_stay_inside_the_package() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(abi::plugin_library_filename("good")), b"").unwrap();
        assert!(library_path(dir.path(), "good").is_ok());
        assert!(matches!(library_path(dir.path(), "missing"), Err(LoadError::BadPath(_))));
        assert!(matches!(library_path(dir.path(), "../x"), Err(LoadError::BadPath(_))));
        assert!(matches!(library_path(dir.path(), "C:evil"), Err(LoadError::BadPath(_))));
    }

    #[test]
    fn non_plugins_fail_to_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(abi::plugin_library_filename("junk"));
        std::fs::write(&path, b"not a library").unwrap();
        assert!(matches!(load_library(&path), Err(LoadError::Open(_))));
    }
}
