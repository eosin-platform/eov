//! Loading plugin libraries and creating plugin instances.
//!
//! Each library is opened with `abi_stable::library::lib_header_from_path`
//! and its root module initialised with `LibHeader::init_root_module`, which
//! checks the `eov-plugin-api` major version and the layout of every type in
//! the interface before any plugin code runs (PLG-1). Unlike
//! `RootModule::load_from_file`, this keeps one module per library, so any
//! number of plugins can be loaded.
//!
//! abi_stable refuses a module with fewer fields than the host's copy of
//! its type, so the root module is checked as plugin API 1.0 defined it
//! (every 1.x module extends that) and, for libraries whose module has the
//! fields added since (the slide source), once more with
//! the full layout.

use abi_stable::prefix_type::PrefixRefTrait;
use abi_stable::std_types::RStr;
use eov_plugin_api::abi::{self, HostBox, PluginBox, PluginModuleRef, SlideSourceBox};
use eov_plugin_api::i18n::{isolate, tr};
use eov_plugin_api::protocol::PluginInit;
use semver::Version;
use std::path::{Path, PathBuf};

/// Why a plugin library could not be loaded or started. `Display` says it in
/// English (the session log, the audit log, `eov batch`);
/// [`LoadError::localized`] in the interface's language, for the Plugins
/// panel and messages. What the library loader or the plugin said stays as
/// it is in both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The library file is missing or outside the package.
    BadPath(BadPath),
    /// The library could not be opened or is not an eov plugin (the
    /// loader's words).
    Open(String),
    /// Built against an incompatible plugin API or ABI.
    Incompatible(Incompatible),
    /// The plugin's `create` failed (the plugin's words).
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

impl LoadError {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        match self {
            LoadError::BadPath(m) => tr!("invalid library path: {}", m.localized()),
            LoadError::Open(m) => tr!("could not load the library: {}", m),
            LoadError::Incompatible(m) => tr!("incompatible plugin: {}", m.localized()),
            LoadError::Create(m) => tr!("the plugin failed to start: {}", m),
        }
    }
}

/// Why a plugin's library path is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BadPath {
    /// The plugin id cannot be a file name.
    NotAFileName(String),
    /// A path that cannot be resolved: the path and the file system's
    /// words.
    Unresolved(String),
    /// The library (its file name) resolves outside the package.
    OutsidePackage(String),
}

impl std::fmt::Display for BadPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BadPath::NotAFileName(id) => write!(f, "plugin id '{id}' is not a file name"),
            BadPath::Unresolved(text) => f.write_str(text),
            BadPath::OutsidePackage(file) => write!(f, "{file} resolves outside the package"),
        }
    }
}

impl BadPath {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        match self {
            BadPath::NotAFileName(id) => tr!("plugin id '{}' is not a file name", isolate(id)),
            BadPath::Unresolved(text) => isolate(text),
            BadPath::OutsidePackage(file) => {
                tr!("{} resolves outside the package", isolate(file))
            }
        }
    }
}

/// How a plugin library does not fit this eov.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Incompatible {
    /// Built for another ABI.
    Abi { plugin: u32, host: u32 },
    /// The plugin API version the library gives cannot be read: the version
    /// and the version library's words.
    ApiVersion { version: String, error: String },
    /// Built for another major version of the plugin API.
    Api { plugin: String, host: String },
    /// What abi_stable found wrong with the library's layout (its words).
    Layout(String),
}

impl std::fmt::Display for Incompatible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Incompatible::Abi { plugin, host } => {
                write!(f, "plugin ABI {plugin}, this eov speaks ABI {host}")
            }
            Incompatible::ApiVersion { version, error } => {
                write!(f, "bad API version '{version}': {error}")
            }
            Incompatible::Api { plugin, host } => {
                write!(f, "built for plugin API {plugin}, this eov provides {host}")
            }
            Incompatible::Layout(text) => f.write_str(text),
        }
    }
}

impl Incompatible {
    /// The reason in the interface's language.
    pub fn localized(&self) -> String {
        match self {
            Incompatible::Abi { plugin, host } => tr!(
                "plugin ABI {plugin}, this eov speaks ABI {host}",
                plugin = plugin,
                host = host
            ),
            Incompatible::ApiVersion { version, error } => tr!(
                "bad API version '{version}': {error}",
                version = isolate(version),
                error = error
            ),
            Incompatible::Api { plugin, host } => tr!(
                "built for plugin API {plugin}, this eov provides {host}",
                plugin = isolate(plugin),
                host = isolate(host)
            ),
            Incompatible::Layout(text) => text.clone(),
        }
    }
}

/// A loaded, version-checked root module.
#[derive(Clone, Copy)]
pub struct Module {
    module: Root,
}

#[derive(Clone, Copy)]
enum Root {
    /// Built before slide sources were added.
    V1_0(abi::v1_0::PluginModuleRef),
    Full(PluginModuleRef),
}

impl Module {
    /// Wraps a module obtained in-process (tests, statically linked plugins).
    pub fn from_ref(module: PluginModuleRef) -> Result<Self, LoadError> {
        check_versions(module.abi_version(), module.api_version().as_str())?;
        Ok(Self {
            module: Root::Full(module),
        })
    }

    /// Wraps a module that has only plugin API 1.0's fields.
    pub fn from_1_0(module: abi::v1_0::PluginModuleRef) -> Result<Self, LoadError> {
        check_versions(module.abi_version(), module.api_version().as_str())?;
        Ok(Self {
            module: Root::V1_0(module),
        })
    }

    pub fn api_version(&self) -> &'static str {
        match self.module {
            Root::V1_0(module) => module.api_version().as_str(),
            Root::Full(module) => module.api_version().as_str(),
        }
    }

    /// Calls the plugin's `create`.
    pub fn instantiate(&self, host: HostBox, init: &PluginInit) -> Result<PluginBox, LoadError> {
        let init_json =
            serde_json::to_string(init).map_err(|e| LoadError::Create(e.to_string()))?;
        let create = match self.module {
            Root::V1_0(module) => module.create(),
            Root::Full(module) => module.create(),
        };
        create(host, RStr::from_str(&init_json))
            .into_result()
            .map_err(|e| LoadError::Create(e.message.into_string()))
    }

    /// Whether the library was built with slide sources.
    #[cfg(test)]
    pub fn has_slide_source(&self) -> bool {
        matches!(self.module, Root::Full(module) if module.slide_source().is_some())
    }

    /// The slide source of `instance`, which this module created; `None`
    /// for libraries built before slide sources were added.
    pub fn slide_source(&self, instance: &PluginBox) -> Option<Result<SlideSourceBox, String>> {
        let Root::Full(module) = self.module else {
            return None;
        };
        let get = module.slide_source()?;
        Some(
            get(instance)
                .into_result()
                .map_err(|error| error.message.into_string()),
        )
    }
}

/// Checks the explicit handshake on top of abi_stable's own checks.
pub fn check_versions(abi_version: u32, api_version: &str) -> Result<(), LoadError> {
    if abi_version != abi::ABI_VERSION {
        return Err(LoadError::Incompatible(Incompatible::Abi {
            plugin: abi_version,
            host: abi::ABI_VERSION,
        }));
    }
    let plugin = Version::parse(api_version).map_err(|e| {
        LoadError::Incompatible(Incompatible::ApiVersion {
            version: api_version.to_string(),
            error: e.to_string(),
        })
    })?;
    let host = Version::parse(abi::API_VERSION).expect("valid crate version");
    if plugin.major != host.major {
        return Err(LoadError::Incompatible(Incompatible::Api {
            plugin: plugin.to_string(),
            host: host.to_string(),
        }));
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
        return Err(LoadError::BadPath(BadPath::NotAFileName(id.to_string())));
    }
    let root = dunce::canonicalize(package_root).map_err(|e| {
        LoadError::BadPath(BadPath::Unresolved(format!(
            "{}: {e}",
            package_root.display()
        )))
    })?;
    let path = dunce::canonicalize(root.join(&file_name))
        .map_err(|e| LoadError::BadPath(BadPath::Unresolved(format!("{file_name}: {e}"))))?;
    if !path.starts_with(&root) {
        return Err(LoadError::BadPath(BadPath::OutsidePackage(file_name)));
    }
    Ok(path)
}

/// Opens a plugin library and initialises its root module.
pub fn load_library(path: &Path) -> Result<Module, LoadError> {
    let header = abi_stable::library::lib_header_from_path(path)
        .map_err(|e| LoadError::Open(e.to_string()))?;
    let incompatible = |e: abi_stable::library::LibraryError| match e {
        abi_stable::library::LibraryError::IncompatibleVersionNumber { .. }
        | abi_stable::library::LibraryError::AbiInstability(_) => {
            LoadError::Incompatible(Incompatible::Layout(e.to_string()))
        }
        other => LoadError::Open(other.to_string()),
    };
    // The layout every 1.x module extends; then the newer fields, when the
    // library has them.
    let base = header
        .init_root_module::<abi::v1_0::PluginModuleRef>()
        .map_err(incompatible)?;
    let has_newer = base
        .to_prefix_ref()
        .field_accessibility()
        .at(abi::SLIDE_SOURCE_FIELD)
        .is_accessible();
    if has_newer {
        match header.init_root_module::<PluginModuleRef>() {
            Ok(full) => return Module::from_ref(full),
            Err(error) => tracing::warn!(
                "{}: its slide formats cannot be used: {error}",
                path.display()
            ),
        }
    }
    Module::from_1_0(base)
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
        assert!(matches!(
            library_path(dir.path(), "missing"),
            Err(LoadError::BadPath(_))
        ));
        assert!(matches!(
            library_path(dir.path(), "../x"),
            Err(LoadError::BadPath(_))
        ));
        assert!(matches!(
            library_path(dir.path(), "C:evil"),
            Err(LoadError::BadPath(_))
        ));
    }

    #[test]
    fn non_plugins_fail_to_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(abi::plugin_library_filename("junk"));
        std::fs::write(&path, b"not a library").unwrap();
        assert!(matches!(load_library(&path), Err(LoadError::Open(_))));
    }
}
