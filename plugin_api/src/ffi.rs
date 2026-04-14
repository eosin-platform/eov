//! Stable ABI types for the plugin dynamic library interface.
//!
//! Uses [`abi_stable`] to define types that can safely cross the Rust dynamic
//! library boundary. The plugin exports a `#[no_mangle]` function that returns
//! a [`PluginVTable`], and the host loads it with
//! [`abi_stable::library::RawLibrary`].

use abi_stable::std_types::{RString, RVec};
use abi_stable::StableAbi;

/// FFI-safe toolbar button registration data.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ToolbarButtonFFI {
    pub button_id: RString,
    pub tooltip: RString,
    pub icon_svg: RString,
    pub action_id: RString,
}

/// FFI-safe response from a plugin action handler.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ActionResponseFFI {
    /// If `true`, the host should open this plugin's `.slint` UI window.
    pub open_window: bool,
}

/// VTable of function pointers exported by each plugin shared library.
///
/// Each plugin crate exports:
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable { ... }
/// ```
#[repr(C)]
#[derive(StableAbi, Copy, Clone)]
pub struct PluginVTable {
    /// Returns the toolbar buttons this plugin wants to register.
    pub get_toolbar_buttons: extern "C" fn() -> RVec<ToolbarButtonFFI>,

    /// Called when a toolbar button registered by this plugin is clicked.
    pub on_action: extern "C" fn(action_id: RString) -> ActionResponseFFI,

    /// Called when a callback defined in the plugin's `.slint` UI is invoked.
    /// `callback_name` is the kebab-case name of the callback as declared in
    /// the `.slint` file.
    pub on_ui_callback: extern "C" fn(callback_name: RString),
}

/// The type of the factory function each plugin exports.
pub type GetPluginVTableFn = unsafe extern "C" fn() -> PluginVTable;

/// Symbol name the host looks for in plugin shared libraries.
pub const PLUGIN_VTABLE_SYMBOL: &[u8] = b"eov_get_plugin_vtable\0";

/// Returns the expected shared library filename for a plugin on the current
/// platform, derived from the plugin id.
pub fn plugin_library_filename(id: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{id}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{id}.dll")
    } else {
        format!("lib{id}.so")
    }
}
