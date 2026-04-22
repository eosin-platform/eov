//! Stable ABI types for the plugin dynamic library interface.
//!
//! Uses [`abi_stable`] to define types that can safely cross the Rust dynamic
//! library boundary. The plugin exports a `#[no_mangle]` function that returns
//! a [`PluginVTable`], and the host loads it with
//! [`abi_stable::library::RawLibrary`].

use abi_stable::StableAbi;
use abi_stable::std_types::{ROption, RResult, RString, RVec};

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ActiveSidebarFFI {
    pub plugin_id: RString,
    pub button_id: ROption<RString>,
    pub width_px: u32,
    pub ui_path: RString,
    pub component: RString,
}

/// FFI-safe toolbar button registration data.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ToolbarButtonFFI {
    pub button_id: RString,
    pub tooltip: RString,
    pub icon_svg: RString,
    pub action_id: RString,
    pub tool_mode: ROption<HostToolModeFFI>,
    pub hotkey: ROption<RString>,
}

/// FFI-safe HUD toolbar button registration data.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct HudToolbarButtonFFI {
    pub button_id: RString,
    pub tooltip: RString,
    pub icon_svg: RString,
    pub action_id: RString,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct UiPropertyFFI {
    pub name: RString,
    pub json_value: RString,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ViewportContextMenuItemFFI {
    pub item_id: RString,
    pub label: RString,
    pub icon: RString,
    pub enabled: bool,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ViewportOverlayPointFFI {
    pub annotation_id: RString,
    pub x_level0: f64,
    pub y_level0: f64,
    pub diameter_px: f32,
    pub ring_red: u8,
    pub ring_green: u8,
    pub ring_blue: u8,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ConfirmationDialogRequestFFI {
    pub title: RString,
    pub message: RString,
    pub confirm_label: RString,
    pub cancel_label: RString,
    pub confirm_callback: ROption<RString>,
    pub confirm_args_json: ROption<RString>,
    pub cancel_callback: ROption<RString>,
    pub cancel_args_json: ROption<RString>,
}

/// FFI-safe response from a plugin action handler.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ActionResponseFFI {
    /// If `true`, the host should open this plugin's `.slint` UI window.
    pub open_window: bool,
}

#[repr(u32)]
#[derive(StableAbi, Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostLogLevelFFI {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

#[repr(u32)]
#[derive(StableAbi, Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostToolModeFFI {
    Navigate = 0,
    RegionOfInterest = 1,
    MeasureDistance = 2,
    PointAnnotation = 3,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct OpenFileInfoFFI {
    pub file_id: i32,
    pub path: RString,
    pub filename: RString,
    pub width: u64,
    pub height: u64,
    pub level_count: u32,
    pub vendor: ROption<RString>,
    pub mpp_x: ROption<f64>,
    pub mpp_y: ROption<f64>,
    pub objective_power: ROption<f64>,
    pub scan_date: ROption<RString>,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ViewportSnapshotFFI {
    pub pane_index: u32,
    pub file_id: i32,
    pub file_path: RString,
    pub filename: RString,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub width: f64,
    pub height: f64,
    pub image_width: f64,
    pub image_height: f64,
    pub bounds_left: f64,
    pub bounds_top: f64,
    pub bounds_right: f64,
    pub bounds_bottom: f64,
}

#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct HostSnapshotFFI {
    pub app_name: RString,
    pub app_version: RString,
    pub render_backend: RString,
    pub filtering_mode: RString,
    pub split_enabled: bool,
    pub focused_pane: u32,
    pub open_files: RVec<OpenFileInfoFFI>,
    pub active_file: ROption<OpenFileInfoFFI>,
    pub active_viewport: ROption<ViewportSnapshotFFI>,
    pub recent_files: RVec<RString>,
    pub active_sidebar: ROption<ActiveSidebarFFI>,
}

#[repr(C)]
#[derive(StableAbi, Copy, Clone)]
pub struct HostApiVTable {
    pub context: u64,
    pub get_snapshot: extern "C" fn(context: u64) -> HostSnapshotFFI,
    pub read_region: extern "C" fn(
        context: u64,
        file_id: i32,
        level: u32,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
    ) -> RResult<RVec<u8>, RString>,
    pub open_file: extern "C" fn(context: u64, path: RString) -> RResult<(), RString>,
    pub set_active_viewport: extern "C" fn(
        context: u64,
        center_x: f64,
        center_y: f64,
        zoom: f64,
    ) -> RResult<(), RString>,
    pub fit_active_viewport: extern "C" fn(context: u64) -> RResult<(), RString>,
    pub frame_active_rect: extern "C" fn(
        context: u64,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    ) -> RResult<(), RString>,
    pub set_active_tool:
        extern "C" fn(context: u64, tool: HostToolModeFFI) -> RResult<(), RString>,
    pub request_render: extern "C" fn(context: u64) -> RResult<(), RString>,
    pub set_toolbar_button_active:
        extern "C" fn(context: u64, button_id: RString, active: bool) -> RResult<(), RString>,
    pub set_hud_toolbar_button_active:
        extern "C" fn(context: u64, button_id: RString, active: bool) -> RResult<(), RString>,
    pub show_sidebar: extern "C" fn(
        context: u64,
        button_id: RString,
        width_px: u32,
        ui_path: RString,
        component: RString,
    ) -> RResult<(), RString>,
    pub refresh_sidebar: extern "C" fn(context: u64) -> RResult<(), RString>,
    pub hide_sidebar: extern "C" fn(context: u64) -> RResult<(), RString>,
    pub show_confirmation_dialog:
        extern "C" fn(context: u64, request: ConfirmationDialogRequestFFI) -> RResult<(), RString>,
    pub save_file_dialog: extern "C" fn(
        context: u64,
        default_file_name: RString,
        filter_name: RString,
        extension: RString,
    ) -> RResult<RString, RString>,
    pub log_message: extern "C" fn(context: u64, level: HostLogLevelFFI, message: RString),
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
    /// Supplies the plugin with a host API vtable so it can query the app and
    /// issue host commands later during callbacks.
    pub set_host_api: extern "C" fn(host_api: HostApiVTable),

    /// Returns the toolbar buttons this plugin wants to register.
    pub get_toolbar_buttons: extern "C" fn() -> RVec<ToolbarButtonFFI>,

    /// Returns HUD toolbar buttons shown inside each viewport.
    pub get_hud_toolbar_buttons: extern "C" fn() -> RVec<HudToolbarButtonFFI>,

    /// Called when a toolbar button registered by this plugin is clicked.
    pub on_action: extern "C" fn(action_id: RString) -> ActionResponseFFI,

    /// Called when a HUD toolbar button is clicked in a specific viewport.
    pub on_hud_action:
        extern "C" fn(action_id: RString, viewport: ViewportSnapshotFFI) -> ActionResponseFFI,

    /// Called when a callback defined in the plugin's `.slint` UI is invoked.
    /// `callback_name` is the kebab-case name of the callback as declared in
    /// the `.slint` file.
    pub on_ui_callback: extern "C" fn(callback_name: RString, args_json: RString),

    /// Returns the current values for public properties on the active sidebar.
    /// Values are encoded as JSON and converted to typed Slint values by the host.
    pub get_sidebar_properties: extern "C" fn() -> RVec<UiPropertyFFI>,

    /// Returns plugin-contributed viewport context menu items for the given viewport.
    pub get_viewport_context_menu_items:
        extern "C" fn(viewport: ViewportSnapshotFFI) -> RVec<ViewportContextMenuItemFFI>,

    /// Called when a plugin-contributed viewport context menu item is selected.
    pub on_viewport_context_menu_action:
        extern "C" fn(item_id: RString, viewport: ViewportSnapshotFFI) -> ActionResponseFFI,

    /// Returns point overlay data that should be rendered over the given viewport.
    pub get_viewport_overlay_points:
        extern "C" fn(viewport: ViewportSnapshotFFI) -> RVec<ViewportOverlayPointFFI>,

    /// Called when the host commits a point annotation click for the active point tool.
    pub on_point_annotation_placed:
        extern "C" fn(viewport: ViewportSnapshotFFI, x_level0: f64, y_level0: f64),

    /// Called when the host moves an existing point annotation for the active point tool.
    pub on_point_annotation_moved: extern "C" fn(
        viewport: ViewportSnapshotFFI,
        annotation_id: RString,
        x_level0: f64,
        y_level0: f64,
    ),

    /// Returns viewport filter descriptors this plugin provides.
    /// May return an empty vec if the plugin has no viewport filters.
    pub get_viewport_filters: extern "C" fn() -> RVec<ViewportFilterFFI>,

    /// Apply a CPU viewport filter in-place.
    /// `filter_id` identifies which filter to apply.
    /// `rgba_data` is a mutable pointer to width*height*4 bytes of RGBA8 data.
    /// Returns `true` if the filter was applied successfully.
    pub apply_filter_cpu: extern "C" fn(
        filter_id: RString,
        rgba_data: *mut u8,
        len: u32,
        width: u32,
        height: u32,
    ) -> bool,

    /// Apply a GPU viewport filter using raw Vulkan handles.
    /// `filter_id` identifies which filter to apply.
    /// `ctx` is a pointer to a `GpuFilterContextFFI` struct with the Vulkan
    /// device, image, and sync fence. The plugin must signal the fence when done.
    /// Returns `true` if the filter was applied successfully.
    pub apply_filter_gpu:
        extern "C" fn(filter_id: RString, ctx: *const GpuFilterContextFFI) -> bool,

    /// Enable or disable a viewport filter.
    pub set_filter_enabled: extern "C" fn(filter_id: RString, enabled: bool),
}

/// FFI-safe GPU filter context passed from the host to the plugin.
///
/// Contains raw Vulkan handles for zero-copy GPU memory access. The plugin
/// creates its own Vulkan pipeline on the shared device and submits commands
/// that operate on `vk_image`. The plugin **must** signal `vk_done_fence`
/// when its GPU work is complete.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct GpuFilterContextFFI {
    /// `VkDevice` handle (as `u64`).
    pub vk_device: u64,
    /// `VkPhysicalDevice` handle (as `u64`).
    pub vk_physical_device: u64,
    /// `VkQueue` handle the plugin may submit to.
    pub vk_queue: u64,
    /// Queue family index for `vk_queue`.
    pub queue_family_index: u32,
    /// `VkImage` handle for the RGBA8 frame.
    pub vk_image: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `VkFormat` value (e.g. `VK_FORMAT_R8G8B8A8_UNORM` = 37).
    pub vk_format: u32,
    /// `VkFence` handle — the plugin **must** signal this when its GPU
    /// commands have finished executing on `vk_image`.
    pub vk_done_fence: u64,
}

/// FFI-safe viewport filter descriptor.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ViewportFilterFFI {
    pub filter_id: RString,
    pub name: RString,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub enabled: bool,
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
