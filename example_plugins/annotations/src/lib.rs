use abi_stable::std_types::{RString, RVec};
use plugin_api::ffi::{
    ActionResponseFFI, HostApiVTable, HostLogLevelFFI, HudToolbarButtonFFI, PluginVTable,
    ToolbarButtonFFI, ViewportFilterFFI, ViewportSnapshotFFI,
};
use std::sync::Mutex;

const ICON_SVG: &str = include_str!("../../../app/ui/icons/annotations.svg");
const BUTTON_ID: &str = "toggle_annotations";
const SIDEBAR_WIDTH_PX: u32 = 250;
const SIDEBAR_UI_PATH: &str = "ui/annotations-sidebar.slint";
const SIDEBAR_COMPONENT: &str = "AnnotationsSidebar";

static HOST_API: Mutex<Option<HostApiVTable>> = Mutex::new(None);

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    *HOST_API.lock().unwrap() = Some(host_api);
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![ToolbarButtonFFI {
        button_id: RString::from(BUTTON_ID),
        tooltip: RString::from("Toggle annotations sidebar"),
        icon_svg: RString::from(ICON_SVG),
        action_id: RString::from(BUTTON_ID),
    }])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    if action_id.as_str() == BUTTON_ID && let Some(host_api) = *HOST_API.lock().unwrap() {
        if let Err(err) = (host_api.show_sidebar)(
            host_api.context,
            RString::from(BUTTON_ID),
            SIDEBAR_WIDTH_PX,
            RString::from(SIDEBAR_UI_PATH),
            RString::from(SIDEBAR_COMPONENT),
        )
        .into_result()
        {
            (host_api.log_message)(
                host_api.context,
                HostLogLevelFFI::Error,
                RString::from(format!("Failed to toggle annotations sidebar: {err}")),
            );
        }
    }

    ActionResponseFFI { open_window: false }
}

extern "C" fn on_hud_action_ffi(
    _action_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_ui_callback_ffi(_callback_name: RString) {}

extern "C" fn get_viewport_filters_ffi() -> RVec<ViewportFilterFFI> {
    RVec::new()
}

extern "C" fn apply_filter_cpu_ffi(
    _filter_id: RString,
    _rgba_data: *mut u8,
    _len: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

extern "C" fn apply_filter_gpu_ffi(
    _filter_id: RString,
    _ctx: *const plugin_api::ffi::GpuFilterContextFFI,
) -> bool {
    false
}

extern "C" fn set_filter_enabled_ffi(_filter_id: RString, _enabled: bool) {}

#[unsafe(no_mangle)]
pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable {
    PluginVTable {
        set_host_api: set_host_api_ffi,
        get_toolbar_buttons: get_toolbar_buttons_ffi,
        get_hud_toolbar_buttons: get_hud_toolbar_buttons_ffi,
        on_action: on_action_ffi,
        on_hud_action: on_hud_action_ffi,
        on_ui_callback: on_ui_callback_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}