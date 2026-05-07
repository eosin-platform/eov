use abi_stable::std_types::RVec;
use plugin_api::ffi::{HostLogLevelFFI, UiPropertyFFI};
use serde_json::json;

use crate::state;

fn parse_args(args_json: &str) -> Vec<serde_json::Value> {
    match serde_json::from_str::<serde_json::Value>(args_json) {
        Ok(serde_json::Value::Array(values)) => values,
        _ => Vec::new(),
    }
}

pub fn get_sidebar_properties() -> RVec<UiPropertyFFI> {
    state::prepare_window_runtime();
    let selected_device_index = state::selected_device_index();
    let profile_names = state::profile_names();
    RVec::from(vec![
        state::property("controller-enabled", json!(state::controller_enabled())),
        state::property("status-text", json!(state::current_status_text())),
        state::property("device-options", json!(state::device_labels())),
        state::property("selected-device-index", json!(selected_device_index)),
        state::property("profile-options", json!(profile_names)),
        state::property(
            "selected-profile-index",
            json!(state::profile_selected_index()),
        ),
        state::property(
            "profile-draft-name",
            json!(state::current_profile_draft_name()),
        ),
        state::property(
            "left-dead-zone",
            json!(state::numeric_value_text("left-dead-zone")),
        ),
        state::property(
            "right-dead-zone",
            json!(state::numeric_value_text("right-dead-zone")),
        ),
        state::property(
            "trigger-dead-zone",
            json!(state::numeric_value_text("trigger-dead-zone")),
        ),
        state::property(
            "pan-sensitivity",
            json!(state::numeric_value_text("pan-sensitivity")),
        ),
        state::property(
            "zoom-sensitivity",
            json!(state::numeric_value_text("zoom-sensitivity")),
        ),
        state::property("axis-action-options", json!(state::axis_action_options())),
        state::property(
            "button-action-options",
            json!(state::button_action_options()),
        ),
        state::property("mapping-rows", state::mapping_rows_json()),
    ])
}

pub fn on_sidebar_callback(callback_name: &str, args_json: &str) {
    state::prepare_window_runtime();
    let args = parse_args(args_json);
    match callback_name {
        "enabled-changed" => {
            if let Some(value) = args.first().and_then(|value| value.as_bool()) {
                state::set_controller_enabled(value);
            }
        }
        "device-selected" => {
            if let Some(label) = args.first().and_then(|value| value.as_str()) {
                state::set_selected_device_by_label(label);
            }
        }
        "profile-name-changed" => {
            if let Some(name) = args.first().and_then(|value| value.as_str()) {
                state::set_profile_draft_name(name.to_string());
            }
        }
        "save-profile-clicked" => {
            let name = state::current_profile_draft_name();
            if let Err(err) = state::save_profile(&name) {
                state::log_message(HostLogLevelFFI::Warn, err);
            }
        }
        "load-profile-selected" => {
            if let Some(name) = args.first().and_then(|value| value.as_str())
                && let Err(err) = state::load_profile(name)
            {
                state::log_message(HostLogLevelFFI::Warn, err);
            }
        }
        "delete-profile-clicked" => {
            let name = state::current_profile_draft_name();
            if let Err(err) = state::delete_profile(&name) {
                state::log_message(HostLogLevelFFI::Warn, err);
            }
        }
        "numeric-field-changed" => {
            if let (Some(field), Some(value)) = (
                args.first().and_then(|value| value.as_str()),
                args.get(1).and_then(|value| value.as_str()),
            ) {
                state::set_numeric_field(field, value);
            }
        }
        "axis-mapping-selected" => {
            if let (Some(control_key), Some(action_label)) = (
                args.first().and_then(|value| value.as_str()),
                args.get(1).and_then(|value| value.as_str()),
            ) {
                state::set_mapping_by_label(control_key, action_label, true);
            }
        }
        "button-mapping-selected" => {
            if let (Some(control_key), Some(action_label)) = (
                args.first().and_then(|value| value.as_str()),
                args.get(1).and_then(|value| value.as_str()),
            ) {
                state::set_mapping_by_label(control_key, action_label, false);
            }
        }
        _ => {}
    }

    state::refresh_window_ui_if_available();
}
