use abi_stable::std_types::RVec;
use plugin_api::ffi::{HostLogLevelFFI, UiPropertyFFI};

use crate::model::{Annotation, SidebarTreeRow, annotation_label, hex_color_to_rgb};
use crate::operations::{
    create_annotation_set_for_active_file, delete_annotation_for_active_file,
    delete_annotation_set_for_active_file, export_active_file_annotations,
    refresh_sidebar_if_available, rename_annotation_set_for_active_file,
    request_delete_annotation_set, request_render_if_available,
    set_annotation_set_color_for_active_file, set_annotation_set_visibility_for_active_file,
    sync_active_file,
};
use crate::state::{
    PluginState, active_file_key, active_viewport_from_snapshot, host_api, host_snapshot,
    log_message, plugin_state,
};

fn sidebar_rows(state: &PluginState) -> Vec<SidebarTreeRow> {
    let Some(active_path) = active_file_key(state) else {
        return Vec::new();
    };
    let Some(loaded) = state.files.get(active_path) else {
        return Vec::new();
    };
    let selected_set_id = state.selected_set_by_file.get(active_path);
    let collapsed_sets = state.collapsed_sets_by_file.get(active_path);
    let hidden_sets = state.hidden_sets_by_file.get(active_path);

    let mut rows = Vec::new();
    for set in &loaded.annotation_sets {
        let is_collapsed = collapsed_sets.is_some_and(|collapsed| collapsed.contains(&set.id));
        let is_visible = !hidden_sets.is_some_and(|hidden| hidden.contains(&set.id));
        let (color_r, color_g, color_b) = hex_color_to_rgb(&set.color_hex);
        rows.push(SidebarTreeRow {
            row_id: set.id.clone(),
            parent_set_id: set.id.clone(),
            label: set.name.clone(),
            annotation_count: set.annotations.len() as i32,
            indent: 0,
            is_set: true,
            is_collapsed,
            is_selected: selected_set_id.is_some_and(|selected| selected == &set.id),
            visible: is_visible,
            color_r: color_r as i32,
            color_g: color_g as i32,
            color_b: color_b as i32,
        });

        if !is_collapsed {
            for annotation in &set.annotations {
                let annotation_id = match annotation {
                    Annotation::Point(point) => point.id.clone(),
                    Annotation::Polygon(polygon) => polygon.id.clone(),
                };
                rows.push(SidebarTreeRow {
                    row_id: annotation_id,
                    parent_set_id: set.id.clone(),
                    label: annotation_label(annotation),
                    annotation_count: 0,
                    indent: 1,
                    is_set: false,
                    is_collapsed: false,
                    is_selected: false,
                    visible: is_visible,
                    color_r: color_r as i32,
                    color_g: color_g as i32,
                    color_b: color_b as i32,
                });
            }
        }
    }
    rows
}

fn parse_callback_args(args_json: &str) -> Vec<serde_json::Value> {
    match serde_json::from_str::<serde_json::Value>(args_json) {
        Ok(serde_json::Value::Array(values)) => values,
        _ => Vec::new(),
    }
}

fn selected_set_name(state: &PluginState) -> String {
    state
        .active_file_path
        .as_deref()
        .and_then(|path| {
            let selected_id = state.selected_set_by_file.get(path)?;
            state.files.get(path).and_then(|loaded| {
                loaded
                    .annotation_sets
                    .iter()
                    .find(|set| &set.id == selected_id)
                    .map(|set| set.name.clone())
            })
        })
        .unwrap_or_default()
}

fn focus_sidebar_row(row_id: &str) -> Result<(), String> {
    sync_active_file()?;

    let annotation_target = {
        let mut state = plugin_state().lock().unwrap();
        let Some(active_path) = active_file_key(&state).map(str::to_string) else {
            return Ok(());
        };

        let is_set = state
            .files
            .get(&active_path)
            .is_some_and(|loaded| loaded.annotation_sets.iter().any(|set| set.id == row_id));
        if is_set {
            state
                .selected_set_by_file
                .insert(active_path, row_id.to_string());
            return Ok(());
        }

        let target = state.files.get(&active_path).and_then(|loaded| {
            loaded.annotation_sets.iter().find_map(|set| {
                set.annotations
                    .iter()
                    .find_map(|annotation| match annotation {
                        Annotation::Point(point) if point.id == row_id => {
                            Some((set.id.clone(), point.x_level0, point.y_level0))
                        }
                        Annotation::Polygon(polygon) if polygon.id == row_id => {
                            let min_x = polygon
                                .vertices
                                .iter()
                                .map(|vertex| vertex.x_level0)
                                .fold(f64::INFINITY, f64::min);
                            let min_y = polygon
                                .vertices
                                .iter()
                                .map(|vertex| vertex.y_level0)
                                .fold(f64::INFINITY, f64::min);
                            let max_x = polygon
                                .vertices
                                .iter()
                                .map(|vertex| vertex.x_level0)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let max_y = polygon
                                .vertices
                                .iter()
                                .map(|vertex| vertex.y_level0)
                                .fold(f64::NEG_INFINITY, f64::max);
                            Some((set.id.clone(), (min_x + max_x) * 0.5, (min_y + max_y) * 0.5))
                        }
                        _ => None,
                    })
            })
        });

        if let Some((set_id, _, _)) = target.as_ref() {
            state
                .selected_set_by_file
                .insert(active_path, set_id.clone());
        }
        target
    };

    let Some((_, x_level0, y_level0)) = annotation_target else {
        return Ok(());
    };

    let snapshot = host_snapshot()?;
    let Some(active_viewport) = active_viewport_from_snapshot(&snapshot) else {
        return Ok(());
    };
    let width = active_viewport.width.max(1.0);
    let height = active_viewport.height.max(1.0);
    let Some(host_api) = host_api() else {
        return Ok(());
    };
    (host_api.frame_active_rect)(
        host_api.context,
        x_level0 - width / 2.0,
        y_level0 - height / 2.0,
        width,
        height,
    )
    .into_result()
    .map_err(|err| format!("failed to frame annotation '{row_id}': {err}"))
}

fn toggle_set_for_active_file(set_id: &str) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_path) = active_file_key(&state).map(str::to_string) else {
        return Ok(());
    };
    let collapsed = state.collapsed_sets_by_file.entry(active_path).or_default();
    if !collapsed.insert(set_id.to_string()) {
        collapsed.remove(set_id);
    }
    Ok(())
}

pub(crate) fn on_sidebar_callback(callback_name: &str, args_json: &str) {
    let args = parse_callback_args(args_json);

    let result = match callback_name {
        "export-clicked" => export_active_file_annotations(),
        "create-set-clicked" => create_annotation_set_for_active_file().map(|_| {
            refresh_sidebar_if_available();
        }),
        "rename-set-committed" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            let Some(serde_json::Value::String(new_name)) = args.get(1) else {
                return;
            };
            rename_annotation_set_for_active_file(set_id, new_name).map(|_| {
                refresh_sidebar_if_available();
            })
        }
        "delete-set-confirmed" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            delete_annotation_set_for_active_file(set_id).map(|_| {
                refresh_sidebar_if_available();
                request_render_if_available();
            })
        }
        "request-delete-set" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            let Some(serde_json::Value::String(set_name)) = args.get(1) else {
                return;
            };
            request_delete_annotation_set(set_id, set_name)
        }
        "delete-annotation-clicked" => {
            let Some(serde_json::Value::String(annotation_id)) = args.first() else {
                return;
            };
            delete_annotation_for_active_file(annotation_id).map(|_| {
                refresh_sidebar_if_available();
                request_render_if_available();
            })
        }
        "source-selected" => Ok(()),
        "row-clicked" => {
            let Some(serde_json::Value::String(row_id)) = args.first() else {
                return;
            };
            focus_sidebar_row(row_id)
        }
        "toggle-set" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            toggle_set_for_active_file(set_id).map(|_| {
                refresh_sidebar_if_available();
            })
        }
        "toggle-set-visibility" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            let Some(serde_json::Value::Bool(visible)) = args.get(1) else {
                return;
            };
            set_annotation_set_visibility_for_active_file(set_id, *visible).map(|_| {
                refresh_sidebar_if_available();
                request_render_if_available();
            })
        }
        "set-set-color" => {
            let Some(serde_json::Value::String(set_id)) = args.first() else {
                return;
            };
            let Some(serde_json::Value::String(color_hex)) = args.get(1) else {
                return;
            };
            set_annotation_set_color_for_active_file(set_id, color_hex).map(|_| {
                refresh_sidebar_if_available();
                request_render_if_available();
            })
        }
        _ => Ok(()),
    };

    if let Err(err) = result {
        log_message(HostLogLevelFFI::Error, err);
    }
}

pub(crate) fn get_sidebar_properties() -> RVec<UiPropertyFFI> {
    if let Err(err) = sync_active_file() {
        log_message(HostLogLevelFFI::Error, err);
    }

    let state = plugin_state().lock().unwrap();
    let rows = sidebar_rows(&state);
    let editing_set_id = state
        .active_file_path
        .as_deref()
        .and_then(|path| state.editing_set_by_file.get(path).cloned())
        .unwrap_or_default();
    let empty_state = if state.active_file_path.is_none() {
        "Open a slide to view its annotation sets.".to_string()
    } else if rows.is_empty() {
        "No annotation sets for this slide yet.".to_string()
    } else {
        String::new()
    };

    RVec::from(vec![
        UiPropertyFFI {
            name: "source-options".into(),
            json_value: "[\"Local\"]".into(),
        },
        UiPropertyFFI {
            name: "source-index".into(),
            json_value: "0".into(),
        },
        UiPropertyFFI {
            name: "tree-items".into(),
            json_value: serde_json::to_string(&rows)
                .unwrap_or_else(|_| "[]".to_string())
                .into(),
        },
        UiPropertyFFI {
            name: "empty-state-text".into(),
            json_value: serde_json::to_string(&empty_state)
                .unwrap_or_else(|_| "\"\"".to_string())
                .into(),
        },
        UiPropertyFFI {
            name: "can-export".into(),
            json_value: (state.active_file_path.is_some()).to_string().into(),
        },
        UiPropertyFFI {
            name: "can-delete-set".into(),
            json_value: state
                .active_file_path
                .as_deref()
                .and_then(|path| state.selected_set_by_file.get(path))
                .is_some()
                .to_string()
                .into(),
        },
        UiPropertyFFI {
            name: "selected-set-id".into(),
            json_value: serde_json::to_string(
                &state
                    .active_file_path
                    .as_deref()
                    .and_then(|path| state.selected_set_by_file.get(path).cloned())
                    .unwrap_or_default(),
            )
            .unwrap_or_else(|_| "\"\"".to_string())
            .into(),
        },
        UiPropertyFFI {
            name: "editing-set-id".into(),
            json_value: serde_json::to_string(&editing_set_id)
                .unwrap_or_else(|_| "\"\"".to_string())
                .into(),
        },
        UiPropertyFFI {
            name: "selected-set-name".into(),
            json_value: serde_json::to_string(&selected_set_name(&state))
                .unwrap_or_else(|_| "\"\"".to_string())
                .into(),
        },
    ])
}
