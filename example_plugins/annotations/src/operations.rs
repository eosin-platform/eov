use common::file_id::hex_digest;
use plugin_api::ffi::{ConfirmationDialogRequestFFI, HostToolModeFFI, ViewportSnapshotFFI};
use rusqlite::params;
use std::fs;
use std::path::Path;
use uuid::Uuid;

use crate::db::{fingerprint_for_file, load_annotation_sets, open_database};
use crate::model::{
    Annotation, AnnotationSet, ExportAnnotation, ExportAnnotationSet, ExportFile,
    ExportPolygonVertex, LoadedFileAnnotations, PointAnnotation, PolygonAnnotation, PolygonVertex,
    choose_annotation_set_color, now_unix_secs, sort_annotation_sets, unique_untitled_set_name,
};
use crate::state::{
    PluginState, active_file_from_snapshot, active_file_key, host_api, host_snapshot, plugin_state,
};

fn ensure_loaded_for_file(
    state: &mut PluginState,
    file_path: &str,
    filename: &str,
) -> Result<(), String> {
    if state.files.contains_key(file_path) {
        return Ok(());
    }

    let path = Path::new(file_path);
    let fingerprint = fingerprint_for_file(path)?;
    let connection = open_database()?;
    let annotation_sets = load_annotation_sets(&connection, &fingerprint)?;
    state.files.insert(
        file_path.to_string(),
        LoadedFileAnnotations {
            file_path: file_path.to_string(),
            filename: filename.to_string(),
            fingerprint,
            annotation_sets,
        },
    );
    Ok(())
}

pub(crate) fn sync_active_file() -> Result<(), String> {
    let snapshot = host_snapshot()?;
    let mut state = plugin_state().lock().unwrap();
    let Some(active_file) = active_file_from_snapshot(&snapshot) else {
        state.active_file_path = None;
        state.active_filename = None;
        return Ok(());
    };

    let file_path = active_file.path.to_string();
    let filename = active_file.filename.to_string();
    ensure_loaded_for_file(&mut state, &file_path, &filename)?;
    state.active_file_path = Some(file_path);
    state.active_filename = Some(filename);
    Ok(())
}

pub(crate) fn ensure_loaded_for_viewport(viewport: &ViewportSnapshotFFI) -> Result<(), String> {
    let file_path = viewport.file_path.to_string();
    if file_path.is_empty() {
        return Ok(());
    }
    let filename = viewport.filename.to_string();
    ensure_loaded_for_file(&mut plugin_state().lock().unwrap(), &file_path, &filename)
}

pub(crate) fn refresh_sidebar_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.refresh_sidebar)(host_api.context).into_result();
    }
}

pub(crate) fn request_render_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.request_render)(host_api.context).into_result();
    }
}

fn ensure_selected_set_for_active_file(state: &mut PluginState) -> Result<Option<String>, String> {
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(None);
    };
    let Some(loaded) = state.files.get(&active_file_path).cloned() else {
        return Ok(None);
    };

    if let Some(selected_id) = state.selected_set_by_file.get(&active_file_path)
        && loaded
            .annotation_sets
            .iter()
            .any(|set| &set.id == selected_id)
    {
        return Ok(Some(selected_id.clone()));
    }

    if let Some(existing) = loaded
        .annotation_sets
        .iter()
        .find(|set| set.name == "Untitled")
    {
        state
            .selected_set_by_file
            .insert(active_file_path, existing.id.clone());
        return Ok(Some(existing.id.clone()));
    }

    let connection = open_database()?;
    let id = Uuid::new_v4().to_string();
    let timestamp = now_unix_secs();
    let color_hex = choose_annotation_set_color(&loaded.annotation_sets);
    connection
		.execute(
			"INSERT INTO annotation_sets (id, fingerprint, name, notes, color, created_at, updated_at) VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6)",
			params![&id, loaded.fingerprint.as_slice(), "Untitled", &color_hex, timestamp, timestamp],
		)
		.map_err(|err| format!("failed to create untitled annotation set: {err}"))?;

    let loaded_entry = state
        .files
        .get_mut(&active_file_path)
        .expect("loaded file missing");
    loaded_entry.annotation_sets.insert(
        0,
        AnnotationSet {
            id: id.clone(),
            name: "Untitled".to_string(),
            notes: None,
            color_hex,
            created_at: timestamp,
            updated_at: timestamp,
            annotations: Vec::new(),
        },
    );
    state
        .selected_set_by_file
        .insert(active_file_path, id.clone());
    Ok(Some(id))
}

pub(crate) fn create_annotation_set_for_active_file() -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let Some(loaded) = state.files.get(&active_file_path).cloned() else {
        return Ok(());
    };

    let name = unique_untitled_set_name(&loaded.annotation_sets);
    let id = Uuid::new_v4().to_string();
    let timestamp = now_unix_secs();
    let color_hex = choose_annotation_set_color(&loaded.annotation_sets);
    let connection = open_database()?;
    connection
		.execute(
			"INSERT INTO annotation_sets (id, fingerprint, name, notes, color, created_at, updated_at) VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6)",
			params![&id, loaded.fingerprint.as_slice(), &name, &color_hex, timestamp, timestamp],
		)
		.map_err(|err| format!("failed to create annotation set '{name}': {err}"))?;

    let loaded_entry = state
        .files
        .get_mut(&active_file_path)
        .ok_or_else(|| format!("active file '{}' is not loaded", active_file_path))?;
    loaded_entry.annotation_sets.push(AnnotationSet {
        id: id.clone(),
        name,
        notes: None,
        color_hex,
        created_at: timestamp,
        updated_at: timestamp,
        annotations: Vec::new(),
    });
    sort_annotation_sets(&mut loaded_entry.annotation_sets);
    state
        .selected_set_by_file
        .insert(active_file_path.clone(), id.clone());
    state.editing_set_by_file.insert(active_file_path, id);
    Ok(())
}

pub(crate) fn set_annotation_set_visibility_for_active_file(
    set_id: &str,
    visible: bool,
) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let hidden_sets = state
        .hidden_sets_by_file
        .entry(active_file_path)
        .or_default();
    if visible {
        hidden_sets.remove(set_id);
    } else {
        hidden_sets.insert(set_id.to_string());
    }
    Ok(())
}

pub(crate) fn set_annotation_set_color_for_active_file(
    set_id: &str,
    color_hex: &str,
) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    state.editing_set_by_file.remove(&active_file_path);

    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
        .execute(
            "UPDATE annotation_sets SET color = ?2, updated_at = ?3 WHERE id = ?1",
            params![set_id, color_hex, timestamp],
        )
        .map_err(|err| format!("failed to update annotation set color for '{set_id}': {err}"))?;

    if let Some(loaded_entry) = state.files.get_mut(&active_file_path)
        && let Some(set) = loaded_entry
            .annotation_sets
            .iter_mut()
            .find(|set| set.id == set_id)
    {
        set.color_hex = color_hex.to_string();
        set.updated_at = timestamp;
    }

    Ok(())
}

pub(crate) fn rename_annotation_set_for_active_file(
    set_id: &str,
    new_name: &str,
) -> Result<(), String> {
    sync_active_file()?;

    let trimmed_name = new_name.trim();
    if trimmed_name.is_empty() {
        return Ok(());
    }

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };

    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
        .execute(
            "UPDATE annotation_sets SET name = ?2, updated_at = ?3 WHERE id = ?1",
            params![set_id, trimmed_name, timestamp],
        )
        .map_err(|err| format!("failed to rename annotation set '{set_id}': {err}"))?;

    let loaded_entry = match state.files.get_mut(&active_file_path) {
        Some(entry) => entry,
        None => return Ok(()),
    };
    if let Some(set) = loaded_entry
        .annotation_sets
        .iter_mut()
        .find(|set| set.id == set_id)
    {
        set.name = trimmed_name.to_string();
        set.updated_at = timestamp;
        sort_annotation_sets(&mut loaded_entry.annotation_sets);
    }
    Ok(())
}

pub(crate) fn delete_annotation_set_for_active_file(set_id: &str) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let connection = open_database()?;
    connection
        .execute("DELETE FROM annotation_sets WHERE id = ?1", params![set_id])
        .map_err(|err| format!("failed to delete annotation set '{set_id}': {err}"))?;

    let next_selected = {
        let loaded_entry = match state.files.get_mut(&active_file_path) {
            Some(entry) => entry,
            None => return Ok(()),
        };
        loaded_entry.annotation_sets.retain(|set| set.id != set_id);
        loaded_entry
            .annotation_sets
            .first()
            .map(|set| set.id.clone())
    };

    if let Some(collapsed) = state.collapsed_sets_by_file.get_mut(&active_file_path) {
        collapsed.remove(set_id);
    }
    if let Some(hidden) = state.hidden_sets_by_file.get_mut(&active_file_path) {
        hidden.remove(set_id);
    }
    if let Some(editing) = state.editing_set_by_file.get(&active_file_path)
        && editing == set_id
    {
        state.editing_set_by_file.remove(&active_file_path);
    }
    match next_selected {
        Some(next_id) => {
            state.selected_set_by_file.insert(active_file_path, next_id);
        }
        None => {
            state.selected_set_by_file.remove(&active_file_path);
        }
    }
    Ok(())
}

pub(crate) fn delete_annotation_for_active_file(annotation_id: &str) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let timestamp = now_unix_secs();
    let connection = open_database()?;

    let updated_set_id = state.files.get(&active_file_path).and_then(|loaded| {
        loaded.annotation_sets.iter().find_map(|set| {
            set.annotations
                .iter()
                .any(|annotation| match annotation {
                    Annotation::Point(point) => point.id == annotation_id,
                    Annotation::Polygon(polygon) => polygon.id == annotation_id,
                })
                .then(|| set.id.clone())
        })
    });

    connection
        .execute(
            "DELETE FROM annotations WHERE id = ?1",
            params![annotation_id],
        )
        .map_err(|err| format!("failed to delete annotation '{annotation_id}': {err}"))?;

    if let Some(set_id) = updated_set_id.as_deref() {
        connection
            .execute(
                "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
                params![set_id, timestamp],
            )
            .map_err(|err| {
                format!("failed to update annotation set timestamp for '{set_id}': {err}")
            })?;
    }

    if let Some(loaded_entry) = state.files.get_mut(&active_file_path) {
        for set in &mut loaded_entry.annotation_sets {
            let before = set.annotations.len();
            set.annotations.retain(|annotation| match annotation {
                Annotation::Point(point) => point.id != annotation_id,
                Annotation::Polygon(polygon) => polygon.id != annotation_id,
            });
            if set.annotations.len() != before {
                set.updated_at = timestamp;
                break;
            }
        }
    }

    Ok(())
}

pub(crate) fn request_delete_annotation_set(set_id: &str, set_name: &str) -> Result<(), String> {
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    let request = ConfirmationDialogRequestFFI {
		title: "Delete Annotation Set".into(),
		message: format!(
			"Are you sure you want to delete annotation set '{set_name}'? This action cannot be undone."
		)
		.into(),
		confirm_label: "Delete Permanently".into(),
		cancel_label: "Cancel".into(),
		confirm_callback: abi_stable::std_types::ROption::RSome("delete-set-confirmed".into()),
		confirm_args_json: abi_stable::std_types::ROption::RSome(
			serde_json::to_string(&vec![set_id])
				.unwrap_or_else(|_| "[]".to_string())
				.into(),
		),
		cancel_callback: abi_stable::std_types::ROption::RNone,
		cancel_args_json: abi_stable::std_types::ROption::RNone,
	};
    (host_api.show_confirmation_dialog)(host_api.context, request)
        .into_result()
        .map_err(|err| format!("failed to show delete annotation set confirmation: {err}"))
}

pub(crate) fn move_point_annotation(
    viewport: &ViewportSnapshotFFI,
    annotation_id: &str,
    x_level0: f64,
    y_level0: f64,
) -> Result<(), String> {
    ensure_loaded_for_viewport(viewport)?;
    let file_path = viewport.file_path.to_string();
    let mut state = plugin_state().lock().unwrap();
    state.active_file_path = Some(file_path.clone());
    state.active_filename = Some(viewport.filename.to_string());

    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
        .execute(
            "UPDATE annotation_points SET x_level0 = ?2, y_level0 = ?3 WHERE annotation_id = ?1",
            params![annotation_id, x_level0, y_level0],
        )
        .map_err(|err| format!("failed to move point annotation '{annotation_id}': {err}"))?;

    let mut updated_set_id = None;
    if let Some(loaded) = state.files.get_mut(&file_path) {
        for set in &mut loaded.annotation_sets {
            for annotation in &mut set.annotations {
                if let Annotation::Point(point) = annotation
                    && point.id == annotation_id
                {
                    point.x_level0 = x_level0;
                    point.y_level0 = y_level0;
                    point.updated_at = timestamp;
                    set.updated_at = timestamp;
                    updated_set_id = Some(set.id.clone());
                    break;
                }
            }
            if updated_set_id.is_some() {
                break;
            }
        }
    }

    connection
        .execute(
            "UPDATE annotations SET updated_at = ?2 WHERE id = ?1",
            params![annotation_id, timestamp],
        )
        .map_err(|err| {
            format!("failed to update annotation timestamp for '{annotation_id}': {err}")
        })?;

    if let Some(set_id) = updated_set_id {
        connection
            .execute(
                "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
                params![set_id, timestamp],
            )
            .map_err(|err| {
                format!("failed to update annotation set timestamp after move: {err}")
            })?;
    }

    Ok(())
}

pub(crate) fn move_polygon_annotation(
    viewport: &ViewportSnapshotFFI,
    annotation_id: &str,
    vertices: &[plugin_api::ffi::ViewportOverlayVertexFFI],
) -> Result<(), String> {
    if vertices.len() < 3 {
        return Ok(());
    }

    ensure_loaded_for_viewport(viewport)?;
    let file_path = viewport.file_path.to_string();
    let mut state = plugin_state().lock().unwrap();
    state.active_file_path = Some(file_path.clone());
    state.active_filename = Some(viewport.filename.to_string());

    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
        .execute(
            "DELETE FROM annotation_polygon_vertices WHERE annotation_id = ?1",
            params![annotation_id],
        )
        .map_err(|err| format!("failed to clear polygon vertices for '{annotation_id}': {err}"))?;
    for (index, vertex) in vertices.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO annotation_polygon_vertices (annotation_id, vertex_index, x_level0, y_level0) VALUES (?1, ?2, ?3, ?4)",
                params![annotation_id, index as i64, vertex.x_level0, vertex.y_level0],
            )
            .map_err(|err| format!("failed to update polygon vertex {index} for '{annotation_id}': {err}"))?;
    }

    let mut updated_set_id = None;
    if let Some(loaded) = state.files.get_mut(&file_path) {
        for set in &mut loaded.annotation_sets {
            for annotation in &mut set.annotations {
                if let Annotation::Polygon(polygon) = annotation
                    && polygon.id == annotation_id
                {
                    polygon.vertices = vertices
                        .iter()
                        .map(|vertex| PolygonVertex {
                            x_level0: vertex.x_level0,
                            y_level0: vertex.y_level0,
                        })
                        .collect();
                    polygon.updated_at = timestamp;
                    set.updated_at = timestamp;
                    updated_set_id = Some(set.id.clone());
                    break;
                }
            }
            if updated_set_id.is_some() {
                break;
            }
        }
    }

    connection
        .execute(
            "UPDATE annotations SET updated_at = ?2 WHERE id = ?1",
            params![annotation_id, timestamp],
        )
        .map_err(|err| {
            format!("failed to update polygon annotation timestamp for '{annotation_id}': {err}")
        })?;

    if let Some(set_id) = updated_set_id {
        connection
            .execute(
                "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
                params![set_id, timestamp],
            )
            .map_err(|err| {
                format!("failed to update annotation set timestamp after polygon move: {err}")
            })?;
    }

    Ok(())
}

pub(crate) fn start_point_annotation_flow() -> Result<(), String> {
    sync_active_file()?;
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    (host_api.set_active_tool)(host_api.context, HostToolModeFFI::PointAnnotation)
        .into_result()
        .map_err(|err| format!("failed to activate point annotation tool: {err}"))?;
    refresh_sidebar_if_available();
    request_render_if_available();
    Ok(())
}

pub(crate) fn start_polygon_annotation_flow() -> Result<(), String> {
    sync_active_file()?;
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    (host_api.set_active_tool)(host_api.context, HostToolModeFFI::PolygonAnnotation)
        .into_result()
        .map_err(|err| format!("failed to activate polygon annotation tool: {err}"))?;
    refresh_sidebar_if_available();
    request_render_if_available();
    Ok(())
}

pub(crate) fn export_active_file_annotations() -> Result<(), String> {
    sync_active_file()?;
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };

    let export_payload = {
        let state = plugin_state().lock().unwrap();
        let Some(active_path) = active_file_key(&state) else {
            return Ok(());
        };
        let loaded = state
            .files
            .get(active_path)
            .ok_or_else(|| format!("active file '{}' is not loaded", active_path))?;
        ExportFile {
            file_path: loaded.file_path.clone(),
            fingerprint_hex: hex_digest(&loaded.fingerprint),
            annotation_sets: loaded
                .annotation_sets
                .iter()
                .map(|set| ExportAnnotationSet {
                    id: set.id.clone(),
                    name: set.name.clone(),
                    notes: set.notes.clone(),
                    color_hex: set.color_hex.clone(),
                    created_at: set.created_at,
                    updated_at: set.updated_at,
                    annotations: set
                        .annotations
                        .iter()
                        .map(|annotation| match annotation {
                            Annotation::Point(point) => ExportAnnotation::Point {
                                id: point.id.clone(),
                                created_at: point.created_at,
                                updated_at: point.updated_at,
                                x_level0: point.x_level0,
                                y_level0: point.y_level0,
                            },
                            Annotation::Polygon(polygon) => ExportAnnotation::Polygon {
                                id: polygon.id.clone(),
                                created_at: polygon.created_at,
                                updated_at: polygon.updated_at,
                                vertices: polygon
                                    .vertices
                                    .iter()
                                    .map(|vertex| ExportPolygonVertex {
                                        x_level0: vertex.x_level0,
                                        y_level0: vertex.y_level0,
                                    })
                                    .collect(),
                            },
                        })
                        .collect(),
                })
                .collect(),
        }
    };

    let default_file_name = {
        let state = plugin_state().lock().unwrap();
        state
            .active_file_path
            .as_deref()
            .and_then(|path| state.files.get(path))
            .map(|loaded| format!("{}_annotations.json", loaded.filename))
            .unwrap_or_else(|| {
                let filename = export_payload
                    .file_path
                    .rsplit_once(std::path::MAIN_SEPARATOR)
                    .map(|(_, name)| name.to_string())
                    .unwrap_or_else(|| export_payload.file_path.clone());
                format!("{}_annotations.json", filename)
            })
    };

    let save_path = match (host_api.save_file_dialog)(
        host_api.context,
        default_file_name.into(),
        "JSON".into(),
        "json".into(),
    )
    .into_result()
    {
        Ok(path) => path.to_string(),
        Err(_) => return Ok(()),
    };

    let json = serde_json::to_string_pretty(&export_payload)
        .map_err(|err| format!("failed to serialize annotation export: {err}"))?;
    fs::write(&save_path, json)
        .map_err(|err| format!("failed to write annotation export '{}': {err}", save_path))?;
    Ok(())
}

pub(crate) fn persist_point_annotation(
    viewport: &ViewportSnapshotFFI,
    x_level0: f64,
    y_level0: f64,
) -> Result<(), String> {
    ensure_loaded_for_viewport(viewport)?;
    let file_path = viewport.file_path.to_string();
    let mut state = plugin_state().lock().unwrap();
    state.active_file_path = Some(file_path.clone());
    state.active_filename = Some(viewport.filename.to_string());
    let Some(annotation_set_id) = ensure_selected_set_for_active_file(&mut state)? else {
        return Ok(());
    };

    let annotation_id = Uuid::new_v4().to_string();
    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
		.execute(
			"INSERT INTO annotations (id, annotation_set_id, type, created_at, updated_at) VALUES (?1, ?2, 'point', ?3, ?4)",
			params![&annotation_id, &annotation_set_id, timestamp, timestamp],
		)
		.map_err(|err| format!("failed to insert point annotation: {err}"))?;
    connection
        .execute(
            "INSERT INTO annotation_points (annotation_id, x_level0, y_level0) VALUES (?1, ?2, ?3)",
            params![&annotation_id, x_level0, y_level0],
        )
        .map_err(|err| format!("failed to insert point annotation geometry: {err}"))?;
    connection
        .execute(
            "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
            params![&annotation_set_id, timestamp],
        )
        .map_err(|err| format!("failed to update annotation set timestamp: {err}"))?;

    let loaded = state
        .files
        .get_mut(&file_path)
        .ok_or_else(|| format!("file '{}' is not loaded in plugin state", file_path))?;
    let set = loaded
        .annotation_sets
        .iter_mut()
        .find(|set| set.id == annotation_set_id)
        .ok_or_else(|| format!("annotation set '{}' is not loaded", annotation_set_id))?;
    set.updated_at = timestamp;
    set.annotations.insert(
        0,
        Annotation::Point(PointAnnotation {
            id: annotation_id,
            created_at: timestamp,
            updated_at: timestamp,
            x_level0,
            y_level0,
        }),
    );
    Ok(())
}

pub(crate) fn persist_polygon_annotation(
    viewport: &ViewportSnapshotFFI,
    vertices: &[plugin_api::ffi::ViewportOverlayVertexFFI],
) -> Result<(), String> {
    if vertices.len() < 3 {
        return Ok(());
    }

    ensure_loaded_for_viewport(viewport)?;
    let file_path = viewport.file_path.to_string();
    let mut state = plugin_state().lock().unwrap();
    state.active_file_path = Some(file_path.clone());
    state.active_filename = Some(viewport.filename.to_string());
    let Some(annotation_set_id) = ensure_selected_set_for_active_file(&mut state)? else {
        return Ok(());
    };

    let annotation_id = Uuid::new_v4().to_string();
    let timestamp = now_unix_secs();
    let connection = open_database()?;
    connection
        .execute(
            "INSERT INTO annotations (id, annotation_set_id, type, created_at, updated_at) VALUES (?1, ?2, 'polygon', ?3, ?4)",
            params![&annotation_id, &annotation_set_id, timestamp, timestamp],
        )
        .map_err(|err| format!("failed to insert polygon annotation: {err}"))?;
    connection
        .execute(
            "INSERT INTO annotation_polygons (annotation_id) VALUES (?1)",
            params![&annotation_id],
        )
        .map_err(|err| format!("failed to insert polygon annotation shell: {err}"))?;
    for (index, vertex) in vertices.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO annotation_polygon_vertices (annotation_id, vertex_index, x_level0, y_level0) VALUES (?1, ?2, ?3, ?4)",
                params![&annotation_id, index as i64, vertex.x_level0, vertex.y_level0],
            )
            .map_err(|err| format!("failed to insert polygon vertex {index}: {err}"))?;
    }
    connection
        .execute(
            "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
            params![&annotation_set_id, timestamp],
        )
        .map_err(|err| format!("failed to update annotation set timestamp: {err}"))?;

    let loaded = state
        .files
        .get_mut(&file_path)
        .ok_or_else(|| format!("file '{}' is not loaded in plugin state", file_path))?;
    let set = loaded
        .annotation_sets
        .iter_mut()
        .find(|set| set.id == annotation_set_id)
        .ok_or_else(|| format!("annotation set '{}' is not loaded", annotation_set_id))?;
    set.updated_at = timestamp;
    set.annotations.insert(
        0,
        Annotation::Polygon(PolygonAnnotation {
            id: annotation_id,
            created_at: timestamp,
            updated_at: timestamp,
            vertices: vertices
                .iter()
                .map(|vertex| PolygonVertex {
                    x_level0: vertex.x_level0,
                    y_level0: vertex.y_level0,
                })
                .collect(),
        }),
    );
    Ok(())
}
