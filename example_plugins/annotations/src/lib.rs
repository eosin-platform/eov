use abi_stable::std_types::{ROption, RString, RVec};
use common::file_id::{compute_fingerprint, hex_digest};
use plugin_api::ffi::{
    ActionResponseFFI, ConfirmationDialogRequestFFI, HostApiVTable, HostLogLevelFFI,
    HostSnapshotFFI, HostToolModeFFI, HudToolbarButtonFFI, OpenFileInfoFFI, PluginVTable,
    ToolbarButtonFFI, UiPropertyFFI, ViewportContextMenuItemFFI, ViewportFilterFFI,
    ViewportOverlayPointFFI, ViewportSnapshotFFI,
};
use rusqlite::{Connection, params};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const SIDEBAR_WIDTH_PX: u32 = 300;
const SIDEBAR_UI_PATH: &str = "ui/annotations-sidebar.slint";
const SIDEBAR_COMPONENT: &str = "AnnotationsSidebar";

const ACTION_TOGGLE_SIDEBAR: &str = "toggle_annotations";
const ACTION_CREATE_POINT: &str = "create_point_annotation";
const VIEWPORT_MENU_CREATE_POINT: &str = "create_point";

const SIDEBAR_ICON_SVG: &str = include_str!("../../../app/ui/icons/annotations.svg");
const POINT_ICON_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.5" fill="currentColor"/></svg>"#;
const SET_COLOR_PALETTE: [&str; 16] = [
    "#FF355E",
    "#FF7A00",
    "#E7FF00",
    "#39FF14",
    "#00F5D4",
    "#00BBF9",
    "#4D96FF",
    "#6C63FF",
    "#9D4EDD",
    "#FF4FD8",
    "#F15BB5",
    "#FF8A5B",
    "#43AA8B",
    "#577590",
    "#FFD166",
    "#FFFFFF",
];

static HOST_API: Mutex<Option<HostApiVTable>> = Mutex::new(None);
static PLUGIN_STATE: OnceLock<Mutex<PluginState>> = OnceLock::new();

#[derive(Default)]
struct PluginState {
    files: HashMap<String, LoadedFileAnnotations>,
    active_file_path: Option<String>,
    active_filename: Option<String>,
    selected_set_by_file: HashMap<String, String>,
    editing_set_by_file: HashMap<String, String>,
    collapsed_sets_by_file: HashMap<String, HashSet<String>>,
    hidden_sets_by_file: HashMap<String, HashSet<String>>,
}

#[derive(Clone)]
struct LoadedFileAnnotations {
    file_path: String,
    filename: String,
    fingerprint: [u8; 32],
    annotation_sets: Vec<AnnotationSet>,
}

#[derive(Clone)]
struct AnnotationSet {
    id: String,
    name: String,
    notes: Option<String>,
    color_hex: String,
    created_at: i64,
    updated_at: i64,
    annotations: Vec<Annotation>,
}

#[derive(Clone)]
enum Annotation {
    Point(PointAnnotation),
}

#[derive(Clone)]
struct PointAnnotation {
    id: String,
    created_at: i64,
    updated_at: i64,
    x_level0: f64,
    y_level0: f64,
}

#[derive(Serialize)]
struct SidebarTreeRow {
    row_id: String,
    parent_set_id: String,
    label: String,
    annotation_count: i32,
    indent: i32,
    is_set: bool,
    is_collapsed: bool,
    is_selected: bool,
    visible: bool,
    color_r: i32,
    color_g: i32,
    color_b: i32,
}

#[derive(Serialize)]
struct ExportFile {
    file_path: String,
    fingerprint_hex: String,
    annotation_sets: Vec<ExportAnnotationSet>,
}

#[derive(Serialize)]
struct ExportAnnotationSet {
    id: String,
    name: String,
    notes: Option<String>,
    color_hex: String,
    created_at: i64,
    updated_at: i64,
    annotations: Vec<ExportAnnotation>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ExportAnnotation {
    Point {
        id: String,
        created_at: i64,
        updated_at: i64,
        x_level0: f64,
        y_level0: f64,
    },
}

fn plugin_state() -> &'static Mutex<PluginState> {
    PLUGIN_STATE.get_or_init(|| Mutex::new(PluginState::default()))
}

fn host_api() -> Option<HostApiVTable> {
    *HOST_API.lock().unwrap()
}

fn log_message(level: HostLogLevelFFI, message: impl Into<String>) {
    if let Some(host_api) = host_api() {
        (host_api.log_message)(host_api.context, level, RString::from(message.into()));
    }
}

fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn annotations_db_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("EOV_ANNOTATIONS_DB") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create annotations db directory '{}': {err}", parent.display())
            })?;
        }
        return Ok(path);
    }

    let config_dir = dirs::config_dir()
        .ok_or_else(|| "could not determine config directory for annotations db".to_string())?
        .join("eov");
    fs::create_dir_all(&config_dir).map_err(|err| {
        format!(
            "failed to create annotations config directory '{}': {err}",
            config_dir.display()
        )
    })?;
    Ok(config_dir.join("annotations.db"))
}

fn hex_color_to_rgb(color_hex: &str) -> (u8, u8, u8) {
    let hex = color_hex.trim_start_matches('#');
    if hex.len() != 6 {
        return (0xF2, 0xF4, 0xF8);
    }
    let red = u8::from_str_radix(&hex[0..2], 16).ok();
    let green = u8::from_str_radix(&hex[2..4], 16).ok();
    let blue = u8::from_str_radix(&hex[4..6], 16).ok();
    match (red, green, blue) {
        (Some(red), Some(green), Some(blue)) => (red, green, blue),
        _ => (0xF2, 0xF4, 0xF8),
    }
}

fn palette_seed() -> usize {
    let uuid = Uuid::new_v4();
    let mut seed = 0usize;
    for byte in uuid.as_bytes().iter().take(std::mem::size_of::<usize>()) {
        seed = (seed << 8) | *byte as usize;
    }
    seed
}

fn choose_annotation_set_color(annotation_sets: &[AnnotationSet]) -> String {
    let mut usage_counts: HashMap<&'static str, usize> =
        SET_COLOR_PALETTE.iter().copied().map(|color| (color, 0)).collect();
    let used_colors: HashSet<&str> = annotation_sets.iter().map(|set| set.color_hex.as_str()).collect();
    for set in annotation_sets {
        if let Some(count) = usage_counts.get_mut(set.color_hex.as_str()) {
            *count += 1;
        }
    }

    let unused_colors: Vec<&str> = SET_COLOR_PALETTE
        .iter()
        .copied()
        .filter(|color| !used_colors.contains(color))
        .collect();
    let seed = palette_seed();
    if !unused_colors.is_empty() {
        return unused_colors[seed % unused_colors.len()].to_string();
    }

    let min_usage = usage_counts.values().copied().min().unwrap_or(0);
    let least_used: Vec<&str> = SET_COLOR_PALETTE
        .iter()
        .copied()
        .filter(|color| usage_counts.get(color).copied().unwrap_or(0) == min_usage)
        .collect();
    least_used[seed % least_used.len()].to_string()
}

fn open_database() -> Result<Connection, String> {
    let path = annotations_db_path()?;
    let connection = Connection::open(&path)
        .map_err(|err| format!("failed to open annotations db '{}': {err}", path.display()))?;
    connection
        .execute_batch(
            r#"
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS annotation_sets (
                id TEXT PRIMARY KEY,
                fingerprint BLOB NOT NULL CHECK(length(fingerprint) = 32),
                name TEXT NOT NULL CHECK(length(name) <= 255),
                notes TEXT,
                color TEXT NOT NULL CHECK(length(color) = 7),
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                annotation_set_id TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('point', 'ellipse', 'polygon', 'bitmask')),
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (annotation_set_id) REFERENCES annotation_sets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_points (
                annotation_id TEXT PRIMARY KEY,
                x_level0 REAL NOT NULL,
                y_level0 REAL NOT NULL,
                FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_ellipses (
                annotation_id TEXT PRIMARY KEY,
                center_x_level0 REAL NOT NULL,
                center_y_level0 REAL NOT NULL,
                radius_x_level0 REAL NOT NULL CHECK(radius_x_level0 > 0),
                radius_y_level0 REAL NOT NULL CHECK(radius_y_level0 > 0),
                rotation_radians REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_polygons (
                annotation_id TEXT PRIMARY KEY,
                FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_polygon_vertices (
                annotation_id TEXT NOT NULL,
                vertex_index INTEGER NOT NULL,
                x_level0 REAL NOT NULL,
                y_level0 REAL NOT NULL,
                PRIMARY KEY (annotation_id, vertex_index),
                FOREIGN KEY (annotation_id) REFERENCES annotation_polygons(annotation_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_bitmasks (
                annotation_id TEXT PRIMARY KEY,
                FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotation_bitmask_strokes (
                id TEXT PRIMARY KEY,
                annotation_id TEXT NOT NULL,
                stroke_index INTEGER NOT NULL,
                brush_radius_level0 REAL NOT NULL CHECK(brush_radius_level0 > 0),
                is_eraser INTEGER NOT NULL DEFAULT 0 CHECK(is_eraser IN (0, 1)),
                created_at INTEGER NOT NULL,
                FOREIGN KEY (annotation_id) REFERENCES annotation_bitmasks(annotation_id) ON DELETE CASCADE,
                UNIQUE (annotation_id, stroke_index)
            );

            CREATE TABLE IF NOT EXISTS annotation_bitmask_stroke_points (
                stroke_id TEXT NOT NULL,
                point_index INTEGER NOT NULL,
                x_level0 REAL NOT NULL,
                y_level0 REAL NOT NULL,
                PRIMARY KEY (stroke_id, point_index),
                FOREIGN KEY (stroke_id) REFERENCES annotation_bitmask_strokes(id) ON DELETE CASCADE
            );
            "#,
        )
        .map_err(|err| format!("failed to initialize annotations db schema: {err}"))?;
    Ok(connection)
}

fn fingerprint_for_file(path: &Path) -> Result<[u8; 32], String> {
    compute_fingerprint(path).map_err(|err| {
        format!(
            "failed to compute WSI fingerprint for '{}': {err}",
            path.display()
        )
    })
}

fn load_annotation_sets(
    connection: &Connection,
    fingerprint: &[u8; 32],
) -> Result<Vec<AnnotationSet>, String> {
    let mut sets_stmt = connection
        .prepare(
            "SELECT id, name, notes, color, created_at, updated_at FROM annotation_sets WHERE fingerprint = ?1 ORDER BY lower(name) DESC, created_at DESC",
        )
        .map_err(|err| format!("failed to prepare annotation set query: {err}"))?;

    let set_rows = sets_stmt
        .query_map(params![fingerprint.as_slice()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(|err| format!("failed to query annotation sets: {err}"))?;

    let mut annotation_stmt = connection
        .prepare(
            r#"
            SELECT a.id, a.created_at, a.updated_at, p.x_level0, p.y_level0
            FROM annotations a
            INNER JOIN annotation_points p ON p.annotation_id = a.id
            WHERE a.annotation_set_id = ?1 AND a.type = 'point'
            ORDER BY a.created_at DESC, a.id DESC
            "#,
        )
        .map_err(|err| format!("failed to prepare point annotation query: {err}"))?;

    let mut sets = Vec::new();
    for set_row in set_rows {
        let (id, name, notes, color_hex, created_at, updated_at) =
            set_row.map_err(|err| format!("failed to read annotation set row: {err}"))?;
        let annotation_rows = annotation_stmt
            .query_map(params![&id], |row| {
                Ok(Annotation::Point(PointAnnotation {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                    x_level0: row.get(3)?,
                    y_level0: row.get(4)?,
                }))
            })
            .map_err(|err| format!("failed to query point annotations for set '{id}': {err}"))?;
        let mut annotations = Vec::new();
        for annotation in annotation_rows {
            annotations.push(
                annotation.map_err(|err| format!("failed to read point annotation row: {err}"))?,
            );
        }
        annotations.sort_by(|left, right| annotation_label(right).cmp(&annotation_label(left)));

        sets.push(AnnotationSet {
            id,
            name,
            notes,
            color_hex,
            created_at,
            updated_at,
            annotations,
        });
    }

    Ok(sets)
}

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

fn host_snapshot() -> Result<HostSnapshotFFI, String> {
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    Ok((host_api.get_snapshot)(host_api.context))
}

fn active_file_from_snapshot(snapshot: &HostSnapshotFFI) -> Option<OpenFileInfoFFI> {
    match &snapshot.active_file {
        ROption::RSome(file) => Some(file.clone()),
        ROption::RNone => None,
    }
}

fn active_viewport_from_snapshot(snapshot: &HostSnapshotFFI) -> Option<ViewportSnapshotFFI> {
    match &snapshot.active_viewport {
        ROption::RSome(viewport) => Some(viewport.clone()),
        ROption::RNone => None,
    }
}

fn sync_active_file() -> Result<(), String> {
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

fn ensure_loaded_for_viewport(viewport: &ViewportSnapshotFFI) -> Result<(), String> {
    let file_path = viewport.file_path.to_string();
    if file_path.is_empty() {
        return Ok(());
    }
    let filename = viewport.filename.to_string();
    ensure_loaded_for_file(&mut plugin_state().lock().unwrap(), &file_path, &filename)
}

fn annotation_label(annotation: &Annotation) -> String {
    match annotation {
        Annotation::Point(_) => "Point".to_string(),
    }
}

fn sort_annotation_sets(annotation_sets: &mut [AnnotationSet]) {
    annotation_sets.sort_by(|left, right| {
        right
            .name
            .to_ascii_lowercase()
            .cmp(&left.name.to_ascii_lowercase())
            .then_with(|| right.created_at.cmp(&left.created_at))
            .then_with(|| right.id.cmp(&left.id))
    });
}

fn unique_untitled_set_name(annotation_sets: &[AnnotationSet]) -> String {
    let existing = annotation_sets
        .iter()
        .map(|set| set.name.as_str())
        .collect::<HashSet<_>>();
    if !existing.contains("Untitled") {
        return "Untitled".to_string();
    }

    let mut suffix = 1;
    loop {
        let candidate = format!("Untitled {suffix}");
        if !existing.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

fn refresh_sidebar_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.refresh_sidebar)(host_api.context).into_result();
    }
}

fn request_render_if_available() {
    if let Some(host_api) = host_api() {
        let _ = (host_api.request_render)(host_api.context).into_result();
    }
}

fn active_file_key(state: &PluginState) -> Option<&str> {
    state.active_file_path.as_deref()
}

fn ensure_selected_set_for_active_file(state: &mut PluginState) -> Result<Option<String>, String> {
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(None);
    };
    let Some(loaded) = state.files.get(&active_file_path).cloned() else {
        return Ok(None);
    };

    if let Some(selected_id) = state.selected_set_by_file.get(&active_file_path)
        && loaded.annotation_sets.iter().any(|set| &set.id == selected_id)
    {
        return Ok(Some(selected_id.clone()));
    }

    if let Some(existing) = loaded.annotation_sets.iter().find(|set| set.name == "Untitled") {
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

    let loaded_entry = state.files.get_mut(&active_file_path).expect("loaded file missing");
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

fn create_annotation_set_for_active_file() -> Result<(), String> {
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
    state.selected_set_by_file.insert(active_file_path.clone(), id.clone());
    state.editing_set_by_file.insert(active_file_path, id);
    Ok(())
}

fn set_annotation_set_visibility_for_active_file(set_id: &str, visible: bool) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let hidden_sets = state.hidden_sets_by_file.entry(active_file_path).or_default();
    if visible {
        hidden_sets.remove(set_id);
    } else {
        hidden_sets.insert(set_id.to_string());
    }
    Ok(())
}

fn set_annotation_set_color_for_active_file(set_id: &str, color_hex: &str) -> Result<(), String> {
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
        && let Some(set) = loaded_entry.annotation_sets.iter_mut().find(|set| set.id == set_id)
    {
        set.color_hex = color_hex.to_string();
        set.updated_at = timestamp;
    }

    Ok(())
}

fn annotation_set_by_point_id<'a>(
    annotation_sets: &'a [AnnotationSet],
    annotation_id: &str,
) -> Option<&'a AnnotationSet> {
    annotation_sets.iter().find(|set| {
        set.annotations.iter().any(|annotation| {
            matches!(annotation, Annotation::Point(point) if point.id == annotation_id)
        })
    })
}

fn rename_annotation_set_for_active_file(set_id: &str, new_name: &str) -> Result<(), String> {
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
    if let Some(set) = loaded_entry.annotation_sets.iter_mut().find(|set| set.id == set_id) {
        set.name = trimmed_name.to_string();
        set.updated_at = timestamp;
        sort_annotation_sets(&mut loaded_entry.annotation_sets);
    }
    Ok(())
}

fn delete_annotation_set_for_active_file(set_id: &str) -> Result<(), String> {
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
        loaded_entry.annotation_sets.first().map(|set| set.id.clone())
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

fn delete_annotation_for_active_file(annotation_id: &str) -> Result<(), String> {
    sync_active_file()?;

    let mut state = plugin_state().lock().unwrap();
    let Some(active_file_path) = state.active_file_path.clone() else {
        return Ok(());
    };
    let timestamp = now_unix_secs();
    let connection = open_database()?;

    let updated_set_id = state.files.get(&active_file_path).and_then(|loaded| {
        loaded.annotation_sets.iter().find_map(|set| {
            set.annotations.iter().any(|annotation| match annotation {
                Annotation::Point(point) => point.id == annotation_id,
            })
            .then(|| set.id.clone())
        })
    });

    connection
        .execute("DELETE FROM annotations WHERE id = ?1", params![annotation_id])
        .map_err(|err| format!("failed to delete annotation '{annotation_id}': {err}"))?;

    if let Some(set_id) = updated_set_id.as_deref() {
        connection
            .execute(
                "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
                params![set_id, timestamp],
            )
            .map_err(|err| format!("failed to update annotation set timestamp for '{set_id}': {err}"))?;
    }

    if let Some(loaded_entry) = state.files.get_mut(&active_file_path) {
        for set in &mut loaded_entry.annotation_sets {
            let before = set.annotations.len();
            set.annotations.retain(|annotation| match annotation {
                Annotation::Point(point) => point.id != annotation_id,
            });
            if set.annotations.len() != before {
                set.updated_at = timestamp;
                break;
            }
        }
    }

    Ok(())
}

fn request_delete_annotation_set(set_id: &str, set_name: &str) -> Result<(), String> {
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
        confirm_callback: ROption::RSome("delete-set-confirmed".into()),
        confirm_args_json: ROption::RSome(
            serde_json::to_string(&vec![set_id]).unwrap_or_else(|_| "[]".to_string()).into(),
        ),
        cancel_callback: ROption::RNone,
        cancel_args_json: ROption::RNone,
    };
    (host_api.show_confirmation_dialog)(host_api.context, request)
        .into_result()
        .map_err(|err| format!("failed to show delete annotation set confirmation: {err}"))
}

fn move_point_annotation(
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
                let Annotation::Point(point) = annotation;
                if point.id == annotation_id {
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
        .map_err(|err| format!("failed to update annotation timestamp for '{annotation_id}': {err}"))?;

    if let Some(set_id) = updated_set_id {
        connection
            .execute(
                "UPDATE annotation_sets SET updated_at = ?2 WHERE id = ?1",
                params![set_id, timestamp],
            )
            .map_err(|err| format!("failed to update annotation set timestamp after move: {err}"))?;
    }

    Ok(())
}

fn start_point_annotation_flow() -> Result<(), String> {
    sync_active_file()?;
    let Some(host_api) = host_api() else {
        return Err("host API is not available".to_string());
    };
    (host_api.set_active_tool)(host_api.context, HostToolModeFFI::PointAnnotation)
        .into_result()
        .map_err(|err| format!("failed to activate point annotation tool: {err}"))?;
    let _ = (host_api.refresh_sidebar)(host_api.context).into_result();
    let _ = (host_api.request_render)(host_api.context).into_result();
    Ok(())
}

fn export_active_file_annotations() -> Result<(), String> {
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
                        })
                        .collect(),
                })
                .collect(),
        }
    };

    let default_file_name = format!(
        "{}_annotations.json",
        export_payload
            .annotation_sets
            .first()
            .and(Some(()))
            .and_then(|_| {
                let state = plugin_state().lock().unwrap();
                state
                    .active_file_path
                    .as_deref()
                    .and_then(|path| state.files.get(path))
                    .map(|loaded| loaded.filename.clone())
            })
            .unwrap_or_else(|| {
                export_payload
                    .file_path
                    .rsplit_once(std::path::MAIN_SEPARATOR)
                    .map(|(_, name)| name.to_string())
                    .unwrap_or_else(|| export_payload.file_path.clone())
            })
    );
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

fn on_sidebar_callback(callback_name: &str, args_json: &str) {
    let args = parse_callback_args(args_json);

    let result = match callback_name {
        "export-clicked" => export_active_file_annotations(),
        "create-set-clicked" => {
            create_annotation_set_for_active_file().map(|_| {
                refresh_sidebar_if_available();
            })
        }
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
            sync_active_file().and_then(|_| {
                let Some(serde_json::Value::String(row_id)) = args.first() else {
                    return Ok(());
                };
                let mut state = plugin_state().lock().unwrap();
                let Some(active_path) = active_file_key(&state).map(str::to_string) else {
                    return Ok(());
                };
                let Some(loaded) = state.files.get(&active_path) else {
                    return Ok(());
                };

                if loaded.annotation_sets.iter().any(|set| &set.id == row_id) {
                    state.selected_set_by_file.insert(active_path, row_id.clone());
                    refresh_sidebar_if_available();
                    return Ok(());
                }

                let annotation_target = loaded.annotation_sets.iter().find_map(|set| {
                    set.annotations.iter().find_map(|annotation| match annotation {
                        Annotation::Point(point) if &point.id == row_id => {
                            Some((set.id.clone(), point.x_level0, point.y_level0))
                        }
                        _ => None,
                    })
                });

                if let Some((set_id, x_level0, y_level0)) = annotation_target {
                    state.selected_set_by_file.insert(active_path, set_id);
                    drop(state);
                    refresh_sidebar_if_available();

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
                    .map_err(|err| format!("failed to frame annotation '{row_id}': {err}"))?;
                }
                Ok(())
            })
        }
        "toggle-set" => {
            sync_active_file().and_then(|_| {
                let Some(serde_json::Value::String(set_id)) = args.first() else {
                    return Ok(());
                };
                let mut state = plugin_state().lock().unwrap();
                let Some(active_path) = active_file_key(&state).map(str::to_string) else {
                    return Ok(());
                };
                let collapsed = state.collapsed_sets_by_file.entry(active_path).or_default();
                if !collapsed.insert(set_id.clone()) {
                    collapsed.remove(set_id);
                }
                refresh_sidebar_if_available();
                Ok(())
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

fn persist_point_annotation(viewport: &ViewportSnapshotFFI, x_level0: f64, y_level0: f64) -> Result<(), String> {
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

extern "C" fn set_host_api_ffi(host_api: HostApiVTable) {
    *HOST_API.lock().unwrap() = Some(host_api);
}

extern "C" fn get_toolbar_buttons_ffi() -> RVec<ToolbarButtonFFI> {
    RVec::from(vec![
        ToolbarButtonFFI {
            button_id: RString::from(ACTION_TOGGLE_SIDEBAR),
            tooltip: RString::from("Toggle annotations sidebar"),
            icon_svg: RString::from(SIDEBAR_ICON_SVG),
            action_id: RString::from(ACTION_TOGGLE_SIDEBAR),
            tool_mode: ROption::RNone,
            hotkey: ROption::RNone,
        },
        ToolbarButtonFFI {
            button_id: RString::from(ACTION_CREATE_POINT),
            tooltip: RString::from("Create point annotation"),
            icon_svg: RString::from(POINT_ICON_SVG),
            action_id: RString::from(ACTION_CREATE_POINT),
            tool_mode: ROption::RSome(HostToolModeFFI::PointAnnotation),
            hotkey: ROption::RSome("1".into()),
        },
    ])
}

extern "C" fn get_hud_toolbar_buttons_ffi() -> RVec<HudToolbarButtonFFI> {
    RVec::new()
}

extern "C" fn on_action_ffi(action_id: RString) -> ActionResponseFFI {
    let result = match action_id.as_str() {
        ACTION_TOGGLE_SIDEBAR => {
            sync_active_file().and_then(|_| {
                let Some(host_api) = host_api() else {
                    return Err("host API is not available".to_string());
                };
                (host_api.show_sidebar)(
                    host_api.context,
                    RString::from(ACTION_TOGGLE_SIDEBAR),
                    SIDEBAR_WIDTH_PX,
                    RString::from(SIDEBAR_UI_PATH),
                    RString::from(SIDEBAR_COMPONENT),
                )
                .into_result()
                .map_err(|err| format!("failed to toggle annotations sidebar: {err}"))
            })
        }
        ACTION_CREATE_POINT => start_point_annotation_flow(),
        _ => Ok(()),
    };

    if let Err(err) = result {
        log_message(HostLogLevelFFI::Error, err);
    }

    ActionResponseFFI { open_window: false }
}

extern "C" fn on_hud_action_ffi(
    _action_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    ActionResponseFFI { open_window: false }
}

extern "C" fn on_ui_callback_ffi(callback_name: RString, args_json: RString) {
    on_sidebar_callback(callback_name.as_str(), args_json.as_str());
}

extern "C" fn get_sidebar_properties_ffi() -> RVec<UiPropertyFFI> {
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
            json_value: serde_json::to_string(&rows).unwrap_or_else(|_| "[]".to_string()).into(),
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
            json_value: serde_json::to_string(
                &state
                    .active_file_path
                    .as_deref()
                    .and_then(|path| {
                        let selected_id = state.selected_set_by_file.get(path)?;
                        state
                            .files
                            .get(path)
                            .and_then(|loaded| {
                                loaded
                                    .annotation_sets
                                    .iter()
                                    .find(|set| &set.id == selected_id)
                                    .map(|set| set.name.clone())
                            })
                    })
                    .unwrap_or_default(),
            )
            .unwrap_or_else(|_| "\"\"".to_string())
            .into(),
        },
    ])
}

extern "C" fn get_viewport_context_menu_items_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportContextMenuItemFFI> {
    if viewport.file_id < 0 || viewport.file_path.is_empty() {
        return RVec::new();
    }

    RVec::from(vec![ViewportContextMenuItemFFI {
        item_id: VIEWPORT_MENU_CREATE_POINT.into(),
        label: "Create Point".into(),
        icon: "point".into(),
        enabled: true,
    }])
}

extern "C" fn on_viewport_context_menu_action_ffi(
    item_id: RString,
    _viewport: ViewportSnapshotFFI,
) -> ActionResponseFFI {
    if item_id.as_str() == VIEWPORT_MENU_CREATE_POINT
        && let Err(err) = start_point_annotation_flow()
    {
        log_message(HostLogLevelFFI::Error, err);
    }

    ActionResponseFFI { open_window: false }
}

extern "C" fn get_viewport_overlay_points_ffi(
    viewport: ViewportSnapshotFFI,
) -> RVec<ViewportOverlayPointFFI> {
    if viewport.file_id < 0 || viewport.file_path.is_empty() {
        return RVec::new();
    }
    if let Err(err) = ensure_loaded_for_viewport(&viewport) {
        log_message(HostLogLevelFFI::Error, err);
        return RVec::new();
    }

    let state = plugin_state().lock().unwrap();
    let Some(loaded) = state.files.get(viewport.file_path.as_str()) else {
        return RVec::new();
    };
    let hidden_sets = state.hidden_sets_by_file.get(viewport.file_path.as_str());

    let points = loaded
        .annotation_sets
        .iter()
        .filter(|set| !hidden_sets.is_some_and(|hidden| hidden.contains(&set.id)))
        .flat_map(|set| set.annotations.iter())
        .filter_map(|annotation| match annotation {
            Annotation::Point(point) => {
                let set = annotation_set_by_point_id(&loaded.annotation_sets, &point.id)?;
                let (ring_red, ring_green, ring_blue) = hex_color_to_rgb(&set.color_hex);
                Some(ViewportOverlayPointFFI {
                    annotation_id: point.id.clone().into(),
                    x_level0: point.x_level0,
                    y_level0: point.y_level0,
                    diameter_px: 12.0,
                    ring_red,
                    ring_green,
                    ring_blue,
                })
            }
        })
        .collect::<Vec<_>>();
    RVec::from(points)
}

extern "C" fn on_point_annotation_placed_ffi(
    viewport: ViewportSnapshotFFI,
    x_level0: f64,
    y_level0: f64,
) {
    match persist_point_annotation(&viewport, x_level0, y_level0) {
        Ok(()) => {
            refresh_sidebar_if_available();
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

extern "C" fn on_point_annotation_moved_ffi(
    viewport: ViewportSnapshotFFI,
    annotation_id: RString,
    x_level0: f64,
    y_level0: f64,
) {
    match move_point_annotation(&viewport, annotation_id.as_str(), x_level0, y_level0) {
        Ok(()) => {
            request_render_if_available();
        }
        Err(err) => log_message(HostLogLevelFFI::Error, err),
    }
}

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
        get_sidebar_properties: get_sidebar_properties_ffi,
        get_viewport_context_menu_items: get_viewport_context_menu_items_ffi,
        on_viewport_context_menu_action: on_viewport_context_menu_action_ffi,
        get_viewport_overlay_points: get_viewport_overlay_points_ffi,
        on_point_annotation_placed: on_point_annotation_placed_ffi,
        on_point_annotation_moved: on_point_annotation_moved_ffi,
        get_viewport_filters: get_viewport_filters_ffi,
        apply_filter_cpu: apply_filter_cpu_ffi,
        apply_filter_gpu: apply_filter_gpu_ffi,
        set_filter_enabled: set_filter_enabled_ffi,
    }
}