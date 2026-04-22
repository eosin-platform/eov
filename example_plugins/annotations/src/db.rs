use common::file_id::compute_fingerprint;
use rusqlite::{Connection, params};
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::{
    Annotation, AnnotationSet, PointAnnotation, PolygonAnnotation, PolygonVertex, annotation_label,
};

fn annotations_db_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("EOV_ANNOTATIONS_DB") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create annotations db directory '{}': {err}",
                    parent.display()
                )
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

pub(crate) fn open_database() -> Result<Connection, String> {
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

pub(crate) fn fingerprint_for_file(path: &Path) -> Result<[u8; 32], String> {
    compute_fingerprint(path).map_err(|err| {
        format!(
            "failed to compute WSI fingerprint for '{}': {err}",
            path.display()
        )
    })
}

pub(crate) fn load_annotation_sets(
    connection: &Connection,
    fingerprint: &[u8; 32],
) -> Result<Vec<AnnotationSet>, String> {
    let mut sets_stmt = connection
        .prepare(
            "SELECT id, name, notes, color, created_at, updated_at FROM annotation_sets WHERE fingerprint = ?1 ORDER BY lower(name) ASC, created_at DESC",
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

    let mut polygon_stmt = connection
        .prepare(
            r#"
            SELECT a.id, a.created_at, a.updated_at, pv.vertex_index, pv.x_level0, pv.y_level0
            FROM annotations a
            INNER JOIN annotation_polygon_vertices pv ON pv.annotation_id = a.id
            WHERE a.annotation_set_id = ?1 AND a.type = 'polygon'
            ORDER BY a.created_at DESC, a.id DESC, pv.vertex_index ASC
            "#,
        )
        .map_err(|err| format!("failed to prepare polygon annotation query: {err}"))?;

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

        let mut polygon_rows = polygon_stmt
            .query(params![&id])
            .map_err(|err| format!("failed to query polygon annotations for set '{id}': {err}"))?;
        let mut current_polygon: Option<PolygonAnnotation> = None;
        while let Some(row) = polygon_rows
            .next()
            .map_err(|err| format!("failed to read polygon annotation row: {err}"))?
        {
            let annotation_id = row
                .get::<_, String>(0)
                .map_err(|err| format!("failed to read polygon annotation id: {err}"))?;
            let created_at = row
                .get::<_, i64>(1)
                .map_err(|err| format!("failed to read polygon created_at: {err}"))?;
            let updated_at = row
                .get::<_, i64>(2)
                .map_err(|err| format!("failed to read polygon updated_at: {err}"))?;
            let vertex = PolygonVertex {
                x_level0: row
                    .get(4)
                    .map_err(|err| format!("failed to read polygon vertex x: {err}"))?,
                y_level0: row
                    .get(5)
                    .map_err(|err| format!("failed to read polygon vertex y: {err}"))?,
            };

            match current_polygon.as_mut() {
                Some(polygon) if polygon.id == annotation_id => polygon.vertices.push(vertex),
                Some(polygon) => {
                    annotations.push(Annotation::Polygon(polygon.clone()));
                    current_polygon = Some(PolygonAnnotation {
                        id: annotation_id,
                        created_at,
                        updated_at,
                        vertices: vec![vertex],
                    });
                }
                None => {
                    current_polygon = Some(PolygonAnnotation {
                        id: annotation_id,
                        created_at,
                        updated_at,
                        vertices: vec![vertex],
                    });
                }
            }
        }
        if let Some(polygon) = current_polygon.take() {
            annotations.push(Annotation::Polygon(polygon));
        }
        annotations.sort_by_key(|annotation| std::cmp::Reverse(annotation_label(annotation)));

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
