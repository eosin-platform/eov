use crate::analysis::AnalyzedTile;
use rusqlite::{Connection, OptionalExtension, params};
use sqlite_vec::sqlite3_vec_init;
use std::fs;
use std::path::PathBuf;
use std::sync::Once;
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;

static SQLITE_VEC_INIT: Once = Once::new();
const CACHE_WRITE_QUEUE_CAPACITY: usize = 4096;

#[derive(Clone, Debug)]
pub struct LatentCacheKey {
    pub model_sha256: String,
    pub model_path: String,
    pub file_sha256: String,
    pub file_path: String,
    pub level: u32,
    pub tile_size: u32,
    pub stride: u32,
    pub embedding_dim: usize,
}

pub struct LatentCacheWriter {
    connection: Connection,
    key: LatentCacheKey,
    cache_id: Option<i64>,
}

pub struct AsyncLatentCacheWriter {
    sender: SyncSender<CacheWriteRequest>,
    handle: Option<thread::JoinHandle<Result<(), String>>>,
}

enum CacheWriteRequest {
    Tile {
        tile: AnalyzedTile,
        embedding: Option<Vec<f32>>,
    },
    Shutdown,
}

pub fn load_tiles_for_positions(
    key: &LatentCacheKey,
    positions: &[(u64, u64)],
) -> Result<Vec<AnalyzedTile>, String> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    let connection = open_database()?;
    let Some(cache_id) = find_cache_id(&connection, key)? else {
        return Ok(Vec::new());
    };

    let mut statement = connection
        .prepare(
            r#"
            SELECT
                x_level0,
                y_level0,
                mean_absolute_error,
                max_error
            FROM latent_cache_tile
            WHERE cache_id = ?1 AND x_level0 = ?2 AND y_level0 = ?3
            "#,
        )
        .map_err(|err| format!("failed to prepare latent cache tile lookup: {err}"))?;

    let mut tiles = Vec::with_capacity(positions.len());
    for (x_level0, y_level0) in positions {
        let tile = statement
            .query_row(
                params![cache_id, *x_level0 as i64, *y_level0 as i64],
                |row| {
                    Ok(AnalyzedTile {
                        x: row.get::<_, i64>(0)? as u64,
                        y: row.get::<_, i64>(1)? as u64,
                        width: key.stride,
                        height: key.stride,
                        sample_width: key.tile_size,
                        sample_height: key.tile_size,
                        reconstruction_rgb: Vec::new(),
                        difference_rgb: Vec::new(),
                        error_map_luma: Vec::new(),
                        mean_absolute_error: row.get(2)?,
                        max_error: row.get::<_, i64>(3)? as u8,
                    })
                },
            )
            .optional()
            .map_err(|err| format!("failed to load latent cache tile: {err}"))?;
        if let Some(tile) = tile {
            tiles.push(tile);
        }
    }
    Ok(tiles)
}

impl LatentCacheWriter {
    pub fn open(key: LatentCacheKey) -> Result<Self, String> {
        Ok(Self {
            connection: open_database()?,
            key,
            cache_id: None,
        })
    }

    pub fn store_tile(
        &mut self,
        tile: &AnalyzedTile,
        embedding: Option<&[f32]>,
    ) -> Result<(), String> {
        let cache_id = self.ensure_cache_id()?;
        let sql_with_embedding = r#"
            INSERT INTO latent_cache_tile (
                cache_id,
                x_level0,
                y_level0,
                embedding,
                mean_absolute_error,
                max_error,
                updated_at
            ) VALUES (
                ?1, ?2, ?3, vec_f32(?4), ?5, ?6, unixepoch()
            )
            ON CONFLICT(cache_id, x_level0, y_level0) DO UPDATE SET
                embedding = excluded.embedding,
                mean_absolute_error = excluded.mean_absolute_error,
                max_error = excluded.max_error,
                updated_at = unixepoch()
        "#;
        let sql_without_embedding = r#"
            INSERT INTO latent_cache_tile (
                cache_id,
                x_level0,
                y_level0,
                embedding,
                mean_absolute_error,
                max_error,
                updated_at
            ) VALUES (
                ?1, ?2, ?3, NULL, ?4, ?5, unixepoch()
            )
            ON CONFLICT(cache_id, x_level0, y_level0) DO UPDATE SET
                embedding = excluded.embedding,
                mean_absolute_error = excluded.mean_absolute_error,
                max_error = excluded.max_error,
                updated_at = unixepoch()
        "#;

        if let Some(embedding) = embedding {
            let embedding_blob = f32_blob(embedding);
            self.connection
                .execute(
                    sql_with_embedding,
                    params![
                        cache_id,
                        tile.x as i64,
                        tile.y as i64,
                        embedding_blob,
                        tile.mean_absolute_error,
                        tile.max_error as i64,
                    ],
                )
                .map_err(|err| {
                    format!("failed to store latent cache tile with embedding: {err}")
                })?;
        } else {
            self.connection
                .execute(
                    sql_without_embedding,
                    params![
                        cache_id,
                        tile.x as i64,
                        tile.y as i64,
                        tile.mean_absolute_error,
                        tile.max_error as i64,
                    ],
                )
                .map_err(|err| format!("failed to store latent cache tile: {err}"))?;
        }

        Ok(())
    }

    fn ensure_cache_id(&mut self) -> Result<i64, String> {
        if let Some(cache_id) = self.cache_id {
            return Ok(cache_id);
        }
        if let Some(cache_id) = find_cache_id(&self.connection, &self.key)? {
            self.cache_id = Some(cache_id);
            return Ok(cache_id);
        }

        self.connection
            .execute(
                r#"
                INSERT INTO latent_cache (
                    model_sha256,
                    model_path,
                    file_sha256,
                    file_path,
                    level,
                    tile_size,
                    stride,
                    embedding_dim,
                    width_level0,
                    height_level0,
                    sample_width,
                    sample_height,
                    updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, unixepoch())
                "#,
                params![
                    &self.key.model_sha256,
                    &self.key.model_path,
                    &self.key.file_sha256,
                    &self.key.file_path,
                    self.key.level as i64,
                    self.key.tile_size as i64,
                    self.key.stride as i64,
                    self.key.embedding_dim as i64,
                    self.key.stride as i64,
                    self.key.stride as i64,
                    self.key.tile_size as i64,
                    self.key.tile_size as i64,
                ],
            )
            .map_err(|err| format!("failed to insert latent cache row: {err}"))?;

        let cache_id = find_cache_id(&self.connection, &self.key)?
            .ok_or_else(|| "latent cache row was not found after insert".to_string())?;
        self.cache_id = Some(cache_id);
        Ok(cache_id)
    }
}

impl AsyncLatentCacheWriter {
    pub fn open(key: LatentCacheKey) -> Result<Self, String> {
        let mut writer = LatentCacheWriter::open(key)?;
        let (sender, receiver) = mpsc::sync_channel(CACHE_WRITE_QUEUE_CAPACITY);
        let handle = thread::spawn(move || -> Result<(), String> {
            while let Ok(request) = receiver.recv() {
                match request {
                    CacheWriteRequest::Tile { tile, embedding } => {
                        writer.store_tile(&tile, embedding.as_deref())?;
                    }
                    CacheWriteRequest::Shutdown => return Ok(()),
                }
            }
            Ok(())
        });

        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    pub fn enqueue_tile(
        &self,
        tile: AnalyzedTile,
        embedding: Option<Vec<f32>>,
    ) -> Result<(), String> {
        self.sender
            .try_send(CacheWriteRequest::Tile { tile, embedding })
            .map_err(|error| match error {
                TrySendError::Full(_) => format!(
                    "persistent latent cache queue is full (capacity {})",
                    CACHE_WRITE_QUEUE_CAPACITY
                ),
                TrySendError::Disconnected(_) => {
                    "persistent latent cache writer thread disconnected".to_string()
                }
            })
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<(), String> {
        let _ = self.sender.send(CacheWriteRequest::Shutdown);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| "latent cache writer thread panicked".to_string())??;
        }
        Ok(())
    }
}

impl Drop for AsyncLatentCacheWriter {
    fn drop(&mut self) {
        let _ = self.finish_inner();
    }
}

fn latent_cache_db_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("EOVAE_LATENT_CACHE_DB") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create latent cache directory '{}': {err}",
                    parent.display()
                )
            })?;
        }
        return Ok(path);
    }

    let config_dir = dirs::config_dir()
        .ok_or_else(|| "could not determine config directory for latent cache".to_string())?
        .join("eov");
    fs::create_dir_all(&config_dir).map_err(|err| {
        format!(
            "failed to create latent cache config directory '{}': {err}",
            config_dir.display()
        )
    })?;
    Ok(config_dir.join("eovae-latent-cache.db"))
}

fn open_database() -> Result<Connection, String> {
    ensure_sqlite_vec_registered();
    let path = latent_cache_db_path()?;
    let connection = Connection::open(&path)
        .map_err(|err| format!("failed to open latent cache db '{}': {err}", path.display()))?;
    connection
        .execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS latent_cache (
                id INTEGER PRIMARY KEY,
                model_sha256 TEXT NOT NULL,
                model_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                file_path TEXT NOT NULL,
                level INTEGER NOT NULL,
                tile_size INTEGER NOT NULL,
                stride INTEGER NOT NULL,
                embedding_dim INTEGER NOT NULL,
                width_level0 INTEGER NOT NULL,
                height_level0 INTEGER NOT NULL,
                sample_width INTEGER NOT NULL,
                sample_height INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                UNIQUE (model_sha256, file_sha256, level, tile_size, stride, embedding_dim)
            );

            CREATE TABLE IF NOT EXISTS latent_cache_tile (
                id INTEGER PRIMARY KEY,
                cache_id INTEGER NOT NULL
                    REFERENCES latent_cache(id)
                    ON DELETE CASCADE,
                x_level0 INTEGER NOT NULL,
                y_level0 INTEGER NOT NULL,
                embedding BLOB,
                mean_absolute_error REAL NOT NULL,
                max_error INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                UNIQUE (cache_id, x_level0, y_level0)
            );

            CREATE INDEX IF NOT EXISTS idx_latent_cache_tile_cache_id
            ON latent_cache_tile(cache_id);
            "#,
        )
        .map_err(|err| format!("failed to initialize latent cache schema: {err}"))?;
    Ok(connection)
}

fn ensure_sqlite_vec_registered() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut i8,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32,
        >(sqlite3_vec_init as *const ())));
    });
}

fn find_cache_id(connection: &Connection, key: &LatentCacheKey) -> Result<Option<i64>, String> {
    connection
        .query_row(
            r#"
            SELECT id
            FROM latent_cache
            WHERE model_sha256 = ?1
              AND file_sha256 = ?2
              AND level = ?3
              AND tile_size = ?4
              AND stride = ?5
              AND embedding_dim = ?6
            "#,
            params![
                &key.model_sha256,
                &key.file_sha256,
                key.level as i64,
                key.tile_size as i64,
                key.stride as i64,
                key.embedding_dim as i64,
            ],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("failed to look up latent cache row: {err}"))
}

fn f32_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}
