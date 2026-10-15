//! The on-disk result cache (VAE-12).
//!
//! - Lives in the plugin's cache directory (never the config directory) and
//!   stores no file paths: slides are known by their identity or content
//!   fingerprint, runs by their [`crate::run::RunSpec`] key and manifest.
//! - Has a schema version (`PRAGMA user_version`) and ordered migrations; a
//!   database written by a newer eovae is left alone.
//! - One thread owns the connection. Writes are queued through a bounded
//!   channel (callers block instead of silently dropping results) and
//!   committed in batched transactions.
//! - Is size-limited: the least recently used runs are deleted when the
//!   database grows past its budget, and it can be cleared from the sidebar.
//! - Tile pixels are stored only when the user enables it (compressed);
//!   scores, background markers and embeddings always are.

use crate::results::{TilePayload, TileState};
use crate::tiling::TileCoord;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DB_FILE: &str = "results.sqlite";

/// Ordered migrations; `user_version` is the number applied.
const MIGRATIONS: &[&str] = &[
    // 1: initial schema.
    r#"
    CREATE TABLE slide (
        id INTEGER PRIMARY KEY,
        identity TEXT UNIQUE,
        fingerprint TEXT UNIQUE,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        last_used INTEGER NOT NULL
    );
    CREATE TABLE run (
        id INTEGER PRIMARY KEY,
        slide_id INTEGER NOT NULL REFERENCES slide(id) ON DELETE CASCADE,
        run_key TEXT NOT NULL,
        manifest TEXT NOT NULL,
        cols INTEGER NOT NULL,
        rows INTEGER NOT NULL,
        created INTEGER NOT NULL,
        last_used INTEGER NOT NULL,
        UNIQUE (slide_id, run_key)
    );
    CREATE INDEX run_last_used ON run(last_used);
    CREATE TABLE tile (
        run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
        col INTEGER NOT NULL,
        row INTEGER NOT NULL,
        state INTEGER NOT NULL,
        mae REAL,
        max_error REAL,
        embedding BLOB,
        payload BLOB,
        PRIMARY KEY (run_id, col, row)
    ) WITHOUT ROWID;
    "#,
];

/// How a slide is known.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlideKeys {
    /// The host's identity (`FileInfo::identity`), once it provides one.
    pub identity: Option<String>,
    /// eovae's fallback content fingerprint.
    pub fingerprint: Option<String>,
    pub width: u64,
    pub height: u64,
}

/// A tile to persist.
#[derive(Clone, Debug)]
pub struct DbTile {
    pub coord: TileCoord,
    pub state: TileState,
    pub embedding: Option<Vec<f32>>,
    /// Stored only when keeping pixels is enabled.
    pub payload: Option<Arc<TilePayload>>,
}

/// One run's writes.
#[derive(Clone, Debug)]
pub struct RunWrite {
    pub slide: SlideKeys,
    pub run_key: String,
    pub manifest: String,
    pub cols: u32,
    pub rows: u32,
    pub tiles: Vec<DbTile>,
}

/// What the cache reports to the sidebar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheStatus {
    pub enabled: bool,
    pub bytes: u64,
    pub budget: u64,
    pub runs: u64,
    pub error: Option<String>,
}

enum Request {
    Write(RunWrite),
    LoadRun {
        slide: SlideKeys,
        run_key: String,
        reply: Sender<Vec<(TileCoord, TileState)>>,
    },
    LoadPayloads {
        slide: SlideKeys,
        run_key: String,
        coords: Vec<TileCoord>,
        reply: Sender<Vec<(TileCoord, TilePayload)>>,
    },
    SetBudget(u64),
    Clear(Sender<Result<(), String>>),
    Flush(Sender<()>),
}

/// Handle to the cache thread.
pub struct ResultCache {
    tx: Mutex<Option<SyncSender<Request>>>,
    status: Arc<Mutex<CacheStatus>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn f32_blob(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Opens (creating or migrating) the database. Returns `Ok(None)` for a
/// database written by a newer schema.
fn open_connection(path: &Path) -> Result<Option<Connection>, String> {
    let connection = Connection::open(path).map_err(|e| e.to_string())?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|e| e.to_string())?;
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;
    if version as usize > MIGRATIONS.len() {
        return Ok(None);
    }
    if version == 0 {
        // Must precede the first table to take effect.
        connection
            .execute_batch("PRAGMA auto_vacuum = INCREMENTAL;")
            .map_err(|e| e.to_string())?;
    }
    connection
        .execute_batch(
            "PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA synchronous = NORMAL;",
        )
        .map_err(|e| e.to_string())?;
    for (index, migration) in MIGRATIONS.iter().enumerate().skip(version as usize) {
        let tx = connection
            .unchecked_transaction()
            .map_err(|e| e.to_string())?;
        tx.execute_batch(migration)
            .map_err(|e| format!("migration {}: {e}", index + 1))?;
        tx.pragma_update(None, "user_version", (index + 1) as i64)
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
    }
    Ok(Some(connection))
}

struct Store {
    connection: Connection,
    budget: u64,
    pending: usize,
    transaction_started: Option<Instant>,
}

impl Store {
    fn size(&self) -> u64 {
        let pragma = |name: &str| -> u64 {
            self.connection
                .query_row(&format!("PRAGMA {name}"), [], |row| row.get::<_, i64>(0))
                .unwrap_or(0)
                .max(0) as u64
        };
        let page = pragma("page_size");
        (pragma("page_count").saturating_sub(pragma("freelist_count"))) * page
    }

    fn runs(&self) -> u64 {
        self.connection
            .query_row("SELECT COUNT(*) FROM run", [], |row| row.get::<_, i64>(0))
            .unwrap_or(0)
            .max(0) as u64
    }

    fn begin(&mut self) -> rusqlite::Result<()> {
        if self.transaction_started.is_none() {
            self.connection.execute_batch("BEGIN")?;
            self.transaction_started = Some(Instant::now());
        }
        Ok(())
    }

    fn commit(&mut self) -> rusqlite::Result<()> {
        if self.transaction_started.take().is_some() {
            self.connection.execute_batch("COMMIT")?;
            self.pending = 0;
        }
        Ok(())
    }

    /// The slide's row, matching identity first, then fingerprint; records
    /// a newly known identity on a fingerprint match (key migration).
    fn slide_id(&mut self, keys: &SlideKeys, create: bool) -> rusqlite::Result<Option<i64>> {
        let by_identity = match &keys.identity {
            Some(identity) => self
                .connection
                .query_row(
                    "SELECT id FROM slide WHERE identity = ?1",
                    params![identity],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?,
            None => None,
        };
        let by_fingerprint = match &keys.fingerprint {
            Some(fingerprint) => self
                .connection
                .query_row(
                    "SELECT id, identity FROM slide WHERE fingerprint = ?1",
                    params![fingerprint],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?)),
                )
                .optional()?,
            None => None,
        };
        let id = match (by_identity, by_fingerprint) {
            (Some(id), _) => Some(id),
            (None, Some((id, known))) => {
                if known.is_none() && keys.identity.is_some() {
                    self.connection.execute(
                        "UPDATE slide SET identity = ?1 WHERE id = ?2",
                        params![keys.identity, id],
                    )?;
                }
                Some(id)
            }
            (None, None) if create && (keys.identity.is_some() || keys.fingerprint.is_some()) => {
                self.connection.execute(
                    "INSERT INTO slide (identity, fingerprint, width, height, last_used) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![keys.identity, keys.fingerprint, keys.width as i64, keys.height as i64, now()],
                )?;
                Some(self.connection.last_insert_rowid())
            }
            (None, None) => None,
        };
        if let Some(id) = id {
            self.connection.execute(
                "UPDATE slide SET last_used = ?1 WHERE id = ?2",
                params![now(), id],
            )?;
        }
        Ok(id)
    }

    fn run_id(&mut self, slide_id: i64, run_key: &str) -> rusqlite::Result<Option<i64>> {
        let id = self
            .connection
            .query_row(
                "SELECT id FROM run WHERE slide_id = ?1 AND run_key = ?2",
                params![slide_id, run_key],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if let Some(id) = id {
            self.connection.execute(
                "UPDATE run SET last_used = ?1 WHERE id = ?2",
                params![now(), id],
            )?;
        }
        Ok(id)
    }

    fn write(&mut self, write: &RunWrite) -> rusqlite::Result<()> {
        self.begin()?;
        let Some(slide_id) = self.slide_id(&write.slide, true)? else {
            return Ok(());
        };
        let run_id = match self.run_id(slide_id, &write.run_key)? {
            Some(id) => id,
            None => {
                self.connection.execute(
                    "INSERT INTO run (slide_id, run_key, manifest, cols, rows, created, last_used) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
                    params![slide_id, write.run_key, write.manifest, write.cols, write.rows, now()],
                )?;
                self.connection.last_insert_rowid()
            }
        };
        let mut statement = self.connection.prepare_cached(
            "INSERT OR REPLACE INTO tile (run_id, col, row, state, mae, max_error, embedding, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;
        for tile in &write.tiles {
            let (mae, max) = match tile.state {
                TileState::Analysed { mae, max } => (Some(f64::from(mae)), Some(f64::from(max))),
                _ => (None, None),
            };
            statement.execute(params![
                run_id,
                tile.coord.col,
                tile.coord.row,
                tile.state.code(),
                mae,
                max,
                tile.embedding.as_deref().map(f32_blob),
                tile.payload.as_ref().map(|p| p.encode()),
            ])?;
        }
        drop(statement);
        self.pending += write.tiles.len();
        Ok(())
    }

    fn load_run(
        &mut self,
        slide: &SlideKeys,
        run_key: &str,
    ) -> rusqlite::Result<Vec<(TileCoord, TileState)>> {
        self.commit()?;
        let Some(slide_id) = self.slide_id(slide, false)? else {
            return Ok(Vec::new());
        };
        let Some(run_id) = self.run_id(slide_id, run_key)? else {
            return Ok(Vec::new());
        };
        let mut statement = self
            .connection
            .prepare("SELECT col, row, state, mae, max_error FROM tile WHERE run_id = ?1")?;
        let rows = statement.query_map(params![run_id], |row| {
            let coord = TileCoord::new(row.get(0)?, row.get(1)?);
            let state = match row.get::<_, i64>(2)? {
                0 => TileState::Analysed {
                    mae: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0) as f32,
                    max: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0) as f32,
                },
                1 => TileState::Background,
                _ => TileState::Failed,
            };
            Ok((coord, state))
        })?;
        rows.collect()
    }

    fn load_payloads(
        &mut self,
        slide: &SlideKeys,
        run_key: &str,
        coords: &[TileCoord],
    ) -> rusqlite::Result<Vec<(TileCoord, TilePayload)>> {
        self.commit()?;
        let Some(slide_id) = self.slide_id(slide, false)? else {
            return Ok(Vec::new());
        };
        let Some(run_id) = self.run_id(slide_id, run_key)? else {
            return Ok(Vec::new());
        };
        let mut statement = self.connection.prepare_cached(
            "SELECT payload FROM tile WHERE run_id = ?1 AND col = ?2 AND row = ?3 AND payload IS NOT NULL",
        )?;
        let mut found = Vec::new();
        for coord in coords {
            let blob: Option<Vec<u8>> = statement
                .query_row(params![run_id, coord.col, coord.row], |row| row.get(0))
                .optional()?;
            if let Some(payload) = blob.as_deref().and_then(TilePayload::decode) {
                found.push((*coord, payload));
            }
        }
        Ok(found)
    }

    /// Deletes least recently used runs until the database fits its budget.
    fn enforce_budget(&mut self) -> rusqlite::Result<()> {
        self.commit()?;
        let mut size = self.size();
        if size <= self.budget {
            return Ok(());
        }
        let target = self.budget / 10 * 9;
        while size > target {
            let oldest: Option<i64> = self
                .connection
                .query_row(
                    "SELECT id FROM run ORDER BY last_used ASC, id ASC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()?;
            let Some(oldest) = oldest else { break };
            self.connection
                .execute("DELETE FROM run WHERE id = ?1", params![oldest])?;
            let freed_before = size;
            size = self.size();
            if size >= freed_before {
                // Deleting released nothing measurable; stop rather than
                // wiping everything.
                break;
            }
        }
        self.connection.execute(
            "DELETE FROM slide WHERE id NOT IN (SELECT slide_id FROM run)",
            [],
        )?;
        self.connection
            .execute_batch("PRAGMA incremental_vacuum;")?;
        Ok(())
    }

    fn clear(&mut self) -> rusqlite::Result<()> {
        self.commit()?;
        self.connection
            .execute_batch("DELETE FROM tile; DELETE FROM run; DELETE FROM slide;")?;
        self.connection
            .execute_batch("PRAGMA incremental_vacuum;")?;
        self.connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }
}

const COMMIT_EVERY: Duration = Duration::from_millis(500);
const COMMIT_TILES: usize = 2048;

fn serve(mut store: Store, rx: Receiver<Request>, status: Arc<Mutex<CacheStatus>>) {
    let mut writes_since_check = 0usize;
    let report = |store: &Store, error: Option<String>| {
        let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
        status.bytes = store.size();
        status.budget = store.budget;
        status.runs = store.runs();
        if error.is_some() {
            status.error = error;
        }
    };
    report(&store, None);
    loop {
        let timeout = match store.transaction_started {
            Some(started) => COMMIT_EVERY.saturating_sub(started.elapsed()),
            None => Duration::from_secs(3600),
        };
        let request = match rx.recv_timeout(timeout) {
            Ok(request) => Some(request),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let result: rusqlite::Result<()> = (|| {
            match request {
                Some(Request::Write(write)) => {
                    store.write(&write)?;
                    writes_since_check += write.tiles.len();
                }
                Some(Request::LoadRun {
                    slide,
                    run_key,
                    reply,
                }) => {
                    let _ = reply.send(store.load_run(&slide, &run_key).unwrap_or_default());
                }
                Some(Request::LoadPayloads {
                    slide,
                    run_key,
                    coords,
                    reply,
                }) => {
                    let _ = reply.send(
                        store
                            .load_payloads(&slide, &run_key, &coords)
                            .unwrap_or_default(),
                    );
                }
                Some(Request::SetBudget(budget)) => {
                    store.budget = budget;
                    store.enforce_budget()?;
                    report(&store, None);
                }
                Some(Request::Clear(reply)) => {
                    let result = store.clear().map_err(|e| e.to_string());
                    let _ = reply.send(result);
                    report(&store, None);
                }
                Some(Request::Flush(reply)) => {
                    store.commit()?;
                    report(&store, None);
                    let _ = reply.send(());
                }
                None => {}
            }
            let due = store
                .transaction_started
                .is_some_and(|started| started.elapsed() >= COMMIT_EVERY)
                || store.pending >= COMMIT_TILES;
            if due {
                store.commit()?;
                if writes_since_check >= 1024 {
                    writes_since_check = 0;
                    store.enforce_budget()?;
                }
                report(&store, None);
            }
            Ok(())
        })();
        if let Err(error) = result {
            let _ = store.connection.execute_batch("ROLLBACK");
            store.transaction_started = None;
            store.pending = 0;
            report(&store, Some(format!("result cache: {error}")));
        }
    }
    let _ = store.commit();
    let _ = store.enforce_budget();
    report(&store, None);
}

impl ResultCache {
    /// Opens the cache in `dir`. Problems disable persistence (results stay
    /// in memory) and are reported through [`Self::status`].
    pub fn open(dir: &Path, budget: u64) -> Self {
        let status = Arc::new(Mutex::new(CacheStatus {
            budget,
            ..Default::default()
        }));
        let path = dir.join(DB_FILE);
        let opened = std::fs::create_dir_all(dir)
            .map_err(|e| e.to_string())
            .and_then(|()| match open_connection(&path) {
                Ok(connection) => Ok(connection),
                Err(error) => {
                    // A damaged cache is moved aside and rebuilt: it only
                    // holds results that can be recomputed.
                    let aside = path.with_extension("sqlite.damaged");
                    let _ = std::fs::rename(&path, &aside);
                    for suffix in ["-wal", "-shm"] {
                        let mut side = path.clone().into_os_string();
                        side.push(suffix);
                        let _ = std::fs::remove_file(PathBuf::from(side));
                    }
                    open_connection(&path).map_err(|again| format!("{error}; then {again}"))
                }
            });
        let connection = match opened {
            Ok(Some(connection)) => connection,
            Ok(None) => {
                status.lock().unwrap_or_else(|e| e.into_inner()).error = Some(
                    "the result cache was created by a newer eovae; results are kept in memory only".to_string(),
                );
                return Self::disabled(status);
            }
            Err(error) => {
                status.lock().unwrap_or_else(|e| e.into_inner()).error = Some(format!(
                    "the result cache is unavailable ({error}); results are kept in memory only"
                ));
                return Self::disabled(status);
            }
        };
        status.lock().unwrap_or_else(|e| e.into_inner()).enabled = true;
        let (tx, rx) = mpsc::sync_channel::<Request>(16);
        let store = Store {
            connection,
            budget,
            pending: 0,
            transaction_started: None,
        };
        let thread_status = Arc::clone(&status);
        let handle = std::thread::Builder::new()
            .name("eovae-cache".to_string())
            .spawn(move || serve(store, rx, thread_status))
            .ok();
        Self {
            tx: Mutex::new(Some(tx)),
            status,
            handle: Mutex::new(handle),
        }
    }

    fn disabled(status: Arc<Mutex<CacheStatus>>) -> Self {
        Self {
            tx: Mutex::new(None),
            status,
            handle: Mutex::new(None),
        }
    }

    fn sender(&self) -> Option<SyncSender<Request>> {
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn status(&self) -> CacheStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Queues writes; blocks while the cache thread is behind.
    pub fn write(&self, write: RunWrite) {
        if let Some(tx) = self.sender() {
            let _ = tx.send(Request::Write(write));
        }
    }

    /// A run's stored tiles. Blocks; never call it on the UI thread.
    pub fn load_run(&self, slide: &SlideKeys, run_key: &str) -> Vec<(TileCoord, TileState)> {
        let Some(tx) = self.sender() else {
            return Vec::new();
        };
        let (reply, wait) = mpsc::channel();
        if tx
            .send(Request::LoadRun {
                slide: slide.clone(),
                run_key: run_key.to_string(),
                reply,
            })
            .is_err()
        {
            return Vec::new();
        }
        wait.recv().unwrap_or_default()
    }

    /// Stored pixels of tiles. Blocks; never call it on the UI thread.
    pub fn load_payloads(
        &self,
        slide: &SlideKeys,
        run_key: &str,
        coords: Vec<TileCoord>,
    ) -> Vec<(TileCoord, TilePayload)> {
        let Some(tx) = self.sender() else {
            return Vec::new();
        };
        let (reply, wait) = mpsc::channel();
        if tx
            .send(Request::LoadPayloads {
                slide: slide.clone(),
                run_key: run_key.to_string(),
                coords,
                reply,
            })
            .is_err()
        {
            return Vec::new();
        }
        wait.recv().unwrap_or_default()
    }

    pub fn set_budget(&self, budget: u64) {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).budget = budget;
        if let Some(tx) = self.sender() {
            let _ = tx.send(Request::SetBudget(budget));
        }
    }

    /// Deletes everything. Blocks; never call it on the UI thread.
    pub fn clear(&self) -> Result<(), String> {
        let Some(tx) = self.sender() else {
            return Err("the result cache is disabled".to_string());
        };
        let (reply, wait) = mpsc::channel();
        tx.send(Request::Clear(reply))
            .map_err(|_| "the result cache stopped".to_string())?;
        wait.recv()
            .map_err(|_| "the result cache stopped".to_string())?
    }

    /// Commits pending writes. Blocks.
    pub fn flush(&self) {
        if let Some(tx) = self.sender() {
            let (reply, wait) = mpsc::channel();
            if tx.send(Request::Flush(reply)).is_ok() {
                let _ = wait.recv_timeout(Duration::from_secs(10));
            }
        }
    }

    /// Commits and closes the database.
    pub fn shutdown(&self) {
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(handle) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ResultCache {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(fingerprint: &str) -> SlideKeys {
        SlideKeys {
            identity: None,
            fingerprint: Some(fingerprint.to_string()),
            width: 1000,
            height: 800,
        }
    }

    fn write(slide: SlideKeys, run: &str, tiles: Vec<DbTile>) -> RunWrite {
        RunWrite {
            slide,
            run_key: run.to_string(),
            manifest: "{}".to_string(),
            cols: 4,
            rows: 4,
            tiles,
        }
    }

    fn analysed(col: u32, mae: f32, payload: Option<Arc<TilePayload>>) -> DbTile {
        DbTile {
            coord: TileCoord::new(col, 0),
            state: TileState::Analysed {
                mae,
                max: mae * 2.0,
            },
            embedding: Some(vec![1.0, 2.0]),
            payload,
        }
    }

    fn payload(size: u32) -> Arc<TilePayload> {
        Arc::new(TilePayload {
            size,
            valid_w: size,
            valid_h: size,
            recon: (0..size * size * 3).map(|i| (i % 251) as u8).collect(),
            error: (0..size * size).map(|i| (i % 13) as u8).collect(),
        })
    }

    #[test]
    fn results_persist_across_reopening_and_are_keyed_by_slide_and_run() {
        let dir = tempfile::tempdir().unwrap();
        {
            let cache = ResultCache::open(dir.path(), 1 << 30);
            assert!(cache.status().enabled);
            cache.write(write(
                keys("fp-a"),
                "run-1",
                vec![
                    analysed(0, 0.01, Some(payload(8))),
                    DbTile {
                        coord: TileCoord::new(1, 0),
                        state: TileState::Background,
                        embedding: None,
                        payload: None,
                    },
                ],
            ));
            cache.shutdown();
        }
        let cache = ResultCache::open(dir.path(), 1 << 30);
        let mut loaded = cache.load_run(&keys("fp-a"), "run-1");
        loaded.sort_by_key(|(c, _)| c.key());
        assert_eq!(loaded.len(), 2);
        assert!(
            matches!(loaded[0].1, TileState::Analysed { mae, .. } if (mae - 0.01).abs() < 1e-6)
        );
        assert_eq!(loaded[1].1, TileState::Background);
        assert!(
            cache.load_run(&keys("fp-b"), "run-1").is_empty(),
            "other slide"
        );
        assert!(
            cache.load_run(&keys("fp-a"), "run-2").is_empty(),
            "other run"
        );
        let payloads = cache.load_payloads(
            &keys("fp-a"),
            "run-1",
            vec![TileCoord::new(0, 0), TileCoord::new(1, 0)],
        );
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].1, *payload(8));
    }

    #[test]
    fn fingerprint_keys_migrate_to_host_identities() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ResultCache::open(dir.path(), 1 << 30);
        cache.write(write(keys("fp-a"), "r", vec![analysed(0, 0.5, None)]));
        // Later the host knows the slide's identity: the old rows are found
        // through the fingerprint and adopt the identity.
        let both = SlideKeys {
            identity: Some("quickhash1:xyz".to_string()),
            ..keys("fp-a")
        };
        assert_eq!(cache.load_run(&both, "r").len(), 1);
        let identity_only = SlideKeys {
            identity: Some("quickhash1:xyz".to_string()),
            fingerprint: None,
            width: 1000,
            height: 800,
        };
        assert_eq!(cache.load_run(&identity_only, "r").len(), 1);
    }

    #[test]
    fn the_budget_evicts_least_recently_used_runs_and_clear_empties() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ResultCache::open(dir.path(), 1 << 30);
        for run in 0..6 {
            let tiles = (0..4)
                .map(|col| analysed(col, 0.1, Some(payload(64))))
                .collect();
            cache.write(write(keys("fp"), &format!("run-{run}"), tiles));
            cache.flush();
        }
        let before = cache.status();
        assert!(before.bytes > 0);
        assert_eq!(before.runs, 6);
        // Shrink the budget to about half: the oldest runs go first.
        cache.set_budget(before.bytes / 2);
        cache.flush();
        let after = cache.status();
        assert!(after.runs < 6 && after.runs > 0, "{after:?}");
        assert!(
            cache.load_run(&keys("fp"), "run-0").is_empty(),
            "oldest evicted"
        );
        assert!(
            !cache.load_run(&keys("fp"), "run-5").is_empty(),
            "newest kept"
        );
        cache.clear().unwrap();
        assert!(cache.load_run(&keys("fp"), "run-5").is_empty());
        assert_eq!(cache.status().runs, 0);
    }

    #[test]
    fn databases_from_newer_versions_are_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        {
            let connection = Connection::open(dir.path().join(DB_FILE)).unwrap();
            connection.pragma_update(None, "user_version", 99).unwrap();
        }
        let cache = ResultCache::open(dir.path(), 1 << 30);
        let status = cache.status();
        assert!(!status.enabled);
        assert!(status.error.unwrap().contains("newer"));
        // Using it is harmless.
        cache.write(write(keys("fp"), "r", vec![]));
        assert!(cache.load_run(&keys("fp"), "r").is_empty());
    }

    #[test]
    fn damaged_databases_are_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(DB_FILE),
            b"this is not a database at all, not even close",
        )
        .unwrap();
        let cache = ResultCache::open(dir.path(), 1 << 30);
        assert!(cache.status().enabled, "{:?}", cache.status());
        cache.write(write(keys("fp"), "r", vec![analysed(0, 0.2, None)]));
        assert_eq!(cache.load_run(&keys("fp"), "r").len(), 1);
    }
}
