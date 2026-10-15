//! An append-only record of what eov ran with (AUDIT §14.6), for research
//! reproducibility: each session's eov version and plugins (id, version,
//! package SHA-256, origin, state), and every plugin install, update,
//! removal, enabling or disabling and application update, from the window
//! or the command line.
//!
//! It is `<data>/audit.jsonl`: one JSON object per line with `time` (UTC,
//! RFC 3339), `unix` (seconds) and `event`. It names no slide paths. When it
//! passes 10 MB it is renamed to `audit.1.jsonl`, replacing the older one.

use serde_json::{Map, Value, json};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_BYTES: u64 = 10 * 1024 * 1024;

/// Serialises writers within this process (other processes append whole
/// lines too, which the OS keeps intact at this size).
static WRITE: Mutex<()> = Mutex::new(());

fn audit_file() -> PathBuf {
    crate::paths::data_dir().join("audit.jsonl")
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// `unix` seconds as `2026-10-15T17:50:08Z`.
fn rfc3339(unix: i64) -> String {
    let (year, month, day) = civil_from_days(unix.div_euclid(86_400));
    let seconds = unix.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}

/// Appends an event; `fields` (a JSON object) are added to it. Failures
/// are logged, never fatal.
pub(crate) fn record(event: &str, fields: Value) {
    if let Err(error) = record_in(&audit_file(), event, fields) {
        tracing::warn!("Could not write the audit log: {error}");
    }
}

fn record_in(path: &Path, event: &str, fields: Value) -> std::io::Result<()> {
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default();
    let mut line = Map::new();
    line.insert("time".into(), json!(rfc3339(unix)));
    line.insert("unix".into(), json!(unix));
    line.insert("event".into(), json!(event));
    if let Value::Object(fields) = fields {
        line.extend(fields);
    }
    let mut text = Value::Object(line).to_string();
    text.push('\n');

    let _guard = WRITE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::metadata(path).is_ok_and(|meta| meta.len() > MAX_BYTES) {
        let _ = fs::rename(path, path.with_extension("1.jsonl"));
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_utc_rfc3339() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_791_999_008), "2026-10-14T17:30:08Z");
        assert_eq!(rfc3339(-1), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn events_are_appended_one_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        record_in(
            &path,
            "plugin_removed",
            json!({"plugin": "a", "version": "1.0.0"}),
        )
        .unwrap();
        record_in(&path, "session_started", json!({"eov": "1.0.0"})).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "plugin_removed");
        assert_eq!(lines[0]["plugin"], "a");
        assert_eq!(lines[1]["eov"], "1.0.0");
        assert!(lines[1]["time"].as_str().unwrap().ends_with('Z'));
    }
}
