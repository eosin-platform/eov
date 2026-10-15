//! Per-tile and per-slide metadata, written incrementally (DS-3).
//!
//! Records are appended (CSV and/or JSON Lines) and flushed as each chunk of
//! a slide completes, so the metadata survives errors, crashes and
//! cancellation, and an interrupted run can be resumed ([`prepare_resume`]
//! repairs what a crash can leave behind). `metadata.json` (a JSON array) is
//! produced from `metadata.jsonl` when a run ends.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::config::MetadataFormat;

/// Metadata of one written tile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileRecord {
    /// Slide path as given or discovered.
    pub slide_path: String,
    /// Sampled slide fingerprint (fast; see `file_id::compute_fingerprint`).
    pub fingerprint: String,
    /// SHA-256 of the slide's files (empty when hashing is disabled).
    pub file_sha256: String,
    /// File stem of the slide (sanitised).
    pub slide_stem: String,
    /// Tile image path relative to the dataset root.
    pub tile_path: String,
    /// Level-0 origin of the tile.
    pub x: u64,
    pub y: u64,
    /// Requested tile size (output pixels).
    pub tile_size: u32,
    /// Written image size (always `tile_size`).
    pub width: u32,
    pub height: u32,
    /// Slide size at level 0.
    pub slide_width: u64,
    pub slide_height: u64,
    /// Pyramid level the pixels were read from.
    pub level: u32,
    /// Slide pixel size at level 0 (µm).
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    /// Unique slide key used in output paths.
    #[serde(default)]
    pub slide_id: String,
    /// Level-0 footprint of the tile.
    #[serde(default)]
    pub level0_width: f64,
    #[serde(default)]
    pub level0_height: f64,
    /// Level-0 pixels per output pixel.
    #[serde(default)]
    pub downsample: f64,
    /// Pixel size of the written tile (µm).
    #[serde(default)]
    pub output_mpp_x: Option<f64>,
    #[serde(default)]
    pub output_mpp_y: Option<f64>,
    /// Fraction of tissue pixels (when a tissue filter is used).
    #[serde(default)]
    pub tissue_fraction: Option<f32>,
    /// Fraction of white or transparent pixels.
    #[serde(default)]
    pub white_fraction: f32,
    /// Majority annotation class and its coverage.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub label_coverage: Option<f32>,
    /// Mask image path relative to the dataset root.
    #[serde(default)]
    pub mask_path: Option<String>,
    #[serde(default)]
    pub vendor: Option<String>,
    #[serde(default)]
    pub objective_power: Option<f64>,
    /// Stain normalisation applied to the tile, if any.
    #[serde(default)]
    pub stain_normalization: Option<String>,
}

/// Outcome of one slide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SlideStatus {
    Complete,
    /// Processed, but some tiles could not be read.
    Partial,
    Skipped,
    Failed,
    Cancelled,
}

/// Metadata of one slide (`slides.csv` / `slides.jsonl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlideRecord {
    pub slide_id: String,
    pub slide_path: String,
    pub status: SlideStatus,
    pub message: Option<String>,
    pub vendor: Option<String>,
    pub objective_power: Option<f64>,
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    pub width: u64,
    pub height: u64,
    pub level_count: u32,
    pub read_level: Option<u32>,
    pub downsample: Option<f64>,
    pub output_mpp: Option<f64>,
    /// Tissue fraction of the whole slide (low-resolution mask).
    pub tissue_fraction: Option<f32>,
    pub candidates: u64,
    pub written: u64,
    pub existing: u64,
    pub skipped_white: u64,
    pub skipped_tissue: u64,
    pub skipped_annotation: u64,
    pub failed: u64,
    pub fingerprint: Option<String>,
    pub file_sha256: Option<String>,
    pub stain_normalization: Option<String>,
    pub annotation_file: Option<String>,
}

pub(crate) const CSV_FILE: &str = "metadata.csv";
pub(crate) const JSONL_FILE: &str = "metadata.jsonl";
pub(crate) const JSON_FILE: &str = "metadata.json";
pub(crate) const SLIDES_CSV_FILE: &str = "slides.csv";
pub(crate) const SLIDES_JSONL_FILE: &str = "slides.jsonl";

fn wants(format: Option<MetadataFormat>) -> (bool, bool, bool) {
    // (csv, jsonl, final json array)
    match format {
        None => (true, true, true),
        Some(MetadataFormat::Csv) => (true, false, false),
        Some(MetadataFormat::Json) => (false, true, true),
        Some(MetadataFormat::JsonLines) => (false, true, false),
    }
}

struct CsvAppender {
    writer: csv::Writer<BufWriter<File>>,
}

impl CsvAppender {
    fn open(path: &Path, append: bool) -> std::io::Result<Self> {
        let existing = append && path.metadata().map(|m| m.len() > 0).unwrap_or(false);
        let file = OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(path)?;
        let writer = csv::WriterBuilder::new()
            .has_headers(!existing)
            .from_writer(BufWriter::new(file));
        Ok(Self { writer })
    }

    fn write<T: Serialize>(&mut self, record: &T) -> std::io::Result<()> {
        self.writer.serialize(record).map_err(std::io::Error::other)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().get_ref().sync_data()
    }
}

struct JsonlAppender {
    writer: BufWriter<File>,
}

impl JsonlAppender {
    fn open(path: &Path, append: bool) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    fn write<T: Serialize>(&mut self, record: &T) -> std::io::Result<()> {
        serde_json::to_writer(&mut self.writer, record).map_err(std::io::Error::other)?;
        self.writer.write_all(b"\n")
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()
    }
}

/// Removes a last record that was only partly written (a crash during an
/// append), so that appending continues on a fresh line instead of gluing
/// the next record onto the fragment. In CSV files a newline inside a
/// quoted field does not end a record. Returns the number of bytes removed.
fn repair_tail(path: &Path, csv: bool) -> std::io::Result<u64> {
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len();
    // Offset just past the last record terminator.
    let mut end = 0u64;
    let mut offset = 0u64;
    let mut in_quotes = false;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        for &byte in &buf[..n] {
            offset += 1;
            if csv && byte == b'"' {
                // Quotes inside quoted fields are doubled, so the parity
                // says whether we are inside a field.
                in_quotes = !in_quotes;
            } else if byte == b'\n' && !in_quotes {
                end = offset;
            }
        }
    }
    if end == len {
        return Ok(0);
    }
    file.set_len(end)?;
    file.sync_data()?;
    warn!(
        "{}: removed a partly written last record ({} bytes) left by an interrupted run",
        path.display(),
        len - end
    );
    Ok(len - end)
}

/// Streams the records of a JSON Lines file; lines that do not parse are
/// skipped with a warning. A missing file has no records.
fn for_each_jsonl<T: DeserializeOwned>(
    path: &Path,
    mut f: impl FnMut(T) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    let mut invalid = 0u64;
    for line in BufReader::new(file).split(b'\n') {
        let line = line?;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice(&line) {
            Ok(record) => f(record)?,
            Err(_) => invalid += 1,
        }
    }
    if invalid > 0 {
        warn!("{}: skipped {invalid} unreadable lines", path.display());
    }
    Ok(())
}

/// Streams the records of a CSV file with a header; rows that do not parse
/// are skipped with a warning. A missing file has no records.
fn for_each_csv<T: DeserializeOwned>(
    path: &Path,
    mut f: impl FnMut(T) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    let mut reader = csv::Reader::from_reader(BufReader::new(file));
    let mut invalid = 0u64;
    for record in reader.deserialize::<T>() {
        match record {
            Ok(record) => f(record)?,
            Err(e) => match e.into_kind() {
                csv::ErrorKind::Io(e) => return Err(e),
                _ => invalid += 1,
            },
        }
    }
    if invalid > 0 {
        warn!("{}: skipped {invalid} unreadable rows", path.display());
    }
    Ok(())
}

/// What earlier sessions recorded in an output directory.
#[derive(Debug, Default)]
pub(crate) struct ResumeState {
    /// Paths (relative to the output directory) of the recorded tiles.
    pub tiles: HashSet<String>,
    /// The latest record of each slide, by slide id.
    pub slides: HashMap<String, SlideRecord>,
}

/// Prepares an output directory for resuming and returns what is already
/// recorded there.
///
/// * Partly written last records (a crash during an append) are removed.
/// * Tile records found in only one of `metadata.csv` and `metadata.jsonl`
///   — a crash between the two appends of a batch, or an earlier session
///   that used another metadata format — are appended to the other file
///   when this run writes it, so both list the same tiles.
pub(crate) fn prepare_resume(
    dir: &Path,
    format: Option<MetadataFormat>,
) -> std::io::Result<ResumeState> {
    for (file, csv) in [
        (CSV_FILE, true),
        (JSONL_FILE, false),
        (SLIDES_CSV_FILE, true),
        (SLIDES_JSONL_FILE, false),
    ] {
        repair_tail(&dir.join(file), csv)?;
    }
    let (want_csv, want_jsonl, _) = wants(format);
    let (csv_path, jsonl_path) = (dir.join(CSV_FILE), dir.join(JSONL_FILE));

    let mut in_jsonl = HashSet::new();
    for_each_jsonl::<TileRecord>(&jsonl_path, |r| {
        in_jsonl.insert(r.tile_path);
        Ok(())
    })?;
    let mut in_csv = HashSet::new();
    let mut jsonl_out = None;
    let mut added_to_jsonl = 0u64;
    for_each_csv::<TileRecord>(&csv_path, |r| {
        if in_csv.insert(r.tile_path.clone()) && want_jsonl && !in_jsonl.contains(&r.tile_path) {
            let out = match &mut jsonl_out {
                Some(out) => out,
                None => jsonl_out.insert(JsonlAppender::open(&jsonl_path, true)?),
            };
            out.write(&r)?;
            added_to_jsonl += 1;
        }
        Ok(())
    })?;
    if let Some(mut out) = jsonl_out {
        out.flush()?;
    }
    if want_csv && in_jsonl.iter().any(|p| !in_csv.contains(p)) {
        let mut out = CsvAppender::open(&csv_path, true)?;
        let mut added = 0u64;
        for_each_jsonl::<TileRecord>(&jsonl_path, |r| {
            if !in_csv.contains(&r.tile_path) {
                out.write(&r)?;
                in_csv.insert(r.tile_path);
                added += 1;
            }
            Ok(())
        })?;
        out.flush()?;
        warn!(
            "{}: added {added} tile records found only in {JSONL_FILE}",
            csv_path.display()
        );
    }
    if added_to_jsonl > 0 {
        warn!(
            "{}: added {added_to_jsonl} tile records found only in {CSV_FILE}",
            jsonl_path.display()
        );
    }
    let mut tiles = in_jsonl;
    tiles.extend(in_csv);
    Ok(ResumeState {
        tiles,
        slides: latest_slide_records(dir)?.into_iter().collect(),
    })
}

/// The latest record of every slide, in the order slides first appear.
/// `slides.jsonl` is authoritative; `slides.csv` fills in slides it lacks.
fn latest_slide_records(dir: &Path) -> std::io::Result<Vec<(String, SlideRecord)>> {
    fn collect(
        records: &mut Vec<(String, SlideRecord)>,
        index: &mut HashMap<String, usize>,
        record: SlideRecord,
    ) {
        match index.get(&record.slide_id) {
            Some(&i) => records[i].1 = record,
            None => {
                index.insert(record.slide_id.clone(), records.len());
                records.push((record.slide_id.clone(), record));
            }
        }
    }
    let mut records = Vec::new();
    let mut index = HashMap::new();
    for_each_jsonl::<SlideRecord>(&dir.join(SLIDES_JSONL_FILE), |r| {
        collect(&mut records, &mut index, r);
        Ok(())
    })?;
    let from_jsonl: HashSet<String> = index.keys().cloned().collect();
    for_each_csv::<SlideRecord>(&dir.join(SLIDES_CSV_FILE), |r| {
        if !from_jsonl.contains(&r.slide_id) {
            collect(&mut records, &mut index, r);
        }
        Ok(())
    })?;
    Ok(records)
}

/// Writes `path` atomically through a `.part` file. Errors name `path`.
fn write_atomically(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".part");
    let tmp = PathBuf::from(tmp);
    let result = (|| {
        let mut out = BufWriter::new(File::create(&tmp)?);
        write(&mut out)?;
        out.flush()?;
        out.get_ref().sync_data()?;
        drop(out);
        std::fs::rename(&tmp, path)
    })();
    result.map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        std::io::Error::new(e.kind(), format!("{}: {e}", path.display()))
    })
}

/// Appends tile and slide records as a run progresses.
pub struct MetadataWriter {
    dir: PathBuf,
    tiles_csv: Option<CsvAppender>,
    tiles_jsonl: Option<JsonlAppender>,
    slides_csv: Option<CsvAppender>,
    slides_jsonl: Option<JsonlAppender>,
    final_json: bool,
    resumed: bool,
}

impl MetadataWriter {
    /// Opens (appending when `append`) the metadata files for `format`.
    /// Call [`prepare_resume`] first when appending.
    pub fn open(dir: &Path, format: Option<MetadataFormat>, append: bool) -> std::io::Result<Self> {
        let (csv, jsonl, final_json) = wants(format);
        Ok(Self {
            dir: dir.to_path_buf(),
            tiles_csv: csv
                .then(|| CsvAppender::open(&dir.join(CSV_FILE), append))
                .transpose()?,
            tiles_jsonl: jsonl
                .then(|| JsonlAppender::open(&dir.join(JSONL_FILE), append))
                .transpose()?,
            slides_csv: csv
                .then(|| CsvAppender::open(&dir.join(SLIDES_CSV_FILE), append))
                .transpose()?,
            slides_jsonl: jsonl
                .then(|| JsonlAppender::open(&dir.join(SLIDES_JSONL_FILE), append))
                .transpose()?,
            final_json,
            resumed: append,
        })
    }

    /// Appends tile records and flushes them to disk.
    pub fn append_tiles(&mut self, records: &[TileRecord]) -> std::io::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        if let Some(w) = &mut self.tiles_csv {
            for r in records {
                w.write(r)?;
            }
            w.flush()?;
        }
        if let Some(w) = &mut self.tiles_jsonl {
            for r in records {
                w.write(r)?;
            }
            w.flush()?;
        }
        Ok(())
    }

    /// Appends one slide record and flushes it.
    pub fn append_slide(&mut self, record: &SlideRecord) -> std::io::Result<()> {
        if let Some(w) = &mut self.slides_csv {
            w.write(record)?;
            w.flush()?;
        }
        if let Some(w) = &mut self.slides_jsonl {
            w.write(record)?;
            w.flush()?;
        }
        Ok(())
    }

    /// Paths of the files this writer maintains (including the final JSON).
    pub fn paths(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if self.tiles_csv.is_some() {
            out.push(self.dir.join(CSV_FILE));
        }
        if self.tiles_jsonl.is_some() {
            out.push(self.dir.join(JSONL_FILE));
        }
        if self.final_json {
            out.push(self.dir.join(JSON_FILE));
        }
        if self.slides_csv.is_some() {
            out.push(self.dir.join(SLIDES_CSV_FILE));
        }
        if self.slides_jsonl.is_some() {
            out.push(self.dir.join(SLIDES_JSONL_FILE));
        }
        out
    }

    /// Flushes everything; after a resumed run, rewrites the slide files
    /// with one record per slide (the latest); and, if requested, converts
    /// `metadata.jsonl` into the `metadata.json` array (streamed, not held
    /// in memory).
    pub fn finish(mut self) -> std::io::Result<Vec<PathBuf>> {
        for w in [&mut self.tiles_csv, &mut self.slides_csv]
            .into_iter()
            .flatten()
        {
            w.flush()?;
        }
        for w in [&mut self.tiles_jsonl, &mut self.slides_jsonl]
            .into_iter()
            .flatten()
        {
            w.flush()?;
        }
        let paths = self.paths();
        let (slides_csv, slides_jsonl) = (self.slides_csv.is_some(), self.slides_jsonl.is_some());
        drop((self.slides_csv.take(), self.slides_jsonl.take()));
        if self.resumed {
            compact_slides(&self.dir, slides_csv, slides_jsonl)?;
        }
        if self.final_json {
            write_json_array(&self.dir.join(JSONL_FILE), &self.dir.join(JSON_FILE))?;
        }
        Ok(paths)
    }
}

/// Rewrites the slide files with one record per slide: an interrupted
/// slide is recorded again when a resumed run finishes it, and the latest
/// record wins (at the position of the slide's first record).
fn compact_slides(dir: &Path, csv: bool, jsonl: bool) -> std::io::Result<()> {
    let records = latest_slide_records(dir)?;
    if jsonl {
        write_atomically(&dir.join(SLIDES_JSONL_FILE), |out| {
            for (_, record) in &records {
                serde_json::to_writer(&mut *out, record).map_err(std::io::Error::other)?;
                out.write_all(b"\n")?;
            }
            Ok(())
        })?;
    }
    if csv {
        write_atomically(&dir.join(SLIDES_CSV_FILE), |out| {
            let mut writer = csv::Writer::from_writer(&mut *out);
            for (_, record) in &records {
                writer.serialize(record).map_err(std::io::Error::other)?;
            }
            writer.flush()
        })?;
    }
    Ok(())
}

/// Writes the JSON Lines file as a JSON array (atomically). Lines that do
/// not parse are skipped with a warning rather than failing the run.
fn write_json_array(jsonl: &Path, json: &Path) -> std::io::Result<()> {
    write_atomically(json, |out| {
        out.write_all(b"[")?;
        let mut first = true;
        for_each_jsonl::<serde_json::Value>(jsonl, |value| {
            out.write_all(if first { b"\n  " } else { b",\n  " })?;
            serde_json::to_writer(&mut *out, &value).map_err(std::io::Error::other)?;
            first = false;
            Ok(())
        })?;
        out.write_all(if first { b"]\n" } else { b"\n]\n" })
    })
}

/// Tile paths recorded in an output directory, from `metadata.jsonl` and
/// `metadata.csv`. Unreadable lines (e.g. a truncated last line) are
/// ignored.
#[cfg(test)]
pub(crate) fn recorded_tiles(dir: &Path) -> std::io::Result<HashSet<String>> {
    let mut out = HashSet::new();
    for_each_jsonl::<TileRecord>(&dir.join(JSONL_FILE), |r| {
        out.insert(r.tile_path);
        Ok(())
    })?;
    for_each_csv::<TileRecord>(&dir.join(CSV_FILE), |r| {
        out.insert(r.tile_path);
        Ok(())
    })?;
    Ok(out)
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample_record(x: u64) -> TileRecord {
        TileRecord {
            slide_path: "slides/test.svs".into(),
            fingerprint: "fingerprint".into(),
            file_sha256: "sha256".into(),
            slide_stem: "test".into(),
            tile_path: format!("slides/test/test_x{x:06}_y000000_s512.png"),
            x,
            y: 0,
            tile_size: 512,
            width: 512,
            height: 512,
            slide_width: 50000,
            slide_height: 40000,
            level: 0,
            mpp_x: Some(0.25),
            mpp_y: Some(0.25),
            slide_id: "test".into(),
            level0_width: 512.0,
            level0_height: 512.0,
            downsample: 1.0,
            output_mpp_x: Some(0.25),
            output_mpp_y: Some(0.25),
            tissue_fraction: Some(0.75),
            white_fraction: 0.1,
            label: Some("Tumor".into()),
            label_coverage: Some(1.0),
            mask_path: None,
            vendor: Some("aperio".into()),
            objective_power: Some(40.0),
            stain_normalization: None,
        }
    }

    #[test]
    fn records_are_appended_and_resumable() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = MetadataWriter::open(dir.path(), None, false).unwrap();
        w.append_tiles(&[sample_record(0), sample_record(512)])
            .unwrap();
        drop(w);
        // A second session appends without repeating the CSV header.
        let mut w = MetadataWriter::open(dir.path(), None, true).unwrap();
        w.append_tiles(&[sample_record(1024)]).unwrap();
        let paths = w.finish().unwrap();
        assert!(paths.iter().any(|p| p.ends_with(JSON_FILE)));
        let csv = std::fs::read_to_string(dir.path().join(CSV_FILE)).unwrap();
        assert_eq!(csv.matches("slide_path").count(), 1, "one header");
        assert_eq!(csv.lines().count(), 4);
        let json: Vec<TileRecord> =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(JSON_FILE)).unwrap())
                .unwrap();
        assert_eq!(json.len(), 3);
        assert_eq!(json[2], sample_record(1024));
        let recorded = recorded_tiles(dir.path()).unwrap();
        assert_eq!(recorded.len(), 3);
    }

    #[test]
    fn truncated_jsonl_lines_are_ignored_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let mut w =
            MetadataWriter::open(dir.path(), Some(MetadataFormat::JsonLines), false).unwrap();
        w.append_tiles(&[sample_record(0)]).unwrap();
        drop(w);
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.path().join(JSONL_FILE))
            .unwrap();
        f.write_all(b"{\"slide_path\": \"trunc").unwrap();
        assert_eq!(recorded_tiles(dir.path()).unwrap().len(), 1);
    }

    #[test]
    fn csv_only_runs_are_resumable_and_old_csvs_parse() {
        let dir = tempfile::tempdir().unwrap();
        // A pre-1.0 CSV without the new columns.
        std::fs::write(
            dir.path().join(CSV_FILE),
            "slide_path,fingerprint,file_sha256,slide_stem,tile_path,x,y,tile_size,width,height,slide_width,slide_height,level,mpp_x,mpp_y\n\
             a.svs,f,s,a,slides/a/a_x000000_y000000_s512.png,0,0,512,512,512,1000,1000,0,0.25,0.25\n",
        )
        .unwrap();
        let recorded = recorded_tiles(dir.path()).unwrap();
        assert!(recorded.contains("slides/a/a_x000000_y000000_s512.png"));
    }

    pub(crate) fn sample_slide(id: &str, status: SlideStatus, written: u64) -> SlideRecord {
        SlideRecord {
            slide_id: id.into(),
            slide_path: format!("{id}.svs"),
            status,
            message: None,
            vendor: None,
            objective_power: None,
            mpp_x: Some(0.25),
            mpp_y: Some(0.25),
            width: 1000,
            height: 1000,
            level_count: 1,
            read_level: Some(0),
            downsample: Some(1.0),
            output_mpp: Some(0.25),
            tissue_fraction: None,
            candidates: 4,
            written,
            existing: 0,
            skipped_white: 0,
            skipped_tissue: 0,
            skipped_annotation: 0,
            failed: 0,
            fingerprint: Some("f".into()),
            file_sha256: None,
            stain_normalization: None,
            // Quotes, a comma and a newline exercise CSV quoting.
            annotation_file: Some("a, \"quoted\"\nname.geojson".into()),
        }
    }

    fn append_raw(path: &Path, bytes: &[u8]) {
        OpenOptions::new()
            .append(true)
            .open(path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
    }

    fn jsonl_records<T: DeserializeOwned>(path: &Path) -> Vec<T> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn csv_records<T: DeserializeOwned>(path: &Path) -> Vec<T> {
        csv::Reader::from_path(path)
            .unwrap()
            .deserialize()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    fn xs(records: Vec<TileRecord>) -> Vec<u64> {
        records.into_iter().map(|r| r.x).collect()
    }

    #[test]
    fn partly_written_records_are_removed_before_appending() {
        let dir = tempfile::tempdir().unwrap();
        let path = |f: &str| dir.path().join(f);
        let mut w = MetadataWriter::open(dir.path(), None, false).unwrap();
        w.append_tiles(&[sample_record(0)]).unwrap();
        w.append_slide(&sample_slide("a", SlideStatus::Complete, 1))
            .unwrap();
        drop(w);
        // The process died in the middle of the next appends.
        append_raw(&path(JSONL_FILE), b"{\"slide_path\":\"slides/te");
        append_raw(&path(CSV_FILE), b"slides/test.svs,fingerprint,sha");
        // Cut inside a quoted field that contains a newline.
        append_raw(&path(SLIDES_CSV_FILE), b"b,b.svs,complete,\"a\nb");
        append_raw(&path(SLIDES_JSONL_FILE), b"{\"slide_id\":");

        let state = prepare_resume(dir.path(), None).unwrap();
        assert_eq!(state.tiles.len(), 1);
        assert_eq!(
            state.slides.keys().collect::<Vec<_>>(),
            vec![&"a".to_string()]
        );
        let mut w = MetadataWriter::open(dir.path(), None, true).unwrap();
        w.append_tiles(&[sample_record(512)]).unwrap();
        w.append_slide(&sample_slide("b", SlideStatus::Complete, 1))
            .unwrap();
        w.finish().unwrap();

        // Every file parses completely and holds the records of both runs.
        let json: Vec<TileRecord> =
            serde_json::from_str(&std::fs::read_to_string(path(JSON_FILE)).unwrap()).unwrap();
        assert_eq!(json, vec![sample_record(0), sample_record(512)]);
        assert_eq!(jsonl_records::<TileRecord>(&path(JSONL_FILE)), json);
        assert_eq!(csv_records::<TileRecord>(&path(CSV_FILE)), json);
        let slides = vec![
            sample_slide("a", SlideStatus::Complete, 1),
            sample_slide("b", SlideStatus::Complete, 1),
        ];
        assert_eq!(csv_records::<SlideRecord>(&path(SLIDES_CSV_FILE)), slides);
        assert_eq!(
            jsonl_records::<SlideRecord>(&path(SLIDES_JSONL_FILE)),
            slides
        );
    }

    #[test]
    fn csv_and_jsonl_tile_records_are_reconciled_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let path = |f: &str| dir.path().join(f);
        // An earlier run wrote CSV only.
        let mut w = MetadataWriter::open(dir.path(), Some(MetadataFormat::Csv), false).unwrap();
        w.append_tiles(&[sample_record(0), sample_record(512)])
            .unwrap();
        w.finish().unwrap();
        // A JSON Lines run adopts the CSV's tiles into its own file and
        // leaves the CSV, which it does not write, alone.
        let state = prepare_resume(dir.path(), Some(MetadataFormat::JsonLines)).unwrap();
        assert_eq!(state.tiles.len(), 2);
        let mut w =
            MetadataWriter::open(dir.path(), Some(MetadataFormat::JsonLines), true).unwrap();
        w.append_tiles(&[sample_record(1024)]).unwrap();
        w.finish().unwrap();
        assert_eq!(xs(jsonl_records(&path(JSONL_FILE))), vec![0, 512, 1024]);
        assert_eq!(xs(csv_records(&path(CSV_FILE))), vec![0, 512]);
        // A run writing both completes each file.
        let state = prepare_resume(dir.path(), None).unwrap();
        assert_eq!(state.tiles.len(), 3);
        assert_eq!(xs(csv_records(&path(CSV_FILE))), vec![0, 512, 1024]);
        assert_eq!(xs(jsonl_records(&path(JSONL_FILE))), vec![0, 512, 1024]);
        // Preparing again changes nothing.
        prepare_resume(dir.path(), None).unwrap();
        assert_eq!(xs(csv_records(&path(CSV_FILE))), vec![0, 512, 1024]);
        assert_eq!(xs(jsonl_records(&path(JSONL_FILE))), vec![0, 512, 1024]);
    }

    #[test]
    fn resumed_runs_keep_one_record_per_slide() {
        let dir = tempfile::tempdir().unwrap();
        let path = |f: &str| dir.path().join(f);
        let mut w = MetadataWriter::open(dir.path(), None, false).unwrap();
        w.append_slide(&sample_slide("a", SlideStatus::Complete, 4))
            .unwrap();
        w.append_slide(&sample_slide("b", SlideStatus::Cancelled, 1))
            .unwrap();
        w.finish().unwrap();
        let state = prepare_resume(dir.path(), None).unwrap();
        assert_eq!(state.slides["b"].status, SlideStatus::Cancelled);
        let mut w = MetadataWriter::open(dir.path(), None, true).unwrap();
        w.append_slide(&sample_slide("b", SlideStatus::Complete, 3))
            .unwrap();
        w.append_slide(&sample_slide("c", SlideStatus::Complete, 2))
            .unwrap();
        w.finish().unwrap();
        let expected = vec![
            sample_slide("a", SlideStatus::Complete, 4),
            sample_slide("b", SlideStatus::Complete, 3),
            sample_slide("c", SlideStatus::Complete, 2),
        ];
        assert_eq!(
            jsonl_records::<SlideRecord>(&path(SLIDES_JSONL_FILE)),
            expected
        );
        assert_eq!(csv_records::<SlideRecord>(&path(SLIDES_CSV_FILE)), expected);
    }

    #[test]
    fn unreadable_lines_do_not_fail_the_final_json() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = MetadataWriter::open(dir.path(), Some(MetadataFormat::Json), false).unwrap();
        w.append_tiles(&[sample_record(0)]).unwrap();
        drop(w);
        // A line glued together by an interrupted append.
        append_raw(&dir.path().join(JSONL_FILE), b"{\"slide_pa{\"x\":1}\n");
        let mut w = MetadataWriter::open(dir.path(), Some(MetadataFormat::Json), true).unwrap();
        w.append_tiles(&[sample_record(512)]).unwrap();
        w.finish().unwrap();
        let json: Vec<TileRecord> =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(JSON_FILE)).unwrap())
                .unwrap();
        assert_eq!(json, vec![sample_record(0), sample_record(512)]);
    }
}
