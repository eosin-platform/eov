//! Per-tile and per-slide metadata, written incrementally (DS-3).
//!
//! Records are appended (CSV and/or JSON Lines) and flushed as each chunk of
//! a slide completes, so the metadata survives errors, crashes and
//! cancellation, and an interrupted run can be resumed. `metadata.json` (a
//! JSON array) is produced from `metadata.jsonl` when a run ends.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
        self.writer.flush()
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

/// Appends tile and slide records as a run progresses.
pub struct MetadataWriter {
    dir: PathBuf,
    tiles_csv: Option<CsvAppender>,
    tiles_jsonl: Option<JsonlAppender>,
    slides_csv: Option<CsvAppender>,
    slides_jsonl: Option<JsonlAppender>,
    final_json: bool,
}

impl MetadataWriter {
    /// Opens (appending when `append`) the metadata files for `format`.
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

    /// Flushes everything and, if requested, converts `metadata.jsonl` into
    /// the `metadata.json` array (streamed, not held in memory).
    pub fn finish(mut self) -> std::io::Result<Vec<PathBuf>> {
        if let Some(w) = &mut self.tiles_csv {
            w.flush()?;
        }
        if let Some(w) = &mut self.tiles_jsonl {
            w.flush()?;
        }
        let paths = self.paths();
        if self.final_json {
            write_json_array(&self.dir.join(JSONL_FILE), &self.dir.join(JSON_FILE))?;
        }
        Ok(paths)
    }
}

/// Writes the JSON Lines file as a pretty JSON array (atomically).
fn write_json_array(jsonl: &Path, json: &Path) -> std::io::Result<()> {
    let tmp = json.with_extension("json.part");
    {
        let mut out = BufWriter::new(File::create(&tmp)?);
        out.write_all(b"[")?;
        let mut first = true;
        if let Ok(file) = File::open(jsonl) {
            for line in BufReader::new(file).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let value: serde_json::Value =
                    serde_json::from_str(&line).map_err(std::io::Error::other)?;
                out.write_all(if first { b"\n  " } else { b",\n  " })?;
                serde_json::to_writer(&mut out, &value).map_err(std::io::Error::other)?;
                first = false;
            }
        }
        out.write_all(if first { b"]\n" } else { b"\n]\n" })?;
        out.flush()?;
    }
    std::fs::rename(&tmp, json)
}

/// Tile paths already recorded in an output directory (for resuming).
/// Reads `metadata.jsonl`, or `metadata.csv` when there is no JSONL file.
/// A truncated last line (from a crash) is ignored.
pub fn recorded_tiles(dir: &Path) -> std::io::Result<HashSet<String>> {
    let mut out = HashSet::new();
    let jsonl = dir.join(JSONL_FILE);
    if jsonl.is_file() {
        for line in BufReader::new(File::open(&jsonl)?).lines() {
            let line = line?;
            if let Ok(record) = serde_json::from_str::<TileRecord>(&line) {
                out.insert(record.tile_path);
            }
        }
        return Ok(out);
    }
    let csv_path = dir.join(CSV_FILE);
    if csv_path.is_file() {
        let mut reader = csv::Reader::from_path(&csv_path).map_err(std::io::Error::other)?;
        for record in reader.deserialize::<TileRecord>().flatten() {
            out.insert(record.tile_path);
        }
    }
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
}
