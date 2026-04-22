//! Per-tile metadata types and serialization (CSV / JSON).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// Metadata record for a single extracted tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileRecord {
    /// Original input path for the slide (as given on the CLI or discovered).
    pub slide_path: String,
    /// Fast sampled slide fingerprint used for indexing/debugging.
    pub fingerprint: String,
    /// Full SHA-256 of the slide file, computed during export.
    pub file_sha256: String,
    /// Stem of the slide filename (no extension).
    pub slide_stem: String,
    /// Relative path to the written tile image (relative to the dataset root).
    pub tile_path: String,
    /// X origin of the tile in level-0 pixels.
    pub x: u64,
    /// Y origin of the tile in level-0 pixels.
    pub y: u64,
    /// Requested tile size.
    pub tile_size: u32,
    /// Actual written width (always == tile_size for full tiles).
    pub width: u32,
    /// Actual written height (always == tile_size for full tiles).
    pub height: u32,
    /// Slide width at level 0.
    pub slide_width: u64,
    /// Slide height at level 0.
    pub slide_height: u64,
    /// Pyramid level used for extraction (always 0 for this pipeline).
    pub level: u32,
    /// Microns per pixel (X) if available.
    pub mpp_x: Option<f64>,
    /// Microns per pixel (Y) if available.
    pub mpp_y: Option<f64>,
}

/// Write tile records as CSV to the given path.
pub fn write_csv(records: &[TileRecord], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut wtr = csv::Writer::from_writer(std::io::BufWriter::new(file));
    for rec in records {
        wtr.serialize(rec).map_err(std::io::Error::other)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Write tile records as a JSON array to the given path.
pub fn write_json(records: &[TileRecord], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, records).map_err(std::io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> TileRecord {
        TileRecord {
            slide_path: "slides/test.svs".into(),
            fingerprint: "fingerprint".into(),
            file_sha256: "sha256".into(),
            slide_stem: "test".into(),
            tile_path: "slides/test/test_x000000_y000000_s512.png".into(),
            x: 0,
            y: 0,
            tile_size: 512,
            width: 512,
            height: 512,
            slide_width: 50000,
            slide_height: 40000,
            level: 0,
            mpp_x: Some(0.25),
            mpp_y: Some(0.25),
        }
    }

    #[test]
    fn test_csv_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("meta.csv");
        let records = vec![sample_record()];
        write_csv(&records, &csv_path).unwrap();

        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert!(content.contains("slide_path"));
        assert!(content.contains("test.svs"));
    }

    #[test]
    fn test_json_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("meta.json");
        let records = vec![sample_record()];
        write_json(&records, &json_path).unwrap();

        let content = std::fs::read_to_string(&json_path).unwrap();
        let parsed: Vec<TileRecord> = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].x, 0);
    }
}
