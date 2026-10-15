//! Input path expansion and slide discovery (DS-6).
//!
//! * DICOM instances in one directory that belong to the same series
//!   (Series Instance UID) form **one** slide.
//! * During directory scans, TIFFs that cannot be brightfield slides —
//!   single-channel images (masks, label maps) or untiled images — are
//!   skipped and reported. Files given explicitly are always used.
//! * Directory symlink loops are detected; hidden entries are ignored.
//! * Multi-file formats (MIRAX, VMS/VMU, DICOM) list all their member files
//!   so the dataset hash covers the image data, not just the index file.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use tracing::debug;

/// File extensions that EOV can open as whole-slide images (via OpenSlide).
/// Keep in sync with the file-picker filter in the app.
pub const SUPPORTED_SLIDE_EXTENSIONS: &[&str] = &[
    "svs", "tif", "tiff", "dcm", "ndpi", "vms", "vmu", "scn", "mrxs", "svslide", "bif", "czi",
];

/// Maximum directory depth searched.
const MAX_DEPTH: usize = 64;

/// Returns `true` if the file extension is a known slide format.
pub fn is_supported_slide_extension(path: &Path) -> bool {
    extension(path).is_some_and(|ext| SUPPORTED_SLIDE_EXTENSIONS.contains(&ext.as_str()))
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

/// A slide to process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlideInput {
    /// The file to open.
    pub path: PathBuf,
    /// Every file that belongs to the slide (for hashing), sorted.
    pub members: Vec<PathBuf>,
    /// DICOM Series Instance UID, for grouped DICOM slides.
    pub series_uid: Option<String>,
}

impl SlideInput {
    fn single(path: PathBuf) -> Self {
        Self {
            members: members_of(&path),
            path,
            series_uid: None,
        }
    }
}

/// A path that was not used, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedInput {
    pub path: PathBuf,
    pub reason: String,
}

/// Result of [`discover_slides`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Discovery {
    pub slides: Vec<SlideInput>,
    /// Files deliberately not used (non-slide TIFFs, extra DICOM instances).
    pub skipped: Vec<SkippedInput>,
    /// Inputs that could not be resolved (missing, unsupported, unreadable).
    pub errors: Vec<(PathBuf, String)>,
}

/// Expands inputs into slides (deduplicated by canonical path, in input
/// order; directory entries sorted by name).
pub fn discover_slides(inputs: &[PathBuf]) -> Discovery {
    let mut discovery = Discovery::default();
    let mut files: Vec<(PathBuf, bool)> = Vec::new(); // (path, explicit)
    for input in inputs {
        let meta = match std::fs::metadata(input) {
            Ok(m) => m,
            Err(_) => {
                discovery
                    .errors
                    .push((input.clone(), "path does not exist".into()));
                continue;
            }
        };
        if meta.is_file() {
            if is_supported_slide_extension(input) {
                files.push((input.clone(), true));
            } else {
                discovery.errors.push((
                    input.clone(),
                    "file extension is not a supported slide format".into(),
                ));
            }
        } else if meta.is_dir() {
            let mut visited = HashSet::new();
            collect_recursive(input, 0, &mut visited, &mut files, &mut discovery);
        } else {
            discovery
                .errors
                .push((input.clone(), "not a file or directory".into()));
        }
    }

    // Deduplicate by canonical path, preserving first-seen order.
    let mut seen = HashSet::new();
    files.retain(|(p, _)| seen.insert(std::fs::canonicalize(p).unwrap_or_else(|_| p.clone())));

    // Group DICOM instances by (directory, series UID); a group takes the
    // position of its first instance.
    enum Entry {
        Slide(SlideInput),
        DicomSeries(PathBuf, String),
    }
    let mut dicom_groups: BTreeMap<(PathBuf, String), Vec<PathBuf>> = BTreeMap::new();
    let mut order: Vec<Entry> = Vec::new();
    for (path, explicit) in files {
        let ext = extension(&path).unwrap_or_default();
        if ext == "dcm" {
            match dicom_series_uid(&path) {
                Some(uid) => {
                    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
                    let group = dicom_groups.entry((dir.clone(), uid.clone())).or_default();
                    if group.is_empty() {
                        order.push(Entry::DicomSeries(dir, uid));
                    }
                    group.push(path);
                }
                None => order.push(Entry::Slide(SlideInput::single(path))),
            }
            continue;
        }
        if !explicit
            && matches!(ext.as_str(), "tif" | "tiff")
            && let Some(reason) = non_slide_tiff_reason(&path)
        {
            discovery.skipped.push(SkippedInput { path, reason });
            continue;
        }
        order.push(Entry::Slide(SlideInput::single(path)));
    }
    for entry in order {
        match entry {
            Entry::Slide(slide) => discovery.slides.push(slide),
            Entry::DicomSeries(dir, uid) => {
                let mut group = dicom_groups.remove(&(dir, uid.clone())).unwrap_or_default();
                group.sort();
                let path = group[0].clone();
                for extra in &group[1..] {
                    discovery.skipped.push(SkippedInput {
                        path: extra.clone(),
                        reason: format!(
                            "DICOM instance of series {uid}, extracted via {}",
                            path.display()
                        ),
                    });
                }
                discovery.slides.push(SlideInput {
                    path,
                    members: group,
                    series_uid: Some(uid),
                });
            }
        }
    }
    debug!(
        "Expanded {} inputs into {} slide(s), {} skipped",
        inputs.len(),
        discovery.slides.len(),
        discovery.skipped.len()
    );
    discovery
}

/// Pre-1.0 entry point: slide paths and input errors (skipped files are
/// reported as errors).
pub fn expand_inputs(inputs: &[PathBuf]) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let discovery = discover_slides(inputs);
    let mut errors = discovery.errors;
    errors.extend(discovery.skipped.into_iter().map(|s| (s.path, s.reason)));
    (
        discovery.slides.into_iter().map(|s| s.path).collect(),
        errors,
    )
}

fn collect_recursive(
    dir: &Path,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    out: &mut Vec<(PathBuf, bool)>,
    discovery: &mut Discovery,
) {
    if depth > MAX_DEPTH {
        discovery.errors.push((
            dir.to_path_buf(),
            format!("directory nesting deeper than {MAX_DEPTH} levels"),
        ));
        return;
    }
    let canonical = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if !visited.insert(canonical) {
        debug!("Skipping already visited directory {}", dir.display());
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            discovery
                .errors
                .push((dir.to_path_buf(), format!("cannot read directory: {err}")));
            return;
        }
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        // Follows symlinks; loops are caught by `visited`.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            // MIRAX data directories hold no slides of their own.
            if is_mirax_data_dir(&path) {
                continue;
            }
            collect_recursive(&path, depth + 1, visited, out, discovery);
        } else if meta.is_file() && is_supported_slide_extension(&path) {
            out.push((path, false));
        }
    }
}

/// `case/` next to `case.mrxs` holds MIRAX data files, not slides.
fn is_mirax_data_dir(dir: &Path) -> bool {
    let mut index = dir.as_os_str().to_os_string();
    index.push(".mrxs");
    Path::new(&index).is_file()
}

/// Every file of a (possibly multi-file) slide, sorted.
pub fn members_of(path: &Path) -> Vec<PathBuf> {
    let mut members = vec![path.to_path_buf()];
    match extension(path).as_deref() {
        Some("mrxs") => {
            if let Some(stem) = path.file_stem() {
                let dir = path.with_file_name(stem);
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    members.extend(
                        entries
                            .filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| p.is_file()),
                    );
                }
            }
        }
        Some("vms") | Some("vmu") => {
            if let (Ok(text), Some(dir)) = (std::fs::read_to_string(path), path.parent()) {
                for line in text.lines() {
                    let Some((key, value)) = line.split_once('=') else {
                        continue;
                    };
                    let key = key.trim();
                    if key.starts_with("ImageFile") || key == "MapFile" || key == "OptimisationFile"
                    {
                        let file = dir.join(value.trim());
                        if file.is_file() {
                            members.push(file);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    members.sort();
    members.dedup();
    members
}

/// Reads the DICOM Series Instance UID (0020,000E) from the first 1 MiB of
/// an explicit-VR file (`DICM` preamble). Returns `None` for anything else.
pub fn dicom_series_uid(path: &Path) -> Option<String> {
    let mut buf = Vec::with_capacity(1 << 16);
    File::open(path)
        .ok()?
        .take(1 << 20)
        .read_to_end(&mut buf)
        .ok()?;
    if buf.len() < 132 || &buf[128..132] != b"DICM" {
        return None;
    }
    // Tag (0020,000E) little-endian + VR "UI" + 16-bit length.
    let pattern = [0x20, 0x00, 0x0E, 0x00, b'U', b'I'];
    let start = buf[132..]
        .windows(pattern.len())
        .position(|w| w == pattern)?
        + 132;
    let len_at = start + pattern.len();
    let len = u16::from_le_bytes([*buf.get(len_at)?, *buf.get(len_at + 1)?]) as usize;
    let value = buf.get(len_at + 2..len_at + 2 + len)?;
    let uid: String = value
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as char)
        .collect::<String>()
        .trim()
        .to_string();
    (!uid.is_empty() && uid.len() <= 64 && uid.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .then_some(uid)
}

/// Why a TIFF found by a directory scan is not a brightfield slide, if it
/// clearly is not: one sample per pixel (masks, label images) or no tiles
/// (OpenSlide's TIFF drivers need tiled images).
pub fn non_slide_tiff_reason(path: &Path) -> Option<String> {
    let mut buf = Vec::with_capacity(1 << 16);
    File::open(path)
        .ok()?
        .take(1 << 20)
        .read_to_end(&mut buf)
        .ok()?;
    let info = first_ifd(&buf)?;
    if info.samples_per_pixel == 1 && info.photometric != 6 {
        return Some("single-channel TIFF (a mask or label image, not a brightfield slide)".into());
    }
    if !info.tiled {
        return Some("untiled TIFF (OpenSlide can only open tiled slides)".into());
    }
    None
}

struct IfdInfo {
    samples_per_pixel: u64,
    photometric: u64,
    tiled: bool,
}

fn first_ifd(buf: &[u8]) -> Option<IfdInfo> {
    let little = match buf.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u64> {
        let b = buf.get(at..at + 2)?;
        Some(if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        } as u64)
    };
    let u32_at = |at: usize| -> Option<u64> {
        let b: [u8; 4] = buf.get(at..at + 4)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        } as u64)
    };
    let u64_at = |at: usize| -> Option<u64> {
        let b: [u8; 8] = buf.get(at..at + 8)?.try_into().ok()?;
        Some(if little {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        })
    };
    let big = match u16_at(2)? {
        42 => false,
        43 => true,
        _ => return None,
    };
    let ifd = if big { u64_at(8)? } else { u32_at(4)? } as usize;
    let (count, entry_size, first) = if big {
        (u64_at(ifd)? as usize, 20usize, ifd + 8)
    } else {
        (u16_at(ifd)? as usize, 12usize, ifd + 2)
    };
    let mut info = IfdInfo {
        samples_per_pixel: 1,
        photometric: 2,
        tiled: false,
    };
    for i in 0..count.min(512) {
        let at = first + i * entry_size;
        let tag = u16_at(at)?;
        let ty = u16_at(at + 2)?;
        let value_at = at + if big { 12 } else { 8 };
        let value = match ty {
            3 => u16_at(value_at)?,
            4 => u32_at(value_at)?,
            16 => u64_at(value_at)?,
            _ => continue,
        };
        match tag {
            262 => info.photometric = value,
            277 => info.samples_per_pixel = value,
            322 => info.tiled = true,
            _ => {}
        }
    }
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_tiled_tiff;
    use std::fs;

    #[test]
    fn test_is_supported_extension() {
        assert!(is_supported_slide_extension(Path::new("slide.svs")));
        assert!(is_supported_slide_extension(Path::new("slide.SVS")));
        assert!(is_supported_slide_extension(Path::new("slide.ndpi")));
        assert!(!is_supported_slide_extension(Path::new("notes.txt")));
        assert!(!is_supported_slide_extension(Path::new("noext")));
    }

    #[test]
    fn missing_and_unsupported_inputs_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("notes.txt");
        fs::write(&txt, "hello").unwrap();
        let d = discover_slides(&[PathBuf::from("/nonexistent/slide.svs"), txt]);
        assert!(d.slides.is_empty());
        assert_eq!(d.errors.len(), 2);
    }

    #[test]
    fn directories_are_searched_recursively_skipping_hidden_entries() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("a.svs"), "").unwrap();
        fs::write(sub.join("b.ndpi"), "").unwrap();
        fs::write(sub.join("c.txt"), "").unwrap();
        fs::write(sub.join("._a.svs"), "").unwrap(); // AppleDouble junk
        let d = discover_slides(&[dir.path().to_path_buf()]);
        let names: Vec<_> = d
            .slides
            .iter()
            .map(|s| s.path.file_name().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["a.svs", "b.ndpi"]);
        let (paths, errors) = expand_inputs(&[dir.path().to_path_buf()]);
        assert_eq!(paths.len(), 2);
        assert!(errors.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_loops_terminate() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("a.svs"), "").unwrap();
        std::os::unix::fs::symlink(dir.path(), sub.join("loop")).unwrap();
        let d = discover_slides(&[dir.path().to_path_buf()]);
        assert_eq!(d.slides.len(), 1);
    }

    fn write_dicom(path: &Path, uid: &str) {
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        // (0008,0060) Modality "SM", then (0020,000E) Series Instance UID.
        bytes.extend_from_slice(&[0x08, 0x00, 0x60, 0x00, b'C', b'S', 2, 0, b'S', b'M']);
        let mut value = uid.as_bytes().to_vec();
        if value.len() % 2 == 1 {
            value.push(0);
        }
        bytes.extend_from_slice(&[0x20, 0x00, 0x0E, 0x00, b'U', b'I']);
        bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&value);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn dicom_instances_of_one_series_are_one_slide() {
        let dir = tempfile::tempdir().unwrap();
        for (name, uid) in [
            ("i1.dcm", "1.2.3"),
            ("i2.dcm", "1.2.3"),
            ("i3.dcm", "1.2.3"),
            ("j1.dcm", "1.2.44"),
        ] {
            write_dicom(&dir.path().join(name), uid);
        }
        assert_eq!(
            dicom_series_uid(&dir.path().join("i2.dcm")).as_deref(),
            Some("1.2.3")
        );
        let d = discover_slides(&[dir.path().to_path_buf()]);
        assert_eq!(d.slides.len(), 2, "{:?}", d.slides);
        let series = d
            .slides
            .iter()
            .find(|s| s.series_uid.as_deref() == Some("1.2.3"))
            .unwrap();
        assert_eq!(series.members.len(), 3);
        assert!(series.path.ends_with("i1.dcm"));
        assert_eq!(d.skipped.len(), 2);
    }

    #[test]
    fn mask_tiffs_are_skipped_in_directory_scans_only() {
        let dir = tempfile::tempdir().unwrap();
        let slide = dir.path().join("slide.tif");
        write_tiled_tiff(&slide, 64, 64, 32, |_, _| [200, 100, 150]).unwrap();
        // A single-channel TIFF: patch SamplesPerPixel (tag 277) to 1.
        let mask = dir.path().join("slide_mask.tif");
        let mut bytes = fs::read(&slide).unwrap();
        let ifd = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let count = u16::from_le_bytes([bytes[ifd], bytes[ifd + 1]]) as usize;
        for i in 0..count {
            let at = ifd + 2 + i * 12;
            if u16::from_le_bytes([bytes[at], bytes[at + 1]]) == 277 {
                bytes[at + 8] = 1;
            }
        }
        fs::write(&mask, bytes).unwrap();
        assert!(non_slide_tiff_reason(&slide).is_none());
        assert!(
            non_slide_tiff_reason(&mask)
                .unwrap()
                .contains("single-channel")
        );
        let d = discover_slides(&[dir.path().to_path_buf()]);
        assert_eq!(d.slides.len(), 1);
        assert_eq!(d.skipped.len(), 1);
        // Explicitly named files are trusted.
        assert_eq!(discover_slides(&[mask]).slides.len(), 1);
    }

    #[test]
    fn mirax_members_include_the_data_directory() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("case.mrxs");
        fs::write(&index, "").unwrap();
        let data = dir.path().join("case");
        fs::create_dir(&data).unwrap();
        fs::write(data.join("Slidedat.ini"), "").unwrap();
        fs::write(data.join("Data0000.dat"), "").unwrap();
        let members = members_of(&index);
        assert_eq!(members.len(), 3);
        let d = discover_slides(&[dir.path().to_path_buf()]);
        assert_eq!(d.slides.len(), 1);
    }
}
