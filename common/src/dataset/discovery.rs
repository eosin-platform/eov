//! Input path expansion and supported-slide discovery.

use std::path::{Path, PathBuf};
use tracing::debug;

/// File extensions that EOV can open as whole-slide images (via OpenSlide).
/// Keep in sync with the file-picker filter in `app/src/callbacks.rs`.
pub const SUPPORTED_SLIDE_EXTENSIONS: &[&str] = &[
    "svs", "tif", "tiff", "dcm", "ndpi", "vms", "vmu", "scn", "mrxs", "svslide", "bif", "czi",
];

/// Returns `true` if the file extension is a known slide format.
pub fn is_supported_slide_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            let lower = ext.to_ascii_lowercase();
            SUPPORTED_SLIDE_EXTENSIONS.iter().any(|&s| s == lower)
        })
}

/// Expand a list of input paths into a deduplicated list of slide file paths.
///
/// Each input may be:
/// * a slide file — included directly
/// * a directory — recursively scanned for files with supported extensions
///
/// Returns `(slides, errors)` where `errors` contains paths that could not be
/// resolved (e.g. non-existent paths). Duplicate slides are removed by
/// canonical path when possible.
pub fn expand_inputs(inputs: &[PathBuf]) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let mut slides: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<(PathBuf, String)> = Vec::new();

    for input in inputs {
        if !input.exists() {
            errors.push((input.clone(), "path does not exist".into()));
            continue;
        }

        if input.is_file() {
            if is_supported_slide_extension(input) {
                slides.push(input.clone());
            } else {
                errors.push((
                    input.clone(),
                    "file extension is not a supported slide format".into(),
                ));
            }
        } else if input.is_dir() {
            collect_slides_recursive(input, &mut slides);
        } else {
            errors.push((input.clone(), "not a file or directory".into()));
        }
    }

    // Deduplicate by canonical path, preserving first-seen order.
    deduplicate_by_canonical(&mut slides);

    debug!("Expanded {} inputs into {} slide(s)", inputs.len(), slides.len());

    (slides, errors)
}

/// Recursively walk a directory and collect files with supported slide extensions.
fn collect_slides_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            debug!("Cannot read directory {}: {err}", dir.display());
            return;
        }
    };

    // Sort entries for deterministic ordering.
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_slides_recursive(&path, out);
        } else if path.is_file() && is_supported_slide_extension(&path) {
            out.push(path);
        }
    }
}

/// Remove duplicate paths by comparing canonical forms.
fn deduplicate_by_canonical(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::new();
    paths.retain(|p| {
        let key = std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.clone());
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_is_supported_extension() {
        assert!(is_supported_slide_extension(Path::new("slide.svs")));
        assert!(is_supported_slide_extension(Path::new("slide.SVS")));
        assert!(is_supported_slide_extension(Path::new("slide.tif")));
        assert!(is_supported_slide_extension(Path::new("slide.ndpi")));
        assert!(!is_supported_slide_extension(Path::new("notes.txt")));
        assert!(!is_supported_slide_extension(Path::new("image.png")));
        assert!(!is_supported_slide_extension(Path::new("noext")));
    }

    #[test]
    fn test_expand_nonexistent_path() {
        let (slides, errors) = expand_inputs(&[PathBuf::from("/nonexistent/slide.svs")]);
        assert!(slides.is_empty());
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_expand_unsupported_extension() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("notes.txt");
        fs::write(&txt, "hello").unwrap();

        let (slides, errors) = expand_inputs(&[txt]);
        assert!(slides.is_empty());
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_expand_directory_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        // Create dummy files with slide extensions (they won't be openable,
        // but expand_inputs only checks extensions).
        fs::write(dir.path().join("a.svs"), "").unwrap();
        fs::write(sub.join("b.tif"), "").unwrap();
        fs::write(sub.join("c.txt"), "").unwrap();

        let (slides, errors) = expand_inputs(&[dir.path().to_path_buf()]);
        assert_eq!(slides.len(), 2);
        assert!(errors.is_empty());
    }
}
