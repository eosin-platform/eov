//! Cases in the series bar (AUDIT §14.1): the Case order groups a folder's
//! slides by the case (accession) they belong to, read from their names.
//!
//! `series_case_pattern` in `config.toml` is a regular expression whose
//! first capture group (or whole match) is a slide's case, for example
//! `"^(S\d{2}-\d+)"` for `S21-12345 A1 HE.svs`. Names it does not match, and
//! every name without a pattern, use the default: a TCGA barcode gives its
//! participant (`TCGA-A1-A0SB`), and other names lose their last part
//! (`C3L-00011-21.svs` is case `C3L-00011`).

use regex::Regex;
use std::sync::OnceLock;

/// The configured pattern, compiled once (an invalid one is reported and
/// ignored).
fn pattern() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| {
            let text = crate::config::load_series_case_pattern()?;
            match Regex::new(&text) {
                Ok(regex) => Some(regex),
                Err(error) => {
                    tracing::warn!(
                        "series_case_pattern is not a valid regular expression: {error}"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// The name without its slide extension (`.svs`, `.ome.tif`, ...).
fn stem(filename: &str) -> &str {
    let stem = filename.rsplit_once('.').map_or(filename, |(stem, _)| stem);
    stem.strip_suffix(".ome")
        .or_else(|| stem.strip_suffix(".OME"))
        .unwrap_or(stem)
}

fn is_separator(c: char) -> bool {
    matches!(c, '-' | '_' | ' ' | '.')
}

/// The default case of a file name.
fn default_case(filename: &str) -> String {
    let stem = stem(filename);
    if stem.starts_with("TCGA-") {
        let participant: Vec<&str> = stem.splitn(4, '-').take(3).collect();
        if participant.len() == 3 {
            return participant.join("-");
        }
    }
    match stem.trim_end_matches(is_separator).rfind(is_separator) {
        Some(end) => stem[..end].trim_end_matches(is_separator).to_string(),
        None => stem.to_string(),
    }
}

/// The case `filename` belongs to, with `pattern` when it matches.
pub(crate) fn case_with(filename: &str, pattern: Option<&Regex>) -> String {
    pattern
        .and_then(|pattern| pattern.captures(filename))
        .and_then(|captures| captures.get(1).or_else(|| captures.get(0)))
        .map(|found| found.as_str().trim().to_string())
        .filter(|case| !case.is_empty())
        .unwrap_or_else(|| default_case(filename))
}

/// The case `filename` belongs to.
pub(crate) fn case_of(filename: &str) -> String {
    case_with(filename, pattern())
}

/// The name shown under a case: what follows it, or the whole name.
pub(crate) fn name_within(filename: &str, case: &str) -> String {
    filename
        .strip_prefix(case)
        // Nothing but the extension left: the whole name.
        .filter(|rest| !rest.starts_with('.'))
        .map(|rest| rest.trim_start_matches(is_separator))
        .filter(|rest| !rest.is_empty())
        .unwrap_or(filename)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_come_from_common_naming_schemes() {
        assert_eq!(default_case("C3L-00011-21.svs"), "C3L-00011");
        assert_eq!(default_case("TCGA-A1-A0SB-01Z-00-DX1.svs"), "TCGA-A1-A0SB");
        assert_eq!(default_case("case12_block3_HE.ome.tiff"), "case12_block3");
        assert_eq!(default_case("single.svs"), "single");
        assert_eq!(default_case("trailing-.svs"), "trailing-");
        let accession = Regex::new(r"^(S\d{2}-\d+)").unwrap();
        assert_eq!(
            case_with("S21-12345 A1 HE.svs", Some(&accession)),
            "S21-12345"
        );
        assert_eq!(
            case_with("other-name-1.svs", Some(&accession)),
            "other-name",
            "names the pattern misses use the default"
        );
        assert_eq!(name_within("C3L-00011-21.svs", "C3L-00011"), "21.svs");
        assert_eq!(name_within("single.svs", "single"), "single.svs");
    }
}
