//! Named views of a slide (AUDIT §14.1), kept in `<data>/bookmarks.toml`
//! and keyed by slide identity, so they survive renames and moves. The file
//! is plain TOML: copy it to share bookmarks.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Bookmarks kept per slide.
pub(crate) const MAX_PER_SLIDE: usize = 50;
/// Longest bookmark name, in characters.
pub(crate) const MAX_NAME_CHARS: usize = 120;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Bookmark {
    pub name: String,
    pub center_x: f64,
    pub center_y: f64,
    /// Screen pixels per level-0 pixel.
    pub zoom: f64,
    /// Clockwise view rotation in degrees (a multiple of 90); absent in
    /// bookmarks made before views could turn.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rotation: u32,
    /// Mirrored left to right (after the rotation).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flipped: bool,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

impl Bookmark {
    /// How the bookmarked view is turned.
    pub fn orientation(&self) -> eov_common::viewport::Orientation {
        eov_common::viewport::Orientation::from_degrees(f64::from(self.rotation), self.flipped)
            .unwrap_or_default()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default)]
    slide: BTreeMap<String, Slide>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Slide {
    /// File name when the first bookmark was added, for people.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    name: String,
    #[serde(default)]
    bookmark: Vec<Bookmark>,
}

fn store_path() -> PathBuf {
    crate::paths::data_dir().join("bookmarks.toml")
}

fn read(path: &Path) -> Store {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

fn write(path: &Path, store: &Store) -> Result<(), String> {
    let text = toml::to_string(store).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text).map_err(|error| error.to_string())?;
    fs::rename(&temp, path).map_err(|error| error.to_string())
}

/// The bookmarks of the slide with identity `key`, oldest first.
pub(crate) fn list(key: &str) -> Vec<Bookmark> {
    list_in(&store_path(), key)
}

fn list_in(path: &Path, key: &str) -> Vec<Bookmark> {
    read(path)
        .slide
        .remove(key)
        .map(|slide| slide.bookmark)
        .unwrap_or_default()
}

/// Adds a bookmark; returns its name.
pub(crate) fn add(key: &str, file_name: &str, bookmark: Bookmark) -> Result<String, String> {
    add_in(&store_path(), key, file_name, bookmark)
}

fn add_in(path: &Path, key: &str, file_name: &str, bookmark: Bookmark) -> Result<String, String> {
    if ![bookmark.center_x, bookmark.center_y, bookmark.zoom]
        .iter()
        .all(|v| v.is_finite())
        || bookmark.zoom <= 0.0
    {
        return Err("the view is not valid".to_string());
    }
    let mut store = read(path);
    let slide = store.slide.entry(key.to_string()).or_default();
    if slide.name.is_empty() {
        slide.name = file_name.to_string();
    }
    if slide.bookmark.len() >= MAX_PER_SLIDE {
        return Err(format!(
            "a slide can have at most {MAX_PER_SLIDE} bookmarks"
        ));
    }
    let name = bookmark.name.clone();
    slide.bookmark.push(bookmark);
    write(path, &store)?;
    Ok(name)
}

/// Removes the bookmark at `index` of the slide's list.
pub(crate) fn remove(key: &str, index: usize) -> Result<(), String> {
    remove_in(&store_path(), key, index)
}

fn remove_in(path: &Path, key: &str, index: usize) -> Result<(), String> {
    let mut store = read(path);
    let Some(slide) = store.slide.get_mut(key) else {
        return Err("the slide has no bookmarks".to_string());
    };
    if index >= slide.bookmark.len() {
        return Err("no such bookmark".to_string());
    }
    slide.bookmark.remove(index);
    if slide.bookmark.is_empty() {
        store.slide.remove(key);
    }
    write(path, &store)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bookmark(name: &str) -> Bookmark {
        Bookmark {
            name: name.into(),
            center_x: 10.0,
            center_y: 20.0,
            zoom: 0.5,
            rotation: 0,
            flipped: false,
        }
    }

    #[test]
    fn bookmarks_are_kept_per_slide() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bookmarks.toml");
        add_in(&path, "slide-a", "a.svs", bookmark("Tumour edge")).unwrap();
        add_in(&path, "slide-a", "a.svs", bookmark("Margin")).unwrap();
        add_in(&path, "slide-b", "b.svs", bookmark("Mitosis")).unwrap();
        let names: Vec<_> = list_in(&path, "slide-a")
            .into_iter()
            .map(|b| b.name)
            .collect();
        assert_eq!(names, ["Tumour edge", "Margin"]);
        remove_in(&path, "slide-a", 0).unwrap();
        assert_eq!(list_in(&path, "slide-a")[0].name, "Margin");
        assert!(remove_in(&path, "slide-a", 5).is_err());
        let invalid = Bookmark {
            zoom: f64::NAN,
            ..bookmark("bad")
        };
        assert!(add_in(&path, "slide-a", "a.svs", invalid).is_err());
        assert_eq!(list_in(&path, "slide-b").len(), 1);
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;

    #[test]
    fn bookmarks_keep_their_turn_and_old_ones_read() {
        let old = "[slide.abc]\nname = \"a.svs\"\n[[slide.abc.bookmark]]\nname = \"x\"\ncenter_x = 1.0\ncenter_y = 2.0\nzoom = 0.5\n";
        let store: Store = toml::from_str(old).unwrap();
        let bookmark = &store.slide["abc"].bookmark[0];
        assert_eq!(
            bookmark.orientation(),
            eov_common::viewport::Orientation::IDENTITY
        );
        let turned = Bookmark {
            rotation: 270,
            flipped: true,
            ..bookmark.clone()
        };
        let text = toml::to_string(&turned).unwrap();
        let back: Bookmark = toml::from_str(&text).unwrap();
        assert_eq!(back, turned);
        assert_eq!(
            back.orientation(),
            eov_common::viewport::Orientation::new(3, true)
        );
        // An upright bookmark is written as before.
        assert!(!toml::to_string(bookmark).unwrap().contains("rotation"));
    }
}
