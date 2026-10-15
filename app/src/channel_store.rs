//! How the channels of multichannel slides are displayed, kept in
//! `<data>/channels/<slide>.json` and keyed by slide identity, so a slide
//! opens with the colours and windows it was last shown with, under any
//! name.

use eov_common::channels::{ChannelDisplay, ChannelSet, ChannelSettings};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SavedChannel {
    /// For people reading the file; channels match by position.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    name: String,
    visible: bool,
    /// `#RRGGBB`.
    color: String,
    min: f64,
    max: f64,
    #[serde(default = "one")]
    gamma: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Saved {
    #[serde(default)]
    version: u32,
    /// File name when last saved, for people reading the file.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    name: String,
    channels: Vec<SavedChannel>,
}

fn store_dir() -> PathBuf {
    crate::paths::data_dir().join("channels")
}

/// A file name for a slide identity key (`eov-slide-1:3f2a…`).
fn file_name(key: &str) -> String {
    let safe: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{safe}.json")
}

fn hex(color: [u8; 3]) -> String {
    format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2])
}

fn parse_hex(text: &str) -> Option<[u8; 3]> {
    let digits = text.trim().strip_prefix('#')?;
    if digits.len() != 6 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let channel = |i: usize| u8::from_str_radix(&digits[i..i + 2], 16).ok();
    Some([channel(0)?, channel(2)?, channel(4)?])
}

/// The saved display of slide `key`, if one was saved for a slide with as
/// many channels as `set`.
pub(crate) fn load(key: &str, set: &ChannelSet) -> Option<ChannelDisplay> {
    load_in(&store_dir(), key, set)
}

fn load_in(dir: &Path, key: &str, set: &ChannelSet) -> Option<ChannelDisplay> {
    let text = fs::read_to_string(dir.join(file_name(key))).ok()?;
    let saved: Saved = match serde_json::from_str(&text) {
        Ok(saved) => saved,
        Err(error) => {
            tracing::warn!("Ignoring unreadable channel settings: {error}");
            return None;
        }
    };
    if saved.channels.len() != set.len() {
        return None;
    }
    let channels = saved
        .channels
        .iter()
        .map(|channel| {
            Some(ChannelSettings {
                visible: channel.visible,
                color: parse_hex(&channel.color)?,
                min: channel.min,
                max: channel.max,
                gamma: channel.gamma,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    ChannelDisplay { channels }.validated(set).ok()
}

/// Saves the display of slide `key` (`name`: its file name).
pub(crate) fn save(
    key: &str,
    name: &str,
    display: &ChannelDisplay,
    set: &ChannelSet,
) -> Result<(), String> {
    save_in(&store_dir(), key, name, display, set)
}

fn save_in(
    dir: &Path,
    key: &str,
    name: &str,
    display: &ChannelDisplay,
    set: &ChannelSet,
) -> Result<(), String> {
    let saved = Saved {
        version: VERSION,
        name: name.to_string(),
        channels: display
            .channels
            .iter()
            .zip(&set.channels)
            .map(|(settings, info)| SavedChannel {
                name: info.name.clone(),
                visible: settings.visible,
                color: hex(settings.color),
                min: settings.min,
                max: settings.max,
                gamma: settings.gamma,
            })
            .collect(),
    };
    let text = serde_json::to_string_pretty(&saved).map_err(|e| e.to_string())?;
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(file_name(key));
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, text).map_err(|e| e.to_string())?;
    fs::rename(&temp, &path).map_err(|e| e.to_string())
}

/// Forgets the display of slide `key`, so it opens with the defaults.
pub(crate) fn forget(key: &str) -> Result<(), String> {
    forget_in(&store_dir(), key)
}

fn forget_in(dir: &Path, key: &str) -> Result<(), String> {
    match fs::remove_file(dir.join(file_name(key))) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_common::channels::{ChannelInfo, ChannelStats, SampleType};

    fn set(names: &[&str]) -> ChannelSet {
        ChannelSet {
            channels: names
                .iter()
                .enumerate()
                .map(|(index, name)| ChannelInfo {
                    index,
                    name: name.to_string(),
                    color: None,
                    fluor: None,
                    emission_nm: None,
                    excitation_nm: None,
                    stats: ChannelStats::default(),
                })
                .collect(),
            sample_type: SampleType::U16,
            significant_bits: None,
            size_z: 1,
            size_t: 1,
        }
    }

    #[test]
    fn displays_persist_by_identity_and_must_match_the_channels() {
        let dir = tempfile::tempdir().unwrap();
        let channels = set(&["DAPI", "CD8"]);
        assert_eq!(load_in(dir.path(), "eov-slide-1:abc", &channels), None);
        let mut display = channels.default_display();
        display.channels[0].color = [1, 2, 3];
        display.channels[1].visible = false;
        display.channels[1].min = 120.0;
        display.channels[1].max = 3000.0;
        display.channels[1].gamma = 1.5;
        save_in(
            dir.path(),
            "eov-slide-1:abc",
            "a.ome.tif",
            &display,
            &channels,
        )
        .unwrap();
        let text = fs::read_to_string(dir.path().join("eov-slide-1_abc.json")).unwrap();
        assert!(
            text.contains("\"#010203\"") && text.contains("CD8"),
            "{text}"
        );
        assert_eq!(
            load_in(dir.path(), "eov-slide-1:abc", &channels),
            Some(display)
        );
        // Another channel count: not this slide's settings.
        assert_eq!(
            load_in(dir.path(), "eov-slide-1:abc", &set(&["a", "b", "c"])),
            None
        );
        forget_in(dir.path(), "eov-slide-1:abc").unwrap();
        forget_in(dir.path(), "eov-slide-1:abc").unwrap();
        assert_eq!(load_in(dir.path(), "eov-slide-1:abc", &channels), None);
    }

    #[test]
    fn damaged_files_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let channels = set(&["a"]);
        fs::write(dir.path().join("k.json"), "{not json").unwrap();
        assert_eq!(load_in(dir.path(), "k", &channels), None);
        fs::write(
            dir.path().join("k.json"),
            r##"{"channels":[{"visible":true,"color":"red","min":0,"max":1}]}"##,
        )
        .unwrap();
        assert_eq!(load_in(dir.path(), "k", &channels), None);
        // A window that makes no sense is repaired, not refused.
        fs::write(
            dir.path().join("k.json"),
            r##"{"channels":[{"visible":true,"color":"#00FF00","min":5,"max":1}]}"##,
        )
        .unwrap();
        let display = load_in(dir.path(), "k", &channels).unwrap();
        assert!(display.channels[0].max > display.channels[0].min);
        assert_eq!(display.channels[0].gamma, 1.0);
        assert_eq!(parse_hex("#0a0B0c"), Some([10, 11, 12]));
        assert_eq!(parse_hex("0a0b0c"), None);
    }
}
