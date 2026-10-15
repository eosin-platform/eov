//! Session restore (AUDIT §14.1): the panes, tabs and views of the last
//! session, in `<eov home>/session.toml`.
//!
//! eov saves the session when it exits (and every minute, so a crash loses
//! little) and reopens it on a start without files on the command line,
//! unless `restore_session = false` in `config.toml`. Slides that are gone
//! simply fail to open; nothing else is checked on disk at start-up.

use crate::state::{AppState, PaneId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
struct Session {
    #[serde(default)]
    focused_pane: usize,
    #[serde(default, rename = "pane")]
    panes: Vec<PaneSession>,
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
struct PaneSession {
    #[serde(default, rename = "tab")]
    tabs: Vec<TabSession>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct TabSession {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    /// The path in the platform's encoding, as hex, when it is not UTF-8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    os_path: Option<String>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    /// Clockwise view rotation in degrees (a multiple of 90).
    #[serde(default, skip_serializing_if = "is_zero")]
    rotation: u32,
    /// Mirrored left to right (after the rotation).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    flipped: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    active: bool,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

impl TabSession {
    fn path(&self) -> Option<PathBuf> {
        let path = match (&self.path, &self.os_path) {
            (_, Some(hex)) => crate::recent::decode_os(hex)?,
            (Some(path), None) => PathBuf::from(path),
            (None, None) => return None,
        };
        // Older versions stored `\\?\` paths on Windows (SH-15).
        Some(crate::paths::simplified(&path))
    }
}

/// Where a slide reopens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SavedView {
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub orientation: eov_common::viewport::Orientation,
}

fn session_path() -> PathBuf {
    crate::paths::session_file()
}

fn capture(state: &AppState) -> Session {
    let panes = (0..state.panes.len())
        .map(|index| {
            let pane = PaneId(index);
            let active = state.active_tab_id_for_pane(pane);
            let tabs = state
                .tabs_for_pane(pane)
                .iter()
                .filter(|tab| !state.is_home_tab(**tab))
                .filter_map(|&tab| {
                    let file = state.get_file(state.resolve_tab_file_id(tab))?;
                    let viewport = &state
                        .saved_tab_view(tab)
                        .filter(|_| active != Some(tab))
                        .or_else(|| file.pane_state(pane))?
                        .viewport
                        .viewport;
                    let (path, os_path) = match file.path.to_str() {
                        Some(path) => (Some(path.to_string()), None),
                        None => (None, Some(crate::recent::encode_os(&file.path))),
                    };
                    Some(TabSession {
                        path,
                        os_path,
                        center_x: viewport.center.x,
                        center_y: viewport.center.y,
                        zoom: viewport.zoom,
                        rotation: viewport.orientation.degrees(),
                        flipped: viewport.orientation.flipped(),
                        active: active == Some(tab),
                    })
                })
                .collect();
            PaneSession { tabs }
        })
        .filter(|pane: &PaneSession| !pane.tabs.is_empty())
        .collect();
    Session {
        focused_pane: state.focused_pane.0,
        panes,
    }
}

/// Saves the open panes, tabs and views.
pub(crate) fn save(state: &AppState) {
    if let Err(error) = write(&session_path(), &capture(state)) {
        tracing::warn!("Could not save the session: {error}");
    }
}

fn write(path: &Path, session: &Session) -> std::io::Result<()> {
    let text = toml::to_string(session).map_err(std::io::Error::other)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text)?;
    fs::rename(&temp, path)
}

fn read(path: &Path) -> Option<Session> {
    let text = fs::read_to_string(path).ok()?;
    match toml::from_str(&text) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!("Ignoring unreadable {}: {error}", path.display());
            None
        }
    }
}

/// Reopens the last session's slides in their panes and views. Returns
/// whether anything was reopened.
pub(crate) fn restore(
    ui: &crate::AppWindow,
    state: &Arc<RwLock<AppState>>,
    tile_cache: &Arc<eov_common::TileCache>,
    render_timer: &std::rc::Rc<slint::Timer>,
) -> bool {
    if !crate::config::load_restore_session() {
        return false;
    }
    let Some(session) = read(&session_path()) else {
        return false;
    };
    if session.panes.is_empty() {
        return false;
    }
    crate::prepare_launch_panes(ui, state, session.panes.len());
    // Slides open in parallel; the placements keep each pane's tabs in
    // their order, with the active one selected.
    let batch = crate::file_ops::new_batch();
    for (index, pane) in session.panes.iter().enumerate() {
        state.write().set_focused_pane(PaneId(index));
        let tabs = pane.tabs.iter().filter_map(|tab| Some((tab, tab.path()?)));
        for (position, (tab, path)) in tabs.enumerate() {
            crate::file_ops::set_initial_view(
                &path,
                SavedView {
                    center_x: tab.center_x,
                    center_y: tab.center_y,
                    zoom: tab.zoom,
                    orientation: eov_common::viewport::Orientation::from_degrees(
                        f64::from(tab.rotation),
                        tab.flipped,
                    )
                    .unwrap_or_default(),
                },
            );
            let placement = crate::file_ops::Placement {
                batch,
                position,
                select: tab.active,
                focus: tab.active && index == session.focused_pane,
            };
            crate::open_file_placed(ui, state, tile_cache, render_timer, path, placement);
        }
    }
    let focused = session.focused_pane.min(session.panes.len() - 1);
    state.write().set_focused_pane(PaneId(focused));
    ui.set_focused_pane(focused as i32);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.toml");
        let session = Session {
            focused_pane: 1,
            panes: vec![
                PaneSession {
                    tabs: vec![TabSession {
                        path: Some("/slides/a.svs".into()),
                        os_path: None,
                        center_x: 100.0,
                        center_y: 200.0,
                        zoom: 0.25,
                        rotation: 90,
                        flipped: true,
                        active: true,
                    }],
                },
                PaneSession {
                    tabs: vec![TabSession {
                        path: None,
                        os_path: Some(crate::recent::encode_os(Path::new("/slides/b.svs"))),
                        center_x: 1.0,
                        center_y: 2.0,
                        zoom: 1.0,
                        rotation: 0,
                        flipped: false,
                        active: false,
                    }],
                },
            ],
        };
        write(&path, &session).unwrap();
        let read_back = read(&path).unwrap();
        assert_eq!(read_back, session);
        assert_eq!(
            read_back.panes[1].tabs[0].path(),
            Some(PathBuf::from("/slides/b.svs"))
        );
    }
}
