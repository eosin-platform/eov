//! Linked navigation across panes showing different slides (AUDIT Phase 3
//! #5): H&E next to IHC, serial sections.
//!
//! While linked, moving the focused pane moves every other pane by the same
//! displacement through an [`Alignment`] between the two slides (level-0
//! coordinates of one to the other: offset, rotation, scale). Zoom follows
//! the alignment's scale, which starts as the ratio of the pixel sizes
//! (equal µm per screen pixel when both slides record one, else equal
//! level-0 pixels); view turns follow so the panes keep turning together.
//!
//! A new pair starts aligned as the panes are, so linking never jumps. The
//! alignment is adjusted by moving or turning one pane alone ("adjust
//! alignment"), or solved from two landmarks placed on each slide. Adjusted alignments are
//! kept per slide pair, by slide identity, in `<data>/alignments.toml`, and
//! used the next time the two slides are linked.
//!
//! The same-slide viewport lock is unchanged: panes of one slide follow
//! each other through it, or through an identity alignment when only the
//! link is on.

use crate::state::{AppState, PaneId};
use eov_common::Viewport;
use eov_common::viewport::Orientation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Level-0 coordinates of one slide → another:
/// `p_b = scale · M · p_a + offset`, where `M` turns `rotation_deg`
/// clockwise (y down) and then mirrors left to right when `mirrored` (as a
/// view's [`Orientation`] does).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Alignment {
    pub scale: f64,
    pub rotation_deg: f64,
    #[serde(default)]
    pub mirrored: bool,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl Alignment {
    pub(crate) fn identity() -> Self {
        Self {
            scale: 1.0,
            rotation_deg: 0.0,
            mirrored: false,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    fn is_valid(&self) -> bool {
        [self.scale, self.rotation_deg, self.offset_x, self.offset_y]
            .iter()
            .all(|v| v.is_finite())
            && self.scale > 0.0
    }

    /// `M` as `[a, b, c, d]`: `x' = a·x + b·y`, `y' = c·x + d·y`.
    fn matrix(&self) -> [f64; 4] {
        let (sin, cos) = self.rotation_deg.to_radians().sin_cos();
        if self.mirrored {
            [-cos, sin, sin, cos]
        } else {
            [cos, -sin, sin, cos]
        }
    }

    /// `M` applied to a direction.
    fn turn_vector(&self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d] = self.matrix();
        (a * x + b * y, c * x + d * y)
    }

    /// Where level-0 point `(x, y)` of the first slide is on the second.
    pub(crate) fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (tx, ty) = self.turn_vector(x, y);
        (
            self.scale * tx + self.offset_x,
            self.scale * ty + self.offset_y,
        )
    }

    /// The alignment of the second slide to the first.
    pub(crate) fn inverse(&self) -> Self {
        // M is orthogonal: its inverse is its transpose, a turn back (or the
        // same mirror).
        let [a, b, c, d] = self.matrix();
        let (x, y) = (-self.offset_x / self.scale, -self.offset_y / self.scale);
        Self {
            scale: 1.0 / self.scale,
            rotation_deg: if self.mirrored {
                self.rotation_deg
            } else {
                -self.rotation_deg
            },
            mirrored: self.mirrored,
            offset_x: a * x + c * y,
            offset_y: b * x + d * y,
        }
    }

    /// This alignment after the first slide's points are turned by `first`
    /// (`M · first`): the alignment once a view turns alone.
    fn after(&self, first: Orientation) -> Self {
        let phi = f64::from(first.degrees());
        let (rotation_deg, mirrored) = if first.flipped() {
            // R(θ)·Mirror = Mirror·R(−θ)
            (phi - self.rotation_deg, !self.mirrored)
        } else {
            (self.rotation_deg + phi, self.mirrored)
        };
        Self {
            rotation_deg: (rotation_deg + 180.0).rem_euclid(360.0) - 180.0,
            mirrored,
            ..*self
        }
    }

    /// The view turn closest to `M` (whole quarter turns).
    pub(crate) fn turn(&self) -> Orientation {
        let quarter_turns = (self.rotation_deg / 90.0).round().rem_euclid(4.0) as i32;
        Orientation::new(quarter_turns, self.mirrored)
    }

    /// The alignment of two views as they are: `b`'s centre is where `a`'s
    /// is, their turns relate as shown, and one pixel of `a`'s slide is
    /// `scale` pixels of `b`'s.
    pub(crate) fn between_views(a: &Viewport, b: &Viewport, scale: f64) -> Self {
        let turn = a.orientation.then(b.orientation.inverse());
        let mut alignment = Self {
            scale,
            rotation_deg: f64::from(turn.degrees()),
            mirrored: turn.flipped(),
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let (x, y) = alignment.apply(a.center.x, a.center.y);
        alignment.offset_x = b.center.x - x;
        alignment.offset_y = b.center.y - y;
        alignment
    }

    /// The alignment that maps landmarks `a[i]` of the first slide onto
    /// `b[i]` of the second (rotation, scale and offset, mirrored when the
    /// sections are). `None` when the landmarks coincide.
    pub(crate) fn from_landmarks(
        a: [[f64; 2]; 2],
        b: [[f64; 2]; 2],
        mirrored: bool,
    ) -> Option<Self> {
        let (ux, uy) = (a[1][0] - a[0][0], a[1][1] - a[0][1]);
        let (mut vx, vy) = (b[1][0] - b[0][0], b[1][1] - b[0][1]);
        let (lu, lv) = (ux.hypot(uy), vx.hypot(vy));
        if lu < 1e-6 || lv < 1e-6 {
            return None;
        }
        if mirrored {
            // Undo the mirror on the second slide's side.
            vx = -vx;
        }
        let rotation_deg = (vy.atan2(vx) - uy.atan2(ux)).to_degrees();
        let mut alignment = Self {
            scale: lv / lu,
            rotation_deg: (rotation_deg + 180.0).rem_euclid(360.0) - 180.0,
            mirrored,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let (x, y) = alignment.apply(a[0][0], a[0][1]);
        alignment.offset_x = b[0][0] - x;
        alignment.offset_y = b[0][1] - y;
        alignment.is_valid().then_some(alignment)
    }
}

/// The focused pane's view as last propagated.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LeaderView {
    pane: u32,
    file_id: i32,
    center: (f64, f64),
    zoom: f64,
    orientation: Orientation,
}

/// Linked navigation state.
#[derive(Default)]
pub(crate) struct SlideLink {
    /// Panes with different slides follow the focused one.
    pub enabled: bool,
    /// Moving a pane moves it alone and adjusts its alignments.
    pub adjusting: bool,
    /// Alignments in use, by `(first, second)` open file, first < second.
    alignments: HashMap<(i32, i32), Alignment>,
    /// Pairs whose saved alignment was looked up already.
    looked_up: HashSet<(i32, i32)>,
    /// Pairs adjusted since they were last saved.
    dirty: HashSet<(i32, i32)>,
    /// Landmarks per open file, for solving alignments.
    pub landmarks: HashMap<i32, [Option<[f64; 2]>; 2]>,
    last: Option<LeaderView>,
    /// Each pane's slide when last synced, and whether its identity was
    /// known: a slide that opens, or is identified, aligns at once.
    last_panes: Vec<(Option<i32>, bool)>,
}

impl SlideLink {
    fn get(&self, a: i32, b: i32) -> Option<Alignment> {
        if a <= b {
            self.alignments.get(&(a, b)).copied()
        } else {
            self.alignments.get(&(b, a)).map(Alignment::inverse)
        }
    }

    fn set(&mut self, a: i32, b: i32, alignment: Alignment) {
        if a <= b {
            self.alignments.insert((a, b), alignment);
        } else {
            self.alignments.insert((b, a), alignment.inverse());
        }
    }

    /// Forgets everything about files that are closed.
    fn retain_files(&mut self, open: &HashSet<i32>) {
        let keep = |pair: &(i32, i32)| open.contains(&pair.0) && open.contains(&pair.1);
        self.alignments.retain(|pair, _| keep(pair));
        self.looked_up.retain(keep);
        self.dirty.retain(keep);
        self.landmarks.retain(|file, _| open.contains(file));
    }
}

fn pair(a: i32, b: i32) -> (i32, i32) {
    (a.min(b), a.max(b))
}

/// Level-0 pixels of `b` per level-0 pixel of `a` for equal µm: the ratio
/// of the pixel sizes when both slides have one, else 1.
fn physical_scale(state: &AppState, a: i32, b: i32) -> f64 {
    let mpp = |id: i32| {
        state
            .get_file(id)
            .and_then(|file| crate::host_ui::calibration(file).average())
            .filter(|mpp| mpp.is_finite() && *mpp > 0.0)
    };
    match (mpp(a), mpp(b)) {
        (Some(a), Some(b)) => a / b,
        _ => 1.0,
    }
}

fn viewport_of(state: &AppState, pane: PaneId) -> Option<(i32, Viewport)> {
    let file_id = state.active_file_id_for_pane(pane)?;
    let viewport = state
        .get_file(file_id)?
        .pane_state(pane)?
        .viewport
        .viewport
        .clone();
    Some((file_id, viewport))
}

/// The slide identity key of an open file, once it is known.
fn identity_key(file_id: i32) -> Option<String> {
    crate::host_ui::with_host(|host| host.identity_key(i64::from(file_id))).flatten()
}

/// The alignment of `a`'s slide to `b`'s: in use, saved for the pair, or
/// the two views as `a_view` and `b_view` are.
fn alignment_for(
    state: &mut AppState,
    a: i32,
    b: i32,
    a_view: &Viewport,
    b_view: &Viewport,
) -> Alignment {
    if a == b {
        return Alignment::identity();
    }
    let key = pair(a, b);
    // A saved alignment wins once both slides are identified (identities
    // are computed in the background after a slide opens).
    if !state.slide_link.looked_up.contains(&key)
        && let (Some(id_a), Some(id_b)) = (identity_key(a), identity_key(b))
    {
        state.slide_link.looked_up.insert(key);
        let alignment = lookup(&id_a, &id_b).unwrap_or_else(|| {
            // Until now the pair was aligned as the views were; derive that
            // again, with a pixel size saved for either slide (it applies
            // once the slide is identified).
            Alignment::between_views(a_view, b_view, physical_scale(state, a, b))
        });
        state.slide_link.set(a, b, alignment);
        return alignment;
    }
    if let Some(alignment) = state.slide_link.get(a, b) {
        return alignment;
    }
    let alignment = Alignment::between_views(a_view, b_view, physical_scale(state, a, b));
    state.slide_link.set(a, b, alignment);
    alignment
}

/// Propagates the focused pane's view to the linked panes (or, while
/// adjusting, re-aligns them to it). Called once per render tick; cheap
/// when nothing moved. Returns whether a pane changed.
pub(crate) fn sync_views(state: &mut AppState) -> bool {
    if !state.slide_link.enabled || !state.split_enabled || state.panes.len() < 2 {
        state.slide_link.last = None;
        return false;
    }
    let open: HashSet<i32> = state.open_files.iter().map(|file| file.id).collect();
    state.slide_link.retain_files(&open);
    let leader = state.active_viewport_pane();
    let Some(uid) = state.pane_uid(leader) else {
        return false;
    };
    // Advance the leader's animation first, so the others follow this frame.
    if let Some(file_id) = state.active_file_id_for_pane(leader)
        && let Some(pane_state) = state
            .get_file_mut(file_id)
            .and_then(|f| f.pane_state_mut(leader))
    {
        pane_state.viewport.update();
    }
    let Some((leader_file, leader_view)) = viewport_of(state, leader) else {
        return false;
    };
    let now = LeaderView {
        pane: uid,
        file_id: leader_file,
        center: (leader_view.center.x, leader_view.center.y),
        zoom: leader_view.zoom,
        orientation: leader_view.orientation,
    };
    let panes: Vec<(Option<i32>, bool)> = (0..state.panes.len())
        .map(|index| {
            let file = state.active_file_id_for_pane(PaneId(index));
            (file, file.is_some_and(|file| identity_key(file).is_some()))
        })
        .collect();
    let previous = state.slide_link.last;
    if previous == Some(now) && state.slide_link.last_panes == panes {
        return false;
    }
    state.slide_link.last_panes = panes;
    let same_leader = previous.filter(|last| last.pane == uid && last.file_id == leader_file);
    // New pairs start aligned as the panes were before this move.
    let before = same_leader
        .map(|last| {
            let mut view = leader_view.clone();
            view.center.x = last.center.0;
            view.center.y = last.center.1;
            view.zoom = last.zoom;
            view.orientation = last.orientation;
            view
        })
        .unwrap_or_else(|| leader_view.clone());
    state.slide_link.last = Some(now);

    let lock = state.viewport_lock_enabled;
    let adjusting = state.slide_link.adjusting;
    // While adjusting, the focused pane also turns alone: the turn since the
    // last frame, in its slide's coordinates.
    let turned_alone = same_leader
        .filter(|last| adjusting && last.orientation != leader_view.orientation)
        .map(|last| leader_view.orientation.then(last.orientation.inverse()));
    let mut changed = false;
    for index in 0..state.panes.len() {
        let pane = PaneId(index);
        if pane == leader {
            continue;
        }
        let Some((file_id, view)) = viewport_of(state, pane) else {
            continue;
        };
        if file_id == leader_file && (lock || adjusting) {
            // The viewport lock moves panes of the same slide.
            continue;
        }
        let mut alignment = alignment_for(state, leader_file, file_id, &before, &view);
        let center = if adjusting && file_id != leader_file {
            // The pane stays; the alignment follows the move.
            if let Some(turn) = turned_alone {
                alignment = alignment.after(turn);
            }
            let (x, y) = alignment.apply(leader_view.center.x, leader_view.center.y);
            alignment.offset_x += view.center.x - x;
            alignment.offset_y += view.center.y - y;
            let key = pair(leader_file, file_id);
            state.slide_link.set(leader_file, file_id, alignment);
            state.slide_link.dirty.insert(key);
            // What was adjusted wins over an alignment saved before.
            state.slide_link.looked_up.insert(key);
            (view.center.x, view.center.y)
        } else {
            alignment.apply(leader_view.center.x, leader_view.center.y)
        };
        let zoom = leader_view.zoom / alignment.scale;
        let orientation = alignment.turn().inverse().then(leader_view.orientation);
        let Some(viewport) = state
            .get_file_mut(file_id)
            .and_then(|file| file.pane_state_mut(pane))
            .map(|pane_state| &mut pane_state.viewport)
        else {
            continue;
        };
        let current = &viewport.viewport;
        let moved = (current.center.x - center.0).abs() > 1e-9
            || (current.center.y - center.1).abs() > 1e-9
            || (current.zoom - zoom).abs() > 1e-12
            || current.orientation != orientation;
        if moved {
            viewport.stop();
            viewport.viewport.orientation = orientation;
            viewport.set_center_zoom(center.0, center.1, zoom);
            changed = true;
        }
    }
    if changed {
        state.request_render();
    }
    changed
}

/// Links (or unlinks) panes showing different slides. Returns whether the
/// link is on.
pub(crate) fn toggle_link(state: &mut AppState) -> bool {
    if state.slide_link.enabled {
        save_dirty(state);
        state.slide_link = SlideLink::default();
        false
    } else {
        state.slide_link.enabled = true;
        state.slide_link.last = None;
        state.request_render();
        true
    }
}

/// Starts or ends adjusting: moving a pane moves it alone and updates its
/// alignments, which are saved when adjusting ends. Returns whether it is
/// on (it needs the link).
pub(crate) fn toggle_adjust(state: &mut AppState) -> bool {
    if !state.slide_link.enabled {
        return false;
    }
    let adjusting = !state.slide_link.adjusting;
    state.slide_link.adjusting = adjusting;
    if !adjusting {
        save_dirty(state);
    }
    adjusting
}

/// Places landmark `index` (0 or 1) of `file_id`'s slide at level-0
/// `point`.
pub(crate) fn set_landmark(state: &mut AppState, file_id: i32, index: usize, point: [f64; 2]) {
    if index < 2 {
        state.slide_link.landmarks.entry(file_id).or_default()[index] = Some(point);
    }
}

pub(crate) fn clear_landmarks(state: &mut AppState) {
    state.slide_link.landmarks.clear();
}

/// `leader`'s slide and the other linked panes' slides that have both
/// landmarks, with whether their views are mirrored to each other.
fn landmark_pairs(state: &AppState, leader: PaneId) -> Vec<(i32, i32, bool)> {
    let Some((leader_file, leader_view)) = viewport_of(state, leader) else {
        return Vec::new();
    };
    let complete = |file: i32| {
        state
            .slide_link
            .landmarks
            .get(&file)
            .is_some_and(|marks| marks.iter().all(Option::is_some))
    };
    if !complete(leader_file) {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for index in 0..state.panes.len() {
        let pane = PaneId(index);
        if pane == leader {
            continue;
        }
        if let Some((file, view)) = viewport_of(state, pane)
            && file != leader_file
            && complete(file)
            && !pairs.iter().any(|(_, f, _)| *f == file)
        {
            let mirrored = leader_view
                .orientation
                .then(view.orientation.inverse())
                .flipped();
            pairs.push((leader_file, file, mirrored));
        }
    }
    pairs
}

/// Whether "align from landmarks" has something to align for `pane`.
pub(crate) fn can_align_from_landmarks(state: &AppState, pane: PaneId) -> bool {
    state.slide_link.enabled && !landmark_pairs(state, pane).is_empty()
}

/// Aligns the other linked panes to the focused one from the landmarks
/// placed on both slides, saves the alignments and moves the panes there.
pub(crate) fn align_from_landmarks(state: &mut AppState) -> Result<String, String> {
    if !state.slide_link.enabled {
        return Err("Link the panes first".to_string());
    }
    let pairs = landmark_pairs(state, state.active_viewport_pane());
    if pairs.is_empty() {
        return Err("Place landmarks 1 and 2 on the focused slide and on another".to_string());
    }
    let mut aligned = 0;
    for (a, b, mirrored) in pairs {
        let marks = |file: i32| {
            let [p, q] = state.slide_link.landmarks[&file];
            [p.unwrap_or_default(), q.unwrap_or_default()]
        };
        let Some(alignment) = Alignment::from_landmarks(marks(a), marks(b), mirrored) else {
            continue;
        };
        state.slide_link.set(a, b, alignment);
        state.slide_link.looked_up.insert(pair(a, b));
        state.slide_link.dirty.insert(pair(a, b));
        aligned += 1;
    }
    if aligned == 0 {
        return Err("The landmarks of a slide must be apart".to_string());
    }
    save_dirty(state);
    // Move the others to the new alignment now.
    state.slide_link.last = None;
    let adjusting = std::mem::replace(&mut state.slide_link.adjusting, false);
    sync_views(state);
    state.slide_link.adjusting = adjusting;
    Ok(format!(
        "Aligned {aligned} slide{} from landmarks",
        if aligned == 1 { "" } else { "s" }
    ))
}

/// Forgets the alignments of the focused pane's slide (also the saved
/// ones): the panes are aligned as they are now.
pub(crate) fn reset_alignment(state: &mut AppState) -> bool {
    let Some(file) = state.active_file_id_for_pane(state.active_viewport_pane()) else {
        return false;
    };
    let pairs: Vec<(i32, i32)> = state
        .slide_link
        .alignments
        .keys()
        .copied()
        .filter(|(a, b)| *a == file || *b == file)
        .collect();
    for key in &pairs {
        state.slide_link.alignments.remove(key);
        state.slide_link.dirty.remove(key);
        state.slide_link.looked_up.insert(*key);
        if let (Some(a), Some(b)) = (identity_key(key.0), identity_key(key.1))
            && let Err(error) = forget(&a, &b)
        {
            tracing::warn!("Could not forget the alignment: {error}");
        }
    }
    state.slide_link.last = None;
    !pairs.is_empty()
}

/// Saves the alignments adjusted since they were last saved.
fn save_dirty(state: &mut AppState) {
    let dirty: Vec<(i32, i32)> = state.slide_link.dirty.drain().collect();
    for (a, b) in dirty {
        let Some(alignment) = state.slide_link.get(a, b) else {
            continue;
        };
        let (Some(id_a), Some(id_b)) = (identity_key(a), identity_key(b)) else {
            tracing::warn!("Cannot save an alignment: a slide is not identified yet");
            continue;
        };
        let name = |id: i32| {
            state
                .get_file(id)
                .map(|file| file.filename.clone())
                .unwrap_or_default()
        };
        if let Err(error) = store(
            &id_a,
            &id_b,
            &alignment,
            &format!("{} → {}", name(a), name(b)),
        ) {
            tracing::warn!("Could not save the alignment: {error}");
        }
    }
}

// ─── Saved alignments ───────────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    #[serde(default)]
    pair: BTreeMap<String, Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    /// The two slides' file names when it was saved, for people.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    names: String,
    #[serde(flatten)]
    alignment: Alignment,
}

fn store_path() -> PathBuf {
    crate::paths::data_dir().join("alignments.toml")
}

/// The store key of a slide pair (ordered) and whether `a` comes first.
fn key_of(a: &str, b: &str) -> (String, bool) {
    if a <= b {
        (format!("{a}|{b}"), true)
    } else {
        (format!("{b}|{a}"), false)
    }
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

/// The saved alignment of slide `a` to slide `b` (by identity), if any.
pub(crate) fn lookup(a: &str, b: &str) -> Option<Alignment> {
    lookup_in(&store_path(), a, b)
}

fn lookup_in(path: &Path, a: &str, b: &str) -> Option<Alignment> {
    let (key, a_first) = key_of(a, b);
    let alignment = read(path).pair.get(&key)?.alignment;
    let alignment = if a_first {
        alignment
    } else {
        alignment.inverse()
    };
    alignment.is_valid().then_some(alignment)
}

fn store(a: &str, b: &str, alignment: &Alignment, names: &str) -> Result<(), String> {
    store_in(&store_path(), a, b, alignment, names)
}

fn store_in(
    path: &Path,
    a: &str,
    b: &str,
    alignment: &Alignment,
    names: &str,
) -> Result<(), String> {
    if !alignment.is_valid() {
        return Err("the alignment is not valid".to_string());
    }
    let (key, a_first) = key_of(a, b);
    let mut store = read(path);
    store.pair.insert(
        key,
        Entry {
            names: names.to_string(),
            alignment: if a_first {
                *alignment
            } else {
                alignment.inverse()
            },
        },
    );
    write(path, &store)
}

fn forget(a: &str, b: &str) -> Result<(), String> {
    forget_in(&store_path(), a, b)
}

fn forget_in(path: &Path, a: &str, b: &str) -> Result<(), String> {
    let mut store = read(path);
    if store.pair.remove(&key_of(a, b).0).is_some() {
        write(path, &store)
    } else {
        Ok(())
    }
}

/// Fixtures shared with the host UI's tests.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    /// Two slides with different pixel sizes side by side.
    pub(crate) fn two_slides() -> (AppState, tempfile::TempDir, i32, i32, PaneId) {
        use std::sync::Arc;
        crate::paths::isolate_for_tests();
        let dir = tempfile::tempdir().unwrap();
        let mut state = AppState::new();
        let open = |state: &mut AppState, name: &str, size: u32, target: PaneId| {
            let path = dir.path().join(name);
            eov_common::test_support::write_tiled_tiff(&path, size, size, 256, |_, _| {
                [200, 200, 200]
            })
            .unwrap();
            let id = state.allocate_file_id();
            let wsi = eov_common::WsiFile::open(&path).unwrap();
            let cache = Arc::new(eov_common::TileCache::with_limits(8, 8 << 20));
            let manager = Arc::new(eov_common::TileManager::new(wsi.clone(), id));
            let loader = Arc::new(crate::tile_loader::TileLoader::new(
                Arc::clone(&manager),
                cache,
            ));
            state.add_file(crate::state::NewOpenFile {
                id,
                path,
                wsi,
                tile_manager: manager,
                tile_loader: loader,
                viewport: eov_common::ViewportState::new(
                    400.0,
                    300.0,
                    f64::from(size),
                    f64::from(size),
                ),
                thumbnail: None,
                target_pane: target,
                activation: crate::state::TabActivation::Focus,
            });
            id
        };
        let a = open(&mut state, "a.tif", 1024, PaneId::PRIMARY);
        let second = state.insert_pane(1).unwrap();
        let b = open(&mut state, "b.tif", 2048, second);
        // A's pixels are 0.5 µm, B's 0.25 µm: B has twice the pixels per µm.
        state.get_file_mut(a).unwrap().calibration_override = Some(0.5);
        state.get_file_mut(b).unwrap().calibration_override = Some(0.25);
        state.set_focused_pane(PaneId::PRIMARY);
        (state, dir, a, b, second)
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::two_slides;
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn alignments_invert_and_solve_from_landmarks() {
        for mirrored in [false, true] {
            let alignment = Alignment {
                scale: 0.5,
                rotation_deg: 33.0,
                mirrored,
                offset_x: 1200.0,
                offset_y: -300.0,
            };
            let p = alignment.apply(400.0, 250.0);
            assert!(close(alignment.inverse().apply(p.0, p.1), (400.0, 250.0)));
            let a = [[100.0, 200.0], [900.0, 650.0]];
            let b = [alignment.apply(100.0, 200.0), alignment.apply(900.0, 650.0)];
            let solved =
                Alignment::from_landmarks(a, [[b[0].0, b[0].1], [b[1].0, b[1].1]], mirrored)
                    .unwrap();
            assert!((solved.scale - 0.5).abs() < 1e-9);
            assert!((solved.rotation_deg - 33.0).abs() < 1e-6);
            assert!(close(solved.apply(400.0, 250.0), p));
            // Turning the first slide's points first is the alignment after
            // that turn.
            for first in Orientation::all() {
                let (x, y) = first.to_screen_xy(400.0, 250.0);
                assert!(close(
                    alignment.after(first).apply(400.0, 250.0),
                    alignment.apply(x, y)
                ));
            }
        }
        assert!(
            Alignment::from_landmarks([[1.0, 1.0]; 2], [[0.0, 0.0], [5.0, 5.0]], false).is_none()
        );
        let near = Alignment {
            rotation_deg: 87.0,
            ..Alignment::identity()
        };
        assert_eq!(near.turn(), Orientation::new(1, false));
    }

    /// Linking keeps the views as they are, and from then on moving one
    /// pane moves the other the same way on screen.
    #[test]
    fn linked_views_move_together_on_screen() {
        for a_turn in Orientation::all() {
            for b_turn in Orientation::all() {
                let mut a = Viewport::new(800.0, 600.0, 20_000.0, 10_000.0);
                a.zoom = 0.2;
                a.center.x = 5000.0;
                a.center.y = 3000.0;
                a.orientation = a_turn;
                let mut b = Viewport::new(800.0, 600.0, 40_000.0, 20_000.0);
                b.center.x = 12_000.0;
                b.center.y = 7000.0;
                b.orientation = b_turn;
                // B's pixels are half as large: twice as many per µm.
                let alignment = Alignment::between_views(&a, &b, 2.0);
                b.zoom = a.zoom / alignment.scale;
                assert!(close(
                    alignment.apply(a.center.x, a.center.y),
                    (b.center.x, b.center.y)
                ));
                assert_eq!(
                    alignment.turn().inverse().then(a.orientation),
                    b.orientation
                );
                // A point under the pointer in A is under it in B.
                let p = a.screen_to_image(120.0, 450.0);
                let q = alignment.apply(p.x, p.y);
                let shown = b.image_to_screen(q.0, q.1);
                assert!(
                    close((shown.x, shown.y), (120.0, 450.0)),
                    "{a_turn:?} {b_turn:?}"
                );
            }
        }
    }

    fn view(state: &AppState, pane: PaneId) -> Viewport {
        viewport_of(state, pane).unwrap().1
    }

    fn set_view(state: &mut AppState, pane: PaneId, x: f64, y: f64, zoom: f64) {
        let file = state.active_file_id_for_pane(pane).unwrap();
        let viewport = &mut state
            .get_file_mut(file)
            .unwrap()
            .pane_state_mut(pane)
            .unwrap()
            .viewport;
        viewport.set_center_zoom(x, y, zoom);
    }

    #[test]
    fn linked_panes_follow_in_physical_units_and_adjust() {
        let (mut state, _dir, a, b, second) = two_slides();
        set_view(&mut state, PaneId::PRIMARY, 500.0, 400.0, 0.5);
        set_view(&mut state, second, 900.0, 1000.0, 0.7);
        assert!(toggle_link(&mut state));
        // Linking moves nothing; the zoom follows the pixel sizes.
        sync_views(&mut state);
        let follower = view(&state, second);
        assert!(close(
            (follower.center.x, follower.center.y),
            (900.0, 1000.0)
        ));
        assert!(
            (follower.zoom - 0.25).abs() < 1e-9,
            "same µm per screen pixel"
        );

        // 100 px of A to the right is 200 px of B.
        set_view(&mut state, PaneId::PRIMARY, 600.0, 400.0, 0.5);
        assert!(sync_views(&mut state));
        let follower = view(&state, second);
        assert!(close(
            (follower.center.x, follower.center.y),
            (1100.0, 1000.0)
        ));
        assert!(!sync_views(&mut state), "nothing moved since");

        // Turning the focused pane turns the other one too.
        assert!(crate::callbacks::rotate_active_viewport(&mut state, 1));
        sync_views(&mut state);
        assert_eq!(view(&state, second).orientation, Orientation::new(1, false));

        // While adjusting, the other pane stays and the alignment follows.
        assert!(toggle_adjust(&mut state));
        set_view(&mut state, PaneId::PRIMARY, 700.0, 450.0, 0.5);
        sync_views(&mut state);
        let follower = view(&state, second);
        assert!(close(
            (follower.center.x, follower.center.y),
            (1100.0, 1000.0)
        ));
        assert!(!toggle_adjust(&mut state));
        set_view(&mut state, PaneId::PRIMARY, 710.0, 450.0, 0.5);
        sync_views(&mut state);
        let follower = view(&state, second);
        assert!(close(
            (follower.center.x, follower.center.y),
            (1120.0, 1000.0)
        ));

        // While adjusting, the focused pane also turns alone; then the panes
        // move the same way on screen again.
        assert!(toggle_adjust(&mut state));
        assert!(crate::callbacks::rotate_active_viewport(&mut state, 1));
        sync_views(&mut state);
        assert_eq!(view(&state, second).orientation, Orientation::new(1, false));
        assert!(!toggle_adjust(&mut state));
        set_view(&mut state, PaneId::PRIMARY, 650.0, 420.0, 0.5);
        sync_views(&mut state);
        let (leader, follower) = (view(&state, PaneId::PRIMARY), view(&state, second));
        assert_eq!(follower.orientation, Orientation::new(1, false));
        let alignment = state.slide_link.get(a, b).unwrap();
        let p = leader.screen_to_image(120.0, 250.0);
        let q = alignment.apply(p.x, p.y);
        let shown = follower.image_to_screen(q.0, q.1);
        assert!(close((shown.x, shown.y), (120.0, 250.0)), "{shown:?}");
        assert!(!toggle_link(&mut state));
    }

    #[test]
    fn landmarks_align_the_linked_panes() {
        let (mut state, _dir, a, b, second) = two_slides();
        assert!(toggle_link(&mut state));
        let truth = Alignment {
            scale: 2.0,
            rotation_deg: 10.0,
            mirrored: false,
            offset_x: 100.0,
            offset_y: -50.0,
        };
        assert!(!can_align_from_landmarks(&state, PaneId::PRIMARY));
        for (index, p) in [[200.0, 300.0], [700.0, 650.0]].into_iter().enumerate() {
            set_landmark(&mut state, a, index, p);
            let q = truth.apply(p[0], p[1]);
            set_landmark(&mut state, b, index, [q.0, q.1]);
        }
        assert!(can_align_from_landmarks(&state, PaneId::PRIMARY));
        align_from_landmarks(&mut state).unwrap();
        set_view(&mut state, PaneId::PRIMARY, 512.0, 512.0, 0.4);
        sync_views(&mut state);
        let follower = view(&state, second);
        let expected = truth.apply(512.0, 512.0);
        assert!(close((follower.center.x, follower.center.y), expected));
        assert!((follower.zoom - 0.2).abs() < 1e-9);
        assert!(reset_alignment(&mut state));
    }

    /// The toolbar toggles reach the link, and the render loop moves the
    /// linked pane.
    #[test]
    fn the_toolbar_links_the_panes() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            use parking_lot::RwLock;
            use std::rc::Rc;
            use std::sync::Arc;
            crate::render_pool::init_global().unwrap();
            let (mut state, _dir, _a, _b, second) = two_slides();
            state.render_backend = eov_common::RenderBackend::Cpu;
            set_view(&mut state, PaneId::PRIMARY, 500.0, 400.0, 0.5);
            set_view(&mut state, second, 900.0, 1000.0, 0.25);
            let state = Arc::new(RwLock::new(state));
            let ui = crate::AppWindow::new().unwrap();
            let tile_cache = Arc::new(eov_common::TileCache::with_limits(16, 16 << 20));
            crate::callbacks::setup_callbacks(
                &ui,
                Arc::clone(&state),
                Arc::clone(&tile_cache),
                Rc::new(slint::Timer::default()),
            );
            ui.invoke_toggle_slide_link_requested();
            assert!(ui.get_slide_link_enabled());
            crate::render::update_and_render(&ui, &state, &tile_cache);
            set_view(&mut state.write(), PaneId::PRIMARY, 550.0, 400.0, 0.5);
            crate::render::update_and_render(&ui, &state, &tile_cache);
            let b = view(&state.read(), second);
            assert!(close((b.center.x, b.center.y), (1000.0, 1000.0)));
            ui.invoke_toggle_alignment_adjust_requested();
            assert!(ui.get_alignment_adjust_enabled());
            ui.invoke_toggle_slide_link_requested();
            assert!(!ui.get_slide_link_enabled() && !ui.get_alignment_adjust_enabled());
        });
    }

    #[test]
    fn alignments_are_saved_per_slide_pair() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alignments.toml");
        let alignment = Alignment {
            scale: 2.0,
            rotation_deg: 90.0,
            mirrored: false,
            offset_x: 10.0,
            offset_y: 20.0,
        };
        assert_eq!(lookup_in(&path, "b", "a"), None);
        store_in(&path, "b", "a", &alignment, "b.svs → a.svs").unwrap();
        let back = lookup_in(&path, "b", "a").unwrap();
        assert!(close(back.apply(3.0, 4.0), alignment.apply(3.0, 4.0)));
        // The other way round is the inverse.
        let other = lookup_in(&path, "a", "b").unwrap();
        let p = alignment.apply(3.0, 4.0);
        assert!(close(other.apply(p.0, p.1), (3.0, 4.0)));
        forget_in(&path, "a", "b").unwrap();
        assert_eq!(lookup_in(&path, "b", "a"), None);
    }
}
