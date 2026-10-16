//! Navigation history (AUDIT §14.1): per pane, the fields of view the user
//! settled on, so Alt+← and Alt+→ step back and forward through them, like
//! a browser.
//!
//! A view is recorded when it settles (the same debounce as plugins'
//! `ViewChanged`), and only when it differs noticeably from the previous
//! one: moved by more than half a screen, or zoomed by more than 1.5×.
//! Each entry keeps the view's turn (any angle, and the mirror), which
//! stepping back or forward restores; turning alone refines the entry.

use std::collections::HashMap;

/// Entries kept per pane in each direction.
const MAX_ENTRIES: usize = 100;

/// A field of view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Visit {
    pub file_id: i32,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    /// Pane size in logical pixels, to judge "half a screen".
    pub width: f64,
    pub height: f64,
    /// How the view was turned.
    pub orientation: eov_common::viewport::Orientation,
}

impl Visit {
    /// Whether going from `self` to `other` is a move worth remembering.
    fn differs_from(&self, other: &Visit) -> bool {
        if self.file_id != other.file_id {
            return true;
        }
        let ratio = (self.zoom / other.zoom.max(1e-12)).max(other.zoom / self.zoom.max(1e-12));
        if ratio > 1.5 {
            return true;
        }
        // Distance in screen pixels at the smaller zoom.
        let zoom = self.zoom.min(other.zoom);
        let dx = (self.center_x - other.center_x).abs() * zoom;
        let dy = (self.center_y - other.center_y).abs() * zoom;
        dx > self.width.min(other.width) / 2.0 || dy > self.height.min(other.height) / 2.0
    }
}

#[derive(Default)]
struct PaneHistory {
    back: Vec<Visit>,
    forward: Vec<Visit>,
    current: Option<Visit>,
    /// The next settled view is where a step back or forward went.
    arriving: bool,
}

/// The histories of all panes, by stable pane id.
#[derive(Default)]
pub(crate) struct History {
    panes: HashMap<u32, PaneHistory>,
}

impl History {
    /// A pane's view settled at `visit`.
    pub fn settled(&mut self, pane: u32, visit: Visit) {
        let history = self.panes.entry(pane).or_default();
        if std::mem::take(&mut history.arriving) {
            history.current = Some(visit);
            return;
        }
        match history.current {
            Some(current) if current.differs_from(&visit) => {
                history.back.push(current);
                if history.back.len() > MAX_ENTRIES {
                    history.back.remove(0);
                }
                history.forward.clear();
                history.current = Some(visit);
            }
            Some(_) => {
                // Small adjustments refine the current entry.
                history.current = Some(visit);
            }
            None => history.current = Some(visit),
        }
    }

    /// Steps back; returns where to go. `is_open` says whether a slide is
    /// still open (entries of closed slides are skipped).
    pub fn back(&mut self, pane: u32, is_open: impl Fn(i32) -> bool) -> Option<Visit> {
        let history = self.panes.get_mut(&pane)?;
        while let Some(visit) = history.back.pop() {
            if !is_open(visit.file_id) {
                continue;
            }
            if let Some(current) = history.current.replace(visit) {
                history.forward.push(current);
            }
            history.arriving = true;
            return Some(visit);
        }
        None
    }

    /// Steps forward again after [`back`](Self::back).
    pub fn forward(&mut self, pane: u32, is_open: impl Fn(i32) -> bool) -> Option<Visit> {
        let history = self.panes.get_mut(&pane)?;
        while let Some(visit) = history.forward.pop() {
            if !is_open(visit.file_id) {
                continue;
            }
            if let Some(current) = history.current.replace(visit) {
                history.back.push(current);
            }
            history.arriving = true;
            return Some(visit);
        }
        None
    }

    /// Forgets panes that no longer exist.
    pub fn retain_panes(&mut self, live: &[u32]) {
        self.panes.retain(|pane, _| live.contains(pane));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, zoom: f64) -> Visit {
        Visit {
            file_id: 1,
            center_x: x,
            center_y: 0.0,
            zoom,
            width: 1000.0,
            height: 800.0,
            orientation: Default::default(),
        }
    }

    /// Entries come back with the angle they were seen at; turning alone
    /// makes no entry.
    #[test]
    fn steps_restore_the_angle_of_each_field() {
        use eov_common::viewport::Orientation;
        let tilted = |x: f64, degrees: f64| Visit {
            orientation: Orientation::from_degrees(degrees, false).unwrap(),
            ..at(x, 1.0)
        };
        let mut history = History::default();
        history.settled(3, tilted(0.0, 37.25));
        history.settled(3, tilted(9000.0, 37.25));
        history.settled(3, tilted(9000.0, 0.0)); // turned only: refines
        let back = history.back(3, |_| true).unwrap();
        assert_eq!(back, tilted(0.0, 37.25));
        assert_eq!(back.orientation.degrees(), 37.25);
        history.settled(3, back);
        assert_eq!(history.forward(3, |_| true), Some(tilted(9000.0, 0.0)));
    }

    #[test]
    fn steps_back_and_forward_through_settled_fields() {
        let mut history = History::default();
        history.settled(7, at(0.0, 1.0));
        history.settled(7, at(100.0, 1.0)); // small: refines
        history.settled(7, at(5000.0, 1.0)); // far: new entry
        history.settled(7, at(5000.0, 4.0)); // zoomed: new entry
        let open = |_| true;
        assert_eq!(history.back(7, open), Some(at(5000.0, 1.0)));
        history.settled(7, at(5000.0, 1.0)); // arriving
        assert_eq!(history.back(7, open), Some(at(100.0, 1.0)));
        history.settled(7, at(100.0, 1.0));
        assert_eq!(history.back(7, open), None);
        assert_eq!(history.forward(7, open), Some(at(5000.0, 1.0)));
        history.settled(7, at(5000.0, 1.0));
        assert_eq!(history.forward(7, open), Some(at(5000.0, 4.0)));
        history.settled(7, at(5000.0, 4.0));
        assert_eq!(history.forward(7, open), None);
    }

    #[test]
    fn new_moves_clear_the_forward_history_and_closed_slides_are_skipped() {
        let mut history = History::default();
        history.settled(1, at(0.0, 1.0));
        history.settled(
            1,
            Visit {
                file_id: 2,
                ..at(0.0, 1.0)
            },
        );
        history.settled(1, at(9000.0, 1.0));
        // Slide 2 was closed: stepping back skips it.
        assert_eq!(history.back(1, |file| file != 2), Some(at(0.0, 1.0)));
        history.settled(1, at(0.0, 1.0));
        history.settled(1, at(-9000.0, 1.0));
        assert_eq!(history.forward(1, |_| true), None);
    }
}
