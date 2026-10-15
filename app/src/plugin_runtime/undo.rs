//! One undo history for the whole application.
//!
//! Host edits (ROI, measurements) and plugin changes share the stack, so
//! Ctrl/⌘+Z always undoes the most recent change to the slide being looked
//! at, whoever made it (PLG-10). Plugin entries are undone by sending the
//! owner `Undo { entry_id }`; host entries carry their own inverse.

use crate::interaction::HostMeasurement;
use eov_plugin_api::protocol::{FileId, Rect};
use std::sync::Arc;

/// Entries kept per stack before the oldest are dropped.
pub const MAX_ENTRIES: usize = 500;

#[derive(Clone, Debug, PartialEq)]
pub enum HostChange {
    Roi {
        before: Option<Rect>,
        after: Option<Rect>,
    },
    MeasurementAdded(HostMeasurement),
    MeasurementsRemoved(Vec<HostMeasurement>),
    MeasurementChanged {
        before: HostMeasurement,
        after: HostMeasurement,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum UndoAction {
    Host(HostChange),
    Plugin { plugin_id: Arc<str>, entry_id: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct UndoEntry {
    /// `None` for changes not tied to one slide.
    pub file_id: Option<FileId>,
    pub label: String,
    pub action: UndoAction,
}

#[derive(Default)]
pub struct UndoStack {
    undo: Vec<UndoEntry>,
    redo: Vec<UndoEntry>,
}

/// What the app must do to perform an undo or redo.
#[derive(Clone, Debug, PartialEq)]
pub enum UndoStep {
    /// Apply this host change (already inverted for undo).
    Host { file_id: Option<FileId>, change: HostChange },
    /// Send `Undo`/`Redo` to the plugin.
    Plugin { plugin_id: Arc<str>, entry_id: String, redo: bool },
}

fn in_scope(entry: &UndoEntry, file_id: Option<FileId>) -> bool {
    entry.file_id.is_none() || entry.file_id == file_id
}

fn invert(change: &HostChange) -> HostChange {
    match change {
        HostChange::Roi { before, after } => HostChange::Roi {
            before: *after,
            after: *before,
        },
        HostChange::MeasurementAdded(m) => HostChange::MeasurementsRemoved(vec![m.clone()]),
        HostChange::MeasurementsRemoved(ms) if ms.len() == 1 => {
            HostChange::MeasurementAdded(ms[0].clone())
        }
        HostChange::MeasurementsRemoved(ms) => HostChange::MeasurementsRemoved(ms.clone()),
        HostChange::MeasurementChanged { before, after } => HostChange::MeasurementChanged {
            before: after.clone(),
            after: before.clone(),
        },
    }
}

impl UndoStack {
    pub fn push(&mut self, entry: UndoEntry) {
        // A new change invalidates the redo history of its slide.
        let file_id = entry.file_id;
        self.redo.retain(|e| !in_scope(e, file_id));
        self.undo.push(entry);
        if self.undo.len() > MAX_ENTRIES {
            self.undo.remove(0);
        }
    }

    pub fn can_undo(&self, file_id: Option<FileId>) -> bool {
        self.undo.iter().any(|e| in_scope(e, file_id))
    }

    pub fn can_redo(&self, file_id: Option<FileId>) -> bool {
        self.redo.iter().any(|e| in_scope(e, file_id))
    }

    pub fn undo_label(&self, file_id: Option<FileId>) -> Option<&str> {
        self.undo
            .iter()
            .rev()
            .find(|e| in_scope(e, file_id))
            .map(|e| e.label.as_str())
    }

    pub fn redo_label(&self, file_id: Option<FileId>) -> Option<&str> {
        self.redo
            .iter()
            .rev()
            .find(|e| in_scope(e, file_id))
            .map(|e| e.label.as_str())
    }

    /// Pops the newest entry for `file_id` and returns how to undo it.
    pub fn undo(&mut self, file_id: Option<FileId>) -> Option<UndoStep> {
        let index = self.undo.iter().rposition(|e| in_scope(e, file_id))?;
        let entry = self.undo.remove(index);
        let step = match &entry.action {
            UndoAction::Host(change) => UndoStep::Host {
                file_id: entry.file_id,
                change: invert(change),
            },
            UndoAction::Plugin {
                plugin_id,
                entry_id,
            } => UndoStep::Plugin {
                plugin_id: Arc::clone(plugin_id),
                entry_id: entry_id.clone(),
                redo: false,
            },
        };
        self.redo.push(entry);
        Some(step)
    }

    pub fn redo(&mut self, file_id: Option<FileId>) -> Option<UndoStep> {
        let index = self.redo.iter().rposition(|e| in_scope(e, file_id))?;
        let entry = self.redo.remove(index);
        let step = match &entry.action {
            UndoAction::Host(change) => UndoStep::Host {
                file_id: entry.file_id,
                change: change.clone(),
            },
            UndoAction::Plugin {
                plugin_id,
                entry_id,
            } => UndoStep::Plugin {
                plugin_id: Arc::clone(plugin_id),
                entry_id: entry_id.clone(),
                redo: true,
            },
        };
        self.undo.push(entry);
        Some(step)
    }

    /// Drops a plugin's entries (of one file, or all).
    pub fn clear_plugin(&mut self, plugin_id: &str, file_id: Option<FileId>) {
        let keep = |e: &UndoEntry| match &e.action {
            UndoAction::Plugin { plugin_id: owner, .. } => {
                &**owner != plugin_id || (file_id.is_some() && e.file_id != file_id)
            }
            UndoAction::Host(_) => true,
        };
        self.undo.retain(keep);
        self.redo.retain(keep);
    }

    /// Drops every entry of a closed file.
    pub fn clear_file(&mut self, file_id: FileId) {
        self.undo.retain(|e| e.file_id != Some(file_id));
        self.redo.retain(|e| e.file_id != Some(file_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::MeasureMode;
    use eov_plugin_api::protocol::Shape;

    fn plugin_entry(file: Option<FileId>, id: &str) -> UndoEntry {
        UndoEntry {
            file_id: file,
            label: id.into(),
            action: UndoAction::Plugin {
                plugin_id: Arc::from("anno"),
                entry_id: id.into(),
            },
        }
    }

    #[test]
    fn undo_is_scoped_to_the_viewed_slide() {
        let mut stack = UndoStack::default();
        stack.push(plugin_entry(Some(1), "a"));
        stack.push(plugin_entry(Some(2), "b"));
        assert_eq!(stack.undo_label(Some(1)), Some("a"));
        assert_eq!(
            stack.undo(Some(1)),
            Some(UndoStep::Plugin {
                plugin_id: Arc::from("anno"),
                entry_id: "a".into(),
                redo: false
            })
        );
        assert!(!stack.can_undo(Some(1)));
        assert!(stack.can_redo(Some(1)));
        assert!(stack.can_undo(Some(2)));
        // Redo brings it back.
        assert!(matches!(stack.redo(Some(1)), Some(UndoStep::Plugin { redo: true, .. })));
        assert!(stack.can_undo(Some(1)));
    }

    #[test]
    fn host_changes_are_inverted() {
        let mut stack = UndoStack::default();
        let roi = Rect {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };
        stack.push(UndoEntry {
            file_id: Some(1),
            label: "Set ROI".into(),
            action: UndoAction::Host(HostChange::Roi {
                before: None,
                after: Some(roi),
            }),
        });
        assert_eq!(
            stack.undo(Some(1)),
            Some(UndoStep::Host {
                file_id: Some(1),
                change: HostChange::Roi {
                    before: Some(roi),
                    after: None
                }
            })
        );
        let m = HostMeasurement {
            id: "line:1".into(),
            mode: MeasureMode::Line,
            shape: Shape::Point { x: 0.0, y: 0.0 },
        };
        stack.push(UndoEntry {
            file_id: Some(1),
            label: "Measure".into(),
            action: UndoAction::Host(HostChange::MeasurementAdded(m.clone())),
        });
        // A new change clears the slide's redo history.
        assert!(!stack.can_redo(Some(1)));
        assert_eq!(
            stack.undo(Some(1)),
            Some(UndoStep::Host {
                file_id: Some(1),
                change: HostChange::MeasurementsRemoved(vec![m])
            })
        );
    }

    #[test]
    fn clearing_removes_plugin_and_file_entries() {
        let mut stack = UndoStack::default();
        stack.push(plugin_entry(Some(1), "a"));
        stack.push(plugin_entry(Some(2), "b"));
        stack.clear_plugin("anno", Some(1));
        assert!(!stack.can_undo(Some(1)));
        assert!(stack.can_undo(Some(2)));
        stack.clear_file(2);
        assert!(!stack.can_undo(Some(2)));
    }
}
