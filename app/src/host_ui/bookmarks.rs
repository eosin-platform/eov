//! Bookmarks (AUDIT §14.1): Ctrl+B names the focused pane's view and keeps
//! it for the slide; Ctrl+Shift+B, or the viewport menu, lists the slide's
//! bookmarks to go to or delete. Storage: [`crate::bookmarks`].

use super::UiHost;
use crate::ContextMenuItem;
use crate::bookmarks::{self, Bookmark};
use crate::shortcuts::Action;
use crate::state::{AppState, PaneId};
use eov_plugin_api::i18n::{self, isolate, tr};
use eov_plugin_api::protocol::NotifyLevel;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::rc::Rc;

/// Bookmarks listed in one menu (the most recent ones).
const MENU_ITEMS: usize = 25;

/// A view waiting for its name in the Add Bookmark dialog.
pub(super) struct PendingBookmark {
    key: String,
    file_name: String,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    orientation: eov_common::viewport::Orientation,
}

/// What the bookmark menu was opened for.
pub(super) struct BookmarkMenu {
    file_id: i32,
    key: String,
    /// `bookmarks::list(key)` when the menu opened; indices refer to it.
    bookmarks: Vec<Bookmark>,
}

/// A number as eov_common writes it (no trailing zeros), with the decimal
/// separator of the interface's language.
fn localized(number: f64) -> String {
    let text = eov_common::format_decimal(number);
    let separator = i18n::current().decimal_separator;
    if separator == '.' {
        text
    } else {
        text.replacen('.', &separator.to_string(), 1)
    }
}

/// An objective-equivalent magnification as people write it ("2.5×").
fn magnification_text(magnification: f64) -> String {
    let rounded = if magnification < 10.0 {
        (magnification * 10.0).round() / 10.0
    } else {
        magnification.round()
    };
    format!("{}×", localized(rounded))
}

/// How a zoom reads: a magnification when the slide's is known.
fn scale_text(native_magnification: Option<f64>, zoom: f64) -> String {
    match native_magnification {
        Some(native) => magnification_text(native * zoom),
        // Translators: a percentage, e.g. "25 %" or "0.4 %".
        None => tr!("{} %", localized((zoom * 100.0).round())),
    }
}

/// The name a new bookmark gets when it is given none: "Bookmark 3" for a
/// slide that has `count` = 2 already.
fn default_name(count: usize) -> String {
    tr!("Bookmark {}", count + 1)
}

/// The slide shown in the pane keyboard commands act on, and that pane.
fn focused_slide(state: &AppState) -> Option<(i32, PaneId)> {
    let pane = state.active_viewport_pane();
    Some((state.active_file_id_for_pane(pane)?, pane))
}

fn slide_key(state: &AppState, file_id: i32) -> Result<String, String> {
    let file = state
        .get_file(file_id)
        .ok_or_else(|| tr!("The slide was closed"))?;
    file.wsi
        .identity()
        .map(|identity| identity.key().to_string())
        // Translators: eov could not work out which slide file this is (its
        // fingerprint, under which bookmarks are kept); not about the patient.
        .map_err(|error| tr!("This slide cannot be identified: {}", error))
}

impl UiHost {
    /// Ctrl+B: asks for a name for the focused pane's view.
    pub(crate) fn begin_bookmark(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let prepared = {
            let state = self.state.read();
            focused_slide(&state).map(|(file_id, pane)| {
                let file = state.get_file(file_id)?;
                let viewport = &file.pane_state(pane)?.viewport.viewport;
                let key = slide_key(&state, file_id);
                Some((
                    key,
                    PendingBookmark {
                        key: String::new(),
                        file_name: file.filename.clone(),
                        center_x: viewport.center.x,
                        center_y: viewport.center.y,
                        zoom: viewport.zoom,
                        orientation: viewport.orientation,
                    },
                    crate::callbacks::native_magnification(file),
                ))
            })
        };
        let Some(Some((key, mut pending, native))) = prepared else {
            self.status(&tr!("Open a slide to bookmark a view"));
            return;
        };
        pending.key = match key {
            Ok(key) => key,
            Err(error) => {
                self.notify(
                    NotifyLevel::Warning,
                    &tr!("Cannot add a bookmark"),
                    Some(&error),
                    None,
                );
                return;
            }
        };
        let count = bookmarks::list(&pending.key).len();
        if count >= bookmarks::MAX_PER_SLIDE {
            let key = crate::shortcuts::current().label(Action::Bookmarks);
            let max = bookmarks::MAX_PER_SLIDE;
            let detail = if key.is_empty() {
                tr!(
                    "A slide can have at most {n} bookmark. Delete some from the bookmark menu."
                        | "A slide can have at most {n} bookmarks. Delete some from the bookmark menu."
                            % max
                )
            } else {
                tr!(
                    "A slide can have at most {n} bookmark. Delete some from the bookmark menu ({key})."
                        | "A slide can have at most {n} bookmarks. Delete some from the bookmark menu ({key})."
                            % max,
                    key = isolate(&key)
                )
            };
            self.notify(
                NotifyLevel::Warning,
                &tr!("Cannot add a bookmark"),
                Some(&detail),
                None,
            );
            return;
        }
        let scale = scale_text(native, pending.zoom);
        ui.set_bookmark_dialog_name(default_name(count).into());
        ui.set_bookmark_dialog_detail(
            // Translators: where a bookmark looks: its magnification or
            // zoom, the point in the middle of the view (level-0 pixels)
            // and the slide's file name.
            tr!(
                "{scale} at {x}, {y} of {file}",
                scale = isolate(&scale),
                x = pending.center_x.round(),
                y = pending.center_y.round(),
                file = isolate(&pending.file_name)
            )
            .into(),
        );
        *self.pending_bookmark.borrow_mut() = Some(pending);
        ui.set_bookmark_dialog_visible(true);
    }

    /// The Add Bookmark dialog was confirmed with `name`.
    pub(crate) fn finish_bookmark(&self, name: &str) {
        if let Some(ui) = self.ui() {
            ui.set_bookmark_dialog_visible(false);
        }
        let Some(pending) = self.pending_bookmark.borrow_mut().take() else {
            return;
        };
        let name = name.trim();
        let name = if name.is_empty() {
            default_name(bookmarks::list(&pending.key).len())
        } else {
            name.chars().take(bookmarks::MAX_NAME_CHARS).collect()
        };
        let added = bookmarks::add(
            &pending.key,
            &pending.file_name,
            Bookmark {
                name,
                center_x: pending.center_x,
                center_y: pending.center_y,
                zoom: pending.zoom,
                rotation: pending.orientation.degrees(),
                flipped: pending.orientation.flipped(),
            },
        );
        match added {
            Ok(name) => {
                let key = crate::shortcuts::current().label(Action::Bookmarks);
                self.status(&if key.is_empty() {
                    tr!("Bookmarked “{}”", name)
                } else {
                    tr!(
                        "Bookmarked “{name}”; {key} lists the slide's bookmarks",
                        name = name,
                        key = isolate(&key)
                    )
                })
            }
            Err(error) => self.notify(
                NotifyLevel::Warning,
                &tr!("Cannot add a bookmark"),
                Some(&error),
                None,
            ),
        }
    }

    pub(crate) fn cancel_bookmark(&self) {
        self.pending_bookmark.borrow_mut().take();
        if let Some(ui) = self.ui() {
            ui.set_bookmark_dialog_visible(false);
        }
    }

    /// Whether slide `file_id` has bookmarks (for the viewport menu).
    pub(super) fn slide_has_bookmarks(&self, file_id: i32) -> bool {
        let state = self.state.read();
        slide_key(&state, file_id).is_ok_and(|key| !bookmarks::list(&key).is_empty())
    }

    /// Ctrl+Shift+B: lists the focused slide's bookmarks at window position
    /// (`x`, `y`); with `deleting`, choosing one deletes it.
    pub(crate) fn open_bookmark_menu(&self, x: f32, y: f32, deleting: bool) {
        let Some(ui) = self.ui() else {
            return;
        };
        let (file_id, key, native) = {
            let state = self.state.read();
            let Some((file_id, _)) = focused_slide(&state) else {
                self.status(&tr!("Open a slide to see its bookmarks"));
                return;
            };
            let native = state
                .get_file(file_id)
                .and_then(crate::callbacks::native_magnification);
            match slide_key(&state, file_id) {
                Ok(key) => (file_id, key, native),
                Err(error) => {
                    self.status(&error);
                    return;
                }
            }
        };
        let list = bookmarks::list(&key);
        if list.is_empty() {
            let key = crate::shortcuts::current().label(Action::BookmarkAdd);
            self.status(&if key.is_empty() {
                tr!("This slide has no bookmarks yet")
            } else {
                tr!(
                    "This slide has no bookmarks yet: {} adds one",
                    isolate(&key)
                )
            });
            return;
        }
        let first = list.len().saturating_sub(MENU_ITEMS);
        let mut items: Vec<ContextMenuItem> = list
            .iter()
            .enumerate()
            .skip(first)
            .map(|(index, bookmark)| ContextMenuItem {
                id: if deleting {
                    format!("host:bookmark-delete:{index}")
                } else {
                    format!("host:bookmark-go:{index}")
                }
                .into(),
                label: bookmark.name.clone().into(),
                icon: if deleting { "delete" } else { "" }.into(),
                shortcut: scale_text(native, bookmark.zoom).into(),
                enabled: true,
                separator_after: false,
            })
            .collect();
        if let Some(last) = items.last_mut() {
            last.separator_after = true;
        }
        items.push(ContextMenuItem {
            id: if deleting {
                "host:bookmarks"
            } else {
                "host:bookmarks-delete"
            }
            .into(),
            label: if deleting {
                tr!("Back to Bookmarks")
            } else {
                tr!("Delete a Bookmark…")
            }
            .into(),
            icon: "".into(),
            shortcut: "".into(),
            enabled: true,
            separator_after: false,
        });
        *self.bookmark_menu.borrow_mut() = Some(BookmarkMenu {
            file_id,
            key,
            bookmarks: list,
        });
        *self.menu.borrow_mut() = None;
        ui.set_context_menu_tab_id(-1);
        ui.set_context_menu_series_path("".into());
        ui.set_context_menu_x(x);
        ui.set_context_menu_y(y);
        ui.set_context_menu_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_context_menu_visible(true);
    }

    /// Where a keyboard-opened menu appears: at the pointer when it is over
    /// the window, otherwise near the window's top centre.
    pub(crate) fn keyboard_menu_position(&self) -> (f32, f32) {
        if let Some(position) = self.last_pointer.get() {
            return position;
        }
        self.ui()
            .map(|ui| {
                let size = ui.window().size().to_logical(ui.window().scale_factor());
                (size.width / 2.0 - 120.0, 80.0)
            })
            .unwrap_or((40.0, 80.0))
    }

    /// `host:bookmark…` menu commands; returns whether it was one.
    pub(super) fn bookmark_command(&self, command: &str) -> bool {
        if command == "bookmark-add" {
            self.begin_bookmark();
            return true;
        }
        if command == "bookmarks" || command == "bookmarks-delete" {
            let (x, y) = self.ui().map_or((40.0, 80.0), |ui| {
                (ui.get_context_menu_x(), ui.get_context_menu_y())
            });
            self.open_bookmark_menu(x, y, command == "bookmarks-delete");
            return true;
        }
        let (go, index) = if let Some(index) = command.strip_prefix("bookmark-go:") {
            (true, index)
        } else if let Some(index) = command.strip_prefix("bookmark-delete:") {
            (false, index)
        } else {
            return false;
        };
        let Ok(index) = index.parse::<usize>() else {
            return true;
        };
        let Some(menu) = self.bookmark_menu.borrow_mut().take() else {
            return true;
        };
        let Some(bookmark) = menu.bookmarks.get(index).cloned() else {
            return true;
        };
        if go {
            self.go_to_bookmark(menu.file_id, &bookmark);
            return true;
        }
        // Deleting by index is only safe if the list did not change since
        // the menu opened (another eov window may have edited it).
        let current = bookmarks::list(&menu.key);
        if current.get(index) != Some(&bookmark) {
            self.status(&tr!("The bookmarks changed; open the menu again"));
            return true;
        }
        match bookmarks::remove(&menu.key, index) {
            Ok(()) => self.status(&tr!("Deleted the bookmark “{}”", bookmark.name)),
            Err(error) => self.notify(
                NotifyLevel::Warning,
                &tr!("Cannot delete the bookmark"),
                Some(&error),
                None,
            ),
        }
        true
    }

    fn go_to_bookmark(&self, file_id: i32, bookmark: &Bookmark) {
        {
            let mut state = self.state.write();
            let pane = state.active_viewport_pane();
            if state.active_file_id_for_pane(pane) != Some(file_id) {
                // The menu's slide is no longer the pane's.
                drop(state);
                self.status(&tr!("The slide of that bookmark is no longer shown here"));
                return;
            }
            if let Some(viewport) = state.active_viewport_mut() {
                viewport.set_orientation(bookmark.orientation());
                viewport.smooth_set_center_zoom(
                    bookmark.center_x,
                    bookmark.center_y,
                    bookmark.zoom,
                );
            }
            state.mirror_locked_viewports_from_active();
            state.request_render();
        }
        self.request_render();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_read_like_objectives() {
        assert_eq!(magnification_text(2.5), "2.5×");
        assert_eq!(magnification_text(19.96), "20×");
        assert_eq!(magnification_text(1.25), "1.3×");
        assert_eq!(scale_text(Some(40.0), 0.5), "20×");
        assert_eq!(scale_text(None, 0.25), "25 %");
    }
}
