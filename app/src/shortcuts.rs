//! The host's keyboard shortcuts (AUDIT §14.1): one table of what eov does
//! and the keys for it.
//!
//! - Keys are written as in menus: `"Ctrl+Shift+B"`, `"Alt+ArrowLeft"`,
//!   `"PageDown"`, `"+"`. `Ctrl` means ⌘ on macOS.
//! - `config.toml`'s `[shortcuts]` table changes them per action:
//!   `bookmark-add = "Ctrl+D"`, `zoom-in = ["+", "="]`, `slide-info = []`
//!   (no key).
//! - The table is what the Keyboard Shortcuts panel (F1) lists and what
//!   plugin hotkeys may not take.

use eov_plugin_api::protocol::{Hotkey, key_name};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

/// What a shortcut does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Action {
    OpenFile,
    CloseTab,
    NextTab,
    PreviousTab,
    NextSlide,
    PreviousSlide,
    SlideInfo,
    CopyView,
    ExportView,
    Quit,
    Fit,
    FitTissue,
    ZoomIn,
    ZoomOut,
    PanLeft,
    PanRight,
    PanUp,
    PanDown,
    /// Objective-equivalent magnification; `None` is one screen pixel per
    /// slide pixel.
    Magnify(Option<Magnification>),
    RotateLeft,
    RotateRight,
    Flip,
    HistoryBack,
    HistoryForward,
    BookmarkAdd,
    Bookmarks,
    ToolNavigate,
    ToolRoi,
    ToolMeasure,
    Measurements,
    Undo,
    Redo,
    ToggleMinimap,
    FullScreen,
    Shortcuts,
}

/// The magnification presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Magnification {
    X1_25,
    X2_5,
    X5,
    X10,
    X20,
    X40,
}

impl Magnification {
    pub fn value(self) -> f64 {
        match self {
            Self::X1_25 => 1.25,
            Self::X2_5 => 2.5,
            Self::X5 => 5.0,
            Self::X10 => 10.0,
            Self::X20 => 20.0,
            Self::X40 => 40.0,
        }
    }
}

/// Sections of the Keyboard Shortcuts panel, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Group {
    Slides,
    Navigation,
    Tools,
    Edit,
    Window,
}

impl Group {
    pub fn label(self) -> &'static str {
        match self {
            Self::Slides => "Slides and tabs",
            Self::Navigation => "Navigation",
            Self::Tools => "Tools",
            Self::Edit => "Editing",
            Self::Window => "Window",
        }
    }
}

/// An action as the table and the panel describe it.
pub(crate) struct ActionInfo {
    pub action: Action,
    /// Its name in `[shortcuts]`.
    pub id: &'static str,
    pub label: &'static str,
    pub group: Group,
    /// Whether holding the key repeats it.
    pub repeats: bool,
    defaults: &'static [&'static str],
}

const fn info(
    action: Action,
    id: &'static str,
    label: &'static str,
    group: Group,
    repeats: bool,
    defaults: &'static [&'static str],
) -> ActionInfo {
    ActionInfo {
        action,
        id,
        label,
        group,
        repeats,
        defaults,
    }
}

use Action as A;
use Group as G;
use Magnification as M;

/// Every action, in the panel's order.
#[rustfmt::skip]
pub(crate) const ACTIONS: &[ActionInfo] = &[
    info(A::OpenFile, "open-file", "Open a slide", G::Slides, false, &["Ctrl+O"]),
    info(A::CloseTab, "close-tab", "Close the tab", G::Slides, false, &["Ctrl+W"]),
    info(A::NextTab, "next-tab", "Next tab in the pane", G::Slides, true, &["Ctrl+Tab"]),
    info(A::PreviousTab, "previous-tab", "Previous tab in the pane", G::Slides, true, &["Ctrl+Shift+Tab"]),
    info(A::NextSlide, "next-slide", "Next slide of the folder", G::Slides, false, &["PageDown"]),
    info(A::PreviousSlide, "previous-slide", "Previous slide of the folder", G::Slides, false, &["PageUp"]),
    info(A::SlideInfo, "slide-info", "Slide information", G::Slides, false, &["I"]),
    info(A::CopyView, "copy-view", "Copy the view as an image", G::Slides, false, &["Ctrl+C"]),
    info(A::ExportView, "export-view", "Export the view", G::Slides, false, &["Ctrl+E"]),
    info(A::Fit, "fit", "Fit the slide to the pane", G::Navigation, false, &["F"]),
    info(A::FitTissue, "fit-tissue", "Fit the tissue to the pane", G::Navigation, false, &["Shift+F"]),
    info(A::ZoomIn, "zoom-in", "Zoom in", G::Navigation, true, &["+", "="]),
    info(A::ZoomOut, "zoom-out", "Zoom out", G::Navigation, true, &["-"]),
    info(A::PanLeft, "pan-left", "Pan left (with Shift: further)", G::Navigation, true, &["ArrowLeft"]),
    info(A::PanRight, "pan-right", "Pan right (with Shift: further)", G::Navigation, true, &["ArrowRight"]),
    info(A::PanUp, "pan-up", "Pan up (with Shift: further)", G::Navigation, true, &["ArrowUp"]),
    info(A::PanDown, "pan-down", "Pan down (with Shift: further)", G::Navigation, true, &["ArrowDown"]),
    info(A::Magnify(Some(M::X1_25)), "zoom-1.25x", "Zoom to 1.25×", G::Navigation, false, &["1"]),
    info(A::Magnify(Some(M::X2_5)), "zoom-2.5x", "Zoom to 2.5×", G::Navigation, false, &["2"]),
    info(A::Magnify(Some(M::X5)), "zoom-5x", "Zoom to 5×", G::Navigation, false, &["3"]),
    info(A::Magnify(Some(M::X10)), "zoom-10x", "Zoom to 10×", G::Navigation, false, &["4"]),
    info(A::Magnify(Some(M::X20)), "zoom-20x", "Zoom to 20×", G::Navigation, false, &["5"]),
    info(A::Magnify(Some(M::X40)), "zoom-40x", "Zoom to 40×", G::Navigation, false, &["6"]),
    info(A::Magnify(None), "zoom-1-to-1", "Zoom to one screen pixel per slide pixel", G::Navigation, false, &["0"]),
    info(A::RotateLeft, "rotate-left", "Rotate the view 90° to the left", G::Navigation, false, &["["]),
    info(A::RotateRight, "rotate-right", "Rotate the view 90° to the right", G::Navigation, false, &["]"]),
    info(A::Flip, "flip", "Mirror the view left to right", G::Navigation, false, &["Shift+H"]),
    info(A::HistoryBack, "back", "Back to the previous view", G::Navigation, false, &["Alt+ArrowLeft"]),
    info(A::HistoryForward, "forward", "Forward to the next view", G::Navigation, false, &["Alt+ArrowRight"]),
    info(A::BookmarkAdd, "bookmark-add", "Bookmark the view", G::Navigation, false, &["Ctrl+B"]),
    info(A::Bookmarks, "bookmarks", "Bookmarks of the slide", G::Navigation, false, &["Ctrl+Shift+B"]),
    info(A::ToolNavigate, "tool-navigate", "Navigate", G::Tools, false, &["N"]),
    info(A::ToolRoi, "tool-roi", "Region of interest", G::Tools, false, &["R"]),
    info(A::ToolMeasure, "tool-measure", "Measure", G::Tools, false, &["D"]),
    info(A::Measurements, "measurements", "The slide's measurements", G::Tools, false, &["Ctrl+Shift+M"]),
    info(A::Undo, "undo", "Undo", G::Edit, true, &["Ctrl+Z"]),
    info(A::Redo, "redo", "Redo", G::Edit, true, &["Ctrl+Shift+Z", "Ctrl+Y"]),
    info(A::ToggleMinimap, "minimap", "Show or hide the minimap", G::Window, false, &["M"]),
    info(A::FullScreen, "full-screen", "Full screen", G::Window, false, &["F11"]),
    info(A::Shortcuts, "shortcuts", "Keyboard shortcuts", G::Window, false, &["F1", "?"]),
    info(A::Quit, "quit", "Quit eov", G::Window, false, &["Ctrl+Shift+W"]),
];

pub(crate) fn action_info(action: Action) -> &'static ActionInfo {
    ACTIONS
        .iter()
        .find(|info| info.action == action)
        .expect("every action is in the table")
}

/// A key with modifiers. `key` is a [`key_name`] ("b", "PageDown", "+").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Chord {
    pub key: String,
    /// Ctrl, or ⌘ on macOS.
    pub primary: bool,
    pub shift: bool,
    pub alt: bool,
}

/// Names people write for keys, beyond [`key_name`]'s own.
fn canonical_key(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    let named = match lower.as_str() {
        "left" => "ArrowLeft",
        "right" => "ArrowRight",
        "up" => "ArrowUp",
        "down" => "ArrowDown",
        "esc" => "Escape",
        "return" => "Enter",
        "del" => "Delete",
        "pgup" => "PageUp",
        "pgdn" | "pgdown" => "PageDown",
        "plus" => "+",
        "minus" => "-",
        "equal" | "equals" => "=",
        "arrowleft" => "ArrowLeft",
        "arrowright" => "ArrowRight",
        "arrowup" => "ArrowUp",
        "arrowdown" => "ArrowDown",
        "escape" => "Escape",
        "enter" => "Enter",
        "tab" => "Tab",
        "space" => "Space",
        "backspace" => "Backspace",
        "delete" => "Delete",
        "home" => "Home",
        "end" => "End",
        "pageup" => "PageUp",
        "pagedown" => "PageDown",
        _ => "",
    };
    if !named.is_empty() {
        return Some(named.to_string());
    }
    if let Some(number) = lower.strip_prefix('f')
        && let Ok(n) = number.parse::<u32>()
        && (1..=24).contains(&n)
    {
        return Some(format!("F{n}"));
    }
    let mut chars = name.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if !c.is_control() && !c.is_whitespace() => Some(key_name(name)),
        _ => None,
    }
}

impl Chord {
    /// Parses `"Ctrl+Shift+B"`, `"Alt+Left"`, `"F2"`, `"+"`, `"Ctrl++"`.
    pub fn parse(text: &str) -> Result<Chord, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("an empty key".to_string());
        }
        // The key is the last part; "+" itself is written "+" or "Ctrl++".
        let (modifiers, key) = if text == "+" {
            ("", "+")
        } else if let Some(rest) = text.strip_suffix("++") {
            (rest, "+")
        } else {
            match text.rsplit_once('+') {
                Some((modifiers, key)) => (modifiers, key),
                None => ("", text),
            }
        };
        let mut chord = Chord {
            key: canonical_key(key.trim()).ok_or_else(|| format!("unknown key '{key}'"))?,
            primary: false,
            shift: false,
            alt: false,
        };
        for modifier in modifiers
            .split('+')
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            match modifier.to_lowercase().as_str() {
                "ctrl" | "control" | "cmd" | "command" | "primary" | "⌘" => chord.primary = true,
                "shift" | "⇧" => chord.shift = true,
                "alt" | "option" | "opt" | "⌥" => chord.alt = true,
                other => return Err(format!("unknown modifier '{other}'")),
            }
        }
        Ok(chord)
    }

    /// As the platform writes it: "Ctrl+Shift+B", or "⇧⌘B" on macOS.
    pub fn label(&self) -> String {
        let key = match self.key.as_str() {
            "ArrowLeft" => "←".to_string(),
            "ArrowRight" => "→".to_string(),
            "ArrowUp" => "↑".to_string(),
            "ArrowDown" => "↓".to_string(),
            key if key.chars().count() == 1 => key.to_uppercase(),
            key => key.to_string(),
        };
        if cfg!(target_os = "macos") {
            format!(
                "{}{}{}{key}",
                if self.alt { "⌥" } else { "" },
                if self.shift { "⇧" } else { "" },
                if self.primary { "⌘" } else { "" },
            )
        } else {
            format!(
                "{}{}{}{key}",
                if self.primary { "Ctrl+" } else { "" },
                if self.alt { "Alt+" } else { "" },
                if self.shift { "Shift+" } else { "" },
            )
        }
    }

    /// A plugin's hotkey, to show it like eov's keys.
    pub fn from_hotkey(hotkey: &Hotkey) -> Chord {
        Chord {
            key: canonical_key(&hotkey.key).unwrap_or_else(|| hotkey.key.clone()),
            primary: hotkey.modifiers.primary || hotkey.modifiers.control,
            shift: hotkey.modifiers.shift,
            alt: hotkey.modifiers.alt,
        }
    }

    fn matches_key(&self, key: &str) -> bool {
        self.key.eq_ignore_ascii_case(key)
    }
}

/// Each action's keys, with the user's changes applied.
#[derive(Clone, Debug)]
pub(crate) struct Table {
    bindings: Vec<(Action, Vec<Chord>)>,
}

impl Default for Table {
    fn default() -> Self {
        let bindings = ACTIONS
            .iter()
            .map(|info| {
                let chords = info
                    .defaults
                    .iter()
                    .map(|text| Chord::parse(text).expect("default shortcuts parse"))
                    .collect();
                (info.action, chords)
            })
            .collect();
        Self { bindings }
    }
}

impl Table {
    /// The defaults with `[shortcuts]` entries (action id → keys) applied.
    /// Returns what could not be applied, for the user.
    pub fn with_overrides(overrides: &[(String, Vec<String>)]) -> (Table, Vec<String>) {
        let mut table = Table::default();
        let mut problems = Vec::new();
        // The user's keys, by action; they win over default keys of other
        // actions.
        let mut chosen: BTreeMap<usize, Vec<Chord>> = BTreeMap::new();
        for (id, keys) in overrides {
            let Some(index) = ACTIONS.iter().position(|info| info.id == id) else {
                problems.push(format!("[shortcuts] {id}: there is no such action"));
                continue;
            };
            let mut chords = Vec::new();
            for key in keys {
                match Chord::parse(key) {
                    Ok(chord) if is_controller_key(&chord) => problems.push(format!(
                        "[shortcuts] {id}: {key} is kept for the viewport and dialogs"
                    )),
                    Ok(chord) => chords.push(chord),
                    Err(error) => problems.push(format!("[shortcuts] {id}: {key}: {error}")),
                }
            }
            chosen.insert(index, chords);
        }
        for (index, chords) in &chosen {
            for chord in chords {
                for (other, other_chords) in &chosen {
                    if other > index && other_chords.contains(chord) {
                        problems.push(format!(
                            "[shortcuts] {} is set for both {} and {}; {} keeps it",
                            chord.label(),
                            ACTIONS[*index].id,
                            ACTIONS[*other].id,
                            ACTIONS[*other].id,
                        ));
                    }
                }
            }
        }
        for (index, chords) in chosen {
            // A key given to this action leaves every other action.
            for (_, others) in &mut table.bindings {
                others.retain(|chord| !chords.contains(chord));
            }
            table.bindings[index].1 = chords;
        }
        (table, problems)
    }

    /// The action bound to a key press. A key without Shift in the table
    /// also matches with Shift held (Shift+arrows pan further, and "+" or
    /// "?" need Shift on many layouts), unless Shift has its own binding.
    pub fn lookup(&self, key: &str, primary: bool, shift: bool, alt: bool) -> Option<Action> {
        let find = |shift: bool| {
            self.bindings.iter().find_map(|(action, chords)| {
                chords
                    .iter()
                    .any(|c| {
                        c.matches_key(key)
                            && c.primary == primary
                            && c.alt == alt
                            && c.shift == shift
                    })
                    .then_some(*action)
            })
        };
        find(shift).or_else(|| if shift { find(false) } else { None })
    }

    pub fn chords(&self, action: Action) -> &[Chord] {
        self.bindings
            .iter()
            .find(|(a, _)| *a == action)
            .map(|(_, chords)| chords.as_slice())
            .unwrap_or_default()
    }

    /// The first key of `action` as menus show it ("" when it has none).
    pub fn label(&self, action: Action) -> String {
        self.chords(action)
            .first()
            .map(Chord::label)
            .unwrap_or_default()
    }

    /// Every key of `action`, for the panel ("Ctrl+Shift+Z or Ctrl+Y").
    pub fn labels(&self, action: Action) -> String {
        self.chords(action)
            .iter()
            .map(Chord::label)
            .collect::<Vec<_>>()
            .join(" or ")
    }
}

/// Keys the viewport and dialogs keep, whatever the table says.
fn is_controller_key(chord: &Chord) -> bool {
    !chord.primary
        && !chord.alt
        && matches!(
            chord.key.as_str(),
            "Escape" | "Enter" | "Delete" | "Backspace" | "Tab"
        )
}

static CURRENT: RwLock<Option<Arc<Table>>> = RwLock::new(None);
/// What [`install_from_config`] could not use, until the window reports it.
static PROBLEMS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The table in use.
pub(crate) fn current() -> Arc<Table> {
    if let Some(table) = CURRENT.read().ok().and_then(|slot| slot.clone()) {
        return table;
    }
    let table = Arc::new(Table::default());
    if let Ok(mut slot) = CURRENT.write() {
        slot.get_or_insert_with(|| Arc::clone(&table));
    }
    table
}

/// Builds the table from `config.toml` and puts it in use. Entries that
/// could not be used are logged and kept for [`take_problems`].
pub(crate) fn install_from_config() {
    let (overrides, mut problems) = crate::config::load_shortcuts();
    let (table, more) = Table::with_overrides(&overrides);
    problems.extend(more);
    for problem in &problems {
        tracing::warn!("{problem}");
    }
    if let Ok(mut slot) = CURRENT.write() {
        *slot = Some(Arc::new(table));
    }
    if let Ok(mut slot) = PROBLEMS.lock() {
        *slot = problems;
    }
}

/// The problems of the last [`install_from_config`], once.
pub(crate) fn take_problems() -> Vec<String> {
    PROBLEMS
        .lock()
        .map(|mut problems| std::mem::take(&mut *problems))
        .unwrap_or_default()
}

/// Whether a plugin hotkey would collide with eov's own keys.
pub(crate) fn is_taken_by_host(key: &str, primary: bool, shift: bool, alt: bool) -> bool {
    let key = canonical_key(key).unwrap_or_else(|| key.to_string());
    let chord = Chord {
        key: key.clone(),
        primary,
        shift,
        alt,
    };
    is_controller_key(&chord)
        || (!primary && !alt && key.starts_with("Arrow"))
        || current().lookup(&key, primary, shift, alt).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chords_parse_and_print() {
        let chord = Chord::parse("ctrl+shift+b").unwrap();
        assert_eq!(
            chord,
            Chord {
                key: "b".into(),
                primary: true,
                shift: true,
                alt: false
            }
        );
        assert_eq!(Chord::parse("Ctrl++").unwrap().key, "+");
        assert_eq!(Chord::parse("+").unwrap().key, "+");
        assert_eq!(Chord::parse("Alt+Left").unwrap().key, "ArrowLeft");
        assert_eq!(Chord::parse("pgdn").unwrap().key, "PageDown");
        assert_eq!(Chord::parse("f11").unwrap().key, "F11");
        assert!(Chord::parse("Hyper+X").is_err());
        assert!(Chord::parse("Ctrl+Banana").is_err());
        if !cfg!(target_os = "macos") {
            assert_eq!(chord.label(), "Ctrl+Shift+B");
            assert_eq!(Chord::parse("Alt+Left").unwrap().label(), "Alt+←");
        }
        // Every default parses (Table::default would panic otherwise).
        let table = Table::default();
        assert_eq!(table.labels(Action::Redo).matches(" or ").count(), 1);
    }

    #[test]
    fn lookup_prefers_the_exact_shift_state() {
        let table = Table::default();
        assert_eq!(table.lookup("z", true, false, false), Some(Action::Undo));
        assert_eq!(table.lookup("z", true, true, false), Some(Action::Redo));
        // Shift+arrow pans further; "+" usually needs Shift.
        assert_eq!(
            table.lookup("ArrowLeft", false, true, false),
            Some(Action::PanLeft)
        );
        assert_eq!(table.lookup("+", false, true, false), Some(Action::ZoomIn));
        assert_eq!(
            table.lookup("?", false, true, false),
            Some(Action::Shortcuts)
        );
        assert_eq!(
            table.lookup("ArrowLeft", false, false, true),
            Some(Action::HistoryBack)
        );
        assert_eq!(table.lookup("b", false, false, false), None);
        assert_eq!(table.lookup("7", false, false, false), None);
    }

    #[test]
    fn overrides_move_keys_between_actions() {
        let overrides = vec![
            ("minimap".to_string(), vec!["F".to_string()]),
            ("slide-info".to_string(), vec![]),
            (
                "bookmark-add".to_string(),
                vec!["Ctrl+D".to_string(), "Nope+X".to_string()],
            ),
            ("no-such-action".to_string(), vec!["X".to_string()]),
            ("fit".to_string(), vec!["Escape".to_string()]),
        ];
        let (table, problems) = Table::with_overrides(&overrides);
        assert_eq!(
            table.lookup("f", false, false, false),
            Some(Action::ToggleMinimap)
        );
        assert_eq!(table.lookup("m", false, false, false), None);
        assert_eq!(table.lookup("i", false, false, false), None);
        assert_eq!(
            table.lookup("d", true, false, false),
            Some(Action::BookmarkAdd)
        );
        assert_eq!(table.lookup("b", true, false, false), None);
        assert_eq!(problems.len(), 3, "{problems:?}");
        // Fit lost F to the minimap and its own new key was refused.
        assert!(table.chords(Action::Fit).is_empty());
    }

    #[test]
    fn plugins_cannot_take_host_keys() {
        assert!(is_taken_by_host("F", false, false, false));
        assert!(is_taken_by_host("f", false, true, false));
        assert!(is_taken_by_host("b", true, false, false));
        assert!(is_taken_by_host("escape", false, false, false));
        assert!(is_taken_by_host("ArrowUp", false, false, false));
        assert!(!is_taken_by_host("7", false, false, false));
        assert!(!is_taken_by_host("q", false, false, false));
        assert!(!is_taken_by_host("ArrowUp", false, false, true));
    }
}
