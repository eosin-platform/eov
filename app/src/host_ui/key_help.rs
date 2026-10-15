//! The Keyboard Shortcuts panel (F1) and the keys menus and tooltips name
//! (AUDIT §14.1). The keys come from [`crate::shortcuts`].

use super::UiHost;
use crate::shortcuts::{self, ACTIONS, Action, Chord};
use crate::{ShortcutLabels, ShortcutRow};
use eov_plugin_api::protocol::NotifyLevel;
use slint::{ModelRc, VecModel};
use std::collections::BTreeMap;
use std::rc::Rc;

fn heading(label: &str) -> ShortcutRow {
    ShortcutRow {
        heading: true,
        label: label.into(),
        id: "".into(),
        keys: "".into(),
    }
}

fn row(label: &str, id: &str, keys: String) -> ShortcutRow {
    ShortcutRow {
        heading: false,
        label: label.into(),
        id: id.into(),
        keys: keys.into(),
    }
}

/// Sets the keys the window's menus and tooltips mention.
pub(crate) fn apply_labels(ui: &crate::AppWindow) {
    let table = shortcuts::current();
    ui.set_shortcut_labels(ShortcutLabels {
        undo: table.label(Action::Undo).into(),
        redo: table.label(Action::Redo).into(),
        navigate: table.label(Action::ToolNavigate).into(),
        roi: table.label(Action::ToolRoi).into(),
        measure: table.label(Action::ToolMeasure).into(),
        close_tab: table.label(Action::CloseTab).into(),
        slide_info: table.label(Action::SlideInfo).into(),
    });
}

impl UiHost {
    pub(crate) fn open_shortcuts_panel(&self) {
        let Some(ui) = self.ui() else {
            return;
        };
        let table = shortcuts::current();
        let mut rows = Vec::new();
        let mut group = None;
        for info in ACTIONS {
            if group != Some(info.group) {
                group = Some(info.group);
                rows.push(heading(info.group.label()));
            }
            rows.push(row(info.label, info.id, table.labels(info.action)));
        }
        let bindings = self.shared.contributions.lock().key_bindings();
        let mut by_plugin: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        for (plugin_id, label, hotkey) in bindings {
            let mut keys = Chord::from_hotkey(&hotkey).label();
            if hotkey.momentary {
                keys.push_str(" (hold)");
            }
            by_plugin
                .entry(self.plugin_name(&plugin_id))
                .or_default()
                .push((label, keys));
        }
        for (plugin, keys) in by_plugin {
            rows.push(heading(&format!("{plugin} (plugin)")));
            for (label, keys) in keys {
                rows.push(row(&label, "", keys));
            }
        }
        let path = crate::config::resolve_config_path()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "config.toml".into());
        ui.set_shortcuts_hint(
            format!(
                "To change keys, add a [shortcuts] table to {path} with the grey names: \
                 bookmark-add = \"Ctrl+D\", zoom-in = [\"+\", \"=\"], or [] for no key. \
                 \"Ctrl\" means ⌘ on macOS. Changes apply when eov starts; plugins cannot \
                 take eov's keys."
            )
            .into(),
        );
        ui.set_shortcut_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
        ui.set_shortcuts_panel_visible(true);
    }

    pub(crate) fn copy_config_path(&self) {
        let Ok(path) = crate::config::resolve_config_path() else {
            return;
        };
        let text = path.display().to_string();
        crate::clipboard::copy_text(text.clone());
        if let Some(ui) = self.ui() {
            super::notices::show_toast(&ui, &format!("Copied {text}"), NotifyLevel::Info);
        }
    }

    /// Tells the user about `[shortcuts]` entries that could not be used.
    pub(super) fn report_shortcut_problems(&self) {
        let problems = shortcuts::take_problems();
        if problems.is_empty() {
            return;
        }
        self.notify(
            NotifyLevel::Warning,
            "Some keyboard shortcuts in config.toml were not used",
            Some(&problems.join("\n")),
            None,
        );
    }
}
