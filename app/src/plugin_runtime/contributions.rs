//! Toolbar buttons, HUD buttons, tools and filters contributed by plugins.
//!
//! Ids are scoped per plugin, registrations are validated, ordering is
//! deterministic (plugin id, declared order, registration order — PLG-16),
//! and hotkeys are checked against the host's shortcuts and each other.

use eov_plugin_api::protocol::{
    ButtonUpdate, FilterSpec, Hotkey, HotkeySpec, HudButtonSpec, MenuItem, PaneId, ToolSpec,
    ToolbarButtonSpec,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Keys the host binds without modifiers (case-insensitive).
pub const RESERVED_PLAIN_KEYS: &[&str] = &[
    "f", "m", "n", "r", "d", "i", "+", "=", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "f11", "pageup", "pagedown",
];
/// Keys the host binds with the primary modifier (Ctrl, or ⌘ on macOS).
pub const RESERVED_PRIMARY_KEYS: &[&str] = &[
    "w", "z", "y", "o", "c", "e", "s", "b", "tab", "+", "=", "-", "0",
];

/// Largest inline SVG accepted for an icon.
const MAX_ICON_BYTES: usize = 64 * 1024;

fn validate_id(kind: &str, id: &str) -> Result<(), String> {
    let ok = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{kind} id '{id}' must be 1–64 characters of letters, digits, '_', '-' or '.'"
        ))
    }
}

fn validate_icon(svg: &str) -> Result<(), String> {
    if svg.len() > MAX_ICON_BYTES {
        return Err(format!("icon SVG is larger than {MAX_ICON_BYTES} bytes"));
    }
    Ok(())
}

fn validate_menu(menu: &[MenuItem]) -> Result<(), String> {
    for item in menu {
        validate_id("menu item", &item.item_id)?;
    }
    if menu.len() > 64 {
        return Err("menus are limited to 64 items".to_string());
    }
    Ok(())
}

/// Normalises a hotkey for comparison.
fn hotkey_signature(hotkey: &Hotkey) -> (String, bool, bool, bool) {
    (
        hotkey.key.to_lowercase(),
        hotkey.modifiers.primary || hotkey.modifiers.control,
        hotkey.modifiers.shift,
        hotkey.modifiers.alt,
    )
}

/// Whether `hotkey` collides with a host shortcut.
pub fn is_reserved(hotkey: &Hotkey) -> bool {
    let (key, primary, _shift, alt) = hotkey_signature(hotkey);
    if alt {
        // Alt+←/→ step through the pane's navigation history.
        return !primary && matches!(key.as_str(), "arrowleft" | "arrowright");
    }
    if primary {
        RESERVED_PRIMARY_KEYS.contains(&key.as_str())
    } else {
        RESERVED_PLAIN_KEYS.contains(&key.as_str())
            || matches!(
                key.as_str(),
                "escape" | "enter" | "return" | "delete" | "backspace" | "tab"
            )
            || key.starts_with("arrow")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolbarEntry {
    pub plugin_id: Arc<str>,
    pub spec: ToolbarButtonSpec,
    pub seq: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaneButtonState {
    pub active: Option<bool>,
    pub menu: Option<Vec<MenuItem>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HudEntry {
    pub plugin_id: Arc<str>,
    pub spec: HudButtonSpec,
    pub per_pane: HashMap<PaneId, PaneButtonState>,
    pub seq: u64,
}

impl HudEntry {
    pub fn active_in(&self, pane: PaneId) -> bool {
        self.per_pane
            .get(&pane)
            .and_then(|state| state.active)
            .unwrap_or(self.spec.active)
    }

    pub fn menu_in(&self, pane: PaneId) -> &[MenuItem] {
        self.per_pane
            .get(&pane)
            .and_then(|state| state.menu.as_deref())
            .unwrap_or(&self.spec.menu)
    }
}

/// A plugin key binding of its own.
#[derive(Clone, Debug, PartialEq)]
pub struct HotkeyEntry {
    pub plugin_id: Arc<str>,
    pub spec: HotkeySpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolEntry {
    pub plugin_id: Arc<str>,
    pub spec: ToolSpec,
    pub seq: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterEntry {
    pub plugin_id: Arc<str>,
    pub spec: FilterSpec,
    pub seq: u64,
}

/// Every plugin contribution. Lives in `HostShared` behind a mutex.
#[derive(Clone, Debug, Default)]
pub struct Contributions {
    toolbar: Vec<ToolbarEntry>,
    hud: Vec<HudEntry>,
    tools: Vec<ToolEntry>,
    hotkeys: Vec<HotkeyEntry>,
    filters: Vec<FilterEntry>,
    seq: u64,
    revision: u64,
}

impl Contributions {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.revision += 1;
        self.seq
    }

    /// Whether another contribution already uses `hotkey`.
    fn hotkey_taken(&self, plugin_id: &str, own_id: &str, hotkey: &Hotkey) -> Option<String> {
        let wanted = hotkey_signature(hotkey);
        let toolbar = self
            .toolbar
            .iter()
            .filter_map(|e| e.spec.hotkey.as_ref().map(|h| (e, &e.spec.button_id, h)))
            .map(|(e, id, h)| (&e.plugin_id, id, h));
        let tools = self.tools.iter().filter_map(|e| {
            e.spec
                .hotkey
                .as_ref()
                .map(|h| (&e.plugin_id, &e.spec.tool_id, h))
        });
        let hotkeys = self
            .hotkeys
            .iter()
            .map(|e| (&e.plugin_id, &e.spec.hotkey_id, &e.spec.hotkey));
        toolbar
            .chain(tools)
            .chain(hotkeys)
            .find(|(owner, id, h)| {
                !(&***owner == plugin_id && id.as_str() == own_id) && hotkey_signature(h) == wanted
            })
            .map(|(owner, id, _)| format!("{owner}:{id}"))
    }

    /// Drops a conflicting hotkey, returning a warning for the user.
    fn vet_hotkey(
        &self,
        plugin_id: &str,
        own_id: &str,
        hotkey: &mut Option<Hotkey>,
    ) -> Option<String> {
        let candidate = hotkey.as_ref()?;
        let problem = if candidate.key.trim().is_empty() {
            Some("an empty key".to_string())
        } else if is_reserved(candidate) {
            Some("a key eov already uses".to_string())
        } else {
            self.hotkey_taken(plugin_id, own_id, candidate)
                .map(|other| format!("the same key as {other}"))
        };
        problem.map(|problem| {
            let key = hotkey.take().map(|h| h.key).unwrap_or_default();
            format!("Hotkey '{key}' of {plugin_id}:{own_id} was ignored: it is {problem}.")
        })
    }

    // -- toolbar ---------------------------------------------------------------

    /// Adds or replaces a toolbar button. Returns a warning if its hotkey was
    /// dropped.
    pub fn add_toolbar(
        &mut self,
        plugin_id: &Arc<str>,
        mut spec: ToolbarButtonSpec,
    ) -> Result<Option<String>, String> {
        validate_id("button", &spec.button_id)?;
        validate_icon(&spec.icon_svg)?;
        if let Some(icon) = &spec.toggled_icon_svg {
            validate_icon(icon)?;
        }
        if self
            .toolbar
            .iter()
            .filter(|e| e.plugin_id == *plugin_id)
            .count()
            >= 32
            && !self
                .toolbar
                .iter()
                .any(|e| e.plugin_id == *plugin_id && e.spec.button_id == spec.button_id)
        {
            return Err("plugins may add at most 32 toolbar buttons".to_string());
        }
        let warning = self.vet_hotkey(plugin_id, &spec.button_id, &mut spec.hotkey);
        let seq = self.next_seq();
        match self
            .toolbar
            .iter_mut()
            .find(|e| e.plugin_id == *plugin_id && e.spec.button_id == spec.button_id)
        {
            Some(entry) => entry.spec = spec,
            None => self.toolbar.push(ToolbarEntry {
                plugin_id: Arc::clone(plugin_id),
                spec,
                seq,
            }),
        }
        Ok(warning)
    }

    pub fn update_toolbar(&mut self, plugin_id: &str, update: &ButtonUpdate) -> Result<(), String> {
        if let Some(icon) = &update.icon_svg {
            validate_icon(icon)?;
        }
        let entry = self
            .toolbar
            .iter_mut()
            .find(|e| &*e.plugin_id == plugin_id && e.spec.button_id == update.button_id)
            .ok_or_else(|| format!("unknown toolbar button '{}'", update.button_id))?;
        if let Some(active) = update.active {
            entry.spec.active = active;
        }
        if let Some(enabled) = update.enabled {
            entry.spec.enabled = enabled;
        }
        if let Some(tooltip) = &update.tooltip {
            entry.spec.tooltip = tooltip.clone();
        }
        if let Some(icon) = &update.icon_svg {
            entry.spec.icon_svg = icon.clone();
        }
        self.revision += 1;
        Ok(())
    }

    pub fn remove_toolbar(&mut self, plugin_id: &str, button_id: &str) -> bool {
        let before = self.toolbar.len();
        self.toolbar
            .retain(|e| !(&*e.plugin_id == plugin_id && e.spec.button_id == button_id));
        let removed = self.toolbar.len() != before;
        if removed {
            self.revision += 1;
        }
        removed
    }

    /// Toolbar buttons in display order.
    pub fn toolbar(&self) -> Vec<&ToolbarEntry> {
        let mut entries: Vec<_> = self.toolbar.iter().collect();
        entries.sort_by(|a, b| {
            a.plugin_id
                .cmp(&b.plugin_id)
                .then(a.spec.order.cmp(&b.spec.order))
                .then(a.seq.cmp(&b.seq))
        });
        entries
    }

    #[cfg(test)]
    pub fn toolbar_button(&self, plugin_id: &str, button_id: &str) -> Option<&ToolbarEntry> {
        self.toolbar
            .iter()
            .find(|e| &*e.plugin_id == plugin_id && e.spec.button_id == button_id)
    }

    // -- HUD -------------------------------------------------------------------

    pub fn add_hud(&mut self, plugin_id: &Arc<str>, spec: HudButtonSpec) -> Result<(), String> {
        validate_id("HUD button", &spec.button_id)?;
        validate_icon(&spec.icon_svg)?;
        validate_menu(&spec.menu)?;
        let seq = self.next_seq();
        match self
            .hud
            .iter_mut()
            .find(|e| e.plugin_id == *plugin_id && e.spec.button_id == spec.button_id)
        {
            Some(entry) => entry.spec = spec,
            None => self.hud.push(HudEntry {
                plugin_id: Arc::clone(plugin_id),
                spec,
                per_pane: HashMap::new(),
                seq,
            }),
        }
        Ok(())
    }

    pub fn update_hud(&mut self, plugin_id: &str, update: &ButtonUpdate) -> Result<(), String> {
        if let Some(menu) = &update.menu {
            validate_menu(menu)?;
        }
        if let Some(icon) = &update.icon_svg {
            validate_icon(icon)?;
        }
        let entry = self
            .hud
            .iter_mut()
            .find(|e| &*e.plugin_id == plugin_id && e.spec.button_id == update.button_id)
            .ok_or_else(|| format!("unknown HUD button '{}'", update.button_id))?;
        match update.pane {
            Some(pane) => {
                let state = entry.per_pane.entry(pane).or_default();
                if update.active.is_some() {
                    state.active = update.active;
                }
                if update.menu.is_some() {
                    state.menu = update.menu.clone();
                }
            }
            None => {
                if let Some(active) = update.active {
                    entry.spec.active = active;
                    // A global update overrides per-pane state.
                    for state in entry.per_pane.values_mut() {
                        state.active = None;
                    }
                }
                if let Some(menu) = &update.menu {
                    entry.spec.menu = menu.clone();
                }
                if let Some(tooltip) = &update.tooltip {
                    entry.spec.tooltip = tooltip.clone();
                }
                if let Some(icon) = &update.icon_svg {
                    entry.spec.icon_svg = icon.clone();
                }
            }
        }
        self.revision += 1;
        Ok(())
    }

    pub fn remove_hud(&mut self, plugin_id: &str, button_id: &str) -> bool {
        let before = self.hud.len();
        self.hud
            .retain(|e| !(&*e.plugin_id == plugin_id && e.spec.button_id == button_id));
        let removed = self.hud.len() != before;
        if removed {
            self.revision += 1;
        }
        removed
    }

    pub fn hud(&self) -> Vec<&HudEntry> {
        let mut entries: Vec<_> = self.hud.iter().collect();
        entries.sort_by(|a, b| {
            a.plugin_id
                .cmp(&b.plugin_id)
                .then(a.spec.order.cmp(&b.spec.order))
                .then(a.seq.cmp(&b.seq))
        });
        entries
    }

    /// Forgets per-pane state of a pane that was closed.
    pub fn forget_pane(&mut self, pane: PaneId) {
        for entry in &mut self.hud {
            entry.per_pane.remove(&pane);
        }
    }

    // -- tools -----------------------------------------------------------------

    pub fn register_tool(
        &mut self,
        plugin_id: &Arc<str>,
        mut spec: ToolSpec,
    ) -> Result<Option<String>, String> {
        validate_id("tool", &spec.tool_id)?;
        validate_icon(&spec.icon_svg)?;
        if !spec.freehand_tolerance_px.is_finite() || spec.freehand_tolerance_px < 0.0 {
            return Err("freehand tolerance must be finite and non-negative".to_string());
        }
        if self
            .tools
            .iter()
            .filter(|e| e.plugin_id == *plugin_id)
            .count()
            >= 16
            && !self
                .tools
                .iter()
                .any(|e| e.plugin_id == *plugin_id && e.spec.tool_id == spec.tool_id)
        {
            return Err("plugins may register at most 16 tools".to_string());
        }
        let warning = self.vet_hotkey(plugin_id, &spec.tool_id, &mut spec.hotkey);
        let seq = self.next_seq();
        match self
            .tools
            .iter_mut()
            .find(|e| e.plugin_id == *plugin_id && e.spec.tool_id == spec.tool_id)
        {
            Some(entry) => entry.spec = spec,
            None => self.tools.push(ToolEntry {
                plugin_id: Arc::clone(plugin_id),
                spec,
                seq,
            }),
        }
        Ok(warning)
    }

    pub fn unregister_tool(&mut self, plugin_id: &str, tool_id: &str) -> bool {
        let before = self.tools.len();
        self.tools
            .retain(|e| !(&*e.plugin_id == plugin_id && e.spec.tool_id == tool_id));
        let removed = self.tools.len() != before;
        if removed {
            self.revision += 1;
        }
        removed
    }

    pub fn tool(&self, plugin_id: &str, tool_id: &str) -> Option<&ToolEntry> {
        self.tools
            .iter()
            .find(|e| &*e.plugin_id == plugin_id && e.spec.tool_id == tool_id)
    }

    pub fn tools(&self) -> Vec<&ToolEntry> {
        let mut entries: Vec<_> = self.tools.iter().collect();
        entries.sort_by(|a, b| {
            a.plugin_id
                .cmp(&b.plugin_id)
                .then(a.spec.order.cmp(&b.spec.order))
                .then(a.seq.cmp(&b.seq))
        });
        entries
    }

    /// The tool or toolbar button bound to a key press.
    pub fn hotkey_target(
        &self,
        key: &str,
        primary: bool,
        shift: bool,
        alt: bool,
    ) -> Option<HotkeyTarget> {
        let wanted = (key.to_lowercase(), primary, shift, alt);
        if let Some(tool) = self.tools().into_iter().find(|e| {
            e.spec
                .hotkey
                .as_ref()
                .is_some_and(|h| hotkey_signature(h) == wanted)
        }) {
            return Some(HotkeyTarget::Tool {
                plugin_id: Arc::clone(&tool.plugin_id),
                tool_id: tool.spec.tool_id.clone(),
                momentary: tool.spec.hotkey.as_ref().is_some_and(|h| h.momentary),
            });
        }
        if let Some(target) = self.toolbar().into_iter().find_map(|e| {
            e.spec
                .hotkey
                .as_ref()
                .filter(|h| hotkey_signature(h) == wanted)
                .map(|_| HotkeyTarget::Button {
                    plugin_id: Arc::clone(&e.plugin_id),
                    button_id: e.spec.button_id.clone(),
                })
        }) {
            return Some(target);
        }
        self.hotkeys
            .iter()
            .find(|e| hotkey_signature(&e.spec.hotkey) == wanted)
            .map(|e| HotkeyTarget::Hotkey {
                plugin_id: Arc::clone(&e.plugin_id),
                hotkey_id: e.spec.hotkey_id.clone(),
            })
    }

    // -- hotkeys -------------------------------------------------------------------

    /// Adds or replaces a plugin key binding. A key that is reserved or taken
    /// is refused with the reason (nothing is registered then).
    pub fn register_hotkey(
        &mut self,
        plugin_id: &Arc<str>,
        spec: HotkeySpec,
    ) -> Result<Option<String>, String> {
        validate_id("hotkey", &spec.hotkey_id)?;
        let exists =
            |e: &HotkeyEntry| e.plugin_id == *plugin_id && e.spec.hotkey_id == spec.hotkey_id;
        if !self.hotkeys.iter().any(exists)
            && self
                .hotkeys
                .iter()
                .filter(|e| e.plugin_id == *plugin_id)
                .count()
                >= 64
        {
            return Err("plugins may register at most 64 hotkeys".to_string());
        }
        let mut hotkey = Some(spec.hotkey.clone());
        if let Some(warning) = self.vet_hotkey(plugin_id, &spec.hotkey_id, &mut hotkey) {
            return Ok(Some(warning));
        }
        self.revision += 1;
        match self.hotkeys.iter_mut().find(|e| exists(e)) {
            Some(entry) => entry.spec = spec,
            None => self.hotkeys.push(HotkeyEntry {
                plugin_id: Arc::clone(plugin_id),
                spec,
            }),
        }
        Ok(None)
    }

    pub fn unregister_hotkey(&mut self, plugin_id: &str, hotkey_id: &str) -> bool {
        let before = self.hotkeys.len();
        self.hotkeys
            .retain(|e| !(&*e.plugin_id == plugin_id && e.spec.hotkey_id == hotkey_id));
        self.revision += 1;
        before != self.hotkeys.len()
    }

    // -- filters -----------------------------------------------------------------

    pub fn register_filter(
        &mut self,
        plugin_id: &Arc<str>,
        spec: FilterSpec,
    ) -> Result<(), String> {
        validate_id("filter", &spec.filter_id)?;
        let seq = self.next_seq();
        match self
            .filters
            .iter_mut()
            .find(|e| e.plugin_id == *plugin_id && e.spec.filter_id == spec.filter_id)
        {
            Some(entry) => entry.spec = spec,
            None => self.filters.push(FilterEntry {
                plugin_id: Arc::clone(plugin_id),
                spec,
                seq,
            }),
        }
        Ok(())
    }

    /// Returns whether the state changed.
    pub fn set_filter_enabled(
        &mut self,
        plugin_id: &str,
        filter_id: &str,
        enabled: bool,
    ) -> Result<bool, String> {
        let entry = self
            .filters
            .iter_mut()
            .find(|e| &*e.plugin_id == plugin_id && e.spec.filter_id == filter_id)
            .ok_or_else(|| format!("unknown filter '{filter_id}'"))?;
        if entry.spec.enabled == enabled {
            return Ok(false);
        }
        entry.spec.enabled = enabled;
        self.revision += 1;
        Ok(true)
    }

    pub fn unregister_filter(&mut self, plugin_id: &str, filter_id: &str) -> bool {
        let before = self.filters.len();
        self.filters
            .retain(|e| !(&*e.plugin_id == plugin_id && e.spec.filter_id == filter_id));
        let removed = self.filters.len() != before;
        if removed {
            self.revision += 1;
        }
        removed
    }

    /// Filters in execution order: priority, then plugin id, then filter id.
    pub fn filters(&self) -> Vec<&FilterEntry> {
        let mut entries: Vec<_> = self.filters.iter().collect();
        entries.sort_by(|a, b| {
            a.spec
                .priority
                .cmp(&b.spec.priority)
                .then(a.plugin_id.cmp(&b.plugin_id))
                .then(a.spec.filter_id.cmp(&b.spec.filter_id))
        });
        entries
    }

    /// Removes everything a plugin contributed.
    pub fn remove_plugin(&mut self, plugin_id: &str) {
        self.toolbar.retain(|e| &*e.plugin_id != plugin_id);
        self.hud.retain(|e| &*e.plugin_id != plugin_id);
        self.tools.retain(|e| &*e.plugin_id != plugin_id);
        self.hotkeys.retain(|e| &*e.plugin_id != plugin_id);
        self.filters.retain(|e| &*e.plugin_id != plugin_id);
        self.revision += 1;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotkeyTarget {
    Tool {
        plugin_id: Arc<str>,
        tool_id: String,
        momentary: bool,
    },
    Button {
        plugin_id: Arc<str>,
        button_id: String,
    },
    Hotkey {
        plugin_id: Arc<str>,
        hotkey_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::protocol::{Modifiers, ToolKind};

    fn plugin(id: &str) -> Arc<str> {
        Arc::from(id)
    }

    fn button(id: &str, order: i32, hotkey: Option<&str>) -> ToolbarButtonSpec {
        ToolbarButtonSpec {
            button_id: id.into(),
            tooltip: id.into(),
            icon_svg: "<svg/>".into(),
            order,
            hotkey: hotkey.map(Hotkey::key),
            ..Default::default()
        }
    }

    #[test]
    fn ordering_is_deterministic() {
        let mut c = Contributions::default();
        c.add_toolbar(&plugin("zeta"), button("a", 0, None))
            .unwrap();
        c.add_toolbar(&plugin("alpha"), button("second", 2, None))
            .unwrap();
        c.add_toolbar(&plugin("alpha"), button("first", 1, None))
            .unwrap();
        let ids: Vec<_> = c
            .toolbar()
            .iter()
            .map(|e| e.spec.button_id.clone())
            .collect();
        assert_eq!(ids, ["first", "second", "a"]);
    }

    #[test]
    fn ids_are_scoped_per_plugin() {
        let mut c = Contributions::default();
        c.add_toolbar(&plugin("a"), button("toggle", 0, None))
            .unwrap();
        c.add_toolbar(&plugin("b"), button("toggle", 0, None))
            .unwrap();
        c.update_toolbar(
            "a",
            &ButtonUpdate {
                button_id: "toggle".into(),
                active: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(c.toolbar_button("a", "toggle").unwrap().spec.active);
        assert!(!c.toolbar_button("b", "toggle").unwrap().spec.active);
        assert!(
            c.add_toolbar(&plugin("a"), button("bad id", 0, None))
                .is_err()
        );
    }

    #[test]
    fn hotkey_conflicts_are_rejected_with_a_warning() {
        let mut c = Contributions::default();
        // Reserved host key.
        let warning = c
            .add_toolbar(&plugin("a"), button("x", 0, Some("f")))
            .unwrap();
        assert!(warning.unwrap().contains("eov already uses"));
        assert!(c.toolbar_button("a", "x").unwrap().spec.hotkey.is_none());
        // First come, first served; case-insensitive.
        assert!(
            c.add_toolbar(&plugin("a"), button("p", 0, Some("p")))
                .unwrap()
                .is_none()
        );
        let warning = c
            .register_tool(
                &plugin("b"),
                ToolSpec::new("t", "T", "<svg/>", ToolKind::Point).with_hotkey(Hotkey::key("P")),
            )
            .unwrap();
        assert!(warning.unwrap().contains("a:p"));
        // Modifiers make keys distinct.
        let warning = c
            .register_tool(
                &plugin("b"),
                ToolSpec::new("t", "T", "<svg/>", ToolKind::Point).with_hotkey(Hotkey {
                    key: "p".into(),
                    modifiers: Modifiers {
                        alt: true,
                        ..Default::default()
                    },
                    momentary: false,
                }),
            )
            .unwrap();
        assert!(warning.is_none());
        assert_eq!(
            c.hotkey_target("P", false, false, false),
            Some(HotkeyTarget::Button {
                plugin_id: plugin("a"),
                button_id: "p".into()
            })
        );
    }

    #[test]
    fn plugin_hotkeys_are_vetted_and_dispatched() {
        let mut c = Contributions::default();
        let spec = |id: &str, key: &str| HotkeySpec {
            hotkey_id: id.into(),
            hotkey: Hotkey::key(key),
            description: String::new(),
        };
        assert!(
            c.register_hotkey(&plugin("a"), spec("next", "j"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            c.hotkey_target("J", false, false, false),
            Some(HotkeyTarget::Hotkey {
                plugin_id: plugin("a"),
                hotkey_id: "next".into()
            })
        );
        // Taken and reserved keys are refused, with the reason.
        let refused = c.register_hotkey(&plugin("b"), spec("mine", "j")).unwrap();
        assert!(refused.unwrap().contains("a:next"));
        assert!(
            c.register_hotkey(&plugin("b"), spec("fit", "f"))
                .unwrap()
                .is_some()
        );
        assert!(c.unregister_hotkey("a", "next"));
        assert_eq!(c.hotkey_target("j", false, false, false), None);
    }

    #[test]
    fn filters_run_by_priority_then_plugin() {
        let mut c = Contributions::default();
        let spec = |id: &str, priority| FilterSpec {
            filter_id: id.into(),
            name: id.into(),
            priority,
            enabled: true,
        };
        c.register_filter(&plugin("b"), spec("x", 0)).unwrap();
        c.register_filter(&plugin("a"), spec("y", 0)).unwrap();
        c.register_filter(&plugin("c"), spec("z", -1)).unwrap();
        let order: Vec<_> = c
            .filters()
            .iter()
            .map(|e| format!("{}:{}", e.plugin_id, e.spec.filter_id))
            .collect();
        assert_eq!(order, ["c:z", "a:y", "b:x"]);
        assert!(!c.set_filter_enabled("a", "y", true).unwrap());
        assert!(c.set_filter_enabled("a", "y", false).unwrap());
    }

    #[test]
    fn hud_state_can_differ_per_pane() {
        let mut c = Contributions::default();
        c.add_hud(
            &plugin("vae"),
            HudButtonSpec {
                button_id: "mode".into(),
                tooltip: "Mode".into(),
                icon_svg: "<svg/>".into(),
                ..Default::default()
            },
        )
        .unwrap();
        c.update_hud(
            "vae",
            &ButtonUpdate {
                button_id: "mode".into(),
                pane: Some(2),
                active: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        let entry = c.hud()[0];
        assert!(entry.active_in(2));
        assert!(!entry.active_in(1));
        c.remove_plugin("vae");
        assert!(c.hud().is_empty());
    }
}
