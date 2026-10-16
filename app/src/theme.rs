//! The window's colour theme: dark (the default), light or high contrast
//! (`theme` in config.toml, or the theme button of the toolbar). The colours
//! themselves are the tokens of `app/ui/theme.slint`.

use crate::{AppWindow, ContextMenuItem, Theme, ThemeKind};
use slint::ComponentHandle;

impl ThemeKind {
    pub(crate) const ALL: [ThemeKind; 3] =
        [ThemeKind::Dark, ThemeKind::Light, ThemeKind::HighContrast];

    /// The name in config.toml.
    pub(crate) fn id(self) -> &'static str {
        match self {
            ThemeKind::Dark => "dark",
            ThemeKind::Light => "light",
            ThemeKind::HighContrast => "high-contrast",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            ThemeKind::Dark => "Dark",
            ThemeKind::Light => "Light",
            ThemeKind::HighContrast => "High Contrast",
        }
    }

    /// Reads a name as written in config.toml (case and `_` for `-` do not
    /// matter).
    pub(crate) fn from_id(id: &str) -> Option<Self> {
        match id.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "dark" => Some(ThemeKind::Dark),
            "light" => Some(ThemeKind::Light),
            "high-contrast" => Some(ThemeKind::HighContrast),
            _ => None,
        }
    }
}

pub(crate) fn current(ui: &AppWindow) -> ThemeKind {
    ui.global::<Theme>().get_kind()
}

/// The theme in config.toml (the dark one when there is none).
pub(crate) fn apply_saved(ui: &AppWindow) {
    if let Some(kind) = crate::config::load_theme() {
        ui.global::<Theme>().set_kind(kind);
    }
}

/// Switches to `kind` and remembers it.
pub(crate) fn choose(ui: &AppWindow, kind: ThemeKind) {
    ui.global::<Theme>().set_kind(kind);
    if let Err(error) = crate::config::save_theme(kind) {
        tracing::warn!("Could not save the theme: {error}");
    }
}

/// The theme menu's items (`theme:<id>`), the current one checked.
pub(crate) fn menu_items(current: ThemeKind) -> Vec<ContextMenuItem> {
    ThemeKind::ALL
        .iter()
        .map(|kind| ContextMenuItem {
            id: format!("theme:{}", kind.id()).into(),
            label: kind.label().into(),
            icon: if *kind == current { "check" } else { "" }.into(),
            shortcut: "".into(),
            enabled: true,
            separator_after: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_read_as_written_in_the_file() {
        for kind in ThemeKind::ALL {
            assert_eq!(ThemeKind::from_id(kind.id()), Some(kind));
        }
        assert_eq!(
            ThemeKind::from_id(" High_Contrast "),
            Some(ThemeKind::HighContrast)
        );
        assert_eq!(ThemeKind::from_id("solarized"), None);
    }

    /// WCAG contrast of `text` over `background` (alpha composited over
    /// `base`).
    fn contrast(text: slint::Color, background: slint::Color, base: slint::Color) -> f64 {
        fn over(top: slint::Color, bottom: [f64; 3]) -> [f64; 3] {
            let a = f64::from(top.alpha()) / 255.0;
            let c = [top.red(), top.green(), top.blue()];
            [0, 1, 2].map(|i| f64::from(c[i]) * a + bottom[i] * (1.0 - a))
        }
        fn luminance(c: [f64; 3]) -> f64 {
            let [r, g, b] = c.map(|v| {
                let v = v / 255.0;
                if v <= 0.03928 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            });
            0.2126 * r + 0.7152 * g + 0.0722 * b
        }
        let base = [base.red(), base.green(), base.blue()].map(f64::from);
        let behind = over(background, base);
        let front = over(text, behind);
        let (a, b) = (luminance(front), luminance(behind));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    /// The text of the high-contrast theme has a contrast of at least 7:1
    /// against what it is drawn on, that of the light theme at least 4.5:1;
    /// a light slide is behind the panels that float over it.
    #[test]
    fn text_is_readable_in_the_light_and_high_contrast_themes() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = AppWindow::new().unwrap();
            let theme = ui.global::<Theme>();
            let slide = slint::Color::from_rgb_u8(0xf0, 0xe6, 0xee);
            for (kind, minimum) in [(ThemeKind::Light, 4.5), (ThemeKind::HighContrast, 7.0)] {
                theme.set_kind(kind);
                let surfaces = [
                    ("background", theme.get_background()),
                    ("surface", theme.get_surface()),
                    ("surface-hover", theme.get_surface_hover()),
                    ("toolbar", theme.get_toolbar_bg()),
                    ("inactive tab", theme.get_tab_inactive()),
                    ("menu", theme.get_context_menu_bg()),
                    ("field", theme.get_input_bg()),
                    ("hud", theme.get_hud_bg()),
                    ("hud panel", theme.get_hud_panel_bg()),
                    ("sunken", theme.get_surface_sunken()),
                ];
                let texts = [
                    ("primary", theme.get_text_primary()),
                    ("secondary", theme.get_text_secondary()),
                    ("label", theme.get_text_label()),
                    ("value", theme.get_text_value()),
                    ("strong", theme.get_text_strong()),
                ];
                let mut pairs: Vec<(String, slint::Color, slint::Color)> = Vec::new();
                for (surface, background) in surfaces {
                    for (name, text) in texts {
                        pairs.push((format!("{name} text on {surface}"), text, background));
                    }
                }
                pairs.extend([
                    (
                        "text on the accent".into(),
                        theme.get_text_on_accent(),
                        theme.get_accent(),
                    ),
                    (
                        "selected text".into(),
                        theme.get_text_selected(),
                        theme.get_selected_fill(),
                    ),
                    (
                        "menu item under the pointer".into(),
                        theme.get_text_primary(),
                        theme.get_context_menu_hover(),
                    ),
                    (
                        "tooltip".into(),
                        theme.get_tooltip_text(),
                        theme.get_tooltip_bg(),
                    ),
                    ("link".into(), theme.get_link(), theme.get_background()),
                    ("error".into(), theme.get_error(), theme.get_surface()),
                    ("warning".into(), theme.get_warning(), theme.get_surface()),
                    (
                        "success".into(),
                        theme.get_success(),
                        theme.get_toolbar_bg(),
                    ),
                    ("danger".into(), theme.get_danger(), theme.get_surface()),
                    (
                        "on danger".into(),
                        theme.get_text_on_danger(),
                        theme.get_danger_fill(),
                    ),
                    (
                        "badge".into(),
                        theme.get_slide_badge_text(),
                        theme.get_slide_badge_bg(),
                    ),
                ]);
                for (what, text, background) in pairs {
                    let ratio = contrast(text, background, slide);
                    assert!(ratio >= minimum, "{kind:?}: {what}: {ratio:.2}");
                }
            }
            theme.set_kind(ThemeKind::Dark);
        });
    }
}
