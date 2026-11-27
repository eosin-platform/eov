//! The language of eov's interface: which one, and applying it.
//!
//! The languages, catalogs and the `tr!` macro are the plugin API's
//! ([`eov_plugin_api::i18n`]), shared with the plugins. The language is
//! chosen in this order:
//!
//! 1. `--language`, for this run only (it is never saved);
//! 2. the language chosen in the settings (`language` in `config.toml`);
//! 3. the system's preferred languages, the first one eov has;
//! 4. English.
//!
//! The window's own texts (`@tr` in `app/ui`) are compiled in by Slint
//! (`app/build.rs`) and switched with [`slint::select_bundled_translation`];
//! the texts made in Rust come from tables `build.rs` makes from the same
//! catalogs. Both change at once when the user picks another language, as
//! do plugins (their catalogs and [`HostEvent::LanguageChanged`]).
//!
//! [`HostEvent::LanguageChanged`]: eov_plugin_api::protocol::HostEvent

use crate::{AppInfo, AppWindow, LanguageEntry, Locale};
use eov_plugin_api::i18n::{self, Catalog, Language, tr};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::cell::Cell;
use std::rc::Rc;

/// The translations of the texts made in Rust for one language.
pub(crate) struct RustCatalog {
    pub(crate) tag: &'static str,
    /// The catalog's `Plural-Forms` header.
    pub(crate) plural_forms: &'static str,
    /// `(context, English text, forms)`.
    pub(crate) messages: &'static [(Option<&'static str>, &'static str, &'static [&'static str])],
}

// BUNDLED_LANGUAGES and RUST_CATALOGS, made by build.rs.
include!(concat!(env!("OUT_DIR"), "/catalogs.rs"));

/// Where the language of this run came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Origin {
    /// `--language`.
    CommandLine,
    /// `language` in config.toml.
    Settings,
    /// The system's preferred languages.
    System,
    /// None of the above gave a language eov has: English.
    Fallback,
}

/// The language of this run and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Choice {
    pub(crate) language: &'static Language,
    pub(crate) origin: Origin,
}

/// Chooses the language: `--language` (`command_line`), then the settings,
/// then the system's languages (most preferred first), then English. An
/// unknown `--language` is passed over; its value is returned so that it
/// can be reported once the language is set.
pub(crate) fn resolve(
    command_line: Option<&str>,
    settings: Option<&'static Language>,
    system: &[String],
) -> (Choice, Option<String>) {
    let mut unknown = None;
    if let Some(value) = command_line {
        match from_command_line(value) {
            Some(language) => {
                return (
                    Choice {
                        language,
                        origin: Origin::CommandLine,
                    },
                    None,
                );
            }
            None => unknown = Some(value.to_string()),
        }
    }
    let choice = match settings {
        Some(language) => Choice {
            language,
            origin: Origin::Settings,
        },
        None => match i18n::negotiate_first(system) {
            Some(language) => Choice {
                language,
                origin: Origin::System,
            },
            None => Choice {
                language: i18n::source(),
                origin: Origin::Fallback,
            },
        },
    };
    (choice, unknown)
}

/// The language `--language` names: a tag (`pt-BR`, also `pt_BR` or
/// `PT-br`), or a locale name for one (`zh_TW` gives Traditional Chinese).
pub(crate) fn from_command_line(value: &str) -> Option<&'static Language> {
    i18n::by_tag(value).or_else(|| i18n::negotiate(value))
}

/// The message for an unknown `--language`, in the language eov then uses.
pub(crate) fn unknown_language_message(value: &str) -> String {
    let tags: Vec<&str> = i18n::LANGUAGES.iter().map(|l| l.tag).collect();
    tr!(
        "Unknown language '{value}' given with --language: it is not one of eov's languages ({languages}). The interface is in {current}.",
        value = i18n::isolate(value),
        languages = i18n::isolate(tags.join(", ")),
        current = language_name(i18n::current()),
    )
}

/// The system's preferred languages, most preferred first (BCP 47 or
/// POSIX names): the user's language list on macOS and Windows; on other
/// systems `LANGUAGE`, `LC_ALL`, `LC_MESSAGES` and `LANG`, as gettext reads
/// them (a `C` or `POSIX` locale means English, whatever `LANGUAGE` says).
pub(crate) fn system_locales() -> Vec<String> {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let effective = ["LC_ALL", "LC_MESSAGES", "LANG"]
            .into_iter()
            .filter_map(|name| std::env::var(name).ok())
            .find(|value| !value.is_empty());
        if let Some(locale) = effective {
            let name = locale.split(['.', '@']).next().unwrap_or_default();
            if name == "C" || name == "POSIX" {
                return Vec::new();
            }
        }
    }
    sys_locale::get_locales().collect()
}

/// The language the system gives (for "Automatic").
pub(crate) fn system_language() -> &'static Language {
    i18n::negotiate_first(system_locales()).unwrap_or(i18n::source())
}

/// The catalog of the texts made in Rust for `language`.
fn rust_catalog(language: &'static Language) -> Option<Catalog> {
    if language.is_source() {
        return None;
    }
    let compiled = RUST_CATALOGS.iter().find(|c| c.tag == language.tag)?;
    let messages = compiled
        .messages
        .iter()
        .map(|(context, msgid, forms)| (*context, *msgid, *forms));
    match Catalog::from_messages(language, compiled.plural_forms, messages) {
        Ok(catalog) => Some(catalog),
        Err(error) => {
            tracing::warn!("The {} translation cannot be used: {error}", language.tag);
            None
        }
    }
}

/// The translation of a text made in Rust into `language` (whatever the
/// current language is), if eov's catalog has one.
pub(crate) fn translation(
    language: &'static Language,
    context: Option<&str>,
    text: &str,
) -> Option<String> {
    rust_catalog(language)?
        .translate(context, text)
        .map(str::to_string)
}

/// Makes `language` the one of the texts made in Rust ([`tr!`]); the window
/// follows with [`apply_to_window`].
pub(crate) fn activate(language: &'static Language) {
    i18n::set_catalog(language, rust_catalog(language));
}

thread_local! {
    /// Whether `--language` chose this run's language (until the user
    /// chooses one in the window).
    static FROM_COMMAND_LINE: Cell<bool> = const { Cell::new(false) };
    /// The choice made in the Language dialog in this run (`Some(None)`:
    /// the system's), which the dialog shows checked even when it could
    /// not be saved.
    static CHOSEN: Cell<Option<Option<&'static Language>>> = const { Cell::new(None) };
}

/// Chooses and activates the language of this run (see the module
/// documentation). Returns the value of an unknown `--language`.
pub(crate) fn init(command_line: Option<&str>) -> Option<String> {
    let (choice, unknown) = resolve(
        command_line,
        crate::config::load_language(),
        &system_locales(),
    );
    activate(choice.language);
    FROM_COMMAND_LINE.with(|from| from.set(choice.origin == Origin::CommandLine));
    tracing::info!(
        "Interface language: {} ({:?})",
        choice.language.tag,
        choice.origin
    );
    unknown
}

/// The window in the current language: its texts (Slint's bundled
/// translations, which also set how Slint writes numbers), the fonts of
/// Chinese, Japanese and Korean text (see [`crate::fonts`]), the writing
/// direction and the Home screen's language link.
pub(crate) fn apply_to_window(ui: &AppWindow) {
    let language = i18n::current();
    // Before the texts change, so that they are laid out once, in the
    // language's font.
    crate::fonts::apply(language);
    // English is the texts as written: Slint selects it as "".
    let bundle = if language.is_source() {
        ""
    } else {
        language.tag
    };
    if let Err(error) = slint::select_bundled_translation(bundle) {
        tracing::warn!(
            "The window's texts have no {} translation: {error}",
            language.tag
        );
    }
    let locale = ui.global::<Locale>();
    locale.set_tag(language.tag.into());
    locale.set_rtl(language.is_rtl());
    locale.set_decimal_separator(language.decimal_separator.to_string().into());
    ui.global::<AppInfo>()
        .set_language_name(language.native_name.into());
}

/// A language's name in the language of the interface ("German" in
/// English, "Allemand" in French).
pub(crate) fn language_name(language: &Language) -> String {
    match language.tag {
        "en" => tr!("Language name" => "English"),
        "ar" => tr!("Language name" => "Arabic"),
        "cs" => tr!("Language name" => "Czech"),
        "da" => tr!("Language name" => "Danish"),
        "de" => tr!("Language name" => "German"),
        "el" => tr!("Language name" => "Greek"),
        "es" => tr!("Language name" => "Spanish"),
        "fa" => tr!("Language name" => "Persian"),
        "fi" => tr!("Language name" => "Finnish"),
        "fr" => tr!("Language name" => "French"),
        "he" => tr!("Language name" => "Hebrew"),
        "hi" => tr!("Language name" => "Hindi"),
        "hu" => tr!("Language name" => "Hungarian"),
        "id" => tr!("Language name" => "Indonesian"),
        "it" => tr!("Language name" => "Italian"),
        "ja" => tr!("Language name" => "Japanese"),
        "ko" => tr!("Language name" => "Korean"),
        "nb" => tr!("Language name" => "Norwegian Bokmål"),
        "nl" => tr!("Language name" => "Dutch"),
        "pl" => tr!("Language name" => "Polish"),
        "pt-BR" => tr!("Language name" => "Portuguese (Brazil)"),
        "ro" => tr!("Language name" => "Romanian"),
        "ru" => tr!("Language name" => "Russian"),
        "sl" => tr!("Language name" => "Slovenian"),
        "sv" => tr!("Language name" => "Swedish"),
        "th" => tr!("Language name" => "Thai"),
        "tr" => tr!("Language name" => "Turkish"),
        "uk" => tr!("Language name" => "Ukrainian"),
        "vi" => tr!("Language name" => "Vietnamese"),
        "zh-Hans" => tr!("Language name" => "Chinese (Simplified)"),
        "zh-Hant" => tr!("Language name" => "Chinese (Traditional)"),
        _ => language.english_name.to_string(),
    }
}

/// The rows of the Language dialog: "Automatic" (the system's language),
/// then every language by its English name's order, each in its own
/// writing with its name in the interface's language.
pub(crate) fn dialog_entries() -> Vec<LanguageEntry> {
    let saved = CHOSEN
        .with(Cell::get)
        .unwrap_or_else(crate::config::load_language);
    let current = i18n::current();
    let from_command_line = FROM_COMMAND_LINE.with(Cell::get);
    let system = system_language();
    let mut languages: Vec<&'static Language> = i18n::LANGUAGES.iter().collect();
    languages.sort_by_key(|language| language.english_name);
    let system_name = language_name(system);
    let automatic = LanguageEntry {
        tag: SharedString::new(),
        native_name: tr!("Automatic (System Default)").into(),
        // The system's language by its own name, and by its name in the
        // interface's language when that differs ("Deutsch — German").
        detail: if system_name == system.native_name {
            tr!("Currently {native}", native = system.native_name)
        } else {
            tr!(
                "Currently {native} — {name}",
                native = system.native_name,
                name = system_name
            )
        }
        .into(),
        selected: !from_command_line && saved.is_none(),
        rtl: false,
    };
    std::iter::once(automatic)
        .chain(languages.into_iter().map(|language| {
            let name = language_name(language);
            LanguageEntry {
                tag: language.tag.into(),
                native_name: language.native_name.into(),
                detail: if name == language.native_name {
                    SharedString::new()
                } else {
                    name.into()
                },
                selected: if from_command_line {
                    std::ptr::eq(language, current)
                } else {
                    saved.is_some_and(|saved| std::ptr::eq(saved, language))
                },
                rtl: language.is_rtl(),
            }
        }))
        .collect()
}

/// What the Language dialog says about `--language`.
fn override_note() -> String {
    if FROM_COMMAND_LINE.with(Cell::get) {
        tr!(
            "This run shows {language} because eov was started with --language. Choosing a language here applies it now and from the next start.",
            language = i18n::current().native_name
        )
    } else {
        String::new()
    }
}

fn refresh_dialog(ui: &AppWindow) {
    let entries = dialog_entries();
    let shown = ui.get_language_entries();
    match shown.as_any().downcast_ref::<VecModel<LanguageEntry>>() {
        // The rows the dialog shows (the list of languages never changes):
        // update them in place, so that the row with the keyboard focus
        // keeps it when a language is chosen.
        Some(rows) if rows.row_count() == entries.len() => {
            for (index, entry) in entries.into_iter().enumerate() {
                if rows.row_data(index).as_ref() != Some(&entry) {
                    rows.set_row_data(index, entry);
                }
            }
        }
        _ => ui.set_language_entries(ModelRc::from(Rc::new(VecModel::from(entries)))),
    }
    ui.set_language_override_note(override_note().into());
}

/// Opens the Language dialog.
pub(crate) fn open_dialog(ui: &AppWindow) {
    refresh_dialog(ui);
    ui.set_language_dialog_visible(true);
}

/// The choice made in the Language dialog: `tag` empty follows the system.
/// It is saved, and the interface and the plugins change at once.
pub(crate) fn choose(ui: &AppWindow, tag: &str) {
    let preference = if tag.is_empty() {
        None
    } else {
        match i18n::by_tag(tag) {
            Some(language) => Some(language),
            None => {
                tracing::warn!("The Language dialog offered an unknown language '{tag}'");
                return;
            }
        }
    };
    FROM_COMMAND_LINE.with(|from| from.set(false));
    CHOSEN.with(|chosen| chosen.set(Some(preference)));
    // The language first, so that a problem saving it is told in the
    // language just chosen.
    switch_to(ui, preference.unwrap_or_else(system_language));
    if let Err(error) = crate::config::save_language(preference) {
        tracing::warn!("Cannot save the language: {error:#}");
        ui.set_status_text(
            tr!(
                "The language could not be saved for the next start: {}",
                format!("{error:#}")
            )
            .into(),
        );
    }
    refresh_dialog(ui);
}

/// Shows the interface in `language` from now on: the window, the texts
/// made in Rust that it shows, and the plugins.
pub(crate) fn switch_to(ui: &AppWindow, language: &'static Language) {
    if std::ptr::eq(language, i18n::current()) {
        return;
    }
    tracing::info!("Interface language changed to {}", language.tag);
    activate(language);
    apply_to_window(ui);
    // The status bar's last message is in the previous language: the bar
    // starts again from what the window says at startup (the same text in
    // the window's catalog). Messages written from here on stay, such as
    // choose()'s when the language cannot be saved.
    ui.set_status_text(
        // Translators: the status bar's text when there is nothing to report.
        tr!("AppWindow" => "Ready").into(),
    );
    crate::host_ui::language_changed(ui, language);
}

/// Connects the Language dialog and the Home screen's language link.
pub(crate) fn setup_callbacks(ui: &AppWindow) {
    let weak = ui.as_weak();
    ui.on_language_dialog_requested(move || {
        if let Some(ui) = weak.upgrade() {
            open_dialog(&ui);
        }
    });
    let weak = ui.as_weak();
    ui.global::<AppInfo>().on_choose_language(move || {
        if let Some(ui) = weak.upgrade() {
            open_dialog(&ui);
        }
    });
    let weak = ui.as_weak();
    ui.on_language_chosen(move |tag| {
        if let Some(ui) = weak.upgrade() {
            choose(&ui, &tag);
        }
    });
    let weak = ui.as_weak();
    ui.on_language_dialog_closed(move || {
        if let Some(ui) = weak.upgrade() {
            ui.set_language_dialog_visible(false);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(choice: (Choice, Option<String>)) -> (&'static str, Origin, Option<String>) {
        (choice.0.language.tag, choice.0.origin, choice.1)
    }

    #[test]
    fn the_command_line_comes_first_then_the_settings_then_the_system() {
        let german = i18n::by_tag("de");
        let system = vec!["fr-FR".to_string(), "en-US".to_string()];
        assert_eq!(
            tag(resolve(Some("ja"), german, &system)),
            ("ja", Origin::CommandLine, None)
        );
        assert_eq!(
            tag(resolve(None, german, &system)),
            ("de", Origin::Settings, None)
        );
        assert_eq!(
            tag(resolve(None, None, &system)),
            ("fr", Origin::System, None)
        );
        assert_eq!(
            tag(resolve(
                None,
                None,
                &["pt-PT".to_string(), "sw".to_string()]
            )),
            ("en", Origin::Fallback, None)
        );
        assert_eq!(
            tag(resolve(None, None, &[])),
            ("en", Origin::Fallback, None)
        );
    }

    #[test]
    fn an_unknown_command_line_language_is_passed_over_and_reported() {
        let german = i18n::by_tag("de");
        assert_eq!(
            tag(resolve(Some("klingon"), german, &[])),
            ("de", Origin::Settings, Some("klingon".to_string()))
        );
        assert_eq!(
            tag(resolve(Some("pt-PT"), None, &["it_IT.UTF-8".to_string()])),
            ("it", Origin::System, Some("pt-PT".to_string()))
        );
    }

    #[test]
    fn the_command_line_takes_tags_and_locale_names() {
        for (value, expected) in [
            ("ja", "ja"),
            ("pt-BR", "pt-BR"),
            ("pt_br", "pt-BR"),
            ("zh-Hant", "zh-Hant"),
            ("zh_TW", "zh-Hant"),
            ("zh-CN", "zh-Hans"),
            ("en", "en"),
            ("EN", "en"),
        ] {
            assert_eq!(
                from_command_line(value).map(|l| l.tag),
                Some(expected),
                "{value}"
            );
        }
        assert!(from_command_line("pt-PT").is_none());
        assert!(from_command_line("").is_none());
    }

    #[test]
    fn every_language_is_compiled_in_and_bundled_in_slint_order() {
        let expected: Vec<&str> = i18n::translations().iter().map(|l| l.tag).collect();
        assert_eq!(BUNDLED_LANGUAGES, expected.as_slice());
        let compiled: Vec<&str> = RUST_CATALOGS.iter().map(|c| c.tag).collect();
        assert_eq!(compiled, expected);
        for language in i18n::translations() {
            let catalog =
                rust_catalog(language).unwrap_or_else(|| panic!("{} has no catalog", language.tag));
            assert!(!catalog.is_empty(), "{} translates nothing", language.tag);
        }
    }

    /// tools/i18n/languages.json is the plugin API's language table for the
    /// translation tools. `EOV_UPDATE_LANGUAGES=1 cargo test -p eov
    /// languages_json` writes it again.
    #[test]
    fn languages_json_is_current() {
        let entries: Vec<serde_json::Value> = i18n::LANGUAGES
            .iter()
            .map(|language| {
                serde_json::json!({
                    "tag": language.tag,
                    "native_name": language.native_name,
                    "english_name": language.english_name,
                    "rtl": language.is_rtl(),
                    "plural_forms": language.plural_forms,
                    "decimal_separator": language.decimal_separator.to_string(),
                })
            })
            .collect();
        let text = serde_json::to_string_pretty(&entries).unwrap() + "\n";
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/i18n/languages.json");
        if std::env::var_os("EOV_UPDATE_LANGUAGES").is_some() {
            std::fs::write(&path, &text).unwrap();
        }
        let current = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        assert_eq!(
            current, text,
            "tools/i18n/languages.json is out of date: run EOV_UPDATE_LANGUAGES=1 cargo test -p eov languages_json"
        );
    }

    /// eov's catalogs and those of the in-tree plugins (submodules that are
    /// not checked out are skipped) are complete, merged with their
    /// templates, and show nothing wrongly (see
    /// `eov_plugin_api::i18n::check_catalog`).
    #[test]
    fn every_catalog_is_complete_and_valid() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let components = [
            ("app/translations", "eov"),
            ("plugin_api/example-plugin/translations", "example_plugin"),
            ("plugins/annotations/translations", "annotations"),
            ("plugins/cellaris/translations", "cellaris"),
            ("plugins/eovae/translations", "eovae"),
            ("plugins/gamepad/translations", "gamepad"),
        ];
        let mut failures = Vec::new();
        for (dir, domain) in components {
            let dir = root.join(dir);
            let template_path = dir.join(format!("{domain}.pot"));
            let Ok(template) = std::fs::read_to_string(&template_path) else {
                assert_ne!(domain, "eov", "{} is missing", template_path.display());
                continue;
            };
            for language in i18n::translations() {
                let path = i18n::catalog_path(&dir, domain, language);
                let Ok(po) = std::fs::read_to_string(&path) else {
                    failures.push(format!("{}: missing", path.display()));
                    continue;
                };
                let report = i18n::check_catalog(language, &po, Some(&template));
                if !report.is_complete() {
                    failures.push(format!(
                        "{}: {} untranslated, {} fuzzy, {} missing ({:?}), problems: {:#?}",
                        path.display(),
                        report.untranslated,
                        report.fuzzy,
                        report.missing.len(),
                        report.missing.iter().take(3).collect::<Vec<_>>(),
                        report.problems
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Texts that the window and Rust code share (the same context and
    /// text) are translated in Rust too: the theme menu names the themes
    /// as the toolbar's tooltip does (`"Theme" => "Dark"`). Their entries
    /// keep the references of both (tools/i18n/i18n.py `extract`), so
    /// build.rs keeps them in the Rust tables.
    #[test]
    fn texts_shared_with_the_window_are_translated_in_rust_too() {
        use crate::ThemeKind;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("translations");
        // Tests run in English.
        let english: Vec<String> = ThemeKind::ALL.iter().map(|kind| kind.label()).collect();
        assert_eq!(english, ["Dark", "Light", "High Contrast"]);
        let mut failures = Vec::new();
        for language in i18n::translations() {
            let catalog = Catalog::load(&dir, "eov", language)
                .unwrap()
                .unwrap_or_else(|| panic!("{} has no catalog", language.tag));
            let rust = rust_catalog(language).unwrap();
            for text in &english {
                let wanted = catalog
                    .translate(Some("Theme"), text)
                    .unwrap_or_else(|| panic!("{}: {text:?} is not translated", language.tag));
                if rust.translate(Some("Theme"), text) != Some(wanted) {
                    failures.push(format!(
                        "{}: {text:?} is not in the Rust tables",
                        language.tag
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        // The menu itself, on this test's thread only (thread-local-language).
        let german = i18n::by_tag("de").unwrap();
        activate(german);
        let labels: Vec<String> = ThemeKind::ALL.iter().map(|kind| kind.label()).collect();
        activate(i18n::source());
        let catalog = Catalog::load(&dir, "eov", german).unwrap().unwrap();
        for (label, text) in labels.iter().zip(&english) {
            assert_eq!(Some(label.as_str()), catalog.translate(Some("Theme"), text));
        }
    }

    /// The macOS bundle declares every language, so that macOS shows its
    /// own panels in the user's language.
    #[test]
    fn the_macos_bundle_declares_every_language() {
        let plist = include_str!("../../assets/macos/Info.plist");
        let declared: Vec<&str> = plist
            .split("<key>CFBundleLocalizations</key>")
            .nth(1)
            .and_then(|rest| rest.split("</array>").next())
            .expect("CFBundleLocalizations")
            .split("<string>")
            .skip(1)
            .filter_map(|item| item.split("</string>").next())
            .collect();
        let tags: Vec<&str> = i18n::LANGUAGES.iter().map(|l| l.tag).collect();
        assert_eq!(declared, tags);
    }

    #[test]
    fn every_language_has_a_name_in_the_interface() {
        for language in i18n::LANGUAGES {
            assert_eq!(language_name(language), language.english_name);
        }
    }
}

/// The language in the window: every language can be chosen, the window's
/// texts, the texts made in Rust and plugin surfaces follow, and right-to-left
/// languages are flagged. These tests switch languages on the Slint test
/// thread only (the plugin API's `thread-local-language` keeps other tests'
/// texts English) and switch back to English when they end.
#[cfg(test)]
mod window_tests {
    use super::*;
    use i_slint_backend_testing::{AccessibleRole, ElementHandle};
    use slint::Model;
    use std::path::{Path, PathBuf};

    fn translations_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("translations")
    }

    /// eov's translation of a text of the window (`@tr` in component
    /// `context`), as its catalog has it.
    fn window_text(language: &'static Language, context: &str, text: &str) -> String {
        Catalog::load(&translations_dir(), "eov", language)
            .unwrap()
            .and_then(|catalog| catalog.translate(Some(context), text).map(str::to_string))
            .unwrap_or_else(|| text.to_string())
    }

    /// Back to English when a test ends, also when it fails.
    struct English<'a>(&'a AppWindow);

    impl Drop for English<'_> {
        fn drop(&mut self) {
            switch_to(self.0, i18n::source());
            CHOSEN.with(|chosen| chosen.set(None));
            self.0.set_language_dialog_visible(false);
        }
    }

    fn window() -> AppWindow {
        let ui = AppWindow::new().unwrap();
        ui.window().set_size(slint::PhysicalSize::new(1400, 900));
        ui.show().unwrap();
        ui
    }

    fn labelled(ui: &AppWindow, role: AccessibleRole, label: &str) -> bool {
        ElementHandle::find_by_accessible_label(ui, label)
            .any(|element| element.accessible_role() == Some(role))
    }

    #[test]
    fn every_language_can_be_chosen_and_the_window_follows() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            apply_to_window(&ui);
            for language in i18n::LANGUAGES {
                switch_to(&ui, language);
                let locale = ui.global::<Locale>();
                assert_eq!(locale.get_tag(), language.tag);
                assert_eq!(locale.get_rtl(), language.is_rtl(), "{}", language.tag);
                assert_eq!(
                    locale.get_decimal_separator(),
                    language.decimal_separator.to_string()
                );
                assert_eq!(
                    ui.global::<AppInfo>().get_language_name(),
                    language.native_name
                );
                // The toolbar's globe: "Language" in the language, and in
                // English too for those who cannot read it.
                let word = window_text(language, "WindowToolbar", "Language");
                let label = if language.is_source() {
                    word.clone()
                } else {
                    format!("{word} (Language)")
                };
                assert!(
                    labelled(&ui, AccessibleRole::Button, &label),
                    "{}: no button labelled {label:?}",
                    language.tag
                );
                if !language.is_source() {
                    assert_ne!(word, "Language", "{} translates nothing", language.tag);
                }
                // Texts made in Rust follow too.
                assert_eq!(i18n::current().tag, language.tag);
            }
        });
    }

    #[test]
    fn the_language_dialog_offers_every_language_and_applies_a_choice() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            open_dialog(&ui);
            assert!(ui.get_language_dialog_visible());
            let entries: Vec<LanguageEntry> = ui.get_language_entries().iter().collect();
            assert_eq!(entries.len(), 1 + i18n::LANGUAGES.len());
            assert!(entries[0].tag.is_empty(), "Automatic comes first");
            for language in i18n::LANGUAGES {
                let entry = entries
                    .iter()
                    .find(|entry| entry.tag == language.tag)
                    .unwrap_or_else(|| panic!("{} is not offered", language.tag));
                assert_eq!(entry.native_name, language.native_name);
                assert_eq!(entry.rtl, language.is_rtl());
            }

            choose(&ui, "de");
            let german = i18n::by_tag("de").unwrap();
            assert_eq!(i18n::current(), german);
            assert_eq!(crate::config::load_language(), Some(german));
            let entries: Vec<LanguageEntry> = ui.get_language_entries().iter().collect();
            let selected: Vec<&str> = entries
                .iter()
                .filter(|entry| entry.selected)
                .map(|entry| entry.tag.as_str())
                .collect();
            assert_eq!(selected, ["de"]);
            // The rows made in Rust are made again in German.
            let catalog = rust_catalog(german).unwrap();
            let automatic = catalog
                .translate(None, "Automatic (System Default)")
                .expect("translated");
            assert_eq!(entries[0].native_name, automatic);
            // German is listed as "Deutsch", with no name repeated.
            let deutsch = entries.iter().find(|e| e.tag == "de").unwrap();
            assert_eq!(deutsch.native_name, "Deutsch");
            assert!(deutsch.detail.is_empty());

            // Automatic forgets the choice.
            choose(&ui, "");
            assert_eq!(crate::config::load_language(), None);
            assert_eq!(i18n::current(), system_language());
        });
    }

    /// The status bar's last message was written in the previous language:
    /// after a change the bar says "Ready" in the new one, as at startup.
    #[test]
    fn a_language_change_starts_the_status_bar_again() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            ui.set_status_text("Opened C3L-00004-21.svs (3 levels)".into());
            let german = i18n::by_tag("de").unwrap();
            switch_to(&ui, german);
            // In German, as eov's tables have it (English until the entry
            // the window shares with Rust code is in them).
            let ready = translation(german, Some("AppWindow"), "Ready")
                .unwrap_or_else(|| "Ready".to_string());
            assert_eq!(ui.get_status_text(), ready);
            // Showing the language already shown changes nothing.
            ui.set_status_text("Gespeichert".into());
            switch_to(&ui, german);
            assert_eq!(ui.get_status_text(), "Gespeichert");
        });
    }

    #[test]
    fn rust_messages_are_translated_with_their_values() {
        // On this test's thread only (thread-local-language).
        let german = i18n::by_tag("de").unwrap();
        let catalog = rust_catalog(german).unwrap();
        let template = catalog
            .translate(
                None,
                "Unknown language '{value}' given with --language: it is not one of eov's languages ({languages}). The interface is in {current}.",
            )
            .expect("translated")
            .to_string();
        activate(german);
        let message = unknown_language_message("xx");
        activate(i18n::source());
        assert!(message.contains("'xx'") || message.contains("„xx“") || message.contains("xx"));
        assert!(message.contains("Deutsch"), "{message}");
        assert_ne!(message, template, "the values are filled in");
        assert!(!message.contains("{value}") && !message.contains("{current}"));
        // English again.
        assert!(unknown_language_message("xx").starts_with("Unknown language"));
    }

    /// A plugin surface compiled with the plugin's catalogs follows the
    /// window's language, also for languages the plugin lacks (English).
    #[test]
    fn plugin_surfaces_follow_the_language_of_the_window() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            let plugin = tempfile::tempdir().unwrap();
            std::fs::write(
                plugin.path().join("plugin.toml"),
                "id = \"demo\"\nname = \"Demo\"\nversion = \"1.0.0\"\n[i18n]\n",
            )
            .unwrap();
            let german = i18n::by_tag("de").unwrap();
            let japanese = i18n::by_tag("ja").unwrap();
            for (language, text) in [(german, "Hallo"), (japanese, "こんにちは")] {
                let folder = plugin
                    .path()
                    .join("translations")
                    .join(language.tag)
                    .join("LC_MESSAGES");
                std::fs::create_dir_all(&folder).unwrap();
                std::fs::write(
                    folder.join("demo.po"),
                    i18n::empty_catalog(language, "demo")
                        + &format!("\nmsgctxt \"Panel\"\nmsgid \"Hello\"\nmsgstr \"{text}\"\n"),
                )
                .unwrap();
            }
            let (bundle, domain) =
                crate::host_ui::plugin_i18n::surface_bundle(plugin.path()).expect("a bundle");
            let mut compiler = slint_interpreter::Compiler::default();
            compiler.set_translation_domain(domain);
            compiler.set_bundled_translations_path(bundle);
            let result = spin_on(compiler.build_from_source(
                "export component Panel inherits Rectangle { out property <string> greeting: @tr(\"Hello\"); }"
                    .into(),
                plugin.path().join("panel.slint"),
            ));
            let panel = result
                .component("Panel")
                .expect("compiles")
                .create()
                .unwrap();
            let greeting = || match panel.get_property("greeting").unwrap() {
                slint_interpreter::Value::String(text) => text.to_string(),
                other => panic!("{other:?}"),
            };
            assert_eq!(greeting(), "Hello");
            switch_to(&ui, german);
            assert_eq!(greeting(), "Hallo");
            switch_to(&ui, japanese);
            assert_eq!(greeting(), "こんにちは");
            // The plugin has no French: English, not another language's.
            switch_to(&ui, i18n::by_tag("fr").unwrap());
            assert_eq!(greeting(), "Hello");
            switch_to(&ui, i18n::by_tag("zh-Hant").unwrap());
            assert_eq!(greeting(), "Hello");
        });
    }

    /// Numbers typed into the window's fields are read whatever the
    /// language: `.` or `,`, the language's separator (Slint writes and reads
    /// numbers with it), and Arabic-Indic or Persian digits.
    #[test]
    fn typed_numbers_are_read_in_every_language() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            for tag in ["en", "de", "fa", "ar", "ja"] {
                switch_to(&ui, i18n::by_tag(tag).unwrap());
                let locale = ui.global::<Locale>();
                for (text, value) in [
                    ("1.5", 1.5),
                    ("1,5", 1.5),
                    ("۱٫۵", 1.5),
                    ("٢٫٥", 2.5),
                    ("256", 256.0),
                    ("\u{2212}3", -3.0),
                ] {
                    let read = locale.invoke_number(text.into());
                    assert!(
                        (read - value).abs() < 1e-5,
                        "{tag}: {text:?} read as {read}"
                    );
                    assert!(locale.invoke_is_number(text.into()), "{tag}: {text:?}");
                }
                assert!(!locale.invoke_is_number("abc".into()), "{tag}");
            }
        });
    }

    /// Fields that take a whole number (pixels, threads, colour values)
    /// read one written with a thousands separator as that number in every
    /// language, `,` and `.` alike (three digits after the separator are no
    /// fraction anyone means there), and round other decimals.
    #[test]
    fn whole_numbers_are_read_in_every_language() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = window();
            let _english = English(&ui);
            for language in i18n::LANGUAGES {
                switch_to(&ui, language);
                let tag = language.tag;
                let locale = ui.global::<Locale>();
                for (text, value) in [
                    ("1,024", 1024),
                    ("1.024", 1024),
                    ("1 024", 1024),
                    ("1\u{A0}024", 1024),
                    ("1\u{202F}024", 1024),
                    ("١٬٠٢٤", 1024),
                    ("۱٬۰۲۴", 1024),
                    ("65,536", 65536),
                    ("512", 512),
                    ("۵۱۲", 512),
                    // Decimals, rounded.
                    ("128.6", 129),
                    ("128,6", 129),
                    ("١٢٨٫٦", 129),
                    ("2,5", 3),
                    ("1,0245", 1),
                    // The Arabic decimal sign never separates thousands,
                    // and a first digit 0 makes a decimal.
                    ("1٫024", 1),
                    ("0,250", 0),
                ] {
                    assert!(
                        locale.invoke_is_whole_number(text.into()),
                        "{tag}: {text:?}"
                    );
                    assert_eq!(
                        locale.invoke_whole_number(text.into()),
                        value,
                        "{tag}: {text:?}"
                    );
                }
                for text in ["", "x", "inf", "1 024 x"] {
                    assert!(
                        !locale.invoke_is_whole_number(text.into()),
                        "{tag}: {text:?}"
                    );
                }
            }
        });
    }

    fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
        use std::task::{Context, Poll, Waker};
        let mut cx = Context::from_waker(Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }
}
