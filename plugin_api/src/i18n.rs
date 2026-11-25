//! Localisation of eov and its plugins.
//!
//! eov's interface is available in the [`LANGUAGES`] listed here. Every text
//! is written in English, the source language, and translated through
//! gettext catalogs (`.po` files) that sit beside the code that uses them:
//!
//! ```text
//! translations/
//! ├── <domain>.pot                      the template: every text, untranslated
//! ├── de/LC_MESSAGES/<domain>.po        German
//! ├── pt-BR/LC_MESSAGES/<domain>.po     Portuguese (Brazil)
//! └── zh-Hant/LC_MESSAGES/<domain>.po   Chinese (Traditional)
//! ```
//!
//! Folders are named by the language's BCP 47 tag ([`Language::tag`]); the
//! spellings other gettext tools use (`pt_BR`, `zh_TW`, `nb_NO`) are found
//! too ([`Language::catalog_folders`]). A text that a
//! catalog lacks, or that is marked fuzzy, is shown in English: a missing
//! translation never shows as an empty or raw text.
//!
//! # Texts in `.slint` files
//!
//! Mark them with Slint's `@tr("…")`. eov compiles a plugin's surfaces with
//! the plugin's catalogs (see the manifest's `[i18n]` table), so they follow
//! the language of the interface and change with it.
//!
//! # Texts made in Rust
//!
//! Use [`tr!`](crate::tr!) for every text a user reads: notifications,
//! status texts, menu items, tooltips, errors. It has the syntax of the
//! [`tr`](https://docs.rs/tr) crate's macro, so its `xtr` tool extracts the
//! texts, and the placeholders of `@tr` (`{}`, `{0}`, `{name}`, `{n}` for
//! the count of a plural text; `{{` and `}}` for braces):
//!
//! ```
//! use eov_plugin_api::i18n::tr;
//! let file = "case-12.svs";
//! let count = 3;
//! assert_eq!(tr!("Opened {}", file), "Opened case-12.svs");
//! let _ = tr!("Toolbar" => "Open"); // with a context
//! assert_eq!(tr!("One region" | "{n} regions" % count), "3 regions");
//! let _ = tr!("Saved {count} regions to {file}", count = count, file = file);
//! // Translators: a comment for translators, extracted with the text.
//! let _ = tr!("Measure");
//! ```
//!
//! eov loads a plugin's catalog for the language of the interface before
//! it calls [`Plugin::create`](crate::sdk::Plugin::create), and again before
//! it delivers [`HostEvent::LanguageChanged`](crate::protocol::HostEvent),
//! so `tr!` always answers in the current language. Texts a plugin handed
//! to eov earlier (toolbar tooltips, tool names) keep their language until
//! the plugin hands them over again: do that when the language changes.
//!
//! Do not translate what is read by programs or other people's tools: file
//! formats, JSON, CSV, log messages, identifiers, or data such as file
//! names and slide properties.
//!
//! # The same text for the user and for the log
//!
//! A problem is often both shown to the user and written to eov's log,
//! which stays English. Write it once, with `tr!`, and get it in both
//! languages from [`shown_and_english`]; [`in_english`] runs any code with
//! this thread's texts written in English:
//!
//! ```
//! use eov_plugin_api::i18n::{self, tr};
//! let name = "case-12.svs";
//! let (shown, english) = i18n::shown_and_english(|| tr!("{} could not be read", i18n::isolate(name)));
//! assert_eq!(english, "case-12.svs could not be read");
//! # assert_eq!(shown, english);
//! // host.info(english); host.notify(NotifyLevel::Warning, shown);
//! ```
//!
//! # Writing direction and numbers
//!
//! Arabic, Persian and Hebrew are written from right to left. Wrap values
//! that are inserted into a translated text, and that are written from left
//! to right whatever the language (file names, paths, numbers with units),
//! with [`isolate`], so the text around them keeps its order. Show
//! decimal numbers with [`format_decimal`] and read numbers that users type
//! with [`parse_decimal`]; both follow the language's decimal separator, as
//! Slint does in the interface.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Translates a text written in English: see the [module documentation](self).
pub use crate::tr;

/// The tag of the source language. Catalogs translate from it; it has none.
pub const SOURCE_TAG: &str = "en";

/// The writing direction of a language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    LeftToRight,
    RightToLeft,
}

/// A language of eov's interface: one of [`LANGUAGES`]. Minor releases may
/// add fields, so code outside this crate reads them and does not build or
/// destructure a `Language` in full.
#[derive(Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Language {
    /// The BCP 47 tag (`"de"`, `"pt-BR"`, `"zh-Hant"`): the name of its
    /// catalog folders, the value of `language` in `config.toml` and of
    /// `--language`, and what eov tells plugins.
    pub tag: &'static str,
    /// The language's name in itself (`"Deutsch"`), with the variant where
    /// that matters (`"Português (Brasil)"`).
    pub native_name: &'static str,
    /// The language's name in English (`"German"`).
    pub english_name: &'static str,
    pub direction: Direction,
    /// The `Plural-Forms` header of the language's catalogs (gettext
    /// syntax): how many forms a text with a number has, and which one a
    /// number takes.
    pub plural_forms: &'static str,
    /// The decimal separator, as the Unicode CLDR has it and as Slint
    /// writes and reads numbers in the interface.
    pub decimal_separator: char,
}

impl Language {
    pub fn is_rtl(&self) -> bool {
        self.direction == Direction::RightToLeft
    }

    pub fn is_source(&self) -> bool {
        self.tag == SOURCE_TAG
    }

    /// The number of plural forms (`nplurals` in [`Self::plural_forms`]).
    pub fn plural_count(&self) -> usize {
        self.plural_forms
            .split(';')
            .find_map(|part| part.trim().strip_prefix("nplurals="))
            .and_then(|count| count.trim().parse().ok())
            .unwrap_or(2)
    }

    /// The tag in POSIX spelling (`pt_BR`), as some tools name folders.
    pub fn posix_tag(&self) -> String {
        self.tag.replace('-', "_")
    }

    /// The folder names gettext tools give this language's catalogs, in
    /// the order eov looks for them: the tag (`zh-Hant`), its POSIX
    /// spelling (`zh_Hant`), and the locale names msginit, Poedit and
    /// Weblate use for it (`zh_TW`, `nb_NO`).
    pub fn catalog_folders(&self) -> Vec<String> {
        let mut folders = vec![self.tag.to_string()];
        let posix = self.posix_tag();
        if posix != self.tag {
            folders.push(posix);
        }
        let conventional: &[&str] = match self.tag {
            "zh-Hans" => &["zh_CN"],
            "zh-Hant" => &["zh_TW"],
            "nb" => &["nb_NO", "no"],
            _ => &[],
        };
        folders.extend(conventional.iter().map(|folder| folder.to_string()));
        folders
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag)
    }
}

const ONE_FORM: &str = "nplurals=1; plural=0;";
const TWO_FORMS: &str = "nplurals=2; plural=(n != 1);";
const ONE_FOR_ZERO_AND_ONE: &str = "nplurals=2; plural=(n > 1);";
const EAST_SLAVIC: &str = "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);";

const fn language(
    tag: &'static str,
    native_name: &'static str,
    english_name: &'static str,
    plural_forms: &'static str,
    decimal_separator: char,
) -> Language {
    Language {
        tag,
        native_name,
        english_name,
        direction: Direction::LeftToRight,
        plural_forms,
        decimal_separator,
    }
}

const fn rtl(language: Language) -> Language {
    Language {
        direction: Direction::RightToLeft,
        ..language
    }
}

/// Every language of the interface: English (the source) first, then the
/// translations by tag. Plural forms follow the Unicode CLDR as gettext
/// writes them; decimal separators are the CLDR's.
///
/// Minor releases add languages: treat this as a list of any length (it is
/// a slice, so its type stays the same when one is added), and expect tags
/// it does not have yet from newer eov versions ([`by_tag`] then answers
/// `None`).
pub static LANGUAGES: &[Language] = &[
    language("en", "English", "English", TWO_FORMS, '.'),
    rtl(language(
        "ar",
        "العربية",
        "Arabic",
        "nplurals=6; plural=(n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : n%100>=3 && n%100<=10 ? 3 : n%100>=11 ? 4 : 5);",
        '.',
    )),
    language(
        "cs",
        "Čeština",
        "Czech",
        "nplurals=3; plural=(n==1 ? 0 : n>=2 && n<=4 ? 1 : 2);",
        ',',
    ),
    language("da", "Dansk", "Danish", TWO_FORMS, ','),
    language("de", "Deutsch", "German", TWO_FORMS, ','),
    language("el", "Ελληνικά", "Greek", TWO_FORMS, ','),
    language("es", "Español", "Spanish", TWO_FORMS, ','),
    rtl(language(
        "fa",
        "فارسی",
        "Persian",
        ONE_FOR_ZERO_AND_ONE,
        '\u{066B}',
    )),
    language("fi", "Suomi", "Finnish", TWO_FORMS, ','),
    language("fr", "Français", "French", ONE_FOR_ZERO_AND_ONE, ','),
    rtl(language("he", "עברית", "Hebrew", TWO_FORMS, '.')),
    language("hi", "हिन्दी", "Hindi", ONE_FOR_ZERO_AND_ONE, '.'),
    language("hu", "Magyar", "Hungarian", TWO_FORMS, ','),
    language("id", "Bahasa Indonesia", "Indonesian", ONE_FORM, ','),
    language("it", "Italiano", "Italian", TWO_FORMS, ','),
    language("ja", "日本語", "Japanese", ONE_FORM, '.'),
    language("ko", "한국어", "Korean", ONE_FORM, '.'),
    language("nb", "Norsk bokmål", "Norwegian Bokmål", TWO_FORMS, ','),
    language("nl", "Nederlands", "Dutch", TWO_FORMS, ','),
    language(
        "pl",
        "Polski",
        "Polish",
        "nplurals=3; plural=(n==1 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);",
        ',',
    ),
    language(
        "pt-BR",
        "Português (Brasil)",
        "Portuguese (Brazil)",
        ONE_FOR_ZERO_AND_ONE,
        ',',
    ),
    language(
        "ro",
        "Română",
        "Romanian",
        "nplurals=3; plural=(n==1 ? 0 : (n==0 || (n%100 > 0 && n%100 < 20)) ? 1 : 2);",
        ',',
    ),
    language("ru", "Русский", "Russian", EAST_SLAVIC, ','),
    language(
        "sl",
        "Slovenščina",
        "Slovenian",
        "nplurals=4; plural=(n%100==1 ? 0 : n%100==2 ? 1 : n%100==3 || n%100==4 ? 2 : 3);",
        ',',
    ),
    language("sv", "Svenska", "Swedish", TWO_FORMS, ','),
    language("th", "ไทย", "Thai", ONE_FORM, '.'),
    language("tr", "Türkçe", "Turkish", TWO_FORMS, ','),
    language("uk", "Українська", "Ukrainian", EAST_SLAVIC, ','),
    language("vi", "Tiếng Việt", "Vietnamese", ONE_FORM, ','),
    language("zh-Hans", "简体中文", "Chinese (Simplified)", ONE_FORM, '.'),
    language(
        "zh-Hant",
        "繁體中文",
        "Chinese (Traditional)",
        ONE_FORM,
        '.',
    ),
];

/// English, the source language.
pub fn source() -> &'static Language {
    &LANGUAGES[0]
}

/// The translations: every language but English.
pub fn translations() -> &'static [Language] {
    &LANGUAGES[1..]
}

/// The language with exactly this tag (letter case and `_` for `-` do not
/// matter: `pt_br` is `pt-BR`). Use [`negotiate`] for locale names such as
/// `pt_BR.UTF-8` or `zh-TW`.
pub fn by_tag(tag: &str) -> Option<&'static Language> {
    let tag = tag.trim().replace('_', "-");
    LANGUAGES
        .iter()
        .find(|language| language.tag.eq_ignore_ascii_case(&tag))
}

/// The parts of a locale name that choose a language.
#[derive(Debug, Default, PartialEq, Eq)]
struct LocaleParts {
    language: String,
    script: Option<String>,
    region: Option<String>,
}

/// Splits a BCP 47 tag (`zh-Hant-TW`) or a POSIX locale name
/// (`pt_BR.UTF-8@euro`) into language, script and region. `C` and `POSIX`
/// (no preference) give `None`.
fn locale_parts(locale: &str) -> Option<LocaleParts> {
    // POSIX: language[_territory][.codeset][@modifier].
    let locale = locale.trim();
    let locale = locale.split(['.', '@']).next().unwrap_or_default();
    let mut subtags = locale.split(['-', '_']).filter(|part| !part.is_empty());
    let language = subtags.next()?.to_ascii_lowercase();
    if !(2..=3).contains(&language.len()) || !language.chars().all(|c| c.is_ascii_alphabetic()) {
        // `C`, `POSIX` or not a language at all.
        return None;
    }
    let mut parts = LocaleParts {
        language,
        ..Default::default()
    };
    for subtag in subtags {
        let is_letters = subtag.chars().all(|c| c.is_ascii_alphabetic());
        if subtag.len() == 4 && is_letters && parts.script.is_none() && parts.region.is_none() {
            parts.script = Some(subtag.to_ascii_lowercase());
        } else if parts.region.is_none()
            && ((subtag.len() == 2 && is_letters)
                || (subtag.len() == 3 && subtag.chars().all(|c| c.is_ascii_digit())))
        {
            parts.region = Some(subtag.to_ascii_uppercase());
        } else if subtag.len() == 1 {
            // An extension (`-u-nu-latn`) or private use (`-x-…`): the rest
            // does not choose the language.
            break;
        }
    }
    Some(parts)
}

/// The language of the interface for a locale name of the system or the
/// user: a BCP 47 tag (`pt-BR`, `zh-Hant-TW`, `es-419`) or a POSIX locale
/// (`de_AT.UTF-8`). `None` when none of the [`LANGUAGES`] fits, and for `C`
/// and `POSIX`.
///
/// Regional variants of a language take its translation (`de-CH` →
/// German, `es-MX` → Spanish, `fr-CA` → French), except where a region
/// writes differently from the translation:
///
/// - Chinese: Traditional characters for `Hant`, Taiwan, Hong Kong and
///   Macao; Simplified characters for `Hans`, mainland China, Singapore,
///   Malaysia and `zh` alone. A script, when given, decides.
/// - Portuguese: the translation is Brazilian, so it is chosen for `pt-BR`
///   and for `pt` alone (which the CLDR takes as Brazil), not for Portugal
///   or Portuguese-speaking Africa, whose spelling and vocabulary differ:
///   they get the next language the system prefers, or English. They can
///   still choose Português (Brasil) in the settings.
///
/// Old and alternative codes are understood: `iw` (Hebrew), `in`
/// (Indonesian), `no` and `nn` (Norwegian: Nynorsk readers get Bokmål,
/// the other written standard every Norwegian reads).
pub fn negotiate(locale: &str) -> Option<&'static Language> {
    let parts = locale_parts(locale)?;
    let region = parts.region.as_deref();
    let tag = match parts.language.as_str() {
        "zh" => match (parts.script.as_deref(), region) {
            (Some("hant"), _) => "zh-Hant",
            (Some("hans"), _) => "zh-Hans",
            (Some(_), _) => return None,
            (None, Some("TW" | "HK" | "MO")) => "zh-Hant",
            (None, _) => "zh-Hans",
        },
        "pt" => match region {
            None | Some("BR") => "pt-BR",
            Some(_) => return None,
        },
        "iw" => "he",
        "in" => "id",
        "no" | "nn" => "nb",
        // Serbian, Uzbek, … written in another script than the translation
        // (none here) would be refused by the script check below.
        other => other,
    };
    let language = by_tag(tag)?;
    // A script the translation is not written in (`ar-Latn`, `ru-Latn`):
    // a reader of that script may not read ours.
    if parts.language != "zh"
        && let Some(script) = parts.script.as_deref()
        && script != default_script(language)
    {
        return None;
    }
    Some(language)
}

/// The ISO 15924 script (lower case) a language's translation is written in.
fn default_script(language: &Language) -> &'static str {
    match language.tag {
        "ar" | "fa" => "arab",
        "he" => "hebr",
        "hi" => "deva",
        "el" => "grek",
        "ru" | "uk" => "cyrl",
        "ja" => "jpan",
        "ko" => "kore",
        "th" => "thai",
        "zh-Hans" => "hans",
        "zh-Hant" => "hant",
        _ => "latn",
    }
}

/// The first of `locales` (in order of preference) that [`negotiate`]
/// fits.
pub fn negotiate_first<I, S>(locales: I) -> Option<&'static Language>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    locales
        .into_iter()
        .find_map(|locale| negotiate(locale.as_ref()))
}

// ---------------------------------------------------------------------------
// The current language
// ---------------------------------------------------------------------------

/// The index into [`LANGUAGES`] of this library's current language. Each
/// plugin library has its own (statics are not shared between libraries);
/// eov and the SDK keep it up to date.
#[cfg(not(feature = "thread-local-language"))]
static CURRENT: AtomicUsize = AtomicUsize::new(0);

// With `thread-local-language` (hosts' and plugins' tests), each thread has
// its own language and catalog, so a test that switches languages does not
// change the texts that tests on other threads compare.
#[cfg(feature = "thread-local-language")]
thread_local! {
    static CURRENT: AtomicUsize = const { AtomicUsize::new(0) };
    static CATALOG: RwLock<Option<Catalog>> = const { RwLock::new(None) };
}

fn current_index() -> usize {
    #[cfg(not(feature = "thread-local-language"))]
    return CURRENT.load(Ordering::Relaxed);
    #[cfg(feature = "thread-local-language")]
    return CURRENT.with(|current| current.load(Ordering::Relaxed));
}

fn set_current_index(index: usize) {
    #[cfg(not(feature = "thread-local-language"))]
    CURRENT.store(index, Ordering::Relaxed);
    #[cfg(feature = "thread-local-language")]
    CURRENT.with(|current| current.store(index, Ordering::Relaxed));
}

thread_local! {
    /// Set while this thread writes its texts in English ([`in_english`]),
    /// whatever the library's (or, with `thread-local-language`, this
    /// thread's) language is.
    static IN_ENGLISH: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread writes its texts in English just now
/// ([`in_english`]).
fn writing_english() -> bool {
    IN_ENGLISH.with(Cell::get)
}

/// The language of the interface (English until eov says otherwise), and
/// English while this thread runs [`in_english`].
pub fn current() -> &'static Language {
    if writing_english() {
        return source();
    }
    &LANGUAGES[current_index().min(LANGUAGES.len() - 1)]
}

/// Runs `write` with this thread's texts written as in English:
/// [`tr!`](crate::tr!) gives the English text (filled in, and with the
/// English plural form), and everything that follows [`current`] does as
/// for English: [`isolate`] adds no marks, [`format_decimal`] writes and
/// [`parse_decimal`] reads numbers as in English, [`is_rtl`] is false;
/// [`join_sentences`] joins as it does in every language. For the
/// English line of eov's log or of a file, from the code that writes the
/// shown text; see [`shown_and_english`] for both at once.
///
/// It affects only this thread, only Rust code (Slint's `@tr` texts keep
/// the interface's language) and only until `write` returns, also when it
/// panics. Calls may be nested. The language and the catalog stay as they
/// are: [`set_current`] and [`set_catalog`] called inside `write` change
/// them, and this thread follows them again once `write` returns.
pub fn in_english<R>(write: impl FnOnce() -> R) -> R {
    /// Puts this thread's language back, also when `write` panics.
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_ENGLISH.with(|english| english.set(self.0));
        }
    }
    let _restore = Restore(IN_ENGLISH.with(|english| english.replace(true)));
    write()
}

/// The text `write` makes in the interface's language (to show) and in
/// English (for eov's log, files, batch output), written by the same code:
///
/// ```
/// use eov_plugin_api::i18n::{self, tr};
/// let name = "model.onnx";
/// let (shown, english) =
///     i18n::shown_and_english(|| tr!("{} could not be loaded", i18n::isolate(name)));
/// assert_eq!(english, "model.onnx could not be loaded");
/// # assert_eq!(shown, english);
/// ```
///
/// `write` is called twice, the second time inside [`in_english`]: it
/// should only make the text. Texts it inserts follow the language being
/// written when they are made in `write` too (with `tr!`, [`isolate`],
/// [`format_decimal`]); a text made earlier is inserted as it is, so keep
/// both forms of such a text and insert the one for [`current`] (English
/// when [`Language::is_source`]).
pub fn shown_and_english(write: impl Fn() -> String) -> (String, String) {
    let shown = write();
    let english = in_english(&write);
    (shown, english)
}

/// Whether the interface is written from right to left.
pub fn is_rtl() -> bool {
    current().is_rtl()
}

/// Makes `language` this library's current language for [`current`],
/// [`isolate`], [`format_decimal`] and [`parse_decimal`], without changing
/// the catalog (see [`set_catalog`]).
pub fn set_current(language: &'static Language) {
    if let Some(index) = LANGUAGES.iter().position(|l| std::ptr::eq(l, language)) {
        set_current_index(index);
    }
}

// ---------------------------------------------------------------------------
// Catalogs
// ---------------------------------------------------------------------------

/// The translations of one domain (eov, or a plugin) into one language,
/// read from a gettext `.po` file.
///
/// Only translated entries are used, as gettext's `msgfmt` compiles them:
/// untranslated, fuzzy and obsolete ones are left out, so their texts show
/// in English. Plural forms follow the catalog's `Plural-Forms` header.
pub struct Catalog {
    language: &'static Language,
    /// The forms of each translated text (one, or one per plural form), by
    /// context and English text.
    messages: HashMap<(Option<String>, String), Vec<String>>,
    plural: PluralRule,
}

impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog")
            .field("language", &self.language.tag)
            .field("messages", &self.messages.len())
            .finish()
    }
}

thread_local! {
    /// Set while this thread reads a `.po` file (see [`reading_catalog`]).
    static READING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is reading a `.po` file. rspolib, the reader Slint
/// uses too, panics on some malformed files; eov catches those panics and
/// reports a bad catalog, and its panic hook asks this so that it does not
/// record them as crashes.
pub fn reading_catalog() -> bool {
    READING.with(std::cell::Cell::get)
}

/// Reads a `.po` text with rspolib, turning its panics on malformed files
/// into errors.
fn read_po(text: &str) -> Result<rspolib::POFile, String> {
    READING.with(|reading| reading.set(true));
    let result = std::panic::catch_unwind(|| rspolib::pofile(text));
    READING.with(|reading| reading.set(false));
    match result {
        Ok(Ok(file)) => Ok(file),
        Ok(Err(error)) => Err(error.to_string()),
        Err(payload) => Err(
            match payload
                .downcast_ref::<&str>()
                .map(|text| text.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
            {
                Some(text) => format!("malformed ({text})"),
                None => "malformed".to_string(),
            },
        ),
    }
}

/// Why a catalog could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogError {
    pub message: String,
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CatalogError {}

impl Catalog {
    /// Reads a catalog of `language` from the contents of a `.po` file.
    ///
    /// The file is read with rspolib (Slint reads its catalogs with it
    /// too). Its plural rule comes from the `Plural-Forms` header, or from
    /// the language when the header has none.
    pub fn parse(language: &'static Language, po: &[u8]) -> Result<Self, CatalogError> {
        use rspolib::prelude::*;
        let error = |message: String| CatalogError { message };
        let text = std::str::from_utf8(po)
            .map_err(|e| error(format!("not UTF-8: {e}")))?
            .trim_start_matches('\u{feff}')
            .replace("\r\n", "\n");
        if !text.contains("msgid") {
            return Err(error("not a .po file: it has no msgid".to_string()));
        }
        // rspolib takes a path or the text of a file; a text spans several
        // lines, so it is never taken for a path.
        let file = read_po(&text).map_err(|e| error(format!("not a valid .po file: {e}")))?;
        let forms = file
            .metadata
            .get("Plural-Forms")
            .map(String::as_str)
            .unwrap_or(language.plural_forms);
        let plural = PluralRule::from_header(forms)
            .map_err(|e| error(format!("Plural-Forms \"{forms}\": {e}")))?;
        let messages = file
            .entries
            .iter()
            .filter(|entry| entry.translated() && !entry.msgid.is_empty())
            .map(|entry| {
                let forms = match &entry.msgstr {
                    Some(text) if entry.msgid_plural.is_none() => vec![text.clone()],
                    _ => entry.msgstr_plural.clone(),
                };
                ((entry.msgctxt.clone(), entry.msgid.clone()), forms)
            })
            .collect();
        Ok(Self {
            language,
            messages,
            plural,
        })
    }

    /// A catalog from texts read beforehand (eov compiles its own into the
    /// program): `(context, English text, forms)`, with the forms of a
    /// plural text in the order of `plural_forms` (a `Plural-Forms`
    /// header; the language's when empty). Texts with no form or an empty
    /// one are left out.
    pub fn from_messages<'a>(
        language: &'static Language,
        plural_forms: &str,
        messages: impl IntoIterator<Item = (Option<&'a str>, &'a str, &'a [&'a str])>,
    ) -> Result<Self, CatalogError> {
        let forms = if plural_forms.trim().is_empty() {
            language.plural_forms
        } else {
            plural_forms
        };
        let plural = PluralRule::from_header(forms).map_err(|e| CatalogError {
            message: format!("Plural-Forms \"{forms}\": {e}"),
        })?;
        let messages = messages
            .into_iter()
            .filter(|(_, msgid, forms)| {
                !msgid.is_empty() && !forms.is_empty() && forms.iter().all(|f| !f.is_empty())
            })
            .map(|(context, msgid, forms)| {
                (
                    (context.map(str::to_string), msgid.to_string()),
                    forms.iter().map(|f| f.to_string()).collect(),
                )
            })
            .collect();
        Ok(Self {
            language,
            messages,
            plural,
        })
    }

    /// Reads the catalog of `domain` for `language` from `dir` (see
    /// [`find_catalog`]); `Ok(None)` when there is none.
    pub fn load(
        dir: &Path,
        domain: &str,
        language: &'static Language,
    ) -> Result<Option<Self>, CatalogError> {
        let Some(path) = find_catalog(dir, domain, language) else {
            return Ok(None);
        };
        let bytes = std::fs::read(&path).map_err(|e| CatalogError {
            message: format!("cannot read {}: {e}", path.display()),
        })?;
        Self::parse(language, &bytes)
            .map(Some)
            .map_err(|e| CatalogError {
                message: format!("{}: {}", path.display(), e.message),
            })
    }

    pub fn language(&self) -> &'static Language {
        self.language
    }

    /// The number of translated messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    fn lookup(&self, msgctxt: Option<&str>, msgid: &str) -> Option<&[String]> {
        // Borrowed lookups would need a custom key type; texts are short.
        self.messages
            .get(&(msgctxt.map(str::to_string), msgid.to_string()))
            .map(Vec::as_slice)
    }

    /// The translation of `msgid` with the context `msgctxt`, if the
    /// catalog has one.
    pub fn translate(&self, msgctxt: Option<&str>, msgid: &str) -> Option<&str> {
        self.lookup(msgctxt, msgid)?.first().map(String::as_str)
    }

    /// The form of a text with a number `n`, if the catalog translates it.
    pub fn ntranslate(&self, msgctxt: Option<&str>, singular: &str, n: u64) -> Option<&str> {
        let forms = self.lookup(msgctxt, singular)?;
        forms
            .get(self.plural.form(n))
            .filter(|form| !form.is_empty())
            .map(String::as_str)
    }

    /// The plural form a number `n` takes in this catalog's language.
    pub fn plural_form(&self, n: u64) -> usize {
        self.plural.form(n)
    }
}

/// A gettext plural rule: the C expression of a `Plural-Forms` header
/// (`n%10==1 && n%100!=11 ? 0 : …`) that gives the form a number takes.
#[derive(Debug, Clone, PartialEq)]
pub struct PluralRule {
    count: usize,
    expression: Expression,
}

#[derive(Debug, Clone, PartialEq)]
enum Expression {
    N,
    Number(u64),
    Not(Box<Expression>),
    Binary(Box<Expression>, BinaryOp, Box<Expression>),
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl BinaryOp {
    /// Binding strength, as in C (higher binds tighter).
    fn precedence(self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::Ne => 3,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => 6,
        }
    }

    fn apply(self, a: u64, b: u64) -> u64 {
        match self {
            BinaryOp::Or => u64::from(a != 0 || b != 0),
            BinaryOp::And => u64::from(a != 0 && b != 0),
            BinaryOp::Eq => u64::from(a == b),
            BinaryOp::Ne => u64::from(a != b),
            BinaryOp::Lt => u64::from(a < b),
            BinaryOp::Le => u64::from(a <= b),
            BinaryOp::Gt => u64::from(a > b),
            BinaryOp::Ge => u64::from(a >= b),
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            // gettext's own evaluator gives 0 for a division by zero.
            BinaryOp::Div => a.checked_div(b).unwrap_or(0),
            BinaryOp::Rem => a.checked_rem(b).unwrap_or(0),
        }
    }
}

impl Expression {
    fn evaluate(&self, n: u64) -> u64 {
        match self {
            Expression::N => n,
            Expression::Number(value) => *value,
            Expression::Not(inner) => u64::from(inner.evaluate(n) == 0),
            Expression::Binary(a, op, b) => match op {
                // Short-circuit, as C does.
                BinaryOp::Or if a.evaluate(n) != 0 => 1,
                BinaryOp::And if a.evaluate(n) == 0 => 0,
                _ => op.apply(a.evaluate(n), b.evaluate(n)),
            },
            Expression::Conditional(condition, then, otherwise) => {
                if condition.evaluate(n) != 0 {
                    then.evaluate(n)
                } else {
                    otherwise.evaluate(n)
                }
            }
        }
    }
}

/// Reads the plural expression (the grammar of GNU gettext's
/// `plural.y`): conditionals, `||`, `&&`, comparisons, arithmetic, `!`,
/// parentheses, `n` and unsigned numbers.
struct ExpressionParser<'a> {
    text: &'a [u8],
    at: usize,
}

impl<'a> ExpressionParser<'a> {
    fn skip_space(&mut self) {
        while self.text.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_space();
        self.text.get(self.at).copied()
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.text[self.at..].starts_with(token.as_bytes()) {
            self.at += token.len();
            true
        } else {
            false
        }
    }

    fn conditional(&mut self) -> Result<Expression, String> {
        let condition = self.binary(1)?;
        if self.eat("?") {
            let then = self.conditional()?;
            if !self.eat(":") {
                return Err(format!("':' expected at {}", self.at));
            }
            let otherwise = self.conditional()?;
            return Ok(Expression::Conditional(
                Box::new(condition),
                Box::new(then),
                Box::new(otherwise),
            ));
        }
        Ok(condition)
    }

    fn operator(&mut self) -> Option<(BinaryOp, usize)> {
        self.skip_space();
        let rest = &self.text[self.at..];
        let two = |token: &[u8]| rest.starts_with(token);
        Some(if two(b"||") {
            (BinaryOp::Or, 2)
        } else if two(b"&&") {
            (BinaryOp::And, 2)
        } else if two(b"==") {
            (BinaryOp::Eq, 2)
        } else if two(b"!=") {
            (BinaryOp::Ne, 2)
        } else if two(b"<=") {
            (BinaryOp::Le, 2)
        } else if two(b">=") {
            (BinaryOp::Ge, 2)
        } else {
            match rest.first()? {
                b'<' => (BinaryOp::Lt, 1),
                b'>' => (BinaryOp::Gt, 1),
                b'+' => (BinaryOp::Add, 1),
                b'-' => (BinaryOp::Sub, 1),
                b'*' => (BinaryOp::Mul, 1),
                b'/' => (BinaryOp::Div, 1),
                b'%' => (BinaryOp::Rem, 1),
                _ => return None,
            }
        })
    }

    /// Binary operators binding at least as tightly as `min` (precedence
    /// climbing; all of them associate to the left).
    fn binary(&mut self, min: u8) -> Result<Expression, String> {
        let mut left = self.unary()?;
        while let Some((op, len)) = self.operator() {
            if op.precedence() < min {
                break;
            }
            self.at += len;
            let right = self.binary(op.precedence() + 1)?;
            left = Expression::Binary(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expression, String> {
        match self.peek() {
            Some(b'!') => {
                self.at += 1;
                Ok(Expression::Not(Box::new(self.unary()?)))
            }
            Some(b'(') => {
                self.at += 1;
                let inner = self.conditional()?;
                if !self.eat(")") {
                    return Err(format!("')' expected at {}", self.at));
                }
                Ok(inner)
            }
            Some(b'n') => {
                self.at += 1;
                Ok(Expression::N)
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.at;
                while self.text.get(self.at).is_some_and(u8::is_ascii_digit) {
                    self.at += 1;
                }
                std::str::from_utf8(&self.text[start..self.at])
                    .ok()
                    .and_then(|digits| digits.parse().ok())
                    .map(Expression::Number)
                    .ok_or_else(|| format!("number too large at {start}"))
            }
            Some(c) => Err(format!("unexpected '{}' at {}", c as char, self.at)),
            None => Err("unexpected end".to_string()),
        }
    }
}

impl PluralRule {
    /// Reads a `Plural-Forms` header value
    /// (`nplurals=2; plural=(n != 1);`).
    pub fn from_header(header: &str) -> Result<Self, String> {
        let mut count = None;
        let mut expression = None;
        for part in header.split(';') {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            match key.trim() {
                "nplurals" => {
                    count = Some(
                        value
                            .trim()
                            .parse::<usize>()
                            .map_err(|_| format!("nplurals is not a number: {value}"))?,
                    )
                }
                "plural" => expression = Some(value.trim()),
                _ => {}
            }
        }
        let count = count.ok_or("nplurals is missing")?;
        if !(1..=6).contains(&count) {
            return Err(format!("nplurals must be 1 to 6, not {count}"));
        }
        let text = expression.ok_or("plural is missing")?;
        let mut parser = ExpressionParser {
            text: text.as_bytes(),
            at: 0,
        };
        let expression = parser.conditional()?;
        parser.skip_space();
        if parser.at != text.len() {
            return Err(format!(
                "unexpected text after the expression: {}",
                &text[parser.at..]
            ));
        }
        Ok(Self { count, expression })
    }

    /// How many forms a text with a number has.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The form `n` takes: always less than [`Self::count`].
    pub fn form(&self, n: u64) -> usize {
        usize::try_from(self.expression.evaluate(n))
            .unwrap_or(usize::MAX)
            .min(self.count - 1)
    }
}

/// What [`check_catalog`] found in a catalog.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogReport {
    pub translated: usize,
    pub fuzzy: usize,
    pub untranslated: usize,
    /// Texts of the template the catalog lacks (`context\u{4}text`).
    pub missing: Vec<String>,
    /// Texts the catalog has but the template no longer does.
    pub obsolete: Vec<String>,
    /// What would show wrongly: a header that does not fit the language,
    /// plural forms that are missing, placeholders that are lost or made up.
    pub problems: Vec<String>,
}

impl CatalogReport {
    /// Whether every text of the template is translated, and nothing is
    /// wrong.
    pub fn is_complete(&self) -> bool {
        self.fuzzy == 0
            && self.untranslated == 0
            && self.missing.is_empty()
            && self.problems.is_empty()
    }
}

/// The values a text inserts, as `format` and Slint fill them: the k-th
/// `{}` and `{k}` are the same value, so a translation may write `{1} … {0}`
/// to put them in another order.
fn inserted_values(text: &str) -> std::collections::BTreeSet<String> {
    let mut next = 0;
    placeholders_in_order(text)
        .into_iter()
        .map(|placeholder| {
            if placeholder == "{}" {
                next += 1;
                format!("{{{}}}", next - 1)
            } else {
                placeholder.to_string()
            }
        })
        .collect()
}

/// The placeholders of a text (`{}`, `{0}`, `{name}`; not `{{` or `}}`).
pub fn placeholders(text: &str) -> std::collections::BTreeSet<&str> {
    placeholders_in_order(text).into_iter().collect()
}

fn placeholders_in_order(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find('{') {
        let tail = &rest[at..];
        if let Some(after) = tail.strip_prefix("{{") {
            rest = after;
            continue;
        }
        match tail.find('}') {
            Some(end) if !tail[1..end].contains('{') => {
                found.push(&tail[..=end]);
                rest = &tail[end + 1..];
            }
            _ => rest = &tail[1..],
        }
    }
    found
}

/// Checks a catalog of `language` (the text of its `.po` file) against
/// what eov needs to show it, and, when given, its template (`.pot`):
///
/// - the header names the language and its plural forms;
/// - a text with a number has as many forms as the language has, and each
///   form that numbers other than one take shows the number (`{n}`) when
///   the English does;
/// - every translation keeps the English text's placeholders and adds none
///   (`{}` and `{0}` stand for values eov inserts).
///
/// It also counts translated, fuzzy and untranslated texts, and lists the
/// template's texts the catalog lacks.
pub fn check_catalog(
    language: &'static Language,
    po: &str,
    template: Option<&str>,
) -> CatalogReport {
    use rspolib::prelude::*;
    let mut report = CatalogReport::default();
    let parse = |text: &str| {
        let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n");
        read_po(&text)
    };
    let file = match parse(po) {
        Ok(file) => file,
        Err(error) => {
            report
                .problems
                .push(format!("not a valid .po file: {error}"));
            return report;
        }
    };
    let compact = |text: &str| text.split_whitespace().collect::<String>();
    match file.metadata.get("Language") {
        Some(tag) if tag.trim() == language.tag => {}
        other => report.problems.push(format!(
            "the header says Language: {}, not {}",
            other.map(String::as_str).unwrap_or("(none)"),
            language.tag
        )),
    }
    let rule = match file.metadata.get("Plural-Forms") {
        Some(forms) if compact(forms) == compact(language.plural_forms) => {
            PluralRule::from_header(forms).ok()
        }
        other => {
            report.problems.push(format!(
                "the header's Plural-Forms is {}, not {}",
                other.map(String::as_str).unwrap_or("(none)"),
                language.plural_forms
            ));
            None
        }
    };
    let key = |context: &Option<String>, msgid: &str| match context {
        Some(context) => format!("{context}\u{4}{msgid}"),
        None => msgid.to_string(),
    };
    let mut keys = std::collections::BTreeSet::new();
    for entry in file
        .entries
        .iter()
        .filter(|e| !e.obsolete && !e.msgid.is_empty())
    {
        let name = key(&entry.msgctxt, &entry.msgid);
        keys.insert(name.clone());
        if entry.fuzzy() {
            report.fuzzy += 1;
            continue;
        }
        if !entry.translated() {
            report.untranslated += 1;
            continue;
        }
        report.translated += 1;
        let plural = entry.msgid_plural.as_deref();
        let sources: std::collections::BTreeSet<String> = inserted_values(&entry.msgid)
            .union(&inserted_values(plural.unwrap_or_default()))
            .cloned()
            .collect();
        let forms: Vec<&str> = match plural {
            Some(_) => entry.msgstr_plural.iter().map(String::as_str).collect(),
            None => entry.msgstr.iter().map(String::as_str).collect(),
        };
        if plural.is_some() && forms.len() != language.plural_count() {
            report.problems.push(format!(
                "{name:?}: {} plural forms, {} has {}",
                forms.len(),
                language.english_name,
                language.plural_count()
            ));
        }
        for (index, form) in forms.iter().enumerate() {
            let found = inserted_values(form);
            let made_up: Vec<&String> = found.difference(&sources).collect();
            if !made_up.is_empty() {
                report
                    .problems
                    .push(format!("{name:?}: the translation adds {made_up:?}"));
            }
            let lost: Vec<&String> = sources
                .iter()
                .filter(|p| p.as_str() != "{n}" && !found.contains(*p))
                .collect();
            if !lost.is_empty() {
                report
                    .problems
                    .push(format!("{name:?}: the translation loses {lost:?}"));
            }
            // A form that numbers other than 1 take must show the number.
            let shows_counts = match (&rule, plural) {
                (Some(rule), Some(plural)) if plural.contains("{n}") => {
                    (0..=200u64).any(|n| n != 1 && rule.form(n) == index)
                }
                (_, None) => entry.msgid.contains("{n}"),
                _ => false,
            };
            if shows_counts && !found.contains("{n}") {
                report.problems.push(format!(
                    "{name:?}: form {index} is used for counts other than 1 but does not show {{n}}"
                ));
            }
        }
    }
    if let Some(template) = template {
        match parse(template) {
            Ok(template) => {
                let wanted: std::collections::BTreeSet<String> = template
                    .entries
                    .iter()
                    .filter(|e| !e.obsolete && !e.msgid.is_empty())
                    .map(|e| key(&e.msgctxt, &e.msgid))
                    .collect();
                report.missing = wanted.difference(&keys).cloned().collect();
                report.obsolete = keys.difference(&wanted).cloned().collect();
            }
            Err(error) => report
                .problems
                .push(format!("the template is not valid: {error}")),
        }
    }
    report
}

/// Where the catalog of `domain` for `language` belongs:
/// `<dir>/<tag>/LC_MESSAGES/<domain>.po`.
pub fn catalog_path(dir: &Path, domain: &str, language: &Language) -> PathBuf {
    dir.join(language.tag)
        .join("LC_MESSAGES")
        .join(format!("{domain}.po"))
}

/// The catalog of `domain` for `language` in `dir`: in the tag's folder
/// (`pt-BR`), else under another spelling gettext tools use
/// ([`Language::catalog_folders`]: `pt_BR`, `zh_TW`). English has none.
pub fn find_catalog(dir: &Path, domain: &str, language: &Language) -> Option<PathBuf> {
    if language.is_source() {
        return None;
    }
    language
        .catalog_folders()
        .into_iter()
        .map(|folder| {
            dir.join(folder)
                .join("LC_MESSAGES")
                .join(format!("{domain}.po"))
        })
        .find(|path| path.is_file())
}

/// A catalog without translations for `language`: just the header, with the
/// language's plural forms. eov puts one in for each language a plugin has
/// no catalog for, so that every bundle lists the same languages.
pub fn empty_catalog(language: &Language, domain: &str) -> String {
    format!(
        "msgid \"\"\nmsgstr \"\"\n\"Project-Id-Version: {domain}\\n\"\n\"Language: {tag}\\n\"\n\
         \"MIME-Version: 1.0\\n\"\n\"Content-Type: text/plain; charset=UTF-8\\n\"\n\
         \"Content-Transfer-Encoding: 8bit\\n\"\n\"Plural-Forms: {plural}\\n\"\n",
        tag = language.tag,
        plural = language.plural_forms,
    )
}

// ---------------------------------------------------------------------------
// Translating
// ---------------------------------------------------------------------------

/// The catalog [`tr!`](crate::tr!) translates with. Each library (eov, and
/// each plugin) has its own: statics are not shared between libraries.
#[cfg(not(feature = "thread-local-language"))]
static CATALOG: RwLock<Option<Catalog>> = RwLock::new(None);

/// Runs `f` with this library's catalog (this thread's with
/// `thread-local-language`).
fn with_catalog<R>(f: impl FnOnce(&RwLock<Option<Catalog>>) -> R) -> R {
    #[cfg(not(feature = "thread-local-language"))]
    return f(&CATALOG);
    #[cfg(feature = "thread-local-language")]
    return CATALOG.with(f);
}

/// Makes `catalog` the one [`tr!`](crate::tr!) translates with, and its
/// language the [`current`] one; `None` shows the English texts. eov
/// and the plugin SDK call this when the language changes.
pub fn set_catalog(language: &'static Language, catalog: Option<Catalog>) {
    set_current(language);
    let catalog = catalog.filter(|c| std::ptr::eq(c.language(), language));
    with_catalog(|slot| {
        *slot
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = catalog
    });
}

/// The translation of `msgid` (with the context `msgctxt`), or `msgid`
/// (always inside [`in_english`]).
pub fn translate(msgctxt: Option<&str>, msgid: &str) -> String {
    if writing_english() {
        return msgid.to_string();
    }
    with_catalog(|slot| {
        let catalog = slot.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        catalog
            .as_ref()
            .and_then(|catalog| catalog.translate(msgctxt, msgid))
            .unwrap_or(msgid)
            .to_string()
    })
}

/// The form of a text with a number `n` in the current language (in
/// English inside [`in_english`]).
pub fn ntranslate(msgctxt: Option<&str>, singular: &str, plural: &str, n: u64) -> String {
    if writing_english() {
        return if n == 1 { singular } else { plural }.to_string();
    }
    with_catalog(|slot| {
        let catalog = slot.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        match catalog
            .as_ref()
            .and_then(|catalog| catalog.ntranslate(msgctxt, singular, n))
        {
            Some(translation) => translation.to_string(),
            // English: one, or many (including none).
            None if n == 1 => singular.to_string(),
            None => plural.to_string(),
        }
    })
}

/// Fills the placeholders of a translated text, as Slint's `@tr` does:
/// `{}` takes the next argument, `{0}` the first, `{name}` the one with that
/// name; `{{` and `}}` are braces. A placeholder without its argument stays
/// as it is.
pub fn format(template: &str, args: &[(&str, &dyn fmt::Display)]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(template.len() + 16);
    let mut next = 0;
    let mut rest = template;
    while let Some(at) = rest.find(['{', '}']) {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
            continue;
        }
        if let Some(after) = tail.strip_prefix('}') {
            out.push('}');
            rest = after;
            continue;
        }
        let Some(end) = tail.find('}') else {
            out.push_str(tail);
            rest = "";
            break;
        };
        let name = tail[1..end].trim();
        let arg = if name.is_empty() {
            next += 1;
            args.get(next - 1)
        } else if let Ok(index) = name.parse::<usize>() {
            args.get(index)
        } else {
            args.iter().find(|(arg_name, _)| *arg_name == name)
        };
        match arg {
            Some((_, value)) => {
                let _ = write!(out, "{value}");
            }
            None => out.push_str(&tail[..=end]),
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Translates a text written in English, in the current language of this
/// library (see the [module documentation](crate::i18n)):
///
/// - `tr!("text")`, `tr!("text with {}", value)`, `tr!("{name}", name = value)`;
/// - `tr!("context" => "text", …)` for a text that needs a context to be
///   translated right (the same English word for different things);
/// - `tr!("one file" | "{n} files" % count, …)` for a text with a number:
///   each language has its own forms, and `{n}` is the number.
///
/// It returns a `String`. Arguments are formatted with `Display`.
#[macro_export]
macro_rules! tr {
    ($msgctxt:tt => $singular:tt | $plural:tt % $n:expr $(, $($args:tt)*)?) => {{
        let n = $n;
        $crate::i18n::format(
            &$crate::i18n::ntranslate(::core::option::Option::Some($msgctxt), $singular, $plural, n as u64),
            $crate::__tr_args!([] $($($args)*)? , n = n),
        )
    }};
    ($msgctxt:tt => $msgid:tt $(, $($args:tt)*)?) => {
        $crate::i18n::format(
            &$crate::i18n::translate(::core::option::Option::Some($msgctxt), $msgid),
            $crate::__tr_args!([] $($($args)*)?),
        )
    };
    ($singular:tt | $plural:tt % $n:expr $(, $($args:tt)*)?) => {{
        let n = $n;
        $crate::i18n::format(
            &$crate::i18n::ntranslate(::core::option::Option::None, $singular, $plural, n as u64),
            $crate::__tr_args!([] $($($args)*)? , n = n),
        )
    }};
    ($msgid:tt $(, $($args:tt)*)?) => {
        $crate::i18n::format(
            &$crate::i18n::translate(::core::option::Option::None, $msgid),
            $crate::__tr_args!([] $($($args)*)?),
        )
    };
}

/// The arguments of [`tr!`] as `(name, value)` pairs; not public API.
#[doc(hidden)]
#[macro_export]
macro_rules! __tr_args {
    ([$($done:tt)*]) => { &[$($done)*] };
    ([$($done:tt)*] ,) => { &[$($done)*] };
    ([$($done:tt)*] , $($rest:tt)+) => { $crate::__tr_args!([$($done)*] $($rest)+) };
    ([$($done:tt)*] $name:ident = $value:expr , $($rest:tt)*) => {
        $crate::__tr_args!([$($done)* (::core::stringify!($name), &$value as &dyn ::core::fmt::Display),] , $($rest)*)
    };
    ([$($done:tt)*] $name:ident = $value:expr) => {
        $crate::__tr_args!([$($done)* (::core::stringify!($name), &$value as &dyn ::core::fmt::Display),])
    };
    ([$($done:tt)*] $name:ident , $($rest:tt)*) => {
        $crate::__tr_args!([$($done)* (::core::stringify!($name), &$name as &dyn ::core::fmt::Display),] , $($rest)*)
    };
    ([$($done:tt)*] $name:ident) => {
        $crate::__tr_args!([$($done)* (::core::stringify!($name), &$name as &dyn ::core::fmt::Display),])
    };
    ([$($done:tt)*] $value:expr , $($rest:tt)*) => {
        $crate::__tr_args!([$($done)* ("", &$value as &dyn ::core::fmt::Display),] , $($rest)*)
    };
    ([$($done:tt)*] $value:expr) => {
        $crate::__tr_args!([$($done)* ("", &$value as &dyn ::core::fmt::Display),])
    };
}

// ---------------------------------------------------------------------------
// Direction and numbers
// ---------------------------------------------------------------------------

/// First strong isolate: starts a run whose direction comes from its first
/// letter, without changing the order of the text around it.
const FSI: char = '\u{2068}';
/// Pop directional isolate: ends the run [`FSI`] started.
const PDI: char = '\u{2069}';

/// `text` for insertion into a translated text: in a right-to-left
/// language it is isolated (U+2068 … U+2069), so that a file name, a path or
/// a number with its unit keeps its own order and does not reorder the text
/// around it. In left-to-right languages it is returned as it is.
pub fn isolate(text: impl fmt::Display) -> String {
    isolate_for(current(), text)
}

/// [`isolate`] for a given language.
pub fn isolate_for(language: &Language, text: impl fmt::Display) -> String {
    if language.is_rtl() {
        format!("{FSI}{text}{PDI}")
    } else {
        text.to_string()
    }
}

/// Separately translated sentences as one message, one after the other:
/// with a space between them, except after a sentence that ends in
/// full-width punctuation (`。`, `！`, `？`, a full-width closing bracket),
/// which Chinese and Japanese follow without a space. Empty sentences are
/// left out.
///
/// Build a message from whole sentences this way rather than with
/// `push(' ')` or `join(" ")`: then each sentence is translated on its own
/// and the space follows the writing of the language, including when one of
/// the sentences falls back to English.
pub fn join_sentences<S: AsRef<str>>(sentences: impl IntoIterator<Item = S>) -> String {
    /// Ends of a sentence after which Chinese and Japanese put no space.
    const FULL_WIDTH_ENDS: [char; 14] = [
        '。', '｡', '．', '！', '？', '）', '」', '』', '】', '〕', '〉', '》', '］', '｝',
    ];
    let mut text = String::new();
    for sentence in sentences {
        let sentence = sentence.as_ref().trim();
        if sentence.is_empty() {
            continue;
        }
        if !text.is_empty() && !text.ends_with(FULL_WIDTH_ENDS) {
            text.push(' ');
        }
        text.push_str(sentence);
    }
    text
}

/// `value` with `decimals` digits after the decimal separator of the
/// current language (`1,50` in German, `1.50` in English).
pub fn format_decimal(value: f64, decimals: usize) -> String {
    format_decimal_for(current(), value, decimals)
}

/// [`format_decimal`] for a given language.
pub fn format_decimal_for(language: &Language, value: f64, decimals: usize) -> String {
    let text = format!("{value:.decimals$}");
    if language.decimal_separator == '.' {
        text
    } else {
        text.replacen('.', &language.decimal_separator.to_string(), 1)
    }
}

/// Reads a number a user typed: with the current language's decimal
/// separator, or with `.`, `,` or the Arabic decimal sign (people type what
/// they are used to), in Western, Arabic-Indic or Persian digits.
///
/// What could as well be a number with a thousands separator is refused
/// rather than guessed: two separators (`1.234,5`), and a whole number
/// written with the language's thousands separator (`1,600` where the
/// decimal separator is not `,`, `1.600` where it is). `0,25`, `2,5` and
/// `1234,5` are read as decimals in every language.
pub fn parse_decimal(text: &str) -> Option<f64> {
    parse_decimal_for(current(), text)
}

/// [`parse_decimal`] for a given language.
pub fn parse_decimal_for(language: &Language, text: &str) -> Option<f64> {
    // Arabic-Indic and Persian digits, as their keyboards type them.
    let text: String = text
        .trim()
        .chars()
        .map(|c| match c {
            '\u{0660}'..='\u{0669}' => char::from(b'0' + (c as u32 - 0x0660) as u8),
            '\u{06F0}'..='\u{06F9}' => char::from(b'0' + (c as u32 - 0x06F0) as u8),
            c => c,
        })
        .collect();
    let thousands = if language.decimal_separator == ',' {
        '.'
    } else {
        ','
    };
    if has_thousands(&text, thousands) {
        return None;
    }
    let normalized: String = text
        .chars()
        .map(|c| match c {
            // The Arabic decimal sign is never a thousands separator (that is
            // U+066C, which is refused like any other character).
            '\u{066B}' | ',' => '.',
            '\u{2212}' => '-',
            c if c == language.decimal_separator => '.',
            c => c,
        })
        .collect();
    if normalized.matches('.').count() > 1 {
        return None;
    }
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Whether `text` (in Western digits) is a whole number written with one
/// `separator` between its thousands (`1,600`, `-12,500`): one to three
/// digits, the first not 0, then the separator and three digits.
fn has_thousands(text: &str, separator: char) -> bool {
    let text = text.strip_prefix(['+', '-', '\u{2212}']).unwrap_or(text);
    let Some((thousands, rest)) = text.split_once(separator) else {
        return false;
    };
    let digits = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
    (1..=3).contains(&thousands.len())
        && !thousands.starts_with('0')
        && digits(thousands)
        && rest.len() == 3
        && digits(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_languages_are_complete_and_ordered() {
        assert_eq!(LANGUAGES.len(), 31);
        assert!(source().is_source());
        let tags: Vec<&str> = translations().iter().map(|l| l.tag).collect();
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        // Slint bundles catalogs in the order of their folder names; this
        // order is that one.
        assert_eq!(tags, sorted);
        for expected in [
            "zh-Hans", "ru", "pt-BR", "fr", "de", "es", "sl", "cs", "uk", "ar", "hi", "el", "ja",
            "ko", "it", "tr", "pl", "zh-Hant", "nl", "id", "vi", "th", "fa", "sv", "ro", "hu",
            "nb", "fi", "da", "he",
        ] {
            assert!(by_tag(expected).is_some(), "{expected} is missing");
        }
        let rtl: Vec<&str> = LANGUAGES
            .iter()
            .filter(|l| l.is_rtl())
            .map(|l| l.tag)
            .collect();
        assert_eq!(rtl, ["ar", "fa", "he"]);
    }

    #[test]
    fn tags_are_matched_without_regard_to_case_and_separator() {
        assert_eq!(by_tag("pt-BR").unwrap().tag, "pt-BR");
        assert_eq!(by_tag("pt_br").unwrap().tag, "pt-BR");
        assert_eq!(by_tag("ZH-hant").unwrap().tag, "zh-Hant");
        assert_eq!(by_tag(" en ").unwrap().tag, "en");
        assert!(by_tag("pt").is_none());
        assert!(by_tag("xx").is_none());
    }

    fn negotiated(locale: &str) -> Option<&'static str> {
        negotiate(locale).map(|l| l.tag)
    }

    #[test]
    fn system_locales_are_negotiated() {
        // POSIX names, with codesets and modifiers.
        assert_eq!(negotiated("de_DE.UTF-8"), Some("de"));
        assert_eq!(negotiated("de_AT.UTF-8@euro"), Some("de"));
        assert_eq!(negotiated("fr_CA"), Some("fr"));
        assert_eq!(negotiated("es-419"), Some("es"));
        assert_eq!(negotiated("en-GB"), Some("en"));
        assert_eq!(negotiated("ja_JP.eucJP"), Some("ja"));
        assert_eq!(negotiated("ar-EG"), Some("ar"));
        assert_eq!(negotiated("fa_IR"), Some("fa"));
        assert_eq!(negotiated("he_IL.UTF-8"), Some("he"));
        // Old codes.
        assert_eq!(negotiated("iw_IL"), Some("he"));
        assert_eq!(negotiated("in_ID"), Some("id"));
        assert_eq!(negotiated("no_NO"), Some("nb"));
        assert_eq!(negotiated("nb-NO"), Some("nb"));
        assert_eq!(negotiated("nn_NO"), Some("nb"));
        // No preference, and languages without a translation.
        assert_eq!(negotiated("C"), None);
        assert_eq!(negotiated("POSIX"), None);
        assert_eq!(negotiated("C.UTF-8"), None);
        assert_eq!(negotiated(""), None);
        assert_eq!(negotiated("sr_RS"), None);
        assert_eq!(negotiated("ms_MY"), None);
        assert_eq!(negotiated("x-klingon"), None);
        // An extension does not change the language.
        assert_eq!(negotiated("de-DE-u-co-phonebk"), Some("de"));
    }

    #[test]
    fn chinese_follows_the_script() {
        for (locale, tag) in [
            ("zh_CN.UTF-8", "zh-Hans"),
            ("zh-SG", "zh-Hans"),
            ("zh-MY", "zh-Hans"),
            ("zh", "zh-Hans"),
            ("zh-Hans", "zh-Hans"),
            ("zh-Hans-HK", "zh-Hans"),
            ("zh_TW", "zh-Hant"),
            ("zh-HK", "zh-Hant"),
            ("zh_MO.UTF-8", "zh-Hant"),
            ("zh-Hant", "zh-Hant"),
            ("zh-Hant-CN", "zh-Hant"),
        ] {
            assert_eq!(negotiated(locale), Some(tag), "{locale}");
        }
        // Chinese in another script (pinyin) has no translation.
        assert_eq!(negotiated("zh-Latn"), None);
    }

    #[test]
    fn portuguese_is_brazilian_only_where_it_is_written_so() {
        assert_eq!(negotiated("pt_BR.UTF-8"), Some("pt-BR"));
        assert_eq!(negotiated("pt-BR"), Some("pt-BR"));
        assert_eq!(negotiated("pt"), Some("pt-BR"));
        for european in ["pt_PT", "pt-PT", "pt-AO", "pt_MZ.UTF-8"] {
            assert_eq!(negotiated(european), None, "{european}");
        }
        // The next preference is taken instead.
        assert_eq!(
            negotiate_first(["pt-PT", "es-ES", "en-US"]).map(|l| l.tag),
            Some("es")
        );
        assert_eq!(negotiate_first(["pt-PT"]).map(|l| l.tag), None);
    }

    #[test]
    fn another_script_is_not_taken_for_the_translation() {
        assert_eq!(negotiated("ru-Cyrl"), Some("ru"));
        assert_eq!(negotiated("ru-Latn"), None);
        assert_eq!(negotiated("ar-Latn"), None);
        assert_eq!(negotiated("de-Latn-CH"), Some("de"));
    }

    #[test]
    fn plural_counts_are_read_from_the_header() {
        assert_eq!(by_tag("ja").unwrap().plural_count(), 1);
        assert_eq!(by_tag("de").unwrap().plural_count(), 2);
        assert_eq!(by_tag("ru").unwrap().plural_count(), 3);
        assert_eq!(by_tag("sl").unwrap().plural_count(), 4);
        assert_eq!(by_tag("ar").unwrap().plural_count(), 6);
    }

    /// A catalog for `language` with one plural text whose forms say their
    /// index, and one translated, one untranslated and one fuzzy text.
    fn sample_catalog(language: &'static Language) -> Catalog {
        let mut po = empty_catalog(language, "test");
        po.push_str("\nmsgid \"Close\"\nmsgstr \"[Close]\"\n");
        po.push_str("\nmsgctxt \"Menu\"\nmsgid \"Close\"\nmsgstr \"[Menu Close]\"\n");
        po.push_str("\nmsgid \"Untranslated\"\nmsgstr \"\"\n");
        po.push_str("\n#, fuzzy\nmsgid \"Fuzzy\"\nmsgstr \"[Fuzzy]\"\n");
        po.push_str("\nmsgid \"{n} file\"\nmsgid_plural \"{n} files\"\n");
        for form in 0..language.plural_count() {
            po.push_str(&format!("msgstr[{form}] \"form {form}: {{n}}\"\n"));
        }
        Catalog::parse(language, po.as_bytes()).unwrap()
    }

    #[test]
    fn catalogs_translate_and_fall_back_to_english() {
        let catalog = sample_catalog(by_tag("de").unwrap());
        assert_eq!(catalog.len(), 3);
        assert_eq!(catalog.translate(None, "Close"), Some("[Close]"));
        assert_eq!(
            catalog.translate(Some("Menu"), "Close"),
            Some("[Menu Close]")
        );
        // Untranslated, fuzzy and unknown texts are not in the catalog, so
        // they show in English (see `the_macro_translates…`).
        assert_eq!(catalog.translate(None, "Untranslated"), None);
        assert_eq!(catalog.translate(None, "Fuzzy"), None);
        assert_eq!(catalog.translate(None, "Unknown"), None);
        assert_eq!(catalog.translate(Some("Other"), "Close"), None);
        assert_eq!(catalog.ntranslate(None, "{n} folder", 5), None);
    }

    #[test]
    fn plural_forms_follow_each_language() {
        // The form each number takes, per language (Unicode CLDR).
        let cases: &[(&str, &[(u64, usize)])] = &[
            ("en", &[(0, 1), (1, 0), (2, 1), (101, 1)]),
            ("fr", &[(0, 0), (1, 0), (2, 1), (100, 1)]),
            ("pt-BR", &[(0, 0), (1, 0), (2, 1)]),
            ("ja", &[(0, 0), (1, 0), (2, 0), (100, 0)]),
            (
                "ru",
                &[
                    (1, 0),
                    (2, 1),
                    (4, 1),
                    (5, 2),
                    (11, 2),
                    (12, 2),
                    (21, 0),
                    (22, 1),
                    (25, 2),
                    (111, 2),
                ],
            ),
            ("uk", &[(1, 0), (3, 1), (7, 2), (31, 0)]),
            (
                "pl",
                &[(1, 0), (2, 1), (5, 2), (12, 2), (22, 1), (25, 2), (101, 2)],
            ),
            ("cs", &[(1, 0), (2, 1), (4, 1), (5, 2), (22, 2)]),
            (
                "sl",
                &[(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (101, 0), (102, 1)],
            ),
            (
                "ro",
                &[(0, 1), (1, 0), (2, 1), (19, 1), (20, 2), (101, 1), (120, 2)],
            ),
            (
                "ar",
                &[
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (10, 3),
                    (11, 4),
                    (99, 4),
                    (100, 5),
                    (102, 5),
                    (103, 3),
                ],
            ),
            ("he", &[(1, 0), (2, 1), (20, 1)]),
            ("hi", &[(0, 0), (1, 0), (2, 1)]),
        ];
        for (tag, expected) in cases {
            let language = by_tag(tag).unwrap();
            let catalog = sample_catalog(language);
            for (n, form) in *expected {
                let text = catalog.ntranslate(None, "{n} file", *n);
                assert_eq!(
                    text,
                    Some(format!("form {form}: {{n}}").as_str()),
                    "{tag} n={n}"
                );
            }
        }
    }

    #[test]
    fn every_language_has_a_valid_plural_rule() {
        for language in LANGUAGES {
            let catalog = sample_catalog(language);
            for n in 0..200 {
                let text = catalog.ntranslate(None, "{n} file", n).unwrap_or_default();
                let form: usize = text
                    .strip_prefix("form ")
                    .and_then(|rest| rest.split(':').next())
                    .and_then(|form| form.parse().ok())
                    .unwrap_or_else(|| panic!("{}: n={n} gave {text:?}", language.tag));
                assert!(form < language.plural_count(), "{}: n={n}", language.tag);
            }
        }
    }

    #[test]
    fn plural_rules_are_read_and_evaluated_as_in_c() {
        let rule = |header: &str| PluralRule::from_header(header).unwrap();
        // Precedence: comparison over &&, && over ||, % over comparison.
        let polish = rule(by_tag("pl").unwrap().plural_forms);
        assert_eq!(polish.count(), 3);
        assert_eq!(
            (0..30).map(|n| polish.form(n)).collect::<Vec<_>>(),
            [
                2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                2, 2
            ]
        );
        assert_eq!(rule("nplurals=2; plural=n*2-3 > 1+1;").form(3), 1);
        assert_eq!(rule("nplurals=2; plural=!(n == 1);").form(1), 0);
        assert_eq!(rule("nplurals=3; plural=n/2 % 3;").form(10), 2);
        // Division by zero is 0, as gettext has it; forms stay in range.
        assert_eq!(rule("nplurals=2; plural=n/0;").form(5), 0);
        assert_eq!(rule("nplurals=2; plural=n;").form(9), 1);
        // A header without spaces, and with them everywhere.
        assert_eq!(rule("nplurals=2;plural=n!=1;").form(2), 1);
        assert_eq!(rule(" nplurals = 2 ; plural = ( n != 1 ) ; ").form(1), 0);
        for bad in [
            "plural=(n != 1);",
            "nplurals=2;",
            "nplurals=x; plural=0;",
            "nplurals=9; plural=0;",
            "nplurals=2; plural=(n != 1;",
            "nplurals=2; plural=n ? 1;",
            "nplurals=2; plural=n $ 1;",
            "nplurals=2; plural=n 1;",
        ] {
            assert!(PluralRule::from_header(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn a_broken_catalog_is_an_error() {
        let german = by_tag("de").unwrap();
        assert!(Catalog::parse(german, b"").is_err());
        assert!(Catalog::parse(german, b"\xff\xfe").is_err());
        let bad_rule = "msgid \"\"\nmsgstr \"Plural-Forms: nplurals=2; plural=n +;\\n\"\n";
        assert!(Catalog::parse(german, bad_rule.as_bytes()).is_err());
        // Windows line ends and a byte order mark are fine.
        let crlf = "\u{feff}msgid \"Close\"\r\nmsgstr \"Schließen\"\r\n";
        let catalog = Catalog::parse(german, crlf.as_bytes()).unwrap();
        assert_eq!(catalog.translate(None, "Close"), Some("Schließen"));
        // Without a header, the language's plural rule applies.
        let polish = by_tag("pl").unwrap();
        let headerless = "msgid \"{n} file\"\nmsgid_plural \"{n} files\"\n\
                          msgstr[0] \"plik\"\nmsgstr[1] \"pliki\"\nmsgstr[2] \"plików\"\n";
        let catalog = Catalog::parse(polish, headerless.as_bytes()).unwrap();
        assert_eq!(catalog.ntranslate(None, "{n} file", 5), Some("plików"));
    }

    #[test]
    fn a_malformed_catalog_is_an_error_not_a_panic() {
        let german = by_tag("de").unwrap();
        // Each of these makes rspolib 0.1.2 panic.
        for broken in [
            "msgstr \"x\"\n",
            "\"\n",
            "# comment\n\"continued\"\n",
            "msgid_plural \"x\"\n",
            "msgid \"a\"\nmsgstr \"b\"\nmsgstr \"c\"\n",
        ] {
            assert!(
                Catalog::parse(german, format!("msgid \"\"\n{broken}").as_bytes()).is_err()
                    || Catalog::parse(german, broken.as_bytes()).is_err(),
                "accepted {broken:?}"
            );
            let _ = check_catalog(german, broken, None);
        }
        assert!(!reading_catalog());
    }

    #[test]
    fn translations_may_reorder_anonymous_placeholders() {
        let japanese = by_tag("ja").unwrap();
        let po = empty_catalog(japanese, "t")
            + "\nmsgid \"{} of {}\"\nmsgstr \"{1} 件中 {0} 件\"\n"
            + "\nmsgid \"{} slides\"\nmsgstr \"{} 枚\"\n";
        let report = check_catalog(japanese, &po, None);
        assert!(report.problems.is_empty(), "{report:?}");
        let wrong =
            empty_catalog(japanese, "t") + "\nmsgid \"{} of {}\"\nmsgstr \"{1} 件中 {2} 件\"\n";
        assert!(!check_catalog(japanese, &wrong, None).problems.is_empty());
        // The formatter fills them in that order.
        let args: &[(&str, &dyn fmt::Display)] = &[("", &12), ("", &340)];
        assert_eq!(format("{1} 件中 {0} 件", args), "340 件中 12 件");
    }

    #[test]
    fn catalogs_are_found_in_conventional_folders() {
        let dir = std::env::temp_dir().join(format!("eov-i18n-folders-{}", std::process::id()));
        let traditional = by_tag("zh-Hant").unwrap();
        let folder = dir.join("zh_TW").join("LC_MESSAGES");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("demo.po"), empty_catalog(traditional, "demo")).unwrap();
        assert_eq!(
            find_catalog(&dir, "demo", traditional),
            Some(folder.join("demo.po"))
        );
        assert!(find_catalog(&dir, "demo", by_tag("zh-Hans").unwrap()).is_none());
        assert_eq!(
            by_tag("nb").unwrap().catalog_folders(),
            ["nb", "nb_NO", "no"]
        );
        assert_eq!(
            by_tag("pt-BR").unwrap().catalog_folders(),
            ["pt-BR", "pt_BR"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn placeholders_are_found() {
        let found: Vec<&str> = placeholders("{} of {0}, {name} {{not}} {n}")
            .into_iter()
            .collect();
        assert_eq!(found, ["{0}", "{name}", "{n}", "{}"]);
        assert!(placeholders("no { placeholder").is_empty());
    }

    #[test]
    fn catalogs_are_checked_against_the_language_and_the_template() {
        let polish = by_tag("pl").unwrap();
        let template = "msgid \"\"\nmsgstr \"\"\n\"Content-Type: text/plain; charset=UTF-8\\n\"\n\
            \nmsgid \"Open {}\"\nmsgstr \"\"\n\
            \nmsgid \"One slide\"\nmsgid_plural \"{n} slides\"\nmsgstr[0] \"\"\nmsgstr[1] \"\"\n\
            \nmsgid \"Close\"\nmsgstr \"\"\n";
        let good = empty_catalog(polish, "t")
            + "\nmsgid \"Open {}\"\nmsgstr \"Otwórz {}\"\n"
            + "\nmsgid \"One slide\"\nmsgid_plural \"{n} slides\"\n\
               msgstr[0] \"Jeden preparat\"\nmsgstr[1] \"{n} preparaty\"\nmsgstr[2] \"{n} preparatów\"\n"
            + "\nmsgid \"Close\"\nmsgstr \"Zamknij\"\n";
        let report = check_catalog(polish, &good, Some(template));
        assert!(report.is_complete(), "{report:?}");
        assert_eq!(report.translated, 3);

        let bad = empty_catalog(polish, "t").replace("Language: pl", "Language: de")
            + "\nmsgid \"Open {}\"\nmsgstr \"Otwórz {x}\"\n"
            + "\nmsgid \"One slide\"\nmsgid_plural \"{n} slides\"\n\
               msgstr[0] \"Jeden preparat\"\nmsgstr[1] \"preparaty\"\n"
            + "\n#, fuzzy\nmsgid \"Gone\"\nmsgstr \"Nie ma\"\n";
        let report = check_catalog(polish, &bad, Some(template));
        assert!(!report.is_complete());
        assert_eq!(report.fuzzy, 1);
        assert_eq!(report.missing, ["Close"]);
        assert_eq!(report.obsolete, ["Gone"]);
        let problems = report.problems.join("\n");
        assert!(problems.contains("Language: de"), "{problems}");
        assert!(problems.contains("adds"), "{problems}");
        assert!(problems.contains("loses"), "{problems}");
        assert!(problems.contains("2 plural forms"), "{problems}");
        assert!(problems.contains("does not show {n}"), "{problems}");
    }

    #[test]
    fn catalogs_are_found_under_either_spelling() {
        let dir = std::env::temp_dir().join(format!("eov-i18n-test-{}", std::process::id()));
        let brazil = by_tag("pt-BR").unwrap();
        let posix = dir.join("pt_BR").join("LC_MESSAGES");
        std::fs::create_dir_all(&posix).unwrap();
        std::fs::write(
            posix.join("demo.po"),
            empty_catalog(brazil, "demo") + "\nmsgid \"Close\"\nmsgstr \"Fechar\"\n",
        )
        .unwrap();
        let catalog = Catalog::load(&dir, "demo", brazil).unwrap().unwrap();
        assert_eq!(catalog.translate(None, "Close"), Some("Fechar"));
        assert!(
            Catalog::load(&dir, "demo", by_tag("de").unwrap())
                .unwrap()
                .is_none()
        );
        assert!(Catalog::load(&dir, "demo", source()).unwrap().is_none());
        std::fs::write(posix.join("demo.po"), "msgid \"unterminated").unwrap();
        assert!(Catalog::load(&dir, "demo", brazil).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn isolation_applies_to_right_to_left_languages_only() {
        let arabic = by_tag("ar").unwrap();
        assert_eq!(
            isolate_for(arabic, "slide.svs"),
            "\u{2068}slide.svs\u{2069}"
        );
        assert_eq!(isolate_for(by_tag("de").unwrap(), "slide.svs"), "slide.svs");
    }

    #[test]
    fn decimals_follow_the_language() {
        let german = by_tag("de").unwrap();
        let persian = by_tag("fa").unwrap();
        assert_eq!(format_decimal_for(german, 1.5, 2), "1,50");
        assert_eq!(format_decimal_for(source(), 1.5, 2), "1.50");
        assert_eq!(format_decimal_for(persian, 0.25, 2), "0\u{066B}25");
        assert_eq!(format_decimal_for(german, -3.0, 0), "-3");
        for (text, value) in [
            ("1,5", Some(1.5)),
            ("1.5", Some(1.5)),
            (" 2 ", Some(2.0)),
            ("1.234,5", None),
            ("", None),
            ("abc", None),
            ("inf", None),
        ] {
            assert_eq!(parse_decimal_for(german, text), value, "{text:?}");
        }
        assert_eq!(parse_decimal_for(persian, "0\u{066B}25"), Some(0.25));
        assert_eq!(parse_decimal_for(source(), "0,25"), Some(0.25));
        // Persian and Arabic-Indic digits, as their keyboards type them,
        // and the Arabic decimal sign in any language.
        assert_eq!(parse_decimal_for(persian, "۰٫۲۵"), Some(0.25));
        assert_eq!(parse_decimal_for(by_tag("ar").unwrap(), "٠٫٢٥"), Some(0.25));
        assert_eq!(parse_decimal_for(persian, "0٫۲۵"), Some(0.25));
        assert_eq!(parse_decimal_for(persian, "۵۰"), Some(50.0));
        assert_eq!(parse_decimal_for(source(), "0\u{066B}5"), Some(0.5));
        assert_eq!(parse_decimal_for(source(), "\u{2212}2"), Some(-2.0));
        // The Arabic thousands separator is not a decimal point.
        assert_eq!(parse_decimal_for(persian, "1\u{066C}000"), None);
    }

    #[test]
    fn thousands_are_refused_rather_than_read_as_decimals() {
        let english = source();
        let german = by_tag("de").unwrap();
        let persian = by_tag("fa").unwrap();
        for text in [
            "1,600",
            "10,000",
            "-12,500",
            "+100,000",
            "\u{2212}1,250",
            "۱,۶۰۰",
        ] {
            assert_eq!(parse_decimal_for(english, text), None, "{text:?}");
        }
        assert_eq!(parse_decimal_for(persian, "1,600"), None);
        // What cannot be a thousands separator is still a decimal comma.
        for (text, value) in [
            ("0,25", 0.25),
            ("2,5", 2.5),
            ("0,125", 0.125),
            ("1234,5", 1234.5),
            ("1,60", 1.6),
            ("1,6000", 1.6),
        ] {
            assert_eq!(parse_decimal_for(english, text), Some(value), "{text:?}");
        }
        // Where the decimal separator is a comma, it is the point.
        assert_eq!(parse_decimal_for(german, "1,600"), Some(1.6));
        assert_eq!(parse_decimal_for(german, "1.600"), None);
        assert_eq!(parse_decimal_for(german, "0.125"), Some(0.125));
        assert_eq!(parse_decimal_for(german, "1.5"), Some(1.5));
        assert_eq!(parse_decimal_for(english, "1.600"), Some(1.6));
    }

    #[test]
    fn sentences_are_joined_as_their_script_writes_them() {
        assert_eq!(
            join_sentences(["One slide.", "Two layers."]),
            "One slide. Two layers."
        );
        assert_eq!(
            join_sentences(["已处理 1 张切片。", "跳过了 1 张切片（见 slides.csv）。"]),
            "已处理 1 张切片。跳过了 1 张切片（见 slides.csv）。"
        );
        assert_eq!(
            join_sentences(["スライドを処理しました！", "Check the file."]),
            "スライドを処理しました！Check the file."
        );
        // A sentence that fell back to English keeps its space.
        assert_eq!(
            join_sentences(["Check the file.", "已处理。"]),
            "Check the file. 已处理。"
        );
        assert_eq!(
            join_sentences(vec![String::new(), "Alone.".into(), " ".into()]),
            "Alone."
        );
        assert_eq!(join_sentences(Vec::<String>::new()), "");
    }

    /// Tests that change this library's language or catalog run one at a
    /// time.
    fn global_language() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn placeholders_are_filled_as_slint_does() {
        let file = "a.svs";
        let args: &[(&str, &dyn fmt::Display)] = &[("", &file), ("n", &3)];
        assert_eq!(format("{} has {n} levels", args), "a.svs has 3 levels");
        assert_eq!(format("{1}/{0}", args), "3/a.svs");
        assert_eq!(format("{{literal}} {}", args), "{literal} a.svs");
        assert_eq!(format("}} {", args), "} {");
        // A placeholder without an argument is shown, not dropped.
        assert_eq!(format("{missing} {5}", args), "{missing} {5}");
        assert_eq!(format("no placeholders", &[]), "no placeholders");
    }

    #[test]
    fn the_macro_translates_with_the_current_catalog() {
        let _guard = global_language();
        let german = by_tag("de").unwrap();
        let po = empty_catalog(german, "test")
            + "\nmsgid \"Open {}\"\nmsgstr \"{} öffnen\"\n"
            + "\nmsgctxt \"Menu\"\nmsgid \"Close\"\nmsgstr \"Schließen\"\n"
            + "\nmsgid \"One slide in {folder}\"\nmsgid_plural \"{n} slides in {folder}\"\n"
            + "msgstr[0] \"Ein Schnitt in {folder}\"\nmsgstr[1] \"{n} Schnitte in {folder}\"\n";
        set_catalog(german, Some(Catalog::parse(german, po.as_bytes()).unwrap()));
        assert_eq!(current().tag, "de");
        let file = "x.svs";
        assert_eq!(crate::tr!("Open {}", file), "x.svs öffnen");
        assert_eq!(crate::tr!("Menu" => "Close"), "Schließen");
        assert_eq!(crate::tr!("Close"), "Close");
        let folder = "/data";
        assert_eq!(
            crate::tr!(
                "One slide in {folder}" | "{n} slides in {folder}" % 1,
                folder
            ),
            "Ein Schnitt in /data"
        );
        assert_eq!(
            crate::tr!(
                "One slide in {folder}" | "{n} slides in {folder}" % 7_usize,
                folder = "/x"
            ),
            "7 Schnitte in /x"
        );
        // Untranslated texts are formatted in English.
        assert_eq!(crate::tr!("Ctx" => "{} of {}", 1, 2), "1 of 2");
        assert_eq!(crate::tr!("One item" | "{n} items" % 2), "2 items");
        // A catalog of another language than the one chosen is not used.
        set_catalog(
            by_tag("fr").unwrap(),
            Some(Catalog::parse(german, po.as_bytes()).unwrap()),
        );
        assert_eq!(crate::tr!("Menu" => "Close"), "Close");
        set_catalog(source(), None);
        assert_eq!(crate::tr!("Open {}", file), "Open x.svs");
        assert_eq!(current().tag, "en");
    }

    #[test]
    fn the_current_language_is_per_library() {
        let _guard = global_language();
        let before = current();
        set_current(by_tag("he").unwrap());
        assert!(is_rtl());
        assert_eq!(isolate("a.svs"), "\u{2068}a.svs\u{2069}");
        set_current(by_tag("de").unwrap());
        assert_eq!(format_decimal(2.25, 1), "2,2");
        assert_eq!(parse_decimal("2,5"), Some(2.5));
        set_current(source());
        assert!(!is_rtl());
        set_current(before);
    }

    /// Back to English, without a catalog, when a test ends (also when it
    /// fails).
    struct BackToEnglish;

    impl Drop for BackToEnglish {
        fn drop(&mut self) {
            set_catalog(source(), None);
        }
    }

    #[test]
    fn texts_are_written_in_english_inside_in_english() {
        let _guard = global_language();
        let _english = BackToEnglish;
        let persian = by_tag("fa").unwrap();
        let po = empty_catalog(persian, "test")
            + "\nmsgid \"Open {}\"\nmsgstr \"باز کردن {}\"\n"
            + "\nmsgctxt \"Menu\"\nmsgid \"Close\"\nmsgstr \"بستن\"\n"
            + "\nmsgid \"One slide\"\nmsgid_plural \"{n} slides\"\n"
            + "msgstr[0] \"{n} اسلاید\"\nmsgstr[1] \"{n} اسلاید\"\n";
        set_catalog(
            persian,
            Some(Catalog::parse(persian, po.as_bytes()).unwrap()),
        );
        let write = || crate::tr!("Open {}", isolate("a.svs"));
        assert_eq!(write(), "باز کردن \u{2068}a.svs\u{2069}");
        let english = in_english(|| {
            assert!(current().is_source());
            assert!(!is_rtl());
            assert_eq!(crate::tr!("Menu" => "Close"), "Close");
            assert_eq!(crate::tr!("One slide" | "{n} slides" % 1), "One slide");
            assert_eq!(crate::tr!("One slide" | "{n} slides" % 0), "0 slides");
            assert_eq!(format_decimal(2.5, 1), "2.5");
            // Nested calls: English until the outer one ends.
            assert_eq!(in_english(write), "Open a.svs");
            assert!(current().is_source());
            write()
        });
        assert_eq!(english, "Open a.svs");
        // The interface's language again, which did not change meanwhile.
        assert_eq!(current().tag, "fa");
        assert_eq!(write(), "باز کردن \u{2068}a.svs\u{2069}");
        assert_eq!(crate::tr!("One slide" | "{n} slides" % 1), "1 اسلاید");
        assert_eq!(format_decimal(2.5, 1), "2\u{066B}5");
        // Numbers a user typed are read as in English, too.
        set_current(by_tag("de").unwrap());
        assert_eq!(parse_decimal("1.600"), None);
        assert_eq!(in_english(|| parse_decimal("1.600")), Some(1.6));
        assert_eq!(
            in_english(|| join_sentences(["Eins.", "Zwei."])),
            "Eins. Zwei."
        );
    }

    #[test]
    fn english_ends_when_the_writing_panics() {
        let _guard = global_language();
        let _english = BackToEnglish;
        set_current(by_tag("he").unwrap());
        let result = std::panic::catch_unwind(|| {
            in_english(|| {
                assert!(!is_rtl());
                panic!("the writing failed")
            })
        });
        assert!(result.is_err());
        assert!(is_rtl());
        assert_eq!(isolate("a.svs"), "\u{2068}a.svs\u{2069}");
    }

    #[test]
    fn english_is_written_on_this_thread_only() {
        let _guard = global_language();
        let _english = BackToEnglish;
        set_current(by_tag("he").unwrap());
        in_english(|| {
            let other = std::thread::spawn(|| {
                // With `thread-local-language`, each thread has its own
                // language; without, this is the library's (already he).
                set_current(by_tag("he").unwrap());
                (is_rtl(), isolate("x"))
            })
            .join()
            .unwrap();
            assert_eq!(other, (true, "\u{2068}x\u{2069}".to_string()));
            assert!(!is_rtl());
        });
        assert!(is_rtl());
    }

    #[test]
    fn shown_and_english_texts_are_written_by_the_same_code() {
        let _guard = global_language();
        let _english = BackToEnglish;
        let german = by_tag("de").unwrap();
        let po = empty_catalog(german, "test")
            + "\nmsgid \"{} of {}\"\nmsgstr \"{} von {}\"\n"
            + "\nmsgid \"{n} tile could not be read\"\nmsgid_plural \"{n} tiles could not be read\"\n"
            + "msgstr[0] \"{n} Kachel war unlesbar\"\nmsgstr[1] \"{n} Kacheln waren unlesbar\"\n";
        set_catalog(german, Some(Catalog::parse(german, po.as_bytes()).unwrap()));
        let (shown, english) =
            shown_and_english(|| crate::tr!("{} of {}", format_decimal(1.5, 1), 3));
        assert_eq!(shown, "1,5 von 3");
        assert_eq!(english, "1.5 of 3");
        let failed = 2;
        assert_eq!(
            shown_and_english(|| crate::tr!(
                "{n} tile could not be read" | "{n} tiles could not be read" % failed
            )),
            (
                "2 Kacheln waren unlesbar".to_string(),
                "2 tiles could not be read".to_string()
            )
        );
        // Only the text is made in English: the language stays.
        assert_eq!(current().tag, "de");
        // In English, both are the same.
        set_catalog(source(), None);
        assert_eq!(
            shown_and_english(|| crate::tr!("{} of {}", 1, 2)),
            ("1 of 2".to_string(), "1 of 2".to_string())
        );
    }
}
