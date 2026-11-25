//! Plugins' translations, as eov uses them (see the plugin API's `i18n` and
//! the manifest's `[i18n]` table): their surfaces are compiled with their
//! catalogs, and the Plugins panel shows their names and descriptions in the
//! language of the interface. The plugins translate their own Rust texts.
//!
//! Slint keeps one language index for the whole window, its own and the
//! plugins' surfaces alike, in the order of eov's bundled languages. A
//! plugin's surfaces are therefore compiled from a bundle with exactly
//! eov's languages, in eov's order: the plugin's catalog where it has one,
//! an empty catalog (English texts) where it has none. Languages a plugin
//! has but eov has not are left out.

use eov_plugin_api::PluginManifest;
use eov_plugin_api::i18n::{self, Catalog, PluralRule};
use eov_plugin_api::manifest::{MANIFEST_FILENAME, MANIFEST_TRANSLATION_CONTEXT};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::{Path, PathBuf};

/// The folder and domain of a plugin's catalogs (its manifest's `[i18n]`).
pub(crate) fn catalogs(plugin_root: &Path) -> Option<(PathBuf, String)> {
    let manifest = PluginManifest::from_file(&plugin_root.join(MANIFEST_FILENAME)).ok()?;
    manifest.catalogs(plugin_root)
}

/// The bundle to compile a plugin's surfaces with: a folder in eov's cache
/// holding `<tag>/LC_MESSAGES/<domain>.po` for each of eov's bundled
/// languages, and the domain. `None` when the plugin is not translated or
/// eov bundles no languages. Bundles are named by their content, so a
/// bundle is never changed while another eov reads it.
pub(crate) fn surface_bundle(plugin_root: &Path) -> Option<(PathBuf, String)> {
    let (dir, domain) = catalogs(plugin_root)?;
    let languages = crate::i18n::BUNDLED_LANGUAGES;
    if languages.is_empty() {
        return None;
    }
    let mut files = Vec::with_capacity(languages.len());
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    for tag in languages {
        let text = i18n::by_tag(tag)
            .and_then(|language| usable_catalog(&dir, &domain, language))
            .unwrap_or_else(|| {
                let language = i18n::by_tag(tag).unwrap_or(i18n::source());
                i18n::empty_catalog(language, &domain)
            });
        digest.update(tag.as_bytes());
        digest.update([0]);
        digest.update(text.as_bytes());
        digest.update([0]);
        files.push((tag, text));
    }
    let name: String = digest
        .finalize()
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let bundles = crate::paths::cache_dir().join("plugin-translations");
    let bundle = bundles.join(&name);
    if bundle.is_dir() {
        return Some((bundle, domain));
    }
    // Written beside it and renamed into place, so a half-written bundle is
    // never used.
    let staging = match tempfile::Builder::new().prefix(".staging-").tempdir_in({
        let _ = std::fs::create_dir_all(&bundles);
        &bundles
    }) {
        Ok(staging) => staging,
        Err(error) => {
            tracing::warn!(
                "Cannot prepare the translations of the plugin in {}: {error}",
                plugin_root.display()
            );
            return None;
        }
    };
    for (tag, text) in &files {
        let folder = staging.path().join(tag).join("LC_MESSAGES");
        let written = std::fs::create_dir_all(&folder)
            .and_then(|()| std::fs::write(folder.join(format!("{domain}.po")), text));
        if let Err(error) = written {
            tracing::warn!("Cannot prepare the plugin's {tag} translation: {error}");
            return None;
        }
    }
    let staging = staging.keep();
    if let Err(error) = std::fs::rename(&staging, &bundle) {
        // Another eov made the same bundle meanwhile.
        let _ = std::fs::remove_dir_all(&staging);
        if !bundle.is_dir() {
            tracing::warn!("Cannot store the plugin's translations: {error}");
            return None;
        }
    }
    Some((bundle, domain))
}

/// The text of a plugin's catalog for `language` as its surfaces get it, if
/// the plugin has a catalog eov can use there. Slint's compiler reads every
/// bundled catalog whenever it compiles a surface, and one it cannot use
/// would fail every surface of the plugin, in every language. So:
///
/// - a file that does not read as a catalog is left out, and so is a
///   catalog whose plural rule gives some number another form than the
///   language's rule does (its forms are in another order): that language
///   shows the plugin's surfaces in English, and the problem is logged;
/// - any spelling of the language's rule is fine (`n != 1`, msginit's
///   `(n==1) ? 0 : …`, or no `Plural-Forms` at all, which means the
///   language's rule): the copy in the bundle gets eov's spelling, which
///   Slint reads;
/// - a text with fewer plural forms than the language has is left
///   untranslated in the copy, so that it shows in English (Slint would
///   show its first form for the numbers that lack one; the SDK shows
///   those in English too). Extra forms are never used, and stay.
///
/// A catalog that needs none of this goes into the bundle as it is. The
/// plugin's Rust texts are translated by the SDK, which reads the plugin's
/// files itself.
fn usable_catalog(dir: &Path, domain: &str, language: &'static i18n::Language) -> Option<String> {
    let path = i18n::find_catalog(dir, domain, language)?;
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text.trim_start_matches('\u{feff}').replace("\r\n", "\n"),
        Err(error) => {
            tracing::warn!("Cannot read {}: {error}", path.display());
            return None;
        }
    };
    match surface_catalog(language, domain, &text) {
        Ok(text) => Some(text),
        Err(reason) => {
            tracing::warn!(
                "The plugin's {} translation is not used in its surfaces ({}): {reason}",
                language.tag,
                path.display()
            );
            None
        }
    }
}

/// The copy of a catalog (its text, BOM and CRLF removed) that goes into a
/// surface bundle, as [`usable_catalog`] describes, or why there is none.
fn surface_catalog(
    language: &'static i18n::Language,
    domain: &str,
    text: &str,
) -> Result<String, String> {
    let wanted = PluralRule::from_header(language.plural_forms)?;
    // The plugin API reads it with rspolib, Slint's reader, and turns
    // rspolib's panics on malformed files into errors. Its rule is the
    // header's, or the language's when there is none.
    let catalog = Catalog::parse(language, text.as_bytes()).map_err(|error| error.message)?;
    if !same_plural_forms(|n| catalog.plural_form(n), |n| wanted.form(n)) {
        return Err(format!(
            "its plural rule gives some numbers other forms than {}'s rule ({})",
            language.english_name, language.plural_forms
        ));
    }
    let po = PoText::scan(text)?;
    let mut edits: Vec<(Range<usize>, Vec<String>)> = Vec::new();
    let mut prefix = None;
    let compact = |text: &str| text.split_whitespace().collect::<String>();
    match po.header() {
        Some(header) => {
            let forms = header_field(header, "Plural-Forms")?;
            if forms.as_deref().map(compact) != Some(compact(language.plural_forms)) {
                edits.push((
                    header.lines.clone(),
                    header_with_plural_forms(header, language.plural_forms)?,
                ));
            }
        }
        None => prefix = Some(i18n::empty_catalog(language, domain)),
    }
    for entry in po.short_plurals(language.plural_count()) {
        for form in entry.forms() {
            edits.push((form.lines.clone(), vec![format!("{} \"\"", form.keyword)]));
        }
    }
    if edits.is_empty() && prefix.is_none() {
        return Ok(text.to_string());
    }
    let copy = po.with_edits(prefix, edits);
    // What Slint gets must read as a catalog with eov's spelling of the rule.
    Catalog::parse(language, copy.as_bytes())
        .map_err(|error| format!("cannot write a copy for the surfaces: {}", error.message))?;
    let written = PoText::scan(&copy)?
        .header()
        .map(|header| header_field(header, "Plural-Forms"))
        .transpose()?
        .flatten();
    if written.as_deref() != Some(language.plural_forms) {
        return Err("cannot write a copy for the surfaces with the language's plural rule".into());
    }
    Ok(copy)
}

/// Whether two plural rules give the same form to every number gettext
/// rules tell apart (they look at the last two or three digits, and at
/// small numbers).
fn same_plural_forms(a: impl Fn(u64) -> usize, b: impl Fn(u64) -> usize) -> bool {
    (0..=10_000u64)
        .chain([
            100_000,
            1_000_000,
            1_000_001,
            1_000_002,
            1_000_005,
            1_000_011,
            1_000_021,
            1_000_100,
            10_000_000,
            u64::from(u32::MAX),
        ])
        .all(|n| a(n) == b(n))
}

/// A `.po` text read line by line, as far as the surface bundle needs it:
/// where each entry's keywords and texts are, its flags, and the header.
/// It reads texts rspolib has read before, and reads them as rspolib does.
struct PoText<'a> {
    lines: Vec<&'a str>,
    entries: Vec<PoEntry>,
}

#[derive(Default)]
struct PoEntry {
    fuzzy: bool,
    fields: Vec<PoField>,
}

struct PoField {
    /// `msgctxt`, `msgid`, `msgid_plural`, `msgstr` or `msgstr[N]`.
    keyword: String,
    /// The keyword's line and the lines that continue its text.
    lines: Range<usize>,
    /// The text, still escaped.
    escaped: String,
}

impl PoEntry {
    fn field(&self, keyword: &str) -> Option<&PoField> {
        self.fields.iter().find(|field| field.keyword == keyword)
    }

    /// The `msgstr[N]` fields.
    fn forms(&self) -> impl Iterator<Item = &PoField> {
        self.fields
            .iter()
            .filter(|field| field.keyword.starts_with("msgstr["))
    }
}

impl<'a> PoText<'a> {
    fn scan(text: &'a str) -> Result<Self, String> {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut entries = Vec::new();
        let mut current = PoEntry::default();
        // Flags apply to the entry whose keywords follow them.
        let mut fuzzy = false;
        let has_msgstr =
            |entry: &PoEntry| entry.fields.iter().any(|f| f.keyword.starts_with("msgstr"));
        for (index, line) in lines.iter().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                if has_msgstr(&current) {
                    entries.push(std::mem::take(&mut current));
                }
                if line.starts_with("#~") {
                    // An obsolete entry's own flags.
                    fuzzy = false;
                } else if let Some(flags) = line.strip_prefix("#,") {
                    fuzzy |= flags.split(',').any(|flag| flag.trim() == "fuzzy");
                }
                continue;
            }
            let unexpected = || format!("line {}: cannot read {line:?}", index + 1);
            if line.starts_with('"') {
                let field = current.fields.last_mut().ok_or_else(unexpected)?;
                field.escaped.push_str(quoted(line).ok_or_else(unexpected)?);
                field.lines.end = index + 1;
                continue;
            }
            let end = line
                .find(|c: char| c.is_whitespace() || c == '"')
                .unwrap_or(line.len());
            let keyword = &line[..end];
            let begins = match keyword {
                "msgctxt" => !current.fields.is_empty(),
                "msgid" => current.fields.iter().any(|f| f.keyword != "msgctxt"),
                "msgid_plural" | "msgstr" => false,
                _ if keyword.starts_with("msgstr[") && keyword.ends_with(']') => false,
                _ => return Err(unexpected()),
            };
            if begins {
                entries.push(std::mem::take(&mut current));
            }
            if current.fields.is_empty() {
                current.fuzzy = std::mem::take(&mut fuzzy);
            }
            current.fields.push(PoField {
                keyword: keyword.to_string(),
                lines: index..index + 1,
                escaped: quoted(line[end..].trim_start())
                    .ok_or_else(unexpected)?
                    .to_string(),
            });
        }
        if !current.fields.is_empty() {
            entries.push(current);
        }
        Ok(Self { lines, entries })
    }

    /// The `msgstr` of the header: the first entry with an empty `msgid`,
    /// as rspolib takes it.
    fn header(&self) -> Option<&PoField> {
        self.entries
            .iter()
            .find(|entry| entry.field("msgid").is_some_and(|f| f.escaped.is_empty()))
            .and_then(|entry| entry.field("msgstr"))
    }

    /// The translated texts with a number that have fewer forms than
    /// `count` (as rspolib reads forms: by their index, a missing one
    /// empty).
    fn short_plurals(&self, count: usize) -> impl Iterator<Item = &PoEntry> {
        self.entries.iter().filter(move |entry| {
            if entry.fuzzy || entry.field("msgid_plural").is_none() {
                return false;
            }
            let index = |field: &PoField| {
                field.keyword["msgstr[".len()..field.keyword.len() - 1]
                    .parse::<usize>()
                    .ok()
            };
            let len = entry.forms().filter_map(index).map(|i| i + 1).max();
            let filled: BTreeSet<usize> = entry
                .forms()
                .filter(|form| !form.escaped.is_empty())
                .filter_map(index)
                .collect();
            len.is_some_and(|len| len < count && (0..len).all(|i| filled.contains(&i)))
        })
    }

    /// The text with lines replaced (`edits` must not overlap) and `prefix`
    /// put before it.
    fn with_edits(
        &self,
        prefix: Option<String>,
        mut edits: Vec<(Range<usize>, Vec<String>)>,
    ) -> String {
        edits.sort_by_key(|(lines, _)| lines.start);
        let mut out: Vec<String> = prefix.into_iter().collect();
        let mut at = 0;
        for (lines, replacement) in edits {
            out.extend(
                self.lines[at..lines.start]
                    .iter()
                    .map(|line| line.to_string()),
            );
            out.extend(replacement);
            at = lines.end;
        }
        out.extend(self.lines[at..].iter().map(|line| line.to_string()));
        out.join("\n")
    }
}

/// The text between a line's quotes.
fn quoted(text: &str) -> Option<&str> {
    text.strip_prefix('"')?.strip_suffix('"')
}

/// A field of the header (its `msgstr`), as rspolib reads it: the text
/// after `Key: `, trimmed, and the values of repeated keys put together.
fn header_field(header: &PoField, key: &str) -> Result<Option<String>, String> {
    let text = unescape(&header.escaped)?;
    let values: Vec<&str> = text
        .split('\n')
        .filter_map(|line| line.split_once(": "))
        .filter(|(name, _)| *name == key)
        .map(|(_, value)| value.trim())
        .collect();
    Ok((!values.is_empty()).then(|| values.concat()))
}

/// The header's `msgstr` lines with `Plural-Forms: {plural_forms}` in place
/// of the catalog's own (or added).
fn header_with_plural_forms(header: &PoField, plural_forms: &str) -> Result<Vec<String>, String> {
    let text = unescape(&header.escaped)?;
    let mut fields = Vec::new();
    let mut replaced = false;
    for line in text.split('\n').filter(|line| !line.is_empty()) {
        if line
            .split_once(": ")
            .is_some_and(|(name, _)| name == "Plural-Forms")
        {
            if !std::mem::replace(&mut replaced, true) {
                fields.push(format!("Plural-Forms: {plural_forms}"));
            }
        } else {
            fields.push(line.to_string());
        }
    }
    if !replaced {
        fields.push(format!("Plural-Forms: {plural_forms}"));
    }
    Ok(std::iter::once("msgstr \"\"".to_string())
        .chain(
            fields
                .iter()
                .map(|field| format!("\"{}\\n\"", escape(field))),
        )
        .collect())
}

/// A text of a `.po` file without its escapes (those rspolib reads).
fn unescape(escaped: &str) -> Result<String, String> {
    let mut text = String::with_capacity(escaped.len());
    let mut chars = escaped.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            text.push(c);
            continue;
        }
        text.push(match chars.next() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('b') => '\u{8}',
            Some('v') => '\u{b}',
            Some('f') => '\u{c}',
            Some(c @ ('"' | '\\')) => c,
            other => return Err(format!("cannot read the escape \\{}", other.unwrap_or(' '))),
        });
    }
    Ok(text)
}

/// A text written for a `.po` file (the inverse of [`unescape`]).
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => escaped.push_str("\\n"),
            '\t' => escaped.push_str("\\t"),
            '\r' => escaped.push_str("\\r"),
            '\u{8}' => escaped.push_str("\\b"),
            '\u{b}' => escaped.push_str("\\v"),
            '\u{c}' => escaped.push_str("\\f"),
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            c => escaped.push(c),
        }
    }
    escaped
}

thread_local! {
    /// The catalogs of plugins' manifest texts, by plugin root and language.
    static MANIFEST_CATALOGS: RefCell<HashMap<(PathBuf, &'static str), Option<Catalog>>> =
        RefCell::new(HashMap::new());
}

/// A text of a plugin's manifest (its name, description, slide format
/// names) in the language of the interface: the plugin's catalog entry
/// with the context `plugin.toml`, or the text as it is.
pub(crate) fn manifest_text(plugin_root: &Path, text: &str) -> String {
    let language = i18n::current();
    if language.is_source() || text.is_empty() {
        return text.to_string();
    }
    MANIFEST_CATALOGS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let catalog = cache
            .entry((plugin_root.to_path_buf(), language.tag))
            .or_insert_with(|| {
                let (dir, domain) = catalogs(plugin_root)?;
                Catalog::load(&dir, &domain, language)
                    .map_err(|error| tracing::warn!("Plugin translations: {error}"))
                    .ok()
                    .flatten()
            });
        catalog
            .as_ref()
            .and_then(|catalog| catalog.translate(Some(MANIFEST_TRANSLATION_CONTEXT), text))
            .unwrap_or(text)
            .to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin_with_translations(dir: &Path) {
        std::fs::write(
            dir.join("plugin.toml"),
            "id = \"demo\"\nname = \"Demo\"\nversion = \"1.0.0\"\n[i18n]\n",
        )
        .unwrap();
        let german = i18n::by_tag("de").unwrap();
        let folder = dir.join("translations").join("de").join("LC_MESSAGES");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(
            folder.join("demo.po"),
            i18n::empty_catalog(german, "demo")
                + "\nmsgctxt \"plugin.toml\"\nmsgid \"Demo\"\nmsgstr \"Vorführung\"\n",
        )
        .unwrap();
    }

    #[test]
    fn surfaces_get_every_language_of_eov_in_its_order() {
        let _home = crate::paths::isolate_for_tests();
        let plugin = tempfile::tempdir().unwrap();
        plugin_with_translations(plugin.path());
        let (bundle, domain) = surface_bundle(plugin.path()).expect("a bundle");
        assert_eq!(domain, "demo");
        let mut folders: Vec<String> = std::fs::read_dir(&bundle)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        folders.sort();
        assert_eq!(folders, crate::i18n::BUNDLED_LANGUAGES);
        let german = std::fs::read_to_string(bundle.join("de/LC_MESSAGES/demo.po")).unwrap();
        assert!(german.contains("Vorführung"));
        // Languages the plugin lacks are empty, with their plural rule.
        let polish = std::fs::read_to_string(bundle.join("pl/LC_MESSAGES/demo.po")).unwrap();
        assert!(polish.contains(i18n::by_tag("pl").unwrap().plural_forms));
        assert!(!polish.contains("Vorführung"));
        // The same catalogs give the same bundle.
        assert_eq!(surface_bundle(plugin.path()).unwrap().0, bundle);
    }

    /// Writes the demo plugin's catalog for `tag`.
    fn write_catalog(plugin: &Path, tag: &str, text: &str) {
        let folder = plugin.join("translations").join(tag).join("LC_MESSAGES");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("demo.po"), text).unwrap();
    }

    /// The copy of `tag`'s catalog in a bundle.
    fn bundled(bundle: &Path, tag: &str) -> String {
        std::fs::read_to_string(bundle.join(tag).join("LC_MESSAGES/demo.po")).unwrap()
    }

    /// A header as other tools write it, with `plural_forms` split over
    /// quoted lines as given (none: no `Plural-Forms`).
    fn header(tag: &str, plural_forms: &[&str]) -> String {
        let mut text = format!(
            "msgid \"\"\nmsgstr \"\"\n\"Project-Id-Version: demo 1.0\\n\"\n\
             \"Last-Translator: A. \\\"Translator\\\" <a@example.org>\\n\"\n\
             \"Language: {tag}\\n\"\n\"MIME-Version: 1.0\\n\"\n\
             \"Content-Type: text/plain; charset=UTF-8\\n\"\n"
        );
        for line in plural_forms {
            text += &format!("\"{line}\"\n");
        }
        text + "\"X-Generator: a PO editor\\n\"\n"
    }

    /// The bundle's copy of `tag` read as eov's surfaces get it: with the
    /// language's plural forms.
    fn bundled_catalog(bundle: &Path, tag: &str) -> Catalog {
        let language = i18n::by_tag(tag).unwrap();
        let catalog = Catalog::parse(language, bundled(bundle, tag).as_bytes()).unwrap();
        let rule = PluralRule::from_header(language.plural_forms).unwrap();
        assert!(same_plural_forms(
            |n| catalog.plural_form(n),
            |n| rule.form(n)
        ));
        catalog
    }

    #[test]
    fn a_broken_plugin_catalog_is_left_out_of_the_bundle() {
        let _home = crate::paths::isolate_for_tests();
        let plugin = tempfile::tempdir().unwrap();
        plugin_with_translations(plugin.path());
        // A file that is not a catalog, one rspolib panics on, and one with
        // another plural rule (German's in a Polish catalog).
        write_catalog(plugin.path(), "fr", "msgstr \"cassé\"\n\"\n");
        write_catalog(
            plugin.path(),
            "it",
            "msgid \"Demo\"\nmsgstr \"Dimostrazione\"\nmsgstr \"rotto\"\n",
        );
        let german = i18n::empty_catalog(i18n::by_tag("de").unwrap(), "demo");
        write_catalog(
            plugin.path(),
            "pl",
            &(german.replace("Language: de", "Language: pl")
                + "\nmsgid \"Demo\"\nmsgstr \"Pokaz\"\n"),
        );
        let (bundle, _) = surface_bundle(plugin.path()).expect("a bundle");
        assert!(!bundled(&bundle, "fr").contains("cassé"));
        assert!(!bundled(&bundle, "it").contains("Dimostrazione"));
        assert!(!bundled(&bundle, "pl").contains("Pokaz"));
        assert!(bundled(&bundle, "pl").contains(i18n::by_tag("pl").unwrap().plural_forms));
        // The good one is kept.
        assert!(bundled(&bundle, "de").contains("Vorführung"));
    }

    /// Weblate, msginit and hand-written catalogs spell plural rules in
    /// their own ways, or leave the header out: the copies keep their
    /// translations, with eov's spelling, which Slint reads.
    #[test]
    fn other_spellings_of_the_plural_rule_are_kept_with_eovs() {
        let _home = crate::paths::isolate_for_tests();
        let plugin = tempfile::tempdir().unwrap();
        plugin_with_translations(plugin.path());
        let entry =
            |text: &str| format!("\nmsgctxt \"plugin.toml\"\nmsgid \"Demo\"\nmsgstr \"{text}\"\n");
        write_catalog(
            plugin.path(),
            "de",
            &(header("de", &["Plural-Forms: nplurals=2; plural=n != 1;\\n"])
                + &entry("Vorführung")),
        );
        write_catalog(
            plugin.path(),
            "fr",
            &(header("fr", &["Plural-Forms: nplurals=2; plural=n > 1;\\n"]) + &entry("Démo")),
        );
        // msginit's Czech rule, wrapped over two lines as msginit does.
        write_catalog(
            plugin.path(),
            "cs",
            &(header(
                "cs",
                &[
                    "Plural-Forms: nplurals=3; plural=(n==1) ? 0 : (n>=2 && n<=4) ? 1 : ",
                    "2;\\n",
                ],
            ) + &entry("Ukázka")),
        );
        // No Plural-Forms, and no header at all.
        write_catalog(
            plugin.path(),
            "ro",
            &(header("ro", &[]) + &entry("Demonstrație")),
        );
        write_catalog(plugin.path(), "pl", &entry("Pokaz"));
        let (bundle, _) = surface_bundle(plugin.path()).expect("a bundle");
        for (tag, text) in [
            ("de", "Vorführung"),
            ("fr", "Démo"),
            ("cs", "Ukázka"),
            ("ro", "Demonstrație"),
            ("pl", "Pokaz"),
        ] {
            let language = i18n::by_tag(tag).unwrap();
            let copy = bundled(&bundle, tag);
            assert!(
                copy.contains(&format!("\"Plural-Forms: {}\\n\"", language.plural_forms)),
                "{tag}:\n{copy}"
            );
            assert_eq!(copy.matches("Plural-Forms").count(), 1, "{tag}:\n{copy}");
            let catalog = bundled_catalog(&bundle, tag);
            assert_eq!(catalog.translate(Some("plugin.toml"), "Demo"), Some(text));
        }
        // The rest of the header stays as it was.
        let czech = bundled(&bundle, "cs");
        assert!(czech.contains("\"Last-Translator: A. \\\"Translator\\\" <a@example.org>\\n\""));
        assert!(czech.contains("\"X-Generator: a PO editor\\n\""));
    }

    /// A text with fewer plural forms than the language has shows in English
    /// in the surfaces (Slint would show its first form for the other
    /// numbers); the language's other texts stay translated, and so do texts
    /// with extra forms, which are never used.
    #[test]
    fn texts_lacking_plural_forms_are_left_untranslated_in_surfaces() {
        let _home = crate::paths::isolate_for_tests();
        let plugin = tempfile::tempdir().unwrap();
        plugin_with_translations(plugin.path());
        let polish = i18n::by_tag("pl").unwrap();
        write_catalog(
            plugin.path(),
            "pl",
            &(i18n::empty_catalog(polish, "demo")
                + "\nmsgid \"{n} file\"\nmsgid_plural \"{n} files\"\n\
                   msgstr[0] \"{n} plik\"\nmsgstr[1] \"{n} pliki\"\n\
                   \nmsgid \"{n} slide\"\nmsgid_plural \"{n} slides\"\n\
                   msgstr[0] \"{n} preparat\"\nmsgstr[1] \"{n} preparaty\"\n\
                   msgstr[2] \"{n} preparatów\"\n\
                   \n#, fuzzy\nmsgid \"{n} layer\"\nmsgid_plural \"{n} layers\"\n\
                   msgstr[0] \"{n} warstwa\"\n\
                   \nmsgid \"Hello\"\nmsgstr \"Cześć\"\n"),
        );
        let japanese = i18n::by_tag("ja").unwrap();
        let japanese_text = i18n::empty_catalog(japanese, "demo")
            + "\nmsgid \"{n} file\"\nmsgid_plural \"{n} files\"\n\
               msgstr[0] \"{n} 個のファイル\"\nmsgstr[1] \"{n} 個のファイル\"\n";
        write_catalog(plugin.path(), "ja", &japanese_text);
        let (bundle, _) = surface_bundle(plugin.path()).expect("a bundle");
        let catalog = bundled_catalog(&bundle, "pl");
        assert_eq!(catalog.ntranslate(None, "{n} file", 5), None);
        assert_eq!(catalog.ntranslate(None, "{n} file", 1), None);
        assert_eq!(
            catalog.ntranslate(None, "{n} slide", 5),
            Some("{n} preparatów")
        );
        assert_eq!(catalog.translate(None, "Hello"), Some("Cześć"));
        // The fuzzy text is not used anyway, and stays as it was.
        assert!(bundled(&bundle, "pl").contains("msgstr[0] \"{n} warstwa\""));
        // Extra forms: as it was.
        assert_eq!(bundled(&bundle, "ja"), japanese_text);
    }

    /// Catalogs eov can use as they are go into bundles unchanged, so
    /// bundles made before keep their names.
    #[test]
    fn usable_catalogs_are_bundled_as_they_are() {
        let german = i18n::by_tag("de").unwrap();
        let text = header("de", &["Plural-Forms: nplurals=2;  plural=(n != 1);\\n"])
            + "\n#~ msgid \"Old\"\n#~ msgstr \"Alt\"\n\
               \n#, fuzzy\nmsgid \"{n} file\"\nmsgid_plural \"{n} files\"\nmsgstr[0] \"{n} Datei\"\n\
               \nmsgctxt \"Panel\"\nmsgid \"\"\n\"Hello\"\nmsgstr \"\"\n\"Hallo\"\n";
        assert_eq!(surface_catalog(german, "demo", &text).unwrap(), text);
        let english_header = i18n::empty_catalog(german, "demo");
        assert_eq!(
            surface_catalog(german, "demo", &english_header).unwrap(),
            english_header
        );
        // So do the catalogs of eov's own plugins (`i18n.py` writes them).
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for (dir, domain) in [
            ("plugin_api/example-plugin/translations", "example_plugin"),
            ("plugins/annotations/translations", "annotations"),
            ("plugins/cellaris/translations", "cellaris"),
            ("plugins/eovae/translations", "eovae"),
            ("plugins/gamepad/translations", "gamepad"),
        ] {
            for language in i18n::translations() {
                let path = i18n::catalog_path(&root.join(dir), domain, language);
                // Submodules that are not checked out have none.
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                assert_eq!(
                    surface_catalog(language, domain, &text).as_ref(),
                    Ok(&text),
                    "{}",
                    path.display()
                );
                // Without its Plural-Forms, the copy gets the language's
                // and keeps every text as it was.
                let variant = text.replace("\"Plural-Forms:", "\"X-Former-Plural-Forms:");
                let copy = surface_catalog(language, domain, &variant).unwrap();
                let (before, after) = (
                    i18n::check_catalog(language, &text, None),
                    i18n::check_catalog(language, &copy, None),
                );
                assert_eq!(before, after, "{}", path.display());
                assert!(copy.contains(&format!("\"Plural-Forms: {}\\n\"", language.plural_forms)));
            }
        }
    }

    /// Slint reads the copies: the forms follow the language's rule
    /// whatever the catalog's spelling of it, and a text that lacks forms
    /// shows in English while the rest of the language stays translated.
    #[test]
    fn surfaces_show_the_forms_of_the_languages_rule() {
        crate::test_support::run_on_slint_ui_test_thread(|| {
            let ui = crate::AppWindow::new().unwrap();
            struct English<'a>(&'a crate::AppWindow);
            impl Drop for English<'_> {
                fn drop(&mut self) {
                    crate::i18n::switch_to(self.0, i18n::source());
                }
            }
            let _english = English(&ui);
            let plugin = tempfile::tempdir().unwrap();
            plugin_with_translations(plugin.path());
            write_catalog(
                plugin.path(),
                "cs",
                &(header(
                    "cs",
                    &[
                        "Plural-Forms: nplurals=3; plural=(n==1) ? 0 : (n>=2 && n<=4) ? 1 : ",
                        "2;\\n",
                    ],
                ) + "\nmsgctxt \"Panel\"\nmsgid \"{n} slide\"\nmsgid_plural \"{n} slides\"\n\
                     msgstr[0] \"{n} preparát\"\nmsgstr[1] \"{n} preparáty\"\n\
                     msgstr[2] \"{n} preparátů\"\n"),
            );
            // No header: Polish forms all the same, not `n != 1`.
            write_catalog(
                plugin.path(),
                "pl",
                "msgctxt \"Panel\"\nmsgid \"{n} slide\"\nmsgid_plural \"{n} slides\"\n\
                 msgstr[0] \"{n} preparat\"\nmsgstr[1] \"{n} preparaty\"\n\
                 msgstr[2] \"{n} preparatów\"\n\
                 \nmsgctxt \"Panel\"\nmsgid \"{n} file\"\nmsgid_plural \"{n} files\"\n\
                 msgstr[0] \"{n} plik\"\nmsgstr[1] \"{n} pliki\"\n\
                 \nmsgctxt \"Panel\"\nmsgid \"Hello\"\nmsgstr \"Cześć\"\n",
            );
            let (bundle, domain) = surface_bundle(plugin.path()).expect("a bundle");
            let mut compiler = slint_interpreter::Compiler::default();
            compiler.set_translation_domain(domain);
            compiler.set_bundled_translations_path(bundle);
            let result = spin_on(
                compiler.build_from_source(
                    "export component Panel inherits Rectangle {\n\
                     in property <int> count;\n\
                     out property <string> slides: @tr(\"{n} slide\" | \"{n} slides\" % count);\n\
                     out property <string> files: @tr(\"{n} file\" | \"{n} files\" % count);\n\
                     out property <string> greeting: @tr(\"Hello\");\n\
                 }"
                    .into(),
                    plugin.path().join("panel.slint"),
                ),
            );
            let errors: Vec<String> = result
                .diagnostics()
                .filter(|d| d.level() == slint_interpreter::DiagnosticLevel::Error)
                .map(|d| d.to_string())
                .collect();
            assert!(errors.is_empty(), "{errors:?}");
            let panel = result
                .component("Panel")
                .expect("compiles")
                .create()
                .unwrap();
            let text = |name: &str, count: i32| {
                panel
                    .set_property("count", slint_interpreter::Value::Number(count.into()))
                    .unwrap();
                match panel.get_property(name).unwrap() {
                    slint_interpreter::Value::String(text) => text.to_string(),
                    other => panic!("{other:?}"),
                }
            };
            crate::i18n::switch_to(&ui, i18n::by_tag("cs").unwrap());
            assert_eq!(text("slides", 1), "1 preparát");
            assert_eq!(text("slides", 3), "3 preparáty");
            assert_eq!(text("slides", 5), "5 preparátů");
            crate::i18n::switch_to(&ui, i18n::by_tag("pl").unwrap());
            assert_eq!(text("slides", 1), "1 preparat");
            assert_eq!(text("slides", 3), "3 preparaty");
            assert_eq!(text("slides", 5), "5 preparatów");
            assert_eq!(text("slides", 22), "22 preparaty");
            // Two forms where Polish has three: English, not "5 plik".
            assert_eq!(text("files", 5), "5 files");
            assert_eq!(text("files", 1), "1 file");
            assert_eq!(text("greeting", 1), "Cześć");
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

    #[test]
    fn plugins_without_translations_get_no_bundle() {
        let _home = crate::paths::isolate_for_tests();
        let plugin = tempfile::tempdir().unwrap();
        std::fs::write(
            plugin.path().join("plugin.toml"),
            "id = \"plain\"\nname = \"Plain\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        assert!(surface_bundle(plugin.path()).is_none());
        assert!(catalogs(plugin.path()).is_none());
    }
}
