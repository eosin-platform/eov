# Translating eov

eov's interface is available in 31 languages: English, the language it is
written in, and 30 translations. This guide is for everyone who wants to
improve a translation, add a text to eov or a plugin, add a language, or
translate a plugin of their own.

**Every translation can be improved.** The translations other than English
were machine-translated in November 2026, then reviewed by independent AI
reviewers, with a back-translation check of the safety-related texts. None
has yet been reviewed by a native speaker, let alone by a pathologist who is
one. If a text is unclear, inaccurate, unnatural or just not what you would
expect from good software in your language, please
[open an issue](https://github.com/eosin-platform/eov/issues/new?title=Translation%3A%20) or
send a pull request. No wording is final, and corrections from native
speakers and domain experts are exactly what the translations need.

## Languages

| Language | Tag | Review status |
|---|---|---|
| English | `en` | Source language |
| العربية (Arabic) | `ar` | Machine-translated; not yet reviewed by a native speaker |
| Čeština (Czech) | `cs` | Machine-translated; not yet reviewed by a native speaker |
| Dansk (Danish) | `da` | Machine-translated; not yet reviewed by a native speaker |
| Deutsch (German) | `de` | Machine-translated; not yet reviewed by a native speaker |
| Ελληνικά (Greek) | `el` | Machine-translated; not yet reviewed by a native speaker |
| Español (Spanish) | `es` | Machine-translated; not yet reviewed by a native speaker |
| فارسی (Persian) | `fa` | Machine-translated; not yet reviewed by a native speaker |
| Suomi (Finnish) | `fi` | Machine-translated; not yet reviewed by a native speaker |
| Français (French) | `fr` | Machine-translated; not yet reviewed by a native speaker |
| עברית (Hebrew) | `he` | Machine-translated; not yet reviewed by a native speaker |
| हिन्दी (Hindi) | `hi` | Machine-translated; not yet reviewed by a native speaker |
| Magyar (Hungarian) | `hu` | Machine-translated; not yet reviewed by a native speaker |
| Bahasa Indonesia (Indonesian) | `id` | Machine-translated; not yet reviewed by a native speaker |
| Italiano (Italian) | `it` | Machine-translated; not yet reviewed by a native speaker |
| 日本語 (Japanese) | `ja` | Machine-translated; not yet reviewed by a native speaker |
| 한국어 (Korean) | `ko` | Machine-translated; not yet reviewed by a native speaker |
| Norsk bokmål (Norwegian Bokmål) | `nb` | Machine-translated; not yet reviewed by a native speaker |
| Nederlands (Dutch) | `nl` | Machine-translated; not yet reviewed by a native speaker |
| Polski (Polish) | `pl` | Machine-translated; not yet reviewed by a native speaker |
| Português (Brasil) (Portuguese, Brazil) | `pt-BR` | Machine-translated; not yet reviewed by a native speaker |
| Română (Romanian) | `ro` | Machine-translated; not yet reviewed by a native speaker |
| Русский (Russian) | `ru` | Machine-translated; not yet reviewed by a native speaker |
| Slovenščina (Slovenian) | `sl` | Machine-translated; not yet reviewed by a native speaker |
| Svenska (Swedish) | `sv` | Machine-translated; not yet reviewed by a native speaker |
| ไทย (Thai) | `th` | Machine-translated; not yet reviewed by a native speaker |
| Türkçe (Turkish) | `tr` | Machine-translated; not yet reviewed by a native speaker |
| Українська (Ukrainian) | `uk` | Machine-translated; not yet reviewed by a native speaker |
| Tiếng Việt (Vietnamese) | `vi` | Machine-translated; not yet reviewed by a native speaker |
| 简体中文 (Chinese, Simplified) | `zh-Hans` | Machine-translated; not yet reviewed by a native speaker |
| 繁體中文 (Chinese, Traditional) | `zh-Hant` | Machine-translated; not yet reviewed by a native speaker |

Each catalog records its status in its header (`X-Review-Status`). When a
native speaker has reviewed a catalog, change it to say so (for example
`reviewed by @name, 2027-01`) and update this table; with the first such
review, also update the review status at the top of this guide and in the
README.

## How eov chooses the language

1. `eov --language <tag>` (for example `--language ja`, `--language pt-BR`,
   `--language zh-Hant`, `--language en`), for that run only;
2. the language chosen in the window (the globe button in the toolbar, or
   the language link at the bottom of the Home screen), kept as
   `language = "<tag>"` in `config.toml`;
3. the system's preferred languages (the user's language list on macOS and
   Windows; `LANGUAGE`, `LC_ALL`, `LC_MESSAGES` and `LANG` elsewhere), the
   first one eov has;
4. English.

Regional variants use the language's translation (`de-AT` → German, `es-MX`
→ Spanish), except where the region writes differently: Traditional Chinese
for Taiwan, Hong Kong and Macao, Simplified Chinese for mainland China,
Singapore and Malaysia, and the Brazilian translation only for Brazil (and
`pt` alone); Portuguese from Portugal or Africa falls back to the next
language of the system, or English, and can still be chosen by hand.

A language chosen in the window applies at once, to eov and to the plugins.

## Where the translations are

```text
app/translations/eov.pot                      eov's template (generated)
app/translations/<tag>/LC_MESSAGES/eov.po     eov in one language
plugins/<plugin>/translations/<id>.pot         a plugin's template
plugins/<plugin>/translations/<tag>/LC_MESSAGES/<id>.po
plugin_api/example-plugin/translations/…      the example plugin
docs/translation-glossary.md                  terms and how to treat them
docs/glossary/<tag>.tsv                       each language's terms
```

They are gettext catalogs (`.po` files): open them with any PO editor
(Poedit, Lokalize, Gtranslator, Weblate) or a text editor. eov compiles its
own catalogs into the program; plugins carry theirs in their packages.

## Correcting a translation

1. Find the text in `app/translations/<tag>/LC_MESSAGES/eov.po` (or the
   plugin's catalog). `msgctxt` is the text's context: for texts of the
   window it is the component it appears in (`LanguageDialog`), for others a
   word that tells same-looking texts apart (`Menu`, `Zoom`). `#.` lines are
   notes for translators; `#:` lines say where the text is used.
2. Change the `msgstr`. Keep every placeholder (`{}`, `{0}`, `{name}`, `{n}`).
3. Check it: `python3 tools/i18n/i18n.py check` (needs GNU gettext and
   polib) and `cargo test -p eov i18n` (checks placeholders, plural forms and
   completeness).
4. If you changed a text whose `msgctxt` is `Desktop entry`, `File type` or
   `AppStream` (application menus, file managers and software centres on
   Linux show them), write the Linux files again with
   `EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types`, and include the
   changed files in `assets/linux` in your pull request. If you cannot build
   eov, say so in the pull request, and a maintainer will regenerate them.
5. Build eov (`cargo run -- --language <tag>`) to see it.
6. Send a pull request. Say what was wrong if it is not obvious; if you are
   a native speaker or work in pathology, say so — it helps reviewers.

If you would rather not edit files, open an issue with the language, the
text as eov shows it, where you saw it, and the wording you suggest.

## Adding or changing texts (developers)

- Texts in `.slint` files: `@tr("Close")`, `@tr("{} of {}", a, b)`,
  `@tr("{n} slide" | "{n} slides" % count)`, `@tr("Zoom" => "Fit")` for a
  context.
- Texts made in Rust: `tr!` from `eov_plugin_api::i18n` with the same
  syntax: `tr!("Opened {}", name)`, `tr!("Menu" => "Close")`,
  `tr!("One slide" | "{n} slides" % count)`. A `// Translators: …` comment
  right above goes into the catalog.
- One whole sentence per text, with placeholders; never put translated
  pieces together. Insert file names and paths with `i18n::isolate(…)`
  (`Locale.isolate(…)` in Slint) and decimal numbers with
  `i18n::format_decimal`. A message made of several sentences, each
  translated on its own, is put together with `i18n::join_sentences`, not
  with `join(" ")`: Chinese and Japanese put no space after `。`.
- A text used both with and without a plural (`tr!("Delete annotation")`
  and `tr!("Delete annotation" | "Delete {n} annotations" % n)`) becomes one
  catalog entry, the plural one, so the text without a count would show a
  plural form with a raw `{n}`: give one of them a context
  (`tr!("Menu" => "Delete annotation")`).
- Do not translate logs, command-line output, JSON, identifiers, or data
  from slides and files. In a plugin, `host.warn(…)` and `host.error(…)`
  go into eov's log and to the user alike (see below): tell the user with
  `host.notify(…)` and `tr!`, and log in English.
- Fields that take numbers accept Western, Arabic-Indic and Persian digits:
  in Slint, a plain text input read with `Locale.number(…)` and
  `Locale.is-number(…)` (not `input-type: number`, which lets only Western
  digits through); in Rust, `i18n::parse_decimal`.

You do not need to edit any catalog. Run `python3 tools/i18n/i18n.py update`
to regenerate the templates and merge every catalog (new texts arrive
untranslated, changed ones as fuzzy); until they are translated eov shows them
in English. CI (`check --strict`) fails when a template is out of date with
the sources, and the eov tests fail while an in-tree catalog is incomplete or
broken, so new texts get translated before a release.

The Linux desktop files carry the translations of eov's catalogs too: the
desktop entry (`assets/linux/io.eosin.eov.desktop`, for application menus),
the MIME package (`assets/linux/io.eosin.eov.xml`, the names file managers
give the slide formats) and the AppStream metadata
(`assets/linux/io.eosin.eov.metainfo.xml`, the summary and description
software centres show). Their texts are declared for the translators in
`app/src/file_types/mod.rs`, and a test checks that each one is; the
metadata's English is edited by hand, its translations never are. The files
are shipped as they are, so after the catalogs change, write them again with
`EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types` (`cargo test -p eov
file_types` fails while they are out of date).

The tools: `cargo install slint-tr-extractor --version 1.19.0 --locked`,
`cargo install xtr --locked`, GNU gettext (`apt install gettext`,
`brew install gettext`), and polib (`apt install python3-polib` or
`pip install polib`).

## Adding a language

1. Add it to `LANGUAGES` in `plugin_api/src/i18n.rs`: its BCP 47 tag, its
   name in itself and in English, its writing direction, its CLDR plural
   rule in gettext syntax, and its decimal separator. Keep the list sorted
   by tag (Slint bundles languages in that order).
2. `EOV_UPDATE_LANGUAGES=1 cargo test -p eov languages_json` writes
   `tools/i18n/languages.json` for the tools.
3. `python3 tools/i18n/i18n.py update` creates its catalogs for eov and the
   in-tree plugins.
4. Translate them (`export` writes what is left as JSON with an id per
   text, and `import` writes translations back, refusing any whose
   placeholders or plural forms do not fit, or whose text has changed
   since the export), add a
   `docs/glossary/<tag>.tsv`, add it to the table above and to
   `language_name` in `app/src/i18n.rs`.
5. `cargo test -p eov i18n` and look at the window with `--language <tag>`,
   including the Language dialog and a plugin panel. If the language's
   users have locales not named after its tag, as Chinese ones are, list
   them in `desktop_locales` in `app/src/file_types/mod.rs`. Then write the
   Linux files again with
   `EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types` (the new catalog
   translates their `Desktop entry`, `File type` and `AppStream` texts
   too), and include the changed files in `assets/linux` in the pull
   request.

## Plugins

A plugin opts into translations with an `[i18n]` table in its `plugin.toml`:

```toml
[i18n]
catalogs = "translations"   # default; <catalogs>/<tag>/LC_MESSAGES/<domain>.po
domain = "my_plugin"        # default: the plugin id
```

- **Surfaces.** eov compiles the plugin's `.slint` files with its catalogs:
  mark texts with `@tr` as above (the default context is the component's
  name, as `slint-tr-extractor` assumes). A surface may declare
  `in property <string> eov-language;` (the tag) and
  `in property <bool> eov-rtl;` (Arabic, Persian, Hebrew); eov sets them and
  keeps them current. Align start-aligned text with
  `eov-rtl ? TextHorizontalAlignment.right : TextHorizontalAlignment.left`.
- **Rust.** `use eov_plugin_api::prelude::*;` brings `tr!`. The SDK loads the
  plugin's catalog for the language eov tells it (`PluginInit::language`)
  before `Plugin::create`, and again before it delivers
  `HostEvent::LanguageChanged { language }`. On that event, hand eov again
  the texts it shows for you (toolbar tooltips, tool names, surface
  properties holding text).
- **Warnings and errors.** eov writes a plugin's `host.warn(…)` and
  `host.error(…)` messages to its log as they are, and shows the same text
  to the user as a notification. The log stays in English, so tell the
  user with `host.notify(…)` or `host.notify_error(…)` and `tr!`, and log
  in English with `host.info(…)` or `host.debug(…)`. For a text that is
  both shown and logged:

  ```rust
  let (shown, english) = i18n::shown_and_english(|| tr!("{} failed", i18n::isolate(&name)));
  host.info(english);
  host.notify(NotifyLevel::Warning, shown);
  ```

  `shown_and_english` runs the closure twice, the second time in English,
  so the closure should only make the text; values made inside it (with
  `tr!`, `isolate`, `format_decimal`) follow the language, while a text
  made earlier is inserted as it is, so keep such a text in both forms.
  `i18n::in_english(|| …)` runs any code as in English: on this thread
  only, for Rust code only (Slint's `@tr` keeps the interface's language),
  until the code returns or panics; calls can be nested. A plugin that
  hands translated text to `warn` or `error` accepts that eov's log gets
  that language. Keep the errors a plugin returns from `Plugin::create`,
  `Plugin::open_slide` and its `SlideReader` in English too: eov logs or
  records them as they are, and shows a slide's error inside its own
  translated message.
- **Labels on the slide.** eov burns overlay labels into exported images
  and copied views with a font that cannot draw every letter (it has most
  Latin letters and the Greek alphabet, but not, for example, Romanian ș
  and ț, Vietnamese letters with two diacritics or Greek letters with an
  accent), and the images travel beyond the interface's language. Give a
  translated label its English form too:
  `OverlayObject::with_export_label(…)`, written from the English text with
  `i18n::format_decimal_for(i18n::source(), …)` and without `isolate`. A
  label without one is burned in as shown.
- **Name and description.** The Plugins panel shows `name` and
  `description` from the catalog entries with the context `plugin.toml`;
  the extraction below adds them (and the slide format names, for when eov
  shows those).
- **Packaging.** Put `<catalogs>/<tag>/LC_MESSAGES/<domain>.po` into the
  `.eop` (the template is not needed). Languages without a catalog, and texts
  a catalog lacks, show in English. A catalog's plural rule must be its
  language's (as `i18n::LANGUAGES` lists it); any spelling that gives every
  number the same form will do, and a catalog without a `Plural-Forms`
  header takes the language's. With another rule, eov leaves the catalog out
  of the plugin's surfaces, which then show in English in that language. A
  text with fewer plural forms than its language has shows in English (in
  surfaces always, in Rust for the numbers whose form is missing).
- **Tools.** From an eov checkout:
  `python3 tools/i18n/i18n.py update --plugin path/to/plugin` (extract and
  merge), `check --plugin …`, `stats --plugin …`. It extracts the `@tr`
  texts of every `.slint` file in the plugin's folder (not in `target/`,
  hidden folders, or the `tests/`, `benches/` and `examples/` folders of
  its crates), the `tr!` texts of the module tree of the plugin's crate
  (from `[lib] path` in its `Cargo.toml`, else `src/lib.rs`) and the
  manifest's texts, and warns about Rust files that call `tr!` outside that
  tree (the texts of a plugin's other crates are not extracted). In tests,
  `eov_plugin_api::i18n::check_catalog` checks a catalog the way eov's own
  tests do. `tr!`'s language belongs to the whole plugin library, and
  creating a test harness sets it (`testing::test_init` gives English):
  run tests that switch languages in a test binary of their own, one at a
  time, or use the plugin API's `thread-local-language` feature in tests
  (see the `testing` module).
- **Compatibility.** The translation API arrived with plugin API 1.1. An eov
  built with plugin API 1.0 ignores `[i18n]`, sends no language
  (`PluginInit::language` is `None`, so `tr!` gives English) and no
  `LanguageChanged`: translated plugins keep working there, in English.
  Plugins built against 1.0 keep working in newer eov, in English; eov still
  sends them `DisplayAdjustments::badge` in English, as 1.0 did, and the
  English `reason` of `HostEvent::RegistrationRefused`. 1.1 plugins show
  its new `shown_reason` (the reason in the interface's language) when it
  is there, else `isolate(reason)`, and register a refused contribution
  again after `LanguageChanged` to get the reason in the new language. In
  source, 1.1 adds public fields to `PluginInit`, `DisplayAdjustments`,
  `PluginManifest`, `OverlayObject` and `HostEvent::RegistrationRefused`:
  code that builds them as struct literals needs `..Default::default()`
  (or `..testing::test_init(..)`, `..OverlayObject::new(..)`), and code
  that matches event variants needs `..`. `i18n::LANGUAGES` is a slice
  (iterate `i18n::LANGUAGES`, not `&LANGUAGES`) and `Language` is
  `#[non_exhaustive]`: minor releases may add languages and fields, so read
  a `Language`'s fields rather than build or destructure one.

## Checks

| Check | What it catches |
|---|---|
| `python3 tools/i18n/i18n.py check --strict` (CI) | templates out of date with the sources, catalogs not merged, invalid catalogs (`msgfmt --check`), wrong `Language`/`Plural-Forms` headers; it warns about a plugin's Rust files that call `tr!` outside its crate's module tree |
| `cargo test -p eov i18n` | incomplete eov and in-tree plugin catalogs, lost or added placeholders, wrong number of plural forms, plural forms without `{n}`, languages missing from the build, the Slint bundle order, locale negotiation, `--language` and config precedence |
| `cargo test -p eov-plugin-api i18n` | plural rules of every language (CLDR), catalog loading, fallback to English, formatting |
| `cargo test -p eov file_types` | the Linux desktop entry, MIME package and AppStream metadata out of date with the catalogs, control characters in their translations |

## Right-to-left languages

Arabic, Persian and Hebrew are written from right to left. Slint shapes and
orders their text by itself; eov aligns text at the start of the line (the
right edge) and isolates inserted file names, paths and numbers with units
so that they keep their order. (Isolation is applied in right-to-left
interfaces only: a file name in Arabic or Hebrew shown in a left-to-right
interface can still move the punctuation next to it.) Every field that takes
a number accepts Western, Arabic-Indic and Persian digits, and `.`, `,` or
the language's decimal separator. The slide, the minimap, scale bars and other
things whose direction is physical are not mirrored, nor are layouts:
Slint 1.19 cannot mirror layouts, so panels keep their left-to-right
arrangement. Translators: write numbers, units and Latin-script names as
you would in a scientific text; eov keeps them in order.

## Layout checks

`python3 tools/i18n/i18n.py pseudo [components] <dir>` writes pseudo-translations
(accented, 40 % longer, bracketed) of every template. Build eov with them to
find cut-off texts and texts that are not translated:

```bash
python3 tools/i18n/i18n.py pseudo eov target/pseudo
EOV_TRANSLATIONS_DIR=$PWD/target/pseudo/eov cargo run -- --language de
```
