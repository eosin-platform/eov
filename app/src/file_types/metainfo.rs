//! The translations in eov's AppStream metadata
//! (`assets/linux/io.eosin.eov.metainfo.xml`), which software centres
//! (GNOME Software, KDE Discover, Flathub) show. The file is maintained by
//! hand, in English. Its summary and the paragraphs and list items of its
//! description get the translations of eov's catalogs as siblings with an
//! `xml:lang`, right after the English element, where itstool and
//! `msgfmt --xml` put them. The tests check the shipped file;
//! `EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types` writes its
//! translations again (see docs/translating.md).
//!
//! The file is read as text, so that everything else in it stays exactly
//! as it is. That needs plain XML (no CDATA, no DTD), each translated
//! element alone on its lines, and plain text in it (with the predefined
//! entities such as `&amp;`).

use super::{Translate, desktop_locales, translations_of, xml_text};
use anyhow::{Context, Result, bail};
use eov_plugin_api::i18n::Language;

/// The context of the metadata's texts in eov's catalogs.
pub(super) const APPSTREAM_CONTEXT: &str = "AppStream";

/// The `xml:lang`s of a language: the locales of the desktop entry
/// (`desktop_locales`) as BCP 47 tags, with the country (`pt-BR`, `zh-HK`).
/// AppStream reads `xml:lang` as BCP 47 since 0.16.2: libappstream matches
/// a user's `pt_BR` locale with `pt-BR` (never with `pt_BR`, which msgfmt
/// writes), and else with the language alone (`zh`), and GNOME Software's
/// libxmlb turns `pt-BR` into the `pt_BR` it matches. Chinese goes by
/// country, as the users' locales do (`zh-Hant` would match no `zh_TW`
/// user).
fn appstream_locales(language: &Language) -> Vec<String> {
    desktop_locales(language)
        .into_iter()
        .map(|locale| locale.replace('_', "-"))
        .collect()
}

/// An element of the metadata that is translated.
struct Element<'a> {
    name: &'a str,
    /// A translation (it has an `xml:lang`), which is replaced.
    translation: bool,
    /// From the `<` of its opening tag to the `>` of its closing tag.
    start: usize,
    end: usize,
    /// Its text as the catalogs have it: references resolved, whitespace
    /// collapsed.
    text: String,
}

/// `metainfo` (with `\n` line ends) with the translations `translate`
/// gives in place of those it has.
pub(super) fn localized(metainfo: &str, translate: Translate) -> Result<String> {
    let elements = elements(metainfo)?;
    let paragraphs: Vec<&Element> = elements
        .iter()
        .filter(|element| !element.translation && element.name != "summary")
        .collect();
    let description = description_translations(&paragraphs, translate);
    let mut localized = String::with_capacity(metainfo.len());
    let mut copied = 0;
    let mut paragraph = 0;
    for element in &elements {
        // The element's lines, from the indentation of its opening tag to
        // the line end after its closing tag.
        let first = metainfo[..element.start].rfind('\n').map_or(0, |at| at + 1);
        let last = metainfo[element.end..]
            .find('\n')
            .map_or(metainfo.len(), |at| element.end + at + 1);
        let indent = &metainfo[first..element.start];
        if !indent.trim().is_empty() || !metainfo[element.end..last].trim().is_empty() {
            bail!("<{}> shares its lines with other markup", element.name);
        }
        if element.translation {
            // The catalogs' translations replace it.
            localized.push_str(&metainfo[copied..first]);
        } else {
            localized.push_str(&metainfo[copied..last]);
            let translations: Vec<(String, String)> = if element.name == "summary" {
                translations_of(translate, APPSTREAM_CONTEXT, &element.text)
                    .into_iter()
                    .flat_map(|(language, translation)| {
                        appstream_locales(language)
                            .into_iter()
                            .map(move |locale| (locale, translation.clone()))
                    })
                    .collect()
            } else {
                paragraph += 1;
                description
                    .iter()
                    .map(|(locale, texts)| (locale.clone(), texts[paragraph - 1].clone()))
                    .collect()
            };
            for (locale, translation) in translations {
                localized.push_str(&format!(
                    "{indent}<{name} xml:lang=\"{locale}\">{}</{name}>\n",
                    xml_text(&translation),
                    name = element.name
                ));
            }
        }
        copied = last;
    }
    localized.push_str(&metainfo[copied..]);
    Ok(localized)
}

/// The translations of the description's paragraphs and list items, by
/// locale (each of a language's), for the languages that translate every
/// one of them. A language's description is made of the elements in that
/// language alone: libappstream drops an incomplete one for the English
/// description, and other readers would show it with paragraphs missing. A
/// language that lacks one keeps the English description.
fn description_translations(
    description: &[&Element],
    translate: Translate,
) -> Vec<(String, Vec<String>)> {
    eov_plugin_api::i18n::translations()
        .iter()
        .filter_map(|language| {
            // Every text is looked up, also after a missing one (the tests
            // collect the texts looked up).
            let texts: Vec<Option<String>> = description
                .iter()
                .map(|element| translate(language, APPSTREAM_CONTEXT, &element.text))
                .collect();
            let texts: Vec<String> = texts.into_iter().collect::<Option<_>>()?;
            // A "translation" that is the English adds nothing.
            let english = description.iter().map(|element| &element.text);
            (!texts.iter().eq(english)).then_some((language, texts))
        })
        .flat_map(|(language, texts)| {
            appstream_locales(language)
                .into_iter()
                .map(move |locale| (locale, texts.clone()))
        })
        .collect()
}

/// Whether an element `name` within `parents` is translated: the summary,
/// and the paragraphs and list items of the description (not those of
/// release notes).
fn translated(parents: &[&str], name: &str) -> bool {
    matches!(
        (parents, name),
        (["component"], "summary")
            | (["component", "description"], "p")
            | (["component", "description", "ul" | "ol"], "li")
    )
}

/// The translated elements of `xml`, in order.
fn elements(xml: &str) -> Result<Vec<Element<'_>>> {
    // The open elements: name, where the opening tag starts and ends, and
    // whether it has an `xml:lang`.
    let mut open: Vec<(&str, usize, usize, bool)> = Vec::new();
    let mut elements = Vec::new();
    let mut at = 0;
    while let Some(offset) = xml[at..].find('<') {
        let start = at + offset;
        let markup = &xml[start..];
        // Comments and the XML declaration stay as they are.
        let close = if markup.starts_with("<!--") {
            "-->"
        } else if markup.starts_with("<?") {
            "?>"
        } else if markup.starts_with("<!") {
            bail!("unsupported markup (CDATA or a DTD) at byte {start}");
        } else {
            ">"
        };
        let end = start
            + markup
                .find(close)
                .with_context(|| format!("unclosed markup at byte {start}"))?
            + close.len();
        at = end;
        if close != ">" {
            continue;
        }
        let tag = &xml[start + 1..end - 1];
        if let Some(name) = tag.strip_prefix('/') {
            let name = name.trim();
            let (opened, opened_start, opened_end, translation) = open
                .pop()
                .with_context(|| format!("</{name}> closes no element"))?;
            if name != opened {
                bail!("</{name}> closes <{opened}>");
            }
            let parents: Vec<&str> = open.iter().map(|(parent, ..)| *parent).collect();
            if translated(&parents, name) {
                let content = &xml[opened_end..start];
                if content.contains('<') {
                    bail!("<{name}> has markup inside: only plain text is translated");
                }
                elements.push(Element {
                    name: opened,
                    translation,
                    start: opened_start,
                    end,
                    text: text(content)?,
                });
            }
        } else if !tag.ends_with('/') {
            let name = tag
                .split([' ', '\t', '\n', '\r'])
                .next()
                .unwrap_or_default();
            open.push((name, start, end, tag.contains("xml:lang")));
        }
    }
    if let Some((name, ..)) = open.last() {
        bail!("<{name}> is not closed");
    }
    Ok(elements)
}

/// The text of an element's `content` as the catalogs have it: XML
/// whitespace collapsed, the predefined entities resolved.
fn text(content: &str) -> Result<String> {
    let collapsed = content
        .split([' ', '\t', '\n', '\r'])
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let mut text = String::with_capacity(collapsed.len());
    let mut rest = collapsed.as_str();
    while let Some(ampersand) = rest.find('&') {
        text.push_str(&rest[..ampersand]);
        let reference = &rest[ampersand + 1..];
        let semicolon = reference
            .find(';')
            .with_context(|| format!("a lone & in \"{collapsed}\": write &amp;"))?;
        text.push(match &reference[..semicolon] {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            other => bail!("&{other}; is not supported: write the character itself"),
        });
        rest = &reference[semicolon + 1..];
    }
    text.push_str(rest);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::super::translations_in;
    use super::*;
    use eov_plugin_api::i18n::{self, Catalog};

    const ENGLISH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!-- A <comment> -->
<component type="desktop-application">
  <id>org.example.Viewer</id>
  <name>Viewer</name>
  <summary>View slides</summary>
  <summary xml:lang="de">Alt</summary>
  <description>
    <p>
      Viewer shows
      H&amp;E slides.
    </p>
    <p xml:lang="fr">Ancien</p>
    <ul>
      <li>Fast</li>
    </ul>
  </description>
  <releases>
    <release version="1.0.0" date="2026-11-23">
      <description>
        <p>First release</p>
      </description>
    </release>
  </releases>
</component>
"#;

    /// Languages with a country get BCP 47 tags (`pt-BR`, `zh-TW`), which
    /// libappstream matches with the users' locales (`pt_BR`, `zh_TW`); a
    /// translation goes to every locale whose users read it in eov
    /// (Traditional Chinese to Hong Kong and Macao, Bokmål to Nynorsk).
    #[test]
    fn translations_follow_the_english_and_replace_the_old_ones() {
        let german = i18n::by_tag("de").unwrap();
        let french = i18n::by_tag("fr").unwrap();
        let norwegian = i18n::by_tag("nb").unwrap();
        let brazilian = i18n::by_tag("pt-BR").unwrap();
        let chinese = i18n::by_tag("zh-Hant").unwrap();
        let summary = (Some(APPSTREAM_CONTEXT), "View slides");
        let paragraph = (Some(APPSTREAM_CONTEXT), "Viewer shows H&E slides.");
        let item = (Some(APPSTREAM_CONTEXT), "Fast");
        let catalogs = [
            Catalog::from_messages(
                german,
                "",
                [
                    (summary.0, summary.1, &["Präparate <ansehen>"][..]),
                    (
                        paragraph.0,
                        paragraph.1,
                        &["Viewer zeigt H&E-Präparate."][..],
                    ),
                    (item.0, item.1, &["Schnell"][..]),
                ],
            )
            .unwrap(),
            // French lacks the list item: its description stays English.
            Catalog::from_messages(
                french,
                "",
                [
                    (summary.0, summary.1, &["Voir des lames"][..]),
                    (
                        paragraph.0,
                        paragraph.1,
                        &["Viewer affiche des lames H&E."][..],
                    ),
                ],
            )
            .unwrap(),
            Catalog::from_messages(
                norwegian,
                "",
                [
                    (summary.0, summary.1, &["Vis preparater"][..]),
                    (
                        paragraph.0,
                        paragraph.1,
                        &["Viewer viser H&E-preparater."][..],
                    ),
                    (item.0, item.1, &["Rask"][..]),
                ],
            )
            .unwrap(),
            Catalog::from_messages(
                brazilian,
                "",
                [
                    (summary.0, summary.1, &["Ver lâminas"][..]),
                    (
                        paragraph.0,
                        paragraph.1,
                        &["Viewer mostra lâminas de H&E."][..],
                    ),
                    (item.0, item.1, &["Rápido"][..]),
                ],
            )
            .unwrap(),
            // Chinese lacks the paragraph: only its summary goes in.
            Catalog::from_messages(
                chinese,
                "",
                [
                    (summary.0, summary.1, &["檢視玻片"][..]),
                    (item.0, item.1, &["快速"][..]),
                ],
            )
            .unwrap(),
            // Release notes are not translated.
            Catalog::from_messages(
                i18n::by_tag("ja").unwrap(),
                "",
                [(Some(APPSTREAM_CONTEXT), "First release", &["初版"][..])],
            )
            .unwrap(),
        ];
        let translate = translations_in(&catalogs);
        let output = localized(ENGLISH, &translate).unwrap();
        assert_eq!(
            output,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!-- A <comment> -->
<component type="desktop-application">
  <id>org.example.Viewer</id>
  <name>Viewer</name>
  <summary>View slides</summary>
  <summary xml:lang="de">Präparate &lt;ansehen&gt;</summary>
  <summary xml:lang="fr">Voir des lames</summary>
  <summary xml:lang="nb">Vis preparater</summary>
  <summary xml:lang="nn">Vis preparater</summary>
  <summary xml:lang="no">Vis preparater</summary>
  <summary xml:lang="pt-BR">Ver lâminas</summary>
  <summary xml:lang="zh-TW">檢視玻片</summary>
  <summary xml:lang="zh-HK">檢視玻片</summary>
  <summary xml:lang="zh-MO">檢視玻片</summary>
  <description>
    <p>
      Viewer shows
      H&amp;E slides.
    </p>
    <p xml:lang="de">Viewer zeigt H&amp;E-Präparate.</p>
    <p xml:lang="nb">Viewer viser H&amp;E-preparater.</p>
    <p xml:lang="nn">Viewer viser H&amp;E-preparater.</p>
    <p xml:lang="no">Viewer viser H&amp;E-preparater.</p>
    <p xml:lang="pt-BR">Viewer mostra lâminas de H&amp;E.</p>
    <ul>
      <li>Fast</li>
      <li xml:lang="de">Schnell</li>
      <li xml:lang="nb">Rask</li>
      <li xml:lang="nn">Rask</li>
      <li xml:lang="no">Rask</li>
      <li xml:lang="pt-BR">Rápido</li>
    </ul>
  </description>
  <releases>
    <release version="1.0.0" date="2026-11-23">
      <description>
        <p>First release</p>
      </description>
    </release>
  </releases>
</component>
"#
        );
        // Again, it changes nothing; without translations, the English is
        // all that is left.
        assert_eq!(localized(&output, &translate).unwrap(), output);
        let english = localized(&output, &|_, _, _| None).unwrap();
        assert_eq!(
            english,
            ENGLISH
                .replace("  <summary xml:lang=\"de\">Alt</summary>\n", "")
                .replace("    <p xml:lang=\"fr\">Ancien</p>\n", "")
        );
    }

    /// Each `xml:lang` of a language is a locale whose users get that
    /// language in eov's window.
    #[test]
    fn every_locale_gets_the_language_eov_shows_there() {
        for language in i18n::translations() {
            for locale in appstream_locales(language) {
                assert_eq!(
                    i18n::negotiate(&locale).map(|negotiated| negotiated.tag),
                    Some(language.tag),
                    "{locale}"
                );
            }
        }
    }

    #[test]
    fn metadata_it_cannot_keep_as_it_is_is_refused() {
        let none: Translate = &|_, _, _| None;
        for (xml, problem) in [
            (
                "<component><summary>A <em>b</em></summary></component>",
                "markup inside",
            ),
            (
                "<component>\n  <summary>A</summary> <name>B</name>\n</component>",
                "shares its lines",
            ),
            (
                "<component><description><p>A &#38; B</p></description></component>",
                "not supported",
            ),
            ("<component><summary>A</component>", "closes <summary>"),
            ("<component><summary>A</summary>", "not closed"),
            ("<component><![CDATA[a]]></component>", "unsupported markup"),
        ] {
            let error = localized(xml, none).unwrap_err().to_string();
            assert!(error.contains(problem), "{xml}: {error}");
        }
    }
}
