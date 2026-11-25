//! The slide formats eov opens and its `eov://` links, as eov declares them
//! to the operating system (SH-8), and `eov register-file-types`.
//!
//! - **Linux.** The shared-mime-info package
//!   `assets/linux/io.eosin.eov.xml` defines the vendor formats
//!   (shared-mime-info knows TIFF and DICOM), and the desktop entry's
//!   `MimeType=` lists them with `x-scheme-handler/eov`. The Flatpak and the
//!   AppImage ship both. An AppImage or a `cargo install` is not installed
//!   by a package manager, so `eov register-file-types` writes the package,
//!   a desktop entry that starts this very eov, and its icon to
//!   `$XDG_DATA_HOME` (`~/.local/share`), and updates the desktop's caches.
//! - **macOS.** `assets/macos/Info.plist` declares the formats
//!   (`CFBundleDocumentTypes`, `UTImportedTypeDeclarations`) and the link
//!   scheme (`CFBundleURLTypes`); macOS registers them when it first sees the
//!   app. The Apple events that carry the files are handled in
//!   `macos_open.rs`.
//! - **Windows** (portable zip, no installer). `eov register-file-types`
//!   writes `HKEY_CURRENT_USER\Software\Classes`: the `eov.Slide` ProgID,
//!   `OpenWithProgids` for every extension (so "Open with" offers eov; it
//!   becomes the default only for vendor formats no other program claims),
//!   `Applications\eov.exe`, and the `eov` URL protocol.
//!   `unregister-file-types` removes exactly those entries.
//!
//! Both commands change only the current user's settings, and only when
//! the user runs them. The tests check that the shipped assets match
//! [`SLIDE_TYPES`], and that it covers every extension eov opens.
//!
//! The Linux files speak the languages of eov's interface: the desktop
//! entry, the MIME package's names of the formats and the AppStream
//! metadata (`assets/linux/io.eosin.eov.metainfo.xml`) carry the
//! translations of eov's catalogs, which the tests check too
//! (`EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types` writes the
//! files again after the catalogs change; see docs/translating.md).

// The Linux files are only generated on Linux (and checked by the tests on
// every platform).
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
mod linux;
// The translations of the AppStream metadata, which is maintained by hand
// and installed as it is shipped: only its tests write it.
#[cfg(test)]
mod metainfo;
// The registry plan is tested everywhere and carried out on Windows.
#[cfg_attr(not(windows), allow(dead_code))]
mod registry;
#[cfg(windows)]
mod windows;

use anyhow::Result;
use eov_plugin_api::i18n::Language;

/// The desktop's id for eov (desktop entry, MIME package, icon).
pub(crate) const APP_ID: &str = "io.eosin.eov";

/// The scheme of eov links (`eov://open?path=…`).
pub(crate) const URL_SCHEME: &str = "eov";

/// How eov's MIME package defines a format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MimeDefinition {
    pub(crate) comment: &'static str,
    /// The format is a TIFF (`sub-class-of image/tiff`).
    pub(crate) tiff: bool,
    /// Another format uses the extension too: a TIFF header tells them
    /// apart.
    pub(crate) sniff_tiff: bool,
}

/// A slide format eov opens.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlideType {
    pub(crate) extension: &'static str,
    pub(crate) mime: &'static str,
    /// `Some` when eov's MIME package defines the type (shared-mime-info
    /// defines the others).
    pub(crate) definition: Option<MimeDefinition>,
    /// The macOS uniform type identifier (declared in Info.plist, which
    /// the tests check against it).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) uti: &'static str,
    /// No other program owns the format, so eov may be its default
    /// program (TIFF and DICOM have their own).
    pub(crate) eov_owns: bool,
}

const fn vendor(
    extension: &'static str,
    mime: &'static str,
    uti: &'static str,
    comment: &'static str,
    tiff: bool,
) -> SlideType {
    SlideType {
        extension,
        mime,
        definition: Some(MimeDefinition {
            comment,
            tiff,
            sniff_tiff: false,
        }),
        uti,
        eov_owns: true,
    }
}

const fn shared(extension: &'static str, mime: &'static str, uti: &'static str) -> SlideType {
    SlideType {
        extension,
        mime,
        definition: None,
        uti,
        eov_owns: false,
    }
}

/// BigTIFF, which large OME-TIFFs use (`.ome.btf`) and eov reads itself.
/// shared-mime-info has no type for it, and other microscopy programs open
/// it too, so eov is only offered for it.
const fn bigtiff(extension: &'static str) -> SlideType {
    SlideType {
        extension,
        mime: "image/x-bigtiff",
        definition: Some(MimeDefinition {
            comment: "BigTIFF image",
            // Classic TIFF readers cannot read it.
            tiff: false,
            sniff_tiff: false,
        }),
        uti: "io.eosin.eov.bigtiff",
        eov_owns: false,
    }
}

/// Every format of `eov_common::dataset::SUPPORTED_SLIDE_EXTENSIONS`.
pub(crate) const SLIDE_TYPES: &[SlideType] = &[
    vendor(
        "svs",
        "image/x-aperio-svs",
        "com.aperio.svs",
        "Aperio whole-slide image",
        true,
    ),
    vendor(
        "ndpi",
        "image/x-hamamatsu-ndpi",
        "jp.co.hamamatsu.ndpi",
        "Hamamatsu NDPI whole-slide image",
        true,
    ),
    vendor(
        "vms",
        "image/x-hamamatsu-vms",
        "jp.co.hamamatsu.vms",
        "Hamamatsu VMS whole-slide image",
        false,
    ),
    vendor(
        "vmu",
        "image/x-hamamatsu-vmu",
        "jp.co.hamamatsu.vmu",
        "Hamamatsu VMU whole-slide image",
        false,
    ),
    SlideType {
        definition: Some(MimeDefinition {
            comment: "Leica SCN whole-slide image",
            tiff: true,
            // Godot Engine scenes are *.scn too.
            sniff_tiff: true,
        }),
        ..vendor("scn", "image/x-leica-scn", "com.leica.scn", "", true)
    },
    vendor(
        "mrxs",
        "image/x-mirax",
        "com.3dhistech.mrxs",
        "MIRAX whole-slide image",
        false,
    ),
    vendor(
        "svslide",
        "image/x-sakura-svslide",
        "jp.co.sakura.svslide",
        "Sakura whole-slide image",
        false,
    ),
    vendor(
        "bif",
        "image/x-ventana-bif",
        "com.ventana.bif",
        "Ventana whole-slide image",
        true,
    ),
    vendor(
        "czi",
        "image/x-zeiss-czi",
        "com.zeiss.czi",
        "Zeiss CZI whole-slide image",
        false,
    ),
    vendor(
        "qptiff",
        "image/x-akoya-qptiff",
        "com.akoya.qptiff",
        "Akoya QPTIFF whole-slide image",
        true,
    ),
    shared("tif", "image/tiff", "public.tiff"),
    shared("tiff", "image/tiff", "public.tiff"),
    shared("dcm", "application/dicom", "org.nema.dicom"),
    bigtiff("btf"),
    bigtiff("tf2"),
    bigtiff("tf8"),
];

/// The context of the names of the formats (the MIME package's comments)
/// in eov's catalogs.
const FILE_TYPE_CONTEXT: &str = "File type";

/// The names of the formats for the catalogs: `xtr` reads `tr!` in the
/// source (this function is never compiled); the MIME package looks the
/// translations up for every language at once. The tests check that every
/// comment of [`SLIDE_TYPES`] is here.
#[cfg(any())]
fn file_type_texts_for_translators() {
    // Translators: the name of a file type, as file managers show it (in
    // the Type column, for example). Aperio, a maker of slide scanners,
    // stays as it is.
    tr!("File type" => "Aperio whole-slide image");
    // Translators: the name of a file type in file managers. Hamamatsu (a
    // maker of slide scanners) and NDPI (its format) stay as they are.
    tr!("File type" => "Hamamatsu NDPI whole-slide image");
    // Translators: the name of a file type in file managers. Hamamatsu (a
    // maker of slide scanners) and VMS (its format) stay as they are.
    tr!("File type" => "Hamamatsu VMS whole-slide image");
    // Translators: the name of a file type in file managers. Hamamatsu (a
    // maker of slide scanners) and VMU (its format) stay as they are.
    tr!("File type" => "Hamamatsu VMU whole-slide image");
    // Translators: the name of a file type in file managers. Leica (a
    // maker of slide scanners) and SCN (its format) stay as they are.
    tr!("File type" => "Leica SCN whole-slide image");
    // Translators: the name of a file type in file managers. MIRAX (the
    // format of 3DHISTECH's slide scanners) stays as it is.
    tr!("File type" => "MIRAX whole-slide image");
    // Translators: the name of a file type in file managers. Sakura, a
    // maker of slide scanners, stays as it is.
    tr!("File type" => "Sakura whole-slide image");
    // Translators: the name of a file type in file managers. Ventana, a
    // maker of slide scanners, stays as it is.
    tr!("File type" => "Ventana whole-slide image");
    // Translators: the name of a file type in file managers. Zeiss (a
    // maker of slide scanners) and CZI (its format) stay as they are.
    tr!("File type" => "Zeiss CZI whole-slide image");
    // Translators: the name of a file type in file managers. Akoya (a
    // maker of slide scanners) and QPTIFF (its format) stay as they are.
    tr!("File type" => "Akoya QPTIFF whole-slide image");
    // Translators: the name of a file type in file managers. BigTIFF, the
    // variant of TIFF for very large images, stays as it is.
    tr!("File type" => "BigTIFF image");
}

/// eov's shared-mime-info package, as shipped in
/// `assets/linux/io.eosin.eov.xml`.
pub(crate) fn mime_package() -> String {
    mime_package_with(&eov_translation)
}

/// [`mime_package`] with the translations `translate` gives.
fn mime_package_with(translate: Translate) -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!--\n  \
         Whole-slide image formats eov opens (through OpenSlide or its own\n  \
         OME-TIFF reader), for file managers (installed as\n  \
         share/mime/packages/io.eosin.eov.xml). shared-mime-info already knows\n  \
         TIFF (image/tiff) and DICOM (application/dicom); the desktop entry\n  \
         lists those too. Generated from SLIDE_TYPES in\n  \
         app/src/file_types/mod.rs, whose tests check this file, and from\n  \
         the translations in eov's catalogs (app/translations).\n\
         -->\n\
         <mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n",
    );
    for (index, slide) in SLIDE_TYPES.iter().enumerate() {
        let Some(definition) = slide.definition else {
            continue;
        };
        // One definition per type, with the globs of all its extensions.
        if SLIDE_TYPES[..index]
            .iter()
            .any(|earlier| earlier.mime == slide.mime)
        {
            continue;
        }
        xml.push_str(&format!("  <mime-type type=\"{}\">\n", slide.mime));
        xml.push_str(&format!(
            "    <comment>{}</comment>\n",
            xml_text(definition.comment)
        ));
        // File managers name the format in the user's language.
        for (language, translation) in
            translations_of(translate, FILE_TYPE_CONTEXT, definition.comment)
        {
            let translation = xml_text(&translation);
            for locale in desktop_locales(language) {
                xml.push_str(&format!(
                    "    <comment xml:lang=\"{locale}\">{translation}</comment>\n"
                ));
            }
        }
        if definition.tiff {
            xml.push_str("    <sub-class-of type=\"image/tiff\"/>\n");
        }
        xml.push_str("    <generic-icon name=\"image-x-generic\"/>\n");
        for same in SLIDE_TYPES.iter().filter(|same| same.mime == slide.mime) {
            xml.push_str(&format!("    <glob pattern=\"*.{}\"/>\n", same.extension));
        }
        if definition.sniff_tiff {
            xml.push_str(
                "    <!-- Another format uses the extension too: a TIFF header tells them\n         \
                 apart. Below image/tiff's priority, so other TIFFs stay image/tiff. -->\n    \
                 <magic priority=\"40\">\n      \
                 <match type=\"string\" value=\"II\\x2a\\x00\" offset=\"0\"/>\n      \
                 <match type=\"string\" value=\"MM\\x00\\x2a\" offset=\"0\"/>\n      \
                 <match type=\"string\" value=\"II\\x2b\\x00\" offset=\"0\"/>\n      \
                 <match type=\"string\" value=\"MM\\x00\\x2b\" offset=\"0\"/>\n    \
                 </magic>\n",
            );
        }
        xml.push_str("  </mime-type>\n");
    }
    xml.push_str("</mime-info>\n");
    xml
}

/// The `MimeType=` list of the desktop entry.
fn desktop_mime_types() -> String {
    let mut types: Vec<&str> = Vec::new();
    for slide in SLIDE_TYPES {
        if !types.contains(&slide.mime) {
            types.push(slide.mime);
        }
    }
    let mut list: String = types.iter().map(|mime| format!("{mime};")).collect();
    list.push_str(&format!("x-scheme-handler/{URL_SCHEME};"));
    list
}

/// eov's desktop entry, starting `exec` (an `Exec=` program, already
/// quoted). `registered` marks an entry written by `register-file-types`.
pub(crate) fn desktop_entry(exec: &str, registered: bool) -> String {
    desktop_entry_with(exec, registered, &eov_translation)
}

/// [`desktop_entry`] with the translations `translate` gives.
fn desktop_entry_with(exec: &str, registered: bool, translate: Translate) -> String {
    let mut entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=eov\n\
         GenericName={DESKTOP_GENERIC_NAME}\n\
         Comment={DESKTOP_COMMENT}\n\
         # %U: eov:// links need URLs, and eov decodes the file:// URIs file managers\n\
         # pass (GIO turns files on gvfs mounts into file:// URIs of local paths).\n\
         Exec={exec} %U\n\
         Icon={APP_ID}\n\
         Terminal=false\n\
         StartupNotify=true\n\
         StartupWMClass={APP_ID}\n\
         Categories=Science;Graphics;Viewer;\n\
         Keywords={DESKTOP_KEYWORDS}\n\
         # The slide formats of {APP_ID}.xml, TIFF and DICOM, and eov:// links.\n\
         MimeType={}\n",
        desktop_mime_types()
    );
    // The menus of desktops show the entry in the user's language: the
    // translations from eov's catalogs, keyed by the POSIX locale names the
    // desktop entry specification uses.
    entry.push_str("# In the languages of eov's interface (app/translations).\n");
    for (key, text) in [
        ("GenericName", DESKTOP_GENERIC_NAME),
        ("Comment", DESKTOP_COMMENT),
        ("Keywords", DESKTOP_KEYWORDS),
    ] {
        for (language, translation) in translations_of(translate, DESKTOP_CONTEXT, text) {
            let translation = if key == "Keywords" {
                keyword_list(&translation)
            } else {
                translation
            };
            // Keywords that turn out to be the English ones add nothing.
            if translation.is_empty() || translation == text {
                continue;
            }
            let value = desktop_value(&translation);
            for locale in desktop_locales(language) {
                entry.push_str(&format!("{key}[{locale}]={value}\n"));
            }
        }
    }
    if registered {
        entry.push_str(
            "# Written by `eov register-file-types`; `eov unregister-file-types` removes it.\n",
        );
        entry.push_str("X-Eov-Registered=true\n");
    }
    entry
}

/// A translation of the `Keywords` as the English writes them, however the
/// translator did: words separated by `;`, without spaces around them,
/// empty words or a `;` at the end.
fn keyword_list(translation: &str) -> String {
    translation
        .split(';')
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(";")
}

/// `text` as the value of a desktop entry key (Desktop Entry Specification,
/// "Possible value types"): backslash, newline, tab and carriage return are
/// escaped, so that a translation can neither split its line nor be dropped
/// by GLib as an invalid escape. `;` is left alone: it separates the
/// Keywords, and GLib drops a whole translated list that contains `\;`.
fn desktop_value(text: &str) -> String {
    let mut value = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' => value.push_str("\\\\"),
            '\n' => value.push_str("\\n"),
            '\t' => value.push_str("\\t"),
            '\r' => value.push_str("\\r"),
            c => value.push(c),
        }
    }
    value
}

/// The context of the desktop entry's texts in eov's catalogs.
const DESKTOP_CONTEXT: &str = "Desktop entry";
const DESKTOP_GENERIC_NAME: &str = "Whole-slide image viewer";
const DESKTOP_COMMENT: &str = "View whole-slide images with OpenSlide-powered tile rendering";
const DESKTOP_KEYWORDS: &str = "whole-slide;wsi;pathology;microscopy;openslide;microbiology";

/// The desktop entry's texts for the catalogs: `xtr` reads `tr!` in the
/// source (this function is never compiled); the entry looks the
/// translations up for every language at once.
#[cfg(any())]
fn desktop_texts_for_translators() {
    // Translators: the kind of program, under its name in application menus.
    tr!("Desktop entry" => "Whole-slide image viewer");
    // Translators: a tooltip in application menus.
    tr!("Desktop entry" => "View whole-slide images with OpenSlide-powered tile rendering");
    // Translators: search words for application menus, separated by
    // semicolons; keep the English ones and add your language's.
    tr!("Desktop entry" => "whole-slide;wsi;pathology;microscopy;openslide;microbiology");
}

/// The AppStream metadata's texts for the catalogs (see
/// `desktop_texts_for_translators`): its summary and the paragraphs of its
/// description, as `metainfo.rs` reads them from
/// `assets/linux/io.eosin.eov.metainfo.xml` (whitespace collapsed). The
/// tests check that every one of them is here.
#[cfg(any())]
fn appstream_texts_for_translators() {
    // Translators: eov's one-line summary in software centres (GNOME
    // Software, KDE Discover, Flathub), next to its name. No full stop.
    tr!("AppStream" => "Fast desktop viewer for whole-slide images");
    // Translators: the description of eov in software centres (GNOME
    // Software, KDE Discover, Flathub). eov, Rust and WSI stay as they are.
    tr!("AppStream" => "eov is a desktop viewer for whole-slide images built in Rust. It fills a niche in the WSI ecosystem: a small, high-performance workbench for quickly viewing WSI files on your local machine.");
}

/// Looks up eov's translation of a text (with its context) into a
/// language: [`eov_translation`], or a test's catalogs.
type Translate<'a> = &'a dyn Fn(&'static Language, &str, &str) -> Option<String>;

/// The translation in the catalogs compiled into eov.
fn eov_translation(language: &'static Language, context: &str, text: &str) -> Option<String> {
    crate::i18n::translation(language, Some(context), text)
}

/// The translations of `text` that differ from it, in the order of eov's
/// languages.
fn translations_of(
    translate: Translate,
    context: &str,
    text: &str,
) -> Vec<(&'static Language, String)> {
    eov_plugin_api::i18n::translations()
        .iter()
        .filter_map(|language| {
            let translation = translate(language, context, text)?;
            (translation != text).then_some((language, translation))
        })
        .collect()
}

/// The locale names under which desktop entries (`Comment[pt_BR]`) and
/// shared-mime-info (`xml:lang`) carry a language's translation: those of
/// the users eov's window shows the language to (`i18n::negotiate`), in
/// POSIX form (GIO does not match `pt-BR`). Desktops look a user's
/// `zh_HK.UTF-8` up as `zh_HK`, then as `zh`, never by another country or
/// by script. So Chinese is written under each country that reads its
/// script, and Simplified also under `zh` alone, for `zh` and the other
/// countries, which `negotiate` gives Simplified (a `zh_HK` user finds
/// `zh_HK` first). Bokmål is also written under `nn` and `no`, whose users
/// get it in eov. Not under `pt`: it would reach Portugal's users, who get
/// no translation. (No glibc locale uses the old codes `iw` and `in`, which
/// `negotiate` reads too.) AppStream takes BCP 47 tags instead
/// (`metainfo.rs`).
fn desktop_locales(language: &Language) -> Vec<String> {
    let locales: &[&str] = match language.tag {
        "zh-Hans" => &["zh_CN", "zh_SG", "zh_MY", "zh"],
        "zh-Hant" => &["zh_TW", "zh_HK", "zh_MO"],
        "nb" => &["nb", "nn", "no"],
        tag => return vec![tag.replace('-', "_")],
    };
    locales.iter().map(|locale| locale.to_string()).collect()
}

/// `text` as the text of an XML element: markup escaped, and the characters
/// XML 1.0 forbids (C0 controls other than tab, LF and CR; U+FFFE, U+FFFF)
/// left out, so that a stray control character in a translation cannot make
/// the MIME package or the metainfo unreadable (update-mime-database skips
/// such a package, and still succeeds).
fn xml_text(text: &str) -> String {
    let mut xml = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => xml.push_str("&amp;"),
            '<' => xml.push_str("&lt;"),
            '>' => xml.push_str("&gt;"),
            '\t' | '\n' | '\r' => xml.push(c),
            c if c < ' ' || c == '\u{FFFE}' || c == '\u{FFFF}' => {}
            c => xml.push(c),
        }
    }
    xml
}

/// Runs `eov register-file-types` (`register`) or
/// `eov unregister-file-types`.
pub(crate) fn run_cli(register: bool, dry_run: bool) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::run(register, dry_run)
    }
    #[cfg(windows)]
    {
        windows::run(register, dry_run)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = (register, dry_run);
        anyhow::bail!(
            "the eov app declares its file types and eov:// links itself; macOS registers \
             them when it first sees the app (open it once from Applications)"
        )
    }
}

/// A lookup in catalogs a test made ([`Catalog::from_messages`]).
///
/// [`Catalog::from_messages`]: eov_plugin_api::i18n::Catalog::from_messages
#[cfg(test)]
fn translations_in(
    catalogs: &[eov_plugin_api::i18n::Catalog],
) -> impl Fn(&'static Language, &str, &str) -> Option<String> + '_ {
    move |language, context, text| {
        catalogs
            .iter()
            .find(|catalog| catalog.language() == language)?
            .translate(Some(context), text)
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eov_plugin_api::i18n::{self, Catalog};

    #[test]
    fn every_extension_eov_opens_is_declared() {
        let mut declared: Vec<&str> = SLIDE_TYPES.iter().map(|slide| slide.extension).collect();
        let mut supported = eov_common::dataset::SUPPORTED_SLIDE_EXTENSIONS.to_vec();
        declared.sort_unstable();
        supported.sort_unstable();
        assert_eq!(declared, supported);
    }

    const SHIPPED_MIME_PACKAGE: &str = include_str!("../../../assets/linux/io.eosin.eov.xml");
    const SHIPPED_DESKTOP_ENTRY: &str = include_str!("../../../assets/linux/io.eosin.eov.desktop");
    const SHIPPED_METAINFO: &str = include_str!("../../../assets/linux/io.eosin.eov.metainfo.xml");

    /// `text` with `\n` line ends (Git may check text files out with CRLF
    /// on Windows).
    fn lf(text: &str) -> String {
        text.replace("\r\n", "\n")
    }

    /// Checks that `assets/linux/<file>` (`shipped`) is what eov generates;
    /// with `EOV_REGENERATE_ASSETS` set, writes it instead. After the table
    /// or the catalogs change: `EOV_REGENERATE_ASSETS=1 cargo test -p eov
    /// file_types` (see docs/translating.md).
    fn check_or_write(file: &str, shipped: &str, generated: &str) {
        if std::env::var_os("EOV_REGENERATE_ASSETS").is_some() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../assets/linux")
                .join(file);
            std::fs::write(&path, generated)
                .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
            return;
        }
        assert_eq!(
            lf(shipped),
            generated,
            "assets/linux/{file} is out of date: run EOV_REGENERATE_ASSETS=1 cargo test -p eov file_types"
        );
    }

    #[test]
    fn the_shipped_mime_package_is_generated_from_the_table() {
        check_or_write(
            &format!("{APP_ID}.xml"),
            SHIPPED_MIME_PACKAGE,
            &mime_package(),
        );
    }

    #[test]
    fn the_shipped_desktop_entry_is_generated_from_the_table() {
        let entry = desktop_entry("eov", false);
        check_or_write(&format!("{APP_ID}.desktop"), SHIPPED_DESKTOP_ENTRY, &entry);
        assert!(entry.contains("x-scheme-handler/eov;"));
    }

    #[test]
    fn the_shipped_metainfo_has_the_translations_of_the_catalogs() {
        let localized = metainfo::localized(&lf(SHIPPED_METAINFO), &eov_translation).unwrap();
        check_or_write(
            &format!("{APP_ID}.metainfo.xml"),
            SHIPPED_METAINFO,
            &localized,
        );
    }

    /// The texts the Linux files translate, with their contexts.
    fn translated_texts() -> std::collections::BTreeSet<(String, String)> {
        let asked = std::cell::RefCell::new(std::collections::BTreeSet::new());
        let record = |_: &'static Language, context: &str, text: &str| {
            asked
                .borrow_mut()
                .insert((context.to_string(), text.to_string()));
            None
        };
        mime_package_with(&record);
        desktop_entry_with("eov", false, &record);
        metainfo::localized(&lf(SHIPPED_METAINFO), &record).unwrap();
        asked.into_inner()
    }

    /// Every text the Linux files translate is declared for `xtr` in this
    /// file (`*_texts_for_translators`), so that it reaches the catalogs.
    #[test]
    fn every_translated_text_is_declared_for_the_translators() {
        let asked = translated_texts();
        for context in [
            FILE_TYPE_CONTEXT,
            DESKTOP_CONTEXT,
            metainfo::APPSTREAM_CONTEXT,
        ] {
            assert!(asked.iter().any(|(asked, _)| asked == context));
        }
        let source = include_str!("mod.rs");
        for (context, text) in asked {
            let declaration = format!("tr!({context:?} => {text:?});");
            assert!(
                source.contains(&declaration),
                "{declaration} is not declared for the translators"
            );
        }
    }

    /// The catalogs translate the Linux files' texts without control
    /// characters. The files would keep one (a line break in a menu's name,
    /// say; XML cannot even hold some, so those are left out), so the
    /// translator hears of it here.
    #[test]
    fn the_linux_files_translations_have_no_control_characters() {
        let texts = translated_texts();
        for language in i18n::translations() {
            for (context, text) in &texts {
                let Some(translation) = eov_translation(language, context, text) else {
                    continue;
                };
                assert!(
                    !translation.chars().any(char::is_control),
                    "the {} translation of {text:?} (context {context:?}) has a control \
                     character: {translation:?}",
                    language.tag
                );
            }
        }
    }

    /// The names a desktop looks a user's POSIX locale
    /// (`lang_COUNTRY.ENCODING@MODIFIER`) up by, best first, as the Desktop
    /// Entry Specification and GIO (`g_get_language_names`) do:
    /// `lang_COUNTRY@MODIFIER`, `lang_COUNTRY`, `lang@MODIFIER`, `lang`.
    fn lookup_names(locale: &str) -> Vec<String> {
        let (name, modifier) = locale.split_once('@').unwrap_or((locale, ""));
        let name = name.split('.').next().unwrap_or_default();
        let language = name.split('_').next().unwrap_or_default();
        let mut names = Vec::new();
        for base in [name, language] {
            if !modifier.is_empty() {
                names.push(format!("{base}@{modifier}"));
            }
            names.push(base.to_string());
        }
        names
    }

    /// Menus and file managers show a user the translation eov's window
    /// shows them (`i18n::negotiate`), also where it is another locale's
    /// (Hong Kong, Singapore, Nynorsk), and none where eov shows English
    /// (Portugal).
    #[test]
    fn desktops_find_the_language_eov_shows() {
        let mut written = Vec::new();
        for language in i18n::translations() {
            for locale in desktop_locales(language) {
                assert_eq!(
                    i18n::negotiate(&locale).map(|negotiated| negotiated.tag),
                    Some(language.tag),
                    "{locale}"
                );
                written.push((locale, language.tag));
            }
        }
        for user in [
            "zh_CN.UTF-8",
            "zh_SG.UTF-8",
            "zh_MY",
            "zh",
            "zh_TW.UTF-8",
            "zh_HK.UTF-8",
            "zh_MO",
            "nb_NO.UTF-8",
            "nn_NO.UTF-8",
            "no_NO",
            "pt_BR.UTF-8",
            "pt_PT.UTF-8",
            "pt_PT@euro",
            "de_AT.UTF-8@euro",
            "ja_JP.eucJP",
            "sr_RS",
            "en_US.UTF-8",
            "C",
        ] {
            let found = lookup_names(user).into_iter().find_map(|name| {
                written
                    .iter()
                    .find(|(locale, _)| *locale == name)
                    .map(|(_, tag)| *tag)
            });
            let shown = i18n::negotiate(user)
                .map(|language| language.tag)
                .filter(|tag| *tag != "en");
            assert_eq!(found, shown, "{user}");
        }
    }

    /// File managers, application menus and software centres get the
    /// translations of the catalogs, by the locale names their files use:
    /// POSIX in the desktop entry and the MIME package (`pt_BR`), BCP 47 in
    /// the AppStream metadata (`pt-BR`), for every locale whose users read
    /// the language in eov (Traditional Chinese for Hong Kong and Macao
    /// too). Whatever a translation holds, the files stay readable.
    #[test]
    fn the_files_carry_the_translations_of_the_catalogs() {
        let german = i18n::by_tag("de").unwrap();
        let portuguese = i18n::by_tag("pt-BR").unwrap();
        let chinese = i18n::by_tag("zh-Hant").unwrap();
        // Keywords are written as the English ones are, without spaces,
        // empty ones or a `;` at the end; those that are the English add
        // nothing.
        let german_keywords = format!("{DESKTOP_KEYWORDS}; Pathologie ;;Mikroskopie;");
        let portuguese_keywords = format!("{DESKTOP_KEYWORDS};");
        let catalogs = [
            Catalog::from_messages(
                german,
                "",
                [
                    (
                        Some(FILE_TYPE_CONTEXT),
                        "Aperio whole-slide image",
                        &["Aperio-Whole-Slide-Image"][..],
                    ),
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_GENERIC_NAME,
                        &["Viewer für Whole-Slide-Images"][..],
                    ),
                    // A line break and a backslash, escaped.
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_COMMENT,
                        &["Ansehen,\nmit Kacheln (Hellfeld\\Fluoreszenz)"][..],
                    ),
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_KEYWORDS,
                        &[german_keywords.as_str()][..],
                    ),
                    (
                        Some(metainfo::APPSTREAM_CONTEXT),
                        "Fast desktop viewer for whole-slide images",
                        &["Schneller Desktop-Viewer für Whole-Slide-Images"][..],
                    ),
                ],
            )
            .unwrap(),
            Catalog::from_messages(
                portuguese,
                "",
                [
                    // Characters XML forbids are left out.
                    (
                        Some(FILE_TYPE_CONTEXT),
                        "BigTIFF image",
                        &["Imagem BigTIFF\u{b} (P&D)\u{11}"][..],
                    ),
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_GENERIC_NAME,
                        &["Visualizador de imagens de lâmina inteira"][..],
                    ),
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_KEYWORDS,
                        &[portuguese_keywords.as_str()][..],
                    ),
                    (
                        Some(metainfo::APPSTREAM_CONTEXT),
                        "Fast desktop viewer for whole-slide images",
                        &["Visualizador rápido\u{c} de imagens de lâmina inteira\u{FFFF}"][..],
                    ),
                ],
            )
            .unwrap(),
            Catalog::from_messages(
                chinese,
                "",
                [
                    (
                        Some(FILE_TYPE_CONTEXT),
                        "Aperio whole-slide image",
                        &["Aperio 全玻片影像"][..],
                    ),
                    (
                        Some(DESKTOP_CONTEXT),
                        DESKTOP_GENERIC_NAME,
                        &["全玻片影像檢視器"][..],
                    ),
                    (
                        Some(metainfo::APPSTREAM_CONTEXT),
                        "Fast desktop viewer for whole-slide images",
                        &["快速的全玻片影像桌面檢視器"][..],
                    ),
                ],
            )
            .unwrap(),
        ];
        let translate = translations_in(&catalogs);

        let package = mime_package_with(&translate);
        assert!(
            package.contains(
                "    <comment>Aperio whole-slide image</comment>\n    \
                 <comment xml:lang=\"de\">Aperio-Whole-Slide-Image</comment>\n    \
                 <comment xml:lang=\"zh_TW\">Aperio 全玻片影像</comment>\n    \
                 <comment xml:lang=\"zh_HK\">Aperio 全玻片影像</comment>\n    \
                 <comment xml:lang=\"zh_MO\">Aperio 全玻片影像</comment>\n    \
                 <sub-class-of type=\"image/tiff\"/>\n"
            ),
            "{package}"
        );
        assert!(
            package.contains(
                "    <comment>BigTIFF image</comment>\n    \
                 <comment xml:lang=\"pt_BR\">Imagem BigTIFF (P&amp;D)</comment>\n    \
                 <generic-icon"
            ),
            "{package}"
        );
        // A format the catalogs do not translate has its English name only.
        assert!(
            package
                .contains("    <comment>Zeiss CZI whole-slide image</comment>\n    <generic-icon")
        );

        let entry = desktop_entry_with("eov", false, &translate);
        assert!(
            entry.contains(
                "\nGenericName[de]=Viewer für Whole-Slide-Images\n\
                 GenericName[pt_BR]=Visualizador de imagens de lâmina inteira\n\
                 GenericName[zh_TW]=全玻片影像檢視器\n\
                 GenericName[zh_HK]=全玻片影像檢視器\n\
                 GenericName[zh_MO]=全玻片影像檢視器\n"
            ),
            "{entry}"
        );
        assert!(
            entry.contains("\nComment[de]=Ansehen,\\nmit Kacheln (Hellfeld\\\\Fluoreszenz)\n"),
            "{entry}"
        );
        assert!(
            entry.contains(&format!(
                "\nKeywords[de]={DESKTOP_KEYWORDS};Pathologie;Mikroskopie\n"
            )),
            "{entry}"
        );
        assert!(!entry.contains("Keywords[pt_BR]"), "{entry}");
        // Every line is still a comment, a group or a key.
        for line in entry.lines() {
            assert!(
                line.is_empty() || line.starts_with(['#', '[']) || line.contains('='),
                "{line:?} in {entry}"
            );
        }

        let metainfo = metainfo::localized(&lf(SHIPPED_METAINFO), &translate).unwrap();
        assert!(
            metainfo.contains(
                "  <summary>Fast desktop viewer for whole-slide images</summary>\n  \
                 <summary xml:lang=\"de\">Schneller Desktop-Viewer für Whole-Slide-Images</summary>\n  \
                 <summary xml:lang=\"pt-BR\">Visualizador rápido de imagens de lâmina inteira</summary>\n  \
                 <summary xml:lang=\"zh-TW\">快速的全玻片影像桌面檢視器</summary>\n  \
                 <summary xml:lang=\"zh-HK\">快速的全玻片影像桌面檢視器</summary>\n  \
                 <summary xml:lang=\"zh-MO\">快速的全玻片影像桌面檢視器</summary>\n"
            ),
            "{metainfo}"
        );
        for file in [&package, &metainfo] {
            assert!(
                !file.contains(['\u{b}', '\u{c}', '\u{11}', '\u{FFFF}']),
                "{file}"
            );
        }
    }

    #[test]
    fn the_mac_app_declares_every_format_and_the_link_scheme() {
        let plist = include_str!("../../../assets/macos/Info.plist");
        let (documents, rest) = plist
            .split_once("<key>UTImportedTypeDeclarations</key>")
            .expect("imported type declarations");
        for slide in SLIDE_TYPES {
            let uti = format!("<string>{}</string>", slide.uti);
            assert!(
                documents.contains(&uti),
                "{} is not a document type",
                slide.uti
            );
            if slide.uti != "public.tiff" {
                assert!(rest.contains(&uti), "{} is not declared", slide.uti);
                assert!(
                    rest.contains(&format!("<string>{}</string>", slide.extension)),
                    ".{} has no type declaration",
                    slide.extension
                );
            }
        }
        assert!(plist.contains("<key>CFBundleURLSchemes</key>"));
        assert!(plist.contains(&format!("<string>{URL_SCHEME}</string>")));
    }
}
