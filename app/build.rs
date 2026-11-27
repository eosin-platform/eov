use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn main() {
    // The plugin surfaces are shown in `ComponentContainer`s, which are still
    // an experimental Slint feature. Slint takes experimental features by
    // name (an unknown name is an error).
    unsafe {
        std::env::set_var("SLINT_ENABLE_EXPERIMENTAL_FEATURES", "component-container");
    }
    // The UI tests find elements by their accessible names and roles
    // (i-slint-backend-testing's ElementHandle), which needs the element
    // information in the generated code. Release builds go without.
    if std::env::var("PROFILE").as_deref() == Ok("debug") {
        unsafe {
            std::env::set_var("SLINT_EMIT_DEBUG_INFO", "1");
        }
    }

    // The translations (see docs/translating.md): gettext catalogs in
    // translations/<language>/LC_MESSAGES/eov.po, compiled into eov. Slint
    // bundles the window's texts (`@tr`) from them; the texts made in Rust
    // (`tr!`) are turned into tables below. Nothing is read at run time, so
    // every package (AppImage, Flatpak, macOS app, Windows zip, cargo
    // install) has every language.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // EOV_TRANSLATIONS_DIR builds with other catalogs, such as the
    // pseudo-translations of `tools/i18n/i18n.py pseudo` (layout checks).
    println!("cargo:rerun-if-env-changed=EOV_TRANSLATIONS_DIR");
    let translations = std::env::var_os("EOV_TRANSLATIONS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("translations"));
    println!("cargo:rerun-if-changed={}", translations.display());
    let catalogs = catalogs(&translations);
    for (_, catalog) in &catalogs {
        println!("cargo:rerun-if-changed={}", catalog.display());
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out_dir.join("catalogs.rs"), rust_catalogs(&catalogs))
        .expect("cannot write catalogs.rs");

    let mut config = slint_build::CompilerConfiguration::new();
    if !catalogs.is_empty() {
        config = config.with_bundled_translations(&translations);
    }
    slint_build::compile_with_config("ui/app-window.slint", config).expect("Slint build failed");

    // OpenSlide is linked through eov-common (see common/build.rs); linking it
    // here would also attach it to eov-update-helper, which must stay
    // self-contained so it can run from a temporary directory.
}

/// The catalogs in `dir` (`<language>/LC_MESSAGES/eov.po`), by language, in
/// the order in which Slint bundles them (folder names, byte by byte).
fn catalogs(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut catalogs: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path().join("LC_MESSAGES").join("eov.po");
            let language = entry.file_name().into_string().ok()?;
            path.is_file().then_some((language, path))
        })
        .collect();
    catalogs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    catalogs
}

/// Rust source for `app/src/i18n.rs`: the bundled languages in Slint's
/// order, and for each language its plural rule and the translated texts
/// that Rust code uses (entries that occur in `.rs` files, or nowhere in
/// particular; a text that a `.slint` file shares with Rust code has the
/// references of both, as `tools/i18n/i18n.py extract` writes them).
/// Untranslated, fuzzy and obsolete entries are left out, as msgfmt does;
/// they show in English.
fn rust_catalogs(catalogs: &[(String, PathBuf)]) -> String {
    use rspolib::prelude::*;
    let mut code = String::from(
        "/// The languages Slint bundled into the window, in its order.\n\
         pub(crate) static BUNDLED_LANGUAGES: &[&str] = &[",
    );
    for (language, _) in catalogs {
        let _ = write!(code, "{language:?}, ");
    }
    code.push_str("];\n\n/// The translations of the texts made in Rust.\npub(crate) static RUST_CATALOGS: &[RustCatalog] = &[\n");
    for (language, path) in catalogs {
        let file = read_catalog(path);
        let plural_forms = file
            .metadata
            .get("Plural-Forms")
            .cloned()
            .unwrap_or_default();
        let _ = writeln!(
            code,
            "    RustCatalog {{\n        tag: {language:?},\n        plural_forms: {plural_forms:?},\n        messages: &["
        );
        for entry in &file.entries {
            let used_in_rust = entry.occurrences.is_empty()
                || entry
                    .occurrences
                    .iter()
                    .any(|(file, _)| file.ends_with(".rs"));
            if !entry.translated() || entry.msgid.is_empty() || !used_in_rust {
                continue;
            }
            let forms: Vec<&str> = match (&entry.msgstr, &entry.msgid_plural) {
                (Some(text), None) => vec![text.as_str()],
                _ => entry.msgstr_plural.iter().map(String::as_str).collect(),
            };
            let _ = writeln!(
                code,
                "            ({:?}, {:?}, &{:?}),",
                entry.msgctxt.as_deref(),
                entry.msgid,
                forms
            );
        }
        code.push_str("        ],\n    },\n");
    }
    code.push_str("];\n");
    code
}

/// Reads a catalog, or stops the build with an error that names it.
/// rspolib panics instead of returning an error on some malformed files (a
/// second `msgstr`, a `msgstr` right after `msgctxt`, a stray quoted line);
/// its own message names neither the file nor the line.
fn read_catalog(path: &Path) -> rspolib::POFile {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| rspolib::pofile(path));
    std::panic::set_hook(hook);
    match result {
        Ok(Ok(file)) => file,
        Ok(Err(error)) => panic!("{} is not a valid .po file: {error}", path.display()),
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|text| text.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            panic!(
                "{} is not a valid .po file: rspolib cannot read it ({reason}); look for a \
                 duplicated or misplaced msgstr or msgctxt, or a stray quoted line. \
                 `msgfmt --check -o /dev/null {}` or `python3 tools/i18n/i18n.py check` \
                 names the line.",
                path.display(),
                path.display()
            )
        }
    }
}
