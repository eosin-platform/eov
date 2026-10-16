use std::path::{Path, PathBuf};

fn main() {
    unsafe {
        std::env::set_var("SLINT_ENABLE_EXPERIMENTAL_FEATURES", "1");
    }
    // The UI tests find elements by their accessible names and roles
    // (i-slint-backend-testing's ElementHandle), which needs the element
    // information in the generated code. Release builds go without.
    if std::env::var("PROFILE").as_deref() == Ok("debug") {
        unsafe {
            std::env::set_var("SLINT_EMIT_DEBUG_INFO", "1");
        }
    }

    // Translations of the interface's texts (`@tr`) are compiled in when
    // there are any: gettext catalogs in
    // translations/<language>/LC_MESSAGES/eov.po (see the README). The
    // language follows the system's; without a catalog for it, and without
    // any catalog, the texts are the English ones.
    let translations =
        Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("translations");
    println!("cargo:rerun-if-changed={}", translations.display());
    let catalogs = catalogs(&translations);
    for catalog in &catalogs {
        println!("cargo:rerun-if-changed={}", catalog.display());
    }
    let mut config = slint_build::CompilerConfiguration::new();
    if !catalogs.is_empty() {
        config = config.with_bundled_translations(&translations);
    }
    slint_build::compile_with_config("ui/app-window.slint", config).expect("Slint build failed");

    // OpenSlide is linked through eov-common (see common/build.rs); linking it
    // here would also attach it to eov-update-helper, which must stay
    // self-contained so it can run from a temporary directory.
}

/// The catalogs in `dir` (`<language>/LC_MESSAGES/eov.po`).
fn catalogs(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("LC_MESSAGES").join("eov.po"))
        .filter(|path| path.is_file())
        .collect()
}
