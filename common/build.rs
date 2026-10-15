//! Links the OpenSlide shared library for every crate that depends on
//! eov-common.
//!
//! eov-common carries its own minimal OpenSlide binding (`src/openslide.rs`),
//! so this script both locates the library and tells rustc to link it
//! (`cargo:rustc-link-lib=dylib=openslide`). **OpenSlide 4.0 or newer is
//! required.**
//!
//! Resolution order for the library directory:
//! 1. `OPENSLIDE_LIB_DIR`: an explicit directory containing libopenslide
//!    (on Windows, the directory holding the `openslide.lib` import library).
//! 2. `pkg-config openslide`: the normal case, including source-built
//!    OpenSlide 4 installs under `/usr/local` that the default linker search
//!    path (notably rust-lld's) does not cover.
//! 3. The linker's default search path.
//!
//! Link directives emitted here propagate to the final link of every binary
//! and cdylib that depends on eov-common.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=OPENSLIDE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
    println!("cargo:rustc-link-lib=dylib=openslide");

    if let Some(dir) = std::env::var_os("OPENSLIDE_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&dir).display()
        );
        return;
    }

    match pkg_config::Config::new()
        .cargo_metadata(false)
        .env_metadata(false)
        .probe("openslide")
    {
        Ok(library) => {
            if let Some(version) = library.version.split('.').next()
                && version.parse::<u32>().is_ok_and(|major| major < 4)
            {
                println!(
                    "cargo:warning=OpenSlide {} found; eov requires OpenSlide 4.0 or newer",
                    library.version
                );
            }
            for path in library.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
        Err(error) => {
            println!(
                "cargo:warning=pkg-config could not locate OpenSlide ({error}); relying on the \
                 default linker search path. Set OPENSLIDE_LIB_DIR if linking fails."
            );
        }
    }
}
