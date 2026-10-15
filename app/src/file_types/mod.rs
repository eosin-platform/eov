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

// The Linux files are only generated on Linux (and checked by the tests on
// every platform).
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
mod linux;
// The registry plan is tested everywhere and carried out on Windows.
#[cfg_attr(not(windows), allow(dead_code))]
mod registry;
#[cfg(windows)]
mod windows;

use anyhow::Result;

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
    shared("tif", "image/tiff", "public.tiff"),
    shared("tiff", "image/tiff", "public.tiff"),
    shared("dcm", "application/dicom", "org.nema.dicom"),
];

/// eov's shared-mime-info package, as shipped in
/// `assets/linux/io.eosin.eov.xml`.
pub(crate) fn mime_package() -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!--\n  \
         Whole-slide image formats eov opens through OpenSlide, for file managers\n  \
         (installed as share/mime/packages/io.eosin.eov.xml). shared-mime-info\n  \
         already knows TIFF (image/tiff) and DICOM (application/dicom); the\n  \
         desktop entry lists those too. Generated from SLIDE_TYPES in\n  \
         app/src/file_types/mod.rs, whose tests check this file.\n\
         -->\n\
         <mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n",
    );
    for slide in SLIDE_TYPES {
        let Some(definition) = slide.definition else {
            continue;
        };
        xml.push_str(&format!("  <mime-type type=\"{}\">\n", slide.mime));
        xml.push_str(&format!("    <comment>{}</comment>\n", definition.comment));
        if definition.tiff {
            xml.push_str("    <sub-class-of type=\"image/tiff\"/>\n");
        }
        xml.push_str("    <generic-icon name=\"image-x-generic\"/>\n");
        xml.push_str(&format!("    <glob pattern=\"*.{}\"/>\n", slide.extension));
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
    let mut entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=eov\n\
         GenericName=Whole-slide image viewer\n\
         Comment=View whole-slide images with OpenSlide-powered tile rendering\n\
         # %U: eov:// links need URLs, and eov decodes the file:// URIs file managers\n\
         # pass (GIO turns files on gvfs mounts into file:// URIs of local paths).\n\
         Exec={exec} %U\n\
         Icon={APP_ID}\n\
         Terminal=false\n\
         StartupNotify=true\n\
         StartupWMClass={APP_ID}\n\
         Categories=Science;Graphics;Viewer;\n\
         Keywords=whole-slide;wsi;pathology;microscopy;openslide;microbiology\n\
         # The slide formats of {APP_ID}.xml, TIFF and DICOM, and eov:// links.\n\
         MimeType={}\n",
        desktop_mime_types()
    );
    if registered {
        entry.push_str(
            "# Written by `eov register-file-types`; `eov unregister-file-types` removes it.\n",
        );
        entry.push_str("X-Eov-Registered=true\n");
    }
    entry
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_extension_eov_opens_is_declared() {
        let mut declared: Vec<&str> = SLIDE_TYPES.iter().map(|slide| slide.extension).collect();
        let mut supported = eov_common::dataset::SUPPORTED_SLIDE_EXTENSIONS.to_vec();
        declared.sort_unstable();
        supported.sort_unstable();
        assert_eq!(declared, supported);
    }

    /// Rewrites the shipped Linux files from the table after a change:
    /// `cargo test -p eov --bin eov write_linux_assets -- --ignored`.
    #[test]
    #[ignore = "rewrites assets/linux"]
    fn write_linux_assets() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/linux");
        std::fs::write(dir.join(format!("{APP_ID}.xml")), mime_package()).unwrap();
        std::fs::write(
            dir.join(format!("{APP_ID}.desktop")),
            desktop_entry("eov", false),
        )
        .unwrap();
    }

    #[test]
    fn the_shipped_mime_package_is_generated_from_the_table() {
        // Git may check text files out with CRLF on Windows.
        let shipped = include_str!("../../../assets/linux/io.eosin.eov.xml").replace("\r\n", "\n");
        assert_eq!(
            shipped,
            mime_package(),
            "regenerate assets/linux/io.eosin.eov.xml from mime_package()"
        );
    }

    #[test]
    fn the_shipped_desktop_entry_is_generated_from_the_table() {
        let shipped =
            include_str!("../../../assets/linux/io.eosin.eov.desktop").replace("\r\n", "\n");
        assert_eq!(
            shipped,
            desktop_entry("eov", false),
            "regenerate assets/linux/io.eosin.eov.desktop from desktop_entry()"
        );
        assert!(shipped.contains("x-scheme-handler/eov;"));
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
