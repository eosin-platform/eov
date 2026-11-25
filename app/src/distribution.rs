//! How this eov build was packaged, which decides how it updates itself.
//!
//! Exactly one `distribution-*` Cargo feature is enabled per build: release
//! packaging selects its own (`--no-default-features --features
//! distribution-appimage`, ...), and `cargo install eov` gets the default,
//! `distribution-cargo`.

use eov_plugin_api::i18n::tr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    MacosBundle,
    Cargo,
    Flatpak,
    AppImage,
    WindowsPortable,
    Unknown,
}

impl Distribution {
    /// A human-readable name, in the language of the interface (the
    /// window's update plan shows it).
    pub fn display_name(self) -> String {
        match self {
            // Translators: how eov was installed.
            Self::MacosBundle => tr!("macOS app bundle"),
            // Translators: how eov was installed: with the command "cargo install" (keep it as it is).
            Self::Cargo => tr!("cargo install"),
            Self::Flatpak => "Flatpak".to_string(),
            Self::AppImage => "AppImage".to_string(),
            // Translators: how eov was installed.
            Self::WindowsPortable => tr!("Windows portable folder"),
            // Translators: how eov was installed.
            Self::Unknown => tr!("unknown packaging"),
        }
    }
}

pub fn distribution() -> Distribution {
    if cfg!(feature = "distribution-cargo") {
        Distribution::Cargo
    } else if cfg!(feature = "distribution-flatpak") {
        Distribution::Flatpak
    } else if cfg!(feature = "distribution-appimage") {
        Distribution::AppImage
    } else if cfg!(feature = "distribution-macos-bundle") {
        Distribution::MacosBundle
    } else if cfg!(feature = "distribution-windows-portable") {
        Distribution::WindowsPortable
    } else {
        Distribution::Unknown
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(
        feature = "distribution-cargo",
        not(any(
            feature = "distribution-appimage",
            feature = "distribution-flatpak",
            feature = "distribution-macos-bundle",
            feature = "distribution-windows-portable"
        ))
    ))]
    #[test]
    fn default_feature_maps_to_cargo() {
        use super::{Distribution, distribution};

        assert_eq!(distribution(), Distribution::Cargo);
    }

    #[cfg(all(
        feature = "distribution-macos-bundle",
        not(any(
            feature = "distribution-cargo",
            feature = "distribution-appimage",
            feature = "distribution-flatpak",
            feature = "distribution-windows-portable"
        ))
    ))]
    #[test]
    fn macos_bundle_feature_maps_to_macos_bundle() {
        use super::{Distribution, distribution};

        assert_eq!(distribution(), Distribution::MacosBundle);
    }

    #[test]
    fn display_names_are_not_debug_names() {
        assert_eq!(
            super::Distribution::MacosBundle.display_name(),
            "macOS app bundle"
        );
    }
}
