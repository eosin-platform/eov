//! How this eov build was packaged, which decides how it updates itself.
//!
//! Exactly one `distribution-*` Cargo feature is enabled per build: release
//! packaging selects its own (`--no-default-features --features
//! distribution-appimage`, ...), and `cargo install eov` gets the default,
//! `distribution-cargo`.

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
    /// A human-readable name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::MacosBundle => "macOS app bundle",
            Self::Cargo => "cargo install",
            Self::Flatpak => "Flatpak",
            Self::AppImage => "AppImage",
            Self::WindowsPortable => "Windows portable folder",
            Self::Unknown => "unknown packaging",
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
