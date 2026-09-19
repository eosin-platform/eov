#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    MacosBundle,
    Cargo,
    Flatpak,
    AppImage,
    WindowsPortable,
    Unknown,
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
    use super::*;

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
        assert_eq!(distribution(), Distribution::Cargo);
    }

    #[cfg(feature = "distribution-macos-bundle")]
    #[test]
    fn macos_bundle_feature_maps_to_macos_bundle() {
        assert_eq!(distribution(), Distribution::MacosBundle);
    }
}
