pub enum Distribution {
    Homebrew,
    Cargo,
    Flatpak,
    AppImage,
    WindowsPortable,
    Unknown,
}

pub fn distribution() -> Distribution {
    if cfg!(feature = "distribution-homebrew") {
        Distribution::Homebrew
    } else if cfg!(feature = "distribution-cargo") {
        Distribution::Cargo
    } else if cfg!(feature = "distribution-flatpak") {
        Distribution::Flatpak
    } else if cfg!(feature = "distribution-appimage") {
        Distribution::AppImage
    } else if cfg!(feature = "distribution-windows-portable") {
        Distribution::WindowsPortable
    } else {
        Distribution::Unknown
    }
}