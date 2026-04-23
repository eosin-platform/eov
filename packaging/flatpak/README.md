# Flatpak packaging

This directory contains the Flatpak manifest for eov.

The manifest keeps OpenSlide dynamically linked for LGPL-2.1 compliance by
building OpenSlide as a separate shared library module inside the Flatpak app
prefix. It does not statically link OpenSlide into the Rust binary.

## Requirements

- `flatpak`
- `flatpak-builder`
- Freedesktop 24.08 runtime and SDK
- `org.freedesktop.Sdk.Extension.rust-stable`

Install the SDK pieces locally if needed:

```bash
flatpak install flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08 org.freedesktop.Sdk.Extension.rust-stable//24.08
```

## Build locally

From the repository root:

```bash
flatpak-builder --user --install --force-clean build-flatpak packaging/flatpak/io.eosin.eov.yml
```

To produce a repo/export instead of installing immediately:

```bash
flatpak-builder --force-clean --repo=repo build-flatpak packaging/flatpak/io.eosin.eov.yml
```

## Notes

- OpenSlide is sourced from
  `Yanstart/openslide@00bebfb6a412485f7fbfe7fdb5919a3868594c13` and built as a
  shared library module.
- If this project ever carries local OpenSlide patches, those patches must be
  made available under LGPL-2.1 terms.
- For fully reproducible Flatpak builds, maintainers may still want to vendor
  Cargo dependencies or add generated Cargo source metadata later.
