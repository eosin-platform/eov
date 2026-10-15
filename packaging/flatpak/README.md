# Flatpak packaging

This directory contains the Flatpak manifest for eov (`io.eosin.eov.yml`).

OpenSlide stays dynamically linked for LGPL-2.1 compliance: the manifest builds
the pinned OpenSlide fork (`Yanstart/openslide@00bebfb6a412485f7fbfe7fdb5919a3868594c13`),
libdicom and uthash as separate modules, and OpenSlide ships as
`/app/lib/libopenslide.so.1`. The licence texts and third-party notices are
installed to `/app/share/doc/eov` and `/app/share/licenses/io.eosin.eov`.

## Permissions

| Permission | Why |
|---|---|
| `--share=network` | `eov update` and `eov plugin install`. Both ask before going online (or need `--allow-network`/`-y`). |
| `--device=dri` | GPU rendering (Vulkan). |
| `--device=input` | Game controllers for the gamepad plugin. Needs Flatpak 1.15.6 or newer; older versions ignore it (see below). |
| `--filesystem=home` | Slides and folder series in your home directory. |
| `--filesystem=/media:ro`, `/run/media:ro`, `/mnt:ro` | Slides on removable and network drives, read-only. |

The file chooser portal hands eov single files only, so formats with sidecar
data (MIRAX `.mrxs` and its directory) and folder series need the directory
itself to be visible inside the sandbox.

### Slides in other locations

Grant access with `flatpak override` (add `--user` for a per-user installation,
or run it with `sudo` and without `--user` for a system-wide one):

```bash
# read-only access to a slide archive
flatpak override --user --filesystem=/srv/slides:ro io.eosin.eov
# everything on the host, read-only
flatpak override --user --filesystem=host:ro io.eosin.eov
# undo all overrides
flatpak override --user --reset io.eosin.eov
```

### Game controllers on Flatpak older than 1.15.6

Older Flatpak versions do not know `--device=input`. To use the gamepad plugin
there, grant all devices:

```bash
flatpak override --user --device=all io.eosin.eov
```

## Updating

The bundle cannot replace itself from inside the sandbox. `eov update`
downloads and verifies the new `.flatpak` bundle, then prints the command to
run on the host, matching the installation eov runs from:

```bash
flatpak install --user --bundle eov-vX.Y.Z-linux-x86_64.flatpak     # per-user
flatpak install --system --bundle eov-vX.Y.Z-linux-x86_64.flatpak   # system-wide
```

## Requirements

- `flatpak` and `flatpak-builder`. Building needs Flatpak **1.15.6 or newer**,
  because `flatpak build-finish` rejects `--device=input` on older versions.
  Ubuntu 24.04 ships 1.14; CI installs a current version from the Flatpak
  team's PPA (`ppa:flatpak/stable`).
- The Freedesktop 25.08 runtime and SDK, and the Rust SDK extension:

```bash
flatpak install flathub org.freedesktop.Platform//25.08 org.freedesktop.Sdk//25.08 org.freedesktop.Sdk.Extension.rust-stable//25.08
```

- For `build.sh`: a Python virtual environment at `.venv` with `aiohttp`,
  `PyYAML` and `tomlkit`, used by `flatpak-builder-tools` to vendor the Cargo
  dependencies (`cargo-sources.json`) so the build runs offline.

## Build locally

From the repository root:

```bash
./packaging/flatpak/build.sh
```

To install the result directly:

```bash
flatpak-builder --user --install --force-clean build-flatpak packaging/flatpak/io.eosin.eov.yml
```

To produce a repository and a single-file bundle, as CI does:

```bash
flatpak build-export repo build stable
flatpak build-bundle repo eov.flatpak io.eosin.eov stable
```

## Notes

- If this project ever carries local OpenSlide patches, those patches must be
  made available under LGPL-2.1 terms. The OpenSlide source of each release is
  also published with the release (`openslide-<commit>-source.tar.gz`).
- `packaging/shared/build-openslide.sh` pins the same sources for the other
  packages; CI checks that the two stay in sync.
