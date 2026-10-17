# TODO after the 1.0 improvement pass

What is left of the work on [AUDIT.md](AUDIT.md). The audit's status section
lists every finding and §14 feature; regenerate it after a change with
`python3 tools/audit_status/gen_status.py` (edit `status_data.py` or
`features_data.py` next to it first). Nothing below has been pushed.

## Open features

- **SH-10:** Copy View cannot put images on the clipboard in the Flatpak
  under Wayland. Either enable arboard's `wayland-data-control` feature
  (its crates were not available offline) or give the Flatpak X11 access.
- **Vendor Z-stacks** (NDPI, VMS, multi-focal DICOM) show one plane:
  OpenSlide exposes only one.
- **Readers for iSyntax, Olympus/Evident VSI, Leica LIF and CZI
  fluorescence**: plugins can now add slide formats (`[[slide_formats]]`,
  see plugin_api/README.md), but no such reader exists yet.
- **Slide formats from plugins, what is left:**
  - URI schemes (remote slides such as DICOMweb or OMERO): manifests may
    declare them, but eov opens files only. `WsiFile::open` would need to
    take URIs, and slide identity would need `SlideInfo::identity`.
  - `eov batch` runs do not record the versions of the plugins that read
    the slides, only the plugin that analysed them.
- **DICOM SR** is neither read nor written. DICOM Microscopy Bulk Simple
  Annotations are.
- **Localisation:**
  - No translation catalogs ship.
  - Texts made in Rust stay English: status messages, menus built in Rust,
    the theme menu and the plugin usage line.
  - No `.pot` template ships, because slint-tr-extractor was not available
    offline.
- **Themes:** panels provided by plugins are not themed or translated.
- **Signed binaries:** the owner has to provide the release signing key,
  Authenticode and notarization certificates.

## Not verified

- **Platforms:** Windows and macOS builds and behaviour. The GUI was only run
  under X11 in nested Xephyr servers; it was not run live on Wayland.
- **Screen reader:** no real one (Orca) was run. Accessibility was checked
  through Slint's testing backend.
- **DICOM bulk annotations:** no third-party reader (highdicom, Slim,
  QuPath) was available to check the exports. Three points rest on reading
  the standard:
  - that the index list counts coordinate values;
  - that the colour attribute is allowed in group items;
  - that a 2D object may carry a Frame of Reference module.
- **Release workflow:** not run. The `v1.0.0` tag does not exist.
- **Flaky test:** under heavy load, the `instance::` socket tests failed once
  with "Resource temporarily unavailable". They pass on reruns.

## Housekeeping

- **Unpushed commits:** master is ahead of origin/master, and the cellaris
  submodule's `improvements/prepare-v1.0.0` ahead of its remote branch.
  Push the submodule before the superproject that points at it.
- **Leftovers in the home directory.** An early test run, before tests were
  isolated, wrote:
  - `[active_sidebar]` in `~/.eov/config.toml`;
  - `~/.config/eov/eovae.json`;
  - about 25 fixture directories in `~/.cache/eov/plugin-packages`.

  Review them and remove them if they are unwanted.
- **Worktrees:** `.claude/worktrees/` holds the agents' worktrees. Their
  work is merged (the slide-source sketch was finished on master), so they
  can be removed with `git worktree remove`; the branches keep their
  commits. Their build directories have been cleaned.
