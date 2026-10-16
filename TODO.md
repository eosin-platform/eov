# TODO after the 1.0 improvement pass

What is left of the work on [AUDIT.md](AUDIT.md). The audit's status section
lists every finding and §14 feature; regenerate it after a change with
`python3 tools/audit_status/gen_status.py` (edit `status_data.py` or
`features_data.py` next to it first). Nothing below has been pushed.

## Unmerged branches

Work that was stopped before it was finished. Each branch is in a git
worktree under `.claude/worktrees/`; its last commit is marked `WIP` when it
holds unfinished, untested changes.

### HDF5 dataset packing: `worktree-agent-a340c90225cbf0327`

- **Goal:** `eov dataset pack <DIR> --format hdf5` writes `patches.h5`, with
  the same arrays and attributes as `patches.zarr`.
- **Design:** the HDF5 library is loaded at run time (libloading), so eov
  has no build-time HDF5 dependency.
- **Done:**
  - `3947e01`: shared patch-record decoding between pack formats.
  - `7de56f1`: the HDF5 writer.
- **WIP:** `c6fbf40`, CLAM-style per-slide coordinate files
  (`--format clam`).
- **Left:**
  - Finish or drop the CLAM part.
  - Run the tests; libhdf5 1.10 is installed on the dev machine, and the
    tests skip themselves where it is missing.
  - Review, merge, and update the README's Dataset section.

### Model store: `worktree-agent-afa8da3ca208fa577`

Downloading and verifying models for AI plugins (AUDIT §14.5, §12.2).

- **Done**, 5 commits:
  - a content-addressed store under `<data>/models/`;
  - the `eov model fetch|list|verify|remove|import` commands;
  - the plugin API's `model.fetch`, `model.resolve` and `model.list`;
  - policy keys;
  - eovae downloading the HistoVAE reference model.
- **WIP:** the cellaris side is in that worktree's own clone,
  `plugins/cellaris`, on branch `models` (commit `6bee97f`).
- **Left:**
  - eovae tests for the download flow.
  - Finish cellaris's "Download model…".
  - Review, and merge both the superproject and the cellaris branch, then
    record the submodule pointer.
  - There is no verified public URL for either model, so sites configure a
    URL and SHA-256 themselves.

### Slide-source plugin type: `worktree-agent-a82d87b0b13804a70`

Only an API sketch (`b225f73`) exists. This mechanism would let plugins add
formats the host cannot read (iSyntax, VSI, LIF, DICOMweb or OMERO), keeping
the host offline (AUDIT §14.3). Design:

- **Entry point:** an optional field appended to the root module
  `PluginModule` after its last prefix field (abi_stable prefix type). It
  returns a new `#[sabi_trait]` source object. The frozen `PluginFfi` and
  `HostFfi` vtables are untouched, so 1.0 plugins still load; add a test
  that proves it.
- **Handle:** opens a path or URI. It gives info as JSON (levels,
  downsamples, mpp, objective, properties, associated images), reads
  regions as RGBA `Pixels`, and reads associated images.
  - It must be Send + Sync, because render workers read concurrently.
  - The plugin is never called with host locks held.
  - A panic quarantines the plugin and fails the read.
- **Manifest:** declares the extensions and URI schemes so the host knows
  them without loading code.
- **Precedence:** built-in readers win for formats they read.
- **Discovery:** the file picker, series bar, drag and drop, command line,
  and dataset and batch discovery accept registered extensions. CLI commands
  that do not load plugins refuse those formats with a clear message.
- **Tests:** an in-tree synthetic format.

### QPTIFF reader: `wip/qptiff` (in the main repository)

Reads Akoya multiplex IF and brightfield QPTIFF. The OME-TIFF reader builds
an OME document from the PerkinElmer-QPI descriptions (`7ba85ea`).

- **Left:**
  - A QPTIFF mode in `common/src/test_support/ome.rs`: a QPI description
    per IFD, resolution tags (it needs a RATIONAL value), and
    reduced-resolution IFDs.
  - Tests for channels, colours, pyramid, mpp, objective, associated
    images, brightfield RGB, and discovery not skipping a `.tif` QPTIFF.
  - The `qptiff` entry in `app/src/file_types`. Regenerate the Linux files
    with `cargo test -p eov --bin eov write_linux_assets -- --ignored` and
    add the `com.akoya.qptiff` UTI to `assets/macos/Info.plist`.
  - README and `app/README.md`, and the `eov dataset` help text in
    `app/src/cli.rs`.

## Open features

- **Focal plane / time point control in the HUD.** The audit asked for a
  focus slider; today only the keys exist (Ctrl+↑/↓ and Ctrl+→/←).
  - Add `HudSettings` fields, e.g. "Z 2/3" and "T 1/5", filled in
    `ui_update.rs`.
  - Add small step controls in `hud-toolbar.slint`, with accessible labels
    and theme tokens.
- **SH-10:** Copy View cannot put images on the clipboard in the Flatpak
  under Wayland. Either enable arboard's `wayland-data-control` feature
  (its crates were not available offline) or give the Flatpak X11 access.
- **Vendor Z-stacks** (NDPI, VMS, multi-focal DICOM) show one plane:
  OpenSlide exposes only one.
- **Readers for iSyntax, Olympus/Evident VSI, Leica LIF and CZI
  fluorescence**, and remote slides (DICOMweb, OMERO). These would come as
  slide-source plugins once that type lands.
- **DICOM SR** is neither read nor written. DICOM Microscopy Bulk Simple
  Annotations are.
- **Localisation:**
  - No translation catalogs ship.
  - Texts made in Rust stay English: status messages, menus built in Rust,
    the theme menu and the plugin usage line.
  - No `.pot` template ships, because slint-tr-extractor was not available
    offline.
- **Themes:**
  - In the light theme, the area around the slide stays dark grey (the
    renderer draws it).
  - Some dark-theme secondary texts are below 4.5:1 contrast.
  - Panels provided by plugins are not themed or translated.
- **Plugins panel:** a grey strip showed below the panel before the
  accessibility branch was merged. Check whether it is still there.
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

- **Unpushed commits:** master, and the submodules annotations (master),
  cellaris (main) and gamepad (master), are ahead of their remotes.

  Push the submodules before the superproject that points at them.
- **Leftovers in the home directory.** An early test run, before tests were
  isolated, wrote:
  - `[active_sidebar]` in `~/.eov/config.toml`;
  - `~/.config/eov/eovae.json`;
  - about 25 fixture directories in `~/.cache/eov/plugin-packages`.

  Review them and remove them if they are unwanted.
- **Worktrees:** `.claude/worktrees/` holds the agents' worktrees. Remove the
  merged ones with `git worktree remove`; the branches keep their commits.
  Their build directories have been cleaned.
