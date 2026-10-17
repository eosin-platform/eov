# TODO after the 1.0 improvement pass

What is left of the work on [AUDIT.md](AUDIT.md). The audit's status section
lists every finding and §14 feature; regenerate it after a change with
`python3 tools/audit_status/gen_status.py` (edit `status_data.py` or
`features_data.py` next to it first). Nothing below has been pushed.

## Unmerged branches

Work that was stopped before it was finished. Each branch is in a git
worktree under `.claude/worktrees/`; its last commit is marked `WIP` when it
holds unfinished, untested changes.

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

## Open features

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
- **Worktrees:** `.claude/worktrees/` holds the agents' worktrees; all but
  the slide-source one (`worktree-agent-a82d87b0b13804a70`) are merged.
  Remove them with `git worktree remove`; the branches keep their commits.
  Their build directories have been cleaned.
