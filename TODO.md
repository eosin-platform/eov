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
- **DICOM SR** is neither read nor written. DICOM Microscopy Bulk Simple
  Annotations are.
- **Localisation** (31 languages; see docs/translating.md):
  - The 30 translations were drafted with machine translation, then
    reviewed by independent AI reviewers with a back-translation check of
    the safety texts; no native speaker or pathologist has reviewed them.
    Ask for reviews (start with the languages most used in the community)
    and record them in each catalog's `X-Review-Status` and in
    docs/translating.md.
  - Layouts are not mirrored for Arabic, Persian and Hebrew: Slint 1.19 has
    no layout direction. Text starts at the right and inserted values are
    isolated, but controls keep their left-to-right places: the order of
    controls in a row, the side of a check box's box and of a list's arrow,
    label and value columns (a right-to-left reader meets some values
    before their labels), and the keys column of menus, which stays at the
    right. `isolate()` acts only in right-to-left interfaces, so
    right-to-left data inside a left-to-right interface is not isolated.
    docs/translating.md lists choices the right-to-left catalogs make
    differently (percent signs, name + version, formulas) for native
    reviewers to settle.
  - CJK fonts: Slint 1.19 passes no locale to font fallback, so eov puts
    the interface language's CJK font first in Slint's fallback lists
    (app/src/fonts.rs: Noto Sans CJK, Source Han Sans and others on Linux),
    which gives Japanese, Chinese and Korean their own glyph forms and
    punctuation when such a font is installed; otherwise the system's
    fallback order decides. The Windows (Yu Gothic UI, Microsoft YaHei UI,
    Microsoft JhengHei UI, Malgun Gothic) and macOS (Hiragino Sans, PingFang,
    Apple SD Gothic Neo) candidates are untested. Text that is not laid out
    again after a language switch keeps its previous glyphs until it
    changes.
  - Line breaking is Slint's: Korean breaks between syllables (as CSS
    `word-break: normal`), not only at spaces, and Thai is split with the
    ICU4X dictionary inside parley, which splits some words wrongly
    (การกระ|จาย) and allows a break at the end of every Thai run, so a
    closing parenthesis can wrap onto a line of its own (the Brush entry of
    the Keyboard Shortcuts panel). Neither can be set from Slint 1.19. For
    Korean, inserting U+2060 word joiners between syllables when the
    catalogs are compiled worked in a probe but is not applied; in Thai a
    word joiner does not stop the break at the end of a run (checked on the
    Brush entry). The proper fix is upstream.
  - Hindi text is at least 12 px (`Locale.text-size` in app/ui/locale.slint,
    `text-size` in each first-party plugin's surfaces): below that the GPU
    renderer (femtovg 0.28) draws Devanagari with overlapping and missing
    pieces at a scale factor of 1, which the software renderer does not.
    Report it upstream; the minimum can go once it is fixed.
  - Plugin popups (the annotations ⋯ menu and lists) close on a click in
    their panel and on a language change, but not on a click elsewhere in
    the window: a surface learns nothing of clicks outside it. A host
    property such as `eov-pointer-down-outside` would let surfaces close
    them. Plugin modals have a fixed size, so the annotations import
    dialog estimates its height from its translated texts
    (src/import_fit.rs); letting modals take their preferred height would
    replace that.
  - Native file dialogs, and the Linux desktop files (menu entry, MIME type
    names, AppStream metainfo — all translated from the catalogs), follow
    the system's language, not the one chosen in eov.
  - File type names on Windows (registry `FriendlyTypeName`, needs a
    STRINGTABLE resource per language) and macOS (`InfoPlist.strings` per
    `.lproj`) are English; both could reuse the catalogs' "File type" texts.
  - Text burned into exported images and copied views is English with `.`
    decimals: the embedded font has basic Latin and Greek letters only.
    User-written labels in other scripts are left blank (the export dialog
    says which letters). Extending the font subset (Romanian ș/ț, Vietnamese,
    Greek tonos letters) needs fontTools: see common/src/overlay/font.rs.
  - Command-line output, `--help`, logs and the audit log stay English by
    design. The update helper's own failure dialog (Windows, macOS) is
    English: it is a separate program without catalogs.
  - The intended-use statement keeps "not for primary diagnosis" (regulatory
    wording); translators were told it means eov must not be used to make a
    diagnosis. Decide whether the English should say "not for diagnosis".
- **Exports ignore a custom stain target:** with stain normalisation on,
  panes use the target chosen in the Stain Target menu, but exports always
  normalise to the reference H&E (`export_dialog.rs` `build_request` never
  sets `color.target`), although the Stain Target notice says exports
  follow. Found during localisation; not fixed.
- **Found while testing the localisation, not caused by it (not fixed; the
  build from before the localisation behaves the same):**
  - The tab context menu stops opening. After a plugin's sidebar was closed
    with its own toolbar button, or Slide Information, the Messages panel,
    Save a Copy, Dataset export or the Plugins panel was closed with
    Escape, right-clicking the tab shows nothing, and the next left click
    on the tab closes the slide, as if an invisible menu's Close item took
    the click. Opening and closing the slide's own context menu makes the
    tab menu work once more. The cause was not found.
  - The "Region of interest" hover tooltip stays on screen when the pointer
    leaves the viewport straight from an ROI; after Ctrl+W it is still
    drawn over the home screen.
  - In a narrow pane (an 800-px window with a sidebar open) the slide's HUD
    toolbar keeps its full width and is cut at the window's edge: its last
    buttons (grid, effects, image adjustments) cannot be reached.
  - The Keyboard Shortcuts panel repeats a category's heading whenever
    consecutive actions alternate between categories.
  - With the Cellaris overlay on, its "PREDICTED" banner is drawn over the
    right end of the slide's HUD toolbar and hides the effects and image
    adjustment buttons (clicks still reach them).
  - The gamepad settings window cannot scroll its last 14 px, so the
    bottom edge of the Mappings card is cut in every language.
- **Themes:** panels provided by plugins are not themed.
- **Signed binaries:** the owner has to provide the release signing key,
  Authenticode and notarization certificates.

## Not verified

- **Platforms:** Windows and macOS builds and behaviour, including the
  system language detection, the CJK font choice and the language of
  native file dialogs there. The GUI was only run under X11 in nested
  Xephyr servers; it was not run live on Wayland.
- **Screen reader:** no real one (Orca) was run. Accessibility was checked
  through Slint's testing backend. eov is built without Slint's
  `accessibility` feature (app/Cargo.toml; Cargo.lock has no AccessKit), so
  no screen reader can see the roles and names at all, although README.md
  says every control has one for screen readers: enable the feature (and
  test with Orca, NVDA and VoiceOver) or correct the README.
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

- **Plugins need an eov release with plugin API 1.1.** Localisation moved
  eov-plugin-api to 1.1.0, and the first-party plugins now require
  `version = "1.1"`. Their CI builds against a pinned eov checkout: gamepad's
  `[package.metadata.eov] ref = "v1.0.0"` (and the `eov_ref` inputs of the
  annotations and cellaris workflows) point at plugin API 1.0, so plugin CI
  fails until an eov tag with plugin API 1.1 exists and the refs move to it.
- **Localisation commits:** a snapshot of the superproject's localisation
  work was committed as `d65aa7c` ("localization"); the later rounds
  (fixes, catalogs, Linux desktop files) are uncommitted. In the
  annotations, cellaris and gamepad submodules none of the localisation is
  committed yet: commit there first, then the superproject's submodule
  pointers.
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
