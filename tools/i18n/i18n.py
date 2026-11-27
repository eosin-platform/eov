#!/usr/bin/env python3
"""Translation catalogs of eov and its in-tree plugins.

    python3 tools/i18n/i18n.py extract            # regenerate the templates (.pot)
    python3 tools/i18n/i18n.py update             # extract, then merge every catalog
    python3 tools/i18n/i18n.py check [--strict]   # validate catalogs (and templates)
    python3 tools/i18n/i18n.py stats              # translated / fuzzy / untranslated
    python3 tools/i18n/i18n.py export --language de out.json   # what is left to translate
    python3 tools/i18n/i18n.py import --language de in.json    # write translations back
    python3 tools/i18n/i18n.py pseudo DIR         # pseudo-translations for layout checks

Every command takes component names (eov, example_plugin, annotations,
cellaris, eovae, gamepad) or `--plugin DIR` for a plugin elsewhere; without
any it works on all components that are present (submodules that are not
checked out are skipped).

A plugin's texts are taken from plugin.toml, from every .slint file in its
folder (except in target/, hidden folders, its catalogs and the tests/,
benches/ and examples/ folders of its crates) and from the module tree of
its crate root (`[lib] path` in its Cargo.toml, else src/lib.rs). Rust files
that call `tr!` outside that tree, such as those of another crate, are
reported: their texts are not extracted.

Extraction uses the standard tools: Slint's `slint-tr-extractor` for `@tr`
in .slint files, the tr crate's `xtr` for `tr!` in Rust, and GNU gettext
(`msgcat`, `msgmerge`, `msgattrib`, `msgfmt`) for everything else:

    cargo install slint-tr-extractor --version 1.19.0 --locked
    cargo install xtr --locked
    sudo apt install gettext        # or: brew install gettext

`update`, `check`, `export`, `import` and `pseudo` also need polib (`pip
install polib`, or the `python3-polib` package). See docs/translating.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES_FILE = Path(__file__).resolve().parent / "languages.json"
ISSUES_URL = "https://github.com/eosin-platform/eov/issues"
# The context of manifest texts in plugin catalogs (the plugin API's
# MANIFEST_TRANSLATION_CONTEXT).
MANIFEST_CONTEXT = "plugin.toml"
# Recorded in every catalog; see docs/translating.md.
DEFAULT_REVIEW_STATUS = "machine-translated; not yet reviewed by a native speaker"
PLACEHOLDER = re.compile(r"\{\{|\}\}|\{[^{}]*\}")

# Plugins in this repository (and its submodules), by directory.
IN_TREE_PLUGINS = [
    "plugin_api/example-plugin",
    "plugins/annotations",
    "plugins/cellaris",
    "plugins/eovae",
    "plugins/gamepad",
]


@dataclass
class Language:
    tag: str
    native_name: str
    english_name: str
    rtl: bool
    plural_forms: str
    decimal_separator: str

    @property
    def plural_count(self) -> int:
        match = re.search(r"nplurals\s*=\s*(\d+)", self.plural_forms)
        return int(match.group(1)) if match else 2


def languages() -> list[Language]:
    """The translations (every language but English), from languages.json,
    which a test of the eov crate keeps equal to the plugin API's table."""
    data = json.loads(LANGUAGES_FILE.read_text(encoding="utf-8"))
    return [Language(**entry) for entry in data if entry["tag"] != "en"]


def plural_examples(language: Language) -> dict[int, list[int]]:
    """Numbers that take each plural form (with gettext's own evaluator)."""
    import gettext

    expression = re.search(r"plural\s*=\s*([^;]+)", language.plural_forms).group(1)
    rule = gettext.c2py(expression.strip())
    examples: dict[int, list[int]] = {form: [] for form in range(language.plural_count)}
    for n in list(range(0, 30)) + [100, 101, 102, 105, 111, 1000, 1001]:
        form = rule(n)
        if len(examples.setdefault(form, [])) < 8:
            examples[form].append(n)
    return examples


def catalog_folders(tag: str) -> list[str]:
    """The folder names of a language's catalogs, in the order eov looks for
    them (eov_plugin_api::i18n::Language::catalog_folders)."""
    folders = [tag]
    if tag.replace("-", "_") != tag:
        folders.append(tag.replace("-", "_"))
    folders += {"zh-Hans": ["zh_CN"], "zh-Hant": ["zh_TW"], "nb": ["nb_NO", "no"]}.get(tag, [])
    return folders


@dataclass
class Component:
    name: str
    root: Path  # the directory commands run in, and references are relative to
    catalogs: Path  # <catalogs>/<tag>/LC_MESSAGES/<domain>.po
    domain: str
    slint: list[Path] = field(default_factory=list)
    rust: list[Path] = field(default_factory=list)
    manifest: Path | None = None
    bugs: str = ISSUES_URL

    @property
    def template(self) -> Path:
        return self.catalogs / f"{self.domain}.pot"

    def catalog(self, tag: str) -> Path:
        """The catalog of a language: in the first folder that has one, as
        the plugin API's find_catalog looks (`pt-BR`, `pt_BR`, `zh_TW`, …),
        else where a new one belongs (the tag's folder)."""
        for folder in catalog_folders(tag):
            path = self.catalogs / folder / "LC_MESSAGES" / f"{self.domain}.po"
            if path.exists():
                return path
        return self.catalogs / tag / "LC_MESSAGES" / f"{self.domain}.po"

    def catalogs_of(self, tag: str) -> list[Path]:
        """Every catalog of a language, under any spelling (more than one is
        a mistake `check` reports)."""
        return [
            path
            for folder in catalog_folders(tag)
            if (path := self.catalogs / folder / "LC_MESSAGES" / f"{self.domain}.po").exists()
        ]


def eov_component() -> Component:
    return Component(
        name="eov",
        root=ROOT,
        catalogs=ROOT / "app" / "translations",
        domain="eov",
        slint=sorted((ROOT / "app" / "ui").glob("*.slint")),
        rust=[ROOT / "app" / "src" / "main.rs"],
    )


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# Folders of a crate (next to its Cargo.toml) that hold other targets than
# the library: their files are not part of a plugin.
CARGO_TARGET_FOLDERS = {"tests", "benches", "examples"}


def plugin_files(directory: Path, suffix: str, catalogs: Path) -> list[Path]:
    """The plugin's files with this suffix: everywhere in its folder but
    target/, hidden folders, its catalogs and the tests/, benches/ and
    examples/ folders of its crates."""
    skipped = catalogs.resolve()
    found = []
    for folder, subfolders, files in os.walk(directory):
        here = Path(folder)
        crate = (here / "Cargo.toml").is_file()
        subfolders[:] = [
            name
            for name in subfolders
            if not name.startswith(".")
            and name != "target"
            and not (crate and name in CARGO_TARGET_FOLDERS)
            and (here / name).resolve() != skipped
        ]
        found += [here / name for name in files if name.endswith(suffix)]
    return sorted(found)


def crate_root(directory: Path) -> Path | None:
    """The root module of the plugin's crate: `[lib] path` in its
    Cargo.toml, else src/lib.rs; None when there is none in its folder."""
    path = "src/lib.rs"
    cargo = directory / "Cargo.toml"
    if cargo.is_file():
        try:
            library = tomllib.loads(cargo.read_text(encoding="utf-8")).get("lib", {})
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            warn(f"{cargo}: cannot read it ({error}); looking for src/lib.rs")
            library = {}
        if isinstance(library, dict) and isinstance(library.get("path"), str):
            path = library["path"]
    root = directory / path
    if not root.is_file():
        return None
    if not root.resolve().is_relative_to(directory.resolve()):
        warn(f"{directory}: the crate root {path} is outside the plugin's folder; its texts are not extracted")
        return None
    return root


def plugin_component(directory: Path) -> Component | None:
    manifest_path = directory / "plugin.toml"
    if not manifest_path.is_file():
        return None
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    i18n = manifest.get("i18n")
    if i18n is None:
        return None
    domain = i18n.get("domain", manifest["id"])
    repository = manifest.get("repository")
    catalogs = directory / i18n.get("catalogs", "translations")
    root = crate_root(directory)
    return Component(
        name=manifest["id"],
        root=directory,
        catalogs=catalogs,
        domain=domain,
        slint=plugin_files(directory, ".slint", catalogs),
        rust=[root] if root is not None else [],
        manifest=manifest_path,
        bugs=f"{repository.rstrip('/')}/issues" if repository else ISSUES_URL,
    )


# A `tr!` call with a text (`tr!("…"`, `i18n::tr!(r"…"`), as xtr finds them.
TR_CALL = re.compile(r'(?<![\w])tr!\s*[(\[{]\s*(?:r#*)?"')


def calls_tr(path: Path) -> bool:
    """Whether a Rust file calls `tr!` with a text, outside `//` comment
    lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    code = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("//"))
    return TR_CALL.search(code) is not None


def referenced_files(pot: Path, root: Path) -> set[Path]:
    """The files the `#:` references of a template name."""
    found = set()
    for line in pot.read_text(encoding="utf-8").split("\n"):
        if line.startswith("#:"):
            for reference in line[2:].split():
                path, _, number = reference.rpartition(":")
                found.add((root / (path if number.isdigit() else reference)).resolve())
    return found


def report_unextracted_rust(component: Component, extracted: set[Path]) -> None:
    """Warns about a plugin's Rust files that call `tr!` but gave no text:
    they are not in the module tree of its crate root (another crate's, a
    binary's), so their texts stay English."""
    missed = [
        path
        for path in plugin_files(component.root, ".rs", component.catalogs)
        if path.resolve() not in extracted and calls_tr(path)
    ]
    if not missed:
        return
    files = ", ".join(relative(missed, component.root))
    if component.rust:
        warn(
            f"{component.name}: these files call tr! outside the module tree of "
            f"{relative(component.rust, component.root)[0]}, so their texts are not extracted "
            f"and stay English (texts of other crates are never extracted): {files}"
        )
    else:
        warn(
            f"{component.name}: no crate root found (neither `[lib] path` in Cargo.toml nor "
            f"src/lib.rs), so no tr! text is extracted; these files call tr!: {files}"
        )


def components(names: list[str], plugin_dirs: list[str]) -> list[Component]:
    found = [eov_component()]
    for directory in IN_TREE_PLUGINS:
        component = plugin_component(ROOT / directory)
        if component is not None:
            found.append(component)
    for directory in plugin_dirs:
        component = plugin_component(Path(directory).resolve())
        if component is None:
            sys.exit(f"{directory}: no plugin.toml with an [i18n] table")
        found.append(component)
    if not names:
        return found if not plugin_dirs else found[-len(plugin_dirs):]
    by_name = {component.name: component for component in found}
    missing = [name for name in names if name not in by_name]
    if missing:
        sys.exit(f"unknown or absent components: {', '.join(missing)} (known: {', '.join(by_name)})")
    return [by_name[name] for name in names]


def run(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command, cwd=cwd, check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        sys.exit(f"{command[0]} is not installed (see the top of {Path(__file__).name})")
    except subprocess.CalledProcessError as error:
        sys.exit(f"{' '.join(command)} failed:\n{error.stderr}")
    return result.stdout


def relative(paths: list[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]


def header(component: Component, language: Language | None) -> str:
    fields = [
        f"Project-Id-Version: {component.name}",
        f"Report-Msgid-Bugs-To: {component.bugs}",
    ]
    if language is not None:
        fields += [
            f"Language: {language.tag}",
            f"Language-Team: {language.english_name}",
        ]
    fields += [
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=UTF-8",
        "Content-Transfer-Encoding: 8bit",
    ]
    if language is not None:
        fields += [
            f"Plural-Forms: {language.plural_forms}",
            f"X-Review-Status: {DEFAULT_REVIEW_STATUS}",
        ]
    lines = ['msgid ""', 'msgstr ""'] + [f'"{field}\\n"' for field in fields]
    return "\n".join(lines) + "\n"


def po_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'


def manifest_template(component: Component) -> str:
    """Entries for the manifest's texts: name, description, format names."""
    assert component.manifest is not None
    manifest = tomllib.loads(component.manifest.read_text(encoding="utf-8"))
    texts = [manifest.get("name"), manifest.get("description")]
    texts += [fmt.get("name") for fmt in manifest.get("slide_formats", [])]
    reference = str(component.manifest.relative_to(component.root))
    entries = []
    for text in dict.fromkeys(text for text in texts if text):
        entries.append(
            f"#: {reference}\nmsgctxt {po_string(MANIFEST_CONTEXT)}\nmsgid {po_string(text)}\nmsgstr \"\"\n"
        )
    return "\n".join(entries)


def extract(component: Component) -> bool:
    """Regenerates the component's template; returns whether it changed."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        parts = []
        if component.slint:
            target = scratch / "slint.pot"
            run(
                ["slint-tr-extractor", "-o", str(target), "--package-name", component.name]
                + relative(component.slint, component.root),
                cwd=component.root,
            )
            repair_comments(target)
            parts.append(target)
        extracted: set[Path] = set()
        if component.rust:
            target = scratch / "rust.pot"
            run(
                ["xtr", "-o", str(target), "--package-name", component.name, "--omit-header"]
                + relative(component.rust, component.root),
                cwd=component.root,
            )
            if target.exists():
                extracted = referenced_files(target, component.root)
            if target.exists() and target.stat().st_size > 0:
                # xtr writes no header with --omit-header; give it one so
                # msgcat accepts its UTF-8.
                target.write_text(header(component, None) + "\n" + target.read_text(encoding="utf-8"), encoding="utf-8")
                parts.append(target)
        if component.manifest is not None:
            report_unextracted_rust(component, extracted)
            target = scratch / "manifest.pot"
            target.write_text(header(component, None) + "\n" + manifest_template(component), encoding="utf-8")
            parts.append(target)
        merged = scratch / "merged.pot"
        if parts:
            # A text that several parts have (a .slint file and Rust code
            # with the same context and text) keeps the references and
            # comments of each: app/build.rs leaves texts without a `.rs`
            # reference out of the Rust tables. (`--use-first` would keep
            # only the first part's.)
            run(["msgcat", "--sort-by-file", "-o", str(merged)] + [str(p) for p in parts])
        else:
            merged.write_text("", encoding="utf-8")
        body = strip_header(merged.read_text(encoding="utf-8")) if merged.exists() else ""
        body = without_part_markers(body)
        text = header(component, None) + ("\n" + body if body else "")
    component.catalogs.mkdir(parents=True, exist_ok=True)
    old = component.template.read_text(encoding="utf-8") if component.template.exists() else None
    if old == text:
        return False
    component.template.write_text(text, encoding="utf-8")
    return True


# msgcat marks the comments of a text that several parts have, where they
# differ, with the part each comes from: `#. #-#-#-#-#  rust.pot (eov)  #-#-#-#-#`.
PART_MARKER = re.compile(r"^#\.? ?#-#-#-#-#  .*  #-#-#-#-#$")


def without_part_markers(body: str) -> str:
    """The entries msgcat merged, without the lines that say which part a
    comment comes from: the comments of the parts follow one another (the
    .slint files' first). Anything else msgcat had to mark (texts that differ
    in more than their comments) stops the extraction."""
    kept = [line for line in body.split("\n") if not PART_MARKER.match(line)]
    marked = [line for line in kept if "#-#-#-#-#" in line]
    if marked:
        sys.exit("msgcat could not merge the extracted texts:\n" + "\n".join(marked))
    return "\n".join(kept)


# A reference line as slint-tr-extractor writes them: `#: file:line …`.
REFERENCE_LINE = re.compile(r"^#: \S+:\d+( \S+:\d+)*$")


def repair_comments(path: Path) -> None:
    """slint-tr-extractor 1.19 writes a translator comment longer than a
    line without its `#.` prefix (rspolib's writer), in pieces that may
    start with anything (`#3498DB`, `"`, `msgid `); put the prefix back.

    Each of its entries is the comment (one line that starts with `#. `, or
    pieces without a prefix), then the references (always there), then the
    keywords. So in a block that has references, every line before them
    belongs to the comment; the header block and everything from the
    references on stay as they are."""
    blocks = path.read_text(encoding="utf-8").split("\n\n")
    for index, block in enumerate(blocks):
        lines = block.split("\n")
        start = next((i for i, line in enumerate(lines) if REFERENCE_LINE.match(line.rstrip())), None)
        if not start or (start == 1 and lines[0].startswith("#. ")):
            continue
        lines[:start] = [f"#. {line.rstrip()}" for line in lines[:start]]
        blocks[index] = "\n".join(lines)
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def strip_header(text: str) -> str:
    """The entries of a .po/.pot text, without its header entry (the one
    whose `msgid ""` is directly followed by `msgstr`)."""
    blocks = re.split(r"\n[ \t]*\n", text.strip("\n"))
    for index, block in enumerate(blocks):
        if re.search(r'(?m)^msgid ""\nmsgstr ', block):
            del blocks[index]
            break
    return "\n\n".join(blocks).strip("\n") + "\n" if blocks else ""


def update(component: Component) -> None:
    extract(component)
    template = component.template
    for language in languages():
        catalog = component.catalog(language.tag)
        if not catalog.exists():
            # A new catalog: just this language's header. msgmerge adds every
            # text of the template, with as many plural forms as the header
            # says.
            catalog.parent.mkdir(parents=True, exist_ok=True)
            catalog.write_text(header(component, language), encoding="utf-8")
        run(["msgmerge", "--quiet", "--update", "--backup=none", "--previous", str(catalog), str(template)])
        run(["msgattrib", "--no-obsolete", "-o", str(catalog), str(catalog)])
        set_header_fields(catalog, component, language)


def set_header_fields(catalog: Path, component: Component, language: Language) -> None:
    """Keeps the header's language, plural forms and review status right
    (msgmerge leaves them as they are; a new catalog gets them from
    `header`)."""
    file = polib().pofile(str(catalog), wrapwidth=0)
    metadata = file.metadata
    before = dict(metadata)
    metadata["Project-Id-Version"] = component.name
    metadata["Report-Msgid-Bugs-To"] = component.bugs
    metadata["Language"] = language.tag
    metadata.setdefault("Language-Team", language.english_name)
    metadata["Plural-Forms"] = language.plural_forms
    metadata.setdefault("X-Review-Status", DEFAULT_REVIEW_STATUS)
    metadata.pop("POT-Creation-Date", None)
    # Untranslated plural texts get as many (empty) forms as the language
    # has: msgmerge keeps the two of the template on texts it already knew.
    repaired = False
    for entry in file:
        if entry.msgid_plural and not entry.translated() and len(entry.msgstr_plural) != language.plural_count:
            entry.msgstr_plural = {form: "" for form in range(language.plural_count)}
            repaired = True
    if dict(metadata) != before or repaired:
        file.save(str(catalog))
        normalize(catalog)


def normalize(catalog: Path) -> None:
    """gettext's own formatting, so that later merges change nothing."""
    run(["msgcat", "-o", str(catalog), str(catalog)])


def msgfmt_statistics(catalog: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        ["msgfmt", "--check", "--statistics", "-o", "/dev/null", str(catalog)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    numbers = {"translated": 0, "fuzzy": 0, "untranslated": 0}
    for count, kind in re.findall(r"(\d+) (translated|fuzzy|untranslated)", result.stderr):
        numbers[kind] = int(count)
    return numbers["translated"], numbers["fuzzy"], numbers["untranslated"]


def notes(path: Path) -> dict[tuple[str, str, str], tuple[str, tuple[str, ...]]]:
    """The note for translators (`#.`, however its lines are wrapped) and
    the files that use the text (the `#:` references without their line
    numbers, which move with every edit) of every live entry, by (context,
    msgid, plural)."""
    return {
        (entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""): (
            " ".join((entry.comment or "").split()),
            tuple(sorted({file for file, _ in entry.occurrences})),
        )
        for entry in polib().pofile(str(path), wrapwidth=0)
        if not entry.obsolete
    }


def differences(have: dict, want: dict) -> str:
    """What differs between the notes of two copies of a catalog: their
    texts, else the notes for translators or the files that use them."""
    if have.keys() != want.keys():
        return "texts"
    if any(have[key][0] != want[key][0] for key in want):
        return "notes for translators"
    return "references (where its texts are used)"


# The numbers at which two plural rules are compared, as eov compares them:
# gettext rules tell numbers apart by their last two or three digits and
# at small numbers.
PLURAL_SAMPLES = list(range(0, 10_001)) + [
    100_000, 1_000_000, 1_000_001, 1_000_002, 1_000_005, 1_000_011, 1_000_021, 1_000_100, 10_000_000, 2**32 - 1,
]


def plural_rule(forms: str):
    """The number of forms and the form function (with gettext's own
    evaluator) of a `Plural-Forms` value; None when it cannot be read."""
    import gettext

    fields = {}
    for part in forms.split(";"):
        key, equals, value = part.partition("=")
        if equals:
            fields[key.strip()] = value.strip()
    try:
        count = int(fields["nplurals"])
        rule = gettext.c2py(fields["plural"])
    except (KeyError, ValueError, SyntaxError, RecursionError):
        return None
    return count, rule


def same_plural_rule(forms: str, language: Language) -> bool:
    """Whether a `Plural-Forms` value is the language's rule, in any
    spelling: as many forms, and the same form for every number."""
    if "".join(forms.split()) == "".join(language.plural_forms.split()):
        return True
    theirs, ours = plural_rule(forms), plural_rule(language.plural_forms)
    if theirs is None or ours is None or theirs[0] != ours[0]:
        return False
    last = ours[0] - 1
    try:
        return all(min(theirs[1](n), last) == min(ours[1](n), last) for n in PLURAL_SAMPLES)
    except ArithmeticError:
        return False


def selected_languages(tags: list[str] | None) -> list[Language]:
    every = languages()
    if not tags:
        return every
    unknown = [tag for tag in tags if tag not in {l.tag for l in every}]
    if unknown:
        sys.exit(f"unknown languages: {', '.join(unknown)}")
    return [l for l in every if l.tag in tags]


def check(component: Component, strict: bool, complete: bool, tags: list[str] | None = None) -> list[str]:
    problems = []
    if strict:
        with tempfile.TemporaryDirectory() as scratch:
            copy = Component(**{**component.__dict__, "catalogs": Path(scratch)})
            extract(copy)
            if not component.template.exists():
                problems.append(f"{component.template}: missing; run `update`")
            else:
                fresh, committed = notes(copy.template), notes(component.template)
                if fresh != committed:
                    problems.append(
                        f"{component.template}: its {differences(committed, fresh)} are out of date with the "
                        f"sources; run `python3 tools/i18n/i18n.py update {component.name}`"
                    )
    template = notes(component.template) if component.template.exists() else {}
    for language in selected_languages(tags):
        spellings = component.catalogs_of(language.tag)
        if len(spellings) > 1:
            problems.append(
                f"{component.name}: {language.tag} has catalogs under several spellings "
                f"({', '.join(str(path.parent.parent.name) for path in spellings)}); eov uses only the first"
            )
        catalog = component.catalog(language.tag)
        if not catalog.exists():
            if component.name == "eov" or complete:
                problems.append(f"{catalog}: missing")
            continue
        try:
            translated, fuzzy, untranslated = msgfmt_statistics(catalog)
        except ValueError as error:
            problems.append(f"{catalog}: {error}")
            continue
        metadata = polib().pofile(str(catalog), wrapwidth=0).metadata
        if metadata.get("Language", "").strip() != language.tag:
            problems.append(f"{catalog}: its header lacks Language: {language.tag}")
        forms = metadata.get("Plural-Forms")
        if forms is None:
            problems.append(f"{catalog}: its header lacks Plural-Forms: {language.plural_forms}")
        elif not same_plural_rule(forms, language):
            problems.append(
                f"{catalog}: Plural-Forms '{forms}' is not the rule of {language.english_name} "
                f"('{language.plural_forms}', in any spelling)"
            )
        if template:
            merged = notes(catalog)
            if merged != template:
                problems.append(
                    f"{catalog}: not merged with the template (its {differences(merged, template)} differ); run `update`"
                )
        if complete and (fuzzy or untranslated):
            problems.append(f"{catalog}: {untranslated} untranslated and {fuzzy} fuzzy messages")
        # The translations themselves, as `import` checks them.
        for entry in polib().pofile(str(catalog), wrapwidth=0):
            if entry.obsolete or "fuzzy" in entry.flags or not entry.translated():
                continue
            translation = (
                [entry.msgstr_plural[i] for i in sorted(entry.msgstr_plural)] if entry.msgid_plural else entry.msgstr
            )
            why = validate_translation(entry, translation, language)
            if why:
                problems.append(f"{catalog}: {entry.msgid!r}: {why}")
    return problems


# ---------------------------------------------------------------------------
# polib-based commands
# ---------------------------------------------------------------------------


def polib():
    try:
        import polib as module
    except ImportError:
        sys.exit("polib is not installed: pip install polib (or the python3-polib package)")
    return module


def placeholders(text: str) -> set[str]:
    """The values a text inserts: the k-th `{}` is `{k}`, as eov's and
    Slint's formatters fill them, so translations may reorder them."""
    found = set()
    anonymous = 0
    for token in PLACEHOLDER.findall(text):
        if token in ("{{", "}}"):
            continue
        if token == "{}":
            token = "{%d}" % anonymous
            anonymous += 1
        found.add(token)
    return found


def entry_id(component: Component, entry) -> str:
    """A message's id in export files: the component and a digest of the
    message's context and texts, so an import never lands on another text,
    even after the catalog changed."""
    identity = "\x04".join([entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""])
    return f"{component.name}#{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"


def stats(component: Component, tags: list[str] | None = None) -> list[tuple[str, int, int, int]]:
    rows = []
    for language in selected_languages(tags):
        catalog = component.catalog(language.tag)
        if not catalog.exists():
            continue
        try:
            rows.append((language.tag, *msgfmt_statistics(catalog)))
        except ValueError:
            rows.append((language.tag, 0, 0, -1))
    return rows


def export(component_list: list[Component], tag: str, everything: bool) -> list[dict]:
    po = polib()
    language = next((l for l in languages() if l.tag == tag), None)
    if language is None:
        sys.exit(f"unknown language {tag}")
    work = []
    examples = {str(form): numbers for form, numbers in plural_examples(language).items()}
    for component in component_list:
        catalog = component.catalog(tag)
        if not catalog.exists():
            continue
        for index, entry in enumerate(po.pofile(str(catalog), wrapwidth=0)):
            if entry.obsolete:
                continue
            if not everything and entry.translated() and "fuzzy" not in entry.flags:
                continue
            work.append(
                {
                    "id": entry_id(component, entry),
                    "context": entry.msgctxt,
                    "msgid": entry.msgid,
                    "msgid_plural": entry.msgid_plural or None,
                    "forms": language.plural_count if entry.msgid_plural else 1,
                    "numbers_per_form": examples if entry.msgid_plural else None,
                    "comment": entry.comment or None,
                    "where": [f"{path}:{line}" if line else path for path, line in entry.occurrences][:3],
                    "current": (
                        [entry.msgstr_plural[i] for i in sorted(entry.msgstr_plural)]
                        if entry.msgid_plural
                        else entry.msgstr
                    )
                    or None,
                }
            )
    return work


# Right-to-left text whose first strong character is a Latin letter is laid
# out from left to right, its words in the wrong order: the first strong
# character sets the direction of a paragraph (UAX #9, P2 and P3). A value
# coming first may be Latin too. A right-to-left mark (U+200F) first makes
# such a text right to left; `import` puts it there, `check` asks for it.
RIGHT_TO_LEFT_MARK = "\u200f"
ISOLATE_STARTS = "\u2066\u2067\u2068"
ISOLATE_END = "\u2069"


# A list of search words (the desktop entry's `Keywords`), which is not
# shown as text: a mark before its first word would stop that word from
# matching. At least three words and no placeholders, so that values joined
# with a semicolon ("{mean}; {dark}") are not taken for one.
WORD_LIST = re.compile(r"[^;\s{}]+(;[^;{}]+){2,};?")

# Values joined into a sentence ("{plugin}: {status}", "Cellaris: {}",
# "{mean}; {dark}", "{path} ({source}): {what}"), or a line that ends a
# sentence after a value when another line follows ("{reason}.\n{hint}").
# Even without letters of the language such a line has to be laid out right
# to left, or a right-to-left reader meets its parts in reverse order once
# eov isolates the values. A single value ("{} × {} px", "{}%",
# "{hours}:{minutes}") is left alone.
JOINED_VALUES = re.compile(r"[})\]\w](?:[:;\u060c\u061b]| \u00b7) ")
SENTENCE_END = re.compile(r"[})\]\w]\.$")


def line_starts_left_to_right(line: str, joined: bool) -> bool:
    """Whether one line, which Slint lays out as a paragraph of its own,
    would be laid out from left to right: a Latin letter or a placeholder
    comes before its first right-to-left letter. Isolated runs are skipped,
    as UAX #9 skips them. A line without right-to-left letters counts only
    when it joins values (`joined`)."""
    if not joined and not any(unicodedata.bidirectional(c) in ("R", "AL") for c in line):
        return False
    depth = 0
    index = 0
    while index < len(line):
        c = line[index]
        if c in ISOLATE_STARTS:
            depth += 1
        elif c == ISOLATE_END:
            depth = max(0, depth - 1)
        elif depth == 0:
            token = PLACEHOLDER.match(line, index)
            if token and token.group() not in ("{{", "}}"):
                return True
            kind = unicodedata.bidirectional(c)
            if kind in ("R", "AL"):
                return False
            if kind == "L":
                return True
        index += 1
    return False


def left_to_right_lines(text: str) -> list[int]:
    """The lines (0-based) of a translation into a right-to-left language
    that would be laid out from left to right. Slint splits a text at "\n"
    and gives each line its own direction (from its first strong letter),
    so each line is judged on its own. Lists of search words are not laid
    out and never count."""
    if WORD_LIST.fullmatch(text.lstrip(RIGHT_TO_LEFT_MARK)):
        return []
    lines = text.split("\n")
    return [
        index
        for index, line in enumerate(lines)
        if line_starts_left_to_right(
            line,
            bool(JOINED_VALUES.search(line)) or (index + 1 < len(lines) and bool(SENTENCE_END.search(line))),
        )
    ]


def starts_left_to_right(text: str) -> bool:
    """Whether any line of a translation into a right-to-left language
    would be laid out from left to right (see left_to_right_lines)."""
    return bool(left_to_right_lines(text))


# Explicit embeddings and overrides (LRE, RLE, PDF, LRO, RLO): isolates do
# their job without leaking into the text around them.
EMBEDDINGS = "\u202a\u202b\u202c\u202d\u202e"


def unbalanced_isolates(text: str) -> bool:
    """Whether an isolate (U+2066-U+2068) is left open or U+2069 closes
    one that was never opened, on any line: an isolate ends with its line
    (each line is a paragraph of its own), and an open one would take the
    rest of the line, and whatever eov puts after the text, out of its
    direction."""
    for line in text.split("\n"):
        depth = 0
        for c in line:
            if c in ISOLATE_STARTS:
                depth += 1
            elif c == ISOLATE_END:
                depth -= 1
                if depth < 0:
                    return True
        if depth != 0:
            return True
    return False


def right_to_left(text: str) -> str:
    """`text` with a right-to-left mark at the start of each line that
    needs one."""
    marked = set(left_to_right_lines(text))
    if not marked:
        return text
    return "\n".join(
        RIGHT_TO_LEFT_MARK + line if index in marked else line for index, line in enumerate(text.split("\n"))
    )


def validate_translation(entry, translation, language: Language) -> str | None:
    """Why a translation cannot be used, or None."""
    sources = placeholders(entry.msgid) | placeholders(entry.msgid_plural or "")
    required = sources - {"{n}"}
    forms = translation if isinstance(translation, list) else [translation]
    if entry.msgid_plural:
        if not isinstance(translation, list) or len(translation) != language.plural_count:
            return f"needs a list of {language.plural_count} plural forms"
    elif not isinstance(translation, str):
        return "needs a single string"
    for form in forms:
        if not isinstance(form, str) or not form.strip():
            return "empty translation"
        found = placeholders(form)
        if found - sources:
            return f"unknown placeholders {sorted(found - sources)}"
        if required - found:
            return f"missing placeholders {sorted(required - found)}"
        if unbalanced_isolates(form):
            return (
                "every U+2066, U+2067 or U+2068 (isolate) needs its own U+2069 after it on the same line, "
                "and the other way round"
            )
        if any(c in EMBEDDINGS for c in form):
            return "uses U+202A-U+202E (embeddings, overrides): isolate the fragment with U+2068 ... U+2069 instead"
        lines = left_to_right_lines(form) if language.rtl else []
        if lines == [0]:
            return (
                "starts with a Latin letter or a value, so it would be laid out from left to right: "
                "put U+200F (right-to-left mark) first (`import` does)"
            )
        if lines:
            shown = ", ".join(str(index + 1) for index in lines)
            return (
                f"line {shown} starts with a Latin letter or a value, so it would be laid out from left "
                "to right (each line is laid out on its own): put U+200F (right-to-left mark) at the "
                "start of that line (`import` does)"
            )
    if not entry.msgid_plural and "{n}" in sources and "{n}" not in placeholders(forms[0]):
        return "missing placeholder {n}"
    if entry.msgid_plural and "{n}" in placeholders(entry.msgid_plural):
        # Every form that a number other than one takes shows the number,
        # as eov's own check (`check_catalog`) requires: in French the form
        # of 1 is also the form of 0.
        for form, numbers in plural_examples(language).items():
            if form < len(forms) and any(n != 1 for n in numbers) and "{n}" not in placeholders(forms[form]):
                shown = ", ".join(str(n) for n in numbers[:4])
                return f"plural form {form} (for {shown}) does not show {{n}}"
    return None


# The ids of exports made before ids named their text (`eov:12`): the
# message's position in the catalog then. Positions change whenever a
# catalog is merged, so such an id may name another text by now.
POSITION_ID = re.compile(r"^[^#:]+:\d+$")


def import_translations(component_list: list[Component], tag: str, translations: dict) -> tuple[int, list[str]]:
    """Writes translations by the ids `export` gave them; returns how many
    were written and why the others were refused. Every id is either
    written or refused: one that names no text of the selected components
    (an old position id, a text that changed since the export, another
    component's) is never applied anywhere."""
    po = polib()
    language = next((l for l in languages() if l.tag == tag), None)
    if language is None:
        sys.exit(f"unknown language {tag}")
    written = 0
    problems = []
    handled = set()
    for key in translations:
        if POSITION_ID.match(key):
            problems.append(f"{key}: an id by position, from an export made before ids named their text; export again")
            handled.add(key)
    for component in component_list:
        prefix = f"{component.name}#"
        keys = [key for key in translations if key.startswith(prefix)]
        handled.update(keys)
        catalog = component.catalog(tag)
        if not catalog.exists():
            problems += [f"{key}: {component.name} has no catalog for {tag} ({catalog})" for key in keys]
            continue
        file = po.pofile(str(catalog), wrapwidth=0)
        changed = False
        by_id = {entry_id(component, entry): entry for entry in file if not entry.obsolete}
        for key in keys:
            entry = by_id.get(key)
            if entry is None:
                problems.append(f"{key}: no such text in {catalog} (the catalog changed; export again)")
                continue
            translation = translations[key]
            if language.rtl:
                if isinstance(translation, str):
                    translation = right_to_left(translation)
                elif isinstance(translation, list):
                    translation = [right_to_left(form) if isinstance(form, str) else form for form in translation]
            problem = validate_translation(entry, translation, language)
            if problem:
                problems.append(f"{key} {entry.msgid!r}: {problem}")
                continue
            if entry.msgid_plural:
                entry.msgstr_plural = {i: form for i, form in enumerate(translation)}
            else:
                entry.msgstr = translation
            if "fuzzy" in entry.flags:
                entry.flags.remove("fuzzy")
            entry.previous_msgid = None
            entry.previous_msgid_plural = None
            entry.previous_msgctxt = None
            written += 1
            changed = True
        if changed:
            file.save(str(catalog))
            normalize(catalog)
    names = ", ".join(component.name for component in component_list)
    for key in translations:
        if key not in handled:
            problems.append(f"{key}: not an id of the selected components ({names})")
    return written, problems


PSEUDO = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "àƀçđéƒĝĥîĵķļɱñöþǫŕšţûṽŵẋýžÀƁÇĐÉƑĜĤÎĴĶĻṀÑÖÞǪŔŠŢÛṼŴẊÝŽ",
)


def pseudo_text(text: str) -> str:
    """Accented, 40 % longer and bracketed, placeholders kept: shows texts
    that are not translated, cut off or put together from pieces."""
    pieces = []
    last = 0
    for match in PLACEHOLDER.finditer(text):
        pieces.append(text[last:match.start()].translate(PSEUDO))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(text[last:].translate(PSEUDO))
    body = "".join(pieces)
    padding = "~" * max(1, round(len(text) * 0.4))
    return f"⟦{body} {padding}⟧"


def pseudo(component_list: list[Component], target: Path, tag: str) -> None:
    po = polib()
    language = next(l for l in languages() if l.tag == tag)
    for component in component_list:
        if not component.template.exists():
            continue
        file = po.pofile(str(component.template), wrapwidth=0)
        file.metadata.update(
            {
                "Language": tag,
                "Plural-Forms": language.plural_forms,
                "Content-Type": "text/plain; charset=UTF-8",
            }
        )
        for entry in file:
            if entry.msgid_plural:
                entry.msgstr_plural = {
                    i: pseudo_text(entry.msgid if i == 0 else entry.msgid_plural)
                    for i in range(language.plural_count)
                }
            else:
                entry.msgstr = pseudo_text(entry.msgid)
        out = target / component.name / tag / "LC_MESSAGES" / f"{component.domain}.po"
        out.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(out))
        print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_components(p):
        p.add_argument("components", nargs="*", help="eov, example_plugin, annotations, … (default: all present)")
        p.add_argument(
            "--plugin",
            action="append",
            default=[],
            metavar="DIR",
            help="a plugin outside this repository (the folder of its plugin.toml; see above for the files read)",
        )
        return p

    with_components(sub.add_parser("extract", help="regenerate the templates"))
    with_components(sub.add_parser("update", help="extract, then merge every catalog with its template"))
    p = with_components(sub.add_parser("check", help="validate catalogs"))
    p.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a template is out of date with the sources (its texts, notes for translators or the files that use them)",
    )
    p.add_argument("--complete", action="store_true", help="also fail on untranslated or fuzzy messages")
    p.add_argument("--language", action="append", help="only this language (repeatable)")
    p = with_components(sub.add_parser("stats", help="translated / fuzzy / untranslated per language"))
    p.add_argument("--language", action="append", help="only this language (repeatable)")
    p = with_components(sub.add_parser("export", help="write what is left to translate as JSON"))
    p.add_argument("--language", required=True)
    p.add_argument("--all", action="store_true", help="every message, translated or not")
    p.add_argument("output", type=Path)
    p = with_components(sub.add_parser("import", help="write translations from JSON ({id: text or [forms]})"))
    p.add_argument("--language", required=True)
    p.add_argument("input", type=Path)
    p = with_components(sub.add_parser("pseudo", help="pseudo-translations for layout checks"))
    p.add_argument("--language", default="de", help="the language whose plural rules to use")
    p.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command in {"update", "check", "export", "import", "pseudo"}:
        polib()  # before anything is written
    selected = components(args.components, args.plugin)
    if args.command == "extract":
        for component in selected:
            print(f"{component.template.relative_to(ROOT) if component.template.is_relative_to(ROOT) else component.template}: {'updated' if extract(component) else 'unchanged'}")
    elif args.command == "update":
        for component in selected:
            update(component)
            print(f"{component.name}: merged {len(languages())} catalogs")
    elif args.command == "check":
        problems = [
            problem
            for component in selected
            for problem in check(component, args.strict, args.complete, args.language)
        ]
        for problem in problems:
            print(problem)
        if problems:
            sys.exit(1)
        print(f"{len(selected)} components: catalogs are valid")
    elif args.command == "stats":
        for component in selected:
            print(f"{component.name}:")
            for tag, translated, fuzzy, untranslated in stats(component, args.language):
                total = translated + fuzzy + max(untranslated, 0)
                share = 100 * translated / total if total else 0
                print(f"  {tag:8} {translated:5} translated {fuzzy:4} fuzzy {untranslated:5} untranslated  {share:5.1f} %")
    elif args.command == "export":
        work = export(selected, args.language, args.all)
        args.output.write_text(json.dumps(work, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(work)} messages to {args.output}")
    elif args.command == "import":
        translations = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(translations, dict):
            what = "a list (an export file?)" if isinstance(translations, list) else "not an object"
            sys.exit(
                f"{args.input}: import takes a JSON object of translations by id "
                '({"<id>": "text" or ["form", …]}, with the ids an export wrote); '
                f"this is {what}"
            )
        written, problems = import_translations(selected, args.language, translations)
        for problem in problems:
            print(f"refused {problem}")
        print(f"{written} translations written, {len(problems)} refused")
        if problems:
            sys.exit(1)
    elif args.command == "pseudo":
        pseudo(selected, args.output, args.language)


if __name__ == "__main__":
    main()
