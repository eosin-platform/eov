#!/usr/bin/env python3
"""Consistency checks for the packaging and CI files (run by CI).

- Every GitHub Action is pinned to a full commit SHA (UPD-14).
- The OpenSlide fork commit and the libdicom/uthash checksums agree between
  packaging/shared/build-openslide.sh, the Flatpak manifest and the
  third-party notices (SH-19).
- AppRun does not export LD_LIBRARY_PATH (UPD-13) or any variable without
  saving its launch value for the programs eov starts (SH-15), and the
  desktop entry's StartupWMClass matches the app id eov sets (SH-19).
- The Flatpak grants network and input access (UPD-5).
- The AppImage and the Flatpak ship the MIME package of the slide formats,
  and the desktop entry declares them and eov:// links (SH-8).

Exit status 1 lists every problem found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*([^\s#]+)", re.MULTILINE)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def check_action_pins(problems: list[str]) -> None:
    files = sorted((ROOT / ".github").rglob("*.yml"))
    for path in files:
        for reference in USES_RE.findall(path.read_text(encoding="utf-8")):
            if reference.startswith("./") or reference.startswith("docker://"):
                continue
            _, _, version = reference.partition("@")
            if not SHA_RE.fullmatch(version):
                problems.append(
                    f"{path.relative_to(ROOT)}: {reference} is not pinned to a commit SHA"
                )


def shell_default(script: str, name: str) -> str | None:
    match = re.search(rf'^{name}="([^"]+)"$', script, re.MULTILINE)
    return match.group(1) if match else None


def check_openslide_pins(problems: list[str]) -> None:
    script = read("packaging/shared/build-openslide.sh")
    flatpak = read("packaging/flatpak/io.eosin.eov.yml")
    notices = read("assets/COPYING.third-party")
    pins = {
        name: shell_default(script, name)
        for name in (
            "DEFAULT_OPENSLIDE_REPO_URL",
            "DEFAULT_OPENSLIDE_COMMIT",
            "DEFAULT_LIBDICOM_VERSION",
            "DEFAULT_LIBDICOM_SHA256",
            "DEFAULT_UTHASH_VERSION",
            "DEFAULT_UTHASH_SHA256",
        )
    }
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        problems.append(f"build-openslide.sh does not define {', '.join(missing)}")
        return
    commit = pins["DEFAULT_OPENSLIDE_COMMIT"]
    expectations = {
        "OpenSlide fork URL": f"url: {pins['DEFAULT_OPENSLIDE_REPO_URL']}",
        "OpenSlide commit": f"commit: {commit}",
        "libdicom URL": (
            "url: https://github.com/ImagingDataCommons/libdicom/releases/download/"
            f"v{pins['DEFAULT_LIBDICOM_VERSION']}/libdicom-{pins['DEFAULT_LIBDICOM_VERSION']}.tar.xz"
        ),
        "libdicom checksum": f"sha256: {pins['DEFAULT_LIBDICOM_SHA256']}",
        "uthash URL": (
            "url: https://github.com/troydhanson/uthash/archive/"
            f"v{pins['DEFAULT_UTHASH_VERSION']}.tar.gz"
        ),
        "uthash checksum": f"sha256: {pins['DEFAULT_UTHASH_SHA256']}",
    }
    for what, line in expectations.items():
        if line not in flatpak:
            problems.append(
                f"packaging/flatpak/io.eosin.eov.yml: {what} differs from build-openslide.sh ({line})"
            )
    if commit not in notices:
        problems.append(f"assets/COPYING.third-party does not name OpenSlide commit {commit}")
    fork = pins["DEFAULT_OPENSLIDE_REPO_URL"].removesuffix(".git")
    if fork not in notices:
        problems.append(f"assets/COPYING.third-party does not name the OpenSlide fork {fork}")
    if f"libdicom {pins['DEFAULT_LIBDICOM_VERSION']}" not in notices:
        problems.append("assets/COPYING.third-party names a different libdicom version")


def check_linux_metadata(problems: list[str]) -> None:
    apprun = read("packaging/appimage/AppRun")
    code = "\n".join(line for line in apprun.splitlines() if not line.lstrip().startswith("#"))
    if "LD_LIBRARY_PATH" in code:
        problems.append("packaging/appimage/AppRun sets LD_LIBRARY_PATH; bundled libraries use RUNPATH")
    # Programs eov starts get the launch environment back (SH-15): every
    # variable AppRun changes needs its launch value saved.
    for name in re.findall(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)", code, re.MULTILINE):
        if name.startswith("EOV_ORIG_") or name == "EOV_SAVED_ENV":
            continue
        if f"EOV_ORIG_{name}" not in code or "EOV_SAVED_ENV" not in code:
            problems.append(
                f"packaging/appimage/AppRun exports {name} without saving its launch value "
                f"(EOV_ORIG_{name} and EOV_SAVED_ENV, see app/src/desktop.rs)"
            )
    desktop = read("assets/linux/io.eosin.eov.desktop")
    app_id = re.search(r'APP_XDG_ID: &str = "([^"]+)"', read("app/src/main.rs"))
    wm_class = re.search(r"^StartupWMClass=(.+)$", desktop, re.MULTILINE)
    if app_id is None or wm_class is None or wm_class.group(1) != app_id.group(1):
        problems.append(
            "assets/linux/io.eosin.eov.desktop: StartupWMClass must equal the app id "
            "passed to slint::set_xdg_app_id"
        )
    flatpak = read("packaging/flatpak/io.eosin.eov.yml")
    for permission in ("--share=network", "--device=input"):
        if f"- {permission}" not in flatpak:
            problems.append(f"packaging/flatpak/io.eosin.eov.yml: missing {permission}")
    if "/app/share/mime/packages/io.eosin.eov.xml" not in flatpak:
        problems.append("packaging/flatpak/io.eosin.eov.yml: does not install the MIME package")
    appimage = read("packaging/appimage/build.sh")
    if '"$MIME_FILE" "$APPDIR_MIME_DIR/$APP_ID.xml"' not in appimage:
        problems.append("packaging/appimage/build.sh: does not install the MIME package")
    mime_types = re.findall(r'<mime-type type="([^"]+)">', read("assets/linux/io.eosin.eov.xml"))
    declared = re.search(r"^MimeType=(.*)$", desktop, re.MULTILINE)
    declared = set(filter(None, declared.group(1).split(";"))) if declared else set()
    for mime in mime_types + ["image/tiff", "application/dicom", "x-scheme-handler/eov"]:
        if mime not in declared:
            problems.append(f"assets/linux/io.eosin.eov.desktop: MimeType= lacks {mime}")
    exec_line = re.search(r"^Exec=(.*)$", desktop, re.MULTILINE)
    if exec_line is None or not exec_line.group(1).endswith("%U"):
        problems.append("assets/linux/io.eosin.eov.desktop: Exec= must take %U (eov:// links)")


def main() -> int:
    problems: list[str] = []
    check_action_pins(problems)
    check_openslide_pins(problems)
    check_linux_metadata(problems)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print("packaging checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
