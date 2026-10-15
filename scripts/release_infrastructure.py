#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import plistlib
import re
import shutil
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path


class ReleaseError(Exception):
    pass


SEMVER_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
STABLE_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
README_URL_RE = re.compile(
    r"https://github\.com/eosin-platform/eov/releases/(?:download|tag)/[^\s)]+"
)

EOV_REPOSITORY = "eosin-platform/eov"
EOV_MANIFEST_START = "<!-- release-install-links:start -->"
EOV_MANIFEST_END = "<!-- release-install-links:end -->"


@dataclass(frozen=True)
class ArtifactSpec:
    section: str
    filename: str


EOV_ARTIFACTS = (
    ArtifactSpec("platform.windows.x86_64", "eov-v{version}-windows-x86_64.zip"),
    ArtifactSpec("platform.windows.arm64", "eov-v{version}-windows-arm64.zip"),
    ArtifactSpec(
        "platform.linux.appimage.arm64", "eov-v{version}-linux-arm64.AppImage"
    ),
    ArtifactSpec("platform.linux.flatpak.arm64", "eov-v{version}-linux-arm64.flatpak"),
    ArtifactSpec(
        "platform.linux.appimage.x86_64", "eov-v{version}-linux-x86_64.AppImage"
    ),
    ArtifactSpec(
        "platform.linux.flatpak.x86_64", "eov-v{version}-linux-x86_64.flatpak"
    ),
    ArtifactSpec("platform.macos.arm64", "eov-v{version}-macos-arm64.zip"),
    ArtifactSpec("platform.macos.x86_64", "eov-v{version}-macos-x86_64.zip"),
)

PLUGIN_ARTIFACTS = {
    "annotations": tuple(
        ArtifactSpec(
            section,
            f"annotations-v{{version}}-{platform}.eop",
        )
        for section, platform in (
            ("platform.windows.x86_64", "windows-x86_64"),
            ("platform.windows.arm64", "windows-arm64"),
            ("platform.linux.arm64", "linux-arm64"),
            ("platform.linux.x86_64", "linux-x86_64"),
            ("platform.macos.arm64", "macos-arm64"),
            ("platform.macos.x86_64", "macos-x86_64"),
        )
    ),
    "gamepad": tuple(
        ArtifactSpec(
            section,
            f"gamepad-v{{version}}-{platform}.eop",
        )
        for section, platform in (
            ("platform.windows.x86_64", "windows-x86_64"),
            ("platform.windows.arm64", "windows-arm64"),
            ("platform.linux.arm64", "linux-arm64"),
            ("platform.linux.x86_64", "linux-x86_64"),
            ("platform.macos.arm64", "macos-arm64"),
            ("platform.macos.x86_64", "macos-x86_64"),
        )
    ),
}

PLUGIN_METADATA = {
    "annotations": (
        "https://github.com/eosin-platform/eov-annotations-plugin",
        "Annotations",
        "Official annotations plugin",
    ),
    "gamepad": (
        "https://github.com/eosin-platform/eov-gamepad-plugin",
        "Gamepad",
        "Provides support for using gamepads as input devices",
    ),
}

MACOS_CASK_SYMBOLS = {
    "10.13": "high_sierra",
    "10.14": "mojave",
    "10.15": "catalina",
    "11.0": "big_sur",
    "12.0": "monterey",
    "13.0": "ventura",
    "14.0": "sonoma",
    "15.0": "sequoia",
    "26.0": "tahoe",
}

# Where eov keeps state on macOS (eov_common::paths): the eov home, the
# platform data and cache directories (the pre-1.0 config directory is the
# same "Application Support/eov"), and what macOS keeps per bundle id.
CASK_ZAP_PATHS = (
    "~/.eov",
    "~/Library/Application Support/eov",
    "~/Library/Caches/eov",
    "~/Library/Application Support/io.eosin.eov",
    "~/Library/Caches/io.eosin.eov",
    "~/Library/HTTPStorages/io.eosin.eov",
    "~/Library/Preferences/io.eosin.eov.plist",
    "~/Library/Saved Application State/io.eosin.eov.savedState",
)

# Platform sections of per-platform plugin artifacts in plugin manifests.
PLUGIN_PLATFORM_SECTIONS = (
    "platform.windows.x86_64",
    "platform.windows.arm64",
    "platform.linux.x86_64",
    "platform.linux.arm64",
    "platform.macos.x86_64",
    "platform.macos.arm64",
)


def validate_version(version: str, *, stable_only: bool = False) -> str:
    pattern = STABLE_VERSION_RE if stable_only else SEMVER_RE
    if pattern.fullmatch(version) is None:
        qualifier = "stable " if stable_only else ""
        raise ReleaseError(f"{qualifier}version must be valid SemVer: {version!r}")
    return version


@functools.total_ordering
@dataclass(frozen=True)
class _SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[int | str, ...] = ()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _SemVer):
            return NotImplemented
        core = (self.major, self.minor, self.patch)
        other_core = (other.major, other.minor, other.patch)
        if core != other_core:
            return core < other_core
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True
        for left, right in zip(self.prerelease, other.prerelease):
            if left == right:
                continue
            if isinstance(left, int) and isinstance(right, str):
                return True
            if isinstance(left, str) and isinstance(right, int):
                return False
            return left < right
        return len(self.prerelease) < len(other.prerelease)


def _parse_semver(version: str) -> _SemVer:
    validate_version(version)
    core_and_pre = version.split("+", 1)[0]
    core, separator, prerelease = core_and_pre.partition("-")
    major, minor, patch = (int(part) for part in core.split("."))
    identifiers: tuple[int | str, ...] = ()
    if separator:
        identifiers = tuple(
            int(identifier) if identifier.isdigit() else identifier
            for identifier in prerelease.split(".")
        )
    return _SemVer(major, minor, patch, identifiers)


def _stable_version(version: str) -> _SemVer:
    parsed = _parse_semver(version)
    if parsed.prerelease:
        raise ReleaseError(f"stable version required: {version!r}")
    return parsed


@dataclass(frozen=True)
class _Comparator:
    """One comparator of a Rust ``semver::VersionReq``."""

    op: str
    major: int
    minor: int | None
    patch: int | None
    prerelease: tuple[int | str, ...] = ()


_IDENTIFIER_RE = re.compile(r"[0-9A-Za-z-]+")


def _numeric_identifier(text: str) -> tuple[int, str]:
    match = re.match(r"[0-9]+", text)
    if match is None:
        raise ReleaseError("expected a number")
    digits = match.group(0)
    if len(digits) > 1 and digits.startswith("0"):
        raise ReleaseError("numbers must not have leading zeros")
    value = int(digits)
    if value > 2**64 - 1:
        raise ReleaseError("number too large")
    return value, text[len(digits) :]


def _dotted_identifiers(text: str, *, numeric_rules: bool) -> tuple[tuple[str, ...], str]:
    identifiers = []
    while True:
        match = _IDENTIFIER_RE.match(text)
        if match is None:
            raise ReleaseError("empty identifier")
        identifier = match.group(0)
        if (
            numeric_rules
            and identifier.isdigit()
            and len(identifier) > 1
            and identifier.startswith("0")
        ):
            raise ReleaseError("numeric identifiers must not have leading zeros")
        identifiers.append(identifier)
        text = text[len(identifier) :]
        if not text.startswith("."):
            return tuple(identifiers), text
        text = text[1:]


def _wildcard(text: str) -> tuple[str, str] | None:
    if text[:1] in {"*", "x", "X"}:
        return text[0], text[1:]
    return None


def _parse_comparator(text: str) -> tuple[_Comparator, str]:
    op = None
    for candidate in (">=", "<=", "=", ">", "<", "~", "^"):
        if text.startswith(candidate):
            op = candidate
            text = text[len(candidate) :]
            break
    default_op = op is None
    op = op or "^"
    text = text.lstrip(" ")
    major, text = _numeric_identifier(text)
    minor: int | None = None
    patch: int | None = None
    has_wildcard = False
    if text.startswith("."):
        text = text[1:]
        wildcard = _wildcard(text)
        if wildcard is not None:
            has_wildcard = True
            if default_op:
                op = "*"
            text = wildcard[1]
        else:
            minor, text = _numeric_identifier(text)
    if text.startswith("."):
        text = text[1:]
        wildcard = _wildcard(text)
        if wildcard is not None:
            if default_op:
                op = "*"
            text = wildcard[1]
        elif has_wildcard:
            raise ReleaseError("unexpected character after wildcard")
        else:
            patch, text = _numeric_identifier(text)
    prerelease: tuple[int | str, ...] = ()
    if patch is not None and text.startswith("-"):
        identifiers, text = _dotted_identifiers(text[1:], numeric_rules=True)
        prerelease = tuple(int(part) if part.isdigit() else part for part in identifiers)
    if patch is not None and text.startswith("+"):
        _, text = _dotted_identifiers(text[1:], numeric_rules=False)
    return _Comparator(op, major, minor, patch, prerelease), text.lstrip(" ")


def parse_requirement(requirement: str) -> tuple[_Comparator, ...]:
    """Parse a requirement exactly like Rust's ``semver::VersionReq::parse``.

    Returns the comparators (empty for ``*``). Raises ``ReleaseError`` for
    everything the Rust crate rejects: ``||``, hyphen ranges, space-separated
    comparators, ``v`` prefixes, leading zeros, empty strings and more.
    """

    text = requirement.lstrip(" ")
    wildcard = _wildcard(text)
    if wildcard is not None:
        rest = wildcard[1].lstrip(" ")
        if not rest:
            return ()
        raise ReleaseError(f"invalid SemVer requirement {requirement!r}: wildcard must stand alone")
    comparators = []
    try:
        while True:
            comparator, text = _parse_comparator(text)
            comparators.append(comparator)
            if not text:
                break
            if not text.startswith(","):
                raise ReleaseError("expected a comma between comparators")
            text = text[1:].lstrip(" ")
            if len(comparators) >= 32:
                raise ReleaseError("too many comparators")
    except ReleaseError as error:
        raise ReleaseError(f"invalid SemVer requirement {requirement!r}: {error}") from error
    return tuple(comparators)


def _pre_key(prerelease: tuple[int | str, ...]) -> _SemVer:
    return _SemVer(0, 0, 0, prerelease)


def _pre_cmp(left: tuple[int | str, ...], right: tuple[int | str, ...]) -> int:
    # A release (empty prerelease) sorts after every prerelease.
    if not left and not right:
        return 0
    if not left:
        return 1
    if not right:
        return -1
    left_key, right_key = _pre_key(left), _pre_key(right)
    return (left_key > right_key) - (left_key < right_key)


def _matches_exact(cmp: _Comparator, ver: _SemVer) -> bool:
    if ver.major != cmp.major:
        return False
    if cmp.minor is not None and ver.minor != cmp.minor:
        return False
    if cmp.patch is not None and ver.patch != cmp.patch:
        return False
    return ver.prerelease == cmp.prerelease


def _matches_greater(cmp: _Comparator, ver: _SemVer) -> bool:
    if ver.major != cmp.major:
        return ver.major > cmp.major
    if cmp.minor is None:
        return False
    if ver.minor != cmp.minor:
        return ver.minor > cmp.minor
    if cmp.patch is None:
        return False
    if ver.patch != cmp.patch:
        return ver.patch > cmp.patch
    return _pre_cmp(ver.prerelease, cmp.prerelease) > 0


def _matches_less(cmp: _Comparator, ver: _SemVer) -> bool:
    if ver.major != cmp.major:
        return ver.major < cmp.major
    if cmp.minor is None:
        return False
    if ver.minor != cmp.minor:
        return ver.minor < cmp.minor
    if cmp.patch is None:
        return False
    if ver.patch != cmp.patch:
        return ver.patch < cmp.patch
    return _pre_cmp(ver.prerelease, cmp.prerelease) < 0


def _matches_tilde(cmp: _Comparator, ver: _SemVer) -> bool:
    if ver.major != cmp.major:
        return False
    if cmp.minor is not None and ver.minor != cmp.minor:
        return False
    if cmp.patch is not None and ver.patch != cmp.patch:
        return ver.patch > cmp.patch
    return _pre_cmp(ver.prerelease, cmp.prerelease) >= 0


def _matches_caret(cmp: _Comparator, ver: _SemVer) -> bool:
    if ver.major != cmp.major:
        return False
    if cmp.minor is None:
        return True
    if cmp.patch is None:
        return ver.minor >= cmp.minor if cmp.major > 0 else ver.minor == cmp.minor
    if cmp.major > 0:
        if ver.minor != cmp.minor:
            return ver.minor > cmp.minor
        if ver.patch != cmp.patch:
            return ver.patch > cmp.patch
    elif cmp.minor > 0:
        if ver.minor != cmp.minor:
            return False
        if ver.patch != cmp.patch:
            return ver.patch > cmp.patch
    elif ver.minor != cmp.minor or ver.patch != cmp.patch:
        return False
    return _pre_cmp(ver.prerelease, cmp.prerelease) >= 0


def _matches_comparator(cmp: _Comparator, ver: _SemVer) -> bool:
    if cmp.op in {"=", "*"}:
        return _matches_exact(cmp, ver)
    if cmp.op == ">":
        return _matches_greater(cmp, ver)
    if cmp.op == ">=":
        return _matches_exact(cmp, ver) or _matches_greater(cmp, ver)
    if cmp.op == "<":
        return _matches_less(cmp, ver)
    if cmp.op == "<=":
        return _matches_exact(cmp, ver) or _matches_less(cmp, ver)
    if cmp.op == "~":
        return _matches_tilde(cmp, ver)
    return _matches_caret(cmp, ver)


def _requirement_matches(comparators: tuple[_Comparator, ...], ver: _SemVer) -> bool:
    if not all(_matches_comparator(cmp, ver) for cmp in comparators):
        return False
    if not ver.prerelease:
        return True
    # A prerelease only satisfies a requirement when a comparator names a
    # prerelease of the same major.minor.patch.
    return any(
        cmp.major == ver.major
        and cmp.minor == ver.minor
        and cmp.patch == ver.patch
        and cmp.prerelease
        for cmp in comparators
    )


def semver_requirement_satisfied(requirement: str, version: str) -> bool:
    """Evaluate a requirement exactly like Rust's ``VersionReq::matches``.

    Cross-tested against the Rust crate through
    ``app/tests/data/semver-cases.json``.
    """

    return _requirement_matches(parse_requirement(requirement), _parse_semver(version))


def bounded_requirement(requirement: str) -> tuple[_Comparator, ...]:
    """eov's bounded form of a plugin requirement (``update/compat.rs``).

    A requirement without an upper bound is capped at the next breaking
    release of its highest lower bound (``>=0.4.1`` means ``<0.5.0``); a
    requirement with no lower bound either (``*``) is rejected.
    """

    comparators = parse_requirement(requirement)
    if any(cmp.op in {"=", "<", "<=", "~", "^", "*"} for cmp in comparators):
        return comparators
    lower_bounds = [
        (cmp.major, cmp.minor or 0, cmp.patch or 0)
        for cmp in comparators
        if cmp.op in {">", ">="}
    ]
    if not lower_bounds:
        raise ReleaseError(
            f"requirement {requirement!r} has no lower bound; "
            'declare a range such as ">=1.0.0, <2.0.0"'
        )
    major, minor, patch = max(lower_bounds)
    if major > 0:
        cap = (major + 1, 0, 0)
    elif minor > 0:
        cap = (0, minor + 1, 0)
    else:
        cap = (0, 0, patch + 1)
    return comparators + (_Comparator("<", *cap),)


def host_compatible(requirement: str, version: str) -> bool:
    """Whether eov ``version`` loads a plugin declaring ``requirement``,
    with the same bounded and prerelease rules as ``update/compat.rs``."""

    try:
        comparators = bounded_requirement(requirement)
    except ReleaseError:
        return False
    host = _parse_semver(version)
    if _requirement_matches(comparators, host):
        return True
    mentions_prerelease = any(cmp.prerelease for cmp in comparators)
    return (
        bool(host.prerelease)
        and not mentions_prerelease
        and _requirement_matches(comparators, _SemVer(host.major, host.minor, host.patch))
    )


def requirement_is_bounded(requirement: str) -> bool:
    """Whether a requirement declares an explicit upper bound."""

    return any(cmp.op in {"=", "<", "<=", "~", "^", "*"} for cmp in parse_requirement(requirement))


def select_compatible_plugin_release(
    eov_version: str, candidates: list[tuple[str, str]]
) -> str:
    """Return the newest stable candidate whose environment accepts EOV."""

    _parse_semver(eov_version)
    compatible: list[tuple[_SemVer, str]] = []
    for version, environment in candidates:
        parsed = _stable_version(version.removeprefix("v"))
        if host_compatible(environment, eov_version):
            compatible.append((parsed, version.removeprefix("v")))
    if not compatible:
        raise ReleaseError(
            f"no stable plugin release is compatible with EOV {eov_version}"
        )
    compatible.sort(reverse=True)
    return compatible[0][1]


def _github_json(url: str) -> object:
    headers = {
        "User-Agent": "eov-release-infrastructure",
        "Accept": "application/vnd.github+json",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ReleaseError(
                    f"GitHub API returned HTTP {response.status} for {url}"
                )
            return json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise ReleaseError(
            f"could not read GitHub API response {url}: {error}"
        ) from error


def published_plugin_candidates(repository: str) -> list[tuple[str, str]]:
    """Read stable plugin release manifests in descending release order."""

    releases: list[object] = []
    for page in range(1, 101):
        page_value = _github_json(
            f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}"
        )
        if not isinstance(page_value, list):
            raise ReleaseError(
                f"GitHub releases response for {repository} is not a list"
            )
        releases.extend(page_value)
        if len(page_value) < 100:
            break
    candidates: list[tuple[str, str]] = []
    for release in releases:
        if (
            not isinstance(release, dict)
            or release.get("draft")
            or release.get("prerelease")
        ):
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue
        version = tag.removeprefix("v")
        try:
            validate_version(version, stable_only=True)
        except ReleaseError:
            continue
        manifest_url = (
            f"https://github.com/{repository}/releases/download/{tag}/release.toml"
        )
        try:
            request = urllib.request.Request(
                manifest_url,
                headers={
                    "User-Agent": "eov-release-infrastructure",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = tomllib.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise ReleaseError(
                f"could not read plugin release manifest {manifest_url}: {error}"
            ) from error
        except (
            OSError,
            urllib.error.URLError,
            UnicodeDecodeError,
            tomllib.TOMLDecodeError,
        ) as error:
            raise ReleaseError(
                f"could not read plugin release manifest {manifest_url}: {error}"
            ) from error
        plugin_data = data.get("plugin")
        manifest_version = (
            plugin_data.get("version")
            if isinstance(plugin_data, dict)
            else _first_artifact_version(data)
        )
        if manifest_version != version:
            raise ReleaseError(
                f"plugin release {tag} declares version {manifest_version!r}, expected {version!r}"
            )
        environment = None
        if isinstance(plugin_data, dict) and isinstance(
            plugin_data.get("environment"), str
        ):
            environment = plugin_data["environment"]
        if environment is None:
            environment = _first_artifact_environment(data)
        if not isinstance(environment, str):
            raise ReleaseError(
                f"plugin release {tag} has no EOV environment requirement"
            )
        candidates.append((version, environment))
    return candidates


def _first_artifact_environment(data: dict[str, object]) -> str | None:
    def visit(value: object) -> str | None:
        if isinstance(value, dict):
            environment = value.get("environment")
            if isinstance(environment, str):
                return environment
            for child in value.values():
                found = visit(child)
                if found is not None:
                    return found
        return None

    platform = data.get("platform")
    return visit(platform)


def _first_artifact_version(data: dict[str, object]) -> str | None:
    def visit(value: object) -> str | None:
        if isinstance(value, dict):
            version = value.get("version")
            if isinstance(version, str) and "sha256" in value and "url" in value:
                return version
            for child in value.values():
                found = visit(child)
                if found is not None:
                    return found
        return None

    return visit(data.get("platform"))


def command_resolve_plugin_versions(args: argparse.Namespace) -> None:
    eov_version = validate_version(args.eov_version)
    for value in args.plugin:
        name, separator, repository = value.partition("=")
        if not separator or not name or not repository:
            raise ReleaseError(f"plugin must be NAME=OWNER/REPOSITORY: {value!r}")
        selected = select_compatible_plugin_release(
            eov_version, published_plugin_candidates(repository)
        )
        print(f"{name}={selected}")


def validate_sha256(value: str, description: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ReleaseError(f"{description} is not a SHA-256 digest: {value!r}")
    return value.lower()


def immutable_url(repository: str, version: str, filename: str) -> str:
    return f"https://github.com/{repository}/releases/download/v{version}/{filename}"


def write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise ReleaseError(f"could not write {path}: {error}") from error


def read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as input_file:
            value = tomllib.load(input_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"could not read TOML file {path}: {error}") from error
    return value


def nested_table(value: dict[str, object], dotted_name: str) -> dict[str, object]:
    current: object = value
    for part in dotted_name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ReleaseError(f"manifest is missing [{dotted_name}]")
        current = current[part]
    if not isinstance(current, dict):
        raise ReleaseError(f"manifest entry [{dotted_name}] is not a table")
    return current


def artifact_specs(
    kind: str, plugin_name: str | None, version: str
) -> tuple[ArtifactSpec, ...]:
    if kind == "eov":
        return tuple(
            ArtifactSpec(spec.section, spec.filename.format(version=version))
            for spec in EOV_ARTIFACTS
        )
    if plugin_name not in PLUGIN_ARTIFACTS:
        raise ReleaseError(f"unsupported plugin name: {plugin_name!r}")
    return tuple(
        ArtifactSpec(spec.section, spec.filename.format(version=version))
        for spec in PLUGIN_ARTIFACTS[plugin_name]
    )


def checksum_sidecar(path: Path, artifact_name: str, actual_sha256: str) -> None:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ReleaseError(
            f"could not read checksum sidecar {path}: {error}"
        ) from error
    if len(lines) != 1:
        raise ReleaseError(f"checksum sidecar {path} must contain exactly one entry")
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+)", lines[0])
    if match is None:
        raise ReleaseError(f"checksum sidecar {path} has an invalid format")
    sidecar_sha, sidecar_name = match.groups()
    if sidecar_name.removeprefix("*") != artifact_name:
        raise ReleaseError(
            f"checksum sidecar {path} names {sidecar_name!r}, expected {artifact_name!r}"
        )
    if sidecar_sha.lower() != actual_sha256:
        raise ReleaseError(f"checksum sidecar {path} does not match {artifact_name}")


def stage_artifacts(
    artifacts_dir: Path, staging_dir: Path, specs: tuple[ArtifactSpec, ...]
) -> dict[str, Path]:
    if not artifacts_dir.is_dir():
        raise ReleaseError(f"artifact directory does not exist: {artifacts_dir}")

    expected_names = {spec.filename for spec in specs}
    artifact_paths: dict[str, Path] = {}
    sidecar_paths: dict[str, Path] = {}
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".sha256"):
            base_name = name.removesuffix(".sha256")
            if base_name not in expected_names:
                raise ReleaseError(f"unexpected checksum artifact: {path}")
            if base_name in sidecar_paths:
                raise ReleaseError(f"duplicate checksum artifact: {base_name}")
            sidecar_paths[base_name] = path
        else:
            if name not in expected_names:
                raise ReleaseError(f"unexpected release artifact: {path}")
            if name in artifact_paths:
                raise ReleaseError(f"duplicate release artifact: {name}")
            artifact_paths[name] = path

    missing = sorted(expected_names - artifact_paths.keys())
    if missing:
        raise ReleaseError(f"missing release artifacts: {', '.join(missing)}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for name in sorted(expected_names):
        source = artifact_paths[name]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        sidecar = sidecar_paths.get(name)
        if sidecar is not None:
            checksum_sidecar(sidecar, name, digest)
        destination = staging_dir / name
        if destination.exists():
            raise ReleaseError(f"staging directory already contains {destination}")
        shutil.copy2(source, destination)
        if sidecar is not None:
            shutil.copy2(sidecar, staging_dir / sidecar.name)
        staged[name] = destination
    return staged


def source_version_from_cargo(path: Path) -> str:
    data = read_toml(path)
    package = data.get("package")
    if not isinstance(package, dict) or not isinstance(package.get("version"), str):
        raise ReleaseError(f"{path} has no package version")
    return package["version"]


def validate_eov_source(root: Path, requested_version: str) -> None:
    validate_version(requested_version)
    package_paths = (
        root / "app/Cargo.toml",
        root / "common/Cargo.toml",
        root / "plugin_api/Cargo.toml",
    )
    versions = {str(path): source_version_from_cargo(path) for path in package_paths}
    if len(set(versions.values())) != 1:
        raise ReleaseError(f"EOV Cargo package versions disagree: {versions}")
    authoritative_version = next(iter(versions.values()))
    if authoritative_version != requested_version:
        raise ReleaseError(
            f"requested version {requested_version} does not match EOV Cargo version {authoritative_version}"
        )

    app_data = read_toml(root / "app/Cargo.toml")
    dependencies = app_data.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ReleaseError("app/Cargo.toml has no dependencies table")
    for dependency in ("eov-common", "eov-plugin-api"):
        dependency_data = dependencies.get(dependency)
        if (
            not isinstance(dependency_data, dict)
            or dependency_data.get("version") != requested_version
        ):
            raise ReleaseError(
                f"app/Cargo.toml {dependency} dependency is not {requested_version}"
            )

    metainfo_path = root / "assets/linux/io.eosin.eov.metainfo.xml"
    try:
        xml_root = ET.parse(metainfo_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ReleaseError(f"could not read {metainfo_path}: {error}") from error
    releases = xml_root.findall("./releases/release")
    if not releases or releases[0].get("version") != requested_version:
        actual = releases[0].get("version") if releases else None
        raise ReleaseError(
            f"metainfo version {actual!r} does not match {requested_version}"
        )


def validate_plugin_source(root: Path, requested_version: str) -> None:
    validate_version(requested_version)
    cargo_version = source_version_from_cargo(root / "Cargo.toml")
    plugin_data = read_toml(root / "plugin.toml")
    manifest_version = plugin_data.get("version")
    if cargo_version != requested_version or manifest_version != requested_version:
        raise ReleaseError(
            f"plugin versions disagree with requested {requested_version}: "
            f"Cargo.toml={cargo_version!r}, plugin.toml={manifest_version!r}"
        )
    plugin_id = plugin_data.get("id")
    plugin_name = plugin_data.get("name")
    if not isinstance(plugin_id, str) or not plugin_id:
        raise ReleaseError("plugin.toml has no valid id")
    if not isinstance(plugin_name, str) or not plugin_name:
        raise ReleaseError("plugin.toml has no valid name")
    description = plugin_data.get("description")
    if description is not None and not isinstance(description, str):
        raise ReleaseError("plugin.toml description must be a string")
    repository = plugin_data.get("repository")
    if repository is not None and not isinstance(repository, str):
        raise ReleaseError("plugin.toml repository must be a string")
    environment = plugin_data.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("version"), str
    ):
        raise ReleaseError("plugin.toml has no valid [environment].version")


@dataclass(frozen=True)
class PluginRelease:
    """An official plugin release, read from its published release.toml."""

    version: str
    environment: str
    release_tag: str
    artifacts: dict[str, dict[str, str]]


def read_plugin_release(
    path: Path, plugin_id: str, repository: str, version: str, eov_version: str
) -> PluginRelease:
    """Read and validate an official plugin's release.toml for the catalog.

    The catalog pins every per-platform artifact's SHA-256, so the eov release
    (which is signed) vouches for the plugin bytes, not just their version.
    """

    data = read_toml(path)
    plugin = data.get("plugin")
    plugin = plugin if isinstance(plugin, dict) else {}
    declared_id = plugin.get("id")
    if declared_id is not None and declared_id != plugin_id:
        raise ReleaseError(f"{path} declares plugin {declared_id!r}, expected {plugin_id!r}")
    environment = plugin.get("environment") or _first_artifact_environment(data)
    if not isinstance(environment, str):
        raise ReleaseError(f"{path} has no EOV environment requirement")
    if not requirement_is_bounded(environment):
        raise ReleaseError(
            f"plugin {plugin_id} {version} declares the open-ended requirement {environment!r}; "
            "official plugins must declare an upper bound such as \">=1.0.0, <2.0.0\""
        )
    if not host_compatible(environment, eov_version):
        raise ReleaseError(
            f"plugin {plugin_id} {version} requires eov {environment!r}, "
            f"which excludes {eov_version}"
        )
    artifacts: dict[str, dict[str, str]] = {}
    tags = set()
    for section in PLUGIN_PLATFORM_SECTIONS:
        try:
            entry = nested_table(data, section)
        except ReleaseError:
            continue
        if entry.get("version") != version:
            raise ReleaseError(f"{path} [{section}] is not version {version}")
        sha256 = entry.get("sha256")
        url = entry.get("url")
        if not isinstance(sha256, str) or not isinstance(url, str):
            raise ReleaseError(f"{path} [{section}] is missing sha256 or url")
        validate_sha256(sha256, f"{path} [{section}].sha256")
        prefix = f"{repository}/releases/download/"
        if not url.startswith(prefix):
            raise ReleaseError(f"{path} [{section}] URL does not belong to {repository}")
        tag, _, file_name = url[len(prefix) :].rpartition("/")
        if not tag or not file_name:
            raise ReleaseError(f"{path} [{section}] URL is not a release asset URL")
        tags.add(tag)
        artifacts[section] = {"sha256": sha256.lower(), "url": url}
    if not artifacts:
        raise ReleaseError(f"{path} lists no per-platform plugin artifacts")
    if len(tags) != 1:
        raise ReleaseError(f"{path} artifacts point at several releases: {sorted(tags)}")
    return PluginRelease(version, environment, tags.pop(), artifacts)


def render_manifest(
    version: str,
    repository: str,
    specs: tuple[ArtifactSpec, ...],
    staged: dict[str, Path],
    plugin_versions: dict[str, str] | None = None,
    environment: str | None = None,
    plugin_metadata: tuple[str, str, str | None, str | None] | None = None,
    official_metadata: dict[str, tuple[str, str, str]] | None = None,
    plugin_releases: dict[str, PluginRelease] | None = None,
    signed: bool = False,
) -> str:
    lines: list[str] = []
    kind = "eov" if plugin_versions is not None else "plugin"
    lines.extend(
        (
            "[manifest]",
            "schema = 1",
            f"kind = {json.dumps(kind)}",
            f"version = {json.dumps(version)}",
            f"repository = {json.dumps(f'https://github.com/{repository}')}",
        )
    )
    if signed:
        lines.append('signature = "minisign"')
    lines.append("")
    if plugin_metadata is not None:
        plugin_id, plugin_name, description, plugin_repository = plugin_metadata
        lines.extend(
            (
                "[plugin]",
                f"id = {json.dumps(plugin_id)}",
                f"name = {json.dumps(plugin_name)}",
                f"version = {json.dumps(version)}",
                f"repository = {json.dumps(plugin_repository or f'https://github.com/{repository}')}",
            )
        )
        if description is not None:
            lines.append(f"description = {json.dumps(description)}")
        if environment is not None:
            lines.append(f"environment = {json.dumps(environment)}")
        lines.append("")
    for spec in specs:
        digest = hashlib.sha256(staged[spec.filename].read_bytes()).hexdigest()
        lines.extend(
            (
                f"[{spec.section}]",
                f"version = {json.dumps(version)}",
                f"sha256 = {json.dumps(digest)}",
                f"url = {json.dumps(immutable_url(repository, version, spec.filename))}",
            )
        )
        if environment is not None:
            lines.append(f"environment = {json.dumps(environment)}")
        lines.append("")

    if plugin_versions:
        lines.extend(
            (
                "# =================================================================",
                "# Plugins section",
                "# =================================================================",
                "",
            )
        )
        for name in ("annotations", "gamepad"):
            if name not in plugin_versions:
                continue
            plugin_repository, plugin_name, description = (
                official_metadata or PLUGIN_METADATA
            )[name]
            lines.extend(
                (
                    f"[plugins.{name}]",
                    f"name = {json.dumps(plugin_name)}",
                    f"repository = {json.dumps(plugin_repository)}",
                    f"version = {json.dumps(plugin_versions[name])}",
                    f"description = {json.dumps(description)}",
                )
            )
            release = (plugin_releases or {}).get(name)
            if release is not None:
                lines.extend(
                    (
                        f"release_tag = {json.dumps(release.release_tag)}",
                        f"environment = {json.dumps(release.environment)}",
                    )
                )
            lines.append("")
            if release is not None:
                for section, artifact in release.artifacts.items():
                    lines.extend(
                        (
                            f"[plugins.{name}.{section}]",
                            f"version = {json.dumps(release.version)}",
                            f"sha256 = {json.dumps(artifact['sha256'])}",
                            f"url = {json.dumps(artifact['url'])}",
                            "",
                        )
                    )
        while lines and lines[-1] == "":
            lines.pop()
        lines.append("")
    return "\n".join(lines)


def validate_manifest_data(
    data: dict[str, object],
    version: str,
    repository: str,
    specs: tuple[ArtifactSpec, ...],
    environment: str | None = None,
    kind: str = "eov",
) -> None:
    validate_version(version)
    metadata = data.get("manifest")
    if metadata is not None:
        if not isinstance(metadata, dict) or metadata.get("schema") != 1:
            raise ReleaseError("manifest metadata must declare schema = 1")
        expected_kind = kind
        if metadata.get("kind") != expected_kind:
            raise ReleaseError("manifest metadata has the wrong kind")
        if metadata.get("version") != version:
            raise ReleaseError("manifest metadata has the wrong version")
        expected_repository = f"https://github.com/{repository}"
        if metadata.get("repository") != expected_repository:
            raise ReleaseError("manifest metadata has the wrong repository")
    for spec in specs:
        entry = nested_table(data, spec.section)
        if entry.get("version") != version:
            raise ReleaseError(f"[{spec.section}] has the wrong version")
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str):
            raise ReleaseError(f"[{spec.section}] has no sha256")
        validate_sha256(sha256, f"[{spec.section}].sha256")
        expected_url = immutable_url(repository, version, spec.filename)
        if entry.get("url") != expected_url:
            raise ReleaseError(
                f"[{spec.section}] does not use its immutable release URL"
            )
        if environment is not None and entry.get("environment") != environment:
            raise ReleaseError(f"[{spec.section}] has the wrong plugin environment")
    if kind == "eov":
        plugins = data.get("plugins", {})
        if not isinstance(plugins, dict):
            raise ReleaseError("[plugins] must be a table")
        for plugin_id, plugin in plugins.items():
            if not isinstance(plugin_id, str) or not isinstance(plugin, dict):
                raise ReleaseError("official plugin catalog entries must be tables")
            required_fields = ["repository", "version"]
            if isinstance(metadata, dict):
                required_fields.extend(("name", "description"))
            for field in required_fields:
                if not isinstance(plugin.get(field), str) or not plugin[field]:
                    raise ReleaseError(
                        f"official plugin catalog entry '{plugin_id}' is missing {field}"
                    )
            validate_version(plugin["version"])
            plugin_environment = plugin.get("environment")
            if plugin_environment is not None:
                if not isinstance(plugin_environment, str) or not requirement_is_bounded(
                    plugin_environment
                ):
                    raise ReleaseError(
                        f"official plugin catalog entry '{plugin_id}' needs a bounded environment"
                    )
            plugin_tag = plugin.get("release_tag", f"v{plugin['version']}")
            platforms = plugin.get("platform", {})
            if not isinstance(platforms, dict):
                raise ReleaseError(f"official plugin '{plugin_id}' platform entries must be tables")
            for section in PLUGIN_PLATFORM_SECTIONS:
                try:
                    artifact = nested_table({"platform": platforms}, section)
                except ReleaseError:
                    continue
                if artifact.get("version") != plugin["version"]:
                    raise ReleaseError(
                        f"official plugin '{plugin_id}' [{section}] has the wrong version"
                    )
                validate_sha256(
                    str(artifact.get("sha256")), f"plugin '{plugin_id}' [{section}].sha256"
                )
                expected_prefix = f"{plugin['repository']}/releases/download/{plugin_tag}/"
                if not str(artifact.get("url", "")).startswith(expected_prefix):
                    raise ReleaseError(
                        f"official plugin '{plugin_id}' [{section}] URL is not part of "
                        f"release {plugin_tag}"
                    )
            repository_url = urllib.parse.urlparse(plugin["repository"])
            if (
                repository_url.scheme != "https"
                or repository_url.netloc != "github.com"
                or len([part for part in repository_url.path.split("/") if part]) != 2
            ):
                raise ReleaseError(
                    f"official plugin catalog entry '{plugin_id}' has an invalid repository"
                )
    elif kind == "plugin" and isinstance(metadata, dict):
        plugin = data.get("plugin")
        if not isinstance(plugin, dict):
            raise ReleaseError("schema-1 plugin manifest is missing [plugin]")
        for field in ("id", "name", "repository", "version", "environment"):
            if not isinstance(plugin.get(field), str) or not plugin[field]:
                raise ReleaseError(f"[plugin] is missing {field}")
        if plugin["version"] != version or plugin["environment"] != environment:
            raise ReleaseError("[plugin] metadata does not match the release")
        if plugin["repository"] != f"https://github.com/{repository}":
            raise ReleaseError(
                "[plugin] repository does not match the release repository"
            )


def load_and_validate_manifest(
    path: Path,
    version: str,
    repository: str,
    kind: str,
    plugin_name: str | None = None,
    plugin_toml: Path | None = None,
) -> tuple[dict[str, object], tuple[ArtifactSpec, ...]]:
    data = read_toml(path)
    specs = artifact_specs(kind, plugin_name, version)
    environment = None
    if plugin_toml is not None:
        plugin_data = read_toml(plugin_toml)
        environment_data = plugin_data.get("environment")
        if isinstance(environment_data, dict) and isinstance(
            environment_data.get("version"), str
        ):
            environment = environment_data["version"]
    validate_manifest_data(data, version, repository, specs, environment, kind)
    return data, specs


def parse_plugin_versions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or name not in PLUGIN_METADATA:
            raise ReleaseError(f"plugin version must be NAME=VERSION: {value!r}")
        validate_version(version)
        result[name] = version
    return result


def command_manifest(args: argparse.Namespace) -> None:
    version = validate_version(args.version)
    specs = artifact_specs(args.kind, args.plugin_name, version)
    environment = None
    plugin_metadata = None
    official_metadata = None
    if args.kind == "plugin":
        if args.plugin_toml is None:
            raise ReleaseError("plugin manifest generation requires --plugin-toml")
        plugin_data = read_toml(args.plugin_toml)
        if plugin_data.get("version") != version:
            raise ReleaseError(
                "plugin.toml version does not match the requested release"
            )
        environment_data = plugin_data.get("environment")
        if not isinstance(environment_data, dict) or not isinstance(
            environment_data.get("version"), str
        ):
            raise ReleaseError("plugin.toml has no valid [environment].version")
        plugin_id = plugin_data.get("id")
        plugin_name = plugin_data.get("name")
        if not isinstance(plugin_id, str) or not plugin_id:
            raise ReleaseError("plugin.toml has no valid id")
        if not isinstance(plugin_name, str) or not plugin_name:
            raise ReleaseError("plugin.toml has no valid name")
        environment = environment_data["version"]
        plugin_id = plugin_data.get("id")
        plugin_name = plugin_data.get("name")
        description = plugin_data.get("description")
        plugin_repository = plugin_data.get("repository")
        if not isinstance(plugin_id, str) or not isinstance(plugin_name, str):
            raise ReleaseError("plugin.toml must define id and name")
        if description is not None and not isinstance(description, str):
            raise ReleaseError("plugin.toml description must be a string")
        if plugin_repository is not None and not isinstance(plugin_repository, str):
            raise ReleaseError("plugin.toml repository must be a string")
        plugin_metadata = (plugin_id, plugin_name, description, plugin_repository)
    elif args.kind == "eov":
        official_metadata = {}
        for plugin_id in PLUGIN_METADATA:
            plugin_data = read_toml(args.root / "plugins" / plugin_id / "plugin.toml")
            plugin_repository = plugin_data.get("repository")
            plugin_name = plugin_data.get("name")
            description = plugin_data.get("description")
            if not all(
                isinstance(value, str) and value
                for value in (plugin_repository, plugin_name, description)
            ):
                raise ReleaseError(
                    f"official plugin {plugin_id} metadata is incomplete in plugin.toml"
                )
            official_metadata[plugin_id] = (
                plugin_repository,
                plugin_name,
                description,
            )

    staged = stage_artifacts(args.artifacts_dir, args.staging_dir, specs)
    plugin_versions = parse_plugin_versions(args.plugin_version)
    plugin_releases: dict[str, PluginRelease] = {}
    for value in args.plugin_manifest:
        name, separator, path = value.partition("=")
        if not separator or name not in plugin_versions:
            raise ReleaseError(
                f"--plugin-manifest must be NAME=PATH for a --plugin-version plugin: {value!r}"
            )
        plugin_releases[name] = read_plugin_release(
            Path(path),
            name,
            (official_metadata or PLUGIN_METADATA)[name][0],
            plugin_versions[name],
            version,
        )
    content = render_manifest(
        version,
        args.repository,
        specs,
        staged,
        plugin_versions if args.kind == "eov" else None,
        environment,
        plugin_metadata,
        official_metadata,
        plugin_releases,
        args.signed,
    )
    write_atomically(args.output, content)
    data, _ = load_and_validate_manifest(
        args.output,
        version,
        args.repository,
        args.kind,
        args.plugin_name,
        args.plugin_toml,
    )
    for spec in specs:
        entry = nested_table(data, spec.section)
        actual = hashlib.sha256(staged[spec.filename].read_bytes()).hexdigest()
        if entry["sha256"] != actual:
            raise ReleaseError(
                f"[{spec.section}] hash does not match the staged artifact"
            )
    print(f"generated {args.output}")


def _macos_symbol_for(minimum_version: object, source: str) -> str:
    if not isinstance(minimum_version, str):
        raise ReleaseError(f"{source} has no LSMinimumSystemVersion")
    parts = minimum_version.split(".")
    major = parts[0]
    key = f"{major}.0" if major != "10" else ".".join(parts[:2])
    if key not in MACOS_CASK_SYMBOLS:
        raise ReleaseError(f"unsupported LSMinimumSystemVersion {minimum_version!r} in {source}")
    return key


def macos_cask_symbol(info_plist_path: Path, app_archives: tuple[Path, ...] = ()) -> str:
    """The cask's ``depends_on macos:`` symbol.

    With ``app_archives`` (the built macOS zips), the highest
    ``LSMinimumSystemVersion`` of the bundles that are actually shipped wins,
    so the cask never promises an older macOS than the binaries support.
    """

    keys = []
    for archive in app_archives:
        try:
            with zipfile.ZipFile(archive) as bundle:
                info_plist = plistlib.loads(bundle.read("eov.app/Contents/Info.plist"))
        except (
            OSError,
            KeyError,
            ValueError,
            zipfile.BadZipFile,
            plistlib.InvalidFileException,
        ) as error:
            raise ReleaseError(f"could not read Info.plist from {archive}: {error}") from error
        if not isinstance(info_plist, dict):
            raise ReleaseError(f"{archive} has an invalid Info.plist")
        keys.append(_macos_symbol_for(info_plist.get("LSMinimumSystemVersion"), str(archive)))
    if not keys:
        try:
            with info_plist_path.open("rb") as plist_file:
                info_plist = plistlib.load(plist_file)
        except (OSError, ValueError, plistlib.InvalidFileException) as error:
            raise ReleaseError(f"could not read {info_plist_path}: {error}") from error
        if not isinstance(info_plist, dict):
            raise ReleaseError(f"{info_plist_path} is not a property list object")
        keys.append(
            _macos_symbol_for(
                info_plist.get("LSMinimumSystemVersion"), str(info_plist_path)
            )
        )
    key = max(keys, key=lambda value: tuple(int(part) for part in value.split(".")))
    return MACOS_CASK_SYMBOLS[key]


def cask_content(
    token: str,
    version: str,
    arm64_sha256: str,
    arm64_url: str,
    x86_64_sha256: str,
    x86_64_url: str,
    macos_symbol: str,
) -> str:
    zap_lines = "\n".join(f"        {json.dumps(path)}," for path in CASK_ZAP_PATHS)
    return f"""cask "{token}" do
    version "{version}"

    on_arm do
        sha256 "{arm64_sha256}"

        url "{arm64_url}"
    end
    on_intel do
        sha256 "{x86_64_sha256}"

        url "{x86_64_url}"
    end

    name "eov"
    desc "Lightweight, cross-platform Whole Slide Image (WSI) viewer for digital pathology"
    homepage "https://eov.sh/"

    depends_on macos: :{macos_symbol}

    app "eov.app"
    binary "#{{appdir}}/eov.app/Contents/MacOS/eov"

    zap trash: [
{zap_lines}
    ]

    caveats <<~EOS
        eov is not notarized by Apple. If macOS prevents it from opening, go to:
            System Settings \u2192 Privacy & Security \u2192 Open Anyway
    EOS
end
"""


def cask_values(data: dict[str, object], version: str) -> tuple[str, str, str, str]:
    arm = nested_table(data, "platform.macos.arm64")
    intel = nested_table(data, "platform.macos.x86_64")
    arm_sha = arm.get("sha256")
    intel_sha = intel.get("sha256")
    arm_url = arm.get("url")
    intel_url = intel.get("url")
    if not all(
        isinstance(value, str) for value in (arm_sha, intel_sha, arm_url, intel_url)
    ):
        raise ReleaseError("manifest is missing macOS cask values")
    validate_sha256(arm_sha, "macOS ARM SHA-256")
    validate_sha256(intel_sha, "macOS Intel SHA-256")
    expected_arm = f"eov-v{version}-macos-arm64.zip"
    expected_intel = f"eov-v{version}-macos-x86_64.zip"
    if urllib.parse.urlparse(arm_url).path.rsplit("/", 1)[-1] != expected_arm:
        raise ReleaseError("macOS ARM cask URL has the wrong filename")
    if urllib.parse.urlparse(intel_url).path.rsplit("/", 1)[-1] != expected_intel:
        raise ReleaseError("macOS Intel cask URL has the wrong filename")
    return arm_sha, arm_url, intel_sha, intel_url


def write_historical(path: Path, content: str) -> bool:
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ReleaseError(
                f"could not read historical cask {path}: {error}"
            ) from error
        if existing == content:
            return False
        raise ReleaseError(f"historical cask differs from deterministic output: {path}")
    write_atomically(path, content)
    return True


def command_cask(args: argparse.Namespace) -> None:
    version = validate_version(args.version, stable_only=True)
    data, _ = load_and_validate_manifest(
        args.manifest,
        version,
        EOV_REPOSITORY,
        "eov",
    )
    arm_sha, arm_url, intel_sha, intel_url = cask_values(data, version)
    symbol = macos_cask_symbol(args.info_plist, tuple(args.app_archive))
    casks_dir = args.tap_root / "Casks"
    casks_dir.mkdir(parents=True, exist_ok=True)
    exact_path = casks_dir / f"eov@{version}.rb"
    exact_content = cask_content(
        f"eov@{version}", version, arm_sha, arm_url, intel_sha, intel_url, symbol
    )
    historical_changed = write_historical(exact_path, exact_content)
    latest_path = casks_dir / "eov.rb"
    latest_content = cask_content(
        "eov", version, arm_sha, arm_url, intel_sha, intel_url, symbol
    )
    latest_changed = (
        not latest_path.exists()
        or latest_path.read_text(encoding="utf-8") != latest_content
    )
    if latest_changed:
        write_atomically(latest_path, latest_content)
    print(
        f"casks: historical={'updated' if historical_changed else 'unchanged'}, "
        f"latest={'updated' if latest_changed else 'unchanged'}"
    )


def validate_cask_file(
    path: Path,
    expected_token: str,
    expected_version: str,
    expected_values: tuple[str, str, str, str],
) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(f"could not read cask {path}: {error}") from error
    token_match = re.search(r'^cask "([^"]+)" do$', content, re.MULTILINE)
    version_match = re.search(r'^    version "([^"]+)"$', content, re.MULTILINE)
    if token_match is None or token_match.group(1) != expected_token:
        raise ReleaseError(f"{path} has the wrong cask token")
    if version_match is None or version_match.group(1) != expected_version:
        raise ReleaseError(f"{path} has the wrong cask version")
    arm_sha, arm_url, intel_sha, intel_url = expected_values
    for architecture, expected_sha, expected_url in (
        ("arm", arm_sha, arm_url),
        ("intel", intel_sha, intel_url),
    ):
        block_match = re.search(
            rf"on_{architecture} do(?P<block>.*?)^    end$",
            content,
            re.MULTILINE | re.DOTALL,
        )
        if block_match is None:
            raise ReleaseError(f"{path} has no on_{architecture} block")
        block = block_match.group("block")
        sha_match = re.search(r'^        sha256 "([^"]+)"$', block, re.MULTILINE)
        url_match = re.search(r'^        url "([^"]+)"$', block, re.MULTILINE)
        if sha_match is None or sha_match.group(1) != expected_sha:
            raise ReleaseError(f"{path} has the wrong {architecture} SHA-256")
        if url_match is None or url_match.group(1) != expected_url:
            raise ReleaseError(f"{path} has the wrong {architecture} URL")


def command_validate_casks(args: argparse.Namespace) -> None:
    version = validate_version(args.version, stable_only=True)
    data, _ = load_and_validate_manifest(args.manifest, version, EOV_REPOSITORY, "eov")
    values = cask_values(data, version)
    validate_cask_file(args.tap_root / "Casks/eov.rb", "eov", version, values)
    validate_cask_file(
        args.tap_root / f"Casks/eov@{version}.rb",
        f"eov@{version}",
        version,
        values,
    )
    print("validated Homebrew casks")


def readme_block(text: str) -> tuple[int, int, str]:
    start = text.find(EOV_MANIFEST_START)
    end = text.find(EOV_MANIFEST_END)
    if start < 0 or end < 0 or end <= start:
        raise ReleaseError(
            "README is missing a valid release-install-links marker block"
        )
    block_start = start + len(EOV_MANIFEST_START)
    return block_start, end, text[block_start:end]


def readme_artifact_url(filename: str, version: str, urls: dict[str, str]) -> str:
    suffixes = {name.removeprefix(f"eov-v{version}"): url for name, url in urls.items()}
    if filename.startswith(f"eov-v{version}") and filename in urls:
        return urls[filename]
    if not filename.startswith("eov-v"):
        raise ReleaseError(
            f"README download filename is not an EOV release artifact: {filename}"
        )
    matches = []
    for suffix, url in suffixes.items():
        if not filename.endswith(suffix):
            continue
        old_version = filename[len("eov-v") : -len(suffix)]
        if STABLE_VERSION_RE.fullmatch(old_version or "") or SEMVER_RE.fullmatch(
            old_version or ""
        ):
            matches.append(url)
    if len(matches) != 1:
        raise ReleaseError(
            f"could not map README artifact filename {filename!r} to the release manifest"
        )
    return matches[0]


def transform_readme(text: str, version: str, data: dict[str, object]) -> str:
    block_start, block_end, block = readme_block(text)
    urls: dict[str, str] = {}
    for section in (
        "platform.windows.x86_64",
        "platform.windows.arm64",
        "platform.macos.arm64",
        "platform.macos.x86_64",
        "platform.linux.appimage.x86_64",
        "platform.linux.appimage.arm64",
        "platform.linux.flatpak.x86_64",
        "platform.linux.flatpak.arm64",
    ):
        entry = nested_table(data, section)
        url = entry.get("url")
        if not isinstance(url, str):
            raise ReleaseError(f"[{section}] has no URL")
        urls[urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]] = url

    def replace_url(match: re.Match[str]) -> str:
        url = match.group(0)
        parsed = urllib.parse.urlparse(url)
        if "/releases/tag/" in parsed.path:
            return f"https://github.com/{EOV_REPOSITORY}/releases/tag/v{version}"
        if "/releases/download/" not in parsed.path:
            raise ReleaseError(f"README contains an unsupported release URL: {url}")
        filename = parsed.path.rsplit("/", 1)[-1]
        return readme_artifact_url(filename, version, urls)

    transformed_block = README_URL_RE.sub(replace_url, block)
    transformed_block, replacements = re.subn(
        r"(brew\s+install\s+--cask\s+eov@)([0-9]+\.[0-9]+\.[0-9]+)",
        rf"\g<1>{version}",
        transformed_block,
    )
    if replacements == 0:
        raise ReleaseError("README marker block has no exact-version Homebrew example")
    return text[:block_start] + transformed_block + text[block_end:]


def validate_readme_text(text: str, version: str, data: dict[str, object]) -> None:
    _, _, block = readme_block(text)
    expected_urls = {}
    for section in (
        "platform.windows.x86_64",
        "platform.windows.arm64",
        "platform.macos.arm64",
        "platform.macos.x86_64",
        "platform.linux.appimage.x86_64",
        "platform.linux.appimage.arm64",
        "platform.linux.flatpak.x86_64",
        "platform.linux.flatpak.arm64",
    ):
        entry = nested_table(data, section)
        url = entry.get("url")
        if not isinstance(url, str):
            raise ReleaseError(f"[{section}] has no URL")
        expected_urls[url] = section
    found_urls = README_URL_RE.findall(block)
    if not found_urls:
        raise ReleaseError("README marker block has no release URLs")
    for url in found_urls:
        parsed = urllib.parse.urlparse(url)
        if "/releases/tag/" in parsed.path:
            if url != f"https://github.com/{EOV_REPOSITORY}/releases/tag/v{version}":
                raise ReleaseError(f"README release page does not target v{version}")
        elif url not in expected_urls:
            raise ReleaseError(
                f"README contains an unexpected or stale release URL: {url}"
            )
    if (
        re.search(rf"brew\s+install\s+--cask\s+eov@{re.escape(version)}\b", block)
        is None
    ):
        raise ReleaseError(
            f"README exact-version Homebrew example does not target {version}"
        )


def command_update_readme(args: argparse.Namespace) -> None:
    version = validate_version(args.version, stable_only=True)
    data, _ = load_and_validate_manifest(args.manifest, version, EOV_REPOSITORY, "eov")
    try:
        original = args.readme.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(f"could not read README {args.readme}: {error}") from error
    updated = transform_readme(original, version, data)
    validate_readme_text(updated, version, data)
    if updated != original:
        write_atomically(args.readme, updated)
    print(f"README {'updated' if updated != original else 'unchanged'}")


def command_validate_readme(args: argparse.Namespace) -> None:
    version = validate_version(args.version, stable_only=True)
    data, _ = load_and_validate_manifest(args.manifest, version, EOV_REPOSITORY, "eov")
    try:
        text = args.readme.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(f"could not read README {args.readme}: {error}") from error
    validate_readme_text(text, version, data)
    print("validated README release links")


def command_verify_assets(args: argparse.Namespace) -> None:
    version = validate_version(args.version)
    data, specs = load_and_validate_manifest(
        args.manifest,
        version,
        args.repository,
        args.kind,
        args.plugin_name,
        args.plugin_toml,
    )
    for spec in specs:
        entry = nested_table(data, spec.section)
        url = entry["url"]
        expected_sha = entry["sha256"]
        request = urllib.request.Request(
            url, headers={"User-Agent": "eov-release-verifier"}
        )
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ReleaseError(
                        f"asset URL returned HTTP {response.status}: {url}"
                    )
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
        except (OSError, urllib.error.URLError) as error:
            raise ReleaseError(
                f"could not verify release asset {url}: {error}"
            ) from error
        if digest.hexdigest() != expected_sha:
            raise ReleaseError(f"release asset hash mismatch for {url}")
    print(f"verified {len(specs)} published assets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EOV release manifest and installation metadata tooling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_source = subparsers.add_parser("validate-source")
    validate_source.add_argument("--kind", choices=("eov", "plugin"), required=True)
    validate_source.add_argument("--root", type=Path, required=True)
    validate_source.add_argument("--version", required=True)
    validate_source.set_defaults(
        function=lambda args: (
            validate_eov_source(args.root, args.version)
            if args.kind == "eov"
            else validate_plugin_source(args.root, args.version)
        )
    )

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--kind", choices=("eov", "plugin"), required=True)
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--artifacts-dir", type=Path, required=True)
    manifest.add_argument("--staging-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--plugin-name", choices=tuple(PLUGIN_ARTIFACTS))
    manifest.add_argument("--plugin-toml", type=Path)
    manifest.add_argument("--plugin-version", action="append", default=[])
    manifest.add_argument(
        "--plugin-manifest",
        action="append",
        default=[],
        help="NAME=PATH of the plugin's published release.toml; pins its per-platform hashes",
    )
    manifest.add_argument(
        "--signed",
        action="store_true",
        help="declare that release.toml.minisig will be published",
    )
    manifest.set_defaults(function=command_manifest)

    cask = subparsers.add_parser("generate-cask")
    cask.add_argument("--version", required=True)
    cask.add_argument("--manifest", type=Path, required=True)
    cask.add_argument("--tap-root", type=Path, required=True)
    cask.add_argument("--info-plist", type=Path, required=True)
    cask.add_argument(
        "--app-archive",
        type=Path,
        action="append",
        default=[],
        help="built macOS zip; its Info.plist decides depends_on macos",
    )
    cask.set_defaults(function=command_cask)

    validate_casks = subparsers.add_parser("validate-casks")
    validate_casks.add_argument("--version", required=True)
    validate_casks.add_argument("--manifest", type=Path, required=True)
    validate_casks.add_argument("--tap-root", type=Path, required=True)
    validate_casks.set_defaults(function=command_validate_casks)

    update_readme = subparsers.add_parser("update-readme")
    update_readme.add_argument("--version", required=True)
    update_readme.add_argument("--manifest", type=Path, required=True)
    update_readme.add_argument("--readme", type=Path, required=True)
    update_readme.set_defaults(function=command_update_readme)

    validate_readme = subparsers.add_parser("validate-readme")
    validate_readme.add_argument("--version", required=True)
    validate_readme.add_argument("--manifest", type=Path, required=True)
    validate_readme.add_argument("--readme", type=Path, required=True)
    validate_readme.set_defaults(function=command_validate_readme)

    verify_assets = subparsers.add_parser("verify-assets")
    verify_assets.add_argument("--kind", choices=("eov", "plugin"), required=True)
    verify_assets.add_argument("--version", required=True)
    verify_assets.add_argument("--repository", required=True)
    verify_assets.add_argument("--manifest", type=Path, required=True)
    verify_assets.add_argument("--plugin-name", choices=tuple(PLUGIN_ARTIFACTS))
    verify_assets.add_argument("--plugin-toml", type=Path)
    verify_assets.set_defaults(function=command_verify_assets)

    resolve_plugins = subparsers.add_parser(
        "resolve-plugin-versions",
        help="select newest stable official plugin releases compatible with an EOV version",
    )
    resolve_plugins.add_argument("--eov-version", required=True)
    resolve_plugins.add_argument("--plugin", action="append", default=[])
    resolve_plugins.set_defaults(function=command_resolve_plugin_versions)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
