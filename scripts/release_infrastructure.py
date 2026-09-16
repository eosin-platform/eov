#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
        "Official annotations plugin",
    ),
    "gamepad": (
        "https://github.com/eosin-platform/eov-gamepad-plugin",
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
}


def validate_version(version: str, *, stable_only: bool = False) -> str:
    pattern = STABLE_VERSION_RE if stable_only else SEMVER_RE
    if pattern.fullmatch(version) is None:
        qualifier = "stable " if stable_only else ""
        raise ReleaseError(f"{qualifier}version must be valid SemVer: {version!r}")
    return version


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
    environment = plugin_data.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("version"), str
    ):
        raise ReleaseError("plugin.toml has no valid [environment].version")


def render_manifest(
    version: str,
    repository: str,
    specs: tuple[ArtifactSpec, ...],
    staged: dict[str, Path],
    plugin_versions: dict[str, str] | None = None,
    environment: str | None = None,
) -> str:
    lines: list[str] = []
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
            plugin_repository, description = PLUGIN_METADATA[name]
            lines.extend(
                (
                    f"[plugins.{name}]",
                    f"repository = {json.dumps(plugin_repository)}",
                    f"version = {json.dumps(plugin_versions[name])}",
                    f"description = {json.dumps(description)}",
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
) -> None:
    validate_version(version)
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
    validate_manifest_data(data, version, repository, specs, environment)
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
        environment = environment_data["version"]

    staged = stage_artifacts(args.artifacts_dir, args.staging_dir, specs)
    plugin_versions = parse_plugin_versions(args.plugin_version)
    content = render_manifest(
        version,
        args.repository,
        specs,
        staged,
        plugin_versions if args.kind == "eov" else None,
        environment,
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


def macos_cask_symbol(info_plist_path: Path) -> str:
    try:
        with info_plist_path.open("rb") as plist_file:
            info_plist = plistlib.load(plist_file)
    except (OSError, ValueError, plistlib.InvalidFileException) as error:
        raise ReleaseError(f"could not read {info_plist_path}: {error}") from error
    if not isinstance(info_plist, dict):
        raise ReleaseError(f"{info_plist_path} is not a property list object")
    minimum_version = info_plist.get("LSMinimumSystemVersion")
    if (
        not isinstance(minimum_version, str)
        or minimum_version not in MACOS_CASK_SYMBOLS
    ):
        raise ReleaseError(f"unsupported LSMinimumSystemVersion: {minimum_version!r}")
    return MACOS_CASK_SYMBOLS[minimum_version]


def cask_content(
    token: str,
    version: str,
    arm64_sha256: str,
    arm64_url: str,
    x86_64_sha256: str,
    x86_64_url: str,
    macos_symbol: str,
) -> str:
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
        "~/Library/Application Support/io.eosin.eov",
        "~/Library/Caches/io.eosin.eov",
        "~/Library/Preferences/io.eosin.eov.plist",
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
    symbol = macos_cask_symbol(args.info_plist)
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
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--artifacts-dir", type=Path, required=True)
    manifest.add_argument("--staging-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--plugin-name", choices=tuple(PLUGIN_ARTIFACTS))
    manifest.add_argument("--plugin-toml", type=Path)
    manifest.add_argument("--plugin-version", action="append", default=[])
    manifest.set_defaults(function=command_manifest)

    cask = subparsers.add_parser("generate-cask")
    cask.add_argument("--version", required=True)
    cask.add_argument("--manifest", type=Path, required=True)
    cask.add_argument("--tap-root", type=Path, required=True)
    cask.add_argument("--info-plist", type=Path, required=True)
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
