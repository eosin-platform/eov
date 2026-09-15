#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Platform:
    section: str
    asset_template: str


@dataclass(frozen=True)
class Plugin:
    name: str
    repo: str
    repository_url: str
    description: str


class ReleaseError(Exception):
    pass


EOV_REPOSITORY = "eosin-platform/eov"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "release.toml"
CASK_OUTPUT_PATH = OUTPUT_PATH.parent / "Casks" / "eov.rb"

PLATFORMS = (
    Platform("platform.windows.x86_64", "eov-v{version}-windows-x86_64.zip"),
    Platform("platform.windows.arm64", "eov-v{version}-windows-arm64.zip"),
    Platform(
        "platform.linux.appimage.arm64",
        "eov-v{version}-linux-arm64.AppImage",
    ),
    Platform(
        "platform.linux.flatpak.arm64",
        "eov-v{version}-linux-arm64.flatpak",
    ),
    Platform(
        "platform.linux.appimage.x86_64",
        "eov-v{version}-linux-x86_64.AppImage",
    ),
    Platform(
        "platform.linux.flatpak.x86_64",
        "eov-v{version}-linux-x86_64.flatpak",
    ),
    Platform("platform.macos.arm64", "eov-v{version}-macos-arm64.zip"),
    Platform("platform.macos.x86_64", "eov-v{version}-macos-x86_64.zip"),
)

PLUGINS = (
    Plugin(
        name="annotations",
        repo="eosin-platform/eov-annotations-plugin",
        repository_url="https://github.com/eosin-platform/eov-annotations-plugin",
        description="Official annotations plugin",
    ),
    Plugin(
        name="gamepad",
        repo="eosin-platform/eov-gamepad-plugin",
        repository_url="https://github.com/eosin-platform/eov-gamepad-plugin",
        description="Provides support for using gamepads as input devices",
    ),
)


def fetch_latest_release(gh_path: str, repository: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            [gh_path, "api", f"repos/{repository}/releases/latest"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ReleaseError(
            "GitHub CLI (`gh`) is required to generate release.toml."
        ) from error
    except subprocess.CalledProcessError as error:
        details = "\n".join(
            output.strip()
            for output in (error.stderr, error.stdout)
            if output and output.strip()
        )
        if "404" in details or "not found" in details.lower():
            raise ReleaseError(
                f"Repository {repository} has no accessible latest release."
            ) from error
        if details:
            raise ReleaseError(
                f"GitHub API request for {repository} failed: {details}"
            ) from error
        raise ReleaseError(
            f"GitHub API request for {repository} failed with exit code "
            f"{error.returncode}."
        ) from error

    try:
        release = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"GitHub API returned malformed JSON for {repository}: {error}"
        ) from error

    if not isinstance(release, dict):
        raise ReleaseError(
            f"GitHub API returned unexpected JSON for {repository}; expected an object."
        )
    return release


def release_version(release: dict[str, object], repository: str) -> str:
    tag = release.get("tag_name")
    if (
        not isinstance(tag, str)
        or not tag
        or tag != tag.strip()
        or any(character.isspace() for character in tag)
    ):
        raise ReleaseError(
            f"Latest release for {repository} has an invalid or missing tag_name."
        )

    version = tag[1:] if tag.startswith("v") else tag
    if not version:
        raise ReleaseError(
            f"Latest release for {repository} has an invalid release tag: {tag!r}."
        )
    return version


def platform_releases(
    release: dict[str, object], version: str, repository: str
) -> list[tuple[Platform, str, str]]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ReleaseError(
            f"Latest release for {repository} has malformed or missing assets."
        )

    assets_by_name: dict[str, dict[str, object]] = {}
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ReleaseError(
                f"Latest release for {repository} has malformed asset at index {index}."
            )
        name = asset.get("name")
        if not isinstance(name, str) or not name:
            raise ReleaseError(
                f"Latest release for {repository} has an asset with an invalid name."
            )
        if name in assets_by_name:
            raise ReleaseError(
                f"Latest release for {repository} contains duplicate asset {name!r}."
            )
        assets_by_name[name] = asset

    discovered: list[tuple[Platform, str, str]] = []
    for platform in PLATFORMS:
        expected_name = platform.asset_template.format(version=version)
        asset = assets_by_name.get(expected_name)
        if asset is None:
            raise ReleaseError(
                f"Latest {repository} release is missing required asset "
                f"{expected_name!r}."
            )

        download_url = asset.get("browser_download_url")
        if not isinstance(download_url, str) or not download_url.strip():
            raise ReleaseError(
                f"Asset {expected_name!r} is missing browser_download_url."
            )

        digest = asset.get("digest")
        if not isinstance(digest, str) or not digest:
            raise ReleaseError(f"Asset {expected_name!r} is missing a digest.")
        if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) is None:
            raise ReleaseError(
                f"Asset {expected_name!r} does not provide a SHA-256 digest."
            )

        discovered.append((platform, digest, download_url))

    return discovered


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_release_toml(
    version: str,
    platforms: list[tuple[Platform, str, str]],
    plugin_versions: dict[str, str],
) -> str:
    lines: list[str] = []
    for platform, digest, download_url in platforms:
        lines.extend(
            (
                f"[{platform.section}]",
                f"version = {toml_string(version)}",
                f"sha256 = {toml_string(digest.removeprefix('sha256:'))}",
                f"url = {toml_string(download_url)}",
                "",
            )
        )

    lines.extend(
        (
            "# =================================================================",
            "# Plugins section",
            "# =================================================================",
            "",
        )
    )

    for index, plugin in enumerate(PLUGINS):
        lines.extend(
            (
                f"[plugins.{plugin.name}]",
                f"repository = {toml_string(plugin.repository_url)}",
                f"version = {toml_string(plugin_versions[plugin.name])}",
                f"description = {toml_string(plugin.description)}",
            )
        )
        if index != len(PLUGINS) - 1:
            lines.append("")

    return "\n".join(lines) + "\n"


def render_cask(version: str, platforms: list[tuple[Platform, str, str]]) -> str:
    assets = {
        platform.section: (digest.removeprefix("sha256:"), download_url)
        for platform, digest, download_url in platforms
    }
    try:
        arm64_sha256, arm64_url = assets["platform.macos.arm64"]
        x86_64_sha256, x86_64_url = assets["platform.macos.x86_64"]
    except KeyError as error:
        raise ReleaseError(
            "Latest eov release is missing a required macOS cask asset."
        ) from error

    return f"""cask "eov" do
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

    depends_on macos: :big_sur

    app "eov.app"
    binary "#{{appdir}}/eov.app/Contents/MacOS/eov"

    zap trash: [
        "~/Library/Application Support/io.eosin.eov",
        "~/Library/Caches/io.eosin.eov",
        "~/Library/Preferences/io.eosin.eov.plist",
    ]

    caveats <<~EOS
        eov is not notarized by Apple. If macOS prevents it from opening, go to:
            System Settings → Privacy & Security → Open Anyway
    EOS
end
"""


def write_atomically(path: Path, content: str) -> None:
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
        raise ReleaseError(f"Could not write {path}: {error}") from error


def main() -> int:
    gh_path = shutil.which("gh")
    if gh_path is None:
        print(
            "error: GitHub CLI (`gh`) is required to generate release.toml.",
            file=sys.stderr,
        )
        return 1

    try:
        eov_release = fetch_latest_release(gh_path, EOV_REPOSITORY)
        eov_version = release_version(eov_release, EOV_REPOSITORY)
        platforms = platform_releases(eov_release, eov_version, EOV_REPOSITORY)

        plugin_versions: dict[str, str] = {}
        for plugin in PLUGINS:
            release = fetch_latest_release(gh_path, plugin.repo)
            plugin_versions[plugin.name] = release_version(release, plugin.repo)

        content = render_release_toml(eov_version, platforms, plugin_versions)
        write_atomically(OUTPUT_PATH, content)
        write_atomically(CASK_OUTPUT_PATH, render_cask(eov_version, platforms))
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Generated {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
