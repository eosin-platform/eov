#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from release_infrastructure import (
    EOV_REPOSITORY,
    ReleaseError,
    cask_content,
    macos_cask_symbol,
    validate_sha256,
    validate_version,
    write_atomically,
    write_historical,
)

STABLE_TAG_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def fetch_releases(repository: str) -> list[dict[str, object]]:
    gh = shutil.which("gh")
    if gh is None:
        raise ReleaseError(
            "GitHub CLI (`gh`) is required to bootstrap historical casks"
        )
    try:
        result = subprocess.run(
            [gh, "api", f"repos/{repository}/releases?per_page=100"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", "") or str(error)
        raise ReleaseError(
            f"could not query releases for {repository}: {details.strip()}"
        ) from error
    try:
        releases = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"GitHub returned malformed release JSON: {error}"
        ) from error
    if not isinstance(releases, list):
        raise ReleaseError("GitHub release API returned a non-list response")
    return [release for release in releases if isinstance(release, dict)]


def asset_digest(asset: dict[str, object]) -> str:
    digest = asset.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        return validate_sha256(digest.removeprefix("sha256:"), "GitHub asset digest")
    url = asset.get("browser_download_url")
    if not isinstance(url, str) or not url:
        raise ReleaseError("historical macOS asset has no digest or download URL")
    request = urllib.request.Request(
        url, headers={"User-Agent": "eov-homebrew-bootstrap"}
    )
    import hashlib

    digest_builder = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest_builder.update(chunk)
    except (OSError, urllib.error.URLError) as error:
        raise ReleaseError(
            f"could not download historical asset {url}: {error}"
        ) from error
    return digest_builder.hexdigest()


def release_cask_values(
    repository: str, release: dict[str, object], version: str
) -> tuple[str, str, str, str]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ReleaseError("release has no valid assets list")
    expected = {
        f"eov-v{version}-macos-arm64.zip": "arm64",
        f"eov-v{version}-macos-x86_64.zip": "x86_64",
    }
    discovered: dict[str, tuple[str, str]] = {}
    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            continue
        name = raw_asset.get("name")
        if name not in expected:
            continue
        url = raw_asset.get("browser_download_url")
        expected_url = (
            f"https://github.com/{repository}/releases/download/v{version}/{name}"
        )
        if url != expected_url:
            raise ReleaseError(f"historical asset {name!r} has an unexpected URL")
        discovered[expected[name]] = (asset_digest(raw_asset), url)
    missing = sorted(set(expected.values()) - discovered.keys())
    if missing:
        raise ReleaseError(f"missing historical macOS assets: {', '.join(missing)}")
    arm_sha, arm_url = discovered["arm64"]
    intel_sha, intel_url = discovered["x86_64"]
    return arm_sha, arm_url, intel_sha, intel_url


def version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap EOV historical Homebrew casks"
    )
    parser.add_argument("--repository", default=EOV_REPOSITORY)
    parser.add_argument("--tap-root", type=Path, required=True)
    parser.add_argument("--info-plist", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        symbol = macos_cask_symbol(args.info_plist)
        releases = fetch_releases(args.repository)
        generated_versions: list[str] = []
        skipped: list[str] = []
        for release in releases:
            tag = release.get("tag_name")
            if not isinstance(tag, str) or STABLE_TAG_RE.fullmatch(tag) is None:
                continue
            version = tag.removeprefix("v")
            validate_version(version, stable_only=True)
            if release.get("draft") or release.get("prerelease"):
                skipped.append(f"{version}: draft or prerelease")
                continue
            try:
                arm_sha, arm_url, intel_sha, intel_url = release_cask_values(
                    args.repository, release, version
                )
                content = cask_content(
                    f"eov@{version}",
                    version,
                    arm_sha,
                    arm_url,
                    intel_sha,
                    intel_url,
                    symbol,
                )
                write_historical(args.tap_root / "Casks" / f"eov@{version}.rb", content)
                generated_versions.append(version)
            except ReleaseError as error:
                if "historical cask differs" in str(error):
                    raise
                skipped.append(f"{version}: {error}")

        if not generated_versions:
            raise ReleaseError(
                "no stable releases with sufficient macOS assets were found"
            )
        latest = max(generated_versions, key=version_key)
        arm_sha, arm_url, intel_sha, intel_url = release_cask_values(
            args.repository,
            next(
                release
                for release in releases
                if release.get("tag_name") == f"v{latest}"
            ),
            latest,
        )
        latest_content = cask_content(
            "eov", latest, arm_sha, arm_url, intel_sha, intel_url, symbol
        )
        latest_path = args.tap_root / "Casks/eov.rb"
        if (
            not latest_path.exists()
            or latest_path.read_text(encoding="utf-8") != latest_content
        ):
            write_atomically(latest_path, latest_content)
    except (ReleaseError, StopIteration) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"generated historical casks: {', '.join(sorted(generated_versions, key=version_key))}"
    )
    if skipped:
        print("skipped releases:")
        print("\n".join(f"- {item}" for item in skipped))
    else:
        print("skipped releases: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
