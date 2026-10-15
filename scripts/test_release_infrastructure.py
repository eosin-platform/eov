#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
import os
import plistlib
import tempfile
import tomllib
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from release_infrastructure import (
    CASK_ZAP_PATHS,
    EOV_MANIFEST_END,
    EOV_MANIFEST_START,
    EOV_REPOSITORY,
    PLUGIN_METADATA,
    ReleaseError,
    artifact_specs,
    bounded_requirement,
    cask_content,
    host_compatible,
    macos_cask_symbol,
    parse_requirement,
    published_plugin_candidates,
    read_plugin_release,
    render_manifest,
    requirement_is_bounded,
    select_compatible_plugin_release,
    semver_requirement_satisfied,
    stage_artifacts,
    transform_readme,
    validate_manifest_data,
    validate_readme_text,
    validate_version,
    write_historical,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SEMVER_TABLE = REPOSITORY_ROOT / "app/tests/data/semver-cases.json"
FIXTURES = REPOSITORY_ROOT / "app/tests/fixtures"
GENERATED_PLUGIN_FIXTURE = FIXTURES / "generated-gamepad-release.toml"
GENERATED_EOV_FIXTURE = FIXTURES / "generated-eov-catalog-release.toml"
FIXTURE_VERSION = "1.0.1"
FIXTURE_ENVIRONMENT = ">=1.0.0, <2.0.0"


def write_artifacts(source_dir: Path, specs) -> None:
    """Fake build outputs with deterministic contents and checksum sidecars."""

    for index, spec in enumerate(specs):
        source = source_dir / f"job-{index}" / spec.filename
        source.parent.mkdir(parents=True)
        source.write_bytes(f"artifact:{spec.filename}".encode("ascii"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (source.parent / f"{spec.filename}.sha256").write_text(
            f"{digest}  {spec.filename}\n", encoding="ascii"
        )


def generate_release_fixtures(root: Path) -> tuple[str, str]:
    """Run the release pipeline's manifest steps on deterministic inputs.

    Returns the gamepad plugin's release.toml and the eov release.toml whose
    catalog pins the plugin's per-platform hashes. The committed copies in
    app/tests/fixtures are parsed by the Rust manifest tests, so both
    implementations are checked against the same bytes.
    """

    plugin_repository = "eosin-platform/eov-gamepad-plugin"
    plugin_specs = artifact_specs("plugin", "gamepad", FIXTURE_VERSION)
    write_artifacts(root / "plugin-artifacts", plugin_specs)
    plugin_staged = stage_artifacts(
        root / "plugin-artifacts", root / "plugin-staging", plugin_specs
    )
    plugin_text = render_manifest(
        FIXTURE_VERSION,
        plugin_repository,
        plugin_specs,
        plugin_staged,
        None,
        FIXTURE_ENVIRONMENT,
        ("gamepad", "Gamepad", PLUGIN_METADATA["gamepad"][2], None),
        signed=True,
    )
    validate_manifest_data(
        tomllib.loads(plugin_text),
        FIXTURE_VERSION,
        plugin_repository,
        plugin_specs,
        FIXTURE_ENVIRONMENT,
        "plugin",
    )
    plugin_path = root / "gamepad-release.toml"
    plugin_path.write_text(plugin_text, encoding="utf-8")
    release = read_plugin_release(
        plugin_path,
        "gamepad",
        PLUGIN_METADATA["gamepad"][0],
        FIXTURE_VERSION,
        FIXTURE_VERSION,
    )

    eov_specs = artifact_specs("eov", None, FIXTURE_VERSION)
    write_artifacts(root / "eov-artifacts", eov_specs)
    eov_staged = stage_artifacts(root / "eov-artifacts", root / "eov-staging", eov_specs)
    eov_text = render_manifest(
        FIXTURE_VERSION,
        EOV_REPOSITORY,
        eov_specs,
        eov_staged,
        {"annotations": FIXTURE_VERSION, "gamepad": FIXTURE_VERSION},
        plugin_releases={"gamepad": release},
        signed=True,
    )
    validate_manifest_data(
        tomllib.loads(eov_text), FIXTURE_VERSION, EOV_REPOSITORY, eov_specs
    )
    return plugin_text, eov_text


def write_plugin_release_toml(
    path: Path, environment: str, repository: str | None = None
) -> None:
    repository = repository or PLUGIN_METADATA["gamepad"][0]
    lines = [
        "[plugin]",
        'id = "gamepad"',
        'version = "1.0.1"',
        f"environment = {json.dumps(environment)}",
        "",
    ]
    for platform in ("linux.x86_64", "macos.arm64"):
        file_name = f"gamepad-v1.0.1-{platform.replace('.', '-')}.eop"
        lines.extend(
            (
                f"[platform.{platform}]",
                'version = "1.0.1"',
                f'sha256 = "{"a" * 64}"',
                f'url = "{repository}/releases/download/v1.0.1/{file_name}"',
                "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_app_archive(path: Path, minimum_system_version: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "eov.app/Contents/Info.plist",
            plistlib.dumps({"LSMinimumSystemVersion": minimum_system_version}),
        )


class ReleaseInfrastructureTests(unittest.TestCase):
    def make_eov_manifest(
        self, root: Path, version: str = "0.4.5"
    ) -> tuple[dict[str, object], Path]:
        import tomllib

        source_dir = root / "actions-artifacts"
        staging_dir = root / "release-files"
        specs = artifact_specs("eov", None, version)
        for index, spec in enumerate(specs):
            source = source_dir / f"job-{index}" / spec.filename
            source.parent.mkdir(parents=True)
            source.write_bytes(f"artifact:{spec.filename}".encode("ascii"))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (source.parent / f"{spec.filename}.sha256").write_text(
                f"{digest}  {spec.filename}\n",
                encoding="ascii",
            )
        staged = stage_artifacts(source_dir, staging_dir, specs)
        manifest_path = root / "release.toml"
        manifest_path.write_text(
            render_manifest(
                version,
                EOV_REPOSITORY,
                specs,
                staged,
                {"annotations": "0.2.1", "gamepad": "0.2.2"},
            ),
            encoding="utf-8",
        )
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_data(data, version, EOV_REPOSITORY, specs)
        return data, manifest_path

    def test_manifest_hashes_and_urls_come_from_staged_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, _ = self.make_eov_manifest(Path(temporary_directory))
            entry = data["platform"]["macos"]["arm64"]
            self.assertEqual(entry["version"], "0.4.5")
            self.assertIn("/releases/download/v0.4.5/", entry["url"])
            self.assertNotIn("/latest/", entry["url"])
            self.assertEqual(len(entry["sha256"]), 64)

    def test_duplicate_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = artifact_specs("eov", None, "0.4.5")[0]
            for directory in (root / "one", root / "two"):
                directory.mkdir()
                (directory / spec.filename).write_bytes(b"duplicate")
            with self.assertRaisesRegex(ReleaseError, "duplicate release artifact"):
                stage_artifacts(root, root / "staging", (spec,))

    def test_missing_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            specs = artifact_specs("eov", None, "0.4.5")
            (root / specs[0].filename).parent.mkdir(parents=True, exist_ok=True)
            (root / specs[0].filename).write_bytes(b"only one artifact")
            with self.assertRaisesRegex(ReleaseError, "missing release artifacts"):
                stage_artifacts(root, root / "staging", specs)

    def test_version_validation_rejects_non_semver(self) -> None:
        with self.assertRaises(ReleaseError):
            validate_version("v0.4.5")
        with self.assertRaises(ReleaseError):
            validate_version("0.4", stable_only=True)
        self.assertEqual(validate_version("0.4.5"), "0.4.5")

    def test_historical_cask_is_immutable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "eov@0.4.5.rb"
            content = cask_content(
                "eov@0.4.5",
                "0.4.5",
                "a" * 64,
                "https://github.com/eosin-platform/eov/releases/download/v0.4.5/eov-v0.4.5-macos-arm64.zip",
                "b" * 64,
                "https://github.com/eosin-platform/eov/releases/download/v0.4.5/eov-v0.4.5-macos-x86_64.zip",
                "big_sur",
            )
            self.assertTrue(write_historical(path, content))
            self.assertFalse(write_historical(path, content))
            with self.assertRaisesRegex(ReleaseError, "historical cask differs"):
                write_historical(
                    path, content.replace('version "0.4.5"', 'version "0.4.6"')
                )

    def test_readme_transform_is_bounded_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, _ = self.make_eov_manifest(Path(temporary_directory))
            original = f"""Unrelated example version 0.4.4.
{EOV_MANIFEST_START}
Release: https://github.com/eosin-platform/eov/releases/tag/v0.4.4
macOS: https://github.com/eosin-platform/eov/releases/download/v0.4.4/eov-v0.4.4-macos-arm64.zip
Install exact: brew install --cask eov@0.4.4
{EOV_MANIFEST_END}
"""
            updated = transform_readme(original, "0.4.5", data)
            self.assertIn("Unrelated example version 0.4.4", updated)
            self.assertIn(
                "/releases/download/v0.4.5/eov-v0.4.5-macos-arm64.zip", updated
            )
            self.assertIn("brew install --cask eov@0.4.5", updated)
            validate_readme_text(updated, "0.4.5", data)
            self.assertEqual(transform_readme(updated, "0.4.5", data), updated)

    def test_readme_without_markers_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, _ = self.make_eov_manifest(Path(temporary_directory))
            with self.assertRaisesRegex(ReleaseError, "marker block"):
                transform_readme("brew install --cask eov", "0.4.5", data)

    def test_stale_readme_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, _ = self.make_eov_manifest(Path(temporary_directory))
            readme = f"""{EOV_MANIFEST_START}
Download: https://github.com/eosin-platform/eov/releases/download/v0.4.4/eov-v0.4.4-linux-unknown.AppImage
Exact: brew install --cask eov@0.4.4
{EOV_MANIFEST_END}
"""
            with self.assertRaisesRegex(ReleaseError, "could not map README artifact"):
                transform_readme(readme, "0.4.5", data)

    def test_plugin_selection_uses_newest_compatible_release(self) -> None:
        self.assertTrue(semver_requirement_satisfied(">=0.4.0, <0.5.0", "0.4.6"))
        self.assertEqual(
            select_compatible_plugin_release(
                "0.4.6",
                [
                    ("0.3.0", ">=0.5.0"),
                    ("0.2.5", ">=0.4.0"),
                    ("0.2.4", ">=0.3.0"),
                ],
            ),
            "0.2.5",
        )

    def test_plugin_selection_fails_when_no_release_is_compatible(self) -> None:
        with self.assertRaisesRegex(ReleaseError, "no stable plugin release"):
            select_compatible_plugin_release("0.4.6", [("0.3.0", ">=0.5.0")])

    def test_legacy_plugin_releases_without_manifests_are_ignored(self) -> None:
        releases = [
            {"tag_name": "v0.2.1", "draft": False, "prerelease": False},
            {"tag_name": "v0.2.3", "draft": False, "prerelease": False},
        ]
        missing_manifest = urllib.error.HTTPError(
            "https://example.test/release.toml",
            404,
            "Not Found",
            {},
            io.BytesIO(),
        )
        manifest = io.BytesIO(
            b'[plugin]\nversion = "0.2.3"\nenvironment = ">=0.4.0, <0.5.0"\n'
        )
        with (
            patch("release_infrastructure._github_json", return_value=releases),
            patch(
                "release_infrastructure.urllib.request.urlopen",
                side_effect=[missing_manifest, manifest],
            ),
        ):
            candidates = published_plugin_candidates("example/plugin")

        self.assertEqual(candidates, [("0.2.3", ">=0.4.0, <0.5.0")])
        self.assertEqual(select_compatible_plugin_release("0.4.6", candidates), "0.2.3")

    def test_prerelease_requirements_follow_semver_rules(self) -> None:
        self.assertFalse(semver_requirement_satisfied(">=0.4.6", "0.4.6-alpha"))
        self.assertTrue(semver_requirement_satisfied(">=0.4.6-alpha", "0.4.6-alpha.1"))

    def test_open_ended_plugin_requirements_are_capped_like_the_app(self) -> None:
        self.assertFalse(requirement_is_bounded(">=0.4.1"))
        self.assertTrue(requirement_is_bounded(">=1.0.0, <2.0.0"))
        self.assertEqual(len(bounded_requirement(">=0.4.1")), 2)
        with self.assertRaisesRegex(ReleaseError, "no lower bound"):
            bounded_requirement("*")
        # 0.5.0 was a breaking release for plugins built against 0.4.
        self.assertFalse(host_compatible(">=0.4.0", "0.5.0"))
        with self.assertRaisesRegex(ReleaseError, "no stable plugin release"):
            select_compatible_plugin_release("0.5.0", [("0.2.5", ">=0.4.0")])

    def test_catalog_pins_plugin_hashes_from_their_release_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, eov_text = generate_release_fixtures(Path(temporary_directory))
        data = tomllib.loads(eov_text)
        self.assertEqual(data["manifest"]["signature"], "minisign")
        gamepad = data["plugins"]["gamepad"]
        self.assertEqual(gamepad["release_tag"], "v1.0.1")
        self.assertEqual(gamepad["environment"], FIXTURE_ENVIRONMENT)
        linux = gamepad["platform"]["linux"]["x86_64"]
        expected = hashlib.sha256(
            b"artifact:gamepad-v1.0.1-linux-x86_64.eop"
        ).hexdigest()
        self.assertEqual(linux["sha256"], expected)
        self.assertTrue(
            linux["url"].endswith(
                "/releases/download/v1.0.1/gamepad-v1.0.1-linux-x86_64.eop"
            )
        )
        self.assertEqual(len(gamepad["platform"]["windows"]), 2)
        # Without a release manifest, the catalog keeps the version-only entry.
        self.assertNotIn("platform", data["plugins"]["annotations"])

    def test_generated_fixtures_parsed_by_the_rust_tests_are_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plugin_text, eov_text = generate_release_fixtures(Path(temporary_directory))
        if os.environ.get("EOV_UPDATE_FIXTURES") == "1":
            GENERATED_PLUGIN_FIXTURE.write_text(plugin_text, encoding="utf-8")
            GENERATED_EOV_FIXTURE.write_text(eov_text, encoding="utf-8")
        hint = (
            "regenerate with: EOV_UPDATE_FIXTURES=1 "
            "python3 -m unittest test_release_infrastructure"
        )
        self.assertEqual(
            GENERATED_PLUGIN_FIXTURE.read_text(encoding="utf-8"), plugin_text, hint
        )
        self.assertEqual(
            GENERATED_EOV_FIXTURE.read_text(encoding="utf-8"), eov_text, hint
        )

    def test_plugin_release_manifests_must_be_bounded_and_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "release.toml"
            repository = PLUGIN_METADATA["gamepad"][0]
            write_plugin_release_toml(path, ">=1.0.0")
            with self.assertRaisesRegex(ReleaseError, "open-ended"):
                read_plugin_release(path, "gamepad", repository, "1.0.1", "1.0.1")
            write_plugin_release_toml(path, ">=2.0.0, <3.0.0")
            with self.assertRaisesRegex(ReleaseError, "excludes 1.0.1"):
                read_plugin_release(path, "gamepad", repository, "1.0.1", "1.0.1")
            write_plugin_release_toml(
                path, FIXTURE_ENVIRONMENT, "https://github.com/attacker/fork"
            )
            with self.assertRaisesRegex(ReleaseError, "does not belong"):
                read_plugin_release(path, "gamepad", repository, "1.0.1", "1.0.1")
            write_plugin_release_toml(path, FIXTURE_ENVIRONMENT)
            with self.assertRaisesRegex(ReleaseError, "not version 1.0.2"):
                read_plugin_release(path, "gamepad", repository, "1.0.2", "1.0.1")
            release = read_plugin_release(path, "gamepad", repository, "1.0.1", "1.0.1")
            self.assertEqual(release.release_tag, "v1.0.1")
            self.assertEqual(
                sorted(release.artifacts),
                ["platform.linux.x86_64", "platform.macos.arm64"],
            )

    def test_catalog_validation_rejects_tampered_plugin_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, eov_text = generate_release_fixtures(Path(temporary_directory))
        specs = artifact_specs("eov", None, FIXTURE_VERSION)
        for old, new, message in (
            (
                "/eov-gamepad-plugin/releases/download/v1.0.1/gamepad-v1.0.1-linux-x86_64.eop",
                "/eov-gamepad-plugin/releases/download/v0.9.0/gamepad-v1.0.1-linux-x86_64.eop",
                "not part of release",
            ),
            (
                'environment = ">=1.0.0, <2.0.0"',
                'environment = ">=1.0.0"',
                "bounded environment",
            ),
        ):
            with self.subTest(message=message):
                tampered = tomllib.loads(eov_text.replace(old, new))
                with self.assertRaisesRegex(ReleaseError, message):
                    validate_manifest_data(
                        tampered, FIXTURE_VERSION, EOV_REPOSITORY, specs
                    )

    def test_cask_removes_every_eov_state_directory(self) -> None:
        content = cask_content(
            "eov", "1.0.1", "a" * 64, "https://a", "b" * 64, "https://b", "big_sur"
        )
        for path in (
            "~/.eov",
            "~/Library/Application Support/eov",
            "~/Library/Caches/eov",
        ):
            self.assertIn(path, CASK_ZAP_PATHS)
            self.assertIn(f'"{path}",', content)

    def test_cask_macos_requirement_comes_from_the_shipped_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            info_plist = root / "Info.plist"
            info_plist.write_bytes(plistlib.dumps({"LSMinimumSystemVersion": "11.0"}))
            self.assertEqual(macos_cask_symbol(info_plist), "big_sur")
            write_app_archive(root / "arm64.zip", "11.0")
            write_app_archive(root / "x86_64.zip", "12.3")
            self.assertEqual(
                macos_cask_symbol(info_plist, (root / "arm64.zip", root / "x86_64.zip")),
                "monterey",
            )
            write_app_archive(root / "tahoe.zip", "26.0")
            self.assertEqual(macos_cask_symbol(info_plist, (root / "tahoe.zip",)), "tahoe")
            write_app_archive(root / "future.zip", "99.0")
            with self.assertRaisesRegex(ReleaseError, "unsupported LSMinimumSystemVersion"):
                macos_cask_symbol(info_plist, (root / "future.zip",))


class SharedSemverTableTests(unittest.TestCase):
    """The table app/tests/semver_cross.rs checks against the Rust crate.

    Passing here means the release tooling and eov agree on which plugin
    releases a given eov version accepts.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = json.loads(SEMVER_TABLE.read_text(encoding="utf-8"))

    def test_requirements_parse_and_match_like_the_rust_crate(self) -> None:
        versions = self.table["versions"]
        self.assertGreater(len(self.table["cases"]), 50)
        for case in self.table["cases"]:
            requirement = case["requirement"]
            with self.subTest(requirement=requirement):
                if not case["valid"]:
                    with self.assertRaises(ReleaseError):
                        parse_requirement(requirement)
                    continue
                parse_requirement(requirement)
                matches = [
                    version
                    for version in versions
                    if semver_requirement_satisfied(requirement, version)
                ]
                self.assertEqual(matches, case["matches"])

    def test_bounded_plugin_compatibility_matches_the_app(self) -> None:
        for case in self.table["bounded_cases"]:
            with self.subTest(requirement=case["requirement"], version=case["version"]):
                self.assertEqual(
                    host_compatible(case["requirement"], case["version"]),
                    case["compatible"],
                )


if __name__ == "__main__":
    unittest.main()
