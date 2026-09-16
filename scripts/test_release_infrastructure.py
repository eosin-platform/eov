#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from release_infrastructure import (
    EOV_MANIFEST_END,
    EOV_MANIFEST_START,
    EOV_REPOSITORY,
    ReleaseError,
    artifact_specs,
    cask_content,
    render_manifest,
    stage_artifacts,
    transform_readme,
    validate_manifest_data,
    validate_readme_text,
    validate_version,
    write_historical,
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


if __name__ == "__main__":
    unittest.main()
