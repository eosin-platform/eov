#!/usr/bin/env python3

from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap_homebrew


def release(version: str) -> dict[str, object]:
    base = f"https://github.com/eosin-platform/eov/releases/download/v{version}"
    return {
        "tag_name": f"v{version}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": f"eov-v{version}-macos-{arch}.zip",
                "browser_download_url": f"{base}/eov-v{version}-macos-{arch}.zip",
                "digest": "sha256:" + digit * 64,
            }
            for arch, digit in (("arm64", "a"), ("x86_64", "b"))
        ],
    }


class BootstrapHomebrewTests(unittest.TestCase):
    def test_existing_historical_casks_are_kept_and_new_ones_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            info_plist = root / "Info.plist"
            info_plist.write_bytes(plistlib.dumps({"LSMinimumSystemVersion": "11.0"}))
            casks = root / "tap" / "Casks"
            casks.mkdir(parents=True)
            # Written by an older template: immutable, so it must survive.
            old = casks / "eov@0.4.5.rb"
            old.write_text('cask "eov@0.4.5" do\n    # old template\nend\n', encoding="utf-8")
            with patch.object(
                bootstrap_homebrew,
                "fetch_releases",
                return_value=[release("0.4.5"), release("1.0.1")],
            ):
                status = bootstrap_homebrew.main(
                    ["--tap-root", str(root / "tap"), "--info-plist", str(info_plist)]
                )
            self.assertEqual(status, 0)
            self.assertIn("old template", old.read_text(encoding="utf-8"))
            new = (casks / "eov@1.0.1.rb").read_text(encoding="utf-8")
            self.assertIn('"~/.eov",', new)
            latest = (casks / "eov.rb").read_text(encoding="utf-8")
            self.assertIn('version "1.0.1"', latest)


if __name__ == "__main__":
    unittest.main()
