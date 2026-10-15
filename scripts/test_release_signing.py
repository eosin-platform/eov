#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from release_signing import (
    SigningError,
    decode_public_key,
    decode_secret_key,
    encode_public_key,
    encode_secret_key,
    generate_key,
    main,
    scrypt_params,
    secret_key_is_encrypted,
    sign_bytes,
    trusted_comment,
    verify_bytes,
)

# A signature produced by the reference minisign implementation (from the
# minisign-verify crate documentation), used to check format compatibility.
MINISIGN_PUBLIC_KEY = "RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3"
MINISIGN_SIGNATURE = """untrusted comment: signature from minisign secret key
RUQf6LRCGA9i559r3g7V1qNyJDApGip8MfqcadIgT9CuhV3EMhHoN1mGTkUidF/z7SrlQgXdy8ofjb7bNJJylDOocrCo8KLzZwo=
trusted comment: timestamp:1633700835\tfile:test\tprehashed
wLMDjy9FLAuxZ3q4NlEvkgtyhrr0gtTu6KC4KBJdITbbOeAi1zBIYo0v4iTgt8jJpIidRJnp94ABQkJAgAooBQ==
"""

# Small KDF limits keep the encrypted-key tests fast.
FAST_OPSLIMIT = 524288
FAST_MEMLIMIT = 16777216


class ReleaseSigningTests(unittest.TestCase):
    def test_verifies_a_reference_minisign_signature(self) -> None:
        comment = verify_bytes(decode_public_key(MINISIGN_PUBLIC_KEY), b"test", MINISIGN_SIGNATURE)
        self.assertEqual(comment, "timestamp:1633700835\tfile:test\tprehashed")
        with self.assertRaises(SigningError):
            verify_bytes(decode_public_key(MINISIGN_PUBLIC_KEY), b"tampered", MINISIGN_SIGNATURE)

    def test_scrypt_parameters_match_libsodium(self) -> None:
        # minisign's defaults (libsodium *_SENSITIVE limits).
        self.assertEqual(scrypt_params(33554432, 1073741824), (1 << 20, 8, 1))
        # Interactive limits.
        self.assertEqual(scrypt_params(524288, 16777216), (1 << 14, 8, 1))
        # opslimit is clamped to at least 32768.
        self.assertEqual(scrypt_params(1, 16777216), (1 << 10, 8, 1))

    def test_unencrypted_and_encrypted_keys_round_trip(self) -> None:
        key = generate_key()
        plain = encode_secret_key(key, None)
        self.assertFalse(secret_key_is_encrypted(plain))
        self.assertEqual(decode_secret_key(plain, None).public_bytes, key.public_bytes)

        encrypted = encode_secret_key(key, b"hunter2", FAST_OPSLIMIT, FAST_MEMLIMIT)
        self.assertTrue(secret_key_is_encrypted(encrypted))
        self.assertEqual(decode_secret_key(encrypted, b"hunter2").key_id, key.key_id)
        with self.assertRaisesRegex(SigningError, "wrong password"):
            decode_secret_key(encrypted, b"wrong")
        with self.assertRaisesRegex(SigningError, "password is required"):
            decode_secret_key(encrypted, None)

    def test_signatures_bind_data_key_and_trusted_comment(self) -> None:
        key = generate_key()
        public = decode_public_key(encode_public_key(key))
        comment = trusted_comment(
            "eov", "1.0.1", "https://github.com/eosin-platform/eov", "release.toml", 1760000000
        )
        signature = sign_bytes(key, b"manifest", comment)
        self.assertEqual(verify_bytes(public, b"manifest", signature), comment)
        with self.assertRaisesRegex(SigningError, "verification failed"):
            verify_bytes(public, b"manifest!", signature)
        forged = signature.replace("version:1.0.1", "version:1.0.2")
        with self.assertRaisesRegex(SigningError, "verification failed"):
            verify_bytes(public, b"manifest", forged)
        other = decode_public_key(encode_public_key(generate_key()))
        with self.assertRaisesRegex(SigningError, "expected"):
            verify_bytes(other, b"manifest", signature)

    def test_trusted_comment_rejects_separators(self) -> None:
        with self.assertRaises(SigningError):
            trusted_comment("eov", "1.0\t", "https://x", "release.toml", 0)

    def test_command_line_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = root / "release.key"
            public = root / "release.pub"
            manifest = root / "release.toml"
            manifest.write_text("[manifest]\nschema = 1\n", encoding="utf-8")
            self.assertEqual(
                main(["keygen", "--secret-key", str(secret), "--public-key", str(public), "--unencrypted"]),
                0,
            )
            if os.name == "posix":
                self.assertEqual(secret.stat().st_mode & 0o077, 0)
            with patch.dict(os.environ, {"EOV_RELEASE_SIGNING_KEY": secret.read_text()}):
                self.assertEqual(
                    main(
                        [
                            "sign",
                            "--secret-key",
                            "-",
                            "--kind",
                            "eov",
                            "--version",
                            "1.0.1",
                            "--repository",
                            "https://github.com/eosin-platform/eov",
                            str(manifest),
                        ]
                    ),
                    0,
                )
            self.assertTrue((root / "release.toml.minisig").is_file())
            self.assertEqual(main(["verify", "--public-key", str(public), str(manifest)]), 0)
            manifest.write_text("tampered", encoding="utf-8")
            self.assertEqual(main(["verify", "--public-key", str(public), str(manifest)]), 1)


if __name__ == "__main__":
    unittest.main()
