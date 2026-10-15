#!/usr/bin/env python3
"""Minisign-compatible signing of eov release metadata.

eov verifies ``release.toml.minisig`` against public keys compiled into the
binary (``app/src/update/keys.rs``). This tool produces and checks those
signatures without needing the ``minisign`` binary; its keys and signatures
are byte-for-byte compatible with minisign 0.11, so either tool can be used.

Commands
--------

``keygen --secret-key PATH --public-key PATH [--unencrypted]``
    Create a new key pair. The secret key is encrypted with a password read
    from ``EOV_SIGNING_KEY_PASSWORD`` (or prompted for), unless
    ``--unencrypted`` is given.

``sign --secret-key PATH --kind eov --version X --repository URL FILE``
    Write ``FILE.minisig``. The signed trusted comment binds the signature to
    the release identity (``kind``, ``version``, ``repository``), so a
    signature cannot be replayed for another release or another manifest
    kind. ``--secret-key -`` reads the key from ``EOV_RELEASE_SIGNING_KEY``.

``verify --public-key KEY FILE``
    Verify ``FILE.minisig`` (``KEY`` is the base64 public key or a ``.pub``
    file) and print the trusted comment.

Setting up release signing (repository owner)
---------------------------------------------

1. ``python3 scripts/release_signing.py keygen --secret-key eov-release.key
   --public-key eov-release.pub`` on an offline machine, with a strong
   password.
2. Store the *contents* of ``eov-release.key`` as the GitHub Actions secret
   ``EOV_RELEASE_SIGNING_KEY`` and the password as
   ``EOV_RELEASE_SIGNING_KEY_PASSWORD`` (use a protected "release"
   environment). Keep an offline backup of the secret key.
3. Add the second line of ``eov-release.pub`` to ``RELEASE_KEYS`` in
   ``app/src/update/keys.rs`` and set ``SIGNED_SINCE`` to the first version
   that will be signed. From that release on, eov refuses unsigned metadata.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature
except ImportError as error:  # pragma: no cover - exercised only without the package
    raise SystemExit(
        "release_signing.py needs the 'cryptography' package: pip install cryptography"
    ) from error


SIG_ALG = b"Ed"
SIG_ALG_PREHASHED = b"ED"
KDF_SCRYPT = b"Sc"
KDF_NONE = b"\x00\x00"
CHK_ALG = b"B2"
# libsodium crypto_pwhash_scryptsalsa208sha256 *_SENSITIVE limits, which is
# what `minisign -G` uses.
DEFAULT_OPSLIMIT = 33554432
DEFAULT_MEMLIMIT = 1073741824


class SigningError(Exception):
    pass


@dataclass(frozen=True)
class SecretKey:
    key_id: bytes
    private_key: Ed25519PrivateKey

    @property
    def public_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )


@dataclass(frozen=True)
class PublicKey:
    key_id: bytes
    public_key: Ed25519PublicKey


def key_id_hex(key_id: bytes) -> str:
    return int.from_bytes(key_id, "little").to_bytes(8, "big").hex().upper()


def scrypt_params(opslimit: int, memlimit: int) -> tuple[int, int, int]:
    """Port of libsodium's pickparams() for scryptsalsa208sha256."""

    opslimit = max(opslimit, 32768)
    r = 8
    if opslimit < memlimit // 32:
        p = 1
        max_n = opslimit // (r * 4)
        n_log2 = 1
        while n_log2 < 63:
            if (1 << n_log2) > max_n // 2:
                break
            n_log2 += 1
    else:
        max_n = memlimit // (r * 128)
        n_log2 = 1
        while n_log2 < 63:
            if (1 << n_log2) > max_n // 2:
                break
            n_log2 += 1
        max_rp = (opslimit // 4) // (1 << n_log2)
        max_rp = min(max_rp, 0x3FFFFFFF)
        p = max_rp // r
    return 1 << n_log2, r, p


def _kdf_stream(password: bytes, salt: bytes, opslimit: int, memlimit: int, length: int) -> bytes:
    n, r, p = scrypt_params(opslimit, memlimit)
    return hashlib.scrypt(
        password, salt=salt, n=n, r=r, p=p, maxmem=128 * r * n * p + 64 * 1024 * 1024, dklen=length
    )


def _checksum(key_id: bytes, secret: bytes) -> bytes:
    return hashlib.blake2b(SIG_ALG + key_id + secret, digest_size=32).digest()


def generate_key() -> SecretKey:
    return SecretKey(secrets.token_bytes(8), Ed25519PrivateKey.generate())


def encode_secret_key(
    key: SecretKey,
    password: bytes | None,
    opslimit: int = DEFAULT_OPSLIMIT,
    memlimit: int = DEFAULT_MEMLIMIT,
) -> str:
    seed = key.private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    secret = seed + key.public_bytes
    keynum_sk = key.key_id + secret + _checksum(key.key_id, secret)
    salt = secrets.token_bytes(32)
    if password is None:
        kdf = KDF_NONE
        opslimit = memlimit = 0
        salt = bytes(32)
        payload = keynum_sk
    else:
        kdf = KDF_SCRYPT
        stream = _kdf_stream(password, salt, opslimit, memlimit, len(keynum_sk))
        payload = bytes(a ^ b for a, b in zip(keynum_sk, stream))
    blob = (
        SIG_ALG
        + kdf
        + CHK_ALG
        + salt
        + opslimit.to_bytes(8, "little")
        + memlimit.to_bytes(8, "little")
        + payload
    )
    comment = (
        "minisign encrypted secret key" if password is not None else "minisign secret key"
    )
    return f"untrusted comment: {comment}\n{base64.b64encode(blob).decode()}\n"


def decode_secret_key(text: str, password: bytes | None) -> SecretKey:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("untrusted comment:"):
        raise SigningError("not a minisign secret key")
    try:
        blob = base64.b64decode(lines[1], validate=True)
    except ValueError as error:
        raise SigningError(f"secret key is not valid base64: {error}") from error
    if len(blob) != 158 or blob[:2] != SIG_ALG or blob[4:6] != CHK_ALG:
        raise SigningError("unsupported minisign secret key format")
    kdf = blob[2:4]
    salt = blob[6:38]
    opslimit = int.from_bytes(blob[38:46], "little")
    memlimit = int.from_bytes(blob[46:54], "little")
    payload = blob[54:]
    if kdf == KDF_SCRYPT:
        if password is None:
            raise SigningError("the secret key is encrypted; a password is required")
        stream = _kdf_stream(password, salt, opslimit, memlimit, len(payload))
        payload = bytes(a ^ b for a, b in zip(payload, stream))
    elif kdf != KDF_NONE:
        raise SigningError("unsupported secret key KDF")
    key_id, secret, checksum = payload[:8], payload[8:72], payload[72:104]
    if not secrets.compare_digest(checksum, _checksum(key_id, secret)):
        raise SigningError("wrong password or corrupted secret key")
    private_key = Ed25519PrivateKey.from_private_bytes(secret[:32])
    key = SecretKey(key_id, private_key)
    if key.public_bytes != secret[32:]:
        raise SigningError("corrupted secret key (public half does not match)")
    return key


def encode_public_key(key: SecretKey) -> str:
    blob = SIG_ALG + key.key_id + key.public_bytes
    return (
        f"untrusted comment: minisign public key {key_id_hex(key.key_id)}\n"
        f"{base64.b64encode(blob).decode()}\n"
    )


def decode_public_key(value: str) -> PublicKey:
    text = value.strip()
    path = Path(text)
    if not text.startswith("RW") and path.is_file():
        text = path.read_text(encoding="ascii")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = lines[-1] if lines else ""
    try:
        blob = base64.b64decode(line, validate=True)
    except ValueError as error:
        raise SigningError(f"public key is not valid base64: {error}") from error
    if len(blob) != 42 or blob[:2] != SIG_ALG:
        raise SigningError("unsupported minisign public key")
    return PublicKey(blob[2:10], Ed25519PublicKey.from_public_bytes(blob[10:]))


def trusted_comment(kind: str, version: str, repository: str, file_name: str, timestamp: int) -> str:
    for name, value in (("kind", kind), ("version", version), ("repository", repository), ("file", file_name)):
        if not value or any(ch in value for ch in "\t\r\n"):
            raise SigningError(f"invalid {name} for the trusted comment: {value!r}")
    return (
        f"timestamp:{timestamp}\tfile:{file_name}\tkind:{kind}"
        f"\tversion:{version}\trepository:{repository}"
    )


def sign_bytes(key: SecretKey, data: bytes, comment: str) -> str:
    digest = hashlib.blake2b(data, digest_size=64).digest()
    signature = key.private_key.sign(digest)
    global_signature = key.private_key.sign(signature + comment.encode("utf-8"))
    return (
        "untrusted comment: signature from eov release key "
        f"{key_id_hex(key.key_id)}\n"
        f"{base64.b64encode(SIG_ALG_PREHASHED + key.key_id + signature).decode()}\n"
        f"trusted comment: {comment}\n"
        f"{base64.b64encode(global_signature).decode()}\n"
    )


def verify_bytes(public_key: PublicKey, data: bytes, signature_text: str) -> str:
    lines = signature_text.splitlines()
    if len(lines) < 4 or not lines[2].startswith("trusted comment: "):
        raise SigningError("malformed minisign signature")
    try:
        blob = base64.b64decode(lines[1], validate=True)
        global_signature = base64.b64decode(lines[3], validate=True)
    except ValueError as error:
        raise SigningError(f"signature is not valid base64: {error}") from error
    if len(blob) != 74 or len(global_signature) != 64:
        raise SigningError("malformed minisign signature")
    algorithm, key_id, signature = blob[:2], blob[2:10], blob[10:]
    if key_id != public_key.key_id:
        raise SigningError(
            f"signed by key {key_id_hex(key_id)}, expected {key_id_hex(public_key.key_id)}"
        )
    if algorithm == SIG_ALG_PREHASHED:
        message = hashlib.blake2b(data, digest_size=64).digest()
    elif algorithm == SIG_ALG:
        message = data
    else:
        raise SigningError("unsupported signature algorithm")
    comment = lines[2][len("trusted comment: ") :]
    try:
        public_key.public_key.verify(signature, message)
        public_key.public_key.verify(global_signature, signature + comment.encode("utf-8"))
    except InvalidSignature as error:
        raise SigningError("signature verification failed") from error
    return comment


def _password(env: str, confirm: bool) -> bytes:
    value = os.environ.get(env)
    if value is None:
        if not sys.stdin.isatty():
            raise SigningError(f"set {env} or run interactively to enter the password")
        value = getpass.getpass("Secret key password: ")
        if confirm and getpass.getpass("Repeat password: ") != value:
            raise SigningError("passwords do not match")
    if not value:
        raise SigningError("the password must not be empty (use --unencrypted instead)")
    return value.encode("utf-8")


def secret_key_is_encrypted(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise SigningError("not a minisign secret key")
    try:
        blob = base64.b64decode(lines[1], validate=True)
    except ValueError as error:
        raise SigningError(f"secret key is not valid base64: {error}") from error
    return blob[2:4] == KDF_SCRYPT


def _read_secret_key(path: str) -> SecretKey:
    if path == "-":
        text = os.environ.get("EOV_RELEASE_SIGNING_KEY")
        if not text:
            raise SigningError("EOV_RELEASE_SIGNING_KEY is not set")
    else:
        text = Path(path).read_text(encoding="ascii")
    password = (
        _password("EOV_RELEASE_SIGNING_KEY_PASSWORD", False)
        if secret_key_is_encrypted(text)
        else None
    )
    return decode_secret_key(text, password)


def _write_new(path: Path, content: str, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write(content)


def command_keygen(args: argparse.Namespace) -> None:
    key = generate_key()
    password = None if args.unencrypted else _password("EOV_SIGNING_KEY_PASSWORD", True)
    _write_new(args.secret_key, encode_secret_key(key, password), 0o600)
    _write_new(args.public_key, encode_public_key(key), 0o644)
    print(f"key id {key_id_hex(key.key_id)}")
    print(encode_public_key(key).splitlines()[1])


def command_sign(args: argparse.Namespace) -> None:
    key = _read_secret_key(args.secret_key)
    data = args.file.read_bytes()
    comment = trusted_comment(
        args.kind, args.version, args.repository, args.file.name, args.timestamp or int(time.time())
    )
    output = args.output or args.file.with_name(args.file.name + ".minisig")
    output.write_text(sign_bytes(key, data, comment), encoding="ascii", newline="\n")
    print(f"signed {args.file} -> {output} with key {key_id_hex(key.key_id)}")


def command_verify(args: argparse.Namespace) -> None:
    public_key = decode_public_key(args.public_key)
    signature = (args.signature or args.file.with_name(args.file.name + ".minisig")).read_text(
        encoding="utf-8"
    )
    print(verify_bytes(public_key, args.file.read_bytes(), signature))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    keygen = subparsers.add_parser("keygen", help="create a signing key pair")
    keygen.add_argument("--secret-key", type=Path, required=True)
    keygen.add_argument("--public-key", type=Path, required=True)
    keygen.add_argument("--unencrypted", action="store_true")
    keygen.set_defaults(function=command_keygen)

    sign = subparsers.add_parser("sign", help="sign a release manifest")
    sign.add_argument("--secret-key", required=True, help="key file, or - for EOV_RELEASE_SIGNING_KEY")
    sign.add_argument("--kind", choices=("eov", "plugin"), required=True)
    sign.add_argument("--version", required=True)
    sign.add_argument("--repository", required=True)
    sign.add_argument("--timestamp", type=int)
    sign.add_argument("--output", type=Path)
    sign.add_argument("file", type=Path)
    sign.set_defaults(function=command_sign)

    verify = subparsers.add_parser("verify", help="verify a signature")
    verify.add_argument("--public-key", required=True)
    verify.add_argument("--signature", type=Path)
    verify.add_argument("file", type=Path)
    verify.set_defaults(function=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.function(args)
    except (SigningError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
