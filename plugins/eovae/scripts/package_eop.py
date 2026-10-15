#!/usr/bin/env python3
"""Package eovae as an eov plugin package (.eop).

An .eop is a plain tar archive (eov does not read compressed packages yet):

    plugin.toml
    libeovae.so | libeovae.dylib | eovae.dll
    ui/...                          sidebar and icons
    onnxruntime/...                 optional ONNX Runtime (see below)

ONNX Runtime policy
-------------------
eovae opens ONNX Runtime at run time, so one plugin build works with any
runtime >= 1.22. A package can carry one:

* ``--fetch-ort`` downloads the pinned official release for ``--platform``
  and ``--ort-flavor`` and verifies its SHA-256 before using it
  (``--ort-archive`` uses an already downloaded archive, verified the same
  way, for offline builds).
* ``--ort-dir DIR`` (or ``EOVAE_ORT_DIR``) takes the libraries from a
  folder, e.g. a custom build.
* ``--no-ort`` packages none: the runtime comes from ``EOVAE_ORT_LIB`` or
  the system.

Only the files ONNX Runtime needs are staged, under their SONAME (no
symlinks, no duplicates): the runtime, the shared provider bridge and, for
GPU flavours, the CUDA provider. CUDA and cuDNN themselves are never
packaged (gigabytes, and licensed separately); eovae finds them on the
system when CUDA is selected. The runtime's LICENSE and ThirdPartyNotices
are included.

The archive is reproducible: sorted entries, fixed timestamps
(``SOURCE_DATE_EPOCH``, default 0) and no owner names.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import pathlib
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ORT_VERSION = "1.28.3"
ORT_URL = "https://github.com/microsoft/onnxruntime/releases/download/v{version}/{name}"

# (platform, flavour) -> (archive, sha256), from the release's published
# digests (github.com/microsoft/onnxruntime/releases/tag/v1.28.3).
ORT_ARCHIVES = {
    ("linux-x64", "cpu"): (
        "onnxruntime-linux-x64-1.28.3.tgz",
        "db14e4863bd37893fc59729d986ab2a0d043d10b7d44da1913c4982b7e3d009c",
    ),
    ("linux-x64", "cuda12"): (
        "onnxruntime-linux-x64-gpu_cuda12-1.28.3.tgz",
        "4e16d2ec66521a24fe917bd848eb8ae3a9107cb8b4aec978fb2b3ab1a640fcb7",
    ),
    ("linux-x64", "cuda13"): (
        "onnxruntime-linux-x64-gpu_cuda13-1.28.3.tgz",
        "33e91f819324449bede2d1e853c8d46bb6253fa8890bac98ae676f041c8f73a6",
    ),
    ("linux-aarch64", "cpu"): (
        "onnxruntime-linux-aarch64-1.28.3.tgz",
        "6c6b1ae96d7b0be9f555092857c6a4f2b0b5587a5298d732d9e52e8253038ce7",
    ),
    ("osx-arm64", "cpu"): (
        "onnxruntime-osx-arm64-1.28.3.tgz",
        "c436bd9f47dbce6f6311ccc21de97f829b3c862c09d24aa321958cb72f0a1d32",
    ),
    ("win-x64", "cpu"): (
        "onnxruntime-win-x64-1.28.3.zip",
        "1d6fab48e85f948436af7c8c971d2c145cf224e2c444755dec894f8b0de11a83",
    ),
    ("win-x64", "cuda12"): (
        "onnxruntime-win-x64-gpu_cuda12-1.28.3.zip",
        "65d7f956649df80a8ccdcafa7843e4b02f38bfc9e6de28711ed8384508040330",
    ),
    ("win-x64", "cuda13"): (
        "onnxruntime-win-x64-gpu_cuda13-1.28.3.zip",
        "a7216e498c14fb4783a42667ffddcf1590ae262429307ed0dc5889d9c365584e",
    ),
    ("win-arm64", "cpu"): (
        "onnxruntime-win-arm64-1.28.3.zip",
        "76d5e5a23fb7fdbc2ca0c3214bad1b58b2a41016c7052e74e60ed8004bea71b1",
    ),
}

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent


def host_platform() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if sys.platform.startswith("linux"):
        return "linux-aarch64" if arch == "arm64" else "linux-x64"
    if sys.platform == "darwin":
        return "osx-arm64"
    if sys.platform.startswith("win"):
        return f"win-{arch}"
    raise SystemExit(f"unsupported platform {sys.platform}/{machine}")


def plugin_id() -> str:
    text = (PLUGIN_ROOT / "plugin.toml").read_text(encoding="utf-8")
    try:
        import tomllib

        return tomllib.loads(text)["id"]
    except ModuleNotFoundError:  # Python < 3.11
        match = re.search(r'^id\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not match:
            raise SystemExit("plugin.toml has no id")
        return match.group(1)


def library_name(plugin: str, target: str) -> str:
    if target.startswith("win"):
        return f"{plugin}.dll"
    if target.startswith("osx"):
        return f"lib{plugin}.dylib"
    return f"lib{plugin}.so"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_files(lib_dir: pathlib.Path, target: str, flavour: str) -> dict[str, pathlib.Path]:
    """Maps staged names to source files (symlinks resolved)."""
    files: dict[str, pathlib.Path] = {}
    names = sorted(p.name for p in lib_dir.iterdir())
    gpu = flavour != "cpu"

    def add(staged: str, source: pathlib.Path) -> None:
        files[staged] = source.resolve(strict=True)

    if target.startswith("win"):
        wanted = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]
        if gpu:
            wanted.append("onnxruntime_providers_cuda.dll")
        for name in wanted:
            if name in names:
                add(name, lib_dir / name)
    elif target.startswith("osx"):
        versioned = [n for n in names if re.fullmatch(r"libonnxruntime\.\d+\.\d+\.\d+\.dylib", n)]
        main = versioned[-1] if versioned else ("libonnxruntime.dylib" if "libonnxruntime.dylib" in names else None)
        if main:
            add(main, lib_dir / main)
    else:
        candidates = [n for n in names if re.fullmatch(r"libonnxruntime\.so(\.\d+)*", n)]
        if candidates:
            # Stage the real library once, under its SONAME.
            add("libonnxruntime.so.1", lib_dir / max(candidates, key=len))
        wanted = ["libonnxruntime_providers_shared.so"]
        if gpu:
            wanted.append("libonnxruntime_providers_cuda.so")
        for name in wanted:
            if name in names:
                add(name, lib_dir / name)
    if not any(name.startswith(("libonnxruntime", "onnxruntime.dll")) for name in files):
        raise SystemExit(f"no ONNX Runtime library found in {lib_dir}")
    if gpu and not any("providers_cuda" in name for name in files):
        raise SystemExit(f"the GPU flavour needs the CUDA provider in {lib_dir}")
    return files


def notices(root: pathlib.Path) -> dict[str, pathlib.Path]:
    found = {}
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        for candidate in (root / name, root.parent / name):
            if candidate.is_file():
                found[name] = candidate
                break
    return found


def fetch_archive(target: str, flavour: str, archive: pathlib.Path | None, cache: pathlib.Path) -> pathlib.Path:
    key = (target, flavour)
    if key not in ORT_ARCHIVES:
        choices = ", ".join(f"{p}/{f}" for p, f in sorted(ORT_ARCHIVES))
        raise SystemExit(f"no pinned ONNX Runtime for {target}/{flavour}; available: {choices}")
    name, expected = ORT_ARCHIVES[key]
    if archive is None:
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / name
        if not archive.is_file() or sha256(archive) != expected:
            url = ORT_URL.format(version=ORT_VERSION, name=name)
            print(f"downloading {url}", file=sys.stderr)
            partial = archive.with_suffix(archive.suffix + ".part")
            with urllib.request.urlopen(url) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(archive)
    actual = sha256(archive)
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {archive}: expected {expected}, got {actual}")
    return archive


def extract(archive: pathlib.Path, into: pathlib.Path) -> pathlib.Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(into, filter="data")
            else:
                t.extractall(into)
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1 or not (roots[0] / "lib").is_dir():
        raise SystemExit(f"unexpected layout in {archive}")
    return roots[0]


def add_file(archive: tarfile.TarFile, name: str, source: pathlib.Path, mtime: int) -> None:
    data = source.read_bytes()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = mtime
    info.mode = 0o755 if source.suffix in (".so", ".dylib", ".dll") or ".so." in source.name else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", required=True, type=pathlib.Path, help="the built plugin library")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="the .eop to write")
    parser.add_argument("--platform", default=None, help="linux-x64, linux-aarch64, osx-arm64, win-x64 or win-arm64 (default: this machine)")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--fetch-ort", action="store_true", help=f"bundle the pinned ONNX Runtime {ORT_VERSION}")
    runtime.add_argument("--ort-dir", type=pathlib.Path, default=os.environ.get("EOVAE_ORT_DIR"), help="bundle ONNX Runtime from this folder (default: $EOVAE_ORT_DIR)")
    runtime.add_argument("--no-ort", action="store_true", help="bundle no ONNX Runtime")
    parser.add_argument("--ort-flavor", default="cpu", choices=["cpu", "cuda12", "cuda13"], help="runtime build (GPU flavours add the CUDA provider)")
    parser.add_argument("--ort-archive", type=pathlib.Path, default=os.environ.get("EOVAE_ORT_ARCHIVE"), help="use this downloaded release archive (still verified)")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()) / "eovae-packaging", help="where downloads are kept")
    # Accepted for older callers (scripts/run-with-plugins.sh).
    parser.add_argument("--plugin-root", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--native-lib-dir", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plugin_root and args.plugin_root.resolve() != PLUGIN_ROOT:
        raise SystemExit(f"--plugin-root must be {PLUGIN_ROOT}")
    if args.native_lib_dir and not (args.fetch_ort or args.no_ort or args.ort_archive):
        args.ort_dir = args.native_lib_dir

    target = args.platform or host_platform()
    plugin = plugin_id()
    expected_library = library_name(plugin, target)
    if args.library.name != expected_library:
        raise SystemExit(f"the library must be named {expected_library} for {target}, got {args.library.name}")
    if not args.library.is_file():
        raise SystemExit(f"missing plugin library: {args.library}")

    entries: dict[str, pathlib.Path] = {
        "plugin.toml": PLUGIN_ROOT / "plugin.toml",
        expected_library: args.library,
    }
    for path in sorted((PLUGIN_ROOT / "ui").rglob("*")):
        if path.is_file() and path.suffix in (".slint", ".svg", ".png"):
            entries[path.relative_to(PLUGIN_ROOT).as_posix()] = path

    with tempfile.TemporaryDirectory() as scratch:
        runtime_root = None
        if args.fetch_ort or args.ort_archive:
            archive = fetch_archive(target, args.ort_flavor, args.ort_archive, args.cache_dir)
            runtime_root = extract(archive, pathlib.Path(scratch))
            lib_dir = runtime_root / "lib"
        elif args.ort_dir and not args.no_ort:
            lib_dir = args.ort_dir if not (args.ort_dir / "lib").is_dir() else args.ort_dir / "lib"
            runtime_root = args.ort_dir
        else:
            lib_dir = None
        if lib_dir is not None:
            for name, source in runtime_files(lib_dir, target, args.ort_flavor).items():
                entries[f"onnxruntime/{name}"] = source
            for name, source in notices(runtime_root).items():
                entries[f"onnxruntime/{name}"] = source

        mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".part")
        sizes = []
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as out:
            for name in sorted(entries):
                add_file(out, name, entries[name], mtime)
                sizes.append((name, entries[name].stat().st_size))
        partial.replace(args.output)

    for name, size in sizes:
        print(f"{size:>12,}  {name}")
    total = sum(size for _, size in sizes)
    print(f"{total:>12,}  total -> {args.output} (sha256 {sha256(args.output)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
