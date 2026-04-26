#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import tarfile


RUNTIME_PATTERNS = (
    "libonnxruntime*.so*",
    "libonnxruntime*.dylib",
    "onnxruntime*.dll",
    "libcublas*.so*",
    "libcudart*.so*",
    "libcudnn*.so*",
    "libcufft*.so*",
    "libcurand*.so*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the eovae plugin as a .eop tarball")
    parser.add_argument("--plugin-root", required=True, help="Path to the plugin repository root")
    parser.add_argument("--library", required=True, help="Compiled plugin shared library to include")
    parser.add_argument("--output", required=True, help="Output .eop file path")
    parser.add_argument(
        "--native-lib-dir",
        help="Directory containing vendored ONNX Runtime shared libraries copied by cargo",
    )
    return parser.parse_args()


def add_directory(archive: tarfile.TarFile, root: pathlib.Path, relative_dir: pathlib.Path) -> None:
    directory = root / relative_dir
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            archive.add(path, arcname=path.relative_to(root).as_posix())


def iter_runtime_libraries(native_lib_dir: pathlib.Path) -> list[pathlib.Path]:
    matches: dict[str, pathlib.Path] = {}
    for pattern in RUNTIME_PATTERNS:
        for path in sorted(native_lib_dir.glob(pattern)):
            if path.is_file() or path.is_symlink():
                matches[path.name] = path
    return list(matches.values())


def add_runtime_libraries(archive: tarfile.TarFile, native_lib_dir: pathlib.Path) -> None:
    runtime_libraries = iter_runtime_libraries(native_lib_dir)
    if not runtime_libraries:
        raise SystemExit(f"missing vendored ONNX Runtime libraries in {native_lib_dir}")

    for library in runtime_libraries:
        source = library.resolve(strict=True) if library.is_symlink() else library
        archive.add(source, arcname=library.name, recursive=False)


def main() -> int:
    args = parse_args()
    plugin_root = pathlib.Path(args.plugin_root).resolve()
    library_path = pathlib.Path(args.library).resolve()
    output_path = pathlib.Path(args.output).resolve()
    native_lib_dir = pathlib.Path(args.native_lib_dir).resolve() if args.native_lib_dir else None

    manifest_path = plugin_root / "plugin.toml"
    ui_dir = plugin_root / "ui"

    if not manifest_path.is_file():
        raise SystemExit(f"missing plugin manifest: {manifest_path}")
    if not ui_dir.is_dir():
        raise SystemExit(f"missing ui directory: {ui_dir}")
    if not library_path.is_file():
        raise SystemExit(f"missing plugin library: {library_path}")
    if native_lib_dir is not None and not native_lib_dir.is_dir():
        raise SystemExit(f"missing native lib directory: {native_lib_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, "w") as archive:
        archive.add(manifest_path, arcname="plugin.toml")
        archive.add(library_path, arcname=library_path.name)
        add_directory(archive, plugin_root, pathlib.Path("ui"))
        if native_lib_dir is not None:
            add_runtime_libraries(archive, native_lib_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())