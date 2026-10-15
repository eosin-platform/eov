#!/usr/bin/env bash
# Downloads the pinned appimagetool and AppImage type-2 runtime for ARCH
# (x86_64 or aarch64) into a directory and verifies their SHA-256s.
#
#   packaging/appimage/fetch-tools.sh <dir>
#
# Prints APPIMAGETOOL=... and APPIMAGE_RUNTIME_FILE=... lines for
# packaging/appimage/build.sh (CI appends them to $GITHUB_ENV).
#
# To update, pick versioned releases of AppImage/appimagetool and
# AppImage/type2-runtime and copy each asset's SHA-256 from the release page
# (GitHub shows a digest per asset).
set -euo pipefail

APPIMAGETOOL_VERSION="1.9.1"
RUNTIME_VERSION="20251108"

dest="${1:?usage: fetch-tools.sh <dir>}"
arch="${ARCH:-$(uname -m)}"

case "$arch" in
    x86_64)
        appimagetool_sha="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
        runtime_sha="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"
        ;;
    aarch64|arm64)
        arch="aarch64"
        appimagetool_sha="f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158"
        runtime_sha="00cbdfcf917cc6c0ff6d3347d59e0ca1f7f45a6df1a428a0d6d8a78664d87444"
        ;;
    *)
        echo "fetch-tools.sh: unsupported architecture $arch" >&2
        exit 1
        ;;
esac

fetch() {
    local url="$1" sha="$2" path="$3"
    curl -fsSL --retry 3 --retry-delay 5 -o "$path.partial" "$url"
    if ! echo "$sha  $path.partial" | sha256sum --check --status; then
        rm -f "$path.partial"
        echo "fetch-tools.sh: checksum mismatch for $url" >&2
        exit 1
    fi
    mv -f "$path.partial" "$path"
}

mkdir -p "$dest"
dest="$(CDPATH='' cd -- "$dest" && pwd)"
fetch "https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-$arch.AppImage" \
    "$appimagetool_sha" "$dest/appimagetool"
fetch "https://github.com/AppImage/type2-runtime/releases/download/$RUNTIME_VERSION/runtime-$arch" \
    "$runtime_sha" "$dest/runtime-$arch"
chmod 755 "$dest/appimagetool"

echo "APPIMAGETOOL=$dest/appimagetool"
echo "APPIMAGE_RUNTIME_FILE=$dest/runtime-$arch"
