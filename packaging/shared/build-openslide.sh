#!/usr/bin/env bash
# Builds the OpenSlide fork eov ships, plus the libdicom and uthash it needs,
# into OPENSLIDE_PREFIX. Used by the AppImage, macOS, Windows (MSYS2) and CI
# builds; the Flatpak manifest pins the same sources.
#
# Every input is pinned: the OpenSlide fork by full commit id (verified after
# checkout), libdicom and uthash by version and SHA-256. Keep these values in
# sync with packaging/flatpak/io.eosin.eov.yml and assets/COPYING.third-party
# (CI checks that they agree).
set -euo pipefail

DEFAULT_OPENSLIDE_REPO_URL="https://github.com/Yanstart/openslide.git"
DEFAULT_OPENSLIDE_COMMIT="00bebfb6a412485f7fbfe7fdb5919a3868594c13"
DEFAULT_LIBDICOM_VERSION="1.2.0"
DEFAULT_LIBDICOM_SHA256="3b8c05ceb6bf667fed997f23b476dd32c3dc6380eee1998185c211d86a7b4918"
DEFAULT_UTHASH_VERSION="2.3.0"
DEFAULT_UTHASH_SHA256="e10382ab75518bad8319eb922ad04f907cb20cccb451a3aa980c9d005e661acc"

OPENSLIDE_REPO_URL="${OPENSLIDE_REPO_URL:-$DEFAULT_OPENSLIDE_REPO_URL}"
OPENSLIDE_COMMIT="${OPENSLIDE_COMMIT:-$DEFAULT_OPENSLIDE_COMMIT}"
OPENSLIDE_PREFIX="${OPENSLIDE_PREFIX:-$(pwd)/.openslide-prefix}"
OPENSLIDE_BUILD_ROOT="${OPENSLIDE_BUILD_ROOT:-$(pwd)/.openslide-build}"
LIBDICOM_VERSION="${LIBDICOM_VERSION:-$DEFAULT_LIBDICOM_VERSION}"
UTHASH_VERSION="${UTHASH_VERSION:-$DEFAULT_UTHASH_VERSION}"

log() {
    echo "[openslide-build] $*"
}

fail() {
    echo "[openslide-build] error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

prepend_env_path() {
    local var_name="$1"
    local dir="$2"
    local current="${!var_name:-}"

    if [[ -n "$current" ]]; then
        export "$var_name=$dir:$current"
    else
        export "$var_name=$dir"
    fi
}

# A checksum is only known for the default version; overriding the version
# requires overriding the checksum too, so nothing is ever fetched unverified.
resolve_checksum() {
    local variable="$1" version="$2" default_version="$3" default_sha="$4" override="$5"
    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
    elif [[ "$version" == "$default_version" ]]; then
        printf '%s\n' "$default_sha"
    else
        fail "version $version has no pinned checksum; set $variable as well"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download_verified() {
    local url="$1" expected="$2" destination="$3"
    local partial="$destination.partial"

    rm -f "$partial"
    curl -fsSL --retry 3 --retry-delay 5 -o "$partial" "$url"
    local actual
    actual="$(sha256_of "$partial")"
    if [[ "$actual" != "$(printf '%s' "$expected" | tr 'A-F' 'a-f')" ]]; then
        rm -f "$partial"
        fail "checksum mismatch for $url: expected $expected, got $actual"
    fi
    mv -f "$partial" "$destination"
}

install_uthash_headers() {
    local source_root="$1"
    local prefix="$2"
    local archive_path="$source_root/uthash-$UTHASH_VERSION.tar.gz"
    local extract_dir="$source_root/uthash-src"
    local sha
    sha="$(resolve_checksum UTHASH_SHA256 "$UTHASH_VERSION" "$DEFAULT_UTHASH_VERSION" \
        "$DEFAULT_UTHASH_SHA256" "${UTHASH_SHA256:-}")"

    log "Installing uthash $UTHASH_VERSION headers"
    download_verified "https://github.com/troydhanson/uthash/archive/v$UTHASH_VERSION.tar.gz" \
        "$sha" "$archive_path"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir" "$prefix/include"
    # Only src/: the archive's include/ is a symlink, which MSYS2 tar cannot
    # create.
    tar -xzf "$archive_path" -C "$extract_dir" "uthash-$UTHASH_VERSION/src"

    local header
    for header in uthash.h utarray.h utlist.h utringbuffer.h utstack.h; do
        install -m 644 "$extract_dir/uthash-$UTHASH_VERSION/src/$header" "$prefix/include/$header"
    done
}

build_libdicom() {
    local source_root="$1"
    local prefix="$2"
    local archive_path="$source_root/libdicom-$LIBDICOM_VERSION.tar.xz"
    local extract_dir="$source_root/libdicom-src"
    local source_dir="$extract_dir/libdicom-$LIBDICOM_VERSION"
    local sha
    sha="$(resolve_checksum LIBDICOM_SHA256 "$LIBDICOM_VERSION" "$DEFAULT_LIBDICOM_VERSION" \
        "$DEFAULT_LIBDICOM_SHA256" "${LIBDICOM_SHA256:-}")"

    log "Building libdicom $LIBDICOM_VERSION"
    download_verified \
        "https://github.com/ImagingDataCommons/libdicom/releases/download/v$LIBDICOM_VERSION/libdicom-$LIBDICOM_VERSION.tar.xz" \
        "$sha" "$archive_path"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    tar -xJf "$archive_path" -C "$extract_dir"

    meson setup "$source_dir/_build" "$source_dir" \
        --prefix="$prefix" \
        --libdir=lib \
        --buildtype=release \
        -Dtests=false \
        --wrap-mode=nofallback
    meson compile -C "$source_dir/_build"
    meson install -C "$source_dir/_build"
}

build_openslide() {
    local source_root="$1"
    local prefix="$2"
    local source_dir="$source_root/openslide-src"

    [[ "$OPENSLIDE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
        || fail "OPENSLIDE_COMMIT must be a full 40-character commit id, got '$OPENSLIDE_COMMIT'"

    log "Building OpenSlide from $OPENSLIDE_REPO_URL at $OPENSLIDE_COMMIT"
    rm -rf "$source_dir"
    git init -q "$source_dir"
    git -C "$source_dir" remote add origin "$OPENSLIDE_REPO_URL"
    git -C "$source_dir" fetch -q --depth 1 origin "$OPENSLIDE_COMMIT"
    git -C "$source_dir" checkout -q --detach FETCH_HEAD
    local checked_out
    checked_out="$(git -C "$source_dir" rev-parse HEAD)"
    [[ "$checked_out" == "$OPENSLIDE_COMMIT" ]] \
        || fail "fetched OpenSlide commit $checked_out, expected $OPENSLIDE_COMMIT"

    meson setup "$source_dir/_build" "$source_dir" \
        --prefix="$prefix" \
        --libdir=lib \
        --buildtype=release \
        -Dtest=disabled \
        --wrap-mode=nofallback
    meson compile -C "$source_dir/_build"
    meson install -C "$source_dir/_build"
}

write_provenance() {
    local prefix="$1"
    mkdir -p "$prefix/share/eov"
    cat > "$prefix/share/eov/openslide-build.txt" <<EOF
OpenSlide: $OPENSLIDE_REPO_URL @ $OPENSLIDE_COMMIT
libdicom: $LIBDICOM_VERSION
uthash: $UTHASH_VERSION (headers, build time only)
EOF
}

main() {
    require_command curl
    require_command git
    require_command meson
    require_command tar

    mkdir -p "$OPENSLIDE_PREFIX" "$OPENSLIDE_BUILD_ROOT"

    prepend_env_path PKG_CONFIG_PATH "$OPENSLIDE_PREFIX/lib/pkgconfig"
    prepend_env_path LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
    prepend_env_path CPATH "$OPENSLIDE_PREFIX/include"

    case "$(uname -s)" in
        Darwin)
            # Build the libraries for the macOS version the bundle declares
            # (assets/macos/Info.plist), not for the build machine's.
            export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
            log "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
            prepend_env_path DYLD_LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
            ;;
        Linux)
            prepend_env_path LD_LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
            ;;
    esac

    install_uthash_headers "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"
    build_libdicom "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"
    build_openslide "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"
    write_provenance "$OPENSLIDE_PREFIX"

    log "Installed OpenSlide into $OPENSLIDE_PREFIX"
}

main "$@"
