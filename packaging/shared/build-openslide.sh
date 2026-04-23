#!/usr/bin/env bash
set -euo pipefail

OPENSLIDE_REPO_URL="${OPENSLIDE_REPO_URL:-https://github.com/Yanstart/openslide.git}"
OPENSLIDE_COMMIT="${OPENSLIDE_COMMIT:-00bebfb6a412485f7fbfe7fdb5919a3868594c13}"
OPENSLIDE_PREFIX="${OPENSLIDE_PREFIX:-$(pwd)/.openslide-prefix}"
OPENSLIDE_BUILD_ROOT="${OPENSLIDE_BUILD_ROOT:-$(pwd)/.openslide-build}"
LIBDICOM_VERSION="${LIBDICOM_VERSION:-1.2.0}"

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

install_uthash_headers() {
    local source_root="$1"
    local prefix="$2"
    local archive_path="$source_root/uthash.tar.gz"
    local extract_dir="$source_root/uthash-src"

    log "Installing uthash headers"
    curl -fsSL -o "$archive_path" "https://github.com/troydhanson/uthash/archive/v2.3.0.tar.gz"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    tar -xzf "$archive_path" -C "$extract_dir"

    install -Dm644 "$extract_dir/uthash-2.3.0/src/uthash.h" "$prefix/include/uthash.h"
    install -Dm644 "$extract_dir/uthash-2.3.0/src/utarray.h" "$prefix/include/utarray.h"
    install -Dm644 "$extract_dir/uthash-2.3.0/src/utlist.h" "$prefix/include/utlist.h"
    install -Dm644 "$extract_dir/uthash-2.3.0/src/utringbuffer.h" "$prefix/include/utringbuffer.h"
    install -Dm644 "$extract_dir/uthash-2.3.0/src/utstack.h" "$prefix/include/utstack.h"
}

build_libdicom() {
    local source_root="$1"
    local prefix="$2"
    local archive_path="$source_root/libdicom.tar.xz"
    local extract_dir="$source_root/libdicom-src"
    local source_dir="$extract_dir/libdicom-$LIBDICOM_VERSION"

    log "Building libdicom $LIBDICOM_VERSION"
    curl -fsSL -o "$archive_path" "https://github.com/ImagingDataCommons/libdicom/releases/download/v$LIBDICOM_VERSION/libdicom-$LIBDICOM_VERSION.tar.xz"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    tar -xJf "$archive_path" -C "$extract_dir"

    meson setup "$source_dir/_build" "$source_dir" \
        --prefix="$prefix" \
        --libdir=lib \
        -Dtests=false \
        --wrap-mode=nofallback
    meson compile -C "$source_dir/_build"
    meson install -C "$source_dir/_build"
}

build_openslide() {
    local source_root="$1"
    local prefix="$2"
    local source_dir="$source_root/openslide-src"

    log "Building OpenSlide from $OPENSLIDE_REPO_URL at $OPENSLIDE_COMMIT"
    rm -rf "$source_dir"
    git init "$source_dir"
    git -C "$source_dir" remote add origin "$OPENSLIDE_REPO_URL"
    git -C "$source_dir" fetch --depth 1 origin "$OPENSLIDE_COMMIT"
    git -C "$source_dir" checkout --detach FETCH_HEAD

    meson setup "$source_dir/_build" "$source_dir" \
        --prefix="$prefix" \
        --libdir=lib \
        -Dtest=disabled \
        --wrap-mode=nofallback
    meson compile -C "$source_dir/_build"
    meson install -C "$source_dir/_build"
}

main() {
    require_command curl
    require_command git
    require_command meson

    mkdir -p "$OPENSLIDE_PREFIX" "$OPENSLIDE_BUILD_ROOT"

    prepend_env_path PKG_CONFIG_PATH "$OPENSLIDE_PREFIX/lib/pkgconfig"
    prepend_env_path LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
    prepend_env_path CPATH "$OPENSLIDE_PREFIX/include"

    case "$(uname -s)" in
        Darwin)
            prepend_env_path DYLD_LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
            ;;
        Linux)
            prepend_env_path LD_LIBRARY_PATH "$OPENSLIDE_PREFIX/lib"
            ;;
    esac

    install_uthash_headers "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"
    build_libdicom "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"
    build_openslide "$OPENSLIDE_BUILD_ROOT" "$OPENSLIDE_PREFIX"

    log "Installed OpenSlide into $OPENSLIDE_PREFIX"
}

main "$@"