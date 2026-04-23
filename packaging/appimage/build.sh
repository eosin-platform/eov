#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

APP_NAME="${APP_NAME:-eov}"
APP_ID="${APP_ID:-io.eosin.eov}"
PRODUCT_NAME="${PRODUCT_NAME:-eov}"
DESKTOP_FILE="${DESKTOP_FILE:-$REPO_ROOT/assets/linux/$APP_ID.desktop}"
METAINFO_FILE="${METAINFO_FILE:-$REPO_ROOT/assets/linux/$APP_ID.metainfo.xml}"
THIRD_PARTY_FILE="${THIRD_PARTY_FILE:-$REPO_ROOT/assets/linux/COPYING.third-party}"
LICENSES_DIR="${LICENSES_DIR:-$REPO_ROOT/assets/linux/licenses}"
ICON_SOURCE="${ICON_SOURCE:-$REPO_ROOT/images/logo_256.png}"
BUILD_ROOT="${BUILD_ROOT:-$REPO_ROOT/packaging/appimage/build}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
APPDIR="${APPDIR:-$BUILD_ROOT/AppDir}"
APPDIR_LIB_DIR="$APPDIR/usr/lib"
APPDIR_BIN_DIR="$APPDIR/usr/bin"
APPDIR_SHARE_DIR="$APPDIR/usr/share"
APPDIR_DOC_DIR="$APPDIR_SHARE_DIR/doc/$APP_NAME"
APPDIR_ICON_DIR="$APPDIR_SHARE_DIR/icons/hicolor/256x256/apps"
APPDIR_APPLICATIONS_DIR="$APPDIR_SHARE_DIR/applications"
APPDIR_METAINFO_DIR="$APPDIR_SHARE_DIR/metainfo"
BINARY_PATH="${BINARY_PATH:-$REPO_ROOT/target/release/$APP_NAME}"
OPENSLIDE_LIB_PATH="${OPENSLIDE_LIB_PATH:-}"
OPENSLIDE_PREFIX="${OPENSLIDE_PREFIX:-$BUILD_ROOT/openslide-prefix}"
OPENSLIDE_BUILD_ROOT="${OPENSLIDE_BUILD_ROOT:-$BUILD_ROOT/openslide-build}"
ARCH="${ARCH:-$(uname -m)}"

BLACKLISTED_LIBRARIES=(
    'linux-vdso.so.'
    'ld-linux'
    'ld64.so'
    'libc.so.'
    'libdl.so.'
    'libm.so.'
    'libpthread.so.'
    'librt.so.'
    'libutil.so.'
    'libanl.so.'
    'libnss_'
    'libresolv.so.'
    'libBrokenLocale.so.'
)

declare -A COPIED_LIBS=()
declare -A SCANNED_ELF_TARGETS=()
declare -A SKIPPED_LIBS=()

log() {
    echo "[appimage] $*"
}

fail() {
    echo "[appimage] error: $*" >&2
    exit 1
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

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "required file not found: $path"
}

configure_openslide_env() {
    if [[ -z "$OPENSLIDE_LIB_PATH" ]]; then
        log "Building OpenSlide from openslide/main"
        OPENSLIDE_PREFIX="$OPENSLIDE_PREFIX" \
        OPENSLIDE_BUILD_ROOT="$OPENSLIDE_BUILD_ROOT" \
            bash "$REPO_ROOT/packaging/shared/build-openslide.sh"

        OPENSLIDE_LIB_PATH="$(find "$OPENSLIDE_PREFIX/lib" -maxdepth 1 -name 'libopenslide.so*' -type f | sort | head -n 1)"
        [[ -n "$OPENSLIDE_LIB_PATH" ]] || fail "failed to locate OpenSlide library in $OPENSLIDE_PREFIX/lib"
        prepend_env_path PKG_CONFIG_PATH "$OPENSLIDE_PREFIX/lib/pkgconfig"
    fi

    prepend_env_path LD_LIBRARY_PATH "$(dirname "$(readlink -f "$OPENSLIDE_LIB_PATH")")"
    prepend_env_path LIBRARY_PATH "$(dirname "$(readlink -f "$OPENSLIDE_LIB_PATH")")"
    export RUSTFLAGS="-L native=$(dirname "$(readlink -f "$OPENSLIDE_LIB_PATH")") ${RUSTFLAGS:-}"
}

is_blacklisted_library() {
    local base
    base="$(basename "$1")"

    for pattern in "${BLACKLISTED_LIBRARIES[@]}"; do
        if [[ "$base" == ${pattern}* ]]; then
            return 0
        fi
    done

    return 1
}

copy_library_file() {
    local source="$1"
    local resolved base_source base_resolved

    require_file "$source"
    resolved="$(readlink -f "$source")"
    require_file "$resolved"

    base_source="$(basename "$source")"
    base_resolved="$(basename "$resolved")"

    if [[ -n "${COPIED_LIBS[$resolved]:-}" ]]; then
        if [[ "$base_source" != "$base_resolved" && ! -e "$APPDIR_LIB_DIR/$base_source" ]]; then
            ln -s "$base_resolved" "$APPDIR_LIB_DIR/$base_source"
        fi
        return 0
    fi

    log "Bundling library $(basename "$resolved")"
    install -Dm755 "$resolved" "$APPDIR_LIB_DIR/$base_resolved"
    COPIED_LIBS[$resolved]=1

    if [[ "$base_source" != "$base_resolved" && ! -e "$APPDIR_LIB_DIR/$base_source" ]]; then
        ln -s "$base_resolved" "$APPDIR_LIB_DIR/$base_source"
    fi
}

bundle_libraries_for() {
    local target="$1"
    local resolved_target
    local dependency

    resolved_target="$(readlink -f "$target")"
    require_file "$resolved_target"

    if [[ -n "${SCANNED_ELF_TARGETS[$resolved_target]:-}" ]]; then
        return 0
    fi
    SCANNED_ELF_TARGETS[$resolved_target]=1

    while IFS= read -r dependency; do
        [[ -n "$dependency" ]] || continue
        [[ -f "$dependency" ]] || continue

        if is_blacklisted_library "$dependency"; then
            dependency="$(readlink -f "$dependency")"
            if [[ -z "${SKIPPED_LIBS[$dependency]:-}" ]]; then
                log "Skipping blacklisted system library $(basename "$dependency")"
                SKIPPED_LIBS[$dependency]=1
            fi
            continue
        fi

        copy_library_file "$dependency"
        bundle_libraries_for "$dependency"
    done < <(
        ldd "$resolved_target" \
            | awk '
                /=>/ && $(NF-1) ~ /^\// { print $(NF-1) }
                /^[[:space:]]*\// { print $1 }
            ' \
            | sort -u
    )
}

resolve_openslide_library() {
    local candidate

    if [[ -n "$OPENSLIDE_LIB_PATH" ]]; then
        echo "$OPENSLIDE_LIB_PATH"
        return 0
    fi

    candidate="$({ ldd "$BINARY_PATH" || true; } | awk '/libopenslide\.so/ && $(NF-1) ~ /^\// { print $(NF-1); exit }')"
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    candidate="$({ ldconfig -p 2>/dev/null || true; } | awk '/libopenslide\.so(\.|$)/ { print $NF; exit }')"
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    fail "could not locate libopenslide.so; set OPENSLIDE_LIB_PATH explicitly"
}

copy_metadata() {
    log "Copying desktop metadata, icon, and notices"

    install -Dm644 "$DESKTOP_FILE" "$APPDIR/$APP_ID.desktop"
    install -Dm644 "$DESKTOP_FILE" "$APPDIR_APPLICATIONS_DIR/$APP_ID.desktop"
    install -Dm644 "$METAINFO_FILE" "$APPDIR_METAINFO_DIR/$APP_ID.metainfo.xml"
    install -Dm644 "$THIRD_PARTY_FILE" "$APPDIR_DOC_DIR/COPYING.third-party"

    if [[ -d "$LICENSES_DIR" ]]; then
        mkdir -p "$APPDIR_DOC_DIR/licenses"
        cp -a "$LICENSES_DIR/." "$APPDIR_DOC_DIR/licenses/"
    fi

    if [[ -f "$ICON_SOURCE" ]]; then
        install -Dm644 "$ICON_SOURCE" "$APPDIR/$APP_ID.png"
        install -Dm644 "$ICON_SOURCE" "$APPDIR_ICON_DIR/$APP_ID.png"
    else
        log "Icon not found at $ICON_SOURCE; leaving icon install step empty"
    fi
}

create_layout() {
    log "Preparing AppDir layout at $APPDIR"
    rm -rf "$APPDIR"
    mkdir -p \
        "$APPDIR_BIN_DIR" \
        "$APPDIR_LIB_DIR" \
        "$APPDIR_DOC_DIR" \
        "$APPDIR_ICON_DIR" \
        "$APPDIR_APPLICATIONS_DIR" \
        "$APPDIR_METAINFO_DIR" \
        "$DIST_DIR"

    install -Dm755 "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
}

build_binary() {
    log "Building $APP_NAME with cargo build --release"

    (
        cd "$REPO_ROOT"
        cargo build --release --bin "$APP_NAME"
    )

    require_file "$BINARY_PATH"
    install -Dm755 "$BINARY_PATH" "$APPDIR_BIN_DIR/$APP_NAME"
}

run_linuxdeploy() {
    log "linuxdeploy detected; using it to finish dependency collection and create the AppImage"

    (
        cd "$DIST_DIR"
        local -a linuxdeploy_args=(
            --appdir "$APPDIR"
            --desktop-file "$APPDIR/$APP_ID.desktop"
            --custom-apprun "$SCRIPT_DIR/AppRun"
            --deploy-deps-only "$APPDIR_BIN_DIR/$APP_NAME"
            --deploy-deps-only "$APPDIR_LIB_DIR"
            --output appimage
        )

        if [[ -f "$APPDIR/$APP_ID.png" ]]; then
            linuxdeploy_args+=(--icon-file "$APPDIR/$APP_ID.png")
        fi

        linuxdeploy "${linuxdeploy_args[@]}"
    )
}

run_appimagetool() {
    local output="$DIST_DIR/${APP_ID}-${ARCH}.AppImage"

    command -v appimagetool >/dev/null 2>&1 || fail "linuxdeploy not found and appimagetool is not installed"

    log "Creating AppImage with appimagetool"
    ARCH="$ARCH" appimagetool "$APPDIR" "$output"
}

main() {
    log "Starting AppImage build for $PRODUCT_NAME ($APP_ID)"
    require_file "$DESKTOP_FILE"
    require_file "$METAINFO_FILE"
    require_file "$THIRD_PARTY_FILE"

    create_layout
    configure_openslide_env
    build_binary
    copy_metadata

    local openslide_source
    openslide_source="$(resolve_openslide_library)"
    log "Using OpenSlide shared library at $openslide_source"

    copy_library_file "$openslide_source"
    bundle_libraries_for "$BINARY_PATH"
    bundle_libraries_for "$openslide_source"

    if command -v linuxdeploy >/dev/null 2>&1; then
        run_linuxdeploy "$openslide_source"
    else
        run_appimagetool
    fi

    log "AppImage build completed; artifacts are in $DIST_DIR"
    log "If OpenSlide is ever patched locally, publish those patches under LGPL-2.1 terms."
}

main "$@"
