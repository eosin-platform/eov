#!/usr/bin/env bash
# Builds the eov AppImage.
#
# Release builds run in an EL8 container (glibc 2.28, see
# install-build-deps-el8.sh and .github/workflows/build-appimage.yml) so the
# AppImage runs on every distribution with glibc 2.28 or newer.
#
# Libraries are bundled into usr/lib and found through RUNPATH ($ORIGIN), not
# LD_LIBRARY_PATH, so nothing leaks into programs eov starts. Libraries every
# desktop provides (glibc, libstdc++, X11/Wayland/GL, fontconfig, ...) are
# never bundled: an old copy loaded first would break newer system
# components, such as GPU drivers, that need the system's version.
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd)"

APP_NAME="${APP_NAME:-eov}"
APP_ID="${APP_ID:-io.eosin.eov}"
PRODUCT_NAME="${PRODUCT_NAME:-eov}"
DESKTOP_FILE="${DESKTOP_FILE:-$REPO_ROOT/assets/linux/$APP_ID.desktop}"
METAINFO_FILE="${METAINFO_FILE:-$REPO_ROOT/assets/linux/$APP_ID.metainfo.xml}"
# Slide formats for file managers (shared-mime-info; see app/src/file_types).
MIME_FILE="${MIME_FILE:-$REPO_ROOT/assets/linux/$APP_ID.xml}"
THIRD_PARTY_FILE="${THIRD_PARTY_FILE:-$REPO_ROOT/assets/COPYING.third-party}"
LICENSES_DIR="${LICENSES_DIR:-$REPO_ROOT/assets/licenses}"
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
APPDIR_MIME_DIR="$APPDIR_SHARE_DIR/mime/packages"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BINARY_PATH="${BINARY_PATH:-$CARGO_TARGET_DIR/release/$APP_NAME}"
OPENSLIDE_LIB_PATH="${OPENSLIDE_LIB_PATH:-}"
OPENSLIDE_PREFIX="${OPENSLIDE_PREFIX:-$BUILD_ROOT/openslide-prefix}"
OPENSLIDE_BUILD_ROOT="${OPENSLIDE_BUILD_ROOT:-$BUILD_ROOT/openslide-build}"
ARCH="${ARCH:-$(uname -m)}"
# The newest glibc the AppImage may require (e.g. 2.28). Unset for local
# builds, which only report the requirement.
APPIMAGE_MAX_GLIBC="${APPIMAGE_MAX_GLIBC:-}"
# A pinned AppImage type-2 runtime; without it appimagetool downloads the
# current one.
APPIMAGE_RUNTIME_FILE="${APPIMAGE_RUNTIME_FILE:-}"
APPIMAGETOOL="${APPIMAGETOOL:-appimagetool}"
OUTPUT_PATH="${OUTPUT_PATH:-$DIST_DIR/${APP_ID}-${ARCH}.AppImage}"

# Libraries that are never bundled: the glibc family plus the AppImage
# community excludelist (https://github.com/AppImageCommunity/pkg2appimage,
# file "excludelist"). Matched as file-name prefixes.
EXCLUDED_LIBRARIES=(
    'linux-vdso.so.'
    'ld-linux'
    'ld64.so'
    'libc.so.'
    'libdl.so.'
    'libm.so.'
    'libmvec.so.'
    'libpthread.so.'
    'librt.so.'
    'libutil.so.'
    'libanl.so.'
    'libcidn.so.'
    'libnss_'
    'libresolv.so.'
    'libthread_db.so.'
    'libBrokenLocale.so.'
    'libstdc++.so.'
    'libgcc_s.so.'
    'libGL.so.'
    'libEGL.so.'
    'libGLdispatch.so.'
    'libGLX.so.'
    'libOpenGL.so.'
    'libdrm.so.'
    'libglapi.so.'
    'libgbm.so.'
    'libxcb.so.'
    'libxcb-dri2.so.'
    'libxcb-dri3.so.'
    'libX11.so.'
    'libX11-xcb.so.'
    'libwayland-client.so.'
    'libasound.so.'
    'libfontconfig.so.'
    'libfreetype.so.'
    'libharfbuzz.so.'
    'libfribidi.so.'
    'libcom_err.so.'
    'libexpat.so.'
    'libgpg-error.so.'
    'libgmp.so.'
    'libICE.so.'
    'libSM.so.'
    'libusb-1.0.so.'
    'libuuid.so.'
    'libz.so.'
    'libjack.so.'
    'libpipewire-0.3.so.'
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

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

configure_openslide_env() {
    if [[ -z "$OPENSLIDE_LIB_PATH" ]]; then
        log "Building OpenSlide from the pinned Yanstart fork commit"
        OPENSLIDE_PREFIX="$OPENSLIDE_PREFIX" \
        OPENSLIDE_BUILD_ROOT="$OPENSLIDE_BUILD_ROOT" \
            bash "$REPO_ROOT/packaging/shared/build-openslide.sh"

        OPENSLIDE_LIB_PATH="$(find "$OPENSLIDE_PREFIX/lib" -maxdepth 1 -name 'libopenslide.so*' -type f | sort | head -n 1)"
        [[ -n "$OPENSLIDE_LIB_PATH" ]] || fail "failed to locate OpenSlide library in $OPENSLIDE_PREFIX/lib"
        prepend_env_path PKG_CONFIG_PATH "$OPENSLIDE_PREFIX/lib/pkgconfig"
    fi

    local openslide_dir
    openslide_dir="$(dirname "$(readlink -f "$OPENSLIDE_LIB_PATH")")"
    # Only for this build (cargo, ldd); the AppImage itself relies on RUNPATH.
    prepend_env_path LD_LIBRARY_PATH "$openslide_dir"
    prepend_env_path LIBRARY_PATH "$openslide_dir"
    export OPENSLIDE_LIB_DIR="$openslide_dir"
    export RUSTFLAGS="-L native=$openslide_dir ${RUSTFLAGS:-}"
}

is_excluded_library() {
    local base
    base="$(basename "$1")"

    for pattern in "${EXCLUDED_LIBRARIES[@]}"; do
        if [[ "$base" == "$pattern"* ]]; then
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

        if is_excluded_library "$dependency"; then
            dependency="$(readlink -f "$dependency")"
            if [[ -z "${SKIPPED_LIBS[$dependency]:-}" ]]; then
                log "Leaving system library $(basename "$dependency") to the host"
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
    log "Copying desktop metadata, icon, licences and notices"

    install -Dm644 "$DESKTOP_FILE" "$APPDIR/$APP_ID.desktop"
    install -Dm644 "$DESKTOP_FILE" "$APPDIR_APPLICATIONS_DIR/$APP_ID.desktop"
    install -Dm644 "$METAINFO_FILE" "$APPDIR_METAINFO_DIR/$APP_ID.metainfo.xml"
    # Desktop integration tools install it with the desktop entry; without
    # one, `eov register-file-types` does.
    install -Dm644 "$MIME_FILE" "$APPDIR_MIME_DIR/$APP_ID.xml"

    local license
    for license in LICENSE-MIT LICENSE-APACHE-2.0 LICENSE-GPLv3; do
        install -Dm644 "$REPO_ROOT/$license" "$APPDIR_DOC_DIR/$license"
    done
    install -Dm644 "$THIRD_PARTY_FILE" "$APPDIR_DOC_DIR/COPYING.third-party"
    mkdir -p "$APPDIR_DOC_DIR/licenses"
    cp -a "$LICENSES_DIR/." "$APPDIR_DOC_DIR/licenses/"

    require_file "$ICON_SOURCE"
    install -Dm644 "$ICON_SOURCE" "$APPDIR/$APP_ID.png"
    install -Dm644 "$ICON_SOURCE" "$APPDIR_ICON_DIR/$APP_ID.png"
    ln -sf "$APP_ID.png" "$APPDIR/.DirIcon"
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
        cargo build --release --locked --bin "$APP_NAME" --no-default-features --features "distribution-appimage"
    )

    require_file "$BINARY_PATH"
    install -Dm755 "$BINARY_PATH" "$APPDIR_BIN_DIR/$APP_NAME"
}

# eov finds usr/lib through its RUNPATH; each bundled library finds its
# bundled dependencies next to itself. RUNPATH is not inherited, so every
# ELF file needs its own.
set_runpaths() {
    log "Setting RUNPATH on the binary and bundled libraries"
    patchelf --set-rpath '$ORIGIN/../lib' "$APPDIR_BIN_DIR/$APP_NAME"

    local library
    while IFS= read -r -d '' library; do
        patchelf --set-rpath '$ORIGIN' "$library"
    done < <(find "$APPDIR_LIB_DIR" -type f -name '*.so*' -print0)
}

verify_appdir() {
    local binary="$APPDIR_BIN_DIR/$APP_NAME"
    log "Verifying the AppDir without LD_LIBRARY_PATH"

    readelf -d "$binary" | grep -q 'RUNPATH.*\$ORIGIN/\.\./lib' \
        || fail "$binary has no \$ORIGIN/../lib RUNPATH"

    local resolution
    resolution="$(env -u LD_LIBRARY_PATH ldd "$binary")"
    if grep -q 'not found' <<<"$resolution"; then
        echo "$resolution" >&2
        fail "$APP_NAME has unresolved libraries without LD_LIBRARY_PATH"
    fi
    local openslide_resolved lib_dir
    openslide_resolved="$(awk '/libopenslide\.so/ && $3 ~ /^\// { print $3; exit }' <<<"$resolution")"
    lib_dir="$(readlink -f "$APPDIR_LIB_DIR")"
    [[ -n "$openslide_resolved" && "$(readlink -f "$openslide_resolved")" == "$lib_dir"/* ]] \
        || fail "libopenslide does not resolve to the bundled copy: $(grep libopenslide <<<"$resolution")"

    local library
    while IFS= read -r -d '' library; do
        if env -u LD_LIBRARY_PATH ldd "$library" | grep -q 'not found'; then
            env -u LD_LIBRARY_PATH ldd "$library" >&2
            fail "bundled $(basename "$library") has unresolved dependencies"
        fi
    done < <(find "$APPDIR_LIB_DIR" -type f -name '*.so*' -print0)

    local version_output
    version_output="$(env -u LD_LIBRARY_PATH "$APPDIR/AppRun" --version)" \
        || fail "the AppDir does not start: $APPDIR/AppRun --version failed"
    log "Smoke test: $version_output"

    bash "$SCRIPT_DIR/check-glibc.sh" "$APPDIR" ${APPIMAGE_MAX_GLIBC:+"$APPIMAGE_MAX_GLIBC"}
}

# Licence files an RPM package ships: its %license files and the licence-like
# %doc files (many EL8 packages keep COPYING under %doc), falling back to the
# -devel package and then to all its documentation (SQLite, public domain,
# only has a README).
rpm_license_files() {
    local package="$1" candidate found
    for candidate in "$package" "$package-devel"; do
        rpm -q "$candidate" >/dev/null 2>&1 || continue
        found="$({
            rpm -qL "$candidate" 2>/dev/null
            rpm -qd "$candidate" 2>/dev/null | grep -Ei '/(copying|copyright|licen[cs]e|notice)[^/]*$' || true
        } | while IFS= read -r file; do [[ -f "$file" ]] && printf '%s\n' "$file"; done)"
        if [[ -n "$found" ]]; then
            printf '%s\n' "$found"
            return 0
        fi
    done
    rpm -qd "$package" 2>/dev/null | while IFS= read -r file; do [[ -f "$file" ]] && printf '%s\n' "$file"; done
}

# Last resort for packages that install no licence file at all (EL8's libxcb):
# the licence files of the upstream source in the package's source RPM.
rpm_source_license_files() {
    local package="$1" dest="$2" work archive member found=1
    command -v dnf >/dev/null 2>&1 && command -v rpm2cpio >/dev/null 2>&1 \
        && command -v cpio >/dev/null 2>&1 || return 1
    work="$(mktemp -d)"
    if ! dnf -q download --source --destdir "$work" "$package" >/dev/null 2>&1; then
        rm -rf "$work"
        return 1
    fi
    if ! (cd "$work" && rpm2cpio ./*.src.rpm | cpio -idm --quiet 2>/dev/null); then
        rm -rf "$work"
        return 1
    fi
    for archive in "$work"/*.tar.*; do
        [[ -f "$archive" ]] || continue
        while IFS= read -r member; do
            mkdir -p "$dest"
            if tar -xf "$archive" -C "$work" "$member" && cp "$work/$member" "$dest/"; then
                found=0
            fi
        done < <(tar -tf "$archive" | grep -Ei '^[^/]+/(copying|copyright|licen[cs]e)[^/]*$' || true)
    done
    rm -rf "$work"
    return "$found"
}

# The licence files of every bundled library, taken from the build system's
# package database (RPM on the EL8 release baseline, dpkg on Debian and
# Ubuntu), plus those of the OpenSlide and libdicom sources built here, into
# usr/share/doc/eov/third-party/<package>/.
collect_bundled_licenses() {
    local dest="$APPDIR_DOC_DIR/third-party"
    local library package file missing=0
    local -A packages=()
    mkdir -p "$dest"
    log "Collecting licence files of the bundled libraries"

    for library in "${!COPIED_LIBS[@]}"; do
        case "$library" in
            "$(readlink -f "$OPENSLIDE_PREFIX")"/*|"$OPENSLIDE_PREFIX"/*) continue ;;
        esac
        package=""
        if command -v rpm >/dev/null 2>&1; then
            package="$(rpm -qf --queryformat '%{NAME}\n' "$library" 2>/dev/null | head -n 1)" || package=""
            [[ "$package" == *"not owned"* ]] && package=""
        elif command -v dpkg-query >/dev/null 2>&1; then
            package="$(dpkg-query -S "$library" 2>/dev/null | head -n 1 | cut -d: -f1)" || package=""
        fi
        if [[ -z "$package" ]]; then
            log "  no package owns $(basename "$library"); its licence is not collected"
            missing=$((missing + 1))
            continue
        fi
        packages[$package]=1
    done

    for package in "${!packages[@]}"; do
        mkdir -p "$dest/$package"
        if command -v rpm >/dev/null 2>&1; then
            while IFS= read -r file; do
                [[ -n "$file" ]] && cp -L "$file" "$dest/$package/"
            done < <(rpm_license_files "$package")
        elif [[ -f "/usr/share/doc/$package/copyright" ]]; then
            cp -L "/usr/share/doc/$package/copyright" "$dest/$package/"
        fi
        if [[ -z "$(ls -A "$dest/$package")" ]] \
            && command -v rpm >/dev/null 2>&1 \
            && rpm_source_license_files "$package" "$dest/$package"; then
            log "  package $package installs no licence file; took it from its source RPM"
        fi
        if [[ -z "$(ls -A "$dest/$package")" ]]; then
            rmdir "$dest/$package"
            log "  package $package lists no licence files"
            missing=$((missing + 1))
        fi
    done

    local source_dir name
    for source_dir in "$OPENSLIDE_BUILD_ROOT/openslide-src" "$OPENSLIDE_BUILD_ROOT"/libdicom-src/libdicom-*; do
        [[ -d "$source_dir" ]] || continue
        name="$(basename "$source_dir")"
        name="${name%-src}"
        for file in "$source_dir"/COPYING* "$source_dir"/LICENSE*; do
            [[ -f "$file" ]] || continue
            mkdir -p "$dest/$name"
            cp -L "$file" "$dest/$name/"
        done
    done

    log "  licence files for ${#packages[@]} package(s) in ${dest#"$APPDIR"/}"
    if [[ "$missing" -gt 0 && "${APPIMAGE_REQUIRE_LICENSES:-0}" == "1" ]]; then
        fail "$missing bundled library or package without licence files"
    fi
}

run_appimagetool() {
    require_command "$APPIMAGETOOL"
    local -a args=()
    if [[ -n "$APPIMAGE_RUNTIME_FILE" ]]; then
        require_file "$APPIMAGE_RUNTIME_FILE"
        args+=(--runtime-file "$APPIMAGE_RUNTIME_FILE")
    fi

    log "Creating $(basename "$OUTPUT_PATH") with $APPIMAGETOOL"
    rm -f "$OUTPUT_PATH"
    ARCH="$ARCH" "$APPIMAGETOOL" "${args[@]}" "$APPDIR" "$OUTPUT_PATH"
    require_file "$OUTPUT_PATH"
}

main() {
    log "Starting AppImage build for $PRODUCT_NAME ($APP_ID)"
    require_file "$DESKTOP_FILE"
    require_file "$METAINFO_FILE"
    require_file "$THIRD_PARTY_FILE"
    require_command patchelf
    require_command readelf

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
    set_runpaths
    collect_bundled_licenses
    verify_appdir
    run_appimagetool

    log "AppImage build completed: $OUTPUT_PATH"
}

main "$@"
