#!/usr/bin/env bash
# Builds eov.app and its release zip.
#
# - eov, eov-update-helper and the libraries built here (OpenSlide, libdicom)
#   target MACOSX_DEPLOYMENT_TARGET (default: the template's
#   LSMinimumSystemVersion).
# - Every library eov loads that does not come with macOS is bundled into
#   Contents/Frameworks, including the ones Homebrew libraries reference
#   through @rpath or @loader_path (libwebp -> @rpath/libsharpyuv.0.dylib),
#   and referenced relative to the bundle (@executable_path, @loader_path).
#   The build fails if any binary still references a library outside the
#   bundle and the OS (e.g. /opt/homebrew, a CI path or @rpath), or if
#   eov-update-helper links anything but system libraries (it runs from a
#   temporary directory during updates).
# - LSMinimumSystemVersion is raised to the newest `minos` of any shipped
#   binary (Homebrew bottles target the build machine's macOS), so macOS
#   refuses to start eov on a release it cannot run on instead of failing in
#   dyld. MACOS_STRICT_MINIMUM=1 makes that an error instead.
# - The bundle is smoke-tested: `eov --version` from a copy of the bundle,
#   with dyld kept from falling back to libraries outside the bundle and macOS
#   and every library it loads checked, and `eov-update-helper --self-test`
#   from a temporary copy.
# - With CODESIGN_IDENTITY set to a Developer ID identity the bundle is signed
#   with the hardened runtime, and with NOTARIZE=1 it is notarized and
#   stapled (credentials: see packaging/macos/README.md).
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd)"

APP_NAME="${APP_NAME:-eov}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-$APP_NAME}"
PRODUCT_NAME="${PRODUCT_NAME:-eov}"
APP_ID="${APP_ID:-io.eosin.eov}"
BUILD_PROFILE="${BUILD_PROFILE:-release}"
RUST_TARGET="${RUST_TARGET:-}"
VERSION="${VERSION:-}"
ARCH_LABEL="${ARCH_LABEL:-}"
APP_BUNDLE_NAME="${APP_BUNDLE_NAME:-$PRODUCT_NAME.app}"
BUILD_ROOT="${BUILD_ROOT:-$REPO_ROOT/packaging/macos/build}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
ICNS_SOURCE="${ICNS_SOURCE:-$REPO_ROOT/assets/macos/eov.icns}"
INFO_PLIST_TEMPLATE="${INFO_PLIST_TEMPLATE:-$REPO_ROOT/assets/macos/Info.plist}"
ENTITLEMENTS_FILE="${ENTITLEMENTS_FILE:-$SCRIPT_DIR/entitlements.plist}"
THIRD_PARTY_FILE="${THIRD_PARTY_FILE:-$REPO_ROOT/assets/COPYING.third-party}"
LICENSES_DIR="${LICENSES_DIR:-$REPO_ROOT/assets/licenses}"
BUNDLE_ICON_NAME="${BUNDLE_ICON_NAME:-$PRODUCT_NAME.icns}"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-}"
CREATE_ZIP="${CREATE_ZIP:-1}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
NOTARIZE="${NOTARIZE:-0}"
MACOS_STRICT_MINIMUM="${MACOS_STRICT_MINIMUM:-0}"
OPENSLIDE_PREFIX="${OPENSLIDE_PREFIX:-$BUILD_ROOT/openslide-prefix}"
OPENSLIDE_BUILD_ROOT="${OPENSLIDE_BUILD_ROOT:-$BUILD_ROOT/openslide-build}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

STAGING_DIR="$BUILD_ROOT/staging"
# Where each bundled dylib came from (for collecting licences).
BUNDLED_SOURCES="$BUILD_ROOT/bundled-sources.txt"
APP_BUNDLE_PATH="$STAGING_DIR/$APP_BUNDLE_NAME"
CONTENTS_DIR="$APP_BUNDLE_PATH/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
MACOS_BINARY_PATH="$MACOS_DIR/$EXECUTABLE_NAME"
MACOS_HELPER_PATH="$MACOS_DIR/eov-update-helper"
ICON_OUTPUT_PATH="$RESOURCES_DIR/$BUNDLE_ICON_NAME"

# The Cargo feature to enable when building the macOS bundle.
# Recommended distribution for macOS is via Homebrew.
DISTRIBUTION_FEATURE_TARGET="${DISTRIBUTION_FEATURE_TARGET:-distribution-macos-bundle}"

log() {
    echo "[macos] $*"
}

warn() {
    echo "[macos] warning: $*" >&2
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::warning title=macOS bundle::$*"
    fi
}

fail() {
    echo "[macos] error: $*" >&2
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

# "a" newer than "b", comparing dotted numeric versions (bash 3.2 safe).
version_gt() {
    local IFS=.
    local -a left right
    read -r -a left <<<"$1"
    read -r -a right <<<"$2"
    local index
    for index in 0 1 2; do
        local l="${left[index]:-0}" r="${right[index]:-0}"
        if ((10#$l > 10#$r)); then
            return 0
        elif ((10#$l < 10#$r)); then
            return 1
        fi
    done
    return 1
}

plist_value() {
    /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null
}

derive_version() {
    if [[ -n "$VERSION" ]]; then
        return 0
    fi

    VERSION="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$REPO_ROOT/app/Cargo.toml" | head -n 1)"
    [[ -n "$VERSION" ]] || fail "failed to derive version from app/Cargo.toml"
}

derive_arch_label() {
    if [[ -n "$ARCH_LABEL" ]]; then
        return 0
    fi

    case "$RUST_TARGET" in
        aarch64-apple-darwin)
            ARCH_LABEL="arm64"
            ;;
        x86_64-apple-darwin)
            ARCH_LABEL="x86_64"
            ;;
        "")
            case "$(uname -m)" in
                arm64|aarch64)
                    ARCH_LABEL="arm64"
                    ;;
                x86_64)
                    ARCH_LABEL="x86_64"
                    ;;
                *)
                    ARCH_LABEL="$(uname -m)"
                    ;;
            esac
            ;;
        *)
            ARCH_LABEL="$RUST_TARGET"
            ;;
    esac
}

derive_archive_basename() {
    if [[ -n "$ARCHIVE_BASENAME" ]]; then
        return 0
    fi

    ARCHIVE_BASENAME="${APP_NAME}-v${VERSION}-macos-${ARCH_LABEL}"
}

derive_deployment_target() {
    if [[ -z "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
        MACOSX_DEPLOYMENT_TARGET="$(plist_value "$INFO_PLIST_TEMPLATE" LSMinimumSystemVersion)"
        [[ -n "$MACOSX_DEPLOYMENT_TARGET" ]] || fail "no LSMinimumSystemVersion in $INFO_PLIST_TEMPLATE"
    fi
    export MACOSX_DEPLOYMENT_TARGET
    log "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
}

configure_build_env() {
    if [[ -z "${OPENSLIDE_LIB_DIR:-}" ]]; then
        log "Building OpenSlide from the pinned eosin-platform fork commit"
        OPENSLIDE_PREFIX="$OPENSLIDE_PREFIX" \
        OPENSLIDE_BUILD_ROOT="$OPENSLIDE_BUILD_ROOT" \
            bash "$REPO_ROOT/packaging/shared/build-openslide.sh"
        export OPENSLIDE_LIB_DIR="$OPENSLIDE_PREFIX/lib"
    fi

    [[ -d "$OPENSLIDE_LIB_DIR" ]] || fail "OpenSlide library directory not found: $OPENSLIDE_LIB_DIR"

    prepend_env_path PKG_CONFIG_PATH "$OPENSLIDE_LIB_DIR/pkgconfig"
    prepend_env_path LIBRARY_PATH "$OPENSLIDE_LIB_DIR"
    prepend_env_path DYLD_LIBRARY_PATH "$OPENSLIDE_LIB_DIR"
    export RUSTFLAGS="-L native=$OPENSLIDE_LIB_DIR ${RUSTFLAGS:-}"
}

target_dir() {
    if [[ -n "$RUST_TARGET" ]]; then
        printf '%s\n' "$CARGO_TARGET_DIR/$RUST_TARGET/$BUILD_PROFILE"
    else
        printf '%s\n' "$CARGO_TARGET_DIR/$BUILD_PROFILE"
    fi
}

prepare_layout() {
    log "Preparing bundle layout at $APP_BUNDLE_PATH"
    rm -rf "$APP_BUNDLE_PATH"
    rm -f "$BUNDLED_SOURCES"
    mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$FRAMEWORKS_DIR" "$DIST_DIR"
}

build_binary() {
    local -a cargo_cmd=(cargo build --locked --bin "$APP_NAME" --bin eov-update-helper --no-default-features --features "$DISTRIBUTION_FEATURE_TARGET")
    if [[ "$BUILD_PROFILE" == "release" ]]; then
        cargo_cmd+=(--release)
    else
        cargo_cmd+=(--profile "$BUILD_PROFILE")
    fi
    if [[ -n "$RUST_TARGET" ]]; then
        cargo_cmd+=(--target "$RUST_TARGET")
    fi

    configure_build_env
    log "Building $APP_NAME with: ${cargo_cmd[*]}"
    (
        cd "$REPO_ROOT"
        "${cargo_cmd[@]}"
    )

    local output_dir
    output_dir="$(target_dir)"
    require_file "$output_dir/$APP_NAME"
    cp "$output_dir/$APP_NAME" "$MACOS_BINARY_PATH"
    chmod 755 "$MACOS_BINARY_PATH"

    require_file "$output_dir/eov-update-helper"
    cp "$output_dir/eov-update-helper" "$MACOS_HELPER_PATH"
    chmod 755 "$MACOS_HELPER_PATH"
}

generate_bundle_icon() {
    require_file "$ICNS_SOURCE"
    log "Copying macOS bundle icon from $(basename "$ICNS_SOURCE")"
    cp "$ICNS_SOURCE" "$ICON_OUTPUT_PATH"
}

copy_licenses() {
    log "Copying licences and third-party notices into Resources"
    local license
    for license in LICENSE-MIT LICENSE-APACHE-2.0 LICENSE-GPLv3; do
        require_file "$REPO_ROOT/$license"
        cp "$REPO_ROOT/$license" "$RESOURCES_DIR/$license"
    done
    require_file "$THIRD_PARTY_FILE"
    cp "$THIRD_PARTY_FILE" "$RESOURCES_DIR/COPYING.third-party"
    mkdir -p "$RESOURCES_DIR/licenses"
    cp -R "$LICENSES_DIR/." "$RESOURCES_DIR/licenses/"
}

render_info_plist() {
    require_file "$INFO_PLIST_TEMPLATE"

    log "Rendering Info.plist"
    sed \
        -e "s|__PRODUCT_NAME__|$PRODUCT_NAME|g" \
        -e "s|__APP_ID__|$APP_ID|g" \
        -e "s|__VERSION__|$VERSION|g" \
        -e "s|__EXECUTABLE_NAME__|$EXECUTABLE_NAME|g" \
        -e "s|__ICON_FILE__|$BUNDLE_ICON_NAME|g" \
        "$INFO_PLIST_TEMPLATE" > "$CONTENTS_DIR/Info.plist"
    plutil -lint "$CONTENTS_DIR/Info.plist" >/dev/null
}

# Libraries that come with macOS (from the dyld shared cache, not on disk).
is_system_library() {
    case "$1" in
        /usr/lib/*|/System/*)
            return 0
            ;;
    esac
    return 1
}

# The libraries a Mach-O file loads (LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, ...),
# one per line; unlike `otool -L`, without a dylib's own install name.
linked_libraries() {
    otool -l "$1" | awk '
        $1 == "cmd" { cmd = $2; next }
        $1 == "name" && cmd ~ /^LC_(LOAD|LOAD_WEAK|REEXPORT|LOAD_UPWARD|LAZY_LOAD)_DYLIB$/ {
            sub(/^[ \t]*name /, ""); sub(/ \(offset [0-9]+\)$/, ""); print
        }'
}

# The LC_RPATH entries of a Mach-O file, one per line, in search order.
rpaths_of() {
    otool -l "$1" | awk '
        $1 == "cmd" { cmd = $2; next }
        $1 == "path" && cmd == "LC_RPATH" {
            sub(/^[ \t]*path /, ""); sub(/ \(offset [0-9]+\)$/, ""); print
        }'
}

# The canonical form of a path whose directory exists, with symlinks and ".."
# resolved (macOS has no realpath(1) before 13, no readlink -f before 12.3).
canonical_path() {
    local path="$1" target dir hops=0
    [[ -n "$path" ]] || return 1
    while [[ -L "$path" ]]; do
        hops=$((hops + 1))
        [[ "$hops" -le 32 ]] || return 1
        target="$(readlink "$path")"
        case "$target" in
            /*) path="$target" ;;
            *) path="$(dirname "$path")/$target" ;;
        esac
    done
    if [[ -d "$path" ]]; then
        (CDPATH='' cd -P -- "$path" && pwd -P)
        return
    fi
    dir="$(CDPATH='' cd -P -- "$(dirname -- "$path")" 2>/dev/null && pwd -P)" || return 1
    printf '%s/%s\n' "${dir%/}" "$(basename -- "$path")"
}

# Whether the existing file or directory `path` is inside the directory `root`.
is_inside() {
    local root path
    root="$(canonical_path "$1")" || return 1
    path="$(canonical_path "$2" 2>/dev/null)" || return 1
    [[ -e "$path" && "$path" == "$root/"* ]]
}

# A load command or LC_RPATH of the Mach-O file `image` with @loader_path
# (the directory of `image`) and @executable_path (Contents/MacOS) expanded.
expand_load_path() {
    local path="$1" image="$2"
    case "$path" in
        @loader_path|@loader_path/*)
            printf '%s%s\n' "$(dirname "$image")" "${path#@loader_path}"
            ;;
        @executable_path|@executable_path/*)
            printf '%s%s\n' "$MACOS_DIR" "${path#@executable_path}"
            ;;
        *)
            printf '%s\n' "$path"
            ;;
    esac
}

# The canonical path of the file dyld loads for the load command `reference`
# of the Mach-O file at `origin`, @rpath standing for each directory in
# `rpaths` (one per line: the file's own LC_RPATHs, then those of the files
# that load it). Unlike dyld, nothing falls back to DYLD_FALLBACK_LIBRARY_PATH:
# a library found only there (/usr/local/lib is a default, and Homebrew's
# prefix on Intel Macs) is missing on Macs without it.
resolve_reference() {
    local reference="$1" origin="$2" rpaths="$3" dir candidate
    case "$reference" in
        @rpath/*)
            while IFS= read -r dir; do
                candidate="$dir/${reference#@rpath/}"
                if [[ -n "$dir" && -f "$candidate" ]]; then
                    canonical_path "$candidate"
                    return
                fi
            done <<<"$rpaths"
            return 1
            ;;
        @loader_path/*|@executable_path/*|/*)
            candidate="$(expand_load_path "$reference" "$origin")"
            [[ -f "$candidate" ]] || return 1
            canonical_path "$candidate"
            ;;
        *)
            # A bare file name, which dyld looks up in the current directory.
            return 1
            ;;
    esac
}

# The name in Contents/Frameworks of the library copied from the canonical
# path $1, if it is bundled (BUNDLED_SOURCES: name<TAB>source lines).
bundled_name() {
    [[ -f "$BUNDLED_SOURCES" ]] || return 1
    awk -F '\t' -v source="$1" '$2 == source { print $1; found = 1; exit } END { exit !found }' \
        "$BUNDLED_SOURCES"
}

# Where the library bundled as Contents/Frameworks/$1 was copied from.
bundled_source() {
    [[ -f "$BUNDLED_SOURCES" ]] || return 1
    awk -F '\t' -v name="$1" '$1 == name { print $2; found = 1; exit } END { exit !found }' \
        "$BUNDLED_SOURCES"
}

# Deletes the LC_RPATHs of a bundled binary that point outside the bundle and
# macOS. One at a time: Apple's install_name_tool deletes every entry with the
# path, LLVM's only the first.
delete_foreign_rpaths() {
    local binary="$1" rpath foreign before
    while :; do
        foreign=""
        while IFS= read -r rpath; do
            [[ -n "$rpath" ]] || continue
            is_system_library "$rpath" && continue
            is_inside "$APP_BUNDLE_PATH" "$(expand_load_path "$rpath" "$binary")" && continue
            foreign="$rpath"
            break
        done < <(rpaths_of "$binary")
        [[ -n "$foreign" ]] || return 0
        before="$(rpaths_of "$binary" | wc -l)"
        install_name_tool -delete_rpath "$foreign" "$binary"
        [[ "$(rpaths_of "$binary" | wc -l)" -lt "$before" ]] \
            || fail "could not delete the LC_RPATH $foreign of $binary"
    done
}

# Copies every library eov loads, directly or through other libraries, that
# does not come with macOS into Contents/Frameworks, and points the references
# at the copies: @executable_path/../Frameworks/<name> in eov and
# @loader_path/<name> between the libraries. References are resolved the way
# dyld resolves them on the build machine, so the libraries Homebrew's
# libraries reference through @rpath or @loader_path are bundled too. The
# LC_RPATHs that point outside the bundle and macOS are deleted; nothing in the
# bundle uses @rpath afterwards.
bundle_libraries() {
    log "Bundling non-system dynamic libraries"
    local frameworks
    frameworks="$(canonical_path "$FRAMEWORKS_DIR")"
    # The files left to process: the path in the bundle, the path the file
    # was copied from (@loader_path and its LC_RPATHs refer to that one), and
    # the directories in the LC_RPATHs of the files that load it, which dyld
    # searches after its own.
    local -a images=("$MACOS_BINARY_PATH") origins=("$MACOS_BINARY_PATH") inherited=("")
    local -a changes
    local count=1 index=0
    local image origin rpaths rpath references reference source name target searched

    while [[ "$index" -lt "$count" ]]; do
        image="${images[index]}"
        origin="${origins[index]}"
        rpaths=""
        while IFS= read -r rpath; do
            if [[ -n "$rpath" ]]; then
                rpaths="$rpaths$(expand_load_path "$rpath" "$origin")"$'\n'
            fi
        done < <(rpaths_of "$origin")
        rpaths="$rpaths${inherited[index]}"

        changes=()
        # Read before install_name_tool rewrites the file.
        references="$(linked_libraries "$image")"
        while IFS= read -r reference; do
            [[ -n "$reference" ]] || continue
            is_system_library "$reference" && continue
            if ! source="$(resolve_reference "$reference" "$origin" "$rpaths")"; then
                case "$reference" in
                    @rpath/*)
                        searched="$(printf '%s' "$rpaths" | tr '\n' ' ')"
                        searched="${searched% }"
                        fail "$origin links $reference, which is in none of the LC_RPATHs of it and the libraries that load it (${searched:-none})"
                        ;;
                    *)
                        fail "$origin links $reference, which does not exist"
                        ;;
                esac
            fi

            if [[ "$(dirname "$source")" == "$frameworks" ]]; then
                name="$(basename "$source")"
            elif ! name="$(bundled_name "$source")"; then
                name="$(basename "$reference")"
                [[ ! -e "$FRAMEWORKS_DIR/$name" ]] \
                    || fail "two different libraries are called $name: $(bundled_source "$name") and $source"
                log "  bundling $name ($source)"
                cp "$source" "$FRAMEWORKS_DIR/$name"
                chmod 644 "$FRAMEWORKS_DIR/$name"
                printf '%s\t%s\n' "$name" "$source" >> "$BUNDLED_SOURCES"
                images[count]="$FRAMEWORKS_DIR/$name"
                origins[count]="$source"
                inherited[count]="$rpaths"
                count=$((count + 1))
            fi

            if [[ "$image" == "$MACOS_BINARY_PATH" ]]; then
                target="@executable_path/../Frameworks/$name"
            else
                target="@loader_path/$name"
            fi
            if [[ "$reference" != "$target" ]]; then
                changes+=(-change "$reference" "$target")
            fi
        done <<<"$references"

        if [[ "$image" != "$MACOS_BINARY_PATH" ]]; then
            changes+=(-id "@executable_path/../Frameworks/$(basename "$image")")
        fi
        if [[ "${#changes[@]}" -gt 0 ]]; then
            install_name_tool "${changes[@]}" "$image"
        fi
        delete_foreign_rpaths "$image"
        index=$((index + 1))
    done
}

# Licence files of the bundled libraries: Homebrew keeps each formula's
# licence files in the root of its keg (Cellar/<formula>/<version>/), and the
# OpenSlide and libdicom sources built here carry their own.
collect_bundled_licenses() {
    local dest="$RESOURCES_DIR/licenses/third-party" real keg formula file
    mkdir -p "$dest"
    log "Collecting licence files of the bundled libraries"
    if [[ -f "$BUNDLED_SOURCES" ]]; then
        # Canonical paths, e.g. /opt/homebrew/Cellar/webp/1.6.0/lib/libwebp.7.dylib.
        while IFS=$'\t' read -r _ real; do
            case "$real" in
                */Cellar/*/*/lib/*)
                    keg="${real%%/lib/*}"
                    formula="$(basename "$(dirname "$keg")")"
                    ;;
                *)
                    continue
                    ;;
            esac
            [[ -d "$dest/$formula" ]] && continue
            for file in "$keg"/COPYING* "$keg"/LICENSE* "$keg"/LICENCE* "$keg"/COPYRIGHT* "$keg"/NOTICE*; do
                [[ -f "$file" ]] || continue
                mkdir -p "$dest/$formula"
                cp "$file" "$dest/$formula/"
            done
            [[ -d "$dest/$formula" ]] || warn "no licence file found in the Homebrew keg of $formula"
        done < <(sort -u "$BUNDLED_SOURCES")
    fi

    local source_dir name
    for source_dir in "$OPENSLIDE_BUILD_ROOT/openslide-src" "$OPENSLIDE_BUILD_ROOT"/libdicom-src/libdicom-*; do
        [[ -d "$source_dir" ]] || continue
        name="$(basename "$source_dir")"
        name="${name%-src}"
        for file in "$source_dir"/COPYING* "$source_dir"/LICENSE*; do
            [[ -f "$file" ]] || continue
            mkdir -p "$dest/$name"
            cp "$file" "$dest/$name/"
        done
    done
}

bundle_binaries() {
    printf '%s\n' "$MACOS_BINARY_PATH" "$MACOS_HELPER_PATH"
    local fw
    for fw in "$FRAMEWORKS_DIR"/*.dylib; do
        [[ -f "$fw" ]] && printf '%s\n' "$fw"
    done
    return 0
}

# Every library a shipped binary loads must come with macOS or be a file in
# the bundle referenced relative to it (@executable_path, @loader_path), and
# the helper may only use system libraries. Fails on absolute paths such as
# /opt/homebrew/... or the CI build prefix (UPD-4), on @rpath references (the
# library they find depends on LC_RPATHs and, failing those, on what is
# installed on the Mac) and on LC_RPATHs that point outside the bundle.
verify_linkage() {
    log "Verifying dynamic library references (otool -l)"
    local errors=0 binary label reference resolved rpath

    while IFS= read -r binary; do
        label="${binary#"$CONTENTS_DIR"/}"
        while IFS= read -r reference; do
            [[ -n "$reference" ]] || continue
            is_system_library "$reference" && continue
            if [[ "$binary" == "$MACOS_HELPER_PATH" ]]; then
                echo "[macos] error: eov-update-helper links $reference; it must only use system libraries" >&2
                errors=$((errors + 1))
                continue
            fi
            case "$reference" in
                @executable_path/*|@loader_path/*)
                    resolved="$(expand_load_path "$reference" "$binary")"
                    if [[ -f "$resolved" ]] && is_inside "$APP_BUNDLE_PATH" "$resolved"; then
                        continue
                    fi
                    echo "[macos] error: $label links $reference, which is not in the bundle" >&2
                    ;;
                *)
                    echo "[macos] error: $label links $reference, outside the bundle" >&2
                    ;;
            esac
            errors=$((errors + 1))
        done < <(linked_libraries "$binary")

        while IFS= read -r rpath; do
            [[ -n "$rpath" ]] || continue
            is_system_library "$rpath" && continue
            is_inside "$APP_BUNDLE_PATH" "$(expand_load_path "$rpath" "$binary")" && continue
            echo "[macos] error: $label has the LC_RPATH $rpath, outside the bundle" >&2
            errors=$((errors + 1))
        done < <(rpaths_of "$binary")
    done < <(bundle_binaries)

    [[ "$errors" -eq 0 ]] || fail "$errors unresolved or non-portable library reference(s)"
}

macho_minos() {
    otool -l "$1" | awk '
        $1 == "cmd" && $2 == "LC_BUILD_VERSION" { build = 1; next }
        build && $1 == "minos" { print $2; exit }
        $1 == "cmd" && $2 == "LC_VERSION_MIN_MACOSX" { legacy = 1; next }
        legacy && $1 == "version" { print $2; exit }'
}

check_minimum_system_version() {
    local plist="$CONTENTS_DIR/Info.plist"
    local declared required binary minos newest_binary=""
    declared="$(plist_value "$plist" LSMinimumSystemVersion)"
    required="$declared"

    log "Deployment targets of the shipped binaries (minos):"
    while IFS= read -r binary; do
        minos="$(macho_minos "$binary")"
        [[ -n "$minos" ]] || fail "could not read the deployment target of $binary"
        log "  $minos  ${binary#"$CONTENTS_DIR"/}"
        if version_gt "$minos" "$required"; then
            required="$minos"
            newest_binary="${binary#"$CONTENTS_DIR"/}"
        fi
    done < <(bundle_binaries)

    if [[ "$required" == "$declared" ]]; then
        log "LSMinimumSystemVersion $declared covers every binary"
        return 0
    fi
    if [[ "$MACOS_STRICT_MINIMUM" == "1" ]]; then
        fail "$newest_binary needs macOS $required, but the bundle declares $declared"
    fi
    warn "$newest_binary needs macOS $required; raising LSMinimumSystemVersion from $declared to $required. Build the dependencies with MACOSX_DEPLOYMENT_TARGET=$declared to support older releases."
    /usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $required" "$plist"
    # Keep it XML: the self-updater reads it from downloaded bundles.
    plutil -convert xml1 "$plist"
}

codesign_bundle() {
    log "Code signing bundled dylibs and binaries with identity '$CODESIGN_IDENTITY'"
    local -a options=(--force --sign "$CODESIGN_IDENTITY")
    local -a app_options=()
    if [[ "$CODESIGN_IDENTITY" != "-" ]]; then
        # Developer ID: hardened runtime and a secure timestamp, as
        # notarization requires.
        options+=(--options runtime --timestamp)
        require_file "$ENTITLEMENTS_FILE"
        app_options+=(--entitlements "$ENTITLEMENTS_FILE")
    fi

    for fw in "$FRAMEWORKS_DIR"/*.dylib; do
        [[ -f "$fw" ]] || continue
        codesign "${options[@]}" "$fw"
    done

    codesign "${options[@]}" "$MACOS_HELPER_PATH"
    # ${array[@]+...}: an empty array is "unbound" under set -u in bash 3.2.
    codesign "${options[@]}" ${app_options[@]+"${app_options[@]}"} "$MACOS_BINARY_PATH"
    codesign "${options[@]}" ${app_options[@]+"${app_options[@]}"} "$APP_BUNDLE_PATH"
    codesign --verify --strict --verbose=2 "$APP_BUNDLE_PATH"
}

# Runs a command the way it runs on a Mac without Homebrew or this build tree:
# without the build's DYLD_LIBRARY_PATH, and with dyld's fallback limited to
# macOS's own directories. By default dyld also falls back to /usr/local/lib,
# Homebrew's prefix on Intel Macs, which hides a library missing from the
# bundle on an Intel build machine.
run_isolated() {
    env -u DYLD_LIBRARY_PATH -u DYLD_FRAMEWORK_PATH -u DYLD_INSERT_LIBRARIES \
        DYLD_FALLBACK_LIBRARY_PATH=/usr/lib \
        DYLD_FALLBACK_FRAMEWORK_PATH=/System/Library/Frameworks \
        "$@"
}

# Starts eov from a copy of the bundle in a temporary directory (nothing may
# depend on the staging path) and checks that every image dyld loaded
# (DYLD_PRINT_LIBRARIES) is in the copy or part of macOS. Then runs
# eov-update-helper --self-test from a temporary directory, where the updater
# runs it (app/src/update/handoff.rs).
smoke_test_bundle() {
    log "Smoke-testing the bundle"
    local scratch app output image loaded=0 errors=0
    scratch="$(mktemp -d)"
    scratch="$(canonical_path "$scratch")"
    app="$scratch/$APP_BUNDLE_NAME"
    ditto "$APP_BUNDLE_PATH" "$app"

    if ! output="$(run_isolated DYLD_PRINT_LIBRARIES=1 "$app/Contents/MacOS/$EXECUTABLE_NAME" --version \
            2>"$scratch/stderr.log")"; then
        cat "$scratch/stderr.log" >&2
        rm -rf "$scratch"
        fail "$APP_BUNDLE_NAME does not start: eov --version failed"
    fi
    log "  $output"

    # dyld 4 prints "dyld[<pid>]: <path>", dyld 3 "dyld: loaded: <path>".
    while IFS= read -r image; do
        loaded=$((loaded + 1))
        is_system_library "$image" && continue
        is_inside "$app" "$image" && continue
        echo "[macos] error: eov loaded $image, which is neither in the bundle nor part of macOS" >&2
        errors=$((errors + 1))
    done < <(sed -n 's|^dyld[^/]*\(/.*\)$|\1|p' "$scratch/stderr.log")
    if [[ "$errors" -gt 0 ]]; then
        rm -rf "$scratch"
        fail "eov loaded $errors image(s) from outside the bundle and macOS"
    fi
    if [[ "$loaded" -eq 0 ]]; then
        warn "dyld ignored DYLD_PRINT_LIBRARIES (hardened runtime?), so only the start of eov was checked"
    else
        log "  dyld loaded $loaded images, each from the bundle or macOS"
    fi

    cp "$MACOS_HELPER_PATH" "$scratch/eov-update-helper"
    if ! output="$(run_isolated "$scratch/eov-update-helper" --self-test 2>"$scratch/stderr.log")"; then
        cat "$scratch/stderr.log" >&2
        rm -rf "$scratch"
        fail "eov-update-helper --self-test failed when run outside the bundle"
    fi
    rm -rf "$scratch"
    log "  $output (from a temporary directory)"
}

notarize_bundle() {
    [[ "$NOTARIZE" == "1" ]] || return 0
    [[ "$CODESIGN_IDENTITY" != "-" ]] || fail "NOTARIZE=1 needs a Developer ID CODESIGN_IDENTITY"
    : "${APPLE_API_KEY_PATH:?NOTARIZE=1 needs APPLE_API_KEY_PATH}"
    : "${APPLE_API_KEY_ID:?NOTARIZE=1 needs APPLE_API_KEY_ID}"
    : "${APPLE_API_ISSUER:?NOTARIZE=1 needs APPLE_API_ISSUER}"

    local submission="$BUILD_ROOT/notarize-$ARCHIVE_BASENAME.zip"
    log "Submitting $APP_BUNDLE_NAME for notarization"
    rm -f "$submission"
    ditto -c -k --keepParent "$APP_BUNDLE_PATH" "$submission"
    xcrun notarytool submit "$submission" \
        --key "$APPLE_API_KEY_PATH" \
        --key-id "$APPLE_API_KEY_ID" \
        --issuer "$APPLE_API_ISSUER" \
        --wait
    rm -f "$submission"
    xcrun stapler staple "$APP_BUNDLE_PATH"
    xcrun stapler validate "$APP_BUNDLE_PATH"
}

create_zip_archive() {
    local zip_path="$DIST_DIR/${ARCHIVE_BASENAME}.zip"
    local checksum_path="${zip_path}.sha256"

    log "Creating zip archive $(basename "$zip_path")"
    rm -f "$zip_path"
    (
        cd "$STAGING_DIR"
        zip -q -r -y "$zip_path" "$APP_BUNDLE_NAME"
    )

    (
        cd "$DIST_DIR"
        shasum -a 256 "$(basename "$zip_path")" > "$(basename "$checksum_path")"
    )
}

print_summary() {
    log "Bundle contents:"
    ls -la "$CONTENTS_DIR" "$MACOS_DIR" "$FRAMEWORKS_DIR"
    log "LSMinimumSystemVersion: $(plist_value "$CONTENTS_DIR/Info.plist" LSMinimumSystemVersion)"

    if compgen -G "$DIST_DIR/${ARCHIVE_BASENAME}*" >/dev/null; then
        log "Artifacts:"
        ls -la "$DIST_DIR"/"${ARCHIVE_BASENAME}"*
    fi
}

main() {
    [[ "$(uname -s)" == "Darwin" ]] || fail "packaging/macos/build.sh must be run on macOS"

    derive_version
    derive_arch_label
    derive_archive_basename
    derive_deployment_target

    prepare_layout
    build_binary
    generate_bundle_icon
    copy_licenses
    render_info_plist
    bundle_libraries
    collect_bundled_licenses
    verify_linkage
    check_minimum_system_version
    codesign_bundle
    smoke_test_bundle
    notarize_bundle

    if [[ "$CREATE_ZIP" == "1" ]]; then
        create_zip_archive
    fi

    print_summary
}

# Sourced (packaging/macos/test-bundle-libraries.sh), the script only defines
# its functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
