#!/usr/bin/env bash
# Builds eov.app and its release zip.
#
# - eov, eov-update-helper and the libraries built here (OpenSlide, libdicom)
#   target MACOSX_DEPLOYMENT_TARGET (default: the template's
#   LSMinimumSystemVersion).
# - Non-system dylibs are bundled into Contents/Frameworks and referenced
#   through @executable_path; the build fails if any binary still references
#   a library outside the bundle and the OS (e.g. /opt/homebrew or a CI path),
#   or if eov-update-helper links anything but system libraries (it runs from
#   a temporary directory during updates).
# - LSMinimumSystemVersion is raised to the newest `minos` of any shipped
#   binary (Homebrew bottles target the build machine's macOS), so macOS
#   refuses to start eov on a release it cannot run on instead of failing in
#   dyld. MACOS_STRICT_MINIMUM=1 makes that an error instead.
# - The bundle is smoke-tested: `eov --version` from the bundle and
#   `eov-update-helper --self-test` from a temporary copy.
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
        log "Building OpenSlide from the pinned Yanstart fork commit"
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

is_system_dependency() {
    case "$1" in
        /usr/lib/*|/System/*|@rpath/*|@executable_path/*|@loader_path/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

linked_libraries() {
    otool -L "$1" | tail -n +2 | awk '{print $1}'
}

bundle_direct_dependencies() {
    log "Bundling non-system dynamic libraries"

    while IFS= read -r dylib; do
        [[ -n "$dylib" ]] || continue
        if is_system_dependency "$dylib"; then
            continue
        fi
        if [[ ! -f "$dylib" ]]; then
            continue
        fi

        local base
        base="$(basename "$dylib")"
        if [[ ! -f "$FRAMEWORKS_DIR/$base" ]]; then
            log "  bundling: $base"
            printf '%s\n' "$dylib" >> "$BUNDLED_SOURCES"
            cp "$dylib" "$FRAMEWORKS_DIR/$base"
            chmod 644 "$FRAMEWORKS_DIR/$base"
        fi

        install_name_tool -change \
            "$dylib" \
            "@executable_path/../Frameworks/$base" \
            "$MACOS_BINARY_PATH"
    done < <(linked_libraries "$MACOS_BINARY_PATH")
}

bundle_transitive_dependencies() {
    local changed=true

    while $changed; do
        changed=false

        for fw in "$FRAMEWORKS_DIR"/*.dylib; do
            [[ -f "$fw" ]] || continue

            while IFS= read -r dep; do
                [[ -n "$dep" ]] || continue
                if is_system_dependency "$dep"; then
                    continue
                fi
                if [[ ! -f "$dep" ]]; then
                    continue
                fi

                local dep_base
                dep_base="$(basename "$dep")"
                if [[ ! -f "$FRAMEWORKS_DIR/$dep_base" ]]; then
                    log "  bundling (transitive): $dep_base"
                    printf '%s\n' "$dep" >> "$BUNDLED_SOURCES"
                    cp "$dep" "$FRAMEWORKS_DIR/$dep_base"
                    chmod 644 "$FRAMEWORKS_DIR/$dep_base"
                    changed=true
                fi

                install_name_tool -change \
                    "$dep" \
                    "@executable_path/../Frameworks/$dep_base" \
                    "$fw"
            done < <(linked_libraries "$fw")
        done
    done
}

rewrite_framework_ids() {
    for fw in "$FRAMEWORKS_DIR"/*.dylib; do
        [[ -f "$fw" ]] || continue
        local base
        base="$(basename "$fw")"
        install_name_tool -id "@executable_path/../Frameworks/$base" "$fw"
    done
}

# Licence files of the bundled libraries: Homebrew keeps each formula's
# licence files in the root of its keg (Cellar/<formula>/<version>/), and the
# OpenSlide and libdicom sources built here carry their own.
collect_bundled_licenses() {
    local dest="$RESOURCES_DIR/licenses/third-party" source real keg formula file
    mkdir -p "$dest"
    log "Collecting licence files of the bundled libraries"
    if [[ -f "$BUNDLED_SOURCES" ]]; then
        while IFS= read -r source; do
            real="$(readlink -f "$source" 2>/dev/null || printf '%s' "$source")"
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

# Every reference must be a system library or a file inside the bundle; the
# helper may only use system libraries. Fails on absolute paths such as
# /opt/homebrew/... or the CI build prefix (UPD-4).
verify_linkage() {
    log "Verifying dynamic library references (otool -L)"
    local errors=0 binary dep resolved

    while IFS= read -r binary; do
        while IFS= read -r dep; do
            [[ -n "$dep" ]] || continue
            case "$dep" in
                /usr/lib/*|/System/Library/*)
                    continue
                    ;;
            esac
            if [[ "$binary" == "$MACOS_HELPER_PATH" ]]; then
                echo "[macos] error: eov-update-helper links $dep; it must only use system libraries" >&2
                errors=$((errors + 1))
                continue
            fi
            case "$dep" in
                @executable_path/*)
                    resolved="$MACOS_DIR/${dep#@executable_path/}"
                    ;;
                @loader_path/*)
                    resolved="$(dirname "$binary")/${dep#@loader_path/}"
                    ;;
                *)
                    echo "[macos] error: $(basename "$binary") references $dep outside the bundle" >&2
                    errors=$((errors + 1))
                    continue
                    ;;
            esac
            if [[ ! -f "$resolved" ]]; then
                echo "[macos] error: $(basename "$binary") references $dep, which is not in the bundle" >&2
                errors=$((errors + 1))
            fi
        done < <(linked_libraries "$binary")

        while IFS= read -r dep; do
            [[ -n "$dep" ]] || continue
            case "$dep" in
                @executable_path/*|@loader_path/*|/usr/lib/*|/System/*) ;;
                *)
                    echo "[macos] error: $(basename "$binary") has the LC_RPATH $dep" >&2
                    errors=$((errors + 1))
                    ;;
            esac
        done < <(otool -l "$binary" | awk '$1 == "cmd" && $2 == "LC_RPATH" { rpath = 1; next }
                                         rpath && $1 == "path" { print $2; rpath = 0 }')
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

smoke_test_bundle() {
    log "Smoke-testing the bundle"
    local output
    output="$(env -u DYLD_LIBRARY_PATH "$MACOS_BINARY_PATH" --version)" \
        || fail "$APP_BUNDLE_NAME does not start: eov --version failed"
    log "  $output"

    # The updater runs the helper from a private temporary directory, away
    # from the bundle (app/src/update/handoff.rs); it must work from there.
    local scratch
    scratch="$(mktemp -d)"
    cp "$MACOS_HELPER_PATH" "$scratch/eov-update-helper"
    if ! output="$(env -u DYLD_LIBRARY_PATH "$scratch/eov-update-helper" --self-test)"; then
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
    ls -la "$CONTENTS_DIR" "$MACOS_DIR"
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
    bundle_direct_dependencies
    bundle_transitive_dependencies
    rewrite_framework_ids
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

main "$@"
