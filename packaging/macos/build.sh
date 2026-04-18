#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

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
BUNDLE_ICON_NAME="${BUNDLE_ICON_NAME:-$PRODUCT_NAME.icns}"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-}"
CREATE_ZIP="${CREATE_ZIP:-1}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

STAGING_DIR="$BUILD_ROOT/staging"
APP_BUNDLE_PATH="$STAGING_DIR/$APP_BUNDLE_NAME"
CONTENTS_DIR="$APP_BUNDLE_PATH/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
MACOS_BINARY_PATH="$MACOS_DIR/$EXECUTABLE_NAME"
ICON_OUTPUT_PATH="$RESOURCES_DIR/$BUNDLE_ICON_NAME"

log() {
    echo "[macos] $*"
}

fail() {
    echo "[macos] error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "required file not found: $path"
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

configure_build_env() {
    if [[ -z "${OPENSLIDE_LIB_DIR:-}" ]] && command -v brew >/dev/null 2>&1; then
        local openslide_prefix
        openslide_prefix="$(brew --prefix openslide 2>/dev/null || true)"
        if [[ -n "$openslide_prefix" ]]; then
            export OPENSLIDE_LIB_DIR="$openslide_prefix/lib"
        fi
    fi
}

binary_source_path() {
    if [[ -n "$RUST_TARGET" ]]; then
        printf '%s\n' "$REPO_ROOT/target/$RUST_TARGET/$BUILD_PROFILE/$APP_NAME"
    else
        printf '%s\n' "$REPO_ROOT/target/$BUILD_PROFILE/$APP_NAME"
    fi
}

prepare_layout() {
    log "Preparing bundle layout at $APP_BUNDLE_PATH"
    rm -rf "$APP_BUNDLE_PATH"
    mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$FRAMEWORKS_DIR" "$DIST_DIR"
}

build_binary() {
    local -a cargo_cmd=(cargo build --bin "$APP_NAME")
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

    local source_binary
    source_binary="$(binary_source_path)"
    require_file "$source_binary"
    cp "$source_binary" "$MACOS_BINARY_PATH"
    chmod 755 "$MACOS_BINARY_PATH"
}

generate_bundle_icon() {
    require_file "$ICNS_SOURCE"
    log "Copying macOS bundle icon from $(basename "$ICNS_SOURCE")"
    cp "$ICNS_SOURCE" "$ICON_OUTPUT_PATH"
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
            cp "$dylib" "$FRAMEWORKS_DIR/$base"
            chmod 644 "$FRAMEWORKS_DIR/$base"
        fi

        install_name_tool -change \
            "$dylib" \
            "@executable_path/../Frameworks/$base" \
            "$MACOS_BINARY_PATH"
    done < <(otool -L "$MACOS_BINARY_PATH" | tail -n +2 | awk '{print $1}')
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
                    cp "$dep" "$FRAMEWORKS_DIR/$dep_base"
                    chmod 644 "$FRAMEWORKS_DIR/$dep_base"
                    changed=true
                fi

                install_name_tool -change \
                    "$dep" \
                    "@executable_path/../Frameworks/$dep_base" \
                    "$fw"
            done < <(otool -L "$fw" | tail -n +2 | awk '{print $1}')
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

codesign_bundle() {
    log "Code signing bundled dylibs and app binary with identity '$CODESIGN_IDENTITY'"

    for fw in "$FRAMEWORKS_DIR"/*.dylib; do
        [[ -f "$fw" ]] || continue
        codesign --force --sign "$CODESIGN_IDENTITY" "$fw"
    done

    codesign --force --sign "$CODESIGN_IDENTITY" "$MACOS_BINARY_PATH"
    codesign --force --sign "$CODESIGN_IDENTITY" "$APP_BUNDLE_PATH"
}

create_zip_archive() {
    local zip_path="$DIST_DIR/${ARCHIVE_BASENAME}.zip"
    local checksum_path="${zip_path}.sha256"

    log "Creating zip archive $(basename "$zip_path")"
    (
        cd "$STAGING_DIR"
        zip -r -y "$zip_path" "$APP_BUNDLE_NAME"
    )

    (
        cd "$DIST_DIR"
        shasum -a 256 "$(basename "$zip_path")" > "$(basename "$checksum_path")"
    )
}

print_summary() {
    log "Bundle contents:"
    ls -la "$CONTENTS_DIR"

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

    prepare_layout
    build_binary
    generate_bundle_icon
    render_info_plist
    bundle_direct_dependencies
    bundle_transitive_dependencies
    rewrite_framework_ids
    codesign_bundle

    if [[ "$CREATE_ZIP" == "1" ]]; then
        create_zip_archive
    fi

    print_summary
}

main "$@"
