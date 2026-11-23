#!/usr/bin/env bash
# Tests the library bundling of packaging/macos/build.sh, on any OS: builds
# small Mach-O libraries laid out like Homebrew's (opt/ symlinks into the
# Cellar, and libwebp linking @rpath/libsharpyuv.0.dylib, which only its
# LC_RPATH @loader_path/../lib finds: eosin-platform/eov#6), bundles them with
# bundle_libraries, checks the bundle, and checks that verify_linkage rejects
# broken bundles and bundle_libraries libraries it cannot find.
#
# On macOS it uses the Xcode command line tools. Elsewhere it uses LLVM's
# clang, ld64.lld, llvm-otool and llvm-install-name-tool from LLVM_BIN, by
# default the newest /usr/lib/llvm-*/bin that has them (Debian and Ubuntu:
# apt install clang lld llvm).
set -euo pipefail

TEST_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ARCH="${TEST_ARCH:-arm64}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# Paths with spaces, as in /Users/me/My Projects/eov.
ROOT="$WORK/fixture root"
HOMEBREW="$ROOT/home brew"
PREFIX="$ROOT/openslide prefix"
failures=0

setup_tools() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        CC=clang
        LD=ld
        SYSLIBROOT="$(xcrun --show-sdk-path)"
        return
    fi

    if [[ -z "${LLVM_BIN:-}" ]]; then
        local candidate version newest=0
        for candidate in /usr/lib/llvm-*/bin; do
            version="${candidate#/usr/lib/llvm-}"
            version="${version%%/*}"
            [[ "$version" =~ ^[0-9]+$ && "$version" -gt "$newest" ]] || continue
            [[ -x "$candidate/ld64.lld" && -x "$candidate/llvm-otool" \
                && -x "$candidate/llvm-install-name-tool" ]] || continue
            newest="$version"
            LLVM_BIN="$candidate"
        done
    fi
    local tool
    for tool in ld64.lld llvm-otool llvm-install-name-tool; do
        [[ -x "${LLVM_BIN:-}/$tool" ]] || {
            echo "error: no $tool; install LLVM (apt install clang lld llvm) or set LLVM_BIN" >&2
            exit 1
        }
    done
    if [[ -x "$LLVM_BIN/clang" ]]; then
        CC="$LLVM_BIN/clang"
    else
        CC=clang
    fi
    LD="$LLVM_BIN/ld64.lld"

    # build.sh runs otool and install_name_tool.
    mkdir -p "$WORK/bin"
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$LLVM_BIN/llvm-otool" > "$WORK/bin/otool"
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$LLVM_BIN/llvm-install-name-tool" > "$WORK/bin/install_name_tool"
    chmod 755 "$WORK/bin/otool" "$WORK/bin/install_name_tool"
    PATH="$WORK/bin:$PATH"

    # Just enough of the macOS SDK to link against libSystem.
    SYSLIBROOT="$WORK/sdk"
    mkdir -p "$SYSLIBROOT/usr/lib"
    cat > "$SYSLIBROOT/usr/lib/libSystem.tbd" <<'EOF'
--- !tapi-tbd
tbd-version:     4
targets:         [ arm64-macos, x86_64-macos ]
install-name:    '/usr/lib/libSystem.B.dylib'
current-version: 1351
exports:
  - targets:     [ arm64-macos, x86_64-macos ]
    symbols:     [ dyld_stub_binder ]
...
EOF
}

# link_macho <output> <ld options and input files...>
link_macho() {
    local output="$1"
    shift
    mkdir -p "$(dirname "$output")"
    "$LD" -arch "$TEST_ARCH" -platform_version macos 11.0 11.0 -headerpad_max_install_names \
        -syslibroot "$SYSLIBROOT" -o "$output" "$@" -lSystem
}

# dylib <file> <install name> [<ld options and libraries to link>...]
dylib() {
    local file="$1" install_name="$2"
    shift 2
    link_macho "$file" -dylib -install_name "$install_name" "$WORK/library.o" "$@"
}

# executable <file> [<ld options and libraries to link>...]
executable() {
    local file="$1"
    shift
    link_macho "$file" -execute "$WORK/main.o" "$@"
}

build_fixture() {
    local cellar="$HOMEBREW/Cellar"
    printf 'int fixture(void) { return 0; }\n' > "$WORK/library.c"
    printf 'int main(void) { return 0; }\n' > "$WORK/main.c"
    "$CC" -target "$TEST_ARCH-apple-macos11" -c "$WORK/library.c" -o "$WORK/library.o"
    "$CC" -target "$TEST_ARCH-apple-macos11" -c "$WORK/main.c" -o "$WORK/main.o"

    mkdir -p "$HOMEBREW/opt"
    ln -s ../Cellar/zstd/1.5.7 "$HOMEBREW/opt/zstd"
    ln -s ../Cellar/webp/1.6.0 "$HOMEBREW/opt/webp"
    ln -s ../Cellar/libtiff/4.7.2 "$HOMEBREW/opt/libtiff"

    dylib "$cellar/zstd/1.5.7/lib/libzstd.1.dylib" "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib"

    # libsharpyuv.0.dylib is a symlink to the real file.
    dylib "$cellar/webp/1.6.0/lib/libsharpyuv.0.1.2.dylib" @rpath/libsharpyuv.0.dylib
    ln -s libsharpyuv.0.1.2.dylib "$cellar/webp/1.6.0/lib/libsharpyuv.0.dylib"
    dylib "$cellar/webp/1.6.0/lib/libwebp.7.dylib" "$HOMEBREW/opt/webp/lib/libwebp.7.dylib" \
        -rpath @loader_path/../lib "$cellar/webp/1.6.0/lib/libsharpyuv.0.dylib"

    dylib "$cellar/libtiff/4.7.2/lib/libtiff.6.dylib" "$HOMEBREW/opt/libtiff/lib/libtiff.6.dylib" \
        "$HOMEBREW/opt/webp/lib/libwebp.7.dylib" "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib"

    # libopenslide links libdicom through @loader_path, and zstd through
    # @rpath, which only an LC_RPATH of eov (the library that loads it) finds.
    dylib "$PREFIX/lib/libdicom.1.dylib" @loader_path/libdicom.1.dylib
    dylib "$PREFIX/lib/libopenslide.1.dylib" "$PREFIX/lib/libopenslide.1.dylib" \
        "$HOMEBREW/opt/libtiff/lib/libtiff.6.dylib" "$PREFIX/lib/libdicom.1.dylib" \
        "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib"
    install_name_tool -change "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib" @rpath/libzstd.1.dylib \
        "$PREFIX/lib/libopenslide.1.dylib"

    executable "$ROOT/eov" "$PREFIX/lib/libopenslide.1.dylib" \
        -rpath "$HOMEBREW/opt/zstd/lib" -rpath "$ROOT/no such directory"
    executable "$ROOT/eov-update-helper"

    # Homebrew keeps each formula's licence in the root of its keg.
    local keg
    for keg in zstd/1.5.7 webp/1.6.0 libtiff/4.7.2; do
        printf 'licence of %s\n' "${keg%/*}" > "$cellar/$keg/LICENSE"
    done

    # An @rpath library that no LC_RPATH finds.
    dylib "$ROOT/missing/libmissing.0.dylib" @rpath/libmissing.0.dylib
    dylib "$ROOT/broken/libbroken.1.dylib" "$ROOT/broken/libbroken.1.dylib" \
        -rpath @loader_path "$ROOT/missing/libmissing.0.dylib"
    rm -r "$ROOT/missing"
    executable "$ROOT/eov-missing-library" "$ROOT/broken/libbroken.1.dylib"

    # Two different libraries with the same file name.
    dylib "$ROOT/first/libsame.1.dylib" "$ROOT/first/libsame.1.dylib"
    dylib "$ROOT/second/libsame.1.dylib" "$ROOT/second/libsame.1.dylib"
    dylib "$ROOT/second/libother.1.dylib" "$ROOT/second/libother.1.dylib" \
        "$ROOT/second/libsame.1.dylib"
    executable "$ROOT/eov-same-names" "$ROOT/first/libsame.1.dylib" "$ROOT/second/libother.1.dylib"
}

# expect <description> <actual> <expected>
expect() {
    if [[ "$2" == "$3" ]]; then
        echo "ok: $1"
        return
    fi
    failures=$((failures + 1))
    {
        echo "FAIL: $1"
        echo "  expected:"
        printf '%s\n' "$3" | sed 's/^/    /'
        echo "  actual:"
        printf '%s\n' "$2" | sed 's/^/    /'
    } >&2
}

# expect_failure <description> <part of the error message> <command...>
expect_failure() {
    local description="$1" message="$2" output
    shift 2
    if output="$("$@" 2>&1)"; then
        echo "FAIL: $description: it succeeded" >&2
        failures=$((failures + 1))
    elif [[ "$output" != *"$message"* ]]; then
        failures=$((failures + 1))
        {
            echo "FAIL: $description: the output does not contain \"$message\":"
            printf '%s\n' "$output" | sed 's/^/    /'
        } >&2
    else
        echo "ok: $description"
    fi
}

# expect_rejected <description> <part of the error message> <command that breaks the bundle...>
expect_rejected() {
    local description="$1" message="$2"
    shift 2
    rm -rf "$WORK/intact.app"
    cp -R "$APP_BUNDLE_PATH" "$WORK/intact.app"
    "$@"
    expect_failure "verify_linkage rejects $description" "$message" verify_linkage
    rm -rf "$APP_BUNDLE_PATH"
    mv "$WORK/intact.app" "$APP_BUNDLE_PATH"
}

sorted() {
    printf '%s\n' "$@" | LC_ALL=C sort
}

references_of() {
    linked_libraries "$1" | LC_ALL=C sort
}

install_name_of() {
    otool -l "$1" | awk '
        $1 == "cmd" { cmd = $2; next }
        $1 == "name" && cmd == "LC_ID_DYLIB" {
            sub(/^[ \t]*name /, ""); sub(/ \(offset [0-9]+\)$/, ""); print
        }'
}

bundle() {
    prepare_layout
    cp "$1" "$MACOS_BINARY_PATH"
    cp "$ROOT/eov-update-helper" "$MACOS_HELPER_PATH"
    bundle_libraries
}

reference_outside_bundle() {
    cp "$FRAMEWORKS_DIR/libzstd.1.dylib" "$STAGING_DIR/libzstd.1.dylib"
    install_name_tool -change @loader_path/libzstd.1.dylib @loader_path/../../../libzstd.1.dylib \
        "$FRAMEWORKS_DIR/libtiff.6.dylib"
}

setup_tools
build_fixture

export BUILD_ROOT="$ROOT/macos build" DIST_DIR="$ROOT/dist"
# shellcheck source=packaging/macos/build.sh
source "$TEST_DIR/build.sh"

bundle "$ROOT/eov"
verify_linkage

expect "bundled libraries" "$(LC_ALL=C ls "$FRAMEWORKS_DIR")" \
    "$(sorted libdicom.1.dylib libopenslide.1.dylib libsharpyuv.0.dylib libtiff.6.dylib \
        libwebp.7.dylib libzstd.1.dylib)"
expect "references of eov" "$(references_of "$MACOS_BINARY_PATH")" \
    "$(sorted /usr/lib/libSystem.B.dylib @executable_path/../Frameworks/libopenslide.1.dylib)"
expect "references of libopenslide" "$(references_of "$FRAMEWORKS_DIR/libopenslide.1.dylib")" \
    "$(sorted /usr/lib/libSystem.B.dylib @loader_path/libdicom.1.dylib @loader_path/libtiff.6.dylib \
        @loader_path/libzstd.1.dylib)"
expect "references of libtiff" "$(references_of "$FRAMEWORKS_DIR/libtiff.6.dylib")" \
    "$(sorted /usr/lib/libSystem.B.dylib @loader_path/libwebp.7.dylib @loader_path/libzstd.1.dylib)"
expect "references of libwebp" "$(references_of "$FRAMEWORKS_DIR/libwebp.7.dylib")" \
    "$(sorted /usr/lib/libSystem.B.dylib @loader_path/libsharpyuv.0.dylib)"
expect "references of the helper" "$(references_of "$MACOS_HELPER_PATH")" /usr/lib/libSystem.B.dylib
for library in "$FRAMEWORKS_DIR"/*.dylib; do
    name="$(basename "$library")"
    expect "install name of $name" "$(install_name_of "$library")" "@executable_path/../Frameworks/$name"
done
for binary in "$MACOS_BINARY_PATH" "$FRAMEWORKS_DIR"/*.dylib; do
    expect "LC_RPATHs of $(basename "$binary")" "$(rpaths_of "$binary")" ""
done
expect "source of libsharpyuv" "$(bundled_source libsharpyuv.0.dylib)" \
    "$(canonical_path "$HOMEBREW/Cellar/webp/1.6.0/lib/libsharpyuv.0.1.2.dylib")"
expect "source of libzstd" "$(bundled_source libzstd.1.dylib)" \
    "$(canonical_path "$HOMEBREW/Cellar/zstd/1.5.7/lib/libzstd.1.dylib")"

collect_bundled_licenses
expect "licences of the bundled Homebrew libraries" \
    "$(cd "$RESOURCES_DIR/licenses/third-party" && LC_ALL=C find . -type f | LC_ALL=C sort)" \
    "$(sorted ./libtiff/LICENSE ./webp/LICENSE ./zstd/LICENSE)"

expect_rejected "an @rpath reference" "links @rpath/libsharpyuv.0.dylib, outside the bundle" \
    install_name_tool -change @loader_path/libsharpyuv.0.dylib @rpath/libsharpyuv.0.dylib \
    "$FRAMEWORKS_DIR/libwebp.7.dylib"
expect_rejected "a Homebrew path" "links $HOMEBREW/opt/zstd/lib/libzstd.1.dylib, outside the bundle" \
    install_name_tool -change @loader_path/libzstd.1.dylib "$HOMEBREW/opt/zstd/lib/libzstd.1.dylib" \
    "$FRAMEWORKS_DIR/libtiff.6.dylib"
expect_rejected "a missing library" "links @loader_path/libdicom.1.dylib, which is not in the bundle" \
    rm "$FRAMEWORKS_DIR/libdicom.1.dylib"
expect_rejected "a library next to the bundle" "links @loader_path/../../../libzstd.1.dylib, which is not in the bundle" \
    reference_outside_bundle
expect_rejected "an LC_RPATH outside the bundle" "has the LC_RPATH /opt/homebrew/lib, outside the bundle" \
    install_name_tool -add_rpath /opt/homebrew/lib "$MACOS_BINARY_PATH"
expect_rejected "a helper linking a bundled library" "eov-update-helper links @executable_path/../Frameworks/libzstd.1.dylib" \
    install_name_tool -change /usr/lib/libSystem.B.dylib @executable_path/../Frameworks/libzstd.1.dylib \
    "$MACOS_HELPER_PATH"

expect_failure "bundle_libraries rejects an @rpath library no LC_RPATH finds" \
    "links @rpath/libmissing.0.dylib, which is in none of the LC_RPATHs" \
    bundle "$ROOT/eov-missing-library"
expect_failure "bundle_libraries rejects two libraries with the same name" \
    "two different libraries are called libsame.1.dylib" \
    bundle "$ROOT/eov-same-names"

if [[ "$failures" -gt 0 ]]; then
    echo "$failures check(s) failed" >&2
    exit 1
fi
echo "All checks passed"
