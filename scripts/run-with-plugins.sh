#!/usr/bin/env bash
# Builds eov and the in-tree plugins (debug), packages the plugins and starts
# eov with them.
#
# Everything lives in a development home (default: target/dev-eov-home), so
# the plugins, configuration, data and caches of your real installation are
# never touched. Override with EOV_HOME (and EOV_CACHE_DIR / EOV_DATA_DIR).
#
#   CUDA_HOME=/usr/local/cuda-12.9 scripts/run-with-plugins.sh [eov args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

export EOV_HOME="${EOV_HOME:-$TARGET_DIR/dev-eov-home}"
export EOV_DATA_DIR="${EOV_DATA_DIR:-$EOV_HOME/data}"
export EOV_CACHE_DIR="${EOV_CACHE_DIR:-$EOV_HOME/cache}"
DEST_DIR="${EOV_PLUGIN_DIR:-$EOV_HOME/plugins}"
export EOV_PLUGIN_DIR="$DEST_DIR"
echo "Development home: $EOV_HOME"

# Extracted packages are keyed by content, so stale extractions are never
# reused; dropping them only keeps the development cache small.
rm -rf "$EOV_CACHE_DIR/plugin-packages"

# GPU inference for eovae, when a CUDA toolkit is available.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "$CUDA_HOME/lib64" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "No CUDA toolkit at $CUDA_HOME (set CUDA_HOME); eovae falls back to the CPU."
fi

# Each plugin is its own Cargo workspace with its own target directory
# (unless CARGO_TARGET_DIR puts everything in one place).
plugin_target_dir() {
    local manifest="$1"
    cargo metadata --format-version 1 --no-deps --manifest-path "$manifest" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
}

library_name() {
    local crate_name="$1"
    case "$(uname -s)" in
        Darwin) echo "lib${crate_name}.dylib" ;;
        MINGW*|MSYS*|CYGWIN*) echo "${crate_name}.dll" ;;
        *) echo "lib${crate_name}.so" ;;
    esac
}

package_plugin() {
    local plugin_src="$1"
    local staging_dir="$2"
    local package_path="$3"

    mkdir -p "$staging_dir"
    cp "$plugin_src/plugin.toml" "$staging_dir/"

    if [ -d "$plugin_src/ui" ]; then
        cp -r "$plugin_src/ui" "$staging_dir/"
    fi

    crate_name="$(grep '^name\s*=' "$plugin_src/Cargo.toml" | head -1 | sed 's/^name\s*=\s*"\(.*\)"/\1/')"
    lib_name="$(library_name "${crate_name//-/_}")"
    lib_dir="$(plugin_target_dir "$plugin_src/Cargo.toml")/debug"
    if [ -f "$lib_dir/$lib_name" ]; then
        cp "$lib_dir/$lib_name" "$staging_dir/"
    else
        echo "  WARNING: $lib_name not found in $lib_dir, skipping library copy."
    fi

    mkdir -p "$DEST_DIR"
    rm -f "$package_path"
    tar -cf "$package_path" -C "$staging_dir" .
}

package_eovae() {
    local plugin_src="$1"
    local package_path="$2"
    local lib_path
    lib_path="$(plugin_target_dir "$plugin_src/Cargo.toml")/debug/$(library_name eovae)"

    if [ ! -f "$lib_path" ]; then
        echo "  ERROR: $lib_path not found. Build the eovae plugin first."
        return 1
    fi

    mkdir -p "$DEST_DIR"
    # ONNX Runtime: EOVAE_ORT_DIR (a folder of its libraries) when set,
    # otherwise none is bundled and eovae looks for one at run time.
    local ort_args=(--no-ort)
    if [ -n "${EOVAE_ORT_DIR:-}" ]; then
        ort_args=(--ort-dir "$EOVAE_ORT_DIR")
    fi
    python3 "$plugin_src/scripts/package_eop.py" \
        --library "$lib_path" \
        "${ort_args[@]}" \
        --output "$package_path"
}

echo "Building eov..."
cargo build -p eov

PLUGINS=(annotations eovae gamepad)

for plugin_name in "${PLUGINS[@]}"; do
    plugin_src="$PLUGINS_DIR/$plugin_name"
    [ -f "$plugin_src/plugin.toml" ] || continue

    cargo build --manifest-path "$plugin_src/Cargo.toml"

    plugin_id="$(grep '^id\s*=' "$plugin_src/plugin.toml" | head -1 | sed 's/^id\s*=\s*"\(.*\)"/\1/')"
    if [ -z "$plugin_id" ]; then
        echo "WARNING: No id found in $plugin_src/plugin.toml, skipping."
        continue
    fi

    package_path="$DEST_DIR/$plugin_id.eop"
    staging_dir="$EOV_CACHE_DIR/dev-plugin-packaging/$plugin_id"

    echo "Packaging plugin '$plugin_id' to $package_path..."
    if [ "$plugin_id" = "eovae" ]; then
        package_eovae "$plugin_src" "$package_path"
    else
        rm -rf "$staging_dir"
        package_plugin "$plugin_src" "$staging_dir" "$package_path"
    fi
done

echo "Launching eov..."
exec "$TARGET_DIR/debug/eov" \
    --window-width 1200 \
    --window-height 800 \
    --window-x 400 \
    --window-y 200 \
    "$@"
