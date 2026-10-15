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
#export EOVAE_ORT_OPT_LEVEL=disable
export EOVAE_DIAG_ALLOW_CPU_FALLBACK=1
export EOVAE_ORT_PROFILE=1
export EOVAE_DEBUG_TIMING=1
export EOV_PLUGIN_TRACE=1

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
    lib_name="lib${crate_name}.so"
    if [ -f "$TARGET_DIR/debug/$lib_name" ]; then
        cp "$TARGET_DIR/debug/$lib_name" "$staging_dir/"
    else
        echo "  WARNING: $lib_name not found in $TARGET_DIR/debug/, skipping library copy."
    fi

    mkdir -p "$DEST_DIR"
    rm -f "$package_path"
    tar -cf "$package_path" -C "$staging_dir" .
}

package_eovae() {
    local plugin_src="$1"
    local package_path="$2"
    local lib_path="$TARGET_DIR/debug/libeovae.so"

    if [ ! -f "$lib_path" ]; then
        echo "  ERROR: $lib_path not found. Build the eovae plugin first."
        return 1
    fi

    mkdir -p "$DEST_DIR"
    python3 "$plugin_src/scripts/package_eop.py" \
        --plugin-root "$plugin_src" \
        --library "$lib_path" \
        --native-lib-dir "$TARGET_DIR/debug" \
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
