#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"
DEST_DIR="$HOME/.eov/plugins"

rm -rf ~/.cache/eov/plugin-packages

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
#export EOVAE_ORT_OPT_LEVEL=disable
export EOVAE_DIAG_ALLOW_CPU_FALLBACK=1
export EOVAE_ORT_PROFILE=1
export EOVAE_DEBUG_TIMING=1
export EOV_PLUGIN_TRACE=1

package_plugin() {
    local plugin_src="$1"
    local plugin_id="$2"
    local staging_dir="$3"
    local package_path="$4"

    mkdir -p "$staging_dir"
    cp "$plugin_src/plugin.toml" "$staging_dir/"

    if [ -d "$plugin_src/ui" ]; then
        cp -r "$plugin_src/ui" "$staging_dir/"
    fi

    crate_name="$(grep '^name\s*=' "$plugin_src/Cargo.toml" | head -1 | sed 's/^name\s*=\s*"\(.*\)"/\1/')"
    lib_name="lib${crate_name}.so"
    if [ -f "$REPO_ROOT/target/debug/$lib_name" ]; then
        cp "$REPO_ROOT/target/debug/$lib_name" "$staging_dir/"
    else
        echo "  WARNING: $lib_name not found in target/debug/, skipping library copy."
    fi

    mkdir -p "$DEST_DIR"
    rm -f "$package_path"
    tar -cf "$package_path" -C "$staging_dir" .
}

package_eovae() {
    local plugin_src="$1"
    local package_path="$2"
    local lib_path="$REPO_ROOT/target/debug/libeovae.so"

    if [ ! -f "$lib_path" ]; then
        echo "  ERROR: $lib_path not found. Build the eovae plugin first."
        return 1
    fi

    python3 "$plugin_src/scripts/package_eop.py" \
        --plugin-root "$plugin_src" \
        --library "$lib_path" \
        --native-lib-dir "$REPO_ROOT/target/debug" \
        --output "$package_path"
}

echo "Building eov with annotation plugin..."

cargo build -p app

PLUGINS=(annotations eovae gamepad)

for plugin_name in "${PLUGINS[@]}"; do
    plugin_src="$PLUGINS_DIR/$plugin_name"

    cargo build --manifest-path "$plugin_src/Cargo.toml"

    [ -f "$plugin_src/plugin.toml" ] || continue

    # Read the plugin id from plugin.toml
    plugin_id="$(grep '^id\s*=' "$plugin_src/plugin.toml" | head -1 | sed 's/^id\s*=\s*"\(.*\)"/\1/')"
    if [ -z "$plugin_id" ]; then
        echo "WARNING: No id found in $plugin_src/plugin.toml, skipping."
        continue
    fi

    package_path="$DEST_DIR/$plugin_id.eop"
    staging_dir="$HOME/.cache/eov/example-plugin-packaging/$plugin_id"

    echo "Packaging plugin '$plugin_id' to $package_path..."
    if [ "$plugin_id" = "eovae" ]; then
        package_eovae "$plugin_src" "$package_path"
    else
        rm -rf "$staging_dir"
        package_plugin "$plugin_src" "$plugin_id" "$staging_dir" "$package_path"
    fi
done

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" \
    --window-width 1200 \
    --window-height 800 \
    --window-x 400 \
    --window-y 200 \
    "$@"
