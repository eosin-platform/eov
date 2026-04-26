#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"
DEST_DIR="$HOME/.eov/plugins"

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
#export EOVAE_ORT_OPT_LEVEL=disable
export EOVAE_DIAG_ALLOW_CPU_FALLBACK=1
export EOVAE_ORT_PROFILE=1
export EOVAE_DEBUG_TIMING=1
#export EOV_PLUGIN_TRACE=1

package_plugin() {
    local plugin_src="$1"
    local plugin_id="$2"
    local language="$3"
    local staging_dir="$4"
    local package_path="$5"

    mkdir -p "$staging_dir"
    cp "$plugin_src/plugin.toml" "$staging_dir/"

    if [ -d "$plugin_src/ui" ]; then
        cp -r "$plugin_src/ui" "$staging_dir/"
    fi

    if [ "$language" = "python" ]; then
        cp "$plugin_src"/*.py "$staging_dir/"

        helper_src_dir="$REPO_ROOT/plugin_api/python"
        if [ -f "$helper_src_dir/eov_plugin_host.py" ] && [ -f "$helper_src_dir/eov_extension.desc" ]; then
            cp "$helper_src_dir/eov_plugin_host.py" "$staging_dir/"
            cp "$helper_src_dir/eov_extension.desc" "$staging_dir/"
        else
            echo "  WARNING: Python host helper files missing in $helper_src_dir"
        fi

        echo "  Creating Python venv for $plugin_id..."
        python3 -m venv "$staging_dir/.venv"
        "$staging_dir/.venv/bin/pip" install --quiet slint
        if [ -f "$plugin_src/requirements.txt" ]; then
            echo "  Installing requirements for $plugin_id..."
            "$staging_dir/.venv/bin/pip" install --quiet -r "$plugin_src/requirements.txt"
        fi
    else
        crate_name="$(grep '^name\s*=' "$plugin_src/Cargo.toml" | head -1 | sed 's/^name\s*=\s*"\(.*\)"/\1/')"
        lib_name="lib${crate_name}.so"
        if [ -f "$REPO_ROOT/target/debug/$lib_name" ]; then
            cp "$REPO_ROOT/target/debug/$lib_name" "$staging_dir/"
        else
            echo "  WARNING: $lib_name not found in target/debug/, skipping library copy."
        fi
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

PLUGINS=(annotations eovae)

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

    # Detect language (defaults to rust if not specified)
    language="$(grep '^language\s*=' "$plugin_src/plugin.toml" | head -1 | sed 's/^language\s*=\s*"\(.*\)"/\1/' || true)"
    [ -z "$language" ] && language="rust"

    package_path="$DEST_DIR/$plugin_id.eop"
    staging_dir="$HOME/.cache/eov/example-plugin-packaging/$plugin_id"

    echo "Packaging plugin '$plugin_id' ($language) to $package_path..."
    if [ "$plugin_id" = "eovae" ]; then
        package_eovae "$plugin_src" "$package_path"
    else
        rm -rf "$staging_dir"
        package_plugin "$plugin_src" "$plugin_id" "$language" "$staging_dir" "$package_path"
    fi
done

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" \
    --extension-host-port 12345 \
    --window-width 1200 \
    --window-height 800 \
    --window-x 400 \
    --window-y 200 \
    "$@"
