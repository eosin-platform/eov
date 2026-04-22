#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/example_plugins"
DEST_DIR="$HOME/.eov/plugins"

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

echo "Building eov and all example plugins..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml"

for plugin_src in "$PLUGINS_DIR"/*/; do
    [ -f "$plugin_src/plugin.toml" ] || continue

    # Read the plugin id from plugin.toml
    plugin_id="$(grep '^id\s*=' "$plugin_src/plugin.toml" | head -1 | sed 's/^id\s*=\s*"\(.*\)"/\1/')"
    if [ -z "$plugin_id" ]; then
        echo "WARNING: No id found in $plugin_src/plugin.toml, skipping."
        continue
    fi

    plugin_dest="$HOME/.eov/plugins/$plugin_id"

    # Detect language (defaults to rust if not specified)
    language="$(grep '^language\s*=' "$plugin_src/plugin.toml" | head -1 | sed 's/^language\s*=\s*"\(.*\)"/\1/' || true)"
    [ -z "$language" ] && language="rust"

    package_path="$DEST_DIR/$plugin_id.eop"
    plugin_dest="$HOME/.cache/eov/example-plugin-packaging/$plugin_id"

    echo "Packaging plugin '$plugin_id' ($language) to $package_path..."
    rm -rf "$plugin_dest"
    package_plugin "$plugin_src" "$plugin_id" "$language" "$plugin_dest" "$package_path"
done

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" --extension-host-port 12345 "$@"
