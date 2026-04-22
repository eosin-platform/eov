#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/example_plugins"

echo "Building eov and all example plugins..."

PLUGINS=(annotations) # only build these plugins

for plugin_name in $PLUGINS; do    
    plugin_src="$PLUGINS_DIR/$plugin_name"

    cargo build --manifest-path "$plugin_src/Cargo.toml"

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

    echo "Installing plugin '$plugin_id' ($language) to $plugin_dest..."
    rm -rf "$plugin_dest"
    mkdir -p "$plugin_dest"
    cp "$plugin_src/plugin.toml" "$plugin_dest/"

    # Copy UI directory if present
    if [ -d "$plugin_src/ui" ]; then
        cp -r "$plugin_src/ui" "$plugin_dest/"
    fi

    if [ "$language" = "python" ]; then
        # Copy Python script(s)
        cp "$plugin_src"/*.py "$plugin_dest/"

        # Copy the shared Python host helper and current protobuf descriptor.
        helper_src_dir="$REPO_ROOT/plugin_api/python"
        if [ -f "$helper_src_dir/eov_plugin_host.py" ] && [ -f "$helper_src_dir/eov_extension.desc" ]; then
            cp "$helper_src_dir/eov_plugin_host.py" "$plugin_dest/"
            cp "$helper_src_dir/eov_extension.desc" "$plugin_dest/"
        else
            echo "  WARNING: Python host helper files missing in $helper_src_dir"
        fi

        # Create a venv inside the plugin directory and install deps.
        if [ ! -d "$plugin_dest/.venv" ]; then
            echo "  Creating Python venv for $plugin_id..."
            python3 -m venv "$plugin_dest/.venv"
            "$plugin_dest/.venv/bin/pip" install --quiet slint
            # Install plugin-specific requirements if present.
            if [ -f "$plugin_src/requirements.txt" ]; then
                echo "  Installing requirements for $plugin_id..."
                "$plugin_dest/.venv/bin/pip" install --quiet -r "$plugin_src/requirements.txt"
            fi
        fi
    else
        # Rust plugin: read crate name from Cargo.toml to find the .so
        crate_name="$(grep '^name\s*=' "$plugin_src/Cargo.toml" | head -1 | sed 's/^name\s*=\s*"\(.*\)"/\1/')"
        lib_name="lib${crate_name}.so"
        if [ -f "$REPO_ROOT/target/debug/$lib_name" ]; then
            cp "$REPO_ROOT/target/debug/$lib_name" "$plugin_dest/"
        else
            echo "  WARNING: $lib_name not found in target/debug/, skipping library copy."
        fi
    fi
done

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" --extension-host-port 12345 "$@"
