#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_SRC="$REPO_ROOT/example_plugin"
PLUGIN_DEST="$HOME/.eov/plugins/example_plugin"

echo "Building eov and example plugin..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml"

echo "Installing example plugin to $PLUGIN_DEST..."
rm -rf "$PLUGIN_DEST"
mkdir -p "$PLUGIN_DEST"
cp "$PLUGIN_SRC/plugin.toml" "$PLUGIN_DEST/"
cp -r "$PLUGIN_SRC/ui" "$PLUGIN_DEST/"
cp "$REPO_ROOT/target/debug/libexample_plugin.so" "$PLUGIN_DEST/"

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" "$@"
