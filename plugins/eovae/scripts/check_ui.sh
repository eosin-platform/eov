#!/usr/bin/env sh
# Compiles the sidebar with the Slint version eov uses and checks that every
# property the plugin pushes exists with a compatible type (the plugin's own
# tests already check that every declared callback is handled).
set -eu
cd "$(dirname "$0")/.."
dump="$(mktemp -d)/sidebar.json"
EOVAE_DUMP_SIDEBAR="$dump" cargo test --quiet --no-default-features the_sidebar_matches_its_slint_contract
cargo run --quiet --release --manifest-path tools/check-ui/Cargo.toml -- \
    ui/eovae-sidebar.slint EovaeSidebar . "$dump"
