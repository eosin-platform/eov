#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -x
exec cargo run -- \
    --window-width 1200 \
    --window-height 800 \
    --window-x 400 \
    --window-y 200 \
    "$@"