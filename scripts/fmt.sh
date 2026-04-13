#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -x
cargo fmt
slint-lsp format -i app/ui/*.slint