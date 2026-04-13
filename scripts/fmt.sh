#!/bin/bash
set -ex
cd "$(dirname "$0")/.."
cargo fmt
slint-lsp format -i app/ui/*.slint