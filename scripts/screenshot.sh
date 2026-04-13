#!/bin/bash
set -ex
cd "$(dirname "$0")/.."
cargo run -- --window-width 1200 --window-height 800 --window-x 400 --window-y 200