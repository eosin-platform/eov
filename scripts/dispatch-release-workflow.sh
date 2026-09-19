#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$1
REPO=eosin-platform/eov
echo "Dispatching release workflow for $REPO version $VERSION"
gh workflow run release.yml \
  --repo $REPO \
  --ref master -f version="$VERSION"