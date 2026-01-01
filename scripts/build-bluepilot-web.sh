#!/usr/bin/env sh
set -eu

if [ ! -d "bluepilot/web/node_modules" ]; then
  echo "bluepilot/web/node_modules not found. Run 'npm install' in bluepilot/web first." >&2
  exit 1
fi

echo "Building Bluepilot web assets..."
(cd bluepilot/web && npm run build)
