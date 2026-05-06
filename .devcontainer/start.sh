#!/usr/bin/env bash
set -euo pipefail

export PORT=5000
export BASE_PATH=/
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/dev

# Ensure postgres is running (for DB projects)
sudo service postgresql start 2>/dev/null || true

# Start API server in background
echo "Starting API server on :5000..."
(cd "$(dirname "$0")/.." && cd artifacts/api-server && PORT=5000 BASE_PATH=/ pnpm dev 2>&1 | sed 's/^/[api] /') &

# Start UI dev server
echo "Starting homophone-explorer on :5173..."
(cd "$(dirname "$0")/.." && cd artifacts/homophone-explorer && PORT=5173 BASE_PATH=/ pnpm dev 2>&1 | sed 's/^/[ui]  /') &

wait
