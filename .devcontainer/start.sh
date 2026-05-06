#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Don't start if already running
lsof -ti:5173 >/dev/null 2>&1 && echo "UI already running on :5173" && exit 0

sudo service postgresql start 2>/dev/null || true

if command -v tmux &>/dev/null; then
  tmux new-session -d -s dev -x 220 -y 50 2>/dev/null || true
  tmux send-keys -t dev "cd $ROOT/artifacts/api-server && PORT=8080 BASE_PATH=/ pnpm dev" Enter
  tmux split-window -h -t dev
  tmux send-keys -t dev "cd $ROOT/artifacts/homophone-explorer && PORT=5173 BASE_PATH=/ API_PROXY_TARGET=http://localhost:8080 pnpm dev" Enter
  tmux attach -t dev
else
  (cd "$ROOT/artifacts/api-server" && PORT=8080 BASE_PATH=/ pnpm dev 2>&1 | sed 's/^/[api] /') &
  (cd "$ROOT/artifacts/homophone-explorer"        && PORT=5173 BASE_PATH=/ API_PROXY_TARGET=http://localhost:8080 pnpm dev 2>&1 | sed 's/^/[ui]  /') &
  wait
fi
