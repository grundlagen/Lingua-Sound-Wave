#!/usr/bin/env bash
# fugu-swarm setup: vendor OpenFugu (the engine), fetch its artifacts, install deps.
# OpenFugu is depended on, not copied — it stays under vendor/ (git-ignored).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${HERE}/vendor"
OPENFUGU="${VENDOR}/OpenFugu"

mkdir -p "${VENDOR}"

echo "==> Vendoring OpenFugu (Apache-2.0, trotsky1997/OpenFugu)"
if [ -d "${OPENFUGU}/.git" ]; then
  git -C "${OPENFUGU}" pull --ff-only || true
else
  git clone --depth 1 https://github.com/trotsky1997/OpenFugu "${OPENFUGU}"
fi

echo "==> Installing Python deps (OpenFugu + fugu-swarm)"
python3 -m pip install -U pip wheel
python3 -m pip install -r "${OPENFUGU}/requirements.txt"
python3 -m pip install pyyaml pytest

echo "==> Fetching router artifacts (Qwen3-0.6B + model_iter_60.npy + fixture)"
echo "    These come from their licensed sources; nothing is redistributed here."
python3 "${OPENFUGU}/scripts/fetch_artifacts.py" || \
  echo "    (artifact fetch skipped/failed — re-run manually; see OpenFugu NOTICE)"

echo
echo "Setup complete. Next:"
echo "  export ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  DEEPSEEK_API_KEY=...  # etc."
echo "  python -m fugu_swarm.run            # preview the serve command"
echo "  python -m fugu_swarm.run --serve    # launch the OpenAI-compatible endpoint"
