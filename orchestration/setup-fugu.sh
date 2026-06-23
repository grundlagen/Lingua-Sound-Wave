#!/usr/bin/env bash
# Bootstrap a Python venv for OpenFugu, install deps, and run its self-test.
# Run ./pull-weights.sh first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUGU_DIR="${HERE}/.fugu"
REPO_DIR="${FUGU_DIR}/OpenFugu"
VENV_DIR="${FUGU_DIR}/venv"

if [ ! -d "${REPO_DIR}" ]; then
  echo "OpenFugu not cloned yet. Run ./pull-weights.sh first." >&2
  exit 1
fi

PY="$(command -v python3 || command -v python)"
echo "==> Creating venv at ${VENV_DIR} (using ${PY})"
"${PY}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel

echo "==> Installing OpenFugu requirements"
if [ -f "${REPO_DIR}/requirements.txt" ]; then
  pip install -r "${REPO_DIR}/requirements.txt"
elif [ -f "${REPO_DIR}/pyproject.toml" ]; then
  pip install -e "${REPO_DIR}"
else
  echo "    No requirements.txt/pyproject.toml found — installing common deps."
  pip install litellm transformers torch pyyaml fastapi uvicorn
fi

echo "==> Running self-test"
cd "${REPO_DIR}"
if [ -f openfugu/mini.py ]; then
  python openfugu/mini.py --self-test || echo "    (self-test reported issues — check worker config)"
else
  echo "    openfugu/mini.py not found; check the repo layout: ${REPO_DIR}"
fi

echo
echo "Setup complete."
echo "  1. cp ${HERE}/workers.example.yaml ${FUGU_DIR}/workers.yaml  &&  edit it"
echo "  2. source ${VENV_DIR}/bin/activate"
echo "  3. cd ${REPO_DIR} && python openfugu/serve.py"
