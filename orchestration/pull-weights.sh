#!/usr/bin/env bash
# Hunt down and fetch the real, downloadable Fugu-style artifacts.
#
# Official Sakana Fugu has NO public weights (API-only). The genuinely
# downloadable pieces are the community OpenFugu reimplementation + its
# Conductor weights. This script fetches those into ./.fugu/ (git-ignored).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUGU_DIR="${HERE}/.fugu"
REPO_DIR="${FUGU_DIR}/OpenFugu"
WEIGHTS_DIR="${FUGU_DIR}/weights/openfugu-conductor-3b"

OPENFUGU_REPO="https://github.com/trotsky1997/OpenFugu"
CONDUCTOR_REPO="di-zhang-fdu/openfugu-conductor-3b"   # ~6 GB safetensors BF16

mkdir -p "${FUGU_DIR}"

echo "==> 1/3  Cloning OpenFugu (open reimplementation of Sakana Fugu)"
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" pull --ff-only || echo "    (pull skipped — using existing clone)"
else
  git clone --depth 1 "${OPENFUGU_REPO}" "${REPO_DIR}"
fi

echo
echo "==> 2/3  TRINITY router"
echo "    Nothing to download: TRINITY is ~19.5K params, trained locally and"
echo "    gradient-free. Train it after setup with:"
echo "        python train/train_trinity.py   # (from inside OpenFugu)"

echo
echo "==> 3/3  Conductor weights: ${CONDUCTOR_REPO}  (~6 GB, BF16 safetensors)"
echo "    Gated by the Llama 3.2 Community License — you may need 'huggingface-cli login'."
read -r -p "    Download ~6 GB of Conductor weights now? [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  mkdir -p "${WEIGHTS_DIR}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${CONDUCTOR_REPO}" \
      --local-dir "${WEIGHTS_DIR}" --local-dir-use-symlinks False
  elif command -v hf >/dev/null 2>&1; then
    hf download "${CONDUCTOR_REPO}" --local-dir "${WEIGHTS_DIR}"
  else
    echo "    huggingface-cli not found. Install with: pip install -U 'huggingface_hub[cli]'"
    echo "    Then re-run, or clone via git-lfs:"
    echo "        git lfs install && git clone https://huggingface.co/${CONDUCTOR_REPO} '${WEIGHTS_DIR}'"
    exit 1
  fi
  echo "    ✓ Conductor weights at: ${WEIGHTS_DIR}"
else
  echo "    Skipped. Low on RAM / on Termux? Options:"
  echo "      - Use TRINITY router + remote API workers (negligible local footprint), or"
  echo "      - Quantize to GGUF for llama.cpp:"
  echo "          python -m llama_cpp.convert '${WEIGHTS_DIR}' --outtype q4_k_m"
fi

echo
echo "Done. Next: ./setup-fugu.sh"
