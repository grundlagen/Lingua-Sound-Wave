#!/usr/bin/env bash
# One-command continual self-learning on a fresh Ubuntu GPU box (Hetzner GPU
# dedicated / RunPod / Vast / Lambda). Idempotent: re-run to update + resume.
# Config via env (all optional except a GPU): GITHUB_REPO BRANCH GITHUB_TOKEN
#   OPENROUTER_API_KEY BASE CKPT
set -euo pipefail
REPO="${GITHUB_REPO:-grundlagen/Lingua-Sound-Wave}"
BRANCH="${BRANCH:-claude/phrase-weave-multiword}"
CKPT="${CKPT:-$HOME/homophonic-carver}"
BASE="${BASE:-Qwen/Qwen2.5-1.5B-Instruct}"

sudo apt-get update -qq
sudo apt-get install -y -qq git espeak-ng python3-pip
AUTH=""; [ -n "${GITHUB_TOKEN:-}" ] && AUTH="${GITHUB_TOKEN}@"
if [ -d lingua/.git ]; then git -C lingua pull --ff-only;
else git clone --depth 1 -b "$BRANCH" "https://${AUTH}github.com/${REPO}.git" lingua; fi
cd lingua/research/homophone-bench
pip3 install -q --upgrade transformers trl datasets accelerate panphon wordfreq numpy
cd selflearn
EVAL=""; [ -n "${OPENROUTER_API_KEY:-}" ] && EVAL="--eval_llm"
echo "[deploy] launching continual trainer (ckpt: $CKPT)"
exec python3 run_continual.py --base "$BASE" --k 8 --keep_thresh 0.55 \
     --ckpt_dir "$CKPT" $EVAL
