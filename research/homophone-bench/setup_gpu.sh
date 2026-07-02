#!/usr/bin/env bash
# One-shot setup for a fresh (terminal-Claude) box. Safe to re-run.
# Usage: cd research/homophone-bench && bash setup_gpu.sh
set -uo pipefail
cd "$(dirname "$0")"

echo "== system deps =="
command -v espeak-ng >/dev/null || (apt-get update -qq && apt-get install -y -qq espeak-ng)
command -v ffmpeg   >/dev/null || apt-get install -y -qq ffmpeg

echo "== python deps (core, CPU-safe) =="
pip install -q numpy panphon wordfreq sentence_transformers 2>&1 | tail -1

echo "== GPU check =="
python3 - <<'PY'
import torch
gpu = torch.cuda.is_available()
print(f"torch {torch.__version__}  CUDA: {gpu}")
if gpu:
    print("GPU path OPEN: also run -> pip install transformers trl datasets accelerate")
    print("then: python selflearn/train_selflearn.py --data ../train-dual-v1.jsonl")
else:
    print("CPU only: full symbolic pipeline + Haiku loops still run.")
PY

echo "== data fetches (idempotent) =="
[ -f /tmp/muse-en-fr.txt ] || curl -sL --max-time 90 -o /tmp/muse-en-fr.txt \
    "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt"
wc -l /tmp/muse-en-fr.txt

echo "== regenerate heavy artifacts if missing =="
[ -f trigram-lm-fr.pkl ] || echo "  trigram LM absent -> fetch OPUS slice + 'python trigram_lm.py build fr <file>' (see FINAL_VERSE.md)"
[ -f train-dual-v1.jsonl ] || python build_train_corpus.py | tail -2

echo "== smoke test (judge of record + a verified line) =="
python3 - <<'PY'
import matcher
def combo(en, fr):
    qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
    return 0.5*matcher._ngram_channel(qi,ci)+0.5*matcher._feat_channel(qi,ci)
s = combo("less debt, less mess", "laisse dette, laisse messe")
print(f"combo('less debt, less mess' ~ 'laisse dette, laisse messe') = {s:.2f}")
assert s > 0.9, "judge broken!"
print("JUDGE OK. Read CLAUDE.md (repo root) then METHODS_STATUS.md. Go.")
PY
