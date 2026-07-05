#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# GPU TRAINING — Run on vast.ai RTX 4090 or similar (24GB VRAM)
#
# How to use:
#   1. SSH to your GPU:  ssh -p 3401 root@117.50.76.44
#   2. Clone if needed:  git clone https://github.com/grundlagen/Lingua-Sound-Wave
#   3. Run this script:  cd Lingua-Sound-Wave/research/homophone-bench && bash deploy_train.sh
#
# Model: 6-layer transformer, 512-dim, 8 heads, ~35M params
# Data:   122,829 training rows from 9 data sources
# Time:   ~3-5 hours on RTX 4090
# Output: homophone_transformer.pt (~150MB)
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
echo "╔══════════════════════════════════════════╗"
echo "║  GPU TRAINING — Transformer Homophone   ║"
echo "╚══════════════════════════════════════════╝"

# ── 0. Check GPU ──────────────────────────────────────────────────────
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: No GPU detected!"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── 1. Install deps ───────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
pip install -q torch torchvision torchaudio 2>/dev/null || pip install -q torch
pip install -q panphon wordfreq 2>/dev/null || true
apt-get install -qq -y espeak-ng >/dev/null 2>&1 || true

# ── 2. Pull latest code ───────────────────────────────────────────────
echo ""
echo "=== Pulling latest code ==="
git pull origin main 2>/dev/null || echo "  (not a git repo or no changes)"

# ── 3. Verify data ────────────────────────────────────────────────────
echo ""
echo "=== Verifying training data ==="
python3 -c "
import json, os
for f in ['graph_aware_training_final.jsonl']:
    if os.path.exists(f):
        count = sum(1 for _ in open(f))
        print(f'  {f}: {count} lines')
    else:
        print(f'  MISSING: {f} — run python3 build_graph_training_data.py first')
"

# ── 4. Quick test (optional, uncomment to verify) ─────────────────────
# echo ""
# echo "=== Quick test (--quick mode, 2 min) ==="
# python3 train_transformer.py --quick --data graph_aware_training_final.jsonl

# ── 5. Full training ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  FULL TRAINING (3-5 hours)               ║"
echo "║  150 epochs, batch=64, 35M params        ║"
echo "║  122,829 training rows                   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

python3 train_transformer.py \
    --data graph_aware_training_final.jsonl \
    --epochs 150 \
    --batch 64 \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --output homophone_transformer.pt

# ── 6. Compress ───────────────────────────────────────────────────────
echo ""
echo "=== Compressing ==="
gzip -f homophone_transformer.pt
SIZE=$(du -h homophone_transformer.pt.gz | cut -f1)
echo "  Model: homophone_transformer.pt.gz ($SIZE)"

# ── 7. Done ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  DONE                                    ║"
echo "║  Model: homophone_transformer.pt.gz      ║"
echo "║  Size:  $SIZE                            ║"
echo "║                                          ║"
echo "║  Copy back:                              ║"
echo "║  scp -P 3401 root@SERVER:research/       ║"
echo "║    homophone-bench/homophone_            ║"
echo "║    transformer.pt.gz .                   ║"
echo "╚══════════════════════════════════════════╝"
