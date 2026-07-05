#!/usr/bin/env bash
# GPU DEPLOYMENT — Transformer model training on RTX 4090 (24GB VRAM)
# Copy repo to GPU server, then: bash deploy_gpu.sh
#
# Model: 6-layer transformer, 512-dim, 8 heads, ~35M params
# Training: 150 epochs, ~2-4 hours on RTX 4090
# Data: graph_aware_training.jsonl (built from all mathematical frameworks)

set -euo pipefail
echo "=== GPU Deployment — Transformer Homophone Model ==="

# Install deps
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q panphon wordfreq
apt-get install -qq -y espeak-ng >/dev/null 2>&1 || true

cd /root/Lingua-Sound-Wave/research/homophone-bench 2>/dev/null || cd ~/Lingua-Sound-Wave/research/homophone-bench

# Step 1: Build graph-aware training dataset
echo ""
echo "=== Step 1: Building graph-aware training dataset ==="
python build_graph_training_data.py

# Step 2: Train transformer
echo ""
echo "=== Step 2: Training transformer (this will take 2-4 hours) ==="
python train_transformer.py --epochs 150 --batch 64 --d_model 512 --nhead 8 --num_layers 6

# Step 3: Compress
echo ""
echo "=== Step 3: Compressing model ==="
gzip -f homophone_transformer.pt

# Step 4: Quick test
echo ""
echo "=== Step 4: Testing ==="
python -c "
import torch, torch.nn as nn, math, json
state = torch.load('homophone_transformer.pt', map_location='cpu', weights_only=False)
print(f'Model: {state.get(\"num_layers\",\"?\")} layers, d={state.get(\"d_model\",\"?\")}')
print(f'Src vocab: {len(state[\"src_c2i\"])}, Tgt: {len(state[\"tgt_c2i\"])}, IPA: {len(state[\"ipa_c2i\"])}')

# Quick size estimate
total = sum(v.numel() for v in state['model'].values())
print(f'Total params: {total:,}')
"

echo ""
echo "=== DONE ==="
echo "Model: homophone_transformer.pt.gz"
echo "Size: $(du -h homophone_transformer.pt.gz | cut -f1)"
echo "Copy back: scp -P 3401 root@SERVER:~/Lingua-Sound-Wave/research/homophone-bench/homophone_transformer.pt.gz ."
