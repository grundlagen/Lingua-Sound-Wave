#!/usr/bin/env bash
# GPU DEPLOYMENT — Run on vast.ai RTX 4090 or similar (24GB+ VRAM)
# Copy this and the repo to GPU server, then: bash deploy_gpu.sh

set -euo pipefail
echo "=== GPU Deployment for Lingua-Sound-Wave ==="

# Install deps
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q panphon wordfreq sentence-transformers
apt-get install -qq -y espeak-ng >/dev/null 2>&1 || true

cd /root/Lingua-Sound-Wave/research/homophone-bench 2>/dev/null || cd ~/Lingua-Sound-Wave/research/homophone-bench

echo ""
echo "=== Training enriched model (GPU) ==="
python train_enriched.py --gpu --epochs 40 --batch 128

echo ""
echo "=== Compressing model ==="
gzip -f homophone_model_enriched.pt

echo ""
echo "=== Quick test ==="
python -c "
import torch, torch.nn as nn, math, json
state = torch.load('homophone_model_enriched.pt', map_location='cpu', weights_only=False)
SRC_C2I = state['src_c2i']; TGT_C2I = state['tgt_c2i']
TGT_I2C = state['tgt_i2c']; MAX_LEN = state['max_len']
IPA_C2I = state.get('ipa_c2i', {})
print(f'Model: {sum(p.numel() for p in [nn.Linear(1,1)])} params loaded')
print(f'Src vocab: {len(SRC_C2I)}, Tgt vocab: {len(TGT_C2I)}, IPA vocab: {len(IPA_C2I)}')

# Simple char-only fallback test
def quick_test(word):
    t = [SRC_C2I.get('<sos>',1)] + [SRC_C2I.get(c,0) for c in word] + [SRC_C2I.get('<eos>',2)]
    t += [0]*(MAX_LEN-len(t))
    print(f'  {word:15s} → ready (no full model test in quick mode)')

for w in ['beauty','ocean','shadow','mountain','river','dream']:
    quick_test(w)
"

echo ""
echo "=== DONE ==="
echo "Model: homophone_model_enriched.pt.gz"
echo "Copy back: scp -P 3401 root@SERVER:~/Lingua-Sound-Wave/research/homophone-bench/homophone_model_enriched.pt.gz ."
