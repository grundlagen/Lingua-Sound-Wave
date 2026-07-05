#!/bin/bash
# gpu_backup.sh — Run ON the GPU instance. Periodically saves epoch checkpoints
# and uploads to Google Drive via the local machine as relay.
# 
# This script runs alongside training. It:
# 1. Copies the model to timestamped checkpoints every N minutes
# 2. Keeps last 5 checkpoints
# 3. Exposes them for the local auto_save.sh to pull
#
# Deploy: scp this to GPU, then: nohup bash gpu_backup.sh &

MODEL="/root/homophone_transformer.pt"
LOG="/root/training3.log"
CKPT_DIR="/root/checkpoints"
INTERVAL=300  # 5 minutes

mkdir -p "$CKPT_DIR"

echo "GPU backup daemon started (interval=${INTERVAL}s)"
echo "Model: $MODEL"
echo "Checkpoints: $CKPT_DIR"

while true; do
    if [ -f "$MODEL" ]; then
        TS=$(date '+%Y%m%d_%H%M%S')
        cp "$MODEL" "$CKPT_DIR/epoch_${TS}.pt"

        # Keep last 5
        ls -1t "$CKPT_DIR"/epoch_*.pt 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true

        # Log status
        EPOCH_LINE=$(tail -3 "$LOG" 2>/dev/null | grep "epoch" | tail -1 || echo "training...")
        echo "[$(date '+%H:%M:%S')] Saved checkpoint. $EPOCH_LINE"
    fi
    sleep $INTERVAL
done
