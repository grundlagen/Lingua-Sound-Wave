#!/bin/bash
# auto_save.sh — Periodic GPU checkpoint pull + GitHub push.
# Run: chmod +x auto_save.sh && ./auto_save.sh
# Keeps pulling the transformer model from vast.ai every 5 min.

GPU_HOST="137.175.76.24"
GPU_PORT="41480"
GPU_KEY="$HOME/.ssh/id_ed25519_vast"
LOCAL_DIR="$HOME/Lingua-Sound-Wave/research/homophone-bench"
REMOTE_MODEL="/root/homophone_transformer.pt"
REMOTE_LOG="/root/training3.log"

echo "Auto-save daemon started. Pulling checkpoints every 5 min."
echo "GPU: $GPU_HOST:$GPU_PORT"
echo "Local: $LOCAL_DIR"
echo ""

while true; do
    # Pull latest checkpoint
    echo "$(date '+%H:%M:%S') Pulling checkpoint..."
    scp -i "$GPU_KEY" -o StrictHostKeyChecking=no -P "$GPU_PORT" \
        "root@$GPU_HOST:$REMOTE_MODEL" "$LOCAL_DIR/homophone_transformer.pt" 2>/dev/null

    # Pull training log
    scp -i "$GPU_KEY" -o StrictHostKeyChecking=no -P "$GPU_PORT" \
        "root@$GPU_HOST:$REMOTE_LOG" "$LOCAL_DIR/training3.log" 2>/dev/null

    # Show latest epoch
    echo "  Latest epoch: $(tail -1 $LOCAL_DIR/training3.log 2>/dev/null | grep 'epoch' || echo 'still training...')"
    echo "  Model size: $(ls -lh $LOCAL_DIR/homophone_transformer.pt 2>/dev/null | awk '{print $5}')"
    echo ""

    # GitHub push every 6th cycle (30 min)
    if [ $(( $(date +%M) / 5 % 6 )) -eq 0 ]; then
        cd "$HOME/Lingua-Sound-Wave" && \
        git add research/homophone-bench/homophone_transformer.pt.gz 2>/dev/null && \
        git commit -m "auto-save: checkpoint $(date '+%Y-%m-%d %H:%M')" 2>/dev/null && \
        git push 2>/dev/null && \
        echo "  GitHub pushed." || true
    fi

    sleep 300
done
