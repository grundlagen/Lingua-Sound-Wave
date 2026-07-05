#!/bin/bash
# auto_save.sh — Periodic GPU checkpoint pull + local save + Google Drive backup.
# Run: chmod +x auto_save.sh && nohup ./auto_save.sh &
# Keeps pulling the transformer model from vast.ai every 5 min,
# keeps last 10 local epoch checkpoints, syncs to Google Drive.

set -euo pipefail

GPU_HOST="${GPU_HOST:-137.175.76.24}"
GPU_PORT="${GPU_PORT:-41480}"
GPU_KEY="${GPU_KEY:-$HOME/.ssh/id_ed25519_vast}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/Lingua-Sound-Wave/research/homophone-bench}"
REMOTE_MODEL="${REMOTE_MODEL:-/root/homophone_transformer.pt}"
REMOTE_LOG="${REMOTE_LOG:-/root/training3.log}"
REMOTE_DIR="/root"
GDRIVE_DIR="gdrive:homophone-models"
CHECKPOINT_DIR="$LOCAL_DIR/checkpoints"

mkdir -p "$CHECKPOINT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  AUTO-SAVE DAEMON — GPU checkpoint → local → GDrive    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  GPU:    $GPU_HOST:$GPU_PORT"
echo "  Local:  $LOCAL_DIR"
echo "  GDrive: $GDRIVE_DIR"
echo "  Pull interval: 5 min  |  GDrive sync: every 2 pulls"
echo ""

PULL_COUNT=0

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date '+%H:%M:%S')] Pull #$((PULL_COUNT + 1))"

    # ── Pull model checkpoint ──
    if scp -i "$GPU_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        -P "$GPU_PORT" "root@$GPU_HOST:$REMOTE_MODEL" \
        "$LOCAL_DIR/homophone_transformer.pt" 2>/dev/null; then

        MODEL_SIZE=$(ls -lh "$LOCAL_DIR/homophone_transformer.pt" 2>/dev/null | awk '{print $5}')
        echo "  ✓ Model pulled ($MODEL_SIZE)"

        # Save timestamped checkpoint copy
        cp "$LOCAL_DIR/homophone_transformer.pt" "$CHECKPOINT_DIR/transformer_${TIMESTAMP}.pt"

        # Rotate: keep last 10 checkpoints
        ls -1t "$CHECKPOINT_DIR"/transformer_*.pt 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true

        echo "  ✓ Checkpoint saved ($(ls "$CHECKPOINT_DIR" | wc -l) in rotation)"
    else
        echo "  ✗ Model pull failed (GPU may be busy — will retry)"
    fi

    # ── Pull training log ──
    if scp -i "$GPU_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -P "$GPU_PORT" "root@$GPU_HOST:$REMOTE_LOG" \
        "$LOCAL_DIR/training3.log" 2>/dev/null; then

        LATEST=$(tail -3 "$LOCAL_DIR/training3.log" 2>/dev/null | grep "epoch" | tail -1 || echo "")
        if [ -n "$LATEST" ]; then
            echo "  ✓ Log pulled — $LATEST"
        else
            echo "  ✓ Log pulled (no epoch line yet)"
        fi
    fi

    # ── Check GPU process health ──
    PROC_COUNT=$(ssh -i "$GPU_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -p "$GPU_PORT" "root@$GPU_HOST" \
        "ps aux | grep 'python.*train' | grep -v grep | wc -l" 2>/dev/null || echo "?")
    echo "  GPU processes: $PROC_COUNT"

    # ── Google Drive sync (every 2 pulls = every ~10 min) ──
    PULL_COUNT=$((PULL_COUNT + 1))
    if [ $((PULL_COUNT % 2)) -eq 0 ]; then
        echo ""
        echo "  ↑ Syncing to Google Drive..."

        # Sync latest model
        if [ -f "$LOCAL_DIR/homophone_transformer.pt" ]; then
            gzip -c "$LOCAL_DIR/homophone_transformer.pt" > "$LOCAL_DIR/homophone_transformer.pt.gz" 2>/dev/null || true
            rclone copy "$LOCAL_DIR/homophone_transformer.pt.gz" "$GDRIVE_DIR/" \
                --progress 2>/dev/null && echo "  ✓ Model synced to GDrive" || echo "  ✗ GDrive sync failed"
        fi

        # Sync training log
        if [ -f "$LOCAL_DIR/training3.log" ]; then
            rclone copy "$LOCAL_DIR/training3.log" "$GDRIVE_DIR/" 2>/dev/null || true
        fi

        # Sync latest 3 checkpoints
        ls -1t "$CHECKPOINT_DIR"/transformer_*.pt 2>/dev/null | head -3 | while read ckpt; do
            gzip -c "$ckpt" > "${ckpt}.gz" 2>/dev/null || true
            rclone copy "${ckpt}.gz" "$GDRIVE_DIR/checkpoints/" 2>/dev/null && \
                rm -f "${ckpt}.gz" || true
        done

        echo "  ✓ GDrive sync cycle complete"
    fi

    echo ""
    sleep 300
done
