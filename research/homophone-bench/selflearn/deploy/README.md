# Deploy: run the self-learning CONTINUALLY on your own GPU box

No Colab, no "press Run" — the trainer runs as a service that auto-restarts and
resumes from checkpoint. Works on any Ubuntu host with an NVIDIA GPU (Hetzner GPU
dedicated, RunPod, Vast, Lambda).

## Important boundary
This is the recipe; **you** launch it on a box you own. Running a GPU server costs
money (Hetzner GPU dedicated ≈ €180+/mo; RunPod/Vast ≈ $0.2–0.4/hr) and I don't
have your cloud credentials, so I can't and won't provision or bill anything —
that stays your call.

## Option A — one command (bare box)
```bash
export OPENROUTER_API_KEY=...        # optional: per-round LLM eval
export GITHUB_TOKEN=...              # optional: private clone + status push
export GITHUB_REPO=grundlagen/Lingua-Sound-Wave
curl -fsSL https://raw.githubusercontent.com/grundlagen/Lingua-Sound-Wave/claude/phrase-weave-multiword/research/homophone-bench/selflearn/deploy/setup.sh | bash
```

## Option B — systemd service (survives reboots, mirrors poly-microtrader)
```bash
sudo mkdir -p /opt/lingua-selflearn && sudo cp setup.sh /opt/lingua-selflearn/
echo "OPENROUTER_API_KEY=..." | sudo tee /opt/lingua-selflearn/.env
sudo cp selflearn.service /etc/systemd/system/
sudo systemctl enable --now selflearn@$USER
journalctl -u selflearn@$USER -f          # watch it learn
```

## Option C — Docker (any GPU host)
```bash
docker build -t lingua-selflearn selflearn/deploy
docker run -d --gpus all --restart unless-stopped \
  -e OPENROUTER_API_KEY=... -v $PWD/ckpt:/ckpt lingua-selflearn
```

All three run `run_continual.py` (loops rounds forever, skips bad rounds,
relaunches on crash) and push `status.json` to the `selflearn-status` branch so
Claude can monitor on request.
