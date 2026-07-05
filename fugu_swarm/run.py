#!/usr/bin/env python3
"""fugu_swarm.run — launch OpenFugu's serve.py with the no-ChatGPT pool.

Reads pool/no-chatgpt.yaml, enforces the no-OpenAI invariant, checks env vars,
then builds (and optionally execs) the OpenFugu serve command. Keeping this as a
thin launcher means we depend on OpenFugu rather than copying it.

    python -m fugu_swarm.run                       # print the serve command
    python -m fugu_swarm.run --serve               # actually exec it
    python -m fugu_swarm.run --pool pool/x.yaml --port 8088
"""
from __future__ import annotations

import argparse
import os
import sys

from .pool import slot_csv, missing_env

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_POOL = os.path.join(ROOT, "pool", "no-chatgpt.yaml")
OPENFUGU = os.path.join(ROOT, "vendor", "OpenFugu")


def load_pool(path: str):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    models = [w["model"] for w in cfg.get("workers", []) if w.get("model")]
    return cfg, models


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch OpenFugu with the no-ChatGPT pool.")
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--allow-openai", action="store_true",
                    help="override the no-ChatGPT policy (not recommended)")
    ap.add_argument("--serve", action="store_true",
                    help="exec the server instead of just printing the command")
    args = ap.parse_args(argv)

    cfg, models = load_pool(args.pool)
    csv = slot_csv(models, allow_openai=args.allow_openai)  # raises if ChatGPT slips in

    miss = missing_env(models)
    if miss:
        print(f"[warn] missing env for this pool: {', '.join(miss)}", file=sys.stderr)

    model_dir = cfg.get("router", {}).get("backbone", "Qwen/Qwen3-0.6B")
    vector = cfg.get("router", {}).get("vector", "artifacts/model_iter_60.npy")
    serve = os.path.join(OPENFUGU, "openfugu", "serve.py")

    cmd = [sys.executable, serve, "--model", model_dir, "--vector", vector,
           "--slot-models", csv, "--port", str(args.port)]
    print(f"# pool ({len(models)} slots, no ChatGPT): {csv}")
    print(" ".join(cmd))

    if args.serve:
        if not os.path.exists(serve):
            sys.exit(f"OpenFugu not found at {serve} — run ./setup.sh first.")
        os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
