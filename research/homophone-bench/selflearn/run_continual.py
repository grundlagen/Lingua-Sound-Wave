"""Supervisor: keep the self-learning running with NO manual restarts.

train_selflearn.py --continual already skips a bad ROUND and keeps going; this
wraps the whole process so that if it DIES (OOM, kernel hiccup, disconnect), it is
relaunched automatically, resuming from the Drive checkpoint. Run this instead of
calling the trainer directly.

  python run_continual.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --ckpt_dir /content/drive/MyDrive/homophonic-carver --eval_llm
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cmd = [sys.executable, os.path.join(HERE, "train_selflearn.py"), "--continual"]
    cmd += sys.argv[1:]
    backoff = 15
    launches = 0
    while True:
        launches += 1
        print(f"\n[supervisor] launch #{launches}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        print(f"[supervisor] trainer exited rc={rc}; relaunch in {backoff}s "
              f"(resumes from checkpoint)", flush=True)
        time.sleep(backoff)
        backoff = 15 if rc == 0 else min(300, backoff * 2)


if __name__ == "__main__":
    main()
