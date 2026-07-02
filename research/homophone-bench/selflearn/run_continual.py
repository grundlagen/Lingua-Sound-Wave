"""Supervisor: keep the self-learning running with NO manual restarts.

train_selflearn.py --continual already skips a bad ROUND and keeps going; this
wraps the whole process so that if it DIES (OOM, kernel hiccup, disconnect), it is
relaunched automatically, resuming from the Drive checkpoint. Run this instead of
calling the trainer directly.

  python run_continual.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --ckpt_dir /content/drive/MyDrive/homophonic-carver --eval_llm
"""
from __future__ import annotations

import base64
import collections
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def gh_put(path, text, msg):
    """PUT a file to the selflearn-status branch (same scheme as status.json)."""
    tok, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO")
    if not tok or not repo:
        return
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    sha = None
    try:
        q = urllib.request.Request(api + "?ref=selflearn-status", headers=hdr)
        sha = json.load(urllib.request.urlopen(q)).get("sha")
    except Exception:
        pass
    body = {"message": msg, "branch": "selflearn-status",
            "content": base64.b64encode(text.encode()).decode()}
    if sha:
        body["sha"] = sha
    try:
        r = urllib.request.Request(api, data=json.dumps(body).encode(),
                                   headers=hdr, method="PUT")
        urllib.request.urlopen(r)
    except Exception as e:
        print(f"[crash-log push skipped: {e}]", flush=True)


def git_pull():
    """Pick up remotely pushed fixes before each relaunch (self-healing loop)."""
    try:
        out = subprocess.run(["git", "pull", "--rebase", "--autostash"],
                             cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=60)
        print(f"[supervisor] git pull: {out.stdout.strip() or out.stderr.strip()}",
              flush=True)
    except Exception as e:
        print(f"[supervisor] git pull skipped: {e}", flush=True)


def main():
    cmd = [sys.executable, os.path.join(HERE, "train_selflearn.py"), "--continual"]
    cmd += sys.argv[1:]
    backoff = 15
    launches = 0
    while True:
        launches += 1
        print(f"\n[supervisor] launch #{launches}: {' '.join(cmd)}", flush=True)
        tail = collections.deque(maxlen=120)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        rc = proc.wait()
        print(f"[supervisor] trainer exited rc={rc}; relaunch in {backoff}s "
              f"(resumes from checkpoint)", flush=True)
        if rc != 0:
            gh_put("selflearn/crash.log",
                   f"launch #{launches} rc={rc} at {time.strftime('%F %T')}\n"
                   + "".join(tail),
                   f"crash log: launch {launches} rc={rc}")
        git_pull()
        time.sleep(backoff)
        backoff = 15 if rc == 0 else min(300, backoff * 2)


if __name__ == "__main__":
    main()
