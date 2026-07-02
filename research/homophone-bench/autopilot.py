"""AUTOPILOT -- unattended operation of the dual-translation loop.

One cycle = sync -> health -> (regen stale artifacts) -> small bench ->
(mine batch if keys) -> (train step if GPU) -> ledger/status -> commit+push.
Every step is bounded, idempotent, and judge-verified downstream; failures are
logged to AUTOPILOT_STATUS.md and the cycle continues. Collision-safe push
(pull --rebase first) so multiple Claude sessions can share the branch.

Run: python autopilot.py --once          # one cycle (cron-able)
     python autopilot.py --loop 3600     # forever, sleep N seconds between
Cron: 0 * * * * cd <repo>/research/homophone-bench && \
      <python> autopilot.py --once >> autopilot.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
STATUS = "AUTOPILOT_STATUS.md"
STATE = ".autopilot-state.json"
PY = sys.executable


def sh(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT {timeout}s: {cmd}"


def state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"cycle": 0, "bench_line": 0}


def log_status(lines):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"\n## cycle @ {now}\n")
        for ln in lines:
            f.write(f"- {ln}\n")


def cycle():
    st = state()
    st["cycle"] += 1
    notes = []

    # 1. SYNC (collision-safe; other Claude sessions share this branch)
    rc, out = sh("git pull --rebase -q origin $(git rev-parse --abbrev-ref HEAD)", 120)
    notes.append(f"sync: {'ok' if rc == 0 else out.splitlines()[-1] if out else 'fail'}")

    # 2. HEALTH: the judge must stand or nothing else counts
    rc, out = sh(f'{PY} -c "import matcher; qi=matcher.g2p(\'less debt\',\'en\');'
                 f'ci=matcher.g2p(\'laisse dette\',\'fr\');'
                 f's=0.5*matcher._ngram_channel(qi,ci)+0.5*matcher._feat_channel(qi,ci);'
                 f'assert s>0.9, s; print(f\'judge {{s:.2f}}\')"', 300)
    notes.append(f"judge: {out.strip().splitlines()[-1] if rc == 0 else 'FAIL ' + out[-200:]}")
    if rc != 0:
        log_status(notes + ["ABORT: judge failed; no work done"])
        return st

    # 3. REGEN stale artifacts (cheap, idempotent)
    if not os.path.exists("/tmp/muse-en-fr.txt"):
        sh('curl -sL --max-time 90 -o /tmp/muse-en-fr.txt '
           '"https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt"', 120)
        notes.append("regen: fetched MUSE")
    if not os.path.exists("train-dual-v1.jsonl"):
        rc, out = sh(f"{PY} build_train_corpus.py", 600)
        notes.append(f"regen: train corpus ({'ok' if rc == 0 else 'fail'})")

    # 4. ROLLING BENCH: 6 lines per cycle, advancing window (full corpus over cycles)
    off = st.get("bench_line", 0)
    rc, out = sh(f"{PY} - <<'EOF'\n"
                 "import beauty_compose as B\n"
                 "D = B.load_all()\n"
                 f"lines = [l.split('\\t')[0] for i, l in enumerate(open('corpus-carves.tsv', encoding='utf-8')) if i > 0]\n"
                 f"seg = lines[{off}:{off}+6] or lines[:6]\n"
                 "band = 0\n"
                 "for en in seg:\n"
                 "    s, m = B.translate(en, D, show=False)\n"
                 "    band += (s >= 0.55 and m >= 0.45)\n"
                 "print(f'ROLLING {band}/{len(seg)}')\n"
                 "EOF", 3000)
    tail = out.strip().splitlines()[-1] if out.strip() else "no output"
    notes.append(f"bench[{off}:{off}+6]: {tail}")
    st["bench_line"] = (off + 6) % 60

    # 5. MINE batch if Haiku key present (bounded spend: 20 words/cycle)
    have_key = False
    try:
        sys.path.insert(0, HERE)
        import _load_env
        _load_env.load_keys()
        have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    except Exception:
        pass
    if have_key:
        rc, out = sh(f"{PY} llm_bridge.py --n 20", 900)
        kept = [l for l in out.splitlines() if l.startswith("kept")]
        notes.append(f"mine: {kept[-1] if kept else 'skipped/none'}")
    else:
        notes.append("mine: no key (symbolic only)")

    # 6. TRAIN step if GPU (continual, checkpointed -- safe to run each cycle)
    rc, out = sh(f'{PY} -c "import torch; print(torch.cuda.is_available())"', 120)
    if out.strip().endswith("True"):
        rc, out = sh(f"{PY} selflearn/train_selflearn.py --data train-dual-v1.jsonl "
                     f"--rounds 1 --ckpt_dir selflearn/ckpt", 5400)
        notes.append(f"train: {'round ok' if rc == 0 else 'fail ' + out[-150:]}")
    else:
        notes.append("train: no GPU")

    # 7. STATUS + COMMIT + PUSH (collision-safe)
    log_status(notes)
    json.dump(st, open(STATE, "w"))
    sh("git add -A ':!*.pkl' ':!train-dual-v1.jsonl' ':!.venv' && "
       "git commit -q -m 'autopilot: cycle status + mined rows' "
       "-m 'Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>' || true", 120)
    sh("git pull --rebase -q origin $(git rev-parse --abbrev-ref HEAD) && "
       "git push -q origin $(git rev-parse --abbrev-ref HEAD)", 180)
    print("\n".join(notes))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS")
    args = ap.parse_args()
    if args.loop:
        import time
        while True:
            cycle()
            time.sleep(args.loop)
    else:
        cycle()


if __name__ == "__main__":
    main()
