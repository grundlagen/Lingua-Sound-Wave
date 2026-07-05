"""Self-experiment harness -- sweep parameters and LOG what wins, WITHOUT ever
editing the existing code.

Preservation guarantee: this imports the live modules read-only and varies their
parameters at RUNTIME (monkeypatch of in-memory constants), never writing to any
.py. Results append to experiments/results.jsonl. Run it on the dedicated
`selflearn-experiments` branch so main code/history stays pristine; promote a
winning setting only by a deliberate, reviewed edit.

Scored on a tiny labelled set: known-good homophone pairs (1) vs bad ones (0).
A config is better if it SEPARATES them more (mean_good - mean_bad).

Run: python experiment.py
"""
from __future__ import annotations

import importlib
import json
import os
import time

import prosody
import matcher

GOOD = [("Humpty Dumpty", "un petit un petit"), ("it for", "est fort"),
        ("to tell", "t elle"), ("the door", "un voile d or"),
        ("Hickory dickory dock", "cries critiques"), ("boot", "bout")]
BAD = [("Humpty Dumpty", "zzz qqq"), ("it for", "bonjour madame"),
       ("to tell", "chie cede telle mi mot"), ("the door", "ordinateur"),
       ("boot", "fromage"), ("cat", "telephone")]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
os.makedirs(OUT, exist_ok=True)


def separation(score_fn):
    g = [score_fn(e, f) for e, f in GOOD]
    b = [score_fn(e, f) for e, f in BAD]
    return round(sum(g) / len(g) - sum(b) / len(b), 3), \
        round(sum(g) / len(g), 3), round(sum(b) / len(b), 3)


def main():
    results = []

    # baseline (current prosody settings)
    base = separation(prosody.prosodic_score)
    results.append({"config": "baseline", "sep": base[0],
                    "mean_good": base[1], "mean_bad": base[2]})

    # sweep 1: how cheap should unstressed syllables be? (W_UNSTRESSED)
    orig = prosody.W_UNSTRESSED
    for w in (0.15, 0.25, 0.3, 0.4, 0.5):
        prosody.W_UNSTRESSED = w
        sep = separation(prosody.prosodic_score)
        results.append({"config": f"W_UNSTRESSED={w}", "sep": sep[0],
                        "mean_good": sep[1], "mean_bad": sep[2]})
    prosody.W_UNSTRESSED = orig

    # sweep 2: diverged blend on/off
    for div in (True, False):
        sep = separation(lambda e, f: prosody.prosodic_score(e, f, diverged=div))
        results.append({"config": f"diverged={div}", "sep": sep[0],
                        "mean_good": sep[1], "mean_bad": sep[2]})

    # sweep 3: plain combo (no prosody) for reference
    sep = separation(lambda e, f: matcher.homophone_score(e, "en", f, "fr")["score"])
    results.append({"config": "plain_combo", "sep": sep[0],
                    "mean_good": sep[1], "mean_bad": sep[2]})

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT, "results.jsonl"), "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"time": stamp, **r}) + "\n")

    results.sort(key=lambda r: -r["sep"])
    print(f"experiment @ {stamp}  (good vs bad separation; higher = better)\n")
    for r in results:
        print(f"  sep {r['sep']:+.3f}  good {r['mean_good']:.2f} bad {r['mean_bad']:.2f}"
              f"   {r['config']}")
    print(f"\nbest: {results[0]['config']}  -> review, then promote by a deliberate "
          f"edit (old code preserved in git).")
    print(f"logged -> experiments/results.jsonl")


if __name__ == "__main__":
    main()
