"""Composition smoke test using S/A (usable_for_composition) entries only.

Takes English sentences, looks up each word in the working dictionary
(usable entries, best score first), chains the French sides, and checks
junctions: flags FR vowel-vowel hiatus across joins and reports the rhythm
budget (cumulative syllable delta). This is the minimal version of the
composition layer — no LLM, no decoder fallback — to verify the v5 fields
support chaining as designed.

Run: python compose_smoke.py "we do see the group" ...
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

entries = json.load(open("dictionary-v5.json"))
by_en = defaultdict(list)
for x in entries:
    if x.get("usable_for_composition") and x.get("direction") == "en_fr":
        by_en[x["en"]].append(x)
for k in by_en:
    by_en[k].sort(key=lambda x: -x["score"])


def compose(sentence: str):
    words = [w.lower().strip(".,!?") for w in sentence.split()]
    picks, missing = [], []
    for w in words:
        if by_en.get(w):
            picks.append(by_en[w][0])
        else:
            picks.append(None)
            missing.append(w)
    fr_line, warnings, delta = [], [], 0
    prev = None
    for w, p in zip(words, picks):
        if p is None:
            fr_line.append(f"[{w}]")
            continue
        fr_line.append(p["fr"])
        delta += p.get("syllable_delta", 0)
        if prev is not None:
            pc = prev.get("fr_coda", "")
            no = p.get("fr_onset", "")
            if pc.endswith("|V") and no.endswith("|V"):
                warnings.append(f"hiatus at '{prev['fr']} {p['fr']}'")
        prev = p
    covered = sum(1 for p in picks if p is not None)
    return {
        "en": sentence,
        "fr": " ".join(fr_line),
        "coverage": f"{covered}/{len(words)}",
        "missing": missing,
        "rhythm_delta": delta,
        "junction_warnings": warnings,
        "scores": [round(p["score"], 2) for p in picks if p],
    }


if __name__ == "__main__":
    tests = sys.argv[1:] or [
        "we do see the group",
        "two men set the test",
        "she said tell me more",
        "the bell is new",
    ]
    for t in tests:
        r = compose(t)
        print(f"\nEN: {r['en']}")
        print(f"FR: {r['fr']}")
        print(f"   coverage {r['coverage']}  rhythm_delta {r['rhythm_delta']}  "
              f"scores {r['scores']}")
        if r["missing"]:
            print(f"   missing: {r['missing']}")
        for w in r["junction_warnings"]:
            print(f"   ! {w}")
