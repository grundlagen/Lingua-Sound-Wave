"""seed_learn.py — learn a pair's pronunciation-correspondence rules from its
own mined seed dictionary. Answers "is the seed enough to learn the rules?"

For any pair built by multilang.py, this aligns the high-confidence (S-tier)
mined pairs phoneme-by-phoneme and counts which source-phoneme ~ target-
phoneme correspondences actually occur (Ristad & Yianilos learned edit
distance, count form). The counts ARE the learned pronunciation rules:
they say "in this language pair, English /ɪ/ regularly lands on Spanish /i/"
without anyone hand-coding it. Output feeds the matcher as a per-pair cost
overlay (pairs/<pair>/learned-costs.json).

    python seed_learn.py en-es

Honest finding it demonstrates: the seed is ENOUGH to learn the cost rules,
because the panphon featural baseline already scores any pair out of the box
and the seed only needs to refine it. No parallel corpus, no training run —
just the dictionary the engine already mined.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import matcher
from matcher import _canonical, _segs, _vecs, _sub_matrix, _gap_cost
import numpy as np


def align(ipa_a: str, ipa_b: str):
    sa, sb = _segs(_canonical(ipa_a)), _segs(_canonical(ipa_b))
    va, vb = _vecs(_canonical(ipa_a)), _vecs(_canonical(ipa_b))
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return []
    sub = _sub_matrix(sa, va, sb, vb)
    ga = [_gap_cost(s) for s in sa]
    gb = [_gap_cost(s) for s in sb]
    cost = np.zeros((n + 1, m + 1))
    for j in range(1, m + 1):
        cost[0, j] = cost[0, j - 1] + gb[j - 1]
    for i in range(1, n + 1):
        cost[i, 0] = cost[i - 1, 0] + ga[i - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + sub[i - 1, j - 1],
                             cost[i - 1, j] + ga[i - 1], cost[i, j - 1] + gb[j - 1])
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i, j] - (cost[i - 1, j - 1] + sub[i - 1, j - 1])) < 1e-9:
            out.append((sa[i - 1], sb[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and abs(cost[i, j] - (cost[i - 1, j] + ga[i - 1])) < 1e-9:
            i -= 1
        else:
            j -= 1
    return out


def main():
    pair = sys.argv[1] if len(sys.argv) > 1 else "en-es"
    dpath = os.path.join("pairs", pair, "dictionary.json")
    if not os.path.exists(dpath):
        print(f"no mined dictionary at {dpath}; run: python multilang.py "
              f"{pair.replace('-', ' ')}")
        return
    entries = [e for e in json.load(open(dpath)) if e["score"] >= 0.70]
    subs = Counter()
    for e in entries:
        for a, b in align(e["src_ipa"], e["tgt_ipa"]):
            if a != b:
                subs[tuple(sorted((a.replace("ː", ""), b.replace("ː", ""))))] += 1

    def curve(n):
        return round(1.0 / (1.0 + math.log(1 + n)), 3)

    learned = {f"{a}|{b}": curve(n) for (a, b), n in subs.items() if n >= 1}
    outpath = os.path.join("pairs", pair, "learned-costs.json")
    json.dump({"pairs": learned, "gaps": {},
               "meta": {"pair": pair, "seed_entries": len(entries),
                        "distinct_correspondences": len(subs)}},
              open(outpath, "w"), ensure_ascii=False, indent=1)
    print(f"{pair}: learned {len(learned)} correspondence rules from "
          f"{len(entries)} seed pairs -> {outpath}")
    print("top learned pronunciation rules (src~tgt, count, cost):")
    for (a, b), n in subs.most_common(15):
        print(f"    {a} ~ {b:3s}  x{n:3d}  cost {curve(n)}")


if __name__ == "__main__":
    main()
