"""Learn the EN->FR phoneme substitution map FROM THE GOLD DATA (data-driven
'embeds' for the sound channel) -- the best practice vs hand-coded EQUIV tables.

Align the IPA of every GOLD sound∧meaning pair (the best dataset) and tally which
English phoneme gets realised as which French phoneme. The result is an empirical
substitution table = a learned phoneme similarity space, which can (a) replace/
augment matcher.EQUIV, (b) seed a phoneme embedding (PMI -> SVD).

Run: python phoneme_map.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

import matcher

GOLD = "dictionary-v7-remined.tsv"
DICT = "dictionary-v7-integrated.json"


def segdist(a, b):
    if a == b:
        return 0.0
    va, vb = matcher._vecs(a), matcher._vecs(b)
    if len(va) == 0 or len(vb) == 0:
        return 0.6
    return min(1.0, float(np.abs(va[0] - vb[0]).sum()) / (2.0 * matcher.N_FEATURES))


def align(sa, sb, GAP=0.42):
    n, m = len(sa), len(sb)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * GAP; bt[i][0] = "U"
    for j in range(1, m + 1):
        D[0][j] = j * GAP; bt[0][j] = "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1][j - 1] + segdist(sa[i - 1], sb[j - 1])
            de, ins = D[i - 1][j] + GAP, D[i][j - 1] + GAP
            best = min(sub, de, ins)
            D[i][j] = best
            bt[i][j] = "D" if best == sub else ("U" if best == de else "L")
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        d = bt[i][j]
        if d == "D":
            pairs.append((sa[i - 1], sb[j - 1])); i -= 1; j -= 1
        elif d == "U":
            i -= 1
        else:
            j -= 1
    return pairs


def main():
    ipa = {}
    for e in json.load(open(DICT, encoding="utf-8")):
        if e.get("en_ipa") and e.get("fr_ipa"):
            ipa[(e["en"], e["fr"])] = (e["en_ipa"], e["fr_ipa"])

    sub = defaultdict(Counter)       # en-phoneme -> Counter(fr-phoneme)
    pairs = 0
    for i, line in enumerate(open(GOLD, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 6 or p[5] != "GOLD":
            continue
        ei_fi = ipa.get((p[0], p[1]))
        if not ei_fi:
            continue
        sa = matcher._segs(ei_fi[0]); sb = matcher._segs(ei_fi[1])
        if not sa or not sb:
            continue
        for a, b in align(sa, sb):
            sub[a][b] += 1
        pairs += 1

    print(f"learned EN->FR phoneme map from {pairs} GOLD pairs "
          f"({len(sub)} English phonemes)\n")
    print("English  ->  top French realisations (count)   [data-driven EQUIV]")
    rows = []
    for en_ph, c in sorted(sub.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        top = "  ".join(f"{fr}×{n}" for fr, n in c.most_common(4))
        if tot >= 8:
            print(f"  /{en_ph}/  ({tot:4d})  ->  {top}")
            for fr, n in c.most_common():
                rows.append((en_ph, fr, n))
    with open("phoneme-map-gold.tsv", "w", encoding="utf-8") as f:
        f.write("en_phoneme\tfr_phoneme\tcount\n")
        for a, b, n in rows:
            f.write(f"{a}\t{b}\t{n}\n")
    print("\nwrote phoneme-map-gold.tsv (the empirical sound model; seeds a "
          "phoneme embedding via PMI->SVD, or augments matcher.EQUIV).")


if __name__ == "__main__":
    main()
