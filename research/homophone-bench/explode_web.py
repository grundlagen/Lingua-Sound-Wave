"""Explode the chain-web so ALL steps are connection points, and certify
the loop interiors as semantic+homophonic translation units.

Answers two questions about chain-web.tsv:

1. "Endpoint-to-beginning only, or all steps?" — the weave recorded only
   seed -> final-endpoint edges. But every intermediate node of every chain
   is itself a join point. This script parses each recorded chain and emits
   EVERY sub-path (node_i -> node_j, j > i+1) as a derived connection, so
   the web's internal steps become first-class, queryable edges
   (chain-web-full.tsv). 1-hop pairs are skipped (already base edges).

2. "Can the loop parts give semantic AND homophonic translation?" — yes:
   a loop certifies that meaning survived the whole round trip, so every
   SOUND edge inside a loop is a meaning-preserving homophone pair. Those
   pairs are extracted with their loop provenance into
   loop-certified-pairs.tsv — the self-certified gold layer for dual
   translation, and the dataset-improvement engine: certified pairs can be
   promoted in the dictionary, which densifies the graph, which yields
   more loops (a bootstrap).

Usage: python explode_web.py [chain-web.tsv chain-loops.tsv | -S]
"""
from __future__ import annotations

import re
import sys
from collections import Counter

SEP = re.compile(r" ([≈=~]) ")


def parse_chain(s: str):
    """'en:a ≈ fr:b ~ fr:c' -> [('en:a',None), ('fr:b','≈'), ('fr:c','~')]"""
    parts = SEP.split(s)
    out = [(parts[0], None)]
    for i in range(1, len(parts), 2):
        out.append((parts[i + 1], parts[i]))
    return out


def main():
    suffix = "-S" if "-S" in sys.argv else ""
    web_f = f"chain-web{suffix}.tsv"
    loops_f = f"chain-loops{suffix}.tsv"

    # ---- 1. explode all sub-paths into connections ----
    best = {}
    n_chains = 0
    with open(web_f, encoding="utf-8") as f:
        next(f)
        for line in f:
            src, dst, hops, quality, chain = line.rstrip("\n").split("\t")
            n_chains += 1
            nodes = parse_chain(chain)
            q = float(quality)
            for i in range(len(nodes)):
                for j in range(i + 2, len(nodes)):  # skip 1-hop (base edges)
                    a, b = nodes[i][0], nodes[j][0]
                    sub = " ".join(
                        n if lab is None or k == i else f"{lab} {n}"
                        for k, (n, lab) in enumerate(nodes[i:j + 1]))
                    key = (a, b)
                    val = (j - i, q, sub)
                    if key not in best or val > best[key]:
                        best[key] = val
    with open(f"chain-web-full{suffix}.tsv", "w", encoding="utf-8") as f:
        f.write("a\tb\thops\tquality\tsubchain\n")
        for (a, b), (h, q, sub) in sorted(best.items(),
                                          key=lambda kv: -kv[1][1]):
            f.write(f"{a}\t{b}\t{h}\t{q:.3f}\t{sub}\n")
    print(f"exploded {n_chains} chains -> {len(best)} all-step connections "
          f"(chain-web-full{suffix}.tsv)")

    # ---- 2. certify loop interiors ----
    cert = Counter()
    example = {}
    n_loops = 0
    try:
        lf = open(loops_f, encoding="utf-8")
    except FileNotFoundError:
        lf = None
    if lf:
        next(lf)
        for line in lf:
            seed, hops, quality, loop = line.rstrip("\n").split("\t")
            n_loops += 1
            nodes = parse_chain(loop)
            for k in range(1, len(nodes)):
                if nodes[k][1] == "≈":
                    a, b = nodes[k - 1][0], nodes[k][0]
                    en = a if a.startswith("en:") else b
                    fr = b if b.startswith("fr:") else a
                    if en.startswith("en:") and fr.startswith("fr:"):
                        pair = (en[3:], fr[3:])
                        cert[pair] += 1
                        example.setdefault(pair, loop)
        lf.close()
    with open(f"loop-certified-pairs{suffix}.tsv", "w", encoding="utf-8") as f:
        f.write("en\tfr\tcertifications\texample_loop\n")
        for (en, fr), n in cert.most_common():
            f.write(f"{en}\t{fr}\t{n}\t{example[(en, fr)]}\n")
    print(f"{n_loops} loops -> {len(cert)} loop-certified sound pairs "
          f"(loop-certified-pairs{suffix}.tsv)")


if __name__ == "__main__":
    main()
