"""Webbing density of the dual-atom alphabet: how do the loop-tiles interconnect,
and what are the best SETS to compose within?

Two connection directions (the 'vertical and horizontal' ladders):

  HORIZONTAL  tile A can be FOLLOWED by tile B in a line -- the EN words chain
              (EN bigram) AND the FR words chain (FR bigram). This advances both
              poems left-to-right.
  VERTICAL    tiles that can occupy the SAME position: they share a word, or their
              EN words are sound-twins (≈) and so are their FR words -- alternate
              rungs stacked at one slot, giving the search room to satisfy the
              horizontal constraint.

Then we find the best SETS: cluster tiles by shared sense (embedding) and score
each cluster by internal webbing density (how many member pairs actually connect).
Dense clusters are the writeable stanza-pools; the hubs are the interchanges.

Run: python webbing.py
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np

import bigram_lm


def load_tiles(idx, vecs):
    tiles = []
    for i, line in enumerate(open("loop-certified-pairs-v7u.tsv", encoding="utf-8")):
        if i == 0:
            continue
        en, fr, cert, *_ = line.rstrip("\n").split("\t")
        ne, nf = f"en:{en}", f"fr:{fr}"
        if ne in idx and nf in idx:
            tiles.append({"en": en, "fr": fr, "cert": int(cert),
                          "vec": (vecs[idx[ne]] + vecs[idx[nf]]) / 2})
    return tiles


def homo_twins(edges):
    """en-word -> set of en-words that are sound-interchangeable (share an ≈ fr)."""
    fr_to_en = defaultdict(set)
    for n, lst in edges.items():
        if not n.startswith("en:"):
            continue
        for m, q, lab, fam in lst:
            if lab == "≈":
                fr_to_en[m].add(n[3:])
    twin = defaultdict(set)
    for fr, ens in fr_to_en.items():
        for e in ens:
            twin[e] |= ens - {e}
    return twin


def main():
    edges = pickle.load(open("graph-v7u.pkl", "rb"))["edges"]
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    EN, FR = bigram_lm.load("en"), bigram_lm.load("fr")
    tiles = load_tiles(idx, vecs)
    N = len(tiles)
    print(f"alphabet: {N} dual-atom tiles\n")

    # ---- HORIZONTAL: directed follow-edges from OBSERVED corpus bigrams ----
    # (cond() is smoothed -> always >0; the real signal is bi[(a,b)] > 0)
    h_out = defaultdict(list)
    both = 0
    for i, a in enumerate(tiles):
        ae, af = a["en"].lower(), a["fr"].split()[-1].lower()
        for j, b in enumerate(tiles):
            if i == j:
                continue
            ec = EN.bi.get((ae, b["en"].lower()), 0)
            fc = FR.bi.get((af, b["fr"].split()[0].lower()), 0)
            if ec or fc:                          # real co-occurrence on a rail
                dual = ec > 0 and fc > 0
                h_out[i].append((j, ec + fc, dual))
                both += dual
    h_deg = sum(len(v) for v in h_out.values())
    follow = sum(1 for i in range(N) if h_out[i])
    print("== HORIZONTAL webbing (OBSERVED corpus bigrams) ==")
    print(f"   follow-edges: {h_deg}  avg out-degree: {h_deg/N:.2f}  "
          f"tiles with >=1 follow: {follow} ({follow/N*100:.0f}%)")
    print(f"   BOTH-rail follow-edges (EN and FR both attested): {both} "
          f"-- the gold transitions")

    # ---- VERTICAL: same-slot interchangeable rungs ----
    twin = homo_twins(edges)
    by_en, by_fr = defaultdict(list), defaultdict(list)
    for i, t in enumerate(tiles):
        by_en[t["en"]].append(i); by_fr[t["fr"]].append(i)
    v_deg = 0
    for i, t in enumerate(tiles):
        alts = set()
        for j in by_en[t["en"]] + by_fr[t["fr"]]:
            alts.add(j)
        for e2 in twin.get(t["en"], ()):           # sound-twin EN words
            for j in by_en.get(e2, ()):
                alts.add(j)
        alts.discard(i)
        v_deg += len(alts)
    print("\n== VERTICAL webbing (interchangeable rungs at one slot) ==")
    print(f"   stack-edges: {v_deg}  avg stack options: {v_deg/N:.2f}")

    # ---- BEST SETS: thematic clusters scored by internal density ----
    V = np.array([t["vec"] for t in tiles])
    K = 16
    rng = np.random.default_rng(0)
    cen = V[rng.choice(N, K, replace=False)]
    for _ in range(12):
        a = np.argmax(V @ cen.T, axis=1)
        for k in range(K):
            if (a == k).any():
                cen[k] = V[a == k].mean(0); cen[k] /= np.linalg.norm(cen[k]) + 1e-9
    print("\n== BEST SETS (thematic clusters by internal webbing density) ==")
    clusters = []
    for k in range(K):
        members = [i for i in range(N) if a[i] == k]
        if len(members) < 4:
            continue
        ms = set(members)
        internal = sum(1 for i in members for j, *_ in h_out[i] if j in ms)
        dens = internal / (len(members) * (len(members) - 1) + 1e-9)
        clusters.append((dens, internal, members))
    clusters.sort(reverse=True)
    for dens, internal, members in clusters[:5]:
        words = ", ".join(f"{tiles[i]['en']}≈{tiles[i]['fr']}" for i in members[:6])
        print(f"   density {dens:.3f} ({internal} internal links, {len(members)} tiles): {words}")

    # ---- hubs: tiles on the most follow-paths ----
    indeg = defaultdict(int)
    for i in range(N):
        for j, *_ in h_out[i]:
            indeg[j] += 1
    hubs = sorted(range(N), key=lambda i: -(indeg[i] + len(h_out[i])))[:12]
    print("\n== INTERCHANGE HUB TILES (highest total degree) ==")
    for i in hubs:
        print(f"   {tiles[i]['en']:12s} ≈ {tiles[i]['fr']:12s}  "
              f"(in {indeg[i]} / out {len(h_out[i])})")
    print(f"\nReading: horizontal coverage ({sum(1 for i in range(N) if h_out[i])/N*100:.0f}% "
          f"of tiles can be followed) is the wall -- denser French-anchored rungs "
          f"and more loops raise it. The dense clusters are the writeable pools.")


if __name__ == "__main__":
    main()
