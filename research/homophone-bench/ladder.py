"""The dual-reading LADDER: build writing where the English reading and the
French reading SOUND the same and MEAN the same -- both rails unanchored.

Implements the four pieces:

  1. TYPED SLOTS   every node gets three SEPARATE buckets, never flattened:
       homophone (≈, cross-language, sound) | translation (=, sense) |
       synonym (~, sense). (the fix for "all 3 is the scrabble")
  2. LADDER        a position is a homophone PAIR (en ≈ fr). Composition lays
       positions so the EN words read as English AND the FR words read as French
       -- the rungs carry sound, the rails (bigram LM each side) carry sense.
  3. LOOP-TILES    the alphabet = loop-certified pairs (en sounds-like fr AND
       means it: a closed ring where both rails reconcile). Finite, so end-sets
       are finite. Each tile carries an embedding sense-vector.
  4. SENSES        polysemy split: a word's translations clustered in embedding
       space (play -> {jouer...} vs {pièce...}) so a rung picks the right sense.

Run: python ladder.py            (demos 1,3,4 + composes dual-reading lines)
     python ladder.py play sea love   (seed words for the composition theme)
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict

import numpy as np

import bigram_lm

SOUND, MEANING = "S", "M"


# ---------- 1. TYPED SLOTS -------------------------------------------------
def typed_slots(edges):
    homo, trans, syn = defaultdict(set), defaultdict(set), defaultdict(set)
    for n, lst in edges.items():
        for m, q, lab, fam in lst:
            if lab == "≈":
                homo[n].add(m)
            elif lab == "=":
                trans[n].add(m)
            elif lab == "~":
                syn[n].add(m)
    return homo, trans, syn


# ---------- 3. LOOP-TILES (the alphabet) -----------------------------------
def load_tiles(idx, vecs, min_cert=1):
    tiles = []
    for i, line in enumerate(open("loop-certified-pairs-v7u.tsv", encoding="utf-8")):
        if i == 0:
            continue
        en, fr, cert, *_ = line.rstrip("\n").split("\t")
        if int(cert) < min_cert:
            continue
        ne, nf = f"en:{en}", f"fr:{fr}"
        if ne in idx and nf in idx:
            v = (vecs[idx[ne]] + vecs[idx[nf]]) / 2.0       # shared sense vector
            tiles.append({"en": en, "fr": fr, "cert": int(cert), "vec": v})
    return tiles


# ---------- 4. SENSE SPLIT (polysemy) --------------------------------------
def sense_clusters(word, trans, idx, vecs, k=2):
    """cluster a word's French translations by embedding -> distinct senses."""
    cands = [t for t in trans.get(f"en:{word}", ()) if t in idx]
    if len(cands) < 3:
        return [cands]
    V = np.array([vecs[idx[c]] for c in cands])
    # tiny k-means (cosine on normalized vecs)
    rng = np.random.default_rng(0)
    cen = V[rng.choice(len(V), k, replace=False)]
    for _ in range(8):
        a = np.argmax(V @ cen.T, axis=1)
        for j in range(k):
            if (a == j).any():
                cen[j] = V[a == j].mean(0)
                cen[j] /= np.linalg.norm(cen[j]) + 1e-9
    return [[cands[i][3:] for i in range(len(cands)) if a[i] == j] for j in range(k)]


# ---------- 2. LADDER COMPOSITION ------------------------------------------
def compose(tiles, EN, FR, seed_vec, length=6, beam=40):
    """Lay loop-tiles so EN words read as English and FR words read as French,
    held near a theme. Each step is a rung (en≈fr) advancing both rails."""
    # start: tiles closest to the theme
    scored0 = sorted(tiles, key=lambda t: -float(t["vec"] @ seed_vec))[:beam]
    beams = [([t], float(t["vec"] @ seed_vec)) for t in scored0]
    for _ in range(length - 1):
        nxt = []
        for chain, sc in beams:
            last = chain[-1]
            used = {t["en"] for t in chain}
            cands = sorted(tiles, key=lambda t: -float(t["vec"] @ seed_vec))[:300]
            best_local = []
            for t in cands:
                if t["en"] in used or t["fr"] == last["fr"]:
                    continue
                en_c = EN.cond(last["en"].lower(), t["en"].lower())
                fr_c = FR.cond(last["fr"].split()[-1].lower(), t["fr"].split()[0].lower())
                theme = float(t["vec"] @ seed_vec)
                step = (en_c + 0.05) * (fr_c + 0.05) * (theme + 0.3)
                best_local.append((sc + np.log(step + 1e-9), chain + [t]))
            best_local.sort(key=lambda x: -x[0])
            nxt.extend(best_local[:3])
        nxt.sort(key=lambda x: -x[0])
        beams = [(c, s) for s, c in [(s, c) for s, c in nxt[:beam]]]
    # final rank by both-side fluency
    out = []
    for chain, _ in beams:
        en_line = [t["en"] for t in chain]
        fr_line = " ".join(t["fr"] for t in chain).split()
        score = EN.fluency([w.lower() for w in en_line]) * FR.fluency([w.lower() for w in fr_line])
        out.append((score, chain))
    out.sort(key=lambda x: -x[0])
    return out


def main():
    seeds = sys.argv[1:] or ["love", "sea", "night", "play"]
    edges = pickle.load(open("graph-v7u.pkl", "rb"))["edges"]
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    homo, trans, syn = typed_slots(edges)
    EN, FR = bigram_lm.load("en"), bigram_lm.load("fr")

    print("=== 1. TYPED SLOTS (three separate buckets, not flattened) ===")
    for w in ["key", "play", "sea"]:
        n = f"en:{w}"
        print(f"  {w}:")
        print(f"     homophone(≈): {sorted(list(homo.get(n, []))[:6])}")
        print(f"     translation(=): {sorted(x[3:] for x in list(trans.get(n, []))[:6])}")
        print(f"     synonym(~): {sorted(x[3:] for x in list(syn.get(n, []))[:6])}")

    print("\n=== 4. SENSE SPLIT (polysemy clustered by embedding) ===")
    for w in ["play", "spring", "bank"]:
        cl = sense_clusters(w, trans, idx, vecs)
        print(f"  {w}: " + "  |  ".join("{" + ", ".join(c[:5]) + "}" for c in cl if c))

    tiles = load_tiles(idx, vecs)
    print(f"\n=== 3. LOOP-TILE ALPHABET: {len(tiles)} dual atoms "
          f"(en sounds-like fr AND means it) ===")
    for t in tiles[:6]:
        print(f"     {t['en']:14s} ≈≈ {t['fr']:14s}  (x{t['cert']})")

    print("\n=== 2. LADDER: dual-reading lines (EN reads English / FR reads French,"
          " same sound, same drift) ===")
    seed_vec = np.mean([vecs[idx[f"en:{s}"]] for s in seeds if f"en:{s}" in idx], axis=0)
    ranked = compose(tiles, EN, FR, seed_vec)
    for score, chain in ranked[:5]:
        en_line = " ".join(t["en"] for t in chain)
        fr_line = " ".join(t["fr"] for t in chain)
        print(f"  EN: {en_line}")
        print(f"  FR: {fr_line}      (both-side fluency {score:.4f})\n")


if __name__ == "__main__":
    main()
