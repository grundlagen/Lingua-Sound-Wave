"""web_poet -- generative homophonic poetry by THEMED WALK on the v7 web.

Outside-the-box move: don't translate a fixed sentence (that forces both rails to
chain, which the webbing showed is ~0). Instead GENERATE -- walk the web so the
poem is whatever the walk finds, constrained to:

  * SOUND CONTINUITY  -- the walk alternates a sound hop (≈, the homophonic
    handoff that makes the line flow when read aloud) with a meaning hop (~/=,
    which steers the sense). Strict typed alternation -- the discipline the
    composition needs (not the 'scrabble' free mix).
  * THEME GRAVITY     -- meaning hops are chosen to stay near a seed theme vector
    (embeddings), so the poem drifts AROUND a subject instead of wandering.
  * DUAL ATOMS as rests -- landing on a loop-certified pair (sound==meaning) is
    rewarded; those are the lines' natural cadence points (★).

The ribbon read aloud is one continuous homophonic stream; read as words it is a
found poem that lives inside the dictionary. Both 'readings' (EN words / FR words)
sound the same because every sound hop is a twin pair.

Run: python web_poet.py sea night love
"""
from __future__ import annotations

import json
import pickle
import sys
from heapq import nlargest

import numpy as np

import bigram_lm

SOUND, MEANING = "S", "M"


def main():
    seeds = sys.argv[1:] or ["sea", "night", "love"]
    edges = pickle.load(open("graph-v7u.pkl", "rb"))["edges"]
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    EN, FR = bigram_lm.load("en"), bigram_lm.load("fr")

    from wordfreq import zipf_frequency
    def ok(n):
        return all(zipf_frequency(t, n[:2]) >= 2.3 for t in n[3:].split())
    valid = set()
    for n in ids:
        if n[3:].replace(" ", "").isalpha() and ok(n):
            valid.add(n)

    atoms = set()
    for i, line in enumerate(open("loop-certified-pairs-v7u-aug.tsv", encoding="utf-8")):
        if i and len(line.split("\t")) >= 2:
            en, fr = line.split("\t")[:2]
            atoms.add(f"en:{en}"); atoms.add(f"fr:{fr}")

    def vec(n):
        return vecs[idx[n]] if n in idx else None

    def walk(theme, steps=11, beam=80):
        starts = [n for n in valid if n.startswith("en:") and n in idx]
        starts = nlargest(beam, starts, key=lambda n: float(vec(n) @ theme))
        # beam item: (score, nodes, last_family)
        B = [(float(vec(s) @ theme), [s], MEANING) for s in starts]
        for _ in range(steps):
            nxt = []
            for sc, nodes, lastf in B:
                cur = nodes[-1]
                want = SOUND if lastf == MEANING else MEANING
                cands = []
                for m, q, lab, fam in edges.get(cur, ()):
                    if m not in valid or m in nodes or m not in idx:
                        continue
                    if want == SOUND and lab != "≈":
                        continue
                    if want == MEANING and lab not in ("~", "="):
                        continue
                    th = float(vec(m) @ theme)
                    if want == SOUND:
                        step = 0.6 * q + 0.4 * th          # flow + stay on theme
                    else:
                        step = th                          # steer by meaning
                    step += 0.15 if m in atoms else 0.0     # rest on dual atoms
                    cands.append((sc + step, nodes + [m], fam))
                nxt.extend(nlargest(4, cands, key=lambda x: x[0]))
            if not nxt:
                break
            B = nlargest(beam, nxt, key=lambda x: x[0])
        return B

    def fluency(nodes):
        en = [n[3:] for n in nodes if n.startswith("en:")]
        fr = " ".join(n[3:] for n in nodes if n.startswith("fr:")).split()
        return (EN.fluency([w.lower() for w in en]) if en else 0,
                FR.fluency([w.lower() for w in fr]) if fr else 0)

    for s in seeds:
        if f"en:{s}" not in idx:
            continue
        theme = vec(f"en:{s}")
        B = walk(theme)
        # rank final ribbons by theme coherence x both-language fluency x atom rests
        ranked = []
        for sc, nodes, _ in B:
            ef, ff = fluency(nodes)
            th = np.mean([float(vec(n) @ theme) for n in nodes])
            rests = sum(1 for n in nodes if n in atoms)
            ranked.append((th * (ef + 0.1) * (ff + 0.1) * (1 + 0.1 * rests),
                           th, ef, ff, rests, nodes))
        ranked.sort(key=lambda x: -x[0])
        print(f"\n{'='*70}\nTHEME: {s}\n{'='*70}")
        for score, th, ef, ff, rests, nodes in ranked[:3]:
            ribbon = "  ".join(
                ("★" if n in atoms else "") + n[3:] for n in nodes)
            print(f"\n  theme {th:.2f}  EN-flow {ef:.2f}  FR-flow {ff:.2f}  "
                  f"rests {rests}")
            print(f"    {ribbon}")
        print("\n  (read the words aloud in a French mouth: the ≈ handoffs keep one "
              "continuous sound;\n   ★ = a dual atom where sound and meaning lock.)")


if __name__ == "__main__":
    main()
