"""Alternation chains: the translation-telephone game over the web.

The user's spec (key = clé ≈ clay = argile ≈ ...): chains must ALTERNATE
between meaning moves and sound moves — that's what makes them surprising.
A chain like "frog =trans grenouille" is correct but dead; the game is
  EN word =(translate)=> FR word ≈(sounds like)=> EN word =(translate)=> ...
with two extra liberties the user named:
  - any node may be swapped for a semantic neighbor (~sem edges), which
    counts as a meaning move;
  - sound moves may go through multiword splits (y+y) — those are ordinary
    sound edges here because v5 multiword entries are first-class.

We search with STRICT family alternation (meaning-family = {=trans, ~sem};
sound-family = {≈snd}), minimum 2 sound hops, depth <= 7, beam-bounded.
Two result kinds:
  chains  best long paths from the seed (ranked by mean edge quality)
  loops   paths that return to the seed's semantic neighborhood — the
          "= key" ending: last node is the seed itself or a ~sem neighbor.

Usage: python chain_game.py key frog sea bear
Output: chain-game-demo.txt
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

MAX_HOPS = 7
BEAM_PER_NODE = 8
MIN_SOUND_HOPS = 2

SOUND, MEANING = "S", "M"


def build_graph(min_sound: float = 0.0):
    entries = json.load(open("dictionary-v5.json"))
    edges = defaultdict(list)   # node -> [(node, quality, label, family)]

    def add(a, b, q, label, fam):
        edges[a].append((b, q, label, fam))
        edges[b].append((a, q, label, fam))

    for e in entries:
        if e.get("usable_for_composition") and e["score"] >= min_sound:
            add(f"en:{e['en']}", f"fr:{e['fr']}", e["score"], "≈", SOUND)
    try:
        with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    add(f"en:{p[0]}", f"fr:{p[1]}", 0.95, "=", MEANING)
    except FileNotFoundError:
        pass

    nodes = sorted(edges)
    words = [n.split(":", 1)[1] for n in nodes]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = np.asarray(model.encode(words, batch_size=256,
                                   normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)
    print(f"semantic kNN over {len(nodes)} nodes...", file=sys.stderr)
    B = 2000
    for i0 in range(0, len(nodes), B):
        sims = vecs[i0:i0 + B] @ vecs.T
        for r in range(sims.shape[0]):
            row = sims[r]
            idx = np.argpartition(-row, 7)[:7]
            for j in idx:
                if j == i0 + r or row[j] < 0.60:
                    continue
                edges[nodes[i0 + r]].append((nodes[j], float(row[j]), "~", MEANING))
    sem_neigh = defaultdict(set)
    for n, lst in edges.items():
        for m, q, lab, fam in lst:
            if fam == MEANING:
                sem_neigh[n].add(m)
    return edges, sem_neigh


def explore(edges, sem_neigh, seed):
    start = f"en:{seed}"
    home = {start} | sem_neigh.get(start, set())
    chains, loops = [], []
    # state: (node, last_family, path_nodes, path_str, qualities, sound_hops)
    frontier = [(start, None, {start}, start, [], 0)]
    for _depth in range(MAX_HOPS):
        nxt = []
        for node, last, seen, pstr, quals, sh in frontier:
            cands = [c for c in edges.get(node, []) if c[3] != last]
            cands.sort(key=lambda c: -c[1])
            for m, q, lab, fam in cands[:BEAM_PER_NODE]:
                nsh = sh + (1 if fam == SOUND else 0)
                npstr = f"{pstr} {lab} {m}"
                nquals = quals + [q]
                if m in home and m != start and nsh >= MIN_SOUND_HOPS:
                    loops.append((sum(nquals) / len(nquals), len(nquals), npstr))
                if m in seen:
                    continue
                if m.startswith("en:") and nsh >= MIN_SOUND_HOPS and len(nquals) >= 4:
                    chains.append((sum(nquals) / len(nquals), len(nquals), npstr))
                nxt.append((m, fam, seen | {m}, npstr, nquals, nsh))
        nxt.sort(key=lambda s: -(sum(s[4]) / max(1, len(s[4]))))
        frontier = nxt[:4000]
    chains.sort(key=lambda c: (-c[1], -c[0]))
    loops.sort(key=lambda c: (-c[1], -c[0]))
    return chains, loops


def main():
    seeds = sys.argv[1:] or ["key", "frog", "sea", "bear"]
    edges, sem_neigh = build_graph()
    out = []
    for s in seeds:
        chains, loops = explore(edges, sem_neigh, s)
        out.append(f"\n=== {s} ===")
        out.append("  longest alternation chains:")
        for q, ln, p in chains[:4]:
            out.append(f"    [{ln} hops, q {q:.2f}] {p}")
        out.append("  loops back home ('= key' endings):")
        if not loops:
            out.append("    (none within depth budget)")
        for q, ln, p in loops[:3]:
            out.append(f"    [{ln} hops, q {q:.2f}] {p}")
    text = "\n".join(out)
    print(text)
    with open("chain-game-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
