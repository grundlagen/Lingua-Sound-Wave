"""Round Rabbit, branch-native: semantic anchor -> homophonic radius lattice,
run on this branch's mapping-web.json (sound + meaning adjacency) with the
best-PRODUCT walk (Fix 1 from DEPS_RABBIT_AND_NOVEL.md), not min-hop BFS.

Idea (round_rabbit): start from a MEANING, walk outward through homophonic
(sound) edges, and collect every bilingual string that sits N sound-hops away.
The result is a themed homophone field: "say something near this meaning that
also sounds like something else."

Walk: Dijkstra in the product semiring over sound-edge weights (best confidence
path, not fewest hops). At each reached node we attach the v5 homophone strings.

Usage: python round_rabbit_run.py "moon star night" "king queen" ...
"""
from __future__ import annotations

import heapq
import json
import sys
from collections import defaultdict


def load():
    g = json.load(open("mapping-web.json", encoding="utf-8"))
    return g["sound"], g["meaning"]


def attachments():
    d = json.load(open("dictionary-v5.json", encoding="utf-8"))
    out = defaultdict(list)
    for e in d:
        if e.get("direction", "en_fr") != "en_fr":
            continue
        out[f"en:{e['en']}"].append((e["en"], e["fr"], e.get("score", 0)))
        out[f"fr:{e['fr']}"].append((e["en"], e["fr"], e.get("score", 0)))
    return out


def best_product_walk(seeds, sound, max_hops=3, frontier=4000):
    """Max edge-product path from any seed node, within max_hops."""
    best = {}                      # node -> (product, hops, path)
    pq = []
    for s in seeds:
        best[s] = (1.0, 0, [s])
        heapq.heappush(pq, (-1.0, 0, s, [s]))
    while pq and len(best) < frontier:
        negp, hops, node, path = heapq.heappop(pq)
        prod = -negp
        if prod < best.get(node, (0,))[0] - 1e-12:
            continue
        if hops >= max_hops:
            continue
        for edge in sound.get(node, [])[:40]:
            nxt, w = edge[0], float(edge[1])
            np_ = prod * w
            if np_ > best.get(nxt, (0.0,))[0] + 1e-12:
                best[nxt] = (np_, hops + 1, path + [nxt])
                heapq.heappush(pq, (-np_, hops + 1, nxt, path + [nxt]))
    return best


def main():
    seeds_in = sys.argv[1:] or ["moon star night", "king queen", "water hill"]
    sound, meaning = load()
    att = attachments()

    for theme in seeds_in:
        words = theme.split()
        # semantic neighbourhood: the seed words + their meaning-adjacent nodes
        sem_nodes = set()
        for w in words:
            for pre in (f"en:{w}", f"fr:{w}"):
                if pre in meaning or pre in sound:
                    sem_nodes.add(pre)
                for nb in meaning.get(pre, []):
                    sem_nodes.add(nb)
        sem_nodes = {n for n in sem_nodes if n in sound}     # must have a sound edge
        if not sem_nodes:
            print(f"\n=== {theme!r}: no sound-connected meaning anchor ===")
            continue
        reached = best_product_walk(sem_nodes, sound, max_hops=3)

        rows = []
        for node, (prod, hops, path) in reached.items():
            for en, fr, sc in att.get(node, []):
                rows.append((prod * sc, prod, hops, en, fr))
        rows.sort(reverse=True)
        seen = set()
        print(f"\n=== {theme!r}: {len(sem_nodes)} meaning anchors -> "
              f"{len(reached)} nodes in homophonic radius ===")
        shown = 0
        for score, prod, hops, en, fr in rows:
            if (en, fr) in seen:
                continue
            seen.add((en, fr))
            print(f"   {hops} hop  path-prod {prod:.2f}   EN: {en:18s} | FR: {fr}")
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    main()
