"""Advanced set matching: chain English sets to French sets through the
sound+meaning web, with multi-hop paths and optimal assignment.

The user's idea: matching need not be direct word<->word. Given an English
pool {key, bear, tie, frog} and a French pool, each English word can reach
the French set through CHAINS — sound hop, then meaning hop, then sound hop
("key ≈ qui", "clos ≈ clothes", "froc ≈ frog") — and the larger the pools,
the more complicated paths become available.

Graph (three edge families, all graded):
  sound     usable v5 entries           cost = 1 - phonetic score
  trans     MUSE translations           cost = 0.10
  sem       embedding kNN neighbors     cost = 1 - cosine  (within AND
            across languages — lets a chain slide to a synonym before
            sounding, which is what makes long chains land)

Search: Dijkstra from each EN-set member, <= MAX_HOPS, no two consecutive
sem hops (keeps chains honest — sound must keep re-anchoring them).
Assignment: Hungarian algorithm over path costs -> one best FR partner per
EN word, plus the full chain provenance for every pair.

Usage:
  python set_match.py key bear tie frog --vs clos froc cle ours
  python set_match.py key bear tie frog            (FR pool = whole web)
"""
from __future__ import annotations

import heapq
import json
import sys
from collections import defaultdict

import numpy as np

MAX_HOPS = 5
SEM_MIN = 0.55
SEM_K = 6


def build_graph():
    entries = json.load(open("dictionary-v5.json"))
    edges = defaultdict(list)   # node -> [(node, cost, label)]

    def add(a, b, cost, label):
        edges[a].append((b, cost, label))
        edges[b].append((a, cost, label))

    for e in entries:
        if e.get("usable_for_composition"):
            add(f"en:{e['en']}", f"fr:{e['fr']}", 1 - e["score"], "≈snd")

    try:
        with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    add(f"en:{p[0]}", f"fr:{p[1]}", 0.10, "=trans")
    except FileNotFoundError:
        pass

    # graded semantic edges over the web vocabulary
    nodes = sorted(edges)
    words = [n.split(":", 1)[1] for n in nodes]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = model.encode(words, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    print(f"semantic kNN over {len(nodes)} nodes...", file=sys.stderr)
    # blockwise kNN to bound memory
    B = 2000
    for i0 in range(0, len(nodes), B):
        sims = vecs[i0:i0 + B] @ vecs.T
        for r in range(sims.shape[0]):
            row = sims[r]
            idx = np.argpartition(-row, SEM_K + 1)[:SEM_K + 1]
            for j in idx:
                if j == i0 + r or row[j] < SEM_MIN:
                    continue
                edges[nodes[i0 + r]].append((nodes[j], float(1 - row[j]), "~sem"))
    print(f"graph: {len(edges)} nodes", file=sys.stderr)
    return edges


def chains_from(edges, start, targets=None):
    """Dijkstra with hop limit; no two consecutive sem hops."""
    best = {}
    pq = [(0.0, start, 0, None, [start])]
    seen = {}
    while pq:
        cost, node, hops, last, path = heapq.heappop(pq)
        key = (node, last == "~sem")
        if seen.get(key, 1e9) <= cost:
            continue
        seen[key] = cost
        if node != start and node.startswith("fr:"):
            if targets is None or node[3:] in targets:
                if node not in best or cost < best[node][0]:
                    best[node] = (cost, path)
        if hops >= MAX_HOPS:
            continue
        for nxt, c, label in edges.get(node, []):
            if label == "~sem" and last == "~sem":
                continue
            if nxt in path:
                continue
            heapq.heappush(pq, (cost + c, nxt, hops + 1, label,
                                path + [f"{label} {nxt}"]))
    return best


def main():
    args = sys.argv[1:]
    if "--vs" in args:
        i = args.index("--vs")
        en_set, fr_set = args[:i], set(args[i + 1:])
    else:
        en_set, fr_set = args or ["key", "bear", "tie", "frog"], None

    edges = build_graph()
    all_paths = {}
    for w in en_set:
        all_paths[w] = chains_from(edges, f"en:{w}", fr_set)
        print(f"  chained en:{w} -> {len(all_paths[w])} french endpoints",
              file=sys.stderr)

    # candidate FR pool = union of reachable endpoints (or the given set)
    pool = sorted(fr_set) if fr_set else sorted(
        {n[3:] for p in all_paths.values() for n in p})[:400]

    # assignment: Hungarian over path costs
    INF = 99.0
    cost = np.full((len(en_set), len(pool)), INF)
    for i, w in enumerate(en_set):
        for j, f in enumerate(pool):
            hit = all_paths[w].get(f"fr:{f}")
            if hit:
                cost[i, j] = hit[0]
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(cost)

    lines = [f"EN set: {en_set}",
             f"FR pool: {'given ' + str(sorted(fr_set)) if fr_set else f'open (web), {len(pool)} reachable'}",
             ""]
    for i, j in zip(ri, ci):
        if cost[i, j] >= INF:
            lines.append(f"  {en_set[i]:10s} -> (no chain found)")
            continue
        c, path = all_paths[en_set[i]][f"fr:{pool[j]}"]
        lines.append(f"  {en_set[i]:10s} -> {pool[j]:14s} cost {c:.2f}")
        lines.append(f"      chain: {' '.join(path)}")
    text = "\n".join(lines)
    print(text)
    with open("set-match-demo.txt", "a") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()
