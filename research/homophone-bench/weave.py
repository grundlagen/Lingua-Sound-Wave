"""Weave: loop all chains into one interconnected transfer web.

Phase 1 (--stats): connectivity census of the full graph (sound + trans +
semantic-kNN edges). Union-find components answer "are we one web or many
islands": giant component size, isolates, degree hubs.

Phase 2 (--full): for EVERY English dictionary headword (en node with at
least one sound edge), run the bounded alternation-chain search and record
its best transfer endpoints (>=1 sound hop AND >=1 meaning hop, endpoint
zipf >= 3.0, junk-free). Each endpoint is itself another chain's start, so
the recorded edges ARE the chains looped to each other: the chain-web.
Loops (chains returning to the seed's own semantic neighborhood) are
recorded separately.

Outputs:
  chain-web-stats.json   census numbers (phase 1 + phase 2 totals)
  chain-web.tsv          src, dst, hops, quality, chain (top 3 per seed)
  chain-loops.tsv        seeds whose chains come home
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

from wordfreq import zipf_frequency

from chain_game import build_graph, SOUND, MEANING

MAX_HOPS = 6
BEAM_PER_NODE = 6
FRONTIER = 1200


def components(edges):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, lst in edges.items():
        for b, *_ in lst:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comps = defaultdict(int)
    for n in edges:
        comps[find(n)] += 1
    sizes = sorted(comps.values(), reverse=True)
    return sizes


def endpoints(edges, sem_neigh, seed):
    start = f"en:{seed}"
    home = {start} | sem_neigh.get(start, set())
    found, loops = {}, []
    frontier = [(start, None, {start}, start, [], 0, 0)]
    for _ in range(MAX_HOPS):
        nxt = []
        for node, last, seen, pstr, quals, sh, mh in frontier:
            cands = [c for c in edges.get(node, []) if c[3] != last]
            cands.sort(key=lambda c: -c[1])
            for m, q, lab, fam in cands[:BEAM_PER_NODE]:
                if m in seen:
                    continue
                nsh = sh + (fam == SOUND)
                nmh = mh + (fam == MEANING)
                nq = quals + [q]
                np_ = f"{pstr} {lab} {m}"
                if m in home and nsh >= 2:
                    loops.append((sum(nq) / len(nq), len(nq), np_))
                if (m.startswith("fr:") and nsh >= 1 and nmh >= 1
                        and zipf_frequency(m[3:], "fr") >= 3.0):
                    quality = sum(nq) / len(nq)
                    prev = found.get(m[3:])
                    if prev is None or (len(nq), quality) > (prev[0], prev[1]):
                        found[m[3:]] = (len(nq), quality, np_)
                nxt.append((m, fam, seen | {m}, np_, nq, nsh, nmh))
        nxt.sort(key=lambda s: -(sum(s[4]) / max(1, len(s[4]))))
        frontier = nxt[:FRONTIER]
    ranked = sorted(found.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    return ranked[:3], loops[:2]


def main():
    t0 = time.time()
    edges, sem_neigh = build_graph()

    sizes = components(edges)
    n_nodes = len(edges)
    stats = {
        "nodes": n_nodes,
        "components": len(sizes),
        "giant_component": sizes[0],
        "giant_fraction": round(sizes[0] / n_nodes, 4),
        "second_component": sizes[1] if len(sizes) > 1 else 0,
        "isolates_or_tiny(<=3)": sum(1 for s in sizes if s <= 3),
    }
    print(f"PHASE1 census: {json.dumps(stats)}", flush=True)
    with open("chain-web-stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    if "--stats" in sys.argv:
        return

    seeds = sorted({n[3:] for n, lst in edges.items()
                    if n.startswith("en:") and any(c[3] == SOUND for c in lst)})
    print(f"PHASE2 weaving {len(seeds)} seeds...", flush=True)
    n_edges = n_loops = 0
    with open("chain-web.tsv", "w") as fw, open("chain-loops.tsv", "w") as fl:
        fw.write("src\tdst\thops\tquality\tchain\n")
        fl.write("seed\thops\tquality\tloop\n")
        for i, s in enumerate(seeds):
            eps, loops = endpoints(edges, sem_neigh, s)
            for dst, (ln, q, p) in eps:
                fw.write(f"{s}\t{dst}\t{ln}\t{q:.3f}\t{p}\n")
                n_edges += 1
            for q, ln, p in loops:
                fl.write(f"{s}\t{ln}\t{q:.3f}\t{p}\n")
                n_loops += 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(seeds)} seeds, {n_edges} web edges, "
                      f"{n_loops} loops ({time.time() - t0:.0f}s)", flush=True)

    stats.update({"seeds": len(seeds), "chain_web_edges": n_edges,
                  "loops": n_loops, "seconds": round(time.time() - t0)})
    with open("chain-web-stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"DONE {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
