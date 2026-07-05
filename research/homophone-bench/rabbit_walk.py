"""Round Rabbit Fix 1, given a go: min-hop BFS vs best-PRODUCT Dijkstra.

DEPS_RABBIT_AND_NOVEL.md Fix 1 claims round_rabbit.py's bfs_component keeps the
fewest-HOPS path to each node and only folds edge quality into the rank
afterwards -- so it can commit to a weak short route over a strong slightly
longer one, then dock it by -0.08*hops. The fix is to keep the best-PRODUCT
path (Dijkstra over -log(edge_score)), which is the WFST shortest-path.

This rebuilds a homophone sound-graph from dictionary-v5 (no dependency on the
other branch's mapping-web.json) and measures, over many seeds, how often and by
how much best-product beats min-hop on the SAME reachable nodes. A graph fact,
not an opinion.

  node  = "en:word" | "fr:word"
  edge  = entry (en <-> fr) weighted by its homophone score
  hop en->fr->en' means en and en' share a French homophone (the rabbit's hop)

Run: python rabbit_walk.py
"""
from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict, deque


def load_graph(path="dictionary-v5.json"):
    entries = json.load(open(path, encoding="utf-8"))
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in entries:
        if not e.get("usable_for_composition"):
            continue
        sc = float(e.get("score", 0.0))
        if sc < 0.5:
            continue
        en, fr = f"en:{e['en']}", f"fr:{e['fr']}"
        adj[en].append((fr, sc))
        adj[fr].append((en, sc))
        # phrase tokens: en <-> each fr word, so the rabbit can tunnel through
        if " " in e["fr"]:
            for tok in e["fr"].split():
                adj[en].append((f"fr:{tok}", sc * 0.9))
                adj[f"fr:{tok}"].append((en, sc * 0.9))
    for n in adj:
        adj[n].sort(key=lambda x: -x[1])
    return adj


def bfs_minhops(adj, seed, max_hops):
    """round_rabbit's policy: keep the fewest-hop path; record its edge product."""
    best: dict[str, tuple[int, float]] = {seed: (0, 1.0)}
    q = deque([(seed, 0, 1.0)])
    while q:
        node, hops, prod = q.popleft()
        if hops >= max_hops:
            continue
        for nxt, w in adj.get(node, [])[:80]:
            if nxt in best and best[nxt][0] <= hops + 1:
                continue            # already reached in <= hops; KEEP min-hop path
            best[nxt] = (hops + 1, prod * w)
            q.append((nxt, hops + 1, prod * w))
    return best


def dijkstra_bestproduct(adj, seed, max_hops):
    """Fix: keep the maximum edge-PRODUCT path within the hop budget."""
    best: dict[str, tuple[float, int]] = {seed: (1.0, 0)}
    # max-heap on product via negative
    pq = [(-1.0, 0, seed)]
    while pq:
        negp, hops, node = heapq.heappop(pq)
        prod = -negp
        if prod < best.get(node, (0, 0))[0] - 1e-12:
            continue
        if hops >= max_hops:
            continue
        for nxt, w in adj.get(node, [])[:80]:
            np_ = prod * w
            if np_ > best.get(nxt, (0.0, 99))[0] + 1e-12:
                best[nxt] = (np_, hops + 1)
                heapq.heappush(pq, (-np_, hops + 1, nxt))
    return best


def main():
    adj = load_graph()
    print(f"sound-graph: {len(adj)} nodes, "
          f"{sum(len(v) for v in adj.values())//2} edges")

    # seeds: well-connected English nodes
    seeds = sorted((n for n in adj if n.startswith("en:") and len(adj[n]) >= 3),
                   key=lambda n: -len(adj[n]))[:200]
    max_hops = 3

    improved = 0
    total_cmp = 0
    gain_sum = 0.0
    examples = []
    for seed in seeds:
        b = bfs_minhops(adj, seed, max_hops)
        d = dijkstra_bestproduct(adj, seed, max_hops)
        for node in d:
            if node == seed or node not in b:
                continue
            bp = b[node][1]          # bfs path product
            dp = d[node][0]          # dijkstra best product
            total_cmp += 1
            if dp > bp + 1e-9:
                improved += 1
                gain_sum += dp - bp
                if dp - bp > 0.15 and len(examples) < 12:
                    examples.append((seed, node, b[node], d[node]))

    print(f"\ncompared {total_cmp} (seed,node) reachable pairs over {len(seeds)} seeds")
    print(f"best-product strictly beats min-hop on {improved} "
          f"({100*improved/max(1,total_cmp):.1f}%)")
    print(f"mean path-quality gain where it differs: "
          f"{gain_sum/max(1,improved):.3f} (product units)\n")
    print("examples (min-hop product, hops) -> (best product, hops):")
    for seed, node, (bh, bp), (dp, dh) in examples:
        print(f"  {seed:18s} -> {node:20s}  bfs {bp:.2f}@{bh}h  "
              f"dijkstra {dp:.2f}@{dh}h")
    print("\nReading: every % here is a route round_rabbit currently ranks on a")
    print("weaker path than exists. Fix 1 (best-product = WFST shortest-path)")
    print("recovers them, which is exactly where deep fragment tunnels live.")


if __name__ == "__main__":
    main()
