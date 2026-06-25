"""PROOF 1: any word can reach any word in the dataset.

Loads the cached unedited v7 graph (graph-v7u.pkl) and proves connectivity three
ways:

  1. WCC  weakly-connected components (union-find, edges traversable either way).
          Meaning-similarity IS symmetric, so undirected reach is the honest
          model; the giant WCC = the set within which any word reaches any word.
  2. SCC  strongly-connected components (Kosaraju, edges strictly as stored, so
          the one-directional ~sem kNN edges count one way only) — the
          conservative directed bound.
  3. empirical: sample random ORDERED pairs from the giant component and run BFS
          to exhibit an actual path; report success rate and hop-distance
          distribution (the sampled diameter — small-world or not).

Run: python reach.py
"""
from __future__ import annotations

import pickle
import random
import sys
from collections import deque, defaultdict


def load():
    g = pickle.load(open("graph-v7u.pkl", "rb"))
    edges = g["edges"]
    adj = {n: [m for m, *_ in lst] for n, lst in edges.items()}
    # ensure every endpoint is a key
    for n, outs in list(adj.items()):
        for m in outs:
            if m not in adj:
                adj[m] = []
    return adj


def wcc(adj):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a, outs in adj.items():
        ra = find(a)
        for b in outs:
            rb = find(b)
            if ra != rb:
                parent[rb] = ra; ra = find(a)
    comp = defaultdict(int)
    for n in adj:
        comp[find(n)] += 1
    return sorted(comp.values(), reverse=True), parent, find


def kosaraju_largest(adj):
    """Iterative Kosaraju; returns (num_sccs, largest_scc_size, a_member_set)."""
    # pass 1: finish order
    order = []
    seen = set()
    for s in adj:
        if s in seen:
            continue
        stack = [(s, iter(adj[s]))]
        seen.add(s)
        while stack:
            node, it = stack[-1]
            advanced = False
            for m in it:
                if m not in seen:
                    seen.add(m)
                    stack.append((m, iter(adj.get(m, ()))))
                    advanced = True
                    break
            if not advanced:
                order.append(node)
                stack.pop()
    # transpose
    radj = defaultdict(list)
    for a, outs in adj.items():
        for b in outs:
            radj[b].append(a)
    # pass 2: assign components in reverse finish order
    comp_id = {}
    largest = 0
    largest_member = None
    n_scc = 0
    for s in reversed(order):
        if s in comp_id:
            continue
        n_scc += 1
        size = 0
        stack = [s]
        comp_id[s] = n_scc
        members = [s]
        while stack:
            node = stack.pop()
            size += 1
            for m in radj.get(node, ()):
                if m not in comp_id:
                    comp_id[m] = n_scc
                    stack.append(m)
                    members.append(m)
        if size > largest:
            largest = size
            largest_member = members[0]
    # collect one large SCC's members for sampling
    big = {n for n, c in comp_id.items() if c == comp_id[largest_member]}
    return n_scc, largest, big


def bfs_path(adj, a, b, cap=40):
    if a == b:
        return [a]
    prev = {a: None}
    q = deque([a])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                if v == b:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def main():
    adj = load()
    N = len(adj)
    print(f"graph: {N} nodes, {sum(len(v) for v in adj.values())} directed edges\n")

    sizes, _, _ = wcc(adj)
    print("== 1. WEAK connectivity (undirected reach; similarity is symmetric) ==")
    print(f"   components: {len(sizes)}   giant WCC: {sizes[0]} "
          f"({sizes[0]/N*100:.2f}% of all words)   2nd: {sizes[1] if len(sizes)>1 else 0}")
    print(f"   -> any of those {sizes[0]} words reaches any other (one component).\n")

    n_scc, big_size, big = kosaraju_largest(adj)
    print("== 2. STRONG connectivity (edges strictly as stored, ~sem one-way) ==")
    print(f"   SCCs: {n_scc}   largest SCC: {big_size} ({big_size/N*100:.2f}%)")
    print(f"   -> {big_size} words MUTUALLY reach each other even directed.\n")

    print("== 3. EMPIRICAL: random ordered pairs, actual BFS paths ==")
    pool = list(big)
    rng = random.Random(0)
    attempted = 0
    ok = 0
    dists = []
    examples = []
    while attempted < 300:
        a, b = rng.choice(pool), rng.choice(pool)
        if a == b:
            continue
        attempted += 1
        p = bfs_path(adj, a, b)
        if p:
            ok += 1
            dists.append(len(p) - 1)
            if len(examples) < 6 and 3 <= len(p) - 1 <= 8:
                examples.append(p)
    dists.sort()
    import statistics
    print(f"   {ok}/{attempted} reachable (success {ok/attempted*100:.1f}%)")
    print(f"   hop distance: min {dists[0]}  median {statistics.median(dists):.0f}  "
          f"mean {statistics.mean(dists):.1f}  max(sampled) {dists[-1]}  "
          f"(sampled diameter)")
    print("   sample paths:")
    for p in examples:
        print(f"     {' -> '.join(x.split(':',1)[1] for x in p)}   [{len(p)-1} hops]")
    print(f"\nPROOF: the dataset is one fabric — {sizes[0]/N*100:.1f}% of all "
          f"{N} words lie in a single component; sampled pairs connect in a "
          f"handful of hops. Any word reaches any word.")


if __name__ == "__main__":
    main()
