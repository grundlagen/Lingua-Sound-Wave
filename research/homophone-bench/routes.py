"""Q2: understand NODES and ROUTES over the whole v7 web (sound + meaning).

Loads the cached unedited graph (graph-v7u.pkl: every node -> [(node, quality,
label, family)], family S=sound M=meaning) and answers three things about ANY
word in the 195k-node system:

  route  A  B   best alternation path A->B (Dijkstra on -log quality).
                --any drops the alternation rule (shortest by quality, any edges).
  node   W      connectivity profile: degree by family, top sound + meaning
                neighbours, k-hop reach (how much of the web W can touch), and
                hub centrality (how often W is an interchange in recorded chains).
  hubs          the global interchanges: nodes that sit on the most chains.

Routing maximises the product of edge qualities = minimises sum of -log(q).
With alternation (default) the family must flip each hop (the telephone-game
rule that makes chains surprising); --any allows any edge.

Run:  python routes.py route key argile
      python routes.py node key
      python routes.py hubs
"""
from __future__ import annotations

import heapq
import math
import pickle
import sys
from collections import defaultdict, Counter

GRAPH = "graph-v7u.pkl"
FULLWEB = "chain-web-full-v7u.tsv"   # for hub centrality (chain appearances)

SOUND, MEANING = "S", "M"
FAMNAME = {SOUND: "sound", MEANING: "meaning"}


def load():
    g = pickle.load(open(GRAPH, "rb"))
    return g["edges"]


def resolve(edges, w):
    """word -> existing node id, trying en: then fr:."""
    if ":" in w and w in edges:
        return w
    for pre in ("en:", "fr:"):
        if pre + w in edges:
            return pre + w
    raise SystemExit(f"'{w}' not in the web (try en:{w} / fr:{w})")


def route(edges, a, b, alternate=True, max_hops=12):
    src, dst = resolve(edges, a), resolve(edges, b)
    # state = (node, last_family); cost = sum(-log q); track path
    start = (src, None)
    pq = [(0.0, 0, src, None, (src,), (None,))]   # cost, hops, node, lastfam, nodes, labels
    best = {start: 0.0}
    while pq:
        cost, hops, node, lastfam, nodes, labs = heapq.heappop(pq)
        if node == dst and hops > 0:
            return cost, nodes, labs
        if hops >= max_hops:
            continue
        for m, q, lab, fam in edges.get(node, []):
            if alternate and fam == lastfam:
                continue
            if m in nodes:
                continue
            nc = cost - math.log(max(q, 1e-6))
            st = (m, fam)
            if nc < best.get(st, 1e18):
                best[st] = nc
                heapq.heappush(pq, (nc, hops + 1, m, fam, nodes + (m,), labs + (lab,)))
    return None


def fmt_path(cost, nodes, labs):
    parts = [nodes[0]]
    for i in range(1, len(nodes)):
        parts.append(f"{labs[i]} {nodes[i]}")
    q = math.exp(-cost / max(1, len(nodes) - 1))   # geo-mean edge quality
    return f"[{len(nodes)-1} hops, mean-q {q:.2f}]  " + " ".join(parts)


def node_profile(edges, w):
    n = resolve(edges, w)
    lst = edges.get(n, [])
    snd = [(m, q, lab) for m, q, lab, fam in lst if fam == SOUND]
    mng = [(m, q, lab) for m, q, lab, fam in lst if fam == MEANING]
    snd.sort(key=lambda t: -t[1]); mng.sort(key=lambda t: -t[1])
    print(f"\n=== {n} ===")
    print(f"  degree: {len(lst)}  ({len(snd)} sound, {len(mng)} meaning)")
    print("  top SOUND neighbours (homophones):")
    for m, q, lab in snd[:8]:
        print(f"     {lab} {m:18s} q{q:.2f}")
    print("  top MEANING neighbours (translation / synonym):")
    for m, q, lab in mng[:8]:
        print(f"     {lab} {m:18s} q{q:.2f}")
    # k-hop reach: how much of the web this word can touch
    seen = {n}; frontier = {n}
    for k in range(1, 5):
        nxt = set()
        for u in frontier:
            for m, *_ in edges.get(u, []):
                if m not in seen:
                    seen.add(m); nxt.add(m)
        frontier = nxt
        print(f"  reach <= {k} hops: {len(seen)-1} nodes")
        if not frontier:
            break


def hub_centrality(top=25, profile_for=None):
    """Interchange score = how many recorded chains a node lies on."""
    cnt = Counter()
    try:
        f = open(FULLWEB, encoding="utf-8")
    except FileNotFoundError:
        print("(chain-web-full not present yet; centrality needs it)"); return cnt
    next(f)
    for line in f:
        a, b, hops, q, sub = line.rstrip("\n").split("\t")
        for tok in sub.split():
            if tok.startswith(("en:", "fr:")):
                cnt[tok] += 1
    f.close()
    if profile_for is not None:
        node = profile_for if ":" in profile_for else None
        # report rank of the given node
        ranked = [n for n, _ in cnt.most_common()]
        for cand in ([profile_for] if node else [f"en:{profile_for}", f"fr:{profile_for}"]):
            if cand in cnt:
                print(f"  hub centrality: {cand} appears on {cnt[cand]} chains "
                      f"(rank {ranked.index(cand)+1} of {len(ranked)})")
        return cnt
    print(f"\n=== top {top} interchange hubs (most chains pass through) ===")
    for n, c in cnt.most_common(top):
        print(f"  {c:5d}  {n}")
    return cnt


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "hubs":
        hub_centrality()
        return
    edges = load()
    if cmd == "route":
        a, b = sys.argv[2], sys.argv[3]
        anyedge = "--any" in sys.argv
        r = route(edges, a, b, alternate=not anyedge)
        kind = "any-edge" if anyedge else "strict-alternation"
        if r:
            print(f"\n{a} -> {b}  ({kind}):\n  " + fmt_path(*r))
        else:
            print(f"\n{a} -> {b}: no path within budget ({kind})")
    elif cmd == "node":
        node_profile(edges, sys.argv[2])
        hub_centrality(profile_for=sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
