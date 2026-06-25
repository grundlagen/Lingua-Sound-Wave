"""Unified hop routing + ideal hop distance for the translation engine.

Two ideas the engine needs:

1. UNIFIED routing: at every hop take WHICHEVER edge is shortest -- homophone
   (≈), translation (=) or synonym (~) -- not forced alternation. `hop A B`
   returns the fewest-hop typed path.

2. IDEAL HOP DISTANCE: in a homophonic translation, every intermediate hop is a
   WRITTEN word, so chain length = output inflation. We measure the shortest
   chain from an English word to a French word that MEANS it (a real MUSE
   translation), over sound+synonym edges and requiring >=1 SOUND hop (that is
   what makes the output sound like the source). The distribution of that length
   is the per-word inflation; French-anchoring should shrink it.

Optionally folds in extra sound edges (fr-anchored-pairs.tsv) to show the effect.

Run: python hoproute.py hop key argile
     python hoproute.py ideal            [--anchor fr-anchored-pairs.tsv]
"""
from __future__ import annotations

import pickle
import random
import sys
from collections import deque

SOUND, MEANING = "S", "M"
LAB = {"≈": "snd", "=": "trans", "~": "syn"}


def load(anchor=None):
    g = pickle.load(open("graph-v7u.pkl", "rb"))
    edges = g["edges"]
    if anchor:
        n = 0
        with open(anchor, encoding="utf-8") as f:
            next(f)
            for line in f:
                fr, en, combo, tier = line.rstrip("\n").split("\t")
                a, b = f"en:{en}", f"fr:{fr}"
                edges.setdefault(a, []).append((b, float(combo), "≈", SOUND))
                edges.setdefault(b, []).append((a, float(combo), "≈", SOUND))
                n += 1
        print(f"folded in {n} French-anchored sound edges", file=sys.stderr)
    return edges


def resolve(edges, w):
    if w in edges:
        return w
    for p in ("en:", "fr:"):
        if p + w in edges:
            return p + w
    raise SystemExit(f"'{w}' not in web")


def hop(edges, a, b, max_hops=14):
    """Fewest-hop path over ANY edge type (unified)."""
    src, dst = resolve(edges, a), resolve(edges, b)
    prev = {src: (None, None)}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for m, qy, lab, fam in edges.get(u, []):
            if m not in prev:
                prev[m] = (u, lab)
                q.append(m)
    if dst not in prev:
        return None
    path = []
    cur = dst
    while cur is not None:
        p, lab = prev[cur]
        path.append((cur, lab))
        cur = p
    return path[::-1]


def fmt(path):
    if not path:
        return "(no path)"
    out = [path[0][0]]
    for node, lab in path[1:]:
        out.append(f"{lab} {node}")
    return f"[{len(path)-1} hops] " + " ".join(out)


def homophonic_adj(edges):
    """sound + synonym edges only (translation '=' excluded: that is the answer
    we want to reach homophonically, not a free hop)."""
    adj = {}
    for n, lst in edges.items():
        keep = [(m, lab, fam) for m, q, lab, fam in lst if lab in ("≈", "~")]
        if keep:
            adj[n] = keep
    return adj


def ideal_distance(edges, sample=500, cap=10, loose=False):
    """Shortest chain en:W -> fr:translation over sound+syn, >=1 sound hop.
    loose=True also accepts landing on a SYNONYM of the translation (a French
    word that still means the source) -- closer to what the engine accepts."""
    adj = homophonic_adj(edges)
    # synonym closure for loose targets
    syn = {}
    if loose:
        for n, lst in edges.items():
            syn[n] = {m for m, q, lab, fam in lst if lab == "~"}
    # MUSE translation targets: en -> {fr,...}
    tgt = {}
    with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                tgt.setdefault(f"en:{p[0]}", set()).add(f"fr:{p[1]}")
    cands = [e for e in tgt if e in adj]
    rng = random.Random(3)
    rng.shuffle(cands)
    dists, reached, n = [], 0, 0
    for src in cands:
        if n >= sample:
            break
        targets = {t for t in tgt[src] if t in adj}
        if loose:
            for t in list(targets):
                targets |= {s for s in syn.get(t, ()) if s.startswith("fr:")}
        if not targets:
            continue
        n += 1
        # BFS over states (node, sound_used); accept target with sound_used
        start = (src, False)
        seen = {start}
        q = deque([(src, False, 0)])
        found = None
        while q:
            node, su, d = q.popleft()
            if d >= cap:
                continue
            for m, lab, fam in adj.get(node, ()):
                nsu = su or (fam == SOUND)
                if m in targets and nsu:
                    found = d + 1
                    break
                st = (m, nsu)
                if st not in seen:
                    seen.add(st)
                    q.append((m, nsu, d + 1))
            if found:
                break
        if found:
            reached += 1
            dists.append(found)
    dists.sort()
    import statistics
    if not dists:
        print("no homophonic translation paths found"); return
    pct = lambda k: sum(1 for d in dists if d <= k) / n * 100
    print(f"sampled {n} EN words with a French translation in the web")
    print(f"  homophonic-route reachable (<= {cap} hops, >=1 sound): "
          f"{reached}/{n} = {reached/n*100:.0f}%")
    print(f"  ideal hop distance: min {dists[0]} median {statistics.median(dists):.0f} "
          f"mean {statistics.mean(dists):.1f} max {dists[-1]}")
    print(f"  reachable in <=2: {pct(2):.0f}%   <=3: {pct(3):.0f}%   "
          f"<=4: {pct(4):.0f}%   <=5: {pct(5):.0f}%")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    anchor = None
    if "--anchor" in sys.argv:
        anchor = sys.argv[sys.argv.index("--anchor") + 1]
    edges = load(anchor)
    if cmd == "hop":
        print(fmt(hop(edges, sys.argv[2], sys.argv[3])))
    elif cmd == "ideal":
        ideal_distance(edges, loose="--loose" in sys.argv)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
