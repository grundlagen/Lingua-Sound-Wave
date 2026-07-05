"""Chain-routed dual translation: every word transfers to French through an
alternation chain, not a direct lookup.

The user's architecture: transfer capacity comes from chain complexity.
A word's French rendering should be reachable through interleaved
sound/meaning moves (key = touche ≈ douche...), so the output carries BOTH
channels by construction — each chosen French word is connected to its
English source by at least one sound hop AND at least one meaning hop
("transfer chains"). Direct single-edge matches are allowed only as
fallback when no transfer chain exists.

Per content word:
  1. explore alternation chains from en:<word> (same engine as chain_game);
  2. collect French endpoints whose chain used >=1 sound and >=1 meaning
     move; score = chain quality (mean edge quality), tie-broken by chain
     LENGTH (longer = more transfer, per the user's instruction);
  3. choose the best endpoint; render the line; report every chain.

Usage: python chain_translate.py "the sea is cold" ...
Output: chain-translate-demo.txt
"""
from __future__ import annotations

import sys

from wordfreq import zipf_frequency

from chain_game import build_graph, SOUND, MEANING, MAX_HOPS, BEAM_PER_NODE

STOPWORDS = {"the", "a", "an", "is", "are", "was", "of", "and", "or", "in",
             "on", "at", "to", "it", "its"}


def fr_endpoints(edges, seed, max_hops=MAX_HOPS):
    """Alternation-chain search; returns fr endpoints with chain stats."""
    start = f"en:{seed}"
    found = {}
    frontier = [(start, None, {start}, start, [], 0, 0)]
    for _ in range(max_hops):
        nxt = []
        for node, last, seen, pstr, quals, sh, mh in frontier:
            cands = [c for c in edges.get(node, []) if c[3] != last]
            cands.sort(key=lambda c: -c[1])
            for m, q, lab, fam in cands[:BEAM_PER_NODE]:
                if m in seen:
                    continue
                nsh = sh + (1 if fam == SOUND else 0)
                nmh = mh + (1 if fam == MEANING else 0)
                npstr = f"{pstr} {lab} {m}"
                nquals = quals + [q]
                if m.startswith("fr:") and nsh >= 1 and nmh >= 1 \
                        and zipf_frequency(m[3:], "fr") >= 3.0:
                    word = m[3:]
                    quality = sum(nquals) / len(nquals)
                    prev = found.get(word)
                    # prefer longer chains at comparable quality (transfer!)
                    rank = (len(nquals), quality)
                    if prev is None or rank > prev[0]:
                        found[word] = (rank, quality, len(nquals), npstr)
                nxt.append((m, fam, seen | {m}, npstr, nquals, nsh, nmh))
        nxt.sort(key=lambda s: -(sum(s[4]) / max(1, len(s[4]))))
        frontier = nxt[:3000]
    return found


def main():
    sentences = sys.argv[1:] or ["the sea is cold", "two men under the moon"]
    edges, _sem = build_graph()
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    out = []
    for sent in sentences:
        words = [w.lower().strip(".,!?") for w in sent.split()]
        line, details = [], []
        for w in words:
            if w in STOPWORDS:
                line.append("·")
                continue
            eps = fr_endpoints(edges, w)
            good = [(rank, q, ln, p, fw) for fw, (rank, q, ln, p) in eps.items()
                    if q >= 0.80]
            if good:
                # anchor: endpoint must still relate to the source word in
                # meaning — chains are the route, not a license to drift
                texts = [w] + [g[4] for g in good]
                vecs = np.asarray(model.encode(texts, normalize_embeddings=True,
                                               show_progress_bar=False))
                anchored = []
                for g, v in zip(good, vecs[1:]):
                    anchor = max(0.0, float(np.dot(vecs[0], v)))
                    transfer = g[2] * g[1] * (0.25 + 0.75 * anchor)
                    anchored.append((transfer, anchor) + g)
                anchored.sort(key=lambda a: -a[0])
                good = [a[2:] for a in anchored]
                anchors = {a[6]: a[1] for a in anchored}
            if not good:
                # fallback: best direct sound edge
                direct = [(q, m[3:], f"en:{w} {lab} {m}")
                          for m, q, lab, fam in edges.get(f"en:{w}", [])
                          if fam == SOUND and m.startswith("fr:")]
                if direct:
                    q, fw, p = max(direct)
                    line.append(fw)
                    details.append(f"    {w} -> {fw}  [direct sound only, q {q:.2f}]")
                else:
                    line.append(f"[{w}]")
                    details.append(f"    {w} -> (no chain, no direct match)")
                continue
            _rank, q, ln, p, fw = good[0]
            line.append(fw)
            details.append(f"    {w} -> {fw}  [{ln} hops, q {q:.2f}, "
                           f"anchor {anchors.get(fw, 0):.2f}]")
            details.append(f"        {p}")
        out.append(f"\nEN: {sent}")
        out.append(f"FR: {' '.join(line)}")
        out.extend(details)

    text = "\n".join(out)
    print(text)
    with open("chain-translate-demo.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
