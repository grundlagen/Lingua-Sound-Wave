"""Van Rooten joint search: ONE French line that SOUNDS like the English and is
coherent French -- letting the ENGLISH SOURCE DRIFT semantically until a clean
homophonic stream settles.

This is the synonym-swap layer generation_engine.py flagged but never built.
Pipeline:
  1. DRIFT  paraphrase the English source by swapping content words for their
            embedding-nearest English synonyms (friend->mate/pal, sea->ocean/water).
  2. CARVE  for each drifted English line, run whole_line_carve: decode its whole
            phoneme stream into French words (boundaries fall on French words),
            ranked by sound(combo) x French coherence.
  3. SETTLE keep the single best (drift, carve) pair by joint = sound x coherence.

Output is ONE French line + the English paraphrase it sounds like -- not two.

Run: python vanrooten.py "friend loves the sea"
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

import poetry_mode as pm
import whole_line_carve as wlc

STOP = {"the", "a", "an", "is", "are", "on", "in", "of", "and", "to", "it",
        "that", "this", "with", "for", "at", "as", "be", "was", "were"}


def load_syn():
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    from wordfreq import zipf_frequency
    en_ok = np.array([n.startswith("en:") and " " not in n and
                      zipf_frequency(n[3:], "en") >= 2.8 for n in ids])
    return vecs, ids, idx, en_ok


def synonyms(word, ctx, k=3):
    vecs, ids, idx, en_ok = ctx
    n = f"en:{word}"
    if n not in idx:
        return [word]
    v = vecs[idx[n]]
    sims = vecs @ v
    sims[~en_ok] = -1
    order = np.argpartition(-sims, 40)[:40]
    order = order[np.argsort(-sims[order])]
    out = [word]
    for j in order:
        w = ids[int(j)][3:]
        if sims[j] < 0.45:
            break
        if w != word and w not in out and not w.startswith(word[:3]):
            out.append(w)
        if len(out) > k:
            break
    return out


def drifts(words, ctx, max_cand=20):
    opts = [[w] if w in STOP else synonyms(w, ctx, k=2) for w in words]
    cands = []
    for combo in itertools.product(*opts):
        cands.append(" ".join(combo))
        if len(cands) >= max_cand:
            break
    # ensure the original is first
    orig = " ".join(words)
    if orig in cands:
        cands.remove(orig)
    return [orig] + cands


def main():
    sent = sys.argv[1] if len(sys.argv) > 1 else "friend loves the sea"
    import phonetic_decoder as pd
    pd.MIN_WORD_SEGS = 1
    pd.WORD_PENALTY = 0.04
    wlc.force_coverage()
    root = pm.build_poetry_trie(min_zipf=2.0)
    ctx = load_syn()

    words = [w.strip(".,!?;:").lower() for w in sent.split() if w.strip(".,!?;:")]
    cands = drifts(words, ctx)
    print(f'SOURCE: "{sent}"   ({len(cands)} semantic drifts tried)\n')

    best = []
    for en_line in cands:
        ipa, nw, scored = wlc.carve_line(en_line, root, beam=420)
        if scored:
            dual, combo, coh, cov, fr, nfr, nf = scored[0]
            best.append((dual, combo, coh, cov, en_line, fr))
    best.sort(reverse=True)

    print("=== best homophonic-semantic settlements (one stream each) ===")
    for dual, combo, coh, cov, en_line, fr in best[:6]:
        print(f"  joint {dual:.2f} (sound {combo:.2f} x frenchness {coh:.2f}, "
              f"cov {cov:.2f})")
        print(f"     sounds like : {en_line}")
        print(f"     FRENCH      : {fr}")
    if best:
        d, cm, ch, cv, en_line, fr = best[0]
        print(f'\nSETTLED: "{fr}"')
        print(f'  (read aloud it sounds like the English drift "{en_line}")')
    else:
        print("no carve settled; loosen coverage or grow the French unit pool.")


if __name__ == "__main__":
    main()
