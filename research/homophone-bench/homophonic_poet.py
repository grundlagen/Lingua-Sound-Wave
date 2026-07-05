"""homophonic_poet -- GENERATIVE Van Rooten: the output really is a homophone.

The fix to web_poet: a graph walk mixing sound+meaning hops does NOT read aloud
as one sound. Real homophony = CARVE a target phoneme stream into French words so
the French, spoken, reconstructs the English. So:

  1. GENERATE a themed English source line (beam over English bigrams pulled
     toward a seed theme vector) -- this is what the piece MEANS / sounds like.
  2. CARVE its phoneme stream into French (whole_line_carve: boundaries fall on
     French words, fillers carry the meter) -- the French SOUNDS like the English.
  3. VERIFY -- report combo(EN line, FR carve): the matcher's homophone score, the
     proof that it is actually a homophone, plus coverage of the sound stream.

We generate many English sources, carve each, and keep the ones whose French is
both a strong homophone (high combo, high coverage) and decent French (coherence).

Run: python homophonic_poet.py sea night love
"""
from __future__ import annotations

import json
import sys
from heapq import nlargest

import numpy as np

import bigram_lm
import poetry_mode as pm
import whole_line_carve as wlc
import matcher


def themed_english(seed_vec, EN, vecs, ids, idx, valid_en, n_lines=18, length=4):
    """beam: build short English lines pulled toward the theme and fluent."""
    starts = nlargest(40, valid_en, key=lambda w: float(vecs[idx["en:" + w]] @ seed_vec))
    beams = [(float(vecs[idx["en:" + w]] @ seed_vec), [w]) for w in starts[:24]]
    for _ in range(length - 1):
        nxt = []
        for sc, words in beams:
            prev = words[-1]
            cands = []
            for w in starts + nlargest(60, valid_en,
                    key=lambda x: EN.cond(prev.lower(), x.lower())):
                if w in words:
                    continue
                th = float(vecs[idx["en:" + w]] @ seed_vec)
                step = (EN.cond(prev.lower(), w.lower()) + 1e-4) * (th + 0.3)
                cands.append((sc + np.log(step), words + [w]))
            nxt.extend(nlargest(3, cands, key=lambda x: x[0]))
        beams = nlargest(n_lines, nxt, key=lambda x: x[0])
    seen, out = set(), []
    for sc, words in beams:
        key = " ".join(words)
        if key not in seen:
            seen.add(key); out.append(key)
    return out


def main():
    seeds = sys.argv[1:] or ["sea", "night", "love"]
    # carve environment (same as whole_line_carve.main)
    import phonetic_decoder as pd
    pd.MIN_WORD_SEGS = 1
    pd.WORD_PENALTY = 0.04
    wlc.force_coverage()
    root = pm.build_poetry_trie(min_zipf=2.0)

    EN = bigram_lm.load("en")
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    from wordfreq import zipf_frequency
    valid_en = [n[3:] for n in ids if n.startswith("en:") and n[3:].isalpha()
                and 2 <= len(n[3:]) <= 11 and zipf_frequency(n[3:], "en") >= 3.3]

    for s in seeds:
        if "en:" + s not in idx:
            continue
        theme = vecs[idx["en:" + s]]
        lines = themed_english(theme, EN, vecs, ids, idx, valid_en)
        results = []
        for en_line in lines:
            ipa, nw, scored = wlc.carve_line(en_line, root, beam=420)
            if scored:
                dual, combo, coh, cov, fr, nfr, nf = scored[0]
                # homophone proof = combo of the WHOLE en line vs the fr carve
                results.append((combo * cov * (coh + 0.1), combo, coh, cov,
                                en_line, fr, ipa))
        results.sort(key=lambda x: -x[0])
        print(f"\n{'='*72}\nTHEME: {s}\n{'='*72}")
        for score, combo, coh, cov, en_line, fr, ipa in results[:3]:
            print(f"\n  EN (means, on theme): {en_line}")
            print(f"  FR (homophone)      : {fr}")
            print(f"  >> sounds-like proof: combo {combo:.2f}  coverage {cov:.2f}"
                  f"   french-fluency {coh:.2f}")
            print(f"     EN sound [{ipa}]")
        print("\n  (the FR line, spoken, reconstructs the EN sound -- combo is the "
              "matcher's homophone score; that is the project's core, verified.)")


if __name__ == "__main__":
    main()
