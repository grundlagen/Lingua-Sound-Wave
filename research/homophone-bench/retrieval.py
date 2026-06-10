"""The real test: can we BUILD a dictionary, not just classify given pairs?

Classification (bench.py) scores a pre-selected pair. A dictionary is the
inverse, much harder problem: given one word, search a whole real lexicon and
return only legitimate, good-sounding matches. This measures that directly.

For each English query we rank the top-N French words (by corpus frequency,
so candidates are real lexical units) with the winning `combo` matcher, then
inspect the top of the list: is the known-good homophone near the top, and
how much garbage rides above the useful threshold?

Run: python retrieval.py
Needs: espeak-ng, panphon, numpy, wordfreq.
"""
from __future__ import annotations

import sys
from wordfreq import top_n_list

from matcher import g2p, _ngram_channel, _feat_channel

# English queries with a documented "ideal" French homophone (or None if the
# point is just to see what the lexicon offers).
QUERIES = [
    ("shoe", "chou"), ("key", "qui"), ("knee", "ni"), ("sea", "si"),
    ("bow", "beau"), ("dough", "dos"), ("low", "leau"), ("sue", "sous"),
    ("mare", "mere"), ("bell", "belle"), ("ray", "raie"), ("say", "ses"),
    ("more", "mort"), ("two", "tout"), ("vee", "vie"), ("pan", "paon"),
    ("ant", "an"), ("on", "on"), ("fee", "fit"), ("coal", "colle"),
]

THRESHOLD = 0.45  # the bench.py decision threshold


def score(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat_channel(qi, ci)


def main():
    n_cand = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    fr_lex = [w for w in top_n_list("fr", n_cand) if w.isalpha() and len(w) > 1]
    # precompute candidate IPA once
    fr_ipa = [(w, g2p(w, "fr")) for w in fr_lex]
    print(f"Lexicon: {len(fr_ipa)} real French words (top {n_cand} by frequency)\n")

    precision_at_1 = 0
    ideal_found_rank = []
    total_above_thr = []

    for q, ideal in QUERIES:
        qi = g2p(q, "en")
        scored = sorted(((score(qi, ci), w, ci) for w, ci in fr_ipa), reverse=True)
        top = scored[:5]
        above = [s for s in scored if s[0] >= THRESHOLD]
        total_above_thr.append(len(above))

        # where does the documented ideal land?
        rank = next((i for i, (s, w, ci) in enumerate(scored) if w == ideal), None)
        ideal_s = next((s for s, w, ci in scored if w == ideal), None)
        if top and top[0][1] == ideal:
            precision_at_1 += 1
        if rank is not None:
            ideal_found_rank.append(rank)

        toplist = ", ".join(f"{w}({s:.2f})" for s, w, ci in top)
        idtxt = (f"ideal {ideal!r}: rank {rank}, score {ideal_s:.2f}"
                 if rank is not None else f"ideal {ideal!r}: NOT in lexicon")
        print(f"{q!r:8s} [{qi}]")
        print(f"    top5: {toplist}")
        print(f"    {idtxt}   | {len(above)} words >= {THRESHOLD}")

    n = len(QUERIES)
    print(f"\n=== Dictionary-readiness ===")
    print(f"  ideal is #1 match:        {precision_at_1}/{n}")
    if ideal_found_rank:
        import statistics
        print(f"  median rank of ideal:     {int(statistics.median(ideal_found_rank))}")
    print(f"  mean candidates >= {THRESHOLD}:   {sum(total_above_thr)/n:.1f} per query")
    print(f"  (every word >= {THRESHOLD} would become a dictionary entry — most are noise)")


if __name__ == "__main__":
    main()
