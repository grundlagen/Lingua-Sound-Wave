"""An alternative to v5's word-for-word selection: re-rank each headword's FR
choice by the matcher (combo) with a frequency tiebreak.

Why: v5 picks the "best" FR per English headword by its BUILD-TIME score, which
saturates -- `wee` has both `ouïe` and `oui` at 1.0, `sea` has six candidates all
at 1.0 -- so among ties v5 chooses arbitrarily and sometimes lands on the rarer/
worse-sounding word (wee->ouïe, not oui). The fix needs no new data: v5 already
CONTAINS the better candidate. Re-rank each headword's existing FR candidates by

    key = (round(combo, 3) desc, zipf_frequency(fr) desc)

i.e. best sound first, then -- among sound-ties -- the more natural (frequent)
French word. combo is computed from v5's OWN stored IPAs (deterministic, no
espeak), the same two channels as matcher.homophone_score.

Outputs dictionary-v5-reranked.tsv (the alternative word-for-word table) + a diff
report. Does NOT mutate dictionary-v5.json.

Run: python rerank_v5.py
"""
from __future__ import annotations

import json
from collections import defaultdict

from wordfreq import zipf_frequency

import matcher
from matcher import _canonical


def combo_from_ipa(en_ipa: str, fr_ipa: str) -> float:
    a = _canonical(en_ipa or "")
    b = _canonical((fr_ipa or "").replace(" ", ""))
    if not a or not b:
        return 0.0
    ng = matcher._ngram_channel(a, b)
    ft = matcher._feat_channel(a, b)
    return 0.5 * ng + 0.5 * ft


def fr_freq(fr: str) -> float:
    """Naturalness tiebreak: min word zipf for a phrase (a phrase is only as
    natural as its rarest word), single zipf for one word."""
    ws = fr.split()
    return min((zipf_frequency(w, "fr") for w in ws), default=0.0)


def main():
    d = json.load(open("dictionary-v5.json"))
    by = defaultdict(list)
    for e in d:
        if e.get("direction", "en_fr") != "en_fr":
            continue
        by[e["en"]].append(e)

    reranked = {}        # en -> (fr, combo, zipf, v5_fr, changed)
    swaps = []
    combo_v5_sum = combo_rr_sum = 0.0
    n = 0
    for en, cands in by.items():
        # v5's pick: highest build-time score (first among ties = arbitrary)
        v5_pick = max(cands, key=lambda e: e.get("score", 0.0))
        # re-rank by (combo, freq) over the SAME candidates
        scored = []
        for c in cands:
            cb = combo_from_ipa(c.get("en_ipa", ""), c.get("fr_ipa", ""))
            scored.append((round(cb, 3), fr_freq(c["fr"]), cb, c))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        rr = scored[0][3]
        rr_combo = scored[0][2]
        v5_combo = combo_from_ipa(v5_pick.get("en_ipa", ""), v5_pick.get("fr_ipa", ""))
        combo_v5_sum += v5_combo
        combo_rr_sum += rr_combo
        n += 1
        changed = rr["fr"] != v5_pick["fr"]
        reranked[en] = (rr["fr"], rr_combo, scored[0][1], v5_pick["fr"], changed)
        if changed:
            swaps.append((en, v5_pick["fr"], round(v5_combo, 2),
                          rr["fr"], round(rr_combo, 2), scored[0][1]))

    # write the alternative table
    with open("dictionary-v5-reranked.tsv", "w", encoding="utf-8") as f:
        f.write("en\tfr_reranked\tcombo\tfr_zipf\tv5_fr\tchanged\n")
        for en in sorted(reranked):
            fr, cb, z, v5fr, ch = reranked[en]
            f.write(f"{en}\t{fr}\t{cb:.3f}\t{z:.2f}\t{v5fr}\t{int(ch)}\n")

    print(f"headwords re-ranked: {n}")
    print(f"selection changed on: {len(swaps)} ({100*len(swaps)//n}%)")
    print(f"mean combo  v5: {combo_v5_sum/n:.4f}  ->  reranked: {combo_rr_sum/n:.4f}"
          f"  ({combo_rr_sum/n - combo_v5_sum/n:+.4f})")
    print("\nsample swaps (v5 pick -> reranked pick):")
    swaps.sort(key=lambda s: (s[4] - s[2]), reverse=True)  # biggest combo gain first
    print(f"  {'en':10s} {'v5 (combo)':22s} {'reranked (combo, zipf)':28s}")
    for en, v5fr, v5c, rrfr, rrc, z in swaps[:18]:
        print(f"  {en:10s} {v5fr+f' ({v5c})':22s} {rrfr+f' ({rrc}, z{z:.1f})':28s}")

    # validate on the documented cases the v5 test flagged
    print("\nvalidate against documented homophones (from dataset.py):")
    from dataset import POS_WORDS
    doc = dict(POS_WORDS)
    fixed = same = 0
    for en in ["wee", "sea", "knee", "key", "shoe", "day", "bell"]:
        if en in reranked and en in doc:
            fr, cb, z, v5fr, ch = reranked[en]
            hit = "== documented" if fr == doc[en] else f"(doc: {doc[en]})"
            print(f"  {en:6s} v5={v5fr:8s} -> reranked={fr:8s} {hit}")
            if fr == doc[en] and v5fr != doc[en]:
                fixed += 1
            elif fr == v5fr:
                same += 1
    print(f"\nwrote dictionary-v5-reranked.tsv "
          f"({len(swaps)} improved word-for-word picks, no new data).")


if __name__ == "__main__":
    main()
