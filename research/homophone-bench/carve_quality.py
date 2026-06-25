"""Push carve quality: measure which levers actually raise the homophone score.

Carve combo tops out ~0.5 on whole lines. We test, on a fixed line set, the
levers that could raise it -- each measured by mean best-combo (the matcher's
homophone score) and mean coverage of the English sound stream:

  baseline    min_zipf 2.0, beam 350, top_n 40, word_penalty 0.04, lm 0.35
  pool+       richer French unit pool (min_zipf 1.3 -> more carve options)
  search+     wider beam (900) + more candidates (top_n 150) to re-rank by combo
  sound1st    de-weight fluency in the beam (lm 0.10) so the search chases SOUND,
              then re-rank survivors by combo (sound-first, fluency as tiebreak)
  ALL         pool+ ∪ search+ ∪ sound1st

Run: python carve_quality.py
"""
from __future__ import annotations

import matcher
import phonetic_decoder as pd
import poetry_mode as pm
import whole_line_carve as wlc

LINES = [
    "Humpty Dumpty sat on a wall",
    "Hickory dickory dock",
    "Jack and Jill went up the hill",
    "twinkle twinkle little star",
    "adored companion in the",
    "evenings may be so",
]


def carve(line, root, beam, top_n, lm_weight, wp, extra=3):
    pd.BEAM = beam
    pd.WORD_PENALTY = wp
    pd.MIN_WORD_SEGS = 1
    ipa = wlc.en_ipa(line)
    nwords = len(line.split())
    cands = pd.decode(ipa, root, top_n=top_n, max_words=nwords + extra,
                      lm=wlc.LM, lm_weight=lm_weight)
    best = None
    for c in cands:
        if c["coverage"] < 0.70:
            continue
        combo = matcher.homophone_score(line, "en", c["fr"], "fr")["score"]
        coh = wlc.coherence(c["fr"])
        if best is None or combo > best[0]:
            best = (combo, coh, c["coverage"], c["fr"])
    return best


def run(name, root, beam, top_n, lm_weight, wp):
    combos, covs, examples = [], [], []
    for line in LINES:
        b = carve(line, root, beam, top_n, lm_weight, wp)
        if b:
            combos.append(b[0]); covs.append(b[2])
            examples.append((line, b[0], b[2], b[3]))
        else:
            combos.append(0.0); covs.append(0.0)
    mc = sum(combos) / len(combos)
    mv = sum(covs) / len(covs)
    print(f"\n== {name} ==  mean combo {mc:.3f}  mean coverage {mv:.3f}")
    for line, c, v, fr in examples:
        print(f"   {c:.2f}/{v:.2f}  {line!r} -> {fr}")
    return mc, mv


def main():
    wlc.force_coverage()
    print("building tries...", flush=True)
    trie_std = pm.build_poetry_trie(min_zipf=2.0)
    trie_rich = pm.build_poetry_trie(min_zipf=1.3)

    results = {}
    results["baseline"] = run("baseline", trie_std, 350, 40, 0.35, 0.04)
    results["pool+"] = run("pool+ (min_zipf 1.3)", trie_rich, 350, 40, 0.35, 0.04)
    results["search+"] = run("search+ (beam 900, top_n 150)", trie_std, 900, 150, 0.35, 0.04)
    results["sound1st"] = run("sound1st (lm 0.10)", trie_std, 900, 150, 0.10, 0.04)
    results["ALL"] = run("ALL (rich pool + wide + sound1st)", trie_rich, 900, 150, 0.10, 0.04)

    print("\n=== summary (mean combo, mean coverage) ===")
    base = results["baseline"][0]
    for k, (mc, mv) in results.items():
        print(f"  {k:10s} combo {mc:.3f} ({mc-base:+.3f})  coverage {mv:.3f}")
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\nbest: {best[0]} at combo {best[1][0]:.3f} "
          f"(+{best[1][0]-base:.3f} over baseline)")


if __name__ == "__main__":
    main()
