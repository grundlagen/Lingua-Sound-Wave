"""Build a candidate cross-lingual homophone dictionary at scale, and measure
how many *legitimate* word-to-word entries we can actually expect.

This is the real feasibility test behind the "un œuf / enough" question:
classification AUC says nothing about whether a usable dictionary exists.
Retrieval over a real lexicon does.

Architecture (also the right way to build this for real):
  1. phonemize the top-N English and French words once (real lexical units,
     so no forced phrases like "un œuf");
  2. BLOCK: cheap phoneme-bigram Dice shortlists ~K French candidates per
     English word (fast set ops over precomputed bigram sets);
  3. RANK: the full `combo` matcher rescat the shortlist;
  4. KEEP the best match per English word, bucketed into quality tiers.

Tiers encode the user's "full laurels" intuition:
  S  score >= 0.90  near-identical, both real words  -> gold dictionary entry
  A  0.78–0.90      strong, usable
  B  0.62–0.78      loose / punning, needs a human
  -  < 0.62         rejected (noise)

Run: python build_dictionary.py [n_en] [n_fr]
"""
from __future__ import annotations

import json
import sys

from wordfreq import top_n_list

from matcher import g2p, _segs, _canonical, _ngram_channel, _feat_channel

S, A, B = 0.90, 0.78, 0.62
SHORTLIST = 25


def bigrams(ipa: str) -> set:
    s = ("#",) + _segs(_canonical(ipa)) + ("#",)
    return {s[i] + s[i + 1] for i in range(len(s) - 1)}


def combo(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * _feat_channel(qi, ci)


def main():
    n_en = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n_fr = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    en = [w for w in top_n_list("en", n_en) if w.isalpha() and len(w) > 1]
    fr = [w for w in top_n_list("fr", n_fr) if w.isalpha() and len(w) > 1]

    fr_data = []
    for w in fr:
        ipa = g2p(w, "fr")
        fr_data.append((w, ipa, bigrams(ipa)))
    print(f"EN queries: {len(en)}   FR lexicon: {len(fr_data)}", file=sys.stderr)

    entries = []
    for w in en:
        qi = g2p(w, "en")
        qb = bigrams(qi)
        if not qb:
            continue
        # block: cheap Dice shortlist
        shortlisted = sorted(
            fr_data,
            key=lambda c: (len(qb & c[2]) / (len(qb) + len(c[2]))) if c[2] else 0,
            reverse=True,
        )[:SHORTLIST]
        # rank: full combo
        best_s, best_w, best_ci = 0.0, None, None
        for cw, ci, _cb in shortlisted:
            if cw == w:  # identical spelling (loanwords) — skip trivial
                continue
            s = combo(qi, ci)
            if s > best_s:
                best_s, best_w, best_ci = s, cw, ci
        if best_w is not None and best_s >= B:
            tier = "S" if best_s >= S else "A" if best_s >= A else "B"
            entries.append({"en": w, "fr": best_w, "score": round(best_s, 3),
                            "tier": tier, "en_ipa": qi, "fr_ipa": best_ci})

    entries.sort(key=lambda e: -e["score"])
    counts = {t: sum(1 for e in entries if e["tier"] == t) for t in "SAB"}
    print(f"\n=== Candidate dictionary from {len(en)} English words ===")
    print(f"  S (>= {S}, gold): {counts['S']}")
    print(f"  A ({A}-{S}):       {counts['A']}")
    print(f"  B ({B}-{A}):       {counts['B']}")
    print(f"  yield (S+A) per 100 EN words: {100*(counts['S']+counts['A'])/len(en):.1f}")

    print("\n--- Top 40 S/A entries ---")
    for e in [e for e in entries if e["tier"] in "SA"][:40]:
        print(f"  {e['tier']} {e['score']:.2f}  {e['en']:12s} ~ {e['fr']:14s}"
              f"  [{e['en_ipa']} | {e['fr_ipa']}]")

    with open("dictionary-sample.json", "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"\nwrote dictionary-sample.json ({len(entries)} entries)")


if __name__ == "__main__":
    main()
