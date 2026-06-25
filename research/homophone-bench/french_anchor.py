"""French-anchor the homophone layer: symmetric to the English-anchored dataset.

The v7 dataset is EN-headword -> FR-homophone, so the FR sound side is sparse
(only French words that happen to sound English). Here we run the SAME AUC-0.993
matcher in REVERSE: for the most frequent FRENCH words, retrieve the best ENGLISH
homophone over the English lexicon (bigram prefilter -> sharpened featural combo).
Result = FR-anchored sound pairs that densify the French side.

Output: fr-anchored-pairs.tsv  (fr, en, combo, tier)   tier S = combo>=0.90
Run from research/homophone-bench: python french_anchor.py
"""
from __future__ import annotations

import sys
from wordfreq import top_n_list

import bench

EN_POOL_N = 8000      # English candidate lexicon
FR_QUERY_N = 4000     # French headwords to anchor
PREFILTER = 70
THRESH = 0.80


def bigrams(ipa):
    segs, _ = bench.segs_and_vecs(bench.canonical(ipa))
    pad = ("#",) + tuple(segs) + ("#",)
    return {pad[i] + pad[i + 1] for i in range(len(pad) - 1)}


def pcombo(ipa_a, lang_a, ipa_b, lang_b):
    ga, gb = bigrams(ipa_a), bigrams(ipa_b)
    dice = 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0
    va, vb = bench.variants(ipa_a, lang_a), bench.variants(ipa_b, lang_b)
    feat = max(bench._nw_sim(x, y, sharpen=True) for x in va for y in vb)
    return 0.5 * dice + 0.5 * feat


def main():
    print("encoding English candidate lexicon...", file=sys.stderr)
    en_cand = []
    for w in top_n_list("en", EN_POOL_N):
        if not (w.isalpha() and 2 <= len(w) <= 13):
            continue
        try:
            ip = bench.g2p_ipa(w, "en")
        except Exception:
            continue
        bg = bigrams(ip)
        if bg:
            en_cand.append((w, ip, bg))
    print(f"  {len(en_cand)} EN candidates", file=sys.stderr)

    fr_words = [w for w in top_n_list("fr", FR_QUERY_N)
                if w.isalpha() and 2 <= len(w) <= 13]
    rows = []
    for i, fw in enumerate(fr_words):
        try:
            fi = bench.g2p_ipa(fw, "fr")
        except Exception:
            continue
        fg = bigrams(fi)
        if not fg:
            continue
        ranked = sorted(en_cand,
                        key=lambda t: -(len(fg & t[2]) / (len(fg) + len(t[2]))))
        best, bestw = 0.0, None
        for ew, ei, _ in ranked[:PREFILTER]:
            s = pcombo(fi, "fr", ei, "en")
            if s > best:
                best, bestw = s, ew
        if best >= THRESH and bestw:
            rows.append((fw, bestw, best))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(fr_words)} fr words, {len(rows)} anchored",
                  file=sys.stderr)

    rows.sort(key=lambda r: -r[2])
    with open("fr-anchored-pairs.tsv", "w", encoding="utf-8") as f:
        f.write("fr\ten\tcombo\ttier\n")
        for fw, ew, s in rows:
            f.write(f"{fw}\t{ew}\t{s:.3f}\t{'S' if s >= 0.90 else 'A'}\n")
    s_tier = sum(1 for r in rows if r[2] >= 0.90)
    print(f"\nanchored {len(rows)}/{len(fr_words)} French words "
          f"({len(rows)/len(fr_words)*100:.1f}%) to an English homophone "
          f">= {THRESH}; {s_tier} at S-tier (>=0.90).")
    print("top:")
    for fw, ew, s in rows[:15]:
        print(f"   {fw:14s} ~sounds~ {ew:14s}  {s:.2f}")


if __name__ == "__main__":
    main()
