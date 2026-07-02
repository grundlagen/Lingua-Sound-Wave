"""DUAL mining at scale: homophonic AND literal translation AT ONCE.

The vision: pairs (and later lines) where the French IS the translation AND
sounds like the English. Meaning is guaranteed by construction (MUSE bilingual
dictionary = literal translations); we score SOUND over all 113k translation
pairs with the rule-aware combo, and keep a tiered ladder:

    DUAL-S  sound >= 0.75   (say it, hear the English; mean it, read the French)
    DUAL-A  sound >= 0.60
    DUAL-B  sound >= 0.45

Cognates will dominate the top (chance/chance) -- they're kept but FLAGGED
(edit distance) so the non-cognate art tier is separable. Synonym-relaxed pairs
(EN synonym's translation, meaning 0.7) widen the net.

Out: dual-pairs.tsv  (en, fr, sound, meaning, cognate, tier)

Run: python dual_mine.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher

import matcher

MUSE = "/tmp/muse-en-fr.txt"
SYN = "muse-pivot-syn.tsv"


def cognate(en: str, fr: str) -> bool:
    return SequenceMatcher(None, en.lower(), fr.lower()).ratio() >= 0.75


def combo(en: str, fr: str) -> float:
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


def tier(s: float) -> str:
    return "DUAL-S" if s >= 0.75 else ("DUAL-A" if s >= 0.60 else
                                       ("DUAL-B" if s >= 0.45 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pairs = []
    seen = set()
    for line in open(MUSE, encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and p[0].isalpha() and p[1].replace("-", "").isalpha():
            k = (p[0].lower(), p[1].lower())
            if k not in seen:
                seen.add(k)
                pairs.append((k[0], k[1], 1.0))          # literal translation
    # synonym-relaxed: en A ~syn en B, B ->fr T  =>  (A, T, 0.7)
    syn = {}
    for line in open(SYN, encoding="utf-8"):
        a, b, _ = line.rstrip("\n").split("\t")
        if a.startswith("en:") and b.startswith("en:"):
            syn.setdefault(a[3:], set()).add(b[3:])
            syn.setdefault(b[3:], set()).add(a[3:])
    trans = {}
    for en, fr, _ in pairs:
        trans.setdefault(en, set()).add(fr)
    for a, bs in syn.items():
        for b in bs:
            for fr in trans.get(b, ()):
                k = (a, fr)
                if k not in seen:
                    seen.add(k)
                    pairs.append((a, fr, 0.7))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"scoring {len(pairs)} literal/near-literal translation pairs for sound",
          file=sys.stderr)

    kept = 0
    with open("dual-pairs.tsv", "w", encoding="utf-8") as f:
        f.write("en\tfr\tsound\tmeaning\tcognate\ttier\n")
        for i, (en, fr, meaning) in enumerate(pairs):
            s = combo(en, fr)
            t = tier(s)
            if t:
                f.write(f"{en}\t{fr}\t{s:.3f}\t{meaning:.1f}\t{int(cognate(en, fr))}\t{t}\n")
                kept += 1
            if i % 5000 == 0:
                print(f"  {i}/{len(pairs)}  kept {kept}", file=sys.stderr)
    print(f"done: {kept} dual pairs -> dual-pairs.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
