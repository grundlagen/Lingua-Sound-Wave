"""Sentence-level DUAL translation: literal AND homophonic at once, per word,
fueled by dual-pairs.tsv (dual_mine.py over all MUSE literal translations).

Per English word: pick the French that is its TRANSLATION and its best
HOMOPHONE, tier fallback DUAL-S -> DUAL-A -> DUAL-B -> [literal] (bracketed =
translation only, no homophony). Seams scored with the FR bigram LM; --juncture
adds the cross-word sandhi lift. --nocognate drops flagged cognates first.

Run: python dual_sentence.py "the sea is deep" [--nocognate]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import matcher

try:
    import bigram_lm
    _FR = bigram_lm.load("fr")
except Exception:
    _FR = None


def load_dual(path="dual-pairs.tsv"):
    d = defaultdict(list)      # en -> [(sound, cognate, fr, tier)]
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i == 0:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6:
            d[p[0]].append((float(p[2]), int(p[4]), p[1], p[5]))
    for v in d.values():
        v.sort(key=lambda t: (-t[0], t[1]))
    return d


def load_literal():
    lit = {}
    for line in open("/tmp/muse-en-fr.txt", encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and p[0] not in lit:
            lit[p[0]] = p[1]
    return lit


def pick(word, dual, lit, nocognate):
    cands = dual.get(word, [])
    if nocognate:
        nc = [c for c in cands if not c[1]]
        cands = nc or cands
    if cands:
        s, cog, fr, tier = cands[0]
        return fr, tier + ("*" if cog else ""), s
    if word in lit:
        return f"[{lit[word]}]", "literal", 0.0
    return f"[{word}]", "miss", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--nocognate", action="store_true")
    ap.add_argument("--juncture", action="store_true")
    args = ap.parse_args()
    dual = load_dual()
    lit = load_literal()
    print(f"(dual lexicon: {len(dual)} EN words with a translation-homophone)\n")
    for sent in args.text:
        words = [w.strip(".,!?;").lower() for w in sent.split()]
        out, tags, sounds = [], [], []
        for w in words:
            fr, tier, s = pick(w, dual, lit, args.nocognate)
            out.append(fr); tags.append(tier); sounds.append(s)
        fr_line = " ".join(out)
        line_sound = sum(sounds) / max(1, len(sounds))
        if args.juncture:
            try:
                import juncture
                js = juncture.best_juncture_score(sent, fr_line.replace("[", "").replace("]", ""))
                line_sound = max(line_sound, js)
            except Exception:
                pass
        flu = _FR.fluency([w.strip("[]").lower() for w in out]) if _FR else 0.0
        print(f"EN : {sent}")
        print(f"FR : {fr_line}")
        print(f"     tiers: {' '.join(tags)}   mean-sound {line_sound:.2f}"
              + (f"   fr-fluency {flu:.4f}" if _FR else ""))
        print()


if __name__ == "__main__":
    main()
