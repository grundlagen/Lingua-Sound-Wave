#!/usr/bin/env python3
"""Quick test with real French LM."""
import subprocess, matcher, poetry_mode as pm, phonetic_decoder as pd, whole_line_carve as wlc
from lexicon_g2p import clean_ipa
from bigram_lm import BigramLM, load as load_lm

lm = load_lm("fr")
print(f"LM: {lm.N:,} tokens")

root = pm.build_poetry_trie(min_zipf=2.0)
wlc.force_coverage()

lines = ["Humpty Dumpty","Humpty Dumpty sat on a wall","Hickory dickory dock","Rain rain go away",
         "Mary had a little lamb","Jack and Jill went up the hill","the cat and the fiddle"]

for line in lines:
    pd.WORD_PENALTY = 0.0; pd.MIN_WORD_SEGS = 1; pd.BEAM = 400
    ipa,nw,scored = wlc.carve_line(line, root, beam=400)
    if scored:
        best_s,best_fr = -1,""
        for dual,combo,coh,cov,fr,nfr,nf in scored:
            toks = [t.lower() for t in fr.replace("'"," ").split() if t]
            flu = lm.fluency(toks)
            fill = sum(1 for t in toks if t in pm.FILLER)
            s = combo*(0.3+0.7*flu)+0.03*fill
            if s>best_s: best_s,best_fr = s,fr
        print(f"EN: {line}")
        print(f"FR: {best_fr}  combined={best_s:.3f}")
        for dual,combo,coh,cov,fr,nfr,nf in scored[:3]:
            toks2 = [t.lower() for t in fr.replace("'"," ").split() if t]
            f2 = lm.fluency(toks2)
            fill2 = sum(1 for t in toks2 if t in pm.FILLER)
            print(f"     alt: {fr:45s} c={combo:.3f} flu={f2:.3f} fill={fill2}")
        print()
