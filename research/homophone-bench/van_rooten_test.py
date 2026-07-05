#!/usr/bin/env python3
"""Van Rooten test — whole-line carve using project's own engine."""
import subprocess, matcher, poetry_mode as pm
import whole_line_carve as wlc
from lexicon_g2p import clean_ipa

try: import bigram_lm; LM = bigram_lm.load("fr")
except: LM = None

def en_ipa(t):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",t],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())

lines = [
    "Humpty Dumpty",
    "Humpty Dumpty sat on a wall",
    "Hickory dickory dock",
    "Jack and Jill went up the hill",
    "Mary had a little lamb",
    "Little Jack Horner sat in a corner",
    "Rain rain go away",
    "twinkle twinkle little star",
    "London bridge is falling down",
    "hey diddle diddle",
    "the cat and the fiddle",
    "Georgie Porgie pudding and pie",
]

print("Building poetry trie...")
root = pm.build_poetry_trie(min_zipf=2.0)
wlc.force_coverage()

print("\n" + "="*70)
print("VAN ROOTEN WHOLE-LINE CARVE")
print("="*70)

for line in lines:
    print(f"\n── {'─'*50}")
    print(f"EN:  {line}")
    try:
        best = None
        for scale in (1.0, 1.6, 2.2):
            matcher.CHEAP_GAP["h"] = 0.08 * scale
            matcher.GAP = min(0.95, 0.42 * scale)
            try:
                ipa, nwords, scored = wlc.carve_line(line, root, beam=350)
                if scored:
                    dual, combo, coh, cov, fr, nfr, nf = scored[0]
                    if best is None or dual > best[0]:
                        best = (dual, combo, coh, cov, fr)
            except Exception:
                pass

        if best:
            dual, combo, coh, cov, fr = best
            print(f"FR:  {fr}")
            print(f"     s={combo:.3f}  flu={coh:.3f}  cov={cov:.0%}  dual={dual:.3f}")
        else:
            print(f"     (no carve)")
    except Exception as e:
        print(f"     error: {e}")
