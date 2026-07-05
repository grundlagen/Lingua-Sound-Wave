#!/usr/bin/env python3
"""Juncture test with FRENCH voice (which models liaison/elision)."""
import subprocess, re, json
from collections import Counter
import numpy as np

def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text], capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

bigrams = Counter()
for fp in ["/tmp/fr-candide.txt","/tmp/fr-monte-cristo.txt"]:
    try:
        txt = open(fp,encoding="utf-8",errors="ignore").read()[:100000]
        words = [w.lower().strip(".,;:!?'\"") for w in txt.split() if w.strip(".,;:!?'\"") and len(w)>=2]
        for a,b in zip(words, words[1:]): bigrams[(a,b)] += 1
    except: pass

print("FRENCH VOICE JUNCTURE — FR voice connected vs separate")
print("="*65)

results = []
for i, ((a,b), count) in enumerate(bigrams.most_common(60)):
    # FR voice: connected vs separate
    bigram_fr = tts(f"{a} {b}", "fr").replace(" ","")
    a_fr = tts(a, "fr").replace(" ","")
    b_fr = tts(b, "fr").replace(" ","")
    concat = a_fr + b_fr
    gap = ndice(bigram_fr, concat) if concat else 1.0
    if len(bigram_fr)>=4 and len(concat)>=4:
        # EN voice reading the bigram (what the EN ear hears)
        bigram_en = tts(f"{a} {b}", "en-us").replace(" ","")
        results.append({"gap":round(gap,3), "bigram":f"{a} {b}", 
                        "fr_conn":bigram_fr, "fr_sep":concat,
                        "en_ear":bigram_en, "count":count})

gaps = [r["gap"] for r in results]
print(f"\nFR voice juncture gap: mean={np.mean(gaps):.3f} std={np.std(gaps):.3f}")
print(f"Strong juncture (gap<0.95): {sum(1 for r in results if r['gap']<0.95)}/{len(results)}")
print(f"(FR voice DOES model liaison/elision between words)\n")

# Show bigrams where FR voice CHANGES at the boundary
changed = [r for r in results if r["gap"] < 0.99]
print(f"Bigrams with FR voice boundary effects ({len(changed)}):")
for r in sorted(changed, key=lambda x: x["gap"])[:15]:
    print(f"  gap={r['gap']:.3f}  \"{r['bigram']:30s}\" ×{r['count']}")
    print(f"    FR connected : [{r['fr_conn']}]")
    print(f"    FR separate  : [{r['fr_sep']}]")
    print(f"    EN ear hears : [{r['en_ear']}]")
    print()

# Show the pattern: which word endings trigger liaison
print("JUNCTURE PATTERNS (FR voice):")
for r in sorted(changed, key=lambda x: x["gap"])[:10]:
    a,b = r["bigram"].split()
    # What changed?
    if r["fr_conn"] != r["fr_sep"]:
        diff_start = 0
        for i in range(min(len(r["fr_conn"]), len(r["fr_sep"]))):
            if r["fr_conn"][i] != r["fr_sep"][i]:
                diff_start = i
                break
        conn_slice = r["fr_conn"][max(0,diff_start-2):diff_start+4]
        sep_slice = r["fr_sep"][max(0,diff_start-2):diff_start+4]
        print(f"  \"{a} {b}\": [{sep_slice}] → [{conn_slice}] at boundary")
