#!/usr/bin/env python3
"""Quick bigram juncture demo with 50 most frequent French bigrams."""
import subprocess, re, json
from collections import Counter
import numpy as np

def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
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

print("BIGRAM JUNCTURE — 50 most frequent from Candide+Monte-Cristo")
print("="*60)

results = []
for i, ((a,b), count) in enumerate(bigrams.most_common(50)):
    bigram_en = tts(f"{a} {b}", "en-us").replace(" ","")
    a_en = tts(a, "en-us").replace(" ","")
    b_en = tts(b, "en-us").replace(" ","")
    concat = a_en + b_en
    gap = ndice(bigram_en, concat) if concat else 1.0
    if len(bigram_en)>=4 and len(concat)>=4:
        results.append({"gap":round(gap,3), "bigram":f"{a} {b}", "conn":bigram_en, "sep":concat,
                        "a_en":a_en, "b_en":b_en, "count":count})

gaps = [r["gap"] for r in results]
print(f"\nJuncture gap: mean={np.mean(gaps):.3f} std={np.std(gaps):.3f}")
print(f"Strong juncture (gap<0.85): {sum(1 for r in results if r['gap']<0.85)}/{len(results)}")
print(f"\nClose to 1.0 = words act independently")
print(f"Below 0.85 = words blend at boundary (elision/liaison effect)\n")

for r in sorted(results, key=lambda x: x["gap"])[:15]:
    print(f"  gap={r['gap']:.3f}  \"{r['bigram']:30s}\" ×{r['count']}")
    print(f"    connected    : [{r['conn']}]")
    print(f"    concatenated : [{r['sep']}]")
    print(f"    A+B: [{r['a_en']}] + [{r['b_en']}]")
    print()

with open("bigram_juncture_quick.json","w") as f:
    json.dump({"results": sorted(results, key=lambda x: x["gap"]),
               "mean": float(np.mean(gaps)), "std": float(np.std(gaps))}, f, indent=2)
print("Saved bigram_juncture_quick.json")
