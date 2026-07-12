# Fix: replace FUNC_GLUE literal translations with zipf-glue homophones
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Build homophone lookup from zipf-glue
FUNC_HOMOPHONE = {}
for i,line in enumerate(open("zipf-glue.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=3:
        en = p[0].strip().lower()
        fr = p[1].strip().lower()
        sound = float(p[2])
        if en not in FUNC_HOMOPHONE or sound > FUNC_HOMOPHONE[en][1]:
            FUNC_HOMOPHONE[en] = (fr, sound)

# Show what's available
for en in ["the","a","an","and","of","to","in","is","it","for","that","but","by","can","do","not"]:
    if en in FUNC_HOMOPHONE:
        print(f"  {en:8s} → {FUNC_HOMOPHONE[en][0]:15s} (s={FUNC_HOMOPHONE[en][1]:.2f})")
    else:
        print(f"  {en:8s} → (no homophone)")

print(f"\n  {len(FUNC_HOMOPHONE)} function word homophones available")
print("  Replace FUNC_GLUE dicts with: FUNC_HOMOPHONE.get(en, en)")
