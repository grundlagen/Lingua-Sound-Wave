#!/usr/bin/env python3
"""Build full training corpus from ALL gold pairs, then train."""
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

all_pairs = []

# strict-gold
for i,line in enumerate(open("strict-gold.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[0] != p[1]:
        all_pairs.append((p[0].lower(), p[1].lower(), 1.0, "strict"))

# v7 gold
for i,line in enumerate(open("dictionary-v7.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=9 and p[3]=="1" and p[7] != p[8]:
        all_pairs.append((p[7].lower(), p[8].lower(), float(p[1]), "v7"))

# tier-ladder (up to 5000 diverse)
for i,line in enumerate(open("tier-ladder.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=12 and p[10]:
        try:
            s = float(p[10])
            if s >= 0.55 and p[1] != p[2]:
                all_pairs.append((p[1].lower(), p[2].lower(), s, "ladder"))
                if len([x for x in all_pairs if x[3]=="ladder"]) >= 5000: break
        except: continue

# Deduplicate
seen = set()
unique = []
for en, fr, s, src in all_pairs:
    key = (en, fr)
    if key not in seen:
        seen.add(key)
        unique.append({"input": f"English word: {en}", "output": fr, "sound": s, "source": src})

print(f"Built {len(unique)} training pairs")
print(f"  strict: {sum(1 for r in unique if r['source']=='strict')}")
print(f"  v7:     {sum(1 for r in unique if r['source']=='v7')}")
print(f"  ladder: {sum(1 for r in unique if r['source']=='ladder')}")
print(f"  Sample: {unique[0]['input']} → {unique[0]['output']}")
print(f"  Sample: {unique[500]['input']} → {unique[500]['output']}")

with open("train-homophonic-full.jsonl","w") as f:
    for r in unique:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved train-homophonic-full.jsonl")
